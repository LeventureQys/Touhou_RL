"""Unit tests for ``tools/rl/check_v104.py`` (S3-6).

The frozen list, the nine evidence indices, their artifacts and both manifests are
**synthetic** in every test here.  That is enough to pin the verdict logic, the
sha/identity checks and the exit codes, and it is exactly why a synthetic all-pass
run is *not* an acceptance result: no real experiment produced these numbers.

What is pinned:

* the 20 flat keys are required and ``architecture`` is not an alias for ``arch``;
  a frozen list that violates the schema exits 2 without writing a verdict;
* every evidence index and every artifact it lists is re-hashed; a changed byte,
  a missing key or an unknown evidence key changes the verdict, not the exit path
  silently;
* A-01 is the full 7,241-frame re-check of the frozen ``state.best`` checkpoint;
  A-02 is the ``0-15`` bucket of the frozen ``policy.pt`` with ``frames > 0``;
  A-03 re-derives the stage sample from both manifests and compares unrounded
  means; A-04 recomputes ``frames_mean``/``hit_step_cap`` from the eval-report and
  needs the startup evidence (a stub run cannot pass); A-05 re-parses the pytest
  log; A-06 needs all three of D9/D10/D11;
* ``fail`` and ``blocked`` coexist: exit is 2 while each item keeps its own status.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from touhou_rl.pipeline.manifest import (
    DatasetManifest,
    ShardRecord,
    file_digest,
    index_digest,
    split_indices,
)
from touhou_rl.vision_data import VisionDataset
from touhou_rl.vmodels import save_vision_checkpoint
from touhou_rl.vmodels2 import VisionNetConfigV2, VisionScorerV2

ROOT = Path(__file__).resolve().parents[2]

REGION_SIZES: dict[str, tuple[int, int]] = {
    "playfield": (8, 8),
    "hud_score": (2, 8),
    "hud_status": (8, 4),
}
A_TOTAL = 72413  # the real A dataset size: the split then holds exactly 7,241 frames
A_SHARDS = 57
B_SHARDS = 4
B_FRAMES = 10
STAGES = (0, 1, 2, 3)
SEED = 7
VAL_FRACTION = 0.1
PER_STAGE = 128


def _load_tool(name: str):
    module_name = f"s36_tools_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "tools" / "rl" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CHECK = _load_tool("check_v104")


# --------------------------------------------------------------------------- #
# synthetic datasets (metadata only: the checker never decodes pixels)
# --------------------------------------------------------------------------- #
def _write_meta_dataset(path: Path, sizes: list[int], *, stages=STAGES) -> list[ShardRecord]:
    """One npz per shard with ``action``/``meta``/region metadata, no pixels.

    ``stage_of`` and ``len`` only need ``action``/``meta``, and check_v104 never
    reads pixels, so a 72k-frame manifest costs a few MB instead of gigabytes.
    """
    path.mkdir(parents=True, exist_ok=True)
    records: list[ShardRecord] = []
    for index, frames in enumerate(sizes):
        stage = int(stages[index % len(stages)])
        meta = np.zeros((frames, 8), dtype=np.float32)
        meta[:, 0] = 2.0 * np.arange(frames)
        meta[:, 1] = float(stage)
        meta[:, 4] = 2.0
        meta[:, 5] = 2.0
        name = f"collect-{index:05d}.npz"
        np.savez(
            path / name,
            region_name=np.asarray(list(REGION_SIZES), dtype="U24"),
            region_hw=np.asarray([REGION_SIZES[key] for key in REGION_SIZES], dtype=np.int32),
            action=np.zeros((frames, 4), dtype=np.uint8),
            meta=meta,
            # ``discover_vision_shards`` needs a frame payload member to recognise
            # the shard; check_v104 never decodes it (meta only).
            frame=np.zeros((frames, 1, 1, 3), dtype=np.uint8),
        )
        records.append(
            ShardRecord(
                name=name,
                frames=frames,
                bytes=(path / name).stat().st_size,
                sha256=file_digest(path / name),
            )
        )
    return records


def _a_layout() -> list[int]:
    base = 1271
    sizes = [base] * (A_SHARDS - 1)
    sizes.append(A_TOTAL - sum(sizes))
    assert sum(sizes) == A_TOTAL
    return sizes


def _write_manifest(dataset: Path, records: list[ShardRecord], *, name: str = "manifest.json") -> Path:
    manifest = DatasetManifest.build(
        modality="vision",
        root=dataset,
        shards=records,
        frame_shape=[448, 384, 3],
        has_teacher_values=True,
        teacher={"name": "pilot", "config": {}},
        seed=SEED,
        val_fraction=VAL_FRACTION,
        stats={},
        provenance={"purpose": "synthetic-test-only"},
    )
    path = dataset / name
    manifest.save(path)
    return path


@pytest.fixture(scope="module")
def shared_datasets(tmp_path_factory):
    """Built once: 72,413 metadata frames for A, 40 for B."""
    base = tmp_path_factory.mktemp("s36-check-fixtures")
    a = base / "dataset-a"
    _write_manifest(a, _write_meta_dataset(a, _a_layout()))
    b = base / "dataset-b"
    _write_manifest(b, _write_meta_dataset(b, [B_FRAMES] * B_SHARDS))
    return {"dataset-a": a, "dataset-b": b}


@pytest.fixture()
def root(tmp_path: Path, shared_datasets) -> Path:
    for name, source in shared_datasets.items():
        shutil.copytree(source, tmp_path / name)
    return tmp_path


# --------------------------------------------------------------------------- #
# fixture writer: checkpoints, summaries, evidence, frozen list
# --------------------------------------------------------------------------- #
def _rel(path: Path, root: Path) -> str:
    return str(Path(path).relative_to(root)).replace("\\", "/")


def _sha(path: Path) -> str:
    return file_digest(path)


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _checkpoint(path: Path, *, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    config = VisionNetConfigV2(
        depth=3,
        playfield_hw=REGION_SIZES["playfield"],
        hud_score_hw=REGION_SIZES["hud_score"],
        hud_status_hw=REGION_SIZES["hud_status"],
        pf_width=4,
        hud_width=2,
        trunk_hidden=16,
        state_hidden=16,
        candidate_hidden=8,
    )
    save_vision_checkpoint(path, VisionScorerV2(config), {"stage": "state", "epoch": 1})
    return path


def expected_selection(dataset_dir: Path, per_stage: int = PER_STAGE) -> list[int]:
    """The stage sample, implemented independently of the tool under test."""
    manifest = DatasetManifest.load(dataset_dir / "manifest.json")
    _train, val = split_indices(manifest.total_frames, SEED, VAL_FRACTION)
    with VisionDataset(dataset_dir) as dataset:
        stages = dataset.stage_of(val).tolist()
    selected: list[int] = []
    for stage in sorted(set(stages)):
        rows = [index for index, value in zip(val, stages) if value == stage]
        selected.extend(rows[:per_stage])
    return selected


def _evidence(root: Path, key: str, observations: dict, artifacts: list[Path], *, status="pass", reason=""):
    entries = [
        {"path": _rel(artifact, root), "sha256": _sha(artifact)} for artifact in artifacts
    ]
    index = {
        "schema": "v104-evidence/1",
        "kind": key,
        "status": status,
        "created_at": "2026-10-19T12:00:00+08:00",
        "producer": "synthetic-fixture",
        "command": "synthetic",
        "cwd": str(root),
        "exit_code": 0,
        "artifacts": entries,
        "observations": observations,
        "reason": reason,
    }
    path = _write_json(root / "evidence" / f"{key}.json", index)
    return {"path": _rel(path, root), "sha256": _sha(path)}


def _json_artifact(root: Path, name: str, payload: dict) -> Path:
    return _write_json(root / "evidence" / "artifacts" / name, payload)


def build_fixture(root: Path) -> dict:
    """A fully consistent synthetic freeze (all six items pass)."""
    a = root / "dataset-a"
    b = root / "dataset-b"
    state = _checkpoint(root / "runs" / "state-pass.pt", seed=11)
    policy = _checkpoint(root / "runs" / "policy.pt", seed=23)
    teacher = _write_json(root / "runs" / "teacher-values.json", {"teacher": "pilot", "n": 18})
    table = _write_json(root / "runs" / "experiment-table.json", {"rows": []})
    manifest_a = DatasetManifest.load(a / "manifest.json")
    manifest_b = DatasetManifest.load(b / "manifest.json")
    val_sha = manifest_a.splits["val_index_sha256"]

    state_summary = _write_json(
        root / "runs" / "state-summary.json",
        {
            "schema": "vision-summary/2",
            "contract_version": "v104-A/1",
            "arch": "v2",
            "stages": {
                "state": {
                    "best": {
                        "epoch": 9,
                        "checkpoint": "state-pass.pt",
                        "checkpoint_sha256": _sha(state),
                    }
                }
            },
        },
    )
    policy_summary = _write_json(
        root / "runs" / "policy-summary.json",
        {
            "schema": "vision-summary/2",
            "contract_version": "v104-A/1",
            "arch": "v2",
            "stages": {
                "policy": {
                    "best": {
                        "epoch": 8,
                        "checkpoint": "policy.pt",
                        "checkpoint_sha256": _sha(policy),
                    }
                }
            },
        },
    )

    a_selection = expected_selection(a)
    b_selection = expected_selection(b)

    state_obs = {
        "checkpoint_sha256": _sha(state),
        "frames": 7241,
        "coverage_sha256": val_sha,
        "player_pos_px": 8.4,
        "danger_corr_perframe_median": 0.86,
        "danger_corr_perframe_valid": 7241,
    }
    policy_obs = dict(state_obs)
    policy_obs.update(
        {
            "checkpoint_sha256": _sha(policy),
            "by_danger": [
                {"bucket": "0-15", "frames": 900, "top1": 0.4, "top3": 0.42, "regret": 1.2},
                {"bucket": "15-30", "frames": 700, "top1": 0.5, "top3": 0.5, "regret": 3.0},
            ],
        }
    )
    cross_obs = {
        "checkpoint_sha256": _sha(policy),
        "a_mae_raw": 10.0,
        "b_mae_raw": 11.0,
        "a_indices_sha256": index_digest(a_selection),
        "b_indices_sha256": index_digest(b_selection),
        "a_frames": len(a_selection),
        "b_frames": len(b_selection),
        "sampling_mode": "stage-sample",
        "per_stage": PER_STAGE,
    }

    episodes = []
    for index, frames in enumerate((4200, 4300, 4100), start=1):
        episodes.append(
            {
                "episode": index,
                "steps": frames // 2,
                "frames_reported": frames,
                "terminated": True,
                "truncated": False,
                "termination_reason": "game over",
                "last_state": {"stage": 2, "frame": frames, "bullets": 0},
            }
        )
    eval_report = _write_json(
        root / "evidence" / "artifacts" / "eval-report.json",
        {
            "checkpoint": ".",
            "policy": "vision",
            "difficulty": 3,
            "team": 0,
            "practice_stage": None,
            "frameskip": 2,
            "vision": {
                "checkpoint": _rel(policy, root),
                "arch": "v2",
                "depth": 3,
                "max_steps": 12000,
                "isolation": "I-1: pixels only",
            },
            "episodes": episodes,
            "summary": {"episodes": 3, "frames_mean": 4200.0, "frames_max": 4300},
        },
    )
    eval_obs = {
        "checkpoint_sha256": _sha(policy),
        "episodes": 3,
        "frames_mean": 4200.0,
        "hit_step_cap": 0,
        "episode_fields_complete": True,
    }
    pytest_log = root / "evidence" / "artifacts" / "pytest.log"
    pytest_log.parent.mkdir(parents=True, exist_ok=True)
    pytest_log.write_text(
        "........................................ 180 passed, 2 skipped in 12.34s\n", encoding="utf-8"
    )
    pytest_obs = {"passed": 180, "failed": 0, "skipped": 2, "command_exit": 0}

    def observation_artifact(name: str, observations: dict) -> Path:
        return _json_artifact(root, name, {"observations": observations})

    state_artifact = observation_artifact("state-recheck.json", state_obs)
    policy_artifact = observation_artifact("policy-full-val.json", policy_obs)
    cross_artifact = observation_artifact("cross-replay.json", cross_obs)
    d9_obs = {
        "expected_frames": 7241,
        "actual_frames": 7241,
        "visited_index_sha256": val_sha,
        "last_batch_size": 41,
    }
    d10_obs = {
        "budget_bytes": 536_870_912,
        "max_resident_bytes": 536_673_901,
        "lru_verified": True,
        "value_equal": True,
    }
    d11_obs = {
        "expected_files": 1626,
        "validated_files": 1626,
        "bytes_total": 16_134_972_672,
        "value_equal": True,
        "manifest_sha": _sha(b / "manifest.json"),
    }
    startup_obs = {
        "real_game_started": True,
        "pixels_only": True,
        "no_crash": True,
        "no_exception_log": True,
        "ui_checked": True,
        "screenshots": ["evidence/artifacts/episode-01.png"],
    }
    d9_artifact = observation_artifact("d9.json", d9_obs)
    d10_artifact = observation_artifact("d10.json", d10_obs)
    d11_artifact = observation_artifact("d11.json", d11_obs)
    startup_artifact = observation_artifact("startup.json", startup_obs)

    evidence = {
        "state_recheck": _evidence(root, "state_recheck", state_obs, [state_artifact]),
        "policy_full_val": _evidence(root, "policy_full_val", policy_obs, [policy_artifact]),
        "cross_replay": _evidence(root, "cross_replay", cross_obs, [cross_artifact]),
        "eval_report": _evidence(root, "eval_report", eval_obs, [eval_report]),
        "pytest": _evidence(root, "pytest", pytest_obs, [pytest_log]),
        "d9": _evidence(root, "d9", d9_obs, [d9_artifact]),
        "d10": _evidence(root, "d10", d10_obs, [d10_artifact]),
        "d11": _evidence(root, "d11", d11_obs, [d11_artifact]),
        "startup": _evidence(root, "startup", startup_obs, [startup_artifact]),
    }
    frozen = {
        "schema": "v104-final/1",
        "state_checkpoint": _rel(state, root),
        "state_checkpoint_sha256": _sha(state),
        "state_summary": _rel(state_summary, root),
        "policy_checkpoint": _rel(policy, root),
        "policy_checkpoint_sha256": _sha(policy),
        "policy_summary": _rel(policy_summary, root),
        "teacher_cache": _rel(teacher, root),
        "teacher_fingerprint": "f" * 64,
        "manifest_a": _rel(a / "manifest.json", root),
        "manifest_a_sha256": _sha(a / "manifest.json"),
        "manifest_b": _rel(b / "manifest.json", root),
        "manifest_b_sha256": _sha(b / "manifest.json"),
        "val_index_sha256": val_sha,
        "arch": "v2",
        "depth": 3,
        "seed": SEED,
        "experiment_table": _rel(table, root),
        "selection_reason": "lowest 0-15 bucket regret on the A validation set",
        "frozen_at": "2026-10-19T12:00:00+08:00",
        "external_guard_uses": 0,
        "evidence": evidence,
    }
    frozen_path = _write_json(root / "final-artifacts.json", frozen)
    return {
        "root": root,
        "frozen": frozen_path,
        "frozen_doc": frozen,
        "evidence": evidence,
        "state": state,
        "policy": policy,
        "eval_report": eval_report,
        "pytest_log": pytest_log,
        "a_selection": a_selection,
        "b_selection": b_selection,
        "val_index_sha256": val_sha,
        "observations": {
            "state_recheck": state_obs,
            "policy_full_val": policy_obs,
            "cross_replay": cross_obs,
            "eval_report": eval_obs,
            "pytest": pytest_obs,
            "d9": d9_obs,
            "d10": d10_obs,
            "d11": d11_obs,
            "startup": startup_obs,
        },
    }


def run(fixture: dict, *, out: bool = False, frozen_doc: dict | None = None, root: Path | None = None):
    root = root or fixture["root"]
    frozen = fixture["frozen"]
    if frozen_doc is not None:
        frozen = _write_json(root / "final-artifacts.json", frozen_doc)
    out_path = root / "verdict.json"
    argv = ["--artifacts", str(frozen), "--root", str(root)]
    if out:
        argv.extend(["--out", str(out_path)])
    code = CHECK.main(argv)
    document = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else None
    return code, document


def _rewrite_evidence(fixture: dict, key: str, observations: dict | None = None, *,
                      status: str | None = None, artifacts=None, sync_artifact: bool = True) -> dict:
    """Rewrite one index (and its frozen sha) inside a copy of the frozen document.

    ``sync_artifact`` also updates the artifact's own ``observations`` node, which
    models a producer that actually recorded those numbers.  Pass
    ``sync_artifact=False`` to create an index/artifact disagreement on purpose.
    """
    root = fixture["root"]
    index_path = root / "evidence" / f"{key}.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if observations is not None:
        index["observations"] = observations
    if status is not None:
        index["status"] = status
    if artifacts is not None:
        index["artifacts"] = artifacts
    if sync_artifact and observations is not None:
        for entry in index.get("artifacts", []):
            artifact_path = root / entry["path"]
            try:
                document = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(document, dict) and isinstance(document.get("observations"), dict):
                document["observations"] = observations
                _write_json(artifact_path, document)
                entry["sha256"] = _sha(artifact_path)
    _write_json(index_path, index)
    frozen_doc = json.loads(json.dumps(fixture["frozen_doc"]))
    frozen_doc["evidence"][key] = {"path": _rel(index_path, root), "sha256": _sha(index_path)}
    fixture["frozen_doc"] = frozen_doc
    return frozen_doc


# --------------------------------------------------------------------------- #
# the all-pass path (synthetic!)
# --------------------------------------------------------------------------- #
def test_all_six_pass_on_consistent_synthetic_evidence(root: Path):
    """Synthetic all-pass: proves the verdict logic, NOT that v1.0.4 was accepted."""
    fixture = build_fixture(root)
    code, document = run(fixture, out=True)
    assert code == 0
    assert document["exit_code"] == 0
    assert {name: item["status"] for name, item in document["items"].items()} == {
        "A-01": "pass",
        "A-02": "pass",
        "A-03": "pass",
        "A-04": "pass",
        "A-05": "pass",
        "A-06": "pass",
    }
    assert document["artifacts"]["sha256"] == _sha(fixture["frozen"])
    assert document["items"]["A-01"]["values"]["player_pos_px"] == 8.4
    assert document["items"]["A-02"]["values"]["regret"] == 1.2
    assert document["items"]["A-03"]["values"]["ratio"] == pytest.approx(1.1)
    assert document["items"]["A-03"]["values"]["a_frames"] == len(fixture["a_selection"])
    assert document["items"]["A-04"]["values"]["frames_mean"] == 4200.0
    assert document["items"]["A-05"]["values"]["passed"] == 180
    assert all(node["ok"] for node in document["evidence"].values())
    # The frozen identity (file shas + summary<->checkpoint binding) is verified.
    assert document["frozen"]["identity_ok"] is True


# --------------------------------------------------------------------------- #
# frozen-list schema
# --------------------------------------------------------------------------- #
def test_missing_required_key_exits_2_without_a_verdict(root: Path, capsys):
    fixture = build_fixture(root)
    document = json.loads(json.dumps(fixture["frozen_doc"]))
    del document["arch"]
    path = _write_json(root / "broken.json", document)
    code = CHECK.main(["--artifacts", str(path), "--root", str(root), "--out", str(root / "v.json")])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.startswith("v104:EVIDENCE:")
    assert "arch" in captured.err
    assert not (root / "v.json").exists()


def test_architecture_is_not_an_alias_for_arch(root: Path, capsys):
    fixture = build_fixture(root)
    document = json.loads(json.dumps(fixture["frozen_doc"]))
    document["architecture"] = document.pop("arch")
    path = _write_json(root / "aliased.json", document)
    code = CHECK.main(["--artifacts", str(path), "--root", str(root)])
    captured = capsys.readouterr()
    assert code == 2
    assert "not an alias" in captured.err


def test_absolute_paths_and_bad_shas_are_refused(root: Path, capsys):
    fixture = build_fixture(root)
    document = json.loads(json.dumps(fixture["frozen_doc"]))
    document["state_checkpoint"] = str(root / "runs" / "state-pass.pt")
    path = _write_json(root / "absolute.json", document)
    assert CHECK.main(["--artifacts", str(path), "--root", str(root)]) == 2
    assert "workspace-relative" in capsys.readouterr().err

    document = json.loads(json.dumps(fixture["frozen_doc"]))
    document["state_checkpoint_sha256"] = "NOT-A-SHA"
    path = _write_json(root / "badsha.json", document)
    assert CHECK.main(["--artifacts", str(path), "--root", str(root)]) == 2
    assert "64 lowercase hex" in capsys.readouterr().err


def test_unknown_evidence_key_is_a_contract_error(root: Path, capsys):
    fixture = build_fixture(root)
    document = json.loads(json.dumps(fixture["frozen_doc"]))
    document["evidence"]["extra"] = document["evidence"]["d9"]
    path = _write_json(root / "extra-evidence.json", document)
    assert CHECK.main(["--artifacts", str(path), "--root", str(root)]) == 2
    assert "unknown key" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# exit codes: fail only, blocked only, both
# --------------------------------------------------------------------------- #
def test_one_fail_exits_1(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["policy_full_val"])
    observations["by_danger"] = [
        {"bucket": "0-15", "frames": 900, "top3": 0.42, "regret": 2.5},
    ]
    document = _rewrite_evidence(fixture, "policy_full_val", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 1
    assert verdict["items"]["A-02"]["status"] == "fail"
    assert verdict["items"]["A-01"]["status"] == "pass"
    assert verdict["items"]["A-06"]["status"] == "pass"


def test_blocked_and_fail_coexist_exit_2(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["policy_full_val"])
    observations["by_danger"] = [{"bucket": "0-15", "frames": 900, "top3": 0.42, "regret": 2.5}]
    _rewrite_evidence(fixture, "policy_full_val", observations)
    document = _rewrite_evidence(fixture, "d9", status="blocked")
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-02"]["status"] == "fail"
    assert verdict["items"]["A-06"]["status"] == "blocked"
    assert verdict["counts"]["fail"] == 1
    assert verdict["counts"]["blocked"] == 1


def test_missing_evidence_entry_blocks_only_its_items(root: Path):
    fixture = build_fixture(root)
    document = json.loads(json.dumps(fixture["frozen_doc"]))
    del document["evidence"]["cross_replay"]
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-03"]["status"] == "blocked"
    assert verdict["items"]["A-01"]["status"] == "pass"
    assert verdict["items"]["A-04"]["status"] == "pass"


def test_artifact_bitflip_is_blocked(root: Path):
    fixture = build_fixture(root)
    artifact = fixture["eval_report"]
    artifact.write_text(artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    code, verdict = run(fixture, out=True)
    assert code == 2
    assert verdict["items"]["A-04"]["status"] == "blocked"
    assert "artifact sha256" in verdict["items"]["A-04"]["reason"]


def test_evidence_index_bitflip_is_blocked(root: Path):
    fixture = build_fixture(root)
    index_path = root / "evidence" / "state_recheck.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["observations"]["player_pos_px"] = 9.9
    _write_json(index_path, payload)
    code, verdict = run(fixture, out=True)
    assert code == 2
    assert verdict["items"]["A-01"]["status"] == "blocked"
    assert "index sha256" in verdict["items"]["A-01"]["reason"]


def test_index_disagreeing_with_its_artifact_is_blocked(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["state_recheck"])
    observations["player_pos_px"] = 7.0  # the artifact still says 8.4
    document = _rewrite_evidence(fixture, "state_recheck", observations, sync_artifact=False)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-01"]["status"] == "blocked"
    assert "disagrees with artifact" in verdict["items"]["A-01"]["reason"]


# --------------------------------------------------------------------------- #
# A-01 / A-02 boundaries
# --------------------------------------------------------------------------- #
def test_a01_requires_the_full_7241_frames(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["state_recheck"])
    observations["frames"] = 7000
    document = _rewrite_evidence(fixture, "state_recheck", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-01"]["status"] == "blocked"
    assert "full validation set is 7241" in verdict["items"]["A-01"]["reason"]


def test_a01_coverage_must_match_the_frozen_split(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["state_recheck"])
    observations["coverage_sha256"] = "0" * 64
    document = _rewrite_evidence(fixture, "state_recheck", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-01"]["status"] == "blocked"
    assert "coverage_sha256" in verdict["items"]["A-01"]["reason"]


def test_a01_thresholds_are_strict(root: Path):
    fixture = build_fixture(root)
    for position, perframe, expected in (
        (10.0, 0.9, "fail"),
        (9.9, 0.8, "fail"),
        (9.9, 0.800001, "pass"),
    ):
        fixture = build_fixture(root)
        observations = dict(fixture["observations"]["state_recheck"])
        observations["player_pos_px"] = position
        observations["danger_corr_perframe_median"] = perframe
        document = _rewrite_evidence(fixture, "state_recheck", observations)
        code, verdict = run(fixture, out=True, frozen_doc=document)
        assert verdict["items"]["A-01"]["status"] == expected, (position, perframe)


def test_a01_wrong_checkpoint_is_blocked(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["state_recheck"])
    observations["checkpoint_sha256"] = "1" * 64
    document = _rewrite_evidence(fixture, "state_recheck", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-01"]["status"] == "blocked"
    assert "state_checkpoint_sha256" in verdict["items"]["A-01"]["reason"]


def test_summary_must_point_at_the_frozen_checkpoint(root: Path):
    fixture = build_fixture(root)
    summary_path = root / "runs" / "state-summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["stages"]["state"]["best"]["checkpoint_sha256"] = "2" * 64
    _write_json(summary_path, payload)
    code, verdict = run(fixture, out=True)
    assert code == 2
    assert verdict["items"]["A-01"]["status"] == "blocked"
    assert any("summary" in error for error in verdict["frozen"]["identity_errors"])


def test_frozen_arch_must_match_the_checkpoint_payload(root: Path):
    """The freeze names the arch; the payload is what a deployment would load."""
    fixture = build_fixture(root)
    document = json.loads(json.dumps(fixture["frozen_doc"]))
    document["arch"] = "v3-st"  # the checkpoints are v2
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-01"]["status"] == "blocked"
    assert any(
        "arch 'v2' != frozen arch 'v3-st'" in error
        for error in verdict["frozen"]["identity_errors"]
    )


def test_frozen_depth_must_match_the_checkpoint_payload(root: Path):
    fixture = build_fixture(root)
    document = json.loads(json.dumps(fixture["frozen_doc"]))
    document["depth"] = 1
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-01"]["status"] == "blocked"
    assert any(
        "config depth 3" in error for error in verdict["frozen"]["identity_errors"]
    )


def test_a02_empty_bucket_cannot_pass(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["policy_full_val"])
    observations["by_danger"] = [{"bucket": "0-15", "frames": 0, "top3": 0.0, "regret": 0.0}]
    document = _rewrite_evidence(fixture, "policy_full_val", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 1
    assert verdict["items"]["A-02"]["status"] == "fail"
    assert "frames=0" in verdict["items"]["A-02"]["reason"]


def test_a02_missing_top3_is_blocked(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["policy_full_val"])
    observations["by_danger"] = [{"bucket": "0-15", "frames": 900, "regret": 1.2}]
    document = _rewrite_evidence(fixture, "policy_full_val", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-02"]["status"] == "blocked"
    assert "top3" in verdict["items"]["A-02"]["reason"]


def test_a02_uses_the_policy_checkpoint_not_the_state_one(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["policy_full_val"])
    observations["checkpoint_sha256"] = _sha(fixture["state"])
    document = _rewrite_evidence(fixture, "policy_full_val", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-02"]["status"] == "blocked"
    assert "policy_checkpoint_sha256" in verdict["items"]["A-02"]["reason"]


# --------------------------------------------------------------------------- #
# A-03
# --------------------------------------------------------------------------- #
def test_a03_selected_indices_are_recomputed(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["cross_replay"])
    observations["a_indices_sha256"] = "3" * 64
    document = _rewrite_evidence(fixture, "cross_replay", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-03"]["status"] == "blocked"
    assert "recomputed" in verdict["items"]["A-03"]["reason"]


def test_a03_frame_counts_are_recomputed(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["cross_replay"])
    observations["b_frames"] = observations["b_frames"] + 1
    document = _rewrite_evidence(fixture, "cross_replay", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-03"]["status"] == "blocked"
    assert "b_frames" in verdict["items"]["A-03"]["reason"]


def test_a03_ratio_rule(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["cross_replay"])
    observations["b_mae_raw"] = 12.51  # 12.51 / 10.0 = 1.251
    document = _rewrite_evidence(fixture, "cross_replay", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 1
    assert verdict["items"]["A-03"]["status"] == "fail"

    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["cross_replay"])
    observations["b_mae_raw"] = 12.5  # exactly 1.25 -> allowed
    document = _rewrite_evidence(fixture, "cross_replay", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert verdict["items"]["A-03"]["status"] == "pass"


def test_a03_requires_stage_sample_and_per_stage_128(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["cross_replay"])
    observations["sampling_mode"] = "full-val"
    document = _rewrite_evidence(fixture, "cross_replay", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-03"]["status"] == "blocked"
    assert "stage-sample" in verdict["items"]["A-03"]["reason"]

    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["cross_replay"])
    observations["per_stage"] = 64
    document = _rewrite_evidence(fixture, "cross_replay", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert "per_stage=128" in verdict["items"]["A-03"]["reason"]


def test_a03_broken_manifest_split_is_blocked(root: Path):
    fixture = build_fixture(root)
    manifest_path = root / "dataset-b" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["splits"]["val_index_sha256"] = "4" * 64
    _write_json(manifest_path, payload)
    document = json.loads(json.dumps(fixture["frozen_doc"]))
    document["manifest_b_sha256"] = _sha(manifest_path)  # re-freeze the broken manifest
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-03"]["status"] == "blocked"
    assert "re-derive" in verdict["items"]["A-03"]["reason"] or "split" in verdict["items"]["A-03"]["reason"]


# --------------------------------------------------------------------------- #
# A-04
# --------------------------------------------------------------------------- #
def _mutate_eval_report(fixture: dict, mutate) -> None:
    payload = json.loads(fixture["eval_report"].read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(fixture["eval_report"], payload)
    # The frozen index must keep pointing at the (changed) artifact.
    index_path = fixture["root"] / "evidence" / "eval_report.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["artifacts"] = [
        {"path": _rel(fixture["eval_report"], fixture["root"]), "sha256": _sha(fixture["eval_report"])}
    ]
    _write_json(index_path, index)
    fixture["frozen_doc"]["evidence"]["eval_report"] = {
        "path": _rel(index_path, fixture["root"]),
        "sha256": _sha(index_path),
    }


def test_a04_recomputes_frames_mean_and_catches_a_lie(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["eval_report"])
    observations["frames_mean"] = 9999.0
    document = _rewrite_evidence(fixture, "eval_report", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-04"]["status"] == "blocked"
    assert "frames_mean" in verdict["items"]["A-04"]["reason"]


def test_a04_hit_step_cap_fails(root: Path):
    fixture = build_fixture(root)

    def mutate(report):
        report["episodes"][2]["termination_reason"] = "max_steps"

    _mutate_eval_report(fixture, mutate)
    # The producer records the cap it actually hit; the checker recomputes it from
    # the eval-report and then fails the item.
    observations = dict(fixture["observations"]["eval_report"])
    observations["hit_step_cap"] = 1
    document = _rewrite_evidence(fixture, "eval_report", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 1
    assert verdict["items"]["A-04"]["status"] == "fail"
    assert "hit_step_cap 1" in verdict["items"]["A-04"]["reason"]


def test_a04_frames_mean_threshold(root: Path):
    fixture = build_fixture(root)

    def mutate(report):
        for row, frames in zip(report["episodes"], (3000, 3100, 3200)):
            row["frames_reported"] = frames

    _mutate_eval_report(fixture, mutate)
    observations = dict(fixture["observations"]["eval_report"])
    observations["frames_mean"] = 3100.0
    document = _rewrite_evidence(fixture, "eval_report", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 1
    assert verdict["items"]["A-04"]["status"] == "fail"
    assert "frames_mean 3100.0" in verdict["items"]["A-04"]["reason"]


def test_a04_needs_the_startup_evidence(root: Path):
    fixture = build_fixture(root)
    document = json.loads(json.dumps(fixture["frozen_doc"]))
    del document["evidence"]["startup"]
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-04"]["status"] == "blocked"
    assert "startup" in verdict["items"]["A-04"]["reason"]


def test_a04_needs_the_step_cap_recorded(root: Path):
    """Without max_steps=12000 somewhere the item cannot be judged, so it blocks."""

    fixture = build_fixture(root)

    def mutate(report):
        report["vision"].pop("max_steps")

    _mutate_eval_report(fixture, mutate)
    code, verdict = run(fixture, out=True, frozen_doc=fixture["frozen_doc"])
    assert code == 2
    assert verdict["items"]["A-04"]["status"] == "blocked"
    assert "max_steps" in verdict["items"]["A-04"]["reason"]


def test_a04_episode_count_must_match_the_report(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["eval_report"])
    observations["episodes"] = 5
    document = _rewrite_evidence(fixture, "eval_report", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-04"]["status"] == "blocked"
    assert "observed episodes 5" in verdict["items"]["A-04"]["reason"]


def test_a04_stub_run_cannot_pass(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["startup"])
    observations["real_game_started"] = False
    document = _rewrite_evidence(fixture, "startup", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 1
    assert verdict["items"]["A-04"]["status"] == "fail"
    assert "real_game_started" in verdict["items"]["A-04"]["reason"]


def test_a04_deploying_state_pass_is_blocked(root: Path):
    fixture = build_fixture(root)

    def mutate(report):
        report["vision"]["checkpoint"] = _rel(fixture["state"], fixture["root"])

    _mutate_eval_report(fixture, mutate)
    code, verdict = run(fixture, out=True, frozen_doc=fixture["frozen_doc"])
    assert code == 2
    assert verdict["items"]["A-04"]["status"] == "blocked"
    assert "policy.pt" in verdict["items"]["A-04"]["reason"]


def test_a04_missing_cap_fields_fail(root: Path):
    """Missing per-episode terminated/truncated/termination_reason is not a zero cap."""

    fixture = build_fixture(root)

    def mutate(report):
        report["episodes"][0].pop("termination_reason")

    _mutate_eval_report(fixture, mutate)
    code, verdict = run(fixture, out=True, frozen_doc=fixture["frozen_doc"])
    assert code == 1
    assert verdict["items"]["A-04"]["status"] == "fail"
    assert "incomplete" in verdict["items"]["A-04"]["reason"]


def test_a04_wrong_difficulty_fails(root: Path):
    fixture = build_fixture(root)
    _mutate_eval_report(fixture, lambda report: report.update({"difficulty": 1}))
    code, verdict = run(fixture, out=True, frozen_doc=fixture["frozen_doc"])
    assert code == 1
    assert verdict["items"]["A-04"]["status"] == "fail"
    assert "difficulty" in verdict["items"]["A-04"]["reason"]


# --------------------------------------------------------------------------- #
# A-05
# --------------------------------------------------------------------------- #
def test_a05_below_178_passed_fails(root: Path):
    fixture = build_fixture(root)
    fixture["pytest_log"].write_text("170 passed in 4.20s\n", encoding="utf-8")
    observations = {"passed": 170, "failed": 0, "skipped": 0, "command_exit": 0}
    document = _rewrite_evidence(
        fixture,
        "pytest",
        observations,
        artifacts=[{"path": _rel(fixture["pytest_log"], root), "sha256": _sha(fixture["pytest_log"])}],
    )
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 1
    assert verdict["items"]["A-05"]["status"] == "fail"
    assert "178" in verdict["items"]["A-05"]["reason"]


def test_a05_failed_test_fails(root: Path):
    fixture = build_fixture(root)
    fixture["pytest_log"].write_text("1 failed, 180 passed in 4.20s\n", encoding="utf-8")
    observations = {"passed": 180, "failed": 1, "skipped": 0, "command_exit": 1}
    document = _rewrite_evidence(
        fixture,
        "pytest",
        observations,
        artifacts=[{"path": _rel(fixture["pytest_log"], root), "sha256": _sha(fixture["pytest_log"])}],
    )
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 1
    assert verdict["items"]["A-05"]["status"] == "fail"


def test_a05_index_contradicting_the_log_is_blocked(root: Path):
    fixture = build_fixture(root)
    fixture["pytest_log"].write_text("160 passed in 4.20s\n", encoding="utf-8")
    index_path = root / "evidence" / "pytest.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["artifacts"] = [
        {"path": _rel(fixture["pytest_log"], root), "sha256": _sha(fixture["pytest_log"])}
    ]
    _write_json(index_path, index)
    fixture["frozen_doc"]["evidence"]["pytest"] = {
        "path": _rel(index_path, root),
        "sha256": _sha(index_path),
    }
    code, verdict = run(fixture, out=True, frozen_doc=fixture["frozen_doc"])
    assert code == 2
    assert verdict["items"]["A-05"]["status"] == "blocked"
    assert "raw log" in verdict["items"]["A-05"]["reason"]


# --------------------------------------------------------------------------- #
# A-06
# --------------------------------------------------------------------------- #
def test_a06_requires_all_three_defects(root: Path):
    fixture = build_fixture(root)
    document = json.loads(json.dumps(fixture["frozen_doc"]))
    del document["evidence"]["d10"]
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-06"]["status"] == "blocked"
    assert "d10" in verdict["items"]["A-06"]["reason"]


def test_a06_rechecks_the_budget_and_coverage(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["d10"])
    observations["max_resident_bytes"] = 600_000_000
    document = _rewrite_evidence(fixture, "d10", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 1
    assert verdict["items"]["A-06"]["status"] == "fail"
    assert "max_resident_bytes" in verdict["items"]["A-06"]["reason"]

    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["d9"])
    observations["actual_frames"] = 7200
    document = _rewrite_evidence(fixture, "d9", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 1
    assert verdict["items"]["A-06"]["status"] == "fail"
    assert "actual_frames" in verdict["items"]["A-06"]["reason"]


def test_a06_visited_index_must_match_the_frozen_split(root: Path):
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["d9"])
    observations["visited_index_sha256"] = "5" * 64
    document = _rewrite_evidence(fixture, "d9", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 1
    assert "visited_index_sha256" in verdict["items"]["A-06"]["reason"]


# --------------------------------------------------------------------------- #
# small pure-function units
# --------------------------------------------------------------------------- #
def test_non_finite_observation_is_blocked(root: Path):
    """A NaN metric is a failure of the producer, never a value to compare."""
    fixture = build_fixture(root)
    observations = dict(fixture["observations"]["state_recheck"])
    observations["player_pos_px"] = float("nan")
    document = _rewrite_evidence(fixture, "state_recheck", observations)
    code, verdict = run(fixture, out=True, frozen_doc=document)
    assert code == 2
    assert verdict["items"]["A-01"]["status"] == "blocked"
    assert "not finite" in verdict["items"]["A-01"]["reason"]


def test_manifest_val_split_of_a_is_7241(root: Path):
    fixture = build_fixture(root)
    manifest = DatasetManifest.load(root / "dataset-a" / "manifest.json")
    assert manifest.total_frames == A_TOTAL
    _train, val = split_indices(manifest.total_frames, SEED, VAL_FRACTION)
    assert len(val) == 7241
    assert index_digest(val) == manifest.splits["val_index_sha256"] == fixture["val_index_sha256"]


def test_body_of_the_verdict_is_json_serialisable(root: Path):
    fixture = build_fixture(root)
    code, verdict = run(fixture, out=True)
    text = json.dumps(verdict)
    assert code == 0
    assert "v104-check/1" in text
    assert set(verdict["items"]) == {"A-01", "A-02", "A-03", "A-04", "A-05", "A-06"}
    for item in verdict["items"].values():
        assert set(item) == {
            "status",
            "source",
            "path",
            "hash",
            "artifacts",
            "values",
            "thresholds",
            "reason",
        }
