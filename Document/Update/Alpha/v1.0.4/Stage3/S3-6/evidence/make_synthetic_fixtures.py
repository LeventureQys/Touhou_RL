"""Build the synthetic fixtures used by the S3-6 CPU CLI evidence runs.

Nothing here is real data or a real experiment.  Two independent fixture trees are
produced:

``<out>/compare``
    Two tiny pixel datasets (8 shards x 20 frames, three regions as PNG blobs),
    their manifests, and a depth-3 v2 checkpoint.  ``dataset-b`` has no
    ``manifest.json`` on purpose: it carries ``manifest-b.json`` so the CLI
    ``--manifest`` path (the S3-1 read-only evaluation list) is exercised, and a
    missing-manifest refusal can be demonstrated by pointing at nothing.

``<out>/check``
    A freeze tree for ``check_v104.py``: ``base/`` holds the two manifests (A is
    metadata-only with 72,413 frames so its split really is the 7,241-frame one),
    two tiny checkpoints, two summaries, the eval-report, the pytest log and the
    evidence artifacts; ``pass/``, ``fail/`` and ``blocked/`` each hold a
    ``final-artifacts.json`` plus its own evidence indices, so the three exit codes
    can be observed on the same structure.

Every number in the ``check`` tree is invented; the tool judges *consistency*, not
whether v1.0.4 was accepted.  The completion report says so explicitly.

Usage::

    python make_synthetic_fixtures.py --out D:\\...\\.pytest-tmp\\S3-6\\cli
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from touhou_rl import obs, spec, vision
from touhou_rl.pipeline.manifest import (
    DatasetManifest,
    ShardRecord,
    file_digest,
    index_digest,
    split_indices,
)
from touhou_rl.vision_data import META_FIELDS
from touhou_rl.vmodels import save_vision_checkpoint
from touhou_rl.vmodels2 import VisionNetConfigV2, VisionScorerV2

REGION_SIZES = {"playfield": (8, 8), "hud_score": (2, 8), "hud_status": (8, 4)}
PIXEL_SHARDS = 8
PIXEL_FRAMES = 20
CHECK_STAGES = (0, 1, 2, 3)
A_TOTAL = 72413
A_SHARDS = 57
B_SHARDS = 4
B_FRAMES = 10
SEED = 7
VAL_FRACTION = 0.1
PER_STAGE = 128


def _sha(path: Path) -> str:
    return file_digest(path)


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _manifest(dataset: Path, records: list[ShardRecord], name: str = "manifest.json") -> Path:
    manifest = DatasetManifest.build(
        modality="vision",
        root=dataset,
        shards=records,
        frame_shape=[448, 384, 3],
        has_teacher_values=True,
        teacher={"name": "pilot", "config": {}},
        seed=SEED,
        val_fraction=VAL_FRACTION,
        stats={"synthetic": True},
        provenance={"purpose": "synthetic-s3-6-cli-fixture"},
    )
    return manifest.save(dataset / name)


def _pixel_dataset(path: Path, *, offset: int = 0) -> list[ShardRecord]:
    path.mkdir(parents=True, exist_ok=True)
    records: list[ShardRecord] = []
    for shard in range(PIXEL_SHARDS):
        stage = int(CHECK_STAGES[shard % len(CHECK_STAGES)])
        blobs: dict[str, list[bytes]] = {name: [] for name in REGION_SIZES}
        offsets: dict[str, np.ndarray] = {}
        for name, (height, width) in REGION_SIZES.items():
            running = [0]
            for row in range(PIXEL_FRAMES):
                value = (shard + offset) * 53 + row * 7 + len(name)
                image = np.full((height, width, 3), value % 256, dtype=np.uint8)
                blob = vision.encode_png(image)
                blobs[name].append(blob)
                running.append(running[-1] + len(blob))
            offsets[name] = np.asarray(running, dtype=np.int64)
        self_feat = np.zeros((PIXEL_FRAMES, obs.N_SELF), dtype=np.float32)
        self_feat[:, 0] = 0.5 + 0.001 * shard
        self_feat[:, 1] = 0.5 - 0.001 * shard
        meta = np.zeros((PIXEL_FRAMES, len(META_FIELDS)), dtype=np.float32)
        meta[:, 0] = 2.0 * np.arange(PIXEL_FRAMES)
        meta[:, 1] = float(stage)
        meta[:, 4] = 2.0
        meta[:, 5] = 2.0
        payload = {
            "region_name": np.asarray(list(REGION_SIZES), dtype="U24"),
            "region_hw": np.asarray([REGION_SIZES[name] for name in REGION_SIZES], dtype=np.int32),
            "self_feat": self_feat,
            "bullets": np.zeros((PIXEL_FRAMES, obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float16),
            "danger": np.zeros((PIXEL_FRAMES, obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8),
            "action": np.zeros((PIXEL_FRAMES, 4), dtype=np.uint8),
            "meta": meta,
        }
        for name in REGION_SIZES:
            payload[f"{name}_blob"] = np.frombuffer(b"".join(blobs[name]), dtype=np.uint8).copy()
            payload[f"{name}_offset"] = offsets[name]
        name = f"collect-{shard:05d}.npz"
        np.savez(path / name, **payload)
        records.append(
            ShardRecord(
                name=name,
                frames=PIXEL_FRAMES,
                bytes=(path / name).stat().st_size,
                sha256=_sha(path / name),
            )
        )
    return records


def _meta_dataset(path: Path, sizes: list[int]) -> list[ShardRecord]:
    """Metadata-only shards: ``stage_of``/``len`` work, pixels are never decoded."""
    path.mkdir(parents=True, exist_ok=True)
    records: list[ShardRecord] = []
    for index, frames in enumerate(sizes):
        stage = int(CHECK_STAGES[index % len(CHECK_STAGES)])
        meta = np.zeros((frames, len(META_FIELDS)), dtype=np.float32)
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
            frame=np.zeros((frames, 1, 1, 3), dtype=np.uint8),
        )
        records.append(
            ShardRecord(
                name=name,
                frames=frames,
                bytes=(path / name).stat().st_size,
                sha256=_sha(path / name),
            )
        )
    return records


def _tiny_checkpoint(path: Path, *, depth: int, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    config = VisionNetConfigV2(
        depth=depth,
        playfield_hw=REGION_SIZES["playfield"],
        hud_score_hw=REGION_SIZES["hud_score"],
        hud_status_hw=REGION_SIZES["hud_status"],
        pf_width=4,
        hud_width=2,
        trunk_hidden=16,
        state_hidden=16,
        candidate_hidden=8,
    )
    save_vision_checkpoint(
        path,
        VisionScorerV2(config),
        {"stage": "state", "epoch": 1, "history_contract": "causal-segment-v1", "depth": depth},
    )
    return path


def build_compare(out: Path) -> dict:
    base = out / "compare"
    a = base / "dataset-a"
    manifest_a = _manifest(a, _pixel_dataset(a))
    b = base / "dataset-b"
    records_b = _pixel_dataset(b, offset=0)  # byte-identical frames -> ratio ~1.0
    manifest_b = _manifest(b, records_b, name="manifest-b.json")
    checkpoint = _tiny_checkpoint(base / "state-depth3.pt", depth=3, seed=11)

    # A copy of A whose recorded validation index hash was rewritten: the shards are
    # intact, so only the split reproducibility check can catch it.
    bad = base / "dataset-a-badmanifest"
    if bad.exists():
        shutil.rmtree(bad)
    shutil.copytree(a, bad)
    bad_payload = json.loads((bad / "manifest.json").read_text(encoding="utf-8"))
    bad_payload["splits"]["val_index_sha256"] = "0" * 64
    _write_json(bad / "manifest.json", bad_payload)

    unknown = base / "unknown-arch.pt"
    torch.save(
        {
            "kind": "vision_scorer",
            "arch": "v9",
            "state_dict": {"weight": torch.zeros(1)},
            "config": {"depth": 3},
            "frame": {"H": vision.VISION_H, "W": vision.VISION_W, "C": 3, "depth": 3},
            "extra": {},
        },
        unknown,
    )

    # Real v3-st checkpoint (G-v3 approved, S3-5 model) with the C6 identity, plus a
    # deliberately stale v3 payload that declares a config layout the model does not
    # accept: the first proves the dispatch, the second proves the refusal.
    v3 = base / "v3-policy.pt"
    from touhou_rl.vmodels3 import VisionSpatialTemporalConfig, VisionSpatialTemporalPolicy

    torch.manual_seed(11)
    save_vision_checkpoint(
        v3,
        VisionSpatialTemporalPolicy(VisionSpatialTemporalConfig()),
        {
            "source_state_sha": "d" * 64,
            "teacher_fingerprint": "a" * 64,
            "manifest_sha": "b" * 64,
            "val_index_sha256": "c" * 64,
            "loss_config": {"objective": "rank", "rank_delta_px": 0.5},
            "stage": "policy",
            "epoch": 1,
            "depth": 3,
            "history_contract": "causal-segment-v1",
        },
    )
    v3_bad = base / "v3-badconfig.pt"
    torch.save(
        {
            "schema": "vision-checkpoint/3",
            "kind": "vision_scorer",
            "arch": "v3-st",
            "config": {
                "depth": 3,
                # Not the S3-5 layout (it serialises playfield_hw/hud_score_hw/
                # hud_status_hw): a stale writer must be rejected, not coerced.
                "region_sizes": {
                    "playfield": [224, 192],
                    "hud_score": [34, 216],
                    "hud_status": [126, 216],
                },
                "roi_native_hw": [128, 128],
                "local_grid_hw": [16, 16],
                "align_corners": False,
                "candidate_order_version": "focus9-move-v1",
                "score_semantics": "preference_logits",
                "history_contract": "causal-segment-v1",
                "state_head": True,
            },
            "state_dict": {"weight": torch.zeros(1)},
            "frame": {"H": vision.VISION_H, "W": vision.VISION_W, "C": 3, "depth": 3},
            "extra": {
                "source_state_sha": "",
                "teacher_fingerprint": "a" * 64,
                "manifest_sha": "b" * 64,
                "val_index_sha256": "c" * 64,
                "loss_config": {"objective": "rank"},
                "stage": "state",
            },
        },
        v3_bad,
    )
    return {
        "base": base,
        "dataset_a": a,
        "dataset_b": b,
        "dataset_a_badmanifest": bad,
        "manifest_a": manifest_a,
        "manifest_b": manifest_b,
        "checkpoint": checkpoint,
        "unknown_arch": unknown,
        "v3": v3,
        "v3_bad": v3_bad,
    }


def _expected_selection(dataset: Path, per_stage: int = PER_STAGE) -> list[int]:
    from touhou_rl.vision_data import VisionDataset

    manifest = DatasetManifest.load(dataset / "manifest.json")
    _train, val = split_indices(manifest.total_frames, SEED, VAL_FRACTION)
    with VisionDataset(dataset) as handle:
        stages = handle.stage_of(val).tolist()
    selected: list[int] = []
    for stage in sorted(set(stages)):
        rows = [index for index, value in zip(val, stages) if value == stage]
        selected.extend(rows[:per_stage])
    return selected


def build_check(out: Path) -> dict:
    root = out / "check"
    base = root / "base"
    a = base / "dataset-a"
    a_sizes = [1271] * (A_SHARDS - 1)
    a_sizes.append(A_TOTAL - sum(a_sizes))
    manifest_a = _manifest(a, _meta_dataset(a, a_sizes))
    b = base / "dataset-b"
    manifest_b = _manifest(b, _meta_dataset(b, [B_FRAMES] * B_SHARDS))
    state = _tiny_checkpoint(base / "runs" / "state-pass.pt", depth=3, seed=11)
    policy = _tiny_checkpoint(base / "runs" / "policy.pt", depth=3, seed=23)
    teacher = _write_json(base / "runs" / "teacher-values.json", {"teacher": "pilot", "candidates": 18})
    table = _write_json(base / "runs" / "experiment-table.json", {"rows": [], "synthetic": True})
    manifest_a_obj = DatasetManifest.load(manifest_a)
    val_sha = manifest_a_obj.splits["val_index_sha256"]

    state_summary = _write_json(
        base / "runs" / "state-summary.json",
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
        base / "runs" / "policy-summary.json",
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
    a_selection = _expected_selection(a)
    b_selection = _expected_selection(b)

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
                {
                    "bucket": "0-15",
                    "frames": 900,
                    "top1": 0.4,
                    "top3": 0.42,
                    "top3_tie": 0.44,
                    "near_optimal_05": 0.3,
                    "regret": 1.2,
                    "regret_p90_px": 4.1,
                },
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
    episodes = [
        {
            "episode": index,
            "steps": frames // 2,
            "frames_reported": frames,
            "terminated": True,
            "truncated": False,
            "termination_reason": "game over",
            "last_state": {"stage": 2, "frame": frames, "bullets": 0},
        }
        for index, frames in enumerate((4200, 4300, 4100), start=1)
    ]
    eval_report = _write_json(
        base / "artifacts" / "eval-report.json",
        {
            "checkpoint": ".",
            "policy": "vision",
            "difficulty": spec.LUNATIC,
            "team": 0,
            "practice_stage": None,
            "frameskip": 2,
            "vision": {
                "checkpoint": "base/runs/policy.pt",
                "arch": "v2",
                "depth": 3,
                "max_steps": 12000,
                "isolation": "I-1: policy input is regions_to_batch(env.render crops) only",
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
    pytest_log = base / "artifacts" / "pytest.log"
    pytest_log.parent.mkdir(parents=True, exist_ok=True)
    pytest_log.write_text(
        "180 passed, 2 skipped in 12.34s\n", encoding="utf-8"
    )
    pytest_obs = {"passed": 180, "failed": 0, "skipped": 2, "command_exit": 0}
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
        "manifest_sha": _sha(manifest_b),
    }
    startup_obs = {
        "real_game_started": True,
        "pixels_only": True,
        "no_crash": True,
        "no_exception_log": True,
        "ui_checked": True,
        "screenshots": ["screenshots/startup.png"],
    }
    observation_artifacts = {}
    for name, observations in (
        ("state-recheck", state_obs),
        ("policy-full-val", policy_obs),
        ("cross-replay", cross_obs),
        ("d9", d9_obs),
        ("d10", d10_obs),
        ("d11", d11_obs),
        ("startup", startup_obs),
    ):
        observation_artifacts[name] = _write_json(
            base / "artifacts" / f"{name}.json", {"observations": observations}
        )

    def write_variant(name: str, *, policy_override=None, d9_status="pass") -> Path:
        variant = root / name
        evidence_dir = variant / "evidence"
        state_index = _write_json(
            evidence_dir / "state_recheck.json",
            {
                "schema": "v104-evidence/1",
                "kind": "state_recheck",
                "status": "pass",
                "created_at": "2026-10-19T12:00:00+08:00",
                "producer": "synthetic-fixture",
                "command": "synthetic",
                "cwd": str(root),
                "exit_code": 0,
                "artifacts": [
                    {
                        "path": "base/artifacts/state-recheck.json",
                        "sha256": _sha(base / "artifacts" / "state-recheck.json"),
                    }
                ],
                "observations": state_obs,
                "reason": "",
            },
        )
        policy_index_path = evidence_dir / "policy_full_val.json"
        policy_index = {
            "schema": "v104-evidence/1",
            "kind": "policy_full_val",
            "status": "pass",
            "created_at": "2026-10-19T12:00:00+08:00",
            "producer": "synthetic-fixture",
            "command": "synthetic",
            "cwd": str(root),
            "exit_code": 0,
            "artifacts": [
                {
                    "path": "base/artifacts/policy-full-val.json",
                    "sha256": _sha(base / "artifacts" / "policy-full-val.json"),
                }
            ],
            "observations": policy_obs,
            "reason": "",
        }
        if policy_override is not None:
            policy_observations = dict(policy_obs)
            policy_observations["by_danger"] = policy_override
            # Rewrite the artifact so the index and the raw data agree.
            _write_json(base / "artifacts" / f"policy-full-val-{name}.json", {"observations": policy_observations})
            artifact = base / "artifacts" / f"policy-full-val-{name}.json"
            policy_index["observations"] = policy_observations
            policy_index["artifacts"] = [{"path": f"base/artifacts/{artifact.name}", "sha256": _sha(artifact)}]
        policy_index = _write_json(policy_index_path, policy_index)
        d9_index = _write_json(
            evidence_dir / "d9.json",
            {
                "schema": "v104-evidence/1",
                "kind": "d9",
                "status": d9_status,
                "created_at": "2026-10-19T12:00:00+08:00",
                "producer": "synthetic-fixture",
                "command": "synthetic",
                "cwd": str(root),
                "exit_code": 0 if d9_status == "pass" else 3,
                "artifacts": [
                    {
                        "path": "base/artifacts/d9.json",
                        "sha256": _sha(base / "artifacts" / "d9.json"),
                    }
                ],
                "observations": d9_obs,
                "reason": "" if d9_status == "pass" else "synthetic blocked D9 for the exit-2 fixture",
            },
        )

        def index_for(key: str, observations: dict, artifact: Path, kind_status="pass") -> dict:
            path = _write_json(
                evidence_dir / f"{key}.json",
                {
                    "schema": "v104-evidence/1",
                    "kind": key,
                    "status": kind_status,
                    "created_at": "2026-10-19T12:00:00+08:00",
                    "producer": "synthetic-fixture",
                    "command": "synthetic",
                    "cwd": str(root),
                    "exit_code": 0,
                    "artifacts": [{"path": f"base/artifacts/{artifact.name}", "sha256": _sha(artifact)}],
                    "observations": observations,
                    "reason": "",
                },
            )
            return {"path": f"{name}/evidence/{key}.json", "sha256": _sha(path)}

        evidence = {
            "state_recheck": {"path": f"{name}/evidence/state_recheck.json", "sha256": _sha(state_index)},
            "policy_full_val": {"path": f"{name}/evidence/policy_full_val.json", "sha256": _sha(policy_index)},
            "cross_replay": index_for("cross_replay", cross_obs, base / "artifacts" / "cross-replay.json"),
            "eval_report": index_for("eval_report", eval_obs, eval_report),
            "pytest": index_for("pytest", pytest_obs, pytest_log),
            "d9": {"path": f"{name}/evidence/d9.json", "sha256": _sha(d9_index)},
            "d10": index_for("d10", d10_obs, base / "artifacts" / "d10.json"),
            "d11": index_for("d11", d11_obs, base / "artifacts" / "d11.json"),
            "startup": index_for("startup", startup_obs, base / "artifacts" / "startup.json"),
        }
        frozen = {
            "schema": "v104-final/1",
            "state_checkpoint": "base/runs/state-pass.pt",
            "state_checkpoint_sha256": _sha(state),
            "state_summary": "base/runs/state-summary.json",
            "policy_checkpoint": "base/runs/policy.pt",
            "policy_checkpoint_sha256": _sha(policy),
            "policy_summary": "base/runs/policy-summary.json",
            "teacher_cache": "base/runs/teacher-values.json",
            "teacher_fingerprint": "f" * 64,
            "manifest_a": "base/dataset-a/manifest.json",
            "manifest_a_sha256": _sha(manifest_a),
            "manifest_b": "base/dataset-b/manifest.json",
            "manifest_b_sha256": _sha(manifest_b),
            "val_index_sha256": val_sha,
            "arch": "v2",
            "depth": 3,
            "seed": SEED,
            "experiment_table": "base/runs/experiment-table.json",
            "selection_reason": "synthetic: lowest 0-15 bucket regret on A",
            "frozen_at": "2026-10-19T12:00:00+08:00",
            "external_guard_uses": 0,
            "evidence": evidence,
        }
        return _write_json(variant / "final-artifacts.json", frozen)

    return {
        "root": root,
        "pass": write_variant("pass"),
        "fail": write_variant(
            "fail",
            policy_override=[{"bucket": "0-15", "frames": 900, "top3": 0.42, "regret": 2.5}],
        ),
        "zerobucket": write_variant(
            "zerobucket",
            policy_override=[{"bucket": "0-15", "frames": 0, "top3": 0.0, "regret": 0.0}],
        ),
        "blocked": write_variant(
            "blocked",
            policy_override=[{"bucket": "0-15", "frames": 900, "top3": 0.42, "regret": 2.5}],
            d9_status="blocked",
        ),
        "a_selection": a_selection,
        "b_selection": b_selection,
        "val_index_sha256": val_sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build synthetic fixtures for S3-6 CLI evidence")
    parser.add_argument("--out", required=True)
    parser.add_argument("--clean", action="store_true", help="delete the output directory first")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    compare = build_compare(out)
    check = build_check(out)
    summary = {
        "schema": "s3-6-synthetic-fixtures/1",
        "note": (
            "synthetic fixtures only: they exercise the tools' contract checks. They are not "
            "evidence that v1.0.4 passed A-01..A-06."
        ),
        "compare": {
            "dataset_a": str(compare["dataset_a"]),
            "dataset_b": str(compare["dataset_b"]),
            "dataset_a_badmanifest": str(compare["dataset_a_badmanifest"]),
            "manifest_a": str(compare["manifest_a"]),
            "manifest_b": str(compare["manifest_b"]),
            "checkpoint": str(compare["checkpoint"]),
            "unknown_arch": str(compare["unknown_arch"]),
            "v3_policy": str(compare["v3"]),
            "v3_bad_config": str(compare["v3_bad"]),
            "checkpoint_sha256": _sha(compare["checkpoint"]),
        },
        "check": {
            "root": str(check["root"]),
            "pass": str(check["pass"]),
            "fail": str(check["fail"]),
            "zerobucket": str(check["zerobucket"]),
            "blocked": str(check["blocked"]),
            "val_index_sha256": check["val_index_sha256"],
            "a_selected_frames": len(check["a_selection"]),
            "b_selected_frames": len(check["b_selection"]),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
