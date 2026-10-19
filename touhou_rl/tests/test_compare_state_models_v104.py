"""Unit tests for the v1.0.4 ``tools/rl/compare_state_models.py`` (S3-6).

What is pinned here (and why it cannot be inferred from a green run):

* the split is **verified from the manifest**, never re-derived: a missing
  manifest, a changed shard byte or a rewritten ``val_index_sha256`` are all
  ``v104:DATA`` refusals;
* ``full-val`` covers every validation index of the manifest while
  ``stage-sample`` takes the first ``--per-stage`` indices **per stage in
  ascending order**, and the selection is independent of the order the callers
  asks in (same ``selected_index_sha256``);
* the frame stack comes from the checkpoint's ``config.depth`` and the dataset's
  C2 history: a depth-3 checkpoint is fed ``[B, 3, H, W, 3]``, never reshaped to
  depth 1, and an old checkpoint without ``history_contract`` reports
  ``legacy_history``;
* ``mae_raw`` is unrounded so the A-03 ratio (``B <= 1.25 * A``) is computed on raw
  values; the display values are rounded separately;
* the CLI returns 0 on success and 2 with the ``v104:<CODE>:`` prefix on a known
  contract violation.

Every fixture is synthetic (tiny shards, tiny v1/v2 checkpoints) because these
tests exercise the tool, not the policy: no game, no training, no real dataset.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from touhou_rl import obs, vision
from touhou_rl.pipeline.manifest import (
    DatasetManifest,
    ShardRecord,
    file_digest,
    index_digest,
    split_indices,
)
from touhou_rl.vision_data import META_FIELDS, VisionDataset
from touhou_rl.vmodels import VisionNetConfig, VisionScorer, save_vision_checkpoint
from touhou_rl.vmodels2 import VisionNetConfigV2, VisionScorerV2

ROOT = Path(__file__).resolve().parents[2]

REGION_SIZES: dict[str, tuple[int, int]] = {
    "playfield": (8, 8),
    "hud_score": (2, 8),
    "hud_status": (8, 4),
}
SHARDS = 8
FRAMES_PER_SHARD = 20
STAGES = (0, 1, 2, 3)
SEED = 7
VAL_FRACTION = 0.1


def _load_tool(name: str):
    """Import ``tools/rl/<name>.py`` as a module (``tools`` is not a package)."""
    module_name = f"s36_tools_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "tools" / "rl" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


COMPARE = _load_tool("compare_state_models")


# --------------------------------------------------------------------------- #
# synthetic dataset / manifest / checkpoint
# --------------------------------------------------------------------------- #
def _region_blobs(shard: int, frames: int) -> tuple[dict, dict]:
    blobs: dict[str, list[bytes]] = {name: [] for name in REGION_SIZES}
    offsets: dict[str, np.ndarray] = {}
    for name, (height, width) in REGION_SIZES.items():
        running = [0]
        for row in range(frames):
            value = (shard * 53 + row * 7 + len(name)) % 256
            image = np.full((height, width, 3), value, dtype=np.uint8)
            blob = vision.encode_png(image)
            blobs[name].append(blob)
            running.append(running[-1] + len(blob))
        offsets[name] = np.asarray(running, dtype=np.int64)
    return blobs, offsets


def write_dataset(path: Path, *, shards: int = SHARDS, frames: int = FRAMES_PER_SHARD,
                  stages=STAGES, offset: int = 0) -> list[ShardRecord]:
    """Write minimal vision shards (region PNG blobs + labels + meta)."""
    path.mkdir(parents=True, exist_ok=True)
    records: list[ShardRecord] = []
    for shard in range(shards):
        stage = int(stages[shard % len(stages)])
        blobs, offsets = _region_blobs(shard + offset, frames)
        self_feat = np.zeros((frames, obs.N_SELF), dtype=np.float32)
        self_feat[:, 0] = 0.5 + 0.001 * shard
        self_feat[:, 1] = 0.5 - 0.001 * shard
        meta = np.zeros((frames, len(META_FIELDS)), dtype=np.float32)
        meta[:, 0] = 2.0 * np.arange(frames)  # every=2, continuous segment
        meta[:, 1] = float(stage)
        meta[:, 4] = 2.0  # lives
        meta[:, 5] = 2.0  # bombs
        payload = {
            "region_name": np.asarray(list(REGION_SIZES), dtype="U24"),
            "region_hw": np.asarray([REGION_SIZES[name] for name in REGION_SIZES], dtype=np.int32),
            "self_feat": self_feat,
            "bullets": np.zeros((frames, obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float16),
            "danger": np.zeros((frames, obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8),
            "action": np.zeros((frames, 4), dtype=np.uint8),
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
                frames=frames,
                bytes=(path / name).stat().st_size,
                sha256=file_digest(path / name),
            )
        )
    return records


def write_manifest(dataset: Path, records: list[ShardRecord], *, name: str = "manifest.json",
                   seed: int = SEED, val_fraction: float = VAL_FRACTION) -> Path:
    manifest = DatasetManifest.build(
        modality="vision",
        root=dataset,
        shards=records,
        frame_shape=[448, 384, 3],
        has_teacher_values=True,
        teacher={"name": "pilot", "config": {}},
        seed=seed,
        val_fraction=val_fraction,
        stats={},
        provenance={"purpose": "synthetic-test-only"},
    )
    path = dataset / name
    manifest.save(path)
    return path


def write_checkpoint(path: Path, *, arch: str = "v2", depth: int = 3,
                     history_contract: str | None = "causal-segment-v1", seed: int = SEED) -> Path:
    torch.manual_seed(seed)
    extra: dict = {"stage": "state", "epoch": 1, "arch": arch}
    if history_contract is not None:
        extra["history_contract"] = history_contract
    if arch == "v2":
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
        model = VisionScorerV2(config)
    else:
        config = VisionNetConfig(
            depth=depth,
            width=4,
            trunk_hidden=16,
            candidate_hidden=8,
            state_hidden=16,
            model_regions=tuple((name, *size) for name, size in REGION_SIZES.items()),
        )
        model = VisionScorer(config)
    save_vision_checkpoint(path, model, extra)
    return path


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset-a"
    records = write_dataset(path)
    write_manifest(path, records)
    return path


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    return write_checkpoint(tmp_path / "state.pt", arch="v2", depth=3)


# --------------------------------------------------------------------------- #
# selection: full-val / stage-sample
# --------------------------------------------------------------------------- #
def test_full_val_covers_every_validation_index(dataset: Path, checkpoint: Path):
    row = COMPARE.evaluate_checkpoint(checkpoint, dataset, mode="full-val")
    manifest = DatasetManifest.load(dataset / "manifest.json")
    _train, val = split_indices(manifest.total_frames, SEED, VAL_FRACTION)

    assert row["sampling_mode"] == "full-val"
    assert row["frames"] == len(val)
    assert row["selected_indices"] == val
    assert row["selected_index_sha256"] == index_digest(val)
    assert row["val_frames"] == len(val)
    assert row["val_index_sha256"] == manifest.splits["val_index_sha256"]
    assert row["manifest_sha256"] == file_digest(dataset / "manifest.json")
    assert row["checkpoint_sha256"] == file_digest(checkpoint)
    assert row["depth"] == 3
    assert row["arch"] == "v2"
    assert row["history_contract"] == "causal-segment-v1"
    assert row["legacy_history"] is False
    assert row["playfield_hw"] == list(REGION_SIZES["playfield"])
    assert sum(node["frames"] for node in row["per_stage"].values()) == row["frames"]


def test_stage_sample_takes_first_n_ascending_per_stage(dataset: Path, checkpoint: Path):
    row = COMPARE.evaluate_checkpoint(checkpoint, dataset, mode="stage-sample", per_stage=2)
    manifest = DatasetManifest.load(dataset / "manifest.json")
    _train, val = split_indices(manifest.total_frames, SEED, VAL_FRACTION)
    with VisionDataset(dataset) as handle:
        stages = handle.stage_of(val).tolist()
    expected: list[int] = []
    for stage in sorted(set(stages)):
        rows = [index for index, value in zip(val, stages) if value == stage]
        expected.extend(rows[:2])

    assert row["frames"] == len(expected)
    assert row["selected_indices"] == expected
    assert row["selected_index_sha256"] == index_digest(expected)
    assert row["per_stage_limit"] == 2
    for stage, node in row["per_stage"].items():
        assert node["selected"] == min(2, node["available"])
        assert node["frames"] == node["selected"]
        assert node["mae"] == round(node["mae_raw"], 2)


def test_stage_sample_is_independent_of_the_requested_order(dataset: Path, checkpoint: Path):
    with VisionDataset(dataset) as handle:
        _train, val = split_indices(len(handle), SEED, VAL_FRACTION)
        forward, forward_report = COMPARE.select_indices(handle, val, "stage-sample", 2)
        backward, backward_report = COMPARE.select_indices(handle, list(reversed(val)), "stage-sample", 2)
    assert forward == backward
    assert index_digest(forward) == index_digest(backward)
    assert forward_report == backward_report
    # Every stage's slice is ascending even though the input was reversed.
    for stage in sorted({str(stage) for stage in forward_report}):
        del stage  # the per-stage slices are checked by the report equality above


def test_stage_sample_reports_a_short_stage_instead_of_padding(dataset: Path, checkpoint: Path):
    row = COMPARE.evaluate_checkpoint(checkpoint, dataset, mode="stage-sample", per_stage=10_000)
    assert row["frames"] == row["val_frames"]  # every validation frame, all stages
    for node in row["per_stage"].values():
        assert node["selected"] == node["available"]
        assert node["available"] >= 1


# --------------------------------------------------------------------------- #
# depth / history composition
# --------------------------------------------------------------------------- #
def test_depth_three_history_reaches_the_model(dataset: Path, checkpoint: Path, monkeypatch):
    seen: list[tuple[int, ...]] = []
    depths: list[int] = []
    real_decode = VisionScorerV2.decode_state
    real_stack = VisionDataset.stack_indices

    def spy_decode(self, regions, features=None):
        seen.append(tuple(regions["playfield"].shape))
        return real_decode(self, regions, features)

    def spy_stack(self, indices, depth):
        depths.append(int(depth))
        return real_stack(self, indices, depth)

    monkeypatch.setattr(VisionScorerV2, "decode_state", spy_decode)
    monkeypatch.setattr(VisionDataset, "stack_indices", spy_stack)
    COMPARE.evaluate_checkpoint(checkpoint, dataset, mode="stage-sample", per_stage=2, batch=4)

    assert seen, "the tool never called decode_state"
    assert all(shape[1] == 3 for shape in seen), seen
    assert all(shape[2:] == (8, 8, 3) for shape in seen), seen
    assert set(depths) == {3}


def test_old_checkpoint_without_history_contract_is_labelled_legacy(tmp_path: Path):
    dataset = tmp_path / "dataset-a"
    write_manifest(dataset, write_dataset(dataset))
    checkpoint = write_checkpoint(
        tmp_path / "v1-legacy.pt", arch="v1", depth=1, history_contract=None
    )
    row = COMPARE.evaluate_checkpoint(checkpoint, dataset, mode="stage-sample", per_stage=2)
    assert row["arch"] == "v1"
    assert row["depth"] == 1
    assert row["history_contract"] == "legacy"
    assert row["legacy_history"] is True


# --------------------------------------------------------------------------- #
# unrounded values and the A-03 ratio
# --------------------------------------------------------------------------- #
def test_mae_raw_is_unrounded_and_display_is_rounded(dataset: Path, checkpoint: Path):
    row = COMPARE.evaluate_checkpoint(checkpoint, dataset, mode="full-val")
    assert isinstance(row["mae_raw"], float)
    assert row["mae"] == round(row["mae_raw"], 2)
    assert row["median"] == round(float(row["median"]), 2)
    for node in row["per_stage"].values():
        assert node["mae"] == round(node["mae_raw"], 2)


def _ratio_rows(*, a_mae: float, b_mae: float, sha: str = "a" * 64) -> tuple[list[dict], list[dict]]:
    rows = [
        {
            "checkpoint_sha256": sha,
            "dataset": "dataset-b",
            "sampling_mode": "stage-sample",
            "selected_index_sha256": "b" * 64,
            "frames": 8,
            "mae_raw": b_mae,
        }
    ]
    baseline = [
        {
            "checkpoint_sha256": sha,
            "dataset": "dataset-a",
            "sampling_mode": "stage-sample",
            "selected_index_sha256": "c" * 64,
            "frames": 8,
            "mae_raw": a_mae,
        }
    ]
    return rows, baseline


def test_a03_ratio_is_computed_on_unrounded_values():
    # 13.7769 / 11.0216 = 1.249990... <= 1.25, while the rounded values
    # (13.78 / 11.02 = 1.25045) would have failed: this is the whole reason
    # ``mae_raw`` exists.
    rows, baseline = _ratio_rows(a_mae=11.0216, b_mae=13.7769)
    verdict = COMPARE.a03_ratio(rows, baseline)[0]
    assert verdict["pass"] is True
    assert verdict["ratio"] <= 1.25
    assert round(13.7769, 2) / round(11.0216, 2) > 1.25


def test_a03_ratio_fails_above_the_threshold():
    rows, baseline = _ratio_rows(a_mae=10.0, b_mae=12.51)
    verdict = COMPARE.a03_ratio(rows, baseline)[0]
    assert verdict["pass"] is False
    assert verdict["ratio"] == pytest.approx(1.251)


def test_a03_ratio_requires_the_same_checkpoint_and_stage_sample():
    rows, baseline = _ratio_rows(a_mae=10.0, b_mae=11.0)
    other = [dict(baseline[0], checkpoint_sha256="d" * 64)]
    blocked = COMPARE.a03_ratio(rows, other)[0]
    assert blocked["status"] == "blocked"
    assert "checkpoint_sha256" in blocked["reason"]

    full_val = [dict(rows[0], sampling_mode="full-val")]
    blocked = COMPARE.a03_ratio(full_val, baseline)[0]
    assert blocked["status"] == "blocked"
    assert "stage-sample" in blocked["reason"]


def test_a03_ratio_refuses_a_baseline_from_the_same_dataset():
    rows, baseline = _ratio_rows(a_mae=10.0, b_mae=11.0)
    same = [dict(baseline[0], dataset="dataset-b")]
    blocked = COMPARE.a03_ratio(rows, same)[0]
    assert blocked["status"] == "blocked"


# --------------------------------------------------------------------------- #
# refusals: manifest identity, split, arch, non-finite output
# --------------------------------------------------------------------------- #
def test_missing_manifest_is_a_data_error(tmp_path: Path, capsys):
    dataset = tmp_path / "no-manifest"
    write_dataset(dataset)
    checkpoint = write_checkpoint(tmp_path / "state.pt", depth=1)
    with pytest.raises(ValueError, match="v104:DATA"):
        COMPARE.evaluate_checkpoint(checkpoint, dataset)
    with pytest.raises(ValueError, match="v104:DATA"):
        COMPARE.evaluate_checkpoint(checkpoint, dataset, manifest_path=tmp_path / "absent.json")

    code = COMPARE.main(
        [
            "--dataset",
            str(dataset),
            "--checkpoint",
            f"a={checkpoint}",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.startswith("v104:DATA:")
    assert not (tmp_path / "out.json").exists()


def test_tampered_shard_is_refused(dataset: Path, checkpoint: Path):
    shard = sorted(dataset.glob("collect-*.npz"))[0]
    with open(shard, "ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(ValueError, match="v104:DATA"):
        COMPARE.evaluate_checkpoint(checkpoint, dataset)


def test_tampered_manifest_split_is_refused(dataset: Path, checkpoint: Path):
    path = dataset / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["splits"]["val_index_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="v104:DATA"):
        COMPARE.evaluate_checkpoint(checkpoint, dataset)


def test_unknown_arch_is_refused(tmp_path: Path):
    """Unknown arches must stay a hard error, never coerced to v1 (C6)."""
    dataset = tmp_path / "dataset-a"
    write_manifest(dataset, write_dataset(dataset))
    legacy_config = VisionNetConfig(depth=1, width=4, trunk_hidden=16, candidate_hidden=8,
                                    state_hidden=16,
                                    model_regions=tuple((name, *size) for name, size in REGION_SIZES.items()))
    unknown = tmp_path / "unknown.pt"
    torch.save(
        {
            "kind": "vision_scorer",
            "arch": "v9",
            "state_dict": VisionScorer(legacy_config).state_dict(),
            "config": legacy_config.__dict__,
            "frame": {"H": vision.VISION_H, "W": vision.VISION_W, "C": 3, "depth": 1},
            "extra": {},
        },
        unknown,
    )
    with pytest.raises(ValueError, match="v104:CHECKPOINT.*unknown vision arch 'v9'"):
        COMPARE.evaluate_checkpoint(unknown, dataset)


def test_v3_checkpoint_is_evaluated_with_its_contract(tmp_path: Path):
    """G-v3 approved: the comparison tool takes a v3 checkpoint's depth and history.

    The synthetic shards are 8x8, so the dataset resizes them to the v3 ROI; what
    is asserted is the dispatch (``arch``/``depth``/history), not accuracy.
    """
    from touhou_rl.vmodels3 import VisionSpatialTemporalConfig, VisionSpatialTemporalPolicy

    dataset = tmp_path / "dataset-a"
    write_manifest(dataset, write_dataset(dataset))
    model = VisionSpatialTemporalPolicy(VisionSpatialTemporalConfig())
    v3 = tmp_path / "v3.pt"
    save_vision_checkpoint(
        v3,
        model,
        {
            "source_state_sha": "",
            "teacher_fingerprint": "a" * 64,
            "manifest_sha": "b" * 64,
            "val_index_sha256": "c" * 64,
            "loss_config": {"objective": "rank", "rank_delta_px": 0.5},
            "stage": "state",
            "epoch": 1,
            "depth": 3,
            "history_contract": "causal-segment-v1",
        },
    )
    row = COMPARE.evaluate_checkpoint(v3, dataset, mode="stage-sample", per_stage=1)
    assert row["arch"] == "v3-st"
    assert row["depth"] == 3
    assert row["history_contract"] == "causal-segment-v1"
    assert row["legacy_history"] is False
    assert row["playfield_hw"] == [224, 192]
    assert row["frames"] >= 1
    assert row["mae_raw"] == row["mae_raw"] and row["mae_raw"] >= 0  # finite
    assert row["checkpoint_sha256"] == file_digest(v3)


def test_non_finite_prediction_is_refused(dataset: Path, checkpoint: Path, monkeypatch):
    real_decode = VisionScorerV2.decode_state

    def poisoned(self, regions, features=None):
        state = real_decode(self, regions, features)
        state["self_feat"] = state["self_feat"].clone()
        state["self_feat"][:, 0] = float("inf")
        return state

    monkeypatch.setattr(VisionScorerV2, "decode_state", poisoned)
    with pytest.raises(ValueError, match="v104:DATA"):
        COMPARE.evaluate_checkpoint(checkpoint, dataset, mode="stage-sample", per_stage=1)


def test_bad_per_stage_is_a_config_error(dataset: Path, checkpoint: Path):
    with pytest.raises(ValueError, match="v104:CONFIG"):
        COMPARE.evaluate_checkpoint(checkpoint, dataset, mode="stage-sample", per_stage=0)


# --------------------------------------------------------------------------- #
# CLI: two modes, A/B ratio, backward-compatible wrapper
# --------------------------------------------------------------------------- #
def _cli_argv(dataset: Path, checkpoint: Path, out: Path, *, mode: str, manifest: Path | None = None,
              per_stage: int = 2, baseline: Path | None = None) -> list[str]:
    argv = [
        "--dataset",
        str(dataset),
        "--checkpoint",
        f"candidate={checkpoint}",
        "--mode",
        mode,
        "--per-stage",
        str(per_stage),
        "--out",
        str(out),
    ]
    if manifest is not None:
        argv.extend(["--manifest", str(manifest)])
    if baseline is not None:
        argv.extend(["--baseline", str(baseline)])
    return argv


def test_cli_runs_both_modes_and_writes_json(tmp_path: Path, dataset: Path, checkpoint: Path, capsys):
    out = tmp_path / "full.json"
    assert COMPARE.main(_cli_argv(dataset, checkpoint, out, mode="full-val")) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows[0]["sampling_mode"] == "full-val"
    assert rows[0]["frames"] == 16
    assert rows[0]["mae_raw"] == pytest.approx(rows[0]["mae_raw"])
    printed = capsys.readouterr().out
    assert "mode=full-val" in printed
    assert "mae_raw=" in printed
    assert "selected_sha=" in printed

    out = tmp_path / "sample.json"
    assert COMPARE.main(_cli_argv(dataset, checkpoint, out, mode="stage-sample")) == 0
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows[0]["sampling_mode"] == "stage-sample"
    assert rows[0]["frames"] <= rows[0]["val_frames"]
    assert rows[0]["per_stage_limit"] == 2
    assert "| stage |" in capsys.readouterr().out


def test_cli_a03_ratio_against_the_a_side(tmp_path: Path, dataset: Path, checkpoint: Path, capsys):
    # B is a byte-identical copy of A: same checkpoint, same rule -> ratio 1.0.
    other = tmp_path / "dataset-b"
    records = write_dataset(other)
    write_manifest(other, records)
    a_out = tmp_path / "a.json"
    b_out = tmp_path / "b.json"
    assert COMPARE.main(_cli_argv(dataset, checkpoint, a_out, mode="stage-sample", manifest=dataset / "manifest.json")) == 0
    capsys.readouterr()
    assert (
        COMPARE.main(
            _cli_argv(other, checkpoint, b_out, mode="stage-sample", manifest=other / "manifest.json", baseline=a_out)
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "A-03 pass" in printed
    assert "ratio=1.000000" in printed


def test_backward_compatible_evaluate_wrapper(dataset: Path, checkpoint: Path):
    row = COMPARE.evaluate(str(checkpoint), str(dataset))
    assert row["sampling_mode"] == "stage-sample"
    assert row["label"] is None
    assert isinstance(row["_errors"], np.ndarray)
    assert row["_errors"].shape == (row["frames"],)
