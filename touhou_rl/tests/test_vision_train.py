"""Unit tests for the v1.0.3 training entry point (``touhou_rl.vision_train``).

CPU only, synthetic data: every case builds a 24-frame, three-region dataset in
``tmp_path`` so the layout the trainer expects (region PNG blobs, privileged
labels, ``meta[:,1]`` = stage, optional ``manifest.json``) is exercised without a
GPU and without the 23 GB archive.

What is pinned here, and why each is worth a test:

* the validation split comes from ``manifest.json`` and a wrong split hash aborts
  training (D1) - otherwise "held out" means whatever the code derived that day;
* ``--train-fraction`` shrinks the training side only, and refuses to run when the
  manifest could not be verified (D5 / Q3);
* selection keys: ``state.pt`` by position, ``state-pass.pt`` once for the joint
  A-01 pass (D2), ``policy.pt`` by regret, and ``best`` in the summary is the same
  epoch as the saved weights (D4);
* ``bullet_count_mae`` flows through :mod:`touhou_rl.vision_metrics`, i.e. reads
  ``self_feat`` index 14 (Q1); the removed in-file formula read index 13;
* training and validation batch limits are independent, so a screening run cannot
  shrink the validation set (design doc 7.4);
* ``runs/pilot-v1`` cannot be written to (D4);
* ``--arch v1`` still trains and simply omits the heatmap metric.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from touhou_rl import obs, vision, vision_metrics, vision_train
from touhou_rl.pipeline.manifest import DatasetManifest, ManifestError, ShardRecord, file_digest
from touhou_rl.vision_data import META_FIELDS, VisionShardWriter
from touhou_rl.vmodels import load_vision_checkpoint

STATE_METRIC_KEYS = (
    "val_self_mae",
    "val_player_pos_px",
    "val_player_pos_p95_px",
    "val_ray_mae",
    "val_bullet_count_mae",
    "val_danger_mae_log",
    "val_danger_corr",
    "val_danger_corr_perframe_mean",
    "val_danger_corr_perframe_median",
    "val_danger_corr_perframe_valid",
    "val_danger_iou",
    "val_danger_cell_acc",
)
BUCKET_LABELS = ["0-15", "15-30", "30-60", "60-120", "120+"]
# C3: every bucket carries these eight keys (the v1.0.3 four plus the v1.0.4
# top3 / top3_tie / near_optimal_05 / regret_p90_px diagnostics).  The set is
# pinned exactly - a `<=` check would silently accept a dropped diagnostic.
BUCKET_KEYS = {
    "bucket",
    "frames",
    "top1",
    "top3",
    "top3_tie",
    "near_optimal_05",
    "regret_px",
    "regret_p90_px",
}
# Small on purpose: 24 frames at batch 2 give several batches per epoch while the
# whole file still runs in seconds on CPU.  The archive stays tiny; the trainer
# resizes to the model's region sizes anyway.
REGION_HW = {"playfield": (56, 48), "hud_score": (17, 54), "hud_status": (31, 54)}
STAGES = (0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2)


def _png(value: int, height: int, width: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = value % 256
    image[: max(1, height // 4), : max(1, width // 4), 1] = (value * 3) % 256
    return vision.encode_png(image)


def _labels(index: int, *, bullet_channel_fix: bool) -> tuple[obs.Observation, dict]:
    rng = np.random.default_rng(1000 + index)
    self_feat = rng.random(obs.N_SELF, dtype=np.float32) * 0.5
    if bullet_channel_fix:
        # Q1: index 13 is the dead stage-clear flag, index 14 is the bullet count.
        self_feat[13] = 1.0
        self_feat[14] = 0.0
    bullets = rng.random((obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float32).astype(np.float16)
    bullets[:, 5] = (rng.random(obs.N_BULLETS) > 0.25).astype(np.float16)
    observation = obs.Observation(
        self_feat=self_feat,
        bullets=bullets,
        danger_map=rng.integers(0, 8, size=(obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8),
    )
    return observation, {name: 0 for name in META_FIELDS}


def _synthetic_dataset(
    root: Path,
    frames: int = 24,
    *,
    bullet_channel_fix: bool = False,
) -> Path:
    """Write a tiny vision dataset (``root/ds``) and return its directory."""
    dataset_dir = root / "ds"
    writer = VisionShardWriter(dataset_dir, shard_size=5, tag="syn")
    for index in range(frames):
        observation, meta = _labels(index, bullet_channel_fix=bullet_channel_fix)
        meta["frame"] = index
        meta["stage"] = STAGES[index % len(STAGES)]
        regions = {name: _png(index * 9 + offset, *hw) for offset, (name, hw) in enumerate(REGION_HW.items())}
        writer.add(regions, observation, (index % 9, 1, 0, 0), meta)
    writer.close()
    return dataset_dir


def _write_manifest(dataset_dir: Path, *, seed: int = 7, val_fraction: float = 0.2) -> DatasetManifest:
    records = []
    for path in sorted(dataset_dir.glob("*.npz")):
        with np.load(path) as data:
            frames = int(data["action"].shape[0])
        records.append(
            ShardRecord(name=path.name, frames=frames, bytes=path.stat().st_size, sha256=file_digest(path))
        )
    manifest = DatasetManifest.build(
        modality="vision",
        root=dataset_dir,
        shards=records,
        frame_shape=None,
        has_teacher_values=False,
        teacher={},
        seed=seed,
        val_fraction=val_fraction,
        stats={},
        provenance={},
    )
    manifest.save(dataset_dir / "manifest.json")
    return manifest


def _cfg(dataset_dir: Path, out_dir: Path, **overrides) -> vision_train.VisionTrainConfig:
    """Config with small widths (CPU speed) and a 4-frame validation split."""
    values = {
        "dataset": dataset_dir,
        "out_dir": out_dir,
        "arch": "v2",
        "stage": "state",
        "epochs_state": 1,
        "epochs_policy": 1,
        "batch_size": 2,
        "depth": 1,
        "device": "cpu",
        "val_fraction": 0.2,
        "allow_missing_manifest": True,
        "max_train_batches": 2,
        "max_val_batches": 0,
        "pf_width": 8,
        "hud_width": 4,
        "width": 8,
        "trunk_hidden": 32,
    }
    values.update(overrides)
    return vision_train.VisionTrainConfig(**values)


def _read_summary(out_dir: Path) -> dict:
    return json.loads((out_dir / "vision-summary.json").read_text(encoding="utf-8"))


def _extra(path: Path) -> dict:
    return load_vision_checkpoint(path, map_location="cpu")[1]


# --------------------------------------------------------------------------- #
# state / policy stage output
# --------------------------------------------------------------------------- #
def test_state_stage_runs_and_writes_summary(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    out = tmp_path / "run"
    summary = vision_train.train(_cfg(dataset_dir, out, arch="v2", stage="state", epochs_state=1))

    assert (out / "vision-summary.json").is_file()
    assert summary["schema"] == "vision-summary/2"
    assert summary["arch"] == "v2"
    history = summary["stages"]["state"]["history"]
    assert len(history) == 1
    row = history[0]
    missing = [key for key in STATE_METRIC_KEYS if key not in row]
    assert missing == []
    # v2 adds the heatmap diagnostic; val_frames is the count actually validated.
    assert "val_heatmap_hit_rate" in row
    assert row["val_frames"] == summary["val_frames"] == 4
    assert summary["manifest_verified"] is False
    assert (out / "state.pt").is_file()


def test_summary_top_level_schema_and_config(tmp_path):
    """S3-5 reads ``config.max_train_batches`` to tell screening from full runs."""
    dataset_dir = _synthetic_dataset(tmp_path)
    out = tmp_path / "run"
    summary = vision_train.train(
        _cfg(dataset_dir, out, arch="v2", stage="state", epochs_state=1, max_train_batches=1)
    )

    required = {
        "schema",
        "arch",
        "note",
        "config",
        "dataset",
        "frames",
        "train_frames",
        "val_frames",
        "train_fraction",
        "manifest_verified",
        "val_index_sha256",
        "device",
        "depth",
        "parameters",
        "frame_shape",
        "started",
        "finished",
        "stages",
    }
    assert required - set(summary) == set()
    config = summary["config"]
    for key in (
        "stage",
        "epochs_state",
        "epochs_policy",
        "batch_size",
        "lr",
        "depth",
        "max_train_batches",
        "max_val_batches",
        "heatmap_sigma",
        "w_heatmap",
        "w_self",
        "w_danger",
        "pf_width",
        "hud_width",
    ):
        assert key in config, key
    assert config["max_train_batches"] == 1
    assert config["max_val_batches"] == 0
    assert summary["frame_shape"]["playfield"] == [224, 192, vision.VISION_CHANNELS]
    assert summary["train_fraction"] == 1.0


def test_policy_stage_records_by_danger_every_epoch(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    out = tmp_path / "run"
    summary = vision_train.train(
        _cfg(dataset_dir, out, stage="policy", epochs_policy=2, max_train_batches=1)
    )

    history = summary["stages"]["policy"]["history"]
    assert [row["epoch"] for row in history] == [1, 2]
    for row in history:
        assert [bucket["bucket"] for bucket in row["by_danger"]] == BUCKET_LABELS
        for bucket in row["by_danger"]:
            # Exact 8-key set (C3): every bucket carries the new diagnostics too.
            assert set(bucket) == BUCKET_KEYS
            for key in sorted(BUCKET_KEYS - {"bucket"}):
                value = bucket[key]
                assert isinstance(value, (int, float)) and not isinstance(value, bool), (key, value)
                assert math.isfinite(float(value)), (key, value)
            assert int(bucket["frames"]) >= 0
        assert row["val_frames"] == 4
        assert row["val_regret_px_bucket_0_15"] == row["by_danger"][0]["regret_px"]
    assert (out / "policy.pt").is_file()


def test_best_node_matches_saved_checkpoint_epoch(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    out = tmp_path / "run"
    summary = vision_train.train(
        _cfg(dataset_dir, out, stage="both", epochs_state=3, epochs_policy=2, max_train_batches=1)
    )

    state = summary["stages"]["state"]
    best_epoch = state["best"]["epoch"]
    assert best_epoch in [row["epoch"] for row in state["history"]]
    # The best node is the *whole* row of that epoch, not the last epoch's numbers.
    assert state["best"] == state["history"][best_epoch - 1]
    assert _extra(out / "state.pt")["epoch"] == best_epoch

    policy = summary["stages"]["policy"]
    policy_epoch = policy["best"]["epoch"]
    assert policy["best"] == policy["history"][policy_epoch - 1]
    assert _extra(out / "policy.pt")["epoch"] == policy_epoch
    # `final` is the last epoch and is allowed to differ from `best`.
    assert policy["final"] == policy["history"][-1]


def test_state_pass_written_on_joint_threshold(tmp_path, monkeypatch):
    """D2: the joint A-01 pass must survive a later, better-position epoch."""
    dataset_dir = _synthetic_dataset(tmp_path)
    scripted = [_fake_state_metrics(9.0, 0.85), _fake_state_metrics(8.0, 0.70)]
    calls: list[int] = []

    def fake_all_state_metrics(pred_self, pred_danger_log, truth_self, truth_danger, *, heatmap_logits=None):
        calls.append(len(calls))
        return dict(scripted[min(len(calls) - 1, len(scripted) - 1)])

    monkeypatch.setattr(vision_metrics, "all_state_metrics", fake_all_state_metrics)
    out = tmp_path / "run"
    summary = vision_train.train(
        _cfg(dataset_dir, out, stage="state", epochs_state=2, max_train_batches=1, max_val_batches=1)
    )

    assert len(calls) == 2
    pass_extra = _extra(out / "state-pass.pt")
    assert pass_extra["epoch"] == 1
    assert pass_extra["stage"] == "state-pass"
    assert pass_extra["a01"] == {"pos_px": 9.0, "perframe_median": 0.85, "epoch": 1}
    # position-wise the later epoch wins, so state.pt holds epoch 2 ...
    assert _extra(out / "state.pt")["epoch"] == 2
    assert summary["stages"]["state"]["best"]["epoch"] == 2
    # ... and the joint pass is neither overwritten nor lost from the summary.
    assert summary["stages"]["state"]["pass_epoch"] == 1
    assert _extra(out / "state-bestdanger.pt")["epoch"] == 1


def _fake_state_metrics(pos_px: float, perframe_median: float) -> dict:
    return {
        "self_mae": 0.1,
        "player_pos_px": pos_px,
        "player_pos_p95_px": pos_px * 2.0,
        "ray_mae": 0.2,
        "bullet_count_mae": 3.0,
        "danger_mae_log": 0.3,
        "danger_corr": 0.4,
        "danger_corr_perframe_mean": perframe_median,
        "danger_corr_perframe_median": perframe_median,
        "danger_corr_perframe_valid": 4,
        "danger_iou": 0.5,
        "danger_cell_acc": 0.6,
        "heatmap_hit_rate": 0.7,
    }


def test_v1_arch_runs_and_omits_heatmap_metric(tmp_path):
    """Regression: --arch v1 must still train (loss_self + w_danger * loss_danger)."""
    dataset_dir = _synthetic_dataset(tmp_path)
    out = tmp_path / "run-v1"
    summary = vision_train.train(
        _cfg(dataset_dir, out, arch="v1", stage="state", epochs_state=1, max_train_batches=1)
    )

    assert summary["arch"] == "v1"
    row = summary["stages"]["state"]["history"][0]
    assert "val_heatmap_hit_rate" not in row
    for key in STATE_METRIC_KEYS:
        assert key in row
    model, extra = load_vision_checkpoint(out / "state.pt", map_location="cpu")
    assert extra["arch"] == "v1"
    assert model.config.depth == 1


# --------------------------------------------------------------------------- #
# manifest gate and splits
# --------------------------------------------------------------------------- #
def test_manifest_gate_used_when_present(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    manifest = _write_manifest(dataset_dir)
    out = tmp_path / "run"
    summary = vision_train.train(_cfg(dataset_dir, out, allow_missing_manifest=False))

    assert summary["manifest_verified"] is True
    assert summary["val_frames"] == manifest.splits["val_frames"] == 4
    assert summary["val_index_sha256"] == manifest.splits["val_index_sha256"]
    assert summary["train_frames"] + summary["val_frames"] == summary["frames"] == 24


def test_manifest_gate_rejects_wrong_hash(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    _write_manifest(dataset_dir)
    path = dataset_dir / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["splits"]["val_index_sha256"] = "deadbeef"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError):
        vision_train.train(_cfg(dataset_dir, tmp_path / "run", allow_missing_manifest=False))


def test_manifest_gate_required_without_flag(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    with pytest.raises(ManifestError):
        vision_train.train(_cfg(dataset_dir, tmp_path / "run", allow_missing_manifest=False))


def test_train_fraction_requires_manifest(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    cfg = _cfg(dataset_dir, tmp_path / "run", allow_missing_manifest=True, train_fraction=0.5)
    with pytest.raises(SystemExit) as error:
        vision_train.train(cfg)
    assert "--train-fraction requires a verified manifest" in str(error.value)


def test_train_fraction_out_of_range_is_rejected(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    _write_manifest(dataset_dir)
    with pytest.raises(SystemExit) as error:
        vision_train.train(_cfg(dataset_dir, tmp_path / "run", train_fraction=1.5, allow_missing_manifest=False))
    assert "(0, 1]" in str(error.value)


def test_train_fraction_reduces_train_frames_only(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    _write_manifest(dataset_dir)
    full = vision_train.train(
        _cfg(dataset_dir, tmp_path / "full", train_fraction=1.0, allow_missing_manifest=False, max_train_batches=1)
    )
    half = vision_train.train(
        _cfg(dataset_dir, tmp_path / "half", train_fraction=0.5, allow_missing_manifest=False, max_train_batches=1)
    )

    assert half["train_frames"] < full["train_frames"]
    assert abs(half["train_frames"] - full["train_frames"] / 2) <= 2
    # the validation side must be byte-identical between the two runs
    assert half["val_frames"] == full["val_frames"] == 4
    assert half["val_index_sha256"] == full["val_index_sha256"]
    assert half["train_fraction"] == 0.5


# --------------------------------------------------------------------------- #
# batch limits, write protection, metric source
# --------------------------------------------------------------------------- #
def test_max_train_batches_does_not_shrink_val(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    unlimited_val = vision_train.train(
        _cfg(dataset_dir, tmp_path / "a", max_train_batches=1, max_val_batches=0)
    )
    limited_val = vision_train.train(
        _cfg(dataset_dir, tmp_path / "b", max_train_batches=1, max_val_batches=1)
    )

    # training was cut to one batch, validation was not: 4 frames = len(val_idx)
    assert unlimited_val["stages"]["state"]["history"][0]["val_frames"] == unlimited_val["val_frames"] == 4
    # and --max-val-batches still works on its own (1 batch x 2 frames)
    assert limited_val["stages"]["state"]["history"][0]["val_frames"] == 2
    assert limited_val["config"]["max_train_batches"] == 1
    assert limited_val["config"]["max_val_batches"] == 1


def test_refuses_to_overwrite_pilot_v1(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    frozen = vision_train.FROZEN_RUN
    for out in (frozen, frozen / "nested", Path("third_party/th08/build/rl/runs/pilot-v1")):
        with pytest.raises(SystemExit) as error:
            vision_train.train(_cfg(dataset_dir, out))
        assert "refusing to overwrite the v1.0.2 evidence baseline" in str(error.value)


def test_bullet_count_metric_uses_index_14(tmp_path, monkeypatch):
    """Q1: the summary's bullet error must come from vision_metrics, index 14."""
    dataset_dir = _synthetic_dataset(tmp_path, bullet_channel_fix=True)
    seen: dict = {}
    real = vision_metrics.bullet_count_mae

    def spy(pred, truth):
        predicted = np.asarray(pred, dtype=np.float64)
        actual = np.asarray(truth, dtype=np.float64)
        seen["truth13"] = float(actual[0, 13])
        seen["truth14"] = float(actual[0, 14])
        seen["index14"] = float(np.abs(predicted[:, 14] - actual[:, 14]).mean()) * 700.0
        seen["index13"] = float(np.abs(predicted[:, 13] - actual[:, 13]).mean()) * 700.0
        return real(pred, truth)

    monkeypatch.setattr(vision_metrics, "bullet_count_mae", spy)
    summary = vision_train.train(_cfg(dataset_dir, tmp_path / "run", epochs_state=1, max_train_batches=1))

    row = summary["stages"]["state"]["history"][0]
    # the fixture makes the two channels tell different stories
    assert (seen["truth13"], seen["truth14"]) == (1.0, 0.0)
    # the reported value is the index-14 formula, computed independently here
    assert row["val_bullet_count_mae"] == pytest.approx(seen["index14"], abs=0.1)
    # ... and the removed in-file formula (index 13) would have reported ~700 instead
    assert abs(seen["index13"] - seen["index14"]) > 1.0


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #
def test_cli_defaults_and_legacy_alias(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_train(cfg):
        captured["cfg"] = cfg
        return {"stages": {}, "schema": vision_train.SCHEMA}

    monkeypatch.setattr(vision_train, "train", fake_train)
    argv = ["--dataset", str(tmp_path), "--out", str(tmp_path / "run")]
    assert vision_train.main(argv) == 0
    cfg = captured["cfg"]
    assert (cfg.arch, cfg.stage) == ("v2", "both")
    assert (cfg.epochs_state, cfg.epochs_policy) == (8, 8)
    assert (cfg.batch_size, cfg.depth, cfg.lr) == (96, 1, pytest.approx(3e-4))
    assert (cfg.device, cfg.seed, cfg.train_fraction, cfg.val_fraction) == ("cuda", 7, 1.0, 0.1)
    assert (cfg.max_train_batches, cfg.max_val_batches) == (0, 0)
    assert (cfg.heatmap_sigma, cfg.w_heatmap, cfg.w_self, cfg.w_danger) == (1.5, 1.0, 1.0, 1.0)
    assert (cfg.pf_width, cfg.hud_width) == (32, 16)
    assert cfg.allow_missing_manifest is False

    # the deprecated name still works, and the new name wins when both are given
    assert vision_train.main(argv + ["--max-batches-per-epoch", "7"]) == 0
    assert captured["cfg"].max_train_batches == 7
    assert captured["cfg"].max_val_batches == 0
    assert vision_train.main(argv + ["--max-train-batches", "3", "--max-batches-per-epoch", "7"]) == 0
    assert captured["cfg"].max_train_batches == 3


def test_cli_requires_out_and_dataset():
    for argv in ([], ["--dataset", "x"], ["--out", "y"]):
        with pytest.raises(SystemExit):
            vision_train.main(argv)


def test_config_records_recipe_knobs(tmp_path):
    """The six new knobs must land in summary["config"] so a run is reproducible."""
    dataset_dir = _synthetic_dataset(tmp_path)
    summary = vision_train.train(
        _cfg(
            dataset_dir,
            tmp_path / "run",
            epochs_state=1,
            max_train_batches=1,
            heatmap_sigma=2.0,
            w_heatmap=0.5,
            w_self=2.0,
            w_danger=3.0,
            pf_width=8,
            hud_width=4,
        )
    )

    config = summary["config"]
    assert (config["heatmap_sigma"], config["w_heatmap"], config["w_self"], config["w_danger"]) == (
        2.0,
        0.5,
        2.0,
        3.0,
    )
    assert (config["pf_width"], config["hud_width"]) == (8, 4)


def test_deprecated_config_limit_still_works(tmp_path, monkeypatch):
    """`pipeline train` still passes max_batches_per_epoch; it must bound training."""
    dataset_dir = _synthetic_dataset(tmp_path)
    calls: list[int] = []
    real_gather = vision_train._gather

    def counting_gather(*args, **kwargs):
        calls.append(len(calls))
        return real_gather(*args, **kwargs)

    monkeypatch.setattr(vision_train, "_gather", counting_gather)
    summary = vision_train.train(
        _cfg(
            dataset_dir,
            tmp_path / "run",
            max_train_batches=0,
            max_batches_per_epoch=1,
            max_val_batches=0,
        )
    )

    # 1 train batch + 2 val batches (4 validation frames at batch 2), not 10 + 2
    assert len(calls) == 3
    # and the summary records the effective limit, so the run is still a "screening" run
    assert summary["config"]["max_train_batches"] == 1

def test_coordinate_loss_matches_the_pixel_metric():
    """0.01 of the field in x plus 0.02 in y is 6.4 px, scaled by 1/100."""
    import torch
    from touhou_rl.vision_train import coordinate_loss

    pred = torch.tensor([[0.5, 0.5]])
    truth = torch.tensor([[0.51, 0.52]])
    assert abs(float(coordinate_loss(pred, truth)) - 0.064) < 1e-6


def test_coordinate_loss_is_zero_for_a_perfect_prediction():
    import torch
    from touhou_rl.vision_train import coordinate_loss

    position = torch.tensor([[0.3, 0.7], [0.9, 0.1]])
    assert float(coordinate_loss(position, position)) == 0.0


def test_w_coord_is_recorded_in_the_summary_config():
    """The recipe must be reproducible from the summary alone."""
    from touhou_rl.vision_train import VisionTrainConfig

    cfg = VisionTrainConfig(dataset=__import__("pathlib").Path("."), out_dir=__import__("pathlib").Path("."))
    assert cfg.w_coord == 1.0
    assert cfg.w_heatmap == 1.0


# --------------------------------------------------------------------------- #
# history de-duplication inside _gather (throughput, must be value-identical)
# --------------------------------------------------------------------------- #
def _naive_gather(dataset, indices, depth, region_sizes):
    """The straightforward implementation: decode every stack slot as asked.

    This is the reference the de-duplicating ``_gather`` must match *exactly* -
    same rows, same order, same dtype.  It is the v1.0.3 implementation.
    """
    stack_ids = dataset.stack_indices(indices, depth).reshape(-1).tolist()
    regions = dataset.region_frames(stack_ids, region_sizes)
    stacked = {
        name: frames.reshape(
            len(indices), depth, region_sizes[name][0], region_sizes[name][1], vision.VISION_CHANNELS
        )
        for name, frames in regions.items()
    }
    center = dataset.labels(indices)
    labels = {
        "self_feat": center["self_feat"].astype(np.float32),
        "danger": center["danger"].astype(np.float32),
        "action": center["action"].astype(np.int64),
        "bullets": center["bullets"].astype(np.float32),
    }
    return stacked, labels


def _assert_gather_is_value_identical(dataset_dir: Path, indices: list[int], depth: int) -> int:
    """Pin value/shape/dtype equality and return how many slots were duplicates."""
    import torch

    from touhou_rl.vision_data import VisionDataset

    region_sizes = dict(REGION_HW)
    dataset = VisionDataset(dataset_dir, history_contract="causal-segment-v1")
    try:
        fast, fast_labels = vision_train._gather(dataset, indices, depth, region_sizes)
        slow, slow_labels = _naive_gather(dataset, indices, depth, region_sizes)
        for name, (height, width) in region_sizes.items():
            assert fast[name].dtype == torch.uint8
            assert fast[name].shape == (len(indices), depth, height, width, vision.VISION_CHANNELS)
            assert np.array_equal(fast[name].numpy(), slow[name]), name
        # sample set / order / labels are untouched by the de-duplication
        assert fast_labels["action"].dtype == np.int64
        for key in ("self_feat", "danger", "action", "bullets"):
            assert np.array_equal(fast_labels[key], slow_labels[key]), key
        stack_ids = dataset.stack_indices(indices, depth).reshape(-1)
        return int(stack_ids.size - np.unique(stack_ids).size)
    finally:
        dataset.close()


def test_gather_dedups_segment_start_repeats_value_identically(tmp_path):
    """A depth-3 causal segment repeats its first frame; the gather must not."""
    dataset_dir = _synthetic_dataset(tmp_path)
    # index 0 starts a shard, and STAGES changes at 3/6/9/... so these samples are
    # segment starts whose three slots collapse onto one frame.
    duplicates = _assert_gather_is_value_identical(dataset_dir, [0, 3, 6, 9, 12, 15], depth=3)
    assert duplicates > 0, "the fixture must actually produce repeated history slots"


def test_gather_dedups_across_shards_value_identically(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    # shard_size=5: these indices span five shards.
    duplicates = _assert_gather_is_value_identical(dataset_dir, [0, 4, 5, 9, 10, 14, 15, 19, 20, 23], depth=3)
    assert duplicates > 0


def test_gather_depth_one_is_value_identical(tmp_path):
    dataset_dir = _synthetic_dataset(tmp_path)
    duplicates = _assert_gather_is_value_identical(dataset_dir, [0, 1, 2, 7, 8, 13, 21, 23], depth=1)
    assert duplicates == 0  # depth 1 has no repeated slot to remove

