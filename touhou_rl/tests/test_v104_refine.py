"""Tests for the S3-7 tool and the trainer extensions it drives (CPU, synthetic).

Covered here, in the order the task book lists them:

* **R2 weight table** - NPZ shape/dtype, exactly-A-training-index identity (no
  duplicate, no validation frame), finite/non-negative errors, the 10/30 px
  bucket boundaries, the separate JSON identity, and the "never infer the global
  index from the row order" rule.
* **R2 reduction** - ``sum(w * per_sample) / sum(w)`` on the heatmap/coordinate
  terms only; self and danger keep their plain means and validation is
  unweighted (constant weights cancel, non-uniform weights do not).
* **R3 danger head** - ``learnable-v104`` round trip, parameter delta, no output
  ReLU, unknown variant rejection, old payloads stay legacy, and a variant
  checkpoint cannot be loaded into a legacy model.
* **v3-st in the trainer** - ``--arch v3-st`` needs the v104 contract and depth 3,
  the policy path encodes once (one ``forward``), the checkpoint chain carries
  C6's five identity keys, and the E4 ablation keeps every parameter slot while
  feeding a zero local vector.
* **E4/E5/E6 single variable** and the **G-v3 gate/waiver** checks.
* **freeze** - 20 flat keys, sha/identity rules, refusal of an existing valid
  list, seed 7 only, ``external_guard_uses=0``, readiness labelling.
* **dry run** - the default writes nothing.

The real GPU runs (E4/E5/E6, R2/R3, the throughput probe) are *not* launched by
these tests: every subprocess is either a dry run or absent.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from touhou_rl import obs, vision, vision_train
from touhou_rl.pipeline.manifest import DatasetManifest, ShardRecord, file_digest, index_digest
from touhou_rl.vision_data import META_FIELDS, VisionShardWriter
from touhou_rl.vmodels import (
    load_vision_checkpoint,
    save_vision_checkpoint,
    validate_v3_payload,
    vision_arch,
)
from touhou_rl.vmodels2 import (
    DANGER_HEAD_LEARNABLE_V104,
    DANGER_HEAD_LEGACY,
    VisionNetConfigV2,
    VisionScorerV2,
)

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("refine_v104", ROOT / "tools" / "rl" / "refine_v104.py")
refine = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(refine)

REGION_HW = {"playfield": (56, 48), "hud_score": (17, 54), "hud_status": (31, 54)}
#: The model's input ROI (the archive is resized by ``regions_to_batch``); the R2
#: identity binds these, not the archive sizes.
MODEL_REGION_HW = {"playfield": (224, 192), "hud_score": (34, 216), "hud_status": (126, 216)}
STAGES = (0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2)
GATE_V3 = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-4" / "gate-v3.json"


# --------------------------------------------------------------------------- #
# synthetic data / checkpoints
# --------------------------------------------------------------------------- #
def _png(value: int, height: int, width: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = value % 256
    image[: max(1, height // 4), : max(1, width // 4), 1] = (value * 3) % 256
    return vision.encode_png(image)


def _labels(index: int) -> tuple[obs.Observation, dict]:
    rng = np.random.default_rng(1000 + index)
    self_feat = rng.random(obs.N_SELF, dtype=np.float32) * 0.5
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


def synthetic_dataset(root: Path, frames: int = 24, *, manifest: bool = True) -> Path:
    dataset_dir = Path(root) / "ds"
    writer = VisionShardWriter(dataset_dir, shard_size=5, tag="syn")
    for index in range(frames):
        observation, meta = _labels(index)
        meta["frame"] = index
        meta["stage"] = STAGES[index % len(STAGES)]
        regions = {
            name: _png(index * 9 + offset, *hw) for offset, (name, hw) in enumerate(REGION_HW.items())
        }
        writer.add(regions, observation, (index % 9, 1, 0, 0), meta)
    writer.close()
    if manifest:
        write_manifest(dataset_dir)
    return dataset_dir


def write_manifest(dataset_dir: Path, *, seed: int = 7, val_fraction: float = 0.2) -> DatasetManifest:
    records = []
    for path in sorted(Path(dataset_dir).glob("*.npz")):
        with np.load(path) as data:
            frames = int(data["action"].shape[0])
        records.append(
            ShardRecord(name=path.name, frames=frames, bytes=path.stat().st_size, sha256=file_digest(path))
        )
    manifest = DatasetManifest.build(
        modality="vision",
        root=Path(dataset_dir),
        shards=records,
        frame_shape=None,
        has_teacher_values=False,
        teacher={},
        seed=seed,
        val_fraction=val_fraction,
        stats={},
        provenance={},
    )
    manifest.save(Path(dataset_dir) / "manifest.json")
    return manifest


def split_of(dataset_dir: Path) -> tuple[list[int], list[int], str]:
    manifest = DatasetManifest.load(Path(dataset_dir) / "manifest.json")
    train, val = manifest.verify(Path(dataset_dir))
    return train, val, str(manifest.splits["val_index_sha256"])


def cfg(dataset_dir: Path, out_dir: Path, **overrides) -> vision_train.VisionTrainConfig:
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


# --------------------------------------------------------------------------- #
# recipe fixtures
# --------------------------------------------------------------------------- #
def train_config_block(**overrides) -> dict:
    block = {
        "arch": "v2",
        "stage": "both",
        "depth": 3,
        "seed": 7,
        "batch_size": 96,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs_state": 72,
        "epochs_policy": 12,
        "max_train_batches": 0,
        "max_val_batches": 0,
        "selection_contract": "v104",
        "history_contract": "causal-segment-v1",
        "policy_objective": "rank",
        "rank_delta_px": 0.5,
        "rank_weight": 1.0,
        "regression_weight": 1.0,
        "w_heatmap": 1.0,
        "w_coord": 1.0,
        "w_danger": 1.0,
        "w_self": 1.0,
        "policy_aux_heatmap": 0.1,
        "policy_aux_coord": 1.0,
        "policy_aux_danger": 0.2,
        "policy_aux_self": 0.1,
        "policy_freeze_epochs": 2,
        "encoder_lr": 1e-4,
        "policy_head_lr": 3e-4,
        "heatmap_sigma": 1.0,
        "local_enabled": True,
        "danger_head_variant": "legacy",
    }
    block.update(overrides)
    return block


def v3_config_block(**overrides) -> dict:
    block = train_config_block(
        arch="v3-st",
        depth=3,
        heatmap_sigma=2.0,
        w_heatmap=1.0,
        w_coord=1.0,
        w_danger=1.0,
        w_self=1.0,
    )
    block.update(overrides)
    return block


def write_recipe(path: Path, *, experiment: str, train_config: dict, route: str = "v3-st", **extra) -> Path:
    payload = {
        "schema": refine.RECIPE_SCHEMA,
        "phase": "v3-matrix",
        "experiment": experiment,
        "route": route,
        "dataset": refine.rel(refine.DATASET_A),
        "run_dir": f"third_party/th08/build/rl/runs/v104-{experiment}-s7",
        "train_config": train_config,
        "notes": "unit-test recipe",
    }
    payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def v3_recipe_trio(tmp_path: Path) -> list[Path]:
    paths = []
    for name in ("E4", "E5", "E6"):
        config = v3_config_block(
            local_enabled=refine.V3_MATRIX[name]["local_enabled"],
        )
        if refine.V3_MATRIX[name]["policy_aux_zero"]:
            for key in refine.POLICY_AUX:
                config[key] = 0.0
        paths.append(
            write_recipe(
                tmp_path / f"{name}.json",
                experiment=name,
                train_config=config,
                gate={
                    "path": refine.rel(GATE_V3),
                    "waiver": {
                        "by": "Q6",
                        "items": ["throughput_le_60s"],
                        "source": "Document/Update/Alpha/v1.0.4/问题清单.md §8 Q6",
                    },
                },
            )
        )
    return paths


# --------------------------------------------------------------------------- #
# R2: bucket values and the weight table
# --------------------------------------------------------------------------- #
def test_position_weight_buckets_and_boundaries():
    errors = np.array([0.0, 9.999, 10.0, 20.0, 30.0, 30.001, 120.0], dtype=np.float32)
    weights = vision_train.position_weight_values(errors)
    assert weights.dtype == np.float32
    assert list(weights) == [1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0]


def _weight_file(
    directory: Path,
    *,
    train_indices,
    val_indices,
    errors=None,
    weights=None,
    frames=None,
    identity_overrides=None,
    history_contract="legacy",
    roi=None,
    source_sha=None,
    drop_identity=False,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    indices = np.asarray(train_indices, dtype=np.int64)
    errs = np.asarray(
        errors if errors is not None else np.linspace(1.0, 40.0, len(indices)), dtype=np.float32
    )
    ws = np.asarray(weights if weights is not None else vision_train.position_weight_values(errs), dtype=np.float32)
    npz = directory / "weights.npz"
    np.savez(npz, train_indices=indices, errors=errs, weights=ws)
    identity = {
        "schema": vision_train.POSITION_WEIGHTS_SCHEMA,
        "route": "v2",
        "frames": int(frames if frames is not None else len(indices)),
        "train_indices_sha256": index_digest(sorted(int(index) for index in indices)),
        "roi": roi or {name: [hw[0], hw[1]] for name, hw in MODEL_REGION_HW.items()},
        "history_contract": history_contract,
        "thresholds": list(vision_train.POSITION_WEIGHT_THRESHOLDS),
        "weight_values": list(vision_train.POSITION_WEIGHT_VALUES),
        "frame_counts": {
            str(value): int(np.count_nonzero(ws == value)) for value in vision_train.POSITION_WEIGHT_VALUES
        },
        "source_state_sha256": source_sha,
    }
    identity.update(identity_overrides or {})
    if not drop_identity:
        (directory / "weights.json").write_text(json.dumps(identity), encoding="utf-8")
    return npz


def _load_weights(npz, train_indices, val_indices, *, frames=None, history_contract="legacy", source_sha=None):
    return vision_train.load_position_sample_weights(
        npz,
        train_indices=list(train_indices),
        val_indices=list(val_indices),
        dataset_frames=int(frames if frames is not None else (len(train_indices) + len(val_indices))),
        region_sizes={name: tuple(hw) for name, hw in MODEL_REGION_HW.items()},
        history_contract=history_contract,
        source_state_sha256=source_sha,
    )


def test_weights_round_trip_maps_by_index_not_by_row_order(tmp_path):
    train = list(range(0, 20))
    val = [20, 21, 22, 23]
    errors = np.linspace(1.0, 40.0, 20).astype(np.float32)
    # Shuffled storage order: row i is *not* train index i.  The loader must use
    # the stored train_indices values, never the row position.
    order = np.array([7, 0, 19, 3, 4, 5, 6, 1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    npz = _weight_file(
        tmp_path,
        train_indices=order,
        val_indices=val,
        errors=errors[order],
    )
    loaded = _load_weights(npz, train, val, frames=24)
    expected = np.ones(24, dtype=np.float32)
    expected[order] = vision_train.position_weight_values(errors[order])
    assert np.array_equal(loaded["weights"], expected)
    assert loaded["train_frames"] == 20
    assert loaded["weights"][val].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert sum(loaded["frame_counts"].values()) == 20


def test_weights_reject_a_validation_index(tmp_path):
    train = list(range(0, 20))
    val = [20, 21, 22, 23]
    npz = _weight_file(tmp_path, train_indices=train[:-1] + [20], val_indices=val)
    with pytest.raises(ValueError, match="v104:DATA: train_indices are not exactly"):
        _load_weights(npz, train, val, frames=24)


def test_weights_reject_a_different_index_set_of_the_same_size(tmp_path):
    train = list(range(0, 20))
    val = [20, 21, 22, 23]
    npz = _weight_file(tmp_path, train_indices=list(range(1, 21)), val_indices=val)
    with pytest.raises(ValueError, match="not exactly A's training split"):
        _load_weights(npz, train, val, frames=24)


def test_weights_reject_duplicates(tmp_path):
    train = list(range(0, 20))
    val = [20, 21, 22, 23]
    npz = _weight_file(tmp_path, train_indices=train[:-1] + [0], val_indices=val)
    with pytest.raises(ValueError, match="duplicates"):
        _load_weights(npz, train, val, frames=24)


def test_weights_reject_non_finite_and_negative_errors(tmp_path):
    train = list(range(0, 20))
    val = [20, 21, 22, 23]
    errors = np.linspace(1.0, 40.0, 20).astype(np.float32)
    errors[3] = np.nan
    npz = _weight_file(tmp_path / "nan", train_indices=train, val_indices=val, errors=errors)
    with pytest.raises(ValueError, match="non-finite"):
        _load_weights(npz, train, val, frames=24)
    errors[3] = -1.0
    npz = _weight_file(tmp_path / "neg", train_indices=train, val_indices=val, errors=errors)
    with pytest.raises(ValueError, match="negative"):
        _load_weights(npz, train, val, frames=24)


def test_weights_reject_a_bucket_mismatch(tmp_path):
    train = list(range(0, 20))
    val = [20, 21, 22, 23]
    errors = np.linspace(1.0, 40.0, 20).astype(np.float32)
    weights = vision_train.position_weight_values(errors)
    weights[0] = 2.0  # e=1 px is an "easy" frame
    npz = _weight_file(
        tmp_path, train_indices=train, val_indices=val, errors=errors, weights=weights
    )
    with pytest.raises(ValueError, match="do not follow the R2 buckets"):
        _load_weights(npz, train, val, frames=24)


def test_weights_reject_wrong_dtypes(tmp_path):
    train = list(range(0, 20))
    val = [20, 21, 22, 23]
    directory = tmp_path / "dtype"
    directory.mkdir()
    npz = directory / "weights.npz"
    np.savez(
        npz,
        train_indices=np.asarray(train, dtype=np.int32),
        errors=np.zeros(20, dtype=np.float32),
        weights=np.ones(20, dtype=np.float32),
    )
    (directory / "weights.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="train_indices dtype"):
        _load_weights(npz, train, val, frames=24)


def test_weights_require_the_json_identity(tmp_path):
    train = list(range(0, 20))
    val = [20, 21, 22, 23]
    npz = _weight_file(tmp_path, train_indices=train, val_indices=val, drop_identity=True)
    with pytest.raises(ValueError, match="a weight table without its identity"):
        _load_weights(npz, train, val, frames=24)


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"train_indices_sha256": "0" * 64}, "train_indices_sha256"),
        ({"frames": 19}, "records 19 frames"),
        ({"thresholds": [5.0, 30.0]}, "thresholds"),
        ({"weight_values": [1.0, 2.0, 3.0]}, "weight_values"),
        ({"history_contract": "causal-segment-v1"}, "history_contract"),
        ({"roi": {"playfield": [1, 1]}}, "roi"),
        ({"frame_counts": {"1.0": 99}}, "frame_counts"),
        ({"schema": "wrong/1"}, "identity schema"),
    ],
)
def test_weights_reject_identity_mismatches(tmp_path, overrides, fragment):
    train = list(range(0, 20))
    val = [20, 21, 22, 23]
    npz = _weight_file(
        tmp_path, train_indices=train, val_indices=val, identity_overrides=overrides
    )
    with pytest.raises(ValueError) as error:
        _load_weights(npz, train, val, frames=24)
    assert "v104:DATA" in str(error.value)
    assert fragment in str(error.value)


def test_weights_bind_the_error_source_checkpoint(tmp_path):
    train = list(range(0, 20))
    val = [20, 21, 22, 23]
    npz = _weight_file(
        tmp_path, train_indices=train, val_indices=val, source_sha="a" * 64
    )
    with pytest.raises(ValueError, match="v104:CHECKPOINT: weight table was computed from"):
        _load_weights(npz, train, val, frames=24, source_sha="b" * 64)


# --------------------------------------------------------------------------- #
# R2: the reduction itself
# --------------------------------------------------------------------------- #
def test_localisation_losses_weights_only_heatmap_and_coord():
    heatmap = torch.tensor([1.0, 2.0, 3.0])
    coord = torch.tensor([0.1, 0.2, 0.3])
    self_loss = torch.tensor(5.0)
    danger_loss = torch.tensor(7.0)
    uniform, parts = vision_train.localisation_losses(
        heatmap_per_sample=heatmap,
        coord_per_sample=coord,
        self_loss=self_loss,
        danger_loss=danger_loss,
        weights=None,
        w_heatmap=1.0,
        w_coord=1.0,
        w_self=1.0,
        w_danger=1.0,
    )
    assert float(parts["heatmap"]) == pytest.approx(2.0)
    assert float(parts["coord"]) == pytest.approx(0.2)
    assert float(parts["self"]) == pytest.approx(5.0)
    assert float(parts["danger"]) == pytest.approx(7.0)
    assert float(uniform) == pytest.approx(2.0 + 0.2 + 5.0 + 7.0)

    constant, constant_parts = vision_train.localisation_losses(
        heatmap_per_sample=heatmap,
        coord_per_sample=coord,
        self_loss=self_loss,
        danger_loss=danger_loss,
        weights=torch.full((3,), 2.0),
        w_heatmap=1.0,
        w_coord=1.0,
        w_self=1.0,
        w_danger=1.0,
    )
    # sum(w*v)/sum(w) == mean(v): a constant weight must cancel exactly.
    assert float(constant_parts["heatmap"]) == pytest.approx(2.0)
    assert float(constant) == pytest.approx(float(uniform))

    weighted, weighted_parts = vision_train.localisation_losses(
        heatmap_per_sample=heatmap,
        coord_per_sample=coord,
        self_loss=self_loss,
        danger_loss=danger_loss,
        weights=torch.tensor([1.0, 1.5, 2.0]),
        w_heatmap=1.0,
        w_coord=1.0,
        w_self=1.0,
        w_danger=1.0,
    )
    expected_heatmap = (1.0 * 1.0 + 1.5 * 2.0 + 2.0 * 3.0) / 4.5
    expected_coord = (1.0 * 0.1 + 1.5 * 0.2 + 2.0 * 0.3) / 4.5
    assert float(weighted_parts["heatmap"]) == pytest.approx(expected_heatmap)
    assert float(weighted_parts["coord"]) == pytest.approx(expected_coord)
    assert float(weighted_parts["self"]) == pytest.approx(5.0)
    assert float(weighted_parts["danger"]) == pytest.approx(7.0)


def test_weighted_mean_rejects_an_empty_weight_batch():
    with pytest.raises(ValueError, match="sum to zero"):
        vision_train.weighted_mean(torch.tensor([1.0, 2.0]), torch.zeros(2))


def test_train_records_the_weight_identity(tmp_path):
    dataset_dir = synthetic_dataset(tmp_path)
    train, val, _sha = split_of(dataset_dir)
    npz = _weight_file(
        tmp_path / "weights",
        train_indices=train,
        val_indices=val,
        history_contract="causal-segment-v1",
    )
    out = tmp_path / "run"
    out = tmp_path / "run"
    summary = vision_train.train(
        cfg(
            dataset_dir,
            out,
            arch="v2",
            stage="state",
            epochs_state=1,
            selection_contract="v104",
            history_contract="causal-segment-v1",
            policy_objective="legacy-reg",
            depth=1,
            max_val_batches=0,
            allow_missing_manifest=False,
            position_sample_weights=npz,
        )
    )
    block = summary["position_weights"]
    assert block["path"].endswith("weights.npz")
    assert block["sha256"] == file_digest(npz)
    assert block["train_frames"] == len(train)
    assert sum(block["frame_counts"].values()) == len(train)
    assert "only" in block["reduction"]
    assert summary["config"]["position_sample_weights"].endswith("weights.npz")


def test_train_passes_the_batch_weights_into_the_localisation_terms(tmp_path, monkeypatch):
    """The weighted reduction must be *entered* with this batch's weights.

    A fresh model gives nearly identical per-sample losses, so the numeric loss of
    a weighted and an unweighted run at epoch 1 look the same to 5 decimals - that
    is not a usable signal.  The seam that decides which reduction runs receives
    ``weights``; this test pins that the trainer hands it the per-frame table
    (and ``None`` when no table is configured).
    """
    dataset_dir = synthetic_dataset(tmp_path)
    train, val, _sha = split_of(dataset_dir)
    errors = np.zeros(len(train), dtype=np.float32)
    # Alternate easy/hard *by index* so that adjacent indices (which is how the
    # shard-chunked batches pair frames) always carry different weights.
    errors[np.asarray(train) % 2 == 0] = 5.0
    errors[np.asarray(train) % 2 == 1] = 40.0
    npz = _weight_file(
        tmp_path / "weights",
        train_indices=train,
        val_indices=val,
        errors=errors,
        history_contract="causal-segment-v1",
    )
    original = vision_train.localisation_losses
    calls = []

    def spy(**kwargs):
        weights = kwargs["weights"]
        calls.append(None if weights is None else [float(value) for value in weights])
        return original(**kwargs)

    monkeypatch.setattr(vision_train, "localisation_losses", spy)
    common = dict(
        arch="v2",
        stage="state",
        epochs_state=1,
        selection_contract="v104",
        history_contract="causal-segment-v1",
        policy_objective="legacy-reg",
        depth=1,
        max_val_batches=0,
        allow_missing_manifest=False,
    )
    vision_train.train(cfg(dataset_dir, tmp_path / "plain", **common))
    plain_calls = list(calls)
    calls.clear()
    vision_train.train(
        cfg(dataset_dir, tmp_path / "weighted", position_sample_weights=npz, **common)
    )
    weighted_calls = list(calls)
    assert plain_calls and all(entry is None for entry in plain_calls)
    assert weighted_calls and all(entry is not None for entry in weighted_calls)
    expected = {float(value) for value in vision_train.position_weight_values(errors)}
    for entry in weighted_calls:
        assert set(entry) <= expected
    # The epoch is train batches first, then the validation pass.  Training mixes
    # easy/hard frames, validation must stay at 1.0 (its indices are not in the
    # table at all), so the last batches are flat ones.
    val_batches = math.ceil(len(val) / 2)  # batch_size=2, validation keeps the tail
    assert any(len(set(entry)) > 1 for entry in weighted_calls[: len(weighted_calls) - val_batches])
    for entry in weighted_calls[len(weighted_calls) - val_batches :]:
        assert set(entry) == {1.0}, f"validation must not be weighted: {entry}"


# --------------------------------------------------------------------------- #
# R3: the learnable danger head
# --------------------------------------------------------------------------- #
def test_learnable_danger_head_round_trip_and_param_delta(tmp_path):
    legacy = VisionScorerV2(VisionNetConfigV2(depth=3))
    variant = VisionScorerV2(
        VisionNetConfigV2(depth=3, danger_head_variant=DANGER_HEAD_LEARNABLE_V104)
    )
    assert variant.num_parameters() - legacy.num_parameters() == 1632
    legacy_keys = set(legacy.state_dict())
    variant_keys = set(variant.state_dict())
    assert "playfield.danger_head.weight" in legacy_keys
    assert "playfield.danger_head.weight" not in variant_keys
    assert {
        "playfield.danger_reduce.weight",
        "playfield.danger_depthwise.weight",
        "playfield.danger_out.weight",
    } <= variant_keys

    path = tmp_path / "variant.pt"
    save_vision_checkpoint(path, variant, {"stage": "state"})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["config"]["danger_head_variant"] == DANGER_HEAD_LEARNABLE_V104
    restored, extra = load_vision_checkpoint(path)
    assert extra["stage"] == "state"
    assert restored.config.danger_head_variant == DANGER_HEAD_LEARNABLE_V104
    for name, tensor in variant.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name


def test_learnable_danger_head_has_no_output_relu():
    model = VisionScorerV2(VisionNetConfigV2(depth=1, danger_head_variant=DANGER_HEAD_LEARNABLE_V104))
    with torch.no_grad():
        # Zero the final conv so the only path to the output is its bias: with a
        # trailing ReLU the result would be 0, without one it stays -4.
        model.playfield.danger_out.weight.zero_()
        model.playfield.danger_out.bias.fill_(-4.0)
    regions = vision.regions_to_batch(
        {name: np.zeros((1, 1, hw[0], hw[1], 3), dtype=np.uint8) for name, hw in REGION_HW.items()}
    )
    with torch.no_grad():
        danger = model.decode_state(regions)["danger_log"]
    assert float(danger.max()) == pytest.approx(-4.0, abs=1e-6)
    assert float(danger.min()) == pytest.approx(-4.0, abs=1e-6)


@pytest.mark.parametrize("variant", ["legacy", "learnable-v104"])
def test_old_payloads_stay_legacy_and_variants_are_not_interchangeable(tmp_path, variant):
    source = VisionNetConfigV2(depth=1)
    payload_config = dict(source.__dict__)
    payload_config.pop("danger_head_variant", None)  # a v1.0.3 payload has no such key
    payload_config["depth"] = 1
    model = VisionScorerV2(VisionNetConfigV2(**payload_config))
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "kind": "vision_scorer",
            "arch": "v2",
            "state_dict": model.state_dict(),
            "config": payload_config,
            "frame": {"H": vision.VISION_H, "W": vision.VISION_W, "C": 3, "depth": 1},
            "extra": {"stage": "state"},
        },
        path,
    )
    restored, _extra = load_vision_checkpoint(path)
    assert restored.config.danger_head_variant == DANGER_HEAD_LEGACY
    if variant == DANGER_HEAD_LEARNABLE_V104:
        other = VisionScorerV2(VisionNetConfigV2(depth=1, danger_head_variant=variant))
        with pytest.raises(RuntimeError):
            other.load_state_dict(restored.state_dict(), strict=True)


def test_unknown_danger_head_variant_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="v104:CONFIG: danger_head_variant"):
        VisionNetConfigV2(danger_head_variant="v9")
    with pytest.raises(ValueError, match="danger_head_variant"):
        vision_train.validate_config(
            cfg(tmp_path / "ds", tmp_path / "run", arch="v2", danger_head_variant="v9")
        )


def test_danger_head_variant_cli_and_route_rules(tmp_path):
    dataset_dir = synthetic_dataset(tmp_path)
    out = tmp_path / "run"
    # The v2 variant is a model structure, not a contract switch: it is legal in
    # both selection contracts, and only a non-v2 architecture may not carry it.
    vision_train.validate_config(
        cfg(dataset_dir, out, arch="v2", danger_head_variant="learnable-v104")
    )
    vision_train.validate_config(
        cfg(
            dataset_dir,
            out,
            arch="v2",
            stage="state",
            danger_head_variant="learnable-v104",
            selection_contract="v104",
            history_contract="causal-segment-v1",
            depth=3,
        )
    )
    with pytest.raises(ValueError, match="v2 R3 variant"):
        vision_train.validate_config(
            cfg(
                dataset_dir,
                out,
                arch="v1",
                danger_head_variant="learnable-v104",
            )
        )


def test_train_writes_a_learnable_danger_head_checkpoint(tmp_path):
    dataset_dir = synthetic_dataset(tmp_path)
    out = tmp_path / "run"
    summary = vision_train.train(
        cfg(
            dataset_dir,
            out,
            arch="v2",
            stage="state",
            epochs_state=1,
            selection_contract="v104",
            history_contract="causal-segment-v1",
            policy_objective="legacy-reg",
            depth=1,
            allow_missing_manifest=False,
            max_val_batches=0,
            danger_head_variant=DANGER_HEAD_LEARNABLE_V104,
        )
    )
    assert summary["input"]["danger_head_variant"] == DANGER_HEAD_LEARNABLE_V104
    payload = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
    assert payload["config"]["danger_head_variant"] == DANGER_HEAD_LEARNABLE_V104
    assert "playfield.danger_reduce.weight" in payload["state_dict"]


# --------------------------------------------------------------------------- #
# v3-st in the trainer
# --------------------------------------------------------------------------- #
def test_v3_requires_the_v104_contract_and_depth_three(tmp_path):
    dataset_dir = synthetic_dataset(tmp_path)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="only reachable through selection_contract='v104'"):
        vision_train.validate_config(cfg(dataset_dir, out, arch="v3-st", depth=3))
    with pytest.raises(ValueError, match="depth 3 only"):
        vision_train.validate_config(
            cfg(
                dataset_dir,
                out,
                arch="v3-st",
                depth=1,
                selection_contract="v104",
                history_contract="causal-segment-v1",
            )
        )
    with pytest.raises(ValueError, match="local_enabled=False is the v3-st E4 ablation"):
        vision_train.validate_config(
            cfg(dataset_dir, out, arch="v2", local_enabled=False)
        )
    with pytest.raises(ValueError, match="arch='v3-st'.*keeps its own danger head"):
        vision_train.validate_config(
            cfg(
                dataset_dir,
                out,
                arch="v3-st",
                depth=3,
                selection_contract="v104",
                history_contract="causal-segment-v1",
                danger_head_variant="learnable-v104",
            )
        )


def test_v3_refuses_a_run_without_a_manifest(tmp_path):
    dataset_dir = synthetic_dataset(tmp_path, manifest=False)
    with pytest.raises(ValueError, match="v104:CHECKPOINT: arch='v3-st' writes vision-checkpoint/3"):
        vision_train.train(
            cfg(
                dataset_dir,
                tmp_path / "run",
                arch="v3-st",
                depth=3,
                stage="state",
                epochs_state=1,
                selection_contract="v104",
                history_contract="causal-segment-v1",
                policy_objective="legacy-reg",
                allow_missing_manifest=True,
            )
        )


def test_v3_policy_forward_encodes_once():
    from touhou_rl.vmodels3 import VisionSpatialTemporalConfig, VisionSpatialTemporalPolicy

    model = VisionSpatialTemporalPolicy(
        VisionSpatialTemporalConfig(pf_width=8, hud_width=4, trunk_hidden=32)
    ).eval()
    calls = {"count": 0}
    original = model.forward_core

    def counting(regions):
        calls["count"] += 1
        return original(regions)

    model.forward_core = counting
    regions = vision.regions_to_batch(
        {
            "playfield": np.zeros((2, 3, 224, 192, 3), dtype=np.uint8),
            "hud_score": np.zeros((2, 3, 34, 216, 3), dtype=np.uint8),
            "hud_status": np.zeros((2, 3, 126, 216, 3), dtype=np.uint8),
        }
    )
    with torch.no_grad():
        scores, aux = vision_train.policy_forward(model, regions)
    assert calls["count"] == 1, "the v3 policy path must use one forward()"
    assert tuple(scores.shape) == (2, 18)
    assert tuple(aux["heatmap_logits"].shape) == (2, 56, 48)
    assert tuple(aux["danger_log"].shape) == (2, 128)
    assert tuple(aux["self_feat"].shape) == (2, 24)


def test_v3_local_ablation_keeps_every_slot_and_feeds_zeros():
    from touhou_rl.vmodels3 import VisionSpatialTemporalConfig, VisionSpatialTemporalPolicy

    on = VisionSpatialTemporalPolicy(
        VisionSpatialTemporalConfig(pf_width=8, hud_width=4, trunk_hidden=32, local_enabled=True)
    )
    off = VisionSpatialTemporalPolicy(
        VisionSpatialTemporalConfig(pf_width=8, hud_width=4, trunk_hidden=32, local_enabled=False)
    )
    assert on.num_parameters() == off.num_parameters()
    assert {name: tuple(t.shape) for name, t in on.state_dict().items()} == {
        name: tuple(t.shape) for name, t in off.state_dict().items()
    }
    f4 = torch.randn(2, 16, 4, 4, requires_grad=True)
    position = torch.rand(2, 2, requires_grad=True)
    vector = off.local_features(f4, position)
    assert tuple(vector.shape) == (2, 128)
    assert int(torch.count_nonzero(vector)) == 0


def test_v3_state_and_policy_checkpoints_carry_the_c6_identity(tmp_path, monkeypatch):
    dataset_dir = synthetic_dataset(tmp_path)
    out = tmp_path / "run"
    # 24 synthetic frames leave the 0-15 danger bucket empty, and the contract
    # then refuses to select a policy checkpoint at all.  The C6 identity of a
    # *policy-stage* v3 checkpoint is what this test is about, so the selection
    # key is replaced for this run only (the empty-bucket rule itself is pinned by
    # test_v104_train_contract.py).
    monkeypatch.setattr(
        vision_train,
        "policy_selection_key",
        lambda row: (
            float(row.get("val_regret_px") or 0.0),
            0.0,
            int(row.get("epoch") or 1),
        ),
    )
    summary = vision_train.train(
        cfg(
            dataset_dir,
            out,
            arch="v3-st",
            depth=3,
            stage="both",
            epochs_state=1,
            epochs_policy=1,
            selection_contract="v104",
            history_contract="causal-segment-v1",
            policy_objective="legacy-reg",
            heatmap_sigma=2.0,
            allow_missing_manifest=False,
            max_val_batches=0,
            policy_freeze_epochs=1,
        )
    )
    assert summary["arch"] == "v3-st"
    assert summary["depth"] == 3
    assert summary["input"]["heatmap_sigma"] == 2.0
    assert summary["contract_version"] == "v104-A/1"
    state_payload = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
    policy_payload = torch.load(out / "policy.pt", map_location="cpu", weights_only=False)
    for payload in (state_payload, policy_payload):
        assert payload["schema"] == "vision-checkpoint/3"
        assert payload["arch"] == "v3-st"
        validate_v3_payload(payload)
        extra = payload["extra"]
        assert extra["teacher_fingerprint"]
        assert len(extra["manifest_sha"]) == 64
        assert len(extra["val_index_sha256"]) == 64
        assert extra["loss_config"]
    assert state_payload["extra"]["source_state_sha"] == ""
    assert len(policy_payload["extra"]["source_state_sha"]) == 64
    # best.pt must be a loadable v3 checkpoint too (same identity rules).
    best_payload = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    validate_v3_payload(best_payload)
    restored, extra = load_vision_checkpoint(out / "policy.pt")
    assert vision_arch(restored) == "v3-st"
    assert extra["stage"] == "policy"


# --------------------------------------------------------------------------- #
# recipes / single variable / gate
# --------------------------------------------------------------------------- #
def test_recipe_requires_every_train_config_key(tmp_path):
    config = v3_config_block()
    config.pop("policy_aux_self")
    path = write_recipe(tmp_path / "E5.json", experiment="E5", train_config=config)
    with pytest.raises(ValueError, match="train_config is missing"):
        refine.validate_recipe(refine.load_recipe(path), path=path, route="v3-st", phase="v3-matrix")


def test_recipe_rejects_an_unknown_key(tmp_path):
    config = v3_config_block()
    config["w_unknown"] = 1.0
    path = write_recipe(tmp_path / "E5.json", experiment="E5", train_config=config)
    with pytest.raises(ValueError, match="cannot express"):
        refine.validate_recipe(refine.load_recipe(path), path=path, route="v3-st", phase="v3-matrix")


def test_recipe_rejects_a_short_validation_and_a_non_seven_seed(tmp_path):
    path = write_recipe(
        tmp_path / "E5.json", experiment="E5", train_config=v3_config_block(max_val_batches=1)
    )
    with pytest.raises(ValueError, match="max_val_batches=1"):
        refine.validate_recipe(refine.load_recipe(path), path=path, route="v3-st", phase="v3-matrix")
    path = write_recipe(
        tmp_path / "E5b.json", experiment="E5", train_config=v3_config_block(seed=11)
    )
    with pytest.raises(ValueError, match="pre-declares seed 7"):
        refine.validate_recipe(refine.load_recipe(path), path=path, route="v3-st", phase="v3-matrix")


def test_recipe_rejects_b_and_frozen_directories(tmp_path):
    path = write_recipe(tmp_path / "E5.json", experiment="E5", train_config=v3_config_block())
    recipe = refine.load_recipe(path)
    recipe["dataset"] = refine.rel(refine.DATASET_B)
    with pytest.raises(ValueError, match="dataset A only"):
        refine.validate_recipe(recipe, path=path, route="v3-st", phase="v3-matrix")
    recipe = refine.load_recipe(path)
    recipe["run_dir"] = "third_party/th08/build/rl/runs/v104-a01-72ep/sub"
    with pytest.raises(ValueError, match="frozen baseline path"):
        refine.validate_recipe(recipe, path=path, route="v3-st", phase="v3-matrix")


def test_recipe_command_comes_only_from_the_recipe(tmp_path):
    config = v3_config_block(policy_aux_danger=0.0)
    path = write_recipe(tmp_path / "E5.json", experiment="E5", train_config=config)
    plan = refine.validate_recipe(
        refine.load_recipe(path), path=path, route="v3-st", phase="v3-matrix"
    )
    command = plan["command"]
    assert command[1:3] == ["-m", "touhou_rl.vision_train"]
    assert "--max-val-batches" in command and command[command.index("--max-val-batches") + 1] == "0"
    assert command[command.index("--policy-aux-danger") + 1] == "0"
    assert command[command.index("--arch") + 1] == "v3-st"
    assert "--local-enabled" in command
    assert command[command.index("--heatmap-sigma") + 1] == "2"
    # No unexpressed recipe key can hide in the command.
    assert "--w-unknown" not in command


def test_recipe_digest_is_key_order_independent(tmp_path):
    a = {"schema": refine.RECIPE_SCHEMA, "x": {"a": 1, "b": 2}, "train_config": {}}
    b = {"train_config": {}, "x": {"b": 2, "a": 1}, "schema": refine.RECIPE_SCHEMA}
    assert refine.recipe_digest(a) == refine.recipe_digest(b)


def test_single_variable_passes_on_the_declared_trio(tmp_path):
    plans = [
        refine.validate_recipe(
            refine.load_recipe(path), path=path, route="v3-st", phase="v3-matrix"
        )
        for path in v3_recipe_trio(tmp_path)
    ]
    result = refine.check_single_variable(plans)
    assert result["single_variable_verified"] is True
    assert result["delta"]["E4"] == {"local_enabled": False}
    assert result["delta"]["E6"] == {"local_enabled": True, "policy_aux_all_zero": True}


@pytest.mark.parametrize(
    "experiment, overrides, fragment",
    [
        ("E5", {"local_enabled": False}, "local_enabled"),
        ("E4", {"local_enabled": True}, "local_enabled"),
        ("E6", {"policy_aux_heatmap": 0.1}, "policy_aux_heatmap"),
        ("E5", {"w_danger": 0.5}, "w_danger"),
        ("E4", {"policy_objective": "centered-reg", "regression_weight": 1.0}, "policy_objective"),
        ("E6", {"epochs_policy": 12, "policy_freeze_epochs": 3}, "policy_freeze_epochs"),
    ],
)
def test_single_variable_rejects_any_other_difference(tmp_path, experiment, overrides, fragment):
    paths = v3_recipe_trio(tmp_path)
    target = next(path for path in paths if f"{experiment}.json" in path.name)
    recipe = refine.load_recipe(target)
    recipe["train_config"].update(overrides)
    target.write_text(json.dumps(recipe), encoding="utf-8")
    plans = [
        refine.validate_recipe(refine.load_recipe(path), path=path, route="v3-st", phase="v3-matrix")
        for path in paths
    ]
    with pytest.raises(ValueError) as error:
        refine.check_single_variable(plans)
    assert "v104:CONFIG" in str(error.value)
    assert fragment in str(error.value)


def test_single_variable_requires_all_three_experiments(tmp_path):
    plans = [
        refine.validate_recipe(refine.load_recipe(path), path=path, route="v3-st", phase="v3-matrix")
        for path in v3_recipe_trio(tmp_path)[:2]
    ]
    with pytest.raises(ValueError, match="needs exactly the recipes"):
        refine.check_single_variable(plans)


def test_gate_requires_a_waiver_and_keeps_the_fail_status():
    if not GATE_V3.is_file():
        pytest.skip("the S3-4 gate verdict is not present")
    recipe = {
        "gate": {
            "path": refine.rel(GATE_V3),
            "waiver": {
                "by": "Q6",
                "items": ["throughput_le_60s"],
                "source": "问题清单 §8 Q6",
            },
        }
    }
    gate = refine.check_v3_gate(GATE_V3, recipe)
    assert gate["conditions"]["throughput_le_60s"] == "fail"
    assert gate["waived_by"] == "Q6"
    assert gate["failed"] == ["throughput_le_60s"]
    assert gate["needs_user_ruling"] is True


def test_gate_refuses_an_unwaived_failure_or_a_wrong_waiver():
    if not GATE_V3.is_file():
        pytest.skip("the S3-4 gate verdict is not present")
    with pytest.raises(ValueError, match="must name the ruling"):
        refine.check_v3_gate(GATE_V3, {"gate": {}})
    with pytest.raises(ValueError, match="must match the verdict exactly"):
        refine.check_v3_gate(
            GATE_V3,
            {"gate": {"waiver": {"by": "Q6", "items": ["sequence_labels_coverage"], "source": "x"}}},
        )


# --------------------------------------------------------------------------- #
# freeze
# --------------------------------------------------------------------------- #
def _freeze_fixture(tmp_path: Path, monkeypatch, *, seed: int = 7, position: float = 8.5,
                    danger: float = 0.86, bucket: float = 1.2) -> dict:
    dataset_dir = synthetic_dataset(tmp_path)
    train, val, val_sha = split_of(dataset_dir)
    monkeypatch.setattr(refine, "VAL_FRAMES", len(val))
    monkeypatch.setattr(refine, "VAL_INDEX_SHA256", val_sha)
    run = tmp_path / "run"
    run.mkdir()
    model = VisionScorerV2(VisionNetConfigV2(depth=1, pf_width=8, hud_width=4, trunk_hidden=32))
    state_path = run / "state.pt"
    policy_path = run / "policy.pt"
    save_vision_checkpoint(state_path, model, {"stage": "state", "epoch": 3})
    save_vision_checkpoint(policy_path, model, {"stage": "policy", "epoch": 4})
    teacher = tmp_path / "teacher.npz"
    np.savez(teacher, values=np.zeros((3, 18), dtype=np.float16))
    summary = {
        "schema": "vision-summary/2",
        "contract_version": "v104-A/1",
        "arch": "v2",
        "depth": 1,
        "recipe": {"seed": seed},
        "data": {"val_index_sha256": val_sha},
        "teacher": {"fingerprint": "a" * 64},
        "stages": {
            "state": {
                "best": {
                    "epoch": 3,
                    "val_player_pos_px": position,
                    "val_danger_corr_perframe_median": danger,
                    "selection_rule": "position",
                    "checkpoint": refine.rel(state_path),
                    "checkpoint_sha256": file_digest(state_path),
                }
            },
            "policy": {
                "best": {
                    "epoch": 4,
                    "val_regret_px_bucket_0_15": bucket,
                    "val_regret_px": 2.0,
                    "selection_rule": "bucket-0-15-regret",
                    "checkpoint": refine.rel(policy_path),
                    "checkpoint_sha256": file_digest(policy_path),
                }
            },
        },
    }
    summary_path = run / "vision-summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return {
        "dataset": dataset_dir,
        "summary": summary_path,
        "teacher": teacher,
        "state": state_path,
        "policy": policy_path,
        "val_sha": val_sha,
    }


def _freeze_args(tmp_path: Path, fixture: dict, **overrides):
    args = refine.build_parser().parse_args(
        [
            "--phase",
            "freeze",
            "--out",
            str(tmp_path / "artifacts"),
            "--state-summary",
            str(fixture["summary"]),
            "--policy-summary",
            str(fixture["summary"]),
            "--teacher-cache",
            str(fixture["teacher"]),
            "--manifest-a",
            str(fixture["dataset"] / "manifest.json"),
            "--manifest-b",
            str(fixture["dataset"] / "manifest.json"),
            "--experiment-table",
            str(fixture["dataset"] / "manifest.json"),
            "--selection-reason",
            "unit test candidate",
            "--execute",
        ]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_freeze_writes_the_twenty_flat_keys(tmp_path, monkeypatch):
    fixture = _freeze_fixture(tmp_path, monkeypatch)
    args = _freeze_args(tmp_path, fixture)
    assert refine.phase_freeze(args) == 0
    target = tmp_path / "artifacts" / "final-artifacts.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "v104-final/1"
    for key in refine.REQUIRED_FLAT:
        assert key in payload, key
    assert payload["state_checkpoint"] == refine.rel(fixture["state"])
    assert payload["state_checkpoint_sha256"] == file_digest(fixture["state"])
    assert payload["policy_checkpoint_sha256"] == file_digest(fixture["policy"])
    assert payload["arch"] == "v2" and payload["depth"] == 1 and payload["seed"] == 7
    assert payload["external_guard_uses"] == 0
    assert payload["evidence"] == {}
    assert payload["readiness"]["status"] == "hard-lines-met"
    assert payload["readiness"]["unmet"] == []
    # frozen_at must carry a timezone (check_v104 refuses a naive stamp).
    from datetime import datetime

    assert datetime.fromisoformat(payload["frozen_at"]).tzinfo is not None


def test_freeze_refuses_an_existing_valid_list(tmp_path, monkeypatch):
    fixture = _freeze_fixture(tmp_path, monkeypatch)
    args = _freeze_args(tmp_path, fixture)
    assert refine.phase_freeze(args) == 0
    with pytest.raises(ValueError, match="already holds a valid"):
        refine.phase_freeze(args)


def test_freeze_refuses_a_non_seven_seed_and_guard_uses(tmp_path, monkeypatch):
    fixture = _freeze_fixture(tmp_path, monkeypatch, seed=17)
    with pytest.raises(ValueError, match="only the pre-selected seed 7"):
        refine.phase_freeze(_freeze_args(tmp_path, fixture))
    fixture = _freeze_fixture(tmp_path / "second", monkeypatch)
    with pytest.raises(ValueError, match="external_guard_uses=2"):
        refine.phase_freeze(_freeze_args(tmp_path / "second", fixture, external_guard_uses=2))


def test_freeze_labels_unmet_hard_lines(tmp_path, monkeypatch):
    fixture = _freeze_fixture(tmp_path, monkeypatch, position=13.1, danger=0.62, bucket=3.05)
    assert refine.phase_freeze(_freeze_args(tmp_path, fixture)) == 0
    payload = json.loads((tmp_path / "artifacts" / "final-artifacts.json").read_text(encoding="utf-8"))
    assert payload["readiness"]["status"] == "candidate-unmet-hard-lines"
    joined = " | ".join(payload["readiness"]["unmet"])
    assert "A-01 position" in joined and "A-01 danger" in joined and "A-02" in joined


def test_freeze_refuses_an_empty_policy_selection(tmp_path, monkeypatch):
    fixture = _freeze_fixture(tmp_path, monkeypatch)
    summary = json.loads(fixture["summary"].read_text(encoding="utf-8"))
    summary["stages"]["policy"]["best"] = {}
    fixture["summary"].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="stages.policy.best has no checkpoint"):
        refine.phase_freeze(_freeze_args(tmp_path, fixture))


def test_freeze_refuses_a_summary_sha_mismatch(tmp_path, monkeypatch):
    fixture = _freeze_fixture(tmp_path, monkeypatch)
    summary = json.loads(fixture["summary"].read_text(encoding="utf-8"))
    summary["stages"]["state"]["best"]["checkpoint_sha256"] = "b" * 64
    fixture["summary"].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="v104:CHECKPOINT"):
        refine.phase_freeze(_freeze_args(tmp_path, fixture))


def test_freeze_dry_run_writes_nothing(tmp_path, monkeypatch):
    """The default is a dry run: the freeze only writes with --execute."""
    fixture = _freeze_fixture(tmp_path, monkeypatch)
    args = _freeze_args(tmp_path, fixture, execute=False)
    assert refine.phase_freeze(args) == 0
    assert not (tmp_path / "artifacts" / "final-artifacts.json").exists()
    assert refine.phase_freeze(_freeze_args(tmp_path, fixture)) == 0
    assert (tmp_path / "artifacts" / "final-artifacts.json").is_file()


def test_freeze_requires_the_declared_inputs(tmp_path):
    args = refine.build_parser().parse_args(["--phase", "freeze", "--out", str(tmp_path / "x")])
    with pytest.raises(ValueError, match="freeze is missing required input"):
        refine.phase_freeze(args)


def test_freeze_requires_a_selection_reason(tmp_path, monkeypatch):
    fixture = _freeze_fixture(tmp_path, monkeypatch)
    args = _freeze_args(tmp_path, fixture, selection_reason="", selection_reason_path="")
    with pytest.raises(ValueError, match="selection-reason"):
        refine.phase_freeze(args)


# --------------------------------------------------------------------------- #
# screening (設計 §5.1): 24 perception epochs, never eligible
# --------------------------------------------------------------------------- #
def test_screening_run_is_never_acceptance_eligible(tmp_path):
    dataset_dir = synthetic_dataset(tmp_path)
    out = tmp_path / "run"
    summary = vision_train.train(
        cfg(
            dataset_dir,
            out,
            arch="v2",
            stage="state",
            epochs_state=1,
            selection_contract="v104",
            history_contract="causal-segment-v1",
            policy_objective="legacy-reg",
            depth=1,
            max_val_batches=0,
            allow_missing_manifest=False,
            screening=True,
        )
    )
    # Full, unrestricted validation - and still not eligible: a screening run may
    # never stand in for the formal budget.
    assert summary["coverage"]["val"]["actual_frames"] == len(split_of(dataset_dir)[1])
    assert summary["coverage"]["val"]["complete"] is True
    assert summary["acceptance_eligible"] is False
    assert summary["recipe"]["screening"] is True
    assert summary["config"]["screening"] is True


def test_screening_recipe_epoch_rule_and_partial_matrix(tmp_path):
    if not (ROOT / "Document/Update/Alpha/v1.0.4/Stage3/S3-4/gate-v3.json").is_file():
        pytest.skip("the S3-4 gate verdict is not present")
    recipes = []
    for name, local in (("E4", False), ("E5", True)):
        config = v3_config_block(
            local_enabled=local,
            epochs_state=24,
            screening=True,
            note="screening 24+12, not a formal comparison",
        )
        recipes.append(
            write_recipe(
                tmp_path / f"{name}-screen.json",
                experiment=name,
                train_config=config,
                gate={
                    "path": refine.rel(GATE_V3),
                    "waiver": {"by": "Q6", "items": ["throughput_le_60s"], "source": "Q6"},
                },
            )
        )
    plans = [
        refine.validate_recipe(refine.load_recipe(path), path=path, route="v3-st", phase="v3-matrix")
        for path in recipes
    ]
    assert all(plan["screening"] for plan in plans)
    result = refine.check_single_variable(plans)
    assert result["experiments"] == ["E4", "E5"]
    assert result["screening_only"] is True
    # 24 perception epochs without the screening flag is refused.
    bad = json.loads(recipes[1].read_text(encoding="utf-8"))
    bad["train_config"]["screening"] = False
    bad_path = tmp_path / "E5-not-screen.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="the declared budget is 72"):
        refine.validate_recipe(refine.load_recipe(bad_path), path=bad_path, route="v3-st", phase="v3-matrix")
    # ... and the partial set is only allowed for declared screening runs.
    bad["train_config"]["screening"] = False
    bad["train_config"]["epochs_state"] = 72
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    full = refine.validate_recipe(refine.load_recipe(bad_path), path=bad_path, route="v3-st", phase="v3-matrix")
    formal = [
        refine.validate_recipe(refine.load_recipe(path), path=path, route="v3-st", phase="v3-matrix")
        for path in recipes[:1]
    ]
    with pytest.raises(ValueError, match="needs exactly the recipes"):
        refine.check_single_variable([full, formal[0]])


# --------------------------------------------------------------------------- #
# dry run / CLI contract
# --------------------------------------------------------------------------- #
def test_v3_matrix_dry_run_creates_nothing(tmp_path, capsys):
    paths = v3_recipe_trio(tmp_path)
    out = tmp_path / "evidence"
    argv = ["--phase", "v3-matrix", "--route", "v3-st", "--out", str(out)]
    for path in paths:
        argv += ["--recipe", str(path)]
    assert refine.main(argv) == 0
    assert not out.exists()
    printed = capsys.readouterr().out
    assert "v3-matrix plan" in printed
    assert "--max-val-batches 0" in printed
    assert "E4" in printed and "E5" in printed and "E6" in printed


def test_r2_dry_run_requires_the_weight_recipe_and_creates_nothing(tmp_path, monkeypatch):
    dataset_dir = synthetic_dataset(tmp_path)
    train, val, _sha = split_of(dataset_dir)
    npz = _weight_file(
        tmp_path / "weights",
        train_indices=train,
        val_indices=val,
        history_contract="causal-segment-v1",
    )
    # The plan validates the weight table against dataset A; point the tool at the
    # synthetic stand-in so the check itself still runs.
    monkeypatch.setattr(refine, "DATASET_A", dataset_dir)
    config = train_config_block(
        arch="v2",
        epochs_state=72,
        epochs_policy=12,
        position_sample_weights=refine.rel(npz),
    )
    recipe = {
        "schema": refine.RECIPE_SCHEMA,
        "phase": "r2",
        "experiment": "R2",
        "route": "v2",
        "dataset": refine.rel(refine.DATASET_A),
        "run_dir": "third_party/th08/build/rl/runs/v104-R2-s7",
        "train_config": config,
        "baseline": {"state_pos_px": 13.0557, "source": "v104-state-d3-s7"},
    }
    path = tmp_path / "r2.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    out = tmp_path / "evidence"
    assert refine.main(["--phase", "r2", "--route", "v2", "--out", str(out), "--recipe", str(path)]) == 0
    assert not out.exists()

    recipe["train_config"]["position_sample_weights"] = ""
    path.write_text(json.dumps(recipe), encoding="utf-8")
    # A known contract error reaches the CLI as exit 2 with the v104 prefix.
    assert refine.main(["--phase", "r2", "--route", "v2", "--out", str(out), "--recipe", str(path)]) == 2


def test_r3_requires_the_learnable_variant(tmp_path):
    config = train_config_block(arch="v2", epochs_state=72, epochs_policy=12)
    recipe = {
        "schema": refine.RECIPE_SCHEMA,
        "phase": "r3",
        "experiment": "R3",
        "route": "v2",
        "dataset": refine.rel(refine.DATASET_A),
        "run_dir": "third_party/th08/build/rl/runs/v104-R3-s7",
        "train_config": config,
        "baseline": {"danger_corr_perframe_median": 0.6198, "source": "v104-state-d3-s7"},
    }
    path = tmp_path / "r3.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    out = tmp_path / "evidence"
    assert refine.main(["--phase", "r3", "--out", str(out), "--recipe", str(path)]) == 2
    config["danger_head_variant"] = "learnable-v104"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    assert refine.main(["--phase", "r3", "--out", str(out), "--recipe", str(path)]) == 0
    assert not out.exists()


def test_phase_switches_are_scoped(tmp_path):
    with pytest.raises(ValueError, match="--throughput is only valid"):
        refine.run_phase(
            refine.build_parser().parse_args(["--phase", "r2", "--out", str(tmp_path), "--throughput"])
        )
    with pytest.raises(ValueError, match="--smoke is only valid"):
        refine.run_phase(
            refine.build_parser().parse_args(["--phase", "r3", "--out", str(tmp_path), "--smoke"])
        )
    with pytest.raises(ValueError, match="is the v3-st matrix"):
        refine.run_phase(
            refine.build_parser().parse_args(["--phase", "v3-matrix", "--out", str(tmp_path)])
        )
    with pytest.raises(ValueError, match="needs --out"):
        refine.run_phase(refine.build_parser().parse_args(["--phase", "hard-frames"]))


def test_hard_frames_dry_run_needs_a_source_and_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match="needs --source-state"):
        refine.run_phase(
            refine.build_parser().parse_args(["--phase", "hard-frames", "--out", str(tmp_path / "e")])
        )
    with pytest.raises(ValueError, match="does not exist"):
        refine.run_phase(
            refine.build_parser().parse_args(
                [
                    "--phase",
                    "hard-frames",
                    "--out",
                    str(tmp_path / "e"),
                    "--source-state",
                    str(tmp_path / "missing.pt"),
                ]
            )
        )


def test_integrate_requires_both_parents_to_have_helped(tmp_path):
    config = train_config_block(
        arch="v2",
        epochs_state=72,
        epochs_policy=12,
        danger_head_variant="learnable-v104",
        position_sample_weights="some/weights.npz",
    )
    parent = tmp_path / "r2-phase.json"
    parent.write_text(
        json.dumps(
            {
                "schema": refine.OUTPUT_SCHEMA,
                "status": "done",
                "results": [{"gain_vs_baseline": {"improved": False, "detail": "no gain"}}],
            }
        ),
        encoding="utf-8",
    )
    recipe = {
        "schema": refine.RECIPE_SCHEMA,
        "phase": "integrate",
        "experiment": "R2R3",
        "route": "v2",
        "dataset": refine.rel(refine.DATASET_A),
        "run_dir": "third_party/th08/build/rl/runs/v104-R2R3-s7",
        "train_config": config,
        "parents": {"r2": str(parent), "r3": str(parent)},
    }
    path = tmp_path / "integrate.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    assert (
        refine.main(
            ["--phase", "integrate", "--route", "v2", "--out", str(tmp_path / "e"), "--recipe", str(path)]
        )
        == 2
    )
    recipe["parents"] = {"r2": str(parent)}
    path.write_text(json.dumps(recipe), encoding="utf-8")
    assert (
        refine.main(
            ["--phase", "integrate", "--route", "v2", "--out", str(tmp_path / "e"), "--recipe", str(path)]
        )
        == 2
    )
    assert not (tmp_path / "e").exists()


def test_estimate_distinguishes_v2_and_unknown_v3():
    v2 = refine.estimate_epoch_seconds("v2", None)
    assert v2["perception"] is not None and "fallback" in v2["source"]
    v3 = refine.estimate_epoch_seconds("v3-st", None)
    assert v3["perception"] is None and v3["policy"] is None
    assert "no v3 throughput evidence" in v3["source"]
