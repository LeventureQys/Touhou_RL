"""Unit tests for the collect / process / train pipeline.

These pin the properties that make the pipeline worth having:

* the split is reproducible and its hash *detects* a changed dataset (otherwise
  "train used the manifest's split" is a claim nobody can check);
* ``process`` refuses malformed shards instead of letting them reach training;
* ``train`` refuses a manifest that no longer matches the directory, and refuses a
  model kind that does not match the dataset modality.

No game and no GPU: everything here runs on synthetic shards under ``tmp_path``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from touhou_rl import obs, vision
from touhou_rl.pipeline.manifest import (
    DatasetManifest,
    ManifestError,
    index_digest,
    split_indices,
)
from touhou_rl.pipeline.process import ProcessSpec, run_process
from touhou_rl.pipeline.train import TrainSpec, run_train


def _meta(index: int) -> dict:
    return {
        "frame": index,
        "stage": 0,
        "score": 0,
        "graze": 0,
        "lives": 2,
        "bombs": 3,
        "power": 0,
        "deaths": 0,
    }


def _meta_rows(frames: int) -> np.ndarray:
    return np.array([[i, 0, 0, 0, 2, 3, 0, 0] for i in range(frames)], dtype=np.float32)


def _write_privileged_shard(path, frames: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    np.savez_compressed(
        path,
        self_feat=rng.random((frames, obs.N_SELF), dtype=np.float32),
        bullets=rng.random((frames, obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float32).astype(np.float16),
        danger=rng.integers(0, 8, size=(frames, obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8),
        action=np.zeros((frames, 4), dtype=np.uint8),
        meta=_meta_rows(frames),
    )


def _write_vision_shard(path, frames: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    np.savez_compressed(
        path,
        frame=rng.integers(0, 255, size=(frames, vision.VISION_H, vision.VISION_W, 3), dtype=np.uint8),
        self_feat=rng.random((frames, obs.N_SELF), dtype=np.float32),
        bullets=rng.random((frames, obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float32).astype(np.float16),
        danger=rng.integers(0, 8, size=(frames, obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8),
        action=np.zeros((frames, 4), dtype=np.uint8),
        meta=_meta_rows(frames),
    )


def test_split_is_deterministic_and_disjoint():
    train, val = split_indices(100, seed=7, val_fraction=0.1)
    assert (len(train), len(val)) == (90, 10)
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(100))
    again_train, again_val = split_indices(100, seed=7, val_fraction=0.1)
    assert train == again_train and val == again_val
    other_train, other_val = split_indices(100, seed=8, val_fraction=0.1)
    assert val != other_val


def test_process_writes_manifest_with_hashes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_privileged_shard(raw / "p-00000.npz", 20, seed=1)
    _write_privileged_shard(raw / "p-00001.npz", 20, seed=2)
    result = run_process(ProcessSpec(raw_dir=raw, compute_teacher=False, min_frames=10, verbose=False))
    manifest = DatasetManifest.load(raw / "manifest.json")
    assert manifest.modality == "privileged"
    assert manifest.total_frames == 40
    assert len(manifest.shards) == 2
    assert all(len(record.sha256) == 64 for record in manifest.shards)
    assert result["total_frames"] == 40
    train, val = manifest.verify(raw)
    assert len(train) + len(val) == 40


def test_process_detects_bad_action_range(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "p-00000.npz"
    _write_privileged_shard(path, 8)
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    payload["action"][0, 0] = 99  # invalid move class
    np.savez_compressed(path, **payload)
    with pytest.raises(ManifestError, match="action"):
        run_process(ProcessSpec(raw_dir=raw, compute_teacher=False, min_frames=1, verbose=False))


def test_process_detects_absurd_features(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "p-00000.npz"
    _write_privileged_shard(path, 8)
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    payload["self_feat"][0, 0] = 1e9  # what a wrong memory offset looks like
    np.savez_compressed(path, **payload)
    with pytest.raises(ManifestError, match="absurd"):
        run_process(ProcessSpec(raw_dir=raw, compute_teacher=False, min_frames=1, verbose=False))


def test_manifest_verify_rejects_appended_shard(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_privileged_shard(raw / "p-00000.npz", 20)
    run_process(ProcessSpec(raw_dir=raw, compute_teacher=False, min_frames=10, verbose=False))
    manifest = DatasetManifest.load(raw / "manifest.json")
    _write_vision_shard(raw / "p-00001.npz", 5)  # dataset grew after process
    with pytest.raises(ManifestError):
        manifest.verify(raw)


def test_manifest_verify_detects_changed_split_seed(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_privileged_shard(raw / "p-00000.npz", 20)
    run_process(ProcessSpec(raw_dir=raw, compute_teacher=False, min_frames=10, seed=7, verbose=False))
    manifest = DatasetManifest.load(raw / "manifest.json")
    manifest.splits["seed"] = 999  # simulates a hand-edited manifest
    with pytest.raises(ManifestError, match="reproduc"):
        manifest.verify(raw)


def test_train_refuses_manifest_without_matching_data(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_privileged_shard(raw / "p-00000.npz", 16)
    run_process(ProcessSpec(raw_dir=raw, compute_teacher=False, min_frames=8, verbose=False))
    (raw / "p-00000.npz").unlink()
    with pytest.raises(ManifestError, match="missing"):
        run_train(
            TrainSpec(
                manifest=raw / "manifest.json",
                out_dir=tmp_path / "run",
                model="privileged_bc",
                epochs=1,
                batch_size=4,
            )
        )


def test_train_refuses_wrong_modality(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_privileged_shard(raw / "p-00000.npz", 16)
    run_process(ProcessSpec(raw_dir=raw, compute_teacher=False, min_frames=8, verbose=False))
    with pytest.raises(ValueError, match="needs a vision dataset"):
        run_train(
            TrainSpec(
                manifest=raw / "manifest.json",
                out_dir=tmp_path / "run",
                model="vision",
                epochs=1,
            )
        )


def test_vision_process_and_manifest(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_vision_shard(raw / "v-00000.npz", 12, seed=3)
    result = run_process(
        ProcessSpec(raw_dir=raw, modality="vision", compute_teacher=False, min_frames=8, verbose=False)
    )
    manifest = DatasetManifest.load(raw / "manifest.json")
    assert manifest.modality == "vision"
    assert manifest.frame_shape == [vision.VISION_H, vision.VISION_W, 3]
    assert result["stats"]["frame_shape"] == [vision.VISION_H, vision.VISION_W, 3]
    assert json.loads((raw / "manifest.json").read_text(encoding="utf-8"))["version"] == 1


def test_index_digest_changes_with_indices():
    assert index_digest([1, 2, 3]) == index_digest([1, 2, 3])
    assert index_digest([1, 2, 3]) != index_digest([1, 2, 4])
