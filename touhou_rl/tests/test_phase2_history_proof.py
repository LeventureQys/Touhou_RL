import json
from pathlib import Path

import numpy as np
import pytest

from touhou_rl import phase2_history_proof as proof


class FakeDataset:
    def __init__(self, root, **kwargs):
        self.offsets = [(Path("student-000.npz"), 0, 4)]
        self.meta = np.asarray([
            [0, 1, 0, 0, 3, 0, 0, 0],
            [2, 1, 0, 0, 3, 0, 0, 0],
            [4, 1, 0, 0, 3, 0, 0, 1],
            [10, 2, 0, 0, 3, 0, 0, 1],
        ], dtype=np.float64)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __len__(self): return 4
    def meta_of_shard(self, path): return self.meta
    def stack_indices(self, centers, depth):
        # Deliberate episode/split crossing for center 1.
        return np.asarray([[0, 0, 0], [1, 1, 2], [2, 2, 2], [3, 3, 3]], dtype=np.int64)


def _fixture(tmp_path, episodes):
    source = tmp_path / "source"; source.mkdir()
    (source / "student-rollout-report.json").write_text('{"fixture":true}', encoding="utf-8")
    cache = tmp_path / "cache"; cache.mkdir()
    np.savez(cache / "chunk-arrays.npz", episode=np.asarray(episodes, dtype=np.int64),
             sample_index=np.arange(len(episodes), dtype=np.int64))
    (cache / "chunk-identity.json").write_text('{}', encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"episodes": {"train": [1], "dev": [2], "test": [3]}}), encoding="utf-8")
    return source, cache, protocol


def test_history_proof_catches_crossing_and_hashes(monkeypatch, tmp_path):
    monkeypatch.setattr(proof, "VisionDataset", FakeDataset)
    source, cache, protocol = _fixture(tmp_path, [1, 1, 2, 3])
    out = tmp_path / "report.json"
    report = proof.verify(source, cache, protocol, out)
    assert report["crossing_count"] == 1
    assert report["passed"] is False
    assert len(report["history_index_sha256"]) == 64
    assert report["total_rows"] == 4
    assert out.exists()
    with pytest.raises(FileExistsError):
        proof.verify(source, cache, protocol, out)


def test_history_proof_unknown_episode_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(proof, "VisionDataset", FakeDataset)
    source, cache, protocol = _fixture(tmp_path, [1, 99, 2, 3])
    with pytest.raises(ValueError, match="unknown episode"):
        proof.verify(source, cache, protocol, tmp_path / "report.json")
