"""Tests of cache identity and grouped training protocol, without game I/O."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from touhou_rl.macro_train import grouped_split, load_cache


def cache_fixture(tmp_path):
    baseline = tmp_path / "base.pt"
    baseline.write_bytes(b"identity-only fixture; not a loadable network")
    values = dict(features=np.zeros((4, 8), np.float32), base_logits=np.zeros((4,18), np.float32),
                  witness_mask=np.ones((4,18), bool), route_target=np.zeros((4,2), np.float32),
                  valid=np.ones(4,bool), route_valid=np.ones(4,bool), groups=np.array(["A","A","B","B"]))
    metadata = {"schema":"v372-macro-cache/1", "source_sha256":hashlib.sha256(baseline.read_bytes()).hexdigest(),
                "input_kind":"synthetic_test_features", "label_semantics":"positive_verified_route_witnesses"}
    def save():
        path = tmp_path / "cache.npz"
        np.savez(path, **values, metadata=np.array(json.dumps(metadata)))
        return path
    return baseline, values, metadata, save


def test_groups_never_leak():
    groups = np.array(["wall-a", "wall-a", "wall-b", "wall-b", "ring", "empty"])
    train, val = grouped_split(groups)
    assert not set(groups[train]) & set(groups[val])
    assert np.array_equal(train, grouped_split(groups)[0])
    assert train.any() and val.any()
    with pytest.raises(ValueError):
        grouped_split(np.array(["one", "one"]))


def test_cache_contract(tmp_path):
    baseline, values, meta, save = cache_fixture(tmp_path)
    data, metadata = load_cache(save(), baseline)
    assert data["features"].shape == (4,8)
    assert metadata["input_kind"] == "synthetic_test_features"
    meta["source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        load_cache(save(), baseline)


def test_unknown_is_not_negative_label(tmp_path):
    baseline, values, meta, save = cache_fixture(tmp_path)
    values["witness_mask"][0] = False
    values["valid"][0] = False
    values["route_valid"][0] = False
    load_cache(save(), baseline)
    values["valid"][0] = True
    with pytest.raises(ValueError, match="no positive"):
        load_cache(save(), baseline)


def test_reject_nonfinite_and_bce_semantics(tmp_path):
    baseline, values, meta, save = cache_fixture(tmp_path)
    meta["label_semantics"] = "beam_failed_equals_dead"
    with pytest.raises(ValueError, match="negative"):
        load_cache(save(), baseline)
    meta["label_semantics"] = "positive_verified_route_witnesses"
    values["features"][0,0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        load_cache(save(), baseline)
