"""Offline paired snapshot cache tests; no game or real checkpoint required."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from touhou_rl import macro_cache as cache
from touhou_rl.macro_train import file_sha, load_cache
from touhou_rl.escape.viability import ViabilityConfig


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(depth=2, region_sizes={name: (3, 4) for name in cache.REGIONS}, trunk_hidden=3)
        self.seen = []

    def policy_head(self, features):
        logits = features[:, :1].repeat(1, 18)
        logits[:, 3] += 1
        return logits

    def forward_core(self, regions):
        assert not self.training and not torch.is_grad_enabled()
        self.seen.append({k: v.clone() for k, v in regions.items()})
        features = torch.stack([regions[name].mean((1, 2, 3, 4)) for name in cache.REGIONS], dim=1)
        logits = features[:, :1].repeat(1, 18)
        logits[:, 3] += 1
        return {"features": features, "policy_logits": logits}


def paired_data(n=3, hazards=0):
    result = {name: np.full((n, 2, 3, 4, 3), (j+1)/4, np.float32) for j, name in enumerate(cache.REGIONS)}
    result.update(player_xy=np.tile(np.array([[192., 224.]], np.float32), (n, 1)),
                  previous_candidate=np.zeros(n, np.int64),
                  positions=np.full((hazards, 2), -1000., np.float64),
                  velocities=np.zeros((hazards, 2), np.float64), halves=np.ones((hazards, 2), np.float64),
                  offsets=np.array([0, hazards] + [hazards]*(n-1), np.int64),
                  coverage_complete=np.ones(n, np.bool_), model_known=np.ones(n, np.bool_),
                  metadata=np.array(json.dumps({"input_kind": "synthetic_pixels"})),
                  groups=np.array([f"episode-{i}" for i in range(n)]))
    return result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"stub baseline identity")
    model = Backbone()
    monkeypatch.setattr(cache, "load_vision_checkpoint", lambda *a, **kw: (model, {}))
    return tmp_path, baseline, model


def test_pixels_witnesses_route_and_train_compatibility(setup):
    tmp, baseline, model = setup
    paired, out = tmp / "paired.npz", tmp / "cache.npz"
    values = paired_data()
    values["coverage_complete"][1] = False
    values["model_known"][2] = False
    np.savez(paired, **values)
    meta = cache.build_cache(paired, baseline, out, config=ViabilityConfig(horizon_frames=2, beam_width=1), batch_size=2)
    data, loaded = load_cache(out, baseline)
    assert loaded == meta
    assert meta["source_sha256"] == file_sha(baseline)
    assert meta["snapshot_sha256"] == file_sha(paired)
    assert meta["input_kind"] == "synthetic_pixels"
    assert "conditional" in meta["forecast_semantics"]
    np.testing.assert_array_equal(data["features"], [[.25, .5, .75]] * 3)
    np.testing.assert_array_equal(data["valid"], [True, False, False])
    np.testing.assert_array_equal(data["route_valid"], data["valid"])
    assert data["witness_mask"][0].all()
    assert not data["witness_mask"][1:].any()
    # First frame settles to previous stand-still, second frame moves right 4px.
    np.testing.assert_allclose(data["route_target"][0], [4/368, 0])
    assert not data["route_target"][1:].any()
    assert len(model.seen) == 2
    for name in cache.REGIONS:
        np.testing.assert_array_equal(model.seen[0][name].numpy(), values[name][:2])
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        cache.build_cache(paired, baseline, out)
    assert out.read_bytes() == before


def test_all_hazards_default_horizon_and_explicit_flags(setup, monkeypatch):
    tmp, baseline, _ = setup
    values = paired_data(hazards=1100)
    values["model_known"][1] = False
    paired = tmp / "paired.npz"
    np.savez(paired, **values)
    calls = []
    original = cache.search_viability

    def inspect(field, start, previous, logits, **kwargs):
        calls.append((field, kwargs))
        assert field.config.horizon_frames == 90
        assert field.config.calibrated_frames == 90
        assert np.all(field.spawn == 0)
        assert np.all(field.despawn == 91)
        # Skip expensive search here; actual search integration is tested above.
        return original(field, start, previous, logits, **{**kwargs, "model_known": False})

    monkeypatch.setattr(cache, "search_viability", inspect)
    cache.build_cache(paired, baseline, tmp / "out.npz")
    assert len(calls[0][0].positions) == 1100
    assert calls[0][0].dropped_over_cap == 0
    assert calls[1][1]["model_known"] is False


@pytest.mark.parametrize("field,value", [
    ("previous_candidate", np.array([0, 1, 18], np.int64)),
    ("previous_candidate", np.zeros(3, np.int32)),
    ("offsets", np.array([0, 1, 0, 0], np.int64)),
    ("offsets", np.zeros(4, np.int32)),
    ("offsets", np.array([0, 0, 0, 1], np.int64)),
    ("coverage_complete", np.ones(3, np.int64)),
    ("model_known", np.ones((3, 1), np.bool_)),
    ("groups", np.array(["a", "", "c"])),
    ("groups", np.array([b"a", b"b", b"c"])),
    ("groups", np.array(["a", "  ", "c"])),
    ("player_xy", np.array([[np.nan, 224]] * 3, np.float32)),
    ("player_xy", np.array([[0, 224]] * 3, np.float32)),
    ("playfield", np.zeros((3, 2, 3, 4, 3), np.float64)),
    ("playfield", np.full((3, 2, 3, 4, 3), np.inf, np.float32)),
    ("playfield", np.full((3, 2, 3, 4, 3), 255, np.float32)),
    ("hud_score", np.zeros((3, 1, 3, 4, 3), np.float32)),
    ("positions", np.zeros((0, 3), np.float32)),
    ("velocities", np.zeros((1, 2), np.float32)),
])
def test_strict_input_validation(tmp_path, field, value):
    data = paired_data()
    data[field] = value
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError):
        cache.load_paired(path)


@pytest.mark.parametrize("field", ["model_known", "coverage_complete", "positions", "groups"])
def test_missing_required_not_defaulted(tmp_path, field):
    data = paired_data()
    del data[field]
    path = tmp_path / "missing.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="missing paired fields"):
        cache.load_paired(path)


@pytest.mark.parametrize("field,value", [
    ("halves", np.array([[-1., 2.]])),
    ("velocities", np.array([[np.nan, 0.]])),
    ("positions", np.array([[1, 2]], np.int64)),
])
def test_hazard_geometry_validation(tmp_path, field, value):
    data = paired_data(hazards=1)
    data[field] = value
    path = tmp_path / "bad.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError):
        cache.load_paired(path)


def test_budget_not_negative_and_cli(setup, capsys):
    tmp, baseline, _ = setup
    paired = tmp / "paired.npz"
    np.savez(paired, **paired_data())
    out = tmp / "out.npz"
    assert cache.main(["--paired", str(paired), "--baseline", str(baseline), "--out", str(out),
                       "--max-frame-checks", "0"]) == 0
    data, meta = load_cache(out, baseline)
    assert not data["valid"].any() and not data["route_valid"].any()
    assert not data["witness_mask"].any()
    assert meta["status_counts"] == {"budget": 3}
    assert "positive_verified_route_witnesses" in capsys.readouterr().out


def test_baseline_shape_and_nonfinite_outputs_fail_without_artifact(setup, monkeypatch):
    tmp, baseline, model = setup
    paired = tmp / "paired.npz"
    values = paired_data()
    values["playfield"] = np.zeros((3, 2, 4, 4, 3), np.float32)
    np.savez(paired, **values)
    out = tmp / "out.npz"
    with pytest.raises(ValueError, match="baseline contract"):
        cache.build_cache(paired, baseline, out)
    assert not out.exists()
    np.savez(paired, **paired_data())
    monkeypatch.setattr(model, "forward_core", lambda r: {"features": torch.full((3, 3), float("nan")), "policy_logits": torch.zeros(3, 18)})
    with pytest.raises(ValueError, match="invalid baseline features"):
        cache.build_cache(paired, baseline, out)
    assert not out.exists()


def test_goal_deadline_uses_declared_target_and_records_diagnostics(setup):
    tmp, baseline, _ = setup
    values = paired_data()
    values["target_xy"] = values["player_xy"] + np.array([4., 0.], np.float32)
    values["target_deadline_frames"] = np.full(3, 2, np.int64)
    paired, out = tmp / "goal.npz", tmp / "goal-cache.npz"
    np.savez(paired, **values)
    meta = cache.build_cache(paired, baseline, out,
                            config=ViabilityConfig(horizon_frames=3, target_radius_px=0, beam_width=1))
    data, _ = load_cache(out, baseline)
    assert data["valid"].all()
    np.testing.assert_allclose(data["route_target"], [[4/368, 0]] * 3)
    assert meta["route_conditioning"] == "goal-conditioned"
    assert meta["target_deadline_frames"] == [2, 2, 2]
    assert meta["valid_count"] == 3
    assert meta["coverage_complete_count"] == meta["model_known_count"] == 3
    assert meta["unknown_reason_counts"] == {}
    assert meta["teacher_config"]["horizon_frames"] == 3


@pytest.mark.parametrize("update", [
    {"target_deadline_frames": np.ones(3, np.int64)},
    {"target_xy": np.zeros((3, 3), np.float32)},
    {"metadata": np.array('{"input_kind":"synthetic_test_features"}')},
    {"metadata": np.array(['{"input_kind":"real_pixels"}'])},
    {"metadata": np.array("not json")},
])
def test_goal_and_provenance_validation(tmp_path, update):
    values = paired_data()
    values.update(update)
    paired = tmp_path / "bad.npz"
    np.savez(paired, **values)
    with pytest.raises(ValueError):
        cache.load_paired(paired)
