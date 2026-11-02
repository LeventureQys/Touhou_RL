"""Forecast tests: time axis, cutoffs, inflation, uncertainty, exact physics match."""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl.escape import contracts, forecast, families, geometry, synth

START = (192.0, 384.0)


def _field(hazards, *, frame=0, config=None):
    frame = frame
    scenario = synth.Scenario("f/0", "TEST", 0, START, tuple(hazards),
                              horizon_frames=200, hold_frames=0)
    return forecast.DangerField.from_scenario(scenario, frame, config or forecast.ForecastConfig())


def test_time_nodes_cover_the_three_bands():
    nodes = forecast.ForecastConfig().time_nodes()
    assert nodes[:15] == tuple(range(2, 31, 2))
    assert nodes[15:25] == tuple(range(36, 91, 6))
    assert nodes[25:] == (105, 120, 135, 150, 165, 180)
    assert all(b > a for a, b in zip(nodes, nodes[1:]))
    assert nodes[-1] == 180


def test_time_axis_is_not_a_union():
    hazard = synth.HazardSpec((192.0, 300.0), (0.0, 0.0), (6.0, 6.0), despawn_frame=30)
    field = _field([hazard])
    assert field.clearance(192.0, 300.0, 10) <= 0.0
    assert field.clearance(192.0, 300.0, 40) == float("inf")


def test_far_wall_is_kept_even_when_it_is_300_px_away():
    scenario = families.build("SC-09", 0)
    field = forecast.DangerField.from_scenario(scenario, 0)
    wall_y = np.asarray([h.position[1] for h in scenario.hazards])
    far_y = wall_y[np.abs(wall_y - 84.0) < 1e-6]
    assert far_y.size >= 20
    kept_y = field.positions[:, 1]
    assert np.sum(np.abs(kept_y - 84.0) < 1e-6) >= far_y.size - 2


def test_irrelevant_hazards_are_dropped_with_a_reason():
    far = [synth.HazardSpec((100.0, -400.0 - 10 * index), (0.0, -1.0), (2.0, 2.0))
           for index in range(5)]
    field = _field(far)
    assert field.positions.shape[0] == 0
    assert field.dropped_irrelevant == 5
    built = forecast.build(field)
    assert built.record.dropped_entities == 5
    assert any("irrelevant" in reason for reason in built.record.drop_reasons)


def test_entity_cap_is_explicit_and_marks_incomplete_coverage():
    config = forecast.ForecastConfig(max_entities=2)
    hazards = [synth.HazardSpec((100.0 + 10 * index, 300.0), (0.0, 0.0), (2.0, 2.0))
               for index in range(10)]
    field = _field(hazards, config=config)
    assert field.positions.shape[0] == 2
    assert field.dropped_over_cap == 8
    built = forecast.build(field)
    assert any("over_cap" in reason for reason in built.record.drop_reasons)


def test_inflation_is_bounded_by_the_gap_geometry():
    gap = 2.0 * contracts.HURTBOX_HALF_PX + 3.0
    walls = families._wall(300.0, gap_x=192.0, gap_width=gap)
    field = _field(walls)
    assert field.clearance(192.0, 300.0, 0, inflation=0.0) == pytest.approx(1.5)
    assert field.clearance(192.0, 300.0, 0, inflation=1.0) == pytest.approx(0.5)
    assert field.clearance(192.0, 300.0, 0, inflation=1.6) < 0.0
    assert geometry.max_inflation_for_gap(gap, contracts.HURTBOX_HALF_PX) == pytest.approx(1.5)


def test_uncertainty_grows_and_calibration_is_declared():
    field = _field([synth.HazardSpec((192.0, 100.0), (0.0, 1.0), (2.0, 2.0))])
    assert field.uncertainty(2) < field.uncertainty(120)
    assert field.calibrated(90) is True
    assert field.calibrated(105) is False
    built = forecast.build(field)
    assert built.record.calibrated is False  # 90 < 180 horizon


def test_beyond_the_horizon_is_unknown_not_safe():
    field = _field([synth.HazardSpec((192.0, 100.0), (0.0, 1.0), (2.0, 2.0))])
    assert field.clearance(192.0, 384.0, 181) is None
    assert field.clearance(192.0, 384.0, -1) is None
    assert field.unknown(181) is True


def test_record_matches_the_contract():
    field = _field([synth.HazardSpec((192.0, 300.0), (0.0, 2.0), (2.0, 2.0))])
    built = forecast.build(field)
    document = built.to_json_dict()
    assert contracts.validate_document(document) == []
    assert len(document["coverage"]) == len(document["time_nodes_frames"])
    rows, cols = document["coverage_shape"]
    assert all(len(row) == rows * cols for row in document["coverage"])
    assert document["valid_until"] == 180  # frame_base + config.horizon_frames


def test_clearance_matches_the_synthetic_physics():
    rng = np.random.default_rng(11)
    hazards = [
        synth.HazardSpec(
            (float(rng.uniform(40, 380)), float(rng.uniform(20, 430))),
            (float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2))),
            (2.0, 2.0),
        )
        for _ in range(30)
    ]
    scenario = synth.Scenario("f/1", "TEST", 0, START, tuple(hazards),
                              horizon_frames=180, hold_frames=0)
    field = forecast.DangerField.from_scenario(scenario, 0)
    for _ in range(50):
        x = float(rng.uniform(8, 376))
        y = float(rng.uniform(16, 432))
        t = int(rng.integers(0, 180))
        expected = synth.frame_clearance(scenario, t, x, y)
        actual = field.clearance(x, y, t, inflation=0.0)
        if expected == float("inf"):
            assert actual == float("inf")
        else:
            assert actual == pytest.approx(expected, abs=1e-9)


def test_swept_tool_sees_a_between_frame_crossing():
    field = _field([synth.HazardSpec((163.0, 384.0), (10.0, 0.0), (2.0, 2.0))])
    assert field.frame_clearance(192.0, 384.0, 2) > 0.0
    assert field.frame_clearance(192.0, 384.0, 4) > 0.0
    swept = field.swept_clearance((192.0, 384.0), (192.0, 384.0), 2, 4)
    assert swept is not None and swept < 0.0


def test_build_is_fast_enough_for_a_400_entity_field():
    import time

    hazards = [
        synth.HazardSpec((40.0 + (index % 20) * 16.0, 40.0 + (index // 20) * 16.0),
                         (0.0, 1.0), (2.0, 2.0))
        for index in range(400)
    ]
    field = _field(hazards)
    started = time.perf_counter()
    built = forecast.build(field)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert len(built.record.time_nodes_frames) == 31
    assert elapsed_ms < 2000.0  # generous; the engineering target is 4 ms
