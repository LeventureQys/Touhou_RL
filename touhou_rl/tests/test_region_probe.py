"""Unit tests for the v3.5.0 S1 offline region probe (no game process)."""

from __future__ import annotations

import numpy as np

from touhou_rl.region_probe import (
    GRID_H,
    GRID_W,
    DecisionFeature,
    _splat,
    auc,
    component_from,
    danger_maps,
    death_labels,
    gate_report,
    nearest_safe_component,
    player_cell,
    safe_component,
)


def test_player_cell_origin_and_clip():
    assert player_cell(32.0, 16.0) == (0, 0)
    assert player_cell(32.0 + 8.0, 16.0 + 8.0) == (1, 1)
    assert player_cell(-1000.0, -1000.0) == (0, 0)
    assert player_cell(10000.0, 10000.0) == (GRID_H - 1, GRID_W - 1)


def test_splat_marks_neighbourhood():
    points = np.array([[32.0 + 8.0 * 10, 16.0 + 8.0 * 20]], dtype=np.float32)
    occ = _splat(points)
    assert occ.shape == (GRID_H, GRID_W)
    assert occ[20, 10]
    assert occ[19:22, 9:12].all()
    assert not occ[0, 0]


def test_danger_maps_shift_with_velocity():
    positions = np.array([[100.0, 100.0]], dtype=np.float32)
    velocities = np.array([[2.0, 0.0]], dtype=np.float32)
    half = np.array([2.0], dtype=np.float32)
    maps = danger_maps(positions, velocities, half)
    y0, x0 = player_cell(100.0, 100.0)
    assert maps[0][y0, x0]
    assert maps[20][y0, x0 + 5]  # 2 px/frame * 20 frames = 40 px = 5 cells


def test_component_from_and_safe_component():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    cells = component_from(mask, (3, 3))
    assert len(cells) == 9
    assert component_from(mask, (0, 0)) == []
    safe = safe_component(mask, (2, 3))
    assert safe and len(safe) > 9  # free region is larger than the obstacle
    safe_edge = safe_component(mask, (0, 0))
    assert safe_edge
    near = nearest_safe_component(mask, (3, 3))
    assert near and len(near) > 9


def test_auc_perfect_and_ties():
    assert auc(np.array([0, 1]), np.array([False, True])) == 1.0
    assert auc(np.array([1, 0]), np.array([False, True])) == 0.0
    assert auc(np.array([1, 1]), np.array([False, True])) == 0.5
    assert auc(np.array([1, 2]), np.array([False, False])) is None


def _feature(decision: int, in_region: bool, death: bool = False, safe: int = 100,
             clearance: float = 20.0) -> DecisionFeature:
    return DecisionFeature(
        decision=decision, frame=decision * 2, player=(192.0, 300.0),
        area_frac=0.1 if in_region else 0.0,
        in_region_20=in_region, in_region_40=in_region,
        region_area_20=100 if in_region else 0, region_area_40=100 if in_region else 0,
        region_direction=(1.0, 0.0) if in_region else None,
        safe_area=safe, clearance=clearance, death=death,
    )


def test_death_labels():
    features = [_feature(1, False), _feature(2, False, death=True), _feature(3, False)]
    labels = death_labels(features, horizon_decisions=1)
    assert labels.tolist() == [True, False, False]


def test_gate_report_with_synthetic_regions():
    features = [_feature(i, True, death=(i == 20), safe=100) for i in range(25)]
    report = gate_report({"ep01": features}, permutations=10, seed=1)
    assert report["gates"]["S1-1"]["pass"] is True
    assert report["gates"]["S1-3"]["pass"] is True
    assert report["totals"]["deaths"] == 1
