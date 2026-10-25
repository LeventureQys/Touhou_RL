"""Unit tests for v3.6.0 T1 ablation (``touhou_rl.ablate36``).

The interesting tests here are not the geometry ones; they are the two leak
guards:

* ``test_loeo_does_not_see_test_fold_labels`` - flipping every label inside one
  episode must not change that episode's out-of-fold prediction.  If it does,
  the "with block" number is contaminated by the answer key and every increment
  downstream is meaningless.
* ``test_loeo_rank_correlates_with_real_signal`` - a feature that genuinely
  carries the label must produce a high Spearman, otherwise the harness is
  silent-broken rather than noisy-wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from touhou_rl import spec
from touhou_rl.ablate36 import (
    BLOCKS,
    CLR_CAP,
    LABELS,
    _lookahead_at,
    _lookahead_delta,
    _lookahead_min,
    binary_auc,
    brier,
    build_labels,
    clearance_now,
    exit_info,
    extract_episode,
    horizon_clearance,
    judge,
    loeo_predict,
    occupancy,
    player_cell,
    run_ablation,
    spearman,
    topk_accuracy,
)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_occupancy_is_conservative_3x3_splat():
    # One bullet at the centre of cell (10, 10) at 8 px resolution.
    px = spec.PLAYFIELD_X + 10 * 8 + 4
    py = spec.PLAYFIELD_Y + 10 * 8 + 4
    occ, den = occupancy(np.array([[px, py]], dtype=np.float32), 8)
    assert den.shape == (56, 48)
    assert den[10, 10] == 1.0
    assert occ[10, 10]
    # The 8 neighbours are covered (splat), the diagonal 2 steps away is not.
    assert occ[9, 9] and occ[11, 11] and occ[9, 11]
    assert not occ[8, 8]
    # "density" is how many centres fall in a cell's 3x3 neighbourhood, so one
    # bullet contributes 1 to each of nine cells.
    assert float(den.sum()) == 9.0


def test_occupancy_drops_offscreen_bullets():
    occ, den = occupancy(np.array([[-500.0, -500.0], [9999.0, 9999.0]], dtype=np.float32), 8)
    assert not occ.any()
    assert float(den.sum()) == 0.0


def test_player_cell_is_clamped_into_grid():
    assert player_cell(spec.MOVE_AREA_X, spec.MOVE_AREA_Y, 8) == (0, 0)
    y, x = player_cell(-1000.0, -1000.0, 8)
    assert (y, x) == (0, 0)
    y, x = player_cell(1e6, 1e6, 8)
    assert (y, x) == (55, 47)


def test_exit_info_free_space_has_ring_zero():
    mask = np.zeros((56, 48), dtype=bool)
    ring, area, dx, dy = exit_info(mask, (28, 24))
    assert ring == 0
    assert area > 100
    assert math.isfinite(dx) and math.isfinite(dy)


def test_exit_info_blocked_world_has_no_exit():
    mask = np.ones((56, 48), dtype=bool)
    ring, area, _dx, _dy = exit_info(mask, (28, 24))
    assert ring > 0
    assert area == 0


def test_exit_info_respects_safe_min_cells():
    mask = np.ones((56, 48), dtype=bool)
    # Single-cell pockets north of the player are not exits: SAFE_MIN_CELLS is 8.
    mask[27, 24] = False
    mask[20, 24] = False
    ring, area, _dx, _dy = exit_info(mask, (28, 24))
    assert area == 0
    assert ring >= 1


def test_exit_info_finds_a_real_island():
    mask = np.ones((56, 48), dtype=bool)
    # A 10-cell free island at Chebyshev ring 6 to the north.
    mask[21:23, 23:28] = False
    ring, area, _dx, _dy = exit_info(mask, (28, 24))
    assert ring == 6
    assert area >= 8


def test_clearance_now_caps_when_empty():
    pos = np.zeros((0, 2), dtype=np.float32)
    half = np.zeros((0,), dtype=np.float32)
    assert clearance_now(192.0, 384.0, pos, half) == CLR_CAP


def test_clearance_now_subtracts_radii():
    pos = np.array([[192.0, 374.0]], dtype=np.float32)
    half = np.array([2.0], dtype=np.float32)
    got = clearance_now(192.0, 384.0, pos, half)
    expected = 10.0 - 2.0 - 0.825 - 0.5
    assert got == pytest.approx(expected, abs=1e-4)


def test_horizon_clearance_prefers_moving_away():
    # A bullet column directly above the player: moving north must be worse than
    # moving south, and the two must be the extreme values of the 18 actions.
    pos = np.array([[192.0, 350.0]], dtype=np.float32)
    vel = np.zeros((1, 2), dtype=np.float32)
    half = np.array([4.0], dtype=np.float32)
    mins, traj = horizon_clearance(192.0, 384.0, pos, vel, half)
    assert mins.shape == (18,) and traj.shape == (18, 10, 2)
    north, south = mins[1], mins[5]
    assert north < south
    assert north == pytest.approx(mins.min(), abs=1e-6)
    assert south == pytest.approx(mins.max(), abs=1e-6)


def test_horizon_clearance_clips_inside_move_area():
    pos = np.zeros((0, 2), dtype=np.float32)
    vel = np.zeros((0, 2), dtype=np.float32)
    half = np.zeros((0,), dtype=np.float32)
    _mins, traj = horizon_clearance(192.0, 20.0, pos, vel, half)
    assert traj[:, :, 1].min() >= spec.MOVE_AREA_Y - 1e-6
    assert traj[:, :, 0].min() >= spec.MOVE_AREA_X - 1e-6
    assert traj[:, :, 0].max() <= spec.MOVE_AREA_X + spec.MOVE_AREA_W + 1e-6


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------


def _primitives(n: int = 120) -> dict[str, np.ndarray]:
    death = np.zeros(n)
    death[60] = 1.0
    item = np.zeros(n)
    item[10] = 1.0
    bombs_used = np.zeros(n)
    bombs_used[15] = 1.0
    boss = np.full(n, np.nan)
    boss[5:25] = 1000.0
    boss[25:] = 900.0
    return {
        "death": death,
        "power": np.arange(n, dtype=float),
        "bombs": np.full(n, 2.0),
        "bombs_used": bombs_used,
        "n_bullets": np.arange(n, dtype=float) * 2.0,
        "boss_life": boss,
        "clearance": np.linspace(10.0, 1.0, n),
        "danger8": np.linspace(0.0, 1.0, n),
        "region20": np.linspace(0.0, 200.0, n),
        "item_collected": item,
        "px": np.zeros(n),
        "py": np.zeros(n),
    }


def test_death_labels_use_forward_window_only():
    prim = _primitives()
    labels = build_labels(prim)
    # The single death is at index 60; death_60 covers the 30 decisions *after* i,
    # so exactly i in [30, 59] is labelled.
    assert labels["death_60"][30] == 1.0
    assert labels["death_60"][59] == 1.0
    assert labels["death_60"][29] == 0.0
    assert labels["death_60"][60] == 0.0  # a death never labels itself
    # Nothing after the death may carry the label.
    assert np.nan_to_num(labels["death_60"][60:]).sum() == 0.0


def test_death_labels_nan_inside_horizon_of_episode_end():
    prim = _primitives()
    labels = build_labels(prim)
    assert np.isnan(labels["death_60"][-1])
    assert np.isnan(labels["death_60"][-30])
    assert not np.isnan(labels["death_60"][0])


def test_min_clearance_label_is_future_minimum():
    prim = _primitives()
    labels = build_labels(prim)
    assert labels["min_clearance_60"][0] == pytest.approx(prim["clearance"][1:31].min())


def test_bomb_labels():
    prim = _primitives()
    labels = build_labels(prim)
    assert labels["bomb_used_60"][13] == 1.0  # bombs_used rises at 15, within 30
    assert labels["bomb_used_60"][15] == 0.0  # the rise itself is not "future"
    # bomb_needed_60 = death within 60 frames AND bombs available
    assert labels["bomb_needed_60"][30] == 1.0
    assert labels["bomb_needed_60"][0] == 0.0
    prim["bombs"][:] = 0.0
    assert build_labels(prim)["bomb_needed_60"][30] == 0.0


def test_boss_life_delta_is_nan_without_boss():
    prim = _primitives()
    labels = build_labels(prim)
    assert np.isnan(labels["boss_life_delta_60"][0])  # no boss at i, so unknown
    assert np.isnan(labels["boss_life_delta_60"][2])  # boss appears at 5, still unknown
    # Boss HP drops 1000 -> 900 at index 25: index 5 + 30 = 35 already dropped.
    assert labels["boss_life_delta_60"][5] == pytest.approx(100.0)


def test_lookahead_helpers_do_not_read_the_past():
    values = np.arange(10, dtype=float)
    assert _lookahead_at(values, 2)[0] == 2.0
    assert _lookahead_delta(values, 3)[0] == 3.0
    assert np.isnan(_lookahead_at(values, 3)[-1])
    assert _lookahead_min(values, 2)[0] == 1.0


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_spearman_matches_known_values():
    x = np.arange(20, dtype=float)
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert spearman(np.zeros(20), x) is None


def test_binary_auc_and_brier():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    assert binary_auc(y, y) == pytest.approx(1.0)
    assert binary_auc(y, -y) == pytest.approx(0.0)
    assert binary_auc(np.zeros_like(y), y) is None
    assert brier(np.array([0.0, 1.0] * 6), np.array([0.0, 1.0] * 6)) == pytest.approx(0.0)
    assert brier(y, np.full(len(y), 0.5)) == pytest.approx(0.25)


def test_topk_accuracy():
    y = np.array([0, 1, 2, 0, 1, 2] * 3, dtype=float)
    scores = np.eye(3)[y.astype(int)]
    assert topk_accuracy(y, scores, k=1) == pytest.approx(1.0)
    assert topk_accuracy(y, np.zeros((len(y), 3)), k=3) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# leave-one-episode-out ridge
# ---------------------------------------------------------------------------


def _toy(num_groups: int = 5, per_group: int = 60, p: int = 3, seed: int = 7):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(num_groups), per_group)
    X = rng.normal(size=(groups.size, p)).astype(np.float32)
    y = X[:, 0] * 2.0 - X[:, 1] + rng.normal(scale=0.1, size=groups.size)
    return X, y, groups


def test_loeo_does_not_see_test_fold_labels():
    X, y, groups = _toy()
    base = loeo_predict(X, y, groups)
    poisoned = y.copy()
    poisoned[groups == 2] += 1e3  # destroy the answer key of one episode
    after = loeo_predict(X, poisoned, groups)
    mask = groups == 2
    assert np.allclose(base[mask], after[mask])
    assert not np.allclose(base[~mask], after[~mask])


def test_loeo_rank_correlates_with_real_signal():
    X, y, groups = _toy()
    pred = loeo_predict(X, y, groups)
    assert all(np.isfinite(pred))
    assert spearman(pred, y) > 0.9


def test_loeo_ignores_pure_noise():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(6), 80)
    X = rng.normal(size=(groups.size, 4)).astype(np.float32)
    y = rng.normal(size=groups.size)
    pred = loeo_predict(X, y, groups)
    assert abs(spearman(pred, y) or 0.0) < 0.15


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------


def _synth_episode(n: int = 40) -> tuple[list[dict], dict]:
    records = []
    positions, velocities, halves = [], [], []
    for i in range(n):
        px = 180.0 + (i % 5) * 4.0
        py = 380.0 - (i % 7) * 3.0
        bullets = 3
        for k in range(bullets):
            positions.append([100.0 + 20.0 * k + i, 100.0 + 5.0 * i])
            velocities.append([0.0, 2.0])
            halves.append(2.0)
        records.append(
            {
                "decision": i + 1,
                "frame_post": 2 * (i + 1),
                "clock": {"stage": 0, "stage_frames": 2 * (i + 1), "current_spell": -1},
                "player": {"x": px, "y": py, "vx": 0.0, "vy": 0.0, "state": 3, "focus_mode": 0},
                "resources": {
                    "power": float(i),
                    "lives": 2.0,
                    "bombs": 3.0,
                    "bombs_used": 0.0,
                    "deaths": 0.0,
                },
                "bosses": [],
                "enemies": [{"slot": 0, "x": 180.0, "y": 60.0, "life": 20, "max_life": 20, "boss": False, "item_drop": 1}],
                "items": [{"slot": 0, "x": 190.0, "y": 120.0, "type": 0, "state": 0, "onscreen": True, "is_max_value": False}],
                "shot": {"point_item_value_line": 128.0},
                "frame_events": {"item_collected": i % 3 == 0},
                "events": ["death"] if i == 20 else [],
            }
        )
    offsets = np.arange(0, (n + 1) * 3, 3, dtype=np.int64)
    bullets = {
        "positions": np.asarray(positions, dtype=np.float32),
        "velocities": np.asarray(velocities, dtype=np.float32),
        "half": np.asarray(halves, dtype=np.float32),
        "offsets": offsets,
    }
    return records, bullets


def test_extract_episode_shapes_and_finiteness():
    records, bullets = _synth_episode()
    payload = extract_episode(records, bullets)
    assert payload["rows"] == len(records)
    for name in BLOCKS:
        block = payload["blocks"][name]
        assert block.shape[0] == payload["rows"]
        assert block.shape[1] == len(payload["columns"][name])
        assert np.isfinite(block).all(), name
    for name, values in payload["labels"].items():
        assert values.shape[0] == payload["rows"]
    # Death at decision 20 labels the 30 decisions before it (see the label test).
    assert payload["labels"]["death_60"][0] == 1.0


def test_run_ablation_and_judge_are_deterministic():
    # LOEO needs at least two episodes, otherwise no fold has a training set.
    records_a, bullets_a = _synth_episode(80)
    records_b, bullets_b = _synth_episode(80)
    payload_a = extract_episode(records_a, bullets_a)
    payload_b = extract_episode(records_b, bullets_b)
    from touhou_rl.ablate36 import CELLS, Dataset

    dataset = Dataset(
        blocks={name: np.vstack([payload_a["blocks"][name], payload_b["blocks"][name]]) for name in BLOCKS},
        columns=payload_a["columns"],
        labels={key: np.concatenate([payload_a["labels"][key], payload_b["labels"][key]]) for key in LABELS},
        groups=np.repeat([0, 1], payload_a["rows"]),
        episodes=["synth-a", "synth-b"],
        primitives={},
    )
    first = run_ablation(dataset, permutations=2, seed=5)
    second = run_ablation(dataset, permutations=2, seed=5)
    assert [c["delta"] for c in first["cells"]] == [c["delta"] for c in second["cells"]]
    verdict = judge(first["cells"])
    assert set(verdict["per_strategy"]) == {cell["strategy"] for cell in CELLS}
    assert all(row["verdict"] in {"pass", "fail", "underpowered", "unstable", "not_tested"} for row in verdict["per_strategy"].values())


def test_labels_registry_covers_every_built_label():
    records, bullets = _synth_episode(10)
    built = set(build_labels(extract_episode(records, bullets)["primitives"]))
    assert set(LABELS) == built
