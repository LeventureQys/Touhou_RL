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

import json
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
    _per_step_disp,
    binary_auc,
    brier,
    build_labels,
    clearance_now,
    exit_info,
    extract_episode,
    horizon_clearance,
    judge,
    load_dataset,
    loeo_predict,
    measure_speeds,
    occupancy,
    player_cell,
    resolve_cells,
    resolve_survival_fallback,
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
                "frameskip": 2,
                "frame_pre": 2 * i,
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


# ---------------------------------------------------------------------------
# D3: attributed_spawns_60 and the d3 gate set
# ---------------------------------------------------------------------------


def test_attributed_spawns_label_is_nan_without_slot_ids():
    records, bullets = _synth_episode(60)
    payload = extract_episode(records, bullets)
    label = payload["labels"]["attributed_spawns_60"]
    # Old traces have no slots, so the label must be unknown - never "zero spawns".
    assert np.isnan(label).all()


def test_attributed_spawns_label_sums_the_forward_window():
    prim = _primitives(120)
    prim["attributed_spawns"] = np.ones(120)
    prim["attributed_spawns_known"] = np.ones(120)
    labels = build_labels(prim)
    assert labels["attributed_spawns_60"][0] == pytest.approx(30.0)
    assert np.isnan(labels["attributed_spawns_60"][-1])
    prim["attributed_spawns_known"] = np.zeros(120)
    assert np.isnan(build_labels(prim)["attributed_spawns_60"]).all()


def test_extract_episode_attributes_spawns_to_nearby_enemies():
    """A bullet that appears next to an enemy must count as that enemy's spawn."""
    frames = 40
    records = []
    positions, velocities, halves, slots, offsets = [], [], [], [], [0]
    for i in range(frames):
        px, py = 192.0, 380.0
        # One persistent bullet (slot 1) plus one fresh bullet each decision that
        # appears next to the enemy at (180, 60).
        positions.append([192.0, 300.0])
        velocities.append([0.0, 0.0])
        halves.append(2.0)
        slots.append(1)
        if i > 0:
            # Fixed spawn point 10 px below the enemy: inside SPAWN_RADIUS (32 px)
            # for every decision, so the attribution radius is not what changes.
            positions.append([180.0, 70.0])
            velocities.append([0.0, 0.0])
            halves.append(2.0)
            # A fresh pool slot per decision is what the game does for a new
            # bullet; reusing one slot id would (correctly) look like a survivor.
            slots.append(100 + i)
        offsets.append(len(slots))
        records.append(
            {
                "decision": i + 1,
                "frameskip": 2,
                "frame_pre": 2 * i,
                "frame_post": 2 * (i + 1),
                "clock": {"stage": 0, "stage_frames": 2 * (i + 1), "current_spell": -1},
                "player": {"x": px, "y": py, "vx": 0.0, "vy": 0.0, "state": 3, "focus_mode": 0},
                "resources": {"power": 10.0, "lives": 2.0, "bombs": 3.0, "bombs_used": 0.0, "deaths": 0.0},
                "bosses": [],
                "enemies": [
                    {
                        "slot": 0,
                        "x": 180.0,
                        "y": 60.0,
                        "life": 20,
                        "max_life": 20,
                        "boss": False,
                        "item_drop": 1,
                        "damageable": True,
                        "enemy_index": 7,
                        "phase_starting_life": 20,
                        "shoot_interval_frames": 30,
                    }
                ],
                "items": [],
                "shot": {"point_item_value_line": 128.0},
                "frame_events": {"item_collected": False},
                "events": [],
            }
        )
    bullets = {
        "positions": np.asarray(positions, dtype=np.float32),
        "velocities": np.asarray(velocities, dtype=np.float32),
        "half": np.asarray(halves, dtype=np.float32),
        "slots": np.asarray(slots, dtype=np.int32),
        "offsets": np.asarray(offsets, dtype=np.int64),
    }
    payload = extract_episode(records, bullets)
    block = payload["blocks"]["ENEMY"]
    column = payload["columns"]["ENEMY"].index("ENEMY_attributed_spawns")
    # Decision 0 has no previous snapshot; every later decision sees one spawn
    # next to the enemy, i.e. exactly one attributed bullet.
    assert block[1:, column].tolist() == [1.0] * (frames - 1)
    assert payload["labels"]["attributed_spawns_60"][0] == pytest.approx(30.0)
    # The new enemy columns are populated, not dead.
    enemy_columns = payload["columns"]["ENEMY"]
    damageable = payload["blocks"]["ENEMY"][:, enemy_columns.index("ENEMY_damageable_count")]
    assert damageable.min() == 1.0
    phase = payload["blocks"]["ENEMY"][:, enemy_columns.index("ENEMY_phase_frac_mean")]
    assert phase.max() <= 1.0 and phase.min() > 0.0


def test_gate_sets_only_move_the_enemy_gate():
    from touhou_rl.ablate36 import CELLS, CELLS_D3, GATE_SETS

    assert set(GATE_SETS) == {"v1", "d3"}
    assert GATE_SETS["v1"] is CELLS
    v1 = {cell["strategy"]: cell for cell in CELLS}
    d3 = {cell["strategy"]: cell for cell in CELLS_D3}
    assert d3["ENEMY"]["primary"] == ("attributed_spawns_60", ("spearman", 0.50))
    assert d3["ENEMY"]["increments"] == (("attributed_spawns_60", "spearman", 0.05, "and"),)
    assert d3["ENEMY"]["extras"] == (("bullet_delta_60", "spearman", 0.50),)
    for strategy in v1:
        if strategy == "ENEMY":
            continue
        assert d3[strategy] == v1[strategy]
    # Thresholds are identical between the two ENEMY gates - only the label moves.
    assert d3["ENEMY"]["primary"][1] == v1["ENEMY"]["primary"][1]
    assert d3["ENEMY"]["increments"][0][2] == v1["ENEMY"]["increments"][0][2]


def test_resolve_cells_keeps_the_d3_enemy_gate():
    from touhou_rl.ablate36 import GATE_SETS

    dataset, _ = _toy_dataset(deaths=0)
    fallback = resolve_survival_fallback(dataset)
    spec = resolve_cells(fallback, base=GATE_SETS["d3"])
    enemy = next(cell for cell in spec if cell["strategy"] == "ENEMY")
    assert enemy["primary"][0] == "attributed_spawns_60"
    v1_enemy = next(cell for cell in resolve_cells(fallback) if cell["strategy"] == "ENEMY")
    assert v1_enemy["primary"][0] == "bullet_delta_60"


# ---------------------------------------------------------------------------
# per-shot-type movement speeds (D2)
# ---------------------------------------------------------------------------


def _moving_records(n: int = 240, speed: float = 5.0, focus_speed: float = 1.9) -> list[dict]:
    from touhou_rl.ablate36 import _MOVE_VECTORS

    records = []
    for i in range(n):
        move = 1 + (i % 8)
        focus = i % 2
        ux, uy = _MOVE_VECTORS[move]
        comp = (focus_speed if focus else speed) / math.sqrt(2.0) if (ux and uy) else (focus_speed if focus else speed)
        records.append(
            {
                "action": [move, 1, focus, 0],
                "player": {"vx": float(ux * comp), "vy": float(uy * comp), "x": 192.0, "y": 300.0},
            }
        )
    return records


def test_measure_speeds_recovers_youmu_speeds():
    speeds = measure_speeds(_moving_records())
    assert speeds["fallback"] == []
    assert speeds["speed"] == pytest.approx(5.0, abs=0.05)
    assert speeds["focus_speed"] == pytest.approx(1.9, abs=0.05)
    assert speeds["speed_diag"] == pytest.approx(5.0 / math.sqrt(2.0), abs=0.05)


def test_measure_speeds_falls_back_when_the_player_never_moves():
    records = [
        {"action": [0, 1, 0, 0], "player": {"vx": 0.0, "vy": 0.0, "x": 192.0, "y": 300.0}} for _ in range(50)
    ]
    speeds = measure_speeds(records)
    assert set(speeds["fallback"]) == {"speed", "focus_speed"}
    assert speeds["speed"] == 4.0
    assert speeds["focus_speed"] == 2.0


def test_per_step_disp_uses_speed_over_sqrt_two_on_diagonals():
    disp = _per_step_disp(frameskip=2, speeds={"speed": 5.0, "focus_speed": 1.9})
    # action 3 = east, unfocused -> (2 * 5.0, 0)
    assert disp[3][0] == pytest.approx(10.0)
    assert disp[3][1] == pytest.approx(0.0)
    # action 2 = north-east, unfocused -> per-axis 5/sqrt(2) * 2, so |d| = 10
    assert math.hypot(disp[2][0], disp[2][1]) == pytest.approx(10.0)
    # focus rows are the 9..17 half
    assert math.hypot(disp[9 + 2][0], disp[9 + 2][1]) == pytest.approx(2 * 1.9)


def test_faster_speed_reaches_further_in_the_horizon():
    pos = np.zeros((0, 2), dtype=np.float32)
    vel = np.zeros((0, 2), dtype=np.float32)
    half = np.zeros((0,), dtype=np.float32)
    slow = _per_step_disp(2, {"speed": 4.0, "focus_speed": 2.0})
    fast = _per_step_disp(2, {"speed": 5.0, "focus_speed": 1.9})
    _m1, traj_slow = horizon_clearance(192.0, 300.0, pos, vel, half, disp=slow)
    _m2, traj_fast = horizon_clearance(192.0, 300.0, pos, vel, half, disp=fast)
    assert abs(traj_fast[3, -1, 0] - 192.0) > abs(traj_slow[3, -1, 0] - 192.0)


def test_extract_episode_speed_mode_is_opt_in():
    records, bullets = _synth_episode(60)
    default = extract_episode(records, bullets)
    config = extract_episode(records, bullets, speed_mode="config")
    assert default["speed_mode"] == "config"
    assert np.array_equal(default["blocks"]["BASE"], config["blocks"]["BASE"])
    measured = extract_episode(records, bullets, speed_mode="measure")
    # The synthetic trace barely moves, so measure must fall back to the same
    # constants and therefore produce the same features.
    assert measured["speeds"]["fallback"]
    assert np.array_equal(measured["blocks"]["BASE"], config["blocks"]["BASE"])
    with pytest.raises(ValueError, match="unknown speed_mode"):
        extract_episode(records, bullets, speed_mode="bogus")


def _write_synth_probe(tmp_path, episodes: int = 2):
    probe = tmp_path / "probe"
    (probe / "trace").mkdir(parents=True)
    (probe / "bullets").mkdir()
    meta = []
    for index in range(episodes):
        name = f"ep{index:02d}"
        records, bullets = _synth_episode(80)
        with (probe / "trace" / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                record["episode"] = name
                handle.write(json.dumps(record) + "\n")
        np.savez(probe / "bullets" / f"{name}.npz", **bullets)
        meta.append({"episode": name})
    (probe / "episodes.jsonl").write_text(
        "\n".join(json.dumps(item) for item in meta) + "\n", encoding="utf-8"
    )
    return probe


def test_cache_records_speed_mode_and_refuses_a_mismatch(tmp_path):
    probe = _write_synth_probe(tmp_path)
    cache = tmp_path / "features.npz"
    first = load_dataset(probe, cache=cache, speed_mode="config")
    assert first.speed_mode == "config"
    second = load_dataset(probe, cache=cache, speed_mode="config")
    for name in BLOCKS:
        assert np.array_equal(first.blocks[name], second.blocks[name])
    measured_cache = tmp_path / "features-measure.npz"
    measured = load_dataset(probe, cache=measured_cache, speed_mode="measure")
    assert measured.speed_mode == "measure"
    assert len(measured.speeds) == 2
    with pytest.raises(ValueError, match="speed_mode"):
        load_dataset(probe, cache=measured_cache, speed_mode="config")


def test_strict_frames_drops_mismatched_decisions():
    records, bullets = _synth_episode(60)
    # Poison two decisions' frame span the way replay polling does.
    records[10]["frame_post"] = records[10]["frame_pre"] + 3
    records[20]["frame_post"] = records[20]["frame_pre"] + 4
    loose = extract_episode(records, bullets)
    strict = extract_episode(records, bullets, strict_frames=True)
    assert loose["rows"] == 60
    assert strict["rows"] == 58


# ---------------------------------------------------------------------------
# survival-label fallback (执行计划-D2补遗 §3)
# ---------------------------------------------------------------------------


def _toy_dataset(deaths: int, n: int = 480, groups: int = 4):
    from touhou_rl.ablate36 import CELLS, Dataset

    rng = np.random.default_rng(11)
    group_ids = np.repeat(np.arange(groups), n // groups)
    size = group_ids.size
    labels: dict[str, np.ndarray] = {}
    for key in LABELS:
        labels[key] = rng.normal(size=size)
    labels["death_60"] = np.zeros(size)
    labels["death_60"][:deaths] = 1.0
    blocks = {}
    columns = {}
    for name in BLOCKS:
        count = 3
        blocks[name] = rng.normal(size=(size, count)).astype(np.float32)
        columns[name] = [f"{name}_x{i}" for i in range(count)]
    dataset = Dataset(
        blocks=blocks,
        columns=columns,
        labels=labels,
        groups=group_ids,
        episodes=[f"g{i}" for i in range(groups)],
        primitives={"death": np.zeros(size)},
    )
    return dataset, CELLS


def test_survival_fallback_not_triggered_with_enough_deaths():
    dataset, _ = _toy_dataset(deaths=200)
    info = resolve_survival_fallback(dataset)
    assert info["fallback"] is False
    assert resolve_cells(info) is not None
    assert resolve_cells(info)[0]["primary"][0] == "death_60"


def test_survival_fallback_rewrites_only_the_survival_label():
    dataset, cells = _toy_dataset(deaths=0)
    info = resolve_survival_fallback(dataset)
    assert info["fallback"] is True and info["positives"] == 0
    resolved = resolve_cells(info)
    by_name = {cell["strategy"]: cell for cell in resolved}
    original = {cell["strategy"]: cell for cell in cells}

    # LocalDodge's absolute gate moves to the continuous clearance label.
    assert by_name["LocalDodge"]["primary"] == ("min_clearance_60", ("spearman", 0.30))
    # REGION keeps its own label as primary and moves the increment to clearance.
    assert by_name["REGION"]["primary"][0] == "in_region_60"
    assert by_name["REGION"]["increments"] == (("min_clearance_60", "spearman", 0.05, "and"),)
    # LURE's two "or" branches collapse into one after the substitution.
    assert by_name["LURE"]["increments"] == (("min_clearance_60", "spearman", 0.05, "or"),)
    # Unrelated strategies are untouched.
    assert by_name["POWER"]["increments"] == original["POWER"]["increments"]
    assert by_name["ENEMY"]["primary"] == original["ENEMY"]["primary"]
    assert by_name["EMERGENCY"]["extras"] == original["EMERGENCY"]["extras"]


def test_fallback_spec_runs_end_to_end():
    dataset, _ = _toy_dataset(deaths=0)
    info = resolve_survival_fallback(dataset)
    spec = resolve_cells(info)
    ablation = run_ablation(dataset, permutations=2, seed=3, cells_spec=spec)
    verdict = judge(ablation["cells"], cells_spec=spec)
    assert set(verdict["per_strategy"]) == {cell["strategy"] for cell in spec}
    # Every judged label must come from the resolved registry, never death_60.
    for row in verdict["per_strategy"].values():
        if row.get("verdict") == "not_tested":
            continue
        assert row["primary_label"] != "death_60"
        for branch in row["increments"]:
            assert branch["label"] != "death_60"
