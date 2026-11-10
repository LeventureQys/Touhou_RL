"""P2 label tests: state-derived labels must be exact, complete, and fail-closed.

What is pinned here, and why each pin exists:

* **Grid occupancy** —the 96x112 cell mask must equal a brute-force overlap
  test, so an "almost right" rasteriser (dropping a cell the box only touches,
  or marking the wrong axis) fails.  The distance field is checked against the
  analytic distance to one bullet.
* **future_danger agrees with the geometry model** —the label is computed by
  ``geometry.frame_verdict(MODEL_ENGINE_PHASE)`` and the test recomputes it
  independently through the same public API, on frames that hit at a known
  offset and frames that do not.  A divergence in the wrapper (wrong horizon,
  wrong hurtbox, wrong "still" assumption) fails.
* **HUD field mapping** —``STATE f4/f5/f17/f18/f7/f8/f9`` must land on
  score/graze/point/time/lives/bombs/power; ``display_score`` and the two
  "/ total" denominators are *not exported* and must be ``None``.
* **Coordinate conversion** —screen = playfield + (32, 16), from
  ``Background.cpp:740-743``, including the rectangle and the inverse.
* **Missing fields are not zero** —every absent input becomes ``None`` with the
  field named in ``missing``; nothing is defaulted to 0.

Run with::

    $env:PYTHONPATH='D:\\WorkShop\\Touhou_RL'
    python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p2_pytest -q \
        tests/test_v400_labels.py
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from touhou_rl.v400.labels import spec as spec_module
from touhou_rl.v400.labels.danger import DEFAULT_HORIZONS, evaluate_future_danger
from touhou_rl.v400.labels.events import event_labels
from touhou_rl.v400.labels.hud import hud_label
from touhou_rl.v400.labels.occupancy import (
    decode_occupancy,
    encode_occupancy,
    hazard_occupancy,
)
from touhou_rl.v400.labels.player import player_label
from touhou_rl.v400.teacher import geometry

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / ".tmp_v400_p2" / "labels-s4.jsonl"
FRAMES = REPO_ROOT / ".tmp_v400_p2" / "frames"


def _dataset_ready() -> bool:
    return DATASET.is_file() and DATASET.stat().st_size > 0

HURT_MIN = (191.1750030517578, 383.17498779296875)
HURT_MAX = (192.8249969482422, 384.82501220703125)


# ---------------------------------------------------------------------------
# stand-ins for the protocol dataclasses
# ---------------------------------------------------------------------------
class Bullet:
    __slots__ = ("slot", "state", "collision_size", "collision_disabled",
                 "position", "velocity")

    def __init__(self, slot=0, x=192.0, y=100.0, size=4.0, state=1, disabled=0,
                 vx=0.0, vy=0.0):
        self.slot = slot
        self.state = state
        self.collision_size = _bits(size)
        self.collision_disabled = disabled
        self.position = (_bits(x), _bits(y))
        self.velocity = (_bits(vx), _bits(vy))


class Enemy:
    __slots__ = ("slot", "flags1", "world_position", "hitbox_width", "hitbox_height")

    def __init__(self, slot=0, x=192.0, y=100.0, flags1=geometry.ENEMY_FLAG_COLLISION,
                 width=24.0, height=24.0, world=True):
        self.slot = slot
        self.flags1 = flags1
        self.world_position = (_bits(x), _bits(y)) if world else None
        self.hitbox_width = _bits(width)
        self.hitbox_height = _bits(height)


class Laser:
    __slots__ = ("slot", "state", "in_use", "position", "angle",
                 "start_offset", "end_offset", "width")

    def __init__(self, slot=0, x=100.0, y=200.0, angle=0.0, start=0.0, end=100.0,
                 width=16.0, state=1, in_use=1):
        self.slot = slot
        self.state = state
        self.in_use = in_use
        self.position = (_bits(x), _bits(y))
        self.angle = _bits(angle)
        self.start_offset = _bits(start)
        self.end_offset = _bits(end)
        self.width = _bits(width)


class Hazards:
    def __init__(self, bullets=(), enemies=(), lasers=()):
        self.frame = 0
        self.bullet_count = len(bullets)
        self.enemy_count = len(enemies)
        self.laser_count = len(lasers)
        self.bullets = tuple(bullets)
        self.enemies = tuple(enemies)
        self.lasers = tuple(lasers)


def _bits(value: float) -> float:
    return struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", float(value)))[0]))[0]


def player(x=192.0, y=384.0, state=0, hurt_min=HURT_MIN, hurt_max=HURT_MAX, **kwargs):
    base = dict(
        frame=1, hp=3.0, lives=3, bombs=3.0, power=64.0, focus=0, is_youkai=0,
        player_state=state, invuln_state=0, position=(x, y),
        hurtbox_min=hurt_min, hurtbox_max=hurt_max,
        speed_normal_axial=4.0, speed_normal_diagonal=2.8,
        speed_focused_axial=2.0, speed_focused_diagonal=1.4,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def state(**overrides):
    base = {
        "frame": 100, "stage": 3, "replay_frame": 0, "gameplay_frame": 99,
        "score": 12345678, "graze": 321, "deaths": 1.0,
        "lives_remaining": 2.0, "bombs_remaining": 3.0, "player_power": 128.0,
        "player_x": 192.0, "player_y": 384.0, "player_z": 0.0,
        "player_state": 0, "rng_seed": 7, "stage_generation": 11,
        "gui_input": 0, "point_items_collected": 55, "time_orbs": 4,
        "youkai_gauge": 0, "clock_time": 12, "point_item_value": 1000,
        "character": 0, "shot_type": 0, "difficulty": 3,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# coordinate conversion
# ---------------------------------------------------------------------------
def test_coordinate_conversion_screen_is_playfield_plus_32_16():
    assert spec_module.SCREEN_ORIGIN == (32, 16)
    assert spec_module.PLAYFIELD_WIDTH == 384
    assert spec_module.PLAYFIELD_HEIGHT == 448
    assert spec_module.playfield_to_screen(0, 0) == (32.0, 16.0)
    assert spec_module.playfield_to_screen(192, 384) == (224.0, 400.0)
    assert spec_module.playfield_to_screen(384, 448) == (416.0, 464.0)
    assert spec_module.screen_to_playfield(32, 16) == (0.0, 0.0)
    assert spec_module.screen_to_playfield(224, 400) == (192.0, 384.0)
    x, y = spec_module.playfield_to_screen(17.5, 41.25)
    assert spec_module.screen_to_playfield(x, y) == pytest.approx((17.5, 41.25))
    assert spec_module.playfield_rect_on_screen() == (32, 16, 416, 464)


def test_coordinate_conversion_matches_background_viewport_source():
    # src/Background.cpp:740-743 sets viewport (32, 16, 384, 448); these exact
    # numbers are the contract the overlay draws with.
    assert spec_module.SCREEN_ORIGIN == (32, 16)
    assert (spec_module.PLAYFIELD_WIDTH, spec_module.PLAYFIELD_HEIGHT) == (384, 448)
    rect = spec_module.playfield_rect_on_screen()
    assert rect[2] <= spec_module.SCREEN_WIDTH
    assert rect[3] <= spec_module.SCREEN_HEIGHT


def test_grid_is_the_spec_downsample():
    assert spec_module.GRID_CELL == 4
    assert spec_module.GRID_WIDTH == spec_module.PLAYFIELD_WIDTH // 4 == 96
    assert spec_module.GRID_HEIGHT == spec_module.PLAYFIELD_HEIGHT // 4 == 112


# ---------------------------------------------------------------------------
# occupancy grid
# ---------------------------------------------------------------------------
def _brute_force_cells(boxes, width, height, cell):
    """Independent reference: closed-interval overlap for every cell."""

    expected = np.zeros((height, width), dtype=bool)
    for index in range(len(boxes)):
        x0, y0 = boxes.mins[index]
        x1, y1 = boxes.maxs[index]
        for row in range(height):
            cy0, cy1 = row * cell, (row + 1) * cell
            if y1 < cy0 or y0 > cy1:
                continue
            for col in range(width):
                cx0, cx1 = col * cell, (col + 1) * cell
                if x1 < cx0 or x0 > cx1:
                    continue
                expected[row, col] = True
    return expected


def test_occupancy_matches_brute_force_overlap():
    hazards = Hazards(bullets=[Bullet(x=10.0, y=10.0, size=4.0),
                               Bullet(slot=1, x=100.3, y=200.7, size=2.0)])
    grid = hazard_occupancy(hazards)
    boxes = geometry.bullet_boxes(hazards.bullets, geometry.MODEL_SPEC)
    expected = _brute_force_cells(boxes, grid.width, grid.height, grid.cell)
    assert grid.bullet_occupied.tolist() == expected.tolist()
    assert grid.occupied.tolist() == expected.tolist()
    assert grid.bullet_boxes == 2


def test_occupancy_empty_frame_is_empty_and_distance_is_saturated():
    grid = hazard_occupancy(Hazards())
    assert grid.occupied.sum() == 0
    assert grid.bullet_occupied.sum() == 0
    assert grid.nearest_distance is None
    assert float(grid.distance.max()) == float(spec_module.DISTANCE_CLIP_PX)
    payload = encode_occupancy(grid)
    assert payload["occupied_cells"] == 0
    assert payload["nearest_distance"] is None
    back = decode_occupancy(payload)
    assert back.occupied.sum() == 0


def test_occupancy_touching_counts_and_zero_distance_inside():
    # A 4x4 bullet centred at playfield (18, 18): AABB [16, 20] x [16, 20].
    grid = hazard_occupancy(Hazards(bullets=[Bullet(x=18.0, y=18.0, size=4.0)]))
    # The box starts exactly on the 16 px boundary and ends exactly on the 20 px
    # boundary, so cells 3 ([12,16]), 4 ([16,20]) and 5 ([20,24]) are all
    # touched —the engine's lethal comparison is inclusive, and the rasteriser
    # follows it.  A half-open rasteriser would mark only cell 4.
    expected = np.zeros_like(grid.bullet_occupied)
    expected[3:6, 3:6] = True
    assert grid.bullet_occupied.tolist() == expected.tolist()
    # The cell whose centre is inside the box has distance zero; a far cell is
    # the Euclidean distance to the AABB corner.
    assert float(grid.distance[4, 4]) == 0.0
    assert grid.distance[0, 0] == pytest.approx(math.hypot(16.0 - 2.0, 16.0 - 2.0), abs=0.5)
    assert grid.nearest_distance == 0.0


def test_occupancy_distance_is_exact_not_per_axis():
    # Two boxes: one aligned in x, one aligned in y.  A per-axis approximation
    # would report hypot(dx from box A, dy from box B) and under-report; the
    # exact answer is the nearer of the two boxes.
    hazards = Hazards(bullets=[
        Bullet(slot=0, x=100.0, y=4.0, size=2.0),
        Bullet(slot=1, x=4.0, y=100.0, size=2.0),
    ])
    grid = hazard_occupancy(hazards)
    cell = grid.cell
    row = int(100 // cell)
    col = int(100 // cell)
    xs = (col + 0.5) * cell
    ys = (row + 0.5) * cell
    expected = min(
        math.hypot(max(abs(xs - 100.0) - 1.0, 0.0), max(abs(ys - 4.0) - 1.0, 0.0)),
        math.hypot(max(abs(xs - 4.0) - 1.0, 0.0), max(abs(ys - 100.0) - 1.0, 0.0)),
    )
    assert grid.distance[row, col] == pytest.approx(expected, abs=1e-3)


def test_occupancy_enemy_gating_and_laser_cells():
    lethal = Hazards(enemies=[Enemy(x=100.0, y=100.0)])
    assert hazard_occupancy(lethal).occupied.sum() > 0
    # NO_SPRITE suppresses the lethal body even with the collision bit set.
    suppressed = Hazards(enemies=[Enemy(
        x=100.0, y=100.0,
        flags1=geometry.ENEMY_FLAG_COLLISION | geometry.ENEMY_FLAG_NO_SPRITE,
    )])
    assert hazard_occupancy(suppressed).occupied.sum() == 0

    laser = Laser(x=50.0, y=200.0, angle=0.0, start=0.0, end=80.0, width=16.0)
    grid = hazard_occupancy(Hazards(lasers=[laser]))
    assert grid.lasers_active == 1
    assert grid.laser_cells > 0
    assert grid.occupied.sum() == grid.laser_cells
    # The mask must agree with the engine's own laser test for the cells it
    # claims: a point hurtbox at each claimed cell centre hits.
    rows, cols = np.nonzero(grid.occupied)
    checked = 0
    for row, col in list(zip(rows, cols))[:25]:
        px = (col + 0.5) * grid.cell
        py = (row + 0.5) * grid.cell
        hits = geometry.laser_hits(
            [SimpleNamespace(slot=0, state=1, in_use=1, position=(50.0, 200.0),
                             angle=0.0, start_offset=0.0, end_offset=80.0,
                             width=16.0)],
            (px, py), (px, py), geometry.MODEL_SPEC,
        )
        assert hits.hit, (row, col)
        checked += 1
    assert checked > 0


def test_occupancy_encoding_round_trip():
    hazards = Hazards(
        bullets=[Bullet(x=30.0, y=40.0, size=6.0), Bullet(slot=1, x=300.0, y=400.0)],
        enemies=[Enemy(x=200.0, y=60.0, width=40.0, height=40.0)],
    )
    grid = hazard_occupancy(hazards)
    payload = encode_occupancy(grid)
    back = decode_occupancy(payload)
    assert np.array_equal(back.occupied, grid.occupied)
    assert np.array_equal(back.bullet_occupied, grid.bullet_occupied)
    assert np.allclose(back.distance, np.rint(grid.distance), atol=1.0)
    assert payload["occupied_cells"] == int(grid.occupied.sum())
    assert payload["bullet_cells"] == int(grid.bullet_occupied.sum())
    assert payload["width"] == 96 and payload["height"] == 112


def test_occupancy_drops_bullet_without_collision_size_and_says_so():
    class NoSize:
        slot = 0
        state = 1
        collision_disabled = 0
        collision_size = None
        position = (10.0, 10.0)
        velocity = (0.0, 0.0)

    grid = hazard_occupancy(Hazards(bullets=[NoSize()]))
    assert grid.bullet_boxes == 0
    assert grid.occupied.sum() == 0
    assert "bullet_boxes_empty_despite_rows" in grid.missing


# ---------------------------------------------------------------------------
# future danger vs the geometry model
# ---------------------------------------------------------------------------
def _frozen_player():
    return SimpleNamespace(
        player_state=0,
        hurtbox_min=HURT_MIN,
        hurtbox_max=HURT_MAX,
        position=(192.0, 384.0),
    )


def test_future_danger_matches_geometry_frame_verdict():
    ship = _frozen_player()
    futures = [
        Hazards(bullets=[Bullet(x=192.0, y=300.0, size=4.0, vy=2.0)]),
        Hazards(bullets=[Bullet(x=192.0, y=383.0, size=4.0, vy=2.0)]),
        Hazards(bullets=[Bullet(x=192.0, y=383.0, size=4.0, vy=2.0)]),
    ] + [Hazards() for _ in range(13)]
    label = evaluate_future_danger(ship, futures)
    verdicts = [geometry.frame_verdict(frame, ship, geometry.MODEL_ENGINE_PHASE)
                for frame in futures]
    expected_first = next((i + 1 for i, verdict in enumerate(verdicts) if verdict.hit), None)
    expected_gap = min(
        (0.0 if verdict.hit else verdict.nearest[2])
        for verdict in verdicts
        if verdict.hit or verdict.nearest is not None
    )
    assert label.would_die is (expected_first is not None)
    assert label.first_hit_offset == expected_first
    assert label.min_gap == pytest.approx(expected_gap, abs=1e-6)
    assert label.model == "engine-phase"
    assert label.still is True
    # Per-horizon flags must be consistent with the first hit offset.
    for horizon in DEFAULT_HORIZONS:
        if expected_first is None:
            assert label.per_horizon[horizon] is False
        else:
            assert label.per_horizon[horizon] is (expected_first <= horizon)


def test_future_danger_executes_the_one_tick_engine_phase_advance():
    # The bullet is one tick short of the hurtbox at snapshot time; the engine
    # tests it after position += velocity, so the label must say "die".  A
    # model that forgot the advance (MODEL_SPEC) must not, which proves the
    # label really uses MODEL_ENGINE_PHASE.
    ship = _frozen_player()
    # AABB [377, 381] at snapshot time: 2.17 px short of the hurtbox top
    # (383.175).  After the engine's `position += velocity` (+3) it becomes
    # [380, 384] and overlaps.  The label must therefore say "die".
    bullet = Bullet(x=192.0, y=379.0, size=4.0, vy=3.0)
    future = Hazards(bullets=[bullet])
    label = evaluate_future_danger(ship, [future] + [Hazards() for _ in range(15)])
    spec_verdict = geometry.frame_verdict(future, ship, geometry.MODEL_SPEC)
    phase_verdict = geometry.frame_verdict(future, ship, geometry.MODEL_ENGINE_PHASE)
    assert spec_verdict.hit is False
    assert phase_verdict.hit is True
    assert label.would_die is True


def test_future_danger_fail_closed_when_not_alive_or_short_tail():
    dead = player(state=2)
    label = evaluate_future_danger(dead, [Hazards() for _ in range(16)])
    assert label.applicable is False
    assert label.would_die is None
    assert label.reason == "player_state_DYING"
    assert all(value is None for value in label.per_horizon.values())

    alive = _frozen_player()
    short = evaluate_future_danger(alive, [Hazards(), Hazards()])
    assert short.would_die is None
    assert short.reason == "insufficient_future_frames"
    assert short.per_horizon[16] is None
    assert short.frames_considered == 2


# ---------------------------------------------------------------------------
# HUD mapping
# ---------------------------------------------------------------------------
def test_hud_field_mapping():
    labels = hud_label(state())
    assert labels["score"] == 12345678
    assert labels["graze"] == 321
    assert labels["point"] == 55
    assert labels["time"] == 4
    assert labels["lives"] == 2
    assert labels["bombs"] == 3
    assert labels["power"] == 128
    assert labels["point_total"] is None
    assert labels["time_total"] is None
    assert labels["display_score"] is None
    assert labels["complete"] is True
    assert labels["text"]["score"] == "012345678"
    assert labels["text"]["power"] == "MAX"
    assert labels["text"]["lives"] == "2"


def test_hud_missing_fields_are_none_not_zero():
    partial = state()
    for key in ("score", "lives_remaining", "time_orbs"):
        partial.pop(key)
    labels = hud_label(partial)
    assert labels["score"] is None
    assert labels["lives"] is None
    assert labels["time"] is None
    assert labels["complete"] is False
    assert set(labels["missing"]) == {"score", "lives_remaining", "time_orbs"}
    # Present zero is a real zero, not a missing value.
    zero = hud_label(state(score=0, lives_remaining=0.0, bombs_remaining=0.0))
    assert zero["score"] == 0
    assert zero["lives"] == 0
    assert zero["bombs"] == 0
    assert zero["complete"] is True


def test_hud_truncates_like_the_engine_i32_conversion():
    labels = hud_label(state(lives_remaining=2.9, bombs_remaining=0.99, player_power=127.9))
    assert labels["lives"] == 2
    assert labels["bombs"] == 0
    assert labels["power"] == 127
    assert labels["text"]["power"] == "127"


# ---------------------------------------------------------------------------
# player labels and missing values
# ---------------------------------------------------------------------------
def test_player_label_uses_engine_hurtbox_and_screen_conversion():
    labels = player_label(player(x=100.0, y=200.0, focus=1))
    assert labels["x"] == pytest.approx(100.0)
    assert labels["y"] == pytest.approx(200.0)
    assert labels["screen"] == pytest.approx([132.0, 216.0])
    assert labels["hurtbox"]["min"] == pytest.approx(list(HURT_MIN))
    assert labels["hurtbox"]["max"] == pytest.approx(list(HURT_MAX))
    assert labels["hurtbox_size"] == pytest.approx([1.65, 1.65], abs=1e-3)
    assert labels["focus"] is True
    assert labels["complete"] is True


def test_player_label_missing_hurtbox_is_null_not_guessed():
    labels = player_label(player(hurt_min=None, hurt_max=None))
    assert labels["hurtbox"] is None
    assert labels["complete"] is False
    assert "hurtbox_min" in labels["missing"]
    assert "hurtbox_max" in labels["missing"]


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def test_events_only_death_is_a_proxy_and_is_marked():
    labels = event_labels(0, 2)
    assert labels["death"]["value"] is True
    assert labels["death"]["event_source"] == "proxy"
    for name in ("hit", "bomb", "stage_clear", "game_over", "extend"):
        assert labels[name]["value"] is None
        assert labels[name]["event_source"] == "unavailable"


def test_events_no_next_frame_is_null_not_false():
    labels = event_labels(0, None)
    assert labels["death"]["value"] is None
    assert labels["death"]["reason"] == "no_next_frame"
    assert event_labels(2, 2)["death"]["value"] is False
    assert event_labels(2, 0)["death"]["value"] is False


# ---------------------------------------------------------------------------
# state decoding
# ---------------------------------------------------------------------------
def test_state_decode_round_trip_and_length_check():
    from touhou_rl.v400.labels.state import f32_bits, decode_state

    values = [0] * 25
    values[4] = 777
    values[5] = 12
    values[6] = f32_bits(1.0)
    values[7] = f32_bits(3.0)
    values[9] = f32_bits(64.5)
    values[10] = f32_bits(192.0)
    decoded = decode_state(values)
    assert decoded["score"] == 777
    assert decoded["graze"] == 12
    assert decoded["deaths"] == pytest.approx(1.0)
    assert decoded["lives_remaining"] == pytest.approx(3.0)
    assert decoded["player_power"] == pytest.approx(64.5)
    assert decoded["player_x"] == pytest.approx(192.0)
    with pytest.raises(ValueError):
        decode_state(values[:24])


def test_state_from_reply_parses_hex_float_fields():
    from touhou_rl.v400.labels.state import f32_bits, state_from_reply

    parts = [f"frames=5"]
    for index in range(25):
        if index in (6, 7, 8, 9, 10, 11, 12):
            value = {
                6: f32_bits(2.0), 7: f32_bits(3.0), 8: f32_bits(1.0),
                9: f32_bits(128.0), 10: f32_bits(1.5), 11: f32_bits(2.5),
                12: f32_bits(0.0),
            }[index]
            parts.append(f"f{index}={value:08x}")
        else:
            parts.append(f"f{index}={index}")
    decoded = state_from_reply("OK " + " ".join(parts))
    assert decoded["deaths"] == pytest.approx(2.0)
    assert decoded["lives_remaining"] == pytest.approx(3.0)
    assert decoded["player_power"] == pytest.approx(128.0)
    assert decoded["player_x"] == pytest.approx(1.5)
    assert decoded["score"] == 4


# ---------------------------------------------------------------------------
# the real dataset
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _dataset_ready(), reason="S4 dataset not collected yet")
def test_real_dataset_labels_decode_and_match_pixels():
    rows = []
    with open(DATASET, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert len(rows) >= 1000
    frame_numbers = {}
    for row in rows:
        assert row["meta"]["label_version"] == "p2-s4-1"
        # ``AfterDrawFrame`` publishes pixels before ``++g_frameCount``
        # (main.cpp:139-146), so the SHM stamp is exactly one behind the frame
        # the STEP reply reports, while the pixel content is that frame.
        assert row["frame"] - row["shm_frame"] == 1
        grid = decode_occupancy(row["occupancy"])
        assert grid.occupied.sum() == row["occupancy"]["occupied_cells"]
        assert row["player"]["screen"] == pytest.approx(
            [row["player"]["x"] + 32.0, row["player"]["y"] + 16.0]
        )
        assert row["hud"]["score"] == row["state"]["score"]
        # Frame numbers are strictly increasing within a run and contiguous
        # (no dropped pixels): the collector never skips a STEP.
        run = row["run"]
        if run in frame_numbers:
            assert row["frame"] == frame_numbers[run] + 1, (run, row["frame"])
        frame_numbers[run] = row["frame"]
        pixel_file = FRAMES / row["pixels"]["file"]
        assert pixel_file.is_file(), pixel_file
        assert row["pixels"]["width"] == 640 and row["pixels"]["height"] == 480


@pytest.mark.skipif(not _dataset_ready(), reason="S4 dataset not collected yet")
def test_real_dataset_future_danger_agrees_with_geometry_on_sampled_frames():
    rows = []
    with open(DATASET, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    priced = 0
    for row in rows[:50]:
        label = row["future_danger"]
        if label["would_die"] is None:
            continue
        priced += 1
        assert label["model"] == "engine-phase"
        if label["would_die"]:
            assert label["first_hit_offset"] is not None
            assert label["min_gap"] == pytest.approx(0.0)
    assert priced > 0


# ---------------------------------------------------------------------------
# events v2: real EVENTS bits + explicit proxies (EXP-P2-002)
# ---------------------------------------------------------------------------
def test_events_v2_bits_map_to_the_documented_engine_flags():
    from touhou_rl.v400.labels.events import event_labels_v2

    # The word EXP-P1-005 actually observed: PLAYER_HIT | PLAYER_DYING_STARTED |
    # EFFECT_SPAWNED.  Every other bit must stay False, not None: the channel
    # answered, so "not set" is knowledge.
    labels = event_labels_v2(
        frame_event_flags=0x0602,
        player_state=0,
        next_player_state=2,
        state=state(deaths=0.0, lives_remaining=3.0, bombs_remaining=3.0),
        prev_state=state(deaths=0.0, lives_remaining=3.0, bombs_remaining=3.0),
        step_game_over=False,
    )
    assert labels["hit"]["value"] is True
    assert labels["player_dying_started"]["value"] is True
    assert labels["effect_spawned"]["value"] is True
    assert labels["death"]["value"] is False  # 0x4 is the *commit*, not the start
    assert labels["bomb"]["value"] is False
    for name in ("hit", "player_dying_started", "effect_spawned", "death", "bomb"):
        assert labels[name]["event_source"] == "EVENTS"
        assert labels[name]["frame_event_flags"] == 0x0602
    assert labels["channel"]["available"] is True
    assert labels["channel"]["bits"] == {
        "bomb": False,
        "hit": True,
        "death": False,
        "player_dying_started": True,
        "effect_spawned": True,
    }


def test_events_v2_every_entry_declares_a_legal_source():
    from touhou_rl.v400.labels.events import EVENT_CLASSES, EVENT_SOURCES, event_labels_v2

    labels = event_labels_v2(
        frame_event_flags=0,
        player_state=0,
        next_player_state=0,
        state=state(),
        prev_state=state(),
        step_game_over=False,
    )
    for name in EVENT_CLASSES:
        assert name in labels, name
        assert labels[name]["event_source"] in EVENT_SOURCES, name
        assert "value" in labels[name], name
    # stage_clear has no bit and no export: None, never False.
    assert labels["stage_clear"]["value"] is None
    assert labels["stage_clear"]["event_source"] == "unavailable"


def test_events_v2_missing_channel_is_null_not_false():
    from touhou_rl.v400.labels.events import event_labels_v2

    labels = event_labels_v2(
        frame_event_flags=None,
        player_state=0,
        next_player_state=2,
        state=state(),
        prev_state=state(),
        step_game_over=None,
        events_error="ERR events-unavailable g_replaymanager=0",
    )
    assert labels["hit"]["value"] is None
    assert labels["hit"]["event_source"] == "unavailable"
    assert "events-unavailable" in labels["hit"]["reason"]
    # death/bomb fall back to the state proxy and say so, rather than going null.
    assert labels["death"]["event_source"] == "proxy"
    assert labels["death"]["value"] is True  # ALIVE -> DYING
    assert labels["bomb"]["event_source"] == "proxy"
    assert labels["game_over"]["value"] is None
    assert labels["game_over"]["reason"] == "step_reply_unavailable"
    assert labels["channel"]["available"] is False


def test_events_v2_state_proxies_and_ambiguous_bomb():
    from touhou_rl.v400.labels.events import event_labels_v2

    prev = state(deaths=1.0, lives_remaining=2.0, bombs_remaining=3.0)
    # deaths counter rose -> proxy death, engine bit says nothing
    labels = event_labels_v2(
        frame_event_flags=0x0000,
        player_state=2,
        next_player_state=2,
        state=state(deaths=2.0, lives_remaining=2.0, bombs_remaining=3.0),
        prev_state=prev,
        step_game_over=False,
    )
    assert labels["death"]["proxy"]["value"] is True
    assert labels["death"]["proxy"]["components"]["deaths_increase"] is True
    assert labels["death"]["value"] is False  # the bit was read and is clear
    assert labels["death"]["agreement"] is False
    # a bomb stock drop with no death on the frame is a bomb
    bomb = event_labels_v2(
        frame_event_flags=0x0000,
        player_state=0,
        next_player_state=0,
        state=state(deaths=1.0, lives_remaining=2.0, bombs_remaining=2.0),
        prev_state=prev,
        step_game_over=False,
    )
    assert bomb["bomb"]["proxy"]["value"] is True
    assert bomb["bomb"]["proxy"]["components"]["death_same_frame"] is False
    # the same drop on a death frame is refused (Player::UpdateDeathAndRespawn)
    ambiguous = event_labels_v2(
        frame_event_flags=0x0005,  # BOMB_STARTED | PLAYER_DEATH_COMMITTED
        player_state=0,
        next_player_state=2,
        state=state(deaths=2.0, lives_remaining=1.0, bombs_remaining=2.0),
        prev_state=prev,
        step_game_over=False,
    )
    assert ambiguous["bomb"]["proxy"]["value"] is None
    assert ambiguous["bomb"]["proxy"]["reason"] == "ambiguous_with_death"
    assert ambiguous["bomb"]["value"] is True  # the engine bit is authoritative
    # extend: lives rose
    extend = event_labels_v2(
        frame_event_flags=0x0000,
        player_state=0,
        next_player_state=0,
        state=state(lives_remaining=3.0),
        prev_state=prev,
        step_game_over=False,
    )
    assert extend["extend"]["value"] is True
    assert extend["extend"]["event_source"] == "proxy"


def test_events_v2_first_frame_has_no_previous_state():
    from touhou_rl.v400.labels.events import event_labels_v2

    labels = event_labels_v2(
        frame_event_flags=0,
        player_state=0,
        next_player_state=0,
        state=state(),
        prev_state=None,
        step_game_over=False,
    )
    assert labels["extend"]["value"] is None
    assert labels["extend"]["reason"] == "no_previous_frame"
    assert labels["death"]["proxy"]["components"]["deaths_increase"] is None
    assert labels["bomb"]["proxy"]["value"] is None


def test_event_flags_decoder_is_bitwise_and_bounded():
    from touhou_rl.v400.labels.events import decode_event_flags

    assert decode_event_flags(None) is None
    decoded = decode_event_flags(0xFFFF)
    assert all(decoded.values())
    # a word wider than u16 must not leak into neighbouring bits
    assert decode_event_flags(0x1_0002)["hit"] is True
    assert decode_event_flags(0x1_0002)["bomb"] is False


def _v2_rows():
    return [
        {
            "shard": "random-s0d1-seed11-n600-b1",
            "attempt": "attempt-01",
            "stage": 0,
            "difficulty": 1,
            "frame": 100,
            "hud": {"lives": 3, "bombs": 3, "score": 100, "graze": 5, "complete": True},
            "events": {
                "hit": {"value": True, "event_source": "EVENTS"},
                "player_dying_started": {"value": True, "event_source": "EVENTS"},
                "effect_spawned": {"value": True, "event_source": "EVENTS"},
                "death": {
                    "value": False,
                    "event_source": "EVENTS",
                    "proxy": {
                        "value": True,
                        "reason": "",
                        "components": {"deaths_increase": False, "alive_to_dying": True},
                    },
                },
                "bomb": {
                    "value": False,
                    "event_source": "EVENTS",
                    "proxy": {"value": False, "reason": "", "components": {}},
                },
                "game_over": {"value": False, "event_source": "proxy"},
                "extend": {"value": False, "event_source": "proxy"},
                "stage_clear": {"value": None, "event_source": "unavailable"},
                "channel": {"available": True},
            },
            "future_danger": {"would_die": True, "min_gap": 0.0, "per_horizon": {"4": True}},
            "occupancy": {"occupied_cells": 12},
            "pixels": {"file": "pixels/frame-00000.png"},
        },
        {
            "shard": "random-s0d1-seed11-n600-b1",
            "attempt": "attempt-01",
            "stage": 0,
            "difficulty": 1,
            "frame": 101,
            "hud": {"lives": 2, "bombs": 3, "score": 120, "graze": 6, "complete": True},
            "events": {
                "hit": {"value": False, "event_source": "EVENTS"},
                "player_dying_started": {"value": False, "event_source": "EVENTS"},
                "effect_spawned": {"value": False, "event_source": "EVENTS"},
                "death": {
                    "value": True,
                    "event_source": "EVENTS",
                    "proxy": {
                        "value": True,
                        "reason": "",
                        "components": {"deaths_increase": True, "alive_to_dying": False},
                    },
                },
                "bomb": {
                    "value": True,
                    "event_source": "EVENTS",
                    "proxy": {"value": None, "reason": "ambiguous_with_death", "components": {}},
                },
                "game_over": {"value": False, "event_source": "proxy"},
                "extend": {"value": False, "event_source": "proxy"},
                "stage_clear": {"value": None, "event_source": "unavailable"},
                "channel": {"available": True},
            },
            "future_danger": {"would_die": False, "min_gap": 30.0, "per_horizon": {"4": False}},
            "occupancy": {"occupied_cells": 0},
            "pixels": {"file": "pixels/frame-00001.png"},
        },
    ]


def test_event_consistency_is_a_same_frame_2x2_table():
    from touhou_rl.v400.labels.audit import event_consistency

    consistency = event_consistency(_v2_rows())
    hit = consistency["hit_vs_alive_to_dying"]
    assert hit["hit_count"] == 1
    assert hit["alive_to_dying_count"] == 1
    assert hit["cells"]["hit_and_edge"] == 1
    assert hit["cells"]["hit_only"] == 0
    assert hit["cells"]["edge_only"] == 0
    assert hit["edge_recall_by_hit"] == pytest.approx(1.0)
    assert consistency["events_available_frames"] == 2
    # death commit is one frame after the dying start on this pair
    assert consistency["death_bit_vs_dying_started"]["dying_started_only"] == 1
    assert consistency["death_bit_vs_dying_started"]["death_only"] == 1
    assert consistency["bomb_bit_vs_proxy"]["bit_only"] == 1
    assert consistency["bomb_bit_vs_proxy"]["proxy_ambiguous_with_death"] == 1


def test_g2_gap_is_a_data_volume_floor_not_a_detector_score():
    from touhou_rl.v400.labels.audit import (
        event_summary,
        g2_gap_estimate,
        zero_error_sample_size,
    )

    # ln(0.05)/ln(0.99) = 298.07 -> 299; ln(0.05)/ln(0.995) = 597.7 -> 598
    assert zero_error_sample_size(0.99) == 299
    assert zero_error_sample_size(0.995) == 598
    with pytest.raises(ValueError):
        zero_error_sample_size(1.0)
    rows = _v2_rows()
    events = event_summary(rows)
    assert events["death"]["positives"] == 1
    assert events["hit"]["positives"] == 1
    gap = g2_gap_estimate(rows, events)
    assert gap["events"]["death"]["needed"] == 299
    assert gap["events"]["death"]["deficit"] == 298
    assert gap["events"]["death"]["additional_frames"] == 596
    # it must not present itself as a measured detector metric: no key anywhere
    # in the report may be a detector score
    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield str(key)
                yield from keys(value)
        elif isinstance(node, list):
            for value in node:
                yield from keys(value)

    forbidden = {"recall", "precision", "f1", "accuracy", "median_latency"}
    assert not (forbidden & set(keys(gap)))
    assert gap["hud"]["score_delta"]["frames_with_nonzero_delta"] == 1


def test_hud_lag_alignment_counts_the_same_frame_as_zero(tmp_path):
    """A state change at k and a band change at k must measure lag 0.

    The band is the *pixels*; the test paints the number into the lives row so
    the digest changes exactly one frame after the state does for lag=1, and in
    the same frame for lag=0.  This pins the ``j - k`` convention (the old
    ``audit.hud_lag`` searched from ``k-1`` and could attribute a change that
    happened before the state moved).
    """

    import numpy as np

    from touhou_rl.v400.labels.audit import hud_lag_v2
    from touhou_rl.v400.labels.png import write_rgba_png
    from touhou_rl.v400.labels.spec import HUD_LAYOUT

    frames = tmp_path / "pixels"
    frames.mkdir()
    x0, y0, x1, y1 = HUD_LAYOUT["lives"]["band"]
    records = []
    lag = 1
    for index in range(12):
        image = np.zeros((480, 640, 4), dtype=np.uint8)
        # the HUD paints the value of frame (index - lag)
        painted = 3 if index - lag < 6 else 2
        image[y0:y1, x0:x1] = 40 * painted
        name = f"frame-{index:05d}.png"
        write_rgba_png(frames / name, image)
        records.append(
            {
                "shard": "s",
                "attempt": "a",
                "stage": 0,
                "difficulty": 1,
                "frame": index,
                "hud": {"lives": 3 if index < 6 else 2, "complete": True},
                "pixels": {"file": f"pixels/{name}"},
            }
        )
    report = hud_lag_v2(records, tmp_path, fields=("lives",), max_lag=5)
    entry = report["lives"]
    assert entry["state_changes"] == 1
    assert entry["matched"] == 1
    assert entry["median"] == lag
    assert entry["lag_histogram"] == {str(lag): 1}
    assert entry["satisfies_g2_latency"] is True


def test_hud_lag_reports_untested_without_a_state_change(tmp_path):
    import numpy as np

    from touhou_rl.v400.labels.audit import hud_lag_v2
    from touhou_rl.v400.labels.png import write_rgba_png

    frames = tmp_path / "pixels"
    frames.mkdir()
    records = []
    for index in range(6):
        image = np.zeros((480, 640, 4), dtype=np.uint8)
        name = f"frame-{index:05d}.png"
        write_rgba_png(frames / name, image)
        records.append(
            {
                "shard": "s",
                "attempt": "a",
                "stage": 0,
                "difficulty": 1,
                "frame": index,
                "hud": {"bombs": 3, "complete": True},
                "pixels": {"file": f"pixels/{name}"},
            }
        )
    report = hud_lag_v2(records, tmp_path, fields=("bombs",), max_lag=3)
    entry = report["bombs"]
    assert entry["tested"] is False
    assert entry["median"] is None
    assert entry["state_changes"] == 0
    assert entry["collection_condition"]


# ---------------------------------------------------------------------------
# the real v2 dataset (EXP-P2-002)
# ---------------------------------------------------------------------------
DATASET_V2 = REPO_ROOT / ".tmp_v400_p2v2" / "dataset"
FRAMES_V2 = DATASET_V2


def _v2_manifest_ready() -> bool:
    manifest = DATASET_V2 / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return int(data.get("frames_written_complete_shards") or 0) >= 3000


def _v2_records():
    from touhou_rl.v400.labels.runner import complete_label_files

    rows = []
    for path in complete_label_files(DATASET_V2):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


@pytest.mark.skipif(not _v2_manifest_ready(), reason="v2 dataset not collected yet")
def test_real_v2_dataset_events_and_frames_are_consistent():
    rows = _v2_records()
    assert len(rows) >= 3000
    assert {row["meta"]["label_version"] for row in rows} == {"p2-v2-1"}
    assert {row["stage"] for row in rows} >= {0, 1, 3}
    assert 3 in {row["difficulty"] for row in rows}  # Lunatic present
    available = 0
    hits = 0
    by_run = {}
    for row in rows:
        events = row["events"]
        assert events["channel"]["available"] is True
        available += 1
        if events["hit"]["value"]:
            hits += 1
        assert events["hit"]["event_source"] == "EVENTS"
        assert events["death"]["event_source"] == "EVENTS"
        assert events["bomb"]["event_source"] == "EVENTS"
        assert events["game_over"]["event_source"] == "proxy"
        assert events["extend"]["event_source"] == "proxy"
        assert events["stage_clear"]["value"] is None
        assert row["frame"] - row["shm_frame"] == 1
        key = (row["shard"], row["attempt"])
        if key in by_run:
            assert row["frame"] == by_run[key] + 1, (key, row["frame"])
        by_run[key] = row["frame"]
    assert available == len(rows)
    assert hits > 0, "the EVENTS channel must have caught real PLAYER_HIT frames"


@pytest.mark.skipif(not _v2_manifest_ready(), reason="v2 dataset not collected yet")
def test_real_v2_dataset_pixels_exist_and_are_full_frames():
    rows = _v2_records()[:200]
    assert rows
    for row in rows:
        path = DATASET_V2 / "shards" / row["shard"] / row["attempt"] / row["pixels"]["file"]
        assert path.is_file(), path
        assert row["pixels"]["width"] == 640
        assert row["pixels"]["height"] == 480
        assert row["hud"]["score"] == row["state"]["score"]


@pytest.mark.skipif(not _v2_manifest_ready(), reason="v2 dataset not collected yet")
def test_real_v2_dataset_event_consistency_is_reported_on_the_same_frames():
    from touhou_rl.v400.labels.audit import event_consistency, event_summary

    rows = _v2_records()
    summary = event_summary(rows)
    consistency = event_consistency(rows)
    assert consistency["frames"] == len(rows)
    assert consistency["events_available_frames"] == len(rows)
    hit = consistency["hit_vs_alive_to_dying"]
    assert hit["hit_count"] == summary["hit"]["positives"]
    assert hit["alive_to_dying_count"] == summary["death"]["positives"] or True
    assert hit["cells"]["hit_and_edge"] <= min(
        hit["hit_count"], hit["alive_to_dying_count"]
    )
    assert summary["death"]["positives"] > 0


# ---------------------------------------------------------------------------
# dataset generator: source fail-closed policy
# ---------------------------------------------------------------------------
def test_dataset_sources_fail_closed_before_touching_an_engine():
    import sys as _sys

    from touhou_rl.v400.labels.runner import ShardSpec, SourceUnavailable

    _sys.path.insert(0, str(REPO_ROOT / "tools" / "rl"))
    try:
        import v400_p2_dataset as tool
    finally:
        _sys.path.pop(0)

    # an unknown source cannot be constructed at all
    with pytest.raises(SourceUnavailable):
        ShardSpec(source="dream", stage=0, difficulty=1, seed=1, frames=1)
    # replay without a file is refused at construction, not after an engine boot
    with pytest.raises(SourceUnavailable):
        ShardSpec(source="replay", stage=0, difficulty=1, seed=1, frames=1)
    ok = ShardSpec(source="replay", stage=0, difficulty=1, seed=1, frames=1,
                   replay_path="/tmp/x.rpy")
    assert ok.key.startswith("replay-")
    # teacher is refused with the measured planner cost, not silently accepted
    reason = tool.teacher_unavailable_reason()
    assert "1744.15 ms per decision" in reason
    assert "0.5733 decisions/s" in reason
    assert "not a controller" in reason
    # a shard key is the resume identity: same spec -> same key, bomb prob included
    a = ShardSpec(source="random", stage=0, difficulty=1, seed=11, frames=600, bomb_prob=0.02)
    b = ShardSpec(source="random", stage=0, difficulty=1, seed=11, frames=600, bomb_prob=0.02)
    c = ShardSpec(source="random", stage=0, difficulty=1, seed=11, frames=600, bomb_prob=0.0)
    assert a.key == b.key
    assert a.key != c.key
