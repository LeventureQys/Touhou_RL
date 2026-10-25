"""Unit tests for bullet tracking and spawn attribution (v3.6.0 label groundwork).

The decisive test is ``test_spawn_rate_is_identifiable_only_with_slot_ids``: it
encodes the measurement that killed the position-matching path.  A survivor whose
motion is not exactly ``pos + vel * frameskip`` (a decelerating bullet, a polar
acceleration, a one-frame read skew) is indistinguishable from a fresh spawn, and
the inferred spawn count then depends on the tolerance instead of converging.
"""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl.bullet_track import (
    attribute_spawns,
    match_bullets,
    spawns_by_slot,
    track_episode,
)


# ---------------------------------------------------------------------------
# slot-based spawn detection (exact)
# ---------------------------------------------------------------------------


def test_spawns_by_slot_marks_only_new_slots():
    prev = np.array([1, 2, 3], dtype=np.int64)
    cur = np.array([2, 3, 4], dtype=np.int64)
    assert spawns_by_slot(prev, cur).tolist() == [False, False, True]


def test_spawns_by_slot_counts_slot_reuse_as_a_spawn():
    # Slot 7 despawned and was reused by a different bullet: still a spawn.
    prev = np.array([5, 6, 7], dtype=np.int64)
    cur = np.array([5, 6, 7], dtype=np.int64)
    assert not spawns_by_slot(prev, cur).any()
    # But if the pool went empty and slot 7 came back, the empty snapshot is what
    # matters - an empty previous pool means everything is new.
    assert spawns_by_slot(np.array([], dtype=np.int64), cur).all()


def test_spawns_by_slot_first_decision_is_all_new():
    assert spawns_by_slot(np.array([], dtype=np.int64), np.array([0, 1])).all()


# ---------------------------------------------------------------------------
# position matching (the fallback, and why it is not identifiable)
# ---------------------------------------------------------------------------


def test_match_bullets_follows_linear_motion():
    prev_pos = np.array([[100.0, 100.0], [200.0, 50.0]], dtype=np.float32)
    prev_vel = np.array([[0.0, 3.0], [-2.0, 0.0]], dtype=np.float32)
    prev_half = np.array([2.0, 4.0], dtype=np.float32)
    cur_pos = prev_pos + prev_vel * 2.0
    cur_half = prev_half.copy()
    assigned, new_mask = match_bullets(prev_pos, prev_vel, prev_half, cur_pos, cur_half, frameskip=2)
    assert assigned.tolist() == [0, 1]
    assert not new_mask.any()


def test_match_bullets_sees_a_spawn_next_to_a_survivor():
    prev_pos = np.array([[100.0, 100.0]], dtype=np.float32)
    prev_vel = np.array([[0.0, 3.0]], dtype=np.float32)
    prev_half = np.array([2.0], dtype=np.float32)
    cur_pos = np.array([[100.0, 106.0], [140.0, 40.0]], dtype=np.float32)
    cur_half = np.array([2.0, 2.0], dtype=np.float32)
    assigned, new_mask = match_bullets(prev_pos, prev_vel, prev_half, cur_pos, cur_half, frameskip=2)
    assert assigned[0] == 0
    assert new_mask.tolist() == [False, True]


def test_match_bullets_rejects_a_half_extent_change():
    prev_pos = np.array([[100.0, 100.0]], dtype=np.float32)
    prev_vel = np.zeros((1, 2), dtype=np.float32)
    prev_half = np.array([2.0], dtype=np.float32)
    cur_pos = np.array([[100.0, 100.0]], dtype=np.float32)
    cur_half = np.array([8.0], dtype=np.float32)
    _assigned, new_mask = match_bullets(prev_pos, prev_vel, prev_half, cur_pos, cur_half)
    assert new_mask.tolist() == [True]


def test_spawn_rate_is_identifiable_only_with_slot_ids():
    """A non-linear survivor inflates the position-matched spawn count.

    Two bullets survive every decision, but one of them decelerates, so
    ``pos + vel * frameskip`` overshoots it.  With a tight tolerance it is
    reported as a spawn; with a loose one it is not.  Slot ids settle it.
    """
    truth_spawns = 0  # nothing actually spawns
    frames = 12
    positions, offsets, slots = [], [0], []
    for step in range(frames):
        # Bullet A: linear.  Bullet B: decelerating (moves 2 px less each step).
        a = np.array([100.0, 100.0 + 3.0 * step], dtype=np.float32)
        b = np.array([200.0, 100.0 + max(0.0, 6.0 * step - 0.5 * step * step)], dtype=np.float32)
        positions.append(np.stack([a, b]))
        slots.append(np.array([0, 1], dtype=np.int32))
        offsets.append(offsets[-1] + 2)
    positions = np.stack(positions)
    speeds = np.zeros_like(positions)
    speeds[:, 0] = [0.0, 3.0]
    speeds[:, 1] = [0.0, 6.0]
    halves = np.full(positions.shape[:2], 2.0, dtype=np.float32)
    base = {
        "positions": positions.reshape(-1, 2),
        "velocities": speeds.reshape(-1, 2),
        "half": halves.reshape(-1),
        "offsets": np.asarray(offsets),
    }
    records = [{"enemies": []} for _ in range(frames)]

    tight = track_episode(base, records, tolerance=0.5)
    loose = track_episode(base, records, tolerance=8.0)
    assert tight["match_method"] == "position"
    assert tight["spawns"] > loose["spawns"]  # tolerance changes the answer
    assert loose["spawns"] > truth_spawns

    with_slots = dict(base)
    with_slots["slots"] = np.stack(slots).reshape(-1)
    exact = track_episode(with_slots, records, tolerance=0.5)
    assert exact["match_method"] == "slot_id"
    assert exact["spawns"] == truth_spawns


# ---------------------------------------------------------------------------
# attribution
# ---------------------------------------------------------------------------


def test_attribute_spawns_picks_the_nearest_enemy_within_radius():
    spawns = np.array([[100.0, 100.0], [300.0, 200.0], [500.0, 500.0]], dtype=np.float32)
    enemies = np.array([[105.0, 100.0], [290.0, 200.0]], dtype=np.float32)
    info = attribute_spawns(spawns, enemies, radius=32.0)
    assert info["counts"].tolist() == [1, 1]
    assert info["unattributed"] == 1
    assert info["mean_distance"] is not None


def test_attribute_spawns_radius_is_respected():
    spawns = np.array([[100.0, 100.0]], dtype=np.float32)
    enemies = np.array([[100.0, 140.0]], dtype=np.float32)
    assert attribute_spawns(spawns, enemies, radius=32.0)["unattributed"] == 1
    assert attribute_spawns(spawns, enemies, radius=64.0)["unattributed"] == 0


def test_attribute_spawns_with_no_enemies_or_no_spawns():
    spawns = np.array([[100.0, 100.0]], dtype=np.float32)
    enemies = np.zeros((0, 2), dtype=np.float32)
    info = attribute_spawns(spawns, enemies)
    assert info["unattributed"] == 1 and info["counts"].size == 0
    info = attribute_spawns(np.zeros((0, 2)), np.array([[1.0, 1.0]], dtype=np.float32))
    assert info["unattributed"] == 0 and info["counts"].tolist() == [0]


def test_track_episode_reports_the_three_way_split():
    frames = 6
    positions, offsets = [], [0]
    for step in range(frames):
        # One survivor, plus one new bullet each decision right next to an enemy.
        row = [[100.0, 100.0 + step], [300.0, 80.0]]
        positions.append(np.asarray(row, dtype=np.float32))
        offsets.append(offsets[-1] + 2)
    positions = np.stack(positions)
    base = {
        "positions": positions.reshape(-1, 2),
        "velocities": np.tile(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32), (frames, 1)),
        "half": np.full(frames * 2, 2.0, dtype=np.float32),
        "offsets": np.asarray(offsets),
        "slots": np.tile(np.array([0, 1], dtype=np.int32), frames),
    }
    records = [{"enemies": [{"x": 300.0, "y": 80.0}]} for _ in range(frames)]
    stats = track_episode(base, records)
    assert stats["match_method"] == "slot_id"
    assert stats["spawns"] == 0  # both slots persist
    assert stats["attribution_rate"] is None  # nothing spawned, so no denominator
    assert stats["unexplained"] == 0
