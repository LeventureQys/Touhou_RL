"""Test for the zeroed phantom pool entry in BulletField.active_mask (v3.6.0)."""

from __future__ import annotations

import struct

import numpy as np

from touhou_rl import spec
from touhou_rl.state import BulletField


def _field(slots: list[tuple[float, float, float, float, int, int]]) -> BulletField:
    """Build a tiny BulletField from (x, y, vx, vy, state, disabled)."""
    n = len(slots)
    position = np.zeros((n, 2), dtype=np.float32)
    velocity = np.zeros((n, 2), dtype=np.float32)
    state = np.zeros((n,), dtype=np.uint16)
    disabled = np.zeros((n,), dtype=np.int8)
    half = np.full((n, 2), 2.0, dtype=np.float32)
    for index, (x, y, vx, vy, st, dis) in enumerate(slots):
        position[index] = (x, y)
        velocity[index] = (vx, vy)
        state[index] = st
        disabled[index] = dis
    return BulletField(
        raw=np.zeros((n, 1), dtype=np.uint8),
        position=position,
        velocity=velocity,
        state=state,
        disabled=disabled,
        half_size=half,
    )


def test_zeroed_pool_entry_is_not_active():
    """The pool keeps one (0,0)/(0,0) entry whose state is neither unused nor despawning."""
    field = _field(
        [
            (0.0, 0.0, 0.0, 0.0, 7, 0),  # the phantom
            (192.0, 300.0, 0.0, 2.0, 1, 0),  # a real bullet
        ]
    )
    mask = field.active_mask()
    assert mask.tolist() == [False, True]


def test_real_bullet_at_origin_edge_is_still_kept():
    """A bullet inside the playfield must never be filtered, even at the left/top edge."""
    field = _field([(spec.PLAYFIELD_X + 0.5, spec.PLAYFIELD_Y + 0.5, 0.0, 2.0, 1, 0)])
    assert field.active_mask().tolist() == [True]


def test_unused_and_despawning_and_disabled_are_filtered():
    field = _field(
        [
            (100.0, 100.0, 0.0, 2.0, spec.BULLET_UNUSED, 0),
            (100.0, 100.0, 0.0, 2.0, spec.BULLET_DESPAWNING, 0),
            (100.0, 100.0, 0.0, 2.0, 1, 1),
            (100.0, 100.0, 0.0, 2.0, 1, 0),
        ]
    )
    assert field.active_mask().tolist() == [False, False, False, True]


def test_zero_position_but_moving_is_kept():
    """Only *both* zero position and zero velocity identify the phantom."""
    field = _field([(0.0, 0.0, 0.0, -3.0, 1, 0)])
    assert field.active_mask().tolist() == [True]
