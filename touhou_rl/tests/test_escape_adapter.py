"""Adapter tests: the same-frame proof, explicit unknowns, and the settle model.

These run with a scripted fake reader.  No game, no memory, no ambiguity about
what "acceptable" means: a snapshot is only usable when the two reads that
bracket the bullet pool agree on ``gameplay_frame``.
"""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl.escape import adapter, contracts, geometry


class FakeState:
    def __init__(
        self,
        frame: int,
        *,
        x: float = 192.0,
        y: float = 384.0,
        vx: float = 0.0,
        vy: float = 0.0,
        focus: int = 0,
        state: int = 0,
        counter: int = 0,
        half: float = 0.825,
    ) -> None:
        self.gameplay_frame = frame
        self.player_x = x
        self.player_y = y
        self.player_vx = vx
        self.player_vy = vy
        self.focus_mode = focus
        self.player_state = state
        self.active_bullets = counter
        self.hurtbox_half_x = half
        self.hurtbox_half_y = half


class FakeField:
    def __init__(self, positions, velocities=None, halves=None, active=None) -> None:
        self.position = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
        count = self.position.shape[0]
        self.velocity = (
            np.asarray(velocities, dtype=np.float32).reshape(-1, 2)
            if velocities is not None
            else np.zeros((count, 2), dtype=np.float32)
        )
        self.half_size = (
            np.asarray(halves, dtype=np.float32).reshape(-1, 2)
            if halves is not None
            else np.full((count, 2), 2.0, dtype=np.float32)
        )
        self.state = np.ones(count, dtype=np.uint16)
        self.disabled = np.zeros(count, dtype=np.int8)
        self._active = np.ones(count, dtype=bool) if active is None else np.asarray(active, dtype=bool)

    def active_mask(self):
        return self._active


class FakeReader:
    def __init__(self, states, field=None, error: bool = False) -> None:
        self.states = list(states)
        self.field = field
        self.error = error
        self.reads = 0

    def read_state(self, with_bullets: bool = False, with_title: bool = False):
        state = self.states[min(self.reads, len(self.states) - 1)]
        self.reads += 1
        return state

    def read_bullets(self):
        if self.error:
            raise OSError("pool read failed")
        return self.field


def test_stable_snapshot_is_accepted_on_first_attempt():
    reader = FakeReader([FakeState(10), FakeState(10)], FakeField([(200.0, 300.0)], [(0.0, 3.0)]))
    frame = adapter.read_synced(reader, framerate_multiplier=1.0)
    assert frame.stable and frame.attempts == 1
    assert frame.frame_id == 10
    assert frame.pool_count == 1
    assert frame.unsupported == ()
    hazard = frame.hazards[0]
    assert hazard.position == (200.0, 300.0)
    assert hazard.velocity == (0.0, 3.0)


def test_retry_until_both_reads_agree():
    states = [FakeState(10), FakeState(11), FakeState(12), FakeState(12)]
    reader = FakeReader(states, FakeField([]))
    frame = adapter.read_synced(reader, attempts=4)
    assert frame.stable and frame.attempts == 2
    assert frame.frame_id == 12


def test_unstable_snapshot_drops_the_pool_and_names_the_reason():
    states = [FakeState(10), FakeState(11), FakeState(12), FakeState(13)]
    reader = FakeReader(states, FakeField([(200.0, 300.0)]))
    frame = adapter.read_synced(reader, attempts=2)
    assert not frame.stable
    assert frame.hazards is None
    assert adapter.REASON_NOT_STABLE in frame.reasons


def test_pool_read_error_is_not_an_empty_pool():
    reader = FakeReader([FakeState(10), FakeState(10)], error=True)
    frame = adapter.read_synced(reader)
    assert frame.stable
    assert frame.hazards is None
    assert adapter.REASON_BULLET_READ_ERROR in frame.reasons
    observation = adapter.to_observation(frame, episode_id="e", decision_id=0)
    assert observation.completeness is contracts.Completeness.PARTIAL
    assert observation.hazards is None
    contracts.validate_document(contracts.to_json_dict(observation))


def test_empty_pool_is_complete_and_valid():
    reader = FakeReader([FakeState(10), FakeState(10)], FakeField([]))
    frame = adapter.read_synced(reader)
    assert frame.stable and frame.hazards == () and frame.pool_count == 0
    observation = adapter.to_observation(frame, episode_id="e", decision_id=0)
    assert observation.completeness is contracts.Completeness.COMPLETE
    contracts.validate_document(contracts.to_json_dict(observation))


def test_frame_regression_is_unreadable():
    reader = FakeReader([FakeState(50), FakeState(50)], FakeField([]))
    frame = adapter.read_synced(reader, last_frame=100)
    assert not frame.stable
    assert adapter.REASON_FRAME_REGRESSION in frame.reasons
    observation = adapter.to_observation(frame, episode_id="e", decision_id=0)
    assert observation.completeness is contracts.Completeness.UNREADABLE


def test_per_axis_half_extents_are_preserved():
    field = FakeField([(200.0, 300.0)], halves=[(6.0, 1.5)])
    reader = FakeReader([FakeState(10), FakeState(10)], field)
    frame = adapter.read_synced(reader)
    assert frame.hazards[0].half == (6.0, 1.5)


def test_bullet_time_is_declared_as_a_coverage_gap():
    reader = FakeReader([FakeState(10), FakeState(10)], FakeField([]))
    unverified = adapter.read_synced(reader)
    assert adapter.COVERAGE_BULLET_TIME_UNVERIFIED in unverified.unsupported
    active = adapter.read_synced(reader, framerate_multiplier=0.5)
    assert adapter.COVERAGE_BULLET_TIME_ACTIVE in active.unsupported
    linear = adapter.read_synced(reader, framerate_multiplier=1.0)
    assert linear.unsupported == ()


def test_observation_carries_units_and_latency():
    reader = FakeReader([FakeState(10, focus=1, counter=3)], FakeField([]))
    frame = adapter.read_synced(reader)
    observation = adapter.to_observation(
        frame, episode_id="e", decision_id=7, previous_action=(1, 1, 1, 0), frameskip=2
    )
    document = contracts.to_json_dict(observation)
    assert document["action_latency_frames"] == 1
    assert document["coordinate_contract"] == contracts.COORDINATE_CONTRACT
    assert document["player"]["focus"] == 1
    assert contracts.validate_document(document) == []


def test_settle_model_matches_the_recorded_trace_exactly():
    """Numbers from v3.6.1 t2-power ep40, decisions 428 -> 429."""
    path = geometry.executed_path((261.574, 353.314), (0.0, -2.0), (2.828, 2.828), frames=2)
    assert path[0].tolist() == pytest.approx([261.574, 353.314])
    assert path[1].tolist() == pytest.approx([261.574, 351.314])
    assert path[2].tolist() == pytest.approx([264.402, 354.142], abs=1e-3)


def test_settle_model_clamps_at_the_wall():
    path = geometry.executed_path((376.0, 432.0), (4.0, 4.0), (4.0, 4.0), frames=2)
    assert path[1].tolist() == [376.0, 432.0]
    assert path[2].tolist() == [376.0, 432.0]


def test_game_collision_is_per_frame_not_continuous():
    player = (192.0, 384.0)
    half = np.array([[1.0, 1.0]])
    # 24 px/frame: no frame overlap, but the bullet crosses the hurtbox at t=0.5.
    path = geometry.player_path(player, player, 2)
    tunnelled = geometry.frame_min_clearance(
        path, np.array([[180.0, 384.0]]), np.array([[24.0, 0.0]]), half, 0.825
    )[0]
    swept = geometry.path_min_clearance(
        path, np.array([[180.0, 384.0]]), np.array([[24.0, 0.0]]), half, 0.825
    )[0]
    assert tunnelled > 0.0
    assert swept < 0.0

    # 6 px/frame: the frame check sees it too.
    slow_path = geometry.player_path(player, player, 2)
    frame_hit = geometry.frame_min_clearance(
        slow_path, np.array([[186.0, 384.0]]), np.array([[6.0, 0.0]]), half, 0.825
    )[0]
    assert frame_hit <= 0.0


def test_gap_inflation_bound_from_the_edge_instances():
    # E-05: a gap 3 px wider than the hurtbox stays passable strictly below 1.5 px.
    gap_wide = 2 * 0.825 + 3.0
    bound = geometry.max_inflation_for_gap(gap_wide, 0.825)
    assert bound == pytest.approx(1.5)
    # E-06: a gap 1 px narrower is already impassable with zero inflation.
    assert geometry.max_inflation_for_gap(2 * 0.825 - 1.0, 0.825) == 0.0
