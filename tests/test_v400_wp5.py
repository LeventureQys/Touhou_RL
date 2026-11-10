"""Tests for the fake/in-process WP5 contract harness."""

import pytest

from touhou_rl.v400.wp5 import InvalidStateError, ShadowProcess, ShadowState


def test_normal_lifecycle_and_park_resume():
    process = ShadowProcess(7)
    assert process.status == "created"
    assert process.start().frame == 0
    assert process.step({"left": True}).frame == 1
    parked = process.park()
    assert process.status == "parked"
    assert process.snapshot() == parked
    assert process.resume().state_digest == parked.state_digest
    assert process.step("fire").frame == 2
    process.close()
    assert process.status == "closed"
    process.close()  # close is idempotent


def test_same_seed_and_actions_are_deterministic():
    actions = [None, "left", {"fire": 1}, ["right", 2]]
    first = ShadowProcess(123)
    second = ShadowProcess(123)
    first.start()
    second.start()
    for action in actions:
        assert first.step(action) == second.step(action)

    other = ShadowProcess(124)
    other.start()
    for action in actions:
        other.step(action)
    assert other.snapshot().state_digest != first.snapshot().state_digest


def test_branch_and_spawn_are_independent():
    parent = ShadowProcess(5)
    parent.start()
    parent.step("a")
    snapshot = parent.snapshot()
    child = parent.branch(snapshot)
    spawned = parent.spawn()
    assert child.snapshot() == snapshot
    assert spawned.snapshot() == snapshot

    child.step("child")
    spawned.step("spawned")
    assert parent.snapshot() == snapshot
    assert child.snapshot().frame == snapshot.frame + 1
    assert spawned.snapshot().frame == snapshot.frame + 1
    assert child.snapshot().state_digest != spawned.snapshot().state_digest


def test_invalid_state_operations_raise():
    process = ShadowProcess(1)
    with pytest.raises(InvalidStateError):
        process.step("before-start")
    with pytest.raises(InvalidStateError):
        process.park()
    process.start()
    with pytest.raises(InvalidStateError):
        process.start()
    process.park()
    with pytest.raises(InvalidStateError):
        process.step("parked")
    process.resume()
    process.close()
    with pytest.raises(InvalidStateError):
        process.snapshot()
    with pytest.raises(InvalidStateError):
        process.resume()


def test_snapshot_is_immutable_state_value():
    process = ShadowProcess(9)
    state = process.start()
    assert isinstance(state, ShadowState)
    process.step("x")
    assert state.frame == 0
