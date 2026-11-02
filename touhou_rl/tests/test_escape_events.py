"""Tests for the independent evacuation-event labelling (B-02)."""

from __future__ import annotations

from touhou_rl.escape import contracts, events


def _hazard(x, y, vx=0.0, vy=0.0, half=2.0):
    return contracts.HazardObs(
        position=(x, y),
        velocity=(vx, vy),
        half=(half, half),
        motion_model=contracts.MotionModel.LINEAR,
        confidence=1.0,
        valid_until=10_000,
    )


def test_no_event_when_the_player_is_safe_holding_still():
    hazards = (_hazard(120.0, 120.0),)
    event = events.classify((192.0, 384.0), hazards, (0, 1, 0, 0), frames=20)
    assert event.danger is False
    assert event.is_event is False


def test_event_when_a_bullet_closes_in_but_a_direction_escapes():
    # A bullet heading straight at the player from the left, far enough to matter.
    hazards = (_hazard(120.0, 384.0, vx=4.0, half=3.0),)
    event = events.classify((192.0, 384.0), hazards, (3, 1, 0, 0), frames=30)
    assert event.danger is True
    assert event.exit_exists is True
    assert event.chosen_safe in (True, False)


def test_chosen_action_is_labelled_against_the_safe_set():
    hazards = (_hazard(120.0, 384.0, vx=4.0, half=3.0),)
    # Moving right (move 3) escapes; standing still does not.
    good = events.classify((192.0, 384.0), hazards, (3, 1, 0, 0), frames=30)
    bad = events.classify((192.0, 384.0), hazards, (0, 1, 0, 0), frames=30)
    assert good.chosen_safe is True
    assert bad.chosen_safe is False
    summary = events.summarise([good, bad])
    assert summary["events"] == 2
    assert summary["timely"] == 1
    assert summary["rate"] == 0.5


def test_no_exit_means_it_is_not_an_event():
    # Bullets on both sides at point-blank range: no direction survives.
    hazards = tuple(
        _hazard(192.0 + dx, 384.0 + dy, half=80.0) for dx in (-60.0, 60.0) for dy in (-60.0, 60.0)
    )
    event = events.classify((192.0, 384.0), hazards, (0, 1, 0, 0), frames=10)
    assert event.danger is True
    assert event.exit_exists is False
    assert event.is_event is False
