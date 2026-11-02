"""C-01: the experiment clock must be the only termination basis.

Every case here is a failure mode observed (or inferred) in v3.7.0's live run:
the counter rolling back, a read failing, the frame jumping because the process
was descheduled, the planning pause letting the game advance, and the loop
stopping at a decision count instead of a frame.
"""

from __future__ import annotations

import pytest

from touhou_rl.escape import clock, fakegame


def _fake(**kwargs):
    time = fakegame.FakeTime()
    game = fakegame.FakeGame(time, **kwargs)
    reader = clock.ExperimentClock(game.read_state, monotonic=time.monotonic, sleep=time.sleep)
    return time, game, reader


# ---------------------------------------------------------------------------
# Frame accounting
# ---------------------------------------------------------------------------
def test_clock_reads_monotonic_frames():
    time, game, reader = _fake()
    first = reader.read()
    time.advance(0.5)  # 30 frames at 60 fps
    second = reader.read()
    assert second.frame > first.frame
    assert second.frame - first.frame == pytest.approx(30, abs=1)
    assert reader.summary()["rollbacks"] == 0


def test_clock_waits_until_a_real_frame():
    time, game, reader = _fake()
    start = reader.read().frame
    outcome = reader.wait_until(start + 10, timeout_s=2.0)
    assert outcome.reached
    assert outcome.reading.frame >= start + 10
    # Every frame on the way was observed, so death attribution has no holes.
    frames = [sample.frame for sample in outcome.samples]
    assert frames == sorted(set(frames))
    assert len(frames) >= 10


def test_rollback_is_an_anomaly_not_a_negative_exposure():
    """A counter that goes backwards must start a new epoch, never go negative."""

    time, game, reader = _fake(rollback_at_read=4, start_frame=500)
    reader.read()
    reader.read()
    for _ in range(6):
        try:
            reader.read()
        except OSError:
            pass
    summary = reader.summary()
    assert summary["rollbacks"] == 1
    assert summary["epoch"] == 1
    kinds = [entry["kind"] for entry in summary["anomalies"]]
    assert clock.ANOMALY_ROLLBACK in kinds
    # The window arithmetic is clamped by construction.
    window = clock.FrameWindow(start_frame=500, frames=1800)
    assert window.exposure(480) == 0


def test_read_failure_is_recorded_then_raises_after_the_budget():
    time, game, reader = _fake(read_fail_at_read=(1, 2, 3))
    with pytest.raises(OSError):
        reader.read()
    assert reader.summary()["anomaly_counts"].get(clock.ANOMALY_READ_FAIL) == 1

    strict = clock.ExperimentClock(
        game.read_state,
        monotonic=time.monotonic,
        sleep=time.sleep,
        config=clock.ClockConfig(max_read_failures=1),
    )
    with pytest.raises(OSError):
        strict.read()
    with pytest.raises(clock.ClockError):
        strict.read()


def test_frame_jump_is_recorded_with_its_delta():
    time, game, reader = _fake(jump_at_frame=(105, 40))
    reader.read()
    time.advance(0.2)
    reader.read()
    summary = reader.summary()
    jumps = [entry for entry in summary["anomalies"] if entry["kind"] == clock.ANOMALY_JUMP]
    assert jumps, "a 40-frame jump must be visible in the audit"
    assert max(entry["delta"] for entry in jumps) >= 20


def test_stalled_counter_times_out_and_is_not_reported_as_reached():
    time, game, reader = _fake(stall_from=102)
    start = reader.read().frame
    outcome = reader.wait_until(start + 50, timeout_s=0.05)
    assert not outcome.reached
    assert outcome.reason == "timeout"
    kinds = [entry["kind"] for entry in reader.summary()["anomalies"]]
    assert clock.ANOMALY_TIMEOUT in kinds


def test_stall_is_detected_without_waiting_for_the_deadline():
    time, game, reader = _fake(
        stall_from=102,
    )
    reader.config = clock.ClockConfig(stall_s=0.01)
    start = reader.read().frame
    outcome = reader.wait_until(start + 50, timeout_s=5.0)
    assert outcome.reason == "stall"
    assert not outcome.reached


def test_left_game_is_reported_rather_than_waited_out():
    time, game, reader = _fake(supervisor_state=3)
    game.supervisor_state = 3
    outcome = reader.wait_until(10_000, timeout_s=1.0)
    assert outcome.reason == "left_game"
    assert not outcome.reached


# ---------------------------------------------------------------------------
# Window / interval classification (C-02 groundwork)
# ---------------------------------------------------------------------------
def test_window_is_half_open_and_never_negative():
    window = clock.FrameWindow(start_frame=100, frames=1800)
    assert window.end_frame == 1900
    assert window.contains(100) and window.contains(1899)
    assert not window.contains(1900)
    assert window.classify(99) == "before"
    assert window.classify(1900) == "at_or_after_end"
    assert window.exposure(9999) == 1800
    assert window.exposure(0) == 0


def test_interval_classification_flags_boundary_unknown():
    window = clock.FrameWindow(start_frame=100, frames=100)
    assert clock.FrameInterval(120, 120).classify(window) == "inside"
    assert clock.FrameInterval(50, 60).classify(window) == "before"
    assert clock.FrameInterval(200, 210).classify(window) == "at_or_after_end"
    # A gap that starts inside and ends outside cannot be attributed.
    assert clock.FrameInterval(198, 205).classify(window) == "boundary_unknown"
    assert clock.FrameInterval(90, 105).classify(window) == "boundary_unknown"


def test_event_ledger_attributes_deaths_to_the_window_or_flags_them():
    window = clock.FrameWindow(start_frame=100, frames=100)
    ledger = clock.EventLedger(window=window)

    def reading(frame, deaths):
        return clock.FrameReading(frame=frame, wall_s=0.0, deaths=deaths, supervisor_state=2)

    ledger.observe(reading(100, 0))
    ledger.observe(reading(101, 0))
    ledger.observe(reading(102, 1))  # exact, inside
    ledger.observe(reading(150, 1))
    ledger.observe(reading(195, 1))
    ledger.observe(reading(205, 2))  # interval 196..205 crosses the end boundary
    summary = ledger.summary()
    assert summary["deaths_in_window"] == 1
    assert summary["boundary_unknown"] == 1
    assert summary["deaths_after_window"] == 0
    assert summary["death_events_total"] == 2


def test_event_ledger_separates_bombs_from_deaths():
    window = clock.FrameWindow(start_frame=0, frames=10)
    ledger = clock.EventLedger(window=window)

    def reading(frame, deaths=0, bombs_used=0, bombs=2.0):
        return clock.FrameReading(
            frame=frame, wall_s=0.0, deaths=deaths, bombs_used=bombs_used, bombs=bombs, supervisor_state=2
        )

    ledger.observe(reading(0))
    ledger.observe(reading(1, bombs_used=1, bombs=1.0))
    ledger.observe(reading(5, deaths=1, bombs_used=1, bombs=1.0))
    summary = ledger.summary()
    assert summary["bomb_count"] == 1
    assert summary["deaths_in_window"] == 1


def test_death_counter_reset_is_flagged_not_counted_negative():
    window = clock.FrameWindow(start_frame=0, frames=100)
    ledger = clock.EventLedger(window=window)
    ledger.observe(clock.FrameReading(frame=0, wall_s=0.0, deaths=2, supervisor_state=2))
    ledger.observe(clock.FrameReading(frame=1, wall_s=0.0, deaths=0, supervisor_state=2))
    summary = ledger.summary()
    assert summary["deaths_in_window"] == 0
    assert any(event["kind"] == "counter_reset" for event in summary["deaths"])


# ---------------------------------------------------------------------------
# Termination is a frame rule, not a decision count
# ---------------------------------------------------------------------------
def test_termination_rule_counts_frames_not_decisions():
    window = clock.FrameWindow(start_frame=100, frames=1800)
    rule = clock.TerminationRule(window=window, frameskip=2)
    assert rule.slots == 900
    assert rule.slot_start(0) == 100
    assert rule.slot_end(899) == 1900
    # A slot whose action would run past the window boundary does not fit.
    assert rule.fits(898, act_lag=1)
    assert not rule.fits(899, act_lag=1)
    assert rule.stop_reason(1899, 899, act_lag=1) == "window_complete"


def test_termination_rule_rejects_a_bad_frameskip():
    with pytest.raises(ValueError):
        clock.TerminationRule(window=clock.FrameWindow(start_frame=0, frames=10), frameskip=0)
    with pytest.raises(ValueError):
        clock.FrameWindow(start_frame=0, frames=0)
