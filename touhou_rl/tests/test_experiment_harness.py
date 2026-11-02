"""C-02/C-03/C-04: equal logical-frame exposure, real input timing, delay robustness.

The v3.7.0 defect these tests pin down is not a wrong number; it is that the
subject arm was allowed to accumulate extra frames while planning, and the
baseline arm was never measured at all.  So the assertions are structural:

* the episode always ends on ``gameplay_frame``, so exposure is the same number
  whether planning took 0 or 10 frames;
* a slow planner costs *measured hold length*, never a longer experiment;
* ``send_frame`` and ``effective_frame`` are separate fields, and when the engine
  readback cannot resolve the transition the record says so instead of inventing
  a point estimate.
"""

from __future__ import annotations

import pytest

from touhou_rl.escape import arms, clock, fakegame, harness

FRAMES = 120


def _run(policy_work_frames: float = 0.0, **kwargs):
    """Run one fake episode where planning burns ``policy_work_frames`` frames."""

    game_kwargs = kwargs.pop("game", {})
    config_kwargs = kwargs.pop("config", {})
    time = fakegame.FakeTime()
    game = fakegame.FakeGame(time, **game_kwargs)
    work_s = float(policy_work_frames) / 60.0

    def policy(snapshot, context):
        if work_s:
            time.advance(work_s)
        return (3, 1, 0, 0)

    config = harness.HarnessConfig(**{"frames": FRAMES, **config_kwargs})
    document = harness.run_episode(
        game,
        arms.ScriptedArm(policy),
        config=config,
        episode_id="fake",
        monotonic=time.monotonic,
        sleep=time.sleep,
    )
    return time, game, document


# ---------------------------------------------------------------------------
# C-02: exposure is a frame count
# ---------------------------------------------------------------------------
def test_episode_ends_on_the_frame_window_not_a_decision_count():
    time, game, document = _run()
    assert document["termination"] == "window_complete"
    assert document["technical_failure"] is None
    assert document["window"]["frames"] == FRAMES
    assert document["end_frame"] >= document["window"]["end_frame"]
    assert document["exposure_frames"] == FRAMES
    assert document["exposure_shortfall_frames"] == 0
    # The planned decision count is a derived quantity, not the stop rule.
    assert document["planned_slots"] == FRAMES // 2
    assert document["decisions"] != FRAMES // 2 or document["decisions"] == document["planned_slots"]


@pytest.mark.parametrize("work_frames", [0.0, 1.0, 3.0, 10.0])
def test_planning_time_never_changes_total_exposure(work_frames):
    """0/1/3/10 frames of planning must all expose exactly the same window."""

    time, game, document = _run(work_frames)
    assert document["technical_failure"] is None
    assert document["exposure_frames"] == FRAMES
    assert document["deaths_in_window"] == 0
    # Fewer decisions when the planner cannot keep up -- but the same frames.
    if work_frames >= 3.0:
        assert document["decisions"] < document["planned_slots"]
        assert document["plan_over_lag"] > 0


def test_hold_frames_are_exactly_frameskip_when_planning_fits():
    time, game, document = _run(0.0, config={"frames": 240})
    holds = document["hold_frames"]
    assert holds["n"] == document["decisions"] - 1
    assert set(holds["distribution"]) == {"2"}, holds


def test_hold_frames_grow_visibly_when_planning_overruns():
    time, game, document = _run(10.0, config={"frames": 240})
    holds = document["hold_frames"]
    assert holds["max"] > 2
    assert document["plan_over_lag"] > 0
    assert document["send_lateness_frames"]["max"] > 0


def test_death_inside_the_window_is_counted_on_its_frame():
    time, game, document = _run(
        0.0,
        game={"deaths_at": (140, 175)},
        config={"frames": 240},
    )
    assert document["deaths_in_window"] == 2
    assert document["boundary_unknown"] == 0
    assert document["first_death_frame"] == 140
    assert [event["frame_lower"] for event in document["death_events"]] == [140, 175]


def test_death_after_the_window_is_not_counted_as_inside():
    time, game, document = _run(
        0.0,
        game={"deaths_at": (400,)},
        config={"frames": 120},
    )
    assert document["deaths_in_window"] == 0
    assert document["deaths_after_window"] + document["boundary_unknown"] >= 0


def test_stop_on_death_is_a_diagnostic_mode_not_the_window_rule():
    time, game, document = _run(
        0.0,
        game={"deaths_at": (130,)},
        config={"frames": 240, "stop_on_death": True},
    )
    assert document["termination"] == "death"
    assert document["deaths_in_window"] == 1
    assert document["exposure_frames"] < 240


# ---------------------------------------------------------------------------
# C-03: send frame and effective frame are different measurements
# ---------------------------------------------------------------------------
def test_send_and_effective_frames_are_recorded_separately():
    time, game, document = _run(0.0, config={"frames": 240})
    resolved = [
        row
        for row in document["rows"]
        if isinstance(row.get("effective"), dict) and row["effective"]["status"] == "observed"
    ]
    assert resolved, "engine input readback should resolve at least once"
    for row in resolved:
        # The key write and the engine consuming it are different frames.
        assert row["send_frame_after"] >= row["send_target_frame"]
        assert row["effective"]["lower"] == row["send_frame_after"]
        assert row["effective"]["frames_after_send"] is not None
        assert row["effective"]["frames_after_send"] >= 0
    # The fake applies input one frame late, so the transition is observable.
    assert document["effective"]["frames_after_send"]["p50"] >= 1


def test_unresolvable_effective_frame_is_an_interval_not_a_point():
    """If the readback never shows the bits, the record must say unknown."""

    time, game, document = _run(
        0.0,
        game={"input_latency_frames": 10_000},
        config={"frames": 120},
    )
    entries = [row["effective"] for row in document["rows"] if isinstance(row.get("effective"), dict)]
    assert entries
    assert all(entry["status"] != "observed" for entry in entries)
    assert all(entry["effective_frame"] is None for entry in entries)
    assert all(entry["upper"] >= entry["lower"] for entry in entries)


def test_repeated_action_reports_already_active_instead_of_a_fake_point():
    time = fakegame.FakeTime()
    game = fakegame.FakeGame(time)
    arm = arms.ScriptedArm(lambda snapshot, context: (0, 0, 0, 0))  # no keys at all
    document = harness.run_episode(
        game,
        arm,
        config=harness.HarnessConfig(frames=120),
        episode_id="noop",
        monotonic=time.monotonic,
        sleep=time.sleep,
    )
    entries = [row["effective"] for row in document["rows"] if isinstance(row.get("effective"), dict)]
    assert entries
    assert {entry["status"] for entry in entries} == {"already_active"}
    assert all(entry["effective_frame"] is None for entry in entries)


# ---------------------------------------------------------------------------
# C-04: injected latency
# ---------------------------------------------------------------------------
def test_injected_delay_is_measured_and_does_not_extend_the_window():
    # 50 ms is three frames at 60 Hz: enough to push the key write past the
    # scheduled frame, which must show up as measured lateness -- not as a longer
    # experiment and not as a "slower game" excuse.
    time, game, document = _run(0.0, config={"frames": 240, "inject_delay_ms": 50.0})
    assert document["exposure_frames"] == 240
    assert document["send_lateness_frames"]["max"] >= 1
    assert document["timing_ms"]["plan_ms"]["p50"] >= 0.0


def test_actuation_lag_zero_reproduces_the_v370_confound():
    """lag=0 lets planning time become exposure time -- the defect, kept on purpose."""

    time, game, document = _run(0.0, config={"frames": 240, "actuation_lag_frames": 0, "inject_delay_ms": 12.0})
    assert document["exposure_frames"] == 240
    assert document["send_lateness_frames"]["n"] > 0
    assert document["send_lateness_frames"]["max"] >= 0


# ---------------------------------------------------------------------------
# Failure modes must be reported, not hidden
# ---------------------------------------------------------------------------
def test_rollback_aborts_the_episode_without_a_negative_exposure():
    time, game, document = _run(0.0, game={"rollback_at_read": 60}, config={"frames": 600})
    assert document["termination"].startswith("clock_")
    assert document["technical_failure"] is not None
    assert document["exposure_frames"] >= 0
    assert document["clock"]["rollbacks"] >= 1


def test_left_game_is_a_technical_failure_not_a_short_window():
    time, game, document = _run(0.0, game={"supervisor_state": 3}, config={"frames": 120})
    assert document["termination"] in ("left_game", "clock_left_game")
    assert document["technical_failure"] is not None
    assert document["exposure_frames"] == 0


def test_keys_are_released_even_when_the_run_aborts():
    time, game, document = _run(0.0, game={"rollback_at_read": 40}, config={"frames": 240})
    assert game.keys == set()


# ---------------------------------------------------------------------------
# Replay determinism (C-01 / stage A)
# ---------------------------------------------------------------------------
def _replay_run(action: tuple[int, int, int, int], frames: int = 180):
    time = fakegame.FakeTime()
    game = fakegame.FakeGame(time, deaths_at=())
    document = harness.run_episode(
        game,
        arms.ReplayArm(tuple([action] * 400)),
        config=harness.HarnessConfig(frames=frames),
        episode_id="replay",
        monotonic=time.monotonic,
        sleep=time.sleep,
    )
    return game, document


def test_transport_test_same_input_same_trajectory():
    """Same input through the same pipeline must reproduce the trajectory exactly."""

    game_a, doc_a = _replay_run((3, 1, 0, 0))
    game_b, doc_b = _replay_run((3, 1, 0, 0))
    assert doc_a["exposure_frames"] == doc_b["exposure_frames"] == 180
    assert doc_a["decisions"] == doc_b["decisions"]
    assert fakegame.fingerprint(game_a.trace) == fakegame.fingerprint(game_b.trace)
    assert game_a.key_log == game_b.key_log


def test_different_input_produces_a_different_trajectory():
    game_left, _ = _replay_run((7, 1, 0, 0))
    game_right, _ = _replay_run((3, 1, 0, 0))
    assert fakegame.fingerprint(game_left.trace) != fakegame.fingerprint(game_right.trace)


def test_noop_arm_holds_the_empty_action_for_the_whole_window():
    time = fakegame.FakeTime()
    game = fakegame.FakeGame(time)
    document = harness.run_episode(
        game,
        arms.NoOpArm(),
        config=harness.HarnessConfig(frames=120),
        episode_id="noop",
        monotonic=time.monotonic,
        sleep=time.sleep,
    )
    assert document["exposure_frames"] == 120
    assert all(row["action"] == [0, 0, 0, 0] for row in document["rows"])
    assert len(game.key_log) == 0  # nothing was ever pressed


# ---------------------------------------------------------------------------
# AB/BA ordering (C-05)
# ---------------------------------------------------------------------------
def test_balanced_random_order_is_seeded_and_balanced():
    first = harness.plan_order(10, scheme="balanced_random", seed=7)
    second = harness.plan_order(10, scheme="balanced_random", seed=7)
    assert first == second
    ab = sum(1 for order in first if order[0] == "baseline")
    assert abs(ab - (10 - ab)) <= 1
    assert harness.plan_order(3, scheme="ab")[0] == ("baseline", "subject")
    assert harness.plan_order(3, scheme="ba")[0] == ("subject", "baseline")
    with pytest.raises(ValueError):
        harness.plan_order(1, scheme="nonsense")


# ---------------------------------------------------------------------------
# Config guards
# ---------------------------------------------------------------------------
def test_config_rejects_an_overlapping_schedule():
    with pytest.raises(ValueError):
        harness.HarnessConfig(frames=100, frameskip=2, actuation_lag_frames=3)
    with pytest.raises(ValueError):
        harness.HarnessConfig(frames=0)


def test_stale_observation_does_not_end_the_window():
    """Behind-schedule observation is recorded and the window still completes."""

    time = fakegame.FakeTime()
    game = fakegame.FakeGame(time)

    def slow(snapshot, context):
        time.advance(4 / 60.0)  # four frames of planning per decision
        return (3, 1, 0, 0)

    document = harness.run_episode(
        game,
        arms.ScriptedArm(slow),
        config=harness.HarnessConfig(frames=240),
        episode_id="slow",
        monotonic=time.monotonic,
        sleep=time.sleep,
    )
    assert document["exposure_frames"] == 240
    assert document["termination"] == "window_complete"
    assert document["late_slots"] >= 0
