"""Stage A: both arms must be driven by one pipeline and differ only in policy.

The v3.7.0 comparison ran the baseline through ``env.step`` and the subject through
``live.run_live_closed_loop``.  Whatever the death numbers meant, they could not
be attributed to the policy alone.  These tests pin the property that replaces it:
same snapshot, same schedule, same exposure, one difference.
"""

from __future__ import annotations

import pytest

from touhou_rl.escape import arms, controller, fakegame, harness

BULLETS = ((200.0, 100.0, 0.0, 2.0, 2.5), (300.0, 60.0, 0.0, 2.0, 2.5))

#: A cheap planner config: these tests are about the *pipeline*, and the live
#: config (h6/c1/beam1) still costs ~14 ms per decision, which would make the
#: suite take minutes for no extra coverage.
FAST_PLANNER = controller.ControllerConfig(
    planner=controller.planner.PlannerConfig(
        horizon_decisions=4, continuation_decisions=1, beam_width=1
    )
)


def _subject(**kwargs):
    return arms.make_arm("subject", controller_config=FAST_PLANNER, **kwargs)


def _episode(arm, *, frames: int = 240, game_kwargs=None, config_kwargs=None):
    time = fakegame.FakeTime()
    game = fakegame.FakeGame(time, bullet_set=BULLETS, **(game_kwargs or {}))
    document = harness.run_episode(
        game,
        arm,
        config=harness.HarnessConfig(**{"frames": frames, **(config_kwargs or {})}),
        episode_id="stage-a",
        monotonic=time.monotonic,
        sleep=time.sleep,
    )
    return game, document


def test_baseline_and_subject_share_the_episode_frame_schedule():
    """The two arms must expose the *same* frame window through the same loop."""

    _game_b, baseline = _episode(arms.make_arm("baseline"))
    _game_s, subject = _episode(_subject())
    assert baseline["technical_failure"] is None
    assert subject["technical_failure"] is None
    assert baseline["window"] == subject["window"]
    assert baseline["exposure_frames"] == subject["exposure_frames"] == 240
    assert baseline["decisions"] == subject["decisions"]
    assert baseline["hold_frames"]["distribution"] == subject["hold_frames"]["distribution"]
    # Same slot grid for both: the observation frames are the schedule, not the policy.
    assert [row["slot_start_frame"] for row in baseline["rows"]] == [
        row["slot_start_frame"] for row in subject["rows"]
    ]


def test_arms_differ_only_in_the_actions_they_choose():
    _b, baseline = _episode(arms.make_arm("baseline"))
    _s, subject = _episode(_subject())
    baseline_actions = [tuple(row["action"]) for row in baseline["rows"]]
    subject_actions = [tuple(row["action"]) for row in subject["rows"]]
    assert baseline["decisions"] == len(baseline_actions)
    assert {row["intent"] for row in baseline["rows"]} == {"baseline"}
    assert {row["intent"] for row in subject["rows"]} <= {"local", "prepare_exit", "evacuate", "hold", "replan", "emergency", "unknown"}
    assert baseline_actions != subject_actions


def test_actions_are_always_in_the_frozen_action_space():
    for arm in (arms.make_arm("baseline"), _subject()):
        _game, document = _episode(arm, frames=120)
        for row in document["rows"]:
            move, shoot, focus, bomb = row["action"]
            assert 0 <= move <= 8
            assert shoot in (0, 1) and focus in (0, 1) and bomb in (0, 1)


def test_subject_arm_resets_between_episodes():
    """No controller residue may cross an episode boundary (or an arm boundary)."""

    arm = _subject()
    game_a, first = _episode(arm, frames=120)
    game_b, second = _episode(arm, frames=120)
    assert [row["action"] for row in first["rows"]] == [row["action"] for row in second["rows"]]


def test_baseline_arm_is_a_fresh_pilot_per_episode():
    arm = arms.make_arm("baseline")
    _game_a, first = _episode(arm, frames=120)
    first_pilot = arm.pilot
    _game_b, second = _episode(arm, frames=120)
    assert arm.pilot is not first_pilot
    assert [row["action"] for row in first["rows"]] == [row["action"] for row in second["rows"]]


def test_baseline_arm_observes_the_current_frame_not_the_reset_frame():
    """v3.7.0's baseline was fed a fresh ``obs`` each step; keep that property."""

    _game, document = _episode(arms.make_arm("baseline"), frames=240)
    observed = [(row["obs_x"], row["obs_y"]) for row in document["rows"]]
    assert len(set(observed)) > 1, "the pilot must see its own movement, not a frozen start"


def test_subject_config_is_the_only_difference_between_two_subject_runs():
    default_arm = _subject()
    wider = arms.make_arm(
        "subject",
        controller_config=controller.ControllerConfig(
            planner=controller.planner.PlannerConfig(horizon_decisions=4, continuation_decisions=1)
        ),
    )
    _g1, first = _episode(default_arm, frames=120)
    _g2, second = _episode(wider, frames=120)
    assert first["exposure_frames"] == second["exposure_frames"] == 120
    assert first["decisions"] == second["decisions"]


def test_replay_arm_rejects_an_empty_sequence():
    with pytest.raises(ValueError):
        arms.ReplayArm(())


def test_make_arm_rejects_unknown_names():
    with pytest.raises(ValueError):
        arms.make_arm("wishful")
