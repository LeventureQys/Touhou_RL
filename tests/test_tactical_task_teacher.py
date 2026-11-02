"""Isolated tests for the privileged task rule teacher; no game input."""
from dataclasses import replace
import json

import pytest

from touhou_rl import tactical_tasks as world
from touhou_rl import tactical_task_teacher as teacher


@pytest.mark.parametrize("task", world.development_tasks(), ids=lambda t: t.event_id)
def test_closed_loop_goal_and_independent_replay(task):
    result = teacher.rollout(task)
    assert result["terminal"]["termination"] == "success"
    assert result["failure"] is None
    assert len(result["decisions"]) == 20
    assert result["metrics"]["goal_success"]
    assert world.verify_route(task, result["actions"]).status == "feasible"
    assert all(d["target"] == tuple(sorted(task.physical_x(x) for x in task.target))
               for d in result["decisions"])
    if task.family == "F5":
        assert result["metrics"]["max_displacement"] <= 10
        assert all(d["option"] == "LOCAL_RECOVER" for d in result["decisions"])


def test_teacher_never_calls_reference_or_diagnostic(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("reference/diagnostic used to choose actions")
    for name in ("solve", "verify_route", "diagnostic_actions"):
        monkeypatch.setattr(world, name, forbidden)
    for task in world.development_tasks():
        assert teacher.rollout(task)["metrics"]["goal_success"]


def test_wait_and_immediate_exit_from_time_corridor():
    wait = world.development_tasks()[0]
    immediate = world.development_tasks()[6]
    a = teacher.decide(wait, world.initial_state(wait))
    b = teacher.decide(immediate, world.initial_state(immediate))
    assert a.action == 0 and a.departure_frame > 0
    assert a.trigger_reason == "wait_for_time_clear_corridor"
    assert abs(b.action) == 2 and b.departure_frame == 0
    assert b.trigger_reason == "time_clear_corridor_transit"


def test_rule_responds_to_danger_not_subtype(monkeypatch):
    task = world.development_tasks()[0]  # subtype remains wait_pass
    def closing_corridor(task, frame, x):
        return frame >= 11 and 18 <= task.canonical_x(x) <= 22
    monkeypatch.setattr(world, "dangerous", closing_corridor)
    assert abs(teacher.decide(task, world.initial_state(task)).action) == 2
    def opening_corridor(task, frame, x):
        return frame <= 16 and 18 <= task.canonical_x(x) <= 22
    monkeypatch.setattr(world, "dangerous", opening_corridor)
    task = replace(task, subtype="immediate_exit")
    assert teacher.decide(task, world.initial_state(task)).action == 0


def test_recomputes_from_current_state():
    task = replace(world.development_tasks()[0], offset=0, mirror=False)
    # The same task at a later frame must transit, not restart a waiting sequence.
    decision = teacher.decide(task, world.State(18, 10))
    assert decision.action == 2
    assert decision.departure_frame == 18
    # Already beyond the closing corridor: no subtype-based delay.
    task = replace(task, subtype="immediate_exit")
    assert teacher.decide(task, world.State(20, 26)).action == 2


def test_f5_two_frame_collision_overrides_target_preference(monkeypatch):
    task = replace(world.development_tasks()[18], offset=0, mirror=False)
    state = world.initial_state(task)
    # Target preference is +1. Only the second intermediate frame is dangerous,
    # so checking frame+1 only or merely current occupancy would fail this test.
    monkeypatch.setattr(world, "dangerous", lambda t, frame, x: frame == 2 and x == 18)
    decision = teacher.decide(task, state)
    assert decision.action != 1
    assert decision.trigger_reason == "immediate_collision_priority"
    assert not world.advance(task, state, decision.action).hit
    record = next(c for c in decision.candidates if c["action"] == 1)
    assert record["collision_2f"]


def test_f5_emergency_can_move_away_from_target(monkeypatch):
    task = replace(world.development_tasks()[12], offset=0, mirror=False)
    state = world.State(8, 20, max_displacement=4)
    monkeypatch.setattr(world, "dangerous", lambda t, frame, x: frame == 10 and x >= 20)
    decision = teacher.decide(task, state)
    assert decision.action < 0
    assert decision.option == "LOCAL_RECOVER"
    assert not world.advance(task, state, decision.action).hit


def test_unknown_and_observed_failure_are_not_success():
    task = world.development_tasks()[0]
    unknown = teacher.rollout(replace(task, model_known=False))
    assert unknown["failure"] == "unsupported_model"
    assert unknown["terminal"]["termination"] == "interrupted"
    decision = teacher.decide(task, world.State(2, task.start_x, hit=True))
    assert decision.termination == "failed"
    assert decision.failure_reason == "collision"


def test_no_local_witness_not_infeasibility(monkeypatch):
    task = world.development_tasks()[12]
    monkeypatch.setattr(world, "dangerous", lambda *args: True)
    result = teacher.rollout(task)
    assert result["decisions"][0]["failure_reason"] == "no_local_witness_not_infeasibility"
    assert result["failure"] == "collision"
    assert not result["metrics"]["goal_success"]


def test_horizon_goal_failure():
    task = world.development_tasks()[0]
    decision = teacher.decide(task, world.State(task.horizon, task.start_x))
    assert decision.termination == "failed"
    assert decision.failure_reason == "goal_or_local_bound_missed"


def test_mirror_and_offset_equivariance():
    for start in (0, 6, 12, 18):
        tasks = world.development_tasks()[start:start + 6]
        actions = [teacher.rollout(task)["actions"] for task in tasks]
        assert actions[0] == actions[2] == actions[4]
        assert actions[1] == actions[3] == actions[5] == [-a for a in actions[0]]


def test_evaluation_preregistration_required(tmp_path):
    with pytest.raises(ValueError, match="preregistration"):
        teacher.evaluate_development(tmp_path)


def test_evaluation_boundary_preserves_failure_and_svg(tmp_path, monkeypatch):
    (tmp_path / "preregistration.md").write_text("Frozen test protocol", encoding="utf-8")
    task = world.development_tasks()[0]
    monkeypatch.setattr(world, "development_tasks", lambda: (task,))
    original = teacher.decide
    def bad_teacher(t, state):
        return replace(original(t, state), action=0)
    monkeypatch.setattr(teacher, "decide", bad_teacher)
    result = teacher.evaluate_development(tmp_path)
    assert result["summary"][task.subtype]["goal_success"] == 0
    assert len(result["summary"][task.subtype]["failures"]) == 1
    row = json.loads((tmp_path / f"{task.event_id}.json").read_text(encoding="utf-8"))
    assert row["teacher_verification"]["status"] == "rejected_witness"
    assert row["reference"]["status"] == "feasible"
    assert row["teacher"]["failure"] == "goal_or_local_bound_missed"
    assert row["formal_training_labels"] is False
    assert "<svg" in (tmp_path / f"{task.event_id}.svg").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="already exist"):
        teacher.evaluate_development(tmp_path)
