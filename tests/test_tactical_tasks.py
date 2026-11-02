from dataclasses import replace
import pytest

from touhou_rl.tactical_tasks import (
    advance, development_tasks, diagnostic_actions, goal_reached, initial_state,
    metrics, replay, solve, verify_route,
)


def test_development_only_grouped_mirrors_and_deterministic_replay():
    tasks = development_tasks()
    assert len(tasks) == 24
    assert len({t.event_id for t in tasks}) == 24
    for left, right in zip(tasks[::2], tasks[1::2]):
        assert left.template_group == right.template_group
        assert left.split == right.split == "development"
        assert left.start_x + right.start_x == 40
        with pytest.raises(ValueError):
            replace(left, split="validation")
        a = diagnostic_actions(left, "immediate_transit")
        b = diagnostic_actions(right, "immediate_transit")
        assert a == tuple(-x for x in b)
        ls, lt = replay(left, a)
        rs, rt = replay(right, b)
        assert ls.hit == rs.hit
        assert [p["x"] for p in lt] == [40 - p["x"] for p in rt]
        assert replay(left, a) == replay(left, a)


@pytest.mark.parametrize("task", development_tasks(), ids=lambda t: t.event_id)
def test_independent_search_finds_and_replays_positive_goals(task):
    result = solve(task)
    assert result.status == "feasible"
    assert result.checked <= 100000
    assert len(result.actions) == 20
    assert verify_route(task, result.actions).status == "feasible"
    assert goal_reached(task, replay(task, result.actions)[0])


def test_f4_wait_then_pass_and_immediate_exit_counterexample():
    for task in development_tasks():
        if task.family != "F4":
            continue
        immediate = metrics(task, diagnostic_actions(task, "immediate_transit"))
        waited = metrics(task, diagnostic_actions(task, "wait_then_transit"))
        assert waited["goal_success"] == (task.subtype == "wait_pass")
        assert immediate["goal_success"] == (task.subtype == "immediate_exit")
        assert waited["hit"] == (task.subtype == "immediate_exit")
        assert immediate["hit"] == (task.subtype == "wait_pass")
        hold = metrics(task, (0,) * 20)
        assert hold["survived"] and not hold["goal_success"]
        assert verify_route(task, (0,) * 20).status == "rejected_witness"


def test_f5_real_local_recovery_not_stationary_or_large_migration():
    for task in development_tasks():
        if task.family != "F5":
            continue
        local = metrics(task, diagnostic_actions(task, "local_recover"))
        hold = metrics(task, (0,) * 20)
        broad = metrics(task, diagnostic_actions(task, "immediate_transit"))
        assert local["survived"] and local["goal_success"]
        assert 0 < local["max_displacement"] <= 10
        assert hold["hit"] and not hold["goal_success"]
        assert not broad["goal_success"]
        # Even safe terminal placement does not erase excessive earlier excursion.
        state = replay(task, diagnostic_actions(task, "local_recover"))[0]
        assert not goal_reached(task, replace(state, max_displacement=11))


def test_unknown_budget_incomplete_and_bad_route_never_prove_no_solution():
    task = development_tasks()[0]
    assert solve(task, 0).status == "unknown"
    assert solve(task, 1).status == "unknown"
    assert verify_route(task, ()).status == "unknown"
    unknown = replace(task, model_known=False)
    assert solve(unknown).status == "unknown"
    assert verify_route(unknown, (0,) * 20).status == "unknown"
    assert verify_route(task, (0,) * 20).status == "rejected_witness"
    for budget in (-1, 1.5, True):
        with pytest.raises(ValueError):
            solve(task, budget)


def test_every_logical_frame_checked_and_collision_sticky():
    task = next(t for t in development_tasks() if t.subtype == "wait_pass" and t.offset == 0 and not t.mirror)
    # One-frame gate entry at x18 cannot be hidden by a later action.
    state = replace(initial_state(task), x=17)
    state = advance(task, state, 1)
    assert state.frame == 2 and state.hit
    assert advance(task, state, -2).hit
    for action in (3, 1.0, True):
        with pytest.raises(ValueError):
            advance(task, initial_state(task), action)


def test_unsupported_geometry_and_arm_fail_closed():
    task = development_tasks()[0]
    for change in ({"horizon": 42}, {"offset": 2}, {"subtype": "unknown"}):
        with pytest.raises(ValueError):
            replace(task, **change)
    with pytest.raises(ValueError):
        diagnostic_actions(task, "local_recover")
    with pytest.raises(ValueError):
        diagnostic_actions(task, "replacement_baseline")
