"""Forensic unit tests: no checkpoint, game, labels, or training."""
from dataclasses import replace
import copy
import pytest
from touhou_rl.tactical_teacher_validation import ValidationWorld, advance, programs
from touhou_rl.tactical_scenarios import initial_state
from touhou_rl.tactical_synthetic_closed_loop import run_event
from touhou_rl import tactical_failure_forensics as f


def world(**kw):
    return ValidationWorld(template_group='fixture', seed=12, event_id='fixture', family='F2', split='validation',
                           start_x=168., horizon=160, **kw)


@pytest.mark.parametrize('family', ['F1', 'F2', 'F3'])
@pytest.mark.parametrize('mirror', [False, True])
def test_independent_logical_physics_matches_world(family, mirror):
    s = ValidationWorld(template_group='fixture', seed=12, event_id='fixture', family=family, split='validation',
                        start_x=200., mirror=mirror, burst=True,
                        depletion_pressure=family == 'F3')
    state = initial_state(s)
    for i in range(s.horizon//2):
        action = f.ACTIONS[i % 5]
        _, independent = f.independent_replay(s, [action], state)
        state = advance(s, state, action)
        assert independent == state


def test_actual_full_trace_and_tamper_rejected():
    s = world()
    trace = run_event(s, 'N1', 'horizontal_masked',
                      lambda *a: (__import__('torch').tensor([0.]+[-1.]*17), None), 0)
    actions, states, report = f.replay_actual(s, trace)
    assert len(actions) == 80 and len(states) == 81
    assert report['verified'] and report['independent_logical_replay_verified']
    tampered = copy.deepcopy(trace)
    tampered['steps'][7]['x_after'] += 1
    with pytest.raises(ValueError, match='mismatch'):
        f.replay_actual(s, tampered)


def test_reference_does_not_use_teacher(monkeypatch):
    s = ValidationWorld(template_group='fixture', seed=12, event_id='safe', family='F1', split='validation',
                        start_x=8., wall_frame=40)
    monkeypatch.setattr(f.TacticalTeacherV2, 'decide', lambda *a: pytest.fail('teacher used'))
    rows = f.reference_candidates(s, [], initial_state(s), budget=1)
    assert len(rows) == 5
    for row in rows:
        assert len(row['first_two_logical_frames']) == 2
        assert row['status'] == 'feasible'
        assert len(row['witness']['actions']) == 80
        assert row['witness']['independent_verification']['status'] == 'feasible'


def test_unknown_not_infeasible_and_guard():
    s = world()
    rows = f.reference_candidates(s, [], initial_state(s), budget=0)
    assert all(r['status'] == 'unknown' and r['witness'] is None for r in rows)
    with pytest.raises(ValueError):
        f.reference_candidates(replace(s, model_known=False), [], initial_state(s))
    with pytest.raises(ValueError):
        f.reference_candidates(s, [], replace(initial_state(s), hit=True))


def test_full_prefix_checked_not_only_suffix():
    s = ValidationWorld(template_group='fixture', seed=12, event_id='collision', family='F1', split='validation',
                        start_x=192., wall_frame=2)
    rows = f.reference_candidates(s, [], initial_state(s), budget=223)
    assert all(r['first_two_hit'] and r['status'] == 'unknown' for r in rows)
    assert all(r['programs_checked'] == 0 for r in rows)


def test_fixed_program_budget_is_complete_finite_class():
    assert len(list(programs(world()))) == 223
    assert f.action_stats([0, 0, 1, -1, -1]) == dict(
        decisions=5, stays=2, action_switches=2, direction_reversals=1)
