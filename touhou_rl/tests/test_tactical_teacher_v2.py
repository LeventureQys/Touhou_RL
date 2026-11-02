"""R9 policy contracts; no sealed R8 test execution."""
from dataclasses import replace
import ast
from pathlib import Path
from touhou_rl.tactical_scenarios import initial_state, WorldState, Bullet
from touhou_rl.tactical_teacher_validation import ValidationWorld, advance
from touhou_rl.tactical_teacher_v2 import TacticalTeacherV2, confirmation_definitions


def world(family='F1', **kw):
    return ValidationWorld('unit', family, 'unit', 9, kw.pop('start_x', 192.), **kw)


def test_actual_internal_gap_recognized():
    s = world(start_x=216., internal_gap=True, gap=(208.,224.), wall_frame=0)
    d = TacticalTeacherV2().decide(s, initial_state(s))
    assert d.action == 0
    assert not advance(s, initial_state(s), d.action).hit


def test_false_old_gap_exited_using_actual_geometry():
    s = world(internal_gap=True, gap=(220.,236.), wall_frame=44)
    d = TacticalTeacherV2().decide(s, initial_state(s))
    assert d.action > 0
    assert d.target_region == (222.,234.)


def test_two_frame_collision_beats_f2_microshift():
    s = world('F2', start_x=200.)
    st = WorldState(40,200.,(Bullet(0,204.,400.,0.,0.,204.),),False,({'frame':40,'aim_x':200.},))
    d = TacticalTeacherV2().decide(s,st)
    assert d.action <= 0
    assert not advance(s,st,d.action).hit


def test_f3_actual_window_threshold_goal():
    s = world('F3', start_x=210., window=(46,112),threshold=172.,goal=148.,depletion_pressure=True)
    p = TacticalTeacherV2()
    assert p.target(s,WorldState(0,210.))[0] > s.threshold
    assert p.target(s,WorldState(46,210.))[1] < s.goal
    assert p.decide(s,WorldState(46,210.)).action < 0


def test_causal_birth_changes_with_action_and_mirror():
    s = world('F2', start_x=200., fire_interval=1)
    a = advance(s,initial_state(s),-2)
    b = advance(s,initial_state(s),2)
    assert a.emissions[1]['aim_x'] != b.emissions[1]['aim_x']
    sm = replace(s,mirror=True,start_x=184.)
    dm = TacticalTeacherV2().decide(sm,initial_state(sm))
    d = TacticalTeacherV2().decide(s,initial_state(s))
    assert dm.action == -d.action


def test_unknown_not_infeasible():
    s = world(model_known=False)
    assert TacticalTeacherV2().decide(s,initial_state(s)).confidence == 'unknown'
    s = world(wall_frame=0,coverage=(8.,376.))
    assert TacticalTeacherV2().decide(s,initial_state(s)).trigger_reason.endswith('UNKNOWN')


def test_policy_has_no_reference_calls():
    import inspect
    code = inspect.getsource(TacticalTeacherV2)
    tree = ast.parse(code)
    calls = [n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert not {'solve','verify','programs'} & set(calls)


def test_new_confirmation_pairs_and_no_r8_parameter_duplicate():
    from touhou_rl.tactical_teacher_validation import definitions
    rows = confirmation_definitions()
    assert len(rows) == 24
    def signature(spec):
        s = ValidationWorld(**spec)
        return (s.family,s.canonical_x(s.start_x),s.coverage,s.gap,s.wall_frame,s.fire_interval,s.window,s.threshold,s.goal,s.origin,s.burst,s.bullet_speed,s.internal_gap)
    old = {signature(s) for s in definitions()}
    assert not old & {signature(s) for s in rows}
    for i in range(0,len(rows),2):
        assert rows[i]['template_group'] == rows[i+1]['template_group']
        assert signature(rows[i]) == signature(rows[i+1])
