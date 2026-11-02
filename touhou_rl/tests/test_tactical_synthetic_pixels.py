from dataclasses import replace
import numpy as np
import pytest
import torch
from touhou_rl import tactical_synthetic_pixels as m
from touhou_rl.tactical_scenarios import initial_state, Bullet
from touhou_rl.tactical_teacher_validation import make_world, advance, verify


def test_split_locked_and_exposed_only():
    specs=m.selected_specs();assert len(specs)==16
    groups={}
    for row in specs:
        s=row['spec'];assert s['split'] in ('development','validation')
        groups.setdefault(s['template_group'],[]).append((row['split'],s['mirror']))
    assert len(groups)==8
    assert all(len(v)==2 and {x[1] for x in v}=={False,True} and len({x[0] for x in v})==1 for v in groups.values())
    assert sum(r['split']=='train' for r in specs)==8
    assert {r['spec']['family'] for r in specs if r['split']=='train'}=={'F1','F2','F3'}


def test_candidate_mapping_and_actual_history():
    assert m.ACTION_MAP=={-2:7,-1:16,0:0,1:12,2:3}
    h,l=m.history_and_labels([-2,-1,0,1,2],True)
    assert h['previous_candidate'].tolist()==[18,7,16,0,12]
    assert h['previous_hold'].tolist()==[0,2,2,2,2]
    assert h['frame_delta'].tolist()==[0,2,2,2,2]
    assert (h['previous_option']==3).all() and not h['target_valid'].any()
    assert l['action_positive'].sum(1).tolist()==[1]*5
    assert l['action_positive'].nonzero()[:,1].tolist()==[7,16,0,12,3]
    _,failed=m.history_and_labels([-2,0],False)
    assert not failed['action_positive'].any() and not failed['action_valid'].any()


def test_no_pre_wall_future_geometry_or_countdown():
    s=make_world(m.selected_specs()[0]['spec']);st=initial_state(s)
    a=m.render_state(s,st)
    future=replace(s,wall_frame=40,coverage=(50,320),internal_gap=True,gap=(210,220))
    assert np.array_equal(a,m.render_state(future,st))
    assert np.array_equal(a,m.render_state(s,replace(st,frame=12,hit=True)))
    assert a.shape==(480,640,3) and a.dtype==np.uint8
    assert not np.array_equal(a,m.render_state(s,replace(st,frame=s.wall_frame)))


def test_future_bullets_not_drawn_and_fixed_hud():
    s=make_world(m.selected_specs()[4]['spec']);st=initial_state(s)
    image=m.render_state(s,st)
    future=replace(st,bullets=(Bullet(10,192,200,0,8,192),))
    assert np.array_equal(image,m.render_state(s,future))
    changed=m.render_state(s,replace(st,x=250,frame=80))
    assert np.array_equal(image[:,424:],changed[:,424:])


def test_full_160_frame_references_exact_even_states():
    s=make_world(m.selected_specs()[4]['spec']);actions=[0]*80
    states=m.full_states(s,actions)
    assert [st.frame for st in states]==list(range(161))
    st=initial_state(s)
    for i,a in enumerate(actions):
        st=advance(s,st,a);assert states[2*i+2]==st


def test_preregister_first_and_freeze_source(tmp_path):
    with pytest.raises(FileNotFoundError):m.collect(tmp_path)
    p=m.preregister(tmp_path)
    assert p['schema']==m.SCHEMA and p['formal_gate']=='NOT_OPENED'
    with pytest.raises(FileExistsError):m.preregister(tmp_path)
