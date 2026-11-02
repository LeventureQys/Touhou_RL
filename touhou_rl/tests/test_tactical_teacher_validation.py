"""Offline contracts for R8. No sealed test world is executed."""
import json
from dataclasses import replace
import pytest
from touhou_rl import tactical_teacher_validation as v


def world(family='F1',split='development',index=0):
    return v.make_world([d for d in v.definitions() if d['family']==family and d['split']==split][index])


def test_split_groups_and_mirrors_are_disjoint():
    groups={}; signatures={}
    for d in v.definitions():
        g=d['template_group']; groups.setdefault(g,[]).append(d)
    for g,rows in groups.items():
        assert len(rows)==2 and {r['mirror'] for r in rows}=={True,False}
        assert len({r['split'] for r in rows})==1
        assert rows[0]['start_x']+rows[1]['start_x']==384
        p={k:z for k,z in rows[0].items() if k not in ('event_id','template_group','seed','split','mirror')}
        sig=json.dumps(p,sort_keys=True)
        assert sig not in signatures
        signatures[sig]=rows[0]['split']


def test_test_is_manifest_only():
    d=next(d for d in v.definitions() if d['split']=='test')
    with pytest.raises(ValueError,match='sealed test'): v.make_world(d)


def test_f2_actual_player_dependency_and_mirror():
    s=world('F2','validation',4)
    a=v.replay(s,[0]*20)[0]; b=v.replay(s,[2]*20)[0]
    assert a.emissions[0]['aim_x']==b.emissions[0]['aim_x']
    assert a.emissions[1]['aim_x']!=b.emissions[1]['aim_x']
    m=replace(s,mirror=True,start_x=384-s.start_x)
    sm=v.replay(m,[-2]*20)[0]
    assert sm.x==384-b.x
    assert len(sm.emissions)==len(b.emissions)
    for e,em in zip(b.emissions,sm.emissions):
        assert em['aim_x']==pytest.approx(384-e['aim_x'])
        assert em['vx']==pytest.approx(-e['vx'])


def test_full_replay_collision_and_incomplete_unknown():
    s=world()
    assert v.verify(s,[0])=='unknown'
    assert v.verify(s,[0]*80)=='rejected_witness'
    r=v.solve(s)
    assert r['status']=='feasible'
    assert len(r['actions'])==80 and v.verify(s,r['actions'])=='feasible'
    assert all(t['actual_hold']==2 for t in v.replay(s,r['actions'])[1][1:])


def test_f3_goal_not_survival_and_pressure():
    s=world('F3')
    assert s.depletion_pressure
    assert v.verify(s,[0]*80)=='rejected_witness'
    assert v.verify(s,[-2]*80)=='rejected_witness'
    assert v.verify(replace(s,depletion_pressure=False,goal=0.),[0]*80)!='feasible'
    r=v.solve(s)
    assert r['status']=='feasible' and v.verify(s,r['actions'])=='feasible'


def test_each_positive_delay_contains_full_correct_prefix():
    s=world()
    for prefix in (0,-2):
        for delay in (0,2,12,48,80,160):
            r=v.solve(s,delay,prefix)
            if r['status']=='feasible':
                assert r['actions'][:delay//2]==[prefix]*(delay//2)
                assert len(r['actions'])==80
                assert v.verify(s,r['actions'])=='feasible'
            else:
                assert r['status']=='unknown' and not r['actions']
    with pytest.raises(ValueError): v.solve(s,3)


def test_nonmonotone_windows_do_not_fill_unknown():
    e=[dict(delay=d,status='feasible' if d in (0,2,8) else 'unknown') for d in range(0,12,2)]
    assert v.windows(e)==[[0,2],[8,8]]


def test_old_teacher_geometry_failure_is_not_repaired():
    s=world('F1','validation',6)
    actions,_=v.rollout(s)
    assert v.verify(s,actions)=='rejected_witness'
    assert v.solve(s)['status']=='feasible'


def test_preregistration_immutable_and_teacher_bound(tmp_path):
    manifest=v.preregister(tmp_path)
    assert manifest['teacher_sha256']==v.source_hash()
    assert sum(d['split']=='test' for d in manifest['groups'])==24
    with pytest.raises(FileExistsError): v.preregister(tmp_path)
