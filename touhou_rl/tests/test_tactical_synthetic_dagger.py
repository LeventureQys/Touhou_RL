from dataclasses import replace
import json
import pytest
import torch
from touhou_rl import tactical_synthetic_dagger as m
from touhou_rl.tactical_teacher_validation import make_world
from touhou_rl.tactical_synthetic_closed_loop import run_event


def test_original_train_only_limits_and_groups():
    rows=m.train_specs()
    assert len(rows)==8
    assert all(r['split']=='train' and r['spec']['split']=='development' for r in rows)
    assert len({r['spec']['template_group'] for r in rows})==4
    assert {f:sum(r['spec']['family']==f for r in rows) for f in ('F1','F2','F3')}=={'F1':4,'F2':2,'F3':2}


def test_prefix_never_self_supervised_and_actual_history():
    h,l=m.mixed_history_labels([0,0,-1,2],2,True,[9,0,16,3])
    assert h['previous_candidate'].tolist()==[18,9,0,16]
    assert (h['previous_option']==3).all() and not h['target_valid'].any()
    assert h['previous_hold'].tolist()==[0,2,2,2]
    assert l['action_valid'].tolist()==[False,False,True,True]
    assert not l['action_positive'][:2].any()
    assert l['action_positive'].nonzero().tolist()==[[2,16],[3,3]]
    _,failed=m.mixed_history_labels([0,0,-1,2],2,False)
    assert not failed['action_valid'].any() and not failed['action_positive'].any()


class Stay:
    def __call__(self,*args):
        logits=torch.zeros(18);logits[9]=1
        return logits,None


def test_rescue_earliest_window_and_fixed_prefix_independent_replay():
    s=make_world(m.train_specs()[0]['spec'])
    trace=run_event(s,'N1','horizontal_masked',Stay(),None)
    assert trace['first_collision_frame'] is not None
    r=m.rescue_event(s,trace)
    assert r['rescue_start_frame']==trace['first_collision_frame']-16
    original=[x['actual_action'] for x in trace['steps']]
    assert r['prefix_actions']==original[:r['prefix_rows']]
    assert r['actions']==r['prefix_actions']+r['suffix_actions']
    result,_=m.independent_replay(s,r['actions'])
    assert result==r['independent_verification']
    assert r['network_score'] is False and len(r['actions'])==80
    assert r['first_N1_collision_logical']<=r['first_N1_collision_frame']


def test_no_collision_no_teacher_labels(monkeypatch):
    s=make_world(next(r['spec'] for r in m.train_specs() if r['spec']['family']=='F2'))
    s=replace(s,bullet_speed=.1)
    trace=run_event(s,'N1','horizontal_masked',Stay(),None)
    assert trace['first_collision_frame'] is None
    monkeypatch.setattr(m,'TacticalTeacherV2',lambda:pytest.fail('no collision must not invoke teacher'))
    r=m.rescue_event(s,trace)
    assert not r['rescue_attempted'] and not r['rescued'] and r['prefix_rows']==80


def test_failed_rescue_retained_without_labels(monkeypatch):
    s=make_world(m.train_specs()[0]['spec'])
    trace=run_event(s,'N1','horizontal_masked',Stay(),None)
    class BadTeacher:
        def decide(self,*args):
            from touhou_rl.tactical_teacher import Decision
            return Decision(0,'HOLD_STREAM','test')
    monkeypatch.setattr(m,'TacticalTeacherV2',BadTeacher)
    r=m.rescue_event(s,trace)
    assert r['rescue_attempted'] and not r['rescued']
    assert r['independent_verification']['status']=='rejected_witness'
    _,labels=m.mixed_history_labels(r['actions'],r['prefix_rows'],r['rescued'])
    assert not labels['action_valid'].any()


def test_preregister_required_and_immutable(tmp_path,monkeypatch):
    with pytest.raises(FileNotFoundError):m.collect(tmp_path,'cpu')
    p=m.preregister(tmp_path)
    assert len(p['events'])==8 and p['validation_asset_loads']==0
    assert p['collection_checkpoint_sha']==m.sha(m.HEADS)
    with pytest.raises(FileExistsError):m.preregister(tmp_path)
    p['identities'][str(m.HEADS)]='wrong'
    (tmp_path/'preregistration.json').write_text(json.dumps(p),encoding='utf-8')
    with pytest.raises(ValueError,match='identity changed'):m.collect(tmp_path,'cpu')
