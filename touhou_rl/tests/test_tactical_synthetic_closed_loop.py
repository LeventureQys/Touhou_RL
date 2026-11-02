import numpy as np
import torch
from dataclasses import replace
from touhou_rl import tactical_synthetic_closed_loop as cl
from touhou_rl.tactical_teacher_validation import definitions, make_world
from touhou_rl.tactical_synthetic_pixels import render_state
from touhou_rl.tactical_scenarios import initial_state


def world(family='F2'):
    return make_world(next(s for s in definitions() if s['split']=='validation' and s['family']==family and not s['mirror']))


class Stub:
    def __init__(self,candidate):self.candidate=candidate;self.calls=[]
    def __call__(self,arm,regions,previous,hidden,reset):
        self.calls.append((previous,hidden,reset,{k:v.clone() for k,v in regions.items()}))
        logits=torch.zeros(18);logits[self.candidate]=10
        return logits,torch.tensor(len(self.calls))


def test_actual_candidate_changes_world_and_future_pixels():
    s=replace(world(),horizon=40)
    left=Stub(7);right=Stub(3)
    a=cl.run_event(s,'N1','horizontal_masked',left,0)
    b=cl.run_event(s,'N1','horizontal_masked',right,0)
    assert a['trace'][1]['x']==s.start_x-8
    assert b['trace'][1]['x']==s.start_x+8
    assert a['environment_metrics']['emission_events'][0]['aim_x']==s.start_x
    assert not torch.equal(left.calls[1][3]['playfield'],right.calls[1][3]['playfield'])
    assert a['steps'][1]['history']['previous_candidate']==7
    assert a['steps'][1]['pixels']['logical_frames']==[0,0,2]
    assert a['steps'][2]['pixels']['logical_frames']==[0,2,4]


def test_unmasked_invalid_terminates_without_fallback_or_horizontal_projection():
    stub=Stub(2)
    result=cl.run_event(world(),'N1','unmasked',stub,0)
    assert result['technical_failure']=='invalid_action'
    assert len(result['trace'])==1 and len(result['steps'])==1
    assert result['steps'][0]['actual_candidate'] is None
    assert not result['full_horizon'] and result['fallback_count']==0
    assert not result['feasible']


def test_fixed_domain_mask_not_state_dependent_shield():
    class Two(Stub):
        def __call__(self,*args):
            logits,hidden=super().__call__(*args);logits[7]=9
            return logits,hidden
    r=cl.run_event(replace(world(),horizon=2),'B0','horizontal_masked',Two(2),0)
    assert r['steps'][0]['raw_unmasked_candidate']==2
    assert r['steps'][0]['candidate']==7 and r['steps'][0]['actual_action']==-2
    assert not r['steps'][0]['fallback']


def test_focus_hold_preserves_candidate_and_aux_bos_event_reset():
    stub=Stub(9);s=replace(world(),horizon=4)
    first=cl.run_event(s,'N1','horizontal_masked',stub,0)
    cl.run_event(s,'N1','horizontal_masked',stub,0)
    assert first['steps'][1]['actual_candidate']==9
    assert first['steps'][1]['history']==dict(previous_candidate=9,previous_option=3,
        previous_hold=2,frame_delta=2,target_valid=False,history_valid=True)
    assert stub.calls[0][1] is None and stub.calls[0][2]
    assert stub.calls[2][1] is None and stub.calls[2][2]
    h=cl.executed_history(torch.zeros(1,1,256),9)
    assert h['previous_candidate'].item()==9 and h['previous_option'].item()==3
    assert not h['target_valid'].item() and h['previous_target'].count_nonzero()==0


def test_causal_padding_matches_dataset_regions_and_hud_fixed():
    s=world();im=render_state(s,initial_state(s))
    regions,ids=cl.pixel_stack([im])
    assert ids==[0,0,0]
    assert all(v.shape[1]==3 for v in regions.values())
    assert torch.equal(regions['playfield'][:,0],regions['playfield'][:,2])
    later=render_state(s,replace(initial_state(s),frame=40,x=100))
    r2,_=cl.pixel_stack([im,later])
    for key in ['hud_score','hud_status']:assert torch.equal(regions[key],r2[key])


def test_full_horizon_after_collision_not_early_success():
    s=world('F1');r=cl.run_event(s,'fixed_hold','horizontal_masked',None,0)
    assert r['first_collision_frame']==s.wall_frame
    assert r['full_horizon'] and len(r['steps'])==80
    assert not r['survived'] and not r['feasible']
