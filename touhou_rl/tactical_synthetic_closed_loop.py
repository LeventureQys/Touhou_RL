"""Frozen round11 causal horizontal toy research. No training/formal-loader changes."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from . import vision
from .tactical_synthetic_pixels import render_state, SIZES, sha, write_json, selected_specs
from .tactical_teacher_validation import make_world, advance, metrics
from .tactical_scenarios import initial_state
from .tactical_policy import TacticalPolicy, TacticalConfig, bos_history
from .vmodels import load_vision_checkpoint

ROOT=Path('Document/Update/Alpha/v3.7.3/evidence')
RECORDS=ROOT/'synthetic-learning-round10/data/records.json'
HEADS=ROOT/'synthetic-learning-round10/run-s7-n1-200/research-heads.pt'
BASELINE=Path('third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt')
OUT=ROOT/'synthetic-closed-loop-round11'
# focus*9+move: stationary, right, left for each focus. Never project vertical/diagonal.
HORIZONTAL={0:0,3:2,7:-2,9:0,12:1,16:-1}
MOVES=((0,0),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1))

def train_mode(records):
    counts=Counter()
    for e in records['events']:
        if e['split']!='train': continue
        p=RECORDS.parent/e['labels']['path']
        if sha(p)!=e['labels']['sha256']:raise ValueError('training labels SHA mismatch')
        lab=torch.load(p,map_location='cpu',weights_only=True)
        for row in lab['action_positive'][lab['action_valid']]:
            if int(row.sum())!=1:raise ValueError('mode requires single route labels')
            counts[int(row.long().argmax())]+=1
    return min(counts,key=lambda c:(-counts[c],c)),dict(sorted(counts.items()))

def preregister(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    records=json.loads(RECORDS.read_text(encoding='utf-8'))
    events=[e for e in records['events'] if e['split']=='validation']
    assert len(events)==8
    mode,counts=train_mode(records)
    specs={i['spec']['event_id']:i['spec'] for i in selected_specs()}
    frozen=dict(schema='v373-synthetic-closed-loop-prereg/1',
        identities={str(p):sha(p) for p in (RECORDS,HEADS,BASELINE,Path(__file__),Path('touhou_rl/tactical_synthetic_pixels.py'),Path('touhou_rl/tactical_teacher_validation.py'),Path('touhou_rl/tactical_policy.py'))},
        events=[dict(event_id=e['event_id'],family=e['family'],group=e['group'],scenario=specs[e['event_id']],scenario_sha256=e['scenario_sha256']) for e in events],
        mode_candidate=mode,train_counts=counts,mode_source='train labels only; smallest candidate tie break',
        arms=['B0','fixed_hold','train_mode','N1'],conditions=['horizontal_masked','unmasked'],
        horizontal_candidates=HORIZONTAL,mask='Same fixed action-domain mask for every arm, before argmax; not state-dependent, not safety shield. Separate horizontal-only research.',
        candidate_schema=[dict(candidate=c,focus=c//9,move=c%9,vector=MOVES[c%9],toy_action=HORIZONTAL.get(c),supported=c in HORIZONTAL) for c in range(18)],
        invalid_action='Unmasked unsupported candidate terminates technical failure, no execution, no fallback, no shield',
        causal_stack='[max(0,f-4),max(0,f-2),f] chronological, depth3, actual executed even frames only; repeat initial padding',
        environment='advance actual candidate every 2 logical frames; horizon160; continue after collision to measure full horizon, sticky hit; collision resolution2',
        renderer='unchanged current state only; fixed HUD; no future/teacher hint',
        history='event reset hidden=None; BOS first; after execution candidate actual/hold2/delta2/optionBOS3/target_valid false; no predicted aux adopted',
        metrics='survival/full-horizon feasible goal (F3 canonical x<=goal at any actual sampled state); first collision at 2-frame resolution; invalid rate on observed decisions',
        plots='first up to3 distinct validation events per family in fixed records order, compare masked arms; actual shortages stated',
        checkpoint_selection='fixed run-s7-n1-200 only, no training/tuning; batch1 real forward',
        development_only=True,validation_previously_used_for_metrics=True,independent_test=False,formal_P5=False,T21_passed=False,game_execution=False)
    write_json(out/'preregistration.json',frozen)
    return frozen

def pixel_stack(images):
    ids=[max(0,len(images)-3),max(0,len(images)-2),len(images)-1]
    regions={name:[] for name in SIZES}
    for i in ids:
        for name,im in vision.extract_regions(images[i]).items():
            regions[name].append(vision.resize_area(im,*SIZES[name]))
    return vision.regions_to_batch({k:np.stack(v)[None] for k,v in regions.items()}),ids

def executed_history(features, previous):
    h=bos_history(features)
    if previous is not None:
        h['previous_candidate'].fill_(previous);h['previous_hold'].fill_(2)
        h['frame_delta'].fill_(2);h['history_valid'].fill_(True)
    return h

class FrozenNetworks:
    def __init__(self, baseline, heads, device):
        self.device=torch.device(device)
        self.backbone,_=load_vision_checkpoint(baseline,map_location='cpu')
        self.backbone.to(device).eval()
        payload=torch.load(heads,map_location='cpu',weights_only=True)
        if payload.get('schema')!='v373-synthetic-render-only-heads/1':raise ValueError('research schema only')
        if payload['report']['baseline_sha256']!=sha(baseline):raise ValueError('baseline binding')
        self.n1=TacticalPolicy(self.backbone,TacticalConfig(arm='N1')).to(device).eval()
        # Deliberately manual research-schema binding, not the formal checkpoint loader.
        self.n1.heads.load_state_dict(payload['heads'],strict=True)
    @torch.inference_mode()
    def __call__(self, arm, regions, previous, hidden, reset):
        regions={k:v.to(self.device) for k,v in regions.items()}
        core=self.backbone.forward_core(regions)
        f=core['features'][:,None];base=core['policy_logits'][:,None]
        if arm=='B0':return base[0,0].cpu(),None
        result=self.n1.forward_features(f,base,executed_history(f,previous),hidden,
            reset_mask=torch.tensor([[reset]],device=self.device))
        return result['policy_logits'][0,0].cpu(),result['hidden']

def run_event(s, arm, condition, network, mode, directory=None):
    state=initial_state(s);images=[render_state(s,state)];steps=[];actions=[]
    hidden=None;previous=None;first_collision=None;invalid=False
    positions=[dict(frame=0,x=state.x,hit=False)]
    while state.frame<s.horizon:
        regions,ids=pixel_stack(images)
        reset=state.frame==0
        if arm in ('fixed_hold','train_mode'):
            logits=torch.full((18,),-1e9);logits[0 if arm=='fixed_hold' else mode]=0
            next_hidden=None
        else:logits,next_hidden=network(arm,regions,previous,hidden,reset)
        if logits.shape!=(18,) or not torch.isfinite(logits).all():raise ValueError('invalid network logits')
        raw=int(logits.argmax());selected=logits.clone()
        if condition=='horizontal_masked':
            selected[[c for c in range(18) if c not in HORIZONTAL]]=-torch.inf
        candidate=int(selected.argmax());action=HORIZONTAL.get(candidate)
        row=dict(frame=state.frame,pixels=dict(path='pixels.npz',key='rgb',indices=ids,logical_frames=[2*i for i in ids]),
            hidden_reset=reset,history=dict(previous_candidate=18 if previous is None else previous,previous_option=3,previous_hold=0 if previous is None else 2,frame_delta=0 if previous is None else 2,target_valid=False,history_valid=previous is not None),
            raw_unmasked_candidate=raw,raw_unsupported=raw not in HORIZONTAL,candidate=candidate,
            actual_candidate=None,actual_action=None,actual_hold=0,x_before=state.x,x_after=state.x,
            invalid_action=action is None,fallback=False,first_collision_frame=first_collision)
        if action is None:
            invalid=True;steps.append(row);break
        before=state
        state=advance(s,state,action);actions.append(action)
        if state.hit and not before.hit:first_collision=state.frame
        row.update(actual_candidate=candidate,actual_action=action,actual_hold=state.frame-before.frame,
            x_after=state.x,first_collision_frame=first_collision,hit=state.hit)
        steps.append(row);positions.append(dict(frame=state.frame,x=state.x,hit=state.hit))
        previous=candidate;hidden=next_hidden;images.append(render_state(s,state))
    complete=state.frame==s.horizon and not invalid
    goal=s.family!='F3' or any(s.canonical_x(t['x'])<=s.goal for t in positions)
    physical=metrics(s,actions) if complete else None
    result=dict(event_id=s.event_id,family=s.family,arm=arm,condition=condition,steps=steps,
        trace=positions,first_collision_frame=first_collision,collision_time_resolution_frames=2,
        full_horizon=complete,goal_required=s.family=='F3',goal_satisfied=goal,
        survived=complete and not state.hit,feasible=complete and not state.hit and goal,
        technical_failure='invalid_action' if invalid else None,invalid_action=invalid,fallback_count=0,
        raw_unsupported_count=sum(r['raw_unsupported'] for r in steps),decisions=len(steps),
        environment_metrics=physical)
    if directory:
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(directory/'pixels.npz',rgb=np.stack(images),frames=np.arange(len(images))*2)
        write_json(directory/'trace.json',result)
    return result

def plot_event(event, rows, path):
    im=Image.new('RGB',(1000,600),'white');d=ImageDraw.Draw(im)
    d.text((25,15),event+' / horizontal-only masked / frozen research, NOT game',fill='black')
    d.rectangle((60,50,860,435),outline='black')
    for row,color in zip(rows,['#3366cc','#666666','#cc9900','#159447']):
        pts=[(60+5*t['frame'],435-t['x']) for t in row['trace']]
        if len(pts)>1:d.line(pts,fill=color,width=3)
        k=['B0','fixed_hold','train_mode','N1'].index(row['arm'])
        d.text((60,460+25*k),f"{row['arm']}: feasible={row['feasible']}; collision={row['first_collision_frame']}; goal={row['goal_satisfied']}",fill=color)
    d.text((60,440),'x: logical frame 0..160; y: actual player x (not canonical)',fill='black')
    im.save(path)

def run(out=OUT,device='cuda'):
    out=Path(out);p=json.loads((out/'preregistration.json').read_text(encoding='utf-8'))
    for file,digest in p['identities'].items():
        if sha(file)!=digest:raise ValueError('preregistered identity changed: '+file)
    dest=out/'run';dest.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    network=FrozenNetworks(BASELINE,HEADS,device)
    rows=[]
    for condition in p['conditions']:
        for e in p['events']:
            s=make_world(e['scenario'])
            for arm in p['arms']:
                r=run_event(s,arm,condition,network,p['mode_candidate'],dest/condition/e['event_id']/arm)
                rows.append(r);print(condition,e['event_id'],arm,r['feasible'],r['technical_failure'],flush=True)
    summaries=[]
    for condition in p['conditions']:
        for family in ['all','F1','F2','F3']:
            for arm in p['arms']:
                group=[r for r in rows if r['condition']==condition and r['arm']==arm and (family=='all' or r['family']==family)]
                summaries.append(dict(condition=condition,family=family,arm=arm,events=len(group),
                    feasible=sum(r['feasible'] for r in group),survived=sum(r['survived'] for r in group),
                    full_horizon=sum(r['full_horizon'] for r in group),invalid_action=sum(r['invalid_action'] for r in group),
                    raw_unsupported_count=sum(r['raw_unsupported_count'] for r in group),decisions=sum(r['decisions'] for r in group)))
    plots={};figdir=dest/'figures';figdir.mkdir()
    for family in ['F1','F2','F3']:
        events=[e['event_id'] for e in p['events'] if e['family']==family][:3];plots[family]=events
        for event in events:plot_event(event,[r for r in rows if r['event_id']==event and r['condition']=='horizontal_masked'],figdir/(event+'.png'))
    result=dict(schema='v373-synthetic-closed-loop-research/1',device=device,batch_size=1,
        prereg_sha256=sha(out/'preregistration.json'),summaries=summaries,plot_events=plots,
        independent_test=False,development_only=True,formal_P5=False,T21_passed=False,training_steps=0,
        identities_unchanged=all(sha(f)==h for f,h in p['identities'].items()))
    write_json(dest/'summary.json',result)
    write_json(dest/'sha256.json',{str(f.relative_to(dest)):sha(f) for f in dest.rglob('*') if f.is_file()})
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preregister','run']);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--device',default='cuda');a=p.parse_args()
    preregister(a.out) if a.command=='preregister' else run(a.out,a.device)
if __name__=='__main__':main()
