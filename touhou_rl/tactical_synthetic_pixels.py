"""R10 rendered toy-world diagnostic, explicitly incompatible with formal manifests."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from . import vision
from .tactical_scenarios import initial_state
from .tactical_teacher_v2 import rollout, hashes
from .tactical_teacher_validation import definitions, make_world, advance, verify

SCHEMA = 'v373-synthetic-rendered-diagnostic/1'
SIZES = {'playfield': (224,192), 'hud_score': (34,216), 'hud_status': (126,216)}
ACTION_MAP = {-2:7, -1:16, 0:0, 1:12, 2:3}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f: json.dump(data,f,indent=2)

def selected_specs():
    wanted = {'development-F1-structure-00':'train','development-F1-structure-01':'train',
              'development-F2-structure-00':'train','development-F3-structure-00':'train',
              'validation-F1-structure-00':'validation','validation-F2-structure-00':'validation',
              'validation-F2-structure-01':'validation','validation-F3-structure-00':'validation'}
    return [dict(spec=s, split=wanted[s['template_group']]) for s in definitions()
            if s['split'] in ('development','validation') and s['template_group'] in wanted]

def renderer_source_hash(): return sha(__file__)

def preregister(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    value=dict(schema=SCHEMA, renderer_sha256=renderer_source_hash(),teacher_source=hashes(),
        events=selected_specs(),native_shape=[480,640,3],region_sizes=SIZES,depth=3,
        history_frames='[max(0,f-4),max(0,f-2),f]; repeat initial padding',
        horizon=160,steps=80,action_map=ACTION_MAP,hold_candidate=0,
        renderer='PIL RGB; fixed background/HUD; actual player and born bullets; currently active danger only',
        hidden_future='F1 pre-wall geometry, future F3 window/goal, teacher target/option/countdown NEVER drawn',
        preregistered_limitations=['F1 imminent wall invisible before onset: identical pixels can require opposite privileged actions; not fully learnable.',
            'F3 window changes are unannounced; goal is not visible. No future hints added.'],
        labels='only successful complete teacher rollout single-route first action; NOT full viable set',
        sealed_R8_test_execution=0,R9_confirmation_execution=0,formal_gate='NOT_OPENED',real_pixel_training=False)
    write_json(out/'preregistration.json',value);return value

def render_state(s,state):
    """Only current visible geometry; never decisions, targets, future births or clocks."""
    image=Image.new('RGB',(640,480),(18,20,28)); d=ImageDraw.Draw(image)
    d.rectangle((32,16,415,463),fill=(10,14,24))
    # This toy's collision zones are physical regions only while active.
    spans=[]
    if s.family=='F1' and s.wall_frame<=state.frame<s.wall_frame+12:
        lo,hi=s.coverage
        spans=[(lo,hi)] if not s.internal_gap else [(lo,min(hi,s.gap[0])),(max(lo,s.gap[1]),hi)]
    if s.family=='F3' and s.depletion_pressure:
        if state.frame<s.window[0]:spans=[(0,s.threshold)]
        elif state.frame>=s.window[1]:spans=[(s.threshold,383)]
    for lo,hi in spans:
        if lo>hi:continue
        if s.mirror:lo,hi=384-hi,384-lo
        d.rectangle((32+max(0,lo),16,32+min(383,hi),463),fill=(95,28,42))
    for b in state.bullets:
        age=state.frame-b.born
        if age<1:continue
        x,y=b.x+b.vx*age,b.y+b.vy*age
        if 2<=x<382 and 2<=y<446:
            d.ellipse((32+x-3,16+y-3,32+x+3,16+y+3),fill=(255,205,65))
    x,y=32+state.x,416
    d.polygon([(x,y-7),(x-5,y+5),(x+5,y+5)],fill=(90,225,255))
    d.ellipse((x-1,y-1,x+1,y+1),fill=(255,255,255))
    # Fixed across events, time, hit status and labels.
    d.text((432,44),'SCORE 000000000',fill=(180,180,190))
    d.text((432,98),'PLAYER  ***\nBOMB    ***\nPOWER   100',fill=(180,180,190))
    return np.asarray(image,dtype=np.uint8).copy()

def full_states(s, actions):
    state=initial_state(s); states=[state]
    for a in actions:
        nxt=advance(s,state,a)
        # Exact intermediate geometry; births at logical frame n are visible at n+1.
        mid=replace(nxt,frame=state.frame+1,x=min(376.,max(8.,state.x+a*2.)),
                    bullets=tuple(b for b in nxt.bullets if b.born<state.frame+1),
                    emissions=tuple(e for e in nxt.emissions if e['frame']<state.frame+1))
        states.extend([mid,nxt]);state=nxt
    assert len(states)==161 and states[-1].frame==160
    return states

def history_and_labels(actions, verified):
    n=len(actions); candidates=torch.tensor([ACTION_MAP[a] for a in actions],dtype=torch.long)
    history={'previous_candidate':torch.cat([torch.tensor([18]),candidates[:-1]]),
        'previous_option':torch.full((n,),3,dtype=torch.long),
        'previous_hold':torch.tensor([0.]+[2.]*(n-1)), 'frame_delta':torch.tensor([0.]+[2.]*(n-1)),
        'previous_target':torch.zeros(n,2), 'target_valid':torch.zeros(n,dtype=torch.bool),
        'history_valid':torch.tensor([False]+[True]*(n-1))}
    positive=torch.zeros(n,18,dtype=torch.bool)
    if verified:positive[torch.arange(n),candidates]=True
    labels={'action_positive':positive,'action_valid':torch.full((n,),verified,dtype=torch.bool)}
    return history,labels

def collect(out):
    out=Path(out); prereg=json.loads((out/'preregistration.json').read_text(encoding='utf-8'))
    if prereg['renderer_sha256']!=renderer_source_hash() or prereg['teacher_source']!=hashes():
        raise ValueError('source changed after preregistration')
    data=out/'data';data.mkdir(exist_ok=False)
    entries=[];failures=[]
    for item in prereg['events']:
        s=make_world(item['spec']);actions,decisions=rollout(s)
        if len(actions)!=80:raise ValueError('incomplete teacher rollout')
        verdict=verify(s,actions);verified=verdict=='feasible';states=full_states(s,actions)
        event=data/s.event_id;event.mkdir()
        native=np.stack([render_state(s,st) for st in states])
        np.savez_compressed(event/'raw.npz',rgb=native,frames=np.arange(161,dtype=np.int32))
        # Exact causal decision frames; stored float32 uses existing region converter.
        resized={name:[] for name in SIZES}
        for f in range(0,160,2):
            for name,im in vision.extract_regions(native[f]).items():
                resized[name].append(vision.resize_area(im,*SIZES[name]))
        pixels={}
        for name,frames in resized.items():
            arr=np.stack(frames); ids=np.maximum(0,np.arange(80)[:,None]+np.array([-2,-1,0]))
            if name.startswith('hud_'):
                # Constant HUD is shared storage, not 80 redundant copies.
                one=vision.regions_to_batch({name:arr[:1,None].repeat(3,axis=1)})[name]
                pixels[name]=one.expand(80,-1,-1,-1,-1)
            else:pixels[name]=vision.regions_to_batch({name:arr[ids]})[name]
        history,labels=history_and_labels(actions,verified)
        for name,obj in [('pixels',pixels),('history',history),('labels',labels)]:torch.save(obj,event/(name+'.pt'))
        raw_meta=dict(scenario=asdict(s),actions=actions,decisions=decisions,states=[asdict(st) for st in states],
                      verdict=verdict,teacher_source=hashes(),privileged_reference_only=True)
        write_json(event/'rollout.json',raw_meta)
        def ref(name):
            p=event/name;return dict(path=p.relative_to(data).as_posix(),sha256=sha(p))
        entry=dict(event_id=s.event_id,group=s.template_group,split=item['split'],family=s.family,mirror=s.mirror,
            original_split=s.split,rows=80,logical_frames=160,teacher_verified=verified,verify_verdict=verdict,
            pixels=ref('pixels.pt'),history=ref('history.pt'),labels=ref('labels.pt'),
            raw=ref('raw.npz'),rollout=ref('rollout.json'),scenario_sha256=hashlib.sha256(json.dumps(asdict(s),sort_keys=True).encode()).hexdigest())
        entries.append(entry)
        if not verified:failures.append(dict(event_id=s.event_id,verdict=verdict))
        print(s.event_id,verdict,flush=True)
    assert hashes()==prereg['teacher_source']
    records=dict(schema=SCHEMA,events=entries,preregistration=dict(path='../preregistration.json',sha256=sha(out/'preregistration.json')),
        teacher_source=hashes(),renderer_sha256=renderer_source_hash(),region_sizes=SIZES,depth=3,
        real_pixel_training=False,formal_manifest_compatible=False,formal_gate='NOT_OPENED',
        sealed_R8_test_execution=0,R9_confirmation_execution=0,failures=failures,
        limitations=prereg['preregistered_limitations'],history_semantics='BOS row0; actual preceding candidate/hold2/delta2; option always BOS3; no teacher target/option inputs')
    write_json(data/'records.json',records);write_json(data/'failures.json',failures)
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preregister','collect']);p.add_argument('--out',required=True)
    a=p.parse_args();preregister(a.out) if a.command=='preregister' else collect(a.out)
if __name__=='__main__':main()
