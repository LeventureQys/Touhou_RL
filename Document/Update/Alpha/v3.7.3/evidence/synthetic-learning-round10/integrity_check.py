"""Postcollection integrity checks; no teacher policy rerun, no sealed worlds."""
import json, hashlib
from pathlib import Path
import numpy as np
import torch
from touhou_rl.tactical_teacher_validation import make_world, verify, advance
from touhou_rl.tactical_scenarios import initial_state
from touhou_rl.tactical_synthetic_pixels import render_state, ACTION_MAP
from touhou_rl import vision
root=Path(__file__).parent
records=json.loads((root/'data/records.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
groups={}; events=[]
for e in records['events']:
    assert e['split'] in ('train','validation')
    assert groups.setdefault(e['group'],e['split'])==e['split']
    refs={k:root/'data'/e[k]['path'] for k in ('pixels','labels','history','raw','rollout')}
    for k,p in refs.items():assert sha(p)==e[k]['sha256']
    meta=json.loads(refs['rollout'].read_text());s=make_world(meta['scenario'])
    assert s.split!='test' and not s.event_id.startswith('confirmation')
    verdict=verify(s,meta['actions']);assert (verdict=='feasible')==e['teacher_verified']
    with np.load(refs['raw']) as raw:
        assert raw['rgb'].shape==(161,480,640,3)
        native=raw['rgb'];pixels=torch.load(refs['pixels'],weights_only=True)
        state=initial_state(s)
        for j,a in enumerate(meta['actions']):
            assert np.array_equal(native[2*j],render_state(s,state))
            state=advance(s,state,a)
        assert np.array_equal(native[-1],render_state(s,state))
        for region in vision.TH08_REGIONS:
            t=pixels[region.name];assert t.dtype==torch.float32 and t.shape[0:2]==(80,3)
            assert torch.isfinite(t).all() and 0<=t.min()<=t.max()<=1
            for row in (0,1,2,40,79):
                for depth,offset in enumerate((-4,-2,0)):
                    f=max(0,row*2+offset)
                    image=vision.resize_area(vision.crop(native[f],region.box),*t.shape[2:4])
                    expected=torch.as_tensor(image.copy()).float().mul_(1/255)
                    assert torch.equal(t[row,depth],expected)
    h=torch.load(refs['history'],weights_only=True);l=torch.load(refs['labels'],weights_only=True)
    assert (h['previous_option']==3).all() and not h['target_valid'].any()
    assert h['previous_candidate'].tolist()==[18]+[ACTION_MAP[a] for a in meta['actions'][:-1]]
    assert l['action_valid'].all()==e['teacher_verified']
    assert (l['action_positive'].sum(1)==(1 if e['teacher_verified'] else 0)).all()
    events.append(dict(event_id=e['event_id'],verified=e['teacher_verified']))
audit=dict(schema='synthetic-render-integrity/1',events=events,groups=len(groups),
           full_even_state_pixels_checked=True,causal_stacks_samples_per_event=15,
           assets_sha256_checked=True,teacher_complete_verify_checked=True,
           formal_gate='NOT_OPENED',R8_test_executed=0,R9_confirmation_executed=0)
with (root/'integrity-audit.json').open('x') as f:json.dump(audit,f,indent=2)
print(json.dumps(audit,indent=2))
