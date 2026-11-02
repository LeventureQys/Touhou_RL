"""Offline controls for fixed synthetic checkpoint; no training or selection."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from touhou_rl.tactical_data import file_sha
from touhou_rl.vmodels import load_vision_checkpoint
from touhou_rl.tactical_policy import TacticalPolicy,TacticalConfig
from touhou_rl.tactical_feature_cache import _fp32_mode


def run(records,baseline,checkpoint,out):
    out=Path(out)
    if out.exists():raise FileExistsError('new control report required')
    torch.set_num_threads(4)
    records=Path(records);spec=json.loads(records.read_text(encoding='utf-8'))
    if spec['schema']!='v373-synthetic-rendered-diagnostic/1':raise ValueError('synthetic only')
    backbone,_=load_vision_checkpoint(baseline,map_location='cpu');backbone=backbone.cuda().eval()
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload['schema']!='v373-synthetic-render-only-heads/1':raise ValueError('research schema only')
    if payload['report']['baseline_sha256']!=file_sha(baseline):raise ValueError('baseline mismatch')
    model=TacticalPolicy(backbone,TacticalConfig()).cuda().eval();model.heads.load_state_dict(payload['heads'])
    counts=Counter()
    for e in spec['events']:
        if e['split']=='train':
            ref=e['labels'];p=records.parent/ref['path']
            if file_sha(p)!=ref['sha256']:raise ValueError('label sha')
            l=torch.load(p,weights_only=True)
            counts.update(l['action_positive'][l['action_valid']].long().argmax(-1).tolist())
    mode=counts.most_common(1)[0][0];rows=[]
    for e in spec['events']:
        data={}
        for key in ('pixels','labels','history'):
            ref=e[key];p=records.parent/ref['path']
            if file_sha(p)!=ref['sha256']:raise ValueError('asset sha')
            data[key]=torch.load(p,weights_only=True)
        feats=[];base=[]
        with _fp32_mode(torch.device('cuda')),torch.inference_mode():
            for start in range(0,e['rows'],16):
                core=backbone.forward_core({k:v[start:start+16].cuda() for k,v in data['pixels'].items()})
                feats.append(core['features']);base.append(core['policy_logits'])
            f=torch.cat(feats)[None];b=torch.cat(base)[None]
            hist={k:v[None].cuda() for k,v in data['history'].items()}
            predicted=model.forward_features(f,b,hist)['policy_logits'].argmax(-1)[0].cpu()
            nohist=model.forward_features(f,b)['policy_logits'].argmax(-1)[0].cpu()
            # Counterfactual sensory ablation only, not a valid trajectory.
            static=model.forward_features(f[:,:1].expand_as(f),b[:,:1].expand_as(b))['policy_logits'].argmax(-1)[0].cpu()
        positive=data['labels']['action_positive'];valid=data['labels']['action_valid']
        def metric(chosen):
            return float(positive.gather(1,chosen[:,None]).squeeze(1)[valid].float().mean())
        rows.append({'event_id':e['event_id'],'family':e['family'],'split':e['split'],'rows':int(valid.sum()),
            'B0':metric(b.argmax(-1)[0].cpu()),'N1':metric(predicted),'N1_no_actual_history':metric(nohist),
            'N1_static_first_visual_no_history':metric(static),'training_mode':metric(torch.full_like(predicted,mode)),
            'hold':metric(torch.zeros_like(predicted))})
    summary={}
    names=('B0','N1','N1_no_actual_history','N1_static_first_visual_no_history','training_mode','hold')
    for split in ('train','validation'):
        summary[split]={}
        for family in ('F1','F2','F3','all'):
            rr=[r for r in rows if r['split']==split and (family=='all' or r['family']==family)]
            summary[split][family]={'events':len(rr),**{n:sum(r[n] for r in rr)/len(rr) if rr else None for n in names}}
    report={'schema':'v373-synthetic-offline-controls/1','checkpoint_sha256':file_sha(checkpoint),
        'records_sha256':file_sha(records),'training_mode_candidate':mode,'training_action_counts':dict(counts),
        'summary':summary,'events':rows,'training_updates':0,'formal_gate_changed':False,
        'warning':'No-history/static are offline ablations, not matched retraining or valid closed-loop trajectories. No safety claim.'}
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('records','baseline','checkpoint','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.records,a.baseline,a.checkpoint,a.out)
