"""Per-action frozen-checkpoint diagnostics, no retraining or checkpoint selection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from touhou_rl.tactical_data import file_sha
from touhou_rl.vmodels import load_vision_checkpoint
from touhou_rl.tactical_policy import TacticalPolicy,TacticalConfig
from touhou_rl.tactical_feature_cache import _fp32_mode


def class_metrics(target,predicted):
    counts=torch.bincount(target,minlength=18)
    confusion=torch.bincount(target*18+predicted,minlength=324).reshape(18,18)
    present=counts>0
    recalls=confusion.diag().float()/counts.clamp_min(1)
    moving=present.clone();moving[0]=False;moving[9]=False
    return {'rows':len(target),'accuracy':float((target==predicted).float().mean()),
        'macro_recall_present_classes':float(recalls[present].mean()) if present.any() else None,
        'moving_macro_recall':float(recalls[moving].mean()) if moving.any() else None,
        'class_support':counts.tolist(),'class_recall':[float(recalls[i]) if counts[i] else None for i in range(18)],
        'confusion':confusion.tolist()}


def run(records,baseline,checkpoint,out):
    out=Path(out)
    if out.exists():raise FileExistsError('new class diagnostic required')
    torch.set_num_threads(4);records=Path(records)
    spec=json.loads(records.read_text(encoding='utf-8'))
    if spec['schema']!='v373-synthetic-rendered-diagnostic/1':raise ValueError('synthetic only')
    backbone,_=load_vision_checkpoint(baseline,map_location='cpu');backbone=backbone.cuda().eval()
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload['schema']!='v373-synthetic-render-only-heads/1' or payload['report']['baseline_sha256']!=file_sha(baseline):
        raise ValueError('research identity mismatch')
    model=TacticalPolicy(backbone,TacticalConfig()).cuda().eval();model.heads.load_state_dict(payload['heads'])
    rows=[]
    for e in spec['events']:
        if e['split']!='validation':continue
        data={}
        for key in ('pixels','history','labels'):
            ref=e[key];p=records.parent/ref['path']
            if file_sha(p)!=ref['sha256']:raise ValueError('asset drift')
            data[key]=torch.load(p,weights_only=True)
        with _fp32_mode(torch.device('cuda')),torch.inference_mode():
            feats=[];base=[]
            for start in range(0,e['rows'],16):
                c=backbone.forward_core({k:v[start:start+16].cuda() for k,v in data['pixels'].items()})
                feats.append(c['features']);base.append(c['policy_logits'])
            result=model.forward_features(torch.cat(feats)[None],torch.cat(base)[None],{k:v[None].cuda() for k,v in data['history'].items()})
            valid=data['labels']['action_valid'];target=data['labels']['action_positive'][valid].long().argmax(-1)
            pred=result['policy_logits'][0].argmax(-1).cpu()[valid]
        rows.append({'event_id':e['event_id'],'family':e['family'],'target':target.tolist(),'predicted':pred.tolist()})
    summaries={}
    for family in ('F1','F2','F3','all'):
        rr=[r for r in rows if family=='all' or r['family']==family]
        summaries[family]=class_metrics(torch.tensor([x for r in rr for x in r['target']]),torch.tensor([x for r in rr for x in r['predicted']]))
    report={'schema':'v373-peraction-diagnostic/1','checkpoint_sha256':file_sha(checkpoint),
        'records_sha256':file_sha(records),'summary':summaries,'events':rows,
        'updates':0,'warning':'Offline single-route label recall is not necessary safety action recall or survival.'}
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:{n:v for n,v in x.items() if n not in ('confusion','class_support','class_recall')} for k,x in summaries.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('records','baseline','checkpoint','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.records,a.baseline,a.checkpoint,a.out)
