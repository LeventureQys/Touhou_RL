"""Explicit synthetic-rendered pixel learning diagnostic, NOT formal T21/training.

Cannot consume real manifest or produce deployable tactical checkpoint schema.
No formal gate status is changed by this separate research-only experiment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from touhou_rl.vmodels import load_vision_checkpoint
from touhou_rl.tactical_policy import TacticalPolicy,TacticalConfig,tactical_policy_loss
from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_metrics import event_metrics,summarize_events
from touhou_rl.tactical_feature_cache import _fp32_mode

SCHEMA='v373-synthetic-rendered-diagnostic/1'


def run(records,baseline,out,steps=200,device='cuda',init_checkpoint=None):
    records=Path(records);out=Path(out)
    if out.exists():raise FileExistsError('new diagnostic run required')
    if not 1<=steps<=2000:raise ValueError('bounded diagnostic steps required')
    spec=json.loads(records.read_text(encoding='utf-8'))
    if spec.get('schema')!=SCHEMA or spec.get('real_pixel_training',False):
        raise ValueError('only explicitly synthetic renderer diagnostic accepted')
    entries=spec.get('events',[])
    if not entries or any(e.get('split') not in ('train','validation') for e in entries):
        raise ValueError('synthetic train/validation events required')
    if {e['split'] for e in entries}!={'train','validation'}:
        raise ValueError('both grouped splits required')
    torch.manual_seed(7);torch.set_num_threads(4)
    backbone,_=load_vision_checkpoint(baseline,map_location='cpu')
    backbone=backbone.to(device).eval()
    data={'train':[],'validation':[]};groups={}
    begin=time.perf_counter()
    for entry in spec['events']:
        group=entry['group'];split=entry['split']
        if split not in data:raise ValueError('no test or other split in this diagnostic')
        if group in groups and groups[group]!=split:raise ValueError('template leakage')
        groups[group]=split
        if not entry['teacher_verified']:continue
        loaded={}
        for key in ('pixels','history','labels'):
            ref=entry[key];path=(records.parent/ref['path']).resolve()
            if not path.is_relative_to(records.parent.resolve()) or file_sha(path)!=ref['sha256']:
                raise ValueError('diagnostic asset identity mismatch')
            loaded[key]=torch.load(path,map_location='cpu',weights_only=True)
        pixels=loaded.pop('pixels');features=[];logits=[]
        with _fp32_mode(torch.device(device)),torch.inference_mode():
            for start in range(0,entry['rows'],16):
                core=backbone.forward_core({k:v[start:start+16].to(device) for k,v in pixels.items()})
                features.append(core['features'].cpu().clone());logits.append(core['policy_logits'].cpu().clone())
        # New normal tensors to avoid inference tensor restrictions in autograd.
        loaded.update(features=torch.cat(features).clone().to(device),base_logits=torch.cat(logits).clone().to(device),event_id=entry['event_id'])
        loaded['history']={k:v.to(device) for k,v in loaded['history'].items()}
        loaded['labels']={k:v.to(device) for k,v in loaded['labels'].items()}
        data[split].append(loaded)
    if min(len(v) for v in data.values())<1:raise ValueError('grouped train/validation events required')
    encode_seconds=time.perf_counter()-begin
    model=TacticalPolicy(backbone,TacticalConfig(arm='N1')).to(device)
    init_sha=None
    if init_checkpoint is not None:
        payload=torch.load(init_checkpoint,map_location='cpu',weights_only=True)
        if payload.get('schema')!='v373-synthetic-render-only-heads/1' or payload['report']['baseline_sha256']!=file_sha(baseline):
            raise ValueError('synthetic research warm start identity mismatch')
        model.heads.load_state_dict(payload['heads'],strict=True)
        init_sha=file_sha(init_checkpoint)
    for events in data.values():
        for e in events:
            with torch.no_grad():
                check=model.forward_features(e['features'][None],e['base_logits'][None],
                    {k:v[None] for k,v in e['history'].items()})
                tactical_policy_loss(check,{k:v[None] for k,v in e['labels'].items()})
    frozen={n:p.detach().clone() for n,p in backbone.named_parameters()}
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    def infer(event,grad=False):
        # Full causal 80step event for diagnostic memory; no future frames as inputs.
        return model.forward_features(event['features'][None],event['base_logits'][None],
            {k:v[None] for k,v in event['history'].items()})
    def evaluate(split):
        model.eval();rows=[]
        with torch.no_grad():
            for e in data[split]:
                labels={k:v[None] for k,v in e['labels'].items()}
                rows.append(event_metrics(infer(e),labels))
        return summarize_events(rows)
    initial={'train':evaluate('train'),'validation':evaluate('validation')}
    losses=[];start_time=time.perf_counter()
    for step in range(steps):
        model.train();e=data['train'][step%len(data['train'])]
        labels={k:v[None] for k,v in e['labels'].items()}
        result=tactical_policy_loss(infer(e),labels)
        if not result['has_supervision']:continue
        optimizer.zero_grad(set_to_none=True);result['loss'].backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step();losses.append(float(result['loss'].detach()))
        if step%25==0:print(json.dumps({'step':step,'loss':losses[-1]}),flush=True)
    if device.startswith('cuda'):torch.cuda.synchronize()
    elapsed=time.perf_counter()-start_time
    final={'train':evaluate('train'),'validation':evaluate('validation')}
    if not all(torch.equal(p,frozen[n]) for n,p in backbone.named_parameters()):raise AssertionError('backbone changed')
    out.mkdir(parents=True)
    report={'schema':'v373-synthetic-pixel-learning/1','source_sha256':file_sha(records),
        'baseline_sha256':file_sha(baseline),'init_checkpoint_sha256':init_sha,'steps':len(losses),'device':device,'seed':7,
        'encode_seconds':encode_seconds,'optimization_seconds':elapsed,'steps_per_second':len(losses)/elapsed,
        'events':{k:len(v) for k,v in data.items()},'initial':initial,'final':final,
        'first_loss':losses[0] if losses else None,'last_loss':losses[-1] if losses else None,
        'backbone_unchanged':True,'T21_passed':False,'real_pixel_training':False,
        'formal_gates_changed':False,'deployable':False,
        'warning':'Synthetic style feature separability only, teacher single-route labels; not exact viable action sets or closed-loop improvement.'}
    torch.save({'schema':'v373-synthetic-render-only-heads/1','heads':model.heads.state_dict(),
        'report':report},out/'research-heads.pt')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'losses.json').write_text(json.dumps(losses),encoding='utf-8')
    (out/'sha256.json').write_text(json.dumps({p.name:file_sha(p) for p in out.iterdir() if p.is_file()},indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('records','baseline','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=200);p.add_argument('--device',default='cuda')
    p.add_argument('--init-checkpoint',type=Path)
    a=p.parse_args();run(a.records,a.baseline,a.out,a.steps,a.device,a.init_checkpoint)


if __name__=='__main__':main()
