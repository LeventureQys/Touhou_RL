"""Round14 matched fresh-arm synthetic diagnostic; never formal training/deployment."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_policy import TacticalPolicy, TacticalConfig, tactical_policy_loss
from touhou_rl.tactical_feature_cache import _fp32_mode
from touhou_rl.vmodels import load_vision_checkpoint
from touhou_rl.tactical_synthetic_closed_loop import run_event, executed_history
from touhou_rl.tactical_teacher_validation import make_world

ROOT=Path('Document/Update/Alpha/v3.7.3/evidence')
OUT=ROOT/'arms-round14'
RECORDS=ROOT/'synthetic-learning-round10/data/records.json'
BASELINE=Path('third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt')
SCHEMA='v373-arm-diagnostic-heads/1'
ARMS=('B1','N0','N1')
CLASSES=(0,7,3,16,12)
NAMES=('hold','left','right','leftfocus','rightfocus')


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)


def validate_spec(spec):
    if spec.get('schema')!='v373-synthetic-rendered-diagnostic/1' or spec.get('real_pixel_training',False):
        raise ValueError('synthetic original records only')
    events=spec['events']
    if len(events)!=16 or any(sum(e['split']==s for e in events)!=8 for s in ('train','validation')):
        raise ValueError('original train8 validation8 only; no test')
    groups={}
    for e in events:
        if e['rows']!=80 or not e['teacher_verified']:raise ValueError('full verified 80step required')
        if e['group'] in groups and groups[e['group']]!=e['split']:raise ValueError('group leakage')
        groups[e['group']]=e['split']
    return events


def identities():
    spec=json.loads(RECORDS.read_text(encoding='utf-8'));events=validate_spec(spec)
    paths=[RECORDS,BASELINE,Path(__file__),Path('touhou_rl/tests/test_v373_arm_diagnostic.py'),
        Path('touhou_rl/tactical_policy.py'),Path('touhou_rl/tactical_feature_cache.py'),
        Path('touhou_rl/tactical_synthetic_closed_loop.py'),Path('touhou_rl/tactical_synthetic_pixels.py'),
        Path('touhou_rl/tactical_teacher_validation.py'),Path('touhou_rl/tactical_scenarios.py'),
        Path('touhou_rl/vision.py'),Path('touhou_rl/vmodels.py'),
        ROOT/'synthetic-closed-loop-round11/preregistration.json']
    for e in events:
        for key in ('pixels','history','labels'):
            ref=e[key];p=(RECORDS.parent/ref['path']).resolve()
            if not p.is_relative_to(RECORDS.parent.resolve()) or file_sha(p)!=ref['sha256']:raise ValueError('asset SHA')
            paths.append(p)
    return {str(p):file_sha(p) for p in paths}


def preregister(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    old=json.loads((ROOT/'synthetic-closed-loop-round11/preregistration.json').read_text(encoding='utf-8'))
    p=dict(schema='v373-arms-prereg/1',identities=identities(),arms=list(ARMS),seed=7,
        initialization='Each arm reset torch seed7 immediately before TacticalPolicy construction, same frozen backbone; fresh default random heads including standard zero residual, no checkpoint/warmstart.',
        train=dict(updates=200,optimizer='Adam',lr=.001,clip=1,sequence=80,batch=1,order='original train event order, step modulo8',validation_selection=False),
        precision='FP32 no AMP, matmul TF32 false, cuDNN TF32 false, float32 precision highest for encoding/training/evaluation/closed-loop',
        features='one genuine CUDA batch16 encode pass over original16x80 rows, shared RAM tensors only; no disk feature cache',
        closed_loop=dict(events=old['events'],condition='horizontal_masked',history='R11 executed_history and actual renderer; independent event hidden reset; sticky hit; continue horizon',network='arm-aware custom callback, manual research head loading, never FrozenNetworks'),
        action_metrics=dict(labels=list(NAMES),candidates=list(CLASSES),other_prediction_column=True,macro='unweighted recall over supported true classes, report null absent classes',offline='unmasked18 and fixed horizontal masked; initial/final each train/validation and family'),
        B0='Existing R11 reference2/8 only; no repeat. Different TF32 precision forbids direct pure-structure causal claim.',
        independent_test=False,development_only=True,dagger=False,game=False,formal_training=False,shader=False,teacher_calls=False,
        failure_policy='keep all outputs; exclusive x writes; one fixed training run, no extra epochs or val tuning; freeze before tests and run')
    write_json(out/'preregistration.json',p);return p


def verify(out):
    p=json.loads((Path(out)/'preregistration.json').read_text(encoding='utf-8'))
    for path,digest in p['identities'].items():
        if file_sha(path)!=digest:raise ValueError('frozen identity changed: '+path)
    return p


def confusion(truth,pred):
    matrix=[[0]*6 for _ in CLASSES]
    for q,r in zip(truth,pred):
        if q not in CLASSES:raise ValueError('unexpected true class')
        matrix[CLASSES.index(q)][CLASSES.index(r) if r in CLASSES else 5]+=1
    support=[sum(r) for r in matrix]
    recall=[matrix[i][i]/n if n else None for i,n in enumerate(support)]
    observed=[r for r in recall if r is not None]
    return dict(true_classes=list(NAMES),predicted_classes=list(NAMES)+['other'],matrix=matrix,support=support,
        recall=recall,macro_balanced_recall=sum(observed)/len(observed) if observed else None,
        agreement=sum(matrix[i][i] for i in range(5))/sum(support) if sum(support) else None)


def encode(backbone,spec,device):
    data=[]
    for e in validate_spec(spec):
        tensors={}
        for key in ('pixels','history','labels'):
            ref=e[key];p=RECORDS.parent/ref['path']
            if file_sha(p)!=ref['sha256']:raise ValueError('asset identity')
            tensors[key]=torch.load(p,map_location='cpu',weights_only=True)
        ff=[];bb=[]
        with torch.inference_mode():
            for start in range(0,80,16):
                core=backbone.forward_core({k:v[start:start+16].to(device) for k,v in tensors['pixels'].items()})
                ff.append(core['features'].cpu().clone());bb.append(core['policy_logits'].cpu().clone())
        data.append(dict(entry=e,features=torch.cat(ff).clone().to(device),base=torch.cat(bb).clone().to(device),
            history={k:v.to(device) for k,v in tensors['history'].items()},labels={k:v.to(device) for k,v in tensors['labels'].items()}))
    return data


def infer(model,e):
    return model.forward_features(e['features'][None],e['base'][None],{k:v[None] for k,v in e['history'].items()})


@torch.no_grad()
def evaluate(model,data):
    model.eval();rows=[]
    for e in data:
        o=infer(model,e);labels=e['labels'];valid=labels['action_valid'];positive=labels['action_positive'][valid]
        if not bool((positive.sum(-1)==1).all()):raise ValueError('single route labels required')
        truth=positive.long().argmax(-1).tolist();logits=o['policy_logits'][0][valid]
        masked=logits.clone();masked[:,[c for c in range(18) if c not in (0,3,7,9,12,16)]]=-torch.inf
        rows.append(dict(event_id=e['entry']['event_id'],family=e['entry']['family'],split=e['entry']['split'],
            loss=float(tactical_policy_loss(o,{k:v[None] for k,v in labels.items()})['loss']),truth=truth,
            unmasked=logits.argmax(-1).tolist(),horizontal_masked=masked.argmax(-1).tolist()))
    summary={}
    for split in ('train','validation'):
        summary[split]={}
        for family in ('all','F1','F2','F3'):
            rr=[r for r in rows if r['split']==split and (family=='all' or r['family']==family)]
            summary[split][family]={condition:confusion(sum((r['truth'] for r in rr),[]),sum((r[condition] for r in rr),[])) for condition in ('unmasked','horizontal_masked')}
    return dict(summary=summary,events=rows)


class ArmNetwork:
    def __init__(self,backbone,path,device):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema')!=SCHEMA or payload['arm'] not in ARMS:raise ValueError('research schema/arm')
        if payload['baseline_sha256']!=file_sha(BASELINE):raise ValueError('baseline SHA')
        self.arm=payload['arm'];self.device=device
        self.model=TacticalPolicy(backbone,TacticalConfig(**payload['config'])).to(device).eval()
        if self.model.tactical_config.arm!=self.arm:raise ValueError('arm mismatch')
        self.model.heads.load_state_dict(payload['heads'],strict=True)
    @torch.inference_mode()
    def __call__(self,arm,regions,previous,hidden,reset):
        if arm!=self.arm:raise ValueError('callback arm mismatch')
        core=self.model.vision_model.forward_core({k:v.to(self.device) for k,v in regions.items()})
        f=core['features'][:,None];b=core['policy_logits'][:,None]
        o=self.model.forward_features(f,b,executed_history(f,previous),hidden,reset_mask=torch.tensor([[reset]],device=self.device))
        return o['policy_logits'][0,0].cpu(),o['hidden']


def run(out=OUT,device='cuda'):
    out=Path(out);p=verify(out)
    dest=out/'run';dest.mkdir(exist_ok=False)
    if not str(device).startswith('cuda') or not torch.cuda.is_available():raise ValueError('real CUDA required')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    with _fp32_mode(torch.device(device)):
        backbone,_=load_vision_checkpoint(BASELINE,map_location='cpu');backbone.to(device).eval()
        frozen={k:v.detach().clone() for k,v in backbone.state_dict().items()}
        begin=time.perf_counter();data=encode(backbone,json.loads(RECORDS.read_text(encoding='utf-8')),device)
        torch.cuda.synchronize();encoding_seconds=time.perf_counter()-begin
        reports={}
        for arm in ARMS:
            directory=dest/arm;directory.mkdir(exist_ok=False)
            torch.manual_seed(7)
            model=TacticalPolicy(backbone,TacticalConfig(arm=arm)).to(device)
            parameters=[v for v in model.parameters() if v.requires_grad]
            initial=evaluate(model,data)
            optimizer=torch.optim.Adam(parameters,lr=.001);losses=[];training=[e for e in data if e['entry']['split']=='train']
            begin=time.perf_counter()
            for step in range(200):
                model.train();e=training[step%8];o=infer(model,e)
                loss=tactical_policy_loss(o,{k:v[None] for k,v in e['labels'].items()})
                if not loss['has_supervision']:raise AssertionError('must perform all200 updates')
                optimizer.zero_grad(set_to_none=True);loss['loss'].backward()
                torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();losses.append(float(loss['loss'].detach()))
            torch.cuda.synchronize();seconds=time.perf_counter()-begin
            final=evaluate(model,data)
            if not all(torch.equal(v,frozen[k]) for k,v in backbone.state_dict().items()):raise AssertionError('backbone changed')
            report=dict(arm=arm,initial=initial,final=final,updates=200,losses=losses,optimization_seconds=seconds,
                parameters=dict(active=sum(v.numel() for v in parameters),heads_total=sum(v.numel() for v in model.heads.parameters()),total=sum(v.numel() for v in model.parameters()),backbone=sum(v.numel() for v in backbone.parameters())),backbone_unchanged=True)
            with (directory/'research-heads.pt').open('xb') as stream:
                torch.save(dict(schema=SCHEMA,arm=arm,config=asdict(model.tactical_config),baseline_sha256=file_sha(BASELINE),heads={k:v.detach().cpu() for k,v in model.heads.state_dict().items()}),stream)
            write_json(directory/'training.json',report)
            network=ArmNetwork(backbone,directory/'research-heads.pt',device);closed=[]
            for entry in p['closed_loop']['events']:
                row=run_event(make_world(entry['scenario']),arm,'horizontal_masked',network,0,directory/'closed-loop'/entry['event_id'])
                closed.append({k:v for k,v in row.items() if k not in ('steps','trace')})
                print(arm,entry['event_id'],row['survived'],row['goal_satisfied'],row['first_collision_frame'],flush=True)
            families={}
            for family in ('all','F1','F2','F3'):
                rr=[r for r in closed if family=='all' or r['family']==family]
                families[family]=dict(events=len(rr),survived=sum(r['survived'] for r in rr),goal_satisfied=sum(r['goal_satisfied'] for r in rr),feasible=sum(r['feasible'] for r in rr),firsthit={r['event_id']:r['first_collision_frame'] for r in rr})
            write_json(directory/'closed-loop.json',dict(families=families,events=closed))
            reports[arm]=dict(parameters=report['parameters'],optimization_seconds=seconds,closed_loop=families,
                initial=initial['summary'],final=final['summary'])
        verify(out)
        summary=dict(schema='v373-arms-result/1',prereg_sha256=file_sha(out/'preregistration.json'),arms=reports,encoding_seconds=encoding_seconds,
            device=torch.cuda.get_device_name(),torch_version=torch.__version__,precision=dict(dtype='float32',matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),
            independent_test=False,formal_training=False,deployable=False,game=False,source_data_unchanged=True)
        write_json(dest/'summary.json',summary)
        write_json(dest/'sha256.json',{str(f.relative_to(dest)):file_sha(f) for f in dest.rglob('*') if f.is_file()})
        return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('preregister','run'));parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--device',default='cuda');a=parser.parse_args()
    preregister(a.out) if a.command=='preregister' else run(a.out,a.device)
