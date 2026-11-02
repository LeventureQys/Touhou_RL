"""R15 single-factor train-only inverse-frequency N1 research diagnostic."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import torch
from tools.rl import v373_arm_diagnostic as r14
from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_policy import TacticalPolicy, TacticalConfig, tactical_policy_loss
from touhou_rl.tactical_feature_cache import _fp32_mode
from touhou_rl.vmodels import load_vision_checkpoint
from touhou_rl.tactical_synthetic_closed_loop import run_event
from touhou_rl.tactical_teacher_validation import make_world

OUT=r14.ROOT/'weighted-round15'
SCHEMA='v373-weighted-diagnostic-heads/1'
ARMS=('unweighted','weighted')
FORMULA='N=sum_c n_c=640 on train action_valid rows only; K_present=|{c:n_c>0}|; u_c=min(5,sqrt(N/(K_present*n_c))); Z=sum_{c:n_c>0} n_c*u_c/N; w_c=u_c/Z. Absent classes have no target and no weight. For each full80 batch, L_action=sum_{i valid} w_yi*(-log softmax(logits_i)[y_i])/sum_{i valid} w_yi; control uses original unweighted action loss; L_total=L_original-L_original_action+L_action, aux coefficients/losses unchanged.'
write_json=r14.write_json


def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        x=v.detach().cpu().contiguous()
        h.update(k.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def targets(labels,valid=None):
    mask=labels['action_valid'] if valid is None else labels['action_valid'] & valid
    pos=labels['action_positive'][mask]
    if pos.dtype!=torch.bool or not bool((pos.sum(-1)==1).all()):raise ValueError('single class teacher labels required')
    return mask,pos.long().argmax(-1)


def weight_spec(entries,require640=False):
    counts={}
    for e in entries:
        if e['entry']['split']!='train':continue
        _,ys=targets(e['labels'])
        for c in ys.tolist():counts[c]=counts.get(c,0)+1
    n=sum(counts.values());k=len(counts)
    if not n or (require640 and n!=640):raise ValueError('train640 action labels required')
    raw={c:min(5.,math.sqrt(n/(k*v))) for c,v in counts.items()}
    z=sum(counts[c]*raw[c] for c in counts)/n
    return dict(total=n,K_present=k,counts={str(c):counts[c] for c in sorted(counts)},raw={str(c):raw[c] for c in sorted(raw)},normalizer=z,weights={str(c):raw[c]/z for c in sorted(raw)},formula=FORMULA)


def action_loss(output,labels,weights):
    original=tactical_policy_loss(output,labels)
    mask,ys=targets(labels,output.get('valid_mask',torch.ones_like(labels['action_valid'])))
    if ys.numel():
        try:w=torch.tensor([weights[str(c)] for c in ys.tolist()],dtype=output['policy_logits'].dtype,device=ys.device)
        except KeyError as exc:raise ValueError('target absent from train weights') from exc
        nll=-output['policy_logits'][mask].log_softmax(-1).gather(-1,ys[:,None]).squeeze(-1)
        weighted=(w*nll).sum()/w.sum()
    else:weighted=original['action_loss']
    return {**original,'loss':original['loss']-original['action_loss']+weighted,'action_loss':weighted}


def label_entries():
    spec=json.loads(r14.RECORDS.read_text(encoding='utf-8'));entries=[]
    for e in r14.validate_spec(spec):
        if e['split']!='train':continue
        ref=e['labels'];path=r14.RECORDS.parent/ref['path']
        if file_sha(path)!=ref['sha256']:raise ValueError('label SHA')
        labels=torch.load(path,map_location='cpu',weights_only=True)
        if any(bool(v.any()) for k,v in labels.items() if k.endswith('_valid') and k!='action_valid'):raise ValueError('expected action only data')
        entries.append(dict(entry=e,labels=labels))
    return entries


def preregister(out=OUT):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    ids=r14.identities()
    for p in (Path(__file__),Path('touhou_rl/tests/test_v373_weighted_diagnostic.py')):ids[str(p)]=file_sha(p)
    backbone,_=load_vision_checkpoint(r14.BASELINE,map_location='cpu')
    torch.manual_seed(7);model=TacticalPolicy(backbone,TacticalConfig(arm='N1'))
    p=dict(schema='v373-weighted-prereg/1',identities=ids,formula=FORMULA,train_weights=weight_spec(label_entries(),True),initial_head_state_sha256=state_sha(model.heads.state_dict()),
        arms=list(ARMS),seed=7,initialization='fresh seed7 reset immediately before each N1 construction after backbone load; identical canonical head-state SHA required; no warmstart',
        train=dict(updates=200,optimizer='Adam',lr=.001,clip=1.,sequence=80,batch=1,order='original train order roundrobin step modulo8; 25 visits each',validation_selection=False),
        backbone='same frozen baseline, active N1 heads, unchanged tensor equality checked',features='one encode of original16x80 in CUDA chunks16, shared RAM tensors between both arms; no disk cache',
        precision='FP32 noAMP float32_matmul_precision highest; CUDA matmul TF32=false cuDNN TF32=false in encode/train/eval/closed-loop',
        closed_loop=dict(events=json.loads((r14.ROOT/'synthetic-closed-loop-round11/preregistration.json').read_text())['events'],condition='horizontal_masked',runner='unchanged R11 run_event / executed_history; sticky collision continue full horizon; per-event reset'),
        metrics='initial/final train/validation each family: accuracy, peraction confusion/recall, supported-class macro and moving macro (exclude hold0/9); raw18 unsupported predictions reported; horizontal mask identical R11; per-family survival/goal/feasible/firsthit; F1 degradation separately',
        execution_order='CPU tests first; freeze this prereg source+tests+formula+train weights+asset SHA second; one dual-arm run third; report last; retain first failure, never overwrite or retry training',
        independent_test=False,development_only=True,formal_gate=False,deployable=False,teacher_calls=False,new_data=False,shield=False,game=False,tuning=False)
    write_json(out/'preregistration.json',p);return p


def evaluate(model,data):
    report=r14.evaluate(model,data)
    for split in report['summary'].values():
        for family in split.values():
            for m in family.values():
                recalls=[v for i,v in enumerate(m['recall']) if i>0 and v is not None]
                m['moving_macro_recall']=sum(recalls)/len(recalls) if recalls else None
                m['accuracy']=m['agreement']
                m['other_predictions']=sum(row[-1] for row in m['matrix'])
    for e in report['events']:
        e['raw_unsupported_horizontal_count']=sum(c not in (0,3,7,9,12,16) for c in e['unmasked'])
        e['raw_unsupported_teacher_class_count']=sum(c not in r14.CLASSES for c in e['unmasked'])
    return report


class Network(r14.ArmNetwork):
    def __init__(self,backbone,path,device):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema')!=SCHEMA or payload.get('intervention') not in ARMS:raise ValueError('research schema')
        if payload['baseline_sha256']!=file_sha(r14.BASELINE):raise ValueError('baseline SHA')
        self.arm='N1';self.device=device
        self.model=TacticalPolicy(backbone,TacticalConfig(**payload['config'])).to(device).eval()
        if self.model.tactical_config.arm!='N1':raise ValueError('N1 only')
        self.model.heads.load_state_dict(payload['heads'],strict=True)


def run(out=OUT,device='cuda'):
    out=Path(out);p=r14.verify(out);dest=out/'run';dest.mkdir(exist_ok=False)
    try:
        if not str(device).startswith('cuda') or not torch.cuda.is_available():raise ValueError('CUDA required')
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        with _fp32_mode(torch.device(device)):
            backbone,_=load_vision_checkpoint(r14.BASELINE,map_location='cpu');backbone.to(device).eval()
            frozen={k:v.detach().clone() for k,v in backbone.state_dict().items()}
            begin=time.perf_counter();data=r14.encode(backbone,json.loads(r14.RECORDS.read_text()),device)
            torch.cuda.synchronize();encoding_seconds=time.perf_counter()-begin
            actual=weight_spec(data,True)
            if actual!=p['train_weights']:raise AssertionError('train weights changed')
            training=[e for e in data if e['entry']['split']=='train'];reports={}
            for arm in ARMS:
                directory=dest/arm;directory.mkdir(exist_ok=False)
                torch.manual_seed(7);model=TacticalPolicy(backbone,TacticalConfig(arm='N1')).to(device)
                initial_sha=state_sha(model.heads.state_dict())
                if initial_sha!=p['initial_head_state_sha256']:raise AssertionError('initial head mismatch')
                write_json(directory/'initialization.json',dict(head_state_sha256=initial_sha,seed=7,feature_storage_shared=True))
                parameters=[v for v in model.parameters() if v.requires_grad]
                initial=evaluate(model,data);optimizer=torch.optim.Adam(parameters,lr=.001);losses=[]
                begin=time.perf_counter()
                for step in range(200):
                    model.train();e=training[step%8];o=r14.infer(model,e);labels={k:v[None] for k,v in e['labels'].items()}
                    loss=action_loss(o,labels,actual['weights']) if arm=='weighted' else tactical_policy_loss(o,labels)
                    if not loss['has_supervision']:raise AssertionError('200 updates required')
                    optimizer.zero_grad(set_to_none=True);loss['loss'].backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
                    losses.append(float(loss['loss'].detach()))
                torch.cuda.synchronize();seconds=time.perf_counter()-begin;final=evaluate(model,data)
                if not all(torch.equal(v,frozen[k]) for k,v in backbone.state_dict().items()):raise AssertionError('backbone changed')
                params=dict(active=sum(v.numel() for v in parameters),heads_total=sum(v.numel() for v in model.heads.parameters()),total=sum(v.numel() for v in model.parameters()))
                report=dict(intervention=arm,initial_head_state_sha256=initial_sha,final_head_state_sha256=state_sha(model.heads.state_dict()),initial=initial,final=final,updates=200,losses=losses,optimization_seconds=seconds,parameters=params,backbone_unchanged=True)
                with (directory/'research-heads.pt').open('xb') as f:torch.save(dict(schema=SCHEMA,intervention=arm,config=asdict(model.tactical_config),baseline_sha256=file_sha(r14.BASELINE),heads={k:v.detach().cpu() for k,v in model.heads.state_dict().items()}),f)
                write_json(directory/'training.json',report)
                network=Network(backbone,directory/'research-heads.pt',device);closed=[]
                for entry in p['closed_loop']['events']:
                    row=run_event(make_world(entry['scenario']),'N1','horizontal_masked',network,0,directory/'closed-loop'/entry['event_id'])
                    closed.append({k:v for k,v in row.items() if k not in ('steps','trace')})
                    print(arm,entry['event_id'],row['survived'],row['goal_satisfied'],row['first_collision_frame'],flush=True)
                families={}
                for family in ('all','F1','F2','F3'):
                    rows=[r for r in closed if family=='all' or r['family']==family]
                    families[family]=dict(events=len(rows),survived=sum(r['survived'] for r in rows),goal_satisfied=sum(r['goal_satisfied'] for r in rows),feasible=sum(r['feasible'] for r in rows),firsthit={r['event_id']:r['first_collision_frame'] for r in rows})
                write_json(directory/'closed-loop.json',dict(families=families,events=closed))
                reports[arm]=dict(parameters=params,initial_head_state_sha256=initial_sha,final_head_state_sha256=report['final_head_state_sha256'],optimization_seconds=seconds,initial=initial['summary'],final=final['summary'],closed_loop=families)
            r14.verify(out)
            summary=dict(schema='v373-weighted-result/1',prereg_sha256=file_sha(out/'preregistration.json'),arms=reports,encoding_seconds=encoding_seconds,device=torch.cuda.get_device_name(),torch_version=torch.__version__,precision=dict(dtype='float32',matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),independent_test=False,formal_gate=False,deployable=False)
            write_json(dest/'summary.json',summary)
            write_json(dest/'sha256.json',{str(f.relative_to(dest)):file_sha(f) for f in dest.rglob('*') if f.is_file()})
            return summary
    except BaseException:
        write_json(dest/'first-failure.json',dict(traceback=traceback.format_exc(),policy='retained; no automatic rerun or overwrite'))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('preregister','run'));parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    preregister(args.out) if args.command=='preregister' else run(args.out,args.device)
