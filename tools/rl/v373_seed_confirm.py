"""Round16 pre-registered 3-seed confirmation of the R15 weighted/unweighted gap.

Research only: no formal gate, no deployable checkpoint, no game execution. The
fresh confirmation worlds are used once, only after all six checkpoints are
trained and SHA-frozen; this module never calls the teacher policy, the reference
solver or teacher labels on them.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback
import torch
from tools.rl import v373_arm_diagnostic as r14
from tools.rl import v373_weighted_diagnostic as wd
from touhou_rl import tactical_teacher_v2 as teacher_v2
from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_feature_cache import _fp32_mode
from touhou_rl.tactical_policy import TacticalPolicy, TacticalConfig, tactical_policy_loss
from touhou_rl.tactical_synthetic_closed_loop import run_event
from touhou_rl.tactical_teacher_validation import ValidationWorld, definitions, make_world
from touhou_rl.vmodels import load_vision_checkpoint

OUT=r14.ROOT/'seed-confirm-round16'
PREREG_SCHEMA='v373-seed-confirm-prereg/1'
SCHEMA='v373-seed-confirm-heads/1'
ARMS=('unweighted','weighted')
SEEDS=(7,17,27)
UPDATES=200
FAMILIES=('all','F1','F2','F3')
CONDITIONS=('unmasked','horizontal_masked')
FORMULA=wd.FORMULA
write_json=wd.write_json
state_sha=wd.state_sha

# Twelve fresh groups (4 per family, 2 mirrors each = 24 worlds). Geometry stays
# inside the R8 validation/R9 confirmation envelope but no complete signature is
# reused. Frozen here, before any training or evaluation.
GROUPS={
 'F1':[dict(start_x=206.,wall_frame=58,coverage=(82.,288.)),
       dict(start_x=178.,wall_frame=46,coverage=(118.,286.)),
       dict(start_x=218.,wall_frame=54,coverage=(90.,296.),internal_gap=True,gap=(196.,212.)),
       dict(start_x=190.,wall_frame=62,coverage=(96.,282.),internal_gap=True,gap=(232.,246.))],
 'F2':[dict(start_x=166.,fire_interval=17,bullet_speed=7.6,origin=(154.,88.)),
       dict(start_x=214.,fire_interval=19,bullet_speed=9.8,origin=(228.,112.)),
       dict(start_x=186.,fire_interval=27,bullet_speed=8.8,burst=True),
       dict(start_x=204.,fire_interval=29,bullet_speed=7.2,origin=(172.,96.),burst=True)],
 'F3':[dict(start_x=342.,fire_interval=17,window=(38,98),threshold=182.,goal=158.),
       dict(start_x=330.,fire_interval=19,window=(30,96),threshold=190.,goal=162.),
       dict(start_x=348.,fire_interval=27,window=(50,106),threshold=170.,goal=146.),
       dict(start_x=334.,fire_interval=29,window=(26,102),threshold=197.,goal=168.,origin=(216.,108.))]}


def new_dir(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False);return path


def _tuple(value):
    return tuple(value) if isinstance(value,(list,tuple)) else value


def signature(spec):
    """Same geometry signature the R9 confirmation test uses, no execution."""
    s=ValidationWorld(**spec)
    return (s.family,s.canonical_x(s.start_x),_tuple(s.coverage),_tuple(s.gap),s.wall_frame,
            s.fire_interval,_tuple(s.window),s.threshold,s.goal,_tuple(s.origin),s.burst,
            s.bullet_speed,s.internal_gap)


def confirmation_specs():
    rows=[]
    for family,params in GROUPS.items():
        for i,p in enumerate(params):
            group=f'confirm-{family}-{i:02d}'
            for mirror in (False,True):
                kw=dict(p);x=kw.pop('start_x')
                rows.append(dict(event_id=f'{group}-m{int(mirror)}',family=family,template_group=group,
                    seed=16000+i,split='confirmation',mirror=mirror,
                    start_x=384.-x if mirror else x,depletion_pressure=family=='F3',**kw))
    return rows


def confirmation_worlds():
    rows=[]
    for spec in confirmation_specs():
        world=ValidationWorld(**spec);payload=asdict(world)
        rows.append(dict(event_id=payload['event_id'],family=payload['family'],group=payload['template_group'],
            scenario=payload,scenario_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()))
    return rows


def worlds_digest(events):
    return hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest()


def existing_signatures():
    r8=[s for s in definitions() if s['split'] in ('development','validation')]
    if any(s['split']=='test' for s in r8):raise AssertionError('sealed R8 test must not be constructed')
    r9=teacher_v2.confirmation_definitions()
    r10=[e['spec'] for e in json.loads((r14.ROOT/'synthetic-learning-round10/preregistration.json').read_text(encoding='utf-8'))['events']]
    return dict(r8=r8,r9=r9,r10=r10)


def assert_disjoint(specs):
    old=existing_signatures();old_sigs={name:{signature(s) for s in rows} for name,rows in old.items()}
    union=set().union(*old_sigs.values());groups={}
    for s in specs:groups.setdefault(s['template_group'],[]).append(s)
    for group,rows in groups.items():
        if len(rows)!=2:raise ValueError('each group needs exactly two mirrors')
        if signature(rows[0])!=signature(rows[1]):raise ValueError('mirror pair geometry mismatch')
        if signature(rows[0]) in union:raise ValueError('confirmation world duplicates existing parameters: '+group)
    if len({signature(rows[0]) for rows in groups.values()})!=len(groups):
        raise ValueError('confirmation groups must have distinct parameters')
    if not old_sigs['r10']<=old_sigs['r8']:raise ValueError('R10 worlds must be R8 development/validation worlds')
    return dict(r8_development_validation_signatures=len(old_sigs['r8']),
        r9_confirmation_signatures=len(old_sigs['r9']),r10_records_signatures=len(old_sigs['r10']),
        r10_subset_of_r8=True,new_groups=len(groups),new_events=len(specs),duplicates=0,
        fields='family,canonical start_x,coverage,gap,wall_frame,fire_interval,window,threshold,goal,origin,burst,bullet_speed,internal_gap')


def structural_probe(spec):
    """Arithmetic reachability only: no rollout, no solver, no labels."""
    s=ValidationWorld(**spec);x=s.canonical_x(s.start_x)
    if s.family=='F1':
        safe=[(8.,s.coverage[0]-4.),(s.coverage[1]+4.,376.)]
        if s.internal_gap and s.gap[1]-s.gap[0]>=8:safe.append((s.gap[0]+2.,s.gap[1]-2.))
        safe=[(lo,hi) for lo,hi in safe if lo<=hi]
        if not safe:raise ValueError('no safe horizontal interval')
        distance=min(max(lo-x,x-hi,0.) for lo,hi in safe)
        return dict(nearest_safe_px=distance,frames_needed=distance/2.,wall_onset=s.wall_frame,
            reachable_before_wall=distance/2.<=s.wall_frame-2)
    if s.family=='F3':
        position=x-2.*s.window[1];goal_frames=(x-s.goal)/2.
        return dict(position_at_window_end_px=position,below_threshold_at_window_end=position<=s.threshold,
            frames_to_goal=goal_frames,goal_reachable_in_horizon=goal_frames<=s.horizon)
    return dict(fire_interval=s.fire_interval,decisions_between_births=s.fire_interval/2.,
        horizontal_space_px=min(x-8.,376.-x))


def assert_reachable(specs):
    probes={}
    for s in specs:
        probe=structural_probe(s);probes[s['event_id']]=probe
        if s['family']=='F1' and not probe['reachable_before_wall']:raise ValueError('unreachable F1: '+s['event_id'])
        if s['family']=='F3' and not (probe['below_threshold_at_window_end'] and probe['goal_reachable_in_horizon']):
            raise ValueError('unreachable F3: '+s['event_id'])
        if s['family']=='F2' and not (probe['fire_interval']>=16 and probe['horizontal_space_px']>=60):
            raise ValueError('degenerate F2: '+s['event_id'])
    return probes


def train_weights():
    return wd.weight_spec(wd.label_entries(),True)


def formula_digest(spec):
    canonical=json.dumps(spec,sort_keys=True,separators=(',',':')).encode()
    return dict(text_sha256=hashlib.sha256(spec['formula'].encode()).hexdigest(),
        spec_sha256=hashlib.sha256(canonical).hexdigest())


def fresh_policy(backbone,seed,config=None):
    torch.manual_seed(seed)
    return TacticalPolicy(backbone,config or TacticalConfig(arm='N1'))


def initial_head_shas():
    backbone,_=load_vision_checkpoint(r14.BASELINE,map_location='cpu')
    return {str(seed):state_sha(fresh_policy(backbone,seed).heads.state_dict()) for seed in SEEDS}


def identity_paths():
    return [Path(__file__),Path('tests/test_v373_seed_confirm.py'),
        r14.ROOT/'weighted-round15/preregistration.json',r14.ROOT/'weighted-round15/run/summary.json',
        r14.ROOT/'teacher-revision-round9/confirmation-manifest.json',
        Path('tools/rl/v373_weighted_diagnostic.py'),Path('touhou_rl/tactical_teacher_v2.py')]


def preregister(out=OUT):
    out=new_dir(out)
    specs=confirmation_specs();check=assert_disjoint(specs);probes=assert_reachable(specs)
    weights=train_weights();digest=formula_digest(weights)
    r15=json.loads((r14.ROOT/'weighted-round15/preregistration.json').read_text(encoding='utf-8'))
    if weights!=r15['train_weights']:raise ValueError('R15 train weights/formula changed')
    if digest['spec_sha256']!=formula_digest(r15['train_weights'])['spec_sha256']:raise ValueError('formula digest')
    initial=initial_head_shas()
    if initial['7']!=r15['initial_head_state_sha256']:raise ValueError('seed7 must reproduce the R15 initial head state exactly')
    worlds=confirmation_worlds()
    r11=json.loads((r14.ROOT/'synthetic-closed-loop-round11/preregistration.json').read_text(encoding='utf-8'))
    ids=r14.identities()
    for p in identity_paths():ids[str(p)]=file_sha(p)
    p=dict(schema=PREREG_SCHEMA,identities=ids,
        arms=list(ARMS),seeds=list(SEEDS),updates=UPDATES,
        initialization=dict(per_seed=initial,seed7_reproduces_round15=True,
            method='fresh seed reset immediately before every N1 construction after backbone load; identical N1 config for both loss arms so the same seed must give byte-identical initial heads; no warmstart, no checkpoint, no R15 weights',
            initial_head_state_sha256=initial['7']),
        formula=dict(text=FORMULA,spec=weights,source=str(Path('tools/rl/v373_weighted_diagnostic.py')),
            source_sha256=file_sha(Path('tools/rl/v373_weighted_diagnostic.py')),**digest,
            equality_with_round15=True),
        train=dict(updates=UPDATES,optimizer='Adam',lr=.001,clip=1.,sequence=80,batch=1,
            data='original R10 train 8 events only; validation 8 never trained and never used for weights/epochs/selection',
            order='original train order roundrobin step modulo8; 25 visits each',validation_selection=False,
            early_stopping=False,tuning=False),
        features='one genuine CUDA batch16 encode pass over the original 16x80 rows, shared RAM tensors between all arms, no disk feature cache',
        precision='FP32 no AMP, matmul TF32 false, cuDNN TF32 false, float32 precision highest in encode/train/evaluate/closed-loop',
        confirmation_worlds=dict(split='confirmation',events=worlds,worlds_sha256=worlds_digest(worlds),groups=len(GROUPS),
            events_per_family={f:8 for f in ('F1','F2','F3')},duplicate_check=check,
            construction='tactical_teacher_validation.ValidationWorld via make_world; same advance/firing/canonical geometry as R8/R9; split field is confirmation',
            usage='closed-loop confirmation only; never trained, never used for weight/epoch/model selection, no teacher label created',
            solver_calls=0,teacher_policy_calls=0,sealed_test_execution=0),
        structural_probes=probes,
        closed_loop=dict(validation_events=r11['events'],confirmation='confirmation_worlds.events',
            condition='horizontal_masked',runner='unchanged R11 run_event / pixel_stack / render_state / executed_history; sticky collision continue full horizon; per-event hidden reset',
            b0='same renderer and runner with the frozen baseline policy logits only; a reference on these renders, not a real B0 capability measurement',
            order='six checkpoints trained, saved with SHA, reloaded and SHA-checked first; confirmation 24 events then old8 validation continuity control',
            metrics='per event feasible/survived/goal/firsthit; per family event counts; decisions, raw-unsupported and invalid counts'),
        independent_test=False,formal_gate=False,deployable=False,teacher_calls=False,
        new_teacher_labels=False,new_training_data=False,game=False,tuning=False,
        failure_policy='retain first failure; one fixed run, no extra epochs, no baseline switch, no weight change, no overwrite')
    write_json(out/'preregistration.json',p);return p


def verify(out):
    out=Path(out);p=json.loads((out/'preregistration.json').read_text(encoding='utf-8'))
    if p.get('schema')!=PREREG_SCHEMA:raise ValueError('preregistration schema')
    for path,digest in p['identities'].items():
        if file_sha(path)!=digest:raise ValueError('frozen identity changed: '+path)
    if worlds_digest(p['confirmation_worlds']['events'])!=p['confirmation_worlds']['worlds_sha256']:
        raise ValueError('confirmation worlds changed')
    return p


class Network:
    """Arm-aware callback over one frozen research checkpoint, manually checked."""
    def __init__(self,backbone,path,device,expected_sha=None):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema')!=SCHEMA:raise ValueError('research schema')
        if payload.get('arm') not in ARMS or payload.get('seed') not in SEEDS:raise ValueError('research arm/seed')
        if payload['baseline_sha256']!=file_sha(r14.BASELINE):raise ValueError('baseline SHA')
        self.arm='N1';self.device=device;self.name=f"{payload['arm']}-s{payload['seed']}"
        self.model=TacticalPolicy(backbone,TacticalConfig(**payload['config'])).to(device).eval()
        if self.model.tactical_config.arm!='N1':raise ValueError('N1 only')
        self.model.heads.load_state_dict(payload['heads'],strict=True)
        self.heads_sha256=state_sha(self.model.heads.state_dict())
        if self.heads_sha256!=payload['final_head_state_sha256']:raise ValueError('checkpoint head SHA drift')
        if expected_sha is not None and self.heads_sha256!=expected_sha:raise ValueError('unexpected checkpoint')
    @torch.inference_mode()
    def __call__(self,arm,regions,previous,hidden,reset):
        if arm!=self.arm:raise ValueError('callback arm mismatch')
        core=self.model.vision_model.forward_core({k:v.to(self.device) for k,v in regions.items()})
        f=core['features'][:,None];b=core['policy_logits'][:,None]
        o=self.model.forward_features(f,b,r14.executed_history(f,previous),hidden,
            reset_mask=torch.tensor([[reset]],device=self.device))
        return o['policy_logits'][0,0].cpu(),o['hidden']


class BaselineNetwork:
    """R11 B0 semantics: frozen baseline logits, no tactical heads, no labels."""
    def __init__(self,backbone,device):
        self.arm='B0';self.device=device;self.name='B0';self.model=backbone
    @torch.inference_mode()
    def __call__(self,arm,regions,previous,hidden,reset):
        if arm!=self.arm:raise ValueError('callback arm mismatch')
        core=self.model.forward_core({k:v.to(self.device) for k,v in regions.items()})
        return core['policy_logits'][0].cpu(),None


def compact(row):
    physical=row.get('environment_metrics') or {}
    return dict(event_id=row['event_id'],family=row['family'],arm=row['arm'],condition=row['condition'],
        first_collision_frame=row['first_collision_frame'],full_horizon=row['full_horizon'],
        goal_satisfied=row['goal_satisfied'],survived=row['survived'],feasible=row['feasible'],
        invalid_action=row['invalid_action'],technical_failure=row['technical_failure'],
        raw_unsupported_count=row['raw_unsupported_count'],decisions=row['decisions'],
        verdict=physical.get('verdict'),reached_goal=physical.get('reached_goal'),
        corridor_occupancy_bullet_frames=physical.get('corridor_occupancy_bullet_frames'),
        final_horizontal_space_px=physical.get('final_horizontal_space_px'),
        steps=[{k:v for k,v in s.items() if k!='pixels'} for s in row['steps']],trace=row['trace'])


def closed_loop(events,network,arm,label):
    rows=[]
    for entry in events:
        row=compact(run_event(make_world(entry['scenario']),arm,'horizontal_masked',network,0,None))
        row['network']=label
        rows.append(row)
        print('closed-loop',label,row['event_id'],'feasible',row['feasible'],'survived',row['survived'],
            'firsthit',row['first_collision_frame'],flush=True)
    return rows


def families(rows):
    result={}
    for family in FAMILIES:
        rr=[r for r in rows if family=='all' or r['family']==family]
        result[family]=dict(events=len(rr),survived=sum(bool(r['survived']) for r in rr),
            goal_satisfied=sum(bool(r['goal_satisfied']) for r in rr),
            feasible=sum(bool(r['feasible']) for r in rr),
            invalid_action=sum(bool(r['invalid_action']) for r in rr),
            raw_unsupported_count=sum(r['raw_unsupported_count'] for r in rr),
            firsthit={r['event_id']:r['first_collision_frame'] for r in rr})
    return result


def offline(report):
    result={}
    for split in ('train','validation'):
        result[split]={}
        for family in sorted(report['summary'][split]):
            result[split][family]={}
            for condition in CONDITIONS:
                m=report['summary'][split][family][condition]
                result[split][family][condition]=dict(accuracy=m['accuracy'],
                    moving_macro_recall=m['moving_macro_recall'],macro_balanced_recall=m['macro_balanced_recall'],
                    support=list(m['support']),recall=list(m['recall']),other_predictions=m['other_predictions'])
    return result


def seed_stats(results):
    stats={}
    for arm in ARMS:
        stats[arm]={}
        for family in FAMILIES:
            entry=dict(seeds=list(SEEDS))
            for key in ('feasible','survived','goal_satisfied'):
                values=[results[f'{arm}-s{s}'][family][key] for s in SEEDS]
                entry[key]=dict(values=values,mean=sum(values)/len(values),minimum=min(values),
                    maximum=max(values),range=max(values)-min(values))
            stats[arm][family]=entry
    return stats


def run(out=OUT,device='cuda'):
    out=Path(out);p=verify(out);dest=new_dir(out/'run')
    try:
        if not str(device).startswith('cuda') or not torch.cuda.is_available():raise ValueError('real CUDA required')
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
        with _fp32_mode(torch.device(device)):
            if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
                raise AssertionError('TF32 must be disabled')
            backbone,_=load_vision_checkpoint(r14.BASELINE,map_location='cpu');backbone.to(device).eval()
            frozen={k:v.detach().clone() for k,v in backbone.state_dict().items()}
            begin=time.perf_counter()
            data=r14.encode(backbone,json.loads(r14.RECORDS.read_text(encoding='utf-8')),device)
            torch.cuda.synchronize();encoding_seconds=time.perf_counter()-begin
            actual=wd.weight_spec(data,True)
            if actual!=p['formula']['spec']:raise AssertionError('train weights changed after encoding')
            training=[e for e in data if e['entry']['split']=='train']
            validation=[e for e in data if e['entry']['split']=='validation']
            if len(training)!=8 or len(validation)!=8:raise AssertionError('train8/validation8 required')
            if {e['entry']['event_id'] for e in training}&{e['entry']['event_id'] for e in validation}:
                raise AssertionError('split leakage')
            new_dir(dest/'train');reports={};paths={}
            for seed in SEEDS:
                for arm in ARMS:
                    name=f'{arm}-s{seed}';directory=new_dir(dest/'train'/name)
                    model=fresh_policy(backbone,seed).to(device)
                    initial_sha=state_sha(model.heads.state_dict())
                    if initial_sha!=p['initialization']['per_seed'][str(seed)]:raise AssertionError('initial head mismatch')
                    write_json(directory/'initialization.json',dict(head_state_sha256=initial_sha,seed=seed,arm=arm,
                        shared_config='both arms are N1; only the action-loss treatment differs',feature_storage_shared=True))
                    parameters=[v for v in model.parameters() if v.requires_grad]
                    initial=wd.evaluate(model,data);optimizer=torch.optim.Adam(parameters,lr=.001);losses=[]
                    begin=time.perf_counter()
                    for step in range(UPDATES):
                        model.train();e=training[step%8];o=r14.infer(model,e)
                        labels={k:v[None] for k,v in e['labels'].items()}
                        loss=wd.action_loss(o,labels,actual['weights']) if arm=='weighted' else tactical_policy_loss(o,labels)
                        if not loss['has_supervision']:raise AssertionError('all 200 updates must be supervised')
                        optimizer.zero_grad(set_to_none=True);loss['loss'].backward()
                        torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
                        losses.append(float(loss['loss'].detach()))
                    torch.cuda.synchronize();seconds=time.perf_counter()-begin
                    final=wd.evaluate(model,data)
                    if not all(torch.equal(v,frozen[k]) for k,v in backbone.state_dict().items()):
                        raise AssertionError('backbone changed')
                    final_sha=state_sha(model.heads.state_dict());path=directory/'research-heads.pt'
                    payload=dict(schema=SCHEMA,arm=arm,seed=seed,config=asdict(model.tactical_config),
                        baseline_sha256=file_sha(r14.BASELINE),final_head_state_sha256=final_sha,
                        heads={k:v.detach().cpu() for k,v in model.heads.state_dict().items()})
                    with path.open('xb') as stream:torch.save(payload,stream)
                    params=dict(active=sum(v.numel() for v in parameters),
                        heads_total=sum(v.numel() for v in model.heads.parameters()),
                        total=sum(v.numel() for v in model.parameters()))
                    report=dict(arm=arm,seed=seed,initial_head_state_sha256=initial_sha,
                        final_head_state_sha256=final_sha,checkpoint_sha256=file_sha(path),
                        checkpoint_path=str(path.relative_to(dest)),updates=UPDATES,losses=losses,
                        optimization_seconds=seconds,parameters=params,backbone_unchanged=True,
                        train_only='validation labels never indexed',initial=offline(initial),final=offline(final))
                    write_json(directory/'training.json',report)
                    reports[name]=report;paths[name]=path
                    print('trained',name,'init',initial_sha[:12],'final',final_sha[:12],f'{seconds:.2f}s',flush=True)
            write_json(dest/'checkpoints.json',{name:{k:v for k,v in report.items() if k!='losses'}
                for name,report in reports.items()})
            if len(list((dest/'train').rglob('research-heads.pt')))!=6:raise AssertionError('six checkpoints required')
            for name,path in paths.items():
                if file_sha(path)!=reports[name]['checkpoint_sha256']:raise AssertionError('checkpoint SHA drift')
            new_dir(dest/'closed-loop')
            world_events=p['confirmation_worlds']['events'];validation_events=p['closed_loop']['validation_events']
            evaluation={}
            begin=time.perf_counter()
            b0=BaselineNetwork(backbone,device);rows=closed_loop(world_events,b0,'B0','B0')
            write_json(dest/'closed-loop'/'B0-confirmation.json',dict(events=rows,families=families(rows)))
            evaluation['B0']=dict(confirmation=families(rows))
            for name in [f'{arm}-s{seed}' for seed in SEEDS for arm in ARMS]:
                network=Network(backbone,paths[name],device,reports[name]['final_head_state_sha256'])
                confirmation=closed_loop(world_events,network,'N1',name)
                write_json(dest/'closed-loop'/f'{name}-confirmation.json',
                    dict(events=confirmation,families=families(confirmation)))
                continuity=closed_loop(validation_events,network,'N1',name)
                write_json(dest/'closed-loop'/f'{name}-validation.json',
                    dict(events=continuity,families=families(continuity)))
                evaluation[name]=dict(confirmation=families(confirmation),validation=families(continuity),
                    confirmation_events=confirmation,validation_events=continuity)
            torch.cuda.synchronize();evaluation_seconds=time.perf_counter()-begin
            confirmation={name:evaluation[name]['confirmation'] for name in evaluation if name!='B0'}
            continuity={name:evaluation[name]['validation'] for name in evaluation if name!='B0'}
            if len(confirmation)!=6 or len(continuity)!=6:raise AssertionError('six checkpoint evaluations required')
            verify(out)
            summary=dict(schema='v373-seed-confirm-result/1',prereg_sha256=file_sha(out/'preregistration.json'),
                seeds=list(SEEDS),arms=list(ARMS),updates=UPDATES,identities_unchanged=True,
                formula=dict(spec_sha256=p['formula']['spec_sha256'],text_sha256=p['formula']['text_sha256'],
                    source_sha256=p['formula']['source_sha256'],weights=p['formula']['spec']['weights'],
                    counts=p['formula']['spec']['counts'],K_present=p['formula']['spec']['K_present']),
                confirmation_worlds=dict(events=len(world_events),worlds_sha256=p['confirmation_worlds']['worlds_sha256'],
                    duplicate_check=p['confirmation_worlds']['duplicate_check'],
                    per_family={f:8 for f in ('F1','F2','F3')}),
                checkpoints={name:{k:v for k,v in reports[name].items() if k not in ('losses','initial','final')} 
                    | dict(initial_offline={s:reports[name]['initial'][s]['all'] for s in ('train','validation')},
                           final_offline={s:reports[name]['final'][s]['all'] for s in ('train','validation')})
                    for name in reports},
                confirmation=dict(results=confirmation,aggregate=seed_stats(confirmation),
                    b0=evaluation['B0']['confirmation'],
                    verification='R11 run_event replays only the actually executed actions for environment_metrics; no solve(), no teacher policy and no teacher label is read on confirmation worlds'),
                validation_control=dict(results=continuity,aggregate=seed_stats(continuity),
                    note='the old eight validation events were repeatedly exposed in R10-R15; continuity control only, not an independent test',
                    r11_b0_external_reference='R11 reported B0 2/8 feasible on the same eight events with cuDNN TF32 true and a different checkpoint/router protocol'),
                encoding_seconds=encoding_seconds,evaluation_seconds=evaluation_seconds,
                training_seconds_total=sum(reports[n]['optimization_seconds'] for n in reports),
                device=torch.cuda.get_device_name(),torch_version=torch.__version__,
                precision=dict(dtype='float32',matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_precision=torch.get_float32_matmul_precision()),
                independent_test=False,formal_gate=False,deployable=False,teacher_calls=False,
                new_teacher_labels=False,game=False,tuning=False,retrained=False)
            write_json(dest/'summary.json',summary)
            write_json(dest/'sha256.json',{str(f.relative_to(dest)):file_sha(f) for f in dest.rglob('*') if f.is_file()})
            return summary
    except BaseException:
        write_json(dest/'first-failure.json',dict(traceback=traceback.format_exc(),
            policy='retained; no automatic rerun, overwrite, retune or baseline switch'))
        raise


def complete(out=OUT):
    """Read-only recovery after the documented post-evaluation aggregation crash.

    Never trains, encodes or runs a closed loop: it verifies the artifacts written
    by the single training/evaluation run and rebuilds the summary that the crashed
    aggregation failed to write. Every frozen identity except this module must still
    match; the module fix is recorded explicitly.
    """
    out=Path(out);p=json.loads((out/'preregistration.json').read_text(encoding='utf-8'))
    if p.get('schema')!=PREREG_SCHEMA:raise ValueError('preregistration schema')
    changed={}
    for path,digest in p['identities'].items():
        current=file_sha(path)
        if current==digest:continue
        if Path(path).resolve()!=Path(__file__).resolve():raise ValueError('frozen identity changed: '+path)
        changed[str(path)]=dict(preregistered_sha256=digest,current_sha256=current)
    if not changed:raise ValueError('completion expects the documented post-preregistration module fix')
    if worlds_digest(p['confirmation_worlds']['events'])!=p['confirmation_worlds']['worlds_sha256']:
        raise ValueError('confirmation worlds changed')
    run=out/'run'
    failure=json.loads((run/'first-failure.json').read_text(encoding='utf-8'))
    if 'TypeError' not in failure['traceback'] or 'families' not in failure['traceback']:
        raise ValueError('unexpected first failure')
    checkpoints=json.loads((run/'checkpoints.json').read_text(encoding='utf-8'))
    if list(checkpoints)!=[f'{arm}-s{seed}' for seed in SEEDS for arm in ARMS]:raise ValueError('checkpoint set')
    training={name:json.loads((run/'train'/name/'training.json').read_text(encoding='utf-8')) for name in checkpoints}
    for name,ckpt in checkpoints.items():
        if file_sha(run/ckpt['checkpoint_path'])!=ckpt['checkpoint_sha256']:raise ValueError('checkpoint SHA drift: '+name)
        if training[name]['final_head_state_sha256']!=ckpt['final_head_state_sha256']:
            raise ValueError('checkpoint record mismatch: '+name)
    closed={}
    for name in ['B0']+list(checkpoints):
        for world in ('confirmation','validation'):
            if name=='B0' and world=='validation':continue
            payload=json.loads((run/'closed-loop'/f'{name}-{world}.json').read_text(encoding='utf-8'))
            if families(payload['events'])!=payload['families']:raise ValueError('stored aggregate mismatch: '+name+'-'+world)
            closed[(name,world)]=payload
    for name in checkpoints:
        if len(closed[(name,'confirmation')]['events'])!=24 or len(closed[(name,'validation')]['events'])!=8:
            raise ValueError('unexpected event counts: '+name)
    if len(closed[('B0','confirmation')]['events'])!=24:raise ValueError('unexpected B0 event count')
    confirmation={name:closed[(name,'confirmation')]['families'] for name in checkpoints}
    continuity={name:closed[(name,'validation')]['families'] for name in checkpoints}
    dest=new_dir(out/'run-completion')
    summary=dict(schema='v373-seed-confirm-result/1',prereg_sha256=file_sha(out/'preregistration.json'),
        seeds=list(SEEDS),arms=list(ARMS),updates=UPDATES,
        identities='all frozen identities except tools/rl/v373_seed_confirm.py still match; the only post-preregistration source change is the documented aggregation fix plus this read-only recovery command',
        formula=dict(spec_sha256=p['formula']['spec_sha256'],text_sha256=p['formula']['text_sha256'],
            source_sha256=p['formula']['source_sha256'],weights=p['formula']['spec']['weights'],
            counts=p['formula']['spec']['counts'],K_present=p['formula']['spec']['K_present']),
        confirmation_worlds=dict(events=len(p['confirmation_worlds']['events']),
            worlds_sha256=p['confirmation_worlds']['worlds_sha256'],
            duplicate_check=p['confirmation_worlds']['duplicate_check'],per_family={f:8 for f in ('F1','F2','F3')}),
        checkpoints={name:{k:v for k,v in training[name].items() if k not in ('losses','initial','final')}
            | dict(initial_offline={s:training[name]['initial'][s]['all'] for s in ('train','validation')},
                   final_offline={s:training[name]['final'][s]['all'] for s in ('train','validation')})
            for name in training},
        confirmation=dict(results=confirmation,aggregate=seed_stats(confirmation),
            b0=closed[('B0','confirmation')]['families'],
            verification='R11 run_event replays only the actually executed actions for environment_metrics; no solve(), no teacher policy and no teacher label was read on confirmation worlds'),
        validation_control=dict(results=continuity,aggregate=seed_stats(continuity),
            note='the old eight validation events were repeatedly exposed in R10-R15; continuity control only, not an independent test',
            r11_b0_external_reference='R11 reported B0 2/8 feasible on the same eight events with cuDNN TF32 true and a different checkpoint/router protocol'),
        encoding_seconds=None,evaluation_seconds=None,
        training_seconds_total=sum(training[n]['optimization_seconds'] for n in training),
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,torch_version=torch.__version__,
        precision=dict(dtype='float32',matmul_tf32=False,cudnn_tf32=False,
            source='the training/evaluation process asserted both TF32 flags false before any step; the completion process does no numeric work'),
        independent_test=False,formal_gate=False,deployable=False,teacher_calls=False,
        new_teacher_labels=False,game=False,tuning=False,retrained=False,
        recovery=dict(reason='the single training/evaluation run wrote all six checkpoints and all 216 closed-loop events, then crashed in the final B0 aggregation',
            first_failure_sha256=file_sha(run/'first-failure.json'),retrained=False,re_encoded=False,re_evaluated=False,
            run_directory_rewritten=False,changed_identities=changed,
            timings='encoding and evaluation wall times were lost with the crashed process; per-checkpoint optimization seconds and every event row are intact'))
    write_json(dest/'summary.json',summary)
    artifacts={str(f.relative_to(run)):file_sha(f) for f in run.rglob('*') if f.is_file()}
    write_json(dest/'notes.json',dict(schema='v373-seed-confirm-completion/1',reason=summary['recovery']['reason'],
        first_failure_last_line=failure['traceback'].strip().splitlines()[-1],
        training_checkpoints=len(checkpoints),closed_loop_events=sum(len(v['events']) for v in closed.values()),
        retrained=False,re_encoded=False,re_evaluated=False,run_directory_rewritten=False,
        changed_identities=changed,unchanged_identities=len(p['identities'])-len(changed),
        run_artifact_sha256=artifacts,summary_sha256=file_sha(dest/'summary.json')))
    write_json(dest/'sha256.json',{str(f.relative_to(dest)):file_sha(f) for f in dest.rglob('*') if f.is_file()})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preregister','run','complete'))
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.command=='preregister':preregister(args.out)
    elif args.command=='run':run(args.out,args.device)
    else:complete(args.out)
