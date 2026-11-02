"""Round17 pre-registered single-factor KL(base||policy) anchor, research only.

Arm A repeats the R16 unweighted arm; arm B adds exactly one term,
1.0*KL(base||policy) over every action_valid row, and nothing else changes. The 24
confirmation worlds are rebuilt by a per-world constant-policy filter before any
training, because R16's confirmation worlds were passable by one constant policy.
No formal gate, no deployable checkpoint, no teacher label, no game execution.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback
import torch
import torch.nn.functional as F
from tools.rl import v373_arm_diagnostic as r14
from tools.rl import v373_seed_confirm as r16
from tools.rl import v373_weighted_diagnostic as wd
from touhou_rl import tactical_teacher_v2 as teacher_v2
from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_feature_cache import _fp32_mode
from touhou_rl.tactical_policy import TacticalPolicy, TacticalConfig, tactical_policy_loss
from touhou_rl.tactical_scenarios import initial_state
from touhou_rl.tactical_teacher_validation import ValidationWorld, advance, definitions, make_world
from touhou_rl.vmodels import load_vision_checkpoint

OUT=r14.ROOT/'kl-anchor-round17'
PREREG_SCHEMA='v373-kl-anchor-prereg/1'
RESULT_SCHEMA='v373-kl-anchor-result/1'
SCHEMA='v373-kl-anchor-heads/1'
ROUND16='seed-confirm-round16'
ROUND15='weighted-round15'
CONTROL='no-anchor'
ANCHOR='kl-anchor'
ARMS=(CONTROL,ANCHOR)
SEEDS=(7,17,27)
UPDATES=200
ANCHOR_WEIGHT=1.0
LR=.001
CLIP=1.
SEQUENCE=80
BATCH=1
FAMILIES=('all','F1','F2','F3')
CONDITIONS=('unmasked','horizontal_masked')
write_json=r16.write_json
state_sha=r16.state_sha
signature=r16.signature
FAMILY_KEYS=('F1','F2','F3')

# Constant policies used by the construction filter. The runner arms are the frozen
# R11 fixed-policy arms; candidate 3 is right (+2 px/frame) and 7 is left (-2).
CONSTANTS=(('hold','fixed_hold',0,0),('right','train_mode',3,2),('left','train_mode',7,-2))

# Pre-filter design, kept only as the "before" side of the construction filter. It
# follows R16's envelope and, like R16, is passable by one constant policy.
CANDIDATES={
 'F1':[dict(start_x=182.,wall_frame=60,coverage=(104.,290.)),
       dict(start_x=198.,wall_frame=46,coverage=(114.,276.)),
       dict(start_x=214.,wall_frame=56,coverage=(88.,298.),internal_gap=True,gap=(204.,220.)),
       dict(start_x=226.,wall_frame=66,coverage=(78.,286.),internal_gap=True,gap=(238.,252.))],
 'F2':[dict(start_x=174.,fire_interval=18,bullet_speed=8.4,origin=(158.,92.)),
       dict(start_x=196.,fire_interval=20,bullet_speed=9.2,origin=(206.,104.)),
       dict(start_x=190.,fire_interval=30,bullet_speed=8.6,burst=True),
       dict(start_x=220.,fire_interval=24,bullet_speed=7.4,origin=(238.,110.),burst=True)],
 'F3':[dict(start_x=318.,fire_interval=18,window=(20,94),threshold=192.,goal=164.),
       dict(start_x=332.,fire_interval=22,window=(36,100),threshold=186.,goal=158.),
       dict(start_x=340.,fire_interval=28,window=(48,102),threshold=176.,goal=150.),
       dict(start_x=356.,fire_interval=16,window=(32,108),threshold=194.,goal=170.,origin=(220.,106.))]}

# Pre-declared regeneration pool. Used in index order for the families the filter
# rejects; every entry is probed itself before it is accepted.
REPLACEMENTS={
 'F1':[dict(start_x=168.,wall_frame=40,coverage=(8.,376.),internal_gap=True,gap=(184.,216.)),
       dict(start_x=248.,wall_frame=52,coverage=(8.,376.),internal_gap=True,gap=(208.,240.)),
       dict(start_x=182.,wall_frame=34,coverage=(8.,376.),internal_gap=True,gap=(136.,168.)),
       dict(start_x=222.,wall_frame=58,coverage=(8.,376.),internal_gap=True,gap=(246.,278.))],
 'F2':[dict(start_x=180.,fire_interval=17,bullet_speed=7.9,origin=(164.,94.)),
       dict(start_x=200.,fire_interval=19,bullet_speed=9.0,origin=(212.,102.)),
       dict(start_x=188.,fire_interval=25,bullet_speed=8.4,burst=True),
       dict(start_x=216.,fire_interval=23,bullet_speed=7.6,origin=(230.,106.),burst=True)],
 'F3':[dict(start_x=328.,fire_interval=20,window=(28,98),threshold=188.,goal=162.),
       dict(start_x=344.,fire_interval=24,window=(42,104),threshold=178.,goal=152.),
       dict(start_x=338.,fire_interval=30,window=(50,110),threshold=180.,goal=154.),
       dict(start_x=352.,fire_interval=18,window=(24,96),threshold=196.,goal=168.,origin=(216.,102.))]}

# Frozen post-filter selection, the only worlds this round evaluates.
FILTERED={
 'F1':[dict(start_x=168.,wall_frame=40,coverage=(8.,376.),internal_gap=True,gap=(184.,216.)),
       dict(start_x=248.,wall_frame=52,coverage=(8.,376.),internal_gap=True,gap=(208.,240.)),
       dict(start_x=182.,wall_frame=34,coverage=(8.,376.),internal_gap=True,gap=(136.,168.)),
       dict(start_x=222.,wall_frame=58,coverage=(8.,376.),internal_gap=True,gap=(246.,278.))],
 'F2':[dict(start_x=174.,fire_interval=18,bullet_speed=8.4,origin=(158.,92.)),
       dict(start_x=196.,fire_interval=20,bullet_speed=9.2,origin=(206.,104.)),
       dict(start_x=190.,fire_interval=30,bullet_speed=8.6,burst=True),
       dict(start_x=220.,fire_interval=24,bullet_speed=7.4,origin=(238.,110.),burst=True)],
 'F3':[dict(start_x=318.,fire_interval=18,window=(20,94),threshold=192.,goal=164.),
       dict(start_x=332.,fire_interval=22,window=(36,100),threshold=186.,goal=158.),
       dict(start_x=340.,fire_interval=28,window=(48,102),threshold=176.,goal=150.),
       dict(start_x=356.,fire_interval=16,window=(32,108),threshold=194.,goal=170.,origin=(220.,106.))]}

FORMULA=('Arm B only, weight fixed at 1.0 before the run. For a full-80 batch let a=policy_logits [1,80,18], '
 'b=the frozen baseline policy_logits of the same encoded rows (detached, no gradient into the backbone), '
 'V={i: action_valid_i and sequence-valid_i}, N=|V|, and KL_anchor=(1/N)*sum_{i in V} sum_c softmax(b_i)_c*'
 '(log softmax(b_i)_c-log softmax(a_i)_c), i.e. KL(base||policy) in nats averaged over every action_valid row '
 '(F.kl_div(log_target=True, reduction="batchmean") on the masked rows). '
 'L_total(B)=L_original+1.0*KL_anchor. L_total(A)=L_original, byte-identical to the R16 unweighted arm. '
 'L_original is the unchanged tactical_policy_loss (action 1.0, option .2, target .5, termination .2, slack .1, '
 'local 1.0) and its local term is exactly the same KL(base||policy) form; that term is empty here because the '
 'R10 action-only batches contain zero local_valid rows, so it contributes exactly 0 to both arms. '
 'This is an intentional single-factor loss/structure revision of the planned L_local_KL, which was specified for '
 'explicit local-regression samples: R17 keeps its functional form and applies it to the action_valid rows as an '
 'anchor to the frozen base policy. The weight is never tuned on any result.')

FILTER_RULE=('Per-world construction filter, executed before any training and frozen in this preregistration: a '
 'candidate world is rejected when hold (candidate 0), constant right (candidate 3, +2 px/frame) or constant left '
 '(candidate 7, -2 px/frame) alone completes it feasibly. Rejected groups are regenerated in index order from the '
 'pre-declared replacement pool (each replacement is probed first and skipped if it is itself constant-solvable or '
 'duplicates a previous-round parameter signature) and the whole 24-world set is re-probed until every world fails '
 'all three constants. Reason: R16 confirmation worlds were passed by constant right 9/24 in aggregate (F1 8/8 by '
 'right and by left, F2 4/8 by hold), so an unfiltered aggregate cannot be read as a capability measure. The probe '
 'runs the frozen advance() dynamics with a fixed action sequence: no pixels, no labels, no teacher, no solver, no '
 'program search, no witness.')


def new_dir(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False);return path


def spec_for(family,index,params,mirror):
    kw=dict(params);x=kw.pop('start_x')
    group=f'klconf-{family}-{index:02d}'
    return dict(event_id=f'{group}-m{int(mirror)}',family=family,template_group=group,seed=17000+index,
        split='confirmation',mirror=mirror,start_x=384.-x if mirror else x,
        depletion_pressure=family=='F3',**kw)


def specs_from(selection):
    return [spec_for(family,index,params,mirror) for family,params in selection.items()
        for index,params in enumerate(params) for mirror in (False,True)]


def confirmation_specs(selection=None):
    return specs_from(FILTERED if selection is None else selection)


def confirmation_worlds(specs=None):
    rows=[]
    for spec in confirmation_specs() if specs is None else specs:
        world=ValidationWorld(**spec);payload=asdict(world)
        rows.append(dict(event_id=payload['event_id'],family=payload['family'],group=payload['template_group'],
            scenario=payload,scenario_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()))
    return rows


def worlds_digest(events):
    return r16.worlds_digest(events)


def constant_rollout(spec,action):
    """One fixed constant action sequence through the frozen advance() dynamics."""
    world=make_world(spec);state=initial_state(world);positions=[state.x];first=None
    while state.frame<world.horizon:
        before=state.hit
        state=advance(world,state,action)
        if state.hit and not before and first is None:first=state.frame
        positions.append(state.x)
    goal=world.family!='F3' or any(world.canonical_x(x)<=world.goal for x in positions)
    return dict(family=world.family,survived=not state.hit,goal_satisfied=bool(goal),
        feasible=bool(not state.hit and goal),first_collision_frame=first,decisions=len(positions)-1)


def constant_probe(specs):
    rows=[]
    for spec in specs:
        for label,arm,candidate,action in CONSTANTS:
            row=dict(event_id=spec['event_id'],template_group=spec['template_group'],policy=label,
                candidate=candidate,action=action,arm=arm,**constant_rollout(spec,action))
            rows.append(row)
    return rows


def constant_counts(rows):
    result={}
    for family in FAMILY_KEYS:
        result[family]={}
        for label,_,_,_ in CONSTANTS:
            rr=[r for r in rows if r['family']==family and r['policy']==label]
            result[family][label]=dict(events=len(rr),feasible=sum(bool(r['feasible']) for r in rr),
                events_passing=sorted(r['event_id'] for r in rr if r['feasible']))
    result['all']={label:dict(events=len([r for r in rows if r['policy']==label]),
        feasible=sum(bool(r['feasible']) for r in rows if r['policy']==label),
        events_passing=sorted(r['event_id'] for r in rows if r['policy']==label and r['feasible']))
        for label,_,_,_ in CONSTANTS}
    return result


def constant_events(rows):
    return sorted({r['event_id'] for r in rows if r['feasible']})


def existing_signatures():
    r8=[s for s in definitions() if s['split'] in ('development','validation')]
    if any(s['split']=='test' for s in r8):raise AssertionError('sealed R8 test must not be constructed')
    r9=teacher_v2.confirmation_definitions()
    r10=[e['spec'] for e in json.loads((r14.ROOT/'synthetic-learning-round10/preregistration.json').read_text(encoding='utf-8'))['events']]
    old16=json.loads((r14.ROOT/ROUND16/'preregistration.json').read_text(encoding='utf-8'))
    r16worlds=[e['scenario'] for e in old16['confirmation_worlds']['events']]
    if len(r16worlds)!=24:raise ValueError('round16 confirmation set must have 24 worlds')
    return dict(r8=r8,r9=r9,r10=r10,r16=r16worlds)


def previous_signature_union():
    return set().union(*[set(signature(s) for s in rows) for rows in existing_signatures().values()])


def assert_disjoint(specs):
    old=existing_signatures();old_sigs={name:set(signature(s) for s in rows) for name,rows in old.items()}
    union=set().union(*old_sigs.values());groups={}
    for s in specs:groups.setdefault(s['template_group'],[]).append(s)
    for group,rows in groups.items():
        if len(rows)!=2:raise ValueError('each group needs exactly two mirrors: '+group)
        if signature(rows[0])!=signature(rows[1]):raise ValueError('mirror pair geometry mismatch: '+group)
        if signature(rows[0]) in union:raise ValueError('confirmation world duplicates existing parameters: '+group)
    if len(set(signature(rows[0]) for rows in groups.values()))!=len(groups):
        raise ValueError('confirmation groups must have distinct parameters')
    if not old_sigs['r10']<=old_sigs['r8']:raise ValueError('R10 worlds must be R8 development/validation worlds')
    return dict(r8_development_validation_signatures=len(old_sigs['r8']),
        r9_confirmation_signatures=len(old_sigs['r9']),r10_records_signatures=len(old_sigs['r10']),
        r16_confirmation_signatures=len(old_sigs['r16']),previous_signatures_union=len(union),
        r10_subset_of_r8=True,new_groups=len(groups),new_events=len(specs),duplicates=0,
        fields='family,canonical start_x,coverage,gap,wall_frame,fire_interval,window,threshold,goal,origin,burst,bullet_speed,internal_gap')


def replacement_clean(family,index,params,union):
    specs=[spec_for(family,index,params,mirror) for mirror in (False,True)]
    if any(signature(s) in union for s in specs):return False
    return not constant_events(constant_probe(specs))


def construction_filter(candidates=CANDIDATES,replacements=REPLACEMENTS):
    """Deterministic per-world constant filter, run before training only."""
    selection={family:list(params) for family,params in candidates.items()}
    used={family:0 for family in selection};union=previous_signature_union();steps=[]
    before_rows=constant_probe(specs_from(selection))
    before=dict(counts=constant_counts(before_rows),events_with_any_constant=constant_events(before_rows),
        unique_events_with_any_constant=len(constant_events(before_rows)))
    for _ in range(12):
        rows=constant_probe(specs_from(selection));bad=constant_events(rows)
        if not bad:
            after=dict(counts=constant_counts(rows),events_with_any_constant=[],
                unique_events_with_any_constant=0)
            return dict(selection=selection,before=before,after=after,steps=steps,
                specs=specs_from(selection))
        order={(family,index):(n,index) for n,(family,params) in enumerate(selection.items())
            for index in range(len(params))}
        rejected=sorted({(event.split('-')[1],int(event.split('-')[2])) for event in bad},
            key=lambda pair:order[pair])
        for family,index in rejected:
            pool=replacements[family];accepted=None
            while used[family]<len(pool):
                candidate=pool[used[family]];used[family]+=1
                if replacement_clean(family,index,candidate,union):accepted=candidate;break
            if accepted is None:raise ValueError('replacement pool exhausted for '+family)
            steps.append(dict(family=family,index=index,template_group=f'klconf-{family}-{index:02d}',
                rejected_worlds=sorted(e for e in bad if e.startswith(f'klconf-{family}-{index:02d}-')),
                rejected_because=sorted({r['policy'] for r in rows
                    if r['feasible'] and r['template_group']==f'klconf-{family}-{index:02d}'}),
                new_params=accepted))
            selection[family][index]=accepted
    raise ValueError('construction filter did not converge')


def structural_probes(specs):
    return r16.assert_reachable(specs)


def local_valid_rows():
    spec=json.loads(r14.RECORDS.read_text(encoding='utf-8'));total=0;present=0;splits=0
    for e in r14.validate_spec(spec):
        ref=e['labels'];path=r14.RECORDS.parent/ref['path']
        if file_sha(path)!=ref['sha256']:raise ValueError('label SHA')
        labels=torch.load(path,map_location='cpu',weights_only=True);splits+=1
        if 'local_valid' in labels:
            present+=1;total+=int(labels['local_valid'].sum())
    return dict(label_files_checked=splits,files_with_local_valid=present,local_valid_true_rows=total,
        note='the frozen tactical_policy_loss local term is KL(base||policy) on this mask; the action-only R10 data leaves it empty, so it is exactly 0 in both arms')


def identity_paths():
    return [Path(__file__),Path('tests/test_v373_kl_anchor.py'),
        r14.ROOT/ROUND16/'preregistration.json',r14.ROOT/ROUND16/'run-completion/summary.json',
        r14.ROOT/ROUND15/'preregistration.json',r14.ROOT/ROUND15/'run/summary.json',
        r14.ROOT/'teacher-revision-round9/confirmation-manifest.json',
        Path('tools/rl/v373_seed_confirm.py'),Path('tools/rl/v373_weighted_diagnostic.py'),
        Path('touhou_rl/tactical_teacher_v2.py')]


def round16_initial_heads():
    p=json.loads((r14.ROOT/ROUND16/'preregistration.json').read_text(encoding='utf-8'))
    return dict(p['initialization']['per_seed'])


def round16_unweighted_final_heads():
    s=json.loads((r14.ROOT/ROUND16/'run-completion/summary.json').read_text(encoding='utf-8'))
    return {str(seed):s['checkpoints'][f'unweighted-s{seed}']['final_head_state_sha256'] for seed in SEEDS}


def formula_digest():
    return dict(text_sha256=hashlib.sha256(FORMULA.encode()).hexdigest(),
        filter_sha256=hashlib.sha256(FILTER_RULE.encode()).hexdigest())


def preregister(out=OUT):
    out=new_dir(out)
    filtered=construction_filter()
    if filtered['selection']!=FILTERED:raise ValueError('construction filter must reproduce the frozen selection')
    specs=filtered['specs'];check=assert_disjoint(specs);probes=structural_probes(specs)
    initial=r16.initial_head_shas()
    if initial!=round16_initial_heads():raise ValueError('fresh initial head states must reproduce round16')
    final16=round16_unweighted_final_heads();local=local_valid_rows()
    if local['local_valid_true_rows']!=0:raise ValueError('action-only data expected: local_valid must be empty')
    worlds=confirmation_worlds(specs)
    r11=json.loads((r14.ROOT/'synthetic-closed-loop-round11/preregistration.json').read_text(encoding='utf-8'))
    ids=r14.identities()
    for path in identity_paths():ids[str(path)]=file_sha(path)
    digest=formula_digest()
    p=dict(schema=PREREG_SCHEMA,identities=ids,arms=list(ARMS),seeds=list(SEEDS),updates=UPDATES,
        anchor_weight=ANCHOR_WEIGHT,
        intervention=dict(factor='one extra KL(base||policy) anchor term in the action loss of arm B only',
            control='arm A = R16 unweighted arm: unchanged tactical_policy_loss, no added term, same seed schedule',
            anchor='arm B = bit-identical to arm A except L_total=L_original+1.0*KL_anchor over every action_valid row',
            unchanged=['original R10 train 8 events','fresh seed reset before every construction','seeds 7/17/27',
                '200 updates','Adam lr 1e-3','grad clip 1','full 80-step sequence','batch 1','FP32 no TF32',
                'one shared encode pass','N1 heads','no early stopping','no validation selection'],
            not_changed=['class weights (no weighted arm)','epochs','data','schedule','baseline checkpoint',
                'optimizer','closed-loop runner'],
            weight_tuning='none: 1.0 fixed before the run, never adjusted on any result'),
        formula=dict(text=FORMULA,**digest,anchor_weight=ANCHOR_WEIGHT,
            extension_of=dict(planned_term='L_local_KL on explicit local-regression samples (local_valid mask)',
                implementation='the same KL(base||policy) functional form applied to the action_valid mask',
                data_fact=local,
                classification='intentional single-factor structural/loss revision, frozen before the run')),
        train=dict(updates=UPDATES,optimizer='Adam',lr=LR,clip=CLIP,sequence=SEQUENCE,batch=BATCH,
            data='original R10 train 8 events only; validation 8 never trained and never used for weights/epochs/selection',
            order='original train order roundrobin step modulo8; 25 visits each',validation_selection=False,
            early_stopping=False,tuning=False),
        initialization=dict(per_seed=initial,initial_head_state_sha256=initial['7'],
            method='fresh seed reset immediately before every N1 construction after backbone load; both arms use the identical N1 config, so the same seed must give a byte-identical initial head state; no warmstart, no checkpoint, no round15/16 weights',
            identical_between_arms=True,
            round16_unweighted_final_head_sha256=final16,
            reproducibility_check='arm A final head SHA is compared per seed with the R16 unweighted final head SHA and recorded; a mismatch would be reported, not silently retrained'),
        features='one genuine CUDA batch16 encode pass over the original 16x80 rows, shared RAM tensors between all six arms, no disk feature cache',
        precision='FP32 no AMP, matmul TF32 false, cuDNN TF32 false, float32 precision highest in encode/train/evaluate/closed-loop',
        confirmation_worlds=dict(split='confirmation',events=worlds,worlds_sha256=worlds_digest(worlds),groups=len(FILTERED),
            events_per_family={f:8 for f in FAMILY_KEYS},duplicate_check=check,
            construction='tactical_teacher_validation.ValidationWorld via make_world; same advance/firing/canonical geometry as R8/R9/R11/R16; split field is confirmation',
            constant_filter=dict(rule=FILTER_RULE,constants=[dict(policy=l,arm=a,candidate=c,action=x)
                for l,a,c,x in CONSTANTS],candidates=CANDIDATES,replacements=REPLACEMENTS,
                filtered=FILTERED,before=filtered['before'],after=filtered['after'],
                replaced=filtered['steps'],replacement_count=len(filtered['steps']),
                executor='frozen advance() from tactical_teacher_validation, the same step function the closed-loop runner uses; fixed action sequences only'),
            usage='closed-loop confirmation only; never trained, never used for weight/epoch/model selection, no teacher label created',
            solver_calls=0,teacher_policy_calls=0,sealed_test_execution=0),
        structural_probes=probes,
        closed_loop=dict(validation_events=r11['events'],confirmation='confirmation_worlds.events',
            condition='horizontal_masked',
            runner='unchanged R11 run_event / pixel_stack / render_state / executed_history; sticky collision, continue full horizon; per-event hidden reset',
            b0='same renderer and runner with the frozen baseline policy logits only; a reference on these renders, not a real B0 capability measurement',
            order='all six checkpoints trained and SHA-frozen first; only then B0 confirmation, then each arm-seed confirmation, then the old 8 validation continuity control',
            metrics='per event feasible/survived/goal/firsthit; per family event counts; decisions, raw-unsupported and invalid counts; per arm action distribution (hold/right/left share, median switches)',
            families='the 24-event aggregate is reported for reference only; conclusions are read per family and on the F2+F3 subset'),
        forgetting=dict(definition='forgetting = B0 F1 feasible count minus the arm F1 feasible count on the same 24 confirmation renders; positive means the finetune lost base F1 behaviour, negative means it gained F1 successes B0 does not have',
            reference='B0 measured once on these renders with the frozen baseline logits only',
            caveat='after the constant filter B0 is expected to fail most F1 worlds, so a non-positive forgetting value does not mean the arm kept a base behaviour'),
        order_of_operations='CPU tests first; freeze this prereg second; one training run of six checkpoints third; evaluate only after all six exist; report last',
        independent_test=False,formal_gate=False,deployable=False,teacher_calls=False,new_teacher_labels=False,
        new_training_data=False,game=False,tuning=False,
        failure_policy='retain first failure; one fixed run, no extra epochs, no baseline switch, no weight change, no overwrite')
    write_json(out/'preregistration.json',p);return p


def verify(out):
    out=Path(out);p=json.loads((out/'preregistration.json').read_text(encoding='utf-8'))
    if p.get('schema')!=PREREG_SCHEMA:raise ValueError('preregistration schema')
    for path,digest in p['identities'].items():
        if file_sha(path)!=digest:raise ValueError('frozen identity changed: '+path)
    if worlds_digest(p['confirmation_worlds']['events'])!=p['confirmation_worlds']['worlds_sha256']:
        raise ValueError('confirmation worlds changed')
    if p['formula']['text_sha256']!=formula_digest()['text_sha256']:raise ValueError('formula text changed')
    return p


def anchor_mask(output,labels):
    return output.get('valid_mask',torch.ones_like(labels['action_valid']))&labels['action_valid']


def kl_anchor(output,labels):
    """KL(base||policy) over every action_valid row; the base logits stay detached."""
    logits=output['policy_logits'];base=output['base_logits']
    if logits.shape!=base.shape or logits.shape[-1]!=18:raise ValueError('policy/base logits shape')
    mask=anchor_mask(output,labels);n=int(mask.sum())
    if not n:raise ValueError('KL anchor needs at least one action_valid row')
    return F.kl_div(logits[mask].log_softmax(-1),base[mask].detach().log_softmax(-1),
        log_target=True,reduction='batchmean')


def arm_loss(arm,output,labels):
    original=tactical_policy_loss(output,labels)
    if arm==CONTROL:return dict(loss=original['loss'],original=original['loss'],anchor=None,parts=original)
    if arm!=ANCHOR:raise ValueError('unknown arm')
    anchor=kl_anchor(output,labels)
    return dict(loss=original['loss']+ANCHOR_WEIGHT*anchor,original=original['loss'],anchor=anchor,parts=original)


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


def behaviour(rows):
    """Action distribution and switch count from stored closed-loop decisions."""
    counts={};actions={};switches=[];decisions=0
    for row in rows:
        steps=row.get('steps') or []
        candidates=[s.get('actual_candidate') for s in steps]
        actual=[s.get('actual_action') for s in steps]
        for c in candidates:
            if c is not None:counts[str(c)]=counts.get(str(c),0)+1
        for a in actual:
            if a is not None:actions[str(a)]=actions.get(str(a),0)+1
        switches.append(sum(1 for i in range(1,len(candidates)) if candidates[i]!=candidates[i-1]))
        decisions+=len(steps)
    total=sum(counts.values()) or 1
    action_total=sum(actions.values()) or 1
    left=actions.get('-2',0)+actions.get('-1',0);right=actions.get('2',0)+actions.get('1',0)
    return dict(events=len(rows),decisions=decisions,
        candidate_counts=dict(sorted(counts.items(),key=lambda kv:int(kv[0]))),
        candidate_shares={k:round(v/total,6) for k,v in sorted(counts.items(),key=lambda kv:int(kv[0]))},
        action_counts=dict(sorted(actions.items(),key=lambda kv:int(kv[0]))),
        action_shares=dict(hold=round(actions.get('0',0)/action_total,6),
            left=round(left/action_total,6),right=round(right/action_total,6)),
        switches=dict(values=switches,median=float(statistics.median(switches)) if switches else None,
            minimum=min(switches) if switches else None,maximum=max(switches) if switches else None,
            zero_switch_events=sum(1 for s in switches if s==0)),
        note='candidates are the frozen horizontal six-class domain; actions are actual px/frame/2 in {-2,-1,0,1,2}')


def forgetting(b0,arm):
    return dict(family='F1',metric='feasible',
        b0_F1_feasible=b0['F1']['feasible'],arm_F1_feasible=arm['F1']['feasible'],
        forgetting=b0['F1']['feasible']-arm['F1']['feasible'],
        b0_F1_survived=b0['F1']['survived'],arm_F1_survived=arm['F1']['survived'],
        survived_forgetting=b0['F1']['survived']-arm['F1']['survived'],
        b0_all_feasible=b0['all']['feasible'],arm_all_feasible=arm['all']['feasible'],
        all_feasible_b0_minus_arm=b0['all']['feasible']-arm['all']['feasible'],
        b0_F2F3_feasible=b0['F2']['feasible']+b0['F3']['feasible'],
        arm_F2F3_feasible=arm['F2']['feasible']+arm['F3']['feasible'],
        F2F3_b0_minus_arm=b0['F2']['feasible']+b0['F3']['feasible']-arm['F2']['feasible']-arm['F3']['feasible'])


def summary_of(values):
    return dict(values=list(values),mean=sum(values)/len(values),minimum=min(values),
        maximum=max(values),range=max(values)-min(values))


def arm_stats(results):
    stats={}
    for arm in ARMS:
        stats[arm]={}
        for family in FAMILIES:
            entry=dict(seeds=list(SEEDS))
            for key in ('feasible','survived','goal_satisfied'):
                entry[key]=summary_of([results[f'{arm}-s{s}']['confirmation'][family][key] for s in SEEDS])
            entry['validation_feasible']=summary_of([results[f'{arm}-s{s}']['validation'][family]['feasible'] for s in SEEDS])
            entry['validation_survived']=summary_of([results[f'{arm}-s{s}']['validation'][family]['survived'] for s in SEEDS])
            entry['forgetting']=summary_of([results[f'{arm}-s{s}']['forgetting']['forgetting'] for s in SEEDS]) if family=='F1' else None
            stats[arm][family]=entry
        stats[arm]['F2F3']=dict(seeds=list(SEEDS),
            feasible=summary_of([results[f'{arm}-s{s}']['confirmation']['F2']['feasible']
                +results[f'{arm}-s{s}']['confirmation']['F3']['feasible'] for s in SEEDS]),
            survived=summary_of([results[f'{arm}-s{s}']['confirmation']['F2']['survived']
                +results[f'{arm}-s{s}']['confirmation']['F3']['survived'] for s in SEEDS]),
            note='constant-free subset of 16 events (8 F2 + 8 F3); the only confirmation subset where no constant policy completes any event')
    return stats


def paired(results):
    per_seed={}
    for seed in SEEDS:
        a=results[f'{CONTROL}-s{seed}'];b=results[f'{ANCHOR}-s{seed}'];per_seed[str(seed)]={}
        for family in FAMILIES:
            per_seed[str(seed)][family]=dict(
                feasible_delta=b['confirmation'][family]['feasible']-a['confirmation'][family]['feasible'],
                survived_delta=b['confirmation'][family]['survived']-a['confirmation'][family]['survived'],
                validation_feasible_delta=b['validation'][family]['feasible']-a['validation'][family]['feasible'])
        per_seed[str(seed)]['forgetting_delta']=b['forgetting']['forgetting']-a['forgetting']['forgetting']
        per_seed[str(seed)]['F2F3']=dict(
            feasible_delta=(b['confirmation']['F2']['feasible']+b['confirmation']['F3']['feasible'])
                -(a['confirmation']['F2']['feasible']+a['confirmation']['F3']['feasible']),
            survived_delta=(b['confirmation']['F2']['survived']+b['confirmation']['F3']['survived'])
                -(a['confirmation']['F2']['survived']+a['confirmation']['F3']['survived']))
    tally={}
    for family in FAMILIES:
        deltas=[per_seed[str(s)][family]['feasible_delta'] for s in SEEDS]
        tally[family]=dict(deltas=deltas,higher=sum(d>0 for d in deltas),equal=sum(d==0 for d in deltas),
            lower=sum(d<0 for d in deltas))
    deltas=[per_seed[str(s)]['F2F3']['feasible_delta'] for s in SEEDS]
    tally['F2F3']=dict(deltas=deltas,higher=sum(d>0 for d in deltas),equal=sum(d==0 for d in deltas),
        lower=sum(d<0 for d in deltas))
    forget=[per_seed[str(s)]['forgetting_delta'] for s in SEEDS]
    tally['forgetting_F1']=dict(deltas=forget,higher=sum(d>0 for d in forget),equal=sum(d==0 for d in forget),
        lower=sum(d<0 for d in forget))
    return dict(per_seed=per_seed,tally=tally,
        note='delta = anchor minus control on the same seed; positive feasible delta means the KL anchor kept more successes')


def assemble(run,environment=None):
    """Rebuild the round summary from the frozen run artifacts; read-only."""
    run=Path(run);out=run.parent
    p=json.loads((out/'preregistration.json').read_text(encoding='utf-8'))
    names=[f'{arm}-s{seed}' for seed in SEEDS for arm in ARMS]
    checkpoints=json.loads((run/'checkpoints.json').read_text(encoding='utf-8'))
    if sorted(checkpoints)!=sorted(names):raise ValueError('checkpoint set changed')
    training={}
    for name in names:
        ck=checkpoints[name];t=json.loads((run/'train'/name/'training.json').read_text(encoding='utf-8'))
        if file_sha(run/ck['checkpoint_path'])!=ck['checkpoint_sha256']:raise ValueError('checkpoint SHA drift: '+name)
        if t['final_head_state_sha256']!=ck['final_head_state_sha256']:raise ValueError('checkpoint record mismatch: '+name)
        if t['initial_head_state_sha256']!=ck['initial_head_state_sha256']:raise ValueError('initial record mismatch: '+name)
        if t['initial_head_state_sha256']!=p['initialization']['per_seed'][str(t['seed'])]:
            raise ValueError('initial head SHA differs from prereg: '+name)
        if t['arm']!=ck['arm'] or t['seed']!=ck['seed']:raise ValueError('arm/seed record mismatch: '+name)
        training[name]=t
    closed={}
    for name in ['B0']+names:
        for world in (('confirmation',) if name=='B0' else ('confirmation','validation')):
            payload=json.loads((run/'closed-loop'/f'{name}-{world}.json').read_text(encoding='utf-8'))
            events=payload['events']
            if r16.families(events)!=payload['families']:raise ValueError('stored aggregate mismatch: '+name+'-'+world)
            closed[(name,world)]=payload
    expected={'confirmation':p['confirmation_worlds']['events'],'validation':p['closed_loop']['validation_events']}
    if len(expected['confirmation'])!=24 or len(expected['validation'])!=8:raise ValueError('prereg event counts')
    for world,members in (('confirmation',['B0']+names),('validation',names)):
        ids=[e['event_id'] for e in expected[world]]
        families={f:sum(e['family']==f for e in expected[world]) for f in FAMILY_KEYS}
        for name in members:
            events=closed[(name,world)]['events']
            if [r['event_id'] for r in events]!=ids:
                raise ValueError('event list differs from prereg: '+name+'-'+world)
            if {f:sum(r['family']==f for r in events) for f in FAMILY_KEYS}!=families:
                raise ValueError('family counts differ from prereg: '+name+'-'+world)
    b0=closed[('B0','confirmation')]['families']
    results={}
    for name in names:
        conf=closed[(name,'confirmation')]['families'];val=closed[(name,'validation')]['families']
        results[name]=dict(arm=training[name]['arm'],seed=training[name]['seed'],confirmation=conf,validation=val,
            firsthit=dict(confirmation=conf['all']['firsthit'],validation=val['all']['firsthit']),
            forgetting=forgetting(b0,conf),
            offline=dict(train=training[name]['final']['train']['all'],
                validation=training[name]['final']['validation']['all']),
            initial_offline=dict(train=training[name]['initial']['train']['all'],
                validation=training[name]['initial']['validation']['all']),
            checkpoint_sha256=training[name]['checkpoint_sha256'],
            final_head_state_sha256=training[name]['final_head_state_sha256'],
            anchor_weight=training[name]['anchor_weight'],
            anchor_loss_at_step0=training[name].get('anchor_loss_at_step0'),
            reproduces_round16_unweighted_final_head=training[name].get('reproduces_round16_unweighted_final_head'))
    summary=dict(schema=RESULT_SCHEMA,prereg_sha256=file_sha(out/'preregistration.json'),
        seeds=list(SEEDS),arms=list(ARMS),updates=UPDATES,anchor_weight=ANCHOR_WEIGHT,
        intervention=dict(control=CONTROL,anchor=ANCHOR,
            note='single factor: 1.0*KL(base||policy) over action_valid rows in arm B only'),
        formula=dict(text_sha256=p['formula']['text_sha256'],filter_sha256=p['formula']['filter_sha256'],
            anchor_weight=p['formula']['anchor_weight'],extension_of=p['formula']['extension_of']),
        confirmation_worlds=dict(events=len(p['confirmation_worlds']['events']),
            worlds_sha256=p['confirmation_worlds']['worlds_sha256'],
            duplicate_check=p['confirmation_worlds']['duplicate_check'],per_family={f:8 for f in FAMILY_KEYS},
            constant_filter=dict(rule=p['confirmation_worlds']['constant_filter']['rule'],
                before=p['confirmation_worlds']['constant_filter']['before'],
                after=p['confirmation_worlds']['constant_filter']['after'],
                replaced=p['confirmation_worlds']['constant_filter']['replaced'])),
        initialization=dict(per_seed=p['initialization']['per_seed'],
            control_reproduces_round16=[bool(training[f'{CONTROL}-s{s}'].get('reproduces_round16_unweighted_final_head')) for s in SEEDS],
            anchor_reproduces_round16=[bool(training[f'{ANCHOR}-s{s}'].get('reproduces_round16_unweighted_final_head')) for s in SEEDS]),
        checkpoints={name:{k:v for k,v in training[name].items()
            if k not in ('losses','original_losses','anchor_losses','initial','final')}
            | dict(initial_offline={s:training[name]['initial'][s]['all'] for s in ('train','validation')},
                   final_offline={s:training[name]['final'][s]['all'] for s in ('train','validation')})
            for name in names},
        confirmation=dict(b0=b0,b0_behaviour=behaviour(closed[('B0','confirmation')]['events']),
            results={name:results[name]['confirmation'] for name in names},
            behaviour={name:behaviour(closed[(name,'confirmation')]['events']) for name in names},
            aggregate=arm_stats(results),paired=paired(results),
            subset_note='F2+F3 (16 events) is the constant-free subset where a decision is required; the 24-event aggregate is reference only',
            verification='R11 run_event replays only the actually executed actions for environment_metrics; no solve(), no teacher policy and no teacher label is read on confirmation worlds'),
        validation_control=dict(results={name:results[name]['validation'] for name in names},
            note='the old eight validation events were repeatedly exposed in R10-R16 and are themselves constant-passable (F1 right 2/2, left 2/2); continuity control only, not an independent test',
            r11_b0_external_reference='R11 reported B0 2/8 feasible on the same eight events with cuDNN TF32 true and a different checkpoint protocol'),
        forgetting={name:results[name]['forgetting'] for name in names},
        independent_test=False,formal_gate=False,deployable=False,teacher_calls=False,new_teacher_labels=False,
        game=False,tuning=False,retrained=False)
    summary['environment']=dict(training_seconds_total=sum(training[n]['optimization_seconds'] for n in names),
        device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,torch_version=torch.__version__,
        precision=dict(dtype='float32',matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision()))
    if environment:summary['environment'].update(environment)
    return summary


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
            training_data=[e for e in data if e['entry']['split']=='train']
            validation_data=[e for e in data if e['entry']['split']=='validation']
            if len(training_data)!=8 or len(validation_data)!=8:raise AssertionError('train8/validation8 required')
            if {e['entry']['event_id'] for e in training_data}&{e['entry']['event_id'] for e in validation_data}:
                raise AssertionError('split leakage')
            final16=p['initialization']['round16_unweighted_final_head_sha256']
            new_dir(dest/'train');reports={};paths={}
            for seed in SEEDS:
                for arm in ARMS:
                    name=f'{arm}-s{seed}';directory=new_dir(dest/'train'/name)
                    model=r16.fresh_policy(backbone,seed).to(device)
                    initial_sha=state_sha(model.heads.state_dict())
                    if initial_sha!=p['initialization']['per_seed'][str(seed)]:raise AssertionError('initial head mismatch')
                    write_json(directory/'initialization.json',dict(head_state_sha256=initial_sha,seed=seed,arm=arm,
                        shared_config='both arms are N1; only the extra KL anchor differs',feature_storage_shared=True,
                        anchor_weight=ANCHOR_WEIGHT if arm==ANCHOR else 0.))
                    parameters=[v for v in model.parameters() if v.requires_grad]
                    initial=wd.evaluate(model,data);optimizer=torch.optim.Adam(parameters,lr=LR)
                    losses=[];originals=[];anchors=[]
                    begin=time.perf_counter()
                    for step in range(UPDATES):
                        model.train();e=training_data[step%8];o=r14.infer(model,e)
                        labels={k:v[None] for k,v in e['labels'].items()}
                        head=arm_loss(arm,o,labels)
                        if not head['parts']['has_supervision']:raise AssertionError('all 200 updates must be supervised')
                        optimizer.zero_grad(set_to_none=True);head['loss'].backward()
                        torch.nn.utils.clip_grad_norm_(parameters,CLIP);optimizer.step()
                        losses.append(float(head['loss'].detach()));originals.append(float(head['original'].detach()))
                        if head['anchor'] is not None:anchors.append(float(head['anchor'].detach()))
                    torch.cuda.synchronize();seconds=time.perf_counter()-begin
                    final=wd.evaluate(model,data)
                    if not all(torch.equal(v,frozen[k]) for k,v in backbone.state_dict().items()):
                        raise AssertionError('backbone changed')
                    final_sha=state_sha(model.heads.state_dict());path=directory/'research-heads.pt'
                    payload=dict(schema=SCHEMA,arm=arm,seed=seed,config=asdict(model.tactical_config),
                        baseline_sha256=file_sha(r14.BASELINE),final_head_state_sha256=final_sha,
                        anchor_weight=ANCHOR_WEIGHT if arm==ANCHOR else 0.,
                        heads={k:v.detach().cpu() for k,v in model.heads.state_dict().items()})
                    with path.open('xb') as stream:torch.save(payload,stream)
                    params=dict(active=sum(v.numel() for v in parameters),
                        heads_total=sum(v.numel() for v in model.heads.parameters()),
                        total=sum(v.numel() for v in model.parameters()))
                    report=dict(arm=arm,seed=seed,initial_head_state_sha256=initial_sha,
                        final_head_state_sha256=final_sha,checkpoint_sha256=file_sha(path),
                        checkpoint_path=str(path.relative_to(dest)),updates=UPDATES,
                        anchor_weight=ANCHOR_WEIGHT if arm==ANCHOR else 0.,
                        losses=losses,original_losses=originals,anchor_losses=anchors,
                        anchor_loss_at_step0=anchors[0] if anchors else None,
                        optimization_seconds=seconds,parameters=params,backbone_unchanged=True,
                        train_only='validation labels never indexed',
                        reproduces_round16_unweighted_final_head=(final_sha==final16[str(seed)] if arm==CONTROL else None),
                        initial=r16.offline(initial),final=r16.offline(final))
                    write_json(directory/'training.json',report)
                    reports[name]=report;paths[name]=path
                    print('trained',name,'init',initial_sha[:12],'final',final_sha[:12],
                        'anchor0',report['anchor_loss_at_step0'],f'{seconds:.2f}s',flush=True)
            write_json(dest/'checkpoints.json',{name:{k:v for k,v in reports[name].items() if k not in ('losses','original_losses','anchor_losses')}
                for name in reports})
            if len(list((dest/'train').rglob('research-heads.pt')))!=6:raise AssertionError('six checkpoints required')
            for name,path in paths.items():
                if file_sha(path)!=reports[name]['checkpoint_sha256']:raise AssertionError('checkpoint SHA drift')
            new_dir(dest/'closed-loop')
            world_events=p['confirmation_worlds']['events'];validation_events=p['closed_loop']['validation_events']
            begin=time.perf_counter()
            b0=r16.BaselineNetwork(backbone,device);rows=r16.closed_loop(world_events,b0,'B0','B0')
            write_json(dest/'closed-loop'/'B0-confirmation.json',dict(events=rows,families=r16.families(rows)))
            names=[f'{arm}-s{seed}' for seed in SEEDS for arm in ARMS]
            for name in names:
                network=Network(backbone,paths[name],device,reports[name]['final_head_state_sha256'])
                confirmation=r16.closed_loop(world_events,network,'N1',name)
                write_json(dest/'closed-loop'/f'{name}-confirmation.json',
                    dict(events=confirmation,families=r16.families(confirmation)))
                continuity=r16.closed_loop(validation_events,network,'N1',name)
                write_json(dest/'closed-loop'/f'{name}-validation.json',
                    dict(events=continuity,families=r16.families(continuity)))
            torch.cuda.synchronize();evaluation_seconds=time.perf_counter()-begin
            verify(out)
            summary=assemble(dest,dict(encoding_seconds=encoding_seconds,evaluation_seconds=evaluation_seconds))
            write_json(dest/'summary.json',summary)
            write_json(dest/'sha256.json',{str(f.relative_to(dest)):file_sha(f) for f in dest.rglob('*') if f.is_file()})
            return summary
    except BaseException:
        write_json(dest/'first-failure.json',dict(traceback=traceback.format_exc(),
            policy='retained; no automatic rerun, overwrite, retune or baseline switch'))
        raise


def summarize(out=OUT):
    """Read-only rebuild of the summary from the frozen run artifacts; never trains."""
    out=Path(out);dest=new_dir(out/'run-summary')
    summary=assemble(out/'run',dict(encoding_seconds=None,evaluation_seconds=None,
        source='rebuilt read-only from run/; the crashed process wall times are not recoverable, per-checkpoint optimization seconds and every event row are intact'))
    write_json(dest/'summary.json',summary)
    write_json(dest/'sha256.json',{str(f.relative_to(dest)):file_sha(f) for f in dest.rglob('*') if f.is_file()})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preregister','run','summarize'))
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.command=='preregister':preregister(args.out)
    elif args.command=='run':run(args.out,args.device)
    else:summarize(args.out)
