"""Round17 CPU contracts, executed before preregistration and the GPU run."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from tools.rl import v373_kl_anchor as ka
from tools.rl import v373_seed_confirm as r16
from touhou_rl.tactical_policy import TacticalConfig, TacticalPolicy, tactical_policy_loss
from touhou_rl.tactical_synthetic_closed_loop import run_event
from touhou_rl.tactical_teacher_validation import make_world

ROOT=Path(__file__).resolve().parents[1]
OLD=('hold','right','left')


class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(trunk_hidden=8);self.weight=nn.Parameter(torch.randn(2))


def toy_model(seed=3):
    torch.manual_seed(seed)
    return TacticalPolicy(Backbone(),TacticalConfig(arm='N1',feature_dim=8,hidden_size=8))


def toy_batch(model,rows=80,seed=5):
    torch.manual_seed(seed)
    features=torch.randn(1,rows,8);base=torch.randn(1,rows,18)
    output=model.forward_features(features,base,valid_mask=torch.ones(1,rows,dtype=torch.bool))
    positive=torch.zeros(1,rows,18,dtype=torch.bool);positive[...,0]=True
    labels=dict(action_valid=torch.ones(1,rows,dtype=torch.bool),action_positive=positive)
    return output,labels


def metric():
    return dict(accuracy=.5,moving_macro_recall=.25,macro_balanced_recall=.4,support=[2,0,0,0,0],
        recall=[1.,None,None,None,None],other_predictions=0)


def families(feasible,survived=None,goal=None):
    survived=feasible if survived is None else survived
    goal=feasible if goal is None else goal
    return dict(events=8,survived=survived,goal_satisfied=goal,feasible=feasible,invalid_action=0,
        raw_unsupported_count=0,firsthit={})


def test_confirmation_worlds_new_unique_mirrored_and_disjoint():
    specs=ka.confirmation_specs()
    assert len(specs)==24 and len({s['template_group'] for s in specs})==12
    assert {s['split'] for s in specs}=={'confirmation'}
    assert not any(s['split']=='test' for s in specs)
    groups={}
    for s in specs:groups.setdefault(s['template_group'],[]).append(s)
    for rows in groups.values():
        assert len(rows)==2 and rows[0]['mirror']!=rows[1]['mirror']
        assert ka.signature(rows[0])==ka.signature(rows[1])
    assert len({ka.signature(v[0]) for v in groups.values()})==12
    old=ka.existing_signatures()
    assert len({ka.signature(s) for s in old['r16']})==12
    union=set().union(*[set(ka.signature(s) for s in rows) for rows in old.values()])
    assert not {ka.signature(v[0]) for v in groups.values()} & union
    check=ka.assert_disjoint(specs)
    assert check['duplicates']==0 and check['new_events']==24 and check['new_groups']==12
    assert check['previous_signatures_union']==45 and check['r16_confirmation_signatures']==12
    assert check['r10_subset_of_r8'] is True
    assert {s['family'] for s in specs if s['depletion_pressure']}=={'F3'}
    assert not any(s['depletion_pressure'] for s in specs if s['family']!='F3')


def test_constant_filter_rejects_every_constant_solvable_world():
    result=ka.construction_filter()
    assert result['selection']==ka.FILTERED
    before=result['before']['counts'];after=result['after']['counts']
    assert before['F1']['right']['feasible']==8 and before['F1']['left']['feasible']==8
    assert before['all']['right']['feasible']==8 and before['all']['hold']['feasible']==2
    assert len(result['before']['events_with_any_constant'])>0
    for family in ('all','F1','F2','F3'):
        for label in OLD:
            assert before[family][label]['events']==(24 if family=='all' else 8)
            assert after[family][label]['feasible']==0
            assert after[family][label]['events_passing']==[]
    assert after['all']['hold']['feasible']==0
    assert result['after']['unique_events_with_any_constant']==0
    assert len(result['steps'])==4 and {s['family'] for s in result['steps']}=={'F1'}
    assert {s['index'] for s in result['steps']}=={0,1,2,3}
    for step in result['steps']:
        assert step['rejected_worlds'] and step['rejected_because']
        assert set(step['rejected_because'])<=set(OLD)
    rows=ka.constant_probe(ka.confirmation_specs())
    for spec in ka.confirmation_specs():
        assert not any(r['feasible'] for r in rows if r['event_id']==spec['event_id'])
    assert len({r['policy'] for r in rows})==3 and len(rows)==72


def test_structural_probes_and_unreachable_world_rejected():
    specs=ka.confirmation_specs();probes=ka.structural_probes(specs)
    assert set(probes)=={s['event_id'] for s in specs}
    for s in specs:
        probe=probes[s['event_id']]
        if s['family']=='F1':
            assert probe['reachable_before_wall'] is True
            assert probe['frames_needed']<=s['wall_frame']-2
            assert s['internal_gap'] is True and tuple(s['coverage'])==(8.,376.)
        elif s['family']=='F3':
            assert probe['below_threshold_at_window_end'] is True
            assert probe['goal_reachable_in_horizon'] is True
            assert s['window'][1]<=make_world(s).horizon-30
        else:
            assert probe['fire_interval']>=16 and probe['horizontal_space_px']>=60
    bad=dict(event_id='bad',family='F1',template_group='bad',seed=1,split='confirmation',mirror=False,
        start_x=190.,depletion_pressure=False,wall_frame=2,coverage=(96.,282.))
    with pytest.raises(ValueError,match='unreachable F1'):
        r16.assert_reachable([bad])


def test_constant_rollout_matches_the_closed_loop_runner():
    specs=[s for s in ka.confirmation_specs() if s['event_id'] in ('klconf-F1-00-m0','klconf-F2-01-m1')]
    assert len(specs)==2
    for spec in specs:
        world=make_world(spec)
        for label,arm,candidate,action in ka.CONSTANTS:
            row=run_event(world,arm,'horizontal_masked',None,candidate,None)
            probe=ka.constant_rollout(spec,action)
            assert row['feasible']==probe['feasible']
            assert row['survived']==probe['survived']
            assert row['goal_satisfied']==probe['goal_satisfied']
            assert row['first_collision_frame']==probe['first_collision_frame']
            assert row['decisions']==probe['decisions']
            assert row['family']==probe['family']


def test_anchor_is_the_frozen_local_kl_term_and_zero_at_initialisation():
    model=toy_model();output,labels=toy_batch(model)
    anchor=ka.kl_anchor(output,labels)
    assert float(anchor.detach())==0.
    local=tactical_policy_loss(output,{**labels,'local_valid':labels['action_valid']},
        weights=dict(action=0.,option=0.,target=0.,termination=0.,slack=0.))['local_loss']
    assert torch.allclose(anchor,local,atol=1e-12)
    with torch.no_grad():model.heads['residual'].bias.add_(torch.arange(18,dtype=model.heads['residual'].bias.dtype)*.5)
    output,_=toy_batch(model)
    moved=ka.kl_anchor(output,labels)
    assert float(moved.detach())>1e-6
    assert torch.allclose(moved,tactical_policy_loss(output,{**labels,'local_valid':labels['action_valid']},
        weights=dict(action=0.,option=0.,target=0.,termination=0.,slack=0.))['local_loss'],atol=1e-12)
    assert output['base_logits'].requires_grad is False
    assert output['policy_logits'].requires_grad is True


def test_anchor_averages_only_action_valid_rows():
    model=toy_model();torch.manual_seed(11)
    features=torch.randn(1,4,8);base=torch.randn(1,4,18)
    valid=torch.tensor([[True,False,True,False]])
    output=model.forward_features(features,base,valid_mask=torch.ones(1,4,dtype=torch.bool))
    positive=torch.zeros(1,4,18,dtype=torch.bool);positive[...,3]=True
    labels=dict(action_valid=valid,action_positive=positive)
    anchor=ka.kl_anchor(output,labels)
    rows=output['policy_logits'][0][valid[0]].log_softmax(-1)
    ref=output['base_logits'][0][valid[0]].detach().log_softmax(-1)
    manual=(ref.exp()*(ref-rows)).sum(-1).mean()
    assert torch.allclose(anchor,manual,atol=1e-12)
    garbage=base.clone();garbage[0][~valid[0]]=500.
    other=model.forward_features(features,garbage,valid_mask=torch.ones(1,4,dtype=torch.bool))
    assert torch.allclose(ka.kl_anchor(other,labels),anchor,atol=1e-12)
    with pytest.raises(ValueError,match='action_valid'):
        ka.kl_anchor(output,dict(action_valid=torch.zeros(1,4,dtype=torch.bool),action_positive=positive))


def test_arm_loss_control_is_untouched_and_anchor_adds_exactly_one_weight():
    assert ka.ANCHOR_WEIGHT==1. and ka.CONTROL=='no-anchor' and ka.ARMS==('no-anchor','kl-anchor')
    model=toy_model();output,labels=toy_batch(model)
    with torch.no_grad():model.heads['residual'].bias.add_(.3)
    output,_=toy_batch(model)
    control=ka.arm_loss(ka.CONTROL,output,labels)
    reference=tactical_policy_loss(output,labels)
    assert control['anchor'] is None
    assert torch.equal(control['loss'],reference['loss'])
    anchored=ka.arm_loss(ka.ANCHOR,output,labels)
    assert torch.allclose(anchored['loss'],reference['loss']+1.*ka.kl_anchor(output,labels),atol=1e-12)
    assert torch.equal(anchored['original'],reference['loss'])
    with pytest.raises(ValueError,match='unknown arm'):
        ka.arm_loss('weighted',output,labels)


def test_fresh_policy_same_seed_identical_heads_and_round16_initial_state():
    class Backbone2(nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(trunk_hidden=8);self.weight=nn.Parameter(torch.randn(2))
    config=TacticalConfig(arm='N1',feature_dim=8,hidden_size=8)
    hashes=[]
    for _ in ka.ARMS:
        model=r16.fresh_policy(Backbone2(),7,config);hashes.append(ka.state_sha(model.heads.state_dict()))
        assert model.heads['gru'].weight_ih_l0.requires_grad
        assert model.heads['residual'].bias.requires_grad
    assert hashes[0]==hashes[1]
    assert ka.state_sha(r16.fresh_policy(Backbone2(),17,config).heads.state_dict())!=hashes[0]
    initial=ka.round16_initial_heads()
    assert set(initial)=={'7','17','27'}
    assert r16.initial_head_shas()==initial


def test_exclusive_json_and_directory(tmp_path):
    path=tmp_path/'a.json';ka.write_json(path,{'a':1})
    assert json.loads(path.read_text(encoding='utf-8'))=={'a':1}
    with pytest.raises(FileExistsError):ka.write_json(path,{'a':2})
    directory=ka.new_dir(tmp_path/'d')
    assert directory.is_dir()
    with pytest.raises(FileExistsError):ka.new_dir(tmp_path/'d')


def test_foreign_checkpoint_schema_rejected(tmp_path):
    path=tmp_path/'bad.pt';torch.save({'schema':r16.SCHEMA},path)
    with pytest.raises(ValueError,match='research schema'):ka.Network(None,path,'cpu')
    torch.save({'schema':ka.SCHEMA,'arm':'weighted','seed':7},path)
    with pytest.raises(ValueError,match='research arm/seed'):ka.Network(None,path,'cpu')


def test_module_never_calls_teacher_solver_or_rollout():
    tree=ast.parse(Path(ka.__file__).read_text(encoding='utf-8'))
    defined={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}-defined
    assert not {'solve','verify','programs','rollout','metrics'} & calls
    imported={a.name for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) for a in n.names}
    assert not {'solve','verify','programs','rollout','metrics','TacticalTeacher','TacticalTeacherV2'} & imported
    assert {'ValidationWorld','make_world','advance','definitions','initial_state'}<=imported
    assert 'run_event' not in imported


def test_evaluation_starts_only_after_all_six_checkpoints_are_frozen():
    source=Path(ka.__file__).read_text(encoding='utf-8')
    frozen=source.index("write_json(dest/'checkpoints.json'")
    counted=source.index('six checkpoints required')
    evaluated=source.index("new_dir(dest/'closed-loop')")
    assert frozen<counted<evaluated
    assert source.index('for step in range(UPDATES)')<frozen


def test_behaviour_action_distribution_and_switches():
    steps=lambda candidates,actions:[dict(actual_candidate=c,actual_action=a) for c,a in zip(candidates,actions)]
    rows=[dict(steps=steps([0,0,3,3],[0,0,2,2])),
        dict(steps=steps([0,7,7,0],[0,-2,-2,0])),
        dict(steps=steps([0,0,0,0],[0,0,0,0]))]
    result=ka.behaviour(rows)
    assert result['events']==3 and result['decisions']==12
    assert result['candidate_counts']=={'0':8,'3':2,'7':2}
    assert result['action_counts']=={'-2':2,'0':8,'2':2}
    assert result['action_shares']=={'hold':round(8/12,6),'left':round(2/12,6),'right':round(2/12,6)}
    assert result['switches']=={'values':[1,2,0],'median':1.,'minimum':0,'maximum':2,'zero_switch_events':1}
    assert ka.behaviour([])['decisions']==0


def test_forgetting_arm_stats_and_paired_deltas():
    b0={family:families(4) for family in ka.FAMILIES}
    b0['F1']=families(0)
    results={}
    for name,base in (('no-anchor-s7',1),('kl-anchor-s7',3),('no-anchor-s17',2),('kl-anchor-s17',2),
                      ('no-anchor-s27',0),('kl-anchor-s27',4)):
        confirmation={family:families(base) for family in ka.FAMILIES}
        validation={family:families(base) for family in ka.FAMILIES}
        results[name]=dict(arm=name.split('-s')[0],seed=int(name.split('-s')[1]),confirmation=confirmation,
            validation=validation,forgetting=ka.forgetting(b0,confirmation))
    assert results['kl-anchor-s7']['forgetting']['forgetting']==-3
    assert results['no-anchor-s27']['forgetting']['forgetting']==0
    assert results['no-anchor-s27']['forgetting']['F2F3_b0_minus_arm']==8-0
    stats=ka.arm_stats(results)
    assert stats['kl-anchor']['F1']['feasible']['values']==[3,2,4]
    assert stats['kl-anchor']['F1']['feasible']['range']==2
    assert stats['no-anchor']['F2F3']['feasible']['values']==[2,4,0]
    assert stats['kl-anchor']['F1']['forgetting']['values']==[-3,-2,-4]
    paired=ka.paired(results)
    assert paired['tally']['F1']['deltas']==[2,0,4]
    assert paired['tally']['F1']['higher']==2 and paired['tally']['F1']['equal']==1
    assert paired['tally']['forgetting_F1']['deltas']==[-2,0,-4]
    assert paired['per_seed']['7']['F2F3']['feasible_delta']==4


def fake_run(root):
    out=root/'ev';ka.new_dir(out)
    specs=ka.confirmation_specs()
    validation=[dict(event_id=f'validation-F{n}-structure-{i:02d}-m0',family=f'F{n}')
        for n,count in ((1,2),(2,4),(3,2)) for i in range(count)]
    initial={str(seed):f'{seed:064x}' for seed in ka.SEEDS}
    prereg=dict(schema=ka.PREREG_SCHEMA,initialization=dict(per_seed=initial),
        formula=dict(text_sha256='a'*64,filter_sha256='b'*64,anchor_weight=1.,
            extension_of=dict(reason='fake')),
        confirmation_worlds=dict(events=[dict(event_id=s['event_id'],family=s['family']) for s in specs],
            worlds_sha256='c'*64,duplicate_check=dict(duplicates=0),
            constant_filter=dict(rule='fake rule',before=dict(counts={}),after=dict(counts={}),replaced=[])),
        closed_loop=dict(validation_events=validation))
    ka.write_json(out/'preregistration.json',prereg)
    run=out/'run';ka.new_dir(run);ka.new_dir(run/'train');ka.new_dir(run/'closed-loop')
    names=[f'{arm}-s{seed}' for seed in ka.SEEDS for arm in ka.ARMS];checkpoints={}
    for number,name in enumerate(names):
        arm,seed=name.rsplit('-s',1);seed=int(seed)
        directory=ka.new_dir(run/'train'/name)
        head=directory/'research-heads.pt';head.write_bytes(f'heads-{name}'.encode())
        final=f'{number+1:064x}'
        training=dict(arm=arm,seed=seed,initial_head_state_sha256=initial[str(seed)],
            final_head_state_sha256=final,checkpoint_sha256=ka.file_sha(head),
            checkpoint_path=str(head.relative_to(run)),optimization_seconds=10.+number,
            anchor_weight=1. if arm==ka.ANCHOR else 0.,anchor_loss_at_step0=None,
            reproduces_round16_unweighted_final_head=True if arm==ka.CONTROL else None,
            initial=dict(train={'all':metric()},validation={'all':metric()}),
            final=dict(train={'all':metric()},validation={'all':metric()}))
        ka.write_json(directory/'training.json',training)
        checkpoints[name]={k:v for k,v in training.items() if k!='losses'}
    ka.write_json(run/'checkpoints.json',checkpoints)
    def rows(events,offset):
        out=[]
        for position,event in enumerate(events):
            feasible=(position+offset)%3==0
            out.append(dict(event_id=event['event_id'],family=event['family'],condition='horizontal_masked',
                first_collision_frame=None if feasible else 40,survived=feasible,goal_satisfied=feasible,
                feasible=feasible,invalid_action=False,raw_unsupported_count=0,decisions=2,
                steps=[dict(actual_candidate=0,actual_action=0),dict(actual_candidate=3,actual_action=2)]))
        return out
    for position,name in enumerate(['B0']+names):
        worlds=[('confirmation',specs)]+([('validation',validation)] if name!='B0' else [])
        for world,events in worlds:
            payload=rows(events,position)
            ka.write_json(run/'closed-loop'/f'{name}-{world}.json',
                dict(events=payload,families=r16.families(payload)))
    return out,run,names,specs,validation


def test_assemble_rebuilds_summary_and_detects_tampering(tmp_path):
    out,run,names,specs,validation=fake_run(tmp_path)
    summary=ka.assemble(run)
    assert summary['schema']==ka.RESULT_SCHEMA
    assert set(summary['confirmation']['results'])==set(names)
    assert set(summary['confirmation']['behaviour'])==set(names)
    assert summary['confirmation']['b0']['F1']['feasible']==3
    assert summary['validation_control']['results']['no-anchor-s7']['all']['events']==8
    b0_f1=summary['confirmation']['b0']['F1']['feasible']
    for name in names:
        arm_f1=summary['confirmation']['results'][name]['F1']['feasible']
        assert summary['forgetting'][name]['forgetting']==b0_f1-arm_f1
        assert summary['confirmation']['behaviour'][name]['switches']['median']==1.
    assert summary['environment']['training_seconds_total']==sum(10.+i for i in range(6))
    assert summary['confirmation']['aggregate']['kl-anchor']['F1']['feasible']['values']
    assert summary['confirmation']['paired']['tally']['F1']['deltas']
    path=run/'closed-loop'/'no-anchor-s7-confirmation.json'
    payload=json.loads(path.read_text(encoding='utf-8'));payload['events'][0]['feasible']=not payload['events'][0]['feasible']
    path.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ValueError,match='stored aggregate mismatch'):
        ka.assemble(run)
    out2,run2,_,_,_=fake_run(tmp_path/'second')
    training=json.loads((run2/'train'/'kl-anchor-s17'/'training.json').read_text(encoding='utf-8'))
    training['final_head_state_sha256']='d'*64
    (run2/'train'/'kl-anchor-s17'/'training.json').write_text(json.dumps(training),encoding='utf-8')
    with pytest.raises(ValueError,match='checkpoint record mismatch'):
        ka.assemble(run2)
    out3,run3,_,_,_=fake_run(tmp_path/'third')
    path=run3/'closed-loop'/'B0-confirmation.json'
    payload=json.loads(path.read_text(encoding='utf-8'));payload['events'].reverse()
    payload['families']=r16.families(payload['events'])
    path.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ValueError,match='event list differs from prereg'):
        ka.assemble(run3)
