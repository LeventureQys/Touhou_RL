"""Round16 CPU contracts, executed before preregistration and the GPU run."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from tools.rl import v373_seed_confirm as sc
from touhou_rl.tactical_policy import TacticalConfig

ROOT=Path(__file__).resolve().parents[1]


def metric():
    return dict(accuracy=.5,moving_macro_recall=.25,macro_balanced_recall=.4,support=[2,0,0,0,0],
        recall=[1.,None,None,None,None],other_predictions=0)


def test_confirmation_worlds_new_unique_and_mirror_pairs():
    specs=sc.confirmation_specs()
    assert len(specs)==24 and len({s['template_group'] for s in specs})==12
    assert {s['split'] for s in specs}=={'confirmation'}
    assert not any(s['split']=='test' for s in specs)
    groups={}
    for s in specs:groups.setdefault(s['template_group'],[]).append(s)
    for rows in groups.values():
        assert len(rows)==2 and rows[0]['mirror']!=rows[1]['mirror']
        assert sc.signature(rows[0])==sc.signature(rows[1])
    new={sc.signature(rows[0]) for rows in groups.values()}
    assert len(new)==12
    for rows in sc.existing_signatures().values():
        assert not new & {sc.signature(s) for s in rows}
    check=sc.assert_disjoint(specs)
    assert check['duplicates']==0 and check['new_events']==24 and check['new_groups']==12
    assert check['r10_subset_of_r8'] is True and check['r9_confirmation_signatures']==12
    assert {s['family'] for s in specs if s['depletion_pressure']}=={'F3'}
    assert not any(s['depletion_pressure'] for s in specs if s['family']!='F3')


def test_structural_probes_and_unreachable_world_rejected():
    specs=sc.confirmation_specs();probes=sc.assert_reachable(specs)
    assert set(probes)=={s['event_id'] for s in specs}
    for s in specs:
        probe=probes[s['event_id']]
        if s['family']=='F1':
            assert probe['reachable_before_wall'] is True
            assert probe['frames_needed']<=s['wall_frame']-2
        elif s['family']=='F3':
            assert probe['below_threshold_at_window_end'] is True
            assert probe['goal_reachable_in_horizon'] is True
        else:
            assert probe['fire_interval']>=16 and probe['horizontal_space_px']>=60
    bad=dict(event_id='bad',family='F1',template_group='bad',seed=1,split='confirmation',mirror=False,
        start_x=190.,depletion_pressure=False,wall_frame=2,coverage=(96.,282.))
    assert sc.structural_probe(bad)['reachable_before_wall'] is False
    with pytest.raises(ValueError,match='unreachable F1'):sc.assert_reachable([bad])


def test_formula_weights_exactly_round15_preregistration():
    weights=sc.train_weights()
    r15=json.loads((sc.r14.ROOT/'weighted-round15/preregistration.json').read_text(encoding='utf-8'))
    assert weights==r15['train_weights']
    assert sc.FORMULA==r15['formula']
    assert weights['total']==640 and weights['K_present']==5
    assert weights['counts']=={'0':530,'3':44,'7':44,'12':11,'16':11}
    digest=sc.formula_digest(weights)
    assert digest==sc.formula_digest(r15['train_weights'])
    assert len(digest['spec_sha256'])==64 and len(digest['text_sha256'])==64


def test_fresh_policy_same_seed_identical_heads_no_warmstart():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(trunk_hidden=8);self.weight=nn.Parameter(torch.randn(2))
    backbone=Backbone();config=TacticalConfig(arm='N1',feature_dim=8,hidden_size=8)
    hashes=[]
    for _ in sc.ARMS:
        model=sc.fresh_policy(backbone,7,config)
        hashes.append(sc.state_sha(model.heads.state_dict()))
        assert model.heads['gru'].weight_ih_l0.requires_grad
        assert model.heads['option_embedding'].weight.requires_grad
        assert not backbone.weight.requires_grad
    assert hashes[0]==hashes[1]
    assert sc.state_sha(sc.fresh_policy(backbone,17,config).heads.state_dict())!=hashes[0]
    with torch.no_grad():model.heads['residual'].bias[0]+=1
    assert sc.state_sha(model.heads.state_dict())!=hashes[0]


def test_exclusive_json_and_directory(tmp_path):
    path=tmp_path/'a.json';sc.write_json(path,{'a':1})
    assert json.loads(path.read_text(encoding='utf-8'))=={'a':1}
    with pytest.raises(FileExistsError):sc.write_json(path,{'a':2})
    directory=sc.new_dir(tmp_path/'d')
    assert directory.is_dir()
    with pytest.raises(FileExistsError):sc.new_dir(tmp_path/'d')


def test_foreign_checkpoint_schema_rejected(tmp_path):
    path=tmp_path/'bad.pt';torch.save({'schema':sc.r14.SCHEMA},path)
    with pytest.raises(ValueError,match='research schema'):sc.Network(None,path,'cpu')


def test_module_never_calls_teacher_solver_or_rollout():
    tree=ast.parse(Path(sc.__file__).read_text(encoding='utf-8'))
    defined={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}-defined
    assert not {'solve','verify','programs','rollout','metrics'} & calls
    imported={a.name for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) for a in n.names}
    assert not {'solve','verify','programs','rollout','metrics','TacticalTeacher','TacticalTeacherV2'} & imported
    assert {'ValidationWorld','make_world','run_event'} <= imported


def test_offline_and_family_aggregation():
    report=dict(summary={split:{family:{c:metric() for c in sc.CONDITIONS}
        for family in ('all','F2')} for split in ('train','validation')})
    out=sc.offline(report)
    assert out['validation']['F2']['unmasked']['moving_macro_recall']==.25
    assert out['train']['all']['horizontal_masked']['accuracy']==.5
    rows=[dict(event_id='a',family='F1',survived=True,goal_satisfied=True,feasible=True,
            invalid_action=False,raw_unsupported_count=1,first_collision_frame=None),
        dict(event_id='b',family='F1',survived=False,goal_satisfied=False,feasible=False,
            invalid_action=False,raw_unsupported_count=0,first_collision_frame=40)]
    fam=sc.families(rows)
    assert fam['all']['events']==2 and fam['F1']['feasible']==1 and fam['F1']['survived']==1
    assert fam['F1']['firsthit']=={'a':None,'b':40}
    assert fam['F2']==dict(events=0,survived=0,goal_satisfied=0,feasible=0,invalid_action=0,
        raw_unsupported_count=0,firsthit={})
    stats=sc.seed_stats({f'{arm}-s{seed}':fam for arm in sc.ARMS for seed in sc.SEEDS})
    assert stats['weighted']['F1']['feasible']['values']==[1,1,1]
    assert stats['unweighted']['all']['survived']['range']==0


def test_preregistration_verify_detects_identity_and_world_tamper(tmp_path):
    events=[dict(event_id='x',scenario=dict(family='F1'))]
    for name in ('identity','worlds'):
        target=tmp_path/f'{name}.txt';target.write_text('a',encoding='utf-8')
        out=tmp_path/f'prereg-{name}';sc.new_dir(out)
        p=dict(schema=sc.PREREG_SCHEMA,identities={str(target):sc.file_sha(target)},
            confirmation_worlds=dict(events=events,worlds_sha256=sc.worlds_digest(events)))
        sc.write_json(out/'preregistration.json',p)
        assert sc.verify(out)['schema']==sc.PREREG_SCHEMA
        if name=='identity':target.write_text('b',encoding='utf-8')
        else:
            p['confirmation_worlds']['events'][0]['event_id']='changed'
            (out/'preregistration.json').write_text(json.dumps(p),encoding='utf-8')
        with pytest.raises(ValueError,match='frozen identity' if name=='identity' else 'confirmation worlds'):
            sc.verify(out)
