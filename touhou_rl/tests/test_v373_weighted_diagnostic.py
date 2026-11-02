"""Round15 CPU contracts, executed before preregistration and GPU run."""
import copy
import math
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from tools.rl import v373_weighted_diagnostic as d
from touhou_rl.tactical_policy import TacticalPolicy,TacticalConfig,tactical_policy_loss


def labels(ys):
    return dict(action_positive=torch.nn.functional.one_hot(torch.tensor(ys),18).bool(),action_valid=torch.ones(len(ys),dtype=torch.bool))


def output(logits):
    shape=logits.shape[:-1]
    return dict(policy_logits=logits,base_logits=torch.zeros_like(logits),option_logits=torch.zeros(*shape,3),target=torch.zeros(*shape,2),termination_logits=torch.zeros(*shape,2),slack=torch.zeros(shape),valid_mask=torch.ones(shape,dtype=torch.bool))


def test_train_only_formula_normalization_and_absent():
    entries=[dict(entry={'split':'train'},labels=labels([0]*9+[7])),dict(entry={'split':'validation'},labels=labels([3]*99))]
    spec=d.weight_spec(entries)
    assert spec['counts']=={'0':9,'7':1} and spec['K_present']==2
    assert spec['raw']['0']==math.sqrt(10/18)
    assert spec['raw']['7']==math.sqrt(5)
    assert sum(spec['counts'][c]*w for c,w in spec['weights'].items())/10==pytest.approx(1)
    assert '3' not in spec['weights']


def test_cap_before_normalization():
    spec=d.weight_spec([dict(entry={'split':'train'},labels=labels([0]*999+[7]))])
    assert spec['raw']['7']==5
    assert spec['weights']['7']==pytest.approx(5/spec['normalizer'])


def test_weighted_nll_denominator_and_gradient():
    logits=torch.zeros(3,18,requires_grad=True);logits.data[0,0]=2
    lab=labels([0,7,3]);lab['action_valid'][2]=False
    o=output(logits);weights={'0':.5,'7':2.}
    result=d.action_loss(o,lab,weights)
    nll=-logits[:2].log_softmax(-1)[[0,1],[0,7]]
    expected=(nll[0]*.5+nll[1]*2)/2.5
    assert torch.allclose(result['loss'],expected)
    result['loss'].backward();assert not logits.grad[2].any()


def test_aux_untouched_and_unit_weights_match_original():
    logits=torch.randn(3,18,requires_grad=True);o=output(logits);lab=labels([0,7,0])
    lab.update(target_valid=torch.tensor([True,False,False]),target=torch.ones(3,2))
    base=tactical_policy_loss(o,lab);weighted=d.action_loss(o,lab,{'0':1.,'7':1.})
    assert torch.allclose(base['loss'],weighted['loss'])
    assert torch.equal(base['target_loss'],weighted['target_loss'])
    changed=d.action_loss(o,lab,{'0':.5,'7':2.})
    assert torch.allclose(changed['loss']-changed['action_loss'],base['loss']-base['action_loss'])


def test_single_class_valid_rows_only_and_unknown_rejected():
    lab=labels([0,7]);lab['action_positive'][0,3]=True
    with pytest.raises(ValueError,match='single class'):d.action_loss(output(torch.zeros(2,18)),lab,{'0':1.,'7':1.})
    lab['action_valid'][0]=False
    d.action_loss(output(torch.zeros(2,18)),lab,{'7':1.})
    with pytest.raises(ValueError,match='absent'):d.action_loss(output(torch.zeros(2,18)),lab,{'0':1.})


def test_no_valid_rows_and_640_contract():
    lab=labels([0]);lab['action_valid'][:]=False
    result=d.action_loss(output(torch.zeros(1,18,requires_grad=True)),lab,{})
    assert not result['has_supervision'] and result['loss']==0
    with pytest.raises(ValueError,match='640'):d.weight_spec([dict(entry={'split':'train'},labels=labels([0]))],True)


def test_exact_fresh_head_sha_and_active_n1():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(trunk_hidden=8);self.weight=nn.Parameter(torch.randn(2))
    backbone=Backbone();hashes=[]
    for _ in d.ARMS:
        torch.manual_seed(7);m=TacticalPolicy(backbone,TacticalConfig(arm='N1',feature_dim=8,hidden_size=8))
        hashes.append(d.state_sha(m.heads.state_dict()))
        assert m.heads['gru'].weight_ih_l0.requires_grad and m.heads['option_embedding'].weight.requires_grad
        assert not backbone.weight.requires_grad
    assert hashes[0]==hashes[1]
    with torch.no_grad():m.heads['residual'].bias[0]+=1
    assert d.state_sha(m.heads.state_dict())!=hashes[0]


def test_exclusive_output_and_research_schema(tmp_path):
    path=tmp_path/'a.json';d.write_json(path,{'a':1})
    with pytest.raises(FileExistsError):d.write_json(path,{'a':2})
    path=tmp_path/'bad.pt';torch.save({'schema':d.r14.SCHEMA},path)
    with pytest.raises(ValueError,match='research schema'):d.Network(None,path,'cpu')
