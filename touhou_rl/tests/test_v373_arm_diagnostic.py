"""CPU contract tests; frozen before execution by R14 preregistration."""
import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from tools.rl import v373_arm_diagnostic as d
from touhou_rl.tactical_policy import TacticalPolicy,TacticalConfig


def test_confusion_other_and_balanced():
    m=d.confusion([0,0,7,3,16,12],[0,1,7,0,16,12])
    assert m['matrix'][0]==[1,0,0,0,0,1]
    assert m['support']==[2,1,1,1,1]
    assert m['macro_balanced_recall']==.7
    assert m['agreement']==4/6


def test_absent_classes_not_fabricated():
    m=d.confusion([0],[0])
    assert m['recall']==[1,None,None,None,None]
    assert m['macro_balanced_recall']==1
    with pytest.raises(ValueError):d.confusion([9],[0])


def test_exclusive_output(tmp_path):
    p=tmp_path/'a.json';d.write_json(p,{'a':1})
    with pytest.raises(FileExistsError):d.write_json(p,{'a':2})


def test_original_records_contract():
    entries=[dict(split='train' if i<8 else 'validation',rows=80,teacher_verified=True,group=str(i)) for i in range(16)]
    spec=dict(schema='v373-synthetic-rendered-diagnostic/1',events=entries)
    assert len(d.validate_spec(spec))==16
    bad=copy.deepcopy(spec);bad['events'][0]['split']='test'
    with pytest.raises(ValueError):d.validate_spec(bad)
    bad=copy.deepcopy(spec);bad['events'][8]['group']='0'
    with pytest.raises(ValueError):d.validate_spec(bad)
    bad=copy.deepcopy(spec);bad['real_pixel_training']=True
    with pytest.raises(ValueError):d.validate_spec(bad)


def test_fresh_arm_branches_and_frozen_backbone():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(trunk_hidden=8);self.weight=nn.Parameter(torch.randn(2))
    backbone=Backbone();original=backbone.weight.clone()
    for arm in d.ARMS:
        torch.manual_seed(7);model=TacticalPolicy(backbone,TacticalConfig(arm=arm,feature_dim=8,hidden_size=8))
        assert not backbone.weight.requires_grad
        assert model.heads['gru'].weight_ih_l0.requires_grad==(arm!='B1')
        assert model.heads['memoryless'].weight.requires_grad==(arm=='B1')
        assert model.heads['option_embedding'].weight.requires_grad==(arm!='N0')
        assert not model.heads['residual'].weight.any()
        assert torch.equal(original,backbone.weight)
        x=torch.randn(1,3,8);base=torch.randn(1,3,18)
        result=model.forward_features(x,base)
        assert result['policy_logits'].shape==(1,3,18)


def test_callback_rejects_wrong_research_schema(tmp_path):
    p=tmp_path/'bad.pt';torch.save({'schema':'v373-synthetic-render-only-heads/1'},p)
    with pytest.raises(ValueError,match='research schema'):d.ArmNetwork(None,p,'cpu')
