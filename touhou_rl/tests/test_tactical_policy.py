"""Synthetic engineering tests, not tactical learning/capability acceptance."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from touhou_rl.tactical_policy import (TacticalConfig, TacticalPolicy, bos_history,
    tactical_policy_loss, save_tactical_checkpoint, load_tactical_checkpoint)


class StubVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(trunk_hidden=8)
        self.policy_head = nn.Linear(8,18)
    def forward_core(self, regions):
        features = regions["pixels"].reshape(-1,8)
        return {"features":features, "policy_logits":self.policy_head(features)}


def make_policy(arm="N1"):
    torch.manual_seed(7)
    return TacticalPolicy(StubVision(), TacticalConfig(arm=arm,feature_dim=8,hidden_size=16))


def inputs(b=2,t=4):
    return torch.randn(b,t,8), torch.randn(b,t,18)


@pytest.mark.parametrize("arm",["N0","N1","B1"])
def test_zero_equivalence_frozen_base_and_predicted_train_eval(arm):
    policy = make_policy(arm)
    f, base = inputs()
    policy.train()
    assert not policy.vision_model.training
    assert all(not p.requires_grad for p in policy.vision_model.parameters())
    train = policy.forward_features(f,base)
    policy.eval()
    evaluation = policy.forward_features(f,base)
    assert torch.equal(train["policy_logits"],base)
    for key in ("policy_logits","option_logits","target","hidden"):
        assert torch.equal(train[key],evaluation[key])
    assert policy.heads["residual"].out_features == 18


def test_option_and_target_condition_actions_and_receive_gradients():
    p = make_policy()
    f,b = inputs()
    with torch.no_grad():
        p.heads["residual"].weight.normal_()
    out = p.forward_features(f,b)
    out["option_logits"].retain_grad()
    out["target"].retain_grad()
    out["policy_logits"][...,0].sum().backward()
    assert out["option_logits"].grad.abs().sum() > 0
    assert out["target"].grad.abs().sum() > 0
    reference = out["policy_logits"].detach()
    with torch.no_grad():
        p.heads["option"].bias[0] += 5
    assert not torch.equal(reference,p.forward_features(f,b)["policy_logits"])
    reference = p.forward_features(f,b)["policy_logits"].detach()
    with torch.no_grad():
        p.heads["target"][0].bias += 3
    assert not torch.equal(reference,p.forward_features(f,b)["policy_logits"])


def test_padding_even_reset_is_inert_and_has_no_feature_gradient():
    p = make_policy()
    f,b = inputs(b=1,t=4)
    valid = torch.tensor([[True,False,True,False]])
    reset = ~valid
    out = p.forward_features(f,b,valid_mask=valid,reset_mask=reset)
    compact = p.forward_features(f[:,[0,2]],b[:,[0,2]])
    assert torch.equal(out["hidden"],compact["hidden"])
    initial = torch.randn(1,1,16)
    allpad = p.forward_features(f,b,hidden=initial,valid_mask=torch.zeros_like(valid),reset_mask=torch.ones_like(valid))
    assert torch.equal(allpad["hidden"],initial)
    f.requires_grad_(); b.requires_grad_()
    p.forward_features(f,b)["policy_logits"].sum().backward()
    assert f.grad is None and b.grad is None
    assert all(x.grad is None for x in p.vision_model.parameters())


def test_reset_matches_fresh_and_causal_prefix():
    p = make_policy()
    f,b = inputs(b=1,t=4)
    reset = torch.tensor([[False,False,True,False]])
    out = p.forward_features(f,b,reset_mask=reset)
    fresh = p.forward_features(f[:,2:],b[:,2:])
    assert torch.equal(out["hidden"],fresh["hidden"])
    first = p.forward_features(f,b)
    altered = f.clone(); altered[:,2:] += 100
    changed = p.forward_features(altered,b)
    assert torch.equal(first["target"][:,:2],changed["target"][:,:2])


def test_n0_omits_hierarchy_b1_ignores_history():
    p = make_policy("N0")
    assert p.heads["residual"].in_features == 24
    p = make_policy("B1")
    f,b = inputs()
    hist = bos_history(f)
    hist["history_valid"].fill_(True)
    hist["previous_candidate"].fill_(4)
    assert torch.equal(p.forward_features(f,b)["target"],p.forward_features(f,b,hist)["target"])
    assert torch.equal(p.forward_features(f,b)["target"][:,2:],p.forward_features(f[:,2:],b[:,2:])["target"])


def test_unknown_loss_zero_all_gradients_and_no_optimizer_step_signal():
    p = make_policy(); f,b = inputs()
    out = p.forward_features(f,b)
    result = tactical_policy_loss(out,{"target":torch.full((2,4,2),float("nan")),"target_valid":torch.zeros(2,4,dtype=torch.bool)})
    assert result["loss"].item() == 0 and not result["has_supervision"]
    result["loss"].backward()
    assert all(x.grad is None or torch.count_nonzero(x.grad)==0 for x in p.parameters())


@pytest.mark.parametrize("kind",["action","option","target","termination","slack","local"])
def test_masks_independent_unknown_rows_no_gradient(kind):
    p=make_policy(); f,b=inputs(b=1,t=3)
    with torch.no_grad(): p.heads["residual"].weight.normal_()
    out=p.forward_features(f,b)
    for key in ("policy_logits","option_logits","target","termination_logits","slack"):
        out[key].retain_grad()
    mask=torch.tensor([[False,True,False]])
    labels={kind+"_valid":mask}
    if kind in {"action","option"}:
        labels[kind+"_positive"]=torch.zeros(1,3,18 if kind=="action" else 3,dtype=torch.bool)
        labels[kind+"_positive"][0,1,0]=True
    elif kind=="target": labels["target"]=torch.tensor([[[float("nan")]*2,[.1,.2],[float("nan")]*2]])
    elif kind=="termination": labels["termination"]=torch.tensor([[-999,1,-999]])
    elif kind=="slack": labels["slack_interval"]=torch.tensor([[[float("nan")]*2,[-.5,-.1],[float("nan")]*2]])
    result=tactical_policy_loss(out,labels)
    assert result["has_supervision"] and result["supervised_count"]==1
    assert result["counts"][kind]==1
    result["loss"].backward()
    for key in ("policy_logits","option_logits","target","termination_logits","slack"):
        grad=out[key].grad
        assert grad is None or torch.count_nonzero(grad[:,[0,2]])==0


def test_positive_mass_and_soft_option_and_padding_loss():
    p=make_policy(); f,b=inputs(b=1,t=2)
    out=p.forward_features(f,b,valid_mask=torch.tensor([[True,False]]))
    positive=torch.ones(1,2,18,dtype=torch.bool)
    labels={"action_positive":positive,"action_valid":torch.ones(1,2,dtype=torch.bool),
            "option_target":torch.tensor([[[.2,.3,.5],[float("nan")]*3]]),
            "option_valid":torch.ones(1,2,dtype=torch.bool)}
    result=tactical_policy_loss(out,labels)
    assert abs(result["action_loss"].item())<1e-6
    assert result["counts"]["action"]==1 and result["counts"]["option"]==1
    labels["action_positive"].zero_()
    with pytest.raises(ValueError,match="positive"): tactical_policy_loss(out,labels)


def test_pixel_cached_equivalence():
    p=make_policy(); f,b=inputs(b=1,t=2)
    with torch.no_grad(): b=torch.stack([p.vision_model.policy_head(f[:,i]) for i in range(2)],1)
    # Stub uses eight pixels; real backbone contract is checked by v3 itself.
    regions={"pixels":f.reshape(1,2,1,1,8,1)}
    assert torch.equal(p.forward_regions(regions)["policy_logits"],p.forward_features(f,b)["policy_logits"])


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_sidecar_sha_roundtrip_frozen_source_and_devices(tmp_path,monkeypatch,device):
    if device=="cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    p=make_policy(); original=deepcopy(p.vision_model)
    monkeypatch.setattr("touhou_rl.vmodels.load_vision_checkpoint",lambda *a,**kw:(deepcopy(original),{}))
    monkeypatch.setattr("touhou_rl.vmodels.vision_arch",lambda m:"v3-st")
    base=tmp_path/"base.pt"; base.write_bytes(b"bound source fixture")
    side=tmp_path/"tactical.pt"
    save_tactical_checkpoint(side,p,base)
    restored,_=load_tactical_checkpoint(side,map_location=device)
    f,b=inputs(); actual=restored.forward_features(f.to(device),b.to(device))
    assert torch.equal(actual["policy_logits"].cpu(),b)
    with pytest.raises(FileExistsError): save_tactical_checkpoint(side,p,base)
    with torch.no_grad(): p.vision_model.policy_head.bias.add_(1)
    with pytest.raises(ValueError,match="backbone"): save_tactical_checkpoint(tmp_path/"bad.pt",p,base)
    base.write_bytes(b"changed")
    with pytest.raises(ValueError,match="SHA256"): load_tactical_checkpoint(side)
