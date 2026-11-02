from __future__ import annotations
import hashlib
import torch
import pytest
from torch import nn
from touhou_rl.temporal_policy import (
    TemporalConfig, TemporalPolicy, TemporalRunner,
    save_temporal_checkpoint, load_temporal_checkpoint,
)

class FakeVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_head = nn.Linear(256, 18)
        self.register_buffer("marker", torch.ones(1))
    def features(self, regions):
        x = next(iter(regions.values()))
        return torch.zeros(x.shape[0], 256, device=x.device)

def make_policy():
    torch.manual_seed(4)
    return TemporalPolicy(FakeVision())

def histories(b,t):
    return torch.full((b,t),18,dtype=torch.long), torch.full((b,t),3,dtype=torch.long)

def test_config_and_frozen_baseline_residual_zero():
    with pytest.raises(ValueError): TemporalConfig(hidden_size=128)
    with pytest.raises(ValueError): TemporalConfig(durations=[1,3,5])
    p=make_policy(); assert not any(x.requires_grad for x in p.vision_model.parameters())
    x=torch.randn(2,3,256); c,d=histories(2,3)
    out=p.forward_features(x,c,d)
    expected=p.vision_model.policy_head(x.reshape(-1,256)).reshape(2,3,18)
    assert torch.allclose(out["candidate_logits"],expected)

def test_sequence_step_equivalence_reset_and_padding():
    p=make_policy(); x=torch.randn(2,4,256); c,d=histories(2,4)
    full=p.forward_features(x,c,d)
    h=None; pieces=[]
    for t in range(4):
        o=p.forward_features(x[:,t:t+1],c[:,t:t+1],d[:,t:t+1],h)
        pieces.append(o["candidate_logits"]); h=o["hidden"]
    assert torch.allclose(full["candidate_logits"],torch.cat(pieces,1), atol=1e-6, rtol=1e-5)
    reset=torch.tensor([[False,True,False,False],[False,False,False,False]])
    o=p.forward_features(x,c,d,reset=reset)
    assert not torch.allclose(o["duration_logits"][0,1],p.forward_features(x,c,d)["duration_logits"][0,1])
    valid=torch.tensor([[True,True,False,False],[True,True,True,False]])
    pad=p.forward_features(x,c,d,valid_mask=valid)
    assert torch.equal(pad["hidden"][:,0],pad["hidden"][:,1]) is False

def test_runner_commit_mapping_and_reset():
    p=make_policy(); r=TemporalRunner(p)
    regions={"x":torch.zeros(2,3,8,8,3)}
    c,d,_=r.step(regions,reset=True)
    assert c.shape==(2,) and set(d.tolist()) <= {1,3,5}
    r.commit(c,3); assert torch.equal(r.previous_duration,torch.ones(2,dtype=torch.long))
    with pytest.raises(ValueError): r.commit(torch.zeros(2,dtype=torch.long),2)
    old=r.hidden; r.step(regions,reset=True); assert r.previous_candidate is None

def test_forward_regions_has_no_privileged_inputs():
    p=make_policy(); c,d=histories(2,2)
    regions={"x":torch.zeros(2,2,3,8,8,3)}
    out=p.forward_regions(regions,c,d)
    assert out["candidate_logits"].shape==(2,2,18)

def test_pixel_risk_gate_preserves_baseline_without_privileged_input():
    p=TemporalPolicy(FakeVision(),TemporalConfig(risk_gate_enabled=True))
    with torch.no_grad():
        p.residual_head.bias.fill_(2)
        for parameter in p.risk_head.parameters():parameter.zero_()
        p.risk_head[-1].bias.fill_(10)
    x=torch.randn(1,2,256);c,d=histories(1,2);out=p.forward_features(x,c,d)
    assert torch.equal(out['candidate_logits'],p.vision_model.policy_head(x))


def test_checkpoint_roundtrip_and_rejection(tmp_path):
    p=make_policy(); baseline=tmp_path/"baseline.pt"; out=tmp_path/"temporal.pt"
    torch.save({"fake":"baseline"},baseline)
    save_temporal_checkpoint(out,p,baseline)
    q=load_temporal_checkpoint(out,vision_model=FakeVision(),baseline_checkpoint=baseline)
    for a,b in zip(p.state_dict().values(),q.state_dict().values()): assert torch.equal(a,b)
    payload=torch.load(out,weights_only=False); payload["state_dict"]["gru.weight_ih_l0"][0,0]=float("nan"); torch.save(payload,out)
    with pytest.raises(ValueError): load_temporal_checkpoint(out,vision_model=FakeVision(),baseline_checkpoint=baseline)
    payload["state_dict"]["gru.weight_ih_l0"][0,0]=0; payload["source_sha256"]="0"*64; torch.save(payload,out)
    with pytest.raises(ValueError): load_temporal_checkpoint(out,vision_model=FakeVision(),baseline_checkpoint=baseline)
