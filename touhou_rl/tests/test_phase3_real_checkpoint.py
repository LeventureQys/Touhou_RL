import json
import torch
from touhou_rl.temporal_policy import TemporalPolicy,TemporalConfig,save_temporal_checkpoint,load_temporal_checkpoint
from touhou_rl.vmodels import load_vision_checkpoint
from pathlib import Path

def test_real_v3_temporal_checkpoint_roundtrip(tmp_path):
    baseline=Path('Document/Update/Alpha/v2.0.0/phase2/full-phase2-r11/head-only-controlled/policy.pt')
    model,_=load_vision_checkpoint(baseline,map_location='cpu');policy=TemporalPolicy(model,TemporalConfig())
    out=tmp_path/'gru.pt';save_temporal_checkpoint(out,policy,baseline,metadata={'input':'pixels-only'})
    loaded=load_temporal_checkpoint(out,map_location='cpu')
    x=torch.randn(1,2,256);c=torch.full((1,2),18,dtype=torch.long);d=torch.full((1,2),3,dtype=torch.long)
    with torch.no_grad():
        a=policy.forward_features(x,c,d);b=loaded.forward_features(x,c,d)
    assert torch.equal(a['candidate_logits'],b['candidate_logits'])
    assert torch.equal(a['duration_logits'],b['duration_logits'])
