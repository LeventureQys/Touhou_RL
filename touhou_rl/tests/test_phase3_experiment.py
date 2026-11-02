import copy
import numpy as np
import torch
from torch import nn
from touhou_rl.temporal_policy import TemporalPolicy
from touhou_rl.phase3_experiment import train,infer
from touhou_rl.phase3_losses import rank_per_frame,duration_loss

class Vision(nn.Module):
    def __init__(self):super().__init__();self.policy_head=nn.Linear(256,18)

def test_temporal_training_freezes_encoder_and_matches_reset_budget():
    torch.manual_seed(2);f=np.random.default_rng(2).normal(size=(24,256)).astype(np.float32)
    pc=np.full(24,18);pd=np.full(24,3);values=np.tile(np.arange(18,dtype=np.float32),(24,1));targets=np.tile(np.arange(18)%3,(24,1));valid=np.ones((24,18),bool)
    exp=dict(learning_rate=.001,seed=7,epochs=1,burn_in=2,danger_weight=5.,anchor_weight=1.,duration_loss_weight=.3,candidate_correction_ce=True)
    model=TemporalPolicy(Vision());control=copy.deepcopy(model)
    a=train(model,f,[list(range(12)),list(range(12,24))],pc,pd,values,targets,valid,np.ones(24)*10,f,values,torch.device('cpu'),exp)
    b=train(control,f,[list(range(12)),list(range(12,24))],pc,pd,values,targets,valid,np.ones(24)*10,f,values,torch.device('cpu'),exp,True)
    assert a['steps']==b['steps']==2 and a['frozen_before']==a['frozen_after']
    scores,dur=infer(model,f,[list(range(12)),list(range(12,24))],pc,pd,torch.device('cpu'))
    assert scores.shape==(24,18) and dur.shape==(24,18,3) and np.isfinite(scores).all()

def test_rank_and_duration_mask_losses_are_finite():
    scores=torch.randn(2,3,18,requires_grad=True);values=torch.arange(18).float().expand(2,3,18)
    rank=rank_per_frame(scores,values);assert rank.shape==(2,3)
    dl=torch.randn(2,3,18,3,requires_grad=True);targets=torch.zeros(2,3,18,dtype=torch.long);valid=torch.ones_like(targets,dtype=torch.bool)
    loss=rank.mean()+duration_loss(dl,targets,valid);loss.backward();assert torch.isfinite(scores.grad).all()
