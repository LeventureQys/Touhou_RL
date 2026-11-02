"""Causal temporal policy on top of a frozen v3 vision scorer."""
from __future__ import annotations
import hashlib, math, os
from dataclasses import dataclass, asdict
from typing import Any
import torch
from torch import nn

N_CANDIDATES = 18

@dataclass
class TemporalConfig:
    hidden_size: int = 256
    durations: tuple[int, ...] = (1, 3, 5)
    feature_dim: int = 256
    risk_gate_enabled: bool = False
    risk_gate_threshold: float = .1
    def __post_init__(self):
        if isinstance(self.hidden_size, bool) or not isinstance(self.hidden_size, int) or self.hidden_size != 256:
            raise ValueError("hidden_size must be exactly 256")
        if isinstance(self.feature_dim, bool) or not isinstance(self.feature_dim, int) or self.feature_dim != 256:
            raise ValueError("feature_dim must be exactly 256")
        if not isinstance(self.durations, tuple) or self.durations != (1, 3, 5):
            raise ValueError("durations must be exactly (1, 3, 5)")

class TemporalPolicy(nn.Module):
    def __init__(self, vision_model: nn.Module, config: TemporalConfig | None = None):
        super().__init__()
        self.vision_model = vision_model
        self.config = config or TemporalConfig()
        self.gru = nn.GRU(self.config.feature_dim + (N_CANDIDATES + 1) + (len(self.config.durations) + 1), self.config.hidden_size, batch_first=True)
        self.residual_head = nn.Linear(self.config.hidden_size, N_CANDIDATES)
        nn.init.zeros_(self.residual_head.weight); nn.init.zeros_(self.residual_head.bias)
        self.duration_head = nn.Linear(self.config.hidden_size, N_CANDIDATES * len(self.config.durations))
        self.risk_head = nn.Sequential(nn.Linear(self.config.feature_dim,64),nn.ReLU(),nn.Linear(64,1))
        self._freeze_vision()
    def _freeze_vision(self):
        for p in self.vision_model.parameters(): p.requires_grad_(False)
        self.vision_model.eval()
    def train(self, mode=True):
        super().train(mode); self.vision_model.eval(); return self
    def _baseline(self, features):
        head = getattr(self.vision_model, "policy_head", None)
        if head is not None: return head(features)
        score = getattr(self.vision_model, "score", None)
        if score is not None: return score(None, features=features)
        raise AttributeError("vision_model must expose policy_head or score")
    def forward_features(self, features, previous_candidate, previous_duration, hidden=None, reset=None, valid_mask=None, reset_mask=None):
        if reset is None: reset = reset_mask
        if features.ndim != 3 or features.size(-1) != self.config.feature_dim: raise ValueError("features must be [B,T,feature_dim]")
        B,T,_ = features.shape
        if B<1 or T<1 or features.dtype!=torch.float32 or not torch.isfinite(features).all():raise ValueError('finite float32 nonempty features required')
        if previous_candidate.dtype!=torch.int64 or previous_duration.dtype!=torch.int64:raise ValueError('history must be int64, never rounded labels')
        for value,name in ((reset,'reset'),(valid_mask,'valid_mask')):
            if value is not None and (value.shape!=(B,T) or value.dtype!=torch.bool):raise ValueError(name+' must be bool [B,T]')
        if hidden is not None and (hidden.shape!=(1,B,self.config.hidden_size) or not torch.isfinite(hidden).all()):raise ValueError('invalid hidden state')
        c = previous_candidate.to(device=features.device, dtype=torch.long)
        d = previous_duration.to(device=features.device, dtype=torch.long)
        if c.shape != (B,T) or d.shape != (B,T): raise ValueError("history shape mismatch")
        if torch.any((c < 0) | (c > N_CANDIDATES)): raise ValueError("candidate history out of range (BOS=18)")
        if torch.any((d < 0) | (d >= len(self.config.durations)+1)): raise ValueError("duration history out of range (BOS=3)")
        inp = torch.cat([features, torch.nn.functional.one_hot(c, N_CANDIDATES+1).float(), torch.nn.functional.one_hot(d, len(self.config.durations)+1).float()], -1)
        h = hidden
        outs=[]
        for t in range(T):
            if reset is not None:
                r = reset[:,t].to(torch.bool)
                if h is not None: h = h.clone(); h[:,r,:] = 0
            h_prev = h
            y,hnew = self.gru(inp[:,t:t+1], h); h = hnew
            if valid_mask is not None:
                v=valid_mask[:,t].to(torch.bool)
                if h is not None and (~v).any():
                    if h_prev is None:
                        h_prev = torch.zeros_like(h)
                    h = torch.where(v.view(1,-1,1), h, h_prev)
            outs.append(y[:,0])
        hs = torch.stack(outs,1)
        base = self._baseline(features.reshape(B*T,-1)).reshape(B,T,N_CANDIDATES)
        risk_logits=self.risk_head(features).squeeze(-1)
        residual=self.residual_head(hs)
        if self.config.risk_gate_enabled:
            allowed=(risk_logits.sigmoid()<self.config.risk_gate_threshold).to(residual.dtype)
            residual=residual*allowed[...,None]
        cand = base + residual
        dur = self.duration_head(hs).reshape(B,T,N_CANDIDATES,len(self.config.durations))
        return {"candidate_logits":cand, "duration_logits":dur, "hidden":h,'risk_logits':risk_logits}
    def forward_regions(self, regions, previous_candidate, previous_duration, hidden=None, reset=None, valid_mask=None):
        # Regions are [B,T,...]; encode one time slice at a time without privileged inputs.
        if not isinstance(regions, dict): raise ValueError("regions must be a mapping of pixel regions")
        T = next(iter(regions.values())).shape[1]
        feats=[]
        with torch.no_grad():
            for t in range(T): feats.append(self.vision_model.features({k:v[:,t] for k,v in regions.items()}))
        return self.forward_features(torch.stack(feats,1), previous_candidate, previous_duration, hidden, reset, valid_mask)
    def act(self, regions, previous_candidate=None, previous_duration=None, hidden=None, reset=False):
        B = next(iter(regions.values())).shape[0]
        device=next(iter(regions.values())).device
        pc=torch.full((B,1),N_CANDIDATES,dtype=torch.long,device=device) if previous_candidate is None else previous_candidate.reshape(B,1)
        pd=torch.full((B,1),len(self.config.durations),dtype=torch.long,device=device) if previous_duration is None else previous_duration.reshape(B,1)
        # A single v3 region is [B,depth,H,W,C] (5D); temporal regions are [B,T,depth,H,W,C].
        temporal_regions = {}
        for k, v in regions.items():
            if v.ndim == 5:
                temporal_regions[k] = v.unsqueeze(1)
            elif v.ndim == 6:
                temporal_regions[k] = v
            else:
                raise ValueError("regions must be 5D single-step or 6D temporal tensors")
        out=self.forward_regions(temporal_regions,pc,pd,hidden,torch.full((B,1),reset,device=device))
        c=out['candidate_logits'][:,-1].argmax(-1)
        didx=out['duration_logits'][:, -1].gather(1,c[:,None,None].expand(-1,1,len(self.config.durations))).squeeze(1).argmax(-1)
        duration_values = torch.tensor(self.config.durations, device=didx.device, dtype=torch.long)
        d = duration_values[didx]
        return c,d,out

class TemporalRunner:
    def __init__(self, policy: TemporalPolicy): self.policy=policy; self.hidden=None; self.previous_candidate=None; self.previous_duration=None
    def reset(self): self.hidden=None; self.previous_candidate=None; self.previous_duration=None
    def step(self, regions, reset=False):
        if reset:
            self.reset()
        B=next(iter(regions.values())).shape[0]; dev=next(iter(regions.values())).device
        pc=torch.full((B,),N_CANDIDATES,dtype=torch.long,device=dev) if self.previous_candidate is None else self.previous_candidate
        pd=torch.full((B,),len(self.policy.config.durations),dtype=torch.long,device=dev) if self.previous_duration is None else self.previous_duration
        c,d,out=self.policy.act(regions,pc,pd,self.hidden,reset)
        self.hidden=out['hidden']; return c,d,out
    def commit(self, action, duration):
        candidate=torch.as_tensor(action,device=self.hidden.device if self.hidden is not None else None)
        if candidate.dtype!=torch.int64 or torch.any((candidate<0)|(candidate>=N_CANDIDATES)):raise ValueError('invalid candidate')
        mapping={v:i for i,v in enumerate(self.policy.config.durations)}
        vals=duration.detach().flatten().tolist() if torch.is_tensor(duration) else [duration]
        if any(type(v) is not int or v not in mapping for v in vals):raise ValueError('duration not in config mapping')
        mapped=[mapping[v] for v in vals]
        if len(mapped)==1 and candidate.numel()>1:mapped=mapped*candidate.numel()
        duration_value=torch.as_tensor(mapped,dtype=torch.long,device=candidate.device).reshape(candidate.shape)
        self.previous_candidate=candidate;self.previous_duration=duration_value

def _sha(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def save_temporal_checkpoint(path, policy, baseline_checkpoint, metadata=None):
    if not os.path.isfile(baseline_checkpoint): raise FileNotFoundError(baseline_checkpoint)
    payload={"schema":"v200-temporal-policy/1","source_checkpoint":os.fspath(baseline_checkpoint),"source_sha256":_sha(baseline_checkpoint),"config":asdict(policy.config),"state_dict":{k:v.detach().cpu() for k,v in policy.state_dict().items()},"metadata":dict(metadata or {})}
    if os.path.exists(path):raise FileExistsError(path)
    if any(not torch.isfinite(v).all() for v in payload['state_dict'].values()):raise ValueError('nonfinite policy cannot be saved')
    temporary=os.fspath(path)+'.pending'
    torch.save(payload,temporary);os.replace(temporary,path)
def load_temporal_checkpoint(path, vision_model=None, baseline_checkpoint=None, map_location='cpu'):
    p=torch.load(path,map_location=map_location,weights_only=False)
    req=('schema','source_checkpoint','source_sha256','config','state_dict')
    if not isinstance(p,dict) or any(k not in p for k in req) or p['schema']!='v200-temporal-policy/1': raise ValueError('invalid temporal checkpoint')
    src=baseline_checkpoint or p['source_checkpoint']
    if not os.path.isfile(src) or _sha(src)!=p['source_sha256']: raise ValueError('baseline checkpoint SHA mismatch')
    if any(not torch.isfinite(v).all() for v in p['state_dict'].values() if torch.is_tensor(v)): raise ValueError('non-finite checkpoint')
    if vision_model is None:
        from .vmodels import load_vision_checkpoint
        vision_model,_=load_vision_checkpoint(src,map_location=map_location)
    policy=TemporalPolicy(vision_model,TemporalConfig(**p['config']))
    policy.load_state_dict(p['state_dict'],strict=True)
    return policy
