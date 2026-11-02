"""Deterministic pixel-only temporal action chunk executor (also usable in stubs)."""
from .temporal_policy import TemporalRunner

class ChunkExecutor:
    """No phase/stage/privilege input; interrupt only on explicit env reset/terminal.

    A duration is a number of base decisions, not a claim about observed game
    frames. Policy is queried once per chunk; history commits the actual completed
    number only if exactly 1/3/5, otherwise resets after interruption.
    """
    def __init__(self,policy):self.runner=TemporalRunner(policy);self.pending=None;self.remaining=0;self.executed=0
    def reset(self):self.runner.reset();self.pending=None;self.remaining=0;self.executed=0
    def choose(self,regions):
        if self.remaining:
            return self.pending,False
        candidate,duration,_=self.runner.step(regions)
        if candidate.numel()!=1:raise ValueError('live chunk executor requires batch1')
        self.pending=(int(candidate.item()),int(duration.item()));self.remaining=self.pending[1];self.executed=0
        return self.pending,True
    def acknowledge(self,terminated=False,truncated=False):
        if not self.remaining:raise RuntimeError('no action pending')
        self.remaining-=1;self.executed+=1
        if terminated or truncated:
            self.reset();return
        if self.remaining==0:
            import torch
            self.runner.commit(torch.tensor([self.pending[0]],device=self.runner.hidden.device),self.executed)
            self.pending=None
