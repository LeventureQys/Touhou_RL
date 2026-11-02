import torch
from touhou_rl.phase3_executor import ChunkExecutor

class FakeRunner:
    def __init__(self):self.calls=0;self.commits=[];self.hidden=torch.ones(1,1,256)
    def step(self,regions):self.calls+=1;return torch.tensor([3]),torch.tensor([3]),{}
    def commit(self,a,d):self.commits.append((a.tolist(),d))
    def reset(self):self.hidden=None;self.commits=[]

def test_chunk_executes_same_action_and_commits_only_actual_duration():
    x=ChunkExecutor(None);x.runner=FakeRunner()
    for i in range(3):
        action,queried=x.choose({'pixels':'stub'})
        assert action==(3,3) and queried==(i==0)
        x.acknowledge()
    assert x.runner.calls==1 and x.runner.commits==[([3],3)]

def test_terminal_clears_pending_hidden_and_history():
    x=ChunkExecutor(None);x.runner=FakeRunner();x.choose({});x.acknowledge(terminated=True)
    assert not x.remaining and x.pending is None and x.runner.hidden is None
