import json
from pathlib import Path
import pytest
from touhou_rl.phase2_chunk_experiment import _load_protocol, _stratified_a


def test_sealed_episode_protocol_is_accepted(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"episodes":{"train":[1,2],"dev":[3],"test":[4]}, "experiment":{}}))
    value = _load_protocol(path)
    assert value["episodes"]["test"] == [4]


def test_mask_rejects_incomplete_targets():
    import numpy as np, torch
    from touhou_rl.phase2_chunk_experiment import _loss_masked
    mask=np.ones((2,18),bool);mask[0,3]=False
    with pytest.raises(ValueError):_loss_masked(torch.zeros((2,18),requires_grad=True),np.zeros((2,18)),mask,torch.device('cpu'))


def test_stratified_sampling_is_unique_and_spans_stages():
    import numpy as np
    class Fake:
        def labels(self,ids):return {'meta':np.array([[0,i//6,0,0,0,0,0,0] for i in ids])}
    selected=_stratified_a(range(12),10,Fake())
    assert len(selected)==len(set(selected))==10
    assert any(i>=6 for i in selected)


def test_matched_budget_and_frozen_features(monkeypatch):
    import numpy as np, torch
    import touhou_rl.phase2_chunk_experiment as tool
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Linear(2,2);self.policy_head=torch.nn.Linear(2,18)
            self.config=type('Config',(),{'region_sizes':{}})()
        def features(self,x):return self.encoder(x)
    calls=[]
    def gather(data,ids,*args):calls.append((data,len(ids)));return np.ones((len(ids),2),np.float32),{}
    monkeypatch.setattr(tool,'_gather',gather)
    monkeypatch.setattr(tool.vision,'regions_to_batch',lambda x,dev:torch.tensor(x,device=dev))
    values=np.tile(np.arange(18,dtype=np.float32),(40,1));mask=np.ones((40,18),bool)
    for mixed in (True,False):
        calls.clear();m=Fake();steps,before,after=tool._train_steps(m,'A','student',list(range(32)),list(range(8)),values,values,mask,torch.device('cpu'),include_student=mixed,seed=7,batch_size=16,anchor_weight=10.,epochs=2)
        assert steps==4 and before==after
        assert sum(n for _,n in calls)==64
        assert sum(n for data,n in calls if data=='student')==(16 if mixed else 0)


def test_danger_weighted_rank_matches_equal_weights():
    import numpy as np,torch
    from touhou_rl.phase2_chunk_experiment import _loss_masked
    scores=torch.randn(3,18,requires_grad=True);target=np.tile(np.arange(18,dtype=np.float32),(3,1));mask=np.ones_like(target,bool)
    ordinary=_loss_masked(scores,target,mask,torch.device('cpu'))
    weighted=_loss_masked(scores,target,mask,torch.device('cpu'),np.ones(3))
    assert torch.allclose(ordinary,weighted)
    danger=_loss_masked(scores,target,mask,torch.device('cpu'),np.array([5,1,1]));danger.backward()
    assert torch.isfinite(scores.grad).all()


def test_protocol_rejects_overlap(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"episodes":{"train":[1],"dev":[1],"test":[2]}}))
    with pytest.raises(ValueError): _load_protocol(path)
