"""Runner transaction tests without game input or formal tactical labels."""
import pytest
import torch
from touhou_rl.tactical_runner import TacticalRunner
from touhou_rl.tests.test_tactical_policy import make_policy


def propose(r,frame=10):
    return r.propose(features=torch.randn(1,8),base_logits=torch.randn(1,18),observation_frame=frame)


def test_propose_cancel_no_committed_mutation_and_actual_override(monkeypatch):
    r=TacticalRunner(make_policy())
    first=propose(r)
    assert r.hidden is None and r.previous_candidate is None
    with pytest.raises(RuntimeError): propose(r)
    r.cancel(first.proposal_id)
    assert r.hidden is None and r.previous_candidate is None
    second=propose(r)
    actual=(second.candidate+1)%18
    r.commit(actual,3,12,actual_option=3,target_adopted=False,proposal_id=second.proposal_id)
    assert r.previous_candidate==actual and r.previous_hold==3 and r.previous_option==3
    assert not r.target_valid and r.previous_target==(0.,0.)
    old_hidden=r.hidden.clone()
    captured={}
    original=r.policy.forward_features
    def wrapper(features,base,history,hidden):
        captured.update(history)
        return original(features,base,history,hidden)
    monkeypatch.setattr(r.policy,"forward_features",wrapper)
    third=propose(r,18)
    assert captured["previous_candidate"].item()==actual
    assert captured["previous_hold"].item()==3
    assert captured["frame_delta"].item()==6
    assert captured["previous_option"].item()==3
    assert torch.equal(r.hidden,old_hidden)
    r.cancel(third.proposal_id)
    assert torch.equal(r.hidden,old_hidden) and r.previous_candidate==actual


def test_failed_commit_atomic_stale_duplicate_and_reset():
    r=TacticalRunner(make_policy())
    with pytest.raises(RuntimeError): r.commit(1,2,1)
    p=propose(r)
    for kwargs in ({"actual_action":18},{"actual_hold":0},{"effective_frame":9},{"proposal_id":p.proposal_id+1}):
        args=dict(actual_action=1,actual_hold=2,effective_frame=12)
        args.update(kwargs)
        with pytest.raises(ValueError): r.commit(**args)
        assert r.hidden is None and r.previous_candidate is None
    r.commit(2,2,12)
    with pytest.raises(RuntimeError): r.commit(2,2,12)
    with pytest.raises(ValueError,match="rollback"): propose(r,11)
    r.on_death()
    assert r.hidden is None and r.effective_frame is None and r.previous_target is None
    q=propose(r,1)
    with pytest.raises(ValueError,match="stale"): r.commit(1,2,2,proposal_id=p.proposal_id)
    r.on_exception()
    with pytest.raises(RuntimeError): r.commit(1,2,2,proposal_id=q.proposal_id)


def test_reset_replays_bos_and_target_only_after_commit():
    r=TacticalRunner(make_policy())
    f=torch.randn(1,8); b=torch.randn(1,18)
    first=r.propose(features=f,base_logits=b,observation_frame=0)
    assert r.previous_target is None
    r.commit(first.candidate,2,1)
    assert r.previous_target==first.target and r.target_valid
    r.reset()
    again=r.propose(features=f,base_logits=b,observation_frame=0)
    assert first.candidate==again.candidate and first.target==again.target


def test_pixels_and_features_runner_same_proposal():
    p=make_policy()
    r=TacticalRunner(p)
    f=torch.randn(1,8)
    b=p.vision_model.policy_head(f).detach()
    cached=r.propose(features=f,base_logits=b)
    r.cancel()
    pixels=r.propose({"pixels":f.reshape(1,1,1,8,1)})
    assert cached.candidate==pixels.candidate and cached.target==pixels.target
