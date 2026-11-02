import torch
import pytest
from touhou_rl.tactical_metrics import event_metrics,summarize_events


def output():
    return {'policy_logits':torch.tensor([[[5.,0.]+[0.]*16,[0.,5.]+[0.]*16]]),
        'option_logits':torch.tensor([[[0.,2.,0.],[3.,0.,0.]]]),
        'target':torch.tensor([[[1.,0.],[0.,0.]]]),
        'termination_logits':torch.tensor([[[0.,4.,0.],[0.,0.,4.]]])}


def test_unknown_not_accuracy_error_and_independent_target_mask():
    o=output(); mask=torch.tensor([[True,False]])
    pos=torch.zeros_like(o['policy_logits'],dtype=torch.bool);pos[0,0,0]=True
    r=event_metrics(o,{'action_positive':pos,'action_valid':mask,
        'target':torch.tensor([[[0.,1.],[float('nan'),float('nan')]]]),'target_valid':mask})
    assert r['positive_set_agreement']==1 and r['action_unknown_count']==1
    assert r['target_angular_degrees']==pytest.approx(90)
    assert r['option_count']==0


def test_zero_target_direction_is_not_made_into_zero_angle():
    o=output(); mask=torch.tensor([[False,True]])
    r=event_metrics(o,{'target':torch.zeros(1,2,2),'target_valid':mask})
    assert r['target_count']==1 and r['target_direction_count']==0
    assert r['target_angular_degrees'] is None


def test_termination_confusion_and_padding():
    o=output();o['valid_mask']=torch.tensor([[True,False]])
    r=event_metrics(o,{'termination':torch.tensor([[1,0]]),'termination_valid':torch.ones(1,2,dtype=torch.bool)})
    assert r['termination_count']==1
    assert r['termination_confusion']==[[0,0,0],[0,1,0],[0,0,0]]


def test_equal_event_mean_not_long_event_dominated():
    r=summarize_events([{'positive_set_agreement':1.,'action_count':1},
                        {'positive_set_agreement':0.,'action_count':9}])
    assert r['positive_set_agreement']['event_mean']==.5
    assert r['positive_set_agreement']['row_weighted_mean']==.1
