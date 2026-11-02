import torch
from tools.rl.v373_action_recall import class_metrics


def test_hold_majority_does_not_hide_zero_moving_recall():
    target=torch.tensor([0]*9+[3])
    result=class_metrics(target,torch.zeros_like(target))
    assert abs(result['accuracy']-.9)<1e-6
    assert result['moving_macro_recall']==0
    assert result['macro_recall_present_classes']==.5
    assert result['class_recall'][7] is None


def test_stationary_focus9_excluded_from_moving_macro():
    target=torch.tensor([9,3,7]);pred=torch.tensor([9,3,0])
    result=class_metrics(target,pred)
    assert result['moving_macro_recall']==.5
