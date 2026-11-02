"""Losses for finite-model Phase3 temporal preference and duration targets."""
import torch
import torch.nn.functional as F


def rank_per_frame(scores,values,delta=.5):
    ij=torch.triu_indices(18,18,offset=1,device=scores.device)
    gap=values[...,ij[0]]-values[...,ij[1]];valid=gap.abs()>delta
    loss=F.softplus(-gap.sign()*(scores[...,ij[0]]-scores[...,ij[1]]))
    return (loss*valid).sum(-1)/valid.sum(-1).clamp(min=1)


def duration_loss(logits,targets,valid,class_weights=None):
    """Each current-state candidate gets its own duration supervision and mask."""
    safe_targets=targets.clamp(min=0)
    raw=F.cross_entropy(logits.reshape(-1,3),safe_targets.reshape(-1),weight=class_weights,reduction='none').reshape(targets.shape)
    return (raw*valid).sum()/valid.sum().clamp(min=1)


def weighted_mean(values,mask,weight=None):
    w=mask.to(values.dtype)
    if weight is not None:w=w*weight
    return (values*w).sum()/w.sum().clamp(min=1)
