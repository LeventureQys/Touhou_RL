"""Event-grouped offline diagnostics; never substitutes for closed-loop behavior."""
from __future__ import annotations
import torch


@torch.no_grad()
def event_metrics(output, labels):
    """Compute independent valid-label counts; unknown is not an error or success.

    Each input represents ONE complete event (or explicitly reported diagnostic
    window). No future trajectory, survival or safety conclusion is fabricated.
    """
    logits=output['policy_logits']
    shape=logits.shape[:-1]
    valid=output.get('valid_mask',torch.ones(shape,dtype=torch.bool,device=logits.device))
    result={'rows':int(valid.sum()),'capability':'offline_diagnostic_only'}
    def mask(name):
        m=labels.get(name+'_valid',torch.zeros_like(valid))
        if m.dtype!=torch.bool or m.shape!=valid.shape:
            raise ValueError(f'{name}_valid shape/dtype')
        return m&valid
    action=mask('action')
    result['action_count']=int(action.sum())
    result['action_unknown_count']=int((valid&~action).sum())
    result['positive_set_agreement']=None
    if action.any():
        positive=labels['action_positive']
        if positive.dtype!=torch.bool or positive.shape!=logits.shape or not positive[action].any(-1).all():
            raise ValueError('reliable action must have positive set')
        chosen=logits.argmax(-1)
        result['positive_set_agreement']=float(positive.gather(-1,chosen[...,None]).squeeze(-1)[action].float().mean())
    option=mask('option');result['option_count']=int(option.sum())
    result['option_set_agreement']=None
    if option.any():
        pred=output['option_logits'].argmax(-1)
        if 'option_positive' in labels:
            positive=labels['option_positive']
        else:
            target=labels['option_target']
            positive=target>0
        if positive.shape!=output['option_logits'].shape or not positive[option].any(-1).all():
            raise ValueError('reliable option positive set required')
        result['option_set_agreement']=float(positive.gather(-1,pred[...,None]).squeeze(-1)[option].float().mean())
    target=mask('target');result['target_count']=int(target.sum())
    result['target_mse']=None;result['target_angular_degrees']=None;result['target_direction_count']=0
    if target.any():
        p=output['target'][target];q=labels['target'][target]
        if not torch.isfinite(p).all() or not torch.isfinite(q).all() or p.shape!=q.shape:
            raise ValueError('finite reliable target required')
        result['target_mse']=float((p-q).square().mean())
        length=p.norm(dim=-1)*q.norm(dim=-1)
        nonzero=length>1e-8
        result['target_direction_count']=int(nonzero.sum())
        if nonzero.any():
            cos=(p*q).sum(-1)[nonzero]/length[nonzero]
            result['target_angular_degrees']=float(torch.rad2deg(cos.clamp(-1,1).acos()).mean())
    term=mask('termination');result['termination_count']=int(term.sum())
    result['termination_confusion']=None
    if term.any():
        p=output['termination_logits'][term].argmax(-1);q=labels['termination'][term]
        if q.dtype!=torch.long or ((q<0)|(q>2)).any():
            raise ValueError('termination labels must be 0..2')
        result['termination_confusion']=torch.bincount(q*3+p,minlength=9).reshape(3,3).cpu().tolist()
    return result


def summarize_events(events):
    """Equal event mean AND valid-row weighted mean, with explicit denominators."""
    metrics=('positive_set_agreement','option_set_agreement','target_mse','target_angular_degrees')
    counts=('action_count','option_count','target_count','target_direction_count')
    out={'events':len(events),'warning':'No departure, reversal, pollution or survival inferred from offline labels.'}
    for metric,count in zip(metrics,counts):
        observed=[e for e in events if e.get(metric) is not None and e.get(count,0)>0]
        n=sum(e[count] for e in observed)
        out[metric]={'events':len(observed),'valid_rows':n,
            'event_mean':sum(e[metric] for e in observed)/len(observed) if observed else None,
            'row_weighted_mean':sum(e[metric]*e[count] for e in observed)/n if n else None}
    return out
