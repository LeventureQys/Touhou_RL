"""Offline lag identification from independently localized pixels, never a detector.

Positive lag means image trails memory. Ambiguous/periodic/static traces stay
unknown; this engineering diagnostic cannot certify actual screenshot freshness.
"""
from __future__ import annotations
import numpy as np


def estimate_lag(memory, pixels, *, max_lag=8, min_samples=30, margin=.25, max_error=1.):
    """memory frame->(x,y); pixels list of frame/xy/confident/phase_verified.

    Fit one constant anchor offset per hypothesis, so constant-velocity motion
    cannot spuriously identify lag from unknown sprite anchor. Missing frames are
    not interpolated. Every lag must have min_samples; otherwise unknown.
    """
    if max_lag<1 or min_samples<3 or margin<=0 or max_error<=0:
        raise ValueError('invalid diagnostic bounds')
    usable=[r for r in pixels if r.get('confident') and r.get('phase_verified')]
    frames=[r['frame'] for r in usable]
    if len(frames)!=len(set(frames)):
        return {'status':'unknown','reason':'duplicate_frame_phase_ambiguity'}
    scores=[]
    for lag in range(-max_lag,max_lag+1):
        pairs=[(np.asarray(r['xy'],float),np.asarray(memory[r['frame']-lag],float))
               for r in usable if r['frame']-lag in memory]
        if len(pairs)<min_samples:
            return {'status':'unknown','reason':'insufficient_common_lag_coverage'}
        q=np.stack([p[0] for p in pairs]); p=np.stack([p[1] for p in pairs])
        if not np.isfinite(q).all() or not np.isfinite(p).all():
            return {'status':'unknown','reason':'nonfinite_positions'}
        offset=np.median(q-p,axis=0)
        error=float(np.median(np.linalg.norm(q-p-offset,axis=1)))
        scores.append({'lag':lag,'error':error,'samples':len(pairs),'anchor_offset':offset.tolist()})
    scores.sort(key=lambda r:r['error'])
    best=scores[0]['error']
    candidates=[r['lag'] for r in scores if r['error']<=best+margin]
    if best>max_error:
        status,reason='unknown','pixel_memory_geometry_disagree'
    elif len(candidates)!=1:
        status,reason='unknown','lag_not_identifiable_static_constant_or_periodic_motion'
    elif abs(candidates[0])>1:
        status,reason='outside_one_frame','unique_lag_exceeds_budget'
    else:
        status,reason='diagnostic_within_one_frame','unique_lag_for_supplied_positions_only'
    return {'status':status,'reason':reason,'lag_candidates':candidates,'scores':scores,
            'real_pixel_gate_passed':False,
            'warning':'Requires independent real pixel localization and phase validation; not a capture certification.'}
