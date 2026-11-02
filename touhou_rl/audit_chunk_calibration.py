"""Observed-motion audit; no counterfactual or exact-consumption claims."""
from __future__ import annotations
import argparse, hashlib, json, math
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .chunk_safety import ChunkSafetyConfig, predict
from .input_evidence import CONTROL_MASK
from . import spec
SCHEMA='v200-chunk-calibration-audit/2'
_DIRECTIONS=np.array([[0,0],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1]],float)

def _pos(row):
    try:
        p=np.asarray(row.get('player_position'),float)
        return p if p.shape==(2,) and np.isfinite(p).all() else None
    except (TypeError,ValueError): return None

def _action(row):
    a=row.get('executed_action')
    return a if isinstance(a,list) and len(a)==4 and all(type(v) is int for v in a) and 0<=a[0]<9 and all(v in (0,1) for v in a[1:]) else None

def _death(row):
    return type(row.get('deaths_before')) is int and type(row.get('deaths_after')) is int and row['deaths_after']>row['deaths_before']

def _observations(row):
    t=row.get('input_timing')
    if not isinstance(t,dict): return [],'input-timing-missing'
    effective=t.get('effective_first_observed'); end=t.get('end'); req=t.get('requested_buttons')
    if not isinstance(effective,dict) or not isinstance(end,dict) or type(req) is not int:return [],'input-timing-unknown'
    observations=t.get('observations')
    if not isinstance(observations,list):return [],'input-observations-missing'
    obs=[o for o in observations if o.get('stage')==effective.get('stage') and type(o.get('frame')) is int and effective.get('frame',-1)<=o['frame']<=end.get('frame',-1)]
    if len(obs)<2 or obs[0]['frame']!=effective['frame'] or obs[-1]['frame']!=end['frame']:return [],'input-endpoints-missing'
    if any(o.get('read_stable') is not True or type(o.get('gui_input_current')) is not int or (o['gui_input_current']&CONTROL_MASK)!=(req&CONTROL_MASK) or _pos(o) is None for o in obs):return [],'input-readback-unstable-or-mismatch'
    if any(b['frame']!=a['frame']+1 for a,b in zip(obs,obs[1:])):return [],'input-frame-gap'
    if len({o.get('deaths') for o in obs})!=1:return [],'death-during-hold'
    phases=[o.get('player_state') for o in obs]
    if any(p not in (0,3) for p in phases):return [],'uncontrollable-or-unknown-player-phase'
    return obs,None

def audit_rows(rows,config=ChunkSafetyConfig()):
    config.validate(); seen=set(); previous=None; actions=[]
    for row in rows:
        key=(row.get('episode'),row.get('decision'))
        if key in seen: raise ValueError('duplicate episode/decision')
        if previous is not None and (key[0]<previous[0] or key[0]==previous[0] and key[1]<=previous[1]):raise ValueError('trace reordered')
        seen.add(key); previous=key
        reasons=[]; a=_action(row); obs,reason=_observations(row)
        if reason:reasons.append(reason)
        if a is None:reasons.append('invalid-action')
        if _death(row):reasons.append('death')
        if a is not None and a[3]:reasons.append('bomb-during-hold')
        residual=None; predicted=None; observed=None; actual_frames=None
        if not reasons:
            actual_frames=obs[-1]['frame']-obs[0]['frame']; p=_pos(obs[0]).copy()
            diagonal=a[0] in (2,4,6,8)
            speed=(config.focus_speed_diag if a[2] else config.speed_diag) if diagonal else (config.focus_speed if a[2] else config.speed)
            delta=_DIRECTIONS[a[0]]*speed
            lo=np.array([spec.MOVE_AREA_X+config.player_half,spec.MOVE_AREA_Y+config.player_half])
            hi=np.array([spec.MOVE_AREA_X+spec.MOVE_AREA_W-config.player_half,spec.MOVE_AREA_Y+spec.MOVE_AREA_H-config.player_half])
            for _ in range(actual_frames):p=np.clip(p+delta,lo,hi)
            predicted=p.tolist();observed=_pos(obs[-1]).tolist();residual=float(np.linalg.norm(p-_pos(obs[-1])))
        clearance=None; clearance_reason='snapshot-to-effective-not-aligned'
        snap=row.get('raw_safety_snapshot')
        if isinstance(snap,dict) and a is not None and obs and snap.get('stage')==obs[0]['stage'] and snap.get('frame')==obs[0]['frame']:
            durations=[config.frameskip*d for d in (1,3,5)]
            if actual_frames in durations:
                value=float(predict(snap,config)['clearance'][a[2]*9+a[0],durations.index(actual_frames)])
                if math.isfinite(value):clearance=value;clearance_reason=None
                else:clearance_reason='model-invalid'
            else:clearance_reason='duration-not-modeled'
        actions.append(dict(episode=key[0],decision=key[1],action=a,usable=not reasons,
                            unavailable_reasons=reasons,displacement_residual=residual,
                            predicted_next_player_position=predicted,observed_next_player_position=observed,
                            actual_observed_hold_frames=actual_frames,model_clearance=clearance,
                            clearance_unavailable_reason=clearance_reason,death_observed_next=_death(row),
                            input_timing_status='input_observed' if obs else 'unknown'))
    canonical=json.dumps([{k:v for k,v in r.items() if k!='_line'} for r in rows],sort_keys=True,separators=(',',':'),allow_nan=False)
    residuals=[a['displacement_residual'] for a in actions if a['displacement_residual'] is not None]
    return dict(schema=SCHEMA,identity_hash=dict(algorithm='sha256',value=hashlib.sha256(canonical.encode()).hexdigest()),
                config=asdict(config),audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                observational_only=True,counterfactual_calibration=False,action_count=len(actions),actions=actions,
                segments=[],residual_sample_count=len(residuals),residual_p50=float(np.percentile(residuals,50)) if residuals else None,
                residual_p95=float(np.percentile(residuals,95)) if residuals else None,
                unmeasured=['counterfactual-safety','full-world-risk','exact-engine-input-consumption'])

def audit_trace(path,config=ChunkSafetyConfig()):
    p=Path(path);rows=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
    report=audit_rows(rows,config);report['input']=dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest());return report

def aggregate_reports(reports):
    actions=[a for r in reports for a in r['actions']]
    residuals=[a['displacement_residual'] for a in actions if a['displacement_residual'] is not None]
    return dict(schema=SCHEMA,report_count=len(reports),action_count=len(actions),usable_count=sum(a['usable'] for a in actions),
                residual_sample_count=len(residuals),residual_p50=float(np.percentile(residuals,50)) if residuals else None,
                residual_p95=float(np.percentile(residuals,95)) if residuals else None,reports=reports,
                observational_only=True,counterfactual_calibration=False)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('-o','--out');a=p.parse_args(argv)
    text=json.dumps(audit_trace(a.trace),ensure_ascii=False,indent=2,allow_nan=False)
    if a.out:Path(a.out).write_text(text,encoding='utf-8')
    else:print(text)
if __name__=='__main__':main()
