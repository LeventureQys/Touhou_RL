"""Observed executed-clearance calibration. No counterfactual validation claim."""
import numpy as np
from .audit_chunk_calibration import _observations
from .chunk_safety import predict,ChunkSafetyConfig


def audit(rows,config=ChunkSafetyConfig()):
    records=[]; exclusions={}
    for row in rows:
        obs,reason=_observations(row)
        if reason:exclusions[reason]=exclusions.get(reason,0)+1;continue
        if row.get('executed_action',[0,0,0,0])[3]:continue
        duration=obs[-1]['frame']-obs[0]['frame']
        if duration not in (config.frameskip,3*config.frameskip,5*config.frameskip):
            exclusions['duration-not-exact']=exclusions.get('duration-not-exact',0)+1;continue
        if any(not isinstance(o.get('bullets'),dict) for o in obs):
            exclusions['pool-missing']=exclusions.get('pool-missing',0)+1;continue
        snapshot=dict(schema='v200-raw-safety-snapshot/1',complete=True,stage=obs[0]['stage'],frame=obs[0]['frame'],
                      player_position=obs[0]['player_position'],player_state=obs[0]['player_state'],bullets=obs[0]['bullets'])
        action=row['executed_action'];candidate=action[2]*9+action[0]
        forecast=float(predict(snapshot,config)['clearance'][candidate,[1,3,5].index(duration//config.frameskip)])
        measured=config.clearance_cap; linear_residual=[]; first_pool=obs[0]['bullets']
        ids={s:i for i,s in enumerate(first_pool['slot_id'])};spawned=0;current_initial=set(ids)
        for o in obs:
            b=o['bullets'];pos=np.asarray(b['position'],float).reshape(-1,2);half=np.asarray(b['half'],float)
            if len(pos):measured=min(measured,float((np.linalg.norm(pos-np.asarray(o['player_position']),axis=1)-half-config.player_half-config.bullet_pad).min()))
            dt=o['frame']-obs[0]['frame']
            for j,slot in enumerate(b['slot_id']):
                if slot in ids:
                    i=ids[slot];pred=np.asarray(first_pool['position'][i])+np.asarray(first_pool['velocity'][i])*dt
                    linear_residual.append(float(np.linalg.norm(pos[j]-pred)))
                else:spawned+=1
        records.append(dict(episode=row['episode'],decision=row['decision'],duration_decisions=duration//config.frameskip,
                            predicted_clearance=forecast,measured_clearance=measured,error_px=abs(forecast-measured),
                            false_safe=forecast>0 and measured<=0,new_slot_observations=spawned,
                            linear_residual_p95=float(np.percentile(linear_residual,95)) if linear_residual else None))
    errors=[r['error_px'] for r in records]
    return dict(schema='v200-executed-clearance-calibration/1',records=records,exclusions=exclusions,
                count=len(records),per_duration={str(d):sum(r['duration_decisions']==d for r in records) for d in (1,3,5)},
                clearance_error_p95=float(np.percentile(errors,95)) if errors else None,
                false_safe_count=sum(r['false_safe'] for r in records),
                semantics='executed action observed per-frame circles; initial snapshot linear prediction; slot reuse unknown; no counterfactual proof')
