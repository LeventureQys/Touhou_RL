"""Summarize Phase2 finite-world mask coverage, not probabilities of safety."""
from collections import Counter
from pathlib import Path
import json
import numpy as np

def summarize(source,cache):
    source,cache=Path(source),Path(cache)
    report=json.loads((source/'student-rollout-report.json').read_text());rows=[]
    for ep in report['episodes']:rows.extend(json.loads(x) for x in (source/ep['trace']).read_text().splitlines())
    with np.load(cache/'chunk-arrays.npz') as z:
        valid=z['model_valid'].astype(bool);clear=z['clearance'];duration=z['duration_class']
        by_duration={str(d):dict(valid_cells=int(valid[:,:,j].sum()),predicted_collision_cells=int(((clear[:,:,j]<=0)&valid[:,:,j]).sum()),
                                positive_model_margin_cells=int(((clear[:,:,j]>0)&valid[:,:,j]).sum())) for j,d in enumerate((1,3,5))}
        hist={str(i):int((duration==i).sum()) for i in (-1,0,1,2)}
        return dict(schema='v200-phase2-mask-coverage/1',decisions=len(rows),samples=report['samples'],
          player_phases=dict(Counter(str(r['raw_safety_snapshot'].get('player_state')) for r in rows)),
          duration=by_duration,longest_model_safe_duration_class_histogram=hist,
          observed_hold_valid=sum(r.get('action_hold_frames_valid') is True for r in rows),
          world_risk_valid_cells=int(z['world_risk_valid'].sum()),
          semantics='collision labels and duration classes are linear geometric predictions, not observed candidate outcomes')
