"""Recompute executed-clearance finite-sample coverage and uncertainty."""
import json
from pathlib import Path
import numpy as np

def summarize(path):
    r=json.loads(Path(path).read_text());rows=r['records'];out={}
    for label,selected in [('all',rows),('danger_under15',[x for x in rows if x['measured_clearance']<15])]:
        n=len(selected);false=sum(x['false_safe'] for x in selected)
        # Exact one-sided zero-event bound; nonzero count is not summarized as 0.
        out[label]=dict(samples=n,false_safe=false,zero_event_95_upper=1-.05**(1/n) if n and false==0 else None,
          error_p50=float(np.percentile([x['error_px'] for x in selected],50)) if n else None,
          error_p95=float(np.percentile([x['error_px'] for x in selected],95)) if n else None,
          linear_residual_p95_p95=float(np.percentile([x['linear_residual_p95'] for x in selected if x['linear_residual_p95'] is not None],95)) if n else None)
    return dict(schema='v200-clearance-calibration-coverage/1',summary=out,
                limits='zero observed false-safe does not prove zero risk; frames/segments are correlated, interval is descriptive independence approximation')
