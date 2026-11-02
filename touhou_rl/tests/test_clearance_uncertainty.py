from pathlib import Path
import json
from touhou_rl.clearance_uncertainty import summarize

def test_zero_events_is_not_zero_risk(tmp_path):
    p=tmp_path/'c.json';p.write_text(json.dumps(dict(records=[dict(false_safe=False,measured_clearance=10,error_px=.1,linear_residual_p95=.01) for _ in range(40)])))
    r=summarize(p)['summary']['danger_under15']
    assert r['samples']==40 and .07<r['zero_event_95_upper']<.08
