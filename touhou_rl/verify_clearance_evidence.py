"""Read-only verifier for saved clearance gate/probe evidence."""
import json
from pathlib import Path
from .executed_clearance import audit
from .audit_chunk_calibration import audit_rows
from .build_chunk_safety_cache import sha

def verify(root):
    root=Path(root);gate=json.loads((root/'calibration-gate.json').read_text());p=json.loads((root/'preregistration.json').read_text())
    trace=root/'executed-clearance-probe/executed-trace.jsonl';rows=[json.loads(l) for l in trace.read_text().splitlines()]
    if gate['protocol_sha256']!=sha(root/'preregistration.json') or gate['trace_sha256']!=sha(trace):raise ValueError('gate provenance mismatch')
    calibration_file=root/'executed-clearance-probe/clearance-calibration.json'
    if gate.get('clearance_calibration_sha256')!=sha(calibration_file):raise ValueError('saved calibration hash mismatch')
    saved=json.loads(calibration_file.read_text())
    from .raw_pool_contract import validate_pool
    for row in rows:
        for observation in row.get('input_timing',{}).get('observations',[]):
            if observation.get('bullets') is not None: validate_pool(observation['bullets'])
    c=audit(rows);m=audit_rows(rows)
    if any(saved.get(k)!=c.get(k) for k in ('count','per_duration','clearance_error_p95','false_safe_count')):raise ValueError('saved calibration content mismatch')
    if c['clearance_error_p95']!=gate['clearance_error_p95_px'] or c['false_safe_count']!=gate['false_safe_count'] or m['residual_p95']!=gate['position_residual_p95_px']:raise ValueError('gate metrics not reproduced')
    return dict(provenance_verified=True,metrics_reproduced=True,gate_passed=gate['passed'],
                per_duration=c['per_duration'],clearance_p95=c['clearance_error_p95'],motion_p95=m['residual_p95'])
