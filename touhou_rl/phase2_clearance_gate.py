"""Gate executed clearance and motion together from full-pool trace evidence."""
from pathlib import Path
import json
from .build_chunk_safety_cache import sha
from .audit_chunk_calibration import audit_rows
from .executed_clearance import audit

def run(root):
    root=Path(root);p=json.loads((root/'preregistration.json').read_text());probe=root/'executed-clearance-probe'
    rows=[json.loads(l) for l in (probe/'executed-trace.jsonl').read_text().splitlines()]
    saved=json.loads((probe/'clearance-calibration.json').read_text())
    if saved['trace_sha256']!=sha(probe/'executed-trace.jsonl') or saved['protocol_sha256']!=sha(root/'preregistration.json'):raise ValueError('probe identity mismatch')
    clearance=audit(rows);motion=audit_rows(rows)
    pools=[o for r in rows for o in r['input_timing']['observations'] if o.get('read_stable') and o.get('bullets')]
    from .raw_pool_contract import validate_pool
    for observation in pools: validate_pool(observation['bullets'])
    limits=p['calibration_gate'];cl=p['clearance_gate']
    gates=dict(motion_count=motion['residual_sample_count']>=limits['min_input_observed_transitions'],
       motion_p95=motion['residual_p95'] is not None and motion['residual_p95']<=limits['position_residual_p95_max_px'],
       raw_snapshot_count=len(pools)>=limits['min_aligned_complete_snapshots'],
       nonempty_bullet_snapshot_count=sum(bool(o['bullets'].get('position')) for o in pools)>=limits['min_nonempty_bullet_snapshots'],
       per_duration_count=all(clearance['per_duration'][str(d)]>=cl['min_each_duration'] for d in (1,3,5)),
       clearance_p95=clearance['clearance_error_p95'] is not None and clearance['clearance_error_p95']<=cl['error_p95_max_px'],
       false_safe_count=clearance['false_safe_count']<=cl['false_safe_max'])
    result=dict(schema='v200-phase2-calibration-gate/2',protocol_sha256=sha(root/'preregistration.json'),
      trace_sha256=sha(probe/'executed-trace.jsonl'),clearance_calibration_sha256=sha(probe/'clearance-calibration.json'),
      gates=gates,passed=all(gates.values()),training_approved=all(gates.values()),position_residual_p95_px=motion['residual_p95'],
      observed_transitions=motion['residual_sample_count'],clearance_count=clearance['count'],
      clearance_error_p95_px=clearance['clearance_error_p95'],false_safe_count=clearance['false_safe_count'],
      validated_against_measured_clearance=True,scope='executed-action linear existing-pool geometry only; full-world and counterfactual still unverified')
    target=root/'calibration-gate.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    import argparse;p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)
