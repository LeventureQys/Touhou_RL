"""Reproduce the sealed engineering calibration gate; never starts training."""
from pathlib import Path
import json
from .audit_chunk_calibration import audit_trace, aggregate_reports
from .build_chunk_safety_cache import audit_source, sha

def run(root=Path('Document/Update/Alpha/v2.0.0'), round_name='r2', attempts_names=None):
    out=root/f'phase2/full-phase2-{round_name}'; protocol=out/'preregistration.json'
    p=json.loads(protocol.read_text(encoding='utf-8')); threshold=p['calibration_gate']; attempts=[]
    for name in (attempts_names or (f'{round_name}-engineering-teacher-a',f'{round_name}-engineering-teacher-b')):
        source=root/'rollouts'/name
        report,rows,aligned,files=audit_source(source)
        if report.get('protocol_sha256')!=sha(protocol):raise ValueError('protocol identity mismatch')
        calibration=aggregate_reports([audit_trace(source/e['trace']) for e in report['episodes']])
        counts=dict(decisions=len(rows),aligned_complete=sum(r['raw_safety_snapshot'].get('complete') is True for r in aligned),
                    nonempty_bullets=sum(bool(r['raw_safety_snapshot'].get('bullets',{}).get('position')) for r in aligned),
                    observed_transitions=calibration['residual_sample_count'])
        gates=dict(input_sample_count=counts['observed_transitions']>=threshold['min_input_observed_transitions'],
                   position_residual_p95=calibration['residual_p95'] is not None and calibration['residual_p95']<=threshold['position_residual_p95_max_px'],
                   aligned_snapshot_count=counts['aligned_complete']>=threshold['min_aligned_complete_snapshots'],
                   nonempty_bullet_count=counts['nonempty_bullets']>=threshold['min_nonempty_bullet_snapshots'])
        attempts.append(dict(name=name,source_report_sha256=sha(source/'student-rollout-report.json'),
                             trace_and_shard_sha256=files,counts=counts,residual_p50=calibration['residual_p50'],
                             residual_p95=calibration['residual_p95'],gates=gates,passed=all(gates.values()),
                             collection_status=report['phase1_collection_status']))
        path=out/(name+'-calibration.json')
        # Audit-derived report is recomputable, not source data. Existing input read first.
        if path.exists(): json.loads(path.read_text())
        path.write_text(json.dumps(calibration,indent=2,allow_nan=False),encoding='utf-8')
    gate=dict(schema='v200-phase2-calibration-gate/1',protocol_sha256=sha(protocol),thresholds=threshold,
              attempts=attempts,passed=bool(attempts[-1]['passed']),training_approved=bool(attempts[-1]['passed']),
              engineering_budget_exhausted=len(attempts)>=p['collection']['max_engineering_attempts'],
              unmeasured=['counterfactual-clearance-accuracy','full-world-risk','laser-enemy-invulnerability','true-engine-generation'],
              decision='GO-label-model-only' if attempts[-1]['passed'] else 'NO-GO-stop-before-student-collection-and-training')
    target=out/'calibration-gate.json'
    if target.exists(): raise FileExistsError(target)
    target.write_text(json.dumps(gate,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(gate,ensure_ascii=False,indent=2))
    return gate
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--round',default='r2');parser.add_argument('--attempts',nargs='*')
    args=parser.parse_args();run(round_name=args.round,attempts_names=args.attempts)
