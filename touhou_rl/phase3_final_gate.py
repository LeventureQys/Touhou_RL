"""Phase3 finite-model research gate. Never advances to Phase4 automatically."""
from pathlib import Path
import json
from .build_chunk_safety_cache import sha

def evaluate(root):
    root=Path(root);p=json.loads((root/'preregistration.json').read_text());r=json.loads((root/'experiment/phase3-report.json').read_text());g=p['gates']
    if r.get('schema')!='v200-phase3-experiment/1':raise ValueError('Phase3 report schema mismatch')
    if r['protocol_sha256']!=sha(root/'preregistration.json'):raise ValueError('protocol mismatch')
    if r.get('baseline_sha256')!=p['baseline_sha256']:raise ValueError('baseline identity mismatch')
    if r.get('cache_identity_sha256')!=sha(root/'student-chunk-cache/chunk-identity.json'):raise ValueError('cache identity mismatch')
    seal=json.loads((root/'training-code-identity.json').read_text())
    if seal['protocol_sha256']!=sha(root/'preregistration.json'):raise ValueError('training seal mismatch')
    for name,value in seal['sources'].items():
        if sha(Path(__file__).with_name(name))!=value:raise ValueError('sealed Phase3 code mismatch '+name)
    for key in ('train','dev','test'):
        if r['splits'][key]['episodes']!=p['episodes'][key]:raise ValueError('split episode identity mismatch')
    if recompute_path := (root/'independent-recompute.json'):
        if recompute_path.exists():
            proof=json.loads(recompute_path.read_text())
            if proof.get('report_sha256')!=sha(root/'experiment/phase3-report.json'):raise ValueError('stale independent proof')
    base=r['metrics']['baseline'];tem=r['metrics']['temporal'];control=r['metrics']['reset_control'];a=tem['test'];b=base['test'];c=control['test']
    def danger(x):return next(v for v in x['by_danger'] if v['bucket']=='0-15')
    fractions=a['chunk']['duration_fraction'];classes=sum(v>=g['min_duration_class_fraction'] for v in fractions)
    latency=json.loads((root/'latency-and-reset.json').read_text()) if (root/'latency-and-reset.json').exists() else {}
    recompute=json.loads((root/'independent-recompute.json').read_text()) if (root/'independent-recompute.json').exists() else {}
    gates=dict(
      runtime_proof_bound=latency.get('report_sha256')==sha(root/'experiment/phase3-report.json') and latency.get('checkpoint_sha256')==sha(root/'experiment/temporal-policy.pt'),
      matched_duration_clearance_nonregression=a['chunk']['duration_matched_clearance_gain']>=-g['student_danger_regret_increase_max_px'],
      finite_test_samples=a['frames']>=g['min_test_frames'],danger_samples=danger(a)['frames']>=g['min_danger_frames'],
      matched_horizon_improvement=a['regret_px']<=b['regret_px']*(1-g['matched_duration3_regret_improvement_min']),
      better_than_reset_control=a['regret_px']<c['regret_px'],
      A_overall_nonregression=tem['A']['regret_px']<=base['A']['regret_px']+g['A_overall_regret_increase_max_px'],
      A_danger_nonregression=danger(tem['A'])['regret_px']<=danger(base['A'])['regret_px']+g['A_danger_regret_increase_max_px'],
      student_danger_nonregression=danger(a)['regret_px']<=danger(b)['regret_px']+g['student_danger_regret_increase_max_px'],
      danger_uncertainty=r['bootstrap']['test']['danger']['hi95']<=g['danger_episode_bootstrap_upper_max'],
      duration_not_collapsed=classes>=g['min_duration_classes_used'] and max(fractions)<=g['max_duration_class_fraction'],
      same_duration_proxy_not_worse=a['chunk']['chosen_collision_rate']<=a['chunk']['duration_matched_baseline_collision_rate']+g['chosen_horizon_collision_increase_max'],
      noncollapse=a['max_candidate_fraction']<=g['max_single_candidate_fraction'],
      encoder_frozen=all(t['frozen_before']==t['frozen_after'] for t in r['training'].values()),
      matched_optimizer_budget=r['training']['temporal']['steps']==r['training']['reset_control']['steps'],
      reset_isolation=latency.get('reset_isolation_passed') is True,
      inference_budget=latency.get('pixel_end_to_end_p95_ms',1e9)<=g['inference_p95_max_ms'],
      independent_model_recompute=recompute.get('passed') is True,
      B_not_used=r['B_used'] is False)
    result=dict(schema='v200-phase3-research-gate/1',passed=all(gates.values()),gates=gates,failed=[k for k,v in gates.items() if not v],
      protocol_sha256=sha(root/'preregistration.json'),report_sha256=sha(root/'experiment/phase3-report.json'),
      summary=dict(baseline_matched3_regret=b['regret_px'],temporal_matched3_regret=a['regret_px'],control_matched3_regret=c['regret_px'],
        duration_fraction=fractions,test_frames=a['frames'],danger_frames=danger(a)['frames']),
      phase4_executed=False,scope='finite linear current-pool safety proxy; not world/survival guarantee')
    target=root/'final-research-gate.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    import argparse;pa=argparse.ArgumentParser();pa.add_argument('root');a=pa.parse_args();evaluate(a.root)
