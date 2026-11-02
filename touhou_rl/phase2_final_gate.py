"""Independent preregistered Phase2 research verdict from a completed report."""
import json
from pathlib import Path
from .build_chunk_safety_cache import sha

def evaluate(report_path,protocol_path,out_path):
    report_path,protocol_path,out_path=map(Path,(report_path,protocol_path,out_path))
    if out_path.exists():raise FileExistsError(out_path)
    r=json.loads(report_path.read_text());p=json.loads(protocol_path.read_text());t=p['final_gate'];exp=p['experiment']
    if r.get('schema')!='v200-phase2-chunk-experiment/1':raise ValueError('report schema mismatch')
    if r['protocol_sha256']!=sha(protocol_path):raise ValueError('protocol identity mismatch')
    for key,expected in [('baseline_checkpoint_sha256',p['checkpoint_sha256']),('A_manifest_sha256',p['A_manifest_sha256']),('A_val_index_sha256',p['A_val_index_sha256'])]:
        if r.get(key)!=expected:raise ValueError(f'{key} mismatch')
    cache=protocol_path.parent/'student-chunk-cache'
    if r.get('cache_identity_sha256')!=sha(cache/'chunk-identity.json'):raise ValueError('cache identity mismatch')
    if r.get('candidate_mapping')!='focus*9+move' or r.get('input_depth')!=3:raise ValueError('candidate/depth mismatch')
    if r['train']['ratio']!=3 or r['train']['epochs']!=exp.get('epochs',1) or r['train']['epochs'] not in (1,2):raise ValueError('preregistered training budget mismatch')
    if r['train'].get('anchor_weight',0.)!=exp.get('anchor_weight',0.):raise ValueError('anchor weight protocol mismatch')
    if r['train'].get('danger_weight',1.)!=exp.get('danger_weight',1.):raise ValueError('danger weight protocol mismatch')
    for name in ('learning_rate','seed','batch_size'):
        if name in r['train'] and r['train'][name]!=exp[name]:raise ValueError(f'{name} runtime protocol mismatch')
    for key,split in [('student_train','train'),('student_dev','dev'),('student_test','test')]:
        if r['splits'][key]['episodes']!=p['episodes'][split]:raise ValueError('split identity mismatch')
    base=r['before'];mixed=r['after']['mixed'];control=r['after']['A_only_control']
    b=base['student']['student_test'];m=mixed['student']['student_test'];c=control['student']['student_test']
    def danger(metric):return next(x for x in metric['by_danger'] if x['bucket']=='0-15')
    d_b,d_m=danger(b),danger(m)
    audit_path=protocol_path.parent/'independent-execution-audit.json'
    independent=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    calibration_path=protocol_path.parent/'calibration-gate.json'
    calibration=json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    clearance_proof=(r.get('executed_clearance_calibration',{}).get('validated_against_measured_clearance') is True
                     and calibration.get('validated_against_measured_clearance') is True
                     and calibration.get('passed') is True
                     and calibration.get('protocol_sha256')==sha(protocol_path)
                     and r.get('calibration_gate_sha256')==sha(calibration_path)) if calibration_path.exists() else False
    gates={
      'independent_execution_identity':independent.get('report_sha256')==sha(report_path) and independent.get('baseline_unchanged') is True and independent.get('A_manifest_unchanged') is True,
      'independent_frozen_model_match':independent.get('actual_frozen_against_baseline') is True,
      'independent_split_identity':independent.get('split_identity_verified') is True,
      'sealed_source_code_match':independent.get('strict_preregistration_code_match') is True,
      'history_episode_split_isolation':r.get('history_isolation',{}).get('episode_crossings')==0 and r.get('history_isolation',{}).get('split_crossings')==0,
      'split_min_samples':all(r['splits'][key]['frames']>=exp[minimum] for key,minimum in [('student_train','min_train_samples'),('student_dev','min_dev_samples'),('student_test','min_test_samples')]),
      # The engineering gate is motion-only. Full Phase2 teacher validation must
      # also provide observed clearance calibration, not self-declared safety.
      'executed_clearance_calibration':clearance_proof,
      'student_test_overall_improvement':m['regret_px']<=b['regret_px']*(1-t['student_regret_relative_improvement_min']),
      'better_than_A_only':m['regret_px']<c['regret_px'],
      'A_val_overall_nonregression':mixed['A']['regret_px']<=base['A']['regret_px']+t['A_overall_regret_increase_max_px'],
      'A_val_danger_nonregression':danger(mixed['A'])['regret_px']<=danger(base['A'])['regret_px']+t['A_danger_regret_increase_max_px'],
      'student_danger_count':d_m['frames']>=exp['min_danger_test_samples'],
      'student_danger_nonregression':d_m['regret_px']<=d_b['regret_px']+t['student_danger_regret_increase_max_px'],
      'not_collapsed':m['max_single_candidate_fraction']<=t['max_single_candidate_fraction'],
      'frozen_mixed':r['frozen_state_sha256']['mixed']['before']==r['frozen_state_sha256']['mixed']['after'],
      'frozen_control':r['frozen_state_sha256']['control']['before']==r['frozen_state_sha256']['control']['after'],
      'matched_steps':r['train']['optimizer_steps']==r['train']['control_optimizer_steps'],
      'B_not_used':r['B_used'] is False,
      'student_danger_uncertainty':r['bootstrap']['student_test']['danger_0_15']['mixed_minus_baseline']['episodes']>=2 and r['bootstrap']['student_test']['danger_0_15']['mixed_minus_baseline']['hi95']<=t['student_danger_regret_increase_max_px'],
    }
    result=dict(schema='v200-phase2-final-research-gate/1',report_sha256=sha(report_path),protocol_sha256=sha(protocol_path),
                passed=all(gates.values()),gates=gates,failed=[k for k,v in gates.items() if not v],
                decision='Phase2-pass-review-required-before-Phase3' if all(gates.values()) else 'Phase2-NO-GO-remain-diagnostic',
                phase3_executed=False,summary=dict(baseline_regret=b['regret_px'],mixed_regret=m['regret_px'],control_regret=c['regret_px'],
                  test_frames=m['frames'],danger_frames=d_m['frames'],baseline_danger_regret=d_b['regret_px'],mixed_danger_regret=d_m['regret_px']),
                limitations=['linear-existing-bullet-model targets, not verified counterfactual world safety',f"{len(p['episodes']['test'])} test episodes give limited bootstrap precision",'no survival benchmark'])
    out_path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);parser.add_argument('--protocol',required=True);parser.add_argument('--out',required=True)
    a=parser.parse_args();evaluate(a.report,a.protocol,a.out)
