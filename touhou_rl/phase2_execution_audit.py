"""Independent executed identity verification; reports protocol deviations."""
from pathlib import Path
import json
from .build_chunk_safety_cache import sha,audit_source,validate

def audit(root):
    root=Path(root);p=json.loads((root/'preregistration.json').read_text());r=json.loads((root/'head-only-controlled/phase2-report.json').read_text())
    cache_info=json.loads((root/'student-chunk-cache/chunk-identity.json').read_text())
    source=Path(cache_info['source'])
    sr,rows,aligned,files=audit_source(source);cache=validate(root/'student-chunk-cache',verify_source=True)
    deviations={name:dict(sealed=value,current=sha(Path('touhou_rl')/name)) for name,value in p.get('source_code_sha256',{}).items() if sha(Path('touhou_rl')/name)!=value}
    checkpoints={}
    from .vmodels import load_vision_checkpoint
    from .phase2_chunk_experiment import _digest_state
    for name in ('policy.pt','A-only-control.pt'):
        path=root/'head-only-controlled'/name;m,_=load_vision_checkpoint(path,map_location='cpu')
        checkpoints[name]=dict(sha256=sha(path),nonpolicy_sha256=_digest_state(m,include_policy=False))
    baseline_path=Path('third_party/th08/build/rl/runs/v105-stage2-formal-w0-s7/policy.pt')
    baseline_model,_=load_vision_checkpoint(baseline_path,map_location='cpu')
    baseline_nonpolicy=_digest_state(baseline_model,include_policy=False)
    from .pipeline.manifest import index_digest
    with __import__('numpy').load(root/'student-chunk-cache/chunk-arrays.npz') as z:
        ep=z['episode'];valid=z['model_valid'][:,:,1].all(axis=1)
        by_index={x['sample_index']:x for x in aligned}
        split_audit={}
        for key,name in [('student_train','train'),('student_dev','dev'),('student_test','test')]:
            ids=[i for i in range(len(ep)) if ep[i] in p['episodes'][name] and valid[i] and not by_index[i].get('terminated') and not by_index[i].get('truncated') and by_index[i].get('stage_delta',0)==0]
            split_audit[key]={'frames':len(ids),'index_sha256':index_digest(ids),'matches_report':len(ids)==r['splits'][key]['frames'] and index_digest(ids)==r['splits'][key]['index_sha256']}
    result=dict(schema='v200-phase2-independent-execution-audit/1',split_audit=split_audit,split_identity_verified=all(v['matches_report'] for v in split_audit.values()),protocol_sha256=sha(root/'preregistration.json'),
      source_report_sha256=sha(source/'student-rollout-report.json'),source_files=files,
      source_rows=len(rows),aligned_rows=len(aligned),cache_identity_sha256=sha(root/'student-chunk-cache/chunk-identity.json'),
      sealed_code_deviations=deviations,strict_preregistration_code_match=not deviations,
      checkpoints=checkpoints,baseline_nonpolicy_sha256=baseline_nonpolicy,
      actual_frozen_against_baseline=all(v['nonpolicy_sha256']==baseline_nonpolicy for v in checkpoints.values()),
      nonpolicy_model_equality=checkpoints['policy.pt']['nonpolicy_sha256']==checkpoints['A-only-control.pt']['nonpolicy_sha256'],
      baseline_unchanged=sha(Path('third_party/th08/build/rl/runs/v105-stage2-formal-w0-s7/policy.pt'))==p['checkpoint_sha256'],
      A_manifest_unchanged=sha(Path('third_party/th08/build/rl/datasets/pilot-ud0059-v2/manifest.json'))==p['A_manifest_sha256'],
      report_sha256=sha(root/'head-only-controlled/phase2-report.json'),
      scientific_scope='diagnostic-model-only; any code deviation prevents strict preregistered acceptance')
    (root/'independent-execution-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
if __name__=='__main__':
    import argparse;parser=argparse.ArgumentParser();parser.add_argument('root');a=parser.parse_args();print(json.dumps(audit(a.root),indent=2))
