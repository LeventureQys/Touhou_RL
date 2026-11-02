"""Read-only independent source, sequence, frozen model and split identities."""
from pathlib import Path
import json
import numpy as np
from .build_chunk_safety_cache import audit_source,validate,sha
from .phase3_sequences import build_sequence_index,previous_history
from .phase3_experiment import frozen_digest
from .temporal_policy import load_temporal_checkpoint,TemporalPolicy
from .vmodels import load_vision_checkpoint
from .pipeline.manifest import index_digest


def audit(root,source,cache):
    root,source,cache=map(Path,(root,source,cache));p=json.loads((root/'preregistration.json').read_text());r=json.loads((root/'experiment/phase3-report.json').read_text())
    sr,rows,aligned,files=audit_source(source);identity=validate(cache,verify_source=True)
    with np.load(cache/'chunk-arrays.npz') as z:ca={k:z[k].copy() for k in z.files}
    epmap={e:s for s,eps in p['episodes'].items() for e in eps};original=build_sequence_index(rows,epmap,p['temporal_data']['max_frame_gap']);valid=ca['model_valid'].all((1,2));segments=[]
    for seg in original:
        current=[]
        for i in seg:
            if valid[i]:current.append(i)
            elif current:segments.append(current);current=[]
        if current:segments.append(current)
    history=previous_history(rows,segments);lookup={x['sample_index']:x for x in aligned};bad=[]
    for seg in segments:
        for previous,current in zip(seg,seg[1:]):
            a,b=lookup[previous],lookup[current]
            if a['episode']!=b['episode'] or b['decision']!=a['decision']+1 or a['deaths_after']!=a['deaths_before'] or a['stage']!=b['stage']:bad.append((previous,current))
    split_proof={}
    for key in ('train','dev','test'):
        ids=sorted(i for seg in segments for i in seg if epmap[ca['episode'][i]]==key)
        split_proof[key]=dict(frames=len(ids),index_sha256=index_digest(ids),matches_report=len(ids)==r['splits'][key]['frames'] and index_digest(ids)==r['splits'][key]['index_sha256'])
    baseline,_=load_vision_checkpoint(Path(p['baseline_checkpoint']));base_digest=frozen_digest(TemporalPolicy(baseline));model_proof={}
    for name in ('temporal-policy.pt','reset-control.pt'):
        path=root/'experiment'/name;model=load_temporal_checkpoint(path,map_location='cpu');d=frozen_digest(model)
        model_proof[name]=dict(sha256=sha(path),frozen_vision_sha256=d,exact_baseline_frozen=d==base_digest)
    seal=json.loads((root/'training-code-identity.json').read_text());source_mismatch=[n for n,v in seal['sources'].items() if sha(Path(__file__).with_name(n))!=v]
    result=dict(schema='v200-phase3-independent-identity-audit/1',protocol_sha256=sha(root/'preregistration.json'),report_sha256=sha(root/'experiment/phase3-report.json'),
       source_report_sha256=sha(source/'student-rollout-report.json'),source_file_sha256=files,cache_identity_sha256=sha(cache/'chunk-identity.json'),
       raw_decisions=len(rows),aligned_samples=len(aligned),segments=len(segments),actual_previous_pairs=len(history)-len(segments),BOS_count=len(segments),
       actual_sequence_boundary_violations=bad[:20],split_proof=split_proof,model_proof=model_proof,code_seal_mismatch=source_mismatch,
       passed=not bad and not source_mismatch and all(x['matches_report'] for x in split_proof.values()) and all(x['exact_baseline_frozen'] for x in model_proof.values()),
       scope='identity and causal sequence proof only; no world safety claim')
    target=root/'independent-identity-audit.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='source_file_sha256'},indent=2));return result
if __name__=='__main__':
    import argparse;pa=argparse.ArgumentParser();pa.add_argument('--root',required=True);pa.add_argument('--source',required=True);pa.add_argument('--cache',required=True);a=pa.parse_args();audit(a.root,a.source,a.cache)
