"""Recompute heldout report metrics from saved checkpoints without training."""
import json
from pathlib import Path
import numpy as np
import torch
from .phase2_chunk_experiment import _eval
from .vmodels import load_vision_checkpoint
from .vision_data import VisionDataset
from .build_chunk_safety_cache import sha

def verify(root):
    root=Path(root);r=json.loads((root/'head-only-controlled/phase2-report.json').read_text());p=json.loads((root/'preregistration.json').read_text())
    cache_info=json.loads((root/'student-chunk-cache/chunk-identity.json').read_text())
    source=Path(cache_info['source'])
    s=VisionDataset(source,history_contract='causal-segment-v1')
    with np.load(root/'student-chunk-cache/chunk-arrays.npz') as z:
        episode=z['episode'].copy();target=z['clearance'][:,:,1].copy();valid=z['model_valid'][:,:,1].all(1)
    # Source trace provides the exact eligibility; match saved split sample count.
    from .build_chunk_safety_cache import audit_source
    _,_,aligned,_=audit_source(source)
    rows={x['sample_index']:x for x in aligned}
    result={}
    for label,path in [('mixed',root/'head-only-controlled/policy.pt'),('A_only_control',root/'head-only-controlled/A-only-control.pt')]:
        model,_=load_vision_checkpoint(path,map_location='cpu');model.to('cuda').eval()
        ids=[i for i in range(len(episode)) if episode[i] in p['episodes']['test'] and valid[i] and not rows[i].get('terminated') and not rows[i].get('truncated') and rows[i].get('stage_delta',0)==0]
        metric=_eval(model,s,ids,target[ids],torch.device('cuda'))
        expected=r['after'][label]['student']['student_test']
        result[label]=dict(frames=len(ids),regret_px=metric['regret_px'],expected_regret_px=expected['regret_px'],
                           exact_candidates_equal=metric['per_frame']['candidate']==expected['per_frame']['candidate'],
                           per_frame_max_regret_difference=float(np.max(np.abs(np.asarray(metric['per_frame']['regret'])-expected['per_frame']['regret']))))
        del model;torch.cuda.empty_cache()
    s.close()
    report=dict(schema='v200-phase2-independent-metric-recompute/1',report_sha256=sha(root/'head-only-controlled/phase2-report.json'),results=result,
                passed=all(v['exact_candidates_equal'] and v['per_frame_max_regret_difference']<1e-4 for v in result.values()),training_executed=False)
    (root/'independent-test-recompute.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report
if __name__=='__main__':
    import argparse;p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();verify(a.root)
