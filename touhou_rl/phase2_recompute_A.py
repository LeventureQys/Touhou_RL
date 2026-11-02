"""Independent fixed A-val checkpoint reevaluation; never updates weights."""
from pathlib import Path
import json
import numpy as np
import torch
from .pipeline.manifest import DatasetManifest,index_digest
from .vision_data import VisionDataset
from .vmodels import load_vision_checkpoint
from .vision_train import teacher_identity,_fingerprint,teacher_cache_path,_inspect_teacher_cache
from .phase2_chunk_experiment import _eval
from .build_chunk_safety_cache import sha

def verify(root):
    root=Path(root);r=json.loads((root/'head-only-controlled/phase2-report.json').read_text());p=json.loads((root/'preregistration.json').read_text())
    adir=Path('third_party/th08/build/rl/datasets/pilot-ud0059-v2');manifest=DatasetManifest.load(adir/'manifest.json');_,val=manifest.verify(adir)
    if index_digest(val)!=p['A_val_index_sha256'] or sha(adir/'manifest.json')!=p['A_manifest_sha256']:raise ValueError('fixed A identity mismatch')
    dataset=VisionDataset(adir,history_contract='causal-segment-v1')
    identity=teacher_identity(dataset,adir,manifest=manifest,val_index_sha256=index_digest(val));fp=_fingerprint(identity)
    target,reason=_inspect_teacher_cache(teacher_cache_path(adir,fp),fp,len(dataset))
    if target is None:raise ValueError('verified A teacher cache absent '+reason)
    results={}
    for label,path in [('baseline',Path('third_party/th08/build/rl/runs/v105-stage2-formal-w0-s7/policy.pt')),('mixed',root/'head-only-controlled/policy.pt'),('A_only_control',root/'head-only-controlled/A-only-control.pt')]:
        model,_=load_vision_checkpoint(path,map_location='cpu');model.to('cuda').eval()
        actual=_eval(model,dataset,val,target[val],torch.device('cuda'))
        expected=r['before']['A'] if label=='baseline' else r['after'][label]['A']
        results[label]=dict(frames=len(val),regret_px=actual['regret_px'],danger_regret_px=next(x for x in actual['by_danger'] if x['bucket']=='0-15')['regret_px'],
          exact_candidates_equal=actual['per_frame']['candidate']==expected['per_frame']['candidate'],
          max_per_frame_regret_difference=float(np.max(np.abs(np.asarray(actual['per_frame']['regret'])-expected['per_frame']['regret']))))
        del model;torch.cuda.empty_cache()
    dataset.close()
    result=dict(schema='v200-phase2-independent-fixed-A-recompute/1',report_sha256=sha(root/'head-only-controlled/phase2-report.json'),A_val_index_sha256=index_digest(val),
                results=results,passed=all(x['exact_candidates_equal'] and x['max_per_frame_regret_difference']==0 for x in results.values()),training_executed=False)
    (root/'independent-A-recompute.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    import argparse;p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();verify(a.root)
