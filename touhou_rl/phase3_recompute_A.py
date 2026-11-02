"""Read-only frozen-baseline A-val reevaluation of saved Phase3 policies."""
from pathlib import Path
import json
import numpy as np
import torch
from .temporal_policy import load_temporal_checkpoint
from .vmodels import load_vision_checkpoint
from .phase3_experiment import infer,metric
from .pipeline.manifest import DatasetManifest,index_digest
from .vision_data import VisionDataset
from .vision_train import teacher_identity,_fingerprint,teacher_cache_path,_inspect_teacher_cache
from .build_chunk_safety_cache import sha


def verify(root):
    root=Path(root);p=json.loads((root/'preregistration.json').read_text());r=json.loads((root/'experiment/phase3-report.json').read_text());device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adir=Path('third_party/th08/build/rl/datasets/pilot-ud0059-v2');manifest=DatasetManifest.load(adir/'manifest.json');train,val=manifest.verify(adir)
    if sha(adir/'manifest.json')!=p['A_manifest_sha256'] or index_digest(val)!=p['A_val_index_sha256']:raise ValueError('A fixed identity mismatch')
    with np.load(root/'experiment/A-features.npz') as z:features=z['features'].copy();bullets=z['bullets'].copy()
    side=json.loads((root/'experiment/A-features.json').read_text());expected=side['source_identity']['indices_sha256']
    # The deterministic stage/shard subset is recomputed without test labels.
    from .phase2_chunk_experiment import _stratified_a
    dataset=VisionDataset(adir,history_contract='causal-segment-v1');n=r['splits']['train']['frames'];subset=_stratified_a(train,min(len(train),3*n),dataset)
    ids=sorted(set(subset+val))
    if index_digest(ids)!=expected:raise ValueError('A feature indices mismatch')
    if sha(root/'experiment/A-features.npz')!=side['array_sha256']:raise ValueError('A feature bank digest mismatch')
    lookup={i:j for j,i in enumerate(ids)};val_local=[lookup[i] for i in val]
    identity=teacher_identity(dataset,adir,manifest=manifest,val_index_sha256=index_digest(val));fp=_fingerprint(identity)
    values,reason=_inspect_teacher_cache(teacher_cache_path(adir,fp),fp,len(dataset));dataset.close()
    if values is None:raise ValueError('A teacher cache absent '+reason)
    target=values[ids].astype(np.float32);result={}
    for label,name in [('temporal','temporal-policy.pt'),('reset_control','reset-control.pt')]:
        model=load_temporal_checkpoint(root/'experiment'/name,map_location='cpu').to(device).eval()
        scores,dur=infer(model,features,[[i] for i in val_local],np.full(len(ids),18),np.full(len(ids),3),device,True)
        actual=metric(scores,dur,target,bullets,val_local,np.zeros(len(ids),int));saved=r['metrics'][label]['A']
        result[label]=dict(frames=len(val),regret_px=actual['regret_px'],candidate_equal=actual['per_frame']['candidate']==saved['per_frame']['candidate'],max_regret_diff=float(np.max(np.abs(np.asarray(actual['per_frame']['regret'])-saved['per_frame']['regret']))))
        del model;torch.cuda.empty_cache()
    report=dict(schema='v200-phase3-independent-A-recompute/1',report_sha256=sha(root/'experiment/phase3-report.json'),A_val_index_sha256=index_digest(val),results=result,passed=all(v['candidate_equal'] and v['max_regret_diff']<1e-5 for v in result.values()),training_executed=False)
    target_path=root/'independent-A-recompute.json'
    if target_path.exists():raise FileExistsError(target_path)
    target_path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));return report
if __name__=='__main__':
    import argparse;pa=argparse.ArgumentParser();pa.add_argument('root');a=pa.parse_args();verify(a.root)
