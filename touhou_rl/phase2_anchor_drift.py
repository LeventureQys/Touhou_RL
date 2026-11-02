"""Train/dev-only baseline logit drift audit for A anchoring mechanism."""
from pathlib import Path
import json
import numpy as np
import torch
from .vmodels import load_vision_checkpoint
from .vision_data import VisionDataset
from .vision_train import _gather
from . import vision

def run(root):
    root=Path(root);protocol=json.loads((root/'preregistration.json').read_text());cache=json.loads((root/'student-chunk-cache/chunk-identity.json').read_text())
    dataset=VisionDataset(Path(cache['source']),history_contract='causal-segment-v1')
    with np.load(root/'student-chunk-cache/chunk-arrays.npz') as z:
        episodes=z['episode'];valid=z['model_valid'][:,:,1].all(1)
    ids=[i for i in range(len(episodes)) if episodes[i] in protocol['episodes']['train']+protocol['episodes']['dev'] and valid[i]]
    ids=ids[::max(1,len(ids)//128)][:128]
    baseline,_=load_vision_checkpoint(Path('third_party/th08/build/rl/runs/v105-stage2-formal-w0-s7/policy.pt'))
    mixed,_=load_vision_checkpoint(root/'head-only-controlled/policy.pt')
    baseline.to('cuda').eval();mixed.to('cuda').eval();drift=[];flip=[]
    with torch.no_grad():
        for start in range(0,len(ids),32):
            frames,_=_gather(dataset,ids[start:start+32],3,baseline.config.region_sizes);batch=vision.regions_to_batch(frames,'cuda')
            a=baseline.score(batch);b=mixed.score(batch);drift.extend(((a-b)**2).mean(1).cpu().tolist());flip.extend((a.argmax(1)!=b.argmax(1)).cpu().tolist())
    dataset.close();r=dict(schema='v200-phase2-anchor-drift/1',splits_used=['train','dev'],test_used=False,frames=len(ids),
      logit_mse_mean=float(np.mean(drift)),candidate_change_fraction=float(np.mean(flip)),anchor_weight=protocol['experiment'].get('anchor_weight',0),
      note='descriptive train/dev drift, no test-guided weight sweep')
    target=root/'train-dev-anchor-drift.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2));return r
if __name__=='__main__':
    import argparse;p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)
