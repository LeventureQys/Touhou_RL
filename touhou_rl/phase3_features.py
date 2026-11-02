"""Frozen pixel feature bank for controlled temporal experiments."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from . import vision
from .vision_train import _gather
from .build_chunk_safety_cache import sha


def bank(dataset,model,indices,device,out,source_identity):
    """Encode each center's legal depth3 pixel stack once; no label concatenation."""
    out=Path(out);indices=list(map(int,indices));identity=dict(schema='v200-phase3-feature-bank/1',source_identity=source_identity,
      indices_sha256=hashlib.sha256(np.asarray(indices,dtype=np.int64).tobytes()).hexdigest(),
      input='three ROI pixels only, causal-segment-v1 depth3',feature_dim=256)
    if out.exists():
        saved=json.loads(out.with_suffix('.json').read_text())
        if any(saved.get(k)!=v for k,v in identity.items()) or saved.get('array_sha256')!=sha(out):raise ValueError('feature bank identity mismatch')
        with np.load(out,allow_pickle=False) as z:return z['features'].copy(),z['bullets'].copy()
    features=[];bullets=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(indices),64):
            selected=indices[start:start+64];frames,labels=_gather(dataset,selected,3,model.config.region_sizes)
            value=model.features(vision.regions_to_batch(frames,device)).cpu().numpy().astype(np.float32)
            features.append(value);bullets.append(labels['bullets'])
            if start%1024==0:print(f'[phase3] encoded {start}/{len(indices)} pixel centers',flush=True)
    feature=np.concatenate(features);bullet=np.concatenate(bullets)
    if feature.shape!=(len(indices),256) or not np.isfinite(feature).all():raise ValueError('invalid pixel feature bank')
    out.parent.mkdir(parents=True,exist_ok=True);np.savez(out,features=feature,bullets=bullet)
    identity['array_sha256']=sha(out);out.with_suffix('.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    return feature,bullet
