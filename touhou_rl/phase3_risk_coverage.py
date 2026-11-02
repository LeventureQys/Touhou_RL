"""Read-only risk gate coverage from pixels/features, no threshold fitting."""
from pathlib import Path
import json
import numpy as np
import torch
from .temporal_policy import load_temporal_checkpoint
from .vision_metrics import _nearest_bullet_px
from .build_chunk_safety_cache import sha


def audit(root):
    root=Path(root);p=json.loads((root/'preregistration.json').read_text());checkpoint=root/'experiment/temporal-policy.pt'
    model=load_temporal_checkpoint(checkpoint,map_location='cpu').eval()
    with np.load(root/'experiment/student-features.npz') as z:features=z['features'].copy();bullets=z['bullets'].copy()
    with np.load(root/'student-chunk-cache/chunk-arrays.npz') as z:episode=z['episode'].copy();valid=z['model_valid'].all((1,2))
    distance=_nearest_bullet_px(bullets)
    with torch.inference_mode():prob=model.risk_head(torch.tensor(features)).sigmoid().squeeze(-1).numpy()
    threshold=model.config.risk_gate_threshold;gate=prob>=threshold;results={}
    for split in ('train','dev','test'):
        mask=np.isin(episode,p['episodes'][split])&valid
        actual=(distance<30)&mask;danger=(distance<15)&mask
        results[split]=dict(frames=int(mask.sum()),residual_allowed_fraction=float((~gate[mask]).mean()),
          current_under30_count=int(actual.sum()),under30_recall=float(gate[actual].mean()) if actual.any() else None,
          critical_under15_count=int(danger.sum()),under15_recall=float(gate[danger].mean()) if danger.any() else None,
          gate_precision=float((distance[mask&gate]<30).mean()) if (mask&gate).any() else None)
    result=dict(schema='v200-phase3-pixel-risk-gate-coverage/1',protocol_sha256=sha(root/'preregistration.json'),checkpoint_sha256=sha(checkpoint),threshold_fixed=threshold,
      source='risk_head(frozen pixel features) only; current distances are audit labels, never inference input',results=results,threshold_fitted_on_test=False)
    target=root/'risk-gate-coverage.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    import argparse;pa=argparse.ArgumentParser();pa.add_argument('root');a=pa.parse_args();audit(a.root)
