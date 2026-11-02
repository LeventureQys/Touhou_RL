"""Only train/dev danger coverage; does not inspect test outcomes."""
from pathlib import Path
import json
import numpy as np
from .vision_data import VisionDataset
from .vision_metrics import _nearest_bullet_px

def run(root):
    root=Path(root);p=json.loads((root/'preregistration.json').read_text());ci=json.loads((root/'student-chunk-cache/chunk-identity.json').read_text())
    with np.load(root/'student-chunk-cache/chunk-arrays.npz') as z:ep=z['episode'];v=z['model_valid'][:,:,1].all(1)
    d=VisionDataset(Path(ci['source']));out={}
    for key in ('train','dev'):
        ids=[i for i in range(len(ep)) if ep[i] in p['episodes'][key] and v[i]];b=d.labels(ids)['bullets'];near=_nearest_bullet_px(b)
        out[key]=dict(total=len(ids),under15=int((near<15).sum()),under30=int((near<30).sum()),under60=int((near<60).sum()),over120=int((near>=120).sum()))
    d.close();result=dict(schema='v200-phase2-train-dev-danger-coverage/1',test_used=False,splits=out)
    path=root/'train-dev-danger-coverage.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    import argparse;p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)
