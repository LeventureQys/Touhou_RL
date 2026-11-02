"""Train/dev-only diagnosis: do rank updates harm already-good baseline states?"""
from pathlib import Path
import json,numpy as np

def run(root):
    root=Path(root);r=json.loads((root/'experiment/phase3-report.json').read_text());b=r['metrics']['baseline']['dev']['per_frame'];a=r['metrics']['temporal']['dev']['per_frame']
    before=np.asarray(b['regret']);after=np.asarray(a['regret']);near=np.asarray(b['nearest_bullet_px']);out={}
    for name,mask in [('baseline_near_optimal',before<=.5),('baseline_regret_gt05',before>.5),('danger_baseline_near_optimal',(near<15)&(before<=.5)),('danger_baseline_regret_gt05',(near<15)&(before>.5))]:
        out[name]=dict(frames=int(mask.sum()),before=float(before[mask].mean()) if mask.any() else None,after=float(after[mask].mean()) if mask.any() else None,
                       newly_harmed=int(((after-before>.1)&mask).sum()))
    result=dict(schema='v200-phase3-dev-correction-diagnostic/1',test_used=False,groups=out)
    target=root/'dev-correction-diagnostic.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    import argparse;p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)
