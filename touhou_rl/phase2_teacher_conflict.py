"""Train/dev-only target conflict audit, never evaluates student test."""
from pathlib import Path
import json
import numpy as np
from .pilot import AvoidPilot,PilotConfig
from .vision_data import VisionDataset
from .vision_train import _compute_teacher_values
from .build_chunk_safety_cache import sha

def run(root):
    root=Path(root);protocol=json.loads((root/'preregistration.json').read_text());ci=json.loads((root/'student-chunk-cache/chunk-identity.json').read_text());source=Path(ci['source'])
    d=VisionDataset(source,history_contract='causal-segment-v1')
    with np.load(root/'student-chunk-cache/chunk-arrays.npz') as z:
        eps=z['episode'];clear=z['clearance'][:,:,1];valid=z['model_valid'][:,:,1].all(1)
    ids=[i for i in range(len(eps)) if eps[i] in protocol['episodes']['train']+protocol['episodes']['dev'] and valid[i]]
    labels=d.labels(ids);teacher=AvoidPilot(PilotConfig(switch_penalty=0))
    old=[]
    for j,i in enumerate(ids):
        obs={k:labels[k][j] for k in ('self_feat','bullets','danger')}
        old.append(teacher.candidate_scores(obs).astype(float))
    old=np.asarray(old);new=clear[ids];old_choice=old.argmax(1);new_choice=new.argmax(1)
    nearest=np.linalg.norm(labels['bullets'][:,:,:2]*np.array([192.,224.]),axis=-1)
    nearest=np.where(labels['bullets'][:,:,5]>.5,nearest,np.inf).min(1)
    result=dict(schema='v200-phase2-teacher-conflict/1',protocol_sha256=sha(root/'preregistration.json'),source_report_sha256=sha(source/'student-rollout-report.json'),
      splits_used=['train','dev'],test_used=False,rows=len(ids),candidate_top1_agreement=float(np.mean(old_choice==new_choice)),
      old_choice_chunk_regret=float(np.mean(new.max(1)-new[np.arange(len(ids)),old_choice])),
      chunk_choice_old_teacher_regret=float(np.mean(old.max(1)-old[np.arange(len(ids)),new_choice])),
      pair_order_disagreement=float(np.mean(np.sign(old[:,:,None]-old[:,None,:])!=np.sign(new[:,:,None]-new[:,None,:]))),
      danger_rows=int((nearest<15).sum()),note='old teacher top64 beam vs full-pool forced hold differ in both horizon semantics and threat coverage; disagreement is not truth error')
    d.close();path=root/'train-dev-teacher-conflict.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    import argparse;p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)
