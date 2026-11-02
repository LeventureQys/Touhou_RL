"""Fixed matched checkpoint closed-loop comparison; synthetic developer research."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from touhou_rl.tactical_synthetic_closed_loop import FrozenNetworks,run_event,BASELINE,RECORDS
from touhou_rl.tactical_teacher_validation import make_world
from touhou_rl.tactical_data import file_sha


def run(checkpoint,out):
    out=Path(out)
    if out.exists():raise FileExistsError('new comparison output required')
    source=Path('Document/Update/Alpha/v3.7.3/evidence/synthetic-closed-loop-round11/preregistration.json')
    p=json.loads(source.read_text(encoding='utf-8'))
    out.mkdir(parents=True);torch.set_num_threads(4)
    identity={'checkpoint_sha256':file_sha(checkpoint),'baseline_sha256':file_sha(BASELINE),
        'events_source_sha256':file_sha(source),'source':file_sha(__file__),'condition':'horizontal_masked',
        'selection':'fixed final step from equal budget warm start','formal_P5':False,'training_steps':0}
    (out/'preregistration.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    network=FrozenNetworks(BASELINE,checkpoint,'cuda');rows=[]
    for e in p['events']:
        result=run_event(make_world(e['scenario']),'N1','horizontal_masked',network,0,out/e['event_id'])
        rows.append(result)
        print(e['event_id'],result['feasible'],result['first_collision_frame'],flush=True)
    summary={}
    for family in ('F1','F2','F3','all'):
        group=[r for r in rows if family=='all' or r['family']==family]
        summary[family]={'events':len(group),'feasible':sum(r['feasible'] for r in group),
            'survived':sum(r['survived'] for r in group),'first_hits':{r['event_id']:r['first_collision_frame'] for r in group}}
    report={**identity,'summary':summary,'checkpoint_unchanged':file_sha(checkpoint)==identity['checkpoint_sha256']}
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.checkpoint,a.out)
