"""Bounded Phase2 collection entry; all outputs are new directories."""
from pathlib import Path
import argparse, json
from .collect_student_states import StudentRolloutConfig, collect_student_states, _sha
from .pilot import PilotConfig

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--episodes',type=int,required=True);p.add_argument('--max-steps',type=int,required=True)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--behavior',choices=['student','teacher'],default='student');p.add_argument('--epsilon',type=float,default=0)
    p.add_argument('--protocol',required=True)
    a=p.parse_args()
    base=Path('third_party/th08/build/rl/runs/v105-stage2-formal-w0-s7/policy.pt')
    r=collect_student_states(StudentRolloutConfig(checkpoint=base,out_dir=Path(a.out),run_dir=Path(a.run_dir),
       episodes=a.episodes,max_steps=a.max_steps,max_seconds=a.max_seconds,seed=a.seed,behavior=a.behavior,
       epsilon=a.epsilon,record_input_evidence=True,protocol_sha256=_sha(Path(a.protocol)),
       pilot=PilotConfig(switch_penalty=0),shard_size=128))
    print(json.dumps({'samples':r.get('samples'),'alignment_ratio':r.get('alignment_ratio'),
                      'episodes':r['episodes'],'status':r.get('phase1_collection_status')},indent=2))
if __name__=='__main__': main()
