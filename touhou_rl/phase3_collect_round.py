"""Bounded Phase3 data collection from a frozen *pixel-only* behavior policy."""
from pathlib import Path
import argparse,json
from .collect_student_states import StudentRolloutConfig,collect_student_states,_sha
from .pilot import PilotConfig

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--out',required=True);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();path=Path(args.protocol);p=json.loads(path.read_text());c=p['collection']
    checkpoint=Path(p['baseline_checkpoint'])
    if _sha(checkpoint)!=p['baseline_sha256']:raise ValueError('frozen Phase2 behavior SHA mismatch')
    r=collect_student_states(StudentRolloutConfig(checkpoint=checkpoint,out_dir=Path(args.out),run_dir=Path(args.run_dir),
      episodes=c['episodes'],max_steps=c['max_steps_each'],max_seconds=c['max_seconds'],seed=c['seed'],
      record_input_evidence=True,protocol_sha256=_sha(path),pilot=PilotConfig(switch_penalty=0),shard_size=128))
    print(json.dumps({k:r[k] for k in ('samples','alignment_ratio','phase1_collection_status','episodes')},indent=2))
if __name__=='__main__':main()
