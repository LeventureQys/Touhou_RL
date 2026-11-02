"""Bounded real-game teacher action-hold clearance probe. No pixel training data."""
from pathlib import Path
from dataclasses import asdict
import json,time
import numpy as np
from .env import Th08Env,EnvConfig
from .pilot import AvoidPilot,PilotConfig
from .collect_student_states import _teacher_query,_state,_sha
from .executed_clearance import audit
from .winapi import keep_awake

def run(protocol,out,run_dir):
    protocol,out,run_dir=map(Path,(protocol,out,run_dir))
    if out.exists():raise FileExistsError(out)
    p=json.loads(protocol.read_text());budget=p['clearance_probe'];env=Th08Env(EnvConfig(run_dir=run_dir,frameskip=2,
      max_steps=budget['max_steps_each'],reset_mode='relaunch',expose_raw_bullets=True,record_input_evidence=True,evidence_full_pool=True))
    rows=[];started=time.monotonic();out.mkdir(parents=True);keep_awake(True)
    try:
        for episode in range(1,budget['episodes']+1):
            observation,info=env.reset(seed=budget['seed']+episode-1);teacher=AvoidPilot(PilotConfig(switch_penalty=0))
            for decision in range(budget['max_steps_each']):
                if time.monotonic()-started>budget['max_seconds']:break
                duration=(1,3,5)[decision%3];env.config.frameskip=2*duration
                action,clearance=_teacher_query(teacher,observation,_state(info))
                observation,_,terminated,truncated,info=env.step(action)
                teacher._prev_move=int(action[0]);teacher._bomb_cooldown=max(0,teacher._bomb_cooldown-1)
                rows.append(dict(episode=episode,decision=decision,executed_action=action.astype(int).tolist(),
                  input_timing=info['input_timing'],duration_decisions=duration))
                if terminated or truncated:break
    finally:
        env.close();keep_awake(False)
        (out/'executed-trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    result=audit(rows);result.update(protocol_sha256=_sha(protocol),trace_sha256=_sha(out/'executed-trace.jsonl'))
    gate=p['clearance_gate'];result['passed']=all(result['per_duration'][str(d)]>=gate['min_each_duration'] for d in (1,3,5)) and result['clearance_error_p95'] is not None and result['clearance_error_p95']<=gate['error_p95_max_px'] and result['false_safe_count']<=gate['false_safe_max']
    (out/'clearance-calibration.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2));return result
if __name__=='__main__':
    import argparse;p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--out',required=True);p.add_argument('--run-dir',required=True);a=p.parse_args();run(a.protocol,a.out,a.run_dir)
