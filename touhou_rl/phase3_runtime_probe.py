"""Pixel-only runtime diagnostic for a saved temporal chunk policy.

This is NOT a Phase4 survival benchmark. It writes decisions/latency only; no
privileged state reaches the model, no training, and owns a fresh game session.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from .temporal_policy import load_temporal_checkpoint
from .phase3_executor import ChunkExecutor
from .env import Th08Env,EnvConfig
from . import vision
from .build_chunk_safety_cache import sha
from .winapi import keep_awake


def run(checkpoint,out,run_dir,episodes=2,max_decisions=100,max_seconds=120):
    checkpoint,out,run_dir=map(Path,(checkpoint,out,run_dir))
    if out.exists():raise FileExistsError(out)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=load_temporal_checkpoint(checkpoint,map_location='cpu').to(device).eval()
    env=Th08Env(EnvConfig(run_dir=run_dir,frameskip=2,max_steps=max_decisions,reset_mode='relaunch'));executor=ChunkExecutor(model)
    sizes=model.vision_model.config.region_sizes;depth=model.vision_model.config.depth
    all_rows=[];started=time.monotonic();out.mkdir(parents=True);keep_awake(True)
    try:
        for episode in range(1,episodes+1):
            env.reset(seed=41000+episode);executor.reset()
            stacks={name:vision.FrameStack(depth=depth,shape=(*hw,3)) for name,hw in sizes.items()}
            for stack in stacks.values():stack.reset()
            initial=vision.extract_regions(env.render())
            for name,stack in stacks.items():
                image=vision.resize_area(initial[name],*sizes[name])
                for _ in range(depth):stack.push(image)
            for decision in range(max_decisions):
                if time.monotonic()-started>max_seconds:break
                screen=env.render();native=vision.extract_regions(screen)
                regions={name:stacks[name].push(vision.resize_area(native[name],*sizes[name])) for name in stacks}
                if device.type=='cuda':torch.cuda.synchronize()
                begin=time.perf_counter()
                with torch.inference_mode():pair,queried=executor.choose(vision.regions_to_batch(regions,device))
                if device.type=='cuda':torch.cuda.synchronize()
                latency=(time.perf_counter()-begin)*1000
                candidate,duration=pair;action=np.array([candidate%9,1,candidate//9,0],np.int64)
                _,_,terminated,truncated,info=env.step(action);executor.acknowledge(terminated,truncated)
                all_rows.append(dict(episode=episode,decision=decision,candidate=candidate,requested_duration_decisions=duration,
                    policy_queried=queried,inference_ms=latency,terminated=terminated,truncated=truncated,
                    termination_reason=info.get('termination_reason','')))
                if terminated or truncated:break
    finally:executor.reset();env.close();keep_awake(False)
    trace=out/'runtime-trace.jsonl';trace.write_text(''.join(json.dumps(r)+'\n' for r in all_rows),encoding='utf-8')
    report=dict(schema='v200-phase3-runtime-diagnostic/1',checkpoint_sha256=sha(checkpoint),trace_sha256=sha(trace),
      episodes=episodes,decisions=len(all_rows),policy_queries=sum(r['policy_queried'] for r in all_rows),
      pixel_only=True,hidden_and_history_reset_between_episodes=True,phase4_executed=False,
      scope='bounded integration diagnostic only; no survival comparative claim')
    (out/'runtime-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--episodes',type=int,default=2);p.add_argument('--max-decisions',type=int,default=100);p.add_argument('--max-seconds',type=int,default=120);a=p.parse_args();run(a.checkpoint,a.out,a.run_dir,a.episodes,a.max_decisions,a.max_seconds)
