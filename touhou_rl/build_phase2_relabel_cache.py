"""Build an identity-bound Phase 2 relabel cache from student traces.

The cache is diagnostic/training material only. It does not alter source shards or
A manifests. Future risk is computed from the recorded per-decision teacher
clearance and death evidence; duration is an auditable contiguous teacher action
run, not a guessed label.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

SCHEMA="v200-student-relabel-cache/2"

def sha(path:Path)->str:
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def build(rollout:Path,out:Path,horizon:int=50):
 if out.exists(): raise FileExistsError(out)
 report=json.loads((rollout/'student-rollout-report.json').read_text(encoding='utf-8'))
 if report.get('phase1_collection_status')!='collection-ready-for-distribution-audit': raise ValueError('rollout did not pass the collection gate')
 if report.get('training_approved') is not False: raise ValueError('source must remain diagnostic-only')
 if horizon<1: raise ValueError('future horizon must be positive')
 for item in report['shards']:
  if sha(rollout/item['name'])!=item['sha256']: raise ValueError(f"student shard SHA mismatch: {item['name']}")
 rows=[]
 for ep in report['episodes']:
  trace=rollout/ep['trace']
  if sha(trace)!=ep['trace_sha256']: raise ValueError(f'trace SHA mismatch: {trace.name}')
  rows.extend(json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines())
 valid=[r for r in rows if r.get('sample_index') is not None]
 if not valid: raise ValueError('no aligned samples')
 if [int(r['sample_index']) for r in valid] != list(range(report['samples'])):
  raise ValueError('trace samples do not match shard order')
 # Map future death evidence by same episode only; death event attribution is already
 # corrected in collector before trace finalization.
 by_ep={}
 for r in rows: by_ep.setdefault(int(r['episode']),[]).append(r)
 labels=[]
 for r in valid:
  ep=by_ep[int(r['episode'])]; pos=next(i for i,x in enumerate(ep) if x is r)
  window=ep[pos:min(len(ep),pos+horizon+1)]
  death=any(int(x['deaths_after'])>int(x['deaths_before']) for x in window)
  contiguous=all(int(x['decision'])==int(r['decision'])+offset and int(x['stage'])==int(r['stage'])
                 and x.get('sample_index') is not None
                 for offset,x in enumerate(window))
  known=death or (len(window)>=horizon+1 and contiguous)
  # A teacher-action run ends at any missing/unaligned state, stage/death
  # transition, terminal, or changed recommendation; only observed length is used.
  action=tuple(r['teacher_action'])
  duration=1
  duration_valid=False
  prior=r
  for nxt in ep[pos+1:pos+5]:
   boundary=(int(nxt['decision'])!=int(prior['decision'])+1 or int(nxt['stage'])!=int(r['stage'])
             or int(prior['deaths_after'])!=int(nxt['deaths_before'])
             or bool(prior.get('terminated')) or bool(prior.get('truncated')))
   if boundary or nxt.get('sample_index') is None or tuple(nxt['teacher_action'])!=action:
    duration_valid=not boundary and nxt.get('sample_index') is not None
    break
   duration+=1;prior=nxt
  if duration==5: duration_valid=True
  duration_class=2 if duration>=5 else 1 if duration>=3 else 0
  labels.append({
   'sample_index':int(r['sample_index']),'episode':int(r['episode']),'decision':int(r['decision']),
   'teacher_action':list(action),'teacher_candidate_index':int(action[2])*9+int(action[0]),
   'teacher_clearance_px':float(r.get('teacher_clearance',0.0)),
   'future_death':bool(death),'future_death_valid':bool(known),'future_horizon_decisions':horizon,
   'duration_decisions':int(duration),'duration_class':int(duration_class),
   'duration_valid':bool(duration_valid),
   'student_action':list(r['executed_action']),
  })
 out.mkdir(parents=True)
 np.savez(out/'relabel-arrays.npz',
  sample_index=np.asarray([x['sample_index'] for x in labels],dtype=np.int64),
  teacher_candidate=np.asarray([x['teacher_candidate_index'] for x in labels],dtype=np.int64),
  teacher_clearance=np.asarray([x['teacher_clearance_px'] for x in labels],dtype=np.float32),
  future_death=np.asarray([x['future_death'] for x in labels],dtype=np.uint8),
  future_death_valid=np.asarray([x['future_death_valid'] for x in labels],dtype=np.uint8),
  duration_decisions=np.asarray([x['duration_decisions'] for x in labels],dtype=np.uint8),
  duration_class=np.asarray([x['duration_class'] for x in labels],dtype=np.uint8),
  duration_valid=np.asarray([x['duration_valid'] for x in labels],dtype=np.uint8),
  episode=np.asarray([x['episode'] for x in labels],dtype=np.int64),
  decision=np.asarray([x['decision'] for x in labels],dtype=np.int64))
 identity={
  'schema':SCHEMA,'rollout':str(rollout.resolve()),'rollout_report_sha256':sha(rollout/'student-rollout-report.json'),
  'checkpoint_sha256':report['checkpoint_sha256'],'teacher_source_sha256':report['teacher_source_sha256'],
  'teacher_config':report['teacher_config'],'student_shards_sha256':[x['sha256'] for x in report['shards']],
  'traces_sha256':[x['trace_sha256'] for x in report['episodes']],
  'rollout_id':report.get('rollout_id'),
  'future_horizon_decisions':horizon,'duration_classes':[1,3,5],'rows':len(labels),
  'future_risk_semantics':'observed-death-under-executed-student-trajectory, not counterfactual per-action safety',
  'duration_semantics':'observed contiguous teacher recommendations; valid only with next aligned step or 5 observed steps',
  'source_training_approved':False,'label_source':'student-trace-teacher-sidecar',
 }
 identity['cache_sha256']=sha(out/'relabel-arrays.npz')
 (out/'relabel-identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
 (out/'relabel-rows.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in labels),encoding='utf-8')
 return identity

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--rollout',required=True);p.add_argument('--out',required=True);p.add_argument('--horizon',type=int,default=50)
 a=p.parse_args(argv); print(json.dumps(build(Path(a.rollout),Path(a.out),a.horizon),ensure_ascii=False,indent=2)); return 0
if __name__=='__main__':raise SystemExit(main())
