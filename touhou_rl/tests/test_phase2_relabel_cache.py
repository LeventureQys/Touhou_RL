"""Contract tests for Phase 2 relabel identity and non-counterfactual labels."""
import json
import numpy as np
import pytest
from pathlib import Path
from touhou_rl.build_phase2_relabel_cache import build,sha


def make_rollout(root:Path):
 root.mkdir()
 trace=[]
 for episode in (1,2):
  for decision in range(4):
   sample=(episode-1)*4+decision
   trace.append(dict(sample_index=sample,episode=episode,decision=decision,stage=0,
       teacher_action=[3,1,0,0] if decision<3 else [5,1,0,0],
       executed_action=[0,1,0,0],teacher_clearance=float(decision),
       deaths_before=0,deaths_after=1 if episode==2 and decision==2 else 0))
 episodes=[]
 for episode in (1,2):
  path=root/f'episode-{episode:03d}-decisions.jsonl'
  path.write_text(''.join(json.dumps(x)+'\n' for x in trace if x['episode']==episode),encoding='utf-8')
  episodes.append(dict(trace=path.name,trace_sha256=sha(path)))
 shard=root/'student-state-00000.npz'; shard.write_bytes(b'fictional-stub-shard')
 report=dict(training_approved=False,phase1_collection_status='collection-ready-for-distribution-audit',
  samples=8,episodes=episodes,checkpoint_sha256='student',teacher_source_sha256='pilot',
  teacher_config={'frameskip':2},shards=[dict(name=shard.name,sha256=sha(shard))])
 (root/'student-rollout-report.json').write_text(json.dumps(report),encoding='utf-8')
 return trace


def test_relabel_cache_episode_boundary_death_and_duration(tmp_path):
 root=tmp_path/'rollout';make_rollout(root)
 identity=build(root,tmp_path/'cache',horizon=2)
 assert identity['rows']==8
 with np.load(tmp_path/'cache'/'relabel-arrays.npz') as data:
  assert data['sample_index'].tolist()==list(range(8))
  assert data['duration_class'].tolist()==[1,0,0,0,1,0,0,0]
  assert data['future_death'].tolist()[:4]==[0,0,0,0]
  assert data['future_death'].tolist()[4:]==[1,1,1,0]
  assert data['future_death_valid'].tolist()[3]==0
  assert sha(tmp_path/'cache'/'relabel-arrays.npz')==identity['cache_sha256']
 with pytest.raises(FileExistsError): build(root,tmp_path/'cache',horizon=2)


def test_relabel_cache_rejects_tampered_trace_and_shard(tmp_path):
 root=tmp_path/'rollout';make_rollout(root)
 (root/'student-state-00000.npz').write_bytes(b'altered')
 with pytest.raises(ValueError,match='shard SHA'):
  build(root,tmp_path/'invalid',horizon=2)
