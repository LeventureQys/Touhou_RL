import json
import pytest
from touhou_rl.phase2_final_gate import evaluate
from touhou_rl.build_chunk_safety_cache import sha

def metric(regret=1.,n=100):return dict(regret_px=regret,frames=n,by_danger=[dict(bucket='0-15',regret_px=regret,frames=n)],max_single_candidate_fraction=.2)
def fixture(tmp_path):
    protocol=tmp_path/'protocol.json'
    protocol.write_text(json.dumps(dict(checkpoint_sha256='a'*64,A_manifest_sha256='b'*64,A_val_index_sha256='c'*64,
      episodes=dict(train=[1],dev=[2],test=[3]),final_gate=dict(student_regret_relative_improvement_min=.05,A_overall_regret_increase_max_px=.1,A_danger_regret_increase_max_px=.15,student_danger_regret_increase_max_px=.1,max_single_candidate_fraction=.95),
      experiment=dict(min_danger_test_samples=50,min_train_samples=50,min_dev_samples=50,min_test_samples=50))))
    cache=tmp_path/'student-chunk-cache';cache.mkdir();(cache/'chunk-identity.json').write_text('{}')
    report=dict(schema='v200-phase2-chunk-experiment/1',protocol_sha256=sha(protocol),baseline_checkpoint_sha256='a'*64,A_manifest_sha256='b'*64,A_val_index_sha256='c'*64,
      cache_identity_sha256=sha(cache/'chunk-identity.json'),candidate_mapping='focus*9+move',input_depth=3,
      splits={k:dict(episodes=[i],frames=100) for k,i in [('student_train',1),('student_dev',2),('student_test',3)]},
      history_isolation=dict(episode_crossings=0,split_crossings=0),before=dict(A=metric(),student=dict(student_test=metric())),
      after=dict(mixed=dict(A=metric(2.),student=dict(student_test=metric(.8))),A_only_control=dict(A=metric(),student=dict(student_test=metric(.9)))),
      frozen_state_sha256={x:dict(before='a',after='a') for x in ('mixed','control')},train=dict(optimizer_steps=4,control_optimizer_steps=4,ratio=3,epochs=1),B_used=False,
      bootstrap=dict(student_test=dict(danger_0_15=dict(mixed_minus_baseline=dict(episodes=2,hi95=-.01)))))
    return protocol,report

def test_independent_final_gate_rejects_harm_and_missing_clearance_evidence(tmp_path):
    protocol,report=fixture(tmp_path);rp=tmp_path/'report.json';rp.write_text(json.dumps(report));r=evaluate(rp,protocol,tmp_path/'gate.json')
    assert not r['passed'] and not r['gates']['A_val_overall_nonregression']
    assert not r['gates']['executed_clearance_calibration']
    assert r['gates']['student_test_overall_improvement']

def test_wrong_source_identity_rejected(tmp_path):
    protocol,report=fixture(tmp_path);report['baseline_checkpoint_sha256']='x'*64
    rp=tmp_path/'report.json';rp.write_text(json.dumps(report))
    with pytest.raises(ValueError):evaluate(rp,protocol,tmp_path/'gate.json')
