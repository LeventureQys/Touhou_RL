"""CPU-only identity and corruption tests for the independent chunk cache."""
import json
import numpy as np
import pytest
from touhou_rl.build_chunk_safety_cache import build, validate, sha


def source(tmp_path, raw=True):
    root=tmp_path/'source'; root.mkdir()
    snapshot=dict(schema='v200-raw-safety-snapshot/1',frame=10,stage=1,complete=True,
                  player_position=[192.,224.],bullets=dict(position=[],velocity=[],half=[]))
    row=dict(rollout_id='r',episode=1,decision=0,sample_index=0,frame=10,stage=1,
             frame_delta=2,stage_delta=0,executed_action=[3,1,0,0],teacher_action=[0,1,1,0],aligned=True,fresh_pixels=True)
    if raw: row['raw_safety_snapshot']=snapshot
    (root/'trace.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    np.savez(root/'shard.npz',meta=np.array([[10,1,0,0,3,0,0,0]]),decision_meta=np.array([[1,0,2,0]]),
             executed_action=np.array([[3,1,0,0]]),action=np.array([[0,1,1,0]]))
    report=dict(training_approved=False,rollout_id='r',frameskip=2,samples=1,checkpoint_sha256='frozen',
                teacher_source_sha256='historic',teacher_config={},episodes=[dict(episode=1,decisions=1,trace='trace.jsonl',trace_sha256=sha(root/'trace.jsonl'))],
                shards=[dict(name='shard.npz',sha256=sha(root/'shard.npz'))])
    (root/'student-rollout-report.json').write_text(json.dumps(report),encoding='utf-8')
    return root


def rewrite_report(root):
    path=root/'student-rollout-report.json'; r=json.loads(path.read_text())
    r['episodes'][0]['trace_sha256']=sha(root/'trace.jsonl'); r['shards'][0]['sha256']=sha(root/'shard.npz')
    path.write_text(json.dumps(r))


def test_build_and_verify(tmp_path):
    root=source(tmp_path); out=tmp_path/'cache'
    identity=build(root,out)
    assert identity['model_valid_cells']==54 and identity['world_risk_valid_cells']==0
    assert validate(out)['rows']==1
    with pytest.raises(FileExistsError): build(root,out)


def test_old_pool_is_unknown(tmp_path):
    root=source(tmp_path,False); out=tmp_path/'cache'; r=build(root,out)
    assert r['model_valid_cells']==0 and r['missing_raw_snapshots']==1
    with np.load(out/'chunk-arrays.npz') as z:
        assert np.isnan(z['clearance']).all()


@pytest.mark.parametrize('mutation',['reorder','duplicate','wrong_action','wrong_stamp','wrong_shape','wrong_hash'])
def test_source_rejects_corruption(tmp_path,mutation):
    root=source(tmp_path); path=root/'trace.jsonl'; r=json.loads(path.read_text())
    if mutation=='reorder': r['decision']=1
    if mutation=='wrong_action': r['executed_action']=[7,1,0,0]
    if mutation=='wrong_stamp': r['raw_safety_snapshot']['frame']=11
    if mutation=='wrong_hash': r['extra_corruption']=True
    if mutation=='duplicate': path.write_text(json.dumps(r)+'\n'+json.dumps(r)+'\n')
    else: path.write_text(json.dumps(r)+'\n')
    if mutation=='wrong_shape':
        with np.load(root/'shard.npz') as z: data={k:z[k] for k in z.files}
        data['executed_action']=np.zeros((0,4)); np.savez(root/'shard.npz',**data)
    if mutation!='wrong_hash': rewrite_report(root)
    with pytest.raises(ValueError): build(root,tmp_path/'cache')
    assert not (tmp_path/'cache').exists()


@pytest.mark.parametrize('mutation',['mapping','mask','jsonl','source','config'])
def test_output_rejects_corruption(tmp_path,mutation):
    root=source(tmp_path); out=tmp_path/'cache'; build(root,out)
    identity_path=out/'chunk-identity.json'; r=json.loads(identity_path.read_text())
    if mutation=='mapping': r['durations']=[5,3,1]
    if mutation=='config': r['config']['frameskip']=3
    if mutation=='jsonl': (out/'chunk-rows.jsonl').write_text('{}\n')
    if mutation=='source': (root/'trace.jsonl').write_text('{}\n')
    if mutation=='mask':
        with np.load(out/'chunk-arrays.npz') as z: data={k:z[k] for k in z.files if k!='model_valid'}
        np.savez(out/'chunk-arrays.npz',**data)
        r['outputs']['chunk-arrays.npz']=sha(out/'chunk-arrays.npz')
    identity_path.write_text(json.dumps(r))
    with pytest.raises(ValueError): validate(out)


def test_raw_capture_contract():
    from touhou_rl.collect_student_states import _raw_safety_snapshot
    info=dict(frame=10,stage=1,raw_player_position=[192,224],bullets_raw=dict(position=[],velocity=[],half=[]))
    r=_raw_safety_snapshot(info,(1,10),(1,10),(1,10))
    assert r['complete'] and r['bullets']['position']==[] and not r['world_complete']
    assert not _raw_safety_snapshot(info,(1,9),(1,10),(1,10))['complete']
    assert not _raw_safety_snapshot(dict(frame=10,stage=1),(1,10),(1,10),(1,10))['complete']
