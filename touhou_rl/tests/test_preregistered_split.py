import json
import hashlib
import pytest
from touhou_rl.preregistered_split import validate_protocol, audit_trace_splits, protocol_digest, SCHEMA

H='a'*64

def protocol(**changes):
    p={'schema':SCHEMA,'checkpoint_sha256':H,'teacher_source_sha256':H,'environment_digest':H,
       'frameskip':2,'episodes':{'train':[1,2],'dev':[3],'test':[4]},'min_samples_each':1,
       'purge_windows':{'future_decisions':1,'history_decisions':1,'chunk_decisions':0},
       'max_attempts':1,'thresholds':{'danger_min':0.0}}
    p.update(changes); return p

def snap(frame, stage=1):
    return {'schema':'v200-raw-safety-snapshot/1','frame':frame,'stage':stage,'complete':True,'player_position':[0,0],'bullets':{'position':[],'velocity':[],'half':[]}}

def rows(ep, n=4, start=0, rollout='new-round'):
    return [dict(episode=ep,decision=i,rollout_id=rollout,sample_index=start+i,frame=i*2,stage=1,
                 raw_safety_snapshot=snap(i*2),aligned=True,fresh_pixels=True) for i in range(n)]

def write_trace(tmp_path, data, name='trace.jsonl'):
    p=tmp_path/name; p.write_text(''.join(json.dumps(r)+'\n' for r in data), encoding='utf-8'); return p

def test_protocol_digest_and_strict_validation():
    p=protocol(); assert len(protocol_digest(p))==64; assert validate_protocol(p)==p
    for bad in ({'x':1}, {'schema':SCHEMA}):
        with pytest.raises(ValueError): validate_protocol(bad)
    with pytest.raises(ValueError): validate_protocol(protocol(episodes={'train':[1,2],'dev':[2],'test':[4]}))
    with pytest.raises(ValueError): validate_protocol(protocol(checkpoint_sha256='A'*64))
    with pytest.raises(ValueError): validate_protocol(protocol(purge_windows={'future_decisions':-1,'history_decisions':0,'chunk_decisions':0}))

def test_split_audit_and_purge(tmp_path):
    path=write_trace(tmp_path, rows(1,4))
    out=audit_trace_splits([path], protocol(), 'train')
    assert out['schema']=='v200-preregistered-rollout-audit/1'
    assert out['decisions']==4 and out['aligned_samples']==4
    assert out['purged_samples']==2 and out['usable_samples']==2
    assert out['leakage'] is False and out['protocol_digest']==protocol_digest(protocol())

def test_order_duplicate_and_cross_split_rejected(tmp_path):
    p=protocol(); data=rows(1,3)
    bad=list(data); bad[1]=dict(bad[1], decision=0)
    with pytest.raises(ValueError, match='duplicate|order'):
        audit_trace_splits([write_trace(tmp_path,bad)],p,'train')
    with pytest.raises(ValueError, match='belongs'):
        audit_trace_splits([write_trace(tmp_path,rows(3,2),'dev.jsonl')],p,'train')
    bad2=rows(1,2); bad2[1]['raw_safety_snapshot']['frame']=99
    with pytest.raises(ValueError, match='mismatch'):
        audit_trace_splits([write_trace(tmp_path,bad2,'mismatch.jsonl')],p,'train')

def test_r1d_is_not_formal_holdout(tmp_path):
    data=rows(1,2,rollout='r1d')
    with pytest.raises(ValueError, match='r1d'):
        audit_trace_splits([write_trace(tmp_path,data)],protocol(),'train')

def test_internal_stage_death_gap_and_missing_purged(tmp_path):
    for kind in ('stage','death','gap','missing'):
        data=rows(1,7)
        if kind=='stage':
            for r in data[3:]:r['stage']=2;r['raw_safety_snapshot']['stage']=2
        if kind=='death':data[3]['deaths_before']=0;data[3]['deaths_after']=1
        if kind=='gap':
            for r in data[3:]:r['frame']+=3;r['raw_safety_snapshot']['frame']+=3
        if kind=='missing':data[3]['aligned']=False;data[3]['sample_index']=None
        out=audit_trace_splits([write_trace(tmp_path,data,kind+'.jsonl')],protocol(),'train')
        assert out['purged_samples']>2


def test_protocol_rejects_unknown_and_bad_threshold():
    with pytest.raises(ValueError): validate_protocol(protocol(extra=True))
    with pytest.raises(ValueError): validate_protocol(protocol(thresholds={'x':True}))
