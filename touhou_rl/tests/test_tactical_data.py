import copy
from touhou_rl.tactical_data import audit_event


def valid_event():
    return {'event_id':'e','template_group':'g','split':'train',
        'inputs':{'pixels':{'path':'input/pixels.npz','sha256':'pixels'},
                  'history':{'path':'input/history.npz','sha256':'history'}},
        'labels':{'path':'labels/targets.npz','sha256':'labels'},
        'steps':[{'frame':10,'capture_interval':[10,10],'alignment_status':'verified',
                  'pool_complete':True,'previous_actual_candidate':2,
                  'previous_effective_frame':8,'previous_actual_hold':2}]}


def test_structural_event_valid_not_capability_certificate():
    assert not audit_event(valid_event())


def test_unknown_alignment_cannot_train_exact_geometry():
    event=valid_event(); event['steps'][0]['alignment_status']='same_bracket_frame_freshness_unverified'
    assert any('alignment not verified' in e for e in audit_event(event))


def test_privileged_or_future_input_rejected():
    event=valid_event(); event['inputs']['teacher_target']='labels/target.npy'
    assert any('privileged' in e for e in audit_event(event))
    event=valid_event(); event['steps'][0]['history_observed_through']=12
    assert any('noncausal' in e for e in audit_event(event))


def test_unexecuted_action_and_missing_hold_rejected():
    event=valid_event(); event['steps'][0]['previous_effective_frame']=11
    assert any('unexecuted' in e for e in audit_event(event))
    event=valid_event(); event['steps'][0]['previous_actual_hold']=None
    assert any('actual hold' in e for e in audit_event(event))


def test_reset_clock_must_split_event():
    event=valid_event(); event['steps'].append(copy.deepcopy(event['steps'][0]))
    assert any('nonincreasing' in e for e in audit_event(event))


def test_manifest_sha_and_group_leakage(tmp_path):
    import json
    from touhou_rl.tactical_data import audit_manifest,file_sha,SCHEMA
    entries=[]
    for split in ('train','validation','test'):
        event=valid_event(); event.update(event_id=split,split=split,template_group=split)
        for name,asset in list(event['inputs'].items())+[('labels',event['labels'])]:
            target=tmp_path/f'{split}-{name}.bin'; target.write_bytes(f'{split}-{name}'.encode())
            asset.update(path=target.name,sha256=file_sha(target))
        source=tmp_path/f'{split}.json'; source.write_text(json.dumps(event),encoding='utf-8')
        entries.append({'path':source.name,'sha256':file_sha(source)})
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema':SCHEMA,'events':entries}),encoding='utf-8')
    assert audit_manifest(manifest)['passed']
    source=tmp_path/'test.json'
    event=json.loads(source.read_text()); event['template_group']='train'
    source.write_text(json.dumps(event),encoding='utf-8')
    assert any('content mismatch' in e for e in audit_manifest(manifest)['errors'])
    entries[-1]['sha256']=file_sha(source)
    manifest.write_text(json.dumps({'schema':SCHEMA,'events':entries}),encoding='utf-8')
    assert any('template leakage' in e for e in audit_manifest(manifest)['errors'])
    event['template_group']='test'
    event['inputs']['pixels']={'path':'train-pixels.bin','sha256':file_sha(tmp_path/'train-pixels.bin')}
    source.write_text(json.dumps(event),encoding='utf-8');entries[-1]['sha256']=file_sha(source)
    manifest.write_text(json.dumps({'schema':SCHEMA,'events':entries}),encoding='utf-8')
    assert any('duplicate pixel asset' in e for e in audit_manifest(manifest)['errors'])


def test_unhashed_or_aliased_assets_rejected():
    event=valid_event(); event['inputs']['pixels']='unhashed.npz'
    assert any('content-addressed asset' in e for e in audit_event(event))
    event=valid_event(); event['labels']['path']=event['inputs']['pixels']['path']
    assert any('physically separate' in e for e in audit_event(event))
