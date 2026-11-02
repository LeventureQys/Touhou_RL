"""Causal tactical event audit. Raw calibration captures are NOT training data."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SCHEMA='v373-paired-events/1'
FORBIDDEN_INPUTS={'future_pixels','teacher_option','teacher_target','privileged_state','full_bullet_pool'}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def audit_event(event):
    errors=[]
    def reject(condition,message):
        if condition:
            errors.append(message)
    reject(not event.get('event_id'),'missing event_id')
    reject(not event.get('template_group'),'missing template_group')
    reject(event.get('split') not in ('train','validation','test'),'invalid split')
    inputs=event.get('inputs',{})
    reject(bool(FORBIDDEN_INPUTS & set(inputs)),'privileged/future information in deployment inputs')
    reject(bool(set(inputs)-{'pixels','history'}),'undeclared deployment input channels')
    reject(not {'pixels','history'} <= set(inputs),'pixels/history physically separate inputs required')
    for name in ('pixels','history'):
        asset=inputs.get(name)
        reject(not isinstance(asset,dict) or not {'path','sha256'}<=set(asset),f'{name}: content-addressed asset required')
    labels=event.get('labels')
    reject(not isinstance(labels,dict) or not {'path','sha256'}<=set(labels),'content-addressed label section required')
    if isinstance(labels,dict):
        input_paths={a.get('path') for a in inputs.values() if isinstance(a,dict)}
        reject(labels.get('path') in input_paths,'input and label assets must be physically separate')
    rows=event.get('steps',[])
    reject(not rows,'empty event')
    previous=None
    for i,row in enumerate(rows):
        frame=row.get('frame')
        reject(type(frame) is not int or frame<0,f'{i}: invalid frame')
        if type(frame) is not int:
            continue
        if previous is not None:
            reject(frame<=previous,f'{i}: nonincreasing clock; split/reset event required')
        previous=frame
        interval=row.get('capture_interval',[])
        reject(len(interval)!=2 or interval!=[frame,frame],f'{i}: capture not proven same frame')
        reject(row.get('alignment_status')!='verified',f'{i}: alignment not verified')
        reject(not row.get('pool_complete',False),f'{i}: incomplete hazard pool')
        observed=row.get('history_observed_through',frame)
        reject(type(observed) is not int or observed>frame,f'{i}: noncausal history')
        action=row.get('previous_actual_candidate')
        reject(action is not None and (type(action) is not int or not 0<=action<18),f'{i}: invalid actual candidate')
        effective=row.get('previous_effective_frame')
        reject(effective is not None and (type(effective) is not int or effective>frame),f'{i}: unexecuted future action in history')
        reject(action is not None and effective is None,f'{i}: execution not acknowledged')
        hold=row.get('previous_actual_hold')
        reject(action is not None and (type(hold) is not int or hold<1),f'{i}: actual hold required')
    return errors


def audit_manifest(path):
    path=Path(path)
    obj=json.loads(path.read_text(encoding='utf-8'))
    errors=[]
    if obj.get('schema')!=SCHEMA:
        errors.append('schema mismatch')
    groups={}
    pixel_splits={}
    ids=set()
    count=0
    for entry in obj.get('events',[]):
        source=(path.parent/entry['path']).resolve()
        if not source.is_relative_to(path.parent.resolve()):
            errors.append(f'event escapes dataset root: {entry["path"]}')
            continue
        if not source.is_file() or file_sha(source)!=entry.get('sha256'):
            errors.append(f'event content mismatch: {entry.get("path")}')
            continue
        event=json.loads(source.read_text(encoding='utf-8'))
        count+=1
        errors.extend(f'{entry["path"]}: {e}' for e in audit_event(event))
        assets=list(event.get('inputs',{}).values())+[event.get('labels')]
        resolved_inputs=set()
        label_path=None
        for index,asset in enumerate(assets):
            if not isinstance(asset,dict) or not {'path','sha256'}<=set(asset):
                continue
            target=(source.parent/asset['path']).resolve()
            if not target.is_relative_to(path.parent.resolve()):
                errors.append(f'asset escapes dataset root: {asset["path"]}')
                continue
            if not target.is_file() or file_sha(target)!=asset['sha256']:
                errors.append(f'asset content mismatch: {asset["path"]}')
            if index==len(assets)-1:
                label_path=target
            else:
                resolved_inputs.add(target)
        if label_path in resolved_inputs:
            errors.append('resolved input/label asset alias')
        eid=event.get('event_id')
        if eid in ids:
            errors.append('duplicate event_id')
        ids.add(eid)
        group=event.get('template_group')
        split=event.get('split')
        pixel=event.get('inputs',{}).get('pixels')
        pixel_sha=pixel.get('sha256') if isinstance(pixel,dict) else None
        if pixel_sha:
            if pixel_sha in pixel_splits and pixel_splits[pixel_sha]!=split:
                errors.append('duplicate pixel asset across splits')
            pixel_splits[pixel_sha]=split
        if group in groups and groups[group]!=split:
            errors.append(f'template leakage: {group}')
        groups[group]=split
    if count==0:
        errors.append('no audited events')
    if set(groups.values())!={'train','validation','test'}:
        errors.append('three nonempty grouped splits required')
    return {'schema':'v373-data-audit/1','manifest_sha256':file_sha(path),
        'audited_events':count,'passed':not errors,'errors':errors,
        'warning':'Structural audit does not independently certify teacher or render freshness.'}
