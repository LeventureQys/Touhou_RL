"""Copy immutable synthetic assets into a new mixed dataset; never move source."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
from touhou_rl.tactical_data import file_sha


def merge(original,augmentation,out):
    original,augmentation,out=map(Path,(original,augmentation,out))
    if out.exists():raise FileExistsError('new mixed diagnostic root required')
    base=json.loads(original.read_text(encoding='utf-8'));aug=json.loads(augmentation.read_text(encoding='utf-8'))
    schema='v373-synthetic-rendered-diagnostic/1'
    if base['schema']!=schema or aug['schema']!=schema or base.get('real_pixel_training') or aug.get('real_pixel_training'):
        raise ValueError('explicit synthetic records only')
    train_groups={e['group'] for e in base['events'] if e['split']=='train'}
    val_groups={e['group'] for e in base['events'] if e['split']=='validation'}
    if train_groups&val_groups:raise ValueError('base leakage')
    for e in aug['events']:
        if e['split']!='train' or e['group'] not in train_groups or e.get('parent_group')!=e['group'] or not e.get('augmentation'):
            raise ValueError('augmentation must belong to original train template')
    out.mkdir(parents=True);entries=[];ids=set()
    for source,items in ((original,base['events']),(augmentation,aug['events'])):
        for old in items:
            e=copy.deepcopy(old)
            if e['event_id'] in ids:raise ValueError('duplicate event id')
            ids.add(e['event_id'])
            folder=out/f'event-{len(entries):03d}';folder.mkdir()
            for key in ('pixels','history','labels','raw','rollout'):
                if key not in e:continue
                ref=e[key];p=(source.parent/ref['path']).resolve()
                if not p.is_relative_to(source.parent.resolve()) or file_sha(p)!=ref['sha256']:
                    raise ValueError('source asset drift')
                dest=folder/(key+p.suffix);shutil.copyfile(p,dest)
                if file_sha(dest)!=ref['sha256']:raise ValueError('copied asset mismatch')
                ref['path']=dest.relative_to(out).as_posix()
            entries.append(e)
    result={'schema':schema,'events':entries,'real_pixel_training':False,'formal_gate':'NOT_OPENED',
        'sources':[{'path':str(p),'sha256':file_sha(p)} for p in (original,augmentation)],
        'augmentation_round':1,'validation_unchanged':True}
    with (out/'records.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original','augmentation','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(len(merge(a.original,a.augmentation,a.out)['events']))
