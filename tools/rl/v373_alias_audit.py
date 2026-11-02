"""Exact observational alias audit, not approximate feature similarity."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import torch
from touhou_rl.tactical_data import file_sha


def tensor_digest(mapping,row):
    h=hashlib.sha256()
    for key,value in sorted(mapping.items()):
        t=value[row].detach().cpu().contiguous()
        h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def conflict_summary(groups):
    conflicts=[];total=0;max_correct=0
    for key,rows in groups.items():
        counts=Counter(r['action'] for r in rows);total+=len(rows);max_correct+=max(counts.values())
        if len(counts)>1:
            conflicts.append({'digest':key,'rows':rows,'label_counts':dict(counts)})
    return {'rows':total,'unique_inputs':len(groups),'conflicting_groups':len(conflicts),
        'rows_in_conflicting_groups':sum(len(c['rows']) for c in conflicts),
        'deterministic_exact_input_upper_bound':max_correct/total if total else None,
        'conflicts':conflicts}


def run(records,out):
    records=Path(records);out=Path(out)
    if out.exists():raise FileExistsError('new alias audit required')
    obj=json.loads(records.read_text(encoding='utf-8'))
    if obj['schema']!='v373-synthetic-rendered-diagnostic/1':raise ValueError('synthetic only')
    groups={name:defaultdict(list) for name in ('pixel_stack','pixel_stack_plus_actual_history','entire_causal_history')}
    by_split={s:defaultdict(list) for s in ('train','validation')}
    for entry in obj['events']:
        data={}
        for key in ('pixels','history','labels'):
            ref=entry[key];p=records.parent/ref['path']
            if file_sha(p)!=ref['sha256']:raise ValueError('asset mismatch')
            data[key]=torch.load(p,weights_only=True)
        causal=hashlib.sha256()
        for i in range(entry['rows']):
            pixel=tensor_digest(data['pixels'],i);history=tensor_digest(data['history'],i)
            combined=hashlib.sha256((pixel+history).encode()).hexdigest()
            causal.update(combined.encode())
            label=data['labels']
            if not bool(label['action_valid'][i]):continue
            positive=label['action_positive'][i]
            if int(positive.sum())!=1:raise ValueError('single action diagnostic only')
            row={'event_id':entry['event_id'],'family':entry['family'],'split':entry['split'],
                 'frame':i*2,'action':int(positive.long().argmax())}
            groups['pixel_stack'][pixel].append(row)
            groups['pixel_stack_plus_actual_history'][combined].append(row)
            groups['entire_causal_history'][causal.hexdigest()].append(row)
            by_split[entry['split']][causal.hexdigest()].append(row)
    result={'schema':'v373-synthetic-observation-alias/1','records_sha256':file_sha(records),
        'global':{k:conflict_summary(v) for k,v in groups.items()},
        'causal_by_split':{k:conflict_summary(v) for k,v in by_split.items()},
        'warning':'Different teacher actions may both be feasible. Conflicting single-route labels show observational alias, not mutually exclusive safety necessity. Upper bound only for exact given inputs and labels, not optimal gameplay.'}
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:({n:{a:b for a,b in x.items() if a!='conflicts'} for n,x in v.items()} if k in ('global','causal_by_split') else v) for k,v in result.items()},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.records,a.out)
