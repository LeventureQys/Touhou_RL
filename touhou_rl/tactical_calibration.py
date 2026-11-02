"""Analyze actual readback intervals; do not infer precise latency from first match."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
from . import spec

CONTROL=spec.BTN_UP|spec.BTN_DOWN|spec.BTN_LEFT|spec.BTN_RIGHT|spec.BTN_FOCUS|spec.BTN_SHOOT|spec.BTN_BOMB
KEYS={'up':spec.BTN_UP,'down':spec.BTN_DOWN,'left':spec.BTN_LEFT,'right':spec.BTN_RIGHT,
      'shift':spec.BTN_FOCUS,'z':spec.BTN_SHOOT,'x':spec.BTN_BOMB}


def summarize_trial(trial):
    expected=sum(KEYS[k] for k in trial['keys'])
    records=[trial['send_after']]+trial['polls']
    channel='gui_input_current'
    if any(channel not in r for r in records):
        return {'window':trial['window'],'category':trial['category'],
            'input_status':'missing_consumed_input_channel','effective_frame_interval':None,
            'send_frame_interval':trial['send_frame_interval'],
            'stable_readback_deltas':[],'rejected_transitions':{},
            'warning':'legacy input_mask reads padding; do not infer actuation from it'}
    unique=[]
    for r in records:
        if not unique or unique[-1]['gameplay_frame']!=r['gameplay_frame']:
            unique.append(r)
        elif unique[-1]!=r:
            # Same frame count can bracket intra-frame memory mutation; preserve uncertainty.
            unique.append(r)
    first_match=next((i for i,r in enumerate(unique) if r[channel] & CONTROL == expected),None)
    interval=None
    status='unobserved'
    if first_match is not None:
        current=unique[first_match]['gameplay_frame']
        if first_match==0:
            interval=[None,current]
            status='already_active_no_lower_bound'
        else:
            interval=[unique[first_match-1]['gameplay_frame'],current]
            status='bracketed'
    deltas=[]
    rejected=Counter()
    for a,b in zip(unique,unique[1:]):
        dt=b['gameplay_frame']-a['gameplay_frame']
        if dt!=1:
            rejected['non_unit_frame']+=1
            continue
        if a[channel]&CONTROL!=expected or b[channel]&CONTROL!=expected:
            rejected['transition_or_wrong_input']+=1
            continue
        if a['player_state']!=0 or b['player_state']!=0:
            rejected['non_alive_player_state']+=1
            continue
        if a['deaths']!=b['deaths']:
            rejected['death']+=1
            continue
        deltas.append({'dx':b['player_x']-a['player_x'],'dy':b['player_y']-a['player_y'],
                       'frame':b['gameplay_frame']})
    return {'window':trial['window'],'category':trial['category'],
        'input_status':status,'effective_frame_interval':interval,
        'send_frame_interval':trial['send_frame_interval'],
        'stable_readback_deltas':deltas,'rejected_transitions':dict(rejected),
        'warning':'memory input latch and render/update phase not independently synchronized'}


def analyze(report):
    rows=[summarize_trial(t) for t in report['trials']]
    summary={}
    for category in sorted({r['category'] for r in rows}):
        group=[r for r in rows if r['category']==category]
        values=[float(np.hypot(d['dx'],d['dy'])) for r in group for d in r['stable_readback_deltas']]
        summary[category]={'trials':len(group),'stable_unit_frame_deltas':len(values),
            'speed_median':float(np.median(values)) if values else None,
            'speed_p05':float(np.quantile(values,.05)) if values else None,
            'speed_p95':float(np.quantile(values,.95)) if values else None,
            'speed_min':min(values) if values else None,'speed_max':max(values) if values else None}
    return {'schema':'v373-input-analysis/1','source_schema':report['schema'],
        'summary':summary,'input_status_counts':dict(Counter(r['input_status'] for r in rows)),
        'trials':rows,'P0_passed':False,'remaining':['boundary tests',
            *([] if 'focus_diagonal' in summary else ['focus diagonal']),
            'within-frame update order','capture freshness','effective release intervals'],
        'warning':'Stable-readback subset is diagnostic, not a frozen calibrated dynamics model.'}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    report=analyze(json.loads(args.source.read_text(encoding='utf-8')))
    with args.out.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2))


if __name__=='__main__':
    main()
