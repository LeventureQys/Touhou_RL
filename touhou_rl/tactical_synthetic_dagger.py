"""Round13 train-only synthetic DAgger data. No training, game, gate or shield."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from . import vision
from .tactical_synthetic_pixels import (SCHEMA, SIZES, ACTION_MAP, selected_specs,
    render_state, history_and_labels, sha, write_json, renderer_source_hash)
from .tactical_synthetic_closed_loop import (ROOT, RECORDS, HEADS, BASELINE,
    FrozenNetworks, run_event)
from .tactical_failure_forensics import independent_replay, logical_step, replay_actual
from .tactical_teacher_v2 import TacticalTeacherV2, hashes
from .tactical_teacher_validation import make_world, advance
from .tactical_scenarios import initial_state

OUT = ROOT/'synthetic-dagger-round13'


def train_specs():
    rows = [r for r in selected_specs() if r['split'] == 'train']
    counts = Counter(r['spec']['family'] for r in rows)
    if len(rows) != 8 or any(n > 4 for n in counts.values()):
        raise ValueError('original eight train events only; maximum four per family')
    if any(r['spec']['split'] != 'development' for r in rows):
        raise ValueError('validation forbidden')
    return rows


def preregister(out=OUT):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    # Read R10 index only. No validation pixels, labels, trajectories or restore points.
    old = json.loads(RECORDS.read_text(encoding='utf-8'))
    entries = {e['event_id']: e for e in old['events'] if e['split'] == 'train'}
    events = []
    for row in train_specs():
        spec = row['spec']; e = entries[spec['event_id']]
        digest = hashlib.sha256(json.dumps(asdict(make_world(spec)), sort_keys=True).encode()).hexdigest()
        if e['group'] != spec['template_group'] or e['scenario_sha256'] != digest:
            raise ValueError('original train identity mismatch')
        events.append(dict(spec=spec, split='train', scenario_sha256=digest))
    files = [Path(__file__), Path('touhou_rl/tactical_synthetic_pixels.py'),
        Path('touhou_rl/tactical_synthetic_closed_loop.py'),
        Path('touhou_rl/tactical_failure_forensics.py'),
        Path('touhou_rl/tactical_teacher_v2.py'), Path('touhou_rl/tactical_teacher_validation.py'),
        Path('touhou_rl/tactical_scenarios.py'), Path('touhou_rl/tactical_policy.py'),
        HEADS, BASELINE, RECORDS]
    value = dict(schema='v373-synthetic-dagger-prereg/1', events=events,
        identities={str(p):sha(p) for p in files}, teacher_source=hashes(),
        renderer_sha256=renderer_source_hash(), collection_checkpoint_sha=sha(HEADS),
        arm='N1', condition='horizontal_masked', collection_events=8, maximum_per_family=4,
        checkpoint_selection='fixed R10 run-s7-n1-200, no tuning or shield',
        rescue_rule='earliest trusted unhit even decision state in [first N1 reported collision-16, collision); no search over rescue points',
        verification='complete fixed N1 prefix + teacherV2 suffix; independent logical-frame replay from initial state',
        supervision='only independently feasible rescue suffix; prefix action_valid=False; failure all invalid',
        history='actual preceding candidate including N1 prefix; option BOS3 always; depth3 actual causal pixels',
        records_path='records/records.json', refs_relative_to='records', augmentation=True,
        validation_asset_loads=0, forensic_restore_point_loads=0, training_steps=0,
        game_execution=False, formal_labels=False, formal_gate='NOT_OPENED',
        limitations=['Privileged synthetic rescue data only; not a network score or formal P5.',
            'No collision means no rescue and no new supervised rows; failed rescue retained without labels.',
            'F1 hidden onset and F3 hidden goal/window remain unchanged.'])
    write_json(out/'preregistration.json', value)
    return value


def rescue_event(s, trace):
    actions, states, source_verification = replay_actual(s, trace)
    collision = trace['first_collision_frame']
    indices = [] if collision is None or not s.model_known else [
        i for i, st in enumerate(states[:-1])
        if max(0, collision-16) <= st.frame < collision and not st.hit]
    index = indices[0] if indices else None
    decisions = []
    mixed = list(actions)
    if index is not None:
        state = states[index]; mixed = list(actions[:index]); teacher = TacticalTeacherV2()
        while state.frame < s.horizon:
            d = teacher.decide(s, state)
            decisions.append(dict(frame=state.frame, **asdict(d)))
            mixed.append(d.action); state = advance(s, state, d.action)
    independent, final = independent_replay(s, mixed)
    success = index is not None and independent['status'] == 'feasible'
    return dict(actions=mixed, prefix_actions=actions if index is None else actions[:index],
        suffix_actions=[] if index is None else mixed[index:], decisions=decisions,
        rescue_start_frame=None if index is None else states[index].frame,
        prefix_rows=80 if index is None else index, rescued=success,
        rescue_attempted=index is not None, first_N1_collision_frame=collision,
        first_N1_collision_logical=source_verification['first_collision_logical'],
        original_N1_feasible=trace['feasible'], source_verification=source_verification,
        independent_verification=independent, network_score=False,
        privileged_reference_only=True, final_state=asdict(final))


def mixed_history_labels(actions, prefix_rows, verified, candidates=None):
    h, lab = history_and_labels(actions, False)
    # Preserve actual N1 candidates (including focus-hold candidate9, not action0 alias0).
    c = torch.tensor(candidates if candidates is not None else [ACTION_MAP[a] for a in actions])
    if len(c) != len(actions): raise ValueError('candidate length mismatch')
    h['previous_candidate'][1:] = c[:-1]
    if verified:
        lab['action_valid'][prefix_rows:] = True
        ids = torch.arange(prefix_rows, len(actions))
        lab['action_positive'][ids, c[prefix_rows:]] = True
    return h, lab


def export_event(data, s, rescue, trace, prereg):
    event_id = s.event_id+'-dagger-r13'; event = data/event_id; event.mkdir()
    actions = rescue['actions']
    # Independent actual one-frame states, avoiding interpolated hit/bullet state.
    state = initial_state(s); states = [state]
    for action in actions:
        for _ in range(2):
            state = logical_step(s, state, action); states.append(state)
    native = np.stack([render_state(s, st) for st in states])
    np.savez_compressed(event/'raw.npz', rgb=native, frames=np.arange(161,dtype=np.int32))
    resized = {name:[] for name in SIZES}
    for f in range(0,160,2):
        for name, image in vision.extract_regions(native[f]).items():
            resized[name].append(vision.resize_area(image,*SIZES[name]))
    pixels = {}; ids = np.maximum(0,np.arange(80)[:,None]+np.array([-2,-1,0]))
    for name, frames in resized.items():
        arr = np.stack(frames)
        if name.startswith('hud_'):
            one = vision.regions_to_batch({name:arr[:1,None].repeat(3,axis=1)})[name]
            pixels[name] = one.expand(80,-1,-1,-1,-1)
        else: pixels[name] = vision.regions_to_batch({name:arr[ids]})[name]
    n = rescue['prefix_rows']
    candidates = [row['actual_candidate'] for row in trace['steps'][:n]] + [ACTION_MAP[a] for a in actions[n:]]
    history, labels = mixed_history_labels(actions,n,rescue['rescued'],candidates)
    for name, obj in [('pixels',pixels),('history',history),('labels',labels)]:
        torch.save(obj,event/(name+'.pt'))
    raw_meta = dict(scenario=asdict(s), actions=actions, candidates=candidates,
        decisions=rescue['decisions'], states=[asdict(st) for st in states],
        verdict=rescue['independent_verification']['status'], teacher_source=hashes(),
        privileged_reference_only=True, rescue=rescue, augmentation=True,
        parent_event_id=s.event_id, parent_group=s.template_group,
        collection_checkpoint_sha=prereg['collection_checkpoint_sha'])
    write_json(event/'rollout.json',raw_meta)
    def ref(name):
        p=event/name; return dict(path=p.relative_to(data).as_posix(),sha256=sha(p))
    return dict(event_id=event_id, parent_event_id=s.event_id, group=s.template_group,
        parent_group=s.template_group, split='train', family=s.family, mirror=s.mirror,
        original_split=s.split, rows=80, logical_frames=160, teacher_verified=rescue['rescued'],
        verify_verdict=rescue['independent_verification']['status'], augmentation=True,
        collection_checkpoint_sha=prereg['collection_checkpoint_sha'],
        pixels=ref('pixels.pt'),history=ref('history.pt'),labels=ref('labels.pt'),
        raw=ref('raw.npz'),rollout=ref('rollout.json'),
        scenario_sha256=hashlib.sha256(json.dumps(asdict(s),sort_keys=True).encode()).hexdigest(),
        rescue_start_frame=rescue['rescue_start_frame'], first_N1_collision_frame=rescue['first_N1_collision_frame'],
        first_N1_collision_logical=rescue['first_N1_collision_logical'],
        prefix_rows=n, supervised_rows=int(labels['action_valid'].sum()),rescue_attempted=rescue['rescue_attempted'])


def collect(out=OUT, device='cuda'):
    out=Path(out); p=json.loads((out/'preregistration.json').read_text(encoding='utf-8'))
    for f, digest in p['identities'].items():
        if sha(f)!=digest: raise ValueError('preregistered identity changed: '+f)
    if p['events'] != [dict(spec=r['spec'],split='train',scenario_sha256=hashlib.sha256(
            json.dumps(asdict(make_world(r['spec'])),sort_keys=True).encode()).hexdigest()) for r in train_specs()]:
        raise ValueError('train-only preregistration mismatch')
    data=out/'records'; data.mkdir(exist_ok=False)
    source=out/'source-N1'; source.mkdir(exist_ok=False)
    torch.set_num_threads(4); network=FrozenNetworks(BASELINE,HEADS,device)
    entries=[]; rows=[]
    for item in p['events']:
        s=make_world(item['spec'])
        trace=run_event(s,'N1','horizontal_masked',network,None,source/s.event_id)
        rescue=rescue_event(s,trace)
        write_json(source/s.event_id/'rescue.json',rescue)
        entry=export_event(data,s,rescue,trace,p); entries.append(entry)
        rows.append(dict(event_id=s.event_id,family=s.family,original_N1_feasible=trace['feasible'],
            original_N1_survived=trace['survived'],first_N1_collision_frame=trace['first_collision_frame'],
            first_N1_collision_logical=rescue['first_N1_collision_logical'],
            rescue_start_frame=rescue['rescue_start_frame'],rescue_attempted=rescue['rescue_attempted'],
            rescue_success=rescue['rescued'],supervised_rows=entry['supervised_rows'],network_score=False))
        print(s.event_id,'N1',trace['feasible'],'rescue',rescue['rescued'],'labels',entry['supervised_rows'],flush=True)
    failures=[dict(event_id=e['event_id'],verdict=e['verify_verdict']) for e in entries if not e['teacher_verified']]
    records=dict(schema=SCHEMA,events=entries,
        preregistration=dict(path='../preregistration.json',sha256=sha(out/'preregistration.json')),
        teacher_source=hashes(),renderer_sha256=renderer_source_hash(),region_sizes=SIZES,depth=3,
        real_pixel_training=False,formal_manifest_compatible=False,formal_gate='NOT_OPENED',
        sealed_R8_test_execution=0,R9_confirmation_execution=0,failures=failures,
        limitations=p['limitations'],history_semantics=p['history'],augmentation=True,
        collection_checkpoint_sha=p['collection_checkpoint_sha'],train_only=True,
        validation_asset_loads=0,forensic_restore_point_loads=0)
    write_json(data/'records.json',records);write_json(data/'failures.json',failures)
    summary=dict(schema='v373-synthetic-dagger-collection/1',events=rows,collection_events=len(rows),
        original_N1_feasible=sum(r['original_N1_feasible'] for r in rows),
        rescue_attempts=sum(r['rescue_attempted'] for r in rows),rescue_successes=sum(r['rescue_success'] for r in rows),
        supervised_rows=sum(r['supervised_rows'] for r in rows),training_steps=0,game_execution=False,
        validation_asset_loads=0,forensic_restore_point_loads=0,formal_gate='NOT_OPENED',
        source_identities_unchanged=all(sha(f)==h for f,h in p['identities'].items()))
    write_json(out/'collection_summary.json',summary)
    write_json(out/'sha256.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preregister','collect'])
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--device',default='cuda')
    a=parser.parse_args();print(json.dumps(preregister(a.out) if a.command=='preregister' else collect(a.out,a.device),indent=2))

if __name__=='__main__': main()
