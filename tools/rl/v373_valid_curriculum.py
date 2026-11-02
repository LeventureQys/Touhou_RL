"""Round20 pre-registered teacher distillation -> grouped-holdout closed loop; research only.

The data is the R18-valid curriculum (28 worlds, 14 groups of base+mirror) joined
with the frozen R19 train/validation split. Only the 14 train events are rendered
and labelled, by the unmodified TacticalTeacherV2 driving the real physics; the 14
validation events are touched by evaluation only. One shared encode pass, then two
loss arms (A = unchanged action loss, B = the same loss plus 1.0*KL(base||policy))
times three seeds, 200 fixed updates each, final step, no selection. All six
checkpoints are SHA-frozen before any closed loop runs. No formal gate, no
deployable checkpoint, no game execution, no official label.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tools.rl import v373_seed_confirm as r16
from tools.rl import v373_world_filter as r18
from touhou_rl import vision
from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_feature_cache import _fp32_mode
from touhou_rl.tactical_policy import TacticalConfig, TacticalPolicy, tactical_policy_loss
from touhou_rl.tactical_synthetic_closed_loop import HORIZONTAL, executed_history, run_event
from touhou_rl.tactical_synthetic_pixels import (ACTION_MAP, SIZES, full_states,
                                                 history_and_labels, render_state)
from touhou_rl.tactical_teacher_v2 import hashes as teacher_hashes
from touhou_rl.tactical_teacher_v2 import rollout as teacher_rollout
from touhou_rl.tactical_teacher_validation import make_world, verify
from touhou_rl.vmodels import load_vision_checkpoint

ROOT = Path('Document/Update/Alpha/v3.7.3/evidence')
OUT = ROOT / 'validcurriculum-round20'
R18 = ROOT / 'worldfilter-round18'
R19 = ROOT / 'validteacher-round19'
MANIFEST = R18 / 'run/manifest.json'
SPLIT = R19 / 'run/split.json'
REPORTS = Path('Document/Update/Alpha/v3.7.3/reports')
BASELINE = Path('third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt')

PREREG_SCHEMA = 'v373-valid-curriculum-prereg/1'
DATA_SCHEMA = 'v373-valid-curriculum-data/1'
EVENT_SCHEMA = 'v373-valid-curriculum-event/1'
SCHEMA = 'v373-valid-curriculum-heads/1'
RESULT_SCHEMA = 'v373-valid-curriculum-result/1'
SPLIT_SCHEMA = 'v373-valid-teacher-split/1'
MANIFEST_SCHEMA = 'v373-world-filter-run/1'

ARMS = ('A', 'B')
SEEDS = (7, 17, 27)
CONTROL, ANCHOR = ARMS
UPDATES = 200
LR = .001
CLIP = 1.
SEQUENCE = 80
BATCH = 1
ANCHOR_WEIGHT = 1.
HORIZON = 160
DECISIONS = 80
CHUNK = 16
FAMILIES = ('all', 'F1', 'F2', 'F3')
FAMILY_KEYS = ('F1', 'F2', 'F3')
CONDITION = 'horizontal_masked'
TRAIN_SPLIT, VALIDATION_SPLIT = 'train', 'validation'
SUPPORTED = tuple(sorted(HORIZONTAL))
SIGNATURE_FIELDS = ['family', 'canonical start_x', 'coverage', 'gap', 'wall_frame', 'fire_interval', 'window',
                    'threshold', 'goal', 'origin', 'burst', 'bullet_speed', 'internal_gap']
EXPECTED = dict(events=28, groups=14, per_family={'F1': 12, 'F2': 4, 'F3': 12},
                groups_per_family={'F1': 6, 'F2': 2, 'F3': 6},
                train_events=14, train_groups=7, validation_events=14, validation_groups=7,
                train_groups_per_family={'F1': 3, 'F2': 1, 'F3': 3},
                validation_groups_per_family={'F1': 3, 'F2': 1, 'F3': 3},
                train_events_per_family={'F1': 6, 'F2': 2, 'F3': 6},
                validation_events_per_family={'F1': 6, 'F2': 2, 'F3': 6},
                checkpoints=6, closed_loop_files=2 + 2 * len(SEEDS) * len(ARMS),
                supervised_events=14, excluded_events=0, baseline_rows=2 * 14 * 6)
HUD_NAMES = ('hud_score', 'hud_status')
STEP_FIELDS = ('frame', 'hidden_reset', 'raw_unmasked_candidate', 'raw_unsupported', 'candidate',
               'actual_candidate', 'actual_action', 'actual_hold', 'x_before', 'x_after',
               'invalid_action', 'fallback', 'first_collision_frame', 'hit')

FORMULA = ('Arm B only, weight fixed at 1.0 before the run. For the full-80 batch let a=policy_logits [1,80,18], '
           'b=the frozen baseline policy_logits of the same encoded rows (detached, no gradient into the backbone), '
           'V={i: action_valid_i}, N=|V|, KL_anchor=(1/N)*sum_{i in V} sum_c softmax(b_i)_c*'
           '(log softmax(b_i)_c-log softmax(a_i)_c), i.e. KL(base||policy) in nats averaged over every action_valid '
           'row (F.kl_div(log_target=True, reduction="batchmean") on the masked rows). L_total(B)=L_original+1.0*'
           'KL_anchor, L_total(A)=L_original. L_original is the unchanged tactical_policy_loss (action 1.0, option .2, '
           'target .5, termination .2, slack .1, local 1.0); the action-only labels leave option/target/termination/'
           'slack/local empty so they contribute exactly 0 in both arms. The R17 functional form is reused unchanged; '
           'the weight is never tuned on any result.')

STORAGE = dict(dtype='uint8 (the renderer output verbatim, never cast to float on disk)',
               container='npz deflate for the pixel regions, torch.save (xb) for history/labels',
               scale='uint8 -> float32 /255 happens only at load through vision.regions_to_batch',
               hud='hud_score/hud_status are byte-identical across all 80 decisions (asserted per event); stored once '
                   'as [1,3,H,W,C] and expanded to 80 rows at load, so the values are identical while the 80 '
                   'redundant copies are not written',
               why='a float32 copy of the same content is 4x larger and adds nothing: every consumer divides by 255 '
                   'immediately')


# ---------------------------------------------------------------- plumbing
def new_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def write_torch(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)


def write_npz(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def name_of(arm, seed):
    return f'{arm}-s{seed}'


def all_names():
    return [name_of(arm, seed) for arm in ARMS for seed in SEEDS]


def depth_indices(rows=DECISIONS):
    """Depth-3 causal rows: decision i stacks resized frames i-2, i-1, i (repeat initial)."""
    return np.maximum(0, np.arange(rows)[:, None] + np.array([-2, -1, 0]))


def depth_frames(rows=DECISIONS):
    """The logical frames actually stacked, [max(0,f-4), max(0,f-2), f] with f=2*i."""
    return [[max(0, 2 * i - 4), max(0, 2 * i - 2), 2 * i] for i in range(rows)]


# ---------------------------------------------------------------- frozen sources
def load_worlds():
    """Read-only load of the R18 manifest, with its own verdicts re-asserted."""
    raw = MANIFEST.read_bytes()
    spec = json.loads(raw)
    if spec.get('schema') != MANIFEST_SCHEMA:
        raise ValueError('R18 manifest schema')
    if not spec.get('development_only') or spec.get('test_split'):
        raise ValueError('development-only R18 set required')
    if spec.get('training_steps') or spec.get('teacher_labels'):
        raise ValueError('the R18 manifest must have no training and no teacher labels')
    worlds = spec['worlds']
    if len(worlds) != EXPECTED['events']:
        raise ValueError('R18 world count')
    if {f: sum(w['family'] == f for w in worlds) for f in FAMILY_KEYS} != EXPECTED['per_family']:
        raise ValueError('R18 family counts')
    if len({w['group'] for w in worlds}) != EXPECTED['groups']:
        raise ValueError('R18 group count')
    if {f: len({w['group'] for w in worlds if w['family'] == f}) for f in FAMILY_KEYS} != EXPECTED['groups_per_family']:
        raise ValueError('R18 groups per family')
    for world in worlds:
        if world['verdict'] != 'valid':
            raise ValueError('R18 world is not valid: ' + world['event_id'])
        if world['passing_baselines']:
            raise ValueError('R18 world has a passing baseline: ' + world['event_id'])
        if world['any_constant_feasible']:
            raise ValueError('R18 world is constant passable: ' + world['event_id'])
        reference = world['reference']
        if reference.get('status') != 'feasible' or len(reference.get('actions') or []) != DECISIONS:
            raise ValueError('R18 reference witness missing: ' + world['event_id'])
        if not reference.get('witness_replay'):
            raise ValueError('R18 reference witness did not replay: ' + world['event_id'])
        if world['scenario']['split'] != 'development' or world['scenario']['horizon'] != HORIZON:
            raise ValueError('R18 scenario split/horizon: ' + world['event_id'])
        for name, actions in world['baseline_actions'].items():
            if len(actions) != DECISIONS:
                raise ValueError('R18 baseline action table: ' + world['event_id'] + '/' + name)
        if len(world['weak']['actions']) != DECISIONS:
            raise ValueError('R18 weak baseline action table: ' + world['event_id'])
    index = {world['event_id']: world for world in worlds}
    if len(index) != len(worlds):
        raise ValueError('duplicate R18 event id')
    return dict(spec=spec, sha256=hashlib.sha256(raw).hexdigest(), index=index)


def load_split(manifest):
    """Read-only load of the R19 split, bound to the exact R18 manifest bytes."""
    raw = SPLIT.read_bytes()
    spec = json.loads(raw)
    if spec.get('schema') != SPLIT_SCHEMA:
        raise ValueError('R19 split schema')
    if spec['source']['manifest_sha256'] != manifest['sha256']:
        raise ValueError('the R19 split is bound to a different R18 manifest')
    if spec['source']['events'] != EXPECTED['events'] or spec['source']['groups'] != EXPECTED['groups']:
        raise ValueError('R19 split source counts')
    groups = spec['groups']
    if len(groups) != EXPECTED['groups']:
        raise ValueError('R19 group count')
    if spec['assertions'].get('test_split'):
        raise ValueError('no test split may exist')
    seen = {}
    for group in groups:
        if group['split'] not in (TRAIN_SPLIT, VALIDATION_SPLIT):
            raise ValueError('unexpected split label: ' + str(group['split']))
        if len(group['events']) != 2:
            raise ValueError('a group must hold exactly two events: ' + group['group'])
        if {bool(event['mirror']) for event in group['events']} != {False, True}:
            raise ValueError('a group must hold one base and one mirror: ' + group['group'])
        for event in group['events']:
            world = manifest['index'][event['event_id']]
            if world['group'] != group['group'] or world['family'] != group['family']:
                raise ValueError('split/manifest group mismatch: ' + event['event_id'])
            if bool(world['mirror']) != bool(event['mirror']):
                raise ValueError('split/manifest mirror mismatch: ' + event['event_id'])
            if abs(float(world['scenario']['start_x']) - float(event['start_x'])) > 1e-9:
                raise ValueError('split/manifest start_x mismatch: ' + event['event_id'])
            if json.loads(json.dumps(r16.signature(world['scenario']))) != group['signature']:
                raise ValueError('split/manifest signature mismatch: ' + event['event_id'])
            if event['event_id'] in seen:
                raise ValueError('event listed twice: ' + event['event_id'])
            seen[event['event_id']] = group['split']
    if set(seen) != set(manifest['index']) or len(seen) != EXPECTED['events']:
        raise ValueError('the split must cover the 28 R18 worlds exactly once')
    return dict(spec=spec, sha256=hashlib.sha256(raw).hexdigest(), groups=groups, assignment=seen)


def split_counts(split):
    events = {name: 0 for name in (TRAIN_SPLIT, VALIDATION_SPLIT)}
    groups = {name: 0 for name in (TRAIN_SPLIT, VALIDATION_SPLIT)}
    by_family = {}
    for group in split['groups']:
        groups[group['split']] += 1
        events[group['split']] += len(group['events'])
        by_family.setdefault(group['family'], {TRAIN_SPLIT: 0, VALIDATION_SPLIT: 0})[group['split']] += 1
    return dict(groups=groups, events=events, by_family=by_family)


def world_events(manifest, split, split_name):
    """The R18 worlds of one split, in R18 manifest order, as frozen event records."""
    keep = {event for event, name in split['assignment'].items() if name == split_name}
    rows = []
    for world in manifest['spec']['worlds']:
        if world['event_id'] not in keep:
            continue
        payload = world['scenario']
        rows.append(dict(event_id=world['event_id'], family=world['family'], group=world['group'],
                         split=split_name, mirror=bool(world['mirror']), scenario=payload,
                         scenario_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()))
    if len(rows) != EXPECTED[f'{split_name}_events']:
        raise ValueError(f'{split_name} event count')
    if {f: sum(r['family'] == f for r in rows) for f in FAMILY_KEYS} != EXPECTED[f'{split_name}_events_per_family']:
        raise ValueError(f'{split_name} event counts per family')
    return rows


def split_checks(manifest, split, train_events, validation_events):
    """Hard assertions over the frozen partition; recorded in the pre-registration."""
    counts = split_counts(split)
    groups = {}
    for group in split['groups']:
        if group['group'] in groups:
            raise ValueError('duplicate group')
        groups[group['group']] = group['split']
    if counts['groups'] != {TRAIN_SPLIT: EXPECTED['train_groups'], VALIDATION_SPLIT: EXPECTED['validation_groups']}:
        raise ValueError('group split counts')
    if counts['events'] != {TRAIN_SPLIT: EXPECTED['train_events'], VALIDATION_SPLIT: EXPECTED['validation_events']}:
        raise ValueError('event split counts')
    if counts['by_family'] != {f: {TRAIN_SPLIT: EXPECTED['train_groups_per_family'][f],
                                  VALIDATION_SPLIT: EXPECTED['validation_groups_per_family'][f]}
                              for f in FAMILY_KEYS}:
        raise ValueError('per-family group split counts')
    ids = [row['event_id'] for row in train_events] + [row['event_id'] for row in validation_events]
    if len(set(ids)) != EXPECTED['events']:
        raise ValueError('every event must belong to exactly one split')
    order = []
    for family in FAMILY_KEYS:
        family_groups = sorted((group for group in split['groups'] if group['family'] == family),
                               key=lambda group: group['index_in_family'])
        order.append(dict(family=family, order=[group['group'] for group in family_groups],
                          splits=[group['split'] for group in family_groups]))
    splits = {row['event_id']: row['split'] for row in train_events + validation_events}
    leaked = sorted(event for group in split['groups'] for event in (e['event_id'] for e in group['events'])
                    if splits[event] != group['split'])
    if leaked:
        raise ValueError('group leakage: ' + ', '.join(leaked))
    return dict(groups=counts['groups'], events=counts['events'], by_family=counts['by_family'],
                group_splits=groups, every_event_once=True, no_group_crosses_split=not leaked,
                mirror_and_base_in_one_group=True, every_family_has_both_splits=True,
                train_events=[row['event_id'] for row in train_events],
                validation_events=[row['event_id'] for row in validation_events],
                family_order=order, duplicate_signatures=0, signature_fields=list(SIGNATURE_FIELDS))


# ---------------------------------------------------------------- data build
def render_event(event):
    """One teacherV2 rollout, its causal render, causal depth-3 regions, history and labels."""
    world = make_world(event['scenario'])
    actions, decisions = teacher_rollout(world)
    if len(actions) != DECISIONS:
        raise ValueError('incomplete teacher rollout: ' + event['event_id'])
    if any(action not in ACTION_MAP for action in actions):
        raise ValueError('teacher action outside the frozen action set: ' + event['event_id'])
    verdict = verify(world, actions)
    verified = verdict == 'feasible'
    states = full_states(world, actions)
    native = np.stack([render_state(world, state) for state in states])
    if native.dtype != np.uint8 or native.shape != (HORIZON + 1, 480, 640, 3):
        raise ValueError('renderer output shape/dtype: ' + event['event_id'])
    resized = {name: [] for name in SIZES}
    for frame in range(0, HORIZON, 2):
        for name, image in vision.extract_regions(native[frame]).items():
            resized[name].append(vision.resize_area(image, *SIZES[name]))
    ids = depth_indices()
    stacks = {}
    for name, frames in resized.items():
        array = np.stack(frames)
        if array.dtype != np.uint8:
            raise ValueError('uint8 frames required: ' + name)
        stacks[name] = np.ascontiguousarray(array[ids])
    hud_constant = {name: len({frame.tobytes() for frame in resized[name]}) == 1 for name in HUD_NAMES}
    if not all(hud_constant.values()):
        raise ValueError('HUD regions must be constant for the single-copy storage: ' + event['event_id'])
    stored = {'playfield': stacks['playfield']}
    for name in HUD_NAMES:
        stored[name] = np.ascontiguousarray(stacks[name][:1])
    history, labels = history_and_labels(actions, verified)
    switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
    return dict(world=world, actions=actions, decisions=decisions, verdict=verdict, verified=verified,
                stored=stored, history=history, labels=labels, hud_constant=hud_constant,
                switches=switches, distinct_actions=sorted(set(actions)),
                action_counts={str(a): actions.count(a) for a in sorted(set(actions))})


def build_event(directory, event):
    payload = render_event(event)
    pixels = directory / 'pixels.npz'
    history = directory / 'history.pt'
    labels = directory / 'labels.pt'
    write_npz(pixels, payload['stored'])
    write_torch(history, payload['history'])
    write_torch(labels, payload['labels'])
    rollout = directory / 'rollout.json'
    row = dict(event_id=event['event_id'], family=event['family'], group=event['group'],
               split=event['split'], mirror=event['mirror'], scenario=event['scenario'],
               scenario_sha256=event['scenario_sha256'],
               teacher='TacticalTeacherV2 (unmodified)', teacher_source=teacher_hashes(),
               actions=payload['actions'], decisions=payload['decisions'],
               decisions_count=len(payload['actions']), horizon=HORIZON,
               verify_verdict=payload['verdict'], teacher_verified=payload['verified'],
               switches=payload['switches'], distinct_actions=payload['distinct_actions'],
               action_counts=payload['action_counts'], hud_constant=payload['hud_constant'],
               depth_frames=depth_frames(), region_sizes=SIZES,
               labels=dict(rows=DECISIONS, action_valid=int(payload['labels']['action_valid'].sum()),
                           positive_per_valid_row=1, semantics='single route first action only'))
    write_json(rollout, row)
    raw_bytes = sum(int(array.nbytes) for array in payload['stored'].values())
    return dict(event=event, payload=payload,
                refs=dict(pixels=dict(path=f"assets/{event['event_id']}/pixels.npz", sha256=file_sha(pixels),
                                      bytes=pixels.stat().st_size, raw_bytes=raw_bytes,
                                      shapes={name: list(array.shape) for name, array in payload['stored'].items()}),
                          history=dict(path=f"assets/{event['event_id']}/history.pt", sha256=file_sha(history),
                                       bytes=history.stat().st_size),
                          labels=dict(path=f"assets/{event['event_id']}/labels.pt", sha256=file_sha(labels),
                                      bytes=labels.stat().st_size),
                          rollout=dict(path=f"assets/{event['event_id']}/rollout.json", sha256=file_sha(rollout),
                                       bytes=rollout.stat().st_size)))


def build_assets(dest, events):
    new_dir(dest / 'assets')
    built = []
    for event in events:
        directory = new_dir(dest / 'assets' / event['event_id'])
        built.append(build_event(directory, event))
        print('rendered', event['event_id'], 'verdict', built[-1]['payload']['verdict'],
              'stored_bytes', built[-1]['refs']['pixels']['bytes'], flush=True)
    return built


def write_records(dest, built):
    entries, failures = [], []
    for item in built:
        event = item['event']
        entries.append(dict(event_id=event['event_id'], family=event['family'], group=event['group'],
                            split=event['split'], mirror=event['mirror'], rows=DECISIONS, horizon=HORIZON,
                            logical_frames=HORIZON, teacher_verified=item['payload']['verified'],
                            verify_verdict=item['payload']['verdict'], scenario_sha256=event['scenario_sha256'],
                            switches=item['payload']['switches'],
                            distinct_actions=item['payload']['distinct_actions'],
                            action_counts=item['payload']['action_counts'], **item['refs']))
        if not item['payload']['verified']:
            failures.append(dict(event_id=event['event_id'], verdict=item['payload']['verdict']))
    records = dict(schema=DATA_SCHEMA, root='assets', events=entries, failures=failures,
                   depth=3, region_sizes=SIZES, depth_frames='[max(0,f-4),max(0,f-2),f]',
                   history_semantics='actual executed candidate (ACTION_MAP), previous_hold 2, frame_delta 2, '
                                     'option BOS3, target_valid false, history_valid from row 1; no teacher target or '
                                     'option is ever fed as input',
                   label_semantics='action_positive = the one-hot of the single successful teacher route first '
                                   'action; action_valid = teacher_verified; nothing else',
                   storage=STORAGE,
                   totals=dict(events=len(entries), verified=len(entries) - len(failures), failed=len(failures),
                               rows=len(entries) * DECISIONS,
                               bytes=sum(e['pixels']['bytes'] + e['history']['bytes'] + e['labels']['bytes']
                                         + e['rollout']['bytes'] for e in entries)))
    write_json(dest / 'records.json', records)
    return records


# ---------------------------------------------------------------- encode / train
def pixel_tensors(path):
    """uint8 npz -> float32 region tensors /255; HUD rows expanded, values unchanged."""
    with np.load(path) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    out = {}
    for name, array in arrays.items():
        if array.dtype != np.uint8:
            raise ValueError('uint8 storage required: ' + name)
        value = vision.regions_to_batch({name: array})[name]
        if value.shape[0] != DECISIONS:
            if value.shape[0] != 1:
                raise ValueError('unexpected stored row count: ' + name)
            value = value.expand(DECISIONS, -1, -1, -1, -1)
        out[name] = value
    return out


def encode(backbone, records, dest, device):
    data = []
    with torch.inference_mode():
        for entry in records['events']:
            pixels = pixel_tensors(dest / entry['pixels']['path'])
            features, base = [], []
            for start in range(0, DECISIONS, CHUNK):
                core = backbone.forward_core({k: v[start:start + CHUNK].to(device) for k, v in pixels.items()})
                features.append(core['features'].to(device).clone())
                base.append(core['policy_logits'].to(device).clone())
            for key in ('history', 'labels'):
                if file_sha(dest / entry[key]['path']) != entry[key]['sha256']:
                    raise ValueError(f'{key} asset SHA drift: ' + entry['event_id'])
            history = torch.load(dest / entry['history']['path'], map_location='cpu', weights_only=True)
            labels = torch.load(dest / entry['labels']['path'], map_location='cpu', weights_only=True)
            data.append(dict(entry=entry, features=torch.cat(features), base=torch.cat(base),
                             history={k: v.to(device) for k, v in history.items()},
                             labels={k: v.to(device) for k, v in labels.items()}))
    return data


def infer(model, entry):
    return model.forward_features(entry['features'][None], entry['base'][None],
                                  {k: v[None] for k, v in entry['history'].items()})


def anchor_mask(output, labels):
    return output.get('valid_mask', torch.ones_like(labels['action_valid'])) & labels['action_valid']


def kl_anchor(output, labels):
    """KL(base||policy) over every action_valid row; the base logits stay detached."""
    logits = output['policy_logits']
    base = output['base_logits']
    if logits.shape != base.shape or logits.shape[-1] != 18:
        raise ValueError('policy/base logits shape')
    mask = anchor_mask(output, labels)
    if not int(mask.sum()):
        raise ValueError('the KL anchor needs at least one action_valid row')
    return F.kl_div(logits[mask].log_softmax(-1), base[mask].detach().log_softmax(-1),
                    log_target=True, reduction='batchmean')


def arm_loss(arm, output, labels):
    original = tactical_policy_loss(output, labels)
    if arm == CONTROL:
        return dict(loss=original['loss'], original=original['loss'], anchor=None, parts=original)
    if arm != ANCHOR:
        raise ValueError('unknown arm: ' + str(arm))
    anchor = kl_anchor(output, labels)
    return dict(loss=original['loss'] + ANCHOR_WEIGHT * anchor, original=original['loss'],
                anchor=anchor, parts=original)


def accuracy(truth, pred):
    return sum(int(a == b) for a, b in zip(truth, pred)) / len(truth) if truth else None


def offline(model, data):
    """Label fit on the rendered (seen) events; the validation split carries no labels at all."""
    model.eval()
    rows = []
    with torch.inference_mode():
        for entry in data:
            output = infer(model, entry)
            labels = entry['labels']
            valid = labels['action_valid']
            positive = labels['action_positive'][valid]
            if int(valid.sum()) and not bool((positive.sum(-1) == 1).all()):
                raise ValueError('single route labels required')
            logits = output['policy_logits'][0][valid]
            masked = logits.clone()
            masked[:, [c for c in range(18) if c not in HORIZONTAL]] = -torch.inf
            rows.append(dict(event_id=entry['entry']['event_id'], family=entry['entry']['family'],
                             rows=int(valid.sum()),
                             loss=float(tactical_policy_loss(output, {k: v[None] for k, v in labels.items()})['loss']),
                             truth=positive.long().argmax(-1).tolist() if int(valid.sum()) else [],
                             unmasked=logits.argmax(-1).tolist(), masked=masked.argmax(-1).tolist()))
    summary = {}
    for family in FAMILIES:
        group = [row for row in rows if family == 'all' or row['family'] == family]
        truth = sum((row['truth'] for row in group), [])
        summary[family] = dict(events=len(group), rows=sum(row['rows'] for row in group),
                               loss=sum(row['loss'] for row in group) / len(group) if group else None,
                               unmasked_accuracy=accuracy(truth, sum((row['unmasked'] for row in group), [])),
                               masked_accuracy=accuracy(truth, sum((row['masked'] for row in group), [])))
    return dict(summary=summary, events=rows)


# ---------------------------------------------------------------- closed loop
def compact(row):
    """Stored closed-loop row: the pixel dump, the per-step history mirror and the trace are dropped."""
    physical = row.get('environment_metrics') or {}
    return dict(event_id=row['event_id'], family=row['family'], arm=row['arm'], condition=row['condition'],
                first_collision_frame=row['first_collision_frame'], full_horizon=row['full_horizon'],
                goal_satisfied=row['goal_satisfied'], survived=row['survived'], feasible=row['feasible'],
                invalid_action=row['invalid_action'], technical_failure=row['technical_failure'],
                raw_unsupported_count=row['raw_unsupported_count'], decisions=row['decisions'],
                verdict=physical.get('verdict'), reached_goal=physical.get('reached_goal'),
                final_horizontal_space_px=physical.get('final_horizontal_space_px'),
                steps=[{key: step.get(key) for key in STEP_FIELDS} for step in row['steps']])


def behaviour(rows):
    """Action distribution and switch structure of the stored decisions."""
    counts, actions, switches, per_event = {}, {}, [], []
    decisions = 0
    for row in rows:
        steps = row.get('steps') or []
        candidates = [step.get('actual_candidate') for step in steps]
        actual = [step.get('actual_action') for step in steps]
        for candidate in candidates:
            if candidate is not None:
                counts[str(candidate)] = counts.get(str(candidate), 0) + 1
        for action in actual:
            if action is not None:
                actions[str(action)] = actions.get(str(action), 0) + 1
        value = sum(1 for i in range(1, len(candidates)) if candidates[i] != candidates[i - 1])
        switches.append(value)
        decisions += len(steps)
        distinct = sorted({c for c in candidates if c is not None})
        per_event.append(dict(event_id=row['event_id'], family=row['family'], feasible=row['feasible'],
                              survived=row['survived'], first_hit=row['first_collision_frame'],
                              decisions=len(steps), switches=value, distinct_candidates=distinct,
                              single_candidate=len(distinct) == 1,
                              action_counts={str(a): actual.count(a) for a in sorted({a for a in actual if a is not None})}))
    total = sum(counts.values()) or 1
    action_total = sum(actions.values()) or 1
    left = actions.get('-2', 0) + actions.get('-1', 0)
    right = actions.get('2', 0) + actions.get('1', 0)
    single = [row['event_id'] for row in per_event if row['single_candidate']]
    twoseg = [row['event_id'] for row in per_event if row['switches'] <= 1]
    return dict(events=len(rows), decisions=decisions,
                candidate_counts=dict(sorted(counts.items(), key=lambda kv: int(kv[0]))),
                candidate_shares={k: round(v / total, 6) for k, v in sorted(counts.items(), key=lambda kv: int(kv[0]))},
                action_counts=dict(sorted(actions.items(), key=lambda kv: int(kv[0]))),
                action_shares=dict(hold=round(actions.get('0', 0) / action_total, 6),
                                   left=round(left / action_total, 6),
                                   right=round(right / action_total, 6)),
                switches=dict(values=switches,
                              median=float(statistics.median(switches)) if switches else None,
                              mean=round(sum(switches) / len(switches), 6) if switches else None,
                              minimum=min(switches) if switches else None,
                              maximum=max(switches) if switches else None,
                              zero_switch_events=sum(1 for value in switches if value == 0),
                              at_most_one_switch_events=sum(1 for value in switches if value <= 1)),
                single_candidate_events=single,
                single_candidate_feasible_events=[row['event_id'] for row in per_event
                                                  if row['single_candidate'] and row['feasible']],
                two_segment_events=twoseg,
                two_segment_feasible_events=[row['event_id'] for row in per_event
                                             if row['switches'] <= 1 and row['feasible']],
                per_event=per_event,
                note='candidates are the frozen horizontal six-class domain; actions are px/frame/2 in {-2,-1,0,1,2}; '
                     'a constant run is "single_candidate" (one distinct candidate over all 80 decisions) and a '
                     'two-segment run is "switches<=1" (at most one change); both are counted, listed and reported, '
                     'never dropped')


def closed_loop(events, network, arm, label):
    rows = []
    for entry in events:
        row = compact(run_event(make_world(entry['scenario']), arm, CONDITION, network, 0, None))
        row['network'] = label
        rows.append(row)
        print('closed-loop', label, entry['event_id'], 'feasible', row['feasible'],
              'survived', row['survived'], 'firsthit', row['first_collision_frame'], flush=True)
    return rows


class BaselineNetwork:
    """R11 B0 semantics: the frozen baseline policy logits only, no tactical heads."""

    def __init__(self, backbone, device):
        self.arm = 'B0'
        self.device = device
        self.name = 'B0'
        self.model = backbone

    @torch.inference_mode()
    def __call__(self, arm, regions, previous, hidden, reset):
        if arm != self.arm:
            raise ValueError('callback arm mismatch')
        core = self.model.forward_core({key: value.to(self.device) for key, value in regions.items()})
        return core['policy_logits'][0].cpu(), None


class Network:
    """Arm-aware callback over one frozen research checkpoint, manually checked."""

    def __init__(self, backbone, path, device, expected_sha=None):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('schema') != SCHEMA:
            raise ValueError('research schema')
        if payload.get('arm') not in ARMS or payload.get('seed') not in SEEDS:
            raise ValueError('research arm/seed')
        if payload['baseline_sha256'] != file_sha(BASELINE):
            raise ValueError('baseline SHA')
        if payload.get('anchor_weight') != (ANCHOR_WEIGHT if payload['arm'] == ANCHOR else 0.):
            raise ValueError('anchor weight record')
        self.arm = 'N1'
        self.device = device
        self.name = name_of(payload['arm'], payload['seed'])
        self.model = TacticalPolicy(backbone, TacticalConfig(**payload['config'])).to(device).eval()
        if self.model.tactical_config.arm != 'N1':
            raise ValueError('N1 only')
        self.model.heads.load_state_dict(payload['heads'], strict=True)
        self.heads_sha256 = r16.state_sha(self.model.heads.state_dict())
        if self.heads_sha256 != payload['final_head_state_sha256']:
            raise ValueError('checkpoint head SHA drift')
        if expected_sha is not None and self.heads_sha256 != expected_sha:
            raise ValueError('unexpected checkpoint')

    @torch.inference_mode()
    def __call__(self, arm, regions, previous, hidden, reset):
        if arm != self.arm:
            raise ValueError('callback arm mismatch')
        core = self.model.vision_model.forward_core({key: value.to(self.device) for key, value in regions.items()})
        features = core['features'][:, None]
        base = core['policy_logits'][:, None]
        output = self.model.forward_features(features, base, executed_history(features, previous), hidden,
                                             reset_mask=torch.tensor([[reset]], device=self.device))
        return output['policy_logits'][0, 0].cpu(), output['hidden']


def six_baselines(events):
    """The frozen R18 six baselines re-executed on the given worlds; no pixels involved."""
    rows = []
    for entry in events:
        world = make_world(entry['scenario'])
        table, _ = r18.policies(world)
        for name in r18.BASELINES + (r18.WEAK,):
            rows.append(dict(event_id=world.event_id, family=world.family, split=entry['split'], policy=name,
                             **r18.outcome(world, r18.causal_loop(world, table[name]))))
    summary = {}
    for family in FAMILIES:
        group = [row for row in rows if family == 'all' or row['family'] == family]
        summary[family] = {}
        for name in r18.BASELINES + (r18.WEAK,):
            subset = [row for row in group if row['policy'] == name]
            summary[family][name] = dict(events=len(subset), feasible=sum(bool(r['feasible']) for r in subset),
                                         survived=sum(bool(r['survived']) for r in subset),
                                         firsthit={r['event_id']: r['first_hit'] for r in subset},
                                         feasible_events=sorted(r['event_id'] for r in subset if r['feasible']))
    passing = sorted({row['event_id'] for row in rows if row['feasible']})
    return dict(events=len(events), rows=len(rows), policies=list(r18.BASELINES + (r18.WEAK,)),
                executor='unchanged R18 policies()/causal_loop()/outcome() on the same worlds, executed here',
                summary=summary, passing_events=passing, filter_valid=not passing, rows_detail=rows)


# ---------------------------------------------------------------- pre-registration
def criterion():
    """The frozen decision rules of this round; any drift aborts the run."""
    return dict(
        round='round20',
        purpose='first honest teacher distillation on the R18-valid curriculum, read out as a grouped-holdout '
                'horizontal_masked closed loop',
        data=dict(
            source='R18 run/manifest.json (28 worlds, 14 groups of base+mirror) joined with the frozen R19 '
                   'run/split.json; no world regenerated, no parameter changed, no budget added, no test split',
            train=dict(events=14, groups=7, groups_per_family={'F1': 3, 'F2': 1, 'F3': 3},
                       use='the only events that are rendered, labelled, encoded or trained on'),
            validation=dict(events=14, groups=7, groups_per_family={'F1': 3, 'F2': 1, 'F3': 3},
                            use='read-only: never trained, never labelled, never encoded; closed loop only'),
            mirror_note='a group is atomic: the mirror is not an independent sample and never lands in the other split'),
        rollout=dict(teacher='TacticalTeacherV2, unmodified, module SHA frozen',
                     driver='touhou_rl.tactical_teacher_v2.rollout: decide() then advance() the actual chosen action, '
                            'so the world is advanced causally by executed actions only',
                     horizon=HORIZON, decisions=DECISIONS,
                     verify='tactical_teacher_validation.verify(world, all 80 actions), full horizon, no early stop',
                     pass_required='verdict == feasible',
                     non_pass_rule='an event whose teacherV2 rollout does not verify to feasible is excluded from '
                                   'supervision (action_valid all False) and retained as a failure; expected 0 because '
                                   'R19 measured 28/28'),
        render=dict(source='touhou_rl.tactical_synthetic_pixels.render_state / full_states, unchanged: the exact R10 '
                           'renderer used for the R10 records',
                    depth=3, causal_frames='[max(0,f-4), max(0,f-2), f] logical frames, chronological, repeat-initial '
                                           'padding',
                    sampling='decision i uses logical frame f=2*i for i in 0..79',
                    regions='unchanged ROI crop + area resize: playfield 224x192, hud_score 34x216, hud_status 126x216',
                    hidden='no future wall/window/goal geometry, no teacher target, option, countdown or hit state is '
                           'ever drawn',
                    storage=STORAGE,
                    labels='action_positive = the one-hot of the single successful teacher route first action; '
                           'action_valid = teacher_verified; nothing else',
                    history='actual executed candidate (ACTION_MAP), previous_hold 2, frame_delta 2, option BOS3, '
                            'target_valid false, history_valid from row 1'),
        training=dict(arms={CONTROL: 'no anchor: the unchanged tactical_policy_loss',
                            ANCHOR: 'L_original + 1.0*KL(base||policy) over every action_valid row'},
                      formula=FORMULA, anchor_weight=ANCHOR_WEIGHT,
                      seeds=list(SEEDS), updates=UPDATES, optimizer='Adam', lr=LR, clip=CLIP,
                      sequence=SEQUENCE, batch=BATCH,
                      precision='FP32 without AMP, autocast disabled, matmul TF32 False, cuDNN TF32 False, float32 '
                                'matmul precision highest',
                      features='one shared encode pass per event over 5 chunks of 16 rows, identical tensors for both '
                               'arms, no disk feature cache',
                      initialization='fresh torch seed reset immediately before every N1 construction after the '
                                     'backbone load; the same seed must produce a byte-identical initial head state in '
                                     'both arms (hard assertion) and must equal the frozen R16/R17 initial head state '
                                     'for that seed',
                      order='verified train events in R18 manifest order, step modulo the supervised count; all 200 '
                            'updates must be supervised',
                      selection='final step only: no early stopping, no validation-based model selection, no tuning of '
                                'lr/clip/weight/epochs/order'),
        evaluation=dict(
            order='all six checkpoints written and SHA-frozen first; only then B0 and the six R18 baselines on the '
                  'same validation renders, then the six checkpoints on validation, then the six on train (seen)',
            condition=CONDITION,
            runner='unchanged R11 run_event / pixel_stack / render_state / executed_history: 18 logits masked to the '
                   'six supported candidates before argmax, candidate -> two-frame action, real advance, sticky '
                   'collision, continue to the full horizon, per-event hidden reset',
            b0='the frozen baseline policy logits only on the identical renders; a reference, never a B0 capability '
               'claim',
            six_baselines='R18 policies()/causal_loop()/outcome() re-executed on the validation worlds: hold, constant '
                          'right (+2), constant left (-2), alternate every 16 decisions, random seed 37318 over the '
                          'six supported candidates, weak teacher-first-action',
            holdout_rule='if ANY of the six R18 baselines is feasible on ANY validation event then the R18 filter has '
                         'failed on the holdout side: report it verbatim, set holdout_filter_valid=False and withhold '
                         'the learning conclusion of this round',
            metrics='per arm per seed per family feasible and survived counts plus every event firsthit; the train '
                    '(seen) split reported separately and marked as seen; action distribution hold/right/left share '
                    'and median switches',
            reporting='per family F1/F2/F3, never only aggregated; n=3 seeds; F2 validation has one group; mirrors '
                      'are not independent samples',
            degeneracy='a run is constant when one candidate occupies all 80 decisions of an event and two-segment '
                       'when switches<=1; both are counted, listed and reported, never dropped silently'),
        claims=dict(
            significance='none: 3 seeds, 7 validation groups, at most 14 independent geometries (F2 has 2 nearly '
                         'identical); this is a descriptive comparison, not a significance test',
            gate='no formal gate opened, no deployable checkpoint produced, no game executed',
            can_show='whether a fixed 200-update supervised head trained on 14 seen teacher rollouts preserves or '
                     'improves the horizontal_masked closed loop on 14 unseen worlds, and whether the frozen-base KL '
                     'anchor changes that',
            cannot_show='generalisation beyond 7 held-out geometries, tactical understanding, or that the teacher is '
                        'learnable in any formal sense'),
        failure_policy='first failure retained; one run; no retrain, rerun, repair, overwrite, extra epoch or baseline '
                       'switch')


def criterion_sha(value=None):
    return hashlib.sha256(json.dumps(criterion() if value is None else value, sort_keys=True).encode()).hexdigest()


def identity_paths():
    return [Path(__file__), Path('tests/test_v373_valid_curriculum.py'),
            Path('touhou_rl/tactical_synthetic_pixels.py'), Path('touhou_rl/tactical_teacher_v2.py'),
            Path('touhou_rl/tactical_teacher_validation.py'), Path('touhou_rl/tactical_teacher.py'),
            Path('touhou_rl/tactical_synthetic_closed_loop.py'), Path('touhou_rl/tactical_scenarios.py'),
            Path('touhou_rl/tactical_policy.py'), Path('touhou_rl/vision.py'), Path('touhou_rl/vmodels.py'),
            Path('tools/rl/v373_kl_anchor.py'), Path('tools/rl/v373_world_filter.py'),
            Path('tools/rl/v373_seed_confirm.py'), Path('tools/rl/v373_arm_diagnostic.py'),
            Path('tools/rl/v373_weighted_diagnostic.py'), Path('tools/rl/v373_valid_teacher_eval.py'),
            BASELINE, MANIFEST, SPLIT,
            REPORTS / 'P1-worldfilter-round18.md', REPORTS / 'P2-validteacher-round19.md']


def identities():
    return {str(path): file_sha(path) for path in identity_paths()}


def preregister(out=OUT):
    out = new_dir(out)
    manifest = load_worlds()
    split = load_split(manifest)
    train_events = world_events(manifest, split, TRAIN_SPLIT)
    validation_events = world_events(manifest, split, VALIDATION_SPLIT)
    checks = split_checks(manifest, split, train_events, validation_events)
    initial = r16.initial_head_shas()
    if sorted(initial) != sorted(str(seed) for seed in SEEDS):
        raise ValueError('initial head state must be available for every seed')
    if len(set(initial.values())) != len(SEEDS):
        raise ValueError('different seeds must give different initial head states')
    frozen = dict(manifest=manifest['sha256'], split=split['sha256'],
                  train_worlds_digest=r16.worlds_digest(train_events),
                  validation_worlds_digest=r16.worlds_digest(validation_events),
                  all_worlds_digest=r16.worlds_digest(train_events + validation_events),
                  train_events=[row['event_id'] for row in train_events],
                  validation_events=[row['event_id'] for row in validation_events],
                  train_scenario_sha256={row['event_id']: row['scenario_sha256'] for row in train_events},
                  validation_scenario_sha256={row['event_id']: row['scenario_sha256'] for row in validation_events})
    value = dict(schema=PREREG_SCHEMA, identities=identities(), arms=list(ARMS), seeds=list(SEEDS),
                 updates=UPDATES, lr=LR, clip=CLIP, sequence=SEQUENCE, batch=BATCH,
                 anchor_weight=ANCHOR_WEIGHT, criterion=criterion(), criterion_sha256=criterion_sha(),
                 sources=dict(manifest=dict(path=str(MANIFEST), sha256=manifest['sha256'], schema=MANIFEST_SCHEMA),
                              split=dict(path=str(SPLIT), sha256=split['sha256'], schema=SPLIT_SCHEMA),
                              baseline=dict(path=str(BASELINE), sha256=file_sha(BASELINE))),
                 teacher=teacher_hashes(), worlds=frozen,
                 split=dict(rule=split['spec']['rule'], counts=split['spec']['counts'],
                            groups=[dict(group=group['group'], family=group['family'],
                                         index_in_family=group['index_in_family'], split=group['split'],
                                         signature=group['signature'], events=group['events'])
                                    for group in split['groups']],
                            assertions=checks),
                 initialization=dict(per_seed=initial, identical_between_arms=True,
                                     method='fresh seed reset immediately before every N1 construction after the '
                                            'backbone load; both arms use the identical N1 config, so the same seed '
                                            'must give a byte-identical initial head state; no warmstart, no '
                                            'checkpoint, no previous-round weights',
                                     cross_round='the per-seed initial head SHA must also equal the frozen R16/R17 '
                                                 'initial head state for that seed'),
                 expected=EXPECTED, storage=STORAGE,
                 independent_test=False, formal_gate=False, deployable=False, game=False, tuning=False,
                 official_labels=False, research_teacher_labels=True, new_training_data=False,
                 failure_policy='first failure retained; one run; no retrain, rerun, repair, overwrite, extra epoch '
                                'or baseline switch')
    write_json(out / 'preregistration.json', value)
    return value


def verify_prereg(out):
    out = Path(out)
    p = json.loads((out / 'preregistration.json').read_text(encoding='utf-8'))
    if p.get('schema') != PREREG_SCHEMA:
        raise ValueError('preregistration schema')
    for path, digest in p['identities'].items():
        if file_sha(path) != digest:
            raise ValueError('frozen identity changed: ' + path)
    if p['criterion_sha256'] != criterion_sha():
        raise ValueError('the criterion drifted from the frozen pre-registration')
    manifest = load_worlds()
    split = load_split(manifest)
    if manifest['sha256'] != p['sources']['manifest']['sha256']:
        raise ValueError('R18 manifest SHA drift')
    if split['sha256'] != p['sources']['split']['sha256']:
        raise ValueError('R19 split SHA drift')
    for name in (TRAIN_SPLIT, VALIDATION_SPLIT):
        rows = world_events(manifest, split, name)
        if r16.worlds_digest(rows) != p['worlds'][f'{name}_worlds_digest']:
            raise ValueError('world digest drift: ' + name)
        if [row['event_id'] for row in rows] != p['worlds'][f'{name}_events']:
            raise ValueError('world event list drift: ' + name)
    if r16.initial_head_shas() != p['initialization']['per_seed']:
        raise ValueError('initial head state drift')
    return p


# ---------------------------------------------------------------- training
def train_all(dest, data, backbone, device, prereg):
    supervised = [entry for entry in data if entry['entry']['teacher_verified']]
    excluded = [entry['entry']['event_id'] for entry in data if not entry['entry']['teacher_verified']]
    if len(supervised) + len(excluded) != len(data) or not supervised:
        raise ValueError('supervision partition')
    new_dir(dest / 'train')
    reports, paths = {}, {}
    for seed in SEEDS:
        per_arm_initial = {}
        for arm in ARMS:
            name = name_of(arm, seed)
            directory = new_dir(dest / 'train' / name)
            model = r16.fresh_policy(backbone, seed).to(device)
            initial_sha = r16.state_sha(model.heads.state_dict())
            if initial_sha != prereg['initialization']['per_seed'][str(seed)]:
                raise AssertionError('initial head state must reproduce the pre-registration: ' + name)
            per_arm_initial[arm] = initial_sha
            write_json(directory / 'initialization.json',
                       dict(head_state_sha256=initial_sha, seed=seed, arm=arm,
                            anchor_weight=ANCHOR_WEIGHT if arm == ANCHOR else 0.,
                            shared_config='both arms are N1; only the extra KL anchor differs',
                            feature_storage_shared=True, supervised_events=len(supervised),
                            excluded_events=excluded))
            parameters = [value for value in model.parameters() if value.requires_grad]
            initial_offline = offline(model, supervised)
            optimizer = torch.optim.Adam(parameters, lr=LR)
            losses, originals, anchors = [], [], []
            begin = time.perf_counter()
            for step in range(UPDATES):
                model.train()
                entry = supervised[step % len(supervised)]
                output = infer(model, entry)
                labels = {key: value[None] for key, value in entry['labels'].items()}
                head = arm_loss(arm, output, labels)
                if not head['parts']['has_supervision']:
                    raise AssertionError('all 200 updates must be supervised')
                optimizer.zero_grad(set_to_none=True)
                head['loss'].backward()
                torch.nn.utils.clip_grad_norm_(parameters, CLIP)
                optimizer.step()
                losses.append(float(head['loss'].detach()))
                originals.append(float(head['original'].detach()))
                if head['anchor'] is not None:
                    anchors.append(float(head['anchor'].detach()))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            final_offline = offline(model, supervised)
            final_sha = r16.state_sha(model.heads.state_dict())
            path = directory / 'research-heads.pt'
            write_torch(path, dict(schema=SCHEMA, arm=arm, seed=seed, config=asdict(model.tactical_config),
                                   baseline_sha256=file_sha(BASELINE), final_head_state_sha256=final_sha,
                                   anchor_weight=ANCHOR_WEIGHT if arm == ANCHOR else 0.,
                                   heads={key: value.detach().cpu() for key, value in model.heads.state_dict().items()}))
            report = dict(arm=arm, seed=seed, name=name, initial_head_state_sha256=initial_sha,
                          final_head_state_sha256=final_sha, checkpoint_sha256=file_sha(path),
                          checkpoint_path=str(path.relative_to(dest)), updates=UPDATES,
                          anchor_weight=ANCHOR_WEIGHT if arm == ANCHOR else 0.,
                          losses=losses, original_losses=originals, anchor_losses=anchors,
                          anchor_loss_at_step0=anchors[0] if anchors else None,
                          anchor_loss_final=anchors[-1] if anchors else None,
                          original_loss_at_step0=originals[0], original_loss_final=originals[-1],
                          loss_at_step0=losses[0], loss_final=losses[-1],
                          optimization_seconds=seconds,
                          parameters=dict(active=sum(value.numel() for value in parameters),
                                          heads_total=sum(value.numel() for value in model.heads.parameters()),
                                          total=sum(value.numel() for value in model.parameters())),
                          supervised_events=[entry['entry']['event_id'] for entry in supervised],
                          excluded_events=excluded,
                          train_only='the validation events are never encoded, labelled or indexed',
                          initial_offline=initial_offline['summary'], final_offline=final_offline['summary'],
                          initial_offline_events=initial_offline['events'],
                          final_offline_events=final_offline['events'])
            write_json(directory / 'training.json', report)
            reports[name] = report
            paths[name] = path
            print('trained', name, 'init', initial_sha[:12], 'final', final_sha[:12],
                  'anchor0', report['anchor_loss_at_step0'], 'loss0', round(losses[0], 4),
                  'lossN', round(losses[-1], 4), f'{seconds:.2f}s', flush=True)
        if len(set(per_arm_initial.values())) != 1:
            raise AssertionError('both arms must start from identical heads for seed ' + str(seed))
    if len(list((dest / 'train').rglob('research-heads.pt'))) != EXPECTED['checkpoints']:
        raise AssertionError('six checkpoints required')
    for name, path in paths.items():
        if file_sha(path) != reports[name]['checkpoint_sha256']:
            raise AssertionError('checkpoint SHA drift: ' + name)
    write_json(dest / 'checkpoints.json',
               {name: {key: value for key, value in reports[name].items()
                       if key not in ('losses', 'original_losses', 'anchor_losses', 'initial_offline_events',
                                      'final_offline_events')} for name in reports})
    return reports, paths


# ---------------------------------------------------------------- evaluation
def event_records(prereg, split_name):
    """The frozen scenario payloads for one split, rebuilt from the SHA-bound manifest."""
    manifest = load_worlds()
    rows = []
    for event_id in prereg['worlds'][f'{split_name}_events']:
        world = manifest['index'][event_id]
        rows.append(dict(event_id=event_id, family=world['family'], group=world['group'], split=split_name,
                         mirror=bool(world['mirror']), scenario=world['scenario'],
                         scenario_sha256=prereg['worlds'][f'{split_name}_scenario_sha256'][event_id]))
    return rows


def evaluate_all(dest, prereg, backbone, device, reports, paths):
    validation = event_records(prereg, VALIDATION_SPLIT)
    train = event_records(prereg, TRAIN_SPLIT)
    new_dir(dest / 'closed-loop')
    begin = time.perf_counter()
    baselines = dict(validation=six_baselines(validation), train=six_baselines(train))
    if sum(payload['rows'] for payload in baselines.values()) != EXPECTED['baseline_rows']:
        raise AssertionError('six baselines x 14 events x 2 splits required')
    write_json(dest / 'baselines.json', baselines)
    closed = {}
    for split_name, events in ((VALIDATION_SPLIT, validation), (TRAIN_SPLIT, train)):
        rows = closed_loop(events, BaselineNetwork(backbone, device), 'B0', 'B0')
        payload = dict(network='B0', split=split_name, condition=CONDITION, events=rows,
                       families=r16.families(rows), behaviour=behaviour(rows))
        write_json(dest / 'closed-loop' / f'B0-{split_name}.json', payload)
        closed[('B0', split_name)] = payload
    for name in all_names():
        network = Network(backbone, paths[name], device, reports[name]['final_head_state_sha256'])
        for split_name, events in ((VALIDATION_SPLIT, validation), (TRAIN_SPLIT, train)):
            rows = closed_loop(events, network, 'N1', name)
            payload = dict(network=name, split=split_name, condition=CONDITION, events=rows,
                           families=r16.families(rows), behaviour=behaviour(rows),
                           heads_sha256=network.heads_sha256)
            write_json(dest / 'closed-loop' / f'{name}-{split_name}.json', payload)
            closed[(name, split_name)] = payload
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    if len(closed) != EXPECTED['closed_loop_files']:
        raise AssertionError('closed-loop file count')
    return baselines, closed, seconds


def write_events(dest, prereg, records, closed, baselines):
    """One JSON per event: the teacher route, the asset binding and every closed-loop row."""
    new_dir(dest / 'events')
    assets = {entry['event_id']: entry for entry in records['events']}
    baseline_rows = {(row['split'], row['event_id'], row['policy']): row
                     for split in baselines.values() for row in split['rows_detail']}
    for event_id in prereg['worlds']['train_events'] + prereg['worlds']['validation_events']:
        split_name = TRAIN_SPLIT if event_id in prereg['worlds']['train_events'] else VALIDATION_SPLIT
        teacher = None
        if event_id in assets:
            teacher = json.loads((dest / assets[event_id]['rollout']['path']).read_text(encoding='utf-8'))
        payloads = {name: closed[(name, split_name)] for name in ['B0'] + all_names()}
        rows = {name: next(row for row in payload['events'] if row['event_id'] == event_id)
                for name, payload in payloads.items()}
        behaviour = {name: next(item for item in payload['behaviour']['per_event'] if item['event_id'] == event_id)
                     for name, payload in payloads.items()}
        value = dict(schema=EVENT_SCHEMA, event_id=event_id, split=split_name,
                     seen=(split_name == TRAIN_SPLIT), family=event_id.split('-')[1],
                     group=event_id.rsplit('-m', 1)[0],
                     teacher_verified=event_id in assets, asset=assets.get(event_id), teacher=teacher,
                     teacher_switch_summary=(dict(switches=teacher['switches'],
                                                  distinct_actions=teacher['distinct_actions'],
                                                  action_counts=teacher['action_counts'])
                                             if teacher else None),
                     r18_six_baselines={name: dict(feasible=baseline_rows[(split_name, event_id, name)]['feasible'],
                                                   survived=baseline_rows[(split_name, event_id, name)]['survived'],
                                                   first_hit=baseline_rows[(split_name, event_id, name)]['first_hit'])
                                        for name in baselines['validation']['policies']},
                     closed_loop={name: dict(feasible=rows[name]['feasible'], survived=rows[name]['survived'],
                                             goal_satisfied=rows[name]['goal_satisfied'],
                                             first_hit=rows[name]['first_collision_frame'],
                                             decisions=rows[name]['decisions'],
                                             raw_unsupported_count=rows[name]['raw_unsupported_count'],
                                             action_counts=behaviour[name]['action_counts'],
                                             switches=behaviour[name]['switches'],
                                             distinct_candidates=behaviour[name]['distinct_candidates'],
                                             single_candidate=behaviour[name]['single_candidate'])
                                  for name in ['B0'] + all_names()})
        write_json(dest / 'events' / f'{event_id}.json', value)


# ---------------------------------------------------------------- summary
def summary_of(values):
    return dict(values=list(values), mean=sum(values) / len(values), minimum=min(values),
                maximum=max(values), range=max(values) - min(values))


def arm_stats(results, key):
    stats = {}
    for arm in ARMS:
        stats[arm] = {}
        for family in FAMILIES:
            entry = dict(seeds=list(SEEDS))
            for metric in ('feasible', 'survived'):
                entry[metric] = summary_of([results[name_of(arm, seed)][key][family][metric] for seed in SEEDS])
            stats[arm][family] = entry
    return stats


def paired(results):
    per_seed = {}
    for seed in SEEDS:
        control, anchor = results[name_of(CONTROL, seed)], results[name_of(ANCHOR, seed)]
        per_seed[str(seed)] = {}
        for key in ('validation', 'train_seen'):
            per_seed[str(seed)][key] = {family: dict(
                feasible_delta=anchor[key][family]['feasible'] - control[key][family]['feasible'],
                survived_delta=anchor[key][family]['survived'] - control[key][family]['survived'])
                for family in FAMILIES}
    tally = {}
    for key in ('validation', 'train_seen'):
        tally[key] = {}
        for family in FAMILIES:
            deltas = [per_seed[str(seed)][key][family]['feasible_delta'] for seed in SEEDS]
            tally[key][family] = dict(deltas=deltas, higher=sum(d > 0 for d in deltas),
                                      equal=sum(d == 0 for d in deltas), lower=sum(d < 0 for d in deltas))
    return dict(per_seed=per_seed, tally=tally,
                note='delta = anchor(B) minus control(A) on the same seed and the same 14 events; positive means the '
                     'KL anchor kept more successes')


def assemble(dest, extra=None):
    """Rebuild the round summary read-only from the frozen artifacts."""
    dest = Path(dest)
    out = dest.parent
    prereg = json.loads((out / 'preregistration.json').read_text(encoding='utf-8'))
    if prereg['criterion_sha256'] != criterion_sha():
        raise ValueError('the criterion drifted from the frozen pre-registration')
    records = json.loads((dest / 'records.json').read_text(encoding='utf-8'))
    checkpoints = json.loads((dest / 'checkpoints.json').read_text(encoding='utf-8'))
    if sorted(checkpoints) != sorted(all_names()):
        raise ValueError('checkpoint set changed')
    training = {}
    for name in all_names():
        record = checkpoints[name]
        payload = json.loads((dest / 'train' / name / 'training.json').read_text(encoding='utf-8'))
        if file_sha(dest / record['checkpoint_path']) != record['checkpoint_sha256']:
            raise ValueError('checkpoint SHA drift: ' + name)
        if payload['final_head_state_sha256'] != record['final_head_state_sha256']:
            raise ValueError('checkpoint record mismatch: ' + name)
        if payload['initial_head_state_sha256'] != prereg['initialization']['per_seed'][str(payload['seed'])]:
            raise ValueError('initial head SHA differs from the pre-registration: ' + name)
        training[name] = payload
    for seed in SEEDS:
        if training[name_of(CONTROL, seed)]['initial_head_state_sha256'] != \
                training[name_of(ANCHOR, seed)]['initial_head_state_sha256']:
            raise ValueError('both arms must start from identical heads for seed ' + str(seed))
    expected = {VALIDATION_SPLIT: prereg['worlds']['validation_events'],
                TRAIN_SPLIT: prereg['worlds']['train_events']}
    closed = {}
    for name in ['B0'] + all_names():
        for split_name in (VALIDATION_SPLIT, TRAIN_SPLIT):
            payload = json.loads((dest / 'closed-loop' / f'{name}-{split_name}.json').read_text(encoding='utf-8'))
            if r16.families(payload['events']) != payload['families']:
                raise ValueError('stored aggregate mismatch: ' + name + '-' + split_name)
            if [row['event_id'] for row in payload['events']] != expected[split_name]:
                raise ValueError('event list differs from the pre-registration: ' + name + '-' + split_name)
            if name != 'B0' and payload['heads_sha256'] != training[name]['final_head_state_sha256']:
                raise ValueError('closed-loop heads SHA mismatch: ' + name)
            closed[(name, split_name)] = payload
    baselines = json.loads((dest / 'baselines.json').read_text(encoding='utf-8'))
    results = {}
    for name in all_names():
        results[name] = dict(arm=training[name]['arm'], seed=training[name]['seed'],
                             validation=closed[(name, VALIDATION_SPLIT)]['families'],
                             train_seen=closed[(name, TRAIN_SPLIT)]['families'],
                             validation_behaviour=closed[(name, VALIDATION_SPLIT)]['behaviour'],
                             train_seen_behaviour=closed[(name, TRAIN_SPLIT)]['behaviour'],
                             checkpoint_sha256=training[name]['checkpoint_sha256'],
                             final_head_state_sha256=training[name]['final_head_state_sha256'],
                             initial_head_state_sha256=training[name]['initial_head_state_sha256'],
                             anchor_weight=training[name]['anchor_weight'],
                             anchor_loss_at_step0=training[name].get('anchor_loss_at_step0'),
                             anchor_loss_final=training[name].get('anchor_loss_final'),
                             loss_at_step0=training[name].get('loss_at_step0'),
                             loss_final=training[name].get('loss_final'),
                             optimization_seconds=training[name]['optimization_seconds'],
                             final_offline=training[name]['final_offline'],
                             initial_offline=training[name]['initial_offline'])
    holdout = baselines['validation']
    summary = dict(schema=RESULT_SCHEMA, prereg_sha256=file_sha(out / 'preregistration.json'),
                   criterion_sha256=prereg['criterion_sha256'], seeds=list(SEEDS), arms=list(ARMS),
                   updates=UPDATES, anchor_weight=ANCHOR_WEIGHT,
                   intervention=dict(control=CONTROL, anchor=ANCHOR,
                                     note='single factor: 1.0*KL(base||policy) over the action_valid rows of arm B '
                                          'only'),
                   formula=dict(text_sha256=hashlib.sha256(FORMULA.encode()).hexdigest(),
                                anchor_weight=ANCHOR_WEIGHT),
                   sources=prereg['sources'], teacher=prereg['teacher'],
                   worlds=dict(train_events=len(prereg['worlds']['train_events']),
                               validation_events=len(prereg['worlds']['validation_events']),
                               train_worlds_digest=prereg['worlds']['train_worlds_digest'],
                               validation_worlds_digest=prereg['worlds']['validation_worlds_digest'],
                               all_worlds_digest=prereg['worlds']['all_worlds_digest'],
                               train_event_ids=prereg['worlds']['train_events'],
                               validation_event_ids=prereg['worlds']['validation_events']),
                   split=dict(rule=prereg['split']['rule'], counts=prereg['split']['counts'],
                              assertions=prereg['split']['assertions']),
                   data=dict(events=len(records['events']), rows=records['totals']['rows'],
                             storage=records['storage'], depth=records['depth'],
                             region_sizes=records['region_sizes'], depth_frames=records['depth_frames'],
                             history_semantics=records['history_semantics'],
                             label_semantics=records['label_semantics'], totals=records['totals'],
                             failures=records['failures'],
                             excluded_from_supervision=[entry['event_id'] for entry in records['events']
                                                        if not entry['teacher_verified']],
                             assets={entry['event_id']: dict(verified=entry['teacher_verified'],
                                                             verdict=entry['verify_verdict'],
                                                             switches=entry['switches'],
                                                             distinct_actions=entry['distinct_actions'],
                                                             action_counts=entry['action_counts'],
                                                             pixels_sha256=entry['pixels']['sha256'],
                                                             pixels_bytes=entry['pixels']['bytes'],
                                                             history_sha256=entry['history']['sha256'],
                                                             labels_sha256=entry['labels']['sha256'],
                                                             rollout_sha256=entry['rollout']['sha256'])
                                                 for entry in records['events']}),
                   initialization=dict(per_seed={str(seed): training[name_of(CONTROL, seed)]['initial_head_state_sha256']
                                                  for seed in SEEDS},
                                       per_seed_per_arm={str(seed): {arm: training[name_of(arm, seed)]['initial_head_state_sha256']
                                                                     for arm in ARMS} for seed in SEEDS},
                                       identical_between_arms=True, reproducible=True),
                   checkpoints={name: {key: value for key, value in training[name].items()
                                       if key not in ('losses', 'original_losses', 'anchor_losses',
                                                      'initial_offline_events', 'final_offline_events',
                                                      'supervised_events')} for name in all_names()},
                   validation=dict(
                       events=EXPECTED['validation_events'], per_family=EXPECTED['validation_events_per_family'],
                       holdout=dict(filter_valid=holdout['filter_valid'], passing_events=holdout['passing_events'],
                                    executor=holdout['executor'], policies=holdout['policies'],
                                    summary=holdout['summary'],
                                    rule=prereg['criterion']['evaluation']['holdout_rule']),
                       b0=dict(families=closed[('B0', VALIDATION_SPLIT)]['families'],
                               behaviour=closed[('B0', VALIDATION_SPLIT)]['behaviour']),
                       results={name: results[name]['validation'] for name in all_names()},
                       behaviour={name: results[name]['validation_behaviour'] for name in all_names()},
                       aggregate=arm_stats(results, 'validation'),
                       per_event_firsthit={name: closed[(name, VALIDATION_SPLIT)]['families']['all']['firsthit']
                                           for name in ['B0'] + all_names()},
                       note='never trained, never labelled, never encoded: the only split that can falsify this '
                            'distillation'),
                   train_seen=dict(
                       events=EXPECTED['train_events'], per_family=EXPECTED['train_events_per_family'],
                       label='seen: these 14 events are exactly the teacher rollouts the heads were trained on, so '
                             'they measure fit, not generalisation',
                       holdout=dict(filter_valid=baselines['train']['filter_valid'],
                                    passing_events=baselines['train']['passing_events'],
                                    summary=baselines['train']['summary']),
                       b0=dict(families=closed[('B0', TRAIN_SPLIT)]['families'],
                               behaviour=closed[('B0', TRAIN_SPLIT)]['behaviour']),
                       results={name: results[name]['train_seen'] for name in all_names()},
                       behaviour={name: results[name]['train_seen_behaviour'] for name in all_names()},
                       aggregate=arm_stats(results, 'train_seen'),
                       per_event_firsthit={name: closed[(name, TRAIN_SPLIT)]['families']['all']['firsthit']
                                           for name in ['B0'] + all_names()}),
                   paired=paired(results),
                   environment=dict(device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                                    torch_version=torch.__version__,
                                    precision=dict(dtype='float32',
                                                   matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                                   cudnn_tf32=torch.backends.cudnn.allow_tf32,
                                                   matmul_precision=torch.get_float32_matmul_precision()),
                                    timings=json.loads((dest / 'timings.json').read_text(encoding='utf-8'))
                                    if (dest / 'timings.json').is_file() else None),
                   flags=dict(independent_test=False, formal_gate=False, deployable=False, game=False,
                              tuning=False, official_labels=False, research_teacher_labels=True,
                              validation_used_for_selection=False, early_stopping=False, retrained=False,
                              rerun=False),
                   limitations=[
                       'n=3 seeds and 7 validation groups; the 14 validation events collapse to 7 geometries because '
                       'a mirror is not an independent sample',
                       'F2 validation is a single group (2 events) built on two nearly identical worlds (start_x 204 '
                       'vs 208), so an F2 validation number is not a rate',
                       'the labels come from a rule/planner teacher that is handed the real world geometry, not from '
                       'human play and not from a learned policy; "distillation" here means fitting that one route',
                       'the F1 win condition is exact parking inside a 12-20 px gap and the F3 win condition is a timed '
                       'threshold crossing; neither is tactical understanding',
                       'the data is synthetic fixed-y, two-frame actions, no shot, no bomb, no real TH08 pixels',
                       'this round produces research heads only: no formal gate, no deployable checkpoint, no game'],
                   independent_test=False, formal_gate=False, deployable=False, game=False, tuning=False)
    if extra:
        summary['environment'].update(extra)
    return summary


def verify_artifacts(out=OUT):
    """Read-only re-check of every frozen artifact and its SHA."""
    out = Path(out)
    prereg = verify_prereg(out)
    dest = out / 'run'
    records = json.loads((dest / 'records.json').read_text(encoding='utf-8'))
    checked = 0
    for entry in records['events']:
        for key in ('pixels', 'history', 'labels', 'rollout'):
            if file_sha(dest / entry[key]['path']) != entry[key]['sha256']:
                raise ValueError('asset SHA drift: ' + entry['event_id'] + '/' + key)
            checked += 1
    checkpoints = json.loads((dest / 'checkpoints.json').read_text(encoding='utf-8'))
    for name, record in checkpoints.items():
        if file_sha(dest / record['checkpoint_path']) != record['checkpoint_sha256']:
            raise ValueError('checkpoint SHA drift: ' + name)
    baselines = json.loads((dest / 'baselines.json').read_text(encoding='utf-8'))
    closed = 0
    for name in ['B0'] + all_names():
        for split_name in (VALIDATION_SPLIT, TRAIN_SPLIT):
            payload = json.loads((dest / 'closed-loop' / f'{name}-{split_name}.json').read_text(encoding='utf-8'))
            if len(payload['events']) != len(prereg['worlds'][f'{split_name}_events']):
                raise ValueError('closed-loop event count: ' + name + '-' + split_name)
            closed += 1
    stored = json.loads((dest / 'sha256.json').read_text(encoding='utf-8'))
    drift = [path for path, digest in stored.items() if file_sha(dest / path) != digest]
    if drift:
        raise ValueError('stored SHA table drift: ' + ', '.join(drift))
    root = json.loads((out / 'sha256.json').read_text(encoding='utf-8'))
    root_drift = [path for path, digest in root.items() if file_sha(out / path) != digest]
    if root_drift:
        raise ValueError('root SHA table drift: ' + ', '.join(root_drift))
    return dict(artifacts=len(stored), root_artifacts=len(root), assets=checked,
                checkpoints=len(checkpoints), closed_loop_files=closed,
                baseline_rows=baselines['validation']['rows'] + baselines['train']['rows'],
                baseline_passing_events_validation=baselines['validation']['passing_events'],
                baseline_passing_events_train=baselines['train']['passing_events'],
                holdout_filter_valid=baselines['validation']['filter_valid'],
                train_events=len(prereg['worlds']['train_events']),
                validation_events=len(prereg['worlds']['validation_events']),
                excluded_from_supervision=[entry['event_id'] for entry in records['events']
                                           if not entry['teacher_verified']],
                event_files=len(list((dest / 'events').glob('*.json'))),
                criterion_sha256=prereg['criterion_sha256'], identities=len(prereg['identities']))


def run(out=OUT, device='cuda'):
    out = Path(out)
    prereg = verify_prereg(out)
    dest = new_dir(out / 'run')
    try:
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise ValueError('real CUDA required')
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        with _fp32_mode(torch.device(device)):
            if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
                raise AssertionError('TF32 must be disabled')
            timings = {}
            begin = time.perf_counter()
            built = build_assets(dest, event_records(prereg, TRAIN_SPLIT))
            records = write_records(dest, built)
            timings['data_seconds'] = round(time.perf_counter() - begin, 3)
            if records['totals']['events'] != EXPECTED['train_events']:
                raise AssertionError('train event count')
            if len(records['failures']) != EXPECTED['excluded_events']:
                raise AssertionError('the teacher must verify all 14 train events; R19 measured 28/28')
            backbone, _ = load_vision_checkpoint(BASELINE, map_location='cpu')
            backbone.to(device).eval()
            frozen = {key: value.detach().clone() for key, value in backbone.state_dict().items()}
            begin = time.perf_counter()
            data = encode(backbone, records, dest, device)
            torch.cuda.synchronize()
            timings['encoding_seconds'] = round(time.perf_counter() - begin, 3)
            if len(data) != EXPECTED['train_events']:
                raise AssertionError('encoded event count')
            reports, paths = train_all(dest, data, backbone, device, prereg)
            if not all(torch.equal(value, frozen[key]) for key, value in backbone.state_dict().items()):
                raise AssertionError('the frozen backbone changed')
            baselines, closed, seconds = evaluate_all(dest, prereg, backbone, device, reports, paths)
            timings['evaluation_seconds'] = round(seconds, 3)
            timings['training_seconds_total'] = round(sum(reports[name]['optimization_seconds']
                                                          for name in reports), 3)
            write_json(dest / 'timings.json', timings)
            write_events(dest, prereg, records, closed, baselines)
            if not baselines['validation']['filter_valid']:
                print('WARNING the R18 filter failed on the holdout side:', baselines['validation']['passing_events'],
                      flush=True)
            verify_prereg(out)
            summary = assemble(dest)
            write_json(dest / 'summary.json', summary)
            write_json(dest / 'sha256.json',
                       {str(path.relative_to(dest)): file_sha(path) for path in dest.rglob('*') if path.is_file()})
            write_json(out / 'sha256.json',
                       {str(path.relative_to(out)): file_sha(path) for path in out.rglob('*') if path.is_file()})
            return summary
    except BaseException:
        write_json(dest / 'first-failure.json',
                   dict(traceback=traceback.format_exc(),
                        policy='retained; no automatic rerun, overwrite, retune or baseline switch'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preregister', 'run', 'summary', 'verify'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'preregister':
        preregister(args.out)
    elif args.command == 'run':
        run(args.out, args.device)
    elif args.command == 'summary':
        print(json.dumps(assemble(args.out / 'run'), indent=2)[:4000])
    else:
        print(json.dumps(verify_artifacts(args.out), indent=2))
