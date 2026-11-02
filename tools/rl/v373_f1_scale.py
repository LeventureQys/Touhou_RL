"""Round25: is *data scale* the reason the F1 mirror group never learns to go left?

Rounds 18/19/21/22/23/24 all trained on the same six F1 groups (three train, three
validation; six F1 train events).  On the frozen R19 F1 validation set the base
group ends up moving toward the gap while the mirror group moves *screen right* in
every single cell (round24: base 9/9, mirror 0/9).  The round24 diagnosis is that
six events are too few for the auxiliary head to do anything but memorise "which
world is this".  This round asks the cheaper question for the policy itself, on F1
only and with no auxiliary head:

    if the F1 curriculum grows from 6 groups to >= 12 groups, does the mirror
    group's toward-the-gap rate become non-zero, and does the base group survive?

Nothing else is changed.  New worlds are searched with round18's generator and
round18's exact six-baseline + reference-witness criterion, under a larger
pre-registered budget and seed, and every frozen signature of R8/R9/R10/R16/R18/R19
is asserted distinct.  The groups are then re-partitioned by group (mirrors atomic)
into >= 12 train and >= 5 held-out groups while the three frozen R19 F1 validation
groups stay held out, so the round24 yardstick remains an unseen cross-set contrast
for the large arm.  Two control (no-anchor) arms are trained with round20/21's
protocol unchanged: S = the old six F1 groups, L = the twelve-group new train split.
Both arms start from the identical per-seed head, use round21's telegraph renderer
and closed loop, 3 seeds, 200 updates, Adam 1e-3, clip 1.0, batch 1, FP32, both
TF32 flags off, real CUDA.

Research only: no formal gate, no deployable checkpoint, no game execution, no
official labels, no test split.  Exactly one search+train+eval run is executed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from tools.rl import v373_seed_confirm as r16
from tools.rl import v373_spatial_fuse as r22
from tools.rl import v373_telegraph as r21
from tools.rl import v373_valid_curriculum as r20
from tools.rl import v373_world_filter as r18
from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_feature_cache import _fp32_mode
from touhou_rl.tactical_teacher_v2 import hashes as teacher_hashes
from touhou_rl.tactical_teacher_validation import ValidationWorld, make_world
from touhou_rl.vmodels import load_vision_checkpoint

ROOT = Path('Document/Update/Alpha/v3.7.3/evidence')
OUT = ROOT / 'f1scale-round25'
R18_EVIDENCE = ROOT / 'worldfilter-round18'
R19_EVIDENCE = ROOT / 'validteacher-round19'
R21_EVIDENCE = ROOT / 'telegraph-round21'
R24_EVIDENCE = ROOT / 'prelaux-round24'
REPORTS = Path('Document/Update/Alpha/v3.7.3/reports')
MANIFEST = R18_EVIDENCE / 'run/manifest.json'
SPLIT = R19_EVIDENCE / 'run/split.json'
R19_WORLDS = R19_EVIDENCE / 'run/worlds/F1'
R21_RECORDS = R21_EVIDENCE / 'run/records.json'
R24_SUMMARY = R24_EVIDENCE / 'run/summary.json'
BASELINE = r20.BASELINE

PREREG_SCHEMA = 'v373-f1-scale-prereg/1'
SEARCH_SCHEMA = 'v373-f1-scale-search/1'
SPLIT_SCHEMA = 'v373-f1-scale-split/1'
DATA_SCHEMA = 'v373-f1-scale-data/1'
EVENT_SCHEMA = 'v373-f1-scale-event/1'
RESULT_SCHEMA = 'v373-f1-scale-result/1'

FAMILY = 'F1'
SEEDS = r20.SEEDS
UPDATES = r20.UPDATES
LR = r20.LR
CLIP = r20.CLIP
CONDITION = r20.CONDITION
CONTROL = r20.CONTROL                  # 'A': the unchanged tactical_policy_loss, no anchor
ARMS = ('S', 'L')                      # training-set sizes, both on the control loss
ARM_TEXT = {
    'S': 'small: all six R18/R19 F1 groups (12 events), i.e. the whole old F1 data',
    'L': 'large: the new twelve-group train split (24 events)',
}
SEARCH = dict(family=FAMILY, seed=25001, budget=400, target_groups=16, group_prefix='wf25',
              mirror='each accepted group contributes base + mirror, start_x -> 384 - start_x')
SPLIT_FLOORS = dict(train_groups=12, heldout_groups=5)
NEW_GROUPS_SHORTFALL = 8               # below this the brief asks for "all available groups"
R19_FROZEN_TRAIN = ('wf18-F1-002', 'wf18-F1-006', 'wf18-F1-009')
R19_FROZEN_VALIDATION = ('wf18-F1-005', 'wf18-F1-007', 'wf18-F1-010')
SIGNATURE_FIELDS = list(r18.SIGNATURE_FIELDS)
HORIZON = r20.HORIZON
DECISIONS = r20.DECISIONS

# Round24's recorded numbers on exactly the six frozen R19 F1 validation events.  They
# are read back from round24's summary and re-asserted, never quoted on trust.
R24_REFERENCE = dict(
    f1_validation_feasible={'A': [2, 2, 2], 'X': [3, 3, 3]},
    direction=dict(base=dict(toward=6, hold=3, away=0, in_gap=3, cells=9),
                   mirror=dict(toward=3, hold=0, away=6, in_gap=3, cells=9),
                   screen=dict(right=12, left=3, hold=3)),
)


# ---------------------------------------------------------------- plumbing
def write_json(path, value):
    return r20.write_json(path, value)


def write_torch(path, value):
    return r20.write_torch(path, value)


def new_dir(path):
    return r20.new_dir(path)


def scenario_of(spec):
    return json.loads(json.dumps(asdict(ValidationWorld(**spec))))


def scenario_sha(scenario):
    return hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest()


def sig_tuple(value):
    """A fully hashable signature, whether it came from a tuple or from JSON lists."""
    return tuple(sig_tuple(item) for item in value) if isinstance(value, (list, tuple)) else value


# ---------------------------------------------------------------- frozen sources
def load_manifest():
    raw = MANIFEST.read_bytes()
    spec = json.loads(raw)
    if spec.get('schema') != r20.MANIFEST_SCHEMA:
        raise ValueError('R18 manifest schema')
    if spec.get('development_only') is not True or spec.get('test_split') is not False:
        raise ValueError('the R18 set must stay development-only')
    if spec.get('training_steps') or spec.get('teacher_labels'):
        raise ValueError('the R18 manifest must carry no training and no labels')
    worlds = spec['worlds']
    if len(worlds) != 28 or len({w['group'] for w in worlds}) != 14:
        raise ValueError('R18 world/group count')
    if {f: sum(w['family'] == f for w in worlds) for f in ('F1', 'F2', 'F3')} != {'F1': 12, 'F2': 4, 'F3': 12}:
        raise ValueError('R18 family counts')
    index = {w['event_id']: w for w in worlds}
    if len(index) != 28:
        raise ValueError('duplicate R18 event id')
    for world in worlds:
        if world['verdict'] != 'valid' or world['passing_baselines']:
            raise ValueError('R18 world is not valid: ' + world['event_id'])
        reference = world['reference']
        if reference.get('status') != 'feasible' or len(reference.get('actions') or []) != DECISIONS:
            raise ValueError('R18 reference witness missing: ' + world['event_id'])
        if not reference.get('witness_replay'):
            raise ValueError('R18 reference witness did not replay: ' + world['event_id'])
    return dict(spec=spec, sha256=hashlib.sha256(raw).hexdigest(), index=index)


def load_split(manifest):
    raw = SPLIT.read_bytes()
    spec = json.loads(raw)
    if spec.get('schema') != r20.SPLIT_SCHEMA:
        raise ValueError('R19 split schema')
    if spec['source']['manifest_sha256'] != manifest['sha256']:
        raise ValueError('the R19 split is bound to a different R18 manifest')
    groups = spec['groups']
    if len(groups) != 14 or spec['counts']['groups']['validation'] != 7:
        raise ValueError('R19 split shape')
    for group in groups:
        if group['family'] != FAMILY:
            continue
        rows = [manifest['index'][event['event_id']] for event in group['events']]
        if {bool(row['mirror']) for row in rows} != {False, True}:
            raise ValueError('mirror pair missing: ' + group['group'])
        for row in rows:
            if json.loads(json.dumps(r16.signature(row['scenario']))) != group['signature']:
                raise ValueError('R19 signature drift: ' + row['event_id'])
    return dict(spec=spec, sha256=hashlib.sha256(raw).hexdigest(), groups=groups)


def r19_frozen_groups(split):
    table = {group['group']: group for group in split['groups']}
    rows = {}
    for name, ids in (('train', R19_FROZEN_TRAIN), ('validation', R19_FROZEN_VALIDATION)):
        for group_id in ids:
            if group_id not in table or table[group_id]['split'] != name:
                raise ValueError('the frozen R19 F1 split changed: ' + group_id)
            rows[group_id] = table[group_id]
    if len(rows) != 6:
        raise ValueError('six frozen F1 groups required')
    return rows


def frozen_signature_union():
    """R8 dev/val, R9, R10, R16, R18 and R19 (R19 re-uses R18) as one signature set."""
    old = r18.existing_signatures()
    census = {name: len({r18.signature(spec) for spec in rows}) for name, rows in old.items()}
    census['r10_subset_of_r8'] = {r18.signature(s) for s in old['r10']} <= {r18.signature(s) for s in old['r8']}
    union = set().union(*({r18.signature(spec) for spec in rows} for rows in old.values()))
    manifest = load_manifest()
    split = load_split(manifest)
    r18_sigs = {r18.signature(world['scenario']) for world in manifest['spec']['worlds']}
    if len(r18_sigs) != 14:
        raise ValueError('R18 groups must have distinct signatures')
    split_sigs = {sig_tuple(group['signature']) for group in split['groups'] if group['family'] == FAMILY}
    if split_sigs != {sig for sig in r18_sigs if sig[0] == FAMILY}:
        raise ValueError('R19 F1 signatures must be exactly the R18 F1 signatures')
    if r18_sigs & union:
        raise ValueError('a R18 world duplicates a frozen R8/R9/R10/R16 signature')
    union |= r18_sigs
    census['r18_worlds'] = len(r18_sigs)
    census['r19_f1_groups'] = len(split_sigs)
    census['frozen_union'] = len(union)
    return union, census


def r19_teacher_actions(event_id):
    path = R19_WORLDS / (event_id + '.json')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema') != 'v373-valid-teacher-world/1':
        raise ValueError('R19 world schema: ' + event_id)
    two = payload['teachers']['teacher_v2']
    actions = list(two['actions'])
    if two.get('verify') != 'feasible' or len(actions) != DECISIONS:
        raise ValueError('R19 teacher_v2 record is not a feasible 80-action route: ' + event_id)
    if hashlib.sha256(json.dumps(actions).encode()).hexdigest() != two['actions_sha256']:
        raise ValueError('R19 teacher action digest drift: ' + event_id)
    return actions


def r21_assets():
    records = json.loads(R21_RECORDS.read_text(encoding='utf-8'))
    table = {entry['event_id']: entry for entry in records['events']}
    if len(table) != 14 or records['totals']['verified'] != 14:
        raise ValueError('R21 asset records')
    return table


def r24_reference():
    summary = json.loads(R24_SUMMARY.read_text(encoding='utf-8'))
    feasible = {arm: [summary['validation']['results'][r20.name_of(arm, seed)]['F1']['feasible']
                      for seed in SEEDS] for arm in ('A', 'X')}
    if feasible != R24_REFERENCE['f1_validation_feasible']:
        raise ValueError('round24 F1 feasible reference drift')
    groups = summary['validation']['direction']['A']['groups']
    base, mirror = groups['base'], groups['mirror']
    got = dict(base=dict(toward=base['toward'], hold=base['hold'], away=base['away'],
                         in_gap=base['in_gap'], cells=base['cells']),
               mirror=dict(toward=mirror['toward'], hold=mirror['hold'], away=mirror['away'],
                           in_gap=mirror['in_gap'], cells=mirror['cells']),
               screen=dict(right=base['screen_right'] + mirror['screen_right'],
                           left=base['screen_left'] + mirror['screen_left'],
                           hold=base['screen_hold'] + mirror['screen_hold']))
    if got != R24_REFERENCE['direction']:
        raise ValueError('round24 direction reference drift: ' + json.dumps(got))
    if summary['judgement']['K1']['control'] != [2, 2, 2]:
        raise ValueError('round24 K1 control reference drift')
    if summary['validation']['holdout']['filter_valid'] is not True:
        raise ValueError('round24 holdout filter reference drift')
    return dict(summary_sha256=file_sha(R24_SUMMARY),
                f1_validation_feasible=feasible, direction=got,
                note='round24 arm A on exactly the six R19-frozen F1 validation events: feasible 2/2/2, '
                     'base 6/9 toward, mirror 3/9 toward, 3/3 parked, screen 12 right / 3 left / 3 hold')


# ---------------------------------------------------------------- search
def new_group_specs(candidate, geometry):
    group = f"{SEARCH['group_prefix']}-{FAMILY}-{candidate:03d}"
    base = dict(event_id=group + '-m0', template_group=group, seed=25000 + candidate,
                split='development', mirror=False, depletion_pressure=False)
    base.update({key: tuple(value) if isinstance(value, list) else value for key, value in geometry.items()})
    rows = []
    for mirror in (False, True):
        spec = dict(base, mirror=mirror, event_id=group + f'-m{int(mirror)}')
        spec['start_x'] = 384. - float(geometry['start_x']) if mirror else float(geometry['start_x'])
        rows.append(spec)
    return rows


def geometry_out(spec):
    keys = ('family', 'start_x', 'coverage', 'gap', 'wall_frame', 'internal_gap', 'origin',
            'fire_interval', 'bullet_speed', 'burst', 'window', 'threshold', 'goal')
    return {key: spec[key] for key in keys if key in spec}


def brief_evidence(evidence):
    keep = ('feasible', 'survived', 'first_hit')
    return dict(verdict=evidence['verdict'], reasons=evidence['reasons'],
                passing_baselines=evidence['passing_baselines'],
                baselines={name: {key: evidence['baselines'][name][key] for key in keep}
                           for name in r18.BASELINES},
                weak={key: evidence['weak'][key] for key in keep},
                reference=dict(status=evidence['reference'].get('status'),
                               checked=evidence['reference'].get('checked'),
                               witness_replay=evidence['reference'].get('witness_replay')),
                any_constant_feasible=evidence.get('any_constant_feasible'))


def assert_new_disjoint(accepted):
    frozen, census = frozen_signature_union()
    signatures = []
    for row in accepted:
        sig = sig_tuple(row['signature'])
        if sig in frozen:
            raise ValueError('new world duplicates a frozen signature: ' + row['group'])
        if sig in signatures:
            raise ValueError('accepted groups must have distinct signatures')
        signatures.append(sig)
    return dict(census, new_groups=len(accepted), new_events=2 * len(accepted), duplicates=0)


def search_new_groups():
    """Round18's generator and round18's per-world criterion at a larger frozen budget."""
    if SEARCH['family'] != FAMILY:
        raise ValueError('this round is F1 only')
    rng = random.Random(SEARCH['seed'])
    frozen, census = frozen_signature_union()
    seen, accepted, discarded = set(), [], []
    drawn = 0
    for candidate in range(SEARCH['budget']):
        if len(accepted) >= SEARCH['target_groups']:
            break
        drawn += 1
        geometry, reason = r18.draw_geometry(FAMILY, rng)
        if geometry is None:
            discarded.append(dict(candidate=candidate, geometry=None, verdict='discarded', reasons=[reason]))
            continue
        specs = new_group_specs(candidate, geometry)
        if r18.signature(specs[0]) != r18.signature(specs[1]):
            raise AssertionError('mirror pair signature mismatch')
        sig = r18.signature(specs[0])
        if sig in frozen or sig in seen:
            discarded.append(dict(candidate=candidate, geometry=geometry_out(specs[0]), verdict='discarded',
                                  reasons=['duplicate_existing_or_accepted_signature']))
            continue
        seen.add(sig)
        base = r18.evaluate_world(make_world(specs[0]))
        if base['verdict'] != 'valid':
            discarded.append(dict(candidate=candidate, geometry=geometry_out(specs[0]),
                                  verdict=base['verdict'], reasons=['base:' + row for row in base['reasons']],
                                  base=brief_evidence(base), mirror='not_evaluated:base_rejected'))
            continue
        mirror = r18.evaluate_world(make_world(specs[1]))
        if mirror['verdict'] != 'valid':
            discarded.append(dict(candidate=candidate, geometry=geometry_out(specs[0]),
                                  verdict=mirror['verdict'],
                                  reasons=['mirror:' + row for row in mirror['reasons']],
                                  base=brief_evidence(base), mirror=brief_evidence(mirror)))
            continue
        accepted.append(dict(candidate=candidate, group=specs[0]['template_group'], signature=list(sig),
                             geometry=geometry_out(specs[0]), specs=specs,
                             base=r18.compact(base), mirror=r18.compact(mirror),
                             scenarios=[scenario_of(spec) for spec in specs]))
    counts = dict(
        drawn=drawn, budget=SEARCH['budget'], target_groups=SEARCH['target_groups'],
        accepted_groups=len(accepted), accepted_events=2 * len(accepted),
        discarded=len(discarded),
        discarded_invalid=sum(row['verdict'] == 'invalid' for row in discarded),
        discarded_unknown=sum(row['verdict'] == 'unknown' for row in discarded),
        discarded_duplicate=sum('duplicate' in ' '.join(row['reasons']) for row in discarded),
        discarded_unusable_draw=sum(row['verdict'] == 'discarded' and
                                    'duplicate' not in ' '.join(row['reasons']) for row in discarded),
        target_met=len(accepted) >= SEARCH['target_groups'],
        budget_exhausted=len(accepted) < SEARCH['target_groups'] and drawn >= SEARCH['budget'])
    return dict(schema=SEARCH_SCHEMA, search=copy.deepcopy(SEARCH),
                generator='tools.rl.v373_world_filter.draw_geometry (unchanged, F1 domain)',
                criterion='tools.rl.v373_world_filter.evaluate_world (unchanged: constant hold/right/left, the '
                          '16-decision alternation, the seeded uniform random policy and the teacher-first weak '
                          'baseline must all fail, and the independent solve() must return a complete 80-action '
                          'witness that replays without early termination; unknown is discarded; no threshold '
                          'loosened)',
                seed=SEARCH['seed'], budget=SEARCH['budget'], target_groups=SEARCH['target_groups'],
                drawn=drawn, counts=counts, frozen_census=census,
                accepted=[dict(candidate=row['candidate'], group=row['group'], signature=row['signature'],
                               geometry=row['geometry'], events=[row['specs'][0]['event_id'],
                                                                 row['specs'][1]['event_id']],
                               base=row['base'], mirror=row['mirror']) for row in accepted],
                discarded=discarded,
                world_ids=[spec['event_id'] for row in accepted for spec in row['specs']],
                scenarios={spec['event_id']: row['scenarios'][index]
                           for row in accepted for index, spec in enumerate(row['specs'])})


def loaded_search(payload):
    """Turn the stored search JSON back into group dicts (split and verify use this)."""
    groups = []
    for row in payload['accepted']:
        specs = []
        for index, mirror in enumerate((False, True)):
            event_id = row['events'][index]
            specs.append(dict(payload['scenarios'][event_id], event_id=event_id, mirror=mirror))
        if r18.signature(specs[0]) != r18.signature(specs[1]):
            raise ValueError('stored mirror pair signature mismatch: ' + row['group'])
        if sig_tuple(r18.signature(specs[0])) != sig_tuple(row['signature']):
            raise ValueError('stored signature drift: ' + row['group'])
        groups.append(dict(group=row['group'], family=FAMILY, source='wf25', candidate=row['candidate'],
                           signature=row['signature'], events=list(row['events']),
                           scenarios={spec['event_id']: scenario_of(spec) for spec in specs}, specs=specs))
    return groups


# ---------------------------------------------------------------- split
def old_groups(manifest, split):
    rows = []
    for group in split['groups']:
        if group['family'] != FAMILY:
            continue
        scenarios = {event['event_id']: manifest['index'][event['event_id']]['scenario']
                     for event in group['events']}
        rows.append(dict(group=group['group'], family=FAMILY, source='r18',
                         signature=list(group['signature']),
                         events=[event['event_id'] for event in group['events']],
                         scenarios=scenarios,
                         specs=[manifest['index'][event['event_id']]['scenario'] for event in group['events']]))
    rows.sort(key=lambda row: row['group'])
    if [row['group'] for row in rows] != sorted(R19_FROZEN_TRAIN + R19_FROZEN_VALIDATION):
        raise ValueError('the frozen F1 group order changed')
    return rows


def partition(groups):
    """Alternate by group, keep the frozen R19 validation groups held out, then apply the floors."""
    ordered = [row['group'] for row in groups]
    forced = list(R19_FROZEN_VALIDATION)
    if any(group_id not in ordered for group_id in forced):
        raise ValueError('a frozen R19 validation group is missing')
    new_groups = [row['group'] for row in groups if row['source'] == 'wf25']
    flips, shortfall = [], None
    if len(new_groups) < NEW_GROUPS_SHORTFALL:
        assignment = {group_id: ('heldout' if group_id in forced else 'train') for group_id in ordered}
        shortfall = 'fewer_than_eight_new_groups:train_uses_every_group_except_the_frozen_heldout'
    else:
        assignment = {group_id: ('train' if index % 2 == 0 else 'heldout')
                      for index, group_id in enumerate(ordered)}
        for group_id in forced:
            if assignment[group_id] != 'heldout':
                assignment[group_id] = 'heldout'
                flips.append(dict(group=group_id, change='heldout_forced', reason='frozen R19 validation group'))
        while sum(value == 'train' for value in assignment.values()) < SPLIT_FLOORS['train_groups'] and \
                sum(value == 'heldout' for value in assignment.values()) > SPLIT_FLOORS['heldout_groups']:
            pick = next(group_id for group_id in reversed(ordered)
                        if assignment[group_id] == 'heldout' and group_id not in forced)
            assignment[pick] = 'train'
            flips.append(dict(group=pick, change='heldout_to_train', reason='train floor 12'))
        while sum(value == 'heldout' for value in assignment.values()) < SPLIT_FLOORS['heldout_groups']:
            pick = next(group_id for group_id in reversed(ordered) if assignment[group_id] == 'train')
            assignment[pick] = 'heldout'
            flips.append(dict(group=pick, change='train_to_heldout', reason='held-out floor 5'))
    train = [group_id for group_id in ordered if assignment[group_id] == 'train']
    heldout = [group_id for group_id in ordered if assignment[group_id] == 'heldout']
    new_heldout = [group_id for group_id in heldout if group_id in new_groups]
    floors = dict(
        train_groups=len(train) >= SPLIT_FLOORS['train_groups'],
        heldout_groups=len(heldout) >= SPLIT_FLOORS['heldout_groups'],
        forced_heldout=all(group_id in heldout for group_id in forced),
        new_heldout_groups=len(new_heldout),
        new_heldout_floor=len(new_heldout) >= SPLIT_FLOORS['heldout_groups'],
        new_groups_floor=len(new_groups) >= NEW_GROUPS_SHORTFALL,
        disjoint=not (set(train) & set(heldout)),
        covers_all=set(train) | set(heldout) == set(ordered))
    return dict(ordered=ordered, assignment=assignment, train=train, heldout=heldout,
                new_heldout=new_heldout, forced=forced, flips=flips, shortfall=shortfall, floors=floors,
                counts=dict(total_groups=len(ordered), train_groups=len(train), heldout_groups=len(heldout),
                            new_groups=len(new_groups), new_heldout_groups=len(new_heldout),
                            train_events=2 * len(train), heldout_events=2 * len(heldout),
                            new_heldout_events=2 * len(new_heldout)),
                rule='groups are ordered R18/R19 F1 order then acceptance order; index even -> train, odd -> '
                     'held-out; the frozen R19 validation groups are forced held-out; while train < 12 and '
                     'held-out > 5 the last held-out group is flipped to train (and, if held-out < 5, the last '
                     'train group is flipped back).  Mirrors are atomic: a group never crosses the split')


def split_payload(partitioned, groups):
    table = {row['group']: row for row in groups}
    rows = []
    for index, group_id in enumerate(partitioned['ordered']):
        group = table[group_id]
        rows.append(dict(group=group_id, family=group['family'], source=group['source'], index=index,
                         split=partitioned['assignment'][group_id], signature=group['signature'],
                         events=[dict(event_id=event_id,
                                      mirror=bool(group['scenarios'][event_id]['mirror']),
                                      start_x=float(group['scenarios'][event_id]['start_x']),
                                      scenario_sha256=scenario_sha(group['scenarios'][event_id]),
                                      scenario=group['scenarios'][event_id])
                                 for event_id in group['events']]))
    payload = dict(schema=SPLIT_SCHEMA, counts=partitioned['counts'], floors=partitioned['floors'],
                   forced_heldout=partitioned['forced'], new_heldout=partitioned['new_heldout'],
                   flips=partitioned['flips'], shortfall=partitioned['shortfall'], rule=partitioned['rule'],
                   groups=rows,
                   train_events=[event['event_id'] for row in rows if row['split'] == 'train'
                                 for event in row['events']],
                   heldout_events=[event['event_id'] for row in rows if row['split'] == 'heldout'
                                   for event in row['events']])
    payload['new_heldout_events'] = [event['event_id'] for row in rows
                                     if row['split'] == 'heldout' and row['source'] == 'wf25'
                                     for event in row['events']]
    payload['r19_frozen_events'] = [event['event_id'] for row in rows
                                    if row['group'] in R19_FROZEN_VALIDATION for event in row['events']]
    return payload


def split_assertions(payload, arm_events):
    errors = []
    groups = {row['group']: row for row in payload['groups']}
    train = {row['group'] for row in payload['groups'] if row['split'] == 'train'}
    heldout = {row['group'] for row in payload['groups'] if row['split'] == 'heldout'}
    if train & heldout:
        errors.append('a group is in both splits')
    if train | heldout != set(groups):
        errors.append('a group is in neither split')
    if not set(R19_FROZEN_VALIDATION) <= heldout:
        errors.append('a frozen R19 validation group is not held out')
    for row in payload['groups']:
        if len(row['events']) != 2 or {event['mirror'] for event in row['events']} != {False, True}:
            errors.append('a group does not hold exactly one base and one mirror: ' + row['group'])
    l_events = {event['event_id'] for row in payload['groups'] if row['group'] in train for event in row['events']}
    s_events = {event['event_id'] for row in payload['groups']
                if row['group'] in R19_FROZEN_TRAIN + R19_FROZEN_VALIDATION for event in row['events']}
    eval_events = set(payload['r19_frozen_events']) | set(payload['new_heldout_events'])
    if arm_events['S'] != s_events:
        errors.append('the small arm is not exactly the six frozen F1 groups')
    if arm_events['L'] != l_events:
        errors.append('the large arm is not exactly the train split')
    if arm_events['S'] & set(payload['new_heldout_events']):
        errors.append('the small arm trains on the new held-out set')
    if arm_events['L'] & eval_events:
        errors.append('the large arm trains on an evaluation event')
    if l_events & eval_events:
        errors.append('the train split leaks into an evaluation set')
    if set(payload['r19_frozen_events']) & set(payload['new_heldout_events']):
        errors.append('the two evaluation sets overlap')
    return dict(errors=errors, ok=not errors, train_groups=len(train), heldout_groups=len(heldout),
                new_heldout_groups=len(payload['new_heldout']), train_events=len(l_events),
                small_events=len(s_events), eval_events=len(eval_events),
                small_arm_trained_on_r19_frozen=bool(s_events & set(payload['r19_frozen_events'])),
                large_arm_trains_on_r19_frozen=bool(l_events & set(payload['r19_frozen_events'])),
                train_heldout_disjoint=not (l_events & eval_events),
                note='the small arm deliberately trains on all six old F1 groups, which includes the three frozen '
                     'R19 validation groups; its r19-frozen reading is therefore a seen reading and is never used '
                     'for a held-out claim')


def event_table(groups):
    rows = []
    for group in groups:
        for event_id in group['events']:
            scenario = group['scenarios'][event_id]
            rows.append(dict(event_id=event_id, family=group['family'], group=group['group'],
                             mirror=bool(scenario['mirror']), scenario=scenario,
                             scenario_sha256=scenario_sha(scenario)))
    return rows


# ---------------------------------------------------------------- assets
def build_assets(dest, events, assets21):
    """Render every event with round21's renderer; bind byte-identity for the old train events."""
    r20.new_dir(dest / 'assets')
    built = []
    for event in events:
        payload = r21.render_event(event)
        world = payload['world']
        window = r21.telegraph_window(world)
        if window is None:
            raise ValueError('F1 events only: ' + event['event_id'])
        expected = [frame for frame in r21.DECISION_FRAMES if window[0] <= frame < window[1]]
        freeze = payload['freeze']
        confined = (freeze['changed_decision_frames'] == len(expected) and
                    freeze['first_changed_frame'] == (expected[0] if expected else None) and
                    freeze['last_changed_frame'] == (expected[-1] if expected else None) and
                    freeze['preview'] == [max(0, window[0]), window[1]])
        if not confined:
            raise AssertionError('the telegraph must change exactly the preview rows: ' + event['event_id'])
        actions = list(payload['actions'])
        frozen_group = event['group'] in R19_FROZEN_TRAIN + R19_FROZEN_VALIDATION
        if frozen_group and actions != r19_teacher_actions(event['event_id']):
            raise AssertionError('the teacher trajectory deviates from the frozen R19 record: ' + event['event_id'])
        directory = r20.new_dir(dest / 'assets' / event['event_id'])
        pixels, history, labels = directory / 'pixels.npz', directory / 'history.pt', directory / 'labels.pt'
        r20.write_npz(pixels, payload['stored'])
        r20.write_torch(history, payload['history'])
        r20.write_torch(labels, payload['labels'])
        reproduced = None
        if event['event_id'] in assets21:
            reference = assets21[event['event_id']]
            reproduced = {key: file_sha({'pixels': pixels, 'history': history, 'labels': labels}[key]) ==
                          reference[key]['sha256'] for key in ('pixels', 'history', 'labels')}
            if not all(reproduced.values()):
                raise AssertionError('the reproduced asset bytes differ from round21: ' + event['event_id'])
        rollout = directory / 'rollout.json'
        row = dict(schema=EVENT_SCHEMA, event_id=event['event_id'], family=event['family'],
                   group=event['group'], split=event['split'], mirror=bool(event['mirror']),
                   scenario=event['scenario'], scenario_sha256=event['scenario_sha256'],
                   teacher='TacticalTeacherV2 (unmodified)', teacher_source=teacher_hashes(),
                   renderer='tools.rl.v373_telegraph.render_state_telegraph (unchanged)',
                   renderer_source_sha256=file_sha(Path(r21.__file__)),
                   telegraph=dict(lead=r21.TELEGRAPH_LEAD, fill=list(r21.TELEGRAPH_FILL),
                                  edge=list(r21.TELEGRAPH_EDGE), edge_width=r21.EDGE_WIDTH,
                                  window=freeze['preview'], spans=freeze['spans'], gap=freeze['gap']),
                   render_freeze=freeze,
                   telegraph_confinement=dict(expected_decision_frames=len(expected), confined=confined,
                                              note='the preview must change exactly the decision rows whose '
                                                   'stacked frames contain a frame of [wall_frame-45, wall_frame); '
                                                   'checked by count, first and last frame'),
                   containment=dict(round19_teacher_actions_equal=True if frozen_group else None,
                                    round21_asset_bytes_equal=reproduced,
                                    round21_assets=event['event_id'] in assets21),
                   actions=actions, actions_sha256=r21.actions_digest(actions),
                   switches=payload['switches'], distinct_actions=payload['distinct_actions'],
                   action_counts=payload['action_counts'], decisions=len(actions), horizon=HORIZON,
                   verify_verdict=payload['verdict'], teacher_verified=bool(payload['verified']),
                   hud_constant=payload['hud_constant'], depth_frames=r20.depth_frames(),
                   region_sizes=r21.SIZES)
        r20.write_json(rollout, row)
        raw_bytes = sum(int(array.nbytes) for array in payload['stored'].values())
        built.append(dict(event=event, payload=payload, refs=dict(
            pixels=dict(path=f"assets/{event['event_id']}/pixels.npz", sha256=file_sha(pixels),
                        bytes=pixels.stat().st_size, raw_bytes=raw_bytes,
                        shapes={name: list(array.shape) for name, array in payload['stored'].items()}),
            history=dict(path=f"assets/{event['event_id']}/history.pt", sha256=file_sha(history),
                         bytes=history.stat().st_size),
            labels=dict(path=f"assets/{event['event_id']}/labels.pt", sha256=file_sha(labels),
                        bytes=labels.stat().st_size),
            rollout=dict(path=f"assets/{event['event_id']}/rollout.json", sha256=file_sha(rollout),
                         bytes=rollout.stat().st_size))))
        print('rendered', event['event_id'], 'verdict', payload['verdict'], 'preview', freeze['preview'],
              'changed_rows', freeze['changed_decision_frames'], 'round21_bytes_equal', reproduced, flush=True)
    return built


def write_records(dest, built):
    entries, failures = [], []
    for item in built:
        event, payload = item['event'], item['payload']
        containment = json.loads((dest / item['refs']['rollout']['path']).read_text(encoding='utf-8'))['containment']
        entries.append(dict(event_id=event['event_id'], family=event['family'], group=event['group'],
                            split=event['split'], mirror=bool(event['mirror']), rows=DECISIONS,
                            horizon=HORIZON, logical_frames=HORIZON,
                            teacher_verified=payload['verified'], verify_verdict=payload['verdict'],
                            scenario_sha256=event['scenario_sha256'], switches=payload['switches'],
                            distinct_actions=payload['distinct_actions'],
                            action_counts=payload['action_counts'], render_freeze=payload['freeze'],
                            containment=containment, **item['refs']))
        if not payload['verified']:
            failures.append(dict(event_id=event['event_id'], verdict=payload['verdict']))
    records = dict(schema=DATA_SCHEMA, root='assets', events=entries, failures=failures,
                   renderer='tools.rl.v373_telegraph.render_state_telegraph (unchanged)',
                   renderer_source_sha256=file_sha(Path(r21.__file__)),
                   teacher='touhou_rl.tactical_teacher_v2 (unmodified)', teacher_source=teacher_hashes(),
                   telegraph=dict(lead=r21.TELEGRAPH_LEAD, fill=list(r21.TELEGRAPH_FILL),
                                  edge=list(r21.TELEGRAPH_EDGE), edge_width=r21.EDGE_WIDTH,
                                  window='[wall_frame-45, wall_frame)',
                                  lethal_window='[wall_frame, wall_frame+12)'),
                   storage=r20.STORAGE,
                   history_semantics='round20/21 verbatim: the actually executed candidate, previous_hold 2, '
                                     'frame_delta 2, option BOS3, target_valid false',
                   label_semantics='round20/21 verbatim: the one-hot of the single successful teacher route first '
                                   'action; action_valid = teacher_verified',
                   totals=dict(events=len(entries), verified=len(entries) - len(failures), failed=len(failures),
                               rows=len(entries) * DECISIONS,
                               bytes=sum(e['pixels']['bytes'] + e['history']['bytes'] + e['labels']['bytes'] +
                                         e['rollout']['bytes'] for e in entries),
                               changed_pixels=sum(e['render_freeze']['changed_pixels'] for e in entries)))
    r20.write_json(dest / 'records.json', records)
    return records


# ---------------------------------------------------------------- training
def train_pair(dest, datasets, backbone, device, prereg):
    """Both arms use the frozen round20 control loss; only the training set differs."""
    r20.new_dir(dest / 'train')
    reports, paths = {}, {}
    for seed in SEEDS:
        per_arm = {}
        for arm in ARMS:
            name = f'{arm}-s{seed}'
            data = datasets[arm]
            supervised = [entry for entry in data if entry['entry']['teacher_verified']]
            excluded = [entry['entry']['event_id'] for entry in data if not entry['entry']['teacher_verified']]
            if not supervised:
                raise ValueError('an arm has no supervised event: ' + name)
            directory = r20.new_dir(dest / 'train' / arm / name)
            model = r16.fresh_policy(backbone, seed).to(device)
            initial_sha = r16.state_sha(model.heads.state_dict())
            if initial_sha != prereg['initialization']['per_seed'][str(seed)]:
                raise AssertionError('the initial head must reproduce the pre-registration: ' + name)
            per_arm[arm] = initial_sha
            r20.write_json(directory / 'initialization.json', dict(
                head_state_sha256=initial_sha, seed=seed, training_set=arm, loss_arm=CONTROL, anchor_weight=0.,
                shared_config='both arms are N1 on the control loss; only the training set differs',
                supervised_events=len(supervised), excluded_events=excluded, updates=UPDATES,
                events_per_update=len(supervised) / UPDATES, epochs=round(UPDATES / len(supervised), 6)))
            parameters = [value for value in model.parameters() if value.requires_grad]
            initial_offline = r20.offline(model, supervised)
            optimizer = torch.optim.Adam(parameters, lr=LR)
            losses = []
            begin = time.perf_counter()
            for step in range(UPDATES):
                model.train()
                entry = supervised[step % len(supervised)]
                output = r20.infer(model, entry)
                labels = {key: value[None] for key, value in entry['labels'].items()}
                head = r20.arm_loss(CONTROL, output, labels)
                if not head['parts']['has_supervision']:
                    raise AssertionError('every update must be supervised')
                optimizer.zero_grad(set_to_none=True)
                head['loss'].backward()
                torch.nn.utils.clip_grad_norm_(parameters, CLIP)
                optimizer.step()
                losses.append(float(head['loss'].detach()))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            final_offline = r20.offline(model, supervised)
            final_sha = r16.state_sha(model.heads.state_dict())
            path = directory / 'research-heads.pt'
            r20.write_torch(path, dict(schema=r20.SCHEMA, arm=CONTROL, training_set=arm, seed=seed,
                                       config=asdict(model.tactical_config),
                                       baseline_sha256=file_sha(BASELINE),
                                       final_head_state_sha256=final_sha, anchor_weight=0.,
                                       dataset=dict(groups=sorted({entry['entry']['group'] for entry in supervised}),
                                                    events=[entry['entry']['event_id'] for entry in supervised],
                                                    events_available=len(data), updates=UPDATES,
                                                    epochs=round(UPDATES / len(supervised), 6)),
                                       heads={key: value.detach().cpu()
                                              for key, value in model.heads.state_dict().items()}))
            report = dict(arm=arm, training_set=arm, seed=seed, name=name,
                          initial_head_state_sha256=initial_sha, final_head_state_sha256=final_sha,
                          checkpoint_sha256=file_sha(path), checkpoint_path=str(path.relative_to(dest)),
                          updates=UPDATES, loss_arm=CONTROL, anchor_weight=0., lr=LR, clip=CLIP,
                          losses=losses, loss_at_step0=losses[0], loss_final=losses[-1],
                          optimization_seconds=round(seconds, 3),
                          supervised_events=[entry['entry']['event_id'] for entry in supervised],
                          excluded_events=excluded, events_per_update=len(supervised) / UPDATES,
                          epochs=round(UPDATES / len(supervised), 6),
                          parameters=dict(active=sum(value.numel() for value in parameters),
                                          heads_total=sum(value.numel() for value in model.heads.parameters()),
                                          total=sum(value.numel() for value in model.parameters())),
                          initial_offline=initial_offline['summary'], final_offline=final_offline['summary'])
            r20.write_json(directory / 'training.json', report)
            reports[name] = report
            paths[name] = path
            print('trained', name, 'set', arm, 'init', initial_sha[:12], 'final', final_sha[:12],
                  'loss0', round(losses[0], 4), 'lossN', round(losses[-1], 4), f'{seconds:.2f}s', flush=True)
        if len(set(per_arm.values())) != 1:
            raise AssertionError('both arms must start from the identical head for seed ' + str(seed))
    if len(list((dest / 'train').rglob('research-heads.pt'))) != len(ARMS) * len(SEEDS):
        raise AssertionError('one checkpoint per arm and seed required')
    for name, path in paths.items():
        if file_sha(path) != reports[name]['checkpoint_sha256']:
            raise AssertionError('checkpoint SHA drift: ' + name)
    r20.write_json(dest / 'checkpoints.json',
                   {name: {key: value for key, value in reports[name].items()
                           if key not in ('losses', 'initial_offline', 'final_offline')}
                    for name in reports})
    return reports, paths


def load_network(backbone, path, device, arm, seed, expected_sha):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != r20.SCHEMA:
        raise ValueError('research checkpoint schema')
    if payload.get('arm') != CONTROL:
        raise ValueError('this round only trains the control loss')
    if payload.get('training_set') != arm or payload.get('seed') != seed:
        raise ValueError('checkpoint does not belong to this arm/seed')
    if payload.get('anchor_weight') != 0.:
        raise ValueError('the control arm must record anchor_weight 0')
    return r20.Network(backbone, path, device, expected_sha)


# ---------------------------------------------------------------- evaluation
def evaluate(dest, sets, backbone, device, reports, paths):
    r20.new_dir(dest / 'closed-loop')
    begin = time.perf_counter()
    baselines = {name: r20.six_baselines(events) for name, events in sets.items()}
    for name, payload in baselines.items():
        if not payload['filter_valid']:
            raise AssertionError('the frozen R18 filter is not valid on the held-out set: ' + name)
    r20.write_json(dest / 'baselines.json', baselines)
    closed = {}
    for set_name, events in sets.items():
        rows = r21.closed_loop(events, r20.BaselineNetwork(backbone, device), 'B0', 'B0')
        payload = dict(network='B0', set=set_name, condition=CONDITION,
                       renderer='tools.rl.v373_telegraph.render_state_telegraph', events=rows,
                       families=r16.families(rows), behaviour=r20.behaviour(rows))
        r20.write_json(dest / 'closed-loop' / f'B0-{set_name}.json', payload)
        closed[('B0', set_name)] = payload
    for arm in ARMS:
        for seed in SEEDS:
            name = f'{arm}-s{seed}'
            network = load_network(backbone, paths[name], device, arm, seed,
                                   reports[name]['final_head_state_sha256'])
            for set_name, events in sets.items():
                rows = r21.closed_loop(events, network, 'N1', name)
                payload = dict(network=name, set=set_name, condition=CONDITION,
                               renderer='tools.rl.v373_telegraph.render_state_telegraph', events=rows,
                               families=r16.families(rows), behaviour=r20.behaviour(rows),
                               heads_sha256=network.heads_sha256)
                r20.write_json(dest / 'closed-loop' / f'{name}-{set_name}.json', payload)
                closed[(name, set_name)] = payload
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    if len(closed) != (1 + len(ARMS) * len(SEEDS)) * len(sets):
        raise AssertionError('closed-loop file count')
    gaps = {}
    for set_name, events in sets.items():
        rows = {}
        for name in ['B0'] + [f'{arm}-s{seed}' for arm in ARMS for seed in SEEDS]:
            by_event = {row['event_id']: row for row in closed[(name, set_name)]['events']}
            for entry in events:
                world = make_world(entry['scenario'])
                rows[f'{name}/{entry["event_id"]}'] = r22.cell_diagnostic(
                    world, by_event[entry['event_id']], name, set_name)
        gaps[set_name] = rows
    r20.write_json(dest / 'f1-gap.json', dict(
        notes='per (network, F1 held-out event): the canonical x series is rebuilt from advance semantics, so the '
              'odd logical frame is included; in_gap_at_wall is the position on wall_frame-1, the last frame '
              'before the band becomes lethal; toward/hold/away is the canonical direction against the gap centre',
        sets=gaps))
    return baselines, closed, gaps, seconds


def screen_share(rows, networks):
    subset = [row for row in rows.values() if row['network'] in networks]
    counts = dict(cells=len(subset),
                  right=sum(1 for row in subset if row['screen_direction'] == 'right'),
                  left=sum(1 for row in subset if row['screen_direction'] == 'left'),
                  hold=sum(1 for row in subset if row['screen_direction'] == 'hold'))
    counts['right_share'] = round(counts['right'] / len(subset), 6) if subset else None
    return counts


def screen_need(row):
    """The screen direction the gap requires from the start, and therefore the prior-free target.

    `toward` in round22's sense is canonical, which equals the screen direction on base worlds and
    flips on mirror worlds.  The frozen policy prior is "screen right", so the cells where it is
    WRONG are exactly the `need_left` cells.  On round19's frozen F1 validation set those are the
    three mirror events; on a re-partitioned held-out set that mixes gap sides they can also be
    base events, which is why the raw mirror toward rate is not a flip measure there.
    """
    gap = row.get('gap')
    if not gap:
        return None
    centre = (float(gap[0]) + float(gap[1])) / 2.
    start = float(row['start_canonical'])
    if abs(centre - start) < 1e-9:
        return 'hold'
    need = 'right' if centre > start else 'left'
    return need if not row['mirror'] else ('left' if need == 'right' else 'right')


def conditioned(rows, networks):
    """Flip/keep counts conditioned on the direction the gap requires (screen coordinates)."""
    subset = [row for row in rows.values() if row['network'] in networks]
    left = [row for row in subset if screen_need(row) == 'left']
    right = [row for row in subset if screen_need(row) == 'right']
    left_base = [row for row in left if not row['mirror']]
    left_mirror = [row for row in left if row['mirror']]
    right_base = [row for row in right if not row['mirror']]
    right_mirror = [row for row in right if row['mirror']]
    return dict(
        cells=len(subset), need_left=len(left), need_right=len(right),
        need_left_base=len(left_base), need_left_mirror=len(left_mirror),
        need_right_base=len(right_base), need_right_mirror=len(right_mirror),
        flip=sum(1 for row in left if row['screen_direction'] == 'left'),
        flip_base=sum(1 for row in left_base if row['screen_direction'] == 'left'),
        flip_mirror=sum(1 for row in left_mirror if row['screen_direction'] == 'left'),
        flip_rate=round(sum(1 for row in left if row['screen_direction'] == 'left') / len(left), 6)
        if left else None,
        keep=sum(1 for row in right if row['screen_direction'] == 'right'),
        keep_base=sum(1 for row in right_base if row['screen_direction'] == 'right'),
        keep_mirror=sum(1 for row in right_mirror if row['screen_direction'] == 'right'),
        keep_rate=round(sum(1 for row in right if row['screen_direction'] == 'right') / len(right), 6)
        if right else None,
        note='need_left = the gap screen centre is left of the player at the start, i.e. the frozen '
             'screen-right prior is wrong; flip = the run went screen-left there. need_right is the '
             'opposite; keep = the run went screen-right there')


def judgement(gaps):
    """The pre-registered read-out: does scale flip the cells the screen-right prior gets wrong?"""
    new = gaps['new_heldout']
    names_by_arm = {'S': [f'S-s{seed}' for seed in SEEDS], 'L': [f'L-s{seed}' for seed in SEEDS],
                    'B0': ['B0']}
    seeds = {}
    for seed in SEEDS:
        entry = {'B0': conditioned(new, ['B0'])}
        for arm in ARMS:
            network = f'{arm}-s{seed}'
            raw = {}
            for group in ('base', 'mirror'):
                subset = [row for row in new.values() if row['network'] == network and
                          row['mirror_group'] == group]
                verdicts = [row['verdict'] for row in subset]
                raw[group] = dict(cells=len(subset), toward=verdicts.count('toward'),
                                  hold=verdicts.count('hold'), away=verdicts.count('away'),
                                  in_gap=sum(1 for row in subset if row['in_gap_at_wall']),
                                  toward_rate=round(verdicts.count('toward') / len(subset), 6)
                                  if subset else None)
            entry[arm] = dict(conditioned=conditioned(new, [network]), raw=raw,
                              feasible=sum(1 for row in new.values()
                                           if row['network'] == network and row['feasible']))
        seeds[str(seed)] = entry
    l_flip = [seeds[str(seed)]['L']['conditioned']['flip'] for seed in SEEDS]
    l_flip_mirror = [seeds[str(seed)]['L']['conditioned']['flip_mirror'] for seed in SEEDS]
    s_flip = [seeds[str(seed)]['S']['conditioned']['flip'] for seed in SEEDS]
    b0_flip = [seeds[str(seed)]['B0']['flip'] for seed in SEEDS]
    l_keep = [seeds[str(seed)]['L']['conditioned']['keep'] for seed in SEEDS]
    l_keep_rate = [seeds[str(seed)]['L']['conditioned']['keep_rate'] for seed in SEEDS]
    left_cells = seeds[str(SEEDS[0])]['L']['conditioned']['need_left']
    right_cells = seeds[str(SEEDS[0])]['L']['conditioned']['need_right']
    positive = sum(1 for value in l_flip if value > 0)
    if positive >= 2:
        p1 = 'scale_helps'
    elif positive == 0:
        p1 = 'scale_not_the_cause'
    else:
        p1 = 'partial_scale'
    base_kept = all(rate is not None and rate >= .5 for rate in l_keep_rate)
    if p1 == 'scale_helps':
        overall = 'scale_helps' if base_kept else 'mixed_scale'
    elif p1 == 'scale_not_the_cause':
        overall = 'scale_not_the_cause'
    else:
        overall = 'partial_scale'
    r19 = {}
    for network in ['B0'] + [f'{arm}-s{seed}' for arm in ARMS for seed in SEEDS]:
        r19[network] = conditioned(gaps['r19_frozen'], [network])
    return dict(
        rule=dict(
            P1='on the NEW held-out set, the large arm\'s flip count per seed over the need-left cells (the gap '
               'is screen-left of the player at the start, so the screen-right prior is wrong); scale_helps iff '
               'at least two of three seeds are > 0, scale_not_the_cause iff all three are 0, partial_scale '
               'otherwise',
            P1_mirror='the mirror-only subset of the need-left cells, reported separately; it is the exact '
                      'analogue of round19/24\'s mirror group, but the new held-out also contains base cells the '
                      'prior gets wrong',
            P2='base survives iff the large arm\'s keep rate over the need_right cells (where screen-right is '
               'correct) is >= 0.5 in every seed',
            P3='the R19-frozen set is reported for both arms against round24\'s recorded arm A; its three '
               'need-left cells are exactly the three mirror events, so its flip count is directly comparable '
               'to round24\'s mirror toward count of 3/9',
            geometry='the new held-out set mixes gap sides, so a RAW "mirror toward" rate can be satisfied by '
                     'the frozen screen-right prior: the baseline B0 itself scores 5/7 mirror-toward there '
                     '(and 2/7 base-toward).  The primary is therefore direction-conditioned, and the raw '
                     'mirror/base toward counts are reported as a caveat, never as the verdict',
            overall='scale_helps iff P1 == scale_helps and P2 holds; mixed_scale iff P1 == scale_helps and P2 '
                    'fails; scale_not_the_cause iff P1 == scale_not_the_cause; partial_scale otherwise'),
        P1=dict(value=p1, large_flip_by_seed=l_flip, cells_per_seed=left_cells,
                positive_seeds=positive, small_flip_by_seed=s_flip, b0_flip_by_seed=b0_flip),
        P1_mirror=dict(value=p1 if any(v > 0 for v in l_flip_mirror) else 'zero',
                       large_flip_mirror_by_seed=l_flip_mirror,
                       cells_per_seed=seeds[str(SEEDS[0])]['L']['conditioned']['need_left_mirror'],
                       small_flip_mirror_by_seed=[seeds[str(seed)]['S']['conditioned']['flip_mirror']
                                                  for seed in SEEDS]),
        P2=dict(value='base_kept' if base_kept else 'base_lost', large_keep_by_seed=l_keep,
                large_keep_rate_by_seed=l_keep_rate, cells_per_seed=right_cells,
                small_keep_by_seed=[seeds[str(seed)]['S']['conditioned']['keep'] for seed in SEEDS]),
        P3=dict(value='reported', r19_frozen=r19,
                round24_arm_A=dict(mirror_toward=3, mirror_cells=9, base_toward=6, base_cells=9,
                                   feasible=[2, 2, 2],
                                   note='round24 arm A on six events: its three need-left cells are the three '
                                        'mirror events, so mirror toward 3/9 is the same quantity as flip 3/9')),
        pooled=dict(large_flip=sum(l_flip), need_left_cells=left_cells * len(SEEDS),
                    large_flip_mirror=sum(l_flip_mirror),
                    need_left_mirror_cells=seeds[str(SEEDS[0])]['L']['conditioned']['need_left_mirror'] * len(SEEDS),
                    small_flip=sum(s_flip), large_keep=sum(l_keep), need_right_cells=right_cells * len(SEEDS),
                    large_feasible=sum(seeds[str(seed)]['L']['feasible'] for seed in SEEDS),
                    small_feasible=sum(seeds[str(seed)]['S']['feasible'] for seed in SEEDS),
                    positive_seeds=positive),
        per_seed=seeds, raw_mirror_toward=dict(
            large=[seeds[str(seed)]['L']['raw']['mirror']['toward'] for seed in SEEDS],
            small=[seeds[str(seed)]['S']['raw']['mirror']['toward'] for seed in SEEDS],
            b0=seeds[str(SEEDS[0])]['B0']['need_right_mirror'],
            note='reported for continuity with rounds 21-24 but geometry-confounded on this held-out set; the '
                 'baseline already scores 5/7'),
        verdict=overall,
        claim_limit='descriptive counts over three seeds and seven new held-out F1 groups; mirrors are not '
                    'independent samples; no significance test is run')


def summarize_sets(sets, closed, gaps, baselines):
    result = {}
    networks = ['B0'] + [f'{arm}-s{seed}' for arm in ARMS for seed in SEEDS]
    for set_name, events in sets.items():
        rows = gaps[set_name]
        direction = {arm: r22.group_summary(rows, group) for arm, group in
                     (('B0', ['B0']), ('S', [f'S-s{seed}' for seed in SEEDS]),
                      ('L', [f'L-s{seed}' for seed in SEEDS]))}
        mechanism = r22.mechanism_block(rows, {'S': [f'S-s{seed}' for seed in SEEDS],
                                               'L': [f'L-s{seed}' for seed in SEEDS], 'B0': ['B0']})
        result[set_name] = dict(
            events=len(events), event_ids=[entry['event_id'] for entry in events],
            groups=len({entry['group'] for entry in events}),
            filter_valid=baselines[set_name]['filter_valid'],
            passing_events=baselines[set_name]['passing_events'],
            families={name: dict(events=closed[(name, set_name)]['families']['F1']['events'],
                                 feasible=closed[(name, set_name)]['families']['F1']['feasible'],
                                 survived=closed[(name, set_name)]['families']['F1']['survived'])
                      for name in networks},
            per_event_firsthit={name: closed[(name, set_name)]['families']['F1']['firsthit']
                                for name in networks},
            behaviour={name: dict(switches=closed[(name, set_name)]['behaviour']['switches'],
                                  action_shares=closed[(name, set_name)]['behaviour']['action_shares'],
                                  two_segment_feasible_events=closed[(name, set_name)]['behaviour']
                                  ['two_segment_feasible_events'])
                       for name in networks},
            direction=direction, mechanism=mechanism,
            conditioned=dict(B0=conditioned(rows, ['B0']),
                             S=conditioned(rows, [f'S-s{seed}' for seed in SEEDS]),
                             L=conditioned(rows, [f'L-s{seed}' for seed in SEEDS]),
                             by_network={name: conditioned(rows, [name]) for name in networks},
                             note='direction-conditioned flip/keep counts; need_left is the set of cells where '
                                  'the frozen screen-right prior is wrong, so a raw mirror-toward count is not a '
                                  'flip measure on a held-out set that mixes gap sides'),
            screen=dict(B0=screen_share(rows, ['B0']), S=screen_share(rows, [f'S-s{seed}' for seed in SEEDS]),
                        L=screen_share(rows, [f'L-s{seed}' for seed in SEEDS]),
                        note='screen direction = sign of the total screen-x displacement of the executed actions'),
            cells=len(rows))
    return result


def environment():
    return dict(cuda_available=bool(torch.cuda.is_available()),
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                torch=str(torch.__version__), cuda=str(torch.version.cuda), python='3.12.10',
                precision='FP32, autocast disabled',
                tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                matmul_precision=torch.get_float32_matmul_precision(),
                note='recorded inside the run; both TF32 flags must read false')


def assemble(prereg, search, split, assessments, records, reports, baselines, closed, gaps, timings):
    groups = {row['group']: row for row in split['groups']}
    sets = {}
    for name, ids in (('r19_frozen', split['r19_frozen_events']), ('new_heldout', split['new_heldout_events'])):
        rows = []
        for group_id, group in groups.items():
            for event in group['events']:
                if event['event_id'] in ids:
                    rows.append(dict(event_id=event['event_id'], family=FAMILY, group=group_id,
                                     mirror=bool(event['mirror']), scenario=event['scenario'],
                                     scenario_sha256=event['scenario_sha256']))
        sets[name] = rows
    table = summarize_sets(sets, closed, gaps, baselines)
    old_train = [entry for entry in records['events']
                 if entry['containment']['round21_assets'] and
                 all(bool(value) for value in entry['containment']['round21_asset_bytes_equal'].values())]
    summary = dict(
        schema=RESULT_SCHEMA, family=FAMILY, seeds=list(SEEDS), arms=list(ARMS), updates=UPDATES,
        lr=LR, clip=CLIP, batch=1, loss_arm=CONTROL, anchor_weight=0., condition=CONDITION,
        hypothesis='data scale, not label or coupling: does the F1 mirror toward-the-gap rate become non-zero '
                   'when the F1 curriculum grows from six groups to twelve or more?',
        search=dict(drawn=search['drawn'], budget=search['budget'], counts=search['counts'],
                    frozen_census=search['frozen_census'],
                    accepted=[dict(group=row['group'], candidate=row['candidate'], signature=row['signature'])
                              for row in search['accepted']]),
        split=dict(counts=split['counts'], floors=split['floors'], forced_heldout=split['forced_heldout'],
                   new_heldout=split['new_heldout'], flips=split['flips'], shortfall=split['shortfall'],
                   rule=split['rule'], assertions=assessments),
        data=dict(events=records['totals']['events'], verified=records['totals']['verified'],
                  failed=records['totals']['failed'], bytes=records['totals']['bytes'],
                  changed_pixels=records['totals']['changed_pixels'],
                  old_train_events_byte_equal_to_round21=len(old_train),
                  old_train_events=[entry['event_id'] for entry in old_train],
                  round19_teacher_records_checked=sum(
                      1 for entry in records['events']
                      if entry['containment']['round19_teacher_actions_equal'] is True),
                  frozen_group_events=sum(1 for entry in records['events']
                                          if entry['group'] in R19_FROZEN_TRAIN + R19_FROZEN_VALIDATION)),
        initialization=dict(per_seed={str(seed): reports[f'S-s{seed}']['initial_head_state_sha256']
                                      for seed in SEEDS},
                            identical_between_arms=all(
                                reports[f'S-s{seed}']['initial_head_state_sha256'] ==
                                reports[f'L-s{seed}']['initial_head_state_sha256'] for seed in SEEDS)),
        checkpoints={name: dict(training_set=reports[name]['training_set'], seed=reports[name]['seed'],
                                final_head_state_sha256=reports[name]['final_head_state_sha256'],
                                checkpoint_sha256=reports[name]['checkpoint_sha256'],
                                loss_at_step0=reports[name]['loss_at_step0'], loss_final=reports[name]['loss_final'],
                                epochs=reports[name]['epochs'],
                                supervised_events=len(reports[name]['supervised_events']),
                                optimization_seconds=reports[name]['optimization_seconds'])
                       for name in reports},
        sets=table, judgement=judgement(gaps),
        round24_reference=dict(summary_sha256=file_sha(R24_SUMMARY),
                               f1_validation_feasible=R24_REFERENCE['f1_validation_feasible'],
                               direction=R24_REFERENCE['direction'],
                               note='round24 arm A on exactly the six R19-frozen F1 validation events'),
        timings=timings, environment=environment(),
        flags=dict(new_worlds=True, new_pixels=True, new_teacher_labels=True, training=True,
                   test_split=False, formal_gate=False, deployable=False, game=False, tuning=False,
                   auxiliary_head=False, official_labels=False, research_head=True),
        limitations=[
            'the small arm trains on all six old F1 groups, which includes the three frozen R19 validation '
            'groups; its r19-frozen numbers are seen numbers, and every primary claim uses the new held-out set',
            'the two arms are not nested: S = the six old groups, L = three old train groups plus nine new '
            'groups, so size and composition change together',
            'seven new held-out groups (14 cells, 7 base + 7 mirror per network); mirrors are not independent '
            'samples and three seeds are not independent runs of the world',
            'updates are fixed at 200, so the small arm sees 16.7 epochs and the large arm 8.3: compute is held '
            'fixed, not epochs',
            'F1 success is a 12-20 px parking window reached before a fixed lethal frame; it is a geometric '
            'construction, not tactical play',
            'the teacher labels come from a planner that is told the world geometry, not from a learned policy',
            'synthetic fixed-y worlds, two-frame actions, no shooting, no bombs, no real TH08 pixels or hitbox',
            'one hyper-parameter configuration; no tuning; no formal gate; no deployable checkpoint',
        ])
    return summary


# ---------------------------------------------------------------- pre-registration
def criterion():
    return dict(
        round='round25',
        purpose='F1-only test of the data-scale hypothesis: with the same generator, criterion, renderer, teacher '
                'and closed loop, does growing the F1 curriculum from six groups to twelve or more make the mirror '
                'group move toward the gap, and does the base group survive?',
        search=dict(family=FAMILY, generator='tools.rl.v373_world_filter.draw_geometry, unchanged, F1 domain',
                    domains=copy.deepcopy(r18.DOMAINS[FAMILY]),
                    criterion='tools.rl.v373_world_filter.evaluate_world, unchanged: constant hold/right/left, the '
                              '16-decision left-right alternation, the seeded uniform random policy and the '
                              'teacher-first weak baseline must all fail, and the independent solve() must return a '
                              'complete 80-action witness that replays without early termination',
                    seed=SEARCH['seed'], budget=SEARCH['budget'], target_groups=SEARCH['target_groups'],
                    mirrors=2, split_label='development', horizon=HORIZON,
                    shortfall='if the budget runs out before the target, the actual number and every per-candidate '
                              'reason are reported; no threshold is loosened, no existing world is changed and no '
                              'budget is extended after the run',
                    extras='round18\'s six-constant diagnostic (candidates 0/3/7/9/12/16) is reported but is not '
                           'part of the criterion, so a constant-passable world is retained and flagged'),
        disjointness=dict(
            signature='round9 confirmation signature (family, canonical start_x, coverage, gap, wall_frame, '
                      'fire_interval, window, threshold, goal, origin, burst, bullet_speed, internal_gap)',
            frozen=['r8_development_validation', 'r9_confirmation', 'r10_records', 'r16_confirmation',
                    'r18_worlds', 'r19_f1_split (the same fourteen R18 groups)'],
            assertion='every accepted group signature must be absent from the union and distinct from every other '
                      'accepted group; a duplicate is discarded before evaluation and is never trained on'),
        split=dict(
            unit='one group = base + mirror, atomic; a group never crosses the split',
            order='the six R18/R19 F1 groups in R19 split order, then the new groups in acceptance order',
            assignment='index even -> train, odd -> held-out',
            forced='the three frozen R19 F1 validation groups (wf18-F1-005/007/010) are always held out',
            floors=dict(train_groups=SPLIT_FLOORS['train_groups'], heldout_groups=SPLIT_FLOORS['heldout_groups']),
            adjustment='while train < 12 and held-out > 5 the last held-out group is flipped to train; if '
                       'held-out < 5 the last train group is flipped back; every flip is recorded',
            shortfall='if fewer than eight new groups are found, the train arm uses every group except the frozen '
                      'R19 validation groups and the shortfall is recorded instead of relaxing anything',
            leakage='assert the train split and both evaluation sets are disjoint; assert the large arm trains on '
                    'no evaluation event; assert the small arm is exactly the six frozen groups'),
        arms=dict(
            S=dict(text=ARM_TEXT['S'], groups=6, events=12,
                   note='the whole old F1 data, including the three frozen R19 validation groups; its r19-frozen '
                        'reading is seen and is never a held-out claim'),
            L=dict(text=ARM_TEXT['L'], groups='the new train split (>= 12 groups)', events='2 x groups',
                   note='never trains on the R19-frozen groups or the new held-out groups')),
        training=dict(loss='the frozen round20 control loss (no KL anchor, no auxiliary head)',
                      seeds=list(SEEDS), updates=UPDATES, optimizer='Adam', lr=LR, clip=CLIP, batch=1,
                      precision='FP32, autocast disabled, matmul TF32 False, cuDNN TF32 False, float32 matmul '
                                'precision highest', real_cuda=True,
                      protocol='tools.rl.v373_valid_curriculum.infer/arm_loss/offline and '
                               'tools.rl.v373_seed_confirm.fresh_policy/state_sha reused verbatim; the loop body is '
                               'round20 train_all\'s with the arm factor replaced by the training set',
                      initialization='fresh seed reset per construction; the same seed must give the identical '
                                     'initial head in both arms and must equal the frozen R16/R20/R21 value',
                      epochs_note='updates are fixed at 200; the small arm therefore sees 200/12 = 16.7 epochs and '
                                  'the large arm 200/24 = 8.3, i.e. compute is held fixed, not epochs',
                      selection='final step only; no early stopping, no held-out selection, no tuning'),
        evaluation=dict(
            sets=dict(r19_frozen='the six events of the three frozen R19 F1 validation groups (cross-set contrast)',
                      new_heldout='the new held-out groups (>= 5 groups, mirrors atomic)'),
            condition=CONDITION,
            runner='tools.rl.v373_telegraph.run_event_rendered via its closed_loop: the frozen R11 run_event with '
                   'only the renderer parameter added',
            renderer='tools.rl.v373_telegraph.render_state_telegraph, unchanged (lead 45, fill/edge colours, edge '
                     'width 4)',
            b0='the frozen baseline policy logits on the identical renders; a reference, not a capability',
            six_baselines='tools.rl.v373_valid_curriculum.six_baselines (the frozen R18 policies/causal_loop/'
                          'outcome); if any baseline is feasible on a held-out event the filter has failed and the '
                          'learning conclusion is withheld',
            metrics='per cell (network x F1 event): toward/hold/away against the gap, in_gap_at_wall, signed '
                    'parking error px, feasible, survived, first hit; per event first hit; the screen-right share '
                    'of every arm and group; and the direction-conditioned reading: need_left cells (the gap screen '
                    'centre is left of the player at the start, i.e. the frozen screen-right prior is wrong), flip '
                    '(the run goes screen-left there), need_right cells and keep',
            direction='round22 cell_diagnostic/group_summary/mechanism_block reused verbatim: toward means the '
                      'canonical net motion points at the gap centre from the start position; the '
                      'direction-conditioned flip/keep counts are computed here because the new held-out set mixes '
                      'gap sides, which the raw mirror/base grouping does not control',
            claim='descriptive only; n = 3 seeds; seven new held-out groups; mirrors are not independent samples'),
        judgement=dict(
            P1='on the new held-out set, the large arm\'s flip count per seed over the need-left cells (seven '
               'cells: five base + two mirror); scale_helps iff at least two of three seeds are > 0, '
               'scale_not_the_cause iff all three are 0, partial_scale otherwise',
            P1_mirror='the mirror-only subset of the need-left cells (two cells), reported separately; it is the '
                      'exact analogue of round19/24\'s mirror group',
            P2='base survives iff the large arm\'s keep rate over the need_right cells (seven cells) is >= 0.5 in '
               'every seed',
            P3='the R19-frozen set is reported for both arms against round24 arm A; its three need-left cells are '
               'exactly the three mirror events, so its flip count is directly comparable to round24\'s mirror '
               'toward 3/9; it is seen for the small arm and held out for the large arm, so it carries no primary '
               'claim',
            geometry='the new held-out set mixes gap sides, so a raw "mirror toward" rate can be satisfied by the '
                     'frozen screen-right prior (B0 scores 5/7 mirror-toward and 2/7 base-toward there); the '
                     'primary is therefore direction-conditioned and the raw counts are reported as a caveat',
            overall='scale_helps iff P1 == scale_helps and P2 holds; mixed_scale iff P1 == scale_helps and P2 '
                    'fails; scale_not_the_cause iff P1 == scale_not_the_cause; partial_scale otherwise',
            no_relaxation='the verdict is written from these rules whether or not it flatters the hypothesis'),
        claims=dict(
            can_show='whether growing the F1 curriculum, with everything else byte-identical, moves the mirror '
                     'group off the screen-right prior and keeps the base group',
            cannot_show='tactical understanding, generalisation beyond these held-out geometries, or that data '
                        'scale is sufficient in general; a null result only says scale alone did not do it here'),
        failure_policy='first failure retained; one run; no retrain, rerun, repair, overwrite, extra epoch, tuning '
                       'or baseline switch')


def criterion_sha(value=None):
    return hashlib.sha256(json.dumps(criterion() if value is None else value, sort_keys=True).encode()).hexdigest()


def identity_paths():
    return [Path('tools/rl/v373_f1_scale.py'), Path('tests/test_v373_f1_scale.py'),
            Path('tools/rl/v373_world_filter.py'), Path('tools/rl/v373_telegraph.py'),
            Path('tools/rl/v373_valid_curriculum.py'), Path('tools/rl/v373_spatial_fuse.py'),
            Path('tools/rl/v373_seed_confirm.py'),
            Path('touhou_rl/tactical_teacher_v2.py'), Path('touhou_rl/tactical_teacher_validation.py'),
            Path('touhou_rl/tactical_teacher.py'), Path('touhou_rl/tactical_synthetic_pixels.py'),
            Path('touhou_rl/tactical_synthetic_closed_loop.py'), Path('touhou_rl/tactical_scenarios.py'),
            Path('touhou_rl/tactical_policy.py'), Path('touhou_rl/tactical_data.py'), Path('touhou_rl/vision.py'),
            Path('touhou_rl/vmodels.py'), Path('touhou_rl/tactical_feature_cache.py'),
            Path('touhou_rl/tactical_sequence_train.py'), BASELINE, MANIFEST, SPLIT, R21_RECORDS, R24_SUMMARY] + \
           [R19_WORLDS / (group + f'-m{mirror}.json') for group in R19_FROZEN_TRAIN + R19_FROZEN_VALIDATION
            for mirror in (0, 1)]


def identities():
    missing = [str(path) for path in identity_paths() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError('frozen identity missing: ' + ','.join(missing))
    return {str(path): file_sha(path) for path in identity_paths()}


def preregister(out=OUT):
    out = new_dir(out)
    manifest = load_manifest()
    split = load_split(manifest)
    r19_frozen_groups(split)
    union, census = frozen_signature_union()
    reference = r24_reference()
    initial = r16.initial_head_shas()
    if len(set(initial.values())) != len(SEEDS):
        raise ValueError('different seeds must give different initial heads')
    value = dict(
        schema=PREREG_SCHEMA, identities=identities(), criterion=criterion(), criterion_sha256=criterion_sha(),
        family=FAMILY, seeds=list(SEEDS), updates=UPDATES, lr=LR, clip=CLIP, condition=CONDITION,
        arms=list(ARMS), arm_text=ARM_TEXT, search=copy.deepcopy(SEARCH), floors=dict(SPLIT_FLOORS),
        shortfall_threshold=NEW_GROUPS_SHORTFALL,
        sources=dict(manifest=dict(path=str(MANIFEST), sha256=manifest['sha256'], schema=r20.MANIFEST_SCHEMA),
                     split=dict(path=str(SPLIT), sha256=split['sha256'], schema=r20.SPLIT_SCHEMA),
                     baseline=dict(path=str(BASELINE), sha256=file_sha(BASELINE)),
                     round21_records=dict(path=str(R21_RECORDS), sha256=file_sha(R21_RECORDS)),
                     round24=reference),
        frozen=dict(census=census, union=len(union),
                    r19_frozen_train=list(R19_FROZEN_TRAIN),
                    r19_frozen_validation=list(R19_FROZEN_VALIDATION),
                    r19_frozen_signatures={group: row['signature']
                                            for group, row in r19_frozen_groups(split).items()}),
        initialization=dict(per_seed=initial, identical_between_arms=True,
                            method='r16.fresh_policy(backbone, seed) resets torch.manual_seed(seed) immediately '
                                   'before every construction, so the same seed gives the identical head in both '
                                   'arms and must reproduce the frozen R16/R20/R21 value'),
        expected=dict(new_groups=SEARCH['target_groups'], new_events=2 * SEARCH['target_groups'],
                      total_groups=6 + SEARCH['target_groups'], train_groups=12, heldout_groups=10,
                      new_heldout_groups=7, r19_frozen_events=6, new_heldout_events=14,
                      checkpoints=len(ARMS) * len(SEEDS), closed_loop_files=(1 + len(ARMS) * len(SEEDS)) * 2),
        independent_test=False, formal_gate=False, deployable=False, game=False, tuning=False,
        official_labels=False, research_teacher_labels=True, new_worlds=True, new_pixels=True,
        failure_policy='first failure retained; one run; no retrain, rerun, repair, overwrite, extra epoch or '
                       'baseline switch')
    write_json(out / 'preregistration.json', value)
    return value


def verify_prereg(out=OUT):
    out = Path(out)
    p = json.loads((out / 'preregistration.json').read_text(encoding='utf-8'))
    if p.get('schema') != PREREG_SCHEMA:
        raise ValueError('pre-registration schema')
    for path, digest in p['identities'].items():
        if file_sha(path) != digest:
            raise ValueError('frozen identity changed: ' + path)
    if p['criterion_sha256'] != criterion_sha():
        raise ValueError('the criterion drifted from the frozen pre-registration')
    if p['search'] != SEARCH:
        raise ValueError('the search procedure changed after pre-registration')
    if p['criterion']['search']['domains'] != r18.DOMAINS[FAMILY]:
        raise ValueError('the round18 F1 generator domain changed')
    manifest = load_manifest()
    split = load_split(manifest)
    if p['sources']['manifest']['sha256'] != manifest['sha256']:
        raise ValueError('R18 manifest SHA drift')
    if p['sources']['split']['sha256'] != split['sha256']:
        raise ValueError('R19 split SHA drift')
    union, census = frozen_signature_union()
    if census['frozen_union'] != p['frozen']['census']['frozen_union']:
        raise ValueError('the frozen signature union changed')
    r19 = r19_frozen_groups(split)
    if {group: row['signature'] for group, row in r19.items()} != p['frozen']['r19_frozen_signatures']:
        raise ValueError('the frozen R19 F1 signatures changed')
    if r16.initial_head_shas() != p['initialization']['per_seed']:
        raise ValueError('the initial head state changed')
    reference = r24_reference()
    if reference['summary_sha256'] != p['sources']['round24']['summary_sha256']:
        raise ValueError('the round24 summary changed')
    if reference['f1_validation_feasible'] != p['sources']['round24']['f1_validation_feasible']:
        raise ValueError('the round24 reference numbers changed')
    return p


# ---------------------------------------------------------------- run
def write_events(dest, sets, closed, gaps, records, arm_events, baselines):
    r20.new_dir(dest / 'events')
    assets = {entry['event_id']: entry for entry in records['events']}
    trained_by = {}
    for event_id in {row['event_id'] for rows in arm_events.values() for row in rows}:
        trained_by[event_id] = [arm for arm in ARMS if event_id in {row['event_id'] for row in arm_events[arm]}]
    networks = ['B0'] + [f'{arm}-s{seed}' for arm in ARMS for seed in SEEDS]
    baseline_rows = {(set_name, row['event_id'], row['policy']): row
                     for set_name, payload in baselines.items() for row in payload['rows_detail']}
    written = 0
    for set_name, events in sets.items():
        for entry in events:
            event_id = entry['event_id']
            rows = {name: next(row for row in closed[(name, set_name)]['events'] if row['event_id'] == event_id)
                    for name in networks}
            behaviour = {name: next(item for item in closed[(name, set_name)]['behaviour']['per_event']
                                    if item['event_id'] == event_id) for name in networks}
            value = dict(schema=EVENT_SCHEMA, event_id=event_id, set=set_name, family=entry['family'],
                         group=entry['group'], mirror=bool(entry['mirror']), scenario=entry['scenario'],
                         scenario_sha256=entry['scenario_sha256'], trained_by=trained_by.get(event_id, []),
                         asset=assets.get(event_id), r18_six_baselines={
                             policy: dict(feasible=baseline_rows[(set_name, event_id, policy)]['feasible'],
                                          survived=baseline_rows[(set_name, event_id, policy)]['survived'],
                                          first_hit=baseline_rows[(set_name, event_id, policy)]['first_hit'])
                             for policy in baselines[set_name]['policies']},
                         closed_loop={name: dict(feasible=rows[name]['feasible'],
                                                 survived=rows[name]['survived'],
                                                 first_hit=rows[name]['first_collision_frame'],
                                                 decisions=rows[name]['decisions'],
                                                 switches=behaviour[name]['switches'],
                                                 action_counts=behaviour[name]['action_counts'],
                                                 single_candidate=behaviour[name]['single_candidate'])
                                      for name in networks},
                         f1_diagnostic={name: gaps[set_name][f'{name}/{event_id}'] for name in networks})
            r20.write_json(dest / 'events' / f'{event_id}.json', value)
            written += 1
    return written


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
            timings, begin = {}, time.perf_counter()
            search = search_new_groups()
            search['prereg_sha256'] = file_sha(out / 'preregistration.json')
            r20.write_json(dest / 'search.json', search)
            assert_new_disjoint(search['accepted'])
            manifest = load_manifest()
            split = load_split(manifest)
            groups = old_groups(manifest, split) + loaded_search(search)
            table = {row['group']: row for row in groups}
            partitioned = partition(groups)
            payload = split_payload(partitioned, groups)
            r20.write_json(dest / 'split.json', payload)
            if not payload['floors']['disjoint'] or not payload['floors']['forced_heldout']:
                raise AssertionError('the split violates the frozen rules')
            if payload['shortfall'] is None and not payload['floors']['train_groups']:
                raise AssertionError('the train floor must hold when the shortfall clause is not active')
            arm_groups = {'S': list(R19_FROZEN_TRAIN + R19_FROZEN_VALIDATION), 'L': list(partitioned['train'])}
            arm_events = {arm: event_table([table[group] for group in ids]) for arm, ids in arm_groups.items()}
            assessments = split_assertions(payload, {arm: {row['event_id'] for row in rows}
                                                     for arm, rows in arm_events.items()})
            if not assessments['ok']:
                raise AssertionError('split assertions failed: ' + '; '.join(assessments['errors']))
            eval_groups = {'r19_frozen': [table[group] for group in R19_FROZEN_VALIDATION],
                           'new_heldout': [table[group] for group in partitioned['new_heldout']]}
            if not eval_groups['new_heldout']:
                raise AssertionError('the new held-out set is empty')
            eval_events = {name: [dict(row, split=name) for row in event_table(rows)]
                           for name, rows in eval_groups.items()}
            all_events, seen = [], set()
            for rows in arm_events.values():
                for row in rows:
                    if row['event_id'] in seen:
                        continue
                    seen.add(row['event_id'])
                    all_events.append(dict(row, split='development'))
            assets21 = r21_assets()
            timings['search_seconds'] = round(time.perf_counter() - begin, 3)
            begin = time.perf_counter()
            built = build_assets(dest, all_events, assets21)
            records = write_records(dest, built)
            timings['data_seconds'] = round(time.perf_counter() - begin, 3)
            if records['failures']:
                raise AssertionError('the teacher must verify every rendered event: ' +
                                     json.dumps(records['failures']))
            backbone, _ = load_vision_checkpoint(BASELINE, map_location='cpu')
            backbone.to(device).eval()
            frozen_backbone = {key: value.detach().clone() for key, value in backbone.state_dict().items()}
            begin = time.perf_counter()
            data = r20.encode(backbone, records, dest, device)
            torch.cuda.synchronize()
            timings['encoding_seconds'] = round(time.perf_counter() - begin, 3)
            by_id = {entry['entry']['event_id']: entry for entry in data}
            datasets = {arm: [by_id[row['event_id']] for row in rows] for arm, rows in arm_events.items()}
            begin = time.perf_counter()
            reports, paths = train_pair(dest, datasets, backbone, device, prereg)
            timings['training_seconds_total'] = round(time.perf_counter() - begin, 3)
            if not all(torch.equal(value, frozen_backbone[key])
                       for key, value in backbone.state_dict().items()):
                raise AssertionError('the frozen backbone changed')
            baselines, closed, gaps, seconds = evaluate(dest, eval_events, backbone, device, reports, paths)
            timings['evaluation_seconds'] = round(seconds, 3)
            r20.write_json(dest / 'timings.json', timings)
            event_files = write_events(dest, eval_events, closed, gaps, records, arm_events, baselines)
            verify_prereg(out)
            summary = assemble(prereg, search, payload, assessments, records, reports, baselines, closed, gaps,
                               timings)
            summary['events_written'] = event_files
            r20.write_json(dest / 'summary.json', summary)
            r20.write_json(dest / 'sha256.json', {str(path.relative_to(dest)): file_sha(path)
                                                  for path in sorted(dest.rglob('*')) if path.is_file()})
            r20.write_json(out / 'sha256.json', {str(path.relative_to(out)): file_sha(path)
                                                 for path in sorted(out.rglob('*')) if path.is_file()})
            return summary
    except BaseException:
        r20.write_json(dest / 'first-failure.json',
                       dict(traceback=traceback.format_exc(),
                            policy='retained; no automatic rerun, overwrite, retune or baseline switch'))
        raise


# ---------------------------------------------------------------- read-only verify
def verify_artifacts(out=OUT):
    out = Path(out)
    prereg = verify_prereg(out)
    run = out / 'run'
    stored = json.loads((run / 'sha256.json').read_text(encoding='utf-8'))
    drift = [path for path, digest in stored.items() if file_sha(run / path) != digest]
    if drift:
        raise ValueError('run artifact drift: ' + ', '.join(drift))
    root = json.loads((out / 'sha256.json').read_text(encoding='utf-8'))
    root_drift = [path for path, digest in root.items() if file_sha(out / path) != digest]
    if root_drift:
        raise ValueError('root artifact drift: ' + ', '.join(root_drift))
    search = json.loads((run / 'search.json').read_text(encoding='utf-8'))
    assert_new_disjoint(search['accepted'])
    manifest = load_manifest()
    split = load_split(manifest)
    groups = old_groups(manifest, split) + loaded_search(search)
    table = {row['group']: row for row in groups}
    payload = json.loads((run / 'split.json').read_text(encoding='utf-8'))
    rebuilt = split_payload(partition(groups), groups)
    if json.dumps(rebuilt, sort_keys=True) != json.dumps(payload, sort_keys=True):
        raise ValueError('the stored split does not rebuild from the stored search')
    arm_events = {'S': {event for group in R19_FROZEN_TRAIN + R19_FROZEN_VALIDATION
                        for event in table[group]['events']},
                  'L': set(payload['train_events'])}
    assessments = split_assertions(payload, arm_events)
    if not assessments['ok']:
        raise ValueError('split assertions failed: ' + '; '.join(assessments['errors']))
    records = json.loads((run / 'records.json').read_text(encoding='utf-8'))
    checked = 0
    for entry in records['events']:
        for key in ('pixels', 'history', 'labels', 'rollout'):
            if file_sha(run / entry[key]['path']) != entry[key]['sha256']:
                raise ValueError('asset SHA drift: ' + entry['event_id'] + '/' + key)
            checked += 1
        if entry['containment']['round21_assets'] and not all(entry['containment']
                                                              ['round21_asset_bytes_equal'].values()):
            raise ValueError('a reproduced round21 asset differs: ' + entry['event_id'])
    reports = json.loads((run / 'checkpoints.json').read_text(encoding='utf-8'))
    if len(reports) != len(ARMS) * len(SEEDS):
        raise ValueError('checkpoint count')
    for name, report in reports.items():
        path = run / report['checkpoint_path']
        if file_sha(path) != report['checkpoint_sha256']:
            raise ValueError('checkpoint SHA drift: ' + name)
    for seed in SEEDS:
        if reports[f'S-s{seed}']['final_head_state_sha256'] == reports[f'L-s{seed}']['final_head_state_sha256']:
            raise ValueError('the two arms ended identical for seed ' + str(seed))
        if reports[f'S-s{seed}']['initial_head_state_sha256'] != \
                reports[f'L-s{seed}']['initial_head_state_sha256']:
            raise ValueError('the two arms did not start identical for seed ' + str(seed))
    baselines = json.loads((run / 'baselines.json').read_text(encoding='utf-8'))
    for name, value in baselines.items():
        if not value['filter_valid'] or value['passing_events']:
            raise ValueError('R18 filter failure stored on ' + name)
    closed = 0
    for name in ['B0'] + [f'{arm}-s{seed}' for arm in ARMS for seed in SEEDS]:
        for set_name in ('r19_frozen', 'new_heldout'):
            value = json.loads((run / 'closed-loop' / f'{name}-{set_name}.json').read_text(encoding='utf-8'))
            if len(value['events']) != len(payload[f'{set_name}_events']):
                raise ValueError('closed-loop event count: ' + name + '/' + set_name)
            closed += 1
    gaps = json.loads((run / 'f1-gap.json').read_text(encoding='utf-8'))['sets']
    expected_rows = (1 + len(ARMS) * len(SEEDS)) * (len(payload['r19_frozen_events']) +
                                                    len(payload['new_heldout_events']))
    if sum(len(rows) for rows in gaps.values()) != expected_rows:
        raise ValueError('f1-gap row count')
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    if json.dumps(judgement(gaps), sort_keys=True) != json.dumps(summary['judgement'], sort_keys=True):
        raise ValueError('the stored judgement does not recompute from the stored cells')
    reference = r24_reference()
    if reference['f1_validation_feasible'] != summary['round24_reference']['f1_validation_feasible']:
        raise ValueError('round24 reference drift')
    return dict(artifacts=len(stored), root_artifacts=len(root), assets=checked,
                search_accepted=len(search['accepted']), search_drawn=search['drawn'],
                checkpoints=len(reports), closed_loop_files=closed, f1_cells=expected_rows,
                train_groups=payload['counts']['train_groups'], heldout_groups=payload['counts']['heldout_groups'],
                new_heldout_groups=payload['counts']['new_heldout_groups'],
                new_heldout_events=len(payload['new_heldout_events']),
                r19_frozen_events=len(payload['r19_frozen_events']),
                filter_valid={name: value['filter_valid'] for name, value in baselines.items()},
                judgement=summary['judgement']['verdict'],
                small_arm_trained_on_r19_frozen=assessments['small_arm_trained_on_r19_frozen'],
                criterion_sha256=prereg['criterion_sha256'], identities=len(prereg['identities']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preregister', 'run', 'verify'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'preregister':
        value = preregister(args.out)
        print(json.dumps({'schema': value['schema'], 'criterion_sha256': value['criterion_sha256'],
                          'identities': len(value['identities']), 'search': value['search']}, indent=2))
    elif args.command == 'run':
        print(json.dumps(run(args.out, args.device), indent=2)[:2000])
    else:
        print(json.dumps(verify_artifacts(args.out), indent=2))


if __name__ == '__main__':
    main()
