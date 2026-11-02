"""Round18 world-validity filter for the fixed-y synthetic development worlds.

Why this exists: R17 showed that constant policies (always hold / always right /
always left) already pass whole families of the R11/R16 worlds, so those worlds
measure wall-hugging or standing still, not play. R18 adds no training and no
evaluation on such worlds; it rebuilds the world *set*:

  1. Per-world validity criterion (frozen in the pre-registration JSON).
     For one candidate world, the same horizontal_masked causal closed loop is
     driven by five non-learning baselines -- constant hold(0), constant right(3),
     constant left(7), a constant left/right alternation every 16 logical frames,
     and a seeded uniform random policy over the six supported candidates -- plus
     one weak baseline that repeats only the teacher's first two-frame action.
     A world is VALID only if all six baselines fail AND the independent
     reference solver (tactical_teacher_validation.solve -> programs/verify)
     produces a complete action sequence that survives a full non-early replay.
     No reference witness => UNKNOWN, discarded and never reported as unsolvable.
     Any baseline that passes => INVALID ("too easy").
  2. Generation. Candidate geometries are drawn from the ValidationWorld
     parameter space per family, filtered one by one, and kept only when both the
     base and its mirror pass. Target 6 valid groups x 2 mirrors per family.
  3. Disjointness. Every complete parameter signature (the R9 confirmation
     signature) is asserted distinct from R8 development/validation, R9
     confirmation, R10 records and R16 confirmation.

Research only: no training, no teacher labels, no policy checkpoint, no test
split, no game execution. Nothing here is deployable.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_scenarios import initial_state
from touhou_rl.tactical_synthetic_closed_loop import HORIZONTAL
from touhou_rl.tactical_teacher import TacticalTeacher
from touhou_rl.tactical_teacher_validation import (ValidationWorld, advance, make_world, solve,
                                                   verify as verify_actions)
from tools.rl import v373_seed_confirm as r16

ROOT = Path('Document/Update/Alpha/v3.7.3/evidence')
OUT = ROOT / 'worldfilter-round18'
R11 = ROOT / 'synthetic-closed-loop-round11/preregistration.json'
R10_RECORDS = ROOT / 'synthetic-learning-round10/data/records.json'
R9_MANIFEST = ROOT / 'teacher-revision-round9/confirmation-manifest.json'
R16_PREREG = ROOT / 'seed-confirm-round16/preregistration.json'
R17_PROBE = ROOT / 'constant-probe-round17/summary.json'
PREREG_SCHEMA = 'v373-world-filter-prereg/1'
RESULT_SCHEMA = 'v373-world-filter-run/1'

# Frozen horizontal_masked action domain shared with R11/R14/R16.
SUPPORTED = tuple(sorted(HORIZONTAL))
CONSTANTS = {'hold': 0, 'right': 3, 'left': 7}
# R8 preregistration text: "fixed R1 broad oscillation: canonical -2/+2 switching
# every 16 decisions (32 logical frames)". One decision is one two-frame advance.
ALTERNATE_BLOCK = 16         # decisions == 32 logical frames
RANDOM_SEED = 37318
BASELINES = ('hold', 'right', 'left', 'alternate', 'random')
WEAK = 'teacher_first_action'
TARGET_GROUPS = 6
# Declared candidate budget per family. F2 is the scarce family: a world where all
# five non-learning baselines fail and a timed witness still exists is rare, so its
# budget was declared larger before the single run.
BUDGET = {'F1': 200, 'F2': 400, 'F3': 200}
SEEDS = {'F1': 18001, 'F2': 18002, 'F3': 18003}
SPLIT = 'development'
HORIZON = 160
SIGNATURE_FIELDS = ('family', 'canonical start_x', 'coverage', 'gap', 'wall_frame', 'fire_interval',
                    'window', 'threshold', 'goal', 'origin', 'burst', 'bullet_speed', 'internal_gap')
ACTION_CANDIDATE = {0: 0, 2: 3, -2: 7, 1: 12, -1: 16}

# Declared candidate domains. Unused family fields stay at ValidationWorld defaults.
DOMAINS = {
    'F1': dict(coverage=[[8., 376.], [8., 372.], [12., 376.], [16., 368.]],
               internal_gap=[True, True, True, True, False],
               gap_width=[12, 14, 16, 20], gap_origin=list(range(72, 305, 8)),
               wall_frame=list(range(40, 78, 4)), start_offset=[24, 32, 40, 48, 64, 80, 96]),
    'F2': dict(origin=[[192., 100.], [176., 88.], [208., 112.], [160., 96.], [224., 108.], [144., 92.]],
               fire_interval=[12, 14, 16, 18, 20, 22, 24],
               bullet_speed=[7.0, 7.5, 8.0, 8.5, 9.0, 10.0],
               burst=[False, True],
               start_x=[160., 164., 168., 172., 176., 180., 184., 188., 192., 196., 200., 204., 208.]),
    # F3 proposals are timing-informed: r = constant-left crossing / window start,
    # so a constant drift crosses the threshold *before* the window opens. The
    # filter still decides; the proposal only avoids obviously dead geometry.
    'F3': dict(window_start=[40, 48, 56, 64, 72, 80, 88],
               window_length=[16, 20, 24, 28, 32], threshold=[168., 174., 180., 186., 192., 198.],
               goal_offset=[16., 20., 24., 28.], crossing_ratio=[.55, .62, .68, .75, .82],
               fire_interval=[14, 16, 18, 20, 22, 24, 26],
               origin=[[192., 100.], [176., 96.], [208., 104.]]),
}


# ---------------------------------------------------------------- plumbing
def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def new_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _tuple(value):
    return tuple(value) if isinstance(value, (list, tuple)) else value


def signature(spec):
    """The R9 confirmation signature, character for character (mirror-collapsed)."""
    s = ValidationWorld(**spec)
    return (s.family, s.canonical_x(s.start_x), _tuple(s.coverage), _tuple(s.gap), s.wall_frame,
            s.fire_interval, _tuple(s.window), s.threshold, s.goal, _tuple(s.origin), s.burst,
            s.bullet_speed, s.internal_gap)


def existing_signatures():
    """R8 development/validation, R9 confirmation, R10 records, R16 confirmation."""
    old = r16.existing_signatures()                      # r8 / r9 / r10; no test constructed
    old['r16_confirmation'] = r16.confirmation_specs()
    if any(s['split'] == 'test' for rows in old.values() for s in rows):
        raise AssertionError('sealed test split must never be constructed')
    for name, rows in old.items():
        for spec in rows:
            if signature(spec) != r16.signature(spec):
                raise AssertionError('signature definition drift against R9/R16: ' + name)
    return old


def frozen_signature_counts():
    """Read-only census of the frozen sets, for the pre-registration record."""
    old = existing_signatures()
    union = set().union(*({signature(spec) for spec in rows} for rows in old.values()))
    return {name: len({signature(spec) for spec in rows}) for name, rows in old.items()} | \
           dict(union=len(union), r10_subset_of_r8=({signature(s) for s in old['r10']} <=
                                                    {signature(s) for s in old['r8']}))


# ---------------------------------------------------------------- closed loop
def one_hot(candidate):
    logits = torch.zeros(18)
    logits[candidate] = 1.
    return logits


def masked_candidate(logits):
    """Identical masking/argmax step as R11 run_event(condition='horizontal_masked')."""
    if logits.shape != (18,) or not torch.isfinite(logits).all():
        raise ValueError('invalid baseline logits')
    selected = logits.clone()
    selected[[c for c in range(18) if c not in HORIZONTAL]] = -torch.inf
    return int(selected.argmax())


def causal_loop(s, policy):
    """R11 closed loop without pixels: mask -> candidate -> 2-frame advance -> sticky hit."""
    state = initial_state(s)
    actions, decisions = [], []
    positions = [dict(frame=0, x=state.x, hit=False)]
    first_collision = None
    while state.frame < s.horizon:
        candidate = masked_candidate(policy(len(decisions), state))
        if candidate not in HORIZONTAL:
            raise ValueError('masked candidate must be supported')
        action = HORIZONTAL[candidate]
        before = state
        state = advance(s, state, action)
        actions.append(action)
        if state.hit and not before.hit:
            first_collision = state.frame
        decisions.append(dict(frame=before.frame, candidate=candidate, action=action,
                              actual_hold=state.frame - before.frame,
                              x_before=before.x, x_after=state.x, hit=state.hit))
        positions.append(dict(frame=state.frame, x=state.x, hit=state.hit))
    return dict(state=state, actions=actions, decisions=decisions, positions=positions,
                first_collision_frame=first_collision, complete=state.frame == s.horizon)


def outcome(s, run):
    """R11 feasible semantics: full horizon, no sticky collision, F3 goal reached."""
    reached = any(s.canonical_x(p['x']) <= s.goal for p in run['positions'])
    goal = s.family != 'F3' or reached
    survived = bool(run['complete'] and not run['state'].hit)
    return dict(feasible=bool(survived and goal), survived=survived, goal_satisfied=bool(goal),
                first_hit=run['first_collision_frame'], decisions=len(run['decisions']),
                final_x=run['state'].x)


def constant_policy(candidate):
    return lambda step, state: one_hot(candidate)


def alternate_policy(block=ALTERNATE_BLOCK):
    """Constant left/right alternation, one 16-decision (32 logical frame) block each, left first."""
    return lambda step, state: one_hot(7 if (step // block) % 2 == 0 else 3)


def random_candidates(seed, count):
    rng = random.Random(seed)
    return [rng.choice(SUPPORTED) for _ in range(count)]


def random_policy(seed, count=HORIZON // 2):
    sequence = random_candidates(seed, count)
    return lambda step, state: one_hot(sequence[step])


def teacher_first(s):
    """Weak baseline: repeat only the teacher's very first two-frame action."""
    decision = TacticalTeacher().decide(s, initial_state(s))
    if decision.action not in ACTION_CANDIDATE:
        raise ValueError('teacher action outside frozen set')
    candidate = ACTION_CANDIDATE[decision.action]
    row = dict(candidate=candidate, action=decision.action, option=decision.option,
               trigger_reason=decision.trigger_reason)
    return row, constant_policy(candidate)


def policies(s):
    result = {'hold': constant_policy(0), 'right': constant_policy(3), 'left': constant_policy(7),
              'alternate': alternate_policy(), 'random': random_policy(RANDOM_SEED)}
    weak, policy = teacher_first(s)
    result[WEAK] = policy
    return result, weak


# ---------------------------------------------------------------- per world
def evaluate_world(s, force_reference=False):
    """Apply the pre-registered criterion to one world."""
    if s.horizon != HORIZON:
        raise ValueError('the frozen criterion is defined on horizon 160')
    table, weak = policies(s)
    rows = {}
    for name in BASELINES + (WEAK,):
        run = causal_loop(s, table[name])
        row = outcome(s, run)
        row['actions'] = run['actions']
        rows[name] = row
    rows[WEAK]['teacher_first'] = weak
    # The teacher's first action is one of hold/left/right after projection, so the
    # weak baseline is recorded together with the named baseline it coincides with.
    rows[WEAK]['equivalent_baseline'] = next(
        (name for name in BASELINES if rows[name]['actions'] == rows[WEAK]['actions']), None)
    passing = [name for name in BASELINES if rows[name]['feasible']] + \
              ([WEAK] if rows[WEAK]['feasible'] else [])
    # Stricter diagnostic, NOT part of the frozen criterion: the mask has six
    # supported candidates, so also try every single one as a constant. R17 only
    # probed 0/3/7; candidates 12 (+1 px/frame) and 16 (-1 px/frame) are the two
    # focus actions and are reported here so the blind spot stays visible.
    extra = {}
    for candidate in SUPPORTED:
        row = outcome(s, causal_loop(s, constant_policy(candidate)))
        extra[str(candidate)] = dict(action=HORIZONTAL[candidate], feasible=row['feasible'],
                                     survived=row['survived'], first_hit=row['first_hit'])
    evidence = dict(event_id=s.event_id, family=s.family, group=s.template_group, split=s.split,
                    mirror=bool(s.mirror), scenario=asdict(s), signature=list(signature(asdict(s))),
                    baselines={name: rows[name] for name in BASELINES}, weak=rows[WEAK],
                    passing_baselines=passing, extra_constants=extra,
                    any_constant_feasible=[c for c, v in extra.items() if v['feasible']])
    if passing and not force_reference:
        evidence['verdict'] = 'invalid'
        evidence['reasons'] = ['baseline_passed:' + name for name in passing]
        evidence['reference'] = dict(status='not_evaluated', reason='baseline_passed_before_reference')
        return evidence
    ref = solve(s)
    witness = list(ref['actions'])
    replayed = verify_actions(s, witness) == 'feasible' if witness else False
    complete = ref['status'] == 'feasible' and len(witness) == s.horizon // 2 and replayed
    evidence['reference'] = dict(status=ref['status'], reason=ref['reason'], checked=ref['checked'],
                                 actions=witness, witness_replay=replayed,
                                 complete_actions=len(witness) == s.horizon // 2)
    if passing:
        evidence['verdict'] = 'invalid'
        evidence['reasons'] = ['baseline_passed:' + name for name in passing] + \
                              ['reference_witness_present' if complete else 'reference_no_complete_replay']
    elif not complete:
        evidence['verdict'] = 'unknown'
        evidence['reasons'] = ['reference_no_complete_replay:' + str(ref['status'])]
    else:
        evidence['verdict'] = 'valid'
        evidence['reasons'] = ['all_six_baselines_failed', 'reference_witness_replayed']
        evidence['reference']['differs_from'] = {name: witness != rows[name]['actions']
                                                 for name in BASELINES + (WEAK,)}
        evidence['reference']['changed_steps_vs_hold'] = sum(a != b for a, b in
                                                             zip(witness, rows['hold']['actions']))
    return evidence


def compact(evidence):
    """Manifest form: baseline feasible/survived/first_hit plus complete reference actions."""
    keep = ('feasible', 'survived', 'goal_satisfied', 'first_hit', 'decisions')
    return dict(event_id=evidence['event_id'], family=evidence['family'], group=evidence['group'],
                mirror=evidence['mirror'], signature=evidence['signature'],
                scenario=evidence['scenario'], verdict=evidence['verdict'],
                reasons=evidence['reasons'], passing_baselines=evidence['passing_baselines'],
                baselines={name: {key: evidence['baselines'][name][key] for key in keep}
                           for name in BASELINES},
                baseline_actions={name: evidence['baselines'][name]['actions'] for name in BASELINES},
                extra_constants=evidence['extra_constants'],
                any_constant_feasible=evidence['any_constant_feasible'],
                weak={**{key: evidence['weak'][key] for key in keep},
                      'teacher_first': evidence['weak']['teacher_first'],
                      'equivalent_baseline': evidence['weak']['equivalent_baseline'],
                      'actions': evidence['weak']['actions']},
                reference=evidence['reference'])


def _evidence_out(evidence):
    keep = ('feasible', 'survived', 'first_hit')
    return dict(verdict=evidence['verdict'], reasons=evidence['reasons'],
                baselines={name: {k: evidence['baselines'][name][k] for k in keep} for name in BASELINES},
                weak={k: evidence['weak'][k] for k in keep},
                any_constant_feasible=evidence.get('any_constant_feasible'),
                reference={k: evidence['reference'][k] for k in ('status', 'checked', 'witness_replay')
                           if k in evidence['reference']})


# ---------------------------------------------------------------- generation
def draw_geometry(family, rng):
    d = DOMAINS[family]
    if family == 'F1':
        coverage = tuple(rng.choice(d['coverage']))
        internal_gap = bool(rng.choice(d['internal_gap']))
        width = rng.choice(d['gap_width'])
        origin = rng.choice(d['gap_origin'])
        gap = (float(origin), float(origin + width))
        start = gap[0] - rng.choice(d['start_offset']) if rng.random() < .5 else \
            gap[1] + rng.choice(d['start_offset'])
        if not 12. <= start <= 372.:
            return None, 'start_out_of_bounds'
        return dict(family='F1', coverage=coverage, internal_gap=internal_gap, gap=gap,
                    wall_frame=int(rng.choice(d['wall_frame'])), start_x=float(start)), None
    if family == 'F2':
        return dict(family='F2', origin=tuple(rng.choice(d['origin'])),
                    fire_interval=int(rng.choice(d['fire_interval'])),
                    bullet_speed=float(rng.choice(d['bullet_speed'])),
                    burst=bool(rng.choice(d['burst'])),
                    start_x=float(rng.choice(d['start_x']))), None
    window_start = int(rng.choice(d['window_start']))
    length = int(rng.choice(d['window_length']))
    threshold = float(rng.choice(d['threshold']))
    offset = 2. * float(rng.choice(d['crossing_ratio'])) * window_start
    start = threshold + offset
    goal = threshold - float(rng.choice(d['goal_offset']))
    if not (threshold + 8. <= start <= 372. and goal >= 16.):
        return None, 'start_or_goal_out_of_bounds'
    return dict(family='F3', start_x=float(start), window=(window_start, window_start + length),
                threshold=threshold, goal=goal, fire_interval=int(rng.choice(d['fire_interval'])),
                origin=tuple(rng.choice(d['origin']))), None


def group_specs(family, candidate, geometry):
    group = f'wf18-{family}-{candidate:03d}'
    base = dict(event_id=group + '-m0', template_group=group, seed=18000 + candidate,
                split=SPLIT, mirror=False, depletion_pressure=family == 'F3')
    base.update(geometry)
    rows = []
    for mirror in (False, True):
        spec = dict(base, mirror=mirror, event_id=group + f'-m{int(mirror)}')
        spec['start_x'] = 384. - geometry['start_x'] if mirror else geometry['start_x']
        rows.append(spec)
    return rows


def _geometry_out(spec):
    keys = ('family', 'start_x', 'coverage', 'gap', 'wall_frame', 'internal_gap', 'origin',
            'fire_interval', 'bullet_speed', 'burst', 'window', 'threshold', 'goal')
    return {k: spec[k] for k in keys if k in spec}


def search_family(family, seed=None, budget=None, target=None):
    seed = SEEDS[family] if seed is None else seed
    budget = BUDGET[family] if budget is None else budget
    target = TARGET_GROUPS if target is None else target
    rng = random.Random(seed)
    accepted, discarded = [], []
    old = {signature(spec) for rows in existing_signatures().values() for spec in rows}
    seen = set()
    drawn = 0
    for candidate in range(budget):
        if len(accepted) >= target:
            break
        drawn += 1
        geometry, reason = draw_geometry(family, rng)
        if geometry is None:
            discarded.append(dict(family=family, candidate=candidate, geometry=None,
                                  verdict='discarded', reasons=[reason]))
            continue
        specs = group_specs(family, candidate, geometry)
        sig = signature(specs[0])
        if sig != signature(specs[1]):
            raise AssertionError('mirror pair geometry mismatch')
        if sig in old or sig in seen:
            discarded.append(dict(family=family, candidate=candidate, geometry=_geometry_out(specs[0]),
                                  verdict='discarded', reasons=['duplicate_existing_or_accepted_signature']))
            continue
        seen.add(sig)
        base = evaluate_world(make_world(specs[0]))
        if base['verdict'] != 'valid':
            discarded.append(dict(family=family, candidate=candidate, geometry=_geometry_out(specs[0]),
                                  verdict=base['verdict'], reasons=['base:' + r for r in base['reasons']],
                                  base=_evidence_out(base), mirror='not_evaluated:base_rejected'))
            continue
        mirror = evaluate_world(make_world(specs[1]))
        if mirror['verdict'] != 'valid':
            discarded.append(dict(family=family, candidate=candidate, geometry=_geometry_out(specs[0]),
                                  verdict=mirror['verdict'], reasons=['mirror:' + r for r in mirror['reasons']],
                                  base=_evidence_out(base), mirror=_evidence_out(mirror)))
            continue
        accepted.append(dict(family=family, group=specs[0]['template_group'], candidate=candidate,
                             signature=list(sig), base=base, mirror=mirror))
    return dict(family=family, seed=seed, budget=budget, target=target, drawn=drawn,
                accepted=accepted, discarded=discarded)


def assert_disjoint(specs):
    """Hard pre-registered assertion: complete signatures never repeat a frozen set."""
    old = existing_signatures()
    old_sigs = {name: {signature(spec) for spec in rows} for name, rows in old.items()}
    union = set().union(*old_sigs.values())
    groups = {}
    for spec in specs:
        groups.setdefault(spec['template_group'], []).append(spec)
    for group, rows in groups.items():
        if len(rows) != 2:
            raise ValueError('each group needs exactly two mirrors: ' + group)
        if {bool(r['mirror']) for r in rows} != {False, True}:
            raise ValueError('mirror flags must differ: ' + group)
        if len({r['split'] for r in rows}) != 1:
            raise ValueError('mirror pair split mismatch: ' + group)
        if signature(rows[0]) != signature(rows[1]):
            raise ValueError('mirror pair signature mismatch: ' + group)
        if signature(rows[0]) in union:
            raise ValueError('world duplicates a frozen parameter signature: ' + group)
    new = {signature(rows[0]) for rows in groups.values()}
    if new & union:
        raise ValueError('accepted worlds duplicate frozen signatures')
    if len(new) != len(groups):
        raise ValueError('accepted groups must have distinct signatures')
    if not old_sigs['r10'] <= old_sigs['r8']:
        raise ValueError('R10 records must remain an R8 development/validation subset')
    return dict(r8_development_validation_signatures=len(old_sigs['r8']),
                r9_confirmation_signatures=len(old_sigs['r9']),
                r10_records_signatures=len(old_sigs['r10']),
                r16_confirmation_signatures=len(old_sigs['r16_confirmation']),
                r10_subset_of_r8=True, frozen_union=len(union),
                new_groups=len(groups), new_events=len(specs), duplicates=0,
                fields=','.join(SIGNATURE_FIELDS))


# ---------------------------------------------------------------- old worlds
def old_world_sets():
    """The two sets R17 probed: R11 validation 8 and R16 confirmation 24."""
    r11 = json.loads(R11.read_text(encoding='utf-8'))
    events = [dict(set='r11_validation', event_id=e['event_id'], scenario=e['scenario'])
              for e in r11['events']]
    prereg = json.loads(R16_PREREG.read_text(encoding='utf-8'))
    worlds = r16.confirmation_worlds()
    if r16.worlds_digest(worlds) != prereg['confirmation_worlds']['worlds_sha256']:
        raise ValueError('R16 confirmation worlds changed')
    stored = {e['event_id']: e['scenario'] for e in prereg['confirmation_worlds']['events']}
    for row in worlds:
        if stored[row['event_id']] != json.loads(json.dumps(row['scenario'])):
            raise ValueError('R16 scenario drift: ' + row['event_id'])
        events.append(dict(set='r16_confirmation', event_id=row['event_id'], scenario=row['scenario']))
    return events


def probe_old_sets(limit=None):
    rows = []
    for entry in old_world_sets()[:limit]:
        s = make_world(entry['scenario'])
        evidence = evaluate_world(s, force_reference=True)
        rows.append(dict(set=entry['set'], event_id=entry['event_id'], family=s.family,
                         verdict=evidence['verdict'], passing_baselines=evidence['passing_baselines'],
                         reference=evidence['reference']['status'],
                         reference_checked=evidence['reference'].get('checked'),
                         reference_actions=len(evidence['reference'].get('actions', [])),
                         any_constant_feasible=evidence['any_constant_feasible'],
                         baselines={name: {k: evidence['baselines'][name][k]
                                           for k in ('feasible', 'survived', 'first_hit')}
                                    for name in BASELINES},
                         weak={**{k: evidence['weak'][k] for k in ('feasible', 'survived', 'first_hit')},
                               'equivalent_baseline': evidence['weak']['equivalent_baseline']}))
    return rows


def summarise(rows):
    result = {}
    for family in ('F1', 'F2', 'F3', 'all'):
        group = [r for r in rows if family == 'all' or r['family'] == family]
        entry = dict(events=len(group))
        for name in BASELINES:
            entry[name] = dict(feasible=sum(r['baselines'][name]['feasible'] for r in group),
                               survived=sum(r['baselines'][name]['survived'] for r in group),
                               first_hit_present=sum(r['baselines'][name]['first_hit'] is not None
                                                     for r in group))
        entry[WEAK] = dict(feasible=sum(r['weak']['feasible'] for r in group),
                           survived=sum(r['weak']['survived'] for r in group),
                           first_hit_present=sum(r['weak']['first_hit'] is not None for r in group))
        entry['reference_feasible'] = sum(r['reference'] == 'feasible' for r in group)
        entry['reference_unknown'] = sum(r['reference'] == 'unknown' for r in group)
        entry['reference_not_evaluated'] = sum(r['reference'] == 'not_evaluated' for r in group)
        entry['baseline_passes_any'] = sum(bool(r['passing_baselines']) for r in group)
        entry['filtered_out'] = sum(r['verdict'] != 'valid' for r in group)
        entry['constant_passable'] = sum(bool(r.get('any_constant_feasible')) for r in group)
        result[family] = entry
    return result


def r17_cross_check(old_rows):
    """The three R17 constants must reproduce the pre-existing constant-probe counts.

    R17 ran hold/right/left through the real rendered run_event on the same worlds;
    matching its counts here proves the pixel-free loop used by this filter has the
    same closed-loop semantics.
    """
    probe = json.loads(R17_PROBE.read_text(encoding='utf-8'))
    checked, mismatches = {}, []
    for set_name in ('r11_validation', 'r16_confirmation'):
        rows = [r for r in old_rows if r['set'] == set_name]
        for family in ('F1', 'F2', 'F3', 'all'):
            group = [r for r in rows if family == 'all' or r['family'] == family]
            for name in CONSTANTS:
                expected = probe['sets'][set_name]['summary'][family][name]
                actual = dict(events=len(group),
                              feasible=sum(r['baselines'][name]['feasible'] for r in group))
                key = f'{set_name}/{family}/{name}'
                checked[key] = dict(r17=expected, recomputed=actual)
                if expected != actual:
                    mismatches.append(key)
    return dict(source=str(R17_PROBE), matches=not mismatches, mismatches=mismatches,
                compared=len(checked), constants=list(CONSTANTS), detail=checked)


def _new_rows(valid_rows):
    """Uniform row shape so the old and rebuilt sets share one comparison table."""
    keep = ('feasible', 'survived', 'first_hit')
    return [dict(set='worldfilter_round18', event_id=r['event_id'], family=r['family'],
                 group=r['group'], mirror=r['mirror'], verdict=r['verdict'],
                 passing_baselines=r['passing_baselines'], reference=r['reference']['status'],
                 baselines={n: {k: r['baselines'][n][k] for k in keep} for n in BASELINES},
                 weak={**{k: r['weak'][k] for k in keep},
                       'equivalent_baseline': r['weak']['equivalent_baseline']},
                 any_constant_feasible=r['any_constant_feasible']) for r in valid_rows]


def comparison(old_rows, valid_rows):
    new_rows = _new_rows(valid_rows)
    sets = {}
    for name in ('r11_validation', 'r16_confirmation'):
        sets[name] = dict(rows=[r for r in old_rows if r['set'] == name],
                          summary=summarise([r for r in old_rows if r['set'] == name]))
    sets['combined_old'] = dict(rows=old_rows, summary=summarise(old_rows))
    return dict(schema='v373-world-filter-comparison/1',
                old_sets=dict(sets=sets,
                              note='the eight R11 validation worlds and the twenty-four R16 confirmation '
                                   'worlds, re-probed read-only with the identical six baselines; the '
                                   'reference solver is forced on every old world to show that old '
                                   'worlds are solvable while constants already pass'),
                new_worlds=dict(rows=new_rows, summary=summarise(new_rows),
                                note='every retained world has all six baselines failing and one '
                                     'complete replayed reference witness'))


# ---------------------------------------------------------------- pre-register
def criterion():
    return dict(
        baselines=dict(constants=CONSTANTS, supported_candidates=list(SUPPORTED),
                       horizontal_mask=HORIZONTAL,
                       alternate=dict(start='left', block_decisions=ALTERNATE_BLOCK,
                                      block_logical_frames=2 * ALTERNATE_BLOCK,
                                      convention='R8/R11 broad oscillation: switching every 16 '
                                                 'decisions (32 logical frames); one decision is one '
                                                 'two-frame advance'),
                       random=dict(seed=RANDOM_SEED, support=list(SUPPORTED), uniform=True,
                                   sequence=random_candidates(RANDOM_SEED, HORIZON // 2),
                                   note='a fresh Random(seed) per world, so every world sees the same '
                                        'eighty-candidate stream'),
                       weak=WEAK,
                       weak_definition='TacticalTeacher().decide(world, frame 0).action repeated for '
                                       'the whole horizon and projected onto the supported candidate '
                                       'with the same mask; one privileged query, not a label'),
        loop='unchanged R11 semantics: mask the eighteen logits onto the six supported candidates, '
             'argmax, project candidate -> two-frame action, advance the real world, sticky collision, '
             'continue to the full horizon 160; a non-learning baseline needs no pixels',
        feasible='full horizon and no collision and (F3 only) canonical x <= goal at some sampled '
                 'state, identical to R11 run_event',
        world_valid='all five non-learning baselines fail AND the weak teacher-first baseline fails '
                    'AND the independent reference solver returns a complete action sequence that '
                    'replays to feasible without early termination',
        world_invalid='at least one of the six baselines is feasible (the world is too easy)',
        world_unknown='all six baselines fail but no reference witness exists (bounded program '
                      'search); unknown is discarded and is never reported as unsolvable',
        reference='tactical_teacher_validation.solve (independent bounded program class, no teacher '
                  'geometry consulted) plus an explicit non-early verify() replay of every witness',
        discard='mirror pairs share one group and one split; a group is retained only if base and '
                'mirror both come out valid; duplicates of any frozen signature are discarded '
                'before evaluation',
        no_relaxation='if a family cannot reach six valid groups inside its budget the actual number '
                      'and the per-candidate reasons are reported; no threshold is loosened',
        extra_constant_diagnostic='NOT part of the criterion: every supported candidate (0,3,7,9,12,16) '
                                  'is also run as a single constant and reported per world, because the '
                                  'requested five baselines cover only three of the six supported '
                                  'candidates; a retained world that a constant 12/16 still passes is '
                                  'reported as constant_passable and is never called constant-free')


def identities():
    paths = [Path(__file__), Path('tests/test_v373_world_filter.py'),
             Path('tools/rl/v373_seed_confirm.py'), Path('touhou_rl/tactical_teacher_validation.py'),
             Path('touhou_rl/tactical_scenarios.py'), Path('touhou_rl/tactical_teacher.py'),
             Path('touhou_rl/tactical_teacher_v2.py'), Path('touhou_rl/tactical_synthetic_closed_loop.py'),
             Path('touhou_rl/tactical_data.py'), R11, R10_RECORDS, R9_MANIFEST, R16_PREREG, R17_PROBE]
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError('pre-registration identity missing: ' + ','.join(missing))
    return {str(p): file_sha(p) for p in paths}


def preregister(out=OUT):
    out = new_dir(out)
    value = dict(schema=PREREG_SCHEMA, identities=identities(),
                 purpose='world-validity filtering and curriculum rebuild only; no training, no policy '
                         'evaluation, no teacher labels, no deployable checkpoint, no game execution',
                 criterion=criterion(),
                 generation=dict(target_groups_per_family=TARGET_GROUPS, mirrors=2,
                                 target_events=TARGET_GROUPS * 2 * 3, budget_per_family=BUDGET,
                                 seeds=SEEDS, split=SPLIT, horizon=HORIZON, domains=DOMAINS,
                                 parameter_space='ValidationWorld fields only; F1 coverage/wall_frame/'
                                                 'internal_gap/gap/start, F2 origin/fire_interval/'
                                                 'bullet_speed/burst/start, F3 window/threshold/goal/'
                                                 'start (plus fire_interval/origin); unused fields stay '
                                                 'at their ValidationWorld defaults',
                                 proposal='seeded uniform draws per declared domain; F1 start is drawn '
                                          'outside the internal gap; F3 geometry is drawn so a constant '
                                          'drift crosses the threshold before the window opens. The '
                                          'filter alone decides validity; the proposal only avoids '
                                          'obviously dead geometry',
                                 unknown_policy='unknown candidates stay discarded and are counted'),
                 disjointness=assert_disjoint([]),
                 signatures=dict(function='ValidationWorld fields, R9 confirmation signature',
                                 fields=list(SIGNATURE_FIELDS),
                                 cross_checked_against='tools.rl.v373_seed_confirm.signature',
                                 frozen_sets=['r8_development_validation', 'r9_confirmation',
                                              'r10_records', 'r16_confirmation']),
                 execution=dict(order=['tests', 'preregister', 'run'],
                                overwrite='directories exist_ok=False, JSON opened in x mode',
                                reruns='exactly one generation run; no rerun, no retune'),
                 development_only=True, test_split=False, sealed_tests_constructed=0,
                 training_steps=0, teacher_labels=0, policy_evaluations=0, formal_gate=False,
                 deployable=False, game=False)
    write_json(out / 'preregistration.json', value)
    return value


def verify_prereg(out=OUT):
    out = Path(out)
    p = json.loads((out / 'preregistration.json').read_text(encoding='utf-8'))
    if p.get('schema') != PREREG_SCHEMA:
        raise ValueError('pre-registration schema')
    for path, sha in p['identities'].items():
        if file_sha(path) != sha:
            raise ValueError('frozen identity changed: ' + path)
    c = p['criterion']
    if c['baselines']['random']['sequence'] != random_candidates(RANDOM_SEED, HORIZON // 2):
        raise ValueError('random baseline sequence drift')
    if c['baselines']['supported_candidates'] != list(SUPPORTED) or \
            {int(k): v for k, v in c['baselines']['horizontal_mask'].items()} != HORIZONTAL:
        raise ValueError('mask drift')
    g = p['generation']
    if g['seeds'] != SEEDS or g['budget_per_family'] != BUDGET or g['domains'] != DOMAINS:
        raise ValueError('search procedure changed after pre-registration')
    if g['target_groups_per_family'] != TARGET_GROUPS or g['horizon'] != HORIZON:
        raise ValueError('target changed after pre-registration')
    return p


def run(out=OUT):
    out = Path(out)
    prereg = verify_prereg(out)
    dest = new_dir(out / 'run')
    searches = [search_family(family) for family in ('F1', 'F2', 'F3')]
    accepted = [row for search in searches for row in search['accepted']]
    specs = [row[key]['scenario'] for row in accepted for key in ('base', 'mirror')]
    if any(spec['split'] == 'test' for spec in specs):
        raise AssertionError('test split must never be constructed')
    disjoint = assert_disjoint(specs)
    compact_rows = {}
    for family in ('F1', 'F2', 'F3'):
        directory = new_dir(dest / 'filters' / family)
        for row in accepted:
            if row['family'] != family:
                continue
            for key in ('base', 'mirror'):
                evidence = row[key]
                compact_rows[evidence['event_id']] = compact(evidence)
                write_json(directory / (evidence['event_id'] + '.json'), compact(evidence))
    valid_rows = [compact_rows[e] for f in ('F1', 'F2', 'F3')
                  for e in sorted(compact_rows) if compact_rows[e]['family'] == f]
    old_rows = probe_old_sets()
    cross = r17_cross_check(old_rows)
    if not cross['matches']:
        raise AssertionError('R17 constant-probe cross-check failed: ' + ','.join(cross['mismatches']))
    counts = {}
    for search in searches:
        family = search['family']
        rows = [r for r in valid_rows if r['family'] == family]
        counts[family] = dict(
            drawn=search['drawn'], budget=search['budget'], groups=len(search['accepted']),
            events=len(rows), target_events=search['target'] * 2,
            discarded=len(search['discarded']),
            discarded_invalid=sum(d['verdict'] == 'invalid' for d in search['discarded']),
            discarded_unknown=sum(d['verdict'] == 'unknown' for d in search['discarded']),
            discarded_duplicate=sum('duplicate' in ' '.join(d['reasons']) for d in search['discarded']),
            discarded_unusable_draw=sum(d['verdict'] == 'discarded' and
                                        'duplicate' not in ' '.join(d['reasons'])
                                        for d in search['discarded']),
            reference_witnesses=sum(r['reference']['status'] == 'feasible' for r in rows),
            baseline_passes=sum(v['feasible'] for r in rows for v in r['baselines'].values()),
            weak_passes=sum(r['weak']['feasible'] for r in rows),
            constant_passable=sum(bool(r['any_constant_feasible']) for r in rows),
            constant_pass_pairs=sum(len(r['any_constant_feasible']) for r in rows),
            weak_equivalent_to_baseline=sum(r['weak'].get('equivalent_baseline') is not None
                                             for r in rows),
            all_three_constants_hit=sum(all(r['baselines'][n]['first_hit'] is not None
                                            for n in ('hold', 'right', 'left')) for r in rows))
    totals = dict(events=len(valid_rows), groups=len(accepted),
                  baseline_passes=sum(v['feasible'] for r in valid_rows for v in r['baselines'].values()),
                  weak_passes=sum(r['weak']['feasible'] for r in valid_rows),
                  reference_witnesses=sum(r['reference']['status'] == 'feasible' for r in valid_rows),
                  all_three_constants_hit=sum(all(r['baselines'][n]['first_hit'] is not None
                                                  for n in ('hold', 'right', 'left')) for r in valid_rows),
                  all_six_baselines_hit=sum(all(r['baselines'][n]['first_hit'] is not None
                                                for n in BASELINES) and
                                            r['weak']['first_hit'] is not None for r in valid_rows),
                  constant_passable=sum(bool(r['any_constant_feasible']) for r in valid_rows),
                  constant_pass_pairs=sum(len(r['any_constant_feasible']) for r in valid_rows),
                  changed_steps_vs_hold_min=min(r['reference']['changed_steps_vs_hold']
                                                for r in valid_rows) if valid_rows else None)
    manifest = dict(schema=RESULT_SCHEMA, prereg_sha256=file_sha(out / 'preregistration.json'),
                    criterion=prereg['criterion'],
                    criterion_signature_fields=list(SIGNATURE_FIELDS),
                    disjointness=disjoint,
                    searches=[dict(family=s['family'], seed=s['seed'], budget=s['budget'],
                                   target=s['target'], drawn=s['drawn'],
                                   accepted=[dict(group=r['group'], candidate=r['candidate'],
                                                  signature=r['signature'],
                                                  events=[r['base']['event_id'], r['mirror']['event_id']])
                                             for r in s['accepted']],
                                   discarded=s['discarded']) for s in searches],
                    worlds=valid_rows, counts=counts, totals=totals,
                    r17_cross_check=cross,
                    development_only=True, test_split=False, training_steps=0, teacher_labels=0,
                    formal_gate=False, deployable=False, game=False)
    summary = dict(schema=RESULT_SCHEMA + '-summary', prereg_sha256=manifest['prereg_sha256'],
                   totals=totals, counts=counts, disjointness=disjoint,
                   r17_cross_check=dict(matches=cross['matches'], compared=cross['compared'],
                                        mismatches=cross['mismatches']),
                   old_sets=comparison(old_rows, valid_rows)['old_sets']['sets'],
                   new_worlds=comparison(old_rows, valid_rows)['new_worlds']['summary'],
                   identities_unchanged=all(file_sha(p) == s for p, s in prereg['identities'].items()),
                   development_only=True, test_split=False, training_steps=0, teacher_labels=0,
                   formal_gate=False, deployable=False, game=False)
    write_json(dest / 'manifest.json', manifest)
    write_json(dest / 'comparison.json', comparison(old_rows, valid_rows))
    write_json(dest / 'summary.json', summary)
    write_json(dest / 'sha256.json', {str(f.relative_to(dest)): file_sha(f)
                                      for f in sorted(dest.rglob('*')) if f.is_file()})
    return summary


def complete(out=OUT):
    """Read-only correction of a derived field in the already-frozen run.

    The single generation run wrote every world, verdict and witness correctly, but
    the old-set probe row omitted `any_constant_feasible`, so the old-set
    `constant_passable` column of run/comparison.json and run/summary.json is stuck
    at 0. This command rebuilds ONLY the derived comparison from stored artifacts
    plus read-only re-probes: it never searches for worlds, never evaluates a
    retained world again, never trains and never overwrites the run directory.
    Exactly this module and its contract test may differ from the pre-registration.
    """
    out = Path(out)
    prereg = json.loads((out / 'preregistration.json').read_text(encoding='utf-8'))
    if prereg.get('schema') != PREREG_SCHEMA:
        raise ValueError('pre-registration schema')
    changed = {}
    allowed = {Path(__file__).resolve(), Path('tests/test_v373_world_filter.py').resolve()}
    for path, sha in prereg['identities'].items():
        current = file_sha(path)
        if current == sha:
            continue
        if Path(path).resolve() not in allowed:
            raise ValueError('frozen identity changed: ' + path)
        changed[str(path)] = dict(preregistered_sha256=sha, current_sha256=current)
    if not changed:
        raise ValueError('completion expects the documented probe-field fix')
    run = out / 'run'
    stored_sha = json.loads((run / 'sha256.json').read_text(encoding='utf-8'))
    drift = [key for key, sha in stored_sha.items() if file_sha(run / key) != sha]
    if drift:
        raise ValueError('generation artifacts changed: ' + ','.join(drift))
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    stored_summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    worlds = manifest['worlds']
    if len(worlds) != 36 or any(w['verdict'] != 'valid' for w in worlds):
        raise ValueError('expected the thirty-six frozen valid worlds')
    if any(v['feasible'] for w in worlds for v in w['baselines'].values()):
        raise ValueError('a retained world has a passing baseline')
    old_rows = probe_old_sets()
    cross = r17_cross_check(old_rows)
    if not cross['matches']:
        raise AssertionError('R17 cross-check failed')
    rebuilt = comparison(old_rows, worlds)
    for family in ('F1', 'F2', 'F3', 'all'):
        for key, value in rebuilt['new_worlds']['summary'][family].items():
            if value != stored_summary['new_worlds'][family][key]:
                raise ValueError(f'new-world table drift: {family}/{key}')
    correction = new_dir(out / 'run-correction')
    corrected = dict(stored_summary, old_sets=rebuilt['old_sets']['sets'],
                     new_worlds=rebuilt['new_worlds']['summary'],
                     identities_unchanged='every frozen identity except tools/rl/v373_world_filter.py and '
                                          'tests/test_v373_world_filter.py still matches; see correction',
                     correction=dict(reason='the old-set probe row omitted any_constant_feasible, so the '
                                            'old-set constant_passable column of the frozen run is zero',
                                     generation_rerun=False, search_rerun=False, worlds_reevaluated=False,
                                     run_directory_rewritten=False,
                                     changed_identities=changed,
                                     generation_artifact_sha256=stored_sha,
                                     manifest_new_world_table_matches_stored=True))
    write_json(correction / 'comparison.json', rebuilt)
    write_json(correction / 'summary.json', corrected)
    write_json(correction / 'notes.json', dict(
        schema='v373-world-filter-correction/1',
        reason=corrected['correction']['reason'],
        old_set_constant_passable={set_name: {family: value['constant_passable']
                                              for family, value in sets['summary'].items()}
                                   for set_name, sets in rebuilt['old_sets']['sets'].items()},
        new_set_constant_passable={family: value['constant_passable']
                                   for family, value in rebuilt['new_worlds']['summary'].items()},
        r17_cross_check=dict(matches=cross['matches'], compared=cross['compared']),
        changed_identities=changed,
        generation_rerun=False, search_rerun=False, worlds_reevaluated=False,
        run_artifacts_unchanged=True, run_artifacts=len(stored_sha),
        correction_artifact_sha256={str(f.relative_to(correction)): file_sha(f)
                                    for f in sorted(correction.rglob('*')) if f.is_file()}))
    write_json(correction / 'sha256.json', {str(f.relative_to(correction)): file_sha(f)
                                            for f in sorted(correction.rglob('*'))
                                            if f.is_file() and f.name != 'sha256.json'})
    return corrected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preregister', 'run', 'complete'))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == 'preregister':
        value = preregister(args.out)
        print(json.dumps({'schema': value['schema'], 'criterion': list(value['criterion']),
                          'budget': value['generation']['budget_per_family']}, indent=2))
    elif args.command == 'run':
        print(json.dumps(run(args.out), indent=2))
    else:
        value = complete(args.out)
        print(json.dumps({'schema': value['schema'], 'correction': value['correction']}, indent=2))


if __name__ == '__main__':
    main()
