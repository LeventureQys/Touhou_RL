"""Round19: can the two frozen teachers actually play the twenty-eight round18 worlds?

Round18 rebuilt the fixed-y development world set so that no constant policy (and
no alternation, no seeded random walk, no repeat-the-teacher's-first-action)
solves any of the twenty-eight worlds, while an independent bounded reference
program still finds a complete witness. Round19 asks the next question without
touching that evidence:

  1. Read the frozen round18 ``run/manifest.json`` (SHA checked against the
     round18 ``run/sha256.json``) and load exactly its twenty-eight worlds. No
     world is regenerated, no parameter is changed, no world is replaced.
  2. Drive the two frozen teachers -- ``TacticalTeacher`` and
     ``TacticalTeacherV2``, both unmodified -- through every world and record
     each decision (action / option / trigger / confidence), the complete
     hundred-and-sixty logical-frame action stream, an independent ``verify``
     replay, ``first_hit``, goal and survival.
  3. Re-run the six baselines frozen by round18 through round18's own
     ``policies`` / ``causal_loop`` / ``outcome`` and require them to reproduce
     the stored numbers exactly; that is the "this is still the same world and
     the same physics" gate.
  4. Mark a success degenerate when the whole episode used a single action. A
     constant teacher episode is by construction the same trajectory as the
     corresponding round18 constant baseline, so it is never teacher capability.
  5. Freeze a train/validation grouping of the fourteen groups (one group = base
     + mirror, atomic) by alternating family-internal index, and write it to
     ``run/split.json`` with the parameter signature of every group.

Research only: synthetic development worlds, no training, no trained policy, no
checkpoint, no teacher labels, no formal gate, no game execution. Nothing here is
deployable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_scenarios import initial_state
from touhou_rl.tactical_teacher import TacticalTeacher
from touhou_rl.tactical_teacher_validation import advance, make_world, metrics
from touhou_rl.tactical_teacher_validation import verify as verify_actions
from touhou_rl.tactical_teacher_v2 import TacticalTeacherV2
from tools.rl import v373_world_filter as r18

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = 'Document/Update/Alpha/v3.7.3/evidence'
REPORTS = 'Document/Update/Alpha/v3.7.3/reports'
R18 = EVIDENCE + '/worldfilter-round18'
MANIFEST = R18 + '/run/manifest.json'
R18_SHA = R18 + '/run/sha256.json'
R18_PREREG = R18 + '/preregistration.json'
R18_REPORT = REPORTS + '/P1-worldfilter-round18.md'
REPORT = REPORTS + '/P2-validteacher-round19.md'
OUT = EVIDENCE + '/validteacher-round19'

PREREG_SCHEMA = 'v373-valid-teacher-prereg/1'
WORLD_SCHEMA = 'v373-valid-teacher-world/1'
SUMMARY_SCHEMA = 'v373-valid-teacher-summary/1'
SPLIT_SCHEMA = 'v373-valid-teacher-split/1'
R18_SCHEMA = 'v373-world-filter-run/1'

FAMILIES = ('F1', 'F2', 'F3')
EXPECTED_EVENTS = {'F1': 12, 'F2': 4, 'F3': 12}
EXPECTED_GROUPS = {'F1': 6, 'F2': 2, 'F3': 6}
HORIZON = 160
DECISIONS = HORIZON // 2
ACTIONS = (-2, -1, 0, 1, 2)
# action -> supported candidate, the frozen round18 projection (same table).
CANDIDATE = dict(r18.ACTION_CANDIDATE)
TEACHERS = {'teacher': TacticalTeacher, 'teacher_v2': TacticalTeacherV2}
TEACHER_FILES = {'teacher': 'touhou_rl/tactical_teacher.py',
                 'teacher_v2': 'touhou_rl/tactical_teacher_v2.py'}
BASELINES = tuple(r18.BASELINES)
WEAK = r18.WEAK
SPLITS = ('train', 'validation')


# ---------------------------------------------------------------- plumbing
def repo_path(relative):
    path = Path(str(relative))
    return path if path.is_absolute() else ROOT / path


def write_json(path, value):
    """Exclusive write: a second run of this tool fails instead of overwriting evidence."""
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def new_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def actions_sha(actions):
    return hashlib.sha256(json.dumps(list(actions)).encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- input
@lru_cache(maxsize=1)
def load_worlds():
    """Read-only load of the frozen round18 worlds, with every count asserted."""
    manifest_path = repo_path(MANIFEST)
    if not manifest_path.is_file():
        raise FileNotFoundError('round18 manifest missing: ' + MANIFEST)
    manifest_sha = file_sha(manifest_path)
    stored = json.loads(repo_path(R18_SHA).read_text(encoding='utf-8'))
    if stored.get('manifest.json') != manifest_sha:
        raise ValueError('round18 manifest does not match its frozen sha256.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != R18_SCHEMA:
        raise ValueError('round18 manifest schema')
    if manifest.get('test_split') is not False or manifest.get('development_only') is not True:
        raise ValueError('round18 manifest is not development only')
    worlds = manifest['worlds']
    if len(worlds) != 28 or len(worlds) != sum(EXPECTED_EVENTS.values()):
        raise ValueError('round18 world count changed')
    for family in FAMILIES:
        rows = [world for world in worlds if world['family'] == family]
        if len(rows) != EXPECTED_EVENTS[family]:
            raise ValueError('round18 event count changed: ' + family)
        if len({world['group'] for world in rows}) != EXPECTED_GROUPS[family]:
            raise ValueError('round18 group count changed: ' + family)
        if manifest['counts'][family]['events'] != EXPECTED_EVENTS[family]:
            raise ValueError('round18 counts table disagrees with the world list: ' + family)
    if len({world['group'] for world in worlds}) != 14:
        raise ValueError('round18 group count changed')
    for world in worlds:
        if world['verdict'] != 'valid' or 'all_six_baselines_failed' not in world['reasons']:
            raise ValueError('a retained world is not valid: ' + world['event_id'])
        if world['scenario']['split'] != 'development':
            raise ValueError('the round18 set must stay development only: ' + world['event_id'])
        if any(row['feasible'] for row in world['baselines'].values()):
            raise ValueError('a round18 baseline passes a retained world: ' + world['event_id'])
        if world['weak']['feasible']:
            raise ValueError('the round18 weak baseline passes a retained world: ' + world['event_id'])
        if world['any_constant_feasible']:
            raise ValueError('a round18 constant passes a retained world: ' + world['event_id'])
        reference = world['reference']
        if reference['status'] != 'feasible' or reference['witness_replay'] is not True:
            raise ValueError('a retained world has no complete witness: ' + world['event_id'])
        if reference['complete_actions'] is not True or len(reference['actions']) != DECISIONS:
            raise ValueError('a retained witness is not complete: ' + world['event_id'])
    ordered = sorted(worlds, key=lambda world: (world['family'], world['group'], int(bool(world['mirror']))))
    return dict(manifest_sha256=manifest_sha, sha256_table=stored, worlds=ordered,
                counts=manifest['counts'], totals=manifest['totals'],
                r18_prereg_sha256=file_sha(repo_path(R18_PREREG)))


def group_rows(worlds):
    """One group = base + mirror, same signature, same split, complementary start_x."""
    groups = {}
    for world in worlds:
        groups.setdefault(world['group'], []).append(world)
    for name, rows in groups.items():
        family = name.split('-')[1]
        if len(rows) != 2:
            raise ValueError('a group must hold exactly the two mirrors: ' + name)
        if {bool(row['mirror']) for row in rows} != {False, True}:
            raise ValueError('mirror flags must differ inside a group: ' + name)
        if {row['family'] for row in rows} != {family}:
            raise ValueError('group family mismatch: ' + name)
        if len({row['scenario']['split'] for row in rows}) != 1:
            raise ValueError('mirror pair split mismatch: ' + name)
        if rows[0]['signature'] != rows[1]['signature']:
            raise ValueError('mirror pair signature mismatch: ' + name)
        if abs(rows[0]['scenario']['start_x'] + rows[1]['scenario']['start_x'] - 384.) > 1e-9:
            raise ValueError('mirror pair geometry mismatch: ' + name)
        if {row['scenario']['template_group'] for row in rows} != {name}:
            raise ValueError('template_group mismatch: ' + name)
    return groups


def family_index(group):
    return int(str(group).rsplit('-', 1)[1])


# ---------------------------------------------------------------- teachers
def episode(s, policy):
    """Drive one frozen teacher through the world with the verifier's own step function.

    Only ``policy.decide`` is called. The physics belong to the driver so both
    teachers and the independent verifier share one ``advance``. The teachers' own
    ``rollout`` helpers are deliberately unused: ``tactical_teacher.rollout`` steps
    through ``tactical_scenarios.advance``, whose coverage/gap/window/threshold are
    hard-coded defaults and therefore are not this world.
    """
    state = initial_state(s)
    actions, decisions = [], []
    while state.frame < s.horizon:
        decision = policy.decide(s, state)
        if decision.action not in ACTIONS:
            raise ValueError('teacher action outside the frozen two-frame action set')
        before = state
        state = advance(s, before, decision.action)
        actions.append(decision.action)
        decisions.append(dict(frame=before.frame, **asdict(decision),
                              candidate=CANDIDATE[decision.action],
                              x_before=before.x, x_after=state.x, hit=bool(state.hit),
                              actual_hold=state.frame - before.frame))
    if len(actions) != DECISIONS or state.frame != s.horizon:
        raise ValueError('episode did not run the full horizon: ' + s.event_id)
    return dict(actions=actions, decisions=decisions, state=state)


def classify_actions(actions, feasible):
    """Constant-episode detection, fixed before the run.

    A single-action episode is degenerate by definition; if it is also feasible the
    success is reported but never counted as teacher capability.
    """
    distinct = sorted(set(actions))
    single = len(distinct) == 1
    return dict(distinct_actions=distinct, single_action=single,
                constant_action=(distinct[0] if single else None),
                switches=sum(a != b for a, b in zip(actions, actions[1:])),
                degenerate=bool(feasible and single),
                capability_success=bool(feasible and not single))


def teacher_row(s, name):
    run = episode(s, TEACHERS[name]())
    actions = run['actions']
    replay = metrics(s, actions)
    survived = bool(replay['survived'])
    goal = bool(s.family != 'F3' or replay['reached_goal'])
    feasible = bool(survived and goal)
    if (replay['verdict'] == 'feasible') != feasible:
        raise AssertionError('independent verify disagrees with the recorded outcome: ' + s.event_id)
    shape = classify_actions(actions, feasible)
    row = dict(teacher=name, module=TEACHER_FILES[name],
               module_sha256=file_sha(repo_path(TEACHER_FILES[name])),
               decisions=run['decisions'], actions=actions, actions_sha256=actions_sha(actions),
               frame_actions=[a for a in actions for _ in range(2)],
               decisions_count=len(actions),
               option_histogram=dict(sorted(Counter(d['option'] for d in run['decisions']).items())),
               trigger_histogram=dict(sorted(Counter(d['trigger_reason'] for d in run['decisions']).items())),
               confidence=sorted({d['confidence'] for d in run['decisions']}),
               verify=replay['verdict'], survived=survived, feasible=feasible,
               goal_satisfied=goal, reached_goal=bool(replay['reached_goal']),
               first_hit=replay['first_collision_frame'],
               final_x=replay['trace'][-1]['x'], **shape)
    if len(row['frame_actions']) != HORIZON:
        raise AssertionError('the recorded action stream is not one hundred and sixty frames')
    return row


def constant_cross_check(world, row):
    """A constant teacher episode must be the round18 constant baseline, verbatim."""
    if not row['single_action']:
        return dict(single_action=False, consistent=None, candidates=[])
    entries = {key: value for key, value in world['extra_constants'].items()
               if value['action'] == row['constant_action']}
    if not entries:
        raise AssertionError('no round18 constant probe for action ' + str(row['constant_action']))
    consistent = bool(any(value['feasible'] for value in entries.values())) == bool(row['feasible'])
    if not consistent:
        raise AssertionError('a constant teacher episode disagrees with the round18 constant probe')
    return dict(single_action=True, consistent=True, candidates=sorted(entries),
                round18_feasible={key: value['feasible'] for key, value in entries.items()})


# ---------------------------------------------------------------- baselines
def baseline_rows(s, world):
    """Re-run the six round18 baselines through round18's own functions."""
    table, weak = r18.policies(s)
    rows = {}
    for name in BASELINES + (WEAK,):
        run = r18.causal_loop(s, table[name])
        row = r18.outcome(s, run)
        row['actions'] = run['actions']
        rows[name] = row
    rows[WEAK]['teacher_first'] = weak
    rows[WEAK]['equivalent_baseline'] = next(
        (name for name in BASELINES if rows[name]['actions'] == rows[WEAK]['actions']), None)
    mismatches = []
    for name in BASELINES + (WEAK,):
        stored = world['weak'] if name == WEAK else world['baselines'][name]
        produced = rows[name]
        for key in ('feasible', 'survived', 'goal_satisfied'):
            if bool(stored[key]) != bool(produced[key]):
                mismatches.append(name + '.' + key)
        if stored['first_hit'] != produced['first_hit']:
            mismatches.append(name + '.first_hit')
        if stored['decisions'] != produced['decisions']:
            mismatches.append(name + '.decisions')
        expected_actions = stored['actions'] if name == WEAK else world['baseline_actions'][name]
        if list(expected_actions) != list(produced['actions']):
            mismatches.append(name + '.actions')
    if world['weak']['teacher_first'] != rows[WEAK]['teacher_first']:
        mismatches.append('weak.teacher_first')
    if world['weak']['equivalent_baseline'] != rows[WEAK]['equivalent_baseline']:
        mismatches.append('weak.equivalent_baseline')
    return rows, mismatches


def reference_row(s, world):
    """Read-only replay of the round18 witness: is the family solvable at all?"""
    stored = world['reference']
    witness = list(stored.get('actions') or [])
    status = verify_actions(s, witness) if witness else None
    return dict(r18_status=stored['status'], r18_checked=stored.get('checked'),
                r18_witness_replay=stored.get('witness_replay'),
                witness_actions=len(witness), replay_status=status,
                complete=len(witness) == DECISIONS,
                matches=bool(stored['status'] == 'feasible' and status == 'feasible'
                             and len(witness) == DECISIONS))


# ---------------------------------------------------------------- per world
def evaluate_world(world):
    spec = world['scenario']
    if spec.get('split') != 'development':
        raise ValueError('only development worlds may be evaluated')
    s = make_world(spec)
    if s.horizon != HORIZON:
        raise ValueError('the round19 criterion is defined on horizon 160')
    baselines, mismatches = baseline_rows(s, world)
    teachers = {}
    for name in TEACHERS:
        row = teacher_row(s, name)
        row['constant_cross_check'] = constant_cross_check(world, row)
        teachers[name] = row
    reference = reference_row(s, world)
    return dict(schema=WORLD_SCHEMA, event_id=s.event_id, family=s.family, group=s.template_group,
                mirror=bool(s.mirror), split=s.split, signature=world['signature'],
                scenario=asdict(s),
                source=dict(manifest=MANIFEST, manifest_sha256=file_sha(repo_path(MANIFEST)),
                            r18_verdict=world['verdict'], r18_reasons=world['reasons'],
                            r18_any_constant_feasible=world['any_constant_feasible'],
                            r18_reference_status=world['reference']['status'],
                            r18_reference_checked=world['reference'].get('checked')),
                teachers=teachers, baselines=baselines,
                baseline_reproduction=dict(mismatches=mismatches, matches=not mismatches),
                reference=reference,
                flags=dict(synthetic_development_world=True, trained_policy=False, checkpoint=None,
                           training_steps=0, formal_gate=False, game=False))


# ---------------------------------------------------------------- split
def split_plan(worlds):
    """Alternate the family-internal group index: 0 train, 1 validation, 2 train, ..."""
    groups = group_rows(worlds)
    plan = []
    for family in FAMILIES:
        names = sorted((name for name in groups if groups[name][0]['family'] == family), key=family_index)
        for index, name in enumerate(names):
            rows = sorted(groups[name], key=lambda row: int(bool(row['mirror'])))
            plan.append(dict(
                group=name, family=family, index_in_family=index,
                split='train' if index % 2 == 0 else 'validation',
                signature=rows[0]['signature'],
                params=dict(zip(r18.SIGNATURE_FIELDS, rows[0]['signature'])),
                events=[dict(event_id=row['event_id'], mirror=bool(row['mirror']),
                             start_x=row['scenario']['start_x']) for row in rows]))
    return plan


def assert_split(plan, worlds):
    groups = group_rows(worlds)
    if len(plan) != 14 or len(groups) != 14:
        raise ValueError('the split plan must cover exactly the fourteen groups')
    if len({row['group'] for row in plan}) != len(plan):
        raise ValueError('a group appears twice in the split plan')
    event_split = {}
    for row in plan:
        if row['split'] not in SPLITS:
            raise ValueError('unknown split label: ' + str(row['split']))
        rows = groups[row['group']]
        if {world['event_id'] for world in rows} != {event['event_id'] for event in row['events']}:
            raise ValueError('split plan events do not match the group: ' + row['group'])
        if {bool(world['mirror']) for world in rows} != {event['mirror'] for event in row['events']}:
            raise ValueError('split plan mirror flags do not match the group: ' + row['group'])
        for world in rows:
            if world['event_id'] in event_split:
                raise ValueError('an event is assigned to two splits: ' + world['event_id'])
            event_split[world['event_id']] = row['split']
    if len(event_split) != 28:
        raise ValueError('the split must cover exactly the twenty-eight events')
    for name, rows in groups.items():
        if len({event_split[world['event_id']] for world in rows}) != 1:
            raise ValueError('a mirror pair crosses a split: ' + name)
    family_counts = {}
    for family in FAMILIES:
        splits = [row['split'] for row in plan if row['family'] == family]
        if set(splits) != set(SPLITS):
            raise ValueError('every family must have both splits: ' + family)
        family_counts[family] = dict(Counter(splits))
    if [row['split'] for row in plan if row['family'] == 'F2'] != ['train', 'validation']:
        raise ValueError('F2 has two groups: one train and one validation')
    counts = dict(groups=dict(Counter(row['split'] for row in plan)),
                  events=dict(Counter(event_split.values())))
    if counts['groups'] != {'train': 7, 'validation': 7}:
        raise ValueError('the fourteen groups must alternate into seven train and seven validation')
    if counts['events'] != {'train': 14, 'validation': 14}:
        raise ValueError('the twenty-eight events must split fourteen and fourteen')
    for family, value in family_counts.items():
        if value.get('train', 0) + value.get('validation', 0) != EXPECTED_GROUPS[family]:
            raise ValueError('family group count changed: ' + family)
    return dict(groups=14, events=28, groups_per_split=counts['groups'],
                events_per_split=counts['events'], family_group_counts=family_counts,
                no_group_crosses_split=True, mirror_and_base_in_one_group_and_split=True,
                every_family_has_both_splits=True,
                f2_shape='two groups: index 0 train, index 1 validation', test_split=False)


# ---------------------------------------------------------------- aggregation
def aggregate(rows):
    """Pure per-teacher, per-family aggregation over already-recorded world rows."""
    switches = [row['switches'] for row in rows]
    ordered = sorted(switches)
    median = None
    if ordered:
        middle = len(ordered) // 2
        median = float(ordered[middle]) if len(ordered) % 2 else \
            (ordered[middle - 1] + ordered[middle]) / 2.
    return dict(
        instances=len(rows),
        feasible=sum(bool(row['feasible']) for row in rows),
        survived=sum(bool(row['survived']) for row in rows),
        goal_satisfied=sum(bool(row['goal_satisfied']) for row in rows),
        first_hit_present=sum(row['first_hit'] is not None for row in rows),
        single_action=sum(bool(row['single_action']) for row in rows),
        constant_success=sum(bool(row['degenerate']) for row in rows),
        degenerate=sum(bool(row['degenerate']) for row in rows),
        capability_success=sum(bool(row['capability_success']) for row in rows),
        switches_total=sum(switches), switches_mean=round(sum(switches) / len(rows), 3) if rows else None,
        switches_median=median, switches_max=max(switches) if switches else None,
        switch_events=sum(count > 0 for count in switches),
        failures=[row['event_id'] for row in rows if not row['feasible']],
        degenerate_events=[row['event_id'] for row in rows if row['degenerate']])


def family_verdict(teacher, teacher_v2, reference_witnesses, reference_replay):
    both = not (teacher['capability_success'] or teacher_v2['capability_success'])
    return dict(teacher='capable' if teacher['capability_success'] else 'not_capable',
                teacher_v2='capable' if teacher_v2['capability_success'] else 'not_capable',
                both_failed=both,
                teacher_feasible=teacher['feasible'], teacher_v2_feasible=teacher_v2['feasible'],
                teacher_capability_success=teacher['capability_success'],
                teacher_v2_capability_success=teacher_v2['capability_success'],
                reference_witnesses=reference_witnesses,
                reference_witness_replay=reference_replay,
                failure_preserved=both)


def summarise(rows, plan):
    teachers, by_split = {}, {}
    for name in TEACHERS:
        teachers[name], by_split[name] = {}, {}
        for family in FAMILIES + ('all',):
            group = [dict(row['teachers'][name], event_id=row['event_id'], family=row['family'])
                     for row in rows if family == 'all' or row['family'] == family]
            teachers[name][family] = aggregate(group)
            by_split[name][family] = {}
            for split in SPLITS:
                events = {event['event_id'] for entry in plan if entry['split'] == split
                          for event in entry['events']}
                by_split[name][family][split] = aggregate(
                    [entry for entry in group if entry['event_id'] in events])
    verdicts = {}
    for family in FAMILIES + ('all',):
        subset = [row for row in rows if family == 'all' or row['family'] == family]
        verdicts[family] = family_verdict(
            teachers['teacher'][family], teachers['teacher_v2'][family],
            sum(row['reference']['r18_status'] == 'feasible' for row in subset),
            sum(row['reference']['matches'] for row in subset))
    return dict(
        schema=SUMMARY_SCHEMA,
        counts=dict(events=len(rows), groups=len({row['group'] for row in rows}),
                    families={family: dict(events=sum(row['family'] == family for row in rows),
                                           groups=len({row['group'] for row in rows
                                                       if row['family'] == family}))
                              for family in FAMILIES}),
        teachers=teachers, family_verdicts=verdicts, by_split=by_split,
        constant_successes={name: [row['event_id'] for row in rows
                                   if row['teachers'][name]['degenerate']] for name in TEACHERS},
        degenerate_success_total=sum(row['teachers'][name]['degenerate']
                                     for row in rows for name in TEACHERS),
        both_failed_families=[family for family in FAMILIES if verdicts[family]['both_failed']],
        failure_preserved={name: teachers[name]['all']['failures'] for name in TEACHERS},
        baseline_reproduction=dict(worlds=len(rows), rows=len(rows) * (len(BASELINES) + 1),
                                   matches=all(row['baseline_reproduction']['matches'] for row in rows),
                                   mismatches=[dict(event_id=row['event_id'],
                                                    mismatches=row['baseline_reproduction']['mismatches'])
                                               for row in rows
                                               if not row['baseline_reproduction']['matches']],
                                   note='the six round18 baselines re-run through round18 policies/'
                                        'causal_loop/outcome must reproduce the frozen manifest exactly'),
        reference_replay=dict(worlds=len(rows),
                              feasible=sum(row['reference']['matches'] for row in rows),
                              status=dict(Counter(row['reference']['replay_status'] for row in rows)),
                              note='read-only replay of every round18 witness with the independent verify()'),
        split=dict(path='run/split.json',
                   groups=dict(Counter(entry['split'] for entry in plan)),
                   events=dict(Counter(entry['split'] for entry in plan for _ in entry['events']))),
        flags=dict(synthetic_development_worlds=True, test_split=False, training_steps=0,
                   trained_policy=False, checkpoint=None, teacher_labels=0, policy_evaluations=0,
                   formal_gate=False, deployable=False, game=False))


# ---------------------------------------------------------------- pre-register
def criterion():
    return dict(
        worlds=dict(source=MANIFEST,
                    load='read-only: the twenty-eight round18 worlds are loaded from the frozen manifest and '
                         'never regenerated; no parameter is changed, no world is replaced, no budget is added',
                    expected=dict(events=28, groups=14,
                                  per_family={family: dict(events=EXPECTED_EVENTS[family],
                                                           groups=EXPECTED_GROUPS[family])
                                              for family in FAMILIES})),
        loop='identical to round18: tactical_teacher_validation.advance, two logical frames per decision, '
             'horizon 160 = eighty decisions, sticky collision, no pixels, no rendering, no game',
        teachers=dict(teacher='touhou_rl.tactical_teacher.TacticalTeacher, unmodified',
                      teacher_v2='touhou_rl.tactical_teacher_v2.TacticalTeacherV2, unmodified',
                      driver='the driver owns the physics: every decision is executed with the verifier own '
                             'advance(); only decide() is called on the teachers. Their rollout() helpers are '
                             'not used because tactical_teacher.rollout replays through '
                             'tactical_scenarios.advance, whose coverage/gap/window/threshold are hard-coded '
                             'defaults and therefore are not this world'),
        metrics=dict(survived='full horizon 160 and no sticky collision',
                     goal_satisfied='F3: canonical x <= goal at some sampled state; F1/F2: true by the '
                                    'round18 convention, they have no crossing goal',
                     feasible='survived and goal_satisfied, identical to round18 outcome()',
                     first_hit='first logical frame with a sticky collision, null if never',
                     verify='tactical_teacher_validation.verify(world, actions) on the complete eighty-decision '
                            'sequence, non-early: feasible / rejected_witness / unknown',
                     counts='reused, not redefined: the six round18 baselines run through round18 '
                            'policies()/causal_loop()/outcome() and must reproduce the stored manifest'),
        degenerate=dict(definition='an episode whose eighty decisions contain exactly one distinct action is '
                                   'constant; if it is also feasible the success is marked degenerate, reported '
                                   'separately and never counted as teacher capability',
                        reason='a constant teacher episode is by construction the same trajectory as the '
                               'corresponding round18 constant baseline, which already fails on these worlds; the '
                               'flag keeps that explicit instead of silently crediting it',
                        capability_success='feasible and not single_action',
                        cross_check='every constant episode is compared with the round18 extra_constants probe and '
                                    'must agree, otherwise the run aborts'),
        action_changes=dict(definition='switches = number of consecutive decision pairs with a different action '
                                       'over the eighty decisions; reported per world and summed/averaged per '
                                       'teacher per family',
                            reported=['switches_total', 'switches_mean', 'switches_median', 'switches_max',
                                      'switch_events']),
        family_verdict=dict(capable='at least one capability success in the family',
                            not_capable='zero capability successes',
                            both_failed='both teachers not_capable: the family is reported as a failure and kept '
                                        'as a failure, next to the round18 reference witness count',
                            no_relaxation='no threshold is loosened, no world is swapped, no teacher is tuned, no '
                                          'action set is widened, no extra attempt is made, no seed is drawn'),
        reference='read-only: every round18 reference witness is replayed with verify() to show the family is '
                  'solvable at all; this is a contrast, never a success criterion for a teacher',
        split=dict(rule='one group = base + mirror, atomic; within each family the groups are ordered by group '
                        'name (candidate index) and alternate train/validation starting with train at index 0',
                   f2='two groups only: index 0 train, index 1 validation',
                   assertions=['no group crosses a split', 'base and mirror are always in one group and therefore '
                               'in one split', 'every family has both splits',
                               'fourteen groups and twenty-eight events, no event in two groups',
                               'seven train and seven validation groups, fourteen and fourteen events']),
        flags=dict(synthetic_development_worlds=True, test_split=False, training_steps=0, trained_policy=False,
                   checkpoint=None, teacher_labels=0, policy_evaluations=0, formal_gate=False,
                   deployable=False, game=False),
        execution=dict(order=['tests', 'preregister', 'tests', 'run', 'tests'],
                       overwrite='directories are created with exist_ok=False and every JSON is opened in x mode',
                       reruns='exactly one evaluation run; no rerun, no retune, no repair pass'))


def identities():
    paths = ['tools/rl/v373_valid_teacher_eval.py', 'tests/test_v373_valid_teacher_eval.py',
             'tools/rl/v373_world_filter.py',
             'touhou_rl/tactical_teacher.py', 'touhou_rl/tactical_teacher_v2.py',
             'touhou_rl/tactical_teacher_validation.py', 'touhou_rl/tactical_scenarios.py',
             'touhou_rl/tactical_synthetic_closed_loop.py', 'touhou_rl/tactical_data.py',
             MANIFEST, R18_SHA, R18_PREREG, R18_REPORT]
    missing = [path for path in paths if not repo_path(path).is_file()]
    if missing:
        raise FileNotFoundError('pre-registration identity missing: ' + ','.join(missing))
    return {path: file_sha(repo_path(path)) for path in paths}


def preregister(out=OUT):
    dest = new_dir(repo_path(out))
    data = load_worlds()
    value = dict(schema=PREREG_SCHEMA, identities=identities(),
                 purpose='teacher capability on the frozen round18 development worlds only: no training, no trained '
                         'policy, no checkpoint, no teacher labels, no formal gate, no game execution',
                 source=dict(manifest=MANIFEST, manifest_sha256=data['manifest_sha256'],
                             r18_preregistration=R18_PREREG,
                             r18_preregistration_sha256=data['r18_prereg_sha256'],
                             r18_report=R18_REPORT, r18_schema=R18_SCHEMA, r18_totals=data['totals']),
                 expected=dict(events=28, groups=14,
                               per_family={family: dict(events=EXPECTED_EVENTS[family],
                                                        groups=EXPECTED_GROUPS[family])
                                           for family in FAMILIES}),
                 criterion=criterion(),
                 split=dict(rule='alternate the family-internal group index, train at index 0',
                            expected=dict(groups={'train': 7, 'validation': 7},
                                          events={'train': 14, 'validation': 14},
                                          f2={'train': 1, 'validation': 1})),
                 flags=dict(synthetic_development_worlds=True, test_split=False, training_steps=0,
                            trained_policy=False, checkpoint=None, teacher_labels=0, formal_gate=False,
                            deployable=False, game=False))
    write_json(dest / 'preregistration.json', value)
    return value


def verify_prereg(out=OUT):
    path = repo_path(out) / 'preregistration.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema') != PREREG_SCHEMA:
        raise ValueError('pre-registration schema')
    for name, sha in value['identities'].items():
        if file_sha(repo_path(name)) != sha:
            raise ValueError('frozen identity changed: ' + name)
    if value['criterion'] != criterion():
        raise ValueError('criterion drift after pre-registration')
    if value['source']['manifest_sha256'] != load_worlds()['manifest_sha256']:
        raise ValueError('the round18 manifest changed after pre-registration')
    return value


# ---------------------------------------------------------------- run
def run(out=OUT):
    dest_root = repo_path(out)
    prereg = verify_prereg(out)
    data = load_worlds()
    worlds = data['worlds']
    dest = new_dir(dest_root / 'run')
    plan = split_plan(worlds)
    assertions = assert_split(plan, worlds)
    write_json(dest / 'split.json', dict(
        schema=SPLIT_SCHEMA,
        source=dict(manifest=MANIFEST, manifest_sha256=data['manifest_sha256'],
                    events=len(worlds), groups=len({world['group'] for world in worlds})),
        rule='one group = base + mirror, atomic; within each family the groups are ordered by group name '
             '(candidate index) and alternate train/validation starting with train at index 0; the two F2 '
             'groups therefore give one train and one validation group',
        signature_fields=list(r18.SIGNATURE_FIELDS),
        counts=dict(groups=assertions['groups_per_split'], events=assertions['events_per_split'],
                    by_family=assertions['family_group_counts']),
        assertions=assertions, groups=plan))
    for family in FAMILIES:
        new_dir(dest / 'worlds' / family)
    rows = []
    for world in worlds:
        row = evaluate_world(world)
        if not row['baseline_reproduction']['matches']:
            raise AssertionError('the six round18 baselines do not reproduce the frozen manifest: '
                                 + row['event_id'])
        if not row['reference']['matches']:
            raise AssertionError('the round18 reference witness does not replay: ' + row['event_id'])
        write_json(dest / 'worlds' / row['family'] / (row['event_id'] + '.json'), row)
        rows.append(row)
    summary = summarise(rows, plan)
    summary['prereg_sha256'] = file_sha(dest_root / 'preregistration.json')
    summary['source'] = dict(manifest=MANIFEST, manifest_sha256=data['manifest_sha256'],
                             r18_preregistration_sha256=data['r18_prereg_sha256'])
    summary['teacher_modules'] = {name: dict(path=TEACHER_FILES[name],
                                             sha256=file_sha(repo_path(TEACHER_FILES[name])))
                                  for name in TEACHERS}
    summary['identities_unchanged'] = all(file_sha(repo_path(name)) == sha
                                          for name, sha in prereg['identities'].items())
    summary['split']['assertions'] = assertions
    write_json(dest / 'summary.json', summary)
    write_json(dest / 'sha256.json', {path.relative_to(dest).as_posix(): file_sha(path)
                                      for path in sorted(dest.rglob('*'))
                                      if path.is_file() and path.name != 'sha256.json'})
    return summary


def verify_run(out=OUT):
    """Read-only check of the frozen artifacts: every stored SHA and the split plan."""
    dest = repo_path(out) / 'run'
    stored = json.loads((dest / 'sha256.json').read_text(encoding='utf-8'))
    drift = [key for key, sha in stored.items() if file_sha(dest / key) != sha]
    if drift:
        raise ValueError('run artifact changed: ' + ','.join(drift))
    summary = json.loads((dest / 'summary.json').read_text(encoding='utf-8'))
    if not summary['baseline_reproduction']['matches']:
        raise ValueError('the round18 baselines are not reproduced')
    split = json.loads((dest / 'split.json').read_text(encoding='utf-8'))
    worlds = load_worlds()['worlds']
    rebuilt = split_plan(worlds)
    if split['groups'] != rebuilt:
        raise ValueError('the frozen split does not match a recomputation')
    assert_split(rebuilt, worlds)
    return dict(artifacts=len(stored), baselines=summary['baseline_reproduction']['rows'],
                reference_replay=summary['reference_replay']['feasible'],
                groups=len(rebuilt), events=summary['counts']['events'],
                degenerate_success_total=summary['degenerate_success_total'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preregister', 'run', 'verify'))
    parser.add_argument('--out', default=OUT)
    args = parser.parse_args()
    if args.command == 'preregister':
        value = preregister(args.out)
        print(json.dumps({'schema': value['schema'], 'events': value['expected']['events'],
                          'groups': value['expected']['groups'],
                          'manifest_sha256': value['source']['manifest_sha256'],
                          'identities': len(value['identities'])}, indent=2))
    elif args.command == 'run':
        print(json.dumps(run(args.out), indent=2))
    else:
        print(json.dumps(verify_run(args.out), indent=2))


if __name__ == '__main__':
    main()
