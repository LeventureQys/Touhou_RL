"""Round19 CPU contracts for the valid-teacher evaluation.

Run before pre-registration, again after it, and once more after the single run:
    python -m pytest tests/test_v373_valid_teacher_eval.py -q -p no:cacheprovider --basetemp .tmp/v373-validteacher-round19-tests-01

The suite is ordered so that nothing here can influence the criterion: the
criterion is a module constant, the pre-registration test writes into a temporary
directory, and the frozen-world checks are read-only.
"""
import ast
import json
from collections import Counter
from pathlib import Path

import pytest

from tools.rl import v373_valid_teacher_eval as te
from touhou_rl.tactical_teacher_validation import make_world
from touhou_rl.tactical_teacher_validation import verify as verify_actions

ROOT = Path(__file__).resolve().parents[1]
WORLD = 'wf18-F1-002-m0'


def world_rows():
    return te.load_worlds()['worlds']


def one_world(event_id=WORLD):
    return next(row for row in world_rows() if row['event_id'] == event_id)


def episode_row(event_id, feasible, switches, single=False, family='F1', first_hit=40):
    return dict(event_id=event_id, family=family, feasible=feasible,
                survived=feasible, goal_satisfied=True, first_hit=None if feasible else first_hit,
                single_action=single, degenerate=bool(feasible and single),
                capability_success=bool(feasible and not single), switches=switches)


# ---------------------------------------------------------------- input
def test_manifest_is_the_frozen_round18_set():
    """Twenty-eight worlds, fourteen groups, and the frozen SHA table still matches."""
    data = te.load_worlds()
    worlds = data['worlds']
    assert len(worlds) == 28 and len({row['group'] for row in worlds}) == 14
    for family in te.FAMILIES:
        rows = [row for row in worlds if row['family'] == family]
        assert len(rows) == te.EXPECTED_EVENTS[family]
        assert len({row['group'] for row in rows}) == te.EXPECTED_GROUPS[family]
    assert json.loads((ROOT / te.R18_SHA).read_text(encoding='utf-8'))['manifest.json'] == \
        data['manifest_sha256'] == te.file_sha(ROOT / te.MANIFEST)
    for row in worlds:
        assert row['verdict'] == 'valid' and 'all_six_baselines_failed' in row['reasons']
        assert row['scenario']['split'] == 'development'
        assert row['any_constant_feasible'] == []
        assert all(not value['feasible'] for value in row['baselines'].values())
        assert row['weak']['feasible'] is False
        assert row['reference']['status'] == 'feasible'
        assert row['reference']['complete_actions'] is True
        assert len(row['reference']['actions']) == te.DECISIONS
    assert te.HORIZON == 160 and te.DECISIONS == 80


def test_every_round18_witness_still_replays():
    """The read-only reference contrast: every family really is solvable."""
    for row in world_rows():
        s = make_world(row['scenario'])
        assert verify_actions(s, row['reference']['actions']) == 'feasible', row['event_id']


# ---------------------------------------------------------------- split
def test_split_alternates_by_family_index_without_leakage():
    worlds = world_rows()
    plan = te.split_plan(worlds)
    groups = te.group_rows(worlds)
    assert [entry['split'] for entry in plan if entry['family'] == 'F1'] == \
        ['train', 'validation', 'train', 'validation', 'train', 'validation']
    assert [entry['split'] for entry in plan if entry['family'] == 'F2'] == ['train', 'validation']
    assert [entry['split'] for entry in plan if entry['family'] == 'F3'] == \
        ['train', 'validation', 'train', 'validation', 'train', 'validation']
    assert Counter(entry['split'] for entry in plan) == {'train': 7, 'validation': 7}
    event_split = {}
    for family in te.FAMILIES:
        ordered = [entry for entry in plan if entry['family'] == family]
        assert [entry['index_in_family'] for entry in ordered] == list(range(len(ordered)))
        assert [entry['group'] for entry in ordered] == \
            sorted((entry['group'] for entry in ordered), key=te.family_index)
    for entry in plan:
        assert len(entry['events']) == 2
        assert {event['mirror'] for event in entry['events']} == {False, True}
        assert abs(entry['events'][0]['start_x'] + entry['events'][1]['start_x'] - 384.) < 1e-9
        assert dict(zip(te.r18.SIGNATURE_FIELDS, entry['signature'])) == entry['params']
        for event in entry['events']:
            assert event['event_id'] not in event_split
            event_split[event['event_id']] = entry['split']
        # the two mirrors of one group never land in two splits
        assert len({event_split[event['event_id']] for event in entry['events']}) == 1
    assert len(event_split) == 28 and \
        sorted(Counter(event_split.values()).items()) == [('train', 14), ('validation', 14)]
    for name, rows in groups.items():
        assert len({event_split[row['event_id']] for row in rows}) == 1, name
    summary = te.assert_split(plan, worlds)
    assert summary['groups'] == 14 and summary['events'] == 28
    assert summary['no_group_crosses_split'] and summary['mirror_and_base_in_one_group_and_split']
    assert summary['every_family_has_both_splits'] and summary['test_split'] is False


def test_split_rejects_tampering():
    """Moving one mirror into another split is refused, and so is a duplicated group."""
    worlds = world_rows()
    plan = te.split_plan(worlds)
    assert te.assert_split(plan, worlds)['groups_per_split']['train'] == 7
    leaked = [dict(entry) for entry in plan]
    first = leaked[0]
    leaked[0] = dict(first, events=[first['events'][0]], split='train')
    leaked.append(dict(first, events=[first['events'][1]], split='validation'))
    with pytest.raises(ValueError):
        te.assert_split(leaked, worlds)
    dropped = plan[:-1]
    with pytest.raises(ValueError, match='fourteen groups'):
        te.assert_split(dropped, worlds)
    renamed = [dict(entry) for entry in plan]
    renamed[0] = dict(renamed[0], split='holdout')
    with pytest.raises(ValueError, match='unknown split label'):
        te.assert_split(renamed, worlds)


# ---------------------------------------------------------------- judgement
def test_constant_success_is_marked_degenerate_and_never_capability():
    """A single-action episode that survives is degenerate, not teacher ability."""
    constant_win = te.classify_actions([0] * te.DECISIONS, True)
    assert constant_win['single_action'] is True and constant_win['switches'] == 0
    assert constant_win['distinct_actions'] == [0] and constant_win['constant_action'] == 0
    assert constant_win['degenerate'] is True
    assert constant_win['capability_success'] is False
    constant_loss = te.classify_actions([-2] * te.DECISIONS, False)
    assert constant_loss['single_action'] is True and constant_loss['degenerate'] is False
    assert constant_loss['capability_success'] is False
    two_phase = te.classify_actions([1] * 12 + [0] * (te.DECISIONS - 12), True)
    assert two_phase['single_action'] is False and two_phase['degenerate'] is False
    assert two_phase['capability_success'] is True and two_phase['switches'] == 1
    churn = te.classify_actions([0, 1, 0, 1], True)
    assert churn['switches'] == 3 and churn['distinct_actions'] == [0, 1]
    # a constant success still contributes to the raw count but not to capability
    counts = te.aggregate([episode_row('a', True, 0, single=True)])
    assert counts['feasible'] == 1 and counts['constant_success'] == 1
    assert counts['degenerate'] == 1 and counts['capability_success'] == 0
    assert counts['degenerate_events'] == ['a'] and counts['failures'] == []


def test_failure_is_kept_as_failure():
    """Zero capability successes stay a failure; a degenerate win does not rescue it."""
    rows = [episode_row('a', False, 0, single=True), episode_row('b', False, 3),
            episode_row('c', True, 0, single=True)]
    counts = te.aggregate(rows)
    assert counts['instances'] == 3 and counts['feasible'] == 1
    assert counts['capability_success'] == 0 and counts['constant_success'] == 1
    assert counts['failures'] == ['a', 'b'] and counts['first_hit_present'] == 2
    verdict = te.family_verdict(counts, counts, 12, 12)
    assert verdict['teacher'] == 'not_capable' and verdict['teacher_v2'] == 'not_capable'
    assert verdict['both_failed'] is True and verdict['failure_preserved'] is True
    assert verdict['reference_witnesses'] == 12 and verdict['reference_witness_replay'] == 12
    capable = te.aggregate([episode_row('d', True, 1)])
    assert te.family_verdict(capable, counts, 4, 4)['both_failed'] is False


def test_aggregate_counts_action_changes():
    rows = [episode_row('a', False, 0), episode_row('b', False, 2),
            episode_row('c', False, 4)]
    counts = te.aggregate(rows)
    assert counts['switches_total'] == 6 and counts['switches_mean'] == 2.0
    assert counts['switches_median'] == 2.0 and counts['switches_max'] == 4
    assert counts['switch_events'] == 2
    constant = te.aggregate([episode_row('d', False, 0)])
    assert constant['switches_median'] == 0.0 and constant['switch_events'] == 0


# ---------------------------------------------------------------- per world
def test_evaluate_world_records_decisions_actions_and_verdicts():
    row = te.evaluate_world(one_world())
    assert row['schema'] == te.WORLD_SCHEMA
    assert row['event_id'] == WORLD and row['family'] == 'F1' and row['split'] == 'development'
    assert set(row['teachers']) == set(te.TEACHERS)
    for name, teacher in row['teachers'].items():
        assert teacher['decisions_count'] == te.DECISIONS == len(teacher['actions'])
        assert len(teacher['frame_actions']) == te.HORIZON
        assert teacher['actions_sha256'] == te.actions_sha(teacher['actions'])
        assert teacher['verify'] in {'feasible', 'rejected_witness', 'unknown'}
        assert (teacher['verify'] == 'feasible') == teacher['feasible']
        assert teacher['feasible'] == (teacher['survived'] and
                                       (row['family'] != 'F3' or teacher['reached_goal']))
        assert teacher['degenerate'] == (teacher['feasible'] and teacher['single_action'])
        assert teacher['capability_success'] == (teacher['feasible'] and not teacher['single_action'])
        assert teacher['switches'] == sum(a != b for a, b in
                                          zip(teacher['actions'], teacher['actions'][1:]))
        for decision in teacher['decisions']:
            assert {'frame', 'action', 'option', 'trigger_reason', 'confidence', 'candidate',
                    'x_before', 'x_after', 'hit'} <= set(decision)
            assert decision['action'] in te.ACTIONS and decision['confidence']
            assert te.CANDIDATE[decision['action']] == decision['candidate']
        assert sum(teacher['option_histogram'].values()) == te.DECISIONS
        assert teacher['constant_cross_check']['consistent'] in (None, True)
    assert row['flags'] == dict(synthetic_development_world=True, trained_policy=False, checkpoint=None,
                                training_steps=0, formal_gate=False, game=False)
    assert row['reference']['matches'] is True
    assert row['reference']['complete'] is True


@pytest.mark.parametrize('event_id', ['wf18-F1-002-m1', 'wf18-F2-119-m0', 'wf18-F3-000-m0'])
def test_baselines_reproduce_the_frozen_manifest(event_id):
    world = one_world(event_id)
    row = te.evaluate_world(world)
    assert row['baseline_reproduction']['matches'] is True
    assert row['baseline_reproduction']['mismatches'] == []
    assert set(row['baselines']) == set(te.BASELINES) | {te.WEAK}
    for name in te.BASELINES:
        assert row['baselines'][name]['first_hit'] == world['baselines'][name]['first_hit']
        assert row['baselines'][name]['actions'] == world['baseline_actions'][name]
    assert row['baselines'][te.WEAK]['actions'] == world['weak']['actions']
    assert row['baselines'][te.WEAK]['equivalent_baseline'] == world['weak']['equivalent_baseline']


# ---------------------------------------------------------------- plumbing
def test_exclusive_json_and_directory(tmp_path):
    path = tmp_path / 'a.json'
    te.write_json(path, {'a': 1})
    assert json.loads(path.read_text(encoding='utf-8')) == {'a': 1}
    with pytest.raises(FileExistsError):
        te.write_json(path, {'a': 2})
    directory = te.new_dir(tmp_path / 'd')
    assert directory.is_dir()
    with pytest.raises(FileExistsError):
        te.new_dir(tmp_path / 'd')


def test_preregister_freezes_the_criterion(tmp_path):
    value = te.preregister(tmp_path / 'prereg')
    assert value['schema'] == te.PREREG_SCHEMA
    assert value['expected'] == {'events': 28, 'groups': 14,
                                 'per_family': {'F1': {'events': 12, 'groups': 6},
                                                'F2': {'events': 4, 'groups': 2},
                                                'F3': {'events': 12, 'groups': 6}}}
    assert value['criterion'] == te.criterion()
    for key in ('worlds', 'loop', 'teachers', 'metrics', 'degenerate', 'action_changes',
                'family_verdict', 'reference', 'split', 'flags', 'execution'):
        assert key in value['criterion'], key
    assert value['flags']['training_steps'] == 0
    assert value['flags']['trained_policy'] is False and value['flags']['formal_gate'] is False
    assert value['flags']['game'] is False and value['flags']['checkpoint'] is None
    assert value['split']['expected']['groups'] == {'train': 7, 'validation': 7}
    assert te.MANIFEST in value['identities'] and te.R18_SHA in value['identities']
    assert te.R18_REPORT in value['identities']
    assert len(value['identities']) >= 13
    assert te.verify_prereg(tmp_path / 'prereg')['schema'] == te.PREREG_SCHEMA
    with pytest.raises(FileExistsError):
        te.preregister(tmp_path / 'prereg')
    path = tmp_path / 'prereg' / 'preregistration.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    frozen['criterion']['degenerate']['capability_success'] = 'feasible'
    path.write_text(json.dumps(frozen), encoding='utf-8')
    with pytest.raises(ValueError, match='criterion drift'):
        te.verify_prereg(tmp_path / 'prereg')


def test_frozen_identities_reject_tampering(tmp_path, monkeypatch):
    te.preregister(tmp_path / 'prereg')
    path = tmp_path / 'prereg' / 'preregistration.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    frozen['identities'][te.R18_REPORT] = '0' * 64
    path.write_text(json.dumps(frozen), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen identity changed'):
        te.verify_prereg(tmp_path / 'prereg')


# ---------------------------------------------------------------- after the run
def run_dir():
    path = te.repo_path(te.OUT) / 'run'
    if not (path / 'summary.json').is_file():
        pytest.skip('the single run has not happened yet')
    return path


def test_post_run_artifacts_are_frozen():
    run_dir()
    checked = te.verify_run()
    assert checked['artifacts'] >= 30 and checked['baselines'] == 168
    assert checked['reference_replay'] == 28 and checked['groups'] == 14
    assert checked['events'] == 28
    split = json.loads((run_dir() / 'split.json').read_text(encoding='utf-8'))
    summary = json.loads((run_dir() / 'summary.json').read_text(encoding='utf-8'))
    assert split['schema'] == te.SPLIT_SCHEMA and len(split['groups']) == 14
    assert split['groups'] == te.split_plan(world_rows())
    assert summary['schema'] == te.SUMMARY_SCHEMA
    assert summary['counts']['events'] == 28 and summary['counts']['groups'] == 14
    assert summary['baseline_reproduction']['matches'] is True
    assert summary['identities_unchanged'] is True
    assert summary['flags']['trained_policy'] is False and summary['flags']['formal_gate'] is False
    assert summary['degenerate_success_total'] == sum(len(value) for value in
                                                      summary['constant_successes'].values())
    worlds = list((run_dir() / 'worlds').rglob('*.json'))
    assert len(worlds) == 28
    for family in te.FAMILIES:
        assert len(list((run_dir() / 'worlds' / family).glob('*.json'))) == te.EXPECTED_EVENTS[family]
    with pytest.raises(FileExistsError):
        te.new_dir(run_dir())


# ---------------------------------------------------------------- boundaries
def test_module_is_read_only_research():
    source = Path(te.__file__).read_text(encoding='utf-8')
    for forbidden in ('torch.optim', '.backward(', 'TacticalPolicy', 'load_state_dict', 'state_dict',
                      'run_event', 'def solve(', "split='test'", 'split="test"', 'train_mode'):
        assert forbidden not in source, forbidden
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                for alias in node.names}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names}
    assert {'TacticalTeacher', 'TacticalTeacherV2', 'advance', 'metrics', 'verify'} <= imported
    assert not {'TacticalPolicy', 'run_event', 'solve'} & imported
    uses = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert 'decide' in uses
    assert te.REPORT.endswith('P2-validteacher-round19.md')
    assert te.OUT.endswith('/validteacher-round19') and 'worldfilter-round18' in te.R18
