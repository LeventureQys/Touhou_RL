"""Round18 CPU contracts for the world-validity filter.

Run before pre-registration and before the single generation run:
    python -m pytest tests/test_v373_world_filter.py -q -p no:cacheprovider --basetemp .tmp/v373-worldfilter-round18-tests-01
"""
import ast
import json
from pathlib import Path

import pytest
import torch

from tools.rl import v373_world_filter as wf
from tools.rl import v373_seed_confirm as r16
from touhou_rl.tactical_synthetic_closed_loop import HORIZONTAL, run_event
from touhou_rl.tactical_teacher_validation import make_world
from touhou_rl.tactical_teacher_validation import verify as verify_actions

ROOT = Path(__file__).resolve().parents[1]
CORRIDOR = dict(coverage=(8., 376.), gap=(200., 216.), wall_frame=64, internal_gap=True)


def spec(**kw):
    base = dict(event_id='probe', family='F1', template_group='probe', seed=1, split='development',
                start_x=192., coverage=(100., 284.), gap=(180., 204.), wall_frame=64, internal_gap=False,
                window=(34, 104), threshold=180., goal=160., origin=(192., 100.), burst=False,
                depletion_pressure=False)
    base.update(kw)
    return base


def test_constant_passing_world_is_invalid():
    """A world a single constant already solves must be rejected, not retained."""
    evidence = wf.evaluate_world(make_world(spec()))
    assert evidence['verdict'] == 'invalid'
    assert {'right', 'left'} <= set(evidence['passing_baselines'])
    assert evidence['reference']['status'] == 'not_evaluated'
    assert evidence['baselines']['right']['feasible'] and evidence['baselines']['left']['feasible']


def test_random_passing_world_is_invalid():
    evidence = wf.evaluate_world(make_world(spec(family='F2', origin=(192., 100.), fire_interval=14,
                                                bullet_speed=4.5, start_x=180.)))
    assert evidence['verdict'] == 'invalid'
    assert evidence['passing_baselines'] == ['random']
    assert evidence['baselines']['random']['feasible'] and evidence['baselines']['random']['first_hit'] is None


def test_missing_reference_witness_is_unknown_not_valid():
    """Full-width lethal band without an internal gap: nothing can survive the wall."""
    evidence = wf.evaluate_world(make_world(spec(coverage=(8., 376.), wall_frame=64, start_x=192.)))
    assert evidence['verdict'] == 'unknown'
    assert evidence['passing_baselines'] == []
    assert evidence['reference']['status'] == 'unknown'
    assert evidence['reference']['actions'] == []
    assert evidence['reasons'] == ['reference_no_complete_replay:unknown']


def test_valid_world_needs_all_six_baselines_failing_and_a_full_witness():
    world = make_world(spec(start_x=248., **CORRIDOR))
    evidence = wf.evaluate_world(world)
    assert evidence['verdict'] == 'valid'
    assert evidence['passing_baselines'] == []
    for name in wf.BASELINES + (wf.WEAK,):
        row = evidence['baselines'].get(name) or evidence['weak']
        assert row['first_hit'] is not None and not row['feasible']
    reference = evidence['reference']
    assert reference['status'] == 'feasible' and reference['witness_replay'] is True
    assert reference['complete_actions'] is True and len(reference['actions']) == world.horizon // 2
    assert verify_actions(world, reference['actions']) == 'feasible'
    assert all(reference['differs_from'].values())
    assert reference['changed_steps_vs_hold'] > 0
    # stricter diagnostic only: a constant action still cannot solve it
    assert evidence['any_constant_feasible'] == []


def test_weak_baseline_always_collapses_onto_a_named_baseline():
    for mirror in (False, True):
        world = make_world(spec(start_x=384. - 248. if mirror else 248., mirror=mirror, **CORRIDOR))
        evidence = wf.evaluate_world(world)
        equivalent = evidence['weak']['equivalent_baseline']
        assert equivalent in {'hold', 'left', 'right'}
        assert evidence['weak']['actions'] == evidence['baselines'][equivalent]['actions']


def test_mask_matches_the_horizontal_condition():
    assert wf.SUPPORTED == (0, 3, 7, 9, 12, 16) and wf.SUPPORTED == tuple(sorted(HORIZONTAL))
    assert HORIZONTAL == {0: 0, 3: 2, 7: -2, 9: 0, 12: 1, 16: -1}
    logits = torch.zeros(18)
    logits[17] = 5.                              # unsupported candidate is the raw argmax
    logits[3] = 1.
    assert int(logits.argmax()) == 17 and wf.masked_candidate(logits) == 3
    single = torch.full((18,), -10.)
    single[7] = 0.
    assert wf.masked_candidate(single) == 7
    with pytest.raises(ValueError, match='invalid baseline logits'):
        wf.masked_candidate(torch.zeros(17))
    with pytest.raises(ValueError, match='invalid baseline logits'):
        wf.masked_candidate(torch.full((18,), float('nan')))


def test_causal_loop_reproduces_r11_run_event():
    """The same horizontal_masked closed loop, verified against the real R11 runner."""
    world = make_world(spec(start_x=248., **CORRIDOR))
    for name, policy in [('hold', wf.constant_policy(0)), ('alternate', wf.alternate_policy())]:
        mine = wf.causal_loop(world, policy)
        sequence = [d['candidate'] for d in mine['decisions']]
        stub = Stub(sequence)
        theirs = run_event(world, 'baseline', 'horizontal_masked', stub, 0)
        assert [step['actual_action'] for step in theirs['steps']] == mine['actions']
        assert theirs['feasible'] == wf.outcome(world, mine)['feasible']
        assert theirs['survived'] == (not mine['state'].hit)
        assert theirs['first_collision_frame'] == mine['first_collision_frame']
        assert stub.calls == len(mine['decisions']) == world.horizon // 2
        assert name in ('hold', 'alternate')


class Stub:
    """Single-route baseline callback: no network, no checkpoint, no pixels used."""
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = 0

    def __call__(self, arm, regions, previous, hidden, reset):
        candidate = self.sequence[self.calls]
        self.calls += 1
        return wf.one_hot(candidate), None


def test_mirror_groups_share_group_split_and_signature():
    geometry, reason = wf.draw_geometry('F1', __import__('random').Random(1))
    rows = wf.group_specs('F1', 7, geometry)
    assert len(rows) == 2 and rows[0]['template_group'] == rows[1]['template_group'] == 'wf18-F1-007'
    assert {r['split'] for r in rows} == {wf.SPLIT} and wf.SPLIT != 'test'
    assert {r['mirror'] for r in rows} == {False, True}
    assert rows[0]['start_x'] + rows[1]['start_x'] == 384.
    assert wf.signature(rows[0]) == wf.signature(rows[1])
    assert rows[0]['event_id'].endswith('-m0') and rows[1]['event_id'].endswith('-m1')
    base = make_world(rows[0])
    mirrored = make_world(rows[1])
    assert base.canonical_x(base.start_x) == mirrored.canonical_x(mirrored.start_x)


def test_accepted_worlds_never_repeat_a_frozen_signature():
    for name, rows in wf.existing_signatures().items():
        for spec_row in rows:
            assert wf.signature(spec_row) == r16.signature(spec_row), name
    search = wf.search_family('F1', budget=12)
    specs = [row[key]['scenario'] for row in search['accepted'] for key in ('base', 'mirror')]
    check = wf.assert_disjoint(specs)
    assert check['duplicates'] == 0 and check['new_events'] == len(specs) and check['new_groups'] > 0
    assert check['r10_subset_of_r8'] is True
    # Injecting a frozen R11 validation world must trip the assertion.
    frozen = json.loads(wf.R11.read_text(encoding='utf-8'))['events'][0]['scenario']
    pair = [dict(frozen, template_group='dup', event_id='dup-m0', mirror=False),
            dict(frozen, template_group='dup', event_id='dup-m1', mirror=True,
                 start_x=384. - frozen['start_x'])]
    assert wf.signature(pair[0]) == wf.signature(pair[1])
    with pytest.raises(ValueError, match='duplicates a frozen parameter signature'):
        wf.assert_disjoint(pair)


def test_search_reports_shortfall_without_relaxing():
    search = wf.search_family('F1', budget=2, target=6)
    assert search['drawn'] <= 2 and len(search['accepted']) < 6
    for row in search['accepted']:
        for key in ('base', 'mirror'):
            assert row[key]['verdict'] == 'valid' and row[key]['passing_baselines'] == []


def test_random_and_alternate_streams_are_frozen_and_recorded():
    sequence = wf.random_candidates(wf.RANDOM_SEED, wf.HORIZON // 2)
    assert len(sequence) == 80 and set(sequence) <= set(wf.SUPPORTED)
    assert sequence == wf.random_candidates(wf.RANDOM_SEED, wf.HORIZON // 2)
    # R8 preregistration: broad oscillation switches every 16 decisions (32 logical frames).
    assert wf.ALTERNATE_BLOCK == 16
    policy = wf.alternate_policy()
    blocks = [int(policy(step, None).argmax()) for step in range(40)]
    assert blocks[:16] == [7] * 16 and blocks[16:32] == [3] * 16 and blocks[32:] == [7] * 8
    criterion = wf.criterion()
    assert criterion['baselines']['random']['sequence'] == sequence
    assert criterion['baselines']['random']['seed'] == wf.RANDOM_SEED
    assert criterion['baselines']['alternate']['block_decisions'] == 16
    assert criterion['baselines']['alternate']['block_logical_frames'] == 32


def test_exclusive_json_and_directory(tmp_path):
    path = tmp_path / 'a.json'
    wf.write_json(path, {'a': 1})
    assert json.loads(path.read_text(encoding='utf-8')) == {'a': 1}
    with pytest.raises(FileExistsError):
        wf.write_json(path, {'a': 2})
    directory = wf.new_dir(tmp_path / 'd')
    assert directory.is_dir()
    with pytest.raises(FileExistsError):
        wf.new_dir(tmp_path / 'd')


def test_preregister_freezes_criterion_and_rejects_tampering(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / 'prereg'
    value = wf.preregister(out)
    assert value['schema'] == wf.PREREG_SCHEMA and value['test_split'] is False
    assert value['training_steps'] == 0 and value['deployable'] is False and value['game'] is False
    assert set(value['criterion']) >= {'baselines', 'loop', 'world_valid', 'world_invalid',
                                       'world_unknown', 'reference', 'no_relaxation'}
    assert value['generation']['budget_per_family'] == wf.BUDGET
    assert value['generation']['target_events'] == 36
    assert value['disjointness']['r16_confirmation_signatures'] == 12
    assert value['disjointness']['new_events'] == 0 and value['disjointness']['duplicates'] == 0
    assert wf.verify_prereg(out)['schema'] == wf.PREREG_SCHEMA
    with pytest.raises(FileExistsError):
        wf.preregister(out)
    path = out / 'preregistration.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    frozen['generation']['budget_per_family'] = {'F1': 1, 'F2': 1, 'F3': 1}
    path.write_text(json.dumps(frozen), encoding='utf-8')
    with pytest.raises(ValueError, match='search procedure changed'):
        wf.verify_prereg(out)


def test_old_set_probe_reports_single_constant_passability():
    """The old R11 F1 worlds are solved by constant right/left, so the probe must say so."""
    rows = wf.probe_old_sets(limit=2)
    assert {r['family'] for r in rows} == {'F1'} and len(rows) == 2
    for row in rows:
        assert row['verdict'] == 'invalid' and row['reference'] == 'feasible'
        assert row['any_constant_feasible'] == ['3', '7']
        assert row['baselines']['right']['feasible'] and row['baselines']['left']['feasible']
    assert wf.summarise(rows)['F1']['constant_passable'] == 2
    assert wf.summarise(rows)['F1']['baseline_passes_any'] == 2


def test_module_is_read_only_research():
    tree = ast.parse(Path(wf.__file__).read_text(encoding='utf-8'))
    source = Path(wf.__file__).read_text(encoding='utf-8')
    for forbidden in ('torch.optim', '.backward(', 'TacticalPolicy', 'load_state_dict',
                      "split='test'", 'train_mode'):
        assert forbidden not in source, forbidden
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                for alias in node.names}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names}
    assert not {'run_event', 'TacticalPolicy', 'tactical_policy'} & imported
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert 'solve' in calls and 'evaluate_world' in calls
    assert not {'rollout', 'metrics', 'run_event'} & calls
    assert wf.SPLIT == 'development' and wf.HORIZON == 160 and wf.TARGET_GROUPS == 6
