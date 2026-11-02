"""Contract tests for the round20 teacher-distillation runner; CPU only, no training."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from tools.rl import v373_seed_confirm as r16
from tools.rl import v373_valid_curriculum as vc
from touhou_rl import vision
from touhou_rl.tactical_policy import tactical_policy_loss
from touhou_rl.tactical_synthetic_closed_loop import HORIZONTAL, run_event
from touhou_rl.tactical_synthetic_pixels import ACTION_MAP, SIZES
from touhou_rl.tactical_teacher_validation import make_world


@pytest.fixture(scope='module')
def frozen():
    manifest = vc.load_worlds()
    split = vc.load_split(manifest)
    return dict(manifest=manifest, split=split,
                train=vc.world_events(manifest, split, vc.TRAIN_SPLIT),
                validation=vc.world_events(manifest, split, vc.VALIDATION_SPLIT))


def _uint8(name, rows):
    shape = (rows, 3, SIZES[name][0], SIZES[name][1], 3)
    return np.random.default_rng(37318 + rows).integers(0, 256, size=shape, dtype=np.uint8)


def test_action_domain_matches_r11():
    assert vc.SUPPORTED == (0, 3, 7, 9, 12, 16)
    assert HORIZONTAL == {0: 0, 3: 2, 7: -2, 9: 0, 12: 1, 16: -1}
    assert vc.ACTION_MAP == ACTION_MAP == {-2: 7, -1: 16, 0: 0, 1: 12, 2: 3}
    assert sorted(ACTION_MAP) == [-2, -1, 0, 1, 2]
    assert set(ACTION_MAP.values()) <= set(HORIZONTAL)


def test_depth_indices_are_causal():
    rows = vc.depth_indices()
    frames = vc.depth_frames()
    assert rows.shape == (vc.DECISIONS, 3)
    assert frames[0] == [0, 0, 0]
    assert frames[1] == [0, 0, 2]
    assert frames[2] == [0, 2, 4]
    assert frames[5] == [6, 8, 10]
    assert frames[-1] == [154, 156, 158]
    for index, row in enumerate(rows):
        assert [2 * int(value) for value in row] == frames[index]
        assert list(row) == sorted(row), 'the depth must stay chronological'
        assert int(row[-1]) == index


def test_labels_single_route_only():
    actions = [0, -2, 0, 1, 2, 0, 0, -1] * 10
    _, labels = vc.history_and_labels(actions, True)
    assert tuple(labels['action_valid'].shape) == (80,)
    assert bool(labels['action_valid'].all())
    positive = labels['action_positive']
    assert tuple(positive.shape) == (80, 18)
    assert bool((positive.sum(-1) == 1).all()), 'exactly one route per row'
    assert positive.long().argmax(-1).tolist() == [ACTION_MAP[a] for a in actions]
    _, failed = vc.history_and_labels(actions, False)
    assert not bool(failed['action_valid'].any())
    assert not bool(failed['action_positive'].any()), 'an unverified event must carry no positive'
    assert set(labels) == {'action_positive', 'action_valid'}


def test_history_is_executed_not_planned():
    actions = [0, -2, 2, 1, -1, 0, 0, 0] * 10
    history, _ = vc.history_and_labels(actions, True)
    assert history['previous_candidate'].tolist() == [18] + [ACTION_MAP[a] for a in actions[:-1]]
    assert history['previous_option'].tolist() == [3] * 80
    assert history['previous_hold'].tolist() == [0.] + [2.] * 79
    assert history['frame_delta'].tolist() == [0.] + [2.] * 79
    assert not bool(history['target_valid'].any())
    assert not bool(history['history_valid'][0]) and bool(history['history_valid'][1:].all())
    assert tuple(history['previous_target'].shape) == (80, 2)
    assert history['previous_candidate'].dtype == torch.long
    assert history['previous_hold'].dtype == torch.float32
    assert history['history_valid'].dtype == torch.bool


def test_hud_single_copy_expansion_is_exact():
    stored = _uint8('hud_score', 1)
    dense = _uint8('hud_score', 80)
    dense[:] = stored[0]
    expanded = torch.from_numpy(np.ascontiguousarray(stored)).expand(80, -1, -1, -1, -1).contiguous()
    assert torch.equal(expanded, torch.from_numpy(np.ascontiguousarray(dense)))
    assert expanded.shape[0] == 80


def test_uint8_storage_roundtrip_and_scale(tmp_path):
    stored = {'playfield': _uint8('playfield', vc.DECISIONS)}
    for name in vc.HUD_NAMES:
        stored[name] = _uint8(name, 1)
    path = tmp_path / 'pixels.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, **stored)
    decoded = vc.pixel_tensors(path)
    for name, array in stored.items():
        value = decoded[name]
        assert value.dtype == torch.float32
        assert tuple(value.shape) == (80, 3, SIZES[name][0], SIZES[name][1], 3)
        assert float(value.min()) >= 0. and float(value.max()) <= 1.
        reference = (torch.from_numpy(np.ascontiguousarray(array)).to(torch.float32) / 255.)
        reference = reference.expand(80, -1, -1, -1, -1).contiguous()
        assert torch.allclose(value.contiguous(), reference, rtol=0, atol=1e-6), \
            'uint8/255 must be the only transform'
        assert float(value.max()) == pytest.approx(float(array.max()) / 255., abs=1e-6)
    with pytest.raises(FileExistsError):
        with path.open('xb'):
            pass


def test_pixel_tensors_rejects_non_uint8(tmp_path):
    path = tmp_path / 'bad.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, playfield=_uint8('playfield', vc.DECISIONS).astype(np.float32))
    with pytest.raises(ValueError):
        vc.pixel_tensors(path)


def _fake_output(delta=0.):
    """A policy whose logits are the base plus a class-specific push on an unsupported class."""
    logits = torch.zeros(1, 4, 18)
    logits[..., 3] = 2.
    base = logits.clone()
    base[..., 5] = delta
    output = dict(policy_logits=logits, base_logits=base, valid_mask=torch.ones(1, 4, dtype=torch.bool))
    output['option_logits'] = torch.zeros(1, 4, 3)
    output['target'] = torch.zeros(1, 4, 2)
    output['termination_logits'] = torch.zeros(1, 4, 3)
    output['slack'] = torch.zeros(1, 4)
    labels = dict(action_positive=torch.zeros(1, 4, 18, dtype=torch.bool),
                  action_valid=torch.tensor([[True, True, False, True]]))
    labels['action_positive'][0, :, 3] = True
    return output, labels


def test_kl_anchor_zero_when_policy_equals_base():
    output, labels = _fake_output()
    assert torch.equal(output['policy_logits'], output['base_logits'])
    assert float(vc.kl_anchor(output, labels)) == 0.


def test_kl_anchor_matches_manual_and_respects_valid_mask():
    output, labels = _fake_output(delta=3.)
    assert vc.anchor_mask(output, labels).tolist() == [[True, True, False, True]]
    value = float(vc.kl_anchor(output, labels))
    logits = output['policy_logits'][vc.anchor_mask(output, labels)].log_softmax(-1)
    base = output['base_logits'][vc.anchor_mask(output, labels)].detach().log_softmax(-1)
    manual = float((base.exp() * (base - logits)).sum(-1).mean())
    assert abs(value - manual) < 1e-6 and value > 0
    with pytest.raises(ValueError):
        vc.kl_anchor(output, dict(labels, action_valid=torch.zeros_like(labels['action_valid'])))


def test_arm_loss_is_single_factor():
    output, labels = _fake_output(delta=3.)
    original = float(tactical_policy_loss(output, labels)['loss'])
    control = vc.arm_loss(vc.CONTROL, output, labels)
    anchor = vc.arm_loss(vc.ANCHOR, output, labels)
    assert float(control['loss']) == pytest.approx(original)
    assert control['anchor'] is None
    assert float(anchor['original']) == pytest.approx(original)
    assert float(anchor['loss']) == pytest.approx(original + vc.ANCHOR_WEIGHT * float(anchor['anchor']))
    assert float(anchor['anchor']) > 0
    assert vc.ANCHOR_WEIGHT == 1.
    with pytest.raises(ValueError):
        vc.arm_loss('C', output, labels)


def test_training_constants_are_frozen():
    assert vc.SEEDS == (7, 17, 27)
    assert vc.UPDATES == 200
    assert vc.LR == .001 and vc.CLIP == 1.
    assert vc.SEQUENCE == 80 and vc.BATCH == 1 and vc.CHUNK == 16
    assert vc.ARMS == ('A', 'B') and (vc.CONTROL, vc.ANCHOR) == ('A', 'B')
    assert vc.DECISIONS == 80 and vc.HORIZON == 160 and vc.CONDITION == 'horizontal_masked'
    assert vc.EXPECTED['checkpoints'] == 6 and vc.EXPECTED['closed_loop_files'] == 14
    assert vc.EXPECTED['baseline_rows'] == 168
    assert vc.all_names() == ['A-s7', 'A-s17', 'A-s27', 'B-s7', 'B-s17', 'B-s27']


def test_exclusive_writes(tmp_path):
    target = tmp_path / 'a.json'
    vc.write_json(target, dict(x=1))
    with pytest.raises(FileExistsError):
        vc.write_json(target, dict(x=2))
    directory = tmp_path / 'd'
    vc.new_dir(directory)
    with pytest.raises(FileExistsError):
        vc.new_dir(directory)
    weights = tmp_path / 'w.pt'
    vc.write_torch(weights, dict(t=torch.zeros(2)))
    with pytest.raises(FileExistsError):
        vc.write_torch(weights, dict(t=torch.ones(2)))


def test_split_is_frozen_and_covers_28_worlds(frozen):
    manifest, split = frozen['manifest'], frozen['split']
    assert manifest['sha256'] == split['spec']['source']['manifest_sha256']
    assert len(manifest['spec']['worlds']) == 28
    assert len(split['assignment']) == 28
    assert set(split['assignment']) == set(manifest['index'])
    assert {name for name in split['assignment'].values()} == {vc.TRAIN_SPLIT, vc.VALIDATION_SPLIT}
    assert manifest['spec']['test_split'] is False and manifest['spec']['development_only'] is True
    assert all(world['scenario']['split'] == 'development' for world in manifest['spec']['worlds'])
    assert all(world['verdict'] == 'valid' for world in manifest['spec']['worlds'])
    assert all(world['passing_baselines'] == [] for world in manifest['spec']['worlds'])


def test_split_shapes_and_group_atomicity(frozen):
    split = frozen['split']
    counts = vc.split_counts(split)
    assert counts['groups'] == {vc.TRAIN_SPLIT: 7, vc.VALIDATION_SPLIT: 7}
    assert counts['events'] == {vc.TRAIN_SPLIT: 14, vc.VALIDATION_SPLIT: 14}
    assert counts['by_family'] == {'F1': {'train': 3, 'validation': 3},
                                   'F2': {'train': 1, 'validation': 1},
                                   'F3': {'train': 3, 'validation': 3}}
    for group in split['groups']:
        assert len(group['events']) == 2
        assert {bool(event['mirror']) for event in group['events']} == {False, True}
        assert len({event['event_id'] for event in group['events']}) == 2
    train = {row['event_id'] for row in frozen['train']}
    validation = {row['event_id'] for row in frozen['validation']}
    assert not train & validation and len(train) + len(validation) == 28
    for group in split['groups']:
        members = {event['event_id'] for event in group['events']}
        assert members <= (train if group['split'] == vc.TRAIN_SPLIT else validation), 'group leakage'
    checks = vc.split_checks(frozen['manifest'], split, frozen['train'], frozen['validation'])
    assert checks['no_group_crosses_split'] is True and checks['duplicate_signatures'] == 0
    assert checks['every_family_has_both_splits'] is True
    assert {f: sum(row['family'] == f for row in frozen['train']) for f in vc.FAMILY_KEYS} == \
           {'F1': 6, 'F2': 2, 'F3': 6}


def test_world_digest_is_deterministic(frozen):
    first = r16.worlds_digest(frozen['train'])
    second = r16.worlds_digest([dict(row) for row in frozen['train']])
    assert first == second
    assert first != r16.worlds_digest(frozen['validation'])


def test_criterion_is_frozen_and_declares_the_holdout_rule():
    value = vc.criterion()
    assert vc.criterion_sha() == vc.criterion_sha()
    assert value['evaluation']['holdout_rule'].startswith('if ANY of the six R18 baselines')
    assert 'withhold' in value['evaluation']['holdout_rule']
    assert value['training']['updates'] == 200 and value['training']['anchor_weight'] == 1.
    assert value['training']['seeds'] == list(vc.SEEDS)
    assert value['data']['train']['events'] == 14 and value['data']['validation']['events'] == 14
    assert value['training']['selection'].startswith('final step only')
    tampered = json.loads(json.dumps(value))
    tampered['training']['anchor_weight'] = 2.
    assert vc.criterion_sha(tampered) != vc.criterion_sha()
    assert 'no retrain' in value['failure_policy']
    assert value['claims']['significance'].startswith('none')


def test_identity_paths_exist():
    paths = vc.identity_paths()
    missing = [str(path) for path in paths if not Path(path).is_file()]
    assert not missing, missing
    hashes = vc.identities()
    assert len(hashes) == len(paths)
    assert all(len(digest) == 64 for digest in hashes.values())
    assert str(vc.MANIFEST) in hashes and str(vc.SPLIT) in hashes and str(vc.BASELINE) in hashes
    assert set(vc.teacher_hashes()) == {'tactical_teacher_v2.py', 'tactical_teacher.py',
                                        'tactical_teacher_validation.py', 'tactical_scenarios.py'}


def test_r18_six_baselines_reproduce_the_manifest(frozen):
    event = frozen['validation'][0]
    world = frozen['manifest']['index'][event['event_id']]
    payload = vc.six_baselines([event])
    assert payload['rows'] == 6 and len(payload['policies']) == 6
    assert payload['filter_valid'] is True and payload['passing_events'] == []
    for name in payload['policies']:
        row = next(item for item in payload['rows_detail'] if item['policy'] == name)
        stored = world['weak'] if name == 'teacher_first_action' else world['baselines'][name]
        assert bool(row['feasible']) == bool(stored['feasible']), name
        assert bool(row['survived']) == bool(stored['survived']), name
        assert row['first_hit'] == stored['first_hit'], name
        assert row['decisions'] == stored['decisions'], name


def test_no_test_split_or_sealed_world(frozen):
    for row in frozen['train'] + frozen['validation']:
        assert row['scenario']['split'] != 'test'
        with pytest.raises(ValueError):
            make_world(dict(row['scenario'], split='test'))
        assert row['scenario']['horizon'] == vc.HORIZON
        assert 'depletion_pressure' in row['scenario']
    assert vc.make_world.__module__ == 'touhou_rl.tactical_teacher_validation'


def test_one_event_end_to_end_asset_contract(tmp_path, frozen):
    event = next(row for row in frozen['train'] if row['event_id'] == 'wf18-F1-002-m0')
    payload = vc.render_event(event)
    assert payload['verdict'] == 'feasible' and payload['verified'] is True
    assert len(payload['actions']) == 80
    assert payload['hud_constant'] == {'hud_score': True, 'hud_status': True}
    assert payload['labels']['action_valid'].all()
    assert int(payload['labels']['action_positive'].sum()) == 80
    assert sorted(payload['stored']) == sorted(SIZES)
    assert payload['stored']['playfield'].shape == (80, 3, 224, 192, 3)
    for name in vc.HUD_NAMES:
        assert payload['stored'][name].shape == (1, 3, SIZES[name][0], SIZES[name][1], 3)
    for array in payload['stored'].values():
        assert array.dtype == np.uint8
    directory = vc.new_dir(tmp_path / 'assets' / event['event_id'])
    item = vc.build_event(directory, event)
    for name in ('pixels.npz', 'labels.pt', 'history.pt', 'rollout.json'):
        assert (directory / name).is_file(), name
    assert item['refs']['pixels']['bytes'] < 8_000_000, 'uint8 + deflate must stay small'
    assert item['refs']['pixels']['bytes'] * 4 < item['refs']['pixels']['raw_bytes'], \
        'the stored asset must be far smaller than a float32 copy of the same frames'
    stored = json.loads((directory / 'rollout.json').read_text(encoding='utf-8'))
    assert stored['teacher'] == 'TacticalTeacherV2 (unmodified)'
    assert stored['verify_verdict'] == 'feasible'
    assert len(stored['decisions']) == 80
    assert stored['depth_frames'][0] == [0, 0, 0] and stored['depth_frames'][-1] == [154, 156, 158]
    tensors = vc.pixel_tensors(directory / 'pixels.npz')
    assert all(tuple(value.shape)[0] == 80 for value in tensors.values())
    assert all(value.dtype == torch.float32 for value in tensors.values())


class _FixedNetwork:
    """Fake callback: argmax on an unsupported candidate, supported ones ordered hold>left>right."""

    def __init__(self):
        self.calls = 0

    def __call__(self, arm, regions, previous, hidden, reset):
        assert arm == 'N1'
        logits = torch.full((18,), -5.)
        logits[5] = 9.
        for candidate, value in ((0, 3.), (7, 2.), (3, 1.), (9, .5), (12, .25), (16, .1)):
            logits[candidate] = value
        self.calls += 1
        assert reset is (self.calls == 1)
        return logits, None


def test_closed_loop_masks_to_the_six_supported_candidates(frozen):
    event = frozen['validation'][0]
    world = make_world(event['scenario'])
    network = _FixedNetwork()
    row = run_event(world, 'N1', vc.CONDITION, network, 0, None)
    assert row['condition'] == 'horizontal_masked' and row['arm'] == 'N1'
    assert len(row['steps']) == 80 and network.calls == 80
    for step in row['steps']:
        assert step['candidate'] in HORIZONTAL
        assert step['raw_unmasked_candidate'] == 5
        assert step['raw_unsupported'] is True
        assert step['actual_action'] == HORIZONTAL[step['candidate']]
        assert step['invalid_action'] is False and step['fallback'] is False
    assert [step['candidate'] for step in row['steps']] == [0] * 80
    compacted = vc.compact(row)
    assert 'trace' not in compacted and 'environment_metrics' not in compacted
    assert set(compacted['steps'][0]) == set(vc.STEP_FIELDS)
    behaviour = vc.behaviour([compacted])
    assert behaviour['single_candidate_events'] == [world.event_id]
    assert behaviour['action_shares']['hold'] == 1. and behaviour['action_shares']['left'] == 0.
    assert behaviour['switches']['median'] == 0.
    assert r16.families([compacted])['all']['events'] == 1
    assert r16.families([compacted])['all']['firsthit'] == {world.event_id: row['first_collision_frame']}
