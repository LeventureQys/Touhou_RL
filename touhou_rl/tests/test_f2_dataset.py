"""Synthetic/structural tests ONLY; monkeypatched deep audit is not real acceptance.

No live capture, no invented human review and no training/test payload execution.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from touhou_rl import f2_dataset as d
from touhou_rl import f2_labels


def write_json(path, obj):
    path.write_text(json.dumps(obj), encoding='utf-8')
    return path


def plan_fixture(tmp_path):
    sources, events = [], []
    for i, split in enumerate(d.SPLITS):
        replay = tmp_path / f'{i}.rpy'
        replay.write_bytes(bytes([i]))
        sources.append(dict(id=f's{i}', replay=d._ref(tmp_path, replay)))
        events.append(dict(event_id=f'e{i}', source_id=f's{i}', stage=0, split=split,
                           template_group=f'g{i}', episode_id=f'ep{i}', derived_group=f'd{i}'))
    obj = dict(schema=d.PLAN_SCHEMA, frozen=True, preregistered_at='2026-01-01T00:00:00+00:00',
               config=deepcopy(d.DEFAULT_CONFIG), config_sha256=f2_labels.evidence_sha(d.DEFAULT_CONFIG),
               sources=sources, events=events)
    return write_json(tmp_path / 'plan.json', obj), obj


@pytest.mark.parametrize('bad', ['../secret', '/tmp/secret', 'C:/secret', 'C:secret', r'..\secret', r'\\server\secret'])
def test_malicious_paths(tmp_path, bad):
    with pytest.raises(d.F2DatasetError):
        d._inside(tmp_path, bad)


def test_symlink_escape(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + '-outside')
    outside.mkdir()
    link = tmp_path / 'link'
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('symlinks unavailable')
    with pytest.raises(d.F2DatasetError, match='escapes'):
        d._inside(tmp_path, 'link/file')


@pytest.mark.parametrize('key', ['template_group', 'episode_id', 'derived_group', 'source_id'])
def test_association_leakage_before_upstream_payload(tmp_path, monkeypatch, key):
    path, plan = plan_fixture(tmp_path)
    plan['events'][1][key] = plan['events'][0][key]
    write_json(path, plan)
    monkeypatch.setattr(d.torch, 'load', lambda *a, **kw: pytest.fail('must reject before reading training/test payload'))
    with pytest.raises(d.F2DatasetError, match='leakage'):
        d._load_plan(path)


def test_three_independent_replay_hashes(tmp_path):
    path, plan = plan_fixture(tmp_path)
    plan['sources'][1]['replay'] = plan['sources'][0]['replay']
    write_json(path, plan)
    with pytest.raises(d.F2DatasetError, match='independent replay hashes'):
        d._load_plan(path)


def test_prereg_config_and_all_splits(tmp_path):
    path, plan = plan_fixture(tmp_path)
    assert d._load_plan(path) == plan
    plan['config']['depth'] = 1
    write_json(path, plan)
    with pytest.raises(d.F2DatasetError, match='config'):
        d._load_plan(path)


@pytest.mark.parametrize('mask', [0, 2, 0x101, 0x201, 0x31, 0xC1])
def test_unknown_bomb_and_unmappable_are_not_hold(mask):
    assert d._candidate(mask) is None


def test_actual_mapping_matches_live():
    assert d._candidate(1) == 0
    assert d._candidate(0x55) == 17


def fake_payload():
    # Small structural stand-ins, NOT actual preprocessed game captures.
    return dict(pixels={'playfield': torch.zeros(48, 3, 1, 1, 3)},
                history={'previous_candidate': torch.full((48,), 18)},
                labels={'action_valid': torch.ones(48, dtype=torch.bool)},
                label_evidence=[{'fixture_only': True}],
                steps=[{'frame': i + 4} for i in range(48)],
                input_pixel_hashes=[{'frame': 4, 'sha256': 'a' * 64}])


def test_structural_build_never_certifies(tmp_path, monkeypatch):
    plan_path, plan = plan_fixture(tmp_path)
    monkeypatch.setattr(d, '_materialize', lambda *a, **kw: (plan, [(e, fake_payload()) for e in plan['events']]))
    manifest = d.build_dataset(plan_path, tmp_path / 'built', deep=False)
    report = d.audit_dataset(manifest)
    assert not report['passed'] and 'structural-only' in report['errors'][0]
    obj = d._json(manifest)
    assert obj['schema'] == d.SCHEMA and not obj['original_G4']
    assert set(obj['split_groups']) == set(d.SPLITS)
    with pytest.raises(d.F2DatasetError):
        d.legacy_training_view(manifest)


def test_auditor_recomputes_tensor_contents_not_passed_strings(tmp_path, monkeypatch):
    # Monkeypatch upstream materialization ONLY to exercise storage audit.
    # A passing assertion here is NOT a real exact/F2 data readiness claim.
    path, plan = plan_fixture(tmp_path)
    monkeypatch.setattr(d, '_materialize', lambda *a, **kw: (plan, [(e, fake_payload()) for e in plan['events']]))
    manifest = d.build_dataset(path, tmp_path / 'built')
    assert d.audit_dataset(manifest)['passed']
    ep = manifest.parent / 'e0/event.json'
    event = d._json(ep)
    labels = ep.parent / 'labels.pt'
    torch.save({'action_valid': torch.zeros(48, dtype=torch.bool)}, labels)
    event['labels'] = d._ref(ep.parent, labels)
    write_json(ep, event)
    obj = d._json(manifest)
    obj['passed'] = True
    obj['events'][0] = d._ref(manifest.parent, ep)
    write_json(manifest, obj)
    report = d.audit_dataset(manifest)
    assert not report['passed'] and 'labels content mismatch' in report['errors'][0]


def test_alignment_diagnostics_persist_and_recompute(tmp_path, monkeypatch):
    path, plan = plan_fixture(tmp_path)
    def materialize(*args, **kwargs):
        results = []
        for event in plan['events']:
            payload = fake_payload()
            payload['alignment_diagnostics'] = dict(ambiguous_stage_frames=[
                dict(stage=0, stage_frame=55, logic_frames=[100, 101])], excluded_samples=2,
                unique_stage_frames=48)
            results.append((event, payload))
        return plan, results
    monkeypatch.setattr(d, '_materialize', materialize)
    manifest = d.build_dataset(path, tmp_path / 'built')
    obj = d._json(manifest)
    assert obj['alignment_diagnostics']['s0']['excluded_samples'] == 2
    report = d.audit_dataset(manifest)
    assert report['passed'] and report['alignment_diagnostics'] == obj['alignment_diagnostics']
    obj['alignment_diagnostics']['s0']['excluded_samples'] = 0
    write_json(manifest, obj)
    report = d.audit_dataset(manifest)
    assert not report['passed'] and 'alignment diagnostics' in report['errors'][0]


def test_immutable_plan_sha_and_output(tmp_path, monkeypatch):
    path, plan = plan_fixture(tmp_path)
    monkeypatch.setattr(d, '_materialize', lambda *a, **kw: (plan, [(e, fake_payload()) for e in plan['events']]))
    manifest = d.build_dataset(path, tmp_path / 'built')
    with pytest.raises(d.F2DatasetError, match='already exists'):
        d.build_dataset(path, tmp_path / 'built')
    path.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('no test payload reads before plan rejection'))
    assert 'plan SHA' in d.audit_dataset(manifest)['errors'][0]


def test_input_label_alias_rejected(tmp_path, monkeypatch):
    path, plan = plan_fixture(tmp_path)
    monkeypatch.setattr(d, '_materialize', lambda *a, **kw: (plan, [(e, fake_payload()) for e in plan['events']]))
    manifest = d.build_dataset(path, tmp_path / 'built')
    ep = manifest.parent / 'e0/event.json'
    event = d._json(ep)
    event['labels'] = event['inputs']['pixels']
    write_json(ep, event)
    obj = d._json(manifest)
    obj['events'][0] = d._ref(manifest.parent, ep)
    write_json(manifest, obj)
    assert not d.audit_dataset(manifest)['passed']


def test_upstream_pixel_duplicates_global(tmp_path, monkeypatch):
    path, plan = plan_fixture(tmp_path)
    monkeypatch.setattr(d, '_load_source', lambda *a, **kw: {})
    monkeypatch.setattr(d, '_event_payload', lambda *a, **kw: fake_payload())
    with pytest.raises(d.F2DatasetError, match='pixel content duplicate'):
        d._materialize(path, deep=True)


def test_same_hold_teacher_recomputed_from_raw():
    proof = dict(logicFrame=100, ticket=7)
    upstream = dict(teacher=Path(__file__), rows={(0, 4): proof}, teachers={100: dict(
        schema='th08-exact-audit-state/1', ticket=7, bulletsStatus='available', readComplete=True,
        activeCount=1, bullets=[dict(slot=2, position=[10, 20])])})
    annotation = dict(emissions=[], tolerances=dict(aim_miss_px=1.5), teacher=dict(
        artifact_sha256=d.file_sha(__file__), tracks=[dict(
            id='slot:2', samples=[dict(frame=4, xy=[42, 36])])]))
    d._verify_teacher({}, annotation, upstream, 0)
    annotation['teacher']['tracks'][0]['samples'][0]['xy'][0] += 1
    with pytest.raises(d.F2DatasetError, match='differs'):
        d._verify_teacher({}, annotation, upstream, 0)


def test_aim_miss_is_recomputed_from_the_raw_pool():
    """A review cannot declare a hitbox miss that the raw pool contradicts."""
    proof = dict(logicFrame=100, ticket=7)
    raw = dict(schema='th08-exact-audit-state/1', ticket=7, bulletsStatus='available',
               readComplete=True, activeCount=1, state=dict(player_x=10, player_y=120),
               bullets=[dict(slot=2, position=[10, 20], velocity=[0, 4])])
    upstream = dict(teacher=Path(__file__), rows={(0, 4): proof, (0, 6): proof},
                    teachers={100: raw})
    emission = dict(frame=4, track_id='slot:2', aim_player_frame=4,
                    aim_player_px=[42, 136])  # the player's actual position: straight below
    annotation = dict(emissions=[emission], tolerances=dict(aim_miss_px=1.5, position_px=1.5),
                      teacher=dict(artifact_sha256=d.file_sha(__file__), tracks=[]))
    d._verify_teacher({}, annotation, upstream, 0)
    # Player unchanged, but the bullet now travels sideways: no longer aimed.
    raw['bullets'][0]['velocity'] = [4, 0]
    with pytest.raises(d.F2DatasetError, match='misses the player hitbox'):
        d._verify_teacher({}, annotation, upstream, 0)


@pytest.mark.parametrize('mask', [0, 2, 0x201, 0xC1])
def test_unknown_actions_have_zero_masks_and_never_hold(mask):
    from touhou_rl.tests.test_f2_labels import fixture
    row = f2_labels.build_label(*fixture(mask))
    assert row['label_status'] == 'unknown'
    assert row['action_valid'] is False and not any(row['action_positive'])
    tensors = f2_labels.label_to_tensors([row])
    assert not tensors['action_valid'].any() and not tensors['action_positive'].any()


def test_context_unknown_is_retained_beside_valid_demonstration():
    from touhou_rl.tests.test_f2_labels import fixture
    observation, annotation, replay = fixture()
    context = f2_labels.build_label(observation, None, replay)
    demonstration = f2_labels.build_label(observation, annotation, replay)
    tensors = f2_labels.label_to_tensors([context, demonstration])
    assert tensors['action_valid'].tolist() == [False, True]
    assert tensors['action_positive'].shape == (2, 18)
    assert not tensors['action_positive'][0].any()
    assert not tensors['target_valid'].any()


def test_paused_stage_frames_exclude_all_ambiguous_samples(tmp_path, monkeypatch):
    # Disk/source integration fixture only: upstream exact audit is deliberately
    # monkeypatched, so this is NOT a real exact-capture acceptance claim.
    capture = tmp_path / 'capture'
    capture.mkdir()
    replay = tmp_path / 'input.rpy'
    replay.write_bytes(b'synthetic-replay')
    teacher = capture / 'audit-state.jsonl'
    teacher.write_text('\n'.join(json.dumps(dict(logicFrame=f)) for f in range(10, 15)), encoding='utf-8')
    rows_path = capture / 'proofs.jsonl'
    rows_path.write_text('\n'.join(json.dumps(dict(logicFrame=logic, stateSummary=dict(stage=0, stageFrames=frame)))
                                   for logic, frame in [(10, 4), (11, 5), (12, 5), (13, 5), (14, 6)]), encoding='utf-8')
    config = {'fixture_only': True}
    config_sha = f2_labels.evidence_sha(config)
    proof = dict(proofRows=d._ref(capture, rows_path), auditState=d._ref(capture, teacher),
                 collectionProvenance=dict(replaySha256=d.file_sha(replay), configSha256=config_sha))
    manifest = write_json(capture / 'frame-proof.json', proof)
    identity = dict(source='replay', frame_proof='exact', replay_sha256=d.file_sha(replay),
                    exact_manifest_sha256=d.file_sha(manifest), collection_config=config, config_sha256=config_sha)
    collect = write_json(tmp_path / 'collect.json', {'capture_identity': identity})
    source = dict(id='s', exact_manifest=d._ref(tmp_path, manifest), replay=d._ref(tmp_path, replay),
                  collect_config=d._ref(tmp_path, collect), teacher_artifact=d._ref(tmp_path, teacher),
                  collection_config_sha256=config_sha)
    calls = []
    monkeypatch.setattr(d.exact_audit, 'audit_exact_dataset', lambda *a, **kw: calls.append(kw))
    upstream = d._load_source(tmp_path, source, deep=True)
    assert set(upstream['rows']) == {(0, 4), (0, 6)}
    assert upstream['alignment_diagnostics'] == dict(ambiguous_stage_frames=[
        dict(stage=0, stage_frame=5, logic_frames=[11, 12, 13])], excluded_samples=3, unique_stage_frames=2,
        stage_clock_lag=dict(minimum=6, maximum=8, values=[6, 8], samples_not_at_minimum=1))
    assert calls == [dict(deep=True, expect_exact=True, require_proof=True)]
    annotation = dict(emissions=[], tolerances=dict(aim_miss_px=1.5), teacher=dict(
        artifact_sha256=d.file_sha(teacher), tracks=[dict(
            id='slot:2', samples=[dict(frame=5, xy=[42, 36])])]))
    with pytest.raises(d.F2DatasetError, match='lacks exact proof'):
        d._verify_teacher({}, annotation, upstream, 0)
    # No raw frame is removed or rewritten; only the unambiguous index changes.
    assert len(rows_path.read_text(encoding='utf-8').splitlines()) == 5


