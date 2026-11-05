"""Engineering-only end-to-end fixtures, NOT human review or G3/real evidence.

Synthetic PE, scripted holds, hand-constructed pixel/review geometry and encoded
replay bytes exercise the actual audits. Only the expensive vision backbone is
substituted; materialization, exact verification, caching and optimizer run.
"""
from copy import deepcopy
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from touhou_rl import f2_dataset as d, f2_labels, frame_proof
from touhou_rl import tactical_feature_cache as cache, tactical_sequence_train as seq
from touhou_rl.tactical_train import validate_training_gate
from touhou_rl.tests import exact_fixtures as exact


def _json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


#: Deterministic 18-way candidate masks. Period 7 deliberately does not divide
#: the fixture's 96-frame stage/logic clock offset, so a stage-frame index would
#: produce a different label set than the logic-frame index (see
#: ``test_replay_clock_not_stage_clock``).
PATTERN = (0x01, 0x41, 0x81, 0x11, 0x21, 0x05, 0x45)


def _mask(logic):
    return PATTERN[logic % len(PATTERN)]


def _replay(path, identity, records=200):
    """Literal-only LZSS TH08 replay; actual codec validates these fixture bytes."""
    data = bytearray(0x134 + 0x24 + 2 * records + 1)
    data[:6] = b'T8RP\x06\x00'
    struct.pack_into('<I', data, 0x20, 0x134)
    struct.pack_into('<I', data, 0x44, len(data) - 1)
    data[0x72] = identity + 65
    struct.pack_into('<II', data, 0x124, 840704, 2724749753)
    data[0x12c:0x132] = b'0100d\0'
    data[0x134 + 0x24:-1] = b''.join(_mask(index).to_bytes(2, 'little') for index in range(records))
    bits = ''.join('1' + format(v, '08b') for v in data[0x68:]) + '0' * 14
    bits += '0' * (-len(bits) % 8)
    compressed = bytes(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
    header = data[:0x68]
    struct.pack_into('<I', header, 0x0c, 0x68 + len(compressed))
    struct.pack_into('<II', header, 0x18, len(compressed), len(data) - 0x68)
    raw = header + compressed
    struct.pack_into('<I', raw, 0x10, (0x3f000318 + sum(raw[0x15:])) & 0xffffffff)
    key = raw[0x15]
    for i in range(0x18, len(raw)):
        raw[i] = (raw[i] + key) & 255
        key = (key + 7) & 255
    path.write_bytes(raw)
    return path


def _capture(root, source_index, duplicate_pixels=False, shift_inputs=False):
    root.mkdir()
    replay = _replay(root / 'engineering.rpy', source_index)
    collection = {'engineering_fixture': True, 'source_index': source_index}
    config_sha = f2_labels.evidence_sha(collection)
    samples = []
    for f in range(52):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :, 2] = 20 if duplicate_pixels else 20 + source_index
        image[0, 0] = [f, 1, 2]
        image[398:403, 198:203] = [255, 255, 255]
        positions = np.array([[168., 84. + f * 10], [168., 64. + f * 10]])
        for x, y in positions + [32, 16]:
            if 1 <= y < 479:
                image[int(y)-1:int(y)+2, int(x)-1:int(x)+2] = [255, 0, 0]
        # Stage frames 0..51, logic frames 100..151: the two clocks differ by 100
        # exactly as a real capture does after a stage-clock stall. The consumed
        # input words are the replay's logicFrame-1 / logicFrame-2 records unless
        # the test deliberately mis-binds them.
        logic = 100 + f
        skew = 1 if shift_inputs else 0
        state = exact.make_state(logic, input_mask=1,
                                 gui_current=_mask(logic - 1 + skew), gui_previous=_mask(logic - 2 + skew))
        state.player_x, state.player_y = 168., 384.
        sample = exact.make_sample(ticket=f+1, logic_frame=logic, state=state, pixels=image)
        sample.bullets = SimpleNamespace(position=positions, velocity=np.array([[0., 10.], [0., 10.]]),
            state=np.ones(2, dtype=np.int32), disabled=np.zeros(2, dtype=np.int32),
            half_size=np.ones((2, 2)), active_mask=lambda: np.ones(2, dtype=bool))
        samples.append(sample)
    _, _, _, capture, _, _ = exact.run_exact(root, samples,
        collection_provenance={'replaySha256': d.file_sha(replay), 'configSha256': config_sha})
    manifest = capture / frame_proof.PROOF_MANIFEST_NAME
    collect = _json(root / 'collection.json', dict(source='replay', frame_proof='exact',
        replay_sha256=d.file_sha(replay), exact_manifest_sha256=d.file_sha(manifest),
        config_sha256=config_sha, collection_config=collection))
    return replay, manifest, collect, config_sha


def _review(manifest, sid):
    capture = manifest.parent
    proof_rows = capture / frame_proof.PROOF_ROWS_NAME
    rows = [json.loads(line) for line in proof_rows.read_text().splitlines()]
    images = {int(r['stateSummary']['stageFrames']): dict(frame=int(r['stateSummary']['stageFrames']), sha256=r['imageSha256'].lower()) for r in rows}
    tracks = [dict(id='slot:0', samples=[dict(frame=f, xy=[200., 100.+f*10], visible=True,
               image_sha256=images[f]['sha256']) for f in (0, 1)]),
              dict(id='slot:1', samples=[dict(frame=f, xy=[200., 100.+(f-2)*10], visible=True,
               image_sha256=images[f]['sha256']) for f in (2, 3)])]
    reviewed = []
    for frame in range(4, 52):
        observation = dict(frame=frame, observed_through=frame, units='image_px', image_size=[640, 480],
            player_px=[200., 400.], source=dict(capture=dict(id=sid, sha256=d.file_sha(manifest)),
            proof=dict(id=frame_proof.PROOF_ROWS_NAME, sha256=d.file_sha(proof_rows))),
            images=[images[f] for f in (0, 1, 2, 3, frame)], bullet_tracks=deepcopy(tracks))
        annotation = dict(observation_sha256=f2_labels.evidence_sha(observation), family='F2', event_type='aimed_stream',
            engineering_fixture=True, reviewer=dict(id='engineering-pixel-fixture-NOT-human', method='independent_pixel_annotation'),
            teacher=dict(producer_id='engineering-scripted-memory', artifact_sha256=d.file_sha(capture/'audit-state.jsonl'), tracks=deepcopy(tracks)),
            tolerances=dict(position_px=0., aim_cross_px_per_frame=0., aim_miss_px=0., aim_lookback_frames=0,
                            emission_recency_frames=64),
            pixel_tracks=deepcopy(tracks),
            emissions=[dict(frame=f, track_id=track, emitter_id='engineering-emitter', aim_player_frame=f,
                aim_image_sha256=images[f]['sha256'], origin_px=[200., 100.], aim_player_px=[200., 400.],
                velocity_px_per_frame=[0., 10.], miss_px=0., player_pixel_annotation=dict(frame=f,
                image_sha256=images[f]['sha256'], xy=[200., 400.])) for f, track in ((0, 'slot:0'), (2, 'slot:1'))])
        assert f2_labels.validate_event_annotation(annotation, observation) == []
        reviewed.append(dict(observation=observation, annotation=annotation))
    return dict(frozen=True, engineering_fixture=True, warning='NOT real independent human review', rows=reviewed)


def _plan(root, duplicate_pixels=False, shift_inputs=False):
    sources, events = [], []
    for i, split in enumerate(d.SPLITS):
        replay, manifest, collect, config_sha = _capture(root/f's{i}', i, duplicate_pixels, shift_inputs)
        review = _json(root/f'review-{i}.json', _review(manifest, f's{i}'))
        sources.append(dict(id=f's{i}', replay=d._ref(root, replay), exact_manifest=d._ref(root, manifest),
            collect_config=d._ref(root, collect), collection_config_sha256=config_sha,
            teacher_artifact=d._ref(root, manifest.parent/'audit-state.jsonl')))
        events.append(dict(event_id=f'e{i}', source_id=f's{i}', stage=0, split=split,
            template_group=f'g{i}', episode_id=f'ep{i}', derived_group=f'd{i}', review=d._ref(root, review)))
    return _json(root/'plan.json', dict(schema=d.PLAN_SCHEMA, frozen=True,
        engineering_fixture=True, preregistered_at='2026-01-01T00:00:00+00:00',
        config=deepcopy(d.DEFAULT_CONFIG), config_sha256=f2_labels.evidence_sha(d.DEFAULT_CONFIG), sources=sources, events=events))


def _label_at(stage_frame):
    """Expected label: the record the *logic* clock points at, not the stage one."""
    return d._candidate(_mask(100 + stage_frame))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(trunk_hidden=4)
        self.policy_head = nn.Linear(4, 18)

    def forward_core(self, regions):
        means = regions['playfield'].mean(dim=(1, 2, 3, 4))
        features = torch.stack([means, means.square(), means*0+1, means*0], dim=-1)
        return dict(features=features, policy_logits=self.policy_head(features))


def test_engineering_exact_review_to_real_cache_and_optimizer(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    manifest = d.build_dataset(plan, tmp_path/'built')
    assert d.audit_dataset(manifest)['passed']
    assert validate_training_gate(manifest)['objective'] == 'f2_behavior_cloning'
    backbone = TinyBackbone()
    frozen = {k: v.clone() for k, v in backbone.state_dict().items()}
    baseline = tmp_path/'engineering-baseline.bin'
    baseline.write_bytes(b'ENGINEERING ONLY tiny backbone, NOT deployable')
    monkeypatch.setattr(cache, 'load_backbone', lambda _: backbone)
    monkeypatch.setattr(seq, 'load_backbone', lambda _: backbone)
    result = cache.build(manifest, baseline, tmp_path/'cache', batch_size=16)
    assert result['rows'] == 96 and not result['test_cache_created']
    assert not result['original_G4'] and not result['real_image_source_independently_certified']
    _, data, _ = seq.prepare(manifest, tmp_path/'cache/index.json', baseline)
    assert set(data) == {'train', 'validation'}
    labels = data['train'][0]['labels']
    assert labels['action_valid'].all()
    assert labels['action_positive'].to(torch.int64).argmax(-1).tolist() == [_label_at(frame) for frame in range(4, 52)]
    calls = []
    original_step = torch.optim.Adam.step
    def step(optimizer, *args, **kwargs):
        before = [p.detach().clone() for group in optimizer.param_groups for p in group['params']]
        value = original_step(optimizer, *args, **kwargs)
        after = [p for group in optimizer.param_groups for p in group['params']]
        calls.append(any(not torch.equal(a, b) for a, b in zip(before, after)))
        return value
    monkeypatch.setattr(torch.optim.Adam, 'step', step)
    report = seq.train(manifest, tmp_path/'cache/index.json', baseline, tmp_path/'training',
                       seq.SequenceConfig(arm='B1', epochs=1, max_updates=1))
    assert report['updates'] == 1 and calls == [True]
    assert not report['deployable'] and not report['test_used']
    assert all(torch.equal(v, frozen[k]) for k, v in backbone.state_dict().items())


@pytest.mark.parametrize('tamper', ['future', 'teacher'])
def test_review_tampering_rejected_even_with_updated_review_hash(tmp_path, tamper):
    plan_path = _plan(tmp_path)
    plan = d._json(plan_path)
    review_path = tmp_path/plan['events'][0]['review']['path']
    review = d._json(review_path)
    item = review['rows'][0]
    if tamper == 'future':
        item['observation']['observed_through'] += 1
        item['annotation']['observation_sha256'] = f2_labels.evidence_sha(item['observation'])
    else:
        # Defeat the annotation's tolerance check too: the raw teacher binding
        # must still reject a changed memory point after all public refs rehash.
        item['annotation']['teacher']['tracks'][0]['samples'][0]['xy'][0] += 1
        item['annotation']['tolerances']['position_px'] = 2.
    _json(review_path, review)
    plan['events'][0]['review'] = d._ref(tmp_path, review_path)
    _json(plan_path, plan)
    with pytest.raises(d.F2DatasetError,
                       match='causal observation|differs from raw upstream|player hitbox|invalid independent F2 human review'):
        d.build_dataset(plan_path, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_cross_split_duplicate_pixels_rejected_from_actual_pngs(tmp_path):
    plan = _plan(tmp_path, duplicate_pixels=True)
    with pytest.raises(d.F2DatasetError, match='cross-split per-frame pixel content duplicate'):
        d.build_dataset(plan, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_replay_clock_is_the_logic_clock_not_the_stage_clock(tmp_path):
    """A stage-frame index would label these rows with different records.

    The fixture's stage clock lags the replay logic clock by 100 frames (as a
    real capture does after a dialogue), so this assertion fails if the builder
    ever goes back to indexing records by ``stageFrames``.
    """
    plan = _plan(tmp_path)
    manifest = d.build_dataset(plan, tmp_path/'built')
    event = d._json(manifest.parent/'e0'/'event.json')
    steps = event['steps']
    assert [row['frame'] for row in steps] == [100 + f for f in range(4, 52)]
    assert [row['stage_frame'] for row in steps] == list(range(4, 52))
    assert all(row['previous_effective_frame'] == row['frame'] - 1 for row in steps)
    stage_labels = [d._candidate(_mask(frame)) for frame in range(4, 52)]
    assert [d._candidate(_mask(row['frame'])) for row in steps] != stage_labels
    labels = torch.load(manifest.parent/'e0'/'labels.pt', weights_only=True)
    assert labels['action_positive'].to(torch.int64).argmax(-1).tolist() == [_label_at(frame) for frame in range(4, 52)]


def test_misbound_consumed_input_rejected(tmp_path):
    """The record index is proven by the input word the game published."""
    plan = _plan(tmp_path, shift_inputs=True)
    with pytest.raises(d.F2DatasetError, match='not the input this frame consumed'):
        d.build_dataset(plan, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()

