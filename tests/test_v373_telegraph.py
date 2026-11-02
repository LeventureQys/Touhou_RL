"""Contract tests for the round21 F1 danger telegraph; CPU only, no training, no CUDA.

The three things these tests exist for:
  1. the new renderer is `render_state` plus one F1 preview and nothing else,
  2. the preview is geometry the real wall will occupy, not a hint or a target,
  3. the preview is not physics: the same actions hit at the same frames.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from tools.rl import v373_telegraph as tel
from tools.rl import v373_valid_curriculum as r20
from touhou_rl import vision
from touhou_rl.tactical_policy import tactical_policy_loss
from touhou_rl.tactical_scenarios import initial_state
from touhou_rl.tactical_synthetic_closed_loop import HORIZONTAL, run_event
from touhou_rl.tactical_synthetic_pixels import ACTION_MAP, SIZES, render_state
from touhou_rl.tactical_teacher_v2 import rollout as teacher_rollout
from touhou_rl.tactical_teacher_validation import advance, make_world


@pytest.fixture(scope='module')
def frozen():
    manifest = r20.load_worlds()
    split = r20.load_split(manifest)
    train = r20.world_events(manifest, split, r20.TRAIN_SPLIT)
    validation = r20.world_events(manifest, split, r20.VALIDATION_SPLIT)
    return dict(manifest=manifest, split=split, train=train, validation=validation,
                all=train + validation)


def _world(events, event_id):
    return make_world(next(row for row in events if row['event_id'] == event_id)['scenario'])


def _f1_worlds(events):
    return [make_world(row['scenario']) for row in events if row['family'] == 'F1']


def _state(world, frame, x=None):
    from touhou_rl.tactical_scenarios import WorldState
    return WorldState(frame, world.start_x if x is None else float(x))


class _PixelHashNetwork:
    """A deterministic network whose 18 logits are a hash of the actual input pixels.

    Nothing about this network is sensible as a policy; that is the point. Two runners
    that see even one different pixel produce different actions, so equality of the whole
    step list is a very tight statement about the rendered input.
    """

    def __init__(self):
        self.calls = []
        self.ids = []

    def __call__(self, arm, regions, previous, hidden, reset):
        assert arm in ('N1',)
        assert reset is (len(self.calls) == 0)
        for name, value in regions.items():
            assert value.dtype == torch.float32 and value.ndim == 5, name
            assert float(value.min()) >= 0. and float(value.max()) <= 1., name
        last = regions['playfield'][0, -1]
        digest = int.from_bytes(hashlib.sha256(last.numpy().tobytes()).digest()[:8], 'big')
        logits = torch.full((18,), -5.)
        for index, candidate in enumerate((0, 3, 7, 9, 12, 16)):
            logits[candidate] = float((digest >> index) & 1) * 2. + (digest >> (index + 8)) % 3 * .1
        self.calls.append(digest)
        return logits, None


# ---------------------------------------------------------------- the renderer
def test_renderer_constants_are_frozen():
    assert tel.TELEGRAPH_LEAD == 45
    assert tel.TELEGRAPH_FILL == (44, 58, 74)
    assert tel.TELEGRAPH_EDGE == (96, 148, 128)
    assert tel.LETHAL_FILL == (95, 28, 42)
    assert tel.EDGE_WIDTH == 4
    assert tel.PLAYFIELD_BOX == (32, 16, 384, 448)
    assert tel.PLAYFIELD_BACKGROUND == (10, 14, 24)
    assert tel.PREVIEW_CLIP == (0, 383)
    assert len({tel.TELEGRAPH_FILL, tel.TELEGRAPH_EDGE, tel.LETHAL_FILL,
                tel.PLAYFIELD_BACKGROUND}) == 4, 'preview, markers, lethal and background must all differ'
    assert tel.DECISION_FRAMES == tuple(range(0, 160, 2)) and len(tel.DECISION_FRAMES) == 80


def test_the_original_renderer_still_paints_its_own_colours(frozen):
    """`LETHAL_FILL` and the playfield rectangle are facts about `render_state`, not claims."""
    world = _world(frozen['all'], 'wf18-F1-010-m0')
    lethal = render_state(world, _state(world, world.wall_frame))
    assert np.any(np.all(lethal == np.array(tel.LETHAL_FILL, dtype=np.uint8), axis=-1))
    plain = render_state(world, _state(world, 0))
    left, top, width, height = tel.PLAYFIELD_BOX
    region = plain[top:top + height, left:left + width]
    assert np.all(region[0, 200] == np.array(tel.PLAYFIELD_BACKGROUND, dtype=np.uint8))


def test_f2_and_f3_are_byte_identical_to_the_original_renderer(frozen):
    for event_id in ('wf18-F2-119-m0', 'wf18-F3-002-m1'):
        world = _world(frozen['all'], event_id)
        actions, _ = teacher_rollout(world)
        states = tel.full_states(world, actions)
        assert len(states) == 161
        for state in states:
            original = render_state(world, state)
            tegraphed = tel.render_state_telegraph(world, state)
            assert original.dtype == tegraphed.dtype == np.uint8
            assert original.shape == tegraphed.shape == (480, 640, 3)
            assert np.array_equal(original, tegraphed), (event_id, state.frame)
        assert tel.telegraph_window(world) is None
        assert tel.preview_columns(world).sum() == 0


def test_f1_is_identical_outside_the_preview_window(frozen):
    world = _world(frozen['all'], 'wf18-F1-002-m0')
    window = tel.telegraph_window(world)
    assert window == (31, 76) and world.wall_frame == 76
    for frame in list(range(0, 31)) + list(range(76, 161)):
        state = _state(world, frame, 120.)
        assert np.array_equal(render_state(world, state), tel.render_state_telegraph(world, state)), frame


def test_preview_start_boundary_is_wall_frame_minus_45(frozen):
    world = _world(frozen['all'], 'wf18-F1-002-m0')
    before = _state(world, world.wall_frame - tel.TELEGRAPH_LEAD - 1, 120.)
    after = _state(world, world.wall_frame - tel.TELEGRAPH_LEAD, 120.)
    assert not tel.telegraph_active(world, before.frame)
    assert tel.telegraph_active(world, after.frame)
    assert np.array_equal(render_state(world, before), tel.render_state_telegraph(world, before))
    assert not np.array_equal(render_state(world, after), tel.render_state_telegraph(world, after))
    assert tel.telegraph_active(world, world.wall_frame - 1)
    assert not tel.telegraph_active(world, world.wall_frame)
    # a family that is not F1 has no window at all
    f3 = _world(frozen['all'], 'wf18-F3-002-m0')
    assert not any(tel.telegraph_active(f3, frame) for frame in range(161))


def test_preview_paints_exactly_the_future_lethal_region(frozen):
    """Shift the wall 12 frames earlier and let the untouched renderer answer the question."""
    for world in _f1_worlds(frozen['all']):
        state = _state(world, world.wall_frame - 1)
        plain = render_state(world, state)
        tegraphed = tel.render_state_telegraph(world, state)
        shifted = replace(world, wall_frame=world.wall_frame - 12)
        lethal = render_state(shifted, state)
        mask_lethal = np.any(lethal != plain, axis=-1)
        mask_preview = np.any(tegraphed != plain, axis=-1)
        assert mask_lethal.any(), world.event_id
        assert np.array_equal(mask_preview, mask_lethal), world.event_id
        assert tel.telegraph_spans(world) and tel.telegraph_gap(world)


def test_gap_is_an_empty_slot_bounded_by_markers(frozen):
    world = _world(frozen['all'], 'wf18-F1-002-m0')
    state = _state(world, world.wall_frame - 1, 200.)
    image = tel.render_state_telegraph(world, state)
    left, top, width, height = tel.PLAYFIELD_BOX
    region = image[top:top + height, left:left + width]
    gap = tel.telegraph_gap(world)
    spans = tel.telegraph_spans(world)
    assert gap == (88, 104) and spans == [(8, 88), (104, 376)]
    fill = np.array(tel.TELEGRAPH_FILL, dtype=np.uint8)
    edge = np.array(tel.TELEGRAPH_EDGE, dtype=np.uint8)
    background = np.array(tel.PLAYFIELD_BACKGROUND, dtype=np.uint8)
    # the slot between the two painted spans is untouched background
    slot = region[:, gap[0] + 1:gap[1], :]
    if slot.shape[1]:
        assert np.all(slot == background)
    # the painted spans are occupied by the preview colour wherever the player sprite is not
    sprite = range(int(state.x) - 6, int(state.x) + 7)
    for lo, hi in spans:
        for column in range(lo, hi + 1):
            if column in sprite:
                continue
            pixels = region[40:400, column, :]
            assert np.all(np.any(pixels == fill, axis=-1) | np.any(pixels == edge, axis=-1)), column
    # exactly two four-pixel marker bars, split evenly on both sides of the slot
    markers = np.all(region == edge, axis=-1)
    assert int(markers.sum()) == 2 * tel.EDGE_WIDTH * height
    columns = sorted({int(value) for value in np.nonzero(markers)[1]})
    assert columns == list(range(gap[0] - tel.EDGE_WIDTH, gap[0])) + \
        list(range(gap[1] + 1, gap[1] + 1 + tel.EDGE_WIDTH))
    # mirroring puts the same slot on the other side, same width, same span order as render_state
    mirrored = _world(frozen['all'], 'wf18-F1-002-m1')
    assert tel.telegraph_gap(mirrored) == (384 - 104, 384 - 88)
    assert tel.telegraph_spans(mirrored) == [(384 - 88, 384 - 8), (384 - 376, 384 - 104)]
    assert sorted(tel.telegraph_spans(mirrored)) == [(8, 280), (296, 376)]


def test_preview_survives_the_model_input_resize(frozen):
    """The playfield tensor the backbone actually sees must carry both band and slot."""
    for event_id in ('wf18-F1-002-m0', 'wf18-F1-010-m1'):
        world = _world(frozen['all'], event_id)
        state = _state(world, world.wall_frame - 1, 200.)
        tegraphed = vision.resize_area(vision.extract_regions(
            tel.render_state_telegraph(world, state))['playfield'], *SIZES['playfield'])
        plain = vision.resize_area(vision.extract_regions(render_state(world, state))['playfield'],
                                   *SIZES['playfield'])
        assert tegraphed.shape == (224, 192, 3) and tegraphed.dtype == np.uint8
        changed = np.any(tegraphed != plain, axis=-1)
        assert int(changed.sum()) > 5000, 'the preview must be unmissable in the model input'
        fill = np.array(tel.TELEGRAPH_FILL, dtype=np.uint8)
        edge = np.array(tel.TELEGRAPH_EDGE, dtype=np.uint8)
        background = np.array(tel.PLAYFIELD_BACKGROUND, dtype=np.uint8)
        assert int(np.all(tegraphed == fill, axis=-1).sum()) > 5000
        assert int(np.all(tegraphed == edge, axis=-1).sum()) > 20, 'the gap markers must survive'
        # one contiguous run of untouched background inside the band: the visible slot
        middle = tegraphed[40:150]
        empty = [column for column in range(8, 184) if np.all(middle[:, column] == background)]
        runs = []
        for column in empty:
            if not runs or column != runs[-1][1] + 1:
                runs.append([column, column])
            else:
                runs[-1][1] = column
        wide = [run for run in runs if run[1] - run[0] >= 3]
        assert len(wide) == 1, runs
        assert 3 <= wide[0][1] - wide[0][0] <= 14, wide


def test_lethal_style_and_preview_style_are_different_and_frozen(frozen):
    world = _world(frozen['all'], 'wf18-F1-002-m0')
    lethal_image = tel.render_state_telegraph(world, _state(world, world.wall_frame, 200.))
    preview_image = tel.render_state_telegraph(world, _state(world, world.wall_frame - 1, 200.))
    lethal = np.array(tel.LETHAL_FILL, dtype=np.uint8)
    fill = np.array(tel.TELEGRAPH_FILL, dtype=np.uint8)
    edge = np.array(tel.TELEGRAPH_EDGE, dtype=np.uint8)
    assert int(np.all(lethal_image == lethal, axis=-1).sum()) > 100000
    assert not np.any(np.all(lethal_image == fill, axis=-1))
    assert not np.any(np.all(lethal_image == edge, axis=-1))
    assert int(np.all(preview_image == fill, axis=-1).sum()) > 100000
    assert int(np.all(preview_image == edge, axis=-1).sum()) > 3000
    assert not np.any(np.all(preview_image == lethal, axis=-1))
    assert np.array_equal(lethal_image, render_state(world, _state(world, world.wall_frame, 200.)))


def test_preview_stops_where_the_lethal_window_starts_and_ends(frozen):
    world = _world(frozen['all'], 'wf18-F1-002-m0')
    samples = {world.wall_frame - 1: 'preview', world.wall_frame: 'lethal',
               world.wall_frame + 11: 'lethal', world.wall_frame + 12: 'clear'}
    fill = np.array(tel.TELEGRAPH_FILL, dtype=np.uint8)
    lethal = np.array(tel.LETHAL_FILL, dtype=np.uint8)
    for frame, expected in samples.items():
        state = _state(world, frame, 200.)
        image = tel.render_state_telegraph(world, state)
        original = render_state(world, state)
        has_fill = bool(np.any(np.all(image == fill, axis=-1)))
        has_lethal = bool(np.any(np.all(image == lethal, axis=-1)))
        if expected == 'preview':
            assert has_fill and not has_lethal, frame
        elif expected == 'lethal':
            assert has_lethal and not has_fill, frame
            assert np.array_equal(image, original), frame
        else:
            assert not has_lethal and not has_fill, frame
            assert np.array_equal(image, original), frame


# ---------------------------------------------------------------- physics
def _replay(world, actions, render=None):
    """Identical action sequence through the real advance, optionally rendering in between."""
    state = initial_state(world)
    trace = [(state.frame, state.x, state.hit)]
    if render:
        render(world, state)
    for action in actions:
        state = advance(world, state, action)
        if render:
            render(world, state)
        trace.append((state.frame, state.x, state.hit))
    return trace


def test_rendering_has_no_side_effects_on_advance(frozen):
    actions = [2] * 8 + [0] * 32 + [-1] * 4 + [0] * 36
    assert len(actions) == 80
    for event_id in ('wf18-F1-002-m0', 'wf18-F1-006-m1', 'wf18-F1-010-m0', 'wf18-F3-002-m0'):
        world = _world(frozen['all'], event_id)
        plain = _replay(world, actions)
        assert _replay(world, actions, render_state) == plain
        assert _replay(world, actions, tel.render_state_telegraph) == plain
        assert _replay(world, actions, tel.render_state_telegraph) == _replay(world, actions, render_state)
        assert [frame for frame, x, hit in plain if hit][:1] == \
               [frame for frame, x, hit in _replay(world, actions, tel.render_state_telegraph) if hit][:1]


def _by_frame(trace):
    return {frame: (x, hit) for frame, x, hit in trace}


def test_standing_in_the_preview_band_is_not_a_collision(frozen):
    """The whole point: the band is drawn 45 frames early and kills on wall_frame, not before."""
    for event_id in ('wf18-F1-002-m0', 'wf18-F1-006-m0', 'wf18-F1-009-m1'):
        world = _world(frozen['all'], event_id)
        hold = [0] * 80
        trace = _by_frame(_replay(world, hold))
        hits = [frame for frame, (x, hit) in sorted(trace.items()) if hit]
        assert hits and hits[0] == world.wall_frame, event_id
        gap = tel.canonical_gap(world)
        window = tel.telegraph_window(world)
        for frame in tel.DECISION_FRAMES:
            if not window[0] <= frame < world.wall_frame:
                continue
            x, hit = trace[frame]
            assert not hit, (event_id, frame)
            assert not (gap[0] <= world.canonical_x(x) <= gap[1]), (event_id, frame)
            state = _state(world, frame, x)
            image = tel.render_state_telegraph(world, state)
            assert tel.telegraph_active(world, frame)
            assert np.any(np.all(image == np.array(tel.TELEGRAPH_FILL, dtype=np.uint8), axis=-1)), \
                f'the band must be on screen while it is harmless: {event_id}/{frame}'
        probe = tel.physics_probe([{'event_id': event_id,
                                   'scenario': next(row['scenario'] for row in frozen['all']
                                                    if row['event_id'] == event_id)}])
        assert probe['probes'][0]['first_hit'] == world.wall_frame


def test_gap_trace_reconstructs_advance_semantics(frozen):
    world = _world(frozen['all'], 'wf18-F1-002-m0')
    actions, _ = teacher_rollout(world)
    trace = tel.gap_trace(world, actions)
    expected = _replay(world, actions)
    assert len(trace) == 161 and len(expected) == 81
    for row, (frame, x, hit) in zip(trace[::2], expected):
        assert row['frame'] == frame
        assert row['x'] == pytest.approx(x)
    # the odd logical frame is the one advance checks but the decision grid never samples
    mid = trace[1]
    assert mid['frame'] == 1 and mid['x'] == pytest.approx(actions[0] * 2. + world.start_x)
    assert all(row['frame'] == index for index, row in enumerate(trace))


# ---------------------------------------------------------------- the diagnostics
def test_gap_diagnostics_see_the_teacher_park_inside_the_gap(frozen):
    rows = []
    for event_id in ('wf18-F1-002-m0', 'wf18-F1-006-m1', 'wf18-F1-009-m0'):
        world = _world(frozen['all'], event_id)
        actions, _ = teacher_rollout(world)
        row = tel.gap_diagnostics(world, actions, hit_frame=None)
        rows.append(row)
        assert row['entered_gap'] is True, event_id
        assert row['in_gap_at_wall'] is True, event_id
        assert row['mechanism'] == 'in_the_gap_at_the_wall', event_id
        assert row['distance_to_gap_at_wall'] == 0.
        gap = tel.canonical_gap(world)
        assert gap[0] <= row['canonical_at_wall'] <= gap[1]
    assert all(row['complete'] for row in rows)


def test_gap_diagnostics_flag_a_route_that_never_enters_the_gap(frozen):
    world = _world(frozen['all'], 'wf18-F1-002-m0')
    hold = [0] * 80
    row = tel.gap_diagnostics(world, hold, hit_frame=world.wall_frame)
    assert row['entered_gap'] is False and row['entered_gap_frame'] is None
    assert row['in_gap_at_wall'] is False and row['frames_in_gap_before_wall'] == 0
    assert row['mechanism'] == 'stopped_left_of_the_gap'
    assert row['direction'] == 'hold' and row['first_move_frame'] is None
    assert row['hit_frame'] == world.wall_frame
    assert row['distance_to_gap_at_wall'] == pytest.approx(tel.canonical_gap(world)[0] - world.start_x)
    overshoot = tel.gap_diagnostics(world, [2] * 80, hit_frame=None)
    assert overshoot['mechanism'] in ('entered_the_gap_then_left_it', 'stopped_right_of_the_gap')
    assert overshoot['entered_gap'] is True


# ---------------------------------------------------------------- the closed loop runner
def test_closed_loop_runner_reproduces_the_frozen_runner(frozen):
    for event_id in ('wf18-F1-002-m0', 'wf18-F2-119-m1'):
        world = _world(frozen['all'], event_id)
        reference = run_event(world, 'N1', r20.CONDITION, _PixelHashNetwork(), 0, None)
        copied = tel.run_event_rendered(world, 'N1', r20.CONDITION, _PixelHashNetwork(), render=render_state)
        assert r20.compact(copied) == r20.compact(reference), event_id
        assert [step['candidate'] for step in copied['steps']] == \
               [step['candidate'] for step in reference['steps']]
        assert copied['first_collision_frame'] == reference['first_collision_frame']
        assert copied['feasible'] == reference['feasible']


def test_f2_f3_closed_loop_is_unchanged_under_the_telegraph(frozen):
    """Non-F1 pixels are identical, so the whole rollout must be identical too."""
    for event_id in ('wf18-F2-119-m0', 'wf18-F3-002-m1'):
        world = _world(frozen['all'], event_id)
        reference = tel.run_event_rendered(world, 'N1', r20.CONDITION, _PixelHashNetwork(),
                                           render=render_state)
        tegraphed = tel.run_event_rendered(world, 'N1', r20.CONDITION, _PixelHashNetwork())
        assert r20.compact(tegraphed) == r20.compact(reference), event_id


def test_closed_loop_runner_masks_to_the_six_supported_candidates(frozen):
    world = _world(frozen['all'], 'wf18-F1-006-m0')

    class Unsupported:
        def __call__(self, arm, regions, previous, hidden, reset):
            logits = torch.full((18,), -5.)
            logits[5] = 9.
            for candidate, value in ((0, 3.), (7, 2.), (3, 1.), (9, .5), (12, .25), (16, .1)):
                logits[candidate] = value
            return logits, None

    row = tel.run_event_rendered(world, 'N1', r20.CONDITION, Unsupported())
    assert len(row['steps']) == 80
    for step in row['steps']:
        assert step['candidate'] in HORIZONTAL
        assert step['raw_unmasked_candidate'] == 5 and step['raw_unsupported'] is True
        assert step['actual_action'] == HORIZONTAL[step['candidate']]
        assert step['invalid_action'] is False
    assert [step['candidate'] for step in row['steps']] == [0] * 80
    assert row['first_collision_frame'] == world.wall_frame
    assert row['survived'] is False and row['feasible'] is False


def test_the_runner_uses_the_real_advance(frozen):
    import touhou_rl.tactical_teacher_validation as validation
    assert tel.advance is validation.advance
    assert tel.render_state is render_state
    assert not hasattr(tel, 'static_collision')
    assert not hasattr(tel, 'full_advance')
    world = _world(frozen['all'], 'wf18-F1-002-m0')
    row = tel.run_event_rendered(world, 'N1', r20.CONDITION, _PixelHashNetwork())
    actions = [step['actual_action'] for step in row['steps']]
    state = initial_state(world)
    for action in actions:
        state = advance(world, state, action)
    assert state.x == pytest.approx(row['steps'][-1]['x_after'])
    assert bool(state.hit) == (row['first_collision_frame'] is not None)


# ---------------------------------------------------------------- frozen data contract
def test_render_freeze_and_the_data_build_agree(frozen):
    event = next(row for row in frozen['train'] if row['event_id'] == 'wf18-F1-006-m0')
    freeze = tel.render_freeze([event])
    payload = tel.render_event(event)
    assert freeze['sha256'] == hashlib.sha256(json.dumps(freeze['events'], sort_keys=True).encode()).hexdigest()
    for key in ('actions_sha256', 'original_sha256', 'telegraph_sha256', 'diff_sha256',
                'changed_pixels', 'changed_decision_frames', 'first_changed_frame', 'last_changed_frame',
                'columns_sha256'):
        assert freeze['events'][0][key] == payload['freeze'][key], key
    assert payload['freeze']['preview'] == [0, 40] and payload['freeze']['changed_decision_frames'] == 20
    assert payload['verdict'] == 'feasible' and payload['verified'] is True


def test_data_build_is_round20_exact_for_f2_f3_and_preview_only_for_f1(tmp_path, frozen):
    for event_id in ('wf18-F3-004-m0', 'wf18-F1-006-m0'):
        event = next(row for row in frozen['train'] if row['event_id'] == event_id)
        directory = r20.new_dir(tmp_path / event_id)
        item = tel.build_event(directory, event)
        checked = item['payload']['freeze']
        reference = r20_asset_payload(event_id)
        assert list(item['payload']['actions']) == reference['actions']
        for name in item['payload']['labels']:
            assert torch.equal(item['payload']['labels'][name], reference['labels'][name])
        for name in item['payload']['history']:
            assert torch.equal(item['payload']['history'][name], reference['history'][name])
        if event['family'] == 'F1':
            assert item['payload']['stored']['playfield'].shape == (80, 3, SIZES['playfield'][0],
                                                                    SIZES['playfield'][1], 3)
            assert checked['changed_pixels'] > 0
            assert item['refs']['pixels']['bytes'] < 8_000_000
        else:
            assert checked['changed_pixels'] == 0
            assert np.array_equal(item['payload']['stored']['playfield'], reference['stored']['playfield'])
        stored = json.loads((directory / 'rollout.json').read_text(encoding='utf-8'))
        assert stored['round20_containment']['actions_equal'] is True
        assert stored['round20_containment']['labels_equal'] is True
        assert stored['telegraph']['lead'] == 45


def r20_asset_payload(event_id):
    """Round20's stored arrays for one train event, read once and cached."""
    cache = r20_asset_payload.__dict__.setdefault('cache', {})
    if event_id not in cache:
        root = tel.R20_EVIDENCE / 'run'
        records = tel.r20_assets()[event_id]
        with np.load(root / records['pixels']['path']) as archive:
            stored = {name: np.array(archive[name]) for name in archive.files}
        rollout = json.loads((root / records['rollout']['path']).read_text(encoding='utf-8'))
        cache[event_id] = dict(stored=stored, actions=rollout['actions'],
                               labels=torch.load(root / records['labels']['path'], map_location='cpu',
                                                 weights_only=True),
                               history=torch.load(root / records['history']['path'], map_location='cpu',
                                                  weights_only=True))
    return cache[event_id]


def test_round20_reference_is_machine_checked():
    reference = tel.r20_reference()
    assert reference['f1_validation_feasible'] == {'A': [3, 3, 3], 'B': [2, 3, 3]}
    assert reference['all_validation_feasible'] == {'A': [4, 3, 4], 'B': [2, 3, 3]}
    assert reference['b0_f1_validation_feasible'] == 0
    assert reference['holdout_filter_valid'] is True
    assert len(reference['initial_head_state_sha256']) == 3
    assert set(reference['f1_validation_firsthit']) == {'A-s7', 'A-s17', 'A-s27', 'B-s7', 'B-s17', 'B-s27'}
    summary = json.loads((tel.R20_EVIDENCE / 'run/summary.json').read_text(encoding='utf-8'))
    assert reference['summary_sha256'] == hashlib.sha256(
        (tel.R20_EVIDENCE / 'run/summary.json').read_bytes()).hexdigest()
    assert reference['initial_head_state_sha256'] == summary['initialization']['per_seed']
    assert tel.R20_F1_VALIDATION_FEASIBLE == {'A': [3, 3, 3], 'B': [2, 3, 3]}


# ---------------------------------------------------------------- the pre-registration
def test_criterion_is_frozen_and_declares_the_comparison():
    value = tel.criterion()
    assert tel.criterion_sha() == tel.criterion_sha()
    assert value['intervention']['lead_frames'] == 45
    assert value['intervention']['style']['preview_fill'] == list(tel.TELEGRAPH_FILL)
    assert value['intervention']['style']['lethal_fill'] == list(tel.LETHAL_FILL)
    assert 'no replanning' in value['data']['teacher']
    assert value['training']['updates'] == 200 and value['training']['anchor_weight'] == 1.
    assert value['training']['seeds'] == [7, 17, 27]
    assert value['evaluation']['comparison'].startswith('round21 F1 validation feasible per seed')
    assert value['evaluation']['holdout_rule'].startswith('if ANY of the six R18 baselines')
    assert value['claims']['cannot_show'].startswith('learnability of the other families')
    assert 'no retrain' in value['failure_policy']
    tampered = json.loads(json.dumps(value))
    tampered['intervention']['lead_frames'] = 30
    assert tel.criterion_sha(tampered) != tel.criterion_sha()
    tampered = json.loads(json.dumps(value))
    tampered['training']['anchor_weight'] = 2.
    assert tel.criterion_sha(tampered) != tel.criterion_sha()


def test_identity_paths_and_expected_counts_exist():
    paths = tel.identity_paths()
    missing = [str(path) for path in paths if not Path(path).is_file()]
    assert not missing, missing
    hashes = tel.identities()
    assert len(hashes) == len(paths) and all(len(digest) == 64 for digest in hashes.values())
    assert str(tel.R20_EVIDENCE / 'run/summary.json') in hashes
    assert str(tel.BASELINE) in hashes and str(r20.MANIFEST) in hashes
    assert tel.EXPECTED['checkpoints'] == 6 and tel.EXPECTED['closed_loop_files'] == 14
    assert tel.EXPECTED['baseline_rows'] == 168
    assert tel.EXPECTED['train_events'] == 14 and tel.EXPECTED['validation_events'] == 14


def test_arms_seeds_and_protocol_are_round20s():
    assert tel.ARMS == r20.ARMS == ('A', 'B')
    assert tel.SEEDS == r20.SEEDS == (7, 17, 27)
    assert tel.CONDITION == r20.CONDITION == 'horizontal_masked'
    assert tel.ACTION_MAP == ACTION_MAP == {-2: 7, -1: 16, 0: 0, 1: 12, 2: 3}
    assert tel.HORIZONTAL == {0: 0, 3: 2, 7: -2, 9: 0, 12: 1, 16: -1}
    assert r20.all_names() == ['A-s7', 'A-s17', 'A-s27', 'B-s7', 'B-s17', 'B-s27']


def test_span_table_is_pure_geometry(frozen):
    table = tel.span_table(frozen['all'])
    assert len(table) == 28
    assert table['wf18-F1-002-m0']['window'] == [31, 76]
    assert table['wf18-F1-002-m0']['spans'] == [[8, 88], [104, 376]]
    assert table['wf18-F1-002-m0']['lethal_window'] == [76, 88]
    for event_id in ('wf18-F2-119-m0', 'wf18-F3-000-m0'):
        assert table[event_id]['window'] is None and table[event_id]['spans'] == []
        assert table[event_id]['columns_sha256'] == hashlib.sha256(np.zeros(384, np.uint8).tobytes()).hexdigest()
    assert table['wf18-F1-002-m0']['columns_sha256'] == \
        hashlib.sha256(tel.preview_columns(_world(frozen['all'], 'wf18-F1-002-m0')).tobytes()).hexdigest()


def test_policy_loss_and_action_map_are_untouched(frozen):
    """The supervision is round20's; this round adds no label and no loss term."""
    assert tactical_policy_loss.__module__ == 'touhou_rl.tactical_policy'
    assert r20.arm_loss.__module__ == 'tools.rl.v373_valid_curriculum'
    actions = [0, -2, 2, 1, -1, 0, 0, 0] * 10
    history, labels = r20.history_and_labels(actions, True)
    assert set(labels) == {'action_positive', 'action_valid'}
    assert labels['action_positive'].long().argmax(-1).tolist() == [ACTION_MAP[a] for a in actions]
    assert history['previous_candidate'].tolist() == [18] + [ACTION_MAP[a] for a in actions[:-1]]
