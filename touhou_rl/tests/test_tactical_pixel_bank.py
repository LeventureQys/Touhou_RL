import copy
import json
import numpy as np
import pytest
from touhou_rl.tactical_pixel_bank import (DEFAULT_CONFIG, extract_bank, locate_bank,
    analyze_bank, motion_coverage, validate_config, write_new, main)
from touhou_rl.tests.test_tactical_pixel_probe import image_at, sprite, record


def test_fixed_slots_never_replace_unstable_or_constant():
    trace = [record(i, (168, 204)) for i in range(61)]
    frames = np.stack([image_at(200, 220) for _ in trace])
    trace[10]['post']['player_x'] += 1
    frames[28] = 0
    templates, anchors, slots = extract_bank(frames, trace)
    assert len(templates) == 4
    assert [a['index'] for a in anchors] == [0, 39, 48, 57]
    assert 'unstable' in slots[1]['reason']
    assert 'constant' in slots[2]['reason']
    assert all(a['index'] < 60 for a in anchors)


def test_duplicate_counter_first_index_no_success_cherry_pick():
    trace = [record(0, (168, 204)), record(1, (168, 204))]
    trace[1]['pre']['gameplay_frame'] = trace[1]['post']['gameplay_frame'] = 0
    frames = np.stack([np.zeros((480, 640, 3), np.uint8), image_at(200, 220)])
    templates, anchors, slots = extract_bank(frames, trace)
    assert len(templates) == 0 and anchors == []
    assert slots[0]['index'] == 0


def test_global_bank_without_memory_and_support_threshold():
    templates = np.stack([sprite(), sprite()])
    for xy in ((70, 60), (380, 420)):
        result = locate_bank(image_at(*xy), templates)
        assert result['confident'] and result['xy'] == list(xy)
    assert not locate_bank(image_at(70, 60), templates[:1])['confident']
    assert not locate_bank(np.zeros((480, 640, 3), np.uint8), templates)['confident']


def test_cross_bank_confident_disagreement_unknown():
    s1 = sprite()
    s2 = np.random.default_rng(13).integers(0, 255, (24, 24, 3), dtype=np.uint8)
    im = image_at(70, 70, s1)
    im[388:412, 368:392] = s2
    found = locate_bank(im, np.stack([s1, s2]))
    assert found['support'] == 2
    assert not found['confident']
    assert found['reason'] == 'cross_template_disagreement_unknown'


def test_duplicate_remote_peaks_unknown():
    im = image_at(70, 70)
    im[388:412, 368:392] = sprite()
    assert not locate_bank(im, np.stack([sprite(), sprite()]))['confident']


def motion_fixture(static=False):
    trace = []
    pos = np.array([150., 200.])
    directions = [(('right',), (2, 0)), (('up',), (0, -2)),
                  (('left',), (-2, 0)), (('down',), (0, 2)), (('right',), (2, 0))]
    for i in range(100):
        keys, delta = directions[min(i // 20, 4)]
        if not static:
            pos += delta
        r = record(i, pos)
        r['keys'] = list(keys)
        trace.append(r)
    rows = [{'index': i, 'xy': [r['pre']['player_x']+32, r['pre']['player_y']+16],
             'confident': True} for i, r in enumerate(trace)]
    return trace, rows


def test_dynamic_and_three_turn_coverage_required_not_static_padding():
    trace, rows = motion_fixture()
    cov = motion_coverage(trace, rows, DEFAULT_CONFIG)
    assert cov['coverage_sufficient'] and cov['covered_turns'] >= 3
    assert cov['dynamic_samples'] >= 12
    trace, rows = motion_fixture(static=True)
    cov = motion_coverage(trace, rows, DEFAULT_CONFIG)
    assert len(cov['common_counters']) > 12
    assert not cov['coverage_sufficient'] and cov['dynamic_samples'] == 0


def test_phase_unknown_and_all_dev_anchors_excluded():
    frames = np.stack([image_at(200, 220), image_at(370, 400)])
    trace = [record(0, (168, 204)), record(1, (338, 384))]
    bank = np.stack([sprite(), sprite()])
    result = analyze_bank(frames, trace, bank, [{'index': 0}], role='development')
    assert result['confident_count'] == 1
    assert result['excluded_anchor_indices'] == [0]
    assert not result['real_pixel_gate_passed']
    assert result['phase_status'].startswith('unknown')
    assert result['counter_lag_diagnostic']['minimum_samples_required'] == 12
    confirmation = analyze_bank(frames, trace, bank, [{'index': 0}], role='confirmation')
    assert confirmation['excluded_anchor_indices'] == []


def test_frozen_thresholds_and_no_silent_resize():
    c = copy.deepcopy(DEFAULT_CONFIG)
    c['min_confidence'] = .64
    with pytest.raises(ValueError, match='frozen'):
        validate_config(c)
    with pytest.raises(ValueError, match='no silent scaling'):
        extract_bank(np.zeros((1, 240, 320, 3), np.uint8), [record(0, (1, 1))])


def test_exclusive_output_and_cli_preflight(tmp_path):
    path = tmp_path/'out.json'
    write_new(path, {'first': True})
    with pytest.raises(FileExistsError):
        write_new(path, {'first': False})
    assert json.loads(path.read_text()) == {'first': True}
    with pytest.raises(FileExistsError):
        main(['run', '--frames', 'missing.npz', '--trace', 'missing.json',
              '--bank-dir', 'missing', '--role', 'development', '--output', str(path)])
