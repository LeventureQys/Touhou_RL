import numpy as np
import pytest

from touhou_rl.tactical_pixel_probe import (
    analyze, counter_lag_diagnostic, extract_privileged_template, locate_template, ncc_map,
)


def sprite():
    rng = np.random.default_rng(8)
    s = np.zeros((24, 24, 3), dtype=np.uint8)
    s[2:22, 3:21] = rng.integers(40, 255, (20, 18, 3), dtype=np.uint8)
    return s


def image_at(x, y, s=None):
    im = np.zeros((480, 640, 3), dtype=np.uint8)
    s = sprite() if s is None else s
    im[y - 12:y + 12, x - 12:x + 12] = s
    return im


def state(frame, xy):
    return {'gameplay_frame': frame, 'player_x': float(xy[0]), 'player_y': float(xy[1]),
            'player_state': 1, 'gui_input_current': 0, 'focus_mode': 0}


def record(i, xy):
    return {'index': i, 'pre': state(i, xy), 'post': state(i, xy),
            'capture_interval': [i, i], 'phase_verified': False}


def test_ncc_agrees_with_bruteforce():
    rng = np.random.default_rng(2)
    im = rng.normal(size=(13, 17))
    t = im[5:9, 6:11].copy()
    actual = ncc_map(im, t)
    expected = np.zeros(actual.shape)
    tc = t - t.mean()
    for y in range(expected.shape[0]):
        for x in range(expected.shape[1]):
            a = im[y:y + 4, x:x + 5]
            a = a - a.mean()
            expected[y, x] = np.sum(a * tc) / np.sqrt(np.sum(a*a) * np.sum(tc*tc))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize('mode', ['raw', 'gradient'])
def test_full_playfield_global_jump_not_current_memory_window(mode):
    template = sprite()
    for xy in ((70, 60), (380, 420), (90, 400)):
        found = locate_template(image_at(*xy), template, mode=mode)
        assert found['confident']
        assert found['xy'] == list(xy)


def test_two_remote_equal_peaks_unknown():
    im = image_at(70, 70)
    im[388:412, 368:392] = sprite()
    result = locate_template(im, sprite())
    assert not result['confident']
    assert 'two_peaks' in result['reason']


def test_blank_or_animation_unknown():
    found = locate_template(np.zeros((480, 640, 3), np.uint8), sprite())
    assert not found['confident']
    assert 'low_confidence' in found['reason']


def path(i):
    return (180 + int(80 * np.sin(i * .19)) + i % 7,
            220 + int(70 * np.cos(i * .13)) + i % 11)


def test_synthetic_sprite_global_motion_and_plus_three_lag():
    # Pixel localization is independent of memory and covers distant full-field positions.
    trace = [record(i, path(i)) for i in range(85)]
    pixels = []
    for i in range(5, 80):
        p = path(i - 3)
        xy = (p[0] + 32, p[1] + 16)
        result = locate_template(image_at(*xy), sprite())
        pixels.append({'index': i, **result})
    diag = counter_lag_diagnostic(trace, pixels, max_lag=5)
    assert diag['lag_candidates'] == [3]
    assert diag['counter_candidate_status'] == 'outside_one_frame'
    assert diag['status'] == 'unknown'
    assert diag['real_pixel_gate_passed'] is False
    assert diag['phase_status'].startswith('unknown')


@pytest.mark.parametrize('lag', [-3, 0, 3])
def test_constant_velocity_anchor_ambiguity_stays_unknown(lag):
    trace = [record(i, (i * 2, 100 + i)) for i in range(90)]
    pixels = [{'index': i, 'xy': [32 + (i - lag) * 2, 116 + (i - lag)], 'confident': True}
              for i in range(10, 80)]
    diag = counter_lag_diagnostic(trace, pixels, max_lag=5)
    assert diag['status'] == 'unknown'
    assert len(diag['lag_candidates']) == 11
    assert 'not_identifiable' in diag['reason']
    assert diag['real_pixel_gate_passed'] is False


def test_conflicting_memory_counter_not_arbitrarily_first_or_last():
    trace = [record(i, path(i)) for i in range(50)]
    extra = record(30, (999, 999))
    extra['index'] = 50
    trace.append(extra)
    pixels = [{'index': i, 'xy': path(i), 'confident': True} for i in range(8, 44)]
    diag = counter_lag_diagnostic(trace, pixels, max_lag=3)
    assert diag['memory_conflicting_counters'] == [30]
    assert diag['samples_common_to_every_lag'] == 29


def test_duplicate_counter_pixels_disagree_unknown():
    trace = [record(i, path(i)) for i in range(50)]
    extra = record(20, path(20))
    extra['index'] = 50
    trace.append(extra)
    pixels = [{'index': i, 'xy': path(i), 'confident': True} for i in range(8, 44)]
    pixels.append({'index': 50, 'xy': [999, 999], 'confident': True})
    diag = counter_lag_diagnostic(trace, pixels, max_lag=3)
    assert diag['pixel_conflicting_duplicate_counters'] == [20]


def test_anchor_is_explicit_privilege_and_excluded_from_lag():
    frames = np.stack([image_at(200, 220), image_at(370, 400)])
    trace = [record(0, (168, 204)), record(1, (338, 384))]
    result = analyze(frames, trace)
    assert result['anchor']['privileged'] is True
    assert 'Not deployment' in result['warning']
    assert result['localizations'][1]['xy'] == [370., 400.]
    assert result['confident_count'] == 1
    assert result['counter_lag_diagnostic']['status'] == 'unknown'
    assert result['real_pixel_gate_passed'] is False
    assert len(result['frame_rgb_sha256']) == 2


def test_insufficient_samples_still_emit_exploratory_scores_without_claim():
    trace = [record(i, path(i)) for i in range(50)]
    pixels = [{'index': i, 'xy': path(i - 3), 'confident': True} for i in range(10, 20)]
    diag = counter_lag_diagnostic(trace, pixels, max_lag=5, min_samples=12)
    assert len(diag['scores']) == 11
    assert diag['scores_are_exploratory_only'] is True
    assert diag['status'] == 'unknown'
    assert diag['real_pixel_gate_passed'] is False
    assert 'insufficient' in diag['reason']


def test_no_stable_template_and_no_silent_resize():
    im = image_at(200, 220)[None]
    row = record(0, (168, 204))
    row['post']['player_x'] += 1
    with pytest.raises(ValueError, match='stable'):
        extract_privileged_template(im, [row])
    with pytest.raises(ValueError, match='no silent scaling'):
        analyze(im[:, ::2], [row])
