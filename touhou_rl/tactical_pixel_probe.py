"""Privileged-anchor, offline pixel diagnostics; NEVER a deployment detector/gate.

One stable memory point bootstraps a single image template. All later NCC searches
are across a fixed full playfield (or explicitly supplied broad rectangle), never
around the current memory point. Counter-based lag hypotheses are not image-phase
proof, even when there is exactly one best hypothesis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


WARNING = ('Privileged first-template extraction uses memory. Not deployment vision, '
           'not independent sprite identity, and not formal screenshot synchronization proof.')


def _gray(rgb):
    a = np.asarray(rgb, dtype=np.float64)
    return a[..., 0] * .299 + a[..., 1] * .587 + a[..., 2] * .114


def _feature(gray, mode):
    if mode == 'raw':
        return gray
    if mode == 'gradient':
        gy, gx = np.gradient(gray)
        return np.hypot(gx, gy)
    raise ValueError('mode must be raw or gradient')


def _window_sum(a, h, w):
    integral = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]


def ncc_map(image, template):
    """Zero-mean normalized cross correlation, valid windows, pure numpy FFT."""
    image = np.asarray(image, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    h, w = template.shape
    ih, iw = image.shape
    if h > ih or w > iw or min(h, w) < 3:
        raise ValueError('template must fit image and have each dimension >= 3')
    t = template - template.mean()
    energy = float(np.sum(t * t))
    if energy < 1e-10:
        return np.full((ih - h + 1, iw - w + 1), -1.)
    shape = (ih + h - 1, iw + w - 1)
    corr = np.fft.irfft2(np.fft.rfft2(image, s=shape) *
                         np.fft.rfft2(t[::-1, ::-1], s=shape), s=shape)
    corr = corr[h - 1:ih, w - 1:iw]
    count = h * w
    variance = np.maximum(_window_sum(image * image, h, w) -
                          _window_sum(image, h, w) ** 2 / count, 0.)
    denom = np.sqrt(variance * energy)
    out = np.full(corr.shape, -1.)
    np.divide(corr, denom, out=out, where=denom > 1e-9)
    return np.clip(out, -1., 1.)


def locate_template(rgb, template_rgb, *, roi=(32, 16, 384, 448), mode='raw',
                    min_confidence=.65, min_peak_margin=.06):
    """Search a fixed rectangle; neither signature nor implementation takes memory."""
    x, y, width, height = map(int, roi)
    if x < 0 or y < 0 or x + width > rgb.shape[1] or y + height > rgb.shape[0]:
        raise ValueError('search ROI outside image')
    # Feature extraction on full images avoids window-boundary gradient artifacts.
    image = _feature(_gray(rgb), mode)[y:y + height, x:x + width]
    template = _feature(_gray(template_rgb), mode)
    scores = ncc_map(image, template)
    iy, ix = np.unravel_index(np.argmax(scores), scores.shape)
    peak = float(scores[iy, ix])
    h, w = template.shape
    others = scores.copy()
    # Suppress only a local peak footprint, not remote look-alike sprites.
    ry, rx = max(3, h // 2), max(3, w // 2)
    others[max(0, iy - ry):iy + ry + 1, max(0, ix - rx):ix + rx + 1] = -1.
    sy, sx = np.unravel_index(np.argmax(others), others.shape)
    second = float(others[sy, sx])
    reason = ('low_confidence_background_or_animation_unknown' if peak < min_confidence else
              'two_peaks_background_or_sprite_identity_unknown' if peak - second < min_peak_margin else
              'single_template_match_not_identity_proof')
    return {'xy': [float(x + ix + w / 2), float(y + iy + h / 2)],
            'confidence': peak, 'second_confidence': second, 'peak_margin': peak - second,
            'second_xy': [float(x + sx + w / 2), float(y + sy + h / 2)],
            'confident': peak >= min_confidence and peak - second >= min_peak_margin,
            'reason': reason, 'mode': mode}


def _position(state):
    return np.asarray([state['player_x'], state['player_y']], dtype=float)


def _stable(record):
    a, b = record['pre'], record['post']
    keys = ('gameplay_frame', 'player_state', 'focus_mode')
    return (all(a.get(k) == b.get(k) for k in keys) and
            np.isfinite(_position(a)).all() and np.array_equal(_position(a), _position(b)))


def extract_privileged_template(frames, trace, *, origin=(32, 16), size=24):
    """Only this function uses memory to locate image content, exactly once."""
    if size not in (24, 32):
        raise ValueError('template size must be 24 or 32')
    for r in trace:
        if not _stable(r):
            continue
        index = int(r['index'])
        center = _position(r['pre']) + np.asarray(origin)
        x, y = np.rint(center - size / 2).astype(int)
        if x < 0 or y < 0 or x + size > frames.shape[2] or y + size > frames.shape[1]:
            continue
        template = frames[index, y:y + size, x:x + size].copy()
        if float(_gray(template).std()) < 1e-5:
            continue
        return template, {'index': index, 'counter': r['pre']['gameplay_frame'],
                          'crop_xywh': [int(x), int(y), size, size],
                          'memory_xy': _position(r['pre']).tolist(),
                          'player_state': r['pre'].get('player_state'),
                          'focus_mode': r['pre'].get('focus_mode'),
                          'privileged': True, 'warning': WARNING,
                          'stable_bracket_is_not_sprite_identity_or_display_phase_proof': True}
    raise ValueError('no stable, nonconstant initial template candidate')


def counter_lag_diagnostic(trace, localizations, *, max_lag=8, min_samples=12,
                           margin=.35, max_error=2.):
    """Fit fixed anchor per lag; ambiguous memory/counter duplicates are omitted.

    This intentionally reports COUNTER hypotheses even without verified phase.
    No interpolation, no first/last selection among conflicting frame positions.
    """
    memory_sets = {}
    for r in trace:
        for side in ('pre', 'post'):
            s = r[side]
            xy = tuple(_position(s))
            if np.isfinite(xy).all():
                memory_sets.setdefault(int(s['gameplay_frame']), set()).add(xy)
    memory = {f: np.asarray(next(iter(v))) for f, v in memory_sets.items() if len(v) == 1}
    conflicts = sorted(f for f, v in memory_sets.items() if len(v) > 1)
    by_index = {int(r['index']): r for r in trace}
    pixel_sets = {}
    excluded_brackets = 0
    for loc in localizations:
        r = by_index[loc['index']]
        if not loc['confident']:
            continue
        if not _stable(r):
            excluded_brackets += 1
            continue
        f = int(r['pre']['gameplay_frame'])
        pixel_sets.setdefault(f, []).append(loc['xy'])
    pixels = {}
    duplicate_conflicts = []
    for f, values in pixel_sets.items():
        a = np.asarray(values)
        if np.max(np.linalg.norm(a - a[0], axis=1)) > 1.:
            duplicate_conflicts.append(f)
        else:
            pixels[f] = np.median(a, axis=0)
    # Exactly the same sample counter set for every lag, avoiding edge selection bias.
    common = sorted(f for f in pixels if all(f - lag in memory
                                            for lag in range(-max_lag, max_lag + 1)))
    result = {'status': 'unknown', 'phase_status': 'unknown_counter_is_not_display_phase',
              'real_pixel_gate_passed': False, 'scope': 'counter_based_candidate_only',
              'memory_conflicting_counters': conflicts,
              'pixel_conflicting_duplicate_counters': sorted(duplicate_conflicts),
              'excluded_unstable_brackets': excluded_brackets,
              'samples_common_to_every_lag': len(common), 'lag_candidates': [], 'scores': [],
              'positive_lag_means': 'image_trails_memory', 'warning': WARNING}
    result['minimum_samples_required'] = min_samples
    if not common:
        result['reason'] = 'insufficient_unambiguous_common_counter_coverage'
        return result
    q = np.stack([pixels[f] for f in common])
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        p = np.stack([memory[f - lag] for f in common])
        offset = np.median(q - p, axis=0)
        errors = np.linalg.norm(q - p - offset, axis=1)
        scores.append({'lag': lag, 'rmse': float(np.sqrt(np.mean(errors ** 2))),
                       'median_error': float(np.median(errors)),
                       'anchor_offset': offset.tolist(), 'samples': len(common)})
    scores.sort(key=lambda s: s['rmse'])
    best = scores[0]['rmse']
    candidates = [s['lag'] for s in scores if s['rmse'] <= best + margin]
    result.update(scores=scores, lag_candidates=candidates)
    if len(common) < min_samples:
        result['reason'] = 'insufficient_unambiguous_common_counter_coverage'
        result['scores_are_exploratory_only'] = True
        return result
    if best > max_error:
        reason = 'pixel_memory_geometry_disagree_or_animation_unknown'
    elif len(candidates) != 1:
        reason = 'lag_not_identifiable_static_constant_or_periodic_motion'
    else:
        reason = 'unique_counter_candidate_not_synchronization_proof'
        result['counter_candidate_status'] = ('outside_one_frame' if abs(candidates[0]) > 1
                                              else 'within_one_frame_candidate_only')
    result['reason'] = reason
    return result


def analyze(frames, trace, *, origin=(32, 16), size=24, roi=None,
            max_lag=8, min_samples=12):
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (480, 640, 3):
        raise ValueError('expected uint8 rgb[N,480,640,3], no silent scaling')
    indices = [int(r['index']) for r in trace]
    if len(indices) != len(set(indices)) or any(i < 0 or i >= len(frames) for i in indices):
        raise ValueError('duplicate or out-of-range capture index')
    roi = tuple(roi or (int(origin[0]), int(origin[1]), 384, 448))
    template, anchor = extract_privileged_template(frames, trace, origin=origin, size=size)
    rows = []
    for r in trace:
        index = int(r['index'])
        modes = {m: locate_template(frames[index], template, roi=roi, mode=m)
                 for m in ('raw', 'gradient')}
        raw, grad = modes['raw'], modes['gradient']
        agreement = float(np.linalg.norm(np.asarray(raw['xy']) - grad['xy']))
        same_context = (r['pre'].get('player_state') == anchor['player_state'] and
                        r['post'].get('player_state') == anchor['player_state'] and
                        r['pre'].get('focus_mode') == anchor['focus_mode'] and
                        r['post'].get('focus_mode') == anchor['focus_mode'])
        confident = raw['confident'] and grad['confident'] and agreement <= 3 and same_context
        reason = ('raw_gradient_agree_candidate_only' if confident else
                  'state_or_focus_changed_animation_unknown' if not same_context else
                  'raw_gradient_disagree_or_low_confidence_or_two_peaks')
        rows.append({'index': index, 'xy': raw['xy'], 'confident': bool(confident),
                     'reason': reason, 'raw_gradient_distance': agreement, **modes})
    # Do not use the anchor capture as independent localization/lag evidence.
    independent = [r for r in rows if r['index'] != anchor['index']]
    lag = counter_lag_diagnostic(trace, independent, max_lag=max_lag, min_samples=min_samples)
    return {'schema': 'privileged_template_pixel_probe_v1', 'real_pixel_gate_passed': False,
            'status': 'offline_diagnostic_only', 'warning': WARNING,
            'anchor': anchor, 'search_roi_xywh': list(roi),
            'search_policy': 'fixed_full_playfield_or_user_broad_envelope_no_current_memory_window',
            'capture_count': len(trace), 'confident_count': sum(r['confident'] for r in independent),
            'unknown_count': sum(not r['confident'] for r in independent),
            'frame_rgb_sha256': [hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest()
                                 for f in frames],
            'localizations': rows, 'counter_lag_diagnostic': lag,
            'limitations': ['Initial crop can lock onto background or non-player feature.',
                            'One animation template cannot establish sprite identity or cover blinking/death.',
                            'NCC confidence is not calibrated probability; two-mode agreement is not independence.',
                            'Stable memory bracket/counter is not a Present fence or verified display phase.',
                            'No real capture gate can pass from this diagnostic.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, choices=(24, 32), default=24)
    parser.add_argument('--origin', type=int, nargs=2, default=(32, 16))
    parser.add_argument('--roi', type=int, nargs=4)
    parser.add_argument('--max-lag', type=int, default=8)
    parser.add_argument('--min-samples', type=int, default=12)
    args = parser.parse_args(argv)
    with np.load(args.frames, allow_pickle=False) as archive:
        frames = archive['rgb']
    trace = json.loads(args.trace.read_text(encoding='utf-8'))
    result = analyze(frames, trace, origin=args.origin, size=args.size, roi=args.roi,
                     max_lag=args.max_lag, min_samples=args.min_samples)
    result['inputs'] = {key: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                        for key, path in (('frames', args.frames), ('trace', args.trace))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('capture_count', 'confident_count', 'unknown_count')},
                     ensure_ascii=False))
    print(json.dumps(result['counter_lag_diagnostic'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
