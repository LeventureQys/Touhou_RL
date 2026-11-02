"""Frozen privileged animation-bank OFFLINE diagnostic; never a formal pixel gate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .tactical_pixel_probe import (WARNING, _gray, _position, _stable,
                                   locate_template, counter_lag_diagnostic)

DEFAULT_CONFIG = {
    'schema': 'privileged_pixel_bank_config_v1',
    'anchor_offsets': [0, 10, 28, 39, 48, 57],
    'anchor_window': 60, 'size': 24, 'origin': [32, 16],
    'roi': [32, 16, 384, 448], 'min_confidence': .65,
    'min_peak_margin': .06, 'mode_distance': 3., 'bank_distance': 3.,
    'min_support': 2, 'max_lag': 8, 'min_samples': 12,
    'min_dynamic_samples': 12, 'min_covered_turns': 3,
    'motion_threshold': .25, 'turn_radius': 3,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def validate_config(c):
    if c != DEFAULT_CONFIG:
        raise ValueError('only the frozen preregistered config is accepted; no confirmation tuning')


def validate_data(frames, trace):
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (480, 640, 3):
        raise ValueError('expected uint8 rgb[N,480,640,3], no silent scaling')
    indices = [int(r['index']) for r in trace]
    if not trace or len(set(indices)) != len(indices) or any(i < 0 or i >= len(frames) for i in indices):
        raise ValueError('empty trace or duplicate/out-of-range indices')


def extract_bank(frames, trace, config=None):
    """Exactly first capture at each fixed logical offset; NEVER retry a failed slot."""
    c = DEFAULT_CONFIG if config is None else config
    validate_config(c)
    validate_data(frames, trace)
    begin = min(int(r['pre']['gameplay_frame']) for r in trace)
    templates, anchors, slots = [], [], []
    for offset in c['anchor_offsets']:
        candidates = sorted((r for r in trace if int(r['pre']['gameplay_frame']) == begin + offset),
                            key=lambda r: int(r['index']))
        slot = {'offset': offset, 'privileged': True, 'selection': 'first_index_at_exact_counter_no_retry'}
        slots.append(slot)
        if not candidates:
            slot['reason'] = 'missing_exact_counter_no_replacement'
            continue
        r = candidates[0]
        index = int(r['index'])
        slot['index'] = index
        if not _stable(r):
            slot['reason'] = 'unstable_first_bracket_no_replacement'
            continue
        size = c['size']
        x, y = np.rint(_position(r['pre']) + c['origin'] - np.asarray(size / 2)).astype(int)
        if x < 0 or y < 0 or x + size > frames.shape[2] or y + size > frames.shape[1]:
            slot['reason'] = 'crop_outside_image_no_replacement'
            continue
        template = frames[index, y:y+size, x:x+size].copy()
        if _gray(template).std() < 1e-5:
            slot['reason'] = 'constant_crop_no_replacement'
            continue
        slot['reason'] = 'extracted_without_ncc_selection'
        anchors.append({**slot, 'counter': begin + offset, 'crop_xywh': [int(x), int(y), size, size],
                        'memory_xy': _position(r['pre']).tolist(),
                        'player_state': r['pre'].get('player_state'),
                        'focus_mode': r['pre'].get('focus_mode'),
                        'keys': r.get('keys', []),
                        'rgb_sha256': hashlib.sha256(template.tobytes()).hexdigest()})
        templates.append(template)
    return np.asarray(templates, dtype=np.uint8).reshape((-1, c['size'], c['size'], 3)), anchors, slots


def locate_bank(rgb, templates, config=None):
    """No current memory/state argument: independent global image search only."""
    c = DEFAULT_CONFIG if config is None else config
    candidates = []
    for i, template in enumerate(templates):
        modes = {m: locate_template(rgb, template, roi=c['roi'], mode=m,
                    min_confidence=c['min_confidence'], min_peak_margin=c['min_peak_margin'])
                 for m in ('raw', 'gradient')}
        distance = float(np.linalg.norm(np.asarray(modes['raw']['xy']) - modes['gradient']['xy']))
        confident = all(v['confident'] for v in modes.values()) and distance <= c['mode_distance']
        candidates.append({'template_id': i, **modes, 'xy': modes['raw']['xy'],
                           'confident': bool(confident), 'raw_gradient_distance': distance})
    good = [r for r in candidates if r['confident']]
    diameter = max((float(np.linalg.norm(np.asarray(a['xy']) - b['xy']))
                    for a in good for b in good), default=0.)
    confident = len(good) >= c['min_support'] and diameter <= c['bank_distance']
    return {'xy': np.median([r['xy'] for r in good], axis=0).tolist() if good else [0., 0.],
            'confident': bool(confident), 'support': len(good), 'candidate_diameter': diameter,
            'reason': ('consistent_bank_candidate_not_identity_proof' if confident else
                       'cross_template_disagreement_unknown' if diameter > c['bank_distance'] else
                       'insufficient_confident_templates_unknown'), 'templates': candidates}


def motion_coverage(trace, rows, c):
    # Match counter_lag_diagnostic's unambiguous common-counter policy exactly.
    memory_sets = {}
    for r in trace:
        for side in ('pre', 'post'):
            s = r[side]
            if np.isfinite(_position(s)).all():
                memory_sets.setdefault(int(s['gameplay_frame']), set()).add(tuple(_position(s)))
    memory = {f: np.array(next(iter(v))) for f, v in memory_sets.items() if len(v) == 1}
    by_index = {r['index']: r for r in trace}
    pixels = {}
    for loc in rows:
        r = by_index[loc['index']]
        if loc['confident'] and _stable(r):
            pixels.setdefault(int(r['pre']['gameplay_frame']), []).append(loc['xy'])
    common = sorted(f for f, values in pixels.items()
                    if max(np.linalg.norm(np.asarray(v) - values[0]) for v in values) <= 1.
                    and all(f-lag in memory for lag in range(-c['max_lag'], c['max_lag']+1)))
    dynamic = [f for f in common if f-1 in memory and
               np.linalg.norm(memory[f] - memory[f-1]) > c['motion_threshold']]
    # Turn events from observed intended keys, not pixel-success-dependent selection.
    directions = {}
    for r in sorted(trace, key=lambda r: r['index']):
        f = int(r['pre']['gameplay_frame'])
        directions.setdefault(f, tuple(k for k in sorted(r.get('keys', []))
                                       if k in ('left', 'right', 'up', 'down')))
    previous = None
    turns = []
    for f, direction in sorted(directions.items()):
        if direction:
            if previous is not None and direction != previous:
                turns.append(f)
            previous = direction
    covered = [f for f in turns if any(abs(f-g) <= c['turn_radius'] for g in dynamic)]
    passed = len(dynamic) >= c['min_dynamic_samples'] and len(covered) >= c['min_covered_turns']
    return {'common_counters': common, 'dynamic_counters': dynamic, 'dynamic_samples': len(dynamic),
            'static_samples': len(common)-len(dynamic), 'turn_counters': turns,
            'covered_turn_counters': covered, 'covered_turns': len(covered),
            'coverage_sufficient': passed, 'reason': 'covered' if passed else 'insufficient_dynamic_or_turn_coverage'}


def analyze_bank(frames, trace, templates, anchors, *, role, config=None):
    c = DEFAULT_CONFIG if config is None else config
    validate_config(c)
    validate_data(frames, trace)
    if role not in ('development', 'confirmation'):
        raise ValueError('explicit development/confirmation role required')
    rows = [{'index': int(r['index']), **locate_bank(frames[int(r['index'])], templates, c)} for r in trace]
    excluded = {a['index'] for a in anchors} if role == 'development' else set()
    independent = [r for r in rows if r['index'] not in excluded]
    lag = counter_lag_diagnostic(trace, independent, max_lag=c['max_lag'], min_samples=c['min_samples'])
    coverage = motion_coverage(trace, independent, c)
    lag['motion_coverage'] = coverage
    if not coverage['coverage_sufficient']:
        lag['scores_are_exploratory_only'] = True
        lag['coverage_reason'] = coverage['reason']
    return {'schema': 'privileged_pixel_bank_v1', 'role': role, 'warning': WARNING,
            'status': 'offline_diagnostic_only', 'phase_status': 'unknown_counter_is_not_display_phase',
            'real_pixel_gate_passed': False, 'capture_count': len(rows),
            'confident_count': sum(r['confident'] for r in independent),
            'unknown_count': sum(not r['confident'] for r in independent),
            'excluded_anchor_indices': sorted(excluded), 'localizations': rows,
            'counter_lag_diagnostic': lag,
            'search_policy': 'fixed_full_playfield_no_current_memory_localization',
            'frame_rgb_sha256': [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--frames', type=Path, required=True)
    freeze.add_argument('--trace', type=Path, required=True)
    freeze.add_argument('--out', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--frames', type=Path, required=True)
    run.add_argument('--trace', type=Path, required=True)
    run.add_argument('--bank-dir', type=Path, required=True)
    run.add_argument('--role', choices=['development', 'confirmation'], required=True)
    run.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.command == 'run' and a.output.exists():
        raise FileExistsError('refusing to overwrite diagnostic output')
    if a.command == 'freeze' and a.out.exists():
        raise FileExistsError('new bank directory required')
    with np.load(a.frames, allow_pickle=False) as z:
        frames = z['rgb']
    trace = json.loads(a.trace.read_text(encoding='utf-8'))
    if a.command == 'freeze':
        a.out.mkdir(parents=True)
        write_new(a.out/'bank-config.json', DEFAULT_CONFIG)
        write_new(a.out/'prereg.json', {'schema': 'pixel_bank_prereg_v1', 'development_only': 'capture-round5',
            'confirmation': 'new capture-round6; reuse frozen bank; no tuning',
            'selection': 'exact offsets 0,10,28,39,48,57 from first counter; first index only; no retry',
            'anchor_input_intent': ['stationary', 'right', 'left+up', 'down', 'right', 'stationary'],
            'unknown_rules': 'low conf, raw-gradient disagreement, fewer than two supports, any confident cross-bank disagreement',
            'phase_status': 'unknown', 'real_pixel_gate_passed': False,
            'coverage': '>=12 dynamic common counters AND >=3 turns covered within +/-3 counters',
            'warning': WARNING})
        write_new(a.out/'source-sha.json', {str(f): sha(f) for f in
            [Path(__file__), Path(__file__).with_name('tactical_pixel_probe.py'),
             Path(__file__).parent/'tests'/'test_tactical_pixel_bank.py']})
        templates, anchors, slots = extract_bank(frames, trace)
        with (a.out/'bank.npz').open('xb') as f:
            np.savez_compressed(f, rgb=templates)
        write_new(a.out/'anchors.json', {'anchors': anchors, 'slots': slots})
        write_new(a.out/'freeze.json', {'schema': 'pixel_bank_freeze_v1',
            'files': {name: sha(a.out/name) for name in ['bank-config.json', 'prereg.json', 'source-sha.json', 'bank.npz', 'anchors.json']},
            'development_inputs': {str(f): sha(f) for f in [a.frames, a.trace]},
            'real_pixel_gate_passed': False})
        print(json.dumps({'frozen': True, 'templates': len(templates), 'slots': slots}))
    else:
        freeze = json.loads((a.bank_dir/'freeze.json').read_text(encoding='utf-8'))
        for name, digest in freeze['files'].items():
            if sha(a.bank_dir/name) != digest:
                raise ValueError('frozen artifact hash mismatch: '+name)
        sources = json.loads((a.bank_dir/'source-sha.json').read_text(encoding='utf-8'))
        if any(sha(path) != digest for path, digest in sources.items()):
            raise ValueError('frozen source hash mismatch')
        c = json.loads((a.bank_dir/'bank-config.json').read_text(encoding='utf-8'))
        with np.load(a.bank_dir/'bank.npz', allow_pickle=False) as z:
            templates = z['rgb']
        anchors = json.loads((a.bank_dir/'anchors.json').read_text(encoding='utf-8'))['anchors']
        result = analyze_bank(frames, trace, templates, anchors, role=a.role, config=c)
        result['inputs'] = {str(f): sha(f) for f in [a.frames, a.trace]}
        result['freeze_sha256'] = sha(a.bank_dir/'freeze.json')
        write_new(a.output, result)
        print(json.dumps({k: result[k] for k in ('role', 'confident_count', 'unknown_count')}))
        print(json.dumps(result['counter_lag_diagnostic']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
