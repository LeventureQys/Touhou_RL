"""Offline set-valued player geometry diagnostic; never Present/phase proof.

freeze reads development bank results ONLY. analyze consumes already-computed
confirmation localizations, never fits an offset and never re-runs image search.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .tactical_pixel_probe import _position, _stable

RULES = {
    'window_radius': 8, 'residual_budget_px': 2.0,
    'pixel_duplicate_diameter_px': 1.0, 'motion_threshold_px': .25,
    'turn_radius': 3, 'inflation_budgets_px': [2., 4., 8.],
    'offset_selection': 'development counter_lag_diagnostic score lag=0 anchor_offset; no refit',
    'memory': 'exact pre/post counters; any nonfinite or unequal duplicate position invalid; no interpolation',
    'pixel': 'each confident q separately; pairwise conflicting stable duplicate counter invalidates all',
    'unknown': 'unstable bracket, conflicting pixel duplicate, or ANY missing/conflicting memory in full +/-8 window',
    'candidate': 'all exact memory frames with Euclidean norm(q-offset-memory)<=2px, inclusive',
    'relative_error': 'candidate memory minus reference-counter memory; component and radial hull, not continuous support',
    'future_label': 'diagnostic only: complete nonempty set and maximum reference-relative displacement <= fixed inflation budget',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def freeze(development, output):
    if Path(output).exists():
        raise FileExistsError('refusing to overwrite prereg')
    dev = json.loads(Path(development).read_text(encoding='utf-8'))
    if dev['role'] != 'development':
        raise ValueError('development result required; confirmation cannot fit offset')
    scores = [s for s in dev['counter_lag_diagnostic']['scores'] if s['lag'] == 0]
    if len(scores) != 1 or not np.isfinite(scores[0]['anchor_offset']).all():
        raise ValueError('unique finite development lag=0 anchor required')
    value = {'schema': 'alignment_intervals_prereg_v1', 'rules': RULES,
             'anchor_offset': scores[0]['anchor_offset'],
             'development_input': str(development), 'development_sha256': sha(development),
             'source_sha256': sha(__file__),
             'probe_source_sha256': sha(Path(__file__).with_name('tactical_pixel_probe.py')),
             'disclosure': 'round6 confirmation bank aggregates already exposed; not a new independent blind confirmation; new clips required for future rule validation',
             'phase_verified': False, 'real_pixel_gate_passed': False}
    write_new(output, value)
    return value


def memory_map(trace):
    values, invalid = {}, set()
    for r in trace:
        for side in ('pre', 'post'):
            s = r[side]
            f, xy = int(s['gameplay_frame']), _position(s)
            if not np.isfinite(xy).all():
                invalid.add(f)
            else:
                values.setdefault(f, set()).add(tuple(xy))
    conflicts = {f for f, v in values.items() if len(v) != 1} | invalid
    return {f: np.array(next(iter(v))) for f, v in values.items() if f not in conflicts}, conflicts


def hull(values):
    return [float(min(values)), float(max(values))] if len(values) else None


def summarize(rows):
    n = len(rows)
    counts = {s: sum(r['status'] == s for r in rows) for s in
              ('unknown', 'complete_empty', 'complete_nonempty')}
    known = [r for r in rows if r['status'] == 'complete_nonempty']
    counts.update(all_candidates_within_one=sum(r['all_candidates_within_one'] is True for r in known),
                  some_candidates_outside_one=sum(r['all_candidates_within_one'] is False for r in known),
                  multi_frame=sum(len(r['candidate_frames']) > 1 for r in known))
    return {'denominator': n, 'counts': counts,
            'fractions': {k: v/n if n else None for k, v in counts.items()},
            'inflation_diagnostic': {str(b): {
                'count': sum(r['max_relative_displacement_px'] <= b for r in known),
                'fraction_of_stratum': sum(r['max_relative_displacement_px'] <= b for r in known)/n if n else None,
                'fraction_of_complete_nonempty': sum(r['max_relative_displacement_px'] <= b for r in known)/len(known) if known else None}
                for b in RULES['inflation_budgets_px']}}


def analyze(trace, bank, prereg):
    if prereg['rules'] != RULES:
        raise ValueError('frozen rules mismatch; no confirmation tuning')
    offset = np.asarray(prereg['anchor_offset'], dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError('invalid frozen offset')
    if bank.get('role') != 'confirmation':
        raise ValueError('confirmation bank result required')
    by_index = {int(r['index']): r for r in trace}
    locs = bank['localizations']
    if len(by_index) != len(trace) or len({int(q['index']) for q in locs}) != len(locs) or set(by_index) != {int(q['index']) for q in locs}:
        raise ValueError('trace/localization indices must be unique and exactly matched')
    memory, memory_conflicts = memory_map(trace)
    stable_pixels = {}
    for q in locs:
        r = by_index[int(q['index'])]
        if q['confident'] and _stable(r):
            stable_pixels.setdefault(int(r['pre']['gameplay_frame']), []).append(np.asarray(q['xy'], dtype=float))
    pixel_conflicts = {f for f, qs in stable_pixels.items() if any(
        not np.isfinite(a).all() or not np.isfinite(b).all() or
        np.linalg.norm(a-b) > RULES['pixel_duplicate_diameter_px'] for a in qs for b in qs)}
    directions = {}
    for r in sorted(trace, key=lambda r: int(r['index'])):
        directions.setdefault(int(r['pre']['gameplay_frame']), tuple(sorted(k for k in r.get('keys', []) if k in ('left','right','up','down'))))
    turns, previous = [], None
    for f, direction in sorted(directions.items()):
        if direction:
            if previous is not None and previous != direction:
                turns.append(f)
            previous = direction
    rows = []
    for q in locs:
        if not q['confident']:
            continue
        r = by_index[int(q['index'])]
        f, xy = int(r['pre']['gameplay_frame']), np.asarray(q['xy'], dtype=float)
        frames = list(range(f-RULES['window_radius'], f+RULES['window_radius']+1))
        missing = [g for g in frames if g not in memory and g not in memory_conflicts]
        conflicts = [g for g in frames if g in memory_conflicts]
        reasons = []
        if not _stable(r): reasons.append('unstable_capture_bracket')
        if f in pixel_conflicts: reasons.append('conflicting_pixel_duplicate_counter')
        if not np.isfinite(xy).all(): reasons.append('nonfinite_pixel')
        if missing: reasons.append('missing_exact_memory_frames')
        if conflicts: reasons.append('conflicting_or_nonfinite_memory_frames')
        comparisons = []
        for g in frames:
            if g not in memory or not np.isfinite(xy).all():
                continue
            error = xy-offset-memory[g]
            residual = float(np.linalg.norm(error))
            relative = memory[g]-memory[f] if f in memory else None
            comparisons.append({'frame': g, 'relative_frame': g-f, 'memory_xy': memory[g].tolist(),
                'pixel_minus_memory_residual_xy': error.tolist(), 'residual_px': residual,
                'within_residual_budget': residual <= RULES['residual_budget_px'],
                'relative_memory_error_xy': relative.tolist() if relative is not None else None,
                'relative_memory_error_px': float(np.linalg.norm(relative)) if relative is not None else None})
        candidates = [c for c in comparisons if c['within_residual_budget']]
        status = 'unknown' if reasons else 'complete_nonempty' if candidates else 'complete_empty'
        # Partial observed candidates are exposed, never claimed as exhaustive when unknown.
        complete = status == 'complete_nonempty'
        relative_candidates = [c for c in candidates if c['relative_memory_error_xy'] is not None]
        rows.append({'index': int(q['index']), 'reference_frame': f, 'q': xy.tolist() if np.isfinite(xy).all() else None,
            'status': status, 'unknown_reasons': reasons, 'candidate_set_exhaustive_in_window': not reasons,
            'missing_frames': missing, 'conflicting_memory_frames': conflicts,
            'candidate_frames': [c['frame'] for c in candidates],
            'candidate_relative_frames': [c['relative_frame'] for c in candidates],
            'relative_frame_interval_hull': hull([c['relative_frame'] for c in candidates]),
            'candidate_residual_interval_px': hull([c['residual_px'] for c in candidates]),
            'relative_memory_error_interval_xy': [hull([c['relative_memory_error_xy'][j] for c in relative_candidates]) for j in range(2)],
            'relative_memory_error_interval_px': hull([c['relative_memory_error_px'] for c in relative_candidates]),
            'max_relative_displacement_px': max(c['relative_memory_error_px'] for c in candidates) if complete else None,
            'all_candidates_within_one': all(abs(c['relative_frame']) <= 1 for c in candidates) if complete else None,
            'dynamic': bool(np.linalg.norm(memory[f]-memory[f-1]) > RULES['motion_threshold_px']) if f in memory and f-1 in memory else None,
            'near_turn': any(abs(f-t) <= RULES['turn_radius'] for t in turns),
            'comparisons': comparisons})
    strata = {'all_confident': rows,
              'dynamic': [r for r in rows if r['dynamic'] is True],
              'static': [r for r in rows if r['dynamic'] is False],
              'motion_unknown': [r for r in rows if r['dynamic'] is None],
              'dynamic_near_turn': [r for r in rows if r['dynamic'] is True and r['near_turn']],
              'near_turn_all': [r for r in rows if r['near_turn']]}
    low = sum(not q['confident'] for q in locs)
    return {'schema': 'alignment_intervals_v1', 'status': 'offline_player_geometry_diagnostic_only',
        'phase_status': 'unknown_no_Present_fence', 'phase_verified': False, 'real_pixel_gate_passed': False,
        'bulletpool_same_frame_proven': False, 'anchor_offset_frozen': offset.tolist(), 'rules': RULES,
        'disclosure': prereg.get('disclosure'),
        'relative_frame_convention': 'candidate memory frame minus stable pre/post reference counter; negative means earlier memory',
        'window_limit': 'sets only cover +/-8; no assertion that physical display frame lies in this window',
        'interval_warning': 'hulls summarize discrete observed candidate sets; not interpolated positions, probability intervals, or guaranteed physical bounds',
        'capture_count': len(locs), 'confident_count': len(rows), 'low_confidence_unknown_count': low,
        'all_capture_unknown_count': low + sum(r['status'] == 'unknown' for r in rows),
        'all_capture_unknown_fraction': (low + sum(r['status'] == 'unknown' for r in rows))/len(locs) if locs else None,
        'memory_conflicting_counters': sorted(memory_conflicts), 'pixel_conflicting_counters': sorted(pixel_conflicts),
        'turn_counters': turns, 'summaries': {name: summarize(group) for name, group in strata.items()}, 'rows': rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--development', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('analyze')
    for name in ('prereg', 'bank', 'trace', 'output'):
        a.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('refusing to overwrite output')
    if args.command == 'freeze':
        result = freeze(args.development, args.output)
        print(json.dumps({'frozen': True, 'anchor_offset': result['anchor_offset']}))
    else:
        prereg = json.loads(args.prereg.read_text(encoding='utf-8'))
        if prereg['source_sha256'] != sha(__file__) or prereg['probe_source_sha256'] != sha(Path(__file__).with_name('tactical_pixel_probe.py')) or prereg['development_sha256'] != sha(prereg['development_input']):
            raise ValueError('frozen source/development SHA mismatch')
        bank = json.loads(args.bank.read_text(encoding='utf-8'))
        trace = json.loads(args.trace.read_text(encoding='utf-8'))
        # Verify that this trace is exactly one of the old bank's hashed inputs.
        if sha(args.trace) not in bank.get('inputs', {}).values():
            raise ValueError('trace does not match bank input hashes')
        result = analyze(trace, bank, prereg)
        result['inputs_sha256'] = {str(path): sha(path) for path in (args.prereg, args.bank, args.trace)}
        write_new(args.output, result)
        print(json.dumps({'summaries': result['summaries'], 'all_capture_unknown_fraction': result['all_capture_unknown_fraction']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
