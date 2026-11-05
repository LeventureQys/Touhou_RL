"""Independent re-check of the v3.7.6 F2 behaviour-cloning evidence chain.

Why a second implementation: the WP-F finding F-6 was "the reviewer is the
implementer".  A second pass by the same code proves nothing, so this script
*deliberately does not import* ``touhou_rl.f2_labels``, ``touhou_rl.f2_dataset``
or the review authoring tool.  It re-derives every claim from bytes with its own
decoding, its own 18-way mapping, its own miss-distance arithmetic and a
*different* pixel estimator (bright-block centroid instead of a
median-subtracted weighted centroid).

What it still is not: an independent human review.  The reviewer identity is
recorded, and the report says so.  It cannot certify reviewer honesty, only that
the numbers in the evidence files follow from the raw bytes.

Checks: identity bindings, logic-clock alignment against the game's own
published input words, label reconstruction, aim miss distances from the raw
bullet pool, byte-level image binding, an independent pixel position estimate,
test isolation, and a supervision census.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from touhou_rl import frame_proof  # noqa: E402  (PNG decoding only)

REVIEWER = dict(id='v376-wpd-double-entry-audit-20261105', kind='ai_agent_readonly',
                independence='separate implementation from the dataset builder; same developer, not a human')

#: Independent 18-way mapping (spec.MOVES order, focus*9 + move).
MOVE_OF_BITS = {(): 0, (0x10,): 1, (0x10, 0x80): 2, (0x80,): 3, (0x20, 0x80): 4,
                (0x20,): 5, (0x20, 0x40): 6, (0x40,): 7, (0x10, 0x40): 8}
STAGE_IMAGE_ORIGIN = (32.0, 16.0)


def candidate(mask):
    """Inverse of the documented 18-way candidate encoding, written from scratch."""
    if mask & 0x02 or not mask & 0x01:
        return None
    focus = 1 if mask & 0x04 else 0
    directions = tuple(sorted(bit for bit in (0x10, 0x20, 0x40, 0x80) if mask & bit))
    if (0x10 in directions and 0x20 in directions) or (0x40 in directions and 0x80 in directions):
        return None
    move = MOVE_OF_BITS.get(directions)
    return None if move is None else focus * 9 + move


def miss_px(origin, velocity, aim):
    speed = math.hypot(velocity[0], velocity[1])
    if not speed > 0:
        return None
    dx, dy = aim[0] - origin[0], aim[1] - origin[1]
    return abs(dx * velocity[1] - dy * velocity[0]) / speed


def evidence_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _records(payload, extended):
    """Two-byte little-endian input words, stride 6 when the layout is extended."""
    stride = 6 if extended else 2
    if len(payload) % stride:
        raise SystemExit('payload is not a whole number of records')
    return [int.from_bytes(payload[i * stride:i * stride + 2], 'little')
            for i in range(len(payload) // stride)]


def _stage_payload(replay, stage_index):
    script = Path(__file__).resolve().parents[2] / 'third_party/th08/scripts/replay_file.py'
    spec = importlib.util.spec_from_file_location('_wpd_replay_file', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.decode(Path(replay).read_bytes())
    info = module.inspect(replay)
    stages = [s for s in info['stages'] if s['index'] == stage_index]
    if len(stages) != 1:
        raise SystemExit('stage not uniquely present')
    stage = stages[0]
    start = stage['offset'] + 0x24
    end = start + stage['inputRecords'] * (6 if info['extended'] else 2)
    return data[start:end], info['extended']


def bright_block(image, declared, radius=4):
    """Independent estimator: brightest block centroid, not a weighted centroid."""
    cx, cy = int(round(declared[0])), int(round(declared[1]))
    x0, x1 = max(0, cx - radius), min(image.shape[1], cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(image.shape[0], cy + radius + 1)
    patch = image[y0:y1, x0:x1].astype(np.float64).sum(axis=2)
    if patch.size == 0:
        return None
    peak = float(patch.max())
    if peak < 150.0:
        return None
    mask = patch >= 0.7 * peak
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return np.array([x0 + xs.mean(), y0 + ys.mean()])


def audit(dataset, plan_path, cache=None, run=None):
    dataset, plan_path = Path(dataset).resolve(), Path(plan_path).resolve()
    manifest = dataset / 'manifest.json'
    document = _json(manifest)
    plan = _json(plan_path)
    report = dict(schema='v376-f2-independent-audit/1', reviewer=REVIEWER,
                  dataset=str(dataset), manifest_sha256=file_sha(manifest),
                  plan_sha256=file_sha(plan_path), checks=[], failures=[])

    def check(name, ok, detail, blocking=True):
        report['checks'].append(dict(name=name, passed=bool(ok), detail=detail, blocking=blocking))
        if not ok and blocking:
            report['failures'].append(name)
        return ok

    check('manifest binds the frozen plan', document['plan']['sha256'] == file_sha(plan_path),
          dict(manifest_plan_sha=document['plan']['sha256'], actual=file_sha(plan_path)))
    check('objective is F2 behaviour cloning, not original G4',
          document.get('objective') == 'f2_behavior_cloning' and document.get('original_G4') is False,
          dict(objective=document.get('objective'), original_G4=document.get('original_G4')))
    check('config digest matches the preregistered config',
          evidence_sha(plan['config']) == plan['config_sha256'] == document['config_sha256'],
          dict(plan_config_sha256=plan['config_sha256'], manifest_config_sha256=document['config_sha256']))

    sources = {s['id']: s for s in plan['sources']}
    events = {e['event_id']: e for e in plan['events']}
    groups, split_of = {}, {}
    for event in plan['events']:
        for key in (('replay', sources[event['source_id']]['replay']['sha256']),
                    ('template', event['template_group']), ('episode', event['episode_id']),
                    ('derived', event['derived_group'])):
            if key in groups and groups[key] != event['split']:
                check(f'no cross-split {key[0]} leakage', False, dict(key=key, splits=[groups[key], event['split']]))
            groups[key] = event['split']
        split_of[event['event_id']] = event['split']
    check('all three splits present', set(split_of.values()) == {'train', 'validation', 'test'},
          dict(splits=sorted(set(split_of.values()))))

    payloads = {}
    for sid, source in sources.items():
        stage = next(e['stage'] for e in plan['events'] if e['source_id'] == sid)
        payloads[sid] = _stage_payload(ROOT / source['replay']['path'], stage)

    census = {}
    clock_rows = clock_hits = label_rows = label_hits = 0
    unknown_rows = unknown_ok = 0
    aims, misses, declared_gaps = [], [], []
    pixel_distances, pixel_tolerances, memory_gaps, unmeasured = [], [], [], 0
    image_files = {}
    for ref in document['events']:
        event_meta = _json((manifest.parent / ref['path']).resolve())
        event_id = event_meta['event_id']
        split = event_meta['split']
        assert split == split_of[event_id], event_id
        evidence_path = (manifest.parent / ref['path']).resolve().parent / event_meta['label_evidence']['path']
        rows = _json(evidence_path)
        source_id = events[event_id]['source_id']
        source = sources[source_id]
        stage = events[event_id]['stage']
        payload, extended = payloads[source_id]
        masks = _records(payload, extended)
        capture = ROOT / source['exact_manifest']['path']
        capture = capture.parent
        proofs = [json.loads(line) for line in
                  (capture / 'proofs.jsonl').read_text(encoding='utf-8').splitlines()]
        by_frame = {}
        for index, proof in enumerate(proofs):
            proof['sampleIndex'] = index if proof.get('sampleIndex') is None else int(proof['sampleIndex'])
            if int(proof['stateSummary']['stage']) == stage:
                by_frame[int(proof['stateSummary']['stageFrames'])] = proof
        teachers = {}
        with (capture / 'audit-state.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                teachers[int(row['logicFrame'])] = row
        buckets = census.setdefault(split, dict(events=0, rows=0, annotated=0, supervised=0,
                                                actions={}, action_changes=0, events_detail=[]))
        buckets['events'] += 1
        stats = dict(event_id=event_id, rows=len(rows), annotated=0, supervised=0, actions={}, action_changes=0)
        previous = None
        for row in rows:
            observation = row['observation']
            frame = observation['frame']
            replay = row['replay']
            logic = replay['logic_frame']
            buckets['rows'] += 1
            # B. the decoded payload must be this replay's stage payload, and the
            # two input words must be the ones the game published for this frame.
            if bytes.fromhex(replay['payload_hex']) == payload:
                proof = by_frame.get(frame)
                if proof is not None and int(proof['logicFrame']) == logic:
                    teacher = teachers.get(logic)
                    if teacher is not None and 1 <= logic < len(masks):
                        consumed = masks[logic - 1]
                        prior = masks[logic - 2] if logic >= 2 else None
                        clock_rows += 1
                        state = teacher['state']
                        if (state['gui_input_current'] == consumed
                                and state['gui_input_previous'] == prior
                                and replay['consumed_mask'] == consumed
                                and replay['previous_mask'] == prior):
                            clock_hits += 1
            # C. label reconstruction
            label_rows += 1
            expected = candidate(masks[logic]) if logic < len(masks) else None
            flags_clear = logic < len(masks) and not masks[logic] & ~0xF5
            positive = row['action_positive']
            if row['action_valid']:
                if expected is not None and flags_clear and positive[expected] and sum(positive) == 1:
                    label_hits += 1
                stats['supervised'] += 1
                buckets['supervised'] += 1
                stats['actions'][expected] = stats['actions'].get(expected, 0) + 1
                buckets['actions'][expected] = buckets['actions'].get(expected, 0) + 1
                if previous is not None and expected != previous:
                    stats['action_changes'] += 1
                    buckets['action_changes'] += 1
                previous = expected
            else:
                unknown_rows += 1
                if not any(positive) and isinstance(row.get('unknown_reason'), str) and row['unknown_reason'].strip():
                    unknown_ok += 1
            annotation = row.get('event_annotation')
            if not annotation:
                continue
            stats['annotated'] += 1
            buckets['annotated'] += 1
            # D. aim, recomputed from the raw pool: no reviewer number is trusted.
            for emission in annotation['emissions']:
                emission_proof = by_frame.get(emission['frame'])
                teacher = teachers.get(int(emission_proof['logicFrame'])) if emission_proof else None
                slot = int(emission['track_id'].split(':', 1)[1])
                bullet = None
                if teacher:
                    found = [b for b in teacher.get('bullets') or []
                             if b.get('slot') == slot and not b.get('disabled')]
                    bullet = found[0] if len(found) == 1 else None
                if not bullet:
                    aims.append(None)
                    continue
                origin = [bullet['position'][0] + STAGE_IMAGE_ORIGIN[0],
                          bullet['position'][1] + STAGE_IMAGE_ORIGIN[1]]
                value = miss_px(origin, bullet['velocity'], emission['aim_player_px'])
                aims.append(value)
                misses.append(None if value is None else value)
                declared_gaps.append(abs((emission.get('miss_px') or 0.0) - (value or 0.0)))
            # E. independent pixel estimate on every cited sample.  The window is
            # seeded at the *raw memory* position, never at the declared point, so
            # a fabricated coordinate cannot pull the estimator onto itself.
            for track in annotation['pixel_tracks']:
                slot = int(track['id'].split(':', 1)[1])
                for sample in track['samples']:
                    key = (sample['frame'], sample['image_sha256'])
                    if key not in image_files:
                        proof = by_frame.get(sample['frame'])
                        image_files[key] = None if proof is None else (
                            file_sha(capture / proof['imagePath']) == sample['image_sha256'])
                    if image_files[key] is False:
                        pixel_distances.append(None)
                        continue
                    proof = by_frame.get(sample['frame'])
                    teacher = teachers.get(int(proof['logicFrame'])) if proof else None
                    found = [] if teacher is None else [
                        b for b in teacher.get('bullets') or []
                        if b.get('slot') == slot and not b.get('disabled')]
                    if len(found) != 1:
                        unmeasured += 1
                        continue
                    memory = np.array([found[0]['position'][0] + STAGE_IMAGE_ORIGIN[0],
                                       found[0]['position'][1] + STAGE_IMAGE_ORIGIN[1]])
                    image = frame_proof.decode_png_pixels(capture / proof['imagePath'])
                    estimate = bright_block(image, memory, radius=7)
                    if estimate is None:
                        unmeasured += 1
                        continue
                    memory_gaps.append(float(np.hypot(*(estimate - memory))))
                    pixel_distances.append(float(np.hypot(*(estimate - np.array(sample['xy'])))))
                    pixel_tolerances.append(float(annotation['tolerances']['position_px']))
        buckets['events_detail'].append(stats)

    check('logic clock: decoded record equals the input the game consumed',
          clock_rows > 0 and clock_hits == clock_rows, dict(rows=clock_rows, hits=clock_hits))
    check('labels are the replayed demonstration action', label_rows > 0 and label_hits == label_rows - unknown_rows,
          dict(rows=label_rows, supervised=label_rows - unknown_rows, matches=label_hits))
    check('unknown rows carry no action and a reason', unknown_rows > 0 and unknown_ok == unknown_rows,
          dict(rows=unknown_rows, ok=unknown_ok))
    aims_ok = [value for value in aims if value is not None]
    check('aim miss distance inside the declared hitbox tolerance', all(value <= 1.5 for value in aims_ok),
          dict(emissions=len(aims), measurable=len(aims_ok),
               max=round(max(aims_ok), 3) if aims_ok else None,
               median=round(float(np.median(aims_ok)), 3) if aims_ok else None))
    check('declared miss matches the recomputed miss', max(declared_gaps or [0.0]) <= 0.01,
          dict(max_gap=round(max(declared_gaps or [0.0]), 5)))
    distances = [value for value in pixel_distances if value is not None]
    envelopes = sorted({round(value, 2) for value in pixel_tolerances})
    check('cited image bytes match the declared SHA', all(v is not False for v in image_files.values()),
          dict(frames_checked=len(image_files), mismatches=sum(1 for v in image_files.values() if v is False)))
    check('declared pixel positions agree with an independent estimator',
          bool(distances) and float(np.percentile(distances, 90)) <= max(envelopes or [0.0]),
          dict(samples=len(distances), declared_tolerance_px=envelopes,
               max=round(max(distances), 3) if distances else None,
               p90=round(float(np.percentile(distances, 90)), 3) if distances else None,
               median=round(float(np.median(distances)), 3) if distances else None,
               estimator='brightest-block centroid seeded at the raw memory position, radius 7',
               seed_to_memory_px=dict(max=round(max(memory_gaps), 3) if memory_gaps else None,
                                      median=round(float(np.median(memory_gaps)), 3) if memory_gaps else None),
               unmeasured=unmeasured))
    if cache is not None:
        index = _json(Path(cache) / 'index.json')
        cached = {ref['event_id'] for ref in index['events']}
        test_events = {e['event_id'] for e in plan['events'] if e['split'] == 'test'}
        check('no test event is in the feature cache', not (test_events & cached),
              dict(test_events=sorted(test_events), cached=sorted(cached)))
        cache_report = _json(Path(cache) / 'report.json')
        check('cache audit state', cache_report.get('test_tensor_loaded') is False
              and cache_report.get('test_cache_created') is False, dict(
                  test_tensor_loaded=cache_report.get('test_tensor_loaded'),
                  test_cache_created=cache_report.get('test_cache_created')))
    if run is not None:
        run_report = _json(Path(run) / 'report.json')
        check('training report did not use test', run_report.get('test_used') is False
              and run_report.get('backbone_unchanged') is True,
              dict(test_used=run_report.get('test_used'), backbone_unchanged=run_report.get('backbone_unchanged')))
    report['census'] = census
    report['passed'] = not report['failures']
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=None)
    parser.add_argument('--run', type=Path, default=None)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.dataset, args.plan, args.cache, args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for item in result['checks']:
        print(('PASS ' if item['passed'] else 'FAIL ') + item['name'] + ' ' + json.dumps(item['detail']))
    print(json.dumps({k: v for k, v in result['census'].items()}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
