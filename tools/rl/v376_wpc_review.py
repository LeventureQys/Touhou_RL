"""Pre-registered, exhaustive F2 review authoring (v3.7.6 WP-C/D replacement).

Why this tool exists (the previous one is kept unchanged as evidence):

* **No row cherry-picking.** ``v376_wpb_author_review`` chose the rows to annotate
  by whether their two emissions could be measured, which is a selection bias
  (WP-F finding F-5).  Here the *windows* are chosen by a frozen, memory-geometry
  rule and then **every** sample in each window is attempted in capture order.
  A row that cannot carry a contract-valid annotation is recorded with its
  reason and becomes an explicit ``context_only`` row; it is never replaced by a
  more convenient row.
* **Aim is a miss distance.** The aim frame is limited to the emission sample or
  the one sample before it (a 2-frame-sampled capture cannot resolve better),
  and the claim is ``|cross| / |v_teacher| <= AIM_MISS_PX`` -- perpendicular
  pixels, not a speed ratio (WP-F finding F-1).
* **Declared numbers are recomputed.** Each emission declares ``miss_px``; the
  contract recomputes it and ``f2_dataset._verify_teacher`` recomputes it again
  from the raw pool.

Two modes:

``select``    scan a capture with privileged geometry only and write the ranked
              window list plus the frozen selection (no pixel claim).
``annotate``  read a frozen selection and write one review JSON per window.

Privileged geometry never substitutes for the pixel side: bullet positions are
measured from the original PNG bytes.  The player point and the track ids remain
contract-bound to same-hold memory (``f2_dataset`` requires an exact match), so
only the bullet coordinates are independently measured.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'tools' / 'rl') not in sys.path:
    sys.path.insert(0, str(ROOT / 'tools' / 'rl'))

import v376_wpb_author_review as base  # noqa: E402
from touhou_rl import f2_dataset, f2_labels, frame_proof  # noqa: E402
from touhou_rl.f2_dataset import file_sha  # noqa: E402

REVIEWER_ID = 'v376-wp-c-agent-visual-20261105'
TEACHER_ID = 'same-hold-memory-audit'
#: Frozen contract tolerances.  ``aim_miss_px`` is the TH08 player hitbox radius
#: (~1.5 px, a ~3 px target): a straight bullet path inside it passes through the
#: player.  It is fixed by the target size, not by measuring the yield.
AIM_MISS_PX = 1.5
AIM_LOOKBACK_FRAMES = 2
#: A cited emission must still be part of the *current* stream: 64 stage frames
#: is about one second, which at the observed 1.2-2.8 px/frame bullet speeds puts
#: the bullet 77-180 px from the muzzle, i.e. still crossing a 384 px playfield.
#: Beyond that the bullets are a spent volley, not the stream being flown now.
EMISSION_RECENCY_FRAMES = 64
VELOCITY_TOLERANCE_PX_PER_FRAME = 2.5
MIN_POSITION_TOLERANCE_PX = 1.5
MAX_MEASUREMENT_DEVIATION = base.MAX_MEASUREMENT_DEVIATION
ORIGIN_CLUSTER_PX = base.ORIGIN_CLUSTER_PX
DEFAULT_WINDOW_ROWS = 96
DEFAULT_WINDOWS_PER_SOURCE = 3


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def strict_aim(data, first_frame, origin, velocity):
    """Aim evidence limited to the emission sample and the one before it.

    Wider past searches "find" a player position that happens to sit on a stray
    bullet's path; the two frames tried here are the only ones a 2-frame-sampled
    capture can justify (the sprite is first seen at ``first_frame`` and may have
    been spawned on the previous held frame).
    """
    best = None
    for aim_frame in (first_frame, base.previous_frame(data, first_frame)):
        if aim_frame is None:
            continue
        player = base._player(data, aim_frame)
        delta = player - origin
        miss = f2_labels.aim_miss(origin, velocity, player)
        if miss is None or delta @ velocity <= 0:
            continue
        if best is None or miss < best[0]:
            best = (miss, aim_frame, player)
    if best is None or best[0] > AIM_MISS_PX:
        return None
    return dict(miss=best[0], aim_frame=best[1], aim_player=best[2])


def memory_pairs(data, frame):
    """Every recency-bounded aimed emission pair at this row (memory only)."""
    prepared = []
    for slot, track in base.candidate_tracks(data, frame):
        first = track['samples'][0][0]
        if frame - first > EMISSION_RECENCY_FRAMES:
            continue
        velocity = np.asarray(track['samples'][0][1]['velocity'], dtype=float)
        aim = strict_aim(data, first, track['memory'][0], velocity)
        if aim is None:
            continue
        prepared.append(dict(slot=slot, track=track, aim=aim, velocity=velocity,
                             origin=track['memory'][0], first=first))
    pairs = []
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            a, b = prepared[i], prepared[j]
            if a['first'] == b['first']:
                continue
            distance = float(np.hypot(*(a['origin'] - b['origin'])))
            if distance > ORIGIN_CLUSTER_PX:
                continue
            age = (frame - a['first']) + (frame - b['first'])
            pairs.append((age, a['aim']['miss'] + b['aim']['miss'] + .01 * distance, a, b))
    pairs.sort(key=lambda item: (item[0], item[1]))
    return pairs


def row_annotated(data, frame, source_id, position_tolerance):
    """One contract-valid F2 row, or ``None`` with the reasons that blocked it."""
    prepared, reasons = [], []
    for slot, track in base.candidate_tracks(data, frame):
        first = track['samples'][0][0]
        if frame - first > EMISSION_RECENCY_FRAMES:
            reasons.append(f'slot:{slot} emitted {frame - first} frames ago (outside stream recency)')
            continue
        velocity = np.asarray(track['samples'][0][1]['velocity'], dtype=float)
        aim = strict_aim(data, first, track['memory'][0], velocity)
        if aim is None:
            reasons.append(f'slot:{slot} not aimed within {AIM_MISS_PX} px at its emission')
            continue
        segment = base.measure_segment(data, track)
        if segment is None:
            reasons.append(f'slot:{slot} not separable in pixels')
            continue
        prepared.append(dict(slot=slot, segment=segment, aim=aim, velocity=velocity,
                             origin=track['memory'][0], first=first, track=track))
    pairs = []
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            a, b = prepared[i], prepared[j]
            if a['first'] == b['first']:
                continue
            distance = float(np.hypot(*(a['origin'] - b['origin'])))
            if distance > ORIGIN_CLUSTER_PX:
                continue
            age = (frame - a['first']) + (frame - b['first'])
            pairs.append((age, a['aim']['miss'] + b['aim']['miss'] + .01 * distance, a, b))
    if not pairs:
        return None, reasons or ['no two aimed emissions from one origin cluster']
    pairs.sort(key=lambda item: (item[0], item[1]))
    _age, _score, first, second = pairs[0]
    chosen = [first, second]
    centre = (chosen[0]['origin'] + chosen[1]['origin']) / 2.0
    emitter_id = 'origin-cluster-{:.0f}-{:.0f}'.format(round(centre[0] / 10) * 10, round(centre[1] / 10) * 10)
    tracks, teacher_tracks, emissions = [], [], []
    for item in chosen:
        track_id = f"slot:{item['slot']}"
        measured = item['segment']['measured']
        tracks.append(dict(id=track_id, samples=[
            dict(frame=g, xy=[round(float(v), 3) for v in point], visible=True,
                 image_sha256=data['by_frame'][g]['imageSha256']) for g, point in measured]))
        # The teacher track must start at the emission frame (the aim geometry is
        # taken there) and cover every measured pixel sample, because the auditor
        # re-checks each teacher point against the raw pool and the contract
        # compares teacher and pixel positions frame by frame.  Frames between
        # those are unnecessary; the first three carry the velocity.
        memory_at = {g: np.asarray([b['position'][0] + base.FLAG[0], b['position'][1] + base.FLAG[1]])
                     for g, b in item['track']['samples']}
        wanted = sorted({item['first'], item['first'] + 2, item['first'] + 4}
                        | {g for g, _point in measured})
        teacher_frames = [g for g in wanted if g in memory_at]
        teacher_tracks.append(dict(id=track_id, samples=[
            dict(frame=g, xy=[float(memory_at[g][0]), float(memory_at[g][1])]) for g in teacher_frames]))
        (fa, ma), (fb, mb) = measured[0], measured[1]
        velocity = (mb - ma) / (fb - fa)
        emissions.append(dict(
            frame=item['first'], track_id=track_id, emitter_id=emitter_id,
            aim_player_frame=item['aim']['aim_frame'],
            aim_image_sha256=data['by_frame'][item['aim']['aim_frame']]['imageSha256'],
            origin_px=[round(float(v), 3) for v in measured[0][1]],
            aim_player_px=[round(float(v), 3) for v in item['aim']['aim_player']],
            velocity_px_per_frame=[round(float(v), 4) for v in velocity],
            miss_px=round(float(item['aim']['miss']), 4),
            player_pixel_annotation=dict(
                frame=item['aim']['aim_frame'],
                image_sha256=data['by_frame'][item['aim']['aim_frame']]['imageSha256'],
                xy=[round(float(v), 3) for v in item['aim']['aim_player']])))
    cited = sorted({s['frame'] for track in teacher_tracks for s in track['samples']}
                   | {g for item in chosen for g, _point in item['segment']['measured']}
                   | {item['aim']['aim_frame'] for item in chosen} | {frame})
    observation = dict(
        frame=frame, observed_through=frame, units='image_px', image_size=[640, 480],
        player_px=[float(v) for v in base._player(data, frame)],
        source=dict(capture=dict(id=source_id, sha256=data['manifest_sha']),
                    proof=dict(id='proofs.jsonl', sha256=data['rows_sha'])),
        images=[dict(frame=g, sha256=data['by_frame'][g]['imageSha256']) for g in cited],
        bullet_tracks=tracks)
    annotation = dict(
        observation_sha256=f2_labels.evidence_sha(observation), family='F2', event_type='aimed_stream',
        reviewer=dict(id=REVIEWER_ID, method='independent_pixel_annotation',
                      note='AI visual reviewer; bullet coordinates measured from the original PNG bytes'),
        teacher=dict(producer_id=TEACHER_ID, artifact_sha256=data['teacher_sha'], tracks=teacher_tracks),
        tolerances=dict(position_px=position_tolerance,
                        aim_cross_px_per_frame=VELOCITY_TOLERANCE_PX_PER_FRAME,
                        aim_miss_px=AIM_MISS_PX, aim_lookback_frames=AIM_LOOKBACK_FRAMES,
                        emission_recency_frames=EMISSION_RECENCY_FRAMES),
        pixel_tracks=tracks, emissions=emissions)
    errors = f2_labels.validate_event_annotation(annotation, observation)
    if errors:
        return None, errors
    deviations = [value for item in chosen for value in item['segment']['deviations']]
    row = dict(status='reviewed_f2', unknown_reason=None, observation=observation, annotation=annotation)
    return (row, deviations), None


def author_window(capture, stage, source_id, event_id, start, rows, out, replay=None, min_rows=48):
    """Attempt every sample of the window in capture order; record what fails."""
    data = base.load_capture(capture, stage, replay)
    indices = list(range(start, start + rows))
    if min(indices) < 6 or max(indices) >= len(data['proofs']):
        raise SystemExit('window outside the capture')
    frames = [int(data['proofs'][index]['stateSummary']['stageFrames']) for index in indices]
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise SystemExit('non-increasing stage frames in the window')
    # The frozen feature config needs the exact history offsets -4/-2/0, and the
    # capture is not guaranteed contiguous (a stage clock can skip a frame).  A
    # sample whose history frames are absent cannot become a row at all, so it is
    # excluded here and listed with its reason; this is a structural rule that
    # never looks at pixels, annotations or actions.
    structural = [frame for frame in frames
                  if not all(g in data['by_frame'] for g in (frame - 4, frame - 2))]
    frames = [frame for frame in frames if frame not in set(structural)]
    # Two passes so the declared position tolerance is the measured maximum, and
    # the rows are then re-validated against it (never widened per row).
    provisional = MAX_MEASUREMENT_DEVIATION + 1.
    first_pass = []
    for frame in frames:
        result, reason = row_annotated(data, frame, source_id, provisional)
        first_pass.append((frame, result, reason))
    deviations = [value for _frame, result, _reason in first_pass if result for value in result[1]]
    position_tolerance = round(max([MIN_POSITION_TOLERANCE_PX] + [d + 1.0 for d in deviations]), 2)
    dropped = [dict(stage_frame=stage_frame,
                    reason='capture gap: required history frames (-4/-2) absent')
               for stage_frame in structural]
    kept = []
    annotated = context = 0
    for frame, result, reason in first_pass:
        if result is not None:
            row, _values = result
            row['annotation']['tolerances']['position_px'] = position_tolerance
            errors = f2_labels.validate_event_annotation(row['annotation'], row['observation'])
            if errors:
                dropped.append(dict(stage_frame=frame, reason='; '.join(errors)))
                continue
            kept.append(row)
            annotated += 1
            continue
        try:
            kept.append(base.row_context(data, frame, source_id))
            context += 1
        except SystemExit as exc:
            dropped.append(dict(stage_frame=frame, reason=str(exc) + ': ' + '; '.join(reason[:4])))
    if len(kept) < min_rows:
        raise SystemExit(f'window {event_id}: only {len(kept)} usable rows after annotation/drop')
    review = dict(
        schema='v376-f2-review/2', frozen=True, source_id=source_id, event_id=event_id, stage=stage,
        capture=Path(capture).as_posix(),
        capture_manifest_sha256=data['manifest_sha'], proof_rows_sha256=data['rows_sha'],
        teacher_artifact_sha256=data['teacher_sha'],
        selection=dict(rule='frozen window list; every sample in the window attempted in capture order',
                       window_start_sample=start, window_rows=rows, window_samples=len(indices),
                       attempted_after_structural=len(frames),
                       structural_rule='a sample whose -4/-2 history frames are absent from the capture cannot become a row'),
        reviewer=dict(id=REVIEWER_ID, kind='ai_agent_visual', method='independent_pixel_annotation',
                      disclaimer='bullet coordinates measured from original PNG pixels; player point and track ids are contract-bound to same-hold memory; independent re-check still required'),
        tolerances=dict(position_px=position_tolerance,
                        aim_cross_px_per_frame=VELOCITY_TOLERANCE_PX_PER_FRAME,
                        aim_miss_px=AIM_MISS_PX, aim_lookback_frames=AIM_LOOKBACK_FRAMES,
                        emission_recency_frames=EMISSION_RECENCY_FRAMES),
        measured_deviation_px=dict(max=round(max(deviations), 3) if deviations else None,
                                   median=round(float(np.median(deviations)), 3) if deviations else None,
                                   samples=len(deviations)),
        yield_=dict(annotated=annotated, context_only=context, dropped=len(dropped), kept=len(kept),
                    structural_drops=len(structural)),
        drops=dropped, rows=kept)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(review, stream, indent=1)
    return dict(out=str(out), rows=len(kept), annotated=annotated, context=context,
                dropped=len(dropped), position_tolerance=position_tolerance,
                sha256=file_sha(out), max_deviation=review['measured_deviation_px']['max'])


def scan_capture(capture, stage, replay=None):
    """Per-sample aimed-pair flag and active-bullet count (privileged, no pixels)."""
    data = base.load_capture(capture, stage, replay)
    frame_of = [int(proof['stateSummary']['stageFrames']) for proof in data['proofs']]
    flags, active = [], []
    for frame in frame_of:
        pairs = memory_pairs(data, frame)
        flags.append(bool(pairs))
        active.append(len(data['bullets'].get(int(data['by_frame'][frame]['sampleIndex']), ())))
    return dict(data=dict(manifest_sha=data['manifest_sha'], rows_sha=data['rows_sha'],
                          teacher_sha=data['teacher_sha']),
                frames=frame_of, aimed_pair=flags, active_bullets=active)


def select(capture, stage, source_id, rows, count, replay=None):
    """Rank windows by frozen rule: most aimed-pair samples, then most live rows."""
    scan = scan_capture(capture, stage, replay)
    flags, active = scan['aimed_pair'], scan['active_bullets']
    total = len(flags)
    scored = []
    for start in range(0, total - rows + 1):
        window = range(start, start + rows)
        aimed = sum(1 for i in window if flags[i])
        live = sum(1 for i in window if active[i] > 0)
        scored.append((aimed, live, -start, start))
    scored.sort(reverse=True)
    chosen, taken = [], []
    for aimed, live, negative_start, start in scored:
        if len(chosen) >= count:
            break
        if any(start < other + rows and other < start + rows for other in taken):
            continue
        taken.append(start)
        chosen.append(dict(source_id=source_id, start_sample=start,
                           stage_frames=[scan['frames'][start], scan['frames'][start + rows - 1]],
                           aimed_pair_rows=aimed, live_rows=live, rows=rows))
    return dict(source_id=source_id, capture=Path(capture).as_posix(), stage=stage,
                samples=total, window_rows=rows, windows=chosen,
                rule=('window score = (rows where a strict aimed pair exists, rows with a live bullet, -start); '
                      'greedy non-overlapping top-k; computed from the privileged bullet pool before any pixel work'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    pick = sub.add_parser('select')
    pick.add_argument('--capture', required=True)
    pick.add_argument('--source-id', required=True)
    pick.add_argument('--stage', type=int, required=True)
    pick.add_argument('--rows', type=int, default=DEFAULT_WINDOW_ROWS)
    pick.add_argument('--count', type=int, default=DEFAULT_WINDOWS_PER_SOURCE)
    pick.add_argument('--replay', default=None)
    pick.add_argument('--out', type=Path, required=True)
    draw = sub.add_parser('annotate')
    draw.add_argument('--capture', required=True)
    draw.add_argument('--source-id', required=True)
    draw.add_argument('--event-id', required=True)
    draw.add_argument('--stage', type=int, required=True)
    draw.add_argument('--start', type=int, required=True)
    draw.add_argument('--rows', type=int, default=DEFAULT_WINDOW_ROWS)
    draw.add_argument('--replay', default=None)
    draw.add_argument('--out', type=Path, required=True)
    draw.add_argument('--min-rows', type=int, default=48,
                      help='smoke-testing guard only; the dataset contract always demands 48')
    args = parser.parse_args(argv)
    if args.mode == 'select':
        result = select(args.capture, args.stage, args.source_id, args.rows, args.count, args.replay)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    else:
        result = author_window(args.capture, args.stage, args.source_id, args.event_id,
                               args.start, args.rows, args.out, args.replay, args.min_rows)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
