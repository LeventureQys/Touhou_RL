"""Round21: one intervention on top of round20, the F1 danger telegraph; research only.

Round20 measured that every single F1 death of all six distilled heads happens exactly
on `wall_frame`, and the follow-up visibility analysis found the cause: before
`wall_frame` the F1 coverage band and its gap are not rendered at all, so the only
difference between a mirror pair on screen is where the player herself stands. The
policy is asked to park inside a 12-20 px slot it has never seen.

This round adds exactly one thing: a second renderer that draws the same coverage band
(with the same internal gap) from `wall_frame-45` as a *preview* (desaturated fill plus
explicit gap-edge markers). `wall_frame` stays the only lethal frame, `advance` and every
world parameter are untouched, the teacher is rerun unchanged and must reproduce the
frozen round20 action trajectory, the labels and the pixel packing are round20's, and the
training protocol is round20's (A = no anchor, B = 1.0*KL(base||policy), seeds 7/17/27,
200 updates, lr 1e-3, clip 1, batch 1, FP32, both TF32 flags off). No formal gate, no
deployable checkpoint, no game execution, no tuning, no rerun.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from tools.rl import v373_seed_confirm as r16
from tools.rl import v373_valid_curriculum as r20
from touhou_rl import vision
from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_feature_cache import _fp32_mode
from touhou_rl.tactical_synthetic_closed_loop import HORIZONTAL, pixel_stack
from touhou_rl.tactical_synthetic_pixels import (ACTION_MAP, SIZES, full_states,
                                                render_state)
from touhou_rl.tactical_scenarios import initial_state
from touhou_rl.tactical_teacher_v2 import hashes as teacher_hashes
from touhou_rl.tactical_teacher_v2 import rollout as teacher_rollout
from touhou_rl.tactical_teacher_validation import advance, make_world, metrics, verify
from touhou_rl.vmodels import load_vision_checkpoint

ROOT = Path('Document/Update/Alpha/v3.7.3/evidence')
OUT = ROOT / 'telegraph-round21'
R20_EVIDENCE = ROOT / 'validcurriculum-round20'
REPORTS = Path('Document/Update/Alpha/v3.7.3/reports')
BASELINE = r20.BASELINE

PREREG_SCHEMA = 'v373-telegraph-prereg/1'
DATA_SCHEMA = 'v373-telegraph-data/1'
EVENT_SCHEMA = 'v373-telegraph-event/1'
RESULT_SCHEMA = 'v373-telegraph-result/1'

# ---------------------------------------------------------------- the intervention
# `render_state` paints its own picture; this renderer only adds the preview band for F1
# inside [wall_frame-TELEGRAPH_LEAD, wall_frame). Everything below is frozen before any
# training by tools.rl.v373_telegraph.preregister, and the report quotes these numbers.
PLAYFIELD_BOX = (32, 16, 384, 448)          # the playfield rectangle `render_state` fills
PLAYFIELD_BACKGROUND = (10, 14, 24)         # that rectangle's fill colour, verbatim
LETHAL_FILL = (95, 28, 42)                  # the colour `render_state` uses while lethal
TELEGRAPH_LEAD = 45                         # the single design number of this round
TELEGRAPH_FILL = (44, 58, 74)               # desaturated blue-gray preview band
TELEGRAPH_EDGE = (96, 148, 128)             # desaturated green gap-boundary markers
EDGE_WIDTH = 4                              # native px, wide enough to survive the /1.714 resize
PREVIEW_CLIP = (0, 383)                     # `render_state` clips the band to these local columns
DECISION_FRAMES = tuple(range(0, r20.DECISIONS * 2, 2))

CONDITION = r20.CONDITION
SEEDS = r20.SEEDS
ARMS = r20.ARMS
FAMILIES = r20.FAMILIES
EXPECTED = dict(events=28, groups=14, train_events=14, validation_events=14,
                train_events_per_family={'F1': 6, 'F2': 2, 'F3': 6},
                validation_events_per_family={'F1': 6, 'F2': 2, 'F3': 6},
                checkpoints=6, closed_loop_files=2 + 2 * len(SEEDS) * len(ARMS),
                baseline_rows=2 * 14 * 6, supervised_events=14, excluded_events=0)

# Round20's recorded numbers, re-read and re-asserted from its frozen summary before this
# round starts. They are the only comparison baseline of round21 and must never be typed
# in by hand without a machine check.
R20_F1_VALIDATION_FEASIBLE = {'A': [3, 3, 3], 'B': [2, 3, 3]}
R20_ALL_VALIDATION_FEASIBLE = {'A': [4, 3, 4], 'B': [2, 3, 3]}
R20_F1_VALIDATION_FIRSTHIT = {
    'A-s7': {'wf18-F1-005-m0': None, 'wf18-F1-005-m1': 48, 'wf18-F1-007-m0': None,
             'wf18-F1-007-m1': None, 'wf18-F1-010-m0': 56, 'wf18-F1-010-m1': 56},
    'A-s17': {'wf18-F1-005-m0': None, 'wf18-F1-005-m1': 48, 'wf18-F1-007-m0': None,
              'wf18-F1-007-m1': None, 'wf18-F1-010-m0': 56, 'wf18-F1-010-m1': 56},
    'A-s27': {'wf18-F1-005-m0': None, 'wf18-F1-005-m1': 48, 'wf18-F1-007-m0': None,
              'wf18-F1-007-m1': None, 'wf18-F1-010-m0': 56, 'wf18-F1-010-m1': 56},
    'B-s7': {'wf18-F1-005-m0': None, 'wf18-F1-005-m1': 48, 'wf18-F1-007-m0': 60,
             'wf18-F1-007-m1': 60, 'wf18-F1-010-m0': None, 'wf18-F1-010-m1': 56},
    'B-s17': {'wf18-F1-005-m0': None, 'wf18-F1-005-m1': 48, 'wf18-F1-007-m0': 60,
              'wf18-F1-007-m1': None, 'wf18-F1-010-m0': None, 'wf18-F1-010-m1': 56},
    'B-s27': {'wf18-F1-005-m0': None, 'wf18-F1-005-m1': 48, 'wf18-F1-007-m0': 60,
              'wf18-F1-007-m1': None, 'wf18-F1-010-m0': None, 'wf18-F1-010-m1': 56}}

PROBE_ACTIONS = ((0, 'hold'), (2, 'right'), (-2, 'left'))


# ---------------------------------------------------------------- the renderer
def telegraph_active(s, frame, lead=TELEGRAPH_LEAD):
    """True exactly on [wall_frame-lead, wall_frame) for F1; the lethal window is separate."""
    return s.family == 'F1' and s.wall_frame - lead <= frame < s.wall_frame


def telegraph_window(s, lead=TELEGRAPH_LEAD):
    """The preview interval as half-open logical frames, or None outside F1."""
    if s.family != 'F1':
        return None
    return (s.wall_frame - lead, s.wall_frame)


def telegraph_spans(s):
    """The screen-x boxes the wall will occupy, computed exactly as `render_state` does.

    Same arithmetic, same mirror, same clip; the only reason this is a function is that
    the preview needs the geometry one window earlier than the lethal renderer draws it.
    """
    if s.family != 'F1':
        return []
    lo, hi = s.coverage
    spans = [(lo, hi)] if not s.internal_gap else [(lo, min(hi, s.gap[0])), (max(lo, s.gap[1]), hi)]
    out = []
    for lo, hi in spans:
        if lo > hi:
            continue
        if s.mirror:
            lo, hi = 384 - hi, 384 - lo
        out.append((max(PREVIEW_CLIP[0], int(lo)), min(PREVIEW_CLIP[1], int(hi))))
    return [span for span in out if span[0] <= span[1]]


def telegraph_gap(s):
    """The screen-x interval of the safe slot, or None when the wall has no gap."""
    if s.family != 'F1' or not s.internal_gap:
        return None
    lo, hi = s.gap
    if s.mirror:
        lo, hi = 384 - hi, 384 - lo
    return (int(lo), int(hi))


def canonical_gap(s):
    """The safe slot in canonical x, i.e. the coordinate `canonical_x` is compared against."""
    if s.family != 'F1' or not s.internal_gap:
        return None
    return (float(s.gap[0]), float(s.gap[1]))


def preview_columns(s):
    """Per playfield column (0..383): 0 untouched, 1 preview band, 2 gap edge marker."""
    columns = np.zeros(PLAYFIELD_BOX[2], dtype=np.uint8)
    for lo, hi in telegraph_spans(s):
        columns[lo:hi + 1] = 1
    gap = telegraph_gap(s)
    if gap is not None:
        for a, b in ((gap[0] - EDGE_WIDTH, gap[0] - 1), (gap[1] + 1, gap[1] + EDGE_WIDTH)):
            a, b = max(0, a), min(PLAYFIELD_BOX[2] - 1, b)
            if a <= b:
                columns[a:b + 1] = 2
    return columns


def render_state_telegraph(s, state):
    """`render_state` plus the F1 preview band; byte-identical everywhere else.

    The preview is painted only where the pixel is still the playfield background, so a
    bullet, the player sprite or any other drawn object is never covered or altered: the
    preview sits *behind* everything that is actually happening. No teacher target, option,
    countdown, clock, future action or hit state is drawn, and nothing outside the preview
    interval changes.
    """
    image = render_state(s, state)
    if not telegraph_active(s, state.frame):
        return image
    columns = preview_columns(s)
    selected = np.nonzero(columns)[0]
    if not selected.size:
        return image
    out = image.copy()
    left, top, width, height = PLAYFIELD_BOX
    region = out[top:top + height, left:left + width, :]
    paint = columns[selected]
    sub = region[:, selected, :]
    background = np.all(sub == np.array(PLAYFIELD_BACKGROUND, dtype=np.uint8), axis=-1)
    rows, cols = np.nonzero(background)
    if rows.size:
        colours = np.array([TELEGRAPH_FILL, TELEGRAPH_EDGE], dtype=np.uint8)
        sub[rows, cols, :] = colours[paint[cols] - 1]
        # `region[:, selected, :]` is a copy, so the painted block goes back explicitly.
        region[:, selected, :] = sub
    return out


# ---------------------------------------------------------------- frozen digests
def frame_digest(frames):
    """Order-explicit digest over native uint8 frames (no compression, no numpy header)."""
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def diff_digest(diff):
    """Digest over a (frames, H, W) boolean difference mask, plus its per-frame counts."""
    diff = np.asarray(diff, dtype=bool)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(diff.shape)).encode())
    digest.update(np.packbits(diff).tobytes())
    counts = [int(value) for value in diff.reshape(diff.shape[0], -1).sum(-1)]
    digest.update(json.dumps(counts).encode())
    return digest.hexdigest(), counts


def actions_digest(actions):
    return hashlib.sha256(json.dumps(list(actions)).encode()).hexdigest()


def render_freeze(events):
    """Exactly what the intervention changes on the given events, as frozen digests.

    Pure and read-only: one unmodified teacher rollout per event, then the same 80
    decision frames rendered by both renderers. It never trains, never reads a label and
    never touches the validation split's pixels.
    """
    rows = []
    for event in events:
        world = make_world(event['scenario'])
        actions, _ = teacher_rollout(world)
        if len(actions) != r20.DECISIONS:
            raise ValueError('incomplete teacher rollout: ' + event['event_id'])
        if verify(world, actions) != 'feasible':
            raise ValueError('the frozen teacher no longer verifies: ' + event['event_id'])
        states = full_states(world, actions)
        original = np.stack([render_state(world, states[f]) for f in DECISION_FRAMES])
        telegraph = np.stack([render_state_telegraph(world, states[f]) for f in DECISION_FRAMES])
        diff = np.any(original != telegraph, axis=-1)
        digest, counts = diff_digest(diff)
        changed = [int(DECISION_FRAMES[i]) for i, count in enumerate(counts) if count]
        preview = telegraph_window(world)
        rows.append(dict(
            event_id=event['event_id'], family=event['family'], mirror=bool(event['mirror']),
            actions_sha256=actions_digest(actions),
            original_sha256=frame_digest(original), telegraph_sha256=frame_digest(telegraph),
            diff_sha256=digest, changed_decision_frames=len(changed),
            changed_pixels=int(diff.sum()), first_changed_frame=changed[0] if changed else None,
            last_changed_frame=changed[-1] if changed else None,
            preview=[max(0, preview[0]), preview[1]] if preview else None,
            spans=[list(span) for span in telegraph_spans(world)],
            gap=list(telegraph_gap(world)) if telegraph_gap(world) else None,
            columns_sha256=hashlib.sha256(preview_columns(world).tobytes()).hexdigest()))
    value = dict(frames=list(DECISION_FRAMES), lead=TELEGRAPH_LEAD, fill=list(TELEGRAPH_FILL),
                 edge=list(TELEGRAPH_EDGE), edge_width=EDGE_WIDTH, events=rows)
    value['sha256'] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return value


def physics_probe(events):
    """Digest of the real `advance` numeric trace under three constant action sequences.

    The telegraph is a renderer; this table exists so that "the physics did not move" is
    a checked number of this round instead of an assertion about intent.
    """
    digest = hashlib.sha256()
    rows = []
    for event in events:
        world = make_world(event['scenario'])
        for action, name in PROBE_ACTIONS:
            state = initial_state(world)
            trace = []
            for _ in range(r20.DECISIONS):
                state = advance(world, state, action)
                trace.append([state.frame, round(float(state.x), 6), bool(state.hit)])
            digest.update(json.dumps([event['event_id'], name, trace]).encode())
            rows.append(dict(event_id=event['event_id'], probe=name,
                             first_hit=next((frame for frame, x, hit in trace if hit), None),
                             final_x=round(float(state.x), 6)))
    return dict(sha256=digest.hexdigest(), probes=rows,
                actions='constant hold (0), constant right (+2), constant left (-2), 80 decisions each',
                executor='touhou_rl.tactical_teacher_validation.advance, unmodified')


def r20_reference():
    """Re-read round20's frozen results and re-assert every number this round quotes."""
    path = R20_EVIDENCE / 'run/summary.json'
    summary = json.loads(path.read_text(encoding='utf-8'))
    recorded = {arm: [summary['validation']['results'][r20.name_of(arm, seed)]['F1']['feasible']
                      for seed in SEEDS] for arm in ARMS}
    if recorded != R20_F1_VALIDATION_FEASIBLE:
        raise ValueError('round20 F1 validation reference drift')
    total = {arm: [summary['validation']['results'][r20.name_of(arm, seed)]['all']['feasible']
                   for seed in SEEDS] for arm in ARMS}
    if total != R20_ALL_VALIDATION_FEASIBLE:
        raise ValueError('round20 total validation reference drift')
    firsthit = {name: summary['validation']['per_event_firsthit'][name] for name in r20.all_names()}
    for name, value in R20_F1_VALIDATION_FIRSTHIT.items():
        subset = {event: firsthit[name][event] for event in value}
        if subset != value:
            raise ValueError('round20 F1 first-hit reference drift: ' + name)
    if summary['validation']['holdout']['filter_valid'] is not True:
        raise ValueError('round20 holdout filter reference drift')
    if summary['validation']['b0']['families']['F1']['feasible'] != 0:
        raise ValueError('round20 B0 F1 reference drift')
    checkpoints = json.loads((R20_EVIDENCE / 'run/checkpoints.json').read_text(encoding='utf-8'))
    return dict(summary_sha256=file_sha(path), checkpoints_sha256=file_sha(R20_EVIDENCE / 'run/checkpoints.json'),
                records_sha256=file_sha(R20_EVIDENCE / 'run/records.json'),
                f1_validation_feasible=recorded, all_validation_feasible=total,
                f1_validation_firsthit={name: {event: firsthit[name][event] for event in value}
                                        for name, value in R20_F1_VALIDATION_FIRSTHIT.items()},
                b0_f1_validation_feasible=0, holdout_filter_valid=True,
                initial_head_state_sha256=summary['initialization']['per_seed'],
                checkpoint_final_head_state_sha256={name: checkpoints[name]['final_head_state_sha256']
                                                    for name in r20.all_names()},
                f1_deaths='round20: 41/41 F1 deaths happen exactly on wall_frame, i.e. never inside the gap '
                          '(P1-f1-visibility-round20.md, reconstructed by the subtask from the stored actions)')


def r20_assets():
    """Round20's stored per-event asset SHA table; the containment check binds to it."""
    records = json.loads((R20_EVIDENCE / 'run/records.json').read_text(encoding='utf-8'))
    return {entry['event_id']: entry for entry in records['events']}


# ---------------------------------------------------------------- pre-registration
def criterion():
    """The frozen decision rules of this round; any drift aborts the run."""
    return dict(
        round='round21',
        purpose='one pre-registered visibility intervention: draw the F1 coverage band and its gap from '
                'wall_frame-45 so the parking task becomes visible before it becomes lethal; then replay the '
                'round20 protocol unchanged and compare F1 holdout against round20',
        intervention=dict(
            change='the F1 preview band drawn by tools.rl.v373_telegraph.render_state_telegraph only',
            lead_frames=TELEGRAPH_LEAD,
            window='[wall_frame-45, wall_frame) logical frames; the lethal window stays [wall_frame, wall_frame+12)',
            style=dict(preview_fill=list(TELEGRAPH_FILL), gap_edge=list(TELEGRAPH_EDGE), edge_width=EDGE_WIDTH,
                       lethal_fill=list(LETHAL_FILL),
                       note='the preview paints only playfield-background pixels, so bullets and the player sprite '
                            'are never covered; the gap itself is left as a dark slot flanked by fixed edge markers'),
            not_drawn='teacher target region, option, countdown, frame clock, future action, hit state, labels, and '
                      'any F1 geometry outside [wall_frame-45, wall_frame)',
            physics='advance() and every collision rule are untouched: inside the preview window standing in the '
                    'band is not a collision; the same action sequence must hit at the same frame as before',
            other_families='F2/F3 output must be byte-identical to the round20 renderer, which is asserted both by '
                           'unit test and by comparing the stored round21 assets against round20 run/assets',
            unchanged='worlds, world parameters, R19 split, teacher module and teacher actions, labels, pixel '
                      'packing (uint8, depth 3, [max(0,f-4),max(0,f-2),f]), training protocol, closed-loop runner'),
        data=dict(source='the unchanged round20 join of the R18 manifest with the frozen R19 split',
                  train=dict(events=14, groups=7, groups_per_family={'F1': 3, 'F2': 1, 'F3': 3},
                             use='the only events rendered, labelled, encoded or trained on'),
                  validation=dict(events=14, groups=7, groups_per_family={'F1': 3, 'F2': 1, 'F3': 3},
                                  use='read-only: never trained, never labelled, never encoded; closed loop only'),
                  teacher='TacticalTeacherV2, unmodified; its 80-action trajectory per train event must equal the '
                          'frozen round20 rollout action list exactly (no replanning, no repair)',
                  labels='round20 labels verbatim: action_positive = the one-hot of the route first action, '
                         'action_valid = teacher_verified'),
        render=dict(renderer='tools.rl.v373_telegraph.render_state_telegraph',
                    base='touhou_rl.tactical_synthetic_pixels.render_state, unchanged and called as-is',
                    visibility_rule='the preview is frozen by two digests per train event (telegraph frames, and the '
                                    'diff mask against the unchanged renderer) plus one combined digest; the unit test '
                                    'additionally asserts the preview survives the 224x192 area resize'),
        training=dict(arms={r20.CONTROL: 'no anchor: the unchanged tactical_policy_loss',
                            r20.ANCHOR: 'L_original + 1.0*KL(base||policy) over every action_valid row'},
                      anchor_weight=r20.ANCHOR_WEIGHT, seeds=list(SEEDS), updates=r20.UPDATES,
                      optimizer='Adam', lr=r20.LR, clip=r20.CLIP, sequence=r20.SEQUENCE, batch=r20.BATCH,
                      precision='FP32 without AMP, autocast disabled, matmul TF32 False, cuDNN TF32 False, float32 '
                                'matmul precision highest',
                      protocol='tools.rl.v373_valid_curriculum.encode/train_all/Network reused verbatim, so every '
                               'hyper-parameter and the initialization rule are round20 code, not a copy',
                      initialization='a fresh seed reset per N1 construction; the same seed must produce a '
                                     'byte-identical initial head state in both arms and must equal the frozen '
                                     'round20 (and R16/R17) initial head state for that seed',
                      selection='final step only: no early stopping, no validation-based selection, no tuning'),
        evaluation=dict(
            order='all six checkpoints written and SHA-frozen first; only then the six R18 baselines, then B0, then '
                  'the six checkpoints, validation before train',
            condition=CONDITION,
            runner='tools.rl.v373_telegraph.run_event_rendered: a line-for-line copy of the frozen R11 run_event '
                   'with one added renderer parameter; a unit test asserts that, called with render_state, it '
                   'reproduces the frozen run_event exactly (same candidates, same first hit, same feasibility)',
            physics='the environment is still touhou_rl.tactical_teacher_validation.advance; the telegraph is input '
                    'pixels only',
            b0='the frozen baseline policy logits on the identical telegraph renders; a reference, not a capability',
            six_baselines='round20.six_baselines, i.e. the frozen R18 policies()/causal_loop()/outcome(); pixel-free, '
                          'so they must equal round20 row for row',
            metrics='per arm per seed per family feasible and survived, every event first hit, the action distribution '
                    'and switch structure, and for F1 every event: whether the decision-time x had entered the gap, '
                    'the canonical x at the last safe frame, and the direction of the first move',
            comparison='round21 F1 validation feasible per seed minus the round20 recorded value for the same seed; '
                       'improved = any seed strictly greater, unchanged = all equal, worse = any seed lower',
            holdout_rule='if ANY of the six R18 baselines is feasible on ANY validation event then the R18 filter has '
                         'failed on the holdout side: report it verbatim and withhold the learning conclusion',
            reporting='per family F1/F2/F3, never only aggregated; n=3 seeds; F2 validation has one group; mirrors '
                      'are not independent samples',
            degeneracy='single_candidate and switches<=1 are counted, listed and reported, never dropped'),
        claims=dict(
            significance='none: 3 seeds, 7 validation groups, at most 14 independent geometries (F2 has 2 nearly '
                         'identical); descriptive only',
            gate='no formal gate opened, no deployable checkpoint produced, no game executed',
            can_show='whether making an already-determined F1 danger visible before it becomes lethal changes the '
                     'grouped-holdout closed loop of the identical distillation; and, if it does, that this is a '
                     'visibility repair, not tactical understanding',
            cannot_show='learnability of the other families, generalisation beyond 7 held-out geometries, or that a '
                        'learned policy understands anything; a success here is evidence about the input, not the agent'),
        failure_policy='first failure retained; one run; no retrain, rerun, repair of numbers, extra epoch, tuning or '
                       'baseline switch')


def criterion_sha(value=None):
    return hashlib.sha256(json.dumps(criterion() if value is None else value, sort_keys=True).encode()).hexdigest()


def identity_paths():
    return [Path(__file__), Path('tests/test_v373_telegraph.py'),
            Path('tools/rl/v373_valid_curriculum.py'), Path('tests/test_v373_valid_curriculum.py'),
            Path('touhou_rl/tactical_synthetic_pixels.py'), Path('touhou_rl/tactical_teacher_v2.py'),
            Path('touhou_rl/tactical_teacher_validation.py'), Path('touhou_rl/tactical_teacher.py'),
            Path('touhou_rl/tactical_synthetic_closed_loop.py'), Path('touhou_rl/tactical_scenarios.py'),
            Path('touhou_rl/tactical_policy.py'), Path('touhou_rl/vision.py'), Path('touhou_rl/vmodels.py'),
            Path('tools/rl/v373_seed_confirm.py'), Path('tools/rl/v373_world_filter.py'), BASELINE,
            r20.MANIFEST, r20.SPLIT,
            R20_EVIDENCE / 'preregistration.json', R20_EVIDENCE / 'run/records.json',
            R20_EVIDENCE / 'run/summary.json', R20_EVIDENCE / 'run/checkpoints.json',
            REPORTS / 'P1-f1-visibility-round20.md', REPORTS / 'P4-validcurriculum-round20.md']


def identities():
    return {str(path): file_sha(path) for path in identity_paths()}


def span_table(events):
    """Frozen geometry of the preview for every event, without rendering anything."""
    rows = {}
    for event in events:
        world = make_world(event['scenario'])
        window = telegraph_window(world)
        gap = telegraph_gap(world)
        rows[event['event_id']] = dict(
            family=world.family, mirror=bool(world.mirror),
            window=None if window is None else [max(0, window[0]), window[1]],
            spans=[list(span) for span in telegraph_spans(world)],
            gap=None if gap is None else list(gap),
            lethal_window=[world.wall_frame, world.wall_frame + 12] if world.family == 'F1' else None,
            columns_sha256=hashlib.sha256(preview_columns(world).tobytes()).hexdigest())
    return rows


def preregister(out=OUT):
    out = r20.new_dir(out)
    manifest = r20.load_worlds()
    split = r20.load_split(manifest)
    train = r20.world_events(manifest, split, r20.TRAIN_SPLIT)
    validation = r20.world_events(manifest, split, r20.VALIDATION_SPLIT)
    checks = r20.split_checks(manifest, split, train, validation)
    freeze = render_freeze(train)
    physics = physics_probe(train)
    reference = r20_reference()
    initial = r16.initial_head_shas()
    if initial != reference['initial_head_state_sha256']:
        raise ValueError('the initial head state changed since round20')
    if len(set(initial.values())) != len(SEEDS):
        raise ValueError('different seeds must give different initial head states')
    assets = r20_assets()
    if sorted(assets) != sorted(row['event_id'] for row in train):
        raise ValueError('round20 assets must cover exactly the 14 train events')
    world = dict(manifest=manifest['sha256'], split=split['sha256'],
                 train_worlds_digest=r16.worlds_digest(train),
                 validation_worlds_digest=r16.worlds_digest(validation),
                 all_worlds_digest=r16.worlds_digest(train + validation),
                 train_events=[row['event_id'] for row in train],
                 validation_events=[row['event_id'] for row in validation],
                 train_scenario_sha256={row['event_id']: row['scenario_sha256'] for row in train},
                 validation_scenario_sha256={row['event_id']: row['scenario_sha256'] for row in validation})
    value = dict(
        schema=PREREG_SCHEMA, identities=identities(), criterion=criterion(), criterion_sha256=criterion_sha(),
        arms=list(ARMS), seeds=list(SEEDS), updates=r20.UPDATES, lr=r20.LR, clip=r20.CLIP,
        sequence=r20.SEQUENCE, batch=r20.BATCH, anchor_weight=r20.ANCHOR_WEIGHT,
        sources=dict(manifest=dict(path=str(r20.MANIFEST), sha256=manifest['sha256'], schema=r20.MANIFEST_SCHEMA),
                     split=dict(path=str(r20.SPLIT), sha256=split['sha256'], schema=r20.SPLIT_SCHEMA),
                     baseline=dict(path=str(BASELINE), sha256=file_sha(BASELINE)),
                     round20=dict(preregistration=file_sha(R20_EVIDENCE / 'preregistration.json'),
                                  records=reference['records_sha256'], summary=reference['summary_sha256'],
                                  checkpoints=reference['checkpoints_sha256'])),
        teacher=teacher_hashes(), worlds=world,
        split=dict(rule=split['spec']['rule'], counts=split['spec']['counts'], assertions=checks,
                   groups=[dict(group=group['group'], family=group['family'], split=group['split'],
                                index_in_family=group['index_in_family'], signature=group['signature'],
                                events=group['events']) for group in split['groups']]),
        render=dict(renderer='tools.rl.v373_telegraph.render_state_telegraph',
                    renderer_source_sha256=file_sha(Path(__file__)), base='render_state (unchanged)',
                    lead=TELEGRAPH_LEAD, fill=list(TELEGRAPH_FILL), edge=list(TELEGRAPH_EDGE),
                    edge_width=EDGE_WIDTH, lethal_fill=list(LETHAL_FILL), playfield_box=list(PLAYFIELD_BOX),
                    background=list(PLAYFIELD_BACKGROUND),
                    freeze=freeze, physics=physics,
                    spans_by_event=span_table(train + validation)),
        round20=dict(reference=reference,
                     assets={event: dict(pixels=entry['pixels']['sha256'], history=entry['history']['sha256'],
                                         labels=entry['labels']['sha256'], rollout=entry['rollout']['sha256'])
                             for event, entry in sorted(assets.items())},
                     containment=dict(actions='the round21 teacher action list must equal round20\'s stored actions',
                                      labels='labels.pt and history.pt must be tensor-equal to round20\'s',
                                      f2_f3='the stored depth-3 playfield stack must be array-equal to round20\'s',
                                      f1='the same stack must be equal on every row outside the preview window and '
                                         'different on exactly the rows that contain a preview frame')),
        initialization=dict(per_seed=initial, identical_between_arms=True, equals_round20=True,
                            method='fresh seed reset immediately before every N1 construction after the backbone '
                                   'load, through the unchanged r16/r20 code path'),
        expected=EXPECTED, independent_test=False, formal_gate=False, deployable=False, game=False, tuning=False,
        official_labels=False, research_teacher_labels=True, new_worlds=False, new_teacher_labels=True,
        failure_policy='first failure retained; one run; no retrain, rerun, repair, overwrite, extra epoch or baseline '
                       'switch')
    r20.write_json(out / 'preregistration.json', value)
    return value


def verify_prereg(out=OUT):
    out = Path(out)
    p = json.loads((out / 'preregistration.json').read_text(encoding='utf-8'))
    if p.get('schema') != PREREG_SCHEMA:
        raise ValueError('preregistration schema')
    for path, digest in p['identities'].items():
        if file_sha(path) != digest:
            raise ValueError('frozen identity changed: ' + path)
    if p['criterion_sha256'] != criterion_sha():
        raise ValueError('the criterion drifted from the frozen pre-registration')
    manifest = r20.load_worlds()
    split = r20.load_split(manifest)
    if manifest['sha256'] != p['sources']['manifest']['sha256']:
        raise ValueError('R18 manifest SHA drift')
    if split['sha256'] != p['sources']['split']['sha256']:
        raise ValueError('R19 split SHA drift')
    for name in (r20.TRAIN_SPLIT, r20.VALIDATION_SPLIT):
        rows = r20.world_events(manifest, split, name)
        if r16.worlds_digest(rows) != p['worlds'][f'{name}_worlds_digest']:
            raise ValueError('world digest drift: ' + name)
        if [row['event_id'] for row in rows] != p['worlds'][f'{name}_events']:
            raise ValueError('world event list drift: ' + name)
    if r16.initial_head_shas() != p['initialization']['per_seed']:
        raise ValueError('initial head state drift')
    train = r20.world_events(manifest, split, r20.TRAIN_SPLIT)
    freeze = render_freeze(train)
    if freeze['sha256'] != p['render']['freeze']['sha256']:
        raise ValueError('the rendered preview changed since the pre-registration')
    for row, stored in zip(freeze['events'], p['render']['freeze']['events']):
        for key in ('actions_sha256', 'original_sha256', 'telegraph_sha256', 'diff_sha256',
                    'changed_pixels', 'changed_decision_frames', 'columns_sha256'):
            if row[key] != stored[key]:
                raise ValueError(f'render freeze drift: {row["event_id"]}/{key}')
    physics = physics_probe(train)
    if physics['sha256'] != p['render']['physics']['sha256']:
        raise ValueError('the physics probe changed since the pre-registration')
    if json.dumps(span_table(train + r20.world_events(manifest, split, r20.VALIDATION_SPLIT)),
                  sort_keys=True) != json.dumps(p['render']['spans_by_event'], sort_keys=True):
        raise ValueError('the frozen preview geometry table changed')
    reference = r20_reference()
    if json.dumps(reference, sort_keys=True) != json.dumps(p['round20']['reference'], sort_keys=True):
        raise ValueError('the round20 reference record changed')
    if sorted(r20_assets()) != sorted(p['round20']['assets']):
        raise ValueError('round20 asset set changed')
    if r20_reference()['summary_sha256'] != p['sources']['round20']['summary']:
        raise ValueError('round20 summary SHA drift')
    return p


# ---------------------------------------------------------------- data build
def event_records(prereg, split_name):
    return r20.event_records(prereg, split_name)


def containment(event, payload):
    """Hard comparison against round20's stored assets for the same event."""
    entry = r20_assets()[event['event_id']]
    root = R20_EVIDENCE / 'run'
    if file_sha(root / entry['pixels']['path']) != entry['pixels']['sha256']:
        raise ValueError('round20 pixels SHA drift: ' + event['event_id'])
    rollout = json.loads((root / entry['rollout']['path']).read_text(encoding='utf-8'))
    with np.load(root / entry['pixels']['path']) as archive:
        stored = {name: np.array(archive[name]) for name in archive.files}
    reference_labels = torch.load(root / entry['labels']['path'], map_location='cpu', weights_only=True)
    reference_history = torch.load(root / entry['history']['path'], map_location='cpu', weights_only=True)
    actions_equal = list(rollout['actions']) == list(payload['actions'])
    labels_equal = all(torch.equal(reference_labels[key], payload['labels'][key]) for key in reference_labels)
    history_equal = all(torch.equal(reference_history[key], payload['history'][key]) for key in reference_history)
    if not (actions_equal and labels_equal and history_equal):
        raise ValueError('the teacher trajectory or the labels deviate from round20: ' + event['event_id'])
    shapes_equal = all(stored[name].shape == payload['stored'][name].shape for name in payload['stored'])
    if not shapes_equal:
        raise ValueError('stored pixel shapes differ from round20: ' + event['event_id'])
    changed_rows = sorted({int(row) for row in np.nonzero(
        np.any(stored['playfield'] != payload['stored']['playfield'], axis=(1, 2, 3, 4)))[0]})
    ids = r20.depth_indices()
    preview = telegraph_window(make_world(event['scenario']))
    changed_frames = {frame for frame in DECISION_FRAMES if preview and preview[0] <= frame < preview[1]}
    expected_rows = sorted({int(i) for i in range(r20.DECISIONS)
                            for frame in ids[i] if int(frame) * 2 in changed_frames})
    pixels_equal = not changed_rows
    if event['family'] == 'F1':
        if pixels_equal:
            raise ValueError('the telegraph changed no pixel on an F1 event: ' + event['event_id'])
        if changed_rows != expected_rows:
            raise ValueError('the F1 pixel change is not confined to the preview rows: ' + event['event_id'])
    elif not pixels_equal:
        raise ValueError(f'the telegraph must not change {event["family"]} pixels: ' + event['event_id'])
    for name in stored:
        if name == 'playfield' and not pixels_equal:
            continue
        if not np.array_equal(stored[name], payload['stored'][name]):
            raise ValueError('stored pixels differ from round20: ' + event['event_id'] + '/' + name)
    return dict(actions_equal=True, labels_equal=True, history_equal=True,
                pixels_equal=pixels_equal, changed_rows=changed_rows,
                expected_preview_rows=expected_rows,
                round20_asset_sha256=dict(pixels=entry['pixels']['sha256'], history=entry['history']['sha256'],
                                          labels=entry['labels']['sha256'], rollout=entry['rollout']['sha256']))


def render_event(event):
    """One teacherV2 rollout + the telegraph render + the identical round20 packing."""
    world = make_world(event['scenario'])
    actions, decisions = teacher_rollout(world)
    if len(actions) != r20.DECISIONS:
        raise ValueError('incomplete teacher rollout: ' + event['event_id'])
    if any(action not in ACTION_MAP for action in actions):
        raise ValueError('teacher action outside the frozen action set: ' + event['event_id'])
    verdict = verify(world, actions)
    verified = verdict == 'feasible'
    states = full_states(world, actions)
    native = np.stack([render_state_telegraph(world, state) for state in states])
    if native.dtype != np.uint8 or native.shape != (r20.HORIZON + 1, 480, 640, 3):
        raise ValueError('renderer output shape/dtype: ' + event['event_id'])
    original = np.stack([render_state(world, states[frame]) for frame in DECISION_FRAMES])
    telegraph = native[np.array(DECISION_FRAMES)]
    diff = np.any(original != telegraph, axis=-1)
    digest, counts = diff_digest(diff)
    changed = [int(frame) for frame, count in zip(DECISION_FRAMES, counts) if count]
    preview = telegraph_window(world)
    freeze = dict(actions_sha256=actions_digest(actions), original_sha256=frame_digest(original),
                  telegraph_sha256=frame_digest(telegraph), diff_sha256=digest,
                  changed_decision_frames=len(changed), changed_pixels=int(diff.sum()),
                  first_changed_frame=changed[0] if changed else None,
                  last_changed_frame=changed[-1] if changed else None,
                  preview=[max(0, preview[0]), preview[1]] if preview else None,
                  spans=[list(span) for span in telegraph_spans(world)],
                  gap=list(telegraph_gap(world)) if telegraph_gap(world) else None,
                  columns_sha256=hashlib.sha256(preview_columns(world).tobytes()).hexdigest())
    resized = {name: [] for name in SIZES}
    for frame in DECISION_FRAMES:
        for name, image in vision.extract_regions(native[frame]).items():
            resized[name].append(vision.resize_area(image, *SIZES[name]))
    ids = r20.depth_indices()
    stacks = {}
    for name, frames in resized.items():
        array = np.stack(frames)
        if array.dtype != np.uint8:
            raise ValueError('uint8 frames required: ' + name)
        stacks[name] = np.ascontiguousarray(array[ids])
    hud_constant = {name: len({frame.tobytes() for frame in resized[name]}) == 1 for name in r20.HUD_NAMES}
    if not all(hud_constant.values()):
        raise ValueError('HUD regions must be constant for the single-copy storage: ' + event['event_id'])
    stored = {'playfield': stacks['playfield']}
    for name in r20.HUD_NAMES:
        stored[name] = np.ascontiguousarray(stacks[name][:1])
    history, labels = r20.history_and_labels(actions, verified)
    switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
    return dict(world=world, actions=actions, decisions=decisions, verdict=verdict, verified=verified,
                stored=stored, history=history, labels=labels, hud_constant=hud_constant, freeze=freeze,
                switches=switches, distinct_actions=sorted(set(actions)),
                action_counts={str(a): actions.count(a) for a in sorted(set(actions))})


def build_event(directory, event):
    payload = render_event(event)
    pixels = directory / 'pixels.npz'
    history = directory / 'history.pt'
    labels = directory / 'labels.pt'
    r20.write_npz(pixels, payload['stored'])
    r20.write_torch(history, payload['history'])
    r20.write_torch(labels, payload['labels'])
    checked = containment(event, payload)
    rollout = directory / 'rollout.json'
    row = dict(event_id=event['event_id'], family=event['family'], group=event['group'],
               split=event['split'], mirror=event['mirror'], scenario=event['scenario'],
               scenario_sha256=event['scenario_sha256'],
               teacher='TacticalTeacherV2 (unmodified)', teacher_source=teacher_hashes(),
               renderer='v373_telegraph.render_state_telegraph',
               renderer_source_sha256=file_sha(Path(__file__)),
               telegraph=dict(lead=TELEGRAPH_LEAD, fill=list(TELEGRAPH_FILL), edge=list(TELEGRAPH_EDGE),
                              edge_width=EDGE_WIDTH, window=payload['freeze']['preview'],
                              spans=payload['freeze']['spans'], gap=payload['freeze']['gap']),
               render_freeze=payload['freeze'], round20_containment=checked,
               actions=payload['actions'], decisions=payload['decisions'],
               decisions_count=len(payload['actions']), horizon=r20.HORIZON,
               verify_verdict=payload['verdict'], teacher_verified=payload['verified'],
               switches=payload['switches'], distinct_actions=payload['distinct_actions'],
               action_counts=payload['action_counts'], hud_constant=payload['hud_constant'],
               depth_frames=r20.depth_frames(), region_sizes=SIZES,
               labels=dict(rows=r20.DECISIONS, action_valid=int(payload['labels']['action_valid'].sum()),
                           positive_per_valid_row=1, semantics='single route first action only'))
    r20.write_json(rollout, row)
    raw_bytes = sum(int(array.nbytes) for array in payload['stored'].values())
    return dict(event=event, payload=payload,
                refs=dict(pixels=dict(path=f"assets/{event['event_id']}/pixels.npz", sha256=file_sha(pixels),
                                      bytes=pixels.stat().st_size, raw_bytes=raw_bytes,
                                      shapes={name: list(array.shape) for name, array in payload['stored'].items()}),
                          history=dict(path=f"assets/{event['event_id']}/history.pt", sha256=file_sha(history),
                                       bytes=history.stat().st_size),
                          labels=dict(path=f"assets/{event['event_id']}/labels.pt", sha256=file_sha(labels),
                                      bytes=labels.stat().st_size),
                          rollout=dict(path=f"assets/{event['event_id']}/rollout.json", sha256=file_sha(rollout),
                                       bytes=rollout.stat().st_size)))


def build_assets(dest, events):
    r20.new_dir(dest / 'assets')
    built = []
    for event in events:
        directory = r20.new_dir(dest / 'assets' / event['event_id'])
        built.append(build_event(directory, event))
        print('rendered', event['event_id'], 'verdict', built[-1]['payload']['verdict'],
              'preview', built[-1]['payload']['freeze']['preview'],
              'changed_pixels', built[-1]['payload']['freeze']['changed_pixels'],
              'stored_bytes', built[-1]['refs']['pixels']['bytes'], flush=True)
    return built


def write_records(dest, built):
    entries, failures = [], []
    for item in built:
        event = item['event']
        entries.append(dict(event_id=event['event_id'], family=event['family'], group=event['group'],
                            split=event['split'], mirror=event['mirror'], rows=r20.DECISIONS,
                            horizon=r20.HORIZON, logical_frames=r20.HORIZON,
                            teacher_verified=item['payload']['verified'],
                            verify_verdict=item['payload']['verdict'],
                            scenario_sha256=event['scenario_sha256'], switches=item['payload']['switches'],
                            distinct_actions=item['payload']['distinct_actions'],
                            action_counts=item['payload']['action_counts'],
                            render_freeze=item['payload']['freeze'], **item['refs']))
        if not item['payload']['verified']:
            failures.append(dict(event_id=event['event_id'], verdict=item['payload']['verdict']))
    records = dict(schema=DATA_SCHEMA, root='assets', events=entries, failures=failures,
                   renderer='v373_telegraph.render_state_telegraph (F1 preview only)',
                   renderer_source_sha256=file_sha(Path(__file__)),
                   telegraph=dict(lead=TELEGRAPH_LEAD, fill=list(TELEGRAPH_FILL), edge=list(TELEGRAPH_EDGE),
                                  edge_width=EDGE_WIDTH, window='[wall_frame-45, wall_frame)',
                                  lethal_window='[wall_frame, wall_frame+12)'),
                   depth=3, region_sizes=SIZES, depth_frames='[max(0,f-4),max(0,f-2),f]',
                   storage=r20.STORAGE,
                   history_semantics='identical to round20: actual executed candidate (ACTION_MAP), previous_hold 2, '
                                     'frame_delta 2, option BOS3, target_valid false, history_valid from row 1',
                   label_semantics='identical to round20: the one-hot of the single successful teacher route first '
                                   'action; action_valid = teacher_verified; nothing else',
                   round20_containment='actions, history and labels are tensor/list-equal to round20 for every event; '
                                       'F2/F3 pixels are array-equal; F1 pixels differ exactly on the depth rows '
                                       'that contain a preview frame',
                   totals=dict(events=len(entries), verified=len(entries) - len(failures), failed=len(failures),
                               rows=len(entries) * r20.DECISIONS,
                               bytes=sum(e['pixels']['bytes'] + e['history']['bytes'] + e['labels']['bytes']
                                         + e['rollout']['bytes'] for e in entries),
                               changed_pixels=sum(e['render_freeze']['changed_pixels'] for e in entries)))
    r20.write_json(dest / 'records.json', records)
    return records


# ---------------------------------------------------------------- closed loop
def run_event_rendered(s, arm, condition, network, render=render_state):
    """The frozen R11 run_event with exactly one added parameter: the renderer.

    Every other line - the depth-3 causal stack, the 18-way mask, the candidate to
    two-frame action mapping, the real advance, the sticky hit, the full horizon, the
    per-event hidden reset - is copied from touhou_rl.tactical_synthetic_closed_loop.
    """
    state = initial_state(s)
    images = [render(s, state)]
    steps = []
    actions = []
    hidden = None
    previous = None
    first_collision = None
    invalid = False
    positions = [dict(frame=0, x=state.x, hit=False)]
    while state.frame < s.horizon:
        regions, ids = pixel_stack(images)
        reset = state.frame == 0
        logits, next_hidden = network(arm, regions, previous, hidden, reset)
        if logits.shape != (18,) or not torch.isfinite(logits).all():
            raise ValueError('invalid network logits')
        raw = int(logits.argmax())
        selected = logits.clone()
        if condition == 'horizontal_masked':
            selected[[c for c in range(18) if c not in HORIZONTAL]] = -torch.inf
        candidate = int(selected.argmax())
        action = HORIZONTAL.get(candidate)
        row = dict(frame=state.frame, pixels=dict(indices=ids, logical_frames=[2 * i for i in ids]),
                   hidden_reset=reset,
                   history=dict(previous_candidate=18 if previous is None else previous, previous_option=3,
                                previous_hold=0 if previous is None else 2,
                                frame_delta=0 if previous is None else 2,
                                target_valid=False, history_valid=previous is not None),
                   raw_unmasked_candidate=raw, raw_unsupported=raw not in HORIZONTAL, candidate=candidate,
                   actual_candidate=None, actual_action=None, actual_hold=0,
                   x_before=state.x, x_after=state.x, invalid_action=action is None, fallback=False,
                   first_collision_frame=first_collision)
        if action is None:
            invalid = True
            steps.append(row)
            break
        before = state
        state = advance(s, state, action)
        actions.append(action)
        if state.hit and not before.hit:
            first_collision = state.frame
        row.update(actual_candidate=candidate, actual_action=action, actual_hold=state.frame - before.frame,
                   x_after=state.x, first_collision_frame=first_collision, hit=state.hit)
        steps.append(row)
        positions.append(dict(frame=state.frame, x=state.x, hit=state.hit))
        previous = candidate
        hidden = next_hidden
        images.append(render(s, state))
    complete = state.frame == s.horizon and not invalid
    goal = s.family != 'F3' or any(s.canonical_x(t['x']) <= s.goal for t in positions)
    physical = metrics(s, actions) if complete else None
    return dict(event_id=s.event_id, family=s.family, arm=arm, condition=condition, steps=steps,
                trace=positions, first_collision_frame=first_collision, collision_time_resolution_frames=2,
                full_horizon=complete, goal_required=s.family == 'F3', goal_satisfied=goal,
                survived=complete and not state.hit, feasible=complete and not state.hit and goal,
                technical_failure='invalid_action' if invalid else None, invalid_action=invalid,
                fallback_count=0, raw_unsupported_count=sum(r['raw_unsupported'] for r in steps),
                decisions=len(steps), environment_metrics=physical)


def closed_loop(events, network, arm, label, render=render_state_telegraph):
    rows = []
    for entry in events:
        row = r20.compact(run_event_rendered(make_world(entry['scenario']), arm, CONDITION, network,
                                             render=render))
        row['network'] = label
        rows.append(row)
        print('closed-loop', label, entry['event_id'], 'feasible', row['feasible'],
              'survived', row['survived'], 'firsthit', row['first_collision_frame'], flush=True)
    return rows


def gap_trace(s, actions):
    """Per logical frame x rebuilt from `advance` semantics (two frames per decision)."""
    state = initial_state(s)
    rows = [dict(frame=0, x=state.x)]
    for action in actions:
        before = state
        state = advance(s, state, action)
        rows.append(dict(frame=before.frame + 1, x=min(376., max(8., before.x + action * 2.))))
        rows.append(dict(frame=state.frame, x=state.x))
    return rows


def gap_diagnostics(s, actions, hit_frame=None):
    """For one F1 event and one executed action list: was the gap ever entered, and when?

    The position series is rebuilt from `advance` semantics, so it contains the odd
    logical frame that `advance` checks but the 2-frame decision grid never samples. That
    matters: every F1 wall_frame in this curriculum is even, so the last safe frame is odd.
    The comparison is in canonical x, which is the coordinate the collision rule uses.
    """
    trace = gap_trace(s, actions)
    gap = canonical_gap(s)
    canonical = [s.canonical_x(row['x']) for row in trace]
    inside = [gap is not None and gap[0] <= value <= gap[1] for value in canonical]
    wall = s.wall_frame
    last_safe = wall - 1
    last_safe_valid = 0 <= last_safe < len(canonical)
    safe = [index for index, row in enumerate(trace) if row['frame'] < wall and inside[index]]
    entered = [index for index, value in enumerate(inside) if value]
    first_move = next((row['frame'] for row in trace[1:] if abs(row['x'] - trace[0]['x']) > 1e-9), None)
    net = canonical[-1] - canonical[0]
    at_wall = canonical[last_safe] if last_safe_valid else None
    in_gap_at_wall = bool(inside[last_safe]) if last_safe_valid else None
    distance = None if (gap is None or at_wall is None) else max(gap[0] - at_wall, at_wall - gap[1], 0.)
    if in_gap_at_wall:
        mechanism = 'in_the_gap_at_the_wall'
    elif entered:
        mechanism = 'entered_the_gap_then_left_it'
    elif distance is not None and at_wall is not None and at_wall < gap[0]:
        mechanism = 'stopped_left_of_the_gap'
    elif distance is not None and at_wall is not None and at_wall > gap[1]:
        mechanism = 'stopped_right_of_the_gap'
    else:
        mechanism = 'never_near_the_gap'
    return dict(
        event_id=s.event_id, group=s.template_group, mirror=bool(s.mirror),
        complete=len(actions) == r20.DECISIONS,
        wall_frame=wall, gap=list(gap) if gap else None,
        gap_screen=list(telegraph_gap(s)) if telegraph_gap(s) else None, coverage=list(s.coverage),
        preview=[max(0, wall - TELEGRAPH_LEAD), wall - 1],
        decisions_in_preview_window=sum(1 for frame in DECISION_FRAMES
                                       if max(0, wall - TELEGRAPH_LEAD) <= frame < wall),
        start_canonical=round(float(s.canonical_x(s.start_x)), 6), net_canonical_move=round(float(net), 6),
        direction='right' if net > 1e-9 else 'left' if net < -1e-9 else 'hold',
        first_move_frame=first_move,
        entered_gap=bool(entered), entered_gap_frame=trace[entered[0]]['frame'] if entered else None,
        entered_gap_decision=entered[0] // 2 if entered else None,
        frames_in_gap_before_wall=len(safe),
        first_frame_in_gap_before_wall=trace[safe[0]]['frame'] if safe else None,
        in_gap_at_wall=in_gap_at_wall,
        canonical_at_wall=round(float(at_wall), 6) if at_wall is not None else None,
        distance_to_gap_at_wall=round(float(distance), 6) if distance is not None else None,
        hit_frame=hit_frame, mechanism=mechanism,
        mechanism_note='advance semantics were rebuilt per logical frame; "in_gap_at_wall" is the position at '
                       'wall_frame-1, the last frame before the band becomes lethal')


def gap_table(closed, split_name, events):
    """One diagnostic row per (network, F1 event) on a stored closed-loop payload."""
    rows = {}
    for name in ['B0'] + r20.all_names():
        by_event = {row['event_id']: row for row in closed[(name, split_name)]['events']}
        for entry in events:
            if entry['family'] != 'F1':
                continue
            world = make_world(entry['scenario'])
            row = by_event[entry['event_id']]
            actions = [step['actual_action'] for step in row['steps'] if step.get('actual_action') is not None]
            value = gap_diagnostics(world, actions, row['first_collision_frame'])
            value.update(network=name, split=split_name, feasible=row['feasible'], survived=row['survived'],
                         decisions=row['decisions'])
            rows[f'{name}/{entry["event_id"]}'] = value
    return rows


def evaluate_all(dest, prereg, backbone, device, reports, paths):
    validation = event_records(prereg, r20.VALIDATION_SPLIT)
    train = event_records(prereg, r20.TRAIN_SPLIT)
    r20.new_dir(dest / 'closed-loop')
    begin = time.perf_counter()
    baselines = dict(validation=r20.six_baselines(validation), train=r20.six_baselines(train))
    if sum(payload['rows'] for payload in baselines.values()) != EXPECTED['baseline_rows']:
        raise AssertionError('six baselines x 14 events x 2 splits required')
    r20.write_json(dest / 'baselines.json', baselines)
    reference = json.loads((R20_EVIDENCE / 'run/baselines.json').read_text(encoding='utf-8'))
    for split_name in ('validation', 'train'):
        now = [(row['event_id'], row['policy'], row['feasible'], row['survived'], row['first_hit'])
               for row in baselines[split_name]['rows_detail']]
        before = [(row['event_id'], row['policy'], row['feasible'], row['survived'], row['first_hit'])
                  for row in reference[split_name]['rows_detail']]
        if now != before:
            raise AssertionError('the frozen R18 baselines changed: ' + split_name)
    closed = {}
    for split_name, events in ((r20.VALIDATION_SPLIT, validation), (r20.TRAIN_SPLIT, train)):
        rows = closed_loop(events, r20.BaselineNetwork(backbone, device), 'B0', 'B0')
        payload = dict(network='B0', split=split_name, condition=CONDITION,
                       renderer='v373_telegraph.render_state_telegraph', events=rows,
                       families=r16.families(rows), behaviour=r20.behaviour(rows))
        r20.write_json(dest / 'closed-loop' / f'B0-{split_name}.json', payload)
        closed[('B0', split_name)] = payload
    for name in r20.all_names():
        network = r20.Network(backbone, paths[name], device, reports[name]['final_head_state_sha256'])
        for split_name, events in ((r20.VALIDATION_SPLIT, validation), (r20.TRAIN_SPLIT, train)):
            rows = closed_loop(events, network, 'N1', name)
            payload = dict(network=name, split=split_name, condition=CONDITION,
                           renderer='v373_telegraph.render_state_telegraph', events=rows,
                           families=r16.families(rows), behaviour=r20.behaviour(rows),
                           heads_sha256=network.heads_sha256)
            r20.write_json(dest / 'closed-loop' / f'{name}-{split_name}.json', payload)
            closed[(name, split_name)] = payload
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    if len(closed) != EXPECTED['closed_loop_files']:
        raise AssertionError('closed-loop file count')
    gaps = {split_name: gap_table(closed, split_name, events)
            for split_name, events in ((r20.VALIDATION_SPLIT, validation), (r20.TRAIN_SPLIT, train))}
    r20.write_json(dest / 'f1-gap.json', dict(
        notes='per (network, F1 event): the x series is rebuilt from advance semantics, so the odd logical frame is '
              'included; "in_gap_at_wall" is the position on wall_frame-1, the last frame before the band is lethal',
        validation=gaps[r20.VALIDATION_SPLIT], train=gaps[r20.TRAIN_SPLIT]))
    return baselines, closed, gaps, seconds


def write_events(dest, prereg, records, closed, baselines, gaps):
    r20.new_dir(dest / 'events')
    assets = {entry['event_id']: entry for entry in records['events']}
    baseline_rows = {(row['split'], row['event_id'], row['policy']): row
                     for split in baselines.values() for row in split['rows_detail']}
    for event_id in prereg['worlds']['train_events'] + prereg['worlds']['validation_events']:
        split_name = r20.TRAIN_SPLIT if event_id in prereg['worlds']['train_events'] else r20.VALIDATION_SPLIT
        teacher = None
        if event_id in assets:
            teacher = json.loads((dest / assets[event_id]['rollout']['path']).read_text(encoding='utf-8'))
        payloads = {name: closed[(name, split_name)] for name in ['B0'] + r20.all_names()}
        rows = {name: next(row for row in payload['events'] if row['event_id'] == event_id)
                for name, payload in payloads.items()}
        behaviour = {name: next(item for item in payload['behaviour']['per_event']
                                if item['event_id'] == event_id)
                     for name, payload in payloads.items()}
        value = dict(schema=EVENT_SCHEMA, event_id=event_id, split=split_name,
                     seen=(split_name == r20.TRAIN_SPLIT), family=event_id.split('-')[1],
                     group=event_id.rsplit('-m', 1)[0], teacher_verified=event_id in assets,
                     asset=assets.get(event_id), teacher=teacher,
                     gap={name: gaps[split_name].get(f'{name}/{event_id}')
                          for name in ['B0'] + r20.all_names()},
                     r18_six_baselines={name: dict(feasible=baseline_rows[(split_name, event_id, name)]['feasible'],
                                                   survived=baseline_rows[(split_name, event_id, name)]['survived'],
                                                   first_hit=baseline_rows[(split_name, event_id, name)]['first_hit'])
                                        for name in baselines['validation']['policies']},
                     closed_loop={name: dict(feasible=rows[name]['feasible'], survived=rows[name]['survived'],
                                             goal_satisfied=rows[name]['goal_satisfied'],
                                             first_hit=rows[name]['first_collision_frame'],
                                             decisions=rows[name]['decisions'],
                                             raw_unsupported_count=rows[name]['raw_unsupported_count'],
                                             action_counts=behaviour[name]['action_counts'],
                                             switches=behaviour[name]['switches'],
                                             distinct_candidates=behaviour[name]['distinct_candidates'],
                                             single_candidate=behaviour[name]['single_candidate'])
                                  for name in ['B0'] + r20.all_names()})
        r20.write_json(dest / 'events' / f'{event_id}.json', value)


# ---------------------------------------------------------------- summary
def _containment_table(dest, entries):
    """The round20 containment result of every stored event, re-read from its rollout.json."""
    table = {}
    for entry in entries:
        payload = json.loads((Path(dest) / entry['rollout']['path']).read_text(encoding='utf-8'))
        checked = payload['round20_containment']
        table[entry['event_id']] = dict(actions_equal=checked['actions_equal'],
                                        labels_equal=checked['labels_equal'],
                                        history_equal=checked['history_equal'],
                                        pixels_equal=checked['pixels_equal'],
                                        changed_rows=checked['changed_rows'],
                                        expected_preview_rows=checked['expected_preview_rows'])
    return table


def comparison(closed, reference):
    """round21 minus round20 on the same seed and the same six F1 validation events."""
    result = {}
    for arm in ARMS:
        seeds = []
        for index, seed in enumerate(SEEDS):
            name = r20.name_of(arm, seed)
            now = closed[(name, r20.VALIDATION_SPLIT)]['families']['F1']['feasible']
            before = reference['f1_validation_feasible'][arm][index]
            total_now = closed[(name, r20.VALIDATION_SPLIT)]['families']['all']['feasible']
            total_before = reference['all_validation_feasible'][arm][index]
            seeds.append(dict(seed=seed, network=name, f1_round20=before, f1_round21=now, f1_delta=now - before,
                              all_round20=total_before, all_round21=total_now, all_delta=total_now - total_before))
        deltas = [row['f1_delta'] for row in seeds]
        result[arm] = dict(per_seed=seeds, deltas=deltas,
                           improved=sum(1 for value in deltas if value > 0),
                           unchanged=sum(1 for value in deltas if value == 0),
                           worse=sum(1 for value in deltas if value < 0),
                           verdict='improved' if any(value > 0 for value in deltas)
                           else 'unchanged' if all(value == 0 for value in deltas) else 'worse')
    pooled = [value for arm in ARMS for value in result[arm]['deltas']]
    result['pooled'] = dict(deltas=pooled, improved=sum(1 for value in pooled if value > 0),
                            unchanged=sum(1 for value in pooled if value == 0),
                            worse=sum(1 for value in pooled if value < 0),
                            note='six (arm, seed) pairs, six F1 validation events each; descriptive only')
    return result


def assemble(dest, extra=None):
    """Rebuild the round summary read-only from the frozen artifacts."""
    dest = Path(dest)
    out = dest.parent
    prereg = json.loads((out / 'preregistration.json').read_text(encoding='utf-8'))
    if prereg['criterion_sha256'] != criterion_sha():
        raise ValueError('the criterion drifted from the frozen pre-registration')
    records = json.loads((dest / 'records.json').read_text(encoding='utf-8'))
    checkpoints = json.loads((dest / 'checkpoints.json').read_text(encoding='utf-8'))
    if sorted(checkpoints) != sorted(r20.all_names()):
        raise ValueError('checkpoint set changed')
    training = {}
    for name in r20.all_names():
        record = checkpoints[name]
        payload = json.loads((dest / 'train' / name / 'training.json').read_text(encoding='utf-8'))
        if file_sha(dest / record['checkpoint_path']) != record['checkpoint_sha256']:
            raise ValueError('checkpoint SHA drift: ' + name)
        if payload['final_head_state_sha256'] != record['final_head_state_sha256']:
            raise ValueError('checkpoint record mismatch: ' + name)
        if payload['initial_head_state_sha256'] != prereg['initialization']['per_seed'][str(payload['seed'])]:
            raise ValueError('initial head SHA differs from the pre-registration: ' + name)
        training[name] = payload
    for seed in SEEDS:
        if training[r20.name_of(r20.CONTROL, seed)]['initial_head_state_sha256'] != \
                training[r20.name_of(r20.ANCHOR, seed)]['initial_head_state_sha256']:
            raise ValueError('both arms must start from identical heads for seed ' + str(seed))
    expected = {r20.VALIDATION_SPLIT: prereg['worlds']['validation_events'],
                r20.TRAIN_SPLIT: prereg['worlds']['train_events']}
    closed = {}
    for name in ['B0'] + r20.all_names():
        for split_name in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT):
            payload = json.loads((dest / 'closed-loop' / f'{name}-{split_name}.json').read_text(encoding='utf-8'))
            if r16.families(payload['events']) != payload['families']:
                raise ValueError('stored aggregate mismatch: ' + name + '-' + split_name)
            if [row['event_id'] for row in payload['events']] != expected[split_name]:
                raise ValueError('event list differs from the pre-registration: ' + name + '-' + split_name)
            if name != 'B0' and payload['heads_sha256'] != training[name]['final_head_state_sha256']:
                raise ValueError('closed-loop heads SHA mismatch: ' + name)
            closed[(name, split_name)] = payload
    baselines = json.loads((dest / 'baselines.json').read_text(encoding='utf-8'))
    gaps = json.loads((dest / 'f1-gap.json').read_text(encoding='utf-8'))
    reference = r20_reference()
    results = {}
    for name in r20.all_names():
        results[name] = dict(arm=training[name]['arm'], seed=training[name]['seed'],
                             validation=closed[(name, r20.VALIDATION_SPLIT)]['families'],
                             train_seen=closed[(name, r20.TRAIN_SPLIT)]['families'],
                             validation_behaviour=closed[(name, r20.VALIDATION_SPLIT)]['behaviour'],
                             train_seen_behaviour=closed[(name, r20.TRAIN_SPLIT)]['behaviour'],
                             checkpoint_sha256=training[name]['checkpoint_sha256'],
                             final_head_state_sha256=training[name]['final_head_state_sha256'],
                             initial_head_state_sha256=training[name]['initial_head_state_sha256'],
                             anchor_weight=training[name]['anchor_weight'],
                             anchor_loss_at_step0=training[name].get('anchor_loss_at_step0'),
                             anchor_loss_final=training[name].get('anchor_loss_final'),
                             loss_at_step0=training[name].get('loss_at_step0'),
                             loss_final=training[name].get('loss_final'),
                             optimization_seconds=training[name]['optimization_seconds'],
                             final_offline=training[name]['final_offline'],
                             initial_offline=training[name]['initial_offline'])
    holdout = baselines['validation']
    summary = dict(
        schema=RESULT_SCHEMA, prereg_sha256=file_sha(out / 'preregistration.json'),
        criterion_sha256=prereg['criterion_sha256'], seeds=list(SEEDS), arms=list(ARMS),
        updates=r20.UPDATES, anchor_weight=r20.ANCHOR_WEIGHT,
        intervention=dict(lead=TELEGRAPH_LEAD, window='[wall_frame-45, wall_frame)',
                          fill=list(TELEGRAPH_FILL), edge=list(TELEGRAPH_EDGE), edge_width=EDGE_WIDTH,
                          lethal_fill=list(LETHAL_FILL),
                          renderer='v373_telegraph.render_state_telegraph',
                          renderer_source_sha256=prereg['render']['renderer_source_sha256'],
                          note='the only change against round20 is that the F1 band and its gap are drawn 45 frames '
                               'before they become lethal'),
        render_freeze=dict(sha256=prereg['render']['freeze']['sha256'],
                           events={row['event_id']: dict(changed_pixels=row['changed_pixels'],
                                                         changed_decision_frames=row['changed_decision_frames'],
                                                         first=row['first_changed_frame'],
                                                         last=row['last_changed_frame'],
                                                         preview=row['preview'], spans=row['spans'],
                                                         gap=row['gap'])
                                    for row in prereg['render']['freeze']['events']}),
        physics=dict(sha256=prereg['render']['physics']['sha256'],
                     probes=prereg['render']['physics']['probes'],
                     note='constant hold/right/left replays through the unmodified advance; unchanged from the '
                          'pre-registration and independent of the renderer'),
        sources=prereg['sources'], teacher=prereg['teacher'],
        worlds=dict(train_events=len(prereg['worlds']['train_events']),
                    validation_events=len(prereg['worlds']['validation_events']),
                    train_worlds_digest=prereg['worlds']['train_worlds_digest'],
                    validation_worlds_digest=prereg['worlds']['validation_worlds_digest'],
                    all_worlds_digest=prereg['worlds']['all_worlds_digest'],
                    train_event_ids=prereg['worlds']['train_events'],
                    validation_event_ids=prereg['worlds']['validation_events']),
        split=dict(rule=prereg['split']['rule'], counts=prereg['split']['counts'],
                   assertions=prereg['split']['assertions']),
        data=dict(events=len(records['events']), rows=records['totals']['rows'], storage=records['storage'],
                  depth=records['depth'], region_sizes=records['region_sizes'],
                  depth_frames=records['depth_frames'], history_semantics=records['history_semantics'],
                  label_semantics=records['label_semantics'], totals=records['totals'],
                  failures=records['failures'],
                  round20_containment=_containment_table(dest, records['events'])),
        initialization=dict(per_seed={str(seed): training[r20.name_of(r20.CONTROL, seed)]['initial_head_state_sha256']
                                      for seed in SEEDS},
                            per_seed_per_arm={str(seed): {arm: training[r20.name_of(arm, seed)]['initial_head_state_sha256']
                                                          for arm in ARMS} for seed in SEEDS},
                            identical_between_arms=True, equals_round20=True),
        checkpoints={name: {key: value for key, value in training[name].items()
                            if key not in ('losses', 'original_losses', 'anchor_losses',
                                           'initial_offline_events', 'final_offline_events',
                                           'supervised_events')} for name in r20.all_names()},
        validation=dict(
            events=EXPECTED['validation_events'], per_family=EXPECTED['validation_events_per_family'],
            holdout=dict(filter_valid=holdout['filter_valid'], passing_events=holdout['passing_events'],
                         executor=holdout['executor'], policies=holdout['policies'],
                         summary=holdout['summary'],
                         rule=prereg['criterion']['evaluation']['holdout_rule']),
            b0=dict(families=closed[('B0', r20.VALIDATION_SPLIT)]['families'],
                    behaviour=closed[('B0', r20.VALIDATION_SPLIT)]['behaviour']),
            results={name: results[name]['validation'] for name in r20.all_names()},
            behaviour={name: results[name]['validation_behaviour'] for name in r20.all_names()},
            per_event_firsthit={name: closed[(name, r20.VALIDATION_SPLIT)]['families']['all']['firsthit']
                                for name in ['B0'] + r20.all_names()},
            f1_gap=gaps['validation'],
            note='never trained, never labelled, never encoded: the only split that can falsify this distillation'),
        train_seen=dict(
            events=EXPECTED['train_events'], per_family=EXPECTED['train_events_per_family'],
            label='seen: exactly the 14 teacher rollouts the heads were trained on',
            holdout=dict(filter_valid=baselines['train']['filter_valid'],
                         passing_events=baselines['train']['passing_events'],
                         summary=baselines['train']['summary']),
            b0=dict(families=closed[('B0', r20.TRAIN_SPLIT)]['families'],
                    behaviour=closed[('B0', r20.TRAIN_SPLIT)]['behaviour']),
            results={name: results[name]['train_seen'] for name in r20.all_names()},
            behaviour={name: results[name]['train_seen_behaviour'] for name in r20.all_names()},
            per_event_firsthit={name: closed[(name, r20.TRAIN_SPLIT)]['families']['all']['firsthit']
                                for name in ['B0'] + r20.all_names()},
            f1_gap=gaps['train']),
        round20_reference=reference,
        comparison=comparison(closed, reference),
        environment=dict(device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                         torch_version=torch.__version__,
                         precision=dict(dtype='float32', matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                        cudnn_tf32=torch.backends.cudnn.allow_tf32,
                                        matmul_precision=torch.get_float32_matmul_precision()),
                         timings=json.loads((dest / 'timings.json').read_text(encoding='utf-8'))
                         if (dest / 'timings.json').is_file() else None),
        flags=dict(independent_test=False, formal_gate=False, deployable=False, game=False, tuning=False,
                   official_labels=False, research_teacher_labels=True, validation_used_for_selection=False,
                   early_stopping=False, retrained=False, rerun=False, physics_changed=False,
                   intervention_count=1),
        limitations=[
            'n=3 seeds and 7 validation groups; the 14 validation events collapse to 7 geometries because a mirror is '
            'not an independent sample',
            'F2 validation is a single group (2 events) built on two nearly identical worlds, so an F2 validation '
            'number is not a rate',
            'the only intervention is a renderer; physics, worlds, split, teacher, labels and training are round20',
            'the teacher is handed the real world geometry; "distillation" means fitting that one route',
            'the F1 win condition is exact parking inside a 12-20 px gap and the F3 win condition is a timed threshold '
            'crossing; neither is tactical understanding, and this round cannot show that a policy understands a '
            'telegraph, only whether the same task became learnable once the danger was visible',
            'the 45-frame lead is a single design choice and is not ablated here',
            'synthetic fixed-y worlds, two-frame actions, no shot, no bomb, no real TH08 pixels',
            'research heads only: no formal gate, no deployable checkpoint, no game'],
        independent_test=False, formal_gate=False, deployable=False, game=False, tuning=False)
    if extra:
        summary['environment'].update(extra)
    return summary


def verify_artifacts(out=OUT):
    """Read-only re-check of every frozen artifact and its SHA."""
    out = Path(out)
    prereg = verify_prereg(out)
    dest = out / 'run'
    records = json.loads((dest / 'records.json').read_text(encoding='utf-8'))
    checked = 0
    for entry in records['events']:
        for key in ('pixels', 'history', 'labels', 'rollout'):
            if file_sha(dest / entry[key]['path']) != entry[key]['sha256']:
                raise ValueError('asset SHA drift: ' + entry['event_id'] + '/' + key)
            checked += 1
    checkpoints = json.loads((dest / 'checkpoints.json').read_text(encoding='utf-8'))
    for name, record in checkpoints.items():
        if file_sha(dest / record['checkpoint_path']) != record['checkpoint_sha256']:
            raise ValueError('checkpoint SHA drift: ' + name)
    baselines = json.loads((dest / 'baselines.json').read_text(encoding='utf-8'))
    closed = 0
    for name in ['B0'] + r20.all_names():
        for split_name in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT):
            payload = json.loads((dest / 'closed-loop' / f'{name}-{split_name}.json').read_text(encoding='utf-8'))
            if len(payload['events']) != len(prereg['worlds'][f'{split_name}_events']):
                raise ValueError('closed-loop event count: ' + name + '-' + split_name)
            closed += 1
    stored = json.loads((dest / 'sha256.json').read_text(encoding='utf-8'))
    drift = [path for path, digest in stored.items() if file_sha(dest / path) != digest]
    if drift:
        raise ValueError('stored SHA table drift: ' + ', '.join(drift))
    root = json.loads((out / 'sha256.json').read_text(encoding='utf-8'))
    root_drift = [path for path, digest in root.items() if file_sha(out / path) != digest]
    if root_drift:
        raise ValueError('root SHA table drift: ' + ', '.join(root_drift))
    gaps = json.loads((dest / 'f1-gap.json').read_text(encoding='utf-8'))
    return dict(artifacts=len(stored), root_artifacts=len(root), assets=checked,
                checkpoints=len(checkpoints), closed_loop_files=closed,
                baseline_rows=baselines['validation']['rows'] + baselines['train']['rows'],
                baseline_passing_events_validation=baselines['validation']['passing_events'],
                holdout_filter_valid=baselines['validation']['filter_valid'],
                f1_gap_rows=len(gaps['validation']) + len(gaps['train']),
                render_freeze_sha256=prereg['render']['freeze']['sha256'],
                physics_sha256=prereg['render']['physics']['sha256'],
                train_events=len(prereg['worlds']['train_events']),
                validation_events=len(prereg['worlds']['validation_events']),
                criterion_sha256=prereg['criterion_sha256'], identities=len(prereg['identities']))


# ---------------------------------------------------------------- execution
def run(out=OUT, device='cuda'):
    out = Path(out)
    prereg = verify_prereg(out)
    dest = r20.new_dir(out / 'run')
    try:
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise ValueError('real CUDA required')
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        with _fp32_mode(torch.device(device)):
            if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
                raise AssertionError('TF32 must be disabled')
            timings = {}
            begin = time.perf_counter()
            built = build_assets(dest, event_records(prereg, r20.TRAIN_SPLIT))
            records = write_records(dest, built)
            timings['data_seconds'] = round(time.perf_counter() - begin, 3)
            if records['totals']['events'] != EXPECTED['train_events']:
                raise AssertionError('train event count')
            if len(records['failures']) != EXPECTED['excluded_events']:
                raise AssertionError('the teacher must verify all 14 train events')
            for entry in records['events']:
                frozen = next(row for row in prereg['render']['freeze']['events']
                              if row['event_id'] == entry['event_id'])
                for key in ('actions_sha256', 'original_sha256', 'telegraph_sha256', 'diff_sha256',
                            'changed_pixels', 'changed_decision_frames'):
                    if entry['render_freeze'][key] != frozen[key]:
                        raise AssertionError('render freeze drift: ' + entry['event_id'] + '/' + key)
            backbone, _ = load_vision_checkpoint(BASELINE, map_location='cpu')
            backbone.to(device).eval()
            frozen = {key: value.detach().clone() for key, value in backbone.state_dict().items()}
            begin = time.perf_counter()
            data = r20.encode(backbone, records, dest, device)
            torch.cuda.synchronize()
            timings['encoding_seconds'] = round(time.perf_counter() - begin, 3)
            if len(data) != EXPECTED['train_events']:
                raise AssertionError('encoded event count')
            reports, paths = r20.train_all(dest, data, backbone, device, prereg)
            if not all(torch.equal(value, frozen[key]) for key, value in backbone.state_dict().items()):
                raise AssertionError('the frozen backbone changed')
            baselines, closed, gaps, seconds = evaluate_all(dest, prereg, backbone, device, reports, paths)
            timings['evaluation_seconds'] = round(seconds, 3)
            timings['training_seconds_total'] = round(sum(reports[name]['optimization_seconds']
                                                          for name in reports), 3)
            r20.write_json(dest / 'timings.json', timings)
            write_events(dest, prereg, records, closed, baselines, gaps)
            if not baselines['validation']['filter_valid']:
                print('WARNING the R18 filter failed on the holdout side:',
                      baselines['validation']['passing_events'], flush=True)
            verify_prereg(out)
            summary = assemble(dest)
            r20.write_json(dest / 'summary.json', summary)
            r20.write_json(dest / 'sha256.json',
                           {str(path.relative_to(dest)): file_sha(path) for path in dest.rglob('*') if path.is_file()})
            r20.write_json(out / 'sha256.json',
                           {str(path.relative_to(out)): file_sha(path) for path in out.rglob('*') if path.is_file()})
            return summary
    except BaseException:
        r20.write_json(dest / 'first-failure.json',
                       dict(traceback=traceback.format_exc(),
                            policy='retained; no automatic rerun, overwrite, retune or baseline switch'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preregister', 'run', 'summary', 'verify'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'preregister':
        preregister(args.out)
    elif args.command == 'run':
        run(args.out, args.device)
    elif args.command == 'summary':
        print(json.dumps(assemble(args.out / 'run'), indent=2)[:4000])
    else:
        print(json.dumps(verify_artifacts(args.out), indent=2))
