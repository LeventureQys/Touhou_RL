"""Round24: one single-factor intervention on top of round23's training-only privileged
auxiliary task - the label's reference point moves from the fixed playfield centre (x=192) to
the player's own visible x; research only.

Round23 established that the auxiliary *objective* can move behaviour (base-group toward rate
6/9 -> 9/9) but that it moved it the wrong way (mirror 3/9 -> 1/9, screen directions 12R/3L/3hold
-> 17R/1L/0hold), and it named the reason itself: the label was a **position fact** (`gap centre
above x=192`, the playfield centre) while the policy needs a **displacement** (`which way must the
player move`). The two coincide only when the player stands on the same side of the screen centre
as the gap; on the round23 holdout they disagree in 2 of 6 F1 events (`007-m0`, `007-m1`), and
those are exactly the two cells the control arm already solved.

This round changes exactly one thing: the label is `sign(gap_screen_centre - player_x)`, i.e. the
gap's screen centre relative to **the player**, not to the screen centre. `gap` is the privileged
world parameter (round21.telegraph_gap, the same arithmetic the R21 renderer draws the band with);
`player_x` is a *visible* state - `render_state` draws the player sprite at exactly `state.x`
(native playfield columns 32+x), and the action acts on that same screen x (`advance` does
`x = clip(x + action*2)`). So the label is exactly "which sign of action closes the distance to the
slot", which is the thing the policy has to output. Nothing else changes: the auxiliary head
(`Linear(256,2)`, 514 parameters, outside `heads`), the GRU-state capture by a forward pre-hook on
`heads['option']`, supervision only on the telegraph-window rows, the fixed weight (0.5 for arm X,
0.0 for arm A), the two arms, the three seeds, 200 updates, the round21 assets, the R19 split, the
closed-loop runner and the evaluation read-out are round23's, reused through `v373_privileged_aux`
rather than copied, so that the two rounds differ in one number and one definition.

Three things are frozen before the run and measured, never asserted:

  1. **Where the label comes from.** The teacher's own trajectory supplies `player_x`: for the 14
     train events the round19 evidence (`validteacher-round19/run/worlds/...`) stores the frozen
     teacher_v2 action list, which is asserted to be *equal* to the round21 stored rollout actions
     (the actions the training pixels were rendered from). For the six F1 validation worlds the same
     frozen list gives the reference label used by the agreement diagnostic and by the label
     read-out; no teacher is executed by this round (`teacher_calls = 0`).
  2. **The sign-flip rule.** `gap_centre - player_x` can change sign inside the preview window if the
     player crosses the gap centre. The frozen rule is `first_preview_row`: the label is the sign at
     the **first row of the telegraph window**, applied to every supervised row of that event. The
     per-row signs, the flip rows and the flip frames are all recorded for every event, train and
     holdout, and the report discloses them; the alternative rule (per-row sign) is not tried.
  3. **The judgement rule**, which replaces round23's "any seed larger = improved". X is `improved`
     over A **iff** K1 holds - F1 holdout feasible is strictly higher on at least two of the three
     seeds and lower on none - and both pooled feasible counts (F1 and all-family) do not fall. If
     K1 holds and either pooled count falls, the verdict is `mixed`, never `improved`.

Arm A is the same code path with the auxiliary weight at zero, so arm A must reproduce round23's
(and therefore round22's) control checkpoint **and** every closed-loop step bitwise. That is
measured per seed and reported (`matches_round23_control`, `equivalence`), never assumed; if it
fails the round reports `control_not_reproduced` and withholds the causal claim.

No new worlds, no teacher calls, no new tactical labels, no re-rendered pixels (round21's frozen
assets, SHA-bound), no formal gate, no deployable checkpoint, no game execution, no tuning.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tools.rl import v373_privileged_aux as r23
from tools.rl import v373_spatial_fuse as r22
from tools.rl import v373_telegraph as r21
from tools.rl import v373_valid_curriculum as r20
from tools.rl import v373_seed_confirm as r16
from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_feature_cache import _fp32_mode
from touhou_rl.tactical_policy import TacticalConfig
from touhou_rl.tactical_teacher_validation import make_world
from touhou_rl.vmodels import load_vision_checkpoint

ROOT = Path('Document/Update/Alpha/v3.7.3/evidence')
OUT = ROOT / 'prelaux-round24'
R23_EVIDENCE = ROOT / 'privaux-round23'
R22_EVIDENCE = ROOT / 'spatialfuse-round22'
R21_EVIDENCE = ROOT / 'telegraph-round21'
R19_EVIDENCE = ROOT / 'validteacher-round19'
REPORTS = Path('Document/Update/Alpha/v3.7.3/reports')
BASELINE = r20.BASELINE

PREREG_SCHEMA = 'v373-prelaux-prereg/1'
DATA_SCHEMA = 'v373-prelaux-data/1'
LABEL_SCHEMA = 'v373-prelaux-label/1'
EVENT_SCHEMA = 'v373-prelaux-event/1'
RESULT_SCHEMA = 'v373-prelaux-result/1'
# The checkpoint contract is round23's, verbatim: same schema string, same 13 payload keys and
# the same `heads` state dict, so `r23.PrivNetwork` reads these files unchanged. Only the label
# (never a tensor of the payload) differs between the rounds.
HEADS_SCHEMA = r23.HEADS_SCHEMA
PAYLOAD_KEYS = r23.PAYLOAD_KEYS

# ---------------------------------------------------------------- the intervention
ARMS = r23.ARMS
CONTROL, AUXILIARY = r23.CONTROL, r23.AUXILIARY
AUX_WEIGHT = r23.AUX_WEIGHT           # the single number the two arms disagree on
SEEDS = r20.SEEDS
LOSS_ARM = r20.CONTROL
FEATURE_DIM = r23.FEATURE_DIM
HIDDEN_SIZE = r23.HIDDEN_SIZE
AUX_DIM = r23.AUX_DIM
AUX_LABELS = r23.AUX_LABELS           # 0 = the gap is screen-left of the player: move left
AUX_PARAMETERS = r23.AUX_PARAMETERS   # 514, unchanged
STATE_HOOK_MODULE = r23.STATE_HOOK_MODULE
POLICY_HEADS_ACTIVE = r23.POLICY_HEADS_ACTIVE
GRU_INPUT_WIDTH = r23.GRU_INPUT_WIDTH
HEAD_MODULES = r23.HEAD_MODULES
AUX_SUPERVISION_WINDOW = r23.AUX_SUPERVISION_WINDOW
LABEL_RULE = 'first_preview_row'
TIE_EPS = 1e-9
REFERENCE_FRAME_CENTRE = 192.         # round23's reference point; kept only to detect a regression

AUX_TASK = ('from the GRU state predict whether the safe gap lies left or right OF THE PLAYER, i.e. '
            'which sign of action closes the distance to the slot: label = sign(gap screen centre - '
            'player screen x), gap from round21.telegraph_gap(s) (privileged world parameter), '
            'player x from the frozen teacher trajectory (train) or the checkpoint\'s own trajectory '
            '(scoring); the label is read at the first row of the telegraph window')
LABEL_RULE_TEXT = ('the sign is taken at the FIRST decision row whose logical frame lies inside '
                   'round21.telegraph_window(s) = [wall_frame-45, wall_frame) and applied to every '
                   'supervised row of that event; per-row signs, flip rows and flip frames are '
                   'recorded for every event and reported, and the per-row variant is not tried')

EXPECTED = dict(events=28, train_events=14, validation_events=14,
                train_events_per_family={'F1': 6, 'F2': 2, 'F3': 6},
                validation_events_per_family={'F1': 6, 'F2': 2, 'F3': 6},
                checkpoints=6, closed_loop_files=14, networks=7, baseline_rows=2 * 14 * 6,
                f1_gap_rows=2 * 7 * 6, event_files=28,
                aux_events=6, aux_train_rows=128, aux_validation_rows=132,
                aux_unlabelled_train_events=8, flip_free_teacher_events=12,
                judgement_seeds=3, judgement_higher_needed=2)

# Round23's recorded results, re-read and re-asserted from its frozen artifacts before this round
# starts. They are the only comparison baseline of round24 and are never typed in without a check.
R23_CONTROL_FINAL_HEAD_SHA = dict(r23.R22_CONTROL_FINAL_HEAD_SHA)
R23_F1_VALIDATION_FEASIBLE = {'A': [2, 2, 2], 'X': [1, 3, 2]}
R23_ALL_VALIDATION_FEASIBLE = {'A': [3, 3, 3], 'X': [1, 3, 2]}
R23_DELTAS = [-1, 1, 0]
R23_VERDICT = 'improved'              # round23's rule: any seed larger
R23_DIRECTION = dict(
    A=dict(base=dict(toward=6, hold=3, away=0), mirror=dict(toward=3, hold=0, away=6)),
    X=dict(base=dict(toward=9, hold=0, away=0), mirror=dict(toward=1, hold=0, away=8)))
R23_USAGE = dict(
    A=dict(aux_correct_cells=9, mirror_aux_correct=4, mirror_away=6, aux_says_left=6,
           aux_says_right=12, base_aux_correct=5, base_toward=6),
    X=dict(aux_correct_cells=11, mirror_aux_correct=8, mirror_away=8, aux_says_left=16,
           aux_says_right=2, base_aux_correct=3, base_toward=9))
R23_AUX_HOLDOUT = {'A': .467172, 'X': .606061}


# ---------------------------------------------------------------- the privileged label
def round19_world(event_id, family):
    """The frozen round19 per-world record of one event (read-only, never re-executed)."""
    return json.loads((R19_EVIDENCE / 'run' / 'worlds' / family / f'{event_id}.json')
                      .read_text(encoding='utf-8'))


def teacher_table(events):
    """The frozen teacher_v2 action list of every event, from round19's stored evidence.

    This is the *same* data the round21 training pixels were rendered from: for the 14 train
    events the stored list is asserted to equal round21's `rollout.json` actions, so "the label
    read the trajectory the pixels show" is a measured identity. Round24 executes no teacher
    (`teacher_calls = 0`): it re-reads a frozen table.
    """
    table, verdicts = {}, {}
    assets = r23.r21_assets()
    for entry in events:
        event_id = entry['event_id']
        record = round19_world(event_id, entry['family'])
        if record.get('event_id') != event_id or record.get('schema') != 'v373-valid-teacher-world/1':
            raise ValueError('round19 world record mismatch: ' + event_id)
        actions = [int(value) for value in record['teachers']['teacher_v2']['actions']]
        if len(actions) != r20.DECISIONS:
            raise ValueError('incomplete stored teacher trajectory: ' + event_id)
        if event_id in assets:
            stored = json.loads((R21_EVIDENCE / 'run' / assets[event_id]['rollout']['path'])
                                .read_text(encoding='utf-8'))['actions']
            if [int(value) for value in stored] != actions:
                raise ValueError('the round19 teacher table is not the round21 rollout: ' + event_id)
        table[event_id] = actions
        verdicts[event_id] = record['teachers']['teacher_v2']['verify']
    return table, verdicts


def player_trace(world, actions):
    """The player's visible screen x on every decision frame, from the frozen actions.

    `render_state` draws the player sprite at `32 + state.x`, and `advance` moves that same
    screen x by `2 * action`; so this list is literally what the pixels show, and the sign of
    `gap centre - x` is the sign of the action that closes the distance.
    """
    positions = r22.positions_per_row(world, actions)
    if len(positions) != r20.DECISIONS:
        raise ValueError('one player x per decision required')
    return positions


def gap_centre(world):
    gap = r21.telegraph_gap(world)
    return None if gap is None else (float(gap[0]) + float(gap[1])) / 2.


def row_sign(offset):
    return 0 if abs(offset) <= TIE_EPS else (1 if offset > 0 else -1)


def aux_terms(world, positions):
    """Everything the loss and the scoring need about one world's label, or None.

    `positions` must be the visible player x of the *same* trajectory the rows came from: the
    teacher's for the train events, the checkpoint's own for the holdout read-out.
    """
    gap = r21.telegraph_gap(world)
    centre = gap_centre(world)
    if gap is None or centre is None:
        return None
    index = r23.preview_indices(world)
    if not index:
        return None
    offsets = [round(float(centre - float(positions[row])), 6) for row in index]
    signs = [row_sign(value) for value in offsets]
    flips = [index[k] for k in range(1, len(index)) if signs[k] != signs[k - 1]]
    first = index[0]
    side = int(offsets[0] > 0)
    mask = np.array([False] * r20.DECISIONS)
    mask[index] = True
    return dict(side=side, label=AUX_LABELS[side], mask=mask, rows=len(index),
                first_row=first, first_frame=int(r21.DECISION_FRAMES[first]),
                first_player_x=round(float(positions[first]), 6), first_offset=offsets[0],
                first_sign=signs[0],
                gap_screen=[int(gap[0]), int(gap[1])], gap_screen_centre=round(centre, 6),
                preview=[int(r21.telegraph_window(world)[0]), int(r21.telegraph_window(world)[1])],
                row_indices=index, row_frames=[int(r21.DECISION_FRAMES[row]) for row in index],
                offsets=offsets, per_row_signs=signs, sign_flips=flips,
                sign_flip_frames=[int(r21.DECISION_FRAMES[row]) for row in flips],
                tie_rows=[index[k] for k, value in enumerate(offsets) if value == 0],
                would_flip=bool(flips),
                positions=[round(float(value), 6) for value in positions],
                offset_from_screen_centre=round(centre - REFERENCE_FRAME_CENTRE, 6),
                round23_side=int(centre > REFERENCE_FRAME_CENTRE),
                distance_from_centre=round(abs(centre - REFERENCE_FRAME_CENTRE), 6))


def aux_target(world, positions):
    """Training-only label: 1 if the gap is screen-right of the player, else 0. None without a gap."""
    terms = aux_terms(world, positions)
    return None if terms is None else terms['side']


def aux_mask(world):
    """The rows the label is supervised on: inside the telegraph window. None if no label."""
    if r21.telegraph_gap(world) is None:
        return None
    window = r21.telegraph_window(world)
    return np.array([window[0] <= frame < window[1] for frame in r21.DECISION_FRAMES], dtype=bool)


def aux_batch(terms):
    """The [1,80] long target and bool mask of one event, or (None, None) without a label."""
    if terms is None:
        return None, None
    target = torch.full((1, r20.DECISIONS), int(terms['side']), dtype=torch.long)
    return target, torch.from_numpy(terms['mask'][None].copy())


def side_name(value):
    return 'none' if value is None else AUX_LABELS[int(value)]


def sign_name(value):
    return 'tie' if value == 0 else AUX_LABELS[1 if value > 0 else 0]


def action_direction(action):
    return 'hold' if int(action) == 0 else ('right' if int(action) > 0 else 'left')


def action_match(action, sign):
    """Does the teacher's action point the same way as `sign`? agree / disagree / hold.

    `sign` is a sign (-1/0/+1) or a class (0 = left, 1 = right); both encode "which way", and a
    hold is neither agreement nor disagreement because the label never predicts it.
    """
    if int(action) == 0:
        return 'hold'
    return 'agree' if (int(action) > 0) == (sign > 0) else 'disagree'


def label_rows(events, teacher):
    """`{event_id: terms}` for every event that carries a label, from the frozen teacher table."""
    out = {}
    for entry in events:
        world = make_world(entry['scenario'])
        positions = player_trace(world, teacher[entry['event_id']])
        terms = aux_terms(world, positions)
        if terms is not None:
            out[entry['event_id']] = terms
    return out


def supervision_table(records, teacher):
    """The frozen auxiliary supervision: which event carries a label, which row, which sign."""
    rows = {}
    for entry in records:
        world = make_world(entry['scenario'])
        terms = aux_terms(world, player_trace(world, teacher[entry['event_id']]))
        value = dict(event_id=entry['event_id'], family=entry['family'], mirror=bool(world.mirror),
                     target_defined=terms is not None)
        if terms is not None:
            value.update({key: terms[key] for key in
                          ('side', 'label', 'gap_screen', 'gap_screen_centre', 'rows', 'first_row',
                           'first_frame', 'first_player_x', 'first_offset', 'preview',
                           'row_indices', 'row_frames', 'offsets', 'per_row_signs', 'sign_flips',
                           'sign_flip_frames', 'tie_rows', 'would_flip', 'positions',
                           'offset_from_screen_centre', 'round23_side')})
            value['supervised_rows'] = terms['rows']
            value['supervised_row_indices'] = [int(row) for row in np.nonzero(terms['mask'])[0]]
        rows[entry['event_id']] = value
    return rows


def agreement_table(events, teacher):
    """Does the player-relative label agree with the teacher's own first action, per split?

    Reported because it bounds what this label can be: if it did not agree with the frozen
    teacher, it could not be read as "the direction the teacher takes". Two readings are given,
    because they answer different questions:
      * `episode_first_action` - the teacher's action at decision 0, i.e. "what does the teacher
        do first", which is the literal question;
      * `anchored_action` - the teacher's action at the row the label is anchored on (the first
        preview row), i.e. "does the teacher move that way at the moment the band appears".
    A `hold` teacher action is neither agreement nor disagreement: the label never predicts hold,
    so holds are counted separately and excluded from the agreement rate.
    """
    rows = {}
    for entry in events:
        event_id = entry['event_id']
        world = make_world(entry['scenario'])
        actions = teacher[event_id]
        positions = player_trace(world, actions)
        terms = aux_terms(world, positions)
        row = dict(event_id=event_id, family=entry['family'], mirror=bool(world.mirror),
                   split=entry['split'], label_defined=terms is not None,
                   teacher_first_action=int(actions[0]),
                   teacher_first_direction=action_direction(actions[0]))
        if terms is None:
            row.update(label=None, episode_first_action='no_label', anchored_action='no_label',
                       anchored_row=None, anchored_teacher_action=None)
        else:
            anchored = int(actions[terms['first_row']])
            row.update(label=terms['label'], side=terms['side'], first_sign=terms['first_sign'],
                       episode_first_action=action_match(actions[0], terms['first_sign']),
                       anchored_row=terms['first_row'], anchored_teacher_action=anchored,
                       anchored_action=action_match(anchored, terms['first_sign']))
        rows[event_id] = row
    counts = {}
    for split_name in (r20.TRAIN_SPLIT, r20.VALIDATION_SPLIT):
        subset = [row for row in rows.values() if row['split'] == split_name]
        labelled = [row for row in subset if row['label_defined']]
        block = dict(events=len(subset), labelled=len(labelled),
                     unlabelled=len(subset) - len(labelled),
                     unlabelled_events=[row['event_id'] for row in subset if not row['label_defined']])
        for key in ('episode_first_action', 'anchored_action'):
            values = [row[key] for row in labelled]
            agree, disagree = values.count('agree'), values.count('disagree')
            holds = values.count('hold')
            block[key] = dict(agree=agree, disagree=disagree, teacher_hold=holds,
                              agreement_rate=(round(agree / (agree + disagree), 6)
                                              if agree + disagree else None),
                              includes_teacher_hold=False)
        block['label_counts'] = {name: sum(1 for row in labelled if row['label'] == name)
                                 for name in AUX_LABELS}
        counts[split_name] = block
    return dict(rows=rows, by_split=counts,
                note='the label never predicts hold, so a hold first action is counted separately and '
                     'excluded from the rate; F2/F3 worlds have no gap and therefore no label')


def round23_collinearity(events, teacher):
    """Was round23's label a function of the player's own visible x?

    Round23's label is `gap screen centre > 192`. On these worlds the gap and the player's start
    are usually on the same side of 192, so that label can collapse to `player x > 192`, i.e. to a
    statement about the player alone. This table measures that collapse per event, without which
    round23's high linear decodability of its label from the GRU state cannot be interpreted.
    """
    rows = {}
    for entry in events:
        world = make_world(entry['scenario'])
        terms = aux_terms(world, player_trace(world, teacher[entry['event_id']]))
        if terms is None:
            continue
        half = int(terms['first_player_x'] > REFERENCE_FRAME_CENTRE)
        rows[entry['event_id']] = dict(
            split=entry['split'], mirror=bool(world.mirror),
            round23_side=terms['round23_side'], round24_side=terms['side'],
            player_screen_half=half,
            round23_equals_player_half=bool(terms['round23_side'] == half),
            first_player_x=terms['first_player_x'], gap_screen_centre=terms['gap_screen_centre'])
    counts = {}
    for split_name in (r20.TRAIN_SPLIT, r20.VALIDATION_SPLIT):
        subset = [row for row in rows.values() if row['split'] == split_name]
        counts[split_name] = dict(
            events=len(subset),
            round23_equals_player_half=sum(1 for row in subset if row['round23_equals_player_half']),
            round24_equals_player_half=sum(1 for row in subset
                                           if bool(int(row['round24_side']) == row['player_screen_half'])),
            differing=[key for key, row in rows.items()
                       if row['split'] == split_name and row['round23_equals_player_half'] is False])
    return dict(rows=rows, by_split=counts,
                note='`round23_side` is round23\'s label (gap centre above 192); `player_screen_half` is '
                     'the player\'s own visible x above 192 at the same row. Where the two coincide, '
                     'round23\'s target was a statement about the player, not about the gap')


def flip_table(labels_by_event):
    """Per event: the per-row signs, the flip rows and frames, and how the rule handles them."""
    rows = {}
    for event_id, terms in sorted(labels_by_event.items()):
        rows[event_id] = dict(
            rule=LABEL_RULE, target_side=terms['side'], target_label=terms['label'],
            first_row=terms['first_row'], first_frame=terms['first_frame'],
            first_offset=terms['first_offset'], first_sign=terms['first_sign'], rows=terms['rows'],
            row_frames=terms['row_frames'], offsets=terms['offsets'],
            per_row_signs=terms['per_row_signs'], sign_flips=terms['sign_flips'],
            sign_flip_frames=terms['sign_flip_frames'], tie_rows=terms['tie_rows'],
            would_flip=terms['would_flip'],
            handling=('the target is the sign at the first preview row and is applied to every '
                     'supervised row; a later flip is recorded here and is NOT supervised, so the '
                     'head is asked "which way at the moment the band appears", not "which way now" '
                     'on each row'))
    return dict(schema=LABEL_SCHEMA, rule=LABEL_RULE, rule_text=LABEL_RULE_TEXT,
                events=rows, flipping_events=sorted(key for key, row in rows.items()
                                                    if row['would_flip']),
                event_count=len(rows),
                flip_count=sum(len(row['sign_flips']) for row in rows.values()),
                tie_count=sum(len(row['tie_rows']) for row in rows.values()),
                note='the teacher trajectory is used here; a checkpoint\'s own trajectory is scored '
                     'with the same rule and its flips are reported in probes.json')


# ---------------------------------------------------------------- round23 reference
def r23_reference():
    """Re-read round23's frozen results and re-assert every number this round quotes."""
    summary_path = R23_EVIDENCE / 'run/summary.json'
    checkpoints_path = R23_EVIDENCE / 'run/checkpoints.json'
    probes_path = R23_EVIDENCE / 'run/probes.json'
    prereg_path = R23_EVIDENCE / 'preregistration.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    checkpoints = json.loads(checkpoints_path.read_text(encoding='utf-8'))
    probes = json.loads(probes_path.read_text(encoding='utf-8'))
    prereg = json.loads(prereg_path.read_text(encoding='utf-8'))
    f1 = {arm: [summary['validation']['results'][r20.name_of(arm, seed)]['F1']['feasible']
                for seed in SEEDS] for arm in ARMS}
    if f1 != R23_F1_VALIDATION_FEASIBLE:
        raise ValueError('round23 F1 validation reference drift: ' + json.dumps(f1))
    total = {arm: [summary['validation']['results'][r20.name_of(arm, seed)]['all']['feasible']
                   for seed in SEEDS] for arm in ARMS}
    if total != R23_ALL_VALIDATION_FEASIBLE:
        raise ValueError('round23 total validation reference drift: ' + json.dumps(total))
    deltas = summary['comparison']['deltas']
    if deltas != R23_DELTAS or summary['comparison']['verdict'] != R23_VERDICT:
        raise ValueError('round23 comparison reference drift')
    control_sha = {r20.name_of(CONTROL, seed): checkpoints[r20.name_of(CONTROL, seed)]
                   ['final_head_state_sha256'] for seed in SEEDS}
    if control_sha != R23_CONTROL_FINAL_HEAD_SHA:
        raise ValueError('round23 control final head reference drift')
    if summary['equivalence']['all_equal'] is not True or \
            summary['equivalence']['control_not_reproduced']:
        raise ValueError('round23 recorded a control that did not reproduce round22')
    direction = {}
    for arm in ARMS:
        groups = summary['validation']['direction'][arm]['groups']
        direction[arm] = {group: {key: groups[group][key] for key in ('toward', 'hold', 'away')}
                          for group in ('base', 'mirror')}
    if direction != R23_DIRECTION:
        raise ValueError('round23 direction reference drift: ' + json.dumps(direction))
    usage = probes['usage']['by_arm']
    recorded = {arm: {key: usage[arm][key] for key in R23_USAGE[arm]} for arm in ARMS}
    if recorded != R23_USAGE:
        raise ValueError('round23 usage reference drift: ' + json.dumps(recorded))
    holdout = {arm: probes['aux']['by_arm'][arm]['holdout']['accuracy'] for arm in ARMS}
    if holdout != R23_AUX_HOLDOUT:
        raise ValueError('round23 auxiliary holdout reference drift: ' + json.dumps(holdout))
    if prereg['auxiliary']['threshold'] != REFERENCE_FRAME_CENTRE:
        raise ValueError('round23 label reference drift')
    if summary['validation']['holdout']['filter_valid'] is not True:
        raise ValueError('round23 recorded a failed holdout filter')
    cells = probes['usage']['cells']
    mirror_correct_and_away = {}
    for arm in ARMS:
        mirror_correct_and_away[arm] = sum(
            1 for row in cells.values()
            if row['arm'] == arm and row['mirror'] and row['aux_correct']
            and row['policy_verdict'] == 'away')
    return dict(preregistration_sha256=file_sha(prereg_path),
                summary_sha256=file_sha(summary_path),
                checkpoints_sha256=file_sha(checkpoints_path),
                probes_sha256=file_sha(probes_path),
                f1_gap_sha256=file_sha(R23_EVIDENCE / 'run/f1-gap.json'),
                records_sha256=file_sha(R23_EVIDENCE / 'run/records.json'),
                criterion_sha256=prereg['criterion_sha256'],
                label='round23 label = gap screen centre above x=192 (the playfield centre), not '
                      'above the player',
                f1_validation_feasible=f1, all_validation_feasible=total,
                comparison=dict(deltas=deltas, verdict=R23_VERDICT,
                                identical_first_hit_cells=summary['comparison']['identical_first_hit_cells'],
                                note='round23\'s rule was "any seed larger = improved"; this round '
                                     'replaces it with K1 and the mixed verdict'),
                control_final_head_state_sha256=control_sha,
                direction=direction, usage=recorded, holdout_accuracy=holdout,
                mirror_aux_correct_and_away=mirror_correct_and_away,
                label_position_table=prereg['auxiliary']['supervised_events']['train'],
                behaviour='round23: screen directions 12R/3L/3hold (A) -> 17R/1L/0hold (X); mirror '
                          'toward rate 3/9 -> 1/9; the auxiliary head read the mirror side correctly '
                          'in 8/9 cells while the policy moved away in 8/9')


# ---------------------------------------------------------------- data (round21 features)
def event_records(prereg, split_name):
    return r23.event_records(prereg, split_name)


def train_records(prereg):
    return r23.train_records(prereg)


def asset_bindings(records):
    return r23.asset_bindings(records)


def encode(backbone, records, device):
    """Round22's encoder, verbatim, through round23's wrapper (same features, same digest)."""
    return r23.encode(backbone, records, device)


def feature_norms(data):
    return r23.feature_norms(data)


def write_records(dest, records, supervision, flips):
    entries, failures = [], []
    assets = r23.r21_assets()
    for entry in records:
        entries.append(dict(event_id=entry['event_id'], family=entry['family'], group=entry['group'],
                            split=entry['split'], mirror=entry['mirror'], rows=r20.DECISIONS,
                            horizon=r20.HORIZON, teacher_verified=entry['teacher_verified'],
                            scenario_sha256=entry['scenario_sha256'],
                            pixels=entry['pixels'], history=entry['history'], labels=entry['labels'],
                            rollout=entry['rollout'], aux=supervision[entry['event_id']],
                            bytes=sum(assets[entry['event_id']][key]['bytes']
                                      for key in ('pixels', 'history', 'labels', 'rollout'))))
        if not entry['teacher_verified']:
            failures.append(dict(event_id=entry['event_id'], verdict='not_teacher_verified'))
    labelled = [row for row in entries if row['aux']['target_defined']]
    value = dict(schema=DATA_SCHEMA, root=str(R21_EVIDENCE / 'run'), events=entries, failures=failures,
                 bindings=asset_bindings(records),
                 source='round21 telegraph assets, reopened read-only and SHA-bound; nothing re-rendered, '
                        'no teacher executed, no tactical label created',
                 features='round22.spatial_vectors reused verbatim through round23; the 94-d spatial '
                          'vector it also computes is not used and only serves as the frozen proof that '
                          'the encode pass is round22\'s',
                 depth=3, region_sizes=r22.SIZES, depth_frames=r20.depth_frames(), storage=r20.STORAGE,
                 teacher='round19 run/worlds/<family>/<event>.json, teacher_v2 action list, asserted equal '
                         'to round21\'s stored rollout actions for all 14 train events',
                 aux=dict(task=AUX_TASK, label_rule=LABEL_RULE, labels=list(AUX_LABELS),
                          supervision_window=AUX_SUPERVISION_WINDOW,
                          reference_point='the player x, not the playfield centre 192',
                          round23_reference_point=REFERENCE_FRAME_CENTRE,
                          labelled_events=len(labelled),
                          unlabelled_events=[row['event_id'] for row in entries
                                             if not row['aux']['target_defined']],
                          labelled_rows=sum(row['aux']['supervised_rows'] for row in labelled),
                          label_counts={name: sum(1 for row in labelled if row['aux']['label'] == name)
                                        for name in AUX_LABELS},
                          label_vs_round23_label={row['event_id']:
                                                  dict(round24=row['aux']['label'],
                                                       round23=side_name(row['aux']['round23_side']),
                                                       differs=row['aux']['label'] !=
                                                       side_name(row['aux']['round23_side']))
                                                  for row in labelled},
                          sign_flips=flips['flipping_events'], flip_count=flips['flip_count'],
                          tie_count=flips['tie_count'],
                          never_an_input='the label is a training-loss target only: it is not a pixel '
                                         'region, not a history field, not a feature channel, and no '
                                         'policy head tensor depends on it'),
                 history_semantics='identical to round21 (round20 packing), unchanged',
                 label_semantics='identical to round21 (round20 tactical labels), unchanged',
                 totals=dict(events=len(entries),
                             teacher_verified=sum(1 for e in entries if e['teacher_verified']),
                             rows=len(entries) * r20.DECISIONS,
                             bytes=sum(e['bytes'] for e in entries)))
    r20.write_json(dest / 'records.json', value)
    return value


# ---------------------------------------------------------------- training
def infer_with_aux(model, entry):
    return r23.infer_with_aux(model, entry)


def train_all(dest, data, backbone, device, prereg, labels):
    """Two arms x three seeds through round23's loop; only the target source changed.

    The loop body, the optimizer, the clip scope (the 429200 policy-head parameters, identically
    in both arms), the data order and every call sequence are round23's `train_all`; the only
    difference is that `targets` is built from `sign(gap centre - player x)` instead of from
    `gap centre > 192`. Because weight 0.0 contributes an exactly-zero gradient, arm A must
    reproduce round23's (and round22's) control head bitwise; that equality is recorded per seed
    and reported, never silently assumed.
    """
    supervised = [entry for entry in data if entry['entry']['teacher_verified']]
    excluded = [entry['entry']['event_id'] for entry in data if not entry['entry']['teacher_verified']]
    if len(supervised) + len(excluded) != len(data) or not supervised:
        raise ValueError('supervision partition')
    targets = {event_id: aux_batch(terms) for event_id, terms in labels.items()}
    labelled = sorted(targets)
    if len(labelled) != EXPECTED['aux_events']:
        raise AssertionError('six F1 train events carry the auxiliary label')
    r20.new_dir(dest / 'train')
    reports, paths = {}, {}
    for seed in SEEDS:
        per_arm_initial = {}
        for arm in ARMS:
            name = r20.name_of(arm, seed)
            directory = r20.new_dir(dest / 'train' / name)
            model = r23.fresh_aux_head(backbone, seed).to(device)
            if model.tactical_config.feature_dim != FEATURE_DIM:
                raise AssertionError('the backbone trunk width is not the frozen 256')
            if model.tactical_config.hidden_size != HIDDEN_SIZE:
                raise AssertionError('the GRU width is not the frozen 256')
            if model.heads['gru'].input_size != GRU_INPUT_WIDTH:
                raise AssertionError('the GRU input width is not 285: an extra channel would mean a '
                                     'privileged field reached the policy input')
            initial_sha = r16.state_sha(model.heads.state_dict())
            if initial_sha != prereg['initialization']['per_seed'][str(seed)]:
                raise AssertionError('initial head state must reproduce the frozen table: ' + name)
            aux_sha = r16.state_sha(model.aux.state_dict())
            if aux_sha != prereg['initialization']['per_seed_aux'][str(seed)]:
                raise AssertionError('initial auxiliary state must match the pre-registration: ' + name)
            per_arm_initial[arm] = initial_sha
            weight = AUX_WEIGHT if arm == AUXILIARY else 0.
            r20.write_json(directory / 'initialization.json',
                           dict(head_state_sha256=initial_sha, aux_state_sha256=aux_sha,
                                seed=seed, arm=arm, aux_weight=weight, loss_arm=LOSS_ARM,
                                aux=model.describe(),
                                label=dict(task=AUX_TASK, rule=LABEL_RULE, labels=list(AUX_LABELS),
                                           reference='player x',
                                           round23_reference=REFERENCE_FRAME_CENTRE),
                                shared_config='both arms are N1 with the round20 control loss and the '
                                              'same auxiliary head from the same RNG state; only the '
                                              'auxiliary loss weight differs (A 0.0, X 0.5)',
                                feature_storage_shared=True, supervised_events=len(supervised),
                                aux_supervised_events=labelled, excluded_events=excluded))
            variant = {id(value) for value in model.aux.parameters()}
            clip_parameters = [value for value in model.parameters()
                               if value.requires_grad and id(value) not in variant]
            aux_parameters = [value for value in model.aux.parameters()]
            if sum(p.numel() for p in clip_parameters) != POLICY_HEADS_ACTIVE:
                raise AssertionError('the policy head must still be the 429200-parameter head')
            if any(id(value) in variant for value in clip_parameters):
                raise AssertionError('the auxiliary head must stay outside the clip')
            if sum(p.numel() for p in aux_parameters) != AUX_PARAMETERS:
                raise AssertionError('the auxiliary head must be the 514-parameter head')
            parameters = clip_parameters + aux_parameters
            initial_offline = r20.offline(model, supervised)
            optimizer = torch.optim.Adam(parameters, lr=r20.LR)
            losses, originals, aux_losses, aux_accuracies, aux_counts, aux_grads = [], [], [], [], [], []
            begin = time.perf_counter()
            for step in range(r20.UPDATES):
                model.train()
                entry = supervised[step % len(supervised)]
                event_id = entry['entry']['event_id']
                output, au_logits = infer_with_aux(model, entry)
                labels_batch = {key: value[None] for key, value in entry['labels'].items()}
                head = r20.arm_loss(LOSS_ARM, output, labels_batch)
                if not head['parts']['has_supervision']:
                    raise AssertionError('all 200 updates must be supervised')
                target, mask = targets.get(event_id, (None, None))
                if target is None:
                    target = torch.zeros((1, r20.DECISIONS), dtype=torch.long)
                    mask = torch.zeros((1, r20.DECISIONS), dtype=torch.bool)
                aux = r23.aux_loss(au_logits, target.to(device), mask.to(device))
                total = head['loss'] + weight * aux['loss']
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                if arm == AUXILIARY:
                    grad = model.aux.classifier.weight.grad
                    aux_grads.append(None if grad is None else float(grad.norm()))
                torch.nn.utils.clip_grad_norm_(clip_parameters, r20.CLIP)
                optimizer.step()
                losses.append(float(total.detach()))
                originals.append(float(head['original'].detach()))
                aux_losses.append(float(aux['loss'].detach()))
                aux_accuracies.append(aux['accuracy'])
                aux_counts.append(aux['count'])
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            final_offline = r20.offline(model, supervised)
            final_sha = r16.state_sha(model.heads.state_dict())
            final_aux_sha = r16.state_sha(model.aux.state_dict())
            steps_with_aux = [index for index, count in enumerate(aux_counts) if count]
            if arm == CONTROL and final_aux_sha != aux_sha:
                raise AssertionError('a zero-weight auxiliary term must leave its head untouched: ' + name)
            if arm == AUXILIARY and not steps_with_aux:
                raise AssertionError('the auxiliary arm must be supervised on some step')
            path = directory / 'research-heads.pt'
            r20.write_torch(path, dict(schema=HEADS_SCHEMA, arm=arm, seed=seed,
                                       config=asdict(model.tactical_config),
                                       baseline_sha256=file_sha(BASELINE),
                                       final_head_state_sha256=final_sha,
                                       final_aux_state_sha256=final_aux_sha,
                                       initial_aux_state_sha256=aux_sha,
                                       aux_weight=weight, loss_arm=LOSS_ARM,
                                       auxiliary_head=model.aux.describe(),
                                       heads={key: value.detach().cpu()
                                              for key, value in model.heads.state_dict().items()},
                                       aux_state={key: value.detach().cpu()
                                                  for key, value in model.aux.state_dict().items()}))
            report = dict(arm=arm, seed=seed, name=name, initial_head_state_sha256=initial_sha,
                          final_head_state_sha256=final_sha,
                          initial_aux_state_sha256=aux_sha, final_aux_state_sha256=final_aux_sha,
                          checkpoint_sha256=file_sha(path),
                          checkpoint_path=str(path.relative_to(dest)), updates=r20.UPDATES,
                          aux_weight=weight, loss_arm=LOSS_ARM, aux=model.describe(),
                          auxiliary_head=model.aux.describe(),
                          label=dict(task=AUX_TASK, rule=LABEL_RULE, labels=list(AUX_LABELS),
                                     reference='player x', supervised_events=labelled,
                                     supervised_rows=sum(int(targets[key][1].sum()) for key in labelled)),
                          clip_scope=dict(parameters=sum(p.numel() for p in clip_parameters),
                                          includes_auxiliary=False, same_in_both_arms=True,
                                          why='the 429200 policy-head parameters, identically in both '
                                              'arms; the 514 auxiliary parameters are outside the clip'),
                          matches_round23_control=(None if arm != CONTROL else
                                                   final_sha == R23_CONTROL_FINAL_HEAD_SHA[name]),
                          aux_unchanged_by_zero_weight=(arm == CONTROL and final_aux_sha == aux_sha),
                          losses=losses, original_losses=originals, aux_losses=aux_losses,
                          aux_accuracy=aux_accuracies, aux_supervised_counts=aux_counts,
                          aux_supervised_steps=len(steps_with_aux),
                          aux_supervised_step_first=steps_with_aux[0] if steps_with_aux else None,
                          aux_supervised_step_last=steps_with_aux[-1] if steps_with_aux else None,
                          original_loss_at_step0=originals[0], original_loss_final=originals[-1],
                          loss_at_step0=losses[0], loss_final=losses[-1],
                          aux_loss_first=(aux_losses[steps_with_aux[0]] if steps_with_aux else None),
                          aux_loss_final=(aux_losses[steps_with_aux[-1]] if steps_with_aux else None),
                          aux_accuracy_first=(aux_accuracies[steps_with_aux[0]]
                                              if steps_with_aux else None),
                          aux_accuracy_final=(aux_accuracies[steps_with_aux[-1]]
                                              if steps_with_aux else None),
                          aux_grad_norm=dict(
                              first=aux_grads[0] if aux_grads else None,
                              last=aux_grads[-1] if aux_grads else None,
                              minimum=min((v for v in aux_grads if v is not None), default=None),
                              maximum=max((v for v in aux_grads if v is not None), default=None),
                              nonzero_steps=sum(1 for v in aux_grads if v),
                              note='L2 norm of d(aux loss)/d(aux weight) after the backward pass, before '
                                   'the clip; arm A has no auxiliary gradient at all'),
                          optimization_seconds=seconds,
                          parameters=dict(active_head=sum(value.numel() for value in clip_parameters),
                                          active_auxiliary=sum(value.numel() for value in aux_parameters),
                                          active=sum(value.numel() for value in parameters),
                                          heads_total=sum(value.numel() for value in model.heads.parameters()),
                                          auxiliary_total=sum(value.numel() for value in model.aux.parameters()),
                                          total=sum(value.numel() for value in model.parameters()),
                                          same_in_both_arms=True),
                          supervised_events=[entry['entry']['event_id'] for entry in supervised],
                          aux_supervised_events=labelled,
                          excluded_events=excluded,
                          train_only='the validation events are never encoded, labelled or indexed',
                          initial_offline=initial_offline['summary'], final_offline=final_offline['summary'],
                          initial_offline_events=initial_offline['events'],
                          final_offline_events=final_offline['events'])
            r20.write_json(directory / 'training.json', report)
            reports[name] = report
            paths[name] = path
            print('trained', name, 'aux_weight', weight, 'init', initial_sha[:12],
                  'final', final_sha[:12], 'auxinit', aux_sha[:10], 'auxfinal', final_aux_sha[:10],
                  'loss0', round(losses[0], 4), 'lossN', round(losses[-1], 4),
                  'aux_loss', report['aux_loss_first'], report['aux_loss_final'],
                  'aux_acc', report['aux_accuracy_first'], report['aux_accuracy_final'],
                  'matches_r23', report['matches_round23_control'],
                  f'{seconds:.2f}s', flush=True)
        if len(set(per_arm_initial.values())) != 1:
            raise AssertionError('both arms must start from identical heads for seed ' + str(seed))
    if len(list((dest / 'train').rglob('research-heads.pt'))) != EXPECTED['checkpoints']:
        raise AssertionError('six checkpoints required')
    for name, path in paths.items():
        if file_sha(path) != reports[name]['checkpoint_sha256']:
            raise AssertionError('checkpoint SHA drift: ' + name)
    r20.write_json(dest / 'checkpoints.json',
                   {name: {key: value for key, value in reports[name].items()
                           if key not in ('initial_offline_events', 'final_offline_events',
                                          'aux_accuracy', 'aux_supervised_counts')}
                    for name in reports})
    return reports, paths


# ---------------------------------------------------------------- evaluation
PrivNetwork = r23.PrivNetwork


def evaluate_all(dest, prereg, backbone, device, reports, paths):
    return r23.evaluate_all(dest, prereg, backbone, device, reports, paths)


def control_equivalence(closed, evidence=R22_EVIDENCE):
    """Is arm A bit-for-bit the stored control arm of `evidence` on the same split and renderer?"""
    result = {}
    for seed in SEEDS:
        name = r20.name_of(CONTROL, seed)
        for split_name in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT):
            path = Path(evidence) / 'run' / 'closed-loop' / f'{name}-{split_name}.json'
            reference = json.loads(path.read_text(encoding='utf-8'))
            now, before = closed[(name, split_name)]['events'], reference['events']
            fields = ('event_id', 'feasible', 'survived', 'goal_satisfied', 'first_collision_frame',
                      'decisions', 'invalid_action')
            same_rows = all(all(a[key] == b[key] for key in fields) for a, b in zip(now, before))
            same_steps = all(
                [(step.get('candidate'), step.get('actual_action'), step.get('x_after'),
                  step.get('hit')) for step in a['steps']]
                == [(step.get('candidate'), step.get('actual_action'), step.get('x_after'),
                     step.get('hit')) for step in b['steps']]
                for a, b in zip(now, before))
            result[f'{name}/{split_name}'] = dict(events=len(now), same_rows=same_rows,
                                                  same_steps=same_steps)
    result['all_equal'] = all(value['same_rows'] and value['same_steps']
                              for value in result.values() if isinstance(value, dict))
    return result


def direction_table(closed, split_name, events):
    return r23.direction_table(closed, split_name, events)


def mechanism_block(rows):
    return r23.mechanism_block(rows)


def timing_cells(rows):
    return r23.timing_cells(rows)


def stored_actions(closed, split_name, events):
    return r23.stored_actions(closed, split_name, events)


# ---------------------------------------------------------------- probes
def trajectory_positions(steps):
    """The player's visible x on every decision of a stored closed-loop trajectory."""
    if len(steps) != r20.DECISIONS:
        raise ValueError('one stored decision per row required')
    return [float(step['x_before']) for step in steps]


def replay_pass(backbone, device, closed, prereg, paths, reports):
    """One replay per (network, F1 validation event), shared by every probe of this round."""
    validation_f1 = [row for row in event_records(prereg, r20.VALIDATION_SPLIT)
                     if row['family'] == 'F1']
    out, fidelity = {}, {}
    for name in r23.all_names():
        model = r23.load_variant(backbone, device, paths[name], reports[name])
        for entry in validation_f1:
            world = make_world(entry['scenario'])
            steps = next(row for row in closed[(name, r20.VALIDATION_SPLIT)]['events']
                         if row['event_id'] == entry['event_id'])['steps']
            replayed = r23.aux_replay(model, device, world, steps)
            fidelity[f'{name}/{entry["event_id"]}'] = round(
                replayed['stored_candidate_matches'] / replayed['steps'], 6)
            positions = trajectory_positions(steps)
            out[(name, entry['event_id'])] = dict(
                world=world, group=world.template_group,
                terms=aux_terms(world, positions), positions=positions,
                index=r23.preview_indices(world), states=replayed['states'],
                logits=replayed['logits'])
        print('replayed', name, 'fidelity', round(sum(value for key, value in fidelity.items()
                                                      if key.startswith(name + '/'))
                                                  / len(validation_f1), 6), flush=True)
    return out, fidelity


def seen_pass(backbone, device, data, reports, paths, prereg):
    return r23.seen_pass(backbone, device, data, reports, paths, prereg)


def aux_probe(closed, replay, seen, fidelity, prereg, reports):
    """K5: the auxiliary head's accuracy, balanced accuracy and 2x2 confusion, seen and held out.

    Two read-outs, neither of which ever feeds the label:
      * `seen` - the stored round21 training features of the six F1 train events: the rows the
        auxiliary loss saw, i.e. fit, not generalisation;
      * `holdout` - the six F1 validation events, replayed along each checkpoint's own stored
        trajectory. The label is scored against the sign at that trajectory's first preview row
        (the frozen rule), so the label of a cell is the label the checkpoint itself faced.
    """
    train_f1 = [row for row in event_records(prereg, r20.TRAIN_SPLIT) if row['family'] == 'F1']
    validation_f1 = [row for row in event_records(prereg, r20.VALIDATION_SPLIT)
                     if row['family'] == 'F1']
    result = dict(seen={}, holdout={}, replay_fidelity=fidelity, rule=LABEL_RULE)
    for split_name in ('seen', 'holdout'):
        for name in r23.all_names():
            truth, pred, frozen_truth = [], [], []
            per_event = {}
            entries = train_f1 if split_name == 'seen' else validation_f1
            for entry in entries:
                world = make_world(entry['scenario'])
                frozen = aux_terms(world, player_trace(world, teacher_of(prereg, entry['event_id'])))
                if split_name == 'seen':
                    positions = player_trace(world, teacher_of(prereg, entry['event_id']))
                else:
                    positions = replay[(name, entry['event_id'])]['positions']
                terms = aux_terms(world, positions)
                index = terms['row_indices']
                logits = (seen[(name, entry['event_id'])][index] if split_name == 'seen'
                          else replay[(name, entry['event_id'])]['logits'][index])
                predicted = int(logits.mean(0).argmax())
                correct = int((logits.argmax(-1) == terms['side']).sum())
                frozen_correct = int((logits.argmax(-1) == frozen['side']).sum())
                per_event[entry['event_id']] = dict(
                    side=terms['side'], label=terms['label'], predicted=side_name(predicted),
                    predicted_side=predicted, correct=correct, rows=int(len(index)),
                    accuracy=round(correct / len(index), 6),
                    frozen_side=frozen['side'], frozen_label=frozen['label'],
                    frozen_correct=frozen_correct,
                    frozen_accuracy=round(frozen_correct / len(index), 6),
                    own_label_equals_frozen=bool(terms['side'] == frozen['side']),
                    own_first_sign=terms['first_sign'], frozen_first_sign=frozen['first_sign'],
                    mean_probability_right=round(float(torch.softmax(torch.from_numpy(logits), -1)[:, 1]
                                                       .mean()), 6),
                    margin=round(float((logits[:, 1] - logits[:, 0]).mean()), 6),
                    gap_screen_centre=terms['gap_screen_centre'],
                    first_player_x=terms['first_player_x'], first_offset=terms['first_offset'],
                    label_would_flip=terms['would_flip'],
                    sign_flips=terms['sign_flips'],
                    replay_fidelity=(None if split_name == 'seen'
                                     else fidelity[f'{name}/{entry["event_id"]}']))
                truth.extend([terms['side']] * len(index))
                pred.extend(logits.argmax(-1).tolist())
                frozen_truth.extend([frozen['side']] * len(index))
            result[split_name][name] = dict(
                arm=reports[name]['arm'], seed=reports[name]['seed'],
                aux_weight=reports[name]['aux_weight'],
                per_event=per_event, overall=r23.aux_summary(truth, pred),
                overall_vs_frozen_teacher=r23.aux_summary(frozen_truth, pred),
                own_label_equals_frozen_teacher=all(row['own_label_equals_frozen']
                                                    for row in per_event.values()),
                differing_events=[key for key, row in per_event.items()
                                  if not row['own_label_equals_frozen']],
                predicts_one_class=(len({row['predicted_side'] for row in per_event.values()}) == 1))
            print('aux', split_name, name, 'accuracy', result[split_name][name]['overall']['accuracy'],
                  'balanced', result[split_name][name]['overall']['balanced_accuracy'],
                  'confusion', result[split_name][name]['overall']['confusion'],
                  'vs_frozen', result[split_name][name]['overall_vs_frozen_teacher']['accuracy'],
                  flush=True)
    result['by_arm'] = {}
    for arm in ARMS:
        result['by_arm'][arm] = {}
        for split_name in ('seen', 'holdout'):
            rows = [result[split_name][name]['overall'] for name in r23.names_of(arm)]
            frozen_rows = [result[split_name][name]['overall_vs_frozen_teacher']
                           for name in r23.names_of(arm)]
            result['by_arm'][arm][split_name] = dict(
                seeds=len(rows),
                accuracy=round(sum(row['correct'] for row in rows)
                               / sum(row['rows'] for row in rows), 6),
                balanced_accuracy=round(sum(row['balanced_accuracy'] for row in rows) / len(rows), 6),
                per_seed_accuracy=[row['accuracy'] for row in rows],
                per_seed_balanced_accuracy=[row['balanced_accuracy'] for row in rows],
                accuracy_vs_frozen_teacher=round(sum(row['correct'] for row in frozen_rows)
                                                 / sum(row['rows'] for row in frozen_rows), 6),
                balanced_accuracy_vs_frozen_teacher=round(
                    sum(row['balanced_accuracy'] for row in frozen_rows) / len(frozen_rows), 6),
                predicts_one_class=[result[split_name][name]['predicts_one_class']
                                    for name in r23.names_of(arm)],
                differing_events={name: result[split_name][name]['differing_events']
                                  for name in r23.names_of(arm)})
    result['design'] = dict(
        task=AUX_TASK, rule=LABEL_RULE, rule_text=LABEL_RULE_TEXT, labels=list(AUX_LABELS),
        supervision=AUX_SUPERVISION_WINDOW,
        rows='the preview-window decisions only, for every event',
        seen='the stored round21 training features of the six F1 train events: exactly the rows the '
             'auxiliary loss trained on, so this is fit',
        holdout='the six F1 validation events, each replayed along that checkpoint\'s own stored '
                'closed-loop trajectory, re-rendered from its own x trace; the label is the sign at '
                'that trajectory\'s first preview row and is never an input',
        chance='0.5 on the frozen teacher table (3 left / 3 right on both splits); a checkpoint whose '
               'own trajectory flips the sign would move its cell label, and that is reported '
               'per cell rather than hidden',
        label_use='the label exists only inside the training loss and inside this scoring function; '
                  '`PrivNetwork` (the closed loop) is the unmodified parent class and cannot read it')
    return result


def teacher_of(prereg, event_id):
    return [int(value) for value in prereg['teachers'][event_id]]


def state_probe(replay, fidelity, reports):
    """Is the same player-relative direction linearly decodable from the GRU state?"""
    result = {}
    for name in r23.all_names():
        rows = [dict(group=value['group'], side=float(value['terms']['side']),
                     states=value['states'][value['index']])
                for (network, _), value in replay.items() if network == name]
        groups = sorted({row['group'] for row in rows})
        folds = []
        for held in groups:
            train_rows = [row for row in rows if row['group'] != held]
            test_rows = [row for row in rows if row['group'] == held]
            x_train = np.concatenate([row['states'] for row in train_rows]).astype(np.float64)
            x_test = np.concatenate([row['states'] for row in test_rows]).astype(np.float64)
            y_train = np.concatenate([[row['side']] * len(row['states']) for row in train_rows])
            y_test = np.concatenate([[row['side']] * len(row['states']) for row in test_rows])
            fold = dict(held_out=held, rows_train=int(len(y_train)), rows_test=int(len(y_test)))
            for lam in r22.RIDGE['lambda_sweep']:
                fitted = r22.ridge_fit(x_train, y_train, lam)
                prediction = (r22.ridge_predict(fitted, x_test) > .5).astype(np.float64)
                fold[str(lam)] = dict(correct=int((prediction == y_test).sum()), rows=int(len(y_test)))
            folds.append(fold)
        sweep = {lam: round(sum(fold[lam]['correct'] for fold in folds)
                            / sum(fold[lam]['rows'] for fold in folds), 6)
                 for lam in [str(value) for value in r22.RIDGE['lambda_sweep']]}
        result[name] = dict(arm=reports[name]['arm'], seed=reports[name]['seed'],
                            groups=groups, folds=folds, lambda_sweep=sweep,
                            side_accuracy=sweep[str(r22.RIDGE['lambda_'])],
                            side_accuracy_min=min(sweep.values()),
                            side_accuracy_max=max(sweep.values()),
                            majority_baseline=0.5,
                            rows=sum(fold['rows_test'] for fold in folds))
    result['by_arm'] = {}
    for arm in ARMS:
        values = [result[name]['side_accuracy'] for name in r23.names_of(arm)]
        sweep_mean = [sum(result[name]['lambda_sweep'].values())
                      / len(result[name]['lambda_sweep']) for name in r23.names_of(arm)]
        result['by_arm'][arm] = dict(seeds=len(values),
                                     side_accuracy=round(sum(values) / len(values), 6),
                                     per_seed=values, minimum=min(values), maximum=max(values),
                                     sweep_mean=round(sum(sweep_mean) / len(sweep_mean), 6))
    result['reference'] = dict(
        note='round23 measured the same probe on ITS label (gap centre above 192) and got sweep means '
             '0.826 (A) / 0.619 (X); the target of this round is a different binary variable, so only '
             'the arm-to-arm comparison and the whole lambda sweep are read',
        round23_sweep_mean={'A': .826, 'X': .619})
    result['design'] = dict(
        target='binary: is the gap screen centre right of the player, at the first preview row',
        rows='the preview-window decisions of the six F1 validation events, from the replay',
        fit='ridge on standardised inputs with an intercept, leave one geometry group out',
        lambda_sweep=list(r22.RIDGE['lambda_sweep']), headline_lambda=r22.RIDGE['lambda_'],
        caveat='132 rows against 256 dimensions and a label that is constant within an event: the '
               'probe separates events as much as frames, so only the whole lambda sweep and the '
               'arm-to-arm comparison are read, never a single value')
    return result


def usage_table(closed, aux, prereg):
    """Per cell: what the auxiliary head read vs what the policy did, on the same trajectory."""
    rows = {}
    for name in r23.all_names():
        for entry in event_records(prereg, r20.VALIDATION_SPLIT):
            if entry['family'] != 'F1':
                continue
            world = make_world(entry['scenario'])
            stored = next(row for row in closed[(name, r20.VALIDATION_SPLIT)]['events']
                          if row['event_id'] == entry['event_id'])
            cell = r22.cell_diagnostic(world, stored, name, r20.VALIDATION_SPLIT)
            terms = aux_terms(world, trajectory_positions(stored['steps']))
            read = aux['holdout'][name]['per_event'][entry['event_id']]
            rows[f'{name}/{entry["event_id"]}'] = dict(
                network=name, arm=name.split('-')[0], seed=int(name.rsplit('-s', 1)[1]),
                event_id=entry['event_id'], mirror=bool(world.mirror),
                aux_true_side=terms['side'], aux_true_label=terms['label'],
                aux_required_action=('right' if terms['side'] else 'left'),
                aux_predicted_side=read['predicted_side'], aux_predicted_label=read['predicted'],
                aux_correct=bool(read['predicted_side'] == terms['side']),
                aux_row_accuracy=read['accuracy'], aux_margin=read['margin'],
                label_would_flip=terms['would_flip'], sign_flips=terms['sign_flips'],
                label_tie=bool(terms['first_sign'] == 0),
                policy_need=cell['need'], policy_direction=cell['direction'],
                policy_screen_direction=cell['screen_direction'],
                policy_verdict=cell['verdict'], in_gap_at_wall=cell['in_gap_at_wall'],
                signed_error_px=cell['signed_error_px'], mechanism=cell['mechanism'],
                first_move_frame=cell['first_move_frame'], entered_gap=cell['entered_gap'],
                hit_frame=cell['hit_frame'], wall_frame=cell['wall_frame'])
    counts = {}
    for arm in ARMS:
        subset = [row for row in rows.values() if row['arm'] == arm]
        mirror = [row for row in subset if row['mirror']]
        base = [row for row in subset if not row['mirror']]
        counts[arm] = dict(
            cells=len(subset),
            aux_correct_cells=sum(1 for row in subset if row['aux_correct']),
            aux_correct_and_toward=sum(1 for row in subset
                                       if row['aux_correct'] and row['policy_verdict'] == 'toward'),
            aux_correct_and_away=sum(1 for row in subset
                                     if row['aux_correct'] and row['policy_verdict'] == 'away'),
            aux_correct_and_hold=sum(1 for row in subset
                                     if row['aux_correct'] and row['policy_verdict'] == 'hold'),
            aux_says_left=sum(1 for row in subset if row['aux_predicted_side'] == 0),
            aux_says_right=sum(1 for row in subset if row['aux_predicted_side'] == 1),
            policy_screen_left=sum(1 for row in subset if row['policy_screen_direction'] == 'left'),
            policy_screen_right=sum(1 for row in subset if row['policy_screen_direction'] == 'right'),
            policy_screen_hold=sum(1 for row in subset if row['policy_screen_direction'] == 'hold'),
            in_gap_at_wall=sum(1 for row in subset if row['in_gap_at_wall']),
            base_cells=len(base), base_aux_correct=sum(1 for row in base if row['aux_correct']),
            base_toward=sum(1 for row in base if row['policy_verdict'] == 'toward'),
            mirror_cells=len(mirror), mirror_aux_correct=sum(1 for row in mirror if row['aux_correct']),
            mirror_toward=sum(1 for row in mirror if row['policy_verdict'] == 'toward'),
            mirror_away=sum(1 for row in mirror if row['policy_verdict'] == 'away'),
            mirror_aux_correct_and_away=sum(1 for row in mirror
                                            if row['aux_correct'] and row['policy_verdict'] == 'away'),
            toward_but_late=sum(1 for row in subset
                                if row['policy_verdict'] == 'toward' and not row['entered_gap']),
            label_flipped_cells=sum(1 for row in subset if row['label_would_flip']),
            label_tie_cells=sum(1 for row in subset if row['label_tie']))
    return dict(cells=rows, by_arm=counts,
                note='one row per (network, F1 validation event); `aux_*` is the auxiliary head on the '
                     'replayed pixels, `policy_*` is the stored closed loop of the same checkpoint; a '
                     'high aux correctness with a low policy toward count is the "the task was learned, '
                     'the behaviour did not use it" reading, cell by cell')


def label_probe(prereg, replay, reports, agreement):
    """The per-cell label read-out: frozen teacher label vs the one the checkpoint's own trace gives."""
    rows = {}
    for name in r23.all_names():
        for entry in event_records(prereg, r20.VALIDATION_SPLIT):
            if entry['family'] != 'F1':
                continue
            world = make_world(entry['scenario'])
            frozen = aux_terms(world, player_trace(world, teacher_of(prereg, entry['event_id'])))
            own = replay[(name, entry['event_id'])]['terms']
            rows[f'{name}/{entry["event_id"]}'] = dict(
                network=name, event_id=entry['event_id'], mirror=bool(world.mirror),
                frozen_side=frozen['side'], frozen_label=frozen['label'],
                own_side=own['side'], own_label=own['label'],
                same=bool(frozen['side'] == own['side']),
                own_first_player_x=own['first_player_x'], frozen_first_player_x=frozen['first_player_x'],
                own_first_offset=own['first_offset'], frozen_first_offset=frozen['first_offset'],
                own_sign_flips=own['sign_flips'], frozen_sign_flips=frozen['sign_flips'],
                own_first_tie=bool(own['first_sign'] == 0),
                frozen_first_tie=bool(frozen['first_sign'] == 0))
    by_arm = {}
    for arm in ARMS:
        subset = [row for row in rows.values() if row['network'] in r23.names_of(arm)]
        by_arm[arm] = dict(cells=len(subset), same=sum(1 for row in subset if row['same']),
                           differing=[row['event_id'] for row in subset if not row['same']],
                           own_flip_cells=sum(1 for row in subset if row['own_sign_flips']),
                           own_tie_cells=sum(1 for row in subset if row['own_first_tie']),
                           frozen_tie_cells=sum(1 for row in subset if row['frozen_first_tie']))
    return dict(cells=rows, by_arm=by_arm, agreement=agreement,
                note='the holdout label is a function of the checkpoint\'s own visible x, so it is '
                     'recomputed per checkpoint; this table is the measurement of how far that label '
                     'drifted from the frozen teacher trajectory it was defined on')


def run_probes(backbone, device, closed, prereg, records, data, paths, reports, gaps, agreement):
    replay, fidelity = replay_pass(backbone, device, closed, prereg, paths, reports)
    seen = seen_pass(backbone, device, data, reports, paths, prereg)
    aux = aux_probe(closed, replay, seen, fidelity, prereg, reports)
    state = state_probe(replay, fidelity, reports)
    usage = usage_table(closed, aux, prereg)
    labels = label_probe(prereg, replay, reports, agreement)
    stored = stored_actions(closed, r20.VALIDATION_SPLIT,
                            [row for row in event_records(prereg, r20.VALIDATION_SPLIT)
                             if row['family'] == 'F1'])
    return dict(aux=aux, state=state, usage=usage, labels=labels,
                prediction=dict(validation=stored,
                                train=stored_actions(closed, r20.TRAIN_SPLIT,
                                                     [row for row in event_records(prereg, r20.TRAIN_SPLIT)
                                                      if row['family'] == 'F1']),
                                note='the masked argmax actually taken, from the stored rows; per event '
                                     'over all 80 decisions and over the telegraph-window decisions only'),
                base_bias=dict(note='the frozen baseline (B0) is the policy head the residual offset is '
                                    'built on; its screen preference on the identical renders is the prior '
                                    'this round is testing against', validation=stored['B0']),
                timing=dict(validation=timing_cells(gaps[r20.VALIDATION_SPLIT]),
                            train=timing_cells(gaps[r20.TRAIN_SPLIT])),
                replay_note='every replay reports its own fidelity: the fraction of steps on which the '
                            'rebuilt stack reproduces the stored masked argmax')


# ---------------------------------------------------------------- the judgement rule
def rule_verdict(higher, lower, k1, k2, k2_all):
    """The frozen ladder, as a pure function of the counts (so it can be unit tested)."""
    if higher == 0 and lower == 0:
        return 'unchanged'
    if k1:
        return 'improved' if (k2 > 0 and (k2_all is None or k2_all >= 0)) else 'mixed'
    if lower and not higher:
        return 'worse'
    if lower and higher:
        return 'not_improved'
    return 'inconclusive'


def judgement(f1_control, f1_auxiliary, all_control=None, all_auxiliary=None):
    """K1/K2 and the verdict rule that replaces round23's "any seed larger = improved".

    K1 = F1 holdout feasible is strictly higher on at least two of the three seeds and lower on
    none. K2 = the two pooled feasible counts (F1 and all-family) over the three seeds. X is
    `improved` iff K1 holds **and** neither pooled count falls; if K1 holds and either falls the
    verdict is `mixed` and the word improved is not used.
    """
    control = [int(value) for value in f1_control]
    auxiliary = [int(value) for value in f1_auxiliary]
    if len(control) != len(auxiliary) or not control:
        raise ValueError('the paired F1 vectors must have the same non-zero length')
    delta = [auxiliary[index] - control[index] for index in range(len(control))]
    higher = sum(1 for value in delta if value > 0)
    lower = sum(1 for value in delta if value < 0)
    equal = sum(1 for value in delta if value == 0)
    k1 = higher >= EXPECTED['judgement_higher_needed'] and lower == 0
    k2 = sum(auxiliary) - sum(control)
    if all_control is None or all_auxiliary is None:
        k2_all = None
    else:
        if len(all_control) != len(control):
            raise ValueError('the all-family vectors must be aligned with the F1 ones')
        k2_all = sum(int(value) for value in all_auxiliary) - sum(int(value) for value in all_control)
    verdict = rule_verdict(higher, lower, k1, k2, k2_all)
    return dict(
        K1=dict(rule='F1 holdout feasible strictly higher on at least %d of %d seeds and lower on none'
                     % (EXPECTED['judgement_higher_needed'], len(control)),
                control=control, auxiliary=auxiliary, per_seed_delta=delta,
                higher=higher, equal=equal, lower=lower, satisfied=bool(k1)),
        K2=dict(rule='pooled feasible over the seeds; improved requires neither pooled count to fall',
                f1=dict(control=sum(control), auxiliary=sum(auxiliary), delta=k2),
                all_family=(None if k2_all is None else
                            dict(control=sum(int(v) for v in all_control),
                                 auxiliary=sum(int(v) for v in all_auxiliary), delta=k2_all)),
                satisfied=bool(k2 > 0 and (k2_all is None or k2_all >= 0))),
        verdict=verdict,
        rule='improved iff K1 and K2 (both pooled counts do not fall); K1 with a falling pooled count '
             'is mixed, never improved; the round23 rule "any seed larger = improved" is retired',
        replaces=dict(round='round23', rule='any seed strictly greater = improved',
                      deltas=R23_DELTAS, verdict_under_old_rule=R23_VERDICT,
                      verdict_under_this_rule=rule_verdict(1, 1, False, -3, -3),
                      note='round23 recorded -1/+1/0 and a lower total; under K1 that is not improved'))


def k_report(closed, mechanism, probes, reference=None):
    """K1..K5 in one block, computed from the frozen artifacts only."""
    reference = r23_reference() if reference is None else reference
    round23_mirror_correct_and_away = reference['mirror_aux_correct_and_away'][AUXILIARY]
    control = [closed[(r20.name_of(CONTROL, seed), r20.VALIDATION_SPLIT)]['families']['F1']['feasible']
               for seed in SEEDS]
    auxiliary = [closed[(r20.name_of(AUXILIARY, seed), r20.VALIDATION_SPLIT)]['families']['F1']['feasible']
                 for seed in SEEDS]
    all_control = [closed[(r20.name_of(CONTROL, seed), r20.VALIDATION_SPLIT)]['families']['all']['feasible']
                   for seed in SEEDS]
    all_auxiliary = [closed[(r20.name_of(AUXILIARY, seed), r20.VALIDATION_SPLIT)]['families']['all']['feasible']
                     for seed in SEEDS]
    block = judgement(control, auxiliary, all_control, all_auxiliary)
    groups = {arm: mechanism[arm]['groups'] for arm in ARMS}
    block['K3'] = dict(
        rule='mirror-group toward rate on the F1 holdout, canonical motion toward the gap',
        mirror={arm: dict(toward=groups[arm]['mirror']['toward'],
                          cells=groups[arm]['mirror']['cells'],
                          toward_rate=groups[arm]['mirror']['toward_rate']) for arm in ARMS},
        base={arm: dict(toward=groups[arm]['base']['toward'],
                        cells=groups[arm]['base']['cells'],
                        toward_rate=groups[arm]['base']['toward_rate']) for arm in ARMS},
        delta={group: groups[AUXILIARY][group]['toward_rate'] - groups[CONTROL][group]['toward_rate']
               for group in ('base', 'mirror')})
    block['K4'] = dict(
        rule='in-gap at wall_frame-1 and the signed parking error in canonical px',
        in_gap={group: {arm: groups[arm][group]['in_gap'] for arm in ARMS}
                for group in ('base', 'mirror')},
        parked_px={group: {arm: dict(mean_abs=groups[arm][group]['parked_px']['mean_abs'],
                                     maximum_abs=groups[arm][group]['parked_px']['maximum_abs'])
                           for arm in ARMS} for group in ('base', 'mirror')},
        cells={arm: sum(groups[arm][group]['cells'] for group in ('base', 'mirror')) for arm in ARMS})
    holdout = probes['aux']['by_arm']
    block['K5'] = dict(
        rule='auxiliary head on the held-out geometry: per-seed accuracy, balanced accuracy and '
             'the 2x2 confusion; "learned" requires all three seeds above 0.5 balanced accuracy',
        holdout={arm: dict(accuracy=holdout[arm]['holdout']['accuracy'],
                           balanced_accuracy=holdout[arm]['holdout']['balanced_accuracy'],
                           per_seed_accuracy=holdout[arm]['holdout']['per_seed_accuracy'],
                           per_seed_balanced_accuracy=holdout[arm]['holdout']['per_seed_balanced_accuracy'])
                 for arm in ARMS},
        seen={arm: dict(accuracy=holdout[arm]['seen']['accuracy'],
                        balanced_accuracy=holdout[arm]['seen']['balanced_accuracy'],
                        per_seed_accuracy=holdout[arm]['seen']['per_seed_accuracy'])
              for arm in ARMS},
        holdout_vs_frozen_teacher={arm: dict(
            accuracy=holdout[arm]['holdout']['accuracy_vs_frozen_teacher'],
            balanced_accuracy=holdout[arm]['holdout']['balanced_accuracy_vs_frozen_teacher'])
            for arm in ARMS},
        one_class_heads={arm: holdout[arm]['holdout']['predicts_one_class'] for arm in ARMS},
        own_label_differs_from_frozen={arm: holdout[arm]['holdout']['differing_events']
                                       for arm in ARMS},
        confusion={name: probes['aux']['holdout'][name]['overall']['confusion']
                   for name in r23.all_names()},
        learned=(all(value > .5 for value in
                     holdout[AUXILIARY]['holdout']['per_seed_balanced_accuracy'])
                 if len(holdout[AUXILIARY]['holdout']['per_seed_balanced_accuracy']) == len(SEEDS)
                 else False),
        round23_reference=dict(holdout=R23_AUX_HOLDOUT,
                               mirror_aux_correct=R23_USAGE[AUXILIARY]['mirror_aux_correct'],
                               mirror_away=R23_USAGE[AUXILIARY]['mirror_away'],
                               mirror_aux_correct_and_away=round23_mirror_correct_and_away))
    usage = probes['usage']['by_arm']
    block['learned_but_unused'] = dict(
        rule='the round23 comparison this round must report: mirror cells where the auxiliary head '
             'read the required direction but the policy moved away',
        round23_mirror_aux_correct=R23_USAGE[AUXILIARY]['mirror_aux_correct'],
        round23_mirror_aux_correct_and_away=round23_mirror_correct_and_away,
        this_round_mirror_aux_correct=usage[AUXILIARY]['mirror_aux_correct'],
        this_round_mirror_aux_correct_and_away=usage[AUXILIARY]['mirror_aux_correct_and_away'],
        this_round_mirror_away=usage[AUXILIARY]['mirror_away'],
        this_round_mirror_cells=usage[AUXILIARY]['mirror_cells'],
        control_mirror_aux_correct=usage[CONTROL]['mirror_aux_correct'],
        control_mirror_aux_correct_and_away=usage[CONTROL]['mirror_aux_correct_and_away'],
        reading='if the head reads the required direction and the policy still moves away, the label '
                'definition was not the binding constraint: the coupling is; if the head stops reading '
                'it, the label itself is not learnable from the GRU state at this budget')
    return block


# ---------------------------------------------------------------- pre-registration
def criterion():
    """The frozen decision rules of this round; any drift aborts the run."""
    return dict(
        round='round24',
        purpose='one pre-registered intervention on top of round23: change the privileged auxiliary '
                'label from "the gap is on which side of the playfield centre" to "the gap is on '
                'which side OF THE PLAYER", i.e. from a position fact to the action sign that closes '
                'the distance, and read the result out with a judgement rule that can say "worse"',
        intervention=dict(
            change='the auxiliary label only: sign(gap screen centre - player screen x) instead of '
                   'gap screen centre > 192; the head, its placement, the loss, the weight, the arms, '
                   'the seeds, the updates, the data and the split are round23\'s',
            task=AUX_TASK,
            label=dict(source='`round21.telegraph_gap(s)` for the gap (the privileged world parameter, '
                              'exactly the arithmetic the R21 renderer draws the band with) and the '
                              'player screen x, which `render_state` draws at 32+state.x and `advance` '
                              'moves by 2*action: so the label is the sign of the action that reduces '
                              'the distance to the slot',
                       reference='the player x, not the playfield centre',
                       round23_reference=REFERENCE_FRAME_CENTRE,
                       classes=list(AUX_LABELS),
                       availability='training loss and scoring only; never an input, never a history '
                                    'field, never a pixel region, never a tensor of the policy head',
                       player_x_source='the frozen round19 teacher_v2 action list replayed through the '
                                       'unmodified `advance` (train), or the checkpoint\'s own stored '
                                       'x_before (scoring); both are the visible state'),
            supervision=dict(window=AUX_SUPERVISION_WINDOW, rule=LABEL_RULE, rule_text=LABEL_RULE_TEXT,
                             why='outside the telegraph window the band is not drawn yet, so the label '
                                 'would ask the head to guess rather than to read; F2/F3 worlds have '
                                 '`internal_gap = False` and therefore no label at all',
                             flips='per-row signs, flip rows and flip frames are recorded for every '
                                   'event and reported (labels.json, probes.json); the per-row variant '
                                   'is NOT tried in this round',
                             labelled_events='the six F1 train events (3 groups x base/mirror)',
                             unlabelled_events='the eight F2/F3 train events contribute an exactly-zero '
                                               'auxiliary term'),
            head=dict(module='Linear(256, 2, bias=True)', parameters=AUX_PARAMETERS,
                      placement='outside `heads`, unchanged from round23'),
            loss='L = L_control + w * CE(aux_logits, side);  w = 0.0 for arm A, 0.5 for arm X',
            weight=AUX_WEIGHT,
            gradient_path='the auxiliary term reaches only the auxiliary head and the shared GRU',
            head_class='round23\'s PrivAuxPolicy, reused: no line of TacticalPolicy is overridden',
            state_source='forward pre-hook on heads["%s"]' % STATE_HOOK_MODULE,
            not_changed='worlds, world parameters, R19 split, round21 renderer and pixels, teacher '
                        'actions, tactical labels, pixel packing, encoder, control loss, optimizer, '
                        'learning rate, clip value, clip scope, sequence, batch, update count, seed '
                        'set, closed-loop runner, condition, masking, head, weight, supervision window',
            parameter_increment='X and A are identical objects: 429200 policy-head parameters + 514 '
                                'auxiliary parameters = 429714 active; the arms differ only in w'),
        data=dict(source='round21 run/records.json assets, reopened read-only and SHA-bound per event; '
                         'nothing re-rendered, no teacher executed, no tactical label created',
                  features='round22.spatial_vectors reused verbatim; its spatial digest must recompute '
                           'to round22\'s frozen 83a75f73... so "the head sees round22\'s features" is a '
                           'measured identity, not a copy',
                  teacher='round19 run/worlds/<family>/<event>.json teacher_v2 action lists; for the 14 '
                          'train events they are asserted equal to round21\'s stored rollout actions',
                  train=dict(events=14, groups=7, groups_per_family={'F1': 3, 'F2': 1, 'F3': 3},
                             use='the only events encoded or trained on'),
                  validation=dict(events=14, groups=7, groups_per_family={'F1': 3, 'F2': 1, 'F3': 3},
                                  use='read-only: never trained, never encoded into the training set; '
                                      'closed loop and probes only. The auxiliary probe reads the label '
                                      'at that checkpoint\'s own first preview row and uses it only to '
                                      'score'),
                  bindings='pixels.npz, history.pt, labels.pt and rollout.json must equal round21\'s '
                           'frozen SHA table for all 14 train events, otherwise the run aborts'),
        training=dict(arms={CONTROL: 'the same auxiliary head with weight 0.0: the round23 control',
                            AUXILIARY: 'the same auxiliary head with weight 0.5'},
                      loss_arm=LOSS_ARM, aux_weight=AUX_WEIGHT, seeds=list(SEEDS), updates=r20.UPDATES,
                      optimizer='Adam', lr=r20.LR, clip=r20.CLIP,
                      clip_scope='the 429200 policy-head parameters, identically in both arms',
                      sequence=r20.SEQUENCE, batch=r20.BATCH,
                      precision='FP32 without AMP, autocast disabled, matmul TF32 False, cuDNN TF32 False, '
                                'float32 matmul precision highest',
                      loop='round23.train_all with one change: the target/mask pair per event comes from '
                           'sign(gap centre - player x) instead of gap centre > 192',
                      initialization='fresh seed reset per construction; the head is built first, so the '
                                     'policy head SHA must equal the frozen R16/R20/R21/R22/R23 table for '
                                     'the seed; the auxiliary head is built afterwards from the same RNG '
                                     'state in both arms, so its SHA must be identical in both arms',
                      control_reproduction='arm A must reproduce round23\'s A checkpoint SHA and every '
                                           'closed-loop row bitwise; if it does not, that is reported as '
                                           '`control_not_reproduced` and the causal claim is withheld',
                      selection='final step only: no early stopping, no validation-based selection, no '
                                'tuning; the auxiliary weight is fixed at 0.5 before the run'),
        evaluation=dict(
            order='all six checkpoints written and SHA-frozen first; then the six R18 baselines, then '
                  'B0, then the six checkpoints, validation before train; then the probes',
            executor='round21.closed_loop reused verbatim through round23, so the evaluation code is '
                     'round21 code and not a copy; the network is a plain TacticalPolicy',
            condition=r20.CONDITION, renderer='v373_telegraph.render_state_telegraph',
            b0='the frozen baseline policy logits on the identical telegraph renders',
            six_baselines='round20.six_baselines re-executed on the same worlds; the 168 rows must equal '
                          'round22\'s',
            holdout_rule='if ANY of the six R18 baselines is feasible on ANY validation event the R18 '
                         'filter has failed on the holdout side: report it verbatim and withhold the '
                         'learning conclusion',
            definitions=dict(
                direction='round21.gap_trace, canonical displacement start -> wall_frame-1',
                toward='the canonical direction points from the start at the gap\'s canonical centre',
                parked='`signed_error_px` at wall_frame-1 in canonical px: 0 inside [gap0, gap1]',
                label='sign(gap screen centre - player screen x) at the first preview row, where '
                      'player x is `render_state`\'s own x and `advance` moves it by 2*action',
                agreement='the label against the teacher\'s own first action (episode decision 0) and '
                          'against the teacher\'s action at the anchoring row; a hold action is counted '
                          'separately and excluded from the rate',
                aux_accuracy='the auxiliary head on the preview-window rows of the six F1 validation '
                             'events, replayed along each checkpoint\'s own stored trajectory'),
            primary_metrics=[
                'K1: F1 holdout feasible per seed, X minus A',
                'K2: pooled feasible over the three seeds, F1 and all-family',
                'K3: mirror-group toward rate (and base), per arm, with the per-seed split',
                'K4: cells inside the gap at wall_frame-1 and the signed parking error in px',
                'K5: the auxiliary head\'s held-out accuracy, balanced accuracy and 2x2 confusion',
                'the label diagnostics: the per-row sign/flip table, the label-vs-teacher agreement on '
                'both splits, and the count of mirror cells where the head read right and the policy '
                'went the other way'],
            probe_caveats=dict(
                aux_accuracy='the frozen teacher labels are balanced 3/3 on both splits, so chance is 0.5; '
                             'a checkpoint whose own trajectory flips the sign moves its cell label, and '
                             'that is reported per cell rather than hidden',
                label_leakage='both the gap and the player are visible in the telegraph window, so the '
                              'auxiliary task adds no information that is not in the pixels - it adds a '
                              'target. What is privileged is the label, not the content',
                decodability='132 rows against 256 dimensions with a label constant within an event: only '
                             'the whole lambda sweep and the arm-to-arm comparison are read',
                replay='the replay re-renders each checkpoint\'s own x trace; its fidelity is reported '
                       'for every cell'),
            reporting='per family, never only aggregated; n=3 seeds; F2 validation has one group; mirrors '
                      'are not independent samples; every F1 cell is listed',
            degeneracy='single_candidate and switches<=1 are counted, listed and reported'),
        judgement=dict(
            K1='F1 holdout feasible is strictly higher on at least two of the three seeds AND lower on '
               'none',
            K2='the pooled feasible counts over the three seeds: F1 and all-family; improved requires '
               'neither to fall',
            K3='mirror-group toward rate, reported as counts and rates, never as significance',
            K4='cells inside the gap at the wall and the signed parking error in px',
            K5='the auxiliary head\'s held-out balanced accuracy per seed; "learned" requires all three '
               'seeds above 0.5',
            rule='X is improved over A iff K1 holds and K2 does not fall; K1 with a falling pooled count '
                 'is `mixed` and the word improved is not used; K1 false is one of `unchanged` (all '
                 'equal), `not_improved` (both a rise and a fall), `worse` (no rise) or `inconclusive` '
                 '(exactly one rise, no fall)',
            replaces='round23\'s rule "any seed strictly greater = improved", which scored -1/+1/0 as '
                     'improved while the pooled total fell from 9 to 6',
            mixed='if K1 holds but K2 falls the round reports `mixed`; mechanism counters are reported '
                  'next to it so the two are never read apart',
            honesty='if the mirror direction does not improve while the auxiliary head is accurate, the '
                    'round must report the "the task was learned, the policy did not use it" reading with '
                    'its cell-level evidence and compare its mirror cell count with round23\'s 8/9; if '
                    'the label\'s agreement with the teacher is low, the report must state that this '
                    'label cannot be the ground truth for "which way to move" and bound the conclusion',
            aux_learned='the auxiliary task counts as learned on the holdout only if the balanced accuracy '
                        'exceeds 0.5 for all three seeds; a single seed above chance is reported as one '
                        'seed, not as learned'),
        claims=dict(significance='none: 3 seeds, 7 validation groups, at most 14 independent geometries; '
                                 'descriptive only',
                    gate='no formal gate opened, no deployable checkpoint produced, no game executed',
                    can_show='whether replacing a position-fact label with the action-sign label changes '
                             'the F1 holdout direction structure, and whether the residual failure is the '
                             'label definition or the coupling between the GRU state and the policy head',
                    cannot_show='tactical understanding, learnability beyond these 7 held-out geometries, '
                                'or that a privileged label at training time is a legitimate route to a '
                                'deployable policy'),
        failure_policy='first failure retained; one run; no retrain, rerun, repair of numbers, extra epoch, '
                       'tuning or baseline switch')


def criterion_sha(value=None):
    return hashlib.sha256(json.dumps(criterion() if value is None else value,
                                     sort_keys=True).encode()).hexdigest()


def identity_paths():
    return r23.identity_paths() + [
        Path(__file__), Path('tests/test_v373_player_relative_aux.py'),
        R23_EVIDENCE / 'preregistration.json', R23_EVIDENCE / 'run/summary.json',
        R23_EVIDENCE / 'run/checkpoints.json', R23_EVIDENCE / 'run/probes.json',
        R23_EVIDENCE / 'run/f1-gap.json', R23_EVIDENCE / 'run/records.json',
        REPORTS / 'P4-privaux-round23.md',
        R19_EVIDENCE / 'run/summary.json'] + \
        [R19_EVIDENCE / 'run' / 'worlds' / 'F1' / f'{event_id}.json' for event_id in F1_EVENTS]


def identities():
    return {str(path): file_sha(path) for path in identity_paths()}


def world_table(events):
    return r22.world_table(events)


def all_events():
    manifest = r20.load_worlds()
    split = r20.load_split(manifest)
    train = r20.world_events(manifest, split, r20.TRAIN_SPLIT)
    validation = r20.world_events(manifest, split, r20.VALIDATION_SPLIT)
    return manifest, split, train, validation


def f1_events():
    """The twelve F1 events of the frozen split, train first, in manifest order."""
    _, _, train, validation = all_events()
    return [row['event_id'] for row in train + validation if row['family'] == 'F1']


# The twelve F1 events are a frozen list: the module needs it for `identity_paths`, so it is read
# once at import time from the frozen manifest (which is itself SHA-frozen).
F1_EVENTS = f1_events()


def preregister(out=OUT, device='cuda'):
    out = r20.new_dir(out)
    manifest, split, train, validation = all_events()
    checks = r20.split_checks(manifest, split, train, validation)
    initial = r16.initial_head_shas()
    if len(set(initial.values())) != len(SEEDS):
        raise ValueError('different seeds must give different initial head states')
    if sorted(r23.r22_assets()) != sorted(row['event_id'] for row in train):
        raise ValueError('round22 assets must cover exactly the 14 train events')
    reference = r23_reference()
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('real CUDA required for the frozen feature statistics')
    backbone, _ = load_vision_checkpoint(BASELINE, map_location='cpu')
    backbone.to(device).eval()
    torch.set_float32_matmul_precision('highest')
    world = dict(manifest=manifest['sha256'], split=split['sha256'],
                 train_worlds_digest=r16.worlds_digest(train),
                 validation_worlds_digest=r16.worlds_digest(validation),
                 all_worlds_digest=r16.worlds_digest(train + validation),
                 train_events=[row['event_id'] for row in train],
                 validation_events=[row['event_id'] for row in validation],
                 train_scenario_sha256={row['event_id']: row['scenario_sha256'] for row in train},
                 validation_scenario_sha256={row['event_id']: row['scenario_sha256'] for row in validation})
    with _fp32_mode(torch.device(device)), torch.inference_mode():
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise AssertionError('TF32 must be disabled')
        records = train_records(dict(worlds=world))
        data, vectors, _ = encode(backbone, records, device)
        digest = r22.spatial_digest(vectors)
        statistics = r22.spatial_table(vectors)
        norms = feature_norms(data)
    if digest != r23.r22_reference()['spatial_sha256']:
        raise ValueError('the encode pass is not round22\'s: spatial digest drift')
    if norms != r23.r22_reference()['feature_norm']:
        raise ValueError('the frozen feature norm table drifted from round22')
    teacher, verdicts = teacher_table(train + validation)
    if any(value != 'feasible' for value in verdicts.values()):
        raise ValueError('the frozen teacher table must verify every world: ' + json.dumps(verdicts))
    train_labels = label_rows(train, teacher)
    validation_labels = label_rows(validation, teacher)
    supervision = supervision_table(records, teacher)
    labelled_train = {key: value for key, value in supervision.items() if value['target_defined']}
    labelled_validation = {key: value for key, value in supervision_table(validation, teacher).items()
                           if value['target_defined']}
    if len(labelled_train) != EXPECTED['aux_events'] or len(labelled_validation) != EXPECTED['aux_events']:
        raise AssertionError('six F1 events per split carry the auxiliary label')
    if sum(value['supervised_rows'] for value in labelled_train.values()) != EXPECTED['aux_train_rows']:
        raise AssertionError('128 supervised auxiliary rows expected on the train split')
    if sum(value['supervised_rows'] for value in labelled_validation.values()) != \
            EXPECTED['aux_validation_rows']:
        raise AssertionError('132 preview-window rows expected on the validation split')
    flips = flip_table(dict(train_labels, **validation_labels))
    if flips['event_count'] != EXPECTED['aux_events'] * 2:
        raise AssertionError('twelve labelled events expected')
    agreement = agreement_table(train + validation, teacher)
    collinearity = round23_collinearity(train + validation, teacher)
    aux_per_seed = r23.per_seed_aux_shas(backbone)
    if sorted(aux_per_seed) != sorted(str(seed) for seed in SEEDS):
        raise AssertionError('one auxiliary initial state per seed required')
    value = dict(
        schema=PREREG_SCHEMA, identities=identities(), criterion=criterion(),
        criterion_sha256=criterion_sha(),
        arms=list(ARMS), seeds=list(SEEDS), updates=r20.UPDATES, lr=r20.LR, clip=r20.CLIP,
        sequence=r20.SEQUENCE, batch=r20.BATCH, loss_arm=LOSS_ARM, aux_weight=AUX_WEIGHT,
        sources=dict(manifest=dict(path=str(r20.MANIFEST), sha256=manifest['sha256'],
                                   schema=r20.MANIFEST_SCHEMA),
                     split=dict(path=str(r20.SPLIT), sha256=split['sha256'], schema=r20.SPLIT_SCHEMA),
                     baseline=dict(path=str(BASELINE), sha256=file_sha(BASELINE)),
                     round19=dict(root=str(R19_EVIDENCE / 'run' / 'worlds'),
                                  teacher='teacher_v2',
                                  events={event_id: file_sha(R19_EVIDENCE / 'run' / 'worlds'
                                                             / ('F1' if event_id.split('-')[1] == 'F1'
                                                                else event_id.split('-')[1])
                                                             / f'{event_id}.json')
                                          for event_id in world['train_events'] + world['validation_events']}),
                     round23=dict(preregistration=reference['preregistration_sha256'],
                                  records=reference['records_sha256'],
                                  summary=reference['summary_sha256'],
                                  checkpoints=reference['checkpoints_sha256'],
                                  probes=reference['probes_sha256'],
                                  f1_gap=reference['f1_gap_sha256'])),
        worlds=world,
        split=dict(rule=split['spec']['rule'], counts=split['spec']['counts'], assertions=checks,
                   groups=[dict(group=group['group'], family=group['family'], split=group['split'],
                                index_in_family=group['index_in_family'], signature=group['signature'],
                                events=group['events']) for group in split['groups']]),
        auxiliary=dict(task=AUX_TASK, label_rule=LABEL_RULE, label_rule_text=LABEL_RULE_TEXT,
                       labels=list(AUX_LABELS),
                       reference='player x', round23_reference_point=REFERENCE_FRAME_CENTRE,
                       dim=AUX_DIM, parameters=AUX_PARAMETERS,
                       supervision_window=AUX_SUPERVISION_WINDOW,
                       head='Linear(256, 2, bias=True) outside `heads` (round23\'s class, reused)',
                       loss='L = L_control + w * CE(aux_logits, side)',
                       weight=dict(control=0., auxiliary=AUX_WEIGHT),
                       state_source='forward pre-hook on heads["%s"]' % STATE_HOOK_MODULE,
                       parent_detach='inert: the auxiliary head is downstream of the GRU',
                       supervised_events=dict(train=labelled_train),
                       unsupervised_train_events=[key for key, item in supervision.items()
                                                  if not item['target_defined']],
                       validation_labels=labelled_validation,
                       train_label_counts={name: sum(1 for value in labelled_train.values()
                                                     if value['label'] == name)
                                           for name in AUX_LABELS},
                       validation_label_counts={name: sum(1 for value in labelled_validation.values()
                                                          if value['label'] == name)
                                                for name in AUX_LABELS},
                       train_rows=sum(value['supervised_rows'] for value in labelled_train.values()),
                       validation_rows=sum(value['supervised_rows']
                                           for value in labelled_validation.values()),
                       label_changes_against_round23=dict(
                           train=[value['event_id'] for value in labelled_train.values()
                                  if value['label'] != side_name(value['round23_side'])],
                           validation=[value['event_id'] for value in labelled_validation.values()
                                       if value['label'] != side_name(value['round23_side'])]),
                       initial_state_sha256_per_seed=aux_per_seed,
                       initial_state_sha256_per_arm=r23.aux_init_shas(),
                       inference='the closed loop builds a plain TacticalPolicy from payload["heads"]; the '
                                 'auxiliary state is never loaded and there is no module to hold it'),
        teachers=teacher,
        teacher_digest=teacher_digest(teacher),
        labels=flips,
        agreement=agreement,
        round23_collinearity=collinearity,
        features=dict(source='round22.spatial_vectors reused verbatim', spatial_sha256=digest,
                      statistics=statistics, feature_norm=norms,
                      note='the 94-d spatial vector that function also returns is not used; it is the '
                           'frozen proof that the encode pass is round22\'s'),
        geometry=world_table(train + validation),
        validation_bindings=asset_bindings(records),
        initialization=dict(per_seed=initial, per_seed_aux=aux_per_seed,
                            per_seed_per_arm={str(seed): {arm: initial[str(seed)] for arm in ARMS}
                                              for seed in SEEDS},
                            identical_between_arms=True, aux_identical_between_arms=True,
                            equals_round16_round20_round21_round22_round23=True,
                            method='fresh seed reset immediately before every N1 construction after the '
                                   'backbone load, through the frozen r16 recipe; the auxiliary head is '
                                   'constructed afterwards, so it cannot move the head initialisation and '
                                   'both arms are byte-identical objects at step 0'),
        round23_reference=reference,
        expected=EXPECTED,
        verification=dict(
            prerequisites=['every training asset equals round21\'s frozen SHA',
                           'the round22 spatial digest recomputes to 83a75f73...',
                           'the round22 feature-norm table recomputes exactly',
                           'the round19 teacher_v2 table equals round21\'s stored rollout actions',
                           'the initial head SHA equals the frozen table for every seed',
                           'the six R18 baselines reproduce round22\'s 168 rows'],
            control='arm A must equal round23\'s A checkpoint SHA and its closed-loop rows for all 3 '
                    'seeds; otherwise it is reported as `control_not_reproduced` and the causal claim '
                    'is withheld',
            step0='a unit test asserts that step 0 of arm X is bitwise arm A and that perturbing the '
                  'auxiliary parameters changes no policy logit'),
        independent_test=False, formal_gate=False, deployable=False, game=False, tuning=False,
        new_worlds=False, new_teacher_labels=False, teacher_calls=0, official_labels=False,
        research_teacher_labels=True, new_pixels=False, privileged_label=True,
        privileged_label_scope='training loss and scoring only',
        failure_policy='first failure retained; one run; no retrain, rerun, repair or overwrite')
    r20.write_json(out / 'preregistration.json', value)
    return value


def teacher_digest(teacher):
    digest = hashlib.sha256()
    for event_id in sorted(teacher):
        digest.update(event_id.encode())
        digest.update(json.dumps([int(value) for value in teacher[event_id]]).encode())
    return digest.hexdigest()


def verify_prereg(out=OUT, device='cuda'):
    out = Path(out)
    p = json.loads((out / 'preregistration.json').read_text(encoding='utf-8'))
    if p.get('schema') != PREREG_SCHEMA:
        raise ValueError('preregistration schema')
    for path, digest in p['identities'].items():
        if file_sha(path) != digest:
            raise ValueError('frozen identity changed: ' + path)
    if p['criterion_sha256'] != criterion_sha():
        raise ValueError('the criterion drifted from the frozen pre-registration')
    manifest, split, train, validation = all_events()
    if manifest['sha256'] != p['sources']['manifest']['sha256']:
        raise ValueError('R18 manifest SHA drift')
    if split['sha256'] != p['sources']['split']['sha256']:
        raise ValueError('R19 split SHA drift')
    for name, rows in ((r20.TRAIN_SPLIT, train), (r20.VALIDATION_SPLIT, validation)):
        if r16.worlds_digest(rows) != p['worlds'][f'{name}_worlds_digest']:
            raise ValueError('world digest drift: ' + name)
        if [row['event_id'] for row in rows] != p['worlds'][f'{name}_events']:
            raise ValueError('world event list drift: ' + name)
    if r16.initial_head_shas() != p['initialization']['per_seed']:
        raise ValueError('initial head state drift')
    if json.dumps(r23_reference(), sort_keys=True) != json.dumps(p['round23_reference'], sort_keys=True):
        raise ValueError('the round23 reference record changed')
    if json.dumps(world_table(train + validation), sort_keys=True) != \
            json.dumps(p['geometry'], sort_keys=True):
        raise ValueError('the frozen geometry table changed')
    if sorted(r23.r22_assets()) != sorted(p['worlds']['train_events']):
        raise ValueError('round22 asset set changed')
    teacher, verdicts = teacher_table(train + validation)
    if teacher_digest(teacher) != p['teacher_digest']:
        raise ValueError('the frozen teacher table changed')
    if teacher != p['teachers']:
        raise ValueError('the frozen teacher actions changed')
    flips = flip_table(dict(label_rows(train, teacher), **label_rows(validation, teacher)))
    if json.dumps(flips, sort_keys=True) != json.dumps(p['labels'], sort_keys=True):
        raise ValueError('the frozen label / flip table changed')
    if json.dumps(agreement_table(train + validation, teacher), sort_keys=True) != \
            json.dumps(p['agreement'], sort_keys=True):
        raise ValueError('the frozen label-vs-teacher agreement table changed')
    if json.dumps(round23_collinearity(train + validation, teacher), sort_keys=True) != \
            json.dumps(p['round23_collinearity'], sort_keys=True):
        raise ValueError('the frozen round23-label collinearity table changed')
    backbone, _ = load_vision_checkpoint(BASELINE, map_location='cpu')
    backbone.to(device).eval()
    with _fp32_mode(torch.device(device)), torch.inference_mode():
        records = train_records(p)
        if asset_bindings(records) != p['validation_bindings']:
            raise ValueError('the asset bindings changed')
        data, vectors, _ = encode(backbone, records, device)
        if r22.spatial_digest(vectors) != p['features']['spatial_sha256']:
            raise ValueError('the encode pass changed since the pre-registration')
        if r22.spatial_table(vectors) != p['features']['statistics']:
            raise ValueError('the feature statistics changed since the pre-registration')
        if feature_norms(data) != p['features']['feature_norm']:
            raise ValueError('the feature norm table changed since the pre-registration')
    supervision = supervision_table(records, teacher)
    labelled = {key: value for key, value in supervision.items() if value['target_defined']}
    if json.dumps(labelled, sort_keys=True) != \
            json.dumps(p['auxiliary']['supervised_events']['train'], sort_keys=True):
        raise ValueError('the auxiliary supervision table changed since the pre-registration')
    validation_labels = {key: value for key, value in supervision_table(validation, teacher).items()
                         if value['target_defined']}
    if json.dumps(validation_labels, sort_keys=True) != \
            json.dumps(p['auxiliary']['validation_labels'], sort_keys=True):
        raise ValueError('the frozen validation auxiliary labels changed')
    if r23.per_seed_aux_shas(backbone) != p['auxiliary']['initial_state_sha256_per_seed']:
        raise ValueError('the auxiliary initialisation drifted')
    if r23.aux_init_shas() != p['auxiliary']['initial_state_sha256_per_arm']:
        raise ValueError('the auxiliary construction drifted')
    return p


# ---------------------------------------------------------------- summary
def comparison(closed):
    """K1/K2 plus the per-seed pairs and the first-hit cell check."""
    per_arm = {}
    for arm in ARMS:
        per_arm[arm] = [dict(seed=seed, network=r20.name_of(arm, seed),
                             f1=closed[(r20.name_of(arm, seed), r20.VALIDATION_SPLIT)]['families']['F1']['feasible'],
                             f1_survived=closed[(r20.name_of(arm, seed), r20.VALIDATION_SPLIT)]['families']['F1']['survived'],
                             all=closed[(r20.name_of(arm, seed), r20.VALIDATION_SPLIT)]['families']['all']['feasible'],
                             f1_train=closed[(r20.name_of(arm, seed), r20.TRAIN_SPLIT)]['families']['F1']['feasible'],
                             all_train=closed[(r20.name_of(arm, seed), r20.TRAIN_SPLIT)]['families']['all']['feasible'])
                         for seed in SEEDS]
    pairs = []
    for index, seed in enumerate(SEEDS):
        control = closed[(r20.name_of(CONTROL, seed), r20.VALIDATION_SPLIT)]['families']['all']['firsthit']
        auxiliary = closed[(r20.name_of(AUXILIARY, seed), r20.VALIDATION_SPLIT)]['families']['all']['firsthit']
        events = sorted(control)
        same_first_hit = [event for event in events if control[event] == auxiliary[event]]
        moved = {event: [control[event], auxiliary[event]] for event in events
                 if control[event] != auxiliary[event]}
        pairs.append(dict(seed=seed, control=per_arm[CONTROL][index]['f1'],
                          auxiliary=per_arm[AUXILIARY][index]['f1'],
                          delta=per_arm[AUXILIARY][index]['f1'] - per_arm[CONTROL][index]['f1'],
                          all_control=per_arm[CONTROL][index]['all'],
                          all_auxiliary=per_arm[AUXILIARY][index]['all'],
                          events=len(events), identical_first_hit_cells=len(same_first_hit),
                          moved_first_hit=moved))
    block = judgement([row['f1'] for row in per_arm[CONTROL]],
                      [row['f1'] for row in per_arm[AUXILIARY]],
                      [row['all'] for row in per_arm[CONTROL]],
                      [row['all'] for row in per_arm[AUXILIARY]])
    block['per_arm'] = per_arm
    block['per_seed'] = pairs
    block['identical_first_hit_cells'] = [row['identical_first_hit_cells'] for row in pairs]
    block['changed_first_hit_cells'] = [row['events'] - row['identical_first_hit_cells'] for row in pairs]
    block['note'] = ('three seed pairs, six F1 validation events each; X minus A; descriptive only. '
                     'K1/K2 above are the frozen rule; the round23 rule that called -1/+1/0 improved is '
                     'retired')
    return block


def assemble(dest, extra=None):
    """Rebuild the round summary read-only from the frozen artifacts."""
    dest = Path(dest)
    out = dest.parent
    prereg = json.loads((out / 'preregistration.json').read_text(encoding='utf-8'))
    if prereg['criterion_sha256'] != criterion_sha():
        raise ValueError('the criterion drifted from the frozen pre-registration')
    records = json.loads((dest / 'records.json').read_text(encoding='utf-8'))
    checkpoints = json.loads((dest / 'checkpoints.json').read_text(encoding='utf-8'))
    if sorted(checkpoints) != sorted(r23.all_names()):
        raise ValueError('checkpoint set changed')
    training = {}
    for name in r23.all_names():
        record = checkpoints[name]
        payload = json.loads((dest / 'train' / name / 'training.json').read_text(encoding='utf-8'))
        if file_sha(dest / record['checkpoint_path']) != record['checkpoint_sha256']:
            raise ValueError('checkpoint SHA drift: ' + name)
        if payload['final_head_state_sha256'] != record['final_head_state_sha256']:
            raise ValueError('checkpoint record mismatch: ' + name)
        if payload['initial_head_state_sha256'] != prereg['initialization']['per_seed'][str(payload['seed'])]:
            raise ValueError('initial head SHA differs from the pre-registration: ' + name)
        if payload['initial_aux_state_sha256'] != prereg['initialization']['per_seed_aux'][str(payload['seed'])]:
            raise ValueError('initial auxiliary SHA differs from the pre-registration: ' + name)
        training[name] = payload
    for seed in SEEDS:
        pair = {arm: training[r20.name_of(arm, seed)]['initial_head_state_sha256'] for arm in ARMS}
        aux_pair = {arm: training[r20.name_of(arm, seed)]['initial_aux_state_sha256'] for arm in ARMS}
        if len(set(pair.values())) != 1 or len(set(aux_pair.values())) != 1:
            raise ValueError('the two arms must start byte-identical for seed ' + str(seed))
    closed = {}
    expected = {r20.VALIDATION_SPLIT: prereg['worlds']['validation_events'],
                r20.TRAIN_SPLIT: prereg['worlds']['train_events']}
    for name in ['B0'] + r23.all_names():
        for split_name in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT):
            payload = json.loads((dest / 'closed-loop' / f'{name}-{split_name}.json')
                                 .read_text(encoding='utf-8'))
            if r16.families(payload['events']) != payload['families']:
                raise ValueError('stored aggregate mismatch: ' + name + '-' + split_name)
            if [row['event_id'] for row in payload['events']] != expected[split_name]:
                raise ValueError('event list differs from the pre-registration: ' + name + '-' + split_name)
            if name != 'B0' and payload['heads_sha256'] != training[name]['final_head_state_sha256']:
                raise ValueError('closed-loop heads SHA mismatch: ' + name)
            closed[(name, split_name)] = payload
    baselines = json.loads((dest / 'baselines.json').read_text(encoding='utf-8'))
    gaps = json.loads((dest / 'f1-gap.json').read_text(encoding='utf-8'))
    probes = json.loads((dest / 'probes.json').read_text(encoding='utf-8'))
    reference = r23_reference()
    equivalence = dict(round22=control_equivalence(closed, R22_EVIDENCE),
                       round23=control_equivalence(closed, R23_EVIDENCE))
    equivalence['all_equal'] = all(block['all_equal'] for block in (equivalence['round22'],
                                                                    equivalence['round23']))
    mechanism = {split_name: mechanism_block(gaps[split_name])
                 for split_name in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT)}
    k = k_report(closed, mechanism[r20.VALIDATION_SPLIT], probes, reference)
    k['K4']['train_seen'] = mechanism[r20.TRAIN_SPLIT]
    results = {}
    for name in r23.all_names():
        results[name] = dict(arm=training[name]['arm'], seed=training[name]['seed'],
                             validation=closed[(name, r20.VALIDATION_SPLIT)]['families'],
                             train_seen=closed[(name, r20.TRAIN_SPLIT)]['families'],
                             validation_behaviour=closed[(name, r20.VALIDATION_SPLIT)]['behaviour'],
                             train_seen_behaviour=closed[(name, r20.TRAIN_SPLIT)]['behaviour'],
                             checkpoint_sha256=training[name]['checkpoint_sha256'],
                             final_head_state_sha256=training[name]['final_head_state_sha256'],
                             initial_head_state_sha256=training[name]['initial_head_state_sha256'],
                             initial_aux_state_sha256=training[name]['initial_aux_state_sha256'],
                             final_aux_state_sha256=training[name]['final_aux_state_sha256'],
                             aux_weight=training[name]['aux_weight'],
                             matches_round23_control=training[name]['matches_round23_control'],
                             loss_at_step0=training[name]['loss_at_step0'],
                             loss_final=training[name]['loss_final'],
                             aux_loss_first=training[name]['aux_loss_first'],
                             aux_loss_final=training[name]['aux_loss_final'],
                             aux_accuracy_first=training[name]['aux_accuracy_first'],
                             aux_accuracy_final=training[name]['aux_accuracy_final'],
                             aux_supervised_steps=training[name]['aux_supervised_steps'],
                             aux_grad_norm=training[name]['aux_grad_norm'],
                             auxiliary_head=training[name]['auxiliary_head'],
                             parameters=training[name]['parameters'],
                             optimization_seconds=training[name]['optimization_seconds'],
                             initial_offline=training[name]['initial_offline'],
                             final_offline=training[name]['final_offline'])
    holdout = baselines['validation']
    summary = dict(
        schema=RESULT_SCHEMA, prereg_sha256=file_sha(out / 'preregistration.json'),
        criterion_sha256=prereg['criterion_sha256'], seeds=list(SEEDS), arms=list(ARMS),
        updates=r20.UPDATES, loss_arm=LOSS_ARM, aux_weight=AUX_WEIGHT, label_rule=LABEL_RULE,
        intervention=dict(auxiliary=prereg['auxiliary'], features=prereg['features'],
                          label=prereg['labels'],
                          note='the only change against round23 is the label\'s reference point: the '
                               'gap relative to the player instead of to the screen centre; arm A is '
                               'round23 A with that weight at zero'),
        sources=prereg['sources'], worlds=prereg['worlds'],
        data=dict(events=len(records['events']), rows=records['totals']['rows'],
                  bindings=records['bindings'], storage=records['storage'],
                  aux=records['aux'], totals=records['totals'], failures=records['failures']),
        initialization=dict(per_seed=prereg['initialization']['per_seed'],
                            per_seed_aux={str(seed): {arm: training[r20.name_of(arm, seed)]
                                                      ['initial_aux_state_sha256'] for arm in ARMS}
                                          for seed in SEEDS},
                            identical_between_arms=True, aux_identical_between_arms=True,
                            equals_round16_round20_round21_round22_round23=True),
        equivalence=dict(control=equivalence, all_equal=equivalence['all_equal'],
                         matches_round23_control={name: training[name]['matches_round23_control']
                                                  for name in r23.names_of(CONTROL)},
                         control_not_reproduced=not equivalence['all_equal'],
                         step0='unit tested: identical policy logits and identical masked action at step '
                               '0; perturbing the auxiliary parameters changes no policy logit'),
        checkpoints={name: {key: value for key, value in training[name].items()
                            if key not in ('initial_offline_events', 'final_offline_events',
                                           'aux_accuracy', 'aux_supervised_counts',
                                           'supervised_events', 'aux_supervised_events')}
                     for name in r23.all_names()},
        judgement=k,
        validation=dict(
            events=EXPECTED['validation_events'], per_family=EXPECTED['validation_events_per_family'],
            holdout=dict(filter_valid=holdout['filter_valid'], passing_events=holdout['passing_events'],
                         executor=holdout['executor'], policies=holdout['policies'],
                         summary=holdout['summary'],
                         rule=prereg['criterion']['evaluation']['holdout_rule']),
            b0=dict(families=closed[('B0', r20.VALIDATION_SPLIT)]['families'],
                    behaviour=closed[('B0', r20.VALIDATION_SPLIT)]['behaviour']),
            results={name: results[name]['validation'] for name in r23.all_names()},
            behaviour={name: results[name]['validation_behaviour'] for name in r23.all_names()},
            per_event_firsthit={name: closed[(name, r20.VALIDATION_SPLIT)]['families']['all']['firsthit']
                                for name in ['B0'] + r23.all_names()},
            direction=mechanism[r20.VALIDATION_SPLIT],
            f1_cells=gaps[r20.VALIDATION_SPLIT]),
        train_seen=dict(
            events=EXPECTED['train_events'], per_family=EXPECTED['train_events_per_family'],
            label='seen: exactly the 14 teacher rollouts the heads were trained on',
            holdout=dict(filter_valid=baselines['train']['filter_valid'],
                         passing_events=baselines['train']['passing_events'],
                         summary=baselines['train']['summary']),
            b0=dict(families=closed[('B0', r20.TRAIN_SPLIT)]['families'],
                    behaviour=closed[('B0', r20.TRAIN_SPLIT)]['behaviour']),
            results={name: results[name]['train_seen'] for name in r23.all_names()},
            per_event_firsthit={name: closed[(name, r20.TRAIN_SPLIT)]['families']['all']['firsthit']
                                for name in ['B0'] + r23.all_names()},
            direction=mechanism[r20.TRAIN_SPLIT],
            f1_cells=gaps[r20.TRAIN_SPLIT]),
        probes=probes,
        label_diagnostics=dict(rule=prereg['labels'], agreement=prereg['agreement'],
                               round23_collinearity=prereg['round23_collinearity']),
        round23_reference=reference,
        comparison=comparison(closed),
        environment=dict(device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                         torch_version=torch.__version__,
                         precision=dict(dtype='float32',
                                        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                        cudnn_tf32=torch.backends.cudnn.allow_tf32,
                                        matmul_precision=torch.get_float32_matmul_precision()),
                         timings=json.loads((dest / 'timings.json').read_text(encoding='utf-8'))
                         if (dest / 'timings.json').is_file() else None),
        flags=dict(independent_test=False, formal_gate=False, deployable=False, game=False, tuning=False,
                   official_labels=False, research_teacher_labels=True, new_worlds=False, new_pixels=False,
                   teacher_calls=0, validation_used_for_selection=False, early_stopping=False,
                   retrained=False, rerun=False, physics_changed=False, intervention_count=1,
                   privileged_label=True, label_reference='player x'),
        limitations=[
            'n=3 seeds and 7 validation groups; the 14 validation events collapse to 7 geometries because '
            'a mirror is not an independent sample, and F1 validation is 3 groups / 6 events',
            'the label is constant within an event and on the frozen teacher table it is perfectly '
            'confounded with the mirror group (all base events need screen-right, all mirror events need '
            'screen-left), so a head could in principle read "which group is this" instead of "where is '
            'the gap"; the per-cell table, the offset column and the confusion are reported so that this '
            'cannot be hidden',
            'the label is a displacement, not a parking instruction: it never says "hold", and the '
            'teacher itself holds at the anchoring row on 6 of 12 labelled events',
            'the auxiliary task is supervised only on the six F1 train events and only inside the '
            'telegraph window, so 8 of the 14 update positions carry an exactly-zero auxiliary term',
            'the auxiliary weight (0.5) is fixed before the run and never tuned',
            'the F1 win condition is exact parking inside a 12-20 px gap; it is not tactical understanding',
            'synthetic fixed-y worlds, two-frame actions, no shot, no bomb, no real TH08 pixels',
            'a privileged training-time label is not a route to a deployable policy; research heads only: '
            'no formal gate, no deployable checkpoint, no game'],
        independent_test=False, formal_gate=False, deployable=False, game=False, tuning=False)
    if extra:
        summary['environment'].update(extra)
    return summary


def write_events(dest, prereg, records, closed, gaps):
    """One JSON per event: the teacher route, the asset binding, every closed-loop result."""
    r20.new_dir(dest / 'events')
    assets = {entry['event_id']: entry for entry in records}
    stored_baselines = json.loads((dest / 'baselines.json').read_text(encoding='utf-8'))
    baseline_rows = {(row['split'], row['event_id'], row['policy']): row
                     for split in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT)
                     for row in stored_baselines[split]['rows_detail']}
    baseline_names = stored_baselines['validation']['policies']
    for event_id in prereg['worlds']['train_events'] + prereg['worlds']['validation_events']:
        split_name = r20.TRAIN_SPLIT if event_id in prereg['worlds']['train_events'] else r20.VALIDATION_SPLIT
        teacher, asset = None, assets.get(event_id)
        if asset is not None:
            teacher = json.loads((R21_EVIDENCE / 'run' / asset['rollout']['path']).read_text(encoding='utf-8'))
        payloads = {name: closed[(name, split_name)] for name in ['B0'] + r23.all_names()}
        rows = {name: next(row for row in payload['events'] if row['event_id'] == event_id)
                for name, payload in payloads.items()}
        behaviour = {name: next(item for item in payload['behaviour']['per_event']
                                if item['event_id'] == event_id)
                     for name, payload in payloads.items()}
        row_families = {name: behaviour[name]['family'] for name in behaviour}
        value = dict(schema=EVENT_SCHEMA, event_id=event_id, split=split_name,
                     seen=(split_name == r20.TRAIN_SPLIT), family=row_families['B0'],
                     group=event_id.rsplit('-m', 1)[0],
                     teacher_verified=asset is not None and asset['teacher_verified'],
                     asset=({key: asset[key] for key in ('pixels', 'history', 'labels', 'rollout')}
                            if asset else None),
                     teacher=teacher,
                     frozen_teacher_actions=prereg['teachers'][event_id],
                     geometry=prereg['geometry'][event_id],
                     auxiliary=prereg['auxiliary']['supervised_events']['train'].get(event_id)
                     or prereg['auxiliary']['validation_labels'].get(event_id) or {},
                     agreement=prereg['agreement']['rows'].get(event_id, {}),
                     gap={name: gaps[split_name].get(f'{name}/{event_id}')
                          for name in ['B0'] + r23.all_names()},
                     r18_six_baselines={name: dict(feasible=baseline_rows[(split_name, event_id, name)]['feasible'],
                                                   survived=baseline_rows[(split_name, event_id, name)]['survived'],
                                                   first_hit=baseline_rows[(split_name, event_id, name)]['first_hit'])
                                        for name in baseline_names},
                     closed_loop={name: dict(feasible=rows[name]['feasible'],
                                             survived=rows[name]['survived'],
                                             goal_satisfied=rows[name]['goal_satisfied'],
                                             first_hit=rows[name]['first_collision_frame'],
                                             decisions=rows[name]['decisions'],
                                             raw_unsupported_count=rows[name]['raw_unsupported_count'],
                                             action_counts=behaviour[name]['action_counts'],
                                             switches=behaviour[name]['switches'],
                                             distinct_candidates=behaviour[name]['distinct_candidates'],
                                             single_candidate=behaviour[name]['single_candidate'])
                                  for name in ['B0'] + r23.all_names()})
        r20.write_json(dest / 'events' / f'{event_id}.json', value)
    if len(list((dest / 'events').glob('*.json'))) != EXPECTED['event_files']:
        raise AssertionError('28 event files required')


def verify_artifacts(out=OUT):
    """Read-only re-check of every frozen artifact and its SHA."""
    out = Path(out)
    prereg = verify_prereg(out)
    dest = out / 'run'
    records = json.loads((dest / 'records.json').read_text(encoding='utf-8'))
    checked = 0
    for entry in records['events']:
        for key in ('pixels', 'history', 'labels', 'rollout'):
            if file_sha(R21_EVIDENCE / 'run' / entry[key]['path']) != entry[key]['sha256']:
                raise ValueError('round21 asset SHA drift: ' + entry['event_id'] + '/' + key)
            checked += 1
    checkpoints = json.loads((dest / 'checkpoints.json').read_text(encoding='utf-8'))
    for name, record in checkpoints.items():
        if file_sha(dest / record['checkpoint_path']) != record['checkpoint_sha256']:
            raise ValueError('checkpoint SHA drift: ' + name)
        payload = torch.load(dest / record['checkpoint_path'], map_location='cpu', weights_only=True)
        if set(payload) != set(PAYLOAD_KEYS):
            raise ValueError('stored checkpoint keys: ' + name)
        if {key.split('.')[0] for key in payload['heads']} != set(HEAD_MODULES):
            raise ValueError('stored checkpoint carries a non-head tensor: ' + name)
        if payload['final_head_state_sha256'] != record['final_head_state_sha256']:
            raise ValueError('stored checkpoint head SHA record: ' + name)
        if payload['final_aux_state_sha256'] != record['final_aux_state_sha256']:
            raise ValueError('stored checkpoint auxiliary SHA record: ' + name)
        if payload['aux_weight'] != record['aux_weight']:
            raise ValueError('stored checkpoint auxiliary weight: ' + name)
    baselines = json.loads((dest / 'baselines.json').read_text(encoding='utf-8'))
    closed = 0
    for name in ['B0'] + r23.all_names():
        for split_name in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT):
            payload = json.loads((dest / 'closed-loop' / f'{name}-{split_name}.json')
                                 .read_text(encoding='utf-8'))
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
    probes = json.loads((dest / 'probes.json').read_text(encoding='utf-8'))
    summary = json.loads((dest / 'summary.json').read_text(encoding='utf-8'))
    events = len(list((dest / 'events').glob('*.json')))
    judgement = summary['judgement']
    return dict(artifacts=len(stored), root_artifacts=len(root), assets=checked,
                checkpoints=len(checkpoints), closed_loop_files=closed, event_files=events,
                baseline_rows=baselines['validation']['rows'] + baselines['train']['rows'],
                baseline_passing_events_validation=baselines['validation']['passing_events'],
                holdout_filter_valid=baselines['validation']['filter_valid'],
                f1_gap_rows=len(gaps['validation']) + len(gaps['train']),
                aux_parameters=AUX_PARAMETERS, aux_weight=AUX_WEIGHT, label_rule=LABEL_RULE,
                aux_holdout_accuracy={name: probes['aux']['holdout'][name]['overall']['accuracy']
                                      for name in r23.all_names()},
                aux_seen_accuracy={name: probes['aux']['seen'][name]['overall']['accuracy']
                                   for name in r23.all_names()},
                state_decodability={name: probes['state'][name]['side_accuracy']
                                    for name in r23.all_names()},
                K1=judgement['K1'], K2=judgement['K2'], verdict=judgement['verdict'],
                K3=judgement['K3'], K4=dict(in_gap=judgement['K4']['in_gap'],
                                            parked_px=judgement['K4']['parked_px']),
                K5=dict(learned=judgement['K5']['learned'],
                        holdout=judgement['K5']['holdout']),
                learned_but_unused=judgement['learned_but_unused'],
                control_not_reproduced=summary['equivalence']['control_not_reproduced'],
                matches_round23_control=summary['equivalence']['matches_round23_control'],
                label_agreement={split: {key: value[key] for key in ('episode_first_action',
                                                                    'anchored_action')}
                                 for split, value in prereg['agreement']['by_split'].items()},
                label_flips=dict(events=len(prereg['labels']['flipping_events']),
                                 flipping_events=prereg['labels']['flipping_events'],
                                 flip_count=prereg['labels']['flip_count']),
                round23_label_equals_player_half=prereg['round23_collinearity']['by_split'],
                aux_holdout_vs_frozen={arm: judgement['K5']['holdout_vs_frozen_teacher'][arm]
                                       for arm in ARMS},
                one_class_heads=judgement['K5']['one_class_heads'],
                train_events=len(prereg['worlds']['train_events']),
                validation_events=len(prereg['worlds']['validation_events']),
                criterion_sha256=prereg['criterion_sha256'], identities=len(prereg['identities']))


# ---------------------------------------------------------------- execution
def run(out=OUT, device='cuda'):
    out = Path(out)
    prereg = verify_prereg(out, device)
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
            records = train_records(prereg)
            teacher = {event_id: [int(value) for value in actions]
                       for event_id, actions in prereg['teachers'].items()}
            labels = label_rows(records, teacher)
            backbone, _ = load_vision_checkpoint(BASELINE, map_location='cpu')
            backbone.to(device).eval()
            frozen = {key: value.detach().clone() for key, value in backbone.state_dict().items()}
            data, vectors, _ = encode(backbone, records, device)
            if r22.spatial_digest(vectors) != prereg['features']['spatial_sha256']:
                raise AssertionError('the encode pass differs from the pre-registration')
            if feature_norms(data) != prereg['features']['feature_norm']:
                raise AssertionError('the feature norm table differs from the pre-registration')
            if len(data) != EXPECTED['train_events']:
                raise AssertionError('encoded event count')
            if json.dumps(flip_table(labels)['events'], sort_keys=True) != json.dumps(
                    {key: value for key, value in prereg['labels']['events'].items()
                     if key in labels}, sort_keys=True):
                raise AssertionError('the training label table differs from the pre-registration')
            write_records(dest, records, supervision_table(records, teacher), prereg['labels'])
            timings['data_seconds'] = round(time.perf_counter() - begin, 3)
            begin = time.perf_counter()
            reports, paths = train_all(dest, data, backbone, device, prereg, labels)
            timings['training_seconds_loop'] = round(time.perf_counter() - begin, 3)
            if not all(torch.equal(value, frozen[key]) for key, value in backbone.state_dict().items()):
                raise AssertionError('the frozen backbone changed')
            baselines, closed, seconds = evaluate_all(dest, prereg, backbone, device, reports, paths)
            timings['evaluation_seconds'] = round(seconds, 3)
            gaps = {split_name: direction_table(closed, split_name, event_records(prereg, split_name))
                    for split_name in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT)}
            r20.write_json(dest / 'f1-gap.json', dict(
                notes='per (network, F1 event): the x series is rebuilt from advance semantics; '
                      '"in_gap_at_wall" is the position on wall_frame-1; "direction" is canonical '
                      '(round21 semantics) and "screen_direction" is the raw x motion',
                validation=gaps[r20.VALIDATION_SPLIT], train=gaps[r20.TRAIN_SPLIT]))
            begin = time.perf_counter()
            probes = run_probes(backbone, device, closed, prereg, records, data, paths, reports, gaps,
                                prereg['agreement'])
            r20.write_json(dest / 'probes.json', probes)
            timings['probes_seconds'] = round(time.perf_counter() - begin, 3)
            write_events(dest, prereg, records, closed, gaps)
            timings['training_seconds_total'] = round(sum(reports[name]['optimization_seconds']
                                                          for name in reports), 3)
            r20.write_json(dest / 'timings.json', timings)
            if not baselines['validation']['filter_valid']:
                print('WARNING the R18 filter failed on the holdout side:',
                      baselines['validation']['passing_events'], flush=True)
            verify_prereg(out, device)
            summary = assemble(dest)
            r20.write_json(dest / 'summary.json', summary)
            r20.write_json(dest / 'sha256.json',
                           {str(path.relative_to(dest)): file_sha(path)
                            for path in dest.rglob('*') if path.is_file()})
            r20.write_json(out / 'sha256.json',
                           {str(path.relative_to(out)): file_sha(path)
                            for path in out.rglob('*') if path.is_file()})
            return summary
    except BaseException:
        r20.write_json(dest / 'first-failure.json',
                       dict(traceback=traceback.format_exc(),
                            policy='retained; no automatic rerun, overwrite, retune or baseline switch'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preregister', 'run', 'summary', 'verify', 'judge'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'preregister':
        preregister(args.out, args.device)
    elif args.command == 'run':
        run(args.out, args.device)
    elif args.command == 'summary':
        print(json.dumps(assemble(args.out / 'run'), indent=2)[:4000])
    elif args.command == 'judge':
        print(json.dumps(judgement([2, 2, 2], [1, 3, 2]), indent=2))
    else:
        print(json.dumps(verify_artifacts(args.out), indent=2))
