"""Round23: one single-factor intervention on top of round22 - a training-only privileged
auxiliary task; research only.

Round22 established two facts at once. (a) The representation carries the answer: the frozen
256-d trunk vector that the N1 head already sees is **98.5% linearly decodable** for which
side of the screen the F1 gap is on (lambda sweep 95.3-98.5%), so "the information is not in
the input" is false. (b) The behaviour does not use it: the mirror-group toward rate is
3/9 -> 3/9 across the two arms, 95-97% of the telegraph-window decisions are `hold`, and the
94-d pooled spatial vector that round added decodes the side at only 67% while its fusion
layer learned a 94.4%-constant re-bias. What remains is that the *objective* never forces the
head to use the position it can already read.

This round changes exactly one thing: a **training-only privileged auxiliary task**. From the
GRU state - the tensor the residual head already conditions on - a `Linear(256, 2)` head
predicts whether the safe gap's screen-x centre lies left or right of the playfield centre
(192 of 384 native px). The label is `round21.telegraph_gap(s)`, i.e. `s.gap` under
`s.mirror`: exactly the arithmetic the R21 renderer uses to draw the band, and available only
while training. The auxiliary cross-entropy enters with a fixed weight (0.5 for arm X, 0.0 for
arm A) and a unit test asserts the inference path cannot read it: no label is ever
concatenated into `features`/`history`/pixels (the GRU input stays 256 + 29 = 285 wide), the
auxiliary parameters live **outside** `heads` (so the policy head state dict, its SHA and the
whole checkpoint contract stay byte-identical to round22's), and the evaluation network is a
plain `TacticalPolicy` that has no auxiliary head to read.

Arm A is the same code path with the auxiliary weight at zero, so the two arms are
byte-identical objects at step 0 and differ in exactly one number. Because `0.0 * L` carries
an exactly-zero gradient, arm A must reproduce round22's control checkpoint bitwise; that is
measured per seed and reported (`matches_round22_control`), never assumed.

Two implementation notes, both unit tested:
  1. The parent `TacticalPolicy.forward_features` starts by detaching `features`. Unlike
     round22's fusion layer (which sat *upstream* of that line and therefore needed it
     removed), the auxiliary head is *downstream* of the GRU: `features` is frozen backbone
     output with no trainable ancestor, so the parent's `detach()` costs the auxiliary head
     nothing and **this round overrides no line of the parent body**. The GRU state is
     captured by a forward pre-hook on the frozen `heads['option']` layer, whose input *is*
     that state; five independent heads are recomputed from the captured tensor and compared
     bitwise, so "this is the state the policy used" is measured, not asserted.
  2. The auxiliary label exists only for F1 worlds (F2/F3 have `internal_gap = False`, i.e. no
     gap), and it is supervised only on the rows inside the telegraph window - the interval in
     which the renderer actually draws the band - because outside that window the label
     describes a wall that has not been telegraphed yet and supervising it would ask the head
     to guess instead of to read. This is frozen in the pre-registration.

No new worlds, no teacher calls, no new tactical labels, no re-rendered training pixels
(round21's frozen assets, SHA-bound), no formal gate, no deployable checkpoint, no game
execution, no tuning.
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
from torch import nn
from torch.nn import functional as F
from tools.rl import v373_seed_confirm as r16
from tools.rl import v373_spatial_fuse as r22
from tools.rl import v373_telegraph as r21
from tools.rl import v373_valid_curriculum as r20
from touhou_rl.macro_policy import _finite
from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_feature_cache import _fp32_mode
from touhou_rl.tactical_policy import (TacticalConfig, TacticalPolicy, tactical_policy_loss)
from touhou_rl.tactical_synthetic_closed_loop import executed_history, pixel_stack
from touhou_rl.tactical_teacher_validation import make_world
from touhou_rl.vmodels import load_vision_checkpoint

ROOT = Path('Document/Update/Alpha/v3.7.3/evidence')
OUT = ROOT / 'privaux-round23'
R22_EVIDENCE = ROOT / 'spatialfuse-round22'
R21_EVIDENCE = ROOT / 'telegraph-round21'
REPORTS = Path('Document/Update/Alpha/v3.7.3/reports')
BASELINE = r20.BASELINE

PREREG_SCHEMA = 'v373-privaux-prereg/1'
DATA_SCHEMA = 'v373-privaux-data/1'
EVENT_SCHEMA = 'v373-privaux-event/1'
HEADS_SCHEMA = 'v373-privaux-heads/1'
RESULT_SCHEMA = 'v373-privaux-result/1'

# ---------------------------------------------------------------- the intervention
ARMS = ('A', 'X')
CONTROL, AUXILIARY = 'A', 'X'
AUX_WEIGHT = .5                       # the single number the two arms disagree on
SEEDS = r20.SEEDS
LOSS_ARM = r20.CONTROL                # both arms are the round-20/21/22 control loss: no KL anchor
FEATURE_DIM = 256                     # vmodels3.TRUNK_HIDDEN, asserted against the checkpoint
HIDDEN_SIZE = 256                     # the GRU width, asserted against the config
AUX_DIM = 2                           # binary: left / right of the playfield centre
AUX_THRESHOLD = 192.                  # screen centre of the 384-px native playfield
AUX_LABELS = ('left', 'right')
AUX_PARAMETERS = HIDDEN_SIZE * AUX_DIM + AUX_DIM        # 514
AUX_TASK = ('from the GRU state predict whether the safe gap\'s screen-x centre lies left or '
            'right of x=192; label = round21.telegraph_gap(s), i.e. s.gap under s.mirror')
AUX_SUPERVISION_WINDOW = ('the rows whose logical frame 2*i lies inside '
                          'round21.telegraph_window(s) = [wall_frame-45, wall_frame)')
STATE_HOOK_MODULE = 'option'
POLICY_HEADS_ACTIVE = 429200
GRU_INPUT_WIDTH = FEATURE_DIM + 29    # candidate19 + option4 + hold/delta2 + target2 + validity2
HEAD_MODULES = ('gru', 'memoryless', 'option', 'target', 'urgency', 'termination', 'slack',
                'option_embedding', 'residual')
PAYLOAD_KEYS = frozenset(('schema', 'arm', 'seed', 'config', 'baseline_sha256',
                          'final_head_state_sha256', 'final_aux_state_sha256',
                          'initial_aux_state_sha256', 'aux_weight', 'loss_arm', 'auxiliary_head',
                          'heads', 'aux_state'))

PARENT_DETACH_LINE = 'features, base_logits = features.detach(), base_logits.detach()'

EXPECTED = dict(events=28, train_events=14, validation_events=14,
                train_events_per_family={'F1': 6, 'F2': 2, 'F3': 6},
                validation_events_per_family={'F1': 6, 'F2': 2, 'F3': 6},
                checkpoints=6, closed_loop_files=14, networks=7, baseline_rows=2 * 14 * 6,
                f1_gap_rows=2 * 7 * 6, event_files=28,
                aux_events=6, aux_train_rows=128, aux_validation_rows=132,
                aux_unlabelled_train_events=8)

# Round22's recorded results, re-read and re-asserted from its frozen artifacts before this
# round starts. They are the only comparison baseline of round23 and are never typed in
# without a machine check.
R22_CONTROL_FINAL_HEAD_SHA = {
    'A-s7': 'd52551bb43e4d49d55566548fb7a579ad9fab7161e7624abcadd600a496d4e72',
    'A-s17': 'fb9233f11bb2c556e4a689e91f2144c0b98b94ca81bce3a5319de9cd5493ea64',
    'A-s27': '49ff13d9384fb2d6768fd3bc6453b40f28bccaa5fa915ee5bf2fc3f885cb77ac'}
R22_SPATIAL_SHA = '83a75f73211b1964d5ff3cbb21220d09ed4b8d0e92fe181c010cd59909f58cd5'
R22_F1_VALIDATION_FEASIBLE = {'A': [2, 2, 2], 'S': [2, 2, 2]}
R22_ALL_VALIDATION_FEASIBLE = {'A': [3, 3, 3], 'S': [3, 3, 3]}
R22_COMPARISON_DELTAS = [0, 0, 0]
R22_DIRECTION = dict(
    A=dict(base=dict(toward=6, hold=3, away=0), mirror=dict(toward=3, hold=0, away=6)),
    S=dict(base=dict(toward=6, hold=0, away=3), mirror=dict(toward=3, hold=1, away=5)),
    six_heads=dict(base=dict(toward=12, hold=3, away=3), mirror=dict(toward=6, hold=1, away=11)))


# ---------------------------------------------------------------- the privileged label
def aux_target(world):
    """Training-only label: which screen half holds the safe gap's centre, or None.

    Provenance is exactly `s.gap` and `s.mirror` (through `round21.telegraph_gap`, the same
    arithmetic the R21 renderer uses to draw the band). F2/F3 worlds have no gap at all
    (`internal_gap = False`), so they carry no label and contribute no auxiliary gradient.
    """
    gap = r21.telegraph_gap(world)
    if gap is None:
        return None
    centre = (float(gap[0]) + float(gap[1])) / 2.
    return int(centre > AUX_THRESHOLD)


def aux_mask(world):
    """The rows the label is supervised on: inside the telegraph window. None if no label."""
    if aux_target(world) is None:
        return None
    window = r21.telegraph_window(world)
    return np.array([window[0] <= frame < window[1] for frame in r21.DECISION_FRAMES], dtype=bool)


def aux_terms(world):
    """Everything the loss and the scoring need about one world's auxiliary task, or None."""
    target = aux_target(world)
    if target is None:
        return None
    mask = aux_mask(world)
    gap = r21.telegraph_gap(world)
    window = r21.telegraph_window(world)
    centre = (float(gap[0]) + float(gap[1])) / 2.
    return dict(side=target, label=AUX_LABELS[target], mask=mask, rows=int(mask.sum()),
                gap_screen=[int(gap[0]), int(gap[1])], gap_screen_centre=round(centre, 6),
                preview=[int(window[0]), int(window[1])],
                distance_from_centre=round(abs(centre - AUX_THRESHOLD), 6))


def side_name(value):
    return 'none' if value is None else AUX_LABELS[int(value)]


# ---------------------------------------------------------------- the auxiliary head
class PrivilegedAux(nn.Module):
    """`Linear(256, 2)`: the training-only classifier over the GRU state.

    It is deliberately *not* part of `heads`. That keeps the policy head state dict, its SHA
    and every checkpoint field byte-identical to round22's, and it makes "the evaluation path
    cannot read the auxiliary head" structural: the evaluation network is a plain
    `TacticalPolicy` built from `payload['heads']`, which contains no key of this module.
    """

    def __init__(self, hidden_size=HIDDEN_SIZE, classes=AUX_DIM):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.classes = int(classes)
        self.classifier = nn.Linear(self.hidden_size, self.classes, bias=True)

    def forward(self, state):
        if not torch.is_tensor(state) or state.ndim != 3 or state.shape[-1] != self.hidden_size:
            raise ValueError('the auxiliary head reads the GRU state [B,T,%d]' % self.hidden_size)
        _finite(state, 'gru state')
        return self.classifier(state)

    def describe(self):
        return dict(layer='Linear(%d, %d, bias=True)' % (self.hidden_size, self.classes),
                    parameters=sum(p.numel() for p in self.parameters()),
                    outside_heads=True,
                    never_an_input=True,
                    note='it never enters `features`, `history` or a pixel region; it only adds '
                         'gradient to the shared GRU while training')


class PrivAuxPolicy(TacticalPolicy):
    """`TacticalPolicy` plus a training-only auxiliary head; **no line of the parent is overridden**.

    Round22 needed a subclass because its fusion layer sat upstream of `forward_features`,
    whose first statement detaches `features` - that would have starved the fusion of gradient
    forever. The auxiliary head of this round sits downstream of the GRU, so the parent's
    `detach()` is inert for it: `features` is frozen backbone output with no trainable
    ancestor. The parent body is therefore used verbatim, and `type(model).forward_features`
    is literally `TacticalPolicy.forward_features`.

    The GRU state is the *input of `heads['option']`*. A forward pre-hook captures it; the unit
    tests recompute all five heads that consume it from the captured tensor and assert bitwise
    equality, so the captured tensor is the state the policy used, not a look-alike.
    """

    def __init__(self, vision_model, config=None):
        super().__init__(vision_model, config)
        self.aux = PrivilegedAux(hidden_size=self.tactical_config.hidden_size)
        self._state = None
        self._handle = self.heads[STATE_HOOK_MODULE].register_forward_pre_hook(self._capture_state)

    def _capture_state(self, module, inputs):
        self._state = inputs[0]
        return None

    def state(self):
        if self._state is None:
            raise RuntimeError('forward_features must run before the GRU state is available')
        return self._state

    def aux_logits(self, state=None):
        return self.aux(self._state if state is None else state)

    def forward_with_aux(self, features, base_logits, history=None, hidden=None,
                         reset_mask=None, valid_mask=None):
        """The parent forward, then the auxiliary logits *read from* its state.

        The auxiliary computation happens strictly after and strictly beside the parent's
        return value, so it cannot perturb `policy_logits`; a unit test asserts that an
        arbitrary perturbation of the auxiliary parameters leaves the policy forward bitwise
        unchanged.
        """
        self._state = None
        output = self.forward_features(features, base_logits, history, hidden, reset_mask, valid_mask)
        return output, self.aux(self.state())

    def describe(self):
        return dict(aux=self.aux.describe(),
                    policy_forward='unchanged: PrivAuxPolicy overrides no method of TacticalPolicy',
                    state_source='forward pre-hook on heads["%s"], whose input is the GRU state'
                                 % STATE_HOOK_MODULE,
                    parent_detach='the parent detaches `features`; that is inert here because the '
                                  'auxiliary head is downstream of the GRU and the backbone is frozen',
                    parameters=dict(policy_heads=POLICY_HEADS_ACTIVE,
                                    auxiliary=sum(p.numel() for p in self.aux.parameters())))


def aux_loss(aux_logits, target, mask):
    """Cross-entropy over the supervised rows only; exactly zero when nothing is supervised.

    `target` is [B,T] long and `mask` is [B,T] bool. Rows outside the mask are never indexed,
    so an event with no label (F2/F3) contributes `0 * sum(logits)`, which keeps the graph
    shape and carries an exactly-zero gradient.
    """
    if not torch.is_tensor(aux_logits) or aux_logits.ndim != 3 or aux_logits.shape[-1] != AUX_DIM:
        raise ValueError('aux_logits must be [B,T,%d]' % AUX_DIM)
    _finite(aux_logits, 'aux_logits')
    if mask.shape != aux_logits.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('aux mask must be bool [B,T]')
    if target.shape != mask.shape or target.dtype != torch.long:
        raise ValueError('aux target must be long [B,T]')
    count = int(mask.sum())
    if not count:
        return dict(loss=aux_logits.sum() * 0., count=0, correct=0, accuracy=None,
                    supervised=False)
    truth = target[mask]
    if bool(((truth < 0) | (truth >= AUX_DIM)).any()):
        raise ValueError('aux target outside the frozen two classes')
    logits = aux_logits[mask]
    loss = F.cross_entropy(logits, truth)
    correct = int((logits.argmax(-1) == truth).sum())
    return dict(loss=loss, count=count, correct=correct, accuracy=correct / count,
                supervised=True)


def fresh_aux_head(backbone, seed):
    """The frozen R16 recipe followed by the auxiliary head: the policy head is built first.

    `torch.manual_seed(seed)` then `TacticalPolicy(...)` is exactly `r16.fresh_policy`, so the
    policy head draws the same numbers as round15/16/17/20/21/22 and the auxiliary head -
    created afterwards from the same RNG state in both arms - cannot move it. A unit test
    asserts the head SHA equals the frozen table and the auxiliary SHA is identical in both
    arms.
    """
    torch.manual_seed(seed)
    return PrivAuxPolicy(backbone, TacticalConfig(arm='N1'))


def aux_init_shas():
    """The frozen initial auxiliary SHA: the same construction for both arms."""
    out = {}
    for arm in ARMS:
        torch.manual_seed(0)
        out[arm] = r16.state_sha(PrivilegedAux().state_dict())
    return out


def per_seed_aux_shas(backbone):
    """The auxiliary initial state that `fresh_aux_head` actually produces, per seed."""
    return {str(seed): r16.state_sha(fresh_aux_head(backbone, seed).aux.state_dict())
            for seed in SEEDS}


# ---------------------------------------------------------------- plumbing
def all_names():
    """The six research checkpoints of this round: two arms x three seeds (arm-major)."""
    return [f'{arm}-s{seed}' for arm in ARMS for seed in SEEDS]


def names_of(arm):
    return [n for n in all_names() if n.startswith(arm + '-')]


def masked_candidate(logits):
    return r22.masked_candidate(logits)


# ---------------------------------------------------------------- round22 reference
def r22_records():
    return json.loads((R22_EVIDENCE / 'run/records.json').read_text(encoding='utf-8'))


def r22_assets():
    return {entry['event_id']: entry for entry in r22_records()['events']}


def r21_assets():
    """Round21's own records: the only place the frozen per-asset byte counts are stored."""
    path = R21_EVIDENCE / 'run/records.json'
    return {entry['event_id']: entry
            for entry in json.loads(path.read_text(encoding='utf-8'))['events']}


def r22_reference():
    """Re-read round22's frozen results and re-assert every number this round quotes."""
    prereg_path = R22_EVIDENCE / 'preregistration.json'
    prereg = json.loads(prereg_path.read_text(encoding='utf-8'))
    summary_path = R22_EVIDENCE / 'run/summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    recorded = {arm: [summary['validation']['results'][r20.name_of(arm, seed)]['F1']['feasible']
                      for seed in SEEDS] for arm in r22.ARMS}
    if recorded != R22_F1_VALIDATION_FEASIBLE:
        raise ValueError('round22 F1 validation reference drift')
    total = {arm: [summary['validation']['results'][r20.name_of(arm, seed)]['all']['feasible']
                   for seed in SEEDS] for arm in r22.ARMS}
    if total != R22_ALL_VALIDATION_FEASIBLE:
        raise ValueError('round22 total validation reference drift')
    checkpoints_path = R22_EVIDENCE / 'run/checkpoints.json'
    checkpoints = json.loads(checkpoints_path.read_text(encoding='utf-8'))
    final = {name: checkpoints[name]['final_head_state_sha256'] for name in R22_CONTROL_FINAL_HEAD_SHA}
    if final != R22_CONTROL_FINAL_HEAD_SHA:
        raise ValueError('round22 control final head reference drift')
    if not all(checkpoints[name]['matches_round21_control'] for name in R22_CONTROL_FINAL_HEAD_SHA):
        raise ValueError('round22 recorded a control that did not reproduce round21')
    if prereg['fusion']['spatial_sha256'] != R22_SPATIAL_SHA:
        raise ValueError('round22 spatial vector reference drift')
    deltas = summary['comparison']['deltas']
    if deltas != R22_COMPARISON_DELTAS or summary['comparison']['verdict'] != 'unchanged':
        raise ValueError('round22 comparison reference drift')
    direction = {}
    for arm in r22.ARMS:
        groups = summary['validation']['direction'][arm]['groups']
        direction[arm] = {group: {key: groups[group][key] for key in ('toward', 'hold', 'away')}
                          for group in ('base', 'mirror')}
    if direction != {arm: R22_DIRECTION[arm] for arm in r22.ARMS}:
        raise ValueError('round22 direction reference drift: ' + json.dumps(direction))
    gaps_path = R22_EVIDENCE / 'run/f1-gap.json'
    gaps = json.loads(gaps_path.read_text(encoding='utf-8'))
    six = r22.direction_counts(gaps['validation'],
                               networks=[r20.name_of(arm, seed) for arm in r22.ARMS for seed in SEEDS])
    if six != R22_DIRECTION['six_heads']:
        raise ValueError('round22 six-head direction reference drift: ' + json.dumps(six))
    return dict(preregistration_sha256=file_sha(prereg_path),
                summary_sha256=file_sha(summary_path),
                checkpoints_sha256=file_sha(checkpoints_path),
                records_sha256=file_sha(R22_EVIDENCE / 'run/records.json'),
                f1_gap_sha256=file_sha(gaps_path),
                baselines_sha256=file_sha(R22_EVIDENCE / 'run/baselines.json'),
                criterion_sha256=prereg['criterion_sha256'],
                spatial_sha256=R22_SPATIAL_SHA,
                feature_norm=prereg['fusion']['feature_norm'],
                f1_validation_feasible=recorded, all_validation_feasible=total,
                control_final_head_state_sha256=final,
                comparison=dict(deltas=deltas, verdict=summary['comparison']['verdict'],
                                identical_first_hit_cells=summary['comparison']['identical_first_hit_cells']),
                direction=direction, six_head_direction=six,
                decodability='round22 probe: 256-d trunk features 0.985 side accuracy on the held-out '
                             'geometry (lambda sweep 0.953-0.985); the 94-d pooled vector 0.674',
                behaviour='round22: 95-97% of telegraph-window decisions are hold; every F1 death is on '
                          'wall_frame; the fusion delta is 94.4% constant')


# ---------------------------------------------------------------- data (round22 features)
def event_records(prereg, split_name):
    """The frozen scenario payloads of one split, rebuilt from the SHA-bound R18 manifest."""
    return r20.event_records(prereg, split_name)


def train_records(prereg):
    """The 14 train events, each binding round21's frozen asset bytes (round22's own reader)."""
    return r22.train_records(prereg)


def asset_bindings(records):
    return r22.asset_bindings(records)


def encode(backbone, records, device):
    """Round22's encoder, verbatim: the same features, the same base logits, the same digest.

    Reusing `r22.spatial_vectors` instead of reimplementing it is what makes "the head sees
    exactly the features round22's head saw" checkable: that function also emits the 94-d
    spatial vector, so `spatial_digest` recomputing to round22's frozen `83a75f73...` proves
    the whole encode pass - pixels, chunking, backbone, dtype - is round22's. The spatial
    vector itself is *not* used by this round.
    """
    return r22.spatial_vectors(backbone, records, R21_EVIDENCE / 'run', device)


def feature_norms(data):
    return {row['entry']['event_id']: round(float(row['features'].norm(dim=-1).mean()), 6)
            for row in data}


def supervision_table(records):
    """The frozen auxiliary supervision: which event carries a label, and on which rows."""
    rows = {}
    for entry in records:
        world = make_world(entry['scenario'])
        terms = aux_terms(world)
        value = dict(event_id=entry['event_id'], family=entry['family'], mirror=bool(world.mirror),
                     target_defined=terms is not None)
        if terms is not None:
            value.update(side=terms['side'], label=terms['label'],
                         gap_screen=terms['gap_screen'],
                         gap_screen_centre=terms['gap_screen_centre'],
                         distance_from_centre=terms['distance_from_centre'],
                         preview=terms['preview'], supervised_rows=terms['rows'],
                         supervised_row_indices=[int(i) for i in np.nonzero(terms['mask'])[0]])
        rows[entry['event_id']] = value
    return rows


def aux_batch(world):
    """The [1,80] long target and bool mask of one world, or (None, None) without a label."""
    terms = aux_terms(world)
    if terms is None:
        return None, None
    target = torch.full((1, r20.DECISIONS), terms['side'], dtype=torch.long)
    mask = torch.from_numpy(terms['mask'][None].copy())
    return target, mask


def write_records(dest, records, supervision):
    entries, failures = [], []
    for entry in records:
        entries.append(dict(event_id=entry['event_id'], family=entry['family'], group=entry['group'],
                            split=entry['split'], mirror=entry['mirror'], rows=r20.DECISIONS,
                            horizon=r20.HORIZON, teacher_verified=entry['teacher_verified'],
                            scenario_sha256=entry['scenario_sha256'],
                            pixels=entry['pixels'], history=entry['history'], labels=entry['labels'],
                            rollout=entry['rollout'], aux=supervision[entry['event_id']],
                            bytes=sum(r21_assets()[entry['event_id']][key]['bytes']
                                      for key in ('pixels', 'history', 'labels', 'rollout'))))
        if not entry['teacher_verified']:
            failures.append(dict(event_id=entry['event_id'], verdict='not_teacher_verified'))
    labelled = [row for row in entries if row['aux']['target_defined']]
    value = dict(schema=DATA_SCHEMA, root=str(R21_EVIDENCE / 'run'), events=entries, failures=failures,
                 bindings=asset_bindings(records),
                 source='round21 telegraph assets, reopened read-only and SHA-bound; nothing re-rendered, '
                        'no teacher called, no tactical label created',
                 features='round22.spatial_vectors reused verbatim; the 94-d spatial vector it also '
                          'computes is not used by this round and only serves as the frozen proof that '
                          'the encode pass is round22\'s',
                 depth=3, region_sizes=r22.SIZES, depth_frames=r20.depth_frames(), storage=r20.STORAGE,
                 aux=dict(task=AUX_TASK, threshold=AUX_THRESHOLD, labels=list(AUX_LABELS),
                          supervision_window=AUX_SUPERVISION_WINDOW,
                          labelled_events=len(labelled),
                          unlabelled_events=[row['event_id'] for row in entries
                                             if not row['aux']['target_defined']],
                          labelled_rows=sum(row['aux']['supervised_rows'] for row in labelled),
                          label_counts={name: sum(1 for row in labelled if row['aux']['label'] == name)
                                        for name in AUX_LABELS},
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
    return model.forward_with_aux(entry['features'][None], entry['base'][None],
                                  {key: value[None] for key, value in entry['history'].items()})


def train_all(dest, data, backbone, device, prereg):
    """Two arms x three seeds through one loop; the only arm difference is the auxiliary weight.

    The loop is round20's `train_all` with three additions: the auxiliary term, its weight and
    the saved auxiliary state. Because weight 0.0 contributes an exactly-zero gradient, arm A
    must reproduce round22's control head bitwise; that equality is recorded per seed and
    reported, never silently assumed.

    The gradient clip is applied to the policy head's parameters only, identically in both
    arms. The frozen round-20/21/22 protocol clipped `model.parameters()`; here that set would
    additionally contain the 514 auxiliary parameters, whose gradient is exactly zero in arm A
    and non-zero in arm X. Keeping them out of the clip (a) makes the clipping rule literally
    the same tensor set in both arms and (b) keeps the control's floating-point path identical
    to round22's, whose parameter list was the head's. Both the choice and its consequence are
    recorded (`clip_scope`, `matches_round22_control`).
    """
    supervised = [entry for entry in data if entry['entry']['teacher_verified']]
    excluded = [entry['entry']['event_id'] for entry in data if not entry['entry']['teacher_verified']]
    if len(supervised) + len(excluded) != len(data) or not supervised:
        raise ValueError('supervision partition')
    worlds = {entry['entry']['event_id']: make_world(entry['entry']['scenario'])
              for entry in supervised}
    targets = {event_id: aux_batch(world) for event_id, world in worlds.items()}
    labelled = sorted(event_id for event_id, (target, _) in targets.items() if target is not None)
    if len(labelled) != EXPECTED['aux_events']:
        raise AssertionError('six F1 train events carry the auxiliary label')
    r20.new_dir(dest / 'train')
    reports, paths = {}, {}
    for seed in SEEDS:
        per_arm_initial = {}
        for arm in ARMS:
            name = r20.name_of(arm, seed)
            directory = r20.new_dir(dest / 'train' / name)
            model = fresh_aux_head(backbone, seed).to(device)
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
                labels = {key: value[None] for key, value in entry['labels'].items()}
                head = r20.arm_loss(LOSS_ARM, output, labels)
                if not head['parts']['has_supervision']:
                    raise AssertionError('all 200 updates must be supervised')
                target, mask = targets[event_id]
                if target is None:
                    target = torch.zeros((1, r20.DECISIONS), dtype=torch.long)
                    mask = torch.zeros((1, r20.DECISIONS), dtype=torch.bool)
                aux = aux_loss(au_logits, target.to(device), mask.to(device))
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
                          aux_weight=weight, loss_arm=LOSS_ARM,
                          auxiliary_head=model.aux.describe(),
                          clip_scope=dict(parameters=sum(p.numel() for p in clip_parameters),
                                          includes_auxiliary=False, same_in_both_arms=True,
                                          why='the 429200 policy-head parameters, identically in both '
                                              'arms; the 514 auxiliary parameters are outside the clip'),
                          matches_round22_control=(None if arm != CONTROL else
                                                   final_sha == R22_CONTROL_FINAL_HEAD_SHA[name]),
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
                                          round22_control_total=2026474,
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
                  'matches_r22', report['matches_round22_control'],
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
class PrivNetwork:
    """One frozen research checkpoint on the **parent** class: no auxiliary head exists here.

    The payload carries the auxiliary state - it is the evidence of what was trained - but this
    class never loads it: `TacticalPolicy` has no `aux` submodule, `heads` carries no auxiliary
    key, and nothing here calls the label function. Only the auxiliary *SHA* is copied, as a
    provenance identifier (never a tensor, never a computation). A unit test asserts that a
    checkpoint whose auxiliary state is replaced by noise produces identical closed-loop logits,
    and that the label function is never called on this path.
    """

    def __init__(self, backbone, path, device, expected_head_sha=None, expected_aux_sha=None):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or set(payload) != set(PAYLOAD_KEYS):
            raise ValueError('research payload keys')
        if payload['schema'] != HEADS_SCHEMA:
            raise ValueError('research schema')
        if payload['arm'] not in ARMS or payload['seed'] not in SEEDS:
            raise ValueError('research arm/seed')
        if payload['baseline_sha256'] != file_sha(BASELINE):
            raise ValueError('baseline SHA')
        weight = payload['aux_weight']
        if weight != (AUX_WEIGHT if payload['arm'] == AUXILIARY else 0.):
            raise ValueError('auxiliary weight does not match the arm')
        if payload['loss_arm'] != LOSS_ARM:
            raise ValueError('loss arm record')
        if {key.split('.')[0] for key in payload['heads']} != set(HEAD_MODULES):
            raise ValueError('the policy head state dict must carry no auxiliary or label tensor')
        self.arm = 'N1'
        self.device = device
        self.name = r20.name_of(payload['arm'], payload['seed'])
        self.model = TacticalPolicy(backbone, TacticalConfig(**payload['config'])).to(device).eval()
        if type(self.model) is not TacticalPolicy:
            raise TypeError('the evaluation network must be the unmodified parent class')
        if self.model.tactical_config.arm != 'N1':
            raise ValueError('N1 only')
        if hasattr(self.model, 'aux'):
            raise ValueError('the evaluation model must not carry an auxiliary head')
        self.model.heads.load_state_dict(payload['heads'], strict=True)
        self.heads_sha256 = r16.state_sha(self.model.heads.state_dict())
        self.aux_sha256 = payload['final_aux_state_sha256']
        if self.heads_sha256 != payload['final_head_state_sha256']:
            raise ValueError('checkpoint head SHA drift')
        if expected_head_sha is not None and self.heads_sha256 != expected_head_sha:
            raise ValueError('unexpected checkpoint')
        if expected_aux_sha is not None and self.aux_sha256 != expected_aux_sha:
            raise ValueError('unexpected auxiliary state')

    @torch.inference_mode()
    def __call__(self, arm, regions, previous, hidden, reset):
        if arm != self.arm:
            raise ValueError('callback arm mismatch')
        core = self.model.vision_model.forward_core({key: value.to(self.device)
                                                     for key, value in regions.items()})
        features = core['features'][:, None]
        base = core['policy_logits'][:, None]
        output = self.model.forward_features(features, base, executed_history(features, previous),
                                             hidden,
                                             reset_mask=torch.tensor([[reset]], device=self.device))
        return output['policy_logits'][0, 0].cpu(), output['hidden']


def evaluate_all(dest, prereg, backbone, device, reports, paths):
    validation = event_records(prereg, r20.VALIDATION_SPLIT)
    train = event_records(prereg, r20.TRAIN_SPLIT)
    r20.new_dir(dest / 'closed-loop')
    begin = time.perf_counter()
    baselines = dict(validation=r20.six_baselines(validation), train=r20.six_baselines(train))
    if sum(payload['rows'] for payload in baselines.values()) != EXPECTED['baseline_rows']:
        raise AssertionError('six baselines x 14 events x 2 splits required')
    r20.write_json(dest / 'baselines.json', baselines)
    reference = json.loads((R22_EVIDENCE / 'run/baselines.json').read_text(encoding='utf-8'))
    for split_name in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT):
        now = [(row['event_id'], row['policy'], row['feasible'], row['survived'], row['first_hit'])
               for row in baselines[split_name]['rows_detail']]
        before = [(row['event_id'], row['policy'], row['feasible'], row['survived'], row['first_hit'])
                  for row in reference[split_name]['rows_detail']]
        if now != before:
            raise AssertionError('the frozen R18 baselines changed: ' + split_name)
    closed = {}
    for split_name, events in ((r20.VALIDATION_SPLIT, validation), (r20.TRAIN_SPLIT, train)):
        rows = r21.closed_loop(events, r20.BaselineNetwork(backbone, device), 'B0', 'B0')
        payload = dict(network='B0', split=split_name, condition=r20.CONDITION,
                       renderer='v373_telegraph.render_state_telegraph', auxiliary=None, events=rows,
                       families=r16.families(rows), behaviour=r20.behaviour(rows))
        r20.write_json(dest / 'closed-loop' / f'B0-{split_name}.json', payload)
        closed[('B0', split_name)] = payload
    for name in all_names():
        network = PrivNetwork(backbone, paths[name], device,
                              reports[name]['final_head_state_sha256'],
                              reports[name]['final_aux_state_sha256'])
        for split_name, events in ((r20.VALIDATION_SPLIT, validation), (r20.TRAIN_SPLIT, train)):
            rows = r21.closed_loop(events, network, 'N1', name)
            payload = dict(network=name, split=split_name, condition=r20.CONDITION,
                           renderer='v373_telegraph.render_state_telegraph',
                           auxiliary=dict(weight=reports[name]['aux_weight'],
                                          head=reports[name]['auxiliary_head'],
                                          inference_used=False,
                                          note='the closed loop runs the unmodified parent class; the '
                                               'auxiliary head is not constructed here at all'),
                           events=rows, families=r16.families(rows), behaviour=r20.behaviour(rows),
                           heads_sha256=network.heads_sha256, aux_sha256=network.aux_sha256)
            r20.write_json(dest / 'closed-loop' / f'{name}-{split_name}.json', payload)
            closed[(name, split_name)] = payload
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    if len(closed) != EXPECTED['closed_loop_files']:
        raise AssertionError('closed-loop file count')
    return baselines, closed, seconds


def control_equivalence(closed):
    """Is arm A bit-for-bit round22's control arm on the same split and renderer?"""
    result = {}
    for seed in SEEDS:
        name = r20.name_of(CONTROL, seed)
        for split_name in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT):
            path = R22_EVIDENCE / 'run' / 'closed-loop' / f'{name}-{split_name}.json'
            reference = json.loads(path.read_text(encoding='utf-8'))
            now, before = closed[(name, split_name)]['events'], reference['events']
            fields = ('event_id', 'feasible', 'survived', 'goal_satisfied', 'first_collision_frame',
                      'decisions', 'invalid_action')
            same_rows = all(all(a[key] == b[key] for key in fields) for a, b in zip(now, before))
            same_steps = all(
                [(s.get('candidate'), s.get('actual_action'), s.get('x_after'), s.get('hit'))
                 for s in a['steps']]
                == [(s.get('candidate'), s.get('actual_action'), s.get('x_after'), s.get('hit'))
                    for s in b['steps']]
                for a, b in zip(now, before))
            result[f'{name}/{split_name}'] = dict(events=len(now), same_rows=same_rows,
                                                  same_steps=same_steps)
    result['all_equal'] = all(value['same_rows'] and value['same_steps']
                              for value in result.values() if isinstance(value, dict))
    return result


# ---------------------------------------------------------------- mechanism tables
def direction_table(closed, split_name, events):
    """One mechanism row per (network, F1 event), with round22's cell diagnostic verbatim."""
    rows = {}
    for name in ['B0'] + all_names():
        by_event = {row['event_id']: row for row in closed[(name, split_name)]['events']}
        for entry in events:
            if entry['family'] != 'F1':
                continue
            world = make_world(entry['scenario'])
            rows[f'{name}/{entry["event_id"]}'] = r22.cell_diagnostic(
                world, by_event[entry['event_id']], name, split_name)
    if len(rows) != EXPECTED['f1_gap_rows'] // 2:
        raise AssertionError('one mechanism row per network and F1 event required')
    return rows


def mechanism_block(rows):
    return r22.mechanism_block(rows, {CONTROL: names_of(CONTROL), AUXILIARY: names_of(AUXILIARY),
                                      'B0': ['B0']})


def timing_cells(rows):
    """Per cell: did a cell that moved toward the gap also reach it before the wall?"""
    out = {}
    for key, row in rows.items():
        out[key] = dict(
            network=row['network'], mirror=bool(row['mirror']), verdict=row['verdict'],
            need=row['need'], direction=row['direction'],
            first_move_frame=row['first_move_frame'], entered_gap=row['entered_gap'],
            entered_gap_frame=row['entered_gap_frame'],
            entered_gap_decision=row['entered_gap_decision'],
            frames_in_gap_before_wall=row['frames_in_gap_before_wall'],
            in_gap_at_wall=row['in_gap_at_wall'], canonical_at_wall=row['canonical_at_wall'],
            signed_error_px=row['signed_error_px'], wall_frame=row['wall_frame'],
            hit_frame=row['hit_frame'], mechanism=row['mechanism'],
            toward_but_late=bool(row['verdict'] == 'toward' and not row['entered_gap']),
            note='toward-but-late is visible here: a cell that moved toward the gap but has '
                 'entered_gap=False never reached the slot before the wall became lethal')
    return out


# ---------------------------------------------------------------- probes
def preview_indices(world):
    window = r21.telegraph_window(world)
    if window is None:
        return []
    return [index for index, frame in enumerate(r21.DECISION_FRAMES) if window[0] <= frame < window[1]]


def stored_actions(closed, split_name, events):
    """Round22's stored-prediction table with this round's arm names (that helper hard-codes
    round22's `S` arm, so it is re-expressed here rather than reached through)."""
    table = {}
    for name in ['B0'] + all_names():
        by_event = {row['event_id']: row for row in closed[(name, split_name)]['events']}
        table[name] = {}
        for entry in events:
            if entry['family'] != 'F1':
                continue
            world = make_world(entry['scenario'])
            window = r21.telegraph_window(world)
            steps = by_event[entry['event_id']]['steps']
            inside = [index for index, step in enumerate(steps)
                      if window[0] <= int(step['frame']) < window[1]]
            for label, subset in (('all', steps), ('preview', [steps[i] for i in inside])):
                directions = [r22.screen_direction(r22.screen_action(step.get('candidate')))
                              for step in subset]
                table[name].setdefault(entry['event_id'], {})[label] = dict(
                    decisions=len(directions),
                    right=sum(1 for value in directions if value == 'right'),
                    left=sum(1 for value in directions if value == 'left'),
                    hold=sum(1 for value in directions if value == 'hold'))
    return table


def load_variant(backbone, device, path, report):
    """The training class with both states loaded: used by the probes, never by the closed loop."""
    model = fresh_aux_head(backbone, report['seed']).to(device)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model.heads.load_state_dict(payload['heads'], strict=True)
    model.aux.load_state_dict(payload['aux_state'], strict=True)
    if r16.state_sha(model.heads.state_dict()) != report['final_head_state_sha256']:
        raise ValueError('probe: head SHA drift')
    if r16.state_sha(model.aux.state_dict()) != report['final_aux_state_sha256']:
        raise ValueError('probe: auxiliary SHA drift')
    return model.eval()


def aux_replay(model, device, world, steps):
    """Replay one stored trajectory through the variant model and read the auxiliary head.

    The replay is the runner's own input sequence (round22's `replay_images`), so the
    auxiliary head is asked exactly the question the policy faced. The masked argmax is
    compared with the stored candidate at every step, so "the probe saw the runner inputs" is
    measured (`stored_candidate_matches`) rather than claimed.
    """
    if world.family != 'F1':
        raise ValueError('the auxiliary replay is F1 only')
    images = r22.replay_images(world, steps)
    hidden, previous = None, None
    matches = 0
    states, logits, candidates = [], [], []
    with torch.inference_mode():
        for index, step in enumerate(steps):
            regions, _ = pixel_stack(images[:index + 1])
            core = model.vision_model.forward_core({key: value.to(device)
                                                    for key, value in regions.items()})
            features = core['features'][:, None]
            base = core['policy_logits'][:, None]
            output, au = model.forward_with_aux(
                features, base, executed_history(features, previous), hidden,
                reset_mask=torch.tensor([[index == 0]], device=device))
            candidate = masked_candidate(output['policy_logits'][0, 0])
            matches += int(candidate == int(step['candidate']))
            states.append(model.state()[0, 0].detach().cpu().numpy().astype(np.float32))
            logits.append(au[0, 0].detach().cpu().numpy().astype(np.float32))
            candidates.append(candidate)
            previous, hidden = candidate, output['hidden']
    return dict(stored_candidate_matches=matches, steps=len(steps),
                states=np.stack(states), logits=np.stack(logits), candidates=candidates)


def replay_pass(backbone, device, closed, prereg, paths, reports):
    """One replay per (network, F1 validation event), shared by every probe of this round."""
    validation_f1 = [row for row in event_records(prereg, r20.VALIDATION_SPLIT)
                     if row['family'] == 'F1']
    out, fidelity = {}, {}
    for name in all_names():
        model = load_variant(backbone, device, paths[name], reports[name])
        for entry in validation_f1:
            world = make_world(entry['scenario'])
            steps = next(row for row in closed[(name, r20.VALIDATION_SPLIT)]['events']
                         if row['event_id'] == entry['event_id'])['steps']
            replayed = aux_replay(model, device, world, steps)
            fidelity[f'{name}/{entry["event_id"]}'] = round(
                replayed['stored_candidate_matches'] / replayed['steps'], 6)
            out[(name, entry['event_id'])] = dict(
                world=world, group=world.template_group, terms=aux_terms(world),
                index=preview_indices(world), states=replayed['states'],
                logits=replayed['logits'])
        print('replayed', name, 'fidelity', round(sum(value for key, value in fidelity.items()
                                                      if key.startswith(name + '/'))
                                                  / len(validation_f1), 6), flush=True)
    return out, fidelity


def seen_pass(backbone, device, data, reports, paths, prereg):
    """The auxiliary head on the stored (seen) training features: one 80-row forward per event."""
    train_f1 = [row['event_id'] for row in event_records(prereg, r20.TRAIN_SPLIT)
                if row['family'] == 'F1']
    by_event = {row['entry']['event_id']: row for row in data}
    out = {}
    for name in all_names():
        model = load_variant(backbone, device, paths[name], reports[name])
        with torch.inference_mode():
            for event_id in train_f1:
                _, au = infer_with_aux(model, by_event[event_id])
                out[(name, event_id)] = au[0].detach().cpu().numpy().astype(np.float32)
    return out


def aux_summary(truth, pred):
    """Accuracy, balanced accuracy and the 2x2 confusion of one auxiliary read-out."""
    confusion = {name: {other: 0 for other in AUX_LABELS} for name in AUX_LABELS}
    for index in range(len(truth)):
        confusion[AUX_LABELS[int(truth[index])]][AUX_LABELS[int(pred[index])]] += 1
    per_class, recalls = {}, []
    for name in AUX_LABELS:
        support = sum(confusion[name].values())
        per_class[name] = dict(support=support, correct=confusion[name][name],
                               recall=round(confusion[name][name] / support, 6) if support else None)
        if support:
            recalls.append(confusion[name][name] / support)
    rows = len(truth)
    correct = sum(confusion[name][name] for name in AUX_LABELS)
    counts = [sum(1 for value in truth if int(value) == index) for index in range(AUX_DIM)]
    return dict(rows=rows, correct=correct,
                accuracy=round(correct / rows, 6) if rows else None,
                chance=0.5,
                majority_baseline=round(max(counts) / rows, 6) if rows else None,
                balanced_accuracy=round(sum(recalls) / len(recalls), 6) if recalls else None,
                confusion=confusion, per_class=per_class)


def aux_probe(closed, replay, seen, fidelity, prereg, reports):
    """Requirement 3: the auxiliary head's accuracy and confusion, seen and held out.

    Two read-outs, neither of which ever feeds the label:
      * `seen` - the stored round21 training features of the six F1 train events, i.e. exactly
        the rows the auxiliary loss saw: this is fit, not generalisation;
      * `holdout` - the six F1 validation events, replayed along each checkpoint's own stored
        closed-loop trajectory (the pixels are re-rendered from that trajectory, so no
        validation asset is read and no teacher is called). The label is used only to *score*
        the prediction, which is what a supervised metric is.
    """
    train_f1 = [row for row in event_records(prereg, r20.TRAIN_SPLIT) if row['family'] == 'F1']
    validation_f1 = [row for row in event_records(prereg, r20.VALIDATION_SPLIT)
                     if row['family'] == 'F1']
    result = dict(seen={}, holdout={}, replay_fidelity=fidelity)
    for split_name in ('seen', 'holdout'):
        for name in all_names():
            truth, pred, per_event, stack = [], [], {}, []
            entries = train_f1 if split_name == 'seen' else validation_f1
            for entry in entries:
                world = make_world(entry['scenario'])
                terms = aux_terms(world)
                index = preview_indices(world)
                logits = (seen[(name, entry['event_id'])][index] if split_name == 'seen'
                          else replay[(name, entry['event_id'])]['logits'][index])
                predicted = int(logits.mean(0).argmax())
                correct = int((logits.argmax(-1) == terms['side']).sum())
                per_event[entry['event_id']] = dict(
                    side=terms['side'], label=terms['label'], predicted=side_name(predicted),
                    predicted_side=predicted, correct=correct, rows=int(len(index)),
                    accuracy=round(correct / len(index), 6),
                    mean_probability_right=round(float(torch.softmax(torch.from_numpy(logits), -1)[:, 1]
                                                       .mean()), 6),
                    margin=round(float((logits[:, 1] - logits[:, 0]).mean()), 6),
                    gap_screen_centre=terms['gap_screen_centre'],
                    distance_from_centre=terms['distance_from_centre'],
                    replay_fidelity=(None if split_name == 'seen'
                                     else fidelity[f'{name}/{entry["event_id"]}']))
                truth.extend([terms['side']] * len(index))
                pred.extend(logits.argmax(-1).tolist())
                stack.append(logits)
            result[split_name][name] = dict(
                arm=reports[name]['arm'], seed=reports[name]['seed'], aux_weight=reports[name]['aux_weight'],
                per_event=per_event, overall=aux_summary(truth, pred))
            print('aux', split_name, name, 'accuracy', result[split_name][name]['overall']['accuracy'],
                  'balanced', result[split_name][name]['overall']['balanced_accuracy'],
                  'confusion', result[split_name][name]['overall']['confusion'], flush=True)
    result['by_arm'] = {}
    for arm in ARMS:
        result['by_arm'][arm] = {}
        for split_name in ('seen', 'holdout'):
            rows = [result[split_name][name]['overall'] for name in names_of(arm)]
            result['by_arm'][arm][split_name] = dict(
                seeds=len(rows),
                accuracy=round(sum(row['correct'] for row in rows)
                               / sum(row['rows'] for row in rows), 6),
                balanced_accuracy=round(sum(row['balanced_accuracy'] for row in rows) / len(rows), 6),
                per_seed_accuracy=[row['accuracy'] for row in rows])
    result['design'] = dict(
        task=AUX_TASK, threshold=AUX_THRESHOLD, labels=list(AUX_LABELS),
        supervision=AUX_SUPERVISION_WINDOW,
        rows='the preview-window decisions only, for every event',
        seen='the stored round21 training features of the six F1 train events: exactly the rows the '
             'auxiliary loss trained on, so this is fit',
        holdout='the six F1 validation events, each replayed along that checkpoint\'s own stored '
                'closed-loop trajectory, re-rendered from its own x trace; the label scores the '
                'prediction and is never an input',
        chance='0.5: the six labels are balanced 3/3 on both splits, so a majority baseline cannot '
               'beat chance',
        label_use='the label exists only inside the training loss and inside this scoring function; '
                  '`PrivNetwork` (the closed loop) is the unmodified parent class and cannot read it')
    return result


def state_probe(replay, fidelity, reports):
    """Requirement 4's first half: is the gap side more linearly decodable from the GRU state?

    Round22 measured 98.5% side accuracy from the frozen 256-d trunk features and 67.4% from
    its 94-d pooled vector, both fitted on train pixels with a leave-one-geometry-out split.
    This probe asks the same question of the **GRU state** - the tensor the auxiliary head
    reads - on the six F1 validation events, leaving one geometry group out. If arm X's state
    decodes the side better than arm A's while the closed loop does not move, the auxiliary
    task shaped the representation without changing the behaviour, which is exactly the
    "learned but unused" reading this round has to be able to report.
    """
    result = {}
    for name in all_names():
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
        values = [result[name]['side_accuracy'] for name in names_of(arm)]
        result['by_arm'][arm] = dict(seeds=len(values),
                                     side_accuracy=round(sum(values) / len(values), 6),
                                     per_seed=values, minimum=min(values), maximum=max(values))
    result['reference'] = dict(
        note='round22 measured this target from the frozen 256-d trunk features at 0.985 and from its '
             '94-d pooled vector at 0.674; the numbers here are about the GRU state, not the trunk '
             'features, so they are not comparable in level - they are comparable *between the two '
             'arms* of this round, which is the question being asked',
        round22_features=0.985, round22_spatial=0.674)
    result['design'] = dict(
        target='binary: the gap\'s screen centre above x=192, i.e. the auxiliary label',
        rows='the preview-window decisions of the six F1 validation events, from the replay',
        fit='ridge on standardised inputs with an intercept, leave one geometry group out',
        lambda_sweep=list(r22.RIDGE['lambda_sweep']), headline_lambda=r22.RIDGE['lambda_'],
        caveat='132 rows against 256 dimensions and a label that is constant within an event: the '
               'probe separates events as much as frames, so only the whole lambda sweep and the '
               'arm-to-arm comparison are read, never a single value')
    return result


def usage_table(closed, aux, prereg):
    """The "learned but unused" table: what the auxiliary head read vs what the policy did."""
    rows = {}
    for name in all_names():
        for entry in event_records(prereg, r20.VALIDATION_SPLIT):
            if entry['family'] != 'F1':
                continue
            world = make_world(entry['scenario'])
            stored = next(row for row in closed[(name, r20.VALIDATION_SPLIT)]['events']
                          if row['event_id'] == entry['event_id'])
            cell = r22.cell_diagnostic(world, stored, name, r20.VALIDATION_SPLIT)
            terms = aux_terms(world)
            read = aux['holdout'][name]['per_event'][entry['event_id']]
            rows[f'{name}/{entry["event_id"]}'] = dict(
                network=name, arm=name.split('-')[0], seed=int(name.rsplit('-s', 1)[1]),
                event_id=entry['event_id'], mirror=bool(world.mirror),
                aux_true_side=terms['side'], aux_true_label=terms['label'],
                aux_predicted_side=read['predicted_side'], aux_predicted_label=read['predicted'],
                aux_correct=bool(read['predicted_side'] == terms['side']),
                aux_row_accuracy=read['accuracy'], aux_margin=read['margin'],
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
            toward_but_late=sum(1 for row in subset
                                if row['policy_verdict'] == 'toward' and not row['entered_gap']))
    return dict(cells=rows, by_arm=counts,
                note='one row per (network, F1 validation event); `aux_*` is the auxiliary head on the '
                     'replayed pixels, `policy_*` is the stored closed loop of the same checkpoint; a '
                     'high aux correctness with a low policy toward count is the "the task was learned, '
                     'the behaviour did not use it" reading, cell by cell')


def run_probes(backbone, device, closed, prereg, records, data, paths, reports, gaps):
    """Every mechanism probe of this round, all read-only on frozen artifacts."""
    replay, fidelity = replay_pass(backbone, device, closed, prereg, paths, reports)
    seen = seen_pass(backbone, device, data, reports, paths, prereg)
    aux = aux_probe(closed, replay, seen, fidelity, prereg, reports)
    state = state_probe(replay, fidelity, reports)
    usage = usage_table(closed, aux, prereg)
    stored = stored_actions(closed, r20.VALIDATION_SPLIT,
                            [row for row in event_records(prereg, r20.VALIDATION_SPLIT)
                             if row['family'] == 'F1'])
    return dict(aux=aux, state=state, usage=usage,
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


# ---------------------------------------------------------------- pre-registration
def criterion():
    """The frozen decision rules of this round; any drift aborts the run."""
    return dict(
        round='round23',
        purpose='one pre-registered intervention on top of round22: add a training-only privileged '
                'auxiliary task - predict which screen half holds the safe gap from the GRU state - so '
                'that the objective, not just the representation, has to use the position round22 '
                'proved is linearly decodable at 98.5%, and read the result out mechanistically',
        intervention=dict(
            change='a Linear(256,2) auxiliary head on the GRU state, plus an auxiliary cross-entropy '
                   'term in the training loss with a fixed weight',
            task=AUX_TASK,
            label=dict(source='s.gap under s.mirror through round21.telegraph_gap, i.e. exactly the '
                              'arithmetic the R21 renderer uses to draw the band',
                       threshold=AUX_THRESHOLD, classes=list(AUX_LABELS),
                       availability='training only; never an input, never a history field, never a pixel '
                                    'region, never a tensor of the policy head'),
            supervision=dict(window=AUX_SUPERVISION_WINDOW,
                             why='outside the telegraph window the band is not drawn yet, so the label '
                                 'would ask the head to guess rather than to read; F2/F3 worlds have '
                                 '`internal_gap = False` and therefore no label at all',
                             labelled_events='the six F1 train events (3 groups x base/mirror)',
                             unlabelled_events='the eight F2/F3 train events contribute an exactly-zero '
                                               'auxiliary term'),
            head=dict(module='Linear(256, 2, bias=True)', parameters=AUX_PARAMETERS,
                      placement='outside `heads`, so the policy head state dict, its SHA and the whole '
                                'checkpoint contract are unchanged from round22'),
            loss='L = L_control + w * CE(aux_logits, side);  w = 0.0 for arm A, 0.5 for arm X',
            weight=AUX_WEIGHT,
            gradient_path='the auxiliary term reaches only the auxiliary head and the shared GRU; the '
                          'residual head, the option/target/urgency/termination/slack heads and the frozen '
                          'backbone are untouched by it',
            head_class='no line of TacticalPolicy is overridden: the auxiliary head is downstream of the '
                       'GRU, so the parent detach of `features` is inert for it (contrast round22, whose '
                       'upstream fusion layer required the detach to be removed)',
            state_source='forward pre-hook on heads["%s"]; the unit tests recompute all five heads that '
                         'consume it from the captured tensor and assert bitwise equality'
                         % STATE_HOOK_MODULE,
            not_changed='worlds, world parameters, R19 split, round21 renderer and pixels, teacher '
                        'actions, tactical labels, pixel packing, encoder, control loss, optimizer, '
                        'learning rate, clip value, clip scope (the policy head in both arms), sequence, '
                        'batch, update count, seed set, closed-loop runner, condition, masking',
            parameter_increment='X and A are identical objects: 429200 policy-head parameters + 514 '
                                'auxiliary parameters = 429714 active; the arms differ only in w'),
        data=dict(source='round21 run/records.json assets, reopened read-only and SHA-bound per event; '
                         'nothing re-rendered, no teacher called, no tactical label created',
                  features='round22.spatial_vectors reused verbatim; its spatial digest must recompute '
                           'to round22\'s frozen 83a75f73... so "the head sees round22\'s features" is a '
                           'measured identity, not a copy',
                  train=dict(events=14, groups=7, groups_per_family={'F1': 3, 'F2': 1, 'F3': 3},
                             use='the only events encoded or trained on'),
                  validation=dict(events=14, groups=7, groups_per_family={'F1': 3, 'F2': 1, 'F3': 3},
                                  use='read-only: never trained, never labelled for training, never '
                                      'encoded into the training set; closed loop and diagnostic probes '
                                      'only. The auxiliary probe re-renders each checkpoint\'s own stored '
                                      'trajectory and uses the label only to score it'),
                  bindings='pixels.npz, history.pt, labels.pt and rollout.json must equal round21\'s '
                           'frozen SHA table for all 14 train events, otherwise the run aborts'),
        training=dict(arms={CONTROL: 'the same auxiliary head with weight 0.0: the round22 control',
                            AUXILIARY: 'the same auxiliary head with weight 0.5'},
                      loss_arm=LOSS_ARM, aux_weight=AUX_WEIGHT, seeds=list(SEEDS), updates=r20.UPDATES,
                      optimizer='Adam', lr=r20.LR, clip=r20.CLIP,
                      clip_scope='the 429200 policy-head parameters, identically in both arms; the '
                                 'auxiliary head is outside the clip so that the clipping rule covers the '
                                 'same tensor set in both arms and the control\'s floating-point path stays '
                                 'round22\'s',
                      sequence=r20.SEQUENCE, batch=r20.BATCH,
                      precision='FP32 without AMP, autocast disabled, matmul TF32 False, cuDNN TF32 False, '
                                'float32 matmul precision highest',
                      loop='round20 train_all with three additions only: the auxiliary term, its weight and '
                           'the saved auxiliary state',
                      initialization='fresh seed reset per construction; the head is built first, so the '
                                     'policy head SHA must equal the frozen R16/R20/R21/R22 table for the '
                                     'seed; the auxiliary head is built afterwards from the same RNG state '
                                     'in both arms, so its SHA must be identical in both arms',
                      control_reproduction='arm A must reproduce round22\'s A checkpoint and its '
                                           'closed-loop rows bitwise; if it does not, that is reported as '
                                           '`control_not_reproduced` and the causal claim is withheld',
                      selection='final step only: no early stopping, no validation-based selection, no '
                                'tuning of lr/clip/weight/epochs/order; the auxiliary weight is fixed at 0.5 '
                                'before the run'),
        evaluation=dict(
            order='all six checkpoints written and SHA-frozen first; then the six R18 baselines, then B0, '
                  'then the six checkpoints, validation before train; then the probes',
            executor='round21.run_event_rendered / round21.closed_loop reused verbatim, so the evaluation '
                     'code is round21 code and not a copy; the network is a plain TacticalPolicy, so the '
                     'auxiliary head has no path into the closed loop',
            condition=r20.CONDITION, renderer='v373_telegraph.render_state_telegraph',
            b0='the frozen baseline policy logits on the identical telegraph renders; a reference, not a '
               'capability',
            six_baselines='round20.six_baselines re-executed on the same worlds; the 168 rows must equal '
                          'round22\'s',
            holdout_rule='if ANY of the six R18 baselines is feasible on ANY validation event the R18 '
                         'filter has failed on the holdout side: report it verbatim and withhold the '
                         'learning conclusion',
            definitions=dict(
                direction='rebuild the logical-frame x trace from the executed actions with `advance` '
                          'semantics (round21.gap_trace), take the canonical displacement start -> '
                          'wall_frame-1: that is `direction`; `screen_direction` is the same displacement '
                          'in raw x',
                toward='the canonical direction points from the start\'s canonical x at the gap\'s '
                       'canonical centre, so `toward` is "moved toward the slot" regardless of the mirror; '
                       '`away` is the opposite and `hold` is zero net motion',
                parked='`signed_error_px` is measured at wall_frame-1 in canonical px: 0 inside '
                       '[gap0, gap1], negative short of the left edge, positive past the right edge',
                mirror_flip_rate='the fraction of mirror-group cells whose canonical motion is toward the '
                                 'gap; because the same screen-right prior is right in the base group and '
                                 'wrong in the mirror group, this is the direction flip this round is about',
                identical_cells='for every seed, the number of the 14 validation events on which the two '
                                'arms have the same first-hit frame',
                toward_but_late='a cell whose verdict is `toward` and whose entered_gap is False: it moved '
                                'the right way and still never reached the slot before the wall',
                aux_accuracy='the auxiliary head on the preview-window rows of the six F1 validation '
                             'events, replayed along each checkpoint\'s own stored trajectory; the label '
                             'scores the prediction and is never an input; chance is 0.5 because the six '
                             'labels are balanced 3/3 on both splits'),
            primary_metrics=[
                'F1 holdout, per group (base / mirror) and per arm: cells that move toward the gap, hold, '
                'or move away, plus the mirror toward rate (the direction flip this round is about)',
                'F1 holdout, per cell: whether the x at wall_frame-1 is inside the gap, the signed parking '
                'error in px, whether the cell moved toward the gap but never reached it before the wall, '
                'and the frame of first movement',
                'per family F1/F2/F3 and pooled: feasible and survived per network, every event first hit, '
                'totals, and for each seed the number of validation events whose first hit is identical '
                'between the arms',
                'the auxiliary head\'s accuracy, balanced accuracy and 2x2 confusion on the held-out '
                'geometry (and on the seen training rows as fit), per arm, per seed, per event',
                'the "learned but unused" read-out: per cell, the auxiliary side read, its correctness, '
                'the policy\'s screen direction, the verdict and the parking error side by side; plus the '
                'ridge decodability of the same side from the GRU state, arm A against arm X'],
            probe_caveats=dict(
                aux_accuracy='the label is balanced 3/3 on both splits, so chance is 0.5 and a majority '
                             'baseline cannot beat it; but the label is constant within an event, so an '
                             'event-level read-out is the honest unit and the per-row accuracy is only a '
                             'summary of it',
                label_leakage='the label is a function of exactly the geometry the renderer draws, so the '
                              'auxiliary task adds no information that is not in the pixels - it adds a '
                              'target. What is privileged is the label, not the content',
                decodability='132 rows against 256 dimensions with a label constant within an event: the '
                              'probe separates events as much as frames, so only the whole lambda sweep and '
                              'the arm-to-arm comparison are read',
                replay='the replay re-renders each checkpoint\'s own x trace; its fidelity (the stored '
                       'masked argmax reproduced step by step) is reported for every cell'),
            reporting='per family, never only aggregated; n=3 seeds; F2 validation has one group; mirrors '
                      'are not independent samples; every F1 cell is listed',
            degeneracy='single_candidate and switches<=1 are counted, listed and reported, never dropped '
                       'silently'),
        verdict_rules=dict(
            equivalence='arm A must reproduce round22\'s A checkpoint SHA and every closed-loop row '
                        'bitwise; if it does not, the round reports `control_not_reproduced` and withholds '
                        'the causal claim',
            improvement='F1 validation feasible per seed, X minus A: any seed strictly greater = improved, '
                        'all equal = unchanged, any seed lower = worse',
            mechanism='the mirror toward rate, the parking errors and the auxiliary accuracies are reported '
                      'as counts, px and rates, never as a significance claim',
            honesty='if the mirror direction does not improve while the auxiliary head is accurate, the '
                    'round must report the "the task was learned, the policy did not use it" reading with '
                    'its cell-level evidence; if the mirror direction flips but the cells die on timing, '
                    'the round must report the flip and the timing failure together; neither may be left '
                    'implicit',
            aux_learned='the auxiliary task counts as learned on the holdout only if the balanced accuracy '
                        'exceeds 0.5 and the event-level read is correct on the same side of chance for '
                        'all three seeds; a single seed above chance is reported as one seed, not as '
                        'learned'),
        claims=dict(significance='none: 3 seeds, 7 validation groups, at most 14 independent geometries; '
                                 'descriptive only',
                    gate='no formal gate opened, no deployable checkpoint produced, no game executed',
                    can_show='whether a fixed-weight privileged auxiliary task on the shared GRU state '
                             'makes the same N1 head stop being dominated by the screen-right prior, and '
                             'if not, whether the auxiliary task was learned at all',
                    cannot_show='tactical understanding, learnability beyond these 7 held-out geometries, '
                                'or that a privileged label at training time is a legitimate route to a '
                                'deployable policy'),
        failure_policy='first failure retained; one run; no retrain, rerun, repair of numbers, extra epoch, '
                       'tuning or baseline switch')


def criterion_sha(value=None):
    return hashlib.sha256(json.dumps(criterion() if value is None else value, sort_keys=True).encode()).hexdigest()


def identity_paths():
    return [Path(__file__), Path('tests/test_v373_privileged_aux.py'),
            Path('tools/rl/v373_spatial_fuse.py'), Path('tests/test_v373_spatial_fuse.py'),
            Path('tools/rl/v373_telegraph.py'), Path('tests/test_v373_telegraph.py'),
            Path('tools/rl/v373_valid_curriculum.py'), Path('tests/test_v373_valid_curriculum.py'),
            Path('tools/rl/v373_seed_confirm.py'), Path('tools/rl/v373_world_filter.py'),
            Path('touhou_rl/tactical_policy.py'), Path('touhou_rl/macro_policy.py'),
            Path('touhou_rl/vmodels3.py'), Path('touhou_rl/vmodels.py'), Path('touhou_rl/vision.py'),
            Path('touhou_rl/tactical_synthetic_pixels.py'),
            Path('touhou_rl/tactical_teacher_validation.py'), Path('touhou_rl/tactical_teacher.py'),
            Path('touhou_rl/tactical_teacher_v2.py'), Path('touhou_rl/tactical_synthetic_closed_loop.py'),
            Path('touhou_rl/tactical_scenarios.py'), Path('touhou_rl/tactical_data.py'),
            Path('touhou_rl/tactical_sequence_train.py'), Path('touhou_rl/tactical_feature_cache.py'),
            BASELINE, r20.MANIFEST, r20.SPLIT,
            R22_EVIDENCE / 'preregistration.json', R22_EVIDENCE / 'run/records.json',
            R22_EVIDENCE / 'run/summary.json', R22_EVIDENCE / 'run/checkpoints.json',
            R22_EVIDENCE / 'run/f1-gap.json', R22_EVIDENCE / 'run/baselines.json',
            REPORTS / 'P4-spatialfuse-round22.md', REPORTS / 'P4-telegraph-round21.md']


def identities():
    return {str(path): file_sha(path) for path in identity_paths()}


def world_table(events):
    return r22.world_table(events)


def preregister(out=OUT, device='cuda'):
    out = r20.new_dir(out)
    manifest = r20.load_worlds()
    split = r20.load_split(manifest)
    train = r20.world_events(manifest, split, r20.TRAIN_SPLIT)
    validation = r20.world_events(manifest, split, r20.VALIDATION_SPLIT)
    checks = r20.split_checks(manifest, split, train, validation)
    initial = r16.initial_head_shas()
    if len(set(initial.values())) != len(SEEDS):
        raise ValueError('different seeds must give different initial head states')
    if sorted(r22_assets()) != sorted(row['event_id'] for row in train):
        raise ValueError('round22 assets must cover exactly the 14 train events')
    reference = r22_reference()
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
    if digest != reference['spatial_sha256']:
        raise ValueError('the encode pass is not round22\'s: spatial digest drift')
    if norms != reference['feature_norm']:
        raise ValueError('the frozen feature norm table drifted from round22')
    supervision = supervision_table(records)
    validation_labels = supervision_table(validation)
    labelled_train = {key: value for key, value in supervision.items() if value['target_defined']}
    labelled_validation = {key: value for key, value in validation_labels.items()
                           if value['target_defined']}
    if len(labelled_train) != EXPECTED['aux_events']:
        raise AssertionError('six F1 train events carry the auxiliary label')
    if sum(value['supervised_rows'] for value in labelled_train.values()) != EXPECTED['aux_train_rows']:
        raise AssertionError('132 supervised auxiliary rows expected on the train split')
    if len(labelled_validation) != EXPECTED['aux_events']:
        raise AssertionError('six F1 validation events carry the auxiliary label')
    if sum(value['supervised_rows'] for value in labelled_validation.values()) != \
            EXPECTED['aux_validation_rows']:
        raise AssertionError('128 preview-window rows expected on the validation split')
    aux_per_seed = per_seed_aux_shas(backbone)
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
                     round22=dict(preregistration=file_sha(R22_EVIDENCE / 'preregistration.json'),
                                  records=reference['records_sha256'],
                                  summary=reference['summary_sha256'],
                                  checkpoints=reference['checkpoints_sha256'],
                                  f1_gap=reference['f1_gap_sha256'],
                                  baselines=reference['baselines_sha256'])),
        worlds=world,
        split=dict(rule=split['spec']['rule'], counts=split['spec']['counts'], assertions=checks,
                   groups=[dict(group=group['group'], family=group['family'], split=group['split'],
                                index_in_family=group['index_in_family'], signature=group['signature'],
                                events=group['events']) for group in split['groups']]),
        auxiliary=dict(task=AUX_TASK, threshold=AUX_THRESHOLD, labels=list(AUX_LABELS),
                       dim=AUX_DIM, parameters=AUX_PARAMETERS,
                       supervision_window=AUX_SUPERVISION_WINDOW,
                       head='Linear(256, 2, bias=True) outside `heads`',
                       loss='L = L_control + w * CE(aux_logits, side)',
                       weight=dict(control=0., auxiliary=AUX_WEIGHT),
                       state_source='forward pre-hook on heads["%s"]' % STATE_HOOK_MODULE,
                       parent_detach='inert: the auxiliary head is downstream of the GRU, so no line of '
                                     'the parent body is overridden',
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
                       initial_state_sha256_per_seed=aux_per_seed,
                       initial_state_sha256_per_arm=aux_init_shas(),
                       inference='the closed loop builds a plain TacticalPolicy from payload["heads"]; the '
                                 'auxiliary state is never loaded and there is no module to hold it'),
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
                            equals_round16_round20_round21_round22=True,
                            method='fresh seed reset immediately before every N1 construction after the '
                                   'backbone load, through the frozen r16 recipe; the auxiliary head is '
                                   'constructed afterwards, so it cannot move the head initialisation and '
                                   'both arms are byte-identical objects at step 0'),
        round22_reference=reference,
        expected=EXPECTED,
        verification=dict(
            prerequisites=['every training asset equals round21\'s frozen SHA',
                           'the round22 spatial digest recomputes to 83a75f73...',
                           'the round22 feature-norm table recomputes exactly',
                           'the initial head SHA equals the frozen table for every seed',
                           'the initial auxiliary SHA equals the pre-registration for every seed and is '
                           'identical in both arms',
                           'the six R18 baselines reproduce round22\'s 168 rows'],
            control='arm A must equal round22\'s A checkpoint SHA and its closed-loop rows for all 3 '
                    'seeds; otherwise it is reported as `control_not_reproduced` and the causal claim is '
                    'withheld',
            step0='a unit test asserts that step 0 of arm X is bitwise arm A (identical policy logits, '
                  'identical masked action) and that perturbing the auxiliary parameters changes no policy '
                  'logit'),
        independent_test=False, formal_gate=False, deployable=False, game=False, tuning=False,
        new_worlds=False, new_teacher_labels=False, teacher_calls=0, official_labels=False,
        research_teacher_labels=True, new_pixels=False, privileged_label=True,
        privileged_label_scope='training loss and scoring only',
        failure_policy='first failure retained; one run; no retrain, rerun, repair, overwrite, extra epoch '
                       'or baseline switch')
    r20.write_json(out / 'preregistration.json', value)
    return value


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
    if json.dumps(r22_reference(), sort_keys=True) != json.dumps(p['round22_reference'], sort_keys=True):
        raise ValueError('the round22 reference record changed')
    train = r20.world_events(manifest, split, r20.TRAIN_SPLIT)
    validation = r20.world_events(manifest, split, r20.VALIDATION_SPLIT)
    if json.dumps(world_table(train + validation), sort_keys=True) != json.dumps(p['geometry'], sort_keys=True):
        raise ValueError('the frozen geometry table changed')
    if sorted(r22_assets()) != sorted(p['worlds']['train_events']):
        raise ValueError('round22 asset set changed')
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
    supervision = supervision_table(records)
    labelled = {key: value for key, value in supervision.items() if value['target_defined']}
    if json.dumps(labelled, sort_keys=True) != \
            json.dumps(p['auxiliary']['supervised_events']['train'], sort_keys=True):
        raise ValueError('the auxiliary supervision table changed since the pre-registration')
    validation_labels = {key: value for key, value in supervision_table(validation).items()
                         if value['target_defined']}
    if json.dumps(validation_labels, sort_keys=True) != \
            json.dumps(p['auxiliary']['validation_labels'], sort_keys=True):
        raise ValueError('the frozen validation auxiliary labels changed')
    if per_seed_aux_shas(backbone) != p['auxiliary']['initial_state_sha256_per_seed']:
        raise ValueError('the auxiliary initialisation drifted')
    if aux_init_shas() != p['auxiliary']['initial_state_sha256_per_arm']:
        raise ValueError('the auxiliary construction drifted')
    return p


# ---------------------------------------------------------------- summary
def comparison(closed):
    """X minus A on the same seed and the same six F1 validation events, plus the cell check."""
    per_arm = {}
    for arm in ARMS:
        per_arm[arm] = [dict(seed=seed, network=r20.name_of(arm, seed),
                             f1=closed[(r20.name_of(arm, seed), r20.VALIDATION_SPLIT)]['families']['F1']['feasible'],
                             f1_survived=closed[(r20.name_of(arm, seed), r20.VALIDATION_SPLIT)]['families']['F1']['survived'],
                             all=closed[(r20.name_of(arm, seed), r20.VALIDATION_SPLIT)]['families']['all']['feasible'],
                             f1_train=closed[(r20.name_of(arm, seed), r20.TRAIN_SPLIT)]['families']['F1']['feasible'])
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
                          events=len(events), identical_first_hit_cells=len(same_first_hit),
                          moved_first_hit=moved))
    values = [row['delta'] for row in pairs]
    return dict(per_arm=per_arm, per_seed=pairs, deltas=values,
                improved=sum(1 for value in values if value > 0),
                unchanged=sum(1 for value in values if value == 0),
                worse=sum(1 for value in values if value < 0),
                identical_first_hit_cells=[row['identical_first_hit_cells'] for row in pairs],
                verdict='improved' if any(value > 0 for value in values)
                else 'unchanged' if all(value == 0 for value in values) else 'worse',
                note='three seed pairs, six F1 validation events each; X minus A; descriptive only')


def assemble(dest, extra=None):
    """Rebuild the round summary read-only from the frozen artifacts."""
    dest = Path(dest)
    out = dest.parent
    prereg = json.loads((out / 'preregistration.json').read_text(encoding='utf-8'))
    if prereg['criterion_sha256'] != criterion_sha():
        raise ValueError('the criterion drifted from the frozen pre-registration')
    records = json.loads((dest / 'records.json').read_text(encoding='utf-8'))
    checkpoints = json.loads((dest / 'checkpoints.json').read_text(encoding='utf-8'))
    if sorted(checkpoints) != sorted(all_names()):
        raise ValueError('checkpoint set changed')
    training = {}
    for name in all_names():
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
    for name in ['B0'] + all_names():
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
    reference = r22_reference()
    equivalence = control_equivalence(closed)
    mechanism = {split_name: mechanism_block(gaps[split_name])
                 for split_name in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT)}
    results = {}
    for name in all_names():
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
                             matches_round22_control=training[name]['matches_round22_control'],
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
        updates=r20.UPDATES, loss_arm=LOSS_ARM, aux_weight=AUX_WEIGHT,
        intervention=dict(auxiliary=prereg['auxiliary'], features=prereg['features'],
                          note='the only change against round22 is the auxiliary head on the GRU state and '
                               'its loss weight; arm A is round22 A with that weight at zero'),
        sources=prereg['sources'], worlds=prereg['worlds'],
        data=dict(events=len(records['events']), rows=records['totals']['rows'],
                  bindings=records['bindings'], storage=records['storage'],
                  aux=records['aux'], totals=records['totals'], failures=records['failures']),
        initialization=dict(per_seed=prereg['initialization']['per_seed'],
                            per_seed_aux={str(seed): {arm: training[r20.name_of(arm, seed)]['initial_aux_state_sha256']
                                                      for arm in ARMS} for seed in SEEDS},
                            identical_between_arms=True, aux_identical_between_arms=True,
                            equals_round16_round20_round21_round22=True),
        equivalence=dict(control=equivalence, all_equal=equivalence['all_equal'],
                         matches_round22_control={name: training[name]['matches_round22_control']
                                                  for name in names_of(CONTROL)},
                         control_not_reproduced=not equivalence['all_equal'],
                         step0='unit tested: identical policy logits and identical masked action at step 0; '
                               'perturbing the auxiliary parameters changes no policy logit'),
        checkpoints={name: {key: value for key, value in training[name].items()
                            if key not in ('initial_offline_events', 'final_offline_events',
                                           'aux_accuracy', 'aux_supervised_counts',
                                           'supervised_events', 'aux_supervised_events')}
                     for name in all_names()},
        validation=dict(
            events=EXPECTED['validation_events'], per_family=EXPECTED['validation_events_per_family'],
            holdout=dict(filter_valid=holdout['filter_valid'], passing_events=holdout['passing_events'],
                         executor=holdout['executor'], policies=holdout['policies'],
                         summary=holdout['summary'],
                         rule=prereg['criterion']['evaluation']['holdout_rule']),
            b0=dict(families=closed[('B0', r20.VALIDATION_SPLIT)]['families'],
                    behaviour=closed[('B0', r20.VALIDATION_SPLIT)]['behaviour']),
            results={name: results[name]['validation'] for name in all_names()},
            behaviour={name: results[name]['validation_behaviour'] for name in all_names()},
            per_event_firsthit={name: closed[(name, r20.VALIDATION_SPLIT)]['families']['all']['firsthit']
                                for name in ['B0'] + all_names()},
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
            results={name: results[name]['train_seen'] for name in all_names()},
            per_event_firsthit={name: closed[(name, r20.TRAIN_SPLIT)]['families']['all']['firsthit']
                                for name in ['B0'] + all_names()},
            direction=mechanism[r20.TRAIN_SPLIT],
            f1_cells=gaps[r20.TRAIN_SPLIT]),
        probes=probes,
        round22_reference=reference,
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
                   privileged_label=True),
        limitations=[
            'n=3 seeds and 7 validation groups; the 14 validation events collapse to 7 geometries because '
            'a mirror is not an independent sample, and F1 validation is 3 groups / 6 events',
            'the auxiliary label is constant within an event and perfectly mirror-paired (3 left / 3 '
            'right on each split), so a head could in principle read "which world is this" instead of '
            '"where is the gap"; the per-cell table, the distance-from-centre column and the confusion '
            'are reported so that this cannot be hidden',
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
    for name in ['B0'] + all_names():
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
    events = len(list((dest / 'events').glob('*.json')))
    return dict(artifacts=len(stored), root_artifacts=len(root), assets=checked,
                checkpoints=len(checkpoints), closed_loop_files=closed, event_files=events,
                baseline_rows=baselines['validation']['rows'] + baselines['train']['rows'],
                baseline_passing_events_validation=baselines['validation']['passing_events'],
                holdout_filter_valid=baselines['validation']['filter_valid'],
                f1_gap_rows=len(gaps['validation']) + len(gaps['train']),
                aux_parameters=AUX_PARAMETERS, aux_weight=AUX_WEIGHT,
                aux_holdout_accuracy={name: probes['aux']['holdout'][name]['overall']['accuracy']
                                      for name in all_names()},
                aux_seen_accuracy={name: probes['aux']['seen'][name]['overall']['accuracy']
                                   for name in all_names()},
                state_decodability={name: probes['state'][name]['side_accuracy'] for name in all_names()},
                train_events=len(prereg['worlds']['train_events']),
                validation_events=len(prereg['worlds']['validation_events']),
                criterion_sha256=prereg['criterion_sha256'], identities=len(prereg['identities']))


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
        payloads = {name: closed[(name, split_name)] for name in ['B0'] + all_names()}
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
                     geometry=prereg['geometry'][event_id],
                     auxiliary=prereg['auxiliary']['supervised_events']['train'].get(event_id, {}),
                     gap={name: gaps[split_name].get(f'{name}/{event_id}')
                          for name in ['B0'] + all_names()},
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
                                  for name in ['B0'] + all_names()})
        r20.write_json(dest / 'events' / f'{event_id}.json', value)
    if len(list((dest / 'events').glob('*.json'))) != EXPECTED['event_files']:
        raise AssertionError('28 event files required')


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
            write_records(dest, records, supervision_table(records))
            timings['data_seconds'] = round(time.perf_counter() - begin, 3)
            begin = time.perf_counter()
            reports, paths = train_all(dest, data, backbone, device, prereg)
            timings['training_seconds_loop'] = round(time.perf_counter() - begin, 3)
            if not all(torch.equal(value, frozen[key]) for key, value in backbone.state_dict().items()):
                raise AssertionError('the frozen backbone changed')
            baselines, closed, seconds = evaluate_all(dest, prereg, backbone, device, reports, paths)
            timings['evaluation_seconds'] = round(seconds, 3)
            gaps = {split_name: direction_table(closed, split_name, event_records(prereg, split_name))
                    for split_name in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT)}
            r20.write_json(dest / 'f1-gap.json', dict(
                notes='per (network, F1 event): the x series is rebuilt from advance semantics, so the odd '
                      'logical frame is included; "in_gap_at_wall" is the position on wall_frame-1, the '
                      'last frame before the band becomes lethal; "direction" is canonical (round21 '
                      'semantics) and "screen_direction" is the raw x motion, which is what the base prior '
                      'acts on',
                validation=gaps[r20.VALIDATION_SPLIT], train=gaps[r20.TRAIN_SPLIT]))
            begin = time.perf_counter()
            probes = run_probes(backbone, device, closed, prereg, records, data, paths, reports, gaps)
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
    parser.add_argument('command', choices=('preregister', 'run', 'summary', 'verify'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'preregister':
        preregister(args.out, args.device)
    elif args.command == 'run':
        run(args.out, args.device)
    elif args.command == 'summary':
        print(json.dumps(assemble(args.out / 'run'), indent=2)[:4000])
    else:
        print(json.dumps(verify_artifacts(args.out), indent=2))
