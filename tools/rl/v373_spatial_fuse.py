"""Round22: one structural intervention on top of round21 - the spatial conditioning of the
frozen backbone's feature map; research only.

Round21 proved that making the F1 wall and its gap visible from `wall_frame-45` is not
enough: the same head on the same world from the same start changed from "park inside the
gap at 248" (round20) to "hold at 216" (round21), the base group moved toward the gap in
15/18 head cells while the mirror group moved away from it in 15/18, and every death was
still on `wall_frame`. The reading is that the 256-d trunk vector the head sees does not let
the gradient say *where on the screen* the slot is: the head is a linear offset on a frozen
vector whose own prior is "screen right".

This round changes exactly one structure - nothing else. The frozen backbone's F8 map
(28x24) is pooled into a 94-d vector (7x6 average pool = 42, column profile = W = 24, row
profile = H = 28); one Linear(94->256) with **exactly zero** weight and bias maps it into the
trunk's 256-d space; that delta is added to the trunk features before the unchanged N1 head.
Arm A is the same code path with the fusion bypassed and frozen at zero (i.e. round21's A);
arm S is the fusion live from that zero initialisation. Both arms use the round-20/21
control loss (no KL anchor), 3 seeds, 200 updates, round21's telegraph pixels, the R19
split and round21's pinned hyper-parameters. The primary read-out is mechanistic (per-cell
direction toward/away the gap, parking error in px), not a total.

Two departures from the literal brief are documented in the pre-registration and the
report, because the frozen code forces them:
  1. `TacticalPolicy.forward_features` detaches its `features` input, so a trainable fusion
     layer upstream of it would receive **no gradient for all 200 updates** and arm S would
     stay bit-identical to arm A (a vacuous round). The variant `SpatialHead` is the parent's
     method verbatim with that single `features.detach()` removed; a unit test asserts the two
     forwards are bitwise identical and that the body is the parent's body with exactly that
     one line changed. The parent module is not modified.
  2. `forward_core` does not return F4/F8, so the maps are taken with
     `backbone.playfield(backbone._playfield_input(backbone._checked_regions(...)))` - the
     same private plumbing in the same order that `forward_core` uses. A unit test recomputes
     `forward_core`'s tail from those two maps and asserts bitwise equality, so "these are the
     tensors the frozen backbone used" is measured, not asserted.

No new worlds, no teacher calls, no new labels, no re-rendered training pixels (they are
round21's frozen assets, SHA-bound), no formal gate, no deployable checkpoint, no game
execution, no tuning.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tools.rl import v373_seed_confirm as r16
from tools.rl import v373_telegraph as r21
from tools.rl import v373_valid_curriculum as r20
from touhou_rl.macro_policy import _finite, _mask
from touhou_rl.tactical_data import file_sha
from touhou_rl.tactical_feature_cache import _fp32_mode
from touhou_rl.tactical_policy import (N_ACTIONS, TacticalConfig, TacticalPolicy, bos_history,
                                       tactical_policy_loss)
from touhou_rl.tactical_scenarios import WorldState
from touhou_rl.tactical_synthetic_closed_loop import HORIZONTAL, executed_history, pixel_stack
from touhou_rl.tactical_synthetic_pixels import SIZES, full_states
from touhou_rl.tactical_teacher_validation import make_world
from touhou_rl.vmodels import load_vision_checkpoint

ROOT = Path('Document/Update/Alpha/v3.7.3/evidence')
OUT = ROOT / 'spatialfuse-round22'
R21_EVIDENCE = ROOT / 'telegraph-round21'
REPORTS = Path('Document/Update/Alpha/v3.7.3/reports')
BASELINE = r20.BASELINE

PREREG_SCHEMA = 'v373-spatial-fuse-prereg/1'
DATA_SCHEMA = 'v373-spatial-fuse-data/1'
EVENT_SCHEMA = 'v373-spatial-fuse-event/1'
HEADS_SCHEMA = 'v373-spatial-fuse-heads/1'
RESULT_SCHEMA = 'v373-spatial-fuse-result/1'

# ---------------------------------------------------------------- the intervention
ARMS = ('A', 'S')
CONTROL, SPATIAL = 'A', 'S'
SEEDS = r20.SEEDS
LOSS_ARM = r20.CONTROL              # both arms are the round-20/21 control loss: no KL anchor
SPATIAL_POOL = (7, 6)               # F8 28x24 -> 7 rows x 6 columns = 42
SPATIAL_COLUMNS = 24                # F8 width  -> column profile
SPATIAL_ROWS = 28                   # F8 height -> row profile
SPATIAL_DIM = 42 + SPATIAL_COLUMNS + SPATIAL_ROWS     # 94
FEATURE_DIM = 256                   # vmodels3.TRUNK_HIDDEN, asserted against the checkpoint
F4_SHAPE = (56, 48)
F8_SHAPE = (28, 24)
PLAYFIELD_HW = (224, 192)
FUSION_PARAMETERS = SPATIAL_DIM * FEATURE_DIM + FEATURE_DIM    # 94*256 + 256 = 24320
FUSION_INIT = 'weight and bias exactly zero (nn.init.zeros_), so the delta is 0 at step 0'
RIDGE = dict(lambda_=1., standardise='fit-fold mean/std', folds='leave one F1 train group out',
             lambda_sweep=[1e-4, 1e-3, 1e-2, .1, 1., 10., 100., 1e3],
             note='lambda=1.0 is the headline value; the sweep is reported so the conclusion does not '
                  'depend on that one choice (measured: the ordering of the two representations is the '
                  'same across seven orders of magnitude)')

PARENT_DETACH_LINE = 'features, base_logits = features.detach(), base_logits.detach()'
VARIANT_DETACH_LINE = 'base_logits = base_logits.detach()'

EXPECTED = dict(events=28, train_events=14, validation_events=14,
                train_events_per_family={'F1': 6, 'F2': 2, 'F3': 6},
                validation_events_per_family={'F1': 6, 'F2': 2, 'F3': 6},
                checkpoints=6, closed_loop_files=14, networks=7, baseline_rows=2 * 14 * 6,
                f1_gap_rows=2 * 7 * 6, spatial_files=14, event_files=28)

# Round21's recorded results, re-read and re-asserted from its frozen summary before this
# round starts. They are the only comparison baseline of round22 and are never typed in
# without a machine check.
R21_F1_VALIDATION_FEASIBLE = {'A': [2, 2, 2], 'B': [2, 2, 2]}
R21_ALL_VALIDATION_FEASIBLE = {'A': [3, 3, 3], 'B': [2, 2, 2]}
R21_CONTROL_FINAL_HEAD_SHA = {
    'A-s7': 'd52551bb43e4d49d55566548fb7a579ad9fab7161e7624abcadd600a496d4e72',
    'A-s17': 'fb9233f11bb2c556e4a689e91f2144c0b98b94ca81bce3a5319de9cd5493ea64',
    'A-s27': '49ff13d9384fb2d6768fd3bc6453b40f28bccaa5fa915ee5bf2fc3f885cb77ac'}
# Six head cells per group in round21 (B0 excluded, as in the round21 report): base 15
# toward / 3 hold / 0 away, mirror 3 toward / 0 hold / 15 away. Arm A alone - the arm this
# round reproduces - is base 6/3/0 and mirror 3/0/6; B0 contributes 3/0/0 and 0/0/3.
R21_DIRECTION = dict(heads=dict(base=dict(toward=15, hold=3, away=0),
                                mirror=dict(toward=3, hold=0, away=15)),
                     control=dict(base=dict(toward=6, hold=3, away=0),
                                  mirror=dict(toward=3, hold=0, away=6)),
                     b0=dict(base=dict(toward=3, hold=0, away=0),
                             mirror=dict(toward=0, hold=0, away=3)))


# ---------------------------------------------------------------- the intervention
class SpatialFusion(nn.Module):
    """F8 -> 94-d spatial vector -> 256-d delta, added to the frozen trunk features.

    The vector has no parameters and is fixed by this round's criterion:
        pooled  = adaptive_avg_pool2d(f8, (7, 6)).mean(channel)   -> 42 (row major)
        columns = f8.mean(channel, row)                            -> 24 (one per F8 column)
        rows    = f8.mean(channel, column)                         -> 28 (one per F8 row)
    The only parameters are one Linear(94, 256) whose weight and bias start at exactly 0, so
    arm S is numerically arm A at step 0 and the first update is the first difference.
    `enabled=False` is arm A: the delta is not computed at all (bypass, not "add zero").
    """

    def __init__(self, spatial_dim=SPATIAL_DIM, feature_dim=FEATURE_DIM, enabled=True):
        super().__init__()
        self.spatial_dim = int(spatial_dim)
        self.feature_dim = int(feature_dim)
        self.enabled = bool(enabled)
        self.projection = nn.Linear(self.spatial_dim, self.feature_dim, bias=True)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        if not self.enabled:
            for parameter in self.projection.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def spatial_maps(backbone, regions):
        """F4/F8 exactly as `forward_core` computes them: the same two private calls, same order."""
        checked = backbone._checked_regions(regions)
        return backbone.playfield(backbone._playfield_input(checked['playfield']))

    @staticmethod
    def vector(f8):
        """[B,C,H,W] -> [B,94]; the definition frozen in this round's criterion."""
        if f8.ndim != 4:
            raise ValueError('F8 must be [B,C,H,W]')
        pooled = F.adaptive_avg_pool2d(f8, SPATIAL_POOL).mean(dim=1).flatten(1)
        columns = f8.mean(dim=(1, 2))
        rows = f8.mean(dim=(1, 3))
        value = torch.cat([pooled, columns, rows], dim=-1)
        if value.shape[-1] != SPATIAL_DIM:
            raise ValueError('spatial vector must be %d-d, got %d' % (SPATIAL_DIM, value.shape[-1]))
        return value

    def forward(self, f8):
        return self.projection(self.vector(f8))

    def fuse(self, features, spatial):
        """features [B,T,256] and spatial [B,T,94] -> the head's `features` argument."""
        if features.shape[:-1] != spatial.shape[:-1] or features.shape[-1] != self.feature_dim \
                or spatial.shape[-1] != self.spatial_dim:
            raise ValueError('fusion shape mismatch: %s vs %s'
                             % (tuple(features.shape), tuple(spatial.shape)))
        if not self.enabled:
            return features
        return features + self.projection(spatial)

    def describe(self):
        return dict(enabled=self.enabled, spatial_dim=self.spatial_dim, feature_dim=self.feature_dim,
                    parameters=sum(p.numel() for p in self.projection.parameters()),
                    active=sum(p.numel() for p in self.projection.parameters() if p.requires_grad),
                    init=FUSION_INIT if self.enabled else 'frozen at zero and bypassed (arm A)')


class SpatialHead(TacticalPolicy):
    """`TacticalPolicy` with exactly one line of `forward_features` changed.

    The parent starts `forward_features` with
    ``features, base_logits = features.detach(), base_logits.detach()``: the head is a
    function of pre-detached inputs by design. That is correct for a head trained on frozen
    features, but it makes an *upstream* trainable fusion layer unreachable - `detach()` cuts
    the only path from the loss to its parameters, so arm S would receive a zero gradient for
    all 200 updates and stay bit-identical to arm A. The variant therefore drops the
    `features` detach and keeps the `base_logits` one (numerically inert: the backbone is
    frozen). Every other statement is the parent's, verbatim. The unit test asserts the two
    forwards are bitwise identical and that the body is the parent's body with exactly this
    one line changed. The parent module is not modified.
    """

    def forward_features(self, features, base_logits, history=None, hidden=None,
                         reset_mask=None, valid_mask=None):
        if features.ndim != 3 or min(features.shape[:2]) < 1:
            raise ValueError("features must be nonempty [B,T,F]")
        b, t, _ = features.shape
        c = self.tactical_config
        _finite(features, "features", (b, t, c.feature_dim))
        _finite(base_logits, "base_logits", (b, t, 18))
        if features.device != base_logits.device or features.dtype != base_logits.dtype:
            raise ValueError("features/base dtype or device mismatch")
        base_logits = base_logits.detach()
        hist = bos_history(features) if history is None else history
        if set(hist) != set(bos_history(features)):
            raise ValueError("history fields mismatch")
        for name, maximum in (("previous_candidate", 18), ("previous_option", 3)):
            x = hist[name]
            if x.shape != (b, t) or x.dtype != torch.long or x.device != features.device or bool(((x < 0) | (x > maximum)).any()):
                raise ValueError(f"invalid {name}")
        for name in ("target_valid", "history_valid"):
            _mask(hist[name], name, (b, t), features.device)
        for name in ("previous_hold", "frame_delta", "previous_target"):
            _finite(hist[name], name, (b, t, 2) if name == "previous_target" else (b, t))
            if hist[name].device != features.device or hist[name].dtype != features.dtype:
                raise ValueError("history dtype or device mismatch")
        if bool(((hist["previous_hold"] < 0) | (hist["frame_delta"] < 0)).any()):
            raise ValueError("hold and frame delta must be nonnegative")
        valid = torch.ones((b,t), dtype=torch.bool, device=features.device) if valid_mask is None else _mask(valid_mask, "valid_mask", (b,t), features.device)
        reset = torch.zeros_like(valid) if reset_mask is None else _mask(reset_mask, "reset_mask", (b,t), features.device)
        h = features.new_zeros((1,b,c.hidden_size)) if hidden is None else hidden
        _finite(h, "hidden", (1,b,c.hidden_size))
        if h.device != features.device or h.dtype != features.dtype:
            raise ValueError("hidden dtype or device mismatch")
        hv = hist["history_valid"]
        pc = torch.where(hv, hist["previous_candidate"], 18)
        po = torch.where(hv, hist["previous_option"], 3)
        tv = hv & hist["target_valid"]
        inp = torch.cat((features, F.one_hot(pc,19).to(features.dtype),
                         F.one_hot(po,4).to(features.dtype),
                         torch.log1p(hist["previous_hold"] * hv)[...,None],
                         torch.log1p(hist["frame_delta"] * hv)[...,None],
                         torch.where(tv[...,None], hist["previous_target"], 0).detach(),
                         tv[...,None].to(features.dtype), hv[...,None].to(features.dtype)), -1)
        states = []
        for i in range(t):
            # A padding position is inert, including reset flags on padding.
            start = torch.where((reset[:,i] & valid[:,i])[None,:,None], 0, h)
            if c.arm == "B1":
                candidate = torch.tanh(self.heads["memoryless"](features[:,i]))[None]
            else:
                _, candidate = self.heads["gru"](inp[:,i:i+1], start)
            h = torch.where(valid[:,i][None,:,None], candidate, h)
            states.append(h[0])
        state = torch.stack(states, 1)
        option = self.heads["option"](state)
        target = self.heads["target"](state)
        conditions = [features, state]
        if c.arm != "N0":
            conditions += [self.heads["option_embedding"](option.softmax(-1)), target]
        residual = self.heads["residual"](torch.cat(conditions, -1))
        return {"features": features, "base_logits": base_logits,
                "policy_logits": base_logits + residual, "candidate_logits": base_logits + residual,
                "residual_logits": residual, "option_logits": option, "target": target,
                "route_target": target, "urgency": self.heads["urgency"](state).sigmoid().squeeze(-1),
                "termination_logits": self.heads["termination"](state),
                "slack": self.heads["slack"](state).squeeze(-1), "hidden": h,
                "valid_mask": valid}


def fresh_head(backbone, seed):
    """`r16.fresh_policy` with the variant class: same reseed, same construction order.

    `r16.fresh_policy` hard-codes `TacticalPolicy`, so the identical RNG recipe is written
    out here; the initial head SHA is then checked against the frozen R16/R20/R21 table, so
    "same recipe" is a measured claim rather than a claim about source text.
    """
    torch.manual_seed(seed)
    return SpatialHead(backbone, TacticalConfig(arm='N1'))


# ---------------------------------------------------------------- plumbing
def spatial_digest(arrays):
    """Order-explicit digest over the 94-d spatial vectors of every train event."""
    digest = hashlib.sha256()
    for event_id, array in arrays.items():
        digest.update(event_id.encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
    return digest.hexdigest()


def spatial_stats(array):
    return dict(rows=int(array.shape[0]), dim=int(array.shape[1]),
                minimum=round(float(array.min()), 6), maximum=round(float(array.max()), 6),
                mean=round(float(array.mean()), 6), std=round(float(array.std()), 6))


def spatial_table(vectors):
    return {event_id: spatial_stats(array) for event_id, array in sorted(vectors.items())}


def masked_candidate(logits):
    """The frozen 18-way mask then argmax, exactly as the closed-loop runner does it."""
    selected = logits.clone()
    selected[[c for c in range(N_ACTIONS) if c not in HORIZONTAL]] = -torch.inf
    return int(selected.argmax())


def screen_action(candidate):
    return None if candidate is None else HORIZONTAL.get(int(candidate))


def screen_direction(action):
    if action is None:
        return None
    return 'right' if action > 0 else 'left' if action < 0 else 'hold'


def all_names():
    """The six research checkpoints of this round: two arms x three seeds (arm-major)."""
    return [f'{arm}-s{seed}' for arm in ARMS for seed in SEEDS]


# ---------------------------------------------------------------- round21 reference
def r21_records():
    return json.loads((R21_EVIDENCE / 'run/records.json').read_text(encoding='utf-8'))


def r21_assets():
    return {entry['event_id']: entry for entry in r21_records()['events']}


def classify_direction(row):
    """`toward` = the canonical direction the gap requires, from round21's stored fields."""
    gap = row.get('gap')
    start = row['start_canonical']
    if not gap:
        return dict(need=None, verdict='no_gap')
    centre = (float(gap[0]) + float(gap[1])) / 2.
    need = 'right' if centre > start else 'left' if centre < start else 'hold'
    direction = row['direction']
    if direction == 'hold':
        verdict = 'hold'
    else:
        verdict = 'toward' if direction == need else 'away'
    return dict(need=need, verdict=verdict)


def direction_counts(rows, networks=None):
    """Toward/hold/away counts for the base and mirror groups of round21-shaped rows."""
    result = {}
    for group, mirror in (('base', False), ('mirror', True)):
        subset = [row for row in rows.values() if bool(row['mirror']) is mirror
                  and (networks is None or row['network'] in networks)]
        verdicts = [classify_direction(row)['verdict'] for row in subset]
        result[group] = {key: verdicts.count(key) for key in ('toward', 'hold', 'away')}
    return result


def r21_reference():
    """Re-read round21's frozen results and re-assert every number this round quotes."""
    summary_path = R21_EVIDENCE / 'run/summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    recorded = {arm: [summary['validation']['results'][r20.name_of(arm, seed)]['F1']['feasible']
                      for seed in SEEDS] for arm in r20.ARMS}
    if recorded != R21_F1_VALIDATION_FEASIBLE:
        raise ValueError('round21 F1 validation reference drift')
    total = {arm: [summary['validation']['results'][r20.name_of(arm, seed)]['all']['feasible']
                   for seed in SEEDS] for arm in r20.ARMS}
    if total != R21_ALL_VALIDATION_FEASIBLE:
        raise ValueError('round21 total validation reference drift')
    checkpoints = json.loads((R21_EVIDENCE / 'run/checkpoints.json').read_text(encoding='utf-8'))
    final = {name: checkpoints[name]['final_head_state_sha256'] for name in R21_CONTROL_FINAL_HEAD_SHA}
    if final != R21_CONTROL_FINAL_HEAD_SHA:
        raise ValueError('round21 control final head reference drift')
    gaps = json.loads((R21_EVIDENCE / 'run/f1-gap.json').read_text(encoding='utf-8'))
    round21_names = [r20.name_of(arm, seed) for arm in r20.ARMS for seed in r20.SEEDS]
    control = direction_counts(gaps['validation'], networks=[n for n in round21_names
                                                             if n.startswith(r20.CONTROL + '-')])
    if control != R21_DIRECTION['control']:
        raise ValueError('round21 control direction reference drift: ' + json.dumps(control))
    heads = direction_counts(gaps['validation'], networks=round21_names)
    if heads != R21_DIRECTION['heads']:
        raise ValueError('round21 six-head direction reference drift: ' + json.dumps(heads))
    b0 = direction_counts(gaps['validation'], networks=['B0'])
    if b0 != R21_DIRECTION['b0']:
        raise ValueError('round21 B0 direction reference drift: ' + json.dumps(b0))
    return dict(summary_sha256=file_sha(summary_path),
                checkpoints_sha256=file_sha(R21_EVIDENCE / 'run/checkpoints.json'),
                records_sha256=file_sha(R21_EVIDENCE / 'run/records.json'),
                f1_gap_sha256=file_sha(R21_EVIDENCE / 'run/f1-gap.json'),
                baselines_sha256=file_sha(R21_EVIDENCE / 'run/baselines.json'),
                f1_validation_feasible=recorded, all_validation_feasible=total,
                control_final_head_state_sha256=final,
                direction=dict(control=R21_DIRECTION['control'], heads=R21_DIRECTION['heads'],
                               b0=R21_DIRECTION['b0']),
                f1_deaths='round21: every F1 death (validation 24 + train 27 + B0 12) is on wall_frame')


# ---------------------------------------------------------------- direction mechanics
def signed_error(gap, at_wall):
    """0 inside [gap0, gap1]; negative short of the left edge; positive past the right edge."""
    if gap is None or at_wall is None:
        return None
    if at_wall < gap[0]:
        return float(at_wall) - float(gap[0])
    if at_wall > gap[1]:
        return float(at_wall) - float(gap[1])
    return 0.


def cell_diagnostic(world, row, network, split_name):
    """One mechanism row for one (network, F1 event): round21's diagnostic plus the verdicts."""
    actions = [step['actual_action'] for step in row['steps'] if step.get('actual_action') is not None]
    value = r21.gap_diagnostics(world, actions, row['first_collision_frame'])
    trace = r21.gap_trace(world, actions)
    screen = trace[-1]['x'] - trace[0]['x']
    value.update(network=network, split=split_name, feasible=bool(row['feasible']),
                 survived=bool(row['survived']), decisions=row['decisions'],
                 screen_direction='right' if screen > 1e-9 else 'left' if screen < -1e-9 else 'hold',
                 screen_net_px=round(float(screen), 6),
                 signed_error_px=signed_error(value['gap'], value['canonical_at_wall']),
                 arm='B0' if network == 'B0' else network.split('-')[0],
                 mirror_group='mirror' if bool(world.mirror) else 'base')
    value.update(classify_direction(value))
    return value


def direction_table(closed, split_name, events):
    rows = {}
    for name in ['B0'] + all_names():
        by_event = {row['event_id']: row for row in closed[(name, split_name)]['events']}
        for entry in events:
            if entry['family'] != 'F1':
                continue
            world = make_world(entry['scenario'])
            rows[f'{name}/{entry["event_id"]}'] = cell_diagnostic(
                world, by_event[entry['event_id']], name, split_name)
    if len(rows) != EXPECTED['f1_gap_rows'] // 2:
        raise AssertionError('one mechanism row per network and F1 event required')
    return rows


def group_summary(rows, networks):
    """Per-group counts, the flip rate and the parking error of one arm."""
    out = {}
    for group, mirror in (('base', False), ('mirror', True)):
        subset = [row for row in rows.values()
                  if row['network'] in networks and row['mirror'] is mirror]
        if not subset:
            continue
        verdicts = [row['verdict'] for row in subset]
        distances = [abs(row['signed_error_px']) for row in subset
                     if row['signed_error_px'] is not None]
        out[group] = dict(
            cells=len(subset),
            toward=verdicts.count('toward'), hold=verdicts.count('hold'),
            away=verdicts.count('away'),
            toward_rate=round(verdicts.count('toward') / len(subset), 6),
            in_gap=sum(1 for row in subset if row['in_gap_at_wall']),
            parked_px=dict(values=[row['signed_error_px'] for row in subset],
                           mean_abs=round(sum(distances) / len(distances), 6) if distances else None,
                           maximum_abs=max(distances) if distances else None),
            never_entered_gap=sum(1 for row in subset if not row['entered_gap']),
            mechanisms={key: sum(1 for row in subset if row['mechanism'] == key)
                        for key in sorted({row['mechanism'] for row in subset})},
            screen_right=sum(1 for row in subset if row['screen_direction'] == 'right'),
            screen_left=sum(1 for row in subset if row['screen_direction'] == 'left'),
            screen_hold=sum(1 for row in subset if row['screen_direction'] == 'hold'))
    return out


def mechanism_block(rows, networks_by_arm):
    block = {}
    for arm, networks in networks_by_arm.items():
        groups = group_summary(rows, networks)
        base, mirror = groups['base'], groups['mirror']
        block[arm] = dict(
            groups=groups,
            flip_rate=dict(
                mirror_toward_rate=mirror['toward_rate'], base_toward_rate=base['toward_rate'],
                mirror_in_gap=mirror['in_gap'], base_in_gap=base['in_gap'],
                note='mirror_toward_rate is the fraction of mirror-group cells whose canonical motion points '
                     'at the gap: that is the direction flip this round is about, because the same screen-right '
                     'prior is wrong there and right in the base group'),
            per_seed={str(seed): group_summary(rows, [n for n in networks if n.endswith(f'-s{seed}')])
                      for seed in SEEDS})
    return block


# ---------------------------------------------------------------- data (round21 pixels)
def event_records(prereg, split_name):
    """The frozen scenario payloads of one split, rebuilt from the SHA-bound R18 manifest."""
    return r20.event_records(prereg, split_name)


def train_records(prereg):
    """The 14 train events, each binding round21's frozen asset bytes."""
    table = r21_assets()
    rows = []
    for row in event_records(prereg, r20.TRAIN_SPLIT):
        stored = table[row['event_id']]
        value = dict(row, teacher_verified=bool(stored['teacher_verified']))
        for key in ('pixels', 'history', 'labels', 'rollout'):
            value[key] = dict(path=stored[key]['path'], sha256=stored[key]['sha256'])
        rows.append(value)
    return rows


def asset_bindings(records):
    return {entry['event_id']: {key: entry[key]['sha256']
                                for key in ('pixels', 'history', 'labels', 'rollout')}
            for entry in records}


def spatial_vectors(backbone, records, root, device):
    """Frozen features/base (round20's encode) plus the 94-d spatial vector of the same row.

    The forward pass runs under `torch.inference_mode()` exactly like `r20.encode`, so the
    features are the same numbers round20/21 trained on. Tensors created inside that mode are
    inference tensors and cannot be inputs of an autograd-tracked layer, so they are cloned
    *outside* the block: a clone of an inference tensor is an ordinary tensor with identical
    values (the documented work-around), which is what lets the fusion layer's Linear receive
    `spatial` during training.
    """
    root = Path(root)
    raw = []
    with torch.inference_mode():
        for entry in records:
            pixels = r20.pixel_tensors(root / entry['pixels']['path'])
            for key in ('pixels', 'history', 'labels'):
                if file_sha(root / entry[key]['path']) != entry[key]['sha256']:
                    raise ValueError(f'{key} asset SHA drift: ' + entry['event_id'])
            features, base, spatial = [], [], []
            for start in range(0, r20.DECISIONS, r20.CHUNK):
                chunk = {key: value[start:start + r20.CHUNK].to(device) for key, value in pixels.items()}
                core = backbone.forward_core(chunk)
                f4, f8 = SpatialFusion.spatial_maps(backbone, chunk)
                if tuple(f4.shape[-2:]) != F4_SHAPE or tuple(f8.shape[-2:]) != F8_SHAPE:
                    raise ValueError('frozen spatial map shape: %s %s'
                                     % (tuple(f4.shape), tuple(f8.shape)))
                features.append(core['features'].clone())
                base.append(core['policy_logits'].clone())
                spatial.append(SpatialFusion.vector(f8).clone())
            history = torch.load(root / entry['history']['path'], map_location='cpu', weights_only=True)
            labels = torch.load(root / entry['labels']['path'], map_location='cpu', weights_only=True)
            raw.append((entry, torch.cat(features), torch.cat(base), torch.cat(spatial),
                        history, labels))
    data, vectors, feature_table = [], {}, {}
    with torch.inference_mode(False):
        for entry, features, base, spatial, history, labels in raw:
            features, base, spatial = features.clone(), base.clone(), spatial.clone()
            row = dict(entry=entry, features=features, base=base, spatial=spatial,
                       history={k: v.to(device) for k, v in history.items()},
                       labels={k: v.to(device) for k, v in labels.items()})
            if row['spatial'].shape != (r20.DECISIONS, SPATIAL_DIM):
                raise ValueError('spatial row shape: ' + entry['event_id'])
            if row['spatial'].is_inference() or row['features'].is_inference():
                raise ValueError('the fusion input must be an ordinary tensor: ' + entry['event_id'])
            data.append(row)
            vectors[entry['event_id']] = row['spatial'].detach().cpu().numpy().astype(np.float32)
            feature_table[entry['event_id']] = row['features'].detach().cpu().numpy().astype(np.float32)
    return data, vectors, feature_table


def write_spatial(dest, vectors):
    r20.new_dir(dest / 'spatial')
    refs = {}
    for event_id, array in sorted(vectors.items()):
        path = dest / 'spatial' / f'{event_id}.npz'
        r20.write_npz(path, {'spatial': array})
        refs[event_id] = dict(path=f'spatial/{event_id}.npz', sha256=file_sha(path),
                              bytes=path.stat().st_size, shape=list(array.shape))
    if len(refs) != EXPECTED['spatial_files']:
        raise AssertionError('14 spatial vector files required')
    return refs


def write_records(dest, records, spatial_refs):
    entries, failures = [], []
    for entry in records:
        entries.append(dict(event_id=entry['event_id'], family=entry['family'], group=entry['group'],
                            split=entry['split'], mirror=entry['mirror'], rows=r20.DECISIONS,
                            horizon=r20.HORIZON, teacher_verified=entry['teacher_verified'],
                            scenario_sha256=entry['scenario_sha256'],
                            pixels=entry['pixels'], history=entry['history'], labels=entry['labels'],
                            rollout=entry['rollout'],
                            bytes=sum(r21_assets()[entry['event_id']][key]['bytes']
                                      for key in ('pixels', 'history', 'labels', 'rollout'))))
        if not entry['teacher_verified']:
            failures.append(dict(event_id=entry['event_id'], verdict='not_teacher_verified'))
    value = dict(schema=DATA_SCHEMA, root=str(R21_EVIDENCE / 'run'), events=entries, failures=failures,
                 bindings=asset_bindings(records),
                 source='round21 telegraph assets, reopened read-only and SHA-bound; nothing re-rendered, '
                        'no teacher called, no label created',
                 depth=3, region_sizes=SIZES, depth_frames=r20.depth_frames(), storage=r20.STORAGE,
                 spatial=spatial_refs,
                 spatial_definition='F8 [B,96,28,24] -> 42 (7x6 pool over channels) + 24 (column profile) '
                                    '+ 28 (row profile) = 94 float32 per decision row, produced by the same '
                                    'frozen backbone pass that produced features/base',
                 history_semantics='identical to round21 (round20 packing), unchanged',
                 label_semantics='identical to round21 (round20 labels), unchanged',
                 totals=dict(events=len(entries), teacher_verified=sum(1 for e in entries if e['teacher_verified']),
                             rows=len(entries) * r20.DECISIONS,
                             bytes=sum(e['bytes'] for e in entries),
                             spatial_bytes=sum(ref['bytes'] for ref in spatial_refs.values())))
    r20.write_json(dest / 'records.json', value)
    return value


# ---------------------------------------------------------------- training
def fused_infer(model, fusion, entry):
    features = entry['features'][None]
    spatial = entry['spatial'][None]
    return model.forward_features(fusion.fuse(features, spatial), entry['base'][None],
                                  {key: value[None] for key, value in entry['history'].items()})


def fused_offline(model, fusion, data):
    """Label fit on the rendered (seen) events; the validation split carries no labels."""
    model.eval()
    rows = []
    with torch.inference_mode():
        for entry in data:
            output = fused_infer(model, fusion, entry)
            labels = entry['labels']
            valid = labels['action_valid']
            positive = labels['action_positive'][valid]
            if int(valid.sum()) and not bool((positive.sum(-1) == 1).all()):
                raise ValueError('single route labels required')
            logits = output['policy_logits'][0][valid]
            masked = logits.clone()
            masked[:, [c for c in range(N_ACTIONS) if c not in HORIZONTAL]] = -torch.inf
            rows.append(dict(event_id=entry['entry']['event_id'], family=entry['entry']['family'],
                             rows=int(valid.sum()),
                             loss=float(tactical_policy_loss(output, {k: v[None] for k, v in labels.items()})['loss']),
                             truth=positive.long().argmax(-1).tolist() if int(valid.sum()) else [],
                             unmasked=logits.argmax(-1).tolist(), masked=masked.argmax(-1).tolist()))
    summary = {}
    for family in r20.FAMILIES:
        group = [row for row in rows if family == 'all' or row['family'] == family]
        truth = sum((row['truth'] for row in group), [])
        summary[family] = dict(events=len(group), rows=sum(row['rows'] for row in group),
                               loss=sum(row['loss'] for row in group) / len(group) if group else None,
                               unmasked_accuracy=r20.accuracy(truth, sum((row['unmasked'] for row in group), [])),
                               masked_accuracy=r20.accuracy(truth, sum((row['masked'] for row in group), [])))
    return dict(summary=summary, events=rows)


def train_all(dest, data, backbone, device, prereg):
    """Two arms x three seeds through one loop; the only arm difference is the fusion layer.

    The loop is round20's `train_all` with two additions: the fused feature and the saved
    fusion state. Arm A must therefore reproduce round21's control head bitwise; that
    equality is recorded per seed (and reported, never silently assumed).
    """
    supervised = [entry for entry in data if entry['entry']['teacher_verified']]
    excluded = [entry['entry']['event_id'] for entry in data if not entry['entry']['teacher_verified']]
    if len(supervised) + len(excluded) != len(data) or not supervised:
        raise ValueError('supervision partition')
    r20.new_dir(dest / 'train')
    reports, paths = {}, {}
    for seed in SEEDS:
        per_arm_initial = {}
        for arm in ARMS:
            name = r20.name_of(arm, seed)
            directory = r20.new_dir(dest / 'train' / name)
            model = fresh_head(backbone, seed).to(device)
            fusion = SpatialFusion(enabled=(arm == SPATIAL)).to(device)
            if model.tactical_config.feature_dim != FEATURE_DIM:
                raise AssertionError('the backbone trunk width is not the frozen 256')
            initial_sha = r16.state_sha(model.heads.state_dict())
            if initial_sha != prereg['initialization']['per_seed'][str(seed)]:
                raise AssertionError('initial head state must reproduce the frozen table: ' + name)
            fusion_sha = r16.state_sha(fusion.state_dict())
            if fusion_sha != prereg['initialization']['fusion_per_arm'][arm]:
                raise AssertionError('the fusion layer must start at the frozen init: ' + name)
            per_arm_initial[arm] = initial_sha
            r20.write_json(directory / 'initialization.json',
                           dict(head_state_sha256=initial_sha, fusion_state_sha256=fusion_sha,
                                seed=seed, arm=arm, anchor_weight=0., loss_arm=LOSS_ARM,
                                fusion=fusion.describe(),
                                shared_config='both arms are N1 with the round20 control loss; only the fusion '
                                              'layer differs (A bypassed and frozen at zero, S live from zero init)',
                                feature_storage_shared=True, supervised_events=len(supervised),
                                excluded_events=excluded))
            parameters = [value for value in model.parameters() if value.requires_grad]
            parameters += [value for value in fusion.parameters() if value.requires_grad]
            initial_offline = fused_offline(model, fusion, supervised)
            optimizer = torch.optim.Adam(parameters, lr=r20.LR)
            losses, originals, delta_norms, fusion_grads = [], [], [], []
            begin = time.perf_counter()
            for step in range(r20.UPDATES):
                model.train()
                fusion.train()
                entry = supervised[step % len(supervised)]
                output = fused_infer(model, fusion, entry)
                labels = {key: value[None] for key, value in entry['labels'].items()}
                head = r20.arm_loss(LOSS_ARM, output, labels)
                if not head['parts']['has_supervision']:
                    raise AssertionError('all 200 updates must be supervised')
                optimizer.zero_grad(set_to_none=True)
                head['loss'].backward()
                torch.nn.utils.clip_grad_norm_(parameters, r20.CLIP)
                if fusion.enabled:
                    grad = fusion.projection.weight.grad
                    fusion_grads.append(None if grad is None else float(grad.norm()))
                    with torch.no_grad():
                        delta_norms.append(float(fusion.projection(entry['spatial'][None]).norm()))
                optimizer.step()
                losses.append(float(head['loss'].detach()))
                originals.append(float(head['original'].detach()))
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            final_offline = fused_offline(model, fusion, supervised)
            final_sha = r16.state_sha(model.heads.state_dict())
            fusion_final = r16.state_sha(fusion.state_dict())
            if not fusion.enabled and any(float(value.abs().max()) != 0. for value in fusion.parameters()):
                raise AssertionError('arm A fusion must stay exactly zero: ' + name)
            path = directory / 'research-heads.pt'
            r20.write_torch(path, dict(schema=HEADS_SCHEMA, arm=arm, seed=seed,
                                       config=asdict(model.tactical_config),
                                       baseline_sha256=file_sha(BASELINE),
                                       final_head_state_sha256=final_sha,
                                       final_fusion_state_sha256=fusion_final,
                                       anchor_weight=0., loss_arm=LOSS_ARM,
                                       fusion=fusion.describe(),
                                       heads={key: value.detach().cpu()
                                              for key, value in model.heads.state_dict().items()},
                                       fusion_state={key: value.detach().cpu()
                                                     for key, value in fusion.state_dict().items()}))
            report = dict(arm=arm, seed=seed, name=name, initial_head_state_sha256=initial_sha,
                          final_head_state_sha256=final_sha,
                          initial_fusion_state_sha256=fusion_sha, final_fusion_state_sha256=fusion_final,
                          checkpoint_sha256=file_sha(path),
                          checkpoint_path=str(path.relative_to(dest)), updates=r20.UPDATES,
                          anchor_weight=0., loss_arm=LOSS_ARM, fusion=fusion.describe(),
                          matches_round21_control=(None if arm != CONTROL else
                                                   final_sha == R21_CONTROL_FINAL_HEAD_SHA[name]),
                          losses=losses, original_losses=originals,
                          anchor_loss_at_step0=0., anchor_loss_final=0.,
                          original_loss_at_step0=originals[0], original_loss_final=originals[-1],
                          loss_at_step0=losses[0], loss_final=losses[-1],
                          fusion_grad_norm=dict(first=fusion_grads[0] if fusion_grads else None,
                                                last=fusion_grads[-1] if fusion_grads else None,
                                                minimum=min(fusion_grads) if fusion_grads else None,
                                                maximum=max(fusion_grads) if fusion_grads else None,
                                                nonzero_steps=sum(1 for value in fusion_grads if value),
                                                note='L2 norm of d(loss)/d(fusion weight) after the backward '
                                                     'pass, before the clip and the step; arm A has no fusion '
                                                     'gradient at all'),
                          delta_norm=dict(first=delta_norms[0] if delta_norms else None,
                                          last=delta_norms[-1] if delta_norms else None,
                                          maximum=max(delta_norms) if delta_norms else None,
                                          note='L2 norm of the 256-d fusion delta on the training row of that '
                                               'step; exactly 0 at step 0 by construction'),
                          optimization_seconds=seconds,
                          parameters=dict(active_head=sum(value.numel() for value in model.parameters()
                                                          if value.requires_grad),
                                          active_fusion=sum(value.numel() for value in fusion.parameters()
                                                            if value.requires_grad),
                                          active=sum(value.numel() for value in parameters),
                                          heads_total=sum(value.numel() for value in model.heads.parameters()),
                                          fusion_total=sum(value.numel() for value in fusion.parameters()),
                                          total=sum(value.numel() for value in model.parameters())
                                                + sum(value.numel() for value in fusion.parameters()),
                                          round21_control_active=429200,
                                          increment_vs_control=sum(value.numel()
                                                                   for value in fusion.parameters()
                                                                   if value.requires_grad)),
                          supervised_events=[entry['entry']['event_id'] for entry in supervised],
                          excluded_events=excluded,
                          train_only='the validation events are never encoded, labelled or indexed',
                          initial_offline=initial_offline['summary'], final_offline=final_offline['summary'],
                          initial_offline_events=initial_offline['events'],
                          final_offline_events=final_offline['events'])
            r20.write_json(directory / 'training.json', report)
            reports[name] = report
            paths[name] = path
            print('trained', name, 'fusion', fusion.enabled, 'init', initial_sha[:12],
                  'final', final_sha[:12], 'loss0', round(losses[0], 4),
                  'lossN', round(losses[-1], 4),
                  'deltaN', None if not delta_norms else round(delta_norms[-1], 4),
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
                           if key not in ('losses', 'original_losses',
                                          'initial_offline_events', 'final_offline_events')}
                    for name in reports})
    return reports, paths


# ---------------------------------------------------------------- evaluation
class SpatialNetwork:
    """One frozen research checkpoint: the variant head plus its fusion layer."""

    def __init__(self, backbone, path, device, expected_head_sha=None, expected_fusion_sha=None):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('schema') != HEADS_SCHEMA:
            raise ValueError('research schema')
        if payload.get('arm') not in ARMS or payload.get('seed') not in SEEDS:
            raise ValueError('research arm/seed')
        if payload['baseline_sha256'] != file_sha(BASELINE):
            raise ValueError('baseline SHA')
        if payload.get('anchor_weight') != 0. or payload.get('loss_arm') != LOSS_ARM:
            raise ValueError('arm weight record')
        self.arm = 'N1'
        self.device = device
        self.name = r20.name_of(payload['arm'], payload['seed'])
        self.model = SpatialHead(backbone, TacticalConfig(**payload['config'])).to(device).eval()
        if self.model.tactical_config.arm != 'N1':
            raise ValueError('N1 only')
        self.model.heads.load_state_dict(payload['heads'], strict=True)
        self.fusion = SpatialFusion(enabled=(payload['arm'] == SPATIAL)).to(device).eval()
        self.fusion.load_state_dict(payload['fusion_state'], strict=True)
        self.heads_sha256 = r16.state_sha(self.model.heads.state_dict())
        self.fusion_sha256 = r16.state_sha(self.fusion.state_dict())
        if self.heads_sha256 != payload['final_head_state_sha256']:
            raise ValueError('checkpoint head SHA drift')
        if self.fusion_sha256 != payload['final_fusion_state_sha256']:
            raise ValueError('checkpoint fusion SHA drift')
        if expected_head_sha is not None and self.heads_sha256 != expected_head_sha:
            raise ValueError('unexpected checkpoint')
        if expected_fusion_sha is not None and self.fusion_sha256 != expected_fusion_sha:
            raise ValueError('unexpected fusion state')

    @torch.inference_mode()
    def __call__(self, arm, regions, previous, hidden, reset):
        if arm != self.arm:
            raise ValueError('callback arm mismatch')
        regions = {key: value.to(self.device) for key, value in regions.items()}
        core = self.model.vision_model.forward_core(regions)
        _, f8 = SpatialFusion.spatial_maps(self.model.vision_model, regions)
        spatial = SpatialFusion.vector(f8)[:, None]
        features = self.fusion.fuse(core['features'][:, None], spatial)
        output = self.model.forward_features(features, core['policy_logits'][:, None],
                                             executed_history(features, previous), hidden,
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
    reference = json.loads((R21_EVIDENCE / 'run/baselines.json').read_text(encoding='utf-8'))
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
                       renderer='v373_telegraph.render_state_telegraph', fusion=None, events=rows,
                       families=r16.families(rows), behaviour=r20.behaviour(rows))
        r20.write_json(dest / 'closed-loop' / f'B0-{split_name}.json', payload)
        closed[('B0', split_name)] = payload
    for name in all_names():
        network = SpatialNetwork(backbone, paths[name], device,
                                 reports[name]['final_head_state_sha256'],
                                 reports[name]['final_fusion_state_sha256'])
        for split_name, events in ((r20.VALIDATION_SPLIT, validation), (r20.TRAIN_SPLIT, train)):
            rows = r21.closed_loop(events, network, 'N1', name)
            payload = dict(network=name, split=split_name, condition=r20.CONDITION,
                           renderer='v373_telegraph.render_state_telegraph',
                           fusion=network.fusion.describe(), events=rows,
                           families=r16.families(rows), behaviour=r20.behaviour(rows),
                           heads_sha256=network.heads_sha256, fusion_sha256=network.fusion_sha256)
            r20.write_json(dest / 'closed-loop' / f'{name}-{split_name}.json', payload)
            closed[(name, split_name)] = payload
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    if len(closed) != EXPECTED['closed_loop_files']:
        raise AssertionError('closed-loop file count')
    return baselines, closed, seconds


def control_equivalence(closed):
    """Is arm A bit-for-bit round21's control arm on the same split and renderer?"""
    result = {}
    for seed in SEEDS:
        name = r20.name_of(CONTROL, seed)
        for split_name in (r20.VALIDATION_SPLIT, r20.TRAIN_SPLIT):
            path = R21_EVIDENCE / 'run' / 'closed-loop' / f'{name}-{split_name}.json'
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


# ---------------------------------------------------------------- probes
def stored_actions(closed, split_name, events):
    """The masked argmax actually taken, straight out of the stored closed-loop rows."""
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
                directions = [screen_direction(screen_action(step.get('candidate')))
                              for step in subset]
                table[name].setdefault(entry['event_id'], {})[label] = dict(
                    decisions=len(directions),
                    right=sum(1 for d in directions if d == 'right'),
                    left=sum(1 for d in directions if d == 'left'),
                    hold=sum(1 for d in directions if d == 'hold'))
    return table


def replay_images(world, steps):
    """The exact image sequence the closed-loop runner rendered, rebuilt from the stored trace.

    The runner renders once before its loop (the initial state) and once after every `advance`,
    so image *i* is the state at the start of decision *i*, and `x_before` is exactly that
    state's x. Decision 0's state is the initial state, so the sequence is one image per step
    and nothing else; an extra leading image would shift every stack by one decision.
    """
    if not steps or int(steps[0]['frame']) != 0:
        raise ValueError('the stored trace must start at decision 0')
    return [r21.render_state_telegraph(world, WorldState(int(step['frame']), float(step['x_before'])))
            for step in steps]


def preview_steps(world, steps):
    window = r21.telegraph_window(world)
    if window is None:
        return []
    return [index for index, step in enumerate(steps) if window[0] <= int(step['frame']) < window[1]]


def flip_probe(network, device, world, steps, name, head_only=False):
    """Replay the stored trajectory; at every preview step also ask with the playfield mirrored.

    The rebuilt stacks are the runner's stacks (the replay's masked argmax is compared with
    the stored candidate step by step), so "same state, mirrored playfield" is measured on
    identical inputs apart from the mirror. A head that reads the gap's screen position must
    reverse its horizontal preference; a head dominated by the frozen prior cannot.
    """
    if world.family != 'F1':
        raise ValueError('the input-mirror probe is F1 only: F1 fires no bullets, so (frame, x) '
                         'determines the rendered image exactly and the rebuild is provably the '
                         'runner input; for F2/F3 the bullet list would also have to be replayed')
    images = replay_images(world, steps)
    wanted = set(preview_steps(world, steps))
    hidden, previous = None, None
    rows = []
    matches = 0
    with torch.inference_mode():
        for index, step in enumerate(steps):
            regions, _ = pixel_stack(images[:index + 1])
            if head_only:
                core = network.forward_core({key: value.to(device) for key, value in regions.items()})
                logits, next_hidden = core['policy_logits'][0], None
            else:
                logits, next_hidden = network('N1', regions, previous, hidden, index == 0)
            candidate = masked_candidate(logits)
            if index in wanted:
                flipped = {key: (value.flip(-2) if key == 'playfield' else value)
                           for key, value in regions.items()}
                if head_only:
                    core = network.forward_core({key: value.to(device) for key, value in flipped.items()})
                    flipped_logits = core['policy_logits'][0]
                else:
                    flipped_logits, _ = network('N1', flipped, previous, hidden, index == 0)
                flipped_candidate = masked_candidate(flipped_logits)
                rows.append(dict(step=index, frame=int(step['frame']), candidate=candidate,
                                 candidate_flipped=flipped_candidate,
                                 action=screen_action(candidate),
                                 action_flipped=screen_action(flipped_candidate)))
            matches += int(candidate == int(step['candidate']))
            previous, hidden = candidate, next_hidden
    return dict(network=name, event_id=world.event_id, steps=len(steps),
                stored_candidate_matches=matches, rows=rows)


def flip_summary(probes):
    """Per network: how often mirroring the playfield reverses the horizontal preference.

    Two rates are reported because the distilled heads hold on almost every step: the
    unconditional flip rate is dominated by "hold stays hold", so the informative number is
    the error rate over the steps where the head actually moves.
    """
    result = {}
    for row in probes:
        entry = result.setdefault(row['network'],
                                  dict(rows=0, flips=0, same=0, holds_true=0, holds_flipped=0,
                                       moving_rows=0, moving_flips=0, events={},
                                       stored_candidate_matches=0, replay_steps=0))
        entry['stored_candidate_matches'] += row['stored_candidate_matches']
        entry['replay_steps'] += row['steps']
        per = entry['events'].setdefault(row['event_id'], dict(rows=0, flips=0, same=0,
                                                              moving_rows=0, moving_flips=0))
        for item in row['rows']:
            a, b = item['action'], item['action_flipped']
            entry['rows'] += 1
            per['rows'] += 1
            if a is None or b is None:
                continue
            if a == 0:
                entry['holds_true'] += 1
            if b == 0:
                entry['holds_flipped'] += 1
            if a * b < 0:
                entry['flips'] += 1
                per['flips'] += 1
            elif a == b:
                entry['same'] += 1
                per['same'] += 1
            if a != 0:
                entry['moving_rows'] += 1
                per['moving_rows'] += 1
                if a * b < 0:
                    entry['moving_flips'] += 1
                    per['moving_flips'] += 1
    for entry in result.values():
        entry['flip_rate'] = round(entry['flips'] / entry['rows'], 6) if entry['rows'] else None
        entry['same_rate'] = round(entry['same'] / entry['rows'], 6) if entry['rows'] else None
        entry['flip_rate_when_moving'] = (round(entry['moving_flips'] / entry['moving_rows'], 6)
                                          if entry['moving_rows'] else None)
        entry['replay_fidelity'] = (round(entry['stored_candidate_matches'] / entry['replay_steps'], 6)
                                    if entry['replay_steps'] else None)
        for per in entry['events'].values():
            per['flip_rate'] = round(per['flips'] / per['rows'], 6) if per['rows'] else None
    return result


def ridge_fit(x, y, lambda_=RIDGE['lambda_']):
    """Ridge regression with an intercept on standardised inputs; deterministic."""
    mean, std = x.mean(0), x.std(0)
    std = np.where(std < 1e-8, 1., std)
    z = (x - mean) / std
    design = np.concatenate([z, np.ones((z.shape[0], 1))], 1)
    penalty = np.eye(design.shape[1]) * lambda_
    penalty[-1, -1] = 0.
    weight = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return dict(mean=mean, std=std, weight=weight)


def ridge_predict(model, x):
    z = (x - model['mean']) / model['std']
    design = np.concatenate([z, np.ones((z.shape[0], 1))], 1)
    return design @ model['weight']


def spatial_probe(vector_table, feature_table, worlds, positions, rows_by_event):
    """Is the gap's screen position linearly decodable from the 94-d vector (and from the
    256-d trunk features)? Fit on two F1 train groups, test on the third; train pixels only.

    Targets, all in native screen px:
      side   - binary: is the gap's screen centre right of the playfield centre (192)?
      centre - the gap's screen centre itself;
      offset - the gap's screen centre minus the player's x on that row, i.e. the signed
               distance the policy would have to travel to park in the gap.
    """
    groups = {}
    for event_id, world in worlds.items():
        groups.setdefault(world.template_group, []).append(event_id)
    rows = []
    for event_id, world in worlds.items():
        gap_screen = r21.telegraph_gap(world)
        centre = (gap_screen[0] + gap_screen[1]) / 2.
        for index in rows_by_event[event_id]:
            rows.append(dict(event_id=event_id, group=world.template_group, row=index,
                             side=float(centre > 192.), centre=centre,
                             offset=centre - float(positions[event_id][index])))
    result = {}
    for name, table in (('spatial_94', vector_table), ('features_256', feature_table)):
        folds = []
        for held in sorted(groups):
            train_ids = [e for g in groups if g != held for e in groups[g]]
            test_ids = set(groups[held])
            train_rows = [row for row in rows if row['event_id'] in train_ids]
            test_rows = [row for row in rows if row['event_id'] in test_ids]
            if not train_rows or not test_rows:
                continue
            x_train = np.stack([table[row['event_id']][row['row']] for row in train_rows]).astype(np.float64)
            x_test = np.stack([table[row['event_id']][row['row']] for row in test_rows]).astype(np.float64)
            fold = dict(held_out=held, train_events=len(train_ids), test_events=len(test_ids),
                        rows_train=len(train_rows), rows_test=len(test_rows))
            for target in ('centre', 'offset'):
                y_train = np.array([row[target] for row in train_rows], dtype=np.float64)
                y_test = np.array([row[target] for row in test_rows], dtype=np.float64)
                model = ridge_fit(x_train, y_train)
                pred = ridge_predict(model, x_test)
                baseline = np.full_like(y_test, y_train.mean())
                spread = float(((y_test - y_test.mean()) ** 2).sum())
                fold[target] = dict(
                    mae_px=round(float(np.abs(pred - y_test).mean()), 6),
                    baseline_mae_px=round(float(np.abs(baseline - y_test).mean()), 6),
                    r2=round(float(1. - ((pred - y_test) ** 2).sum() / spread), 6)
                    if spread > 1e-9 else None)
            y_train = np.array([row['side'] for row in train_rows], dtype=np.float64)
            y_test = np.array([row['side'] for row in test_rows], dtype=np.float64)
            model = ridge_fit(x_train, y_train)
            pred = (ridge_predict(model, x_test) > .5).astype(np.float64)
            fold['side'] = dict(accuracy=round(float((pred == y_test).mean()), 6),
                                baseline_accuracy=round(float((y_test == (y_train.mean() > .5)).mean()), 6))
            sweep = {}
            for lam in RIDGE['lambda_sweep']:
                centre_model = ridge_fit(x_train, np.array([row['centre'] for row in train_rows],
                                                           dtype=np.float64), lam)
                centre_truth = np.array([row['centre'] for row in test_rows], dtype=np.float64)
                centre_pred = ridge_predict(centre_model, x_test)
                side_model = ridge_fit(x_train, y_train, lam)
                side_pred = (ridge_predict(side_model, x_test) > .5).astype(np.float64)
                sweep[str(lam)] = dict(side_correct=int((side_pred == y_test).sum()),
                                       side_rows=int(len(y_test)),
                                       centre_abs_error=float(np.abs(centre_pred - centre_truth).sum()))
            fold['lambda_sweep'] = sweep
            folds.append(fold)
        result[name] = dict(
            folds=folds, held_out_groups=[fold['held_out'] for fold in folds],
            side_accuracy=round(sum(fold['side']['accuracy'] for fold in folds) / len(folds), 6),
            side_baseline_accuracy=round(sum(fold['side']['baseline_accuracy'] for fold in folds)
                                         / len(folds), 6),
            centre_mae_px=round(sum(fold['centre']['mae_px'] for fold in folds) / len(folds), 6),
            centre_baseline_mae_px=round(sum(fold['centre']['baseline_mae_px'] for fold in folds)
                                         / len(folds), 6),
            offset_mae_px=round(sum(fold['offset']['mae_px'] for fold in folds) / len(folds), 6),
            offset_baseline_mae_px=round(sum(fold['offset']['baseline_mae_px'] for fold in folds)
                                         / len(folds), 6),
            lambda_sweep={lam: dict(
                side_accuracy=round(sum(fold['lambda_sweep'][lam]['side_correct'] for fold in folds)
                                    / sum(fold['lambda_sweep'][lam]['side_rows'] for fold in folds), 6),
                centre_mae_px=round(sum(fold['lambda_sweep'][lam]['centre_abs_error'] for fold in folds)
                                    / sum(fold['lambda_sweep'][lam]['side_rows'] for fold in folds), 6))
                for lam in [str(value) for value in RIDGE['lambda_sweep']]})
    result['design'] = dict(rows='decisions whose logical frame 2*i is inside [max(0, wall_frame-45), wall_frame)',
                            fit='F1 train events only (6 events, 3 groups, base + mirror)',
                            ridge=RIDGE,
                            targets='screen px: gap side, gap centre, gap centre minus the player x',
                            note='read-only measurement of the frozen representation; it trains nothing and '
                                 'reads no validation pixel')
    return result


def mirror_symmetry(vectors, worlds):
    """Are the column profiles of a mirror pair mirror images? A test of the geometry."""
    by_group = {}
    for event_id, world in worlds.items():
        by_group.setdefault(world.template_group, {})[bool(world.mirror)] = event_id
    rows = []
    for group, pair in sorted(by_group.items()):
        if set(pair) != {False, True}:
            continue
        left, right = vectors[pair[False]], vectors[pair[True]]
        columns_left = left[:, 42:42 + SPATIAL_COLUMNS].reshape(-1)
        columns_right = right[:, 42:42 + SPATIAL_COLUMNS][:, ::-1].reshape(-1)
        scale = float(np.std(np.concatenate([columns_left, columns_right])))
        rows.append(dict(group=group, rows=int(left.shape[0]),
                         correlation=round(float(np.corrcoef(columns_left, columns_right)[0, 1]), 6),
                         mean_abs_difference=round(float(np.abs(columns_left - columns_right).mean()), 6),
                         pooled_std=round(scale, 6),
                         relative_difference=round(float(np.abs(columns_left - columns_right).mean()
                                                         / scale), 6) if scale else None))
    return dict(pairs=rows,
                note='the column profile of m0 against the reversed column profile of m1 on the same group; '
                     'the F8 column axis is the screen-x axis, so a layout-reading backbone must be close to '
                     'mirror-symmetric here')


def fusion_ablation(backbone, device, closed, prereg, paths, reports):
    """Same S head weights, delta on vs delta off, on the six F1 holdout events.

    This is a within-checkpoint ablation, not a second arm: it changes no parameter and trains
    nothing. It answers "did the delta cause the behavioural difference, and did switching it
    off recover arm A's behaviour?" - which is exactly the question a 5.6%-conditioning layer
    makes urgent.
    """
    f1_validation = [row for row in event_records(prereg, r20.VALIDATION_SPLIT)
                     if row['family'] == 'F1']
    cells, summary = {}, {}
    for name in [n for n in all_names() if n.startswith(SPATIAL + '-')]:
        network = SpatialNetwork(backbone, paths[name], device,
                                 reports[name]['final_head_state_sha256'],
                                 reports[name]['final_fusion_state_sha256'])
        with_fusion = {row['event_id']: row for row in closed[(name, r20.VALIDATION_SPLIT)]['events']}
        network.fusion.enabled = False
        without = r21.closed_loop(f1_validation, network, 'N1', name + '-no-delta')
        network.fusion.enabled = True
        changed, total = 0, 0
        for row in without:
            world = make_world(next(entry['scenario'] for entry in f1_validation
                                    if entry['event_id'] == row['event_id']))
            reference = with_fusion[row['event_id']]
            left = [step.get('candidate') for step in row['steps']]
            right = [step.get('candidate') for step in reference['steps']]
            changed += sum(1 for a, b in zip(left, right) if a != b)
            total += min(len(left), len(right))
            cells[f'{name}/{row["event_id"]}'] = dict(
                network=name, event_id=row['event_id'], mirror=bool(world.mirror),
                with_delta=cell_diagnostic(world, reference, name, 'validation'),
                without_delta=cell_diagnostic(world, row, name, 'validation-no-delta'),
                first_hit_changed=(row['first_collision_frame'] != reference['first_collision_frame']))
        summary[name] = dict(events=len(without), decisions_compared=total,
                             decisions_changed=changed,
                             changed_share=round(changed / total, 6) if total else None,
                             feasible_with=sum(1 for row in with_fusion.values() if row['feasible']),
                             feasible_without=sum(1 for row in without if row['feasible']))
    return dict(cells=cells, summary=summary,
                method='the trained S checkpoint is rerun with `fusion.enabled = False` (the delta is '
                       'bypassed, no parameter changes, no retraining) on the six F1 holdout events; '
                       'feasibility, first hit, direction and parking error are then recomputed the same way',
                caveat='the S head weights were trained *with* the delta, so "delta off" is a '
                       'counterfactual probe of what the delta does, not arm A')


def fusion_diagnostics(networks, vectors, feature_table):
    """What did the fusion layer actually learn: a constant re-bias, or spatial conditioning?

    `constant` is `projection(mean spatial vector)` and `conditioning` is `delta - constant`,
    i.e. the part that varies because the screen varies. The 94-d vector's own mean dominates
    its spread (measured: mean 0.179 vs std 0.016), so a layer that has mostly learned a
    constant has re-biased the head rather than conditioned it on the screen - which is a
    different (and weaker) intervention than the one this round intends to test.
    """
    order = sorted(vectors)
    stacked = np.concatenate([vectors[event] for event in order]).astype(np.float32)
    features = np.concatenate([feature_table[event] for event in order]).astype(np.float32)
    feature_norm = float(np.linalg.norm(features, axis=1).mean())
    result = {}
    for name in [n for n in networks if n.startswith(SPATIAL + '-')]:
        fusion = networks[name].fusion
        if not fusion.enabled:
            raise ValueError('only the spatial arm carries a live fusion layer')
        device = fusion.projection.weight.device
        tensor = torch.from_numpy(stacked).to(device)
        mean_vector = torch.from_numpy(stacked.mean(0, keepdims=True)).to(device)
        with torch.no_grad():
            delta = fusion.projection(tensor).cpu().numpy()
            constant = fusion.projection(mean_vector).cpu().numpy()[0]
        conditioned = delta - constant
        delta_norm = np.linalg.norm(delta, axis=1)
        conditioned_norm = np.linalg.norm(conditioned, axis=1)
        result[name] = dict(
            rows=int(delta.shape[0]),
            weight_frobenius=round(float(fusion.projection.weight.detach().norm()), 6),
            bias_norm=round(float(fusion.projection.bias.detach().norm()), 6),
            delta_norm_mean=round(float(delta_norm.mean()), 6),
            delta_norm_max=round(float(delta_norm.max()), 6),
            constant_norm=round(float(np.linalg.norm(constant)), 6),
            conditioning_norm_mean=round(float(conditioned_norm.mean()), 6),
            conditioning_share=round(float(conditioned_norm.mean() / delta_norm.mean()), 6),
            frozen_feature_norm_mean=round(feature_norm, 6),
            delta_to_feature_ratio=round(float(delta_norm.mean() / feature_norm), 6))
    result['design'] = dict(
        rows='all 80 decision rows of the 14 train events, on the stored frozen vectors; read-only',
        note='constant = projection(mean spatial vector); conditioning = delta - constant. Both arms are '
             'otherwise identical, so this separates "the fusion re-biased the head" from "the fusion '
             'conditioned it on the screen"')
    return result


def positions_per_row(world, actions):
    """The player's x on every decision frame, from the stored teacher actions (physics only)."""
    states = full_states(world, actions)
    if len(states) != r20.HORIZON + 1:
        raise ValueError('the stored rollout does not replay to the full horizon')
    return [float(states[frame].x) for frame in r21.DECISION_FRAMES]


def run_probes(backbone, device, closed, prereg, records, vectors, feature_table, paths, reports):
    """Every mechanism probe of this round, all read-only on frozen artifacts."""
    validation = event_records(prereg, r20.VALIDATION_SPLIT)
    train = event_records(prereg, r20.TRAIN_SPLIT)
    f1_validation = [row for row in validation if row['family'] == 'F1']
    f1_train = [row for row in train if row['family'] == 'F1']
    networks = {'B0': None}
    for name in all_names():
        networks[name] = SpatialNetwork(backbone, Path(paths[name]), device,
                                        reports[name]['final_head_state_sha256'],
                                        reports[name]['final_fusion_state_sha256'])
    probes = []
    for entry in f1_validation:
        world = make_world(entry['scenario'])
        for name in ['B0'] + all_names():
            steps = next(row for row in closed[(name, r20.VALIDATION_SPLIT)]['events']
                         if row['event_id'] == entry['event_id'])['steps']
            if name == 'B0':
                probes.append(flip_probe(backbone, device, world, steps, name, head_only=True))
            else:
                probes.append(flip_probe(networks[name], device, world, steps, name))
    flip = flip_summary(probes)
    stored = stored_actions(closed, r20.VALIDATION_SPLIT, validation)
    worlds = {}
    for entry in f1_train:
        worlds[entry['event_id']] = make_world(entry['scenario'])
    rows_by_event, positions = {}, {}
    for entry in f1_train:
        world = worlds[entry['event_id']]
        rollout = json.loads((R21_EVIDENCE / 'run' / next(
            row for row in records if row['event_id'] == entry['event_id'])['rollout']['path'])
            .read_text(encoding='utf-8'))
        positions[entry['event_id']] = positions_per_row(world, rollout['actions'])
        window = r21.telegraph_window(world)
        rows_by_event[entry['event_id']] = [index for index, frame in enumerate(r21.DECISION_FRAMES)
                                            if window[0] <= frame < window[1]]
    return dict(
        flip=dict(networks=flip, events=[dict(network=row['network'], event_id=row['event_id'],
                                              steps=row['steps'],
                                              stored_candidate_matches=row['stored_candidate_matches'])
                                         for row in probes],
                  method='each stored closed-loop trajectory is re-rendered from its own x trace and replayed; '
                         'at every decision inside the telegraph window the same network state also sees the '
                         'playfield mirrored horizontally (hud untouched) and the two masked argmax screen '
                         'actions are compared'),
        prediction=dict(validation=stored, train=stored_actions(closed, r20.TRAIN_SPLIT, train),
                        note='the masked argmax actually taken, from the stored rows; per event, over all 80 '
                             'decisions and over the telegraph-window decisions only'),
        base_bias=dict(
            note='the frozen baseline (B0) is the same backbone policy head the residual offset is built on; '
                 'its per-decision screen preference on the identical telegraph renders is the base prior '
                 'this round is testing against',
            validation=stored['B0']),
        spatial=spatial_probe(vectors, feature_table, worlds, positions, rows_by_event),
        symmetry=mirror_symmetry(vectors, worlds),
        fusion=fusion_diagnostics(networks, vectors, feature_table),
        ablation=fusion_ablation(backbone, device, closed, prereg, paths, reports),
        replay_note='the flip probe also reports replay fidelity: the fraction of steps on which the rebuilt '
                    'stack reproduces the stored masked argmax, i.e. that the probe sees the runner inputs')


# ---------------------------------------------------------------- pre-registration
def criterion():
    """The frozen decision rules of this round; any drift aborts the run."""
    return dict(
        round='round22',
        purpose='one pre-registered structural intervention on top of round21: give the unchanged N1 head a '
                'pooled spatial view of the frozen backbone F8 map, so the screen position of the F1 gap is '
                'linearly reachable from the head input, and read the result out mechanistically',
        intervention=dict(
            change='a new fusion layer between the frozen trunk features and the unchanged N1 head',
            spatial_vector=dict(source='F8 [B,96,28,24] from the frozen backbone; F4 [B,64,56,48] is taken '
                                       'from the same call and shape-locked but not used',
                                pooled_7x6=42, columns_W=SPATIAL_COLUMNS, rows_H=SPATIAL_ROWS,
                                total=SPATIAL_DIM,
                                definition='concat(adaptive_avg_pool2d(f8, (7,6)).mean(channel).flatten, '
                                           'f8.mean(channel, row), f8.mean(channel, column))'),
            projection=dict(layer='Linear(94, 256, bias=True)', parameters=FUSION_PARAMETERS,
                            init=FUSION_INIT),
            fusion='fused = features + projection(spatial); arm A bypasses the layer and freezes it at zero',
            head='TacticalPolicy.forward_features with exactly one line changed: the incoming `features` is '
                 'not detached (the parent detaches it, which would leave the fusion layer with zero gradient '
                 'for all 200 updates, making arm S vacuous); a unit test asserts the two forwards are bitwise '
                 'identical on identical inputs and that the body is the parent body with that one line changed',
            base_logits='still backbone.policy_head(original features), never recomputed from the fused vector, '
                        'so the residual head keeps its round20/21 meaning',
            not_changed='worlds, world parameters, R19 split, round21 renderer and pixels, teacher actions, '
                        'labels, pixel packing, encoder, loss weights, optimizer, learning rate, clip, sequence, '
                        'batch, update count, seed set, closed-loop runner, condition, masking',
            parameter_increment='S adds exactly %d active parameters (94*256+256); A adds 0 and keeps '
                                'round21\'s 429200' % FUSION_PARAMETERS),
        data=dict(source='round21 run/records.json assets, reopened read-only and SHA-bound per event; nothing '
                         're-rendered, no teacher called, no new label created',
                  train=dict(events=14, groups=7, groups_per_family={'F1': 3, 'F2': 1, 'F3': 3},
                             use='the only events encoded or trained on'),
                  validation=dict(events=14, groups=7, groups_per_family={'F1': 3, 'F2': 1, 'F3': 3},
                                  use='read-only: never trained, never labelled, never encoded into the training '
                                      'set; closed loop and input probes only'),
                  bindings='pixels.npz, history.pt, labels.pt and rollout.json must equal round21\'s frozen SHA '
                           'table for all 14 train events, otherwise the run aborts'),
        training=dict(arms={CONTROL: 'fusion bypassed and frozen at zero: round21 arm A',
                            SPATIAL: 'fusion live from a zero initialisation'},
                      loss_arm=LOSS_ARM, anchor_weight=0., seeds=list(SEEDS), updates=r20.UPDATES,
                      optimizer='Adam', lr=r20.LR, clip=r20.CLIP, sequence=r20.SEQUENCE, batch=r20.BATCH,
                      precision='FP32 without AMP, autocast disabled, matmul TF32 False, cuDNN TF32 False, '
                                'float32 matmul precision highest',
                      loop='round20 train_all with two additions only: the fused feature and the saved fusion '
                           'state',
                      initialization='fresh seed reset per construction; the initial head SHA must equal the '
                                     'frozen R16/R20/R21 table for the seed and the two arms must agree '
                                     'bitwise; the fusion layer must start at exactly zero in both arms',
                      equivalence='because the fusion starts at exactly zero and arm A is bypassed, step 0 of '
                                  'arm S is bitwise arm A; a unit test asserts identical policy logits and '
                                  'identical masked actions on real data rows',
                      control_reproduction='arm A must reproduce round21\'s A checkpoint and its closed-loop '
                                           'rows bitwise; if it does not, that is reported as '
                                           '`control_not_reproduced` and the S-vs-A comparison is withheld',
                      selection='final step only: no early stopping, no validation-based selection, no tuning'),
        evaluation=dict(
            order='all six checkpoints written and SHA-frozen first; then the six R18 baselines, then B0, then '
                  'the six checkpoints, validation before train',
            executor='round21.run_event_rendered / round21.closed_loop reused verbatim, so the evaluation code '
                     'is round21 code and not a copy',
            condition=r20.CONDITION, renderer='v373_telegraph.render_state_telegraph',
            b0='the frozen baseline policy logits on the identical telegraph renders; a reference, not a '
               'capability',
            six_baselines='round20.six_baselines re-executed on the same worlds; the 168 rows must equal '
                          'round21\'s',
            holdout_rule='if ANY of the six R18 baselines is feasible on ANY validation event the R18 filter '
                         'has failed on the holdout side: report it verbatim and withhold the learning conclusion',
            definitions=dict(
                direction='rebuild the logical-frame x trace from the executed actions with `advance` semantics '
                          '(round21.gap_trace), take the canonical displacement start -> wall_frame-1: that is '
                          '`direction`; `screen_direction` is the same displacement in raw x',
                toward='the canonical direction points from the start\'s canonical x at the gap\'s canonical '
                       'centre, so `toward` is "moved toward the slot" regardless of the mirror; `away` is the '
                       'opposite and `hold` is zero net motion',
                parked='`signed_error_px` is measured at wall_frame-1 in canonical px: 0 inside [gap0, gap1], '
                       'negative short of the left edge, positive past the right edge',
                mirror_flip_rate='the fraction of mirror-group cells whose canonical motion is toward the gap; '
                                 'because the same screen-right prior is right in the base group and wrong in '
                                 'the mirror group, this is the direction flip this round is about',
                identical_cells='for every seed, the number of the 14 validation events on which the two arms '
                                'have the same first-hit frame (a stricter "unchanged" than the F1 count alone)'),
            primary_metrics=[
                'F1 holdout, per group (base / mirror) and per arm: cells that move toward the gap, hold, or '
                'move away, plus the mirror toward rate (the direction flip rate this round is about)',
                'F1 holdout, per cell: whether the decision-time x is inside the gap at wall_frame-1 and the '
                'signed parking error in px (negative = short of the left edge, positive = past the right edge)',
                'per family F1/F2/F3 and pooled: feasible and survived per network, every event first hit, totals, '
                'and for each seed the number of validation events whose first hit is identical between the arms',
                'discriminating evidence for "base prior" vs "spatial information unavailable": the stored '
                'prediction distribution per mirror group, an input-mirror flip probe on every head, a linear '
                'decodability probe of the 94-d vector and of the 256-d trunk features, the frozen B0 bias, a '
                'decomposition of the learned delta into a constant re-bias and a screen-dependent part, and a '
                'within-checkpoint delta-on/delta-off ablation that changes no parameter'],
            probe_caveats=dict(
                flip='the probe mirrors the whole playfield, so the player sprite is mirrored too; a flipped '
                     'preference therefore cannot be attributed to the gap alone. It is reported with its '
                     'moving-step denominator, and the distilled heads hold on almost every telegraph-window '
                     'decision, so its power is small by construction',
                decodability='the probe fits on F1 train pixels only and leaves one geometry group out; its '
                             'conclusion is reported together with a seven-order lambda sweep so that it does not '
                             'rest on one shrinkage value, and p >> n means a small lambda is an interpolator',
                ablation='the S head was trained with the delta, so "delta off" is a counterfactual, not arm A'),
            reporting='per family, never only aggregated; n=3 seeds; F2 validation has one group; mirrors are '
                      'not independent samples; every F1 cell is listed',
            degeneracy='single_candidate and switches<=1 are counted, listed and reported, never dropped'),
        verdict_rules=dict(
            equivalence='arm A must reproduce round21\'s A checkpoint SHA and every closed-loop row bitwise; if it '
                        'does not, the round reports `control_not_reproduced` and withholds the S-vs-A conclusion',
            improvement='F1 validation feasible per seed, S minus A: any seed strictly greater = improved, all '
                        'equal = unchanged, any seed lower = worse',
            mechanism='the mirror toward rate and the parking errors are reported as counts and px, never as a '
                      'significance claim; a failure to move them is reported as a failure',
            honesty='if the mirror direction does not improve, the round must state which of the two readings the '
                    'evidence supports (base prior vs vector too coarse) instead of leaving it open'),
        claims=dict(significance='none: 3 seeds, 7 validation groups, at most 14 independent geometries; '
                                 'descriptive only',
                    gate='no formal gate opened, no deployable checkpoint produced, no game executed',
                    can_show='whether a 94-d pooled spatial view of the same frozen backbone is enough for the '
                             'same N1 head to stop being dominated by the screen-right prior, and if not, '
                             'whether the reason is the prior or the vector',
                    cannot_show='tactical understanding, learnability beyond these 7 held-out geometries, or '
                                'that any policy reads a telegraph the way a player would'),
        failure_policy='first failure retained; one run; no retrain, rerun, repair of numbers, extra epoch, '
                       'tuning or baseline switch')


def criterion_sha(value=None):
    return hashlib.sha256(json.dumps(criterion() if value is None else value, sort_keys=True).encode()).hexdigest()


def identity_paths():
    return [Path(__file__), Path('tests/test_v373_spatial_fuse.py'),
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
            R21_EVIDENCE / 'preregistration.json', R21_EVIDENCE / 'run/records.json',
            R21_EVIDENCE / 'run/summary.json', R21_EVIDENCE / 'run/checkpoints.json',
            R21_EVIDENCE / 'run/f1-gap.json', R21_EVIDENCE / 'run/baselines.json',
            REPORTS / 'P4-telegraph-round21.md', REPORTS / 'P1-f1-visibility-round20.md']


def identities():
    return {str(path): file_sha(path) for path in identity_paths()}


def world_table(events):
    """Frozen geometry of every event this round touches, read from its world."""
    rows = {}
    for event in events:
        world = make_world(event['scenario'])
        gap_screen = r21.telegraph_gap(world)
        window = r21.telegraph_window(world)
        rows[event['event_id']] = dict(
            family=world.family, mirror=bool(world.mirror), group=world.template_group,
            start_x=float(world.start_x), start_canonical=float(world.canonical_x(world.start_x)),
            wall_frame=int(world.wall_frame), coverage=list(world.coverage),
            internal_gap=bool(world.internal_gap),
            gap=list(world.gap) if world.internal_gap else None,
            gap_screen=list(gap_screen) if gap_screen else None,
            preview_window=[max(0, window[0]), window[1]] if window else None,
            last_preview_frame=(world.wall_frame - 1) if window else None,
            decisions_in_preview=(sum(1 for frame in r21.DECISION_FRAMES
                                      if window[0] <= frame < window[1]) if window else None))
    return rows


def fusion_init_shas():
    out = {}
    for arm in ARMS:
        torch.manual_seed(0)
        out[arm] = r16.state_sha(SpatialFusion(enabled=(arm == SPATIAL)).state_dict())
    return out


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
    if sorted(r21_assets()) != sorted(row['event_id'] for row in train):
        raise ValueError('round21 assets must cover exactly the 14 train events')
    reference = r21_reference()
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('real CUDA required for the frozen spatial statistics')
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
        data, vectors, feature_table = spatial_vectors(backbone, records, R21_EVIDENCE / 'run', device)
        statistics = spatial_table(vectors)
        digest = spatial_digest(vectors)
        feature_norm = {row['entry']['event_id']: round(float(row['features'].norm(dim=-1).mean()), 6)
                        for row in data}
    shapes = dict(f4=list(F4_SHAPE), f8=list(F8_SHAPE), playfield=list(PLAYFIELD_HW),
                  spatial=list(vectors[train[0]['event_id']].shape),
                  pooled=42, columns=SPATIAL_COLUMNS, rows=SPATIAL_ROWS, total=SPATIAL_DIM,
                  projection=dict(in_features=SPATIAL_DIM, out_features=FEATURE_DIM,
                                  parameters=FUSION_PARAMETERS))
    value = dict(
        schema=PREREG_SCHEMA, identities=identities(), criterion=criterion(),
        criterion_sha256=criterion_sha(),
        arms=list(ARMS), seeds=list(SEEDS), updates=r20.UPDATES, lr=r20.LR, clip=r20.CLIP,
        sequence=r20.SEQUENCE, batch=r20.BATCH, loss_arm=LOSS_ARM, anchor_weight=0.,
        sources=dict(manifest=dict(path=str(r20.MANIFEST), sha256=manifest['sha256'],
                                   schema=r20.MANIFEST_SCHEMA),
                     split=dict(path=str(r20.SPLIT), sha256=split['sha256'], schema=r20.SPLIT_SCHEMA),
                     baseline=dict(path=str(BASELINE), sha256=file_sha(BASELINE)),
                     round21=dict(preregistration=file_sha(R21_EVIDENCE / 'preregistration.json'),
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
        fusion=dict(shapes=shapes, spatial_sha256=digest, statistics=statistics,
                    feature_norm=feature_norm, init=FUSION_INIT,
                    init_state_sha256=fusion_init_shas(),
                    definition='F8 -> 7x6 average pool over channels (42) + channel-and-row mean per column (24) '
                               '+ channel-and-column mean per row (28) = 94, then Linear(94,256) initialised at '
                               'exactly zero; arm A bypasses the layer and keeps it frozen at zero',
                    private_calls='backbone._checked_regions, backbone._playfield_input and backbone.playfield '
                                  'are called in the same order forward_core calls them; a unit test recomputes '
                                  'forward_core\'s tail from the extracted F4/F8 and asserts bitwise equality',
                    resolution=dict(native_playfield_width=384, model_playfield_width=192,
                                    f8_columns=SPATIAL_COLUMNS, native_px_per_column=384 // SPATIAL_COLUMNS,
                                    pooled_cell_native_px=384 // SPATIAL_POOL[1],
                                    note='the F8 column axis is the screen-x axis: 192 model columns / 24 = 8 '
                                         'model px per column, i.e. 16 native playfield px, so a 12-20 px gap is '
                                         'about one F8 column wide and the frozen vector is coarse by construction')),
        geometry=world_table(train + validation),
        validation_bindings=asset_bindings(records),
        initialization=dict(per_seed=initial,
                            per_seed_per_arm={str(seed): {arm: initial[str(seed)] for arm in ARMS}
                                              for seed in SEEDS},
                            identical_between_arms=True, equals_round16_round20_round21=True,
                            fusion_per_arm=fusion_init_shas(),
                            method='fresh seed reset immediately before every N1 construction after the '
                                   'backbone load, through the frozen r16 recipe; the fusion layer is built '
                                   'afterwards and zero-initialised, so it cannot move the head initialisation'),
        round21_reference=reference,
        expected=EXPECTED,
        verification=dict(
            prerequisites=['every training asset equals round21\'s frozen SHA',
                           'the spatial vectors recompute to the frozen digest',
                           'the initial head SHA equals the frozen table for every seed',
                           'the six R18 baselines reproduce round21\'s 168 rows'],
            control='arm A must equal round21\'s A checkpoint SHA and its closed-loop rows for all 3 seeds; '
                    'otherwise it is reported as `control_not_reproduced` and the S-vs-A comparison is withheld',
            step0='a unit test asserts that step 0 of arm S is bitwise arm A (identical policy logits, '
                  'identical masked action, delta exactly 0)'),
        independent_test=False, formal_gate=False, deployable=False, game=False, tuning=False,
        new_worlds=False, new_teacher_labels=False, teacher_calls=0, official_labels=False,
        research_teacher_labels=True, new_pixels=False,
        failure_policy='first failure retained; one run; no retrain, rerun, repair, overwrite, extra epoch or '
                       'baseline switch')
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
    if fusion_init_shas() != p['initialization']['fusion_per_arm']:
        raise ValueError('fusion initialisation drift')
    if json.dumps(r21_reference(), sort_keys=True) != json.dumps(p['round21_reference'], sort_keys=True):
        raise ValueError('the round21 reference record changed')
    train = r20.world_events(manifest, split, r20.TRAIN_SPLIT)
    validation = r20.world_events(manifest, split, r20.VALIDATION_SPLIT)
    if json.dumps(world_table(train + validation), sort_keys=True) != json.dumps(p['geometry'], sort_keys=True):
        raise ValueError('the frozen geometry table changed')
    if sorted(r21_assets()) != sorted(p['worlds']['train_events']):
        raise ValueError('round21 asset set changed')
    backbone, _ = load_vision_checkpoint(BASELINE, map_location='cpu')
    backbone.to(device).eval()
    with _fp32_mode(torch.device(device)), torch.inference_mode():
        records = train_records(p)
        if asset_bindings(records) != p['validation_bindings']:
            raise ValueError('the asset bindings changed')
        _, vectors, _ = spatial_vectors(backbone, records, R21_EVIDENCE / 'run', device)
        if spatial_digest(vectors) != p['fusion']['spatial_sha256']:
            raise ValueError('the spatial vectors changed since the pre-registration')
        if spatial_table(vectors) != p['fusion']['statistics']:
            raise ValueError('the spatial statistics changed since the pre-registration')
    return p


# ---------------------------------------------------------------- summary
def comparison(closed):
    """S minus A on the same seed and the same six F1 validation events, plus the cell check."""
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
        spatial = closed[(r20.name_of(SPATIAL, seed), r20.VALIDATION_SPLIT)]['families']['all']['firsthit']
        events = sorted(control)
        same_first_hit = [event for event in events if control[event] == spatial[event]]
        moved = {event: [control[event], spatial[event]] for event in events
                 if control[event] != spatial[event]}
        pairs.append(dict(seed=seed, control=per_arm[CONTROL][index]['f1'],
                          spatial=per_arm[SPATIAL][index]['f1'],
                          delta=per_arm[SPATIAL][index]['f1'] - per_arm[CONTROL][index]['f1'],
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
                note='three seed pairs, six F1 validation events each; S minus A; descriptive only')


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
        training[name] = payload
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
    reference = r21_reference()
    equivalence = control_equivalence(closed)
    networks_by_arm = {CONTROL: [n for n in all_names() if n.startswith(CONTROL + '-')],
                       SPATIAL: [n for n in all_names() if n.startswith(SPATIAL + '-')],
                       'B0': ['B0']}
    mechanism = {split_name: mechanism_block(gaps[split_name], networks_by_arm)
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
                             final_fusion_state_sha256=training[name]['final_fusion_state_sha256'],
                             matches_round21_control=training[name]['matches_round21_control'],
                             loss_at_step0=training[name]['loss_at_step0'],
                             loss_final=training[name]['loss_final'],
                             fusion=training[name]['fusion'],
                             fusion_grad_norm=training[name]['fusion_grad_norm'],
                             delta_norm=training[name]['delta_norm'],
                             parameters=training[name]['parameters'],
                             optimization_seconds=training[name]['optimization_seconds'],
                             initial_offline=training[name]['initial_offline'],
                             final_offline=training[name]['final_offline'])
    holdout = baselines['validation']
    summary = dict(
        schema=RESULT_SCHEMA, prereg_sha256=file_sha(out / 'preregistration.json'),
        criterion_sha256=prereg['criterion_sha256'], seeds=list(SEEDS), arms=list(ARMS),
        updates=r20.UPDATES, loss_arm=LOSS_ARM, anchor_weight=0.,
        intervention=dict(shapes=prereg['fusion']['shapes'],
                          spatial_sha256=prereg['fusion']['spatial_sha256'],
                          definition=prereg['fusion']['definition'],
                          private_calls=prereg['fusion']['private_calls'],
                          resolution=prereg['fusion']['resolution'],
                          feature_norm=prereg['fusion']['feature_norm'],
                          head='SpatialHead: TacticalPolicy.forward_features with the input `features` detach '
                               'removed and nothing else',
                          note='the only change against round21 is the fusion layer between the frozen trunk '
                               'features and the same N1 head; A is round21 A'),
        sources=prereg['sources'], worlds=prereg['worlds'],
        data=dict(events=len(records['events']), rows=records['totals']['rows'],
                  bindings=records['bindings'], storage=records['storage'],
                  spatial=records['spatial'], spatial_definition=records['spatial_definition'],
                  totals=records['totals'], failures=records['failures']),
        initialization=dict(per_seed=prereg['initialization']['per_seed'],
                            per_seed_per_arm={str(seed): {arm: training[r20.name_of(arm, seed)]['initial_head_state_sha256']
                                                          for arm in ARMS} for seed in SEEDS},
                            identical_between_arms=True,
                            fusion_per_arm=prereg['initialization']['fusion_per_arm'],
                            equals_round16_round20_round21=True),
        equivalence=dict(control=equivalence, all_equal=equivalence['all_equal'],
                         matches_round21_control={name: training[name]['matches_round21_control']
                                                  for name in all_names() if name.startswith(CONTROL + '-')},
                         step0='unit tested: identical policy logits and identical masked action at step 0'),
        checkpoints={name: {key: value for key, value in training[name].items()
                            if key not in ('losses', 'original_losses',
                                           'initial_offline_events', 'final_offline_events',
                                           'supervised_events')} for name in all_names()},
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
        round21_reference=reference,
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
                   retrained=False, rerun=False, physics_changed=False, intervention_count=1),
        limitations=[
            'n=3 seeds and 7 validation groups; the 14 validation events collapse to 7 geometries because a '
            'mirror is not an independent sample',
            'F2 validation is a single group (2 events), so an F2 validation number is not a rate',
            'the intervention is a 94-d pooled view of a 28x24 map: one F8 column is 16 native px, so a 12-20 px '
            'gap is about one column wide and the vector is coarse by construction',
            'the fusion layer is the only trained addition; the frozen backbone and the head geometry are round20',
            'the F1 win condition is exact parking inside a 12-20 px gap; it is not tactical understanding',
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
            if file_sha(R21_EVIDENCE / 'run' / entry[key]['path']) != entry[key]['sha256']:
                raise ValueError('round21 asset SHA drift: ' + entry['event_id'] + '/' + key)
            checked += 1
    for event_id, ref in records['spatial'].items():
        if file_sha(dest / ref['path']) != ref['sha256']:
            raise ValueError('stored spatial vector SHA drift: ' + event_id)
        checked += 1
    checkpoints = json.loads((dest / 'checkpoints.json').read_text(encoding='utf-8'))
    for name, record in checkpoints.items():
        if file_sha(dest / record['checkpoint_path']) != record['checkpoint_sha256']:
            raise ValueError('checkpoint SHA drift: ' + name)
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
                fuse_parameter_increment=FUSION_PARAMETERS,
                spatial_sha256=prereg['fusion']['spatial_sha256'],
                flip_probe=f"{len(probes['flip']['networks'])} networks",
                train_events=len(prereg['worlds']['train_events']),
                validation_events=len(prereg['worlds']['validation_events']),
                criterion_sha256=prereg['criterion_sha256'], identities=len(prereg['identities']))


def write_events(dest, prereg, records, closed, gaps, spatial_refs):
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
        if asset is not None:
            with np.load(dest / spatial_refs[event_id]['path']) as archive:
                value['spatial'] = spatial_stats(np.array(archive['spatial']))
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
            data, vectors, feature_table = spatial_vectors(backbone, records, R21_EVIDENCE / 'run', device)
            if spatial_digest(vectors) != prereg['fusion']['spatial_sha256']:
                raise AssertionError('the spatial vectors differ from the pre-registration')
            if spatial_table(vectors) != prereg['fusion']['statistics']:
                raise AssertionError('the spatial statistics differ from the pre-registration')
            if len(data) != EXPECTED['train_events']:
                raise AssertionError('encoded event count')
            spatial_refs = write_spatial(dest, vectors)
            records_value = write_records(dest, records, spatial_refs)
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
                      'logical frame is included; "in_gap_at_wall" is the position on wall_frame-1, the last '
                      'frame before the band becomes lethal; "direction" is canonical (round21 semantics) and '
                      '"screen_direction" is the raw x motion, which is what the base prior acts on',
                validation=gaps[r20.VALIDATION_SPLIT], train=gaps[r20.TRAIN_SPLIT]))
            begin = time.perf_counter()
            probes = run_probes(backbone, device, closed, prereg, records, vectors, feature_table,
                                paths, reports)
            r20.write_json(dest / 'probes.json', probes)
            timings['probes_seconds'] = round(time.perf_counter() - begin, 3)
            write_events(dest, prereg, records, closed, gaps, spatial_refs)
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
