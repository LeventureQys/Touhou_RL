"""CPU-only causal sequence and *model proxy*, never world-safety, contracts.

Centres already contain encoded depth-3 pixel features. Shard boundaries therefore
are not recurrent boundaries. Evaluation must traverse windows in returned order,
carry hidden state between windows, and reset only at reset_mask (not every batch).
Duration classes 0/1/2 mean exactly 1/3/5 decisions; BOS values are 18 and 3.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

DURATIONS = np.asarray([1, 3, 5], dtype=np.int64)
BOS_CANDIDATE = 18
BOS_DURATION = 3


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _audit_rows(rows):
    """Reject mixed rollout identities, reordered episodes/decisions and samples."""
    identity = None
    previous_episode = previous_decision = previous_sample = None
    for row in rows:
        rid = row.get('rollout_id')
        if not isinstance(rid, str) or not rid:
            raise ValueError('missing rollout identity')
        if identity is None:
            identity = rid
        if rid != identity:
            raise ValueError('rows must have one unique rollout identity')
        ep, decision = row.get('episode'), row.get('decision')
        if not _integer(ep) or not _integer(decision) or ep < 0 or decision < 0:
            raise ValueError('invalid episode/decision')
        if previous_episode is not None:
            if ep < previous_episode or (ep == previous_episode and decision <= previous_decision):
                raise ValueError('episode/decision order or duplicate')
        sample = row.get('sample_index')
        if sample is not None:
            if not _integer(sample) or sample < 0 or (previous_sample is not None and sample <= previous_sample):
                raise ValueError('sample order or duplicate')
            previous_sample = sample
        previous_episode, previous_decision = ep, decision


def _trusted(row):
    raw = row.get('raw_safety_snapshot') or {}
    return (row.get('aligned') is True and row.get('fresh_pixels') is True
            and raw.get('complete') is True and raw.get('player_state') in (0, 3)
            and _integer(row.get('frame')) and _integer(row.get('stage'))
            and raw.get('frame') == row['frame'] and raw.get('stage') == row['stage']
            and _integer(row.get('deaths_before')) and _integer(row.get('deaths_after')))


def build_sequence_index(rows, episode_splits, max_frame_gap=30):
    """Return continuous segments of global sample indexes; never sort trace rows.

    ``episode_splits`` maps each episode id to its train/dev/test split. Missing
    assignments fail rather than accidentally putting unknown episodes in train.
    An outcome death/termination belongs to the current row; the *next* row resets.
    """
    rows = list(rows)
    _audit_rows(rows)
    if not _integer(max_frame_gap) or max_frame_gap < 1:
        raise ValueError('max_frame_gap must be positive integer')
    if not isinstance(episode_splits, Mapping):
        raise ValueError('episode_splits must map episode to split')
    segments, current, previous = [], [], None
    for row in rows:
        if row['episode'] not in episode_splits or episode_splits[row['episode']] is None:
            raise ValueError('missing episode split')
        trusted = row.get('sample_index') is not None and _trusted(row)
        continuous = (trusted and previous is not None
                      and row['episode'] == previous['episode']
                      and episode_splits[row['episode']] == episode_splits[previous['episode']]
                      and row['decision'] == previous['decision'] + 1
                      and row['stage'] == previous['stage']
                      and 0 < row['frame'] - previous['frame'] <= max_frame_gap
                      and row['deaths_before'] == previous['deaths_after']
                      and previous['deaths_after'] == previous['deaths_before']
                      and not previous.get('terminated', False)
                      and not previous.get('truncated', False)
                      and not previous.get('stage_delta', 0))
        if not continuous and current:
            segments.append(current)
            current = []
        if trusted:
            current.append(int(row['sample_index']))
        previous = row if trusted else None
    if current:
        segments.append(current)
    return segments


def _segment_audit(segments):
    seen = set()
    for segment in segments:
        last = -1
        for sample in segment:
            if not _integer(sample) or sample < 0 or sample in seen or sample <= last:
                raise ValueError('invalid/duplicate/out-of-order segment sample')
            seen.add(sample)
            last = sample


def previous_history(rows, segments):
    """Causal previous executed (candidate, duration class), otherwise paired BOS.

    Never reads teacher_action, death_window, future labels or legacy game-frame
    deltas. Missing/unrecognized duration masks the entire previous-action pair.
    """
    rows, segments = list(rows), list(segments)
    _audit_rows(rows)
    _segment_audit(segments)
    lookup = {r['sample_index']: r for r in rows if r.get('sample_index') is not None}
    result = {}
    for segment in segments:
        previous = None
        for sample in segment:
            if sample not in lookup:
                raise ValueError('segment sample absent from rows')
            row = lookup[sample]
            if previous is not None and (row['episode'] != previous['episode'] or row['decision'] != previous['decision'] + 1):
                raise ValueError('history crosses episode/decision boundary')
            pair = (BOS_CANDIDATE, BOS_DURATION)
            if previous is not None:
                action, duration = previous.get('executed_action'), previous.get('action_duration_decisions')
                if (isinstance(action, (list, tuple, np.ndarray)) and len(action) == 4
                        and all(_integer(x) for x in action) and 0 <= action[0] < 9
                        and all(x in (0, 1) for x in action[1:])
                        and _integer(duration) and duration in (1, 3, 5)):
                    pair = (int(action[2] * 9 + action[0]), (1, 3, 5).index(duration))
            result[sample] = pair
            previous = row
    return result


def build_windows(segments, length=16, burn_in=4):
    """Fixed nonoverlap padded windows in segment order, suitable for true carry.

    Only the first ``burn_in`` centres of each *segment* have loss masked; later
    windows carry real hidden state, so do not burn them in again. BOS is only
    segment start. Never train/evaluate padding or index features at sentinel -1.
    Evaluation uses valid_mask, including burn-in, and no shuffled one-frame pass.
    """
    if not _integer(length) or length < 1 or not _integer(burn_in) or not 0 <= burn_in < length:
        raise ValueError('require length > burn_in >= 0')
    segments = list(segments)
    _segment_audit(segments)
    windows = []
    for sid, segment in enumerate(segments):
        for start in range(0, len(segment), length):
            n = min(length, len(segment) - start)
            indices = np.full(length, -1, dtype=np.int64)
            indices[:n] = segment[start:start+n]
            valid = np.arange(length) < n
            reset = np.zeros(length, dtype=bool)
            reset[0] = start == 0
            loss = valid & (start + np.arange(length) >= burn_in)
            windows.append(dict(sample_indices=indices, valid_mask=valid, loss_mask=loss,
                                reset_mask=reset, bos_mask=reset.copy(), segment_id=sid,
                                offset=start, carry=start > 0))
    return windows


def _proxy_arrays(clearance, model_valid=None):
    c = np.asarray(clearance, dtype=np.float64)
    if c.ndim != 3 or c.shape[1:] != (18, 3):
        raise ValueError('clearance must be [N,18,3]')
    if model_valid is None:
        valid = np.isfinite(c)
    else:
        mask = np.asarray(model_valid)
        if mask.shape != c.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError('model_valid must be binary [N,18,3]')
        valid = mask.astype(bool) & np.isfinite(c)
    # Prefix-minimum clearance cannot improve over a longer forced hold.
    for a, b in ((0, 1), (1, 2), (0, 2)):
        if np.any(valid[..., a] & valid[..., b] & (c[..., b] > c[..., a] + 1e-6)):
            raise ValueError('duration clearance must be monotone nonincreasing')
    return c, valid


def chunk_targets(clearance, model_valid, safety_threshold_px=2., duration_discount=.25):
    """Model-only targets, with full 3-duration support required per candidate.

    joint_target is discounted clearance c/(1+discount*(duration-1)); its flattened
    argmax uses candidate*3+duration_class. best_duration instead selects longest
    strictly safe (>threshold) hold, or class0 if none safe (still valid/dangerous).
    No observed death or world-risk label is consumed. Counts are never rebalanced.
    """
    if not np.isfinite(safety_threshold_px) or not np.isfinite(duration_discount) or duration_discount < 0:
        raise ValueError('invalid threshold/discount')
    c, valid = _proxy_arrays(clearance, model_valid)
    duration_valid = valid.all(axis=-1)
    safe = valid & (c > safety_threshold_px)
    best = np.max(np.where(safe, np.arange(3), 0), axis=-1).astype(np.int64)
    best[~duration_valid] = -1
    joint = np.where(valid, c / (1 + duration_discount * (DURATIONS - 1)), np.nan)
    flat = np.where(valid, joint, -np.inf).reshape(len(c), 54)
    joint_class = flat.argmax(axis=-1) if len(c) else np.empty(0, dtype=np.int64)
    joint_valid = valid.any(axis=(1, 2))
    joint_class[~joint_valid] = -1
    return dict(joint_target=joint, joint_class=joint_class, joint_valid=joint_valid,
                valid_mask=valid, best_duration=best, duration_valid=duration_valid,
                risk_dangerous=duration_valid & ~safe.any(axis=-1),
                duration_class_counts=np.bincount(best[duration_valid], minlength=3),
                semantics='linear existing-pool model proxy only; world risk unknown')


def evaluate_joint(clearance, candidate_pred, duration_pred):
    """Paired proxy metrics versus same-candidate fixed-duration3 baseline.

    Predicted durations are *classes* 0/1/2, not decision counts. NaN cells are
    excluded using a common full-row mask for paired comparisons. Regret compares
    candidate choices at exactly the same horizon, never credits shorter exposure.
    positive matched_horizon_regret_diff means worse than fixed3 candidate regret.
    """
    c, valid = _proxy_arrays(clearance)
    n = len(c)
    candidate, duration = np.asarray(candidate_pred), np.asarray(duration_pred)
    if (candidate.shape != (n,) or duration.shape != (n,)
            or candidate.dtype.kind not in 'iu' or duration.dtype.kind not in 'iu'
            or np.any((candidate < 0) | (candidate >= 18))
            or np.any((duration < 0) | (duration >= 3))):
        raise ValueError('predictions must be integer [N] candidate and duration classes')
    mask = valid.all(axis=(1, 2))
    x, a, d = c[mask], candidate[mask], duration[mask]
    k = np.arange(len(x))
    chosen, fixed = x[k, a, d], x[k, a, 1]
    regret = x[k, :, d].max(axis=1) - chosen
    fixed_regret = x[:, :, 1].max(axis=1) - fixed
    def mean(v):
        return float(np.mean(v)) if len(v) else None
    def metrics(v, durations):
        return dict(mean_clearance=mean(v), collision_rate=mean(v <= 0),
                    short_duration_rate=mean(durations == 0), mean_horizon=mean(DURATIONS[durations]))
    matched = {}
    for cls, horizon in enumerate(DURATIONS):
        selected = d == cls
        matched[int(horizon)] = dict(count=int(selected.sum()),
            mean_clearance=mean(chosen[selected]), fixed3_mean_clearance=mean(fixed[selected]),
            matched_horizon_regret=mean(regret[selected]),
            fixed3_regret=mean(fixed_regret[selected]),
            matched_horizon_regret_diff=mean((regret-fixed_regret)[selected]))
    return dict(valid_mask=mask, count=int(mask.sum()), chosen_chunk_clearance=chosen,
                chosen=metrics(chosen, d), fixed_duration3=metrics(fixed, np.ones(len(x), dtype=int)),
                duration_class_counts=np.bincount(d, minlength=3), duration_matching=matched,
                matched_horizon_regret=mean(regret), fixed3_regret=mean(fixed_regret),
                matched_horizon_regret_diff=mean(regret-fixed_regret),
                semantics='paired model proxy only; shorter alone is not improvement')
