"""Evaluation metrics for the vision student, in pure numpy.

Why this is its own module
--------------------------
The metrics are the only part of training that a human reads as a verdict, so
they must be runnable and unit-testable without a GPU, a model, or an image
decoder.  Importing :mod:`touhou_rl.vision_train` would drag in torch; importing
PIL would drag in a decode stack.  Neither happens here.

Calibration against v1.0.2
--------------------------
Every metric that v1.0.2 reported keeps its exact v1 formula so old numbers stay
comparable, with two adjudicated fixes:

* ``bullet_count_mae`` reads index 14 (``active_count / 700``).  v1 read index
  13, which is ``stage_clear_flag`` and is constant 1.0 on all 72,472 recorded
  frames - the old "15.1 bullets of error" was measuring a dead channel.
* ``danger_metrics`` adds per-frame Pearson next to the flattened one.  A trivial
  predictor that emits only the per-frame bullet total already scores ~0.557 on
  the flattened correlation, so the flattened number cannot decide whether the
  model localises bullets; A-01 reads the per-frame median instead.

Conventions
-----------
* all float outputs are rounded (4 decimals, except ``regret_px``/
  ``regret_p90_px``/3 and ``bullet_count_mae``/1) and are never ``NaN``: a
  degenerate but finite input yields ``0.0`` so a broken run cannot silently
  pass a ``nan`` comparison;
* a *non-finite* prediction or label is a hard failure and raises
  ``ValueError("v104:DATA: ...")``.  v1.0.3 relied on the ``_round`` map to 0.0
  above, which turns ``nan`` position error into "0.0 px" - a perfect score.
  Nothing here may report a number for a non-finite input;
* ``N == 0`` is a valid input and yields zeros, never an exception;
* v1.0.4 keeps every v1 key and value, and adds the ranking diagnostics
  ``top3``/``top3_tie``/``near_optimal_05``/``regret_p90_px`` to
  :func:`policy_metrics` (globally and per danger bucket).  ``regret`` is always
  the raw, unclipped teacher clearance in pixels.
"""

from __future__ import annotations

import numpy as np

from .spec import PLAYFIELD_H, PLAYFIELD_W

BUCKETS = ((0.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, 120.0), (120.0, float("inf")))
BUSY_THRESHOLD = 0.2
DANGER_SCALE = 8.0
N_CANDIDATES = 18
TOP_K = 3
NEAR_OPTIMAL_PX = 0.5
REGRET_P90 = 90.0


def _round(value: float, digits: int) -> float:
    """Round for reporting, mapping every non-finite value to 0.0."""
    if not np.isfinite(value):
        return 0.0
    return round(float(value), digits)


def _require_finite(name: str, value: np.ndarray) -> None:
    """Reject a non-finite prediction or label instead of reporting a number.

    ``_round`` maps non-finite to 0.0, which is the right answer for a
    *degenerate* (constant) input and a catastrophically wrong one for a ``nan``
    prediction: 0.0 px of position error would pass A-01.  Callers that feed
    model output here must see the failure, not a zero.
    """
    if value.size and not bool(np.isfinite(value).all()):
        raise ValueError(f"v104:DATA: {name} contains non-finite values")


def _as_2d(array: np.ndarray | list, width: int | None = None) -> np.ndarray:
    """``[N, ...]`` -> ``[N, width]`` without tripping over ``N == 0``."""
    out = np.asarray(array, dtype=np.float64)
    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
        out = out.reshape(1, -1) if out.size else out.reshape(0, width or 0)
    if width is not None:
        return out.reshape(out.shape[0], width)
    if out.shape[0] == 0:
        return out
    return out.reshape(out.shape[0], -1)


def player_position_mae_px(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Player position error in game pixels, ``(mean, p95)``.

    Single-frame error is ``(|dx| * 384 + |dy| * 448) / 2`` - the v1 definition.
    Deliberately not the Euclidean norm: the Euclidean form inflates every value
    by 1.0-1.4x and would make the 70.97 px v1.0.2 baseline non-comparable.
    """
    predicted = _as_2d(pred)
    actual = _as_2d(truth)
    _require_finite("pred_self", predicted)
    _require_finite("truth_self", actual)
    if predicted.shape[0] == 0:
        return (0.0, 0.0)
    error = (
        np.abs(predicted[:, 0] - actual[:, 0]) * PLAYFIELD_W
        + np.abs(predicted[:, 1] - actual[:, 1]) * PLAYFIELD_H
    ) / 2.0
    return (_round(float(error.mean()), 4), _round(float(np.percentile(error, 95)), 4))


def _pearson(flat_pred: np.ndarray, flat_true: np.ndarray) -> float:
    if flat_pred.size <= 1:
        return 0.0
    if flat_pred.std() == 0.0 or flat_true.std() == 0.0:
        return 0.0
    value = float(np.corrcoef(flat_pred, flat_true)[0, 1])
    return value if np.isfinite(value) else 0.0


def danger_metrics(pred_log: np.ndarray, truth: np.ndarray) -> dict:
    """Danger-map metrics, global and per-frame.

    ``pred_log`` is the model's ``log1p(cells / 8)`` output; ``truth`` is the raw
    uint8 per-cell bullet count.  Both the flattened Pearson (v1, kept for
    comparability) and the per-frame Pearson (the A-01 criterion) are reported:
    the flattened form is dominated by frames with many bullets, so it rewards a
    predictor that only knows "this frame is busy".
    """
    predicted = _as_2d(pred_log)
    target = _as_2d(truth)
    _require_finite("pred_danger_log", predicted)
    _require_finite("truth_danger", target)
    if predicted.shape[0] == 0 or target.shape[0] == 0:
        return {
            "danger_mae_log": 0.0,
            "danger_corr": 0.0,
            "danger_corr_perframe_mean": 0.0,
            "danger_corr_perframe_median": 0.0,
            "danger_corr_perframe_valid": 0,
            "danger_iou": 0.0,
            "danger_cell_acc": 0.0,
        }
    if predicted.shape[0] != target.shape[0]:
        raise ValueError(
            f"danger_metrics: {predicted.shape[0]} predictions vs {target.shape[0]} targets"
        )
    truth_log = np.log1p(target / DANGER_SCALE)

    busy_true = truth_log > BUSY_THRESHOLD
    busy_pred = predicted > BUSY_THRESHOLD
    union = (busy_true | busy_pred).sum(axis=1)
    intersection = (busy_true & busy_pred).sum(axis=1)
    busy_frames = union > 0
    iou = (
        float(np.mean(intersection[busy_frames] / union[busy_frames]))
        if bool(busy_frames.any())
        else 0.0
    )

    # Per-frame Pearson: a frame whose prediction or target is constant across the
    # 128 cells carries no spatial information, so it is dropped rather than
    # filled with 0.0 (filling would drag the median toward "no correlation").
    centred_pred = predicted - predicted.mean(axis=1, keepdims=True)
    centred_true = truth_log - truth_log.mean(axis=1, keepdims=True)
    norm_pred = np.sqrt((centred_pred * centred_pred).sum(axis=1))
    norm_true = np.sqrt((centred_true * centred_true).sum(axis=1))
    finite = np.isfinite(predicted).all(axis=1) & np.isfinite(truth_log).all(axis=1)
    valid = finite & (norm_pred > 0.0) & (norm_true > 0.0)
    per_frame = np.empty(int(valid.sum()), dtype=np.float64)
    if per_frame.size:
        rows = np.flatnonzero(valid)
        per_frame = (centred_pred[rows] * centred_true[rows]).sum(axis=1) / (
            norm_pred[rows] * norm_true[rows]
        )
        per_frame = np.clip(per_frame, -1.0, 1.0)

    return {
        "danger_mae_log": _round(float(np.abs(predicted - truth_log).mean()), 4),
        "danger_corr": _round(_pearson(predicted.ravel(), truth_log.ravel()), 4),
        "danger_corr_perframe_mean": _round(float(per_frame.mean()) if per_frame.size else 0.0, 4),
        "danger_corr_perframe_median": _round(
            float(np.median(per_frame)) if per_frame.size else 0.0, 4
        ),
        "danger_corr_perframe_valid": int(per_frame.size),
        "danger_iou": _round(iou, 4),
        "danger_cell_acc": _round(float(np.mean(busy_true == busy_pred)), 4),
    }


def bullet_count_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute bullet-count error in bullets (index 14, ``active_count/700``).

    Index 13 is ``stage_clear_flag`` and is 1.0 on every recorded frame; reading it
    produced v1's phantom "15.1 bullets" error.
    """
    predicted = _as_2d(pred, 24)
    actual = _as_2d(truth, 24)
    _require_finite("pred_self", predicted)
    _require_finite("truth_self", actual)
    if predicted.shape[0] == 0:
        return 0.0
    return _round(float(np.abs(predicted[:, 14] - actual[:, 14]).mean()) * 700.0, 1)


def ray_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute error of the eight ray distances (indices 16:24)."""
    predicted = _as_2d(pred, 24)
    actual = _as_2d(truth, 24)
    _require_finite("pred_self", predicted)
    _require_finite("truth_self", actual)
    if predicted.shape[0] == 0:
        return 0.0
    return _round(float(np.abs(predicted[:, 16:24] - actual[:, 16:24]).mean()), 4)


def self_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute error over all 24 self-feature dimensions (v1 definition)."""
    predicted = _as_2d(pred, 24)
    actual = _as_2d(truth, 24)
    _require_finite("pred_self", predicted)
    _require_finite("truth_self", actual)
    if predicted.shape[0] == 0:
        return 0.0
    return _round(float(np.abs(predicted - actual).mean()), 4)


def _bucket_label(low: float, high: float) -> str:
    return f"{int(low)}-{int(high)}" if np.isfinite(high) else f"{int(low)}+"


def _nearest_bullet_px(bullets: np.ndarray) -> np.ndarray:
    """Nearest valid bullet distance in game pixels, ``inf`` when there is none.

    ``dx``/``dy`` are the normalised offsets of :mod:`touhou_rl.obs` scaled back
    to the 384x448 playfield.  The distance is computed in float64: in float32 a
    bullet at exactly 15 px can round to 14.9999995 and land in the 0-15 bucket,
    which is the bucket A-02 is judged on.
    """
    frames = bullets.shape[0]
    if bullets.shape[1] == 0:
        return np.full(frames, np.inf, dtype=np.float64)
    dx = bullets[:, :, 0].astype(np.float64) * 192.0
    dy = bullets[:, :, 1].astype(np.float64) * 224.0
    valid = bullets[:, :, 5] > 0.5
    with np.errstate(invalid="ignore"):
        distance = np.sqrt(dx * dx + dy * dy)
    # Invalid slots become inf so the row minimum only sees real bullets; a frame
    # with no valid slot therefore keeps inf and lands in the 120+ bucket.
    return np.where(valid, distance, np.inf).min(axis=1)


def _empty_policy_bucket(low: float, high: float) -> dict:
    return {
        "bucket": _bucket_label(low, high),
        "frames": 0,
        "top1": 0.0,
        "top3": 0.0,
        "top3_tie": 0.0,
        "near_optimal_05": 0.0,
        "regret_px": 0.0,
        "regret_p90_px": 0.0,
    }


def _policy_aggregate(index: np.ndarray, hits: dict, regret: np.ndarray) -> dict:
    """Hit rates and regret statistics over the frames selected by ``index``."""
    if index.size == 0:
        return {
            "frames": 0,
            "top1": 0.0,
            "top3": 0.0,
            "top3_tie": 0.0,
            "near_optimal_05": 0.0,
            "regret_px": 0.0,
            "regret_p90_px": 0.0,
        }
    return {
        "frames": int(index.size),
        "top1": _round(float(hits["top1"][index].mean()), 4),
        "top3": _round(float(hits["top3"][index].mean()), 4),
        "top3_tie": _round(float(hits["top3_tie"][index].mean()), 4),
        "near_optimal_05": _round(float(hits["near_optimal_05"][index].mean()), 4),
        "regret_px": _round(float(regret[index].mean()), 3),
        "regret_p90_px": _round(float(np.percentile(regret[index], REGRET_P90)), 3),
    }


def policy_metrics(pred: np.ndarray, teacher: np.ndarray, bullets: np.ndarray) -> dict:
    """Teacher-agreement metrics, overall and split by current danger.

    ``top1`` is how often the model's preferred candidate is the teacher's
    favourite; ``regret_px`` is how many pixels of raw (unclipped) teacher
    clearance that mistake costs.  v1.0.4 adds the diagnostics A-02 reports next
    to the 0-15 px regret:

    * ``top3`` - the student's argmax is inside the teacher's strict top three,
      ordered by ``(-teacher_value, candidate_index)`` (stable, no 0.5 slack);
    * ``top3_tie`` - the same, but the set is every candidate whose raw teacher
      value is at least the third-best value.  It is the honest ceiling when the
      teacher cannot separate places, so ``top3_tie >= top3`` always holds;
    * ``near_optimal_05`` - ``max(v) - v[argmax(s)] <= 0.5`` px: the choice is
      effectively optimal even when it is not the teacher's first pick;
    * ``regret_p90_px`` - the 90th percentile of the per-frame regret, so a tail
      of catastrophic frames cannot hide behind the mean.

    ``chosen`` is the student's row argmax; ``regret`` always uses the raw
    teacher px (never the ``[-8, 40]`` training clip).  The danger split matters
    because an average hides the only region a policy actually has to get right:
    the 0-15 px bucket.
    """
    predicted = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(teacher, dtype=np.float64)
    shot = np.asarray(bullets, dtype=np.float32)
    if predicted.ndim != 2 or predicted.shape[1] != N_CANDIDATES:
        raise ValueError(
            f"v104:DATA: pred must be [N, {N_CANDIDATES}], got shape {predicted.shape}"
        )
    if actual.ndim != 2 or actual.shape[1] != N_CANDIDATES:
        raise ValueError(
            f"v104:DATA: teacher must be [N, {N_CANDIDATES}], got shape {actual.shape}"
        )
    if shot.ndim != 3 or shot.shape[2] != 6:
        raise ValueError(f"v104:DATA: bullets must be [N, B, 6], got shape {shot.shape}")
    frames = predicted.shape[0]
    if actual.shape[0] != frames:
        raise ValueError(
            f"v104:DATA: {frames} predictions vs {actual.shape[0]} teachers"
        )
    if shot.shape[0] != frames:
        raise ValueError(f"v104:DATA: {frames} predictions vs {shot.shape[0]} bullet frames")
    empty = {
        "top1": 0.0,
        "top3": 0.0,
        "top3_tie": 0.0,
        "near_optimal_05": 0.0,
        "regret_px": 0.0,
        "regret_p90_px": 0.0,
        "by_danger": [_empty_policy_bucket(low, high) for low, high in BUCKETS],
    }
    if frames == 0:
        return empty
    _require_finite("pred", predicted)
    _require_finite("teacher", actual)
    _require_finite("bullets", shot)

    chosen = predicted.argmax(axis=1)
    rows = np.arange(frames)
    order = np.argsort(-actual, axis=1, kind="stable")
    third_best = np.sort(actual, axis=1)[:, -TOP_K]
    tie_mask = actual >= third_best[:, None]
    hits = {
        "top1": chosen == actual.argmax(axis=1),
        "top3": (order[:, :TOP_K] == chosen[:, None]).any(axis=1),
        "top3_tie": np.take_along_axis(tie_mask, chosen[:, None], axis=1)[:, 0],
        "near_optimal_05": (actual.max(axis=1) - actual[rows, chosen]) <= NEAR_OPTIMAL_PX,
    }
    regret = actual.max(axis=1) - actual[rows, chosen]
    danger = _nearest_bullet_px(shot)

    overall = _policy_aggregate(rows, hits, regret)
    overall.pop("frames")
    buckets = []
    for low, high in BUCKETS:
        # ``high == inf`` must mean "no upper bound": a frame whose 64 bullet slots
        # are all invalid has distance inf and belongs in the 120+ bucket.  Using
        # ``danger < high`` there would silently drop those frames from every bucket.
        if np.isfinite(high):
            mask = (danger >= low) & (danger < high)
        else:
            mask = danger >= low
        entry = {"bucket": _bucket_label(low, high)}
        entry.update(_policy_aggregate(np.flatnonzero(mask), hits, regret))
        buckets.append(entry)
    overall["by_danger"] = buckets
    return overall


def heatmap_hit_rate(logits: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of frames whose heatmap argmax lands within 1 cell of the truth.

    ``logits`` is ``[N, H, W]`` (rows = height, cols = width; 28x24 for the v2
    head, 56x48 for the v3-st head) and ``truth`` is the normalised
    ``self_feat``.  The grid is read from the array, never assumed: cell
    coordinates follow the same convention the target rasterisation uses,
    ``x * W - 0.5`` / ``y * H - 0.5``, rounded and clipped into the grid.  The
    v1.0.3 code hard-coded 24 columns / 28 rows, which silently scored a 56x48
    head against a grid half its size (every frame missed by up to 2x the
    intended tolerance).

    Diagnostic only - it is a coarse "did the head look at the right place"
    check, not a substitute for ``player_pos_px``.
    """
    scores = np.asarray(logits, dtype=np.float64)
    actual = np.asarray(truth, dtype=np.float64)
    if scores.ndim != 3:
        raise ValueError(f"v104:DATA: logits must be [N, H, W], got shape {scores.shape}")
    if actual.ndim != 2 or actual.shape[1] < 2:
        raise ValueError(f"v104:DATA: truth must be [N, >=2], got shape {actual.shape}")
    frames, height, width = scores.shape
    if height < 1 or width < 1:
        raise ValueError(f"v104:DATA: logits grid must be at least 1x1, got {height}x{width}")
    if frames == 0 or actual.shape[0] == 0:
        return 0.0
    if actual.shape[0] != frames:
        raise ValueError(f"v104:DATA: {frames} logits vs {actual.shape[0]} truth rows")
    _require_finite("heatmap logits", scores)
    _require_finite("heatmap truth", actual[:, :2])
    grid = scores.reshape(frames, -1).argmax(axis=1)
    hit_rows = grid // width
    hit_cols = grid % width
    want_x = np.clip(np.rint(actual[:, 0] * width - 0.5), 0, width - 1).astype(np.int64)
    want_y = np.clip(np.rint(actual[:, 1] * height - 0.5), 0, height - 1).astype(np.int64)
    hit = (np.abs(hit_rows - want_y) <= 1) & (np.abs(hit_cols - want_x) <= 1)
    return _round(float(hit.mean()), 4)


def all_state_metrics(
    pred_self: np.ndarray,
    pred_danger_log: np.ndarray,
    truth_self: np.ndarray,
    truth_danger: np.ndarray,
    *,
    heatmap_logits: np.ndarray | None = None,
) -> dict:
    """Every state-stage metric, keyed without the ``val_`` prefix.

    S3-3 adds ``val_`` and writes the result into ``vision-summary.json``; the keys
    here must stay in one-to-one correspondence with the summary schema.
    """
    mean_px, p95_px = player_position_mae_px(pred_self, truth_self)
    metrics = {
        "self_mae": self_mae(pred_self, truth_self),
        "player_pos_px": mean_px,
        "player_pos_p95_px": p95_px,
        "ray_mae": ray_mae(pred_self, truth_self),
        "bullet_count_mae": bullet_count_mae(pred_self, truth_self),
    }
    metrics.update(danger_metrics(pred_danger_log, truth_danger))
    if heatmap_logits is not None:
        metrics["heatmap_hit_rate"] = heatmap_hit_rate(heatmap_logits, truth_self)
    return metrics
