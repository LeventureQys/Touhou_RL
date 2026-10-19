"""Teacher-preference losses for the v1.0.4 policy head, in pure torch.

Why this is its own module
--------------------------
The v1.0.3 policy loss lives inline in :mod:`touhou_rl.vision_train`
(``F.smooth_l1_loss(predicted, clip(value, -8, 40) / 10)``).  Two defects make
that loss unfit for the v1.0.4 ranking experiment:

* the absolute teacher scale drifts between frames (one frame's 12 px is another
  frame's ceiling), so a regression loss spends capacity on reproducing the
  frame's offset instead of its ordering, and the ``[-8, 40]`` clip throws away
  the extreme frames that carry the safest/unsafest decisions;
* nothing in it prefers the teacher's best candidate over its worst - a
  prediction that is 5 px too high on every candidate scores exactly as well as
  one that is 5 px too high only on the teacher's favourite.

The three objectives here are the ones the experiment matrix (E0/E1/E2/E3a-c)
compares:

``legacy-reg``
    ``smooth_l1(s, clip(v, -8, 40) / 10).mean`` - the v1.0.3 objective, kept
    bit-identical so E0/E1 stay comparable with the old runs.
``centered-reg``
    ``smooth_l1(s - mean(s), (v - mean(v)) / 10).mean`` - the same shape of loss
    with the per-frame offset removed and no clipping (the clip is what the
    centre is meant to replace, not to duplicate).
``rank``
    every ``i < j`` pair whose teacher values differ by more than
    ``rank_delta_px`` contributes ``softplus(-sign(v_i - v_j) * (s_i - s_j))``.
    The logistic form is the pairwise logistic (RankNet) loss: it is zero for a
    correct large margin and grows linearly for a wrong large margin, which is
    exactly "prefer the teacher's choice" rather than "match its number".
    Optionally mixed with ``centered-reg`` via ``regression_weight``.

Accounting
----------
Teacher values arrive as ``float16`` (the pilot's choice, kept for this round);
pairs whose difference collapses to zero under that quantisation are *excluded*
rather than treated as either order - the local rank of two candidates the
label itself cannot separate is not a fact to learn.  The reported
``valid_pairs`` / ``frames_with_pairs`` / ``frames_without_pairs`` make that
auditable, and a frame with no usable pair contributes ``scores.sum() * 0``
(not ``NaN``) so a fully tied frame can still be trained on without poisoning
the mean.

Errors
------
Every rejection is a ``ValueError`` carrying the v1.0.4 prefix that the CLI
converts into ``exit 2``: ``v104:CONFIG:`` for a bad objective/weight/delta and
``v104:DATA:`` for a bad tensor (shape, dtype, device, non-finite value).  A
misconfigured weight must never silently produce a zero loss.
"""

from __future__ import annotations

import math
import numbers

import torch
import torch.nn.functional as F

N_CANDIDATES = 18
N_PAIRS = N_CANDIDATES * (N_CANDIDATES - 1) // 2  # 153
OBJECTIVES = ("legacy-reg", "centered-reg", "rank")
VALUE_CLIP = (-8.0, 40.0)
VALUE_SCALE = 10.0
DEFAULT_RANK_DELTA_PX = 0.5

# Pair indices are device constants; rebuilding them per batch would add an
# allocation and a host->device copy to every training step.
_PAIR_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _pairs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(row, col)`` index tensors for the 153 upper-triangular pairs."""
    key = str(device)
    cached = _PAIR_CACHE.get(key)
    if cached is None:
        rows, cols = torch.triu_indices(N_CANDIDATES, N_CANDIDATES, offset=1, device=device)
        cached = (rows, cols)
        _PAIR_CACHE[key] = cached
    return cached


def _config_error(detail: str) -> ValueError:
    return ValueError(f"v104:CONFIG: {detail}")


def _data_error(detail: str) -> ValueError:
    return ValueError(f"v104:DATA: {detail}")


def _number(name: str, value) -> float:
    """Accept a real scalar (python or numpy) and return it as a finite float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise _config_error(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise _config_error(f"{name} must be finite, got {value!r}")
    return out


def _check_config(
    objective: str, rank_delta_px, rank_weight, regression_weight
) -> tuple[str, float, float, float]:
    if not isinstance(objective, str) or objective not in OBJECTIVES:
        raise _config_error(f"objective must be one of {list(OBJECTIVES)}, got {objective!r}")
    delta = _number("rank_delta_px", rank_delta_px)
    rank_weight = _number("rank_weight", rank_weight)
    regression_weight = _number("regression_weight", regression_weight)
    if delta < 0.0:
        raise _config_error(f"rank_delta_px must be >= 0, got {delta}")
    if rank_weight < 0.0:
        raise _config_error(f"rank_weight must be >= 0, got {rank_weight}")
    if regression_weight < 0.0:
        raise _config_error(f"regression_weight must be >= 0, got {regression_weight}")
    if rank_weight + regression_weight <= 0.0:
        raise _config_error(
            f"total weight must be > 0, got rank_weight={rank_weight} "
            f"and regression_weight={regression_weight}"
        )
    if objective != "rank":
        # A regression objective with a rank weight would silently train a
        # different thing than its name says; the design fixes the convention.
        if rank_weight != 0.0:
            raise _config_error(f"{objective} requires rank_weight=0, got {rank_weight}")
        if regression_weight != 1.0:
            raise _config_error(
                f"{objective} requires regression_weight=1, got {regression_weight}"
            )
    return objective, delta, rank_weight, regression_weight


def _check_scores(scores) -> None:
    if not isinstance(scores, torch.Tensor):
        raise _data_error(f"scores must be a torch.Tensor, got {type(scores).__name__}")
    if scores.dim() != 2 or scores.shape[1] != N_CANDIDATES or scores.shape[0] < 1:
        raise _data_error(
            f"scores must be [{N_CANDIDATES} candidates] with at least one frame, "
            f"got shape {tuple(scores.shape)}"
        )
    if scores.dtype != torch.float32:
        raise _data_error(f"scores must be torch.float32, got {scores.dtype}")
    if not bool(torch.isfinite(scores).all()):
        raise _data_error("scores contain non-finite values")


def _check_teacher(teacher_px, scores: torch.Tensor) -> torch.Tensor:
    if not isinstance(teacher_px, torch.Tensor):
        raise _data_error(f"teacher_px must be a torch.Tensor, got {type(teacher_px).__name__}")
    if tuple(teacher_px.shape) != tuple(scores.shape):
        raise _data_error(
            f"teacher_px must have shape {tuple(scores.shape)}, got {tuple(teacher_px.shape)}"
        )
    if not teacher_px.is_floating_point():
        raise _data_error(f"teacher_px must be a floating point tensor, got {teacher_px.dtype}")
    if teacher_px.device != scores.device:
        raise _data_error(
            f"teacher_px is on {teacher_px.device} but scores are on {scores.device}"
        )
    if not bool(torch.isfinite(teacher_px).all()):
        raise _data_error("teacher_px contains non-finite values")
    # float16 teacher labels are the recorded precision; upcasting keeps their
    # quantisation (ties stay ties) while giving the pair arithmetic fp32.
    return teacher_px.to(torch.float32)


def _legacy_regression(scores: torch.Tensor, teacher_px: torch.Tensor) -> torch.Tensor:
    target = torch.clamp(teacher_px, *VALUE_CLIP) / VALUE_SCALE
    return F.smooth_l1_loss(scores, target)


def _centered_regression(scores: torch.Tensor, teacher_px: torch.Tensor) -> torch.Tensor:
    # keepdim=True keeps [B, 1] so the subtraction broadcasts over candidates;
    # a [B] tensor would align with the *frame* axis and silently broadcast as
    # a [B, B] difference matrix.
    centred_scores = scores - scores.mean(dim=1, keepdim=True)
    centred_teacher = (teacher_px - teacher_px.mean(dim=1, keepdim=True)) / VALUE_SCALE
    return F.smooth_l1_loss(centred_scores, centred_teacher)


def _rank_loss(
    scores: torch.Tensor, teacher_px: torch.Tensor, delta: float
) -> tuple[torch.Tensor, int, int, int]:
    """Pairwise logistic loss, averaged within a frame and then over frames."""
    rows, cols = _pairs(scores.device)
    score_gap = scores.index_select(1, rows) - scores.index_select(1, cols)
    teacher_gap = teacher_px.index_select(1, rows) - teacher_px.index_select(1, cols)
    valid = teacher_gap.abs() > delta
    pair_loss = F.softplus(-torch.sign(teacher_gap) * score_gap)
    counts = valid.sum(dim=1)
    per_frame = (pair_loss * valid).sum(dim=1) / counts.clamp(min=1).to(scores.dtype)
    # A frame whose teacher values are all tied has no opinion; it must add 0.0
    # without breaking the graph (backward still has to reach every parameter).
    placeholder = scores.sum(dim=1) * 0.0
    rank_loss = torch.where(counts > 0, per_frame, placeholder).mean()
    return (
        rank_loss,
        int(counts.sum().item()),
        int((counts > 0).sum().item()),
        int((counts == 0).sum().item()),
    )


def policy_loss(
    scores: torch.Tensor,
    teacher_px: torch.Tensor,
    objective: str,
    rank_delta_px: float = DEFAULT_RANK_DELTA_PX,
    rank_weight: float = 1.0,
    regression_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Teacher-preference loss for one batch of candidate scores.

    ``scores`` is the student's raw preference (float32 ``[B, 18]``, order
    ``focus * 9 + move``); ``teacher_px`` is :class:`pilot.AvoidPilot`'s raw
    clearance in playfield pixels, *before* any clipping, for the same 18
    candidates.  Returns ``(loss, stats)`` where ``loss`` is a differentiable
    0-dim tensor and ``stats`` carries the diagnostics the summary records:

    ======================  ==================================================
    ``objective``           the objective string that was used
    ``rank_loss``           frame-mean of the per-frame pair mean (0 if unused)
    ``regression_loss``     the regression term (legacy or centred)
    ``valid_pairs``         pairs with ``abs(v_i - v_j) > rank_delta_px``
    ``frames_with_pairs``   frames that contributed at least one pair
    ``frames_without_pairs`` frames that fell back to the zero placeholder
    ======================  ==================================================

    For ``rank`` the total is ``rank_weight * rank_loss +
    regression_weight * regression_loss``; the regression term is always
    computed so its value is reportable even at weight 0 (it does not change
    the loss or the gradient when the weight is 0).
    """
    objective, delta, rank_weight, regression_weight = _check_config(
        objective, rank_delta_px, rank_weight, regression_weight
    )
    _check_scores(scores)
    teacher = _check_teacher(teacher_px, scores)
    frames = int(scores.shape[0])

    if objective == "rank":
        rank_loss, valid_pairs, frames_with, frames_without = _rank_loss(scores, teacher, delta)
        regression_loss = _centered_regression(scores, teacher)
        loss = rank_weight * rank_loss + regression_weight * regression_loss
    else:
        regression_loss = (
            _legacy_regression(scores, teacher)
            if objective == "legacy-reg"
            else _centered_regression(scores, teacher)
        )
        rank_loss = None
        valid_pairs, frames_with, frames_without = 0, 0, frames
        loss = regression_weight * regression_loss

    stats = {
        "objective": objective,
        "rank_loss": 0.0 if rank_loss is None else float(rank_loss.detach()),
        "regression_loss": float(regression_loss.detach()),
        "valid_pairs": int(valid_pairs),
        "frames_with_pairs": int(frames_with),
        "frames_without_pairs": int(frames_without),
    }
    return loss, stats
