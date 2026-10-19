"""Unit tests for the v1.0.4 teacher-preference losses.

Each objective is pinned on the property an experiment cannot survive losing:

* ``legacy-reg`` must stay bit-identical to the v1.0.3 in-training loss
  (``smooth_l1(score, clip(value, -8, 40) / 10).mean``), otherwise E0/E1 stop
  being comparable with the old runs;
* ``centered-reg`` must ignore a common per-frame offset (that is its whole
  point), must not clip, and must centre with ``keepdim=True`` - a ``[B]`` mean
  silently broadcasts against the candidate axis and produces a different loss;
* ``rank`` must carry gradient across candidate groups (``focus`` blocks), must
  average within a frame *before* averaging over the batch so a busy frame does
  not outvote a quiet one, must exclude a pair at exactly ``rank_delta_px`` and
  pairs that float16 quantisation collapsed, and must return a finite zero (with
  a working backward) for a fully tied frame;
* every rejected configuration raises ``v104:CONFIG`` / ``v104:DATA``: a silent
  zero loss would look like a converged run.

Everything here is CPU-only: no CUDA, no dataset, no game window.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from touhou_rl import vision_losses as vl

PAIRS = [
    (i, j) for i in range(vl.N_CANDIDATES) for j in range(i + 1, vl.N_CANDIDATES)
]
LEGACY_KWARGS = {"rank_weight": 0.0, "regression_weight": 1.0}


# -- reference implementations (deliberately naive) -----------------------
def _smooth_l1_mean(delta: torch.Tensor) -> torch.Tensor:
    absolute = delta.abs()
    return torch.where(absolute < 1.0, 0.5 * delta * delta, absolute - 0.5).mean()


def _manual_legacy(scores: torch.Tensor, teacher: torch.Tensor) -> float:
    target = torch.clamp(teacher, *vl.VALUE_CLIP) / vl.VALUE_SCALE
    return float(_smooth_l1_mean(scores - target))


def _manual_centered(scores: torch.Tensor, teacher: torch.Tensor) -> float:
    centred_scores = scores - scores.mean(dim=1, keepdim=True)
    centred_teacher = (teacher - teacher.mean(dim=1, keepdim=True)) / vl.VALUE_SCALE
    return float(_smooth_l1_mean(centred_scores - centred_teacher))


def _softplus(value: float) -> float:
    """Stable softplus, matching ``torch.nn.functional.softplus``."""
    if value > 20.0:
        return value
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _frame_rank(scores: torch.Tensor, teacher: torch.Tensor, row: int, delta: float) -> float:
    terms = []
    for i, j in PAIRS:
        gap = float(teacher[row, i] - teacher[row, j])
        if abs(gap) > delta:
            sign = math.copysign(1.0, gap)
            terms.append(_softplus(-sign * float(scores[row, i] - scores[row, j])))
    return sum(terms) / len(terms) if terms else 0.0


def _manual_rank(scores: torch.Tensor, teacher: torch.Tensor, delta: float) -> float:
    """Per-frame mean of the valid pair losses, then the batch mean."""
    frames = scores.shape[0]
    return sum(_frame_rank(scores, teacher, row, delta) for row in range(frames)) / frames


def _scores(frames: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(frames, vl.N_CANDIDATES, generator=generator)


def _value(loss: torch.Tensor) -> float:
    """Scalar value of a loss tensor (detached: conversion warns otherwise)."""
    return float(loss.detach())


def _teacher_px(frames: int, seed: int = 1, scale: float = 12.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(frames, vl.N_CANDIDATES, generator=generator) * scale


# -- legacy-reg -----------------------------------------------------------
def test_pair_count_is_the_full_153_candidate_pairs():
    assert vl.N_PAIRS == 153
    assert len(PAIRS) == 153
    assert vl.N_CANDIDATES == 18
    assert vl.OBJECTIVES == ("legacy-reg", "centered-reg", "rank")


def test_legacy_regression_is_the_v1_smooth_l1():
    scores = _scores(4, seed=7)
    teacher = _teacher_px(4, seed=8, scale=30.0)
    loss, stats = vl.policy_loss(scores, teacher, "legacy-reg", **LEGACY_KWARGS)
    v1_reference = F.smooth_l1_loss(
        scores, torch.clamp(teacher, *vl.VALUE_CLIP) / vl.VALUE_SCALE
    )
    assert _value(loss) == pytest.approx(_value(v1_reference), abs=1e-7)
    assert _value(loss) == pytest.approx(_manual_legacy(scores, teacher), abs=2e-7)
    assert stats["objective"] == "legacy-reg"
    assert stats["regression_loss"] == pytest.approx(_value(v1_reference), abs=1e-7)
    assert stats["rank_loss"] == 0.0
    assert stats["valid_pairs"] == 0
    assert stats["frames_with_pairs"] == 0
    assert stats["frames_without_pairs"] == 4


def test_legacy_regression_matches_v1_elementwise():
    """Every element must follow the v1 formula, not just the batch mean."""
    scores = _scores(3, seed=11)
    teacher = _teacher_px(3, seed=12, scale=100.0)
    target = torch.clamp(teacher, *vl.VALUE_CLIP) / vl.VALUE_SCALE
    delta = scores - target
    absolute = delta.abs()
    elementwise = torch.where(absolute < 1.0, 0.5 * delta * delta, absolute - 0.5)
    loss, _stats = vl.policy_loss(scores, teacher, "legacy-reg", **LEGACY_KWARGS)
    assert _value(loss) == pytest.approx(float(elementwise.mean()), abs=1e-7)
    # the quadratic branch must actually be exercised by this fixture
    assert bool((absolute < 1.0).any())


# -- centered-reg ---------------------------------------------------------
def test_centered_regression_removes_a_common_offset():
    scores = _scores(4, seed=21)
    teacher = _teacher_px(4, seed=22)
    base, _ = vl.policy_loss(scores, teacher, "centered-reg", **LEGACY_KWARGS)
    shifted, _ = vl.policy_loss(scores + 3.5, teacher + 3.5, "centered-reg", **LEGACY_KWARGS)
    assert _value(shifted) == pytest.approx(_value(base), abs=1e-6)
    # ... while the legacy objective does depend on that offset, so the fixture
    # cannot pass for the wrong reason.
    legacy_base, _ = vl.policy_loss(scores, teacher, "legacy-reg", **LEGACY_KWARGS)
    legacy_shifted, _ = vl.policy_loss(scores + 3.5, teacher + 3.5, "legacy-reg", **LEGACY_KWARGS)
    assert abs(_value(legacy_shifted) - _value(legacy_base)) > 1e-3


def test_centered_regression_does_not_clip_teacher_px():
    scores = torch.zeros(1, vl.N_CANDIDATES)
    teacher = torch.zeros(1, vl.N_CANDIDATES)
    scores[0, 0], scores[0, 1] = 60.0, -60.0
    teacher[0, 0], teacher[0, 1] = 1000.0, -1000.0
    loss, _stats = vl.policy_loss(scores, teacher, "centered-reg", **LEGACY_KWARGS)
    assert _value(loss) == pytest.approx(_manual_centered(scores, teacher), abs=1e-6)
    # clipping the teacher before centring (the legacy behaviour) would give
    # 55.5 per touched element instead of 39.5
    clipped = torch.clamp(teacher, *vl.VALUE_CLIP)
    wrong = _manual_centered(scores, clipped)
    assert abs(_value(loss) - wrong) > 1.0


def test_centered_regression_uses_per_frame_mean_keepdim():
    """A ``[B]`` mean would align with the candidate axis, not the frame axis."""
    scores = _scores(18, seed=31)
    teacher = _teacher_px(18, seed=32)
    loss, _stats = vl.policy_loss(scores, teacher, "centered-reg", **LEGACY_KWARGS)
    assert _value(loss) == pytest.approx(_manual_centered(scores, teacher), abs=1e-6)
    # with B == 18 the missing keepdim silently subtracts *column* means instead
    # of *row* means (with any other B it would raise), so this compares values
    broadcast = scores - scores.mean(dim=1)  # the bug: (18,18) - (18,)
    target = (teacher - teacher.mean(dim=1, keepdim=True)) / vl.VALUE_SCALE
    wrong = float(_smooth_l1_mean(broadcast - target))
    assert abs(_value(loss) - wrong) > 1e-3


# -- rank -----------------------------------------------------------------
def test_rank_gradient_crosses_candidate_groups_and_lowers_loss():
    scores = torch.zeros(1, vl.N_CANDIDATES, requires_grad=True)
    # 1.4 vs 1.0 with the other 16 candidates at 1.2: at delta 0.3 the only
    # separable pair is (2, 12), which straddles the two focus groups.
    teacher = torch.full((1, vl.N_CANDIDATES), 1.2)
    teacher[0, 2] = 1.4  # focus = 0 group
    teacher[0, 12] = 1.0  # focus = 1 group
    loss, stats = vl.policy_loss(scores, teacher, "rank", rank_delta_px=0.3)
    assert stats["valid_pairs"] == 1
    assert stats["frames_with_pairs"] == 1
    assert stats["frames_without_pairs"] == 0
    loss.backward()
    grad = scores.grad.detach()[0]
    assert float(grad[2]) < 0.0  # raising score[2] lowers the loss
    assert float(grad[12]) > 0.0  # lowering score[12] lowers the loss
    untouched = torch.ones(vl.N_CANDIDATES, dtype=torch.bool)
    untouched[[2, 12]] = False
    assert float(grad[untouched].abs().sum()) == 0.0

    before = _value(loss)
    with torch.no_grad():
        scores -= 1.0 * scores.grad
    after, _ = vl.policy_loss(scores.detach(), teacher, "rank", rank_delta_px=0.3)
    assert _value(after) < before


def test_rank_prefers_the_teacher_ordering():
    teacher = torch.zeros(1, vl.N_CANDIDATES)
    teacher[0, 0] = 2.0  # the only valid pairs are (0, j >= 1)
    correct = torch.full((1, vl.N_CANDIDATES), -1.0)
    correct[0, 0] = 1.0  # every gap is +2 -> loss = softplus(-2)
    wrong = torch.full((1, vl.N_CANDIDATES), 1.0)
    wrong[0, 0] = -1.0  # every gap is -2 -> loss = softplus(+2)
    good, stats_good = vl.policy_loss(correct, teacher, "rank")
    bad, stats_bad = vl.policy_loss(wrong, teacher, "rank")
    assert stats_good["valid_pairs"] == stats_bad["valid_pairs"] == 17
    assert _value(good) == pytest.approx(_softplus(-2.0), abs=1e-6)
    assert _value(bad) == pytest.approx(_softplus(2.0), abs=1e-6)
    assert _value(good) < _value(bad)


def test_rank_averages_within_frame_then_over_batch():
    # frame 0: one value above the rest -> 17 valid pairs
    # frame 1: values 0.0 / 1.0 / (16 x 0.5) -> exactly one valid pair
    scores = _scores(2, seed=41)
    teacher = torch.zeros(2, vl.N_CANDIDATES)
    teacher[0, 0] = 5.0
    teacher[1, 1] = 1.0
    teacher[1, 2:] = 0.5
    loss, stats = vl.policy_loss(scores, teacher, "rank")
    frame0 = _frame_rank(scores, teacher, 0, 0.5)
    frame1 = _frame_rank(scores, teacher, 1, 0.5)
    assert stats["valid_pairs"] == 18
    assert _value(loss) == pytest.approx((frame0 + frame1) / 2.0, abs=1e-6)
    # a global pair mean would weight frame 0's 17 pairs 17x more
    global_mean = (17.0 * frame0 + 1.0 * frame1) / 18.0
    assert abs(_value(loss) - global_mean) > 1e-3


def test_rank_matches_reference_on_random_labels():
    scores = _scores(5, seed=51)
    teacher = _teacher_px(5, seed=52, scale=8.0)
    for delta in (0.0, 0.5, 2.0):
        loss, stats = vl.policy_loss(scores, teacher, "rank", rank_delta_px=delta)
        assert _value(loss) == pytest.approx(_manual_rank(scores, teacher, delta), abs=2e-6)
        expected_pairs = sum(
            1
            for row in range(5)
            for i, j in PAIRS
            if abs(float(teacher[row, i] - teacher[row, j])) > delta
        )
        assert stats["valid_pairs"] == expected_pairs


def test_rank_objective_weight_combinations_ra_b_rb_rc():
    scores = _scores(3, seed=61)
    teacher = _teacher_px(3, seed=62, scale=15.0)
    rank_reference = _manual_rank(scores, teacher, 0.5)
    centered_reference = _manual_centered(scores, teacher)
    for rank_weight, regression_weight in ((1.0, 0.0), (1.0, 0.1), (1.0, 1.0)):
        loss, stats = vl.policy_loss(
            scores,
            teacher,
            "rank",
            rank_weight=rank_weight,
            regression_weight=regression_weight,
        )
        expected = rank_weight * rank_reference + regression_weight * centered_reference
        assert _value(loss) == pytest.approx(expected, abs=2e-6), (rank_weight, regression_weight)
        # the reported terms do not depend on the weights
        assert stats["rank_loss"] == pytest.approx(rank_reference, abs=1e-6)
        assert stats["regression_loss"] == pytest.approx(centered_reference, abs=1e-6)


def test_rank_all_tied_frames_are_finite_zero_with_zero_gradient():
    scores = torch.zeros(3, vl.N_CANDIDATES, requires_grad=True)
    teacher = torch.full((3, vl.N_CANDIDATES), 7.0)
    loss, stats = vl.policy_loss(scores, teacher, "rank")
    assert _value(loss) == 0.0
    assert math.isfinite(_value(loss))
    assert stats["valid_pairs"] == 0
    assert stats["frames_with_pairs"] == 0
    assert stats["frames_without_pairs"] == 3
    assert stats["rank_loss"] == 0.0
    loss.backward()
    assert scores.grad is not None
    assert bool(torch.isfinite(scores.grad).all())
    assert float(scores.grad.abs().sum()) == 0.0


def test_rank_tied_frame_inside_a_batch_contributes_zero():
    scores = torch.zeros(2, vl.N_CANDIDATES)
    teacher = torch.zeros(2, vl.N_CANDIDATES)
    teacher[0, 0] = 5.0
    teacher[1, :] = 3.0
    scores.requires_grad_(True)
    loss, stats = vl.policy_loss(scores, teacher, "rank")
    assert stats["frames_with_pairs"] == 1
    assert stats["frames_without_pairs"] == 1
    assert _value(loss) == pytest.approx(_frame_rank(scores.detach(), teacher, 0, 0.5) / 2.0, abs=1e-6)
    loss.backward()
    assert float(scores.grad[1].abs().sum()) == 0.0
    assert float(scores.grad[0, 0]) < 0.0


def test_rank_delta_boundary_is_exclusive():
    scores = torch.zeros(1, vl.N_CANDIDATES)
    teacher = torch.zeros(1, vl.N_CANDIDATES)
    teacher[0, 0] = 1.0
    teacher[0, 1] = 0.5  # exactly rank_delta_px -> not a pair
    loss, stats = vl.policy_loss(scores, teacher, "rank", rank_delta_px=0.5)
    assert stats["valid_pairs"] == 16  # only (0, j >= 2)
    assert _value(loss) == pytest.approx(math.log(2.0), abs=1e-6)
    # a hair below the difference and the pair comes in (the comparison is
    # strict, not ">= delta - epsilon")
    _loss, looser = vl.policy_loss(scores, teacher, "rank", rank_delta_px=0.49)
    assert looser["valid_pairs"] == 33


def test_rank_float16_labels_quantise_to_ties():
    """float16 labels cannot separate 40.00 and 40.01, so that pair is not a fact."""
    scores = _scores(1, seed=311)
    teacher32 = torch.zeros(1, vl.N_CANDIDATES)
    teacher32[0, 0] = 40.0
    teacher32[0, 1] = 40.01
    teacher16 = teacher32.half()
    assert float(teacher16[0, 0]) == float(teacher16[0, 1])
    loss32, stats32 = vl.policy_loss(scores, teacher32, "rank", rank_delta_px=0.0)
    loss16, stats16 = vl.policy_loss(scores, teacher16, "rank", rank_delta_px=0.0)
    assert stats32["valid_pairs"] == 33
    assert stats16["valid_pairs"] == 32  # the collapsed pair is excluded
    assert math.isfinite(_value(loss16))
    assert _value(loss16) != pytest.approx(_value(loss32), abs=1e-6)


def test_rank_single_frame_batch():
    scores = torch.zeros(1, vl.N_CANDIDATES, requires_grad=True)
    teacher = torch.zeros(1, vl.N_CANDIDATES)
    teacher[0, 0], teacher[0, 9] = 3.0, 0.0
    loss, stats = vl.policy_loss(scores, teacher, "rank")
    assert loss.dim() == 0
    assert stats["frames_with_pairs"] == 1
    assert _value(loss) == pytest.approx(math.log(2.0), abs=1e-6)
    loss.backward()
    assert float(scores.grad.abs().sum()) > 0.0


def test_rank_huge_delta_has_no_pairs():
    scores = _scores(2, seed=81)
    teacher = _teacher_px(2, seed=82)
    loss, stats = vl.policy_loss(scores, teacher, "rank", rank_delta_px=1e9)
    assert _value(loss) == 0.0
    assert math.isfinite(_value(loss))
    assert stats["valid_pairs"] == 0
    assert stats["frames_without_pairs"] == 2


# -- stats and differentiability -----------------------------------------
@pytest.mark.parametrize(
    "objective, kwargs",
    [
        ("legacy-reg", LEGACY_KWARGS),
        ("centered-reg", LEGACY_KWARGS),
        ("rank", {}),
        ("rank", {"regression_weight": 0.1}),
    ],
)
def test_policy_loss_contract_shape_and_stats(objective, kwargs):
    scores = _scores(3, seed=91).requires_grad_(True)
    teacher = _teacher_px(3, seed=92)
    loss, stats = vl.policy_loss(scores, teacher, objective, **kwargs)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.requires_grad
    assert loss.grad_fn is not None
    assert set(stats) == {
        "objective",
        "rank_loss",
        "regression_loss",
        "valid_pairs",
        "frames_with_pairs",
        "frames_without_pairs",
    }
    assert stats["objective"] == objective
    for key in ("rank_loss", "regression_loss"):
        assert isinstance(stats[key], float)
        assert math.isfinite(stats[key])
    for key in ("valid_pairs", "frames_with_pairs", "frames_without_pairs"):
        assert isinstance(stats[key], int)
        assert math.isfinite(stats[key])
    assert stats["valid_pairs"] >= 0
    assert stats["frames_with_pairs"] + stats["frames_without_pairs"] == 3
    loss.backward()
    assert scores.grad is not None
    assert bool(torch.isfinite(scores.grad).all())
    assert float(scores.grad.abs().sum()) > 0.0


# -- error contract -------------------------------------------------------
@pytest.mark.parametrize(
    "objective, kwargs",
    [
        ("bogus", {}),
        ("RANK", {}),
        ("", {}),
        ("rank", {"rank_weight": -0.1}),
        ("rank", {"regression_weight": -1.0}),
        ("rank", {"rank_weight": 0.0, "regression_weight": 0.0}),
        ("rank", {"rank_delta_px": -0.5}),
        ("rank", {"rank_delta_px": float("nan")}),
        ("rank", {"rank_delta_px": float("inf")}),
        ("rank", {"rank_weight": float("inf")}),
        ("rank", {"rank_weight": "1.0"}),
        ("legacy-reg", {"rank_weight": 1.0, "regression_weight": 0.0}),
        ("legacy-reg", {"rank_weight": 0.0, "regression_weight": 0.5}),
        ("centered-reg", {"rank_weight": 0.0, "regression_weight": 0.0}),
        ("centered-reg", {"rank_weight": 0.0}),
    ],
)
def test_policy_loss_rejects_bad_config(objective, kwargs):
    scores = torch.zeros(2, vl.N_CANDIDATES)
    teacher = torch.zeros(2, vl.N_CANDIDATES)
    with pytest.raises(ValueError, match="v104:CONFIG"):
        vl.policy_loss(scores, teacher, objective, **kwargs)


@pytest.mark.parametrize(
    "scores, teacher",
    [
        (np.zeros((2, vl.N_CANDIDATES), dtype=np.float32), None),
        (torch.zeros(vl.N_CANDIDATES), None),
        (torch.zeros(2, vl.N_CANDIDATES - 1), None),
        (torch.zeros(2, vl.N_CANDIDATES + 1), None),
        (torch.zeros(0, vl.N_CANDIDATES), None),
        (torch.zeros(2, 3, vl.N_CANDIDATES), None),
        (torch.zeros(2, vl.N_CANDIDATES, dtype=torch.float64), None),
        (torch.zeros(2, vl.N_CANDIDATES, dtype=torch.float16), None),
        (torch.zeros(2, vl.N_CANDIDATES, dtype=torch.int64), None),
        (None, torch.zeros(2, vl.N_CANDIDATES - 1)),
        (None, torch.zeros(3, vl.N_CANDIDATES)),
        (None, torch.zeros(2, vl.N_CANDIDATES, dtype=torch.int64)),
        (None, np.zeros((2, vl.N_CANDIDATES), dtype=np.float32)),
    ],
)
def test_policy_loss_rejects_bad_tensors(scores, teacher):
    if scores is None:
        scores = torch.zeros(2, vl.N_CANDIDATES)
    if teacher is None:
        teacher = torch.zeros(2, vl.N_CANDIDATES)
    with pytest.raises(ValueError, match="v104:DATA"):
        vl.policy_loss(scores, teacher, "rank")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_policy_loss_rejects_non_finite_inputs(bad):
    scores = torch.zeros(2, vl.N_CANDIDATES)
    teacher = torch.zeros(2, vl.N_CANDIDATES)
    scores[0, 3] = bad
    with pytest.raises(ValueError, match="v104:DATA"):
        vl.policy_loss(scores, teacher, "rank")
    scores = torch.zeros(2, vl.N_CANDIDATES)
    teacher[1, 4] = bad
    with pytest.raises(ValueError, match="v104:DATA"):
        vl.policy_loss(scores, teacher, "rank")
    with pytest.raises(ValueError, match="v104:DATA"):
        vl.policy_loss(scores, teacher, "legacy-reg", **LEGACY_KWARGS)
