"""Synthetic unit tests for ``touhou_rl.v430.ppo`` (W04A).

**合成测试，不是游戏/能力证据**（synthetic tests, not game or capability
evidence）.  Every number in this file is hand-computed from a few scalars; no
rollout, no engine, no captured data and no checkpoint is involved.  Passing
this file only shows that the loss/diagnostic primitives are arithmetically
correct and fail-closed — it says nothing about TH08 performance.

Coverage map (requirement -> test):
 1  ratio identity                    -> test_ratio_identity_is_one
 2  clip boundary + exact fraction    -> test_clip_boundary_and_clip_fraction_exact
 3  hand-computed surrogate           -> test_clipped_surrogate_hand_computed
 4  clip_ratio <= 0 / inf             -> test_clip_ratio_invalid_rejected
 5  value loss == MSE without clip    -> test_value_loss_without_clip_is_mse
 6  clip without old_values raises    -> test_value_loss_clip_without_old_values_raises
 7  entropy sign                      -> test_entropy_bonus_is_negative_weighted_mean
 8  coef < 0 rejected                 -> test_entropy_coefficient_negative_rejected
 9  kl returns two, zero at identity  -> test_kl_estimators_returns_two_and_zero_at_identity
10  k1 may be negative, k3 never      -> test_kl_k1_can_be_negative_k3_cannot
11  total_loss weighting              -> test_total_loss_weighted_sum_hand_computed
12  gradient diagnostics              -> test_gradient_diagnostics_norm_skip_and_cosine_conflict
13  shape/length/empty/NaN rejected   -> test_shape_mismatch_empty_and_nan_rejected
14  synthetic-scope header + imports  -> test_file_header_declares_synthetic_scope,
                                        test_ppo_module_imports_no_engine_or_learn
Plus extras: clipped value loss hand-check, strict old_values-without-clip refusal.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from touhou_rl.v430 import ppo
from touhou_rl.v430.ppo import PpoError


def _t(values, dtype=torch.float64) -> torch.Tensor:
    """Small helper: build a tensor from Python scalars without touching globals."""
    return torch.tensor(values, dtype=dtype)


# ---------------------------------------------------------------------------
# 1. ratio
# ---------------------------------------------------------------------------


def test_ratio_identity_is_one() -> None:
    """Requirement 1: identical log-probs give exactly 1.0."""
    old = _t([-0.5, -1.25, 2.0])
    r = ppo.ratio(old.clone(), old.clone())
    assert torch.equal(r, torch.ones_like(old))
    assert r.dtype == old.dtype and r.shape == old.shape


def test_ratio_equals_exp_of_difference() -> None:
    """ratio is exp(new - old), not a log/exp round trip."""
    new = _t([0.0, math.log(2.0)])
    old = _t([0.0, 0.0])
    r = ppo.ratio(new, old)
    assert r[0].item() == 1.0
    assert r[1].item() == pytest.approx(2.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. clip boundary and exact clip_fraction
# ---------------------------------------------------------------------------


def test_clip_boundary_and_clip_fraction_exact() -> None:
    """Requirement 2: A>0 clips at r>1+c, A<0 clips at r<1-c; fraction is exact.

    Four samples, clip_ratio=0.2 -> band [0.8, 1.2]:
      * A=+1 with r=1.50 -> r > 1.2 -> clipped
      * A=-1 with r=0.50 -> r < 0.8 -> clipped
      * A=+1 with r=1.10 -> inside      -> not clipped
      * A=-1 with r=0.95 -> inside      -> not clipped
    Exactly 2 of 4 are clipped, so clip_fraction must be exactly 0.5.
    """
    adv = _t([1.0, -1.0, 1.0, -1.0])
    logp_old = _t([0.0, 0.0, 0.0, 0.0])
    logp_new = _t(
        [
            math.log(1.50),
            math.log(0.50),
            math.log(1.10),
            math.log(0.95),
        ]
    )
    out = ppo.clipped_surrogate(adv, logp_new, logp_old, clip_ratio=0.2)

    assert out.clip_fraction == 0.5  # exact, not approx
    assert out.ratio_mean == pytest.approx((1.5 + 0.5 + 1.1 + 0.95) / 4.0, abs=1e-12)

    # Directional claims: the clipped branch is the one that binds.
    r = ppo.ratio(logp_new, logp_old)
    assert r[0].item() > 1.2 and adv[0].item() > 0
    assert r[1].item() < 0.8 and adv[1].item() < 0
    assert 0.8 < r[2].item() < 1.2
    assert 0.8 < r[3].item() < 1.2

    # The loss is the negative of the mean surrogate (never positive here).
    assert out.loss.item() == pytest.approx(
        -((1.2 * 1.0) + (0.8 * -1.0) + (1.1 * 1.0) + (0.95 * -1.0)) / 4.0,
        abs=1e-12,
    )


def test_clip_fraction_zero_when_all_inside_band() -> None:
    """A batch entirely inside the trust region reports exactly 0.0."""
    adv = _t([1.0, -2.0])
    logp_old = _t([0.0, 0.0])
    logp_new = _t([math.log(1.1), math.log(0.9)])
    out = ppo.clipped_surrogate(adv, logp_new, logp_old, clip_ratio=0.2)
    assert out.clip_fraction == 0.0
    assert out.ratio_mean == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. hand-computed surrogate
# ---------------------------------------------------------------------------


def test_clipped_surrogate_hand_computed() -> None:
    """Requirement 3: -mean(min(rA, clip(r)A)) vs hand arithmetic, tol 1e-6.

    clip_ratio=0.2, A=[1, -1, 2], r=[1.5, 0.5, 1.1]:
      s0: min(1.5*1, 1.2*1)  = min(1.5, 1.2)  = 1.2
      s1: min(0.5*-1, 0.8*-1)= min(-0.5, -0.8)= -0.8
      s2: min(1.1*2, 1.2*2)  = min(2.2, 2.4)  = 2.2
    mean = 2.6/3, loss = -2.6/3.
    """
    adv = _t([1.0, -1.0, 2.0])
    logp_old = _t([0.0, 0.0, 0.0])
    logp_new = _t([math.log(1.5), math.log(0.5), math.log(1.1)])

    out = ppo.clipped_surrogate(adv, logp_new, logp_old, clip_ratio=0.2)

    expected_surrogate = (1.2 + -0.8 + 2.2) / 3.0
    assert out.loss.item() == pytest.approx(-expected_surrogate, abs=1e-6)
    assert out.clip_fraction == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert out.policy_loss is out.loss


# ---------------------------------------------------------------------------
# 4. illegal clip_ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -0.2, float("inf"), float("-inf"), float("nan")])
def test_clip_ratio_invalid_rejected(bad: float) -> None:
    """Requirement 4: clip_ratio must be finite and > 0."""
    adv = _t([1.0])
    logp_old = _t([0.0])
    logp_new = _t([0.1])
    with pytest.raises(PpoError):
        ppo.clipped_surrogate(adv, logp_new, logp_old, clip_ratio=bad)


def test_clip_ratio_bool_and_text_rejected() -> None:
    """bool/str coefficients are bugs, not values (fail-closed)."""
    adv = _t([1.0])
    logp_old = _t([0.0])
    logp_new = _t([0.1])
    for bad in (True, "0.2", None):
        with pytest.raises(PpoError):
            ppo.clipped_surrogate(adv, logp_new, logp_old, clip_ratio=bad)


# ---------------------------------------------------------------------------
# 5. value loss == MSE without clip
# ---------------------------------------------------------------------------


def test_value_loss_without_clip_is_mse() -> None:
    """Requirement 5: no clip -> plain MSE, hand-checked.

    values=[1,2,3], returns=[1.5,1.5,4] -> diffs=[-0.5,0.5,-1]
    squares=[0.25,0.25,1] -> mean = 0.5.
    """
    loss = ppo.value_loss(_t([1.0, 2.0, 3.0]), _t([1.5, 1.5, 4.0]))
    assert loss.item() == pytest.approx(0.5, abs=1e-12)
    assert loss.item() == pytest.approx(
        float(torch.mean((_t([1.0, 2.0, 3.0]) - _t([1.5, 1.5, 4.0])) ** 2)),
        abs=1e-12,
    )


def test_value_loss_clipped_matches_hand_computation() -> None:
    """Clipped value loss: max((v-R)^2, (clip(v, v_old±c)-R)^2).

    Sample A: v=2.0, R=1.0, v_old=1.0, c=0.5 -> clip(v)=1.5
              max((2-1)^2, (1.5-1)^2) = max(1.0, 0.25) = 1.0
    Sample B: v=1.4, R=1.0, v_old=1.0, c=0.5 -> clip(v)=1.4 (unchanged)
              max(0.16, 0.16) = 0.16
    mean = (1.0 + 0.16)/2 = 0.58
    """
    loss = ppo.value_loss(
        _t([2.0, 1.4]),
        _t([1.0, 1.0]),
        clip=0.5,
        old_values=_t([1.0, 1.0]),
    )
    assert loss.item() == pytest.approx(0.58, abs=1e-12)


# ---------------------------------------------------------------------------
# 6. clip without old_values must raise (no silent MSE fallback)
# ---------------------------------------------------------------------------


def test_value_loss_clip_without_old_values_raises() -> None:
    """Requirement 6: clip alone is an error, never a silent MSE downgrade."""
    values = _t([1.0, 2.0])
    returns = _t([0.0, 0.0])
    with pytest.raises(PpoError, match="old_values"):
        ppo.value_loss(values, returns, clip=0.5)

    # And the reverse ambiguity is refused too (reference would be ignored).
    with pytest.raises(PpoError, match="old_values"):
        ppo.value_loss(values, returns, old_values=_t([1.0, 2.0]))


def test_value_loss_clip_invalid_rejected() -> None:
    """Non-positive / non-finite clip for the value loss is rejected."""
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(PpoError):
            ppo.value_loss(
                _t([1.0]), _t([0.0]), clip=bad, old_values=_t([1.0])
            )


# ---------------------------------------------------------------------------
# 7. entropy bonus sign
# ---------------------------------------------------------------------------


def test_entropy_bonus_is_negative_weighted_mean() -> None:
    """Requirement 7: bonus == -coef * mean(H), i.e. always <= 0 for H >= 0."""
    h = _t([0.5, 1.5, 2.5])  # mean 1.5
    term = ppo.entropy_bonus(h, coef=0.001)
    assert term.item() == pytest.approx(-0.001 * 1.5, abs=1e-15)
    assert term.item() < 0.0


def test_entropy_bonus_zero_coefficient_is_zero() -> None:
    """coef=0 is legal and must produce exactly 0, not a sign-flipped term."""
    term = ppo.entropy_bonus(_t([0.5, 1.5]), coef=0.0)
    assert term.item() == 0.0


# ---------------------------------------------------------------------------
# 8. negative entropy coefficient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.001, float("nan"), float("inf"), True, "0.001"])
def test_entropy_coefficient_negative_rejected(bad) -> None:
    """Requirement 8: coef < 0 (and non-scalar junk) raises PpoError."""
    with pytest.raises(PpoError):
        ppo.entropy_bonus(_t([1.0]), coef=bad)


# ---------------------------------------------------------------------------
# 9. kl_estimators returns two, zero at identity
# ---------------------------------------------------------------------------


def test_kl_estimators_returns_two_and_zero_at_identity() -> None:
    """Requirement 9: exactly two estimates; both exactly 0 when new == old."""
    logp = _t([-0.5, -1.0, 0.3])
    out = ppo.kl_estimators(logp.clone(), logp.clone())
    assert isinstance(out, tuple) and len(out) == 2
    k1, k3 = out
    assert isinstance(k1, float) and isinstance(k3, float)
    assert k1 == 0.0
    assert k3 == 0.0
    assert k3 >= 0.0


# ---------------------------------------------------------------------------
# 10. k1 may be negative, k3 never
# ---------------------------------------------------------------------------


def test_kl_k1_can_be_negative_k3_cannot() -> None:
    """Requirement 10: construct k1 < 0 while k3 stays >= 0.

    new-old = [0.1, 0.2] -> k1 = -mean([0.1,0.2]) = -0.15 (< 0)
                            k3 = mean([0.01,0.04])/2 = 0.0125 (>= 0)
    """
    logp_new = _t([-1.0, -2.0])
    logp_old = _t([-1.1, -2.2])
    k1, k3 = ppo.kl_estimators(logp_new, logp_old)

    assert k1 == pytest.approx(-0.15, abs=1e-12)
    assert k1 < 0.0
    assert k3 == pytest.approx(0.0125, abs=1e-12)
    assert k3 > 0.0

    # k3 is invariant to the sign of the deviation, k1 is not (do not mix them).
    k1_swapped, k3_swapped = ppo.kl_estimators(logp_old, logp_new)
    assert k1_swapped == pytest.approx(0.15, abs=1e-12)
    assert k3_swapped == pytest.approx(k3, abs=1e-15)


# ---------------------------------------------------------------------------
# 11. total loss weighting
# ---------------------------------------------------------------------------


def test_total_loss_weighted_sum_hand_computed() -> None:
    """Requirement 11: policy + value_coef*value + entropy_coef*entropy_term.

    1.0 + 0.5*2.0 + 0.001*(-0.5) = 1.0 + 1.0 - 0.0005 = 1.9995
    """
    total = ppo.total_loss(
        1.0, 2.0, -0.5, value_coef=0.5, entropy_coef=0.001
    )
    assert total.item() == pytest.approx(1.9995, abs=1e-12)


def test_total_loss_keeps_gradient_path() -> None:
    """The weighted sum must stay differentiable through its terms."""
    policy = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    total = ppo.total_loss(
        policy, 2.0, -0.5, value_coef=0.5, entropy_coef=0.001
    )
    total.backward()
    assert policy.grad is not None
    assert policy.grad.item() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value_coef": -0.5, "entropy_coef": 0.001},
        {"value_coef": 0.5, "entropy_coef": -0.001},
        {"value_coef": float("nan"), "entropy_coef": 0.001},
        {"value_coef": 0.5, "entropy_coef": float("inf")},
    ],
)
def test_total_loss_invalid_coefficients_rejected(kwargs) -> None:
    """Illegal coefficients are rejected instead of producing a NaN loss."""
    with pytest.raises(PpoError):
        ppo.total_loss(1.0, 2.0, -0.5, **kwargs)


# ---------------------------------------------------------------------------
# 12. gradient diagnostics
# ---------------------------------------------------------------------------


def test_gradient_diagnostics_norm_skip_and_cosine_conflict() -> None:
    """Requirement 12: [3,4] -> norm 5.0; None skipped; non-finite counted;
    opposite reference -> cos == -1 and conflict True.
    """
    first = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    (first * _t([3.0, 4.0])).sum().backward()  # grad = [3, 4] -> L2 = 5

    never_used = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    assert never_used.grad is None  # must be skipped, not counted as zero

    zero_grad = torch.tensor([1.0, -1.0], dtype=torch.float64, requires_grad=True)
    zero_grad.grad = torch.zeros(2, dtype=torch.float64)

    inf_grad = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    inf_grad.grad = _t([float("inf")])

    params = [first, None, never_used, zero_grad, inf_grad]
    diag = ppo.gradient_diagnostics(params)

    assert diag.grad_norm == 5.0  # exact in float64
    assert diag.n_nonfinite == 1  # inf_grad
    assert diag.n_zero == 1  # zero_grad (never_used has no .grad at all)
    assert diag.n_finite == 2  # first + zero_grad
    assert diag.cosine is None and diag.conflict is None

    reference = [_t([-3.0, -4.0]), None, None, _t([0.0, 0.0]), _t([1.0])]
    diag_ref = ppo.gradient_diagnostics(params, reference_grads=reference)
    assert diag_ref.grad_norm == 5.0
    assert diag_ref.cosine == pytest.approx(-1.0, abs=1e-12)
    assert diag_ref.conflict is True
    assert diag_ref.n_reference_pairs == 2

    # Same direction -> cos == +1, no conflict.
    same = [_t([3.0, 4.0]), None, None, _t([0.0, 0.0]), _t([1.0])]
    diag_same = ppo.gradient_diagnostics(params, reference_grads=same)
    assert diag_same.cosine == pytest.approx(1.0, abs=1e-12)
    assert diag_same.conflict is False


def test_gradient_diagnostics_accepts_generator_parameters() -> None:
    """A one-shot iterable must work (length check must not consume it twice)."""
    param = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    (param * _t([6.0])).sum().backward()
    diag = ppo.gradient_diagnostics(
        (p for p in [param]), reference_grads=[_t([-6.0])]
    )
    assert diag.grad_norm == 6.0
    assert diag.cosine == pytest.approx(-1.0, abs=1e-12)
    assert diag.conflict is True


def test_gradient_diagnostics_rejects_bad_reference_alignment() -> None:
    """Misaligned or unusable references raise instead of guessing."""
    param = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    (param * _t([1.0])).sum().backward()

    with pytest.raises(PpoError, match="positionally aligned"):
        ppo.gradient_diagnostics([param], reference_grads=[_t([1.0]), _t([2.0])])
    with pytest.raises(PpoError, match="positionally aligned"):
        ppo.gradient_diagnostics([param], reference_grads=[])
    with pytest.raises(PpoError):
        ppo.gradient_diagnostics(param)  # a bare tensor is not an iterable
    with pytest.raises(PpoError, match="zero norm"):
        ppo.gradient_diagnostics([param], reference_grads=[_t([0.0])])
    with pytest.raises(PpoError, match="no finite"):
        ppo.gradient_diagnostics(
            [param], reference_grads=[_t([float("nan")])]
        )
    with pytest.raises(PpoError):
        ppo.gradient_diagnostics([param], reference_grads=[_t([1.0, 2.0])])


# ---------------------------------------------------------------------------
# 13. shape / length / empty / NaN rejection
# ---------------------------------------------------------------------------


def test_shape_mismatch_empty_and_nan_rejected() -> None:
    """Requirement 13: every malformed input raises PpoError."""
    good = _t([0.0, 0.0, 0.0])
    # length / shape mismatch
    with pytest.raises(PpoError):
        ppo.ratio(good, _t([0.0, 0.0]))
    with pytest.raises(PpoError):
        ppo.clipped_surrogate(
            _t([1.0, 1.0]), good, _t([0.0, 0.0]), clip_ratio=0.2
        )
    with pytest.raises(PpoError):
        ppo.value_loss(good, _t([0.0]))
    with pytest.raises(PpoError):
        ppo.kl_estimators(good, _t([0.0]))
    # empty batch
    with pytest.raises(PpoError, match="empty"):
        ppo.ratio(_t([]), _t([]))
    with pytest.raises(PpoError, match="empty"):
        ppo.clipped_surrogate(_t([]), _t([]), _t([]), clip_ratio=0.2)
    with pytest.raises(PpoError, match="empty"):
        ppo.value_loss(_t([]), _t([]))
    with pytest.raises(PpoError, match="empty"):
        ppo.entropy_bonus(_t([]), coef=0.001)
    with pytest.raises(PpoError, match="empty"):
        ppo.kl_estimators(_t([]), _t([]))
    # NaN / Inf inputs
    with pytest.raises(PpoError, match="NaN or Inf"):
        ppo.ratio(_t([float("nan")]), _t([0.0]))
    with pytest.raises(PpoError, match="NaN or Inf"):
        ppo.ratio(_t([0.0]), _t([float("inf")]))
    with pytest.raises(PpoError, match="NaN or Inf"):
        ppo.clipped_surrogate(
            _t([float("nan")]), _t([0.0]), _t([0.0]), clip_ratio=0.2
        )
    with pytest.raises(PpoError, match="NaN or Inf"):
        ppo.value_loss(_t([float("inf")]), _t([0.0]))
    with pytest.raises(PpoError, match="NaN or Inf"):
        ppo.value_loss(
            _t([float("nan")]), _t([0.0]), clip=0.5, old_values=_t([0.0])
        )
    with pytest.raises(PpoError, match="NaN or Inf"):
        ppo.entropy_bonus(_t([float("nan")]), coef=0.001)
    with pytest.raises(PpoError, match="NaN or Inf"):
        ppo.kl_estimators(_t([float("nan")]), _t([0.0]))
    # wrong types / dtypes
    with pytest.raises(PpoError):
        ppo.ratio([0.0], [0.0])
    with pytest.raises(PpoError):
        ppo.ratio(torch.tensor([0, 1]), torch.tensor([0, 1]))
    with pytest.raises(PpoError):
        ppo.clipped_surrogate(
            _t([1.0]), _t([0.0]), _t([0.0]), clip_ratio=_t([0.2, 0.3])
        )


def test_ratio_overflow_is_rejected_not_clamped() -> None:
    """A ratio that overflows to Inf is an error, not a value to clamp."""
    with pytest.raises(PpoError, match="overflow"):
        ppo.ratio(_t([1e300]), _t([-1e300]))


# ---------------------------------------------------------------------------
# 14. synthetic scope header, no engine/learn imports, purity
# ---------------------------------------------------------------------------


def test_file_header_declares_synthetic_scope() -> None:
    """Requirement 14: the test file says it is synthetic, not capability evidence."""
    text = Path(__file__).read_text(encoding="utf-8")
    assert "合成测试，不是游戏/能力证据" in text
    assert "no engine" in text.lower()


def test_ppo_module_imports_no_engine_or_learn() -> None:
    """``ppo`` must stay pure: torch + stdlib only, no engine/v400/v420/learn."""
    source = Path(ppo.__file__).read_text(encoding="utf-8")
    import_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("touhou_rl.v400", "touhou_rl.v420", "v430.learn", "th08", "os", "random")
    for line in import_lines:
        assert not line.startswith("import touhou_rl"), line
        assert not line.startswith("from touhou_rl"), line
        for needle in forbidden:
            assert needle not in line.split("#")[0], f"forbidden import: {line}"


def test_functions_are_reproducible_and_do_not_move_the_input() -> None:
    """Same inputs -> identical outputs; inputs are not mutated in place."""
    adv = _t([1.0, -1.0])
    new = _t([0.1, -0.2])
    old = _t([0.0, 0.0])
    before = (adv.clone(), new.clone(), old.clone())

    a = ppo.clipped_surrogate(adv, new, old, clip_ratio=0.2)
    b = ppo.clipped_surrogate(adv, new, old, clip_ratio=0.2)
    assert torch.equal(a.loss, b.loss)
    assert a.clip_fraction == b.clip_fraction
    assert a.ratio_mean == b.ratio_mean

    assert torch.equal(adv, before[0])
    assert torch.equal(new, before[1])
    assert torch.equal(old, before[2])

    k1a, k3a = ppo.kl_estimators(new, old)
    k1b, k3b = ppo.kl_estimators(new, old)
    assert k1a == k1b and k3a == k3b
