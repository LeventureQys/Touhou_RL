"""PPO loss and diagnostic primitives (v4.3.0 work package W04A).

This module is the *pure-function* part of the v4.3.0 PPO objective described in
``DEVELOPMENT_GUIDE.md`` §7.3–§7.4 and in the ``ppo`` block of
``PLAN_SPEC.json``.  It deliberately contains **no** rollout, **no** engine
access, **no** optimizer, **no** data loading and **no** training loop: every
function here is a total function of its tensor arguments.

Design rules (all enforced by this file, and by ``tests/test_v430_ppo.py``):

* **pure / reproducible** — no global state is read or written, no seed is set,
  no randomness is used, no environment variable or file is touched.  The same
  inputs always produce bit-identical outputs on the same device/dtype.
* **torch + stdlib only** — the module imports :mod:`torch`, :mod:`math`,
  :mod:`dataclasses` and :mod:`typing` and nothing else.  In particular it does
  *not* import the engine, :mod:`touhou_rl.v400`, :mod:`touhou_rl.v420`,
  :mod:`touhou_rl.v430.learn`, or any model module; head shapes/log-probs come
  in as tensors from the caller (see :mod:`touhou_rl.v430.models.actor`).
* **fail-closed** — every invalid input (shape/length mismatch, empty batch,
  NaN/Inf, illegal coefficient, missing ``old_values``) raises
  :class:`PpoError`.  Nothing silently degrades, e.g. a clipped value loss is
  never quietly replaced by a plain MSE, and a negative entropy coefficient is
  never silently flipped.
* **no hidden device sync** — scalar reads happen only through ``float(...)``
  on a detached CPU scalar (the ``.item()``-equivalent), and only to return
  diagnostics.  Gradients never leave the autograd graph inside a loss function.

What is intentionally *not* here (kept out of this slice): advantage
normalisation, GAE/return computation (:mod:`touhou_rl.v430.sequence`), the BC
loss (:mod:`touhou_rl.v430.learn`), gradient clipping at ``max_grad_norm``,
optimizer steps, KL-based early stop, and minibatch iteration.  Those consume
these functions; they are not re-implemented here.

Coefficient values are **not** hard-coded from ``PLAN_SPEC.json`` on purpose:
``clip_ratio=0.2``, ``value_coefficient=0.5``, ``entropy_coefficient=0.001`` and
``max_grad_norm=0.5`` are development parameters passed in by the caller, so a
config review can change them without touching this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "PpoError",
    "ClippedSurrogate",
    "GradientDiagnostics",
    "ratio",
    "clipped_surrogate",
    "value_loss",
    "entropy_bonus",
    "kl_estimators",
    "total_loss",
    "gradient_diagnostics",
]


class PpoError(ValueError):
    """Raised for every invalid input to a PPO helper.

    Subclasses :class:`ValueError` so that callers which only guard against
    bad values keep working, while still allowing a targeted ``except
    PpoError``.  The message always names the offending argument and why it is
    rejected; no exception is raised for a merely *unusual* but valid input.
    """


# ---------------------------------------------------------------------------
# validation helpers (private; all raise PpoError, none mutate their input)
# ---------------------------------------------------------------------------


def _is_real_scalar(value: Any) -> bool:
    """True for an ``int``/``float`` that is not a ``bool``."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_scalar(name: str, value: Any) -> float:
    """Return ``value`` as a finite Python float, or raise :class:`PpoError`.

    Accepts a Python real or a 0-dim/1-element tensor, so callers can pass a
    config value or a tensor coefficient interchangeably.  ``bool`` is
    rejected: ``True`` as a coefficient is a bug, not a value of 1.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise PpoError(
                f"{name} must be a scalar, got a tensor of shape {tuple(value.shape)}"
            )
        value = value.detach().to(torch.float64).reshape(()).item()
    if not _is_real_scalar(value):
        raise PpoError(
            f"{name} must be a real scalar, got {type(value).__name__}: {value!r}"
        )
    out = float(value)
    if not math.isfinite(out):
        raise PpoError(f"{name} must be finite, got {out!r}")
    return out


def _float_tensor(name: str, value: Any) -> torch.Tensor:
    """Require a floating-point :class:`torch.Tensor` (no silent casting)."""
    if not isinstance(value, torch.Tensor):
        raise PpoError(
            f"{name} must be a torch.Tensor, got {type(value).__name__}"
        )
    if not value.is_floating_point():
        raise PpoError(
            f"{name} must be a floating-point tensor, got dtype {value.dtype}"
        )
    return value


def _nonempty(name: str, tensor: torch.Tensor) -> None:
    if tensor.numel() == 0:
        raise PpoError(f"{name} is an empty batch (numel == 0)")


def _finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise PpoError(f"{name} contains NaN or Inf")


def _same_shape(name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor) -> None:
    if tuple(a.shape) != tuple(b.shape):
        raise PpoError(
            f"{name_a} and {name_b} must have the same shape, "
            f"got {tuple(a.shape)} vs {tuple(b.shape)}"
        )


def _sample_vector(name: str, value: Any) -> torch.Tensor:
    """Validate a per-sample tensor: float, non-empty, all finite."""
    tensor = _float_tensor(name, value)
    _nonempty(name, tensor)
    _finite(name, tensor)
    return tensor


def _loss_scalar(name: str, value: Any) -> torch.Tensor:
    """Validate a scalar loss term, returning a 0-dim tensor that keeps grad."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise PpoError(
                f"{name} must be a scalar, got a tensor of shape {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise PpoError(
                f"{name} must be a floating-point tensor, got dtype {value.dtype}"
            )
        scalar = value.reshape(())
        _finite(name, scalar)
        return scalar
    if not _is_real_scalar(value):
        raise PpoError(
            f"{name} must be a scalar tensor or a real number, "
            f"got {type(value).__name__}"
        )
    out = float(value)
    if not math.isfinite(out):
        raise PpoError(f"{name} must be finite, got {out!r}")
    return torch.as_tensor(out, dtype=torch.float64)


# ---------------------------------------------------------------------------
# 1. ratio
# ---------------------------------------------------------------------------


def ratio(logp_new: Any, logp_old: Any) -> torch.Tensor:
    """Return the per-sample probability ratio ``exp(new - old)``.

    Identical log-probs therefore give exactly ``1.0``; the subtraction happens
    in the inputs' common dtype, so no precision is lost to a redundant
    ``exp``/``log`` round trip.

    Raises:
        PpoError: on a non-tensor/non-float input, non-empty/shape mismatch,
            any NaN/Inf in the inputs, or a ratio that overflows to Inf/NaN
            (e.g. a log-prob difference beyond the dtype's range).  Such a
            ratio is a broken update, not something to clamp silently.

    The result stays attached to the autograd graph through ``logp_new``.
    """
    new = _sample_vector("logp_new", logp_new)
    old = _sample_vector("logp_old", logp_old)
    _same_shape("logp_new", new, "logp_old", old)
    out = torch.exp(new - old)
    if not bool(torch.isfinite(out).all()):
        raise PpoError(
            "ratio overflowed to a non-finite value; the log-prob difference "
            "is too large for the dtype"
        )
    return out


# ---------------------------------------------------------------------------
# 2. clipped surrogate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClippedSurrogate:
    """Result of :func:`clipped_surrogate`.

    Attributes:
        loss: 0-dim tensor, ``-mean(min(r*A, clip(r, 1-c, 1+c)*A))``; carries
            the autograd graph.  ``policy_loss`` is an alias.
        clip_fraction: Python float, the fraction of samples whose clipped
            branch was strictly smaller than the unclipped branch (i.e. the
            samples where the min actually selected the clipped term).
        ratio_mean: Python float, ``mean(r)`` over the batch.
    """

    loss: torch.Tensor
    clip_fraction: float
    ratio_mean: float

    @property
    def policy_loss(self) -> torch.Tensor:
        """Alias for :attr:`loss` (the name used in the guide)."""
        return self.loss


def clipped_surrogate(
    advantages: Any,
    logp_new: Any,
    logp_old: Any,
    *,
    clip_ratio: Any,
) -> ClippedSurrogate:
    """PPO clipped surrogate loss (negated, i.e. ready to minimise).

    ``r = ratio(logp_new, logp_old)``; the returned loss is

    ``-mean( min(r*A, clip(r, 1-clip_ratio, 1+clip_ratio)*A) )``

    which is the standard maximisation objective with the sign flipped, so a
    caller can add it directly to a value/entropy term and call
    ``backward()``.

    With ``A > 0`` the min picks the clipped term once ``r > 1 + clip_ratio``;
    with ``A < 0`` it picks the clipped term once ``r < 1 - clip_ratio``.  That
    is exactly the set counted by ``clip_fraction``.

    Args:
        advantages: per-sample advantage tensor (finite, non-empty).
        logp_new: log-prob of the acted joint action under the current policy.
        logp_old: log-prob of the same action under the behaviour policy.
        clip_ratio: positive finite epsilon of the trust region (e.g. ``0.2``).

    Raises:
        PpoError: illegal ``clip_ratio`` (``<= 0``, non-finite, non-real), any
            shape/length mismatch, empty batch, or NaN/Inf in an input.

    Note:
        No advantage normalisation happens here (the guide asks for it over the
        valid actor samples *before* this call); no ``.item()`` is taken inside
        the loss path, and no global state is touched.
    """
    clip = _finite_scalar("clip_ratio", clip_ratio)
    if clip <= 0.0:
        raise PpoError(f"clip_ratio must be > 0, got {clip!r}")

    adv = _sample_vector("advantages", advantages)
    new = _sample_vector("logp_new", logp_new)
    old = _sample_vector("logp_old", logp_old)
    _same_shape("advantages", adv, "logp_new", new)
    _same_shape("logp_new", new, "logp_old", old)

    r = ratio(new, old)
    unclipped = r * adv
    clamped = torch.clamp(r, 1.0 - clip, 1.0 + clip)
    clipped = clamped * adv
    surrogate = torch.minimum(unclipped, clipped)
    loss = -surrogate.mean()

    is_clipped = clipped < unclipped
    clip_fraction = float(is_clipped.detach().to(torch.float64).mean())
    ratio_mean = float(r.detach().to(torch.float64).mean())
    return ClippedSurrogate(
        loss=loss, clip_fraction=clip_fraction, ratio_mean=ratio_mean
    )


# ---------------------------------------------------------------------------
# 3. value loss
# ---------------------------------------------------------------------------


def value_loss(
    values: Any,
    returns: Any,
    *,
    clip: Any = None,
    old_values: Any = None,
) -> torch.Tensor:
    """Critic loss: plain MSE, or PPO's clipped value loss when asked for.

    * ``clip is None`` and ``old_values is None`` -> ``mean((v - R) ** 2)``.
    * ``clip is not None`` -> ``mean(max((v-R)**2, (clip(v, v_old ± clip) - R)**2))``
      where the clamp is around ``old_values`` (the pre-update critic output).

    Refusals (fail-closed, no silent fallback):

    * ``clip`` without ``old_values`` -> :class:`PpoError`.  A clipped value
      loss without the reference point is *not* defined, and degrading to MSE
      would change the objective behind the caller's back.
    * ``old_values`` without ``clip`` -> :class:`PpoError`.  The reference is
      then unused; accepting it would hide a caller bug.
    * non-positive/non-finite ``clip``, shape mismatch, empty batch, NaN/Inf.

    The guide keeps the critic objective separate from the actor loss; that
    separation is preserved here — this function only ever returns the critic
    term and never combines it with anything else.
    """
    v = _sample_vector("values", values)
    ret = _sample_vector("returns", returns)
    _same_shape("values", v, "returns", ret)

    if clip is None:
        if old_values is not None:
            raise PpoError(
                "old_values was given without clip; the clipped value loss "
                "needs both. Refusing to silently ignore old_values."
            )
        return ((v - ret) ** 2).mean()

    if old_values is None:
        raise PpoError(
            "clip was given without old_values; the PPO clipped value loss "
            "needs old_values and will not be silently degraded to plain MSE."
        )

    c = _finite_scalar("clip", clip)
    if c <= 0.0:
        raise PpoError(f"clip must be > 0, got {c!r}")

    old = _sample_vector("old_values", old_values)
    _same_shape("values", v, "old_values", old)
    _same_shape("values", v, "returns", ret)

    clipped_v = old + torch.clamp(v - old, -c, c)
    mse = (v - ret) ** 2
    clipped_mse = (clipped_v - ret) ** 2
    return torch.maximum(mse, clipped_mse).mean()


# ---------------------------------------------------------------------------
# 4. entropy bonus
# ---------------------------------------------------------------------------


def entropy_bonus(entropies: Any, *, coef: Any) -> torch.Tensor:
    """Entropy term for a **maximised** entropy: returns ``-coef * mean(H)``.

    The sign is negative on purpose.  Entropy is maximised by the trainer, and
    this module returns losses that are minimised, so the term enters the total
    loss with a minus.  ``coef`` is the non-negative weight (e.g. ``0.001``);
    a negative value is rejected rather than silently flipped, because flipping
    it would turn exploration pressure into entropy collapse.

    Raises:
        PpoError: ``coef < 0``, ``coef`` non-finite/non-real, empty batch,
            non-float or non-finite entropies.
    """
    c = _finite_scalar("coef", coef)
    if c < 0.0:
        raise PpoError(
            f"coef must be >= 0 (entropy is maximised, the minus sign is "
            f"applied by this function), got {c!r}"
        )
    h = _sample_vector("entropies", entropies)
    return -c * h.mean()


# ---------------------------------------------------------------------------
# 5. KL estimators
# ---------------------------------------------------------------------------


def kl_estimators(logp_new: Any, logp_old: Any) -> tuple[float, float]:
    """Return **two** KL estimates ``(k1, k3)`` as Python floats.

    With ``d = logp_new - logp_old``:

    * ``k1 = mean(old - new) = mean(-d)`` — unbiased, but it *can* be negative
      on a finite sample even when the true KL is positive.
    * ``k3 = mean((new - old) ** 2) / 2 = mean(d**2) / 2`` — always ``>= 0``,
      biased (it is the second-order expansion), so it overestimates for large
      deviations.

    **Do not mix them.**  Pick one estimator for a run and report which one was
    used: ``k1`` is the one that can go negative (so a threshold test must not
    assume ``k1 >= 0``), ``k3`` is the conservative non-negative one (so an
    epoch-stop threshold calibrated on ``k3`` is not comparable to one
    calibrated on ``k1``).  Both are returned together so the caller — and the
    diagnostics in the result JSON — can record the pair instead of an
    unlabelled number.

    Both are computed in float64 after detaching: KL is a *diagnostic*, never
    part of the loss graph (``logp_new`` may require grad; the graph is
    discarded here, so measuring KL cannot accidentally add a gradient path).

    Raises:
        PpoError: shape mismatch, empty batch, NaN/Inf, non-float tensors.
    """
    new = _sample_vector("logp_new", logp_new)
    old = _sample_vector("logp_old", logp_old)
    _same_shape("logp_new", new, "logp_old", old)

    a = new.detach().to(torch.float64).reshape(-1)
    b = old.detach().to(torch.float64).reshape(-1)
    diff = a - b
    k1 = float((-diff).mean())
    k3 = float((diff * diff).mean() / 2.0)
    if not (math.isfinite(k1) and math.isfinite(k3)):
        raise PpoError(
            f"kl_estimators produced a non-finite value: k1={k1!r}, k3={k3!r}"
        )
    # Mathematically impossible, kept as a fail-closed invariant guard.
    if k3 < 0.0:  # pragma: no cover - defensive
        raise PpoError(f"k3 must be >= 0 by construction, got {k3!r}")
    return (k1, k3)


# ---------------------------------------------------------------------------
# 6. total loss
# ---------------------------------------------------------------------------


def total_loss(
    policy_loss: Any,
    value_loss: Any,
    entropy_term: Any,
    *,
    value_coef: Any,
    entropy_coef: Any,
) -> torch.Tensor:
    """Weighted sum ``policy + value_coef*value + entropy_coef*entropy_term``.

    ``entropy_term`` is expected to be the output of :func:`entropy_bonus`
    (already negated).  Passing the raw mean entropy instead would silently
    flip the exploration pressure, so callers must go through
    :func:`entropy_bonus`.

    Coefficients must be finite and non-negative; ``value_coef >= 0`` and
    ``entropy_coef >= 0``.  A negative weight is rejected rather than accepted
    as "just a sign": the individual terms already carry their own signs.

    Raises:
        PpoError: illegal coefficient, non-scalar/non-finite loss term.
    """
    vc = _finite_scalar("value_coef", value_coef)
    if vc < 0.0:
        raise PpoError(f"value_coef must be >= 0, got {vc!r}")
    ec = _finite_scalar("entropy_coef", entropy_coef)
    if ec < 0.0:
        raise PpoError(f"entropy_coef must be >= 0, got {ec!r}")

    p = _loss_scalar("policy_loss", policy_loss)
    v = _loss_scalar("value_loss", value_loss)
    e = _loss_scalar("entropy_term", entropy_term)
    return p + vc * v + ec * e


# ---------------------------------------------------------------------------
# 7. gradient diagnostics (§7.4 "record per-term gradient norms and conflict")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GradientDiagnostics:
    """Result of :func:`gradient_diagnostics` (all read-only numbers).

    Attributes:
        grad_norm: L2 norm over all *finite* gradients, in float64.
        n_nonfinite: number of gradient tensors skipped for containing NaN/Inf.
        n_zero: number of gradient tensors that are exactly all-zero.
        n_finite: number of gradient tensors that entered :attr:`grad_norm`.
        cosine: cosine similarity against ``reference_grads``, or ``None`` when
            no reference was supplied.
        conflict: ``cosine < 0`` (strict), or ``None`` with no reference.
        n_reference_pairs: number of gradient tensors that had a finite
            reference counterpart and entered the cosine.
    """

    grad_norm: float
    n_nonfinite: int
    n_zero: int
    n_finite: int
    cosine: float | None = None
    conflict: bool | None = None
    n_reference_pairs: int = 0


def _iter_grads(parameters: Any, *, label: str) -> list[tuple[int, torch.Tensor]]:
    """Collect ``(position, flat float64 grad)`` for gradients that exist.

    ``None`` parameters and parameters whose ``.grad`` is ``None`` are skipped
    entirely: a missing gradient is not a zero gradient and must not inflate
    ``n_zero`` or dilute the norm.  Positions refer to the caller's iterable,
    so ``reference_grads`` can be aligned by position.
    """
    if isinstance(parameters, torch.Tensor):
        raise PpoError(
            f"{label} must be an iterable of tensors, not a single tensor"
        )
    try:
        items = list(parameters)
    except TypeError as exc:
        raise PpoError(f"{label} must be an iterable of tensors: {exc}") from exc

    grads: list[tuple[int, torch.Tensor]] = []
    for position, item in enumerate(items):
        if item is None:
            continue
        param = _float_tensor(f"{label}[{position}]", item)
        grad = param.grad
        if grad is None:
            continue
        grad = _float_tensor(f"{label}[{position}].grad", grad)
        grads.append((position, grad.detach().to(torch.float64).reshape(-1)))
    return grads


def gradient_diagnostics(
    parameters: Any,
    *,
    reference_grads: Any = None,
) -> GradientDiagnostics:
    """Summarise gradients for §7.4 diagnostics (norm, zeros, direction conflict).

    Args:
        parameters: iterable of tensors carrying ``.grad`` (``None`` entries are
            allowed and are skipped).  The iterable order defines the position
            used to align ``reference_grads``.
        reference_grads: optional iterable **positionally aligned** with
            ``parameters`` where each entry is a tensor (an independent
            gradient, e.g. the BC loss's gradient w.r.t. the same parameter) or
            ``None`` when that parameter has no reference.  It must have the
            same length as ``parameters``.

    Returns:
        :class:`GradientDiagnostics`.  ``grad_norm`` is ``sqrt(sum(g**2))`` over
        the finite gradients only; tensors containing NaN/Inf are counted in
        ``n_nonfinite`` and excluded from the norm, exactly-zero tensors are
        counted in ``n_zero`` and contribute zero.  With a reference, ``cosine``
        is the cosine similarity between the concatenated own gradient and the
        concatenated reference gradient; ``conflict`` is ``cosine < 0``.

    Raises:
        PpoError: ``parameters`` not iterable / a tensor itself, non-float
            tensor, ``reference_grads`` length mismatch, reference shape
            mismatch, no comparable finite pair, or a zero-norm vector (cosine
            undefined).

    This function reads scalars only at the end, via ``float(...)`` on detached
    CPU scalars (the ``.item()``-equivalent); it never mutates ``.grad`` and
    never touches global state.
    """
    items = _materialise(parameters)
    own = _iter_grads(items, label="parameters")

    sq_sum = 0.0
    n_nonfinite = 0
    n_zero = 0
    n_finite = 0
    for _, grad in own:
        if not bool(torch.isfinite(grad).all()):
            n_nonfinite += 1
            continue
        n_finite += 1
        if bool((grad == 0).all()):
            n_zero += 1
        sq_sum += float((grad * grad).sum())
    grad_norm = math.sqrt(sq_sum)
    if not math.isfinite(grad_norm):
        raise PpoError(f"grad_norm is not finite: {grad_norm!r}")

    if reference_grads is None:
        return GradientDiagnostics(
            grad_norm=grad_norm,
            n_nonfinite=n_nonfinite,
            n_zero=n_zero,
            n_finite=n_finite,
        )

    if isinstance(reference_grads, torch.Tensor):
        raise PpoError(
            "reference_grads must be an iterable of tensors, not a single tensor"
        )
    try:
        ref_items = list(reference_grads)
    except TypeError as exc:
        raise PpoError(
            f"reference_grads must be an iterable: {exc}"
        ) from exc

    if len(ref_items) != len(items):
        raise PpoError(
            "reference_grads must be positionally aligned with parameters: "
            f"got {len(ref_items)} reference entries for {len(items)} parameters"
        )

    dot = 0.0
    a_sq = 0.0
    b_sq = 0.0
    n_pairs = 0
    for position, grad in own:
        ref = ref_items[position]
        if ref is None:
            continue
        if not bool(torch.isfinite(grad).all()):
            continue
        ref_t = _float_tensor(f"reference_grads[{position}]", ref)
        _same_shape("gradient", grad, f"reference_grads[{position}]", ref_t)
        if not bool(torch.isfinite(ref_t).all()):
            continue
        flat = ref_t.detach().to(torch.float64).reshape(-1)
        dot += float((grad * flat).sum())
        a_sq += float((grad * grad).sum())
        b_sq += float((flat * flat).sum())
        n_pairs += 1

    if n_pairs == 0:
        raise PpoError(
            "reference_grads was given but no finite gradient/reference pair "
            "exists, so a cosine similarity cannot be computed"
        )
    denom = math.sqrt(a_sq) * math.sqrt(b_sq)
    if denom == 0.0:
        raise PpoError(
            "cosine similarity is undefined: the own or the reference gradient "
            "has zero norm"
        )
    cosine = dot / denom
    cosine = max(-1.0, min(1.0, cosine))
    return GradientDiagnostics(
        grad_norm=grad_norm,
        n_nonfinite=n_nonfinite,
        n_zero=n_zero,
        n_finite=n_finite,
        cosine=cosine,
        conflict=cosine < 0.0,
        n_reference_pairs=n_pairs,
    )


def _materialise(parameters: Any) -> list[Any]:
    """Return ``parameters`` as a list (helper for length checks only)."""
    if isinstance(parameters, torch.Tensor):
        raise PpoError("parameters must be an iterable of tensors, not a tensor")
    try:
        return list(parameters)
    except TypeError as exc:
        raise PpoError(f"parameters must be an iterable of tensors: {exc}") from exc
