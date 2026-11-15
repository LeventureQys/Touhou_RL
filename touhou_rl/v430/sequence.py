"""Discount/lambda, GAE and sequence segmentation for ``v4.3.0``.

This module implements five pure functions from ``DEVELOPMENT_GUIDE``:

* §7.2 (Return 与 bootstrap) — :func:`discounts_and_lambdas` (the per-decision
  ``gamma_t`` / ``lambda_t`` derived from ``effective_frames``) and
  :func:`gae` (the backward advantage recursion and the value target);
* §7.1 (序列训练) — :func:`segment_episode` (one episode → loss windows with a
  burn-in context), :func:`segment_batch` (group segments for a minibatch) and
  :func:`padding_statistics` (prove padding never enters the advantage/entropy
  denominator).

It is deliberately **not** the training path: the hidden state, the
exact-prefix recomputation, the model, the optimiser and the rollout loop are
out of scope and are not implemented here.  Nothing in this module opens a
file, starts a process, imports ``torch``/``numpy`` or mutates global state.

Contract (§7.2), with two frames as the reference step:

```text
gamma_t        = exp(-effective_frames_t / (hz * tau_s))
lambda_t       = lambda_base ** (effective_frames_t / 2)
delta_t        = r_t + gamma_t * bootstrap_mask_t * V_next_t - V(obs_t)
A_t            = delta_t + gamma_t * lambda_t * continuation_mask_t * A_{t+1}
value_target_t = stop_gradient(A_t + V_old(obs_t))
```

The defaults are the ``PLAN_SPEC.json`` ``ppo`` block values
(``gamma_time_constant_seconds = 200``, ``reference_hz = 60``,
``lambda_at_two_frames = 0.97``); ``reference_hz`` is reused from
:mod:`touhou_rl.v430.contracts` so the two constants cannot drift.

Fail-closed: every malformed input raises :class:`SequenceError` instead of
returning a sentinel that a caller could mistake for a number.

Open points for the next slice are marked ``TODO(W03C)``; the ones this slice
inherited (``TODO(W03B)``) are answered in
``Document/Update/Alpha/v4.3.0/evidence/W03/W03B.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from touhou_rl.v430.contracts import REFERENCE_HZ

__all__ = [
    "SequenceError",
    "TAU_S",
    "REFERENCE_HZ",
    "LAMBDA_BASE",
    "ROLLOUT_DECISIONS_PER_ENV",
    "BURN_IN_DECISIONS",
    "LOSS_UNROLL_DECISIONS",
    "SEQUENCES_PER_MINIBATCH",
    "EFFECTIVE_MINIBATCH_STEPS",
    "discounts_and_lambdas",
    "gae",
    "segment_episode",
    "segment_batch",
    "padding_statistics",
]


class SequenceError(ValueError):
    """Raised for any malformed discount/GAE input (fail-closed)."""


#: ``PLAN_SPEC.ppo.gamma_time_constant_seconds`` — gamma time constant (s).
TAU_S: float = 200.0
#: ``PLAN_SPEC.ppo.reference_hz`` — reference decision rate (Hz); imported from
#: :mod:`touhou_rl.v430.contracts`, where it is ``REFERENCE_HZ = 60``.
#: ``PLAN_SPEC.ppo.lambda_at_two_frames`` — lambda at the two-frame step.
LAMBDA_BASE: float = 0.97

#: ``PLAN_SPEC.ppo.rollout_decisions_per_env`` — one env's rollout length; the
#: default segment shape (``512 = 8 * 64``) is what makes one env's rollout
#: exactly ``sequences_per_minibatch`` full loss windows when the episode does
#: not end earlier.
ROLLOUT_DECISIONS_PER_ENV: int = 512
#: ``PLAN_SPEC.ppo.burn_in_decisions`` — default burn-in context of
#: :func:`segment_episode` (a **maximum**: an episode start gets only the steps
#: that really exist, never padding).
BURN_IN_DECISIONS: int = 64
#: ``PLAN_SPEC.ppo.loss_unroll_decisions`` — default loss window length of
#: :func:`segment_episode`.
LOSS_UNROLL_DECISIONS: int = 64
#: ``PLAN_SPEC.ppo.sequences_per_minibatch`` — default grouping of
#: :func:`segment_batch`.
SEQUENCES_PER_MINIBATCH: int = 8
#: ``PLAN_SPEC.ppo.effective_minibatch_steps`` — ``8 * 64 = 512`` real loss
#: decisions per minibatch; **training steps, not tensor cells** (see
#: :func:`padding_statistics`).
EFFECTIVE_MINIBATCH_STEPS: int = 512


def _require_positive_int(value: Any, name: str, *,
                          hint: str = "effective_frames are whole frames") -> int:
    """Return ``value`` as a positive ``int`` or raise :class:`SequenceError`."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise SequenceError(
            f"{name} must be an int, got {value!r} ({type(value).__name__}); "
            f"{hint}")
    if value <= 0:
        raise SequenceError(
            f"{name} must be positive, got {value}; a decision that consumed the "
            f"clock cannot have a non-positive frame count")
    return value


def _require_non_negative_int(value: Any, name: str, *,
                             hint: str = "burn-in may be 0 (no context)") -> int:
    """Return ``value`` as a non-negative ``int`` or raise :class:`SequenceError`."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise SequenceError(
            f"{name} must be an int, got {value!r} ({type(value).__name__}); "
            f"{hint}")
    if value < 0:
        raise SequenceError(
            f"{name} must be >= 0, got {value}; {hint}")
    return value


def _require_positive_finite(value: Any, name: str) -> float:
    """Return ``value`` as a positive finite ``float`` or raise."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SequenceError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SequenceError(f"{name} {number!r} is not finite")
    if number <= 0.0:
        raise SequenceError(f"{name} must be positive, got {number}")
    return number


def _require_lambda_base(value: Any) -> float:
    """Return ``value`` as a float in the open interval ``(0, 1)`` or raise."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SequenceError(f"lambda_base must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SequenceError(f"lambda_base {number!r} is not finite")
    if not 0.0 < number < 1.0:
        raise SequenceError(
            f"lambda_base must be in the open interval (0, 1), got {number}")
    return number


def _is_sequence(value: Any) -> bool:
    """True for list/tuple/other non-str, non-mapping iterables."""

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    return hasattr(value, "__iter__")


def _gammas_lambdas(frames: list[int], tau: float, hz: float,
                    lambda_base: float) -> tuple[list[float], list[float]]:
    """The scalar formulas, applied to an already validated frame list."""

    horizon = hz * tau
    gammas = [math.exp(-count / horizon) for count in frames]
    lambdas = [lambda_base ** (count / 2.0) for count in frames]
    return gammas, lambdas


def discounts_and_lambdas(effective_frames: Any, *, tau_s: float = TAU_S,
                          hz: int = REFERENCE_HZ,
                          lambda_base: float = LAMBDA_BASE
                          ) -> tuple[Any, Any]:
    """Return ``(gamma, lambda)`` for one decision, or lists for a sequence.

    ``gamma_t = exp(-effective_frames_t / (hz * tau_s))`` and
    ``lambda_t = lambda_base ** (effective_frames_t / 2)``.  With the defaults a
    two-frame step gives ``gamma = exp(-2/12000)`` and ``lambda = 0.97``; a
    four-frame step gives ``lambda = 0.97 ** 2``.

    ``effective_frames`` may be a single ``int`` (result: two floats) or a
    sequence of ``int`` (result: two lists of the same length).  The output
    shape therefore matches the input shape, as §7.2's per-decision indexing
    assumes: it is always index-aligned with the transition it came from.

    Fail-closed: a non-integer frame count (including ``2.0`` and ``True``), a
    frame count ``<= 0``, ``tau_s <= 0``, ``hz <= 0`` or a ``lambda_base``
    outside ``(0, 1)`` raises :class:`SequenceError`.
    """

    tau = _require_positive_finite(tau_s, "tau_s")
    # ``hz`` is an int in the contract (``contracts._require_int``): a fractional
    # rate would make ``effective_frames = hz * seconds`` a float and hide the
    # fact that the frame count is no longer whole frames.
    rate = float(_require_positive_int(
        hz, "hz",
        hint="the reference rate is a whole-frame rate and must not be a bool"))
    lam_base = _require_lambda_base(lambda_base)

    if _is_sequence(effective_frames):
        frames = [_require_positive_int(item, f"effective_frames[{index}]")
                  for index, item in enumerate(effective_frames)]
        return _gammas_lambdas(frames, tau, rate, lam_base)

    single = _require_positive_int(effective_frames, "effective_frames")
    gammas, lambdas = _gammas_lambdas([single], tau, rate, lam_base)
    return gammas[0], lambdas[0]


def _field(record: Any, name: str, *, default: Any = ..., index: int = -1) -> Any:
    """Read ``name`` from a dict-like or attribute-like transition record."""

    if isinstance(record, Mapping):
        if name in record:
            return record[name]
        if default is not ...:
            return default
        raise SequenceError(f"transitions[{index}] is missing {name!r}")
    value = getattr(record, name, ...)
    if value is not ...:
        return value
    if default is not ...:
        return default
    raise SequenceError(f"transitions[{index}] is missing {name!r}")


def _require_bool(value: Any, name: str, index: int) -> bool:
    """A mask flag must be a real ``bool``: ``0``/``1``/``None`` are refused."""

    if not isinstance(value, bool):
        raise SequenceError(
            f"transitions[{index}].{name} must be a bool, got {value!r}; "
            f"0/1 or None would silently change the bootstrap mask")
    return value


def _require_real(value: Any, name: str, index: int) -> float:
    """A reward/value must be a finite real number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SequenceError(f"transitions[{index}].{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SequenceError(f"transitions[{index}].{name} {number!r} is not finite")
    return number


def _episode_id(record: Any, index: int) -> Any:
    """The episode key; ``None`` is refused because it would join everything."""

    episode_id = _field(record, "episode_id", index=index)
    if episode_id is None:
        raise SequenceError(
            f"transitions[{index}].episode_id is None; a missing episode key "
            f"cannot be used to cut the advantage recursion")
    return episode_id


def _step_masks(records: list[Any], values: list[Any],
                index: int) -> tuple[int, float, int]:
    """Return ``(bootstrap_mask, V_next, continuation_mask)`` for one step.

    This is the single place that decides §7.2's masks, so the forward and
    backward passes cannot disagree about them.

    ``segment_end`` is read **only** to validate that it is a real ``bool``: it
    is a *data-layout* fact (where the rollout writer cut the tensor), not an
    episode-end fact, so it **does not** change either mask.  Treating a
    segment boundary as a truncation would silently zero the continuation and
    change every advantage in front of it — see :func:`segment_episode`, which
    cuts segments without rewriting any flag.
    """

    record = records[index]
    terminated = _require_bool(_field(record, "terminated", index=index),
                               "terminated", index)
    truncated = _require_bool(_field(record, "truncated", index=index),
                              "truncated", index)
    _require_bool(_field(record, "segment_end", default=False, index=index),
                  "segment_end", index)
    if terminated and truncated:
        raise SequenceError(
            f"transitions[{index}] has terminated and truncated both true; a "
            f"terminal end and a sampling interruption are mutually exclusive")

    has_successor = index + 1 < len(records)
    same_episode_successor = (
        has_successor
        and _episode_id(records[index + 1], index + 1) == _episode_id(record, index))

    if terminated:
        # The terminal reward *is* the value target: no V(next_obs) term and no
        # continuation into a next transition.
        return 0, 0.0, 0

    if truncated:
        # Bootstrapping is legal, but only from the last genuine observation,
        # which the caller supplies as terminal_value (there is no fallback to
        # the post-reset next observation).
        terminal_value = _field(record, "terminal_value", default=None, index=index)
        if terminal_value is None:
            raise SequenceError(
                f"transitions[{index}] has truncated=True but no terminal_value; "
                f"bootstrapping an interrupted step without its last genuine "
                f"value would silently use 0")
        return 1, _require_real(terminal_value, "terminal_value", index), 0

    if same_episode_successor:
        # The ordinary step bootstraps from the next observation's value, which
        # the caller already evaluated as values[t+1].
        return 1, _require_real(values[index + 1], "values", index + 1), 1

    # No same-episode successor.
    if has_successor:
        raise SequenceError(
            f"transitions[{index}].episode_id="
            f"{_episode_id(record, index)!r} is followed by a different "
            f"episode_id but the step is flagged neither terminated nor "
            f"truncated; how an episode ended has to be recorded, it cannot be "
            f"inferred from the sample order")
    terminal_value = _field(record, "terminal_value", default=None, index=index)
    if terminal_value is None:
        # Data simply ends here: there is no successor value to bootstrap from.
        return 0, 0.0, 0
    return 1, _require_real(terminal_value, "terminal_value", index), 0


def gae(transitions: Any, values: Any, *, tau_s: float = TAU_S,
        hz: int = REFERENCE_HZ, lambda_base: float = LAMBDA_BASE) -> dict[str, list]:
    """Generalised advantage estimation over one contiguous sample (§7.2).

    Parameters
    ----------
    transitions:
        Sequence of dict-or-object records with ``episode_id``, ``reward``,
        ``terminated``, ``truncated`` and ``effective_frames``.  The optional
        ``terminal_value`` is the bootstrap value at a truncation (``V`` on the
        last genuine observation) and is *required* when the step truncates.
        ``segment_end`` is accepted and validated (it must be a real ``bool``)
        but **changes nothing**: a segment boundary is a data-layout cut, not an
        episode end, so it neither bootstraps nor breaks the recursion.  A
        caller that *does* want a bootstrap at a cut must say so with
        ``truncated=True`` *and* supply ``terminal_value``, or end the sequence
        there with a ``terminal_value`` — never by abusing ``segment_end``.
    values:
        ``V(obs_t)`` for every transition, same length.

    Returns
    -------
    dict with

    * ``advantages`` — ``A_t``, the **un-normalised** GAE advantage (any
      normalisation is a later, whole-minibatch step and is not done here);
    * ``value_targets`` — ``stop_gradient(A_t + V_old(obs_t))``.  This module has
      no tensor library, so the "stop gradient" part is the caller's obligation:
      here it is exactly ``A_t + V_t`` and must not be differentiated through
      ``V_t``;
    * ``deltas`` — the TD residual ``delta_t`` before the recursion;
    * ``valid`` — ``True`` for every emitted step.  A step is either computed or
      the call raises; this slice never pads, so the mask only gives a future
      segmenting caller a place to put its padding.

    Masks (§7.2)
    ------------
    * ``terminated=True`` → ``bootstrap_mask=0``, ``continuation_mask=0``;
    * ``truncated=True`` → ``bootstrap_mask=1`` with ``V_next=terminal_value``,
      ``continuation_mask=0``;
    * ordinary step → ``bootstrap_mask=1`` with ``V_next=values[t+1]``,
      ``continuation_mask=1``;
    * ``episode_id`` change → ``continuation_mask=0``; the previous episode's
      advantage is never carried across the boundary;
    * last element of the sample → ``continuation_mask=0``, and when it is
      flagged neither ``terminated`` nor ``truncated`` also
      ``bootstrap_mask=0`` (there is nothing to bootstrap from) — unless the
      record carries an explicit ``terminal_value``, in which case that value is
      the bootstrap and ``bootstrap_mask=1``.

    Deliberate strictness (kept, not relaxed)
    -----------------------------------------
    A mid-sample ``episode_id`` boundary whose last step is flagged neither
    ``terminated`` nor ``truncated`` is **rejected** rather than silently zeroed
    or bootstrapped: the caller has to say how that episode ended.  Segmenting
    never hides that error — :func:`segment_episode` only accepts a single
    episode and never rewrites the end flags.  ``segment_end`` is **not** an
    end flag that satisfies this rule (see ``W03B.md`` §4).
    """

    tau = _require_positive_finite(tau_s, "tau_s")
    rate = float(_require_positive_int(
        hz, "hz",
        hint="the reference rate is a whole-frame rate and must not be a bool"))
    lam_base = _require_lambda_base(lambda_base)

    if isinstance(transitions, (str, bytes, bytearray, Mapping)):
        raise SequenceError("transitions must be a sequence of records, not a mapping")
    records = list(transitions)
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise SequenceError("values must be a sequence of numbers, not a mapping")
    value_list = list(values)
    if len(records) != len(value_list):
        raise SequenceError(
            f"len(transitions)={len(records)} != len(values)={len(value_list)}; "
            f"V(obs_t) is needed for every transition")

    count = len(records)
    frames = [_require_positive_int(_field(record, "effective_frames", index=index),
                                    f"effective_frames[{index}]")
              for index, record in enumerate(records)]
    gammas, lambdas = _gammas_lambdas(frames, tau, rate, lam_base)

    deltas: list[float] = []
    continuations: list[int] = []
    for index, record in enumerate(records):
        reward = _require_real(_field(record, "reward", index=index), "reward", index)
        value_here = _require_real(value_list[index], "values", index)
        bootstrap_mask, v_next, continuation_mask = _step_masks(
            records, value_list, index)
        deltas.append(reward + gammas[index] * bootstrap_mask * v_next - value_here)
        continuations.append(continuation_mask)

    advantages: list[float] = [0.0] * count
    next_advantage = 0.0
    for index in range(count - 1, -1, -1):
        if continuations[index]:
            advantages[index] = (
                deltas[index] + gammas[index] * lambdas[index] * next_advantage)
        else:
            advantages[index] = deltas[index]
        next_advantage = advantages[index]

    value_targets = [
        advantages[index] + _require_real(value_list[index], "values", index)
        for index in range(count)]

    return {
        "advantages": advantages,
        "value_targets": value_targets,
        "deltas": deltas,
        "valid": [True] * count,
    }


# ---------------------------------------------------------------------------
# §7.1 序列训练 — segmentation and burn-in (slice W03B)
# ---------------------------------------------------------------------------

def _as_records(value: Any, name: str) -> list[Any]:
    """Materialise a container into a list, refusing mappings/str/scalars."""

    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise SequenceError(
            f"{name} must be a sequence of records, not a mapping or a string")
    try:
        return list(value)
    except TypeError:
        raise SequenceError(
            f"{name} must be an iterable of records, got {value!r}") from None


def _segment_int(segment: Any, name: str, index: int) -> int:
    """Read one integer bound out of a segment record (dict or attribute)."""

    value = _field(segment, name, index=index)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SequenceError(
            f"segments[{index}].{name} must be an int, got {value!r} "
            f"({type(value).__name__})")
    return value


def _mask_true_positions(mask: Any, name: str, index: int) -> list[int]:
    """Indices where a mask is ``True``; a non-``bool`` entry is refused.

    Refusing ``0``/``1``/``None`` here is the same rule
    :func:`_require_bool` applies to the transition flags: a mask that is not
    strictly boolean cannot be summed into a denominator safely.
    """

    if isinstance(mask, (str, bytes, bytearray, Mapping)):
        raise SequenceError(
            f"segments[{index}].{name} must be a sequence of bools, got {mask!r}")
    try:
        entries = list(mask)
    except TypeError:
        raise SequenceError(
            f"segments[{index}].{name} must be iterable, got {mask!r}") from None
    positions: list[int] = []
    for offset, entry in enumerate(entries):
        if not isinstance(entry, bool):
            raise SequenceError(
                f"segments[{index}].{name}[{offset}] must be a bool, got "
                f"{entry!r}")
        if entry:
            positions.append(offset)
    return positions


def segment_episode(transitions: Any, *, burn_in: int = BURN_IN_DECISIONS,
                    unroll: int = LOSS_UNROLL_DECISIONS
                    ) -> list[dict[str, Any]]:
    """Cut **one** episode into loss windows, each with its burn-in context.

    ``DEVELOPMENT_GUIDE`` §7.1: *每环境 rollout 512 决策，切为 64 决策 loss 段，
    加前 64 步 burn-in 上下文；cold start 不足 64 步时按真实有效 mask 填充；
    跨 episode 不得串接。burn-in 更新 hidden 但不计 loss。*

    The cut is purely positional (``loss_start`` advances by ``unroll`` from
    ``0``), so the result is a deterministic function of ``len(transitions)``:

    ```text
    loss_start  = k * unroll                     for k = 0, 1, 2, ...
    loss_end    = min(loss_start + unroll, n)    (the tail may be short)
    context_start = max(0, loss_start - burn_in) (never negative)
    ```

    Guarantees
    ----------
    * ``loss_steps == loss_end - loss_start == min(unroll, n - loss_start)``:
      a loss window is **exactly** the real remaining steps of the episode, so
      the tail may be shorter than ``unroll`` but is never padded up to it.
    * ``context_steps == loss_start - context_start <= burn_in``: the context is
      a **maximum**.  At the episode start only the steps that really exist are
      marked (``0`` for the first window), never a synthetic prefix.  Hence
      ``context_mask`` is a *real-data* mask, as §7.1 demands for a cold start.
    * ``segment_index`` is ``0, 1, 2, ...`` in increasing order and
      ``loss_start`` is non-decreasing, so **the order inside a segment equals
      the input order** (this function never shuffles and never sorts).
    * Loss windows **never overlap**: ``segment[k]["loss_end"] <=
      segment[k+1]["loss_start"]`` (they are adjacent, so the union of the loss
      windows is the whole episode).  The *context* of a segment may overlap the
      previous segment's loss window — that is the design intent, since the
      hidden state has to be rebuilt from real steps and those steps are also
      real data.
    * **One episode only**: this function never joins two episodes.  If the input
      carries more than one ``episode_id`` it raises :class:`SequenceError`
      instead of cutting across the boundary — §7.1's *跨 episode 不得串接* is
      enforced, not documented.

    Segment boundaries and ``segment_end`` (final decision, W03B)
    ------------------------------------------------------------
    A segmentation boundary is **not** a truncation and this function does
    **not** rewrite ``terminated`` / ``truncated`` / ``segment_end``: it only
    reports positions.  Cutting the tensor is a data-layout fact; the task did
    not end there.  Consequences:

    * running :func:`gae` over the episode with or without the boundaries gives
      **bit-identical** advantages (a test asserts exact ``==``);
    * if a caller wants a bootstrap value at a cut, it must say so explicitly —
      ``truncated=True`` plus ``terminal_value``, or the sequence ending there
      with a ``terminal_value``.  Abusing ``segment_end`` for it was the
      previous slice's stop-gap and is gone.

    Parameters
    ----------
    transitions:
        The whole decision sequence of **one** episode (dict or object records
        with at least ``episode_id``).  Empty input is refused: there is no
        episode to segment.
    burn_in:
        Maximum burn-in context in decisions (``>= 0``; ``0`` means no context).
    unroll:
        Loss window length in decisions (``> 0``).

    Returns
    -------
    list of dicts, one per segment, each with

    ``episode_id``, ``segment_index``, ``episode_steps`` (``n``),
    ``context_start``, ``loss_start``, ``loss_end``,
    ``context_steps``, ``loss_steps``,
    ``steps`` (the whole window ``loss_end - context_start``),
    ``context_mask`` / ``loss_mask`` (tuples of ``n`` bools).

    Both masks are **full-episode length** and index-aligned with the input
    list and with :func:`gae`'s output arrays, so a segment can be consumed by
    plain positional slicing (``transitions[context_start:loss_end]``) without
    ever indexing the wrong decision; ``context_mask`` is ``True`` exactly on
    ``[context_start, loss_start)`` (burn-in: feeds the hidden state, **never**
    the loss) and ``loss_mask`` exactly on ``[loss_start, loss_end)``.  The two
    masks are disjoint.
    """

    records = _as_records(transitions, "transitions")
    if not records:
        raise SequenceError(
            "transitions is empty; there is no episode to segment (an empty "
            "episode is a recording error, not a segment list)")
    burn = _require_non_negative_int(burn_in, "burn_in")
    step = _require_positive_int(
        unroll, "unroll",
        hint="the loss unroll is a whole number of decisions")

    episode_id = _episode_id(records[0], 0)
    for index in range(1, len(records)):
        other = _episode_id(records[index], index)
        if other != episode_id:
            raise SequenceError(
                f"transitions[{index}].episode_id={other!r} differs from "
                f"transitions[0].episode_id={episode_id!r}; segment_episode "
                f"takes exactly one episode and must never join two (§7.1: "
                f"cross-episode concatenation is forbidden)")

    count = len(records)
    segments: list[dict[str, Any]] = []
    loss_start = 0
    while loss_start < count:
        loss_end = min(loss_start + step, count)
        context_start = max(0, loss_start - burn)
        segments.append({
            "episode_id": episode_id,
            "segment_index": len(segments),
            "episode_steps": count,
            "context_start": context_start,
            "loss_start": loss_start,
            "loss_end": loss_end,
            "context_steps": loss_start - context_start,
            "loss_steps": loss_end - loss_start,
            "steps": loss_end - context_start,
            "context_mask": tuple(
                context_start <= index < loss_start for index in range(count)),
            "loss_mask": tuple(
                loss_start <= index < loss_end for index in range(count)),
        })
        loss_start = loss_end
    return segments


def segment_batch(episodes: Any, *, burn_in: int = BURN_IN_DECISIONS,
                  unroll: int = LOSS_UNROLL_DECISIONS,
                  sequences_per_minibatch: int = SEQUENCES_PER_MINIBATCH
                  ) -> list[dict[str, Any]]:
    """Segment several episodes and group the segments into minibatches.

    ``episodes`` is either a mapping ``{episode_id: transitions}`` or an
    iterable of ``(episode_id, transitions)`` pairs.  Episode order is the input
    order (a mapping's insertion order), and inside every episode the decision
    order is the input order, so the whole operation is deterministic and
    **never shuffles** (shuffling a minibatch is the sampler's job, not this
    pure function's).

    Each episode is validated to contain exactly its own ``episode_id`` — a
    record whose ``episode_id`` disagrees with the pair key raises
    :class:`SequenceError`, because a mislabelled episode would silently join
    two episodes inside one sequence.

    Returns
    -------
    list of dicts, one per minibatch group (consecutive segments, in the order
    produced above):

    ``minibatch_index``, ``segments`` (tuple of segment dicts),
    ``sequences``, ``effective_steps`` (Σ ``loss_steps`` — the real loss
    denominator), ``context_steps`` (Σ, burn-in only), ``window_steps``
    (Σ ``steps``, i.e. the cells that carry real data), ``pad_width``
    (``max(window steps)``: the rectangular width a tensor batch would need) and
    ``is_full`` (``sequences == sequences_per_minibatch``).

    The last group may be short; it is returned with ``is_full=False`` rather
    than dropped, so a caller that needs 8 sequences per minibatch has to check
    the flag (dropping silently would change the step count of an epoch).

    With the ``PLAN_SPEC`` defaults one env's non-ending rollout of 512
    decisions yields ``8`` full windows of ``64`` loss decisions, i.e. exactly
    ``sequences_per_minibatch * loss_unroll_decisions ==
    effective_minibatch_steps == 512`` real training steps.
    """

    burn = _require_non_negative_int(burn_in, "burn_in")
    step = _require_positive_int(
        unroll, "unroll",
        hint="the loss unroll is a whole number of decisions")
    per_group = _require_positive_int(
        sequences_per_minibatch, "sequences_per_minibatch",
        hint="a minibatch holds at least one sequence")

    if isinstance(episodes, Mapping):
        pairs: list[Any] = list(episodes.items())
    else:
        pairs = _as_records(episodes, "episodes")
    if not pairs:
        raise SequenceError(
            "episodes is empty; there is nothing to segment")

    seen: list[Any] = []
    ordered: list[tuple[Any, list[Any]]] = []
    for index, pair in enumerate(pairs):
        entries = _as_records(pair, f"episodes[{index}]")
        if len(entries) != 2:
            raise SequenceError(
                f"episodes[{index}] must be an (episode_id, transitions) pair, "
                f"got {len(entries)} item(s)")
        episode_id, records = entries[0], entries[1]
        if episode_id is None:
            raise SequenceError(
                f"episodes[{index}] has episode_id=None; a missing episode key "
                f"cannot order the minibatches")
        if seen:
            for earlier in seen:
                if earlier == episode_id:
                    raise SequenceError(
                        f"episodes[{index}] repeats episode_id={episode_id!r}; a "
                        f"duplicate key would segment the same episode twice")
        seen.append(episode_id)
        episode_records = _as_records(records, f"episodes[{index}][1]")
        if not episode_records:
            raise SequenceError(
                f"episodes[{index}] (episode_id={episode_id!r}) is empty; an "
                f"empty episode is a recording error")
        for offset, record in enumerate(episode_records):
            record_id = _episode_id(record, offset)
            if record_id != episode_id:
                raise SequenceError(
                    f"episodes[{index}][1][{offset}].episode_id={record_id!r} "
                    f"does not match the episode key {episode_id!r}; a "
                    f"mislabelled record would join two episodes")
        ordered.append((episode_id, episode_records))

    flat: list[dict[str, Any]] = []
    for _episode_id_value, episode_records in ordered:
        flat.extend(segment_episode(episode_records, burn_in=burn, unroll=step))

    groups: list[dict[str, Any]] = []
    for start in range(0, len(flat), per_group):
        chunk = flat[start:start + per_group]
        groups.append({
            "minibatch_index": len(groups),
            "segments": tuple(chunk),
            "sequences": len(chunk),
            "effective_steps": sum(segment["loss_steps"] for segment in chunk),
            "context_steps": sum(segment["context_steps"] for segment in chunk),
            "window_steps": sum(segment["steps"] for segment in chunk),
            "pad_width": max(segment["steps"] for segment in chunk),
            "is_full": len(chunk) == per_group,
        })
    return groups


def padding_statistics(segments: Any, *,
                       pad_to: Any = None) -> dict[str, Any]:
    """Count real vs masked steps of a segment batch, failing closed on drift.

    This is the function that answers §7.1's *padding 不得进入优势/熵统计*
    with two numbers instead of a promise:

    * ``effective_steps`` — Σ ``loss_steps``, the **only** denominator an
      advantage or entropy statistic may divide by;
    * ``naive_denominator`` — ``sequences * pad_width``, the denominator you get
      if every cell of the padded rectangular batch is treated as a valid step.

    ``masked_steps = naive_denominator - effective_steps`` is therefore the
    number of cells that must stay out of the statistics (padding cells plus the
    burn-in context, which updates the hidden state but is never a loss step).
    ``padding_slots = naive_denominator - window_steps`` counts only the cells
    with no data at all.

    ``pad_to`` is the rectangular **window** width to assume, in decisions of
    context *plus* loss (so a full ``burn_in=64`` / ``unroll=64`` row is ``128``
    wide, not ``64``); it defaults to the widest segment (``max(steps)``), i.e.
    no padding at all.  A ``pad_to`` narrower than the widest segment raises
    :class:`SequenceError` — padding may never truncate a real step.

    Fail-closed checks (each raises :class:`SequenceError`)
    -----------------------------------------------------
    * a missing/ non-int bound, or ``not 0 <= context_start <= loss_start <
      loss_end``;
    * ``loss_mask`` whose ``True`` positions are not exactly
      ``[loss_start, loss_end)`` — a mask that leaks a padding cell into the
      loss is rejected, not reported;
    * ``context_mask`` whose ``True`` positions are not exactly
      ``[context_start, loss_start)``;
    * a non-``bool`` mask entry, or ``episode_steps < loss_end``.

    Returns
    -------
    dict with ``sequences``, ``window_steps``, ``context_steps``,
    ``effective_steps``, ``loss_mask_true``, ``context_mask_true`` (the last two
    recomputed from the masks as a cross-check and equal to the sums above),
    ``pad_width``, ``naive_denominator``, ``padding_slots``, ``masked_steps``.
    """

    records = _as_records(segments, "segments")
    if not records:
        raise SequenceError("segments is empty; there is nothing to count")

    sequences = len(records)
    window_steps = 0
    context_steps = 0
    effective_steps = 0
    context_mask_true = 0
    loss_mask_true = 0
    widths: list[int] = []

    for index, segment in enumerate(records):
        context_start = _segment_int(segment, "context_start", index)
        loss_start = _segment_int(segment, "loss_start", index)
        loss_end = _segment_int(segment, "loss_end", index)
        if not 0 <= context_start <= loss_start < loss_end:
            raise SequenceError(
                f"segments[{index}] has context_start={context_start}, "
                f"loss_start={loss_start}, loss_end={loss_end}; the bounds must "
                f"satisfy 0 <= context_start <= loss_start < loss_end")
        episode_steps = _field(segment, "episode_steps", default=None, index=index)
        if episode_steps is not None:
            if (isinstance(episode_steps, bool)
                    or not isinstance(episode_steps, int)
                    or episode_steps < loss_end):
                raise SequenceError(
                    f"segments[{index}].episode_steps={episode_steps!r} must be "
                    f"an int >= loss_end={loss_end}")

        expected_loss = list(range(loss_start, loss_end))
        expected_context = list(range(context_start, loss_start))
        observed_loss = _mask_true_positions(
            _field(segment, "loss_mask", index=index), "loss_mask", index)
        observed_context = _mask_true_positions(
            _field(segment, "context_mask", index=index), "context_mask", index)
        if observed_loss != expected_loss:
            raise SequenceError(
                f"segments[{index}].loss_mask marks {observed_loss!r} but the "
                f"declared loss window is {expected_loss!r}; a mask like that "
                f"would let a padding cell into the loss statistics")
        if observed_context != expected_context:
            raise SequenceError(
                f"segments[{index}].context_mask marks {observed_context!r} but "
                f"the declared burn-in window is {expected_context!r}; a mask "
                f"like that would change the statistics through the hidden state")

        widths.append(loss_end - context_start)
        window_steps += loss_end - context_start
        context_steps += loss_start - context_start
        effective_steps += loss_end - loss_start
        context_mask_true += len(observed_context)
        loss_mask_true += len(observed_loss)

    if loss_mask_true != effective_steps or context_mask_true != context_steps:
        raise SequenceError(
            f"mask totals disagree with the declared windows: loss_mask_true="
            f"{loss_mask_true} vs effective_steps={effective_steps}, "
            f"context_mask_true={context_mask_true} vs context_steps="
            f"{context_steps}")

    widest = max(widths)
    if pad_to is None:
        width = widest
    else:
        width = _require_positive_int(
            pad_to, "pad_to", hint="pad_to is a tensor width in decisions")
        if width < widest:
            raise SequenceError(
                f"pad_to={width} is narrower than the widest segment "
                f"({widest} steps); padding may never truncate a real step")

    naive_denominator = sequences * width
    return {
        "sequences": sequences,
        "window_steps": window_steps,
        "context_steps": context_steps,
        "effective_steps": effective_steps,
        "loss_mask_true": loss_mask_true,
        "context_mask_true": context_mask_true,
        "pad_width": width,
        "naive_denominator": naive_denominator,
        "padding_slots": naive_denominator - window_steps,
        "masked_steps": naive_denominator - effective_steps,
    }


# TODO(W03C): the hidden state itself is not modelled here.  ``segment_episode``
# hands out real-step context masks, but the exact-prefix recomputation and the
# "burn-in approximation error" measurement of §7.1 need the recurrent model and
# belong to the model/training slice.
# TODO(W03C): ``segment_batch`` groups by consecutive segments; whether the
# sampler should shuffle *whole* minibatch groups (never inside a sequence) is
# the rollout/sampler slice's decision, not this pure function's.

