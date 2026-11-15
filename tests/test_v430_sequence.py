"""W03A synthetic unit tests for :mod:`touhou_rl.v430.sequence`.

**These are synthetic tests, not evidence of game capability.**  They exercise
two pure functions (``discounts_and_lambdas`` / ``gae``) on hand-built records;
no engine, no game asset, no dataset, no model and no training run is involved.
Passing this file says nothing about TH08 play strength.

Every expected number below is recomputed **in the test itself from the §7.2
formulas written out longhand**, never by calling the implementation back.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest

from touhou_rl.v430 import sequence
from touhou_rl.v430.sequence import (
    BURN_IN_DECISIONS,
    EFFECTIVE_MINIBATCH_STEPS,
    LAMBDA_BASE,
    LOSS_UNROLL_DECISIONS,
    REFERENCE_HZ,
    ROLLOUT_DECISIONS_PER_ENV,
    SEQUENCES_PER_MINIBATCH,
    TAU_S,
    SequenceError,
    discounts_and_lambdas,
    gae,
    padding_statistics,
    segment_batch,
    segment_episode,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_SPEC = REPO_ROOT / "Document" / "Update" / "Alpha" / "v4.3.0" / "PLAN_SPEC.json"

#: ``gamma`` at the two-frame reference step, written out longhand.
GAMMA_2 = math.exp(-2.0 / (60.0 * 200.0))
#: ``lambda`` at the two-frame reference step.
LAMBDA_2 = 0.97


def _transition(*, episode_id="ep", reward, terminated=False, truncated=False,
                effective_frames=2, terminal_value=None, segment_end=None):
    """Build one plain dict transition; ``terminal_value`` is omitted if None."""

    record = {
        "episode_id": episode_id,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "effective_frames": effective_frames,
    }
    if terminal_value is not None:
        record["terminal_value"] = terminal_value
    if segment_end is not None:
        record["segment_end"] = segment_end
    return record


# ---------------------------------------------------------------------------
# 1-2. discount/lambda values at the two- and four-frame steps
# ---------------------------------------------------------------------------

def test_two_frame_step_matches_exp_and_lambda_base():
    gamma, lam = discounts_and_lambdas(2)
    # gamma_2 = exp(-2 / (60 * 200)) = exp(-1/6000) ~= 0.9998333472214507
    assert abs(gamma - math.exp(-2 / 12000)) < 1e-15
    assert gamma < 1.0
    # lambda_2 = 0.97 ** (2/2) = 0.97 exactly at the reference step
    assert abs(lam - 0.97) < 1e-15


def test_four_frame_step_squares_lambda():
    gamma, lam = discounts_and_lambdas(4)
    # lambda_4 = 0.97 ** (4/2) = 0.97 ** 2 = 0.9409
    assert abs(lam - 0.97 ** 2) < 1e-15
    # gamma_4 = exp(-4 / 12000)
    assert abs(gamma - math.exp(-4 / 12000)) < 1e-15
    # A four-frame decision covers more time, so its gamma must be smaller.
    assert gamma < math.exp(-2 / 12000)


def test_sequence_input_is_index_aligned():
    gammas, lambdas = discounts_and_lambdas([2, 4, 8])
    assert len(gammas) == len(lambdas) == 3
    for index, frames in enumerate((2, 4, 8)):
        assert abs(gammas[index] - math.exp(-frames / 12000)) < 1e-15
        assert abs(lambdas[index] - 0.97 ** (frames / 2)) < 1e-15


# ---------------------------------------------------------------------------
# 3. fail-closed inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_frames", [0, -1, -2, 2.5, 2.0, True, None, "2"])
def test_bad_effective_frames_raise(bad_frames):
    with pytest.raises(SequenceError):
        discounts_and_lambdas(bad_frames)


@pytest.mark.parametrize("bad_sequence", [[0], [-1], [2, 2.5], [2, 0]])
def test_bad_effective_frames_in_sequence_raise(bad_sequence):
    with pytest.raises(SequenceError):
        discounts_and_lambdas(bad_sequence)


@pytest.mark.parametrize("kwargs", [
    # tau_s must be positive and finite: nan has no order, so it cannot be a
    # time constant either (nan > 0 is False, but nan <= 0 is also False, which
    # is exactly the kind of value that slips through a naive comparison).
    {"tau_s": 0.0}, {"tau_s": -200.0}, {"tau_s": float("inf")},
    {"tau_s": float("nan")},
    # hz is an int in the contract (``contracts._require_int``): a fractional
    # rate would turn the frame count into a float.  ``True`` is an ``int``
    # subclass, so it has to be refused explicitly.
    {"hz": 0}, {"hz": -60}, {"hz": 60.5}, {"hz": 60.0}, {"hz": "60"},
    {"hz": None}, {"hz": True}, {"hz": False},
    # lambda_base is strictly inside (0, 1): the endpoints are not discounts.
    {"lambda_base": 0.0}, {"lambda_base": 1.0}, {"lambda_base": 1.5},
    {"lambda_base": -0.97}, {"lambda_base": float("nan")},
])
def test_bad_hyperparameters_raise(kwargs):
    with pytest.raises(SequenceError):
        discounts_and_lambdas(2, **kwargs)


def test_bad_hyperparameters_raise_in_gae_too():
    transitions = [_transition(reward=1.0)]
    for kwargs in ({"hz": 60.5}, {"hz": True}, {"tau_s": float("nan")},
                   {"lambda_base": 1.0}):
        with pytest.raises(SequenceError):
            gae(transitions, [0.1], **kwargs)


# ---------------------------------------------------------------------------
# 4. single episode, no termination: a three-step sequence by hand
# ---------------------------------------------------------------------------

def test_single_episode_three_steps_by_hand():
    """Hand computation for a 3-step, 2-frame, non-terminated sequence.

    Inputs::
        t :  0      1      2
        r :  1.0    0.5   -0.25
        V :  0.3    0.4    0.2

    All steps consume ``effective_frames = 2``, so for every ``t``::

        gamma = G = exp(-2/12000) = 0.9998333472214507
        lambda = L = 0.97 ** 1 = 0.97

    Masks: steps 0 and 1 are ordinary in-episode steps (bootstrap from
    ``V[t+1]``, continue the recursion).  Step 2 is the tail of the sample with
    ``terminated = truncated = False``, so there is no ``V(3)``: it does **not**
    bootstrap and does **not** continue.

    Hand computation::

        d0 = r0 + G*V1 - V0 = 1.0  + G*0.4 - 0.3
        d1 = r1 + G*V2 - V1 = 0.5  + G*0.2 - 0.4
        d2 = r2 + 0    - V2 = -0.25          - 0.2   (bootstrap_mask = 0)

        A2 = d2
        A1 = d1 + G*L*A2
        A0 = d0 + G*L*A1

        value_target_t = A_t + V_t
    """

    g = GAMMA_2  # exp(-2/12000)
    lam = LAMBDA_2  # 0.97
    rewards = [1.0, 0.5, -0.25]
    values = [0.3, 0.4, 0.2]

    d0 = rewards[0] + g * values[1] - values[0]
    d1 = rewards[1] + g * values[2] - values[1]
    d2 = rewards[2] - values[2]

    a2 = d2
    a1 = d1 + g * lam * a2
    a0 = d0 + g * lam * a1

    expected_deltas = [d0, d1, d2]
    expected_advantages = [a0, a1, a2]
    expected_targets = [a0 + values[0], a1 + values[1], a2 + values[2]]

    transitions = [_transition(episode_id="single", reward=reward)
                   for reward in rewards]
    result = gae(transitions, values)

    assert result["deltas"] == expected_deltas
    assert result["advantages"] == expected_advantages
    assert result["value_targets"] == expected_targets
    assert result["valid"] == [True, True, True]


# ---------------------------------------------------------------------------
# 5. terminated last step does not bootstrap
# ---------------------------------------------------------------------------

def test_terminated_last_step_does_not_bootstrap():
    """``A_t = r_t - V_t`` on the terminated step: no ``gamma * V_next`` term."""

    values = [0.4, 0.9]
    transitions = [
        _transition(episode_id="term", reward=0.5),
        _transition(episode_id="term", reward=2.0, terminated=True),
    ]
    result = gae(transitions, values)

    # Last step: d = r + gamma * 0 * V_next - V, i.e. exactly r - V.
    assert result["deltas"][1] == 2.0 - 0.9
    assert result["advantages"][1] == 2.0 - 0.9
    # And it is not the bootstrapped value, which would have been larger.
    assert result["advantages"][1] != 2.0 + GAMMA_2 * values[1] - 0.9
    # The previous step still continues into it: d0 + G*L*A1.
    assert result["advantages"][0] == 0.5 + GAMMA_2 * values[1] - values[0] \
        + GAMMA_2 * LAMBDA_2 * (2.0 - 0.9)
    assert result["value_targets"][1] == (2.0 - 0.9) + 0.9  # == 2.0 - 0.0


# ---------------------------------------------------------------------------
# 6. truncated does bootstrap from terminal_value
# ---------------------------------------------------------------------------

def test_truncated_bootstraps_from_terminal_value():
    values = [0.3, 0.4]
    terminal_value = 2.0
    transitions = [
        _transition(episode_id="trunc", reward=0.5),
        _transition(episode_id="trunc", reward=1.0, truncated=True,
                    terminal_value=terminal_value),
    ]
    result = gae(transitions, values)

    expected_last = 1.0 + GAMMA_2 * 1 * terminal_value - 0.4
    assert result["deltas"][1] == expected_last
    assert result["advantages"][1] == expected_last
    # The truncation term is really in there.  (1.0 - 0.4) is what a
    # non-bootstrapping termination would have produced.
    assert result["advantages"][1] != 1.0 - 0.4
    assert result["advantages"][1] - (1.0 - 0.4) == GAMMA_2 * terminal_value
    # Mask is 0, so the recursion does not continue past the truncation.
    assert result["advantages"][0] == (
        0.5 + GAMMA_2 * values[1] - values[0]
        + GAMMA_2 * LAMBDA_2 * expected_last)


def test_truncated_without_terminal_value_raises():
    transitions = [
        _transition(episode_id="trunc", reward=0.5),
        _transition(episode_id="trunc", reward=1.0, truncated=True),
    ]
    with pytest.raises(SequenceError):
        gae(transitions, [0.3, 0.4])


def test_terminated_and_truncated_together_raise():
    transitions = [_transition(episode_id="x", reward=1.0, terminated=True,
                               truncated=True, terminal_value=1.0)]
    with pytest.raises(SequenceError):
        gae(transitions, [0.5])


# ---------------------------------------------------------------------------
# 7. cross-episode cut
# ---------------------------------------------------------------------------

def test_cross_episode_cut_matches_second_episode_alone():
    """Episode 2 must be bit-identical to running it on its own.

    Episode 1 ends with ``terminated=True`` (2 steps); episode 2 has three
    ordinary steps with mixed frame counts (2, 4, 2).  The concatenated run is
    compared **element by element with exact equality** against the standalone
    run of episode 2: any leak of episode 1's ``A`` (or of its ``gamma``/
    ``lambda``/``V``) through the boundary would change at least one float.
    """

    ep1 = [
        _transition(episode_id="ep1", reward=1.0),
        _transition(episode_id="ep1", reward=2.0, terminated=True),
    ]
    ep1_values = [0.1, 0.2]
    ep2 = [
        _transition(episode_id="ep2", reward=0.5, effective_frames=2),
        _transition(episode_id="ep2", reward=-1.0, effective_frames=4),
        _transition(episode_id="ep2", reward=0.25, effective_frames=2),
    ]
    ep2_values = [1.0, 0.5, 0.0]

    joined = gae(ep1 + ep2, ep1_values + ep2_values)
    alone = gae(ep2, ep2_values)

    for key in ("advantages", "value_targets", "deltas", "valid"):
        # Exact equality, not a tolerance: the arithmetic must be the same run.
        assert joined[key][2:] == alone[key], f"{key} leaked across episodes"

    # The last step of episode 1 is terminated, so it cannot reach into ep2;
    # episode 2 is then checked against its own longhand recursion.
    assert joined["advantages"][:2] == [
        1.0 + GAMMA_2 * ep1_values[1] - ep1_values[0]
        + GAMMA_2 * LAMBDA_2 * (2.0 - ep1_values[1]),
        2.0 - ep1_values[1],
    ]
    # Episode 2 step 0 (2 frames), step 1 (4 frames), step 2 (2 frames, tail).
    g4 = math.exp(-4 / 12000)
    l4 = 0.97 ** 2
    a4 = 0.25 - ep2_values[2]  # tail: no bootstrap, no continuation
    a3 = -1.0 + g4 * ep2_values[2] - ep2_values[1] + g4 * l4 * a4
    a2 = 0.5 + GAMMA_2 * ep2_values[1] - ep2_values[0] + GAMMA_2 * LAMBDA_2 * a3
    assert joined["advantages"][2] == a2
    assert joined["advantages"][3] == a3
    assert joined["advantages"][4] == a4


def test_mid_sample_episode_boundary_requires_an_end_flag():
    """An episode may not end mid-sample without saying how it ended."""

    transitions = [
        _transition(episode_id="ep1", reward=1.0),  # neither terminated nor ...
        _transition(episode_id="ep2", reward=1.0),  # ... truncated, new episode
    ]
    with pytest.raises(SequenceError):
        gae(transitions, [0.1, 0.2])


# ---------------------------------------------------------------------------
# 8. constants agree with PLAN_SPEC.json
# ---------------------------------------------------------------------------

def test_constants_match_plan_spec():
    ppo = json.loads(PLAN_SPEC.read_text(encoding="utf-8"))["ppo"]
    assert TAU_S == ppo["gamma_time_constant_seconds"] == 200.0
    assert REFERENCE_HZ == ppo["reference_hz"] == 60
    assert LAMBDA_BASE == ppo["lambda_at_two_frames"] == 0.97
    # Context constants recorded by this slice (segmentation is not implemented).
    assert sequence.ROLLOUT_DECISIONS_PER_ENV == ppo["rollout_decisions_per_env"] == 512
    assert sequence.BURN_IN_DECISIONS == ppo["burn_in_decisions"] == 64
    assert sequence.LOSS_UNROLL_DECISIONS == ppo["loss_unroll_decisions"] == 64


def test_default_arguments_are_the_contract_values():
    for function in (discounts_and_lambdas, gae):
        parameters = inspect.signature(function).parameters
        assert parameters["tau_s"].default == 200.0
        assert parameters["hz"].default == 60
        assert parameters["lambda_base"].default == 0.97


# ---------------------------------------------------------------------------
# shape / fail-closed behaviour of gae itself
# ---------------------------------------------------------------------------

def test_gae_rejects_length_mismatch():
    with pytest.raises(SequenceError):
        gae([_transition(reward=1.0)], [0.1, 0.2])
    with pytest.raises(SequenceError):
        gae([_transition(reward=1.0), _transition(reward=2.0)], [0.1])


def test_gae_rejects_non_bool_flags():
    bad = _transition(reward=1.0)
    bad["terminated"] = 0  # int instead of bool would change the mask silently
    with pytest.raises(SequenceError):
        gae([bad], [0.1])


def test_gae_returns_one_value_per_transition():
    transitions = [_transition(reward=0.1) for _ in range(5)]
    result = gae(transitions, [0.0, 0.1, 0.2, 0.3, 0.4])
    for key in ("advantages", "value_targets", "deltas", "valid"):
        assert len(result[key]) == 5
    assert set(result) == {"advantages", "value_targets", "deltas", "valid"}


# ===========================================================================
# W03B — §7.1 sequence segmentation / burn-in / padding (synthetic tests only)
#
# Every expected number below is derived in the test from the segment bounds
# (or read from PLAN_SPEC.json), never by calling the implementation back.
# ===========================================================================

def _episode(episode_id="ep", count=1, *, reward=0.0, effective_frames=2):
    """A flat episode of ``count`` ordinary (non-terminal) decisions."""

    return [_transition(episode_id=episode_id, reward=reward,
                        effective_frames=effective_frames)
            for _ in range(count)]


def _window_mask(count, start, end):
    """The expected full-episode-length mask for the half-open ``[start, end)``."""

    return tuple(start <= index < end for index in range(count))


# ---------------------------------------------------------------------------
# 9. 512 / 64 / 64 / 8 agree with PLAN_SPEC.json
# ---------------------------------------------------------------------------

def test_segmentation_constants_match_plan_spec():
    ppo = json.loads(PLAN_SPEC.read_text(encoding="utf-8"))["ppo"]
    assert sequence.ROLLOUT_DECISIONS_PER_ENV == ppo["rollout_decisions_per_env"] == 512
    assert sequence.BURN_IN_DECISIONS == ppo["burn_in_decisions"] == 64
    assert sequence.LOSS_UNROLL_DECISIONS == ppo["loss_unroll_decisions"] == 64
    assert sequence.SEQUENCES_PER_MINIBATCH == ppo["sequences_per_minibatch"] == 8
    assert sequence.EFFECTIVE_MINIBATCH_STEPS == ppo["effective_minibatch_steps"] == 512
    # The four numbers are not independent: they are the manifest's own
    # arithmetic, recomputed here from the JSON values (not from literals).
    assert (ppo["sequences_per_minibatch"] * ppo["loss_unroll_decisions"]
            == ppo["effective_minibatch_steps"])
    assert (ppo["rollout_decisions_per_env"] // ppo["loss_unroll_decisions"]
            == ppo["sequences_per_minibatch"])


def test_segmentation_defaults_are_the_contract_values():
    parameters = inspect.signature(segment_episode).parameters
    assert parameters["burn_in"].default == BURN_IN_DECISIONS
    assert parameters["unroll"].default == LOSS_UNROLL_DECISIONS
    batch_parameters = inspect.signature(segment_batch).parameters
    assert batch_parameters["burn_in"].default == BURN_IN_DECISIONS
    assert batch_parameters["unroll"].default == LOSS_UNROLL_DECISIONS
    assert (batch_parameters["sequences_per_minibatch"].default
            == SEQUENCES_PER_MINIBATCH)


# ---------------------------------------------------------------------------
# 10. 200 decisions / burn_in=64 / unroll=64 -> 64 + 64 + 64 + 8
# ---------------------------------------------------------------------------

def test_two_hundred_steps_give_windows_of_64_64_64_8():
    count = 200
    transitions = _episode(count=count)
    segments = segment_episode(transitions, burn_in=64, unroll=64)

    assert len(segments) == 4
    assert [segment["segment_index"] for segment in segments] == [0, 1, 2, 3]

    # (context_start, loss_start, loss_end, context_steps, loss_steps)
    expected = [
        (0, 0, 64, 0, 64),
        (0, 64, 128, 64, 64),
        (64, 128, 192, 64, 64),
        (128, 192, 200, 64, 8),
    ]
    for segment, (context_start, loss_start, loss_end,
                  context_steps, loss_steps) in zip(segments, expected):
        assert segment["episode_id"] == "ep"
        assert segment["episode_steps"] == count
        assert segment["context_start"] == context_start
        assert segment["loss_start"] == loss_start
        assert segment["loss_end"] == loss_end
        assert segment["context_steps"] == context_steps == loss_start - context_start
        assert segment["loss_steps"] == loss_steps == loss_end - loss_start
        # The loss window is exactly min(unroll, the episode's real remainder).
        assert segment["loss_steps"] == min(64, count - loss_start)
        assert segment["steps"] == loss_end - context_start
        # Masks are full-episode length, so positional slicing is unambiguous.
        assert len(segment["context_mask"]) == count
        assert len(segment["loss_mask"]) == count
        assert segment["context_mask"] == _window_mask(count, context_start, loss_start)
        assert segment["loss_mask"] == _window_mask(count, loss_start, loss_end)

    # The first window starts the episode: no real step exists before it, so the
    # context mask is empty instead of being padded up to 64.
    assert segments[0]["context_steps"] == 0
    assert segments[0]["context_mask"] == (False,) * count
    # From the second window on a full 64-step burn-in is available.
    assert all(segment["context_steps"] == 64 for segment in segments[1:])
    # The tail is short (8) and is *not* padded to 64.
    assert segments[-1]["loss_steps"] == 8
    assert sum(segment["loss_steps"] for segment in segments) == count


# ---------------------------------------------------------------------------
# 11. padding never enters the statistics: two explicit denominators
# ---------------------------------------------------------------------------

def test_short_episode_padding_does_not_enter_the_denominator():
    transitions = _episode(count=10)
    segments = segment_episode(transitions, burn_in=64, unroll=64)
    assert len(segments) == 1
    segment = segments[0]
    assert (segment["context_start"], segment["loss_start"],
            segment["loss_end"]) == (0, 0, 10)

    # A 10-step episode inside the 64-wide contract window: 54 cells of padding.
    statistics = padding_statistics(segments, pad_to=64)
    assert statistics["effective_steps"] == 10          # the real denominator
    assert statistics["loss_mask_true"] == 10
    assert statistics["context_steps"] == 0             # no synthetic prefix
    assert statistics["context_mask_true"] == 0
    assert statistics["window_steps"] == 10
    assert statistics["pad_width"] == 64
    assert statistics["naive_denominator"] == 64        # padding counted as valid
    assert statistics["masked_steps"] == 54
    assert statistics["padding_slots"] == 64 - 10
    assert (statistics["naive_denominator"] - statistics["effective_steps"]
            == statistics["masked_steps"])
    # The two denominators really are different numbers: 10 vs 64.
    assert statistics["effective_steps"] != statistics["naive_denominator"]

    # Without padding (width = the widest real window) both denominators meet at
    # 10, which is exactly why the padding width has to be stated explicitly.
    unpadded = padding_statistics(segments)
    assert unpadded["pad_width"] == 10
    assert unpadded["naive_denominator"] == 10 == unpadded["effective_steps"]
    assert unpadded["masked_steps"] == 0
    assert unpadded["padding_slots"] == 0


def test_contract_rollout_yields_exactly_effective_minibatch_steps():
    episodes = _episode(count=ROLLOUT_DECISIONS_PER_ENV)  # 512
    segments = segment_episode(episodes)
    assert len(segments) == SEQUENCES_PER_MINIBATCH == 8

    full_row = BURN_IN_DECISIONS + LOSS_UNROLL_DECISIONS  # a 64+64 tensor row
    statistics = padding_statistics(segments, pad_to=full_row)
    assert statistics["sequences"] == 8
    assert statistics["effective_steps"] == EFFECTIVE_MINIBATCH_STEPS == 512
    assert statistics["loss_mask_true"] == 512
    # 7 windows have a full 64-step context; the first one has none.
    assert statistics["context_steps"] == 7 * 64 == 448
    assert statistics["window_steps"] == 64 + 7 * (64 + 64) == 960
    # Counting every cell of the 8 x 128 rectangle would double the denominator.
    assert statistics["pad_width"] == 128
    assert statistics["naive_denominator"] == 8 * 128 == 1024
    assert statistics["masked_steps"] == 1024 - 512 == 512
    assert statistics["padding_slots"] == 1024 - 960 == 64
    # ... and 512 masked cells = 448 burn-in + 64 padding, none of them loss.
    assert statistics["masked_steps"] == statistics["context_steps"] + statistics["padding_slots"]


# ---------------------------------------------------------------------------
# 12. burn-in feeds the hidden state, never the loss
# ---------------------------------------------------------------------------

def test_burn_in_context_never_carries_loss():
    count = 200
    segments = segment_episode(_episode(count=count), burn_in=64, unroll=64)
    for segment in segments:
        context_start = segment["context_start"]
        loss_start = segment["loss_start"]
        loss_end = segment["loss_end"]
        # context_mask is True on the whole burn-in window ...
        assert segment["context_mask"] == _window_mask(count, context_start, loss_start)
        assert all(segment["context_mask"][context_start:loss_start])
        # ... and loss_mask is False there, i.e. no burn-in step can be a loss.
        assert not any(segment["loss_mask"][context_start:loss_start])
        # The two masks are disjoint: no step is both context and loss.
        assert not any(context and loss
                       for context, loss in zip(segment["context_mask"],
                                                segment["loss_mask"]))
        # Every loss step is inside the loss window and vice versa.
        assert segment["context_mask"] != segment["loss_mask"]

    assert segments[1]["context_start"] == 0
    assert segments[1]["context_mask"][0:64] == (True,) * 64
    assert segments[1]["loss_mask"][0:64] == (False,) * 64
    assert segments[1]["loss_mask"][64:128] == (True,) * 64


# ---------------------------------------------------------------------------
# 13. segmentation does not change the advantages (exact `==`, no tolerance)
# ---------------------------------------------------------------------------

def test_segmentation_does_not_change_advantages():
    count = 200
    transitions = _episode(count=count)
    values = [0.01 * index for index in range(count)]
    segments = segment_episode(transitions, burn_in=64, unroll=64)
    full = gae(transitions, values)

    # The data-layer cut the segmenter reports: mark every segment boundary.
    marked = [dict(transition) for transition in transitions]
    for segment in segments[:-1]:
        marked[segment["loss_end"] - 1]["segment_end"] = True
    assert sum(1 for transition in marked if transition.get("segment_end")) == 3
    again = gae(marked, values)

    for key in ("advantages", "value_targets", "deltas"):
        # Exact equality, not approx: a boundary treated as a truncation would
        # zero the continuation and change almost every float.
        assert again[key] == full[key], f"{key} changed at a segment boundary"

    for segment in segments:
        window = slice(segment["loss_start"], segment["loss_end"])
        assert again["advantages"][window] == full["advantages"][window]
        assert again["value_targets"][window] == full["value_targets"][window]

    # Independent run: the last window ends at the true data end, so running gae
    # on that window alone must reproduce the full run bit for bit.
    last = segments[-1]
    standalone = gae(transitions[last["context_start"]:],
                     values[last["context_start"]:])
    assert standalone["advantages"] == full["advantages"][last["context_start"]:]
    assert standalone["value_targets"] == full["value_targets"][last["context_start"]:]


def test_a_segment_boundary_is_not_a_truncation():
    transitions = _episode(count=4)
    values = [0.1, 0.2, 0.3, 0.4]
    segments = segment_episode(transitions, burn_in=2, unroll=2)
    assert [(segment["context_start"], segment["loss_start"], segment["loss_end"])
            for segment in segments] == [(0, 0, 2), (0, 2, 4)]
    boundary = segments[0]["loss_end"] - 1  # decision index 1
    plain = gae(transitions, values)

    # If the boundary were a truncation it would *require* a terminal_value ...
    truncated = [dict(transition) for transition in transitions]
    truncated[boundary]["truncated"] = True
    with pytest.raises(SequenceError):
        gae(truncated, values)
    # ... and even with one supplied the advantages change.
    truncated[boundary]["terminal_value"] = 0.0
    changed = gae(truncated, values)
    assert changed["advantages"] != plain["advantages"]
    assert changed["advantages"][boundary] != plain["advantages"][boundary]

    # With segment_end instead: identical, exactly.
    marked = [dict(transition) for transition in transitions]
    marked[boundary]["segment_end"] = True
    assert gae(marked, values)["advantages"] == plain["advantages"]


# ---------------------------------------------------------------------------
# 14. order is preserved; segments are never shuffled
# ---------------------------------------------------------------------------

def test_segment_order_equals_input_order():
    count = 200
    transitions = [_transition(episode_id="ep", reward=float(index))
                   for index in range(count)]
    segments = segment_episode(transitions)
    seen = []
    for segment in segments:
        loss_indices = [index for index, flag in enumerate(segment["loss_mask"])
                        if flag]
        # Ascending, contiguous, and exactly the positional slice of the input.
        assert loss_indices == list(range(segment["loss_start"], segment["loss_end"]))
        rewards = [transitions[index]["reward"] for index in loss_indices]
        assert rewards == [float(index) for index in loss_indices]
        assert rewards == sorted(rewards)          # never reordered
        assert len(set(rewards)) == len(rewards)   # every decision exactly once
        seen.extend(loss_indices)
    assert seen == list(range(count))              # no gap, no repeat, no shuffle


# ---------------------------------------------------------------------------
# 15. cross-episode input is refused
# ---------------------------------------------------------------------------

def test_segment_episode_refuses_two_episodes():
    two = _episode("ep1", 3) + _episode("ep2", 3)
    with pytest.raises(SequenceError):
        segment_episode(two)
    # The refusal is about the boundary, not about the length: the same six
    # decisions as a single episode segment fine.
    assert len(segment_episode(_episode("ep1", 6))) == 1


def test_segment_episode_rejects_empty_and_bad_arguments():
    with pytest.raises(SequenceError):
        segment_episode([])
    with pytest.raises(SequenceError):
        segment_episode(_episode(count=4), burn_in=-1)
    with pytest.raises(SequenceError):
        segment_episode(_episode(count=4), burn_in=True)
    with pytest.raises(SequenceError):
        segment_episode(_episode(count=4), unroll=0)
    with pytest.raises(SequenceError):
        segment_episode(_episode(count=4), unroll=2.5)
    # burn_in=0 is legal: no context at all.
    no_context = segment_episode(_episode(count=70), burn_in=0)
    assert [segment["context_steps"] for segment in no_context] == [0, 0]
    assert [segment["loss_steps"] for segment in no_context] == [64, 6]


# ---------------------------------------------------------------------------
# 16. overlap rules: loss windows disjoint, context may overlap
# ---------------------------------------------------------------------------

def test_loss_windows_never_overlap_but_context_may():
    count = 200
    segments = segment_episode(_episode(count=count), burn_in=64, unroll=64)
    loss_ranges = [(segment["loss_start"], segment["loss_end"])
                   for segment in segments]
    for (start_a, end_a), (start_b, end_b) in zip(loss_ranges, loss_ranges[1:]):
        assert end_a <= start_b, "loss windows must not overlap"
    assert loss_ranges[0][0] == 0
    assert loss_ranges[-1][1] == count
    assert sum(end - start for start, end in loss_ranges) == count

    # The burn-in of a later window does overlap the previous loss window: that
    # is the design intent (the hidden state must be rebuilt from real steps).
    assert segments[2]["context_start"] == segments[1]["loss_start"]
    assert segments[2]["context_start"] < segments[1]["loss_end"]
    assert segments[2]["context_steps"] == 64
    # ... but the overlap is context only: no loss step is computed twice.
    for index, segment_a in enumerate(segments):
        for segment_b in segments[index + 1:]:
            assert not any(a and b for a, b in zip(segment_a["loss_mask"],
                                                   segment_b["loss_mask"]))


# ---------------------------------------------------------------------------
# 17. the strict mid-sample rule is kept — segment_end does not rescue it
# ---------------------------------------------------------------------------

def test_segment_end_does_not_rescue_a_mid_sample_episode_boundary():
    transitions = [
        _transition(episode_id="ep1", reward=1.0, segment_end=True),
        _transition(episode_id="ep2", reward=1.0),
    ]
    with pytest.raises(SequenceError):
        gae(transitions, [0.1, 0.2])


def test_gae_rejects_non_bool_segment_end():
    transitions = [_transition(reward=1.0)]
    transitions[0]["segment_end"] = 1  # int instead of bool
    with pytest.raises(SequenceError):
        gae(transitions, [0.1])


# ---------------------------------------------------------------------------
# 18. the data tail keeps bootstrap_mask=0 unless a value is given explicitly
# ---------------------------------------------------------------------------

def test_tail_without_end_flag_keeps_bootstrap_mask_zero():
    values = [0.3, 0.5, 0.2]
    transitions = [
        _transition(episode_id="tail", reward=1.0),
        _transition(episode_id="tail", reward=-0.5),
        _transition(episode_id="tail", reward=0.25),
    ]
    result = gae(transitions, values)
    # No successor: d = r - V (no gamma * V_next term).
    assert result["deltas"][-1] == 0.25 - 0.2
    assert result["advantages"][-1] == 0.25 - 0.2
    assert result["value_targets"][-1] == (0.25 - 0.2) + 0.2

    # segment_end on the tail does not bootstrap it either.
    marked = [dict(transition) for transition in transitions]
    marked[-1]["segment_end"] = True
    assert gae(marked, values)["advantages"] == result["advantages"]

    # An explicit terminal_value at the data end is the documented way to
    # bootstrap there, and it really changes the last advantage.
    bootstrapped = [dict(transition) for transition in transitions]
    bootstrapped[-1]["terminal_value"] = 2.0
    other = gae(bootstrapped, values)
    assert other["deltas"][-1] == 0.25 + GAMMA_2 * 2.0 - 0.2
    assert other["advantages"][-1] != result["advantages"][-1]


# ---------------------------------------------------------------------------
# 19. segment_batch grouping
# ---------------------------------------------------------------------------

def test_segment_batch_groups_by_sequences_per_minibatch():
    episodes = {f"ep{index}": _episode(f"ep{index}", 128) for index in range(4)}
    groups = segment_batch(episodes)
    # 4 episodes * 2 windows of 64 = 8 segments = exactly one full minibatch.
    assert len(groups) == 1
    group = groups[0]
    assert group["minibatch_index"] == 0
    assert group["sequences"] == SEQUENCES_PER_MINIBATCH == 8
    assert group["is_full"] is True
    assert group["effective_steps"] == 8 * 64 == EFFECTIVE_MINIBATCH_STEPS
    assert [segment["episode_id"] for segment in group["segments"]] == [
        "ep0", "ep0", "ep1", "ep1", "ep2", "ep2", "ep3", "ep3"]
    assert [segment["segment_index"] for segment in group["segments"]] == [
        0, 1, 0, 1, 0, 1, 0, 1]


def test_segment_batch_accepts_pairs_and_flags_a_short_group():
    pairs = [("a", _episode("a", 70)), ("b", _episode("b", 10))]
    groups = segment_batch(pairs)
    assert len(groups) == 1
    assert [group["sequences"] for group in groups] == [3]
    assert groups[0]["is_full"] is False       # reported, not dropped
    assert groups[0]["effective_steps"] == 70 + 10
    assert [segment["episode_id"] for segment in groups[0]["segments"]] == [
        "a", "a", "b"]
    assert [segment["loss_steps"] for segment in groups[0]["segments"]] == [
        64, 6, 10]


def test_segment_batch_rejects_bad_input():
    with pytest.raises(SequenceError):
        segment_batch([])
    # the key and the records disagree: two episodes in one sequence
    with pytest.raises(SequenceError):
        segment_batch({"b": _episode("a", 4)})
    with pytest.raises(SequenceError):
        segment_batch([("a", _episode("a", 4)), ("a", _episode("a", 4))])
    with pytest.raises(SequenceError):
        segment_batch({"a": []})
    with pytest.raises(SequenceError):
        segment_batch({"a": _episode("a", 4)}, sequences_per_minibatch=0)


# ---------------------------------------------------------------------------
# 20. padding_statistics fails closed on a mask that leaks padding
# ---------------------------------------------------------------------------

def test_padding_statistics_refuses_a_mask_that_leaks_padding():
    segments = segment_episode(_episode(count=10))
    tampered = [dict(segments[0])]
    loss_mask = list(tampered[0]["loss_mask"])
    loss_mask.append(True)  # a cell outside the loss window marked as loss
    tampered[0]["loss_mask"] = tuple(loss_mask)
    with pytest.raises(SequenceError):
        padding_statistics(tampered, pad_to=64)

    # 0/1 instead of bools is refused: a mask that cannot be summed safely.
    ints = [dict(segments[0])]
    ints[0]["loss_mask"] = tuple(int(flag) for flag in segments[0]["loss_mask"])
    with pytest.raises(SequenceError):
        padding_statistics(ints)

    with pytest.raises(SequenceError):
        padding_statistics([])
    with pytest.raises(SequenceError):
        padding_statistics(segments, pad_to=1)  # narrower than the real window

