"""Unit tests for the v4.3.0 actor candidate (W03 slice D).

**These are synthetic tests, not game or capability evidence.**  Nothing here
starts the engine, reads a capture, trains a step or claims a score: the tests
assert the *architecture* of ``DEVELOPMENT_GUIDE.md`` §4.3 (three CNN streams, a
65-dim history projection, one GRU, four independent action heads), the §5.1
arithmetic (joint log-probability and joint entropy are the **sum** of the four
heads) and the six fail-closed refusals of
:mod:`touhou_rl.v430.models.actor`.  The observation fed in is built by the real
``observation.encode_o2`` from four synthetic RGBA frames, so the shapes are the
frozen O2 shapes; the pixels are noise and mean nothing.

Requirement map (the twelve items of the work-package brief):

1  shapes                     ``test_head_logit_shapes_unbatched_and_batched``
2  parameter count            ``test_parameter_report_total_within_plan_range``
3  joint log-prob is a sum    ``test_joint_log_prob_is_the_sum_of_four_heads``
4  joint entropy is a sum     ``test_joint_entropy_is_the_sum_of_four_heads``
5  72 combinations normalise  ``test_72_joint_probabilities_normalise``
6  zero-update determinism    ``test_eval_mode_is_deterministic_and_has_no_dropout``
7  no BatchNorm               ``test_no_batch_norm_anywhere``
8  privileged input refused   ``test_privileged_input_is_refused``
9  wrong shape refused        ``test_wrong_image_shape_is_refused``
10 reproducible sampling      ``test_sampling_is_reproducible_with_generator``
11 hidden state handling      ``test_hidden_state_zero_init_and_passthrough``
12 synthetic-data note        this docstring

plus the model-layer half of "no automatic OR SHOOT"
(``test_shoot_head_is_independent_and_never_forced``).
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from touhou_rl.v430 import contracts, observation
from touhou_rl.v430.models import actor

#: Head sizes the contract froze (``PLAN_SPEC.action.heads``).
HEAD_SIZES = {"move": 9, "focus": 2, "shoot": 2, "bomb": 2}
HEADS = ("move", "focus", "shoot", "bomb")


# ---------------------------------------------------------------------------
# fixtures: one real O2 encoding (of synthetic noise), reused read-only
# ---------------------------------------------------------------------------
def _synthetic_frames(count: int = 4, seed: int = 7) -> list[np.ndarray]:
    """``count`` distinct opaque 480x640 RGBA frames of seeded noise.

    Alpha is forced to 255 so the encoder's alpha gate accepts the buffer; the
    frames are distinct because each draw uses the same seeded stream.  Nothing
    here resembles a game frame, and that is the point: the tests below are about
    shapes and arithmetic, not about pixels.
    """

    generator = np.random.default_rng(seed)
    frames = []
    for _ in range(count):
        frame = generator.integers(0, 256, size=(480, 640, 4), dtype=np.uint8)
        frame[..., 3] = 255
        frames.append(frame)
    return frames


@pytest.fixture(scope="module")
def o2_observation() -> observation.O2Observation:
    """A genuine ``encode_o2`` result (noise pixels, real frozen shapes)."""

    return observation.encode_o2(
        _synthetic_frames(),
        key_history=None,
        history_valid=[False] * 8,
        elapsed_capture_s=0.016,
    )


@pytest.fixture(scope="module")
def actor_observation(o2_observation: observation.O2Observation
                      ) -> contracts.ActorObservation:
    """The same observation with a zero ``memory`` (cold start)."""

    return o2_observation.to_actor_observation(np.zeros(256, dtype=np.float32))


@pytest.fixture(scope="module")
def policy() -> actor.ActorPolicy:
    torch.manual_seed(0)
    model = actor.build_actor()
    model.eval()
    return model


@pytest.fixture(scope="module")
def head_logits(policy: actor.ActorPolicy, actor_observation: contracts.ActorObservation
                ) -> actor.HeadLogits:
    with torch.no_grad():
        logits, _ = policy(actor_observation)
    return logits.detach()


def _batch_of(observation_like: observation.O2Observation, batch: int) -> dict[str, Any]:
    """Stack an encoded observation into an unbatched->batched tensor bundle."""

    fields = observation_like.as_dict()
    return {name: np.stack([fields[name]] * batch, axis=0) for name in fields}


# ---------------------------------------------------------------------------
# 1. shapes
# ---------------------------------------------------------------------------
def test_head_logit_shapes_unbatched_and_batched(
        policy: actor.ActorPolicy, actor_observation: contracts.ActorObservation,
        o2_observation: observation.O2Observation) -> None:
    """One observation -> ``(9,) (2,) (2,) (2)``; a batch -> ``(B, head)``."""

    with torch.no_grad():
        logits, hidden = policy(actor_observation)
    assert isinstance(logits, actor.HeadLogits)
    assert tuple(logits.move.shape) == (9,)
    assert tuple(logits.focus.shape) == (2,)
    assert tuple(logits.shoot.shape) == (2,)
    assert tuple(logits.bomb.shape) == (2,)
    assert logits.batched is False and logits.batch_size == 1
    # The GRU state keeps its layer axis even for one observation.
    assert tuple(hidden.shape) == (1, 1, 256)

    for batch in (1, 3):
        with torch.no_grad():
            batched, hidden_b = policy(_batch_of(o2_observation, batch))
        assert tuple(batched.move.shape) == (batch, 9)
        for head, size in HEAD_SIZES.items():
            assert tuple(batched[head].shape) == (batch, size)
        assert batched.batched is True and batched.batch_size == batch
        assert tuple(hidden_b.shape) == (1, batch, 256)

    # ``HeadLogits`` refuses a per-head width other than the contract's.
    with pytest.raises(actor.ShapeMismatchError):
        actor.HeadLogits(move=torch.zeros(10), focus=torch.zeros(2),
                         shoot=torch.zeros(2), bomb=torch.zeros(2))
    # ... and refuses heads that disagree on rank.
    with pytest.raises(actor.ShapeMismatchError):
        actor.HeadLogits(move=torch.zeros(4, 9), focus=torch.zeros(2),
                         shoot=torch.zeros(2), bomb=torch.zeros(2))


# ---------------------------------------------------------------------------
# 2. parameter count
# ---------------------------------------------------------------------------
def test_parameter_report_total_within_plan_range(policy: actor.ActorPolicy) -> None:
    """Total in ``PLAN_SPEC.model.parameter_range_millions`` and actor-only."""

    report = policy.parameter_report()
    total = report["total"]
    print(f"\nACTUAL ACTOR PARAMETER TOTAL: {total:,} ({total / 1e6:.6f} M)")
    print(f"  streams: global={report['parts']['global_stream']:,} "
          f"play={report['parts']['play_stream']:,} "
          f"hud={report['parts']['hud_stream']:,}")
    print(f"  history={report['parts']['history_branch']:,} "
          f"fusion={report['parts']['fusion']:,} "
          f"gru={report['parts']['memory_gru']:,} "
          f"heads={report['parts']['action_heads']:,}")

    low, high = 1.0, 3.0
    assert low <= total / 1e6 <= high, (
        f"{total:,} parameters is outside PLAN_SPEC's [{low}, {high}] M range")
    assert report["within_parameter_range"] is True
    assert report["total_millions"] == pytest.approx(total / 1e6, abs=1e-9)

    # The report must add up, both to itself and to an independent recount.
    assert total == sum(report["parts"].values())
    assert total == sum(int(p.numel()) for p in policy.parameters())
    for head in HEADS:
        assert report["action_heads"][head] == sum(
            int(p.numel()) for p in policy.heads[head].parameters())

    # The guide's "约 2.73M" is the definition this table produces.
    assert total == pytest.approx(2_730_000, rel=0.02), (
        f"{total:,} is not within 2% of guide §4.3's '约 2.73M'")

    # ---- actor-only declaration: no critic / auxiliary head may hide in here.
    assert report["scope"] == "actor"
    assert report["includes_critic"] is False
    assert report["includes_auxiliary_heads"] is False
    assert report["parameter_scope"] == actor.MODEL_SPEC["parameter_scope"]
    assert report["auxiliary_heads_not_included"] == list(
        actor.MODEL_SPEC["auxiliary_heads"])
    assert len(report["auxiliary_heads_not_included"]) == 6
    # No part of the report is named after a critic or an auxiliary head.
    for name in report["parts"]:
        assert "critic" not in name and "aux" not in name and "value" not in name
    for name in report["auxiliary_heads_not_included"]:
        assert name not in report["parts"]
    # ... and no module/parameter of the built actor is a critic or aux head.
    names = [name for name, _ in policy.named_parameters()]
    assert not any("critic" in name or "value" in name or "aux" in name
                   for name in names), names
    linear_outs = sorted(module.out_features for module in policy.modules()
                         if isinstance(module, nn.Linear))
    # three stream projections + history + fusion + the four heads, nothing else
    assert linear_outs == [2, 2, 2, 9, 32, 64, 128, 128, 256], linear_outs
    assert sorted(policy.heads) == sorted(HEADS)


# ---------------------------------------------------------------------------
# 3. joint log-probability is the SUM
# ---------------------------------------------------------------------------
def _known_logits(move_probs: list[float], focus_probs: list[float],
                  shoot_probs: list[float], bomb_probs: list[float]
                  ) -> actor.HeadLogits:
    """Logits whose softmax is exactly the given distribution (``logit = log p``)."""

    def logits(probs: list[float]) -> torch.Tensor:
        return torch.log(torch.tensor(probs, dtype=torch.float32))

    return actor.HeadLogits(move=logits(move_probs), focus=logits(focus_probs),
                            shoot=logits(shoot_probs), bomb=logits(bomb_probs))


def test_joint_log_prob_is_the_sum_of_four_heads() -> None:
    """``joint_log_prob`` == sum of the four log-probs, not product/mean (hand-calc)."""

    # (a) uniform heads: every head contributes exactly -ln(size).
    uniform = actor.HeadLogits(
        move=torch.zeros(9), focus=torch.zeros(2),
        shoot=torch.zeros(2), bomb=torch.zeros(2))
    action = contracts.JointAction(move=0, focus=0, shoot=0, bomb=0)
    expected = -(math.log(9) + math.log(2) + math.log(2) + math.log(2))
    measured = float(actor.joint_log_prob(uniform, action))
    assert measured == pytest.approx(expected, rel=1e-6), (measured, expected)
    # Every one of the 72 combinations has the same log-probability here.
    assert float(actor.joint_log_prob(uniform, [4, 1, 1, 1])) == pytest.approx(
        expected, rel=1e-6)

    # (b) non-uniform heads, so sum / product-of-logs / mean are all distinct.
    move = [1 / 9] * 9
    focus = [0.7, 0.3]
    shoot = [0.9, 0.1]
    bomb = [0.25, 0.75]
    logits = _known_logits(move, focus, shoot, bomb)
    taken = (0, 0, 1, 0)
    head_log_probs = [
        math.log(move[taken[0]]), math.log(focus[taken[1]]),
        math.log(shoot[taken[2]]), math.log(bomb[taken[3]]),
    ]
    summed = sum(head_log_probs)
    product_of_logs = math.prod(head_log_probs)
    mean_of_logs = summed / 4

    measured = float(actor.joint_log_prob(logits, contracts.JointAction(*taken)))
    assert measured == pytest.approx(summed, rel=1e-5)
    # The joint *probability* is the product of the four probabilities ...
    assert math.exp(measured) == pytest.approx(
        (1 / 9) * 0.7 * 0.1 * 0.25, rel=1e-5)
    # ... which is the exponential of the log-probability sum, nothing else.
    assert measured != pytest.approx(product_of_logs, rel=1e-3)
    assert measured != pytest.approx(mean_of_logs, rel=1e-3)

    # Per-head decomposition is available for logging, and it is the same sum.
    per_head = {head: float(torch.log_softmax(logits[head], dim=-1)[taken[index]])
                for index, head in enumerate(HEADS)}
    assert sum(per_head.values()) == pytest.approx(measured, rel=1e-6)
    # The contract's own float-level check agrees with the tensor path.
    assert contracts.joint_logprob_of(per_head) == pytest.approx(measured, rel=1e-6)

    # A missing head raises: no head may stand in for another (v4.2.0 defect).
    with pytest.raises(actor.ModelError):
        actor.joint_log_prob(logits, {"move": 0, "focus": 0, "shoot": 0})
    with pytest.raises(actor.ModelError):
        actor.joint_log_prob(logits, [0, 0, 0])          # three heads
    with pytest.raises(actor.ModelError):
        actor.joint_log_prob(logits, [9, 0, 0, 0])       # move outside [0, 9)


# ---------------------------------------------------------------------------
# 4. joint entropy is the SUM
# ---------------------------------------------------------------------------
def test_joint_entropy_is_the_sum_of_four_heads() -> None:
    """``joint_entropy`` == sum of the four head entropies (hand-calc)."""

    uniform = actor.HeadLogits(
        move=torch.zeros(9), focus=torch.zeros(2),
        shoot=torch.zeros(2), bomb=torch.zeros(2))
    expected = math.log(9) + 3 * math.log(2)
    measured = float(actor.joint_entropy(uniform))
    assert measured == pytest.approx(expected, rel=1e-6), (measured, expected)
    # Not a product and not a mean.
    assert measured != pytest.approx(math.log(9) * math.log(2) ** 3, rel=1e-3)
    assert measured != pytest.approx(expected / 4, rel=1e-3)

    move = [1 / 9] * 9
    focus = [0.7, 0.3]
    shoot = [0.9, 0.1]
    bomb = [0.25, 0.75]
    logits = _known_logits(move, focus, shoot, bomb)

    def entropy(probs: list[float]) -> float:
        return -sum(p * math.log(p) for p in probs if p > 0.0)

    summed = entropy(move) + entropy(focus) + entropy(shoot) + entropy(bomb)
    measured = float(actor.joint_entropy(logits))
    assert measured == pytest.approx(summed, rel=1e-5)
    assert measured >= 0.0

    # Per-head entropies sum to the same number, and agree with the contract.
    per_head = {head: float(-(torch.softmax(logits[head], -1)
                             * torch.log_softmax(logits[head], -1)).sum())
                for head in HEADS}
    assert sum(per_head.values()) == pytest.approx(measured, rel=1e-6)
    assert contracts.joint_entropy_of(per_head) == pytest.approx(measured, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. the 72 combinations are a normalised distribution
# ---------------------------------------------------------------------------
def test_72_joint_probabilities_normalise(head_logits: actor.HeadLogits) -> None:
    """``sum(exp(joint_log_prob))`` over the 72 combinations == 1 (tolerance 1e-6)."""

    assert contracts.joint_inverse()
    assert contracts.JOINT_COUNT == 72
    total = 0.0
    seen = set()
    for joint_action in contracts.iter_joint_actions():
        log_prob = float(actor.joint_log_prob(head_logits, joint_action))
        assert math.isfinite(log_prob) and log_prob <= 0.0
        total += math.exp(log_prob)
        seen.add(joint_action.joint)
    assert len(seen) == 72
    print(f"\nSUM OVER 72 JOINT PROBABILITIES = {total!r}")
    assert total == pytest.approx(1.0, abs=1e-6), total
    # The same total comes out of one batched call, i.e. per-row normalisation.
    stacked = torch.stack([
        torch.stack([torch.log_softmax(head_logits[head], -1)[
            contracts.decode_joint(joint)[index]] for index, head in enumerate(HEADS)])
        for joint in range(72)])
    batched_total = float(torch.exp(stacked.sum(dim=1)).sum())
    assert batched_total == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 6. determinism (no dropout, no batch statistics, no update)
# ---------------------------------------------------------------------------
def test_eval_mode_is_deterministic_and_has_no_dropout(
        policy: actor.ActorPolicy, actor_observation: contracts.ActorObservation
        ) -> None:
    """Same weights + same input -> bitwise-equal logits; no stochastic layer."""

    policy.eval()
    with torch.no_grad():
        first, hidden_first = policy(actor_observation)
        second, hidden_second = policy(actor_observation)
    for head in HEADS:
        assert torch.equal(first[head], second[head]), head
        assert torch.equal(first[head], first[head].clone())
    assert torch.equal(hidden_first, hidden_second)

    # ``train()`` is also deterministic for this architecture: it has no dropout,
    # no BatchNorm in training mode and no stochastic depth.
    policy.train()
    with torch.no_grad():
        third, _ = policy(actor_observation)
    policy.eval()
    for head in HEADS:
        assert torch.equal(first[head], third[head]), head

    # The architecture has no stochastic layer at all, so a "seed" cannot change
    # a forward pass.
    stochastic = [type(module).__name__ for module in policy.modules()
                  if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d,
                                         nn.Dropout3d, nn.AlphaDropout,
                                         nn.GaussianNLLLoss))]
    assert stochastic == []
    for module in policy.modules():
        if isinstance(module, nn.GRU):
            assert module.dropout == 0.0
    # Convolutional blocks are conv -> GroupNorm -> SiLU and nothing else.
    for name in ("global", "play", "hud"):
        kinds = [type(module).__name__ for module in policy.streams[name].body]
        assert kinds == ["Conv2d", "GroupNorm", "SiLU"] * 3, kinds

    # No optimizer step happened: the parameters are unchanged and every stream
    # still receives gradient (a frozen branch would train silently wrong).
    with torch.no_grad():
        before = [parameter.detach().clone() for parameter in policy.parameters()]
    policy.zero_grad(set_to_none=True)
    logits, _ = policy(actor_observation)
    loss = sum(logits[head].sum() for head in HEADS)
    loss.backward()
    for parameter, snapshot in zip(policy.parameters(), before):
        assert torch.equal(parameter.detach(), snapshot)
    reached = {name for name, parameter in policy.named_parameters()
               if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0}
    for prefix in ("streams.global", "streams.play", "streams.hud",
                   "history_branch", "fusion", "recurrent", "heads.move",
                   "heads.focus", "heads.shoot", "heads.bomb"):
        assert any(name.startswith(prefix) for name in reached), prefix
    policy.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# 7. no BatchNorm
# ---------------------------------------------------------------------------
def test_no_batch_norm_anywhere(policy: actor.ActorPolicy) -> None:
    """Neither the module tree nor the source imports a batch-statistics layer."""

    offenders = [(name, type(module).__name__)
                 for name, module in policy.named_modules()
                 if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                                        nn.BatchNorm3d, nn.SyncBatchNorm))]
    assert offenders == [], offenders
    # Nothing tracks running statistics either (BatchNorm's mechanism).
    statistics = [name for name, module in policy.named_modules()
                  if module.__class__.__name__.endswith("Norm")
                  and getattr(module, "track_running_stats", False)]
    assert statistics == [], statistics
    buffers = [(name, tensor.shape) for name, tensor in policy.named_buffers()
               if "running_" in name]
    assert buffers == [], buffers

    # GroupNorm everywhere in the spatial stacks, LayerNorm in the fusion, and the
    # group count is PLAN_SPEC's 4.
    norm_kinds = {type(module).__name__ for module in policy.modules()
                  if isinstance(module, (nn.GroupNorm, nn.LayerNorm))}
    assert norm_kinds == {"GroupNorm", "LayerNorm"}
    for name in ("global", "play", "hud"):
        groups = {module.num_groups for module in policy.streams[name].modules()
                  if isinstance(module, nn.GroupNorm)}
        assert groups == {actor.MODEL_SPEC["encoder_candidate"]["conv_norm_groups"]}

    # Belt and braces: the source never *references* a batch-norm class at all,
    # not even in the detection code (which works structurally instead).
    source = Path(actor.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm"}
    hits = [node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in banned]
    hits += [node.id for node in ast.walk(tree)
             if isinstance(node, ast.Name) and node.id in banned]
    assert hits == [], hits
    assert policy.parameter_report()["has_mutable_batch_statistics"] is False
    assert policy.parameter_report()["batch_norm_modules"] == []


# ---------------------------------------------------------------------------
# 8. privileged input is refused
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["stage_id", "seed", "engine_frame",
                                   "teacher_action", "resource_truth",
                                   "player_center", "enemy_table"])
def test_privileged_input_is_refused(
        policy: actor.ActorPolicy, actor_observation: contracts.ActorObservation,
        field: str) -> None:
    """``stage_id``/``seed``/``engine_frame``/... can never reach ``forward``."""

    data = actor_observation.as_dict()
    data[field] = 3
    with pytest.raises(actor.PrivilegedModelInputError) as excinfo:
        policy(data)
    assert field in str(excinfo.value)
    # The exception is a ModelError (and a ContractError), so a caller can catch
    # one type for every model-layer refusal.
    assert isinstance(excinfo.value, actor.ModelError)
    assert isinstance(excinfo.value, contracts.ContractError)

    # A keyword argument is the same leak through a different door, and so is a
    # nested dict carrying the name one level down.
    with pytest.raises(actor.PrivilegedModelInputError):
        policy(actor_observation, **{field: 3})
    nested = actor_observation.as_dict()
    nested["extra"] = {"inner": {field: 3}}
    with pytest.raises(actor.PrivilegedModelInputError):
        policy(nested)

    # An unregistered (non-privileged) name is refused too: the surface is closed.
    with pytest.raises(actor.PrivilegedModelInputError):
        policy({**actor_observation.as_dict(), "harmless_typo": 0})
    with pytest.raises(actor.PrivilegedModelInputError):
        policy(actor_observation, temperature=0.5)
    # ``build_actor`` has no architecture knobs to smuggle a wider input into.
    with pytest.raises(actor.PrivilegedModelInputError):
        actor.build_actor(hidden_size=512)


def test_obs_must_be_an_observation_record(policy: actor.ActorPolicy) -> None:
    """A bare tensor/string/number is not an observation and is refused."""

    for bad in (torch.zeros(3, 240, 320), "observation", 3.5, None):
        with pytest.raises(actor.ModelError):
            policy(bad)
    # A missing field is refused rather than zero-filled.
    with pytest.raises(actor.ModelError):
        policy({"global_rgb": np.zeros((3, 240, 320), np.uint8)})


# ---------------------------------------------------------------------------
# 9. wrong shapes are refused, never repaired
# ---------------------------------------------------------------------------
def test_wrong_image_shape_is_refused(
        policy: actor.ActorPolicy, o2_observation: observation.O2Observation) -> None:
    """``(3,240,319)`` and friends raise; nothing is interpolated or padded."""

    def one(**overrides: Any) -> dict[str, Any]:
        fields = dict(o2_observation.as_dict())
        fields.update(overrides)
        return fields

    # The exact case from the brief: one column short.
    with pytest.raises(actor.ShapeMismatchError):
        policy(one(global_rgb=np.zeros((3, 240, 319), np.uint8)))
    # ... and the same wrong shape with a leading batch axis.
    with pytest.raises(actor.ShapeMismatchError):
        policy(one(global_rgb=np.zeros((1, 3, 240, 319), np.uint8)))
    # Every stream refuses its own wrong shape.
    for field, shape in (("global_rgb", (3, 240, 320)),
                         ("play_gray", (4, 224, 192)),
                         ("hud_gray", (1, 448, 160))):
        wrong = shape[:-1] + (shape[-1] + 1,)
        with pytest.raises(actor.ShapeMismatchError):
            policy(one(**{field: np.zeros(wrong, np.uint8)}))
        # A transposed image is a different shape, not the same one.
        with pytest.raises(actor.ShapeMismatchError):
            policy(one(**{field: np.zeros(shape[::-1], np.uint8)}))
    # The 2-D image and the 5-D stack are refused as well (no auto squeeze/pad).
    with pytest.raises(actor.ShapeMismatchError):
        policy(one(global_rgb=np.zeros((240, 320), np.uint8)))
    with pytest.raises(actor.ShapeMismatchError):
        policy(one(play_gray=np.zeros((1, 4, 224, 192), np.uint8)))

    # A float image means it was normalised outside the model: refuse it.
    with pytest.raises(actor.ShapeMismatchError) as excinfo:
        policy(one(global_rgb=np.zeros((3, 240, 320), np.float32)))
    assert "255" in str(excinfo.value)
    # Wrong dtypes on the non-image fields are refused too, never cast.
    with pytest.raises(actor.ShapeMismatchError):
        policy(one(history_valid=np.zeros(8, np.int64)))
    with pytest.raises(actor.ShapeMismatchError):
        policy(one(elapsed_capture_s=np.float64(0.016)))
    # Mixed batchedness across fields cannot be broadcast into a batch.
    batched = _batch_of(o2_observation, 2)
    batched["hud_gray"] = batched["hud_gray"][0]
    with pytest.raises(actor.ShapeMismatchError):
        policy(batched)
    # Values outside 0/1 in the executed key history are refused.
    keys = np.zeros((8, 7), np.uint8)
    keys[0, 0] = 2
    with pytest.raises(actor.ShapeMismatchError):
        policy(one(executed_key_history=keys))
    # A non-finite interval is refused (it would poison the projection).
    with pytest.raises(actor.ShapeMismatchError):
        policy(one(elapsed_capture_s=np.float32("nan")))


# ---------------------------------------------------------------------------
# 10. reproducible sampling
# ---------------------------------------------------------------------------
def test_sampling_is_reproducible_with_generator(head_logits: actor.HeadLogits) -> None:
    """Same generator seed -> same action; different seeds -> different actions."""

    def draw(seed: int, steps: int = 1) -> list[tuple[int, ...]]:
        generator = torch.Generator().manual_seed(seed)
        return [actor.sample(head_logits, generator=generator)[0].as_tuple()
                for _ in range(steps)]

    with torch.no_grad():
        assert draw(1234, steps=12) == draw(1234, steps=12)
        # Re-seeding one generator rewinds it.
        generator = torch.Generator().manual_seed(99)
        first = [actor.sample(head_logits, generator=generator)[0].as_tuple()
                 for _ in range(8)]
        generator.manual_seed(99)
        again = [actor.sample(head_logits, generator=generator)[0].as_tuple()
                 for _ in range(8)]
        assert first == again
        # Different seeds must differ over 64 draws (a collision in every one of
        # 64 steps is not a coincidence, it is broken seeding).
        seeds = [draw(seed, steps=64) for seed in (1, 2)]
        assert seeds[0] != seeds[1]
        differing = sum(1 for left, right in zip(seeds[0], seeds[1]) if left != right)
        assert differing > 8, differing

        # The sampled log-probability is the joint log-probability of the drawn
        # action (the rollout logs what it sampled, not something adjacent).
        action, log_prob = actor.sample(head_logits,
                                        generator=torch.Generator().manual_seed(5))
        assert isinstance(action, contracts.JointAction)
        assert 0 <= action.joint < contracts.JOINT_COUNT
        assert float(log_prob) == pytest.approx(
            float(actor.joint_log_prob(head_logits, action)), rel=1e-6)
        assert float(log_prob) <= 0.0

        # A batched draw returns one action per row with a (B,) log-probability.
        batched = actor.HeadLogits(**{
            head: torch.stack([value, value, value])
            for head, value in head_logits.as_dict().items()})
        actions, batch_log_prob = actor.sample(
            batched, generator=torch.Generator().manual_seed(7))
        assert isinstance(actions, tuple) and len(actions) == 3
        assert all(isinstance(item, contracts.JointAction) for item in actions)
        assert tuple(batch_log_prob.shape) == (3,)

        # Temperature is a real knob and the logged probability follows it.
        cold_action, cold_log_prob = actor.sample(
            head_logits, generator=torch.Generator().manual_seed(11),
            temperature=0.5)
        assert math.isfinite(float(cold_log_prob))
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(actor.ModelError):
                actor.sample(head_logits, temperature=bad)
        with pytest.raises(actor.ModelError):
            actor.sample(head_logits, generator=torch.zeros(4))


def test_sampling_does_not_need_global_rng(head_logits: actor.HeadLogits) -> None:
    """Sampling with an explicit generator leaves ``torch``'s global RNG alone."""

    with torch.no_grad():
        torch.manual_seed(0)
        before = torch.random.get_rng_state()
        actor.sample(head_logits, generator=torch.Generator().manual_seed(3))
        assert torch.equal(before, torch.random.get_rng_state())
        # Without a generator it is allowed to use the global stream (documented),
        # and it then obeys torch.manual_seed.
        torch.manual_seed(21)
        left = actor.sample(head_logits)[0].as_tuple()
        torch.manual_seed(21)
        right = actor.sample(head_logits)[0].as_tuple()
        assert left == right


# ---------------------------------------------------------------------------
# 11. hidden state
# ---------------------------------------------------------------------------
def test_hidden_state_zero_init_and_passthrough(
        policy: actor.ActorPolicy, actor_observation: contracts.ActorObservation,
        o2_observation: observation.O2Observation) -> None:
    """``hidden=None`` is zeros; a returned state round-trips; the batch follows."""

    with torch.no_grad():
        logits_none, hidden_none = policy(actor_observation)
        zeros = torch.zeros(1, 1, 256, dtype=torch.float32)
        logits_zero, hidden_zero = policy(actor_observation, zeros)
        logits_initial, hidden_initial = policy(actor_observation, hidden=None)
    # ``hidden=None`` is *exactly* a zero initial state, not "whatever".
    assert torch.equal(hidden_none, hidden_zero)
    for head in HEADS:
        assert torch.equal(logits_none[head], logits_zero[head])
        assert torch.equal(logits_none[head], logits_initial[head])
    assert torch.equal(hidden_initial, hidden_zero)

    # The returned state feeds straight back in and keeps its shape.
    with torch.no_grad():
        second, hidden_second = policy(actor_observation, hidden_none)
        third, hidden_third = policy(actor_observation, hidden_second)
    assert tuple(second.move.shape) == (9,)
    for state in (hidden_none, hidden_second, hidden_third):
        assert tuple(state.shape) == (1, 1, 256)
        assert state.dtype is torch.float32
        assert bool(torch.isfinite(state).all())
    # The hidden state is not the input: two different prefixes differ.
    assert not torch.equal(hidden_none, hidden_second)

    # The batch dimension follows the input, including a batch change.
    for batch in (1, 2, 5):
        with torch.no_grad():
            _, hidden_batch = policy(_batch_of(o2_observation, batch))
        assert tuple(hidden_batch.shape) == (1, batch, 256)
        with torch.no_grad():
            _, again = policy(_batch_of(o2_observation, batch), hidden_batch)
        assert tuple(again.shape) == (1, batch, 256)

    # Every shape that is not (1, B, 256) is refused, never reshaped.
    for bad in (torch.zeros(256), torch.zeros(1, 256), torch.zeros(2, 1, 256),
                torch.zeros(1, 1, 128), torch.zeros(1, 2, 256)):
        with pytest.raises(actor.HiddenStateError):
            policy(actor_observation, bad)
    with pytest.raises(actor.HiddenStateError):
        policy(actor_observation, torch.zeros(1, 1, 256, dtype=torch.float64))
    with pytest.raises(actor.HiddenStateError):
        policy(actor_observation, torch.full((1, 1, 256), float("nan")))
    with pytest.raises(actor.HiddenStateError):
        policy(actor_observation, np.zeros((1, 1, 256), np.float32))


# ---------------------------------------------------------------------------
# the model-layer half of "no automatic OR SHOOT"
# ---------------------------------------------------------------------------
def test_shoot_head_is_independent_and_never_forced(
        policy: actor.ActorPolicy, actor_observation: contracts.ActorObservation,
        head_logits: actor.HeadLogits) -> None:
    """``shoot=0`` is a real action with a real logit; nothing ORs the bit on."""

    # The contract's switch is still off, and the policy refuses to run if it is on.
    assert contracts.forced_shoot() is False
    assert contracts.truth_action_mask() is False
    actor.assert_no_forced_shoot()

    # The shoot head is an ordinary two-way classifier with both logits finite.
    assert policy.heads["shoot"].out_features == 2
    assert tuple(policy.heads["shoot"].weight.shape) == (2, 256)
    assert sorted(policy.heads) == sorted(HEADS)
    assert tuple(head_logits.shoot.shape) == (2,)
    assert bool(torch.isfinite(head_logits.shoot).all())
    shoot_probability = torch.softmax(head_logits.shoot, dim=-1)
    assert float(shoot_probability.sum()) == pytest.approx(1.0, rel=1e-6)
    assert float(shoot_probability[0]) > 0.0     # "do not shoot" is reachable
    assert float(shoot_probability[1]) > 0.0

    # ``joint_log_prob`` uses the shoot logits verbatim: log_softmax([0, ln3])
    # gives exactly -ln4 for shoot=0, so no bias or mask was added behind our back.
    probe = actor.HeadLogits(
        move=torch.zeros(9), focus=torch.zeros(2),
        shoot=torch.tensor([0.0, math.log(3.0)]), bomb=torch.zeros(2))
    no_shoot = float(actor.joint_log_prob(probe, [0, 0, 0, 0]))
    shoot = float(actor.joint_log_prob(probe, [0, 0, 1, 0]))
    assert no_shoot == pytest.approx(-(math.log(9) + math.log(2) + math.log(4)
                                       + math.log(2)), rel=1e-6)
    assert shoot > no_shoot     # with these logits shooting is the likelier branch
    # The two branches are distinct actions with distinct probabilities: if the
    # model OR-ed shoot on, these two numbers would have to be equal.
    assert math.exp(no_shoot) != pytest.approx(math.exp(shoot))
    assert math.exp(shoot) / math.exp(no_shoot) == pytest.approx(3.0, rel=1e-5)

    # A head that all but forbids shooting never emits shoot=1, and the mirror
    # image never emits shoot=0: the sampler does not "help" either way.
    with torch.no_grad():
        favoured = actor.HeadLogits(move=torch.zeros(9), focus=torch.zeros(2),
                                    shoot=torch.tensor([20.0, -20.0]),
                                    bomb=torch.zeros(2))
        generator = torch.Generator().manual_seed(17)
        draws = [actor.sample(favoured, generator=generator)[0].shoot
                 for _ in range(64)]
        assert set(draws) == {0}
        mirrored = dataclasses.replace(favoured, shoot=torch.tensor([-20.0, 20.0]))
        mirrored_draws = [actor.sample(mirrored, generator=generator)[0].shoot
                          for _ in range(64)]
        assert set(mirrored_draws) == {1}

        # Sampling never mutates the logits (an in-place "force shoot" would).
        before = {head: value.clone() for head, value in head_logits.as_dict().items()}
        actor.sample(head_logits, generator=torch.Generator().manual_seed(23))
        actor.joint_log_prob(head_logits, [0, 0, 0, 0])
        actor.joint_entropy(head_logits)
        for head in HEADS:
            assert torch.equal(before[head], head_logits[head]), head

        # The forward pass has no module that could force the bit: exactly four
        # heads, one linear each, and no path that adds a constant to a head.
        head_linears = [module for module in policy.heads.values()
                        if isinstance(module, nn.Linear)]
        assert len(head_linears) == 4
        assert all(module.bias is not None for module in head_linears)
        # A real forward pass produces shoot logits that differ between the two
        # actions' probabilities (a dead head would be constant).
        assert float(torch.softmax(head_logits.shoot, -1)[0]) != pytest.approx(
            float(torch.softmax(head_logits.shoot, -1)[1]), rel=1e-9)

    # ``evaluate`` recomputes the same numbers as ``forward`` for PPO.
    with torch.no_grad():
        evaluation = policy.evaluate(actor_observation, ids=[0, 0, 0, 0])
    assert isinstance(evaluation, actor.Evaluation)
    for head in HEADS:
        assert torch.equal(evaluation.head_logits[head], head_logits[head])
    assert float(evaluation.joint_log_prob) == pytest.approx(
        float(actor.joint_log_prob(head_logits, [0, 0, 0, 0])), rel=1e-6)
    assert evaluation.joint_entropy is not None
