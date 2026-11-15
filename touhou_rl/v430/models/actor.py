"""TH08 **O2 actor policy** (v4.3.0 work package W03, slice D).

Specification: ``Document/Update/Alpha/v4.3.0/DEVELOPMENT_GUIDE.md`` §4.3 ("第一个
可实现网络候选" — its table is the normative shape list) plus §5.1 (the four action
heads: ``move∈[0,9)``, ``focus/shoot/bomb∈{0,1}``; the joint log-probability and
the joint entropy are the **sum** of the four heads) and ``PLAN_SPEC.json``
``model`` / ``action``.

The whole encoder table lives in one place, :data:`MODEL_SPEC` (a verbatim mirror
of ``PLAN_SPEC.model``).  The module is built **from** that table, so the network
cannot drift from the plan by a copied constant; the cross-checks at import time
tie it to :mod:`touhou_rl.v430.observation` (channel counts, 65 = 56+8+1, the
352-dim fusion) and to :mod:`touhou_rl.v430.contracts` (head sizes, the 256-dim
``memory`` field).

Network (one class, :class:`ActorPolicy`)::

    global   3->16/5/2, 16->32/3/2, 32->32/3/2   pad=k//2, GroupNorm(4)+SiLU
             adaptive_pool(6,8)   -> flatten -> Linear(32*6*8  -> 128)
    play     4->32/5/2, 32->64/3/2, 64->64/3/2   pad=k//2, GroupNorm(4)+SiLU
             adaptive_pool(14,12) -> flatten -> Linear(64*14*12 -> 128)
    hud      1->16/3/1, 16->32/3/2, 32->32/3/2   pad=k//2, GroupNorm(4)+SiLU
             adaptive_pool(28,10) -> flatten -> Linear(32*28*10 -> 64)
    history  65 = 56 executed key bits + 8 validity flags + 1 capture interval
             Linear(65->32) + SiLU
    fusion   concat 352 -> Linear(352->256) -> LayerNorm(256) -> SiLU
    memory   GRU(input=256, hidden=256, num_layers=1, bias=True)
    heads    Linear(256->9), Linear(256->2), Linear(256->2), Linear(256->2)

Every conv and every linear carries a bias (GRU keeps its standard two biases).
The pooled grid is kept as a **spatial grid** and flattened afterwards — the
streams never average all positions away before the projection.

Deliberate refusals (each one is a way a broken caller could otherwise produce a
plausible-looking but wrong training signal):

1. **no automatic SHOOT** — ``shoot`` is an ordinary two-way classification head;
   index 0 means "do not shoot" and is never OR-ed on, biased, or renormalised
   away (the v4.2.0 defect that :mod:`touhou_rl.v430.actions` already fixed must
   not come back in the model layer).  :func:`_assert_no_forced_shoot` refuses to
   run the model at all if ``contracts.forced_shoot()`` ever becomes ``True`` or
   ``contracts.truth_action_mask()`` becomes ``True``.
2. **no privileged input** — see :meth:`ActorPolicy._as_batch`; ``stage_id``,
   ``seed``, ``engine_frame``, ... raise :class:`PrivilegedModelInputError`
   through the :mod:`touhou_rl.v430.contracts` /
   :mod:`touhou_rl.v430.observation` sentinel, and an unknown keyword raises too.
3. **no silent shape repair** — a wrong shape raises :class:`ShapeMismatchError`;
   the model never interpolates, pads, reshapes or squeezes an image into
   something that merely runs.
4. **explicit GRU state** — ``hidden=None`` means "zero initial state"; a supplied
   hidden is shape-asserted to ``(1, B, 256)`` and refused otherwise.
5. **no batch statistics** — only ``GroupNorm`` / ``LayerNorm``; ``BatchNorm`` is
   absent by construction, and :meth:`ActorPolicy.parameter_report` reports the
   (empty) list of statistics-tracking layers so a test can assert it.
6. **reproducible sampling** — :func:`sample` accepts a ``torch.Generator`` and
   never touches global RNG state.

Normalisation: ``/255.0`` happens **inside** :meth:`ActorPolicy.forward`, once,
on the three image streams only.  A float image handed in from outside is refused
(it has already been normalised somewhere else — a silent 1/65025 error), and key
bits / validity flags / the capture interval are never divided.

The ``memory`` field of an :class:`~touhou_rl.v430.contracts.ActorObservation` is
validated for shape/dtype but is **not** used as the GRU state: the recurrent
state must be passed explicitly as ``hidden``, so no teacher- or truth-filled
``memory`` vector can steer the recurrence.  This is stricter than the guide
requires and is deliberate.

This module is a **synthetic unit-test target**: importing it and running its
tests proves the architecture and the interface arithmetic, not game capability.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from touhou_rl.v430 import contracts, observation

__all__ = [
    # spec
    "MODEL_SPEC",
    "HEAD_ORDER",
    "HEAD_SIZES",
    "JOINT_COUNT",
    "ACTOR_PARAMETER_SCOPE",
    "AUXILIARY_HEADS",
    "OBS_FIELD_KEYS",
    "IMAGE_FIELDS",
    "HISTORY_FEATURES",
    # errors
    "ModelError",
    "PrivilegedModelInputError",
    "ShapeMismatchError",
    "HiddenStateError",
    # structures
    "HeadLogits",
    "Evaluation",
    # model
    "ActorPolicy",
    "build_actor",
    # pure functions on logits
    "joint_log_prob",
    "joint_entropy",
    "sample",
    # report / self-checks
    "parameter_report",
    "assert_no_forced_shoot",
    "activation_of",
]

# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class ModelError(contracts.ContractError):
    """Base class of every actor-model contract violation (a ``ContractError``)."""


class PrivilegedModelInputError(ModelError):
    """Truth / teacher / judge state (or an unknown field) reached the model."""


class ShapeMismatchError(ModelError):
    """An input tensor does not have the shape the observation contract froze."""


class HiddenStateError(ModelError):
    """The recurrent state is not a finite ``(1, B, 256)`` float32 tensor."""


# ---------------------------------------------------------------------------
# the spec table: a verbatim mirror of PLAN_SPEC.model, used to build the module
# ---------------------------------------------------------------------------
#: ``PLAN_SPEC.model`` — kept as one dict so every number below is read once.
MODEL_SPEC: dict[str, Any] = {
    "candidate": "three-stream spatial CNN plus GRU",
    "hidden_size": 256,
    "parameter_range_millions": [1, 3],
    "parameter_range_is_gate": False,
    "normalization": "GroupNorm or LayerNorm; no mutable batch statistics",
    "activation": "SiLU",
    "encoder_candidate": {
        "global": {
            # (in_channels, out_channels, kernel_size, stride), padding = k // 2
            "convs": [[3, 16, 5, 2], [16, 32, 3, 2], [32, 32, 3, 2]],
            "adaptive_pool_hw": [6, 8],
            "project_features": 128,
        },
        "play": {
            "convs": [[4, 32, 5, 2], [32, 64, 3, 2], [64, 64, 3, 2]],
            "adaptive_pool_hw": [14, 12],
            "project_features": 128,
        },
        "hud": {
            "convs": [[1, 16, 3, 1], [16, 32, 3, 2], [32, 32, 3, 2]],
            "adaptive_pool_hw": [28, 10],
            "project_features": 64,
        },
        "history": {"flattened_features": 65, "project_features": 32},
        "fusion_features": 352,
        "fusion_projection": 256,
        "conv_padding": "kernel_size//2",
        "conv_norm_groups": 4,
    },
    "action_heads_input": 256,
    "auxiliary_heads": [
        "player", "visible_entities", "motion_and_danger",
        "visible_resources", "tactical_goal", "future_outcome",
    ],
    "parameter_scope": (
        "actor encoder, memory and action heads; critic and auxiliary heads "
        "counted separately"),
}

#: ``PLAN_SPEC.model.parameter_scope`` restated as the actor-only scope string the
#: parameter report carries, so "actor" can never silently include a critic.
ACTOR_PARAMETER_SCOPE: str = MODEL_SPEC["parameter_scope"]
#: The six auxiliary heads of ``PLAN_SPEC.model.auxiliary_heads``.  They are *not*
#: implemented in this slice, and the report lists them as excluded.
AUXILIARY_HEADS: tuple[str, ...] = tuple(MODEL_SPEC["auxiliary_heads"])

#: The four action heads, in ``contracts.HEAD_ORDER`` (also the joint-index order).
HEAD_ORDER: tuple[str, ...] = contracts.HEAD_ORDER
HEAD_SIZES: dict[str, int] = dict(contracts.HEAD_SIZES)
JOINT_COUNT: int = contracts.JOINT_COUNT

#: ``PLAN_SPEC.observation.O2`` field names, in contract order (``memory`` last).
OBS_FIELD_KEYS: tuple[str, ...] = contracts.ACTOR_OBSERVATION_KEYS
#: The three pixel streams; ``/255.0`` applies to these and nothing else.
IMAGE_FIELDS: tuple[str, ...] = ("global_rgb", "play_gray", "hud_gray")
#: The six fields the model needs from an observation (``memory`` is optional and
#: is never used as the recurrent state, see the module docstring).
REQUIRED_FIELDS: tuple[str, ...] = tuple(
    name for name in OBS_FIELD_KEYS if name != "memory")

#: ``(C, H, W)`` of each image stream, reused from the observation contract.
O2_SHAPES: dict[str, tuple[int, ...]] = observation.O2_SHAPES

_NUM_GROUPS: int = int(MODEL_SPEC["encoder_candidate"]["conv_norm_groups"])
_HIDDEN: int = int(MODEL_SPEC["hidden_size"])
_FUSION_FEATURES: int = int(MODEL_SPEC["encoder_candidate"]["fusion_features"])
_HISTORY_FEATURES: int = int(
    MODEL_SPEC["encoder_candidate"]["history"]["flattened_features"])
_HISTORY_PROJECT: int = int(
    MODEL_SPEC["encoder_candidate"]["history"]["project_features"])
#: 65 = 56 executed key bits + 8 validity flags + 1 capture interval (guide §4.3).
HISTORY_FEATURES: int = _HISTORY_FEATURES

#: ``PLAN_SPEC.observation.O2.normalize_divisor``; the only place ``/255`` happens.
_NORMALIZE_DIVISOR: float = float(contracts.O2_NORMALIZE_DIVISOR)

# ---------------------------------------------------------------------------
# import-time cross-checks: the plan, the observation contract and this module
# must be the same table (a frozen constant that drifts is worse than none)
# ---------------------------------------------------------------------------
def _stream_in_channels() -> dict[str, int]:
    candidate = MODEL_SPEC["encoder_candidate"]
    return {name: int(candidate[name]["convs"][0][0]) for name in IMAGE_FIELDS}


def _check_spec_consistency() -> None:
    candidate = MODEL_SPEC["encoder_candidate"]
    expected_channels = {
        "global_rgb": O2_SHAPES["global_rgb"][0],
        "play_gray": O2_SHAPES["play_gray"][0],
        "hud_gray": O2_SHAPES["hud_gray"][0],
    }
    streams = {"global": "global_rgb", "play": "play_gray", "hud": "hud_gray"}
    for stream, field in streams.items():
        declared = int(candidate[stream]["convs"][0][0])
        observed = int(expected_channels[field])
        if declared != observed:
            raise ModelError(  # pragma: no cover - frozen tables
                f"encoder stream {stream!r} takes {declared} input channel(s) but "
                f"O2 field {field!r} has {observed}; the network and the "
                f"observation contract disagree")
    projections = [int(candidate[name]["project_features"])
                   for name in streams] + [_HISTORY_PROJECT]
    if sum(projections) != _FUSION_FEATURES:
        raise ModelError(  # pragma: no cover - frozen tables
            f"stream projections {projections} sum to {sum(projections)}, but "
            f"PLAN_SPEC fusion_features is {_FUSION_FEATURES}")
    if _HISTORY_FEATURES != 56 + 8 + 1:
        raise ModelError(  # pragma: no cover - frozen table
            f"history flattened_features is {_HISTORY_FEATURES}, but guide §4.3 "
            f"defines 56 key bits + 8 validity flags + 1 interval = 65")
    if _HIDDEN != O2_SHAPES["memory"][0]:
        raise ModelError(  # pragma: no cover - frozen tables
            f"GRU hidden size {_HIDDEN} != observation memory shape "
            f"{O2_SHAPES['memory']}; the recurrent state is the memory field")
    if int(MODEL_SPEC["action_heads_input"]) != _HIDDEN:
        raise ModelError(  # pragma: no cover - frozen table
            f"action_heads_input {MODEL_SPEC['action_heads_input']} != hidden "
            f"size {_HIDDEN}")
    if tuple(HEAD_SIZES) != HEAD_ORDER:
        raise ModelError(  # pragma: no cover - frozen table
            f"head order {HEAD_ORDER} disagrees with sizes {tuple(HEAD_SIZES)}")
    if JOINT_COUNT != math.prod(HEAD_SIZES[head] for head in HEAD_ORDER):
        raise ModelError(  # pragma: no cover - frozen table
            f"joint_count {JOINT_COUNT} != product of head sizes {HEAD_SIZES}")
    if _NUM_GROUPS < 1:
        raise ModelError(  # pragma: no cover - frozen table
            f"conv_norm_groups {_NUM_GROUPS} < 1")


_check_spec_consistency()


def activation_of(name: str) -> nn.Module:
    """Build the plan's activation (``PLAN_SPEC.model.activation`` == ``SiLU``)."""

    if name != "SiLU":  # pragma: no cover - frozen table
        raise ModelError(f"PLAN_SPEC activation {name!r} is not implemented here")
    return nn.SiLU()


# ---------------------------------------------------------------------------
# the no-forced-shoot gate
# ---------------------------------------------------------------------------
def assert_no_forced_shoot() -> None:
    """Refuse to run the actor if the contract ever forces the shoot bit.

    The v4.2.0 defect was ``shoot=0`` becoming indistinguishable from ``shoot=1``
    because the compiler OR-ed the bit on.  The model layer must not reintroduce
    it from its side, so this reads the contract's two switches and raises rather
    than trusting the caller.  It is a *self-check*, not a gate on the game.
    """

    if contracts.forced_shoot():
        raise ModelError(
            "contracts.forced_shoot() is True: the shoot head would collapse to "
            "'always shoot' and shoot=0 would stop meaning 'not shooting'. The "
            "actor refuses to run under a forced-shoot contract")
    if contracts.truth_action_mask():
        raise ModelError(
            "contracts.truth_action_mask() is True: no truth-derived action "
            "masking may reach the actor")


def _assert_no_forced_shoot() -> None:  # internal alias used in hot paths
    assert_no_forced_shoot()


# ---------------------------------------------------------------------------
# the four head logits
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True, eq=False)
class HeadLogits:
    """The four action heads' **raw logits** (never probabilities, never masked).

    ``move`` is ``(..., 9)`` and ``focus``/``shoot``/``bomb`` are ``(..., 2)``;
    the leading batch dimension is present for a batched input and absent for a
    single observation.  ``shoot[..., 0]`` is a genuine "do not shoot" logit: no
    head is biased, masked or renormalised here.
    """

    move: torch.Tensor
    focus: torch.Tensor
    shoot: torch.Tensor
    bomb: torch.Tensor

    def __post_init__(self) -> None:
        tensors = self.as_dict()
        ranks = set()
        leading = set()
        for head in HEAD_ORDER:
            value = tensors[head]
            if not isinstance(value, torch.Tensor):
                raise ModelError(
                    f"head {head!r} logits are {type(value).__name__}, expected a "
                    f"torch.Tensor")
            if not torch.is_floating_point(value):
                raise ModelError(
                    f"head {head!r} logits dtype is {value.dtype}, expected a "
                    f"floating dtype")
            expected = HEAD_SIZES[head]
            if value.shape[-1] != expected:
                raise ShapeMismatchError(
                    f"head {head!r} has {value.shape[-1]} logit(s) on its last "
                    f"axis, expected {expected} (contracts.HEAD_SIZES)")
            ranks.add(value.ndim)
            leading.add(tuple(value.shape[:-1]))
        if len(ranks) != 1:
            raise ShapeMismatchError(
                f"the four heads disagree on rank: {sorted(ranks)}; they must all "
                f"be unbatched (-,head) or all batched (B,head)")
        if len(leading) != 1:
            raise ShapeMismatchError(
                f"the four heads disagree on their leading shape: {sorted(leading)}")

    # -- containers ---------------------------------------------------------
    def as_dict(self) -> dict[str, torch.Tensor]:
        return {head: getattr(self, head) for head in HEAD_ORDER}

    def __getitem__(self, head: str) -> torch.Tensor:
        if head not in HEAD_ORDER:
            raise ModelError(f"unknown head {head!r}; known {HEAD_ORDER}")
        return getattr(self, head)

    def head(self, name: str) -> torch.Tensor:
        """The logits of one head, by name."""
        return self[name]

    # -- geometry ----------------------------------------------------------
    @property
    def batched(self) -> bool:
        """``True`` when the heads carry a leading batch dimension."""
        return self.move.ndim == 2

    @property
    def batch_size(self) -> int:
        """The batch size, or ``1`` for an unbatched observation."""
        return int(self.move.shape[0]) if self.batched else 1

    # -- logging -----------------------------------------------------------
    def detach(self) -> "HeadLogits":
        """A detached copy, for writing a log without holding the graph."""
        return HeadLogits(**{head: value.detach() for head, value in self.as_dict().items()})

    def tolist(self) -> dict[str, list[float]]:
        """Plain Python floats per head (a record must not carry a device tensor)."""
        return {head: [float(x) for x in value.detach().reshape(-1)]
                for head, value in self.as_dict().items()}


@dataclasses.dataclass(frozen=True, eq=False)
class Evaluation:
    """The result of :meth:`ActorPolicy.evaluate` (the PPO recompute path).

    ``joint_log_prob``/``joint_entropy`` are filled only when the caller passed the
    action ids that were actually taken.
    """

    head_logits: HeadLogits
    hidden_next: torch.Tensor
    joint_log_prob: torch.Tensor | None = None
    joint_entropy: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# one convolutional stream: conv/norm/activation stack -> grid pool -> flatten
# ---------------------------------------------------------------------------
def _conv_output_size(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


class _ConvStream(nn.Module):
    """One spatial stream of guide §4.3: three convs, a grid pool, a projection.

    The pool is :class:`torch.nn.AdaptiveAvgPool2d` onto the plan's **grid**
    (``6x8`` / ``14x12`` / ``28x10``) and the grid is flattened only afterwards, so
    the projection keeps positional structure.  Every conv has ``bias=True`` and
    ``padding = kernel // 2``; normalisation is ``GroupNorm(groups, C)`` (never
    BatchNorm, which would make the output depend on the batch it is run in).
    """

    def __init__(self, *, name: str, spec: Mapping[str, Any], input_hw: tuple[int, int],
                 out_channels: int | None = None) -> None:
        super().__init__()
        self.name = str(name)
        convs = [tuple(int(value) for value in conv) for conv in spec["convs"]]
        pool_hw = tuple(int(value) for value in spec["adaptive_pool_hw"])
        project_features = int(spec["project_features"])
        if len(pool_hw) != 2 or pool_hw[0] < 1 or pool_hw[1] < 1:  # pragma: no cover
            raise ModelError(f"{self.name}: adaptive_pool_hw {pool_hw} is not a grid")
        blocks: list[nn.Module] = []
        height, width = input_hw
        channels = None
        for index, (in_channels, out_channels_i, kernel, stride) in enumerate(convs):
            if index and in_channels != channels:  # pragma: no cover - frozen table
                raise ModelError(
                    f"{self.name}: conv {index} takes {in_channels} channels but the "
                    f"previous conv emits {channels}")
            if stride < 1 or kernel < 1:  # pragma: no cover - frozen table
                raise ModelError(f"{self.name}: conv {index} k={kernel} s={stride}")
            if out_channels_i % _NUM_GROUPS:
                raise ModelError(
                    f"{self.name}: conv {index} emits {out_channels_i} channels, "
                    f"which GroupNorm({_NUM_GROUPS}) cannot split")
            padding = kernel // 2
            blocks.append(nn.Conv2d(in_channels, out_channels_i, kernel, stride=stride,
                                    padding=padding, bias=True))
            blocks.append(nn.GroupNorm(_NUM_GROUPS, out_channels_i))
            blocks.append(activation_of(MODEL_SPEC["activation"]))
            height = _conv_output_size(height, kernel, stride, padding)
            width = _conv_output_size(width, kernel, stride, padding)
            channels = out_channels_i
        if height < 1 or width < 1:  # pragma: no cover - frozen table
            raise ModelError(f"{self.name}: the conv stack empties a {input_hw} input")
        self.input_hw = (int(input_hw[0]), int(input_hw[1]))
        self.input_channels = int(convs[0][0])
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(pool_hw)
        self.out_channels = int(channels)  # type: ignore[arg-type]
        self.pool_hw = pool_hw
        self.pre_pool_hw = (height, width)
        self.project = nn.Linear(
            self.out_channels * pool_hw[0] * pool_hw[1], project_features, bias=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        expected = (self.input_channels, *self.input_hw)
        if tuple(images.shape[1:]) != expected:
            raise ShapeMismatchError(
                f"{self.name} stream expects (B, C, H, W) with C,H,W = {expected}, "
                f"got {tuple(images.shape)}")
        features = self.body(images)
        if tuple(features.shape[2:]) != self.pre_pool_hw:  # pragma: no cover
            raise ModelError(
                f"{self.name}: the conv stack produced a {tuple(features.shape[2:])} "
                f"grid, expected {self.pre_pool_hw}")
        grid = self.pool(features)
        if tuple(grid.shape[2:]) != self.pool_hw:  # pragma: no cover
            raise ModelError(
                f"{self.name}: pooling produced {tuple(grid.shape[2:])}, expected "
                f"{self.pool_hw}")
        return self.project(torch.flatten(grid, start_dim=1))


# ---------------------------------------------------------------------------
# input handling: validate the observation, refuse everything else
# ---------------------------------------------------------------------------
def _as_field_mapping(obs: Any) -> dict[str, Any]:
    """``obs`` -> ``{field: value}`` for an observation record or a mapping."""

    if isinstance(obs, Mapping):
        return dict(obs)
    if dataclasses.is_dataclass(obs) and not isinstance(obs, type):
        return {field.name: getattr(obs, field.name)
                for field in dataclasses.fields(obs)}
    raise ModelError(
        f"obs must be an ActorObservation / O2Observation / mapping of the O2 "
        f"fields / a tensor bundle, got {type(obs).__name__}; the model has no "
        f"other input surface")


def _to_tensor(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    try:
        import numpy as np  # local: the model must not need numpy to import

        array = np.asarray(value)
    except Exception as exc:  # noqa: BLE001 - an unreadable value is a caller error
        raise ShapeMismatchError(
            f"{name}: value of type {type(value).__name__} is neither a torch "
            f"tensor nor convertible to an array: {exc}") from exc
    try:
        return torch.as_tensor(array)
    except Exception as exc:  # noqa: BLE001
        raise ShapeMismatchError(
            f"{name}: array of dtype {getattr(array, 'dtype', '?')} and shape "
            f"{getattr(array, 'shape', '?')} cannot become a tensor: {exc}") from exc


_DTYPES: dict[str, torch.dtype] = {
    "uint8": torch.uint8,
    "bool": torch.bool,
    "float32": torch.float32,
}


def _check_field(name: str, value: Any) -> torch.Tensor:
    """Dtype-check one O2 field and return it as a tensor (nothing is cast)."""

    tensor = _to_tensor(value, name=name)
    expected_dtype = _DTYPES[contracts.FIELD_SPECS[name].dtype]
    if tensor.dtype is not expected_dtype:
        hint = ""
        if tensor.dtype.is_floating_point and expected_dtype is torch.uint8:
            hint = ("; a float image has already been normalised — /255 belongs "
                    "inside the model (forward), not to the caller")
        raise ShapeMismatchError(
            f"{name}: dtype {tensor.dtype} != required {expected_dtype}{hint}")
    return tensor


def _split_batch(name: str, tensor: torch.Tensor, expected: tuple[int, ...]
                 ) -> tuple[torch.Tensor, bool]:
    """Return ``(tensor, batched)`` after refusing every shape but the frozen one."""

    shape = tuple(int(dim) for dim in tensor.shape)
    if shape == expected:
        return tensor, False
    if len(shape) == len(expected) + 1 and shape[1:] == expected:
        return tensor, True
    raise ShapeMismatchError(
        f"{name}: shape {shape} is neither {expected} nor (B,)+{expected}; the "
        f"model never interpolates, pads or reshapes an observation into another "
        f"shape (guide §4.2: '禁止自动插值成另一形状')")


# ---------------------------------------------------------------------------
# the actor
# ---------------------------------------------------------------------------
class ActorPolicy(nn.Module):
    """The guide §4.3 actor: three CNN streams + history -> 256 -> GRU -> 4 heads.

    Parameters
    ----------
    None.  Every number comes from :data:`MODEL_SPEC`; constructing an actor with a
    different architecture requires editing that table (and then
    :func:`_check_spec_consistency` re-checks it against the observation contract).

    Examples
    --------
    >>> policy = build_actor()                     # doctest: +SKIP
    >>> logits, hidden = policy(obs)               # doctest: +SKIP
    >>> action, log_prob = sample(logits)          # doctest: +SKIP
    """

    def __init__(self) -> None:
        super().__init__()
        candidate = MODEL_SPEC["encoder_candidate"]
        self.streams = nn.ModuleDict()
        for stream, field in (("global", "global_rgb"), ("play", "play_gray"),
                              ("hud", "hud_gray")):
            shape = O2_SHAPES[field]
            self.streams[stream] = _ConvStream(
                name=stream, spec=candidate[stream], input_hw=(shape[1], shape[2]))
        self.history_branch = nn.Sequential(
            nn.Linear(_HISTORY_FEATURES, _HISTORY_PROJECT, bias=True),
            activation_of(MODEL_SPEC["activation"]),
        )
        self.fusion = nn.Sequential(
            nn.Linear(_FUSION_FEATURES, _HIDDEN, bias=True),
            nn.LayerNorm(_HIDDEN),
            activation_of(MODEL_SPEC["activation"]),
        )
        self.recurrent = nn.GRU(_HIDDEN, _HIDDEN, num_layers=1, bias=True,
                                batch_first=False)
        self.heads = nn.ModuleDict(
            {head: nn.Linear(_HIDDEN, HEAD_SIZES[head], bias=True)
             for head in HEAD_ORDER})

    # -- geometry ----------------------------------------------------------
    @property
    def hidden_size(self) -> int:
        """The GRU hidden size (also the observation ``memory`` length)."""
        return _HIDDEN

    @property
    def num_layers(self) -> int:
        return int(self.recurrent.num_layers)

    @property
    def fusion_features(self) -> int:
        return _FUSION_FEATURES

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -- input validation --------------------------------------------------
    def _as_batch(self, obs: Any, *, extra: Mapping[str, Any] | None = None
                  ) -> tuple[dict[str, torch.Tensor], bool]:
        """Validate ``obs`` and return ``({field: batched tensor}, batched)``.

        The six O2 fields are mandatory and must agree on batchedness; ``memory``
        is optional and only dtype/shape-checked (it is not the GRU state, see the
        module docstring).  A privileged name raises
        :class:`PrivilegedModelInputError`; an unknown name raises
        :class:`ModelError`.
        """

        data = _as_field_mapping(obs)
        if extra:
            data = {**data, **dict(extra)}
        try:
            observation.assert_actor_inputs_clean(data, what="actor model input")
        except observation.ObservationError as exc:
            raise PrivilegedModelInputError(str(exc)) from exc
        except contracts.ContractError as exc:  # pragma: no cover - defensive
            raise PrivilegedModelInputError(str(exc)) from exc
        unknown = sorted(name for name in data if name not in OBS_FIELD_KEYS)
        if unknown:
            raise PrivilegedModelInputError(
                f"actor model input carries unregistered field(s) {unknown}; the "
                f"input surface is exactly {list(OBS_FIELD_KEYS)} "
                f"(contract forbids {list(contracts.FORBIDDEN_ACTOR_FIELDS)})")
        missing = [name for name in REQUIRED_FIELDS if name not in data]
        if missing:
            raise ShapeMismatchError(
                f"actor model input is missing field(s) {missing}; a missing "
                f"observation field cannot be zero-filled")
        checked: dict[str, torch.Tensor] = {}
        batched_flags: set[bool] = set()
        for name in REQUIRED_FIELDS:
            tensor = _check_field(name, data[name])
            expected = tuple(int(dim) for dim in O2_SHAPES[name])
            checked[name], batched = _split_batch(name, tensor, expected)
            batched_flags.add(batched)
        if len(batched_flags) != 1:
            raise ShapeMismatchError(
                f"the observation fields disagree on batchedness: "
                f"{ {name: tuple(value.shape) for name, value in checked.items()} }; "
                f"one decision's fields are all unbatched or all batched")
        batched = batched_flags.pop()
        batch_size = int(checked["global_rgb"].shape[0]) if batched else 1
        for name, tensor in checked.items():
            observed = int(tensor.shape[0]) if batched else 1
            if observed != batch_size:
                raise ShapeMismatchError(
                    f"{name}: batch size {observed} != {batch_size} from "
                    f"global_rgb; a short field cannot be broadcast")
        if not batched:
            checked = {name: tensor.unsqueeze(0) for name, tensor in checked.items()}
        self._check_field_values(checked)
        if "memory" in data:
            self._check_memory(data["memory"], batched=batched, batch_size=batch_size)
        return checked, batched

    def _check_field_values(self, batch: Mapping[str, torch.Tensor]) -> None:
        """Refuse key bits outside 0/1, a holed validity window, a bad interval."""

        keys = batch["executed_key_history"]
        if keys.numel():
            allowed = torch.tensor([0, 1], dtype=keys.dtype, device=keys.device)
            if not bool(torch.isin(keys, allowed).all()):
                values = sorted({int(v) for v in torch.unique(keys).tolist()})
                raise ShapeMismatchError(
                    f"executed_key_history holds values outside {{0,1}}: {values}")
        for row in batch["history_valid"]:
            try:
                contracts.check_history_valid_prefix([bool(flag) for flag in row])
            except contracts.ContractError as exc:
                raise ShapeMismatchError(f"history_valid: {exc}") from exc
        interval = batch["elapsed_capture_s"]
        if not bool(torch.isfinite(interval).all()):
            raise ShapeMismatchError(
                "elapsed_capture_s holds a non-finite value; it would poison the "
                "history projection")
        if bool((interval < 0).any()):
            raise ShapeMismatchError("elapsed_capture_s holds a negative interval")

    def _check_memory(self, memory: Any, *, batched: bool, batch_size: int) -> None:
        tensor = _check_field("memory", memory)
        expected = tuple(int(dim) for dim in O2_SHAPES["memory"])
        _, memory_batched = _split_batch("memory", tensor, expected)
        if memory_batched != batched:
            raise ShapeMismatchError(
                f"memory: batchedness {memory_batched} disagrees with the "
                f"observation's {batched}")
        if memory_batched and int(tensor.shape[0]) != batch_size:  # pragma: no cover
            raise ShapeMismatchError(
                f"memory: batch size {int(tensor.shape[0])} != {batch_size}")
        if not bool(torch.isfinite(tensor).all()):
            raise ShapeMismatchError("memory holds a non-finite value")

    def _check_hidden(self, hidden: Any, *, batch_size: int,
                      device: torch.device) -> torch.Tensor:
        """Assert the GRU state is ``(num_layers, B, 256)`` float32 and finite."""

        expected = (self.num_layers, batch_size, _HIDDEN)
        if hidden is None:
            return torch.zeros(expected, dtype=torch.float32, device=device)
        if not isinstance(hidden, torch.Tensor):
            raise HiddenStateError(
                f"hidden must be a torch.Tensor of shape {expected} or None (which "
                f"means a zero initial state), got {type(hidden).__name__}")
        shape = tuple(int(dim) for dim in hidden.shape)
        if shape != expected:
            raise HiddenStateError(
                f"hidden shape {shape} != {expected} (num_layers, batch, "
                f"hidden_size); the recurrent state is never reshaped or "
                f"broadcast to fit")
        if hidden.dtype is not torch.float32:
            raise HiddenStateError(
                f"hidden dtype is {hidden.dtype}, required float32")
        if not bool(torch.isfinite(hidden).all()):
            raise HiddenStateError("hidden holds a non-finite value")
        if hidden.device != device:
            raise HiddenStateError(
                f"hidden is on {hidden.device} but the model is on {device}; move "
                f"the state yourself so a silent device copy cannot mask a bug")
        return hidden

    def _check_device(self, batch: Mapping[str, torch.Tensor]) -> None:
        device = self.device
        for name, tensor in batch.items():
            if tensor.device != device:
                raise ModelError(
                    f"{name} is on {tensor.device} but the model is on {device}; "
                    f"move the observation yourself (silent device moves are not "
                    f"performed)")

    # -- encoding ----------------------------------------------------------
    def _encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """One observation batch -> the 256-dim fused feature vector."""

        parts: list[torch.Tensor] = []
        for stream, field in (("global", "global_rgb"), ("play", "play_gray"),
                              ("hud", "hud_gray")):
            images = batch[field].to(torch.float32) / _NORMALIZE_DIVISOR
            parts.append(self.streams[stream](images))
        keys = batch["executed_key_history"].reshape(batch["global_rgb"].shape[0], -1)
        history = torch.cat(
            [keys.to(torch.float32),
             batch["history_valid"].to(torch.float32),
             batch["elapsed_capture_s"].to(torch.float32).reshape(-1, 1)],
            dim=1)
        if history.shape[1] != HISTORY_FEATURES:  # pragma: no cover - derived
            raise ModelError(
                f"history vector is {history.shape[1]}-dim, expected "
                f"{HISTORY_FEATURES}")
        parts.append(self.history_branch(history))
        fused = torch.cat(parts, dim=1)
        if fused.shape[1] != _FUSION_FEATURES:  # pragma: no cover - derived
            raise ModelError(
                f"fusion vector is {fused.shape[1]}-dim, expected {_FUSION_FEATURES}")
        return self.fusion(fused)

    def _head_logits(self, features: torch.Tensor) -> HeadLogits:
        return HeadLogits(**{head: self.heads[head](features) for head in HEAD_ORDER})

    # -- forward -----------------------------------------------------------
    def forward(self, obs: Any, hidden: Any = None, **unexpected: Any
                ) -> tuple[HeadLogits, torch.Tensor]:
        """Encode one observation (or a batch) and step the GRU once.

        Parameters
        ----------
        obs:
            An :class:`~touhou_rl.v430.contracts.ActorObservation`, an
            :class:`~touhou_rl.v430.observation.O2Observation`, or a mapping of the
            O2 fields to ``uint8``/``bool``/``float32`` tensors (or arrays).  A
            single observation is unbatched; adding a leading ``B`` makes it a
            batch.  Every other input is refused.
        hidden:
            ``None`` for a zero initial state, else a finite ``float32``
            ``(1, B, 256)`` tensor (an earlier ``hidden_next``).

        Returns
        -------
        (HeadLogits, torch.Tensor)
            The four raw head logits (no batch dim when the input had none) and the
            next GRU state, shape ``(1, B, 256)``.
        """

        if unexpected:
            raise PrivilegedModelInputError(
                f"forward() was handed unknown keyword(s) {sorted(unexpected)}; the "
                f"input surface is obs + hidden only")
        _assert_no_forced_shoot()
        batch, batched = self._as_batch(obs)
        self._check_device(batch)
        batch_size = int(batch["global_rgb"].shape[0])
        state = self._check_hidden(hidden, batch_size=batch_size, device=self.device)
        features = self._encode(batch)
        out, next_state = self.recurrent(features.unsqueeze(0), state)
        logits = self._head_logits(out.squeeze(0))
        if not batched:
            logits = HeadLogits(**{head: value.squeeze(0)
                                   for head, value in logits.as_dict().items()})
        return logits, next_state

    # -- PPO recompute -----------------------------------------------------
    def evaluate(self, obs: Any, *, hidden: Any = None, ids: Any = None) -> Evaluation:
        """Recompute the four head logits for an observation (no sampling, no RNG).

        This is the PPO path: given a stored observation and the action ids that
        were taken, it returns the *same* numbers the rollout logged (same weights,
        same shapes, same temperature 1.0).  ``ids`` is optional; supply it to also
        get the joint log-probability and the joint entropy.
        """

        logits, next_state = self.forward(obs, hidden)
        if ids is None:
            return Evaluation(head_logits=logits, hidden_next=next_state)
        return Evaluation(
            head_logits=logits,
            hidden_next=next_state,
            joint_log_prob=_joint_log_prob(logits, ids),
            joint_entropy=_joint_entropy(logits),
        )

    # -- report ------------------------------------------------------------
    def parameter_report(self) -> dict[str, Any]:
        """Parameter counts per stream / branch / head, and the total.

        The scope is the **actor only**: ``includes_critic`` and
        ``includes_auxiliary_heads`` are ``False`` and the six auxiliary heads of
        ``PLAN_SPEC.model.auxiliary_heads`` are listed as excluded, so an actor
        number can never absorb a critic number (guide §4.3: "critic和辅助头单列
        参数…不能藏进actor指标").
        """

        return parameter_report(self)


# ---------------------------------------------------------------------------
# parameter report
# ---------------------------------------------------------------------------
def _count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def _tracks_batch_statistics(module: nn.Module) -> bool:
    """``True`` for any layer holding mutable per-batch statistics.

    Detection is by mechanism, not by class name: a normalisation layer that runs
    with the batch's own statistics keeps ``running_mean``/``running_var``
    buffers.  The actor has none, and this module deliberately never names a
    ``BatchNorm`` class (a test asserts that over the AST), so the property is
    checked structurally.
    """

    buffers = getattr(module, "_buffers", {})
    return any(name in buffers for name in ("running_mean", "running_var",
                                            "num_batches_tracked"))


def _count_all(modules: Sequence[nn.Module]) -> int:
    return sum(_count(module) for module in modules)


def _stream_report(stream: _ConvStream) -> dict[str, Any]:
    convs = [module for module in stream.body if isinstance(module, nn.Conv2d)]
    norms = [module for module in stream.body if isinstance(module, nn.GroupNorm)]
    return {
        "convs": [list(conv.weight.shape) for conv in convs],
        "input_hw": [int(dim) for dim in stream.input_hw],
        "pre_pool_hw": [int(dim) for dim in stream.pre_pool_hw],
        "adaptive_pool_hw": [int(dim) for dim in stream.pool_hw],
        "out_channels": stream.out_channels,
        "project_features": int(stream.project.out_features),
        "groups": sorted({int(norm.num_groups) for norm in norms}),
        "conv_and_norm_parameters": _count_all(convs + norms),
        "projection_parameters": _count(stream.project),
        "total": _count(stream),
    }


def parameter_report(policy: ActorPolicy | None = None) -> dict[str, Any]:
    """Count the actor's parameters, per stream / branch / head, plus the total.

    Refuses to answer if the parts do not add up to ``sum(p.numel())`` — a report
    that does not add up is worse than no report.
    """

    policy = build_actor() if policy is None else policy
    if not isinstance(policy, ActorPolicy):
        raise ModelError(
            f"parameter_report expects an ActorPolicy, got {type(policy).__name__}")
    streams = {name: _stream_report(policy.streams[name])
               for name in ("global", "play", "hud")}
    heads = {head: _count(policy.heads[head]) for head in HEAD_ORDER}
    parts = {
        "global_stream": streams["global"]["total"],
        "play_stream": streams["play"]["total"],
        "hud_stream": streams["hud"]["total"],
        "history_branch": _count(policy.history_branch),
        "fusion": _count(policy.fusion),
        "memory_gru": _count(policy.recurrent),
        "action_heads": sum(heads.values()),
    }
    total = sum(parts.values())
    measured = _count(policy)
    if total != measured:
        raise ModelError(
            f"parameter report adds up to {total} but the module holds {measured}; "
            f"a stream or a head is not being counted")
    batch_norm = sorted({
        type(module).__name__ for module in policy.modules()
        if _tracks_batch_statistics(module)})
    report = {
        "scope": "actor",
        "parameter_scope": ACTOR_PARAMETER_SCOPE,
        "includes_critic": False,
        "includes_auxiliary_heads": False,
        "auxiliary_heads_not_included": list(AUXILIARY_HEADS),
        "model_spec_parameter_range_millions": list(
            MODEL_SPEC["parameter_range_millions"]),
        "parameter_range_is_gate": bool(MODEL_SPEC["parameter_range_is_gate"]),
        "streams": streams,
        "history": {
            "flattened_features": HISTORY_FEATURES,
            "project_features": _HISTORY_PROJECT,
            "total": parts["history_branch"],
        },
        "fusion": {
            "features": _FUSION_FEATURES,
            "projection": _HIDDEN,
            "total": parts["fusion"],
        },
        "memory_gru": {
            "hidden_size": _HIDDEN,
            "num_layers": policy.num_layers,
            "total": parts["memory_gru"],
        },
        "action_heads": dict(heads),
        "joint_count": JOINT_COUNT,
        "parts": parts,
        "total": total,
        "total_millions": round(total / 1e6, 6),
        "within_parameter_range": bool(
            MODEL_SPEC["parameter_range_millions"][0]
            <= total / 1e6
            <= MODEL_SPEC["parameter_range_millions"][1]),
        "normalization_layers": "GroupNorm + LayerNorm only",
        "batch_norm_modules": batch_norm,
        "has_mutable_batch_statistics": bool(batch_norm),
        "activation": MODEL_SPEC["activation"],
        "conv_padding": MODEL_SPEC["encoder_candidate"]["conv_padding"],
        "measurement": "sum(p.numel() for p in policy.parameters())",
    }
    return report


# ---------------------------------------------------------------------------
# pure functions on head logits (module level; the methods delegate here)
# ---------------------------------------------------------------------------
def _check_logits(head_logits: Any) -> HeadLogits:
    if not isinstance(head_logits, HeadLogits):
        raise ModelError(
            f"expected HeadLogits (the four raw head logits), got "
            f"{type(head_logits).__name__}")
    return head_logits


def _action_indices(head_logits: HeadLogits, actions: Any
                    ) -> dict[str, torch.Tensor]:
    """Normalise ``actions`` to one ``long`` index per head (no clamping)."""

    batched = head_logits.batched
    batch_size = head_logits.batch_size
    per_head: dict[str, list[Any]] = {head: [] for head in HEAD_ORDER}

    def take(row: Sequence[Any]) -> None:
        items = list(row)
        if len(items) != len(HEAD_ORDER):
            raise ModelError(
                f"a joint action needs {len(HEAD_ORDER)} head values "
                f"{HEAD_ORDER}, got {len(items)}: {items!r}")
        for head, value in zip(HEAD_ORDER, items):
            per_head[head].append(value)

    if isinstance(actions, contracts.JointAction):
        if batched and batch_size != 1:
            raise ModelError(
                f"a single JointAction cannot describe a batch of {batch_size}; "
                f"pass one action per row")
        take(actions.as_tuple())
    elif isinstance(actions, Mapping):
        unknown = [key for key in actions if key not in HEAD_ORDER]
        if unknown:
            raise ModelError(
                f"actions mapping has unknown head(s) {unknown}; known {HEAD_ORDER}")
        missing = [head for head in HEAD_ORDER if head not in actions]
        if missing:
            raise ModelError(
                f"actions mapping is missing head(s) {missing}; all four heads are "
                f"required (a missing head cannot default to 'always shoot')")
        take([actions[head] for head in HEAD_ORDER])
    elif isinstance(actions, torch.Tensor):
        tensor = actions
        if tensor.ndim == 1:
            if tensor.shape[0] != len(HEAD_ORDER):
                raise ModelError(
                    f"action tensor has {tensor.shape[0]} values, expected "
                    f"{len(HEAD_ORDER)}")
            take(tensor.tolist())
        elif tensor.ndim == 2:
            if int(tensor.shape[1]) != len(HEAD_ORDER):
                raise ModelError(
                    f"action tensor has {int(tensor.shape[1])} columns, expected "
                    f"{len(HEAD_ORDER)}")
            if not batched and int(tensor.shape[0]) != 1:
                raise ModelError(
                    f"{int(tensor.shape[0])} action rows for an unbatched "
                    f"observation")
            if batched and int(tensor.shape[0]) != batch_size:
                raise ModelError(
                    f"{int(tensor.shape[0])} action rows for a batch of "
                    f"{batch_size}")
            rows = tensor.tolist()
            for row in rows:
                take(row)
        else:
            raise ModelError(
                f"action tensor must be (4,) or (B,4), got {tuple(tensor.shape)}")
    elif isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
        items = list(actions)
        if batched and items and all(
                isinstance(item, (Sequence, torch.Tensor))
                and not isinstance(item, (str, bytes)) for item in items):
            if len(items) != batch_size:
                raise ModelError(
                    f"{len(items)} action rows for a batch of {batch_size}")
            for row in items:
                take(list(row))
        else:
            take(items)
    else:
        raise ModelError(
            f"actions must be a JointAction, a 4-sequence, a (4,)/(B,4) tensor or "
            f"a mapping of the four heads, got {type(actions).__name__}")

    indices: dict[str, torch.Tensor] = {}
    device = head_logits.move.device
    for head in HEAD_ORDER:
        values = per_head[head]
        if batched and len(values) != batch_size:
            raise ModelError(
                f"head {head!r} received {len(values)} index/indices for a batch of "
                f"{batch_size}")
        if not batched and len(values) != 1:
            raise ModelError(
                f"head {head!r} received {len(values)} index/indices for a single "
                f"observation")
        try:
            index = torch.as_tensor(
                [int(value) if not isinstance(value, bool) else int(value)
                 for value in values],
                dtype=torch.long, device=device)
        except (TypeError, ValueError) as exc:
            raise ModelError(
                f"head {head!r} action indices {values!r} are not integers: {exc}"
            ) from exc
        upper = HEAD_SIZES[head]
        if bool((index < 0).any()) or bool((index >= upper).any()):
            raise ModelError(
                f"head {head!r} action index {values!r} outside [0, {upper})")
        indices[head] = index if batched else index.reshape(())
    return indices


def _head_log_prob(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    scores = torch.log_softmax(logits, dim=-1)
    if scores.ndim == 1:
        return scores[index]
    return scores.gather(-1, index.reshape(-1, 1)).squeeze(-1)


def _head_entropy(logits: torch.Tensor) -> torch.Tensor:
    scores = torch.log_softmax(logits, dim=-1)
    return -(scores.exp() * scores).sum(dim=-1)


def _sum_head_log_probs(logits: HeadLogits,
                        indices: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Sum one already-validated index per head (the shared arithmetic)."""

    total: torch.Tensor | None = None
    for head in HEAD_ORDER:
        part = _head_log_prob(logits[head], indices[head])
        if bool((part > 1e-6).any()):  # pragma: no cover - a broken gather
            raise ModelError(
                f"head {head!r} produced log-probability {float(part.max())} > 0; a "
                f"probability above 1 means the logits/gather are wrong")
        total = part if total is None else total + part
    assert total is not None
    return total


def _joint_log_prob(head_logits: Any, actions: Any) -> torch.Tensor:
    """The joint log-probability of the four independent heads: the **sum**.

    Not a product, not an average.  Every head must be given: there is no
    parameter that can stand in for one, so ``shoot`` cannot silently become
    "always 1" (that substitution is the v4.2.0 defect in probability space).
    """

    logits = _check_logits(head_logits)
    return _sum_head_log_probs(logits, _action_indices(logits, actions))


def _joint_entropy(head_logits: Any) -> torch.Tensor:
    """The joint entropy of the four independent heads: the **sum**."""

    logits = _check_logits(head_logits)
    total: torch.Tensor | None = None
    for head in HEAD_ORDER:
        part = _head_entropy(logits[head])
        if bool((part < -1e-6).any()):  # pragma: no cover - a broken softmax
            raise ModelError(
                f"head {head!r} produced entropy {float(part.min())} < 0")
        total = part if total is None else total + part
    assert total is not None
    return total


def _sample(head_logits: Any, *, generator: torch.Generator | None = None,
            temperature: float = 1.0) -> tuple[Any, torch.Tensor]:
    """Draw one index per head from the four independent categorical heads.

    ``temperature`` divides the logits before the softmax, and the returned
    log-probability is the one of the distribution actually sampled from (not of
    the unscaled logits), so a rollout cannot log a probability it did not use.
    """

    logits = _check_logits(head_logits)
    _assert_no_forced_shoot()
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ModelError(f"temperature must be a real number, got {temperature!r}")
    scale = float(temperature)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ModelError(f"temperature must be finite and > 0, got {scale!r}")
    if isinstance(generator, torch.Tensor):
        raise ModelError(
            "generator must be a torch.Generator; a tensor is not a random source")
    scaled = {head: (value if scale == 1.0 else value / scale)
              for head, value in logits.as_dict().items()}
    scaled_logits = HeadLogits(**scaled)
    indices: dict[str, torch.Tensor] = {}
    for head in HEAD_ORDER:
        scores = scaled_logits[head]
        probabilities = torch.softmax(scores, dim=-1)
        drawn = torch.multinomial(probabilities.reshape(-1, scores.shape[-1]), 1,
                                  generator=generator)
        drawn = drawn.reshape(scores.shape[:-1])
        indices[head] = drawn
    log_prob = _sum_head_log_probs(scaled_logits, indices)
    if not logits.batched:
        action = contracts.JointAction(
            *[int(indices[head].reshape(-1)[0]) for head in HEAD_ORDER])
        return action, log_prob
    rows = [contracts.JointAction(*[int(indices[head][index])
                                    for head in HEAD_ORDER])
            for index in range(logits.batch_size)]
    return tuple(rows), log_prob


#: Public module-level API (the class methods delegate to the same code paths).
def joint_log_prob(head_logits: Any, actions: Any) -> torch.Tensor:
    """Sum of the four heads' log-probabilities at ``actions`` (see internals)."""
    return _joint_log_prob(head_logits, actions)


def joint_entropy(head_logits: Any) -> torch.Tensor:
    """Sum of the four heads' entropies (see internals)."""
    return _joint_entropy(head_logits)


def sample(head_logits: Any, *, generator: torch.Generator | None = None,
           temperature: float = 1.0) -> tuple[Any, torch.Tensor]:
    """Sample one joint action; same generator seed -> same action."""
    return _sample(head_logits, generator=generator, temperature=temperature)


def build_actor(**unexpected: Any) -> ActorPolicy:
    """Build the guide §4.3 actor.  It takes no architecture arguments."""

    if unexpected:
        raise PrivilegedModelInputError(
            f"build_actor() takes no arguments, got {sorted(unexpected)}; every "
            f"number comes from MODEL_SPEC (editing it re-runs the consistency "
            f"checks)")
    return ActorPolicy()
