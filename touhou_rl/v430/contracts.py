"""The ``v430-contract/1`` data contract (v4.3.0 work package W01).

One training sample, one deployment decision and one evaluation replay must
describe the *same* instant, the *same* action and the *same* duration.  This
module is the single place that decides whether a record satisfies that, and it
is deliberately:

* **pure** — every function is a total function of its arguments.  Nothing here
  opens a file, starts a process, reads an env var, imports an engine or mutates
  global state.  The only I/O in the whole module is :func:`o2_digest`'s hashing
  of an in-memory dict, which is a pure function of that dict;
* **stdlib only** — no numpy, no torch, no ``touhou_rl.v400`` import.  The
  contract is testable anywhere, including a machine with no game assets.  The
  button/direction *tables* are not duplicated either: they are reused from
  :mod:`touhou_rl.v400.gym.actions` / :mod:`touhou_rl.spec` by
  :mod:`touhou_rl.v430.actions`, which is the only module that builds a mask;
* **fail-closed** — every checker raises :class:`ContractError` instead of
  returning a sentinel that a caller could mistake for "passed".  There is no
  ``True``/``None`` fall-through: a function either returns the validated value
  or raises.

Sections:

1. joint action (:data:`JOINT_COUNT` = 72, encode/decode are exact inverses);
2. joint probability semantics (four independent heads, summed log-prob and
   entropy — **no** implicit "always shoot" rule anywhere);
3. :class:`ActorObservation` shapes/dtypes/domains and the privileged-field
   sentinel;
4. O2 encoding parameters and :func:`o2_digest` (parameters only: the real pixel
   encoder is W02 and is *not* implemented here);
5. the time/step contract (``hold_game_frames = 2``, two independent duration
   sources, an explicit ``invalid`` source);
6. transition fields (``terminated`` / ``truncated`` / ``segment_end`` are three
   different fields) and the bootstrap rule;
7. result JSON / exit-code semantics with ``capability_gate`` defaulting to
   ``not_run``.

Constants are mirrored in ``configs/v430/contracts.json``;
``tests/test_v430_contracts.py`` asserts the two agree, so neither side can drift
silently.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = [
    # contract identity
    "CONTRACT_VERSION",
    "SPEC_REVISION",
    "contract_constants",
    "action_parameters",
    "o2_parameters",
    "o2_digest",
    "time_parameters",
    "result_parameters",
    # 1. joint action
    "HEAD_ORDER",
    "HEAD_SIZES",
    "JOINT_COUNT",
    "N_MOVE",
    "BINARY_HEADS",
    "JointAction",
    "encode_joint",
    "decode_joint",
    "iter_joint_actions",
    "check_joint_action",
    "joint_inverse",
    "forced_shoot",
    "truth_action_mask",
    "check_no_forced_shoot_rule",
    # 2. joint probability
    "check_head_logprob",
    "check_head_entropy",
    "check_joint_logprob",
    "check_joint_entropy",
    "joint_logprob_of",
    "joint_entropy_of",
    # 3. ActorObservation
    "ACTOR_OBSERVATION_FIELDS",
    "ACTOR_OBSERVATION_KEYS",
    "KEY_ORDER",
    "FIELD_SPECS",
    "FORBIDDEN_ACTOR_FIELDS",
    "PRIVILEGED_FIELD_NAMES",
    "FieldSpec",
    "ActorObservation",
    "check_actor_observation",
    "check_history_valid_prefix",
    "check_executed_key_history",
    "find_privileged_fields",
    "assert_not_privileged",
    "check_no_privileged_fields",
    "schema_digest",
    # 4. O2 parameters
    "SOURCE_RGB_WH",
    "O2_PLAY_CROP_XYWH",
    "O2_HUD_CROP_XYWH",
    "O2_GRAY_WEIGHTS",
    "O2_GRAY_SHIFT",
    "O2_GRAY_RULE",
    "O2_DOWNSAMPLE_FACTOR",
    "O2_DOWNSAMPLE_RULE",
    "O2_STORAGE_DTYPE",
    "O2_NORMALIZATION",
    "O2_NORMALIZE_DIVISOR",
    "O2_IMAGE_BYTES_PER_DECISION",
    "RGB_CHANNEL_ORDER",
    "RGBA_INPUT_ORDER",
    "O2_PLAY_TEMPORAL_SPACING",
    # 5. time / step
    "HOLD_GAME_FRAMES",
    "REFERENCE_HZ",
    "DURATION_SOURCES",
    "DURATION_SOURCE_ENGINE",
    "DURATION_SOURCE_EXTERNAL",
    "DURATION_SOURCE_INVALID",
    "DURATION_TOLERANCE_FRAMES",
    "Duration",
    "check_duration",
    "effective_frames_from_capture",
    "assert_normal_two_frame",
    # 6. transition
    "TRANSITION_REQUIRED_FIELDS",
    "TRANSITION_OPTIONAL_FIELDS",
    "TRANSITION_FIELDS",
    "TRANSITION_END_FLAGS",
    "TRANSITION_ALIAS_KEYS",
    "TRUNCATION_REASONS",
    "CONTROLLER_SOURCES",
    "SESSION_ACTOR_LOSS_MASK",
    "ACTOR_LOSS_MASK_ON",
    "check_transition",
    "check_bootstrap",
    "check_end_flags",
    # 7. result / exit codes
    "EXIT_OK",
    "EXIT_INVALID",
    "EXIT_GATE_FAIL",
    "EXIT_INCOMPLETE",
    "EXIT_CODES",
    "EXIT_CODE_BY_EXECUTION_STATUS",
    "RESULT_FIELDS",
    "EXECUTION_STATUSES",
    "PROTOCOL_VALIDITY_VALUES",
    "RESULT_STATUSES",
    "CAPABILITY_GATE_VALUES",
    "CAPABILITY_GATE_DEFAULT",
    "TRAINING_COMMAND_CLASSES",
    "ContractError",
    "new_result",
    "assert_result_schema",
    "set_capability_gate",
    "exit_code_for",
]


class ContractError(RuntimeError):
    """A record, an observation, an action or a result violates the contract.

    Raised by *every* failing check in this module.  There is no sentinel return
    path, so a caller cannot accidentally read "did not raise" as "valid" when it
    forgot to inspect a return value.
    """


# ---------------------------------------------------------------------------
# 0. contract identity
# ---------------------------------------------------------------------------
#: Version of the data contract.  A change to any constant below must change it.
CONTRACT_VERSION = "v430-contract/1"
#: The plan revision (``PLAN_SPEC.json``) this contract implements.
SPEC_REVISION = "v430-plan-r2"
#: Schema tag of ``configs/v430/contracts.json``.
CONFIG_SCHEMA = "v430-contracts/1"


# ---------------------------------------------------------------------------
# 1. joint action: 9 x 2 x 2 x 2 = 72 combinations
# ---------------------------------------------------------------------------
#: Head order of the joint id, exactly as ``PLAN_SPEC.action.heads`` lists it.
HEAD_ORDER: tuple[str, ...] = ("move", "focus", "shoot", "bomb")
#: Size of each head: 9 directions, three binary gates.
HEAD_SIZES: dict[str, int] = {"move": 9, "focus": 2, "shoot": 2, "bomb": 2}
N_MOVE = HEAD_SIZES["move"]
BINARY_HEADS: tuple[str, ...] = ("focus", "shoot", "bomb")
#: ``9 * 2 * 2 * 2``: every combination has its own final mask.
JOINT_COUNT = N_MOVE * 2 * 2 * 2

if JOINT_COUNT != 72:  # pragma: no cover - frozen arithmetic
    raise RuntimeError(f"JOINT_COUNT is {JOINT_COUNT}, the plan pins 72")


def _require_int(value: Any, name: str) -> int:
    """``value`` must be a real ``int``; ``bool`` is refused on purpose.

    ``True``/``False`` are ``int`` subclasses in Python, so accepting them here
    would let a comparison result (or a numpy/torch byte) silently masquerade as
    an action index.  A contract that cannot tell 1 from ``True`` cannot tell an
    action from a predicate.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an int, got {value!r}")
    return value


def _require_binary(value: Any, name: str) -> int:
    number = _require_int(value, name)
    if number not in (0, 1):
        raise ContractError(f"{name} must be 0 or 1, got {number}")
    return number


@dataclasses.dataclass(frozen=True)
class JointAction:
    """One of the 72 requested actions, as the actor samples it.

    The four heads are sampled **independently**; this record only carries the
    joint index of the combination that was drawn.  ``shoot`` is a real head:
    ``shoot=0`` and ``shoot=1`` are different actions with different masks.
    """

    move: int
    focus: int
    shoot: int
    bomb: int

    def __post_init__(self) -> None:
        encode_joint(self.move, self.focus, self.shoot, self.bomb)

    @property
    def joint(self) -> int:
        """The joint index in ``[0, 72)``."""

        return encode_joint(self.move, self.focus, self.shoot, self.bomb)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.move, self.focus, self.shoot, self.bomb)

    def as_dict(self) -> dict[str, int]:
        return {"move": self.move, "focus": self.focus,
                "shoot": self.shoot, "bomb": self.bomb}


def encode_joint(move: Any, focus: Any, shoot: Any, bomb: Any) -> int:
    """``(move, focus, shoot, bomb)`` -> joint id in ``[0, 72)``.

    ``joint = ((move * 2 + focus) * 2 + shoot) * 2 + bomb``, i.e. the head order
    of :data:`HEAD_ORDER`.  This is a **new** encoding for v4.3.0 and is
    deliberately *not* ``touhou_rl.spec.encode_flat`` (which orders the bits
    ``move, shoot, focus, bomb``); mixing the two would relabel every focused
    shot as a focused bomb.  The inverse is :func:`decode_joint`.
    """

    move = _require_int(move, "move")
    if not 0 <= move < N_MOVE:
        raise ContractError(f"move {move} outside [0, {N_MOVE})")
    focus = _require_binary(focus, "focus")
    shoot = _require_binary(shoot, "shoot")
    bomb = _require_binary(bomb, "bomb")
    return ((move * 2 + focus) * 2 + shoot) * 2 + bomb


def decode_joint(joint: Any) -> tuple[int, int, int, int]:
    """Inverse of :func:`encode_joint`; refuses anything outside ``[0, 72)``."""

    joint = _require_int(joint, "joint")
    if not 0 <= joint < JOINT_COUNT:
        raise ContractError(f"joint {joint} outside [0, {JOINT_COUNT})")
    move, rest = divmod(joint, 8)
    focus, rest = divmod(rest, 4)
    shoot, bomb = divmod(rest, 2)
    return move, focus, shoot, bomb


def joint_of(action: JointAction | Sequence[int]) -> int:
    """Accept a :class:`JointAction` or a 4-sequence and return its joint id."""

    if isinstance(action, JointAction):
        return action.joint
    items = _as_tuple(action, what="action")
    if len(items) != len(HEAD_ORDER):
        raise ContractError(
            f"a joint action needs {len(HEAD_ORDER)} heads {HEAD_ORDER}, got "
            f"{len(items)} values")
    return encode_joint(*items)


def iter_joint_actions() -> Iterator[JointAction]:
    """Yield all :data:`JOINT_COUNT` actions in joint-id order."""

    for joint in range(JOINT_COUNT):
        yield JointAction(*decode_joint(joint))


def check_joint_action(action: Any) -> JointAction:
    """Validate a requested action and return it as a :class:`JointAction`."""

    if isinstance(action, JointAction):
        return JointAction(*decode_joint(action.joint))
    return JointAction(*decode_joint(joint_of(action)))


def joint_inverse(rounds: int = JOINT_COUNT) -> bool:
    """``decode(encode(a)) == a`` and ``encode(decode(j)) == j`` for every value.

    A *self-check* the tests use, not a gate: it recomputes the two directions
    over all 72 combinations and returns whether they are exact inverses and
    produce exactly 72 distinct ids.
    """

    rounds = _require_int(rounds, "rounds")
    if rounds != JOINT_COUNT:
        raise ContractError(f"rounds {rounds} != JOINT_COUNT {JOINT_COUNT}")
    seen: set[int] = set()
    for action in iter_joint_actions():
        joint = action.joint
        if decode_joint(joint) != action.as_tuple():
            return False
        if joint in seen:
            return False
        seen.add(joint)
    return len(seen) == JOINT_COUNT and min(seen) == 0 and max(seen) == JOINT_COUNT - 1


#: ``PLAN_SPEC.action.forced_shoot`` — shooting is a sampled head, never forced.
FORCED_SHOOT = False
#: ``PLAN_SPEC.action.truth_action_mask`` — actions are never filtered by truth.
TRUTH_ACTION_MASK = False


def forced_shoot() -> bool:
    """``False``: the compiler must never OR the shoot bit into a request."""

    return FORCED_SHOOT


def truth_action_mask() -> bool:
    """``False``: no truth-derived action masking is part of the contract."""

    return TRUTH_ACTION_MASK


def _as_tuple(values: Any, *, what: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ContractError(f"{what} must be a sequence, got {values!r}")
    return tuple(values)


def check_no_forced_shoot_rule(masks: Sequence[int], *,
                               shoot_requests: Sequence[int]) -> tuple[int, ...]:
    """Refuse a mask list whose SHOOT bit disagrees with the ``shoot`` head.

    ``mask & shoot_bit`` must equal ``shoot_request`` frame by frame.  This is
    the fail-closed inverse of the v4.2.0 defect where ``cand_to_mask`` forced
    the shoot bit on and ``shoot=0`` became indistinguishable from ``shoot=1``.
    The bit itself is passed in so this contract module never needs to import
    the engine's button table.
    """

    mask_items = _as_tuple(masks, what="masks")
    shoot_items = _as_tuple(shoot_requests, what="shoot_requests")
    if len(mask_items) != len(shoot_items):
        raise ContractError(
            f"{len(mask_items)} masks vs {len(shoot_items)} shoot requests")
    if not shoot_items:
        raise ContractError("no frames to check")
    shoot_bit = 1 << shoot_bit_index()
    for index, (mask, shoot) in enumerate(zip(mask_items, shoot_items)):
        mask = _require_int(mask, f"masks[{index}]")
        shoot = _require_binary(shoot, f"shoot_requests[{index}]")
        actual = 1 if mask & shoot_bit else 0
        if actual != shoot:
            raise ContractError(
                f"frame {index}: requested shoot={shoot} but mask 0x{mask:04x} has "
                f"shoot bit {actual}; the shoot head must not be OR-ed on")
    return mask_items


def shoot_bit_index() -> int:
    """Bit index of SHOOT in the engine mask (``TH08`` Global.hpp: bit 0).

    Kept as a number, not a table: :mod:`touhou_rl.spec` owns the authoritative
    ``BTN_SHOOT`` value and :mod:`touhou_rl.v430.actions` asserts the two agree
    at import time.
    """

    return 0


# ---------------------------------------------------------------------------
# 2. joint probability semantics: four independent categorical heads
# ---------------------------------------------------------------------------
def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        # A NaN or +-inf head value poisons the sum *and* the PPO ratio
        # ``exp(logp_new - logp_old)``.  Masking an unreachable head is done by
        # the sampler contract, never by writing -inf into the record.
        raise ContractError(f"{name} must be finite, got {number!r}")
    return number


def check_head_logprob(value: Any, head: str = "head") -> float:
    """A per-head log-probability: finite and ``<= 0``."""

    number = _require_finite(value, f"{head}_logprob")
    if number > 0.0:
        raise ContractError(f"{head}_logprob {number} > 0; a probability cannot exceed 1")
    return number


def check_head_entropy(value: Any, head: str = "head") -> float:
    """A per-head entropy: finite and ``>= 0``."""

    number = _require_finite(value, f"{head}_entropy")
    if number < 0.0:
        raise ContractError(f"{head}_entropy {number} < 0")
    return number


def check_joint_logprob(move_lp: Any, focus_lp: Any, shoot_lp: Any, bomb_lp: Any) -> float:
    """Joint log-probability of the four independent heads: the **sum**.

    ``log_prob(move, focus, shoot, bomb) = sum(head_logprob)``.  All four heads
    are mandatory: there is no parameter that could stand in for a head, so no
    caller can hand in "shoot always happens" and get a well-formed number back.
    ``shoot_lp=None`` raises rather than silently substituting 0.0 (which would
    mean "shoot is certain") — that substitution is exactly the v4.2.0 defect in
    probability space.
    """

    parts = (("move", move_lp), ("focus", focus_lp),
             ("shoot", shoot_lp), ("bomb", bomb_lp))
    total = 0.0
    for head, value in parts:
        total += check_head_logprob(value, head)
    return total


def check_joint_entropy(move_h: Any, focus_h: Any, shoot_h: Any, bomb_h: Any) -> float:
    """Joint entropy of the four independent heads: the **sum**."""

    parts = (("move", move_h), ("focus", focus_h),
             ("shoot", shoot_h), ("bomb", bomb_h))
    total = 0.0
    for head, value in parts:
        total += check_head_entropy(value, head)
    return total


def _head_mapping(values: Mapping[str, Any], what: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ContractError(f"{what} must be a mapping, got {type(values).__name__}")
    missing = [head for head in HEAD_ORDER if head not in values]
    if missing:
        raise ContractError(f"{what} is missing heads {missing}; all four are required")
    extra = [key for key in values if key not in HEAD_ORDER]
    if extra:
        raise ContractError(f"{what} has unregistered heads {extra}; known {HEAD_ORDER}")
    return {head: float(values[head]) for head in HEAD_ORDER}


def joint_logprob_of(head_logprobs: Mapping[str, Any]) -> float:
    """:func:`check_joint_logprob` over a ``{head: value}`` mapping."""

    heads = _head_mapping(head_logprobs, "head_logprobs")
    return check_joint_logprob(*(heads[head] for head in HEAD_ORDER))


def joint_entropy_of(head_entropies: Mapping[str, Any]) -> float:
    """:func:`check_joint_entropy` over a ``{head: value}`` mapping."""

    heads = _head_mapping(head_entropies, "head_entropies")
    return check_joint_entropy(*(heads[head] for head in HEAD_ORDER))


# ---------------------------------------------------------------------------
# 3. ActorObservation
# ---------------------------------------------------------------------------
#: ``executed_key_history`` column order: left/right/up/down/focus/shoot/bomb.
KEY_ORDER: tuple[str, ...] = ("left", "right", "up", "down", "focus", "shoot", "bomb")
#: Fields the actor may see, in the order of ``PLAN_SPEC.observation.O2``.
ACTOR_OBSERVATION_KEYS: tuple[str, ...] = (
    "global_rgb", "play_gray", "hud_gray", "executed_key_history",
    "history_valid", "elapsed_capture_s", "memory",
)

#: Names that must never reach the actor.  They are engine/teacher/judge truth:
#: the actor only sees pixels and its own executed keys.
FORBIDDEN_ACTOR_FIELDS: tuple[str, ...] = (
    "engine_frame", "seed", "script_id", "stage_truth", "enemy_table",
    "resource_truth", "teacher_action", "future_tape",
)
#: Further names the actor-facing surface refuses.  These are not in the
#: ``PLAN_SPEC`` list, but they are the same class of leak (privileged state that
#: a hand-built dict could smuggle in), so the sentinel covers them too and the
#: bidirectional config/constant test keeps :data:`FORBIDDEN_ACTOR_FIELDS`
#: exactly the plan's list.
_PRIVILEGED_EXTRA_NAMES: tuple[str, ...] = (
    "privileged_ref", "privileged_state", "oracle_state", "lives_truth",
    "bomb_count_truth", "power_truth", "rng_state", "replay_input_tape",
)
PRIVILEGED_FIELD_NAMES: tuple[str, ...] = FORBIDDEN_ACTOR_FIELDS + _PRIVILEGED_EXTRA_NAMES

#: Max recursion depth of the privileged-name scan (a nested audit dict may
#: legitimately hold one of these names one level down; deeper than this is
#: treated as "cannot be audited" only when it is not a plain container).
_PRIVILEGED_SCAN_DEPTH = 6


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    """Shape/dtype/domain of one actor observation field.

    ``value_domain`` is ``None`` when the dtype bounds are the whole domain
    (``uint8`` 0..255, ``float32`` finite) and a concrete tuple when the legal
    values are stricter than the dtype (key history is 0/1 inside a ``uint8``
    tensor, ``history_valid`` is a boolean flag).
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    minimum: float | None
    maximum: float | None
    value_domain: tuple[Any, ...] | None
    semantics: str


ACTOR_OBSERVATION_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("global_rgb", (3, 240, 320), "uint8", 0, 255, None,
              "the current full framed region, RGB, including menus and HUD"),
    FieldSpec("play_gray", (4, 224, 192), "uint8", 0, 255, None,
              "current and three previous *genuine consecutive* play frames"),
    FieldSpec("hud_gray", (1, 448, 160), "uint8", 0, 255, None,
              "TH08 native-pixel HUD strip; other games need their own check"),
    FieldSpec("executed_key_history", (8, 7), "uint8", 0, 1, (0, 1),
              "oldest to newest actual executed key state, columns KEY_ORDER"),
    FieldSpec("history_valid", (8,), "bool", None, None, (False, True),
              "cold start / missing history must not be faked as executed idle"),
    FieldSpec("elapsed_capture_s", (), "float32", 0.0, None, None,
              "the capture clock's own interval, not an engine frame number"),
    FieldSpec("memory", (256,), "float32", None, None, None,
              "model-internal recurrent state; a teacher must never fill it"),
)
FIELD_SPECS: dict[str, FieldSpec] = {spec.name: spec for spec in ACTOR_OBSERVATION_FIELDS}

if tuple(FIELD_SPECS) != ACTOR_OBSERVATION_KEYS:  # pragma: no cover - frozen table
    raise RuntimeError(
        f"FIELD_SPECS keys {tuple(FIELD_SPECS)} != ACTOR_OBSERVATION_KEYS "
        f"{ACTOR_OBSERVATION_KEYS}")

#: Leaves this module is willing to materialise for a value scan.  Smaller
#: fields are scanned element-wise even when they are typed arrays (the key
#: history is only 56 bytes); the three image fields are checked by dtype alone,
#: because ``uint8`` *is* the 0..255 guarantee and scanning 474 kB per check in
#: pure Python would be a hidden cost, not a guarantee.
_SCAN_LEAF_LIMIT = 4096


def _normalise_dtype(value: Any, name: str) -> str:
    """``torch.uint8`` / ``np.uint8`` / ``uint8`` -> ``uint8``.

    Accepts a duck-typed ``.dtype`` (numpy, torch and every array library expose
    one) and never imports the library that produced it.
    """

    text = str(value).strip().lower()
    for prefix in ("torch.", "numpy.", "np."):
        if text.startswith(prefix):
            text = text[len(prefix):]
    aliases = {"uint8": "uint8", "bool": "bool", "bool_": "bool",
               "float32": "float32", "float": "float32",
               "float_": "float32"}
    if text not in aliases:
        raise ContractError(
            f"{name}: dtype {value!r} is not one of uint8/bool/float32")
    return aliases[text]


def _shape_of(value: Any, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            dims = tuple(shape)
        except TypeError as exc:
            raise ContractError(f"{name}: shape {shape!r} is not iterable") from exc
        out = []
        for dim in dims:
            dim = _require_int(dim, f"{name}.shape entry")
            if dim < 0:
                raise ContractError(f"{name}: negative shape entry {dim}")
            out.append(dim)
        return tuple(out)
    if isinstance(value, (list, tuple)):
        return _scan_nested(value, name)[0]
    if isinstance(value, (bool, int, float)):
        return ()
    raise ContractError(
        f"{name}: value of type {type(value).__name__} exposes no shape; pass an "
        f"array with .shape or a nested list")


def _scan_nested(value: Any, name: str, depth: int = 0
                 ) -> tuple[tuple[int, ...], str | None, float | None, float | None]:
    """One pass over a nested list: shape, leaf kind, min, max (ragged refused)."""

    if depth > 8:
        raise ContractError(f"{name}: nested deeper than 8 levels")
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,), None, None, None
        sub_shape: tuple[int, ...] | None = None
        kind: str | None = None
        low: float | None = None
        high: float | None = None
        for item in value:
            item_shape, item_kind, item_low, item_high = _scan_nested(item, name, depth + 1)
            if sub_shape is None:
                sub_shape = item_shape
            elif item_shape != sub_shape:
                raise ContractError(
                    f"{name}: ragged nested sequence ({sub_shape} vs {item_shape})")
            if item_kind is not None:
                kind = item_kind if kind is None or kind == item_kind else "mixed"
            if item_low is not None:
                low = item_low if low is None else min(low, item_low)
            if item_high is not None:
                high = item_high if high is None else max(high, item_high)
        return (len(value),) + (sub_shape or ()), kind, low, high
    if isinstance(value, bool):
        return (), "bool", float(value), float(value)
    if isinstance(value, int):
        return (), "int", float(value), float(value)
    if isinstance(value, float):
        return (), "float", float(value), float(value)
    dtype_attr = getattr(value, "dtype", None)
    if dtype_attr is not None:
        # A numpy/torch scalar inside a plain list: trust its dtype, not its
        # Python type (``np.bool_`` is not a ``bool`` subclass).
        normal = _normalise_dtype(dtype_attr, name)
        kind = {"bool": "bool", "uint8": "int", "float32": "float"}[normal]
        number = float(bool(value) if normal == "bool" else float(value))
        return (), kind, number, number
    raise ContractError(f"{name}: leaf {value!r} is not a number or bool")


def _coerce_leaf(leaf: Any, dtype: str, name: str, index: int) -> Any:
    """Return ``leaf`` as a Python scalar of ``dtype``, or raise.

    Array scalars (``np.uint8``, ``torch.float32``, ``np.bool_``) are accepted by
    their declared dtype and coerced, because none of them is an instance of the
    Python ``int``/``float``/``bool`` it represents.  Nothing is imported: the
    dtype string is all that is read.
    """

    if dtype == "bool":
        if isinstance(leaf, bool):
            return leaf
        dtype_attr = getattr(leaf, "dtype", None)
        if dtype_attr is not None and _normalise_dtype(dtype_attr, name) == "bool":
            return bool(leaf)
        raise ContractError(
            f"{name}: leaf {index} is {leaf!r}; a bool field cannot store "
            f"{type(leaf).__name__} (0/1 ints are refused, they are not flags)")
    if dtype == "uint8":
        if not isinstance(leaf, bool) and isinstance(leaf, int):
            return int(leaf)
        dtype_attr = getattr(leaf, "dtype", None)
        if dtype_attr is not None and _normalise_dtype(dtype_attr, name) == "uint8":
            return int(leaf)
        raise ContractError(f"{name}: leaf {index} is {leaf!r}, not a uint8")
    if not isinstance(leaf, bool) and isinstance(leaf, (int, float)):
        return float(leaf)
    dtype_attr = getattr(leaf, "dtype", None)
    if dtype_attr is not None and _normalise_dtype(dtype_attr, name) == "float32":
        return float(leaf)
    raise ContractError(f"{name}: leaf {index} is {leaf!r}, not a number")


def _iter_leaves(value: Any, name: str) -> Iterator[Any]:
    """Iterate leaves of a nested sequence or of a typed array (no numpy import)."""

    shape = getattr(value, "shape", None)
    if shape is not None and not isinstance(value, (list, tuple)):
        try:
            if tuple(shape) == ():
                # A 0-d array scalar is a leaf, not an iterable.
                yield value
                return
        except TypeError:
            pass
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (list, tuple)):
                yield from _iter_leaves(item, name)
            else:
                yield item
        return
    if isinstance(value, (bool, int, float)):
        yield value
        return
    try:
        for item in value:
            yield from _iter_leaves(item, name)
    except TypeError as exc:
        raise ContractError(
            f"{name}: value of type {type(value).__name__} is neither a sequence "
            f"nor iterable; its values cannot be checked") from exc


def _element_count(spec: FieldSpec) -> int:
    count = 1
    for dim in spec.shape:
        count *= dim
    return count


def _check_array(name: str, spec: FieldSpec, value: Any) -> tuple[tuple[int, ...], str]:
    """Validate shape, dtype and (when affordable) value domain of one field."""

    shape = _shape_of(value, name)
    if shape != spec.shape:
        raise ContractError(
            f"{name}: shape {shape} != required {spec.shape}; resizing or "
            f"interpolation into another shape is refused, not adapted")
    dtype_attr = getattr(value, "dtype", None)
    if dtype_attr is not None:
        dtype = _normalise_dtype(dtype_attr, name)
    else:
        _shape, kind, _low, _high = _scan_nested(value, name)
        if kind is None:
            raise ContractError(f"{name}: empty sequence, dtype cannot be inferred")
        if kind == "mixed":
            raise ContractError(
                f"{name}: mixed leaf types; a dtype cannot be inferred honestly")
        dtype = {"bool": "bool", "int": "uint8", "float": "float32"}[kind]
        if kind == "int":
            # Inference is only honest when the values really fit the dtype.
            _lo, _hi = _domain_bounds(spec, name)
            leaves = _iter_leaves(value, name)
            for index, leaf in enumerate(leaves):
                number = _require_leaf_int(leaf, name, index)
                if not _lo <= number <= _hi:
                    raise ContractError(
                        f"{name}: value {number} at leaf {index} outside [{_lo}, "
                        f"{_hi}]; it does not fit uint8")
    if dtype != spec.dtype:
        raise ContractError(
            f"{name}: dtype {dtype} != required {spec.dtype}" +
            ("; a Python list of ints infers uint8 — pass floats or a typed array "
             "for a float32 field" if spec.dtype == "float32" else ""))
    count = _element_count(spec)
    if count <= _SCAN_LEAF_LIMIT:
        for index, leaf in enumerate(_iter_leaves(value, name)):
            _check_leaf(spec, name, index, leaf)
    return shape, dtype


def _domain_bounds(spec: FieldSpec, name: str) -> tuple[float, float]:
    low = spec.minimum if spec.minimum is not None else -math.inf
    high = spec.maximum if spec.maximum is not None else math.inf
    if spec.value_domain is not None:
        low = max(low, float(min(spec.value_domain)))
        high = min(high, float(max(spec.value_domain)))
    if low == -math.inf or high == math.inf:
        raise ContractError(
            f"{name}: inferred-integer dtype needs a bounded domain, got "
            f"[{low}, {high}]")
    return low, high


def _require_leaf_int(leaf: Any, name: str, index: int) -> int:
    return int(_coerce_leaf(leaf, "uint8", name, index))


def _check_leaf(spec: FieldSpec, name: str, index: int, leaf: Any) -> None:
    if spec.dtype == "bool":
        _coerce_leaf(leaf, "bool", name, index)
        return
    if spec.dtype == "uint8":
        number = int(_coerce_leaf(leaf, "uint8", name, index))
        if spec.value_domain is not None and number not in spec.value_domain:
            raise ContractError(
                f"{name}: leaf {index} value {number} not in {spec.value_domain}")
        if not 0 <= number <= 255:
            raise ContractError(f"{name}: leaf {index} value {number} is not a uint8")
        return
    # float32
    number = float(_coerce_leaf(leaf, "float32", name, index))
    if not math.isfinite(number):
        raise ContractError(f"{name}: leaf {index} {number!r} is not finite")
    if spec.maximum is not None and number > spec.maximum:
        raise ContractError(f"{name}: leaf {index} {number} > maximum {spec.maximum}")
    if spec.minimum is not None and number < spec.minimum:
        raise ContractError(f"{name}: leaf {index} {number} < minimum {spec.minimum}")


def find_privileged_fields(obj: Any, *, path: str = "", depth: int = 0) -> tuple[str, ...]:
    """Every privileged name reachable in ``obj``, as ``dotted.path`` strings.

    Scans mapping keys, object attributes (``__dict__``/dataclass fields) and
    sequences, to :data:`_PRIVILEGED_SCAN_DEPTH` levels.  Returns ``()`` when the
    object is clean; it never returns ``None``, so ``if find_privileged_fields``
    cannot be misread.
    """

    found: list[str] = []
    if depth > _PRIVILEGED_SCAN_DEPTH:
        return ()
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if isinstance(key, str) and key in PRIVILEGED_FIELD_NAMES:
                found.append(f"{path}.{key}" if path else key)
                continue
            child = f"{path}.{key}" if path else str(key)
            found.extend(find_privileged_fields(value, path=child, depth=depth + 1))
        return tuple(found)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            name = field.name
            if name in PRIVILEGED_FIELD_NAMES:
                found.append(f"{path}.{name}" if path else name)
                continue
            child = f"{path}.{name}" if path else name
            found.extend(find_privileged_fields(getattr(obj, name), path=child,
                                                depth=depth + 1))
        return tuple(found)
    if isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            found.extend(find_privileged_fields(
                item, path=f"{path}[{index}]" if path else f"[{index}]",
                depth=depth + 1))
        return tuple(found)
    attributes = getattr(obj, "__dict__", None)
    if isinstance(attributes, dict):
        for name, value in attributes.items():
            if name in PRIVILEGED_FIELD_NAMES:
                found.append(f"{path}.{name}" if path else name)
                continue
            child = f"{path}.{name}" if path else name
            found.extend(find_privileged_fields(value, path=child, depth=depth + 1))
    return tuple(found)


def check_no_privileged_fields(obj: Any, *, what: str = "actor input") -> None:
    """Raise if ``obj`` carries any :data:`PRIVILEGED_FIELD_NAMES` field."""

    found = find_privileged_fields(obj)
    if found:
        raise ContractError(
            f"{what} carries privileged fields {list(found)}; the actor may see "
            f"pixels and its own executed keys only "
            f"(forbidden: {list(FORBIDDEN_ACTOR_FIELDS)})")


def assert_not_privileged(obj: Any, *, what: str = "actor input") -> Any:
    """Sentinel: refuse ``obj`` when it carries truth/teacher/judge fields.

    Returns ``obj`` unchanged when clean, so it can be used inline
    (``compile(context=assert_not_privileged(ctx))``) without hiding a failure:
    the failure is the exception, never a falsy return.
    """

    check_no_privileged_fields(obj, what=what)
    return obj


@dataclasses.dataclass(frozen=True)
class ActorObservation:
    """The actor's *whole* input surface, as a named record.

    A caller may hand a plain mapping to :func:`check_actor_observation` instead;
    this dataclass exists so the seven field names cannot be typo'd and so the
    forbidden-field sentinel has something explicit to inspect.
    """

    global_rgb: Any
    play_gray: Any
    hud_gray: Any
    executed_key_history: Any
    history_valid: Any
    elapsed_capture_s: Any
    memory: Any

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in ACTOR_OBSERVATION_KEYS}


def check_history_valid_prefix(history_valid: Any) -> tuple[bool, ...]:
    """``history_valid`` must be a leading run of ``False`` then ``True``.

    Cold start has no history at all (all ``False``); after *k* executed
    decisions the first ``k-8`` entries may be missing.  A hole in the middle
    ("we have frame 5 but lost frames 3 and 6") cannot be represented by a
    boolean flag, and filling such a hole with an executed-idle row would be a
    fabrication, so it is refused.
    """

    items = tuple(_iter_leaves(history_valid, "history_valid"))
    expected = _element_count(FIELD_SPECS["history_valid"])
    if len(items) != expected:
        raise ContractError(
            f"history_valid: {len(items)} flags, expected {expected}")
    flags = []
    for index, item in enumerate(items):
        flags.append(bool(_coerce_leaf(item, "bool", "history_valid", index)))
    seen_valid = False
    for index, flag in enumerate(flags):
        if flag:
            seen_valid = True
        elif seen_valid:
            raise ContractError(
                f"history_valid[{index}] is False after a valid entry; missing "
                f"history in the middle of the window cannot be represented "
                f"(a zero row would be read as executed idle)")
    return tuple(flags)


def check_executed_key_history(history: Any) -> tuple[tuple[int, ...], ...]:
    """``executed_key_history`` must be ``[8, 7]`` of 0/1 in :data:`KEY_ORDER`."""

    spec = FIELD_SPECS["executed_key_history"]
    _check_array("executed_key_history", spec, history)
    rows: list[tuple[int, ...]] = []
    leaves = list(_iter_leaves(history, "executed_key_history"))
    if len(leaves) != 8 * 7:
        raise ContractError(
            f"executed_key_history: {len(leaves)} leaves, expected {8 * 7}")
    for row in range(8):
        values = []
        for column in range(7):
            values.append(int(_coerce_leaf(leaves[row * 7 + column], "uint8",
                                           "executed_key_history", row * 7 + column)))
        rows.append(tuple(values))
    return tuple(rows)


def check_actor_observation(obs: Any) -> dict[str, tuple[tuple[int, ...], str]]:
    """Validate one actor observation; return ``{field: (shape, dtype)}``.

    Refused, all with :class:`ContractError`:

    * a non-mapping, or a mapping whose key set is not exactly
      :data:`ACTOR_OBSERVATION_KEYS` (a missing field cannot be defaulted and an
      extra field could be anything, including a leak);
    * any privileged name anywhere in the structure (see
      :func:`assert_not_privileged`);
    * a wrong shape (no interpolation to another shape is accepted), a wrong
      dtype, or — for the fields small enough to scan — a value outside the
      declared domain (key history 0/1, ``history_valid`` strictly boolean and a
      leading-``False`` prefix, no NaN/inf in ``memory``, no negative capture
      interval);
    * a ``history_valid`` window with a hole in the middle.

    Arrays are duck-typed (``.shape`` / ``.dtype``), so this stays stdlib-only.
    """

    if dataclasses.is_dataclass(obs) and not isinstance(obs, type):
        obs = obs.as_dict() if hasattr(obs, "as_dict") else dataclasses.asdict(obs)
    if not isinstance(obs, Mapping):
        raise ContractError(
            f"an actor observation must be a mapping, got {type(obs).__name__}")
    check_no_privileged_fields(obs, what="ActorObservation")
    keys = tuple(obs)
    missing = [name for name in ACTOR_OBSERVATION_KEYS if name not in obs]
    if missing:
        raise ContractError(
            f"actor observation missing fields {missing}; the actor input is a "
            f"closed set and a missing field cannot be zero-filled")
    extra = [name for name in keys if name not in ACTOR_OBSERVATION_KEYS]
    if extra:
        raise ContractError(
            f"actor observation has extra fields {extra}; only "
            f"{list(ACTOR_OBSERVATION_KEYS)} may be read by the actor")
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for spec in ACTOR_OBSERVATION_FIELDS:
        result[spec.name] = _check_array(spec.name, spec, obs[spec.name])
    check_executed_key_history(obs["executed_key_history"])
    check_history_valid_prefix(obs["history_valid"])
    scalar = obs["elapsed_capture_s"]
    if isinstance(scalar, (list, tuple)):
        raise ContractError(
            "elapsed_capture_s must be a 0-d scalar, got a sequence; a batch of "
            "intervals is not one observation")
    try:
        # 0-d array scalars coerce; the shape/dtype check above already refused a
        # sequence, a bool and a non-finite value.
        seconds = float(scalar)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            f"elapsed_capture_s must be a 0-d numeric scalar, got {scalar!r}") from exc
    if not math.isfinite(seconds):
        raise ContractError(f"elapsed_capture_s {seconds!r} is not finite")
    if seconds < 0.0:
        raise ContractError(f"elapsed_capture_s {seconds} < 0")
    return result


def schema_digest(payload: Mapping[str, Any]) -> str:
    """Stable sha256 of a JSON-serialisable mapping (sorted keys, no spaces)."""

    if not isinstance(payload, Mapping):
        raise ContractError(f"payload must be a mapping, got {type(payload).__name__}")
    try:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"payload is not stably serialisable: {exc}") from exc
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 4. O2 encoding parameters (parameters + digest only; the encoder is W02)
# ---------------------------------------------------------------------------
#: Source pixel size of the TH08 capture.
SOURCE_RGB_WH: tuple[int, int] = (640, 480)
#: Playfield crop, ``(x, y, w, h)`` in source pixels.
O2_PLAY_CROP_XYWH: tuple[int, int, int, int] = (32, 16, 384, 448)
#: HUD crop, ``(x, y, w, h)``: the full right-hand 160x448 strip.
O2_HUD_CROP_XYWH: tuple[int, int, int, int] = (480, 16, 160, 448)
#: BT.601 integer luma weights ``(R, G, B)`` and the shift.
O2_GRAY_WEIGHTS: tuple[int, int, int] = (77, 151, 28)
O2_GRAY_SHIFT = 8
O2_GRAY_RULE = "(77*R+151*G+28*B)>>8"
#: 2x2 integer mean: ``uint32`` sum first, then floor division by 4.
O2_DOWNSAMPLE_FACTOR = 2
O2_DOWNSAMPLE_ACCUMULATOR = "uint32"
O2_DOWNSAMPLE_RULE = "2x2 mean; uint32 sum then floor division by 4"
O2_STORAGE_DTYPE = "uint8"
O2_NORMALIZATION = "divide by 255 in minibatch"
O2_NORMALIZE_DIVISOR = 255.0
#: Source channel order, and the RGBA order a Windows capture may hand over
#: (alpha is dropped, never blended in).
RGB_CHANNEL_ORDER: tuple[str, ...] = ("R", "G", "B")
RGBA_INPUT_ORDER: tuple[str, ...] = ("R", "G", "B", "A")
O2_PLAY_TEMPORAL_SPACING = "consecutive genuine captured game frames"
#: Bytes of the three image fields per decision (history/flags/memory excluded).
O2_IMAGE_BYTES_PER_DECISION = (
    _element_count(FIELD_SPECS["global_rgb"])
    + _element_count(FIELD_SPECS["play_gray"])
    + _element_count(FIELD_SPECS["hud_gray"])
)

if O2_IMAGE_BYTES_PER_DECISION != 474112:  # pragma: no cover - frozen arithmetic
    raise RuntimeError(
        f"O2 image bytes {O2_IMAGE_BYTES_PER_DECISION} != PLAN_SPEC 474112; the "
        f"observation shapes changed without the plan")


def o2_parameters() -> dict[str, Any]:
    """The canonical O2 parameter block :func:`o2_digest` hashes.

    Everything that changes the encoded bytes is here and nothing else: crop
    rectangles, the luma rule, the downsample rule, storage dtype, the
    normalisation point and the key order.  The real encoder is W02; freezing the
    parameters now is what lets a run identity bind to "O2" and fail when someone
    silently changes a crop.
    """

    return {
        "variant": "O2_candidate",
        "source_rgb_wh": list(SOURCE_RGB_WH),
        "rgb_channel_order": list(RGB_CHANNEL_ORDER),
        "rgba_input_order": list(RGBA_INPUT_ORDER),
        "play_crop_xywh": list(O2_PLAY_CROP_XYWH),
        "hud_crop_xywh": list(O2_HUD_CROP_XYWH),
        "gray_rule": O2_GRAY_RULE,
        "gray_weights_rgb": list(O2_GRAY_WEIGHTS),
        "gray_shift": O2_GRAY_SHIFT,
        "downsample_factor": O2_DOWNSAMPLE_FACTOR,
        "downsample_accumulator": O2_DOWNSAMPLE_ACCUMULATOR,
        "downsample_rule": O2_DOWNSAMPLE_RULE,
        "storage_dtype": O2_STORAGE_DTYPE,
        "normalization": O2_NORMALIZATION,
        "normalize_divisor": O2_NORMALIZE_DIVISOR,
        "play_temporal_spacing": O2_PLAY_TEMPORAL_SPACING,
        "global_rgb_chw": list(FIELD_SPECS["global_rgb"].shape),
        "play_gray_chw": list(FIELD_SPECS["play_gray"].shape),
        "hud_gray_chw": list(FIELD_SPECS["hud_gray"].shape),
        "executed_key_history_shape": list(FIELD_SPECS["executed_key_history"].shape),
        "history_valid_shape": list(FIELD_SPECS["history_valid"].shape),
        "memory_shape": list(FIELD_SPECS["memory"].shape),
        "elapsed_capture_s_rank": len(FIELD_SPECS["elapsed_capture_s"].shape),
        "key_order": list(KEY_ORDER),
        "image_bytes_per_decision": O2_IMAGE_BYTES_PER_DECISION,
        "forbidden_actor_fields": list(FORBIDDEN_ACTOR_FIELDS),
    }


def o2_digest(parameters: Mapping[str, Any] | None = None) -> str:
    """Stable sha256 of the O2 parameter block (run-identity binding).

    ``parameters=None`` hashes :func:`o2_parameters`.  An explicit mapping is
    hashed as given, which is how a caller proves that two runs used the same
    encoding *and* how a test proves the digest is sensitive to a single changed
    parameter.
    """

    return schema_digest(o2_parameters() if parameters is None else parameters)


# ---------------------------------------------------------------------------
# 5. time / step contract
# ---------------------------------------------------------------------------
#: A normal request holds the keys for two game frames (``PLAN_SPEC``
#: ``action.hold_game_frames``).
HOLD_GAME_FRAMES = 2
#: Nominal capture rate used to convert the external controller clock to frames.
REFERENCE_HZ = 60
#: ``elapsed_game_frames`` comes from the engine (R2-AC only, and it is truth the
#: R1 actor must not read); ``elapsed_capture_s`` comes from the controller's own
#: clock; ``invalid`` means the interval is not usable at all.
DURATION_SOURCE_ENGINE = "engine_elapsed_frames"
DURATION_SOURCE_EXTERNAL = "external_controller_clock"
DURATION_SOURCE_INVALID = "invalid"
DURATION_SOURCES: tuple[str, ...] = (
    DURATION_SOURCE_ENGINE, DURATION_SOURCE_EXTERNAL, DURATION_SOURCE_INVALID)
#: Half a frame at :data:`REFERENCE_HZ`: the largest deviation a segment may show
#: and still count as the nominal two-frame step.  A dropped or duplicated frame
#: moves the measured interval by a whole frame and is therefore refused by
#: :func:`assert_normal_two_frame` instead of being rounded back to 2.
DURATION_TOLERANCE_FRAMES = 0.5


@dataclasses.dataclass(frozen=True)
class Duration:
    """How long one decision actually took, and where that number came from.

    ``effective_frames`` is ``None`` exactly when the interval is unusable
    (``duration_source == "invalid"``), so a consumer has no numeric value to
    mistake for "two normal frames".
    """

    duration_source: str
    elapsed_game_frames: int | None
    elapsed_capture_s: float | None
    effective_frames: float | None
    hz: int

    @property
    def valid(self) -> bool:
        return self.duration_source != DURATION_SOURCE_INVALID

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration_source": self.duration_source,
            "elapsed_game_frames": self.elapsed_game_frames,
            "elapsed_capture_s": self.elapsed_capture_s,
            "effective_frames": self.effective_frames,
            "hz": self.hz,
            "valid": self.valid,
        }


def effective_frames_from_capture(elapsed_capture_s: Any, hz: Any = REFERENCE_HZ, *,
                                  duration_source: str = DURATION_SOURCE_EXTERNAL
                                  ) -> float:
    """``effective_frames = hz * elapsed_capture_s`` for the **external** clock.

    Only defined for ``duration_source == "external_controller_clock"``: the
    engine-frame source has its own number and must never be reconstructed by
    multiplying a capture interval, and the ``invalid`` source has none at all.
    Both refusals are inside this function, so a caller cannot reach the
    conversion by forgetting a branch.
    """

    if duration_source != DURATION_SOURCE_EXTERNAL:
        raise ContractError(
            f"effective_frames_from_capture is only defined for "
            f"duration_source={DURATION_SOURCE_EXTERNAL!r}, got "
            f"{duration_source!r}; the engine source reports whole frames and the "
            f"invalid source has no usable interval")
    hz = _require_int(hz, "hz")
    if hz <= 0:
        raise ContractError(f"hz must be positive, got {hz}")
    if isinstance(elapsed_capture_s, bool) or not isinstance(elapsed_capture_s, (int, float)):
        raise ContractError(
            f"elapsed_capture_s must be a number, got {elapsed_capture_s!r}")
    seconds = float(elapsed_capture_s)
    if not math.isfinite(seconds):
        raise ContractError(f"elapsed_capture_s {seconds!r} is not finite")
    if seconds <= 0.0:
        raise ContractError(
            f"elapsed_capture_s {seconds} <= 0; a decision that consumed the "
            f"clock cannot have a non-positive interval")
    return hz * seconds


def check_duration(*, duration_source: Any, elapsed_game_frames: Any = None,
                   elapsed_capture_s: Any = None, hz: Any = REFERENCE_HZ) -> Duration:
    """Validate one decision's duration record against its declared source.

    * ``engine_elapsed_frames`` — ``elapsed_game_frames`` is a positive int and is
      the effective frame count; ``elapsed_capture_s`` may be ``None`` or a real
      measurement but is **not** used to derive frames;
    * ``external_controller_clock`` — ``elapsed_capture_s`` is a positive finite
      float and is converted with :func:`effective_frames_from_capture`;
      ``elapsed_game_frames`` must be ``None`` because the R1 actor may not read
      the engine's frame counter at all;
    * ``invalid`` — neither source may be presented as the frame count:
      ``elapsed_game_frames`` must be ``None`` and ``effective_frames`` is
      ``None``.  A fault, a pause or a dropped frame cannot masquerade as a normal
      two-frame step, because there is no number left to report as one.  A capture
      interval is still accepted here for *accounting* only; it is never converted.
    """

    if not isinstance(duration_source, str):
        raise ContractError(f"duration_source must be a string, got {duration_source!r}")
    if duration_source not in DURATION_SOURCES:
        raise ContractError(
            f"duration_source {duration_source!r} not in {list(DURATION_SOURCES)}")
    hz = _require_int(hz, "hz")
    if hz <= 0:
        raise ContractError(f"hz must be positive, got {hz}")

    if elapsed_game_frames is not None:
        frames = _require_int(elapsed_game_frames, "elapsed_game_frames")
        if frames <= 0:
            raise ContractError(f"elapsed_game_frames must be > 0, got {frames}")
    else:
        frames = None

    seconds: float | None = None
    if elapsed_capture_s is not None:
        if isinstance(elapsed_capture_s, bool) or not isinstance(elapsed_capture_s, (int, float)):
            raise ContractError(
                f"elapsed_capture_s must be a number, got {elapsed_capture_s!r}")
        seconds = float(elapsed_capture_s)
        if not math.isfinite(seconds):
            raise ContractError(f"elapsed_capture_s {seconds!r} is not finite")

    if duration_source == DURATION_SOURCE_ENGINE:
        if frames is None:
            raise ContractError(
                "duration_source=engine_elapsed_frames requires "
                "elapsed_game_frames; the engine reports whole frames and no "
                "other field may stand in for it")
        if seconds is not None and seconds <= 0.0:
            raise ContractError(
                f"elapsed_capture_s {seconds} <= 0 while the clock advanced")
        return Duration(duration_source, frames, seconds, float(frames), hz)

    if duration_source == DURATION_SOURCE_EXTERNAL:
        if frames is not None:
            raise ContractError(
                "duration_source=external_controller_clock must not carry "
                "elapsed_game_frames; the engine frame counter is truth the R1 "
                "actor may not read")
        if seconds is None:
            raise ContractError(
                "duration_source=external_controller_clock requires "
                "elapsed_capture_s; the two duration sources are independent and "
                "neither may be borrowed from the other")
        effective = effective_frames_from_capture(seconds, hz,
                                                 duration_source=duration_source)
        return Duration(duration_source, None, seconds, effective, hz)

    # invalid
    if frames is not None:
        raise ContractError(
            "duration_source=invalid must not carry elapsed_game_frames; a fault, "
            "a pause or a dropped frame may not be recorded as a normal frame "
            "count")
    if seconds is not None and seconds <= 0.0:
        raise ContractError(
            f"elapsed_capture_s {seconds} <= 0; an unusable interval is not a "
            f"clock reading either")
    return Duration(duration_source, None, seconds, None, hz)


def assert_normal_two_frame(duration: Duration,
                            tolerance_frames: float = DURATION_TOLERANCE_FRAMES
                            ) -> Duration:
    """Refuse a duration that is not the nominal :data:`HOLD_GAME_FRAMES` step.

    An ``invalid`` record raises.  A record whose effective frame count is more
    than ``tolerance_frames`` away from two raises: that is a dropped, duplicated
    or paused interval, and it must be re-recorded (or rejected as a PPO sample),
    not silently rounded back to the nominal step.
    """

    if not isinstance(duration, Duration):
        raise ContractError(f"expected a Duration, got {type(duration).__name__}")
    if not duration.valid or duration.effective_frames is None:
        raise ContractError(
            f"duration_source={duration.duration_source!r} has no effective frame "
            f"count; an incomplete or invalid segment cannot pass as two frames")
    if isinstance(tolerance_frames, bool) or not isinstance(tolerance_frames, (int, float)):
        raise ContractError(f"tolerance_frames must be a number, got {tolerance_frames!r}")
    tolerance = float(tolerance_frames)
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ContractError(f"tolerance_frames must be finite and >= 0, got {tolerance}")
    delta = abs(duration.effective_frames - HOLD_GAME_FRAMES)
    if delta > tolerance:
        raise ContractError(
            f"effective_frames {duration.effective_frames} is {delta} frames away "
            f"from the nominal {HOLD_GAME_FRAMES} (tolerance {tolerance}); a "
            f"dropped/duplicated frame or a pause must be recorded as invalid, not "
            f"rounded to the normal step")
    return duration


# ---------------------------------------------------------------------------
# 6. transition fields and bootstrap
# ---------------------------------------------------------------------------
#: The three end flags are **three fields**, never one ``done``.
TRANSITION_END_FLAGS: tuple[str, ...] = ("terminated", "truncated", "segment_end")
#: Single-field names that would re-merge the three semantics.  A record using
#: one of them is refused instead of being silently reinterpreted.
TRANSITION_ALIAS_KEYS: tuple[str, ...] = (
    "done", "is_done", "episode_done", "end", "terminal", "is_terminal",
)
#: Why a *non-terminal* episode was cut.  Closed set: an unregistered reason
#: cannot be checked, so it must be added here (and to the config) first.
TRUNCATION_REASONS: tuple[str, ...] = (
    "rollout_limit", "frame_budget", "time_limit", "resource_limit",
    "owner_preempt", "delivery_timeout", "capture_gap", "protocol_invalid",
    "other",
)
#: Only one owner may hold the keys at a time.
CONTROLLER_SOURCES: tuple[str, ...] = ("actor", "session")
#: A session-driven action carries no policy gradient (``PLAN_SPEC.action``).
SESSION_ACTOR_LOSS_MASK = 0
ACTOR_LOSS_MASK_ON = 1

TRANSITION_REQUIRED_FIELDS: tuple[str, ...] = (
    "record_id", "episode_id", "obs_ref", "next_obs_ref",
    "requested_action", "executed_masks", "controller_source",
    "terminated", "truncated", "segment_end", "terminal_observation_ref",
    "elapsed_game_frames", "actor_loss_mask", "value_loss_mask",
    "privileged_ref",
)
TRANSITION_OPTIONAL_FIELDS: tuple[str, ...] = (
    "ancestor_id", "event_id", "split", "history_valid", "obs_digest",
    "action_digest", "behavior_policy_sha", "behavior_log_prob",
    "recurrent_state_version", "reward_components", "reward_total",
    "reward_source", "reward_digest", "capture_timestamps",
    "delivery_timestamps", "label_validity", "protocol_validity",
    "truncation_reason",
)
TRANSITION_FIELDS: tuple[str, ...] = (
    TRANSITION_REQUIRED_FIELDS + TRANSITION_OPTIONAL_FIELDS)


def check_end_flags(record: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    """Validate the three independent end flags of one record.

    ``terminated`` (the task really ended), ``truncated`` (sampling/resource
    interruption of a task that could continue) and ``segment_end`` (the data
    block ended) are separate booleans.  Refused: a missing flag, a non-bool,
    ``terminated`` together with ``truncated`` (a real end is not an
    interruption), and ``truncated`` without a registered
    :data:`TRUNCATION_REASONS` entry.
    """

    if not isinstance(record, Mapping):
        raise ContractError(f"record must be a mapping, got {type(record).__name__}")
    for alias in TRANSITION_ALIAS_KEYS:
        if alias in record:
            raise ContractError(
                f"record carries {alias!r}; terminated / truncated / segment_end "
                f"are three different facts and may not share one done field")
    flags: dict[str, bool] = {}
    for name in TRANSITION_END_FLAGS:
        if name not in record:
            raise ContractError(
                f"record is missing {name!r}; all of {list(TRANSITION_END_FLAGS)} "
                f"are required, and none of them may substitute for another")
        value = record[name]
        if not isinstance(value, bool):
            raise ContractError(f"{name} must be a bool, got {value!r}")
        flags[name] = value
    if flags["terminated"] and flags["truncated"]:
        raise ContractError(
            "terminated and truncated are both true; a terminal end and a "
            "sampling interruption are mutually exclusive")
    if flags["truncated"]:
        reason = record.get("truncation_reason")
        if not isinstance(reason, str) or reason not in TRUNCATION_REASONS:
            raise ContractError(
                f"truncated=True requires truncation_reason in "
                f"{list(TRUNCATION_REASONS)}, got {reason!r}")
    return flags["terminated"], flags["truncated"], flags["segment_end"]


def check_bootstrap(*, terminated: Any, truncated: Any, segment_end: Any,
                    bootstrap_mask: Any, terminal_observation_ref: Any = None,
                    next_obs_ref: Any = None,
                    reset_observation_ref: Any = None) -> int:
    """Decide whether bootstrapping is legal for this record.

    * ``terminated`` — the value target *is* the terminal reward:
      ``bootstrap_mask`` must be 0;
    * ``truncated`` — bootstrapping is legal, but only from the last genuine
      observation: a non-zero mask requires ``terminal_observation_ref``;
    * neither — the step continues inside the episode, so a non-zero mask
      requires ``next_obs_ref``, and that reference may not be the post-reset
      observation (a reset image is not the next state of this episode).

    A missing ``terminal_observation_ref`` therefore never bootstraps: there is
    no default and no fallback to ``next_obs_ref``.
    """

    for name, value in (("terminated", terminated), ("truncated", truncated),
                        ("segment_end", segment_end)):
        if not isinstance(value, bool):
            raise ContractError(f"{name} must be a bool, got {value!r}")
    if terminated and truncated:
        raise ContractError(
            "terminated and truncated are both true; a terminal end and a "
            "sampling interruption are mutually exclusive")
    mask = _require_binary(bootstrap_mask, "bootstrap_mask")

    def _ref(value: Any, name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"{name} must be a non-empty string or None, got {value!r}")
        return value

    terminal_ref = _ref(terminal_observation_ref, "terminal_observation_ref")
    next_ref = _ref(next_obs_ref, "next_obs_ref")
    reset_ref = _ref(reset_observation_ref, "reset_observation_ref")

    if terminated:
        if mask != 0:
            raise ContractError(
                "bootstrap_mask must be 0 on a terminated episode; the terminal "
                "reward is the value target, not V(next_obs)")
        return 0
    if mask == 0:
        return 0
    if reset_ref is not None and next_ref is not None and reset_ref == next_ref:
        raise ContractError(
            f"next_obs_ref {next_ref!r} is the post-reset observation; a reset "
            f"image is not the next state of this episode and cannot bootstrap it")
    if truncated:
        if terminal_ref is None:
            raise ContractError(
                "truncated=True with bootstrap_mask=1 requires "
                "terminal_observation_ref; bootstrapping an interrupted segment "
                "without its last genuine observation would use a reset image")
        return 1
    if next_ref is None:
        raise ContractError(
            "bootstrap_mask=1 on a continuing step requires next_obs_ref; there "
            "is no default next observation")
    return 1


def check_transition(record: Any, *, for_actor: bool = False,
                     bootstrap_mask: Any = None,
                     reset_observation_ref: Any = None) -> dict[str, Any]:
    """Validate one transition record (``DEVELOPMENT_GUIDE`` §6.2).

    Beyond the field set (:data:`TRANSITION_REQUIRED_FIELDS`, unknown names
    refused) and the three end flags (:func:`check_end_flags`) this enforces the
    couplings that make a record usable:

    * ``controller_source == "session"`` implies ``actor_loss_mask == 0`` — a
      session key press has no policy gradient, and no caller may claim one;
    * ``requested_action`` is a valid joint id and ``executed_masks`` has 1 or
      :data:`HOLD_GAME_FRAMES` entries in ``0..0xFFFF`` (a terminal step may
      advance a single frame);
    * ``bootstrap_mask`` (when supplied) is checked by :func:`check_bootstrap`;
    * ``for_actor=True`` refuses ``privileged_ref`` and any privileged field name.
    """

    if not isinstance(record, Mapping):
        raise ContractError(f"record must be a mapping, got {type(record).__name__}")
    for alias in TRANSITION_ALIAS_KEYS:
        if alias in record:
            raise ContractError(
                f"record carries {alias!r}; terminated / truncated / segment_end "
                f"are three different facts and may not share one done field")
    missing = [name for name in TRANSITION_REQUIRED_FIELDS if name not in record]
    if missing:
        raise ContractError(f"transition is missing required fields {missing}")
    unknown = [name for name in record if name not in TRANSITION_FIELDS]
    if unknown:
        raise ContractError(
            f"transition carries unregistered fields {unknown}; known fields are "
            f"{list(TRANSITION_FIELDS)}")
    terminated, truncated, segment_end = check_end_flags(record)

    source = record["controller_source"]
    if source not in CONTROLLER_SOURCES:
        raise ContractError(
            f"controller_source {source!r} not in {list(CONTROLLER_SOURCES)}")
    actor_loss_mask = _require_binary(record["actor_loss_mask"], "actor_loss_mask")
    value_loss_mask = _require_binary(record["value_loss_mask"], "value_loss_mask")
    if source == "session" and actor_loss_mask != SESSION_ACTOR_LOSS_MASK:
        raise ContractError(
            f"controller_source='session' requires actor_loss_mask="
            f"{SESSION_ACTOR_LOSS_MASK}, got {actor_loss_mask}; a session key press "
            f"is not an actor decision and carries no policy gradient")
    if source == "actor" and actor_loss_mask not in (0, 1):
        raise ContractError(f"actor_loss_mask {actor_loss_mask} is not 0/1")

    decode_joint(record["requested_action"])
    masks = _as_tuple(record["executed_masks"], what="executed_masks")
    if len(masks) not in (1, HOLD_GAME_FRAMES):
        raise ContractError(
            f"executed_masks has {len(masks)} frames; a decision covers "
            f"{HOLD_GAME_FRAMES} frames (a terminal step may cover 1)")
    for index, mask in enumerate(masks):
        mask = _require_int(mask, f"executed_masks[{index}]")
        if not 0 <= mask <= 0xFFFF:
            raise ContractError(
                f"executed_masks[{index}] 0x{mask:04x} outside 0..0xFFFF")

    if for_actor:
        ref = record["privileged_ref"]
        if ref is not None and ref != "" and ref != ():
            raise ContractError(
                f"an actor-visible transition must not carry privileged_ref "
                f"({ref!r}); only the training/judge side may read it")
        # ``privileged_ref`` is a *registered* transition field; it is checked for
        # emptiness above and must not trip the name scan, which is about fields
        # that have no business being in a transition at all.
        visible = {key: value for key, value in record.items()
                   if key != "privileged_ref"}
        check_no_privileged_fields(visible, what="transition")

    if bootstrap_mask is not None:
        check_bootstrap(terminated=terminated, truncated=truncated,
                        segment_end=segment_end, bootstrap_mask=bootstrap_mask,
                        terminal_observation_ref=record["terminal_observation_ref"],
                        next_obs_ref=record["next_obs_ref"],
                        reset_observation_ref=reset_observation_ref)
    return dict(record)


# ---------------------------------------------------------------------------
# 7. result JSON and exit codes
# ---------------------------------------------------------------------------
#: ``0`` — the requested work completed, or the requested gate passed.  It is
#: **not** a capability claim (a training command exiting 0 says only that the
#: training phase finished under its protocol).
EXIT_OK = 0
#: ``2`` — invalid inputs, protocol or environment.
EXIT_INVALID = 2
#: ``3`` — a valid measurement failed the requested gate.
EXIT_GATE_FAIL = 3
#: ``4`` — incomplete because of resources or interruption.
EXIT_INCOMPLETE = 4
EXIT_CODES: dict[int, str] = {
    EXIT_OK: "requested execution completed or requested gate passed; not automatic "
             "capability success",
    EXIT_INVALID: "invalid inputs protocol or environment",
    EXIT_GATE_FAIL: "valid measurement failed requested gate",
    EXIT_INCOMPLETE: "incomplete due to resources or interruption",
}
#: The four result fields every result JSON must separate.
RESULT_FIELDS: tuple[str, ...] = (
    "execution_status", "protocol_validity", "research_result", "capability_gate")
EXECUTION_STATUSES: tuple[str, ...] = ("ok", "invalid", "gate_fail", "incomplete")
EXIT_CODE_BY_EXECUTION_STATUS: dict[str, int] = {
    "ok": EXIT_OK,
    "invalid": EXIT_INVALID,
    "gate_fail": EXIT_GATE_FAIL,
    "incomplete": EXIT_INCOMPLETE,
}
PROTOCOL_VALIDITY_VALUES: tuple[str, ...] = ("valid", "invalid", "not_run")
RESULT_STATUSES: tuple[str, ...] = (
    "not_run", "pass", "fail", "inconclusive", "invalid", "incomplete")
CAPABILITY_GATE_VALUES: tuple[str, ...] = (
    "not_run", "not_applicable", "pass", "fail", "invalid")
#: Never defaulted to anything else: a plain run cannot claim a capability.
CAPABILITY_GATE_DEFAULT = "not_run"
#: Gates a *result constructor* may never set: only an independent judgement
#: (:func:`set_capability_gate`) may, and it has to carry evidence.
CAPABILITY_GATE_JUDGED_ONLY: tuple[str, ...] = ("pass", "fail")
#: Command classes whose exit 0 must never promote ``capability_gate``.
TRAINING_COMMAND_CLASSES: tuple[str, ...] = ("training",)
#: Command classes a v430 CLI may declare.
COMMAND_CLASSES: tuple[str, ...] = (
    "audit", "contract", "training", "evaluation", "confirmation", "adaptation",
    "settlement", "planning", "data", "teacher",
)


def _check_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(f"{name} must be a 64-character lowercase hex sha256, got {value!r}")
    if any(char not in "0123456789abcdef" for char in value):
        raise ContractError(f"{name} must be lowercase hex, got {value!r}")
    return value


def new_result(*, execution_status: Any = "ok", protocol_validity: Any = "valid",
               research_result: Mapping[str, Any] | None = None,
               capability_gate: Any = None,
               **extra: Any) -> dict[str, Any]:
    """Build the four-field result skeleton of a v430 command.

    ``capability_gate`` defaults to :data:`CAPABILITY_GATE_DEFAULT`
    (``"not_run"``) and **cannot** be set to ``"pass"``/``"fail"`` here: the
    constructor is what an exit-0 code path uses, so it must not be able to
    promote a capability claim.  Use :func:`set_capability_gate` with independent
    judge evidence for that.

    ``research_result`` is either a mapping containing a ``status`` from
    :data:`RESULT_STATUSES` or ``None`` for ``{"status": "not_run"}``.
    """

    if execution_status not in EXECUTION_STATUSES:
        raise ContractError(
            f"execution_status {execution_status!r} not in {list(EXECUTION_STATUSES)}")
    if protocol_validity not in PROTOCOL_VALIDITY_VALUES:
        raise ContractError(
            f"protocol_validity {protocol_validity!r} not in "
            f"{list(PROTOCOL_VALIDITY_VALUES)}")
    if capability_gate is None:
        gate = CAPABILITY_GATE_DEFAULT
    elif capability_gate in CAPABILITY_GATE_JUDGED_ONLY:
        raise ContractError(
            f"capability_gate={capability_gate!r} cannot be set by a result "
            f"constructor; only an independent judgement may set it "
            f"(set_capability_gate with evidence), and an exit-0 command does not "
            f"promote a capability")
    elif capability_gate not in CAPABILITY_GATE_VALUES:
        raise ContractError(
            f"capability_gate {capability_gate!r} not in {list(CAPABILITY_GATE_VALUES)}")
    else:
        gate = capability_gate

    if research_result is None:
        research: dict[str, Any] = {"status": "not_run"}
    else:
        if not isinstance(research_result, Mapping):
            raise ContractError(
                f"research_result must be a mapping, got "
                f"{type(research_result).__name__}")
        status = research_result.get("status")
        if status not in RESULT_STATUSES:
            raise ContractError(
                f"research_result.status {status!r} not in {list(RESULT_STATUSES)}")
        research = dict(research_result)

    result: dict[str, Any] = {
        "execution_status": execution_status,
        "protocol_validity": protocol_validity,
        "research_result": research,
        "capability_gate": gate,
    }
    for key in extra:
        if key in RESULT_FIELDS:
            raise ContractError(
                f"{key!r} is a result field and cannot be passed as an extra")
    result.update(extra)
    return result


def set_capability_gate(result: Mapping[str, Any], gate: Any, *,
                        judged_from: Mapping[str, Any]) -> dict[str, Any]:
    """Set ``capability_gate`` from an **independent** judgement.

    ``judged_from`` must carry the judge identity and the sha256 of the sealed
    evidence it read; an empty or partial record is refused.  The execution
    status is *not* touched, so a run that exited 0 keeps saying that it merely
    completed.
    """

    if not isinstance(result, Mapping):
        raise ContractError(f"result must be a mapping, got {type(result).__name__}")
    if gate not in CAPABILITY_GATE_VALUES:
        raise ContractError(
            f"capability gate {gate!r} not in {list(CAPABILITY_GATE_VALUES)}")
    if not isinstance(judged_from, Mapping) or not judged_from:
        raise ContractError(
            "setting capability_gate requires a non-empty judged_from record "
            "(independent judge identity and sealed-evidence sha256)")
    judge = judged_from.get("judge")
    if not isinstance(judge, str) or not judge.strip():
        raise ContractError("judged_from.judge must be a non-empty identifier")
    _check_sha256(judged_from.get("sha256"), "judged_from.sha256")
    if gate in CAPABILITY_GATE_JUDGED_ONLY:
        pass
    elif judged_from.get("force_clear") is not True:
        raise ContractError(
            f"clearing capability_gate to {gate!r} needs judged_from.force_clear="
            f"true together with the evidence; it is not the default")
    updated = dict(result)
    updated["capability_gate"] = gate
    updated["capability_judgement"] = dict(judged_from)
    return updated


def exit_code_for(result: Mapping[str, Any]) -> int:
    """The process exit code implied by a result's ``execution_status``."""

    if not isinstance(result, Mapping):
        raise ContractError(f"result must be a mapping, got {type(result).__name__}")
    status = result.get("execution_status")
    if status not in EXIT_CODE_BY_EXECUTION_STATUS:
        raise ContractError(
            f"execution_status {status!r} not in {list(EXECUTION_STATUSES)}")
    return EXIT_CODE_BY_EXECUTION_STATUS[status]


def assert_result_schema(result: Any) -> dict[str, Any]:
    """Validate a result JSON against :data:`RESULT_FIELDS` and the exit codes.

    Refused: a missing field, an unknown ``execution_status``/``protocol_validity``
    /``capability_gate``, ``execution_status="ok"`` with an invalid protocol or a
    mismatching ``exit_code``, a ``capability_gate`` of ``pass``/``fail`` without
    the ``capability_judgement`` evidence block, and any result claiming a
    capability while its ``command_class`` is a training class.
    """

    if not isinstance(result, Mapping):
        raise ContractError(f"result must be a mapping, got {type(result).__name__}")
    missing = [name for name in RESULT_FIELDS if name not in result]
    if missing:
        raise ContractError(f"result is missing fields {missing}")
    execution_status = result["execution_status"]
    if execution_status not in EXECUTION_STATUSES:
        raise ContractError(
            f"execution_status {execution_status!r} not in {list(EXECUTION_STATUSES)}")
    protocol_validity = result["protocol_validity"]
    if protocol_validity not in PROTOCOL_VALIDITY_VALUES:
        raise ContractError(
            f"protocol_validity {protocol_validity!r} not in "
            f"{list(PROTOCOL_VALIDITY_VALUES)}")
    research = result["research_result"]
    if not isinstance(research, Mapping):
        raise ContractError(
            f"research_result must be a mapping, got {type(research).__name__}")
    if research.get("status") not in RESULT_STATUSES:
        raise ContractError(
            f"research_result.status {research.get('status')!r} not in "
            f"{list(RESULT_STATUSES)}")
    gate = result["capability_gate"]
    if gate not in CAPABILITY_GATE_VALUES:
        raise ContractError(
            f"capability_gate {gate!r} not in {list(CAPABILITY_GATE_VALUES)}")
    if "exit_code" in result:
        expected = EXIT_CODE_BY_EXECUTION_STATUS[execution_status]
        if _require_int(result["exit_code"], "exit_code") != expected:
            raise ContractError(
                f"exit_code {result['exit_code']} does not match "
                f"execution_status={execution_status!r} (expected {expected})")
    if protocol_validity == "invalid" and execution_status != "invalid":
        raise ContractError(
            f"protocol_validity='invalid' requires execution_status='invalid' "
            f"(exit {EXIT_INVALID}), got {execution_status!r}")
    if execution_status == "ok" and protocol_validity != "valid":
        raise ContractError(
            f"execution_status='ok' requires protocol_validity='valid', got "
            f"{protocol_validity!r}; a run cannot complete under an invalid protocol")
    if gate in CAPABILITY_GATE_JUDGED_ONLY:
        judgement = result.get("capability_judgement")
        if not isinstance(judgement, Mapping) or not judgement:
            raise ContractError(
                f"capability_gate={gate!r} requires a capability_judgement record "
                f"from an independent judge; nothing may promote itself")
        judge = judgement.get("judge")
        if not isinstance(judge, str) or not judge.strip():
            raise ContractError("capability_judgement.judge must be a non-empty identifier")
        _check_sha256(judgement.get("sha256"), "capability_judgement.sha256")
        command_class = result.get("command_class")
        if command_class in TRAINING_COMMAND_CLASSES:
            raise ContractError(
                f"a {command_class!r} command may not report "
                f"capability_gate={gate!r}; training exit 0 is not a capability")
    if "command_class" in result and result["command_class"] not in COMMAND_CLASSES:
        raise ContractError(
            f"command_class {result['command_class']!r} not in "
            f"{list(COMMAND_CLASSES)}")
    return dict(result)


# ---------------------------------------------------------------------------
# constant blocks mirrored by ``configs/v430/contracts.json``
# ---------------------------------------------------------------------------
def action_parameters() -> dict[str, Any]:
    """The action block the config mirrors (and the digest of it)."""

    return {
        "heads": {head: HEAD_SIZES[head] for head in HEAD_ORDER},
        "head_order": list(HEAD_ORDER),
        "joint_count": JOINT_COUNT,
        "joint_formula": "((move*2+focus)*2+shoot)*2+bomb",
        "factorization": "independent categorical heads; summed log-probability "
                         "and entropy",
        "hold_game_frames": HOLD_GAME_FRAMES,
        "forced_shoot": FORCED_SHOOT,
        "truth_action_mask": TRUTH_ACTION_MASK,
        "actual_key_history": True,
        "owner_exclusive": True,
        "session_actor_loss_mask": SESSION_ACTOR_LOSS_MASK,
        "controller_sources": list(CONTROLLER_SOURCES),
        "state_machine_semantics": "requested action and emitted masks both logged; "
                                   "compile identically in sampling and evaluation",
        "bomb_semantics": "press-edge trigger recorded separately from the held "
                          "level; no hidden hit state",
    }


def time_parameters() -> dict[str, Any]:
    """The time/step block the config mirrors."""

    return {
        "hold_game_frames": HOLD_GAME_FRAMES,
        "reference_hz": REFERENCE_HZ,
        "duration_sources": list(DURATION_SOURCES),
        "duration_tolerance_frames": DURATION_TOLERANCE_FRAMES,
        "effective_frames_from_capture": "hz * elapsed_capture_s, external source "
                                         "only",
        "independent_sources": True,
    }


def result_parameters() -> dict[str, Any]:
    """The result-JSON/exit-code block the config mirrors."""

    return {
        "exit_codes": {str(code): text for code, text in EXIT_CODES.items()},
        "result_fields": list(RESULT_FIELDS),
        "execution_statuses": list(EXECUTION_STATUSES),
        "exit_code_by_execution_status": dict(EXIT_CODE_BY_EXECUTION_STATUS),
        "protocol_validity_values": list(PROTOCOL_VALIDITY_VALUES),
        "result_statuses": list(RESULT_STATUSES),
        "capability_gate_values": list(CAPABILITY_GATE_VALUES),
        "capability_gate_default": CAPABILITY_GATE_DEFAULT,
        "capability_gate_judged_only": list(CAPABILITY_GATE_JUDGED_ONLY),
        "training_command_classes": list(TRAINING_COMMAND_CLASSES),
        "command_classes": list(COMMAND_CLASSES),
    }


def observation_parameters() -> dict[str, Any]:
    """The full O2 block plus the digest, as stored in the config."""

    block = o2_parameters()
    block["digest"] = o2_digest(block)
    block["fields"] = [
        {"name": spec.name, "shape": list(spec.shape), "dtype": spec.dtype,
         "minimum": spec.minimum, "maximum": spec.maximum,
         "value_domain": (list(spec.value_domain)
                          if spec.value_domain is not None else None),
         "semantics": spec.semantics}
        for spec in ACTOR_OBSERVATION_FIELDS
    ]
    return block


def contract_constants() -> dict[str, Any]:
    """Every constant ``configs/v430/contracts.json`` mirrors.

    The test asserts ``config["constants"] == contract_constants()``, so a change
    on either side fails the suite instead of drifting.
    """

    return {
        "contract_version": CONTRACT_VERSION,
        "spec_revision": SPEC_REVISION,
        "config_schema": CONFIG_SCHEMA,
        "action": action_parameters(),
        "observation": observation_parameters(),
        "time": time_parameters(),
        "results": result_parameters(),
        "forbidden_actor_fields": list(FORBIDDEN_ACTOR_FIELDS),
        "privileged_field_names": list(PRIVILEGED_FIELD_NAMES),
        "transition": {
            "end_flags": list(TRANSITION_END_FLAGS),
            "alias_keys": list(TRANSITION_ALIAS_KEYS),
            "required_fields": list(TRANSITION_REQUIRED_FIELDS),
            "optional_fields": list(TRANSITION_OPTIONAL_FIELDS),
            "truncation_reasons": list(TRUNCATION_REASONS),
            "bootstrap_rule": "terminated -> 0; truncated -> needs "
                              "terminal_observation_ref; continuing -> needs "
                              "next_obs_ref; a reset image never bootstraps",
        },
    }
