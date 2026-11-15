"""The 72-combination action state machine (v4.3.0 work package W01).

One decision = one :class:`~touhou_rl.v430.contracts.JointAction` =
``(move, focus, shoot, bomb)`` held for
:data:`~touhou_rl.v430.contracts.HOLD_GAME_FRAMES` game frames.  This module turns
that request into the per-frame engine button masks and keeps the *actual key
state* so the next request is compiled from what was really sent.

The four properties the work package is built around:

* **no implicit SHOOT.**  The mask's shoot bit is the request's ``shoot`` bit,
  frame by frame.  :mod:`touhou_rl.v400.gym.actions.cand_to_mask` ORs SHOOT on
  unconditionally (it encodes the teacher's 18 candidates, where "not shooting"
  is not a choice); it is *not* used to build a v4.3.0 mask, and
  :func:`touhou_rl.v430.contracts.check_no_forced_shoot_rule` re-checks the
  result.  This is the v4.2.0 defect made impossible by construction instead of
  by caller discipline.
* **bomb press edge.**  A bomb is triggered once by the 0 -> 1 transition of the
  key, not by every frame in which the key is down.  The compiled record carries
  ``requested``, the emitted level and the edge frames, so a replay can audit
  exactly one trigger per press.  No hidden "already bombed" state is consulted:
  the only history is the previously **emitted** level.
* **single owner.**  :class:`KeyOwnership` hands the keyboard to exactly one of
  ``actor`` / ``session``.  A second acquisition raises; switching is an explicit
  release + acquire.  Session decisions carry ``actor_loss_mask = 0``.
* **no truth.**  The compiler never sees bullets, lives or bomb counts.  An
  optional ``context`` argument is scanned by
  :func:`touhou_rl.v430.contracts.assert_not_privileged` and otherwise ignored, so
  a caller that *has* truth cannot pass it in, and a request that spends a bomb
  the character does not own is still emitted and recorded.

Sampling and evaluation share :meth:`ActionCompiler.compile_request`; there is no
evaluation-only branch, no rescue heuristic and no second mask builder.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Sequence

from touhou_rl.spec import BTN_BOMB, BTN_FOCUS, BTN_SHOOT
from touhou_rl.v400.gym.actions import (
    DIRECTION_MASK,
    MOVES_PER_CAND,
    N_CAND,
    _MOVE_BITS,
    cand_to_mask,
)
from touhou_rl.v430.contracts import (
    ACTOR_LOSS_MASK_ON,
    CONTRACT_VERSION,
    CONTROLLER_SOURCES,
    HOLD_GAME_FRAMES,
    JOINT_COUNT,
    N_MOVE,
    SESSION_ACTOR_LOSS_MASK,
    ContractError,
    JointAction,
    assert_not_privileged,
    check_joint_action,
    check_no_forced_shoot_rule,
    decode_joint,
    encode_joint,
)

__all__ = [
    "AGENT_MASK_V430",
    "DIRECTION_MASKS",
    "CompiledAction",
    "CompilerState",
    "FramePlan",
    "KeyOwnership",
    "MaskRangeError",
    "OwnershipError",
    "OwnerConflictError",
    "PrivilegedInputError",
    "ActionCompiler",
    "all_joint_masks",
    "check_agent_mask",
    "check_mask",
    "compile_masks",
    "mask_for",
    "assert_not_privileged",
]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class ActionError(ContractError):
    """Base class of the action-layer contract violations."""


class MaskRangeError(ActionError):
    """A button mask is outside ``0..0xFFFF`` or contains system-input bits."""


class OwnershipError(ActionError):
    """The keyboard ownership bookkeeping is inconsistent."""


class OwnerConflictError(OwnershipError):
    """A second owner tried to take the keyboard while it was held."""


class PrivilegedInputError(ActionError):
    """Truth/teacher input was handed to the action compiler."""


# ---------------------------------------------------------------------------
# button tables (reused, never copied)
# ---------------------------------------------------------------------------
#: The direction bits of every ``move``, straight from
#: :mod:`touhou_rl.v400.gym.actions`.  There is exactly one direction table in the
#: repository; v4.3.0 indexes it, it does not restate it.
DIRECTION_MASKS: tuple[int, ...] = _MOVE_BITS
#: Bits the agent may ever emit: the four directions plus shoot/bomb/focus.
AGENT_MASK_V430 = DIRECTION_MASK | BTN_SHOOT | BTN_BOMB | BTN_FOCUS


def _fold(bits: Iterable[int]) -> int:
    total = 0
    for bit in bits:
        total |= bit
    return total


# Import-time guards, in the spirit of the v4.2.0 contract module: if the frozen
# v4.0.0 tables ever change under us, fail loudly here instead of emitting a wrong
# key for every decision of a run.
if len(DIRECTION_MASKS) != N_MOVE:  # pragma: no cover - frozen table
    raise RuntimeError(
        f"touhou_rl.v400.gym.actions has {len(DIRECTION_MASKS)} direction masks, "
        f"v4.3.0 pins N_MOVE={N_MOVE}")
if N_CAND != N_MOVE * 2 or MOVES_PER_CAND != N_MOVE:  # pragma: no cover - frozen
    raise RuntimeError(
        f"the reused v4.0.0 action table is not a {N_MOVE}-move x 2-focus table: "
        f"N_CAND={N_CAND}, MOVES_PER_CAND={MOVES_PER_CAND}")
if _fold(DIRECTION_MASKS) != DIRECTION_MASK:  # pragma: no cover - frozen table
    raise RuntimeError("DIRECTION_MASK is not the OR of the per-move masks")
for _move in range(N_MOVE):  # pragma: no cover - frozen table
    if DIRECTION_MASKS[_move] | BTN_SHOOT != cand_to_mask(_move, 0):
        raise RuntimeError(
            f"direction bits for move {_move} disagree with "
            f"touhou_rl.v400.gym.actions.cand_to_mask; the reused table changed")
if (BTN_SHOOT & BTN_BOMB) or (BTN_SHOOT & BTN_FOCUS) or (BTN_BOMB & BTN_FOCUS):  # pragma: no cover
    raise RuntimeError("shoot/bomb/focus button bits overlap")
if any(DIRECTION_MASK & bit for bit in (BTN_SHOOT, BTN_BOMB, BTN_FOCUS)):  # pragma: no cover
    raise RuntimeError("a direction bit collides with an action button bit")


# ---------------------------------------------------------------------------
# mask construction and validation
# ---------------------------------------------------------------------------
def check_mask(mask: Any, *, name: str = "mask") -> int:
    """``mask`` must be an ``int`` in ``0..0xFFFF`` (the engine's 16-bit word)."""

    if isinstance(mask, bool) or not isinstance(mask, int):
        raise MaskRangeError(f"{name} must be an int, got {mask!r}")
    if not 0 <= mask <= 0xFFFF:
        raise MaskRangeError(
            f"{name} 0x{mask & 0xFFFFFFFF:08x} outside 0..0xFFFF; the engine input "
            f"word is 16 bits")
    return mask


def check_agent_mask(mask: Any, *, name: str = "mask") -> int:
    """As :func:`check_mask`, and the mask may only contain agent-emittable bits.

    Menu/skip/enter and every other system input is *not* the policy's to press
    (``touhou_rl.spec.BTN_MENU`` etc.), so a mask carrying one is refused rather
    than forwarded to the engine.
    """

    mask = check_mask(mask, name=name)
    extra = mask & ~AGENT_MASK_V430
    if extra:
        raise MaskRangeError(
            f"{name} 0x{mask:04x} contains non-agent bits 0x{extra:04x}; the policy "
            f"may only emit directions plus shoot/bomb/focus")
    return mask


def mask_for(move: Any, focus: Any, shoot: Any, bomb: Any) -> int:
    """The engine mask of one frame of ``(move, focus, shoot, bomb)``.

    This is the *only* mask builder in v4.3.0.  Note that it does not call
    ``cand_to_mask``: that function forces SHOOT on, which is exactly what this
    work package must not do.
    """

    if isinstance(move, bool) or not isinstance(move, int) or not 0 <= move < N_MOVE:
        raise ActionError(f"move {move!r} outside [0, {N_MOVE})")
    for name, value in (("focus", focus), ("shoot", shoot), ("bomb", bomb)):
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            raise ActionError(f"{name} must be 0 or 1, got {value!r}")
    mask = DIRECTION_MASKS[move]
    if shoot:
        mask |= BTN_SHOOT
    if focus:
        mask |= BTN_FOCUS
    if bomb:
        mask |= BTN_BOMB
    return mask


def all_joint_masks() -> tuple[tuple[int, tuple[int, int, int, int]], ...]:
    """``(mask, action)`` for all :data:`JOINT_COUNT` combinations.

    The import-time guard below proves the 72 masks are *distinct*: if two
    combinations collapsed onto one mask, the joint distribution and the executed
    key would disagree for one of them and the PPO ratio would be computed on an
    action that was never sent.
    """

    out: list[tuple[int, tuple[int, int, int, int]]] = []
    for joint in range(JOINT_COUNT):
        action = decode_joint(joint)
        out.append((mask_for(*action), action))
    return tuple(out)


if len({mask for mask, _action in all_joint_masks()}) != JOINT_COUNT:  # pragma: no cover
    raise RuntimeError(
        f"the {JOINT_COUNT} v4.3.0 combinations do not produce {JOINT_COUNT} "
        f"distinct masks; two actions would be indistinguishable on the wire")


def compile_masks(action: JointAction | Sequence[int], *,
                  frames: int = HOLD_GAME_FRAMES) -> tuple[int, ...]:
    """Stateless mask list of one request (no ownership, no bomb edge accounting).

    This is the pure part of the state machine: it is what the compiler emits for
    a hold window before the bomb edge bookkeeping is applied.  Callers that need
    the bomb edge and the actor/session mask must use
    :meth:`ActionCompiler.compile_request`, which is the single entry point the
    rollout and the evaluator share.
    """

    joint = check_joint_action(action)
    if isinstance(frames, bool) or not isinstance(frames, int):
        raise ActionError(f"frames must be an int, got {frames!r}")
    if frames != HOLD_GAME_FRAMES:
        raise ActionError(
            f"frames {frames} != HOLD_GAME_FRAMES {HOLD_GAME_FRAMES}; a different "
            f"hold would change the meaning of every logged decision")
    mask = mask_for(*joint.as_tuple())
    return tuple(check_agent_mask(mask) for _ in range(frames))


# ---------------------------------------------------------------------------
# keyboard ownership
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class OwnershipToken:
    """Proof that one source currently owns the keyboard."""

    source: str
    serial: int

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "serial": self.serial}


def _check_source(source: Any) -> str:
    if not isinstance(source, str):
        raise OwnershipError(f"controller_source must be a string, got {source!r}")
    if source not in CONTROLLER_SOURCES:
        raise OwnershipError(
            f"controller_source {source!r} not in {list(CONTROLLER_SOURCES)}")
    return source


class KeyOwnership:
    """At most one key owner at a time; switching is explicit.

    ``actor`` and ``session`` must never press keys simultaneously — two sources
    of input would make the executed key history and the logged action describe
    different users.  Acquisition while the keyboard is held raises
    :class:`OwnerConflictError`, and a released token can never send again: every
    compile re-checks its token against the bus.
    """

    def __init__(self) -> None:
        self._token: OwnershipToken | None = None
        self._serial = 0

    @property
    def holder(self) -> str | None:
        return None if self._token is None else self._token.source

    @property
    def token(self) -> OwnershipToken | None:
        return self._token

    def acquire(self, source: str) -> OwnershipToken:
        """Take the keyboard for ``source``; raise if anyone (even itself) holds it."""

        source = _check_source(source)
        if self._token is not None:
            raise OwnerConflictError(
                f"{source!r} cannot take the keyboard while "
                f"{self._token.source!r} holds it (token #{self._token.serial}); "
                f"switching owners must be explicit: release the current token "
                f"first")
        self._serial += 1
        self._token = OwnershipToken(source=source, serial=self._serial)
        return self._token

    def release(self, token: OwnershipToken) -> None:
        """Release ``token``; a stale or foreign token is refused."""

        if not isinstance(token, OwnershipToken):
            raise OwnershipError(f"expected an OwnershipToken, got {token!r}")
        if self._token is None:
            raise OwnershipError(
                f"no owner holds the keyboard; cannot release #{token.serial}")
        if token != self._token:
            raise OwnershipError(
                f"token #{token.serial} ({token.source}) is not the current "
                f"holder #{self._token.serial} ({self._token.source}); a stale "
                f"token cannot release the keyboard")
        self._token = None

    def assert_token(self, token: OwnershipToken, *, what: str = "compile") -> None:
        """Raise unless ``token`` is the live owner."""

        if not isinstance(token, OwnershipToken):
            raise OwnershipError(f"expected an OwnershipToken, got {token!r}")
        if self._token is None:
            raise OwnerConflictError(
                f"{token.source!r} tried to {what} after the keyboard was released")
        if self._token != token:
            raise OwnerConflictError(
                f"{token.source!r} (token #{token.serial}) tried to {what} while "
                f"{self._token.source!r} (token #{self._token.serial}) owns the "
                f"keyboard; concurrent owners are refused")

    def as_dict(self) -> dict[str, Any]:
        return {"holder": self.holder, "serial": self._serial}


# ---------------------------------------------------------------------------
# compiled decision
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FramePlan:
    """One frame of a compiled decision: level bits plus the bomb edge flag."""

    index: int
    mask: int
    bomb_level: int
    bomb_edge: bool

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "mask": self.mask,
                "mask_hex": f"0x{self.mask:04x}",
                "bomb_level": self.bomb_level, "bomb_edge": self.bomb_edge}


@dataclasses.dataclass(frozen=True)
class CompiledAction:
    """The audit record of one compiled decision.

    ``requested`` and ``executed_masks`` are both kept: a zero-update replay must
    reproduce the masks bit for bit, and a reader must be able to see that
    ``shoot=0`` really produced a mask without the shoot bit.
    """

    requested: JointAction
    executed_masks: tuple[int, ...]
    frames: tuple[FramePlan, ...]
    move: int
    focus: int
    shoot: int
    bomb_requested: int
    bomb_emitted: tuple[int, ...]
    bomb_edge_frames: tuple[int, ...]
    bomb_edge_consumed: bool
    bomb_held: bool
    controller_source: str
    actor_loss_mask: int
    hold_game_frames: int
    decision_index: int
    bomb_edges_total: int

    def __post_init__(self) -> None:
        if not isinstance(self.requested, JointAction):
            raise ActionError(
                f"requested must be a JointAction, got {type(self.requested).__name__}")
        if len(self.executed_masks) != self.hold_game_frames:
            raise ActionError(
                f"{len(self.executed_masks)} masks for a "
                f"{self.hold_game_frames}-frame decision")
        if len(self.frames) != len(self.executed_masks):
            raise ActionError(
                f"{len(self.frames)} frame plans for {len(self.executed_masks)} masks")
        for index, mask in enumerate(self.executed_masks):
            check_agent_mask(mask, name=f"executed_masks[{index}]")
        for index, plan in enumerate(self.frames):
            check_agent_mask(plan.mask, name=f"frames[{index}].mask")
            if plan.mask != self.executed_masks[index]:
                raise ActionError(
                    f"frame {index} mask 0x{plan.mask:04x} != executed mask "
                    f"0x{self.executed_masks[index]:04x}")
        if self.controller_source not in CONTROLLER_SOURCES:
            raise ActionError(
                f"controller_source {self.controller_source!r} not in "
                f"{list(CONTROLLER_SOURCES)}")
        expected_mask = (SESSION_ACTOR_LOSS_MASK if self.controller_source == "session"
                         else ACTOR_LOSS_MASK_ON)
        if self.actor_loss_mask != expected_mask and not (
                self.controller_source == "actor" and self.actor_loss_mask == 0):
            raise ActionError(
                f"actor_loss_mask {self.actor_loss_mask} is wrong for "
                f"controller_source={self.controller_source!r}")
        if self.bomb_edge_consumed != bool(self.bomb_edge_frames):
            raise ActionError(
                "bomb_edge_consumed disagrees with bomb_edge_frames "
                f"({self.bomb_edge_consumed} vs {list(self.bomb_edge_frames)})")

    @property
    def mask(self) -> int:
        """The first executed mask (the one the engine sees first)."""

        return self.executed_masks[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": CONTRACT_VERSION,
            "requested": self.requested.as_dict(),
            "requested_joint": self.requested.joint,
            "executed_masks": list(self.executed_masks),
            "executed_masks_hex": [f"0x{mask:04x}" for mask in self.executed_masks],
            "frames": [plan.as_dict() for plan in self.frames],
            "move": self.move,
            "focus": self.focus,
            "shoot": self.shoot,
            "bomb_requested": self.bomb_requested,
            "bomb_emitted": list(self.bomb_emitted),
            "bomb_edge_frames": list(self.bomb_edge_frames),
            "bomb_edge_consumed": self.bomb_edge_consumed,
            "bomb_held": self.bomb_held,
            "bomb_edges_total": self.bomb_edges_total,
            "controller_source": self.controller_source,
            "actor_loss_mask": self.actor_loss_mask,
            "hold_game_frames": self.hold_game_frames,
            "decision_index": self.decision_index,
        }


@dataclasses.dataclass(frozen=True)
class CompilerState:
    """The whole mutable state of a compiler, as a value.

    It is *only* the keys that were actually emitted: the last mask and the last
    bomb level (plus counters for auditing).  There is deliberately no field for
    "did the bomb succeed", "how many bombs are left" or "was the player hit" —
    the compiler has no hidden hit state to reflect on.
    """

    last_mask: int = 0
    last_bomb_level: int = 0
    decision_index: int = 0
    bomb_edges_total: int = 0

    def __post_init__(self) -> None:
        check_agent_mask(self.last_mask, name="state.last_mask")
        if self.last_bomb_level not in (0, 1):
            raise ActionError(
                f"state.last_bomb_level {self.last_bomb_level!r} is not 0/1")
        for name, value in (("decision_index", self.decision_index),
                            ("bomb_edges_total", self.bomb_edges_total)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ActionError(f"state.{name} must be a non-negative int, got {value!r}")
        if self.bomb_edges_total > self.decision_index:
            raise ActionError(
                f"bomb_edges_total {self.bomb_edges_total} > decision_index "
                f"{self.decision_index}")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# the compiler
# ---------------------------------------------------------------------------
class ActionCompiler:
    """Turn requests into per-frame masks, holding the actual key state.

    One instance is one key owner.  Constructing it acquires the keyboard on its
    :class:`KeyOwnership` bus; constructing a second compiler for the *other*
    source on the same bus raises :class:`OwnerConflictError` until the first is
    released.  :meth:`compile_request` is the same call in rollout and in
    evaluation — there is no mode flag that could make the evaluated policy
    behave differently from the trained one.
    """

    def __init__(self, source: str = "actor", *,
                 ownership: KeyOwnership | None = None,
                 hold_frames: int = HOLD_GAME_FRAMES,
                 context: Any = None) -> None:
        self._source = _check_source(source)
        if isinstance(hold_frames, bool) or not isinstance(hold_frames, int):
            raise ActionError(f"hold_frames must be an int, got {hold_frames!r}")
        if hold_frames != HOLD_GAME_FRAMES:
            raise ActionError(
                f"hold_frames {hold_frames} != HOLD_GAME_FRAMES "
                f"{HOLD_GAME_FRAMES}; the step size is part of the contract")
        self._hold_frames = hold_frames
        self.ownership = KeyOwnership() if ownership is None else ownership
        self._token = self.ownership.acquire(self._source)
        self._state = CompilerState()
        if context is not None:
            self._check_context(context)

    # -- identity ---------------------------------------------------------
    @property
    def source(self) -> str:
        return self._source

    @property
    def token(self) -> OwnershipToken:
        return self._token

    @property
    def hold_game_frames(self) -> int:
        return self._hold_frames

    @property
    def state(self) -> CompilerState:
        return self._state

    @property
    def actor_loss_mask(self) -> int:
        """``0`` for a session compiler, ``1`` for an actor compiler."""

        return (SESSION_ACTOR_LOSS_MASK if self._source == "session"
                else ACTOR_LOSS_MASK_ON)

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": CONTRACT_VERSION,
            "controller_source": self._source,
            "actor_loss_mask": self.actor_loss_mask,
            "hold_game_frames": self._hold_frames,
            "state": self._state.as_dict(),
            "ownership": self.ownership.as_dict(),
            "token": self._token.as_dict(),
        }

    # -- lifecycle --------------------------------------------------------
    def release(self) -> None:
        """Give the keyboard back; the compiler can no longer compile."""

        self.ownership.release(self._token)

    def close(self) -> None:
        """Alias of :meth:`release` (context-manager friendly)."""

        self.release()

    def __enter__(self) -> "ActionCompiler":
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.ownership.holder == self._source:
            self.release()

    # -- state snapshot / restore ----------------------------------------
    def snapshot(self) -> CompilerState:
        """The current state, as a frozen value (for replay reproduction)."""

        return self._state

    def restore(self, state: CompilerState) -> None:
        """Reset the state machine to ``state``; used to reproduce a replay."""

        if not isinstance(state, CompilerState):
            raise ActionError(
                f"expected a CompilerState, got {type(state).__name__}")
        # ``CompilerState.__post_init__`` already validated the ranges; rebuild it
        # so a subclass cannot smuggle an unvalidated instance through.
        self._state = CompilerState(state.last_mask, state.last_bomb_level,
                                    state.decision_index, state.bomb_edges_total)

    # -- privileged-input sentinel ---------------------------------------
    @staticmethod
    def _check_context(context: Any) -> None:
        try:
            assert_not_privileged(context, what="action-compiler context")
        except ContractError as exc:
            raise PrivilegedInputError(str(exc)) from exc

    # -- the one compile entry point -------------------------------------
    def compile_request(self, move: Any, focus: Any, shoot: Any, bomb: Any, *,
                        context: Any = None) -> CompiledAction:
        """Compile ``(move, focus, shoot, bomb)`` into ``hold_game_frames`` masks.

        The request is honoured literally:

        * ``shoot`` is the mask's shoot bit — ``0`` really is "not shooting";
        * ``focus``/``move`` are the focus bit and the reused direction bits;
        * ``bomb`` asserts the bomb bit for the hold window and triggers exactly
          one rising edge per 0 -> 1 transition of the request stream; the edge is
          reported in ``bomb_edge_frames``/``bomb_edge_consumed`` whether or not
          the character had a bomb left, because the compiler cannot know that and
          must not ask.

        ``context`` is *never* used to alter the action: it is scanned by
        :func:`assert_not_privileged` (so truth cannot even be handed in) and
        otherwise ignored.  Two calls with and without a clean context return
        identical masks.
        """

        self.ownership.assert_token(self._token, what="compile")
        action = JointAction(*decode_joint(encode_joint(move, focus, shoot, bomb)))
        if context is not None:
            self._check_context(context)

        previous = self._state
        level_mask = mask_for(*action.as_tuple())
        edge = action.bomb == 1 and previous.last_bomb_level == 0

        frames: list[FramePlan] = []
        masks: list[int] = []
        bomb_emitted: list[int] = []
        edge_frames: list[int] = []
        for index in range(self._hold_frames):
            is_edge = bool(edge and index == 0)
            frame_mask = check_agent_mask(level_mask, name=f"frame[{index}]")
            frames.append(FramePlan(index=index, mask=frame_mask,
                                    bomb_level=action.bomb, bomb_edge=is_edge))
            masks.append(frame_mask)
            bomb_emitted.append(action.bomb)
            if is_edge:
                edge_frames.append(index)

        decision_index = previous.decision_index + 1
        edges_total = previous.bomb_edges_total + (1 if edge else 0)
        compiled = CompiledAction(
            requested=action,
            executed_masks=tuple(masks),
            frames=tuple(frames),
            move=action.move,
            focus=action.focus,
            shoot=action.shoot,
            bomb_requested=action.bomb,
            bomb_emitted=tuple(bomb_emitted),
            bomb_edge_frames=tuple(edge_frames),
            bomb_edge_consumed=edge,
            bomb_held=bool(action.bomb == 1 and not edge),
            controller_source=self._source,
            actor_loss_mask=self.actor_loss_mask,
            hold_game_frames=self._hold_frames,
            decision_index=decision_index,
            bomb_edges_total=edges_total,
        )
        # The shoot invariant is re-checked against the *emitted* masks, not the
        # request, so a future edit to ``mask_for`` cannot quietly reintroduce the
        # forced-SHOOT defect.
        check_no_forced_shoot_rule(masks, shoot_requests=[action.shoot] * len(masks))
        self._state = CompilerState(
            last_mask=masks[-1],
            last_bomb_level=action.bomb,
            decision_index=decision_index,
            bomb_edges_total=edges_total,
        )
        return compiled

    def compile_action(self, action: JointAction | Sequence[int], *,
                       context: Any = None) -> CompiledAction:
        """``compile_request`` for a :class:`JointAction` or a 4-sequence.

        Delegates instead of re-implementing: sampling hands in a joint action,
        evaluation hands in the recorded four heads, and both reach the same
        state machine.
        """

        joint = check_joint_action(action)
        return self.compile_request(*joint.as_tuple(), context=context)

    def compile_sequence(self, requests: Iterable[Sequence[int]], *,
                         context: Any = None) -> tuple[CompiledAction, ...]:
        """Compile several requests on one state machine, in order."""

        out = []
        for request in requests:
            out.append(self.compile_action(request, context=context))
        return tuple(out)
