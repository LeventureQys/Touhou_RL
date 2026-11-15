"""Pixel-only session state machine (v4.3.0 work package W01, slice B).

The *session* is the second (and only other) key owner of a run.  When the game
is not under actor control -- a title menu, a stage banner, the ending, the
game-over prompt, a death animation -- somebody has to press keys, and that
somebody is this module.  ``DEVELOPMENT_GUIDE`` §5.2 and ``PLAN_SPEC.action``
fix the four properties it is built around:

* **pixels and the session's own key history only.**  The machine is fed
  :class:`PixelEvidence` records that describe what a *pixel* detector saw
  (which scene, whether the player sprite was located, whether a hit flash was
  visible, how many life pips are drawn) plus its own capture clock.  It never
  receives ``stage_id``, ``game_over``, ``stage_clear`` or any other engine
  truth: a mapping passed as evidence is scanned for
  :data:`SESSION_TRUTH_FIELDS` before it is parsed and a hit raises
  :class:`SessionTruthError`.  The engine may still record those values in an
  independent judge stream; it may not steer this state machine.
* **one owner.**  The machine shares one
  :class:`touhou_rl.v430.actions.KeyOwnership` bus with the actor.  It acquires
  the bus only while a session state actually presses keys and releases it as
  soon as the play field is live again; a concurrent actor compiler makes
  :meth:`SessionMachine.take_control` raise
  :class:`touhou_rl.v430.actions.OwnerConflictError`.  Every emitted decision
  carries ``controller_source='session'`` and ``actor_loss_mask=0`` (the actor's
  own compiler carries ``1``); the mask is re-checked at emit time instead of
  being trusted from the compiler.
* **``unknown`` is a real state.**  When the pixels do not identify the screen,
  the state is ``unknown`` -- not ``playing``, not ``game_over``, and it presses
  *nothing*.  There is no zero-fill, no "assume idle" and no default action: a
  consumer that reads ``state == 'unknown'`` must widen its uncertainty window
  rather than continue as if the run were live.
* **no lost reward while the session drives.**  Every sample's capture interval
  is attributed to the span of the state that was in force *during* it, and
  :meth:`SessionMachine.attach_reward` / :meth:`SessionMachine.note_event` let a
  caller stamp reward components and events onto the currently open span (a
  stage banner or a game-over prompt is exactly when the engine pays the clear
  or the loss).  :meth:`SessionLedger.assert_complete` then proves, sample by
  sample, that the spans tile the whole session with no gap in frame index, no
  gap in capture time and no sample left out.

What this module deliberately does **not** do: decide PPO/GAE semantics, own the
pixel detector (``feedback`` is W05), record engine truth, or press a key in a
state it cannot identify.  Rewards are *carried*, not interpreted.

States (``DEVELOPMENT_GUIDE`` §9 lists the permitted set)::

    playing / hit_pending / respawn / transition / game_over / menu / ending /
    unknown

Transition rules (:meth:`SessionMachine._infer`; every input is a pixel
feature, in this order):

=========================================  ==============================
pixel evidence                             resulting state
=========================================  ==============================
``ending_credits`` scene                   ``ending``
``game_over_prompt`` scene                 ``game_over``
``menu_overlay`` scene                     ``menu``
``stage_banner`` scene                     ``transition``
hit flash / life-pip drop in ``playing``   ``hit_pending``
live scene, play field >90% dark           ``unknown`` (features disagree)
live scene, player sprite located          ``playing``
live scene, no player sprite, pending      ``respawn``
live scene, no player sprite, otherwise    ``unknown``
``unrecognised`` scene                     keep ``hit_pending``/``respawn``,
                                           else ``unknown``
=========================================  ==============================

A terminal state is left only by a positive pixel signal (a menu or a live play
field), and leaving ``game_over`` / ``ending`` is what increments
:attr:`SessionMachine.restart_count` -- with the reason recorded in a
:class:`RestartRecord`.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import math
from typing import Any, ClassVar, Iterator, Mapping, Sequence

import touhou_rl.v430.contracts as C
from touhou_rl.v430.actions import ActionCompiler, KeyOwnership

__all__ = [
    # states
    "PLAYING",
    "HIT_PENDING",
    "RESPAWN",
    "TRANSITION",
    "GAME_OVER",
    "MENU",
    "ENDING",
    "UNKNOWN",
    "SESSION_STATES",
    "SESSION_ACTION_STATES",
    "SESSION_ACTIONS",
    # pixel scenes
    "SCENE_PLAY",
    "SCENE_BANNER",
    "SCENE_MENU",
    "SCENE_ENDING",
    "SCENE_OVER",
    "SCENE_UNRECOGNISED",
    "SESSION_SCENES",
    # closed vocabularies
    "SESSION_EVENTS",
    "RESTART_REASONS",
    "MANUAL_RESTART_REASONS",
    "SESSION_TRUTH_FIELDS",
    "DIGEST_SCOPE_PIXELS",
    "DARK_PLAY_RATIO_MAX",
    "SESSION_ACTOR_LOSS_MASK",
    "SESSION_SOURCE_IMPORTS_ALLOWED",
    "SESSION_SOURCE_IMPORTS_FORBIDDEN",
    # errors
    "SessionError",
    "SessionEvidenceError",
    "SessionTruthError",
    "SessionClockError",
    "SessionStateError",
    "SessionLedgerError",
    # records
    "PixelEvidence",
    "SessionAction",
    "RestartRecord",
    "SessionSpan",
    "SessionStep",
    "SessionLedger",
    "SessionMachine",
    # helpers
    "assert_pixel_only",
    "find_session_truth_fields",
    "session_action_for_state",
    "session_source_audit",
]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class SessionError(C.ContractError):
    """Base class of the session-layer contract violations.

    Subclasses :class:`touhou_rl.v430.contracts.ContractError`, so a caller that
    only knows the contract layer still catches it.  Nothing here returns a
    sentinel for a failure: the failure is always the exception.
    """


class SessionEvidenceError(SessionError):
    """The pixel evidence record is malformed (missing/extra/ill-typed field)."""


class SessionTruthError(SessionError):
    """Engine truth (stage / game_over / lives / ...) was handed to the session."""


class SessionClockError(SessionError):
    """The capture clock is not monotonic or an interval is not a real one."""


class SessionStateError(SessionError):
    """The state machine was used out of order (e.g. finished, then advanced)."""


class SessionLedgerError(SessionError):
    """The span ledger does not tile the session; time/reward would be lost."""


# ---------------------------------------------------------------------------
# states
# ---------------------------------------------------------------------------
#: The play field is live and the actor owns the keys.
PLAYING = "playing"
#: A hit was seen in pixels; the run is in its death/knockback beat.
HIT_PENDING = "hit_pending"
#: The run is between the death and the reappearance of the player sprite.
RESPAWN = "respawn"
#: A stage banner / stage-clear interstitial is on screen.
TRANSITION = "transition"
#: The game-over prompt is on screen and the run has terminated.
GAME_OVER = "game_over"
#: A title/menu overlay is on screen.
MENU = "menu"
#: The ending credits are on screen.
ENDING = "ending"
#: The pixels do not identify the screen.  A legal, *sticky* state.
UNKNOWN = "unknown"

#: Every state the session may report (``DEVELOPMENT_GUIDE`` §9).
SESSION_STATES: tuple[str, ...] = (
    PLAYING, HIT_PENDING, RESPAWN, TRANSITION, GAME_OVER, MENU, ENDING, UNKNOWN)

#: States in which the session itself presses keys, i.e. acquires the keyboard.
SESSION_ACTION_STATES: tuple[str, ...] = (MENU, TRANSITION, ENDING, GAME_OVER)

#: What the session presses while it owns the keyboard: advance / confirm.
#: ``move=0`` is the idle move (no direction bit) and ``shoot=1`` is the confirm
#: button (Z in TH08).  Every value is agent-emittable
#: (:func:`touhou_rl.v430.actions.check_agent_mask`), so no menu/skip system bit
#: is smuggled in.
SESSION_ACTIONS: dict[str, tuple[int, int, int, int]] = {
    MENU: (0, 0, 1, 0),
    TRANSITION: (0, 0, 1, 0),
    ENDING: (0, 0, 1, 0),
    GAME_OVER: (0, 0, 1, 0),
}

#: ``PLAN_SPEC.action.session_actor_loss_mask`` / ``contracts.SESSION_ACTOR_LOSS_MASK``.
SESSION_ACTOR_LOSS_MASK = C.SESSION_ACTOR_LOSS_MASK


# ---------------------------------------------------------------------------
# pixel scenes (produced by the pixel detector, never by the engine)
# ---------------------------------------------------------------------------
SCENE_PLAY = "play_field_live"
SCENE_BANNER = "stage_banner"
SCENE_MENU = "menu_overlay"
SCENE_ENDING = "ending_credits"
SCENE_OVER = "game_over_prompt"
SCENE_UNRECOGNISED = "unrecognised"

#: Closed vocabulary of the pixel scene label.  A detector that cannot classify
#: the frame must say ``unrecognised``; it may not guess ``play_field_live``.
SESSION_SCENES: tuple[str, ...] = (
    SCENE_PLAY, SCENE_BANNER, SCENE_MENU, SCENE_ENDING, SCENE_OVER,
    SCENE_UNRECOGNISED)

#: Above this fraction of near-black play-field pixels the "live play field"
#: label is contradicted by the pixels themselves and the session refuses to call
#: the frame live.  Two independent pixel features disagreeing is *unknown*, not
#: a tie-break in favour of the convenient answer.
DARK_PLAY_RATIO_MAX = 0.9


# ---------------------------------------------------------------------------
# closed vocabularies
# ---------------------------------------------------------------------------
#: Events a span may carry.  Auto-recorded by the state machine or attached by a
#: caller through :meth:`SessionMachine.note_event`; an unregistered string is
#: refused so a downstream consumer can enumerate the space.
SESSION_EVENTS: tuple[str, ...] = (
    "play_field_live",
    "stage_banner_seen",
    "menu_seen",
    "ending_seen",
    "game_over_prompt_seen",
    "hit_flash_seen",
    "life_pips_dropped",
    "respawn_wait",
    "player_visible_again",
    "play_field_without_player",
    "dark_play_field",
    "scene_unrecognised",
    "restart_observed",
    "manual_restart",
)

#: Why a run was restarted, as observed from pixels (the last three are the
#: operator-only ones and are never inferred).
RESTART_REASONS: tuple[str, ...] = (
    "game_over_restart",     # left game_over straight into menu/play
    "ending_restart",        # left ending into menu/play (a new loop)
    "manual_reset",          # the operator restarted the run from the menu
    "scene_reload",          # the capture/engine was reloaded under the session
    "harness_reconnect",     # the harness reconnected and re-opened the session
)

#: Reasons only a human/operator call may claim.
MANUAL_RESTART_REASONS: tuple[str, ...] = (
    "manual_reset", "scene_reload", "harness_reconnect")

#: Field names the session must never see.  These are the engine's answers to
#: "what stage is it", "is the run over", "how many lives are left" -- the very
#: facts the session is only allowed to *infer from pixels*.  A mapping carrying
#: one of these is refused before it is parsed, so a caller that happens to have
#: truth cannot accidentally condition the key presser on it.
SESSION_TRUTH_FIELDS: tuple[str, ...] = (
    # stage identity
    "stage_id", "stage", "stage_number", "stage_index", "stage_truth",
    "stage_clear", "stage_clear_flag", "stage_clear_screen_state",
    # terminal truth
    "game_over", "game_over_flag", "is_game_over", "game_manager_state",
    "supervisor_state",
    # resource truth
    "lives", "life", "lives_truth", "bomb_count", "bomb_stock",
    "bomb_count_truth", "power", "power_truth", "resource_truth",
    "resource_state",
    # engine clock / identity
    "engine_frame", "engine_frame_index", "engine_frames", "seed", "script_id",
    "enemy_table",
    # teacher / judge surfaces
    "teacher_action", "future_tape", "privileged_ref", "privileged_state",
    "oracle_state", "rng_state", "replay_input_tape",
)

#: ``digest_scope`` of a captured frame's bytes: pixels are binary, so they are
#: hashed raw and never line-ending-normalised
#: (``evidence/W00/SHA_LINE_ENDING_FINDING.md``).  The text-scope counterpart is
#: :data:`touhou_rl.v430.evaluate.DIGEST_SCOPE_TEXT`; ``tests/test_v430_session.py``
#: asserts the two vocabularies agree, so this literal cannot drift.
DIGEST_SCOPE_PIXELS = "binary/raw"

#: Modules this file may import.  Anything engine-, teacher- or training-shaped
#: is refused by :func:`session_source_audit`.
SESSION_SOURCE_IMPORTS_ALLOWED: tuple[str, ...] = (
    "__future__", "ast", "dataclasses", "hashlib", "math", "typing",
    "touhou_rl.v430.contracts", "touhou_rl.v430.actions",
)
SESSION_SOURCE_IMPORTS_FORBIDDEN: tuple[str, ...] = (
    "touhou_rl.state", "touhou_rl.memory", "touhou_rl.env", "touhou_rl.session",
    "touhou_rl.v400.gym.env", "touhou_rl.v400.gym.protocol",
    "touhou_rl.v400.gym.shm", "touhou_rl.v400.teacher",
    "touhou_rl.v430.teacher", "touhou_rl.v430.rl", "touhou_rl.v430.learn",
    "touhou_rl.v430.models", "touhou_rl.v430.curriculum", "torch", "numpy",
)


# ---------------------------------------------------------------------------
# privileged-field scan (session-specific names, on top of the contract's)
# ---------------------------------------------------------------------------
def _iter_named_fields(value: Any, path: str = "", depth: int = 0
                       ) -> Iterator[str]:
    if depth > 6:
        return
    if isinstance(value, Mapping):
        for name, child in value.items():
            if not isinstance(name, str):
                continue
            child_path = f"{path}.{name}" if path else name
            if name in SESSION_TRUTH_FIELDS:
                yield child_path
                continue
            yield from _iter_named_fields(child, child_path, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            yield from _iter_named_fields(item, child_path, depth + 1)
        return
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        yield from _iter_named_fields(attributes, path, depth + 1)


def find_session_truth_fields(obj: Any) -> tuple[str, ...]:
    """Paths of every :data:`SESSION_TRUTH_FIELDS` name reachable from ``obj``."""

    return tuple(_iter_named_fields(obj))


def assert_pixel_only(obj: Any, *, what: str = "session pixel evidence") -> Any:
    """Sentinel: refuse ``obj`` if it carries engine truth or a contract-forbidden field.

    Returns ``obj`` unchanged when clean (so it can be used inline); the failure
    is the exception, never a falsy return.  Two independent name sets are
    checked: this module's :data:`SESSION_TRUTH_FIELDS` and the frozen
    :data:`touhou_rl.v430.contracts.PRIVILEGED_FIELD_NAMES`.
    """

    found = find_session_truth_fields(obj)
    if found:
        raise SessionTruthError(
            f"{what} carries engine truth {list(found)}; the session may only "
            f"read pixels and its own key history, so stage/game-over/resource "
            f"values may not condition a key press "
            f"(forbidden: {list(SESSION_TRUTH_FIELDS)})")
    try:
        C.assert_not_privileged(obj, what=what)
    except C.ContractError as exc:
        raise SessionTruthError(str(exc)) from exc
    return obj


# ---------------------------------------------------------------------------
# small validators
# ---------------------------------------------------------------------------
def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SessionEvidenceError(f"{name} must be an int, got {value!r}")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SessionEvidenceError(f"{name} must be a bool, got {value!r}")
    return value


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SessionEvidenceError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SessionEvidenceError(f"{name} {number!r} is not finite")
    return number


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SessionEvidenceError(
            f"{name} must be a 64-character lowercase hex sha256, got {value!r}")
    if any(char not in "0123456789abcdef" for char in value):
        raise SessionEvidenceError(f"{name} must be lowercase hex, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# pixel evidence
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class PixelEvidence:
    """One captured frame as the pixel detector describes it.

    Every field is either a pixel measurement or the session's *own* capture
    clock.  There is deliberately no field for the stage number, the game-over
    flag, the life counter or the frame counter of the engine: those are truth,
    and :meth:`from_mapping` refuses them by name.

    ``elapsed_capture_s`` is the interval the capture clock advanced since the
    previous sample -- the same quantity the actor's observation carries
    (``DEVELOPMENT_GUIDE`` §4.1) and what the span durations are built from, so
    the session period is measured on the shared external clock (§5.3) and never
    on the engine's frame counter.
    """

    frame_index: int
    frame_digest: str
    captured_at_s: float
    elapsed_capture_s: float
    scene: str
    play_field_dark_ratio: float
    player_sprite_visible: bool
    hit_flash_visible: bool
    visible_life_pips: int | None = None
    frame_digest_scope: str = DIGEST_SCOPE_PIXELS

    #: Exactly the keys :meth:`from_mapping` accepts (a ``ClassVar``, so it is
    #: not a dataclass field and cannot be passed as one).
    FIELDS: ClassVar[tuple[str, ...]] = (
        "frame_index", "frame_digest", "captured_at_s", "elapsed_capture_s",
        "scene", "play_field_dark_ratio", "player_sprite_visible",
        "hit_flash_visible", "visible_life_pips", "frame_digest_scope")

    def __post_init__(self) -> None:
        _require_int(self.frame_index, "frame_index")
        if self.frame_index < 0:
            raise SessionEvidenceError(
                f"frame_index {self.frame_index} is negative; the capture clock "
                f"starts at zero")
        _require_sha256(self.frame_digest, "frame_digest")
        if self.frame_digest_scope != DIGEST_SCOPE_PIXELS:
            raise SessionEvidenceError(
                f"frame_digest_scope {self.frame_digest_scope!r} must be "
                f"{DIGEST_SCOPE_PIXELS!r}; captured pixels are binary and are "
                f"never line-ending-normalised")
        captured = _require_finite(self.captured_at_s, "captured_at_s")
        if captured < 0.0:
            raise SessionEvidenceError(f"captured_at_s {captured} < 0")
        elapsed = _require_finite(self.elapsed_capture_s, "elapsed_capture_s")
        if elapsed <= 0.0:
            raise SessionEvidenceError(
                f"elapsed_capture_s {elapsed} <= 0; a captured frame consumed a "
                f"positive interval of the capture clock, and a fake zero would "
                f"drop time from the reward chain")
        if elapsed > captured + 1e-9:
            raise SessionEvidenceError(
                f"elapsed_capture_s {elapsed} exceeds captured_at_s {captured}; "
                f"the interval cannot start before the session clock")
        if not isinstance(self.scene, str) or self.scene not in SESSION_SCENES:
            raise SessionEvidenceError(
                f"scene {self.scene!r} not in {list(SESSION_SCENES)}; a detector "
                f"that cannot classify a frame must say {SCENE_UNRECOGNISED!r}")
        dark = _require_finite(self.play_field_dark_ratio, "play_field_dark_ratio")
        if not 0.0 <= dark <= 1.0:
            raise SessionEvidenceError(
                f"play_field_dark_ratio {dark} outside [0, 1]")
        _require_bool(self.player_sprite_visible, "player_sprite_visible")
        _require_bool(self.hit_flash_visible, "hit_flash_visible")
        if self.visible_life_pips is not None:
            _require_int(self.visible_life_pips, "visible_life_pips")
            if self.visible_life_pips < 0:
                raise SessionEvidenceError(
                    f"visible_life_pips {self.visible_life_pips} is negative")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PixelEvidence":
        """Build evidence from a mapping, refusing truth names and unknown keys.

        Order matters and is fail-closed: the truth scan runs **first**, so
        ``{"game_over": True, ...}`` reports *"engine truth"* rather than a
        generic "unknown field", and the diagnostic names the red line that was
        crossed.
        """

        if not isinstance(mapping, Mapping):
            raise SessionEvidenceError(
                f"pixel evidence must be a mapping, got {type(mapping).__name__}")
        assert_pixel_only(mapping, what="session pixel evidence")
        unknown = [key for key in mapping if key not in cls.FIELDS]
        if unknown:
            raise SessionEvidenceError(
                f"pixel evidence carries unregistered fields {unknown}; the "
                f"session accepts exactly {list(cls.FIELDS)} and nothing else")
        optional = ("visible_life_pips", "frame_digest_scope")
        missing = [key for key in cls.FIELDS
                   if key not in mapping and key not in optional]
        if missing:
            raise SessionEvidenceError(f"pixel evidence is missing fields {missing}")
        return cls(**{key: mapping[key] for key in mapping})

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "frame_digest": self.frame_digest,
            "frame_digest_scope": self.frame_digest_scope,
            "captured_at_s": self.captured_at_s,
            "elapsed_capture_s": self.elapsed_capture_s,
            "scene": self.scene,
            "play_field_dark_ratio": self.play_field_dark_ratio,
            "player_sprite_visible": self.player_sprite_visible,
            "hit_flash_visible": self.hit_flash_visible,
            "visible_life_pips": self.visible_life_pips,
            "source": "pixels",
        }


def _coerce_evidence(value: Any) -> PixelEvidence:
    if isinstance(value, PixelEvidence):
        assert_pixel_only(value, what="session pixel evidence")
        return value
    return PixelEvidence.from_mapping(value)


# ---------------------------------------------------------------------------
# session action
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SessionAction:
    """The key request the session makes in one state."""

    state: str
    move: int
    focus: int
    shoot: int
    bomb: int

    def __post_init__(self) -> None:
        if self.state not in SESSION_ACTION_STATES:
            raise SessionStateError(
                f"state {self.state!r} presses no keys; only "
                f"{list(SESSION_ACTION_STATES)} are session-driven")
        for name, value in (("move", self.move), ("focus", self.focus),
                            ("shoot", self.shoot), ("bomb", self.bomb)):
            _require_int(value, name)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.move, self.focus, self.shoot, self.bomb)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "requested": list(self.as_tuple()),
            "controller_source": "session",
            "actor_loss_mask": SESSION_ACTOR_LOSS_MASK,
        }


def session_action_for_state(state: str) -> SessionAction | None:
    """The session's key request in ``state``, or ``None`` when the actor drives.

    ``unknown`` maps to ``None`` *by construction*: a state the pixels did not
    identify never presses a key, so it cannot be silently treated as playing
    (which would hand the game to the actor) or as game-over (which would
    restart the run).
    """

    if state not in SESSION_STATES:
        raise SessionStateError(f"unknown session state {state!r}")
    request = SESSION_ACTIONS.get(state)
    if request is None:
        return None
    # Positional expansion: ``state`` must not be passed by keyword *and* be the
    # first positional argument of ``*request`` (that raised TypeError for every
    # session-driven state, i.e. the session could not press a key at all).
    return SessionAction(state, *request)


# ---------------------------------------------------------------------------
# restart record
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class RestartRecord:
    """One audited restart: when, why, from which state into which."""

    restart_index: int
    reason: str
    source: str
    from_state: str
    to_state: str
    frame_index: int
    frame_digest: str
    frame_digest_scope: str
    captured_at_s: float
    session_elapsed_s: float

    def __post_init__(self) -> None:
        _require_int(self.restart_index, "restart_index")
        if self.restart_index < 1:
            raise SessionStateError(
                f"restart_index {self.restart_index} must be >= 1")
        if self.reason not in RESTART_REASONS:
            raise SessionStateError(
                f"restart reason {self.reason!r} not in {list(RESTART_REASONS)}")
        if self.source not in ("pixels", "operator"):
            raise SessionStateError(
                f"restart source {self.source!r} must be 'pixels' or 'operator'")
        if self.from_state not in SESSION_STATES or self.to_state not in SESSION_STATES:
            raise SessionStateError(
                f"restart {self.from_state!r} -> {self.to_state!r} uses an "
                f"unregistered state")
        _require_sha256(self.frame_digest, "restart.frame_digest")
        if self.frame_digest_scope != DIGEST_SCOPE_PIXELS:
            raise SessionStateError(
                f"restart.frame_digest_scope {self.frame_digest_scope!r} must be "
                f"{DIGEST_SCOPE_PIXELS!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "restart_index": self.restart_index,
            "reason": self.reason,
            "source": self.source,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "frame_index": self.frame_index,
            "frame_digest": self.frame_digest,
            "frame_digest_scope": self.frame_digest_scope,
            "captured_at_s": self.captured_at_s,
            "session_elapsed_s": self.session_elapsed_s,
        }


# ---------------------------------------------------------------------------
# spans and the ledger
# ---------------------------------------------------------------------------
def _span_duration(elapsed_capture_s: float, hz: int) -> dict[str, Any]:
    """Validate a span's duration on the shared external clock.

    A positive interval is a normal :func:`touhou_rl.v430.contracts.check_duration`
    record for ``external_controller_clock``.  A **zero** interval (only possible
    for a trailing span that never received a sample: the state changed on the
    very last captured frame) is recorded as ``duration_source='invalid'`` with
    ``effective_frames=None`` -- the contract's own language for "this interval
    is not a usable frame count".  Nothing is rounded up to the nominal step.
    """

    if elapsed_capture_s > 0.0:
        return C.check_duration(
            duration_source=C.DURATION_SOURCE_EXTERNAL,
            elapsed_capture_s=elapsed_capture_s, hz=hz).as_dict()
    return {
        "duration_source": C.DURATION_SOURCE_INVALID,
        "elapsed_game_frames": None,
        "elapsed_capture_s": 0.0,
        "effective_frames": None,
        "hz": hz,
        "valid": False,
        "note": (
            "zero-length trailing span: the state changed on the last captured "
            "frame, so no capture interval elapsed under the new state. It is "
            "not padded to a nominal step and carries no reward unless a caller "
            "attached one."),
    }


@dataclasses.dataclass(frozen=True)
class SessionSpan:
    """One contiguous period in which the session state did not change.

    ``duration`` is a validated :class:`touhou_rl.v430.contracts.Duration` on the
    shared *external* controller clock, so the span can be spliced into the
    caller's return computation with the same units as an actor step.  Nothing
    here says how the reward should be discounted.
    """

    span_index: int
    state: str
    key_owner: str | None
    actor_loss_mask: int | None
    start_frame_index: int
    end_frame_index: int
    start_sample_index: int | None
    end_sample_index: int | None
    start_captured_at_s: float
    end_captured_at_s: float
    elapsed_capture_s: float
    duration: dict[str, Any]
    sample_count: int
    start_frame_digest: str
    end_frame_digest: str
    events: tuple[str, ...]
    reward_components: dict[str, float]
    session_decisions: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_index": self.span_index,
            "state": self.state,
            "key_owner": self.key_owner,
            "actor_loss_mask": self.actor_loss_mask,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "start_sample_index": self.start_sample_index,
            "end_sample_index": self.end_sample_index,
            "start_captured_at_s": self.start_captured_at_s,
            "end_captured_at_s": self.end_captured_at_s,
            "elapsed_capture_s": self.elapsed_capture_s,
            "duration": dict(self.duration),
            "sample_count": self.sample_count,
            "start_frame_digest": self.start_frame_digest,
            "start_frame_digest_scope": DIGEST_SCOPE_PIXELS,
            "end_frame_digest": self.end_frame_digest,
            "end_frame_digest_scope": DIGEST_SCOPE_PIXELS,
            "events": list(self.events),
            "reward_components": dict(self.reward_components),
            "session_decisions": self.session_decisions,
        }


class SessionLedger:
    """Every span of one session, plus the proof that none of them is missing."""

    def __init__(self, *, spans: Sequence[SessionSpan],
                 samples: Sequence[PixelEvidence], restarts: Sequence[RestartRecord],
                 hz: int, finished: bool,
                 discarded_empty_spans: int = 0) -> None:
        self._spans = tuple(spans)
        self._samples = tuple(samples)
        self._restarts = tuple(restarts)
        self._hz = hz
        self._finished = finished
        self._discarded = discarded_empty_spans

    # -- accessors ---------------------------------------------------------
    @property
    def spans(self) -> tuple[SessionSpan, ...]:
        return self._spans

    @property
    def samples(self) -> tuple[PixelEvidence, ...]:
        return self._samples

    @property
    def restarts(self) -> tuple[RestartRecord, ...]:
        return self._restarts

    @property
    def finished(self) -> bool:
        return self._finished

    def totals(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        decisions: dict[str, int] = {}
        for span in self._spans:
            by_state[span.state] = by_state.get(span.state, 0) + 1
            if span.session_decisions:
                decisions[span.state] = (decisions.get(span.state, 0)
                                         + span.session_decisions)
        total_elapsed = sum(span.elapsed_capture_s for span in self._spans)
        return {
            "spans": len(self._spans),
            "samples": len(self._samples),
            "restarts": len(self._restarts),
            "discarded_empty_spans": self._discarded,
            "span_states": dict(sorted(by_state.items())),
            "session_decisions": dict(sorted(decisions.items())),
            "total_elapsed_capture_s": total_elapsed,
            "total_effective_frames": self._hz * total_elapsed,
            "hz": self._hz,
            "menu_waits": by_state.get(MENU, 0),
            "transition_waits": by_state.get(TRANSITION, 0),
            "ending_waits": by_state.get(ENDING, 0),
            "game_over_waits": by_state.get(GAME_OVER, 0),
            "unknown_waits": by_state.get(UNKNOWN, 0),
        }

    def reward_rollup(self) -> dict[str, dict[str, Any]]:
        """Every reward component attached during the session, never dropped.

        The session does not decide what a reward means; it guarantees that a
        component stamped onto a span (the clear reward paid while a stage banner
        is up, say) survives into this rollup together with the spans that
        carried it and the states those spans were in.
        """

        rollup: dict[str, dict[str, Any]] = {}
        for span in self._spans:
            for name, value in span.reward_components.items():
                row = rollup.setdefault(
                    name, {"sum": 0.0, "count": 0, "spans": [], "states": [],
                           "elapsed_capture_s": 0.0})
                row["sum"] += value
                row["count"] += 1
                row["spans"].append(span.span_index)
                row["elapsed_capture_s"] += span.elapsed_capture_s
                if span.state not in row["states"]:
                    row["states"].append(span.state)
        return rollup

    def events(self) -> tuple[str, ...]:
        out: list[str] = []
        for span in self._spans:
            for event in span.events:
                if event not in out:
                    out.append(event)
        return tuple(out)

    # -- completeness ------------------------------------------------------
    def assert_complete(self) -> dict[str, Any]:
        """Prove the spans tile the session; raise :class:`SessionLedgerError`.

        Six independent checks, so a dropped span cannot hide behind a
        compensating duplicate:

        1. every span either covers at least one sample or is the zero-length
           final span (marked ``duration_source='invalid'``);
        2. the spans' sample indices partition ``0 .. n-1`` exactly once each;
        3. every span's ``elapsed_capture_s`` equals the sum of its samples'
           intervals, and the total equals the samples' total;
        4. consecutive spans meet exactly (same frame index, same clock reading);
        5. every span's ``elapsed`` equals ``end_captured_at_s -
           start_captured_at_s``;
        6. the sample frame indices strictly increase (no frame counted twice).
        """

        if not self._spans:
            raise SessionLedgerError(
                "the ledger is empty; a session with no spans cannot be spliced "
                "into a return chain")
        last_index = len(self._spans) - 1
        for span in self._spans:
            zero = span.elapsed_capture_s == 0.0
            if zero:
                if span.span_index != last_index:
                    raise SessionLedgerError(
                        f"span {span.span_index} ({span.state}) has a zero capture "
                        f"interval but is not the final span; a zero-length span "
                        f"in the middle would hide the time of an unrecorded "
                        f"state change")
                if span.duration.get("duration_source") != C.DURATION_SOURCE_INVALID:
                    raise SessionLedgerError(
                        f"the zero-length final span {span.span_index} claims "
                        f"duration_source="
                        f"{span.duration.get('duration_source')!r}; a zero "
                        f"interval may not be presented as a normal step")
                if span.duration.get("effective_frames") is not None:
                    raise SessionLedgerError(
                        f"the zero-length final span {span.span_index} carries "
                        f"effective_frames="
                        f"{span.duration.get('effective_frames')!r}; a zero "
                        f"interval has no frame count")
            elif span.sample_count < 1:
                raise SessionLedgerError(
                    f"span {span.span_index} ({span.state}) covers no sample but "
                    f"has a positive interval {span.elapsed_capture_s}; the "
                    f"caller could not account for that time")
            if span.sample_count > 0:
                if span.start_sample_index is None or span.end_sample_index is None:
                    raise SessionLedgerError(
                        f"span {span.span_index} ({span.state}) covers "
                        f"{span.sample_count} samples without sample indices")
                if (span.end_sample_index - span.start_sample_index + 1
                        != span.sample_count):
                    raise SessionLedgerError(
                        f"span {span.span_index} ({span.state}) claims "
                        f"{span.sample_count} samples but its index range "
                        f"[{span.start_sample_index}, {span.end_sample_index}] "
                        f"holds "
                        f"{span.end_sample_index - span.start_sample_index + 1}")
        covered: list[int] = []
        for span in self._spans:
            if span.sample_count:
                covered.extend(range(span.start_sample_index,
                                     span.end_sample_index + 1))
        if covered != list(range(len(self._samples))):
            missing = sorted(set(range(len(self._samples))) - set(covered))
            duplicates = sorted({index for index in covered
                                 if covered.count(index) > 1})
            raise SessionLedgerError(
                f"the spans do not partition the {len(self._samples)} captured "
                f"samples (missing={missing[:10]}, duplicated={duplicates[:10]}); "
                f"the session period would lose time or reward")
        sample_seconds = sum(sample.elapsed_capture_s for sample in self._samples)
        span_seconds = 0.0
        for span in self._spans:
            if span.sample_count:
                own = sum(self._samples[index].elapsed_capture_s
                          for index in range(span.start_sample_index,
                                             span.end_sample_index + 1))
                if not math.isclose(own, span.elapsed_capture_s,
                                    rel_tol=0.0, abs_tol=1e-9):
                    raise SessionLedgerError(
                        f"span {span.span_index} ({span.state}) records "
                        f"{span.elapsed_capture_s}s but its samples add up to "
                        f"{own}s")
            span_seconds += span.elapsed_capture_s
            if not math.isclose(span.end_captured_at_s - span.start_captured_at_s,
                                span.elapsed_capture_s, rel_tol=0.0, abs_tol=1e-6):
                raise SessionLedgerError(
                    f"span {span.span_index} ({span.state}) spans "
                    f"{span.start_captured_at_s} -> {span.end_captured_at_s} "
                    f"({span.end_captured_at_s - span.start_captured_at_s}s) but "
                    f"records {span.elapsed_capture_s}s of intervals")
        if not math.isclose(span_seconds, sample_seconds, rel_tol=0.0, abs_tol=1e-9):
            raise SessionLedgerError(
                f"the spans cover {span_seconds:.9f}s of capture clock but the "
                f"samples recorded {sample_seconds:.9f}s; the difference is "
                f"duration the caller would lose")
        for previous, following in zip(self._spans, self._spans[1:]):
            if previous.end_frame_index != following.start_frame_index:
                raise SessionLedgerError(
                    f"span {previous.span_index} ends at frame "
                    f"{previous.end_frame_index} but span "
                    f"{following.span_index} starts at "
                    f"{following.start_frame_index}; the frame index would have a "
                    f"gap")
            if not math.isclose(previous.end_captured_at_s,
                                following.start_captured_at_s,
                                rel_tol=0.0, abs_tol=1e-9):
                raise SessionLedgerError(
                    f"span {previous.span_index} ends at "
                    f"{previous.end_captured_at_s} but span "
                    f"{following.span_index} starts at "
                    f"{following.start_captured_at_s}; the capture clock would "
                    f"have a gap")
        frame_numbers = [sample.frame_index for sample in self._samples]
        for earlier, later in zip(frame_numbers, frame_numbers[1:]):
            if later <= earlier:
                raise SessionLedgerError(
                    f"frame index did not advance ({earlier} -> {later}); two "
                    f"samples claim the same captured frame")
        return {
            "complete": True,
            "checks": {
                "every_span_has_samples_or_is_zero_final": True,
                "sample_indices_partition": True,
                "per_span_intervals_add_up": True,
                "capture_intervals_add_up": True,
                "frame_index_contiguous": True,
                "clock_contiguous": True,
                "span_bounds_match_intervals": True,
                "frame_index_strictly_increasing": True,
            },
            "totals": self.totals(),
            "reward_rollup": self.reward_rollup(),
            "events": list(self.events()),
            "restart_count": len(self._restarts),
            "restart_reasons": [record.reason for record in self._restarts],
            "restart_sources": [record.source for record in self._restarts],
            "duration_source": C.DURATION_SOURCE_EXTERNAL,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "v430-session-ledger/1",
            "finished": self._finished,
            "hz": self._hz,
            "duration_source": C.DURATION_SOURCE_EXTERNAL,
            "spans": [span.as_dict() for span in self._spans],
            "restarts": [record.as_dict() for record in self._restarts],
            "totals": self.totals(),
            "reward_rollup": self.reward_rollup(),
            "events": list(self.events()),
        }


# ---------------------------------------------------------------------------
# one processed sample
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SessionStep:
    """What one captured frame did to the session."""

    index: int
    previous_state: str
    state: str
    changed: bool
    events: tuple[str, ...]
    evidence: PixelEvidence
    action: Any = None            # CompiledAction | None
    session_owns_keys: bool = False
    restarts: tuple[RestartRecord, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "previous_state": self.previous_state,
            "state": self.state,
            "changed": self.changed,
            "events": list(self.events),
            "evidence": self.evidence.as_dict(),
            "action": None if self.action is None else self.action.as_dict(),
            "session_owns_keys": self.session_owns_keys,
            "restarts": [record.as_dict() for record in self.restarts],
        }


# ---------------------------------------------------------------------------
# the state machine
# ---------------------------------------------------------------------------
class SessionMachine:
    """Turn pixel evidence into a session state, key presses and an audited ledger.

    The machine is the *only* code path allowed to take the keyboard away from
    the actor, and it does so through a shared
    :class:`touhou_rl.v430.actions.KeyOwnership` bus, so a concurrent actor
    compiler is a hard error rather than a race.
    """

    def __init__(self, *, ownership: KeyOwnership | None = None,
                 capture_hz: int = C.REFERENCE_HZ,
                 context: Any = None) -> None:
        if isinstance(capture_hz, bool) or not isinstance(capture_hz, int):
            raise SessionError(f"capture_hz must be an int, got {capture_hz!r}")
        if capture_hz <= 0:
            raise SessionError(f"capture_hz must be positive, got {capture_hz}")
        if context is not None:
            # Sentinel: a caller that happens to hold engine truth cannot hand it
            # to the session "for convenience".
            assert_pixel_only(context, what="session context")
        self._ownership = KeyOwnership() if ownership is None else ownership
        self._hz = capture_hz
        self._compiler: ActionCompiler | None = None
        self._state = UNKNOWN
        self._samples: list[PixelEvidence] = []
        self._spans: list[SessionSpan] = []
        self._restarts: list[RestartRecord] = []
        self._open: dict[str, Any] | None = None
        self._finished = False
        self._last_life_pips: int | None = None
        self._discarded = 0

    # -- identity ---------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def ownership(self) -> KeyOwnership:
        return self._ownership

    @property
    def hz(self) -> int:
        return self._hz

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def restarts(self) -> tuple[RestartRecord, ...]:
        return tuple(self._restarts)

    @property
    def restart_count(self) -> int:
        return len(self._restarts)

    @property
    def session_holds_keys(self) -> bool:
        return self._compiler is not None

    @property
    def waiting(self) -> bool:
        """True while the session (not the actor) drives: a menu/banner/ending."""

        return self._state in SESSION_ACTION_STATES

    @property
    def action_for_state(self) -> SessionAction | None:
        return session_action_for_state(self._state)

    # -- keyboard ownership ------------------------------------------------
    def take_control(self) -> None:
        """Acquire the shared keyboard bus for ``session``.

        Raises :class:`touhou_rl.v430.actions.OwnerConflictError` -- propagated
        unchanged from the action layer -- if the actor, or anything else,
        currently holds the bus.  There is no "steal" path and no priority
        argument: switching owners is always an explicit release then acquire.
        """

        if self._compiler is not None:
            return
        self._compiler = ActionCompiler("session", ownership=self._ownership,
                                        hold_frames=C.HOLD_GAME_FRAMES)

    def release_control(self) -> None:
        """Give the keyboard back; safe to call when it is not held."""

        if self._compiler is None:
            return
        self._compiler.release()
        self._compiler = None

    def compile_session_action(self, move: Any = None, focus: Any = None,
                               shoot: Any = None, bomb: Any = None) -> Any:
        """Compile the session's request for the current state.

        With no explicit request the state's registered request is used.  The
        result is re-checked here (``controller_source='session'``,
        ``actor_loss_mask=0``) instead of trusting the compiler, so a future
        change to the action layer cannot quietly give a session press a policy
        gradient.
        """

        if self._compiler is None:
            raise SessionStateError(
                "the session does not hold the keyboard; call take_control() "
                "first (an implicit acquire would race the actor)")
        action = session_action_for_state(self._state)
        if action is None:
            raise SessionStateError(
                f"state {self._state!r} presses no keys; the session only drives "
                f"{list(SESSION_ACTION_STATES)}")
        request = (action.move if move is None else move,
                   action.focus if focus is None else focus,
                   action.shoot if shoot is None else shoot,
                   action.bomb if bomb is None else bomb)
        compiled = self._compiler.compile_request(*request)
        if compiled.controller_source != "session":
            raise SessionStateError(
                f"compiled decision claims controller_source="
                f"{compiled.controller_source!r}, expected 'session'")
        if compiled.actor_loss_mask != SESSION_ACTOR_LOSS_MASK:
            raise SessionStateError(
                f"compiled session decision carries actor_loss_mask="
                f"{compiled.actor_loss_mask}; a session key press has no policy "
                f"gradient ({SESSION_ACTOR_LOSS_MASK})")
        return compiled

    # -- span bookkeeping --------------------------------------------------
    def _open_new_span(self, *, state: str, start_frame_index: int,
                       start_captured_at_s: float, start_frame_digest: str) -> None:
        self._open = {
            "span_index": len(self._spans),
            "state": state,
            "key_owner": ("session" if state in SESSION_ACTION_STATES
                          else ("actor" if state == PLAYING else None)),
            "actor_loss_mask": (SESSION_ACTOR_LOSS_MASK
                                if state in SESSION_ACTION_STATES
                                else (C.ACTOR_LOSS_MASK_ON
                                      if state == PLAYING else None)),
            "start_frame_index": start_frame_index,
            "end_frame_index": start_frame_index,
            "start_sample_index": None,
            "end_sample_index": None,
            "start_captured_at_s": start_captured_at_s,
            "end_captured_at_s": start_captured_at_s,
            "elapsed_capture_s": 0.0,
            "sample_count": 0,
            "start_frame_digest": start_frame_digest,
            "end_frame_digest": start_frame_digest,
            "events": [],
            "reward_components": {},
            "session_decisions": 0,
        }

    def _attribute_sample(self, sample: PixelEvidence, sample_index: int) -> None:
        """Charge one sample's capture interval to the currently open span."""

        raw = self._open
        assert raw is not None
        if raw["start_sample_index"] is None:
            raw["start_sample_index"] = sample_index
        raw["end_sample_index"] = sample_index
        raw["sample_count"] += 1
        raw["elapsed_capture_s"] += sample.elapsed_capture_s
        raw["end_frame_index"] = sample.frame_index
        raw["end_captured_at_s"] = sample.captured_at_s
        raw["end_frame_digest"] = sample.frame_digest

    def _close_span(self) -> SessionSpan:
        assert self._open is not None
        raw = self._open
        span = SessionSpan(
            span_index=raw["span_index"],
            state=raw["state"],
            key_owner=raw["key_owner"],
            actor_loss_mask=raw["actor_loss_mask"],
            start_frame_index=raw["start_frame_index"],
            end_frame_index=raw["end_frame_index"],
            start_sample_index=raw["start_sample_index"],
            end_sample_index=raw["end_sample_index"],
            start_captured_at_s=raw["start_captured_at_s"],
            end_captured_at_s=raw["end_captured_at_s"],
            elapsed_capture_s=raw["elapsed_capture_s"],
            duration=_span_duration(raw["elapsed_capture_s"], self._hz),
            sample_count=raw["sample_count"],
            start_frame_digest=raw["start_frame_digest"],
            end_frame_digest=raw["end_frame_digest"],
            events=tuple(raw["events"]),
            reward_components=dict(raw["reward_components"]),
            session_decisions=raw["session_decisions"],
        )
        self._open = None
        self._spans.append(span)
        return span

    def _record_event(self, *events: str) -> None:
        if self._open is None:
            return
        for event in events:
            if event not in SESSION_EVENTS:
                raise SessionStateError(
                    f"event {event!r} not in {list(SESSION_EVENTS)}")
            if event not in self._open["events"]:
                self._open["events"].append(event)

    # -- public annotations ------------------------------------------------
    def attach_reward(self, name: Any, value: Any) -> None:
        """Stamp a reward component onto the currently open span.

        This is the interface that keeps a stage-clear (or a game-over) reward
        from being lost because the actor was not driving: the session does not
        interpret the number, it records it on the span that covers the time in
        which it was paid, and :meth:`SessionLedger.reward_rollup` surfaces it.
        A second value for the same name in the same span is refused (that would
        be an ambiguous duplicate, not a sum).
        """

        if self._open is None:
            raise SessionStateError(
                "no open session span to attach a reward to; call advance() with "
                "the first pixel evidence before attaching rewards")
        if not isinstance(name, str) or not name.strip():
            raise SessionError(
                f"reward component name must be a non-empty string, got {name!r}")
        number = _require_finite(value, f"reward component {name!r}")
        if name in self._open["reward_components"]:
            raise SessionStateError(
                f"span {self._open['span_index']} already carries reward "
                f"component {name!r}; attach it once and sum upstream")
        self._open["reward_components"][name] = number

    def note_event(self, event: Any) -> None:
        """Attach a caller-observed event from :data:`SESSION_EVENTS`."""

        if self._open is None:
            raise SessionStateError(
                "no open session span; call advance() before note_event()")
        self._record_event(event)

    # -- the inference -----------------------------------------------------
    def _life_pips_dropped(self, evidence: PixelEvidence) -> bool:
        """True when two *known* pixel readings show fewer life pips.

        ``None`` means "the pixel reader could not decide" and is never treated
        as zero: an unknown reading can neither create nor hide a hit.
        """

        if self._last_life_pips is None or evidence.visible_life_pips is None:
            return False
        return evidence.visible_life_pips < self._last_life_pips

    def _infer(self, previous_state: str, evidence: PixelEvidence
               ) -> tuple[str, tuple[str, ...]]:
        scene = evidence.scene
        if scene == SCENE_ENDING:
            return ENDING, ("ending_seen",)
        if scene == SCENE_OVER:
            return GAME_OVER, ("game_over_prompt_seen",)
        if scene == SCENE_MENU:
            return MENU, ("menu_seen",)
        if scene == SCENE_BANNER:
            return TRANSITION, ("stage_banner_seen",)
        if previous_state == PLAYING:
            if evidence.hit_flash_visible:
                return HIT_PENDING, ("hit_flash_seen",)
            if self._life_pips_dropped(evidence):
                return HIT_PENDING, ("life_pips_dropped",)
        if scene == SCENE_PLAY:
            if evidence.play_field_dark_ratio > DARK_PLAY_RATIO_MAX:
                return UNKNOWN, ("dark_play_field",)
            if evidence.player_sprite_visible:
                if previous_state in (HIT_PENDING, RESPAWN):
                    return PLAYING, ("player_visible_again",)
                return PLAYING, ("play_field_live",)
            if previous_state == HIT_PENDING:
                return RESPAWN, ("respawn_wait",)
            if previous_state == RESPAWN:
                return RESPAWN, ()
            return UNKNOWN, ("play_field_without_player",)
        # `unrecognised`: never upgraded, but a pending death beat stays pending.
        if previous_state in (HIT_PENDING, RESPAWN):
            return previous_state, ("scene_unrecognised",)
        return UNKNOWN, ("scene_unrecognised",)

    # -- restarts ----------------------------------------------------------
    def _new_restart(self, *, reason: str, source: str, previous_state: str,
                     new_state: str, evidence: PixelEvidence) -> RestartRecord:
        first = self._samples[0]
        return RestartRecord(
            restart_index=len(self._restarts) + 1,
            reason=reason,
            source=source,
            from_state=previous_state,
            to_state=new_state,
            frame_index=evidence.frame_index,
            frame_digest=evidence.frame_digest,
            frame_digest_scope=evidence.frame_digest_scope,
            captured_at_s=evidence.captured_at_s,
            session_elapsed_s=evidence.captured_at_s - first.captured_at_s,
        )

    def _auto_restart(self, previous_state: str, new_state: str,
                      evidence: PixelEvidence) -> RestartRecord | None:
        if previous_state == GAME_OVER and new_state in (MENU, PLAYING):
            reason = "game_over_restart"
        elif previous_state == ENDING and new_state in (MENU, PLAYING):
            reason = "ending_restart"
        else:
            return None
        record = self._new_restart(reason=reason, source="pixels",
                                   previous_state=previous_state,
                                   new_state=new_state, evidence=evidence)
        self._restarts.append(record)
        return record

    def record_manual_restart(self, evidence: Any, reason: Any = "manual_reset"
                              ) -> RestartRecord:
        """Record an operator-driven restart (never inferred from pixels).

        ``reason`` must come from :data:`MANUAL_RESTART_REASONS`, so an
        unattended run cannot claim a human reset it did not have, and a human
        reset cannot hide behind a pixel inference.
        """

        if self._finished:
            raise SessionStateError("the session is finished; it cannot be annotated")
        if reason not in MANUAL_RESTART_REASONS:
            raise SessionStateError(
                f"manual restart reason {reason!r} not in "
                f"{list(MANUAL_RESTART_REASONS)}; pixel-inferred reasons are "
                f"produced by advance(), not by the operator")
        sample = _coerce_evidence(evidence)
        if self._samples:
            last = self._samples[-1]
            if sample.frame_index <= last.frame_index:
                raise SessionClockError(
                    f"manual restart frame {sample.frame_index} is not after the "
                    f"last processed frame {last.frame_index}")
            self._samples.append(sample)
            if self._open is not None:
                self._attribute_sample(sample, len(self._samples) - 1)
        else:
            self._samples.append(sample)
        record = self._new_restart(reason=reason, source="operator",
                                   previous_state=self._state,
                                   new_state=self._state, evidence=sample)
        self._restarts.append(record)
        self._record_event("manual_restart")
        return record

    # -- the main entry point ---------------------------------------------
    def advance(self, evidence: Any) -> SessionStep:
        """Process one captured frame and return what changed.

        ``evidence`` is a :class:`PixelEvidence` or a mapping with exactly its
        fields.  Engine truth in a mapping raises
        :class:`SessionTruthError` *before* anything is parsed.
        """

        if self._finished:
            raise SessionStateError(
                "the session is finished; finish() already froze the ledger and "
                "a later sample would be outside it")
        sample = _coerce_evidence(evidence)

        if self._samples:
            previous_sample = self._samples[-1]
            if sample.frame_index <= previous_sample.frame_index:
                raise SessionClockError(
                    f"frame index did not advance "
                    f"({previous_sample.frame_index} -> {sample.frame_index}); the "
                    f"capture clock must move forward or two different games "
                    f"would share one frame number")
            expected = sample.captured_at_s - previous_sample.captured_at_s
            if not math.isclose(expected, sample.elapsed_capture_s,
                                rel_tol=0.0, abs_tol=1e-6):
                raise SessionClockError(
                    f"elapsed_capture_s {sample.elapsed_capture_s} disagrees with "
                    f"the capture clock difference {expected}; the interval is "
                    f"the only duration the session may use")

        sample_index = len(self._samples)
        previous_state = self._state
        self._samples.append(sample)

        if self._open is None:
            # Opening sample: its interval belongs to the state it establishes,
            # and the span starts one interval before it (the session clock
            # reading at which the first interval began).
            new_state, events = self._infer(previous_state, sample)
            self._state = new_state
            self._open_new_span(
                state=new_state,
                start_frame_index=sample.frame_index,
                start_captured_at_s=sample.captured_at_s - sample.elapsed_capture_s,
                start_frame_digest=sample.frame_digest)
            self._attribute_sample(sample, sample_index)
            self._record_event(*events)
            changed = new_state != previous_state
            restarts: list[RestartRecord] = []
        else:
            new_state, events = self._infer(previous_state, sample)
            changed = new_state != previous_state
            if not changed:
                self._attribute_sample(sample, sample_index)
                events = ()
                restarts = []
            else:
                # The interval belongs to the state that was in force during it,
                # i.e. the one that is ending here; the new state starts now.
                self._attribute_sample(sample, sample_index)
                self._close_span()
                restarts = []
                record = self._auto_restart(previous_state, new_state, sample)
                if record is not None:
                    restarts.append(record)
                self._state = new_state
                self._open_new_span(
                    state=new_state,
                    start_frame_index=sample.frame_index,
                    start_captured_at_s=sample.captured_at_s,
                    start_frame_digest=sample.frame_digest)
                # Events describe the state that has just been entered, so they
                # are stamped on the new span: a stage banner really is a
                # property of the ``transition`` span (and therefore travels
                # with the time it covers when the caller keeps the reward).
                self._record_event(*events)
                if record is not None:
                    self._record_event("restart_observed")

        self._last_life_pips = sample.visible_life_pips
        action = self._sync_key_control()
        return SessionStep(
            index=sample_index,
            previous_state=previous_state,
            state=self._state,
            changed=changed,
            events=events,
            evidence=sample,
            action=action,
            session_owns_keys=self.session_holds_keys,
            restarts=tuple(restarts),
        )

    def _sync_key_control(self) -> Any:
        """Take the keyboard in a session-driven state, release it otherwise."""

        if self._state in SESSION_ACTION_STATES:
            self.take_control()
            compiled = self.compile_session_action()
            if self._open is not None:
                self._open["session_decisions"] += 1
            return compiled
        self.release_control()
        return None

    # -- finishing ---------------------------------------------------------
    def finish(self) -> SessionLedger:
        """Close the open span and freeze the ledger (idempotent)."""

        if self._finished:
            return self.ledger
        if self._open is not None:
            raw = self._open
            if raw["sample_count"] == 0 and not (
                    raw["reward_components"] or raw["events"]
                    or raw["session_decisions"]):
                # The state changed on the very last captured frame and nothing
                # was charged to it: dropping an empty, reward-free span loses
                # nothing. It is counted, not hidden.
                self._open = None
                self._discarded += 1
            else:
                self._close_span()
        if self.session_holds_keys:
            self.release_control()
        self._finished = True
        return self.ledger

    @property
    def ledger(self) -> SessionLedger:
        spans = list(self._spans)
        if self._open is not None:
            raw = self._open
            spans.append(SessionSpan(
                span_index=raw["span_index"],
                state=raw["state"],
                key_owner=raw["key_owner"],
                actor_loss_mask=raw["actor_loss_mask"],
                start_frame_index=raw["start_frame_index"],
                end_frame_index=raw["end_frame_index"],
                start_sample_index=raw["start_sample_index"],
                end_sample_index=raw["end_sample_index"],
                start_captured_at_s=raw["start_captured_at_s"],
                end_captured_at_s=raw["end_captured_at_s"],
                elapsed_capture_s=raw["elapsed_capture_s"],
                duration=_span_duration(raw["elapsed_capture_s"], self._hz),
                sample_count=raw["sample_count"],
                start_frame_digest=raw["start_frame_digest"],
                end_frame_digest=raw["end_frame_digest"],
                events=tuple(raw["events"]),
                reward_components=dict(raw["reward_components"]),
                session_decisions=raw["session_decisions"],
            ))
        return SessionLedger(spans=spans, samples=self._samples,
                             restarts=self._restarts, hz=self._hz,
                             finished=self._finished,
                             discarded_empty_spans=self._discarded)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "v430-session/1",
            "contract": C.CONTRACT_VERSION,
            "state": self._state,
            "states": list(SESSION_STATES),
            "action_states": list(SESSION_ACTION_STATES),
            "waiting": self.waiting,
            "hz": self._hz,
            "duration_source": C.DURATION_SOURCE_EXTERNAL,
            "session_holds_keys": self.session_holds_keys,
            "ownership": self._ownership.as_dict(),
            "restart_count": self.restart_count,
            "restarts": [record.as_dict() for record in self._restarts],
            "ledger": self.ledger.as_dict(),
            "forbidden_truth_fields": list(SESSION_TRUTH_FIELDS),
        }


# ---------------------------------------------------------------------------
# source-level dependency audit (a second, independent "no truth" proof)
# ---------------------------------------------------------------------------
def _module_path() -> str:
    return __file__


def _imported_modules(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source, filename=path)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return sorted(set(modules))


def session_source_audit(path: str | None = None) -> dict[str, Any]:
    """Prove this file imports no engine/teacher/training module.

    The behavioural proof is the truth-injection sentinel; this is the static
    one.  It reads this module's own source, extracts every ``import`` and
    refuses (a) any module in :data:`SESSION_SOURCE_IMPORTS_FORBIDDEN` and (b) any
    module outside :data:`SESSION_SOURCE_IMPORTS_ALLOWED`.  A future edit that
    reaches for the engine to "just read the stage id" fails this check as well
    as the sentinel.
    """

    source_path = path or _module_path()
    modules = _imported_modules(source_path)
    forbidden_hits = [name for name in modules
                      if any(name == bad or name.startswith(bad + ".")
                             for bad in SESSION_SOURCE_IMPORTS_FORBIDDEN)]
    unknown = [name for name in modules
               if name not in SESSION_SOURCE_IMPORTS_ALLOWED]
    if forbidden_hits or unknown:
        raise SessionTruthError(
            f"{source_path} imports modules the session may not use: "
            f"forbidden={forbidden_hits}, outside the allow-list={unknown}; the "
            f"session reads pixels and its own key history only")
    with open(source_path, "rb") as handle:
        payload = handle.read()
    return {
        "path": source_path,
        "imports": modules,
        "forbidden_present": forbidden_hits,
        "allow_list": list(SESSION_SOURCE_IMPORTS_ALLOWED),
        "forbidden_list": list(SESSION_SOURCE_IMPORTS_FORBIDDEN),
        "sha256": hashlib.sha256(
            payload.replace(b"\r\n", b"\n")).hexdigest(),
        "digest_scope": "text/lf-normalized",
    }
