"""Privileged snapshot adapter with an explicit same-frame proof (D1).

Why the old chain is not reusable as-is
---------------------------------------
The v3.6.x trace marks a decision ``complete`` when eight field groups were
readable.  That says nothing about whether the bullet pool and the player state
belong to the *same* instant: in 12,000 recorded decisions the pool size and the
engine's ``activeBulletCount`` agreed only 26.8% of the time, with differences up
to ±416 (see ``reports/接口与基线核查.md`` §3).  A planner that asks "is this
route safe from *now*" cannot be built on that.

This adapter therefore makes one thing mandatory: the two state reads that
bracket the bullet read must carry the same ``gameplay_frame``.  If they never
agree within the attempt budget, the bullets are **dropped** (``hazards=None``)
and the reason is recorded, because a pool from an unknown frame is worse than
no pool at all.

It also preserves per-axis bullet half extents.  ``env.py`` collapses them to
``max(half_x, half_y)`` for the old pilot's circle model; doing that here would
silently turn every bullet into a square and break the AABB clearance that
``geometry.py`` computes exactly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .contracts import (
    OBSERVATION_SCHEMA,
    Completeness,
    ContractError,
    EscapeObservation,
    HazardObs,
    MotionModel,
    PlayerObs,
    Source,
)

#: ``Player::playerState`` values (Player.hpp): 0 alive, 1 spawning, 2 dying, 3 invulnerable.
PLAYER_STATE_ALIVE = 0
PLAYER_STATE_INVULNERABLE = 3

#: Reasons this module can record.  Kept as constants so traces can be grouped.
REASON_BULLET_READ_ERROR = "bullets: read failed"
REASON_NOT_STABLE = "frame: state and pool never agreed on gameplay_frame"
REASON_FRAME_REGRESSION = "frame: gameplay_frame went backwards"

#: A capability gap, not a missing read: the current reader exposes no
#: ``framerateMultiplier``, so "bullet time" cannot be ruled out.  FR-13 requires
#: this to be visible rather than assumed away.
COVERAGE_BULLET_TIME_UNVERIFIED = "framerateMultiplier: not observable, linear forecast assumes 1.0"
COVERAGE_BULLET_TIME_ACTIVE = "framerateMultiplier != 1.0: bullet-time frames are unsupported"


class SnapshotReader(Protocol):
    """The read-only surface :func:`read_synced` needs (see ``state.GameStateReader``)."""

    def read_state(self, with_bullets: bool = False, with_title: bool = False) -> Any: ...

    def read_bullets(self) -> Any: ...


@dataclass(frozen=True)
class SyncedFrame:
    """One accepted snapshot plus how it was obtained."""

    frame_id: int
    player: PlayerObs
    hazards: tuple[HazardObs, ...] | None
    pool_count: int | None
    engine_counter: int | None
    attempts: int
    stable: bool
    reasons: tuple[str, ...]
    unsupported: tuple[str, ...]
    bombs: int | None = None
    #: The raw ``GameState`` and bullet pool the snapshot was decoded from.
    #: v3.7.1 needs them so both experiment arms consume *one* read: the
    #: baseline pilot wants ``(state, bullet pool)`` and the escape controller
    #: wants the contract observation, and two separate reads would let them see
    #: different frames (the v3.7.0 comparison did exactly that).
    state: Any = None
    field: Any = None


def _player(state: Any) -> tuple[PlayerObs, tuple[str, ...]]:
    reasons: list[str] = []
    state_code = int(getattr(state, "player_state", -1))
    if state_code not in (0, 1, 2, 3):
        reasons.append(f"player: unknown player_state {state_code}")
    alive = state_code in (PLAYER_STATE_ALIVE, PLAYER_STATE_INVULNERABLE)
    invulnerable = state_code == PLAYER_STATE_INVULNERABLE
    half_x = float(getattr(state, "hurtbox_half_x", 0.0))
    half_y = float(getattr(state, "hurtbox_half_y", 0.0))
    # The hurtbox is square in TH08; a zero read means the field was torn down.
    half = max(half_x, half_y, 0.0)
    if half <= 0.0:
        reasons.append("player: hurtbox half extent unreadable")
        half = 0.825  # the measured value, recorded so the plan is not silently wrong
    player = PlayerObs(
        x=float(getattr(state, "player_x", 0.0)),
        y=float(getattr(state, "player_y", 0.0)),
        focus=int(getattr(state, "focus_mode", 0)) & 1,
        hurtbox_half=half,
        alive=alive,
        invulnerable=invulnerable,
        speed_model="border-team-lunatic-1.00d",
    )
    return player, tuple(reasons)


def _hazards(field: Any, frame_id: int, supported_frames: int) -> tuple[tuple[HazardObs, ...], int]:
    mask = np.asarray(field.active_mask())
    positions = np.asarray(field.position)[mask]
    velocities = np.asarray(field.velocity)[mask]
    halves = np.asarray(field.half_size)[mask]
    count = int(positions.shape[0])
    hazards = tuple(
        HazardObs(
            position=(float(positions[index, 0]), float(positions[index, 1])),
            velocity=(float(velocities[index, 0]), float(velocities[index, 1])),
            half=(float(halves[index, 0]), float(halves[index, 1])),
            motion_model=MotionModel.LINEAR,
            confidence=1.0,
            valid_until=frame_id + max(0, supported_frames),
        )
        for index in range(count)
    )
    return hazards, count


def read_synced(
    reader: SnapshotReader,
    *,
    last_frame: int | None = None,
    attempts: int = 4,
    supported_frames: int = 180,
    framerate_multiplier: float | None = None,
    sleep: float = 0.0,
) -> SyncedFrame:
    """Read a player+pool snapshot that is *proven* to be from one frame.

    ``last_frame`` enables the monotonicity check: a snapshot whose
    ``gameplay_frame`` went backwards means the game restarted or the read
    straddled two sessions, and it is reported as unreadable instead of being
    used as "now".
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    reasons: list[str] = []
    unsupported: list[str] = []
    if framerate_multiplier is None:
        unsupported.append(COVERAGE_BULLET_TIME_UNVERIFIED)
    elif abs(framerate_multiplier - 1.0) > 1e-6:
        unsupported.append(COVERAGE_BULLET_TIME_ACTIVE)

    state: Any = None
    before: Any = None
    hazards: tuple[HazardObs, ...] | None = None
    pool_count: int | None = None
    engine_counter: int | None = None
    stable = False
    used = 0

    for attempt in range(1, attempts + 1):
        used = attempt
        before = reader.read_state(with_bullets=False, with_title=False)
        try:
            field = reader.read_bullets()
            read_error = False
        except OSError:
            field = None
            read_error = True
        after = reader.read_state(with_bullets=False, with_title=False)
        frame_before = int(getattr(before, "gameplay_frame", -1))
        frame_after = int(getattr(after, "gameplay_frame", -1))
        state = after
        if frame_after >= 0 and last_frame is not None and frame_after < int(last_frame):
            reasons = [REASON_FRAME_REGRESSION]
            hazards, pool_count, engine_counter = None, None, None
            stable = False
            break
        if frame_before != frame_after or frame_after < 0:
            # The pool was read while the frame advanced: try again.
            if sleep:
                time.sleep(sleep)
            continue
        stable = True
        engine_counter = int(getattr(after, "active_bullets", -1))
        if engine_counter < 0:
            engine_counter = None
        if read_error or field is None:
            reasons = [REASON_BULLET_READ_ERROR]
            hazards, pool_count = None, None
        else:
            hazards, pool_count = _hazards(field, frame_after, supported_frames)
        break

    player, player_reasons = _player(state if state is not None else before)
    reasons = list(reasons) + list(player_reasons)
    if not stable:
        hazards = None
        if REASON_NOT_STABLE not in reasons:
            reasons.append(REASON_NOT_STABLE)
        pool_count = None
    return SyncedFrame(
        frame_id=int(getattr(state, "gameplay_frame", -1)) if state is not None else -1,
        player=player,
        hazards=hazards,
        pool_count=pool_count,
        engine_counter=engine_counter,
        attempts=used,
        stable=stable,
        reasons=tuple(dict.fromkeys(reasons)),
        unsupported=tuple(unsupported),
        bombs=int(getattr(state, "bombs", 0)) if state is not None else None,
        state=state,
        field=field,
    )


def to_observation(
    frame: SyncedFrame,
    *,
    episode_id: str,
    decision_id: int,
    previous_action: tuple[int, int, int, int] | None = None,
    frameskip: int = 2,
    action_latency_frames: int = 1,
    source: Source = Source.PRIVILEGED,
) -> EscapeObservation:
    """Convert an accepted frame into the frozen record (and validate it)."""

    if not frame.stable:
        completeness = Completeness.UNREADABLE
    elif frame.hazards is None:
        completeness = Completeness.PARTIAL
    else:
        completeness = Completeness.COMPLETE
    observation = EscapeObservation(
        schema=OBSERVATION_SCHEMA,
        episode_id=episode_id,
        decision_id=int(decision_id),
        frame_id=int(frame.frame_id),
        source=source,
        player=frame.player,
        frameskip=int(frameskip),
        observed_frame_age=0,
        action_latency_frames=int(action_latency_frames),
        completeness=completeness,
        coordinate_contract=_coordinate_contract(),
        hazards=frame.hazards,
        previous_action=previous_action,
        missing_reasons=frame.reasons,
        unsupported_hazards=frame.unsupported,
        bombs=int(frame.bombs or 0),
        bullet_pool_count=frame.pool_count,
    )
    _assert_valid(observation)
    return observation


def _coordinate_contract() -> dict[str, Any]:
    # Imported lazily so this module stays importable from tools that only need
    # the reader protocol.
    from .contracts import COORDINATE_CONTRACT

    return dict(COORDINATE_CONTRACT)


def _assert_valid(observation: EscapeObservation) -> None:
    from .contracts import to_json_dict, validate_document

    issues = validate_document(to_json_dict(observation))
    if issues:
        raise ContractError("; ".join(issues))
