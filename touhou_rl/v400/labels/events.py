"""Event labels v1 (proxy) and v2 (real ``EVENTS`` bits + explicit proxies).

**v1** (``event_labels``, kept for the EXP-P2-001 dataset) was written for the
``build/rl_hazard`` binary, which had no ``EVENTS`` command: only the death
proxy was observable and five classes were ``event_source="unavailable"``.

**v2** (``event_labels_v2``) runs against ``build/rl_hazard2`` (SHA
``f172399f…``), whose ``EVENTS`` replies the engine's own
``ReplayManager::frameEventFlags`` word (``ReplayManager.hpp:161-172``).  Five
classes are therefore **real event bits**, not proxies:

    BOMB_STARTED           0x0001   -> "bomb"
    PLAYER_HIT             0x0002   -> "hit"
    PLAYER_DEATH_COMMITTED 0x0004   -> "death"
    PLAYER_DYING_STARTED   0x0200   -> "player_dying_started"
    EFFECT_SPAWNED         0x0400   -> "effect_spawned"

Three classes have no bit at all and stay state-derived, and are labelled
``event_source="proxy"`` with the exact criterion spelled out in the label:

* ``game_over`` — the ``STEP`` reply's ``game_over=`` field, which the harness
  fills from ``g_GameManager.showRetryMenu`` (``rlharness.cpp`` ``HandleStep``),
  raised only on the frame the chain asks to exit;
* ``extend``    — ``livesRemaining`` rises between consecutive frames;
* ``stage_clear`` — **unavailable**: ``GameManager.flags`` bit9 is never
  exported and no replay-event bit covers it.  It is ``None``, never ``False``.

Every event dict carries ``event_source`` ∈ ``{"EVENTS", "proxy",
"unavailable"}``.  Where a state-derived proxy also exists (``death``, ``bomb``,
``extend``) the label keeps it under ``proxy`` together with an ``agreement``
flag, so the same frame can be scored twice and the two channels compared
(EXP-P2-002 §4).  A missing input is always ``None`` plus a ``reason``; nothing
is defaulted to ``False`` meaning "did not happen", because a channel that was
never read is not evidence of absence.
"""

from __future__ import annotations

from typing import Any, Mapping

from touhou_rl.v400.labels.state import PLAYER_STATE_ALIVE, PLAYER_STATE_DYING

#: ``ReplayManager.hpp:161-172`` frames event bits (mirrored in
#: ``touhou_rl.v400.gym.protocol``; restated here so the label module does not
#: depend on the gym package for a pure-data constant).
REPLAY_FRAME_EVENT_BOMB_STARTED = 0x0001
REPLAY_FRAME_EVENT_PLAYER_HIT = 0x0002
REPLAY_FRAME_EVENT_PLAYER_DEATH_COMMITTED = 0x0004
REPLAY_FRAME_EVENT_PLAYER_DYING_STARTED = 0x0200
REPLAY_FRAME_EVENT_EFFECT_SPAWNED = 0x0400

#: Event class -> the real engine bit it is read from (v2).
EVENT_BITS: dict[str, int] = {
    "bomb": REPLAY_FRAME_EVENT_BOMB_STARTED,
    "hit": REPLAY_FRAME_EVENT_PLAYER_HIT,
    "death": REPLAY_FRAME_EVENT_PLAYER_DEATH_COMMITTED,
    "player_dying_started": REPLAY_FRAME_EVENT_PLAYER_DYING_STARTED,
    "effect_spawned": REPLAY_FRAME_EVENT_EFFECT_SPAWNED,
}

#: The three classes that have no engine bit and stay state-derived.
PROXY_EVENTS: tuple[str, ...] = ("game_over", "extend")

#: The label vocabulary, in report order.
EVENT_CLASSES: tuple[str, ...] = (
    "hit",
    "player_dying_started",
    "effect_spawned",
    "death",
    "bomb",
    "game_over",
    "extend",
    "stage_clear",
)

#: Legal ``event_source`` values.
EVENT_SOURCES: tuple[str, ...] = ("EVENTS", "proxy", "unavailable")

#: Criterion text for the state-derived half of every event that has one.
PROXY_CRITERIA: dict[str, str] = {
    "death": (
        "proxy = (deaths(k) > deaths(k-1)) OR (playerState(k)==ALIVE and "
        "playerState(k+1)==DYING).  The second clause is EXP-P1-004 §1's edge; "
        "the first is globals.deaths (STATE f6) increasing, which also catches a "
        "death whose DYING frame was not sampled"
    ),
    "bomb": (
        "proxy = bombsRemaining(k) < bombsRemaining(k-1) on a frame with no death "
        "commit / DYING edge.  Player::UpdateDeathAndRespawn also moves the bomb "
        "stock on a lost-life frame, so a drop on a death frame is reported as "
        "ambiguous (ambiguous_with_death=True) rather than counted as a bomb"
    ),
    "extend": (
        "proxy = livesRemaining(k) > livesRemaining(k-1).  Lives also rise after "
        "a continue, which the harness never reaches, so on this data the rise is "
        "the extend path"
    ),
    "game_over": (
        "proxy = the STEP reply's game_over= field (g_GameManager.showRetryMenu), "
        "which the harness raises only on the frame the chain asks to exit; "
        "lives_remaining==0 is recorded alongside as a cross-check, not as the "
        "criterion"
    ),
}

#: Why each event type cannot be read from the engine in this build.
UNAVAILABLE_REASONS: dict[str, str] = {
    "hit": (
        "REPLAY_FRAME_EVENT_PLAYER_HIT (0x2) lives in ReplayManager::"
        "frameEventFlags, which rlharness never exports (no STATE field, not in "
        "HASH/DEBUG/PLAYER); no proxy is claimed here"
    ),
    "death": (
        "REPLAY_FRAME_EVENT_PLAYER_DEATH_COMMITTED (0x4) is not exported; the "
        "ALIVE->DYING proxy below is used instead"
    ),
    "bomb": (
        "bombsUsed is not exported; a bombsRemaining drop is also produced by "
        "death handling, so no bomb proxy is claimed in this slice"
    ),
    "stage_clear": (
        "GameManager.flags bit9 (stageClearSequenceActive) is not exported"
    ),
    "game_over": (
        "showRetryMenu is not exported; only the harness STEP reply's game_over "
        "flag is observable, and it is not part of the state export"
    ),
    "extend": (
        "livesRemaining rises on extend but also on other paths (continue); no "
        "extend proxy is claimed in this slice"
    ),
}


def event_labels(
    player_state: int | None,
    next_player_state: int | None,
) -> dict[str, Any]:
    """Per-frame event labels for one frame and its recorded successor.

    ``next_player_state=None`` means there is no successor frame (run boundary):
    the death proxy is then ``None`` with ``reason="no_next_frame"`` rather than
    a guessed ``False``.
    """

    if player_state is None:
        death_value: bool | None = None
        death_reason: str = "player_state_missing"
    elif next_player_state is None:
        death_value = None
        death_reason = "no_next_frame"
    else:
        death_value = bool(
            int(player_state) == PLAYER_STATE_ALIVE
            and int(next_player_state) == PLAYER_STATE_DYING
        )
        death_reason = ""

    labels: dict[str, Any] = {
        "death": {
            "value": death_value,
            "event_source": "proxy",
            "criterion": (
                "playerState(k)==ALIVE(0) and playerState(k+1)==DYING(2); "
                "a strict subset of frameEventFlags & 0x4, not the event bit"
            ),
            "reason": death_reason,
            "player_state": player_state,
            "player_state_next": next_player_state,
        }
    }
    for name in ("hit", "bomb", "stage_clear", "game_over", "extend"):
        labels[name] = {
            "value": None,
            "event_source": "unavailable",
            "reason": UNAVAILABLE_REASONS[name],
        }
    labels["note"] = (
        "EVENTS/frameEventFlags are not observable in this rl_hazard build; "
        "only the death proxy is a value, and it is marked event_source=proxy"
    )
    return labels


# ---------------------------------------------------------------------------
# v2: real EVENTS bits + explicit state-derived proxies
# ---------------------------------------------------------------------------
def decode_event_flags(flags: int | None) -> dict[str, bool] | None:
    """Decode the five known event bits, or ``None`` when the word is absent."""

    if flags is None:
        return None
    word = int(flags) & 0xFFFF
    return {name: bool(word & bit) for name, bit in EVENT_BITS.items()}


def _bit(flags: int | None, name: str) -> bool | None:
    """One bit of the event word, or ``None`` when the channel returned nothing."""

    if flags is None:
        return None
    return bool(int(flags) & EVENT_BITS[name])


def _number(state: Mapping[str, Any] | None, key: str) -> float | None:
    """A float field of a decoded STATE mapping, or ``None`` when absent."""

    if not state:
        return None
    value = state.get(key)
    if value is None:
        return None
    return float(value)


def _alive_to_dying(
    player_state: int | None, next_player_state: int | None
) -> tuple[bool | None, str]:
    """EXP-P1-004 §1's death edge, or ``None`` with the reason it is unknown."""

    if player_state is None:
        return None, "player_state_missing"
    if next_player_state is None:
        return None, "no_next_frame"
    return (
        bool(
            int(player_state) == PLAYER_STATE_ALIVE
            and int(next_player_state) == PLAYER_STATE_DYING
        ),
        "",
    )


def _rose(current: float | None, previous: float | None) -> tuple[bool | None, str]:
    """``current > previous`` as a proxy edge, or ``None`` when either is absent."""

    if current is None or previous is None:
        return None, "no_previous_frame" if previous is None else "state_missing"
    return bool(current > previous), ""


def _fell(current: float | None, previous: float | None) -> tuple[bool | None, str]:
    if current is None or previous is None:
        return None, "no_previous_frame" if previous is None else "state_missing"
    return bool(current < previous), ""


def _agree(primary: bool | None, proxy: bool | None) -> bool | None:
    if primary is None or proxy is None:
        return None
    return bool(primary) == bool(proxy)


def event_labels_v2(
    *,
    frame_event_flags: int | None,
    player_state: int | None = None,
    next_player_state: int | None = None,
    state: Mapping[str, Any] | None = None,
    prev_state: Mapping[str, Any] | None = None,
    step_game_over: bool | None = None,
    events_error: str | None = None,
) -> dict[str, Any]:
    """Per-frame v2 event labels.

    ``frame_event_flags`` is the ``EVENTS`` word for *this* frame (after the
    ``STEP`` that advanced it) or ``None`` when the channel is unavailable
    (``events_error`` then says why).  ``state``/``prev_state`` are decoded
    ``STATE`` mappings for this frame and its predecessor *in the same run*;
    ``prev_state=None`` at a run's first frame makes every change-based proxy
    ``None`` (``reason="no_previous_frame"``) instead of comparing across runs.

    The primary ``value`` of ``hit``/``player_dying_started``/``effect_spawned``/
    ``death``/``bomb`` is the engine bit (``event_source="EVENTS"``).  When the
    channel is unavailable, ``death``/``bomb`` fall back to their state-derived
    proxy and say so; the other three become ``None`` with
    ``event_source="unavailable"``.  ``game_over``/``extend`` are always proxies
    (no bit exists), and ``stage_clear`` is always ``None``.
    """

    flags = None if frame_event_flags is None else int(frame_event_flags) & 0xFFFF
    available = flags is not None
    bits = decode_event_flags(flags)

    deaths = _number(state, "deaths")
    deaths_prev = _number(prev_state, "deaths")
    lives = _number(state, "lives_remaining")
    lives_prev = _number(prev_state, "lives_remaining")
    bombs = _number(state, "bombs_remaining")
    bombs_prev = _number(prev_state, "bombs_remaining")

    edge, edge_reason = _alive_to_dying(player_state, next_player_state)
    deaths_rose, deaths_rose_reason = _rose(deaths, deaths_prev)
    if deaths_rose is None:
        proxy_death: bool | None = None
        proxy_death_reason = deaths_rose_reason
    else:
        # A death is the OR of the two observables; the edge alone can be None.
        proxy_death = bool(deaths_rose or (edge is True))
        proxy_death_reason = ""

    def with_bit(name: str) -> dict[str, Any]:
        value = _bit(flags, name)
        entry: dict[str, Any] = {
            "value": value,
            "event_source": "EVENTS" if available else "unavailable",
            "criterion": f"EVENTS reply frameEventFlags & 0x{EVENT_BITS[name]:04x}",
            "reason": "" if available else (events_error or "events_channel_missing"),
        }
        if available:
            entry["frame_event_flags"] = flags
        return entry

    labels: dict[str, Any] = {
        "hit": with_bit("hit"),
        "player_dying_started": with_bit("player_dying_started"),
        "effect_spawned": with_bit("effect_spawned"),
    }

    # -- death: engine bit when present, else the state proxy ----------------
    death_value = _bit(flags, "death")
    death_source = "EVENTS" if available else "proxy"
    if not available:
        death_value = proxy_death
    labels["death"] = {
        "value": death_value,
        "event_source": death_source,
        "frame_event_flags": flags,
        "criterion": (
            f"EVENTS reply frameEventFlags & 0x{EVENT_BITS['death']:04x} "
            "(PLAYER_DEATH_COMMITTED)"
            if available
            else PROXY_CRITERIA["death"]
        ),
        "reason": "" if available else (proxy_death_reason or "events_channel_missing"),
        "proxy": {
            "value": proxy_death,
            "criterion": PROXY_CRITERIA["death"],
            "reason": proxy_death_reason,
            "components": {
                "deaths_increase": deaths_rose,
                "alive_to_dying": edge,
            },
        },
        "agreement": _agree(_bit(flags, "death"), proxy_death),
        "deaths": deaths,
        "deaths_prev": deaths_prev,
        "player_state": player_state,
        "player_state_next": next_player_state,
        "proxy_edge_reason": edge_reason,
    }

    # -- bomb: engine bit when present, else stock drop without a death ------
    bombs_fell, bombs_fell_reason = _fell(bombs, bombs_prev)
    death_same_frame = (
        None
        if (edge is None and deaths_rose is None)
        else bool(edge is True or deaths_rose is True)
    )
    if bombs_fell is None:
        proxy_bomb: bool | None = None
        proxy_bomb_reason = bombs_fell_reason
    elif bombs_fell and death_same_frame is True:
        proxy_bomb = None
        proxy_bomb_reason = "ambiguous_with_death"
    else:
        proxy_bomb = bool(bombs_fell)
        proxy_bomb_reason = ""
    bomb_value = _bit(flags, "bomb")
    if not available:
        bomb_value = proxy_bomb
    labels["bomb"] = {
        "value": bomb_value,
        "event_source": "EVENTS" if available else "proxy",
        "frame_event_flags": flags,
        "criterion": (
            f"EVENTS reply frameEventFlags & 0x{EVENT_BITS['bomb']:04x} "
            "(BOMB_STARTED)"
            if available
            else PROXY_CRITERIA["bomb"]
        ),
        "reason": "" if available else (proxy_bomb_reason or "events_channel_missing"),
        "proxy": {
            "value": proxy_bomb,
            "criterion": PROXY_CRITERIA["bomb"],
            "reason": proxy_bomb_reason,
            "components": {
                "bombs_decrease": bombs_fell,
                "death_same_frame": death_same_frame,
            },
        },
        "agreement": _agree(_bit(flags, "bomb"), proxy_bomb),
        "bombs": bombs,
        "bombs_prev": bombs_prev,
    }

    # -- game_over: no bit exists; the STEP reply carries the engine flag ----
    if step_game_over is None:
        game_over_value: bool | None = None
        game_over_reason = "step_reply_unavailable"
    else:
        game_over_value = bool(step_game_over)
        game_over_reason = ""
    labels["game_over"] = {
        "value": game_over_value,
        "event_source": "proxy",
        "criterion": PROXY_CRITERIA["game_over"],
        "reason": game_over_reason,
        "step_game_over": step_game_over,
        "lives": lives,
        "lives_zero": None if lives is None else bool(lives <= 0.0),
    }

    # -- extend: lives rise -------------------------------------------------
    lives_rose, lives_rose_reason = _rose(lives, lives_prev)
    labels["extend"] = {
        "value": lives_rose,
        "event_source": "proxy",
        "criterion": PROXY_CRITERIA["extend"],
        "reason": lives_rose_reason,
        "proxy": {
            "value": lives_rose,
            "criterion": PROXY_CRITERIA["extend"],
            "reason": lives_rose_reason,
            "components": {"lives_increase": lives_rose},
        },
        "lives": lives,
        "lives_prev": lives_prev,
    }

    labels["stage_clear"] = {
        "value": None,
        "event_source": "unavailable",
        "reason": UNAVAILABLE_REASONS["stage_clear"],
    }

    labels["channel"] = {
        "name": "EVENTS",
        "available": available,
        "frame_event_flags": flags,
        "bits": bits,
        "error": None if available else (events_error or "events_channel_missing"),
    }
    labels["note"] = (
        "v2: hit/player_dying_started/effect_spawned/death/bomb are engine "
        "EVENTS bits; game_over/extend are state proxies; stage_clear is "
        "unavailable (None).  proxy/agreement are kept per frame for the "
        "EVENTS-vs-proxy consistency report"
    )
    return labels
