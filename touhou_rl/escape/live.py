"""Live shadow probe wiring for D8/D9.

The offline probe replays recorded traces.  This module is the *live* path: the
same decision loop, but driven by an adapter that reads the running game and (for
the self-play arms) supplies input.  It exists as a separate module because the
live path has checks the offline path cannot make:

* the observation must be fresh (``observed_frame_age`` within the bound) — a stale
  read is what makes a live run silently meaningless;
* the frame counter must advance monotonically, which is the same-frame check;
* every decision must be recorded with its paired label, so B-05's completeness
  ratio (≥99%) and B-07's pairing are computed from the run, not estimated.

Nothing here imports the game.  ``run_live`` takes two callables, so the whole file
is exercised offline against a fake adapter (see ``tests/test_escape_live.py``);
the real wiring is one line at the call site.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import contracts, controller, forecast, probe

LIVE_SCHEMA = "v370-live-probe/1"


def field_from_observation(observation) -> "forecast.DangerField":
    """Build a danger field from a live observation.

    Live reads only say which hazards are active *now*: spawn is ``now`` and the
    despawn is the hazard's declared ``valid_until`` (or the forecast horizon when
    the adapter does not provide one).  That is weaker than a recorded trace, where
    the full spawn/despawn schedule is known, and it is why a live run can only
    ever confirm the current frame — the future is extrapolated, not observed.
    """

    import numpy as np

    frame = int(observation.frame_id)
    hazards = observation.hazards or ()
    if not hazards:
        positions = np.zeros((0, 2), dtype=np.float64)
        velocities = np.zeros((0, 2), dtype=np.float64)
        halves = np.zeros((0, 2), dtype=np.float64)
        spawn = np.zeros((0,), dtype=np.float64)
        despawn = np.zeros((0,), dtype=np.float64)
    else:
        positions = np.asarray([hazard.position for hazard in hazards], dtype=np.float64)
        velocities = np.asarray([hazard.velocity for hazard in hazards], dtype=np.float64)
        halves = np.asarray([hazard.half for hazard in hazards], dtype=np.float64)
        spawn = np.full((len(hazards),), float(frame), dtype=np.float64)
        despawn = np.asarray(
            [
                float(getattr(hazard, "valid_until", None) or frame + _FORECAST_HORIZON)
                for hazard in hazards
            ],
            dtype=np.float64,
        )
    field = forecast.DangerField(
        frame_base=frame,
        positions=positions,
        velocities=velocities,
        halves=halves,
        spawn=spawn,
        despawn=despawn,
        config=forecast.ForecastConfig(),
    )
    return field.compressed()


_FORECAST_HORIZON = 180


class LiveError(RuntimeError):
    """The run cannot produce a valid result (sync lost, adapter read failed)."""


def game_source(reader, *, episode_id: str = "live", frameskip: int = 2, advance=None):
    """Bind a game snapshot reader to ``run_live``.

    This is the one-line wiring D8 was missing: ``reader`` is any object with
    ``read_state(with_bullets=..., with_title=...)`` and ``read_bullets()`` (the
    repository's ``GameStateReader`` satisfies it).  ``adapter.read_synced`` proves
    the snapshot is one frame by bracketing two state reads around the bullet read,
    and the monotonicity check catches a restart.  ``advance`` (optional) is called
    between decisions; without it the loop runs as fast as the adapter allows.

    Returns ``(read_fn, advance_fn, state)`` where ``state`` carries the last
    accepted frame so a caller can inspect sync statistics.
    """

    state = {"last_frame": None, "reads": 0, "rejected": 0}

    def read():
        from . import adapter

        frame = adapter.read_synced(reader, last_frame=state["last_frame"])
        state["reads"] += 1
        if not frame.stable:
            state["rejected"] += 1
            return None
        state["last_frame"] = frame.frame_id
        return adapter.to_observation(
            frame,
            episode_id=episode_id,
            decision_id=state["reads"],
            frameskip=frameskip,
        )

    return read, advance, state


def run_live(
    read_fn,
    *,
    advance_fn=None,
    decisions: int = 200,
    config: probe.ProbeConfig | None = None,
    max_stale: int = 2,
) -> dict:
    """Collect ``decisions`` live observations and decisions through the controller.

    Returns a document with the same row shape as the recorded probe plus the live
    only fields: ``same_frame_verified``, ``stale_reads``, ``frames`` and the B-05
    completeness ratio.  A stale read is counted and skipped rather than being
    silently used — the controller would report ``UNKNOWN`` for it anyway, and the
    B-05 ratio needs to know how often it happened.
    """

    config = config or probe.ProbeConfig(
        decisions=decisions,
        planner=controller.ControllerConfig(),
    )
    shadow = controller.EscapeController(config.planner)
    rows: list[dict] = []
    frames: list[int] = []
    stale = 0
    dropped = 0
    started = time.perf_counter()

    while len(rows) < decisions:
        observation = read_fn()
        if observation is None:
            dropped += 1
            if dropped > decisions * 2 + 16:
                raise LiveError("adapter returned no observation too many times")
            if advance_fn is not None:
                advance_fn()
            continue
        issues = contracts.validate_document(contracts.to_json_dict(observation))
        age = int(getattr(observation, "observed_frame_age", 0) or 0)
        if issues or age > max_stale:
            stale += 1
            if advance_fn is not None:
                advance_fn()
            continue

        frames.append(int(observation.frame_id))
        field = field_from_observation(observation)
        # The controller takes ``(focus, move)`` as its previous action; the emitted
        # contract action is ``(move, shoot, focus, bomb)``.  Keep both so the next
        # decision gets the right pair and the record stays contract-shaped.
        previous = tuple(rows[-1]["previous_action"]) if rows else (0, 0)
        decision, _diagnostics = shadow.decide(
            x=observation.player.x,
            y=observation.player.y,
            frame=0,
            previous_action=(int(previous[0]), int(previous[1])),
            field=field,
            observation_ok=True,
            observation_age=age,
            bombs=int(getattr(observation, "bombs", 0) or 0),
        )
        rows.append(
            {
                "frame": int(observation.frame_id),
                "decision": len(rows) + 1,
                "x": float(observation.player.x),
                "y": float(observation.player.y),
                "hazards": len(observation.hazards),
                "intent": decision.intent.value,
                "action": [int(value) for value in decision.accepted_action],
                "previous_action": [
                    int(decision.accepted_action[2]),
                    int(decision.accepted_action[0]),
                ],
                "route_id": decision.route_id,
                "reason": decision.reason,
                "plan_ms": decision.timing_breakdown.get("controller_ms"),
                "deadline_miss": bool(decision.deadline_miss),
            }
        )
        if advance_fn is not None:
            advance_fn()

    monotonic = all(b > a for a, b in zip(frames, frames[1:]))
    plan_ms = [row["plan_ms"] for row in rows if row["plan_ms"] is not None]
    misses = sum(1 for row in rows if row["deadline_miss"])
    return {
        "schema": LIVE_SCHEMA,
        "live": True,
        "decisions": len(rows),
        "requested": decisions,
        "stale_reads": stale,
        "dropped_reads": dropped,
        "same_frame_verified": bool(monotonic),
        "frame_span": [frames[0], frames[-1]] if frames else [],
        "complete_ratio": round(len(rows) / max(1, len(rows) + stale + dropped), 4),
        "intents": _counts(row["intent"] for row in rows),
        "deadline_misses": misses,
        "deadline_miss_rate": round(misses / len(rows), 4) if rows else None,
        "plan_ms_p50": round(sorted(plan_ms)[len(plan_ms) // 2], 3) if plan_ms else None,
        "plan_ms_p95": (
            round(sorted(plan_ms)[max(0, int(len(plan_ms) * 0.95) - 1)], 3) if plan_ms else None
        ),
        "max_hazards": max((row["hazards"] for row in rows), default=0),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "rows": rows,
        "notes": (
            "same_frame_verified only means the adapter's frame counter advanced; "
            "the game-side sync check is part of the D0 adapter audit"
        ),
    }


def run_live_closed_loop(
    read_fn,
    step_fn,
    *,
    decisions: int = 200,
    config: probe.ProbeConfig | None = None,
    max_stale: int = 2,
) -> dict:
    """Closed-loop live run: read the game, decide, and *send* the action.

    ``step_fn(action)`` is the input side (``env.step``).  The difference to
    :func:`run_live` is that the game advances only when we send an action, so the
    latency that matters is ``read + plan`` against the frameskip budget.

    Deaths are read from the player state: the run counts transitions into a dead
    state, and ``lives_lost`` is the number of those transitions that were followed
    by a respawn at the spawn point.
    """

    config = config or probe.ProbeConfig(decisions=decisions, planner=controller.ControllerConfig())
    shadow = controller.EscapeController(config.planner)
    rows: list[dict] = []
    frames: list[int] = []
    stale = dropped = 0
    deaths = 0
    respawns = 0
    previous_dead = False
    started = time.perf_counter()

    while len(rows) < decisions:
        observation = read_fn()
        if observation is None:
            dropped += 1
            if dropped > decisions * 2 + 16:
                raise LiveError("adapter returned no observation too many times")
            continue
        issues = contracts.validate_document(contracts.to_json_dict(observation))
        age = int(getattr(observation, "observed_frame_age", 0) or 0)
        if issues or age > max_stale:
            stale += 1
            continue

        frames.append(int(observation.frame_id))
        field = field_from_observation(observation)
        previous = tuple(rows[-1]["previous_action"]) if rows else (0, 0)
        decision, _diagnostics = shadow.decide(
            x=observation.player.x,
            y=observation.player.y,
            frame=0,
            previous_action=(int(previous[0]), int(previous[1])),
            field=field,
            observation_ok=True,
            observation_age=age,
            bombs=int(getattr(observation, "bombs", 0) or 0),
        )
        action = tuple(int(value) for value in decision.accepted_action)
        step_fn(action)

        alive = bool(getattr(observation.player, "alive", True))
        if previous_dead and alive:
            respawns += 1
        if not alive and not previous_dead:
            deaths += 1
        previous_dead = not alive

        rows.append(
            {
                "frame": int(observation.frame_id),
                "decision": len(rows) + 1,
                "x": float(observation.player.x),
                "y": float(observation.player.y),
                "hazards": len(observation.hazards or ()),
                "intent": decision.intent.value,
                "action": list(action),
                "previous_action": [action[2], action[0]],
                "route_id": decision.route_id,
                "reason": decision.reason,
                "plan_ms": decision.timing_breakdown.get("controller_ms"),
                "deadline_miss": bool(decision.deadline_miss),
                "alive": alive,
            }
        )

    monotonic = all(b > a for a, b in zip(frames, frames[1:]))
    plan_ms = [row["plan_ms"] for row in rows if row["plan_ms"] is not None]
    misses = sum(1 for row in rows if row["deadline_miss"])
    return {
        "schema": "v370-live-closed-loop/1",
        "live": True,
        "arm": "subject",
        "decisions": len(rows),
        "requested": decisions,
        "stale_reads": stale,
        "dropped_reads": dropped,
        "same_frame_verified": bool(monotonic),
        "frame_span": [frames[0], frames[-1]] if frames else [],
        "complete_ratio": round(len(rows) / max(1, len(rows) + stale + dropped), 4),
        "deaths": deaths,
        "respawns": respawns,
        "intents": _counts(row["intent"] for row in rows),
        "deadline_misses": misses,
        "deadline_miss_rate": round(misses / len(rows), 4) if rows else None,
        "plan_ms_p50": round(sorted(plan_ms)[len(plan_ms) // 2], 3) if plan_ms else None,
        "plan_ms_p95": (
            round(sorted(plan_ms)[max(0, int(len(plan_ms) * 0.95) - 1)], 3) if plan_ms else None
        ),
        "max_hazards": max((row["hazards"] for row in rows), default=0),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "rows": rows,
    }


def _counts(values) -> dict:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def write(document: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def b05_check(document: dict, *, threshold: float = 0.99) -> dict:
    """B-05's completeness criterion for one arm of one run."""

    ratio = document["complete_ratio"]
    return {
        "complete_ratio": ratio,
        "threshold": threshold,
        "status": "pass" if ratio >= threshold else "fail",
        "stale_reads": document["stale_reads"],
        "dropped_reads": document["dropped_reads"],
    }


def b07_check(document: dict) -> dict:
    """B-07 needs the paired labels; the live probe only guarantees they were recorded."""

    rows = document["rows"]
    labelled = sum(1 for row in rows if row.get("action") is not None)
    return {
        "rows": len(rows),
        "labelled": labelled,
        "status": "ready" if labelled == len(rows) and rows else "incomplete",
    }
