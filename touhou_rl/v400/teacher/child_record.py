"""Child-process recording: the fork walks the horizon, the main process only commits.

11-P1 §2 step 1 asks for a *branch* of the decision point that records every
lethal's geometry.  Until ``build/rl_hazard2`` the only process that could answer
``HAZARDS`` was the authoritative shadow itself, so recording moved it ``H``
frames forward and the next decision had to pay a new process + ``RESET`` + replay
(~1.6 s, EXP-P1-006).  With ``CHILD <h> HAZARDS/PLAYER`` the branch exists:

1. ``PARK`` once at the decision point (the template is a one-way snapshot);
2. ``SPAWN`` one child from that template;
3. the child ``STEP``s one frame at a time and answers ``HAZARDS`` + ``PLAYER``
   for every frame — all of it sent as **one** pipelined batch
   (:meth:`TeacherEngine.ask_many`), because 144 child round trips at ~0.12 ms
   each would otherwise dominate the phase;
4. the child is dropped (``CHILD <h> QUIT``) and the template stays valid for
   verification.

The main process never advances during recording.  Its only motion is the
``R`` committed sub-decisions of :mod:`touhou_rl.v400.teacher.decide`, and it is
never reset — the 1.6 s restore is gone rather than hidden (EXP-P1-007 §3).

Fail-closed rules (unchanged from EXP-P1-006, now the *only* path):

* ``probe_child_hazards()`` must say the build can serve ``CHILD HAZARDS``; a
  ``False`` raises :class:`~touhou_rl.v400.teacher.hazard_tape.ChildHazardsUnavailable`
  before a single frame is recorded — no silent fall back to a main-process
  recording (the CLI has an explicit ``--allow-legacy-restore`` switch instead);
* a refused or truncated child reply is an error, never a tape with holes;
* the tape's rows come from the child's own ``HAZARDS``/``PLAYER`` replies and
  row 0 is the decision point itself (``HAZARDS`` before the first ``STEP``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from touhou_rl.v400.gym.actions import cand_to_mask
from touhou_rl.v400.gym.protocol import parse_hazards_reply, parse_player_reply
from touhou_rl.v400.teacher.engine import EngineError, parse_fields
from touhou_rl.v400.teacher.hazard_tape import (
    ChildHazardsUnavailable,
    HazardTape,
    expand_actions,
    record,
)


@dataclass
class ChildRecord:
    """One child recording, with the numbers needed to price the phase."""

    tape: HazardTape
    template: int
    handle: int
    commands: int = 0
    steps: int = 0
    reply_bytes: int = 0
    max_reply_bytes: int = 0
    spawn_ms: float = 0.0
    advance_ms: float = 0.0
    frames: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "source": "child",
            "template": int(self.template),
            "commands": int(self.commands),
            "steps": int(self.steps),
            "frames": int(self.frames),
            "reply_bytes": int(self.reply_bytes),
            "max_reply_bytes": int(self.max_reply_bytes),
            "spawn_ms": round(float(self.spawn_ms), 3),
            "advance_ms": round(float(self.advance_ms), 3),
            "ms_per_frame": round(float(self.advance_ms) / self.frames, 3) if self.frames else None,
        }


def record_child_tape(
    engine,
    *,
    horizon: int,
    planned_actions=(),
    decision_frames: int = 2,
    template: int | None = None,
) -> ChildRecord:
    """Record ``horizon`` frames in a ``SPAWN``ed child and leave the shadow alone.

    ``template`` is the ``PARK`` taken at the decision point; when it is ``None``
    one is taken here.  The template is *not* released: the caller reuses it for
    verification, which must judge the same state the tape describes.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    probe = getattr(engine, "probe_child_hazards", None)
    if not callable(probe):
        raise ChildHazardsUnavailable(
            "engine has no probe_child_hazards(); refusing to record without a "
            "measured CHILD HAZARDS capability"
        )
    verdict = probe()
    if not verdict.get("available"):
        raise ChildHazardsUnavailable(
            "engine reports child hazards unavailable: "
            f"{verdict.get('reason')!r}; the main process must not be used as a "
            "silent fallback (see --allow-legacy-restore)"
        )
    slots = expand_actions(planned_actions, horizon, decision_frames)

    started = time.perf_counter()
    if template is None:
        template = int(engine.park())
    handle = int(engine.spawn(template))
    spawn_ms = (time.perf_counter() - started) * 1000.0

    # Row 0 is the decision point: HAZARDS/PLAYER *before* the first STEP, so the
    # tape cannot label frame d+1 as frame d (EXP-P1-003 §5).
    commands = [
        f"CHILD {handle} HASH",
        f"CHILD {handle} HAZARDS",
        f"CHILD {handle} PLAYER",
    ]
    for index in range(1, horizon):
        commands.append(f"CHILD {handle} STEP {cand_to_mask(slots[index], 0)} 1")
        commands.append(f"CHILD {handle} HAZARDS")
        commands.append(f"CHILD {handle} PLAYER")

    started = time.perf_counter()
    try:
        replies = engine.ask_many(commands)
    except EngineError as error:
        raise ChildHazardsUnavailable(
            f"child recording batch failed after {len(commands)} commands: {error}"
        ) from error
    finally:
        engine.child_quit(handle)
    advance_ms = (time.perf_counter() - started) * 1000.0

    payloads: list[dict[str, object]] = []
    cursor = 1
    current_hash = str(parse_fields(replies[0]).get("hash", ""))
    try:
        for index in range(horizon):
            if index:
                step = parse_fields(replies[cursor])
                cursor += 1
                current_hash = str(step.get("hash", ""))
            hazard = parse_hazards_reply(replies[cursor])
            cursor += 1
            player = parse_player_reply(replies[cursor])
            cursor += 1
            payloads.append(
                {
                    "frame": int(player.frame),
                    "hash": current_hash,
                    "player": player,
                    "bullets": hazard.bullets,
                    "lasers": hazard.lasers,
                    "enemies": hazard.enemies,
                }
            )
    except (IndexError, ValueError) as error:  # pragma: no cover - protocol drift
        raise ChildHazardsUnavailable(
            f"child reply batch is not parseable at command {cursor}: {error}"
        ) from error

    tape = record(
        engine,
        horizon=horizon,
        planned_actions=planned_actions,
        decision_frames=decision_frames,
        source="child",
        child_callable=lambda index: payloads[index],
    )
    return ChildRecord(
        tape=tape,
        template=int(template),
        handle=int(handle),
        commands=len(commands),
        steps=horizon - 1,
        reply_bytes=input_bytes(replies),
        max_reply_bytes=max((len(reply) for reply in replies), default=0),
        spawn_ms=spawn_ms,
        advance_ms=advance_ms,
        frames=horizon,
    )


def input_bytes(replies) -> int:
    """Total bytes the child's replies occupied (evidence for the batch size)."""

    return int(sum(len(reply) for reply in replies))
