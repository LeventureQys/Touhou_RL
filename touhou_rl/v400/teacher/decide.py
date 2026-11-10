"""``teacher-v1`` core: one decision, timed in four phases (11-P1 §2, §5).

One decision is:

1. **record**  — freeze the decision point (``PARK``), read the frame-event word
   (``EVENTS``) and record the live horizon ``H`` frames with the committed plan
   plus "不动(聚焦)" (:mod:`.hazard_tape` / :mod:`.child_record`);
2. **search**  — beam search on that tape over the 18 first actions
   (:mod:`.search`) → ``geo_score[18]``;
3. **verify**  — roll the top ``k`` plans out in real forks of the same state and
   let the engine decide life or death (:mod:`.verify`);
4. **commit**  — the caller's job: advance the authoritative process ``R``
   sub-decisions and read ``STATE``.  It is measured by the CLI, not here,
   because only the caller knows how many sub-decisions it commits.

``record_source`` selects **where** the tape is recorded:

* ``"child"`` (the shipped path): ``PARK`` once, ``SPAWN`` a copy, let the copy
  ``STEP`` + answer ``CHILD HAZARDS``/``CHILD PLAYER`` for every frame, drop the
  copy.  The shadow does not move, so nothing has to be restored — the ~1.6 s
  ``RESET`` + replay of EXP-P1-006 disappears from the critical path.
* ``"main"`` (legacy, used only by ``--allow-legacy-restore``): the measured
  EXP-P1-003/006 forward walk, which advances the shadow ``H`` frames and needs
  the CLI to restore it with a new process + ``RESET`` + replay.

The child path is fail-closed: if ``probe_child_hazards()`` says the build cannot
serve ``CHILD HAZARDS``, :class:`~touhou_rl.v400.teacher.hazard_tape.ChildHazardsUnavailable`
is raised.  There is no silent fall back to ``"main"``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from touhou_rl.v400.teacher.child_record import ChildRecord, record_child_tape
from touhou_rl.v400.teacher.config import N_CAND, TeacherConfig
from touhou_rl.v400.teacher.hazard_tape import HazardTape, record
from touhou_rl.v400.teacher.label import TeacherLabel
from touhou_rl.v400.teacher.search import Plan, SearchResult, beam_search, score_plan
from touhou_rl.v400.teacher.verify import Consistency, VerifiedPlan, select_plans, verify_plans

#: 11-P1 §4 rule 2: the death-bomb window is ``bombs * 6`` frames, capped at 15.
DEATH_BOMB_FRAMES_PER_BOMB = 6
DEATH_BOMB_WINDOW_CAP = 15
#: ``Player.cpp``: with no bombs left the window is 2 frames — too short to bomb.
DEATH_BOMB_WINDOW_NO_BOMBS = 2
#: 11-P1 §4 rule 2: the mask is applied one frame late, so one frame is gone.
INPUT_LATENCY_FRAMES = 1


@dataclass
class Decision:
    """Everything one decision produced, including how long each phase took."""

    label: TeacherLabel
    tape: HazardTape
    search: SearchResult
    verified: list[VerifiedPlan]
    consistency: Consistency
    timing_ms: dict[str, float] = field(default_factory=dict)
    bombs: float = 0.0
    chosen: Plan | None = None
    bomb_death_available: bool = False
    child_record: ChildRecord | None = None
    events_frame: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "label": self.label.as_dict(),
            "timing_ms": {key: round(float(value), 3) for key, value in self.timing_ms.items()},
            "consistency": self.consistency.as_dict(),
            "verified": [verified.as_dict() for verified in self.verified],
            "tape": self.tape.summary(),
            "events_frame": dict(self.events_frame),
            "child_record": None if self.child_record is None else self.child_record.as_dict(),
        }


def death_bomb_window(
    bombs: float,
    *,
    time_point_bonus: bool = False,
    spell_card: bool = False,
    ship_multiplier: float = 1.0,
) -> int:
    """Frames of the death-bomb window (11-P1 §4 rule 2, ``Player.cpp:534-590``).

    Base ``bombs * 6``; ``+7`` when the time-point threshold is reached; capped at
    15; doubled inside a spell card (cap 30); ships 0/4/5 multiply by 9/5.  Only
    the base term is observable through the control protocol — time points, the
    spell-card flag and the ship class are not exported — so the default call
    returns the **lower bound**, which is the conservative direction for "can a
    bomb still be pressed in time?".
    """

    if bombs < 0:
        raise ValueError(f"bombs must be >= 0, got {bombs}")
    count = int(bombs)
    if count <= 0:
        return DEATH_BOMB_WINDOW_NO_BOMBS
    window = count * DEATH_BOMB_FRAMES_PER_BOMB
    if time_point_bonus:
        window += 7
    window = min(window, DEATH_BOMB_WINDOW_CAP)
    if spell_card:
        window = min(window * 2, DEATH_BOMB_WINDOW_CAP * 2)
    if ship_multiplier != 1.0:
        window = int(window * ship_multiplier)
    return max(window, DEATH_BOMB_WINDOW_NO_BOMBS)


def death_bomb_feasible(bombs: float, window: int | None = None) -> bool:
    """Can a death bomb still be pressed inside the window? (11-P1 §4 rule 2)

    The engine applies the mask one frame late and a window of ``<= 2`` frames is
    the "no bombs" case, which the spec says is too short: rule 1 (active bomb)
    is the only option there.
    """

    if bombs <= 0:
        return False
    frames = death_bomb_window(bombs) if window is None else int(window)
    return frames - INPUT_LATENCY_FRAMES > DEATH_BOMB_WINDOW_NO_BOMBS


def decide(
    engine,
    config: TeacherConfig,
    *,
    committed=(),
    extra_actions=(),
    frame: int = 0,
    stage: int = 0,
    difficulty: int = 0,
    consistency: Consistency | None = None,
    bombs: float | None = None,
    include_items: bool | None = None,
    record_source: str = "main",
    shoot_release: bool = False,
) -> Decision:
    """Run one full decision against ``engine`` (which sits on the decision point).

    ``committed`` is the previous plan's remainder: 11-P1 §2 step 1 walks the
    recording with "上一轮计划的剩余部分 + 不动(聚焦)".  ``record_source="main"``
    moves the process (the caller must restore it); ``record_source="child"``
    does not, and is the only path the shipped CLI uses.

    ``shoot_release`` records EXP-P1-009's SHOOT release frame in the label: the
    *caller* owns the commit (it knows the decision index), so it decides when
    the pulse is due and hands the verdict in for the label to carry.
    """
    if record_source not in ("main", "child"):
        raise ValueError(f"record_source must be 'main' or 'child', got {record_source!r}")
    consistency = consistency or Consistency()

    # ---- phase 0a: the decision point's event word (11-P1 §4 rule 2) ------
    # Read before anything advances the process: ``EVENTS`` is the *last advanced
    # frame's* bitmask (the engine clears it at the start of every frame), so it
    # must be sampled on the frame the decision is taken from.
    started = time.perf_counter()
    events_frame = engine.events_snapshot()
    events_ms = (time.perf_counter() - started) * 1000.0

    # ---- phase 0b: freeze the decision point -----------------------------
    # Both the recording (child path) and verification fork from this template,
    # so they judge the same state.  Its cost belongs to "record": it exists to
    # produce the tape and is reused by verification for free.
    started = time.perf_counter()
    template = engine.park()
    park_ms = (time.perf_counter() - started) * 1000.0

    # ---- phase 1: record -------------------------------------------------
    started = time.perf_counter()
    child_record: ChildRecord | None = None
    if record_source == "child":
        child_record = record_child_tape(
            engine,
            horizon=config.horizon,
            planned_actions=committed,
            decision_frames=config.decision_frames,
            template=template,
        )
        tape = child_record.tape
    else:
        tape = record(
            engine,
            horizon=config.horizon,
            planned_actions=committed,
            decision_frames=config.decision_frames,
        )
    record_ms = park_ms + (time.perf_counter() - started) * 1000.0

    # ---- phase 2: geometric beam search ----------------------------------
    started = time.perf_counter()
    result = beam_search(tape, config)
    search_ms = (time.perf_counter() - started) * 1000.0

    # ---- phase 3: engine verification ------------------------------------
    started = time.perf_counter()
    candidates = select_plans(result, config.verify_k, extra_actions)
    verified = verify_plans(engine, tape, candidates, config=config, template=template)
    verify_ms = (time.perf_counter() - started) * 1000.0
    for verified_plan in verified:
        consistency.add(verified_plan.plan, verified_plan)

    # ---- selection (11-P1 §2 step 5, §4 rules) ---------------------------
    bombs_available = float(tape.player["bombs"][0] if bombs is None else bombs)
    survivors = [item for item in verified if not item.died]
    chosen: Plan | None = None
    events: list[str] = []
    if survivors:
        best = max(survivors, key=lambda item: item.plan.score)
        chosen = _replace(best.plan, verified=True)
        events.append("survive")
    else:
        # Every verified plan dies.  Prefer the one that dies latest, as 11-P1 §3
        # orders dead plans, and let the bomb rule below decide.
        fallback = max(verified, key=lambda item: item.plan.score) if verified else None
        chosen = None if fallback is None else _replace(fallback.plan, verified=True)
        events.append("all_verified_plans_die")
    if chosen is None:
        chosen = result.best()
        if chosen is not None:
            events.append("unverified_fallback")

    # 11-P1 §4 rule 1: every *verified* plan dies and there is a bomb left.
    # The rule is about step 5's engine verdict ("发现所有计划都会死"), not about
    # the geometry model: a plan the engine kills can still have
    # ``death_frame is None`` in the tape, and that case is exactly where the
    # bomb matters (EXP-P1-007 §3).
    all_verified_die = bool(verified) and not survivors
    shortest_death_frame = min(
        (int(item.death_frame) for item in verified if item.died and item.death_frame is not None),
        default=None,
    )
    rule1_window = config.bomb_rule1_max_death_frame
    # EXP-P1-015 §3: the v1 logs show rule 1 firing on 48-frame forecasts (the
    # shortest verified death was 27-47 frames away on every bomb of Stage4A/
    # Stage6A), so ``v2a``/``v2b`` only spend a bomb when the engine-confirmed
    # death is at most ``rule1_window`` frames away.
    rule1_imminent = rule1_window is None or (
        shortest_death_frame is not None and shortest_death_frame <= int(rule1_window)
    )
    # EXP-P1-015 §3b (v2c): keep the last bomb for the real death bomb, and only
    # spend one when the plan is already tight (``bomb_rule1_max_margin``).
    min_bombs = int(getattr(config, "bomb_rule1_min_bombs_left", 1) or 0)
    rule1_enough_bombs = bombs_available > min_bombs
    max_margin = getattr(config, "bomb_rule1_max_margin", None)
    chosen_margin = float(chosen.margin) if chosen is not None else float("nan")
    rule1_margin_ok = max_margin is None or (
        chosen_margin == chosen_margin and chosen_margin < float(max_margin)
    )
    bomb = 0
    if config.bomb_active and all_verified_die and bombs_available > 0:
        if rule1_imminent and rule1_enough_bombs and rule1_margin_ok:
            bomb = 1
            events.append("active_bomb")
        else:
            reason = []
            if not rule1_imminent:
                reason.append(f"death_in_{shortest_death_frame}f")
            if not rule1_enough_bombs:
                reason.append(f"keeping_last_bombs_left_{bombs_available:.0f}")
            if not rule1_margin_ok:
                reason.append(f"margin_{chosen_margin:.1f}px")
            events.append("active_bomb_suppressed:" + ",".join(reason))
    elif config.bomb_active and all_verified_die and bombs_available <= 0:
        events.append("active_bomb_no_bombs_left")

    # 11-P1 §4 rule 2: a real PLAYER_HIT on the main process means the next
    # decision bombs immediately.  ``EVENTS`` is the only source of that bit;
    # when the engine refuses it the label says so and no bomb is fabricated.
    bomb_death_available = bool(events_frame.get("available"))
    frame_event_flags = events_frame.get("frame_event_flags")
    player_hit = bool(events_frame.get("player_hit"))
    if config.bomb_death:
        if not bomb_death_available:
            events.append(f"death_bomb_unavailable:{events_frame.get('reason')}")
        elif player_hit:
            if bombs_available <= 0:
                events.append("death_bomb_player_hit_but_no_bombs")
            elif death_bomb_feasible(bombs_available):
                bomb = 1
                events.append("death_bomb_player_hit")
            else:
                events.append("death_bomb_window_too_short")
        else:
            events.append("death_bomb_no_player_hit")

    margin = float(chosen.margin) if chosen is not None else float("nan")
    scores = np.array(result.scores, dtype=np.float64, copy=True)
    verified_mask = np.zeros(N_CAND, dtype=bool)
    for verified_plan in verified:
        cand = int(verified_plan.plan.first_action)
        scores[cand] = float(_replace(verified_plan.plan, verified=True).score)
        verified_mask[cand] = True
    for cand in range(N_CAND):
        if not verified_mask[cand] and np.isfinite(scores[cand]):
            scores[cand] -= float(config.weights.unverified_discount)

    label = TeacherLabel(
        scores=scores,
        bomb=bomb,
        chosen_plan=chosen,
        margin=margin,
        verified_mask=verified_mask,
        bomb_death_available=bomb_death_available,
        frame_event_flags=None if frame_event_flags is None else int(frame_event_flags),
        bombs=bombs_available,
        shoot=bool(config.shoot_always),
        shoot_release=bool(shoot_release),
        frame=int(frame),
        stage=int(stage),
        difficulty=int(difficulty),
        events=events,
        verified=[item.as_dict() for item in verified],
        timing_ms={
            "record": record_ms,
            "search": search_ms,
            "verify": verify_ms,
            "events": events_ms,
        },
        extra={
            "horizon": config.horizon,
            "beam": config.beam,
            "mode": config.mode,
            "record_source": record_source,
            "target_x": result.target_x,
            "slots": result.slots,
            "expansions": result.detail.get("expansions", 0),
            "roots": result.detail.get("roots", 0),
            "items_recorded": bool(include_items if include_items is not None else config.record_items),
            "park_ms": round(park_ms, 3),
            "event_names": list(events_frame.get("names") or []),
            "death_bomb_window": death_bomb_window(bombs_available),
            "death_bomb_feasible": death_bomb_feasible(bombs_available),
            # EXP-P1-015 §3: what rule 1 saw, so the bombe timing can be audited
            # from the JSONL without replaying the decision.
            "bomb_rule1_max_death_frame": rule1_window,
            "shortest_verified_death_frame": shortest_death_frame,
            "bomb_rule1_min_bombs_left": min_bombs,
            "bomb_rule1_max_margin": max_margin,
            "bomb_rule1_enough_bombs": bool(rule1_enough_bombs),
            "bomb_rule1_margin_ok": bool(rule1_margin_ok),
            "bomb_rule1_imminent": bool(rule1_imminent),
            "bomb_rule1_suppressed": bool(
                config.bomb_active
                and all_verified_die
                and bombs_available > 0
                and not rule1_imminent
            ),
            # EXP-P1-009: SHOOT is always on; the release frame is the rising
            # edge the stage transition waits for (10-P0 §4.2).
            "shoot_always": bool(config.shoot_always),
            "shoot_release": bool(shoot_release),
            "shoot_pulse_period": int(config.shoot_pulse_period),
            "child_record": None if child_record is None else child_record.as_dict(),
        },
    )
    timing = {
        "record": record_ms,
        "search": search_ms,
        "verify": verify_ms,
        "events": events_ms,
    }
    return Decision(
        label=label,
        tape=tape,
        search=result,
        verified=verified,
        consistency=consistency,
        timing_ms=timing,
        bombs=bombs_available,
        chosen=chosen,
        bomb_death_available=bomb_death_available,
        child_record=child_record,
        events_frame=dict(events_frame),
    )


def _replace(plan: Plan, *, verified: bool) -> Plan:
    """A copy of ``plan`` with the ``verified`` flag set (frozen-friendly)."""
    return Plan(
        first_action=plan.first_action,
        duration=plan.duration,
        slots=plan.slots,
        margin=plan.margin,
        death_frame=plan.death_frame,
        end_xy=plan.end_xy,
        score=plan.score,
        verified=verified,
        align=plan.align,
        edge=plan.edge,
        high=plan.high,
        items=plan.items,
        bottom=plan.bottom,
        escape=plan.escape,
        # EXP-P1-017 §2 / EXP-P1-020 §2: the diagnostic cost terms travel with the
        # copy too (``score`` is already copied verbatim, so this changes no
        # decision; it only keeps the label's breakdown complete).
        boss=plan.boss,
        top=plan.top,
        corner=plan.corner,
        laser=plan.laser,
        geo_score=plan.score,
    )


def restore(engine, history, *, target_frames: int, stage: int, difficulty: int, team: int, seed: int):
    """Put the shadow back on ``target_frames`` by RESET + replay.

    ``history`` is one entry per committed sub-decision, oldest first; an entry is
    an ``int`` candidate or a ``(candidate, bomb)`` pair (the bomb bit changes the
    world, so a replay that drops it would land on a different state).  Returns
    the wall time in milliseconds.  This is the price of recording on the main
    process (EXP-P1-003 §2/§4); the child path does not call it.
    """
    from touhou_rl.v400.gym.actions import cand_to_mask

    started = time.perf_counter()
    engine.reset(stage=stage, difficulty=difficulty, team=team, seed=seed)
    advanced = 0
    for item in history:
        if advanced >= target_frames:
            break
        cand, bomb = _history_item(item)
        frames = min(2, target_frames - advanced)
        engine.step(cand_to_mask(cand, bomb), frames)
        advanced += frames
    while advanced < target_frames:
        engine.step(cand_to_mask(9, 0), 1)
        advanced += 1
    return (time.perf_counter() - started) * 1000.0


def _history_item(item) -> tuple[int, int]:
    """``int`` or ``(cand, bomb)`` -> ``(cand, bomb)``."""

    if isinstance(item, (tuple, list)):
        return int(item[0]), int(item[1]) if len(item) > 1 else 0
    return int(item), 0
