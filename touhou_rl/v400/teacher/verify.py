"""Engine verification of the search's top plans (11-P1 §2 steps 3-4).

Verification is the only place a plan's death is decided: the geometry search
proposes, the engine judges.  Each candidate is rolled out in a real fork of the
current state (``PARK`` once, ``SPAWN`` per plan, ``CHILD <h> STEP`` per decision
slot), and death is read from the engine's own counters — never inferred from a
geometry score.

Consistency (11-P1 §2 step 4)
-----------------------------
A verification rollout reports, per frame, the ship's position and the death
counter.  Two things are compared against the search:

* **positions**: the plan's exact motion model must reproduce the engine's
  positions bit-for-bit-ish (tolerance ``POSITION_TOLERANCE``).  A mismatch means
  the motion model is wrong, not that the world changed;
* **death**: a plan the geometry called alive that the engine kills (or the
  reverse).  Those are the counts reported by the CLI and are *measurements*,
  not errors to be smoothed over.

If the rolled-out trace disagrees with the recorded tape (a different ship
position changes aimed bullets, so the tape is stale), :func:`reconcile` asks
the caller to re-record, up to ``config.max_iterations`` times (11-P1 §2 step 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from touhou_rl.v400.teacher.config import TeacherConfig
from touhou_rl.v400.teacher.engine import decode_state_fields, parse_fields
from touhou_rl.v400.teacher.search import Plan, simulate_plan

#: Allowed disagreement between the model's positions and the engine's, in px.
POSITION_TOLERANCE = 1e-3


@dataclass
class VerifiedPlan:
    """One plan's real engine rollout."""

    plan: Plan
    died: bool
    death_frame: int | None
    margin: float
    positions: list[tuple[float, float]]
    deaths: list[float]
    hashes: list[str]
    events: list[str] = field(default_factory=list)
    max_position_error: float = 0.0
    trajectory_matches: bool = True
    frames_advanced: int = 0

    def as_dict(self) -> dict:
        return {
            "first_action": int(self.plan.first_action),
            "duration": int(self.plan.duration),
            "died": bool(self.died),
            "death_frame": None if self.death_frame is None else int(self.death_frame),
            "margin": round(float(self.margin), 4),
            "frames_advanced": int(self.frames_advanced),
            "max_position_error": round(float(self.max_position_error), 6),
            "trajectory_matches": bool(self.trajectory_matches),
            "events": list(self.events),
            "end_xy": [
                round(float(self.positions[-1][0]), 3),
                round(float(self.positions[-1][1]), 3),
            ]
            if self.positions
            else None,
        }


def verify_plans(
    engine,
    tape,
    plans,
    *,
    config: TeacherConfig,
    tolerance: float = POSITION_TOLERANCE,
    template: int | None = None,
    batch: bool = True,
) -> list[VerifiedPlan]:
    """Roll every plan out in a real fork and judge it with the engine's counters.

    A single ``PARK`` serves all plans: the template is a copy of the live state
    at the decision point, and each ``SPAWN`` is an independent copy of it, so the
    plans cannot influence each other.  Each child is advanced one decision at a
    time (``STEP <mask> 2``) and read with ``STATE`` so a death is attributed to a
    two-frame decision slot instead of to the whole horizon.

    ``template`` must be a ``PARK`` taken **at the decision point**.  Recording
    advances the shadow ``H`` frames (EXP-P1-003 §3), so parking after the
    recording would verify a different world — the caller therefore parks before
    it records and hands the handle in here.
    """
    if not plans:
        return []
    if template is None:
        template = engine.park()
    results: list[VerifiedPlan] = []
    for plan in plans:
            handle = engine.spawn(template)
            positions: list[tuple[float, float]] = []
            deaths: list[float] = []
            hashes: list[str] = []
            events: list[str] = []
            position_slots: list[int] = []
            death_frame: int | None = None
            advanced = 0
            try:
                slots = _plan_slots(plan, tape)
                runs = _plan_runs(slots)
                if batch:
                    for end_slot, state, advanced_frames, state_hash in _run_rollout(engine, handle, runs):
                        advanced += advanced_frames
                        hashes.append(state_hash)
                        if state["px"] is not None:
                            positions.append((float(state["px"]), float(state["py"])))
                            position_slots.append(end_slot)
                        if state["deaths"] is not None:
                            deaths.append(float(state["deaths"]))
                    if len(deaths) > 1 and deaths[-1] > deaths[0]:
                        # A run-level pass proves the death.  Refine it on a fresh
                        # child so the two-frame attribution and the per-slot
                        # positions are kept: the refinement is another real
                        # rollout of the same template, never an inference.
                        engine.child_quit(handle)
                        handle = engine.spawn(template)
                        positions, deaths, hashes = [], [], []
                        advanced = 0
                        for slot, state, advanced_frames, state_hash in _slot_rollout(
                            engine, handle, slots
                        ):
                            advanced += advanced_frames
                            hashes.append(state_hash)
                            if state["px"] is not None:
                                positions.append((float(state["px"]), float(state["py"])))
                                position_slots.append(slot)
                            if state["deaths"] is not None:
                                deaths.append(float(state["deaths"]))
                            if len(deaths) > 1 and deaths[-1] > deaths[0] and death_frame is None:
                                death_frame = 2 * slot + 1
                                events.append("death")
                else:
                    baseline = decode_state_fields(parse_fields(engine.child(handle, "STATE")))
                    if baseline["deaths"] is not None:
                        deaths.append(float(baseline["deaths"]))
                    if baseline["px"] is not None:
                        positions.append((float(baseline["px"]), float(baseline["py"])))
                        position_slots.append(-1)
                    for slot, cand in slots:
                        step = parse_fields(engine.child(handle, f"STEP {_mask(cand)} 2"))
                        advanced += int(step.get("advanced", 0))
                        hashes.append(str(step.get("hash", "")))
                        state = decode_state_fields(parse_fields(engine.child(handle, "STATE")))
                        if state["px"] is not None:
                            positions.append((float(state["px"]), float(state["py"])))
                            position_slots.append(slot)
                        if state["deaths"] is not None:
                            deaths.append(float(state["deaths"]))
                        if len(deaths) > 1 and deaths[-1] > deaths[0] and death_frame is None:
                            death_frame = 2 * slot + 1
                            events.append("death")
            finally:
                engine.child_quit(handle)
            # Compare the engine's positions with the search's exact model.
            modelled_margin, modelled_end, modelled_death, modelled_margins = simulate_plan(
                tape, plan.slots, config=config
            )
            error = 0.0
            model_positions = _modelled_positions(tape, plan.slots, config)
            # ``positions[i]`` is the engine's STATE after the frames up to slot
            # ``position_slots[i]``: a step from frame ``2s`` to ``2s+2`` is the
            # model's ``2s+2``-th entry (the model starts at the decision point).
            for index in range(min(len(positions), len(position_slots))):
                model_index = 2 * position_slots[index] + 2
                if model_index >= len(model_positions):
                    break
                error = max(
                    error,
                    abs(positions[index][0] - model_positions[model_index][0]),
                    abs(positions[index][1] - model_positions[model_index][1]),
                )
            survives = death_frame is None
            verified = VerifiedPlan(
                plan=plan,
                died=not survives,
                death_frame=death_frame,
                margin=float(modelled_margin),
                positions=positions,
                deaths=deaths,
                hashes=hashes,
                events=events,
                max_position_error=error,
                trajectory_matches=error <= tolerance,
                frames_advanced=advanced,
            )
            results.append(verified)
    return results


def _plan_slots(plan, tape) -> list[tuple[int, int]]:
    """``(slot, candidate)`` pairs the rollout actually advances."""
    slots: list[tuple[int, int]] = []
    for slot, cand in enumerate(plan.slots):
        if 2 * slot >= tape.horizon:
            break
        slots.append((slot, int(cand)))
    return slots


def _plan_runs(slots) -> list[tuple[int, int, int]]:
    """Collapse a plan into ``(mask, frames, last_slot)`` runs of one mask.

    A decision plan holds one mask per slot, so consecutive equal candidates are
    one longer ``STEP``.  The sampled plans are mostly ``(first, first, idle,
    idle, ...)``, which is 2-4 engine round trips instead of 24; the death is
    still proven by the engine's own counter at every run boundary, and a death
    is refined per slot by :func:`_slot_rollout`.
    """
    runs: list[tuple[int, int, int]] = []
    for slot, cand in slots:
        mask = _mask(cand)
        if runs and runs[-1][0] == mask:
            previous = runs[-1]
            runs[-1] = (mask, previous[1] + 2, slot)
        else:
            runs.append((mask, 2, slot))
    return runs


def _run_rollout(engine, handle: int, runs) -> list[tuple[int, object, int, str]]:
    """One pipelined ``STEP``/``STATE`` pair per run; returns run-boundary samples.

    The first sample is the child's ``STATE`` **before** any step, so the death
    baseline is the decision point's counter (the per-slot path of EXP-P1-003
    compared every reading to the first *post-step* one, which silently missed a
    death that happened in slot 0 and never incremented again).
    """
    commands: list[str] = [f"CHILD {handle} STATE"]
    for mask, frames, _end in runs:
        commands.append(f"CHILD {handle} STEP {mask} {frames}")
        commands.append(f"CHILD {handle} STATE")
    replies = engine.ask_many(commands)
    baseline = decode_state_fields(parse_fields(replies[0]))
    out: list[tuple[int, object, int, str]] = [(-1, baseline, 0, "")]
    for index, (_mask, _frames, end_slot) in enumerate(runs):
        step = parse_fields(replies[1 + 2 * index])
        state = decode_state_fields(parse_fields(replies[2 + 2 * index]))
        out.append((end_slot, state, int(step.get("advanced", 0)), str(step.get("hash", ""))))
    return out


def _slot_rollout(engine, handle: int, slots) -> list[tuple[int, object, int, str]]:
    """One pipelined ``STEP 2``/``STATE`` pair per decision slot (+ baseline)."""
    commands: list[str] = [f"CHILD {handle} STATE"]
    for _slot, cand in slots:
        commands.append(f"CHILD {handle} STEP {_mask(cand)} 2")
        commands.append(f"CHILD {handle} STATE")
    replies = engine.ask_many(commands)
    baseline = decode_state_fields(parse_fields(replies[0]))
    out: list[tuple[int, object, int, str]] = [(-1, baseline, 0, "")]
    for index, (slot, _cand) in enumerate(slots):
        step = parse_fields(replies[1 + 2 * index])
        state = decode_state_fields(parse_fields(replies[2 + 2 * index]))
        out.append((slot, state, int(step.get("advanced", 0)), str(step.get("hash", ""))))
    return out


def _mask(cand: int) -> int:
    from touhou_rl.v400.gym.actions import cand_to_mask

    return cand_to_mask(int(cand), 0)


def _modelled_positions(tape, slots, config: TeacherConfig) -> list[tuple[float, float]]:
    """Frame positions from the exact motion model, one entry per decision slot."""
    center = np.array(tape.start_position, dtype=np.float64)
    deltas = __import__(
        "touhou_rl.v400.teacher.search", fromlist=["delta_table"]
    ).delta_table(tape.speeds)
    out: list[tuple[float, float]] = [tuple(center)]  # type: ignore[list-item]
    for cand in slots:
        step = deltas[cand]
        center = center + step
        center[0] = float(np.clip(center[0], *config.border_x))
        center[1] = float(np.clip(center[1], *config.border_y))
        out.append((float(center[0]), float(center[1])))
        center = center + step
        center[0] = float(np.clip(center[0], *config.border_x))
        center[1] = float(np.clip(center[1], *config.border_y))
        out.append((float(center[0]), float(center[1])))
    return out


@dataclass
class Consistency:
    """The geometry-versus-engine bookkeeping the card reports."""

    geo_dead_engine_alive: int = 0
    geo_alive_engine_dead: int = 0
    agree_alive: int = 0
    agree_dead: int = 0
    trajectory_mismatch: int = 0

    def add(self, plan: Plan, verified: VerifiedPlan) -> None:
        geo_dead = plan.death_frame is not None
        if geo_dead and not verified.died:
            self.geo_dead_engine_alive += 1
        elif (not geo_dead) and verified.died:
            self.geo_alive_engine_dead += 1
        elif geo_dead and verified.died:
            self.agree_dead += 1
        else:
            self.agree_alive += 1
        if not verified.trajectory_matches:
            self.trajectory_mismatch += 1

    def as_dict(self) -> dict[str, int]:
        return {
            "geo_dead_engine_alive": self.geo_dead_engine_alive,
            "geo_alive_engine_dead": self.geo_alive_engine_dead,
            "agree_alive": self.agree_alive,
            "agree_dead": self.agree_dead,
            "trajectory_mismatch": self.trajectory_mismatch,
            "plans_verified": (
                self.geo_dead_engine_alive
                + self.geo_alive_engine_dead
                + self.agree_alive
                + self.agree_dead
            ),
        }


def select_plans(result, k: int, extra=()) -> list[Plan]:
    """The ``k`` best first actions, plus any action that must be labelled.

    11-P1 §5: ``k`` verified plans in full mode, ``1 + the action to label`` in
    fast mode (DAgger).  Duplicates are dropped but order is preserved, so the
    caller can tell which plan came from the search and which was appended.
    """
    chosen: list[Plan] = []
    seen: set[int] = set()
    for cand, _score in result.ranking():
        if len(chosen) >= k:
            break
        plan = result.plans.get(cand)
        if plan is None or cand in seen:
            continue
        chosen.append(plan)
        seen.add(cand)
    for cand in extra:
        plan = result.plans.get(int(cand))
        if plan is not None and int(cand) not in seen:
            chosen.append(plan)
            seen.add(int(cand))
    return chosen


def reconcile(*_args, **_kwargs):
    """Placeholder kept out of the first slice.

    Re-recording a *verified* trajectory needs another forward walk of the shadow
    (see :mod:`touhou_rl.v400.teacher.hazard_tape`); the CLI performs the loop by
    calling :func:`touhou_rl.v400.teacher.hazard_tape.record` again with the
    verified plan as the nominal, which is exactly 11-P1 §2 step 4.  This symbol
    exists so the intent is visible in the API and a caller cannot think the loop
    is silently skipped.
    """
    raise NotImplementedError(
        "the re-record loop is driven by the CLI (see tools/rl/v400_teacher.py)"
    )
