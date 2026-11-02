"""Intent state machine and execution layer (D5).

What this layer owns
--------------------
The planner (D4) answers "which route is safe".  This module answers "what is the
*campaign* doing right now" and turns it into one action per decision:

    LOCAL ──(closure_eta approaches)──▶ PREPARE_EXIT ──▶ EVACUATE ──▶ HOLD
      ▲                                    │               │
      └────────────(threat gone)───────────┴──(route invalid)──▶ REPLAN

plus ``UNKNOWN`` (no trustworthy observation) and ``EMERGENCY`` (no route and
already close to a hazard).  Four rules make the behaviour auditable:

1. **Intent persists but keys never lock.**  The state is kept for at least
   ``min_hold_decisions`` *only* when the previous route is still valid and no
   emergency is detected; every decision re-verifies the first action with the
   exact per-frame predicate (the planner already did, this is a second check).
2. **Late is a state, not a surprise.**  Evacuation starts from
   ``closure_eta − horizon − margin``, not from "a bullet is close".
3. **Failure is explicit.**  No route + hazards + no bombs is ``EMERGENCY`` with
   a recorded fallback; no observation is ``UNKNOWN`` with a held action, never a
   silent "keep going".
4. **One bomb arbiter.**  The controller is the only place that may set the bomb
   bit, with a cooldown; the planner never does.

The controller is pure with respect to the game: it consumes an
``EscapeObservation``-shaped record plus a ``DangerField`` and returns a
``ControlDecision``.  ``run_episode`` wires it to the synthetic testbed for
evaluation; the live probe (D8) will wire it to the adapter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from . import geometry, planner
from .contracts import (
    ControlDecision,
    DECISION_SCHEMA,
    Intent,
    RejectReason,
    RejectedAction,
)

ACTION_LABELS = tuple(f"f{focus}-m{move}" for focus, move in planner.ACTIONS)


@dataclass(frozen=True)
class ControllerConfig:
    #: Start evacuating when the exit closes within this many frames of the plan end.
    prepare_margin_frames: int = 90
    #: Escalate to EVACUATE when the margin is below this.
    evacuate_margin_frames: int = 45
    #: Treat "no route and clearance at or below this" as EMERGENCY.
    emergency_clearance_px: float = 2.0
    #: Minimum decisions an intent is kept when nothing forces a change.
    min_hold_decisions: int = 2
    #: How many decisions a committed route may be followed before replanning.
    #: Measured trade-off (D7d): 1 (= no continuation) keeps E-05 alive at 20x4
    #: but SC-15 flips and dies at 44; 2-4 lifts SC-15 to 125-156 frames yet
    #: breaks E-05 (50) and even SC-15 at 12x2 (148-156).  No value fixes both,
    #: so the default stays off until corridor commitment exists.
    continuation_decisions: int = 1
    #: Decisions the committed corridor (region sequence) is kept before refresh.
    corridor_decisions: int = 20
    #: Decisions to wait after a bomb.
    bomb_cooldown: int = 20
    #: Per-decision budget; a decision slower than this is reported as a miss.
    #: 33.3 ms = two frames at 60 fps (frameskip 2), the B-06 budget.
    deadline_ms: float | None = 33.3
    #: Decisions older than this (in frames) are treated as unusable.
    max_observation_age: int = 2
    #: How many decisions a plan may be reused before a replan is forced.
    route_max_age: int = 1
    planner: planner.PlannerConfig = field(default_factory=planner.PlannerConfig)


@dataclass
class _State:
    intent: Intent = Intent.LOCAL
    route: planner.Route | None = None
    plan: planner.PlannerResult | None = None
    route_age: int = 0
    hold: int = 0
    bomb_cooldown: int = 0
    decision_id: int = 0
    #: Remaining actions of the committed route, and how many were executed.
    plan_actions: tuple[tuple[int, int], ...] = ()
    plan_cursor: int = 0
    #: Committed region sequence and its age.
    corridor: tuple = ()
    corridor_age: int = 0


class EscapeController:
    """One controller instance per episode; ``reset`` guarantees no residue."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self.state = _State()
        self._score_mode: str | None = None

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        self.state = _State()
        self._score_mode = None
        self._freedom = None
        self._freedom_field = None

    # -- helpers ---------------------------------------------------------
    def _clearance(self, field, x: float, y: float, frame: int) -> float | None:
        return field.clearance(x, y, frame, inflation=self.config.planner.inflation_px)

    def _rejected(self, plan: planner.PlannerResult | None, chosen: tuple[int, int] | None) -> tuple[RejectedAction, ...]:
        if plan is None:
            return ()
        out: list[RejectedAction] = []
        for action, status, reason in zip(planner.ACTIONS, plan.first_action_status, plan.first_action_reason):
            if chosen is not None and action == chosen:
                continue
            if status == "immediate_collision":
                code = RejectReason.IMMEDIATE_COLLISION
            elif "dead_end" in reason:
                code = RejectReason.PREFER_SAFER_ROUTE
            elif status == "feasible":
                code = RejectReason.PREFER_SAFER_ROUTE
            else:
                code = RejectReason.UNKNOWN
            out.append(
                RejectedAction(
                    action=(int(action[1]), 1, int(action[0]), 0),
                    reason=code,
                    evidence=reason[:160],
                    min_clearance=None,
                )
            )
        return tuple(out[:8])

    # -- decision --------------------------------------------------------
    def decide(
        self,
        *,
        x: float,
        y: float,
        frame: int,
        previous_action: tuple[int, int],
        field,
        observation_ok: bool = True,
        observation_reason: str = "",
        observation_age: int = 0,
        bombs: int = 0,
    ) -> tuple[ControlDecision, dict]:
        started = time.perf_counter()
        state = self.state
        state.decision_id += 1
        if state.bomb_cooldown > 0:
            state.bomb_cooldown -= 1
        diagnostics: dict = {"frame": frame, "observation_age": observation_age}

        # A-01 asks for a missing/late-frame test: a stale observation is not a
        # decision base.  Age is compared against a configured bound, so the same
        # code path covers "adapter returned an old frame" and "frame dropped".
        if observation_ok and observation_age > self.config.max_observation_age:
            observation_ok = False
            observation_reason = (
                f"stale observation: age {observation_age} > {self.config.max_observation_age}"
            )

        if not observation_ok:
            # No trustworthy "now": hold the previous action, never pretend.
            action = (previous_action[1], 1, previous_action[0], 0)
            decision = self._record(
                state,
                Intent.UNKNOWN,
                action,
                route_id=None,
                reason=f"observation unusable: {observation_reason or 'unknown'}",
                rejected=(),
                fallback="hold_previous_action",
                timing=started,
            )
            state.intent = Intent.UNKNOWN
            return decision, diagnostics

        clearance = self._clearance(field, x, y, frame)
        state.route_age += 1
        state.corridor_age += 1
        # Route continuity: while the committed route still has actions and the
        # next one re-verifies, keep executing it.  Re-selecting the best route
        # every decision made SC-15 flip between the dead-end side and the safe
        # mouth at long horizons (44 frames vs 180).
        pending = None
        if (
            state.plan_actions
            and state.plan_cursor < len(state.plan_actions)
            and state.plan_cursor < self.config.continuation_decisions
        ):
            candidate = state.plan_actions[state.plan_cursor]
            if self._verify_first_action(field, x, y, frame, candidate, previous_action):
                pending = candidate
        needs_replan = pending is None and (
            state.plan is None
            or state.plan.route is None
            or state.route_age >= self.config.route_max_age
            or self._route_invalid(field, x, y, frame, state.plan)
        )
        if needs_replan:
            plan_config = self.config.planner
            if plan_config.score_mode == "auto":
                # Resolve the mode once per episode.  Re-resolving per decision made
                # the controller oscillate (SC-16: 180 -> 133 frames), because the
                # closure margin changes as soon as the player starts moving.
                if self._score_mode is None:
                    from . import exit as exit_module

                    table = exit_module.closure_map(field)
                    self._score_mode, _diag = exit_module.adaptive_mode(field, (x, y), table)
                plan_config = replace(plan_config, score_mode=self._score_mode)
            committed = state.corridor if state.corridor_age < self.config.corridor_decisions else ()
            plan = planner.plan(
                field,
                (x, y),
                previous_action=previous_action,
                config=plan_config,
                t_offset=frame,
                corridor=committed,
            )
            state.plan = plan
            state.route = plan.route
            state.route_age = 0
            if not committed and plan.route is not None and plan.record.corridor_by_time:
                # Commit to the region sequence (not to individual actions) and
                # keep it fixed for ``corridor_decisions``: refreshing it every
                # decision made SC-15 flip exactly as before.
                state.corridor = tuple(plan.record.corridor_by_time)
                state.corridor_age = 0
            elif plan.route is None:
                state.corridor = ()
                state.corridor_age = 0
            if plan.route is not None and plan.record.action_sequence:
                state.plan_actions = tuple(
                    (int(action[2]), int(action[0])) for action in plan.record.action_sequence
                )
                state.plan_cursor = 0
            else:
                state.plan_actions = ()
                state.plan_cursor = 0
        plan = state.plan
        diagnostics["planner_status"] = plan.status.value if plan is not None else None
        diagnostics["plan_ms"] = plan.elapsed_ms if plan is not None else None

        route = plan.route if plan is not None else None
        closure = plan.record.closure_eta if plan is not None else None
        # "How long does the area I am standing in stay open" — that is the clock
        # the evacuation phase has to beat, not the closure of the route's end.
        current_closure = self._current_closure(field, x, y)
        margin = None if current_closure is None else int(current_closure - frame)
        diagnostics["current_closure"] = current_closure
        diagnostics["route_closure"] = closure

        if route is None:
            emergency = clearance is not None and clearance <= self.config.emergency_clearance_px
            intent = Intent.EMERGENCY if emergency else Intent.UNKNOWN
            bomb, bomb_reason = self._maybe_bomb(clearance, bombs)
            action = (0, 1, 1, int(bomb)) if bomb else (previous_action[1], 1, previous_action[0], 0)
            decision = self._record(
                state,
                intent,
                action,
                route_id=None,
                reason=(
                    f"no route: {plan.reasons[0] if plan and plan.reasons else 'unknown'}; "
                    f"clearance={clearance}"
                ),
                rejected=self._rejected(plan, None),
                fallback="bomb" if bomb else "hold_previous_action",
                bomb_reason=bomb_reason,
                emergency_override=bool(emergency),
                timing=started,
            )
            state.intent = intent
            return decision, diagnostics

        # A route exists: decide the campaign phase from the closure margin.
        if margin is not None and margin <= self.config.evacuate_margin_frames:
            proposed = Intent.EVACUATE
        elif margin is not None and margin <= self.config.prepare_margin_frames:
            proposed = Intent.PREPARE_EXIT
        else:
            proposed = Intent.LOCAL
        if (
            state.intent in (Intent.EVACUATE, Intent.PREPARE_EXIT)
            and state.hold < self.config.min_hold_decisions
            and proposed == Intent.LOCAL
        ):
            intent = state.intent  # do not drop an evacuation on a single quiet frame
        else:
            intent = proposed
        if intent != state.intent:
            state.hold = 0
        else:
            state.hold += 1

        if pending is not None:
            # Continue the committed route; it was verified when planned and the
            # next step just re-verified, so no re-selection is needed.
            action = pending
            state.plan_cursor += 1
            verified = True
            continuation = True
        else:
            contract_action = plan.record.action_sequence[0]
            action = (int(contract_action[2]), int(contract_action[0]))
            verified = self._verify_first_action(field, x, y, frame, action, previous_action)
            continuation = False
        if not verified:
            # The plan said safe, the exact re-check says otherwise: replan once.
            replanned = planner.plan(
                field,
                (x, y),
                previous_action=previous_action,
                config=self.config.planner,
                t_offset=frame,
            )
            state.plan = replanned
            if replanned.route is not None and replanned.record.action_sequence:
                state.route = replanned.route
                route = replanned.route
                plan = replanned
                state.plan_actions = tuple(
                    (int(item[2]), int(item[0])) for item in replanned.record.action_sequence
                )
                state.plan_cursor = 1
                contract_action = replanned.record.action_sequence[0]
                action = (int(contract_action[2]), int(contract_action[0]))
                intent = Intent.REPLAN
            else:
                state.plan_actions = ()
                state.plan_cursor = 0
                bomb, bomb_reason = self._maybe_bomb(clearance, bombs)
                decision = self._record(
                    state,
                    Intent.EMERGENCY,
                    (0, 1, 1, int(bomb)) if bomb else (previous_action[1], 1, previous_action[0], 0),
                    route_id=None,
                    reason="planned action failed exact re-check and replanning found nothing",
                    rejected=self._rejected(plan, None),
                    fallback="bomb" if bomb else "hold_previous_action",
                    bomb_reason=bomb_reason,
                    emergency_override=True,
                    timing=started,
                )
                state.intent = Intent.EMERGENCY
                return decision, diagnostics
        elif not continuation:
            state.plan_cursor = 1

        decision = self._record(
            state,
            intent,
            (int(action[1]), 1, int(action[0]), 0),
            route_id=plan.record.route_id,
            reason=(
                f"{intent.value}: {'continuing' if continuation else 'new'} route, "
                f"route_closure={closure}, current_closure={current_closure}, margin={margin}, "
                f"min_clearance={plan.record.min_clearance}"
            ),
            rejected=self._rejected(plan, action),
            timing=started,
        )
        state.intent = intent
        return decision, diagnostics

    # -- pieces ----------------------------------------------------------
    def _current_closure(self, field, x: float, y: float) -> int | None:
        table = getattr(self, "_freedom", None)
        if table is None or getattr(self, "_freedom_field", None) is not field:
            from . import exit as exit_module

            table = exit_module.closure_map(field)
            table = np.where(exit_module.playable_mask(field), table, -1)
            self._freedom = table
            self._freedom_field = field
        from . import exit as exit_module

        value = int(exit_module.cell_freedom(field, x, y, table))
        if value < 0 or value > field.config.horizon_frames:
            return None
        return value

    def _route_invalid(self, field, x: float, y: float, frame: int, plan: planner.PlannerResult) -> bool:
        route = plan.route
        if route is None:
            return True
        # Reuse is only allowed while the current position can still join the route:
        # compare against the first knot and require it to be within one step.
        knots = plan.record.position_time_knots
        if not knots:
            return True
        first = knots[0]
        distance = ((first.x - x) ** 2 + (first.y - y) ** 2) ** 0.5
        return distance > 8.0

    def _verify_first_action(self, field, x, y, frame, action, previous) -> bool:
        model = geometry.MoveModel()
        velocity = model.displacement(action[0], action[1], frames=1)
        previous_velocity = model.displacement(previous[0], previous[1], frames=1)
        px, py = x, y
        for step in range(1, self.config.planner.frameskip + 1):
            delta = previous_velocity if step <= self.config.planner.settle_frames else velocity
            px, py = model.clamp(px + delta[0], py + delta[1])
            clearance = self._clearance(field, px, py, frame + step)
            if clearance is None:
                return False
            if clearance <= 0.0:
                return False
        return True

    def _maybe_bomb(self, clearance: float | None, bombs: int) -> tuple[bool, str | None]:
        if bombs <= 0 or self.state.bomb_cooldown > 0:
            return False, None
        if clearance is None or clearance > 0.0:
            return False, None
        self.state.bomb_cooldown = self.config.bomb_cooldown
        return True, f"no route and clearance {clearance:.3f} <= 0 with {bombs} bomb(s)"

    def _record(
        self,
        state: _State,
        intent: Intent,
        action: tuple[int, int, int, int],
        *,
        route_id: str | None,
        reason: str,
        rejected: tuple[RejectedAction, ...],
        fallback: str | None = None,
        bomb_reason: str | None = None,
        emergency_override: bool = False,
        timing: float,
    ) -> ControlDecision:
        elapsed_ms = round((time.perf_counter() - timing) * 1000.0, 3)
        # A-01 asks for a timeout test.  "Missed the deadline" is a measured fact,
        # not a placeholder: it is true when this decision took longer than the
        # configured budget.  The synthetic loop never misses it; the recorded
        # probe and the live loop can.
        deadline_miss = self.config.deadline_ms is not None and elapsed_ms > self.config.deadline_ms
        return ControlDecision(
            schema=DECISION_SCHEMA,
            decision_id=state.decision_id,
            intent=intent,
            accepted_action=action,
            reason=reason,
            route_id=route_id,
            rejected_actions=rejected,
            emergency_override=bool(emergency_override),
            fallback=fallback,
            bomb_reason=bomb_reason,
            deadline_miss=bool(deadline_miss),
            timing_breakdown={"controller_ms": elapsed_ms, "budget_ms": self.config.deadline_ms},
        )


def run_episode(
    scenario,
    *,
    config: ControllerConfig | None = None,
    forecast_config=None,
    max_decisions: int | None = None,
) -> dict:
    """Closed loop through the controller against the synthetic world."""

    from . import forecast as forecast_module
    from . import synth as synth_module

    config = config or ControllerConfig()
    forecast_config = forecast_config or forecast_module.ForecastConfig()
    env = synth_module.SyntheticEnv(scenario)
    field = forecast_module.DangerField.from_scenario(scenario, 0, forecast_config)
    controller = EscapeController(config)
    decisions: list[dict] = []
    intents: dict[str, int] = {}
    limit = max_decisions or scenario.horizon_frames // config.planner.frameskip
    while not env.hit and env.frame < scenario.horizon_frames and len(decisions) < limit:
        decision, _diag = controller.decide(
            x=env.x,
            y=env.y,
            frame=env.frame,
            previous_action=env.previous_action,
            field=field,
            observation_ok=True,
            bombs=3,
        )
        intents[decision.intent.value] = intents.get(decision.intent.value, 0) + 1
        decisions.append(
            {
                "decision_id": decision.decision_id,
                "frame": env.frame,
                "x": env.x,
                "y": env.y,
                "intent": decision.intent.value,
                "action": list(decision.accepted_action),
                "route_id": decision.route_id,
                "plan_status": (
                    controller.state.plan.status.value
                    if controller.state.plan is not None
                    else None
                ),
                "reason": decision.reason,
                "fallback": decision.fallback,
                "bomb": bool(decision.accepted_action[3]),
            }
        )
        env.step((int(decision.accepted_action[2]), int(decision.accepted_action[0])))
    return {
        "scenario_id": scenario.scenario_id,
        "frames": env.frame,
        "hit": env.hit,
        "survived": (not env.hit) and env.frame >= scenario.horizon_frames,
        "intents": intents,
        "decisions": decisions,
        "min_clearance": None if env.min_clearance == float("inf") else round(env.min_clearance, 4),
    }
