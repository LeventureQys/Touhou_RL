"""Experiment arms for the v3.7.1 shared pipeline.

An *arm* is only a decision function: it receives the same ``Snapshot`` that
every other arm receives and returns an action.  It never reads the game, never
writes input and never owns a clock -- that is the point.  v3.7.0's comparison
ran two different loops (``env.step`` for baseline, ``live.run_live_closed_loop``
for subject), so the loop itself was a confound; here the only difference
between arms is the policy.

Arms provided
-------------
``BaselineArm``
    The unmodified v3.7.0 baseline: ``AvoidPilot(PilotConfig())``.  One
    behavioural change is forced by the pipeline, not chosen: the observation it
    receives is the *same synchronized snapshot* the other arm sees, instead of
    a separate read.
``SubjectArm``
    The escape controller, fed the contract observation exactly as
    ``live.run_live_closed_loop`` did (``frame=0`` because the field base is
    "now").
``HybridVetoArm``
    Baseline local control plus one macro-route veto.  This is the P1
    single-factor arm: it differs from the baseline in exactly one way.
``DelayedArm``
    Another arm plus a fixed planning cost, i.e. the C-04 latency control: it
    separates "different decisions" from "slower decisions".
``ReplayArm``
    A recorded action sequence, replayed open-loop.  This is the C-01
    "same input -> same trajectory" probe: two replay runs through the same
    pipeline must produce the same positions, which is what proves the pipeline
    is not itself injecting state.
``NoOpArm``
    Sends the empty action every slot.  Used for the stage-A pipeline check.
``ScriptedArm``
    A fixed callable, for tests and for scripted calibration runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from . import controller, harness, live


class BaselineArm:
    """``AvoidPilot`` behind the shared pipeline (v3.7.0 baseline, unchanged)."""

    name = "baseline"

    def __init__(self, config=None, *, pilot_factory: Callable | None = None) -> None:
        from ..pilot import AvoidPilot, PilotConfig

        self._factory = pilot_factory or (lambda: AvoidPilot(config or PilotConfig()))
        self.pilot = self._factory()

    def reset(self, episode_id: str) -> None:
        # A fresh pilot per episode: AvoidPilot keeps _prev_move / _bomb_cooldown
        # across calls and residue from the previous arm would leak between arms.
        self.pilot = self._factory()

    def decide(self, snapshot: harness.Snapshot, context: harness.ArmContext) -> harness.ArmDecision:
        obs = snapshot.network_obs()
        bullets = snapshot.raw_bullets()
        action = self.pilot.act(obs, bullets, bombs=int(context.bombs))
        move, shoot, focus, bomb = (int(v) for v in action)
        return harness.ArmDecision(
            action=(move, shoot, focus, bomb),
            intent="baseline",
            reason="AvoidPilot(PilotConfig())",
            diagnostics={},
        )


@dataclass
class SubjectArm:
    """The escape controller behind the shared pipeline."""

    config: controller.ControllerConfig = field(default_factory=controller.ControllerConfig)
    name: str = "subject"

    def __post_init__(self) -> None:
        self.controller = controller.EscapeController(self.config)

    def reset(self, episode_id: str) -> None:
        self.controller = controller.EscapeController(self.config)

    def decide(self, snapshot: harness.Snapshot, context: harness.ArmContext) -> harness.ArmDecision:
        observation = snapshot.observation
        if observation is None:  # pragma: no cover - the harness rejects these
            return harness.ArmDecision(action=(0, 0, 0, 0), intent="unknown", reason="no observation")
        field = live.field_from_observation(observation)
        decision, _diagnostics = self.controller.decide(
            x=observation.player.x,
            y=observation.player.y,
            frame=0,
            previous_action=(int(context.previous_action[0]), int(context.previous_action[1])),
            field=field,
            observation_ok=True,
            observation_age=int(getattr(observation, "observed_frame_age", 0) or 0),
            bombs=int(context.bombs),
        )
        return harness.ArmDecision(
            action=tuple(int(v) for v in decision.accepted_action),
            intent=decision.intent.value,
            route_id=decision.route_id,
            reason=decision.reason,
            deadline_miss=bool(decision.deadline_miss),
            diagnostics={
                "controller_ms": decision.timing_breakdown.get("controller_ms"),
                "plan_status": (
                    None
                    if self.controller.state.plan is None
                    else self.controller.state.plan.status.value
                ),
            },
        )


@dataclass
class HybridVetoArm:
    """Baseline local control plus **one** macro-route veto (v3.7.1 P1).

    This is the single-factor arm the v3.7.0 comparison never had.  It runs the
    unmodified baseline pilot and the escape controller on the *same* snapshot,
    and only when the controller reports a high-confidence sealed future
    (``intent`` is evacuate/emergency **and** it found a route) does it replace
    the pilot's ``move`` and ``focus``.  ``shoot`` and ``bomb`` always come from
    the pilot, so the bomb protocol is bit-for-bit unchanged -- which is what
    "the subject must not change the bomb protocol" has to mean if it is to mean
    anything.

    The controller still runs every decision, so its cost is charged to the
    experiment as measured planning time.  A veto that only pays when it fires
    would be a free lunch that does not exist on the real machine.
    """

    config: controller.ControllerConfig = field(default_factory=controller.ControllerConfig)
    pilot_config: object = None
    name: str = "hybrid_veto"

    def __post_init__(self) -> None:
        self._base = BaselineArm(self.pilot_config)
        self._ctrl = SubjectArm(self.config)
        self.veto_fires = 0
        self.decisions = 0

    def reset(self, episode_id: str) -> None:
        self._base.reset(episode_id)
        self._ctrl.reset(episode_id)
        self.veto_fires = 0
        self.decisions = 0

    @staticmethod
    def veto_condition(decision: harness.ArmDecision) -> bool:
        """High-confidence sealed future: an evacuation intent *and* a route."""

        return decision.intent in ("evacuate", "emergency") and decision.route_id is not None

    def decide(self, snapshot: harness.Snapshot, context: harness.ArmContext) -> harness.ArmDecision:
        base = self._base.decide(snapshot, context)
        proposed = self._ctrl.decide(snapshot, context)
        self.decisions += 1
        fired = self.veto_condition(proposed)
        if fired:
            self.veto_fires += 1
            action = (proposed.action[0], base.action[1], proposed.action[2], base.action[3])
        else:
            action = base.action
        return harness.ArmDecision(
            action=action,
            intent="veto" if fired else "baseline",
            route_id=proposed.route_id if fired else None,
            reason=(proposed.reason if fired else "baseline local control (veto idle)")[:200],
            deadline_miss=bool(proposed.deadline_miss),
            diagnostics={
                "veto_fired": bool(fired),
                "controller_intent": proposed.intent,
                "controller_route": proposed.route_id,
                "controller_ms": proposed.diagnostics.get("controller_ms"),
                "baseline_action": list(base.action),
                "controller_action": list(proposed.action),
            },
        )


@dataclass
class DelayedArm:
    """Another arm plus a fixed planning cost (the C-04 latency control, in vivo).

    "The new policy is worse" and "the new policy is slower, and being slower is
    what hurts" are different claims.  This wrapper makes the baseline pay the
    same per-decision cost as the arm it is compared with, so the two can be told
    apart.  The delay is charged *inside* ``decide`` and therefore shows up in
    the harness's measured ``plan_ms``, which is the point.
    """

    inner: object
    delay_ms: float
    sleeper: Callable[[float], None] | None = None
    name: str = "delayed"

    def __post_init__(self) -> None:
        if self.delay_ms < 0:
            raise ValueError("delay_ms must be >= 0")
        self._sleep = self.sleeper or time.sleep

    def reset(self, episode_id: str) -> None:
        self.inner.reset(episode_id)

    def decide(self, snapshot: harness.Snapshot, context: harness.ArmContext) -> harness.ArmDecision:
        if self.delay_ms > 0:
            self._sleep(self.delay_ms / 1000.0)
        decision = self.inner.decide(snapshot, context)
        decision.diagnostics = {
            **(decision.diagnostics or {}),
            "injected_plan_ms": float(self.delay_ms),
        }
        return decision


@dataclass
class ReplayArm:
    """Open-loop replay of a recorded action sequence (C-01 determinism probe)."""

    actions: tuple[tuple[int, int, int, int], ...]
    name: str = "replay"

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("ReplayArm needs at least one action")
        self.actions = tuple(tuple(int(v) for v in action) for action in self.actions)

    def reset(self, episode_id: str) -> None:
        self.index = 0

    def decide(self, snapshot: harness.Snapshot, context: harness.ArmContext) -> harness.ArmDecision:
        position = min(getattr(self, "index", 0), len(self.actions) - 1)
        self.index = position + 1
        return harness.ArmDecision(
            action=self.actions[position],
            intent="replay",
            reason=f"replay step {position}",
        )


@dataclass
class NoOpArm:
    """Empty action every slot: the pipeline's null arm (stage A)."""

    action: tuple[int, int, int, int] = (0, 0, 0, 0)
    name: str = "noop"

    def reset(self, episode_id: str) -> None:
        return None

    def decide(self, snapshot: harness.Snapshot, context: harness.ArmContext) -> harness.ArmDecision:
        return harness.ArmDecision(
            action=tuple(int(v) for v in self.action), intent="noop", reason="no keys"
        )


@dataclass
class ScriptedArm:
    """Fixed callable: ``fn(snapshot, context) -> action``."""

    fn: Callable[[harness.Snapshot, harness.ArmContext], tuple[int, int, int, int]]
    name: str = "scripted"
    deadline_miss: bool = False

    def reset(self, episode_id: str) -> None:
        return None

    def decide(self, snapshot: harness.Snapshot, context: harness.ArmContext) -> harness.ArmDecision:
        return harness.ArmDecision(
            action=tuple(int(v) for v in self.fn(snapshot, context)),
            intent="scripted",
            deadline_miss=bool(self.deadline_miss),
        )


ARM_NAMES = (
    "baseline",
    "subject",
    "hybrid_veto",
    "delayed_baseline",
    "noop",
    "replay",
)


def make_arm(
    kind: str,
    *,
    controller_config=None,
    replay=None,
    pilot_factory=None,
    pilot_config=None,
    delay_ms: float = 0.0,
    sleeper: Callable[[float], None] | None = None,
):
    """Resolve an arm by name so a config file can describe an experiment."""

    if kind == "baseline":
        return BaselineArm(pilot_config, pilot_factory=pilot_factory)
    if kind == "subject":
        return SubjectArm(config=controller_config or controller.ControllerConfig())
    if kind == "hybrid_veto":
        return HybridVetoArm(
            config=controller_config or controller.ControllerConfig(),
            pilot_config=pilot_config,
        )
    if kind == "delayed_baseline":
        return DelayedArm(
            inner=BaselineArm(pilot_config, pilot_factory=pilot_factory),
            delay_ms=float(delay_ms),
            sleeper=sleeper,
            name="delayed_baseline",
        )
    if kind == "noop":
        return NoOpArm()
    if kind == "replay":
        if not replay:
            raise ValueError("replay arm needs an action sequence")
        return ReplayArm(tuple(tuple(int(v) for v in action) for action in replay))
    raise ValueError(f"unknown arm {kind!r}")
