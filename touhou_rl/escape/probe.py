"""Read-only shadow probe (D8 skeleton) and paired decision records (D9).

What this module guarantees
---------------------------
* **Read-only.**  The probe consumes observations and never writes input.  In the
  live configuration the game is driven by whatever is already playing; the probe
  only decides and records.  This is why it can be exercised offline against a
  replay source without any game process.
* **Same-frame honesty.**  Every row records the observation age, the completeness
  and the frame id it was derived from; a row whose observation is unusable is
  written as ``UNKNOWN`` with the reason, never as a decision.
* **Paired viability.**  For each decision the probe also stores a *privileged*
  label: whether the chosen action survives the next ``viability_frames`` in the
  synthetic world, and whether the omniscient reference finds any surviving
  action.  That is the D9 "would a perfect player have survived here" column; it
  is a teacher label and is never fed to the controller.

Why the source is an interface
------------------------------
``FrameSource`` is the seam between synthetic replay (tests, this round) and the
live adapter (D8, needs the game).  ``ReplaySource`` replays a synthetic scenario
frame by frame; ``AdapterSource`` (live) will wrap ``escape.adapter.read_synced``.
Tests exercise the probe through ``ReplaySource`` only, so the live path stays
unclaimed until it is run on the real game.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from . import adapter, contracts, controller, forecast, geometry, reference, synth
from .contracts import Completeness, EscapeObservation, Source


class FrameSource(Protocol):
    """Something that can hand out a synchronized observation."""

    def read(self) -> EscapeObservation | None: ...
    def advance(self) -> None: ...
    def done(self) -> bool: ...


@dataclass
class ReplaySource:
    """Offline stand-in for the live adapter: replays a synthetic scenario."""

    scenario: synth.Scenario
    frame: int = 0
    previous_action: tuple[int, int] = (0, 0)
    source: Source = Source.SYNTHETIC

    def __post_init__(self) -> None:
        self._field = forecast.DangerField.from_scenario(self.scenario, 0)

    def read(self) -> EscapeObservation | None:
        field = self._field
        if field.dropped_over_cap:
            return None
        mask = field._active(self.frame)
        positions = (field.positions + field.velocities * float(self.frame))[mask]
        velocities = field.velocities[mask]
        halves = field.halves[mask]
        from .contracts import HazardObs, MotionModel, PlayerObs

        hazards = tuple(
            HazardObs(
                position=(float(positions[i, 0]), float(positions[i, 1])),
                velocity=(float(velocities[i, 0]), float(velocities[i, 1])),
                half=(float(halves[i, 0]), float(halves[i, 1])),
                motion_model=MotionModel.LINEAR,
                confidence=1.0,
                valid_until=self.scenario.horizon_frames,
            )
            for i in range(positions.shape[0])
        )
        return EscapeObservation(
            schema="v370-escape/observation",
            episode_id=self.scenario.scenario_id,
            decision_id=self.frame,
            frame_id=self.frame,
            source=self.source,
            player=PlayerObs(
                x=float(self.scenario.player_start[0]),
                y=float(self.scenario.player_start[1]),
                focus=self.previous_action[0],
                hurtbox_half=contracts.HURTBOX_HALF_PX,
            ),
            frameskip=2,
            observed_frame_age=0,
            action_latency_frames=1,
            completeness=Completeness.COMPLETE,
            coordinate_contract=dict(contracts.COORDINATE_CONTRACT),
            hazards=hazards,
            bullet_pool_count=len(hazards),
        )

    def advance(self) -> None:
        self.frame += 2

    def done(self) -> bool:
        return self.frame >= self.scenario.horizon_frames


@dataclass
class AdapterSource:
    """Live path: wraps ``adapter.read_synced`` behind the same interface.

    It is constructed with two callables so that nothing here imports the game:
    ``read_fn`` returns an ``EscapeObservation`` or ``None`` (unsynchronised), and
    ``advance_fn`` blocks until the next decision.  The live wiring (D8) will pass
    ``lambda: adapter.read_synced(game)``.  This class is exercised offline with a
    fake ``read_fn``; the real wiring is not claimed until it runs on the game.
    """

    read_fn: object
    advance_fn: object = None
    live: bool = True

    def read(self):
        return self.read_fn()

    def advance(self) -> None:
        if self.advance_fn is not None:
            self.advance_fn()

    def done(self) -> bool:
        return False


@dataclass(frozen=True)
class ProbeConfig:
    decisions: int = 200
    viability_frames: int = 24
    reference_budget: int = 400
    survey: bool = False
    planner: controller.ControllerConfig = field(default_factory=controller.ControllerConfig)


def _viability(scenario: synth.Scenario, frame: int, action: tuple[int, int]) -> dict:
    """Privileged label: does this action survive the next 24 frames from here?"""

    env = synth.SyntheticEnv(scenario)
    env.frame = int(frame)
    env.x, env.y = scenario.player_start
    env.previous_action = (0, 0)
    env.hit = False
    env.min_clearance = float("inf")
    env._check(int(frame))
    env.step(action)
    for _ in range(12):
        env.step(action)
    return {
        "action_survives_24": not env.hit,
        "min_clearance": round(float(env.min_clearance), 4),
    }


def _player_xy(observation: EscapeObservation) -> tuple[float, float]:
    """Accept both the contract dataclass and its JSON form (adapters return either)."""

    player = observation.player
    if isinstance(player, dict):
        return float(player["x"]), float(player["y"])
    return float(player.x), float(player.y)


def run_probe(
    source: FrameSource,
    scenario: synth.Scenario,
    *,
    config: ProbeConfig | None = None,
) -> dict:
    """Shadow decisions in a loop; never writes input to the source."""

    config = config or ProbeConfig()
    field = forecast.DangerField.from_scenario(scenario, 0)
    shadow = controller.EscapeController(config.planner)
    rows: list[dict] = []
    last_action: tuple[int, int] = (0, 0)
    for index in range(config.decisions):
        if source.done():
            break
        observation = source.read()
        if observation is None:
            rows.append(
                {
                    "decision": index,
                    "frame": source.frame if hasattr(source, "frame") else index * 2,
                    "status": "unreadable",
                    "intent": "unknown",
                    "action": None,
                    "reason": "source returned no synchronized observation",
                    "pair": None,
                }
            )
            source.advance()
            continue
        issues = contracts.validate_document(contracts.to_json_dict(observation))
        player_x, player_y = _player_xy(observation)
        decision, _diag = shadow.decide(
            x=player_x,
            y=player_y,
            frame=observation.frame_id,
            previous_action=last_action,
            field=field,
            observation_ok=not issues,
            observation_reason="; ".join(issues[:2]),
            observation_age=int(getattr(observation, "observed_frame_age", 0) or 0),
            bombs=0,
        )
        chosen = (int(decision.accepted_action[2]), int(decision.accepted_action[0]))
        last_action = chosen
        label = _viability(scenario, observation.frame_id, chosen)
        if config.survey:
            from . import paired

            teacher = paired.survey(scenario, observation.frame_id)
            label["classification"] = paired.classify(
                label["action_survives_24"], teacher["survivable_actions"] > 0
            )
            label["survivable_actions"] = teacher["survivable_actions"]
            label["best_min_clearance"] = teacher["best_min_clearance"]
        rows.append(
            {
                "decision": index,
                "frame": observation.frame_id,
                "observation_age": observation.observed_frame_age,
                "completeness": observation.completeness.value,
                "status": "ok",
                "intent": decision.intent.value,
                "action": list(decision.accepted_action),
                "route_id": decision.route_id,
                "reason": decision.reason[:160],
                "pair": label,
                "contract_issues": issues[:3],
            }
        )
        source.advance()
    ok_rows = [row for row in rows if row["status"] == "ok"]
    survived = sum(1 for row in ok_rows if row["pair"] and row["pair"]["action_survives_24"])
    document = {
        "schema": "v370-probe/1",
        "scenario": scenario.scenario_id,
        "decisions": len(rows),
        "readable": len(ok_rows),
        "unreadable": len(rows) - len(ok_rows),
        "chosen_action_survives": survived,
        "chosen_action_survives_rate": round(survived / len(ok_rows), 4) if ok_rows else None,
        "intents": _counts(row["intent"] for row in ok_rows),
        "rows": rows,
        "live": False,
        "note": "offline replay source; the live adapter path is not exercised here",
    }
    if config.survey:
        from . import paired

        document["paired"] = paired.summarise(rows)
    return document


def run_recorded_probe(source, *, decisions: int = 100, config: ProbeConfig | None = None) -> dict:
    """Shadow decisions over a *recorded game trace* (no game process).

    The forecast is rebuilt from the recorded bullet pool at each decision, so the
    planner sees real bullet patterns.  There is no scenario, hence no privileged
    24-frame pair label: the pairing column stays ``None`` and the result is marked
    ``live: false`` and ``source: "recorded-game-trace"``.
    """

    config = config or ProbeConfig()
    shadow = controller.EscapeController(config.planner)
    rows: list[dict] = []
    last_action: tuple[int, int] = (0, 0)
    for index in range(decisions):
        if source.done():
            break
        observation = source.read()
        if observation is None:
            rows.append(
                {
                    "decision": index,
                    "frame": source.frame_id(),
                    "status": "unreadable",
                    "intent": "unknown",
                    "action": None,
                    "reason": "recorded observation missing player state",
                }
            )
            source.advance()
            continue
        field = source.field()
        issues = contracts.validate_document(contracts.to_json_dict(observation))
        decision, _diag = shadow.decide(
            x=observation.player.x,
            y=observation.player.y,
            # Each recorded decision gets a fresh field whose base is "now", so the
            # offset is 0.  Passing the absolute game frame here double-counts the
            # offset and made every plan return "horizon exhausted".
            frame=0,
            previous_action=last_action,
            field=field,
            observation_ok=not issues,
            observation_reason="; ".join(issues[:2]),
            observation_age=int(getattr(observation, "observed_frame_age", 0) or 0),
            bombs=0,
        )
        chosen = (int(decision.accepted_action[2]), int(decision.accepted_action[0]))
        last_action = chosen
        plan = shadow.state.plan
        rows.append(
            {
                "decision": index,
                "frame": observation.frame_id,
                "status": "ok",
                "intent": decision.intent.value,
                "action": list(decision.accepted_action),
                "plan_status": plan.status.value if plan is not None else None,
                "plan_ms": plan.elapsed_ms if plan is not None else None,
                "deadline_miss": bool(decision.deadline_miss),
                "hazards": int(field.positions.shape[0]),
                "dropped_over_cap": int(field.dropped_over_cap),
                "reason": decision.reason[:160],
            }
        )
        source.advance()
    ok_rows = [row for row in rows if row["status"] == "ok"]
    plan_ms = [row["plan_ms"] for row in ok_rows if row.get("plan_ms") is not None]
    statuses: dict[str, int] = {}
    for row in ok_rows:
        status = row.get("plan_status") or "none"
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "schema": "v370-recorded-probe/1",
        "source": "recorded-game-trace",
        "live": False,
        "note": (
            "recorded bullet pool replayed behind the v3.7.0 observation/forecast/"
            "controller; same-frame sync and input timing are NOT exercised"
        ),
        "decisions": len(rows),
        "readable": len(ok_rows),
        "unreadable": len(rows) - len(ok_rows),
        "intents": _counts(row["intent"] for row in ok_rows),
        "plan_statuses": statuses,
        "plan_ms_p50": round(float(np.median(plan_ms)), 3) if plan_ms else None,
        "plan_ms_p95": round(float(np.percentile(plan_ms, 95)), 3) if plan_ms else None,
        "deadline_misses": sum(1 for row in ok_rows if row.get("deadline_miss")),
        "deadline_miss_rate": (
            round(sum(1 for row in ok_rows if row.get("deadline_miss")) / len(ok_rows), 4)
            if ok_rows
            else None
        ),
        "budget_ms": config.planner.deadline_ms,
        "max_hazards": max((row["hazards"] for row in ok_rows), default=0),
        "over_cap_decisions": sum(1 for row in ok_rows if row.get("dropped_over_cap")),
        "rows": rows,
    }


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def write_probe(document: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
