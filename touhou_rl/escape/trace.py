"""v3.7.0 trace sidecar: records, attribution and offline replay (D6).

What a record contains
----------------------
One JSONL line per decision, with the four records that already have contracts:

* ``EscapeObservation`` (synthetic source in this module; live will use the
  adapter), including the frozen ``coordinate_contract``;
* a compact ``DangerForecast`` summary (node count, dropped entities, calibrated
  flag, closure ETA, build time) rather than the full 31×672 coverage grid, which
  would be ~200 KB per decision;
* the full ``RoutePlan`` (action sequence, first-action statuses and reasons,
  clearance, closure, timing);
* the full ``ControlDecision`` (intent, accepted action, rejected actions with
  evidence, fallback, bomb reason).

Hazards are stored once per decision in an NPZ sidecar (positions, velocities,
halves, offsets), the same layout the v3.5.0 writer used, so an offline replay can
redraw a frame without the game.

Why the forecast summary is not the full grid
---------------------------------------------
A 90-decision episode with the full grid would be ~18 MB of JSON per episode.
The summary keeps the fields a reviewer needs (how many entities, what was
dropped, when the exit closes) while the grid can be recomputed from the sidecar
plus the config.  This is a deliberate, documented lossy choice; it is recorded in
each line as ``forecast_summary: true``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import contracts, controller, forecast, geometry, synth
from .contracts import (
    Completeness,
    EscapeObservation,
    HazardObs,
    MotionModel,
    PlayerObs,
    Source,
)

TRACE_SCHEMA = "v370-trace/1"
SUMMARY_SCHEMA = "v370-trace-summary/1"


def _observation_dict(
    field: forecast.DangerField,
    frame: int,
    x: float,
    y: float,
    focus: int,
    count: int,
) -> dict:
    """Contract-shaped observation with the hazard list referenced, not inlined.

    Inlining hundreds of hazard boxes would make a 90-decision trace ~4 MB of
    JSON.  The full list is in the NPZ sidecar (``hazards_ref``); every scalar
    field keeps the contract's meaning, so ``validate_document`` can still be run
    after re-inserting the list from the sidecar.
    """

    return {
        "schema": "v370-escape/observation",
        "episode_id": "trace",
        "decision_id": 0,
        "frame_id": int(field.frame_base + frame),
        "source": Source.SYNTHETIC.value,
        "player": {
            "x": float(x),
            "y": float(y),
            "focus": int(focus) & 1,
            "hurtbox_half": contracts.HURTBOX_HALF_PX,
            "alive": True,
            "invulnerable": False,
            "speed_model": "border-team-lunatic-1.00d",
        },
        "frameskip": 2,
        "observed_frame_age": 0,
        "action_latency_frames": 1,
        "completeness": Completeness.COMPLETE.value,
        "coordinate_contract": dict(contracts.COORDINATE_CONTRACT),
        "bullet_pool_count": int(count),
        "hazards_ref": {"sidecar": True, "count": int(count)},
    }


@dataclass
class TraceWriter:
    out_dir: Path
    episode: str
    _records: list[dict] = None  # type: ignore[assignment]
    _positions: list[np.ndarray] = None  # type: ignore[assignment]
    _velocities: list[np.ndarray] = None  # type: ignore[assignment]
    _halves: list[np.ndarray] = None  # type: ignore[assignment]
    _offsets: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self._records = []
        self._positions = []
        self._velocities = []
        self._halves = []
        self._offsets = [0]

    def add(
        self,
        *,
        decision: dict,
        plan: dict | None,
        observation: dict,
        forecast_summary: dict,
        timing: dict,
        hazards: tuple[np.ndarray, np.ndarray, np.ndarray],
        outcome: dict,
    ) -> None:
        index = len(self._records) + 1
        positions, velocities, halves = hazards
        self._positions.append(np.asarray(positions, dtype=np.float32).reshape(-1, 2))
        self._velocities.append(np.asarray(velocities, dtype=np.float32).reshape(-1, 2))
        self._halves.append(np.asarray(halves, dtype=np.float32).reshape(-1, 2))
        self._offsets.append(self._offsets[-1] + int(positions.shape[0]))
        observation = dict(observation)
        observation["episode_id"] = self.episode
        observation["decision_id"] = index - 1
        self._records.append(
            {
                "schema": TRACE_SCHEMA,
                "episode": self.episode,
                "decision": index,
                "forecast_summary": True,
                "observation": observation,
                "forecast": forecast_summary,
                "plan": plan,
                "decision_record": decision,
                "timing": timing,
                "outcome": outcome,
                "sidecar_index": index - 1,
            }
        )

    def close(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.out_dir / f"{self.episode}.jsonl"
        with trace_path.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        positions = np.concatenate(self._positions, axis=0) if self._positions else np.zeros((0, 2), np.float32)
        velocities = np.concatenate(self._velocities, axis=0) if self._velocities else np.zeros((0, 2), np.float32)
        halves = np.concatenate(self._halves, axis=0) if self._halves else np.zeros((0, 2), np.float32)
        np.savez(
            self.out_dir / f"{self.episode}.npz",
            positions=positions.astype(np.float32),
            velocities=velocities.astype(np.float32),
            halves=halves.astype(np.float32),
            offsets=np.asarray(self._offsets, dtype=np.int64),
        )
        summary = {
            "schema": SUMMARY_SCHEMA,
            "episode": self.episode,
            "decisions": len(self._records),
            "trace": trace_path.name,
            "sidecar": trace_path.with_suffix(".npz").name,
            "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        }
        (self.out_dir / f"{self.episode}.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary


def record_episode(
    scenario: synth.Scenario,
    *,
    out_dir: Path | str,
    config: controller.ControllerConfig | None = None,
    forecast_config: forecast.ForecastConfig | None = None,
    max_decisions: int | None = None,
) -> dict:
    """Run the controller on a synthetic scenario and write a replayable trace."""

    config = config or controller.ControllerConfig()
    forecast_config = forecast_config or forecast.ForecastConfig()
    env = synth.SyntheticEnv(scenario)
    field = forecast.DangerField.from_scenario(scenario, 0, forecast_config)
    planner_obj = controller.EscapeController(config)
    writer = TraceWriter(Path(out_dir), scenario.scenario_id.replace("/", "-"))
    limit = max_decisions or scenario.horizon_frames // config.planner.frameskip
    started = time.perf_counter()
    decisions = 0
    while not env.hit and env.frame < scenario.horizon_frames and decisions < limit:
        frame_before = env.frame
        decision, diagnostics = planner_obj.decide(
            x=env.x, y=env.y, frame=env.frame,
            previous_action=env.previous_action,
            field=field, bombs=3,
        )
        plan = planner_obj.state.plan
        mask = field._active(env.frame)
        centres, halves = field._active_at(env.frame)
        velocities = field.velocities[mask]
        observation = _observation_dict(
            field, env.frame, env.x, env.y, env.previous_action[0], int(centres.shape[0])
        )
        summary = {
            "model_version": forecast.MODEL_VERSION,
            "nodes": len(field.config.time_nodes()),
            "horizon_frames": field.config.horizon_frames,
            "dropped_entities": int(field.dropped_irrelevant + field.dropped_over_cap),
            "calibrated": field.config.calibrated_frames >= field.config.horizon_frames,
            "plan_status": plan.status.value if plan is not None else None,
            "plan_ms": plan.elapsed_ms if plan is not None else None,
            "planner_mode": plan.score_mode if plan is not None else None,
        }
        env.step((int(decision.accepted_action[2]), int(decision.accepted_action[0])))
        decisions += 1
        writer.add(
            decision=contracts.to_json_dict(decision),
            plan=contracts.to_json_dict(plan.record) if plan is not None else None,
            observation=observation,
            forecast_summary=summary,
            timing={"controller_ms": decision.timing_breakdown.get("controller_ms", 0.0)},
            hazards=(np.asarray(centres), np.asarray(velocities), np.asarray(halves)),
            outcome={"hit": env.hit, "frame": env.frame, "advanced": env.frame - frame_before},
        )
    summary = writer.close()
    summary.update(
        {
            "scenario": scenario.scenario_id,
            "hit": env.hit,
            "frames": env.frame,
            "survived": (not env.hit) and env.frame >= scenario.horizon_frames,
            "elapsed_s": round(time.perf_counter() - started, 2),
        }
    )
    (Path(out_dir) / f"{scenario.scenario_id.replace('/', '-')}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# reading / attribution
# ---------------------------------------------------------------------------
def load_trace(path: Path | str) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_sidecar(path: Path | str) -> dict:
    data = np.load(Path(path))
    return {key: np.asarray(data[key]) for key in data.files}


def attribute(trace: list[dict], *, window: int = 8) -> dict:
    """Classify the last decisions before a hit (or the end of a clean run)."""

    if not trace:
        raise ValueError("empty trace")
    hit_records = [record for record in trace if record["outcome"].get("hit")]
    hit_frame = hit_records[0]["outcome"]["frame"] if hit_records else None
    start = max(0, len(trace) - window)
    window_records = trace[start:]
    intents: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for record in window_records:
        intent = record["decision_record"]["intent"]
        intents[intent] = intents.get(intent, 0) + 1
        status = (record["forecast"] or {}).get("plan_status") or "none"
        statuses[status] = statuses.get(status, 0) + 1
    no_route = any(status != "feasible" for status in statuses)
    classification = (
        "no_route_at_decision" if no_route else ("hit_while_route_feasible" if hit_frame else "clean")
    )
    return {
        "episode": trace[0]["episode"],
        "decisions": len(trace),
        "hit_frame": hit_frame,
        "classification": classification,
        "window_intents": intents,
        "window_plan_statuses": statuses,
        "window": [
            {
                "decision": record["decision"],
                "intent": record["decision_record"]["intent"],
                "action": record["decision_record"]["accepted_action"],
                "route_id": record["decision_record"].get("route_id"),
                "plan_status": (record["forecast"] or {}).get("plan_status"),
                "closure_eta": (record["plan"] or {}).get("closure_eta"),
                "min_clearance": (record["plan"] or {}).get("min_clearance"),
                "fallback": record["decision_record"].get("fallback"),
            }
            for record in window_records
        ],
    }
