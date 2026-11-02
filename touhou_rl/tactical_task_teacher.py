"""Development-only privileged rule teacher for the frozen tactical_tasks world.

No game IO, training labels, reference search or diagnostic-arm action selection.
F4 checks a straight corridor against its time-dependent occupancy; F5 filters
all actions by their actual two-frame collision result before local preferences.
The independent evaluator alone calls the existing reference functions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from . import tactical_tasks as world


@dataclass(frozen=True)
class Decision:
    frame: int
    action: int
    option: str
    trigger_reason: str
    target: tuple[int, int]
    termination: str
    failure_reason: str | None = None
    departure_frame: int | None = None
    candidates: tuple[dict, ...] = ()


def _distance(task, state):
    x = task.canonical_x(state.x)
    lo, hi = task.target
    return max(lo - x, 0, x - hi)


def _toward(task, state, speed):
    x = task.canonical_x(state.x)
    lo, hi = task.target
    delta = lo - x if x < lo else hi - x if x > hi else 0
    canonical = max(-speed, min(speed, delta))
    return int(-canonical if task.mirror else canonical)


def _corridor_clear(task, state):
    """Check one frozen straight transit policy, not a route/action search.

    There is no branching or reference witness. This is only a sufficient
    corridor test; its failure is never called task infeasibility.
    """
    while state.frame < task.horizon:
        state = world.advance(task, state, _toward(task, state, 2))
        if state.hit:
            return False
    return _distance(task, state) == 0


def _departure(task, state):
    """First clear straight corridor after safe stationary waiting, if any."""
    while state.frame < task.horizon:
        if _corridor_clear(task, state):
            return state.frame
        state = world.advance(task, state, 0)
        if state.hit:
            return None
    return None


def decide(task: world.Task, state: world.State) -> Decision:
    """Recompute from current privileged task/state, without cached sequences."""
    target = tuple(sorted(task.physical_x(x) for x in task.target))
    if not task.model_known:
        return Decision(state.frame, 0, "UNKNOWN", "unsupported_model", target,
                        "interrupted", "unsupported_model")
    if state.hit:
        return Decision(state.frame, 0, "UNKNOWN", "collision_already_observed", target,
                        "failed", "collision")
    if state.frame >= task.horizon:
        success = world.goal_reached(task, state)
        return Decision(state.frame, 0, "HOLD_STREAM", "horizon_reached", target,
                        "success" if success else "failed", None if success else "goal_or_local_bound_missed")
    candidates = []
    valid = []
    for action in world.ACTIONS:
        nxt = world.advance(task, state, action)
        local_ok = task.family != "F5" or nxt.max_displacement <= 10
        safe = not nxt.hit and local_ok
        candidates.append({"action": action, "collision_2f": nxt.hit,
                           "local_bound_ok": local_ok, "safe_2f": safe,
                           "next_x": nxt.x, "target_distance": _distance(task, nxt)})
        if safe:
            valid.append((action, nxt))
    evidence = tuple(candidates)
    if not valid:
        return Decision(state.frame, 0, "LOCAL_RECOVER", "no_safe_two_frame_candidate", target,
                        "interrupted", "no_local_witness_not_infeasibility", candidates=evidence)
    if task.family == "F5":
        preferred = _toward(task, state, 1)
        preferred_x = world.advance(task, state, preferred).x
        action, _ = min(valid, key=lambda pair: (abs(pair[1].x - preferred_x),
                                                abs(pair[0]), world.ACTIONS.index(pair[0])))
        hold_safe = next(c["safe_2f"] for c in candidates if c["action"] == 0)
        preferred_safe = next(c["safe_2f"] for c in candidates if c["action"] == preferred)
        reason = "immediate_collision_priority" if not hold_safe or not preferred_safe else "local_target_recovery"
        return Decision(state.frame, action, "LOCAL_RECOVER", reason, target,
                        "continue", candidates=evidence)
    departure = _departure(task, state)
    if departure == state.frame:
        action = _toward(task, state, 2)
        reason = "target_region_hold" if action == 0 else "time_clear_corridor_transit"
        return Decision(state.frame, action, "HOLD_STREAM" if action == 0 else "TRANSIT",
                        reason, target, "continue", departure_frame=departure, candidates=evidence)
    if departure is not None and any(a == 0 for a, _ in valid):
        return Decision(state.frame, 0, "HOLD_STREAM", "wait_for_time_clear_corridor", target,
                        "continue", departure_frame=departure, candidates=evidence)
    # Failed corridor check is not an impossibility claim. Keep the failed
    # development rollout, using only an immediate collision-filtered fallback.
    action, _ = min(valid, key=lambda pair: (_distance(task, pair[1]), abs(pair[0])))
    return Decision(state.frame, action, "LOCAL_RECOVER", "corridor_unknown_local_fallback", target,
                    "interrupted", "no_straight_corridor_witness", candidates=evidence)


def rollout(task):
    state = world.initial_state(task)
    decisions, actions = [], []
    while state.frame < task.horizon and not state.hit:
        decision = decide(task, state)
        decisions.append(asdict(decision))
        actions.append(decision.action)
        state = world.advance(task, state, decision.action)
    terminal = decide(task, state)
    return {"actions": actions, "decisions": decisions, "terminal": asdict(terminal),
            "failure": None if terminal.termination == "success" else terminal.failure_reason,
            "metrics": world.metrics(task, actions)}


def _svg(task, row):
    """Space/time occupancy and full teacher/reference trajectories, synthetic units."""
    sx = lambda x: 40 + 12 * x
    sy = lambda t: 30 + 10 * t
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="500" viewBox="0 0 640 500">',
             '<rect width="640" height="500" fill="white"/>',
             f'<text x="20" y="18" font-size="12">{task.event_id}: x synthetic units / time downward</text>']
    for frame in range(1, task.horizon + 1):
        for x in range(41):
            if world.dangerous(task, frame, x):
                parts.append(f'<rect x="{sx(x)-6}" y="{sy(frame)-5}" width="12" height="10" fill="#efb3b3"/>')
    lo, hi = sorted(task.physical_x(x) for x in task.target)
    parts.append(f'<rect x="{sx(lo)}" y="30" width="{sx(hi)-sx(lo)}" height="400" fill="#8dd59a" opacity="0.3"/>')
    for trace, color in ((row["reference_metrics"]["trace"], "#888888"),
                         (row["teacher"]["metrics"]["trace"], "#1546bd")):
        points = " ".join(f'{sx(s["x"])},{sy(s["frame"])}' for s in trace)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for d in row["teacher"]["decisions"]:
        parts.append(f'<text x="545" y="{sy(d["frame"])+4}" font-size="9">{d["option"][:5]} {d["action"]}</text>')
    parts.append('<text x="20" y="465" font-size="12">Blue: teacher; grey: independent BFS; red: danger; green: target</text>')
    parts.append(f'<text x="20" y="485" font-size="12">Terminal: {row["teacher"]["terminal"]["termination"]}; formal labels: false</text></svg>')
    return "\n".join(parts)


def evaluate_development(output):
    """Evaluation boundary: references never supply a teacher action."""
    out = Path(output)
    prereg = out / "preregistration.md"
    if not prereg.is_file():
        raise ValueError("preregistration.md must exist before evaluation")
    tasks = world.development_tasks()
    # Refuse result overwrite, preserving development failures across runs.
    if any((out / f"{t.event_id}.json").exists() for t in tasks) or (out / "summary.json").exists():
        raise ValueError("evaluation results already exist; preserve previous runs")
    summary = {}
    for task in tasks:
        teacher = rollout(task)  # Complete before the independent reference runs.
        verdict = world.verify_route(task, teacher["actions"])
        reference = world.solve(task)
        row = {"task": asdict(task), "task_sha256": hashlib.sha256(json.dumps(asdict(task), sort_keys=True).encode()).hexdigest(),
               "teacher": teacher, "teacher_verification": asdict(verdict),
               "reference": asdict(reference), "reference_metrics": world.metrics(task, reference.actions),
               "formal_training_labels": False, "split": "development"}
        (out / f"{task.event_id}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        (out / f"{task.event_id}.svg").write_text(_svg(task, row), encoding="utf-8")
        group = summary.setdefault(task.subtype, {"instances": 0, "goal_success": 0, "survived": 0,
                                                "reference_feasible": 0, "failures": []})
        group["instances"] += 1
        group["goal_success"] += teacher["metrics"]["goal_success"]
        group["survived"] += teacher["metrics"]["survived"]
        group["reference_feasible"] += reference.status == "feasible"
        if teacher["failure"]:
            group["failures"].append({"event_id": task.event_id, "reason": teacher["failure"]})
    result = {"split": "development", "formal_training_labels": False, "instances": len(tasks),
              "preregistration_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
              "teacher_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "task_source_sha256": hashlib.sha256(Path(world.__file__).read_bytes()).hexdigest(),
              "summary": summary, "holdout_status": "pending_task_contract_development_only",
              "limitations": "Privileged fixed-y toy rules only; no independent held-out result, game control, B0 comparison, formal supervision or complete P1/P2 claim."}
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_development(args.output), indent=2))


if __name__ == "__main__":
    main()
