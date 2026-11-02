"""Development-only F4/F5 toy tasks; no game, training or existing teacher imports.

Fixed y=400, synthetic x units (not calibrated pixels), two-frame decisions.
Success is a replayed task goal, NEVER mere survival. Finite search returns only
positive witnesses or unknown; a rejected route is not an infeasibility proof.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

ACTIONS = (0, -1, 1, -2, 2)


@dataclass(frozen=True)
class Task:
    event_id: str
    family: str
    subtype: str
    template_group: str
    seed: int
    offset: int = 0
    mirror: bool = False
    split: str = "development"
    model_known: bool = True
    horizon: int = 40

    def __post_init__(self):
        if self.split != "development":
            raise ValueError("only development tasks supported")
        if (self.family, self.subtype) not in (("F4", "wait_pass"),
                ("F4", "immediate_exit"), ("F5", "narrow_gap"),
                ("F5", "scattered")):
            raise ValueError("unsupported task")
        if self.offset not in (-1, 0, 1) or self.horizon != 40:
            raise ValueError("outside preregistered geometry/horizon")

    def canonical_x(self, x):
        return (40 - x if self.mirror else x) - self.offset

    def physical_x(self, x):
        x += self.offset
        return 40 - x if self.mirror else x

    @property
    def start_x(self):
        return self.physical_x(10 if self.family == "F4" else 16)

    @property
    def target(self):
        return (30, 34) if self.family == "F4" else ((20, 22) if self.subtype == "narrow_gap" else (24, 26))


@dataclass(frozen=True)
class State:
    frame: int
    x: int
    hit: bool = False
    max_displacement: int = 0
    first_target_frame: int | None = None


@dataclass(frozen=True)
class Verdict:
    status: str
    reason: str
    actions: tuple[int, ...] = ()
    checked: int = 0


def development_tasks():
    return tuple(Task(f"dev-task-{subtype}-{i:02d}", family, subtype,
                      f"dev-task-{subtype}-offset-{offset}", 73730 + i,
                      offset, bool(i % 2))
                 for family, subtype in (("F4", "wait_pass"), ("F4", "immediate_exit"),
                                         ("F5", "narrow_gap"), ("F5", "scattered"))
                 for i, offset in enumerate((-1, -1, 0, 0, 1, 1)))


def initial_state(task):
    return State(0, task.start_x)


def dangerous(task, frame, x):
    if hasattr(task, "dangerous"):
        return task.dangerous(frame, x)
    cx = task.canonical_x(x)
    if task.subtype == "wait_pass":
        return frame <= 16 and 18 <= cx <= 22
    if task.subtype == "immediate_exit":
        return frame >= 11 and 18 <= cx <= 22
    if task.subtype == "narrow_gap":
        return 12 <= frame <= 20 and not 18 <= cx <= 22
    return (6 <= frame <= 10 and 15 <= cx <= 17) or (14 <= frame <= 18 and 19 <= cx <= 21)


def advance(task, state, action):
    if type(action) is not int or action not in ACTIONS:
        raise ValueError("action must be an integer speed in -2..2")
    x, hit, displacement, arrival = state.x, state.hit, state.max_displacement, state.first_target_frame
    for frame in range(state.frame + 1, min(state.frame + 2, task.horizon) + 1):
        x = min(40, max(0, x + action))
        hit = hit or dangerous(task, frame, x)
        displacement = max(displacement, abs(x - task.start_x))
        if arrival is None and task.target[0] <= task.canonical_x(x) <= task.target[1]:
            arrival = frame
    return State(min(state.frame + 2, task.horizon), x, hit, displacement, arrival)


def replay(task, actions):
    state = initial_state(task)
    trace = [asdict(state)]
    for action in actions:
        if state.frame == task.horizon:
            break
        state = advance(task, state, action)
        trace.append({**asdict(state), "action": action, "actual_hold": 2})
    return state, trace


def goal_reached(task, state):
    return (state.frame == task.horizon and not state.hit
            and task.target[0] <= task.canonical_x(state.x) <= task.target[1]
            and (task.family != "F5" or state.max_displacement <= 10))


def verify_route(task, actions):
    actions = tuple(actions)
    if not task.model_known:
        return Verdict("unknown", "unsupported_model")
    state, _ = replay(task, actions)
    if state.frame != task.horizon:
        return Verdict("unknown", "incomplete_horizon")
    if not goal_reached(task, state):
        return Verdict("rejected_witness", "collision_or_goal_missed_not_infeasibility")
    return Verdict("feasible", "full_horizon_goal_replayed", actions)


def solve(task, max_expansions=100000):
    """Independent BFS, not teacher-guided; no failure is an unsolvable verdict."""
    if type(max_expansions) is not int or max_expansions < 0:
        raise ValueError("nonnegative integer search budget required")
    if not task.model_known:
        return Verdict("unknown", "unsupported_model")
    queue = deque([(initial_state(task), ())])
    seen = set()
    checked = 0
    while queue and checked < max_expansions:
        state, actions = queue.popleft()
        checked += 1
        if goal_reached(task, state):
            result = verify_route(task, actions)
            return Verdict(result.status, result.reason, result.actions, checked)
        if state.frame == task.horizon:
            continue
        for action in ACTIONS:
            nxt = advance(task, state, action)
            if nxt.hit or (task.family == "F5" and nxt.max_displacement > 10):
                continue
            key = (nxt.frame, nxt.x, nxt.max_displacement)
            if key not in seen:
                seen.add(key)
                queue.append((nxt, actions + (action,)))
    return Verdict("unknown", "bounded_search_no_witness", checked=checked)


def diagnostic_actions(task, arm):
    if arm not in ("hold", "immediate_transit", "wait_then_transit", "local_recover"):
        raise ValueError("unknown frozen diagnostic arm")
    if arm == "local_recover" and task.family != "F5":
        raise ValueError("local diagnostic is F5 only")
    if arm == "wait_then_transit" and task.family != "F4":
        raise ValueError("wait diagnostic is F4 only")
    state = initial_state(task)
    actions = []
    sign = -1 if task.mirror else 1
    while state.frame < task.horizon:
        cx = task.canonical_x(state.x)
        if arm == "hold" or (arm == "wait_then_transit" and state.frame < 16):
            action = 0
        elif arm == "local_recover":
            target = 20 if task.subtype == "narrow_gap" else 24
            action = sign if cx < target else 0
        else:
            action = 2 * sign if cx < 32 else 0
        actions.append(action)
        state = advance(task, state, action)
    return tuple(actions)


def metrics(task, actions):
    state, trace = replay(task, actions)
    return {"survived": state.frame == task.horizon and not state.hit,
            "goal_success": goal_reached(task, state), **asdict(state), "trace": trace}


def evaluate_development(output):
    out = Path(output)
    prereg = out / "preregistration.md"
    if not prereg.is_file():
        raise ValueError("preregistration.md must exist before evaluation")
    rows, summary = [], {}
    for task in development_tasks():
        reference = solve(task)
        arms = ["hold", "immediate_transit", "wait_then_transit" if task.family == "F4" else "local_recover"]
        row = {"task": asdict(task), "task_sha256": hashlib.sha256(json.dumps(asdict(task), sort_keys=True).encode()).hexdigest(),
               "reference": asdict(reference), "reference_metrics": metrics(task, reference.actions),
               "diagnostics": {arm: metrics(task, diagnostic_actions(task, arm)) for arm in arms}}
        rows.append(row)
        (out / f"{task.event_id}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        group = summary.setdefault(task.subtype, {"instances": 0, "reference_feasible": 0, "arms": {}})
        group["instances"] += 1
        group["reference_feasible"] += reference.status == "feasible"
        for arm, m in row["diagnostics"].items():
            counts = group["arms"].setdefault(arm, {"survived": 0, "goal_success": 0})
            for name in counts:
                counts[name] += m[name]
    checks = {"reference_24_of_24": sum(g["reference_feasible"] for g in summary.values()) == 24,
              "f4_wait_pass": summary["wait_pass"]["arms"]["wait_then_transit"]["goal_success"] == 6 and summary["wait_pass"]["arms"]["immediate_transit"]["goal_success"] == 0,
              "f4_immediate_counterexample": summary["immediate_exit"]["arms"]["immediate_transit"]["goal_success"] == 6 and summary["immediate_exit"]["arms"]["wait_then_transit"]["goal_success"] == 0,
              "f5_local_recovery": all(summary[s]["arms"]["local_recover"]["goal_success"] == 6 for s in ("narrow_gap", "scattered")),
              "hold_never_goal": all(g["arms"]["hold"]["goal_success"] == 0 for g in summary.values()),
              "f5_large_transit_never_goal": all(summary[s]["arms"]["immediate_transit"]["goal_success"] == 0 for s in ("narrow_gap", "scattered"))}
    result = {"split": "development", "formal_training_labels": False, "preregistration_sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
              "summary": summary, "construction_checks": checks,
              "limitations": "fixed-y synthetic units; no B0, learned teacher, held-out test, formal noninferiority or complete P1/P2 claim"}
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_development(args.output), indent=2))


if __name__ == "__main__":
    main()
