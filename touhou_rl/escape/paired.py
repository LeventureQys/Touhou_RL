"""Paired decision statistics (D9): was the shadow's action survivable, and was
a survivable action available at all?

The pair is deliberately asymmetric:

* the **chosen** side is what the shadow produced from a normal observation;
* the **teacher** side is a privileged survey of all 18 actions, available only
  because this is an offline probe.  It answers "would a perfect player have had
  a survivable move here", which is the counterfactual the acceptance's D9 asks
  for.  The survey never reaches the controller.

Both sides are measured with the same 24-frame "hold this action" rollout, so the
comparison is apples to apples even though the rollout itself is a crude proxy
for a full policy.
"""

from __future__ import annotations

from . import planner, synth

#: Frames of the hold-action rollout used for both sides of the pair.
PAIR_FRAMES = 24


def action_survives(scenario: synth.Scenario, frame: int, action: tuple[int, int], *, frames: int = PAIR_FRAMES) -> dict:
    """Roll ``action`` forward from (start position, ``frame``) and report survival."""

    env = synth.SyntheticEnv(scenario)
    env.frame = int(frame)
    env.x, env.y = scenario.player_start
    env.previous_action = (0, 0)
    env.hit = False
    env.min_clearance = float("inf")
    env._check(int(frame))
    decisions = max(1, frames // 2)
    for _ in range(decisions):
        if env.hit:
            break
        env.step(action)
    return {
        "survives": not env.hit,
        "min_clearance": round(float(env.min_clearance), 4),
        "frames": env.frame - int(frame),
    }


def survey(scenario: synth.Scenario, frame: int, *, frames: int = PAIR_FRAMES) -> dict:
    """Privileged survey of all 18 actions at this decision."""

    results = {
        action: action_survives(scenario, frame, action, frames=frames) for action in planner.ACTIONS
    }
    survivable = [action for action, result in results.items() if result["survives"]]
    best = max(results.values(), key=lambda item: item["min_clearance"])
    return {
        "survivable_actions": len(survivable),
        "survivable": [[int(a[0]), int(a[1])] for a in survivable],
        "best_min_clearance": best["min_clearance"],
        "per_action": {
            f"{int(action[0])}-{int(action[1])}": result["survives"]
            for action, result in results.items()
        },
    }


def classify(chosen_survives: bool, any_survives: bool) -> str:
    if chosen_survives and any_survives:
        return "safe_choice"
    if chosen_survives and not any_survives:
        return "lucky"  # should not happen: the survey includes the chosen action
    if not chosen_survives and any_survives:
        return "missed"
    return "doomed"


def summarise(rows: list[dict]) -> dict:
    """Aggregate the paired rows of one probe run."""

    classified: dict[str, int] = {}
    readable = 0
    for row in rows:
        pair = row.get("pair") or {}
        if "classification" not in pair:
            continue
        readable += 1
        classified[pair["classification"]] = classified.get(pair["classification"], 0) + 1
    total = sum(classified.values())
    return {
        "decisions": readable,
        "classified": classified,
        "safe_choice_rate": round(classified.get("safe_choice", 0) / total, 4) if total else None,
        "missed_rate": round(classified.get("missed", 0) / total, 4) if total else None,
        "doomed_rate": round(classified.get("doomed", 0) / total, 4) if total else None,
        "note": "teacher survey is privileged and offline-only; it never feeds the controller",
    }
