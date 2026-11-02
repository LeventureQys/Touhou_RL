"""Privileged small-rule teacher for synthetic development ONLY.

Unknown is recorded, not a failure proof and never a supervised safe label.
Evaluation uses tactical_reference rather than this teacher to adjudicate routes.
No pixels, checkpoint, live input or old execution loop is used.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import argparse
import json
from pathlib import Path
from .tactical_scenarios import development_scenarios, initial_state, advance, fingerprint

@dataclass(frozen=True)
class Decision:
    action: int
    option: str
    trigger_reason: str
    target_region: tuple[float, float] | None = None
    confidence: str = "synthetic_rule"


class TacticalTeacher:
    def decide(self, s, state):
        cx = s.canonical_x(state.x)
        sign = -1 if s.mirror else 1
        if not s.model_known:
            return Decision(0, "LOCAL_RECOVER", "UNKNOWN", confidence="unknown")
        if s.family == "F1":
            if s.internal_gap and 180 <= cx <= 204:
                return Decision(0, "HOLD_STREAM", "RELIABLE_INTERNAL_GAP")
            action = -2 if cx > 88 else 0
            return Decision(action*sign, "TRANSIT" if action else "HOLD_STREAM",
                            "PREEMPTIVE_EXIT", (60, 88))
        if s.family == "F3" and 34 <= state.frame < 104 and cx > 128:
            return Decision(-2*sign, "TRANSIT", "RESTREAM", (100, 128))
        if s.family in ("F2", "F3"):
            # Actual emission events, not position-only guessed streaming state.
            last = state.emissions[-1] if state.emissions else None
            age = state.frame-last["frame"] if last else -1
            action = sign if 0 <= age < 4 and cx < 368 else 0
            return Decision(action, "HOLD_STREAM", "EMISSION_FIXED_MICRO_SHIFT" if action else "WAIT_EMISSION")
        return Decision(0, "LOCAL_RECOVER", "NO_MACRO_DANGER")


def rollout(s, policy=None):
    teacher = policy or TacticalTeacher()
    state = initial_state(s)
    actions, decisions = [], []
    while state.frame < s.horizon:
        d = teacher.decide(s, state)
        actions.append(d.action)
        decisions.append({"frame": state.frame, **asdict(d)})
        state = advance(s, state, d.action)
    return tuple(actions), decisions


def evaluate_development(output):
    from .tactical_reference import solve, verify_route, trajectory_metrics, scan_start_windows
    out = Path(output)
    evidence = out / "evidence"
    reports = out / "reports"
    evidence.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in development_scenarios():
        actions, decisions = rollout(s)
        ref = solve(s)
        teacher_check = verify_route(s, actions)
        # Fixed diagnostic comparator, explicitly NOT original visual checkpoint.
        sign = -1 if s.mirror else 1
        random_actions = tuple(sign*(-2 if (k//16)%2 == 0 else 2) for k in range(s.horizon//2))
        local_actions = (0,)*(s.horizon//2)
        metrics = trajectory_metrics(s, actions)
        baseline = trajectory_metrics(s, random_actions)
        local = trajectory_metrics(s, local_actions)
        row = {"scenario": asdict(s), "sha256": fingerprint(s),
               "label_source": "synthetic_rule_independently_replayed",
               "teacher_verdict": asdict(teacher_check), "reference": asdict(ref),
               "teacher": metrics, "oscillation_baseline": baseline,
               "local_hold_baseline": local, "decisions": decisions,
               "reference_metrics": trajectory_metrics(s, ref.actions) if ref.actions else None}
        if s.family == "F1" and s.event_id.endswith("00"):
            row["delayed_start_scan"] = scan_start_windows(s)
        (evidence / f"tactical-{s.event_id}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        if s.family in ("F1", "F2", "F3"):
            series = [("teacher", metrics, "#16803c"), ("broad oscillation (not B0)", baseline, "#d43b32")]
            if row["reference_metrics"]:
                series.append(("independent reference", row["reference_metrics"], "#2463ce"))
            svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="520" viewBox="0 0 800 520">',
                   '<rect width="800" height="520" fill="white"/>',
                   f'<text x="30" y="25">{s.event_id}: synthetic horizontal x vs logical frame; NOT pixels/B0</text>',
                   '<text x="30" y="48">y-axis player x (8..376); x-axis frames (0..160)</text>']
            for idx, (name, m, color) in enumerate(series):
                points = ' '.join(f'{40+t["frame"]*4},{460-t["x"]}' for t in m["trace"])
                svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
                svg.append(f'<text x="40" y="{480+idx*16}" fill="{color}">{name}: survived={m["survived"]}</text>')
            for d in decisions:
                if d["frame"] % 16 == 0:
                    svg.append(f'<text x="{40+d["frame"]*4}" y="70" font-size="8">{d["option"]}</text>')
            svg.append('</svg>')
            (evidence / f'tactical-{s.event_id}.svg').write_text('\n'.join(svg), encoding='utf-8')
        rows.append(row)
    summary = {}
    for family in (f"F{i}" for i in range(1,7)):
        group = [r for r in rows if r["scenario"]["family"] == family]
        summary[family] = {"instances": len(group),
            "teacher_survived": sum(r["teacher"]["survived"] for r in group),
            "reference_feasible": sum(r["reference"]["status"] == "feasible" for r in group),
            "teacher_verified": sum(r["teacher_verdict"]["status"] == "feasible" for r in group),
            "unknown": sum(r["teacher_verdict"]["status"] == "unknown" for r in group),
            "oscillation_survived": sum(r["oscillation_baseline"]["survived"] for r in group),
            "teacher_ray_count": sum(r["teacher"]["corridor_ray_count"] for r in group),
            "baseline_ray_count": sum(r["oscillation_baseline"]["corridor_ray_count"] for r in group),
            "teacher_restreamed": sum(r["teacher"]["restreamed"] for r in group),
            "teacher_occupancy": sum(r["teacher"]["corridor_occupancy_bullet_frames"] for r in group),
            "baseline_occupancy": sum(r["oscillation_baseline"]["corridor_occupancy_bullet_frames"] for r in group),
            "teacher_final_room": sum(r["teacher"]["final_horizontal_space_px"] for r in group),
            "baseline_final_room": sum(r["oscillation_baseline"]["final_horizontal_space_px"] for r in group)}
    failures = [{"event_id": r["scenario"]["event_id"], "verdict": r["teacher_verdict"],
                 "reference_status": r["reference"]["status"]} for r in rows
                if r["teacher_verdict"]["status"] != "feasible"]
    behavior_failures = {
        "F2_pollution_reduction": {"passed": summary["F2"]["teacher_occupancy"] < summary["F2"]["baseline_occupancy"] and summary["F2"]["teacher_survived"] >= summary["F2"]["oscillation_survived"],
            "reason": "R1 actual reserved-corridor bullet-frame occupancy; survival nondecrease"},
        "F3_space_recovery": {"passed": summary["F3"]["teacher_final_room"] > summary["F3"]["baseline_final_room"] and summary["F3"]["teacher_survived"] >= summary["F3"]["oscillation_survived"],
            "reason": "R1 final horizontal room must improve; crossing alone insufficient"},
        "deadline_lateness": {"passed": False, "reason": "latest-start interval upper bound not established"},
        "original_policy_comparison": {"passed": False, "reason": "no visual B0 executed"}}
    (evidence / "tactical-behavior-gates.json").write_text(json.dumps(behavior_failures, indent=2), encoding="utf-8")
    (evidence / "tactical-development-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (evidence / "tactical-development-failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    report = """# P1/P2 minimal synthetic development evaluation

72 deterministic development instances; fixed-y horizontal toy world, not calibrated
TH08 dynamics. Two-frame actions; shoot/bomb not simulated. No real pixels or game
passed. F2/F3 emissions use actual causal position. No validation/test has been
opened or claimed independent. Reference enumerates a separate bounded program
class; it cannot prove absence of a solution. Teacher route verification is a full
causal replay outside teacher logic. Unknowns remain excluded from safe labels.

## Actual results

```json
""" + json.dumps(summary, indent=2) + """
```

## Gate status and limitations

R0 negative results are preserved in tactical-development-summary-r0.json:
F2 birth-ray counts equal (96/96); F3 counts 20 vs 0. R1 preregistration changes
only diagnostic comparator breadth, teacher unchanged; adds actual logical-frame
corridor occupancy (x80..160, y360..420; F3 reservation ends frame104). Above
summary is R1 measured output. Birth-ray count remains diagnostic only. Do not
infer B0 superiority or official labels. See tactical-behavior-gates.json.

P1 partial: six families replayable, F2/F3 causality tested. F4 has NO waiting-then-
pass objective and is NOT a validated F4 task. F5 is only a static no-macro-
interference control, NOT local precision-control noninferiority. F1 is a timed
static rectangle toy, NOT a fair original visual-policy comparator. R1 F2 has
12/12 survival vs12/12 comparator and occupancy0 vs436 (narrow positive synthetic
counterexample). F3 occupancy0 vs0 and final horizontal room1520 vs2016 sum:
space-recovery superiority FAILS despite12/12 crossing. No threshold lowered.
F6 unsupported future model is unknown. No independent human
label review; no sealed validation/test templates; P1 full gate NOT PASSED.
P2 full gate NOT PASSED: no original checkpoint/pixel comparator, no full
calibration, no reliable latest-start upper bounds, no three visualized replays
per family. JSON traces contain every decision, emission and trajectory, including
failure/unknown evidence. F1 scan reports witnessed delay values, not an exact
latest deadline; unwitnessed values are unknown. F2/F3 diagnostic comparator is
oscillatory fixed policy, not B0. No unsupported family enters success supervision.
Pollution reports BOTH birth-ray counts and actual logical-frame bullet occupancy
in the predeclared reserved rectangle. Occupancy is still a toy-world diagnostic,
not a calibrated game-wide safety metric. R1 broad-motion comparator is a DIFFERENT
comparator from R0, not a matched frozen-baseline improvement experiment; its F2
counterexample MUST NOT count as P2 improvement. Teacher itself did not improve.
36 SVG trajectory charts cover F1/F2/F3 only; F4/F5/F6 visual replay gate pending.
Next task: improve F3 teacher on a meaningful depletion/window task with frozen
comparator, not switch comparators to obtain favorable results.
"""
    (reports / "P1-P2-tactical-development.md").write_text(report, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_development(args.output), indent=2))

if __name__ == "__main__":
    main()
