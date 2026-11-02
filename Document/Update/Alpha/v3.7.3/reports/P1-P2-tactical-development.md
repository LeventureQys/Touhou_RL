# P1/P2 minimal synthetic development evaluation

72 deterministic development instances; fixed-y horizontal toy world, not calibrated
TH08 dynamics. Two-frame actions; shoot/bomb not simulated. No real pixels or game
passed. F2/F3 emissions use actual causal position. No validation/test has been
opened or claimed independent. Reference enumerates a separate bounded program
class; it cannot prove absence of a solution. Teacher route verification is a full
causal replay outside teacher logic. Unknowns remain excluded from safe labels.

## Actual results

```json
{
  "F1": {
    "instances": 12,
    "teacher_survived": 12,
    "reference_feasible": 12,
    "teacher_verified": 12,
    "unknown": 0,
    "oscillation_survived": 0,
    "teacher_ray_count": 0,
    "baseline_ray_count": 0,
    "teacher_restreamed": 0,
    "teacher_occupancy": 0,
    "baseline_occupancy": 0,
    "teacher_final_room": 1360.0,
    "baseline_final_room": 672.0
  },
  "F2": {
    "instances": 12,
    "teacher_survived": 12,
    "reference_feasible": 12,
    "teacher_verified": 12,
    "unknown": 0,
    "oscillation_survived": 12,
    "teacher_ray_count": 96,
    "baseline_ray_count": 86,
    "teacher_restreamed": 0,
    "teacher_occupancy": 0,
    "baseline_occupancy": 436,
    "teacher_final_room": 1824.0,
    "baseline_final_room": 672.0
  },
  "F3": {
    "instances": 12,
    "teacher_survived": 12,
    "reference_feasible": 12,
    "teacher_verified": 12,
    "unknown": 0,
    "oscillation_survived": 12,
    "teacher_ray_count": 20,
    "baseline_ray_count": 0,
    "teacher_restreamed": 12,
    "teacher_occupancy": 0,
    "baseline_occupancy": 0,
    "teacher_final_room": 1520.0,
    "baseline_final_room": 2016.0
  },
  "F4": {
    "instances": 12,
    "teacher_survived": 12,
    "reference_feasible": 12,
    "teacher_verified": 12,
    "unknown": 0,
    "oscillation_survived": 12,
    "teacher_ray_count": 0,
    "baseline_ray_count": 0,
    "teacher_restreamed": 0,
    "teacher_occupancy": 0,
    "baseline_occupancy": 0,
    "teacher_final_room": 2144.0,
    "baseline_final_room": 672.0
  },
  "F5": {
    "instances": 12,
    "teacher_survived": 12,
    "reference_feasible": 12,
    "teacher_verified": 12,
    "unknown": 0,
    "oscillation_survived": 12,
    "teacher_ray_count": 0,
    "baseline_ray_count": 0,
    "teacher_restreamed": 0,
    "teacher_occupancy": 0,
    "baseline_occupancy": 0,
    "teacher_final_room": 2144.0,
    "baseline_final_room": 672.0
  },
  "F6": {
    "instances": 12,
    "teacher_survived": 10,
    "reference_feasible": 0,
    "teacher_verified": 0,
    "unknown": 12,
    "oscillation_survived": 12,
    "teacher_ray_count": 96,
    "baseline_ray_count": 86,
    "teacher_restreamed": 0,
    "teacher_occupancy": 0,
    "baseline_occupancy": 436,
    "teacher_final_room": 2144.0,
    "baseline_final_room": 672.0
  }
}
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
Pollution reports BOTH birth-ray counts and logical-frame bullet occupancy in
the reserved rectangle. R1 comparator differs from R0; favorable R1 F2 findings
are a broad-motion diagnostic counterexample, NOT frozen-baseline P2 improvement.
Teacher unchanged. 36 SVG charts cover F1/F2/F3; other visualization gates pending.

## R2 isolated F3 task-pressure revision

Preregistered in evidence/tactical-r2-prereg.md; all R1 evidence retained untouched.
Only opt-in F3 world pressure changed: canonical x<=180 unsafe before frame34;
x>=180 unsafe from frame104 onward. Same original initial worlds, same state-
dependent emissions, unchanged teacher, fixed R1 comparator. This tests a real
within-toy-world crossing window rather than merely accepting arbitrary survival.

Actual 12-world results: teacher12 feasible, independent reference12 feasible,
fixed R1 broad baseline0, stationary0, premature crossing0, late crossing0.
Individual JSON and SVG traces are in evidence/tactical-r2/. This is a narrow
synthetic task-validity result, not P2 superiority on frozen R1 worlds or real
pixels. Full P1/P2 gates remain NOT PASSED. Teacher rule was not revised.

`python -m pytest touhou_rl/tests/test_tactical_scenarios.py touhou_rl/tests/test_tactical_teacher.py --basetemp .tmp/tactical-synth-tests-20261101-r2 -p no:cacheprovider -q`
=>11 passed in0.11s, exit0. R2 inline Python evaluation exit0. Tests verify that
R2 teacher actions equal R1 teacher actions and the three timing controls fail.
Next task: independent F3 template groups and reviewed timing labels; F4 objective
and F5 meaningful local hazard controls. No further factor changes this round.
