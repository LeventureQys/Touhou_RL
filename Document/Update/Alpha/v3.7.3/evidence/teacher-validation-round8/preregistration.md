# Round8 preregistration — frozen before execution

2026-11-01; synthetic-only, no game, labels, checkpoint or training.

## Fixed intervention and split

New independent `tactical_teacher_validation.py` only; existing TacticalTeacher,
scenario/reference modules remain unchanged. Teacher source SHA256 and all new
world definitions are frozen in `manifest.json` BEFORE any rollout. Manifest
includes complete test definitions but test worlds are NEVER instantiated/evaluated
by the evaluator. Mirrors always share template_group and split.

Development: three parameter/structure groups per F1/F2/F3, each mirrored (18).
Validation: four disjoint parameter/structure groups per family, mirrored (24).
Test: four separately named and parametrized groups per family, mirrored (24),
manifest-only. No overlap in complete group parameter signatures across splits.
Validation definitions are locked before teacher/reference evaluation; no tuning.

F1 varies coverage span, hazard timing and internal gap placement/width. F2 varies
actual-player-aimed emitter origin, cadence/speed and optional two-shot bursts;
each shot aims at actual x at its birth, not a frozen future. F3 retains R2
`depletion_pressure=True`, early-left/late-right pressure, actual aimed emissions
and finite crossing goal, with independent start/threshold/window parameter groups.
This explicitly probes old teacher geometry constants; failures must be retained.

## Frozen comparators and finite reference

Unmodified teacher; fixed R1 broad oscillation: canonical -2/+2 switching every
16 decisions (32 logical frames). Also fixed hold, premature left-cross and
late (frame112) left-cross controls. No comparator replacement after validation.
Measure full-horizon survival, full replay witness verdict, first collision,
actual corridor bullet-frame occupancy, birth-ray count, final horizontal room,
F3 goal and emission/decision traces. Report family and split separately.
Independent bounded open-loop program generator uses no teacher actions/options;
fixed maximum 223 schedules (hold; 11 starts x 4 directions x 5 durations; two
cadence micro-shifts), stopping at first witness. Starts and durations are frozen
in source before manifest registration. All 42 non-test events are evaluated and
all F1/F3 events scanned. Development F3 uses exactly R2 thresholds/window;
validation F3 explicitly varies crossing-window and threshold geometry in this
new module only, retaining the R2 depletion-pressure mechanism. This is a limited
synthetic structural/parameter holdout, not new game dynamics.
Full two-frame causal replay certifies positive witnesses only. Search misses are
unknown, never infeasible. Reference covers only this finite synthetic horizon.

## Delays and lateness

F1/F3: scan EVERY even delay 0..horizon inclusive under TWO frozen prefixes:
repeat initial current teacher action; hold (0). Reference suffix search is
independent, and each positive includes complete two-frame actions and is fully
replayed from the original initial state with the prefix. No binary search or
monotonicity assumption. Save all unknowns and reasons. Witnessed windows are
consecutive seen feasible delays; their endpoints are not exact deadlines.
Late-start diagnostic compares teacher sustained target progress (>=8px over
6 logical frames) ONLY to greatest already witnessed delay for that prefix.
No exact upper bound/no exact <=10% lateness gate passed; no witness => unknown.

## Outputs and nonclaims

All 42 development/validation events saved, including failures/unknown; at least
three validation SVGs per F1/F2/F3 (all validation events rendered), showing teacher,
fixed broad comparator, reference, options/targets and witnessed-only deadlines.
Full P1/P2 gate remains NOT PASSED: no B0 pixels, real calibration, F4/F5/F6 or
reviewed precise labels. No official labels or training; test remains sealed.
