# Tasks round1 preregistration (before implementation/evaluation)

2026-11-01. Scope: independent fixed-y horizontal synthetic world only; no existing module edits, game control, labels, training, held-out evaluation or P1/P2 completion claim.

## Frozen task contract
- World x=0..40, y=400 fixed; 2 logical frames/decision, action speeds {-2,-1,0,1,2} units/frame; horizon40. Bounds clamp. Hazards checked EVERY logical frame after motion. Units are synthetic, not calibrated pixels.
- 24 development instances: F4 6 wait/pass +6 immediate-exit; F5 6 narrow-gap +6 scattered-hazard. Each subtype has 3 offsets(-1,0,1), each mirrored. Mirrors share template group; no validation/test created/opened.
- F4 start x10, final target[30,34], gate[18,22]. Wait/pass gate blocked frames1..16, then open; immediate-exit gate open through10, blocked11..40. Success requires collision-free full horizon AND final target; survival at start is NOT success.
- F5 start16. Narrow: frames12..20 all positions outside[18,22] dangerous; final target[20,22]. Scattered: [15,17] dangerous frames6..10; [19,21] frames14..18; final target[24,26]. Both require collision-free horizon, final target AND maximum displacement <=10. Large migration is not local recovery. Offsets shift all x geometry; mirrors reflect around20.
- Unknown model, incomplete replay or budget exhaustion => unknown, NEVER infeasible. A collision/missed goal rejects only that supplied witness.

## Frozen diagnostic arms (not B0 / original policy)
- hold: action0 throughout.
- immediate_transit: canonical action+2 until x>=32, then hold (all families).
- wait_then_transit: F4 hold until frame16 then action+2 to32.
- local_recover: F5 canonical action+1 toward20(narrow) or24(scattered), then hold.
- Independent reference: breadth-first finite time-position search, action order(0,-1,1,-2,2), budget100000 state expansions; replay recovered positive route from initial world. Search does not import existing tactical teacher/reference and is not a policy comparator.

## Metrics/gates (report all, no comparator replacement)
Report per subtype and each arm: full-horizon survival, goal success, hit, final x, max displacement, first target arrival, action/position trace. Reference should witness24/24. F4 wait/pass must show wait_then success6/6 and immediate failure6/6; immediate-exit must show immediate success6/6 and wait_then failure6/6. F5 local_recover should succeed12/12; hold must not receive success; large transit should not receive success. These are task-construction checks, NOT learned-policy benefit, formal noninferiority, or full P1/P2 gates. Preserve any failure unchanged.

## Planned commands
`python -m touhou_rl.tactical_tasks --output Document/Update/Alpha/v3.7.3/evidence/tasks-round1` (development-only, no split option).
`python -m pytest tests/test_tactical_tasks.py -q -p no:cacheprovider --basetemp <fresh unique directory>`.
Also run existing scenario/teacher tests with a second fresh basetemp; record commands and exit codes.
