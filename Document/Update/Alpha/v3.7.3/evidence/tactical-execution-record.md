# Tactical synthesis execution record

Working directory D:\WorkShop\Touhou_RL. Runtime system Python 3.12.10.
No game launched, no pixels consumed, no evaluate/old files edited.

1. `python -m pytest touhou_rl/tests/test_tactical_scenarios.py touhou_rl/tests/test_tactical_teacher.py --basetemp .tmp/tactical-synth-tests-20261101-a -q`
   Exit0, 10 passed, cache permission warning (not a test failure).
2. `python -m touhou_rl.tactical_teacher --output Document/Update/Alpha/v3.7.3`
   R0 exit0. Summary preserved as tactical-development-summary-r0.json.
3. R1 registered in tactical-revision-r1-prereg.md before execution; comparator
   breadth only behavioral factor. Teacher rules unchanged. Corridor metric fixed
   separately with documented geometrical definition.
4. `python -m touhou_rl.tactical_teacher --output Document/Update/Alpha/v3.7.3`
   R1 exit0. 72 scenario JSON traces; 36 SVG charts F1/F2/F3 (12 per family).
5. `python -m pytest touhou_rl/tests/test_tactical_scenarios.py touhou_rl/tests/test_tactical_teacher.py --basetemp .tmp/tactical-synth-tests-20261101-b -p no:cacheprovider -q`
   Final exit0, 10 passed, no warnings. Final evaluator rerun uses command (4).

## Interfaces

- tactical_scenarios: Scenario/WorldState/Bullet dataclasses, ACTIONS horizontal
  speed set {-4,-2,0,2,4}px/frame, development_scenarios(), initial_state(),
  advance(s,state,action,frames=2), replay(s,actions), fingerprint(s).
- tactical_teacher: Decision(action, option, trigger_reason, target_region,
  confidence), TacticalTeacher.decide(s,state), rollout(s), evaluate_development.
- tactical_reference: ReferenceResult(status,actions,checked,reason), solve(),
  verify_route(), scan_start_windows(), trajectory_metrics(). No teacher imports.
  feasible = positive finite-horizon witness incl F3 target; rejected_witness =
  collision of particular route, not universal failure; unknown stays unknown.

P1/P2 full gates NOT PASSED. F2 R1 narrow counterexample positive only. F3
space-recovery superiority fails. No official supervision exported. No B0/model
comparison; no independent sealed holdout or human review; no real pixels/live.
SVG is x-coordinate versus logical-frame chart, not rendered TH08 screenshot.
