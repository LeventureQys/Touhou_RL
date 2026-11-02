from dataclasses import replace
from touhou_rl.tactical_scenarios import development_scenarios, initial_state
from touhou_rl.tactical_teacher import TacticalTeacher, rollout
from touhou_rl.tactical_reference import solve, verify_route, scan_start_windows, trajectory_metrics


def test_reference_budget_never_negative_proof():
    s = development_scenarios()[0]
    r = solve(s, max_programs=0)
    assert r.status == "unknown"
    assert not r.actions
    scan = scan_start_windows(s, delays=(0, 8, 16), max_programs=0)
    assert [r["delay"] for r in scan] == [0, 8, 16]
    assert all(r["status"] == "unknown" for r in scan)


def test_unknown_model_not_success_despite_survival():
    s = next(s for s in development_scenarios() if s.family == "F6")
    actions, decisions = rollout(s)
    assert all(d["confidence"] == "unknown" for d in decisions)
    assert solve(s).status == "unknown"
    assert verify_route(s, actions).status == "unknown"


def test_independent_reference_checks_full_route_and_rejects_collision():
    s = development_scenarios()[0]
    assert verify_route(s, (0,)*80).status == "rejected_witness"
    assert verify_route(s, (-2,)).status == "unknown"
    result = solve(s)
    assert result.status == "feasible"
    assert verify_route(s, result.actions).status == "feasible"


def test_f1_teacher_preempts_wall_and_preserves_internal_gap():
    for s in development_scenarios():
        if s.family != "F1":
            continue
        actions, _ = rollout(s)
        assert verify_route(s, actions).status == "feasible"
    s = next(s for s in development_scenarios() if s.family == "F1" and s.internal_gap and s.start_x == 192)
    d = TacticalTeacher().decide(s, initial_state(s))
    assert d.action == 0 and d.option == "HOLD_STREAM"


def test_f3_window_actual_migration_and_mirror():
    worlds = [s for s in development_scenarios() if s.family == "F3"]
    for s in worlds:
        actions, ds = rollout(s)
        assert any(d["trigger_reason"] == "RESTREAM" for d in ds)
        m = trajectory_metrics(s, actions)
        assert m["restreamed"]


def test_r2_f3_depletion_requires_window_and_teacher_unchanged():
    for original in development_scenarios():
        if original.family != "F3":
            continue
        s = replace(original, depletion_pressure=True)
        sign = -1 if s.mirror else 1
        still = (0,)*80
        early = tuple(-2*sign if k < 30 else 0 for k in range(80))
        late = tuple(-2*sign if 2*k >= 104 else 0 for k in range(80))
        assert verify_route(s, still).status == "rejected_witness"
        assert verify_route(s, early).status == "rejected_witness"
        assert verify_route(s, late).status == "rejected_witness"
        # No pressure-aware teacher branch: exact same policy actions in both worlds.
        assert rollout(s)[0] == rollout(original)[0]
        assert verify_route(s, rollout(s)[0]).status == "feasible"

