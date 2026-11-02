"""Reference-solver tests: proofs vs bounded search, brute-force cross-check."""

from __future__ import annotations

import pytest

from touhou_rl.escape import families, reference, synth


def _tiny(hazards=(), *, start=(192.0, 384.0), horizon=4, frameskip=2):
    return synth.Scenario("tiny/0", "TEST", 0, start, tuple(hazards),
                          horizon_frames=horizon, hold_frames=0, frameskip=frameskip)


def test_bruteforce_finds_what_the_solver_finds():
    """Exhaustive enumeration is the yardstick on tiny instances."""
    cases = {
        "open": _tiny(),
        "one_bullet": _tiny([synth.HazardSpec((196.0, 384.0), (0.0, 0.0), (2.0, 2.0))], horizon=4),
        "wall": _tiny(tuple(synth.HazardSpec((x, 300.0), (0.0, 0.0), (2.0, 2.0))
                            for x in range(4, 380, 4)), horizon=4),
    }
    for name, scenario in cases.items():
        exact = reference.bruteforce(scenario, decisions=2)
        result = reference.solve(scenario, node_budget=5000)
        if exact["exists"]:
            assert result.status is reference.SolutionStatus.FEASIBLE, name
            outcome = synth.simulate(scenario, list(result.actions))
            assert outcome["survived"] is True
        else:
            assert result.status is not reference.SolutionStatus.FEASIBLE, name


def test_bruteforce_detects_impossible_start():
    scenario = _tiny([synth.HazardSpec((192.0, 384.0), (0.0, 0.0), (2.0, 2.0))])
    exact = reference.bruteforce(scenario, decisions=1)
    assert exact["survivors"] == 0


def test_certified_block_requires_a_proof():
    open_field = _tiny()
    assert reference.certify_blocked(open_field) is None
    filled = families.build("SC-12", 0)
    proof = reference.certify_blocked(filled)
    assert proof is not None
    assert proof.status is reference.SolutionStatus.NO_SOLUTION_CERTIFIED
    assert "area_covered" in proof.reason


def test_a_failed_search_is_not_a_proof():
    scenario = families.edge_instances()["E-06"]  # narrow gap + rising floor
    result = reference.solve(scenario, node_budget=300)
    assert result.status is reference.SolutionStatus.BUDGET_EXHAUSTED


def test_budget_exhaustion_is_reported_for_a_trivial_budget():
    result = reference.solve(families.edge_instances()["E-12"], node_budget=1)
    assert result.status is reference.SolutionStatus.BUDGET_EXHAUSTED
    assert "node budget" in result.reason


def test_unsupported_family_is_unknown_not_blocked():
    result = reference.solve(families.build("SC-13", 0))
    assert result.status is reference.SolutionStatus.MODEL_UNKNOWN
    assert "unsupported" in result.reason


def test_feasible_routes_always_re_simulate():
    for scenario_id in ("E-01", "E-05", "E-09", "SC-15/0"):
        if scenario_id.startswith("SC-"):
            family, seed = scenario_id.split("/")
            scenario = families.build(family, int(seed))
        else:
            scenario = families.edge_instances()[scenario_id]
        result = reference.solve(scenario, node_budget=5000)
        if result.status is reference.SolutionStatus.FEASIBLE:
            outcome = reference.validate(scenario, result.actions)
            assert outcome["survived"], scenario_id


def test_mirror_instance_stays_mirrored():
    base = families.edge_instances()["E-20"]
    mirrored = synth.Scenario(
        "E-20m", "SC-14", 0, (448.0 - base.player_start[0], base.player_start[1]),
        tuple(
            synth.HazardSpec((448.0 - h.position[0], h.position[1]), (-h.velocity[0], h.velocity[1]), h.half)
            for h in base.hazards
        ),
        horizon_frames=base.horizon_frames, hold_frames=0,
    )
    left = reference.solve(base, node_budget=2000)
    right = reference.solve(mirrored, node_budget=2000)
    assert left.status is right.status is reference.SolutionStatus.FEASIBLE
    for hazard, mirror in zip(base.hazards, mirrored.hazards):
        assert mirror.position[0] == pytest.approx(448.0 - hazard.position[0])
        assert mirror.velocity[0] == pytest.approx(-hazard.velocity[0])
