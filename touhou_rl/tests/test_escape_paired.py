"""Pairing tests for D9 (privileged survey, classification, aggregation)."""

from __future__ import annotations

from touhou_rl.escape import families, paired, probe, synth


def test_action_survives_inside_the_arena():
    scenario = synth.Scenario("empty/0", "TEST", 0, (192.0, 384.0), (),
                              horizon_frames=40, hold_frames=0)
    result = paired.action_survives(scenario, 0, (0, 3))
    assert result["survives"] is True
    assert result["frames"] > 0


def test_survey_counts_all_actions_and_reports_the_best():
    scenario = families.edge_instances()["E-09"]
    report = paired.survey(scenario, 0)
    assert len(report["per_action"]) == 18
    assert 0 <= report["survivable_actions"] <= 18
    assert report["best_min_clearance"] >= -1.0


def test_classification_covers_the_four_cases():
    assert paired.classify(True, True) == "safe_choice"
    assert paired.classify(True, False) == "lucky"
    assert paired.classify(False, True) == "missed"
    assert paired.classify(False, False) == "doomed"


def test_probe_with_survey_emits_a_paired_summary():
    scenario = families.edge_instances()["E-09"]
    document = probe.run_probe(
        probe.ReplaySource(scenario), scenario, config=probe.ProbeConfig(decisions=4, survey=True)
    )
    summary = document["paired"]
    assert summary["decisions"] == 4
    assert sum(summary["classified"].values()) == 4
    assert summary["safe_choice_rate"] is not None
    for row in document["rows"]:
        assert row["pair"]["classification"] in {"safe_choice", "lucky", "missed", "doomed"}
        assert row["pair"]["survivable_actions"] >= 0


def test_aggregation_ignores_rows_without_a_pair():
    rows = [
        {"pair": {"classification": "safe_choice"}},
        {"pair": {"classification": "missed"}},
        {"pair": None},
        {"status": "unreadable"},
    ]
    summary = paired.summarise(rows)
    assert summary["decisions"] == 2
    assert summary["classified"]["safe_choice"] == 1
    assert summary["classified"]["missed"] == 1
