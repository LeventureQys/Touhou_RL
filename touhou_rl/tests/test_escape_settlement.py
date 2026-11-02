"""Settlement-table tests: the machine-readable gates must match ``gates()``."""

from __future__ import annotations

from touhou_rl.escape import evaluate


def _row(scenario_id, family, *, survived, baseline_survived, expected=None, reference=None):
    return {
        "scenario_id": scenario_id,
        "family": family,
        "expected": expected,
        "reference": reference,
        "horizon": 60,
        "subject": {
            "scenario_id": scenario_id, "arm": "subject", "survived": survived, "hit": not survived,
            "hit_frame": None, "frames": 60, "min_clearance": 1.0,
            "entered_pocket": False, "evacuate_frame": 0, "pocket_frames": 0,
            "final_status": "feasible", "final_reason": "ok",
        },
        "baseline": {
            "scenario_id": scenario_id, "arm": "baseline", "survived": baseline_survived,
            "hit": not baseline_survived, "hit_frame": None, "frames": 60, "min_clearance": 0.5,
            "entered_pocket": False, "evacuate_frame": None, "pocket_frames": 0,
            "final_status": None, "final_reason": None,
        },
    }


def test_settle_matches_gates_and_reports_family_denominators():
    document = {
        "split": "dev",
        "rows": [
            _row("SC-17/0", "SC-17", survived=True, baseline_survived=True, reference="feasible"),
            _row("SC-17/1", "SC-17", survived=True, baseline_survived=False, reference="feasible"),
            _row("SC-15/0", "SC-15", survived=True, baseline_survived=True),
        ],
    }
    settled = evaluate.settle(document)
    direct = evaluate.gates(document)
    assert settled["scenarios"] == 3
    assert settled["split"] == "dev"
    for gate in ("A-01", "A-02", "A-04", "A-06", "A-07"):
        assert settled["gates"][gate]["status"] == direct[gate]["status"]
    assert settled["family_breakdown"]["SC-17"]["rows"] == 2
    assert settled["family_breakdown"]["SC-17"]["subject_rate"] == 1.0
    assert settled["family_breakdown"]["SC-17"]["baseline_rate"] == 0.5
    # A-08 differs by design: settle derives sparse_success from the SC-17 rows.
    assert settled["gates"]["A-08"]["status"] == "pass"
    assert settled["gates"]["A-08"]["rate"] == 1.0


def test_render_markdown_contains_the_gate_and_family_tables():
    document = {
        "split": "dev",
        "rows": [
            _row("SC-17/0", "SC-17", survived=True, baseline_survived=True, reference="feasible"),
            _row("SC-16/0", "SC-16", survived=True, baseline_survived=False),
        ],
    }
    settlement = evaluate.settle(document)
    markdown = evaluate.render_markdown(settlement)
    assert "| A-01 |" in markdown
    assert "| A-02 | pass |" in markdown
    assert "| SC-17 | 1 |" in markdown
    assert "| SC-16 | 1 |" in markdown
    assert "A-02" in markdown and "存活/可行" in markdown


def test_settle_keeps_the_reference_labels_it_used():
    document = {
        "split": "dev",
        "rows": [_row("E-05", "E-05", survived=True, baseline_survived=False, reference="feasible")],
    }
    settled = evaluate.settle(document, labels={"E-05": 25})
    assert settled["reference_labels"] == {"E-05": 25}
    assert settled["gates"]["A-03"]["denominator"] == 1
