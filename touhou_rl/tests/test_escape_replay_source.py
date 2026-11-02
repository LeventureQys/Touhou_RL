"""Recorded-trace replay tests (D8 readiness without the game)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from touhou_rl.escape import controller, families, planner, probe, replay_source, trace

SHORT = controller.ControllerConfig(
    planner=planner.PlannerConfig(horizon_decisions=4, continuation_decisions=1, beam_width=1)
)


def _record(tmp_path: Path) -> Path:
    scenario = families.edge_instances()["E-01"]
    summary = trace.record_episode(scenario, out_dir=tmp_path, config=SHORT)
    return tmp_path / summary["trace"]


def test_recorded_source_reads_a_v370_trace(tmp_path):
    path = _record(tmp_path)
    source = replay_source.RecordedSource(path)
    observation = source.read()
    assert observation is not None
    assert observation.player is not None
    assert observation.frame_id >= 0
    field = source.field()
    assert field.positions.ndim == 2
    source.advance()
    assert not source.done()


def test_recorded_source_skips_the_warmup_start(tmp_path):
    path = _record(tmp_path)
    first = replay_source.RecordedSource(path, start=0, limit=1)
    later = replay_source.RecordedSource(path, start=3, limit=1)
    assert first.read().frame_id != later.read().frame_id


def test_recorded_probe_marks_itself_offline(tmp_path):
    path = _record(tmp_path)
    document = probe.run_recorded_probe(
        replay_source.RecordedSource(path), decisions=4, config=probe.ProbeConfig()
    )
    assert document["source"] == "recorded-game-trace"
    assert document["live"] is False
    assert "same-frame sync" in document["note"]
    assert document["decisions"] == 4
    assert document["readable"] > 0
    for row in document["rows"]:
        if row["status"] == "ok":
            assert row["plan_status"] in {
                "feasible", "budget_exhausted", "no_solution_certified", "model_unknown"
            }


def test_recorded_probe_reports_plan_latency(tmp_path):
    path = _record(tmp_path)
    document = probe.run_recorded_probe(
        replay_source.RecordedSource(path), decisions=3, config=probe.ProbeConfig()
    )
    assert document["plan_ms_p50"] is not None
    assert document["max_hazards"] >= 0
