"""Trace sidecar, attribution and replay tests (D6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from touhou_rl.escape import contracts, controller, families, planner, render, trace

SHORT = controller.ControllerConfig(
    planner=planner.PlannerConfig(horizon_decisions=4, continuation_decisions=1, beam_width=1)
)


def _record(name: str, scenario, tmp_path: Path):
    summary = trace.record_episode(scenario, out_dir=tmp_path, config=SHORT)
    return summary, tmp_path / summary["trace"], tmp_path / summary["sidecar"]


def test_recorded_trace_has_contract_records(tmp_path):
    summary, trace_path, sidecar_path = _record("E-01", families.edge_instances()["E-01"], tmp_path)
    assert trace_path.is_file() and sidecar_path.is_file()
    records = trace.load_trace(trace_path)
    assert len(records) == summary["decisions"]
    frames = [record["observation"]["frame_id"] for record in records]
    assert frames == sorted(frames)
    sidecar = trace.load_sidecar(sidecar_path)
    for index, record in enumerate(records):
        assert record["schema"] == trace.TRACE_SCHEMA
        assert record["observation"]["hazards_ref"]["count"] == int(
            sidecar["offsets"][index + 1] - sidecar["offsets"][index]
        )
        assert contracts.validate_document(record["plan"]) == []
        assert contracts.validate_document(record["decision_record"]) == []
        assert record["forecast"]["plan_status"] in {
            "feasible", "budget_exhausted", "no_solution_certified", "model_unknown"
        }


def test_attribution_finds_the_hit_and_classifies(tmp_path):
    summary, trace_path, _sidecar = _record("E-11", families.edge_instances()["E-11"], tmp_path)
    assert summary["hit"] is True
    records = trace.load_trace(trace_path)
    report = trace.attribute(records)
    assert report["hit_frame"] == summary["frames"]
    assert report["classification"] in {"no_route_at_decision", "hit_while_route_feasible"}
    assert report["window"]


def test_clean_episode_is_classified_clean(tmp_path):
    _summary, trace_path, _sidecar = _record("E-01", families.edge_instances()["E-01"], tmp_path)
    report = trace.attribute(trace.load_trace(trace_path))
    assert report["classification"] == "clean"
    assert report["hit_frame"] is None


def test_text_render_contains_the_decision_table(tmp_path):
    _summary, trace_path, _sidecar = _record("E-01", families.edge_instances()["E-01"], tmp_path)
    text = render.render_text(trace.load_trace(trace_path))
    assert "| # | frame | intent |" in text
    assert "classification:" in text
    assert "feasible" in text


def test_png_render_writes_a_layered_image(tmp_path):
    _summary, trace_path, sidecar_path = _record("E-01", families.edge_instances()["E-01"], tmp_path)
    out = tmp_path / "replay.png"
    path = render.render_png(
        trace.load_trace(trace_path), trace.load_sidecar(sidecar_path), out, scale=1
    )
    assert path.is_file() and path.stat().st_size > 1000
    from PIL import Image

    with Image.open(path) as image:
        assert image.size == (384, 448)


def test_render_episode_writes_both_views(tmp_path):
    _summary, trace_path, sidecar_path = _record("E-01", families.edge_instances()["E-01"], tmp_path)
    result = render.render_episode(trace_path, sidecar_path, tmp_path / "views")
    assert Path(result["text"]).is_file()
    assert Path(result["png"]).is_file()
    assert "classification" in result["attribution"]


def test_recording_is_deterministic_in_decisions(tmp_path):
    scenario = families.edge_instances()["E-01"]
    first = trace.record_episode(scenario, out_dir=tmp_path / "a", config=SHORT)
    second = trace.record_episode(scenario, out_dir=tmp_path / "b", config=SHORT)
    rows_a = [
        (r["decision_record"]["intent"], r["decision_record"]["accepted_action"], r["observation"]["frame_id"])
        for r in trace.load_trace(tmp_path / "a" / first["trace"])
    ]
    rows_b = [
        (r["decision_record"]["intent"], r["decision_record"]["accepted_action"], r["observation"]["frame_id"])
        for r in trace.load_trace(tmp_path / "b" / second["trace"])
    ]
    assert rows_a == rows_b
