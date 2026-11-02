"""CLI tests for ``python -m touhou_rl.escape`` (D0 commands only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from touhou_rl.escape import cli, contracts


def _observation(valid: bool = True) -> dict:
    document = contracts.to_json_dict(
        contracts.EscapeObservation(
            schema=contracts.OBSERVATION_SCHEMA,
            episode_id="cli-0",
            decision_id=0,
            frame_id=4,
            source=contracts.Source.PRIVILEGED,
            player=contracts.PlayerObs(x=192.0, y=384.0, focus=1, hurtbox_half=0.825),
            frameskip=2,
            observed_frame_age=0,
            action_latency_frames=1,
            completeness=contracts.Completeness.COMPLETE,
            coordinate_contract=contracts.COORDINATE_CONTRACT,
            hazards=(),
            bullet_pool_count=0,
        )
    )
    if not valid:
        document["coordinate_contract"] = dict(document["coordinate_contract"], length_unit="px_per_s")
    return document


def test_help_lists_the_implemented_and_stub_commands(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for command in ("schema", "validate", "baseline-fixture", "snapshot", "evidence"):
        assert command in out
    for command in ("synth", "evaluate", "render", "benchmark"):
        assert command in out


def test_schema_command_writes_generated_schema(tmp_path):
    out = tmp_path / "schema.json"
    assert cli.main(["schema", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == contracts.contract_schema()


def test_validate_command_reports_both_outcomes(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_observation()), encoding="utf-8")
    assert cli.main(["validate", str(good)]) == 0

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_observation(valid=False)), encoding="utf-8")
    assert cli.main(["validate", str(bad)]) == 1
    assert "issue(s)" in capsys.readouterr().out


def test_validate_accepts_a_list_of_documents(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps([_observation(), _observation()]), encoding="utf-8")
    assert cli.main(["validate", str(path)]) == 0


def test_baseline_fixture_write_then_check(tmp_path, capsys):
    out = tmp_path / "golden.json"
    assert cli.main(["baseline-fixture", "--out", str(out)]) == 0
    assert cli.main(["baseline-fixture", "--out", str(out), "--check"]) == 0
    # A missing fixture is a failure, not a silent pass.
    assert cli.main(["baseline-fixture", "--out", str(tmp_path / "nope.json"), "--check"]) == 1


def test_snapshot_records_repository_state(tmp_path):
    payload = tmp_path / "manifest.json"
    assert cli.main(["snapshot", "--out", str(payload), "--root", "."]) == 0
    document = json.loads(payload.read_text(encoding="utf-8"))
    assert document["git"]["available"] is True
    assert len(document["git"]["head"]) == 40
    assert document["baseline_surface_digest"]
    assert all(not record.get("missing") for record in document["contract_files"])


def test_snapshot_survives_missing_git(tmp_path, monkeypatch):
    from touhou_rl.escape import snapshot

    monkeypatch.setattr(snapshot, "_git", lambda *args, **kwargs: (127, "git unavailable: test"))
    state = snapshot.git_state(tmp_path)
    assert state["available"] is False
    assert "unavailable" in state["reason"]


def test_evidence_reports_missing_artifacts(tmp_path, capsys):
    base = tmp_path / "version"
    base.mkdir()
    (base / "present.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "evidence.json"
    code = cli.main(
        [
            "evidence",
            "--base",
            str(base),
            "--out",
            str(out),
            "--include",
            "present.json,absent.json",
        ]
    )
    assert code == 1
    assert "missing: absent.json" in capsys.readouterr().out
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["entries"][0]["bytes"] == 2
    assert manifest["entries"][1]["missing"] is True


@pytest.mark.parametrize("command", ["benchmark"])
def test_unimplemented_commands_exit_nonzero(command, capsys):
    assert cli.main([command]) == 3
    assert "not implemented" in capsys.readouterr().err


def test_synth_lists_families_and_edges(tmp_path, capsys):
    assert cli.main(["synth", "--list"]) == 0
    out = capsys.readouterr().out
    assert "SC-17" in out and "E-20" in out

    payload = tmp_path / "scenario.json"
    assert cli.main(["synth", "--edge", "E-05", "--out", str(payload)]) == 0
    document = json.loads(payload.read_text(encoding="utf-8"))
    assert document["scenario_id"] == "E-05"
    assert len(document["signature"]) == 16


def test_reference_reports_a_proof(tmp_path, capsys):
    payload = tmp_path / "reference.json"
    assert cli.main(["reference", "--edge", "E-11", "--out", str(payload)]) == 0
    document = json.loads(payload.read_text(encoding="utf-8"))
    assert document["result"]["status"] == "no_solution_certified"
    assert "area_covered" in document["result"]["reason"]


def test_reference_sweep_writes_status_counts(tmp_path):
    payload = tmp_path / "sweep.json"
    assert cli.main(["reference", "--sweep", "--budget", "50", "--out", str(payload)]) == 0
    document = json.loads(payload.read_text(encoding="utf-8"))
    from touhou_rl.escape import families

    expected = len(families.GENERATORS) + len(families.edge_ids())
    assert len(document["rows"]) == expected
    assert sum(document["status_counts"].values()) == expected


def test_forecast_writes_a_valid_record(tmp_path):
    payload = tmp_path / "forecast.json"
    assert cli.main(["forecast", "--edge", "E-01", "--out", str(payload)]) == 0
    document = json.loads(payload.read_text(encoding="utf-8"))
    assert contracts.validate_document(document) == []
    assert len(document["time_nodes_frames"]) == 31
    assert document["coverage_shape"] == [28, 24]


def test_forecast_benchmark_reports_timings(tmp_path):
    payload = tmp_path / "bench.json"
    assert cli.main(["forecast", "--family", "SC-01", "--benchmark", "2", "--out", str(payload)]) == 0
    document = json.loads(payload.read_text(encoding="utf-8"))
    assert document["build_ms"]["runs"] == 2
    assert document["build_ms"]["p95"] >= 0.0
    assert document["query_ms"]["runs"] == 1000


def test_record_and_render_roundtrip(tmp_path):
    out_dir = tmp_path / "trace"
    assert cli.main([
        "record", "--edge", "E-01", "--out", str(out_dir),
        "--plan-horizon", "4", "--plan-continuation", "1",
    ]) == 0
    trace_path = out_dir / "E-01.jsonl"
    assert trace_path.is_file()
    views = tmp_path / "views"
    assert cli.main(["render", "--trace", str(trace_path), "--out", str(views)]) == 0
    assert (views / "E-01.png").is_file()
    assert (views / "E-01.md").is_file()
