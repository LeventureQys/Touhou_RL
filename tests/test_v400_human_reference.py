"""Fail-closed human benchmark replay evidence checks, and the measured reference.

The measured-reference tests read EXP-P0-046's engine-measured evidence: the
``STATE`` counters the real engine produced while it played the two retail
replays.  They never invent a measurement; when the evidence is absent the test
is skipped rather than filled in.
"""
import copy
import json
from pathlib import Path

import pytest

from touhou_rl.v400.bench.human_reference import (
    CAP_MULTIPLIER, SCHEMA, HumanReferenceError, MissingHumanMeasurements,
    build_human_reference, build_reference_from_measurements, cap_for,
    cap_for_suite, cap_from_human_frames, inspect_human_replay, load_reference,
    validate_reference, write_reference,
)
from touhou_rl.v400.bench.suites import SUITES
from tools.rl.v400_human_reference import main

REPLAY = Path(__file__).parents[1] / "th08_replay" / "Th8_ud0059.rpy"
SHA = "62b02a172780d846f4109fd0dcc762692c5205692670c0f7400c2026c409a5e9"
MEASURED = Path(__file__).parents[1] / ".tmp_v400_human_autoplay" / "EXP-P0-046_measured_stages.json"
NORMAL_SHA = "e621086d814fba4838a2c84e4081162524c3c7e84d1bc4ca3f6621f9ca52d96f"
MEASURED_AT = "2026-11-08T23:43:36+08:00"
#: sha256 of the exact harness ELF that produced the measured numbers (EXP-P0-046).
ENGINE_SHA = "591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb"
needs_measured = pytest.mark.skipif(not MEASURED.is_file(),
                                    reason="EXP-P0-046 engine-measured evidence is absent")



def test_real_replay_stream_lengths_not_invented_gameplay_metrics():
    evidence = inspect_human_replay(REPLAY)
    assert evidence["source_sha256"] == SHA
    assert evidence["executable"] == {"size": 840704, "checksum": 2724749753, "version": "0100d"}
    assert [(s["stage_index"], s["input_records"]) for s in evidence["stages"]] == [
        (0, 16665), (1, 18942), (2, 22305), (4, 26820), (5, 30854), (7, 35524)]
    assert [s["retail_expected_trailing_input_records"] for s in evidence["stages"]] == [[2, 3]] * 5 + [[6, 7]]
    assert all(s["gameplay_frames"] is s["deaths"] is s["bombs_used"] is None for s in evidence["stages"])
    assert len(evidence["required_real_game_state"]) >= 4
    assert evidence["human_reference_created"] is False


def test_reference_builder_rejects_and_does_not_write(tmp_path):
    output = tmp_path / "human_reference.json"
    with pytest.raises(MissingHumanMeasurements, match="death and bomb-use events"):
        build_human_reference(REPLAY, output)
    assert not output.exists()


def test_cli_inspects_and_fails_closed(tmp_path, capsys):
    evidence_file = tmp_path / "replay_evidence.json"
    output = tmp_path / "human_reference.json"
    assert main([str(REPLAY), "--evidence-output", str(evidence_file), "--output", str(output)]) == 3
    report = json.loads(evidence_file.read_text(encoding="utf-8"))
    assert report["source_sha256"] == SHA
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_cli_rejects_evidence_named_as_reference(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main([str(REPLAY), "--evidence-output", str(tmp_path / "human_reference.json")])
    assert exc.value.code == 2
    assert not (tmp_path / "human_reference.json").exists()


def test_corrupt_replay_never_generates_reference(tmp_path):
    bad = tmp_path / "bad.rpy"
    bad.write_bytes(b"not a replay")
    assert main([str(bad), "--output", str(tmp_path / "human_reference.json")]) == 3
    assert not (tmp_path / "human_reference.json").exists()


# -- measured reference: built from engine STATE, never from input records --
def measured_evidence() -> dict:
    return json.loads(MEASURED.read_text(encoding="utf-8"))


def reference_from_measured() -> dict:
    return build_reference_from_measurements(
        measured_evidence(), measured_at=MEASURED_AT,
        measured_at_source="test", evidence_path=str(MEASURED),
        engine_sha256=ENGINE_SHA,
    )


def fixture_measured(frames: int = 100) -> dict:
    """Plumbing fixture only — not a human measurement, never written to bench/."""
    return {
        "kind": "real-engine-replay-playback",
        "human_reference_created": True,
        "runs": [{
            "difficulty": "normal",
            "replay": {"file": "fixture.rpy", "sha256": "a" * 64},
            "engine": {"binary": "/tmp/th08-modern", "binary_sha256": "c" * 64},
            "stages": [{"stage_index": 0, "difficulty": 1, "shot": 6,
                        "gameplay_frames": frames, "deaths": 0, "bombs_used": 0}],
        }],
    }


@needs_measured
def test_reference_carries_the_measured_engine_numbers():
    document = reference_from_measured()
    assert document["schema"] == SCHEMA
    assert document["collection_method"] == "engine STATE during real replay playback"
    assert document["not_from_input_records"] is True
    assert document["cap_multiplier"] == CAP_MULTIPLIER == 1.5
    assert document["cap_rule"] == "ceil(cap_multiplier * measured human gameplay_frames of that stage)"
    assert document["measured_at"] == MEASURED_AT
    assert len(document["caveats"]) >= 4

    normal = next(entry for entry in document["sets"] if entry["difficulty"] == "normal")
    assert normal["source_replay_sha256"] == NORMAL_SHA
    assert (normal["shot"], normal["shot_name"]) == (6, "Marisa")
    stage1 = next(stage for stage in normal["stages"] if stage["stage_index"] == 0)
    assert (stage1["stage_name"], stage1["gameplay_frames"], stage1["deaths"],
            stage1["bombs_used"]) == ("Stage1", 13258, 0, 0)
    assert stage1["replay_input_frames_cross_check"] == 13189

    lunatic = next(entry for entry in document["sets"] if entry["difficulty"] == "lunatic")
    assert lunatic["source_replay_sha256"] == SHA
    assert (lunatic["shot"], lunatic["shot_name"]) == (10, "Youmu")
    assert [(stage["stage_name"], stage["gameplay_frames"], stage["deaths"], stage["bombs_used"])
            for stage in lunatic["stages"]] == [
        ("Stage1", 16732, 0, 1), ("Stage2", 19009, 0, 0), ("Stage3", 22394, 0, 0),
        ("Stage4B", 26909, 0, 2), ("Stage5", 30943, 0, 1), ("Stage6B", 35959, 1, 2)]
    assert [(stage["stage_name"], stage["gameplay_frames"]) for stage in normal["stages"]] == [
        ("Stage1", 13258), ("Stage2", 18525), ("Stage3", 20009),
        ("Stage4A", 27722), ("Stage5", 25675), ("Stage6B", 36403)]


@needs_measured
def test_coverage_caveats_name_the_missing_stages():
    document = reference_from_measured()
    assert document["coverage"]["complete"] is False
    assert document["coverage"]["missing"]["normal"] == ["Stage4B", "Stage6A", "Extra"]
    assert document["coverage"]["missing"]["lunatic"] == ["Stage4A", "Stage6A", "Extra"]
    joined = " ".join(document["caveats"])
    for name in ("Stage6A", "Stage4A", "Stage4B", "Extra"):
        assert name in joined
    assert "single normal corpus" in joined
    assert "six stages" in joined


def test_cap_rule_is_ceil_of_one_and_a_half_human_frames():
    assert cap_from_human_frames(13258) == 19887
    assert cap_from_human_frames(13259) == 19889
    assert cap_from_human_frames(2) == 3
    for bad in (0, -1, True, 1.5, None):
        with pytest.raises(HumanReferenceError):
            cap_from_human_frames(bad)


@needs_measured
def test_cap_for_suite_uses_the_measured_stage_and_refuses_unmeasured_ones():
    document = reference_from_measured()
    sn = cap_for_suite(document, SUITES["S1N"])
    assert (sn.stage_name, sn.human_frames, sn.multiplier, sn.cap) == ("Stage1", 13258, 1.5, 19887)
    assert sn.source_replay_sha256 == NORMAL_SHA
    sl = cap_for_suite(document, SUITES["S1L"])
    assert (sl.human_frames, sl.cap) == (16732, 25098)
    with pytest.raises(HumanReferenceError, match="declares 8 stages"):
        cap_for_suite(document, SUITES["PSN"])
    with pytest.raises(HumanReferenceError, match="no measured play"):
        cap_for(document, difficulty="normal", stage_name_value="Stage6A")


def test_reference_round_trip_and_strict_reload(tmp_path):
    output = tmp_path / "human_reference.json"
    report = write_reference(
        build_reference_from_measurements(fixture_measured(), measured_at=MEASURED_AT,
                                          measured_at_source="test"),
        output,
    )
    assert report["sha256"] and report["sets"] == 1
    stored = load_reference(output)
    assert cap_for(stored, difficulty="normal", stage_name_value="Stage1").cap == 150


@pytest.mark.parametrize("mutation", [
    lambda doc: doc.update(schema="v400-wp7-human-reference/0"),
    lambda doc: doc.update(not_from_input_records=False),
    lambda doc: doc.update(collection_method="decoded from input records"),
    lambda doc: doc.update(cap_multiplier=2.0),
    lambda doc: doc.update(caveats=[]),
    lambda doc: doc.update(measured_at="2026-11-08T23:43:36"),  # no timezone
    lambda doc: doc.update(sets=[]),
    lambda doc: doc["sets"][0].update(source_replay_sha256="not-a-digest"),
    lambda doc: doc["sets"][0].update(difficulty="extra"),
    lambda doc: doc["sets"][0]["stages"][0].update(gameplay_frames=None),
    lambda doc: doc["sets"][0]["stages"][0].update(gameplay_frames=-3),
    lambda doc: doc["sets"][0]["stages"][0].update(deaths=None),
    lambda doc: doc["sets"][0]["stages"][0].update(bombs_used="2"),
    lambda doc: doc["sets"][0]["stages"][0].update(stage_name="Stage9"),
])
def test_loader_rejects_a_reference_that_cannot_support_a_cap(mutation):
    document = build_reference_from_measurements(fixture_measured(), measured_at=MEASURED_AT,
                                                 measured_at_source="test")
    tampered = copy.deepcopy(document)
    mutation(tampered)
    with pytest.raises(HumanReferenceError):
        validate_reference(tampered)


def test_builder_fails_closed_on_unmeasured_evidence(tmp_path):
    broken = fixture_measured()
    broken["runs"][0]["stages"][0]["gameplay_frames"] = None
    with pytest.raises(HumanReferenceError, match="gameplay_frames"):
        build_reference_from_measurements(broken, measured_at=MEASURED_AT, measured_at_source="test")
    wrong_kind = fixture_measured()
    wrong_kind["kind"] = "input-record-arithmetic"
    with pytest.raises(HumanReferenceError, match="kind"):
        build_reference_from_measurements(wrong_kind, measured_at=MEASURED_AT,
                                          measured_at_source="test")
    skipped_crosscheck = fixture_measured()
    skipped_crosscheck["runs"][0]["stages"][0]["replay_input_frames"] = 1_000
    with pytest.raises(HumanReferenceError, match="replay input frames"):
        build_reference_from_measurements(skipped_crosscheck, measured_at=MEASURED_AT,
                                          measured_at_source="test")
    no_engine_sha = fixture_measured()
    del no_engine_sha["runs"][0]["engine"]["binary_sha256"]
    with pytest.raises(HumanReferenceError, match="engine binary sha256"):
        build_reference_from_measurements(no_engine_sha, measured_at=MEASURED_AT,
                                          measured_at_source="test")


@needs_measured
def test_cli_requires_the_engine_binary_sha_for_real_evidence(tmp_path, capsys):
    """The measured JSON has no engine digest, so the build must be given one."""
    output = tmp_path / "human_reference.json"
    assert main(["--measured", str(MEASURED), "--output", str(output)]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked" and "engine binary sha256" in payload["reason"]
    assert not output.exists()


@needs_measured
def test_cli_builds_the_reference_from_measured_evidence(tmp_path):
    output = tmp_path / "human_reference.json"
    assert main(["--measured", str(MEASURED), "--output", str(output),
                 "--engine-sha256", "d" * 64]) == 0
    document = load_reference(output)
    assert document["schema"] == SCHEMA
    assert document["measured_at_source"].startswith("file mtime")
    assert document["engine"]["binary_sha256"] == "d" * 64
    assert cap_for_suite(document, SUITES["S1N"]).cap == 19887


@needs_measured
def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    output = tmp_path / "human_reference.json"
    assert main(["--measured", str(MEASURED), "--dry-run", "--output", str(output),
                 "--engine-sha256", "d" * 64]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "dry_run"
    assert {cap["cap"] for cap in report["caps"]} >= {19887}
    assert not output.exists()


@needs_measured
def test_cli_refuses_tampered_measured_evidence(tmp_path, capsys):
    broken = measured_evidence()
    broken["runs"][1]["stages"][0]["gameplay_frames"] = None
    evidence = tmp_path / "measured.json"
    evidence.write_text(json.dumps(broken), encoding="utf-8")
    output = tmp_path / "human_reference.json"
    assert main(["--measured", str(evidence), "--output", str(output),
                 "--engine-sha256", "d" * 64]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked" and payload["human_reference_created"] is False
    assert not output.exists()

