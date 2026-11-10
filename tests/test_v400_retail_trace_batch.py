"""Unit tests for the serial retail trace batch driver (EXP-P0-072).

No game is launched and no retail instance is touched: the observer is replaced
by a stub script that speaks the same CLI/report contract, and the ``th08.exe``
guard probe is injected.  The canonical-hash expectations come from the
official ``compare-replay-traces.trace_sha256``, never from a file hash.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/rl/v400_retail_trace_batch.py"
OFFICIAL_COMPARISON = ROOT / "third_party/th08/scripts/compare-replay-traces.py"
OFFICIAL_SCHEMA = ROOT / "third_party/th08/scripts/replay-external-schema.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


batch = _load("v400_retail_trace_batch", MODULE_PATH)
sys.path.insert(0, str(OFFICIAL_COMPARISON.parent))
comparison = _load("th08_compare_replay_traces", OFFICIAL_COMPARISON)
SCHEMA = json.loads(OFFICIAL_SCHEMA.read_text(encoding="utf-8"))
FIELDS = SCHEMA["fields"]


STUB_OBSERVER = r'''
"""Stub observer for the batch-driver tests; it never launches a game."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FIELDS = ["calc", "stage", "frame", "gameFrame", "score", "graze", "deaths", "lives",
          "bombs", "power", "x", "y", "z", "playerState", "rngSeed", "rngGeneration",
          "input", "pointItems", "timeOrbs", "gauge", "clock", "pointValue",
          "character", "shot", "difficulty"]


def main() -> int:
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1])
    replay = Path(argv[argv.index("--replay") + 1])
    if out.exists() and any(out.iterdir()):
        print(f"refusing to reuse non-empty {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(os.environ["V400_BATCH_STUB_MANIFEST"]).read_text(encoding="utf-8"))
    mode = os.environ.get("V400_BATCH_STUB_MODE", "ok")
    record = next(item for item in manifest["fixtures"] if item["file"] == replay.name)
    trace = record["retailTrace"]
    expected = int(trace["calculationFrames"])
    if mode == "crash":
        print("stub crash", file=sys.stderr)
        return 9
    rows = expected if mode != "partial" else expected - 3
    stage_indexes = [int(stage["index"]) for stage in record["stages"]]
    final_index = stage_indexes[-1]
    stage_frames = {}
    for stage in record["stages"]:
        # the official tail invariant: the final stage keeps 6/7 input records,
        # a cleared stage 2/3 -- so sum(stageFrames) is smaller than rows
        tail = 6 if int(stage["index"]) == final_index else 2
        stage_frames[str(stage["index"])] = int(stage["inputRecords"]) - tail
    complete = {
        "rows": rows,
        "ended": rows == expected,
        "errors": [],
        "stages": stage_indexes,
        "stageFrames": stage_frames,
        "endScores": {str(stage["index"]): int(stage["endScore"]) for stage in record["stages"]},
        "rngOrigins": {str(stage["index"]): position
                       for position, stage in enumerate(record["stages"])},
        "naturalReplayExit": rows == expected,
    }
    (out / "complete.json").write_text(json.dumps(complete) + "\n", encoding="utf-8")
    if mode == "noreport":
        with (out / "rows.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for number in range(rows):
                stream.write(json.dumps([number % 1000] * len(FIELDS)) + "\n")
        print("stub finished rows without a report", file=sys.stderr)
        return 7
    sha = trace["sha256"] if mode == "ok" else "0" * 64
    stage_table = []
    for stage in record["stages"]:
        tail = 6 if int(stage["index"]) == final_index else 2
        last_frame = stage_frames[str(stage["index"])]
        remaining = int(stage["inputRecords"]) - last_frame
        stage_table.append({
            "stage": int(stage["index"]),
            "name": stage.get("name"),
            "inputRecords": int(stage["inputRecords"]),
            "rows": rows,
            "lastFrame": last_frame,
            "unconsumedInputRecords": remaining,
            "expectedTail": [tail, tail + 1],
            "tailOk": remaining in (tail, tail + 1),
            "manifestEndScore": int(stage["endScore"]),
            "observedEndScore": int(stage["endScore"]),
            "endScoreMatch": True,
        })
    verdict = {
        "rows": rows,
        "calculationFrames": expected,
        "rowsMatch": rows == expected,
        "canonicalTraceSha256": sha,
        "retailTraceSha256": trace["sha256"],
        "shaMatch": sha == trace["sha256"],
        "stageTable": stage_table,
    }
    status = "ok" if (verdict["rowsMatch"] and verdict["shaMatch"]) else "mismatch"
    report = {
        "schema": "stub-observer/1",
        "phase": "rows",
        "status": status,
        "result": {"reason": "stub " + mode, "exitCode": 0 if status == "ok" else 6},
        "verdict": verdict,
        "elapsedSeconds": 0.01,
        "manifest": {"fixtureId": record["id"], "file": record["file"]},
    }
    (out / "report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
    with (out / "rows.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for number in range(rows):
            stream.write(json.dumps([number] * len(FIELDS)) + "\n")
    print(json.dumps({"phase": "rows", "status": status}))
    return 0 if status == "ok" else 6


if __name__ == "__main__":
    raise SystemExit(main())
'''


# ---------------------------------------------------------------------------
# fixtures for the tests themselves
# ---------------------------------------------------------------------------
def _record(index: int, frames: int = 10) -> dict:
    return {
        "id": f"fx-{index:02d}",
        "file": f"th8_fx{index:02d}.rpy",
        "sha256": hashlib.sha256(f"replay-{index}".encode()).hexdigest(),
        "retailTrace": {
            "schemaVersion": 4,
            "calculationFrames": frames,
            "sha256": hashlib.sha256(f"trace-{index}".encode()).hexdigest(),
        },
        "stages": [{"index": 0, "name": "Stage 1", "inputRecords": frames,
                    "endScore": 1000 + index}],
    }


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Temp manifest + replay dir + stub observer, wired through the env."""
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    records = [_record(0), _record(1), _record(2)]
    for record in records:
        (replay_dir / record["file"]).write_bytes(b"stub replay " + record["id"].encode())
    manifest_path = tmp_path / "fixtures.json"
    manifest_path.write_text(
        json.dumps({"version": 1, "targetSha256": "0" * 64, "fixtures": records}),
        encoding="utf-8")
    stub = tmp_path / "stub_observer.py"
    stub.write_text(STUB_OBSERVER, encoding="utf-8")
    monkeypatch.setenv("V400_BATCH_STUB_MANIFEST", str(manifest_path))
    monkeypatch.setenv("V400_BATCH_STUB_MODE", "ok")
    monkeypatch.setattr(batch, "INSTANCE_PROBE", lambda: [])
    return {
        "root": tmp_path / "batch",
        "manifest": manifest_path,
        "replay_dir": replay_dir,
        "stub": stub,
        "records": records,
        "tmp": tmp_path,
    }


def _run_argv(sandbox, *extra: str) -> list[str]:
    return [
        "--root", str(sandbox["root"]),
        "--fixtures", str(sandbox["manifest"]),
        "--replay-dir", str(sandbox["replay_dir"]),
        "--observer", str(sandbox["stub"]),
        "--python", sys.executable,
        "--budget-seconds", "20",
        "--guard-timeout", "0.05",
        "--guard-poll", "0.01",
        "--wallclock-slack", "60",
        "--heartbeat-seconds", "3600",
        *extra,
    ]


def _journal(sandbox) -> list[dict]:
    return batch.read_journal(sandbox["root"] / "journal.jsonl")


def _report(sandbox) -> dict:
    return json.loads((sandbox["root"] / "report.json").read_text(encoding="utf-8"))


def _ok_entry(fixture_id: str, rows: int, manifest_sha: str, actual: str) -> dict:
    return {
        "fixture_id": fixture_id, "file": f"{fixture_id}.rpy",
        "manifest_sha256": manifest_sha, "actual_sha256": actual,
        "rows": rows, "calculationFrames": rows,
        "rows_equal": True, "sha_equal": actual == manifest_sha,
        "elapsed_s": 12.5, "exit_code": 0 if actual == manifest_sha else 6,
        "status": "ok" if actual == manifest_sha else "mismatch", "error": None,
        "out": f"/tmp/{fixture_id}", "endedAt": "2026-11-09T00:00:00+08:00",
        "attempt": 1,
    }


# ---------------------------------------------------------------------------
# manifest / journal / resume semantics
# ---------------------------------------------------------------------------
def test_load_fixtures_reads_the_real_manifest():
    fixtures = batch.load_fixtures()
    assert len(fixtures) == 108
    assert len({item["id"] for item in fixtures}) == 108
    assert sum(item["calculationFrames"] for item in fixtures) == 12311460
    assert all(item["present"] for item in fixtures)
    assert all(len(item["retailTraceSha256"]) == 64 for item in fixtures)
    assert {item["file"] for item in fixtures} >= {"th8_ud2cd2.rpy", "th8_ud2cdc.rpy"}


def test_load_fixtures_rejects_missing_trace_and_duplicates(tmp_path):
    good = _record(0)
    broken = json.loads(json.dumps(good))
    broken["id"] = "fx-broken"
    del broken["retailTrace"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"fixtures": [good, broken]}), encoding="utf-8")
    with pytest.raises(ValueError, match="calculationFrames"):
        batch.load_fixtures(path, tmp_path)

    broken["retailTrace"] = {"schemaVersion": 4, "calculationFrames": 10}
    path.write_text(json.dumps({"fixtures": [good, broken]}), encoding="utf-8")
    with pytest.raises(ValueError, match="retailTrace.sha256"):
        batch.load_fixtures(path, tmp_path)

    duplicated = json.loads(json.dumps(good))
    path.write_text(json.dumps({"fixtures": [good, duplicated]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate fixture id"):
        batch.load_fixtures(path, tmp_path)


def test_append_journal_fsyncs_each_entry(tmp_path, monkeypatch):
    calls = []
    real_fsync = batch.os.fsync

    def counting_fsync(fd):
        calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(batch.os, "fsync", counting_fsync)
    journal = tmp_path / "journal.jsonl"
    batch.append_journal(journal, {"fixture_id": "a", "status": "ok", "rows": 1})
    batch.append_journal(journal, {"fixture_id": "b", "status": "failed", "rows": None})
    assert len(calls) == 2
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["fixture_id"] == "a"
    assert json.loads(lines[1])["status"] == "failed"
    assert len(batch.read_journal(journal)) == 2


def test_terminal_entries_only_skip_ok_and_mismatch():
    entries = [
        _ok_entry("a", 10, "m", "m"),
        _ok_entry("b", 10, "m", "0"),
        {"fixture_id": "c", "status": "failed", "error": "boom", "rows": None,
         "rows_equal": None, "sha_equal": None, "elapsed_s": 1.0, "exit_code": 7},
        _ok_entry("a", 10, "m", "m"),
    ]
    done = batch.terminal_entries(entries)
    assert set(done) == {"a", "b"}
    assert done["b"]["status"] == "mismatch"
    assert batch.latest_entries(entries)["c"]["status"] == "failed"


def _failed_entry(fixture_id: str, attempt: int = 1,
                  error: str = "navigation hang; observer budget not enforced") -> dict:
    return {
        "fixture_id": fixture_id, "file": f"{fixture_id}.rpy", "status": "failed",
        "error": error, "rows": 0, "rows_equal": None, "sha_equal": None,
        "elapsed_s": 2441.0, "exit_code": None, "attempt": attempt,
        "out": f"/tmp/{fixture_id}", "endedAt": "2026-11-09T13:01:51+08:00",
    }


def test_failed_entry_is_retried_once_then_left_exhausted():
    """EXP-P0-075: failed is not final, but it is not retried forever either."""
    entries = [_failed_entry("a")]
    assert batch.terminal_entries(entries) == {}          # gets a retry
    assert batch.exhausted_entries(entries, 2) == {}      # ... this one
    entries.append(_failed_entry("a", attempt=2))
    assert set(batch.exhausted_entries(entries, 2)) == {"a"}
    assert batch.exhausted_entries(entries, 3) == {}


def test_retry_policy_is_recorded_for_audit():
    policy = batch.retry_policy(2)
    assert policy["retryFailed"] is True
    assert policy["maxAttemptsPerFixture"] == 2
    assert policy["terminalStatuses"] == ["ok", "mismatch"]
    assert batch.retry_policy(2, skip_failed=True)["retryFailed"] is False
    assert batch.DEFAULT_MAX_ATTEMPTS == 2


def test_main_retries_a_failed_fixture_once_and_keeps_the_old_evidence(sandbox):
    fixtures = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])
    first = fixtures[0]["id"]
    sandbox["root"].mkdir(parents=True, exist_ok=True)
    # a failed attempt that left its (non-empty) evidence directory behind
    (sandbox["root"] / first).mkdir()
    (sandbox["root"] / first / "debugger.log").write_text("[2.5] armed 15 threads\n",
                                                          encoding="utf-8")
    batch.append_journal(sandbox["root"] / "journal.jsonl", _failed_entry(first))

    code = batch.main(_run_argv(sandbox, "--only", first))

    assert code == batch.EXIT_PARTIAL                       # fx-01/fx-02 still pending
    journal = _journal(sandbox)
    assert [entry["status"] for entry in journal] == ["failed", "ok"]
    assert journal[1]["attempt"] == 2
    assert (sandbox["root"] / first / "attempt-2").is_dir()  # retry in a new dir
    assert (sandbox["root"] / first / "debugger.log").is_file()  # nothing deleted
    assert _report(sandbox)["retryPolicy"]["maxAttemptsPerFixture"] == 2


def test_main_leaves_a_twice_failed_fixture_exhausted(sandbox):
    fixtures = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])
    first = fixtures[0]["id"]
    sandbox["root"].mkdir(parents=True, exist_ok=True)
    journal_path = sandbox["root"] / "journal.jsonl"
    batch.append_journal(journal_path, _failed_entry(first, attempt=1))
    batch.append_journal(journal_path, _failed_entry(first, attempt=2))

    code = batch.main(_run_argv(sandbox, "--only", first))

    assert code == batch.EXIT_FAILURES
    assert len(_journal(sandbox)) == 2                      # exhausted: nothing ran
    assert not (sandbox["root"] / first).exists()
    report = _report(sandbox)
    assert report["failed"] == 1
    assert report["retryPolicy"]["maxAttemptsPerFixture"] == 2


def test_next_attempt_never_deletes(tmp_path):
    assert batch.next_attempt(tmp_path, "fx-00") == 1
    base = tmp_path / "fx-00"
    base.mkdir()
    assert batch.next_attempt(tmp_path, "fx-00") == 1  # empty directory is reusable
    (base / "rows.jsonl").write_text("[]\n", encoding="utf-8")
    assert batch.next_attempt(tmp_path, "fx-00") == 2
    (base / "attempt-2").mkdir()
    (base / "attempt-2" / "rows.jsonl").write_text("[]\n", encoding="utf-8")
    assert batch.next_attempt(tmp_path, "fx-00") == 3
    assert batch.attempt_dir(tmp_path, "fx-00", 1) == base
    assert batch.attempt_dir(tmp_path, "fx-00", 2) == base / "attempt-2"


# ---------------------------------------------------------------------------
# single-instance guard
# ---------------------------------------------------------------------------
def test_wait_for_no_instances_polls_until_clear():
    sequence = iter([[7], [7], []])
    sleeps = []
    outcome = batch.wait_for_no_instances(
        5.0, probe=lambda: next(sequence, []), poll_s=0.01, sleep=sleeps.append)
    assert outcome["ok"] is True
    assert outcome["pids"] == []
    assert len(sleeps) == 2


def test_wait_for_no_instances_gives_up_after_the_timeout():
    outcome = batch.wait_for_no_instances(
        0.0, probe=lambda: [4242], poll_s=0.01, sleep=lambda _s: None)
    assert outcome["ok"] is False
    assert outcome["pids"] == [4242]
    assert outcome["timeoutSeconds"] == 0.0


# ---------------------------------------------------------------------------
# report aggregation and exit codes
# ---------------------------------------------------------------------------
def test_summarize_and_exit_codes():
    fixtures = batch.load_fixtures()
    subset = fixtures[:2]
    done = [_ok_entry(subset[0]["id"], subset[0]["calculationFrames"],
                      subset[0]["retailTraceSha256"], subset[0]["retailTraceSha256"])]
    report = batch.build_report(subset, done, root=Path("/tmp"), state="running")
    assert (report["total"], report["compared"], report["identical"],
            report["mismatch"], report["failed"], report["remaining"]) == (2, 1, 1, 0, 0, 1)
    assert report["totalRowsPerSide"]["captured"] == subset[0]["calculationFrames"]
    assert report["totalRowsPerSide"]["allManifest"] == (
        subset[0]["calculationFrames"] + subset[1]["calculationFrames"])
    assert batch.exit_code_for(report) == batch.EXIT_PARTIAL

    mismatched = done + [_ok_entry(subset[1]["id"], subset[1]["calculationFrames"],
                                   subset[1]["retailTraceSha256"], "0" * 64)]
    report = batch.build_report(subset, mismatched, root=Path("/tmp"))
    assert report["mismatch"] == 1
    assert report["mismatches"][0]["actualSha256"] == "0" * 64
    assert report["mismatches"][0]["manifestSha256"] == subset[1]["retailTraceSha256"]
    assert batch.exit_code_for(report) == batch.EXIT_FAILURES

    identical = [
        _ok_entry(fixture["id"], fixture["calculationFrames"],
                  fixture["retailTraceSha256"], fixture["retailTraceSha256"])
        for fixture in subset
    ]
    report = batch.build_report(subset, identical, root=Path("/tmp"))
    assert batch.exit_code_for(report) == batch.EXIT_ALL_IDENTICAL


def test_summarize_timing_and_projection():
    fixtures = batch.load_fixtures()[:2]
    entries = [_ok_entry(fixtures[0]["id"], fixtures[0]["calculationFrames"],
                         fixtures[0]["retailTraceSha256"], fixtures[0]["retailTraceSha256"])]
    entries[0]["elapsed_s"] = 100.0
    summary = batch.summarize(fixtures, entries)
    timing = summary["timing"]
    assert timing["totalElapsedSeconds"] == 100.0
    assert timing["averageSecondsPerFixture"] == 100.0
    assert timing["averageRowsPerSecond"] == round(
        fixtures[0]["calculationFrames"] / 100.0, 2)
    assert timing["projectedRemainingSecondsByFixture"] == 100.0
    assert timing["projectedRemainingSecondsByRows"] == round(
        fixtures[1]["calculationFrames"] / (fixtures[0]["calculationFrames"] / 100.0), 1)


def test_count_rows_handles_a_growing_file(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b"[1,2]\n[3,4]\n[5,6]")
    assert batch.count_rows(rows) == (2, len(b"[1,2]\n[3,4]\n[5,6]"))
    rows.write_bytes(b"[1,2]\n[3,4]\n[5,6]\n")
    assert batch.count_rows(rows)[0] == 3
    assert batch.count_rows(tmp_path / "missing.jsonl") == (0, 0)


# ---------------------------------------------------------------------------
# one fixture through the stub observer
# ---------------------------------------------------------------------------
def test_run_one_ok_records_both_hash_kinds(sandbox):
    fixture = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])[0]
    entry = batch.run_one(fixture, root=sandbox["root"], budget_seconds=20.0,
                          python=sys.executable, observer=sandbox["stub"],
                          guard_timeout=0.05, log=lambda _m: None,
                          probe=lambda: [], guard_poll=0.01)
    assert entry["status"] == "ok"
    assert entry["exit_code"] == 0
    assert entry["rows"] == fixture["calculationFrames"]
    assert entry["rows_equal"] is True
    assert entry["sha_equal"] is True
    assert entry["actual_sha256"] == fixture["retailTraceSha256"]
    assert entry["manifest_sha256"] == fixture["retailTraceSha256"]
    assert entry["replay_sha256"] == fixture["replay_sha256"]
    assert entry["elapsed_s"] > 0
    assert entry["guard_before"]["ok"] is True and entry["guard_after"]["ok"] is True
    assert entry["out"] == str(sandbox["root"] / fixture["id"])
    assert entry["attempt"] == 1
    # the file hash is recorded but is *not* the canonical hash
    rows_file = Path(entry["out"]) / "rows.jsonl"
    assert entry["rows_file_sha256"] == batch.sha256_file(rows_file)
    assert entry["rows_file_sha256"] != entry["actual_sha256"]
    assert entry["stage_frames_equal"] is True
    assert entry["end_scores_equal"] is True
    assert entry["ended"] is True


def test_run_one_mismatch_keeps_both_shas(sandbox, monkeypatch):
    monkeypatch.setenv("V400_BATCH_STUB_MODE", "mismatch")
    fixture = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])[1]
    entry = batch.run_one(fixture, root=sandbox["root"], budget_seconds=20.0,
                          python=sys.executable, observer=sandbox["stub"],
                          guard_timeout=0.05, log=lambda _m: None,
                          probe=lambda: [], guard_poll=0.01)
    assert entry["status"] == "mismatch"
    assert entry["exit_code"] == 6
    assert entry["rows_equal"] is True
    assert entry["sha_equal"] is False
    assert entry["actual_sha256"] == "0" * 64
    assert entry["manifest_sha256"] == fixture["retailTraceSha256"]


def test_run_one_partial_rows_is_failed(sandbox, monkeypatch):
    monkeypatch.setenv("V400_BATCH_STUB_MODE", "partial")
    fixture = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])[0]
    entry = batch.run_one(fixture, root=sandbox["root"], budget_seconds=20.0,
                          python=sys.executable, observer=sandbox["stub"],
                          guard_timeout=0.05, log=lambda _m: None,
                          probe=lambda: [], guard_poll=0.01)
    assert entry["status"] == "failed"
    assert entry["rows_equal"] is False
    assert entry["rows"] == fixture["calculationFrames"] - 3
    assert "incomplete" in entry["error"] or "stub partial" in entry["error"]


def test_run_one_crash_is_recorded_not_raised(sandbox, monkeypatch):
    monkeypatch.setenv("V400_BATCH_STUB_MODE", "crash")
    fixture = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])[0]
    entry = batch.run_one(fixture, root=sandbox["root"], budget_seconds=20.0,
                          python=sys.executable, observer=sandbox["stub"],
                          guard_timeout=0.05, log=lambda _m: None,
                          probe=lambda: [], guard_poll=0.01)
    assert entry["status"] == "failed"
    assert entry["exit_code"] == 9
    assert entry["error"]
    assert "stub crash" in (entry.get("stderrTail") or "")


def test_run_one_without_report_uses_the_official_canonical_hash(sandbox, monkeypatch):
    monkeypatch.setenv("V400_BATCH_STUB_MODE", "noreport")
    fixture = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])[0]
    entry = batch.run_one(fixture, root=sandbox["root"], budget_seconds=20.0,
                          python=sys.executable, observer=sandbox["stub"],
                          guard_timeout=0.05, log=lambda _m: None,
                          probe=lambda: [], guard_poll=0.01)
    rows_file = Path(entry["out"]) / "rows.jsonl"
    rows = [json.loads(line) for line in rows_file.read_text(encoding="utf-8").splitlines()]
    expected = comparison.trace_sha256({"schema": SCHEMA}, rows)
    assert entry["actual_sha256"] == expected
    assert entry["rows_file_sha256"] == batch.sha256_file(rows_file)
    assert entry["actual_sha256"] != entry["rows_file_sha256"]
    assert entry.get("canonicalComputedByDriver") is True
    assert entry["rows"] == fixture["calculationFrames"]
    # rows are complete but the canonical hash is not the published one
    assert entry["status"] == "mismatch"
    assert entry["exit_code"] == 7
    assert entry["ended"] is True
    assert entry["stage_frames_equal"] is True


def test_stage_tails_follow_the_official_invariant():
    stages = [
        {"index": 0, "inputRecords": 100, "endScore": 1},
        {"index": 6, "inputRecords": 200, "endScore": 2},
    ]
    assert batch._stage_tails_ok(stages, {"0": 98, "6": 194}) is True   # 2 / 6 remain
    assert batch._stage_tails_ok(stages, {"0": 97, "6": 193}) is True   # 3 / 7 remain
    assert batch._stage_tails_ok(stages, {"0": 96, "6": 194}) is False  # 4 remain
    assert batch._stage_tails_ok(stages, {"0": 98, "6": 192}) is False  # 8 remain
    assert batch._stage_tails_ok(stages, {"0": 98}) is None
    assert batch._stage_tails_ok([], {}) is None


def test_run_one_guard_blocks_before_creating_a_directory(sandbox):
    fixture = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])[0]
    entry = batch.run_one(fixture, root=sandbox["root"], budget_seconds=20.0,
                          python=sys.executable, observer=sandbox["stub"],
                          guard_timeout=0.0, log=lambda _m: None,
                          probe=lambda: [4242], guard_poll=0.01)
    assert entry["status"] == "failed"
    assert "residual th08.exe" in entry["error"]
    assert entry["rows"] is None
    assert not (sandbox["root"] / fixture["id"]).exists()


def test_run_one_retry_uses_a_new_attempt_directory(sandbox):
    fixture = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])[0]
    base = sandbox["root"] / fixture["id"]
    base.mkdir(parents=True)
    (base / "rows.jsonl").write_text("[]\n", encoding="utf-8")
    entry = batch.run_one(fixture, root=sandbox["root"], budget_seconds=20.0,
                          python=sys.executable, observer=sandbox["stub"],
                          guard_timeout=0.05, log=lambda _m: None,
                          probe=lambda: [], guard_poll=0.01)
    assert entry["attempt"] == 2
    assert entry["out"] == str(base / "attempt-2")
    assert entry["retryOf"] == str(base)
    assert (base / "rows.jsonl").is_file()  # nothing was deleted


def test_observer_command_contract(sandbox, monkeypatch):
    recorded = []
    real_run = subprocess.run

    def recording_run(command, **kwargs):
        recorded.append(list(command))
        return real_run(command, **kwargs)

    monkeypatch.setattr(batch.subprocess, "run", recording_run)
    fixture = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])[0]
    entry = batch.run_one(fixture, root=sandbox["root"], budget_seconds=1800.0,
                          python=sys.executable, observer=sandbox["stub"],
                          guard_timeout=0.05, log=lambda _m: None,
                          probe=lambda: [], guard_poll=0.01)
    observer_command = [command for command in recorded
                        if any(str(part).endswith("stub_observer.py") for part in command)]
    assert len(observer_command) == 1
    command = observer_command[0]
    assert command[1:3] == ["-u", str(sandbox["stub"])]
    assert command[command.index("--phase") + 1] == "rows"
    assert command[command.index("--budget-seconds") + 1] == "1800"
    assert command[command.index("--replay") + 1] == str(fixture["replay"])
    assert "--with-score-dat" not in command  # the default skip is the required one
    assert entry["status"] == "ok"


# ---------------------------------------------------------------------------
# the whole driver
# ---------------------------------------------------------------------------
def test_main_runs_two_fixtures_and_writes_the_report(sandbox):
    code = batch.main(_run_argv(sandbox, "--limit", "2"))
    assert code == batch.EXIT_PARTIAL
    journal = _journal(sandbox)
    assert len(journal) == 2
    assert {entry["status"] for entry in journal} == {"ok"}
    report = _report(sandbox)
    assert (report["total"], report["compared"], report["identical"],
            report["mismatch"], report["failed"], report["remaining"]) == (3, 2, 2, 0, 0, 1)
    assert report["totalRowsPerSide"]["captured"] == 20
    assert report["totalRowsPerSide"]["expected"] == 20
    assert report["totalRowsPerSide"]["allManifest"] == 30
    assert report["timing"]["averageSecondsPerFixture"] > 0
    assert report["state"] == "finished"
    assert report["fixtures"][0]["status"] == "ok"
    assert report["fixtures"][2]["status"] == "pending"


def test_main_resumes_and_skips_ok_and_mismatch(sandbox):
    fixtures = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])
    sandbox["root"].mkdir(parents=True, exist_ok=True)
    batch.append_journal(sandbox["root"] / "journal.jsonl",
                         _ok_entry(fixtures[0]["id"], 10,
                                   fixtures[0]["retailTraceSha256"],
                                   fixtures[0]["retailTraceSha256"]))
    code = batch.main(_run_argv(sandbox))
    assert code == batch.EXIT_ALL_IDENTICAL  # fx-00 was already final, fx-01/02 ran
    journal = _journal(sandbox)
    assert len(journal) == 3
    assert [entry["fixture_id"] for entry in journal] == ["fx-00", "fx-01", "fx-02"]
    assert not (sandbox["root"] / fixtures[0]["id"]).exists()  # nothing re-ran fx-00
    report = _report(sandbox)
    assert report["identical"] == 3
    assert report["remaining"] == 0


def test_main_reports_a_mismatch_with_both_shas(sandbox, monkeypatch):
    monkeypatch.setenv("V400_BATCH_STUB_MODE", "mismatch")
    code = batch.main(_run_argv(sandbox, "--limit", "1"))
    assert code == batch.EXIT_FAILURES
    report = _report(sandbox)
    assert report["mismatch"] == 1
    assert report["mismatches"][0]["actualSha256"] == "0" * 64
    assert report["mismatches"][0]["manifestSha256"] == sandbox["records"][0]["retailTrace"]["sha256"]
    journal = _journal(sandbox)
    assert journal[0]["status"] == "mismatch"
    assert journal[0]["actual_sha256"] == "0" * 64
    assert journal[0]["manifest_sha256"] == sandbox["records"][0]["retailTrace"]["sha256"]


def test_main_stops_when_a_residual_instance_never_goes_away(sandbox, monkeypatch):
    monkeypatch.setattr(batch, "INSTANCE_PROBE", lambda: [999])
    code = batch.main(_run_argv(sandbox, "--max-guard-failures", "1"))
    assert code == batch.EXIT_FAILURES
    report = _report(sandbox)
    assert report["state"] == "aborted"
    assert "residual" in (report["abortedReason"] or "")
    assert report["remaining"] == 2
    journal = _journal(sandbox)
    assert len(journal) == 1
    assert journal[0]["status"] == "failed"
    assert "residual th08.exe" in journal[0]["error"]
    assert not (sandbox["root"] / "fx-00").exists()


def test_main_dry_run_writes_nothing(sandbox):
    code = batch.main(_run_argv(sandbox, "--dry-run"))
    assert code == batch.EXIT_PARTIAL
    assert not (sandbox["root"] / "journal.jsonl").exists()
    assert not (sandbox["root"] / "report.json").exists()


def test_main_only_runs_the_named_fixture(sandbox):
    code = batch.main(_run_argv(sandbox, "--only", "fx-01"))
    assert code == batch.EXIT_PARTIAL
    assert [entry["fixture_id"] for entry in _journal(sandbox)] == ["fx-01"]


def test_main_stop_file_stops_before_the_next_fixture(sandbox):
    sandbox["root"].mkdir(parents=True, exist_ok=True)
    (sandbox["root"] / batch.STOP_NAME).write_text("stop\n", encoding="utf-8")
    code = batch.main(_run_argv(sandbox))
    assert code == batch.EXIT_PARTIAL
    assert _journal(sandbox) == []
    report = _report(sandbox)
    assert report["remaining"] == 3
    assert "stop file" in (report["abortedReason"] or "")


def test_heartbeat_snapshot_carries_counts_and_rows(sandbox):
    fixtures = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])
    entries = [_ok_entry(fixtures[0]["id"], 10, fixtures[0]["retailTraceSha256"],
                         fixtures[0]["retailTraceSha256"])]
    out = sandbox["tmp"] / "live"
    out.mkdir()
    (out / "rows.jsonl").write_text("[]\n" * 5, encoding="utf-8")
    (out / "status.json").write_text(json.dumps({
        "phase": "playing", "rows": 5, "playbackStarted": True, "gamePid": 1234,
        "foregroundOk": True, "navigationError": None,
        "lastDebugEventAgeSeconds": 0.01, "elapsedSeconds": 42.0,
        "updatedAt": "2026-11-09T10:00:00+0800", "ignored": "x",
    }), encoding="utf-8")
    progress = batch.Progress()
    progress.update(state="running", fixtureId=fixtures[1]["id"], out=str(out),
                    expectedRows=10)
    snapshot = progress.snapshot(fixtures, entries)
    assert snapshot["fixtureId"] == fixtures[1]["id"]
    assert snapshot["rowsSoFar"] == 5
    assert snapshot["fractionOfFixture"] == 0.5
    assert snapshot["observer"]["phase"] == "playing"
    assert snapshot["observer"]["gamePid"] == 1234
    assert "ignored" not in snapshot["observer"]
    assert snapshot["counts"]["identical"] == 1
    assert snapshot["counts"]["remaining"] == 2
    assert snapshot["pid"] == batch.os.getpid()


# ---------------------------------------------------------------------------
# EXP-P0-076: adaptive per-fixture budget
# ---------------------------------------------------------------------------
def test_fixture_budget_formula_and_examples():
    # max(1800, ceil(frames / 50) + 300)
    assert batch.fixture_budget_seconds(149_986) == 3300.0      # the largest fixture
    assert batch.fixture_budget_seconds(135_726) == 3015.0      # normal-6-b
    assert batch.fixture_budget_seconds(135_000) == 3000.0      # the >135k cut
    assert batch.fixture_budget_seconds(99_113) == 2283.0       # 1983 + 300
    assert batch.fixture_budget_seconds(10) == 1800.0           # floor
    assert batch.fixture_budget_seconds(0) == 1800.0
    assert batch.fixture_budget_seconds(None) == 1800.0         # never raises
    assert batch.fixture_budget_seconds(-5) == 1800.0
    # the formula is ceil, not round
    assert batch.fixture_budget_seconds(75_001) == 1801.0
    assert batch.fixture_budget_seconds(75_000) == 1800.0       # exactly the floor


def test_fixture_budget_cap_and_overrides():
    assert batch.BUDGET_MAX_SECONDS == 5400.0
    assert batch.fixture_budget_seconds(400_000) == 5400.0      # 8000 + 300 -> capped
    assert batch.fixture_budget_seconds(149_986, maximum=2000.0) == 2000.0
    # a cap below the floor must not shorten a run below the G0.1 authorization
    assert batch.fixture_budget_seconds(149_986, maximum=600.0) == 1800.0
    assert batch.fixture_budget_seconds(400_000, minimum=2000.0, maximum=1500.0) == 2000.0
    # a measured-rate plan (61.5 rows/s) is shorter than the 50 rows/s plan
    assert batch.fixture_budget_seconds(149_986, rows_per_second=61.5) == 2739.0
    assert batch.fixture_budget_seconds(149_986, minimum=2000.0) == 3300.0
    assert batch.budget_policy()["rowsPerSecondFloor"] == 50.0


def test_the_twelve_large_fixtures_get_a_raised_budget():
    """EXP-P0-075 §6: >135k frames cannot finish inside a fixed 1800 s budget."""
    fixtures = batch.load_fixtures()
    budgets = {fixture["id"]: batch.fixture_budget_seconds(fixture["calculationFrames"])
               for fixture in fixtures}
    large = [fixture["id"] for fixture in fixtures if fixture["calculationFrames"] > 135_000]
    assert set(large) == {
        "normal-6-b", "normal-8-b", "hard-1-b", "hard-8-b", "lunatic-3-b",
        "lunatic-4-b", "lunatic-8-b", "lunatic-11-b", "normal-8-a", "hard-8-a",
        "lunatic-8-a", "lunatic-10-a",
    }
    assert len(large) == 12
    for fixture_id in large:
        assert budgets[fixture_id] > 1800.0
    frames = {fixture["id"]: fixture["calculationFrames"] for fixture in fixtures}
    assert budgets["lunatic-3-b"] == 3300.0                     # 149,986 frames
    assert frames["lunatic-3-b"] == 149_986
    assert max(budgets.values()) == 3300.0
    assert all(value <= batch.BUDGET_MAX_SECONDS for value in budgets.values())
    # the formula raises the budget exactly for the fixtures it needs to:
    # ceil(frames / 50) + 300 > 1800  <=>  frames > 75,000
    raised = {fixture_id for fixture_id, value in budgets.items() if value > 1800.0}
    assert raised == {fixture_id for fixture_id, value in frames.items() if value > 75_000}


# ---------------------------------------------------------------------------
# EXP-P0-076: --start-stage for the [8] Extra fixtures
# ---------------------------------------------------------------------------
def test_start_stage_is_the_first_manifest_stage_index():
    def stage(index):
        return {"index": index, "name": None, "inputRecords": 1, "endScore": 1}

    assert batch.start_stage_for({"stages": [stage(8)]}) == 8
    assert batch.start_stage_for({"stages": [stage(0), stage(1)]}) == 0
    assert batch.start_stage_for({"stages": [stage(8), stage(0)]}) == 8
    assert batch.start_stage_for({"stages": []}) == 0
    assert batch.start_stage_for({}) == 0
    assert batch.first_stage_index({"stages": [{"index": None}, stage(8)]}) == 8


def test_the_twelve_extra_fixtures_use_start_stage_eight():
    fixtures = batch.load_fixtures()
    stages = {fixture["id"]: [step["index"] for step in fixture["stages"]]
              for fixture in fixtures}
    extras = sorted(fixture_id for fixture_id, value in stages.items() if value == [8])
    assert set(extras) == {f"extra-{index}-extra" for index in range(12)}
    for fixture in fixtures:
        expected = 8 if fixture["id"] in extras else 0
        assert batch.start_stage_for(fixture) == expected, fixture["id"]
    # the observer must also receive a start stage that is present in the replay
    for fixture in fixtures:
        assert batch.start_stage_for(fixture) in stages[fixture["id"]]


def _direct_fixture(fixture_id: str = "fx-big", frames: int = 149_986,
                    stage_indexes: tuple[int, ...] = (8,)) -> dict:
    """A fixture dict as ``load_fixtures`` builds it (no manifest, no replay file)."""
    return {
        "id": fixture_id,
        "file": f"{fixture_id}.rpy",
        "replay": ROOT / "third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/"
                          "th8_ud2c65.rpy",
        "replay_sha256": "0" * 64,
        "calculationFrames": frames,
        "retailTraceSha256": hashlib.sha256(fixture_id.encode()).hexdigest(),
        "schemaVersion": 4,
        "stages": [{"index": index, "name": f"Stage {index}",
                    "inputRecords": frames, "endScore": 1000}
                   for index in stage_indexes],
        "present": True,
    }


def _report_for(fixture: dict, *, rows: int, status: str = "ok",
                exit_code: int = 0, sha: str | None = None) -> dict:
    sha = sha or fixture["retailTraceSha256"]
    return {
        "schema": "stub-observer/1",
        "phase": "rows",
        "status": status,
        "result": {"reason": "stub " + status, "exitCode": exit_code},
        "verdict": {
            "rows": rows,
            "calculationFrames": fixture["calculationFrames"],
            "rowsMatch": rows == fixture["calculationFrames"],
            "canonicalTraceSha256": sha,
            "retailTraceSha256": fixture["retailTraceSha256"],
            "shaMatch": sha == fixture["retailTraceSha256"],
            "stageTable": [],
        },
        "elapsedSeconds": 12.5,
    }


def test_run_one_passes_the_adaptive_budget_and_start_stage(tmp_path, monkeypatch):
    recorded = []
    logs = []

    def recording_run(command, **kwargs):
        recorded.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(batch.subprocess, "run", recording_run)
    fixture = _direct_fixture("extra-2-extra", frames=149_986, stage_indexes=(8,))
    entry = batch.run_one(fixture, root=tmp_path, budget_seconds=1800.0,
                          python=sys.executable, observer=tmp_path / "observer.py",
                          guard_timeout=0.05, log=logs.append, python_env={},
                          timeout_slack=60.0, probe=lambda: [], guard_poll=0.01)
    assert len(recorded) == 1
    command = recorded[0]
    assert command[command.index("--budget-seconds") + 1] == "3300"
    assert command[command.index("--start-stage") + 1] == "8"
    assert entry["budgetSeconds"] == 3300.0
    assert entry["budgetBaseSeconds"] == 1800.0
    assert entry["budgetMaxSeconds"] == 5400.0
    assert entry["wallclockGuardSeconds"] == 3360.0
    assert entry["startStage"] == 8
    assert entry["timed_out"] is False
    assert any("budget=3300 s, startStage=8" in message for message in logs)


def test_run_one_keeps_the_floor_for_a_normal_fixture(tmp_path, monkeypatch):
    recorded = []

    def recording_run(command, **kwargs):
        recorded.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(batch.subprocess, "run", recording_run)
    fixture = _direct_fixture("fx-small", frames=10, stage_indexes=(0, 1))
    entry = batch.run_one(fixture, root=tmp_path, budget_seconds=1800.0,
                          python=sys.executable, observer=tmp_path / "observer.py",
                          guard_timeout=0.05, log=lambda _m: None, python_env={},
                          timeout_slack=60.0, probe=lambda: [], guard_poll=0.01)
    command = recorded[0]
    assert command[command.index("--budget-seconds") + 1] == "1800"
    assert command[command.index("--start-stage") + 1] == "0"
    assert entry["budgetSeconds"] == 1800.0
    assert entry["startStage"] == 0


def test_a_wallclock_timeout_is_journalled_as_timeout_and_never_as_success(
        tmp_path, monkeypatch):
    """EXP-P0-076 §3: the rows of a killed run are evidence, not a success."""
    fixture = _direct_fixture("fx-timeout", frames=10, stage_indexes=(0,))
    report = _report_for(fixture, rows=10)          # a complete-looking report

    def timing_out(command, **kwargs):
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "rows.jsonl").write_text("".join(
            json.dumps([index % 1000] * len(FIELDS)) + "\n" for index in range(10)),
            encoding="utf-8")
        (out / "report.json").write_text(json.dumps(report), encoding="utf-8")
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    monkeypatch.setattr(batch.subprocess, "run", timing_out)
    entry = batch.run_one(fixture, root=tmp_path, budget_seconds=1800.0,
                          python=sys.executable, observer=tmp_path / "observer.py",
                          guard_timeout=0.05, log=lambda _m: None, python_env={},
                          timeout_slack=60.0, probe=lambda: [], guard_poll=0.01)
    assert entry["status"] == "timeout"
    assert entry["status"] != batch.STATUS_OK
    assert entry["timed_out"] is True
    assert entry["timeoutSource"] == "batch-wallclock"
    assert entry["rowsAreEvidenceOnly"] is True
    # the row count and the hash comparison survive as evidence ...
    assert entry["rows"] == 10
    assert entry["rows_equal"] is True
    assert entry["sha_equal"] is True
    # ... but they do not grade the run as a success
    assert "wall-clock timeout after 1860 s (budget 1800 s)" in entry["error"]
    assert "evidence only" in entry["error"]


def test_an_observer_watchdog_timeout_is_also_status_timeout(tmp_path, monkeypatch):
    fixture = _direct_fixture("fx-watchdog", frames=10, stage_indexes=(0,))
    report = _report_for(fixture, rows=10, status="timeout", exit_code=7)
    report["budgetSeconds"] = 3300.0

    def watchdog_exit(command, **kwargs):
        out = Path(command[command.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "rows.jsonl").write_text("".join(
            json.dumps([index] * len(FIELDS)) + "\n" for index in range(10)),
            encoding="utf-8")
        (out / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(command, 7, "", "")

    monkeypatch.setattr(batch.subprocess, "run", watchdog_exit)
    entry = batch.run_one(fixture, root=tmp_path, budget_seconds=1800.0,
                          python=sys.executable, observer=tmp_path / "observer.py",
                          guard_timeout=0.05, log=lambda _m: None, python_env={},
                          timeout_slack=60.0, probe=lambda: [], guard_poll=0.01)
    assert entry["status"] == "timeout"
    assert entry["timed_out"] is True
    assert entry["timeoutSource"] == "observer-watchdog"
    assert entry["rowsAreEvidenceOnly"] is True
    assert entry["budgetSeconds"] == 3300.0        # the observer's own budget wins
    assert entry["rows_equal"] is True and entry["sha_equal"] is True
    assert "watchdog" in entry["error"]


def test_timeout_is_retryable_and_counted_as_a_failure():
    fixtures = batch.load_fixtures()[:1]
    fixture_id = fixtures[0]["id"]

    def timeout_entry(attempt: int = 1) -> dict:
        return {"fixture_id": fixture_id, "file": f"{fixture_id}.rpy", "status": "timeout",
                "error": "wall-clock timeout", "rows": 100, "rows_equal": True,
                "sha_equal": False, "elapsed_s": 2100.0, "exit_code": None,
                "timed_out": True, "attempt": attempt,
                "out": "/tmp/a", "endedAt": "2026-11-09T14:00:00+08:00"}

    # not terminal -> it is retried
    assert batch.terminal_entries([timeout_entry()]) == {}
    assert batch.exhausted_entries([timeout_entry()], 2) == {}
    # ... once, like a failure
    assert set(batch.exhausted_entries([timeout_entry(1), timeout_entry(2)], 2)) == {
        fixture_id}

    report = batch.build_report(fixtures, [timeout_entry()], root=Path("/tmp"),
                                budget_seconds=1800.0,
                                retry_policy_info=batch.retry_policy(2))
    assert report["timeout"] == 1
    assert report["failed"] == 0
    assert report["remaining"] == 0
    assert report["failures"][0]["status"] == "timeout"
    assert report["failures"][0]["budgetSeconds"] == batch.fixture_budget_seconds(
        fixtures[0]["calculationFrames"])
    assert batch.exit_code_for(report) == batch.EXIT_FAILURES
    assert "timeout" in report["retryPolicy"]["retryableStatuses"]


def test_report_and_heartbeat_carry_the_budget(sandbox):
    fixtures = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])
    report = batch.build_report(fixtures, [], root=Path("/tmp"), budget_seconds=1800.0,
                                retry_policy_info=batch.retry_policy(2))
    assert report["budgetPolicy"]["rowsPerSecondFloor"] == 50.0
    assert report["budgetPolicy"]["maximumSeconds"] == 5400.0
    assert report["budgetPlan"]["baseSeconds"] == 1800.0
    assert report["budgetPlan"]["plannedSeconds"] == 3 * 1800.0
    assert [detail["budgetSeconds"] for detail in report["fixtures"]] == [1800.0] * 3
    assert [detail["startStage"] for detail in report["fixtures"]] == [0, 0, 0]
    assert "budgetSeconds" in report["journalFieldNotes"]
    assert "startStage" in report["journalFieldNotes"]

    progress = batch.Progress(budget_base=1800.0, budget_max=5400.0)
    progress.update(state="running", fixtureId=fixtures[0]["id"], budgetSeconds=3300.0,
                    startStage=8)
    snapshot = progress.snapshot(fixtures, [])
    assert snapshot["budgetSeconds"] == 3300.0
    assert snapshot["startStage"] == 8
    assert snapshot["budgetPolicy"]["formula"].startswith("max(minimumSeconds")
    assert snapshot["budgetPlan"]["maxSeconds"] == 5400.0


def test_main_retries_a_timed_out_fixture_once_and_keeps_its_evidence(sandbox):
    fixtures = batch.load_fixtures(sandbox["manifest"], sandbox["replay_dir"])
    first = fixtures[0]["id"]
    sandbox["root"].mkdir(parents=True, exist_ok=True)
    (sandbox["root"] / first).mkdir()
    (sandbox["root"] / first / "rows.jsonl").write_text("[]\n", encoding="utf-8")
    batch.append_journal(sandbox["root"] / "journal.jsonl", {
        "fixture_id": first, "file": f"{first}.rpy", "status": "timeout",
        "error": "wall-clock timeout after 2700 s (budget 1800 s)",
        "rows": 1000, "rows_equal": True, "sha_equal": False, "elapsed_s": 2700.0,
        "exit_code": None, "timed_out": True, "attempt": 1,
        "out": str(sandbox["root"] / first), "endedAt": "2026-11-09T14:00:00+08:00",
    })

    code = batch.main(_run_argv(sandbox, "--only", first, "--budget-seconds", "1800"))

    assert code == batch.EXIT_PARTIAL                    # fx-01/fx-02 still pending
    journal = _journal(sandbox)
    assert [entry["status"] for entry in journal] == ["timeout", "ok"]
    assert journal[1]["attempt"] == 2
    assert journal[1]["budgetSeconds"] == 1800.0          # frames=10 -> the floor
    assert journal[1]["startStage"] == 0
    assert (sandbox["root"] / first / "rows.jsonl").is_file()   # nothing deleted
    report = _report(sandbox)
    assert report["timeout"] == 0
    assert report["identical"] == 1
    assert report["budgetPolicy"]["minimumSeconds"] == 1800.0


def test_main_dry_run_prints_the_budget_and_start_stage(sandbox, capsys):
    code = batch.main(_run_argv(sandbox, "--dry-run", "--limit", "1",
                                "--budget-seconds", "1800"))
    assert code == batch.EXIT_PARTIAL
    out = capsys.readouterr().out
    assert "budget=1800s startStage=0" in out
    assert "formula=max(floor, ceil(frames/50)+300)" in out

