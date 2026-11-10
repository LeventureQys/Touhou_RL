"""EXP-P0-072 (v4.0.0 WP1): serial batch driver for the retail trace observer.

The retail TH08 binary is a single-instance, single-window, real-time simulator:
``v400_retail_trace_debugapi.py`` navigates the in-game replay menu with
``SendInput`` scan codes, so it owns the foreground window and there may be
exactly one ``th08.exe`` at a time.  A parallel attempt in EXP-P0-068 killed one
of the two traces, so this driver runs the 108 fixtures of
``third_party/th08/config/replay-fixtures.json`` **strictly serially** and does
nothing else.

What it guarantees (checked, not just documented):

* one fixture at a time, one child process, one ``th08.exe``;
* a pre-run and a post-run guard that waits up to ``--guard-timeout`` (120 s)
  for a residual ``th08.exe`` and refuses to start (or continues after) a run
  while one exists;
* one private output directory per fixture, ``<root>/<fixture_id>``, and a new
  ``attempt-N`` subdirectory instead of deleting a previous attempt;
* a crash-safe journal ``<root>/journal.jsonl``: one ``fsync``-ed JSON line per
  finished fixture with the fields ``fixture_id``, ``file``, ``manifest_sha256``,
  ``actual_sha256``, ``rows``, ``calculationFrames``, ``rows_equal``,
  ``sha_equal``, ``elapsed_s``, ``exit_code``, ``error``.  ``actual_sha256`` is
  always the *official canonical* trace hash (``compare-replay-traces.trace_sha256``:
  compact schema JSON + 25 little-endian uint32 per row), never the SHA-256 of
  ``rows.jsonl``; the file hash is reported separately as ``rows_file_sha256``;
* resumability: a fixture whose journal has a terminal ``ok`` or ``mismatch``
  entry is skipped; a ``failed`` or ``timeout`` entry is retried
  (``--skip-failed`` opts out);
* ``<root>/report.json`` after every fixture (atomic replace) with
  ``total/compared/identical/mismatch/failed/remaining``,
  ``totalRowsPerSide``, cumulative and average timing, a wall-clock projection
  and the per-fixture details;
* ``<root>/heartbeat.json`` every ``--heartbeat-seconds`` so a progress check
  can see the fixture in flight, its row count so far and the batch totals;
* exit codes: ``0`` = all 108 identical, ``3`` = partial (fixtures remaining),
  ``1`` = at least one mismatch, failed or timed-out fixture.

EXP-P0-076 (v4.0.0 WP1) -- the two known-failing classes of the G0.1 batch:

* **per-fixture budget**: ``--budget-seconds`` is now a *floor*.  Each fixture
  gets ``max(floor, ceil(calculationFrames / 50) + 300)`` seconds, capped at
  ``BUDGET_MAX_SECONDS`` (5400 s).  The 12 fixtures above 135k calculation
  frames need 2,195-2,439 s at the measured 61.5-100 rows/s, so a fixed 1800 s
  budget truncated them by construction (truncation => too few rows =>
  ``mismatch``).  The value used is journalled (``budgetSeconds``), reported
  (``fixtures[].budgetSeconds`` / ``budgetPolicy``) and heartbeated.
* **Extra navigation**: the observer's ``--start-stage`` is the *official*
  ``startStage`` (the replay's first stage index), not a row filter.  Passing
  ``--start-stage 0`` for a ``[8]`` replay makes the closed-loop navigator tap
  against the wrong replay-list cursor for 40 s and only then fall back to the
  35 s blind sequence (~48 s); the official value 8 reaches the Extra entry in
  ~7 s (measured on ``.tmp_v400_trace_fmt1/full_extra2`` / EXP-P0-068 §9.6).
  This driver now always passes ``--start-stage <first manifest stage index>``.
* **timeout is not success**: a run killed by the wall-clock guard, or one whose
  observer watchdog fired (``report.status == "timeout"``), is journalled with
  ``status="timeout"`` and is never graded ``ok`` even if the rows happen to
  match.  ``timeout`` is retryable (``--max-attempts``, default 2).

Usage::

    python tools/rl/v400_retail_trace_batch.py --root .tmp_v400_trace_batch
    python tools/rl/v400_retail_trace_batch.py --only easy-0-a        # one fixture
    python tools/rl/v400_retail_trace_batch.py --dry-run --limit 3    # plan only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES_JSON = ROOT / "third_party/th08/config/replay-fixtures.json"
REPLAY_DIR = ROOT / "third_party/th08/tests/fixtures/replays/th08-1.00d/accepted"
OBSERVER = ROOT / "tools/rl/v400_retail_trace_debugapi.py"
DEFAULT_ROOT = ROOT / ".tmp_v400_trace_batch"

JOURNAL_NAME = "journal.jsonl"
REPORT_NAME = "report.json"
HEARTBEAT_NAME = "heartbeat.json"
STOP_NAME = "STOP"

REPORT_SCHEMA = "v400-retail-trace-batch/2"
JOURNAL_SCHEMA = "v400-retail-trace-batch-journal/2"

STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_FAILED = "failed"
STATUS_PENDING = "pending"
#: EXP-P0-076: a run cut off by the wall-clock guard or by the observer's own
#: watchdog.  Its rows are evidence, never a success -- even if the row count and
#: the canonical hash happen to match (a killed run cannot be trusted to have
#: finished its detach/complete.json path).
STATUS_TIMEOUT = "timeout"
#: the only states that make a fixture final for resume purposes.
TERMINAL_STATUSES = (STATUS_OK, STATUS_MISMATCH)
#: EXP-P0-076: states a later session may retry (``ok``/``mismatch`` never are).
RETRYABLE_STATUSES = (STATUS_FAILED, STATUS_TIMEOUT)
#: EXP-P0-075: a failed fixture gets one retry by default, then stays exhausted.
DEFAULT_MAX_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# EXP-P0-076: adaptive per-fixture budget
# ---------------------------------------------------------------------------
#: The plan rate.  EXP-P0-075 §6 measured 61.5 rows/s on ``easy-2-b``, EXP-P0-074
#: §9 measured 79-100 rows/s on the fast fixtures, EXP-P0-068 §9.5 83.8 rows/s;
#: 50 rows/s is the slow floor, so a fixture that runs at the measured rate has
#: a 19-100 % time reserve (the 300 s margin covers attach + navigation + detach).
BUDGET_ROWS_PER_SECOND = 50.0
#: attach (~3 s) + navigation (7-48 s) + detach (~8 s) + slack.
BUDGET_MARGIN_SECONDS = 300.0
#: the G0.1 authorization floor (EXP-P0-068: one continuous run, up to 1800 s).
BUDGET_MIN_SECONDS = 1800.0
#: hard cap so one wedged fixture cannot eat a whole session.  This mirrors
#: ``v400_retail_trace_debugapi.MAX_BUDGET_SECONDS`` (the observer rejects a
#: larger ``--budget-seconds``), asserted equal by the unit tests.
BUDGET_MAX_SECONDS = 5400.0

EXIT_ALL_IDENTICAL = 0
EXIT_FAILURES = 1
EXIT_PARTIAL = 3


def fixture_budget_seconds(calculation_frames,
                           *, minimum: float = BUDGET_MIN_SECONDS,
                           rows_per_second: float = BUDGET_ROWS_PER_SECOND,
                           margin: float = BUDGET_MARGIN_SECONDS,
                           maximum: float = BUDGET_MAX_SECONDS) -> float:
    """EXP-P0-076: per-fixture observer budget in seconds.

    ``max(minimum, ceil(calculationFrames / rows_per_second) + margin)``, capped
    at ``maximum``.  The cap is applied last and can never return less than
    ``minimum`` (a caller asking for ``maximum < minimum`` gets the floor, which
    keeps the G0.1 authorization intact instead of silently shortening runs).

    ``149_986`` frames -> ``ceil(2999.72) + 300 = 3300`` s.
    """
    try:
        frames = max(0, int(calculation_frames))
    except (TypeError, ValueError):
        frames = 0
    try:
        rate = float(rows_per_second)
    except (TypeError, ValueError):
        rate = 0.0
    if not rate > 0:
        rate = BUDGET_ROWS_PER_SECOND
    try:
        floor = float(minimum)
    except (TypeError, ValueError):
        floor = BUDGET_MIN_SECONDS
    try:
        cap = float(maximum)
    except (TypeError, ValueError):
        cap = BUDGET_MAX_SECONDS
    need = float(math.ceil(frames / rate) + float(margin))
    # the cap can never go below the floor: the floor is the G0.1 authorization
    cap = max(cap, floor)
    return float(min(max(floor, need), cap))


def budget_policy(*, minimum: float = BUDGET_MIN_SECONDS,
                  maximum: float = BUDGET_MAX_SECONDS) -> dict:
    """The budget rule, recorded in the report and the heartbeat."""
    return {
        "formula": ("max(minimumSeconds, ceil(calculationFrames / rowsPerSecondFloor) "
                    "+ marginSeconds), capped at maximumSeconds"),
        "minimumSeconds": float(minimum),
        "rowsPerSecondFloor": BUDGET_ROWS_PER_SECOND,
        "marginSeconds": BUDGET_MARGIN_SECONDS,
        "maximumSeconds": float(maximum),
        "source": "EXP-P0-076-WP1-batch-budget-extra",
        "note": ("--budget-seconds is the floor, not the value: the 12 fixtures above 135k "
                 "calculation frames need 2,195-2,439 s at the measured 61.5-100 rows/s "
                 "(EXP-P0-075 §6), so a fixed 1800 s budget truncated them by construction"),
    }


def first_stage_index(fixture: dict) -> int | None:
    """The replay's first manifest stage index (``None`` when it has no stages)."""
    for stage in fixture.get("stages") or []:
        index = stage.get("index")
        if index is not None:
            return int(index)
    return None


def start_stage_for(fixture: dict) -> int:
    """The official ``startStage`` to pass to the observer.

    ``replay-external-observer.menu()`` compares the replay-menu cursor against
    ``settings["startStage"]`` and ``capture-replay.py`` defaults ``startStage``
    to ``replay["stages"][0]["index"]``.  So for the 12 Extra fixtures
    (``stages=[8]``) the value is **8**, which is what reaches the Extra entry in
    ~7 s instead of the 40 s closed-loop timeout + 35 s blind fallback.
    """
    index = first_stage_index(fixture)
    return 0 if index is None else index


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` so a reader never sees a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str | None:
    """SHA-256 of a file's bytes (never used as the canonical trace hash)."""
    import hashlib

    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count_rows(rows_path: Path) -> tuple[int, int]:
    """``(lines, bytes)`` of a ``rows.jsonl`` that may still be growing."""
    rows_path = Path(rows_path)
    try:
        size = rows_path.stat().st_size
    except OSError:
        return 0, 0
    lines = 0
    try:
        with rows_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                lines += chunk.count(b"\n")
    except OSError:
        return 0, size
    return lines, size


# ---------------------------------------------------------------------------
# fixtures / journal / report (pure, unit-tested without a game)
# ---------------------------------------------------------------------------
def load_fixtures(path: Path = FIXTURES_JSON, replay_dir: Path = REPLAY_DIR) -> list[dict]:
    """The 108 manifest entries, normalised for the driver."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fixtures: list[dict] = []
    seen: set[str] = set()
    for record in data.get("fixtures", []):
        trace = record.get("retailTrace") or {}
        fixture_id = record.get("id")
        if not fixture_id:
            raise ValueError(f"fixture without an id in {path}")
        if fixture_id in seen:
            raise ValueError(f"duplicate fixture id {fixture_id!r} in {path}")
        seen.add(fixture_id)
        for key, value in (("calculationFrames", trace.get("calculationFrames")),
                           ("retailTrace.sha256", trace.get("sha256"))):
            if value is None:
                raise ValueError(f"fixture {fixture_id!r} has no {key}")
        replay = Path(replay_dir) / record["file"]
        fixtures.append({
            "id": fixture_id,
            "file": record["file"],
            "replay": replay,
            "replay_sha256": record.get("sha256"),
            "calculationFrames": int(trace["calculationFrames"]),
            "retailTraceSha256": trace["sha256"],
            "schemaVersion": trace.get("schemaVersion"),
            "stages": [
                {key: stage.get(key) for key in
                 ("index", "name", "inputRecords", "endScore")}
                for stage in record.get("stages", [])
            ],
            "present": replay.is_file(),
        })
    return fixtures


def append_journal(path: Path, entry: dict) -> None:
    """Append one entry and fsync it before returning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_journal(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("fixture_id"):
            entries.append(entry)
    return entries


def latest_entries(entries: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for entry in entries:
        latest[entry["fixture_id"]] = entry
    return latest


def terminal_entries(entries: list[dict]) -> dict[str, dict]:
    """fixture_id -> its last ``ok``/``mismatch`` entry (the resume set)."""
    done: dict[str, dict] = {}
    for entry in entries:
        if entry.get("status") in TERMINAL_STATUSES:
            done[entry["fixture_id"]] = entry
    return done


def exhausted_entries(entries: list[dict], max_attempts: int) -> dict[str, dict]:
    """fixture_id -> its last failed/timeout entry once it used up ``max_attempts``.

    EXP-P0-075: a ``failed`` record is *not* final (``TERMINAL_STATUSES``), so by
    default a fixture is retried -- but only up to ``--max-attempts`` times
    (default 2, i.e. one retry).  A fixture that hangs twice is left for a human
    with its evidence intact instead of being retried on every session restart
    forever.  ``--skip-failed`` still means "never retry".

    EXP-P0-076: ``timeout`` is retryable exactly like ``failed`` (a truncated run
    gets one more chance with the adaptive budget).
    """
    attempts: dict[str, int] = {}
    for entry in entries:
        fixture_id = entry.get("fixture_id")
        if fixture_id:
            attempts[fixture_id] = attempts.get(fixture_id, 0) + 1
    exhausted: dict[str, dict] = {}
    for fixture_id, entry in latest_entries(entries).items():
        if (entry.get("status") in RETRYABLE_STATUSES
                and attempts.get(fixture_id, 0) >= max(1, int(max_attempts))):
            exhausted[fixture_id] = entry
    return exhausted


def retry_policy(max_attempts: int, skip_failed: bool = False) -> dict:
    """The resume rule, recorded in the report so it is auditable."""
    return {
        "terminalStatuses": list(TERMINAL_STATUSES),
        "retryableStatuses": list(RETRYABLE_STATUSES),
        "skipTerminal": True,
        "retryFailed": not skip_failed,
        "maxAttemptsPerFixture": max(1, int(max_attempts)),
        "defaultMaxAttempts": DEFAULT_MAX_ATTEMPTS,
        "rule": ("a fixture whose last entry is ok/mismatch is skipped; a failed or timeout "
                 "entry is retried until it has maxAttemptsPerFixture entries, then left "
                 "exhausted for a human"),
    }


def attempt_dir(root: Path, fixture_id: str, attempt: int) -> Path:
    base = Path(root) / fixture_id
    return base if attempt <= 1 else base / f"attempt-{attempt}"


def next_attempt(root: Path, fixture_id: str) -> int:
    """First attempt uses ``<root>/<id>``; retries never delete anything."""
    base = Path(root) / fixture_id
    if not base.is_dir():
        return 1
    try:
        occupied = any(base.iterdir())
    except OSError:
        occupied = True
    if not occupied:
        return 1
    used = [1]
    for child in base.glob("attempt-*"):
        if child.is_dir():
            try:
                used.append(int(child.name.split("-", 1)[1]))
            except (IndexError, ValueError):
                continue
    return max(used) + 1


def summarize(fixtures: list[dict], entries: list[dict], *,
              budget_base: float = BUDGET_MIN_SECONDS,
              budget_max: float = BUDGET_MAX_SECONDS) -> dict:
    """Counts + per-fixture detail over every journal entry so far."""
    latest = latest_entries(entries)
    counts = {STATUS_PENDING: 0, STATUS_OK: 0, STATUS_MISMATCH: 0, STATUS_FAILED: 0,
              STATUS_TIMEOUT: 0}
    details: list[dict] = []
    for fixture in fixtures:
        entry = latest.get(fixture["id"])
        status = entry.get("status") if entry else STATUS_PENDING
        if status not in counts:
            status = STATUS_FAILED
        counts[status] += 1
        budget = fixture_budget_seconds(fixture["calculationFrames"],
                                       minimum=budget_base, maximum=budget_max)
        details.append({
            "fixtureId": fixture["id"],
            "file": fixture["file"],
            "status": status,
            "attempts": sum(1 for item in entries
                            if item["fixture_id"] == fixture["id"]),
            "rows": (entry or {}).get("rows"),
            "calculationFrames": fixture["calculationFrames"],
            "rowsEqual": (entry or {}).get("rows_equal"),
            "shaEqual": (entry or {}).get("sha_equal"),
            "manifestSha256": fixture["retailTraceSha256"],
            "actualSha256": (entry or {}).get("actual_sha256"),
            "elapsedSeconds": (entry or {}).get("elapsed_s"),
            "exitCode": (entry or {}).get("exit_code"),
            "error": (entry or {}).get("error"),
            "out": (entry or {}).get("out"),
            "endedAt": (entry or {}).get("endedAt"),
            # EXP-P0-076: the plan is visible before the fixture runs
            "budgetSeconds": budget,
            "timedOut": bool((entry or {}).get("timed_out")),
            "startStage": start_stage_for(fixture),
        })

    counted = [detail for detail in details if detail["rows"] is not None]
    captured = sum(int(detail["rows"]) for detail in counted)
    expected = sum(int(detail["calculationFrames"]) for detail in counted)
    elapsed = sum(float(detail["elapsedSeconds"] or 0.0) for detail in details)
    attempts = sum(int(detail["attempts"]) for detail in details)
    pending_frames = sum(int(detail["calculationFrames"])
                         for detail in details if detail["status"] == STATUS_PENDING)
    rows_per_second = (captured / elapsed) if elapsed > 0 and captured else None
    seconds_per_fixture = (elapsed / attempts) if attempts else None
    planned = sum(float(detail["budgetSeconds"]) for detail in details)
    planned_pending = sum(float(detail["budgetSeconds"]) for detail in details
                          if detail["status"] == STATUS_PENDING)
    return {
        "counts": {
            "total": len(fixtures),
            "compared": counts[STATUS_OK] + counts[STATUS_MISMATCH],
            "identical": counts[STATUS_OK],
            "mismatch": counts[STATUS_MISMATCH],
            "failed": counts[STATUS_FAILED],
            "timeout": counts[STATUS_TIMEOUT],
            "remaining": counts[STATUS_PENDING],
        },
        "details": details,
        "budget": {
            "baseSeconds": float(budget_base),
            "maxSeconds": float(budget_max),
            "plannedSeconds": round(planned, 1),
            "plannedSecondsForPending": round(planned_pending, 1),
            "fixturesAboveBase": sum(1 for detail in details
                                     if float(detail["budgetSeconds"]) > float(budget_base)),
            "uncappedFixedBudgetSeconds": round(len(fixtures) * float(budget_base), 1),
        },
        "totalRowsPerSide": {
            "definition": (
                "captured = rows summed over the fixtures that produced a row count; "
                "expected = the manifest calculationFrames of those same fixtures; "
                "allManifest = every fixture in replay-fixtures.json"
            ),
            "captured": captured,
            "expected": expected,
            "allManifest": sum(int(fixture["calculationFrames"]) for fixture in fixtures),
        },
        "timing": {
            "totalElapsedSeconds": round(elapsed, 3),
            "attempts": attempts,
            "averageSecondsPerFixture": (round(seconds_per_fixture, 3)
                                         if seconds_per_fixture is not None else None),
            "averageRowsPerSecond": (round(rows_per_second, 2)
                                     if rows_per_second is not None else None),
            "projectedRemainingSecondsByFixture": (
                round(counts[STATUS_PENDING] * seconds_per_fixture, 1)
                if seconds_per_fixture is not None else None),
            "projectedRemainingSecondsByRows": (
                round(pending_frames / rows_per_second, 1)
                if rows_per_second else None),
        },
    }


def build_report(fixtures: list[dict], entries: list[dict], *, root: Path,
                 state: str = "running", started_at: str | None = None,
                 aborted_reason: str | None = None,
                 budget_seconds: float | None = None,
                 retry_policy_info: dict | None = None,
                 budget_max_seconds: float = BUDGET_MAX_SECONDS) -> dict:
    budget_base = (BUDGET_MIN_SECONDS if budget_seconds is None else float(budget_seconds))
    summary = summarize(fixtures, entries, budget_base=budget_base,
                        budget_max=budget_max_seconds)
    counts = summary["counts"]
    mismatches = [
        {"fixtureId": detail["fixtureId"], "file": detail["file"],
         "manifestSha256": detail["manifestSha256"],
         "actualSha256": detail["actualSha256"],
         "rows": detail["rows"], "calculationFrames": detail["calculationFrames"]}
        for detail in summary["details"] if detail["status"] == STATUS_MISMATCH
    ]
    failures = [
        {"fixtureId": detail["fixtureId"], "file": detail["file"],
         "status": detail["status"],
         "error": detail["error"], "exitCode": detail["exitCode"],
         "budgetSeconds": detail["budgetSeconds"],
         "rows": detail["rows"], "calculationFrames": detail["calculationFrames"]}
        for detail in summary["details"]
        if detail["status"] in (STATUS_FAILED, STATUS_TIMEOUT)
    ]
    return {
        "schema": REPORT_SCHEMA,
        "generatedAt": iso_now(),
        "state": state,
        "root": str(root),
        "journal": str(Path(root) / JOURNAL_NAME),
        "startedAt": started_at,
        "abortedReason": aborted_reason,
        "budgetSeconds": budget_seconds,
        "budgetPolicy": budget_policy(minimum=budget_base, maximum=budget_max_seconds),
        "budgetPlan": summary["budget"],
        "total": counts["total"],
        "compared": counts["compared"],
        "identical": counts["identical"],
        "mismatch": counts["mismatch"],
        "failed": counts["failed"],
        "timeout": counts["timeout"],
        "remaining": counts["remaining"],
        "totalRowsPerSide": summary["totalRowsPerSide"],
        "timing": summary["timing"],
        "mismatches": mismatches,
        "failures": failures,
        "retryPolicy": retry_policy_info,
        "journalFieldNotes": {
            "manifest_sha256": "the fixture's retailTrace.sha256 (canonical expectation)",
            "actual_sha256": ("the captured trace's official canonical hash "
                              "(compare-replay-traces.trace_sha256)"),
            "rows_file_sha256": "SHA-256 of rows.jsonl bytes -- not the canonical hash",
            "replay_sha256": "SHA-256 of the .rpy replay file (manifest field sha256)",
            "budgetSeconds": ("EXP-P0-076: the per-fixture --budget-seconds this entry ran "
                              "with (adaptive, >= --budget-seconds, capped at 5400)"),
            "timed_out": ("true when the run was cut off (wall clock or observer watchdog); "
                          "such an entry is status=timeout and is never a success"),
            "startStage": ("EXP-P0-076: the observer --start-stage (the manifest's first "
                           "stage index; Extra fixtures use 8)"),
        },
        "fixtures": summary["details"],
    }


def exit_code_for(report: dict) -> int:
    if report["mismatch"] or report["failed"] or report.get("timeout"):
        return EXIT_FAILURES
    if report["remaining"]:
        return EXIT_PARTIAL
    if report["identical"] == report["total"] and report["total"] > 0:
        return EXIT_ALL_IDENTICAL
    return EXIT_PARTIAL


# ---------------------------------------------------------------------------
# single-instance guard
# ---------------------------------------------------------------------------
def instance_probe() -> list[int]:
    """Live ``th08.exe`` PIDs (``tasklist``); tests replace ``INSTANCE_PROBE``."""
    return running_th08_pids()


#: indirection so a unit test can prove the guard without a game running.
INSTANCE_PROBE = instance_probe


def running_th08_pids() -> list[int]:
    """PIDs of live ``th08.exe`` processes (the tool's own parser)."""
    try:
        completed = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq th08.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    pids: list[int] = []
    for line in (completed.stdout or "").splitlines():
        parts = [part.strip().strip('"') for part in line.split('","')]
        if parts and parts[0].strip('"').lower() == "th08.exe" and len(parts) > 1:
            try:
                pids.append(int(parts[1]))
            except ValueError:
                continue
    return pids


def wait_for_no_instances(timeout_s: float, *, probe=None, poll_s: float = 5.0,
                          sleep=time.sleep) -> dict:
    """Wait until no ``th08.exe`` is alive, at most ``timeout_s`` seconds."""
    return wait_for_instance_limit(0, timeout_s, probe=probe, poll_s=poll_s, sleep=sleep)


def wait_for_instance_limit(limit: int, timeout_s: float, *, probe=None,
                            poll_s: float = 5.0, sleep=time.sleep) -> dict:
    """Wait until at most ``limit`` ``th08.exe`` are alive (``limit=0`` = none).

    EXP-P0-073: the serial mode needs ``limit=0``; the parallel pipeline allows the
    ``N-1`` instances that are still playing in the same group.
    """
    probe = probe or INSTANCE_PROBE
    limit = max(0, int(limit))
    started = time.monotonic()
    pids = list(probe())
    while len(pids) > limit:
        waited = time.monotonic() - started
        if waited >= timeout_s:
            break
        sleep(min(poll_s, max(0.1, timeout_s - waited)))
        pids = list(probe())
    return {
        "ok": len(pids) <= limit,
        "pids": pids,
        "limit": limit,
        "waitedSeconds": round(time.monotonic() - started, 3),
        "timeoutSeconds": timeout_s,
    }


# ---------------------------------------------------------------------------
# canonical trace hash (the only accepted criterion)
# ---------------------------------------------------------------------------
def _observer_module():
    spec = importlib.util.spec_from_file_location("v400_retail_trace_batch_observer",
                                                  OBSERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_canonical_sha(rows_path: Path) -> tuple[str | None, str | None]:
    """Official canonical hash of a captured ``rows.jsonl``.

    Returns ``(sha, error)``.  The observer's ``canonical_trace_sha256`` is the
    single source of truth (it imports ``compare-replay-traces.trace_sha256``),
    so the driver never re-implements the format.
    """
    try:
        return _observer_module().canonical_trace_sha256(Path(rows_path)), None
    except Exception as exc:  # noqa: BLE001 - a bad row must be reported, not hidden
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# one fixture
# ---------------------------------------------------------------------------
def _ends_with(stages: list[dict], observed: dict | None) -> bool | None:
    if not stages or not isinstance(observed, dict):
        return None
    return all(observed.get(str(stage["index"])) == stage.get("endScore")
               for stage in stages)


def _stage_tails_ok(stages: list[dict], stage_frames: dict | None) -> bool | None:
    """The official per-stage unconsumed-input tail invariant.

    ``replay_file.valid_input_tail``: a cleared stage keeps 2/3 input records, the
    final stage 6/7.  ``stageFrames`` is the last calculation frame of the stage,
    so ``inputRecords - stageFrames`` must land in that window.
    """
    if not stages or not isinstance(stage_frames, dict) or not stage_frames:
        return None
    final_index = stages[-1].get("index")
    for stage in stages:
        last_frame = stage_frames.get(str(stage["index"]))
        if last_frame is None or stage.get("inputRecords") is None:
            return None
        tail = (6, 7) if stage.get("index") == final_index else (2, 3)
        if int(stage["inputRecords"]) - int(last_frame) not in tail:
            return False
    return True


def run_one(fixture: dict, *, root: Path, budget_seconds: float, python: str,
            observer: Path, guard_timeout: float, log, python_env=None,
            timeout_slack: float = 900.0, probe=None, sleep=time.sleep,
            guard_poll: float = 5.0, allow_existing: int = 0,
            nav_lock: Path | None = None, nav_lock_timeout: float = 300.0,
            slot: int | None = None, extra_entry: dict | None = None,
            budget_max: float = BUDGET_MAX_SECONDS) -> dict:
    """Run the observer for one fixture and return its journal entry.

    ``budget_seconds`` is the *floor* (``--budget-seconds``): the effective
    per-fixture budget is ``fixture_budget_seconds(calculationFrames,
    minimum=budget_seconds, maximum=budget_max)`` (EXP-P0-076) and is recorded in
    the entry as ``budgetSeconds``.  ``--start-stage`` is
    always passed from the manifest's first stage index (Extra replays = 8).

    ``allow_existing`` > 0 is the parallel pipeline: that many *other* instances may
    be alive already (they are playing, so they need no foreground window), and the
    observer is told ``--allow-existing-instances`` plus the shared ``--nav-lock``
    that serialises the menu navigation.
    """
    fixture_id = fixture["id"]
    attempt = next_attempt(root, fixture_id)
    out_dir = attempt_dir(root, fixture_id, attempt)
    budget = fixture_budget_seconds(fixture["calculationFrames"],
                                   minimum=budget_seconds, maximum=budget_max)
    start_stage = start_stage_for(fixture)
    wallclock = budget + float(timeout_slack)
    entry: dict = {
        # the fields the batch journal contract requires
        "fixture_id": fixture_id,
        "file": fixture["file"],
        "manifest_sha256": fixture["retailTraceSha256"],
        "actual_sha256": None,
        "rows": None,
        "calculationFrames": fixture["calculationFrames"],
        "rows_equal": None,
        "sha_equal": None,
        "elapsed_s": 0.0,
        "exit_code": None,
        "error": None,
        # extras that make an entry auditable on its own
        "schema": JOURNAL_SCHEMA,
        "status": STATUS_FAILED,
        "attempt": attempt,
        "out": str(out_dir),
        "replay": str(fixture["replay"]),
        "replay_sha256": fixture["replay_sha256"],
        "manifest_retail_trace_sha256": fixture["retailTraceSha256"],
        "rows_file_sha256": None,
        "stage_frames_equal": None,
        "end_scores_equal": None,
        "ended": None,
        "natural_replay_exit": None,
        "tool_elapsed_s": None,
        "timed_out": False,
        "timeoutSource": None,
        "guard_before": None,
        "guard_after": None,
        "parallelSlot": slot,
        "parallelAllowExisting": allow_existing,
        # EXP-P0-076: the plan this attempt actually ran with
        "budgetSeconds": budget,
        "budgetBaseSeconds": float(budget_seconds),
        "budgetMaxSeconds": float(budget_max),
        "wallclockGuardSeconds": wallclock,
        "startStage": start_stage,
        "beganAt": iso_now(),
        "endedAt": None,
    }
    if extra_entry:
        entry.update(extra_entry)

    guard = wait_for_instance_limit(allow_existing, guard_timeout, probe=probe,
                                    poll_s=guard_poll, sleep=sleep)
    entry["guard_before"] = guard
    if not guard["ok"]:
        entry["error"] = (f"residual th08.exe after {guard_timeout:g} s: "
                          f"pids={guard['pids']}; refusing to start")
        entry["endedAt"] = iso_now()
        return entry
    if not fixture["present"]:
        entry["error"] = f"replay file missing: {fixture['replay']}"
        entry["endedAt"] = iso_now()
        return entry
    if attempt > 1:
        entry["retryOf"] = str(attempt_dir(root, fixture_id, attempt - 1))

    command = [
        python, "-u", str(observer),
        "--phase", "rows",
        "--out", str(out_dir),
        "--replay", str(fixture["replay"]),
        "--budget-seconds", f"{budget:g}",
        # EXP-P0-076: the official startStage (first manifest stage index).  For
        # the 12 Extra fixtures this is 8 and turns the 40 s closed-loop miss +
        # 35 s blind fallback into a ~7 s closed-loop hit.
        "--start-stage", str(start_stage),
    ]
    if allow_existing > 0:
        # parallel pipeline: other th08.exe may be playing, and menu navigation is
        # serialised across the group by a cross-process lock
        command.append("--allow-existing-instances")
        if nav_lock is not None:
            command += ["--nav-lock", str(nav_lock),
                        "--nav-lock-timeout", f"{nav_lock_timeout:g}"]
        entry["navLock"] = str(nav_lock) if nav_lock is not None else None
    log(f"[{fixture_id}] attempt {attempt} -> {out_dir} "
        f"(budget={budget:g} s, startStage={start_stage})")
    log(f"[{fixture_id}] {' '.join(command)}")
    started = time.monotonic()
    completed = None
    failure = None
    try:
        completed = subprocess.run(
            command, cwd=str(ROOT), env=python_env, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=wallclock,
        )
    except subprocess.TimeoutExpired as exc:
        entry["timed_out"] = True
        entry["timeoutSource"] = "batch-wallclock"
        failure = f"wall-clock timeout after {wallclock:g} s (budget {budget:g} s)"
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        entry["stderrTail"] = stderr[-2000:]
    except OSError as exc:
        failure = f"could not start the observer: {type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    entry["elapsed_s"] = round(elapsed, 3)

    report: dict = {}
    report_path = out_dir / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failure = failure or f"unreadable report.json: {type(exc).__name__}: {exc}"
            report = {}
    verdict = report.get("verdict") or {}
    # EXP-P0-073: navigation accounting for the parallel pipeline.  ``navLockHeld``
    # is the only window during which this instance owned the keyboard.
    parallel_info = report.get("parallel") or {}
    entry["playbackStartedAt"] = parallel_info.get("playbackStartedAt")
    entry["navPath"] = (report.get("navigation") or {}).get("path")
    entry["navLockWaitSeconds"] = (report.get("navLock") or {}).get("waitedSeconds")
    entry["navLockHeldSeconds"] = ((report.get("navLock") or {}).get("release")
                                   or {}).get("heldSeconds")
    complete: dict = {}
    complete_path = out_dir / "complete.json"
    if complete_path.is_file():
        try:
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            complete = {}

    rows = verdict.get("rows")
    actual_sha = verdict.get("canonicalTraceSha256")
    rows_equal = verdict.get("rowsMatch")
    sha_equal = verdict.get("shaMatch")
    rows_path = out_dir / "rows.jsonl"
    if rows is None and rows_path.is_file():
        lines, _ = count_rows(rows_path)
        if lines:
            rows = lines
            entry["rowsSource"] = "rows.jsonl line count (report.json has no verdict)"
    if actual_sha is None and rows_path.is_file():
        actual_sha, sha_error = compute_canonical_sha(rows_path)
        entry["canonicalComputedByDriver"] = actual_sha is not None
        if sha_error:
            entry["canonicalError"] = sha_error
    if rows is not None and fixture["calculationFrames"] is not None:
        if rows_equal is None:
            rows_equal = int(rows) == int(fixture["calculationFrames"])
    if actual_sha is not None:
        if sha_equal is None:
            sha_equal = actual_sha == fixture["retailTraceSha256"]

    entry["rows"] = int(rows) if rows is not None else None
    entry["calculationFrames"] = int(verdict.get("calculationFrames")
                                     or fixture["calculationFrames"])
    entry["actual_sha256"] = actual_sha
    entry["rows_equal"] = bool(rows_equal) if rows_equal is not None else None
    entry["sha_equal"] = bool(sha_equal) if sha_equal is not None else None
    entry["rows_file_sha256"] = sha256_file(rows_path) if rows_path.is_file() else None
    # ``stageFrames`` is *not* a row count: one calculation frame can be hit more
    # than once (the verified ``th8_ud2cd2`` trace has 874 duplicate rows), so
    # ``sum(stageFrames) != rows`` is normal.  The official invariant is the
    # unconsumed-input tail of each stage (``replay_file.valid_input_tail``), which
    # the observer already grades per stage in ``verdict.stageTable``.
    stage_table = verdict.get("stageTable") or []
    if stage_table:
        entry["stage_frames_equal"] = all(step.get("tailOk") is True for step in stage_table)
        entry["end_scores_equal"] = all(step.get("endScoreMatch") is True
                                        for step in stage_table)
    else:
        entry["stage_frames_equal"] = _stage_tails_ok(fixture["stages"],
                                                      complete.get("stageFrames"))
        entry["end_scores_equal"] = _ends_with(fixture["stages"], complete.get("endScores"))
    entry["ended"] = complete.get("ended")
    entry["natural_replay_exit"] = complete.get("naturalReplayExit")
    entry["tool_elapsed_s"] = report.get("elapsedSeconds")

    if completed is not None:
        entry["exit_code"] = completed.returncode
        stdout_tail = (completed.stdout or "")[-800:]
        if stdout_tail:
            entry["stdoutTail"] = stdout_tail
        stderr_tail = (completed.stderr or "")[-2000:]
        if stderr_tail:
            entry["stderrTail"] = stderr_tail
    reason = (report.get("result") or {}).get("reason")
    # EXP-P0-076: the observer's own watchdog writes ``status="timeout"`` (exit 7)
    # when it had to force the ending.  Treat that exactly like a batch wall-clock
    # kill -- the process was cut off, so whatever it wrote is evidence only.
    if not entry["timed_out"] and report.get("status") == STATUS_TIMEOUT:
        entry["timed_out"] = True
        entry["timeoutSource"] = "observer-watchdog"
        reported_budget = report.get("budgetSeconds")
        if isinstance(reported_budget, (int, float)) and reported_budget > 0:
            entry["budgetSeconds"] = float(reported_budget)
        failure = failure or (f"observer watchdog fired (exit {entry['exit_code']}, "
                              f"budget {entry['budgetSeconds']:g} s)")
    if entry["timed_out"]:
        # fail-closed: a cut-off run is *never* a success, even when the rows happen
        # to equal calculationFrames and the canonical hash matches.  The rows stay
        # in the journal as evidence, clearly marked as not accepted.
        entry["status"] = STATUS_TIMEOUT
        entry["rowsAreEvidenceOnly"] = True
        rows_note = (f"; {entry['rows']} rows captured, kept as evidence only"
                     if entry["rows"] is not None else "")
        entry["error"] = ((failure or "timed out")
                          + (f" ({reason})" if reason else "") + rows_note)
    elif failure:
        entry["error"] = failure + (f" ({reason})" if reason else "")
    elif entry["rows_equal"] and entry["sha_equal"]:
        if entry["exit_code"] == 0:
            entry["status"] = STATUS_OK
        else:
            entry["error"] = (f"observer exited {entry['exit_code']} although rows and the "
                              f"canonical hash match")
    elif entry["rows_equal"]:
        entry["status"] = STATUS_MISMATCH
        entry["error"] = reason or (f"rows match but the canonical hash {actual_sha} "
                                    f"differs from {fixture['retailTraceSha256']}")
    else:
        entry["error"] = reason or (
            f"incomplete trace: {entry['rows']} of {entry['calculationFrames']} rows "
            f"(exit {entry['exit_code']})")

    post = wait_for_instance_limit(allow_existing, guard_timeout, probe=probe,
                                   poll_s=guard_poll, sleep=sleep)
    entry["guard_after"] = post
    if not post["ok"]:
        note = f"residual th08.exe after the run: pids={post['pids']}"
        entry["error"] = (entry["error"] + "; " + note) if entry["error"] else note
    entry["endedAt"] = iso_now()
    if entry["status"] == STATUS_OK:
        log(f"[{fixture_id}] ok rows={entry['rows']} sha={actual_sha} "
            f"({entry['elapsed_s']:.1f} s)")
    else:
        log(f"[{fixture_id}] {entry['status']} rows={entry['rows']} "
            f"sha={actual_sha} error={entry['error']}")
    return entry


def read_observer_status(path: Path) -> dict | None:
    """The observer's live ``<out>/status.json``, trimmed for the heartbeat."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    keys = ("phase", "rows", "playbackStarted", "gamePid", "foregroundOk",
            "navigationError", "lastDebugEventAgeSeconds", "elapsedSeconds", "updatedAt")
    return {key: data.get(key) for key in keys if key in data}


# ---------------------------------------------------------------------------
# heartbeat
# ---------------------------------------------------------------------------
class Progress:
    """Shared, thread-safe view of the batch for ``heartbeat.json``."""

    def __init__(self, *, budget_base: float = BUDGET_MIN_SECONDS,
                 budget_max: float = BUDGET_MAX_SECONDS) -> None:
        self._lock = threading.Lock()
        self._payload: dict = {"state": "starting", "fixtureId": None, "out": None,
                               "budgetSeconds": None, "startStage": None}
        self._started = time.monotonic()
        #: EXP-P0-076: the heartbeat carries the budget rule and the current value.
        self.budget_base = float(budget_base)
        self.budget_max = float(budget_max)

    def update(self, **fields) -> None:
        with self._lock:
            self._payload.update(fields)

    def snapshot(self, fixtures: list[dict], entries: list[dict]) -> dict:
        with self._lock:
            payload = dict(self._payload)
        out = payload.get("out")
        rows_so_far = rows_bytes = None
        if out:
            rows_so_far, rows_bytes = count_rows(Path(out) / "rows.jsonl")
            payload["rowsSoFar"] = rows_so_far
            payload["rowsBytes"] = rows_bytes
            status = read_observer_status(Path(out) / "status.json")
            if status:
                payload["observer"] = status
            expected = payload.get("expectedRows")
            if expected and rows_so_far is not None:
                payload["fractionOfFixture"] = round(rows_so_far / expected, 4)
        summary = summarize(fixtures, entries, budget_base=self.budget_base,
                            budget_max=self.budget_max)
        payload.update({
            "updatedAt": iso_now(),
            "pid": os.getpid(),
            "batchElapsedSeconds": round(time.monotonic() - self._started, 1),
            "counts": summary["counts"],
            "timing": summary["timing"],
            "budgetPolicy": budget_policy(minimum=self.budget_base,
                                          maximum=self.budget_max),
            "budgetPlan": summary["budget"],
        })
        return payload


def start_heartbeat(path: Path, progress: Progress, fixtures: list[dict],
                    entries: list[dict], interval: float, log) -> threading.Event:
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval):
            try:
                write_json_atomic(path, progress.snapshot(fixtures, entries))
            except Exception as exc:  # noqa: BLE001 - a heartbeat must never kill a run
                log(f"heartbeat failed: {type(exc).__name__}: {exc}")

    threading.Thread(target=loop, name="v400-batch-heartbeat", daemon=True).start()
    return stop


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_JSON)
    parser.add_argument("--replay-dir", type=Path, default=REPLAY_DIR)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--observer", type=Path, default=OBSERVER)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--budget-seconds", type=float, default=BUDGET_MIN_SECONDS,
                        help="floor of the per-fixture budget (EXP-P0-076): each fixture "
                             "gets max(this, ceil(calculationFrames / 50) + 300) seconds")
    parser.add_argument("--budget-max-seconds", type=float, default=BUDGET_MAX_SECONDS,
                        help="hard cap for the adaptive per-fixture budget "
                             f"(default {BUDGET_MAX_SECONDS:g}, also the observer's cap)")
    parser.add_argument("--guard-timeout", type=float, default=120.0)
    parser.add_argument("--guard-poll", type=float, default=5.0)
    parser.add_argument("--max-guard-failures", type=int, default=2)
    parser.add_argument("--wallclock-slack", type=float, default=900.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="run at most N pending fixtures (smoke tests)")
    parser.add_argument("--only", default=None,
                        help="comma-separated fixture ids to run (in this order)")
    parser.add_argument("--first", default=None,
                        help="comma-separated fixture ids to run before the rest")
    parser.add_argument("--skip-failed", action="store_true",
                        help="do not retry fixtures whose last attempt failed")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help="journal entries a fixture may accumulate before a failed or "
                             "timed-out record is left exhausted (EXP-P0-075/076; "
                             "default 2 = one retry)")
    parser.add_argument("--stop-file", type=Path, default=None,
                        help=f"stop before the next fixture when this file exists "
                             f"(default <root>/{STOP_NAME})")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and write nothing")
    args = parser.parse_args(argv)

    if args.budget_seconds < 1.0:
        parser.error("--budget-seconds must be >= 1")
    if not 1.0 <= args.budget_max_seconds <= BUDGET_MAX_SECONDS:
        parser.error(f"--budget-max-seconds must be in [1, {BUDGET_MAX_SECONDS:g}] "
                     f"(the observer refuses a larger --budget-seconds)")
    if args.budget_seconds > args.budget_max_seconds:
        parser.error("--budget-seconds (floor) must not exceed --budget-max-seconds (cap)")

    root = Path(args.root)
    log_path = Path(args.log) if args.log else root / "batch.log"
    log_stream = None
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("a", encoding="utf-8", newline="\n")
    session_started = iso_now()

    def log(message: str) -> None:
        line = f"[{iso_now()}] {message}"
        print(line, flush=True)
        if log_stream is not None:
            log_stream.write(line + "\n")
            log_stream.flush()

    fixtures = load_fixtures(args.fixtures, args.replay_dir)
    journal_path = root / JOURNAL_NAME
    entries = read_journal(journal_path)
    done = terminal_entries(entries)
    policy = retry_policy(args.max_attempts, args.skip_failed)
    exhausted: dict[str, dict] = {}
    if args.skip_failed:
        for fixture_id, entry in latest_entries(entries).items():
            if entry.get("status") in RETRYABLE_STATUSES:
                done.setdefault(fixture_id, entry)
    else:
        # EXP-P0-075: a failed record is retried, but only until --max-attempts
        # entries exist for that fixture.
        exhausted = exhausted_entries(entries, args.max_attempts)
        for fixture_id, entry in exhausted.items():
            done.setdefault(fixture_id, entry)

    pending = [fixture for fixture in fixtures if fixture["id"] not in done]
    if args.only:
        wanted = [item.strip() for item in args.only.split(",") if item.strip()]
        by_id = {fixture["id"]: fixture for fixture in fixtures}
        unknown = [item for item in wanted if item not in by_id]
        if unknown:
            log(f"unknown fixture ids: {unknown}")
            if log_stream:
                log_stream.close()
            return 2
        pending = [by_id[item] for item in wanted if item not in done]
    elif args.first:
        wanted = [item.strip() for item in args.first.split(",") if item.strip()]
        by_id = {fixture["id"]: fixture for fixture in fixtures}
        head = [by_id[item] for item in wanted if item in by_id and item not in done]
        head_ids = {fixture["id"] for fixture in head}
        pending = head + [fixture for fixture in pending if fixture["id"] not in head_ids]
    if args.limit is not None:
        pending = pending[: max(0, args.limit)]

    log("=" * 78)
    log(f"session start {session_started} pid={os.getpid()}")
    log(f"argv: {' '.join(sys.argv[1:])}")
    log(f"fixtures={len(fixtures)} journal={journal_path}")
    log(f"resume: {len(done)} terminal of {len(fixtures)}; "
        f"pending {len(fixtures) - len(done)}, this session {len(pending)}")
    log(f"retry policy: skip {list(TERMINAL_STATUSES)}; retry "
        f"{list(RETRYABLE_STATUSES)} up to "
        f"{policy['maxAttemptsPerFixture']} attempts"
        + (" (--skip-failed: never)" if args.skip_failed else "")
        + (f"; exhausted {sorted(exhausted)}" if exhausted else ""))
    planned_budgets = {fixture["id"]: fixture_budget_seconds(
        fixture["calculationFrames"], minimum=args.budget_seconds,
        maximum=args.budget_max_seconds) for fixture in fixtures}
    raised = sorted(fixture_id for fixture_id, value in planned_budgets.items()
                    if value > args.budget_seconds)
    log(f"budget: floor={args.budget_seconds:g} s cap={args.budget_max_seconds:g} s "
        f"formula=max(floor, ceil(frames/{BUDGET_ROWS_PER_SECOND:g})"
        f"+{BUDGET_MARGIN_SECONDS:g}); raised for {len(raised)} of {len(fixtures)} "
        f"fixtures, max={max(planned_budgets.values()) if planned_budgets else 0:g} s")
    log(f"start-stage: passed from each fixture's first manifest stage index "
        f"(Extra fixtures use 8)")
    log(f"guard={args.guard_timeout:g} s observer={args.observer} python={args.python}")
    if not fixtures:
        log("no fixtures; nothing to do")
        if log_stream:
            log_stream.close()
        return 2

    if args.dry_run:
        for fixture in pending:
            attempt = next_attempt(root, fixture["id"])
            budget = fixture_budget_seconds(fixture["calculationFrames"],
                                           minimum=args.budget_seconds,
                                           maximum=args.budget_max_seconds)
            log(f"would run {fixture['id']} ({fixture['file']}, "
                f"{fixture['calculationFrames']} frames) -> "
                f"{attempt_dir(root, fixture['id'], attempt)} "
                f"budget={budget:g}s startStage={start_stage_for(fixture)}")
        if log_stream:
            log_stream.close()
        return EXIT_PARTIAL

    python_env = dict(os.environ)
    python_env.pop("PYTHONPATH", None)
    python_env["PYTHONIOENCODING"] = "utf-8"
    stop_file = Path(args.stop_file) if args.stop_file else root / STOP_NAME

    progress = Progress(budget_base=args.budget_seconds,
                        budget_max=args.budget_max_seconds)
    heartbeat = start_heartbeat(root / HEARTBEAT_NAME, progress, fixtures, entries,
                                max(1.0, args.heartbeat_seconds), log)
    aborted_reason = None
    guard_failures = 0
    report: dict = {}
    try:
        report = build_report(fixtures, entries, root=root, state="running",
                              started_at=session_started,
                              budget_seconds=args.budget_seconds, retry_policy_info=policy,
                              budget_max_seconds=args.budget_max_seconds)
        write_json_atomic(root / REPORT_NAME, report)
        for position, fixture in enumerate(pending, 1):
            if stop_file.exists():
                aborted_reason = f"stop file present: {stop_file}"
                log(aborted_reason)
                break
            attempt = next_attempt(root, fixture["id"])
            progress.update(state="running", fixtureId=fixture["id"],
                            out=str(attempt_dir(root, fixture["id"], attempt)),
                            expectedRows=fixture["calculationFrames"], attempt=attempt,
                            budgetSeconds=planned_budgets.get(fixture["id"]),
                            startStage=start_stage_for(fixture))
            log(f"--- fixture {position}/{len(pending)} {fixture['id']} "
                f"({fixture['file']}, {fixture['calculationFrames']} frames, "
                f"budget {planned_budgets.get(fixture['id'], args.budget_seconds):g} s, "
                f"startStage {start_stage_for(fixture)})")
            entry = run_one(
                fixture, root=root, budget_seconds=args.budget_seconds,
                python=args.python, observer=args.observer,
                guard_timeout=args.guard_timeout, log=log, python_env=python_env,
                timeout_slack=args.wallclock_slack, guard_poll=args.guard_poll,
                budget_max=args.budget_max_seconds,
            )
            entries.append(entry)
            if not args.dry_run:
                append_journal(journal_path, entry)
            report = build_report(fixtures, entries, root=root, state="running",
                                  started_at=session_started,
                                  aborted_reason=aborted_reason,
                                  budget_seconds=args.budget_seconds,
                                  retry_policy_info=policy,
                                  budget_max_seconds=args.budget_max_seconds)
            write_json_atomic(root / REPORT_NAME, report)
            log(f"progress: {report['identical']}/{report['total']} identical, "
                f"{report['mismatch']} mismatch, {report['failed']} failed, "
                f"{report['timeout']} timeout, "
                f"{report['remaining']} remaining, "
                f"avg {report['timing']['averageSecondsPerFixture']} s/fixture")
            if entry["status"] in RETRYABLE_STATUSES and not (
                    entry.get("guard_before") or {}).get("ok", True):
                guard_failures += 1
                if guard_failures >= max(1, args.max_guard_failures):
                    aborted_reason = (
                        f"{guard_failures} consecutive pre-run guards found a residual "
                        f"th08.exe; stopping instead of burning one guard timeout per fixture")
                    log(aborted_reason)
                    break
            else:
                guard_failures = 0
            if (entry.get("guard_after") or {}).get("ok") is False:
                aborted_reason = (
                    f"th08.exe survived {args.guard_timeout:g} s after {fixture['id']}: "
                    f"pids={entry['guard_after']['pids']}")
                log(aborted_reason)
                break
    except KeyboardInterrupt:
        aborted_reason = "KeyboardInterrupt"
        log(aborted_reason)
    finally:
        heartbeat.set()
        report = build_report(fixtures, entries, root=root,
                              state="aborted" if aborted_reason else "finished",
                              started_at=session_started,
                              aborted_reason=aborted_reason,
                              budget_seconds=args.budget_seconds, retry_policy_info=policy,
                              budget_max_seconds=args.budget_max_seconds)
        write_json_atomic(root / REPORT_NAME, report)
        code = exit_code_for(report)
        log(f"session end: total={report['total']} identical={report['identical']} "
            f"mismatch={report['mismatch']} failed={report['failed']} "
            f"timeout={report['timeout']} "
            f"remaining={report['remaining']} exit={code}"
            + (f" aborted={aborted_reason}" if aborted_reason else ""))
        log(f"report: {root / REPORT_NAME}")
        progress.update(state="stopped")
        if log_stream:
            log_stream.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
