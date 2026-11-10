#!/usr/bin/env python3
"""WP1 fallback C: differential replay retrace (harness ON vs portable OFF).

Both builds run the same real replay through the ordinary developer-replay
autoplay path with ``TH08_REPLAY_TRACE_RECORD_ONLY=1``.  The recorder then
writes its 25-field row stream without comparing against a retail reference,
and the two row streams are compared byte for byte (and the two
``complete.json`` files too).

This is a differential check (ON vs OFF), *not* a bit-exact comparison against
the retail executable.  See ``Document/Update/Alpha/v4.0.0/experiments/
EXP-P0-051-WP1-differential-replay-retrace.md`` and ``EXP-P0-055`` for the
reduced proof strength.

The tool is fail-closed:

* both builds must expose ``th08::modern::ReplayTraceRecord(`` through ``nm -C``;
* every run must write ``complete.json`` with ``ended`` and
  ``naturalReplayExit`` true, an empty ``errors`` list, and a row count that
  matches the row file size exactly (25 little-endian u32 per row);
* a case only counts as identical when both runs are clean and the row files
  *and* the completion files are byte-identical;
* the official accepted corpus is checked against
  ``third_party/th08/config/replay-fixtures.json`` (names, count and SHA-256)
  before any replay is played.

Modes
-----

Single/repeated explicit replays (original mode)::

    python3 tools/rl/v400_replay_retrace.py \
        --on-build  /root/Touhou_RL/third_party/th08-rl/build/rl_retrace/th08-modern \
        --off-build /root/Touhou_RL/third_party/th08-rl/build/portable-retrace/th08-modern \
        --replay /mnt/d/WorkShop/Touhou_RL/th08_replay/Th8_ud1858.rpy \
        --output-dir .tmp_v400_replay_retrace/run-20261109

Sharded, resumable full-corpus run (108 official fixtures, 4 shards of 27)::

    python3 tools/rl/v400_replay_retrace.py --all --shard 1/4 \
        --on-build  .../build/rl_retrace/th08-modern \
        --off-build .../build/portable-retrace/th08-modern \
        --output-dir .tmp_v400_replay_retrace_108 --resume \
        --timeout 300 --retries 1

Every finished case is appended (and fsynced) to ``<out>/shard-<n>.jsonl``;
restarting with ``--resume`` skips every case already present in the journal.
``--merge`` folds all journals into ``<out>/report.json``::

    python3 tools/rl/v400_replay_retrace.py --merge --expected-total 108 \
        --on-build ... --off-build ... --output-dir .tmp_v400_replay_retrace_108

Both variants must be built in their own build directories; the shared
``build/rl`` and ``build/portable-linux-x86_64`` artifacts are not used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import time

ROW_FIELD_COUNT = 25
ROW_STRUCT = struct.Struct("<25I")
# Mirrors the row[] initializer order in
# third_party/th08-rl/src/modern/linux/replay_trace.cpp.
ROW_FIELDS = (
    "frame", "stage", "replayFrame", "gameplayFrame", "score", "graze",
    "deaths", "lives", "bombs", "power", "x", "y", "z", "playerState",
    "rngSeed", "rngGeneration", "input", "pointItems", "timeOrbs", "gauge",
    "clock", "pointItemValue", "character", "shotType", "difficulty",
)
TRACE_ENV_PREFIX = "TH08_REPLAY_TRACE_"
PASSTHROUGH_ENV_PREFIXES = ("TH08_", "SDL_", "LP_", "LIBGL_", "GALLIUM_")
DEFAULT_TIMEOUT_SECONDS = 300
MAXIMUM_TIMEOUT_SECONDS = 300
DEFAULT_MAXIMUM_RETRIES = 1
DEFAULT_GAME_DATA_DIR = Path("/root/v400-data/game")
DEFAULT_REPLAY_HELPER = Path("/root/Touhou_RL/third_party/th08-rl/scripts/replay_file.py")
DEFAULT_FIXTURES_DIR = Path(
    "/root/Touhou_RL/third_party/th08/tests/fixtures/replays/th08-1.00d/accepted")
DEFAULT_MANIFEST = Path("/root/Touhou_RL/third_party/th08/config/replay-fixtures.json")
DEFAULT_EXPECTED_TOTAL = 108
DEFAULT_SHARDS = 4
PROOF_STRENGTH = ("differential (harness ON vs portable OFF); not a bit-exact "
                  "comparison against the retail executable")
TIMEOUT_EXIT_CODES = (124, 137)
CASE_STATUSES = ("identical", "different", "timeout", "failed")


class CorpusError(RuntimeError):
    """The accepted corpus does not match its manifest exactly."""


def muted_config() -> bytes:
    """Supervisor::LoadConfig defaults with BGM/SFX off and a windowed display.

    Byte-for-byte the same 60-byte configuration the upstream portable replay
    suite writes (``scripts/capture-portable-replay.py`` -> ``muted_config``).
    """
    cfg = bytearray(60)
    struct.pack_into("<10hI2h", cfg, 0, 0, 1, 2, 4, -1, -1, -1, -1, 3, 0, 0x80001, 600, 600)
    cfg[0x1C:0x27] = bytes([2, 3, 0, 0, 0, 1, 1, 0, 2, 0, 0])
    return bytes(cfg)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_environment(base: dict, runtime: Path, output: Path, stage: int) -> dict:
    """Build the identical environment both builds must run under.

    The suite's documented playback profile (fast clock, no rasterization,
    dummy audio) is used so a replay of this size finishes inside the bound.
    """
    environment = {key: value for key, value in base.items() if not key.startswith(TRACE_ENV_PREFIX)}
    environment.update(
        SDL_AUDIODRIVER="dummy",
        LIBGL_ALWAYS_SOFTWARE="1",
        GALLIUM_DRIVER="llvmpipe",
        LP_NUM_THREADS="0",
        LP_NO_RAST="true",
        TH08_AUTOPLAY_REPLAY="./replay/audit.rpy",
        TH08_AUTOPLAY_STAGE=str(stage),
        TH08_RENDER_AUDIT_FAST="1",
        TH08_RENDER_AUDIT="0",
        TH08_PORTABLE_CAPTURE=str(output),
        TH08_REPLAY_TRACE_RECORD_ONLY="1",
        TH08_REPLAY_TRACE_ROWS=str(runtime / "rows.bin"),
        TH08_REPLAY_TRACE_COMPLETION=str(output / "complete.json"),
    )
    return environment


def capture_command(executable: Path, runtime: Path, timeout: int) -> list:
    return [
        "xvfb-run", "-a", "-s", "-screen 0 1024x768x24",
        "timeout", "--signal=TERM", "--kill-after=5s", f"{timeout}s",
        str(executable), "--data-dir", str(runtime),
    ]


def first_difference(on_bytes: bytes, off_bytes: bytes):
    """Return None for equal streams, else the first differing row or length."""
    row_size = ROW_STRUCT.size
    common_rows = min(len(on_bytes), len(off_bytes)) // row_size
    for index in range(common_rows):
        start = index * row_size
        if on_bytes[start:start + row_size] != off_bytes[start:start + row_size]:
            on_row = struct.unpack_from(ROW_STRUCT.format, on_bytes, start)
            off_row = struct.unpack_from(ROW_STRUCT.format, off_bytes, start)
            fields = [name for name, left, right in zip(ROW_FIELDS, on_row, off_row) if left != right]
            return {
                "kind": "row",
                "rowIndex": index,
                "rowNumber": index + 1,
                "differingFields": fields,
                "on": dict(zip(ROW_FIELDS, on_row)),
                "off": dict(zip(ROW_FIELDS, off_row)),
            }
    if len(on_bytes) == len(off_bytes):
        return None
    return {
        "kind": "length",
        "rowsCompared": common_rows,
        "onRows": len(on_bytes) // row_size,
        "offRows": len(off_bytes) // row_size,
        "onBytes": len(on_bytes),
        "offBytes": len(off_bytes),
    }


def compare_rows_files(on_path: Path, off_path: Path) -> dict:
    if not Path(on_path).is_file() or not Path(off_path).is_file():
        return {
            "identical": False,
            "onRows": 0,
            "offRows": 0,
            "onBytes": 0,
            "offBytes": 0,
            "onSha256": None,
            "offSha256": None,
            "firstDifference": {"kind": "missing", "onExists": Path(on_path).is_file(),
                                "offExists": Path(off_path).is_file()},
        }
    on_bytes = Path(on_path).read_bytes()
    off_bytes = Path(off_path).read_bytes()
    row_size = ROW_STRUCT.size
    result = {
        "identical": on_bytes == off_bytes,
        "onRowBytes": len(on_bytes),
        "offRowBytes": len(off_bytes),
        "onRows": len(on_bytes) // row_size,
        "offRows": len(off_bytes) // row_size,
        "onSha256": sha256_bytes(on_bytes),
        "offSha256": sha256_bytes(off_bytes),
        "firstDifference": first_difference(on_bytes, off_bytes),
    }
    if len(on_bytes) % row_size or len(off_bytes) % row_size:
        result["invalidRowSize"] = True
    return result


def first_complete_difference(on_text, off_text):
    """Describe the first differing top-level key of two completion documents."""
    if on_text is None or off_text is None:
        return {"kind": "missing", "onPresent": on_text is not None,
                "offPresent": off_text is not None}
    try:
        on_value, off_value = json.loads(on_text), json.loads(off_text)
    except ValueError:
        return {"kind": "unparsable"}
    if not isinstance(on_value, dict) or not isinstance(off_value, dict):
        return {"kind": "shape"}
    for key in sorted(set(on_value) | set(off_value)):
        if on_value.get(key) != off_value.get(key):
            return {"kind": "key", "key": key, "on": on_value.get(key), "off": off_value.get(key)}
    return {"kind": "text", "detail": "documents serialize differently but parse equal"}


def compare_completion_files(on_path: Path, off_path: Path) -> dict:
    on_path, off_path = Path(on_path), Path(off_path)
    on_exists, off_exists = on_path.is_file(), off_path.is_file()
    if not on_exists or not off_exists:
        return {
            "identical": False,
            "onExists": on_exists,
            "offExists": off_exists,
            "onSha256": sha256_file(on_path) if on_exists else None,
            "offSha256": sha256_file(off_path) if off_exists else None,
            "firstDifference": {"kind": "missing", "onPresent": on_exists, "offPresent": off_exists},
        }
    on_bytes, off_bytes = on_path.read_bytes(), off_path.read_bytes()
    return {
        "identical": on_bytes == off_bytes,
        "onExists": True,
        "offExists": True,
        "onBytes": len(on_bytes),
        "offBytes": len(off_bytes),
        "onSha256": sha256_bytes(on_bytes),
        "offSha256": sha256_bytes(off_bytes),
        "firstDifference": None if on_bytes == off_bytes else first_complete_difference(
            on_bytes, off_bytes),
    }


def complete_problems(complete, rows_path: Path) -> list:
    """Fail-closed checks on the recorder's completion metadata."""
    problems = []
    if not isinstance(complete, dict):
        return ["complete.json is not a JSON object"]
    if complete.get("ended") is not True:
        problems.append("complete.json ended != true")
    if complete.get("naturalReplayExit") is not True:
        problems.append("complete.json naturalReplayExit != true")
    errors = complete.get("errors")
    if errors != []:
        problems.append(f"complete.json errors = {errors!r}")
    rows = complete.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        problems.append(f"complete.json rows is not a positive integer: {rows!r}")
        return problems
    if not Path(rows_path).is_file():
        problems.append("rows file missing")
        return problems
    size = Path(rows_path).stat().st_size
    if size != rows * ROW_STRUCT.size:
        problems.append(f"rows file size {size} != rows {rows} * {ROW_STRUCT.size}")
    return problems


def run_capture(executable: Path, replay: Path, stage: int, game_data: Path,
                case_dir: Path, timeout: int) -> dict:
    """Run one side once.  The caller owns retries via distinct directories."""
    case_dir = Path(case_dir)
    if case_dir.exists():
        raise FileExistsError(f"refusing to reuse capture directory: {case_dir}")
    runtime = case_dir / "runtime"
    (runtime / "replay").mkdir(parents=True)
    for name in ("th08.dat", "thbgm.dat"):
        (runtime / name).symlink_to(Path(game_data) / name)
    (runtime / "th08.cfg").write_bytes(muted_config())
    shutil.copyfile(Path(replay), runtime / "replay" / "audit.rpy")
    rows_path = runtime / "rows.bin"
    complete_path = case_dir / "complete.json"
    environment = capture_environment(dict(os.environ), runtime, case_dir, stage)
    command = capture_command(Path(executable), runtime, timeout)
    (case_dir / "command.json").write_text(json.dumps({
        "command": command,
        "environment": {key: environment[key] for key in sorted(environment)
                        if key.startswith(PASSTHROUGH_ENV_PREFIXES)},
        "timeoutSeconds": timeout,
    }, indent=2) + "\n")

    started = time.monotonic()
    timed_out = False
    exit_code = None
    with (case_dir / "run.log").open("wb") as log:
        try:
            completed = subprocess.run(command, cwd=str(case_dir), env=environment,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       timeout=timeout + 60)
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    seconds = round(time.monotonic() - started, 3)
    if exit_code in TIMEOUT_EXIT_CODES:
        timed_out = True

    problems = []
    if timed_out:
        problems.append(f"per-run timeout ({timeout}s) or runner safety timeout")
    elif exit_code != 0:
        problems.append(f"process exit code {exit_code}")
    complete = None
    if not complete_path.is_file():
        problems.append("complete.json missing")
    else:
        try:
            complete = json.loads(complete_path.read_text())
        except (OSError, ValueError) as error:
            problems.append(f"complete.json unreadable: {error}")
        if complete is not None:
            problems.extend(complete_problems(complete, rows_path))

    result = {
        "executable": str(executable),
        "executableSha256": sha256_file(Path(executable)),
        "exitCode": exit_code,
        "timedOut": timed_out,
        "seconds": seconds,
        "rowsPath": str(rows_path),
        "rowsPresent": rows_path.is_file(),
        "rowsBytes": rows_path.stat().st_size if rows_path.is_file() else 0,
        "rowsSha256": sha256_file(rows_path) if rows_path.is_file() else None,
        "completePath": str(complete_path),
        "completeSha256": sha256_file(complete_path) if complete_path.is_file() else None,
        "complete": complete,
        "runtimeScoreSha256": (sha256_file(runtime / "score.dat")
                               if (runtime / "score.dat").is_file() else None),
        "runtimeConfigSha256": (sha256_file(runtime / "th08.cfg")
                                if (runtime / "th08.cfg").is_file() else None),
        "problems": problems,
    }
    (case_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def side_status(result: dict) -> str:
    """Classify one side: clean / timeout / failed (fail-closed)."""
    if result.get("timedOut") or result.get("exitCode") in TIMEOUT_EXIT_CODES:
        return "timeout"
    if result.get("problems"):
        return "failed"
    return "clean"


def run_side(executable: Path, replay: Path, stage: int, game_data: Path,
             side_dir: Path, timeout: int, retries: int = DEFAULT_MAXIMUM_RETRIES) -> dict:
    """Run one side, retrying the whole process once (or ``retries`` times).

    Attempt directories are numbered past any that already exist, so a run that
    was killed mid-case can be resumed without deleting earlier evidence.
    """
    side_dir = Path(side_dir)
    side_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in side_dir.glob("attempt-*"):
        try:
            existing.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    first_attempt = max(existing) + 1 if existing else 1
    attempts = []
    last = None
    for attempt in range(first_attempt, first_attempt + retries + 1):
        last = run_capture(executable, replay, stage, game_data,
                           side_dir / f"attempt-{attempt}", timeout)
        attempts.append({
            "attempt": attempt,
            "exitCode": last["exitCode"],
            "timedOut": last["timedOut"],
            "seconds": last["seconds"],
            "rowsBytes": last["rowsBytes"],
            "completeSha256": last["completeSha256"],
            "problems": last["problems"],
        })
        if not last["problems"]:
            break
    rows = last["rowsBytes"] // ROW_STRUCT.size
    complete = last.get("complete")
    summary = {
        "executable": last["executable"],
        "executableSha256": last["executableSha256"],
        "attempts": len(attempts),
        "attemptLog": attempts,
        "status": side_status(last),
        "exitCode": last["exitCode"],
        "timedOut": last["timedOut"],
        "seconds": last["seconds"],
        "totalSeconds": round(sum(item["seconds"] for item in attempts), 3),
        "rowsPath": last["rowsPath"],
        "rowsBytes": last["rowsBytes"],
        "rows": rows,
        "rowsSha256": last["rowsSha256"],
        "completePath": last["completePath"],
        "completeSha256": last["completeSha256"],
        "completeRows": complete.get("rows") if isinstance(complete, dict) else None,
        "completeEnded": complete.get("ended") if isinstance(complete, dict) else None,
        "completeErrors": complete.get("errors") if isinstance(complete, dict) else None,
        "runtimeScoreSha256": last.get("runtimeScoreSha256"),
        "runtimeConfigSha256": last.get("runtimeConfigSha256"),
        "problems": last["problems"],
        "directory": str(side_dir),
    }
    (side_dir / "side.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def inspect_build(executable: Path) -> dict:
    executable = Path(executable)
    problems = []
    symbol_present = False
    if not executable.is_file():
        problems.append("executable is missing")
    else:
        try:
            symbols = subprocess.run(["nm", "-C", str(executable)], capture_output=True,
                                     text=True, timeout=120, check=False)
            symbol_present = "th08::modern::ReplayTraceRecord(" in symbols.stdout
            if not symbol_present:
                problems.append("ReplayTraceRecord symbol missing (TH08_ENABLE_REPLAY_TRACE=OFF?)")
        except (OSError, subprocess.SubprocessError) as error:
            problems.append(f"cannot inspect symbols: {error}")
    return {
        "path": str(executable),
        "sha256": sha256_file(executable) if executable.is_file() else None,
        "bytes": executable.stat().st_size if executable.is_file() else 0,
        "recordSymbolPresent": symbol_present,
        "problems": problems,
    }


def describe_repository(executable: Path) -> dict:
    """Record the source identity of the tree that produced the builds."""
    repository = Path(executable).resolve().parents[2]
    try:
        head = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repository), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        head, dirty = None, None
    recorder = repository / "src/modern/linux/replay_trace.cpp"
    return {
        "worktree": str(repository),
        "head": head,
        "dirty": dirty,
        "recorderSource": str(recorder),
        "recorderSourceSha256": sha256_file(recorder) if recorder.is_file() else None,
    }


def load_replay_module(path: Path):
    spec = importlib.util.spec_from_file_location("v400_replay_retrace_replay_file", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load replay helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


# --------------------------------------------------------------------------
# Sharded corpus mode
# --------------------------------------------------------------------------

def load_manifest(path: Path) -> list:
    data = json.loads(Path(path).read_text())
    fixtures = data.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise CorpusError(f"manifest has no fixtures: {path}")
    return fixtures


def select_cases(fixtures_dir: Path, manifest_path: Path, only=None) -> list:
    """Verify the accepted corpus against its manifest and return case records.

    Fail-closed: every manifest fixture must be present, the accepted directory
    must contain nothing else, and every SHA-256 must match the manifest.
    """
    fixtures_dir = Path(fixtures_dir)
    if not fixtures_dir.is_dir():
        raise CorpusError(f"fixtures directory is missing: {fixtures_dir}")
    fixtures = load_manifest(manifest_path)
    present = {path.name: path for path in fixtures_dir.glob("*.rpy")}
    known = {fixture["file"] for fixture in fixtures}
    missing = sorted(known - set(present))
    extra = sorted(set(present) - known)
    if missing:
        raise CorpusError(f"manifest fixtures missing from corpus ({len(missing)}): {missing[:5]}")
    if extra:
        raise CorpusError(f"unmanifested .rpy files in corpus ({len(extra)}): {extra[:5]}")

    mismatched = []
    cases = []
    for order, fixture in enumerate(fixtures):
        path = present[fixture["file"]]
        digest = sha256_file(path)
        if digest != fixture["sha256"]:
            mismatched.append(fixture["file"])
        frames = fixture.get("retailTrace", {}).get("calculationFrames")
        cases.append({
            "case": path.stem,
            "fixtureId": fixture.get("id"),
            "file": path.name,
            "path": str(path),
            "sha256": digest,
            "manifestSha256": fixture["sha256"],
            "bytes": path.stat().st_size,
            "difficulty": fixture.get("difficultyName"),
            "shot": fixture.get("shotName"),
            "stages": [entry["index"] for entry in fixture.get("stages", [])],
            "inputRecords": sum(entry.get("inputRecords", 0) for entry in fixture.get("stages", [])),
            "manifestFrames": frames,
            "manifestOrder": order,
        })
    if mismatched:
        raise CorpusError(f"fixture SHA-256 differs from manifest ({len(mismatched)}): {mismatched[:5]}")

    if only:
        tokens = [only] if isinstance(only, str) else list(only)
        wanted = {token.strip() for token in tokens if token.strip()}
        selected = [case for case in cases
                    if case["case"] in wanted or case["fixtureId"] in wanted]
        unknown = wanted - {case["case"] for case in selected} - {case["fixtureId"] for case in selected}
        if unknown:
            raise CorpusError(f"--only names unknown cases: {sorted(unknown)}")
        cases = selected
    return cases


def balanced_order(cases: list) -> list:
    """Largest-predicted-load first, so round-robin shards stay balanced."""
    return sorted(cases, key=lambda case: (-(case.get("manifestFrames") or 0), case["case"]))


def assign_shards(cases: list, shards: int, shard: int) -> list:
    """Round-robin over the balanced global order; each shard gets its own slice."""
    if not 1 <= shard <= shards:
        raise ValueError(f"shard {shard} is outside 1..{shards}")
    ordered = balanced_order(cases)
    return [case for index, case in enumerate(ordered) if index % shards == shard - 1]


def journal_path(output: Path, shard: int) -> Path:
    return Path(output) / f"shard-{shard}.jsonl"


def append_jsonl(path: Path, record: dict) -> None:
    """Append one finished case and force it to disk (crash-safe resume)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def read_journal(path: Path):
    """Return (records-by-case, corrupt-line-count); later lines win."""
    path = Path(path)
    records = {}
    corrupt = 0
    if not path.is_file():
        return records, corrupt
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            corrupt += 1
            continue
        if isinstance(record, dict) and record.get("case"):
            records[record["case"]] = record
        else:
            corrupt += 1
    return records, corrupt


def case_status(on: dict, off: dict, rows: dict, completion: dict) -> str:
    if on["status"] == "timeout" or off["status"] == "timeout":
        return "timeout"
    if on["status"] != "clean" or off["status"] != "clean":
        return "failed"
    if rows.get("identical") and completion.get("identical"):
        return "identical"
    return "different"


def build_case_record(case: dict, info: dict, stage: int, on: dict, off: dict,
                      started: str, seconds: float, shard_info: dict) -> dict:
    rows = compare_rows_files(Path(on["rowsPath"]), Path(off["rowsPath"]))
    completion = compare_completion_files(Path(on["completePath"]), Path(off["completePath"]))
    status = case_status(on, off, rows, completion)
    expected_sides = [side for side in ("on", "off") if (on if side == "on" else off)["status"] == "clean"]
    manifest_frames = case.get("manifestFrames")
    frames_match = {}
    for name, side in (("on", on), ("off", off)):
        frames_match[name] = (None if manifest_frames is None
                              else side["rows"] == manifest_frames)
    record = {
        "case": case["case"],
        "fixtureId": case.get("fixtureId"),
        "file": case["file"],
        "status": status,
        "identical": status == "identical",
        "shard": shard_info.get("shard"),
        "shards": shard_info.get("shards"),
        "startedUtc": started,
        "finishedUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(seconds, 3),
        "replay": {
            "path": case["path"],
            "sha256": case["sha256"],
            "manifestSha256": case.get("manifestSha256"),
            "bytes": case.get("bytes"),
            "difficulty": info.get("difficultyName", case.get("difficulty")),
            "shot": info.get("shotName", case.get("shot")),
            "stages": [entry["index"] for entry in info["stages"]],
            "inputRecords": sum(entry["inputRecords"] for entry in info["stages"]),
            "startStage": stage,
            "manifestFrames": manifest_frames,
            "manifestOrder": case.get("manifestOrder"),
        },
        "on": {key: on[key] for key in (
            "status", "exitCode", "timedOut", "seconds", "attempts", "rows", "rowsBytes",
            "rowsSha256", "completeSha256", "problems")},
        "off": {key: off[key] for key in (
            "status", "exitCode", "timedOut", "seconds", "attempts", "rows", "rowsBytes",
            "rowsSha256", "completeSha256", "problems")},
        "comparison": {
            "rows": rows,
            "completeJson": completion,
            "identical": rows.get("identical") and completion.get("identical"),
            "firstDifference": rows.get("firstDifference") or completion.get("firstDifference"),
        },
        "rowsMatchCalculationFrames": frames_match,
        "expectedSides": expected_sides,
    }
    return record


def run_case(case: dict, stage_override, on_build: Path, off_build: Path, game_data: Path,
             output: Path, timeout: int, replay_module, retries: int,
             parallel_sides: bool, shard_info: dict) -> dict:
    replay = Path(case["path"])
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    clock = time.monotonic()
    info = replay_module.inspect(str(replay), require_retail=False)
    stage = stage_override if stage_override is not None else info["stages"][0]["index"]
    case_dir = Path(output) / case["case"]
    arguments = dict(replay=replay, stage=stage, game_data=game_data,
                     timeout=timeout, retries=retries)
    if parallel_sides:
        with ThreadPoolExecutor(max_workers=2) as pool:
            on_future = pool.submit(run_side, on_build, side_dir=case_dir / "on", **arguments)
            off_future = pool.submit(run_side, off_build, side_dir=case_dir / "off", **arguments)
            on, off = on_future.result(), off_future.result()
    else:
        on = run_side(on_build, side_dir=case_dir / "on", **arguments)
        off = run_side(off_build, side_dir=case_dir / "off", **arguments)
    record = build_case_record(case, info, stage, on, off, started,
                               time.monotonic() - clock, shard_info)
    write_json(case_dir / "case.json", record)
    return record


def summarize(records: dict, expected_total: int) -> dict:
    counts = {status: 0 for status in CASE_STATUSES}
    for record in records.values():
        status = record.get("status")
        if status in counts:
            counts[status] += 1
    compared = counts["identical"] + counts["different"]
    recorded = len(records)
    total = expected_total if expected_total is not None else recorded
    return {
        "total": total,
        "recorded": recorded,
        "compared": compared,
        "identical": counts["identical"],
        "different": counts["different"],
        "timeout": counts["timeout"],
        "failed": counts["failed"],
        "remaining": max(total - recorded, 0),
        "partial": recorded < total,
        "allIdentical": recorded == total and counts["identical"] == total and total > 0,
        "totalRowsPerSide": sum(int(record.get("on", {}).get("rows") or 0)
                                for record in records.values()),
    }


def collect_journals(output: Path):
    output = Path(output)
    journals = {}
    corrupt = {}
    for path in sorted(output.glob("shard-*.jsonl")):
        try:
            shard = int(path.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        records, bad = read_journal(path)
        journals[shard] = (path, records)
        corrupt[shard] = bad
    merged = {}
    for shard in sorted(journals):
        for name, record in journals[shard][1].items():
            record.setdefault("shard", shard)
            merged[name] = record
    return journals, merged, corrupt


def merge_report(output: Path, expected_total: int, builds=None, repository=None,
                 game_data: Path = None, run_config=None) -> dict:
    output = Path(output)
    journals, records, corrupt = collect_journals(output)
    summary = summarize(records, expected_total)
    shard_summaries = []
    for shard in sorted(journals):
        path, shard_records = journals[shard]
        statuses = {status: 0 for status in CASE_STATUSES}
        for record in shard_records.values():
            if record.get("status") in statuses:
                statuses[record["status"]] += 1
        shard_summaries.append({
            "shard": shard,
            "journal": str(path),
            "records": len(shard_records),
            "corruptLines": corrupt.get(shard, 0),
            "statuses": statuses,
            "manifestFrames": sum(int(record.get("replay", {}).get("manifestFrames") or 0)
                                  for record in shard_records.values()),
        })
    cases = sorted(records.values(), key=lambda record: (
        record.get("replay", {}).get("manifestOrder") is None,
        record.get("replay", {}).get("manifestOrder") or 0,
        record.get("case") or ""))
    report = {
        "version": 2,
        "kind": "wp1-differential-replay-retrace-sharded",
        "proofStrength": PROOF_STRENGTH,
        "total": summary["total"],
        "recorded": summary["recorded"],
        "compared": summary["compared"],
        "identical": summary["identical"],
        "different": summary["different"],
        "timeout": summary["timeout"],
        "failed": summary["failed"],
        "partial": summary["partial"],
        "remaining": summary["remaining"],
        "allIdentical": summary["allIdentical"],
        "totalRowsPerSide": summary["totalRowsPerSide"],
        "expectedTotal": expected_total,
        "generatedUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runConfig": run_config or {},
        "builds": builds or {},
        "repository": repository or {},
        "gameData": game_data_snapshot(game_data) if game_data else {},
        "shards": shard_summaries,
        "differences": [
            {"case": record["case"], "firstDifference": record.get("comparison", {}).get("firstDifference")}
            for record in cases if record.get("status") == "different"
        ],
        "notIdentical": [
            {"case": record["case"], "status": record.get("status"),
             "onProblems": record.get("on", {}).get("problems"),
             "offProblems": record.get("off", {}).get("problems")}
            for record in cases if record.get("status") in ("timeout", "failed")
        ],
        "cases": cases,
        "summary": summary,
    }
    write_json(output / "report.json", report)
    return report


def game_data_snapshot(game_data: Path) -> dict:
    game_data = Path(game_data)
    snapshot = {}
    for name in ("score.dat", "th08.cfg", "th08.dat"):
        path = game_data / name
        snapshot[name] = {
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    return snapshot


def parse_shard(value: str):
    try:
        left, right = value.split("/", 1)
        shard, shards = int(left), int(right)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("expected SHARD/SHARDS, e.g. 1/4")
    if shards < 1 or not 1 <= shard <= shards:
        raise argparse.ArgumentTypeError(f"{value} does not name a shard inside 1..{shards}")
    return shard, shards


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--on-build", type=Path, required=True,
                        help="harness build (TH08_RL_HARNESS=ON) executable")
    parser.add_argument("--off-build", type=Path, required=True,
                        help="portable build (no harness, trace recorder ON) executable")
    parser.add_argument("--replay", type=Path, action="append", default=None,
                        help="real .rpy recording; repeat for every case")
    parser.add_argument("--all", action="store_true",
                        help="run the whole accepted corpus from --manifest/--fixtures-dir")
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR,
                        help="accepted corpus directory (with --all)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="replay-fixtures.json used to verify the corpus (with --all)")
    parser.add_argument("--shard", type=parse_shard, metavar="SHARD/SHARDS",
                        help="run one round-robin shard, e.g. 2/4 (with --all)")
    parser.add_argument("--only", default=None,
                        help="comma-separated fixture ids or stems (smoke subsets)")
    parser.add_argument("--start-stage", type=int,
                        help="internal stage index to start from; default: first recorded stage")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help=f"per-run deadline in seconds (1..{MAXIMUM_TIMEOUT_SECONDS})")
    parser.add_argument("--retries", type=int, default=DEFAULT_MAXIMUM_RETRIES,
                        help="whole-process retries per side (0..5, default 1)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="cases processed concurrently inside this shard (1..8)")
    parser.add_argument("--parallel-sides", action="store_true",
                        help="run the ON and OFF process of one case concurrently")
    parser.add_argument("--resume", action="store_true",
                        help="keep an existing --output-dir and skip journaled cases")
    parser.add_argument("--reprocess-failed", action="store_true",
                        help="with --resume, retry cases previously recorded as timeout/failed")
    parser.add_argument("--merge", action="store_true",
                        help="only fold shard-*.jsonl into report.json, run nothing")
    parser.add_argument("--progress", action="store_true",
                        help="print journal progress and exit")
    parser.add_argument("--expected-total", type=int, default=DEFAULT_EXPECTED_TOTAL,
                        help="expected number of cases for the report (default 108)")
    parser.add_argument("--game-data-dir", type=Path, default=DEFAULT_GAME_DATA_DIR)
    parser.add_argument("--replay-helper", type=Path, default=DEFAULT_REPLAY_HELPER)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="evidence directory; must be new unless --resume/--merge/--progress")
    args = parser.parse_args(argv)

    if not 1 <= args.timeout <= MAXIMUM_TIMEOUT_SECONDS:
        parser.error(f"--timeout must be within 1..{MAXIMUM_TIMEOUT_SECONDS} seconds")
    if not 0 <= args.retries <= 5:
        parser.error("--retries must be within 0..5")
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be within 1..8")
    if args.all and args.replay:
        parser.error("--all and --replay are mutually exclusive")
    if not args.all and not args.replay and not (args.merge or args.progress):
        parser.error("one of --all, --replay, --merge or --progress is required")
    if args.shard and not args.all:
        parser.error("--shard requires --all")
    reusable = args.resume or args.merge or args.progress
    if args.output_dir.exists() and not reusable:
        parser.error(f"--output-dir must not exist: {args.output_dir}")
    for name in ("th08.dat", "thbgm.dat"):
        if not (args.game_data_dir / name).is_file():
            parser.error(f"--game-data-dir is missing {name}")
    if not args.replay_helper.is_file():
        parser.error(f"--replay-helper is missing: {args.replay_helper}")

    on_build = inspect_build(args.on_build)
    off_build = inspect_build(args.off_build)
    builds = {"on": on_build, "off": off_build}
    if not args.merge and not args.progress:
        if on_build["problems"] or off_build["problems"]:
            for label, build in (("ON", on_build), ("OFF", off_build)):
                for problem in build["problems"]:
                    print(f"{label} build: {problem}", file=sys.stderr)
            return 2
    repository = describe_repository(args.on_build)
    run_config = {
        "timeoutSeconds": args.timeout,
        "retries": args.retries,
        "jobs": args.jobs,
        "parallelSides": args.parallel_sides,
        "shard": list(args.shard) if args.shard else None,
        "expectedTotal": args.expected_total,
        "proofStrength": PROOF_STRENGTH,
        "mutedConfigSha256": sha256_bytes(muted_config()),
        "environmentTemplate": {
            "SDL_AUDIODRIVER": "dummy", "LIBGL_ALWAYS_SOFTWARE": "1",
            "GALLIUM_DRIVER": "llvmpipe", "LP_NUM_THREADS": "0", "LP_NO_RAST": "true",
            "TH08_AUTOPLAY_REPLAY": "./replay/audit.rpy",
            "TH08_RENDER_AUDIT_FAST": "1", "TH08_RENDER_AUDIT": "0",
            "TH08_REPLAY_TRACE_RECORD_ONLY": "1",
        },
    }

    if args.progress:
        journals, records, corrupt = collect_journals(args.output_dir)
        summary = summarize(records, args.expected_total)
        print(json.dumps({"journals": {str(path): len(items) for path, items in
                                       ((entry[0], entry[1]) for entry in journals.values())},
                          "corruptLines": corrupt, **summary}, indent=2))
        return 0

    if args.merge:
        report = merge_report(args.output_dir, args.expected_total, builds, repository,
                              args.game_data_dir, run_config)
        print(json.dumps({key: report[key] for key in (
            "total", "recorded", "compared", "identical", "different", "timeout",
            "failed", "partial", "remaining")}, indent=2))
        if report["partial"]:
            return 3
        return 0 if report["allIdentical"] else 1

    replay_module = load_replay_module(args.replay_helper)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard, shards = args.shard if args.shard else (1, 1)
    shard_info = {"shard": shard, "shards": shards}
    journal = journal_path(args.output_dir, shard)
    completed, corrupt = read_journal(journal)
    skip = set(completed)
    if args.reprocess_failed:
        skip = {name for name, record in completed.items()
                if record.get("status") not in ("timeout", "failed")}
    if completed:
        print(f"resume: {len(completed) - len(skip)} of {len(completed)} journaled cases skipped",
              flush=True)

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    corpus = {}
    if args.all:
        try:
            cases = select_cases(args.fixtures_dir, args.manifest,
                                 only=args.only.split(",") if args.only else None)
        except CorpusError as error:
            print(f"corpus: {error}", file=sys.stderr)
            return 2
        cases = assign_shards(cases, shards, shard)
        corpus = {
            "fixturesDir": str(args.fixtures_dir),
            "manifest": str(args.manifest),
            "manifestSha256": sha256_file(args.manifest),
            "casesInShard": len(cases),
            "manifestFramesInShard": sum(case.get("manifestFrames") or 0 for case in cases),
        }
        print(f"shard {shard}/{shards}: {len(cases)} cases, "
              f"{corpus['manifestFramesInShard']} calculation frames", flush=True)
    else:
        cases = [{"case": Path(replay).stem, "fixtureId": None, "file": Path(replay).name,
                  "path": str(replay), "sha256": sha256_file(replay),
                  "manifestFrames": None} for replay in args.replay]

    pending = [case for case in cases if case["case"] not in skip]
    print(f"pending {len(pending)} of {len(cases)} cases", flush=True)
    lock = threading.Lock()
    finished = []

    def handle(case):
        try:
            record = run_case(case, args.start_stage, args.on_build, args.off_build,
                              args.game_data_dir, args.output_dir, args.timeout,
                              replay_module, args.retries, args.parallel_sides, shard_info)
        except Exception as error:  # fail-closed: record and keep the shard alive
            record = {
                "case": case["case"], "fixtureId": case.get("fixtureId"),
                "file": case.get("file"), "status": "failed", "identical": False,
                "shard": shard, "shards": shards,
                "finishedUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "replay": {"path": case.get("path"), "sha256": case.get("sha256"),
                           "manifestFrames": case.get("manifestFrames"),
                           "manifestOrder": case.get("manifestOrder")},
                "on": {"status": "failed", "problems": [f"runner exception: {error}"]},
                "off": {"status": "failed", "problems": [f"runner exception: {error}"]},
                "comparison": {"firstDifference": {"kind": "exception", "message": str(error)}},
                "runnerError": repr(error),
            }
        with lock:
            append_jsonl(journal, record)
            finished.append(record)
            print(f"[{len(skip) + len(finished)}/{len(cases)}] {record['case']} "
                  f"{record['status']} rows on/off="
                  f"{record.get('on', {}).get('rows')}/{record.get('off', {}).get('rows')} "
                  f"seconds on/off={record.get('on', {}).get('seconds')}/"
                  f"{record.get('off', {}).get('seconds')}", flush=True)
        return record

    if args.jobs == 1:
        for case in pending:
            handle(case)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(handle, pending))

    journals, records, corrupt_lines = collect_journals(args.output_dir)
    report = merge_report(args.output_dir, args.expected_total, builds, repository,
                          args.game_data_dir, run_config)
    report["corpus"] = corpus
    report["startedUtc"] = started_utc
    report["journalCorruptLines"] = corrupt_lines
    write_json(args.output_dir / "report.json", report)
    print(json.dumps({key: report[key] for key in (
        "total", "recorded", "compared", "identical", "different", "timeout",
        "failed", "partial")}, indent=2), flush=True)
    return 0 if report["allIdentical"] else (3 if report["partial"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
