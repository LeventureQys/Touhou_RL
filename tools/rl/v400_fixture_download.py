"""Download every TH08 replay fixture in the manifest and verify bytes + SHA-256.

The `.rpy` files fetched here are replay **inputs** for WP1/G0.1 scale validation.
They are not, and this tool never treats them as, retail reference trajectories.

Exit codes: 0 all fixtures downloaded and verified; 1 any mismatch/failure;
2 manifest or output-directory problem. Existing directories are never deleted
or overwritten: if the requested output directory exists, a timestamped sibling
is created instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DEFAULT = ROOT / "third_party/th08/config/replay-fixtures.json"
OUT_DIR_DEFAULT = ROOT / ".tmp_v400_retail_fixtures_all"
REPORT_NAME = "manifest_check.json"
SCHEMA = "v400-fixture-corpus-check/1"

MAX_CONCURRENCY = 4
MAX_DEADLINE_SECONDS = 1500.0  # hard cap: 25 minutes
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_ATTEMPTS = 2  # one verification retry, as required
USER_AGENT = "touhou-rl-fixture-check/1.0 (+WP1/G0.1 input corpus)"

STATUS_VERIFIED = "verified"
STATUS_MISMATCH = "mismatch"
STATUS_FAILED = "failed"

# (status, body, error) for one HTTP attempt; status is None on transport error.
FetchResult = tuple[int | None, bytes | None, str | None]
Fetch = Callable[[str, float], FetchResult]


class ManifestError(ValueError):
    """The fixture manifest is unreadable or violates the expected schema."""


@dataclass(frozen=True)
class Fixture:
    id: str
    file: str
    url: str
    sha256: str
    bytes: int


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_hex_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(c in "0123456789abcdef" for c in value)


def load_manifest(path: Path) -> list[Fixture]:
    """Parse id/file/url/sha256/bytes for every fixture, rejecting unsafe entries."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest {path} is not valid JSON: {exc}") from exc
    entries = raw.get("fixtures") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f"manifest {path} has no non-empty 'fixtures' list")
    fixtures: list[Fixture] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"fixture #{index} is not an object")
        missing = [k for k in ("id", "file", "url", "sha256", "bytes") if k not in entry]
        if missing:
            raise ManifestError(f"fixture #{index} missing keys: {missing}")
        fixture_id, name, url = entry["id"], entry["file"], entry["url"]
        digest, size = entry["sha256"], entry["bytes"]
        if not isinstance(fixture_id, str) or not fixture_id:
            raise ManifestError(f"fixture #{index} has an empty id")
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise ManifestError(f"fixture {fixture_id} has an unsafe file name: {name!r}")
        if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
            raise ManifestError(f"fixture {fixture_id} has a non-HTTP url: {url!r}")
        if not _is_hex_digest(digest):
            raise ManifestError(f"fixture {fixture_id} has an invalid sha256: {digest!r}")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ManifestError(f"fixture {fixture_id} has an invalid byte size: {size!r}")
        if fixture_id in seen_ids:
            raise ManifestError(f"duplicate fixture id: {fixture_id}")
        if name in seen_names:
            raise ManifestError(f"duplicate fixture file name: {name}")
        seen_ids.add(fixture_id)
        seen_names.add(name)
        fixtures.append(Fixture(fixture_id, name, url, digest, size))
    return fixtures


def make_output_dir(base: Path, now: Callable[[], datetime] | None = None) -> Path:
    """Create a brand new output directory; never delete or reuse an existing one."""
    base = Path(base)
    clock = now or (lambda: datetime.now(timezone.utc))
    stamp = clock().strftime("%Y%m%d-%H%M%S")
    candidates = [base, base.with_name(f"{base.name}-{stamp}")]
    candidates += [base.with_name(f"{base.name}-{stamp}-{n:02d}") for n in range(1, 100)]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise ManifestError(f"cannot create a new output directory next to {base}")


def http_fetch(url: str, timeout: float) -> FetchResult:
    """Return (http_status, body, error) for one GET attempt."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read(), None
    except urllib.error.HTTPError as exc:
        return int(exc.code), None, f"HTTPError: {exc.code} {exc.reason}"
    except Exception as exc:  # noqa: BLE001 - transport errors must be recorded, not raised
        return None, None, f"{type(exc).__name__}: {exc}"


def _store(out_dir: Path, name: str, body: bytes, *, verified: bool) -> str:
    """Write the payload atomically; mismatched payloads keep a .mismatch suffix."""
    final_name = name if verified else f"{name}.mismatch"
    staging = out_dir / f".{final_name}.part"
    staging.write_bytes(body)
    staging.replace(out_dir / final_name)
    return final_name


def download_fixture(
    fixture: Fixture,
    out_dir: Path,
    *,
    fetch: Fetch = http_fetch,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    attempts: int = DEFAULT_ATTEMPTS,
    stop_at: float | None = None,
    now: Callable[[], float] = time.monotonic,
) -> dict:
    """Fetch one fixture, verifying byte size and SHA-256, retrying once on mismatch."""
    record = {
        "id": fixture.id,
        "file": fixture.file,
        "url": fixture.url,
        "sha256": fixture.sha256,
        "bytes": fixture.bytes,
        "status": STATUS_FAILED,
        "http": None,
        "attempts": 0,
        "actual_sha256": None,
        "actual_bytes": None,
        "stored": False,
        "path": None,
        "error": None,
    }
    last_body: bytes | None = None
    mismatch_seen = False
    error: str | None = None
    for attempt in range(1, max(1, attempts) + 1):
        if stop_at is not None:
            remaining = stop_at - now()
            if remaining <= 0:
                error = "deadline exceeded"
                break
            timeout = max(1.0, min(timeout, remaining))
        record["attempts"] = attempt
        status, body, fetch_error = fetch(fixture.url, timeout)
        record["http"] = status
        if fetch_error is not None or status != 200 or body is None:
            error = fetch_error or f"unexpected HTTP status {status}"
            continue
        actual_sha, actual_size = sha256_hex(body), len(body)
        record["actual_sha256"], record["actual_bytes"] = actual_sha, actual_size
        if actual_size == fixture.bytes and actual_sha == fixture.sha256:
            stored = _store(out_dir, fixture.file, body, verified=True)
            record.update(status=STATUS_VERIFIED, stored=True, path=stored, error=None)
            return record
        last_body = body
        mismatch_seen = True
        record["error"] = (
            f"mismatch: bytes {actual_size} != {fixture.bytes} or "
            f"sha256 {actual_sha} != {fixture.sha256}"
        )
    if mismatch_seen and last_body is not None:
        stored = _store(out_dir, fixture.file, last_body, verified=False)
        record.update(status=STATUS_MISMATCH, stored=False, path=stored)
    else:
        record["error"] = error or record["error"] or "download failed"
    return record


def _summarize(records: Sequence[dict]) -> dict:
    verified = sum(1 for r in records if r["status"] == STATUS_VERIFIED)
    mismatch = sum(1 for r in records if r["status"] == STATUS_MISMATCH)
    failed = sum(1 for r in records if r["status"] == STATUS_FAILED)
    downloaded = sum(1 for r in records if r["http"] == 200 and r["actual_bytes"] is not None)
    return {
        "total": len(records),
        "downloaded": downloaded,
        "verified": verified,
        "mismatch": mismatch,
        "failed": failed,
    }


def run(
    manifest_path: Path = MANIFEST_DEFAULT,
    out_dir_base: Path = OUT_DIR_DEFAULT,
    *,
    concurrency: int = MAX_CONCURRENCY,
    deadline_seconds: float = MAX_DEADLINE_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    attempts: int = DEFAULT_ATTEMPTS,
    fetch: Fetch | None = None,
    limit: int | None = None,
    log: Callable[[str], None] | None = None,
    now: Callable[[], float] = time.monotonic,
) -> tuple[Path, dict]:
    """Download and verify all fixtures; return (output_dir, report)."""
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be 1..{MAX_CONCURRENCY}, got {concurrency}")
    fetch = fetch or http_fetch
    deadline = min(float(deadline_seconds), MAX_DEADLINE_SECONDS)
    if deadline <= 0:
        raise ValueError("deadline_seconds must be positive")
    fixtures = load_manifest(manifest_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        fixtures = fixtures[:limit]
    out_dir = make_output_dir(out_dir_base)
    started = now()
    stop_at = started + deadline
    emit = log or (lambda message: None)
    lock = threading.Lock()

    def task(fixture: Fixture) -> dict:
        try:
            record = download_fixture(
                fixture, out_dir, fetch=fetch, timeout=timeout, attempts=attempts,
                stop_at=stop_at, now=now,
            )
        except Exception as exc:  # noqa: BLE001 - one broken item must not abort the corpus
            record = {
                "id": fixture.id, "file": fixture.file, "url": fixture.url,
                "sha256": fixture.sha256, "bytes": fixture.bytes,
                "status": STATUS_FAILED, "http": None, "attempts": None,
                "actual_sha256": None, "actual_bytes": None, "stored": False,
                "path": None, "error": f"internal_error: {type(exc).__name__}: {exc}",
            }
        with lock:
            emit(
                f"[{fixture.id}] {record['status']} http={record['http']} "
                f"bytes={record['actual_bytes']}/{fixture.bytes} attempts={record['attempts']}"
                + (f" error={record['error']}" if record["error"] else "")
            )
        return record

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(task, fixtures))

    elapsed = now() - started
    counts = _summarize(records)
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(Path(manifest_path)),
        "out_dir": str(out_dir),
        "role": "replay inputs for WP1/G0.1; not retail reference trajectories",
        "concurrency": concurrency,
        "deadline_seconds": deadline,
        "elapsed_seconds": round(elapsed, 3),
        **counts,
        "items": records,
    }
    if counts["verified"] + counts["mismatch"] + counts["failed"] != counts["total"]:
        raise AssertionError("inconsistent fixture status accounting")
    (out_dir / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out_dir, report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--deadline-seconds", type=float, default=MAX_DEADLINE_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        out_dir, report = run(
            args.manifest,
            args.out_dir,
            concurrency=args.concurrency,
            deadline_seconds=args.deadline_seconds,
            timeout=args.timeout_seconds,
            limit=args.limit,
            log=lambda message: print(message, flush=True),
        )
    except (ManifestError, ValueError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    print(
        f"total={report['total']} downloaded={report['downloaded']} "
        f"verified={report['verified']} mismatch={report['mismatch']} "
        f"failed={report['failed']}",
        flush=True,
    )
    print(f"report={out_dir / REPORT_NAME}", flush=True)
    return 0 if report["verified"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
