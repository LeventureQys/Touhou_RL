"""Offline tests for the TH08 replay-fixture downloader (no external network)."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools.rl import v400_fixture_download as fd

ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST = ROOT / "third_party/th08/config/replay-fixtures.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(path: Path, entries: list[dict], version: int = 1) -> Path:
    path.write_text(json.dumps({"version": version, "fixtures": entries}), encoding="utf-8")
    return path


def entry(fixture_id: str, name: str, body: bytes, url: str = "https://example.invalid/x.rpy") -> dict:
    return {"id": fixture_id, "file": name, "url": url, "sha256": sha(body), "bytes": len(body),
            "shot": 0, "difficulty": 0, "retailTrace": {"calculationFrames": 1, "sha256": "0" * 64}}


def fake_fetch(payloads: dict[str, list[tuple[int | None, bytes | None, str | None]]]):
    """Return a fetch that serves queued (status, body, error) triples per URL."""
    calls: list[str] = []
    lock = threading.Lock()

    def fetch(url: str, timeout: float):
        with lock:
            calls.append(url)
            queue = payloads[url]
            return queue.pop(0) if len(queue) > 1 else queue[0]

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


# ---------------------------------------------------------------- manifest


def test_real_manifest_lists_108_unique_fixtures():
    fixtures = fd.load_manifest(REAL_MANIFEST)
    assert len(fixtures) == 108
    assert len({f.id for f in fixtures}) == 108
    assert len({f.file for f in fixtures}) == 108
    for fixture in fixtures:
        assert fixture.file.endswith(".rpy")
        assert len(fixture.sha256) == 64 and fixture.sha256.islower()
        assert fixture.bytes > 0
        assert fixture.url.startswith("https://")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.pop("sha256"),
        lambda e: e.update(sha256="not-a-digest"),
        lambda e: e.update(file="../escape.rpy"),
        lambda e: e.update(file=""),
        lambda e: e.update(url="ftp://example.invalid/x.rpy"),
        lambda e: e.update(bytes=0),
        lambda e: e.update(bytes="30418"),
        lambda e: e.update(id=""),
    ],
)
def test_manifest_rejects_invalid_entries(tmp_path, mutate):
    bad = entry("a", "a.rpy", b"data")
    mutate(bad)
    manifest = write_manifest(tmp_path / "m.json", [bad])
    with pytest.raises(fd.ManifestError):
        fd.load_manifest(manifest)


def test_manifest_rejects_duplicates(tmp_path):
    body = b"data"
    manifest = write_manifest(tmp_path / "m.json", [entry("a", "a.rpy", body), entry("a", "b.rpy", body)])
    with pytest.raises(fd.ManifestError, match="duplicate fixture id"):
        fd.load_manifest(manifest)


def test_manifest_rejects_empty_list_and_broken_json(tmp_path):
    with pytest.raises(fd.ManifestError):
        fd.load_manifest(write_manifest(tmp_path / "empty.json", []))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(fd.ManifestError):
        fd.load_manifest(broken)
    with pytest.raises(fd.ManifestError):
        fd.load_manifest(tmp_path / "missing.json")


# ---------------------------------------------------------------- directories


def test_output_dir_is_created_new(tmp_path):
    base = tmp_path / "fixtures_all"
    out = fd.make_output_dir(base)
    assert out == base and out.is_dir()


def test_existing_output_dir_is_never_reused_or_deleted(tmp_path):
    base = tmp_path / "fixtures_all"
    base.mkdir()
    keep = base / "keep.bin"
    keep.write_bytes(b"keep")
    clock = lambda: datetime(2026, 11, 8, 12, 34, 56, tzinfo=timezone.utc)  # noqa: E731
    out = fd.make_output_dir(base, now=clock)
    assert out == tmp_path / "fixtures_all-20261108-123456"
    assert out.is_dir()
    assert keep.read_bytes() == b"keep"
    assert base.is_dir()


def test_output_dir_suffix_collision_falls_through(tmp_path):
    base = tmp_path / "fixtures_all"
    base.mkdir()
    (tmp_path / "fixtures_all-20261108-123456").mkdir()
    clock = lambda: datetime(2026, 11, 8, 12, 34, 56, tzinfo=timezone.utc)  # noqa: E731
    out = fd.make_output_dir(base, now=clock)
    assert out.name == "fixtures_all-20261108-123456-01"


# ---------------------------------------------------------------- one fixture


def test_download_verified(tmp_path):
    body = b"T8RP" + bytes(range(64))
    fixture = fd.Fixture("f1", "f1.rpy", "https://example.invalid/f1.rpy", sha(body), len(body))
    fetch = fake_fetch({fixture.url: [(200, body, None)]})
    record = fd.download_fixture(fixture, tmp_path, fetch=fetch)
    assert record["status"] == fd.STATUS_VERIFIED
    assert record["http"] == 200 and record["attempts"] == 1
    assert record["actual_sha256"] == sha(body) and record["actual_bytes"] == len(body)
    assert record["stored"] is True and record["path"] == "f1.rpy"
    assert (tmp_path / "f1.rpy").read_bytes() == body
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".part")] == []


def test_mismatch_is_retried_once_then_reported(tmp_path):
    expected = b"expected-bytes"
    corrupt = b"corrupted-bytes"
    fixture = fd.Fixture("f1", "f1.rpy", "https://example.invalid/f1.rpy", sha(expected), len(expected))
    fetch = fake_fetch({fixture.url: [(200, corrupt, None)]})
    record = fd.download_fixture(fixture, tmp_path, fetch=fetch)
    assert record["attempts"] == 2
    assert len(fetch.calls) == 2
    assert record["status"] == fd.STATUS_MISMATCH
    assert record["stored"] is False
    assert record["actual_sha256"] == sha(corrupt) and record["actual_bytes"] == len(corrupt)
    assert "mismatch" in record["error"]
    assert not (tmp_path / "f1.rpy").exists()
    assert (tmp_path / "f1.rpy.mismatch").read_bytes() == corrupt


def test_short_payload_is_reported_as_mismatch(tmp_path):
    expected = b"aaaaaaaa"
    fixture = fd.Fixture("f1", "f1.rpy", "https://example.invalid/f1.rpy", sha(expected), len(expected))
    fetch = fake_fetch({fixture.url: [(200, b"aaaaaaa", None)]})
    record = fd.download_fixture(fixture, tmp_path, fetch=fetch)
    assert record["status"] == fd.STATUS_MISMATCH
    assert record["actual_bytes"] == 7


def test_mismatch_then_success_verifies(tmp_path):
    body = b"payload"
    fixture = fd.Fixture("f1", "f1.rpy", "https://example.invalid/f1.rpy", sha(body), len(body))
    fetch = fake_fetch({fixture.url: [(200, b"corrupt", None), (200, body, None)]})
    record = fd.download_fixture(fixture, tmp_path, fetch=fetch)
    assert record["status"] == fd.STATUS_VERIFIED
    assert record["attempts"] == 2
    assert (tmp_path / "f1.rpy").read_bytes() == body


def test_http_error_records_status_without_faking_success(tmp_path):
    fixture = fd.Fixture("f1", "f1.rpy", "https://example.invalid/f1.rpy", "a" * 64, 10)
    fetch = fake_fetch({fixture.url: [(404, None, "HTTPError: 404 Not Found")]})
    record = fd.download_fixture(fixture, tmp_path, fetch=fetch)
    assert record["status"] == fd.STATUS_FAILED
    assert record["http"] == 404 and record["attempts"] == 2
    assert "404" in record["error"]
    assert record["stored"] is False
    assert list(tmp_path.iterdir()) == []


def test_transport_error_records_exception(tmp_path):
    fixture = fd.Fixture("f1", "f1.rpy", "https://example.invalid/f1.rpy", "a" * 64, 10)
    fetch = fake_fetch({fixture.url: [(None, None, "URLError: name resolution failed")]})
    record = fd.download_fixture(fixture, tmp_path, fetch=fetch)
    assert record["status"] == fd.STATUS_FAILED
    assert record["http"] is None
    assert "URLError" in record["error"]


# ---------------------------------------------------------------- whole run


def test_run_report_contract_and_counts(tmp_path):
    manifest = write_manifest(
        tmp_path / "m.json",
        [
            entry("ok-1", "ok1.rpy", b"one", "https://example.invalid/one.rpy"),
            entry("ok-2", "ok2.rpy", b"two", "https://example.invalid/two.rpy"),
            entry("bad-1", "bad1.rpy", b"three", "https://example.invalid/three.rpy"),
        ],
    )
    fetch = fake_fetch({
        "https://example.invalid/one.rpy": [(200, b"one", None)],
        "https://example.invalid/two.rpy": [(200, b"two", None)],
        "https://example.invalid/three.rpy": [(200, b"WRONG", None)],
    })
    lines: list[str] = []
    out_dir, report = fd.run(manifest, tmp_path / "out", fetch=fetch, concurrency=4, log=lines.append)
    report_path = out_dir / fd.REPORT_NAME
    assert report_path.is_file()
    on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    assert on_disk == report
    for key in ("total", "downloaded", "verified", "mismatch", "failed"):
        assert isinstance(on_disk[key], int)
        assert key in on_disk
    assert (on_disk["total"], on_disk["downloaded"], on_disk["verified"],
            on_disk["mismatch"], on_disk["failed"]) == (3, 3, 2, 1, 0)
    assert on_disk["concurrency"] == 4
    assert len(on_disk["items"]) == 3
    for item in on_disk["items"]:
        for key in ("id", "file", "sha256", "bytes", "status", "http"):
            assert key in item
    assert [item["id"] for item in on_disk["items"]] == ["ok-1", "ok-2", "bad-1"]
    assert (out_dir / "ok1.rpy").is_file() and (out_dir / "bad1.rpy.mismatch").is_file()
    assert len(lines) == 3


def test_run_records_deadline_failures_without_downloading(tmp_path):
    manifest = write_manifest(
        tmp_path / "m.json",
        [entry("a", "a.rpy", b"a", "https://example.invalid/a.rpy"),
         entry("b", "b.rpy", b"b", "https://example.invalid/b.rpy")],
    )
    fetch = fake_fetch({"https://example.invalid/a.rpy": [(200, b"a", None)],
                        "https://example.invalid/b.rpy": [(200, b"b", None)]})
    ticks = iter([0.0, 100.0, 200.0, 300.0, 400.0, 500.0])

    def clock() -> float:
        return next(ticks)

    out_dir, report = fd.run(manifest, tmp_path / "out", fetch=fetch,
                             deadline_seconds=10.0, now=clock)
    assert report["verified"] == 0
    assert report["failed"] == 2
    assert fetch.calls == []
    assert all(item["error"] == "deadline exceeded" for item in report["items"])
    assert (out_dir / fd.REPORT_NAME).is_file()


def test_run_caps_deadline_at_25_minutes(tmp_path):
    manifest = write_manifest(tmp_path / "m.json", [entry("a", "a.rpy", b"a", "https://example.invalid/a.rpy")])
    fetch = fake_fetch({"https://example.invalid/a.rpy": [(200, b"a", None)]})
    _, report = fd.run(manifest, tmp_path / "out", fetch=fetch, deadline_seconds=99999)
    assert report["deadline_seconds"] == 1500.0


@pytest.mark.parametrize("concurrency", [0, -1, 5, 8])
def test_run_rejects_concurrency_above_four(tmp_path, concurrency):
    manifest = write_manifest(tmp_path / "m.json", [entry("a", "a.rpy", b"a", "https://example.invalid/a.rpy")])
    with pytest.raises(ValueError, match="concurrency"):
        fd.run(manifest, tmp_path / "out", fetch=fake_fetch({}), concurrency=concurrency)


def test_run_limit_slices_corpus(tmp_path):
    manifest = write_manifest(
        tmp_path / "m.json",
        [entry("a", "a.rpy", b"a", "https://example.invalid/a.rpy"),
         entry("b", "b.rpy", b"b", "https://example.invalid/b.rpy")],
    )
    fetch = fake_fetch({"https://example.invalid/a.rpy": [(200, b"a", None)],
                        "https://example.invalid/b.rpy": [(200, b"b", None)]})
    _, report = fd.run(manifest, tmp_path / "out", fetch=fetch, limit=1)
    assert report["total"] == 1 and report["items"][0]["id"] == "a"


def test_main_exit_codes(tmp_path, monkeypatch):
    body = b"body"
    manifest = write_manifest(tmp_path / "m.json", [entry("a", "a.rpy", body, "https://example.invalid/a.rpy")])
    good = fake_fetch({"https://example.invalid/a.rpy": [(200, body, None)]})
    monkeypatch.setattr(fd, "http_fetch", good)
    code = fd.main(["--manifest", str(manifest), "--out-dir", str(tmp_path / "out"), "--concurrency", "2"])
    assert code == 0

    bad_manifest = write_manifest(tmp_path / "bad.json", [entry("a", "a.rpy", body, "https://example.invalid/a.rpy")])
    corrupt = fake_fetch({"https://example.invalid/a.rpy": [(200, b"other", None)]})
    monkeypatch.setattr(fd, "http_fetch", corrupt)
    code = fd.main(["--manifest", str(bad_manifest), "--out-dir", str(tmp_path / "out2")])
    assert code == 1

    code = fd.main(["--manifest", str(tmp_path / "absent.json"), "--out-dir", str(tmp_path / "out3")])
    assert code == 2
    assert not (tmp_path / "out3").exists()


# ---------------------------------------------------------------- real HTTP


@pytest.fixture()
def local_server(tmp_path):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):  # keep pytest output clean
            return None

    serve = tmp_path / "www"
    serve.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(serve)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield serve, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_real_urllib_download_over_local_http(tmp_path, local_server):
    serve, base_url = local_server
    body = b"T8RP-local-fixture-payload"
    (serve / "ok.rpy").write_bytes(body)
    manifest = write_manifest(
        tmp_path / "m.json",
        [
            entry("ok", "ok.rpy", body, f"{base_url}/ok.rpy"),
            entry("missing", "missing.rpy", b"gone", f"{base_url}/missing.rpy"),
        ],
    )
    out_dir, report = fd.run(manifest, tmp_path / "out", concurrency=2)
    assert report["total"] == 2
    assert report["downloaded"] == 1 and report["verified"] == 1
    assert report["failed"] == 1 and report["mismatch"] == 0
    assert (out_dir / "ok.rpy").read_bytes() == body
    by_id = {item["id"]: item for item in report["items"]}
    assert by_id["ok"]["http"] == 200 and by_id["ok"]["status"] == "verified"
    assert by_id["missing"]["status"] == "failed" and by_id["missing"]["http"] == 404
    assert not (out_dir / "missing.rpy").exists()
