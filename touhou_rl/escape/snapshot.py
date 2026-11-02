"""Code/worktree snapshot for D0: hashes, not a bare ``git HEAD``.

Why not just record the commit
------------------------------
The tree that D0 audited had uncommitted modifications to ``pilot.py``,
``env.py`` and ``evaluate.py`` (see ``git status`` in the manifest).  A commit
hash alone would name a baseline that is *not* the code that produced the
fixtures.  The snapshot therefore hashes the audited files themselves and keeps
the dirty-file list next to the digest.

The snapshot is evidence, not a test input: it contains a timestamp, so it is
never compared byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA = "v370-code-snapshot/1"

#: Files whose behaviour the baseline arm depends on.  A change to any of them
#: requires a new snapshot and a new golden fixture.
BASELINE_FILES: tuple[str, ...] = (
    "touhou_rl/pilot.py",
    "touhou_rl/env.py",
    "touhou_rl/obs.py",
    "touhou_rl/spec.py",
    "touhou_rl/state.py",
    "touhou_rl/trace35.py",
    "touhou_rl/tactics.py",
    "touhou_rl/session.py",
    "touhou_rl/winapi.py",
    "touhou_rl/menu.py",
    "touhou_rl/reward.py",
    "touhou_rl/input_evidence.py",
    "touhou_rl/probe35.py",
)

#: The new package is recorded too, so the contract that was frozen is auditable
#: even before it is committed.
CONTRACT_FILES: tuple[str, ...] = (
    "touhou_rl/escape/__init__.py",
    "touhou_rl/escape/contracts.py",
    "touhou_rl/escape/geometry.py",
    "touhou_rl/escape/adapter.py",
    "touhou_rl/escape/shadow.py",
    "touhou_rl/escape/synth.py",
    "touhou_rl/escape/families.py",
    "touhou_rl/escape/reference.py",
    "touhou_rl/escape/forecast.py",
    "touhou_rl/escape/planner.py",
    "touhou_rl/escape/exit.py",
    "touhou_rl/escape/controller.py",
    "touhou_rl/escape/trace.py",
    "touhou_rl/escape/render.py",
    "touhou_rl/escape/evaluate.py",
    "touhou_rl/escape/probe.py",
    "touhou_rl/escape/paired.py",
    "touhou_rl/escape/replay_source.py",
    "touhou_rl/escape/vision.py",
    "touhou_rl/escape/student.py",
    "touhou_rl/escape/baseline.py",
    "touhou_rl/escape/audit.py",
    "touhou_rl/escape/seeds.py",
    "touhou_rl/escape/snapshot.py",
    "touhou_rl/escape/cli.py",
    "touhou_rl/escape/__main__.py",
    "touhou_rl/tests/test_escape_contracts.py",
    "touhou_rl/tests/test_escape_geometry.py",
    "touhou_rl/tests/test_escape_adapter.py",
    "touhou_rl/tests/test_escape_shadow.py",
    "touhou_rl/tests/test_escape_synth.py",
    "touhou_rl/tests/test_escape_families.py",
    "touhou_rl/tests/test_escape_reference.py",
    "touhou_rl/tests/test_escape_forecast.py",
    "touhou_rl/tests/test_escape_planner.py",
    "touhou_rl/tests/test_escape_exit.py",
    "touhou_rl/tests/test_escape_controller.py",
    "touhou_rl/tests/test_escape_trace.py",
    "touhou_rl/tests/test_escape_evaluate.py",
    "touhou_rl/tests/test_escape_probe.py",
    "touhou_rl/tests/test_escape_paired.py",
    "touhou_rl/tests/test_escape_replay_source.py",
    "touhou_rl/tests/test_escape_vision.py",
    "touhou_rl/tests/test_escape_baseline.py",
    "touhou_rl/tests/test_escape_cli.py",
    "touhou_rl/tests/test_escape_seeds_audit.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env specific
        return 127, f"git unavailable: {exc}"
    return completed.returncode, completed.stdout.strip()


def git_state(root: Path) -> dict:
    code, head = _git(root, "rev-parse", "HEAD")
    if code != 0:
        return {"available": False, "reason": head or "git rev-parse failed"}
    _code, branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    status_code, status = _git(root, "status", "--porcelain=v1")
    lines = [line for line in status.splitlines() if line.strip()]
    modified = [line[3:] for line in lines if line[:2].strip() and not line.startswith("??")]
    untracked = [line[3:] for line in lines if line.startswith("??")]
    return {
        "available": status_code == 0,
        "head": head,
        "branch": branch,
        "dirty": bool(lines),
        "change_count": len(lines),
        "modified": modified,
        "untracked_count": len(untracked),
        "untracked": untracked,
        "note": (
            "HEAD alone is not the audited baseline while 'dirty' is true; the file "
            "hashes below are the snapshot that matters."
        ),
    }


def file_records(root: Path, paths: tuple[str, ...]) -> list[dict]:
    records = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            records.append({"path": relative, "missing": True})
            continue
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        )
    return records


def _surface_digest(records: list[dict]) -> str:
    payload = [[record["path"], record.get("sha256")] for record in records]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def code_snapshot(root: Path | str) -> dict:
    root = Path(root)
    baseline = file_records(root, BASELINE_FILES)
    contract = file_records(root, CONTRACT_FILES)
    return {
        "schema": SCHEMA,
        "captured": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(root),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "git": git_state(root),
        "baseline_files": baseline,
        "contract_files": contract,
        "baseline_surface_digest": _surface_digest(baseline),
        "contract_surface_digest": _surface_digest(contract),
        "note": (
            "baseline_surface_digest covers the old controller and its readers; a different "
            "digest means the comparison arm changed and the golden fixture must be re-pinned."
        ),
    }


def write_snapshot(path: Path, root: Path | str) -> dict:
    snapshot = code_snapshot(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snapshot


def evidence_manifest(base: Path | str, include: tuple[str, ...]) -> dict:
    """Hash a fixed list of artifacts relative to ``base`` (for ``evidence/``)."""
    base = Path(base)
    entries = []
    for relative in include:
        path = base / relative
        if not path.is_file():
            entries.append({"path": relative, "missing": True})
            continue
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema": "v370-evidence/1",
        "captured": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base": str(base),
        "entries": entries,
        "digest": _surface_digest(entries),
    }
