"""Synthetic self-check for the W00 ``audit`` implementation (tools/rl/v430.py).

**This is an implementation self-check, not game or capability evidence.**
Every case runs the real entry point as a subprocess against fixtures created in
a throw-away temporary directory; no engine, no game, no training and no
repository file is touched.  Passing this file says the audit decision logic
behaves as specified.  It says nothing about TH08 playing ability and it does
not pass G430-E0.

Cases: all checks satisfied (0), missing file (3), sha256 mismatch (3), plan
spec drift (3), sync-manifest coverage gap (3), oversized object without a
content hash (3), oversized object with ``hash_policy=stat_only`` (0), WSL
distro unavailable (3, rows must be ``unreachable``), invalid config schema (2),
``frozen_at`` without a timezone (2), existing ``--out`` refused (2, file
untouched) and a reserved subcommand (2, nothing written).

The report is regenerated on every run so it always describes the last executed
code; it is not an immutable experimental artefact.  ``--keep`` keeps the
fixtures and ``--report`` overrides the report path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPORT = HERE / "verify_audit.json"
SELFCHECK_SCHEMA = "v430-audit-selfcheck/1"
PLAN_SCHEMA = "v430-selfcheck/1"
PLAN_REVISION = "v430-selfcheck-r1"


def find_root(start: Path) -> Path:
    for candidate in start.parents:
        if (candidate / "AGENTS.md").is_file():
            return candidate
    return start.parents[6]


ROOT = find_root(HERE)
TOOL = ROOT / "tools" / "rl" / "v430.py"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_manifest(entries: list[tuple[str, str]]) -> bytes:
    return "".join(f"{digest}  {path}\n" for digest, path in entries).encode("utf-8")


def invoke(argv: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        [sys.executable, str(TOOL), *argv],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def status_of(payload: dict | None, ident: str) -> str | None:
    for check in (payload or {}).get("checks", []):
        if check["id"] == ident:
            return check["status"]
    return None


def stats(payload: dict | None) -> dict:
    return (payload or {}).get("statistics", {}) or {}


def met_of(payload: dict | None, ident: str) -> bool | None:
    for check in (payload or {}).get("checks", []):
        if check["id"] == ident:
            return check["requested_check_met"]
    return None


Case = dict[str, Any]


def base_config(case_dir: Path, assets: list[dict]) -> dict:
    return {
        "schema": "v430-assets/1",
        "experiment_id": "v430-selfcheck",
        "spec_revision": PLAN_REVISION,
        "frozen_at": "2026-11-15T00:00:00+08:00",
        "wsl": {"distro": "Ubuntu"},
        "plan_spec": {
            "path": str(case_dir / "PLAN_SPEC.json"),
            "schema_expected": PLAN_SCHEMA,
            "revision_expected": PLAN_REVISION,
        },
        "assets": assets,
    }


def with_manifest(config: dict, case_dir: Path) -> dict:
    payload = (case_dir / "ASSET_SYNC.sha256").read_bytes()
    config = dict(config)
    config["sync_manifest"] = {
        "path": str(case_dir / "ASSET_SYNC.sha256"),
        "sha256_expected": sha256_bytes(payload),
        "entry_count_expected": sum(1 for line in payload.decode().splitlines() if line.strip()),
    }
    return config


def absolute_file_row(ident: str, path: Path, digest: str | None, *, manifest: bool = False,
                      **extra: Any) -> dict:
    row: dict[str, Any] = {
        "id": ident,
        "kind": "file",
        "require": "hash" if digest else "presence",
        "path": str(path),
        "path_base": "absolute",
    }
    if digest:
        row["sha256_expected"] = digest
    if manifest:
        row["manifest"] = {"file": "ASSET_SYNC.sha256", "entry_path": str(path)}
    row.update(extra)
    return row


# --------------------------------------------------------------------------- #
# cases: builder(case_dir, out_path) -> Case
# --------------------------------------------------------------------------- #
def case_all_pass(case_dir: Path, out: Path) -> Case:
    payload_alpha = b"alpha-payload\n"
    payload_beta = b"beta-payload\n"
    alpha = case_dir / "alpha.bin"
    beta = case_dir / "beta.bin"
    alpha.write_bytes(payload_alpha)
    beta.write_bytes(payload_beta)
    nested = case_dir / "tree"
    nested.mkdir()
    (nested / "one.dat").write_bytes(b"1")
    (nested / "two.dat").write_bytes(b"2")
    write_json(case_dir / "PLAN_SPEC.json", {"schema": PLAN_SCHEMA, "revision": PLAN_REVISION})
    (case_dir / "ASSET_SYNC.sha256").write_bytes(
        build_manifest([(sha256_bytes(payload_alpha), str(alpha)),
                        (sha256_bytes(payload_beta), str(beta))])
    )
    assets = [
        absolute_file_row("alpha", alpha, sha256_bytes(payload_alpha), manifest=True),
        absolute_file_row("beta", beta, sha256_bytes(payload_beta), manifest=True),
        {
            "id": "tree",
            "kind": "dir",
            "require": "presence",
            "path": str(nested),
            "path_base": "absolute",
            "expect": {"contains": ["one.dat", "two.dat"]},
        },
        {
            "id": "python",
            "kind": "python",
            "require": "presence",
            "path": sys.executable,
            "path_base": "absolute",
            "expect": {"implementation": platform.python_implementation(),
                       "version": platform.python_version()},
        },
    ]
    return {
        "expected_exit": 0,
        "config": with_manifest(base_config(case_dir, assets), case_dir),
        "out": out,
        "checks": [
            ("exit 0", lambda p, r: r == 0),
            ("all rows satisfied", lambda p, r: stats(p).get("unmet") == 0),
            ("3 verified + 1 present_unhashed", lambda p, r: (
                stats(p).get("verified") == 6 and stats(p).get("present_unhashed") == 1)),
            ("manifest fully covered", lambda p, r: stats(p).get("manifest_entries_covered") == 2),
            ("directory row is present_unhashed but met", lambda p, r: (
                status_of(p, "tree") == "present_unhashed" and met_of(p, "tree") is True)),
            ("protocol_validity valid", lambda p, r: (p or {}).get("protocol_validity") == "valid"),
            ("result records exit_code 0", lambda p, r: (p or {}).get("exit_code") == 0),
        ],
    }


def case_missing_file(case_dir: Path, out: Path) -> Case:
    present = case_dir / "present.bin"
    present.write_bytes(b"here\n")
    absent = case_dir / "gone.bin"
    write_json(case_dir / "PLAN_SPEC.json", {"schema": PLAN_SCHEMA, "revision": PLAN_REVISION})
    assets = [
        absolute_file_row("present", present, sha256_bytes(present.read_bytes())),
        absolute_file_row("gone", absent, "0" * 64),
    ]
    return {
        "expected_exit": 3,
        "config": base_config(case_dir, assets),
        "out": out,
        "checks": [
            ("exit 3", lambda p, r: r == 3),
            ("absent row is missing", lambda p, r: status_of(p, "gone") == "missing"),
            ("present row still verified", lambda p, r: status_of(p, "present") == "present_verified"),
            ("statistics.missing == 1", lambda p, r: stats(p).get("missing") == 1),
            ("unmet names the row", lambda p, r: stats(p).get("unmet_ids") == ["gone"]),
            ("not downgraded to pass", lambda p, r: (p or {}).get("research_result", {}).get("passed") is False),
        ],
    }


def case_sha_mismatch(case_dir: Path, out: Path) -> Case:
    target = case_dir / "drifted.bin"
    target.write_bytes(b"content-after-the-pin\n")
    pinned = sha256_bytes(b"content-at-the-pin\n")
    write_json(case_dir / "PLAN_SPEC.json", {"schema": PLAN_SCHEMA, "revision": PLAN_REVISION})
    assets = [absolute_file_row("drifted", target, pinned)]
    return {
        "expected_exit": 3,
        "config": base_config(case_dir, assets),
        "out": out,
        "checks": [
            ("exit 3", lambda p, r: r == 3),
            ("row is present_mismatch", lambda p, r: status_of(p, "drifted") == "present_mismatch"),
            ("statistics.mismatch == 1", lambda p, r: stats(p).get("mismatch") == 1),
            ("both hashes are recorded", lambda p, r: all(
                check.get("sha256_expected") and check.get("sha256_actual")
                for check in (p or {}).get("checks", []) if check["id"] == "drifted")),
        ],
    }


def case_plan_spec_drift(case_dir: Path, out: Path) -> Case:
    write_json(case_dir / "PLAN_SPEC.json", {"schema": PLAN_SCHEMA, "revision": "v430-selfcheck-r9"})
    target = case_dir / "ok.bin"
    target.write_bytes(b"ok\n")
    assets = [absolute_file_row("ok", target, sha256_bytes(b"ok\n"))]
    return {
        "expected_exit": 3,
        "config": base_config(case_dir, assets),
        "out": out,
        "checks": [
            ("exit 3", lambda p, r: r == 3),
            ("plan_spec pin is present_mismatch", lambda p, r: status_of(p, "plan_spec") == "present_mismatch"),
            ("unmet names plan_spec", lambda p, r: stats(p).get("unmet_ids") == ["plan_spec"]),
        ],
    }


def case_manifest_coverage_gap(case_dir: Path, out: Path) -> Case:
    first = case_dir / "first.bin"
    second = case_dir / "second.bin"
    first.write_bytes(b"first\n")
    second.write_bytes(b"second\n")
    write_json(case_dir / "PLAN_SPEC.json", {"schema": PLAN_SCHEMA, "revision": PLAN_REVISION})
    (case_dir / "ASSET_SYNC.sha256").write_bytes(
        build_manifest([(sha256_bytes(b"first\n"), str(first)), (sha256_bytes(b"second\n"), str(second))])
    )
    assets = [absolute_file_row("first", first, sha256_bytes(b"first\n"), manifest=True)]
    return {
        "expected_exit": 3,
        "config": with_manifest(base_config(case_dir, assets), case_dir),
        "out": out,
        "checks": [
            ("exit 3", lambda p, r: r == 3),
            ("manifest itself verifies", lambda p, r: status_of(p, "sync_manifest") == "present_verified"),
            ("coverage row is present_mismatch", lambda p, r: (
                status_of(p, "sync_manifest:coverage") == "present_mismatch")),
            ("statistics exposes 1 of 2 covered", lambda p, r: (
                stats(p).get("manifest_entries") == 2 and stats(p).get("manifest_entries_covered") == 1)),
        ],
    }


def _oversize_case(case_dir: Path, out: Path, *, stat_only: bool) -> Case:
    target = case_dir / "big.bin"
    target.write_bytes(b"0123456789")
    write_json(case_dir / "PLAN_SPEC.json", {"schema": PLAN_SCHEMA, "revision": PLAN_REVISION})
    digest = sha256_bytes(target.read_bytes())
    row = absolute_file_row("big", target, digest)
    if stat_only:
        row["hash_policy"] = "stat_only"
    config = base_config(case_dir, [row])
    config["audit_policy"] = {"max_full_hash_bytes": 1}
    return {
        "expected_exit": 0 if stat_only else 3,
        "config": config,
        "out": out,
        "checks": [
            ("oversized row never counts as verified",
             lambda p, r: status_of(p, "big") == "present_unhashed"),
            ("row is not left unmeasured", lambda p, r: met_of(p, "big") is (True if stat_only else False)),
            ("exit matches the policy", lambda p, r: r == (0 if stat_only else 3)),
            ("reason explains the size limit", lambda p, r: any(
                "max_full_hash_bytes" in (check.get("reason") or "")
                for check in (p or {}).get("checks", []) if check["id"] == "big")),
        ],
    }


def case_oversize_blocked(case_dir: Path, out: Path) -> Case:
    return _oversize_case(case_dir, out, stat_only=False)


def case_oversize_stat_only(case_dir: Path, out: Path) -> Case:
    return _oversize_case(case_dir, out, stat_only=True)


def case_wsl_unreachable(case_dir: Path, out: Path) -> Case:
    write_json(case_dir / "PLAN_SPEC.json", {"schema": PLAN_SCHEMA, "revision": PLAN_REVISION})
    distro = "v430-selfcheck-absent-distro"
    assets = [
        {
            "id": "wsl-file",
            "kind": "wsl_file",
            "require": "hash",
            "wsl": {"distro": distro, "path": "/root/v430-selfcheck/engine"},
            "sha256_expected": "0" * 64,
        },
        {
            "id": "wsl-dir",
            "kind": "wsl_dir",
            "require": "presence",
            "wsl": {"distro": distro, "path": "/root/v430-selfcheck/data"},
            "expect": {"contains": ["th08.dat"]},
        },
        {
            "id": "wsl-python",
            "kind": "wsl_python",
            "require": "presence",
            "wsl": {"distro": distro, "path": "/root/v430-selfcheck/python"},
            "expect": {"version": "3.10.12"},
        },
    ]
    config = base_config(case_dir, assets)
    config["wsl"] = {"distro": distro}
    return {
        "expected_exit": 3,
        "config": config,
        "out": out,
        "checks": [
            ("exit 3", lambda p, r: r == 3),
            ("all three WSL rows are unreachable", lambda p, r: [
                status_of(p, name) for name in ("wsl-file", "wsl-dir", "wsl-python")
            ] == ["unreachable"] * 3),
            ("nothing becomes missing", lambda p, r: stats(p).get("missing") == 0),
            ("statistics.unreachable == 3", lambda p, r: stats(p).get("unreachable") == 3),
            ("wsl availability recorded as false", lambda p, r: (
                (p or {}).get("identities", {}).get("wsl", {}).get("available") is False)),
            ("no row is counted as satisfied", lambda p, r: all(
                met_of(p, name) is False for name in ("wsl-file", "wsl-dir", "wsl-python"))),
        ],
    }


def case_bad_schema(case_dir: Path, out: Path) -> Case:
    write_json(case_dir / "PLAN_SPEC.json", {"schema": PLAN_SCHEMA, "revision": PLAN_REVISION})
    config = base_config(case_dir, [])
    config["schema"] = "v430-assets/nope"
    config["assets"] = [{"id": "x", "kind": "file", "require": "presence", "path": str(case_dir / "x.bin")}]
    return {
        "expected_exit": 2,
        "config": config,
        "out": out,
        "checks": [
            ("exit 2", lambda p, r: r == 2),
            ("result says invalid_inputs", lambda p, r: (p or {}).get("execution_status") == "invalid_inputs"),
            ("protocol_validity invalid", lambda p, r: (p or {}).get("protocol_validity") == "invalid"),
            ("nothing measured", lambda p, r: stats(p).get("denominator") == 0),
            ("gap names the schema", lambda p, r: any(
                "v430-assets/1" in gap for gap in (p or {}).get("remaining_gaps", []))),
        ],
    }


def case_bad_frozen_at(case_dir: Path, out: Path) -> Case:
    write_json(case_dir / "PLAN_SPEC.json", {"schema": PLAN_SCHEMA, "revision": PLAN_REVISION})
    config = base_config(case_dir, [])
    config["frozen_at"] = "2026-11-15T00:00:00"
    config["assets"] = [{"id": "x", "kind": "file", "require": "presence", "path": str(case_dir / "x.bin")}]
    return {
        "expected_exit": 2,
        "config": config,
        "out": out,
        "checks": [
            ("exit 2", lambda p, r: r == 2),
            ("gap explains the missing timezone", lambda p, r: any(
                "timezone" in gap for gap in (p or {}).get("remaining_gaps", []))),
        ],
    }


def case_out_exists(case_dir: Path, out: Path) -> Case:
    target = case_dir / "kept.bin"
    target.write_bytes(b"kept\n")
    write_json(case_dir / "PLAN_SPEC.json", {"schema": PLAN_SCHEMA, "revision": PLAN_REVISION})
    sentinel = b'{"sentinel": "must not be overwritten"}\n'
    out.write_bytes(sentinel)
    return {
        "expected_exit": 2,
        "config": base_config(case_dir, [absolute_file_row("kept", target, sha256_bytes(b"kept\n"))]),
        "out": out,
        "keep_out": True,
        "checks": [
            ("exit 2", lambda p, r: r == 2),
            ("no result was parsed from the old file", lambda p, r: p is None),
            ("existing file is byte-identical", lambda p, r: out.read_bytes() == sentinel),
        ],
    }


def case_reserved_subcommand(case_dir: Path, out: Path) -> Case:
    return {
        "expected_exit": 2,
        "argv": ["train", "--config", str(case_dir / "absent.json"), "--out", str(out)],
        "out": out,
        "checks": [
            ("exit 2", lambda p, r: r == 2),
            ("notice says not_implemented", lambda p, r: "not_implemented" in CASE_OUTPUT.get("stdout", "")),
            ("no result file was written", lambda p, r: not out.exists()),
        ],
    }


CASE_OUTPUT: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
BUILDERS: list[tuple[str, Callable[[Path, Path], Case]]] = [
    ("all_pass", case_all_pass),
    ("missing_file", case_missing_file),
    ("sha_mismatch", case_sha_mismatch),
    ("plan_spec_drift", case_plan_spec_drift),
    ("manifest_coverage_gap", case_manifest_coverage_gap),
    ("oversize_blocked", case_oversize_blocked),
    ("oversize_stat_only", case_oversize_stat_only),
    ("wsl_unreachable", case_wsl_unreachable),
    ("bad_schema", case_bad_schema),
    ("bad_frozen_at", case_bad_frozen_at),
    ("out_exists", case_out_exists),
    ("reserved_subcommand", case_reserved_subcommand),
]


def run_case(name: str, builder: Callable[[Path, Path], Case], workspace: Path) -> dict:
    case_dir = workspace / name
    case_dir.mkdir(parents=True)
    out = case_dir / "result.json"
    case = builder(case_dir, out)
    out = case["out"]
    if case.get("argv") is not None:
        argv = case["argv"]
    else:
        config_path = case_dir / "assets.json"
        write_json(config_path, case["config"])
        argv = ["audit", "--config", str(config_path), "--out", str(out)]

    rc, stdout, stderr = invoke(argv)
    CASE_OUTPUT["stdout"] = stdout
    CASE_OUTPUT["stderr"] = stderr
    payload = None
    if out.exists() and not case.get("keep_out"):
        try:
            payload = read_json(out)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
    elif out.exists() and case.get("keep_out"):
        payload = None

    rows = []
    ok = True
    for label, predicate in case["checks"]:
        try:
            passed = bool(predicate(payload, rc))
        except Exception as exc:  # noqa: BLE001 - a raised predicate is a failed assertion
            passed = False
            label = f"{label} (predicate raised {type(exc).__name__}: {exc})"
        rows.append({"label": label, "ok": passed})
        ok = ok and passed
    ok = ok and rc == case["expected_exit"]
    return {
        "name": name,
        "expected_exit": case["expected_exit"],
        "actual_exit": rc,
        "ok": ok,
        "stdout_tail": stdout.strip().splitlines()[-2:],
        "stderr_tail": stderr.strip().splitlines()[-2:],
        "assertions": rows,
        "observed": {
            "statistics": stats(payload),
            "statuses": {check["id"]: check["status"] for check in (payload or {}).get("checks", [])},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", default=str(REPORT), help="where to write the self-check report")
    parser.add_argument("--keep", action="store_true", help="keep the fixture directory for inspection")
    args = parser.parse_args()

    if not TOOL.is_file():
        print(f"self-check: missing entry point {TOOL}", file=sys.stderr)
        return 1
    started = datetime.now().astimezone()
    workspace = Path(tempfile.mkdtemp(prefix="v430_audit_selfcheck_"))
    cases: list[dict] = []
    try:
        for name, builder in BUILDERS:
            case = run_case(name, builder, workspace)
            cases.append(case)
            print(f"[{'PASS' if case['ok'] else 'FAIL'}] {name}: expected {case['expected_exit']}, "
                  f"got {case['actual_exit']}")
            if not case["ok"]:
                for row in case["assertions"]:
                    if not row["ok"]:
                        print(f"        failed assertion: {row['label']}")
                if case["stderr_tail"]:
                    print(f"        stderr: {case['stderr_tail']}")
    finally:
        if args.keep:
            print(f"fixtures kept at {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)

    passed = sum(1 for case in cases if case["ok"])
    failed = len(cases) - passed
    finished = datetime.now().astimezone()
    report = {
        "schema": SELFCHECK_SCHEMA,
        "evidence_class": "implementation_self_check",
        "not_capability_evidence": True,
        "statement": (
            "This exercises the decision logic of tools/rl/v430.py audit on synthetic fixtures in a "
            "temporary directory. It is implementation self-checking only: it is not game, engine, "
            "asset or capability evidence, it does not validate any registered real asset, and it does "
            "not pass G430-E0. Regenerated on every run, so it always describes the last executed code."
        ),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "target": {
            "tool": "tools/rl/v430.py",
            "sha256": sha256_bytes(TOOL.read_bytes()),
            "python": sys.executable,
            "python_version": platform.python_version(),
        },
        "workspace": "temporary directory, removed after the run" if not args.keep else str(workspace),
        "summary": {"cases": len(cases), "passed": passed, "failed": failed},
        "cases": cases,
        "exit_code": 0 if failed == 0 else 1,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(f"self-check: {passed}/{len(cases)} cases passed | report {report_path}")
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
