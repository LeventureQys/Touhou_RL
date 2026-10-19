#!/usr/bin/env python
"""S3-8 negative self-tests of the evidence chain (temporary copies only).

The task book asks for four boundary checks that must not be satisfied by the real
artifacts:

1. a B manifest that is missing from a **copy** of the frozen list must stop the
   machine judgement (``exit 2``, blocked), not pass;
2. an empty 0-15 danger bucket (``frames = 0``) must never be accepted for A-02;
3. a wrong checkpoint sha256 must stop the startup path *before* the game is
   launched (``exit 2``);
4. an eval-report whose episodes lack ``termination_reason`` must not pass A-04.

Everything is written under ``.pytest-tmp/S3-8/boundary`` (scratch); no delivered
artifact, run directory, manifest or checkpoint is modified.  The script prints a
machine-readable summary and exits 0 only when every negative test behaved as
required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "touhou_rl" / "vision_data.py").exists():
            return candidate
    raise SystemExit("cannot locate the repository root")


ROOT = _repo_root(Path(__file__).resolve().parent)
EVIDENCE = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-8" / "evidence"
FROZEN = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "final-artifacts.json"
SCRATCH = ROOT / ".pytest-tmp" / "S3-8" / "boundary"
PY = sys.executable


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def dig(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_check(artifacts: Path, out: Path) -> dict:
    completed = subprocess.run(
        [
            PY,
            str(ROOT / "tools" / "rl" / "check_v104.py"),
            "--artifacts",
            str(artifacts),
            "--out",
            str(out),
            "--root",
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    result = {"exit_code": completed.returncode, "stdout": completed.stdout.splitlines(), "stderr": completed.stderr.splitlines()}
    if out.is_file():
        payload = json.loads(out.read_text(encoding="utf-8"))
        result["items"] = {key: value["status"] for key, value in payload.get("items", {}).items()}
        result["counts"] = payload.get("counts")
    return result


def main(argv: list[str] | None = None) -> int:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    checks = []

    # --- 1. missing B manifest in a copy of the frozen list -------------------
    case = SCRATCH / "missing-manifest-b"
    case.mkdir()
    mutated = json.loads(json.dumps(frozen))
    mutated["manifest_b"] = "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/does-not-exist.json"
    artifacts = write(case / "final-artifacts.json", mutated)
    result = run_check(artifacts, case / "check.json")
    checks.append(
        {
            "case": "missing_manifest_b",
            "expect": "exit 2 and A-01..A-04 blocked",
            "exit_code": result["exit_code"],
            "items": result.get("items"),
            "pass": bool(
                result["exit_code"] == 2
                and result.get("items")
                and all(result["items"].get(key) == "blocked" for key in ("A-01", "A-02", "A-03", "A-04"))
            ),
        }
    )

    # --- 2. empty 0-15 bucket must not pass A-02 ----------------------------
    case = SCRATCH / "empty-bucket"
    case.mkdir()
    index_src = EVIDENCE / "indices" / "policy_full_val.json"
    index = json.loads(index_src.read_text(encoding="utf-8"))
    for row in index["observations"].get("by_danger", []):
        if row.get("bucket") == "0-15":
            row["frames"] = 0
            row["top1"] = 0.0
            row["top3"] = 0.0
            row["regret_px"] = 0.0
    index["status"] = "pass"  # the index lies about itself on purpose
    index_path = write(case / "policy_full_val-empty-bucket.json", index)
    mutated = json.loads(json.dumps(frozen))
    mutated["evidence"]["policy_full_val"] = {"path": rel(index_path), "sha256": dig(index_path)}
    artifacts = write(case / "final-artifacts.json", mutated)
    result = run_check(artifacts, case / "check.json")
    checks.append(
        {
            "case": "empty_0_15_bucket",
            "expect": "A-02 never pass (fail or blocked)",
            "exit_code": result["exit_code"],
            "items": result.get("items"),
            "pass": bool(result.get("items", {}).get("A-02") in ("fail", "blocked") and result["exit_code"] != 0),
        }
    )

    # --- 3. wrong checkpoint sha must stop the startup path before the game --
    case = SCRATCH / "wrong-checkpoint-sha"
    case.mkdir()
    mutated = json.loads(json.dumps(frozen))
    mutated["policy_checkpoint_sha256"] = "0" * 64
    artifacts = write(case / "final-artifacts.json", mutated)
    completed = subprocess.run(
        [
            PY,
            str(EVIDENCE / "startup_smoke.py"),
            "--artifacts",
            rel(artifacts),
            "--out",
            str(case),
            "--device",
            "cuda",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    summary = json.loads((case / "startup-smoke.json").read_text(encoding="utf-8"))
    checks.append(
        {
            "case": "wrong_checkpoint_sha",
            "expect": "exit 2 and no game launch (real_game_started false, no pid recorded)",
            "exit_code": completed.returncode,
            "reason": summary.get("reason"),
            "checkpoint_matches_frozen": summary.get("checkpoint_matches_frozen"),
            "pass": bool(
                completed.returncode == 2
                and summary.get("checkpoint_matches_frozen") is False
                and not summary.get("exe")
            ),
        }
    )

    # --- 4. an eval-report without termination_reason must not pass A-04 ----
    case = SCRATCH / "missing-termination-reason"
    case.mkdir()
    report = {
        "checkpoint": frozen["policy_checkpoint"],
        "policy": "vision",
        "difficulty": 3,
        "team": 0,
        "practice_stage": None,
        "frameskip": 2,
        "deterministic": False,
        "device": "cuda",
        "started": "2026-10-20T00:00:00",
        "finished": "2026-10-20T00:10:00",
        "vision": {
            "checkpoint": frozen["policy_checkpoint"],
            "arch": "v3-st",
            "depth": 3,
            "max_steps": 12000,
        },
        "episodes": [
            {"episode": index, "steps": 900, "frames_reported": 1800, "stages_reached": 0, "deaths": 3,
             "terminated": True, "truncated": False}
            for index in (1, 2, 3)
        ],
        "summary": {"episodes": 3, "frames_mean": 1800.0},
    }
    report_path = write(case / "eval-report.json", report)
    synthetic_index = {
        "schema": "v104-evidence/1",
        "kind": "eval_report",
        "status": "pass",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "producer": "negative self-test (synthetic, scratch only)",
        "command": "negative self-test",
        "cwd": str(ROOT),
        "exit_code": 0,
        "artifacts": [{"path": rel(report_path), "sha256": dig(report_path)}],
        "observations": {
            "checkpoint_sha256": frozen["policy_checkpoint_sha256"],
            "episodes": 3,
            "frames_mean": 1800.0,
            "hit_step_cap": 0,
            "episode_fields_complete": False,
            "max_steps": 12000,
        },
        "reason": "synthetic eval-report whose episodes lack termination_reason",
    }
    index_path = write(case / "eval_report-index.json", synthetic_index)
    # A self-consistent synthetic startup artifact + index: both claim the gate facts,
    # so the only thing that can reject the run is the missing termination_reason.
    startup_src = EVIDENCE / "indices" / "startup.json"
    startup = json.loads(startup_src.read_text(encoding="utf-8"))
    startup_obs = {
        **startup["observations"],
        "real_game_started": True,
        "pixels_only": True,
        "no_crash": True,
        "no_exception_log": True,
        "ui_checked": True,
        "screenshots": startup["observations"]["screenshots"],
    }
    startup_evidence = {
        "schema": "v104-s3-8-startup/1",
        "kind": "startup",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "producer": "negative self-test (synthetic, scratch only)",
        "command": "negative self-test",
        "cwd": str(ROOT),
        "exit_code": 0,
        "observations": startup_obs,
        "reason": "synthetic startup evidence for the termination_reason boundary test",
    }
    startup_evidence_path = write(case / "startup-evidence.json", startup_evidence)
    startup["status"] = "pass"
    startup["observations"] = startup_obs
    startup["artifacts"] = [{"path": rel(startup_evidence_path), "sha256": dig(startup_evidence_path)}]
    startup_path = write(case / "startup-index.json", startup)
    mutated = json.loads(json.dumps(frozen))
    mutated["evidence"]["eval_report"] = {"path": rel(index_path), "sha256": dig(index_path)}
    mutated["evidence"]["startup"] = {"path": rel(startup_path), "sha256": dig(startup_path)}
    artifacts = write(case / "final-artifacts.json", mutated)
    result = run_check(artifacts, case / "check.json")
    checks.append(
        {
            "case": "missing_termination_reason",
            "expect": "A-04 fail (fields incomplete), never pass",
            "exit_code": result["exit_code"],
            "items": result.get("items"),
            "pass": bool(result.get("items", {}).get("A-04") in ("fail", "blocked")),
        }
    )

    summary = {
        "schema": "v104-s3-8-boundary-checks/1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/boundary_checks.py",
        "command": [sys.executable, *sys.argv],
        "cwd": str(ROOT),
        "scratch_root": rel(SCRATCH),
        "note": "all mutations live in .pytest-tmp/S3-8/boundary; delivered artifacts are read-only here",
        "checks": checks,
        "all_pass": all(node["pass"] for node in checks),
    }
    out = EVIDENCE / "boundary-checks.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": summary["all_pass"], "cases": [(c["case"], c["exit_code"], c["pass"]) for c in checks]}, ensure_ascii=False))
    print(f"written: {rel(out)}")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
