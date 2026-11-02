"""v3.7.1 evidence manifest: SHA-256 of code, configs, data and the environment.

Why this file exists
--------------------
C-08 says another agent must be able to recompute every table in the v3.7.1
reports *without* the chat history.  That needs three things pinned:

1. which code produced the numbers (file hashes + git state, including the
   untracked files, because this worktree has uncommitted work from other
   agents that must not be overwritten);
2. which frozen config the run used;
3. which raw data file each table came from.

The v3.7.0 reports recorded a bare ``git HEAD`` while the tree had uncommitted
changes, which is not a reproducible claim.  This one hashes the working tree.

    python tools/rl/v371_manifest.py --out Document/Update/Alpha/v3.7.1/evidence/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

VERSION = "v3.7.1"
SCHEMA = "v371-evidence-manifest/1"

#: Files whose exact content defines the experiment.  Missing files are recorded
#: as absent rather than skipped, so a partial checkout is visible.
CODE = [
    "touhou_rl/escape/clock.py",
    "touhou_rl/escape/harness.py",
    "touhou_rl/escape/arms.py",
    "touhou_rl/escape/experiment.py",
    "touhou_rl/escape/fakegame.py",
    "touhou_rl/escape/adapter.py",
    "touhou_rl/escape/controller.py",
    "touhou_rl/escape/planner.py",
    "touhou_rl/escape/live.py",
    "touhou_rl/escape/cli.py",
    "touhou_rl/pilot.py",
    "touhou_rl/env.py",
    "touhou_rl/state.py",
    "touhou_rl/spec.py",
    "touhou_rl/tests/test_experiment_clock.py",
    "touhou_rl/tests/test_experiment_harness.py",
    "touhou_rl/tests/test_experiment_stage_a.py",
    "tools/rl/v371_manifest.py",
    "tools/rl/v371_recompute.py",
]

#: v3.7.0 scripts that lived in ``.tmp`` and are superseded by the formal tools.
#: They are archived verbatim (with hashes) because the v3.7.0 numbers came from
#: them; the reports must not depend on ``.tmp`` still existing.
LEGACY = [
    ".tmp/live_pairs.py",
    ".tmp/live_arm.py",
    ".tmp/live_timing.py",
    ".tmp/live_ablation.py",
    ".tmp/live_events.py",
    ".tmp/live_pairing.py",
    ".tmp/live_output.py",
    ".tmp/live_subject.py",
    ".tmp/live_summary.py",
    ".tmp/live_b01.py",
    ".tmp/live_extrapolation.py",
    ".tmp/live_student.py",
]

_SUPERSEDES = {
    ".tmp/live_pairs.py": "touhou_rl/escape/experiment.py:run_pair_session (frame-based window, AB/BA order)",
    ".tmp/live_arm.py": "touhou_rl/escape/experiment.py:run_live_episode",
    ".tmp/live_timing.py": "touhou_rl/escape/harness.py (per-decision read/plan/send timing)",
    ".tmp/live_ablation.py": "touhou_rl/escape/experiment.py + configs/p2-ablation-v1.json",
    ".tmp/live_events.py": "touhou_rl/escape/clock.py:EventLedger",
    ".tmp/live_pairing.py": "touhou_rl/escape/harness.py:ArmContext/ArmDecision",
    ".tmp/live_output.py": "touhou_rl/escape/experiment.py:write",
    ".tmp/live_subject.py": "touhou_rl/escape/arms.py:SubjectArm",
    ".tmp/live_summary.py": "tools/rl/v371_recompute.py",
    ".tmp/live_b01.py": "tools/rl/v371_recompute.py",
    ".tmp/live_extrapolation.py": "touhou_rl/escape/arms.py:ReplayArm + experiment.replay_check",
    ".tmp/live_student.py": "not migrated: student stays out of scope in v3.7.1 (README §9)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return out.stdout.strip()
    except Exception as error:  # noqa: BLE001 - a missing git must not fail the manifest
        return f"<git error: {error}>"


def build(root: Path, *, extra: list[str] | None = None) -> dict:
    files = {}
    for relative in CODE + LEGACY + (extra or []):
        path = root / relative
        files[relative] = (
            {"sha256": sha256(path), "bytes": path.stat().st_size} if path.is_file() else {"absent": True}
        )

    status = _git(root, "status", "--porcelain")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "root": str(root),
        "git": {
            "head": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len([line for line in status.splitlines() if line.strip()]),
            "dirty": bool(status.strip()),
            "note": (
                "the worktree is dirty by design: other agents' uncommitted work must not be "
                "overwritten, so file hashes -- not git HEAD -- identify the code"
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": _numpy_version(),
        },
        "files": files,
        "legacy_superseded_by": dict(_SUPERSEDES),
    }


def _numpy_version() -> str:
    try:
        import numpy

        return str(numpy.__version__)
    except Exception:  # noqa: BLE001
        return "<missing>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="write the v3.7.1 evidence manifest")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--extra", nargs="*", default=[], help="additional files to hash")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    document = build(root, extra=args.extra)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    present = sum(1 for entry in document["files"].values() if "sha256" in entry)
    print(f"wrote {out} ({present}/{len(document['files'])} files hashed)")
    print(f"  git head {document['git']['head'][:12]} dirty={document['git']['dirty']} "
          f"({document['git']['dirty_files']} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
