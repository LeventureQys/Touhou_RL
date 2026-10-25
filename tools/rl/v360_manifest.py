"""v3.6.0 evidence bookkeeping: data inventory + sha256 manifest.

Kept as a separate, dumb script (no analytics) so the audit trail is a pure
function of the artefacts on disk.  Anything that reads numbers lives in
``touhou_rl.ablate36``; this only hashes and indexes.

    python tools/rl/v360_manifest.py \
        --root Document/Update/Alpha/v3.6.0 \
        --ablation Document/Update/Alpha/v3.6.0/experiments/T1-ablation/ablation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

SCHEMA = "v360-evidence-manifest/1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_summary(root: Path) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=30, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": len([line for line in run("status", "--porcelain").splitlines() if line.strip()]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/rl/v360_manifest.py")
    parser.add_argument("--root", type=Path, required=True, help="v3.6.0 document root")
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--name", default="D2", help="inventory name, e.g. D1 or D2")
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    root = args.root
    report = json.loads(args.ablation.read_text(encoding="utf-8"))
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    inventory = {
        "schema": "v360-data-inventory/1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "name": args.name,
        "source_probe": report["source"],
        "protocol": report["protocol"],
        "survival_fallback": report.get("survival_fallback"),
        "corpus": report["inventory"],
    }
    (data_dir / f"{args.name}-inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    evidence_dir = root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    files = []
    excluded: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if evidence_dir in path.parents and path.name == "manifest.json":
            continue
        if "__pycache__" in path.parts:
            continue
        # ``np.savez`` stamps zip entries with the current time, so the cache hash
        # is not reproducible across runs and cannot verify anything.  It is a
        # derivative of the S0 traces, not evidence: hash the inputs
        # (ablation.json / inventory) instead, and list the cache here by name.
        if path.suffix == ".npz" and path.parent.name == "data":
            excluded.append(str(path.relative_to(root)).replace("\\", "/"))
            continue
        # Probe traces are large and gitignored by the same reasoning; the episode
        # index (episodes.jsonl) is kept so the corpus composition is still hashed.
        if path.parent.name in ("trace", "bullets") and "probe" in path.parts:
            excluded.append(str(path.relative_to(root)).replace("\\", "/"))
            continue
        files.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema": SCHEMA,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(root).replace("\\", "/"),
        "git": git_summary(args.repo),
        # Requirement doc §7 lists a checkpoint per experiment.  There is none on
        # purpose: every v3.6.0 experiment is a ridge regression (T1) whose
        # reproducible artefact is the feature cache + seeds, or a rule-based
        # controller (T2) whose artefact is the config + arm seed.
        "checkpoint": None,
        "files": files,
        "excluded_caches": excluded,
        "totals": {
            "files": len(files),
            "bytes": sum(item["bytes"] for item in files),
        },
    }
    target = evidence_dir / "manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(target), "files": len(files), "git": manifest["git"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
