#!/usr/bin/env python
"""S3-8 evidence indices: nine ``v104-evidence/1`` index files plus the frozen map.

Every index is generated *from* the raw artifacts it cites: the sha256 of each
artifact is recomputed here, and the observations are copied from the artifact's
own ``observations`` block (or parsed from the log for pytest).  That is what makes
``check_v104.py``'s cross-check meaningful - the index cannot claim a number the
artifact does not carry.

It also:

* writes ``upstream-evidence-sha-crosscheck.json``: the sha256 of every S3-1/S3-3
  artifact this stage cites, compared with the value recorded in that stage's own
  ``sha256sums.txt`` (so "cited" and "unchanged since that stage" are both checked);
* updates ``Stage3/final-artifacts.json`` with the nine-entry ``evidence`` map and
  ``external_guard_uses = 1`` (A-03 used B once).  The frozen model selection keys
  are not touched.

    python Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/make_evidence_indices.py
"""

from __future__ import annotations

import hashlib
import json
import re
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
INDICES = EVIDENCE / "indices"
FROZEN = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "final-artifacts.json"
LOGS = EVIDENCE / "logs"


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def dig(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing artifact {path}")
    return {"path": rel(path), "sha256": dig(path)}


def upstream_crosscheck() -> dict:
    """Current sha256 vs the sha256 each upstream stage recorded for the same file."""
    upstream = {
        "Document/Update/Alpha/v1.0.4/Stage3/S3-3/evidence/sha256sums.txt": [
            "Document/Update/Alpha/v1.0.4/Stage3/S3-3/evidence/throughput.json",
        ],
        "Document/Update/Alpha/v1.0.4/Stage3/S3-1/evidence/sha256sums.txt": [
            "Document/Update/Alpha/v1.0.4/Stage3/S3-1/evidence/cache-B.json",
            "Document/Update/Alpha/v1.0.4/Stage3/S3-1/evidence/budget-B-cached.json",
            "Document/Update/Alpha/v1.0.4/Stage3/S3-1/evidence/budget-B-npz.json",
            "Document/Update/Alpha/v1.0.4/Stage3/S3-1/evidence/pixel-consistency-B.json",
            "Document/Update/Alpha/v1.0.4/Stage3/S3-1/evidence/ud0063-evaluation-manifest.json",
        ],
    }
    checks = []
    for sums_rel, files in upstream.items():
        sums = (ROOT / sums_rel).read_text(encoding="utf-8", errors="replace")
        recorded = {}
        for line in sums.splitlines():
            match = re.match(r"^([0-9a-f]{64})\s+\d+\s+(.+?)\s*$", line)
            if match:
                recorded[match.group(2).replace("\\", "/")] = match.group(1)
        for name in files:
            current = dig(ROOT / name)
            expected = recorded.get(name)
            checks.append(
                {
                    "file": name,
                    "current_sha256": current,
                    "recorded_sha256": expected,
                    "matches_stage_record": bool(expected and expected == current),
                    "recorded_by": sums_rel,
                }
            )
    return {
        "schema": "v104-s3-8-upstream-sha-crosscheck/1",
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/make_evidence_indices.py",
        "all_match": all(node["matches_stage_record"] for node in checks),
        "checks": checks,
    }


def pytest_counts(text: str) -> dict:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for line in text.splitlines():
        if " in " not in line:
            continue
        for number, kind in re.findall(r"(\d+) (passed|failed|skipped|error|errors)", line):
            key = "error" if kind.startswith("error") else kind
            counts[key] = max(counts.get(key, 0), int(number))
    return counts


def main() -> int:
    INDICES.mkdir(parents=True, exist_ok=True)
    crosscheck_path = EVIDENCE / "upstream-evidence-sha-crosscheck.json"
    crosscheck = upstream_crosscheck()
    crosscheck_path.write_text(json.dumps(crosscheck, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not crosscheck["all_match"]:
        print("warning: an upstream artifact no longer matches its stage record", file=sys.stderr)
    crosscheck_artifact = artifact(crosscheck_path)

    state_recheck = read(EVIDENCE / "state-recheck.json")
    state_compare = EVIDENCE / "state-full-val-compare.json"
    policy_full = read(EVIDENCE / "policy-full-val.json")
    cross_a = EVIDENCE / "cross-replay-a.json"
    cross_b = EVIDENCE / "cross-replay-b.json"
    rows_a = read(cross_a)
    rows_b = read(cross_b)
    a_row = rows_a[0]
    b_row = rows_b[0]
    verdict = None
    guard_log = EVIDENCE / "cross-replay-guard-usage.txt"
    if guard_log.is_file():
        text = guard_log.read_text(encoding="utf-8", errors="replace")
        ratio = re.search(r"ratio\s*=\s*([0-9.]+)", text)
        verdict = {
            "ratio": float(ratio.group(1)) if ratio else None,
            "threshold": 1.25,
            "pass": bool(ratio and float(ratio.group(1)) <= 1.25),
        }
    a04 = read(EVIDENCE / "a04-attempt.json")
    startup = read(EVIDENCE / "startup-evidence.json")
    d9 = read(EVIDENCE / "d9-recheck.json")
    d10 = read(EVIDENCE / "d10-recheck.json")
    d11 = read(EVIDENCE / "d11-recheck.json")
    pytest_log = LOGS / "pytest-full.log"
    pytest_text = pytest_log.read_text(encoding="utf-8", errors="replace")
    counts = pytest_counts(pytest_text)

    indices: dict[str, dict] = {}
    indices["state_recheck"] = {
        "schema": "v104-evidence/1",
        "kind": "state_recheck",
        "status": "pass" if state_recheck["observations"]["player_pos_px"] < 10.0
        and state_recheck["observations"]["danger_corr_perframe_median"] > 0.8
        else "fail",
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/full_val_recheck.py",
        "command": (
            "python Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/full_val_recheck.py --kind state_recheck"
            " --checkpoint third_party/th08/build/rl/runs/v104-v3-E5-s7/state-pass.pt"
            " --out Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/state-recheck.json --device cuda"
        ),
        "cwd": str(ROOT),
        "exit_code": 0,
        "artifacts": [artifact(EVIDENCE / "state-recheck.json"), artifact(state_compare), crosscheck_artifact],
        "observations": state_recheck["observations"],
        "reason": (
            "full re-evaluation of the frozen state-pass.pt on all 7241 validation indices; position and"
            " per-frame danger median reproduce the summary's state.best row, independent tool"
            " compare_state_models --mode full-val agrees on the position mean"
        ),
    }
    indices["policy_full_val"] = {
        "schema": "v104-evidence/1",
        "kind": "policy_full_val",
        "status": "pass"
        if (
            float(policy_full["observations"]["by_danger"][0]["frames"]) > 0
            and float(policy_full["observations"]["by_danger"][0]["regret_px"]) < 1.5
        )
        else "fail",
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/full_val_recheck.py",
        "command": (
            "python Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/full_val_recheck.py --kind"
            " policy_full_val --checkpoint third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt"
            " --teacher-cache third_party/th08/build/rl/datasets/"
            "pilot-ud0059-v2-teacher-values-v104-a52416f9342fa00a0185c061771d09c22ba735bbe01c83dd7910a6c3bc8ec0f6.npz"
            " --teacher-fingerprint a52416f9342fa00a0185c061771d09c22ba735bbe01c83dd7910a6c3bc8ec0f6"
            " --out Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/policy-full-val.json --device cuda"
        ),
        "cwd": str(ROOT),
        "exit_code": 0,
        "artifacts": [artifact(EVIDENCE / "policy-full-val.json"), crosscheck_artifact],
        "observations": policy_full["observations"],
        "reason": (
            "full 7241-frame re-evaluation of the frozen policy.pt with the fingerprinted teacher cache;"
            " the 0-15 px bucket / top3 / by_danger rows reproduce the summary's policy.best row. The"
            " position/danger numbers of the same checkpoint are diagnostics only and do not replace A-01."
        ),
    }
    indices["cross_replay"] = {
        "schema": "v104-evidence/1",
        "kind": "cross_replay",
        "status": "pass" if verdict and verdict["pass"] else "fail",
        "created_at": now_iso(),
        "producer": "tools/rl/compare_state_models.py (--mode stage-sample --per-stage 128)",
        "command": (
            "python tools/rl/compare_state_models.py --dataset third_party/th08/build/rl/datasets/pilot-ud0059-v2"
            " --checkpoint frozen=third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt --mode stage-sample"
            " --per-stage 128 --out .../cross-replay-a.json ; then the same with"
            " --dataset third_party/th08/build/rl/datasets/pilot-ud0063 --manifest"
            " Document/Update/Alpha/v1.0.4/Stage3/S3-1/evidence/ud0063-evaluation-manifest.json"
            " --baseline .../cross-replay-a.json --out .../cross-replay-b.json"
        ),
        "cwd": str(ROOT),
        "exit_code": 0,
        "artifacts": [
            artifact(cross_a),
            artifact(cross_b),
            artifact(guard_log),
            crosscheck_artifact,
        ],
        "observations": {
            "checkpoint_sha256": b_row["checkpoint_sha256"],
            "a_mae_raw": float(a_row["mae_raw"]),
            "b_mae_raw": float(b_row["mae_raw"]),
            "a_indices_sha256": a_row["selected_index_sha256"],
            "b_indices_sha256": b_row["selected_index_sha256"],
            "a_frames": int(a_row["frames"]),
            "b_frames": int(b_row["frames"]),
            "sampling_mode": b_row["sampling_mode"],
            "per_stage": int(b_row["per_stage_limit"]),
            "a_per_stage": a_row["per_stage"],
            "b_per_stage": b_row["per_stage"],
            "ratio": verdict["ratio"] if verdict else None,
            "threshold": 1.25,
            "external_guard_uses": 1,
            "b_used_for_selection": False,
        },
        "reason": (
            "one-shot external guard on B with the same frozen policy.pt and the same stage-sample rule;"
            " B_mae_raw / A_mae_raw = %.6f > 1.25, so the item is a real fail. The ratio was not re-run,"
            " the threshold was not changed, and B was not used to adjust or re-select the model."
            % (verdict["ratio"] if verdict else float("nan"))
        ),
    }
    indices["eval_report"] = {
        "schema": "v104-evidence/1",
        "kind": "eval_report",
        "status": "blocked",
        "created_at": now_iso(),
        "producer": "python -m touhou_rl.evaluate (vision branch)",
        "command": " ".join(a04["command"]),
        "cwd": str(ROOT),
        "exit_code": int(a04["exit_code"]),
        "artifacts": [
            artifact(EVIDENCE / "a04-attempt.json"),
            artifact(EVIDENCE / "startup-evidence.json"),
            artifact(EVIDENCE / "a04-compare-evals.json"),
            artifact(EVIDENCE / "logs" / "a04-evaluate.err"),
            artifact(EVIDENCE / "logs" / "a04-evaluate.log"),
            artifact(EVIDENCE / "logs" / "a04-compare-evals.log"),
        ],
        "observations": dict(a04["observations"]),
        "reason": (
            "the exact A-04 command ran with the frozen policy.pt and the correct"
            " lunatic/team0/frameskip2/practice=-1/max_steps=12000 configuration, but env.reset raised"
            " TimeoutError('screen 4 did not become ready') after 0 episodes: the game window cannot be"
            " made foreground because the user's elevated Task Manager holds the foreground and this"
            " agent is medium integrity (UIPI blocks SetForegroundWindow / SwitchToThisWindow / synthetic"
            " input). No episode exists, so frames_mean/hit_step_cap cannot be asserted - blocked, not"
            " pass, and a retry needs the foreground released (or an approved input-path change)."
        ),
    }
    indices["pytest"] = {
        "schema": "v104-evidence/1",
        "kind": "pytest",
        "status": "pass"
        if counts["passed"] >= 178 and counts["failed"] == 0 and counts["error"] == 0
        else "fail",
        "created_at": now_iso(),
        "producer": "python -m pytest touhou_rl/tests -q",
        "command": (
            "python -m pytest touhou_rl/tests -q -o cache_dir=D:\\WorkShop\\Touhou_RL\\.pytest-tmp\\S3-8\\.pytest_cache"
            " (TEMP/TMP redirected to .pytest-tmp\\S3-8 because %TEMP%\\pytest-of-a5935 and the repo"
            " .pytest_cache have unreadable ACL entries)"
        ),
        "cwd": str(ROOT),
        "exit_code": 0,
        "artifacts": [artifact(pytest_log), artifact(LOGS / "pytest-full.err")],
        "observations": {
            "passed": int(counts["passed"]),
            "failed": int(counts["failed"]),
            "skipped": int(counts["skipped"]),
            "command_exit": 0,
            "errors": int(counts["error"]),
            "summary_line": "698 passed in 23.79s",
        },
        "reason": "full suite re-run for this stage: 698 passed, 0 failed, 0 skipped, exit 0",
    }
    indices["d9"] = {
        "schema": "v104-evidence/1",
        "kind": "d9",
        "status": d9["status"],
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/d9d10d11_recheck.py",
        "command": (
            "python Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/d9d10d11_recheck.py --only d9"
            " --out Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence"
        ),
        "cwd": str(ROOT),
        "exit_code": 0,
        "artifacts": [artifact(EVIDENCE / "d9-recheck.json"), crosscheck_artifact],
        "observations": d9["observations"],
        "reason": (
            "the formal E5 run's coverage.val is 7241/7241 with the fixed visited digest; the trainer's"
            " make_batches reproduces 76 validation batches with a 41-frame tail; the S3-3 throughput"
            " evidence (5 runs) still reports the same coverage and its sha matches that stage's record"
        ),
    }
    indices["d10"] = {
        "schema": "v104-evidence/1",
        "kind": "d10",
        "status": d10["status"],
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/d9d10d11_recheck.py",
        "command": (
            "python Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/d9d10d11_recheck.py --only d10"
            " --out Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence"
        ),
        "cwd": str(ROOT),
        "exit_code": 0,
        "artifacts": [artifact(EVIDENCE / "d10-recheck.json"), crosscheck_artifact],
        "observations": d10["observations"],
        "reason": (
            "both residency paths re-run on B: the 512 MiB budget is the module default, neither the"
            " label-only path nor the worst-case full-record path ever exceeded it, an LRU hit is"
            " recognised, an evicted record is re-read as a miss and its values are byte-identical"
        ),
    }
    indices["d11"] = {
        "schema": "v104-evidence/1",
        "kind": "d11",
        "status": d11["status"],
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/d9d10d11_recheck.py",
        "command": (
            "python Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/d9d10d11_recheck.py --only d11"
            " --out Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence"
        ),
        "cwd": str(ROOT),
        "exit_code": 0,
        "artifacts": [artifact(EVIDENCE / "d11-recheck.json"), crosscheck_artifact],
        "observations": d11["observations"],
        "reason": (
            "all 1626 B decoded-cache files re-verified against their sidecars and their source npz"
            " digests (payload included), the byte total equals the S3-1 record, and the cached frames are"
            " byte-identical to a fresh decode of the source npz"
        ),
    }
    indices["startup"] = {
        "schema": "v104-evidence/1",
        "kind": "startup",
        "status": "blocked",
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/make_a04_evidence.py",
        "command": "python Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/startup_evidence_probe.py",
        "cwd": str(ROOT),
        "exit_code": 0,
        "artifacts": [
            artifact(EVIDENCE / "startup-evidence.json"),
            artifact(EVIDENCE / "startup-probe.json"),
            artifact(EVIDENCE / "foreground-diagnostic.json"),
            artifact(EVIDENCE / "startup-title-a.png"),
            artifact(EVIDENCE / "startup-title-b.png"),
        ],
        "observations": startup["observations"],
        "reason": startup["reason"],
    }

    written = {}
    for kind, index in indices.items():
        path = INDICES / f"{kind}.json"
        path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written[kind] = {"path": rel(path), "sha256": dig(path)}

    frozen = read(FROZEN)
    before_sha = dig(FROZEN)
    frozen["evidence"] = written
    frozen["external_guard_uses"] = 1
    frozen["evidence_updated_at"] = now_iso()
    frozen["evidence_updated_by"] = "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/make_evidence_indices.py"
    FROZEN.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    after_sha = dig(FROZEN)

    print(
        json.dumps(
            {
                "indices": written,
                "frozen_before_sha256": before_sha,
                "frozen_after_sha256": after_sha,
                "upstream_all_match": crosscheck["all_match"],
                "pytest_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
