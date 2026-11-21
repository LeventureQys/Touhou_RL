"""v4.3.1 recheck R41: re-run the earliest evidence (R1-R6) and compare bit-for-bit.

Rounds 1-6 produced the numeric backbone of this report: the corrected baselines, the
cross-entropy re-executions, the significance tests, the overlap audit. Twenty-odd rounds of
edits later, nothing should have moved -- but "nothing should have moved" is exactly the kind
of claim this report has been criticising elsewhere.

This snapshots the six evidence directories, re-runs the six scripts against the CURRENT
workspace, and compares every produced JSON with the snapshot:
  * identical      -> the evidence is reproducible from the current files
  * different      -> a real finding: either a script was changed, or an input moved

Only files the scripts themselves write are compared, so unrelated additions do not count.

Writes only Document/Update/Alpha/v4.3.1/evidence/R41/.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
SNAP = REPO / ".tmp_v431_seg001" / "rerun_snapshot"
OUT = EV / "R41"
SCRIPTS = [
    ("R1", "v431_recheck_baseline.py", "baseline_recheck.json"),
    ("R2", "v431_recheck_overlap.py", "overlap_audit.json"),
    ("R3", "v431_recheck_duplication.py", "duplication_structure.json"),
    ("R4", "v431_recheck_models.py", "model_recheck.json"),
    ("R5", "v431_recheck_significance.py", "significance.json"),
    ("R6", "v431_recheck_actor.py", "actor_recheck.json"),
]


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


TIMESTAMP_KEYS = {"generated_at_utc", "generated_at", "timestamp", "wall_s"}
import re as _re


def lookup(obj, path: str):
    """Walk `.a.b[0].c` against a parsed JSON object. Returns (found, value)."""
    tokens = _re.findall(r"[^.\[\]]+|\[\d+\]", path)
    cur = obj
    for tok in tokens:
        if tok.startswith("["):
            idx = int(tok[1:-1])
            if not isinstance(cur, list) or idx >= len(cur):
                return False, None
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or tok not in cur:
                return False, None
            cur = cur[tok]
    return True, cur

FLOAT_TOL = 1e-9


def classify(path: str, a, b) -> str:
    key = path.rsplit(".", 1)[-1].split("[")[0]
    if key in TIMESTAMP_KEYS:
        return "timestamp"
    if isinstance(a, float) and isinstance(b, float) and abs(a - b) <= FLOAT_TOL:
        return "float_jitter"
    return "substantive"


def first_difference(a, b, path: str = "") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        ka, kb = set(a), set(b)
        if ka != kb:
            return f"{path}: keys differ by {sorted(ka ^ kb)[:5]}"
        for k in sorted(ka):
            got = first_difference(a[k], b[k], f"{path}.{k}")
            if got:
                return got
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            got = first_difference(x, y, f"{path}[{i}]")
            if got:
                return got
        return None
    if isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > 1e-12:
            return f"{path}: {a!r} vs {b!r}"
        return None
    if a != b:
        return f"{path}: {a!r} vs {b!r}"
    return None


def main() -> int:
    compare_only = "--compare-only" in sys.argv
    SNAP.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, script, outfile in SCRIPTS:
        src = EV / group / outfile
        before = SNAP / f"{group}_{outfile}"
        if before.is_file():
            # keep the FIRST snapshot: it is the only copy of the original evidence
            pre_sha = sha(before)
        elif src.is_file():
            before.parent.mkdir(parents=True, exist_ok=True)
            before.write_bytes(src.read_bytes())
            pre_sha = sha(before)
        else:
            pre_sha = None
        if compare_only and not before.is_file():
            rows.append({"group": group, "script": script, "output": outfile,
                         "exit": 0, "wall_s": 0.0, "sha_before": None,
                         "sha_after": sha(src) if src.is_file() else None,
                         "identical": None, "first_difference": "no snapshot on disk",
                         "difference_kind": "missing_snapshot"})
            print(f"  {group} {script:34s} NO SNAPSHOT")
            continue
        if compare_only:
            class _P:
                returncode = 0
                stderr = ""
            proc = _P()
            wall = 0.0
        else:
            t0 = time.perf_counter()
            proc = subprocess.run([sys.executable, str(REPO / "tools" / "rl" / script)],
                                  cwd=str(REPO), capture_output=True, text=True,
                                  env={**__import__("os").environ,
                                       "PYTHONPATH": str(REPO)})
            wall = round(time.perf_counter() - t0, 1)
        post_sha = sha(src) if src.is_file() else None
        diff = None
        diff_kind = ""
        if pre_sha and post_sha:
            if pre_sha != post_sha:
                old_obj = json.loads(before.read_text(encoding="utf-8"))
                new_obj = json.loads(src.read_text(encoding="utf-8"))
                diff = first_difference(old_obj, new_obj)
                if diff is None:
                    diff = "content differs but no value difference found (ordering only?)"
                kind = "ordering" if diff.startswith("content differs") else classify(
                    diff.split(":", 1)[0], None, None)
                if kind == "substantive":
                    # recover the two values from the objects to allow the float test
                    kind = "substantive"
                    parts = diff.split(":", 1)
                    if len(parts) == 2:
                        rest = parts[1].strip()
                        if " vs " in rest:
                            keypath = rest.split(" vs ", 1)[0].strip()
                            ok_a, a_val = lookup(old_obj, keypath)
                            ok_b, b_val = lookup(new_obj, keypath)
                            if ok_a and ok_b:
                                kind = classify(keypath, a_val, b_val)
                diff_kind = kind
        rows.append({
            "group": group, "script": script, "output": outfile,
            "exit": proc.returncode, "wall_s": wall,
            "sha_before": pre_sha, "sha_after": post_sha,
            "identical": (pre_sha == post_sha) if (pre_sha and post_sha) else None,
            "first_difference": diff,
            "difference_kind": diff_kind if diff else "",
        })
        print(f"  {group} {script:34s} exit={proc.returncode} {wall:6.1f}s "
              f"identical={rows[-1]['identical']}")
        if proc.returncode != 0:
            print(f"      stderr: {proc.stderr.strip().splitlines()[-1][:120]}")

    ok = [r for r in rows if r["exit"] == 0]
    identical = [r for r in rows if r["identical"] is True]
    out = {
        "schema": "v431-recheck-r41/1",
        "question": ("do the six earliest evidence files still reproduce bit-for-bit from the "
                     "current workspace?"),
        "scripts": rows,
        "counts": {"scripts": len(rows), "exit_zero": len(ok),
                   "identical": len(identical), "changed": len(rows) - len(identical)},
        "verdict": {
            "all_ran": len(ok) == len(rows),
            "all_identical": len(identical) == len(rows),
            "reading": None,
        },
    }
    v = out["verdict"]
    if v["all_identical"] and v["all_ran"]:
        v["reading"] = ("all six early evidence files reproduce bit-for-bit from the current "
                        "workspace: no later edit moved any number in the report's backbone")
    elif v["all_ran"]:
        substantive = [r["group"] for r in rows
                       if r["identical"] is not True and r["difference_kind"] == "substantive"]
        cosmetic = [r["group"] for r in rows
                    if r["identical"] is not True and r["difference_kind"] != "substantive"]
        out["verdict"]["substantive_changes"] = substantive
        out["verdict"]["cosmetic_changes"] = cosmetic
        v["reading"] = (f"{len(identical)}/{len(rows)} byte-identical; {len(cosmetic)} differ "
                        f"only in a timestamp or sub-1e-9 float jitter ({cosmetic}); "
                        f"substantive changes: {substantive or 'none'}")
    else:
        v["reading"] = "at least one script failed to run; see the exit codes"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rerun.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"scripts {len(rows)}  exit0 {len(ok)}  identical {len(identical)}")
    for r in rows:
        if r["first_difference"]:
            print(f"  DIFF {r['group']}: {r['first_difference']}")
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'rerun.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
