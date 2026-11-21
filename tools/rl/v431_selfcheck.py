"""v4.3.1 selfcheck: one command that runs every evidence-level checker and prints a verdict table.

The eight checks added in R63-R73 live in eight scripts, three of which take minutes. A reader who
wants to know "is this report still sound?" should not have to discover that ordering.

Two modes:
    --fast   (default) the five cheap checks: R69 headers, R70 percentages, R71 cost chain,
             R72 K rows, R73 leftovers. Each is a few seconds. This is the "5-minute" path.
    --full   adds R63 (artifact absence), R64 (N4/N5), R65 (evidence types), R66 (split scripts),
             R67 (agreement), R68 (sign conventions). R63/R64 walk ~150k files (~4 min each).

Each checker is a separate process, so one crashing does not hide the others, and the exit code of
this script is non-zero if any check fails.

Deliberately NOT part of run_all_audits.ps1: that suite is the assertion gate (v431_verify_report.py
runs last over all stored evidence). This is a convenience aggregator for a human reader, and
duplicating the slow walks inside the gate would double its runtime for no extra assurance.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
PY = sys.executable

FAST = [
    ("R69 表头口径", "v431_headers.py"),
    ("R70 百分比自证", "v431_percentages.py"),
    ("R71 成本链", "v431_cost_chain.py"),
    ("R72 K 表引用", "v431_k_rows.py"),
    ("R73 旧值残留", "v431_value_leftovers.py"),
]
FULL_EXTRA = [
    ("R63 产物缺席（约 4 min）", "v431_artifact_absence.py"),
    ("R64 N4/N5（约 4 min）", "v431_n4_n5.py"),
    ("R65 证据类型", "v431_evidence_types.py"),
    ("R66 切分脚本", "v431_split_scripts.py"),
    ("R67 全称断言", "v431_agreement_18.py"),
    ("R68 符号口径", "v431_sign_conventions.py"),
]


def run(name: str, script: str) -> tuple[str, int, float, str]:
    path = REPO / "tools" / "rl" / script
    if not path.is_file():
        return name, -1, 0.0, f"missing: {script}"
    t0 = time.time()
    proc = subprocess.run([PY, str(path)], cwd=str(REPO), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    dt = time.time() - t0
    out = (proc.stdout or "").strip().splitlines()
    # prefer the checker's own verdict line over the trailing "wrote:" line
    verdict = [ln for ln in out if ln.startswith("verdict:")]
    if verdict:
        last = verdict[-1][len("verdict:"):].strip()[:110]
    elif out:
        last = out[-1][:110]
    else:
        err = (proc.stderr or "").strip().splitlines()
        last = err[-1][:110] if err else "(no output)"
    return name, proc.returncode, dt, last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run the slow tree walks")
    ap.add_argument("--list", action="store_true", help="print what would run")
    args = ap.parse_args()

    plan = FAST + (FULL_EXTRA if args.full else [])
    if args.list:
        for name, script in plan:
            print(f"  {script:34s} {name}")
        return 0

    print(f"mode: {'full' if args.full else 'fast'}   checks: {len(plan)}")
    print()
    results = []
    for name, script in plan:
        n, code, dt, last = run(name, script)
        results.append((n, code, dt, last))
        flag = "ok  " if code == 0 else f"FAIL({code})"
        print(f"  [{flag}] {n:<26s} {dt:6.2f}s  {last}")

    bad = [r for r in results if r[1] != 0]
    print()
    print(f"non-zero: {len(bad)} of {len(results)}   总耗时 {sum(r[2] for r in results):.1f}s")
    if bad:
        print("failed:")
        for n, code, dt, last in bad:
            print(f"  - {n}: exit {code}  {last}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
