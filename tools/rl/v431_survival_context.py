"""v4.3.1 recheck R49: every survival RATE must carry its sample size and its threshold.

R48 fixed the same problem for "admissible": the verdict is dominated by a setting, so the setting
must travel with the number. A survival rate has TWO such settings -- how many episodes (n) and
what counts as surviving (300 decisions / 600 frames / 全长) -- and "6 倍存活率优势" is unreadable
without both: at n=40 the comparison was 4/40 vs 0/40, at n=80 it is 6/80 vs 1/80.

Rule: a line that states a survival RATE or RATIO (a percentage or "N 倍" near 存活) must also
carry an n-marker (n=, /80, /40, /20, 80 次, episode 数) and a threshold-marker
(300 决策 / 全长 / 600 帧 / 180 帧 / alive@).

Writes only Document/Update/Alpha/v4.3.1/evidence/R49/.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DOCS = {
    "report": REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "ROUTE_RECHECK_2026-11-15.md",
    "summary": REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "SUMMARY_FOR_DECISION.md",
    "self_audit": REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "SELF_AUDIT.md",
}
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R49"

RATE = re.compile(r"\d+(?:\.\d+)?\s*%|\d+\s*倍")
N_MARK = ("n=", "n≈", "n = ", "/80", "/40", "/20", "/10", "/128", "80 次", "40 次",
          "episode 数", "80 episode", "129 episode")
THRESH = ("300 决策", "全长", "600 帧", "180 帧", "alive@", "跑满")


def main() -> int:
    findings = []
    scanned = 0
    for tag, path in DOCS.items():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "存活" not in line:
                continue
            scanned += 1
            if not RATE.search(line):
                continue
            has_n = any(m in line for m in N_MARK)
            has_thr = any(m in line for m in THRESH)
            if has_n and has_thr:
                continue
            missing = []
            if not has_n:
                missing.append("n")
            if not has_thr:
                missing.append("threshold")
            findings.append({
                "doc": tag, "line": lineno,
                "has_n": has_n, "has_threshold": has_thr,
                "missing": "+".join(missing),
                "text": line.strip()[:150],
            })

    out = {
        "schema": "v431-recheck-r49/1",
        "question": ("does every survival RATE carry both its sample size and its threshold?"),
        "rule": {"rate_pattern": RATE.pattern, "n_markers": list(N_MARK),
                 "threshold_markers": list(THRESH),
                 "why": ("the same comparison is 4/40 vs 0/40 at n=40 and 6/80 vs 1/80 at n=80, "
                         "so a rate without n is not comparable across the report")},
        "counts": {"lines_mentioning_survival": scanned, "violations": len(findings)},
        "violations": findings,
        "verdict": {"clean": not findings, "reading": None},
    }
    v = out["verdict"]
    v["reading"] = (
        f"{scanned} lines mention survival; every one that states a RATE also carries n and the "
        "threshold" if not findings else
        f"{len(findings)} lines state a survival rate without n and/or the threshold")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "survival_context.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"lines mentioning survival : {scanned}")
    print(f"rates missing context     : {len(findings)}")
    for f in findings:
        print(f"  {f['doc']} L{f['line']:>5d} [{f['missing']:>13s}] {f['text'][:96]}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'survival_context.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
