"""v4.3.1 recheck R48: every admissibility RESULT must carry its segment length.

R47 showed the admissibility verdict depends almost entirely on the segment length: at 3 s none of
the nine dense windows is admissible, at 10 s eight of nine are. A sentence that says "8 of 9
windows are admissible" without saying "at 10 s" is therefore not merely imprecise -- read in
another context it means the opposite.

Rule enforced here: any line that states a NUMBER of admissible windows/states (patterns such as
"8 个合格", "2/3", "合格率 = 0 / 4") must also carry a segment-length marker on the same line
(3 s / 10 s / 180 帧 / 600 帧 / 段长 / 521 / 552 / 8.68 / 9.2 s), unless the line is a
DELIBERATELY length-agnostic rule (the S1b stop condition itself, which is defined per side and
per segment by construction).

Writes only Document/Update/Alpha/v4.3.1/evidence/R48/.
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
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R48"

MARKERS = ("3 s", "3s", "10 s", "10s", "180 帧", "600 帧", "段长", "521", "552",
           "8.68", "9.2 s", "300 决策")
# a line states a RESULT if it reports how many windows/states passed
RESULT = re.compile(r"(\d+\s*个合格|\d+\s*/\s*\d+\s*合格|合格率\s*=?\s*\d|"
                    r"合格\s*\*\*\d|\d+\s*个窗口.*合格|\d+\s*个状态.*合格)")
# rules are length-agnostic on purpose
RULE_MARKERS = ("S1b", "每侧 ≥4", "合格状态 < 4", "停止条件")


def main() -> int:
    findings = []
    scanned = 0
    for tag, path in DOCS.items():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "合格" not in line:
                continue
            scanned += 1
            if not RESULT.search(line):
                continue
            if any(m in line for m in MARKERS):
                continue
            if any(m in line for m in RULE_MARKERS):
                continue
            findings.append({"doc": tag, "line": lineno, "text": line.strip()[:150]})

    out = {
        "schema": "v431-recheck-r48/1",
        "question": ("does every admissibility RESULT in the deliverables carry its segment "
                     "length?"),
        "rule": {
            "result_pattern": RESULT.pattern,
            "segment_markers": list(MARKERS),
            "length_agnostic_exemptions": list(RULE_MARKERS),
            "why": ("at 3 s none of the nine dense windows is admissible; at 10 s eight are, so "
                    "a result without its segment length can be read as the opposite"),
        },
        "counts": {"lines_mentioning_admissibility": scanned, "violations": len(findings)},
        "violations": findings,
        "verdict": {"clean": not findings, "reading": None},
    }
    v = out["verdict"]
    v["reading"] = (
        f"{scanned} lines mention admissibility; every one that states a RESULT also carries its "
        "segment length" if not findings else
        f"{len(findings)} lines state an admissibility result without a segment length")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "segment_context.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"lines mentioning admissibility : {scanned}")
    print(f"results without a segment     : {len(findings)}")
    for f in findings:
        print(f"  {f['doc']} L{f['line']}: {f['text'][:110]}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'segment_context.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
