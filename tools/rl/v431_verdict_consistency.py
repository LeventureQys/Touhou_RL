"""v4.3.1 recheck R56: do the three deliverables agree about arm A?

The report, the decision summary and the preregistration all state a verdict about arm A. After
R52-R55 that verdict became two-directional ("upper tail yes, typical no"), and a reader who only
opens one of the three files must not receive a materially different claim.

This checks consistency mechanically: every place that characterises arm A's outcome must contain
the same THREE load-bearing pieces of information --

  1. the pre-registered tests were not significant (so it is not "learned");
  2. the upper tail does favour it (so it is not "no difference");
  3. the typical episode does NOT favour it (so the mean alone is not the whole story).

A missing piece is not a typo: it is a file that will mislead whichever reader opens it first.

Writes only Document/Update/Alpha/v4.3.1/evidence/R56/.
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
    "prereg": REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "PREREGISTRATION_SEG_001.md",
}
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R56"

# the three pieces, each with the phrases that count as carrying it
PIECES = {
    "not_significant": ("不显著", "0.1167", "0.0584", "未判定"),
    "upper_tail_favours": ("294.8", "上尾", "154.5", "15/80", "前 10"),
    "typical_does_not": ("中位数", "−15.0", "Q1", "典型"),
}


def main() -> int:
    rows = []
    for tag, path in DOCS.items():
        text = path.read_text(encoding="utf-8")
        has = {k: [p for p in v if p in text] for k, v in PIECES.items()}
        rows.append({
            "doc": tag,
            "pieces": {k: {"present": bool(v), "matched": v} for k, v in has.items()},
            "complete": all(bool(v) for v in has.values()),
            "arm_a_mentions": len(re.findall(r"臂 A", text)),
        })

    # stronger check for the 2-page summary: the three pieces must sit inside the SAME section,
    # otherwise a reader who stops after §3 gets a one-sided claim
    sm = DOCS["summary"].read_text(encoding="utf-8")
    start = sm.find("### 臂 A 的最终数字")
    nxt = sm.find("\n### ", start + 1)
    section = sm[start:nxt] if start != -1 and nxt != -1 else ""
    sec_has = {k: [x for x in v if x in section] for k, v in PIECES.items()}
    section_ok = all(bool(v) for v in sec_has.values())
    out_section = {"section": "SUMMARY §3 臂 A 的最终数字",
                   "chars": len(section),
                   "pieces": {k: {"present": bool(v), "matched": v} for k, v in sec_has.items()},
                   "all_in_one_section": section_ok}

    incomplete = [r["doc"] for r in rows if not r["complete"]]
    if not section_ok:
        incomplete.append("summary_section")
    out = {
        "schema": "v431-recheck-r56/1",
        "question": "三份交付物对臂 A 的结论是否一致（三个要素是否都在）？",
        "pieces_required": {k: list(v) for k, v in PIECES.items()},
        "docs": rows,
        "summary_section": out_section,
        "counts": {"docs": len(rows),
                   "complete": len(rows) - len([d for d in incomplete if d != "summary_section"]),
                   "incomplete": len(incomplete)},
        "incomplete_docs": incomplete,
        "verdict": {"consistent": not incomplete, "reading": None},
    }
    v = out["verdict"]
    v["reading"] = (
        "all three deliverables carry all three pieces: the pre-registered tests are not "
        "significant, the upper tail favours arm A, and the typical episode does not -- so a "
        "reader of any single file gets the same two-directional verdict"
        if not incomplete else
        f"{incomplete} are missing at least one piece of the arm A verdict")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verdict_consistency.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'doc':>8s} {'arm A mentions':>14s} " + " ".join(f"{k:>18s}" for k in PIECES))
    print("-" * 84)
    for r in rows:
        flags = " ".join(f"{('yes' if r['pieces'][k]['present'] else 'NO'):>18s}" for k in PIECES)
        print(f"{r['doc']:>8s} {r['arm_a_mentions']:14d} {flags}")
    print()
    print(f"SUMMARY §3 (one section, {out_section['chars']} chars): " +
          " ".join(f"{k}={'yes' if out_section['pieces'][k]['present'] else 'NO'}" for k in PIECES))
    print()
    print(f"complete: {out['counts']['complete']}/{out['counts']['docs']}  "
          f"incomplete: {incomplete or 'none'}")
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'verdict_consistency.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
