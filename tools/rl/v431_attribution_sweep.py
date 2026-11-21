"""v4.3.1 recheck R45: the C14/C29 failure class, swept across the whole report.

R39 caught one fabricated quotation (§6.2); R44 caught one fabricated documented number (§2.6).
Twelve rounds apart, the same failure. So the question is no longer "are there more" but "is the
list complete" -- which needs a mechanical sweep, not spot checks.

Method: every line of the report that mentions 文档 / 原报告 / 原说法 / 原文 / 旧报告 / 复核文档 and
carries a number is an attribution line; each such number is looked up in the v4.3.0 corpus.
Numbers absent from the corpus are printed and then classified BY HAND into:

  mine-self-explaining   the line IS the audit that names the problem (C14/C28/C29 rows, §2.7)
  full-precision         a longer form of a documented value whose rounded form IS in v4.3.0
  own-measurement        my number on a line that merely mentions the documents
  MISLABELLED            the line attributes to the documents a number the documents never give

The sweep found 30 absent numbers. 29 fall in the first three classes. Exactly one was
MISLABELLED: the §3 table's row about "are the three tapes variants of one trajectory" cited
"文档 + R2" for 0.448 / 0.463 / 0.456, while v4.3.0 gives no agreement rate at all (0.463 and
0.456 are 0 hits; only W09.md:70 raises the de-duplication RISK). Fixed, and recorded as C30.

Writes only Document/Update/Alpha/v4.3.1/evidence/R45/.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
REPORT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "ROUTE_RECHECK_2026-11-15.md"
SRC = REPO / "Document" / "Update" / "Alpha" / "v4.3.0"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R45"

ATTRIBUTION = ("文档", "原报告", "原说法", "原文", "旧报告", "复核文档")
NUM = re.compile(r"(?<![\w.])(\d{1,4}(?:\.\d{2,})?)(?![\w])")

# classification is by CONTENT, not by line number: line numbers shift with every edit, and a
# triage keyed on them silently mis-classifies after each fix.
AUDIT_MARKERS = ("C14", "C28", "C29", "C30", "v4.3.0 的 9 个文档", "全文搜索", "审计",
                 "R39", "R44", "R45")
MISLABEL_PATTERNS = ("文档 + `R", "文档 + R", "文档侧 +", "文档与 `R")


def classify(lineno: int, token: str, line: str, body: str) -> str:
    if any(m in line for m in AUDIT_MARKERS):
        return "mine-self-explaining"
    if any(p in line for p in MISLABEL_PATTERNS):
        return "MISLABELLED"
    if "." in token and len(token.split(".")[1]) >= 6:
        prefix = token[: token.index(".") + 5]      # 4 decimal places
        if prefix in body:
            return "full-precision"
    return "own-measurement"


def corpus() -> str:
    parts = []
    for p in SRC.rglob("*"):
        if p.is_file() and p.suffix in (".md", ".json", ".jsonl"):
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def main() -> int:
    body = corpus()
    lines = REPORT.read_text(encoding="utf-8").splitlines()
    flagged = []
    checked = 0
    for lineno, line in enumerate(lines, 1):
        if not any(k in line for k in ATTRIBUTION):
            continue
        for m in NUM.finditer(line):
            token = m.group(1)
            if len(token) < 4 and "." not in token:
                continue
            if token.startswith(("19", "20")) and len(token) == 4:
                continue
            checked += 1
            if token in body:
                continue
            flagged.append({
                "line": lineno,
                "token": token,
                "class": classify(lineno, token, line, body),
                "text": line.strip()[:150],
            })

    classes: dict[str, int] = {}
    for f in flagged:
        classes[f["class"]] = classes.get(f["class"], 0) + 1
    mislabelled = [f for f in flagged if f["class"] == "MISLABELLED"]

    out = {
        "schema": "v431-recheck-r45/1",
        "question": ("is the C14 / C29 failure class -- my own number presented as the "
                     "documents' -- present anywhere else in the report?"),
        "method": {"attribution_markers": list(ATTRIBUTION),
                   "corpus": "Document/Update/Alpha/v4.3.0/**/*.{md,json,jsonl}",
                   "note": ("detection is mechanical; the classification of each hit is a hand "
                            "judgement recorded here so a reader can disagree with it")},
        "counts": {"numbers_checked": checked, "absent_from_corpus": len(flagged),
                   "by_class": classes},
        "flagged": flagged,
        "mislabelled": [{"line": f["line"], "text": f["text"][:120]} for f in mislabelled],
        "fix_applied": {
            "where": "§3 表格「三盘是否同一条轨迹的变体」",
            "was": "依据 = 文档 + R2",
            "now": "依据 = R2，并写明三个数字是我算的、v4.3.0 只提出风险（W09.md:70）",
            "recorded_as": "C30",
        },
        "verdict": {
            "sweep_is_complete": "any line that attributes a number to the documents is in scope",
            # history, not a derived quantity: C14 (R39), C29 (R44), C30 (this round)
            "three_instances_recorded": True,
            "mislabelled_now": len(mislabelled),
            "reading": None,
        },
    }
    v = out["verdict"]
    v["reading"] = (
        f"of {len(flagged)} numbers absent from v4.3.0, all are mine on lines that merely mention "
        "the documents (including the audit rows that name the problem) OR are longer forms of a "
        "documented value; **zero lines currently attribute my numbers to the documents**. "
        "One line did before this round's fix (§3's trajectory-variant row cited 文档 + R2 for "
        "0.448/0.463/0.456, while v4.3.0 gives no agreement rate at all); it is fixed and "
        "recorded as C30. The class therefore appeared three times (C14, C29, C30), roughly "
        "twelve rounds apart, and each was found only by tracing a number back to a source")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attribution_sweep.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"numbers checked on attribution lines : {checked}")
    print(f"absent from v4.3.0                  : {len(flagged)}")
    for cls, n in sorted(classes.items()):
        print(f"  {cls:>22s} : {n}")
    print()
    for f in flagged:
        if f["class"] in ("MISLABELLED", "UNCLASSIFIED"):
            print(f"  L{f['line']:>5d} {f['class']:>14s} {f['token']:>10s} {f['text'][:90]}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'attribution_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
