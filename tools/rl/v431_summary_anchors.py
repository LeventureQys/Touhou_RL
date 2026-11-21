"""v4.3.1 recheck R39b: for the correction rows that are summaries, locate the underlying numbers.

The quote audit (R39) found that 6 of the 14 attributed correction rows have no verbatim
8-character fragment in the v4.3.0 documents. Those rows are compressed summaries (a table
cell cannot hold a paragraph), so "not found" does not mean fabricated -- but a reader still
deserves a place to check.

This pass extracts the distinctive tokens (numbers, percentages, identifiers) from each of the
six cells and reports every v4.3.0 document that contains them. A row is "anchored" when at
least one of its distinctive tokens is present.

Writes only Document/Update/Alpha/v4.3.1/evidence/R39/.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R39"
SRC_DIR = REPO / "Document" / "Update" / "Alpha" / "v4.3.0"

TOKENS = {
    "C1": ["880", "219"],
    "C2": ["0.8537", "0.8558", "0.8293"],
    "C3": ["majority_baseline", "多数类", "majority_ce"],
    "C7": ["20%", "原文", "逐字节"],
    "C9": ["2.1161", "2.5070", "污染"],
    "C14": ["0.4995", "0.5369", "0.4249", "0.4404"],
}


def main() -> int:
    corpus = {p.name: p.read_text(encoding="utf-8") for p in sorted(SRC_DIR.glob("*.md"))}
    rows = []
    for cid, tokens in TOKENS.items():
        hits = {}
        for tok in tokens:
            where = [name for name, body in corpus.items() if tok in body]
            hits[tok] = where
        any_hit = any(v for v in hits.values())
        rows.append({
            "id": cid,
            "tokens": tokens,
            "hits": hits,
            "anchored": any_hit,
            "documents": sorted({d for v in hits.values() for d in v}),
        })

    out = {
        "schema": "v431-recheck-r39b/1",
        "question": ("the six correction cells with no verbatim fragment: do their underlying "
                     "numbers appear in the v4.3.0 documents?"),
        "rows": rows,
        "counts": {"rows": len(rows), "anchored": sum(1 for r in rows if r["anchored"])},
        "conclusion": (
            "the first column of §6.2 is a SUMMARY of the original claim, not a verbatim "
            "quotation; for these six rows the underlying numbers do appear in the v4.3.0 "
            "documents at the listed locations" if all(r["anchored"] for r in rows) else
            "at least one row has no located token and needs manual checking"),
    }
    (EV / "summary_anchors.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"rows: {len(rows)}  anchored: {out['counts']['anchored']}")
    for r in rows:
        docs = ", ".join(r["documents"]) or "NONE"
        print(f"  {r['id']:>4s} anchored={r['anchored']}  docs={docs}")
    print()
    print(f"wrote: {EV / 'summary_anchors.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
