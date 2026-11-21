"""v4.3.1 recheck R39: audit my own quotations against the original v4.3.0 documents.

The report's §6.1/§6.2 tables quote the original review's claims ("原说法"). If any of those
quotes is my paraphrase presented as a quotation, the whole correction list is weaker than it
looks -- a reader cannot check a claim that does not exist in the source.

This walks every K/C row, extracts the "original claim" cell, and searches the v4.3.0 document
set for a long enough literal fragment. Nothing is judged semantically: a row is FOUND when at
least one 8-character fragment of the cell appears verbatim in a v4.3.0 document, and NOT FOUND
otherwise. Both outcomes are reported honestly, including the ones that turn out to be my
paraphrase.

Writes only Document/Update/Alpha/v4.3.1/evidence/R39/.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
REPORT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "ROUTE_RECHECK_2026-11-15.md"
SRC_DIR = REPO / "Document" / "Update" / "Alpha" / "v4.3.0"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R39"
FRAG = 8
STRIP = str.maketrans({c: " " for c in "*`「」\"'“”‘’（）()【】[]|\\/_,.;:!?—–…§"})


def clean(text: str) -> str:
    return re.sub(r"\s+", "", text.translate(STRIP))


def load_corpus() -> dict[str, str]:
    out = {}
    for p in sorted(SRC_DIR.glob("*.md")):
        out[p.name] = clean(p.read_text(encoding="utf-8"))
    return out


def find_fragment(claim: str, corpus: dict[str, str]) -> tuple[str, str] | None:
    """Return (document, fragment) for the first verbatim fragment found, else None."""
    text = clean(claim)
    if not text:
        return None
    size = FRAG if len(text) >= FRAG else len(text)
    if size < 4:
        return None
    for start in range(0, len(text) - size + 1):
        frag = text[start:start + size]
        # skip fragments that are mostly punctuation or too generic
        if sum(1 for ch in frag if "\u4e00" <= ch <= "\u9fff" or ch.isalnum()) < size - 1:
            continue
        for name, body in corpus.items():
            if frag in body:
                return name, frag
    return None


def cell(row: str, index: int) -> str:
    parts = [p.strip() for p in row.strip().strip("|").split("|")]
    return parts[index] if index < len(parts) else ""


def main() -> int:
    report = REPORT.read_text(encoding="utf-8")
    corpus = load_corpus()
    rows = []
    for line in report.splitlines():
        m = re.match(r"^\|\s*([KC])(\d+)\s*\|", line)
        if not m:
            continue
        kind, num = m.group(1), int(m.group(2))
        if kind != "C":
            continue  # §6.1 keep rows are my endorsements, not quotations
        claim = cell(line, 1)
        if not claim or claim.startswith("（我自己的"):
            rows.append({"id": f"{kind}{num}", "claim": claim,
                         "status": "internal", "where": "",
                         "fragment": "",
                         "note": "the claim is already labelled as my own, so no source quote "
                                 "is owed"})
            continue
        got = find_fragment(claim, corpus)
        rows.append({
            "id": f"{kind}{num}", "claim": claim,
            "status": "found" if got else "not_found",
            "where": got[0] if got else "",
            "fragment": got[1] if got else "",
            "note": "" if got else "no 8-character fragment appears verbatim in any v4.3.0 doc",
        })

    found = [r for r in rows if r["status"] == "found"]
    not_found = [r for r in rows if r["status"] == "not_found"]
    internal = [r for r in rows if r["status"] == "internal"]
    out = {
        "schema": "v431-recheck-r39/1",
        "question": ("does every claim the report attributes to the original v4.3.0 review "
                     "actually appear in it?"),
        "method": {"fragment_chars": FRAG,
                   "match": "verbatim 8-character fragment, no semantic matching",
                   "corpus": sorted(corpus.keys())},
        "counts": {"rows": len(rows), "found": len(found), "not_found": len(not_found),
                   "internal": len(internal)},
        "rows": rows,
        "verdict": {
            "all_attributed_claims_found": len(not_found) == 0,
            "reading": None,
        },
    }
    v = out["verdict"]
    if not_found:
        v["reading"] = (f"{len(not_found)} of {len(found) + len(not_found)} attributed claims "
                        f"have no verbatim fragment in the v4.3.0 documents: those are either my "
                        f"paraphrases presented as quotes (to be fixed) or quotes of files "
                        f"outside this directory")
    else:
        v["reading"] = ("every attributed claim has a verbatim fragment in a v4.3.0 document, so "
                        "no correction rests on a fabricated quotation")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "quote_audit.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"corpus: {len(corpus)} v4.3.0 documents, fragment size {FRAG} chars")
    print(f"rows: {len(rows)}  found: {len(found)}  not found: {len(not_found)}  "
          f"internal: {len(internal)}")
    print()
    lines = []
    for r in rows:
        mark = {"found": "OK ", "not_found": "MISS", "internal": "own "}[r["status"]]
        where = f" <- {r['where']}" if r["where"] else ""
        lines.append(f"  {mark} {r['id']:>4s} {r['claim'][:44]}{where}")
    (OUT / "quote_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in rows:
        mark = {"found": "OK ", "not_found": "MISS", "internal": "own "}[r["status"]]
        where = r["where"] or ""
        try:
            print(f"  {mark} {r['id']:>4s} {r['claim'][:40]} {where}")
        except UnicodeEncodeError:
            print(f"  {mark} {r['id']:>4s} (see quote_audit.txt for the text) {where}")
    print()
    print(f"verdict: see quote_audit.json")
    print(f"wrote: {OUT / 'quote_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
