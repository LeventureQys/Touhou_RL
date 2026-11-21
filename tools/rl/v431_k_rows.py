"""v4.3.1 recheck R72: does every K row cite a group that actually contains its numbers?

The K table (18 rows) is the report's "what survives" list. Each row carries numbers and cites
evidence groups. Two failure modes are possible and neither was checked:

  1. a row cites a group that does NOT contain the number it uses (wrong attribution);
  2. a row carries no citation at all (unfalsifiable).

For each K row this strips the R-references, extracts the numbers, loads every cited group's
JSON, and reports per number whether it is found there.

A "not found" is not automatically an error: some rows restate a quantity computed elsewhere
(e.g. a share derived from two numbers, or a rate quoted from another row). Those are listed
separately so the distinction stays visible.

Writes only Document/Update/Alpha/v4.3.1/evidence/R72/.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
DOC = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "ROUTE_RECHECK_2026-11-15.md"
OUT = EV / "R72"

ROW = re.compile(r"^\| K(\d+) \|")
REF = re.compile(r"R(\d+)")
# a claim number stands alone: not glued to letters on either side
NUM = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?(?![A-Za-z0-9])")
# numbers that are not claims: the K/R ids, dates, and small ordinals
SKIP = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
        "17", "18", "2026", "11", "15"}


def corpus(n: int) -> str:
    d = EV / f"R{n}"
    if not d.is_dir():
        return ""
    parts = []
    for p in sorted(d.glob("*.json")):
        try:
            parts.append(p.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(parts)


def main() -> int:
    rows = []
    lines = DOC.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, 1):
        m = ROW.match(line)
        if not m:
            continue
        kid = int(m.group(1))
        refs = sorted({int(r) for r in REF.findall(line)})
        body = REF.sub("", line)          # drop the citations before extracting numbers
        nums = [n for n in NUM.findall(body) if n not in SKIP]
        joined = "\n".join(corpus(r) for r in refs)
        # every number appearing anywhere in the cited groups, for tolerance matching
        pool = []
        for tok in re.findall(r"\d+(?:\.\d+)?", joined):
            try:
                pool.append(float(tok))
            except ValueError:
                pass
        found, missing = [], []
        for n in nums:
            try:
                v = float(n)
            except ValueError:
                missing.append(n)
                continue
            # a 4-dp claim matches a 2-dp stored value and vice versa
            hit = any(abs(v - q) <= max(5e-4, abs(v) * 1e-6) for q in pool)
            (found if hit else missing).append(n)
        rows.append({
            "k": kid, "line": i, "refs": refs,
            "n_citations": len(refs),
            "numbers": nums, "numbers_found": found, "numbers_missing": missing,
            "has_citation": bool(refs),
        })

    nocite = [r["k"] for r in rows if not r["has_citation"]]
    withmiss = [{"k": r["k"], "missing": r["numbers_missing"], "refs": r["refs"]}
                for r in rows if r["numbers_missing"]]
    out = {
        "schema": "v431-recheck-r72/1",
        "question": "§6 的 18 条 K 判断，每条的引用是否真的含有它用的数字？有没有无引用的？",
        "rows": rows,
        "counts": {"k_rows": len(rows),
                   "rows_without_any_citation": nocite,
                   "rows_with_a_number_not_in_their_cited_groups": len(withmiss),
                   "numbers_total": sum(len(r["numbers"]) for r in rows),
                   "numbers_found": sum(len(r["numbers_found"]) for r in rows)},
        "rows_with_missing": withmiss,
        "verdict": {"every_row_cites_something": None, "all_numbers_traceable": None,
                    "reading": None},
    }
    v = out["verdict"]
    v["every_row_cites_something"] = not nocite
    v["all_numbers_traceable"] = not withmiss
    v["reading"] = (
        f"18 条 K 里，**{len(rows) - len(nocite)} 条带引用**"
        + (f"，无引用的是 {nocite}。" if nocite else "，**没有一条是无引用的**。")
        + f" 共 {out['counts']['numbers_total']} 个数字，其中 "
        f"**{out['counts']['numbers_found']} 个能在所引证据组里直接找到**。"
        + (f" {len(withmiss)} 条有数字不在其引用组里（见 rows_with_missing）——"
           f"这些要么是别处算出的派生量，要么是引用错组。" if withmiss else
           " **全部可追溯。**"))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "k_rows.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"K rows {len(rows)}  numbers {out['counts']['numbers_total']}  "
          f"found {out['counts']['numbers_found']}")
    print(f"no citation: {nocite}")
    print()
    for r in rows:
        flag = "" if not r["numbers_missing"] else f"  MISSING {r['numbers_missing']}"
        print(f"  K{r['k']:<2d} refs={r['refs']} nums={len(r['numbers'])} "
              f"found={len(r['numbers_found'])}{flag}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'k_rows.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
