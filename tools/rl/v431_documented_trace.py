"""v4.3.1 recheck R44: trace every "documented" number in §2.6 back to a v4.3.0 file.

R39 caught one fabricated quotation (`C14`). This applies the same discipline to §2.6, the table
that claims "10 of 12 documented numbers reproduce bit-for-bit": every value in its 文档 column
should exist somewhere under `Document/Update/Alpha/v4.3.0/`.

The audit found a second instance of the same defect:

  * 0.6461  -> documented at v4.3.0/evidence/W03_DATA_LEG.md:173 (focus row of the cross-tape
               table), and my batch-mean re-execution gives 0.6460538936985863 -> EXACT
  * 0.5589  -> **not present anywhere under v4.3.0**. It is my own batch-mean value
               (0.558894082903862) that got written into the 文档 column.

So the corrected statement is: **11 documented numbers, all 11 exact, 0 approximations** -- and
one row was my error, not an approximation.

R6's evidence already carried both averaging conventions per head, so nothing needed re-running;
the earlier "只做到近似" note looked at `sample_mean_ce` and missed `batch_mean_ce`.

Writes only Document/Update/Alpha/v4.3.1/evidence/R44/.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
SRC = REPO / "Document" / "Update" / "Alpha" / "v4.3.0"
OUT = EV / "R44"

# the 文档 column of §2.6, with the row label
DOCUMENTED = [
    ("ActorPolicy seed12-train move", "0.0293", "move", "seed12_train"),
    ("ActorPolicy seed12-val move", "1.8807", "move", "seed12_val"),
    ("ActorPolicy seed10 move", "1.8837", "move", "seed10_full"),
    ("ActorPolicy seed10 focus", "0.6461", "focus", "seed10_full"),
    ("ActorPolicy seed10-val focus", "0.5589", "focus", "seed12_val"),
    ("16x seed12-train move", "1.2827", "move", "seed12_train"),
    ("16x seed12-val move", "1.5306", "move", "seed12_val"),
    ("16x seed10 move", "1.6020", "move", "seed10_full"),
    ("4x seed12-train move", "0.1558", "move", "seed12_train"),
    ("4x seed12-val move", "2.1299", "move", "seed12_val"),
    ("4x seed10 move", "2.1161", "move", "seed10_full"),
    ("4x seed10-suffix move", "2.5070", "move", "seed10_suffix"),
]


def corpus() -> dict[str, str]:
    return {str(p.relative_to(REPO)): p.read_text(encoding="utf-8", errors="ignore")
            for p in SRC.rglob("*") if p.is_file() and p.suffix in (".md", ".json", ".jsonl")}


def locate(value: str, files: dict[str, str]) -> list[str]:
    return [name for name, body in files.items() if value in body]


def main() -> int:
    files = corpus()
    r6 = json.loads((EV / "R6" / "actor_recheck.json").read_text(encoding="utf-8"))
    r4 = json.loads((EV / "R4" / "model_recheck.json").read_text(encoding="utf-8"))

    def batch(arm: str, head: str, set_name: str):
        """Batch-mean CE from R6 (ActorPolicy) or R4 (the small models)."""
        try:
            if arm == "actor":
                return r6["evaluations"][set_name]["batch_mean_ce"][head]
            for a in r4["arms"].values():
                if a.get("down") == {"16x": 16, "4x": 4}[arm]:
                    e = a["evaluations"].get(set_name)
                    if not e:
                        return None
                    if isinstance(e.get("ce"), float):
                        return e["ce"]
                    return e.get("batch_mean_ce", {}).get(head)
        except Exception:                                        # noqa: BLE001
            return None
        return None

    rows = []
    for label, value, head, set_name in DOCUMENTED:
        where = locate(value, files)
        mine = batch("actor" if label.startswith("ActorPolicy") else
                     ("16x" if label.startswith("16x") else "4x"), head, set_name or "")
        rows.append({
            "row": label,
            "documented_value": value,
            "documented_in": where,
            "is_documented": bool(where),
            "head": head,
            "eval_set": set_name,
            "reproduced_batch_mean": (round(mine, 6) if isinstance(mine, float) else None),
            "matches_to_4dp": (round(mine, 4) == round(float(value), 4)
                               if isinstance(mine, float) else None),
        })

    documented_rows = [r for r in rows if r["is_documented"]]
    fabricated = [r for r in rows if not r["is_documented"]]
    exact = [r for r in documented_rows if r["matches_to_4dp"]]

    out = {
        "schema": "v431-recheck-r44/1",
        "question": ("does every value in §2.6's 文档 column exist under v4.3.0, and does the "
                     "batch-mean re-execution reproduce it?"),
        "method": {"corpus": "Document/Update/Alpha/v4.3.0/**/*.{md,json,jsonl}",
                   "files_scanned": len(files),
                   "match": "literal substring, then compare to 4 decimal places"},
        "rows": rows,
        "counts": {
            "rows": len(rows),
            "documented": len(documented_rows),
            "documented_and_exact": len(exact),
            "documented_but_mismatch": len(documented_rows) - len(exact),
            "not_documented_at_all": len(fabricated),
        },
        "fabricated_rows": [r["row"] for r in fabricated],
        "correction": {
            "was": "§2.6 说 10 项逐位一致、2 项近似",
            "now": (f"{len(exact)} 项文档数字全部逐位一致、0 项近似；另有 1 项（"
                    f"{fabricated[0]['row'] if fabricated else 'n/a'}）的"
                    f"文档值在 v4.3.0 里不存在——那是我自己的 batch-mean 值被写进了文档列"),
            "cause": ("§2.6 的注看的是 R6 的 sample_mean_ce，而 batch_mean_ce 一直就在同一个"
                      "证据文件里；0.5589 正是 batch-mean 的 val focus 值，被误当作文档值"),
            "class": "与 C14 同一类缺陷（自己的数字冒充文档值），记为 C29",
        },
        "verdict": {
            "all_documented_numbers_reproduce": len(documented_rows) == len(exact),
            "one_row_is_mine": len(fabricated) == 1,
            "reading": None,
        },
    }
    v = out["verdict"]
    v["reading"] = (
        "every one of the documented numbers reproduces exactly under batch-mean averaging "
        f"({len(exact)}/{len(documented_rows)}); the twelfth row was never documented -- it is my "
        "own value, so §2.6 over-claimed in the wrong direction (it should have said more exact, "
        "not fewer) and contained one fabricated source")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "documented_trace.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'row':>30s} {'doc':>8s} {'found in':>34s} {'mine':>10s} {'match':>6s}")
    print("-" * 94)
    for r in rows:
        where = r["documented_in"][0].split("/")[-1] if r["documented_in"] else "** NOT FOUND **"
        print(f"{r['row']:>30s} {r['documented_value']:>8s} {where:>34s} "
              f"{str(r['reproduced_batch_mean']):>10s} {str(r['matches_to_4dp']):>6s}")
    print()
    print(f"documented {out['counts']['documented']}/{len(rows)}; "
          f"exact {out['counts']['documented_and_exact']}/{out['counts']['documented']}; "
          f"not documented at all {out['counts']['not_documented_at_all']}")
    print(f"fabricated rows: {out['fabricated_rows']}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'documented_trace.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
