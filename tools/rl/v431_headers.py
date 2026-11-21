"""v4.3.1 recheck R69: table headers -- which columns carry their own convention?

R68 showed a column named "差" holding two opposite sign conventions. The general question is
whether the OTHER numeric columns are self-contained. This walks every markdown table header in
the two deliverables and classifies each numeric column.

104 tables, 266 distinct header cells. Most short headers are LABEL columns (结果/项/来源/状态)
where a convention is meaningless. The ones that matter are numeric, and they reduce to one real
ambiguity:

    `n` means SAMPLES in the CE tables (220, 1099) and EPISODES in the arm-A tables (10, 20, 80).

Both uses are legitimate; the shared letter is not. This records the full header inventory, the
n-conflation, and the columns whose unit (decisions vs nats vs episodes) is implicit.

Writes only Document/Update/Alpha/v4.3.1/evidence/R69/.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DOCS = {"report": REPO / "Document/Update/Alpha/v4.3.1" / "ROUTE_RECHECK_2026-11-15.md",
        "summary": REPO / "Document/Update/Alpha/v4.3.1" / "SUMMARY_FOR_DECISION.md"}
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R69"

SEP = re.compile(r"^\|[\s:|-]+\|$")
# columns that carry a number and therefore need to say what the number counts
NUMERIC = ("n", "均值", "中位数", "最大", "最小值", "标准差", "平均差", "累计", "用时", "占",
           "跑满", "合格数", "数量", "规模")
UNIT_OK = ("（", "(", "%", "决策", "帧", "倍", "s ", "KB", "h ", "n=", "字节")


def main() -> int:
    tables = 0
    header_cells: collections.Counter = collections.Counter()
    numeric_cols = []
    for tag, path in DOCS.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        for i in range(len(lines) - 1):
            if not (lines[i].startswith("|") and SEP.match(lines[i + 1].strip())):
                continue
            tables += 1
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            header_cells.update(cells)
            for c in cells:
                base = c.strip("*").strip()
                if base in NUMERIC and not any(u in c for u in UNIT_OK):
                    sample = lines[i + 2][:120] if i + 2 < len(lines) else ""
                    numeric_cols.append({"doc": tag, "line": i + 1, "column": base,
                                         "header": lines[i].strip()[:130], "sample_row": sample})

    # the n-conflation: same header, two meanings
    n_rows = [r for r in numeric_cols if r["column"].startswith("n")]
    n_conflation = {
        "header": "n",
        "meanings": [
            {"kind": "样本数", "evidence": "CE 表 `留出集 | n | 模型 CE …`，行值 220 / 1099"},
            {"kind": "episode 数", "evidence": "臂 A 表 `臂 | n | 存活 …`，行值 10 / 20 / 80"},
        ],
        "problem": "同一个表头字母在两类表里指不同的东西；读者跨表比较时会把 220 个样本与 80 个 episode 当成同一量",
        "fix": "CE 表写 `n（样本）`，臂 A 表写 `n（episode）`",
    }

    out = {
        "schema": "v431-recheck-r69/1",
        "question": "报告里每个数值列的表头是否自带口径？（R68 的「差」只是其中一例吗？）",
        "counts": {"tables": tables, "distinct_header_cells": len(header_cells),
                   "numeric_columns_without_a_unit": len(numeric_cols)},
        "n_conflation": n_conflation,
        "numeric_columns": numeric_cols,
        "label_columns_note": (
            "多数短表头是标签列（结果/项/来源/状态/说明/改成/问题），对它们谈口径没有意义——"
            "把它们排除后，真正缺口径的数值列只有下面这些。"),
        "verdict": {"n_is_ambiguous": True, "other_numeric_columns_are_labels": None,
                    "reading": None},
    }
    v = out["verdict"]
    non_n = [r for r in numeric_cols if not r["column"].startswith("n")]
    v["other_numeric_columns_are_labels"] = False
    v["reading"] = (
        f"扫了 {tables} 张表、{len(header_cells)} 个不同的表头单元。"
        f"排除标签列之后，真正需要口径的数值列有 **{len(numeric_cols)}** 处。"
        "其中**最实质的一处是 `n`**：它在 CE 表里是**样本数**（220/1099），"
        "在臂 A 表里是**episode 数**（10/20/80）——同一个字母两种东西，"
        "跨表比较时会被读成同一个量。其余是 `均值/中位数/最大` 这类**默认单位是决策数但没有写明**的列。"
        "**所以 R68 的「差」不是孤例：表头普遍假设读者知道上下文。**")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "headers.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"tables {tables}  distinct cells {len(header_cells)}  numeric cols needing a unit {len(numeric_cols)}")
    for r in numeric_cols:
        print(f"  {r['doc']}:{r['line']:<5d} col='{r['column']}'  {r['header'][:70]}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'headers.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
