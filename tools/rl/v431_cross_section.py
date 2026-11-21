"""v4.3.1 recheck R59: cross-section logical conflicts that no assertion can catch.

1,218 assertions check numbers and structure. None of them checks whether the ARGUMENT is
self-consistent across sections. This is the record of the conflicts found by reading the four
deliverables end to end.

The conflict types follow the audit brief:

  A  the same fact described differently in different sections
  B  a conclusion stronger than its evidence
  C  two criteria conflated as one "better"
  D  a claim later refuted but not annotated where it is made
  E  circular or self-referential support
  F  mixed measurement conventions inside one comparison

Every entry records where it was, which type, what was wrong, and what was changed. The point is
not that the report is clean -- it is that these are the classes the assertion suite is blind to,
and they are listed so the next reviewer knows where to look.

Writes only Document/Update/Alpha/v4.3.1/evidence/R59/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R59"

FOUND = [
    {"id": 1, "where": "§0 结语", "type": "D",
     "what": "结语仍写\"文档里 10 个 CE 数字\"，而 §2.6/`R44` 已把口径改成 11/11",
     "fix": "改为 11 个，并注明 `R44` 逐个回源"},
    {"id": 2, "where": "§0 结语", "type": "D",
     "what": "结语写\"改用 A1 之后最小实验从 22 h 降到 4–5 h\"，而同一张表第 9 行已判该预算**不成立**",
     "fix": "改为\"这个预算是我的推断、已被本表第 9 行否决\"，并给出真实路径与实际 13.5 h"},
    {"id": 3, "where": "§4.2r 标题", "type": "D",
     "what": "标题断言\"在留出磁带上比随机还差\"（n=20），`R38`（n=80）已把它弱化为不显著",
     "fix": "标题标 n=20，并在节首加前向说明：数字有效、模态被削弱（见 `C24`）"},
    {"id": 4, "where": "§4.2w 标题与首句", "type": "A",
     "what": "写\"28 条更正\"，而更正表已增到 37 条；`R39` 审计现在覆盖 37 行",
     "fix": "标题去掉数字；正文写\"当时 28 条，现在 37 条\""},
    {"id": 5, "where": "§5.5 `C-const-best` 行", "type": "F",
     "what": "仍写\"dev 最强恒定动作 245.5–272.0\"，`R51` 已区分神谕（逐窗口最大，均值 258.8）与可部署（172.4）",
     "fix": "拆成神谕 258.8（单窗口最大 272.0）+ 可部署 172.4 + 池均值 111.7"},
    {"id": 6, "where": "§5.5 任务成功率判据行", "type": "B",
     "what": "写\"存活率在所有臂与基线上都是 0\"，而 `R38` 已测到 `C-fair` **1/80**（`C27`）",
     "fix": "限定为\"n=20 时都是 0\"，并写明扩到 n=80 后 `C-fair` 也跑成了一次"},
    {"id": 7, "where": "§7.1 成本表", "type": "A",
     "what": "表只覆盖 `R1`–`R38`，`R39`–`R57` 的 19 组离线审计不在表内，读者无法从中看出\"13.5 h 里没有新增引擎\"",
     "fix": "加一行：19 组 + 1 流程步骤、引擎时间 0、1,218 条断言；并在成本句里显式写出"},
]

# findings contributed by the delegated full-read audit
DELEGATED = json.loads((OUT.parent / "R59" / "delegated.json").read_text(encoding="utf-8")) \
    if (OUT / "delegated.json").is_file() else None


def main() -> int:
    by_type: dict[str, int] = {}
    for f in FOUND:
        by_type[f["type"]] = by_type.get(f["type"], 0) + 1

    out = {
        "schema": "v431-recheck-r59/1",
        "question": ("跨节逻辑冲突：1,218 条断言覆盖不到的那一类问题"),
        "scope": ["§0", "§4.2r", "§4.2w", "§5.5", "§7.1"],
        "found_by_self": FOUND,
        "found_by_delegated_read": DELEGATED,
        "counts": {"self": len(FOUND), "by_type_self": by_type,
                   "delegated_total": DELEGATED["counts"]["total"] if DELEGATED else None,
                   "delegated_by_type": ({"A": DELEGATED["counts"]["A"],
                                          "B": DELEGATED["counts"]["B"],
                                          "C": DELEGATED["counts"]["C"],
                                          "D": DELEGATED["counts"]["D"],
                                          "E": DELEGATED["counts"]["E"],
                                          "F": DELEGATED["counts"]["F"]})
                   if DELEGATED else None},
        "verdict": {
            "assertion_suite_is_blind_to": [f["what"] for f in FOUND],
            "reading": None,
        },
    }
    v = out["verdict"]
    v["reading"] = (
        f"自读找到 {len(FOUND)} 处跨节冲突，全部属 D/A/B/F 型："
        f"**后文修正了前文的结论，而前文没有回头标注**（`D` 三处）、"
        f"**计数与口径随轮次变了而文字没变**（`A` 两处）、"
        f"**绝对化表述强于证据**（`B` 一处）、**同一比较里混用神谕与可部署口径**（`F` 一处）。"
        "全部已就地修正。这一类问题**在结构上无法由断言发现**："
        "断言能查\"这个数字是否等于证据\"，查不了\"这句话是否还成立\"。"
        + (f" 另由一次**全文通读**（委派，逐条回原文复核后）找到 **{DELEGATED['counts']['total']}** 处，"
           f"按类型 A={DELEGATED['counts']['A']} B={DELEGATED['counts']['B']} "
           f"C={DELEGATED['counts']['C']} D={DELEGATED['counts']['D']} "
           f"E={DELEGATED['counts']['E']} F={DELEGATED['counts']['F']}；"
           "其中最要紧的几处（摘要 3/4 vs 2/4、S2 三口径、预注册冻结快照、172.4 的 n=4）已一并修正。"
           if DELEGATED else ""))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cross_section.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'id':>3s} {'where':>12s} {'type':>5s}  what")
    print("-" * 100)
    for f in FOUND:
        print(f"{f['id']:3d} {f['where']:>12s} {f['type']:>5s}  {f['what'][:70]}")
    print()
    print(f"self-found {len(FOUND)}; by type {by_type}")
    print(f"delegated  {out['counts']['delegated_total']}  {out['counts']['delegated_by_type']}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'cross_section.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
