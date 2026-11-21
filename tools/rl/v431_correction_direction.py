"""v4.3.1 recheck R54: do my own corrections all point the same way?

Thirty-six corrections across eighteen rounds. If every one of them made arm A look worse, that
would be a sign to distrust the review: a critic who only ever moves in one direction is not
auditing, they are building a case.

This classifies each C row on two axes:

  subject    original_report | my_process | my_method | my_result
  direction  on ARM A's evaluation: helps | hurts | neutral

"helps" means the correction made arm A look BETTER than my earlier text claimed (i.e. I had
overstated a negative). That is the direction a biased-against-arm-A reviewer would never produce.

The classification is a hand judgement recorded row by row so a reader can disagree with any line;
the tally is then mechanical.

Writes only Document/Update/Alpha/v4.3.1/evidence/R54/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R54"

# id: (subject, direction_on_arm_a, one-line reason)
ROWS = {
    "C1": ("original_report", "neutral", "880/219 从未存在 → 879/220"),
    "C2": ("original_report", "neutral", "0.8537 不是 CE；正确基线 1.6586"),
    "C3": ("original_report", "neutral", "基线用了评测标签"),
    "C4": ("original_report", "helps_bc", "“全部不优”太绝对 → 16× 在 CE 上优（帮 BC，不帮臂 A）"),
    "C5": ("original_report", "neutral", "准确率上原说法对"),
    "C6": ("original_report", "neutral", "219 是动作层不是输入层"),
    "C7": ("original_report", "neutral", "近重复对，不是原文"),
    "C8": ("original_report", "neutral", "共同前缀 225"),
    "C9": ("original_report", "neutral", "2.5070 溯源到 cross_tape_honest.py"),
    "C10": ("original_report", "neutral", "≥9/10 的断言不成立"),
    "C11": ("original_report", "neutral", "键历史相同不构成输入重复"),
    "C12": ("original_report", "neutral", "“窗口内无死亡”的口径"),
    "C13": ("original_report", "neutral", "A1 已提供现成资产"),
    "C14": ("my_process", "neutral", "我的数字冒充文档值"),
    "C15": ("my_method", "neutral", "砍掉 C4 是错的"),
    "C16": ("my_method", "neutral", "state 0 在 180 帧口径下不合格"),
    "C17": ("my_method", "neutral", "“便宜 5 倍”不成立"),
    "C18": ("my_method", "neutral", "后期密集窗口可执行（确认）"),
    "C19": ("my_method", "neutral", "3 s → 10 s"),
    "C20": ("my_method", "neutral", "PARK 本就存在"),
    "C21": ("my_method", "neutral", "PARK 是结构性拒绝"),
    "C22": ("my_method", "hurts", "C-fair 太弱，真门是恒定动作基线"),
    "C23": ("my_result", "helps", "“臂 A 没有学习信号”是错的：它有"),
    "C24": ("my_result", "helps", "“显著差于随机”过强：不显著"),
    "C25": ("my_method", "neutral", "STEP 批量不提速"),
    "C26": ("my_method", "neutral", "客户端已无余量"),
    "C27": ("my_result", "hurts", "“基线 0 次存活”是小样本假象：C-fair 也活过一次"),
    "C28": ("my_process", "neutral", "§6.2 原说法列可混入我的数字"),
    "C29": ("my_process", "neutral", "0.5589 不是文档值"),
    "C30": ("my_process", "neutral", "§3 依据夸大"),
    "C31": ("my_process", "neutral", "25 项应为 24 项"),
    "C32": ("my_method", "hurts", "停止条件 3 条不可执行"),
    "C33": ("my_method", "hurts", "漏掉恒定动作池基线：臂 A 不如固定动作"),
    "C34": ("my_method", "hurts", "C-const-best 是神谕；可部署版更严格"),
    "C35": ("my_method", "hurts", "“6 倍优势”只在均值成立"),
    "C36": ("my_method", "hurts", "判定本身依赖判据"),
}


def main() -> int:
    rows = [{"id": k, "subject": v[0], "direction": v[1], "reason": v[2]}
            for k, v in ROWS.items()]
    assert len(rows) == 36, len(rows)

    by_dir: dict[str, list[str]] = {}
    by_subject: dict[str, dict[str, int]] = {}
    for r in rows:
        by_dir.setdefault(r["direction"], []).append(r["id"])
        by_subject.setdefault(r["subject"], {}).setdefault(r["direction"], 0)
        by_subject[r["subject"]][r["direction"]] += 1

    out = {
        "schema": "v431-recheck-r54/1",
        "question": ("我的 36 处自更正是都朝同一方向（对臂 A 不利），还是有来有回？"),
        "rows": rows,
        "counts": {k: len(v) for k, v in sorted(by_dir.items())},
        "ids_by_direction": by_dir,
        "by_subject": by_subject,
        "structural_note": (
            "加基线只能让候选者更难过关：每新增一个基线/判据，被测对象的排名只会下降或不变。"
            "所以“hurts”条数多于“helps”有一部分是**结构性的**，不是判断倾向。"),
        "verdict": {"one_directional": None, "reading": None},
    }
    v = out["verdict"]
    hurts, helps = out["counts"].get("hurts", 0), out["counts"].get("helps", 0)
    v["one_directional"] = helps == 0
    v["reading"] = (
        f"36 处自更正里 **{helps} 处让臂 A 看起来更好**（{by_dir.get('helps', [])}）、"
        f"**{hurts} 处让它更差**、{out['counts'].get('neutral', 0)} 处中性。"
        f"所以**不是单向的**：`C23`（我曾说\"臂 A 没有学习信号\"）与 `C24`（我曾说\"显著差于随机\"）"
        f"都是**我把负面说过头了**，后来被我自己的重算纠正回来。"
        f"（另有 `C4` 一处 `helps_bc`：它让 **BC** 看起来更好，已**刻意排除在臂 A 这条轴之外**——"
        f"两个候选者不能混在一个方向里统计。）"
        f"同时必须承认一个**结构性不对称**：每新增一个基线或判据，被测者只会更难过关，"
        f"所以\"更差\"的条数偏多是**方法本身的性质**，不是我有意挑刺。"
        if helps > 0 else
        "36 处自更正**全部朝同一方向**，这本身是不可信的信号，必须复查。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "correction_direction.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'id':>4s} {'subject':>16s} {'direction':>9s}  reason")
    print("-" * 92)
    for r in rows:
        if r["direction"] != "neutral":
            print(f"{r['id']:>4s} {r['subject']:>16s} {r['direction']:>9s}  {r['reason']}")
    print("-" * 92)
    for k, v2 in sorted(out["counts"].items()):
        print(f"  {k:>8s} : {v2}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'correction_direction.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
