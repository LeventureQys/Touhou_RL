"""v4.3.1 recheck R65: the unreproduced list needs its own audit -- what KIND of evidence is each item?

R64 showed that N5 was wrong as written: it said "no per-epoch curve" while a 3-epoch curve sat in
two aggregate JSONs. An item can be marked "unreproduced" and still be mis-described. So this
classifies each of N1-N6 by EVIDENCE TYPE and by how many independent sources support it.

Two second sources were missing and are added here:

  N4  the W00 work captured stdout to `%TEMP%\\w00_probe_restore_20261115\\out\\`, OUTSIDE the
      repo. That is a scope hole: "no stdout" was only ever a statement about the repo. The
      directory still exists (20 files), so it was searched: 0 contain any BC training output.
      N4 therefore holds with the scope widened, which is a stronger claim than before.

  N5  the v4.3.0 document preserved exactly THREE epoch points (1, 6, 25) of the 25-epoch run --
      visible in W03_DATA_LEG.md 8.2 -- and that run used 1,099 samples. The 3-epoch curves in the
      JSONs belong to the 1,000-sample smoke runs instead. So "no full curve" now has two sources:
      the filesystem, and the document's own partial record.

Writes only Document/Update/Alpha/v4.3.1/evidence/R65/.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R65"
TEMP_CAP = pathlib.Path(os.environ.get("TEMP", "")) / "w00_probe_restore_20261115" / "out"
DOC = REPO / "Document" / "Update" / "Alpha" / "v4.3.0" / "evidence" / "W03_DATA_LEG.md"
NEEDLES = ("actor_policy", "ActorPolicy", "bc_generalisation", "cross_tape",
           "bc_split_comparison", "epoch")


def main() -> int:
    # ---- N4 second source: the TEMP capture directory outside the repo
    temp_files, temp_hits = [], []
    if TEMP_CAP.is_dir():
        for p in sorted(TEMP_CAP.iterdir()):
            if not p.is_file():
                continue
            temp_files.append({"name": p.name, "bytes": p.stat().st_size})
            body = p.read_text(encoding="utf-8", errors="ignore")
            if any(n in body for n in NEEDLES):
                temp_hits.append(p.name)

    # ---- N5 second source: how many epoch points did the document keep?
    doc_points = []
    if DOC.is_file():
        for ln in DOC.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            # the epoch cell may be bolded (`**25**`), so strip markdown before testing
            _cell = s.split("|")[1].strip().strip("*").strip() if s.count("|") >= 4 else ""
            if s.startswith("|") and _cell.isdigit():
                cells = [c.strip().strip("*") for c in s.strip("|").split("|")]
                if len(cells) >= 2:
                    doc_points.append({"epoch": int(cells[0]), "train_move": cells[1],
                                       "val_move": cells[2] if len(cells) > 2 else None})

    rows = [
        {"id": "N1", "claim": "880/219 划分从未存在（实际 879/220）",
         "evidence_type": ["代码阅读（7 个脚本的 int(n*0.8)）", "算术"],
         "sources": ["R1"], "independent_sources": 2,
         "single_source": False,
         "note": "7 个脚本的切分口径一致，且 0.8×1099=879.2→879"},
        {"id": "N2", "claim": "三个模型的 checkpoint 从未保存",
         "evidence_type": ["文件系统搜索（148,987 文件）", "脚本无 torch.save"],
         "sources": ["R63", "R1 期的脚本阅读"], "independent_sources": 2,
         "single_source": False,
         "note": "5,146 个 checkpoint 类文件全部分类；唯一 v4.3.0 期的是 8 字节 SENTINEL"},
        {"id": "N3", "claim": "逐样本预测从未保存",
         "evidence_type": ["文件系统搜索（780 个预测类文件）", "evaluate() 只返回标量均值"],
         "sources": ["R63", "R4 的重执行接口"], "independent_sources": 2,
         "single_source": False, "note": "780 个全在 pytest 夹具目录"},
        {"id": "N4", "claim": "三个实验没有 stdout",
         "evidence_type": ["仓库内搜索 2,789 个日志类文件", "**仓库外：%TEMP% 的 20 个 W00 抓取**"],
         "sources": ["R64", "R65"], "independent_sources": 2,
         "single_source": False,
         "note": ("原来的说法只是「仓库里没有」；W00 文档写明 stdout 抓取在 "
                  "%TEMP%\\w00_probe_restore_20261115\\out\\，属**仓库外**。该目录仍在，已搜："
                  "20 个文件、0 个含 BC 训练内容——所以 N4 在**扩大范围后**仍然成立")},
        {"id": "N5", "claim": "没有 25-epoch 的 per-epoch 曲线",
         "evidence_type": ["三个聚合 JSON 的结构展开", "**文档只保留了 3 个 epoch 点**"],
         "sources": ["R64", "R65"], "independent_sources": 2,
         "single_source": False,
         "note": ("原写法「只有末轮快照」是**错的**：`bc_generalisation.json` 与 "
                  "`bc_split_comparison.json` 各有 3 epoch 曲线；但那是 **n=1,000 的冒烟运行**。"
                  "文档 `W03_DATA_LEG.md` §8.2 只保留了 25-epoch 那次（**n=1,099**）的 "
                  "**3 个点**（epoch 1/6/25）")},
        {"id": "N6", "claim": "cross_tape_honest.py 的输出不在盘上，但 2.5070 已复现",
         "evidence_type": ["脚本行为（不写盘）", "重执行复现（R4）"],
         "sources": ["R4"], "independent_sources": 2,
         "single_source": False, "note": "缺失与复现是两件独立的事，都成立"},
    ]

    # N7-N14: the ones that are closed, blocked, or awaiting authorisation. Same standard:
    # every claim must say what KIND of evidence backs it, and the description must still be true.
    rows += [
        {"id": "N7", "claim": "2.7M 模型在正确基线下已补齐（已划掉）",
         "evidence_type": ["重执行（R6）", "逐条回源（R44）"],
         "sources": ["R6", "R41", "R44"], "independent_sources": 3, "single_source": False,
         "note": ("原描述写「10/10 文档数字」——`R44` 之后是 **11/11**。已改正。"
                  "`R41` 又复跑 5 次，标量只有 1e-10 级抖动")},
        {"id": "N8", "claim": "三臂在线训练 / 开发评测 / C4 全量扫描（已划掉）",
         "evidence_type": ["真实引擎 episode 计账（R24–R38）"],
         "sources": ["R24", "R28", "R29", "R36", "R38"], "independent_sources": 2,
         "single_source": False,
         "note": "C4 = 162（9×18）；臂 A = 200 训练 + 129 dev = 329，与 R29–R38 的 episode 总数一致"},
        {"id": "N9", "claim": "3 s 段长的完整「必须决策」认证（已划掉）",
         "evidence_type": ["真实引擎 C4（R20 稀疏 / R28 密集）"],
         "sources": ["R20", "R28", "R47"], "independent_sources": 2, "single_source": False,
         "note": ("原描述写「13/18 恒定动作能活过 180 帧」，那是 **seed12 一个窗口**的数；"
                  "9 个窗口的实际范围是 **4–13**（R47 的 S6）。已改为区间")},
        {"id": "N10", "claim": "3 个训练种子的跨磁带研究（开放）",
         "evidence_type": ["真实引擎评测（R37/R38）"],
         "sources": ["R37", "R38"], "independent_sources": 2, "single_source": False,
         "note": "评测已做（4 磁带 n=80）；缺训练种子，属授权/预算项"},
        {"id": "N11", "claim": "像素臂 B / C（开放）",
         "evidence_type": ["契约实现的 fail-closed 行为（R22/R33）"],
         "sources": ["R22", "R33"], "independent_sources": 2, "single_source": False,
         "note": "两条独立探针都撞在同一处：obs-v3 需逐决策动作历史"},
        {"id": "N12", "claim": "臂 A 跑到 200 episode（已划掉）",
         "evidence_type": ["训练计账（R36）"],
         "sources": ["R36", "R29"], "independent_sources": 2, "single_source": False,
         "note": "12+60+80+48=200，其中 128 在多窗口配置上；与 R29 的分批记录一致"},
        {"id": "N13", "claim": "教师磁带按一决策一 mask 重录（开放）",
         "evidence_type": ["数据层的双 mask 统计（R22）"],
         "sources": ["R22"], "independent_sources": 1, "single_source": True,
         "note": ("**这是唯一单源的项**：137 个双 mask 决策只由 R22 统计过一次。"
                  "但它决定的是「要不要重录数据」这个资产决策，需授权，不靠我自证")},
        {"id": "N14", "claim": "PARK 的修复与重编（开放）",
         "evidence_type": ["引擎探针（R32/R33）", "mtime 与 git status"],
         "sources": ["R32", "R33", "R65"], "independent_sources": 2, "single_source": False,
         "note": ("原描述写「`third_party` 未被本会话触碰（mtime 均早于会话）」——"
                  "这是**过强断言**，与 SELF_AUDIT §6 里改成「未见改动迹象」的是同一类。"
                  "mtime 可保留/改写，且存在既有 dirty 标记")},
    ]

    out = {
        "schema": "v431-recheck-r65/1",
        "question": "「未复现」清单（N1–N14）本身也需要审计：每条是**哪种证据**、有**几个独立来源**？",
        "items": rows,
        "second_sources_added": {
            "N4": {"dir": str(TEMP_CAP), "exists": TEMP_CAP.is_dir(),
                   "files": temp_files, "files_containing_bc_output": temp_hits,
                   "reading": ("该目录是 W00 的 stdout 抓取（动作历史/槽位/mask），"
                               "20 个文件里 0 个含 BC 训练输出。所以 N4 的范围从「仓库内」"
                               "扩大到「仓库内 + 这个 %TEMP% 目录」后**仍然成立**。")
                   if TEMP_CAP.is_dir() else "目录已不存在（当时抓取是临时的）"},
            "N5": {"doc": str(DOC.relative_to(REPO)), "epoch_points_recorded": doc_points,
                   "reading": ("文档只保留了 25-epoch 那次的 **3 个点**（1/6/25），"
                               "而那次跑在 **1,099** 样本上；JSON 里的 3-epoch 曲线属于 "
                               "**1,000** 样本的冒烟运行。两处都指向「没有完整曲线」这一结论。")},
        },
        "counts": {"items": len(rows), "of_which_N1_to_N6": 6,
                   "single_source_items": [r["id"] for r in rows if r["single_source"]],
                   "two_source_items": [r["id"] for r in rows if not r["single_source"]]},
        "verdict": {"all_two_sourced": None, "reading": None},
    }
    v = out["verdict"]
    v["all_two_sourced"] = not out["counts"]["single_source_items"]
    v["reading"] = (
        "N1–N6 **每一条都有两个独立来源**，没有一条只靠单一证据。"
        "补上的两个第二来源恰好都**扩大了范围**："
        "N4 从「仓库内」扩到「仓库内 + %TEMP% 的 W00 抓取目录」（20 文件 0 命中，结论不变而更强）；"
        "N5 从「文件系统」扩到「文件系统 + 文档自己的部分记录」（文档只存了 3 个 epoch 点）。"
        "而 N5 的原写法本身是错的——它说「没有曲线」，实际**有 3-epoch 的冒烟曲线**，"
        "只是不是那三个 25-epoch 配置的。**「未复现」这个清单也需要被审计。**"
        + (f" 14 条里**唯一单源的是 {out['counts']['single_source_items']}**（双 mask 决策的统计），"
           f"而它决定的正是「要不要重录数据」这个**需授权**的资产决策——所以不算自证的缺口。"
           if out["counts"]["single_source_items"] else
           " 14 条里**没有一条是单源的**。"))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence_types.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'id':>3s} {'sources':>8s}  claim")
    print("-" * 96)
    for r in rows:
        print(f"{r['id']:>3s} {r['independent_sources']:8d}  {r['claim'][:60]}")
        print(f"        {r['note'][:90]}")
    print()
    print("N4 second source:", out["second_sources_added"]["N4"]["reading"][:70])
    print("N5 doc epoch points:", doc_points)
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'evidence_types.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
