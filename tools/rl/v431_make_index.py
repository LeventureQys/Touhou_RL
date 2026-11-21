"""v4.3.1 recheck R62: generate the handover index from what is actually on disk.

The four deliverables grew to ~290 KB across 61 evidence groups. The entry point was buried in a
line of SUMMARY_FOR_DECISION.md, which is fine for a reader who already knows the layout and
useless for one who does not.

This generates `README_INDEX.md` instead of hand-writing it: every row is derived from the
evidence directories and the documents themselves, so the index cannot drift from the contents.
Re-run it after adding evidence and the index updates.

Writes Document/Update/Alpha/v4.3.1/README_INDEX.md and evidence/R62/index.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DOC = REPO / "Document" / "Update" / "Alpha" / "v4.3.1"
EV = DOC / "evidence"
OUT_JSON = EV / "R62" / "index.json"



# Curated one-liners. A group without an entry FAILS the script, so adding evidence forces a
# description -- an index that silently degrades to "v431-recheck-r3/1" is worse than no index.
DESCRIPTIONS = {
    1: "重算划分与正确基线（训练集频率 MLE/Laplace），无模型",
    2: "跨磁带重叠：动作前缀相同不等于像素相同",
    3: "重复结构：盘内重复行、相邻样本差、SE 重复",
    4: "三配置重执行：CE / 准确率 / 逐类（新身份，非复现）",
    5: "显著性：配对 bootstrap 4,000 次",
    6: "ActorPolicy 重执行 + 剥离近重复前缀的翻负实验",
    7: "片段候选与成本排序（离线筛选）",
    8: "磁带哈希链与后期密集窗口定位",
    10: "A1 资产核验（byte-exact，sha256 pin）",
    11: "A1 安全集证据：184 个 proper 状态",
    12: "状态成本画像（每状态每秒代价）",
    14: "音频 cfg 审计：105 个 cfg 全部可出声",
    15: "候选空间：18 个 (focus,move)，mask 注入性",
    16: "判定可靠性：Wilson / bootstrap / Fisher",
    18: "A1 state 0 的 18 臂复现，36,180 次前缀哈希 0 不符",
    19: "3 个策略驱动 episode（稀疏早期状态）",
    20: "C4 @3 s 在 4 个稀疏状态上：0/4 合格 → S1b 触发",
    21: "A1 池上限：学生轨迹全部止于 first_death",
    22: "replay() 阻断：137 个双 mask 决策，首个 frame 94",
    23: "手动逐帧重放到达 frame 8,664（密集区间可达性）",
    24: "C4 @10 s 单窗口：13/18 恒动作活过 180 帧",
    25: "三窗口复跑：2/3 合格，臂 A 观测定义有错",
    26: "HAZARDS 可用性与字段（env-role 可答）",
    27: "臂 A 观测特征 106 维与自测 18/18",
    28: "9 窗口筛选：8 合格，train/dev 各 4（冻结切分）",
    29: "臂 A 首轮训练曲线",
    30: "前缀 run 结构 + STEP 批量微基准（0.992×）",
    31: "C-fair 基线画像（逐决策均匀随机）",
    32: "PARK/SPAWN/CHILD 探针：结构上关着",
    33: "PARK 在 5 个不同帧上重试：同一组拒绝原因",
    34: "单磁带→跨磁带迁移（n=20）",
    35: "多窗口训练（4 磁带）与首个不跨窗的更正",
    36: "200 episode 预算与训练后段趋势",
    37: "四窗口 n=40 评测",
    38: "n=80 评测：R37 的乐观结论变弱",
    39: "引文回源：更正表第一列逐字/数字定位",
    41: "早期证据 R1–R6 重跑（发现我自己的假快照）",
    42: "逐窗口方差与「窗口宽阔度」",
    43: "平滑敏感性：MLE vs Laplace（4.34e-4）",
    44: "§2.6 文档值回源：11/11 精确，1 行非文档",
    45: "全篇「归因数字」扫描（C14/C29/C30 那类）",
    46: "证据引用完整性：45 引用 / 0 缺失 / 0 孤儿",
    47: "停止条件可判定性：8 条里 2 触发 / 2 不可用",
    48: "段长上下文纪律：合格结论必须带段长",
    49: "存活率上下文纪律：比率必须带 n 与门槛",
    50: "均值截断 + 漏掉的「固定一个动作」基线",
    51: "基线阶梯：神谕 258.8 / 可部署 172.4 / 池 111.7",
    52: "统计量翻转：均值 +14.4 / 中位数 −15.0",
    53: "判据敏感性：8 条可算判据 2 支持 / 5 反对",
    54: "36 处自更正的方向统计（2 帮臂 A / 7 损）",
    55: "反向检验：最有利判据 6/8 支持，单侧 p=0.0584",
    56: "三份交付物对臂 A 的结论一致性",
    57: "全套离线审计：22 脚本 / 幂等",
    59: "跨节逻辑冲突通读：自读 7 + 委派 24",
    60: "结论强于证据专项：11 处（10 确认）",
    61: "判据混用专项：3 处（含「种子」两义）",
    62: "本索引的生成记录",
    63: "产物缺席搜索：148,987 文件里没有一个是训练好的 BC 权重",
    64: "N4/N5 穷尽：0 个日志提到 BC；但有 3 epoch 曲线，且两个聚合文件是 n=1,000",
    65: "未复现清单的证据类型：N1–N6 各 2 个独立来源，N4 范围扩到 %TEMP%",
    66: "切分脚本逐条清点：5 个 0.8 + 1 个固定 200，两种语料（1,099 与 1,000）",
    67: "两条全称断言：18 个恒定动作成立；一致率 0.448/0.463/0.456 无法复现",
    68: "两张表的「差」列符号相反：模型−基线 vs 基线−模型，同一对显示为 −0.128 与 +0.1280",
    69: "104 张表的表头审计：n 在 CE 表是样本、在臂 A 表是 episode；25 处数值列补齐单位",
    70: "107 个百分比：9 个自证全成立；我的第一版检查器按距离配对报了 25 个假错",
    71: "成本链 9/9 可重算；70.4 s 是满长上界（实测 17.8）；抓到 90→72",
    72: "K 表引用核验：6 条无引用、K9 仍带已证伪的数、K3 引用错组（seed11 在 R4）",
    73: "按值反查残留：63 处全部人工分类、0 处真残留；假阳性是主题/正确用法/子串/自身说明",
    74: "臂 B/C 的阻断其实在 replay()：到不了起始帧，不是不能提交两个 mask",
    75: "存活率检验的功效与粒度：163/臂 → 84%@7.5%；实际 α 0.1%；p=0.0584 不是\"差一点\"",
    76: "判据校准：Fisher 0.0584（α 实际 0.126%）vs Boschloo ≤0.0343 ⇒ 未通过判据 ≠ 无效应",
}
PROCESS_STEPS = {13: "流程步骤：报告自检（无独立文件）",
                 17: "流程步骤：判据算术自测（无独立文件）",
                 40: "流程步骤：未复现/未执行清单复核（无独立文件）"}


def describe(group_dir: pathlib.Path) -> str:
    """One line for an evidence group, taken from its own JSON."""
    jsons = sorted(group_dir.glob("*.json"))
    for p in jsons:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                        # noqa: BLE001
            continue
        if isinstance(d, dict):
            for key in ("question", "note", "what", "schema"):
                v = d.get(key)
                if isinstance(v, str) and len(v) > 8:
                    return v.replace("\n", " ")[:110]
    files = [p.name for p in group_dir.iterdir() if p.is_file()]
    return f"{len(files)} file(s): " + ", ".join(files[:3])


def main() -> int:
    docs = []
    for name in ("SUMMARY_FOR_DECISION.md", "ROUTE_RECHECK_2026-11-15.md",
                 "SELF_AUDIT.md", "PREREGISTRATION_SEG_001.md",
                 "evidence/COMMANDS.md"):
        p = DOC / name
        if p.is_file():
            docs.append({"file": name, "bytes": p.stat().st_size,
                         "lines": len(p.read_text(encoding="utf-8").splitlines())})

    groups = []
    for d in sorted(EV.iterdir(), key=lambda x: (len(x.name), x.name)):
        if not d.is_dir() or not d.name.startswith("R"):
            continue
        try:
            n = int(d.name[1:])
        except ValueError:
            continue
        total = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        about = DESCRIPTIONS.get(n)
        assert about, f"evidence group {d.name} has no curated description"
        groups.append({"group": d.name, "n": n,
                       "files": sum(1 for p in d.rglob("*") if p.is_file()),
                       "bytes": total, "about": about})
    groups.sort(key=lambda g: g["n"])

    out = {
        "schema": "v431-recheck-r62/1",
        "question": "接手者的入口：索引直接从磁盘生成，避免与内容漂移",
        "documents": docs,
        "evidence_groups": groups,
        "counts": {"documents": len(docs), "groups": len(groups),
                   "evidence_bytes": sum(g["bytes"] for g in groups)},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- the index itself
    L = []
    L.append("# 索引 — v4.3.0 路线复核（接手从这里开始）\n")
    L.append("> 本文件由 `tools/rl/v431_make_index.py` **生成**，不是手写；"
             "新增证据后重跑即可，不会与内容漂移。\n")
    L.append("## 1. 你是谁，先读哪份\n")
    L.append("| 你的目的 | 读这份 | 规模 |\n| --- | --- | --- |")
    idx = {d["file"]: d for d in docs}
    def size(name):
        d = idx.get(name)
        return f"{d['lines']} 行 / {d['bytes']//1024} KB" if d else "—"
    L.append(f"| **要做一个决定**（只有两件事等你） | `SUMMARY_FOR_DECISION.md` | {size('SUMMARY_FOR_DECISION.md')} |")
    L.append(f"| **要核对某个数字** | `ROUTE_RECHECK_2026-11-15.md` | {size('ROUTE_RECHECK_2026-11-15.md')} |")
    L.append(f"| **要复跑我的检查** | `SELF_AUDIT.md`（§0 一条命令跑 22 个脚本） | {size('SELF_AUDIT.md')} |")
    L.append(f"| **要知道判据当初怎么定的** | `PREREGISTRATION_SEG_001.md`（正文是 rev E 冻结快照，已加横幅） | {size('PREREGISTRATION_SEG_001.md')} |")
    L.append(f"| **要查某条命令的原始输出** | `evidence/COMMANDS.md` | {size('evidence/COMMANDS.md')} |")
    L.append("")
    L.append("## 2. 三十秒版本\n")
    L.append("- 原报告**数值可重现**（11 个 CE 数字逐位一致），但**基线定义错了**、`880/219` 这个划分**从未存在**。")
    L.append("- 换对基线后，16× 小模型在 **CE** 上从\"不优于\"翻成\"优于\"，但**准确率仍输给恒预测多数类**——BC 既未被证明有效，也未被证明无效。")
    L.append("- 三臂里只有**臂 A** 跑得起来（B/C 被 `_pair_cand` 契约挡住）。它的判定**双向**：上尾判据 6/8 支持它，典型判据 5/8 反对它，最宽松的单侧检验也只到 **p=0.0584**。")
    L.append("- 一个**只在训练窗口上选出**的固定动作拿 **172.4**，是臂 A 的 **1.83 倍**（它是 4 个确定性值，非 n=80）。")
    L.append("- 等你决定的只有两件：**批准修 `PARK`**（→ 234.7 h 变 ≈9 h，臂 A 可判定），或**就此定稿**。")
    L.append("")
    L.append("## 3. 证据组一览（自动生成）\n")
    L.append(f"共 **{len(groups)}** 组有独立文件，**{out['counts']['evidence_bytes']//1024} KB**；"
             f"另有 **{len(PROCESS_STEPS)}** 个流程步骤（输出是断言本身，无独立文件）。\n")
    L.append("| 组 | 文件 | 大小 | 这一组在做什么 |\n| --- | ---: | ---: | --- |")
    for g in groups:
        about = g["about"].replace("|", "\\|")
        L.append(f"| `{g['group']}` | {g['files']} | {g['bytes']//1024} KB | {about} |")
    for n, what in sorted(PROCESS_STEPS.items()):
        L.append(f"| `R{n}` | — | — | {what} |")
    L.append("")
    L.append("## 4. 五分钟自查（接手者最短路径）")
    L.append("")
    L.append("```powershell")
    L.append("$env:PYTHONPATH='D:\\WorkShop\\Touhou_RL'")
    L.append("python tools\\rl\\v431_selfcheck.py          # 5 个廉价检查，约 0.3 s")
    L.append("python tools\\rl\\v431_selfcheck.py --full   # 加 6 个（含两次约 4 min 的全树遍历）")
    L.append("```")
    L.append("")
    L.append("它按**证据层**而不是按节检查：表头口径（`R69`）、百分比自证性（`R70`）、成本链（`R71`）、K 表引用（`R72`）、旧值残留（`R73`），`--full` 再加产物缺席（`R63`）、N4/N5（`R64`）、证据类型（`R65`）、切分脚本（`R66`）、全称断言（`R67`）、符号口径（`R68`）。")
    L.append("")
    L.append("> **为什么单独一条命令而不是并进套件**：套件是**断言门**（`v431_verify_report.py` 最后跑，断言覆盖全部已存证据）；这条是给**人**用的聚合器。把两次全树遍历塞进断言门只会让套件慢一倍，不增加任何保证。")
    L.append("")
    L.append("## 5. 一条命令复跑全部检查")
    L.append("")
    L.append("```powershell")
    L.append("$env:PYTHONPATH='D:\\WorkShop\\Touhou_RL'")
    L.append("& .tmp_v431_seg001\\run_all_audits.ps1   # 期望: non-zero: 0 of 23（连跑两遍仍全绿）")
    L.append("```")
    L.append("")
    L.append("## 6. 资产与范围")
    L.append("")
    L.append("- 完整报告与预注册在 `Document/Update/Alpha/v4.3.1/`；引擎产物在 `.tmp_v431_seg001/`（gitignored）。")
    L.append("- **本复核未修改任何既有文件**、未 `git add`/`commit`、未恢复 WP05、未触碰资产红线路径。")
    L.append("- 引擎时间 **13.5 h**；`R39` 起的 38 轮**全部离线**。")
    L.append("")
    (DOC / "README_INDEX.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"documents {len(docs)}  groups {len(groups)}  evidence {out['counts']['evidence_bytes']//1024} KB")
    print("wrote:", DOC / "README_INDEX.md")
    print("wrote:", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
