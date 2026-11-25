# -*- coding: utf-8 -*-
"""图 5：本机资产规模（对数横轴）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
import numpy as np

fig, ax = F.canvas(12.4, 6.4)

F.text(ax, 4, 94, "五十八个 GB，没有一条能进 Git", size=15.5, weight="bold")
F.text(ax, 4, 90, "引擎、数据集、checkpoint 全在本机；仓库里只有代码、配置和文档。",
       size=9.2, color=F.MUTED)

items = [
    ("D1 规范语料", 28.76, "469,216 决策 / 137 分片", F.ACC),
    ("v4.2 双帧语料", 16.0, "19,772 个槽位", F.ACC),
    ("DAgger round1", 2.85, "56,000 决策", F.ACC2),
    ("DAgger round2", 2.67, "52,500 决策", F.ACC2),
    ("DAgger round3", 2.60, "50,787 决策 · 已暂停", F.ACC2),
    ("S1 语料", 2.34, "66,143 决策", F.ACC2),
    ("v8 证据", 1.95, "29 个版本目录", F.GOLD),
    ("游戏数据", 0.474, "学生与 DAgger 共用", F.MUTED),
]

axL = fig.add_axes([0.245, 0.20, 0.62, 0.55])
axL.set_facecolor(F.BG)
axL.set_xscale("log")
for s in ["top", "right"]:
    axL.spines[s].set_visible(False)
axL.spines["left"].set_color(F.RULE)
axL.spines["bottom"].set_color(F.RULE)
axL.grid(axis="x", color=F.RULE, lw=0.6, alpha=0.7)
axL.set_axisbelow(True)

ys = np.arange(len(items))[::-1]
vals = [it[1] for it in items]
axL.barh(ys, vals, height=0.52, color=[it[3] for it in items], zorder=3)
for y, (name, v, note, col) in zip(ys, items):
    axL.text(v * 1.13, y, f"{v:g} GB", va="center", fontsize=8.6, color=F.INK)
    axL.text(v * 1.13, y - 0.30, note, va="center", fontsize=7.0, color=F.MUTED)
axL.set_yticks(ys)
axL.set_yticklabels([it[0] for it in items], fontsize=8.8, color=F.INK2)
axL.set_xlim(0.3, 120)
axL.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
axL.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"], fontsize=7.8)
axL.set_xlabel("GB（对数刻度）", fontsize=8.0, color=F.MUTED, labelpad=6)
axL.tick_params(colors=F.MUTED, length=3)

# 右侧总量标记
F.box(ax, 4, 10, 18.5, 15, fc=F.PANEL, ec=F.RULE)
F.text(ax, 6.0, 20.6, "合计", size=8.0, color=F.MUTED)
F.text(ax, 6.0, 16.0, "≈ 58 GB", size=16, color=F.INK, weight="bold")
F.text(ax, 6.0, 12.2, "1,424 条 SHA 校验行", size=7.4, color=F.MUTED)

F.caption(ax, "图 5", "本机资产。数据量在涨，能力没涨——这也是路线被质疑的地方。", y=2.0)
F.save(fig, "fig05_assets.png")
