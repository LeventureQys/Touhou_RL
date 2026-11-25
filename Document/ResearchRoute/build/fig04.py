# -*- coding: utf-8 -*-
"""图 4：v4.0 的 P0–P4 依赖链与各自的门。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F

fig, ax = F.canvas(13.2, 6.4)

F.text(ax, 4, 94, "五段流水线，五道门", size=15.5, weight="bold")
F.text(ax, 4, 90, "设计上「门不过就不进入依赖它的阶段」。实际执行时，仪器先崩了，人只能回头修仪器。",
       size=9.2, color=F.MUTED)

stages = [
    ("P0", "训练场", "真实引擎 + 确定性回放\n进程内重开 + 共享内存像素", "G0", "108/108 回放一致\n渲染 819.7 / 2,000 fps", "未过", "f"),
    ("P1", "搜索教师", "HazardTape → 几何束搜索\n引擎 fork 验证计划", "G1", "开发 20/20 过面\n确认 100 种子未跑", "未结", "p"),
    ("P2", "感知与回报", "死亡 / bomb / 过面事件\nHUD 读数与奖励", "G2", "死亡召回 19.47%\n精确 4.58%", "未过", "f"),
    ("P3", "像素学生", "BC → DAgger\n18 动作 + bomb", "G3", "160 局 0 过关\n最深 3,914 / 13,195 帧", "未过", "f"),
    ("P4", "自我改进", "专家迭代 ‖ 像素回报 RL\n四臂 B0–B3", "G4", "未启动", "未启", "p"),
]
colors = {"f": F.FAIL, "p": F.GOLD, "n": F.ACC2}

x0, w, gap = 3.5, 16.6, 3.4
for i, (p, name, desc, g, res, st, k) in enumerate(stages):
    x = x0 + i * (w + gap)
    col = colors[k]
    F.box(ax, x, 36, w, 42, fc=F.PANEL, ec=F.RULE)
    F.text(ax, x + 1.6, 74.0, p, size=8.6, color=F.MUTED, weight="bold")
    F.text(ax, x + 1.6, 69.2, name, size=11.4, color=F.INK, weight="bold")
    F.text(ax, x + 1.6, 61.5, desc, size=7.0, color=F.INK2, lh=1.55)
    F.rule(ax, x + 1.6, 52.2, x + w - 1.6, 52.2)
    F.text(ax, x + 1.6, 48.8, g, size=8.2, color=col, weight="bold")
    F.text(ax, x + 5.0, 48.8, st, size=7.8, color=col, weight="bold")
    F.text(ax, x + 1.6, 42.6, res, size=6.9, color=F.INK2, lh=1.45)
    if i < 4:
        F.arrow(ax, x + w + 0.4, 58, x + w + gap - 0.4, 58, color="#A9A296", lw=1.2)
        F.text(ax, x + w + gap / 2, 62.0, "过门", size=6.6, color=F.MUTED, ha="center")

# 底部两条观察
F.rule(ax, 4, 32, 96, 32, color=F.RULE)
F.text(ax, 4, 26.5, "唯一的正式门结论是 G0 未过。随后 11 天里，P3 的实验仍在跑，理由是：不过门的原因指向仪器，不指向算法。", size=8.8, color=F.INK2)
F.text(ax, 4, 21.0, "代价是依赖链失去约束力。到 v4.2 只能靠事后审计补口径，八条结论里六条是这么被推翻的。", size=8.8, color=F.ACC)
# 底部四个统计块
stats = [
    (4.0, "D1 语料 · 帧", "1,876,864"),
    (28.0, "D1 语料 · 决策", "469,216"),
    (50.0, "D1 语料 · 分片", "137"),
    (68.0, "差分回放 · 行/侧", "12,311,460"),
]
for x, lab, num in stats:
    F.text(ax, x, 16.4, lab, size=7.2, color=F.MUTED)
    F.text(ax, x, 11.6, num, size=12.0, weight="bold", color=F.INK)

F.caption(ax, "图 4", "P0–P4 依赖链。门是设计出来的，执行时被仪器问题拽回去两次。", y=4.0)
F.save(fig, "fig04_chain.png")
