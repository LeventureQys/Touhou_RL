# -*- coding: utf-8 -*-
"""图 3：离线指标与闭环成绩的落差。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
import numpy as np

fig, ax = F.canvas(12.6, 6.6)

F.text(ax, 4, 94, "离线分数与闭环成绩，是两张脸", size=15.5, weight="bold")
F.text(ax, 4, 90, "左边两组是同一个模型在测试集与真实对局里的表现；右边是 v1.0.4 的真机存活帧数。",
       size=9.0, color=F.MUTED)

# 左：斜率图
axL = fig.add_axes([0.055, 0.16, 0.42, 0.64])
axL.set_facecolor(F.BG)
axL.set_ylim(0, 1.0)
axL.set_xlim(0, 1)
for s in ["top", "right"]:
    axL.spines[s].set_visible(False)
axL.spines["left"].set_color(F.RULE)
axL.spines["bottom"].set_color(F.RULE)
axL.set_xticks([0.06, 0.94])
axL.set_xticklabels(["离线", "闭环"], fontsize=9.5, color=F.INK2)
axL.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axL.set_yticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=8.2)
axL.tick_params(colors=F.MUTED, length=3)
axL.grid(axis="y", color=RULE if False else F.RULE, lw=0.6, alpha=0.7)
axL.set_axisbelow(True)

cases = [
    ("合成像素 · masked 准确率", 0.913, 0.262, 0.190, F.ACC),
    ("规则蒸馏 · 离线拟合", 0.920, 0.290, 0.140, F.ACC2),
]
for i, (lab, off, hi, lo, col) in enumerate(cases):
    axL.plot([0.06, 0.94], [off, hi], color=col, lw=1.5, alpha=0.85, zorder=2,
             label=lab)
    axL.scatter([0.06], [off], s=64, color=col, zorder=4, edgecolor=F.BG, linewidth=1.3)
    axL.plot([0.94, 0.94], [lo, hi], color=col, lw=6, alpha=0.22, solid_capstyle="round", zorder=3)
    axL.scatter([0.94, 0.94], [hi, lo], s=48, color=col, zorder=4, edgecolor=F.BG, linewidth=1.2)
    axL.text(1.005, hi + 0.018, f"{hi*100:.0f}%", fontsize=9.0, color=col, weight="bold", va="bottom")
    axL.text(1.005, lo - 0.022, f"{lo*100:.0f}%", fontsize=9.0, color=col, weight="bold", va="top")
axL.text(0.075, 0.955, "0.91 — 0.92", fontsize=8.6, color=F.MUTED, ha="left", va="center")
axL.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), frameon=False, fontsize=7.2,
           handlelength=1.4, labelspacing=0.5)
axL.set_title("同一个模型，换一个口径", fontsize=10.5, color=F.INK, loc="left", pad=10)

# 右：真机存活
axR = fig.add_axes([0.60, 0.16, 0.35, 0.64])
axR.set_facecolor(F.BG)
for s in ["top", "right"]:
    axR.spines[s].set_visible(False)
axR.spines["left"].set_color(F.RULE)
axR.spines["bottom"].set_color(F.RULE)
bars = axR.bar([0, 1], [1627.7, 2418], width=0.5, color=[F.ACC, "#9AA0A6"], zorder=3)
axR.axhline(4000, color=F.FAIL, lw=1.1, ls=(0, (4, 3)), zorder=2)
axR.text(1.48, 4060, "过关门槛 4,000 帧", fontsize=8.0, color=F.FAIL, ha="right")
axR.set_ylim(0, 4600)
axR.set_xticks([0, 1])
axR.set_xticklabels(["v1.0.4 策略", "随机策略"], fontsize=9.0, color=F.INK2)
axR.set_yticks([0, 1000, 2000, 3000, 4000])
axR.set_yticklabels(["0", "1,000", "2,000", "3,000", "4,000"], fontsize=8.0)
axR.tick_params(colors=F.MUTED, length=3)
axR.grid(axis="y", color=F.RULE, lw=0.6, alpha=0.7)
axR.set_axisbelow(True)
for x, v in zip([0, 1], [1627.7, 2418]):
    axR.text(x, v + 90, f"{v:,.1f}", ha="center", fontsize=9.0, color=F.INK)
axR.set_title("真机存活帧数（越高越好）", fontsize=10.5, color=F.INK, loc="left", pad=10)

# 底部注解
F.rule(ax, 4, 11.5, 96, 11.5, color=F.RULE)
F.text(ax, 4, 8.4, "自机位置误差已经压到 2.41 px，0–15 px 危险桶的教师后悔值只有 1.27 px，逐帧 argmax 却翻转 50.7%。",
       size=8.6, color=F.INK2)
F.text(ax, 4, 5.2, "离线指标量的是「像不像教师」，闭环量的是「活不活得下去」。两者之间没有单调关系。",
       size=8.6, color=F.ACC, weight="bold")

F.caption(ax, "图 3", "离线与闭环的落差。三个案例都指向同一件事：准确率不是能力。", y=1.6)
F.save(fig, "fig03_gap.png")
