# -*- coding: utf-8 -*-
"""图 8：四组配置在四个预算上的存活率。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
import numpy as np

fig, ax = F.canvas(12.8, 6.6)

F.text(ax, 4, 94.5, "把「何时放雷」写成规则后，曲线抬起来了", size=15.5, weight="bold")
F.text(ax, 4, 90.5, "真实引擎，按决策数计预算。端到端与学放雷两列在 40 秒档没有可用数据。",
       size=9.0, color=F.MUTED)

axB = fig.add_axes([0.075, 0.22, 0.88, 0.58])
axB.set_facecolor(F.BG)
for s in ["top", "right"]:
    axB.spines[s].set_visible(False)
axB.spines["left"].set_color(F.RULE)
axB.spines["bottom"].set_color(F.RULE)
axB.grid(axis="y", color=F.RULE, lw=0.6, alpha=0.7)
axB.set_axisbelow(True)

labels = ["10 秒", "20 秒", "30 秒", "40 秒"]
data = {
    "纯规则 R5": [100.0, 77.11, 17.61, 1.00],
    "端到端学习": [100.0, 31.11, 5.00, None],
    "学放雷（最好）": [99.86, 34.72, 8.33, None],
    "紧迫度自适应 α=0.5": [100.0, 99.58, 60.83, 14.17],
}
colors = {"纯规则 R5": "#8E8B84", "端到端学习": F.ACC, "学放雷（最好）": F.GOLD, "紧迫度自适应 α=0.5": F.ACC2}

n = len(labels)
w = 0.19
x = np.arange(n)
for i, (name, vals) in enumerate(data.items()):
    xs = x + (i - 1.5) * w
    vv = [v if v is not None else 0 for v in vals]
    axB.bar(xs, vv, width=w * 0.92, color=colors[name], zorder=3,
            label=name, edgecolor="none")
    for xi, v in zip(xs, vals):
        if v is None:
            continue
        else:
            axB.text(xi, v + 1.6, f"{v:.1f}" if v < 100 else "100", ha="center",
                     va="bottom", fontsize=6.6, color=F.INK2)

axB.text(3.0, 2.2, "未测", ha="center", va="bottom", fontsize=7.0, color=F.MUTED)

axB.set_xticks(x)
axB.set_xticklabels(labels, fontsize=9.6, color=F.INK2)
axB.set_ylim(0, 108)
axB.set_yticks([0, 20, 40, 60, 80, 100])
axB.set_yticklabels(["0", "20", "40", "60", "80", "100%"], fontsize=8.2)
axB.tick_params(colors=F.MUTED, length=3)
axB.set_ylabel("存活率", fontsize=8.6, color=F.MUTED, labelpad=8)
axB.legend(loc="upper right", bbox_to_anchor=(1.0, 1.14), ncol=4, frameon=False,
           fontsize=7.8, handlelength=1.0, columnspacing=1.4, handletextpad=0.5)

# 差值标注在分组标签下方
axB.annotate("α − 规则 = +22.5pp", xy=(1, -0.115), xycoords=("data", "axes fraction"),
             ha="center", va="top", fontsize=7.2, color=F.ACC2, annotation_clip=False)
axB.annotate("α − 规则 = +43.2pp", xy=(2, -0.115), xycoords=("data", "axes fraction"),
             ha="center", va="top", fontsize=7.2, color=F.ACC2, annotation_clip=False)

F.text(ax, 4, 11.6, "样本：自适应列 n=240；规则列 n=3,600；端到端与学放雷各约 n=3,600。",
       size=7.8, color=F.INK2)
F.text(ax, 4, 8.4, "30 秒档对预注册固定臂 35/15/3 的 48.75% 单侧 Fisher p=0.0051；对 40/20/5 的 58.75% 为 p=0.355。",
       size=7.8, color=F.INK2)

F.caption(ax, "图 8", "四组配置的分档存活率。规则与学习各管一半，合起来才过 30 秒。", y=2.0)
F.save(fig, "fig08_survival.png")
