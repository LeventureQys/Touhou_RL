# -*- coding: utf-8 -*-
"""图 7：十次「学放雷」的三种失败形态。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F

fig, ax = F.canvas(13.0, 6.6)

F.text(ax, 4, 94.5, "十次让学习接管放雷，三种失败形态", size=15.5, weight="bold")
F.text(ax, 4, 90.5, "同一副冻结的移动头，只换放雷决策。十次里没有一次稳定超过手写规则。",
       size=9.0, color=F.MUTED)

cards = [
    ("形态一", "时机被学成了常量", "6", "v8.3 / v8.5 / v8.6 / v8.17 / v8.18 / v8.28",
     "20 s 最好 29.58%\n对照规则 77.11%", "首雷恒在决策 0；\n用雷数恒为 3.00", F.FAIL),
    ("形态二", "策略退化成一直按雷", "1", "v8.24（900 决策预算）",
     "30 s 4.08%\n每回合输出 65–136 次", "靠引擎的每局限次\n才勉强停在 3 颗", F.FAIL),
    ("形态三", "换奖励、换读出，仍不过规则", "3", "v8.21（600）\nv8.27 密集奖励 D1 / D2",
     "30 s 4.92%\n密集奖励 0.61–0.78%", "20 s 的 90.17% 是 argmax 读出；\n同读出规则 91.67%", F.FAIL),
]

for i, (kicker, title, n, vers, score, mech, col) in enumerate(cards):
    x = 3.6 + i * 31.6
    F.box(ax, x, 32, 29.6, 48, fc=F.PANEL, ec="#DED6C6")
    F.text(ax, x + 1.8, 75.8, kicker, size=7.6, color=F.MUTED, weight="bold")
    F.text(ax, x + 1.8, 70.6, title, size=10.8, color=F.INK, weight="bold")
    F.text(ax, x + 1.8, 63.8, n, size=21, color=col, weight="bold")
    F.text(ax, x + 7.0, 63.8, "次尝试", size=7.6, color=F.MUTED)
    F.text(ax, x + 1.8, 55.8, vers, size=7.0, color=F.INK2, lh=1.5)
    F.rule(ax, x + 1.8, 49.0, x + 27.6, 49.0)
    F.text(ax, x + 1.8, 43.0, score, size=8.6, color=col, weight="bold", lh=1.5)
    F.text(ax, x + 1.8, 34.8, mech, size=7.2, color=F.INK2, lh=1.5)

# 机制解释
F.text(ax, 4, 24.0, "为什么学不会", size=8.6, color=F.MUTED, weight="bold")
F.text(ax, 4, 18.6, "花雷的代价约在 946 帧后才出现，反事实标签的视界只有 120 帧：48,168 对标签里，「放」从不比「不放」差。",
       size=8.8, color=F.INK2)
F.text(ax, 4, 13.4, "单步「106 维观测 → 2 类 logits」又没有剩余雷数、剩余决策、距死亡时间这类输入，最稳的解就是开局放掉。",
       size=8.8, color=F.INK2)
F.text(ax, 4, 8.2, "拆开之后两边的贡献才量得出来：学到的躲闪 +21.64pp（20 秒），收紧阈值 +31.14pp（30 秒）。",
       size=8.6, color=F.ACC)

F.caption(ax, "图 7", "十次学放雷的失败形态。不是信号太稀疏，是表示与视界看不到代价。", y=2.0)
F.save(fig, "fig07_attempts.png")
