# -*- coding: utf-8 -*-
"""图 9：当前站位与两条待走的路。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F

fig, ax = F.canvas(13.2, 7.2)

F.text(ax, 4, 95, "现在站在哪里，下一步花哪笔引擎时间", size=15.5, weight="bold")
F.text(ax, 4, 91.2, "左边是已经停下的，右边是计划。所有数字都是计划量或历史实测，不是新成绩。",
       size=8.8, color=F.MUTED)

# 主线泳道
F.text(ax, 4, 84.5, "主线 v4.4：先证明「状态臂真的在看弹幕」", size=9.2, color=F.ACC, weight="bold")
F.rule(ax, 4, 82.4, 96.5, 82.4, color="#D8C9B4", lw=0.8)

main = [
    ("现在", "WP05 暂停", "三轮 DAgger\n15.9 万决策\n20 开发局 0/20", F.FAIL),
    ("E0", "身份与裁判", "1 真实成功 +\n1 真实死亡\n新进程复跑", F.INK2),
    ("E1a", "开发配对", "8 祖先 × 10 局\n= 80 局", F.INK2),
    ("E1b", "独立验证", "256 祖先 × 2 臂\n512 局 · 8–10 h", F.INK2),
    ("E2", "因果诊断", "3 配方 × 3 种子\n≤1,800 局 · 28–35 h", F.INK2),
    ("E3", "像素片段", "B/C 各 1.2 万决策\n960 个评测局", F.INK2),
    ("E4", "课程与自然开局", "接管点逐段前移\n自然 S1 另建清单", F.INK2),
]
x0, w, gap = 3.6, 12.2, 1.55
for i, (k, name, desc, col) in enumerate(main):
    x = x0 + i * (w + gap)
    fc = "#F6E7E1" if i == 0 else F.PANEL
    F.box(ax, x, 54, w, 25, fc=fc, ec=("#D9B3A5" if i == 0 else F.RULE))
    F.text(ax, x + 1.1, 76.0, k, size=7.0, color=F.MUTED, weight="bold")
    F.text(ax, x + 1.1, 71.6, name, size=8.5, color=col, weight="bold")
    F.text(ax, x + 1.1, 63.0, desc, size=6.5, color=F.INK2, lh=1.5)
    if i < 6:
        F.arrow(ax, x + w + 0.15, 66.5, x + w + gap - 0.15, 66.5, color="#B4AC9C", ms=7)

# 放雷线泳道
F.text(ax, 4, 48.5, "放雷线 v8.29：先确认「G1 的优势不是换起点换出来的」", size=9.2, color=F.GOLD, weight="bold")
F.rule(ax, 4, 46.4, 96.5, 46.4, color="#D8C9B4", lw=0.8)

bomb = [
    ("V829-INFO-01", "只读盘点", "主机 / HEAD / 三份权重 SHA\n历史逐窗计数核对", F.GOLD),
    ("冒烟", "旧窗复跑", "恢复哈希一致\n抽样次数 / 两帧 mask", F.INK2),
    ("PAIR-01", "192 个单臂回合", "8 个新祖先 × 3 ckpt × 2 RNG\n两档 · ≤403,200 帧", F.INK2),
    ("判据", "±5pp", "Δ30 ≥ +5pp 且 ≥5/8 祖先为正\n→ 提名 G1", F.ACC2),
]
x0, w, gap = 3.6, 20.6, 3.0
for i, (k, name, desc, col) in enumerate(bomb):
    x = x0 + i * (w + gap)
    F.box(ax, x, 28, w, 16.4, fc=F.PANEL, ec=F.RULE)
    F.text(ax, x + 1.2, 40.8, k, size=7.2, color=col, weight="bold")
    F.text(ax, x + 1.2, 36.6, name, size=9.0, color=F.INK, weight="bold")
    F.text(ax, x + 1.2, 31.4, desc, size=6.6, color=F.INK2, lh=1.45)
    if i < 3:
        F.arrow(ax, x + w + 0.2, 36.0, x + w + gap - 0.2, 36.0, color="#B4AC9C", ms=7)

# 阻塞项
F.text(ax, 4, 23.0, "卡住的三件事", size=8.4, color=F.MUTED, weight="bold")
blockers = [
    ("G410-1 的 100 种子 8px 审计未做", 4),
    ("WP06 的 HUD / 玩家定位未实现", 32),
    ("V829-INFO-01 回执未到，旧窗入口要重写", 60),
]
for s, x in blockers:
    F.text(ax, x, 18.6, "×", size=11, color=F.FAIL, weight="bold")
    F.text(ax, x + 2.2, 18.6, s, size=7.6, color=F.INK2)

# 底线
F.box(ax, 4, 8, 92.5, 6.2, fc=F.INK, ec=F.INK, r=0.8)
F.text(ax, 50, 11.1, "G410-2 的判据没变：确认 100 局至少 50 次过面，Wilson 下界 ≥35%，另加 20 局 Windows 零售复验。",
       size=8.2, color=F.BG, ha="center")

F.caption(ax, "图 9", "当前站位。主线继续卡在「像素学生不会玩」，放雷线卡在一份还没回来的只读回执。", y=2.4)
F.save(fig, "fig09_position.png")
