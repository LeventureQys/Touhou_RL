# -*- coding: utf-8 -*-
"""图 6：G410 八道门的状态表。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F

fig, ax = F.canvas(13.2, 7.4)

F.text(ax, 4, 95, "八道门，只过了一道", size=15.5, weight="bold")
F.text(ax, 4, 91.2, "状态取 2026-11-13 的暂停快照。开发局用 seed 1000–1099，确认局 61000–61099 一局未动。",
       size=8.8, color=F.MUTED)

cols = [(4, 12.5), (12.5, 34), (34, 63), (63, 97)]
heads = ["门", "要证明的能力", "判据（确认口径）", "现状"]
for (a, b), h in zip(cols, heads):
    F.text(ax, a + 0.6, 85.0, h, size=8.2, color=F.MUTED, weight="bold")
F.rule(ax, 4, 82.8, 96.5, 82.8, color="#BDB6A8", lw=0.9)

rows = [
    ("G410-0", "实验、环境、数据、裁判可信", "资产三态核验；≥3 训练种子 × 3 次重开 × 1,000 帧一致", "已通过", F.OK),
    ("G410-1", "教师质量、标签可学习、数据腿", "第一面 ≥90/100；PSN 每面死亡 ≤0.5；S1 ≥5 万唯一决策", "进行中：教师 100/100 过面\n严格不一致 80.95%，8px 审计未做", F.PART),
    ("G410-2", "像素学生过第一面并真部署", "确认 100 局 ≥50%；Wilson 下界 ≥35%；真实 20 局保留 ≥70%", "未开始：开发 20 局 0/20\n最好不到 4,000 帧", F.FAIL),
    ("G410-3", "连续像素事件与终局", "召回/精确下界各 ≥99%，≥400 事件；死亡延迟 P95 ≤10 帧", "未开始：只有规格，HUD 与玩家定位未实现", F.FAIL),
    ("G410-4", "只靠像素回报继续训练", "相对基线 ≥+15pp 且差下界 >0；3 个训练种子同向", "未开始", F.FAIL),
    ("G410-5", "同一策略完整一周目，不续关", "50% / 下界 35%；真实零售保留 70%", "未开始", F.FAIL),
    ("G410-6", "未见设置上的有限交互适应", "两个干净留出家族，AULC 改善下界 >0", "未开始", F.FAIL),
    ("G410-7", "第二作严格像素适应并通关", "50% / 下界 35%，真实复验", "未开始", F.FAIL),
]

y = 78.0
for i, (g, cap, crit, st, col) in enumerate(rows):
    h = 8.5
    if i % 2 == 0:
        F.box(ax, 4, y - h + 1.2, 92.5, h, fc="#F4F0E8", ec="none", r=0.4, z=1)
    F.text(ax, 5.0, y - 4.2, g, size=8.6, weight="bold", color=col)
    F.text(ax, 13.6, y - 4.2, cap, size=8.0, color=F.INK)
    F.text(ax, 34.6, y - 4.2, crit, size=6.9, color=F.INK2)
    F.text(ax, 63.6, y - 4.2, st, size=6.9, color=col, weight="bold", lh=1.45)
    y -= 8.6

# 工作包状态
F.rule(ax, 4, 9.8, 96.5, 9.8, color=F.RULE)
F.text(ax, 4, 6.4, "工作包", size=8.0, color=F.MUTED, weight="bold")
wp = [("00", "通过", F.OK), ("01", "通过", F.OK), ("02", "进行中", F.PART), ("03", "进行中", F.PART),
      ("04", "进行中", F.PART), ("05", "进行中", F.PART), ("06", "进行中", F.PART), ("07–12", "未启动", F.FAIL)]
x = 13.0
for name, st, col in wp:
    F.text(ax, x, 6.4, f"WP{name}", size=7.2, color=F.INK2)
    F.text(ax, x + 1.6 + (2.6 if len(name) > 2 else 1.4), 6.4, st, size=7.2, color=col, weight="bold")
    x += 10.4

F.caption(ax, "图 6", "G410 门禁与工作包。八道门里七道没开始，唯一通过的 G410-0 只保证「实验本身可信」。", y=2.0)
F.save(fig, "fig06_gates.png")
