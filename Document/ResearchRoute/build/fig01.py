# -*- coding: utf-8 -*-
"""图 1：系统边界。特权只活在训练期，部署只看框选像素。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F

fig, ax = F.canvas(12.4, 6.9)

# 顶部说明
F.text(ax, 4, 94.5, "同一条流水线，两套输入权限", size=16, weight="bold")
F.text(ax, 4, 90.2, "训练期可以读内存，部署期只能读框内画面；红色虚线是两者的分界。", size=9.6, color=F.MUTED)

# 列标题
F.text(ax, 15, 84, "真实引擎", size=11, weight="bold", ha="center", color=F.INK)
F.text(ax, 15, 80.6, "TH08 1.00d · 60 Hz", size=8.6, ha="center", color=F.MUTED)
F.text(ax, 50, 84, "特权层", size=11, weight="bold", ha="center", color=F.ACC)
F.text(ax, 50, 80.6, "仅训练期，产物是标签与裁判日志", size=8.6, ha="center", color=F.MUTED)
F.text(ax, 84.5, 84, "学生策略", size=11, weight="bold", ha="center", color=F.ACC2)
F.text(ax, 84.5, 80.6, "部署期唯一允许的输入", size=8.6, ha="center", color=F.MUTED)

# 引擎面板
F.box(ax, 4, 26, 22, 50, fc=F.PANEL)
F.text(ax, 6, 71.5, "进程外只读", size=8.8, color=F.INK, weight="bold")
for i, s in enumerate(["OpenProcess + ReadProcessMemory", "GetDC + BitBlt 抓帧", "SendInput / PostMessage 送键", "不注入 · 不 hook · 不写内存"]):
    F.text(ax, 6, 66.5 - i * 6.2, s, size=7.2, color=F.INK2)
F.rule(ax, 6, 39.5, 24, 39.5)
F.text(ax, 6, 36, "抓帧中位 2.46 ms/帧", size=7.9, color=F.ACC)
F.text(ax, 6, 31.8, "按键生效延迟约 1 逻辑帧", size=7.9, color=F.ACC)
F.text(ax, 6, 27.6, "窗口不同步时会一直返回旧画面", size=7.6, color=F.FAIL)

# 特权面板
F.box(ax, 31, 26, 38, 50, fc="#F7F1E7", ec="#E2D6C2")
priv = [
    ("搜索教师", "18 个动作 × 持续帧，给出可执行计划"),
    ("标签与掩码", "已验证 / 存活 / 死亡 / 未知，四类分开"),
    ("独立裁判", "只写评测日志，不回流给策略"),
    ("回放与证据", "差分回放一致，SHA-256 逐文件登记"),
]
for i, (h, s) in enumerate(priv):
    y = 66.5 - i * 10.2
    F.text(ax, 33.5, y, h, size=9.6, weight="bold", color=F.ACC)
    F.text(ax, 33.5, y - 4.1, s, size=7.8, color=F.INK2)

# 学生面板
F.box(ax, 73, 26, 23, 50, fc="#EAF2F0", ec="#C9DCD8")
stud = [
    ("框选画面", "96 × 112，叠 4 个连续帧"),
    ("自身动作历史", "只含已经执行过的动作"),
    ("动作头", "18 个移动/聚焦 + bomb 二分类"),
    ("决策节拍", "每 2 个逻辑帧一次"),
]
for i, (h, s) in enumerate(stud):
    y = 66.5 - i * 10.2
    F.text(ax, 75.5, y, h, size=9.6, weight="bold", color=F.ACC2)
    F.text(ax, 75.5, y - 4.1, s, size=7.8, color=F.INK2)

# 箭头
F.arrow(ax, 26.2, 62, 30.8, 62, color=F.ACC)
F.text(ax, 28.5, 64.6, "读状态", size=7.4, ha="center", color=F.ACC)

F.arrow(ax, 26.2, 45, 30.8, 45, color=F.ACC2, ls="--")
F.text(ax, 28.5, 47.6, "抓帧", size=7.4, ha="center", color=F.ACC2)

F.arrow(ax, 69.2, 55, 72.8, 55, color=F.ACC)
F.text(ax, 71, 57.6, "标签", size=7.4, ha="center", color=F.ACC)

F.arrow(ax, 84.5, 25.6, 84.5, 18.5, color=F.ACC2)
F.text(ax, 84.5, 21.8, "送键", size=7.4, ha="center", color=F.ACC2)

# 部署边界
ax.plot([70.6, 70.6], [14, 78], color=F.FAIL, lw=1.2, ls=(0, (4, 3)), zorder=3)
F.text(ax, 70.6, 12.2, "▲ 部署边界：跨过这条线，特权输入即失效", size=8.4, color=F.FAIL, ha="center")

# 底部评测条
F.box(ax, 4, 6, 92, 4.6, fc=F.INK, ec=F.INK, r=0.8)
F.text(ax, 50, 8.3, "判据只有三件：真实成功、真实死亡、同帧逐帧复算     ·     离线准确率不在此列",
       size=8.8, color=F.BG, ha="center")

F.caption(ax, "图 1", "系统边界。引擎、特权层与像素学生三者的输入权限。", y=2.2)
F.save(fig, "fig01_boundary.png")
