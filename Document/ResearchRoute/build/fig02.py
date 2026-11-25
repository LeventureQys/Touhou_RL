# -*- coding: utf-8 -*-
"""图 2：37 天时间线，按旬分三栏，避免右侧拥挤。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F

fig, ax = F.canvas(13.4, 7.2)

F.text(ax, 4, 96, "三十七天，五十二个版本目录", size=16.5, weight="bold")
F.text(ax, 4, 92.2, "每个节点都是一次判定：墨绿成立，锈红失败，金色转向或部分成立。",
       size=9.4, color=F.MUTED)
cols = [
    ("10 月 18 日 — 10 月 31 日", [
        ("10-18", "v1.0 首次提交", "抓帧、裁剪、共享 CNN 与 collect/process/train 流水线", "n"),
        ("10-20", "v1.0.4 空间网络", "A-01/A-02 过；真机 1,627.7 帧，低于随机 2,418", "f"),
        ("10-25", "v3.6 七策略消融", "ENEMY 只 +3.63%（门槛 15%）；POWER 死亡 1.95→4.60", "f"),
        ("10-28", "v3.7 时空选路", "合成 117/117、封死降 92.3%；真机 1.333 死/局，更差", "p"),
    ]),
    ("11 月 1 日 — 11 月 12 日", [
        ("11-02", "v3.7.4 同帧证明", "240/240 帧；旧读法 20.83% 跨 ticket", "n"),
        ("11-05", "v3.7.6 BC 示范", "验证 4.96%，保持上一动作基线 97.52%", "f"),
        ("11-10", "v4.0 P0 训练场", "108/108 回放逐位一致；渲染 819.7 fps，门槛 2,000", "p"),
        ("11-12", "v4.0 D1 语料", "137 分片 / 46.9 万决策 / 187.7 万帧", "n"),
    ]),
    ("11 月 13 日 — 11 月 24 日", [
        ("11-13", "v4.1 WP05 暂停", "DAgger 三轮 15.9 万决策；20 个开发局全 0 过关", "f"),
        ("11-15", "v4.2 路线结论", "160 局 0 过关；八条结论，六条被推翻", "f"),
        ("11-21", "v4.3 复核 / v4.4 计划", "拆成 H1–H5 与 E0–E4；先修仪器再谈算法", "p"),
        ("11-22", "v8 放雷线总报告", "20 s 99.58% / 30 s 60.83%；10 次学放雷全败", "p"),
        ("11-23", "v8.29 复验设计", "G1 对 T2，192 个单臂回合，等只读回执", "p"),
    ]),
]

for ci, (head, items) in enumerate(cols):
    x0 = 4 + ci * 31.5
    w = 29.0
    step = 16.2 if ci < 2 else 13.4
    last_y = 77.8 - (len(items) - 1) * step
    F.rule(ax, x0 + 2.6, 83.8, x0 + 2.6, last_y - 5.0, color="#C9C2B4", lw=1.1)
    F.text(ax, x0 + 2.6, 87.2, head, size=9.2, weight="bold", color=F.INK)
    for i, (d, t, s, k) in enumerate(items):
        y = 77.8 - i * step
        col = {"n": F.ACC2, "f": F.FAIL, "p": F.GOLD}[k]
        ax.scatter([x0 + 2.6], [y], s=44, color=col, edgecolor=F.BG, linewidth=1.2, zorder=5)
        F.text(ax, x0 + 5.4, y + 1.6, d, size=7.6, color=F.MUTED)
        F.text(ax, x0 + 5.4, y - 1.7, t, size=8.9, weight="bold", color=col)
        F.text(ax, x0 + 5.4, y - 5.6, s, size=7.3, color=F.INK2)

# 底部统计条
F.box(ax, 4, 6.5, 92, 5.2, fc=F.PANEL, ec=F.RULE, r=0.9)
F.text(ax, 6.2, 9.1, "37 天", size=8.6, color=F.INK2, weight="bold")
F.text(ax, 12.5, 9.1, "52 个版本目录", size=8.6, color=F.INK2)
F.text(ax, 27.0, 9.1, "1,424 条 SHA-256 校验行", size=8.6, color=F.INK2)
F.text(ax, 49.0, 9.1, "约 58 GB 本机资产", size=8.6, color=F.INK2)
F.text(ax, 67.0, 9.1, "0 / 160 局过关", size=8.6, color=F.FAIL, weight="bold")

F.caption(ax, "图 2", "研究时间线。失败集中在十月下旬到十一月中旬，每一次转向都发生在数字出来之后。", y=2.2)
F.save(fig, "fig02_timeline.png")
