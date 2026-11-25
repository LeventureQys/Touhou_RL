# -*- coding: utf-8 -*-
"""统一的图版样式：米白底、墨黑字、锈红/墨绿两个强调色。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Polygon
import matplotlib.patheffects as pe

FONT = r"C:\Windows\Fonts\msyh.ttc"
fm.fontManager.addfont(FONT)

INK = "#1E2227"
INK2 = "#3C4046"
MUTED = "#71767C"
RULE = "#D8D3C8"
BG = "#FBFAF6"
PANEL = "#F2EEE5"
ACC = "#AD431F"      # 锈红：主线
ACC2 = "#0E6660"     # 墨绿：有效结果
GOLD = "#B4862C"
OK = "#2E6B4A"
FAIL = "#9E3A2A"
PART = "#A9721C"
BLUE = "#2F5C8A"

plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "axes.unicode_minus": False,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": INK,
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "figure.dpi": 110,
})


def canvas(w=12.0, h=6.75):
    # 画布按 0.78 缩放：字号不变，相对字号变大，缩到正文宽度后仍可读。
    fig = plt.figure(figsize=(w * 0.78, h * 0.78))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc=PANEL, ec=RULE, lw=0.9, r=1.2, z=2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def text(ax, x, y, s, size=10, color=INK, weight="normal", ha="left", va="center",
         style="normal", z=5, rotation=0, family="Microsoft YaHei", lh=1.35):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, style=style, zorder=z, rotation=rotation,
                   family=family, linespacing=lh)


def arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.1, style="-|>", ms=8, ls="-", z=4,
          connectionstyle="arc3,rad=0"):
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                        color=color, linewidth=lw, linestyle=ls, zorder=z,
                        shrinkA=0, shrinkB=0, connectionstyle=connectionstyle)
    ax.add_patch(p)
    return p


def rule(ax, x1, y1, x2, y2, color=RULE, lw=0.8, ls="-", z=1):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="butt")


def tag(ax, x, y, s, fc=INK, tc=BG, size=8.5, pad=0.9, z=6, weight="bold"):
    t = ax.text(x, y, s, fontsize=size, color=tc, ha="center", va="center",
                zorder=z + 1, fontweight=weight)
    t.set_bbox(dict(boxstyle=f"round,pad={pad/10},rounding_size=0.6", fc=fc, ec="none"))
    return t


def caption(ax, num, s, y=3.0, x=4.0):
    text(ax, x, y, num, size=10, color=ACC, weight="bold")
    text(ax, x + 4.2, y, s, size=9.2, color=MUTED)


def save(fig, name):
    import os
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", name)
    fig.savefig(out, dpi=220, facecolor=BG)
    plt.close(fig)
    print("saved", out)
