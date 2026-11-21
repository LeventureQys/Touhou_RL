"""v4.3.1 recheck R71: does the cost chain recompute? And is 70.4 s the right unit for every episode?

Every hour figure in the report descends from two numbers: the per-episode cost (70.4 s) and an
episode count. A single slip in that chain silently moves a decision-relevant number (234.7 h vs
9 h is literally what the PARK decision is about).

Two things to separate:

  1. ARITHMETIC -- does each claim follow from its own episode count x 70.4 s?
  2. UNIT -- 70.4 s is the cost of a FULL-LENGTH episode (600 frames = 300 decisions). Episodes
     that die early cost less. So "N x 70.4" is an UPPER BOUND whenever the group contains deaths.
     R20 measured 72 episodes in 1,282 s = 17.8 s/episode, and R31 measured 40 in 2,264 s =
     56.6 s/episode -- both far below 70.4. Using 70.4 for those groups overstates the cost.

That distinction is the finding: the hour figures are upper bounds, not measurements, except where
a group actually measured its own wall clock.

Writes only Document/Update/Alpha/v4.3.1/evidence/R71/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
OUT = EV / "R71"
UNIT = 70.4          # seconds per full-length episode (300 decisions, 600 frames)
SIM_FLOOR = 8663 * 7.7 / 1000.0   # seconds, from the measured 7.7 ms/frame


def h(x: float) -> float:
    return x / 3600.0


def main() -> int:
    claims = []

    def add(name, episodes, stated_h, note=""):
        implied = h(episodes * UNIT)
        claims.append({"claim": name, "episodes": episodes, "stated_h": stated_h,
                       "implied_h": round(implied, 2),
                       "delta_h": round(implied - stated_h, 2),
                       "holds": abs(implied - stated_h) <= 0.06,
                       "note": note})

    add("C4 密集 9 窗口（162 episode）", 162, 3.2)
    add("n=80 的评测（80 episode）", 80, 1.6)
    add("补到 163/臂（166 episode）", 166, 3.3)
    add("200 训练 episode", 200, 3.9)
    add("1,000 episode 单窗口", 1000, 19.6)
    add("3 训练种子 × 4 窗口 × 1,000 episode", 12000, 234.7)
    add("计账 643 episode", 643, 12.6)
    add("12 训练 episode", 12, 0.25)
    add("C4 每窗口（18 episode）", 18, round(h(18 * UNIT), 2), "报告写 ≈21 min")

    # ---- measured per-group costs, DERIVED from the evidence where the evidence has them
    measured = []
    r20_path = EV / "R20" / "c4_screen.json"
    if r20_path.is_file():
        d20 = json.loads(r20_path.read_text(encoding="utf-8"))
        states = len(d20.get("states") or [])
        complete = sum(1 for st in d20.get("states") or [] if st.get("complete_18"))
        eps = complete * 18
        measured.append({
            "group": "R20", "derived_from": "states_complete x 18",
            "states": states, "states_complete": complete, "episodes": eps,
            "wall_s": 1282, "wall_s_source": "报告与运行记录（不在 R20 的 JSON 里）",
            "s_per_episode": round(1282 / eps, 1) if eps else None,
            "exists_in_evidence_json": False,
        })
    r31_path = EV / "R31" / "baseline_profile.json"
    if r31_path.is_file():
        d31 = json.loads(r31_path.read_text(encoding="utf-8"))
        cf = d31.get("c_fair_pooled") or {}
        measured.append({
            "group": "R31", "derived_from": "c_fair_pooled.n",
            "episodes": cf.get("n"), "wall_s": 2264,
            "wall_s_source": "报告与运行记录（**R31 的 JSON 里没有 wall time**）",
            "s_per_episode": round(2264 / cf["n"], 1) if cf.get("n") else None,
            "exists_in_evidence_json": False,
        })

    out = {
        "schema": "v431-recheck-r71/1",
        "question": ("报告里每个小时数能不能从\"episode 数 × 70.4 s\"重算出来？"
                     "70.4 s 对每一类 episode 都适用吗？"),
        "unit_seconds": UNIT,
        "simulation_floor_seconds": round(SIM_FLOOR, 1),
        "claims": claims,
        "measured_per_group": measured,
        "unit_caveat": {
            "what_70_4_is": "满长 episode（300 决策 / 600 帧）的成本；早死的 episode 更便宜",
            "measured_below_unit": [
                {"source": "R20", "s_per_episode": 17.8,
                 "why": "C4 在稀疏早期状态上，恒定动作很快死"},
                {"source": "R31", "s_per_episode": 56.6,
                 "why": "C-fair 平均 79.7 决策就死，比满长短"},
            ],
            "consequence": ("凡是用 70.4 s 乘一个**含死亡**的组，得到的是**上界**而不是实测；"
                            "只有真正计过 wall clock 的组才是实测值"),
        },
        "counts": {"claims": len(claims), "hold": sum(1 for c in claims if c["holds"]),
                   "measured_groups": len(measured)},
        "verdict": {"all_claims_recompute": None, "unit_is_an_upper_bound": None,
                    "reading": None},
    }
    v = out["verdict"]
    v["all_claims_recompute"] = all(c["holds"] for c in claims)
    v["unit_is_an_upper_bound"] = True
    bad = [c["claim"] for c in claims if not c["holds"]]
    v["reading"] = (
        f"{len(claims)} 条小时数里 **{sum(1 for c in claims if c['holds'])} 条可从"
        f"「episode 数 × 70.4 s」重算出来**，偏差 ≤0.06 h；"
        + (f"不成立的是 {bad}。" if bad else "**全部成立**。")
        + " **但 70.4 s 的口径需要说清**：它是**满长 episode**（300 决策）的成本，"
        f"而实测里 R20 是 **17.8 s/episode**、R31 是 **56.6 s/episode**——早死的 episode 便宜得多。"
        "所以凡是用 70.4 s 乘一个**含死亡**的组，得到的是**上界**而非实测；"
        "只有真正计过 wall clock 的组才是实测值。**这不改变任何决策的符号，但改变\"13.5 h\"这类数的含义。**")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cost_chain.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"unit {UNIT} s/episode   simulation floor {SIM_FLOOR:.1f} s")
    print(f"{'claim':>42s} {'eps':>7s} {'stated':>8s} {'implied':>8s} {'ok':>4s}")
    print("-" * 76)
    for c in claims:
        print(f"{c['claim']:>42s} {c['episodes']:7d} {c['stated_h']:8.2f} "
              f"{c['implied_h']:8.2f} {str(c['holds']):>4s}")
    print()
    for m in measured:
        print(f"  measured {m['group']}: {m['episodes']} eps in {m['wall_s']} s "
              f"= {m['s_per_episode']} s/episode   [{m['derived_from']}; "
              f"wall time in evidence json: {m['exists_in_evidence_json']}]")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'cost_chain.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
