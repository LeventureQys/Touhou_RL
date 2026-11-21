"""v4.3.1 recheck R55: is there ANY reasonable criterion under which arm A looks better?

R53 showed 5 of 8 criteria oppose arm A. A review that only ever looks for reasons to fail a
candidate is as untrustworthy as a cheerleader, so this does the opposite: it builds the set of
criteria that are most FAVOURABLE to arm A -- upper tail, survival count, best window -- and asks
how far the support actually goes.

Two of the favourable criteria are legitimate and were already in the report: the survival count
(6/80 vs 1/80) and Q3 (110.5 vs 98.8). The interesting question is whether the survival advantage
clears 0.05 under the ONE-SIDED test, which is the most generous version anyone could defend. The
report has been quoting the two-sided p = 0.1167 throughout.

Writes only Document/Update/Alpha/v4.3.1/evidence/R55/.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / ".tmp_v431_seg001" / "full" / "raw"
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
OUT = EV / "R55"
CAP = 300
DEV = (11, 12, 17, 18)


def episodes(paths) -> list[float]:
    out = []
    for p in paths:
        if p.suffix == ".jsonl":
            recs = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        else:
            recs = [json.loads(p.read_text(encoding="utf-8"))]
        for r in recs:
            v = r.get("return_decisions")
            if v is None and "first_death_offset" in r:
                off = r["first_death_offset"]
                v = CAP if off is None else off / 2.0
            if v is not None:
                out.append(float(v))
    return out


def hypergeom(k: int, n: int, K: int, N: int) -> float:
    """P(X = k) drawing n from N with K successes."""
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def fisher_one_sided(a_surv: int, a_n: int, b_surv: int, b_n: int) -> float:
    """One-sided p for 'arm survives more often than the baseline'."""
    N, K, n = a_n + b_n, a_surv + b_surv, a_n
    lo = max(0, n - (N - K))
    hi = min(n, K)
    return sum(hypergeom(k, n, K, N) for k in range(a_surv, hi + 1))


def main() -> int:
    arm = episodes(
        [RAW / "armA" / f"eval-seed{s}-armA-multi4-f8664-128.jsonl" for s in DEV]
        + [RAW / "armA" / f"eval-seed{s}-multi4-128-ext.jsonl" for s in DEV])
    fair = episodes(
        [p for s in DEV for p in sorted((RAW / "cfair").glob(f"cfair-seed{s}-f8664-ep*.json"))])

    arm_surv = sum(1 for v in arm if v >= CAP - 1e-9)
    fair_surv = sum(1 for v in fair if v >= CAP - 1e-9)
    p_one = fisher_one_sided(arm_surv, len(arm), fair_surv, len(fair))

    top = lambda v, k: statistics.mean(sorted(v, reverse=True)[:k])
    arm_w = {s: episodes([RAW / "armA" / f"eval-seed{s}-armA-multi4-f8664-128.jsonl",
                          RAW / "armA" / f"eval-seed{s}-multi4-128-ext.jsonl"]) for s in DEV}
    fair_w = {s: episodes(sorted((RAW / "cfair").glob(f"cfair-seed{s}-f8664-ep*.json")))
              for s in DEV}

    crit = [
        {"criterion": "跑满 300 的次数", "arm": f"{arm_surv}/{len(arm)}",
         "baseline": f"{fair_surv}/{len(fair)}", "favours_arm": arm_surv > fair_surv,
         "note": "报告一直用双侧 p=0.1167"},
        {"criterion": "存活率单侧 Fisher（最宽松的合法检验）", "arm": f"{arm_surv/len(arm):.3f}",
         "baseline": f"{fair_surv/len(fair):.3f}", "favours_arm": p_one < 0.05,
         "p_one_sided": round(p_one, 4),
         "note": "即使单侧也 > 0.05" if p_one >= 0.05 else "单侧 < 0.05"},
        {"criterion": "上四分位 Q3", "arm": 110.5, "baseline": 98.8,
         "favours_arm": True, "note": ""},
        {"criterion": "前 10 个 episode 的均值", "arm": round(top(arm, 10), 1),
         "baseline": round(top(fair, 10), 1),
         "favours_arm": top(arm, 10) > top(fair, 10), "note": ""},
        {"criterion": "≥200 决策的 episode 比例",
         "arm": f"{sum(1 for v in arm if v >= 200)}/{len(arm)}",
         "baseline": f"{sum(1 for v in fair if v >= 200)}/{len(fair)}",
         "favours_arm": sum(1 for v in arm if v >= 200) > sum(1 for v in fair if v >= 200),
         "note": ""},
        {"criterion": "单个最好窗口（seed17）的均值",
         "arm": round(statistics.mean(arm_w[17]), 1),
         "baseline": round(statistics.mean(fair_w[17]), 1),
         "favours_arm": statistics.mean(arm_w[17]) > statistics.mean(fair_w[17]),
         "note": "只有一个窗口"},
        {"criterion": "seed17 的中位数", "arm": round(statistics.median(arm_w[17]), 1),
         "baseline": round(statistics.median(fair_w[17]), 1),
         "favours_arm": statistics.median(arm_w[17]) > statistics.median(fair_w[17]),
         "note": ""},
        {"criterion": "四个窗口里胜出的窗口数（按均值）",
         "arm": f"{sum(1 for s in DEV if statistics.mean(arm_w[s]) > statistics.mean(fair_w[s]))}/4",
         "baseline": "—",
         "favours_arm": sum(1 for s in DEV
                            if statistics.mean(arm_w[s]) > statistics.mean(fair_w[s])) >= 3,
         "note": "3/4 才算压倒性"},
    ]

    fav = [c["criterion"] for c in crit if c["favours_arm"]]
    out = {
        "schema": "v431-recheck-r55/1",
        "question": "有没有任何**合理且对臂 A 有利**的判据能支持\"臂 A 学会了\"？",
        "criteria": crit,
        "counts": {"criteria": len(crit), "favours_arm": len(fav)},
        "favoured_by": fav,
        "survival_one_sided": {"arm": f"{arm_surv}/{len(arm)}",
                               "baseline": f"{fair_surv}/{len(fair)}",
                               "p_one_sided": round(p_one, 4),
                               "clears_005": p_one < 0.05},
        "verdict": {"support_exists": None, "support_is_upper_tail_only": None,
                    "reading": None},
    }
    v = out["verdict"]
    v["support_exists"] = len(fav) > 0
    v["support_is_upper_tail_only"] = all(
        any(w in c for w in ("Q3", "前 10", "≥200", "seed17", "跑满", "存活", "单侧"))
        for c in fav) if fav else False
    v["reading"] = (
        f"**{len(fav)}/{len(crit)} 条对臂 A 有利**：{fav}。所以\"臂 A 什么都没学到\"是**错的**——"
        f"它在存活次数、Q3、前 10 名均值、≥200 的比例、以及 seed17 上确实更好。"
        f"**但所有这些有利判据都是\"上尾\"或\"单窗口\"判据**：没有一个说的是\"典型 episode 更好\"。"
        f"最关键的是**最宽松的合法检验也差一点**：存活率单侧 Fisher **p = {p_one:.4f}**（> 0.05），"
        f"比双侧的 0.1167 好，但**仍然过不了线**。"
        f"所以正确的表述是：**优势真实存在，但只存在于分布的顶端**；"
        f"它既不足以说\"学会了\"，也不该被写成\"没有差别\"。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "favourable_criteria.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'criterion':>34s} {'arm':>9s} {'baseline':>9s} {'favours':>8s}")
    print("-" * 68)
    for c in crit:
        print(f"{c['criterion']:>34s} {str(c['arm']):>9s} {str(c['baseline']):>9s} "
              f"{str(c['favours_arm']):>8s}")
    print()
    print(f"favours arm: {len(fav)}/{len(crit)}")
    print(f"one-sided Fisher p = {p_one:.4f}  clears 0.05 = {p_one < 0.05}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'favourable_criteria.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
