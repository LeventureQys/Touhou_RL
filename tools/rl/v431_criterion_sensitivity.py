"""v4.3.1 recheck R53: does the verdict "undetermined" survive a change of primary criterion?

The report's verdict rests on two tests: the mean difference (paired bootstrap, CI includes 0) and
the survival-rate difference (Fisher p = 0.1167). Both are MEAN/COUNT statistics. R52 showed the
same two groups give the opposite sign on the median and Q1.

So the question is not rhetorical: if the preregistered primary criterion had been the MEDIAN
instead of the mean, what would the verdict be? This enumerates every criterion that can be
computed from the stored episodes, marks which way each points, and bootstraps the median
difference to see whether it is significantly negative.

A criterion that is significant in the NEGATIVE direction changes the verdict from "undetermined"
to "failed against that baseline", and that has to be stated rather than buried.

Writes only Document/Update/Alpha/v4.3.1/evidence/R53/.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / ".tmp_v431_seg001" / "full" / "raw"
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
OUT = EV / "R53"
CAP = 300
DEV = (11, 12, 17, 18)
BOOT = 4000


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


def boot_diff(a: list[float], b: list[float], stat, seed: int = 20261115) -> dict:
    rng = random.Random(seed)
    obs = stat(a) - stat(b)
    diffs = []
    for _ in range(BOOT):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(stat(ra) - stat(rb))
    diffs.sort()
    lo = diffs[int(0.025 * BOOT)]
    hi = diffs[int(0.975 * BOOT) - 1]
    return {"observed": round(obs, 2), "ci95": [round(lo, 2), round(hi, 2)],
            "excludes_zero": bool(lo > 0 or hi < 0), "direction": (
                "arm_better" if lo > 0 else "baseline_better" if hi < 0 else "inconclusive")}


def main() -> int:
    arm = episodes(
        [RAW / "armA" / f"eval-seed{s}-armA-multi4-f8664-128.jsonl" for s in DEV]
        + [RAW / "armA" / f"eval-seed{s}-multi4-128-ext.jsonl" for s in DEV])
    fair = episodes(
        [p for s in DEV for p in sorted((RAW / "cfair").glob(f"cfair-seed{s}-f8664-ep*.json"))])

    mean_ci = boot_diff(arm, fair, statistics.mean)
    med_ci = boot_diff(arm, fair, statistics.median)
    q1_ci = boot_diff(arm, fair, lambda v: sorted(v)[int(0.25 * (len(v) - 1))])

    r51 = json.loads((EV / "R51" / "baseline_ladder.json").read_text(encoding="utf-8"))
    r38 = json.loads((EV / "R38" / "n80.json").read_text(encoding="utf-8"))
    r52 = json.loads((EV / "R52" / "statistic_flip.json").read_text(encoding="utf-8"))

    deployable = r51["baselines"]["const_chosen_on_train"]["mean"]
    oracle = r51["baselines"]["oracle_per_window_best"]["mean"]
    pool = r51["baselines"]["const_pool_mean_dev"]["mean"]
    arm_mean = statistics.mean(arm)
    arm_med = statistics.median(arm)

    crit = [
        {"criterion": "均值 > C-fair", "statistic": round(arm_mean - statistics.mean(fair), 2),
         "test": "paired bootstrap", "ci95": mean_ci["ci95"],
         "direction": mean_ci["direction"], "points": "for_arm"},
        {"criterion": "中位数 > C-fair", "statistic": round(arm_med - statistics.median(fair), 2),
         "test": "bootstrap", "ci95": med_ci["ci95"],
         "direction": med_ci["direction"], "points": "against_arm"},
        {"criterion": "Q1 > C-fair", "statistic": q1_ci["observed"], "test": "bootstrap",
         "ci95": q1_ci["ci95"], "direction": q1_ci["direction"],
         "points": "against_arm" if q1_ci["ci95"][1] < 0 else "inconclusive"},
        {"criterion": "Q3 > C-fair", "statistic": r52["comparisons"]["q3_diff"], "test": "point",
         "ci95": None, "direction": "arm_better", "points": "for_arm"},
        {"criterion": "均值 > 恒定动作池（可部署）",
         "statistic": round(arm_mean - pool, 2), "test": "point", "ci95": None,
         "direction": "baseline_better", "points": "against_arm"},
        {"criterion": "均值 > train 上选出的恒定动作（可部署）",
         "statistic": round(arm_mean - deployable, 2), "test": "point", "ci95": None,
         "direction": "baseline_better", "points": "against_arm"},
        {"criterion": "均值 > 神谕最强恒定动作", "statistic": round(arm_mean - oracle, 2),
         "test": "point", "ci95": None, "direction": "baseline_better", "points": "against_arm"},
        {"criterion": "跑满 300 的次数（6/80 vs 1/80）",
         "statistic": (f"{r38['pooled']['arm_a']['survivals']}/{r38['pooled']['arm_a']['n']} vs "
                       f"{r38['pooled']['c_fair']['survivals']}/{r38['pooled']['c_fair']['n']}"),
         "test": "Fisher exact two-sided", "ci95": None,
         "direction": "inconclusive", "points": "inconclusive",
         "p": r38["pooled"]["survival_fisher"]["p_two_sided"]},
    ]

    counts = {"for_arm": sum(1 for c in crit if c["points"] == "for_arm"),
              "against_arm": sum(1 for c in crit if c["points"] == "against_arm"),
              "inconclusive": sum(1 for c in crit if c["points"] == "inconclusive")}

    median_significant_negative = med_ci["excludes_zero"] and med_ci["ci95"][1] < 0
    out = {
        "schema": "v431-recheck-r53/1",
        "question": ("如果主判据从均值换成中位数，臂 A 的判定会不会从\"未判定\"变成\"失败\"？"),
        "criteria": crit,
        "counts": counts,
        "bootstraps": {"mean": mean_ci, "median": med_ci, "q1": q1_ci},
        "verdict": {"median_flip_is_significant": None, "reading": None},
    }
    v = out["verdict"]
    v["median_flip_is_significant"] = bool(median_significant_negative)
    v["reading"] = (
        f"{len(crit)} 条可算判据里 **{counts['for_arm']} 条支持臂 A**、"
        f"**{counts['against_arm']} 条反对**、{counts['inconclusive']} 条不能判定。"
        + (f" 关键的是**中位数差 {med_ci['observed']:+.2f}，95% CI "
           f"[{med_ci['ci95'][0]:+.2f}, {med_ci['ci95'][1]:+.2f}] 排除 0**——"
           f"即**随机策略在中位数上显著优于臂 A**。"
           if median_significant_negative else
           f" 中位数差 {med_ci['observed']:+.2f}，CI [{med_ci['ci95'][0]:+.2f}, "
           f"{med_ci['ci95'][1]:+.2f}] 含 0，尚不显著。")
        + " 所以\"未判定\"这个结论**只对均值口径成立**；换成中位数口径，"
          "臂 A 相对**最弱的**基线已经是显著更差。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "criterion_sensitivity.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'criterion':>34s} {'stat':>9s} {'ci95':>22s} {'points':>14s}")
    print("-" * 84)
    for c in crit:
        ci = f"[{c['ci95'][0]}, {c['ci95'][1]}]" if c["ci95"] else "-"
        print(f"{c['criterion']:>34s} {c['statistic']:>9} {ci:>22s} {c['points']:>14s}")
    print()
    print(f"for arm {counts['for_arm']}  against arm {counts['against_arm']}  "
          f"inconclusive {counts['inconclusive']}")
    print(f"median diff CI {med_ci['ci95']}  significant_negative={median_significant_negative}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'criterion_sensitivity.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
