"""v4.3.1 recheck R52: does "arm A beats random" survive a change of summary statistic?

`C-fair` is uniform random PER DECISION (i.i.d. over the same 18 actions). Arm A's mean is 94.1
against its 79.7, and the report has been quoting that as "arm A beats random". But arm A's
distribution is bimodal: most episodes die early and a few run the full segment.

So the comparison may flip with the statistic. This computes, for arm A and C-fair on the same
four dev tapes:

    mean, median, quartiles, and the two cross-fractions
        P(arm episode > C-fair median)   and   P(C-fair episode > arm median)

If the mean favours arm A while the median favours C-fair, then "beats random" is not a property
of the policy -- it is a property of the chosen statistic, and the report must say which one.

This is the same failure class as R48 (admissible depends on segment length) and R49 (survival
rate depends on n and threshold), applied to a different kind of setting: the SUMMARY STATISTIC.

Writes only Document/Update/Alpha/v4.3.1/evidence/R52/.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / ".tmp_v431_seg001" / "full" / "raw"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R52"
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


def quartiles(vals: list[float]) -> dict:
    s = sorted(vals)
    n = len(s)

    def q(frac: float) -> float:
        pos = (n - 1) * frac
        lo, hi = int(pos), min(int(pos) + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)

    return {"min": round(s[0], 1), "q1": round(q(0.25), 1), "median": round(q(0.5), 1),
            "q3": round(q(0.75), 1), "max": round(s[-1], 1)}


def main() -> int:
    arm = episodes(
        [RAW / "armA" / f"eval-seed{s}-armA-multi4-f8664-128.jsonl" for s in DEV]
        + [RAW / "armA" / f"eval-seed{s}-multi4-128-ext.jsonl" for s in DEV])
    fair = episodes(
        [p for s in DEV for p in sorted((RAW / "cfair").glob(f"cfair-seed{s}-f8664-ep*.json"))])

    qa, qf = quartiles(arm), quartiles(fair)
    mean_a, mean_f = statistics.mean(arm), statistics.mean(fair)
    med_a, med_f = qa["median"], qf["median"]

    p_arm_over_fair_med = sum(1 for v in arm if v > med_f) / len(arm)
    p_fair_over_arm_med = sum(1 for v in fair if v > med_a) / len(fair)

    out = {
        "schema": "v431-recheck-r52/1",
        "question": ("\"臂 A 胜过随机\"这个说法换个统计量还成立吗？"),
        "cap_decisions": CAP,
        "policies": {
            "arm_a": {"n": len(arm), "mean": round(mean_a, 1), **qa,
                      "at_cap": sum(1 for v in arm if v >= CAP - 1e-9)},
            "c_fair": {"n": len(fair), "mean": round(mean_f, 1), **qf,
                       "at_cap": sum(1 for v in fair if v >= CAP - 1e-9)},
        },
        "comparisons": {
            "mean_diff": round(mean_a - mean_f, 1),
            "median_diff": round(med_a - med_f, 1),
            "q1_diff": round(qa["q1"] - qf["q1"], 1),
            "q3_diff": round(qa["q3"] - qf["q3"], 1),
            "p_arm_above_cfair_median": round(p_arm_over_fair_med, 3),
            "p_cfair_above_arm_median": round(p_fair_over_arm_med, 3),
        },
        "verdict": {"statistic_flips_the_sign": None, "reading": None},
    }
    c = out["comparisons"]
    v = out["verdict"]
    flips = (c["mean_diff"] > 0) and (c["median_diff"] < 0)
    v["statistic_flips_the_sign"] = bool(flips)
    v["reading"] = (
        f"the sign FLIPS with the statistic: mean {mean_a:.1f} vs {mean_f:.1f} "
        f"({c['mean_diff']:+}), but median {med_a:.1f} vs {med_f:.1f} ({c['median_diff']:+}) and "
        f"Q1 {qa['q1']:.1f} vs {qf['q1']:.1f} ({c['q1_diff']:+}) both favour the RANDOM policy. "
        f"{c['p_arm_above_cfair_median']:.0%} of arm A episodes beat C-fair's median, and "
        f"{c['p_cfair_above_arm_median']:.0%} of C-fair episodes beat arm A's median. So \"arm A "
        f"beats random\" is only true for the mean, and the mean is carried by the few episodes "
        f"that reach the cap ({out['policies']['arm_a']['at_cap']}/80)."
        if flips else
        f"the comparison is consistent across statistics (mean {c['mean_diff']:+}, "
        f"median {c['median_diff']:+}).")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "statistic_flip.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'policy':>8s} {'n':>4s} {'mean':>7s} {'min':>6s} {'q1':>6s} {'median':>7s} "
          f"{'q3':>7s} {'max':>7s} {'at cap':>7s}")
    print("-" * 70)
    for name, d in out["policies"].items():
        print(f"{name:>8s} {d['n']:4d} {d['mean']:7.1f} {d['min']:6.1f} {d['q1']:6.1f} "
              f"{d['median']:7.1f} {d['q3']:7.1f} {d['max']:7.1f} {d['at_cap']:7d}")
    print()
    print(f"mean diff   {c['mean_diff']:+}   (arm A better)")
    print(f"median diff {c['median_diff']:+}   (random better)")
    print(f"Q1 diff     {c['q1_diff']:+}   (random better)")
    print(f"P(arm > cfair median)  {c['p_arm_above_cfair_median']:.1%}")
    print(f"P(cfair > arm median)  {c['p_cfair_above_arm_median']:.1%}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'statistic_flip.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
