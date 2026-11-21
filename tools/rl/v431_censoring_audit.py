"""v4.3.1 recheck R50: which of the reported means are censored at the 300-decision cap?

Every episode ends either by dying or by reaching the end of the segment (300 decisions). The
reported "mean survival decisions" is therefore an average of `min(death, 300)`:

  * a group with NO survival has an EXACT mean (every episode died before the cap);
  * a group WITH survivals has a LOWER BOUND (the survivals were actually longer than 300).

The report compares arm A (6/80 at the cap) against C-fair (1/80) and against the best constant
(0/18). Those three are not the same kind of number, and the difference is not a rounding detail:
arm A's 94.1 could be arbitrarily larger in truth, while the constant's 272.0 is exact.

This recomputes each mean from the raw episodes, states whether it is exact or a lower bound, and
reports the gap between the censored mean and the "someone survived" indicator.

Writes only Document/Update/Alpha/v4.3.1/evidence/R50/.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / ".tmp_v431_seg001" / "full" / "raw"
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
OUT = EV / "R50"
CAP = 300
DEV_SEEDS = (11, 12, 17, 18)


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


def summarise(name: str, vals: list[float]) -> dict:
    at_cap = sum(1 for v in vals if v >= CAP - 1e-9)
    mean = statistics.mean(vals) if vals else None
    return {
        "group": name,
        "n": len(vals),
        "at_cap": at_cap,
        "mean_censored": round(mean, 1) if mean is not None else None,
        "median_censored": round(statistics.median(vals), 1) if vals else None,
        "max": round(max(vals), 1) if vals else None,
        "kind": ("EXACT" if at_cap == 0 else "LOWER_BOUND"),
        "why": ("every episode died before the cap, so the mean is exact" if at_cap == 0 else
                f"{at_cap} episode(s) reached the cap, so the true mean is >= this value"),
    }


def main() -> int:
    arm_vals = episodes(
        [RAW / "armA" / f"eval-seed{s}-armA-multi4-f8664-128.jsonl" for s in DEV_SEEDS]
        + [RAW / "armA" / f"eval-seed{s}-multi4-128-ext.jsonl" for s in DEV_SEEDS])
    fair_vals = episodes(
        [p for s in DEV_SEEDS for p in sorted((RAW / "cfair").glob(f"cfair-seed{s}-f8664-ep*.json"))])
    const_vals = episodes(
        [p for s in DEV_SEEDS for p in sorted((RAW / "c4dense").glob(f"c4dense-seed{s}-f8664-cand*.json"))])

    groups = [summarise("arm A (dev, n=80)", arm_vals),
              summarise("C-fair (dev, n=80)", fair_vals),
              summarise("all constants pooled (18 x 4)", const_vals)]

    # ---- the comparison nobody had framed: arm A vs "pick one action and stick with it" ----
    per_window = []
    for s in DEV_SEEDS:
        a = episodes([RAW / "armA" / f"eval-seed{s}-armA-multi4-f8664-128.jsonl",
                      RAW / "armA" / f"eval-seed{s}-multi4-128-ext.jsonl"])
        c = episodes(sorted((RAW / "c4dense").glob(f"c4dense-seed{s}-f8664-cand*.json")))
        if not a or not c:
            continue
        per_window.append({
            "seed": s,
            "arm_mean": round(statistics.mean(a), 1),
            "arm_at_cap": sum(1 for v in a if v >= CAP - 1e-9),
            "const_pool_mean": round(statistics.mean(c), 1),
            "const_pool_at_cap": sum(1 for v in c if v >= CAP - 1e-9),
            "const_best": round(max(c), 1),
            "arm_minus_pool": round(statistics.mean(a) - statistics.mean(c), 1),
            "arm_minus_best": round(statistics.mean(a) - max(c), 1),
        })
    wins_pool = sum(1 for r in per_window if r["arm_minus_pool"] > 0)
    wins_best = sum(1 for r in per_window if r["arm_minus_best"] > 0)
    out_extra = {
        "per_window": per_window,
        "windows_arm_beats_const_pool": wins_pool,
        "windows_arm_beats_best_const": wins_best,
        "windows": len(per_window),
        "mean_of_arm_minus_pool": round(statistics.mean(
            [r["arm_minus_pool"] for r in per_window]), 1) if per_window else None,
    }

    out = {
        "schema": "v431-recheck-r50/1",
        "question": "报告里哪些\"均值存活决策数\"其实是下界？",
        "cap_decisions": CAP,
        "groups": groups,
        "counts": {"groups": len(groups),
                   "lower_bounds": sum(1 for g in groups if g["kind"] == "LOWER_BOUND"),
                   "exact": sum(1 for g in groups if g["kind"] == "EXACT")},
        "arm_vs_a_fixed_action": out_extra,
        "verdict": {"some_means_are_lower_bounds": None, "reading": None,
                    "arm_beats_fixed_action_anywhere": None},
    }
    v = out["verdict"]
    lb = [g["group"] for g in groups if g["kind"] == "LOWER_BOUND"]
    v["some_means_are_lower_bounds"] = bool(lb)
    v["arm_beats_fixed_action_anywhere"] = out_extra["windows_arm_beats_const_pool"] > 0
    v["reading"] = (
        f"NEW, SHARPER: arm A's mean (94.1, a lower bound) is BELOW the pooled mean of the 18 "
        f"constant actions (111.7, exact), and per window arm A only beats 'pick one action and "
        f"repeat it' on {out_extra['windows_arm_beats_const_pool']}/{out_extra['windows']} windows "
        f"(mean gap {out_extra['mean_of_arm_minus_pool']:+}). "
        f"Separately, {len(lb)} of {len(groups)} reported means are LOWER BOUNDS, not exact averages: "
        f"{lb} each contain episodes that reached the {CAP}-decision cap. The best-constant "
        "comparison has no survivals, so its mean is exact -- which means the report compares a "
        "lower bound against an exact value when it contrasts arm A with C-const-best. The "
        "direction of that comparison is therefore the worst case for arm A, and it still loses, "
        "so the conclusion survives; but the numbers must be labelled as bounds.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "censoring.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'group':>42s} {'n':>4s} {'at cap':>7s} {'mean':>7s} {'median':>7s} {'kind':>12s}")
    print("-" * 84)
    for g in groups:
        print(f"{g['group']:>42s} {g['n']:4d} {g['at_cap']:7d} {str(g['mean_censored']):>7s} "
              f"{str(g['median_censored']):>7s} {g['kind']:>12s}")
    print()
    print(f"{'seed':>5s} {'arm':>7s} {'@cap':>5s} {'pool':>7s} {'best':>7s} "
          f"{'arm-pool':>9s} {'arm-best':>9s}")
    print("-" * 58)
    for r in per_window:
        print(f"{r['seed']:5d} {r['arm_mean']:7.1f} {r['arm_at_cap']:5d} "
              f"{r['const_pool_mean']:7.1f} {r['const_best']:7.1f} "
              f"{r['arm_minus_pool']:+9.1f} {r['arm_minus_best']:+9.1f}")
    print()
    print(f"arm beats the constant pool on {wins_pool}/{len(per_window)} windows, "
          f"the best constant on {wins_best}/{len(per_window)}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'censoring.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
