"""v4.3.1 recheck R38: n=80 per arm — does the cross-tape survival become decidable?

R37 pooled 40 arm A episodes against 40 C-fair episodes on four unseen tapes:
  arm A  mean 104.8, median 49.5, 4/40 full survivals on 3 of 4 windows
  C-fair mean  76.2, median 59.5, 0/40
  mean difference +28.82, CI [-3.24, +63.48]   (spans zero, by 3.24)
  survival Fisher p = 0.1156

Training is finished (200 episodes), so the only missing quantity was evaluation sample size.
This run adds 10 more episodes per window per arm and redoes both tests at n=80.

Two honest outcomes:
  * Fisher reaches significance -> arm A has a demonstrated, if rare, cross-tape skill and the
    decision rule can finally be applied;
  * it does not -> the skill stays suggestive, and a much larger evaluation (or the fork-server)
    is needed before anyone may claim it.

Writes only Document/Update/Alpha/v4.3.1/evidence/R38/.
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
OUT = EV / "R38"
DECISIONS = 300
BOOT = 4000
DEV_SEEDS = (11, 12, 17, 18)
FIRST_TAG = "armA-multi4-f8664-128"
EXT_TAG = "multi4-128-ext"


def read_jsonl(p: pathlib.Path) -> list[dict]:
    if not p.is_file():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def bootstrap_diff(a: list[float], b: list[float], seed: int = 0) -> dict:
    rng = random.Random(seed)
    diffs = []
    for _ in range(BOOT):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(sum(ra) / len(ra) - sum(rb) / len(rb))
    diffs.sort()
    lo, hi = diffs[int(0.025 * BOOT)], diffs[int(0.975 * BOOT) - 1]
    return {"mean_diff": round(statistics.mean(diffs), 2),
            "ci95": [round(lo, 2), round(hi, 2)],
            "excludes_zero": bool(lo > 0 or hi < 0), "boot": BOOT}


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    from math import comb
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def prob(k: int) -> float:
        return comb(r1, k) * comb(n - r1, c1 - k) / comb(n, c1)

    observed = prob(a)
    total = 0.0
    for k in range(max(0, c1 - (n - r1)), min(r1, c1) + 1):
        pk = prob(k)
        if pk <= observed + 1e-12:
            total += pk
    return round(min(total, 1.0), 4)


def cfair_values(seed: int) -> list[float]:
    out = []
    for f in sorted((RAW / "cfair").glob(f"cfair-seed{seed}-f8664-ep*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        off = d.get("first_death_offset")
        out.append(DECISIONS if off is None else off / 2.0)
    return out


def arm_values(seed: int) -> tuple[list[float], list[float]]:
    first = [float(r["return_decisions"]) for r in
             read_jsonl(RAW / "armA" / f"eval-seed{seed}-{FIRST_TAG}.jsonl")]
    ext = [float(r["return_decisions"]) for r in
           read_jsonl(RAW / "armA" / f"eval-seed{seed}-{EXT_TAG}.jsonl")]
    return first, ext


def main() -> int:
    rows = []
    all_arm: list[float] = []
    all_fair: list[float] = []
    for seed in DEV_SEEDS:
        first, ext = arm_values(seed)
        arm = first + ext
        fair = cfair_values(seed)
        all_arm.extend(arm)
        all_fair.extend(fair)
        rows.append({
            "seed": seed,
            "arm_first_10": len(first), "arm_ext_10": len(ext),
            "arm_n": len(arm),
            "arm_mean": round(statistics.mean(arm), 1) if arm else None,
            "arm_median": round(statistics.median(arm), 1) if arm else None,
            "arm_survivals": sum(1 for v in arm if v >= DECISIONS),
            "arm_ext_mean": round(statistics.mean(ext), 1) if ext else None,
            "arm_ext_survivals": sum(1 for v in ext if v >= DECISIONS),
            "cfair_n": len(fair),
            "cfair_mean": round(statistics.mean(fair), 1) if fair else None,
            "cfair_survivals": sum(1 for v in fair if v >= DECISIONS),
        })

    r37 = json.loads((EV / "R37" / "dev_all_windows.json").read_text(encoding="utf-8"))
    out = {
        "schema": "v431-recheck-r38/1",
        "checkpoint": FIRST_TAG,
        "protocol": {"dev_seeds": list(DEV_SEEDS), "decisions": DECISIONS,
                     "episodes_per_window": 20, "updates": "none (eval only)"},
        "per_window": rows,
        "pooled": {
            "arm_a": {"n": len(all_arm),
                      "mean": round(statistics.mean(all_arm), 1) if all_arm else None,
                      "median": round(statistics.median(all_arm), 1) if all_arm else None,
                      "survivals": sum(1 for v in all_arm if v >= DECISIONS),
                      "windows_with_a_survival": sorted(
                          r["seed"] for r in rows if r["arm_survivals"] > 0)},
            "c_fair": {"n": len(all_fair),
                       "mean": round(statistics.mean(all_fair), 1) if all_fair else None,
                       "median": round(statistics.median(all_fair), 1) if all_fair else None,
                       "survivals": sum(1 for v in all_fair if v >= DECISIONS)},
            "arm_a_vs_c_fair": bootstrap_diff(all_arm, all_fair)
            if all_arm and all_fair else None,
        },
        "comparison_n40": {
            "arm_mean": r37["pooled"]["arm_a"]["mean"],
            "arm_survivals": r37["pooled"]["arm_a"]["survivals"],
            "cfair_mean": r37["pooled"]["c_fair"]["mean"],
            "mean_diff_ci": r37["pooled"]["arm_a_vs_c_fair"]["ci95"],
            "fisher_p": r37["pooled"]["survival_fisher"]["p_two_sided"],
        },
        "cost": {"episodes": len(all_arm) + len(all_fair),
                 "engine_hours": round((len(all_arm) + len(all_fair)) * 70.4 / 3600.0, 2)},
        "verdict": {"fisher_significant": None, "mean_significant": None,
                    "reading": None},
        "bug_found_and_fixed": {
            "symptom": ("the first attempt at the extension reproduced the first batch exactly: "
                        "seed11 extension mean 78.7 with 1/10 survivals, identical to the first "
                        "ten episodes"),
            "cause": ("the evaluation runner did not pass --policy-seed, so both batches drew "
                      "from the same torch RNG stream and sampled the SAME ten action sequences. "
                      "Those 40 episodes were duplicates, not new samples"),
            "why_it_matters": ("using them would have doubled the apparent sample size with "
                               "copies and produced a fake n=80 significance result"),
            "fix": ["pass --policy-seed 100 for the extension batch",
                    "guard in the runner: abort if the new batch equals the first value-for-"
                    "value"],
            "scope": ("R37's n=40 is unaffected: those 40 episodes come from four DIFFERENT "
                      "windows, so they are distinct even though each used policy-seed 0"),
            "how_it_was_caught": "the two batches reported exactly the same mean to one decimal",
        },
        "limitations": [
            "still one training run; this is a bigger evaluation, not a new seed of training",
            "the four dev windows are the only ones in R28's frozen split, so n grows by "
            "repeating windows rather than by adding tapes",
            "the distribution is bimodal (median below C-fair), so the mean is not a sufficient "
            "summary; both the mean and the survival rate are reported",
        ],
    }

    p = out["pooled"]
    surv_a, surv_c = p["arm_a"]["survivals"], p["c_fair"]["survivals"]
    n_a, n_c = p["arm_a"]["n"], p["c_fair"]["n"]
    p["survival_fisher"] = {
        "table": [[surv_a, n_a - surv_a], [surv_c, n_c - surv_c]],
        "rates": [round(surv_a / n_a, 4), round(surv_c / n_c, 4)],
        "p_two_sided": fisher_two_sided(surv_a, n_a - surv_a, surv_c, n_c - surv_c),
    }

    v = out["verdict"]
    v["fisher_significant"] = p["survival_fisher"]["p_two_sided"] < 0.05
    v["mean_significant"] = bool(p["arm_a_vs_c_fair"] and p["arm_a_vs_c_fair"]["excludes_zero"])
    if v["fisher_significant"]:
        v["reading"] = ("at n=80 per arm the survival difference is significant: arm A shows a "
                        "rare but real cross-tape skill that no constant and no random policy "
                        "reproduces")
    elif v["mean_significant"]:
        v["reading"] = ("the mean difference clears zero at n=80 while the survival counts do "
                        "not: the effect is real on average but still rare")
    else:
        v["reading"] = ("neither test clears at n=80: the skill stays suggestive and a much "
                        "larger evaluation (or the fork-server) is needed")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "n80.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'seed':>5s} {'arm n':>6s} {'arm mean':>9s} {'median':>7s} {'surv':>5s} "
          f"{'ext mean':>9s} {'ext surv':>9s} {'cfair n':>8s} {'cfair mean':>10s}")
    print("-" * 82)
    for r in rows:
        print(f"{r['seed']:5d} {r['arm_n']:6d} {str(r['arm_mean']):>9s} "
              f"{str(r['arm_median']):>7s} {r['arm_survivals']:5d} "
              f"{str(r['arm_ext_mean']):>9s} {r['arm_ext_survivals']:9d} "
              f"{r['cfair_n']:8d} {str(r['cfair_mean']):>10s}")
    print()
    print(f"pooled arm A : n={p['arm_a']['n']} mean={p['arm_a']['mean']} "
          f"median={p['arm_a']['median']} survivals={p['arm_a']['survivals']} "
          f"windows={p['arm_a']['windows_with_a_survival']}")
    print(f"pooled C-fair: n={p['c_fair']['n']} mean={p['c_fair']['mean']} "
          f"median={p['c_fair']['median']} survivals={p['c_fair']['survivals']}")
    if p["arm_a_vs_c_fair"]:
        b = p["arm_a_vs_c_fair"]
        print(f"arm A - C-fair: {b['mean_diff']:+} CI95 {b['ci95']} "
              f"excludes_zero={b['excludes_zero']}")
    f = p["survival_fisher"]
    print(f"survival rates: {f['rates'][0]:.3f} vs {f['rates'][1]:.3f}  "
          f"table={f['table']}  Fisher p={f['p_two_sided']}")
    print()
    print(f"n=40 was: arm {out['comparison_n40']['arm_mean']} "
          f"({out['comparison_n40']['arm_survivals']}/40), "
          f"cfair {out['comparison_n40']['cfair_mean']}, "
          f"Fisher p={out['comparison_n40']['fisher_p']}")
    print()
    print(f"verdict: {v['reading']}")
    if "bug_found_and_fixed" in out:
        print(f"bug note: {out['bug_found_and_fixed']['cause'][:80]}...")
    print(f"wrote: {OUT / 'n80.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
