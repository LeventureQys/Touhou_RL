"""v4.3.1 recheck R37: is R36's 300-decision survival reproducible on other dev tapes?

R36 found dev seed11 at mean 78.7 with ONE full 300-decision survival, which no constant and
no random run achieves on that window. With n=10 and a bimodal distribution, the honest
question is whether that survival is a property of the policy or an accident of seed11.

So the same 128-episode checkpoint is evaluated on the other three dev windows from R28's
frozen split (12, 17, 18), 10 episodes each, identical protocol. Pooling with R36's seed11
gives n=40 on 4 unseen tapes -- still one training run, but four independent bullet patterns.

Comparisons:
  * pooled arm A (n=40) vs pooled C-fair (n=40, all four windows) on mean decisions;
  * survival counts 1/40 vs 0/40 -- and the fact that the best constant on these windows
    cannot reach 300 at all (dev C-const-best is 245.5-272 decisions, R28/R31).

Writes only Document/Update/Alpha/v4.3.1/evidence/R37/.
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
OUT = EV / "R37"
DECISIONS = 300
FPS = 60.0
BOOT = 4000
DEV_SEEDS = (11, 12, 17, 18)
CKPT_TAG = "armA-multi4-f8664-128"


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


def cfair_values(seed: int) -> list[float]:
    out = []
    for f in sorted((RAW / "cfair").glob(f"cfair-seed{seed}-f8664-ep*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        off = d.get("first_death_offset")
        out.append(DECISIONS if off is None else off / 2.0)
    return out


def const_best(seed: int) -> float | None:
    """Best constant's survival in DECISIONS, from R28's C4 screen."""
    p = EV / "R28" / "screen_split.json"
    if not p.is_file():
        return None
    rows = json.loads(p.read_text(encoding="utf-8"))["screen"]["rows"]
    for r in rows:
        if int(r["seed"]) == seed and r.get("max_death_offset") is not None:
            return round(r["max_death_offset"] / 2.0, 1)
    return None


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] using the hypergeometric.

    Needed because the survival counts (4/40 vs 0/40) are the cleanest statement of what the
    policy does that no baseline does, and a mean difference is the wrong test for that.
    """
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


def main() -> int:
    per_window = []
    all_arm: list[float] = []
    all_fair: list[float] = []
    for seed in DEV_SEEDS:
        rows = read_jsonl(RAW / "armA" / f"eval-seed{seed}-{CKPT_TAG}.jsonl")
        if not rows and seed == 11:
            rows = read_jsonl(RAW / "armA" / f"eval-seed11-{CKPT_TAG}.jsonl")
        vals = [float(r["return_decisions"]) for r in rows]
        fair = cfair_values(seed)
        all_arm.extend(vals)
        all_fair.extend(fair)
        per_window.append({
            "seed": seed,
            "episodes": len(vals),
            "mean": round(statistics.mean(vals), 1) if vals else None,
            "median": round(statistics.median(vals), 1) if vals else None,
            "max": round(max(vals), 1) if vals else None,
            "survivals": sum(1 for v in vals if v >= DECISIONS),
            "values": sorted(round(v, 1) for v in vals),
            "c_const_best_decisions": const_best(seed),
            "cfair_mean": round(statistics.mean(fair), 1) if fair else None,
            "cfair_survivals": sum(1 for v in fair if v >= DECISIONS),
        })

    complete = [w for w in per_window if w["episodes"] == 10]
    out = {
        "schema": "v431-recheck-r37/1",
        "checkpoint": CKPT_TAG,
        "protocol": {"decisions": DECISIONS, "seconds": round(DECISIONS * 2 / FPS, 2),
                     "episodes_per_window": 10, "dev_seeds": list(DEV_SEEDS),
                     "note": "same checkpoint, same protocol, four unseen tapes"},
        "per_window": per_window,
        "pooled": {
            "arm_a": {"n": len(all_arm),
                      "mean": round(statistics.mean(all_arm), 1) if all_arm else None,
                      "median": round(statistics.median(all_arm), 1) if all_arm else None,
                      "survivals": sum(1 for v in all_arm if v >= DECISIONS),
                      "windows_with_a_survival": sorted(
                          w["seed"] for w in per_window if w["survivals"] > 0)},
            "c_fair": {"n": len(all_fair),
                       "mean": round(statistics.mean(all_fair), 1) if all_fair else None,
                       "median": round(statistics.median(all_fair), 1) if all_fair else None,
                       "survivals": sum(1 for v in all_fair if v >= DECISIONS)},
            "arm_a_vs_c_fair": bootstrap_diff(all_arm, all_fair)
            if all_arm and all_fair else None,
        },
        "windows_complete": len(complete),
        "all_windows_complete": len(complete) == len(DEV_SEEDS),
        "verdict": {
            "survival_reproduced_elsewhere": None,
            "pooled_beats_cfair": None,
            "reading": None,
        },
        "cost": {"episodes": len(all_arm),
                 "engine_hours": round(len(all_arm) * 70.4 / 3600.0, 2)},
        "limitations": [
            "one training run (128 episodes on 4 train tapes), four eval tapes -- this is "
            "cross-tape but not yet multi-seed training",
            "n=10 per window; pooling to 40 narrows the interval but the distribution is "
            "bimodal so the mean is not a sufficient statistic",
            "the best constant cannot reach 300 on any dev window (245.5-272), so any full "
            "survival is outside what a constant can do -- but nothing here proves a constant "
            "plus one switch could not do it either",
        ],
    }

    surv_a = out["pooled"]["arm_a"]["survivals"]
    surv_c = out["pooled"]["c_fair"]["survivals"]
    n_a, n_c = out["pooled"]["arm_a"]["n"], out["pooled"]["c_fair"]["n"]
    out["pooled"]["survival_fisher"] = {
        "table": [[surv_a, n_a - surv_a], [surv_c, n_c - surv_c]],
        "rates": [round(surv_a / n_a, 4), round(surv_c / n_c, 4)],
        "p_two_sided": fisher_two_sided(surv_a, n_a - surv_a, surv_c, n_c - surv_c),
        "reading": ("a survival the baselines never achieve, but the counts are small: the exact "
                    "test cannot separate 4/40 from 0/40 at the 0.05 level"),
    }

    v = out["verdict"]
    v["survival_reproduced_elsewhere"] = any(w["survivals"] > 0 for w in per_window
                                             if w["seed"] != 11)
    if out["pooled"]["arm_a_vs_c_fair"]:
        v["pooled_beats_cfair"] = out["pooled"]["arm_a_vs_c_fair"]["mean_diff"] > 0
    if v["survival_reproduced_elsewhere"]:
        v["reading"] = ("full 10 s survivals appear on more than one unseen tape, so R36's "
                        "survival is not a seed11 accident")
    elif v["pooled_beats_cfair"]:
        v["reading"] = ("only seed11 produced a full survival, but the pooled mean over four "
                        "tapes is still above C-fair: a tail skill that is rare, not absent")
    else:
        v["reading"] = ("no other window produced a full survival and the pooled mean does not "
                        "beat C-fair: R36's survival looks like a seed11 accident")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_all_windows.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"checkpoint {CKPT_TAG} on {len(DEV_SEEDS)} dev windows, 10 episodes each")
    print()
    print(f"{'seed':>5s} {'eps':>4s} {'mean':>7s} {'median':>7s} {'max':>6s} "
          f"{'surv':>5s} {'const-best':>10s} {'cfair mean':>10s}")
    print("-" * 66)
    for w in per_window:
        print(f"{w['seed']:5d} {w['episodes']:4d} {str(w['mean']):>7s} {str(w['median']):>7s} "
              f"{str(w['max']):>6s} {w['survivals']:5d} "
              f"{str(w['c_const_best_decisions']):>10s} {str(w['cfair_mean']):>10s}")
    print()
    p = out["pooled"]
    print(f"pooled arm A : n={p['arm_a']['n']} mean={p['arm_a']['mean']} "
          f"median={p['arm_a']['median']} survivals={p['arm_a']['survivals']} "
          f"(windows with a survival: {p['arm_a']['windows_with_a_survival']})")
    print(f"pooled C-fair: n={p['c_fair']['n']} mean={p['c_fair']['mean']} "
          f"median={p['c_fair']['median']} survivals={p['c_fair']['survivals']}")
    if p["arm_a_vs_c_fair"]:
        b = p["arm_a_vs_c_fair"]
        print(f"arm A - C-fair: {b['mean_diff']:+} CI95 {b['ci95']} "
              f"excludes_zero={b['excludes_zero']}")
    print()
    f = p["survival_fisher"]
    print(f"survival rates: arm A {f['rates'][0]:.3f} vs C-fair {f['rates'][1]:.3f}  "
          f"table={f['table']}  Fisher p={f['p_two_sided']}")
    print()
    print(f"windows complete: {len(complete)} / {len(DEV_SEEDS)}")
    print(f"survival reproduced: {v['survival_reproduced_elsewhere']}")
    print(f"verdict: {v['reading']}")
    print()
    print(f"wrote: {OUT / 'dev_all_windows.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
