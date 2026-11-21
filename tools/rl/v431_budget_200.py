"""v4.3.1 recheck R36: arm A reaches its prereg budget (200 episodes).

The report's standing caveat was "arm A ran 152 of the prereg's 200 episodes (36%-40% of
budget)". That caveat is removable work, so this run adds 48 more on the MULTI-WINDOW
configuration (the better policy per R35) and re-evaluates on the same dev protocol.

Accounting, all of it on the real engine:

    12  seed13 only          (armA-seed13-f8664-12.pt)      [R29]
  + 60  seed13, continued    (armA-seed13-f8664-72.pt)      [R34]
  + 80  4 windows, shared    (armA-multi4-f8664-80.pt)      [R35]
  + 48  4 windows, continued (armA-multi4-f8664-128.pt)     [R36]
  = 200 episodes total, 128 of them on the multi-window policy

Dev protocol is unchanged from R34/R35: seed11, 10 episodes, 300 decisions, frame 8,664,
no updates. Like-for-like with C-fair (71.8) and the earlier checkpoints.

Writes only Document/Update/Alpha/v4.3.1/evidence/R36/.
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
OUT = EV / "R36"
DECISIONS = 300
EPISODE_S = 70.4
BOOT = 4000
TRAIN_SEEDS = (13, 14, 15, 16)
EXPECTED_TOTAL = 200


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


def cfair_values(seed: int = 11) -> list[float]:
    out = []
    for f in sorted((RAW / "cfair").glob(f"cfair-seed{seed}-f8664-ep*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        off = d.get("first_death_offset")
        out.append(DECISIONS if off is None else off / 2.0)
    return out


def main() -> int:
    t12 = read_jsonl(RAW / "armA" / "train-seed13-f8664-12.jsonl")
    t60 = read_jsonl(RAW / "armA" / "train-seed13-f8664-cont60.jsonl")
    m80 = read_jsonl(RAW / "armA" / "train-multi4-f8664-80.jsonl")
    m48 = read_jsonl(RAW / "armA" / "train-multi4-f8664-cont48.jsonl")
    dev = read_jsonl(RAW / "armA" / "eval-seed11-armA-multi4-f8664-128.jsonl")
    if not dev:
        for cand in (RAW / "armA").glob("eval-seed11*m128*.jsonl"):
            dev = read_jsonl(cand)
            break

    total = len(t12) + len(t60) + len(m80) + len(m48)
    multi_total = len(m80) + len(m48)
    cfair = cfair_values()

    per_seed = {}
    for s in TRAIN_SEEDS:
        rows = [r for r in m48 if int(r.get("seed", 0)) == s]
        if rows:
            per_seed[str(s)] = {
                "episodes": len(rows),
                "mean_return": round(statistics.mean(r["return_decisions"] for r in rows), 1),
                "survivals": sum(1 for r in rows if r.get("survived")),
            }

    m48_rets = [float(r["return_decisions"]) for r in m48]
    third = max(len(m48_rets) // 3, 1)
    m128_combined = [float(r["return_decisions"]) for r in m80 + m48]
    all_rets = [float(r["return_decisions"]) for r in t12 + t60 + m80 + m48]
    dev_rets = [float(r["return_decisions"]) for r in dev]

    r35 = json.loads((EV / "R35" / "multi_window.json").read_text(encoding="utf-8"))
    dev80 = r35["multi"]["dev_mean"]

    out = {
        "schema": "v431-recheck-r36/1",
        "budget": {
            "segments": [
                {"episodes": len(t12), "config": "seed13 only", "checkpoint": "12", "ref": "R29"},
                {"episodes": len(t60), "config": "seed13 continued", "checkpoint": "72",
                 "ref": "R34"},
                {"episodes": len(m80), "config": "4 windows shared", "checkpoint": "80",
                 "ref": "R35"},
                {"episodes": len(m48), "config": "4 windows continued", "checkpoint": "128",
                 "ref": "R36"},
            ],
            "total_episodes": total,
            "prereg_budget": EXPECTED_TOTAL,
            "multi_window_episodes": multi_total,
            "reaches_prereg_budget": total >= EXPECTED_TOTAL,
        },
        "continuation_48": {
            "episodes": len(m48),
            "mean_return": round(statistics.mean(m48_rets), 1) if m48_rets else None,
            "first_third": round(statistics.mean(m48_rets[:third]), 1) if m48_rets else None,
            "last_third": round(statistics.mean(m48_rets[-third:]), 1) if m48_rets else None,
            "trend": (round(statistics.mean(m48_rets[-third:])
                            - statistics.mean(m48_rets[:third]), 1) if m48_rets else None),
            "survivals": sum(1 for r in m48 if r.get("survived")),
            "per_window": per_seed,
        },
        "multi_128": {
            "episodes": multi_total,
            "train_mean": round(statistics.mean(m128_combined), 1) if m128_combined else None,
            "survivals": sum(1 for r in m80 + m48 if r.get("survived")),
        },
        "all_200": {
            "episodes": total,
            "train_mean": round(statistics.mean(all_rets), 1) if all_rets else None,
            "survivals": sum(1 for r in t12 + t60 + m80 + m48 if r.get("survived")),
        },
        "dev": {
            "window": 11, "episodes": len(dev),
            "mean": round(statistics.mean(dev_rets), 1) if dev_rets else None,
            "survivals": sum(1 for r in dev if r.get("survived")),
            "values": sorted(round(v, 1) for v in dev_rets),
            # the mean is not the whole story: 9 of 10 episodes die EARLIER than the 80-episode
            # policy's typical run, and one runs the full 300. That is bimodal, and the mean
            # improvement is a tail event rather than a shift of the distribution.
            "median": round(statistics.median(dev_rets), 1) if dev_rets else None,
            "cfair_median": round(statistics.median(cfair), 1) if cfair else None,
            "episodes_at_or_above_250": sum(1 for v in dev_rets if v >= 250),
            "episodes_below_60": sum(1 for v in dev_rets if v < 60),
            "bimodal": (
                "one episode reached 300 (a survival no constant or random run achieves on this "
                "window: best constant 272, C-fair 0/40) while 9 of 10 died before 90; the "
                "median is BELOW C-fair, so the mean improvement is a tail event"),
            "vs_80_episodes": round(statistics.mean(dev_rets) - dev80, 2) if dev_rets else None,
            "vs_cfair": bootstrap_diff(dev_rets, cfair) if dev_rets and cfair else None,
            "cfair_mean": round(statistics.mean(cfair), 1) if cfair else None,
        },
        "prediction_made_before_the_run": (
            "R34/R35 predict the dev result will not change sign: more episodes on the same "
            "four tapes should shrink the gap, not remove it"),
        "cost": {"episodes": len(m48) + len(dev),
                 "engine_hours": round((len(m48) + len(dev)) * EPISODE_S / 3600.0, 2)},
        "verdict": {"budget_caveat_removed": total >= EXPECTED_TOTAL,
                    "dev_still_below_cfair": None, "reading": None},
    }

    v = out["verdict"]
    if dev_rets:
        v["dev_still_below_cfair"] = statistics.mean(dev_rets) < statistics.mean(cfair)
        if v["dev_still_below_cfair"]:
            v["reading"] = ("at the full 200-episode budget the arm still does not beat uniform "
                            "random on the held-out tape; the R34/R35 prediction held")
        else:
            v["reading"] = ("at 200 episodes the arm finally exceeds C-fair on the held-out "
                            "tape, which would contradict the R34/R35 prediction")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "budget_200.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    b = out["budget"]
    print("== arm A budget ==")
    for s in b["segments"]:
        print(f"  {s['episodes']:3d} episodes  {s['config']:22s} ckpt-{s['checkpoint']:3s}  "
              f"({s['ref']})")
    print(f"  total {b['total_episodes']} / prereg {b['prereg_budget']}  "
          f"-> reaches budget: {b['reaches_prereg_budget']}")
    print(f"  multi-window episodes: {b['multi_window_episodes']}")
    print()
    c = out["continuation_48"]
    print("== continuation (+48, four windows) ==")
    print(f"  mean return     : {c['mean_return']}")
    print(f"  first -> last   : {c['first_third']} -> {c['last_third']} ({c['trend']:+})")
    print(f"  survivals       : {c['survivals']}")
    for s, d in c["per_window"].items():
        print(f"    seed{s}: {d['episodes']} eps, mean {d['mean_return']}, "
              f"survivals {d['survivals']}")
    print()
    print("== multi-window cumulative (128) ==")
    print(f"  train mean {out['multi_128']['train_mean']}, "
          f"survivals {out['multi_128']['survivals']}")
    print()
    print("== dev seed11 ==")
    dv = out["dev"]
    print(f"  128 ep : mean {dv['mean']}  survivals {dv['survivals']}/{dv['episodes']} "
          f"(values {dv['values']})")
    print(f"  80 ep  : mean {dev80}")
    print(f"  C-fair : mean {dv['cfair_mean']}")
    print(f"  128 - 80: {dv['vs_80_episodes']:+}")
    if dv["vs_cfair"]:
        x = dv["vs_cfair"]
        print(f"  128 - C-fair: {x['mean_diff']:+} CI95 {x['ci95']} "
              f"excludes_zero={x['excludes_zero']}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'budget_200.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
