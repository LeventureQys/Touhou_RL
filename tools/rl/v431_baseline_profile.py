"""v4.3.1 recheck R31: the baseline profile — how hard is the task, really?

Three references exist for the same (window, 10 s segment) cell:

  * C-const-best : the strongest of the 18 constant actions, measured by the C4 sweep
                   (R24/R25/R28). This is the "one key held down" floor.
  * C-fair       : uniform over the 18 candidates, resampled every decision, n=10 per dev
                   window (R29 for seed11, this round for 12/17/18).
  * arm A        : the learned policy after 12 training episodes, dev seed11 only (R29).

The point is not to rank arms but to say what a *success* has to beat. If C-const-best
survives 259 decisions where the segment is 300, then a policy must beat a very strong
myopic baseline, not just random flailing.

Writes only Document/Update/Alpha/v4.3.1/evidence/R31/.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / ".tmp_v431_seg001" / "full" / "raw"
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
OUT = EV / "R31"
DECISIONS = 300
FPS = 60.0
DEV_SEEDS = (11, 12, 17, 18)
TRAIN_SEEDS = (13, 14, 15, 16)


def cfair_decisions(seed: int) -> list[float]:
    out = []
    for f in sorted((RAW / "cfair").glob(f"cfair-seed{seed}-f8664-ep*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        off = d.get("first_death_offset")
        out.append(DECISIONS if off is None else off / 2.0)
    return out


def main() -> int:
    screen = json.loads((EV / "R28" / "screen_split.json").read_text(encoding="utf-8"))
    const_by_seed = {r["seed"]: r["max_death_offset"] for r in screen["screen"]["rows"]}

    arm_records = []
    for p in sorted((RAW / "armA").glob("eval-seed11-f8664.jsonl")):
        arm_records = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()
                       if ln.strip()]
    arm = [float(r["return_decisions"]) for r in arm_records]

    rows = []
    for seed in DEV_SEEDS:
        fair = cfair_decisions(seed)
        const_frames = const_by_seed.get(seed)
        rows.append({
            "seed": seed,
            "side": "dev",
            "c_const_best_frames": const_frames,
            "c_const_best_decisions": (round(const_frames / 2.0, 1)
                                       if const_frames is not None else None),
            "c_fair_n": len(fair),
            "c_fair_survived": sum(1 for v in fair if v >= DECISIONS),
            "c_fair_mean_decisions": round(statistics.mean(fair), 1) if fair else None,
            "c_fair_median_decisions": round(statistics.median(fair), 1) if fair else None,
            "c_fair_max_decisions": round(max(fair), 1) if fair else None,
            "arm_a_mean_decisions": (round(statistics.mean(arm), 1) if seed == 11 and arm
                                     else None),
            "arm_a_n": len(arm) if seed == 11 else 0,
        })

    train_rows = []
    for seed in TRAIN_SEEDS:
        frames = const_by_seed.get(seed)
        train_rows.append({
            "seed": seed, "side": "train",
            "c_const_best_frames": frames,
            "c_const_best_decisions": (round(frames / 2.0, 1)
                                       if frames is not None else None),
        })

    all_fair = [v for s in DEV_SEEDS for v in cfair_decisions(s)]
    out = {
        "schema": "v431-recheck-r31/1",
        "cell": {"frame": 8664, "segment_decisions": DECISIONS,
                 "segment_frames": DECISIONS * 2, "segment_seconds": DECISIONS * 2 / FPS},
        "dev_windows": rows,
        "train_windows": train_rows,
        "c_fair_pooled": {
            "n": len(all_fair),
            "survived": sum(1 for v in all_fair if v >= DECISIONS),
            "mean_decisions": round(statistics.mean(all_fair), 1) if all_fair else None,
            "median_decisions": round(statistics.median(all_fair), 1) if all_fair else None,
            "max_decisions": round(max(all_fair), 1) if all_fair else None,
            "survival_rate": round(sum(1 for v in all_fair if v >= DECISIONS) / len(all_fair), 4)
            if all_fair else None,
        },
        "arm_a_dev11": {
            "n": len(arm),
            "survived": sum(1 for v in arm if v >= DECISIONS),
            "mean_decisions": round(statistics.mean(arm), 1) if arm else None,
            "note": "12 training episodes only; see R29 for the bootstrap comparison",
        },
        "difficulty_reading": {
            "best_constant_beats_random_by": None,
            "insight": None,
        },
        "step_batching": {
            "measured": True,
            "result": "no speed-up",
            "detail": ("group 1 = 7.655 ms/frame, group 5 = 7.718, group 25 = 7.837; "
                       "speedup 0.992x, so the cost is the engine simulating a frame, not "
                       "per-command overhead"),
            "consequence": ("the 70.4 s episode cost is already at the engine's simulation "
                            "floor (8,663 x 7.7 ms = 66.7 s); the only remaining lever is a "
                            "save-state"),
        },
    }

    best = max((r for r in rows if r["c_const_best_decisions"] is not None),
               key=lambda r: r["c_const_best_decisions"])
    fair_mean = out["c_fair_pooled"]["mean_decisions"]
    if fair_mean:
        out["difficulty_reading"]["best_constant_beats_random_by"] = round(
            best["c_const_best_decisions"] / fair_mean, 2)
    out["difficulty_reading"]["insight"] = (
        f"the strongest constant action survives {best['c_const_best_decisions']} decisions "
        f"(seed{best['seed']}) versus {fair_mean} for uniform random, i.e. "
        f"{out['difficulty_reading']['best_constant_beats_random_by']}x. A learned policy has "
        f"to beat THAT, not random, so 'beats C-fair' is a weak gate and 'beats "
        f"C-const-best' is the real one")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline_profile.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"cell: frame {out['cell']['frame']}, {DECISIONS} decisions "
          f"({out['cell']['segment_seconds']} s)")
    print()
    print(f"{'seed':>5s} {'side':>6s} {'const-best dec':>15s} {'fair n':>7s} "
          f"{'fair surv':>10s} {'fair mean':>10s} {'fair max':>9s} {'armA mean':>10s}")
    print("-" * 82)
    for r in rows:
        print(f"{r['seed']:5d} {r['side']:>6s} {str(r['c_const_best_decisions']):>15s} "
              f"{r['c_fair_n']:7d} {r['c_fair_survived']:10d} "
              f"{str(r['c_fair_mean_decisions']):>10s} {str(r['c_fair_max_decisions']):>9s} "
              f"{str(r['arm_a_mean_decisions']):>10s}")
    print()
    p = out["c_fair_pooled"]
    print(f"C-fair pooled : n={p['n']} survived={p['survived']} "
          f"mean={p['mean_decisions']} median={p['median_decisions']} max={p['max_decisions']}")
    print(f"difficulty    : {out['difficulty_reading']['insight']}")
    print()
    print(f"STEP batching : {out['step_batching']['detail']}")
    print(f"consequence   : {out['step_batching']['consequence']}")
    print()
    print(f"wrote: {OUT / 'baseline_profile.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
