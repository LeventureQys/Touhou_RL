"""v4.3.1 recheck R34: arm A improves on the TRAIN window and degrades on DEV.

This is the first result in the whole review where the train/dev split actually decides
something, so it is recorded carefully and with its limitations.

What was run (all on the real engine, 10 s segments, frame 8,664):

  train  seed13, 12 episodes  -> checkpoint A12
  train  seed13, +60 episodes from A12 -> checkpoint A72   (72 cumulative episodes)
  dev    seed11, 10 episodes with A12
  dev    seed11, 10 episodes with A72
  base   seed11, C-fair (uniform random), 10 episodes       [R29/R31]

Measured:

  train, first 20 of the 60  : mean return 87.7 decisions
  train, last 20 of the 60   : mean return 108.3 decisions   (+20.6, +23%)
  train, full 300-decision survivals: 4 / 60  (no constant action can do this on seed13,
                                     where the best constant dies at 259 decisions)
  dev at 12 episodes         : mean 75.1 decisions, 0/10 survivals
  dev at 72 episodes         : mean 48.5 decisions, 0/10 survivals
  dev C-fair                 : mean 71.8 decisions, 0/10 survivals

So the policy learned something that pays on the window it trained on and COSTS on the
held-out window. That is overfitting to one tape's bullet pattern, not a dodging skill.

Writes only Document/Update/Alpha/v4.3.1/evidence/R34/.
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
OUT = EV / "R34"
DECISIONS = 300
EPISODE_S = 70.4
BOOT = 4000


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
    return {"mean_diff": round(statistics.mean(diffs), 2),
            "ci95": [round(diffs[int(0.025 * BOOT)], 2),
                     round(diffs[int(0.975 * BOOT) - 1], 2)],
            "excludes_zero": bool(diffs[int(0.025 * BOOT)] > 0
                                  or diffs[int(0.975 * BOOT) - 1] < 0),
            "boot": BOOT}


def cfair_values() -> list[float]:
    out = []
    for f in sorted((RAW / "cfair").glob("cfair-seed11-f8664-ep*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        off = d.get("first_death_offset")
        out.append(DECISIONS if off is None else off / 2.0)
    return out


def main() -> int:
    cont = read_jsonl(RAW / "armA" / "train-seed13-f8664-cont60.jsonl")
    first = read_jsonl(RAW / "armA" / "train-seed13-f8664-12.jsonl")
    # The 72-episode eval overwrote the 12-episode raw file (same output path), so the 12-ep
    # per-episode values are recovered from R29's frozen evidence instead. That is a real
    # process defect and is recorded in `limitations`.
    r29 = json.loads((EV / "R29" / "arm_a_run.json").read_text(encoding="utf-8"))
    dev12_values = [float(v) for v in r29["no_signal_on_dev"]["arm_a"]["all"]]
    dev12_surv = int(r29["no_signal_on_dev"]["arm_a"]["survived"])
    dev72_rows = read_jsonl(RAW / "armA" / "eval-seed11-f8664.jsonl")
    cfair = cfair_values()

    def rets(rows: list[dict]) -> list[float]:
        return [float(r["return_decisions"]) for r in rows]

    t12, t60 = rets(first), rets(cont)
    n20 = max(len(t60) // 3, 1)
    train_stats = {
        "episodes_pretrain_12": len(t12),
        "episodes_continuation_60": len(t60),
        "episodes_total": len(t12) + len(t60),
        "first_20_mean": round(statistics.mean(t60[:n20]), 1) if t60 else None,
        "last_20_mean": round(statistics.mean(t60[-n20:]), 1) if t60 else None,
        "delta": (round(statistics.mean(t60[-n20:]) - statistics.mean(t60[:n20]), 1)
                  if t60 else None),
        "full_survivals_60": sum(1 for r in cont if r.get("survived")),
        "full_survivals_12": sum(1 for r in first if r.get("survived")),
        "best_constant_decisions_seed13": 259,
        "note": ("no constant action survives 300 decisions on seed13 (best constant dies at "
                 "259), so these survivals are not something a constant can produce"),
    }

    d12 = dev12_values
    d72 = rets(dev72_rows)
    dev_stats = {
        "episodes": len(d72),
        "dev12_mean": round(statistics.mean(d12), 1) if d12 else None,
        "dev12_survivals": dev12_surv,
        "dev72_mean": round(statistics.mean(d72), 1) if d72 else None,
        "dev72_survivals": sum(1 for r in dev72_rows if r.get("survived")),
        "cfair_mean": round(statistics.mean(cfair), 1) if cfair else None,
        "cfair_survivals": sum(1 for v in cfair if v >= DECISIONS),
        "dev72_vs_cfair": bootstrap_diff(d72, cfair) if d72 and cfair else None,
        "dev72_vs_dev12": bootstrap_diff(d72, d12) if d72 and d12 else None,
        "dev_values": sorted(round(v, 1) for v in d72),
        "cfair_values": sorted(round(v, 1) for v in cfair),
    }

    transferred = (dev_stats["dev72_mean"] is not None
                   and dev_stats["cfair_mean"] is not None
                   and dev_stats["dev72_mean"] > dev_stats["cfair_mean"])
    out = {
        "schema": "v431-recheck-r34/1",
        "cell": {"frame": 8664, "decisions": DECISIONS, "seconds": 10.0,
                 "train_window": {"seed": 13}, "dev_window": {"seed": 11}},
        "train": train_stats,
        "dev": dev_stats,
        "verdict": {
            "train_improved": (train_stats["delta"] or 0) > 0,
            "train_full_survivals": train_stats["full_survivals_60"],
            "dev_transferred": transferred,
            "dev_degraded_vs_12ep": (dev_stats["dev72_mean"] is not None
                                     and dev_stats["dev12_mean"] is not None
                                     and dev_stats["dev72_mean"] < dev_stats["dev12_mean"]),
            "reading": None,
        },
        "cost": {
            "episodes": len(t12) + len(t60) + len(d12) + len(d72) + len(cfair),
            "engine_hours": round((len(t12) + len(t60) + len(d12) + len(d72)) * EPISODE_S
                                  / 3600.0, 2),
        },
        "limitations": [
            "one train window and one dev window: this is a transfer test, not a seed study",
            f"{len(t12) + len(t60)} cumulative training episodes is 36% of the prereg's 200",
            "dev n=10 gives a wide interval; the sign of the dev change is the robust part",
            "the overfitting reading assumes seed11 and seed13 differ mainly by bullet RNG, "
            "which R2/R3 established for the three original tapes but not specifically here",
            "the eval runner reused one output filename, so the 12-episode raw dev file was "
            "overwritten by the 72-episode run; the 12-episode values are taken from R29's "
            "frozen evidence rather than from the raw file",
        ],
    }
    v = out["verdict"]
    if v["train_improved"] and not v["dev_transferred"]:
        v["reading"] = ("the arm learned something that pays on the training tape and costs on "
                        "the held-out tape: overfitting to one bullet pattern, not a general "
                        "dodging skill")
    elif v["train_improved"] and v["dev_transferred"]:
        v["reading"] = "train and dev both improved: possible transfer, needs more seeds"
    else:
        v["reading"] = "no train-side improvement either; the run is inconclusive"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "transfer.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("== train (seed13) ==")
    print(f"  cumulative episodes        : {train_stats['episodes_total']} "
          f"(12 pretrain + {train_stats['episodes_continuation_60']})")
    print(f"  first 20 mean              : {train_stats['first_20_mean']} decisions")
    print(f"  last 20 mean               : {train_stats['last_20_mean']} decisions "
          f"({train_stats['delta']:+})")
    print(f"  full 300-decision survivals: {train_stats['full_survivals_60']} / "
          f"{train_stats['episodes_continuation_60']}  "
          f"(best constant on seed13 dies at {train_stats['best_constant_decisions_seed13']})")
    print()
    print("== dev (seed11, never trained on) ==")
    print(f"  at 12 episodes : mean {dev_stats['dev12_mean']}  "
          f"survivals {dev_stats['dev12_survivals']}/{len(d12)}")
    print(f"  at 72 episodes : mean {dev_stats['dev72_mean']}  "
          f"survivals {dev_stats['dev72_survivals']}/{len(d72)}")
    print(f"  C-fair         : mean {dev_stats['cfair_mean']}  "
          f"survivals {dev_stats['cfair_survivals']}/{len(cfair)}")
    for name, key in (("dev72 - C-fair", "dev72_vs_cfair"),
                      ("dev72 - dev12 ", "dev72_vs_dev12")):
        b = dev_stats[key]
        if b:
            print(f"  {name}: {b['mean_diff']:+} CI95 {b['ci95']} "
                  f"excludes_zero={b['excludes_zero']}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"cost   : {out['cost']['episodes']} episodes, "
          f"{out['cost']['engine_hours']} engine hours")
    print()
    print(f"wrote: {OUT / 'transfer.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
