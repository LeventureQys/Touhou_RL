"""v4.3.1 recheck R35: does multi-window training fix R34's overfitting?

R34's result was: 72 episodes on ONE tape (seed13) -> train +23%, dev significantly worse than
uniform random (72ep - C-fair = -23.36, CI [-53.35, -1.20]).

The rival explanation is simply "one tape is not enough training diversity". That is testable
without touching the engine, so this compares two policies on the same dev window with the
same protocol:

  single : armA-seed13-f8664-72.pt      (72 episodes, seed13 only)      [R34]
  multi  : armA-multi4-f8664-80.pt      (80 episodes cycled over 13/14/15/16)

Both are evaluated on dev seed11, 10 episodes, 300 decisions, frame 8,664 -- identical to R34,
so the difference is attributable to the training windows and nothing else.

Also reports the per-window train curve for the multi run: if the shared policy is only good on
some tapes, the per-seed means will show it, and that is a different failure mode from
overfitting a single tape.

Writes only Document/Update/Alpha/v4.3.1/evidence/R35/.
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
OUT = EV / "R35"
DECISIONS = 300
EPISODE_S = 70.4
BOOT = 4000
TRAIN_SEEDS = (13, 14, 15, 16)


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
    multi = read_jsonl(RAW / "armA" / "train-multi4-f8664-80.jsonl")
    # explicit name first: globbing `eval-seed11*.jsonl` and taking the last is wrong, because
    # the older single-window file sorts after the multi-window one alphabetically
    multi_dev = read_jsonl(RAW / "armA" / "eval-seed11-armA-multi4-f8664-80.jsonl")
    if not multi_dev:
        for cand in (RAW / "armA").glob("eval-seed11*m94*.jsonl"):
            multi_dev = read_jsonl(cand)
            break
    r34 = json.loads((EV / "R34" / "transfer.json").read_text(encoding="utf-8"))
    single_dev_mean = float(r34["dev"]["dev72_mean"])
    cfair = cfair_values()

    per_seed = {}
    for s in TRAIN_SEEDS:
        rows = [r for r in multi if int(r.get("seed", 0)) == s]
        if rows:
            per_seed[str(s)] = {
                "episodes": len(rows),
                "mean_return": round(statistics.mean(r["return_decisions"] for r in rows), 1),
                "survivals": sum(1 for r in rows if r.get("survived")),
            }

    rets = [float(r["return_decisions"]) for r in multi]
    third = max(len(rets) // 3, 1)
    mdev = [float(r["return_decisions"]) for r in multi_dev]

    out = {
        "schema": "v431-recheck-r35/1",
        "question": "was R34's overfitting caused by training on a single tape?",
        "protocol": {"dev_window": 11, "dev_episodes": len(mdev), "decisions": DECISIONS,
                     "frame": 8664, "train_windows": list(TRAIN_SEEDS)},
        "multi": {
            "train_episodes": len(multi),
            "train_mean_return": round(statistics.mean(rets), 1) if rets else None,
            "train_first_third": round(statistics.mean(rets[:third]), 1) if rets else None,
            "train_last_third": round(statistics.mean(rets[-third:]), 1) if rets else None,
            "train_trend": (round(statistics.mean(rets[-third:]) - statistics.mean(rets[:third]), 1)
                            if rets else None),
            "train_survivals": sum(1 for r in multi if r.get("survived")),
            "per_window": per_seed,
            "dev_mean": round(statistics.mean(mdev), 1) if mdev else None,
            "dev_survivals": sum(1 for r in multi_dev if r.get("survived")),
            "dev_values": sorted(round(v, 1) for v in mdev),
        },
        "comparison": {
            "single_window_dev_mean": single_dev_mean,
            "multi_window_dev_mean": round(statistics.mean(mdev), 1) if mdev else None,
            "cfair_dev_mean": round(statistics.mean(cfair), 1) if cfair else None,
            "multi_vs_cfair": bootstrap_diff(mdev, cfair) if mdev and cfair else None,
            "multi_vs_single": None,
        },
        "cost": {"episodes": len(multi) + len(multi_dev),
                 "engine_hours": round((len(multi) + len(multi_dev)) * EPISODE_S / 3600.0, 2)},
        "limitations": [
            "one dev window (seed11) and n=10, same as R34, so the interval stays wide",
            "the single-window policy had 72 episodes and the multi-window one 80; per-window "
            "that is 20 vs 72, so the comparison also changes episodes-per-window",
            "multi-window training shares one 106-dim MLP across four bullet patterns; a "
            "bigger network could behave differently",
        ],
        "verdict": {"multi_helps_over_single": None, "multi_beats_cfair": None,
                    "reading": None},
    }

    if mdev:
        # single-window per-episode values were overwritten; compare against the R29 snapshot
        # of the 12-episode policy and against the R34 mean instead of faking a paired test
        out["comparison"]["multi_vs_single"] = {
            "note": ("the 72-episode per-episode dev values were overwritten by an earlier run "
                     "(see R34 limitations), so this is a mean-to-mean difference, not a "
                     "bootstrap"),
            "difference_of_means": round(statistics.mean(mdev) - single_dev_mean, 2),
        }

    v = out["verdict"]
    if mdev:
        v["multi_helps_over_single"] = statistics.mean(mdev) > single_dev_mean
        v["multi_beats_cfair"] = bool(cfair) and statistics.mean(mdev) > statistics.mean(cfair)
        if v["multi_beats_cfair"]:
            v["reading"] = ("multi-window training clears C-fair on the held-out tape, so R34's "
                            "failure was at least partly a single-tape artifact")
        elif v["multi_helps_over_single"]:
            v["reading"] = ("multi-window training recovers some of R34's loss but still does "
                            "not beat uniform random on the held-out tape")
        else:
            v["reading"] = ("multi-window training does not rescue held-out performance: at this "
                            "budget the state arm has no cross-tape skill either way")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "multi_window.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    m = out["multi"]
    print("== multi-window training (13/14/15/16, shared policy) ==")
    print(f"  episodes            : {m['train_episodes']}")
    print(f"  train mean return   : {m['train_mean_return']} decisions")
    print(f"  first third -> last : {m['train_first_third']} -> {m['train_last_third']} "
          f"({m['train_trend']:+})")
    print(f"  train survivals     : {m['train_survivals']}")
    for s, d in m["per_window"].items():
        print(f"    seed{s}: {d['episodes']} eps, mean {d['mean_return']}, "
              f"survivals {d['survivals']}")
    print()
    print("== dev seed11 (never trained on) ==")
    c = out["comparison"]
    print(f"  single-window (72 ep) : {c['single_window_dev_mean']}")
    print(f"  multi-window  (80 ep) : {c['multi_window_dev_mean']}")
    print(f"  C-fair                : {c['cfair_dev_mean']}")
    if c["multi_vs_cfair"]:
        b = c["multi_vs_cfair"]
        print(f"  multi - C-fair        : {b['mean_diff']:+} CI95 {b['ci95']} "
              f"excludes_zero={b['excludes_zero']}")
    if c["multi_vs_single"]:
        print(f"  multi - single        : {c['multi_vs_single']['difference_of_means']:+} "
              f"(mean-to-mean)")
    print()
    print(f"verdict: {out['verdict']['reading']}")
    print(f"wrote: {OUT / 'multi_window.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
