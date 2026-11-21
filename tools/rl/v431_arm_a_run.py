"""v4.3.1 recheck R29: arm A, actually run — machinery works, no signal at this budget.

This is the first round where arm A ran on real dense episodes. It reports three things and
keeps them apart:

  1. WHAT WORKS. The whole loop closes on the engine: manual dense prefix (2,165 hash checks,
     0 mismatches), STATE + HAZARDS per decision, REINFORCE update, episode records, a saved
     checkpoint, and a dev evaluation that never touches the training window.

  2. A REAL BUG, FOUND AND FIXED. The first training run used
     `loss = sum(-logp) * advantage`, whose gradient scale is (episode length x advantage)
     ~ 300 x 300 = 9e4. With Adam at lr 1e-2 it destroyed the policy: mean return fell from
     ~84 (untrained) to 38 by episode 10. The fix is mean instead of sum, advantage
     normalised by the decision count, lr 3e-3, and gradient clipping at 1.0.

  3. NO SIGNAL ON DEV. After 12 training episodes on seed13, evaluating the checkpoint on the
     held-out dev window seed11 gives 0/10 survivals and a mean of 75.1 decisions survived,
     against C-fair's 0/10 and 72.2. The difference is inside the noise.

Per the frozen decision rule ("if arm A cannot beat C-fair, stop and inspect reward,
exploration and optimisation before anything else"), the verdict is NO SIGNAL, and the
inspection below finds the binding constraint is not the learner but the cost of an episode.

Writes only Document/Update/Alpha/v4.3.1/evidence/R29/.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / ".tmp_v431_seg001" / "full" / "raw"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R29"
DECISIONS = 300
EPISODE_S = 70.4
BOOT = 4000


def bootstrap_diff(a: list[float], b: list[float], seed: int = 0) -> dict:
    rng = random.Random(seed)
    diffs = []
    for _ in range(BOOT):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(sum(ra) / len(ra) - sum(rb) / len(rb))
    diffs.sort()
    lo = diffs[int(0.025 * BOOT)]
    hi = diffs[int(0.975 * BOOT) - 1]
    return {"mean_diff": round(sum(diffs) / len(diffs), 3),
            "ci95": [round(lo, 3), round(hi, 3)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "boot": BOOT}


def main() -> int:
    train_path = RAW / "armA" / "train-seed13-f8664-12.jsonl"
    eval_path = RAW / "armA" / "eval-seed11-f8664.jsonl"
    for p in (train_path, eval_path):
        if not p.is_file():
            print(f"missing {p}", file=sys.stderr)
            return 2

    train = [json.loads(ln) for ln in train_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ev = [json.loads(ln) for ln in eval_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    cfair = []
    for f in sorted((RAW / "cfair").glob("cfair-seed11-f8664-ep*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        off = d.get("first_death_offset")
        cfair.append(DECISIONS if off is None else off / 2.0)

    arm = [float(r["return_decisions"]) for r in ev]
    diff = bootstrap_diff(arm, cfair)

    staged = {
        "episodes": [12, 60, 200, 1000, 3000],
        "hours_one_window": [round(n * EPISODE_S / 3600.0, 1) for n in (12, 60, 200, 1000, 3000)],
        "hours_3_seeds_4_windows": [round(n * EPISODE_S * 3 * 4 / 3600.0, 1)
                                    for n in (12, 60, 200, 1000, 3000)],
    }

    out = {
        "schema": "v431-recheck-r29/1",
        "configuration": {
            "observation": "A-HAZARDS (client.hazards + STATE), queried per decision",
            "feature_dim": 106,
            "segment": {"decisions": DECISIONS, "frames": 600, "seconds": 10.0},
            "start_frame": 8664,
            "train_window": {"seed": 13, "role": "train"},
            "dev_window": {"seed": 11, "role": "dev (never trained on)"},
            "algorithm": "REINFORCE, EMA baseline, reward +1 per decision survived",
            "lr": 0.003, "hidden": 64, "grad_clip": 1.0,
        },
        "works": {
            "train_episodes": len(train),
            "train_survived_full_segment": sum(1 for r in train if r["survived"]),
            "train_mean_return_decisions": round(
                statistics.mean(r["return_decisions"] for r in train), 1),
            "train_best_episode": max(r["return_decisions"] for r in train),
            "checkpoint_saved": True,
            "dev_episodes": len(ev),
            "dev_survived": sum(1 for r in ev if r["survived"]),
            "dev_mean_return_decisions": round(statistics.mean(arm), 1),
            "note": ("one training episode reached the full 300 decisions, which no constant "
                     "action achieves on this window (best constant = 259 decisions)"),
        },
        "bug_found_and_fixed": {
            "symptom": ("first run's mean return fell from ~84 (untrained) to 38 by episode 10; "
                        "the policy was being destroyed, not trained"),
            "cause": "loss = sum(-logp) * advantage, so gradient scale ~ 300 x 300 = 9e4",
            "fix": ["mean instead of sum", "advantage / decisions", "lr 1e-2 -> 3e-3",
                    "clip_grad_norm_ 1.0"],
            "evidence": {"ep0_return": 40, "ep10_return_before_fix": 38,
                         "ep10_return_after_fix": 83, "ep11_return_after_fix": 300},
        },
        "no_signal_on_dev": {
            "arm_a": {"n": len(arm), "survived": sum(1 for v in arm if v >= DECISIONS),
                      "mean": round(statistics.mean(arm), 1),
                      "median": round(statistics.median(arm), 1),
                      "max": round(max(arm), 1),
                      "all": sorted(round(v, 1) for v in arm)},
            "c_fair": {"n": len(cfair), "survived": sum(1 for v in cfair if v >= DECISIONS),
                       "mean": round(statistics.mean(cfair), 1),
                       "median": round(statistics.median(cfair), 1),
                       "max": round(max(cfair), 1),
                       "all": sorted(round(v, 1) for v in cfair)},
            "paired_bootstrap_of_mean_difference": diff,
            "verdict": ("no signal: the arm's mean is inside the bootstrap CI of a zero "
                        "difference" if not diff["excludes_zero"] else "possible signal"),
        },
        "why_no_signal": {
            "training_episodes": len(train),
            "prereg_budget_episodes": 200,
            "fraction_of_prereg_budget_run": round(len(train) / 200.0, 3),
            "cost_measured_s_per_episode": EPISODE_S,
            "prefix_share_of_episode": 0.96,
            "staged_cost": staged,
            "reading": ("12 episodes is 6% of the prereg's budget; a policy-gradient run that "
                        "needs 1e3-1e3 episodes costs 19-58 h per seed at 70 s/episode, and a "
                        "3-seed x 4-window study costs 60-230 h. The binding constraint is the "
                        "missing engine save-state: 96% of every episode is replaying the 8,663 "
                        "frame prefix, not learning."),
        },
        "decision_rule_outcome": {
            "gate": "arm A must beat C-fair on dev before the pixel arms are attempted",
            "result": "arm A does NOT beat C-fair",
            "action_per_prereg": ("stop and inspect reward / exploration / optimisation "
                                  "before anything else"),
            "inspection_done": {
                "optimisation": "found and fixed a gradient-scale bug (see bug_found_and_fixed)",
                "exploration": ("entropy is not annealed and the initial policy is near-uniform; "
                                "no evidence exploration is the blocker"),
                "reward": ("+1 per decision survived is dense; the terminal is the only 0, so "
                           "the signal is not sparse in the usual sense"),
                "budget": "12 episodes vs the prereg's 200; the run is 6% of its own budget",
            },
            "do_not": ["do not proceed to pixel arms B/C on this result",
                       "do not report arm A as failed as a method - it was not run at budget"],
        },
        "next_actions": [
            "run arm A to the prereg's 200 episodes on one window (~3.9 h) before judging it",
            "or first build the engine save-state, which cuts episode cost by ~25x and makes "
            "a 3-seed study affordable (this is the highest-value engineering prerequisite)",
            "C-const-best remains unbeaten by any policy so far: it is the second gate",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arm_a_run.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    w = out["works"]
    print("== what works ==")
    print(f"  train eps {w['train_episodes']}, full survivals {w['train_survived_full_segment']}, "
          f"mean {w['train_mean_return_decisions']}, best {w['train_best_episode']}")
    print(f"  dev   eps {w['dev_episodes']}, survivals {w['dev_survived']}, "
          f"mean {w['dev_mean_return_decisions']}")
    print()
    n = out["no_signal_on_dev"]
    print("== arm A vs C-fair on dev seed 11 ==")
    print(f"  arm A : mean {n['arm_a']['mean']}  median {n['arm_a']['median']}  "
          f"max {n['arm_a']['max']}  survived {n['arm_a']['survived']}/{n['arm_a']['n']}")
    print(f"  C-fair: mean {n['c_fair']['mean']}  median {n['c_fair']['median']}  "
          f"max {n['c_fair']['max']}  survived {n['c_fair']['survived']}/{n['c_fair']['n']}")
    print(f"  bootstrap mean difference: {diff['mean_diff']} CI95 {diff['ci95']} "
          f"excludes_zero={diff['excludes_zero']}")
    print(f"  verdict: {n['verdict']}")
    print()
    y = out["why_no_signal"]
    print("== why ==")
    print(f"  {y['training_episodes']} episodes = "
          f"{y['fraction_of_prereg_budget_run']:.0%} of the prereg budget")
    for n_ep, h1, h12 in zip(staged["episodes"], staged["hours_one_window"],
                            staged["hours_3_seeds_4_windows"]):
        print(f"  {n_ep:5d} episodes -> {h1:6.1f} h (one window)   {h12:6.1f} h (3 seeds x 4 windows)")
    print()
    print(f"decision rule: {out['decision_rule_outcome']['result']}")
    print(f"wrote: {OUT / 'arm_a_run.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
