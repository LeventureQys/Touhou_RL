"""v4.3.1 recheck R6b: does the ActorPolicy re-execution reproduce the saved numbers?

`cross_tape_generalisation.py` averaged per-head loss over *batches* of 128, so its CE is
not the per-sample mean when the last batch is short. This script prints both averages so
the documented 0.0293 / 1.8807 / 1.8837 can be attributed to the exact averaging rule.
"""

from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
P = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R6" / "actor_recheck.json"


def main() -> None:
    d = json.loads(P.read_text(encoding="utf-8"))
    doc = d["documented"]
    print("documented:", doc)
    print()
    hdr = f"{'set':14s} {'n':>5s} {'move batch-mean':>16s} {'move sample-mean':>17s} {'doc value':>10s}"
    print(hdr)
    print("-" * len(hdr))
    mapping = {"seed12_train": doc["train_move"], "seed12_val": doc["val_move"],
               "seed10_full": doc["seed10_move"]}
    for s, ev in d["evaluations"].items():
        print(f"{s:14s} {ev['n']:5d} {ev['batch_mean_ce']['move']:16.4f} "
              f"{ev['sample_mean_ce']['move']:17.4f} "
              f"{mapping.get(s, float('nan')):10.4f}")
    print()
    print("focus head, batch-mean vs sample-mean:")
    for s, ev in d["evaluations"].items():
        h = ev["heads"]["focus"]
        print(f"  {s:14s} model {ev['sample_mean_ce']['focus']:7.4f}  "
              f"baseline {h['baseline_ce_train_mle']:7.4f}  "
              f"delta {h['delta_vs_train_mle']:+7.4f}  "
              f"CI [{h['bootstrap']['ci95_low']:+7.4f},{h['bootstrap']['ci95_high']:+7.4f}]  "
              f"excludes0={h['bootstrap']['significant']}")
    print()
    print("accuracy vs majority accuracy (the CE/top-1 inversion):")
    for s, ev in d["evaluations"].items():
        h = ev["heads"]["move"]
        flag = ">" if h["model_accuracy"] > h["baseline_majority_accuracy"] else "<="
        print(f"  {s:14s} model {h['model_accuracy']:.4f} {flag} majority "
              f"{h['baseline_majority_accuracy']:.4f}")


if __name__ == "__main__":
    main()
