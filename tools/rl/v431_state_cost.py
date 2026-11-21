"""v4.3.1 recheck R11: cost and coverage profile of the A1 safe-set states.

The A1 pass froze 200 states and measured 18 first-action outcomes at each. That is a
much better anchor for a "segment that needs a combat decision" than an ink heuristic:
the decision-dependence is already engine-measured. This script profiles what it would
cost to use those states as online-learning start points, and how many independent
ancestors they give.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "Document" / "Update" / "Alpha" / "v4.2.0" / "evidence" / "T08" / "a1_safe_set_full.json"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R11"
FPS = 121.7          # measured engine frames/s from R10


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(SRC.read_text(encoding="utf-8"))
    states = d["states"]

    prefix = np.array([int(s["prefix_frames"]) for s in states])
    safe = np.array([int(s["safe_size"]) for s in states])
    seeds = np.array([int(s["seed"]) for s in states])
    offs = np.array([int(s["offset_to_death"]) for s in states])
    complete = np.array([bool(s["complete"]) for s in states])
    mism = np.array([int(s["prefix_hash_mismatches"]) for s in states])
    wall = np.array([float(s["state_wall_s"]) for s in states], dtype=np.float64)

    proper_mask = (safe > 0) & (safe < 18) & complete
    proper = [s for s, m in zip(states, proper_mask) if m]

    def prof(a: np.ndarray) -> dict:
        return {"min": int(a.min()), "p25": int(np.percentile(a, 25)),
                "median": int(np.median(a)), "p75": int(np.percentile(a, 75)),
                "max": int(a.max()), "mean": round(float(a.mean()), 1)}

    out = {
        "schema": "v431-recheck-r11/1",
        "source": str(SRC.relative_to(REPO)),
        "states": len(states),
        "proper_states": len(proper),
        "distinct_seeds": int(len(set(seeds.tolist()))),
        "seed_range": [int(seeds.min()), int(seeds.max())],
        "offsets_to_death": sorted(set(int(x) for x in offs)),
        "prefix_frames_profile": prof(prefix),
        "prefix_hash_mismatch_total": int(mism.sum()),
        "prefix_hash_mismatch_states": int((mism > 0).sum()),
        "safe_size_profile": prof(safe),
        "state_wall_s_profile": {"min": round(float(wall.min()), 1),
                                 "median": round(float(np.median(wall)), 1),
                                 "max": round(float(wall.max()), 1),
                                 "mean": round(float(wall.mean()), 1)},
        "proper_being_death_rich": {
            "definition": "share of the 18 first actions that end in engine_death",
            "min": None, "median": None, "max": None,
        },
        "cost_model": {},
    }

    # how many of the 18 actions die at a proper state
    share = []
    for s in proper:
        deaths = sum(1 for v in s["outcomes"].values() if v == "engine_death")
        share.append(deaths / 18.0)
    share = np.array(share)
    out["proper_being_death_rich"].update({
        "min": round(float(share.min()), 3),
        "median": round(float(np.median(share)), 3),
        "max": round(float(share.max()), 3),
        "mean": round(float(share.mean()), 3),
        "n_states_where_all_18_die": int((share == 1.0).sum()),
        "n_states_where_no_action_dies": int((share == 0.0).sum()),
    })

    # ---- cost model for an online attempt ----
    # prefix from the measured prefix_frames (median of proper states) + the segment
    med_prefix = int(np.median([int(s["prefix_frames"]) for s in proper]))
    for seg_s, seg_frames in ((2, 120), (3, 180), (5, 300), (10, 600)):
        per_attempt_s = (med_prefix + seg_frames) / FPS
        out["cost_model"][f"segment_{seg_s}s"] = {
            "prefix_frames": med_prefix,
            "segment_frames": seg_frames,
            "prefix_s": round(med_prefix / FPS, 1),
            "segment_s": round(seg_frames / FPS, 1),
            "attempt_s": round(per_attempt_s, 1),
            "prefix_share": round((med_prefix / FPS) / per_attempt_s, 3),
            "h_200_attempts": round(200 * per_attempt_s / 3600, 2),
            "h_3_arms_200_each": round(3 * 200 * per_attempt_s / 3600, 2),
        }

    (OUT / "state_cost_profile.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"states={out['states']} proper={out['proper_states']} "
          f"seeds={out['distinct_seeds']} ({out['seed_range'][0]}..{out['seed_range'][1]})")
    print(f"offsets_to_death={out['offsets_to_death']}")
    print(f"prefix_frames: {out['prefix_frames_profile']}")
    print(f"safe_size: {out['safe_size_profile']}")
    print(f"prefix hash mismatches: {out['prefix_hash_mismatch_total']} "
          f"over {out['prefix_hash_mismatch_states']} states")
    print(f"state_wall_s: {out['state_wall_s_profile']}")
    print()
    p = out["proper_being_death_rich"]
    print(f"death share of the 18 actions at proper states: "
          f"min {p['min']} median {p['median']} max {p['max']} mean {p['mean']}")
    print(f"  states where all 18 die: {p['n_states_where_all_18_die']}   "
          f"where none die: {p['n_states_where_no_action_dies']}")
    print()
    print("== online-attempt cost at the median proper-state prefix ==")
    for k, v in out["cost_model"].items():
        print(f"  {k:12s} prefix {v['prefix_s']:6.1f}s + segment {v['segment_s']:5.1f}s "
              f"= {v['attempt_s']:5.1f}s/attempt   prefix share {v['prefix_share']:.1%}   "
              f"200 attempts {v['h_200_attempts']:.2f} h   3 arms {v['h_3_arms_200_each']:.2f} h")
    print()
    print(f"wrote: {OUT / 'state_cost_profile.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
