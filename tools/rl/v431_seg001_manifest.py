"""v4.3.1 recheck R12: freeze the train/dev state manifest for SEG-001 rev B.

Anchoring SEG-001 on the A1 safe-set states means the "requires a combat decision" claim
is already engine-measured instead of inferred from an ink heuristic, and the prefix
replay drops from 86.7 s to 11.4 s per attempt.

This script writes the frozen manifest:

  * a seed-level split of the 50 A1 train seeds (40 train / 10 dev), so dev start points
    come from ANOTHER TRAJECTORY ANCESTOR rather than another offset of the same one;
  * the selected states, restricted to the 184 proper ones (0 < |S| < 18, all 18 arms
    judged), and to a bounded safe-set band so the task is neither trivial nor hopeless;
  * the safe set, the death set, the restoration hash and the prefix length for each.

Rules are fixed here, before any training run, and are not re-tuned afterwards.

Writes only new files under Document/Update/Alpha/v4.3.1/evidence/R12/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "Document" / "Update" / "Alpha" / "v4.2.0" / "evidence" / "T08" / "a1_safe_set_full.json"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R12"

# ---- frozen rules ----
TRAIN_SEED_MAX = 3081      # seeds 3042..3081 = 40 train
DEV_SEED_MIN = 3082        # seeds 3082..3091 = 10 dev (held out, never updated on)
SAFE_MIN, SAFE_MAX = 2, 8  # |S| band: random policy survives 11%-44% of the time
FPS = 121.7
SEGMENT_FRAMES = 180       # 3 s at 60 fps
CERTIFIED_FRAMES = 120     # the A1 horizon


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(SRC.read_text(encoding="utf-8"))
    states = d["states"]

    def proper(s: dict) -> bool:
        return (bool(s["complete"]) and int(s["safe_size"]) > 0
                and int(s["safe_size"]) < 18
                and not s.get("unknown_candidates")
                and int(s["prefix_hash_mismatches"]) == 0)

    def summarise(s: dict) -> dict:
        return {
            "index": int(s["index"]),
            "seed": int(s["seed"]),
            "ancestor": s["ancestor"],
            "decision_frame": int(s["decision_frame"]),
            "death_frame": int(s["death_frame"]),
            "offset_to_death": int(s["offset_to_death"]),
            "stratum": s["stratum"],
            "prefix_frames": int(s["prefix_frames"]),
            "expected_hash": s["expected_hash"],
            "recorded_action": int(s["recorded_action"]),
            "recorded_action_survives": s["outcomes"].get(str(s["recorded_action"])) == "engine_alive",
            "safe_size": int(s["safe_size"]),
            "safe_set": [int(x) for x in s["safe_set"]],
            "death_set": [int(x) for x in s["death_candidates"]],
            "prefix_replay_s": round(int(s["prefix_frames"]) / FPS, 1),
            "segment_frames": SEGMENT_FRAMES,
            "attempt_s": round((int(s["prefix_frames"]) + SEGMENT_FRAMES) / FPS, 1),
        }

    proper_states = [s for s in states if proper(s)]
    train_pool = [s for s in proper_states if int(s["seed"]) <= TRAIN_SEED_MAX]
    dev_pool = [s for s in proper_states if int(s["seed"]) >= DEV_SEED_MIN]
    train_sel = [s for s in train_pool if SAFE_MIN <= int(s["safe_size"]) <= SAFE_MAX]
    dev_sel = [s for s in dev_pool if SAFE_MIN <= int(s["safe_size"]) <= SAFE_MAX]

    out = {
        "schema": "v431-recheck-r12/1",
        "source": str(SRC.relative_to(REPO)),
        "frozen_rules": {
            "train_seeds": f"3042..{TRAIN_SEED_MAX}",
            "dev_seeds": f"{DEV_SEED_MIN}..3091",
            "safe_size_band": [SAFE_MIN, SAFE_MAX],
            "segment_frames": SEGMENT_FRAMES,
            "segment_seconds": round(SEGMENT_FRAMES / 60.0, 2),
            "certified_frames": CERTIFIED_FRAMES,
            "certified_seconds": round(CERTIFIED_FRAMES / 60.0, 2),
            "proper_state_definition": (
                "complete AND 0 < |S| < 18 AND no unknown arms AND prefix hash mismatches == 0"),
            "measured_fps": FPS,
        },
        "counts": {
            "states_total": len(states),
            "proper": len(proper_states),
            "train_pool": len(train_pool),
            "dev_pool": len(dev_pool),
            "train_selected": len(train_sel),
            "dev_selected": len(dev_sel),
            "train_seeds_selected": sorted({int(s["seed"]) for s in train_sel}),
            "dev_seeds_selected": sorted({int(s["seed"]) for s in dev_sel}),
        },
        "budget": {
            "attempt_s_train": (round(sum((int(s["prefix_frames"]) + SEGMENT_FRAMES) / FPS
                                           for s in train_sel) / len(train_sel), 1)
                                 if train_sel else None),
            "attempt_s_dev": (round(sum((int(s["prefix_frames"]) + SEGMENT_FRAMES) / FPS
                                         for s in dev_sel) / len(dev_sel), 1)
                               if dev_sel else None),
        },
        "train_states": [summarise(s) for s in train_sel],
        "dev_states": [summarise(s) for s in dev_sel],
    }

    # 200 episodes spread over the train states, plus dev evaluation cost
    if train_sel:
        per = out["budget"]["attempt_s_train"]
        out["budget"]["episode_cost_s"] = per
        out["budget"]["h_200_episodes_one_arm"] = round(200 * per / 3600, 2)
        out["budget"]["h_three_arms_200_each"] = round(3 * 200 * per / 3600, 2)
    if dev_sel:
        per_dev = out["budget"]["attempt_s_dev"]
        n_dev_attempts = 3 * 4 * len(dev_sel)   # arms x checkpoints x dev states
        out["budget"]["n_dev_attempts"] = n_dev_attempts
        out["budget"]["h_dev_evaluation"] = round(n_dev_attempts * per_dev / 3600, 2)

    (OUT / "seg001_state_manifest.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    c = out["counts"]
    print(f"states {c['states_total']}  proper {c['proper']}")
    print(f"train pool {c['train_pool']} -> selected {c['train_selected']} "
          f"across {len(c['train_seeds_selected'])} seeds")
    print(f"dev   pool {c['dev_pool']} -> selected {c['dev_selected']} "
          f"across {len(c['dev_seeds_selected'])} seeds")
    print(f"dev seeds: {c['dev_seeds_selected']}")
    print()
    print("== budget ==")
    for k, v in out["budget"].items():
        print(f"  {k} = {v}")
    print()
    print(f"{'idx':>4s} {'seed':>5s} {'frame':>6s} {'|S|':>4s} {'prefix_s':>9s} "
          f"{'attempt_s':>10s}  safe_set")
    print("-" * 88)
    for s in out["train_states"][:10]:
        print(f"{s['index']:4d} {s['seed']:5d} {s['decision_frame']:6d} {s['safe_size']:4d} "
              f"{s['prefix_replay_s']:9.1f} {s['attempt_s']:10.1f}  {s['safe_set']}")
    print("   ...")
    for s in out["dev_states"][:8]:
        print(f"{s['index']:4d} {s['seed']:5d} {s['decision_frame']:6d} {s['safe_size']:4d} "
              f"{s['prefix_replay_s']:9.1f} {s['attempt_s']:10.1f}  {s['safe_set']}")
    print()
    print(f"wrote: {OUT / 'seg001_state_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
