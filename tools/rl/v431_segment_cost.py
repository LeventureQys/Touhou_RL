"""v4.3.1 recheck R7c: cost-aware segment selection.

Restoring a segment start requires replaying the recorded masks from RESET to that
slot (`touhou_rl/v420/reslot.py` does exactly this). At the measured replay rate
(13,195 frames in 125.9 s wall -> ~104.8 frames/s, W01 REVIEWER_VERDICT), the prefix
replay cost is  start_slot * 2 / 104.8  seconds, which for a late-stage window
dominates the segment itself by an order of magnitude.

This script ranks the R7 candidates by (combat strength) / (prefix cost) so the
preregistration can name both a cheap window and the best-combat window.
"""

from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1].parents[0]
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R7"
FPS_MEASURED = 13195 / 125.9      # frames per wall second, from W01 seed12 replay
DECISION_FRAMES = 2


def main() -> None:
    d = json.loads((OUT / "segment_candidates.json").read_text(encoding="utf-8"))
    wins = d["all_windows"]

    def cost(slot: int) -> float:
        return slot * DECISION_FRAMES / FPS_MEASURED

    def ok(w):
        return (w["mean_bullet_load_percentile"] >= 60.0
                and w["distinct_executed_cand"] >= 4
                and w["slots_with_margin_below_8"] >= 1)

    good = [w for w in wins if ok(w)]
    for w in good:
        w["prefix_replay_s"] = round(cost(w["start_slot"]), 1)
        w["segment_play_s"] = round(w["width_slots"] * DECISION_FRAMES / FPS_MEASURED, 1)
        w["cost_ratio"] = round(w["prefix_replay_s"] / max(w["segment_play_s"], 1e-9), 1)

    by_combat = sorted(good, key=lambda w: -w["mean_bullet_load_percentile"])
    by_cost = sorted([w for w in good if w["width_slots"] == 150],
                     key=lambda w: w["prefix_replay_s"])

    print(f"measured replay rate: {FPS_MEASURED:.1f} frames/s wall")
    print(f"passing windows: {len(good)}")
    print()
    print("== best combat, any width ==")
    for w in by_combat[:6]:
        print(f"  slot {w['start_slot']:5d}+{w['width_slots']:3d} "
              f"({w['duration_s']:5.2f}s) bullet%={w['mean_bullet_load_percentile']:6.2f} "
              f"cands={w['distinct_executed_cand']:3d} prefix={w['prefix_replay_s']:7.1f}s "
              f"ratio={w['cost_ratio']:5.1f}x")
    print()
    print("== cheapest 5 s windows (width=150) ==")
    for w in by_cost[:6]:
        print(f"  slot {w['start_slot']:5d}+{w['width_slots']:3d} "
              f"({w['duration_s']:5.2f}s) bullet%={w['mean_bullet_load_percentile']:6.2f} "
              f"cands={w['distinct_executed_cand']:3d} prefix={w['prefix_replay_s']:7.1f}s "
              f"ratio={w['cost_ratio']:5.1f}x")
    print()
    print("== any passing window starting before slot 3000 ==")
    early = [w for w in good if w["start_slot"] < 3000]
    print(f"  count = {len(early)}")
    for w in sorted(early, key=lambda w: -w["mean_bullet_load_percentile"])[:6]:
        print(f"  slot {w['start_slot']:5d}+{w['width_slots']:3d} "
              f"({w['duration_s']:5.2f}s) bullet%={w['mean_bullet_load_percentile']:6.2f} "
              f"cands={w['distinct_executed_cand']:3d} prefix={w['prefix_replay_s']:7.1f}s")

    out = {
        "schema": "v431-recheck-r7c/1",
        "measured_replay_frames_per_s": round(FPS_MEASURED, 2),
        "measured_source": "W01 REVIEWER_VERDICT: seed12 replay 13,195 frames / 125.9 s wall",
        "decision_frames": DECISION_FRAMES,
        "n_passing": len(good),
        "best_combat": by_combat[:8],
        "cheapest_150": by_cost[:8],
        "early_candidates": sorted(early, key=lambda w: -w["mean_bullet_load_percentile"])[:8],
    }
    (OUT / "segment_cost_ranking.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print()
    print(f"wrote: {OUT / 'segment_cost_ranking.json'}")


if __name__ == "__main__":
    main()
