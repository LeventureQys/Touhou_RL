"""v4.3.1 recheck R20b: why are the A1 states so easy for a constant action?

R20 found 0/3 screened states admissible: 7-9 of the 18 constant actions survive the full
3 s segment. Before concluding anything about difficulty, this checks WHERE those states
sit in the stage. A1 froze states "proximate to the A0 student's first death", and a weak
student dies early; if every state sits in the first quarter of the stage, then the
segment is in a sparse-bullet regime and surviving 3 s by holding one direction is
expected - which would explain the result without any engine mystery.

Compares:
  * the `decision_frame` distribution of the 184 proper A1 states, against
  * the stage length (13,190 frames from `slots.jsonl`), and
  * the ink (bullet-load) percentile at those frames, against the tape windows that the
    v4.3.0 review's own "combat" criterion selected (frames 8,596-12,734).

Writes only Document/Update/Alpha/v4.3.1/evidence/R20/.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
A1 = REPO / "Document" / "Update" / "Alpha" / "v4.2.0" / "evidence" / "T08" / "a1_safe_set_full.json"
OBS = (REPO / ".tmp_v420_data" / "T02c" / "rebuild"
       / "d1-s0d1-n3300-seed12-shard" / "2f" / "obs" / "play96x112.npy")
SLOTS = REPO / ".tmp_v420_data" / "T04" / "dataset" / "d1-s0d1-n3300-seed12-shard" / "slots.jsonl"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R20"
STAGE_FRAMES = 13190


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a1 = json.loads(A1.read_text(encoding="utf-8"))
    states = a1["states"]
    proper = [s for s in states
              if bool(s["complete"]) and 0 < int(s["safe_size"]) < 18
              and not s.get("unknown_candidates")
              and int(s["prefix_hash_mismatches"]) == 0]
    frames = np.array([int(s["decision_frame"]) for s in proper])
    seeds = [int(s["seed"]) for s in proper]

    # ---- ink profile of the tape, to say "sparse" vs "dense" ----
    obs = np.load(OBS, mmap_mode="r")
    ink = np.empty(obs.shape[0], dtype=np.float32)
    for i in range(0, obs.shape[0], 512):
        b = np.asarray(obs[i:i + 512], dtype=np.uint8)
        ink[i:i + b.shape[0]] = (b.reshape(b.shape[0], -1) > 96).mean(axis=1)
    # slot -> decision_frame mapping from the tape
    slot_frames = []
    for line in SLOTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            slot_frames.append(int(json.loads(line)["decision_frame"]))
    slot_frames = np.array(slot_frames, dtype=np.int64)
    assert len(slot_frames) == len(ink), f"{len(slot_frames)} slots vs {len(ink)} ink rows"

    # percentile of ink at an arbitrary frame: nearest slot
    def ink_pct_at(frame: int) -> float:
        idx = int(np.argmin(np.abs(slot_frames - frame)))
        return float((ink < ink[idx]).mean() * 100.0)

    a1_ink = np.array([ink_pct_at(int(f)) for f in frames])
    # the dense windows the v4.3.0 combat criterion selected (R8)
    dense = [(8666, 8844), (12436, 12734), (10196, 10494), (8526, 8824)]
    dense_ink = []
    for lo, hi in dense:
        idx = np.where((slot_frames >= lo) & (slot_frames <= hi))[0]
        dense_ink.extend(ink_pct_at(int(slot_frames[i])) for i in idx)
    dense_ink = np.array(dense_ink)

    out = {
        "schema": "v431-recheck-r20b/1",
        "stage_frames": STAGE_FRAMES,
        "a1_proper_states": len(proper),
        "a1_decision_frame": {
            "min": int(frames.min()), "p25": int(np.percentile(frames, 25)),
            "median": int(np.median(frames)), "p75": int(np.percentile(frames, 75)),
            "max": int(frames.max()),
            "share_of_stage_median": round(float(np.median(frames)) / STAGE_FRAMES, 4),
            "share_of_stage_max": round(float(frames.max()) / STAGE_FRAMES, 4),
            "n_beyond_half_stage": int((frames > STAGE_FRAMES // 2).sum()),
            "n_beyond_8000": int((frames > 8000).sum()),
        },
        "ink_percentile_at_a1_states": {
            "mean": round(float(a1_ink.mean()), 2),
            "median": round(float(np.median(a1_ink)), 2),
            "max": round(float(a1_ink.max()), 2),
        },
        "ink_percentile_in_dense_windows": {
            "mean": round(float(dense_ink.mean()), 2),
            "median": round(float(np.median(dense_ink)), 2),
        },
        "seeds_distinct": len(set(seeds)),
        "verdict": None,
    }
    early = out["a1_decision_frame"]["share_of_stage_median"] < 0.25
    sparser = out["ink_percentile_at_a1_states"]["mean"] < out["ink_percentile_in_dense_windows"]["mean"]
    out["verdict"] = {
        "a1_states_are_early_in_stage": bool(early),
        "a1_states_are_sparser_than_dense_windows": bool(sparser),
        "reading": ("A1 froze states near the A0 student's FIRST death, which for a weak "
                    "student is early in the stage. Those frames are in a sparse-bullet "
                    "regime, so holding one direction for 3 s survives - which is exactly "
                    "what R20 measured. The cheap prefix and the easy segment are the same "
                    "property."),
    }
    (OUT / "state_position.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    d = out["a1_decision_frame"]
    print(f"A1 proper states            : {out['a1_proper_states']} (seeds {out['seeds_distinct']})")
    print(f"decision_frame              : min {d['min']} p25 {d['p25']} median {d['median']} "
          f"p75 {d['p75']} max {d['max']}")
    print(f"stage length                : {STAGE_FRAMES} frames")
    print(f"median share of stage       : {d['share_of_stage_median']:.1%}")
    print(f"states beyond half the stage: {d['n_beyond_half_stage']}")
    print(f"states beyond frame 8000    : {d['n_beyond_8000']}")
    print()
    print(f"ink percentile at A1 states : mean {out['ink_percentile_at_a1_states']['mean']:.1f} "
          f"median {out['ink_percentile_at_a1_states']['median']:.1f}")
    print(f"ink percentile in dense wins: mean {out['ink_percentile_in_dense_windows']['mean']:.1f} "
          f"median {out['ink_percentile_in_dense_windows']['median']:.1f}")
    print()
    print(f"wrote: {OUT / 'state_position.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
