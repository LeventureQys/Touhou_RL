"""v4.3.1 recheck R7b: render the candidate segment so the "combat" claim is visual.

The R7 selection used an ink-density proxy for bullet load. An ink proxy cannot tell a
bullet curtain from a dark background with a large boss sprite, so this script writes
a montage of the actual play field for the candidate window and its neighbours.

Writes only new files under Document/Update/Alpha/v4.3.1/evidence/R7/.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
OBS = (REPO / ".tmp_v420_data" / "T02c" / "rebuild"
       / "d1-s0d1-n3300-seed12-shard" / "2f" / "obs" / "play96x112.npy")
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R7"


def main() -> int:
    # pick the 3 s / 5 s / 10 s windows R7 ranked first
    cand = json.loads((OUT / "segment_candidates.json").read_text(encoding="utf-8"))
    top = cand["top_windows"]
    picks = {}
    for want in (90, 150, 300):
        for c in top:
            if c["width_slots"] == want:
                picks[want] = c
                break

    obs = np.load(OBS, mmap_mode="r")
    print("picks:", {k: (v["start_slot"], v["end_slot"]) for k, v in picks.items()})

    from PIL import Image

    for width, c in picks.items():
        rows = []
        # 4 evenly spaced slots, each showing the 4 stacked frames as a column strip
        slots = np.linspace(c["start_slot"], c["end_slot"] - 1, 4).astype(int)
        strip_rows = []
        for s in slots:
            block = np.asarray(obs[s], dtype=np.uint8)  # (4,112,96)
            strip_rows.append(np.concatenate([block[i] for i in range(block.shape[0])], axis=1))
        montage = np.concatenate(strip_rows, axis=0)
        img = Image.fromarray(montage, mode="L")
        name = f"segment_{c['duration_s']:.0f}s_slot{c['start_slot']}_{c['end_slot']}.png"
        img.save(OUT / name)
        print(f"  wrote {name}  {img.size}  frames {c['start_frame']}..{c['end_frame']} "
              f"bullet%={c['mean_bullet_load_percentile']} cands={c['distinct_executed_cand']}")

    # a "quiet" control window for comparison: lowest bullet load, same width
    allw = cand["all_windows"]
    quiet = min((w for w in allw if w["width_slots"] == 150),
                key=lambda w: w["mean_bullet_load_percentile"])
    slots = np.linspace(quiet["start_slot"], quiet["end_slot"] - 1, 4).astype(int)
    strip_rows = []
    for s in slots:
        block = np.asarray(obs[s], dtype=np.uint8)
        strip_rows.append(np.concatenate([block[i] for i in range(block.shape[0])], axis=1))
    montage = np.concatenate(strip_rows, axis=0)
    Image.fromarray(montage, mode="L").save(OUT / "control_quiet_5s.png")
    print(f"  wrote control_quiet_5s.png slot {quiet['start_slot']}..{quiet['end_slot']} "
          f"bullet%={quiet['mean_bullet_load_percentile']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
