"""v4.3.1 recheck R8b: render the frozen R8 windows so the selection is visually checked.

The selection is only as good as the ink proxy. This writes one montage per frozen
window (4 moments x 4 stacked frames) so a human or a later agent can confirm the
window is real combat and not a dialogue, a menu, or a boss death animation.

Writes only new files under Document/Update/Alpha/v4.3.1/evidence/R8/.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOT = REPO / ".tmp_v420_data"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R8"


def obs_path(seed: int) -> pathlib.Path:
    return (ROOT / "T02c" / "rebuild" / f"d1-s0d1-n3300-seed{seed}-shard"
            / "2f" / "obs" / "play96x112.npy")


def main() -> int:
    spec = json.loads((OUT / "segment_spec_final.json").read_text(encoding="utf-8"))
    for role, seg in spec["selection"].items():
        if seg is None:
            continue
        obs = np.load(obs_path(seg["episode_seed"]), mmap_mode="r")
        slots = np.linspace(seg["start_slot"], seg["end_slot"] - 1, 4).astype(int)
        rows = []
        for s in slots:
            block = np.asarray(obs[s], dtype=np.uint8)     # (4,112,96)
            rows.append(np.concatenate([block[i] for i in range(block.shape[0])], axis=1))
        montage = np.concatenate(rows, axis=0)
        name = f"{role}_{seg['tape']}_{seg['duration_s']:.0f}s.png"
        Image.fromarray(montage, mode="L").save(OUT / name)
        print(f"{role:16s} slots {seg['start_slot']}..{seg['end_slot']} -> {name} "
              f"({montage.shape[1]}x{montage.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
