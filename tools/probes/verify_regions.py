"""Region layout for TH08 screens and a tool to verify it against a screenshot.

Why regions
-----------
One monolithic 640x480 frame wastes most of its area on decoration, and any
downscale that keeps the playfield readable turns the HUD digits into mush.  The
HUD is not decoration: lives, bombs, score and power drive real decisions (bomb
now or die), so it must be recorded at 1:1 and fed to the model as its own stream.

TH08's layout (verified against a real 640x480 client-area screenshot):

* playfield  (32, 16, 384, 448)  - the action
* hud_score  (424, 36, 216, 34)  - HiScore + Score digits
* hud_status (424, 88, 216, 126) - Player/Spell stars, Power, Graze, Point, Time

``python tools/probes/verify_regions.py --shot .../episode-01.png`` renders every
region next to the source screenshot so the boxes can be checked by eye.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from touhou_rl import vision

ROOT = Path(__file__).resolve().parents[2]

REGIONS: tuple[tuple[str, vision.PlayfieldBox], ...] = (
    ("playfield", vision.PlayfieldBox(32, 16, 384, 448)),
    ("hud_score", vision.PlayfieldBox(424, 36, 216, 34)),
    ("hud_status", vision.PlayfieldBox(424, 88, 216, 126)),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify TH08 region boxes on a screenshot")
    parser.add_argument("--shot", required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--out", default=str(ROOT / "third_party/th08/build/rl/probe-v103/regions.png"))
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    shot = np.asarray(Image.open(args.shot).convert("RGB"), dtype=np.uint8)
    tiles = []
    rows = []
    for name, box in REGIONS:
        crop = vision.crop(shot, box)
        rows.append({"name": name, "box": [box.x, box.y, box.w, box.h], "crop_shape": list(crop.shape)})
        if args.scale > 1:
            crop = np.repeat(np.repeat(crop, args.scale, axis=0), args.scale, axis=1)
        tiles.append(crop)
    # right-align tiles into one sheet (different heights)
    height = max(tile.shape[0] for tile in tiles) + 2 * 8
    width = sum(tile.shape[1] for tile in tiles) + 8 * (len(tiles) + 1)
    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    x = 8
    for tile in tiles:
        sheet[8 : 8 + tile.shape[0], x : x + tile.shape[1]] = tile
        x += tile.shape[1] + 8
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    print(json.dumps({"shot": args.shot, "sheet": str(out), "regions": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
