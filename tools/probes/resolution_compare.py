"""Compare what the vision model sees at different frame resolutions.

Why this exists
---------------
The frame tensor is 128x112 (1/3.4 of the 384x448 playfield).  That is a real
information decision, not a storage accident: a bullet that is 8-10 px across in
the game becomes 2-3 px in the tensor, and the player's hitbox (measured: 0.825 px
half-extent) becomes sub-pixel.  Whether that is enough depends on what the model
is asked to predict - it is fine for the 16x8 danger map, and questionable for
per-bullet geometry.

This tool renders the same real screenshot at several target resolutions, upscales
them back to the playfield size, and prints an edge-energy / contrast summary, so
the choice can be made on evidence instead of vibes - *before* spending another
20+ hours of real-time capture.

Usage::

    python tools/probes/resolution_compare.py --shot .../eval/vision-v1/episode-01.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare vision frame resolutions")
    parser.add_argument("--shot", required=True, help="full client-area screenshot (640x480)")
    parser.add_argument(
        "--sizes",
        default="384x448,192x224,160x186,128x112",
        help="comma separated HxW targets (first should be the native crop)",
    )
    parser.add_argument("--out", default=str(ROOT / "third_party/th08/build/rl/probe-v103/resolution-compare.png"))
    args = parser.parse_args(argv)

    from PIL import Image

    from touhou_rl import spec

    shot = np.asarray(Image.open(args.shot).convert("RGB"), dtype=np.uint8)
    x, y = int(spec.PLAYFIELD_X), int(spec.PLAYFIELD_Y)
    crop = shot[y : y + int(spec.PLAYFIELD_H), x : x + int(spec.PLAYFIELD_W)]

    tiles = []
    rows = []
    for spec_text in args.sizes.split(","):
        height, width = (int(value) for value in spec_text.lower().split("x"))
        small = np.asarray(Image.fromarray(crop).resize((width, height), Image.BOX))
        back = np.asarray(Image.fromarray(small).resize((crop.shape[1], crop.shape[0]), Image.NEAREST))
        tiles.append(back)
        gray = small.astype(np.float32).mean(axis=2)
        # Edge energy: how much local structure survives at this resolution.
        edge = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
        bright = gray > gray.mean() + 2.0 * gray.std()
        rows.append(
            {
                "target": f"{height}x{width}",
                "pixels": int(height * width),
                "bytes_per_frame": int(height * width * 3),
                "edge_energy": round(float(edge), 3),
                "bright_pixel_ratio": round(float(bright.mean()), 5),
                "bright_components_px": int(bright.sum()),
            }
        )
    sheet = np.concatenate(tiles, axis=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)
    print(json.dumps({"shot": args.shot, "sheet": str(out), "tiles": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
