"""Dump frames from a vision shard as a PNG contact sheet (so a human can look).

Vision shards are ``.npz`` arrays, not video - which is great for training and
useless for "did it actually record the game?".  This probe renders the frames the
model sees (already cropped to the playfield and downscaled to 128x112) side by
side, and prints the privileged labels that accompany them.

Usage::

    python tools/probes/preview_vision_shard.py \
        --shard third_party/th08/build/rl/datasets/vision-all/ud0059-00010.npz \
        --indices 0,120,240,360 --out preview.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render frames from a vision shard to PNG")
    parser.add_argument("--shard", required=True)
    parser.add_argument("--indices", default="0,100,200,300", help="comma separated frame indices")
    parser.add_argument("--out", default="vision-preview.png")
    parser.add_argument("--scale", type=int, default=3, help="upscale factor for legibility")
    args = parser.parse_args(argv)

    from PIL import Image

    path = Path(args.shard)
    with np.load(path) as data:
        frames = data["frame"]
        self_feat = data["self_feat"]
        action = data["action"]
        meta = data["meta"]
        bullets = data["bullets"]
    indices = [int(value) for value in args.indices.split(",") if value.strip()]

    tiles = []
    for index in indices:
        index = max(0, min(index, frames.shape[0] - 1))
        image = np.asarray(frames[index], dtype=np.uint8)
        if args.scale > 1:
            image = np.repeat(np.repeat(image, args.scale, axis=0), args.scale, axis=1)
        tiles.append(image)
    sheet = np.concatenate(tiles, axis=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out)

    rows = []
    for index in indices:
        index = max(0, min(index, frames.shape[0] - 1))
        rows.append(
            {
                "index": index,
                "stage": int(meta[index, 1]),
                "game_frame": int(meta[index, 0]),
                "player_rel": [round(float(self_feat[index, 0]), 3), round(float(self_feat[index, 1]), 3)],
                "bullets_near": int(round(float(self_feat[index, 13]) * 700)),
                "action": [int(value) for value in action[index]],
                "bullets_in_view": int((bullets[index, :, 5] > 0.5).sum()),
            }
        )
    print(json.dumps({"shard": str(path), "frames": int(frames.shape[0]), "sheet": str(out), "rows": rows},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
