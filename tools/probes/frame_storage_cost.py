"""Measure lossless storage cost of a *native* resolution TH08 frame.

Why: the project currently stores 128x112 frames because that fit a guessed disk
budget.  The real question is "what does native cost?", and the answer depends on
the encoder, not just the pixel count - ``np.savez_compressed`` deflates raw RGB
without image-specific filtering, which is close to worst case for a game frame.
This probe compares, on real screenshots:

* playfield crop 384x448 and full client area 640x480;
* raw ``npz`` (deflate) vs lossless PNG vs WebP;
* decode cost per frame (a training loader has to pay it).

Usage::

    python tools/probes/frame_storage_cost.py --shot .../eval/vision-v1/episode-01.png
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _npz_size(frame: np.ndarray) -> int:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, frame=frame[None, ...])
    return buffer.getbuffer().nbytes


def _png_size(frame: np.ndarray) -> int:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getbuffer().nbytes


def _webp_size(frame: np.ndarray) -> int:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="WEBP", lossless=True, quality=100, method=4)
    return buffer.getbuffer().nbytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure native-resolution frame storage cost")
    parser.add_argument("--shot", action="append", required=True, help="640x480 client-area screenshot")
    parser.add_argument("--samples", type=int, default=2_290_566, help="total samples for a full corpus")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    from PIL import Image

    rows = []
    for shot_path in args.shot:
        shot = np.asarray(Image.open(shot_path).convert("RGB"), dtype=np.uint8)
        height, width = shot.shape[:2]
        playfield = shot[16 : 16 + 448, 32 : 32 + 384]
        variants = [
            ("full client", shot),
            ("playfield crop", playfield),
        ]
        for name, frame in variants:
            encoded = {}
            for label, fn in (("npz_deflate", _npz_size), ("png", _png_size), ("webp_lossless", _webp_size)):
                started = time.perf_counter()
                encoded[label] = fn(frame)
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "shot": Path(shot_path).name,
                        "region": name,
                        "shape": list(frame.shape[:2]),
                        "raw_bytes": int(frame.nbytes),
                        "codec": label,
                        "bytes": int(encoded[label]),
                        "ratio_vs_raw": round(frame.nbytes / max(1, encoded[label]), 2),
                        "encode_ms": round(elapsed * 1000, 2),
                        "corpus_gb": round(encoded[label] * args.samples / 1e9, 1),
                    }
                )
    # decode cost (matters for the training loader)
    shot = np.asarray(Image.open(args.shot[0]).convert("RGB"), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(shot).save(buffer, format="PNG")
    buffer.seek(0)
    started = time.perf_counter()
    for _ in range(20):
        buffer.seek(0)
        Image.open(buffer).load()
    rows.append({"decode": "png_full_client_ms", "ms": round((time.perf_counter() - started) / 20 * 1000, 2)})

    payload = {"client_size": [int(width), int(height)], "samples_assumed": args.samples, "rows": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
