"""Minimal lossless PNG writer (stdlib only) for the WSL-side collector.

The collector runs inside the WSL guest, where the project's virtualenv has
numpy but not Pillow, and installing a package is out of scope.  A PNG only
needs zlib and CRC32, both in the standard library, so this writes a plain RGBA8
non-interlaced PNG with filter 0.  It is lossless by construction: the pixel
bytes written are exactly the ``SHM`` readback bytes.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def rgba_to_png_bytes(pixels: np.ndarray, *, level: int = 1) -> bytes:
    """Encode ``uint8[H, W, 4]`` top-down RGBA into PNG bytes."""

    array = np.asarray(pixels)
    if array.dtype != np.uint8:
        raise ValueError(f"pixels must be uint8, got {array.dtype}")
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"pixels must be HxWx4 RGBA, got {array.shape}")
    height, width = array.shape[:2]
    raw = np.empty((height, width * 4 + 1), dtype=np.uint8)
    raw[:, 0] = 0  # filter type 0 (None) per scanline
    raw[:, 1:] = array.reshape(height, width * 4)
    header = bytearray()
    header += b"\x89PNG\r\n\x1a\n"
    header += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    header += _chunk(b"IDAT", zlib.compress(raw.tobytes(), level))
    header += _chunk(b"IEND", b"")
    return bytes(header)


def write_rgba_png(path: Path, pixels: np.ndarray, *, level: int = 1) -> dict:
    """Write one frame and return its provenance digests."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = rgba_to_png_bytes(pixels, level=level)
    path.write_bytes(payload)
    return {
        "file": path.name,
        "png_bytes": len(payload),
        "png_sha256": hashlib.sha256(payload).hexdigest(),
        "rgba_sha256": hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest(),
        "width": int(pixels.shape[1]),
        "height": int(pixels.shape[0]),
    }
