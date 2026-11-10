"""Reader for the harness shared-memory block (10-P0 §3).

Layout, all little endian::

    [0, 64)     header: magic u32, version u32, frame u64, state hash u64,
                        pixel width u32, pixel height u32, state bytes u32, flags u32
    [64, 192)   the 25 tracked fields as u32
    [192, ...)  pixels, RGBA8, pixel_width x pixel_height

The engine is the only writer.  A reader takes the header's frame counter to tell
whether it is looking at a frame it has already consumed.

Since T7.1/T7.3 (task book, ``build/rl_v3``) the engine publishes this block
itself at the end of every ``RESET`` and every ``STEP``, so ``STATE``/``HASH``
are no longer a prerequisite for reading a current state: the header's frame
number is the one the command's reply reports.  :class:`touhou_rl.v400.gym.env.Th08Env`
verifies exactly that pairing and raises ``StaleStateError`` when it does not
hold.
"""

from __future__ import annotations

import mmap
import os
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x30303456  # "V400"
VERSION = 1
HEADER = struct.Struct("<IIQQIIII")
HEADER_SIZE = 64
STATE_OFFSET = 64
STATE_FIELDS = 25
STATE_BYTES = STATE_FIELDS * 4
PIXELS_OFFSET = 192

FLAG_STATE_VALID = 1 << 0
FLAG_PIXELS_VALID = 1 << 1

#: Linux exposes POSIX shared memory as files under here.
SHM_DIR = Path("/dev/shm")


def segment_path(name: str) -> Path:
    """Resolve a shm name to its ``/dev/shm`` file, rejecting path escapes."""
    cleaned = name.lstrip("/")
    if not cleaned or "/" in cleaned or cleaned in (".", ".."):
        raise ValueError(f"invalid shared-memory name: {name!r}")
    return SHM_DIR / cleaned


class SharedBlock:
    """Maps the engine's shared block read-only."""

    def __init__(self, name: str) -> None:
        self.name = name.lstrip("/")
        self.path = segment_path(self.name)
        self._file = open(self.path, "rb")
        size = os.fstat(self._file.fileno()).st_size
        if size < PIXELS_OFFSET:
            self._file.close()
            raise ValueError(f"shared block too small: {size} bytes at {self.path}")
        self._map = mmap.mmap(self._file.fileno(), size, access=mmap.ACCESS_READ)
        self.size = size

    def __enter__(self) -> "SharedBlock":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._map.close()
        finally:
            self._file.close()

    # -- header -----------------------------------------------------------
    def header(self) -> dict[str, int]:
        magic, version, frame, state_hash, width, height, state_bytes, flags = (
            HEADER.unpack_from(self._map, 0)
        )
        if magic != MAGIC:
            raise ValueError(f"bad magic 0x{magic:08x} (expected 0x{MAGIC:08x})")
        return {
            "version": version,
            "frame": frame,
            "state_hash": state_hash,
            "width": width,
            "height": height,
            "state_bytes": state_bytes,
            "flags": flags,
        }

    # -- payloads ---------------------------------------------------------
    def state(self) -> np.ndarray:
        """The 25 tracked fields as ``uint32[25]``, in replay-trace order."""
        return np.frombuffer(self._map, dtype="<u4", count=STATE_FIELDS, offset=STATE_OFFSET).copy()

    def pixels(self) -> np.ndarray:
        """The frame as ``uint8[H, W, 4]``, top row first."""
        width, height = self._dimensions()
        count = width * height * 4
        if self.size < PIXELS_OFFSET + count:
            raise ValueError(
                f"shared block holds {self.size - PIXELS_OFFSET} pixel bytes, "
                f"need {count} for {width}x{height}"
            )
        raw = np.frombuffer(self._map, dtype=np.uint8, count=count, offset=PIXELS_OFFSET)
        # glReadPixels hands back the bottom row first; flip to top-down.
        return raw.reshape(height, width, 4)[::-1].copy()

    def _dimensions(self) -> tuple[int, int]:
        header = self.header()
        return header["width"], header["height"]
