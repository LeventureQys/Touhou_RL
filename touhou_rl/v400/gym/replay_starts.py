"""Decode TH08 replay files into provenance-safe replay start records.

This module deliberately does not pretend to attach to a running environment.  It
uses the retail replay codec and exposes the exact per-stage input streams, seeds,
and resource identity needed to construct a verified start index.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
REPLAY_SCRIPTS = ROOT / "third_party" / "th08" / "scripts"
if str(REPLAY_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REPLAY_SCRIPTS))

from replay_file import DIFFICULTIES, SHOTS, STAGES, decode  # type: ignore
import struct

RETAIL_IDENTITY = (840704, 2724749753, "0100d")


@dataclass(frozen=True)
class ReplayStart:
    source: str
    source_sha256: str
    stage_index: int
    stage_name: str
    stage_frame: int
    global_frame: int
    rng_seed: int
    input_records: int
    shot: int
    shot_name: str
    difficulty: int
    difficulty_name: str
    extended: bool
    executable: dict
    resource_metadata: dict

    @property
    def takeover_supported(self) -> bool:
        """Whether this record can attach to a live environment (always false here)."""
        return False

    def as_dict(self) -> dict:
        return {
            "source": self.source, "source_sha256": self.source_sha256,
            "stage_index": self.stage_index, "stage_name": self.stage_name,
            "stage_frame": self.stage_frame, "global_frame": self.global_frame,
            "rng_seed": self.rng_seed, "input_records": self.input_records, "shot": self.shot,
            "difficulty": self.difficulty, "difficulty_name": self.difficulty_name,
            "extended": self.extended, "executable": self.executable,
            "resource_metadata": self.resource_metadata,
            "takeover_supported": False,
        }


def _metadata(header: bytes, raw: bytes, path: Path) -> tuple[dict, dict]:
    identity = {"size": struct.unpack_from("<I", header, 0x124)[0],
                "checksum": struct.unpack_from("<I", header, 0x128)[0],
                "version": header[0x12C:0x132].split(b"\0", 1)[0].decode("ascii", "replace")}
    resource = {"replay_bytes": len(raw), "replay_sha256": hashlib.sha256(raw).hexdigest(),
                "executable": identity, "codec": "TH08-T8RP-v6-LZSS",
                "source_name": path.name}
    return identity, resource


def decode_replay(path: str | Path, require_retail: bool = True) -> list[ReplayStart]:
    path = Path(path)
    raw = path.read_bytes()
    data = decode(raw)
    source_sha = hashlib.sha256(raw).hexdigest()
    identity, resource = _metadata(data, raw, path)
    exe = (840704, 2724749753, "0100d")
    actual = (identity["size"], identity["checksum"], identity["version"])
    if require_retail and actual != exe:
        raise ValueError(f"Replay executable identity is {actual}, expected {exe}")
    shot, difficulty = data[0x6A:0x6C]
    if shot >= len(SHOTS) or difficulty >= len(DIFFICULTIES):
        raise ValueError("Invalid replay shot or difficulty")
    if data[0xB4 + 0x25]:
        raise ValueError("Slow-mode replay")
    extended = data[6]
    if extended not in (0, 1):
        raise ValueError("Invalid replay input encoding")
    extended = bool(extended)
    stride = 6 if extended else 2
    stage_offsets = struct.unpack_from("<9I", data, 0x20)
    fps_offsets = struct.unpack_from("<9I", data, 0x44)
    offsets = sorted(v for v in (*stage_offsets, *fps_offsets) if v)
    if len(set(offsets)) != len(offsets) or any(not 0x134 <= v < len(data) for v in offsets):
        raise ValueError("Invalid replay stage/FPS offsets")
    starts = []
    global_frame = 0
    for index, offset in enumerate(stage_offsets):
        if not offset:
            if fps_offsets[index]:
                raise ValueError("FPS stream without a replay stage")
            continue
        if not fps_offsets[index]:
            raise ValueError("Replay stage without FPS stream")
        end = min(v for v in (*offsets, len(data)) if v > offset)
        size = end - offset - 0x24
        if size < stride or size % stride:
            raise ValueError("Invalid replay input extent")
        count = size // stride
        starts.append(ReplayStart(str(path.resolve()), source_sha, index, STAGES[index], 0,
            global_frame, struct.unpack_from("<H", data, offset + 0x1A)[0], count,
            shot, SHOTS[shot], difficulty, DIFFICULTIES[difficulty], extended, identity, resource))
        global_frame += count
    if not starts:
        raise ValueError("Replay contains no stages")
    return starts


def build_start_index(paths: Iterable[str | Path], require_retail: bool = True, sample_every: int = 600) -> list[dict]:
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")
    result = []
    for path in paths:
        for start in decode_replay(path, require_retail):
            for frame in range(0, start.input_records, sample_every):
                row = start.as_dict()
                row["stage_frame"] = frame
                row["global_frame"] = start.global_frame + frame
                row["sample_every"] = sample_every
                result.append(row)
    return result


# Friendly aliases for callers/tools.
replay_starts = decode_replay

__all__ = ["ReplayStart", "decode_replay", "replay_starts", "build_start_index"]
