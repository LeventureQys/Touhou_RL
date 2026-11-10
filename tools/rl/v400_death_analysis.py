"""P1-WP4 death analysis: rewind the last 3 s of a teacher death, render it, explain it.

The tool answers one question per death — *what killed the teacher v1* — with
artifacts instead of adjectives: 180 frames of replayed pixel data, one GIF, a
``death.json`` whose classification is produced by the fixed rules in
:class:`Rules`, and a summary table over every analysed death.

Pipeline (``--mode run``)
-------------------------

1. **Tape** — the run's own JSONL (``v400_teacher.py --out``) already carries
   the committed mask sequence per decision (``step_masks``: ``0x0041x2`` means
   "mask 0x0041 held for 2 frames").  Concatenating every decision's tokens
   reproduces the exact per-frame input of the original run, so **the teacher is
   not re-run**: only the authoritative engine process is replayed.  The tape is
   validated against the report (``survival_frames``, per-decision ``frame``
   counter, ``committed_frames_per_decision``); any inconsistency is fail-closed.
2. **Replay** — one ``th08-modern`` process in the ``env`` role (``PARK`` is not
   needed here, only pixels) with ``READBACK 1`` + ``SHM``.  Frames before the
   window are stepped in groups; from ``death_frame - pre_frames`` on, every
   frame is stepped **one frame at a time** and read back.
3. **Capture** — per window frame: 640x480 RGBA from ``/dev/shm`` -> PNG,
   ``PLAYER`` (position/hurtbox/state/bombs), ``HAZARDS`` -> nearest lethal
   geometry via the teacher's own ``geometry.frame_verdict``, ``EVENTS``
   (``frameEventFlags``), ``STATE``, plus the plan the teacher had chosen for
   that frame's decision (``chosen_plan``/``margin``/``geo_score``).
4. **Reproduction gate** — the replay must show the death at ``--death-frame``.
   A replay that dies elsewhere (or not at all) exits ``3`` and writes nothing
   that could be mistaken for an analysis.
5. **Classify** — :func:`classify_death` emits *multi-label* causes plus one
   primary cause from :data:`PRIORITY`.  The rule text travels in the JSON.

``--mode summary`` scans ``<root>/*/death.json`` and writes ``summary.md`` +
``summary.json`` + ``summary.csv``.

Roles and dependencies
----------------------

* Runs **inside WSL** (the engine is a Linux ELF) with ``xvfb-run`` for the GL
  context; ``--mode run`` refuses a missing engine and a SHA mismatch.
* ``--mode summary``, the tape parser and the classifier are pure Python and
  also work on the Windows host, which is what the unit tests exercise.
* GIF: Pillow when it is importable (the Windows host has it), otherwise a
  stdlib-only GIF89a encoder (:func:`write_gif`).  Nothing is installed at run
  time — the WSL virtualenv has numpy but no Pillow, and that is a hard rule.

Exit codes: ``0`` ok, ``1`` usage, ``3`` fail-closed (data/replay disagree),
``4`` engine SHA mismatch, ``5`` engine error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):  # allow ``python tools/rl/v400_death_analysis.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_FAIL_CLOSED = 3
EXIT_ENGINE_SHA = 4
EXIT_ENGINE = 5

#: The ``rl_hazard2`` build the P1 corpus was recorded on (EXP-P1-011/012/013).
DEFAULT_ENGINE = "/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2"
EXPECTED_ENGINE_SHA256 = "f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e"

#: 60 fps is the engine's real-time reference; 180 frames == 3 s.
REFERENCE_FPS = 60.0
DEFAULT_PRE_FRAMES = 180

SCHEMA = "v400-p1-wp4-death-analysis-1"

#: ``ReplayManager.hpp:161-172`` frame-event bits (mirror of
#: :mod:`touhou_rl.v400.labels.events`; restated so the report needs no import).
EVENT_BIT_NAMES: dict[int, str] = {
    0x0001: "BOMB_STARTED",
    0x0002: "PLAYER_HIT",
    0x0004: "PLAYER_DEATH_COMMITTED",
    0x0020: "ENEMY_REMOVED",
    0x0040: "ITEM_COLLECTED",
    0x0100: "PAUSE_RECORDED",
    0x0200: "PLAYER_DYING_STARTED",
    0x0400: "EFFECT_SPAWNED",
    0x0800: "BULLET_PATTERN_SPAWNED",
    0x1000: "ENEMY_SPAWNED",
    0x2000: "PLAYER_GRAZED",
}

#: ``ReplayFrameEvent::PLAYER_DEATH_COMMITTED`` / ``PLAYER_HIT``.
EVENT_DEATH = 0x0004
EVENT_HIT = 0x0002
EVENT_DYING = 0x0200

#: ``ENEMY_FLAG_BOSS`` (``touhou_rl/v400/teacher/engine.py:46``): bit 1 of
#: ``flags1``.  Restated here so the pure half of this module needs no engine
#: imports; a wrong value here would silently report every enemy as a boss.
ENEMY_FLAG_BOSS = 1 << 1

#: ``playerState`` for a dying ship (``PlayerState::PlayerState_Dying``).
PLAYER_STATE_DYING = 2

#: Cause priority: the first label that fired is the primary cause.  The order
#: is "immediate lethal mechanism > arena context > contributing condition", so
#: a bomb shortage can never be reported as the thing that killed you.
PRIORITY: tuple[str, ...] = (
    "laser",
    "bullet_wall",
    "boss_pattern",
    "bottom_pin",
    "bomb_exhausted",
)
ALL_LABELS: tuple[str, ...] = PRIORITY + ("unknown",)

#: Stage index -> the 面 name EXP-P1-012 used.
STAGE_NAMES: dict[int, str] = {
    0: "Stage1",
    1: "Stage2",
    2: "Stage3",
    3: "Stage4A",
    4: "Stage4B",
    5: "Stage5",
    6: "Stage6A",
    7: "Stage6B",
    8: "Extra",
}

#: The six EXP-P1-012 PSN deaths this card analyses (``stage index: death frame``).
EXP_P1_012_DEATHS: tuple[tuple[int, int], ...] = (
    (2, 21308),
    (3, 28616),
    (4, 12908),
    (5, 9172),
    (6, 13484),
    (7, 7652),
)


class DeathAnalysisError(RuntimeError):
    """A precondition, a protocol reply or the replay contract failed."""


class FailClosed(DeathAnalysisError):
    """The inputs disagree with each other; refuse to produce an analysis."""


class EngineShaMismatch(DeathAnalysisError):
    """The engine binary is not the build the P1 corpus was recorded on."""


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rules:
    """Every threshold the classification uses, in one place.

    Defaults are fixed *before* the six deaths were analysed; ``death.json``
    carries this dataclass verbatim so a reader can re-derive every label.
    """

    #: ``bottom_pin``: mean ``py`` of the last ``bottom_window_frames`` frames.
    bottom_band_y: float = 400.0
    bottom_window_frames: int = 60
    #: ``bomb_exhausted``: ``bombs`` on the death frame is at most this.
    bomb_exhausted_at: float = 0.0
    #: ``bullet_wall``: of the last ``bullet_wall_window_frames`` frames, at
    #: least ``bullet_wall_min_frames`` must hold ``>= bullet_wall_boxes``
    #: lethal AABBs (bullets + enemy bodies, the teacher's own box model).
    bullet_wall_window_frames: int = 10
    bullet_wall_min_frames: int = 6
    bullet_wall_boxes: int = 40
    #: ``laser``: an *active* laser within ``laser_near_px`` of the hurtbox
    #: centre on any of the last ``laser_window_frames`` frames.
    laser_window_frames: int = 10
    laser_near_px: float = 32.0
    #: ``boss_pattern``: boss alive and closer than this (playfield height).
    boss_radius_px: float = 448.0

    def as_dict(self) -> dict:
        return {
            "bottom_band_y": self.bottom_band_y,
            "bottom_window_frames": self.bottom_window_frames,
            "bomb_exhausted_at": self.bomb_exhausted_at,
            "bullet_wall_window_frames": self.bullet_wall_window_frames,
            "bullet_wall_min_frames": self.bullet_wall_min_frames,
            "bullet_wall_boxes": self.bullet_wall_boxes,
            "laser_window_frames": self.laser_window_frames,
            "laser_near_px": self.laser_near_px,
            "boss_radius_px": self.boss_radius_px,
        }

    def as_text(self) -> dict[str, str]:
        return {
            "bottom_pin": (
                f"mean(py) over the last {self.bottom_window_frames} captured frames "
                f"> {self.bottom_band_y:g}"
            ),
            "bomb_exhausted": f"bombs on the death frame <= {self.bomb_exhausted_at:g}",
            "bullet_wall": (
                f"at least {self.bullet_wall_min_frames} of the last "
                f"{self.bullet_wall_window_frames} frames carry >= "
                f"{self.bullet_wall_boxes} lethal AABBs (bullets + enemy bodies)"
            ),
            "laser": (
                f"an active laser is within {self.laser_near_px:g} px of the hurtbox "
                f"centre on any of the last {self.laser_window_frames} frames"
            ),
            "boss_pattern": (
                f"a boss enemy is alive and closer than {self.boss_radius_px:g} px "
                "on the death frame"
            ),
            "unknown": "none of the five rules above fired",
            "priority": " > ".join(PRIORITY),
        }


DEFAULT_RULES = Rules()


def classify_death(
    *,
    py_series: list[float],
    boxes_series: list[int],
    laser_gaps: list[float | None],
    bombs_at_death: float,
    boss_alive: bool,
    boss_distance: float | None,
    rules: Rules = DEFAULT_RULES,
) -> dict:
    """Turn one window's series into multi-label causes plus a primary cause.

    Pure function: no engine, no files, no clock.  ``py_series`` and
    ``boxes_series`` are chronological and end on the death frame;
    ``laser_gaps`` is the distance from the hurtbox centre to the nearest
    *active* laser per frame (``None`` when no laser is active).

    Returns ``{"labels": [...], "primary": str, "evidence": {...},
    "missing": [...]}``.  ``labels`` is ordered by :data:`PRIORITY`.
    """

    evidence: dict[str, dict] = {}
    missing: list[str] = []

    bottom_window = py_series[-max(1, rules.bottom_window_frames):]
    if bottom_window:
        bottom_mean = sum(bottom_window) / len(bottom_window)
    else:  # pragma: no cover - the caller always passes a non-empty window
        bottom_mean = float("nan")
    bottom_hit = bool(bottom_window) and bottom_mean > rules.bottom_band_y
    evidence["bottom_pin"] = {
        "mean_py": round(bottom_mean, 4),
        "threshold_py": rules.bottom_band_y,
        "frames": len(bottom_window),
        "min_py": round(min(bottom_window), 4) if bottom_window else None,
        "max_py": round(max(bottom_window), 4) if bottom_window else None,
        "fired": bottom_hit,
    }

    bomb_hit = bombs_at_death <= rules.bomb_exhausted_at
    evidence["bomb_exhausted"] = {
        "bombs_at_death": bombs_at_death,
        "threshold": rules.bomb_exhausted_at,
        "fired": bomb_hit,
    }

    wall_window = boxes_series[-max(1, rules.bullet_wall_window_frames):]
    above = sum(1 for count in wall_window if count >= rules.bullet_wall_boxes)
    wall_hit = bool(wall_window) and above >= rules.bullet_wall_min_frames
    evidence["bullet_wall"] = {
        "boxes_threshold": rules.bullet_wall_boxes,
        "frames_considered": len(wall_window),
        "frames_at_or_above": above,
        "frames_required": rules.bullet_wall_min_frames,
        "boxes_series": list(wall_window),
        "max_boxes": max(wall_window) if wall_window else None,
        "median_boxes": (
            sorted(wall_window)[len(wall_window) // 2] if wall_window else None
        ),
        "fired": wall_hit,
    }

    laser_window = laser_gaps[-max(1, rules.laser_window_frames):]
    near = [gap for gap in laser_window if gap is not None and gap <= rules.laser_near_px]
    nearest_laser = min(
        (gap for gap in laser_window if gap is not None), default=None
    )
    laser_hit = bool(near)
    evidence["laser"] = {
        "near_px": rules.laser_near_px,
        "frames_considered": len(laser_window),
        "frames_with_laser": sum(1 for gap in laser_window if gap is not None),
        "frames_within_near": len(near),
        "nearest_gap_px": None if nearest_laser is None else round(nearest_laser, 4),
        "fired": laser_hit,
    }

    boss_hit = bool(
        boss_alive
        and boss_distance is not None
        and boss_distance < rules.boss_radius_px
    )
    evidence["boss_pattern"] = {
        "boss_alive": bool(boss_alive),
        "boss_distance": None if boss_distance is None else round(boss_distance, 4),
        "radius_px": rules.boss_radius_px,
        "fired": boss_hit,
    }

    fired = {
        "bottom_pin": bottom_hit,
        "bomb_exhausted": bomb_hit,
        "bullet_wall": wall_hit,
        "boss_pattern": boss_hit,
        "laser": laser_hit,
    }
    labels = [name for name in PRIORITY if fired[name]]
    primary = labels[0] if labels else "unknown"
    if not labels:
        labels = ["unknown"]
    if not py_series:
        missing.append("py_series")
    if not boxes_series:
        missing.append("boxes_series")
    return {
        "labels": labels,
        "primary": primary,
        "evidence": evidence,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# replay tape (pure)
# ---------------------------------------------------------------------------
_STEP_TOKEN = re.compile(r"^0x([0-9a-fA-F]{1,4})x(\d+)$")


def parse_step_masks(tokens: list[str]) -> list[tuple[int, int]]:
    """``["0x0041x2", "0x0040x1"]`` -> ``[(0x41, 2), (0x40, 1)]``."""

    out: list[tuple[int, int]] = []
    for token in tokens:
        match = _STEP_TOKEN.match(str(token).strip())
        if match is None:
            raise FailClosed(f"unparsable step_masks token: {token!r}")
        mask = int(match.group(1), 16)
        count = int(match.group(2))
        if not 0 <= mask <= 0xFFFF:
            raise FailClosed(f"mask out of range: {token!r}")
        if count <= 0:
            raise FailClosed(f"non-positive frame count: {token!r}")
        out.append((mask, count))
    return out


def load_replay(path: str | Path) -> dict:
    """Read one teacher JSONL and rebuild its per-frame input tape.

    Fail-closed on: no report line, no decision lines, no death, a missing
    ``step_masks`` token, a decision ``frame`` counter that does not match the
    cumulative tape length, or a tape whose length is not
    ``committed_frames_per_decision * decisions``.
    """

    path = Path(path)
    if not path.is_file():
        raise FailClosed(f"replay JSONL not found: {path}")
    digest = hashlib.sha256()
    report: dict | None = None
    decisions: list[dict] = []
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FailClosed(f"{path.name}: unparsable JSONL line: {error}") from error
            kind = record.get("kind")
            if kind == "report":
                report = record
            elif kind == "decision":
                decisions.append(record)
    if report is None:
        raise FailClosed(f"{path.name}: no kind=report line")
    if not decisions:
        raise FailClosed(f"{path.name}: no kind=decision lines")
    if report.get("status") != "ok":
        raise FailClosed(f"{path.name}: report status is {report.get('status')!r}")
    if int(report.get("deaths_observed", 0)) < 1:
        raise FailClosed(
            f"{path.name}: report observed {report.get('deaths_observed')} deaths, "
            "there is no death to analyse"
        )

    frames_per_decision = int(report.get("committed_frames_per_decision") or 0)
    tape: list[tuple[int, int]] = []
    per_decision: list[dict] = []
    for index, record in enumerate(decisions):
        expected_index = record.get("decision")
        if expected_index is not None and int(expected_index) != index:
            raise FailClosed(
                f"{path.name}: decision {index} is labelled {expected_index}"
            )
        tokens = record.get("step_masks")
        if not tokens:
            raise FailClosed(f"{path.name}: decision {index} has no step_masks")
        pairs = parse_step_masks(tokens)
        advanced = sum(count for _, count in pairs)
        tape.extend(pairs)
        counted = sum(count for _, count in tape)
        reported_frame = record.get("frame")
        if reported_frame is not None and int(reported_frame) != counted:
            raise FailClosed(
                f"{path.name}: decision {index} frame={reported_frame} but the tape "
                f"reaches {counted}"
            )
        per_decision.append(
            {
                "decision": index,
                "frame": counted,
                "frames": advanced,
                "first_action": (record.get("chosen_plan") or {}).get("first_action"),
                "duration": (record.get("chosen_plan") or {}).get("duration"),
                "slots": (record.get("chosen_plan") or {}).get("slots"),
                "margin": record.get("margin"),
                "score": (record.get("chosen_plan") or {}).get("score"),
                "geo_score": record.get("geo_score"),
                "verified": (record.get("chosen_plan") or {}).get("verified"),
                "bomb": record.get("bomb"),
                "frame_event_flags": record.get("frame_event_flags"),
            }
        )

    total_frames = sum(count for _, count in tape)
    survival = report.get("survival_frames")
    if survival is not None and int(survival) != total_frames:
        raise FailClosed(
            f"{path.name}: report survival_frames={survival} but the tape holds "
            f"{total_frames} frames"
        )
    if frames_per_decision and frames_per_decision * len(decisions) != total_frames:
        raise FailClosed(
            f"{path.name}: {len(decisions)} decisions x {frames_per_decision} frames "
            f"!= tape {total_frames}"
        )

    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "report": report,
        "decisions": per_decision,
        "tape": tape,
        "total_frames": total_frames,
        "frames_per_decision": frames_per_decision or None,
    }


def tape_frames(tape: list[tuple[int, int]]) -> list[int]:
    """Flatten ``[(mask, count)]`` into one mask per frame."""

    out: list[int] = []
    for mask, count in tape:
        out.extend([mask] * count)
    return out


def decision_for_frame(frame: int, frames_per_decision: int) -> int:
    """Which decision owns ``frame`` (1-based), given 4 frames per decision."""

    if frames_per_decision <= 0:
        raise ValueError("frames_per_decision must be positive")
    if frame <= 0:
        raise ValueError(f"frame must be positive, got {frame}")
    return (frame - 1) // frames_per_decision


# ---------------------------------------------------------------------------
# event bit decomposition
# ---------------------------------------------------------------------------
def decompose_event_flags(value: int) -> dict:
    """Split a 16-bit ``frameEventFlags`` word into named bits."""

    value = int(value)
    bits = []
    for bit in range(16):
        mask = 1 << bit
        bits.append(
            {
                "bit": bit,
                "mask": f"0x{mask:04x}",
                "name": EVENT_BIT_NAMES.get(mask),
                "set": bool(value & mask),
            }
        )
    names = [entry["name"] for entry in bits if entry["set"] and entry["name"]]
    return {
        "raw": f"0x{value:04x}",
        "value": value,
        "bits": bits,
        "names": names,
        "unknown_bits": [
            entry["mask"] for entry in bits if entry["set"] and not entry["name"]
        ],
        "player_hit": bool(value & EVENT_HIT),
        "death_committed": bool(value & EVENT_DEATH),
        "dying_started": bool(value & EVENT_DYING),
    }


def annotate_flags(series: list[int]) -> dict:
    """Bit decomposition for a whole window: death frame plus OR/AND masks."""

    if not series:
        return {"death_frame": decompose_event_flags(0), "frames": 0}
    union = 0
    intersection = 0xFFFF
    for value in series:
        union |= int(value)
        intersection &= int(value)
    return {
        "frames": len(series),
        "death_frame": decompose_event_flags(series[-1]),
        "union": decompose_event_flags(union),
        "intersection": decompose_event_flags(intersection),
        "frames_with_hit": sum(1 for v in series if int(v) & EVENT_HIT),
        "frames_with_death": sum(1 for v in series if int(v) & EVENT_DEATH),
        "series": [f"0x{int(v):04x}" for v in series],
    }


# ---------------------------------------------------------------------------
# GIF: Pillow when present, stdlib GIF89a otherwise
# ---------------------------------------------------------------------------
#: 6x6x6 colour cube, index = r*36 + g*6 + b (216 entries), then 40 greys.
_CUBE_LEVELS = (0, 51, 102, 153, 204, 255)


def gif_palette() -> list[tuple[int, int, int]]:
    """The 256-entry palette both writers use (216 cube + 40 greys)."""

    palette = [
        (r, g, b) for r in _CUBE_LEVELS for g in _CUBE_LEVELS for b in _CUBE_LEVELS
    ]
    palette.extend((value, value, value) for value in range(216, 256))
    palette.extend([(0, 0, 0)] * (256 - len(palette)))
    return palette[:256]


def quantize_rgba(rgba: np.ndarray) -> np.ndarray:
    """Map ``uint8[H, W, 4]`` RGBA onto :func:`gif_palette` indices."""

    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"expected HxWx3/4 pixels, got {array.shape}")
    rgb = array[:, :, :3].astype(np.uint16)
    level = (rgb * 5 + 127) // 255
    index = level[:, :, 0] * 36 + level[:, :, 1] * 6 + level[:, :, 2]
    return index.astype(np.uint8)


def _lzw_encode(data: bytes, min_code_size: int = 8) -> bytes:
    """GIF-variant LZW (LSB-first codes, 12-bit ceiling with clear codes)."""

    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    code_size = min_code_size + 1
    next_code = end_code + 1
    table: dict[int, int] = {}
    out = bytearray()
    bit_buffer = 0
    bit_count = 0

    def emit(code: int) -> None:
        nonlocal bit_buffer, bit_count
        bit_buffer |= code << bit_count
        bit_count += code_size
        while bit_count >= 8:
            out.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8

    emit(clear_code)
    if not data:
        emit(end_code)
    else:
        prefix = int(data[0])
        for byte in data[1:]:
            key = (prefix << 8) | int(byte)
            found = table.get(key)
            if found is not None:
                prefix = found
                continue
            emit(prefix)
            if next_code < 4096:
                table[key] = next_code
                next_code += 1
                # The width grows as soon as the *assigned* code no longer fits,
                # i.e. one entry after the table fills the current width.  A GIF
                # decoder can only know the width of the next code from the
                # entries it already holds, so the encoder must lag by exactly
                # that one entry; verified by decoding every GIF back through
                # Pillow in ``tests/test_v400_death_analysis.py``.
                if next_code > (1 << code_size) and code_size < 12:
                    code_size += 1
            else:
                emit(clear_code)
                table.clear()
                code_size = min_code_size + 1
                next_code = end_code + 1
            prefix = int(byte)
        emit(prefix)
        emit(end_code)
    while bit_count > 0:
        out.append(bit_buffer & 0xFF)
        bit_buffer >>= 8
        bit_count -= 8
    return bytes(out)


def _sub_blocks(payload: bytes) -> bytes:
    out = bytearray()
    for start in range(0, len(payload), 255):
        chunk = payload[start:start + 255]
        out.append(len(chunk))
        out.extend(chunk)
    out.append(0)
    return bytes(out)


def write_gif_stdlib(
    path: str | Path,
    indices: list[np.ndarray],
    *,
    palette: list[tuple[int, int, int]] | None = None,
    delay_cs: int = 5,
    loop: int = 0,
) -> dict:
    """Write an animated GIF89a with nothing but the standard library.

    Used when Pillow is absent (the WSL venv).  ``indices`` are palette indices,
    so no quantisation happens here; the output is decoded back by Pillow in
    ``tests/test_v400_death_analysis.py`` to prove the encoder is real.
    """

    if not indices:
        raise ValueError("a GIF needs at least one frame")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette = list(palette or gif_palette())
    if len(palette) > 256:
        raise ValueError(f"palette has {len(palette)} entries, GIF caps at 256")
    height, width = indices[0].shape[:2]
    for frame in indices:
        if frame.shape[:2] != (height, width):
            raise ValueError("every GIF frame must have the same size")
        if frame.dtype != np.uint8:
            raise ValueError(f"GIF indices must be uint8, got {frame.dtype}")

    out = bytearray()
    out += b"GIF89a"
    out += struct.pack("<HH", width, height)
    out += bytes([0xF7, 0, 0])  # GCT present, 256 entries, 8-bit colour res
    table = bytearray()
    for entry in palette:
        table.extend(bytes(int(max(0, min(255, channel))) for channel in entry))
    table.extend(b"\x00" * (768 - len(table)))
    out += table
    out += b"\x21\xFF\x0BNETSCAPE2.0\x03\x01" + struct.pack("<H", int(loop)) + b"\x00"
    delay = max(0, min(0xFFFF, int(delay_cs)))
    for frame in indices:
        out += b"\x21\xF9\x04" + bytes([0x04]) + struct.pack("<H", delay) + b"\x00\x00"
        out += b"\x2C" + struct.pack("<HHHH", 0, 0, width, height) + b"\x00"
        out += bytes([8])  # LZW minimum code size
        out += _sub_blocks(_lzw_encode(np.ascontiguousarray(frame).tobytes(), 8))
    out += b"\x3B"
    path.write_bytes(bytes(out))
    return {
        "writer": "stdlib",
        "frames": len(indices),
        "width": int(width),
        "height": int(height),
        "bytes": len(out),
        "sha256": hashlib.sha256(out).hexdigest(),
    }


def write_gif_pillow(
    path: str | Path,
    indices: list[np.ndarray],
    *,
    palette: list[tuple[int, int, int]] | None = None,
    delay_cs: int = 5,
    loop: int = 0,
) -> dict:
    """Same frames through Pillow (the host has it; the WSL venv does not)."""

    from PIL import Image  # noqa: PLC0415 - optional dependency by design

    if not indices:
        raise ValueError("a GIF needs at least one frame")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    palette = list(palette or gif_palette())
    flat: list[int] = []
    for entry in palette:
        flat.extend(int(max(0, min(255, channel))) for channel in entry)
    flat.extend([0] * (768 - len(flat)))
    frames = []
    for frame in indices:
        contiguous = np.ascontiguousarray(frame)
        if contiguous.dtype != np.uint8:
            raise ValueError(f"GIF indices must be uint8, got {contiguous.dtype}")
        height, width = contiguous.shape[:2]
        image = Image.frombytes("P", (width, height), contiguous.tobytes())
        image.putpalette(flat)
        frames.append(image)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(delay_cs * 10)),
        loop=int(loop),
        optimize=False,
        disposal=1,
    )
    payload = path.read_bytes()
    return {
        "writer": "pillow",
        "frames": len(indices),
        "width": int(indices[0].shape[1]),
        "height": int(indices[0].shape[0]),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def pillow_available() -> bool:
    try:
        import PIL  # noqa: F401,PLC0415
    except Exception:
        return False
    return True


def write_gif(
    path: str | Path,
    indices: list[np.ndarray],
    *,
    palette: list[tuple[int, int, int]] | None = None,
    delay_cs: int = 5,
    loop: int = 0,
    prefer_pillow: bool = True,
) -> dict:
    """Write the animation with Pillow if possible, else with the stdlib writer."""

    if prefer_pillow and pillow_available():
        try:
            return write_gif_pillow(
                path, indices, palette=palette, delay_cs=delay_cs, loop=loop
            )
        except Exception as error:  # pragma: no cover - Pillow is present on the host
            record = write_gif_stdlib(
                path, indices, palette=palette, delay_cs=delay_cs, loop=loop
            )
            record["pillow_error"] = f"{type(error).__name__}: {error}"
            return record
    return write_gif_stdlib(
        path, indices, palette=palette, delay_cs=delay_cs, loop=loop
    )


# ---------------------------------------------------------------------------
# engine side
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(reply: str) -> dict[str, str]:
    out: dict[str, str] = {"_verb": reply.split(" ", 1)[0]}
    for part in reply.split()[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def prepare_data_dir(template: str | Path, target: str | Path) -> dict:
    """Copy a data dir's small files and symlink its archives (never modify one)."""

    template = Path(template)
    target = Path(target)
    if not template.is_dir():
        raise FailClosed(f"data-dir template not found: {template}")
    target.mkdir(parents=True, exist_ok=True)
    linked = []
    copied = []
    for name in ("th08.dat", "thbgm.dat"):
        source = template / name
        if not source.exists():
            raise FailClosed(f"{template} has no {name}")
        link = target / name
        if link.is_symlink() or link.exists():
            link.unlink()
        try:
            link.symlink_to(source.resolve())
            linked.append({"name": name, "source": str(source.resolve()),
                           "mode": "symlink"})
        except OSError:
            # Windows without the symlink privilege: copy instead of failing.
            # The WSL replay path always takes the symlink branch, so this only
            # keeps the helper usable (and testable) on the host.
            link.write_bytes(source.read_bytes())
            linked.append({"name": name, "source": str(source.resolve()),
                           "mode": "copy"})
    for name in ("th08.cfg", "score.dat"):
        source = template / name
        if source.is_file() and not source.is_symlink():
            (target / name).write_bytes(source.read_bytes())
            copied.append(name)
    for sub in ("replay", "backup"):
        (target / sub).mkdir(exist_ok=True)
    return {"data_dir": str(target), "linked": linked, "copied": copied}


def _maybe_reexec_under_xvfb(argv: list[str]) -> None:
    """The ``env`` role needs a GL context; re-exec under Xvfb once."""

    import os  # noqa: PLC0415 - only used on the replay path

    if os.environ.get("V400_DEATH_XVFB") == "1":
        return
    anchor = sys.argv[0] if sys.argv and sys.argv[0] else __file__
    new_argv = ["xvfb-run", "-a", sys.executable, str(Path(anchor).resolve()), *argv]
    env = dict(os.environ, V400_DEATH_XVFB="1")
    os.execvpe("xvfb-run", new_argv, env)


def _blank_pixels(pixels: np.ndarray) -> bool:
    return int(pixels.max()) == 0


def capture(
    *,
    run: dict,
    out_dir: Path,
    death_frame: int,
    pre_frames: int,
    engine_dir: str,
    expected_sha: str,
    data_template: str,
    timeout: float,
    save_pngs: bool = True,
) -> dict:
    """Replay the tape and capture the pre-death window.  WSL + engine only."""

    import os  # noqa: PLC0415

    # ``protocol``/``shm``/``png`` are imported here so that the pure half of
    # this module (tape, classifier, GIF) still imports on the Windows host,
    # where ``protocol`` cannot be imported at all (top-level ``import select``).
    from touhou_rl.v400.gym.protocol import (  # noqa: PLC0415
        HarnessError,
        parse_events_reply,
        parse_hazards_reply,
        parse_player_reply,
    )
    from touhou_rl.v400.gym.shm import PIXELS_OFFSET, SharedBlock, segment_path  # noqa: PLC0415
    from touhou_rl.v400.labels.png import write_rgba_png  # noqa: PLC0415
    from touhou_rl.v400.labels.runner import ControlClient  # noqa: PLC0415
    from touhou_rl.v400.labels.state import state_from_reply  # noqa: PLC0415
    from touhou_rl.v400.teacher import geometry  # noqa: PLC0415
    from touhou_rl.v400.teacher.engine import ENEMY_FLAG_BOSS  # noqa: PLC0415

    report = run["report"]
    stage = int(report["stage"])
    difficulty = int(report["difficulty"])
    team = int(report.get("team") or 0)
    seed = int(report["seed"])
    frames_per_decision = run["frames_per_decision"] or 4

    if death_frame > run["total_frames"]:
        raise FailClosed(
            f"death frame {death_frame} is beyond the tape's {run['total_frames']} frames"
        )
    first_frame = death_frame - pre_frames
    if first_frame < 1:
        raise FailClosed(
            f"{pre_frames} frames of lookback do not fit before frame {death_frame}"
        )

    binary = str(Path(engine_dir) / "th08-modern")
    if not Path(binary).is_file():
        raise FailClosed(f"engine binary not found: {binary}")
    actual_sha = sha256_file(binary)
    if actual_sha != expected_sha:
        raise EngineShaMismatch(
            f"{binary}: sha256 {actual_sha} != expected {expected_sha}"
        )

    data = prepare_data_dir(data_template, out_dir / "data" / "game")
    shm_name = f"v400death-{os.getpid()}-{int(time.time() * 1000) % 10_000_000}"
    log_path = out_dir / "engine.stderr.log"
    client = ControlClient(
        binary, data["data_dir"], role="env", timeout=timeout, stderr_path=log_path
    )
    block = None
    frames_dir = out_dir / "frames"
    if save_pngs:
        frames_dir.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "engine_binary": binary,
        "engine_sha256": actual_sha,
        "expected_sha256": expected_sha,
        "sha_match": actual_sha == expected_sha,
        "role": "env",
        "readback": None,
        "shm": shm_name,
        "data_dir": data,
        "reset_command": f"RESET {stage} {difficulty} {team} {seed} 0",
        "window": {"pre_frames": pre_frames, "first_frame": first_frame,
                   "last_frame": death_frame},
        "frames": [],
        "death_signals": {},
        "pixel_mismatches": 0,
        "blank_frames": 0,
        "advanced": 0,
    }

    started = time.perf_counter()
    try:
        client.start()
        hello = _tokens(client.request("HELLO"))
        record["hello"] = hello
        if hello.get("role") != "env":
            raise FailClosed(f"HELLO did not confirm the env role: {hello!r}")
        client.request(f"SHM /{shm_name}")
        readback = _tokens(client.request("READBACK 1"))
        record["readback"] = readback.get("readback")
        if readback.get("readback") != "1":
            raise FailClosed(f"READBACK 1 refused: {readback!r}")
        block = SharedBlock(shm_name)
        expected_bytes = PIXELS_OFFSET + 640 * 480 * 4
        if block.size != expected_bytes:
            raise FailClosed(
                f"shared block is {block.size} B, expected {expected_bytes}"
            )
        reset = _tokens(client.request(record["reset_command"], timeout=timeout))
        if reset.get("ready") != "1":
            raise FailClosed(f"RESET did not become ready: {reset!r}")
        frame_now = int(client.request("FRAME").split("=", 1)[1])
        if frame_now != 0:
            raise FailClosed(f"RESET left the frame counter at {frame_now}, expected 0")
        record["reset_ready_frame"] = frame_now

        def step(mask: int, count: int) -> dict:
            reply = _tokens(client.request(f"STEP {int(mask)} {int(count)}"))
            advanced = int(reply.get("advanced", 0))
            if advanced != int(count):
                raise FailClosed(
                    f"STEP {int(mask)} {int(count)} advanced {advanced} frames: {reply!r}"
                )
            record["advanced"] += advanced
            return reply

        # --- fast-forward to the window, in the tape's own grouping ---------
        target = first_frame - 1
        done = 0
        index = 0
        tape = run["tape"]
        while done < target and index < len(tape):
            mask, count = tape[index]
            take = min(count, target - done)
            step(mask, take)
            done += take
            if take == count:
                index += 1
            else:
                tape = list(tape)
                tape[index] = (mask, count - take)
        if done != target:
            raise FailClosed(
                f"tape ended at frame {done}, window starts at {first_frame}"
            )
        record["ff_frames"] = done
        record["ff_seconds"] = round(time.perf_counter() - started, 3)

        # ---- capture the window, one frame per STEP ------------------------
        per_frame_masks = tape_frames(run["tape"])
        baseline_state = state_from_reply(client.request("STATE"))
        deaths_baseline = float(baseline_state.get("deaths") or 0.0)
        record["deaths_baseline"] = deaths_baseline
        observed_death: int | None = None
        death_signal: str | None = None
        first_hit: int | None = None
        first_dying: int | None = None
        for frame in range(first_frame, death_frame + 1):
            mask = per_frame_masks[frame - 1]
            step(mask, 1)
            header = block.header()
            if int(header["frame"]) != frame - 1:
                record["pixel_mismatches"] += 1
                if record["pixel_mismatches"] > 8:
                    raise FailClosed(
                        f"frame {frame}: shm header frame {header['frame']} does not "
                        f"trail the step frame by exactly 1"
                    )
            pixels = block.pixels()
            if _blank_pixels(pixels):
                record["blank_frames"] += 1
                raise FailClosed(
                    f"frame {frame}: the readback is all zero bytes (no render)"
                )
            player = parse_player_reply(client.request("PLAYER"))
            hazards = parse_hazards_reply(client.request("HAZARDS"))
            events = parse_events_reply(client.request("EVENTS"))
            state = state_from_reply(client.request("STATE"))
            verdict_error = None
            try:
                verdict = geometry.frame_verdict(hazards, player)
                verdict_payload = verdict.as_dict()
                boxes = geometry.hazard_box_count(hazards)
                laser = geometry.laser_hits(
                    hazards.lasers, player.hurtbox_min, player.hurtbox_max
                )
                laser_payload = {
                    "considered": int(laser.considered),
                    "skipped_inactive": int(laser.skipped_inactive),
                    "gap": None if laser.nearest is None else float(laser.nearest[2]),
                }
            except ValueError as error:  # missing hurtbox on the dying frame
                verdict_error = f"{type(error).__name__}: {error}"
                verdict_payload = {"error": verdict_error}
                boxes = 0
                laser_payload = {"considered": 0, "skipped_inactive": 0, "gap": None}
            decisions = run["decisions"]
            decision_index = decision_for_frame(frame, frames_per_decision)
            plan = decisions[decision_index] if decision_index < len(decisions) else {}

            png = None
            if save_pngs:
                png = write_rgba_png(frames_dir / f"frame_{frame:06d}.png", pixels)

            frame_record = {
                "frame": int(frame),
                "decision": int(decision_index),
                "mask": f"0x{mask:04x}",
                "shm_frame": int(header["frame"]),
                "shm_state_hash": f"{int(header['state_hash']):016x}",
                "player": {
                    "px": float(player.position[0]),
                    "py": float(player.position[1]),
                    "hurtbox_min": list(player.hurtbox_min or ()),
                    "hurtbox_max": list(player.hurtbox_max or ()),
                    "player_state": int(player.player_state),
                    "invuln_state": int(player.invuln_state),
                    "bombs": float(player.bombs),
                    "lives": int(player.lives),
                    "power": float(player.power),
                    "focus": int(player.focus),
                },
                "hazards": {
                    "bullets": int(hazards.bullet_count),
                    "lasers": int(hazards.laser_count),
                    "enemies": int(hazards.enemy_count),
                    "boxes": int(boxes),
                    "verdict": verdict_payload,
                    "laser": laser_payload,
                },
                "boss": boss_distance_from(hazards, player),
                "events": {
                    "raw": f"0x{events.frame_event_flags:04x}",
                    "value": int(events.frame_event_flags),
                    "names": list(events.names),
                },
                "state": {
                    key: state.get(key)
                    for key in (
                        "frame", "stage", "deaths", "lives_remaining",
                        "bombs_remaining", "player_x", "player_y", "player_state",
                        "stage_generation",
                    )
                },
                "teacher": {
                    "decision": plan.get("decision"),
                    "first_action": plan.get("first_action"),
                    "duration": plan.get("duration"),
                    "margin": plan.get("margin"),
                    "score": plan.get("score"),
                    "verified": plan.get("verified"),
                },
                "png": None if png is None else png["file"],
                "png_sha256": None if png is None else png["png_sha256"],
                "rgba_sha256": None if png is None else png["rgba_sha256"],
            }
            record["frames"].append(frame_record)

            deaths_now = float(state.get("deaths") or 0.0)
            signals = {
                "state_deaths": deaths_now > deaths_baseline,
                "event_death": bool(events.frame_event_flags & EVENT_DEATH),
                "event_hit": bool(events.frame_event_flags & EVENT_HIT),
                "player_dying": int(player.player_state) == PLAYER_STATE_DYING,
            }
            if signals["event_hit"] and first_hit is None:
                first_hit = int(frame)
            if signals["player_dying"] and first_dying is None:
                first_dying = int(frame)
            # The *commit* signals are the death counter and the replay's
            # PLAYER_DEATH_COMMITTED bit.  PLAYER_HIT / DYING are leading
            # indicators that fire a few frames earlier and must never be
            # reported as the death frame.
            commit = None
            if signals["state_deaths"]:
                commit = "state_deaths"
            elif signals["event_death"]:
                commit = "event_death"
            if commit is not None and observed_death is None:
                observed_death = int(frame)
                death_signal = commit
            if observed_death is not None:
                break

        record["death_signals"] = {
            "observed_death_frame": observed_death,
            "first_signal": death_signal,
            "expected_death_frame": death_frame,
            "first_hit_frame": first_hit,
            "first_dying_frame": first_dying,
            "tolerance_frames": frames_per_decision,
        }
        record["wall_seconds"] = round(time.perf_counter() - started, 3)
        record["engine_stderr_tail"] = _tail(log_path)
        if observed_death is None:
            raise FailClosed(
                f"the replay never committed a death inside the window "
                f"{first_frame}..{death_frame} (first hit={first_hit}, "
                f"first dying={first_dying})"
            )
        # The JSONL pins the death to a *decision* (4 committed frames), not to
        # one frame: the teacher only compared the death counter after it had
        # committed the whole decision.  So the replay's commit frame must land
        # inside the recorded decision, and the exact frame is reported.
        if not (death_frame - frames_per_decision <= observed_death <= death_frame):
            raise FailClosed(
                f"the replay committed the death at frame {observed_death} "
                f"({death_signal}), outside the recorded decision "
                f"{death_frame - frames_per_decision + 1}..{death_frame}"
            )
        if observed_death != death_frame:
            # Trim so the analysis window ends on the frame the death was
            # actually committed on; the PNGs of the later frames stay on disk
            # as provenance but are not part of the 3 s replay.
            record["frames"] = [
                frame for frame in record["frames"] if frame["frame"] <= observed_death
            ]
        record["window"]["anchored_last_frame"] = death_frame
        record["window"]["last_frame"] = observed_death
        return record
    except HarnessError as error:
        raise DeathAnalysisError(f"engine protocol error: {error}") from error
    finally:
        if block is not None:
            try:
                block.close()
            except Exception:  # pragma: no cover
                pass
        try:
            client.request("QUIT", timeout=20.0)
        except Exception:
            pass
        client.close()
        try:
            segment_path(shm_name).unlink()
        except (OSError, ValueError):
            pass


def _tail(path: Path, limit: int = 2000) -> str:
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:  # pragma: no cover
        pass
    return ""


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def analyse_window(record: dict, rules: Rules) -> dict:
    """Build the classification inputs from the captured window and classify."""

    frames = record["frames"]
    if not frames:
        raise FailClosed("no captured frames")
    py_series = [frame["player"]["py"] for frame in frames]
    boxes_series = [frame["hazards"]["boxes"] for frame in frames]
    # ``laser_gaps`` is the distance to the nearest *active* laser, taken from
    # the teacher's own ``geometry.laser_hits``; ``None`` when no laser was
    # being tested that frame.
    laser_gaps: list[float | None] = [
        (frame.get("hazards", {}).get("laser") or {}).get("gap") for frame in frames
    ]
    last = frames[-1]
    bombs = last["player"]["bombs"]
    boss = last.get("boss") or {}

    result = classify_death(
        py_series=py_series,
        boxes_series=boxes_series,
        laser_gaps=laser_gaps,
        bombs_at_death=float(bombs),
        boss_alive=bool(boss.get("alive", False)),
        boss_distance=boss.get("distance"),
        rules=rules,
    )
    result["inputs"] = {
        "py_series": py_series,
        "boxes_series": boxes_series,
        "laser_gaps": laser_gaps,
        "bombs_at_death": float(bombs),
        "boss_alive": bool(boss.get("alive", False)),
        "boss_distance": boss.get("distance"),
        "boss": boss,
    }
    return result


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def build_summary(root: str | Path) -> dict:
    """Scan ``<root>/*/death.json`` and aggregate the causes."""

    root = Path(root)
    if not root.is_dir():
        raise FailClosed(f"summary root not found: {root}")
    deaths = []
    for path in sorted(root.glob("*/death.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FailClosed(f"{path}: unreadable death.json: {error}") from error
        deaths.append(payload)
    if not deaths:
        raise FailClosed(f"no */death.json under {root}")
    deaths.sort(key=lambda item: item.get("source", {}).get("stage_index", 99))

    counts = {label: 0 for label in ALL_LABELS}
    primaries = {label: 0 for label in ALL_LABELS}
    for item in deaths:
        for label in item.get("labels") or []:
            counts[label] = counts.get(label, 0) + 1
        primary = item.get("primary_cause") or "unknown"
        primaries[primary] = primaries.get(primary, 0) + 1

    hypothesis = {
        "statement": (
            "EXP-P1-012 §6.4: all six deaths happen after the three bombs are "
            "gone, at py ~ 416-432 (bottom of the playfield)"
        ),
        "deaths": len(deaths),
        "bottom_pin": counts.get("bottom_pin", 0),
        "bomb_exhausted": counts.get("bomb_exhausted", 0),
        "both": sum(
            1 for item in deaths
            if "bottom_pin" in (item.get("labels") or [])
            and "bomb_exhausted" in (item.get("labels") or [])
        ),
        "primary_is_bottom_pin": primaries.get("bottom_pin", 0),
        "primary_is_bomb_exhausted": primaries.get("bomb_exhausted", 0),
    }
    hypothesis["consistent"] = (
        hypothesis["bottom_pin"] == len(deaths)
        and hypothesis["bomb_exhausted"] == len(deaths)
    )
    return {
        "schema": SCHEMA + "-summary",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(root),
        "deaths": deaths,
        "label_counts": counts,
        "primary_counts": primaries,
        "priority": list(PRIORITY),
        "hypothesis": hypothesis,
    }


SUMMARY_COLUMNS = (
    "stage", "death_frame", "primary", "labels", "mean_py", "bottom_pin", "bombs",
    "boxes_median", "boxes_max", "laser_gap", "boss_dist", "flags", "gif", "frames",
)


def summary_rows(summary: dict) -> list[dict]:
    rows = []
    for item in summary["deaths"]:
        evidence = item.get("classification", {}).get("evidence", {})
        death = item.get("death", {})
        rows.append(
            {
                "stage": item.get("source", {}).get("stage_name"),
                "stage_index": item.get("source", {}).get("stage_index"),
                "death_frame": item.get("death", {}).get("frame"),
                "primary": item.get("primary_cause"),
                "labels": "+".join(item.get("labels") or []),
                "mean_py": evidence.get("bottom_pin", {}).get("mean_py"),
                "bottom_pin": evidence.get("bottom_pin", {}).get("fired"),
                "bombs": evidence.get("bomb_exhausted", {}).get("bombs_at_death"),
                "boxes_median": evidence.get("bullet_wall", {}).get("median_boxes"),
                "boxes_max": evidence.get("bullet_wall", {}).get("max_boxes"),
                "laser_gap": evidence.get("laser", {}).get("nearest_gap_px"),
                "boss_dist": evidence.get("boss_pattern", {}).get("boss_distance"),
                "flags": (
                    death.get("frame_event_flags", {})
                    .get("death_frame", {})
                    .get("raw")
                ),
                "gif": item.get("files", {}).get("gif"),
                "frames": item.get("window", {}).get("frames"),
            }
        )
    return rows


def write_summary(summary: dict, out_dir: Path) -> dict:
    rows = summary_rows(summary)
    write_json(out_dir / "summary.json", summary)
    lines = [
        "# EXP-P1-016 death analysis summary",
        "",
        f"- generated: {summary['generated_at']}",
        f"- deaths analysed: **{len(summary['deaths'])}**",
        f"- priority (primary = first fired): {' > '.join(summary['priority'])}",
        "",
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(SUMMARY_COLUMNS)) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_cell(row.get(column)) for column in SUMMARY_COLUMNS) + " |"
        )
    lines.append("")
    lines.append("## label counts (multi-label)")
    lines.append("")
    lines.append("| label | deaths | primary |")
    lines.append("|---|---|---|")
    for label in ALL_LABELS:
        lines.append(
            f"| {label} | {summary['label_counts'].get(label, 0)} | "
            f"{summary['primary_counts'].get(label, 0)} |"
        )
    hypothesis = summary["hypothesis"]
    lines += [
        "",
        "## hypothesis check",
        "",
        f"- statement: {hypothesis['statement']}",
        f"- bottom_pin: {hypothesis['bottom_pin']}/{hypothesis['deaths']}",
        f"- bomb_exhausted: {hypothesis['bomb_exhausted']}/{hypothesis['deaths']}",
        f"- both together: {hypothesis['both']}/{hypothesis['deaths']}",
        f"- primary == bottom_pin: {hypothesis['primary_is_bottom_pin']}",
        f"- primary == bomb_exhausted: {hypothesis['primary_is_bomb_exhausted']}",
        f"- consistent with the EXP-P1-012 claim: **{hypothesis['consistent']}**",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_lines = [",".join(SUMMARY_COLUMNS)]
    for row in rows:
        csv_lines.append(",".join(_csv(row.get(column)) for column in SUMMARY_COLUMNS))
    (out_dir / "summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return {"rows": len(rows), "markdown": str(out_dir / "summary.md"),
            "json": str(out_dir / "summary.json"), "csv": str(out_dir / "summary.csv")}


def _cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _csv(value) -> str:
    text = "" if value is None else _cell(value)
    if "," in text:
        return '"' + text.replace('"', '""') + '"'
    return text


# ---------------------------------------------------------------------------
# boss helper
# ---------------------------------------------------------------------------
def boss_distance_from(hazards, player) -> dict:
    """Nearest boss row: alive flag plus the distance to the hurtbox centre."""

    centre_x, centre_y = player.position
    if player.hurtbox_min is not None and player.hurtbox_max is not None:
        centre_x = (player.hurtbox_min[0] + player.hurtbox_max[0]) / 2.0
        centre_y = (player.hurtbox_min[1] + player.hurtbox_max[1]) / 2.0
    best = None
    for enemy in hazards.enemies:
        if not int(getattr(enemy, "flags1", 0)) & ENEMY_FLAG_BOSS:
            continue
        world = getattr(enemy, "world_position", None) or enemy.position
        distance = math.hypot(float(world[0]) - centre_x, float(world[1]) - centre_y)
        if best is None or distance < best["distance"]:
            best = {
                "alive": True,
                "slot": int(enemy.slot),
                "life": int(enemy.life),
                "x": float(world[0]),
                "y": float(world[1]),
                "distance": round(distance, 4),
            }
    return best or {"alive": False, "distance": None}


# ---------------------------------------------------------------------------
# run mode
# ---------------------------------------------------------------------------
def run_one(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_xvfb:
        # The ``env`` role needs a GL context; this re-execs the whole command
        # under ``xvfb-run -a`` exactly once (V400_DEATH_XVFB guards the loop).
        _maybe_reexec_under_xvfb(sys.argv[1:])
    rules = Rules(
        bottom_band_y=args.bottom_band_y,
        bottom_window_frames=args.bottom_window,
        bullet_wall_boxes=args.bullet_wall_boxes,
        bullet_wall_min_frames=args.bullet_wall_min_frames,
        bullet_wall_window_frames=args.bullet_wall_window,
        laser_near_px=args.laser_near_px,
        boss_radius_px=args.boss_radius,
    )
    try:
        run = load_replay(args.replay)
        report = run["report"]
        stage_index = int(report["stage"])
        seed = int(report["seed"])
        if args.stage is not None and int(args.stage) != stage_index:
            raise FailClosed(
                f"--stage {args.stage} disagrees with the replay's stage {stage_index}"
            )
        if args.seed is not None and int(args.seed) != seed:
            raise FailClosed(
                f"--seed {args.seed} disagrees with the replay's seed {seed}"
            )
        death_frame = int(args.death_frame)
        survival = report.get("survival_frames")
        if survival is not None and death_frame != int(survival):
            raise FailClosed(
                f"--death-frame {death_frame} is not the report's survival_frames "
                f"{survival}; the replayed death would not be the recorded one"
            )
        record = capture(
            run=run,
            out_dir=out_dir,
            death_frame=death_frame,
            pre_frames=int(args.pre_frames),
            engine_dir=args.engine,
            expected_sha=args.expected_sha,
            data_template=args.data_template,
            timeout=float(args.timeout),
            save_pngs=not args.no_png,
        )
        classification = analyse_window(record, rules)
    except EngineShaMismatch as error:
        print(json.dumps({"status": "engine_sha_mismatch", "reason": str(error)},
                         ensure_ascii=False), flush=True)
        return EXIT_ENGINE_SHA
    except FailClosed as error:
        print(json.dumps({"status": "fail_closed", "reason": str(error)},
                         ensure_ascii=False), flush=True)
        return EXIT_FAIL_CLOSED
    except DeathAnalysisError as error:
        print(json.dumps({"status": "engine_error", "reason": str(error)},
                         ensure_ascii=False), flush=True)
        return EXIT_ENGINE

    frames = record["frames"]
    last = frames[-1]
    gif_path = out_dir / "replay.gif"
    gif_record = None
    if not args.no_gif:
        indices = []
        for frame in frames:
            png = out_dir / "frames" / frame["png"] if frame["png"] else None
            if png is None or not png.is_file():
                raise SystemExit("frames were not saved but a GIF was requested")
            indices.append(quantize_rgba(_read_png_rgba(png)))
        gif_record = write_gif(
            gif_path,
            indices,
            delay_cs=int(args.gif_delay_cs),
            prefer_pillow=not args.force_stdlib_gif,
        )
        gif_record["path"] = str(gif_path)

    flags = annotate_flags([frame["events"]["value"] for frame in frames])
    nearest_window = None
    for frame in frames:
        candidate = frame["hazards"]["verdict"]["nearest"]
        if candidate is None:
            continue
        if nearest_window is None or float(candidate["gap"]) < float(nearest_window["gap"]):
            nearest_window = {
                "kind": candidate["kind"],
                "slot": candidate["slot"],
                "gap": float(candidate["gap"]),
                "frame": frame["frame"],
            }
    decisions_in_window = sorted({frame["decision"] for frame in frames})
    death_payload = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": {
            "replay": run["path"],
            "replay_sha256": run["sha256"],
            "stage_index": stage_index,
            "stage_name": STAGE_NAMES.get(stage_index, f"stage{stage_index}"),
            "seed": seed,
            "difficulty": int(report["difficulty"]),
            "team": int(report.get("team") or 0),
            "report_survival_frames": report.get("survival_frames"),
            "report_deaths": report.get("deaths_final"),
            "report_bombs_final": report.get("bombs_final"),
            "report_stop_reason": report.get("stop_reason"),
            "tape_frames": run["total_frames"],
            "decisions": len(run["decisions"]),
            "frames_per_decision": run["frames_per_decision"],
        },
        "engine": {
            "binary": record["engine_binary"],
            "sha256": record["engine_sha256"],
            "expected_sha256": record["expected_sha256"],
            "match": record["sha_match"],
            "role": record["role"],
            "readback": record["readback"],
            "shm": record["shm"],
            "reset": record["reset_command"],
            "stderr_tail": record.get("engine_stderr_tail", ""),
        },
        "replay_check": {
            "expected_death_frame": death_frame,
            "observed_death_frame": record["death_signals"]["observed_death_frame"],
            "first_signal": record["death_signals"]["first_signal"],
            "first_hit_frame": record["death_signals"].get("first_hit_frame"),
            "first_dying_frame": record["death_signals"].get("first_dying_frame"),
            "offset_frames": (
                record["death_signals"]["observed_death_frame"] - death_frame
            ),
            "tolerance_frames": record["death_signals"]["tolerance_frames"],
            "match": True,
            "match_rule": (
                "the commit signal (STATE deaths or PLAYER_DEATH_COMMITTED) must "
                "land inside the recorded decision, i.e. within "
                "frames_per_decision frames of --death-frame"
            ),
            "fast_forward_frames": record.get("ff_frames"),
            "fast_forward_seconds": record.get("ff_seconds"),
            "wall_seconds": record.get("wall_seconds"),
            "pixel_stamp_mismatches": record["pixel_mismatches"],
            "blank_frames": record["blank_frames"],
        },
        "window": {
            "pre_frames": int(args.pre_frames),
            "first_frame": record["window"]["first_frame"],
            "last_frame": record["window"]["last_frame"],
            "frames": len(frames),
            "fps": REFERENCE_FPS,
            "seconds": round(len(frames) / REFERENCE_FPS, 4),
            "decisions": decisions_in_window,
        },
        "death": {
            "frame": int(last["frame"]),
            "decision": int(last["decision"]),
            "stage_frame": last["state"].get("f1"),
            "state_frame": last["state"].get("f0"),
            "player": last["player"],
            "nearest_hazard": last["hazards"]["verdict"]["nearest"],
            "nearest_in_window": nearest_window,
            "hazard_counts": {
                "bullets": last["hazards"]["bullets"],
                "lasers": last["hazards"]["lasers"],
                "enemies": last["hazards"]["enemies"],
                "lethal_boxes": last["hazards"]["boxes"],
            },
            "bombs_at_death": last["player"]["bombs"],
            "bottom_band": {
                "mean_py": classification["evidence"]["bottom_pin"]["mean_py"],
                "threshold_py": rules.bottom_band_y,
                "window_frames": rules.bottom_window_frames,
                "inside": classification["evidence"]["bottom_pin"]["fired"],
            },
            "frame_event_flags": flags,
        },
        "labels": classification["labels"],
        "primary_cause": classification["primary"],
        "classification": classification,
        "rules": rules.as_dict(),
        "rules_text": rules.as_text(),
        "teacher_at_death": last["teacher"],
        "files": {
            "frames_dir": "frames",
            "frame_pngs": len(frames) if not args.no_png else 0,
            "gif": gif_record,
            "frames_jsonl": "frames.jsonl",
            "decisions_jsonl": "decisions.jsonl",
        },
    }
    write_jsonl(out_dir / "frames.jsonl", frames)
    write_jsonl(
        out_dir / "decisions.jsonl",
        [
            run["decisions"][index]
            for index in decisions_in_window
            if index < len(run["decisions"])
        ],
    )
    write_json(out_dir / "death.json", death_payload)
    print(
        json.dumps(
            {
                "status": "ok",
                "stage": STAGE_NAMES.get(stage_index),
                "death_frame": death_frame,
                "labels": classification["labels"],
                "primary": classification["primary"],
                "gif": None if gif_record is None else gif_record["path"],
                "out": str(out_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return EXIT_OK


def _read_png_rgba(path: Path) -> np.ndarray:
    """Decode a PNG this tool wrote (RGBA8, filter 0, non-interlaced)."""

    import zlib  # noqa: PLC0415

    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise FailClosed(f"{path} is not a PNG")
    offset = 8
    width = height = None
    idat = bytearray()
    while offset < len(payload):
        length = struct.unpack_from(">I", payload, offset)[0]
        tag = payload[offset + 4:offset + 8]
        data = payload[offset + 8:offset + 8 + length]
        offset += 12 + length
        if tag == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", data)
            if (depth, color, interlace) != (8, 6, 0):
                raise FailClosed(f"{path}: unsupported PNG (depth/color/interlace)")
        elif tag == b"IDAT":
            idat += data
        elif tag == b"IEND":
            break
    if width is None:
        raise FailClosed(f"{path}: PNG has no IHDR")
    raw = zlib.decompress(bytes(idat))
    stride = width * 4
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride + 1)
    if int(rows[:, 0].max()) != 0:
        raise FailClosed(f"{path}: unexpected PNG filter type")
    return rows[:, 1:].reshape(height, width, 4).copy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--replay", type=Path, default=None,
                        help="teacher JSONL of the run whose death is analysed")
    parser.add_argument("--death-frame", type=int, default=None,
                        help="the frame the death was committed on (== survival_frames)")
    parser.add_argument("--stage", type=int, default=None,
                        help="stage index; checked against the replay, never assumed")
    parser.add_argument("--seed", type=int, default=None,
                        help="reset seed; checked against the replay, never assumed")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--mode", choices=("run", "summary"), default="run")
    parser.add_argument("--summary-root", type=Path, default=None,
                        help="--mode summary: directory holding */death.json")
    parser.add_argument("--pre-frames", type=int, default=DEFAULT_PRE_FRAMES,
                        help="lookback frames captured before the death (default 180)")
    parser.add_argument("--engine", default=DEFAULT_ENGINE,
                        help="engine build directory containing th08-modern")
    parser.add_argument("--expected-sha", default=EXPECTED_ENGINE_SHA256)
    parser.add_argument("--data-template", default="/root/v400-data/g1/inst0",
                        help="data dir to copy/link the archives from")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--no-png", action="store_true", help="skip the PNG sequence")
    parser.add_argument("--no-gif", action="store_true", help="skip the GIF")
    parser.add_argument("--gif-delay-cs", type=int, default=5,
                        help="GIF frame delay in centiseconds (5 == 20 fps)")
    parser.add_argument("--force-stdlib-gif", action="store_true",
                        help="ignore Pillow and use the bundled encoder")
    parser.add_argument("--no-xvfb", action="store_true",
                        help="do not re-exec under xvfb-run (DISPLAY must already work)")
    parser.add_argument("--bottom-band-y", type=float,
                        default=DEFAULT_RULES.bottom_band_y)
    parser.add_argument("--bottom-window", type=int,
                        default=DEFAULT_RULES.bottom_window_frames)
    parser.add_argument("--bullet-wall-boxes", type=int,
                        default=DEFAULT_RULES.bullet_wall_boxes)
    parser.add_argument("--bullet-wall-min-frames", type=int,
                        default=DEFAULT_RULES.bullet_wall_min_frames)
    parser.add_argument("--bullet-wall-window", type=int,
                        default=DEFAULT_RULES.bullet_wall_window_frames)
    parser.add_argument("--laser-near-px", type=float,
                        default=DEFAULT_RULES.laser_near_px)
    parser.add_argument("--boss-radius", type=float,
                        default=DEFAULT_RULES.boss_radius_px)
    parser.add_argument("--list-deaths", action="store_true",
                        help="print the EXP-P1-012 death table and exit")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_deaths:
        print(json.dumps(
            [
                {"stage_index": stage, "stage": STAGE_NAMES.get(stage),
                 "death_frame": frame, "replay": f".tmp_v400_g1/psn/psn_stage{stage}.jsonl"}
                for stage, frame in EXP_P1_012_DEATHS
            ],
            ensure_ascii=False, indent=2,
        ))
        return EXIT_OK
    try:
        if args.mode == "summary":
            root = args.summary_root or args.out
            if root is None:
                print(json.dumps({"status": "usage",
                                  "reason": "--mode summary needs --summary-root"},
                                 ensure_ascii=False), flush=True)
                return EXIT_USAGE
            summary = build_summary(root)
            written = write_summary(summary, Path(root))
            print(json.dumps({"status": "ok", **written}, ensure_ascii=False), flush=True)
            return EXIT_OK
        missing = [
            name for name, value in (
                ("--replay", args.replay), ("--death-frame", args.death_frame),
                ("--out", args.out),
            ) if value is None
        ]
        if missing:
            print(json.dumps({"status": "usage", "reason": f"missing {missing}"},
                             ensure_ascii=False), flush=True)
            return EXIT_USAGE
        return run_one(args)
    except FailClosed as error:
        print(json.dumps({"status": "fail_closed", "reason": str(error)},
                         ensure_ascii=False), flush=True)
        return EXIT_FAIL_CLOSED


if __name__ == "__main__":
    raise SystemExit(main())
