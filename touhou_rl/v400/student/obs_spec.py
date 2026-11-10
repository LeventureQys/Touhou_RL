"""Frozen observation spec **obs-v1** for the v4.0.0 pixel student (P2-WP5 / P3).

The spec is a *contract*, not a hyper-parameter bag: P3, P4 and P5 all consume
observations through this module, and any change to the observation meaning (not
to a training hyper-parameter) requires a new :data:`OBS_SPEC_VERSION` and a new
document, ``obs-spec-v2.md``.  :func:`obs_spec_digest` is the sha256 of the
canonical JSON of :func:`obs_spec`; checkpoints record it and refuse to load
against a different digest (see :class:`ObsSpecMismatch`).

The frozen content follows the P2-WP4 resolution comparison (EXP-P2-003 §6):
``play64`` — 64x64, 6x7 block mean of the native 384x448 playfield — was both
cheaper and stronger than ``play96`` on the same recipe, so v1 freezes 64x64 and
keeps 96x112 as an optional, non-default variant.  That advantage was measured
only in the ``dive`` collection policy; it is not a claim that 64x64 wins in
general (EXP-P2-003 §6).

The encoder below is a byte-exact reimplementation of the P2 collector's
``encode_frame`` (``tools/rl/v400_p2_bulk_collect.py``): integer BT.601 luminance
``(77R + 151G + 28B) >> 8`` and integer block means ``sum // n``.  No float
rounding and no dithering, so a live evaluation frame and a stored dataset frame
are the same bytes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

import numpy as np

OBS_SPEC_VERSION: Final = "obs-v1"

#: Decision period in *logic frames*: one student action is held this long.
#: Must equal ``touhou_rl.v400.gym.env.DECISION_FRAMES`` and the teacher's
#: ``touhou_rl.v400.teacher.config.DECISION_FRAMES`` (03 §3.3).
DECISION_FRAMES: Final = 2

#: Number of stacked playfield frames: ``t-3 .. t`` (oldest first).
FRAME_STACK: Final = 4

#: Stacking order documented for the tensor layout below.
STACK_ORDER: Final = "oldest-first t-3,t-2,t-1,t"

#: Play plane name/shape/dtype in the P2 dataset (``play64.u8``).
PLAY_PLANE: Final = "play64"
PLAY_SHAPE: Final[tuple[int, int]] = (64, 64)

#: HUD plane name/shape/dtype in the P2 dataset (``hud.u8``): the score/lives/
#: bombs numeric bands at native resolution, green channel only.
HUD_PLANE: Final = "hud"
HUD_SHAPE: Final[tuple[int, int]] = (56, 160)
HUD_FRAMES: Final = 1

#: Grayscale conversion, frozen to the P2 collector's integer BT.601 weights.
GRAY_WEIGHTS: Final[tuple[int, int, int]] = (77, 151, 28)
GRAY_SHIFT: Final = 8

#: Playfield viewport on the 640x480 readback (Background.cpp:740-743).
PLAYFIELD_ORIGIN: Final[tuple[int, int]] = (32, 16)
PLAYFIELD_SHAPE: Final[tuple[int, int]] = (384, 448)  # (width, height)

#: HUD bands and columns (touhou_rl.v400.labels.spec.HUD_LAYOUT).
HUD_BANDS: Final[tuple[tuple[int, int], ...]] = ((48, 72), (80, 96), (96, 112))
HUD_X0: Final = 480
HUD_X1: Final = 640

#: Normalisation: ``float32(uint8) / 255.0`` per plane, range [0, 1], no
#: per-channel mean/std subtraction.  Same as the P2 detector (EXP-P2-003 §4).
NORMALIZATION: Final = "float32 / 255.0; per plane; no mean/std; range [0,1]"

#: Action space: ``cand = focus * 9 + move`` (18 move-focus classes) x bomb 2.
N_MOVE_CLASSES: Final = 18
N_BOMB_CLASSES: Final = 2
ACTION_DURATION_DECISIONS: Final = 1
ACTION_DURATION_FRAMES: Final = DECISION_FRAMES
#: The teacher searches (first_action, duration) roots with duration in decision
#: slots; the student emits exactly duration == 1, i.e. one decision == 2 logic
#: frames.  A teacher ``Plan.duration`` of 2 therefore equals two consecutive
#: student decisions, never one.
TEACHER_DURATION_UNITS: Final = "decision slots (1 slot = DECISION_FRAMES logic frames)"

#: Soft-label temperature for ``pi_teacher(a) ∝ exp(score(a) / tau)``.
#: Part of the *training config*, not of the observation meaning: P3 DAgger may
#: calibrate it from the teacher score distribution and record it in the
#: checkpoint, without a new obs version.
TEACHER_TAU_DEFAULT: Final = 1.0

#: Inputs deliberately **not** in obs-v1, listed so a reader does not have to
#: guess.  The 8-decision action history of 13-P3 §2.1 is reserved for obs-v2
#: (it needs a decision-history contract shared by DAgger and deployment).
EXCLUDED_INPUTS: Final[tuple[str, ...]] = (
    "privileged engine state (position, bullets, lives as numbers)",
    "recent action history (reserved for obs-v2)",
    "pixel-reward head outputs from P2",
)


class ObsSpecError(ValueError):
    """Raised when an observation does not match the frozen spec."""


class ObsSpecMismatch(ObsSpecError):
    """Raised when a checkpoint was trained under a different obs spec."""


def obs_spec() -> dict[str, object]:
    """The frozen spec as JSON-serialisable data (the digest covers this)."""

    return {
        "version": OBS_SPEC_VERSION,
        "play": {
            "plane": PLAY_PLANE,
            "shape": list(PLAY_SHAPE),
            "dtype": "uint8",
            "channels": 1,
            "content": "BT.601 integer luminance of the 384x448 playfield, "
                       "6x7 block means at native 64x64",
            "source_region": {
                "screen_origin": list(PLAYFIELD_ORIGIN),
                "playfield": list(PLAYFIELD_SHAPE),
            },
            "stack": FRAME_STACK,
            "stack_order": STACK_ORDER,
        },
        "hud": {
            "plane": HUD_PLANE,
            "shape": list(HUD_SHAPE),
            "dtype": "uint8",
            "channels": 1,
            "content": "green channel of the score/lives/bombs bands at native "
                       "resolution, columns 480:640",
            "bands": [list(band) for band in HUD_BANDS],
            "stack": HUD_FRAMES,
        },
        "decision_frames": DECISION_FRAMES,
        "normalization": NORMALIZATION,
        "action": {
            "move_focus_classes": N_MOVE_CLASSES,
            "bomb_classes": N_BOMB_CLASSES,
            "encoding": "cand = focus*9 + move (0..17); mask = cand_to_mask(cand, bomb)",
            "duration_decisions": ACTION_DURATION_DECISIONS,
            "duration_frames": ACTION_DURATION_FRAMES,
            "teacher_duration_units": TEACHER_DURATION_UNITS,
        },
        "teacher_interface": {
            "soft_label": "pi_teacher(a) = softmax(scores[18] / tau)",
            "tau_default": TEACHER_TAU_DEFAULT,
            "kl_direction": "KL(pi_teacher || pi_student)",
            "verified_mask": "per-candidate trust mask; unverified scores are "
                             "already discounted by the teacher (11-P1 §3)",
            "bomb_target": "teacher bomb label, BCE against student bomb logits",
            "student_output": "move_logits[18] + bomb_logits[2]",
        },
        "excluded_inputs": list(EXCLUDED_INPUTS),
    }


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def obs_spec_digest() -> str:
    """sha256 of the canonical spec; stored in every checkpoint."""

    return hashlib.sha256(_canonical(obs_spec())).hexdigest()


def check_obs_spec(version: str | None, digest: str | None) -> None:
    """Fail closed if a checkpoint/sample was produced under another spec."""

    if version != OBS_SPEC_VERSION:
        raise ObsSpecMismatch(
            f"obs spec version {version!r} is not {OBS_SPEC_VERSION!r}; "
            "retrain or bump the spec version explicitly"
        )
    expected = obs_spec_digest()
    if digest != expected:
        raise ObsSpecMismatch(
            f"obs spec digest {digest!r} does not match {expected!r}"
        )


def _luminance(crop: np.ndarray) -> np.ndarray:
    """Integer BT.601 luminance of an RGBA crop, as uint32."""

    red = crop[:, :, 0].astype(np.uint16)
    green = crop[:, :, 1].astype(np.uint16)
    blue = crop[:, :, 2].astype(np.uint16)
    return ((red * GRAY_WEIGHTS[0] + green * GRAY_WEIGHTS[1] + blue * GRAY_WEIGHTS[2])
            >> GRAY_SHIFT).astype(np.uint32)


def encode_planes(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """640x480 RGBA ``uint8`` readback -> ``(play64, hud)`` uint8 planes.

    Byte-exact with the P2 collector.  Raises :class:`ObsSpecError` rather than
    slicing a frame that does not contain the playfield, so an eval bug cannot
    silently train on background pixels.
    """

    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ObsSpecError(
            f"pixels must be uint8 HxWx4, got {pixels.dtype} {pixels.shape}"
        )
    x0, y0 = PLAYFIELD_ORIGIN
    width, height = PLAYFIELD_SHAPE
    crop = pixels[y0:y0 + height, x0:x0 + width]
    if crop.shape[0] != height or crop.shape[1] != width:
        raise ObsSpecError(
            f"readback {pixels.shape} does not contain the {width}x{height} playfield"
        )
    lum = _luminance(crop)

    oh, ow = PLAY_SHAPE
    bh, bw = height // oh, width // ow
    play = (lum.reshape(oh, bh, ow, bw).sum(axis=(1, 3)) // (bh * bw)).astype(np.uint8)
    if play.shape != PLAY_SHAPE:
        raise ObsSpecError(f"play plane shape {play.shape} != {PLAY_SHAPE}")

    hud_rows = [pixels[a:b, HUD_X0:HUD_X1, 1] for (a, b) in HUD_BANDS]
    hud = np.concatenate(hud_rows, axis=0)
    if hud.shape != HUD_SHAPE:
        raise ObsSpecError(f"hud plane shape {hud.shape} != {HUD_SHAPE}")
    return play, hud


def normalize_play(stack: np.ndarray) -> np.ndarray:
    """``uint8 (S,64,64)`` -> ``float32 (S,64,64)`` in [0,1]."""

    arr = np.asarray(stack, dtype=np.uint8)
    if arr.shape != (FRAME_STACK, *PLAY_SHAPE):
        raise ObsSpecError(f"play stack shape {arr.shape} != {(FRAME_STACK, *PLAY_SHAPE)}")
    return arr.astype(np.float32) / 255.0


def normalize_hud(hud: np.ndarray) -> np.ndarray:
    """``uint8 (56,160)`` -> ``float32 (1,56,160)`` in [0,1]."""

    arr = np.asarray(hud, dtype=np.uint8)
    if arr.shape != HUD_SHAPE:
        raise ObsSpecError(f"hud shape {arr.shape} != {HUD_SHAPE}")
    return arr.astype(np.float32)[None, ...] / 255.0


class ObservationBuffer:
    """Rolling ``FRAME_STACK`` playfield frames plus the current HUD strip.

    At episode start the buffer repeats the first frame until it is full.  That
    padding is part of the spec (it is deterministic and recorded), so a live
    evaluation never invents motion it did not see.
    """

    def __init__(self) -> None:
        self._play: list[np.ndarray] = []
        self._hud: np.ndarray | None = None
        self.frames_seen = 0

    def reset(self) -> None:
        self._play = []
        self._hud = None
        self.frames_seen = 0

    def push(self, play: np.ndarray, hud: np.ndarray) -> None:
        play = np.asarray(play, dtype=np.uint8)
        hud = np.asarray(hud, dtype=np.uint8)
        if play.shape != PLAY_SHAPE:
            raise ObsSpecError(f"play frame shape {play.shape} != {PLAY_SHAPE}")
        if hud.shape != HUD_SHAPE:
            raise ObsSpecError(f"hud frame shape {hud.shape} != {HUD_SHAPE}")
        self._play.append(play.copy())
        if len(self._play) > FRAME_STACK:
            del self._play[0]
        self._hud = hud.copy()
        self.frames_seen += 1

    def stack(self) -> np.ndarray:
        if not self._play:
            raise ObsSpecError("observation buffer is empty; push at least one frame")
        frames = list(self._play)
        while len(frames) < FRAME_STACK:
            frames.insert(0, frames[0])
        return np.stack(frames, axis=0).astype(np.uint8)

    def hud(self) -> np.ndarray:
        if self._hud is None:
            raise ObsSpecError("observation buffer has no HUD frame yet")
        return self._hud

    def arrays(self) -> dict[str, np.ndarray]:
        """Model-ready arrays: ``play (4,64,64) float32``, ``hud (1,56,160)``."""

        return {
            "play": normalize_play(self.stack()),
            "hud": normalize_hud(self.hud()),
        }


__all__ = [
    "OBS_SPEC_VERSION",
    "DECISION_FRAMES",
    "FRAME_STACK",
    "PLAY_PLANE",
    "PLAY_SHAPE",
    "HUD_PLANE",
    "HUD_SHAPE",
    "HUD_FRAMES",
    "N_MOVE_CLASSES",
    "N_BOMB_CLASSES",
    "ACTION_DURATION_FRAMES",
    "TEACHER_TAU_DEFAULT",
    "ObsSpecError",
    "ObsSpecMismatch",
    "ObservationBuffer",
    "check_obs_spec",
    "encode_planes",
    "normalize_hud",
    "normalize_play",
    "obs_spec",
    "obs_spec_digest",
]
