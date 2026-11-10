"""Native-resolution playfield pixels, and obs-v1's exact 64x64 block mean.

The readback is the same 640x480 RGBA8 block the P2 collector and obs-v1 use
(``rlharness.cpp`` ``PublishPixels``; playfield viewport ``(32,16,384,448)``
from ``Background.cpp:740-743``).  obs-v1 throws 6x7 blocks away before the
student ever sees them; this module keeps the 384x448 grayscale field so the
probe can ask what those blocks cost.

Two functions matter:

* :func:`gray_playfield` -- the luminance obs-v1 computes (integer BT.601,
  ``(77R + 151G + 28B) >> 8``), *before* the block mean.  Byte-identical to
  ``obs_spec._luminance`` on the same crop, which the tests prove.
* :func:`obs_v1_play64` -- obs-v1's own 64x64 plane, implemented by calling
  ``obs_spec.encode_planes`` and keeping the play plane.  It is never
  reimplemented here, so input A cannot drift from the frozen spec.

The other resolutions (96x112, 192x224) are integer block means of the native
field with exactly dividing blocks (4x4 and 2x2), so they are deterministic
and reproducible; no interpolation is involved anywhere.
"""

from __future__ import annotations

import numpy as np

from touhou_rl.v400.labels.spec import (
    PLAYFIELD_HEIGHT,
    PLAYFIELD_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_ORIGIN,
    SCREEN_WIDTH,
)
from touhou_rl.v400.student import obs_spec

#: Re-exported so callers do not have to import two modules.
NATIVE_SHAPE = (PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH)  # (height, width)

#: obs-v1 block geometry: 384/64 = 6 columns, 448/64 = 7 rows per plane pixel.
OBS_V1_BLOCK = (PLAYFIELD_HEIGHT // obs_spec.PLAY_SHAPE[0],
                PLAYFIELD_WIDTH // obs_spec.PLAY_SHAPE[1])


class NativeFrameError(ValueError):
    """The readback is not a 640x480 RGBA8 frame with a playfield in it."""


def check_readback(pixels: np.ndarray) -> None:
    """Fail closed instead of slicing a frame that has no playfield."""

    if not isinstance(pixels, np.ndarray):
        raise NativeFrameError(f"pixels must be a numpy array, got {type(pixels)!r}")
    if pixels.dtype != np.uint8:
        raise NativeFrameError(f"pixels must be uint8, got {pixels.dtype}")
    expected = (SCREEN_HEIGHT, SCREEN_WIDTH, 4)
    if pixels.shape != expected:
        raise NativeFrameError(f"pixels must be {expected}, got {pixels.shape}")


def gray_playfield(pixels: np.ndarray) -> np.ndarray:
    """640x480 RGBA8 -> native 384x448 uint8 luminance (no downsampling).

    Equivalent to ``obs_spec._luminance`` on the playfield crop; the same
    integer BT.601 weights and the same shift, so a block mean of this array is
    exactly what obs-v1 stores.
    """

    check_readback(pixels)
    x0, y0 = SCREEN_ORIGIN
    crop = pixels[y0:y0 + PLAYFIELD_HEIGHT, x0:x0 + PLAYFIELD_WIDTH]
    if crop.shape[:2] != NATIVE_SHAPE:
        raise NativeFrameError(
            f"readback {pixels.shape} does not contain the "
            f"{PLAYFIELD_WIDTH}x{PLAYFIELD_HEIGHT} playfield")
    red = crop[:, :, 0].astype(np.uint16)
    green = crop[:, :, 1].astype(np.uint16)
    blue = crop[:, :, 2].astype(np.uint16)
    weights = obs_spec.GRAY_WEIGHTS
    lum = ((red * weights[0] + green * weights[1] + blue * weights[2])
           >> obs_spec.GRAY_SHIFT)
    return lum.astype(np.uint8)


def obs_v1_play64(pixels: np.ndarray) -> np.ndarray:
    """obs-v1's ``play64`` plane, produced by the frozen encoder itself."""

    play, _hud = obs_spec.encode_planes(pixels)
    return play


def block_mean(native: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Integer block mean of a native field down to ``shape`` (rows, cols).

    ``sum // n`` over an exactly dividing block, which is what obs-v1 does for
    6x7; no float is involved, so two runs give identical bytes.
    """

    height, width = native.shape
    out_h, out_w = int(shape[0]), int(shape[1])
    if out_h <= 0 or out_w <= 0:
        raise NativeFrameError(f"target shape must be positive, got {shape}")
    if height % out_h or width % out_w:
        raise NativeFrameError(
            f"{height}x{width} does not divide into {out_h}x{out_w} equal blocks")
    block_h, block_w = height // out_h, width // out_w
    out = native.reshape(out_h, block_h, out_w, block_w).sum(axis=(1, 3))
    return (out // (block_h * block_w)).astype(np.uint8)


def encode_native(pixels: np.ndarray) -> dict[str, np.ndarray]:
    """One readback -> the four pre-registered full-field resolutions.

    ``64x64`` is obs-v1's own plane; ``96x112`` and ``192x224`` are block means
    of the native field.  Input D is *not* built here because it needs the
    player position; see :mod:`touhou_rl.v400.student.probe.inputs`.
    """

    native = gray_playfield(pixels)
    play64 = obs_v1_play64(pixels)
    # The native block mean must reproduce obs-v1 byte for byte; if it does not,
    # one of the two encoders moved and the probe must not silently pick one.
    if not np.array_equal(block_mean(native, obs_spec.PLAY_SHAPE), play64):
        raise NativeFrameError(
            "native block mean disagrees with obs_spec.encode_planes; "
            "refusing to build the probe inputs from two different encoders")
    return {
        "native": native,
        "play64": play64,
        "play96x112": block_mean(native, (112, 96)),
        "play192x224": block_mean(native, (224, 192)),
    }
