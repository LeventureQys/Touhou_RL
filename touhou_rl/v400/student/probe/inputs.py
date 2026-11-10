"""The four pre-registered input tensors A/B/C/D, built from one native frame.

EXP-P3-010 fixes the candidate set; this module is the only place that turns a
384x448 native playfield into those tensors, so training and evaluation cannot
disagree about what "input B" means.

======================  ======================  =============================
code                    tensor                  how
======================  ======================  =============================
``A`` 64x64             ``(1, 64, 64)``         obs-v1's own plane (frozen)
``B`` 96x112            ``(1, 112, 96)``        4x4 integer block mean
``C`` 192x224           ``(1, 224, 192)``       2x2 integer block mean
``D`` 64x64 + 128x128   ``(2, 128, 128)``       ch0 = native 128x128 window
                                                around the ship (zero outside
                                                the playfield), ch1 = A grown
                                                to 128x128 by 2x nearest
======================  ======================  =============================

``D``'s window is aligned to the label window: its origin is
``anchor - 16 px``, so the 96x96 scored window sits exactly at ``[16:112]`` of
the 128x128 crop.  That makes "which input pixel belongs to which label cell" a
pure index arithmetic fact, not an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from touhou_rl.v400.labels.spec import PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH
from touhou_rl.v400.student.probe.labels import CELL_PX, WINDOW_CELLS, WINDOW_PX
from touhou_rl.v400.student.probe.native import block_mean

#: Margin (px) around the scored window that input D also sees.
LOCAL_MARGIN_PX = 16
#: D's actual window side: 96 + 2 * 16 = 128.
LOCAL_SIDE_PX = WINDOW_PX + 2 * LOCAL_MARGIN_PX


@dataclass(frozen=True)
class InputSpec:
    """One pre-registered candidate input."""

    name: str
    channels: int
    height: int
    width: int
    #: Human-readable definition, copied into every card and checkpoint.
    definition: str
    #: True when the tensor needs the player position (input D only).
    needs_player: bool = False
    #: True when the tensor is a local crop rather than the whole field.
    local: bool = False

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)

    @property
    def pixels(self) -> int:
        """Input pixels per frame; the tie-break the selection rule uses."""

        return int(self.height) * int(self.width)


INPUT_SPECS: dict[str, InputSpec] = {
    "A": InputSpec(
        name="A", channels=1, height=64, width=64,
        definition="64x64 whole field (obs-v1 plane, 6x7 block mean)",
    ),
    "B": InputSpec(
        name="B", channels=1, height=112, width=96,
        definition="96x112 whole field (4x4 block mean of the native field)",
    ),
    "C": InputSpec(
        name="C", channels=1, height=224, width=192,
        definition="192x224 whole field (2x2 block mean, half native resolution)",
    ),
    "D": InputSpec(
        name="D", channels=2, height=LOCAL_SIDE_PX, width=LOCAL_SIDE_PX,
        definition=(
            "64x64 whole field + 128x128 native window around the ship "
            "(ch0 native, ch1 = A at 2x nearest)"),
        needs_player=True, local=True,
    ),
}

#: Input pixels per frame, the selection rule's size order.
INPUT_AREAS: dict[str, int] = {name: spec.pixels for name, spec in INPUT_SPECS.items()}


def smallest_first() -> list[str]:
    """The pre-registered "smallest input first" order."""

    return sorted(INPUT_SPECS, key=lambda name: (INPUT_AREAS[name], name))


def input_pixels(name: str) -> int:
    return INPUT_AREAS[name]


def _as_native(native: np.ndarray) -> np.ndarray:
    array = np.asarray(native)
    if array.dtype != np.uint8:
        raise ValueError(f"native playfield must be uint8, got {array.dtype}")
    if array.shape != (PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH):
        raise ValueError(
            f"native playfield must be {(PLAYFIELD_HEIGHT, PLAYFIELD_WIDTH)}, "
            f"got {array.shape}")
    return array


def local_window(native: np.ndarray, anchor_x: int, anchor_y: int) -> np.ndarray:
    """``128x128`` native crop whose ``[16:112]`` is the scored window.

    Pixels outside the playfield are 0 (black).  The ship stays centred because
    the anchor is the snapped label-window origin, never a clamped edge.
    """

    array = _as_native(native)
    origin_x = int(anchor_x) - LOCAL_MARGIN_PX
    origin_y = int(anchor_y) - LOCAL_MARGIN_PX
    out = np.zeros((LOCAL_SIDE_PX, LOCAL_SIDE_PX), dtype=np.uint8)
    src_x0, src_y0 = max(0, origin_x), max(0, origin_y)
    src_x1 = min(PLAYFIELD_WIDTH, origin_x + LOCAL_SIDE_PX)
    src_y1 = min(PLAYFIELD_HEIGHT, origin_y + LOCAL_SIDE_PX)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - origin_x
        dst_y0 = src_y0 - origin_y
        out[dst_y0:dst_y0 + (src_y1 - src_y0),
            dst_x0:dst_x0 + (src_x1 - src_x0)] = array[src_y0:src_y1, src_x0:src_x1]
    return out


def build_input(
    name: str,
    fields: Mapping[str, np.ndarray],
    *,
    anchor_x: int | None = None,
    anchor_y: int | None = None,
) -> np.ndarray:
    """Build one candidate tensor as ``float32`` in ``[0, 1]``.

    ``fields`` is the mapping :func:`touhou_rl.v400.student.probe.native.encode_native`
    returns, so the block means are computed once per frame and shared.
    """

    spec = INPUT_SPECS.get(str(name))
    if spec is None:
        raise KeyError(f"unknown input {name!r}; expected one of {sorted(INPUT_SPECS)}")
    native = _as_native(fields["native"])

    if name == "A":
        plane = fields.get("play64")
        if plane is None:
            plane = block_mean(native, (64, 64))
        return (np.asarray(plane, dtype=np.uint8).astype(np.float32) / 255.0)[None, :, :]
    if name == "B":
        plane = fields.get("play96x112")
        if plane is None:
            plane = block_mean(native, (112, 96))
        return (np.asarray(plane, dtype=np.uint8).astype(np.float32) / 255.0)[None, :, :]
    if name == "C":
        plane = fields.get("play192x224")
        if plane is None:
            plane = block_mean(native, (224, 192))
        return (np.asarray(plane, dtype=np.uint8).astype(np.float32) / 255.0)[None, :, :]

    # D: native local window + the whole field as coarse context.
    if anchor_x is None or anchor_y is None:
        raise ValueError("input D needs the snapped window anchor (anchor_x, anchor_y)")
    window = local_window(native, int(anchor_x), int(anchor_y))
    whole = fields.get("play64")
    if whole is None:
        whole = block_mean(native, (64, 64))
    grown = np.asarray(whole, dtype=np.uint8).repeat(2, axis=0).repeat(2, axis=1)
    if grown.shape != window.shape:  # pragma: no cover - shapes are fixed
        raise ValueError(f"context {grown.shape} != window {window.shape}")
    stacked = np.stack([window, grown], axis=0).astype(np.float32) / 255.0
    if stacked.shape != spec.shape:
        raise ValueError(f"input D shape {stacked.shape} != {spec.shape}")
    return stacked


def label_crop_spec(name: str, input_hw: tuple[int, int],
                    anchor_x: int, anchor_y: int) -> tuple[slice, slice]:
    """Input-pixel rectangle that the scored window occupies, per input.

    Returned as ``(rows, cols)`` slices of the input tensor.  For the whole
    field inputs the rectangle is fractional (A's cell is smaller than a pixel),
    which is why the head pools logits to the *label grid* first and the window
    is then a cell slice; this helper is for the local window (D) where the
    rectangle is integral.
    """

    if name != "D":
        raise ValueError("whole-field inputs are sliced on the label grid, not on pixels")
    height, width = input_hw
    if (height, width) != (LOCAL_SIDE_PX, LOCAL_SIDE_PX):
        raise ValueError(f"D's window must be {LOCAL_SIDE_PX}px, got {height}x{width}")
    # The window origin is anchor - LOCAL_MARGIN, and the scored window starts
    # at `anchor`, hence a constant 16-px offset in D's pixels.
    return (slice(LOCAL_MARGIN_PX, LOCAL_MARGIN_PX + WINDOW_PX),
            slice(LOCAL_MARGIN_PX, LOCAL_MARGIN_PX + WINDOW_PX))


def window_cell_slice(anchor_x: int, anchor_y: int) -> tuple[slice, slice]:
    """The scored window as 4-px cells inside the 96x112 label grid."""

    col0 = int(anchor_x) // CELL_PX
    row0 = int(anchor_y) // CELL_PX
    return (slice(row0, row0 + WINDOW_CELLS), slice(col0, col0 + WINDOW_CELLS))


__all__ = [
    "INPUT_AREAS",
    "INPUT_SPECS",
    "InputSpec",
    "LOCAL_MARGIN_PX",
    "LOCAL_SIDE_PX",
    "build_input",
    "input_pixels",
    "label_crop_spec",
    "local_window",
    "smallest_first",
    "window_cell_slice",
]
