"""The 24x24 cell window around the player: occupancy, future danger, validity.

Everything scored by EXP-P3-010 lives in this window: 24 cells of 4 playfield
pixels, i.e. 96x96 px centred on the ship's ``PLAYER`` position (48 px in every
direction).  The window is **snapped to the frozen 4-px grid** of
``touhou_rl.v400.labels.spec`` so its cells are literally a crop of the P2
label grid -- no second rasteriser, no interpolation, no drift.

Three arrays per frame:

* ``cells``  -- ``occupied`` (bullets | enemy bodies | lasers), the P2 label;
* ``danger`` -- the future-8 danger map: cells whose distance to the nearest
  lethal AABB is ``<= margin_px`` in **any** of the next 8 frames, with the
  distance field coming from ``occupancy.hazard_occupancy`` (P2 geometry, the
  same ``MODEL_SPEC`` the P2 dataset used).  ``margin_px = 8`` matches the
  "danger state" definition of EXP-P3-011 (teacher margin < 8 px), so the two
  cards mean the same thing by "dangerous";
* ``valid``  -- cells that exist (a ship near the edge has a window that leaves
  the 96x112 grid; those cells are *not scored* rather than counted as empty).

The window never invents a label: cells outside the grid are ``valid=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from touhou_rl.v400.labels.occupancy import hazard_occupancy
from touhou_rl.v400.labels.spec import GRID_CELL, GRID_HEIGHT, GRID_WIDTH

#: Cells per side of the scored window (24 cells x 4 px = 96 px = 2 x 48 px).
WINDOW_CELLS = 24
#: Playfield pixels per side of the scored window.
WINDOW_PX = WINDOW_CELLS * GRID_CELL
#: One cell's size in playfield pixels (the P2 grid's own cell).
CELL_PX = GRID_CELL
#: Cells within this many pixels of a lethal AABB count as a danger cell.
DANGER_MARGIN_PX = 8.0
#: Horizon of the danger map, in logic frames (task book: future 8 frames).
DANGER_HORIZON = 8


class WindowError(ValueError):
    """The player position or the grids cannot produce a window."""


@dataclass(frozen=True)
class WindowLabel:
    """One decision instant's scored window."""

    player_x: float
    player_y: float
    #: Top-left playfield pixel of the window, snapped to ``CELL_PX``.
    anchor_x: int
    anchor_y: int
    #: ``(24, 24)`` bool, row-major, ``[row, col]``.
    cells: np.ndarray
    #: ``(24, 24)`` bool, cells that exist.
    valid: np.ndarray
    #: ``(24, 24)`` bool, future-8 danger map (``False`` where invalid).
    danger: np.ndarray
    #: How many future hazard frames the danger map actually saw (<= 8).
    danger_frames: int
    #: Scalar P2 semantics for the still ship, when the caller supplied them.
    would_die_8: bool | None = None

    @property
    def positives(self) -> int:
        return int(self.cells[self.valid].sum())

    @property
    def scored_cells(self) -> int:
        return int(self.valid.sum())

    @property
    def danger_cells(self) -> int:
        return int(self.danger[self.valid].sum())

    def as_summary(self) -> dict[str, Any]:
        return {
            "player_x": self.player_x,
            "player_y": self.player_y,
            "anchor_x": self.anchor_x,
            "anchor_y": self.anchor_y,
            "positives": self.positives,
            "scored_cells": self.scored_cells,
            "danger_cells": self.danger_cells,
            "danger_frames": self.danger_frames,
            "would_die_8": self.would_die_8,
            "full_window": bool(self.valid.all()),
        }

    def pack(self) -> dict[str, np.ndarray]:
        """Storage form: ``uint8`` anchors plus packed bits (72 B per array)."""

        return {
            "cells": np.packbits(self.cells.reshape(-1)),
            "valid": np.packbits(self.valid.reshape(-1)),
            "danger": np.packbits(self.danger.reshape(-1)),
        }


def snap_anchor(player_x: float, player_y: float) -> tuple[int, int]:
    """Window top-left in playfield pixels, snapped to the 4-px label grid.

    The window is centred on the ship up to a ``+-2 px`` snap: the label cells
    must line up with the P2 grid, and a ship never sits exactly on it.
    """

    if not (np.isfinite(player_x) and np.isfinite(player_y)):
        raise WindowError(f"player position is not finite: ({player_x}, {player_y})")
    half = WINDOW_PX // 2
    col = int(np.floor((float(player_x) - half) / CELL_PX + 0.5))
    row = int(np.floor((float(player_y) - half) / CELL_PX + 0.5))
    return col * CELL_PX, row * CELL_PX


def cell_origin(player_x: float, player_y: float) -> tuple[int, int]:
    """Same snap, in cells instead of pixels."""

    anchor_x, anchor_y = snap_anchor(player_x, player_y)
    return anchor_x // CELL_PX, anchor_y // CELL_PX


def grid_arrays(hazards: Any) -> tuple[np.ndarray, np.ndarray]:
    """One ``HAZARDS`` snapshot -> ``(occupied, distance)`` on the P2 grid."""

    grid = hazard_occupancy(hazards)
    return grid.occupied, grid.distance


def min_distance(grids: Iterable[np.ndarray]) -> np.ndarray | None:
    """Element-wise minimum of several 96x112 distance fields."""

    stacked = None
    for field in grids:
        array = np.asarray(field, dtype=np.float32)
        if array.shape != (GRID_HEIGHT, GRID_WIDTH):
            raise WindowError(
                f"distance field {array.shape} != {(GRID_HEIGHT, GRID_WIDTH)}")
        stacked = array if stacked is None else np.minimum(stacked, array)
    return stacked


def window_label(
    occupied: np.ndarray,
    future_distance: np.ndarray | None,
    player_x: float,
    player_y: float,
    *,
    danger_frames: int = 0,
    margin_px: float = DANGER_MARGIN_PX,
    would_die_8: bool | None = None,
) -> WindowLabel:
    """Crop the 24x24 window around the ship out of the P2 grid arrays.

    ``occupied`` is ``hazard_occupancy(...).occupied`` (96x112) and
    ``future_distance`` is the element-wise minimum of the next 8 frames'
    ``distance`` fields (``None`` when the run tail has no future frames).
    """

    occupied = np.asarray(occupied, dtype=bool)
    if occupied.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise WindowError(f"occupied {occupied.shape} != {(GRID_HEIGHT, GRID_WIDTH)}")
    col0, row0 = cell_origin(player_x, player_y)
    anchor_x, anchor_y = col0 * CELL_PX, row0 * CELL_PX

    cells = np.zeros((WINDOW_CELLS, WINDOW_CELLS), dtype=bool)
    valid = np.zeros((WINDOW_CELLS, WINDOW_CELLS), dtype=bool)
    danger = np.zeros((WINDOW_CELLS, WINDOW_CELLS), dtype=bool)

    src_r0, src_r1 = max(0, row0), min(GRID_HEIGHT, row0 + WINDOW_CELLS)
    src_c0, src_c1 = max(0, col0), min(GRID_WIDTH, col0 + WINDOW_CELLS)
    if src_r1 > src_r0 and src_c1 > src_c0:
        dst_r0, dst_c0 = src_r0 - row0, src_c0 - col0
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        cells[dst_r0:dst_r1, dst_c0:dst_c1] = occupied[src_r0:src_r1, src_c0:src_c1]
        valid[dst_r0:dst_r1, dst_c0:dst_c1] = True
        if future_distance is not None:
            field = np.asarray(future_distance, dtype=np.float32)
            if field.shape != (GRID_HEIGHT, GRID_WIDTH):
                raise WindowError(
                    f"distance {field.shape} != {(GRID_HEIGHT, GRID_WIDTH)}")
            danger[dst_r0:dst_r1, dst_c0:dst_c1] = (
                field[src_r0:src_r1, src_c0:src_c1] <= float(margin_px))

    return WindowLabel(
        player_x=float(player_x),
        player_y=float(player_y),
        anchor_x=int(anchor_x),
        anchor_y=int(anchor_y),
        cells=cells,
        valid=valid,
        danger=danger,
        danger_frames=int(danger_frames),
        would_die_8=would_die_8,
    )


def unpack_window(payload: dict[str, np.ndarray], *, count: int = WINDOW_CELLS) -> np.ndarray:
    """Inverse of :meth:`WindowLabel.pack` for one array (tests and replay)."""

    bits = np.unpackbits(np.asarray(payload, dtype=np.uint8))
    return bits[: count * count].astype(bool).reshape(count, count)


__all__ = [
    "CELL_PX",
    "DANGER_HORIZON",
    "DANGER_MARGIN_PX",
    "WINDOW_CELLS",
    "WINDOW_PX",
    "WindowError",
    "WindowLabel",
    "cell_origin",
    "grid_arrays",
    "min_distance",
    "snap_anchor",
    "unpack_window",
    "window_label",
]
