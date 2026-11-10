"""Lethal-occupancy grid labels on the fixed playfield grid (P2 §3.3).

The grid is the P2 specification's own: the playfield is 384x448 playfield
pixels, rasterised at native resolution and downsampled to 1/4, i.e. 96x112
cells of 4x4 px (:data:`touhou_rl.v400.labels.spec.GRID_CELL`).  Cell
``(row, col)`` covers playfield ``[col*4, (col+1)*4) x [row*4, (row+1)*4)``.

Three channels are stored per frame, each derived from ``HAZARDS`` only:

* ``occupied``      — any lethal family (bullet / enemy body / laser), P2 §3.3;
* ``bullet_occupied``— bullets only, the literal "bullet occupancy" of the task;
* ``distance``      — per-cell distance in playfield px to the nearest lethal
  bullet AABB (0 inside), clipped to 255 for storage.

The bullet/enemy AABBs come from :mod:`touhou_rl.v400.teacher.geometry`, so the
grid cannot drift from the model validated in EXP-P1-004.  A box that merely
touches a cell counts as occupying it (the engine's own comparison is
inclusive), which is why the rasteriser rounds *out*: it never marks fewer
cells than the box overlaps.

The distance field is exact, not a per-axis approximation: for each cell the
minimum over boxes of ``hypot(dx, dy)`` is taken.  The implementation chunks the
boxes so the (boxes x 96 x 112) intermediate stays small.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from touhou_rl.v400.labels.spec import (
    DISTANCE_CLIP_PX,
    GRID_CELL,
    GRID_HEIGHT,
    GRID_WIDTH,
    PLAYFIELD_HEIGHT,
    PLAYFIELD_WIDTH,
)
from touhou_rl.v400.teacher import geometry

#: Boxes per chunk for the distance field; keeps the temporary at a few MB.
_DISTANCE_CHUNK = 32


@dataclass(frozen=True)
class GridOccupancy:
    """One frame's occupancy label, arrays in ``(row, col)`` order."""

    width: int
    height: int
    cell: int
    occupied: np.ndarray          # bool[height, width], any lethal family
    bullet_occupied: np.ndarray   # bool[height, width], bullets only
    distance: np.ndarray          # float32[height, width], px to nearest bullet
    bullet_boxes: int
    enemy_boxes: int
    lasers_active: int
    laser_cells: int
    model: str
    missing: tuple[str, ...] = ()

    @property
    def occupied_cells(self) -> int:
        return int(self.occupied.sum())

    @property
    def bullet_cells(self) -> int:
        return int(self.bullet_occupied.sum())

    @property
    def nearest_distance(self) -> float | None:
        """Smallest distance to a lethal bullet; ``None`` when there are none."""

        if self.bullet_boxes == 0:
            return None
        return float(self.distance.min())

    def as_summary(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "cell": self.cell,
            "model": self.model,
            "bullet_boxes": self.bullet_boxes,
            "enemy_boxes": self.enemy_boxes,
            "lasers_active": self.lasers_active,
            "occupied_cells": self.occupied_cells,
            "bullet_cells": self.bullet_cells,
            "laser_cells": self.laser_cells,
            "nearest_distance": self.nearest_distance,
        }


def _cell_bounds(minimum: float, maximum: float, cell: int, count: int) -> tuple[int, int]:
    """Inclusive cell index range a closed interval touches on one axis.

    ``ceil(min/cell) - 1`` includes the cell whose right edge the box touches at
    exactly ``min``; ``floor(max/cell)`` includes the cell whose left edge it
    touches at exactly ``max``.  Both are the inclusive semantics the engine
    uses for its lethal test, so a box on a cell boundary marks both cells.
    """

    low = int(math.ceil(float(minimum) / cell)) - 1
    high = int(math.floor(float(maximum) / cell))
    low = max(0, low)
    high = min(count - 1, high)
    return low, high


def _rasterize_boxes(
    boxes: geometry.Boxes, width: int, height: int, cell: int
) -> np.ndarray:
    """Boolean mask of every cell any AABB overlaps (inclusive)."""

    mask = np.zeros((height, width), dtype=bool)
    for index in range(len(boxes)):
        x0, x1 = float(boxes.mins[index, 0]), float(boxes.maxs[index, 0])
        y0, y1 = float(boxes.mins[index, 1]), float(boxes.maxs[index, 1])
        col0, col1 = _cell_bounds(x0, x1, cell, width)
        row0, row1 = _cell_bounds(y0, y1, cell, height)
        if col0 > col1 or row0 > row1:
            continue
        mask[row0 : row1 + 1, col0 : col1 + 1] = True
    return mask


def _distance_to_boxes(
    boxes: geometry.Boxes,
    width: int,
    height: int,
    cell: int,
    clip: int = DISTANCE_CLIP_PX,
) -> np.ndarray:
    """Exact per-cell Euclidean distance to the nearest AABB, clipped and uint8."""

    if len(boxes) == 0:
        return np.full((height, width), clip, dtype=np.float32)

    xs = (np.arange(width, dtype=np.float32) + 0.5) * np.float32(cell)
    ys = (np.arange(height, dtype=np.float32) + 0.5) * np.float32(cell)
    best = np.full((height, width), np.float32(clip), dtype=np.float32)
    mins = np.asarray(boxes.mins, dtype=np.float32)
    maxs = np.asarray(boxes.maxs, dtype=np.float32)
    for start in range(0, len(boxes), _DISTANCE_CHUNK):
        stop = min(start + _DISTANCE_CHUNK, len(boxes))
        dx = np.maximum(
            mins[start:stop, None, 0] - xs[None, :],
            xs[None, :] - maxs[start:stop, None, 0],
        )
        dx = np.maximum(dx, np.float32(0.0))
        dy = np.maximum(
            mins[start:stop, None, 1] - ys[None, :],
            ys[None, :] - maxs[start:stop, None, 1],
        )
        dy = np.maximum(dy, np.float32(0.0))
        d2 = (
            dy[:, :, None].astype(np.float32) ** 2
            + dx[:, None, :].astype(np.float32) ** 2
        )
        best = np.minimum(best, np.sqrt(d2).min(axis=0))
    return np.minimum(best, np.float32(clip)).astype(np.float32)


def _laser_masks(
    lasers: Iterable[Any], width: int, height: int, cell: int
) -> tuple[np.ndarray, int, int]:
    """Cell mask for active lasers, using ``Player::CalcLaserHitbox`` geometry.

    The engine rotates the tested point into the laser frame and compares it to
    an axis-aligned rectangle whose centre is ``(origin.x + start + len/2,
    origin.y)``, half length ``len/2`` (``*0.7`` once ``start > 0``) and half
    thickness ``width/4``.  Here the tested point is the cell centre, so the
    mask is the set of cells whose centre lies inside the beam.  Zero active
    lasers were observed in the P1 corpus, so this path is implemented from
    source and unit-tested against :func:`geometry.laser_hits` rather than
    being claimed as measured.
    """

    mask = np.zeros((height, width), dtype=bool)
    active = 0
    for laser in lasers:
        if getattr(laser, "in_use", None) == 0:
            continue
        if int(getattr(laser, "state", -1)) != geometry.LASER_STATE_ACTIVE:
            continue
        active += 1
        start = float(laser.start_offset)
        end = float(laser.end_offset)
        length = float(np.float32(end - start))
        half_len = length / 2.0 if start <= 0.0 else length * 0.7 / 2.0
        half_thick = float(np.float32(float(laser.width))) / 4.0
        ox, oy = float(laser.position[0]), float(laser.position[1])
        cx = ox + start + length / 2.0
        cy = oy
        xs = (np.arange(width, dtype=np.float32) + np.float32(0.5)) * np.float32(cell)
        ys = (np.arange(height, dtype=np.float32) + np.float32(0.5)) * np.float32(cell)
        px = xs[None, :] - np.float32(ox)
        py = ys[:, None] - np.float32(oy)
        angle = -float(laser.angle)
        cos = np.float32(math.cos(angle))
        sin = np.float32(math.sin(angle))
        rx = px * cos - py * sin + np.float32(ox)
        ry = px * sin + py * cos + np.float32(oy)
        dx = np.abs(rx - np.float32(cx)) - np.float32(half_len)
        dy = np.abs(ry - np.float32(cy)) - np.float32(half_thick)
        mask |= (dx <= 0.0) & (dy <= 0.0)
    return mask, active, int(mask.sum())


def hazard_occupancy(
    hazards: Any,
    *,
    width: int = GRID_WIDTH,
    height: int = GRID_HEIGHT,
    cell: int = GRID_CELL,
    model: geometry.Model = geometry.MODEL_SPEC,
) -> GridOccupancy:
    """Build one frame's occupancy label from a ``HAZARDS`` snapshot.

    ``model`` selects the AABB rules; the default ``MODEL_SPEC`` is the literal
    P2 §2.1 geometry (bullet at the snapshot position).  The grid is about what
    is lethal *now*, so the one-tick engine advance used for future danger is
    deliberately not applied here.
    """

    if width * cell > PLAYFIELD_WIDTH or height * cell > PLAYFIELD_HEIGHT:
        raise ValueError(
            f"grid {width}x{height} at {cell} px exceeds playfield "
            f"{PLAYFIELD_WIDTH}x{PLAYFIELD_HEIGHT}"
        )
    missing: list[str] = []
    bullets = geometry.bullet_boxes(getattr(hazards, "bullets", ()), model)
    enemies = geometry.enemy_boxes(getattr(hazards, "enemies", ()), model)
    lasers = tuple(getattr(hazards, "lasers", ()))

    # A hazard row without a collision width is dropped by geometry (never
    # guessed); surface that as a missing-input count instead of silently
    # producing an empty grid.
    exported_bullets = len(getattr(hazards, "bullets", ()))
    if exported_bullets and len(bullets) == 0:
        missing.append("bullet_boxes_empty_despite_rows")
    exported_enemies = len(getattr(hazards, "enemies", ()))
    if exported_enemies and len(enemies) == 0:
        missing.append("enemy_boxes_empty_despite_rows")

    bullet_mask = _rasterize_boxes(bullets, width, height, cell)
    enemy_mask = _rasterize_boxes(enemies, width, height, cell)
    laser_mask, lasers_active, laser_cells = _laser_masks(lasers, width, height, cell)
    distance = _distance_to_boxes(bullets, width, height, cell)

    return GridOccupancy(
        width=width,
        height=height,
        cell=cell,
        occupied=bullet_mask | enemy_mask | laser_mask,
        bullet_occupied=bullet_mask,
        distance=distance,
        bullet_boxes=len(bullets),
        enemy_boxes=len(enemies),
        lasers_active=lasers_active,
        laser_cells=laser_cells,
        model=model.name,
        missing=tuple(missing),
    )


def encode_occupancy(grid: GridOccupancy) -> dict[str, Any]:
    """Serialise a grid for the JSONL dataset (bit-pack + base64, no float text)."""

    packed = np.packbits(grid.occupied.astype(np.uint8), axis=None)
    packed_bullets = np.packbits(grid.bullet_occupied.astype(np.uint8), axis=None)
    quantised = np.clip(np.rint(grid.distance), 0, DISTANCE_CLIP_PX).astype(np.uint8)
    payload = {
        **grid.as_summary(),
        "distance_clip_px": int(DISTANCE_CLIP_PX),
        "occupied_packed_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
        "bullet_occupied_packed_b64": base64.b64encode(
            packed_bullets.tobytes()
        ).decode("ascii"),
        "distance_uint8_b64": base64.b64encode(quantised.tobytes()).decode("ascii"),
        "missing": list(grid.missing),
    }
    return payload


def decode_occupancy(payload: dict[str, Any]) -> GridOccupancy:
    """Inverse of :func:`encode_occupancy`; used by the tests and the audit."""

    width = int(payload["width"])
    height = int(payload["height"])
    cell = int(payload["cell"])
    count = width * height
    occupied = np.unpackbits(
        np.frombuffer(base64.b64decode(payload["occupied_packed_b64"]), dtype=np.uint8)
    )[:count].astype(bool).reshape(height, width)
    bullets = np.unpackbits(
        np.frombuffer(
            base64.b64decode(payload["bullet_occupied_packed_b64"]), dtype=np.uint8
        )
    )[:count].astype(bool).reshape(height, width)
    distance = (
        np.frombuffer(base64.b64decode(payload["distance_uint8_b64"]), dtype=np.uint8)
        .reshape(height, width)
        .astype(np.float32)
    )
    return GridOccupancy(
        width=width,
        height=height,
        cell=cell,
        occupied=occupied,
        bullet_occupied=bullets,
        distance=distance,
        bullet_boxes=int(payload.get("bullet_boxes", 0)),
        enemy_boxes=int(payload.get("enemy_boxes", 0)),
        lasers_active=int(payload.get("lasers_active", 0)),
        laser_cells=int(payload.get("laser_cells", 0)),
        model=str(payload.get("model", "")),
        missing=tuple(payload.get("missing", ())),
    )


def occupied_cell_rects(
    grid: GridOccupancy, mask: np.ndarray | None = None
) -> list[tuple[int, int, int, int]]:
    """Screen rectangles of occupied cells, for the overlay.

    Returns half-open ``(x0, y0, x1, y1)`` in **playfield** pixels; the overlay
    adds :data:`SCREEN_ORIGIN` itself so the coordinate conversion stays in one
    place.
    """

    use = grid.occupied if mask is None else mask
    rows, cols = np.nonzero(use)
    cell = grid.cell
    return [
        (int(col) * cell, int(row) * cell, (int(col) + 1) * cell, (int(row) + 1) * cell)
        for row, col in zip(rows, cols)
    ]


def cells_within(distance: np.ndarray, radius_px: float) -> np.ndarray:
    """Cells whose centre is within ``radius_px`` of a lethal bullet (diagnostic)."""

    return distance <= float(radius_px)


def grid_from_hazards_sequence(
    hazards: Sequence[Any], *, model: geometry.Model = geometry.MODEL_SPEC
) -> list[GridOccupancy]:
    """Convenience for tests: one grid per hazard frame."""

    return [hazard_occupancy(frame, model=model) for frame in hazards]
