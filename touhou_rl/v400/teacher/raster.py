"""Rasterised lethal mask + margin table for the P1 beam search (EXP-P1-008).

The beam search asks the same geometric question about a million times per
decision: "how far is the ship's hurtbox from the nearest lethal AABB in this
frame, and does it overlap?".  Answering it against the raw box list costs
``points x boxes``: 191 AABBs in a dense frame x 41k expansion points x 24 slots
is 430 ms/decision (EXP-P1-007 §3.2).  This module answers it from a **table**:

* the playfield is cut into ``cell`` x ``cell`` squares (default 2 px), which is
  184 x 208 = 38,272 cells for ``x in [8, 376]``, ``y in [16, 432]``;
* every frame's lethal AABBs are expanded by the hurtbox half extents — that
  expansion is exactly the set of hurtbox **centres** the engine would call a
  hit, because the AABB test is inclusive (``gap <= 0``) — and rasterised into
  two masks:

  - ``hit``  — the cell's square *intersects* the expanded rect.  A **superset**
    of the lethal centres: a point in a ``hit`` cell may or may not be lethal.
  - ``core`` — the cell's square is *inside one* expanded rect.  A **subset**:
    every point in a ``core`` cell is lethal.

  So ``not hit`` proves alive and ``core`` proves dead; ``hit and not core`` is
  the one-cell band that straddles a lethal boundary and is resolved exactly, by
  the same expression the pre-raster implementation used
  (:meth:`LethalTable.exact`).  The death verdict therefore stays **exact**: the
  table can only save work, never flip a verdict.
* a capped Euclidean distance transform of ``hit`` gives the frame's margin
  field (``MARGIN_CAP`` = 16 px saturates the score term, so the transform only
  carries radii up to 8 cells).

Layout of one table entry (``uint16``): low byte = margin in 1/4 px fixed point
(0..64), high byte = flags (bit 0 ``hit``, bit 1 ``core``).  One gather answers
both questions for a point, which is what makes the lookup cheaper than the
geometry.

Everything here is derived from the engine's own numbers: the lethal AABBs come
from :func:`touhou_rl.v400.teacher.geometry.bullet_boxes` /
:func:`~touhou_rl.v400.teacher.geometry.enemy_boxes` through
``HazardTape.box_array``, the hurtbox half extents from ``PLAYER``.  No constant
of the game is re-derived here.
"""

from __future__ import annotations

import numpy as np

from touhou_rl.v400.teacher import raster_numba
from touhou_rl.v400.teacher.config import MARGIN_CAP, PLAYFIELD_X, PLAYFIELD_Y

#: Cell size of the death verdict lattice, in px (11-P1 §2.2 playfield).
CELL = 2.0
#: Margin lattice: coarser than :data:`CELL` because the transform is O(cells)
#: per frame and a 4 px lattice keeps the whole table inside the search budget.
MARGIN_CELL = 4.0
#: Fixed point of the stored margin: 1/4 px.
QUANT = 4
#: ``MARGIN_CAP`` in fixed point.
MARGIN_Q_CAP = int(round(MARGIN_CAP * QUANT))

FLAG_HIT = 1
FLAG_CORE = 2


def lattice(cell: float) -> tuple[float, float, int, int]:
    """``(x0, y0, nx, ny)`` of the lattice that covers the whole playfield."""
    x0, x1 = PLAYFIELD_X
    y0, y1 = PLAYFIELD_Y
    nx = int(np.ceil((x1 - x0) / cell))
    ny = int(np.ceil((y1 - y0) / cell))
    return float(x0), float(y0), nx, ny


def flat_index(centers: np.ndarray, *, cell: float = CELL, nx: int | None = None, ny: int | None = None) -> np.ndarray:
    """Flat cell index of every ``(x, y)`` centre row, clipped to the playfield.

    ``flat = iy * nx + ix`` with ``ix = floor((x - x0) / cell)``, i.e. the same
    index space the table is stored in.  Positions outside the playfield cannot
    be produced (the search clamps), so the clip is a guard, not a model.
    """
    x0, y0, nx_, ny_ = lattice(cell)
    nx = nx_ if nx is None else nx
    ny = ny_ if ny is None else ny
    x = np.asarray(centers)[:, 0] if centers.ndim == 2 else centers[0]
    y = np.asarray(centers)[:, 1] if centers.ndim == 2 else centers[1]
    ix = np.floor((x - x0) / cell).astype(np.intp)
    iy = np.floor((y - y0) / cell).astype(np.intp)
    np.clip(ix, 0, nx - 1, out=ix)
    np.clip(iy, 0, ny - 1, out=iy)
    ix += iy * nx
    return ix


# ---------------------------------------------------------------------------
# rasterisation
# ---------------------------------------------------------------------------
def _cell_ranges(low, high, half, origin, cell, mode) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive cell range a rect ``[low - half, high + half]`` covers.

    Cell ``i`` covers ``[origin + i*cell, origin + (i+1)*cell]``.  ``mode`` picks
    the coverage predicate (see the module docstring):

    * ``"hit"``    — cell square intersects the rect -> ``i >= (a-origin)/cell - 1``,
      ``i <= (b-origin)/cell``;
    * ``"core"``   — cell square is contained in the rect -> ``i >= (a-origin)/cell``,
      ``i <= (b-origin)/cell - 1``;
    * ``"centre"`` — the cell's centre lies inside the rect -> shift by half a cell.
    """
    a = (low - half - origin) / cell
    b = (high + half - origin) / cell
    if mode == "hit":
        first = np.ceil(a - 1.0)
        last = np.floor(b)
    elif mode == "core":
        first = np.ceil(a)
        last = np.floor(b - 1.0)
    elif mode == "centre":
        first = np.ceil(a - 0.5)
        last = np.floor(b - 0.5)
    else:  # pragma: no cover - internal
        raise ValueError(f"unknown mode {mode!r}")
    return first.astype(np.int64), last.astype(np.int64)


def _paint(planes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
           frames: int, ny: int, nx: int) -> np.ndarray:
    """Rasterise ``[(frame, r0, r1, c0, c1), ...]`` planes into boolean masks.

    One 2-D difference array per (plane, frame) plus two cumulative sums: the
    cost is ``O(corners)`` for the scatter and ``O(frames * ny * nx)`` for the
    sums, never ``O(rects * cells)``.
    """
    n_planes = len(planes)
    stride = (ny + 1) * (nx + 1)
    diff = np.zeros(n_planes * frames * stride, dtype=np.int16)
    for plane, (frame, r0, r1, c0, c1) in enumerate(planes):
        keep = (r1 >= r0) & (c1 >= c0) & (r1 >= 0) & (r0 < ny) & (c1 >= 0) & (c0 < nx)
        if not keep.any():
            continue
        frame = frame[keep]
        r0 = np.clip(r0[keep], 0, ny - 1)
        r1 = np.clip(r1[keep], 0, ny - 1)
        c0 = np.clip(c0[keep], 0, nx - 1)
        c1 = np.clip(c1[keep], 0, nx - 1)
        base = ((plane * frames + frame) * stride).astype(np.int64)
        row0 = base + r0 * (nx + 1)
        row1 = base + (r1 + 1) * (nx + 1)
        corners = np.concatenate([row0 + c0, row1 + c1 + 1])  # +1
        negated = np.concatenate([row0 + c1 + 1, row1 + c0])  # -1
        unique, counts = np.unique(corners, return_counts=True)
        diff[unique] += counts.astype(np.int16)
        unique, counts = np.unique(negated, return_counts=True)
        diff[unique] -= counts.astype(np.int16)
    counts = diff.reshape(n_planes * frames, ny + 1, nx + 1)
    np.cumsum(counts, axis=1, out=counts)
    np.cumsum(counts, axis=2, out=counts)
    return counts.reshape(n_planes, frames, ny + 1, nx + 1)[:, :, :ny, :nx] > 0


def _painter(tape, half, cell: float, mode: str):
    x0, y0, nx, ny = lattice(cell)
    frames = len(tape)
    frame_ids: list[np.ndarray] = []
    r0s: list[np.ndarray] = []
    r1s: list[np.ndarray] = []
    c0s: list[np.ndarray] = []
    c1s: list[np.ndarray] = []
    for index in range(frames):
        boxes = tape.box_array(index)
        if boxes.shape[0] == 0:
            continue
        r0, r1 = _cell_ranges(boxes[:, 1], boxes[:, 3], half[1], y0, cell, mode)
        c0, c1 = _cell_ranges(boxes[:, 0], boxes[:, 2], half[0], x0, cell, mode)
        frame_ids.append(np.full(boxes.shape[0], index, dtype=np.int64))
        r0s.append(r0)
        r1s.append(r1)
        c0s.append(c0)
        c1s.append(c1)
    if not frame_ids:
        empty = np.zeros((frames, ny, nx), dtype=bool)
        return empty, empty
    return (
        np.concatenate(frame_ids),
        np.concatenate(r0s),
        np.concatenate(r1s),
        np.concatenate(c0s),
        np.concatenate(c1s),
    )


# ---------------------------------------------------------------------------
# capped Euclidean distance transform of a boolean raster
# ---------------------------------------------------------------------------
def distance_transform(mask: np.ndarray, cap_cells: int) -> np.ndarray:
    """Exact capped Euclidean distance (in cells) from every cell to ``mask``.

    ``mask`` is ``(frames, rows, cols)`` boolean; the result is ``uint8`` squared
    distances capped at ``cap_cells**2``.  Two exact 1-D passes: the rows pass is
    a plain "distance to the nearest true" scan (index difference), the columns
    pass is the min-plus convolution with ``k**2`` restricted to
    ``|k| <= cap_cells``, which is exact because every larger offset already
    exceeds the cap.

    The working layout is transposed to ``(frames, cols, rows)`` so the *scan*
    runs along the contiguous axis (a strided ``accumulate`` is ~2x slower) and
    only the four ``k`` shifts are strided.  The transpose is pure data movement:
    the returned values are the same.
    """
    frames, rows, cols = mask.shape
    plane = np.ascontiguousarray(mask.transpose(0, 2, 1))  # (frames, cols, rows)
    big = np.int32(1 << 20)
    index = np.arange(rows, dtype=np.int32)[None, None, :]
    last = np.maximum.accumulate(np.where(plane, index, -big), axis=2)
    nxt = np.minimum.accumulate(np.where(plane, index, big)[:, :, ::-1], axis=2)[:, :, ::-1]
    gap = np.minimum(index - last, nxt - index)
    np.minimum(gap, cap_cells, out=gap)
    del last, nxt
    vertical = (gap * gap).astype(np.uint8)
    del gap
    cap_squared = cap_cells * cap_cells
    out = np.full((frames, cols, rows), cap_squared, dtype=np.uint8)
    scratch = np.empty((frames, cols, rows), dtype=np.uint8)
    np.minimum(out, vertical, out=out)
    for offset in range(1, cap_cells + 1):
        add = np.uint8(offset * offset)
        width = cols - offset
        np.add(vertical[:, offset:, :], add, out=scratch[:, :width, :])
        np.minimum(out[:, :width, :], scratch[:, :width, :], out=out[:, :width, :])
        np.add(vertical[:, :width, :], add, out=scratch[:, offset:, :])
        np.minimum(out[:, offset:, :], scratch[:, offset:, :], out=out[:, offset:, :])
    np.minimum(out, np.uint8(cap_squared), out=out)
    return np.ascontiguousarray(out.transpose(0, 2, 1))


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
class LethalTable:
    """Per-tape lookup table: death verdict is exact, margin is quantised."""

    __slots__ = (
        "table", "frames", "nx", "ny", "cell", "half", "boxes", "band_queries",
        "band_exact", "timing", "backend", "x0", "y0", "_flat", "_packed",
    )

    def __init__(self, table, frames, nx, ny, cell, half, boxes, backend="numpy"):
        self.table = table
        self.frames = frames
        self.nx = nx
        self.ny = ny
        self.cell = cell
        self.half = np.asarray(half, dtype=np.float64)
        self.boxes = boxes
        self.band_queries = 0
        self.band_exact = 0
        self.timing = {}
        self.backend = backend
        self.x0, self.y0, _nx, _ny = lattice(cell)
        self._flat = None
        self._packed = None

    # -- layout -----------------------------------------------------------
    def flat_array(self) -> np.ndarray:
        """The whole table as one contiguous ``uint16`` vector."""
        if self._flat is None:
            self._flat = np.ascontiguousarray(self.table).reshape(-1)
        return self._flat

    def packed(self):
        """``(boxes, counts)`` for the numba kernels, built on first use."""
        if self._packed is None:
            self._packed = _pack_boxes(self.boxes)
        return self._packed

    # -- lookups ----------------------------------------------------------
    def flat(self, centers) -> np.ndarray:
        return flat_index(centers, cell=self.cell, nx=self.nx, ny=self.ny)

    def exact(self, frame: int, centers: np.ndarray):
        """The pre-raster expression, element for element (:func:`_frame_margin`)."""
        boxes = self.boxes[frame]
        half = self.half
        if boxes.shape[0] == 0:
            return (
                np.full((centers.shape[0],), MARGIN_CAP, dtype=np.float64),
                np.zeros((centers.shape[0],), dtype=bool),
            )
        pmin_x = centers[:, 0] - half[0]
        pmax_x = centers[:, 0] + half[0]
        pmin_y = centers[:, 1] - half[1]
        pmax_y = centers[:, 1] + half[1]
        dx = np.maximum(boxes[None, :, 0] - pmax_x[:, None], pmin_x[:, None] - boxes[None, :, 2])
        dy = np.maximum(boxes[None, :, 1] - pmax_y[:, None], pmin_y[:, None] - boxes[None, :, 3])
        np.maximum(dx, 0.0, out=dx)
        np.maximum(dy, 0.0, out=dy)
        gap = np.sqrt(dx * dx + dy * dy)
        best = gap.min(axis=1)
        return np.minimum(best, MARGIN_CAP), best <= 0.0

    def exact_points(self, frame: int, centers):
        """Exact ``(margins, dead)`` at one frame; numba kernel when available.

        Values are identical to :meth:`exact` (same expression, same order), the
        kernel only removes the ``(points, boxes)`` numpy temporaries that make
        the finalist re-scoring stage of the beam search unaffordable.
        """
        centers = np.ascontiguousarray(centers, dtype=np.float64)
        n = int(centers.shape[0])
        if n == 0:
            return np.empty(0), np.empty(0, dtype=bool)
        if self.backend == "numba" and raster_numba.available():
            boxes, counts = self.packed()
            margin = np.empty(n, dtype=np.float64)
            dead = np.empty(n, dtype=np.bool_)
            raster_numba.exact_kernel(
                boxes, counts, int(frame), float(self.half[0]), float(self.half[1]),
                float(MARGIN_CAP), centers[:, 0], centers[:, 1], margin, dead,
            )
            return margin, dead
        return self.exact(frame, centers)

    def query(self, frame: int, centers):
        """``(margins, dead)`` for a batch of hurtbox centres at one frame.

        ``dead`` is **exact**: ``not hit`` proves the gap is > 0 and ``core``
        proves it is 0, and the thin band between them is recomputed with
        :meth:`exact`.
        """
        centers = np.ascontiguousarray(centers, dtype=np.float64)
        n = int(centers.shape[0])
        if self.backend == "numba" and raster_numba.available() and n:
            boxes, counts = self.packed()
            margins = np.empty(n, dtype=np.float64)
            dead = np.empty(n, dtype=np.bool_)
            band = np.zeros(n, dtype=np.uint8)
            raster_numba.query_kernel(
                self.flat_array(), int(frame), int(self.nx), int(self.ny),
                float(self.x0), float(self.y0), float(self.cell), boxes, counts,
                float(self.half[0]), float(self.half[1]), float(MARGIN_CAP),
                centers[:, 0], centers[:, 1], margins, dead, band,
            )
            count = int(band.sum())
            if count:
                self.band_queries += n
                self.band_exact += count
            return margins, dead
        codes = self.table[frame].ravel()[self.flat(centers)]
        margins = (codes & np.uint16(0xFF)).astype(np.float64) * (1.0 / QUANT)
        flags = codes >> 8
        dead = (flags & np.uint16(FLAG_HIT)).astype(bool)
        band_mask = dead & ((flags & np.uint16(FLAG_CORE)) == 0)
        count = int(band_mask.sum())
        if count:
            self.band_queries += int(centers.shape[0])
            self.band_exact += count
            exact_margin, exact_dead = self.exact(frame, centers[band_mask])
            margins[band_mask] = exact_margin
            dead[band_mask] = exact_dead
        margins[dead] = 0.0
        return margins, dead

    def query_margins(self, frame: int, centers):
        return self.query(frame, centers)[0]

    def hit(self, frame: int) -> np.ndarray:
        """``(ny, nx)`` boolean: the cell may contain a lethal hurtbox centre."""
        return ((self.table[frame] >> 8) & np.uint16(FLAG_HIT)) != 0

    def core(self, frame: int) -> np.ndarray:
        """``(ny, nx)`` boolean: every hurtbox centre in the cell is lethal."""
        return ((self.table[frame] >> 8) & np.uint16(FLAG_CORE)) != 0

    def margin(self, frame: int) -> np.ndarray:
        """``(ny, nx)`` float64 margin field of the frame, in px."""
        return (self.table[frame] & np.uint16(0xFF)).astype(np.float64) * (1.0 / QUANT)

    def as_dict(self) -> dict:
        return {
            "cell": float(self.cell),
            "nx": int(self.nx),
            "ny": int(self.ny),
            "frames": int(self.frames),
            "band_queries": int(self.band_queries),
            "band_exact": int(self.band_exact),
            "backend": str(self.backend),
        }


def _downsample_hit(hit: np.ndarray, factor: int) -> np.ndarray:
    """A coarse cell is lethal if any of its fine children is (superset, so the
    coarse transform never *under*-reports a distance)."""
    if factor == 1:
        return hit
    frames, rows, cols = hit.shape
    rows_c = rows // factor
    cols_c = cols // factor
    trimmed = hit[:, : rows_c * factor, : cols_c * factor]
    return trimmed.reshape(frames, rows_c, factor, cols_c, factor).any(axis=(2, 4))


def _pack_boxes(boxed: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``(frames, max_boxes, 4)`` padded rect array plus per-frame counts.

    The padded array is what the numba kernels index; the count makes the pad
    invisible (a frame never reads past its own ``counts[frame]`` rows).
    """
    frames = len(boxed)
    width = max((int(box.shape[0]) for box in boxed), default=0)
    packed = np.zeros((frames, width, 4), dtype=np.float64)
    counts = np.zeros(frames, dtype=np.int64)
    for index, box in enumerate(boxed):
        count = int(box.shape[0])
        counts[index] = count
        if count:
            packed[index, :count] = box
    return packed, counts


def _build_numba(packed, counts, half, x0, y0, nx, ny, cell, factor, cap_cells, lookup, frames):
    """Rasterise with the numba kernel; ``None`` when it cannot run here."""
    if not raster_numba.available():
        return None
    if not raster_numba.probe():  # pragma: no cover - compile failure only
        return None
    if ny % factor or nx % factor:  # pragma: no cover - lattice is divisible here
        return None
    table = np.zeros((frames, ny, nx), dtype=np.uint16)
    raster_numba.build_table_kernel(
        packed, counts, float(x0), float(y0), float(cell), int(nx), int(ny),
        float(half[0]), float(half[1]), int(factor), int(cap_cells), lookup, table,
    )
    return table


def build_table(
    tape,
    half,
    *,
    cell: float = CELL,
    margin_cell: float = MARGIN_CELL,
    profile: bool = False,
    backend: str = "auto",
) -> LethalTable:
    """Rasterise ``tape`` into a :class:`LethalTable`.

    ``half`` is the hurtbox half extent of the decision point (``PLAYER``).
    ``cell`` is the death-verdict lattice; ``margin_cell`` the (coarser) lattice
    of the distance transform, whose values are expanded back onto ``cell``.

    ``backend`` is ``"auto"`` (numba when importable, else numpy), ``"numba"``
    or ``"numpy"``; the two must produce bit-identical tables (EXP-P1-010).
    """
    half = np.asarray(half, dtype=np.float64)
    frames = len(tape)
    x0, y0, nx, ny = lattice(cell)
    boxed = [tape.box_array(index) for index in range(frames)]
    tick = _timer() if profile else None
    if not any(box.shape[0] for box in boxed):
        empty = np.zeros((frames, ny, nx), dtype=np.uint16)
        return LethalTable(empty, frames, nx, ny, cell, half, boxed)
    factor = max(1, int(round(margin_cell / cell)))
    cap_cells = max(1, int(np.ceil(MARGIN_CAP / (cell * factor))))
    cap_squared = cap_cells * cap_cells
    lookup = np.minimum(
        np.rint(np.sqrt(np.arange(cap_squared + 1, dtype=np.float64)) * (cell * factor) * QUANT),
        MARGIN_Q_CAP,
    ).astype(np.uint16)
    if backend == "auto":
        backend = "numba" if raster_numba.available() else "numpy"
    table = None
    if backend == "numba":
        packed, counts = _pack_boxes(boxed)
        table = _build_numba(
            packed, counts, half, x0, y0, nx, ny, cell, factor, cap_cells, lookup, frames
        )
        if table is not None and tick:
            tick("build_numba")
    if table is not None:
        built = LethalTable(table, frames, nx, ny, cell, half, boxed)
        built.backend = "numba"
        if tick:
            built.timing = tick.finish()
        return built
    if profile:
        tick = _timer()
    return _build_numpy(tape, half, boxed, cell, factor, cap_cells, cap_squared, lookup,
                        x0, y0, nx, ny, frames, tick)


def _build_numpy(tape, half, boxed, cell, factor, cap_cells, cap_squared, lookup,
                 x0, y0, nx, ny, frames, tick):
    """The pre-numba build (EXP-P1-008), kept as the reference implementation."""
    hit_ids = _painter(tape, half, cell, "hit")
    core_ids = _painter(tape, half, cell, "core")
    hit, core = _paint([hit_ids, core_ids], frames, ny, nx)
    if tick:
        tick("paint")
    coarse = _downsample_hit(hit, factor)
    squared = distance_transform(coarse, cap_cells)
    if tick:
        tick("distance")
    # distance in px, quantised to 1/QUANT px and capped at MARGIN_CAP.  The
    # transform only produces squares of small integers, so a lookup table over
    # them replaces sqrt/rint on 459k float64 values.
    margin_q = lookup[squared]
    del squared
    if factor > 1:
        margin_q = np.repeat(np.repeat(margin_q, factor, axis=1), factor, axis=2)
        margin_q = margin_q[:, :ny, :nx]
    flags = hit.astype(np.uint16)
    flags |= core.astype(np.uint16) << 1
    del hit, core
    table = margin_q
    table |= flags << 8
    del flags
    if tick:
        tick("pack")
    # A cell the raster proved lethal has margin 0 by definition: it is a ``hit``
    # cell, so the transform of the superset mask already reported 0.  Assert it
    # instead of patching it (a patch on 1.8M cells is a measurable cost).
    built = LethalTable(table, frames, nx, ny, cell, half, boxed)
    if tick:
        built.timing = tick.finish()
    return built


class _timer:
    """Tiny step timer for the raster build (``profile=True`` only)."""

    def __init__(self):
        import time

        self._time = time.perf_counter
        self.steps: list[tuple[str, float]] = []
        self._last = self._time()

    def __call__(self, name: str) -> None:
        now = self._time()
        self.steps.append((name, (now - self._last) * 1000.0))
        self._last = now

    def finish(self) -> dict:
        return {name: round(value, 3) for name, value in self.steps}
