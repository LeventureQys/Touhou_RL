"""numba kernels for the rasterised lethal table (EXP-P1-010).

Optional accelerator for :mod:`touhou_rl.v400.teacher.raster`.  When ``numba``
is importable the three hot pieces run as ``njit(cache=True)`` kernels:

* :func:`build_table_kernel` — paint (rect rasterisation), the capped Euclidean
  distance transform and the ``uint16`` pack, one ``prange`` task per frame;
* :func:`query_kernel` — the table lookup: flat index, one ``uint16`` gather,
  flag decode and the exact AABB fallback for the one-cell boundary band;
* :func:`exact_kernel` — the exact capped AABB gap for a batch of centres (used
  by the beam search's finalist re-scoring stage and by ``_root_nodes``).

**Both backends must agree bit for bit.**  Every kernel mirrors the numpy
expression it replaces, operation by operation: same division (not a reciprocal
multiply) in the cell index, same ``sqrt``, same subtraction order, same integer
distance transform.  ``tests/test_v400_teacher.py`` pins the equality of the two
builds and of the two lookups.

``numba`` is *not* a new dependency: it was already present in the WSL venv
(0.68.1) before this experiment.  If it is missing, or if compilation fails,
:func:`available` returns ``False`` and ``raster.py`` falls back to its pure
numpy implementation unchanged.
"""

from __future__ import annotations

import numpy as np

_NUMBA = False
_REASON = "not probed"

try:  # pragma: no cover - import probe
    from numba import njit, prange

    _NUMBA = True
    _REASON = "numba"
except Exception as exc:  # pragma: no cover - no numba on this interpreter
    _REASON = f"numba import failed: {exc.__class__.__name__}: {exc}"
    njit = None  # type: ignore[assignment]
    prange = range  # type: ignore[assignment]


def available() -> bool:
    """True when the numba kernels can be used."""
    return bool(_NUMBA)


def reason() -> str:
    """Why :func:`available` answered what it did."""
    return _REASON


def _fallback(reason: str) -> None:
    global _NUMBA, _REASON
    _NUMBA = False
    _REASON = reason


if _NUMBA:

    @njit(cache=True, parallel=True, nogil=True)
    def build_table_kernel(  # noqa: D103 - internal, documented in the module docstring
        boxes,        # (frames, max_boxes, 4) float64
        counts,       # (frames,) int64
        x0,
        y0,
        cell,
        nx,
        ny,
        halfx,
        halfy,
        factor,       # margin lattice coarsening (2 == 4 px lattice on a 2 px mask)
        cap_cells,    # cap of the distance transform, in coarse cells
        lookup,       # (cap_cells**2 + 1,) uint16 -- squared distance -> 1/QUANT px
        out,          # (frames, ny, nx) uint16 -- low byte margin, high byte flags
    ):
        frames = out.shape[0]
        nyc = ny // factor
        nxc = nx // factor
        cap_sq = cap_cells * cap_cells
        big = np.int32(1 << 20)
        for f in prange(frames):
            hit = np.zeros((ny, nx), np.uint8)
            core = np.zeros((ny, nx), np.uint8)
            n = counts[f]
            for b in range(n):
                lx = boxes[f, b, 0] - halfx
                hx = boxes[f, b, 2] + halfx
                ly = boxes[f, b, 1] - halfy
                hy = boxes[f, b, 3] + halfy
                # hit: the cell square intersects the expanded rect
                c0 = int(np.ceil((lx - x0) / cell - 1.0))
                c1 = int(np.floor((hx - x0) / cell))
                r0 = int(np.ceil((ly - y0) / cell - 1.0))
                r1 = int(np.floor((hy - y0) / cell))
                if c0 < 0:
                    c0 = 0
                if r0 < 0:
                    r0 = 0
                if c1 > nx - 1:
                    c1 = nx - 1
                if r1 > ny - 1:
                    r1 = ny - 1
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        hit[r, c] = 1
                # core: the cell square is contained in the expanded rect
                c0 = int(np.ceil((lx - x0) / cell))
                c1 = int(np.floor((hx - x0) / cell - 1.0))
                r0 = int(np.ceil((ly - y0) / cell))
                r1 = int(np.floor((hy - y0) / cell - 1.0))
                if c0 < 0:
                    c0 = 0
                if r0 < 0:
                    r0 = 0
                if c1 > nx - 1:
                    c1 = nx - 1
                if r1 > ny - 1:
                    r1 = ny - 1
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        core[r, c] = 1
            # downsample: a coarse cell is "hit" when any of its children is
            coarse = np.zeros((nyc, nxc), np.uint8)
            for r in range(nyc):
                for c in range(nxc):
                    for dr in range(factor):
                        for dc in range(factor):
                            if hit[r * factor + dr, c * factor + dc]:
                                coarse[r, c] = 1
            # capped Euclidean distance transform, in coarse cells
            dist = np.empty(nyc, np.int32)
            vertical = np.empty((nyc, nxc), np.int32)
            for c in range(nxc):
                last = -big
                for r in range(nyc):
                    if coarse[r, c]:
                        last = r
                    dist[r] = r - last
                nxt = big
                for r in range(nyc - 1, -1, -1):
                    if coarse[r, c]:
                        nxt = r
                    d = nxt - r
                    if d < dist[r]:
                        dist[r] = d
                for r in range(nyc):
                    d = dist[r]
                    if d > cap_cells:
                        d = cap_cells
                    vertical[r, c] = d * d
            squared = np.full((nyc, nxc), cap_sq, np.int32)
            for r in range(nyc):
                for c in range(nxc):
                    best = vertical[r, c]
                    if best > cap_sq:
                        best = cap_sq
                    for k in range(1, cap_cells + 1):
                        kk = k * k
                        if c - k >= 0:
                            v = vertical[r, c - k] + kk
                            if v < best:
                                best = v
                        if c + k < nxc:
                            v = vertical[r, c + k] + kk
                            if v < best:
                                best = v
                    squared[r, c] = best
            # pack: low byte margin, high byte flags (bit0 hit, bit1 core)
            for r in range(ny):
                cr = r // factor
                for c in range(nx):
                    flags = np.int32(hit[r, c])
                    if core[r, c]:
                        flags += 2
                    out[f, r, c] = np.uint16(
                        np.int32(lookup[squared[cr, c // factor]]) + (flags << 8)
                    )

    @njit(cache=True, nogil=True)
    def query_kernel(  # noqa: D103
        table_flat,   # (frames * ny * nx,) uint16 view of the table
        frame,
        nx,
        ny,
        x0,
        y0,
        cell,
        boxes,
        counts,
        halfx,
        halfy,
        margin_cap,
        xs,
        ys,
        out_margin,   # (n,) float64
        out_dead,     # (n,) bool_
        out_band,     # (n,) uint8 -- 1 when the exact AABB fallback ran
    ):
        base = frame * nx * ny
        n = xs.shape[0]
        for i in prange(n):
            ix = int(np.floor((xs[i] - x0) / cell))
            iy = int(np.floor((ys[i] - y0) / cell))
            if ix < 0:
                ix = 0
            elif ix > nx - 1:
                ix = nx - 1
            if iy < 0:
                iy = 0
            elif iy > ny - 1:
                iy = ny - 1
            code = np.int32(table_flat[base + iy * nx + ix])
            hit = (code >> 8) & 1
            core = (code >> 8) & 2
            margin = np.float64(code & 0xFF) * 0.25
            dead = False
            band = np.uint8(0)
            if hit == 0:
                dead = False
            elif core != 0:
                dead = True
                margin = 0.0
            else:
                # one-cell band: the original AABB expression, point by point
                band = np.uint8(1)
                pminx = xs[i] - halfx
                pmaxx = xs[i] + halfx
                pminy = ys[i] - halfy
                pmaxy = ys[i] + halfy
                best = np.inf
                for b in range(counts[frame]):
                    dx = boxes[frame, b, 0] - pmaxx
                    other = pminx - boxes[frame, b, 2]
                    if other > dx:
                        dx = other
                    dy = boxes[frame, b, 1] - pmaxy
                    other = pminy - boxes[frame, b, 3]
                    if other > dy:
                        dy = other
                    if dx < 0.0:
                        dx = 0.0
                    if dy < 0.0:
                        dy = 0.0
                    gap = np.sqrt(dx * dx + dy * dy)
                    if gap < best:
                        best = gap
                if best > margin_cap:
                    best = margin_cap
                dead = best <= 0.0
                margin = 0.0 if dead else best
            out_margin[i] = margin
            out_dead[i] = dead
            out_band[i] = band

    @njit(cache=True, nogil=True)
    def exact_kernel(  # noqa: D103
        boxes,
        counts,
        frame,
        halfx,
        halfy,
        margin_cap,
        xs,
        ys,
        out_margin,
        out_dead,
    ):
        n = xs.shape[0]
        for i in prange(n):
            pminx = xs[i] - halfx
            pmaxx = xs[i] + halfx
            pminy = ys[i] - halfy
            pmaxy = ys[i] + halfy
            best = np.inf
            for b in range(counts[frame]):
                dx = boxes[frame, b, 0] - pmaxx
                other = pminx - boxes[frame, b, 2]
                if other > dx:
                    dx = other
                dy = boxes[frame, b, 1] - pmaxy
                other = pminy - boxes[frame, b, 3]
                if other > dy:
                    dy = other
                if dx < 0.0:
                    dx = 0.0
                if dy < 0.0:
                    dy = 0.0
                gap = np.sqrt(dx * dx + dy * dy)
                if gap < best:
                    best = gap
            if best > margin_cap:
                best = margin_cap
            out_dead[i] = best <= 0.0
            out_margin[i] = 0.0 if out_dead[i] else best

    def warmup() -> None:
        """Compile every kernel once so the first measured call is JIT-free.

        Compilation is not part of the reported build time (see EXP-P1-010 §2),
        so the benchmark calls this before timing.
        """
        out = np.zeros((1, 4, 4), dtype=np.uint16)
        boxes = np.zeros((1, 1, 4), dtype=np.float64)
        counts = np.zeros(1, dtype=np.int64)
        lookup = np.zeros(1, dtype=np.uint16)
        build_table_kernel(boxes, counts, 0.0, 0.0, 2.0, 4, 4, 0.5, 0.5, 2, 1, lookup, out)
        xs = np.zeros(1, dtype=np.float64)
        ys = np.zeros(1, dtype=np.float64)
        margin = np.zeros(1, dtype=np.float64)
        dead = np.zeros(1, dtype=np.bool_)
        band = np.zeros(1, dtype=np.uint8)
        query_kernel(
            out.ravel(), 0, 4, 4, 0.0, 0.0, 2.0, boxes, counts, 0.5, 0.5, 16.0,
            xs, ys, margin, dead, band,
        )
        exact_kernel(boxes, counts, 0, 0.5, 0.5, 16.0, xs, ys, margin, dead)

else:  # pragma: no cover - exercised only on interpreters without numba
    build_table_kernel = None
    query_kernel = None
    exact_kernel = None

    def warmup() -> None:
        return None


def probe() -> bool:
    """Run one tiny kernel call; disable the backend if compilation fails."""
    if not _NUMBA:
        return False
    try:
        warmup()
    except Exception as exc:  # pragma: no cover - compiler/runtime failure
        _fallback(f"numba kernel failed: {exc.__class__.__name__}: {exc}")
        return False
    return True
