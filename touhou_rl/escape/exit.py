"""Coarse exit analysis on top of the time-indexed forecast (D4b).

Why a separate module
---------------------
The D4 planner ranked routes by "clearance at the end of the plan".  That is a
local proxy: a wide pocket that fills ten frames later scores better than a
narrow corridor that stays open, which is exactly the failure SC-15 is meant to
catch.  This module adds the missing quantity — **how long does the region at the
end of this route stay open** — computed on the coarse grid, cheaply enough to be
run for every candidate route.

What it does *not* do
---------------------
It does not decide collisions.  The grid is 16 px, so a 4 px gap is invisible
here.  The coarse result only ranks and vetoes routes; the hard predicate stays
`geometry.frame_min_clearance` on the exact per-frame path.  This separation is
the whole reason the coarse layer can be cheap.

The closure estimate is a *cell* property: the route's end is mapped to a grid
cell, and the estimate is the first future time node at which that cell is
occupied.  "Never" means "no closure inside the forecast horizon".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import spec
from .forecast import DangerField

#: Continuation probes are separated by this many frames (coarser than the
#: forecast nodes: we only need the order of magnitude of "when does it close").
CLOSURE_STEP = 6


@dataclass(frozen=True)
class ExitReport:
    cell: tuple[int, int]
    component_size: int
    free_area_px2: float
    closure_eta: int | None
    horizon_frames: int
    #: Distance (px) to the nearest cell that stays free for the whole horizon.
    #: ``None`` when no cell does.  This is the "progress to a free region" term
    #: the D7f ranking needs: at long horizons no route reaches the free side, so
    #: comparing closure of the *end cell* picks a dead end that closes later.
    free_distance_px: float | None = None
    #: Latest closure still reachable from the route end (time-expanded BFS).
    future_freedom: int | None = None

    @property
    def closes_inside_horizon(self) -> bool:
        return self.closure_eta is not None and self.closure_eta <= self.horizon_frames


def cell_of(field: DangerField, x: float, y: float) -> tuple[int, int]:
    cell = field.config.cell_px
    rows, cols = field.occupancy(1).shape
    col = int(np.clip(np.floor((x - spec.PLAYFIELD_X) / cell), 0, cols - 1))
    row = int(np.clip(np.floor((y - spec.PLAYFIELD_Y) / cell), 0, rows - 1))
    return row, col


def playable_mask(field: DangerField) -> np.ndarray:
    """Cells whose centre lies inside the movement rectangle.

    Without this mask the coarse layer routes around the ends of a wall that
    spans the movement area but not the whole playfield — the first version of
    the adaptive mode did exactly that and declared a narrow gap "reachable".
    """

    from . import geometry

    rows, cols = field.occupancy(1).shape
    cell = field.config.cell_px
    xs = spec.PLAYFIELD_X + (np.arange(cols) + 0.5) * cell
    ys = spec.PLAYFIELD_Y + (np.arange(rows) + 0.5) * cell
    model = geometry.MoveModel()
    inside_x = (xs >= model.x_min) & (xs <= model.x_max)
    inside_y = (ys >= model.y_min) & (ys <= model.y_max)
    return inside_y[:, None] & inside_x[None, :]


def _components(grid: np.ndarray, playable: np.ndarray | None = None) -> tuple[np.ndarray, dict[int, int]]:
    """4-connected components of free (``grid == 0``) cells inside the playable area."""

    rows, cols = grid.shape
    labels = np.full((rows, cols), -1, dtype=np.int64)
    sizes: dict[int, int] = {}
    current = 0
    blocked = grid != 0
    if playable is not None:
        blocked = blocked | ~playable
    for row in range(rows):
        for col in range(cols):
            if blocked[row, col] or labels[row, col] != -1:
                continue
            stack = [(row, col)]
            labels[row, col] = current
            size = 0
            while stack:
                r, c = stack.pop()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not blocked[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = current
                        stack.append((nr, nc))
            sizes[current] = size
            current += 1
    return labels, sizes


def future_freedom(field: DangerField, x: float, y: float, t_rel: int) -> int | None:
    """Latest closure reachable from ``(x, y)`` starting at ``t_rel``.

    Now a lookup into the backward DP (``future_freedom_map``); the older
    per-call BFS cost 18 searches per plan and dominated the planner's latency.
    """

    return future_freedom_at(field, x, y, t_rel)


def _legacy_future_freedom_bfs(field: DangerField, x: float, y: float, t_rel: int) -> int | None:
    """Kept for reference/tests: the original per-call BFS implementation."""

    row, col = cell_of(field, x, y)
    table = closure_map(field)
    reachable = coarse_reachable(field, (x, y), t_offset=int(t_rel))
    if not reachable.any():
        return None
    return int(table[reachable].max())


def future_freedom_map(field: DangerField) -> tuple[np.ndarray, tuple[int, ...]]:
    """Backward DP over the time-expanded graph: best closure reachable from a cell.

    ``future_freedom`` used to run one breadth-first search per candidate route
    (18 per plan).  This computes the same idea for *every* cell and node in one
    pass, so scoring a route is an O(1) lookup:

        best[t][c] = max(closure[c], max over 8-neighbours of best[t+1][n])

    The step between nodes is the forecast band step (6 frames = up to 24 px,
    i.e. one 16 px cell), so one cell per node is the right move size.
    """

    cached = getattr(field, "_future_freedom_map", None)
    if cached is not None:
        return cached
    nodes = field.config.time_nodes()
    closure = closure_map(field)
    rows, cols = closure.shape
    horizon = field.config.horizon_frames
    best = np.full((len(nodes), rows, cols), -1, dtype=np.int32)
    for index in range(len(nodes) - 1, -1, -1):
        node = nodes[index]
        occupied = field.occupancy(node) != 0
        here = np.where(occupied, -1, closure)
        if index == len(nodes) - 1:
            best[index] = np.where(occupied, -1, np.where(closure > horizon, horizon + 1, closure))
            continue
        nxt = best[index + 1]
        padded = np.pad(nxt, 1, mode="constant", constant_values=-1)
        neighbours = np.maximum.reduce(
            [
                padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
            ]
        )
        best[index] = np.where(occupied, -1, np.maximum(closure, neighbours))
    field._future_freedom_map = (best, nodes)
    return best, nodes


def future_freedom_at(field: DangerField, x: float, y: float, t_rel: int) -> int | None:
    """O(1) lookup into :func:`future_freedom_map` (max closure reachable from a cell)."""

    best, nodes = future_freedom_map(field)
    row, col = cell_of(field, x, y)
    # First node at or after ``t_rel``.
    index = 0
    for position, node in enumerate(nodes):
        if node >= t_rel:
            index = position
            break
    else:
        index = len(nodes) - 1
    value = int(best[index][row, col])
    return None if value < 0 else value


def free_distance_map(field: DangerField) -> np.ndarray | None:
    """Distance in cell units from every cell to the nearest never-closing cell.

    Breadth-first search without obstacle constraints: the number is a
    "how far is safety from here" proxy for ranking, not a path length.  Returns
    ``None`` when no cell stays free (a fully closing field), in which case the
    ranking falls back to closure time.
    """

    cached = getattr(field, "_free_distance", None)
    if cached is not None:
        return cached
    table = closure_map(field)
    free = table > field.config.horizon_frames
    if not free.any():
        field._free_distance = None
        return None
    rows, cols = free.shape
    big = np.int32(rows + cols + 8)
    distance = np.where(free, np.int32(0), big).astype(np.int32)
    for _ in range(rows + cols + 8):
        padded = np.pad(distance, 1, mode="edge")
        neighbours = np.minimum.reduce(
            [
                padded[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
            ]
        )
        updated = np.where(free, np.int32(0), np.minimum(distance, neighbours + 1)).astype(np.int32)
        if np.array_equal(updated, distance):
            break
        distance = updated
    field._free_distance = distance
    return distance


def components_at(field: DangerField, t_rel: int) -> tuple[np.ndarray, dict[int, int]]:
    """Cached ``(labels, sizes)`` of the free components at one time node.

    ``analyze`` ran a full flood fill for every candidate route (18 per plan); the
    label field only depends on the node, so it is computed once and looked up.
    """

    cache = getattr(field, "_component_cache", None)
    if cache is None:
        cache = {}
        field._component_cache = cache
    t_rel = int(max(0, min(t_rel, field.config.horizon_frames)))
    cached = cache.get(t_rel)
    if cached is None:
        cached = _components(field.occupancy(t_rel), playable_mask(field))
        cache[t_rel] = cached
    return cached


def analyze(field: DangerField, x: float, y: float, t_rel: int) -> ExitReport:
    """Size of the free region around ``(x, y)`` and when that cell closes."""

    t_rel = int(max(0, min(t_rel, field.config.horizon_frames)))
    labels, sizes = components_at(field, t_rel)
    row, col = cell_of(field, x, y)
    label = labels[row, col]
    component = int(sizes.get(int(label), 0)) if label >= 0 else 0
    cell_area = field.config.cell_px ** 2
    closure: int | None = None
    probe = t_rel + CLOSURE_STEP
    while probe <= field.config.horizon_frames:
        if field.occupancy(probe)[row, col] != 0:
            closure = probe
            break
        probe += CLOSURE_STEP
    free_map = free_distance_map(field)
    free_distance = None
    if free_map is not None:
        free_distance = float(free_map[row, col]) * field.config.cell_px
    return ExitReport(
        cell=(int(row), int(col)),
        component_size=component,
        free_area_px2=round(component * cell_area, 2),
        closure_eta=closure,
        horizon_frames=field.config.horizon_frames,
        free_distance_px=free_distance,
        future_freedom=future_freedom(field, x, y, t_rel),
    )


def rank_key(report: ExitReport) -> tuple:
    """Future freedom first, then how long the end cell stays open.

    The D4 order was "closure of the end cell" first.  At a 40-frame horizon no
    route reaches the free side of SC-15, so every end cell closes and the dead
    end (closes at 147) beat the safe approach (closes at ~120).  Ranking by
    "latest closure still reachable from the end" fixes that: the pocket's
    reachable set is the pocket itself (max 154), while the safe approach can
    still reach the never-closing side (181).
    """

    if report.future_freedom is None:
        future = float("-inf")
    else:
        future = float(report.future_freedom)
    closure = float("inf") if report.closure_eta is None else float(report.closure_eta)
    return (future, closure, report.component_size)


def closure_map(field: DangerField) -> np.ndarray:
    """Per-cell closure time: the first time node at which the cell is occupied.

    Cells free through the whole forecast get ``horizon + 1``.  Computed once per
    plan from the forecast's (cached) occupancy grids, so scoring a beam node is
    an O(1) lookup instead of a per-node flood fill.
    """

    nodes = field.config.time_nodes()
    rows, cols = field.occupancy(nodes[0]).shape
    closure = np.full((rows, cols), field.config.horizon_frames + 1, dtype=np.int64)
    closed = np.zeros((rows, cols), dtype=bool)
    for node in nodes:
        occupied = field.occupancy(node) != 0
        first = occupied & ~closed
        closure[first] = node
        closed |= occupied
    return closure


def cell_freedom(field: DangerField, x: float, y: float, table: np.ndarray) -> int:
    row, col = cell_of(field, x, y)
    return int(table[row, col])


def coarse_reachable(
    field: DangerField,
    start: tuple[float, float],
    *,
    t_offset: int = 0,
    step: int = 6,
) -> np.ndarray:
    """Time-expanded reachability on the coarse grid (8-connected, ``step`` frames).

    Used to decide whether the latest-closing region is reachable at all at this
    resolution.  A narrow gap that the grid cannot represent makes the region
    unreachable, which is the signal to fall back to the clearance-first scorer.
    """

    rows, cols = field.occupancy(t_offset + 1).shape
    playable = playable_mask(field)
    reachable = np.zeros((rows, cols), dtype=bool)
    row0, col0 = cell_of(field, *start)
    if field.occupancy(t_offset)[row0, col0] != 0 or not playable[row0, col0]:
        return reachable
    frontier = {(row0, col0)}
    reachable[row0, col0] = True
    t = t_offset
    while frontier and t + step <= field.config.horizon_frames:
        t += step
        grid = field.occupancy(t)
        expanded: set[tuple[int, int]] = set()
        for row, col in frontier:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if not playable[nr, nc] or grid[nr, nc] != 0:
                        continue
                    expanded.add((nr, nc))
                    reachable[nr, nc] = True
        frontier = expanded
    return reachable


def adaptive_mode(
    field: DangerField,
    start: tuple[float, float],
    freedom: np.ndarray,
    *,
    t_offset: int = 0,
    slack_frames: int = 60,
) -> tuple[str, dict]:
    """Pick ``freedom`` when a coarsely reachable region stays open much longer.

    The comparison is against the *current* cell, not the global maximum: a
    global "is the best cell reachable" test sat exactly on the classification
    boundary for SC-16 (150 vs 181 − 30) and flipped on one frame.  Asking "is
    there a reachable region that stays open at least ``slack_frames`` longer
    than where I am" is stable and has a direct reading (one second at 60 Hz).

    Returns ``(mode, diagnostics)`` so the choice can be recorded and audited.
    """

    reachable = coarse_reachable(field, start, t_offset=t_offset)
    row, col = cell_of(field, *start)
    current = int(freedom[row, col])
    reachable_max = int(freedom[reachable].max()) if reachable.any() else -1
    mode = "freedom" if reachable_max >= current + slack_frames else "clearance"
    return mode, {
        "current_cell_closure": current,
        "reachable_closure": reachable_max,
        "slack_frames": slack_frames,
        "criterion": (
            "freedom if a coarsely reachable cell stays open at least slack_frames "
            "longer than the current cell"
        ),
    }
