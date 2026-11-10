"""Geometric beam search over one :class:`HazardTape` (11-P1 搂2 step 2, 搂3).

The model of the ship is exact and comes from the engine, not from constants:

* four speeds read out of ``PLAYER`` (per-shot ``.sht`` data, 11-P1 搂2.2);
* a diagonal step moves by the diagonal speed on **both** axes;
* the playfield clamp is ``x鈭圼8,376]``, ``y鈭圼16,432]``;
* the harness has a **one frame input latency**: the mask written in frame ``t``
  first moves the ship between ``t`` and ``t+1``, so the position of frame ``t``
  is the result of the masks up to ``t-1``.  A decision holds one mask for
  :data:`~touhou_rl.v400.teacher.config.DECISION_FRAMES` frames, therefore it
  moves the ship twice.

Scoring is 11-P1 搂3 verbatim; the death branch is ``death_frame / H * 10``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from touhou_rl.v400.teacher.config import MARGIN_CAP, N_CAND, TeacherConfig
from touhou_rl.v400.teacher.motion import SpeedTable, step_delta


@dataclass(frozen=True)
class Plan:
    """One planned continuation: the first action, how long it is held, the rest."""

    first_action: int
    duration: int
    slots: tuple[int, ...]
    margin: float
    death_frame: int | None
    end_xy: tuple[float, float]
    score: float
    verified: bool = False
    items: float = 0.0
    align: float = 0.0
    edge: float = 0.0
    high: float = 0.0
    #: EXP-P1-015 §2: the bottom-band penalty (>= 0) and the upward-escape reward
    #: that produced this score.  Both are 0.0 for the v1 weights.
    bottom: float = 0.0
    escape: float = 0.0
    #: EXP-P1-017 §2 (v2e): the Boss-body proxemity cost (>= 0, 0.0 = off).
    boss: float = 0.0
    #: EXP-P1-020 §2 (v2f): the continuous top-band cost and the flat top-corner
    #: surcharge (both >= 0, 0.0 = off, i.e. v1/v2a).
    top: float = 0.0
    corner: float = 0.0
    #: EXP-P1-021 (v2h): the laser cost, charged when the terminal position sits
    #: inside — or within ``laser_band`` px of — a laser the engine itself marked
    #: ``damageable``.  0.0 for every earlier profile.
    laser: float = 0.0
    geo_score: float | None = None

    @property
    def masks(self) -> tuple[int, ...]:
        from touhou_rl.v400.gym.actions import cand_to_mask

        out: list[int] = []
        for cand in self.slots:
            out.extend([cand_to_mask(cand, 0)] * 2)
        return tuple(out)

    def as_dict(self) -> dict:
        return {
            "first_action": int(self.first_action),
            "duration": int(self.duration),
            "slots": [int(x) for x in self.slots],
            "margin": round(float(self.margin), 4),
            "death_frame": None if self.death_frame is None else int(self.death_frame),
            "end_xy": [round(float(self.end_xy[0]), 3), round(float(self.end_xy[1]), 3)],
            "score": round(float(self.score), 4),
            "verified": bool(self.verified),
            "align": round(float(self.align), 4),
            "edge": float(self.edge),
            "high": float(self.high),
            "items": float(self.items),
            "bottom": round(float(self.bottom), 4),
            "escape": round(float(self.escape), 4),
            "boss": round(float(self.boss), 4),
            "top": round(float(self.top), 4),
            "corner": round(float(self.corner), 4),
            "laser": round(float(self.laser), 4),
            "geo_score": None if self.geo_score is None else round(float(self.geo_score), 4),
        }


@dataclass
class SearchResult:
    """``geo_score[18]`` plus the best continuation found for each candidate."""

    scores: np.ndarray
    plans: dict[int, Plan]
    target_x: float
    slots: int
    beam: int
    durations: tuple[int, ...]
    detail: dict = field(default_factory=dict)

    def best(self) -> Plan | None:
        living = [p for p in self.plans.values() if p.death_frame is None]
        if living:
            return max(living, key=lambda p: p.score)
        dead = list(self.plans.values())
        if not dead:
            return None
        return max(dead, key=lambda p: p.score)

    def ranking(self) -> list[tuple[int, float]]:
        return sorted(
            ((int(cand), float(self.scores[cand])) for cand in range(N_CAND)),
            key=lambda pair: pair[1],
            reverse=True,
        )


def delta_table(speeds: SpeedTable) -> np.ndarray:
    """``(18, 2)`` per-frame displacement for every candidate action."""
    return np.asarray([step_delta(cand, speeds) for cand in range(N_CAND)], dtype=np.float64)


def target_x(tape) -> float:
    """Target x for ``align``: the nearest colliding enemy body, else the ship.

    11-P1 搂3 says "Boss 鎴栨渶杩戞晫浜虹殑 worldPosition.x".  ``boss_slot`` is not
    exported, so this uses the nearest enemy that the engine itself would judge
    as lethal (``flags1`` bit 2) at the tape's first frame, and falls back to the
    ship's own x when no enemy qualifies (which makes ``align`` neutral).
    """
    from touhou_rl.v400.teacher.geometry import ENEMY_FLAG_COLLISION

    start = np.array(tape.start_position)
    best = None
    best_d = float("inf")
    for enemy in tape.enemies[0]:
        if not (int(enemy.flags1) & ENEMY_FLAG_COLLISION):
            continue
        world = enemy.world_position or enemy.position
        dx = float(world[0]) - start[0]
        dy = float(world[1]) - start[1]
        distance = dx * dx + dy * dy
        if distance < best_d:
            best_d = distance
            best = float(world[0])
    return float(start[0]) if best is None else best


def score_plan(
    *,
    margin: float,
    death_frame: int | None,
    end_xy: tuple[float, float],
    target: float,
    horizon: int,
    config: TeacherConfig,
    escape: float | None = None,
    boss_dist: float | None = None,
    laser_gap: float | None = None,
) -> tuple[float, dict[str, float]]:
    """11-P1 搂3, one term at a time.

    ``min(margin, 16)`` is the margin term, ``align`` is
    ``1 - |end_x - target| / align_range`` clipped to ``[0, 1]``, ``items`` is
    capped at 1, ``edge`` is 1 within 24 px of a vertical border and ``high`` is
    1 when the final ``y < 200``.  A dead plan scores ``death_frame / H * 10``,
    which is below 100 for every in-horizon death.

    ``boss_dist`` (EXP-P1-017 §2) is the AABB gap from the terminal position to
    the nearest live Boss body; ``None``/``inf`` (no Boss on the terminal frame,
    or the v1 weights) adds nothing.  ``top``/``corner`` (EXP-P1-020 §2) are the
    terminal ``y``/``x`` of the same candidate; with the v2f weights they charge
    the continuous top band and the flat top-corner surcharge, and with every
    earlier profile (``top_penalty == corner_penalty == 0``) they are exactly 0.
    ``laser_gap`` (EXP-P1-021) is the pixel gap from the terminal position to the
    nearest laser whose engine-side ``damageable`` flag is 1; ``laser_penalty``
    grades it and is 0.0 for every profile before v2h.
    """
    weights = config.weights
    if death_frame is not None:
        death = weights.death_scale * (death_frame / float(horizon))
        return death, {
            "margin": 0.0,
            "align": 0.0,
            "items": 0.0,
            "edge": 0.0,
            "high": 0.0,
            "bottom": 0.0,
            "escape": 0.0,
            "boss": 0.0,
            "top": 0.0,
            "corner": 0.0,
            "laser": 0.0,
            "death": death,
        }
    margin_term = min(float(margin), weights.margin_cap)
    align = 1.0 - abs(float(end_xy[0]) - float(target)) / max(weights.align_range, 1e-9)
    align = float(min(max(align, 0.0), 1.0))
    items = 0.0  # see TeacherConfig.record_items: 0 for every plan when off
    edge = 1.0 if min(
        float(end_xy[0]) - config.border_x[0], config.border_x[1] - float(end_xy[0])
    ) < weights.edge_band else 0.0
    high = 1.0 if float(end_xy[1]) < weights.high_y else 0.0
    bottom = weights.bottom_penalty_at(float(end_xy[1]))
    boss = weights.boss_penalty_at(boss_dist)
    top = weights.top_penalty_at(float(end_xy[1]))
    corner = weights.corner_penalty_at(float(end_xy[0]), float(end_xy[1]))
    laser = weights.laser_penalty_at(laser_gap)
    escape_points = 0.0
    if (
        weights.escape_weight > 0.0
        and escape is not None
        and float(end_xy[1]) > weights.bottom_y
    ):
        span = max(float(weights.escape_span), 1e-9)
        escape_points = weights.escape_weight * min(max(float(escape), 0.0), span) / span
    total = (
        weights.base
        + margin_term
        + weights.align_weight * align
        + weights.items_weight * min(items, weights.items_cap)
        - weights.edge_penalty * edge
        - weights.high_penalty * high
        - bottom
        - boss
        - top
        - corner
        - laser
        + escape_points
    )
    return float(total), {
        "margin": margin_term,
        "align": align,
        "items": items,
        "edge": edge,
        "high": high,
        "bottom": bottom,
        "escape": escape_points,
        "boss": boss,
        "top": top,
        "corner": corner,
        "laser": laser,
        "death": 0.0,
    }


def escape_clearance(boxes: np.ndarray, end_xy: tuple[float, float], half) -> float:
    """Upward room, in pixels, from ``end_xy`` to the nearest lethal AABB above.

    A box only blocks the upward exit when its x-interval overlaps the ship's own
    column, so a bullet off to the side does not count.  ``inf`` means nothing is
    above the ship in this frame (a fully open upward exit).  This is the
    terminal-frame observable EXP-P1-015 §2 asks for; :func:`_escape_rows` is its
    vectorised form and evaluates the same expression.
    """

    if boxes.shape[0] == 0:
        return float("inf")
    x = float(end_xy[0])
    y = float(end_xy[1])
    columns = (boxes[:, 2] >= x - half[0]) & (boxes[:, 0] <= x + half[0])
    above = boxes[columns & (boxes[:, 3] < y - half[1])]
    if above.shape[0] == 0:
        return float("inf")
    return float(np.min(y - half[1] - above[:, 3]))


def _escape_rows(boxes: np.ndarray, centers: np.ndarray, half, chunk: int = 4096) -> np.ndarray:
    """``escape_clearance`` for many terminal positions, in bounded chunks."""

    rows = centers.shape[0]
    if boxes.shape[0] == 0:
        return np.full((rows,), np.inf, dtype=np.float64)
    min_x, max_x, max_y = boxes[:, 0], boxes[:, 2], boxes[:, 3]
    out = np.empty((rows,), dtype=np.float64)
    for begin in range(0, rows, chunk):
        end = min(begin + chunk, rows)
        block = centers[begin:end]
        overlaps = (max_x[None, :] >= block[:, 0, None] - half[0]) & (
            min_x[None, :] <= block[:, 0, None] + half[0]
        )
        above = max_y[None, :] < block[:, 1, None] - half[1]
        gap = block[:, 1, None] - half[1] - max_y[None, :]
        out[begin:end] = np.where(overlaps & above, gap, np.inf).min(axis=1)
    return out


def boss_body_boxes(enemies) -> np.ndarray:
    """``(n, 4)`` AABBs of the live **Boss** bodies on one frame.

    EXP-P1-017 §2 (v2e).  A Boss body is a lethal AABB like any other enemy body,
    so the shipped :func:`touhou_rl.v400.teacher.geometry.enemy_boxes` expression
    is reused verbatim on the rows whose ``flags1`` carries
    ``ENEMY_FLAG_BOSS`` (``engine.py:46``, bit 1) instead of re-deriving the
    ``hitboxDimensions / 3`` half extent here.  Returns an empty ``(0, 4)`` array
    when no Boss is present, so the caller's penalty is exactly 0.
    """

    from touhou_rl.v400.teacher.engine import ENEMY_FLAG_BOSS
    from touhou_rl.v400.teacher.geometry import enemy_boxes

    bosses = [row for row in enemies if int(getattr(row, "flags1", 0)) & ENEMY_FLAG_BOSS]
    if not bosses:
        return np.zeros((0, 4), dtype=np.float64)
    boxes = enemy_boxes(bosses)
    if boxes.mins.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return np.concatenate([boxes.mins, boxes.maxs], axis=1).astype(np.float64)


def boss_clearance(boxes: np.ndarray, end_xy: tuple[float, float]) -> float:
    """AABB gap, px, from ``end_xy`` to the nearest Boss body (``inf`` if none).

    ``0.0`` means the ship's centre is inside a body.  This is the scalar form of
    :func:`_boss_rows`, and the two evaluate the same expression (the tests pin
    that equality) so the reference search and the vectorised search agree.
    """

    if boxes is None or boxes.shape[0] == 0:
        return float("inf")
    x = float(end_xy[0])
    y = float(end_xy[1])
    dx = np.maximum(boxes[:, 0] - x, x - boxes[:, 2])
    dy = np.maximum(boxes[:, 1] - y, y - boxes[:, 3])
    np.maximum(dx, 0.0, out=dx)
    np.maximum(dy, 0.0, out=dy)
    return float(np.sqrt(dx * dx + dy * dy).min())


def _boss_rows(boxes: np.ndarray, centers: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """:func:`boss_clearance` for many terminal centres, in bounded chunks."""

    rows = centers.shape[0]
    if boxes is None or boxes.shape[0] == 0:
        return np.full((rows,), np.inf, dtype=np.float64)
    min_x, min_y, max_x, max_y = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    out = np.empty((rows,), dtype=np.float64)
    for begin in range(0, rows, chunk):
        end = min(begin + chunk, rows)
        center = centers[begin:end]
        dx = np.maximum(min_x[None, :] - center[:, 0, None], center[:, 0, None] - max_x[None, :])
        dy = np.maximum(min_y[None, :] - center[:, 1, None], center[:, 1, None] - max_y[None, :])
        np.maximum(dx, 0.0, out=dx)
        np.maximum(dy, 0.0, out=dy)
        out[begin:end] = np.sqrt(dx * dx + dy * dy).min(axis=1)
    return out


def _laser_rows_gap(
    lasers, centers: np.ndarray, half, chunk: int = 1024
) -> np.ndarray:
    """Gap, px, from each terminal centre to the nearest **damageable** laser.

    EXP-P1-021.  The per-laser arithmetic is
    :func:`touhou_rl.v400.teacher.geometry.laser_gap_rows`, so this vectorised
    path and the scalar :func:`laser_clearance` the reference search uses cannot
    disagree; the chunking only bounds the size of the temporaries.

    ``inf`` for a frame whose lasers are all outside their judgement window
    (``damageable == 0``), which is exactly the case the old, window-less row
    could not distinguish from a live beam.
    """

    batch = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    rows = batch.shape[0]
    if rows == 0 or not lasers:
        return np.full((rows,), np.inf, dtype=np.float64)
    from touhou_rl.v400.teacher.geometry import laser_gap_rows

    out = np.empty((rows,), dtype=np.float64)
    for begin in range(0, rows, chunk):
        end = min(begin + chunk, rows)
        out[begin:end] = laser_gap_rows(lasers, batch[begin:end], half)
    return out


def laser_clearance(lasers, end_xy: tuple[float, float], half) -> float:
    """Scalar form of :func:`_laser_rows_gap` (``inf`` when no beam is live)."""

    point = np.asarray([[float(end_xy[0]), float(end_xy[1])]], dtype=np.float64)
    return float(_laser_rows_gap(lasers, point, half)[0])


def _margin_rows(boxes: np.ndarray, centers: np.ndarray, half: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """``_frame_margin`` for many centre rows, in bounded-size chunks.

    Element for element the same expression as :func:`_frame_margin` (same
    subtraction order, same ``sqrt``, same cap); chunking only bounds the size
    of the ``(rows, boxes)`` temporaries, it cannot change a value.
    """
    rows = centers.shape[0]
    if boxes.shape[0] == 0:
        return np.full((rows,), MARGIN_CAP, dtype=np.float64)
    out = np.empty((rows,), dtype=np.float64)
    min_x, min_y = boxes[:, 0], boxes[:, 1]
    max_x, max_y = boxes[:, 2], boxes[:, 3]
    for begin in range(0, rows, chunk):
        end = min(begin + chunk, rows)
        center = centers[begin:end]
        pmin_x = center[:, 0] - half[0]
        pmax_x = center[:, 0] + half[0]
        pmin_y = center[:, 1] - half[1]
        pmax_y = center[:, 1] + half[1]
        dx = np.maximum(min_x[None, :] - pmax_x[:, None], pmin_x[:, None] - max_x[None, :])
        dy = np.maximum(min_y[None, :] - pmax_y[:, None], pmin_y[:, None] - max_y[None, :])
        np.maximum(dx, 0.0, out=dx)
        np.maximum(dy, 0.0, out=dy)
        gap = np.sqrt(dx * dx + dy * dy)
        out[begin:end] = np.minimum(gap.min(axis=1), MARGIN_CAP)
    return out


def _frame_margin(tape, frame: int, center: np.ndarray, half: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Capped Euclidean gap from every ``center`` row to the frame's lethals."""
    if boxes.shape[0] == 0:
        return np.full((center.shape[0],), MARGIN_CAP, dtype=np.float64)
    pmin = center - half
    pmax = center + half
    dx = np.maximum(boxes[None, :, 0] - pmax[:, None, 0], pmin[:, None, 0] - boxes[None, :, 2])
    dy = np.maximum(boxes[None, :, 1] - pmax[:, None, 1], pmin[:, None, 1] - boxes[None, :, 3])
    np.maximum(dx, 0.0, out=dx)
    np.maximum(dy, 0.0, out=dy)
    gap = np.sqrt(dx * dx + dy * dy)
    return np.minimum(gap.min(axis=1), MARGIN_CAP)


def simulate_plan(tape, slots, *, config: TeacherConfig, start_center=None, half=None):
    """Walk one plan through the tape; return per-frame margins and the death frame.

    The 1-frame latency is modelled by moving on ``mask[t]`` into frame ``t+1``.
    """
    center = np.array(tape.start_position if start_center is None else start_center, dtype=np.float64)
    half = np.asarray(tape.hurtbox_half if half is None else half, dtype=np.float64)
    deltas = delta_table(tape.speeds)
    margins: list[float] = []
    death: int | None = None
    for slot, cand in enumerate(slots):
        frame0 = 2 * slot
        if frame0 >= len(tape):
            break
        step = deltas[cand]
        positions = np.stack([center, center + step], axis=0)
        positions[:, 0] = np.clip(positions[:, 0], *config.border_x)
        positions[:, 1] = np.clip(positions[:, 1], *config.border_y)
        for offset in (0, 1):
            frame = frame0 + offset
            if frame >= len(tape):
                break
            gap = float(_frame_margin(tape, frame, positions[offset : offset + 1], half, tape.box_array(frame))[0])
            margins.append(gap)
            if gap <= 0.0 and death is None:
                death = frame
        center = center + 2.0 * step
        center[0] = float(np.clip(center[0], *config.border_x))
        center[1] = float(np.clip(center[1], *config.border_y))
    end = center
    margin = float(min(margins)) if margins else float("nan")
    return margin, end, death, margins


def beam_search(
    tape,
    config: TeacherConfig,
    *,
    target: float | None = None,
    use_raster: bool = True,
    finalist_extra: int | None = None,
) -> SearchResult:
    """Vectorised beam search; **bit-identical** to :func:`beam_search_reference`.

    Every root of the search is a ``(first action, duration)`` pair.  The
    reference implementation expands one root at a time, which costs 36 Python
    loops per decision slot; this implementation keeps all roots in one stacked
    array and does one numpy call per slot for the expansion, the AABB margins
    and the per-root selection.  The selection is exact, not approximate:

    * the reference orders the ``beam * 18`` expansions of a root by
      ``(died_now, -margin, action, row)`` and keeps the first ``beam``;
    * here the same total order is written as a numeric key
      ``key = 1000 + (16 - margin)`` for dead nodes and ``key = -margin`` for
      living ones (margins are capped at ``MARGIN_CAP``, so every dead key is
      larger than every living one) plus a tie code ``action * n + row``.  A
      partition finds the ``beam``-th key per root and the tie code — never a
      float comparison with a tolerance — breaks the boundary exactly as the
      reference's stable ``lexsort`` does.

    ``use_raster`` (default) answers the per-frame margin from
    :class:`touhou_rl.v400.teacher.raster.LethalTable` instead of the raw AABB
    list.  The **death verdict stays exact** there (``not hit`` proves alive,
    ``core`` proves dead, and the one-cell band between them is recomputed with
    the original expression); only the margin is quantised to the table's
    lattice.

    ``finalist_extra`` (EXP-P1-010 §3) adds the **finalist re-scoring stage**:
    after the raster table has ranked every expansion, the ``keep_width +
    finalist_extra`` best nodes of each root — the only ones that can still reach
    the top ``keep_width`` — are re-scored with the exact AABB expression, and
    the selection runs on those exact numbers.  ``finalist_extra=0`` restores
    the EXP-P1-008 behaviour (raster ranking only).

    ``use_raster=False`` is the pre-raster implementation, kept callable so the
    two can be measured and compared on the same tape.

    ``PYTHONPATH=... pytest tests/test_v400_teacher.py`` pins the equality on a
    real tape and on random tapes; a NaN margin (impossible for the recorded
    engine data, but cheap to guard) falls back to the reference path for that
    slot so the fast path can never invent an order.
    """
    if float(config.beam) < 1:
        raise ValueError("beam must be >= 1")
    half = np.asarray(tape.hurtbox_half, dtype=np.float64)
    deltas = delta_table(tape.speeds)
    slots = config.slots
    limit = min(slots, len(tape) // 2)
    goal = float(target_x(tape) if target is None else target)
    detail: dict[str, int] = {"roots": 0, "expansions": 0}
    if limit <= 0:
        return SearchResult(
            scores=np.full(N_CAND, -np.inf, dtype=np.float64),
            plans={},
            target_x=goal,
            slots=limit,
            beam=config.beam,
            durations=tuple(config.durations),
            detail=detail,
        )
    lo = np.array([config.border_x[0], config.border_y[0]], dtype=np.float64)
    hi = np.array([config.border_x[1], config.border_y[1]], dtype=np.float64)
    box_frames = [tape.box_array(index) for index in range(2 * limit)]
    # The rasterised table is the fast path: death verdicts stay exact (see
    # :mod:`touhou_rl.v400.teacher.raster`), only the *margin* is quantised to
    # its lattice.  An empty tape keeps the closed-form path below, which needs
    # no table at all.
    table = None
    if use_raster and any(int(box.shape[0]) for box in box_frames):
        from touhou_rl.v400.teacher.raster import build_table

        table = build_table(tape, half)
        detail["raster_cells"] = int(table.nx * table.ny)
        detail["raster_cell_px"] = int(table.cell)
        detail["raster_backend"] = table.backend
    if finalist_extra is None:
        # EXP-P1-010 §3 measured the finalist stage on 85 real tapes and it did
        # **not** reduce the plan-level drift (39/1,530 `death_frame` differences
        # with and without it), while costing ~10 ms per decision.  It is
        # therefore implemented, tested and available, but off by default.
        finalist_extra = 0
    if finalist_extra:
        detail["finalist_extra"] = int(finalist_extra)
    durations = tuple(int(d) for d in config.durations if int(d) <= limit)
    if not durations:
        return SearchResult(
            scores=np.full(N_CAND, -np.inf, dtype=np.float64),
            plans={},
            target_x=goal,
            slots=limit,
            beam=config.beam,
            durations=tuple(config.durations),
            detail=detail,
        )
    start = np.array(tape.start_position, dtype=np.float64)
    roots: list[tuple[int, int]] = []
    root_pos: dict[int, np.ndarray] = {}
    root_margin: dict[int, np.ndarray] = {}
    root_death: dict[int, np.ndarray] = {}
    for duration in durations:
        end, margin, death = _root_nodes(start, deltas, duration, box_frames, half, lo, hi, table)
        root_pos[duration] = end
        root_margin[duration] = margin
        root_death[duration] = death
        for first in range(N_CAND):
            roots.append((first, duration))
            detail["roots"] += 1
    # ``roots`` is first-major (the reference's loop order), so the per-first
    # "keep the best duration" tie-break below sees the same order.
    by_duration: dict[int, list[int]] = {duration: [] for duration in durations}
    for index, (_first, duration) in enumerate(roots):
        by_duration[duration].append(index)
    count = np.ones(len(roots), dtype=np.int64)
    capacity = max(int(config.beam), 1)
    pos = np.zeros((len(roots), capacity, 2), dtype=np.float64)
    margin = np.full((len(roots), capacity), np.inf, dtype=np.float64)
    death = np.full((len(roots), capacity), -1, dtype=np.int64)
    actions = np.zeros((len(roots), capacity, limit), dtype=np.int16)
    for duration, indices in by_duration.items():
        idx = np.asarray(indices, dtype=np.int64)
        firsts = np.asarray([roots[i][0] for i in indices], dtype=np.int16)
        pos[idx, 0] = root_pos[duration]
        margin[idx, 0] = root_margin[duration]
        death[idx, 0] = root_death[duration]
        actions[idx, 0, :duration] = firsts[:, None]
    start_slot = np.asarray([duration for _first, duration in roots], dtype=np.int64)
    tie_cache: dict[int, np.ndarray] = {}
    # No frame of the tape holds a lethal AABB: every margin is exactly
    # ``MARGIN_CAP`` (the cap the per-frame query itself applies), ``died_now``
    # is always false and the accumulated death frame stays -1, so the whole
    # margin/selection stack reduces to "keep the first ``beam`` expansions in
    # (action, row) order".  That is the same selection the general path
    # computes, only without materialising 18 * beam * roots anything.
    empty_tape = not any(int(box.shape[0]) for box in box_frames)
    for slot in range(1, limit):
        active = np.flatnonzero(start_slot <= slot)
        if active.size == 0:
            continue
        counts = count[active]
        for block_count in np.unique(counts):
            idx = active[counts == block_count]
            if idx.size == 0:
                continue
            width = int(block_count)
            n = width * N_CAND
            keep_width = min(capacity, n)
            pos_b = pos[idx, :width, :]
            moved = pos_b[:, :, None, :] + deltas[None, None, :, :]
            np.clip(moved, lo, hi, out=moved)
            detail["expansions"] += int(idx.size * width * N_CAND)
            if empty_tape:
                selected = np.broadcast_to(_tie_order(n, tie_cache)[:keep_width], (idx.size, keep_width))
                rows = np.arange(idx.size)
                node_index = selected // N_CAND
                action = selected % N_CAND
                flat_rows = selected + (rows * n)[:, None]
                new_pos = moved.reshape(-1, 2)[flat_rows.ravel()].reshape(idx.size, keep_width, 2)
                new_pos = new_pos + deltas[action]
                np.clip(new_pos, lo, hi, out=new_pos)
                new_margin = np.full((idx.size, keep_width), MARGIN_CAP, dtype=np.float64)
                new_death = np.full((idx.size, keep_width), -1, dtype=np.int64)
                new_actions = np.take_along_axis(
                    actions[idx, :width, :], node_index[:, :, None], axis=1
                )
                new_actions[:, :, slot] = action.astype(np.int16)
                pos[idx, :keep_width] = new_pos
                margin[idx, :keep_width] = new_margin
                death[idx, :keep_width] = new_death
                actions[idx, :keep_width] = new_actions
                count[idx] = keep_width
                continue
            if table is None:
                gap0 = _margin_rows(
                    box_frames[2 * slot], pos_b.reshape(-1, 2), half
                ).reshape(-1, width)
                gap1 = _margin_rows(
                    box_frames[2 * slot + 1], moved.reshape(-1, 2), half
                ).reshape(-1, width, N_CAND)
                gaps = np.minimum(gap0[:, :, None], gap1)
                died_now = gaps <= 0.0
            else:
                margin0, dead0 = table.query(2 * slot, pos_b.reshape(-1, 2))
                margin1, dead1 = table.query(2 * slot + 1, moved.reshape(-1, 2))
                gaps = np.minimum(
                    margin0.reshape(-1, width)[:, :, None],
                    margin1.reshape(-1, width, N_CAND),
                )
                died_now = dead0.reshape(-1, width)[:, :, None] | dead1.reshape(
                    -1, width, N_CAND
                )
                if finalist_extra > 0:
                    # EXP-P1-010 §3: the raster table ranks the beam, but the
                    # nodes that can still reach the top ``keep_width`` are
                    # re-scored with the exact AABB expression before the
                    # selection runs, so the quantised margin cannot decide a
                    # survivor.
                    gaps, died_now = _refine_finalists(
                        table, 2 * slot, pos_b, moved, margin[idx, :width],
                        gaps, died_now, keep_width, finalist_extra, tie_cache, detail,
                    )
            margins = np.minimum(margin[idx, :width, None], gaps)
            # The reference orders by ``died_now`` (this expansion's gap <= 0),
            # *not* by the accumulated death frame: a node that died earlier but
            # survives this expansion sorts as living.  Reproduce that exactly.
            keys = np.where(died_now, (1000.0 + MARGIN_CAP) - margins, -margins).reshape(-1, n)
            if np.isfinite(keys).all():
                keep_view = _exact_selection(keys, n, keep_width, tie_cache)
                selected = _tie_order(n, tie_cache)[keep_view]
            else:
                # Only reachable with a non-finite margin, which the recorded
                # engine numbers cannot produce; fall back to the reference's
                # own lexsort for this slot instead of guessing an order.
                selected = _lexsort_selection(died_now, margins, keep_width)
            rows = np.arange(selected.shape[0])
            node_index = selected // N_CAND
            # ``selected`` is block-local; the gathers below are over the whole
            # stacked batch, so shift each root's block by its own offset.
            flat_rows = selected + (rows * n)[:, None]
            action = selected % N_CAND
            new_pos = moved.reshape(-1, 2)[flat_rows.ravel()].reshape(
                selected.shape[0], keep_width, 2
            ) + deltas[action]
            np.clip(new_pos, lo, hi, out=new_pos)
            new_margin = margins.reshape(-1, n)[rows[:, None], selected]
            previous = death[idx, :width][rows[:, None], node_index]
            died_here = died_now.reshape(-1, n)[rows[:, None], selected]
            new_death = np.where(previous >= 0, previous, np.where(died_here, 2 * slot + 1, -1))
            new_actions = np.take_along_axis(
                actions[idx, :width, :], node_index[:, :, None], axis=1
            )
            new_actions[:, :, slot] = action.astype(np.int16)
            pos[idx, :keep_width] = new_pos
            margin[idx, :keep_width] = new_margin
            death[idx, :keep_width] = new_death
            actions[idx, :keep_width] = new_actions
            count[idx] = keep_width
    # EXP-P1-015 §2: the terminal frame's lethal boxes, the input of the upward
    # escape term ("how far can the ship still go up from where the plan ends?").
    escape_frame = min(2 * limit, len(tape) - 1)
    escape_boxes = tape.box_array(escape_frame)
    # EXP-P1-017 §2: the same terminal frame's Boss bodies (the proxemity cost).
    boss_boxes = boss_body_boxes(tape.enemies[escape_frame])
    # EXP-P1-021: the same terminal frame's lasers.  The rows are passed
    # unfiltered; the damageable gate lives in the geometry helper so the search
    # and ``frame_verdict`` cannot drift apart.
    laser_rows = tape.lasers[escape_frame] if config.weights.laser_penalty > 0.0 else None
    plans = _finish_roots(
        roots, count, pos, margin, death, actions, goal, config,
        half=half, escape_boxes=escape_boxes, boss_boxes=boss_boxes,
        laser_rows=laser_rows,
    )
    if table is not None:
        # How much of the answer had to be recomputed exactly (the one-cell band
        # that straddles a lethal boundary); the rest came from the table.
        detail["raster_band_queries"] = int(table.band_queries)
        detail["raster_band_exact"] = int(table.band_exact)
    scores = np.full(N_CAND, -np.inf, dtype=np.float64)
    for cand, plan in plans.items():
        scores[cand] = plan.score
    return SearchResult(
        scores=scores,
        plans=plans,
        target_x=goal,
        slots=limit,
        beam=config.beam,
        durations=tuple(config.durations),
        detail=detail,
    )


def _root_nodes(start, deltas, duration, box_frames, half, lo, hi, table=None):
    """Every first action held for ``duration`` slots, in one vectorised pass.

    Mirrors :func:`simulate_plan` operation by operation: the position of frame
    ``2j`` is the clamped centre, ``2j+1`` is ``centre + step`` clamped, and the
    next centre is ``centre + 2 * step`` clamped (the reference adds ``2*step``
    once, so the two implementations must not be "simplified" into each other).
    With a raster table the per-frame query is a lookup; without one (empty
    tape) it is the original AABB expression.
    """
    center = np.tile(np.asarray(start, dtype=np.float64), (N_CAND, 1))
    margin = np.full(N_CAND, np.inf, dtype=np.float64)
    death = np.full(N_CAND, -1, dtype=np.int64)
    for j in range(duration):
        first = center + deltas
        np.clip(first, lo, hi, out=first)
        if table is None:
            m0 = _margin_rows(box_frames[2 * j], center, half)
            m1 = _margin_rows(box_frames[2 * j + 1], first, half)
            d0 = m0 <= 0.0
            d1 = m1 <= 0.0
        else:
            # The roots are 18 points per frame, so they are always scored with
            # the exact expression: an approximate root margin would propagate
            # into every continuation of that root (EXP-P1-010 §3).
            m0, d0 = table.exact_points(2 * j, center)
            m1, d1 = table.exact_points(2 * j + 1, first)
        margin = np.minimum(margin, np.minimum(m0, m1))
        death = np.where((death < 0) & d0, 2 * j, death)
        death = np.where((death < 0) & d1, 2 * j + 1, death)
        center = center + 2.0 * deltas
        np.clip(center, lo, hi, out=center)
    return center, margin, death


def _tie_order(n: int, cache: dict[int, np.ndarray]) -> np.ndarray:
    """Flat rows of a block sorted by the reference's ``(action, row)`` tie code."""
    order = cache.get(n)
    if order is None:
        flat = np.arange(n, dtype=np.int64)
        order = np.argsort((flat % N_CAND) * n + flat)
        cache[n] = order
    return order


def _tie_orders(n: int, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_tie_order` plus the inverse permutation (``cache`` may hold both)."""
    entry = cache.get(("both", n))
    if entry is None:
        order = _tie_order(n, cache)
        inverse = np.empty(n, dtype=np.int64)
        inverse[order] = np.arange(n, dtype=np.int64)
        entry = (order, inverse)
        cache[("both", n)] = entry
    return entry


def _refine_finalists(
    table,
    frame0: int,
    pos_b: np.ndarray,
    moved: np.ndarray,
    prev_margin: np.ndarray,
    gaps: np.ndarray,
    died_now: np.ndarray,
    keep_width: int,
    extra: int,
    tie_cache: dict,
    detail: dict,
):
    """Re-score the nodes that can still reach the beam with the exact AABB gap.

    EXP-P1-010 §3.  The raster table ranks every expansion of the slot; the
    quantisation of its margin (4 px lattice, 1/4 px fixed point) is the only
    thing that can reorder them.  Only ``keep_width + extra`` nodes per root can
    still be kept, so those — and only those — are recomputed with the same
    expression the exact search uses, which makes the selection below exact
    whenever the real margin of a dropped node is outside the ``extra`` slack.

    Returns the same ``(gaps, died_now)`` arrays with the finalist entries
    replaced by their exact values; every other entry keeps its table value.
    """
    blocks, width = gaps.shape[0], gaps.shape[1]
    n = width * N_CAND
    if n <= 0:
        return gaps, died_now
    margins = np.minimum(prev_margin[:, :, None], gaps).reshape(-1, n)
    keys = np.where(died_now.reshape(-1, n), (1000.0 + MARGIN_CAP) - margins, -margins)
    k2 = min(n, keep_width + extra)
    # ``argpartition`` (not a full sort, and not the tie-ordered view the exact
    # selection uses): the finalist set only has to *contain* the nodes that can
    # reach the beam, and the exact selection below re-applies the reference's
    # ``(action, row)`` tie code to the re-scored values.
    picked = np.argpartition(keys, k2 - 1, axis=1)[:, :k2]
    blocks_axis = np.repeat(np.arange(gaps.shape[0], dtype=np.int64), k2)
    cols = picked.reshape(-1)
    rows = blocks_axis
    parents = rows * width + cols // N_CAND
    base = rows * n + cols
    unique_parents, parent_of = np.unique(parents, return_inverse=True)
    parent_margin, parent_dead = table.exact_points(
        frame0, pos_b.reshape(-1, 2)[unique_parents]
    )
    child_margin, child_dead = table.exact_points(
        frame0 + 1, moved.reshape(-1, 2)[base]
    )
    exact_gap = np.minimum(parent_margin[parent_of], child_margin)
    exact_dead = parent_dead[parent_of] | child_dead
    flat_gaps = gaps.reshape(-1, n)
    flat_died = died_now.reshape(-1, n)
    flat_gaps[rows, cols] = exact_gap
    flat_died[rows, cols] = exact_dead
    detail["finalist_points"] = int(detail.get("finalist_points", 0)) + int(rows.size)
    detail["finalist_parents"] = int(detail.get("finalist_parents", 0)) + int(
        unique_parents.size
    )
    return flat_gaps.reshape(gaps.shape), flat_died.reshape(died_now.shape)


def _exact_selection(keys, n, keep_width, tie_cache):
    """The ``keep_width`` smallest ``(key, tie)`` rows of every root block.

    ``keys`` is ``(G, n)`` in *flat* row order; the returned array is ``(G, k)``
    of flat row positions, already sorted by ``(key, tie)`` — exactly what the
    reference's ``lexsort((action, -margin, died_now))[:beam]`` produces.
    """
    tie_order = _tie_order(n, tie_cache)
    view = keys[:, tie_order]
    threshold = np.partition(view, keep_width - 1, axis=1)[:, keep_width - 1]
    lower = view < threshold[:, None]
    equal = view == threshold[:, None]
    need = keep_width - lower.sum(axis=1)
    rank = np.cumsum(equal, axis=1)
    selected = lower | (equal & (rank <= need[:, None]))
    rows, where = np.nonzero(selected)
    slot = np.cumsum(selected, axis=1)[rows, where] - 1
    positions = np.empty((selected.shape[0], keep_width), dtype=np.int64)
    positions[rows, slot] = where
    values = np.take_along_axis(view, positions, axis=1)
    block = np.repeat(np.arange(selected.shape[0]), keep_width)
    order = np.lexsort((positions.ravel(), values.ravel(), block))
    return positions.ravel()[order].reshape(selected.shape[0], keep_width)


def _lexsort_selection(died_now, margins, keep_width):
    """The reference's per-block ``lexsort`` selection, used for non-finite keys."""
    width, per = died_now.shape[0], died_now.shape[1] * died_now.shape[2]
    action = np.tile(np.arange(N_CAND), died_now.shape[1])
    out = np.empty((width, keep_width), dtype=np.int64)
    flat_died = died_now.reshape(width, per).astype(np.int8)
    flat_margin = margins.reshape(width, per)
    for row in range(width):
        out[row] = np.lexsort((action, -flat_margin[row], flat_died[row]))[:keep_width]
    return out


def _finish_roots(
    roots, count, pos, margin, death, actions, goal, config, *,
    half=None, escape_boxes=None, boss_boxes=None, laser_rows=None,
):
    """Score every surviving node and keep the best plan per first action.

    ``half``/``escape_boxes`` are the terminal-frame inputs of EXP-P1-015's
    escape term; when the weights keep ``escape_weight == 0`` they are ignored and
    the score is the v1 expression.  ``boss_boxes`` (EXP-P1-017 §2) is the
    terminal frame's Boss-body AABB array: it is subtracted element for element,
    so with ``boss_penalty == 0`` (or no Boss) the score is bit-identical to v1.
    ``laser_rows`` (EXP-P1-021) are the terminal frame's laser rows; only the ones
    the engine marked ``damageable`` become hazards, and with
    ``laser_penalty == 0`` the term is exactly 0 for every plan.
    """
    weights = config.weights
    horizon = float(config.horizon)
    plans: dict[int, Plan] = {}
    bx0, bx1 = config.border_x
    align_range = max(weights.align_range, 1e-9)
    use_escape = bool(
        weights.escape_weight > 0.0 and half is not None and escape_boxes is not None
    )
    use_bottom = bool(weights.bottom_penalty > 0.0 or weights.bottom_extra_penalty > 0.0)
    use_boss = bool(
        weights.boss_penalty > 0.0 and boss_boxes is not None and boss_boxes.shape[0] > 0
    )
    # EXP-P1-021 (v2h): the terminal-frame laser cost.  ``laser_rows`` is the
    # frame's full laser list; ``laser_gap_rows`` drops every row the engine did
    # not mark ``damageable`` (and every pre-EXP-P1-021 row that is not ACTIVE),
    # so a beam that is still fading in is not a hazard.
    use_laser = bool(weights.laser_penalty > 0.0 and half is not None and laser_rows)
    laser_span = max(float(weights.laser_band), 1e-9)
    # EXP-P1-015 §2b (v2c): the hard corridor on the terminal position.
    clip_y = getattr(config, "clip_y", None)
    clip_top = None if clip_y is None else float(clip_y[0])
    clip_bottom = None if clip_y is None else float(clip_y[1])
    for index, (first, duration) in enumerate(roots):
        width = int(count[index])
        end = pos[index, :width]
        margins = margin[index, :width]
        deaths = death[index, :width]
        slots = actions[index, :width]
        alive = deaths < 0
        margin_term = np.minimum(margins, weights.margin_cap)
        align = np.clip(1.0 - np.abs(end[:, 0] - goal) / align_range, 0.0, 1.0)
        edge = (np.minimum(end[:, 0] - bx0, bx1 - end[:, 0]) < weights.edge_band).astype(np.float64)
        high = (end[:, 1] < weights.high_y).astype(np.float64)
        items = 0.0
        # EXP-P1-015 §2, element for element the scalar ``Weights`` helpers.
        bottom = np.zeros(width, dtype=np.float64)
        if weights.bottom_penalty > 0.0:
            bottom = bottom + weights.bottom_penalty * np.maximum(
                end[:, 1] - weights.bottom_y, 0.0
            ) / max(float(weights.bottom_span), 1e-9)
        if weights.bottom_extra_penalty > 0.0:
            bottom = bottom + weights.bottom_extra_penalty * np.maximum(
                end[:, 1] - weights.bottom_extra_y, 0.0
            ) / max(float(weights.bottom_extra_span), 1e-9)
        # EXP-P1-020 §2 (v2f): the continuous top band and the top-corner
        # surcharge, element for element the scalar ``Weights`` helpers.
        top = np.zeros(width, dtype=np.float64)
        if weights.top_penalty > 0.0:
            top = top + weights.top_penalty * np.maximum(
                weights.top_y - end[:, 1], 0.0
            ) / max(float(weights.top_span), 1e-9)
        corner = np.zeros(width, dtype=np.float64)
        if weights.corner_penalty > 0.0:
            in_corner_x = (end[:, 0] < weights.corner_x[0]) | (end[:, 0] > weights.corner_x[1])
            corner = np.where(
                in_corner_x & (end[:, 1] < weights.high_y),
                float(weights.corner_penalty),
                0.0,
            )
        escape_points = np.zeros(width, dtype=np.float64)
        if use_escape and width:
            clearance = _escape_rows(escape_boxes, end, half)
            inside = end[:, 1] > weights.bottom_y
            span = max(float(weights.escape_span), 1e-9)
            escape_points = np.where(
                inside,
                weights.escape_weight * np.clip(clearance, 0.0, span) / span,
                0.0,
            )
        # EXP-P1-017 §2: the Boss-body proxemity cost, element for element the
        # scalar ``Weights.boss_penalty_at`` (clip(radius - d, 0, radius)/radius).
        boss = np.zeros(width, dtype=np.float64)
        if use_boss and width:
            boss_gap = _boss_rows(boss_boxes, end)
            boss_span = max(float(weights.boss_radius), 1e-9)
            boss = weights.boss_penalty * np.clip(boss_span - boss_gap, 0.0, boss_span) / boss_span
        # EXP-P1-021 (v2h): the same shape one family over — the gap to the
        # nearest damageable laser, clipped to ``laser_band`` and graded by
        # ``laser_penalty`` (0 inside the beam's rectangle).
        laser = np.zeros(width, dtype=np.float64)
        laser_gap = np.full(width, np.inf, dtype=np.float64)
        if use_laser and width:
            laser_gap = _laser_rows_gap(laser_rows, end, half)
            laser = weights.laser_penalty * np.clip(
                laser_span - laser_gap, 0.0, laser_span
            ) / laser_span
        total = (
            weights.base
            + margin_term
            + weights.align_weight * align
            + weights.items_weight * min(items, weights.items_cap)
            - weights.edge_penalty * edge
            - weights.high_penalty * high
            - bottom
            - boss
            - top
            - corner
            - laser
            + escape_points
        )
        death_score = weights.death_scale * (deaths / horizon)
        score = np.where(alive, total, death_score)
        # EXP-P1-015 §2b: the hard corridor.  A living plan outside it is pushed
        # under every living plan inside it (and under the death branch, so a
        # surviving out-of-corridor plan still beats dying), but its stored score
        # stays the real one — the corridor is a *selection* rule, not a weight.
        key = score
        if clip_top is not None or clip_bottom is not None:
            inside = np.ones(width, dtype=bool)
            if clip_top is not None:
                inside &= end[:, 1] >= clip_top
            if clip_bottom is not None:
                inside &= end[:, 1] <= clip_bottom
            inside_alive = alive & inside
            if inside_alive.any():
                key = np.where(inside_alive, score, score - 1000.0)
        if not np.isfinite(score).all():
            best_index = -1
            best_key = -np.inf
            for candidate in range(width):
                frame = None if int(deaths[candidate]) < 0 else int(deaths[candidate])
                value, _parts = score_plan(
                    margin=float(margins[candidate]),
                    death_frame=frame,
                    end_xy=(float(end[candidate, 0]), float(end[candidate, 1])),
                    target=goal,
                    horizon=config.horizon,
                    config=config,
                    escape=float(escape_points[candidate]) if use_escape else None,
                    boss_dist=float(boss_gap[candidate]) if use_boss else None,
                    laser_gap=float(laser_gap[candidate]) if use_laser else None,
                )
                candidate_key = float(key[candidate])
                if best_index < 0 or candidate_key > best_key:
                    best_index = candidate
                    best_key = candidate_key
            chosen = best_index
        else:
            chosen = int(np.argmax(key))
        death_frame = None if int(deaths[chosen]) < 0 else int(deaths[chosen])
        # ``score_plan`` reports align/edge/high as 0.0 for a dead plan (the
        # death branch has no geometry terms); mirror that instead of leaking
        # the living terms into the label.
        plan = Plan(
            first_action=first,
            duration=duration,
            slots=tuple(int(x) for x in slots[chosen]),
            margin=float(margins[chosen]),
            death_frame=death_frame,
            end_xy=(float(end[chosen, 0]), float(end[chosen, 1])),
            score=float(score[chosen]),
            align=0.0 if death_frame is not None else float(align[chosen]),
            edge=0.0 if death_frame is not None else float(edge[chosen]),
            high=0.0 if death_frame is not None else float(high[chosen]),
            items=float(items),
            bottom=0.0 if death_frame is not None else float(bottom[chosen]),
            escape=0.0 if death_frame is not None else float(escape_points[chosen]),
            boss=0.0 if death_frame is not None else float(boss[chosen]),
            top=0.0 if death_frame is not None else float(top[chosen]),
            corner=0.0 if death_frame is not None else float(corner[chosen]),
            laser=0.0 if death_frame is not None else float(laser[chosen]),
        )
        existing = plans.get(first)
        if existing is None or plan.score > existing.score:
            plans[first] = plan
    return plans


def beam_search_reference(tape, config: TeacherConfig, *, target: float | None = None) -> SearchResult:
    """Beam search from every ``(first action, duration)`` root.

    This is the **pre-vectorisation implementation**, kept verbatim as the
    equivalence oracle for :func:`beam_search`.  It is never used by a decision.

    For each of the 18 candidates the first action is held for each duration in
    ``config.durations`` decision slots and the remaining slots are searched with
    width ``config.beam``.  ``geo_score[18]`` is the best score of that
    candidate's continuations (11-P1 搂3: the unverified discount is applied by
    the caller once verification has run).
    """
    half = np.asarray(tape.hurtbox_half, dtype=np.float64)
    deltas = delta_table(tape.speeds)
    slots = config.slots
    limit = min(slots, len(tape) // 2)
    goal = float(target_x(tape) if target is None else target)
    # EXP-P1-015 §2: the same terminal frame as :func:`beam_search` (the frame
    # before the plan's end position), so the two paths stay comparable.
    escape_frame = min(2 * limit, len(tape) - 1)
    escape_boxes = tape.box_array(escape_frame)
    # EXP-P1-017 §2 (v2e): the terminal Boss bodies, the proxemity cost's input.
    boss_boxes = boss_body_boxes(tape.enemies[escape_frame])
    # EXP-P1-021 (v2h): the terminal lasers, the laser cost's input.
    laser_rows = tape.lasers[escape_frame] if config.weights.laser_penalty > 0.0 else None
    clip_y = getattr(config, "clip_y", None)
    clip_top = None if clip_y is None else float(clip_y[0])
    clip_bottom = None if clip_y is None else float(clip_y[1])
    plans: dict[int, Plan] = {}
    detail: dict[str, int] = {"roots": 0, "expansions": 0}
    start = np.array(tape.start_position, dtype=np.float64)

    for first in range(N_CAND):
        best: Plan | None = None
        for duration in config.durations:
            if duration > limit:
                continue
            detail["roots"] += 1
            margin, pos, death, _ = simulate_plan(tape, [first] * duration, config=config)
            nodes_pos = np.asarray(pos, dtype=np.float64)[None, :].copy()
            nodes_margin = np.array([margin], dtype=np.float64)
            nodes_death = np.array([-1 if death is None else int(death)], dtype=np.int64)
            nodes_actions = np.zeros((1, limit), dtype=np.int16)
            nodes_actions[0, :duration] = first
            for slot in range(duration, limit):
                count = nodes_pos.shape[0]
                moved = nodes_pos[:, None, :] + deltas[None, :, :]  # frame 2*slot+1
                moved[:, :, 0] = np.clip(moved[:, :, 0], *config.border_x)
                moved[:, :, 1] = np.clip(moved[:, :, 1], *config.border_y)
                gap0 = _frame_margin(tape, 2 * slot, nodes_pos, half, tape.box_array(2 * slot))
                gap1 = _frame_margin(
                    tape, 2 * slot + 1, moved.reshape(-1, 2), half, tape.box_array(2 * slot + 1)
                )
                gaps = np.minimum(np.repeat(gap0, N_CAND), gap1)
                detail["expansions"] += int(gaps.size)
                margins = np.minimum(np.repeat(nodes_margin, N_CAND), gaps)
                died_now = gaps <= 0.0
                previous = np.repeat(nodes_death, N_CAND)
                death_frames = np.where(previous >= 0, previous, np.where(died_now, 2 * slot + 1, -1))
                successor = moved + deltas[None, :, :]  # frame 2*slot+2
                successor[:, :, 0] = np.clip(successor[:, :, 0], *config.border_x)
                successor[:, :, 1] = np.clip(successor[:, :, 1], *config.border_y)
                flat_successor = successor.reshape(-1, 2)
                # Living first, then the largest margin, then a stable action id.
                order = np.lexsort(
                    (
                        np.tile(np.arange(N_CAND), count),
                        -margins,
                        died_now.astype(np.int8),
                    )
                )
                keep = order[: config.beam]
                actions = np.repeat(nodes_actions, N_CAND, axis=0)[keep]
                actions[:, slot] = keep % N_CAND
                nodes_pos = flat_successor[keep]
                nodes_margin = margins[keep]
                nodes_death = death_frames[keep]
                nodes_actions = actions
            candidates: list[tuple[float, Plan]] = []
            for index in range(nodes_pos.shape[0]):
                death_frame = None if int(nodes_death[index]) < 0 else int(nodes_death[index])
                end_xy = (float(nodes_pos[index, 0]), float(nodes_pos[index, 1]))
                score, parts = score_plan(
                    margin=float(nodes_margin[index]),
                    death_frame=death_frame,
                    end_xy=end_xy,
                    target=goal,
                    horizon=config.horizon,
                    config=config,
                    escape=(
                        escape_clearance(escape_boxes, end_xy, half)
                        if config.weights.escape_weight > 0.0
                        else None
                    ),
                    boss_dist=(
                        boss_clearance(boss_boxes, end_xy)
                        if config.weights.boss_penalty > 0.0
                        else None
                    ),
                    laser_gap=(
                        laser_clearance(laser_rows, end_xy, half)
                        if config.weights.laser_penalty > 0.0
                        else None
                    ),
                )
                plan = Plan(
                    first_action=first,
                    duration=duration,
                    slots=tuple(int(x) for x in nodes_actions[index]),
                    margin=float(nodes_margin[index]),
                    death_frame=death_frame,
                    end_xy=end_xy,
                    score=score,
                    align=parts["align"],
                    edge=parts["edge"],
                    high=parts["high"],
                    items=parts["items"],
                    bottom=parts["bottom"],
                    escape=parts["escape"],
                    boss=parts["boss"],
                    top=parts["top"],
                    corner=parts["corner"],
                    laser=parts["laser"],
                )
                candidates.append((score, plan))
            # EXP-P1-015 §2b: the same corridor rule as ``_finish_roots`` — the
            # decision is made per root, over every node of this (first, duration).
            if clip_top is not None or clip_bottom is not None:
                inside_alive = False
                for score, plan in candidates:
                    if plan.death_frame is not None:
                        continue
                    y = plan.end_xy[1]
                    if clip_top is not None and y < clip_top:
                        continue
                    if clip_bottom is not None and y > clip_bottom:
                        continue
                    inside_alive = True
                    break
                if inside_alive:
                    candidates = [
                        (
                            score
                            if plan.death_frame is None
                            and (clip_top is None or plan.end_xy[1] >= clip_top)
                            and (clip_bottom is None or plan.end_xy[1] <= clip_bottom)
                            else score - 1000.0,
                            plan,
                        )
                        for score, plan in candidates
                    ]
            for score, plan in candidates:
                if best is None or score > best[0]:
                    best = (score, plan)
        if best is not None:
            plans[first] = best[1]

    scores = np.full(N_CAND, -np.inf, dtype=np.float64)
    for cand, plan in plans.items():
        scores[cand] = plan.score
    return SearchResult(
        scores=scores,
        plans=plans,
        target_x=goal,
        slots=limit,
        beam=config.beam,
        durations=tuple(config.durations),
        detail=detail,
    )
