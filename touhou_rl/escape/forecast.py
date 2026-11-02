"""Time-indexed danger forecast with explicit uncertainty (D3).

Design rules that come from the earlier stages
----------------------------------------------
1. **Time is kept as an axis.**  A cell that is dangerous at frame 30 and free at
   frame 60 must be representable; the representation is a set of snapshots, not
   a union.  SC-07 exists precisely to catch a union.
2. **The forecast never decides safety.**  It ranks and warns.  The hard
   predicate stays :func:`touhou_rl.escape.geometry.frame_min_clearance` on the
   actually-simulated path (per-frame AABB, D1 revision), because a coarse grid
   can miss a narrow gap (E-05/E-06).
3. **Anything outside the horizon is unknown, not safe.**  Queries past
   ``valid_until`` return ``None`` and a separate ``unknown`` flag.
4. **Cutting entities is explicit.**  A hazard whose swept bounding box never
   touches the movement rectangle can be dropped soundly ("irrelevant"); above
   the entity cap the rest are dropped as "over_cap" and the coverage is marked
   incomplete.  Dropped counts and reasons travel with the forecast.
5. **Uncertainty grows with time** and long-range nodes are uncalibrated, so a
   planner may use them for preference but not for hard refusal (plan §3.2).

The class works on a snapshot in absolute game coordinates: positions at
``frame_base`` plus per-frame velocities.  It does not read memory and does not
know about the game process.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import spec
from .contracts import (
    DangerForecast,
    FORECAST_SCHEMA,
    HURTBOX_HALF_PX,
)
from . import geometry, synth

MODEL_VERSION = "v370-forecast/1"


@dataclass(frozen=True)
class ForecastConfig:
    """Candidate values (dev-set tunable until D7 freezes them)."""

    #: (start, stop, step) bands in game frames from the decision instant.
    bands: tuple[tuple[int, int, int], ...] = ((0, 30, 2), (30, 90, 6), (90, 180, 15))
    horizon_frames: int = 180
    #: Stored macro grid; queries are exact and do not use this resolution.
    cell_px: float = 16.0
    inflation_px: float = 0.0
    #: Beyond this offset the linear model is not calibrated.
    calibrated_frames: int = 90
    uncertainty_base_px: float = 0.0
    uncertainty_per_frame: float = 0.02
    max_entities: int = 1024
    player_half: float = HURTBOX_HALF_PX

    def time_nodes(self) -> tuple[int, ...]:
        nodes: list[int] = []
        for start, stop, step in self.bands:
            if step < 1:
                raise ValueError("band step must be >= 1")
            node = start + step
            while node <= min(stop, self.horizon_frames):
                if not nodes or node > nodes[-1]:
                    nodes.append(int(node))
                node += step
        return tuple(nodes)

    def validate(self) -> None:
        if self.horizon_frames < 1:
            raise ValueError("horizon must be >= 1")
        if self.cell_px <= 0.0:
            raise ValueError("cell_px must be positive")
        if self.inflation_px < 0.0:
            raise ValueError("inflation must be >= 0")
        if self.max_entities < 1:
            raise ValueError("max_entities must be >= 1")


@dataclass
class DangerField:
    """Hazards in absolute coordinates plus a base frame for relative queries."""

    frame_base: int
    positions: np.ndarray  # (N, 2) at frame_base
    velocities: np.ndarray  # (N, 2) per frame
    halves: np.ndarray  # (N, 2)
    spawn: np.ndarray  # (N,) absolute frames
    despawn: np.ndarray  # (N,) absolute frames
    config: ForecastConfig = field(default_factory=ForecastConfig)
    #: Indices dropped as irrelevant / over cap, with reasons.
    dropped_irrelevant: int = 0
    dropped_over_cap: int = 0
    kept: np.ndarray | None = None
    #: Pre-cap relevant entity arrays, kept only so a coverage proof can use
    #: every entity instead of the capped view (see ``coverage_proof``).
    all_positions: np.ndarray | None = None
    all_velocities: np.ndarray | None = None
    all_halves: np.ndarray | None = None
    all_spawn: np.ndarray | None = None
    all_despawn: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64).reshape(-1, 2)
        self.velocities = np.asarray(self.velocities, dtype=np.float64).reshape(-1, 2)
        self.halves = np.asarray(self.halves, dtype=np.float64).reshape(-1, 2)
        self.spawn = np.asarray(self.spawn, dtype=np.int64).reshape(-1)
        self.despawn = np.asarray(self.despawn, dtype=np.int64).reshape(-1)
        count = self.positions.shape[0]
        for name, array in (
            ("velocities", self.velocities),
            ("halves", self.halves),
            ("spawn", self.spawn),
            ("despawn", self.despawn),
        ):
            if array.shape[0] != count:
                raise ValueError(f"{name} has {array.shape[0]} entries, expected {count}")
        self.config.validate()
        if self.kept is None:
            self.kept = np.arange(count, dtype=np.int64)
        if self.all_positions is None:
            self.all_positions = self.positions
            self.all_velocities = self.velocities
            self.all_halves = self.halves
            self.all_spawn = self.spawn
            self.all_despawn = self.despawn
        else:
            self.all_positions = np.asarray(self.all_positions, dtype=np.float64).reshape(-1, 2)
            self.all_velocities = np.asarray(self.all_velocities, dtype=np.float64).reshape(-1, 2)
            self.all_halves = np.asarray(self.all_halves, dtype=np.float64).reshape(-1, 2)
            self.all_spawn = np.asarray(self.all_spawn, dtype=np.int64).reshape(-1)
            self.all_despawn = np.asarray(self.all_despawn, dtype=np.int64).reshape(-1)
        # Cache active centres/halves per time node: the planner calls clearance
        # tens of thousands of times per decision, and rebuilding the mask and
        # fancy-indexed arrays dominated the runtime.
        self._active_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._occupancy_cache: dict[int, np.ndarray] = {}

    def _active_at(self, t_rel: float) -> tuple[np.ndarray, np.ndarray]:
        cached = self._active_cache.get(t_rel)
        if cached is None:
            mask = self._active(self.frame_base + float(t_rel))
            cached = (
                (self.positions + self.velocities * float(t_rel))[mask],
                self.halves[mask],
            )
            self._active_cache[t_rel] = cached
        return cached

    # -- construction ----------------------------------------------------
    @classmethod
    def from_scenario(cls, scenario: synth.Scenario, frame: int, config: ForecastConfig | None = None):
        config = config or ForecastConfig()
        positions = np.asarray([hazard.at(frame) for hazard in scenario.hazards], dtype=np.float64)
        field = cls(
            frame_base=int(frame),
            positions=positions.reshape(-1, 2),
            velocities=scenario.velocities(),
            halves=scenario.halves(),
            spawn=scenario.spawn_frames,
            despawn=scenario.despawn_frames,
            config=config,
        )
        return field.compressed()

    # -- cutoffs ---------------------------------------------------------
    def _swept_box_intersects_area(self) -> np.ndarray:
        """Sound relevance test: could this hazard ever touch the movement area?"""

        model = geometry.MoveModel()
        area_min = np.array([model.x_min, model.y_min])
        area_max = np.array([model.x_max, model.y_max])
        horizon = float(self.config.horizon_frames)
        # Positions at the *visible* interval within the forecast window.
        t0 = np.maximum(self.spawn - self.frame_base, 0).astype(np.float64)
        t1 = np.minimum(self.despawn - self.frame_base, horizon).astype(np.float64)
        visible = t1 >= t0
        p0 = self.positions + self.velocities * t0[:, None]
        p1 = self.positions + self.velocities * t1[:, None]
        low = np.minimum(p0, p1) - self.halves
        high = np.maximum(p0, p1) + self.halves
        overlaps = (high[:, 0] >= area_min[0]) & (low[:, 0] <= area_max[0])
        overlaps &= (high[:, 1] >= area_min[1]) & (low[:, 1] <= area_max[1])
        return overlaps & visible

    def compressed(self) -> "DangerField":
        relevant = self._swept_box_intersects_area()
        indices = np.flatnonzero(relevant)
        dropped_irrelevant = int((~relevant).sum())
        over_cap = 0
        kept = indices
        if indices.size > self.config.max_entities:
            # Keep the ones that arrive earliest; the rest are an explicit
            # coverage gap, not a silent omission.
            t0 = np.maximum(self.spawn[indices] - self.frame_base, 0)
            order = np.argsort(t0, kind="stable")[: self.config.max_entities]
            over_cap = int(indices.size - order.size)
            kept = indices[order]
        return DangerField(
            frame_base=self.frame_base,
            positions=self.positions[kept],
            velocities=self.velocities[kept],
            halves=self.halves[kept],
            spawn=self.spawn[kept],
            despawn=self.despawn[kept],
            config=self.config,
            dropped_irrelevant=dropped_irrelevant,
            dropped_over_cap=over_cap,
            kept=kept,
            all_positions=self.positions[indices],
            all_velocities=self.velocities[indices],
            all_halves=self.halves[indices],
            all_spawn=self.spawn[indices],
            all_despawn=self.despawn[indices],
        )

    # -- queries ---------------------------------------------------------
    def unknown(self, t_rel: float) -> bool:
        return t_rel > self.config.horizon_frames or t_rel < 0

    def calibrated(self, t_rel: float) -> bool:
        return t_rel <= self.config.calibrated_frames

    def uncertainty(self, t_rel: float) -> float:
        t = max(0.0, float(t_rel))
        return self.config.uncertainty_base_px + self.config.uncertainty_per_frame * t

    def _active(self, frame_abs: float) -> np.ndarray:
        return (self.spawn <= frame_abs) & (frame_abs < self.despawn)

    def positions_at(self, t_rel: float) -> np.ndarray:
        return self.positions + self.velocities * float(t_rel)

    def clearance_many(self, points, t_rel: float, *, inflation: float | None = None):
        """Batch clearance for ``M`` points at one instant; ``None`` beyond the horizon.

        Returns ``(M,)`` where each entry is the signed box gap of that point to
        the nearest active hazard.  This is what makes the planner's 18-action
        expansion one NumPy call instead of eighteen Python calls.

        The per-axis formulation avoids the (M, N, 2) temporary and one reduce
        pass of the older stacked version; it is the planner's inner loop.
        """

        if self.unknown(t_rel):
            return None
        margin = self.config.inflation_px if inflation is None else float(inflation)
        centres, halves = self._active_at(float(t_rel))
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if centres.shape[0] == 0:
            return np.full(points.shape[0], np.inf, dtype=np.float64)
        inflate_x = self.config.player_half + halves[:, 0] + margin
        inflate_y = self.config.player_half + halves[:, 1] + margin
        dx = np.abs(points[:, 0][:, None] - centres[None, :, 0]) - inflate_x[None, :]
        dy = np.abs(points[:, 1][:, None] - centres[None, :, 1]) - inflate_y[None, :]
        return np.maximum(dx, dy).min(axis=1)

    def clearance(self, x: float, y: float, t_rel: float, *, inflation: float | None = None) -> float | None:
        """Signed box separation at one instant (``None`` beyond the horizon)."""

        if self.unknown(t_rel):
            return None
        margin = self.config.inflation_px if inflation is None else float(inflation)
        centres, halves = self._active_at(float(t_rel))
        if centres.shape[0] == 0:
            return float("inf")
        gap = np.abs(np.array([x, y])[None, :] - centres) - (self.config.player_half + halves + margin)
        return float(np.max(gap, axis=1).min())

    def frame_clearance(self, x: float, y: float, t_rel: int) -> float | None:
        """The game's predicate at an integer frame offset (no inflation)."""

        return self.clearance(x, y, int(t_rel), inflation=0.0)

    def swept_clearance(self, start, end, t0: float, t1: float, *, inflation: float | None = None) -> float | None:
        """Conservative continuous sweep between two instants (margin tool, not the predicate).

        Evaluated segment by segment with :func:`geometry.swept_box_clearance`, so
        a bullet crossing the hurtbox between two frames is caught here even
        though the game's per-frame predicate would not flag it.
        """

        if self.unknown(t0) or self.unknown(t1):
            return None
        margin = self.config.inflation_px if inflation is None else float(inflation)
        frames = max(1, int(round(t1 - t0)))
        a = np.asarray(start, dtype=np.float64)
        b = np.asarray(end, dtype=np.float64)
        velocity = (b - a) / float(frames)
        best = float("inf")
        for step in range(frames):
            t = t0 + step
            mask = self._active(self.frame_base + t) | self._active(self.frame_base + t + 1)
            if not np.any(mask):
                continue
            centres = (self.positions + self.velocities * t)[mask]
            halves = self.halves[mask]
            velocities = self.velocities[mask]
            player = a + velocity * step
            radius = self.config.player_half + margin
            for index in range(centres.shape[0]):
                gap = geometry.swept_box_clearance(
                    player, velocity, centres[index], velocities[index], radius, halves[index], 1.0
                )
                best = min(best, gap)
        return float(best)

    def occupancy(self, t_rel: int) -> np.ndarray:
        """Macro occupancy grid at one time node (rows=y, cols=x).

        Rasterised with a difference array (four corner updates + 2-D prefix sum)
        instead of one Python slice per hazard: the per-hazard loop dominated the
        build time (measured 9.9 ms at 400 entities) while the vectorised version
        is O(hazards + cells).
        """

        config = self.config
        cols = int(np.ceil(spec.PLAYFIELD_W / config.cell_px))
        rows = int(np.ceil(spec.PLAYFIELD_H / config.cell_px))
        cached = self._occupancy_cache.get(int(t_rel))
        if cached is not None:
            return cached
        mask = self._active(self.frame_base + float(t_rel))
        if not np.any(mask):
            return np.zeros((rows, cols), dtype=np.float32)
        centres = (self.positions + self.velocities * float(t_rel))[mask]
        halves = self.halves[mask]
        left = centres[:, 0] - halves[:, 0]
        right = centres[:, 0] + halves[:, 0]
        top = centres[:, 1] - halves[:, 1]
        bottom = centres[:, 1] + halves[:, 1]
        x0 = np.clip(np.floor((left - spec.PLAYFIELD_X) / config.cell_px).astype(np.int64), 0, cols - 1)
        x1 = np.clip(np.floor((right - spec.PLAYFIELD_X) / config.cell_px).astype(np.int64), 0, cols - 1)
        y0 = np.clip(np.floor((top - spec.PLAYFIELD_Y) / config.cell_px).astype(np.int64), 0, rows - 1)
        y1 = np.clip(np.floor((bottom - spec.PLAYFIELD_Y) / config.cell_px).astype(np.int64), 0, rows - 1)
        diff = np.zeros((rows + 1, cols + 1), dtype=np.int32)
        np.add.at(diff, (y0, x0), 1)
        np.add.at(diff, (y0, x1 + 1), -1)
        np.add.at(diff, (y1 + 1, x0), -1)
        np.add.at(diff, (y1 + 1, x1 + 1), 1)
        prefix = diff.cumsum(axis=0).cumsum(axis=1)
        grid = (prefix[:rows, :cols] > 0).astype(np.float32)
        grid.setflags(write=False)
        self._occupancy_cache[int(t_rel)] = grid
        return grid


@dataclass
class Forecast:
    """A danger field plus the contract record that describes it."""

    field: DangerField
    record: DangerForecast
    build_ms: float = 0.0

    def to_json_dict(self) -> dict:
        from .contracts import to_json_dict

        return to_json_dict(self.record)


def build(field: DangerField, *, elapsed_ms: float = 0.0) -> Forecast:
    nodes = field.config.time_nodes()
    if not nodes:
        raise ValueError("forecast needs at least one time node")
    rows, cols = field.occupancy(nodes[0]).shape
    coverage = tuple(tuple(float(v) for v in field.occupancy(node).reshape(-1)) for node in nodes)
    uncertainty = tuple(field.uncertainty(node) for node in nodes)
    reasons: list[str] = []
    if field.dropped_irrelevant:
        reasons.append("irrelevant: swept box never touches the movement area")
    if field.dropped_over_cap:
        reasons.append("over_cap: entity cap reached, coverage incomplete")
    record = DangerForecast(
        schema=FORECAST_SCHEMA,
        model_version=MODEL_VERSION,
        time_nodes_frames=tuple(int(node) for node in nodes),
        coverage_shape=(int(rows), int(cols)),
        coverage=coverage,
        uncertainty=uncertainty,
        valid_until=field.frame_base + field.config.horizon_frames,
        calibrated=field.config.calibrated_frames >= field.config.horizon_frames,
        dropped_entities=int(field.dropped_irrelevant + field.dropped_over_cap),
        drop_reasons=tuple(reasons),
    )
    return Forecast(field=field, record=record, build_ms=float(elapsed_ms))
