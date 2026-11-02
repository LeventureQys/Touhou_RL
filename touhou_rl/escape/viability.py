"""v3.7.2 finite-witness macro teacher; deliberately independent of live control.

A false ``feasible_mask`` entry means *not witnessed*, NEVER infeasible.  Only
``constraint_mask`` may filter a policy: it selects positive witnesses when the
whole search finished on a complete, uncut, calibrated forecast.  On budget,
unknown, or no witnesses it is all True.  Thus beam pruning is never a negative
label or a proof of closure.  This optional positive-only curriculum can exclude
unexplored safe actions; consumers must not reinterpret it as a viability kernel.

Actions are focus*9+move.  Each decision's first frame uses the preceding action
(the measured one-frame input settle); every remaining frame uses the requested
action.  Movement and clearance are checked at EVERY logic frame, including the
settle frame.  No coarse grid participates in verification.  Coverage/model flags
are caller attestations about the source observation, not inferred from geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isnan
from typing import Sequence

import numpy as np

from .forecast import DangerField
from .geometry import MoveModel

ACTIONS = tuple((focus, move) for focus in (0, 1) for move in range(9))


@dataclass(frozen=True)
class ViabilityConfig:
    horizon_frames: int = 90
    frameskip: int = 6
    beam_width: int = 6
    spatial_cell_px: float = 12.0
    max_frame_checks: int = 500_000
    inflation_px: float = 0.0
    target_radius_px: float = 16.0
    target_deadline_frames: int | None = None

    def __post_init__(self):
        for name in ("horizon_frames", "frameskip", "beam_width"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.max_frame_checks, int) or self.max_frame_checks < 0:
            raise ValueError("max_frame_checks must be a nonnegative integer")
        if not np.isfinite(self.spatial_cell_px) or self.spatial_cell_px <= 0:
            raise ValueError("spatial_cell_px must be finite and positive")
        if not np.isfinite(self.target_radius_px) or self.target_radius_px < 0:
            raise ValueError("target_radius_px must be finite and nonnegative")
        deadline = self.target_deadline_frames
        if deadline is not None and (not isinstance(deadline, int) or isinstance(deadline, bool)
                                     or not 0 <= deadline <= self.horizon_frames):
            raise ValueError("target_deadline_frames must be in 0..horizon_frames")
        if not np.isfinite(self.inflation_px) or self.inflation_px < 0:
            raise ValueError("inflation_px must be finite and nonnegative")


@dataclass(frozen=True)
class VerifiedRoute:
    first_action: int
    actions: tuple[int, ...]
    positions: tuple[tuple[float, float], ...]  # origin plus EVERY frame
    clearances: tuple[float, ...]
    t_offset: int
    arrival_frame: int | None = None  # relative to decision, not field base


@dataclass(frozen=True)
class ViabilityResult:
    status: str  # feasible / unknown / budget; never infeasible
    feasible_mask: tuple[bool, ...]  # positive witnesses, not negative labels
    constraint_mask: tuple[bool, ...]  # safe to pass to policy selection
    chosen_candidate: int  # original policy argmax inside constraint_mask
    verified_route: VerifiedRoute | None
    routes: tuple[VerifiedRoute | None, ...]
    frame_checks: int
    reasons: tuple[str, ...]
    constraints_enabled: bool

    @property
    def witness_mask(self):
        """Positive-only supervision: False must be ignored, never BCE negatives."""
        return self.feasible_mask

    @property
    def known_negative_mask(self):
        """This bounded beam teacher never certifies a negative action label."""
        return (False,) * 18

    @property
    def route(self):
        return self.verified_route


def select_policy_action(policy_logits: Sequence[float], mask: Sequence[bool]) -> int:
    """Preserve original logits and their ordering; no teacher-label replacement."""
    logits = np.asarray(policy_logits, dtype=float)
    allowed = np.asarray(mask, dtype=bool)
    if logits.shape != (18,) or allowed.shape != (18,):
        raise ValueError("policy logits and mask must have 18 entries")
    if not np.isfinite(logits).all():
        raise ValueError("policy logits must be finite")
    if not allowed.any():
        raise ValueError("empty mask is not a valid policy constraint")
    return int(np.argmax(np.where(allowed, logits, -np.inf)))


def apply_policy_constraint(action: Sequence[int], result: ViabilityResult) -> tuple[int, ...]:
    """Replace only (focus, move); preserve shoot/bomb/helper tail verbatim.

    This helper expects routing-first layout, NOT an implicit game action layout.
    The teacher itself never sends inputs or calls shoot/bomb/helper controllers.
    """
    if len(action) < 2:
        raise ValueError("expected routing-first (focus, move, *auxiliary) action")
    return (*ACTIONS[result.chosen_candidate], *action[2:])


@dataclass(frozen=True)
class _Node:
    positions: tuple[tuple[float, float], ...]
    clearances: tuple[float, ...]
    actions: tuple[int, ...]
    previous: int
    minimum: float = float("inf")
    arrival_frame: int | None = None


def _diverse(nodes: list[_Node], config: ViabilityConfig, target) -> list[_Node]:
    """One best representative per spatial bin before duplicates, farthest first.

    Clearance and optional macro target distance rank representatives but never
    establish collision safety. An explicit deadline prioritises already-arrived
    branches so arrival cannot be erased by a nearer, not-yet-arrived state.
    Max-min spatial selection prevents a high-clearance cluster from
    consuming the whole beam when a tighter lateral escape remains available.
    """
    def rank(node):
        x, y = node.positions[-1]
        distance = 0.0 if target is None else hypot(x - target[0], y - target[1])
        reached = config.target_deadline_frames is not None and node.arrival_frame is not None
        return (reached, -distance, node.minimum)

    ordered = sorted(nodes, key=rank, reverse=True)
    cells: dict[tuple[int, int], _Node] = {}
    for node in ordered:
        x, y = node.positions[-1]
        cells.setdefault((int(x // config.spatial_cell_px), int(y // config.spatial_cell_px)), node)
    representatives = list(cells.values())
    chosen = representatives[:1]
    remaining = representatives[1:]
    while remaining and len(chosen) < config.beam_width:
        def distance(node):
            x, y = node.positions[-1]
            return min((x - c.positions[-1][0]) ** 2 + (y - c.positions[-1][1]) ** 2 for c in chosen)
        index = max(range(len(remaining)), key=lambda i: distance(remaining[i]))
        chosen.append(remaining.pop(index))
    # Fill unused capacity without wasting it on identical movement states.
    keys = {(n.positions[-1], n.previous) for n in chosen}
    for node in ordered:
        if len(chosen) >= config.beam_width:
            break
        key = (node.positions[-1], node.previous)
        if key not in keys:
            chosen.append(node)
            keys.add(key)
    return chosen


def search_viability(
    field: DangerField,
    start: tuple[float, float],
    previous: tuple[int, int],
    policy_logits: Sequence[float],
    *,
    config: ViabilityConfig | None = None,
    model: MoveModel | None = None,
    t_offset: int = 0,
    coverage_complete: bool = True,
    model_known: bool = True,
    target: tuple[float, float] | None = None,
) -> ViabilityResult:
    """Search all 18 first actions for calibrated per-frame positive witnesses.

    ``t_offset`` is relative to DangerField.frame_base, NOT absolute game time.
    Caller must set coverage_complete/model_known False for torn observations,
    unsupported/nonlinear dynamics, unknown movement calibration, etc.  Dropped
    over-cap entities and out-of-calibration offsets independently disable output.
    Budget counts queried points, not vectorised calls; no realtime guarantee.
    With target_deadline_frames set, target is a caller-declared goal (never
    derived from coarse closure): reach its radius by the relative deadline,
    then survive until horizon. Leaving the goal after arrival is permitted.
    Without a deadline, target remains only an optional ranking preference.
    """
    config = config or ViabilityConfig()
    model = model or MoveModel()
    select_policy_action(policy_logits, (True,) * 18)  # validate before searching
    if previous not in ACTIONS:
        raise ValueError("previous must be a (focus, move) action")
    if not isinstance(t_offset, int) or t_offset < 0:
        raise ValueError("t_offset must be a nonnegative integer")
    if len(start) != 2 or not np.isfinite(start).all():
        raise ValueError("start must be a finite position")
    if model.clamp(*start) != tuple(start):
        raise ValueError("start outside movement bounds")
    if target is not None and (len(target) != 2 or not np.isfinite(target).all()):
        raise ValueError("target must be a finite position")
    if config.target_deadline_frames is not None and target is None:
        raise ValueError("target is required when target_deadline_frames is set")
    routes: list[VerifiedRoute | None] = [None] * 18
    checks = 0

    def finish(status, reasons):
        positive = tuple(route is not None for route in routes)
        enabled = status == "feasible" and any(positive)
        mask = positive if enabled else (True,) * 18
        chosen = select_policy_action(policy_logits, mask)
        return ViabilityResult(status, positive, mask, chosen, routes[chosen],
                               tuple(routes), checks, tuple(reasons), enabled)

    end = t_offset + config.horizon_frames
    if not model_known or not coverage_complete or field.dropped_over_cap:
        return finish("unknown", ("model_or_coverage_unknown",))
    if any(field.unknown(t) or not field.calibrated(t) for t in range(t_offset, end + 1)):
        return finish("unknown", ("outside_known_calibrated_horizon",))
    if checks >= config.max_frame_checks:
        return finish("budget", ("frame_check_budget",))
    checks += 1
    origin_clearance = field.clearance(*start, t_offset, inflation=config.inflation_px)
    if origin_clearance is None or isnan(float(origin_clearance)):
        return finish("unknown", ("clearance_unknown",))
    if origin_clearance <= 0:
        return finish("unknown", ("no_witness_at_origin",))

    velocities = model.action_displacements(frames=1)  # per-model, never global cache
    previous_index = ACTIONS.index(previous)
    goal_enabled = config.target_deadline_frames is not None
    at_goal = goal_enabled and hypot(start[0] - target[0], start[1] - target[1]) <= config.target_radius_px
    root = _Node((tuple(start),), (), (), previous_index, arrival_frame=0 if at_goal else None)
    beams = [[root] for _ in ACTIONS]
    elapsed = 0
    while elapsed < config.horizon_frames:
        duration = min(config.frameskip, config.horizon_frames - elapsed)
        for first, beam in enumerate(beams):
            pairs = [(node, action) for node in beam
                     for action in ([first] if elapsed == 0 else range(18))]
            if not pairs:
                continue
            count = len(pairs)
            xy = np.asarray([node.positions[-1] for node, _ in pairs], dtype=float)
            histories = [[] for _ in pairs]
            gaps = [[] for _ in pairs]
            arrivals = [node.arrival_frame for node, _ in pairs]
            alive = np.ones(count, dtype=bool)
            for step in range(duration):
                indices = np.flatnonzero(alive)
                if not len(indices):
                    break
                if checks + len(indices) > config.max_frame_checks:
                    return finish("budget", ("frame_check_budget",))
                actions = [pairs[i][0].previous if step == 0 else pairs[i][1] for i in indices]
                xy[indices] += velocities[actions]
                xy[indices, 0] = np.clip(xy[indices, 0], model.x_min, model.x_max)
                xy[indices, 1] = np.clip(xy[indices, 1], model.y_min, model.y_max)
                t = t_offset + elapsed + step + 1
                checks += len(indices)
                values = field.clearance_many(xy[indices], t, inflation=config.inflation_px)
                if values is None or np.isnan(values).any():
                    return finish("unknown", ("clearance_unknown",))
                for i, value in zip(indices, values):
                    if value <= 0:
                        alive[i] = False
                    else:
                        histories[i].append(tuple(map(float, xy[i])))
                        gaps[i].append(float(value))
                        relative_frame = elapsed + step + 1
                        if goal_enabled and arrivals[i] is None:
                            if (relative_frame <= config.target_deadline_frames
                                    and hypot(xy[i, 0] - target[0], xy[i, 1] - target[1])
                                    <= config.target_radius_px):
                                arrivals[i] = relative_frame
                            elif relative_frame >= config.target_deadline_frames:
                                alive[i] = False
            children = []
            for i in np.flatnonzero(alive):
                node, action = pairs[i]
                children.append(_Node(node.positions + tuple(histories[i]),
                                      node.clearances + tuple(gaps[i]),
                                      node.actions + (action,), action,
                                      min(node.minimum, min(gaps[i])), arrivals[i]))
            beams[first] = _diverse(children, config, target)
        elapsed += duration
    for first, beam in enumerate(beams):
        eligible = [node for node in beam if not goal_enabled or node.arrival_frame is not None]
        if eligible:
            best = max(eligible, key=lambda node: node.minimum)
            routes[first] = VerifiedRoute(first, best.actions, best.positions,
                                          best.clearances, t_offset, best.arrival_frame)
    if any(route is not None for route in routes):
        return finish("feasible", ("positive_witnesses_only",))
    return finish("unknown", ("beam_found_no_witness_not_infeasibility",))
