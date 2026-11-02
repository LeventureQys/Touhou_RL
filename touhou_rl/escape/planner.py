"""First-action-conditioned route search (D4).

What this module is responsible for
-----------------------------------
FR-06/FR-07: search a *route*, not a first action; verify every frame with the
game's predicate; report per-first-action status so a search failure is never
dressed up as "no route"; order candidate routes by hard safety first and local
clearance last.

How it differs from the D2 reference
------------------------------------
The reference (``reference.py``) exists to *label* instances: it is exhaustive
within a budget and deliberately conservative about claims.  This planner is the
subject under test: it is a beam search with a small width, one beam per first
action, and a continuation tail.  Its failure is a failure of the subject, not a
property of the world — which is why the two are separate modules and why the
evaluation compares them instead of merging them.

What the search optimises
-------------------------
Per decision it simulates ``frameskip`` frames with the measured 1-frame input
settle, clamps each frame, and rejects a child on the first frame whose box
clearance (with configured inflation) is <= 0.  Among surviving children it keeps
``beam_width`` per first action, ranked by accumulated minimum clearance.

Across the 18 first actions it ranks feasible routes by:

1. exit clearance at the end of the plan (a proxy for "the route still has room"),
2. accumulated minimum clearance over the whole plan,
3. fewer direction switches.

Local clearance is therefore the *last* tie-breaker, matching the requirement
that a slightly tighter but better-exiting route wins.  Exit clearance is a proxy;
it is not the full successor/exit-width analysis of the final system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .. import spec
from . import geometry
from . import exit as exit_module
from .contracts import (
    FirstActionStatus,
    PositionKnot,
    ROUTE_SCHEMA,
    RoutePlan,
    SolutionStatus,
)
from .forecast import DangerField, ForecastConfig
from .synth import SyntheticEnv

ACTIONS: tuple[tuple[int, int], ...] = tuple(
    (focus, move) for focus in (0, 1) for move in range(9)
)
ACTION_IDS: tuple[str, ...] = tuple(f"f{focus}-m{move}" for focus, move in ACTIONS)


@dataclass(frozen=True)
class PlannerConfig:
    frameskip: int = 2
    horizon_decisions: int = 30
    continuation_decisions: int = 10
    beam_width: int = 2
    inflation_px: float = 0.0
    switch_penalty: float = 0.02
    exit_clearance_px: float = 4.0
    max_nodes: int = 200_000
    settle_frames: int = 1
    #: ``"freedom"`` scores by the coarse closure map first (closing regions),
    #: ``"clearance"`` by local clearance first (narrow gaps the 16 px grid cannot
    #: see), ``"auto"`` picks one from coarse reachability and records why.
    score_mode: str = "auto"

    def validate(self) -> None:
        if self.frameskip < 1:
            raise ValueError("frameskip must be >= 1")
        if self.horizon_decisions < 1:
            raise ValueError("horizon_decisions must be >= 1")
        if self.continuation_decisions < 0:
            raise ValueError("continuation_decisions must be >= 0")
        if self.beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.inflation_px < 0.0:
            raise ValueError("inflation must be >= 0")
        if self.score_mode not in ("freedom", "clearance", "auto"):
            raise ValueError(f"unknown score_mode {self.score_mode!r}")


@dataclass
class _Node:
    x: float
    y: float
    previous: tuple[int, int]
    actions: tuple[tuple[int, int], ...]
    clearances: tuple[float, ...]
    switches: int

    @property
    def min_clearance(self) -> float:
        return min(self.clearances) if self.clearances else float("inf")


@dataclass(frozen=True)
class Route:
    first: tuple[int, int]
    actions: tuple[tuple[int, int], ...]
    clearances: tuple[float, ...]
    min_clearance: float
    end_clearance: float
    switches: int


@dataclass(frozen=True)
class PlannerResult:
    status: SolutionStatus
    route: Route | None
    first_action_status: tuple[str, ...]
    first_action_reason: tuple[str, ...]
    alternatives: tuple[str, ...]
    nodes: int
    elapsed_ms: float
    record: RoutePlan
    reasons: tuple[str, ...] = field(default=())
    score_mode: str = "clearance"
    mode_diagnostics: dict = field(default_factory=dict)


def _action_velocities(model: geometry.MoveModel) -> np.ndarray:
    """One-frame displacement of every action; constant, so computed once."""

    global _VELOCITIES
    if _VELOCITIES is None:
        _VELOCITIES = np.asarray(
            [model.displacement(action[0], action[1], frames=1) for action in ACTIONS],
            dtype=np.float64,
        )
    return _VELOCITIES


_VELOCITIES: np.ndarray | None = None


def _expand_batch(
    field: DangerField,
    node: _Node,
    config: PlannerConfig,
    model: geometry.MoveModel,
    t_offset: int,
):
    """Expand all 18 actions of one node with two vectorised clearance calls.

    The first frame of a decision uses the previous action, so it is identical for
    every candidate; only the remaining frames differ.  Doing this per candidate
    in Python was the planner's dominant cost.
    """

    velocities = _action_velocities(model)
    previous = model.displacement(node.previous[0], node.previous[1], frames=1)
    x1, y1 = model.clamp(node.x + previous[0], node.y + previous[1])
    first_frame = t_offset + len(node.clearances) + 1
    first_clearance = field.clearance(x1, y1, first_frame, inflation=config.inflation_px)
    if first_clearance is None or first_clearance <= 0.0:
        return []

    positions = np.tile(np.array([x1, y1], dtype=np.float64), (len(ACTIONS), 1))
    frame_values: list[np.ndarray] = [np.full(len(ACTIONS), float(first_clearance))]
    alive = np.ones(len(ACTIONS), dtype=bool)
    for step in range(2, config.frameskip + 1):
        positions[:, 0] = np.clip(positions[:, 0] + velocities[:, 0], model.x_min, model.x_max)
        positions[:, 1] = np.clip(positions[:, 1] + velocities[:, 1], model.y_min, model.y_max)
        values = field.clearance_many(positions, first_frame + step - 1, inflation=config.inflation_px)
        if values is None:
            return []
        alive &= values > 0.0
        frame_values.append(np.asarray(values, dtype=np.float64))

    children: list[tuple[int, _Node]] = []
    alive_indices = np.flatnonzero(alive)
    for index in alive_indices:
        action = ACTIONS[int(index)]
        # One entry per frame, exactly like the scalar implementation: the time
        # index of the next expansion is derived from the length of this tuple.
        clearances = node.clearances + tuple(float(values[index]) for values in frame_values)
        switches = node.switches + (1 if action != node.previous else 0)
        children.append(
            (
                int(index),
                _Node(
                    float(positions[index, 0]),
                    float(positions[index, 1]),
                    action,
                    node.actions + (action,),
                    clearances,
                    switches,
                ),
            )
        )
    return children


def _search_first_action(
    field: DangerField,
    start: tuple[float, float],
    previous: tuple[int, int],
    action: tuple[int, int],
    config: PlannerConfig,
    model: geometry.MoveModel,
    total_decisions: int,
    t_offset: int,
    freedom: "np.ndarray | None" = None,
    target: tuple[float, float] | None = None,
    mode: str = "clearance",
):
    """Beam search restricted to one first action; returns (route, reason, expansions)."""

    def score_many(nodes: list[_Node]) -> list[tuple]:
        """Vectorised score for a batch of nodes.

        The previous version called ``cell_of`` once per node from inside
        ``list.sort(key=...)`` — 85k Python calls per 20 decisions, ~0.7 s.  One
        numpy gather does the same work.
        """

        if not nodes:
            return []
        if mode == "freedom" and freedom is not None and target is not None:
            xs = np.fromiter((node.x for node in nodes), dtype=np.float64, count=len(nodes))
            ys = np.fromiter((node.y for node in nodes), dtype=np.float64, count=len(nodes))
            cell = field.config.cell_px
            rows, cols = freedom.shape
            col = np.clip(np.floor((xs - spec.PLAYFIELD_X) / cell), 0, cols - 1).astype(np.int64)
            row = np.clip(np.floor((ys - spec.PLAYFIELD_Y) / cell), 0, rows - 1).astype(np.int64)
            closure = freedom[row, col]
            distance = np.hypot(xs - target[0], ys - target[1])
            return [
                (int(closure[i]), float(-distance[i]), nodes[i].min_clearance, -nodes[i].switches)
                for i in range(len(nodes))
            ]
        return [(node.min_clearance, -node.switches, 0.0) for node in nodes]

    def score(node: _Node) -> tuple:
        return score_many([node])[0]

    root = _Node(start[0], start[1], previous, (), (), 0)
    index = ACTIONS.index(action)
    expansions = len(ACTIONS)
    first_frame = t_offset + 1
    first_children = _expand_batch(field, root, config, model, t_offset)
    child = next((node for i, node in first_children if i == index), None)
    if child is None:
        if field.unknown(first_frame):
            return None, "unknown: beyond the forecast horizon", expansions
        return None, "collision", expansions
    beam = [child]
    for depth in range(1, total_decisions):
        children: list[_Node] = []
        for node in beam:
            batch = _expand_batch(field, node, config, model, t_offset)
            expansions += len(ACTIONS)
            children.extend(candidate for _index, candidate in batch)
        if not children:
            if field.unknown(t_offset + len(beam[0].clearances) + 1):
                return None, "unknown: beyond the forecast horizon", expansions
            return None, f"beam_exhausted: no continuation survives at depth {depth}", expansions
        keys = score_many(children)
        order = sorted(range(len(children)), key=lambda i: keys[i], reverse=True)
        beam = [children[i] for i in order[: config.beam_width]]
    best = max(beam, key=score)
    route = Route(
        first=action,
        actions=best.actions,
        clearances=best.clearances,
        min_clearance=best.min_clearance,
        end_clearance=best.clearances[-1] if best.clearances else float("inf"),
        switches=best.switches,
    )
    return route, "", expansions


def _knots(field: DangerField, start, actions, config: PlannerConfig, previous, model) -> tuple[tuple[float, float], ...]:
    x, y = start
    prev_velocity = model.displacement(previous[0], previous[1], frames=1)
    knots = [(x, y)]
    for action in actions:
        velocity = model.displacement(action[0], action[1], frames=1)
        for step in range(1, config.frameskip + 1):
            delta = prev_velocity if step <= config.settle_frames else velocity
            x, y = model.clamp(x + delta[0], y + delta[1])
            knots.append((x, y))
        prev_velocity = velocity
    return tuple(knots)


def plan(
    field: DangerField,
    start: tuple[float, float],
    *,
    previous_action: tuple[int, int] = (0, 0),
    config: PlannerConfig | None = None,
    t_offset: int = 0,
    corridor: tuple = (),
) -> PlannerResult:
    """Plan a route for the next ``horizon_decisions`` decisions.

    ``t_offset`` shifts every query on the field, so one field can be reused by a
    closed-loop runner that advances in game frames instead of rebuilding the
    forecast each decision.

    ``corridor`` is an optional committed region sequence (``CorridorBox``): when
    given, routes whose end falls inside the corridor's box for their end frame
    are ranked first.  This is how the controller commits to a region while still
    re-aiming actions inside it.
    """

    config = config or PlannerConfig()
    config.validate()
    model = geometry.MoveModel()
    if field.dropped_over_cap:
        # The capped view cannot support safety claims, but the *full* entity set
        # can still prove that everything is covered.  Try that before giving up.
        proof = coverage_proof(field, config)
        if proof is not None:
            return _unknown_result(
                field,
                config,
                f"coverage_proof: every lattice point is covered at frame {proof}",
                status=SolutionStatus.NO_SOLUTION_CERTIFIED,
            )
        return _unknown_result(
            field,
            config,
            f"coverage_incomplete: {field.dropped_over_cap} entities dropped over the cap",
        )
    available = (field.config.horizon_frames - t_offset) // config.frameskip
    total = min(config.horizon_decisions + config.continuation_decisions, available)
    if total < 1:
        return _unknown_result(field, config, "forecast horizon exhausted at this offset")
    horizon = min(config.horizon_decisions, total)
    if config.max_nodes < len(ACTIONS):
        return _unknown_result(
            field,
            config,
            f"budget: max_nodes={config.max_nodes} is below one action sweep",
            status=SolutionStatus.BUDGET_EXHAUSTED,
        )

    started = time.perf_counter()
    routes: dict[int, Route] = {}
    statuses: list[str] = []
    reasons: list[str] = []
    nodes = 0
    freedom, target = _freedom_target(field, start)
    mode = config.score_mode
    mode_diagnostics: dict = {}
    if mode == "auto":
        mode, mode_diagnostics = exit_module.adaptive_mode(
            field, start, freedom, t_offset=t_offset
        )
    else:
        mode_diagnostics = {"criterion": "explicit score_mode", "best_closure": int(freedom.max())}
    start_clearance = field.clearance(start[0], start[1], t_offset, inflation=config.inflation_px)
    start_blocked = start_clearance is not None and start_clearance <= 0.0
    candidates: list[Route] = []
    for index, action in enumerate(ACTIONS):
        found: list[Route] = []
        last_reason = ""
        expansions_total = 0
        if start_blocked:
            last_reason = "immediate_collision: player already overlaps a hazard"
        else:
            route, why, expansions = _search_first_action(
                field, start, previous_action, action, config, model, total, t_offset,
                freedom=freedom, target=target, mode=mode,
            )
            expansions_total += expansions
            if route is not None:
                found.append(route)
            else:
                last_reason = why
        nodes += expansions_total
        if found:
            candidates.extend(found)
            statuses.append(FirstActionStatus.FEASIBLE.value)
            reasons.append("feasible")
        else:
            statuses.append(
                FirstActionStatus.IMMEDIATE_COLLISION.value
                if last_reason == "collision" or last_reason.startswith("immediate_collision")
                else FirstActionStatus.UNKNOWN.value
            )
            reasons.append(last_reason)
        if nodes > config.max_nodes:
            for rest in range(index + 1, len(ACTIONS)):
                statuses.append(FirstActionStatus.UNKNOWN.value)
                reasons.append("budget: node cap reached")
            break

    feasible = sorted(
        candidates,
        key=lambda route: (route.end_clearance, route.min_clearance, -route.switches),
        reverse=True,
    )
    if feasible:
        # Coarse exit analysis at the end of each feasible route: prefer routes
        # whose region stays open longest, then the larger region, and only then
        # the local clearance.  This is the "future feasibility over local
        # clearance" ordering of FR-07.
        analysed = []
        for route in feasible:
            actions = route.actions[:horizon]
            knots = _knots(field, start, actions, config, previous_action, model)
            report = exit_module.analyze(
                field, knots[-1][0], knots[-1][1], len(actions) * config.frameskip
            )
            analysed.append((route, knots, report))
        if corridor:
            # Region commitment: prefer routes that end inside the committed
            # corridor.  Without this the long-horizon ranking flips between the
            # dead end and the safe mouth every decision (SC-15 @20x4, 44 frames).
            inside = [
                item
                for item in analysed
                if _in_corridor(item[1][-1], len(item[0].actions) * config.frameskip, corridor)
            ]
            if inside:
                analysed = inside
        # Fourth key: when every route ends in the same coarse cell (common at long
        # horizons, where all first actions converge), the ranking above is a tie
        # and the sort would keep ACTIONS order — whose first entry is *idle*.
        # SC-15 @20x4 spent its first decisions standing still because of that.
        # Break the tie by how much the first action moves toward the target.
        target_dx = target[0] - start[0]
        target_dy = target[1] - start[1]
        norm = max(1e-9, (target_dx ** 2 + target_dy ** 2) ** 0.5)

        def approach(item) -> float:
            first = item[0].first
            velocity = model.displacement(first[0], first[1], frames=1)
            return (velocity[0] * target_dx + velocity[1] * target_dy) / norm

        analysed.sort(
            key=lambda item: (
                exit_module.rank_key(item[2]),
                approach(item),
                item[0].end_clearance,
                item[0].min_clearance,
                -item[0].switches,
            ),
            reverse=True,
        )
        chosen, knots, chosen_exit = analysed[0]
        status = SolutionStatus.FEASIBLE
        route_reasons = ("route survives the plan horizon and continuation tail",)
        for other, _knots_other, other_exit in analysed[1:]:
            if other_exit.closes_inside_horizon:
                reasons[ACTIONS.index(other.first)] = (
                    f"dead_end: region closes at frame {other_exit.closure_eta}, "
                    f"a verified alternative exists"
                )
    elif all(value == FirstActionStatus.IMMEDIATE_COLLISION.value for value in statuses):
        chosen = None
        chosen_exit = None
        status = SolutionStatus.NO_SOLUTION_CERTIFIED
        route_reasons = ("all_first_actions_collide: 18/18 die inside the first decision",)
    else:
        chosen = None
        chosen_exit = None
        proof = None
        if field.dropped_over_cap > 0:
            # The field is a capped view; a coverage proof must use every entity.
            proof = coverage_proof(field, config)
        if proof is not None:
            status = SolutionStatus.NO_SOLUTION_CERTIFIED
            route_reasons = (
                f"coverage_proof: every lattice point is covered at frame {proof} "
                f"({field.dropped_over_cap} entities beyond the field cap)",
            )
        else:
            status = SolutionStatus.BUDGET_EXHAUSTED
            route_reasons = ("no_route_found_within_beam: not a proof of impossibility",)

    if chosen is not None:
        actions = chosen.actions[:horizon]
        clearances = chosen.clearances[: horizon * config.frameskip]
        exit_eta = _exit_eta(clearances, config)
        closure_eta = chosen_exit.closure_eta if chosen_exit is not None else None
        slack = (
            None
            if exit_eta is None or closure_eta is None
            else int(closure_eta - horizon * config.frameskip)
        )
        record = RoutePlan(
            schema=ROUTE_SCHEMA,
            route_id=f"first-{ACTION_IDS[ACTIONS.index(chosen.first)]}",
            solution_status=status,
            first_action_status=tuple(statuses),
            first_action_reason=tuple(reasons),
            action_sequence=tuple(_route_action(action) for action in actions),
            position_time_knots=tuple(
                PositionKnot(frame=index * 1, x=float(x), y=float(y))
                for index, (x, y) in enumerate(knots)
            ),
            corridor_by_time=_corridor(field, knots, config),
            min_clearance=round(float(chosen.min_clearance), 4),
            exit_eta=exit_eta,
            closure_eta=closure_eta,
            slack=slack,
            continuation_window=config.continuation_decisions * config.frameskip,
            confidence=1.0,
            valid_until=field.frame_base + horizon * config.frameskip,
            alternatives=tuple(
                f"first-{ACTION_IDS[ACTIONS.index(route.first)]}" for route in feasible[1:]
            ),
            node_budget=config.max_nodes,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    else:
        record = RoutePlan(
            schema=ROUTE_SCHEMA,
            route_id="none",
            solution_status=status,
            first_action_status=tuple(statuses),
            first_action_reason=tuple(reasons),
            continuation_window=config.continuation_decisions * config.frameskip,
            confidence=0.0,
            valid_until=field.frame_base + horizon * config.frameskip,
            node_budget=config.max_nodes,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    return PlannerResult(
        status=status,
        route=chosen,
        first_action_status=tuple(statuses),
        first_action_reason=tuple(reasons),
        alternatives=record.alternatives,
        nodes=nodes,
        elapsed_ms=record.elapsed_ms,
        record=record,
        reasons=tuple(route_reasons),
        score_mode=mode,
        mode_diagnostics=dict(mode_diagnostics),
    )


def coverage_proof(field: DangerField, config: PlannerConfig, *, step: float = 8.0) -> int | None:
    """Return the first frame at which *every* reachable lattice point is covered.

    Used only when the field dropped entities over its cap: the capped field
    cannot certify coverage, but the full entity arrays can.  The lattice is
    inflated by the hurtbox, so "covered" means the point has no positive
    clearance; a proof at one frame is sufficient because the player cannot leave
    the covered region.

    Cost is bounded by an early exit: the loop stops at the first covered frame,
    and points are processed in chunks so the (points × entities) array stays
    small.
    """

    import numpy as np

    from . import geometry

    model = geometry.MoveModel()
    xs = np.arange(model.x_min, model.x_max + 1e-9, step)
    ys = np.arange(model.y_min, model.y_max + 1e-9, step)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    horizon = min(field.config.horizon_frames, config.horizon_decisions * config.frameskip)
    for t in range(1, horizon + 1):
        absolute = field.frame_base + t
        active = (field.all_spawn <= absolute) & (field.all_despawn > absolute)
        if not active.any():
            continue
        inflate = (field.config.player_half + field.all_halves[active]).reshape(1, -1, 2)
        centres = (field.all_positions[active] + field.all_velocities[active] * float(t)).reshape(1, -1, 2)
        covered = np.ones(grid.shape[0], dtype=bool)
        for start in range(0, grid.shape[0], 512):
            chunk = grid[start : start + 512].reshape(-1, 1, 2)
            gaps = np.abs(chunk - centres) - inflate - field.config.inflation_px
            free = gaps.max(axis=2).min(axis=1) > 0.0
            if free.any():
                covered[start : start + 512] = False
        if covered.all():
            return t
    return None


def _in_corridor(point: tuple[float, float], frame: int, corridor: tuple, tolerance: float = 8.0) -> bool:
    """Is ``point`` inside the corridor box closest in time to ``frame``?"""

    if not corridor:
        return True
    best = min(corridor, key=lambda box: abs(box.frame - frame))
    x, y = point
    return (
        best.x0 - tolerance <= x <= best.x1 + tolerance
        and best.y0 - tolerance <= y <= best.y1 + tolerance
    )


def _corridor(field: DangerField, knots, config: PlannerConfig, *, half: float = 24.0) -> tuple:
    """Compress a verified route into one box every few frames.

    A box is a region the route is known to pass through, inflated by ``half`` px
    so that re-aiming inside it stays useful.  Boxes are emitted every
    ``2 * frameskip`` frames, which keeps a 40-frame plan to ~10 boxes.
    """

    from .contracts import CorridorBox

    model = geometry.MoveModel()
    boxes: list[CorridorBox] = []
    step = max(1, config.frameskip * 2)
    for index, (x, y) in enumerate(knots):
        if index % step:
            continue
        boxes.append(
            CorridorBox(
                frame=index,
                x0=max(model.x_min, float(x) - half),
                y0=max(model.y_min, float(y) - half),
                x1=min(model.x_max, float(x) + half),
                y1=min(model.y_max, float(y) + half),
            )
        )
    return tuple(boxes)


def _freedom_target(field: DangerField, start: tuple[float, float]):
    """Closure table plus the centre of the cell that stays open longest.

    Two alternatives were measured and both were worse (see the D32/D33 notes):

    * ranking route progress before the direction prior made SC-15 collapse
      (1/13 survived) because routes beeline at the target and ignore corridors;
    * requiring the target to lie *outside* the player's current component also
      collapsed SC-15 (1/13), sending the route into the dead end.

    The current rule — nearest maximum-closure cell, ties by scan order — is kept
    because it is the one that passes the gates.  Its known cost is the mirror
    asymmetry documented as an xfail in the test suite.
    """

    import numpy as np

    from .. import spec

    table = exit_module.closure_map(field)
    playable = exit_module.playable_mask(field)
    table = np.where(playable, table, -1)
    best = int(table.max())
    if best < 0:
        return table, start
    rows, cols = np.nonzero(table == best)
    cell = field.config.cell_px
    xs = spec.PLAYFIELD_X + (cols + 0.5) * cell
    ys = spec.PLAYFIELD_Y + (rows + 0.5) * cell
    distance = (xs - start[0]) ** 2 + (ys - start[1]) ** 2
    index = int(np.argmin(distance))
    return table, (float(xs[index]), float(ys[index]))


def _exit_eta(clearances, config: PlannerConfig) -> int | None:
    """First frame after which the (inflated) clearance stays above the exit threshold."""

    window = max(1, config.continuation_decisions * config.frameskip)
    for index in range(max(0, len(clearances) - window + 1)):
        if all(value >= config.exit_clearance_px for value in clearances[index : index + window]):
            return int(index)
    return None


def _route_action(action: tuple[int, int]) -> tuple[int, int, int, int]:
    """Routing action -> the game's ``[move, shoot, focus, bomb]`` contract."""
    focus, move = action
    return (int(move), 1, int(focus), 0)


def _unknown_result(
    field: DangerField,
    config: PlannerConfig,
    reason: str,
    status: SolutionStatus = SolutionStatus.MODEL_UNKNOWN,
) -> PlannerResult:
    statuses = tuple(FirstActionStatus.UNKNOWN.value for _ in ACTIONS)
    reasons = tuple(reason for _ in ACTIONS)
    record = RoutePlan(
        schema=ROUTE_SCHEMA,
        route_id="none",
        solution_status=status,
        first_action_status=statuses,
        first_action_reason=reasons,
        continuation_window=config.continuation_decisions * config.frameskip,
        confidence=0.0,
        valid_until=field.frame_base + config.horizon_decisions * config.frameskip,
        node_budget=config.max_nodes,
        elapsed_ms=0.0,
    )
    return PlannerResult(
        status=status,
        route=None,
        first_action_status=statuses,
        first_action_reason=reasons,
        alternatives=(),
        nodes=0,
        elapsed_ms=0.0,
        record=record,
        reasons=(reason,),
    )


def run_closed_loop(
    scenario,
    *,
    plan_config: PlannerConfig | None = None,
    forecast_config=None,
    max_decisions: int | None = None,
) -> dict:
    """Receding-horizon evaluation of the D4 planner against the synthetic world.

    The field is built once at frame 0 and queried with ``t_offset``: in the
    synthetic world hazard motion is deterministic, so a privileged teacher may
    look ahead.  This is an **R2 evaluation harness**, not a deployment claim;
    the controller (D5) will rebuild the field from the adapter each decision.

    The first action of each plan is executed in :class:`synth.SyntheticEnv`, so
    the outcome is decided by the game's per-frame predicate, not by the plan's
    own bookkeeping.
    """

    from . import synth as synth_module

    plan_config = plan_config or PlannerConfig()
    config = forecast_config or ForecastConfig()
    env = SyntheticEnv(scenario)
    field = DangerField.from_scenario(scenario, 0, config)
    chosen_mode = plan_config.score_mode
    if chosen_mode == "auto":
        # Decide the scoring mode once, from the initial state.  Re-deciding every
        # decision made the controller oscillate between freedom and clearance
        # mid-run (measured: SC-16 regressed from 180 frames to 149).
        table = exit_module.closure_map(field)
        chosen_mode, _diagnostics = exit_module.adaptive_mode(field, scenario.player_start, table)
        plan_config = replace(plan_config, score_mode=chosen_mode)
    decisions = max_decisions or scenario.horizon_frames // plan_config.frameskip
    statuses: dict[str, int] = {}
    fallbacks = 0
    plan_ms: list[float] = []
    first_actions: list[tuple[int, int]] = []
    plans = 0
    while not env.hit and env.frame < scenario.horizon_frames and plans < decisions:
        result = plan(
            field,
            (env.x, env.y),
            previous_action=env.previous_action,
            config=plan_config,
            t_offset=env.frame,
        )
        statuses[result.status.value] = statuses.get(result.status.value, 0) + 1
        plan_ms.append(result.elapsed_ms)
        plans += 1
        if result.route is None:
            fallbacks += 1
            action = (0, 0)
        else:
            contract_action = result.record.action_sequence[0]
            action = (int(contract_action[2]), int(contract_action[0]))
        first_actions.append(action)
        env.step(action)
    return {
        "scenario_id": scenario.scenario_id,
        "plans": plans,
        "frames": env.frame,
        "hit": env.hit,
        "hit_frame": env.frame if env.hit else None,
        "survived": (not env.hit) and env.frame >= scenario.horizon_frames,
        "fallbacks": fallbacks,
        "status_counts": statuses,
        "plan_ms_p50": round(float(sorted(plan_ms)[len(plan_ms) // 2]), 3) if plan_ms else None,
        "first_actions": first_actions,
        "min_clearance": None if env.min_clearance == float("inf") else round(env.min_clearance, 4),
    }

