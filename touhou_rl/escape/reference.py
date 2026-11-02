"""Independent reference solver for the synthetic testbed (D2).

What "reference" means here
---------------------------
The acceptance document requires that a failed search is **not** reported as
"no route", and that a ``certified_blocked`` claim carries conservative evidence.
So this module separates three things that are easy to conflate:

* :func:`certify_blocked` — a *proof* that no sequence survives: either the
  player already overlaps a hazard at frame 0, or the active static hazards cover
  the entire movement rectangle from some frame on.  Coverage is checked by
  sampling the rectangle and, because the covering hazards are static, coverage
  persists, so every reachable position is hit.  No search involved.
* :func:`solve` — a bounded search that either finds a route (``feasible``) or
  reports ``budget_exhausted``.  Exhausting the *discretised* state graph is not
  a proof and is deliberately reported as a budget outcome with a reason string.
* :func:`bruteforce` — exhaustive enumeration of every action sequence up to a
  small depth.  Used to cross-check :func:`solve` on tiny instances; it is exact
  and therefore the yardstick, but exponential.

The solver shares the world physics with the testbed (that is the definition of
the world) and implements its own search and validation.  Routes are always
re-simulated exactly before being returned, so a discretisation artefact cannot
be reported as a route.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from . import geometry, synth
from .contracts import SolutionStatus

#: Grid used only to deduplicate visited states, never to compute positions.
DEDUP_QUANTUM = 0.25

ACTIONS: tuple[tuple[int, int], ...] = tuple(
    (focus, move) for focus in (0, 1) for move in range(9)
)


@dataclass(frozen=True)
class ReferenceResult:
    status: SolutionStatus
    actions: tuple[tuple[int, int], ...] = ()
    survived_frames: int = 0
    min_clearance: float | None = None
    nodes: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "actions": [list(action) for action in self.actions],
            "survived_frames": self.survived_frames,
            "min_clearance": self.min_clearance,
            "nodes": self.nodes,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# proof of no route (no search)
# ---------------------------------------------------------------------------
def _lattice_covers(scenario: synth.Scenario, frame: int) -> bool:
    """Sound coverage test for a full static lattice of equal boxes.

    This is the shape ``families._area_fill`` produces: positions on a regular
    lattice, equal half extents, spacing at most ``2 * half`` on each axis, and
    the lattice extending past the movement rectangle by at least one half
    extent.  Under those conditions the union of boxes provably covers the whole
    rectangle, so every reachable position is hit.

    Sampling the rectangle instead would be *unsound*: a gap narrower than the
    sample step could be missed.  That is why this function checks structure.
    """
    positions, _velocities, halves = scenario.active_positions(frame)
    if positions.shape[0] == 0:
        return False
    if not np.allclose(halves, halves[0]):
        return False
    half = halves[0]
    if np.any(half <= 0.0):
        return False
    xs = np.unique(np.round(positions[:, 0], 6))
    ys = np.unique(np.round(positions[:, 1], 6))
    if xs.size * ys.size != positions.shape[0]:
        return False
    if xs.size > 1 and np.max(np.diff(xs)) > 2.0 * half[0] + 1e-9:
        return False
    if ys.size > 1 and np.max(np.diff(ys)) > 2.0 * half[1] + 1e-9:
        return False
    model = geometry.MoveModel()
    return bool(
        xs[0] - half[0] <= model.x_min + 1e-9
        and xs[-1] + half[0] >= model.x_max - 1e-9
        and ys[0] - half[1] <= model.y_min + 1e-9
        and ys[-1] + half[1] >= model.y_max - 1e-9
    )


def certify_blocked(scenario: synth.Scenario) -> ReferenceResult | None:
    """Return a proof-carrying result, or ``None`` when no proof applies."""

    initial = synth.initial_clearance(scenario)
    if initial <= 0.0:
        return ReferenceResult(
            status=SolutionStatus.NO_SOLUTION_CERTIFIED,
            survived_frames=0,
            min_clearance=initial,
            reason="initial_overlap: the player already overlaps a hazard at frame 0",
        )
    model = geometry.MoveModel()
    for frame in range(1, scenario.horizon_frames + 1):
        positions, velocities, halves = scenario.active_positions(frame)
        if positions.shape[0] == 0:
            continue
        # Soundness requires the cover to persist: every covering hazard is
        # static and alive through the horizon.
        active_specs = [h for h in scenario.hazards if h.active(frame)]
        if any(h.velocity != (0.0, 0.0) for h in active_specs):
            continue
        if any(h.despawn_frame <= scenario.horizon_frames for h in active_specs):
            continue
        if _lattice_covers(scenario, frame):
            return ReferenceResult(
                status=SolutionStatus.NO_SOLUTION_CERTIFIED,
                survived_frames=frame,
                reason=f"area_covered: static hazards cover the whole movement area from frame {frame}",
            )
    # A cheap but sound one-decision proof: every first action collides.
    survivors = 0
    for action in ACTIONS:
        result = synth.advance(scenario, 0, *scenario.player_start, action, (0, 0), model=model)
        if not result.hit:
            survivors += 1
    if survivors == 0:
        return ReferenceResult(
            status=SolutionStatus.NO_SOLUTION_CERTIFIED,
            survived_frames=0,
            reason="all_first_actions_collide: 18/18 first actions die within one decision",
        )
    return None


# ---------------------------------------------------------------------------
# bounded search
# ---------------------------------------------------------------------------
def _key(frame: int, x: float, y: float, previous: tuple[int, int]) -> tuple:
    return (
        frame,
        int(round(x / DEDUP_QUANTUM)),
        int(round(y / DEDUP_QUANTUM)),
        previous[0],
        previous[1],
    )


def solve(
    scenario: synth.Scenario,
    *,
    node_budget: int = 20000,
    model: geometry.MoveModel | None = None,
) -> ReferenceResult:
    """Depth-first, safest-first search for *any* surviving action sequence."""

    if scenario.unsupported:
        return ReferenceResult(
            status=SolutionStatus.MODEL_UNKNOWN,
            reason=f"unsupported hazard class: {scenario.unsupported}",
        )
    proof = certify_blocked(scenario)
    if proof is not None:
        return proof
    model = model or geometry.MoveModel()
    start = (0, scenario.player_start[0], scenario.player_start[1], (0, 0))
    parents: dict[tuple, tuple] = {}
    stack: list[tuple] = [start]
    visited: set[tuple] = {_key(*start)}
    nodes = 0
    while stack:
        frame, x, y, previous = stack.pop()
        nodes += 1
        if nodes > node_budget:
            return ReferenceResult(
                status=SolutionStatus.BUDGET_EXHAUSTED,
                nodes=nodes,
                reason=f"node budget {node_budget} exhausted at frame {frame}",
            )
        if frame >= scenario.horizon_frames:
            route: list[tuple[int, int]] = []
            key = _key(frame, x, y, previous)
            while key in parents:
                key, action = parents[key]
                route.append(action)
            route.reverse()
            outcome = synth.simulate(scenario, route)
            if not outcome["survived"]:
                raise AssertionError(
                    f"{scenario.scenario_id}: reference route failed exact re-simulation"
                )
            return ReferenceResult(
                status=SolutionStatus.FEASIBLE,
                actions=tuple(route),
                survived_frames=outcome["frames"],
                min_clearance=outcome["min_clearance"],
                nodes=nodes,
                reason="route survives the horizon and passes exact re-simulation",
            )
        children = []
        for action in ACTIONS:
            result = synth.advance(scenario, frame, x, y, action, previous, model=model)
            if result.hit:
                continue
            key = _key(result.frame, result.x, result.y, action)
            if key in visited:
                continue
            children.append((result, action, key))
        children.sort(key=lambda item: -item[0].min_clearance)
        for result, action, key in children:
            visited.add(key)
            parents[key] = (_key(frame, x, y, previous), action)
            stack.append((result.frame, result.x, result.y, action))
    return ReferenceResult(
        status=SolutionStatus.BUDGET_EXHAUSTED,
        nodes=nodes,
        reason=(
            "discretised state graph exhausted with no surviving route; this is not a "
            "proof of impossibility"
        ),
    )


def latest_safe_start(
    scenario: synth.Scenario,
    *,
    cell_px: float = 8.0,
    horizon: int | None = None,
) -> dict:
    """Last frame at which a *fresh* start from the initial position can survive.

    This is the A-03 label ("最迟安全启动").  It is computed with a backward
    reachability sweep on a regular cell lattice, using only scenario geometry —
    no planner, no controller, so the subject cannot grade its own homework.

    Method and its honest limit:

    * a cell is usable at frame ``t`` when its centre has strictly positive
      clearance to every hazard active at ``t``;
    * ``R_t`` = usable at ``t`` and (from ``t+1``) at least one of its 8 neighbours
      is in ``R_{t+1}``; at the horizon ``R_H`` is simply the usable set;
    * the label is ``max { t : start cell ∈ R_t }``.

    The lattice is a *superset* approximation of where the player can be (cells
    are 8 px apart, the player moves at most 4 px/frame), so a label is
    optimistic at the boundary.  It is used only to check "did the controller
    start evacuating before the last safe moment", with a margin, on a handful of
    instances; it is not a collision proof.
    """

    model = geometry.MoveModel()
    horizon = scenario.horizon_frames if horizon is None else horizon
    xs = np.arange(model.x_min, model.x_max + 1e-9, cell_px)
    ys = np.arange(model.y_min, model.y_max + 1e-9, cell_px)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    rows, cols = xs.size, ys.size

    usable: list[np.ndarray] = []
    # A cell counts as usable only when its whole square is clear: inflating by
    # ~0.75 cells stops the 8-connected sweep from jumping diagonally through a
    # wall (that bug made E-06's 0.65 px wall look crossable on the 8 px lattice).
    cell_inflation = synth.HURTBOX_HALF_PX + cell_px * 0.75
    for frame in range(horizon + 1):
        positions, _velocities, halves = scenario.active_positions(frame)
        if positions.shape[0] == 0:
            usable.append(np.ones(grid.shape[0], dtype=bool))
            continue
        gaps = np.abs(grid[:, None, :] - positions[None, :, :]) - (
            cell_inflation + halves
        )[None, :, :]
        clearance = gaps.max(axis=2).min(axis=1)
        usable.append(clearance > 0.0)

    reach = [None] * (horizon + 1)
    reach[horizon] = usable[horizon].reshape(rows, cols)
    for frame in range(horizon - 1, -1, -1):
        nxt = reach[frame + 1].copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                shifted = np.zeros_like(nxt)
                src = nxt[max(0, -dr) : rows - max(0, dr), max(0, -dc) : cols - max(0, dc)]
                shifted[max(0, dr) : rows - max(0, -dr), max(0, dc) : cols - max(0, -dc)] = src
                nxt |= shifted
        reach[frame] = usable[frame].reshape(rows, cols) & nxt

    row0 = int(round((scenario.player_start[1] - model.y_min) / cell_px))
    col0 = int(round((scenario.player_start[0] - model.x_min) / cell_px))
    row0 = max(0, min(row0, rows - 1))
    col0 = max(0, min(col0, cols - 1))

    # "Last safe start" is only meaningful while a do-nothing player is still
    # alive: after that frame the question is moot because the episode is over.
    lazy_death = None
    for frame in range(horizon + 1):
        positions, _velocities, halves = scenario.active_positions(frame)
        if positions.shape[0] == 0:
            continue
        gaps = np.abs(np.asarray(scenario.player_start)[None, :] - positions) - (
            synth.HURTBOX_HALF_PX + halves
        )
        if float(gaps.max(axis=1).min()) <= 0.0:
            lazy_death = frame
            break

    safe_frames = [frame for frame in range(horizon + 1) if reach[frame][row0, col0]]
    if lazy_death is not None:
        safe_frames = [frame for frame in safe_frames if frame < lazy_death]
    latest = max(safe_frames) if safe_frames else None
    return {
        "method": "backward_reachability_8px",
        "cell_px": cell_px,
        "horizon": horizon,
        "latest_safe_start": latest,
        "lazy_death_frame": lazy_death,
        "safe_start_count": len(safe_frames),
        "start_cell": (row0, col0),
    }


def validate(scenario: synth.Scenario, actions: tuple[tuple[int, int], ...]) -> dict:
    """Re-simulate a candidate route exactly (never trust the search's own bookkeeping)."""

    outcome = synth.simulate(scenario, list(actions))
    min_clearance = outcome["min_clearance"]
    if min_clearance is not None and min_clearance <= 0.0:
        raise AssertionError(f"{scenario.scenario_id}: route reported safe but minimum clearance <= 0")
    return {
        "scenario_id": scenario.scenario_id,
        "frames": outcome["frames"],
        "hit": outcome["hit"],
        "hit_frame": outcome["hit_frame"],
        "survived": outcome["survived"],
        "min_clearance": min_clearance,
    }


# ---------------------------------------------------------------------------
# exhaustive cross-check
# ---------------------------------------------------------------------------
def bruteforce(scenario: synth.Scenario, decisions: int, model: geometry.MoveModel | None = None) -> dict:
    """Enumerate every action sequence of ``decisions`` steps.  Exact, exponential."""

    model = model or geometry.MoveModel()
    sequences = 0
    survivors = 0
    first_route: tuple[tuple[int, int], ...] | None = None
    for candidate in itertools.product(range(18), repeat=decisions):
        actions = [(index // 9, index % 9) for index in candidate]
        sequences += 1
        frame, x, y = 0, scenario.player_start[0], scenario.player_start[1]
        previous = (0, 0)
        alive = True
        for action in actions:
            result = synth.advance(scenario, frame, x, y, action, previous, model=model)
            frame, x, y = result.frame, result.x, result.y
            previous = action
            if result.hit:
                alive = False
                break
        if alive:
            survivors += 1
            if first_route is None:
                first_route = tuple(actions)
    return {
        "decisions": decisions,
        "sequences": sequences,
        "survivors": survivors,
        "first_route": first_route,
        "exists": survivors > 0,
    }
