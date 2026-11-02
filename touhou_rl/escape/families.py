"""Seventeen scenario families and twenty hand-designed edge instances (D2).

Purpose
-------
The acceptance matrix names SC-01 … SC-17 and E-01 … E-20.  This module turns
the names into deterministic, executable geometry:

* a family is a **function of its seed** (same seed -> byte-identical scenario,
  checked by ``synth.scenario_signature``), so the frozen seed list in
  ``preregistration/seeds-v1.json`` selects reproducible instances;
* the parameters are deliberately simple (linear walls, fans, rings, windows) so
  that a human can verify a failure by looking at the numbers;
* scenarios that need a hazard class the testbed cannot model (accelerating or
  turning bullets, lasers) are marked ``unsupported`` instead of being
  approximated into a false pass.

These generators are *not* tuned to the planner.  They are allowed to be hard.
Their pass/fail accounting belongs to D7 (A-01 … A-08); here they only have to be
well-formed, deterministic and solvable/unsolvable for a stated reason.
"""

from __future__ import annotations

import math

import numpy as np

from .contracts import HURTBOX_HALF_PX
from .synth import HazardSpec, Scenario

#: Default player start (low centre of the movement rectangle).
START = (192.0, 384.0)
#: Movement rectangle (centre coordinates).
X_MIN, X_MAX = 8.0, 376.0
Y_MIN, Y_MAX = 16.0, 432.0

FAMILY_IDS = tuple(f"SC-{index:02d}" for index in range(1, 18))


# ---------------------------------------------------------------------------
# small builders
# ---------------------------------------------------------------------------
def _wall(
    y: float,
    *,
    gap_x: float | None = None,
    gap_width: float = 24.0,
    step: float = 4.0,
    x_from: float = 4.0,
    x_to: float = 380.0,
    half: float = 2.0,
    velocity: tuple[float, float] = (0.0, 0.0),
    spawn_frame: int = 0,
) -> tuple[HazardSpec, ...]:
    """A horizontal wall with at most one gap of *exactly* ``gap_width``.

    Boxes are placed outward from the gap edges, so the free opening is the
    requested width.  Placing them on a fixed lattice and merely omitting the box
    nearest the gap would leave ``step - 2*half`` of extra room (12 px instead of
    0.65 px for the E-06 instance), which silently turns an infeasible gap into a
    passable one.
    """
    centres: list[float] = []
    if gap_x is None:
        x = x_from
        while x <= x_to + 1e-9:
            centres.append(x)
            x += step
    else:
        left_edge = gap_x - gap_width / 2.0
        right_edge = gap_x + gap_width / 2.0
        x = right_edge + half
        while x <= x_to + 1e-9:
            centres.append(x)
            x += step
        x = left_edge - half
        while x >= x_from - 1e-9:
            centres.append(x)
            x -= step
    return tuple(
        HazardSpec((float(x), y), velocity, (half, half), spawn_frame=spawn_frame)
        for x in sorted(centres)
    )


def _fan(
    origin: tuple[float, float],
    *,
    count: int,
    spread_deg: float,
    speed: float,
    half: float = 2.0,
    start_deg: float = -90.0,
) -> tuple[HazardSpec, ...]:
    hazards = []
    for index in range(count):
        fraction = 0.0 if count == 1 else index / (count - 1)
        angle = math.radians(start_deg - spread_deg / 2.0 + spread_deg * fraction)
        hazards.append(
            HazardSpec(origin, (speed * math.cos(angle), speed * math.sin(angle)), (half, half))
        )
    return tuple(hazards)


def _ring(
    centre: tuple[float, float],
    radius: float,
    count: int,
    *,
    speed: float = 0.0,
    half: float = 2.0,
    skip_angle_deg: float | None = None,
    skip_width_deg: float = 0.0,
) -> tuple[HazardSpec, ...]:
    hazards = []
    for index in range(count):
        angle_deg = 360.0 * index / count
        if skip_angle_deg is not None and abs(((angle_deg - skip_angle_deg + 180.0) % 360.0) - 180.0) <= skip_width_deg / 2.0:
            continue
        angle = math.radians(angle_deg)
        position = (centre[0] + radius * math.cos(angle), centre[1] + radius * math.sin(angle))
        velocity = (speed * math.cos(angle), speed * math.sin(angle))
        hazards.append(HazardSpec(position, velocity, (half, half)))
    return tuple(hazards)


def _vertical_wall(
    x: float,
    *,
    gap_y: float | None = None,
    gap_height: float = 24.0,
    step: float = 4.0,
    y_from: float = 12.0,
    y_to: float = 436.0,
    half: float = 2.0,
    velocity: tuple[float, float] = (0.0, 0.0),
    spawn_frame: int = 0,
) -> tuple[HazardSpec, ...]:
    centres: list[float] = []
    if gap_y is None:
        y = y_from
        while y <= y_to + 1e-9:
            centres.append(y)
            y += step
    else:
        top_edge = gap_y - gap_height / 2.0
        bottom_edge = gap_y + gap_height / 2.0
        y = bottom_edge + half
        while y <= y_to + 1e-9:
            centres.append(y)
            y += step
        y = top_edge - half
        while y >= y_from - 1e-9:
            centres.append(y)
            y -= step
    return tuple(
        HazardSpec((x, float(y)), velocity, (half, half), spawn_frame=spawn_frame)
        for y in sorted(centres)
    )


# ---------------------------------------------------------------------------
# families SC-01 … SC-17
# ---------------------------------------------------------------------------
def sc01(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    origin = (192.0 + rng.integers(-20, 21), 40.0)
    spread = 120.0 + rng.uniform(-10.0, 10.0)
    hazards = _fan(origin, count=25, spread_deg=spread, speed=2.0 + rng.uniform(0.0, 0.6))
    return Scenario(f"SC-01/{seed}", "SC-01", seed, START, hazards,
                    notes="wide fan from above; the outer sides stay reachable")


def sc02(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    gap_x = float(rng.integers(140, 261))
    gap_width = 3.0 * HURTBOX_HALF_PX + 6.0 + rng.uniform(0.0, 4.0)
    hazards = _wall(200.0, gap_x=gap_x, gap_width=gap_width)
    return Scenario(f"SC-02/{seed}", "SC-02", seed, START, hazards,
                    notes=f"single static gap at x={gap_x:.0f}, width {gap_width:.2f} px")


def sc03(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    hazards = _ring((192.0, 250.0), 90.0 + rng.uniform(0.0, 20.0), 40,
                    skip_angle_deg=90.0, skip_width_deg=26.0)
    return Scenario(f"SC-03/{seed}", "SC-03", seed, (192.0, 384.0), hazards,
                    notes="ring with an inner gap; going around the outside is not reachable")


def sc04(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    left_gap = 300.0
    right_gap = 120.0 + rng.uniform(-10.0, 10.0)
    hazards = _vertical_wall(120.0, gap_y=left_gap, gap_height=20.0,
                             velocity=(-0.6, 0.0), spawn_frame=0)
    hazards += _vertical_wall(300.0, gap_y=right_gap, gap_height=30.0)
    return Scenario(f"SC-04/{seed}", "SC-04", seed, START, hazards,
                    notes="left gap closes as its wall slides left; right gap stays open")


def sc05(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    first = _wall(260.0, gap_x=150.0, gap_width=26.0)
    second = _wall(120.0, gap_x=260.0 + rng.uniform(-20.0, 20.0), gap_width=26.0,
                   spawn_frame=90)
    return Scenario(f"SC-05/{seed}", "SC-05", seed, START, first + second,
                    notes="second wall appears at frame 90; the first wave's exit is a false-safe end")


def sc06(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    left = _vertical_wall(60.0, velocity=(0.8, 0.0))
    bottom = _wall(440.0, velocity=(0.0, -0.8))
    return Scenario(f"SC-06/{seed}", "SC-06", seed, (100.0, 360.0), left + bottom,
                    notes="corner closes from the left and the bottom")


def sc07(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    # A column is occupied for 40 frames, then vacated: waiting is the answer.
    hazards = tuple(
        HazardSpec((192.0, 300.0 - 10.0 * index), (0.0, 0.0), (6.0, 6.0),
                   spawn_frame=0, despawn_frame=40 + 2 * index)
        for index in range(6)
    )
    return Scenario(f"SC-07/{seed}", "SC-07", seed, START, hazards,
                    notes="the same cells are occupied early and free later")


def sc08(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    speed = 18.0 + rng.uniform(0.0, 6.0)
    hazards = (HazardSpec((192.0 - 60.0, 384.0), (speed, 0.0), (2.0, 2.0)),)
    return Scenario(f"SC-08/{seed}", "SC-08", seed, START, hazards,
                    notes=f"crossing at {speed:.1f} px/frame, between two observation frames")


def sc09(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    far = _wall(84.0, gap_x=210.0 + rng.uniform(-30.0, 30.0), gap_width=30.0)
    near = (HazardSpec((120.0, 300.0), (0.0, 1.0), (2.0, 2.0)),)
    return Scenario(f"SC-09/{seed}", "SC-09", seed, START, far + near,
                    notes="the closing wall is ~300 px away, outside the old 240 px near radius")


def sc10(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    wall = _vertical_wall(200.0, gap_y=200.0, gap_height=28.0, velocity=(0.0, 1.0))
    return Scenario(f"SC-10/{seed}", "SC-10", seed, START, wall,
                    notes="the gap slides down while the player evacuates")


def sc11(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    # A wide, tempting corridor on the right that closes at frame 120, and a
    # narrower but permanently open route on the left.
    tempting = _wall(180.0, gap_x=300.0, gap_width=60.0, spawn_frame=0, velocity=(0.0, 0.9))
    safe = _vertical_wall(90.0, gap_y=300.0, gap_height=26.0)
    return Scenario(f"SC-11/{seed}", "SC-11", seed, START, tempting + safe,
                    notes="the wide right corridor closes; the narrow left route stays open")


def _area_fill(frame: int, *, step: float = 8.0, half: float = 4.0) -> tuple[HazardSpec, ...]:
    """Hazards that tile the whole movement rectangle from ``frame`` on.

    Covering the rectangle is what makes "no route" *provable*: every reachable
    position is inside a box, so no action sequence survives that frame.  The
    boxes are static, so coverage persists.
    """
    hazards = []
    y = Y_MIN - 8.0
    while y <= Y_MAX + 8.0:
        x = X_MIN - 8.0
        while x <= X_MAX + 8.0:
            hazards.append(HazardSpec((x, y), (0.0, 0.0), (half, half), spawn_frame=frame))
            x += step
        y += step
    return tuple(hazards)


def sc12(seed: int) -> Scenario:
    return Scenario(f"SC-12/{seed}", "SC-12", seed, START, _area_fill(10),
                    notes="the whole movement area fills at frame 10: certified no-route")


def sc13(seed: int) -> Scenario:
    return Scenario(f"SC-13/{seed}", "SC-13", seed, START,
                    (HazardSpec((192.0, 200.0), (0.0, 2.0), (2.0, 2.0)),),
                    notes="needs accelerating/turning bullets, which this testbed does not model",
                    unsupported="accelerating/turning bullets")


def sc14(seed: int) -> Scenario:
    base = sc01(seed)
    mirrored = tuple(
        HazardSpec((448.0 - h.position[0], h.position[1]), (-h.velocity[0], h.velocity[1]), h.half)
        for h in base.hazards
    )
    return Scenario(f"SC-14/{seed}", "SC-14", seed, (448.0 - START[0], START[1]), mirrored,
                    notes="x-mirror of SC-01; decisions must follow the geometry, not a side")


def sc15(seed: int) -> Scenario:
    # Two openings in one wall: a wide one (40 px) at y=200 that leads into a
    # pocket closed at x=340, and a 20 px one at y=390 that opens into permanent
    # free space.  Both are wide enough for the 16 px coarse grid to see, so the
    # closure map can tell the dead end from the exit; a clearance-greedy planner
    # prefers the wide mouth and dies when the pressure wall reaches the pocket.
    def rows(start_y: float, stop_y: float) -> list[HazardSpec]:
        out = []
        y = start_y
        while y <= stop_y + 1e-9:
            out.append(HazardSpec((280.0, y), (0.0, 0.0), (2.0, 2.0)))
            y += 4.0
        return out

    wall = tuple(rows(16.0, 176.0) + rows(224.0, 376.0) + rows(404.0, 432.0))
    pocket = _vertical_wall(340.0, y_from=170.0, y_to=240.0, step=4.0)
    pressure = _vertical_wall(60.0, velocity=(1.5, 0.0))
    return Scenario(
        f"SC-15/{seed}",
        "SC-15",
        seed,
        START,
        wall + pocket + pressure,
        notes=(
            "wide mouth at y=200 leads into a pocket closed at x=340; the 20 px "
            "mouth at y=390 opens into free space"
        ),
    )


def sc16(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    close_frame = 90 + int(rng.integers(0, 91))  # 90..180
    gap = 40.0
    gap_centre = 192.0
    speed = gap / 2.0 / close_frame
    wall = _wall(300.0, gap_x=gap_centre, gap_width=gap)
    # Two shutters slide inward and close the gap at ``close_frame``.  A sweeping
    # full-width wall (the first version of this family) does not model a closing
    # exit at all: it can be outrun and it never removes an option.
    left = HazardSpec((gap_centre - gap, 300.0), (speed, 0.0), (gap / 2.0, 2.0))
    right = HazardSpec((gap_centre + gap, 300.0), (-speed, 0.0), (gap / 2.0, 2.0))
    # Pressure source: without it, standing still below the wall survives and the
    # family cannot distinguish "used the gap in time" from "did nothing".
    floor = (
        _wall(470.0, velocity=(0.0, -1.0))
        + _wall(466.0, velocity=(0.0, -1.0))
        + _wall(462.0, velocity=(0.0, -1.0))
    )
    return Scenario(f"SC-16/{seed}", "SC-16", seed, START, wall + (left, right) + floor,
                    notes=(
                        f"locally empty; the only gap closes at frame {close_frame}; "
                        "a rising floor reaches the player's row near frame 86"
                    ))


def sc17(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    hazards = tuple(
        HazardSpec((80.0 + 60.0 * index + rng.uniform(-8.0, 8.0), 200.0 + rng.uniform(-40.0, 40.0)),
                   (0.4 * rng.uniform(-1.0, 1.0), 1.2), (2.0, 2.0))
        for index in range(4)
    )
    return Scenario(f"SC-17/{seed}", "SC-17", seed, START, hazards,
                    notes="isolated sparse bullets: local skill must stay intact")


GENERATORS = {
    "SC-01": sc01, "SC-02": sc02, "SC-03": sc03, "SC-04": sc04, "SC-05": sc05,
    "SC-06": sc06, "SC-07": sc07, "SC-08": sc08, "SC-09": sc09, "SC-10": sc10,
    "SC-11": sc11, "SC-12": sc12, "SC-13": sc13, "SC-14": sc14, "SC-15": sc15,
    "SC-16": sc16, "SC-17": sc17,
}


def build(family: str, seed: int) -> Scenario:
    if family not in GENERATORS:
        raise KeyError(f"unknown family {family}; expected {FAMILY_IDS}")
    return GENERATORS[family](int(seed))


# ---------------------------------------------------------------------------
# edge instances E-01 … E-20
# ---------------------------------------------------------------------------
def _edge(scenario_id: str, family: str, hazards, *, start=START, horizon=20,
          expected: str, notes: str, unsupported: str | None = None) -> Scenario:
    return Scenario(scenario_id, family, 0, start, tuple(hazards),
                    horizon_frames=horizon, hold_frames=0,
                    notes=notes, expected=expected, unsupported=unsupported)


def mirror_x(scenario: Scenario, *, axis: float | None = None) -> Scenario:
    """Reflect a scenario across the vertical centre line (SC-14 invariant test).

    The movement rectangle is symmetric about ``x = 192`` (8 and 376), so a
    mirrored scenario must produce a mirrored trajectory.  This is how SC-14's
    "behaviour follows the geometry, not a memorised left/right" is checked.
    """

    axis = (X_MIN + X_MAX) / 2.0 if axis is None else float(axis)
    hazards = tuple(
        HazardSpec(
            position=(2.0 * axis - hazard.position[0], hazard.position[1]),
            velocity=(-hazard.velocity[0], hazard.velocity[1]),
            half=(hazard.half[0], hazard.half[1]),
            spawn_frame=hazard.spawn_frame,
            despawn_frame=hazard.despawn_frame,
        )
        for hazard in scenario.hazards
    )
    start = (2.0 * axis - scenario.player_start[0], scenario.player_start[1])
    return Scenario(
        f"{scenario.scenario_id}/mirror",
        scenario.family,
        scenario.seed,
        start,
        hazards,
        horizon_frames=scenario.horizon_frames,
        hold_frames=scenario.hold_frames,
        notes=f"x-mirror of {scenario.scenario_id}",
        expected=scenario.expected,
        unsupported=scenario.unsupported,
    )


def edge_instances() -> dict[str, Scenario]:
    """The twenty boundary instances, keyed by id."""

    wide_gap = 2.0 * HURTBOX_HALF_PX + 3.0
    narrow_gap = 2.0 * HURTBOX_HALF_PX - 1.0
    # A rising floor forces the player through the gap before it arrives.  It is
    # three rows thick on purpose: with one thin row the discrete per-frame check
    # can be "surfed" between two frames (measured: a 0.005 px clearance route in
    # the first version of this instance), which would make E-06 feasible for a
    # reason unrelated to the gap width.
    rising = _wall(470.0, velocity=(0.0, -3.0)) + _wall(466.0, velocity=(0.0, -3.0)) + _wall(
        462.0, velocity=(0.0, -3.0)
    )

    instances = {
        "E-01": _edge("E-01", "SC-08",
                      [HazardSpec((163.0, 384.0), (10.0, 0.0), (2.0, 2.0))],
                      expected="feasible",
                      notes="crosses the hurtbox at t=3, between the t=2 and t=4 samples"),
        "E-02": _edge("E-02", "SC-08",
                      [HazardSpec((163.0, 345.0), (10.0, 10.0), (2.0, 2.0))],
                      expected="feasible",
                      notes="diagonal crossing through a hurtbox corner"),
        "E-03": _edge("E-03", "SC-07",
                      [HazardSpec((192.0, 300.0), (0.0, 0.0), (6.0, 6.0), despawn_frame=30),
                       HazardSpec((192.0, 300.0), (0.0, 0.0), (6.0, 6.0), spawn_frame=60)],
                      horizon=80, expected="feasible",
                      notes="same cell occupied [0,30) and [60,80): a window exists"),
        "E-04": _edge("E-04", "SC-07",
                      [HazardSpec((192.0, 300.0), (0.0, 0.0), (6.0, 6.0))],
                      expected="feasible",
                      notes="occupancy never returns, so a permanent block is correct here"),
        "E-05": _edge("E-05", "SC-02",
                      _wall(300.0, gap_x=192.0, gap_width=wide_gap) + rising,
                      horizon=60, expected="feasible",
                      notes=f"gap {wide_gap:.2f} px = hurtbox + 3 px, reachable before the floor arrives"),
        "E-06": _edge("E-06", "SC-02",
                      _wall(300.0, gap_x=192.0, gap_width=narrow_gap) + rising,
                      horizon=60, expected="infeasible_unproven",
                      notes=f"gap {narrow_gap:.2f} px = hurtbox - 1 px: no route, and a bounded "
                            "search must not claim one"),
        "E-07": _edge("E-07", "SC-06",
                      _vertical_wall(40.0, velocity=(1.0, 0.0)),
                      start=(60.0, 360.0), horizon=30, expected="feasible",
                      notes="player starts in the corner and must leave before the wall arrives"),
        "E-08": _edge("E-08", "SC-06",
                      [HazardSpec((400.0, 460.0), (0.0, 0.0), (2.0, 2.0))],
                      start=(376.0, 432.0), horizon=20, expected="feasible",
                      notes="centre exactly on the clamp boundary; the hazard is outside the move area"),
        "E-09": _edge("E-09", "SC-04",
                      _vertical_wall(120.0, gap_y=300.0, gap_height=20.0, velocity=(-0.7, 0.0))
                      + _vertical_wall(300.0, gap_y=150.0, gap_height=24.0),
                      horizon=60, expected="feasible",
                      notes="the nearer exit closes first; the farther one stays open"),
        "E-10": _edge("E-10", "SC-04",
                      _vertical_wall(120.0, gap_y=300.0, gap_height=24.0)
                      + _vertical_wall(300.0, gap_y=300.0, gap_height=24.0),
                      horizon=20, expected="feasible",
                      notes="both exits are equivalent: a tie that must be broken explicitly"),
        "E-11": _edge("E-11", "SC-12", _area_fill(6),
                      horizon=20, expected="blocked",
                      notes="the whole area is covered from frame 6: a coverage proof, not a timeout"),
        "E-12": _edge("E-12", "SC-01",
                      _fan((192.0, 60.0), count=3, spread_deg=20.0, speed=1.0),
                      horizon=60, expected="unknown",
                      notes="open field with a tiny budget: must report budget_exhausted, not blocked"),
        "E-13": _edge("E-13", "SC-16",
                      _wall(384.0, spawn_frame=180),
                      horizon=180, expected="feasible",
                      notes="a full-width row appears at frame 180: the player must already be elsewhere"),
        "E-14": _edge("E-14", "SC-16",
                      _wall(384.0, spawn_frame=181),
                      horizon=180, expected="feasible",
                      notes="the same row appears at frame 181, outside the horizon: feasible, but the "
                            "reason is the horizon, not safety"),
        "E-15": _edge("E-15", "SC-15",
                      _wall(120.0, gap_x=192.0, gap_width=60.0)
                      + _wall(60.0, gap_x=192.0, gap_width=8.0)
                      + _vertical_wall(320.0, gap_y=384.0, gap_height=10.0),
                      horizon=60, expected="feasible",
                      notes="the widest local clearance points at a closed pocket"),
        "E-16": _edge("E-16", "SC-15",
                      _wall(120.0, gap_x=192.0, gap_width=2.0 * HURTBOX_HALF_PX + 1.2),
                      horizon=40, expected="unknown",
                      notes="a 1.2 px-clearance gap: only an exact route through it counts, otherwise unknown"),
        "E-17": _edge("E-17", "SC-17",
                      [HazardSpec((150.0, 250.0), (0.0, 1.0), (2.0, 2.0))],
                      horizon=40, expected="feasible",
                      notes="one slow bullet: local behaviour only"),
        "E-18": _edge("E-18", "SC-17",
                      [HazardSpec((100.0, 250.0), (1.0, 0.0), (2.0, 2.0)),
                       HazardSpec((300.0, 250.0), (-1.0, 0.0), (2.0, 2.0))],
                      horizon=40, expected="feasible",
                      notes="two bullets crossing far from the player"),
        "E-19": _edge("E-19", "SC-13",
                      [HazardSpec((192.0, 250.0), (0.0, 2.0), (2.0, 2.0))],
                      horizon=40, expected="feasible",
                      notes="linear-model control: the hazard is exactly linear"),
        "E-20": _edge("E-20", "SC-14",
                      _fan((192.0, 60.0), count=9, spread_deg=60.0, speed=1.5),
                      horizon=60, expected="feasible",
                      notes="base instance of an x-mirror pair; geometry must decide the side"),
        # E-21/E-22: two *provably* blocked instances with different timing.  Both
        # are a full-area cover that appears once and stays, so the reference's
        # coverage proof applies (static hazards, alive through the horizon) and
        # the label is a proof rather than an assertion.  They extend the A-05
        # negative set to four instances with two distinct failure modes.
        # E-21/E-22: two *provably* blocked instances whose cover appears **after**
        # the planner's 24-frame horizon.  The reference proves them
        # (``area_covered`` from frame 40/120), but a 24-frame plan cannot know:
        # the honest planner answer is ``model_unknown``/``budget_exhausted``, not
        # "certified".  They still belong in A-05 (must not survive) and are the
        # reason the label is ``blocked_late`` rather than ``blocked``.
        "E-21": _edge("E-21", "SC-12",
                      _area_fill(40, step=12.0, half=7.0),
                      horizon=80, expected="blocked_late",
                      notes="second wave: the whole area is covered from frame 40 and stays covered"),
        "E-22": _edge("E-22", "SC-12",
                      _area_fill(120, step=12.0, half=7.0),
                      horizon=140, expected="blocked_late",
                      notes="late cover: nothing happens for 120 frames, then the area closes for good"),
    }
    return instances


def edge_ids() -> tuple[str, ...]:
    return tuple(f"E-{index:02d}" for index in range(1, 23))
