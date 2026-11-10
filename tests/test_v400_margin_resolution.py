"""EXP-P3-005: the ``v2i`` margin-resolution profile and the clamp it runs into.

The task the profile was registered for is narrow: EXP-P3-004 §7.3 (P0) asked for
**geometric resolution**, because 82 % of the decisions have an 18-way top1-top2
*geometric* gap below 0.06 px and the shipped margin term saturates at 16 px.
``v2i`` therefore raises ``Weights.margin_cap`` 16 -> 64 and changes nothing else.

These tests pin four things:

1. ``v2i`` **is** ``v2h`` with one number changed (and ``FAST`` / ``V2A`` / ``V2H``
   are still their published selves — bit for bit, field for field);
2. the margin term itself is monotone and saturating, so raising the cap *would*
   change a score if the score ever saw an uncapped margin;
3. it never does: the search clamps the per-frame clearance at the module
   constant :data:`MARGIN_CAP` (16 px) **before** ``Weights.margin_cap`` is
   applied, so ``v2h`` and ``v2i`` produce identical ``geo_score`` on a real tape
   whose true clearance is ~41 px — the measurement EXP-P3-005 §3 reports;
4. the raster table's byte budget caps any *real* raise at 63.75 px
   (``MARGIN_Q_CAP = 64`` at 1/4 px in the low byte of a ``uint16``), i.e. the
   requested 64.0 is one quantisation step past what the table can store.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from touhou_rl.v400.gym.protocol import BulletRow, PlayerSnapshot
from touhou_rl.v400.teacher.config import (
    CONFIGS,
    FAST,
    MARGIN_CAP,
    PROFILES,
    V2A,
    V2H,
    V2I,
    Weights,
    get_config,
)
from touhou_rl.v400.teacher.hazard_tape import HazardTape
from touhou_rl.v400.teacher.motion import SpeedTable
from touhou_rl.v400.teacher.raster import MARGIN_Q_CAP, QUANT
from touhou_rl.v400.teacher.search import (
    beam_search,
    beam_search_reference,
    score_plan,
    simulate_plan,
)

#: team 0's measured speeds (EXP-P1-001 §3), the same table the teacher tests use.
SPEEDS = SpeedTable(4.0, 2.828427, 2.0, 1.414214)

#: The true clearance of the test wall: bullets at ``y = 428`` with a 4 px collision
#: size put their top edge at 426.0, and the ship's hurtbox bottom is at
#: ``384 + 0.825 = 384.825`` -> ``41.175 px``.  Well past :data:`MARGIN_CAP`.
WALL_Y = 428.0
WALL_CLEARANCE = (WALL_Y - 2.0) - (384.0 + 0.825)


def bullet(x: float, y: float, size: float = 4.0) -> BulletRow:
    return BulletRow(
        slot=0,
        state=1,
        color=0,
        position=(x, y, 0.0),
        velocity=(0.0, 0.0),
        timer=10,
        collision_size=size,
        collision_disabled=0,
    )


def player_row(x: float = 192.0, y: float = 384.0, half: float = 0.825) -> PlayerSnapshot:
    return PlayerSnapshot(
        frame=0,
        hp=2.0,
        lives=2,
        bombs=3.0,
        power=0.0,
        focus=0,
        is_youkai=0,
        player_state=0,
        invuln_state=0,
        position=(x, y),
        hurtbox_min=(x - half, y - half),
        hurtbox_max=(x + half, y + half),
        speed_normal_axial=SPEEDS.normal_axial,
        speed_normal_diagonal=SPEEDS.normal_diagonal,
        speed_focused_axial=SPEEDS.focused_axial,
        speed_focused_diagonal=SPEEDS.focused_diagonal,
    )


def make_tape(horizon: int = 16, *, bullets: tuple[BulletRow, ...] | None = None) -> HazardTape:
    """A stationary tape whose only hazard is a wall ``WALL_CLEARANCE`` px below."""

    wall = bullets if bullets is not None else (bullet(192.0, WALL_Y),)
    row = player_row()
    player = {
        "px": np.full(horizon, row.position[0]),
        "py": np.full(horizon, row.position[1]),
        "min_x": np.full(horizon, row.hurtbox_min[0]),
        "min_y": np.full(horizon, row.hurtbox_min[1]),
        "max_x": np.full(horizon, row.hurtbox_max[0]),
        "max_y": np.full(horizon, row.hurtbox_max[1]),
        "state": np.zeros(horizon),
        "invuln": np.zeros(horizon),
        "focus": np.zeros(horizon),
        "bombs": np.full(horizon, 3.0),
        "lives": np.full(horizon, 2.0),
        "power": np.zeros(horizon),
    }
    return HazardTape(
        horizon=horizon,
        decision_frames=2,
        frames=np.arange(100, 100 + horizon, dtype=np.int64),
        hashes=tuple("0" * 16 for _ in range(horizon)),
        player=player,
        speeds=SPEEDS,
        bullets=tuple(tuple(wall) for _ in range(horizon)),
        lasers=tuple(() for _ in range(horizon)),
        enemies=tuple(() for _ in range(horizon)),
        masks=tuple(0 for _ in range(horizon)),
        actions=tuple(0 for _ in range(horizon)),
        meta={"source": "test-v400-margin"},
    )


# ---------------------------------------------------------------------------
# §1 the profile is one number away from v2h and nothing else moved
# ---------------------------------------------------------------------------
def test_v2i_is_v2h_with_the_margin_cap_raised_and_nothing_else():
    """``replace(v2i, mode="v2h", margin_cap=16) == v2h`` — the whole definition."""

    reverted = dataclasses.replace(
        V2I, mode="v2h", weights=dataclasses.replace(V2I.weights, margin_cap=MARGIN_CAP))
    assert reverted == V2H
    assert V2I.mode == "v2i"
    assert V2I.weights.margin_cap == 64.0
    assert V2H.weights.margin_cap == MARGIN_CAP == 16.0
    # the task's list, item by item: bottom 1.5 / 2.0, high 3.0, laser 12 / band 8,
    # rule-1 12 frames, horizon 48, beam 64, verify_k 3, max_iterations 3.
    assert (V2I.horizon, V2I.beam, V2I.verify_k, V2I.max_iterations) == (48, 64, 3, 3)
    assert V2I.bomb_rule1_max_death_frame == 12
    assert V2I.durations == (1, 2)
    assert V2I.weights.bottom_penalty == 1.5
    assert V2I.weights.bottom_extra_penalty == 2.0
    assert V2I.weights.high_penalty == 3.0
    assert V2I.weights.laser_penalty == 12.0
    assert V2I.weights.laser_band == 8.0
    # the discount is untouched: EXP-P3-004 measured it is not the problem, and
    # v2i must not change two things at once.
    assert V2I.weights.unverified_discount == V2H.weights.unverified_discount == 5.0


def test_v2i_is_registered_and_selectable_by_both_clis():
    assert CONFIGS["v2i"] is V2I
    assert V2I in PROFILES
    assert set(CONFIGS) >= {"v2i", "v2h", "fast", "fast3", "full"}

    script = "tools/rl/v400_d0_dual.py"
    parsed = _argparse_choices(script, "--mode")
    assert "v2i" in parsed and "v2h" in parsed
    teacher = _argparse_choices("tools/rl/v400_teacher.py", "--mode")
    assert "v2i" in teacher


def _argparse_choices(relative: str, option: str) -> list[str]:
    """The ``choices=`` list of one CLI flag, read without importing ``__main__``."""

    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / relative).read_text(encoding="utf-8")
    match = re.search(
        r'"' + re.escape(option) + r'",\s*\n?\s*choices=([^,\n]+)', text)
    assert match, f"{relative}: no {option} choices found"
    expression = match.group(1).strip()
    if expression.startswith("tuple(sorted(") or expression.startswith("tuple("):
        return sorted(CONFIGS)
    return [part.strip().strip('"') for part in expression.strip("()").split(",") if part.strip()]


def test_fast_v2a_and_v2h_keep_their_published_numbers():
    """The three shipped profiles the task protects, literally."""

    assert (FAST.horizon, FAST.beam, FAST.verify_k, FAST.max_iterations) == (32, 24, 1, 1)
    assert (V2A.horizon, V2A.beam, V2A.verify_k, V2A.max_iterations) == (48, 64, 3, 3)
    assert (V2A.weights.bottom_penalty, V2A.weights.bottom_extra_penalty) == (1.5, 2.0)
    assert V2A.weights.margin_cap == MARGIN_CAP
    assert V2H.weights.laser_penalty == 12.0 and V2H.weights.laser_band == 8.0
    assert dataclasses.replace(V2H.weights, laser_penalty=0.0) == V2A.weights
    for profile in (FAST, V2A, V2H, V2I):
        assert profile.weights.unverified_discount == 5.0
    assert get_config("v2i") is V2I
    # EXP-P3-007 registers ``v2j`` (the duration-set profile), so the
    # "not registered" guard moves one name further out.  Semantic-preserving:
    # the assertion still says "a name past the newest experiment is unknown".
    with pytest.raises(ValueError):
        get_config("v2k")


# ---------------------------------------------------------------------------
# §2 the weight itself is live: monotone, saturating, and 4x the range
# ---------------------------------------------------------------------------
def _plan_score(margin: float, config) -> float:
    total, parts = score_plan(
        margin=margin, death_frame=None, end_xy=(192.0, 384.0), target=192.0,
        horizon=config.horizon, config=config)
    assert parts["margin"] == min(float(margin), config.weights.margin_cap)
    return total


def test_the_margin_term_is_monotone_and_saturating_under_each_cap():
    grid = [0.0, 0.5, 4.0, 15.999, 16.0, 24.0, 40.0, 63.0, 64.0, 200.0]
    for config, cap in ((V2H, 16.0), (V2I, 64.0)):
        scores = [_plan_score(margin, config) for margin in grid]
        assert scores == sorted(scores), f"{config.mode}: the margin term is not monotone"
        # strictly increasing until the cap, exactly flat afterwards
        for left, right in zip(grid, grid[1:]):
            step = _plan_score(right, config) - _plan_score(left, config)
            expected = min(max(right - left, 0.0), max(cap - left, 0.0))
            assert step == pytest.approx(expected, abs=1e-9)


def test_raising_the_cap_changes_a_score_that_sees_an_uncapped_margin():
    """48 points of headroom appear, *if* the caller hands over the real margin."""

    assert _plan_score(WALL_CLEARANCE, V2I) - _plan_score(WALL_CLEARANCE, V2H) == pytest.approx(
        WALL_CLEARANCE - MARGIN_CAP, abs=1e-9)
    # below the old cap the two profiles are the same function
    assert _plan_score(3.0, V2I) == pytest.approx(_plan_score(3.0, V2H), abs=1e-12)
    with pytest.raises(TypeError):
        score_plan(margin=1.0)  # keyword-only contract


# ---------------------------------------------------------------------------
# §3 ... but the search clamps the margin at 16 px before the weight sees it
# ---------------------------------------------------------------------------
def test_simulate_plan_reports_a_capped_margin_for_a_real_41_px_clearance():
    tape = make_tape()
    margin, _end, death, margins = simulate_plan(tape, [0, 0], config=V2I)
    assert death is None
    assert WALL_CLEARANCE == pytest.approx(41.175)
    assert margin == pytest.approx(MARGIN_CAP)          # <-- the clamp
    assert all(value <= MARGIN_CAP for value in margins)
    assert margin < WALL_CLEARANCE


def test_v2i_and_v2h_score_every_candidate_identically_on_a_real_tape():
    """The measurement the card publishes: raising the cap moves no label."""

    tape = make_tape()
    for search in (beam_search_reference, beam_search):
        reference = search(tape, V2H, target=192.0)
        candidate = search(tape, V2I, target=192.0)
        np.testing.assert_array_equal(candidate.scores, reference.scores)
        assert candidate.target_x == reference.target_x
        assert candidate.plans.keys() == reference.plans.keys()
        # non-degenerate: all 18 candidates were scored, so the equality above is
        # an 18-way comparison and not an empty loop.
        assert len(reference.scores) == 18
        assert len(reference.plans) == 18
        for first, plan in reference.plans.items():
            assert candidate.plans[first].score == plan.score
            assert candidate.plans[first].margin == plan.margin == MARGIN_CAP


# ---------------------------------------------------------------------------
# §4 a *real* raise cannot reach 64: the raster table stores 1/4 px in one byte
# ---------------------------------------------------------------------------
def test_the_raster_byte_budget_caps_a_real_raise_at_63_75_px():
    assert QUANT == 4
    assert MARGIN_Q_CAP == int(round(MARGIN_CAP * QUANT)) == 64
    representable_max = 255 / QUANT
    assert representable_max == 63.75
    # v2i's requested cap is one quantisation step past the last storable value...
    assert V2I.weights.margin_cap > representable_max
    # ...so any *real* raise has to stop at 63.75 px and thread through
    # raster.MARGIN_Q_CAP / search.MARGIN_CAP, not only through Weights.
    assert int(V2I.weights.margin_cap * QUANT) == 256 > 255
