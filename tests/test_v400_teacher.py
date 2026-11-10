"""Contract tests for the P1 search teacher (11-P1 §2-§5).

No engine is started: the tape is assembled from real protocol row types, so the
geometry, the motion model, the scoring formula and the beam search are all
exercised as pure functions.  The engine-shaped tests use fake processes and
check the two fail-closed contracts (a refused ``HAZARDS``, and a refused
``CHILD HAZARDS``) plus the batched-verification equivalence.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import struct
from pathlib import Path

import numpy as np
import pytest

from touhou_rl.v400.gym.actions import cand_to_mask
from touhou_rl.v400.teacher import geometry

from touhou_rl.v400.gym.protocol import (
    BulletRow,
    EnemyRow,
    LaserRow,
    PlayerSnapshot,
    parse_events_reply,
    parse_laser_row,
)

from touhou_rl.v400.teacher.child_record import record_child_tape
from touhou_rl.v400.teacher.config import (
    CONFIGS,
    FAST,
    FULL,
    MARGIN_CAP,
    N_CAND,
    PROFILES,
    get_config,
)
from touhou_rl.v400.teacher.decide import (
    death_bomb_feasible,
    death_bomb_window,
    decide,
    restore,
)
from touhou_rl.v400.teacher.engine import (
    EXPECTED_ENGINE_SHA256,
    EngineError,
    EngineShaMismatch,
    check_engine_binary,
    decode_state_fields,
    parse_fields,
)
from touhou_rl.v400.teacher.hazard_tape import (
    ChildHazardsUnavailable,
    HazardExportUnavailable,
    HazardTape,
    record,
)
from touhou_rl.v400.teacher.label import TeacherLabel
from touhou_rl.v400.teacher.motion import SpeedTable, step_delta
from touhou_rl.v400.teacher.pool import RecorderPool
from touhou_rl.v400.teacher.raster import (
    CELL,
    build_table,
    flat_index,
    lattice,
)
from touhou_rl.v400.teacher.search import (
    beam_search,
    beam_search_reference,
    score_plan,
    simulate_plan,
)
from touhou_rl.v400.teacher.verify import (
    Consistency,
    VerifiedPlan,
    select_plans,
    verify_plans,
)

ROOT = Path(__file__).resolve().parents[1]

#: team 0's measured speeds (EXP-P1-001 §3), reported here only by the fake tape.
SPEEDS = SpeedTable(4.0, 2.828427, 2.0, 1.414214)


def bullet(x, y, size=4.0, state=1, disabled=0, slot=0):
    return BulletRow(
        slot=slot,
        state=state,
        color=0,
        position=(x, y, 0.0),
        velocity=(0.0, 0.0),
        timer=10,
        collision_size=size,
        collision_disabled=disabled,
    )


def enemy(x, y, hitbox=24.0, flags1=1 << 2, slot=0):
    return EnemyRow(
        slot=slot,
        flags1=flags1,
        position=(x, y, 0.0),
        life=100,
        timer=5,
        world_position=(x, y, 0.0),
        hitbox_width=hitbox,
        hitbox_height=hitbox,
    )


def player_row(x=192.0, y=384.0, half=0.825, bombs=3.0, state=0):
    return PlayerSnapshot(
        frame=0,
        hp=2.0,
        lives=2,
        bombs=bombs,
        power=0.0,
        focus=0,
        is_youkai=0,
        player_state=state,
        invuln_state=0,
        position=(x, y),
        hurtbox_min=(x - half, y - half),
        hurtbox_max=(x + half, y + half),
        speed_normal_axial=SPEEDS.normal_axial,
        speed_normal_diagonal=SPEEDS.normal_diagonal,
        speed_focused_axial=SPEEDS.focused_axial,
        speed_focused_diagonal=SPEEDS.focused_diagonal,
    )


def make_tape(frames, *, speeds=SPEEDS, hurtbox=(192.0, 384.0, 0.825), start_frame=100):
    """Build a tape from ``frames`` = list of (bullets, enemies) per frame."""
    horizon = len(frames)
    x, y, half = hurtbox
    player = {
        "px": np.full(horizon, x),
        "py": np.full(horizon, y),
        "min_x": np.full(horizon, x - half),
        "min_y": np.full(horizon, y - half),
        "max_x": np.full(horizon, x + half),
        "max_y": np.full(horizon, y + half),
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
        frames=np.arange(start_frame, start_frame + horizon, dtype=np.int64),
        hashes=tuple("0" * 16 for _ in range(horizon)),
        player=player,
        speeds=speeds,
        bullets=tuple(tuple(frame[0]) for frame in frames),
        lasers=tuple(() for _ in range(horizon)),
        enemies=tuple(tuple(frame[1]) for frame in frames),
        masks=tuple(0 for _ in range(horizon)),
        actions=tuple(0 for _ in range(horizon)),
        meta={"source": "test"},
    )


def empty_frames(count):
    return [((), ()) for _ in range(count)]


# ---------------------------------------------------------------------------
# §2.1 geometry
# ---------------------------------------------------------------------------
def test_collision_size_is_a_full_width():
    """A 4.0-wide bullet kills at ``position ± 2``, not ``position ± 4``."""
    tape = make_tape([((bullet(192.0, 380.0),), ()) for _ in range(2)])
    mins, maxs = tape.boxes(0)
    assert mins[0].tolist() == pytest.approx([190.0, 378.0])
    assert maxs[0].tolist() == pytest.approx([194.0, 382.0])

    # The ship's hurtbox top is 383.175, so a bullet whose bottom edge is 382.0
    # is 1.175 px away: alive.  A legacy half-width reading would have made the
    # bullet 4 px half-size and killed this position.
    margin, _end, death, _ = simulate_plan(tape, [0, 0], config=FAST)
    assert death is None
    assert margin == pytest.approx(1.175, abs=1e-6)


def test_touching_counts_as_a_hit():
    """11-P1 §2.1: the comparison includes equality."""
    # Exactly representable numbers on both sides, so ``==`` is a real equality
    # test rather than a float32-rounding test: bullet bottom edge 383.0 == the
    # hurtbox's top edge 383.0.
    tape = make_tape([((bullet(192.0, 381.0),), ()) for _ in range(2)], hurtbox=(192.0, 383.825, 0.825))
    _margin, _end, death, margins = simulate_plan(tape, [0, 0], config=FAST)
    assert margins[0] == pytest.approx(0.0)
    assert death == 0


def test_enemy_body_uses_world_position_and_third_extent():
    tape = make_tape([((), (enemy(192.0, 376.0),)) for _ in range(2)])
    mins, maxs = tape.boxes(0)
    assert mins[0].tolist() == pytest.approx([184.0, 368.0])  # 24 / 3 = 8 half extent
    assert maxs[0].tolist() == pytest.approx([200.0, 384.0])


def test_enemy_without_collision_flag_is_not_lethal():
    tape = make_tape([((), (enemy(192.0, 384.0, flags1=0),)) for _ in range(2)])
    mins, _maxs = tape.boxes(0)
    assert mins.shape[0] == 0


def test_non_fired_or_disabled_bullets_are_not_lethal():
    tape = make_tape(
        [((bullet(192.0, 384.0, state=6),), ()), ((bullet(192.0, 384.0, disabled=1),), ())]
    )
    assert tape.boxes(0)[0].shape[0] == 0
    assert tape.boxes(1)[0].shape[0] == 0


# ---------------------------------------------------------------------------
# §2.2 motion model
# ---------------------------------------------------------------------------
def test_speeds_come_from_the_table_not_from_constants():
    slow = SpeedTable(1.0, 0.7071, 0.5, 0.35355)
    assert step_delta(3, slow) == pytest.approx((1.0, 0.0))  # east, unfocused
    assert step_delta(16, slow) == pytest.approx((-0.5, 0.0))  # west, focused
    assert step_delta(2, slow) == pytest.approx((0.7071, -0.7071))  # NE diagonal
    assert step_delta(0, slow) == (0.0, 0.0)
    assert step_delta(9, slow) == (0.0, 0.0)


def test_diagonal_magnitude_equals_axial_speed():
    dx, dy = step_delta(2, SPEEDS)  # NE unfocused
    assert (dx * dx + dy * dy) ** 0.5 == pytest.approx(SPEEDS.normal_axial, rel=1e-5)


def test_playfield_bounds_are_clamped():
    tape = make_tape(empty_frames(8), hurtbox=(10.0, 20.0, 0.825))
    # 8 slots of north-west, 2 frames each: far beyond the 8/16 borders.
    margin, end, death, _ = simulate_plan(tape, [8] * 8, config=FAST)
    assert death is None
    assert end[0] == pytest.approx(8.0)
    assert end[1] == pytest.approx(16.0)


def test_one_frame_input_latency():
    """The mask written in frame t first moves the ship into frame t+1."""
    tape = make_tape(empty_frames(4))
    _m, end, _d, _ = simulate_plan(tape, [3], config=FAST)  # one slot east
    assert end[0] == pytest.approx(192.0 + 2 * SPEEDS.normal_axial)
    # Frame 0 is untouched: a hazard sitting on the start point kills at frame 0
    # even when the plan moves away immediately.
    deadly = make_tape([((bullet(192.0, 384.0),), ()) for _ in range(4)])
    _m, _e, death, _ = simulate_plan(deadly, [3, 3], config=FAST)
    assert death == 0


# ---------------------------------------------------------------------------
# §3 scoring, term by term
# ---------------------------------------------------------------------------
def test_score_terms():
    weights = FULL.weights
    base = score_plan(
        margin=10.0, death_frame=None, end_xy=(192.0, 300.0), target=192.0, horizon=48, config=FULL
    )[0]
    assert base == pytest.approx(weights.base + 10.0 + weights.align_weight * 1.0)
    capped = score_plan(
        margin=999.0, death_frame=None, end_xy=(192.0, 300.0), target=192.0, horizon=48, config=FULL
    )[0]
    assert capped == pytest.approx(weights.base + MARGIN_CAP + weights.align_weight)
    far = score_plan(
        margin=0.0, death_frame=None, end_xy=(192.0, 300.0), target=192.0 + 96.0, horizon=48, config=FULL
    )[0]
    assert far == pytest.approx(weights.base + 0.0 + weights.align_weight * 0.5)
    edge = score_plan(
        margin=0.0, death_frame=None, end_xy=(20.0, 300.0), target=20.0, horizon=48, config=FULL
    )[0]
    assert edge == pytest.approx(weights.base + weights.align_weight - weights.edge_penalty)
    high = score_plan(
        margin=0.0, death_frame=None, end_xy=(192.0, 150.0), target=192.0, horizon=48, config=FULL
    )[0]
    assert high == pytest.approx(weights.base + weights.align_weight - weights.high_penalty)


def test_dead_plan_scores_death_frame_over_horizon():
    score = score_plan(
        margin=0.0, death_frame=24, end_xy=(192.0, 384.0), target=192.0, horizon=48, config=FULL
    )[0]
    assert score == pytest.approx(24 / 48 * 10.0)
    # 11-P1 §3: a dead plan must score below every living one.
    alive = score_plan(
        margin=0.0, death_frame=None, end_xy=(192.0, 384.0), target=192.0, horizon=48, config=FULL
    )[0]
    assert score < alive


# ---------------------------------------------------------------------------
# beam search vs brute force
# ---------------------------------------------------------------------------
def test_beam_search_matches_brute_force_on_a_small_tape():
    """With ``beam >= 18**2`` the beam cannot prune, so it must equal brute force."""
    # A wall of slow bullets slightly north of the ship: moving north is fatal,
    # the other directions are free and score differently through ``align``.
    frames = []
    for _ in range(6):
        frames.append(((bullet(180.0, 370.0, size=6.0), bullet(204.0, 370.0, size=6.0)), ()))
    tape = make_tape(frames)
    config = get_config("full")
    config = type(config)(
        mode="test",
        horizon=6,
        beam=400,
        verify_k=1,
        max_iterations=1,
        durations=(1, 2),
        replan_decisions=2,
        decision_frames=2,
        weights=config.weights,
    )
    result = beam_search(tape, config, use_raster=False)
    # Brute force: every plan of 3 slots.
    best = {cand: -1e9 for cand in range(N_CAND)}
    for first in range(N_CAND):
        for rest in itertools.product(range(N_CAND), repeat=2):
            slots = [first, *rest]
            margin, end, death, _ = simulate_plan(tape, slots, config=config)
            score, _parts = score_plan(
                margin=margin,
                death_frame=death,
                end_xy=(float(end[0]), float(end[1])),
                target=result.target_x,
                horizon=config.horizon,
                config=config,
            )
            best[first] = max(best[first], score)
    for first in range(N_CAND):
        assert result.scores[first] == pytest.approx(best[first], abs=1e-9)


def test_beam_search_reports_every_first_action_and_survives_a_wall():
    frames = [((bullet(192.0, 380.0, size=200.0),), ()) for _ in range(8)]
    tape = make_tape(frames)
    config = type(FULL)(
        mode="test", horizon=8, beam=8, verify_k=1, max_iterations=1, durations=(1,)
    )
    result = beam_search(tape, config)
    assert result.scores.shape == (N_CAND,)
    assert set(result.plans) == set(range(N_CAND))
    # A bullet that spans the whole lane kills every plan.
    assert all(plan.death_frame is not None for plan in result.plans.values())


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def test_label_serialisation_round_trip():
    plan = beam_search(
        make_tape(empty_frames(4)),
        type(FAST)(mode="fast", horizon=4, beam=4, verify_k=1, max_iterations=1, durations=(1,)),
    ).best()
    label = TeacherLabel(
        scores=np.arange(N_CAND, dtype=np.float64),
        bomb=1,
        chosen_plan=plan,
        margin=7.5,
        verified_mask=np.array([True] + [False] * (N_CAND - 1)),
        bomb_death_available=False,
        bombs=2.0,
        frame=120,
        stage=1,
        difficulty=1,
        events=["active_bomb"],
        timing_ms={"record": 1.0, "search": 2.0, "verify": 3.0},
    )
    payload = label.as_dict()
    restored = TeacherLabel.from_dict(payload)
    assert restored.scores.tolist() == label.scores.tolist()
    assert restored.verified_mask.tolist() == label.verified_mask.tolist()
    assert restored.bomb == 1
    assert restored.margin == pytest.approx(7.5)
    assert restored.bomb_death_available is False
    assert restored.chosen_plan.slots == label.chosen_plan.slots
    assert restored.to_json() == label.to_json()


def test_label_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        TeacherLabel(
            scores=np.zeros(3),
            bomb=0,
            chosen_plan=None,
            margin=0.0,
            verified_mask=np.zeros(N_CAND, dtype=bool),
        )
    with pytest.raises(ValueError):
        TeacherLabel(
            scores=np.zeros(N_CAND),
            bomb=2,
            chosen_plan=None,
            margin=0.0,
            verified_mask=np.zeros(N_CAND, dtype=bool),
        )


def test_unverified_actions_are_discounted():
    label = TeacherLabel(
        scores=np.full(N_CAND, 100.0),
        bomb=0,
        chosen_plan=None,
        margin=0.0,
        verified_mask=np.zeros(N_CAND, dtype=bool),
    )
    plan = beam_search(
        make_tape(empty_frames(4)),
        type(FAST)(mode="fast", horizon=4, beam=4, verify_k=1, max_iterations=1, durations=(1,)),
    ).best()
    verified = VerifiedPlan(
        plan=plan,
        died=False,
        death_frame=None,
        margin=plan.margin,
        positions=[],
        deaths=[],
        hashes=[],
    )
    label.with_verification(None, [verified], FULL.weights)
    assert label.verified_mask[plan.first_action]
    assert label.scores[plan.first_action] == pytest.approx(plan.score)
    others = [cand for cand in range(N_CAND) if cand != plan.first_action]
    assert label.scores[others[0]] == pytest.approx(100.0 - FULL.weights.unverified_discount)


# ---------------------------------------------------------------------------
# fail closed
# ---------------------------------------------------------------------------
class FakeEngine:
    """A process that answers everything except ``HAZARDS``."""

    def __init__(self, *, hazards_error=True):
        self.hazards_error = hazards_error
        self.frame = 0

    def step(self, mask, frames=1):
        self.frame += frames
        return {"advanced": str(frames), "frame": str(self.frame), "hash": "0" * 16}

    def hash(self):
        return "0" * 16

    def hazards(self):
        if self.hazards_error:
            raise RuntimeError("ERR unknown-command HAZARDS")
        from touhou_rl.v400.gym.protocol import HazardFrame

        return HazardFrame(frame=self.frame, bullet_count=0, laser_count=0, enemy_count=0)

    def player(self):
        return player_row()


def test_record_fails_closed_when_hazards_are_unavailable():
    with pytest.raises(HazardExportUnavailable):
        record(FakeEngine(), horizon=4, planned_actions=[])


def test_record_pads_with_idle_focused_and_keeps_horizon():
    tape = record(FakeEngine(hazards_error=False), horizon=4, planned_actions=[3])
    assert tape.horizon == 4
    assert tape.actions == (3, 3, 9, 9)  # 1 decision, then "不动(聚焦)"
    assert tape.speeds.normal_axial == pytest.approx(4.0)
    assert tape.player["px"].shape == (4,)


def test_consistency_counts_both_directions():
    tape = make_tape(empty_frames(2))
    plan = beam_search(
        tape,
        type(FAST)(mode="fast", horizon=2, beam=2, verify_k=1, max_iterations=1, durations=(1,)),
    ).best()
    from touhou_rl.v400.teacher.search import Plan

    dead = Plan(
        first_action=plan.first_action,
        duration=plan.duration,
        slots=plan.slots,
        margin=0.0,
        death_frame=3,
        end_xy=plan.end_xy,
        score=1.0,
    )
    consistency = Consistency()
    consistency.add(dead, VerifiedPlan(plan, True, 1, 0.0, [], [], []))   # geo dead, engine dead
    consistency.add(plan, VerifiedPlan(plan, True, 1, 0.0, [], [], []))   # geo alive, engine dead
    consistency.add(dead, VerifiedPlan(plan, False, None, 0.0, [], [], []))  # geo dead, engine alive
    consistency.add(plan, VerifiedPlan(plan, False, None, 0.0, [], [], []))  # both alive
    payload = consistency.as_dict()
    assert payload["geo_alive_engine_dead"] == 1
    assert payload["geo_dead_engine_alive"] == 1
    assert payload["agree_dead"] == 1
    assert payload["agree_alive"] == 1
    assert payload["plans_verified"] == 4


# ---------------------------------------------------------------------------
# §5 parameters
# ---------------------------------------------------------------------------
def test_mode_parameters_match_the_specification():
    assert (FULL.horizon, FULL.beam, FULL.verify_k, FULL.max_iterations) == (48, 64, 3, 3)
    assert (FAST.horizon, FAST.beam, FAST.verify_k, FAST.max_iterations) == (32, 24, 1, 1)
    assert FULL.slots == 24 and FAST.slots == 16
    assert get_config("full") is FULL
    with pytest.raises(ValueError):
        get_config("turbo")


# ---------------------------------------------------------------------------
# vectorised search == the pre-vectorisation implementation, bit for bit
# ---------------------------------------------------------------------------
def _dense_frames(count, *, seed=0, bullets=12):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(count):
        rows = tuple(
            bullet(
                192.0 + float(rng.uniform(-40.0, 40.0)),
                384.0 - abs(float(rng.uniform(0.0, 60.0))),
                size=float(rng.choice([2.6, 4.0, 6.0])),
                slot=index,
            )
            for index in range(bullets)
        )
        frames.append((rows, (enemy(192.0, 340.0, hitbox=60.0),)))
    return frames


def _assert_same_search(tape, config):
    # ``use_raster=False`` is the pre-raster implementation: the bit-for-bit
    # contract pinned here is "vectorised == reference", which the raster table
    # deliberately trades for speed (its own equivalence tests are below).
    got = beam_search(tape, config, use_raster=False)
    want = beam_search_reference(tape, config)
    assert np.array_equal(got.scores, want.scores)
    assert set(got.plans) == set(want.plans)
    fields = (
        "first_action",
        "duration",
        "slots",
        "margin",
        "death_frame",
        "end_xy",
        "score",
        "align",
        "edge",
        "high",
        "items",
    )
    for cand, expected in want.plans.items():
        actual = got.plans[cand]
        for name in fields:
            assert getattr(actual, name) == getattr(expected, name), (cand, name)
    assert got.detail == want.detail
    return got


def test_vectorised_search_matches_the_reference_on_dense_and_empty_tapes():
    config = type(FULL)(mode="test", horizon=24, beam=8, verify_k=1, max_iterations=1)
    _assert_same_search(make_tape(_dense_frames(24, seed=1)), config)
    _assert_same_search(make_tape(empty_frames(24)), config)  # MARGIN_CAP fast path
    # A tape with boxes only in the second half exercises the mixed path.
    frames = empty_frames(12) + _dense_frames(12, seed=2)
    _assert_same_search(make_tape(frames), config)
    # beam wider than the expansion cannot prune and must still agree.
    wide = type(FULL)(mode="test", horizon=8, beam=400, verify_k=1, max_iterations=1)
    _assert_same_search(make_tape(_dense_frames(8, seed=3)), wide)


def test_vectorised_search_finds_unknown_closed_form_on_a_wall():
    """A lane-wide bullet kills every plan; both implementations must say so."""
    frames = [((bullet(192.0, 380.0, size=200.0),), ()) for _ in range(8)]
    config = type(FULL)(mode="test", horizon=8, beam=8, verify_k=1, max_iterations=1)
    result = _assert_same_search(make_tape(frames), config)
    assert all(plan.death_frame is not None for plan in result.plans.values())


# ---------------------------------------------------------------------------
# batched verification == per-command verification (same engine verdicts)
# ---------------------------------------------------------------------------
def _float_bits(value: float) -> str:
    return f"{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"


class FakeRolloutEngine:
    """A deterministic stand-in for ``PARK``/``SPAWN``/``CHILD STEP``/``STATE``.

    Each spawned child walks the same 2-frame steps from the same origin; the
    death counter rises exactly once, when the child's frame counter passes
    ``dies_at``.  ``ask_many`` answers the same commands the per-command path
    sends, in order, which is what the real harness guarantees (one line in, one
    line out, strictly sequential).
    """

    def __init__(self, *, dies_at: int | None = None, start=(192.0, 384.0)):
        self.dies_at = dies_at
        self.start = start
        self.children: dict[int, dict] = {}
        self.next_handle = 0

    def park(self):
        return 1

    def spawn(self, template):
        assert template == 1
        self.next_handle += 1
        self.children[self.next_handle] = {"frame": 0, "x": self.start[0], "y": self.start[1], "deaths": 0.0}
        return self.next_handle

    def child_quit(self, handle):
        self.children.pop(handle, None)

    def _answer(self, handle, command):
        state = self.children[handle]
        parts = command.split()
        if parts[0] == "STEP":
            frames = int(parts[2])
            for _ in range(frames):
                state["frame"] += 1
                state["x"] += 4.0
                if self.dies_at is not None and state["frame"] > self.dies_at and state["deaths"] == 0.0:
                    state["deaths"] = 1.0
            return (
                f"OK frames={state['frame']} advanced={frames} "
                f"hash={state['frame']:016x}"
            )
        if parts[0] == "STATE":
            return (
                f"OK frames={state['frame']} f6={_float_bits(state['deaths'])} "
                f"f7={_float_bits(2.0)} f8={_float_bits(3.0)} "
                f"f10={_float_bits(state['x'])} f11={_float_bits(state['y'])}"
            )
        return "ERR child-command"

    def child_step(self, handle, mask, frames=1):
        reply = self._answer(handle, f"STEP {mask} {frames}")
        return dict(part.split("=", 1) for part in reply.split()[1:] if "=" in part)

    def child(self, handle, command):
        return self._answer(handle, command)

    def ask_many(self, commands):
        assert len(commands) > 0
        return [
            self._answer(self._handle_of(command), " ".join(command.split()[2:]))
            for command in commands
        ]

    def _handle_of(self, command):
        return int(command.split()[1])


def _fake_plans(tape, slots_list):
    from touhou_rl.v400.teacher.search import Plan

    plans = []
    for index, slots in enumerate(slots_list):
        plans.append(
            Plan(
                first_action=slots[0],
                duration=1,
                slots=tuple(slots),
                margin=16.0,
                death_frame=None,
                end_xy=(192.0, 384.0),
                score=100.0 + index,
            )
        )
    return plans


def test_batched_verification_matches_the_per_command_judgement():
    tape = make_tape(empty_frames(8))
    plan_slots = [
        [7, 7, 0, 0],
        [3, 4, 3, 4],
        [0, 0, 0, 0],
    ]
    plans = _fake_plans(tape, plan_slots)
    for dies_at in (None, 1, 5):
        engine_a = FakeRolloutEngine(dies_at=dies_at)
        engine_b = FakeRolloutEngine(dies_at=dies_at)
        old = verify_plans(engine_a, tape, plans, config=FULL, template=1, batch=False)
        new = verify_plans(engine_b, tape, plans, config=FULL, template=1, batch=True)
        for left, right in zip(old, new):
            assert left.died == right.died
            assert left.death_frame == right.death_frame
            assert left.frames_advanced == right.frames_advanced
            # Run boundaries are a subset of the per-slot samples; the final
            # sample must be the same frame in both.
            assert left.positions[-1] == right.positions[-1]


def test_batched_verification_detects_a_slot_zero_death():
    """The death baseline is the child *before* the first step, not after it."""
    tape = make_tape(empty_frames(8))
    plans = _fake_plans(tape, [[7, 7, 0, 0]])
    engine = FakeRolloutEngine(dies_at=0)
    verified = verify_plans(engine, tape, plans, config=FULL, template=1, batch=True)
    assert verified[0].died is True
    assert verified[0].death_frame is not None


# ---------------------------------------------------------------------------
# source="child": fail closed, and use the callable when the engine allows it
# ---------------------------------------------------------------------------
class FakeChildProcess(FakeEngine):
    """A process whose ``CHILD HAZARDS`` capability is set by ``available``."""

    def __init__(self, *, available: bool, hazards_error: bool = False):
        super().__init__(hazards_error=hazards_error)
        self.available = available
        self.steps = 0
        self.probes = 0

    def step(self, mask, frames=1):
        self.steps += frames
        return super().step(mask, frames)

    def probe_child_hazards(self, **_kwargs):
        self.probes += 1
        if self.available:
            return {"available": True, "reason": "CHILD <h> HAZARDS accepted"}
        return {"available": False, "reason": "'CHILD 1 HAZARDS' -> ERR child-command"}

    def park(self):
        return 1

    def spawn(self, template):
        return 1


def test_record_child_source_fails_closed_without_the_capability():
    process = FakeChildProcess(available=False, hazards_error=False)
    calls: list[int] = []

    def callable_(frame):
        calls.append(frame)
        return {"bullets": (), "lasers": (), "enemies": ()}

    with pytest.raises(ChildHazardsUnavailable) as error:
        record(process, horizon=4, source="child", child_callable=callable_)
    assert "ERR child-command" in str(error.value)
    assert calls == []  # the engine's refusal is consulted before any frame
    assert process.steps == 0  # and the main process is never advanced
    assert process.probes == 1


def test_record_child_source_requires_a_callable_and_is_a_bad_source_value():
    process = FakeChildProcess(available=True)
    with pytest.raises(ChildHazardsUnavailable):
        record(process, horizon=4, source="child")
    with pytest.raises(ValueError):
        record(process, horizon=4, source="fork")


def test_record_child_source_walks_the_callable_and_leaves_the_main_process_alone():
    process = FakeChildProcess(available=True)
    seen: list[int] = []

    def callable_(frame):
        seen.append(frame)
        return {
            "frame": 100 + frame,
            "hash": f"{frame:016x}",
            "player": player_row(x=192.0 + frame),
            "bullets": (bullet(192.0, 380.0),),
            "lasers": (),
            "enemies": (enemy(192.0, 340.0),),
        }

    tape = record(process, horizon=4, planned_actions=[3], source="child", child_callable=callable_)
    assert seen == [0, 1, 2, 3]
    assert tape.horizon == 4
    assert tape.actions == (3, 3, 9, 9)
    assert tape.meta["source"] == "child"
    assert tape.frames.tolist() == [100, 101, 102, 103]
    assert tape.speeds.normal_axial == pytest.approx(SPEEDS.normal_axial)
    assert tape.lethal_counts()["bullets"] == 4
    assert tape.box_array(0).shape[0] == 2  # one bullet + one enemy body
    assert tape.start_position[0] == pytest.approx(192.0)  # row 0's PLAYER
    assert process.steps == 0  # the shadow did not move


def test_record_child_source_wraps_a_failing_callable():
    process = FakeChildProcess(available=True)

    def callable_(frame):
        raise RuntimeError("child died")

    with pytest.raises(ChildHazardsUnavailable) as error:
        record(process, horizon=4, source="child", child_callable=callable_)
    assert "child died" in str(error.value)


def test_reference_selection_is_used_when_aliasing_cannot_be_avoided(monkeypatch):
    """``_reference_selection`` is kept for the NaN guard and must agree with lexsort."""
    from touhou_rl.v400.teacher import search as search_module

    rng = np.random.default_rng(0)
    died = rng.random((3, 4, N_CAND)) < 0.4
    margins = rng.integers(0, 4, (3, 4, N_CAND)).astype(np.float64)
    got = search_module._lexsort_selection(died, margins, 8)
    for row in range(3):
        want = np.lexsort(
            (
                np.tile(np.arange(N_CAND), 4),
                -margins[row].reshape(-1),
                died[row].reshape(-1).astype(np.int8),
            )
        )[:8]
        assert got[row].tolist() == want.tolist()


# ---------------------------------------------------------------------------
# recorder template pool: honest refusal until the engine can serve CHILD HAZARDS
# ---------------------------------------------------------------------------
def test_recorder_pool_reports_the_engine_capability_and_fails_closed():
    refused = RecorderPool()
    process = FakeChildProcess(available=False)
    status = refused.status(process)
    assert status["available"] is False
    assert "ERR child-command" in status["reason"]
    assert status["budget"]["fixed_ms"] == 0.0
    with pytest.raises(ChildHazardsUnavailable):
        refused.acquire(process)

    allowed = RecorderPool()
    process_ok = FakeChildProcess(available=True)
    assert allowed.status(process_ok)["available"] is True
    handle = allowed.acquire(process_ok)  # parks once, then spawns
    assert handle == 1
    assert allowed.status(process_ok)["spawned"] == 1


def test_restore_budget_separates_fixed_from_poolable_cost():
    from touhou_rl.v400.teacher.pool import RestoreBudget

    budget = RestoreBudget(start_ms=1180.8, reset_ms=415.3, replay_ms=19.9, park_spawn_ms=0.6)
    assert budget.fixed_ms == pytest.approx(1596.1)
    assert budget.poolable_ms == pytest.approx(19.9)
    assert budget.total_ms == pytest.approx(1616.0)
    assert budget.as_dict()["frames_replayed"] == 0


# ---------------------------------------------------------------------------
# child recording (EXP-P1-007): one pipelined batch, shadow untouched
# ---------------------------------------------------------------------------
def _fbits(value: float) -> str:
    return f"{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"


def _bullet_row(x, y, size=4.0, state=1, slot=0):
    """One ``HAZARDS`` bullet row in the engine's own field order."""

    return (
        f"{slot}:{state}:0:{_fbits(x)}:{_fbits(y)}:{_fbits(0.0)}:"
        f"{_fbits(0.0)}:{_fbits(0.0)}:10:{_fbits(size)}:0"
    )


class FakeWorldEngine:
    """A process whose world is a pure function of its frame counter.

    That is the property the real ``rl_hazard2`` measured for ``CHILD`` replies
    (byte-identical to the parent's), so a child recording and a main-process
    recording of the same world must produce the same tape.  Children are real
    copies of the ``PARK``ed state, and every step/death goes through the same
    engine-shaped replies the real harness emits.
    """

    def __init__(
        self,
        *,
        kills_at=None,
        wall=False,
        events_flags=0x0,
        events_error=None,
        bombs=3.0,
        child_hazards=True,
        child_reply_error=False,
    ):
        self.kills_at = kills_at
        self.wall = wall
        self.events_flags = int(events_flags)
        self.events_error = events_error
        self.bombs = float(bombs)
        self.child_hazards = child_hazards
        self.child_reply_error = child_reply_error
        self.main = {"frame": 0, "x": 192.0, "y": 384.0, "deaths": 0.0}
        self.templates: dict[int, dict] = {}
        self.children: dict[int, dict] = {}
        self._next_template = 0
        self._next_handle = 0
        self.batches: list[list[str]] = []
        self.main_steps = 0
        self.main_masks: list[int] = []
        self.probes = 0
        self.quits = 0

    # -- world ------------------------------------------------------------
    def _bullets(self, frame):
        if self.wall:
            # A lane-wide lethal body: every plan dies geometrically.
            return [_bullet_row(192.0, 384.0, 400.0)]
        if frame in (2, 3):
            return [_bullet_row(100.0 + frame, 380.0, 4.0)]
        return []

    def _hazards_reply(self, state):
        rows = self._bullets(state["frame"])
        return (
            f"OK frames={state['frame']} nb={len(rows)} nl=0 ne=0 "
            f"bullets={','.join(rows)} lasers= enemies="
        )

    def _player_reply(self, state):
        x, y, half = state["x"], state["y"], 0.825
        return (
            f"OK frames={state['frame']} hp={_fbits(2.0)} lives=2 "
            f"bombs={_fbits(self.bombs)} power={_fbits(0.0)} focus=0 isYoukai=0 "
            f"playerState=0 invulnState=0 px={_fbits(x)} py={_fbits(y)} "
            f"hurtboxMinX={_fbits(x - half)} hurtboxMinY={_fbits(y - half)} "
            f"hurtboxMaxX={_fbits(x + half)} hurtboxMaxY={_fbits(y + half)} "
            f"speedNormalAxial={_fbits(4.0)} speedNormalDiagonal={_fbits(2.828427)} "
            f"speedFocusedAxial={_fbits(2.0)} speedFocusedDiagonal={_fbits(1.414214)}"
        )

    def _state_reply(self, state):
        return (
            f"OK frames={state['frame']} f0={state['frame']} f1=1 f2=0 "
            f"f3={state['frame']} f4=0 f5=0 f6={_fbits(state['deaths'])} "
            f"f7={_fbits(2.0)} f8={_fbits(self.bombs)} f9={_fbits(0.0)} "
            f"f10={_fbits(state['x'])} f11={_fbits(state['y'])} f12={_fbits(0.0)} "
            f"f13=0 f14=0 f15=0 f16=0 f17=0 f18=0 f19=0 f20=0 f21=0 f22=0 f23=0 f24=1"
        )

    def _advance(self, state, frames):
        for _ in range(frames):
            state["frame"] += 1
            if self.kills_at is not None and state["frame"] >= self.kills_at:
                state["deaths"] = 1.0
        return f"OK advanced={frames} frame={state['frame']} hash={state['frame']:016x}"

    def _dispatch(self, state, command):
        verb = command.split(" ", 1)[0]
        if verb == "HASH":
            return f"OK hash={state['frame']:016x} frames={state['frame']}"
        if verb == "HAZARDS":
            return self._hazards_reply(state)
        if verb == "PLAYER":
            return self._player_reply(state)
        if verb == "STATE":
            return self._state_reply(state)
        if verb == "STEP":
            return self._advance(state, int(command.split()[2]))
        raise AssertionError(f"unexpected command {command!r}")

    # -- engine surface ---------------------------------------------------
    def probe_child_hazards(self, **_kwargs):
        self.probes += 1
        if self.child_hazards:
            return {"available": True, "reason": "CHILD <h> HAZARDS accepted"}
        return {"available": False, "reason": "'CHILD 1 HAZARDS' -> ERR child-command"}

    def park(self):
        self._next_template += 1
        self.templates[self._next_template] = dict(self.main)
        return self._next_template

    def spawn(self, template):
        self._next_handle += 1
        self.children[self._next_handle] = dict(self.templates[template])
        return self._next_handle

    def child_quit(self, handle):
        self.quits += 1
        self.children.pop(handle, None)

    def ask_many(self, commands):
        commands = list(commands)
        self.batches.append(commands)
        replies: list[str] = []
        for command in commands:
            if command.startswith("CHILD "):
                _, handle, rest = command.split(" ", 2)
                if self.child_reply_error and rest.startswith("HAZARDS"):
                    raise EngineError("'CHILD 1 HAZARDS' -> ERR child-command")
                replies.append(self._dispatch(self.children[int(handle)], rest))
            else:
                replies.append(self._dispatch(self.main, command))
        return replies

    def hash(self):
        return f"{self.main['frame']:016x}"

    def hazards(self):
        from touhou_rl.v400.gym.protocol import parse_hazards_reply

        return parse_hazards_reply(self._hazards_reply(self.main))

    def player(self):
        from touhou_rl.v400.gym.protocol import parse_player_reply

        return parse_player_reply(self._player_reply(self.main))

    def step(self, mask, frames=1):
        reply = self._advance(self.main, frames)
        self.main_steps += frames
        self.main_masks.append(int(mask))
        return dict(part.split("=", 1) for part in reply.split()[1:] if "=" in part)

    def state(self):
        fields = parse_fields(self._state_reply(self.main))
        decoded = decode_state_fields(fields)
        return {
            "frame": decoded["frame"],
            "raw": fields,
            "deaths": decoded["deaths"],
            "lives": decoded["lives"],
            "bombs": decoded["bombs"],
            "px": decoded["px"],
            "py": decoded["py"],
            "has_frame_event_flags": False,
        }

    def events_snapshot(self):
        if self.events_error:
            return {
                "available": False,
                "reason": self.events_error,
                "frame": None,
                "frame_event_flags": None,
                "names": [],
                "player_hit": False,
            }
        event = parse_events_reply(
            f"OK frames={self.main['frame']} frameEventFlags={self.events_flags:04x}"
        )
        return {
            "available": True,
            "reason": "EVENTS accepted",
            "frame": int(event.frame),
            "frame_event_flags": int(event.frame_event_flags),
            "names": list(event.names),
            "player_hit": bool(event.player_hit),
        }

    def reset(self, *, stage=1, difficulty=1, team=0, seed=1, practice=False):
        self.main = {"frame": 0, "x": 192.0, "y": 384.0, "deaths": 0.0}
        self.main_masks.clear()
        return {"reset": "1", "ready": "1"}


def test_child_recording_is_one_batch_and_matches_the_main_process_tape():
    child_engine = FakeWorldEngine()
    main_engine = FakeWorldEngine()
    child = record_child_tape(child_engine, horizon=6, planned_actions=[3])
    tape = child.tape
    reference = record(main_engine, horizon=6, planned_actions=[3], source="main")

    # The shadow never moved, and the whole recording was one pipelined batch.
    assert child_engine.main_steps == 0
    assert len(child_engine.batches) == 1
    assert child.commands == 3 + 3 * (6 - 1)
    assert child.steps == 5
    assert child.frames == 6
    assert child.reply_bytes > 0
    assert child_engine.quits == 1  # the child is dropped, the template survives

    batch = child_engine.batches[0]
    assert batch[0].endswith(" HASH")
    assert batch[1].endswith(" HAZARDS")
    assert batch[2].endswith(" PLAYER")
    assert batch[3].split()[2] == "STEP"
    assert batch[4].endswith(" HAZARDS")
    assert batch[5].endswith(" PLAYER")

    # Same world, same commands -> the child's tape must be the main path's tape.
    assert tape.actions == reference.actions == (3, 3, 9, 9, 9, 9)
    assert tape.frames.tolist() == reference.frames.tolist() == [0, 1, 2, 3, 4, 5]
    assert tape.hashes == reference.hashes
    assert tape.speeds.as_tuple() == reference.speeds.as_tuple()
    for key in (
        "px",
        "py",
        "min_x",
        "min_y",
        "max_x",
        "max_y",
        "state",
        "invuln",
        "focus",
        "bombs",
        "lives",
        "power",
    ):
        assert np.array_equal(tape.player[key], reference.player[key]), key
    for index in range(6):
        assert tape.bullets[index] == reference.bullets[index]
        assert tape.lasers[index] == reference.lasers[index]
        assert tape.enemies[index] == reference.enemies[index]
    assert np.array_equal(tape.box_array(2), reference.box_array(2))
    # EXP-P1-021: `lethal_counts()` also reports the engine's `damageable` laser
    # rows (the laser judging window), so the exact contract is these four keys.
    assert tape.lethal_counts() == reference.lethal_counts() == {
        "bullets": 2,
        "lasers": 0,
        "lasers_damageable": 0,
        "enemies": 0,
    }


def test_child_recording_fails_closed_before_touching_anything():
    engine = FakeWorldEngine(child_hazards=False)
    with pytest.raises(ChildHazardsUnavailable) as error:
        record_child_tape(engine, horizon=4, planned_actions=[3])
    assert "ERR child-command" in str(error.value)
    assert engine.probes == 1
    assert engine.batches == []
    assert engine.main_steps == 0
    assert engine.children == {}


def test_child_recording_wraps_a_refused_child_reply_and_still_drops_the_child():
    engine = FakeWorldEngine(child_reply_error=True)
    with pytest.raises(ChildHazardsUnavailable) as error:
        record_child_tape(engine, horizon=4, planned_actions=[3])
    assert "ERR child-command" in str(error.value)
    assert engine.main_steps == 0
    assert engine.quits == 1  # the failed child is not leaked


# ---------------------------------------------------------------------------
# EVENTS: the two bomb rules of 11-P1 §4
# ---------------------------------------------------------------------------
DECISION_TEST = type(FULL)(mode="test", horizon=8, beam=8, verify_k=1, max_iterations=1)


def test_death_bomb_window_and_feasibility_follow_the_spec():
    assert death_bomb_window(0) == 2  # no bombs at all: 2-frame window
    assert death_bomb_window(1) == 6
    assert death_bomb_window(2) == 12
    assert death_bomb_window(3) == 15  # cap
    assert death_bomb_window(3, time_point_bonus=True) == 15
    assert death_bomb_window(3, spell_card=True) == 30  # doubled inside a spell card
    assert death_bomb_window(3, ship_multiplier=1.8) == 27  # ships 0/4/5
    assert death_bomb_feasible(0) is False
    assert death_bomb_feasible(1) is True
    assert death_bomb_feasible(3) is True


def test_decision_death_bombs_on_a_real_player_hit():
    engine = FakeWorldEngine(events_flags=0x0602)
    decision = decide(engine, DECISION_TEST, record_source="child")
    assert decision.label.bomb == 1
    assert "death_bomb_player_hit" in decision.label.events
    assert decision.bomb_death_available is True
    assert decision.label.frame_event_flags == 0x0602
    assert "PLAYER_HIT" in decision.label.extra["event_names"]
    assert decision.label.extra["death_bomb_feasible"] is True
    assert engine.main_steps == 0


def test_decision_does_not_death_bomb_without_a_player_hit():
    engine = FakeWorldEngine(events_flags=0x0400)
    decision = decide(engine, DECISION_TEST, record_source="child")
    assert decision.label.bomb == 0
    assert "death_bomb_no_player_hit" in decision.label.events
    assert decision.bomb_death_available is True
    assert decision.label.frame_event_flags == 0x0400
    assert "PLAYER_HIT" not in decision.label.extra["event_names"]


def test_decision_does_not_death_bomb_without_a_bomb():
    engine = FakeWorldEngine(events_flags=0x0002, bombs=0.0)
    decision = decide(engine, DECISION_TEST, record_source="child")
    assert decision.label.bomb == 0
    assert "death_bomb_player_hit_but_no_bombs" in decision.label.events
    assert decision.label.bomb_death_available is True
    assert decision.bombs == 0.0


def test_decision_reports_events_unavailable_instead_of_fabricating_a_zero():
    engine = FakeWorldEngine(events_error="'EVENTS' -> ERR events-unavailable g_replaymanager=0")
    decision = decide(engine, DECISION_TEST, record_source="child")
    assert decision.label.bomb == 0
    assert decision.bomb_death_available is False
    assert decision.label.frame_event_flags is None
    assert any(
        event.startswith("death_bomb_unavailable") for event in decision.label.events
    ), decision.label.events
    assert "ERR events-unavailable" in json_events(decision.label.events)


def json_events(events):
    return " ".join(events)


def test_decision_active_bomb_rule_is_unchanged():
    engine = FakeWorldEngine(wall=True, kills_at=0, events_flags=0x0)
    decision = decide(engine, DECISION_TEST, record_source="child")
    assert decision.label.bomb == 1
    assert "active_bomb" in decision.label.events
    assert "all_verified_plans_die" in decision.label.events
    assert "death_bomb_player_hit" not in decision.label.events


def test_death_verdict_agrees_between_child_recording_and_legacy_restore():
    child_engine = FakeWorldEngine(kills_at=3)
    legacy_engine = FakeWorldEngine(kills_at=3)
    child = decide(child_engine, DECISION_TEST, record_source="child")
    legacy = decide(legacy_engine, DECISION_TEST, record_source="main")

    assert child_engine.main_steps == 0  # child recording leaves the shadow alone
    assert legacy_engine.main_steps > 0  # the legacy path walked it
    assert np.array_equal(child.label.scores, legacy.label.scores)
    assert child.label.chosen_plan is not None
    assert child.label.chosen_plan.as_dict() == legacy.label.chosen_plan.as_dict()
    child_verdicts = [(item.plan.first_action, item.died, item.death_frame) for item in child.verified]
    legacy_verdicts = [(item.plan.first_action, item.died, item.death_frame) for item in legacy.verified]
    assert child_verdicts == legacy_verdicts
    assert any(item.died for item in child.verified)  # the death was really exercised
    assert child.label.bomb == legacy.label.bomb == 1  # rule 1: all plans die
    assert child.bomb_death_available is True and legacy.bomb_death_available is True


def test_legacy_restore_replays_the_bomb_bit():
    engine = FakeWorldEngine()
    restore(
        engine,
        [(3, 1), (4, 0)],
        target_frames=4,
        stage=1,
        difficulty=1,
        team=0,
        seed=1,
    )
    assert engine.main_masks == [cand_to_mask(3, 1), cand_to_mask(4, 0)]


# ---------------------------------------------------------------------------
# engine gate: the build the code was measured on, or nothing
# ---------------------------------------------------------------------------
def test_engine_sha_gate_refuses_a_foreign_build(tmp_path):
    import hashlib

    binary = tmp_path / "th08-modern"
    binary.write_bytes(b"not-the-engine")
    actual = hashlib.sha256(b"not-the-engine").hexdigest()
    gate = check_engine_binary(binary, actual)
    assert gate["match"] is True
    assert gate["sha256"] == actual
    recorded = check_engine_binary(binary, None)
    assert recorded["match"] is False and recorded["expected"] is None
    with pytest.raises(EngineShaMismatch):
        check_engine_binary(binary, EXPECTED_ENGINE_SHA256)
    with pytest.raises(EngineShaMismatch):
        check_engine_binary(tmp_path / "missing-engine", actual)


def test_cli_defaults_to_the_verified_child_build_and_keeps_legacy_opt_in(tmp_path):
    cli = _load_cli()
    args = cli.parse_args([])
    assert Path(args.engine).name == "rl_hazard2"
    assert args.binary is None
    assert args.expected_sha == EXPECTED_ENGINE_SHA256
    assert args.allow_legacy_restore is False
    assert cli.EXIT_ENGINE_MISMATCH == 4
    assert cli.EXIT_FAIL_CLOSED == 3
    # A missing/foreign binary fails closed before any process is started.
    assert cli.main(["--engine", str(tmp_path / "no-such-build")]) == cli.EXIT_ENGINE_MISMATCH


def _load_cli():
    path = ROOT / "tools/rl/v400_teacher.py"
    spec = importlib.util.spec_from_file_location("v400_teacher_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# rasterised lethal table (EXP-P1-008)
# ---------------------------------------------------------------------------
def _grid_centres(table):
    x0, y0, nx, ny = lattice(table.cell)
    xs = x0 + (np.arange(nx) + 0.5) * table.cell
    ys = y0 + (np.arange(ny) + 0.5) * table.cell
    return np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)


def test_raster_cell_index_covers_the_playfield():
    x0, y0, nx, ny = lattice(CELL)
    assert (x0, y0, nx, ny) == (8.0, 16.0, 184, 208)
    corners = np.array([[8.0, 16.0], [376.0, 432.0], [7.9, 15.9], [400.0, 500.0]])
    index = flat_index(corners)
    assert index[0] == 0
    assert index[1] == (ny - 1) * nx + (nx - 1)
    assert index[2] == 0  # clamped, not wrapped
    assert index[3] == index[1]
    centres = np.array([[9.0, 17.0], [9.0 + 2 * 5, 17.0 + 2 * 3]])
    assert list(flat_index(centres)) == [0, 3 * nx + 5]


def test_raster_hit_is_a_superset_and_core_a_subset_of_the_lethal_region():
    """The soundness property the exact fallback rests on.

    ``core`` => lethal for every point of the cell, and ``not hit`` => the cell
    cannot hold a lethal point at all.  If either direction broke, the table
    could invent or miss a death.
    """
    tape = make_tape(_dense_frames(12, seed=7, bullets=8))
    table = build_table(tape, np.asarray(tape.hurtbox_half))
    grid = _grid_centres(table)
    hits = 0
    cores = 0
    for frame in range(0, 12, 4):
        _margin, dead = table.exact(frame, grid)
        hit = table.hit(frame).ravel()
        core = table.core(frame).ravel()
        hits += int(hit.sum())
        cores += int(core.sum())
        assert np.all(dead[core])          # core is a subset of the region
        assert not np.any(dead & ~hit)     # not hit proves alive
    assert hits > 0 and cores > 0 and cores < hits


def test_raster_death_verdict_is_exact_on_random_points():
    """``LethalTable.query`` must agree with the AABB expression, point for point."""
    tape = make_tape(_dense_frames(12, seed=11, bullets=10))
    table = build_table(tape, np.asarray(tape.hurtbox_half))
    rng = np.random.default_rng(3)
    total = 0
    for frame in range(0, 12, 3):
        points = np.stack(
            [rng.uniform(8.0, 376.0, 3000), rng.uniform(16.0, 432.0, 3000)], axis=1
        )
        _exact_margin, exact_dead = table.exact(frame, points)
        _margin, dead = table.query(frame, points)
        assert np.array_equal(dead, exact_dead)
        total += int(dead.sum())
    assert total > 0  # the sample really does contain lethal points


def test_raster_margin_stays_within_a_cell_of_the_exact_gap():
    tape = make_tape(_dense_frames(8, seed=5, bullets=6))
    table = build_table(tape, np.asarray(tape.hurtbox_half))
    grid = _grid_centres(table)
    worst = 0.0
    for frame in range(8):
        exact_margin, exact_dead = table.exact(frame, grid)
        margin, _dead = table.query(frame, grid)
        alive = ~exact_dead
        assert alive.any()
        worst = max(worst, float(np.abs(margin[alive] - exact_margin[alive]).max()))
    # One margin-lattice cell is ``margin_cell`` = 2 x CELL px; the sampling
    # error of a lattice value is bounded by the cell diagonal, so this is the
    # documented quantisation budget, not a measured lucky number.
    assert worst <= 2.0 * CELL


def test_raster_search_keeps_every_death_verdict_of_the_exact_search():
    config = type(FULL)(mode="test", horizon=24, beam=8, verify_k=1, max_iterations=1)
    tape = make_tape(_dense_frames(24, seed=1))
    exact = beam_search(tape, config, use_raster=False)
    raster = beam_search(tape, config)
    assert set(exact.plans) == set(raster.plans)
    for cand in exact.plans:
        assert (exact.plans[cand].death_frame is None) == (
            raster.plans[cand].death_frame is None
        ), cand
    # The score can only move through the quantised margin term.
    assert np.max(np.abs(exact.scores - raster.scores)) <= 2.0 * CELL


def test_raster_search_is_faster_than_the_exact_search_on_a_dense_tape():
    import time

    config = type(FULL)(mode="test", horizon=48, beam=32, verify_k=1, max_iterations=1)
    tape = make_tape(_dense_frames(48, seed=2, bullets=16))
    started = time.perf_counter()
    beam_search(tape, config, use_raster=False)
    exact_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    # The table's own claim is the ranking path; the finalist stage is measured
    # separately (EXP-P1-010) because it needs the numba exact kernel to be
    # affordable.
    beam_search(tape, config, finalist_extra=0)
    raster_ms = (time.perf_counter() - started) * 1000.0
    assert raster_ms < 0.8 * exact_ms, (raster_ms, exact_ms)

# ---------------------------------------------------------------------------
# numba build/lookup backend + finalist re-scoring (EXP-P1-010)
# ---------------------------------------------------------------------------
def _numba_ready():
    from touhou_rl.v400.teacher import raster_numba

    return raster_numba.available() and raster_numba.probe()


def test_raster_numba_build_is_identical_to_the_numpy_build():
    """The accelerated build must produce the same table, bit for bit."""
    tape = make_tape(_dense_frames(16, seed=13, bullets=9))
    half = np.asarray(tape.hurtbox_half)
    reference = build_table(tape, half, backend="numpy")
    assert reference.backend == "numpy"
    if not _numba_ready():
        pytest.skip("numba is not importable on this interpreter")
    fast = build_table(tape, half, backend="numba")
    assert fast.backend == "numba"
    assert np.array_equal(reference.table, fast.table)
    assert (fast.nx, fast.ny) == (reference.nx, reference.ny)


def test_raster_numba_lookup_matches_the_numpy_lookup():
    """Query/exact kernels must answer exactly like the numpy expressions."""
    tape = make_tape(_dense_frames(12, seed=17, bullets=10))
    half = np.asarray(tape.hurtbox_half)
    table = build_table(tape, half, backend="numpy")
    if not _numba_ready():
        pytest.skip("numba is not importable on this interpreter")
    fast = build_table(tape, half, backend="numba")
    rng = np.random.default_rng(23)
    points = np.stack(
        [rng.uniform(8.0, 376.0, 4000), rng.uniform(16.0, 432.0, 4000)], axis=1
    )
    for frame in range(0, 12, 3):
        got_margin, got_dead = table.query(frame, points)
        assert np.array_equal(table.exact(frame, points)[1], got_dead)
        fast_margin, fast_dead = fast.query(frame, points)
        assert np.array_equal(got_dead, fast_dead)
        assert np.array_equal(got_margin, fast_margin)
        exact_margin, exact_dead = table.exact(frame, points)
        kernel_margin, kernel_dead = fast.exact_points(frame, points)
        assert np.array_equal(exact_margin, kernel_margin)
        assert np.array_equal(exact_dead, kernel_dead)


def _best_action(result):
    return max(result.plans.values(), key=lambda plan: plan.score).first_action


def test_finalist_stage_matches_the_exact_search():
    """Raster pre-screen + exact finalist re-scoring == the exact search.

    EXP-P1-010 §3: the table may rank the beam, but the nodes that can still be
    kept are re-scored with the exact AABB expression, so the drift that
    EXP-P1-008 measured at plan level has to disappear on this tape.
    """
    config = type(FULL)(mode="test", horizon=24, beam=8, verify_k=1, max_iterations=1)
    tape = make_tape(_dense_frames(24, seed=4, bullets=12))
    exact = beam_search(tape, config, use_raster=False)
    finalist = beam_search(tape, config, finalist_extra=8)
    assert set(exact.plans) == set(finalist.plans)
    for cand in exact.plans:
        assert exact.plans[cand].death_frame == finalist.plans[cand].death_frame, cand
    assert _best_action(exact) == _best_action(finalist)
    # The survivor's margin is the exact one whenever it was a finalist; a node
    # that entered the beam from outside the finalist set keeps the table value,
    # so the score can still move inside the quantisation budget.
    assert np.max(np.abs(exact.scores - finalist.scores)) <= 2.0 * CELL
    # the stage really ran: it re-scored nodes on top of the raster pass
    assert finalist.detail["finalist_points"] > 0


def test_finalist_stage_is_off_at_zero_and_keeps_the_table_verdict():
    """``finalist_extra=0`` is the EXP-P1-008 ranking; it still cannot kill wrongly."""
    config = type(FULL)(mode="test", horizon=16, beam=6, verify_k=1, max_iterations=1)
    tape = make_tape(_dense_frames(16, seed=6, bullets=8))
    plain = beam_search(tape, config, finalist_extra=0)
    assert "finalist_points" not in plain.detail
    exact = beam_search(tape, config, use_raster=False)
    for cand in exact.plans:
        assert (exact.plans[cand].death_frame is None) == (
            plain.plans[cand].death_frame is None
        ), cand


def test_beam_search_without_raster_still_runs():
    """The exact path stays available: it is the equivalence oracle and the fallback."""
    config = type(FULL)(mode="test", horizon=16, beam=6, verify_k=1, max_iterations=1)
    tape = make_tape(_dense_frames(16, seed=9, bullets=7))
    fast = beam_search(tape, config, use_raster=False)
    reference = beam_search_reference(tape, config)
    assert set(fast.plans) == set(reference.plans)
    for cand in fast.plans:
        assert fast.plans[cand].death_frame == reference.plans[cand].death_frame
    assert np.array_equal(fast.scores, reference.scores)


# ---------------------------------------------------------------------------
# SHOOT always on + the stage-advance release frame (EXP-P1-009)
# ---------------------------------------------------------------------------
def test_shoot_bit_is_in_every_candidate_mask():
    """13-P3 §2.2 / D-08: SHOOT is a constant, not a search dimension.

    The 18 candidates are ``focus * 9 + move``; if any of them ever stopped
    setting ``BTN_SHOOT`` the teacher would spend a whole run not shooting
    (EXP-P1-009 §1 hypothesis 1), so the bit is asserted on the encoder itself.
    """
    from touhou_rl.spec import BTN_BOMB, BTN_FOCUS, BTN_SHOOT
    from touhou_rl.v400.teacher.shoot import commit_frames

    assert N_CAND == 18
    for cand in range(N_CAND):
        mask = cand_to_mask(cand)
        assert mask & BTN_SHOOT, cand
        assert not mask & BTN_BOMB, cand
        assert bool(mask & BTN_FOCUS) == (cand // 9), cand
        # A decision is 2 frames by default and both of them shoot.
        frames = commit_frames(cand, decision_frames=2)
        assert frames == [(mask, 2)]
        assert all(m & BTN_SHOOT for m, _ in frames)


def test_release_frame_is_last_exactly_once_and_only_drops_shoot():
    """The stage-advance edge: one SHOOT-free frame, then SHOOT again.

    ``GameManager.cpp:381-389`` needs ``(g_CurFrameInput & 0x1001)`` to differ
    from the previous frame's word, so the block must end without SHOOT and the
    next block must start with it.  Frame count, bomb placement and every other
    button bit must be untouched.
    """
    from touhou_rl.spec import BTN_BOMB, BTN_SHOOT
    from touhou_rl.v400.teacher.shoot import commit_block, mask_summary

    plain = commit_block([7, 11], bomb=1, decision_frames=2)
    pulsed = commit_block([7, 11], bomb=1, decision_frames=2, release_shoot=True)
    assert sum(c for _, c in plain) == sum(c for _, c in pulsed) == 4
    assert [c for _, c in pulsed] == [2, 1, 1]
    assert pulsed[0] == plain[0]
    assert pulsed[1][0] == plain[1][0] and pulsed[1][1] + 1 == plain[1][1]
    assert pulsed[-1][0] & BTN_SHOOT == 0
    assert pulsed[-1][0] & BTN_BOMB == 0  # a release frame never spends a bomb
    # The only difference is the SHOOT bit itself.
    assert pulsed[-1][0] == plain[-1][0] & ~BTN_SHOOT & ~BTN_BOMB
    # The next decision's first frame re-asserts SHOOT: that is the rising edge.
    follow = commit_block([9], bomb=0, decision_frames=2)[0][0]
    assert follow & BTN_SHOOT
    assert (pulsed[-1][0] & BTN_SHOOT) != (follow & BTN_SHOOT)
    summary = mask_summary(pulsed)
    assert summary == {"frames": 4, "shoot_frames": 3, "bomb_frames": 2, "release_frames": 1}


def test_release_due_is_periodic_and_zero_disables_it():
    from touhou_rl.v400.teacher.shoot import SHOOT_PULSE_PERIOD, release_due

    assert SHOOT_PULSE_PERIOD > 0
    due = [index for index in range(3 * SHOOT_PULSE_PERIOD) if release_due(index, SHOOT_PULSE_PERIOD)]
    assert due == [SHOOT_PULSE_PERIOD - 1, 2 * SHOOT_PULSE_PERIOD - 1, 3 * SHOOT_PULSE_PERIOD - 1]
    assert release_due(0, 0) is False
    assert release_due(10**6, 0) is False
    assert release_due(0, 1) is True  # period 1 == release every decision


def test_config_and_label_carry_the_shoot_decision():
    """The fix has to be visible in the config and in the emitted label."""
    config = get_config("full")
    assert config.shoot_always is True
    assert config.shoot_pulse_period == 24
    with pytest.raises(ValueError):
        type(FULL)(mode="bad", horizon=16, beam=2, verify_k=1, max_iterations=1,
                   shoot_pulse_period=-1)
    label = TeacherLabel(
        scores=np.zeros(N_CAND),
        bomb=0,
        chosen_plan=None,
        margin=float("nan"),
        verified_mask=np.zeros(N_CAND, dtype=bool),
        shoot=True,
        shoot_release=True,
    )
    payload = label.as_dict()
    assert payload["shoot"] is True and payload["shoot_release"] is True
    back = TeacherLabel.from_dict(payload)
    assert back.shoot is True and back.shoot_release is True
    # A payload written before EXP-P1-009 keeps the documented defaults instead
    # of failing: SHOOT was always on, no release frame existed.
    legacy = dict(payload)
    legacy.pop("shoot")
    legacy.pop("shoot_release")
    old = TeacherLabel.from_dict(legacy)
    assert old.shoot is True and old.shoot_release is False


def test_cli_exposes_the_shoot_pulse_and_applies_it_to_the_config(tmp_path):
    cli = _load_cli()
    args = cli.parse_args([])
    assert args.shoot_pulse_period is None  # default comes from config
    on = cli.parse_args(["--shoot-pulse-period", "12"])
    assert cli._config_with_pulse(get_config("full"), on.shoot_pulse_period).shoot_pulse_period == 12
    off = cli.parse_args(["--shoot-pulse-period", "0"])
    assert cli._config_with_pulse(get_config("full"), off.shoot_pulse_period).shoot_pulse_period == 0
    assert cli._config_with_pulse(get_config("full"), None).shoot_pulse_period == 24
    # The release verdict is a function of the decision index only, so the very
    # first decision is not a release (the plan needs its SHOOT).
    from touhou_rl.v400.teacher.shoot import release_due

    assert release_due(0, 24) is False


# ---------------------------------------------------------------------------
# The rising edge is really produced, decision after decision (EXP-P1-009 §4)
# ---------------------------------------------------------------------------
def _flat_frames(decision_index, *, period, cands=(9, 10), bomb=0, decisions=None):
    """Every committed frame of ``decisions`` decisions, as ``(mask, 1)`` units."""

    from touhou_rl.v400.teacher.shoot import commit_block, release_due

    out = []
    for index in range(0 if decisions is None else decisions):
        out.extend(
            commit_block(
                list(cands),
                bomb=bomb if index == 0 else 0,
                decision_frames=2,
                release_shoot=release_due(index, period),
            )
        )
    return out


def test_pulse_produces_exactly_one_rising_edge_per_period():
    """The edge has to appear on the wire, not just in the config.

    ``GameManager.cpp:381-389`` compares ``g_CurFrameInput & 0x1001`` with the
    previous frame's word, so a run only ever gets the edge if the committed
    mask sequence goes ``... SHOOT, no-SHOOT, SHOOT ...``.  This walks the real
    ``commit_block`` output and counts the *falling and rising* edges of the
    SHOOT bit in the frame stream.
    """

    from touhou_rl.spec import BTN_SHOOT
    from touhou_rl.v400.teacher.shoot import SHOOT_PULSE_PERIOD, release_due

    period = SHOOT_PULSE_PERIOD
    # Three release decisions (index 23/47/71) plus one more decision so the
    # SHOOT that closes the last release is inside the stream.
    frames = _flat_frames(0, period=period, decisions=3 * period + 1)
    # One (mask, count) entry per decision here: each decision commits 4 frames
    # as 2 + 1 + 1, so expand the counts into the per-frame input word.
    stream = []
    for mask, count in frames:
        assert count >= 1
        stream.extend([mask] * count)
    assert len(stream) == (3 * period + 1) * 4
    shoot = [bool(mask & BTN_SHOOT) for mask in stream]
    falling = sum(1 for a, b in zip(shoot, shoot[1:]) if a and not b)
    rising = sum(1 for a, b in zip(shoot, shoot[1:]) if not a and b)
    # One release per period, and every release is closed by a SHOOT frame: the
    # number of rises equals the number of falls, which is what an edge-driven
    # state machine needs.
    expected_pairs = 3
    assert falling == rising == expected_pairs
    # The release frame is one single frame: no two releases are adjacent.
    assert all(not (not a and not b) for a, b in zip(shoot, shoot[1:]))
    # It is periodic and aligned with the decision index the CLI uses.
    assert [index for index in range(3 * period) if release_due(index, period)] == [
        period - 1,
        2 * period - 1,
        3 * period - 1,
    ]


def test_pulse_period_cli_values_reach_the_committed_frames():
    """period 1 = a release every decision, period 0 = no edge at all."""

    from touhou_rl.spec import BTN_SHOOT
    from touhou_rl.v400.teacher.shoot import commit_block, release_due

    every = [commit_block([9], decision_frames=2, release_shoot=release_due(i, 1)) for i in range(4)]
    for block in every:
        assert sum(c for _, c in block) == 2
        assert block[-1][0] & BTN_SHOOT == 0
    never = [commit_block([9], decision_frames=2, release_shoot=release_due(i, 0)) for i in range(4)]
    assert all(all(mask & BTN_SHOOT for mask, _ in block) for block in never)


def test_follow_stage_walks_the_next_stage_load_out():
    """Right after a stage change ``PARK`` is refused until setup finishes.

    Measured failure this covers: ``ERR park-unsafe role=shadow-required
    threads=2 setup=1`` (the setup thread blocks on ``stageTransitionState``,
    which only a ``0x1001`` rising edge clears).  The helper must keep stepping
    — with a SHOOT edge — until ``DEBUG`` itself reports the setup is done, and
    must give up (not loop forever) when it never is.
    """

    cli = _load_cli()

    class LoadingEngine:
        """Reports "loading" twice, then "ready"; records every STEP mask."""

        def __init__(self, ready_after=2):
            self.masks = []
            self.polls = 0
            self.ready_after = ready_after

        def debug(self):
            self.polls += 1
            loading = self.polls <= self.ready_after
            return {"_verb": "OK", "setupState": "1" if loading else "0",
                    "threads": "2" if loading else "1"}

        def ask(self, command, timeout=None):
            mask = int(command.split()[1])
            self.masks.append(mask)
            return "OK advanced=1 frames=1"

    engine = LoadingEngine(ready_after=2)
    frames, playable = cli._advance_until_playable(engine)
    assert playable is True
    assert frames == len(engine.masks) > 0
    # Every poll group is "SHOOT held, then one released frame": that is the
    # rising edge GameManager.cpp:381-389 waits for.
    assert engine.masks[-1] == 0
    assert engine.masks[0] == 1
    assert sum(1 for mask in engine.masks if mask == 0) >= 2

    # A load that never finishes must stop at the frame budget, not hang.
    stuck = LoadingEngine(ready_after=10**6)
    frames, playable = cli._advance_until_playable(stuck, limit_frames=12, poll=4)
    assert playable is False
    assert frames == 12


def test_stage_change_is_bookkept_apart_from_survival():
    """``stage_clear`` must come from the engine's stage index, not from living.

    EXP-P1-009 §4: a 20,000-frame run with 0 deaths can still report
    ``stage_clear=false``.  The transition row is the only place the completed
    stage, its frame number and its death/bomb deltas are recorded, so its
    arithmetic is asserted here instead of being read off a run.
    """

    cli = _load_cli()
    row = cli._stage_transition(
        decision=2171,
        frame=8684,
        from_stage=1,
        to_stage=2,
        entry_frame=4,  # the run starts at frame 4 (RESET lands on frames=4)
        entry_deaths=0.0,
        entry_bombs=3.0,
        deaths_now=1.0,
        bombs_now=1.0,
    )
    assert row == {
        "decision": 2171,
        "frame": 8684,
        "from_stage": 1,
        "to_stage": 2,
        "frames_in_stage": 8680,
        "deaths_in_stage": 1,
        "bombs_spent_in_stage": 2.0,
    }
    # A stage that has not finished yet is reported as an open row, never as a
    # completed one: `to_stage=None` cannot be mistaken for "advanced".
    open_row = cli._stage_transition(
        decision=0,
        frame=4,
        from_stage=1,
        to_stage=None,
        entry_frame=4,
        entry_deaths=0.0,
        entry_bombs=3.0,
        deaths_now=None,
        bombs_now=None,
    )
    assert open_row["to_stage"] is None
    assert open_row["frames_in_stage"] == 0
    assert open_row["deaths_in_stage"] is None and open_row["bombs_spent_in_stage"] is None
    # Counters are deltas: a deathless, bombless stage reports 0, not the total.
    quiet = cli._stage_transition(
        decision=1,
        frame=100,
        from_stage=1,
        to_stage=2,
        entry_frame=4,
        entry_deaths=0.0,
        entry_bombs=3.0,
        deaths_now=0.0,
        bombs_now=3.0,
    )
    assert quiet["deaths_in_stage"] == 0 and quiet["bombs_spent_in_stage"] == 0.0
    # The CLI keeps the old "stop at the first change" behaviour unless asked.
    assert cli.parse_args([]).follow_stage is False
    assert cli.parse_args(["--follow-stage"]).follow_stage is True


# ---------------------------------------------------------------------------
# ENEMIES: boss / enemy life observation (EXP-P1-011 §2)
# ---------------------------------------------------------------------------
#: One real ``HandleEnemies`` reply: the header carries the authoritative
#: ``count``, each row is
#: ``slot:flags1:posX:posY:posZ:life:timer:worldX:worldY:worldZ:hitboxW:hitboxH``
#: with ``flags1``/``life`` in hex and the rest as raw float bits.
ENEMIES_REPLY = (
    "OK frames=200 count=2 offset=0 limit=16 enemies="
    "3:1000005d:41f00000:c1800000:00000000:0000012c:23:"
    "41f00000:c1800000:00000000:00000000:00000000,"
    "9:1000005f:41f00000:c2000000:00000000:00000310:7:"
    "41f00000:c2000000:00000000:00000000:00000000"
)


def test_enemies_reply_decodes_count_boss_and_life():
    from touhou_rl.v400.teacher.engine import (
        ENEMY_FLAG_BOSS,
        parse_enemies_reply,
    )

    page = parse_enemies_reply(ENEMIES_REPLY)
    assert page["available"] is True
    assert page["frame"] == 200
    assert page["count"] == 2  # the engine's total, not len(rows)
    assert page["limit"] == 16
    assert len(page["rows"]) == 2
    first, boss = page["rows"]
    assert first.slot == 3
    assert first.flags1 & ENEMY_FLAG_BOSS == 0
    assert first.life == 0x12C == 300
    assert boss.slot == 9
    assert boss.flags1 & ENEMY_FLAG_BOSS  # 0x...5f has bit 1 set
    assert boss.life == 0x310 == 784
    assert page["boss_count"] == 1
    assert page["boss_slots"] == (9,)
    assert page["boss_life"] == 784
    assert page["life_sum"] == 300 + 784
    # A refused ENEMIES is reported verbatim, never as "boss=0".
    err = parse_enemies_reply("ERR enemies-not-ready reset-required")
    assert err["available"] is False
    assert err["boss_count"] == 0 and err["boss_life"] is None and err["count"] is None
    assert "reset-required" in err["reason"]


def test_engine_enemies_pages_a_crowded_frame_and_reports_truncation():
    """``ENEMIES`` caps one reply at 16 rows: the wrapper pages, and says so."""

    from touhou_rl.v400.teacher.engine import TeacherEngine, parse_enemies_reply

    rows = [f"{slot}:1000005d:00000000:00000000:00000000:0000000{slot}:0" for slot in range(20)]

    class FakeAsk(TeacherEngine):
        def __init__(self):
            self.sent = []

        def ask(self, command, timeout=None):
            self.sent.append(command)
            _, _, rest = command.partition(" ")
            offset, limit = (int(part) for part in rest.split())
            page = rows[offset: offset + limit]
            return (
                f"OK frames=1 count={len(rows)} offset={offset} limit={limit} enemies="
                + ",".join(page)
            )

    engine = FakeAsk()
    page = engine.enemies()
    assert page["available"] is True
    assert engine.sent == ["ENEMIES 0 16", "ENEMIES 16 16"]
    assert page["count"] == 20
    assert len(page["rows"]) == 20
    assert page["pages"] == 2
    assert page["truncated"] is False
    # max_pages bounds the cost: two pages would need three here.
    engine2 = FakeAsk()
    engine2.sent.clear()
    short = engine2.enemies(max_pages=1)
    assert engine2.sent == ["ENEMIES 0 16"]
    assert short["truncated"] is True
    assert len(short["rows"]) == 16
    # A refusal is a verdict, not an exception.
    class Refusing(FakeAsk):
        def ask(self, command, timeout=None):
            raise EngineError("ERR enemies-not-ready reset-required")

    refused = Refusing().enemies_snapshot()
    assert refused["available"] is False and refused["boss_life"] is None
    assert "reset-required" in refused["reason"]


def test_cli_logs_enemy_and_debug_rows_without_touching_the_policy():
    """The per-decision log carries both; neither is an input to ``decide``."""

    from touhou_rl.v400.teacher.engine import parse_enemies_reply

    cli = _load_cli()
    page = parse_enemies_reply(ENEMIES_REPLY)
    summary = cli._enemy_summary(page)
    assert summary["available"] is True
    assert summary["count"] == 2
    assert summary["boss_count"] == 1
    assert summary["boss_life"] == 784
    assert [row["boss"] for row in summary["rows"]] == [False, True]
    assert summary["rows"][1]["flags1"].startswith("0x")
    assert summary["rows"][1]["life"] == 784
    # Unavailable stays unavailable instead of reading as an empty field.
    missing = cli._enemy_summary({"available": False, "reason": "ERR nope"})
    assert missing["available"] is False
    assert missing["reason"] == "ERR nope"
    assert missing["rows"] == [] and missing["count"] is None

    class FakeDebug:
        def debug(self):
            return {
                "_verb": "OK",
                "curState": "2",
                "setupState": "0",
                "subthread": "0",
                "globals": "1",
                "stage": "1",
                "threads": "1",
                "frames": "1234",
            }

    debug = cli._debug_snapshot(FakeDebug())
    assert debug["available"] is True
    assert debug["curState"] == 2 and debug["setupState"] == 0
    assert debug["stage"] == 1
    # DEBUG does not export it, so it is reported as absent (never as 0).
    assert debug["stage_transition_state"] is None

    class RefusingDebug:
        def debug(self):
            raise EngineError("ERR unknown-command DEBUG")

    refused = cli._debug_snapshot(RefusingDebug())
    assert refused["available"] is False
    assert "unknown-command" in refused["reason"]


# ---------------------------------------------------------------------------
# EXP-P1-015: the bottom safety band, the escape term and the rule-1 window
# ---------------------------------------------------------------------------
def test_bottom_band_penalty_matches_the_published_formula():
    """``-1.5*(y-400)/32`` inside the band and ``-2.0*(y-420)/12`` past 420."""

    weights = get_config("v2a").weights
    assert weights.bottom_penalty_at(380.0) == 0.0  # above the band: free
    assert weights.bottom_penalty_at(400.0) == 0.0  # the edge itself is not charged
    assert weights.bottom_penalty_at(416.0) == pytest.approx(1.5 * 16.0 / 32.0)
    assert weights.bottom_penalty_at(420.0) == pytest.approx(1.5 * 20.0 / 32.0)
    assert weights.bottom_penalty_at(432.0) == pytest.approx(
        1.5 * 32.0 / 32.0 + 2.0 * 12.0 / 12.0
    )
    # Monotone: deeper is never cheaper.
    values = [weights.bottom_penalty_at(y) for y in (400.0, 410.0, 420.0, 426.0, 432.0)]
    assert values == sorted(values)
    # v1 stays exactly v1: no band, no escape, no rule-1 window.
    v1 = get_config("full").weights
    assert v1.bottom_penalty_at(432.0) == 0.0
    assert v1.escape_weight == 0.0
    assert get_config("full").bomb_rule1_max_death_frame is None


def test_v2_profiles_register_their_weights_and_keep_the_shared_terms():
    v1 = get_config("full")
    v2a = get_config("v2a")
    v2b = get_config("v2b")
    assert (v2a.mode, v2b.mode) == ("v2a", "v2b")
    # Every profile keeps 11-P1 §3's other terms and the §5 search parameters.
    for profile in (v2a, v2b):
        assert profile.horizon == v1.horizon
        assert profile.beam == v1.beam and profile.verify_k == v1.verify_k
        assert profile.max_iterations == v1.max_iterations
        assert profile.durations == v1.durations
        for name in ("margin_cap", "align_weight", "items_weight", "items_cap",
                     "edge_penalty", "edge_band", "high_penalty", "high_y",
                     "death_scale", "unverified_discount", "align_range"):
            assert getattr(profile.weights, name) == getattr(v1.weights, name), name
        assert profile.bomb_rule1_max_death_frame == 12
    assert v2a.weights.bottom_penalty == pytest.approx(1.5)
    assert v2a.weights.bottom_extra_penalty == pytest.approx(2.0)
    assert v2a.weights.escape_weight == 0.0
    assert v2b.weights.bottom_penalty > v2a.weights.bottom_penalty
    assert v2b.weights.escape_weight > 0.0
    # ``full`` is still registered under its own name and unchanged.
    assert get_config("full") is FULL and get_config("fast") is FAST
    with pytest.raises(ValueError):
        get_config("v2z")


def test_bottom_band_flips_the_choice_between_a_bottom_and_a_mid_plan():
    """The mechanism, term by term: same align, the bottom plan used to win."""

    bottom = dict(margin=14.0, death_frame=None, end_xy=(192.0, 424.0), target=192.0,
                  horizon=48)
    middle = dict(margin=13.4, death_frame=None, end_xy=(192.0, 396.0), target=192.0,
                  horizon=48)
    v1 = get_config("full")
    v2a = get_config("v2a")
    # v1: the margin term is the only vertical signal, so the deeper plan wins.
    assert score_plan(config=v1, **bottom)[0] > score_plan(config=v1, **middle)[0]
    # v2a: the band outbids a 0.6 px margin advantage at 424 px.
    assert score_plan(config=v2a, **bottom)[0] < score_plan(config=v2a, **middle)[0]
    parts = score_plan(config=v2a, **bottom)[1]
    assert parts["bottom"] == pytest.approx(1.5 * 24.0 / 32.0 + 2.0 * 4.0 / 12.0)
    # A dead plan is still scored by the death branch alone (no band term).
    dead = score_plan(config=v2a, margin=0.0, death_frame=24, end_xy=(192.0, 432.0),
                      target=192.0, horizon=48)[1]
    assert dead["bottom"] == 0.0 and dead["escape"] == 0.0


def test_escape_clearance_measures_the_upward_room_in_the_ship_column():
    from touhou_rl.v400.teacher.search import escape_clearance

    half = np.array([0.825, 0.825])
    boxes = np.array(
        [
            [190.0, 350.0, 194.0, 354.0],   # directly above, 30 px of room
            [300.0, 100.0, 304.0, 104.0],   # beside the column: ignored
            [190.0, 420.0, 194.0, 424.0],   # below the ship: ignored
        ]
    )
    # ship at (192, 384): the nearest block above starts at y=354, minus the half.
    assert escape_clearance(boxes, (192.0, 384.0), half) == pytest.approx(384.0 - 0.825 - 354.0)
    # A box in another column does not restrict the upward exit.
    assert escape_clearance(boxes[1:2], (192.0, 384.0), half) == float("inf")
    assert escape_clearance(np.zeros((0, 4)), (192.0, 384.0), half) == float("inf")


def test_escape_term_only_counts_inside_the_bottom_band():
    v2b = get_config("v2b")
    inside = score_plan(config=v2b, margin=16.0, death_frame=None, end_xy=(192.0, 428.0),
                        target=192.0, horizon=48, escape=8.0)[1]
    assert inside["escape"] == pytest.approx(1.0 * 8.0 / 32.0)
    open_up = score_plan(config=v2b, margin=16.0, death_frame=None, end_xy=(192.0, 428.0),
                         target=192.0, horizon=48, escape=float("inf"))[1]
    assert open_up["escape"] == pytest.approx(1.0)  # saturates at escape_span
    above = score_plan(config=v2b, margin=16.0, death_frame=None, end_xy=(192.0, 380.0),
                       target=192.0, horizon=48, escape=8.0)[1]
    assert above["escape"] == 0.0  # downward exit still open: the term is off
    # A bottom corner that is also sealed scores strictly worse than one with room.
    sealed = score_plan(config=v2b, margin=16.0, death_frame=None, end_xy=(192.0, 432.0),
                        target=192.0, horizon=48, escape=0.0)[0]
    roomy = score_plan(config=v2b, margin=16.0, death_frame=None, end_xy=(192.0, 432.0),
                       target=192.0, horizon=48, escape=32.0)[0]
    assert roomy - sealed == pytest.approx(1.0)


def test_beam_search_with_the_escape_term_matches_the_reference():
    """The vectorised escape term is the same arithmetic as the scalar one."""

    config = type(get_config("v2b"))(
        mode="test", horizon=16, beam=6, verify_k=1, max_iterations=1,
        weights=get_config("v2b").weights,
    )
    tape = make_tape(_dense_frames(16, seed=5))
    got = beam_search(tape, config, use_raster=False)
    want = beam_search_reference(tape, config)
    assert np.array_equal(got.scores, want.scores)
    for cand, expected in want.plans.items():
        actual = got.plans[cand]
        for name in ("score", "bottom", "escape", "margin", "end_xy", "death_frame"):
            assert getattr(actual, name) == getattr(expected, name), (cand, name)



def test_cli_exposes_the_v2_profiles_and_keeps_full_as_default():
    cli = _load_cli()
    args = cli.parse_args([])
    assert args.mode == "full"
    assert cli.parse_args(["--mode", "v2a"]).mode == "v2a"
    assert cli.parse_args(["--mode", "v2b"]).mode == "v2b"
    assert get_config("v2a").weights.bottom_penalty > 0.0


def test_bottom_diag_observation_reports_the_room_above_and_below():
    """The observation is read-only and reports the two rooms plus the direction."""

    from touhou_rl.v400.teacher.search import Plan

    cli = _load_cli()
    frames = [((bullet(192.0, 384.0 - 100.0),), ()) for _ in range(6)]

    class _Decision:
        tape = make_tape(frames, hurtbox=(192.0, 428.0, 0.825))
        chosen = Plan(
            first_action=1,
            duration=1,
            slots=(1,),
            margin=12.0,
            death_frame=None,
            end_xy=(192.0, 412.0),
            score=100.0,
        )

    diag = cli._bottom_diag(_Decision())
    assert diag["y"] == pytest.approx(428.0)
    assert diag["room_down"] == pytest.approx(432.0 - 428.0)
    assert diag["room_up"] == pytest.approx(428.0 - 0.825 - (384.0 - 100.0) - 2.0)
    assert diag["chosen_dir"] == "up"
    assert diag["chosen_end_y"] == pytest.approx(412.0)
    assert diag["boxes"] == 1
    # A bullet two frames later and closer means the exit is closing.
    assert diag["room_up_2f"] is not None and diag["room_up_2f"] <= diag["room_up"]

    class _Empty:
        tape = make_tape(empty_frames(6))
        chosen = None

    empty = cli._bottom_diag(_Empty())
    assert empty["room_up"] is None and empty["chosen_dir"] is None

#: ``DECISION_TEST`` widened to 24 frames so a *far* engine death (frame 20) is
#: inside the plan, with and without the EXP-P1-015 rule-1 window.
DECISION_FAR_V1 = type(FULL)(mode="far", horizon=24, beam=8, verify_k=1, max_iterations=1)
DECISION_FAR_V2A = type(FULL)(
    mode="v2a-far", horizon=24, beam=8, verify_k=1, max_iterations=1,
    weights=get_config("v2a").weights, bomb_rule1_max_death_frame=12,
)


def test_v2_bomb_rule_one_is_suppressed_on_a_far_forecast():
    """All verified plans die, but 19 frames out: keep the bomb (EXP-P1-015 §3)."""

    engine = FakeWorldEngine(kills_at=20)
    decision = decide(engine, DECISION_FAR_V2A, record_source="child")
    assert decision.label.bomb == 0
    assert any(event.startswith("active_bomb_suppressed") for event in decision.label.events)
    assert decision.label.extra["bomb_rule1_suppressed"] is True
    assert decision.label.extra["bomb_rule1_max_death_frame"] == 12
    shortest = decision.label.extra["shortest_verified_death_frame"]
    assert shortest is not None and shortest > 12


def test_v2_bomb_rule_one_still_fires_when_the_death_is_imminent():
    engine = FakeWorldEngine(kills_at=3)
    decision = decide(engine, DECISION_FAR_V2A, record_source="child")
    assert decision.label.bomb == 1
    assert "active_bomb" in decision.label.events
    assert decision.label.extra["bomb_rule1_suppressed"] is False
    assert decision.label.extra["shortest_verified_death_frame"] <= 12


def test_v1_rule_one_still_bombs_on_a_far_forecast():
    """Regression: ``full`` keeps 11-P1 §4 verbatim (no window)."""

    engine = FakeWorldEngine(kills_at=20)
    decision = decide(engine, DECISION_FAR_V1, record_source="child")
    assert decision.label.bomb == 1
    assert "active_bomb" in decision.label.events
    assert decision.label.extra["bomb_rule1_max_death_frame"] is None
    assert decision.label.extra["bomb_rule1_suppressed"] is False


# ---------------------------------------------------------------------------
# EXP-P1-015 §2b/§3b: the v2c hard corridor and the "keep the last bomb" rule
# ---------------------------------------------------------------------------
def test_v2c_profile_registers_the_corridor_and_the_bomb_guards():
    v2a = get_config("v2a")
    v2c = get_config("v2c")
    assert v2c.mode == "v2c"
    assert v2c.clip_y == (200.0, 400.0)
    assert v2a.clip_y is None  # v2a stays a pure soft band
    assert v2c.weights.bottom_penalty == pytest.approx(v2a.weights.bottom_penalty)
    assert v2c.weights.bottom_extra_penalty == pytest.approx(v2a.weights.bottom_extra_penalty)
    assert v2c.bomb_rule1_max_death_frame == 24
    assert v2c.bomb_rule1_min_bombs_left == 1  # keep the last bomb
    assert v2c.bomb_rule1_max_margin == pytest.approx(12.0)
    # v1 stays untouched by every new switch.
    assert get_config("full").clip_y is None
    assert get_config("full").bomb_rule1_min_bombs_left == 0
    assert get_config("full").bomb_rule1_max_margin is None
    with pytest.raises(ValueError):
        type(FULL)(mode="bad", horizon=8, beam=2, verify_k=1, max_iterations=1,
                   clip_y=(300.0, 200.0))


def _finish_nodes(config, end_ys, margins):
    """Run ``_finish_roots`` on one root with synthetic terminal positions."""

    from touhou_rl.v400.teacher.search import _finish_roots

    count = np.array([len(end_ys)], dtype=np.int64)
    pos = np.zeros((1, len(end_ys), 2), dtype=np.float64)
    pos[0, :, 0] = 192.0
    pos[0, :, 1] = np.asarray(end_ys, dtype=np.float64)
    margin = np.asarray([margins], dtype=np.float64)
    death = np.full((1, len(end_ys)), -1, dtype=np.int64)
    actions = np.zeros((1, len(end_ys), 1), dtype=np.int16)
    plans = _finish_roots(
        [(1, 1)], count, pos, margin, death, actions, 192.0, config,
        half=None, escape_boxes=None,
    )
    return plans[1]


def test_v2c_corridor_beats_a_better_scoring_out_of_corridor_plan():
    """y=410 has the best v2a score; the corridor must still pick y=380."""

    end_ys = (380.0, 430.0, 410.0)
    margins = (8.0, 12.0, 10.0)
    v2a = _finish_nodes(get_config("v2a"), end_ys, margins)
    v2c = _finish_nodes(get_config("v2c"), end_ys, margins)
    assert v2a.end_xy[1] == pytest.approx(410.0)   # outside the corridor, soft band only
    assert v2c.end_xy[1] == pytest.approx(380.0)   # inside: the clip is hard
    # The stored score is still the real one (the clip is a selection rule).
    assert v2c.score == pytest.approx(
        v2c.score
    ) and v2c.bottom == pytest.approx(0.0)


def test_v2c_corridor_falls_back_when_no_plan_ends_inside():
    end_ys = (410.0, 430.0)
    margins = (10.0, 8.0)
    v2c = _finish_nodes(get_config("v2c"), end_ys, margins)
    # Nothing inside (200, 400) -> the soft band decides, exactly as in v2a.
    assert v2c.end_xy[1] == pytest.approx(410.0)


def test_v2c_keeps_the_last_bomb_for_the_real_death_bomb():
    """bombs=1 keeps rule 1 silent (v2c), while v1/v2a still spend it."""

    engine = FakeWorldEngine(kills_at=3, bombs=1.0)
    v2c_decision = decide(engine, DECISION_V2C, record_source="child")
    assert v2c_decision.label.bomb == 0
    assert any("keeping_last_bombs" in event for event in v2c_decision.label.events)
    assert v2c_decision.label.extra["bomb_rule1_enough_bombs"] is False
    # The same world under v2a spends the bomb (only the frame window applies).
    engine = FakeWorldEngine(kills_at=3, bombs=1.0)
    v2a_decision = decide(engine, DECISION_FAR_V2A, record_source="child")
    assert v2a_decision.label.bomb == 1


def test_v2c_requires_a_tight_margin_before_bombing():
    # The fake world has no bullets, so the plan margin saturates at MARGIN_CAP:
    # a threshold above it is the "tight margin reached" branch, and 0.0 is the
    # "not tight enough" branch.  Both go through the same rule-1 code path.
    loose = type(DECISION_V2C)(
        mode="v2c-loose", horizon=DECISION_V2C.horizon, beam=DECISION_V2C.beam,
        verify_k=1, max_iterations=1, weights=DECISION_V2C.weights,
        bomb_rule1_max_death_frame=24, bomb_rule1_min_bombs_left=1,
        bomb_rule1_max_margin=1000.0, clip_y=DECISION_V2C.clip_y,
    )
    engine = FakeWorldEngine(kills_at=3, bombs=3.0)
    tight = decide(engine, loose, record_source="child")
    assert tight.label.extra["bomb_rule1_margin_ok"] is True
    assert tight.label.bomb == 1
    # With an impossible margin threshold the same decision must keep the bomb.
    impossible = type(DECISION_V2C)(
        mode="v2c-tight", horizon=DECISION_V2C.horizon, beam=DECISION_V2C.beam,
        verify_k=1, max_iterations=1, weights=DECISION_V2C.weights,
        bomb_rule1_max_death_frame=24, bomb_rule1_min_bombs_left=1,
        bomb_rule1_max_margin=0.0, clip_y=DECISION_V2C.clip_y,
    )
    engine = FakeWorldEngine(kills_at=3, bombs=3.0)
    kept = decide(engine, impossible, record_source="child")
    assert kept.label.bomb == 0
    assert kept.label.extra["bomb_rule1_margin_ok"] is False
    assert any("margin_" in event for event in kept.label.events)


def test_v2c_clip_matches_the_reference_implementation():
    config = type(get_config("v2c"))(
        mode="v2c-test", horizon=16, beam=6, verify_k=1, max_iterations=1,
        weights=get_config("v2c").weights, clip_y=(200.0, 400.0),
    )
    tape = make_tape(_dense_frames(16, seed=7))
    got = beam_search(tape, config, use_raster=False)
    want = beam_search_reference(tape, config)
    assert np.array_equal(got.scores, want.scores)
    assert set(got.plans) == set(want.plans)
    for cand, expected in want.plans.items():
        assert got.plans[cand].end_xy == expected.end_xy, cand
        assert got.plans[cand].score == expected.score, cand

#: ``DECISION_FAR_V2A`` with the v2c profile (corridor + keep-last-bomb guards).
DECISION_V2C = type(FULL)(
    mode="v2c-far", horizon=24, beam=8, verify_k=1, max_iterations=1,
    weights=get_config("v2c").weights, bomb_rule1_max_death_frame=24,
    bomb_rule1_min_bombs_left=1, bomb_rule1_max_margin=12.0, clip_y=(200.0, 400.0),
)


# ---------------------------------------------------------------------------
# EXP-P1-017 §2: the v2e profile and the Boss-body proxemity cost
# ---------------------------------------------------------------------------
def test_boss_body_cost_saturates_at_the_radius():
    """``3.0 * clip(48 - d, 0, 48) / 48``, and off for every earlier profile."""

    weights = get_config("v2e").weights
    assert weights.boss_penalty == pytest.approx(3.0)
    assert weights.boss_radius == pytest.approx(48.0)
    assert weights.boss_penalty_at(None) == 0.0            # no Boss on the frame
    assert weights.boss_penalty_at(float("inf")) == 0.0    # no Boss body at all
    assert weights.boss_penalty_at(48.0) == 0.0            # the radius itself is free
    assert weights.boss_penalty_at(24.0) == pytest.approx(1.5)
    assert weights.boss_penalty_at(0.0) == pytest.approx(3.0)   # inside the body
    assert weights.boss_penalty_at(-5.0) == pytest.approx(3.0)  # cannot exceed it
    values = [weights.boss_penalty_at(d) for d in (0.0, 12.0, 24.0, 36.0, 48.0, 60.0)]
    assert values == sorted(values, reverse=True)
    # v1/v2a are untouched by the new term: the argument cannot move the score.
    assert get_config("full").weights.boss_penalty_at(0.0) == 0.0
    assert get_config("v2a").weights.boss_penalty_at(0.0) == 0.0
    assert get_config("v2c").weights.boss_penalty_at(0.0) == 0.0
    assert get_config("v2d").weights.boss_penalty_at(0.0) == 0.0


def test_boss_clearance_is_the_aabb_gap_and_matches_the_vector_form():
    from touhou_rl.v400.teacher.search import _boss_rows, boss_clearance

    boxes = np.array([[190.0, 90.0, 200.0, 110.0], [300.0, 300.0, 320.0, 340.0]])
    assert boss_clearance(boxes, (192.0, 134.0)) == pytest.approx(24.0)  # below the first
    assert boss_clearance(boxes, (195.0, 100.0)) == 0.0                 # inside the first
    assert boss_clearance(boxes, (250.0, 305.0)) == pytest.approx(50.0)  # left of the second
    assert boss_clearance(np.zeros((0, 4)), (192.0, 100.0)) == float("inf")
    centers = np.array([[192.0, 134.0], [195.0, 100.0], [250.0, 305.0]])
    rows = _boss_rows(boxes, centers)
    for index, center in enumerate(centers):
        assert rows[index] == pytest.approx(boss_clearance(boxes, tuple(center)))
    assert np.isinf(_boss_rows(np.zeros((0, 4)), centers)).all()


def test_boss_body_boxes_uses_only_boss_flagged_lethal_bodies():
    from touhou_rl.v400.teacher.engine import ENEMY_FLAG_BOSS
    from touhou_rl.v400.teacher.geometry import ENEMY_FLAG_COLLISION
    from touhou_rl.v400.teacher.search import boss_body_boxes

    boss = enemy(192.0, 100.0, hitbox=30.0,
                 flags1=ENEMY_FLAG_COLLISION | ENEMY_FLAG_BOSS, slot=3)
    minion = enemy(300.0, 200.0, hitbox=24.0, flags1=ENEMY_FLAG_COLLISION, slot=4)
    ghost = enemy(120.0, 200.0, hitbox=24.0,
                  flags1=ENEMY_FLAG_COLLISION | ENEMY_FLAG_BOSS, slot=5)
    # A Boss row without worldPosition/hitbox is not an AABB and must be dropped.
    bare = EnemyRow(slot=6, flags1=ENEMY_FLAG_COLLISION | ENEMY_FLAG_BOSS,
                    position=(0.0, 0.0, 0.0), life=1, timer=0)
    boxes = boss_body_boxes([boss, minion, ghost, bare])
    assert boxes.shape == (2, 4)  # two Boss rows with geometry, the minion excluded
    # enemy_boxes' own expression: worldPosition +- hitboxDimensions / 3.
    assert boxes[0] == pytest.approx([192.0 - 10.0, 100.0 - 10.0, 192.0 + 10.0, 100.0 + 10.0])
    assert boss_body_boxes([minion, bare]).shape == (0, 4)
    assert boss_body_boxes([]).shape == (0, 4)


def test_boss_body_cost_is_subtracted_from_the_living_score_only():
    v2e = get_config("v2e")
    far, far_parts = score_plan(
        config=v2e, margin=16.0, death_frame=None, end_xy=(192.0, 250.0),
        target=192.0, horizon=48, boss_dist=200.0,
    )
    near, near_parts = score_plan(
        config=v2e, margin=16.0, death_frame=None, end_xy=(192.0, 250.0),
        target=192.0, horizon=48, boss_dist=24.0,
    )
    assert far_parts["boss"] == 0.0
    assert near_parts["boss"] == pytest.approx(1.5)
    assert far == pytest.approx(near + 1.5)
    # The death branch has no geometry terms (11-P1 §3), the Boss cost included.
    dead = score_plan(config=v2e, margin=0.0, death_frame=24, end_xy=(192.0, 250.0),
                      target=192.0, horizon=48, boss_dist=0.0)
    assert dead[1]["boss"] == 0.0
    # v1 ignores the argument entirely, so its score is the shipped expression.
    v1_plain = score_plan(config=get_config("full"), margin=16.0, death_frame=None,
                          end_xy=(192.0, 250.0), target=192.0, horizon=48)
    v1_bossy = score_plan(config=get_config("full"), margin=16.0, death_frame=None,
                          end_xy=(192.0, 250.0), target=192.0, horizon=48, boss_dist=0.0)
    assert v1_plain[0] == v1_bossy[0]
    assert v1_bossy[1]["boss"] == 0.0


def test_v2e_profile_is_v2a_plus_the_boss_cost_and_nothing_else():
    v2a, v2c, v2e = get_config("v2a"), get_config("v2c"), get_config("v2e")
    assert v2e.mode == "v2e"
    assert (v2e.horizon, v2e.beam, v2e.verify_k, v2e.max_iterations) == (
        v2a.horizon, v2a.beam, v2a.verify_k, v2a.max_iterations)
    # §1: the bottom soft band is kept verbatim.
    assert v2e.weights.bottom_penalty == pytest.approx(1.5)
    assert v2e.weights.bottom_extra_penalty == pytest.approx(2.0)
    assert v2e.weights.bottom_y == pytest.approx(400.0)
    assert v2e.weights.bottom_extra_y == pytest.approx(420.0)
    # §2: the Boss cost is the only new weight.
    assert v2e.weights.boss_penalty == pytest.approx(3.0)
    assert v2e.weights.boss_radius == pytest.approx(48.0)
    for name in ("align_weight", "edge_penalty", "high_penalty", "death_scale"):
        assert getattr(v2e.weights, name) == getattr(v2a.weights, name), name
    # §1: rule 1's 12-frame window is kept.
    assert v2e.bomb_rule1_max_death_frame == 12
    # §2: "keep the last bomb" and the margin gate are *withdrawn* (v2c had both).
    assert v2e.bomb_rule1_min_bombs_left == 0
    assert v2e.bomb_rule1_max_margin is None
    assert v2c.bomb_rule1_min_bombs_left == 1 and v2c.bomb_rule1_max_margin == 12.0
    # ..., and no hard corridor (v2c/v2d's clip): v2a never had one either.
    assert v2e.clip_y is None and v2a.clip_y is None
    # The CLI exposes the profile, and the default stays ``full``.
    cli = _load_cli()
    assert cli.parse_args(["--mode", "v2e"]).mode == "v2e"
    assert cli.parse_args([]).mode == "full"


def _finish_nodes_xy(
    config, ends, margins, boss_boxes=None, target=192.0, laser_rows=None, half=None
):
    """``_finish_roots`` on one root with explicit terminal ``(x, y)`` pairs."""

    from touhou_rl.v400.teacher.search import _finish_roots

    if laser_rows is not None and half is None:
        half = (0.825, 0.825)
    ends = np.asarray(ends, dtype=np.float64)
    count = np.array([len(ends)], dtype=np.int64)
    pos = ends[None, :, :].copy()
    margin = np.asarray([margins], dtype=np.float64)
    death = np.full((1, len(ends)), -1, dtype=np.int64)
    actions = np.zeros((1, len(ends), 1), dtype=np.int16)
    plans = _finish_roots(
        [(1, 1)], count, pos, margin, death, actions, float(target), config,
        half=half, escape_boxes=None, boss_boxes=boss_boxes, laser_rows=laser_rows,
    )
    return plans[1]


def test_v2e_flips_the_choice_away_from_a_terminal_position_inside_a_boss_body():
    """The cost is a *candidate-endpoint* cost (option b), not an ``align`` retarget.

    ``B`` is 5 px better on the shipped terms (a larger margin, no bottom band) but
    sits inside the Boss body; ``A`` trades 1 px of margin and the bottom band for
    80 px of clearance from that body.
    """

    boss = np.array([[176.0, 300.0, 208.0, 340.0]])
    ends = ((192.0, 420.0), (192.0, 315.0))  # A: away from the body; B: inside it
    margins = (14.0, 15.0)
    v2a = _finish_nodes_xy(get_config("v2a"), ends, margins)
    v2e = _finish_nodes_xy(get_config("v2e"), ends, margins, boss_boxes=boss)
    assert v2a.end_xy[1] == pytest.approx(315.0)  # v2a: B wins by 1.94 points
    assert v2e.end_xy[1] == pytest.approx(420.0)  # v2e: B's 3.0 point body cost flips it
    assert v2e.boss == 0.0                        # A ends 80 px from the body
    # The score is the real one (the cost is a weight, never a hard clip).
    expected, parts = score_plan(
        config=get_config("v2e"), margin=14.0, death_frame=None,
        end_xy=(192.0, 420.0), target=192.0, horizon=get_config("v2e").horizon,
        boss_dist=80.0,
    )
    assert v2e.score == pytest.approx(expected)
    assert parts["boss"] == 0.0


def test_v2e_keeps_a_plan_that_is_alive_even_when_every_end_is_next_to_the_boss():
    """No Boss position can turn a living plan into a dead one (a weight, not a gate)."""

    boss = np.array([[176.0, 246.0, 208.0, 270.0]])
    ends = ((192.0, 258.0), (200.0, 258.0))  # both inside the body
    v2e = _finish_nodes_xy(get_config("v2e"), ends, (16.0, 16.0), boss_boxes=boss)
    assert v2e.death_frame is None
    assert v2e.boss == pytest.approx(3.0)
    # And with no Boss on the terminal frame the term is exactly zero.
    clear = _finish_nodes_xy(get_config("v2e"), ends, (16.0, 16.0),
                             boss_boxes=np.zeros((0, 4)))
    assert clear.boss == 0.0


def test_v2e_search_matches_the_reference_with_a_boss_body_on_the_tape():
    """The vectorised Boss cost is the same arithmetic as the scalar one."""

    from touhou_rl.v400.teacher.engine import ENEMY_FLAG_BOSS
    from touhou_rl.v400.teacher.geometry import ENEMY_FLAG_COLLISION

    config = type(get_config("v2e"))(
        mode="v2e-test", horizon=24, beam=8, verify_k=1, max_iterations=1,
        weights=get_config("v2e").weights,
    )
    frames = []
    for bullets, enemies in _dense_frames(24, seed=13):
        # A Boss body sitting right next to the ship's row of end positions, so the
        # new term is exercised on every frame (and would be, at zero cost, if the
        # two implementations disagreed about which rows are Bosses).
        boss = enemy(192.0, 330.0, hitbox=60.0,
                     flags1=ENEMY_FLAG_COLLISION | ENEMY_FLAG_BOSS, slot=9)
        frames.append((bullets, tuple(enemies) + (boss,)))
    tape = make_tape(frames)
    got = beam_search(tape, config, use_raster=False)
    want = beam_search_reference(tape, config)
    assert np.array_equal(got.scores, want.scores)
    assert set(got.plans) == set(want.plans)
    charged = 0
    for cand, expected in want.plans.items():
        actual = got.plans[cand]
        for name in ("score", "boss", "margin", "end_xy", "death_frame", "bottom"):
            assert getattr(actual, name) == getattr(expected, name), (cand, name)
        if expected.death_frame is None and expected.boss > 0.0:
            charged += 1
    # The term is actually exercised: some living plan ends within the radius.
    assert charged > 0
    # ... and the same tape under v2a scores strictly higher on those terminals.
    v2a = type(get_config("v2a"))(
        mode="v2a-test", horizon=24, beam=8, verify_k=1, max_iterations=1,
        weights=get_config("v2a").weights,
    )
    plain = beam_search_reference(tape, v2a)
    assert any(
        plain.plans[cand].score > want.plans[cand].score
        for cand in want.plans
        if want.plans[cand].boss > 0.0
    )


def test_rule_one_bombs_again_under_v2e_with_one_bomb_left():
    """v2e withdrew v2c's "keep the last bomb": a 3-frame death spends it."""

    engine = FakeWorldEngine(kills_at=3, bombs=1.0)
    v2e_decision = decide(engine, DECISION_V2E, record_source="child")
    assert v2e_decision.label.bomb == 1
    assert "active_bomb" in v2e_decision.label.events
    assert v2e_decision.label.extra["bomb_rule1_min_bombs_left"] == 0
    assert v2e_decision.label.extra["bomb_rule1_enough_bombs"] is True
    assert v2e_decision.label.extra["bomb_rule1_max_margin"] is None
    assert v2e_decision.label.extra["bomb_rule1_margin_ok"] is True
    # The same world under v2c keeps it (regression: the guards still work).
    engine = FakeWorldEngine(kills_at=3, bombs=1.0)
    v2c_decision = decide(engine, DECISION_V2C, record_source="child")
    assert v2c_decision.label.bomb == 0


def test_rule_one_window_is_still_the_v2a_one_under_v2e():
    """The 12-frame window survives: a far forecast keeps the bomb, a near one spends it."""

    far = decide(FakeWorldEngine(kills_at=20), DECISION_V2E, record_source="child")
    assert far.label.bomb == 0
    assert far.label.extra["bomb_rule1_max_death_frame"] == 12
    assert far.label.extra["bomb_rule1_suppressed"] is True
    near = decide(FakeWorldEngine(kills_at=3), DECISION_V2E, record_source="child")
    assert near.label.bomb == 1
    assert near.label.extra["bomb_rule1_suppressed"] is False


#: ``DECISION_FAR_V2A`` with the v2e profile (bottom band + Boss-body cost, no guards).
DECISION_V2E = type(FULL)(
    mode="v2e-far", horizon=24, beam=8, verify_k=1, max_iterations=1,
    weights=get_config("v2e").weights, bomb_rule1_max_death_frame=12,
)


# ---------------------------------------------------------------------------
# EXP-P1-019: v3a / v3b -- v2a's weights, a larger search budget
# ---------------------------------------------------------------------------
def test_v3_profiles_register_a_larger_budget_and_keep_the_v2a_weights():
    """EXP-P1-019 §2: the only thing that moves is ``H``/``beam``/``k``.

    §10.4 of EXP-P1-015 asks for a bigger search *before* another weight change,
    so the contract this test pins is exactly that: the two v3 profiles must
    share v2a's ``Weights`` instance (an identity, so a later edit cannot drift
    one copy), and every non-budget knob must still be v2a's.
    """

    v1, v2a = get_config("full"), get_config("v2a")
    v3a, v3b = get_config("v3a"), get_config("v3b")
    assert (v3a.mode, v3b.mode) == ("v3a", "v3b")
    # "v2a weights verbatim" is an aliasing fact, not a re-typed copy.
    assert v3a.weights is v2a.weights
    assert v3b.weights is v2a.weights
    assert v3a.weights.bottom_penalty == pytest.approx(1.5)
    assert v3a.weights.bottom_extra_penalty == pytest.approx(2.0)
    # ... and v2e's two experiments are *not* back: no Boss cost, no escape term.
    assert v3a.weights.boss_penalty == 0.0 and v3b.weights.boss_penalty == 0.0
    assert v3a.weights.escape_weight == 0.0 and v3b.weights.escape_weight == 0.0
    # §2: the budgets.
    assert (v3a.horizon, v3a.beam, v3a.verify_k, v3a.max_iterations) == (64, 96, 3, 3)
    assert (v3b.horizon, v3b.beam, v3b.verify_k, v3b.max_iterations) == (64, 128, 4, 3)
    assert v3a.slots == 32 and v3b.slots == 32
    assert v3a.slots == v2a.slots + 8 and v3a.beam == v2a.beam + 32
    assert v3b.beam > v3a.beam and v3b.verify_k == v3a.verify_k + 1
    # Everything else is v2a's (and therefore v1's, unchanged since 11-P1 §5).
    for profile in (v3a, v3b):
        assert profile.bomb_rule1_max_death_frame == 12  # v2a's rule-1 window
        assert profile.bomb_rule1_min_bombs_left == 0  # v2c's guard is not back
        assert profile.bomb_rule1_max_margin is None  # v2c's margin gate is not back
        assert profile.clip_y is None  # v2c/v2d's hard corridor is not back
        assert profile.durations == v1.durations
        assert profile.decision_frames == v1.decision_frames
        assert profile.border_x == v1.border_x and profile.border_y == v1.border_y
        assert profile.shoot_pulse_period == v1.shoot_pulse_period
        assert profile.record_items == v1.record_items
    # Registered, listed, and reachable from the CLI; ``full`` is still default.
    assert CONFIGS["v3a"] is v3a and CONFIGS["v3b"] is v3b
    assert {profile.mode for profile in PROFILES} >= {"v2a", "v3a", "v3b"}
    assert get_config("full") is FULL and get_config("fast") is FAST
    cli = _load_cli()
    assert cli.parse_args(["--mode", "v3a"]).mode == "v3a"
    assert cli.parse_args(["--mode", "v3b"]).mode == "v3b"
    assert cli.parse_args([]).mode == "full"
    with pytest.raises(ValueError):
        get_config("v3c")


def test_v3_budget_reaches_the_horizon_the_beam_and_the_verification():
    """H, beam and k are actually consumed, not just stored on the dataclass."""

    # ---- H: the number of decision slots the search walks ------------------
    # ``SearchResult.slots == min(horizon / 2, frames / 2)``; a 96-frame tape is
    # long enough that only the horizon can bind, so v2a stops at 24 slots and
    # both v3 profiles walk 32.  ``beam=4`` only keeps this cheap.
    long_tape = make_tape(empty_frames(96))
    assert beam_search(long_tape, dataclasses.replace(get_config("v2a"), beam=4)).slots == 24
    assert beam_search(long_tape, dataclasses.replace(get_config("v3a"), beam=4)).slots == 32
    assert beam_search(long_tape, dataclasses.replace(get_config("v3b"), beam=4)).slots == 32

    # ---- beam: how many nodes per root survive one slot --------------------
    # The recorded expansion counts are the closed form of "every active node
    # expands into 18 candidates and only ``beam`` of them are kept", measured on
    # this exact tape (limit = 10 slots, 36 roots).  They are pinned because a
    # change here means either the beam or the horizon moved.
    tape = make_tape(_dense_frames(20, seed=21, bullets=8))
    expansions = {
        mode: beam_search(tape, get_config(mode)).detail["expansions"]
        for mode in ("v2a", "v3a", "v3b")
    }
    assert expansions == {"v2a": 281880, "v3a": 416664, "v3b": 551448}
    # The *weights* are not what moved: v2e's weights on v3a's budget expand
    # exactly the same number of nodes, so the extra work is budget, not scoring.
    assert (
        beam_search(
            tape, dataclasses.replace(get_config("v3a"), weights=get_config("v2e").weights)
        ).detail["expansions"]
        == expansions["v3a"]
    )

    # ---- k: how many plans the engine is asked to verify -------------------
    result = beam_search(tape, get_config("v3b"))
    assert len(select_plans(result, get_config("v2a").verify_k)) == 3
    assert len(select_plans(result, get_config("v3a").verify_k)) == 3
    assert len(select_plans(result, get_config("v3b").verify_k)) == 4


def test_v3_search_stays_within_the_measured_performance_ceiling():
    """The §3 gate is <= 200 ms per **decision**; this pins the search inside it.

    The engine measurement lives in EXP-P1-019 §3 (Stage5, 30 decisions, shipped
    CLI): **93.22 ms/decision** for v3a and **111.95 ms/decision** for v3b,
    against a 52.86 ms v2a baseline -- both under the 200 ms gate, so neither
    profile is rolled back.  A unit test cannot re-run the engine, so the two
    things that *are* deterministic are pinned instead:

    * the work grows by a bounded factor over v2a (a ratio, so it cannot be
      hidden by a faster machine);
    * one search finishes under a deliberately loose absolute ceiling.  The
      *relative* bound is the real assertion: an absolute one would flake when
      the box is busy running the 8-face suite.
    """

    import time

    tape = make_tape(_dense_frames(20, seed=21, bullets=8))
    timings, expansions = {}, {}
    for mode in ("v2a", "v3a", "v3b"):
        config = get_config(mode)
        started = time.perf_counter()
        result = beam_search(tape, config)
        timings[mode] = (time.perf_counter() - started) * 1000.0
        expansions[mode] = result.detail["expansions"]
    # 416,664 / 281,880 = 1.478 and 551,448 / 281,880 = 1.956: the budget grew,
    # it did not multiply.
    assert 1.4 <= expansions["v3a"] / expansions["v2a"] <= 1.6
    assert 1.8 <= expansions["v3b"] / expansions["v2a"] <= 2.1
    # The search phase was 43.5 ms (v3a) / 56.8 ms (v3b) of the 93 / 112 ms
    # decision on the engine; even a slow, loaded box must stay under 1 s here.
    assert timings["v3a"] < 1000.0, timings
    assert timings["v3b"] < 1200.0, timings
    assert timings["v3a"] < 3.0 * timings["v2a"], timings



# ---------------------------------------------------------------------------
# EXP-P1-020: v2f / v2g -- the continuous top band and the top-corner surcharge
# ---------------------------------------------------------------------------
def test_top_band_penalty_matches_the_published_formula():
    """``top_penalty * (200 - y) / 180`` above y=200, exactly the bottom shape."""

    weights = get_config("v2f").weights
    assert weights.top_penalty == pytest.approx(1.5)
    assert (weights.top_y, weights.top_span) == (200.0, 180.0)
    assert weights.top_penalty_at(210.0) == 0.0
    assert weights.top_penalty_at(200.0) == 0.0  # the edge itself is not charged
    assert weights.top_penalty_at(110.0) == pytest.approx(0.75)
    assert weights.top_penalty_at(20.0) == pytest.approx(1.5)  # the task's example
    # The playfield top is 16 px, 4 px past the span: like the bottom band, the
    # formula is not clipped there (1.5 * 184 / 180).
    assert weights.top_penalty_at(16.0) == pytest.approx(1.5 * 184.0 / 180.0)
    values = [weights.top_penalty_at(y) for y in (200.0, 180.0, 120.0, 60.0, 16.0)]
    assert values == sorted(values)  # deeper (smaller y) is never cheaper
    # v2g moves this one knob and nothing else.
    v2g = get_config("v2g").weights
    assert v2g.top_penalty == pytest.approx(3.0)
    assert v2g.top_penalty_at(20.0) == pytest.approx(3.0)
    # Every earlier profile is exactly zero: the term is off at 0.0.
    for mode in ("full", "v2a", "v2b", "v2c", "v2d", "v2e", "v3a", "v3b"):
        assert get_config(mode).weights.top_penalty_at(16.0) == 0.0, mode


def test_top_corner_surcharge_is_flat_and_only_inside_the_top_band():
    """``x < 64`` or ``x > 312`` **and** ``y < 200`` costs a flat 1.0."""

    weights = get_config("v2f").weights
    assert weights.corner_penalty == pytest.approx(1.0)
    assert weights.corner_x == (64.0, 312.0)
    # Inside a top corner: charged exactly once, whatever the depth or side.
    assert weights.corner_penalty_at(20.0, 100.0) == pytest.approx(1.0)
    assert weights.corner_penalty_at(20.0, 16.0) == pytest.approx(1.0)
    assert weights.corner_penalty_at(350.0, 190.0) == pytest.approx(1.0)
    # The threshold is strict: x == 64 and x == 312 are not corners.
    assert weights.corner_penalty_at(64.0, 100.0) == 0.0
    assert weights.corner_penalty_at(312.0, 100.0) == 0.0
    assert weights.corner_penalty_at(63.9, 100.0) == pytest.approx(1.0)
    # It is the *top* corner: outside the high band there is no surcharge.
    assert weights.corner_penalty_at(20.0, 200.0) == 0.0
    assert weights.corner_penalty_at(20.0, 250.0) == 0.0
    assert weights.corner_penalty_at(192.0, 100.0) == 0.0  # middle of the top band
    # Earlier profiles do not have the term at all.
    for mode in ("full", "v2a", "v2e"):
        assert get_config(mode).weights.corner_penalty_at(16.0, 16.0) == 0.0, mode


def test_v2f_profile_is_v2a_plus_the_top_band_and_keeps_high_penalty():
    v1, v2a = get_config("full"), get_config("v2a")
    v2f, v2g = get_config("v2f"), get_config("v2g")
    assert (v2f.mode, v2g.mode) == ("v2f", "v2g")
    assert (v2f.horizon, v2f.beam, v2f.verify_k, v2f.max_iterations) == (
        v2a.horizon, v2a.beam, v2a.verify_k, v2a.max_iterations)
    # v2a's bottom band and every shared term survive field for field.
    for name in ("bottom_penalty", "bottom_y", "bottom_span",
                 "bottom_extra_penalty", "bottom_extra_y", "bottom_extra_span",
                 "escape_weight", "align_weight", "items_weight", "items_cap",
                 "edge_penalty", "edge_band", "high_y", "death_scale",
                 "unverified_discount", "align_range", "margin_cap"):
        assert getattr(v2f.weights, name) == getattr(v2a.weights, name), name
        assert getattr(v2g.weights, name) == getattr(v2a.weights, name), name
    # §2 decision: the binary high term is *kept* on top of the continuous one.
    assert v2a.weights.high_penalty == pytest.approx(3.0)
    assert v2f.weights.high_penalty == pytest.approx(3.0)
    assert v2f.weights.top_penalty == pytest.approx(1.5)
    assert v2f.weights.corner_penalty == pytest.approx(1.0)
    assert v2g.weights.top_penalty == pytest.approx(3.0)
    assert v2g.weights.corner_penalty == pytest.approx(1.0)
    # No Boss cost (EXP-P1-017 was not the fix) and no escape term (v2b).
    assert v2f.weights.boss_penalty == 0.0 and v2g.weights.boss_penalty == 0.0
    assert v2f.weights.escape_weight == 0.0
    # v2a's non-weight knobs are untouched: 12-frame window, no keep-bomb, no
    # margin gate, no hard corridor (all three were falsified by EXP-P1-015/017).
    for profile in (v2f, v2g):
        assert profile.bomb_rule1_max_death_frame == 12
        assert profile.bomb_rule1_min_bombs_left == 0
        assert profile.bomb_rule1_max_margin is None
        assert profile.clip_y is None
        assert profile.durations == v1.durations
        assert profile.decision_frames == v1.decision_frames
    # Registered, listed, CLI-exposed; ``full`` stays the default.
    assert CONFIGS["v2f"] is v2f and CONFIGS["v2g"] is v2g
    assert {profile.mode for profile in PROFILES} >= {"v2f", "v2g"}
    assert get_config("full") is FULL and get_config("fast") is FAST
    cli = _load_cli()
    assert cli.parse_args(["--mode", "v2f"]).mode == "v2f"
    assert cli.parse_args(["--mode", "v2g"]).mode == "v2g"
    assert cli.parse_args([]).mode == "full"
    # EXP-P1-020 left the ``v2h`` name free; EXP-P1-021 is the experiment that
    # registers it, so the "not registered" guard now points one name further out.
    # EXP-P3-005 registers ``v2i`` and EXP-P3-007 registers ``v2j`` the same way,
    # so the guard moves to ``v2k``.
    assert get_config("v2h").mode == "v2h"
    assert cli.parse_args(["--mode", "v2h"]).mode == "v2h"
    assert get_config("v2i").mode == "v2i"
    assert get_config("v2j").mode == "v2j"
    with pytest.raises(ValueError):
        get_config("v2k")


def test_top_band_stacks_with_high_penalty_on_the_living_score_only():
    """Both terms fire together; the death branch carries neither."""

    v2a, v2f = get_config("v2a"), get_config("v2f")
    corner, parts = score_plan(config=v2f, margin=16.0, death_frame=None,
                               end_xy=(32.0, 20.0), target=32.0, horizon=48)
    assert parts["high"] == 1.0                 # the binary term still fires
    assert parts["top"] == pytest.approx(1.5)   # ... and the continuous one too
    assert parts["corner"] == pytest.approx(1.0)
    old, old_parts = score_plan(config=v2a, margin=16.0, death_frame=None,
                                end_xy=(32.0, 20.0), target=32.0, horizon=48)
    assert old_parts["high"] == 1.0
    assert old_parts["top"] == 0.0 and old_parts["corner"] == 0.0
    assert corner == pytest.approx(old - 1.5 - 1.0)  # exactly the two new terms
    # Above the band the ranking is v2a's: no top and no corner, bit for bit.
    top, top_parts = score_plan(config=v2f, margin=9.0, death_frame=None,
                                end_xy=(192.0, 205.0), target=192.0, horizon=48)
    plain, _ = score_plan(config=v2a, margin=9.0, death_frame=None,
                          end_xy=(192.0, 205.0), target=192.0, horizon=48)
    assert top == plain
    assert top_parts["top"] == 0.0 and top_parts["corner"] == 0.0
    # A dead plan is scored by the death branch alone (11-P1 §3).
    dead = score_plan(config=v2f, margin=0.0, death_frame=24, end_xy=(32.0, 20.0),
                      target=32.0, horizon=48)[1]
    assert dead["top"] == 0.0 and dead["corner"] == 0.0 and dead["high"] == 0.0


def test_top_band_flips_the_choice_from_the_top_corner_to_the_midfield():
    """The mechanism: 2.5 points of top cost outbid a 3.5 px margin edge."""

    ends = ((63.9, 20.0), (64.0, 210.0))  # A: top corner; B: just below the band
    margins = (16.0, 12.5)
    v2a = _finish_nodes_xy(get_config("v2a"), ends, margins, target=64.0)
    v2f = _finish_nodes_xy(get_config("v2f"), ends, margins, target=64.0)
    # v2a: the 3.5 px margin edge beats the binary high term, so A wins.
    assert v2a.end_xy[1] == pytest.approx(20.0)
    # v2f: A additionally loses 1.5 (top) + 1.0 (corner) -> B wins.
    assert v2f.end_xy == pytest.approx((64.0, 210.0))
    assert v2f.top == 0.0 and v2f.corner == 0.0
    chosen, parts = score_plan(
        config=get_config("v2f"), margin=12.5, death_frame=None,
        end_xy=(64.0, 210.0), target=64.0, horizon=get_config("v2f").horizon,
    )
    assert v2f.score == pytest.approx(chosen)
    assert parts["top"] == 0.0 and parts["corner"] == 0.0
    # ... and A's frozen score under v2f would have been lower (not a tie).
    rejected, rejected_parts = score_plan(
        config=get_config("v2f"), margin=16.0, death_frame=None,
        end_xy=(63.9, 20.0), target=64.0, horizon=get_config("v2f").horizon,
    )
    assert rejected_parts["top"] == pytest.approx(1.5)
    assert rejected_parts["corner"] == pytest.approx(1.0)
    assert rejected < chosen


def test_v2f_search_matches_the_reference_on_a_top_heavy_tape():
    """The vectorised top/corner arithmetic is the scalar formula, term by term."""

    config = dataclasses.replace(
        get_config("v2f"), mode="v2f-test", horizon=16, beam=6,
        verify_k=1, max_iterations=1,
    )
    # Start the ship in the top-left corner so both new terms are exercised on
    # almost every terminal position.
    tape = make_tape(_dense_frames(16, seed=11, bullets=6), hurtbox=(48.0, 150.0, 0.825))
    got = beam_search(tape, config, use_raster=False)
    want = beam_search_reference(tape, config)
    assert np.array_equal(got.scores, want.scores)
    assert set(got.plans) == set(want.plans)
    charged_top = charged_corner = 0
    for cand, expected in want.plans.items():
        actual = got.plans[cand]
        for name in ("score", "top", "corner", "margin", "end_xy", "death_frame",
                     "bottom", "high"):
            assert getattr(actual, name) == getattr(expected, name), (cand, name)
        if expected.death_frame is None and expected.top > 0.0:
            charged_top += 1
        if expected.death_frame is None and expected.corner > 0.0:
            charged_corner += 1
    assert charged_top > 0 and charged_corner > 0
    # The same tape under v2a scores strictly higher on every charged terminal.
    v2a = dataclasses.replace(
        get_config("v2a"), mode="v2a-test", horizon=16, beam=6,
        verify_k=1, max_iterations=1,
    )
    plain = beam_search_reference(tape, v2a)
    assert any(
        plain.plans[cand].score > want.plans[cand].score
        for cand in want.plans
        if want.plans[cand].top > 0.0 or want.plans[cand].corner > 0.0
    )


def test_v2f_at_zero_top_weights_is_bit_identical_to_v2a():
    """``top_penalty = corner_penalty = 0`` reproduces v2a exactly."""

    v2a, v2f = get_config("v2a"), get_config("v2f")
    zeroed = dataclasses.replace(v2f.weights, top_penalty=0.0, corner_penalty=0.0)
    assert zeroed == v2a.weights  # the band is the *only* difference
    tape = make_tape(_dense_frames(20, seed=3, bullets=8))
    a = beam_search(tape, v2a, use_raster=False)
    b = beam_search(tape, dataclasses.replace(v2f, weights=zeroed), use_raster=False)
    assert np.array_equal(a.scores, b.scores)  # exact equality, not approx
    for cand, plan in a.plans.items():
        other = b.plans[cand]
        for name in ("score", "end_xy", "margin", "bottom", "top", "corner",
                     "high", "align", "death_frame"):
            assert getattr(plan, name) == getattr(other, name), (cand, name)
        assert plan.top == 0.0 and plan.corner == 0.0


def test_top_terms_round_trip_through_the_plan_and_the_label():
    plan = _finish_nodes_xy(get_config("v2f"), ((32.0, 20.0),), (16.0,), target=32.0)
    assert plan.top == pytest.approx(1.5) and plan.corner == pytest.approx(1.0)
    payload = plan.as_dict()
    assert payload["top"] == pytest.approx(1.5)
    assert payload["corner"] == pytest.approx(1.0)
    label = TeacherLabel(
        scores=np.zeros(N_CAND), bomb=0, chosen_plan=plan, margin=16.0,
        verified_mask=np.zeros(N_CAND, dtype=bool),
    )
    again = TeacherLabel.from_dict(label.as_dict())
    assert again.chosen_plan is not None
    assert again.chosen_plan.top == plan.top
    assert again.chosen_plan.corner == plan.corner
    # An older JSONL (no ``top``/``corner`` keys) still deserialises.
    legacy = dict(payload)
    del legacy["top"], legacy["corner"]
    restored = TeacherLabel.from_dict(
        {**label.as_dict(), "chosen_plan": legacy}
    ).chosen_plan
    assert restored is not None and restored.top == 0.0 and restored.corner == 0.0


# ---------------------------------------------------------------------------
# EXP-P1-021: the laser judgement window (startTime/.../damageable) and v2h
# ---------------------------------------------------------------------------
def laser_row(
    state=1,
    timer=10,
    *,
    origin=(0.0, 384.0),
    angle=0.0,
    start=0.0,
    end=400.0,
    width=8.0,
    start_time=60,
    hitbox_start_time=45,
    duration=120,
    despawn_duration=30,
    hitbox_end_delay=12,
    damageable=None,
    slot=0,
    in_use=1,
):
    """One ``LaserRow`` shaped like the engine's, with an explicit window.

    ``damageable=None`` builds the **pre-EXP-P1-021** row (no window fields at
    all), which is the backward-compatibility case the tests pin down.
    """

    return LaserRow(
        slot=slot,
        state=state,
        position=(origin[0], origin[1], 0.0),
        angle=angle,
        start_offset=start,
        end_offset=end,
        current_width=width,
        width=width,
        timer=timer,
        in_use=in_use,
        start_time=start_time,
        hitbox_start_time=hitbox_start_time,
        duration=duration,
        despawn_duration=despawn_duration,
        hitbox_end_delay=hitbox_end_delay,
        damageable=damageable,
    )


def test_laser_row_parses_the_appended_window_and_keeps_old_prefixes_readable():
    """EXP-P1-021 row: 18 fields decode; a 12-field G0/P1 row still parses."""

    row = "3:1:00000000:00000000:00000000:00000000:00000000:00000000:00000000:00000000:7:1:60:45:120:30:12:1"
    parsed = parse_laser_row(row)
    assert parsed.slot == 3 and parsed.state == 1 and parsed.timer == 7
    assert (parsed.start_time, parsed.hitbox_start_time, parsed.duration) == (60, 45, 120)
    assert (parsed.despawn_duration, parsed.hitbox_end_delay) == (30, 12)
    assert parsed.damageable == 1 and parsed.windows_available is True
    assert parsed.carries_hitbox is True
    # 16 fields (inUse + the five numbers, no flag) is a truncated-but-tolerated row.
    truncated = parse_laser_row(row.rsplit(":", 1)[0])
    assert truncated.damageable is None and truncated.duration == 120
    # The G0 prefix alone: 11 fields *plus* the P1 inUse, exactly as before.
    legacy = parse_laser_row(":".join(row.split(":")[:12]))
    assert legacy.damageable is None and legacy.windows_available is False
    assert legacy.in_use == 1 and legacy.state == 1
    assert legacy.carries_hitbox is True  # ACTIVE fallback


def test_laser_hits_uses_the_engine_damageable_flag_not_the_state():
    """The flag decides, not ``state`` — that is the whole point of P1's export.

    A ``LASER_STATE_STARTING`` beam whose ``timer`` already passed
    ``hitboxStartTime`` is lethal even though the old state-only rule skipped it;
    an ``ACTIVE`` beam the engine marked ``damageable = 0`` is not.
    """

    hurt_min = (192.0 - 0.825, 384.0 - 0.825)
    hurt_max = (192.0 + 0.825, 384.0 + 0.825)

    starting_live = laser_row(state=0, timer=46, damageable=1)
    verdict = geometry.laser_hits([starting_live], hurt_min, hurt_max)
    assert verdict.considered == 1 and verdict.hit is True

    starting_old = laser_row(state=0, timer=46, damageable=None)
    legacy = geometry.laser_hits([starting_old], hurt_min, hurt_max)
    assert legacy.considered == 0 and legacy.hit is False
    assert legacy.skipped_inactive == 1

    active_flagged_off = laser_row(state=1, timer=10, damageable=0)
    off = geometry.laser_hits([active_flagged_off], hurt_min, hurt_max)
    assert off.considered == 0 and off.hit is False and off.skipped_inactive == 1

    active_flagged_on = laser_row(state=1, timer=10, damageable=1)
    on = geometry.laser_hits([active_flagged_on], hurt_min, hurt_max)
    assert on.hit is True and on.nearest is not None and on.nearest[2] == 0.0


def test_laser_gap_rows_and_laser_hits_agree_on_the_damageable_gate():
    """The search's gap helper and the frame verdict share one gate."""

    beam = laser_row(state=0, timer=46, damageable=1)
    half = (0.825, 0.825)
    inside = np.array([[192.0, 384.0], [100.0, 384.0], [192.0, 300.0]])
    gaps = geometry.laser_gap_rows([beam], inside, half)
    assert gaps[0] == 0.0 and gaps[1] == 0.0
    assert gaps[2] == pytest.approx(384.0 - 300.0 - 0.825 - 2.0, abs=1e-6)
    for index in range(inside.shape[0]):
        hit = geometry.laser_hits(
            [beam],
            (inside[index, 0] - half[0], inside[index, 1] - half[1]),
            (inside[index, 0] + half[0], inside[index, 1] + half[1]),
        ).hit
        assert hit == (gaps[index] == 0.0), index
    # A fading beam is not a hazard for either function.
    fading = laser_row(state=0, timer=46, damageable=0)
    assert np.all(np.isinf(geometry.laser_gap_rows([fading], inside, half)))
    assert geometry.laser_hits([fading], (-1.0, -1.0), (1.0, 1.0)).hit is False


def test_v2h_flips_the_choice_out_of_a_damageable_laser_and_not_out_of_a_fading_one():
    """v2h = v2a plus one terminal-frame laser cost, gated by ``damageable``."""

    from touhou_rl.v400.teacher.config import V2H

    ends = ((192.0, 300.0), (192.0, 384.0))  # A: above the beam, B: inside it
    margins = (14.0, 15.0)
    live = [laser_row(state=1, timer=10, damageable=1)]
    v2a = _finish_nodes_xy(get_config("v2a"), ends, margins, laser_rows=live)
    v2h = _finish_nodes_xy(V2H, ends, margins, laser_rows=live)
    assert v2a.end_xy[1] == pytest.approx(384.0)  # v2a: B wins by 1.94 points
    assert v2a.laser == 0.0                       # v2a never charges a laser
    assert v2h.end_xy[1] == pytest.approx(300.0)  # v2h: B's 12.0 point beam cost flips it
    assert v2h.laser == 0.0                       # the chosen plan is in clear air

    expected, parts = score_plan(
        config=V2H, margin=15.0, death_frame=None, end_xy=(192.0, 384.0),
        target=192.0, horizon=V2H.horizon, laser_gap=0.0,
    )
    assert parts["laser"] == pytest.approx(12.0)

    # Same geometry, but the engine says the beam has no hitbox: no charge, no flip.
    fading = [laser_row(state=0, timer=10, damageable=0)]
    neutral = _finish_nodes_xy(V2H, ends, margins, laser_rows=fading)
    assert neutral.end_xy[1] == pytest.approx(384.0)
    assert neutral.laser == 0.0


def test_v2h_with_a_zero_laser_penalty_is_bit_identical_to_v2a():
    """The laser term is the *only* difference between v2h and v2a."""

    from touhou_rl.v400.teacher.config import V2H

    zeroed = dataclasses.replace(V2H.weights, laser_penalty=0.0)
    assert zeroed == get_config("v2a").weights
    assert CONFIGS["v2h"] is V2H and V2H in PROFILES
    assert V2H.weights.laser_penalty == pytest.approx(12.0)
    frame_bullets = _dense_frames(20, seed=3, bullets=8)
    tape = make_tape(frame_bullets)
    # A wide beam covering the playfield band the dense tape's terminals live in,
    # so "does the shipped weight charge anything at all" is not a coin flip.
    wide = laser_row(
        state=1, timer=10, damageable=1, slot=0,
        origin=(0.0, 300.0), end=800.0, width=400.0,
    )
    beams = tuple((wide,) for _ in range(len(tape)))
    tape = dataclasses.replace(tape, lasers=beams)
    a = beam_search(tape, get_config("v2a"), use_raster=False)
    b = beam_search(tape, dataclasses.replace(V2H, weights=zeroed), use_raster=False)
    assert np.array_equal(a.scores, b.scores)
    for cand, plan in a.plans.items():
        other = b.plans[cand]
        for name in ("score", "end_xy", "margin", "top", "corner", "laser", "death_frame"):
            assert getattr(plan, name) == getattr(other, name), (cand, name)
        assert plan.laser == 0.0
    # ... and with the shipped weight the term is real on the same tape.
    c = beam_search(tape, V2H, use_raster=False)
    assert any(plan.laser > 0.0 for plan in c.plans.values())


def test_laser_term_round_trips_through_the_plan_and_the_label():
    from touhou_rl.v400.teacher.config import V2H

    beam = [laser_row(state=1, timer=10, damageable=1)]
    plan = _finish_nodes_xy(V2H, ((192.0, 384.0),), (15.0,), laser_rows=beam)
    assert plan.laser == pytest.approx(12.0)
    payload = plan.as_dict()
    assert payload["laser"] == pytest.approx(12.0)
    label = TeacherLabel(
        scores=np.zeros(N_CAND), bomb=0, chosen_plan=plan, margin=16.0,
        verified_mask=np.zeros(N_CAND, dtype=bool),
    )
    again = TeacherLabel.from_dict(label.as_dict())
    assert again.chosen_plan is not None and again.chosen_plan.laser == plan.laser
    legacy = dict(payload)
    del legacy["laser"]
    restored = TeacherLabel.from_dict({**label.as_dict(), "chosen_plan": legacy}).chosen_plan
    assert restored is not None and restored.laser == 0.0


# ---------------------------------------------------------------------------
# EXP-P3-007: v2j -- the first-action duration set {1,2} -> {1,2,3,4}
# ---------------------------------------------------------------------------
def test_v2j_is_v2h_with_the_duration_set_widened_and_nothing_else():
    """EXP-P3-007 §1: one field moves (``durations``); everything else is v2h.

    The profile exists to test EXP-P3-005 §8's P0 hypothesis ("the 18 candidates
    are almost the same action"), so the experiment is only interpretable if the
    *only* difference is the root grid.  That is pinned two ways: the weights are
    the **same instance** (an identity, not a re-typed copy), and folding the one
    moved field back reproduces ``v2h`` field for field.
    """

    v2h, v2j = get_config("v2h"), get_config("v2j")
    assert v2j.mode == "v2j"
    assert v2j.durations == (1, 2, 3, 4)
    assert v2h.durations == (1, 2)  # v2h itself is untouched
    assert v2j.weights is v2h.weights
    # ... and folding the one moved field back is exactly v2h.
    assert dataclasses.replace(v2j, mode="v2h", durations=(1, 2)) == v2h
    for name in (
        "horizon", "beam", "verify_k", "max_iterations", "replan_decisions",
        "decision_frames", "border_x", "border_y", "record_items",
        "shoot_always", "shoot_pulse_period", "bomb_active", "bomb_death",
        "bomb_rule1_max_death_frame", "bomb_rule1_min_bombs_left",
        "bomb_rule1_max_margin", "clip_y",
    ):
        assert getattr(v2j, name) == getattr(v2h, name), name
    assert (v2j.horizon, v2j.beam, v2j.verify_k, v2j.max_iterations) == (48, 64, 3, 3)
    assert v2j.weights.laser_penalty == pytest.approx(12.0)
    assert v2j.weights.laser_band == pytest.approx(8.0)
    # Registered, listed, CLI-reachable; ``full`` is still the default.
    assert CONFIGS["v2j"] is v2j
    assert {profile.mode for profile in PROFILES} >= {"v2h", "v2i", "v2j"}
    cli = _load_cli()
    assert cli.parse_args(["--mode", "v2j"]).mode == "v2j"
    assert cli.parse_args([]).mode == "full"
    with pytest.raises(ValueError):
        get_config("v2k")


def test_v2h_v2i_and_the_earlier_profiles_keep_their_duration_set():
    """``v2j`` must not leak the wider set into any already-published profile.

    EXP-P1-015/017/020/021/EXP-P3-005 numbers were all produced with
    ``durations = (1, 2)``; a profile that silently inherits ``(1, 2, 3, 4)``
    would invalidate every one of them.  Re-declaring ``v2h`` field for field
    also pins that *nothing* about it moved.
    """

    from touhou_rl.v400.teacher.config import V2A, V2H, V2I, Weights

    assert V2H == type(V2H)(
        mode="v2h", horizon=48, beam=64, verify_k=3, max_iterations=3,
        weights=Weights(bottom_penalty=1.5, bottom_extra_penalty=2.0,
                        laser_penalty=12.0, laser_band=8.0),
        bomb_rule1_max_death_frame=12,
    )
    assert V2H.durations == (1, 2)
    assert V2I.durations == (1, 2)
    assert V2A.durations == (1, 2)
    assert get_config("full").durations == (1, 2)
    assert get_config("fast").durations == (1, 2)
    assert get_config("fast3").durations == (1, 2)
    for mode in ("v2a", "v2b", "v2c", "v2d", "v2e", "v2f", "v2g", "v2h", "v2i",
                 "v3a", "v3b"):
        assert get_config(mode).durations == (1, 2), mode


def test_v2j_durations_reach_the_search_roots_and_the_executed_plan():
    """The four durations are actually searched, and the hold length moves pixels.

    Two independent facts: (a) the root grid is ``18 x len(durations)`` and the
    plans carry the duration they were searched with; (b) a longer hold really
    displaces the ship further -- ``simulate_plan`` walks ``2 * duration`` frames
    of the same per-frame delta, so 4 slots travel exactly twice as far as 2.
    """

    tape = make_tape(_dense_frames(20, seed=21, bullets=8))
    short = dataclasses.replace(get_config("v2h"), beam=6)
    wide = dataclasses.replace(get_config("v2j"), beam=6)
    a, b = beam_search(tape, short), beam_search(tape, wide)
    assert short.durations == (1, 2) and wide.durations == (1, 2, 3, 4)
    assert a.detail["roots"] == N_CAND * 2  # 36
    assert b.detail["roots"] == N_CAND * 4  # 72
    assert b.durations == (1, 2, 3, 4)
    assert len(a.plans) == N_CAND and len(b.plans) == N_CAND
    assert all(plan.duration in (1, 2) for plan in a.plans.values())
    assert all(plan.duration in (1, 2, 3, 4) for plan in b.plans.values())
    # The wider set is *used*, not merely allowed: at least one candidate's best
    # plan needs a hold longer than 2 slots, otherwise the new roots never win.
    assert any(plan.duration > 2 for plan in b.plans.values())

    # ---- the hold length is a per-frame walk, so the endpoint scales with it ---
    # Candidate 1 = "N" (up), unfocused: 4.0 px per logic frame, 2 frames per slot.
    # An empty tape keeps the travel exact (a bullet would end the walk early).
    clean = make_tape(empty_frames(20))
    start = np.asarray(clean.start_position, dtype=np.float64)
    for duration in (1, 2, 3, 4):
        margin, end, death, _ = simulate_plan(
            clean, [1] * duration, config=get_config("v2j"))
        assert death is None
        travel = float(start[1] - np.asarray(end, dtype=np.float64)[1])
        assert travel == pytest.approx(4.0 * 2 * duration), (duration, travel)
    # ... and the idle candidate moves nowhere, so the set really is only about
    # *how long* an action is held, not about a new action space.
    _, idle_end, _, _ = simulate_plan(clean, [9] * 4, config=get_config("v2j"))
    assert np.allclose(np.asarray(idle_end, dtype=np.float64), start)


def test_v2j_price_is_bounded_by_the_root_grid():
    """EXP-P3-005 §8 budgeted "search ~2x"; this pins that it is a ratio, not a blow-up.

    The extra work is exactly the extra roots: each duration ``d`` skips the
    first ``d`` expansion rounds of its own root, so the expansion count is
    ``18 * sum(limit - d)`` over ``d in durations`` against ``18 * sum(limit - d)``
    over ``(1, 2)``.  With ``limit`` = 10 slots that is 30/17 = 1.76; the test
    bounds the measured ratio to the bracket the card reports.
    """

    import time

    tape = make_tape(_dense_frames(20, seed=21, bullets=8))
    results, timings = {}, {}
    for mode in ("v2h", "v2j"):
        config = get_config(mode)
        started = time.perf_counter()
        results[mode] = beam_search(tape, config)
        timings[mode] = (time.perf_counter() - started) * 1000.0
    assert results["v2h"].slots == results["v2j"].slots == 10
    roots = results["v2j"].detail["roots"] / results["v2h"].detail["roots"]
    expansions = (results["v2j"].detail["expansions"]
                  / results["v2h"].detail["expansions"])
    assert roots == 2.0
    assert 1.6 <= expansions <= 2.0, expansions
    # 18 * (9 + 8 + 7 + 6) / (18 * (9 + 8)) = 30 / 17 on a full beam.
    assert expansions == pytest.approx(30 / 17, rel=0.05)
    # Relative bound is the real assertion (an absolute one flakes on a busy box).
    assert timings["v2j"] < 4.0 * timings["v2h"], timings
    assert timings["v2j"] < 2000.0, timings


def test_v2j_folded_back_to_two_durations_is_bit_identical_to_v2h():
    """With the duration set folded back, the search cannot tell the two apart.

    This is the "v2h is unchanged" half of EXP-P3-007 §1 at the *output* level:
    the same tape, the same weights instance, the same budget, and the same
    ``durations`` must reproduce every score and every endpoint bit for bit.
    """

    tape = make_tape(_dense_frames(20, seed=21, bullets=8))
    folded = dataclasses.replace(get_config("v2j"), durations=(1, 2))
    # ``use_raster=False`` so the exact closed-form path is compared with the
    # reference implementation (the raster table trades margin resolution for speed).
    a = beam_search(tape, get_config("v2h"), use_raster=False)
    b = beam_search(tape, folded, use_raster=False)
    assert np.array_equal(a.scores, b.scores)
    assert np.array_equal(a.scores, beam_search_reference(tape, folded).scores)
    for cand, plan in a.plans.items():
        other = b.plans[cand]
        assert plan.duration == other.duration
        assert plan.score == other.score
        assert plan.end_xy == other.end_xy
        assert plan.margin == other.margin
    # ... and the wider set *can* move a label: with an align target 88 px to the
    # right, holding "east" (cand 3, +4 px/frame unfocused) for 4 slots ends much
    # closer to it than 2 slots can, so the extra roots win on align alone.  This
    # is the mechanism EXP-P3-007 hypothesises; a wide set that could never change
    # a score would make the whole profile inert.
    goal_frames = [((), (enemy(280.0, 300.0, hitbox=60.0),)) for _ in range(20)]
    goal = make_tape(goal_frames)
    near = beam_search(goal, get_config("v2h"), use_raster=False)
    far = beam_search(goal, get_config("v2j"), use_raster=False)
    east = 3
    assert far.plans[east].duration > near.plans[east].duration
    assert far.plans[east].score > near.plans[east].score
    assert far.plans[east].end_xy[0] > near.plans[east].end_xy[0]
