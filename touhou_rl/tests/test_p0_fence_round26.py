"""Unit tests for the P0 re-run tool (round 26) — no game, no fence section.

The live part of ``tools/rl/p0_fence_round26.py`` needs the running game; what is
tested here is everything that turns recorded frames into a verdict:

* the input-mask constants must match ``spec.py`` (a silent mismatch would make
  every "input was consumed" claim meaningless),
* lag / displacement extraction from a frame sequence,
* boundary saturation and reversal-carry extraction,
* the pre-registered gate aggregation (G1..G11) including its failure paths.
"""

from __future__ import annotations

import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools" / "rl") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "rl"))

import p0_fence_round26 as probe  # noqa: E402

from touhou_rl import framefence, spec  # noqa: E402


# ---------------------------------------------------------------------------
# Constants and masks
# ---------------------------------------------------------------------------
def test_mask_constants_match_spec():
    assert probe.BTN_SHOOT == spec.BTN_SHOOT
    assert probe.BTN_FOCUS == spec.BTN_FOCUS
    assert probe.BTN_UP == spec.BTN_UP
    assert probe.BTN_DOWN == spec.BTN_DOWN
    assert probe.BTN_LEFT == spec.BTN_LEFT
    assert probe.BTN_RIGHT == spec.BTN_RIGHT
    assert probe.DIRECTION_BITS == (
        spec.BTN_UP | spec.BTN_DOWN | spec.BTN_LEFT | spec.BTN_RIGHT
    )


def test_expected_mask_ignores_shoot_when_comparing():
    assert probe.expected_mask({"z"}) == spec.BTN_SHOOT
    assert probe.expected_mask({"z", "right"}) == spec.BTN_SHOOT | spec.BTN_RIGHT
    assert probe.compared_mask(probe.expected_mask({"z", "right"})) == spec.BTN_RIGHT
    focus = probe.expected_mask({"z", "shift", "left"})
    assert probe.compared_mask(focus) == spec.BTN_FOCUS | spec.BTN_LEFT
    diagonal = probe.expected_mask({"z", "right", "up"})
    assert probe.compared_mask(diagonal) == spec.BTN_RIGHT | spec.BTN_UP


# ---------------------------------------------------------------------------
# Frame-sequence extraction
# ---------------------------------------------------------------------------
def _row(index, *, x=192.0, y=384.0, vx=0.0, vy=0.0, mask=spec.BTN_SHOOT, frame=100, **kw):
    row = {
        "ticket": index,
        "logic_frame": frame + index,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "input_mask": mask,
        "focus": kw.get("focus", 0),
        "player_state": kw.get("player_state", 0),
    }
    return row


def test_consumption_and_motion_indices():
    mask = probe.expected_mask({"z", "right"})
    seq = [
        _row(0),
        _row(1),
        _row(2, x=196.0, vx=4.0, mask=mask),
        _row(3, x=200.0, vx=4.0, mask=mask),
    ]
    assert probe.consumption_index(seq, mask) == 2
    assert probe.motion_index(seq, "x", 192.0) == 2
    # The anchor frame itself must never be counted as consumption/motion.
    assert probe.consumption_index(seq, mask, start=3) == 3
    assert probe.motion_index(seq, "x", 200.0, start=3) is None


def test_is_static_and_state_helpers():
    a = _row(0)
    b = _row(1)
    moving = _row(2, x=196.0, vx=4.0)
    assert probe.is_static(a, b)
    assert not probe.is_static(b, moving)


def test_trial_metrics_normal_and_diagonal():
    keys = frozenset({"z", "right"})
    seq = [
        _row(0),
        _row(1, x=196.0, vx=4.0, mask=probe.expected_mask(keys)),
    ]
    metrics = probe.trial_metrics(seq, keys, "x")
    assert metrics["lag_in"] == 1
    assert metrics["lag_mv"] == 1
    assert metrics["delta"] == 4.0
    assert metrics["delta_equals_velocity"] is True

    diagonal = frozenset({"z", "right", "up"})
    step = 4.0 / (2.0**0.5)
    seq = [
        _row(0, y=384.0),
        _row(1, x=192.0 + step, y=384.0 - step, vx=step, vy=-step, mask=probe.expected_mask(diagonal)),
    ]
    metrics = probe.trial_metrics(seq, diagonal, "x")
    assert abs(abs(metrics["delta"]) - step) <= 1e-4 * step
    assert metrics["lag_in"] == 1


def test_summarize_class_pass_and_failure_paths():
    keys = frozenset({"z", "right"})
    good = []
    for index in range(20):
        seq = [_row(0), _row(1, x=196.0, vx=4.0, mask=probe.expected_mask(keys))]
        good.append(
            {
                "class": "C1_lateral_normal",
                "metrics": probe.trial_metrics(seq, keys, "x"),
            }
        )
    summary = probe.summarize_class(good, probe.TRIAL_CLASSES[0])
    assert summary["pass_lag_in"] and summary["pass_lag_mv"] and summary["pass_delta"]
    assert summary["lagInMatch"] == 20

    # One trial with a two-frame input lag must fail the pre-registered gate.
    bad = list(good)
    late_seq = [
        _row(0),
        _row(1),
        _row(2, x=196.0, vx=4.0, mask=probe.expected_mask(keys)),
    ]
    bad[0] = {
        "class": "C1_lateral_normal",
        "metrics": probe.trial_metrics(late_seq, keys, "x"),
    }
    summary = probe.summarize_class(bad, probe.TRIAL_CLASSES[0])
    assert summary["pass_lag_in"] is False
    assert summary["lagInMatch"] == 19

    # ... and so must a displacement that is not exactly one step.
    off = list(good)
    off[1] = {
        "class": "C1_lateral_normal",
        "metrics": probe.trial_metrics(
            [_row(0), _row(1, x=193.0, vx=4.0, mask=probe.expected_mask(keys))],
            keys,
            "x",
        ),
    }
    assert probe.summarize_class(off, probe.TRIAL_CLASSES[0])["pass_delta"] is False


def test_gap_summary_detects_skipped_tickets():
    rows = [_row(i, frame=100) for i in range(5)]
    assert probe.gap_summary(probe.ticket_gaps(rows))["allOne"]
    rows[3]["ticket"] = rows[2]["ticket"] + 2
    rows[4]["ticket"] = rows[3]["ticket"] + 1
    summary = probe.gap_summary(probe.ticket_gaps(rows))
    assert summary["allOne"] is False
    assert summary["counts"] == {"1": 3, "2": 1}
    assert summary["max"] == 2


def test_qpc_rate_uses_game_side_clock():
    assert probe.qpc_rate([], 10) is None
    rows = [{"qpc": 0}, {"qpc": 10_000_000}, {"qpc": 20_000_000}]
    assert probe.qpc_rate(rows, 10_000_000) == pytest.approx(1.0)
    # A backwards clock must not be silently turned into a rate.
    assert probe.qpc_rate([{"qpc": 5}, {"qpc": 4}, {"qpc": 3}], 10) is None


# ---------------------------------------------------------------------------
# Boundaries and carry
# ---------------------------------------------------------------------------
def test_saturation_from_rows():
    wall = [_row(i, x=min(8.0 + 4.0 * i, 376.0)) for i in range(96)]
    result = probe.saturation_from_rows(wall, "x", 1)
    assert result["saturated"] is True
    assert result["boundary"] == 376.0
    assert result["staticTail"] >= 2
    assert result["observedMax"] == 376.0

    still_moving = [_row(i, x=8.0 + 4.0 * i) for i in range(5)]
    assert probe.saturation_from_rows(still_moving, "x", 1)["saturated"] is False

    never_moved = [_row(i, x=200.0) for i in range(5)]
    assert probe.saturation_from_rows(never_moved, "x", 1)["saturated"] is False


def test_carry_metrics_one_frame_of_old_direction():
    keys_ok = [
        _row(0, x=200.0, vx=4.0),
        _row(1, x=204.0, vx=4.0),   # first frame after the reversal still goes right
        _row(2, x=200.0, vx=-4.0),  # then it turns
        _row(3, x=196.0, vx=-4.0),
    ]
    carry = probe.carry_metrics(keys_ok, "x", 1)
    assert carry["oldCarryFrames"] == 1
    assert carry["oldCarryPx"] == 4.0
    assert carry["reversedOnNextFrame"] is True

    immediate = [_row(0, x=200.0), _row(1, x=196.0), _row(2, x=192.0)]
    carry = probe.carry_metrics(immediate, "x", 1)
    assert carry["oldCarryFrames"] == 0
    assert carry["reversedOnNextFrame"] is True


# ---------------------------------------------------------------------------
# Sample checks
# ---------------------------------------------------------------------------
def test_check_summary_and_all_ok():
    samples = [
        {"checks": {"hold_stable": True, "pixels_match_backbuffer": True}},
        {"checks": {"hold_stable": True, "pixels_match_backbuffer": False}},
    ]
    summary = probe.check_summary(samples)
    assert summary["hold_stable"] == {"passed": 2, "total": 2}
    assert summary["pixels_match_backbuffer"] == {"passed": 1, "total": 2}
    assert probe.sample_all_ok(samples[0]) is True
    assert probe.sample_all_ok(samples[1]) is False
    assert probe.sample_all_ok({"checks": {}}) is False


# ---------------------------------------------------------------------------
# Verdict aggregation (G1..G11)
# ---------------------------------------------------------------------------
def _trial(class_name, lag_in=1, lag_mv=1, delta=4.0):
    return {
        "class": class_name,
        "metrics": {
            "lag_in": lag_in,
            "lag_mv": lag_mv,
            "delta": delta,
            "velocity_at_motion": delta,
            "delta_equals_velocity": True,
        },
    }


def _passing_evidence():
    classes = []
    for trial_spec in probe.TRIAL_CLASSES:
        trials = [
            _trial(trial_spec["name"], delta=trial_spec["expected_axis_delta"])
            for _ in range(20)
        ]
        classes.append(probe.summarize_class(trials, trial_spec))
    boundaries = [
        {"name": name, "boundary": expected, "expected": expected, "saturated": True,
         "match": True}
        for name, _axis, _keys, expected in probe.BOUNDARY_PUSHES
    ]
    carry_trials = [
        {"carry": {"oldCarryFrames": 1, "oldCarryPx": 4.0}} for _ in range(20)
    ]
    checks = {
        name: {"passed": 60, "total": 60}
        for name in (
            "hold_stable",
            "player_position",
            "player_velocity",
            "player_state",
            "focus_mode",
            "gameplay_frame",
            "stage_frames",
            "input_mask",
            "active_bullets",
            "bullet_digest",
            "enemy_digest",
            "pixels_frozen",
            "pixels_match_backbuffer",
        )
    }
    return {
        "sessions": {
            "demo": {
                "phases": {
                    "M1_clock": {
                        "ticketGaps": {"allOne": True},
                        "logicGaps": {"allOne": True},
                        "presentHz": 59.9,
                    },
                    "M5_samples": {
                        "accepted": 60,
                        "discarded": 0,
                        "allChecksPassed": 60,
                        "checkCounts": checks,
                    },
                    "M6_legacy": {
                        "reads": 60,
                        "ticketChanged": 12,
                        "bulletDigestMismatch": 3,
                        "enemyDigestMismatch": 9,
                    },
                }
            },
            "practice": {
                "phases": {
                    "M2_M3_trials": {"classes": classes},
                    "M4_boundaries": boundaries,
                    "M4_carry": {
                        "reversals": [{"name": "x_right_to_left", "trials": carry_trials}],
                        "axisSwitches": [],
                    },
                },
                "clockAfterTrials": {
                    "ticketGaps": {"allOne": True},
                    "logicGaps": {"allOne": True},
                },
            },
        }
    }


def test_verdict_passes_when_every_gate_passes():
    verdict = probe.build_verdict(_passing_evidence())
    assert verdict["p0UnfreezePassed"] is True
    assert verdict["decisions"]["exactLabelsAllowed"] is True
    assert all(
        verdict["gates"][name]["passed"] is True
        for name in verdict["gates"]
        if not name.startswith("G11")
    )
    # G11 is a measurement, never a pass/fail gate.
    assert verdict["gates"]["G11_legacy_still_unknown"]["passed"] is None


@pytest.mark.parametrize(
    "mutate, failing_gate",
    [
        (lambda data: data["sessions"]["demo"]["phases"]["M1_clock"].update(
            ticketGaps={"allOne": False}), "G1_ticket_step"),
        (lambda data: data["sessions"]["demo"]["phases"]["M1_clock"].update(
            presentHz=41.0), "G3_present_rate"),
        (lambda data: data["sessions"]["practice"]["phases"]["M2_M3_trials"]["classes"].pop(0),
         "G4_lag_in_strict"),
        (lambda data: data["sessions"]["practice"]["phases"]["M4_boundaries"].pop(),
         "G8_boundaries"),
        (lambda data: data["sessions"]["practice"]["phases"]["M4_carry"]["reversals"][0].update(
            trials=[{"carry": {"oldCarryFrames": 0, "oldCarryPx": 0.0}}]), "G9_reversal_carry"),
        (lambda data: data["sessions"]["demo"]["phases"]["M5_samples"].update(
            allChecksPassed=59), "G10_pixel_state"),
        (lambda data: data["sessions"]["demo"]["phases"]["M5_samples"]["checkCounts"].update(
            pixels_match_backbuffer={"passed": 59, "total": 60}), "G10_pixel_state"),
    ],
)
def test_verdict_refuses_to_unfreeze_when_a_gate_fails(mutate, failing_gate):
    data = _passing_evidence()
    mutate(data)
    verdict = probe.build_verdict(data)
    assert verdict["p0UnfreezePassed"] is False
    assert verdict["gates"][failing_gate]["passed"] is False
    assert verdict["decisions"]["exactLabelsAllowed"] is False


def test_verdict_treats_missing_session_as_failure():
    verdict = probe.build_verdict({"sessions": {}})
    assert verdict["p0UnfreezePassed"] is False


# ---------------------------------------------------------------------------
# Task queue / aggregation (the resumable practice protocol)
# ---------------------------------------------------------------------------
def test_task_queue_covers_every_pre_registered_measurement():
    from collections import Counter

    args = SimpleNamespace(per_class=20, carry_repeats=5)
    tasks = probe._task_queue(args)
    kinds = Counter(task["kind"] for task in tasks)
    assert kinds == {"trial": 80, "boundary": 4, "reversal": 20, "switch": 10}
    assert len(tasks) == 80 + 4 + 20 + 10

    trials = [task for task in tasks if task["kind"] == "trial"]
    per_class = Counter(task["spec"]["name"] for task in trials)
    assert per_class == {spec["name"]: 20 for spec in probe.TRIAL_CLASSES}
    # Directions alternate inside each class so the player does not drift.
    first_class = [task["keys"] for task in trials if task["spec"]["name"] == "C1_lateral_normal"]
    assert first_class[0] != first_class[1]
    assert first_class[0] == first_class[2]


def test_merge_gap_summaries():
    first = {"n": 2, "allOne": True, "counts": {"1": 2}, "max": 1}
    second = {"n": 1, "allOne": False, "counts": {"2": 1}, "max": 2}
    merged = probe._merge_gap_summaries([first, second])
    assert merged["n"] == 3
    assert merged["allOne"] is False
    assert merged["counts"] == {"1": 2, "2": 1}
    assert merged["max"] == 2
    # No sessions at all is not "all one" (absence must not read as success).
    assert probe._merge_gap_summaries([])["allOne"] is False


# ---------------------------------------------------------------------------
# FenceFrameSampler.on_hold: the hook that types keys inside the hold
# ---------------------------------------------------------------------------
class _AdvancingFence:
    """Fake section: reports one stable ticket, then always the next one."""

    def __init__(self, ticket=10):
        self._ticket = ticket
        self.hold_phase = 1
        self.reads = 0
        self.acks: list[int] = []

    @property
    def ticket(self):
        self.reads += 1
        if self.reads > 2:
            self._ticket = 11
        return self._ticket

    def telemetry(self):
        return self._telemetry

    def ack(self, ticket):
        self.acks.append(ticket)


def _fake_sampler_setup():
    bits = tuple(
        struct.unpack("<I", struct.pack("<f", value))[0] for value in (192.5, 384.25, 0.0, 0.0)
    )
    telemetry = framefence.FenceTelemetry(
        ticket=11,
        logic_frame=5000,
        stage_frames=4000,
        run_frames=4000,
        player_state=0,
        focus_mode=0,
        input_mask=spec.BTN_SHOOT,
        gui_input_current=spec.BTN_SHOOT,
        active_bullets=7,
        bullet_digest=0x1234ABCD,
        enemy_digest=0x5678EF01,
        digest_ticket=11,
        hold_phase=1,
        _bits=bits,
    )
    state = SimpleNamespace(
        player_x=192.5,
        player_y=384.25,
        player_vx=0.0,
        player_vy=0.0,
        player_state=0,
        focus_mode=0,
        gameplay_frame=5000,
        stage_frames=4000,
        input_mask=spec.BTN_SHOOT,
        active_bullets=7,
    )
    raw = np.zeros((spec.BULLET_COUNT, spec.BULLET_SIZE), dtype=np.uint8)
    bullets = SimpleNamespace(raw=raw)
    reader = SimpleNamespace(read_state=lambda with_bullets=True: state,
                             read_bullets=lambda: bullets)
    session = SimpleNamespace(reader=reader, grab=lambda: np.zeros((480, 640, 3), np.uint8))
    fence = _AdvancingFence()
    fence._telemetry = telemetry
    digest = framefence.FenceStateDigest(reader=None).bullet_digest(bullets)
    telemetry.bullet_digest = digest
    sampler = framefence.FenceFrameSampler(
        session, fence, with_pixels=False, with_enemies=False, verify_pixels=False
    )
    return sampler, fence, digest


def test_on_hold_is_called_inside_the_hold():
    sampler, fence, _digest = _fake_sampler_setup()
    seen = []

    sampler.on_hold = lambda telemetry: seen.append(telemetry.ticket)
    sample = sampler.next(timeout=0.5, attempts=1)

    assert sample is not None and sample.ok
    assert seen == [11]
    assert sampler.on_hold_calls == 1
    # The ack must come after the hook ran, exactly once for the accepted ticket.
    assert fence.acks == [11]


def test_on_hold_errors_are_recorded_and_do_not_break_the_sample():
    sampler, _fence, _digest = _fake_sampler_setup()

    def boom(_telemetry):
        raise RuntimeError("hook exploded")

    sampler.on_hold = boom
    sample = sampler.next(timeout=0.5, attempts=1)
    assert sample is not None and sample.ok
    assert sampler.on_hold_calls == 0
    assert sampler.on_hold_errors and "hook exploded" in sampler.on_hold_errors[0]


# ---------------------------------------------------------------------------
# accept_current: consume the hold that is in force instead of waiting
# ---------------------------------------------------------------------------
class _StaticFence:
    """Already holding one ticket; the game never advances on its own here."""

    def __init__(self, ticket=11):
        self._ticket = ticket
        self.hold_phase = 1
        self.acks: list[int] = []
        self._telemetry = None

    @property
    def ticket(self):
        return self._ticket

    def telemetry(self):
        return self._telemetry

    def ack(self, ticket):
        self.acks.append(ticket)


def _static_setup(accept_current: bool, last_ticket: int):
    sampler, _fence, digest = _fake_sampler_setup()
    static = _StaticFence(11)
    # Re-use the telemetry the fake sampler created for ticket 11.
    static._telemetry = _fence_telemetry_from(sampler)
    sampler.fence = static
    sampler.accept_current = accept_current
    sampler.last_ticket = last_ticket
    return sampler, static


def _fence_telemetry_from(sampler):
    bits = tuple(
        struct.unpack("<I", struct.pack("<f", value))[0] for value in (192.5, 384.25, 0.0, 0.0)
    )
    digest = sampler.digest.bullet_digest(
        type("B", (), {"raw": np.zeros((spec.BULLET_COUNT, spec.BULLET_SIZE), np.uint8)})()
    )
    return framefence.FenceTelemetry(
        ticket=11,
        logic_frame=5000,
        stage_frames=4000,
        run_frames=4000,
        player_state=0,
        focus_mode=0,
        input_mask=spec.BTN_SHOOT,
        gui_input_current=spec.BTN_SHOOT,
        active_bullets=7,
        bullet_digest=digest,
        digest_ticket=11,
        hold_phase=1,
        _bits=bits,
    )


def test_accept_current_consumes_the_held_ticket():
    sampler, fence = _static_setup(accept_current=True, last_ticket=10)
    started = time.perf_counter()
    sample = sampler.next(timeout=0.2, attempts=1)
    elapsed = time.perf_counter() - started
    assert sample is not None and sample.ticket == 11 and sample.ok
    assert fence.acks == [11]
    assert sampler.last_ticket == 11
    # It must not have waited for a change that never comes.
    assert elapsed < 0.1


def test_wait_for_change_still_ignores_the_held_ticket_by_default():
    sampler, fence = _static_setup(accept_current=False, last_ticket=10)
    assert sampler.next(timeout=0.05, attempts=1) is None
    assert fence.acks == []


def test_accept_current_does_not_reuse_a_consumed_ticket():
    sampler, fence = _static_setup(accept_current=True, last_ticket=11)
    assert sampler.next(timeout=0.05, attempts=1) is None
    assert fence.acks == []
