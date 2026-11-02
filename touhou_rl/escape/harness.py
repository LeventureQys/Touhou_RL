"""The single observation/input pipeline both experiment arms must share (v3.7.1 P0).

Why a new loop instead of ``live.run_live_closed_loop``
-------------------------------------------------------
The v3.7.0 confirmation loop:

* terminated on ``decisions == frames // 2`` (900), never on ``gameplay_frame``;
* let the game run while planning, so the *previous* action's hold span was
  ``frameskip + planning time`` -- arm-dependent by construction;
* recorded only subject-side timing, so baseline exposure was unknown.

This module fixes all three by construction:

1. **One clock.**  The loop is driven by :class:`escape.clock.ExperimentClock`
   and stops on ``FrameWindow.end_frame``, never on a decision count.  After the
   last action slot it still waits for the window boundary, so a death on the
   final frames is observed instead of being truncated away.
2. **One frame schedule.**  Decision ``k`` observes at
   ``start + k*frameskip`` and its action is sent at
   ``obs_frame + actuation_lag_frames``; the next action replaces it at
   ``obs_frame + frameskip + actuation_lag_frames``.  Every action is therefore
   exposed for exactly ``frameskip`` logical frames regardless of how long
   planning took.  Planning that exceeds the lag becomes a recorded deadline
   miss *and* a measured shorter hold, not a silent unequal exposure.  An
   ``actuation_lag_frames`` of 0 reproduces the v3.7.0 confound and exists only
   for the C-04 latency comparison.
3. **One snapshot per decision.**  Both arms are handed the same
   ``adapter.read_synced`` result, so baseline and subject cannot see different
   frames (v3.7.0 issued its own reads, one per arm).

What is recorded per decision
-----------------------------
``obs_frame``/``obs_late_frames``; ``plan_start_frame``; ``send_frame`` sampled
before and after the key write; ``hold_frames`` (the previous action's real
exposure); effective-input timing from the engine's ``g_CurFrameInput``
readback, expressed as a *point* only when a non-matching frame precedes the
match, otherwise as an explicit interval with status ``already_active`` or
``unknown_interval``; the first observable motion frame; and the per-stage wall
times (read / plan / send / wait).

That is C-01, C-02, C-03 and C-04 of the v3.7.1 gate list.  Nothing here imports
the game, so the whole file runs against a fake port.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from .. import spec
from . import adapter, clock as clock_module, contracts

HARNESS_SCHEMA = "v371-harness/1"

#: Buttons whose engine readback proves an action was consumed.
CONTROL_MASK = (
    spec.BTN_UP
    | spec.BTN_DOWN
    | spec.BTN_LEFT
    | spec.BTN_RIGHT
    | spec.BTN_SHOOT
    | spec.BTN_FOCUS
    | spec.BTN_BOMB
)

_KEY_ORDER = (
    (spec.BTN_UP, "up"),
    (spec.BTN_DOWN, "down"),
    (spec.BTN_LEFT, "left"),
    (spec.BTN_RIGHT, "right"),
    (spec.BTN_SHOOT, "z"),
    (spec.BTN_FOCUS, "shift"),
    (spec.BTN_BOMB, "x"),
)


def keys_for_buttons(buttons: int) -> set[str]:
    """Buttons -> the key names the keyboard backends understand."""

    return {name for bit, name in _KEY_ORDER if int(buttons) & bit}


def keys_for_action(action: tuple[int, int, int, int]) -> set[str]:
    return keys_for_buttons(spec.action_to_buttons(tuple(int(v) for v in action)))


# ---------------------------------------------------------------------------
# Ports and snapshots
# ---------------------------------------------------------------------------
class GamePort(Protocol):
    """The three calls the pipeline needs from a game (or a fake in tests)."""

    def read_state(self, with_bullets: bool = False, with_title: bool = False) -> Any: ...

    def read_bullets(self) -> Any: ...

    def set_keys(self, keys: set[str]) -> None: ...


@dataclass
class Snapshot:
    """One synchronized read, shared by every arm of this decision."""

    observation: contracts.EscapeObservation | None
    frame: int
    stable: bool
    attempts: int
    reasons: tuple[str, ...]
    completeness: str
    hazards: int
    state: Any = None
    bullet_field: Any = None
    _network_obs: dict | None = field(default=None, repr=False)

    def network_obs(self) -> dict | None:
        """The gym-style observation the legacy pilot consumes (built once)."""

        if self._network_obs is None and self.state is not None:
            from .. import obs as obs_module

            encoded = obs_module.encode(self.state, self.bullet_field)
            self._network_obs = {
                "self_feat": encoded.self_feat,
                "bullets": encoded.bullets,
                "danger": encoded.danger_map,
            }
        return self._network_obs

    def raw_bullets(self) -> dict | None:
        """Absolute-coordinate bullet dict in the shape ``AvoidPilot.act`` expects."""

        if self.bullet_field is None:
            return None
        mask = self.bullet_field.active_mask()
        half = np.asarray(self.bullet_field.half_size[mask], dtype=np.float32)
        return {
            "position": np.array(self.bullet_field.position[mask], dtype=np.float32),
            "velocity": np.array(self.bullet_field.velocity[mask], dtype=np.float32),
            "half": np.maximum(half.max(axis=1, initial=0.5), 0.5) if half.size else half,
            "half_xy": half,
            "slot_id": np.flatnonzero(mask).astype(np.int32),
            "state": np.array(self.bullet_field.state[mask], dtype=np.uint16),
            "collision_disabled": np.array(self.bullet_field.disabled[mask], dtype=np.int8),
        }


def read_snapshot(
    port: GamePort,
    *,
    last_frame: int | None,
    attempts: int = 4,
    frameskip: int = 2,
    episode_id: str = "live",
    decision_id: int = 0,
) -> Snapshot:
    """One ``read_synced`` call, degraded to a ``Snapshot`` when it is unstable."""

    synced = adapter.read_synced(port, last_frame=last_frame, attempts=attempts, sleep=0.0)
    observation = None
    completeness = "unreadable"
    hazards = 0
    if synced.stable:
        try:
            observation = adapter.to_observation(
                synced,
                episode_id=episode_id,
                decision_id=decision_id,
                frameskip=frameskip,
            )
            completeness = observation.completeness.value
            hazards = 0 if observation.hazards is None else len(observation.hazards)
        except contracts.ContractError:
            observation = None
            completeness = "unreadable"
    return Snapshot(
        observation=observation,
        frame=int(synced.frame_id),
        stable=bool(synced.stable),
        attempts=int(synced.attempts),
        reasons=tuple(synced.reasons),
        completeness=completeness,
        hazards=int(hazards),
        state=synced.state,
        bullet_field=synced.field,
    )


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmContext:
    """Everything an arm may condition on besides the snapshot itself."""

    slot: int
    obs_frame: int
    frameskip: int
    act_lag: int
    previous_action: tuple[int, int]  # (focus, move), the controller's convention
    bombs: int
    episode_id: str


@dataclass
class ArmDecision:
    """One arm's output, plus whatever provenance the arm wants recorded."""

    action: tuple[int, int, int, int]
    intent: str = "n/a"
    route_id: str | None = None
    reason: str = ""
    deadline_miss: bool = False
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        move, shoot, focus, bomb = (int(v) for v in self.action)
        if not 0 <= move <= 8 or shoot not in (0, 1) or focus not in (0, 1) or bomb not in (0, 1):
            raise ValueError(f"invalid action {self.action}")
        self.action = (move, shoot, focus, bomb)


class Arm(Protocol):
    name: str

    def reset(self, episode_id: str) -> None: ...

    def decide(self, snapshot: Snapshot, context: ArmContext) -> ArmDecision: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HarnessConfig:
    """Protocol constants.  Every one of them is identical for both arms."""

    #: Window length in *logical frames* (the only termination basis).
    frames: int = 1800
    #: Logical frames each action is held.  Never derived from a decision count.
    frameskip: int = 2
    #: Frames between the observation and the key write.  1 removes planning
    #: time from the action schedule; 0 reproduces the v3.7.0 confound.
    actuation_lag_frames: int = 1
    #: Extra transport latency injected *after* planning (C-04), milliseconds.
    inject_delay_ms: float = 0.0
    #: Frames after a send during which effective-input readback is attempted.
    effective_probe_frames: int = 3
    #: Hard cap on decisions; exceeding it is a technical failure, not a window end.
    max_decisions: int = 4000
    #: Wall-clock budget for the whole episode.  Exceeding it aborts with
    #: ``wall_budget``; it never ends the window successfully.
    wall_budget_s: float = 180.0
    #: Poll period while waiting for a frame (2 ms ~ 0.12 frames at 60 Hz).
    poll_s: float = 0.002
    #: Same-frame read attempts per snapshot.
    read_attempts: int = 4
    #: Frames a snapshot may be ahead of its slot before it counts as stale.
    max_stale: int = 2
    #: Transition width in frames beyond which the clock records a jump.
    jump_threshold: int = 12
    #: Fixed start frame (None = the first frame observed after reset).
    start_frame: int | None = None
    #: Finish the episode at the first death (diagnostic mode only).
    stop_on_death: bool = False
    #: Keep the per-frame series (position/deaths/input).  Off halves the JSON.
    record_frames: bool = True

    def __post_init__(self) -> None:
        if self.frames < 1:
            raise ValueError("frames must be >= 1")
        if self.frameskip < 1:
            raise ValueError("frameskip must be >= 1")
        if not 0 <= self.actuation_lag_frames <= self.frameskip:
            # A lag > frameskip would make consecutive actions overlap, which puts
            # exposure back under the control of execution order.
            raise ValueError("actuation_lag_frames must be in 0..frameskip")


# ---------------------------------------------------------------------------
# Effective-input probe
# ---------------------------------------------------------------------------
@dataclass
class _Probe:
    """Pending effective-input measurement for one sent action."""

    requested_buttons: int
    send_frame: int
    obs_x: float
    obs_y: float
    expected: tuple[float, float]
    mask_first: int | None = None
    nonzero_seen_before: bool = False
    spurious: bool = False
    motion_frame: int | None = None
    frames: int = 0
    last_frame: int = -1

    def feed(self, reading: clock_module.FrameReading) -> None:
        """Fold one observed frame into the measurement (call in frame order)."""

        frame = int(reading.frame)
        if frame <= self.last_frame or frame < self.send_frame:
            return
        self.last_frame = frame
        self.frames += 1
        mask = int(reading.input_mask) & CONTROL_MASK
        want = self.requested_buttons & CONTROL_MASK
        if mask == want:
            if self.mask_first is None:
                self.mask_first = frame
        else:
            if self.mask_first is None:
                # A frame with our input *not* applied: the next match is a real
                # transition, not a readback coincidence.
                self.nonzero_seen_before = True
            else:
                # It matched and then stopped matching on its own: not evidence.
                self.spurious = True
        if self.motion_frame is None and self.expected != (0.0, 0.0):
            dx = float(reading.x) - self.obs_x
            dy = float(reading.y) - self.obs_y
            if dx * self.expected[0] + dy * self.expected[1] > 0.05:
                self.motion_frame = frame

    def summary(self, probe_frames: int) -> dict:
        """Point-vs-interval answer to "when did the input take effect?"."""

        motion = None if self.motion_frame is None else int(self.motion_frame)
        motion_after = None if motion is None else motion - int(self.send_frame)
        if self.mask_first is not None and self.nonzero_seen_before and not self.spurious:
            return {
                "status": "observed",
                "effective_frame": int(self.mask_first),
                "lower": int(self.send_frame),
                "upper": int(self.mask_first),
                "frames_after_send": int(self.mask_first) - int(self.send_frame),
                "motion_frame": motion,
                "motion_frames_after_send": motion_after,
                "observed_frames": int(self.frames),
            }
        if self.mask_first is not None and not self.spurious:
            # Our bits were already the engine's input on the first frame we
            # looked at, so "this action becomes effective" happened at or before
            # the send.  Reporting send_frame as the effective frame would be
            # a fabricated point estimate.
            return {
                "status": "already_active",
                "effective_frame": None,
                "lower": int(self.send_frame),
                "upper": int(self.send_frame),
                "frames_after_send": 0,
                "motion_frame": motion,
                "motion_frames_after_send": motion_after,
                "observed_frames": int(self.frames),
            }
        return {
            "status": "unknown_interval",
            "effective_frame": None,
            "lower": int(self.send_frame),
            "upper": int(self.send_frame) + int(probe_frames),
            "frames_after_send": None,
            "motion_frame": motion,
            "motion_frames_after_send": motion_after,
            "observed_frames": int(self.frames),
            "note": "engine input readback never showed the requested bits",
        }


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def run_episode(
    port: GamePort,
    arm: Arm,
    *,
    config: HarnessConfig | None = None,
    episode_id: str = "live",
    arm_name: str | None = None,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    start_reading: clock_module.FrameReading | None = None,
) -> dict:
    """Drive one episode, from ``reset`` to the frame-window end.

    Returns a JSON-safe document with the per-decision trace, the event ledger,
    the exposure accounting and the clock audit.  A run that could not keep its
    protocol (rollback, stall, left game, wall budget, stale reads) is returned
    with ``technical_failure`` set rather than silently shortened.
    """

    config = config or HarnessConfig()
    mono = monotonic or time.perf_counter
    napper = sleep or time.sleep
    started = mono()

    clock = clock_module.ExperimentClock(
        port.read_state,
        monotonic=mono,
        sleep=napper,
        config=clock_module.ClockConfig(jump_threshold=config.jump_threshold),
    )

    # -- establish the start frame -------------------------------------
    first = start_reading if start_reading is not None else clock.read()
    start_frame = int(config.start_frame if config.start_frame is not None else first.frame)
    window = clock_module.FrameWindow(start_frame=start_frame, frames=config.frames)
    rule = clock_module.TerminationRule(window=window, frameskip=config.frameskip)
    ledger = clock_module.EventLedger(window=window)
    ledger.observe(first)

    arm.reset(episode_id)
    arm_label = arm_name or getattr(arm, "name", type(arm).__name__)

    rows: list[dict] = []
    frame_series: list[dict] = [] if config.record_frames else []
    samples_seen: set[int] = set()
    observed: dict[int, clock_module.FrameReading] = {}

    def note_reading(reading: clock_module.FrameReading) -> None:
        frame = int(reading.frame)
        if frame not in samples_seen:
            samples_seen.add(frame)
            ledger.observe(reading)
            if config.record_frames:
                frame_series.append(
                    {
                        "frame": frame,
                        "x": round(float(reading.x), 4),
                        "y": round(float(reading.y), 4),
                        "deaths": int(reading.deaths),
                        "bombs": round(float(reading.bombs), 3),
                        "player_state": int(reading.player_state),
                        "input_mask": int(reading.input_mask),
                    }
                )
        observed.setdefault(frame, reading)
        # The pending effective-input probe sees every frame between two sends,
        # not just the hold: a one-frame readback window would miss the transition
        # half the time and then have to claim an interval it never measured.
        if probe is not None:
            probe.feed(reading)

    def note_samples(outcome: clock_module.WaitOutcome) -> None:
        for reading in outcome.samples:
            note_reading(reading)

    previous_action: tuple[int, int] = (0, 0)
    previous_send: int | None = None
    previous_buttons: int | None = None
    probe: _Probe | None = None
    pending_row: int | None = None
    termination = "running"
    technical_failure: str | None = None
    slot = 0
    rejected = 0
    late_slots = 0
    current = first
    end_frame = int(start_frame)

    def fail(reason: str) -> None:
        nonlocal termination, technical_failure
        termination = reason
        technical_failure = reason

    try:
        while True:
            end_frame = int(current.frame)
            if not current.in_game:
                fail("left_game")
                break
            if clock.epoch > 0:
                fail("clock_rollback")
                break
            if config.stop_on_death and ledger.deaths:
                termination = "death"
                break
            if not rule.fits(slot, config.actuation_lag_frames):
                termination = "window_complete"
                break
            if slot >= config.max_decisions:
                fail("max_decisions")
                break
            if mono() - started > config.wall_budget_s:
                fail("wall_budget")
                break

            slot_start = rule.slot_start(slot)
            wait = clock.wait_until(
                slot_start,
                timeout_s=max(1.0, 6.0 * config.frameskip / 60.0),
                poll_s=config.poll_s,
            )
            note_samples(wait)
            if wait.reason in ("left_game", "rollback", "timeout", "stall"):
                fail(f"clock_{wait.reason}")
                current = wait.reading
                break

            # -- observe -------------------------------------------------
            t_read0 = mono()
            snapshot = read_snapshot(
                port,
                last_frame=clock.last_frame,
                attempts=config.read_attempts,
                frameskip=config.frameskip,
                episode_id=episode_id,
                decision_id=slot,
            )
            t_read1 = mono()
            obs_late = int(snapshot.frame) - int(slot_start)
            if snapshot.observation is None:
                # Retry on the next slot instead of aborting: an unreadable frame
                # is a *skipped* decision and must be counted, not silently dropped
                # and not allowed to end the window early.
                rows.append(_rejected_row(slot, snapshot, obs_late, t_read1 - t_read0))
                rejected += 1
                slot += 1
                if rejected > max(8, config.max_decisions // 4):
                    fail("too_many_unstable_snapshots")
                    break
                current = clock.last or current
                continue
            if obs_late > 0:
                # The planner is behind its own slot.  Record it and keep going:
                # the frame window still ends on ``end_frame``, so a slow planner
                # costs measured exposure, never an unequal experiment length.
                late_slots += 1

            observation = snapshot.observation
            bombs = int(getattr(observation, "bombs", 0) or 0)
            context = ArmContext(
                slot=slot,
                obs_frame=int(snapshot.frame),
                frameskip=config.frameskip,
                act_lag=config.actuation_lag_frames,
                previous_action=previous_action,
                bombs=bombs,
                episode_id=episode_id,
            )

            # -- plan ----------------------------------------------------
            t_plan0 = mono()
            decision = arm.decide(snapshot, context)
            t_plan1 = mono()
            if config.inject_delay_ms > 0:
                napper(config.inject_delay_ms / 1000.0)

            # -- send at the scheduled frame -----------------------------
            send_target = int(snapshot.frame) + int(config.actuation_lag_frames)
            send_wait = clock.wait_until(
                send_target,
                timeout_s=max(1.0, 6.0 * config.frameskip / 60.0),
                poll_s=config.poll_s,
            )
            note_samples(send_wait)
            if send_wait.reason in ("left_game", "rollback", "timeout", "stall"):
                fail(f"clock_{send_wait.reason}")
                current = send_wait.reading
                break

            buttons = spec.action_to_buttons(decision.action)
            keys = keys_for_buttons(buttons)
            frame_before_send = clock.last_frame
            t_send0 = mono()
            try:
                port.set_keys(keys)
            except OSError as error:
                fail(f"input_failure: {error}")
                current = clock.last or current
                break
            t_send1 = mono()
            send_reading = clock.read()
            note_reading(send_reading)
            frame_after_send = int(send_reading.frame)

            # Finalize the previous action's effective-input measurement now that
            # its whole exposure window has been observed, then start a new one.
            if probe is not None and pending_row is not None:
                rows[pending_row]["effective"] = probe.summary(config.effective_probe_frames)
            probe = _Probe(
                requested_buttons=int(buttons),
                send_frame=frame_after_send,
                obs_x=float(observation.player.x),
                obs_y=float(observation.player.y),
                expected=_expected_displacement(decision.action, config.frameskip),
            )
            pending_row = len(rows)
            probe.feed(send_reading)

            send_late = frame_after_send - int(send_target)
            hold = None if previous_send is None else frame_after_send - int(previous_send)
            previous_send = frame_after_send
            previous_action = (int(decision.action[2]), int(decision.action[0]))
            buttons_changed = previous_buttons is not None and int(buttons) != int(previous_buttons)
            previous_buttons = int(buttons)

            row: dict = {
                "decision": int(slot),
                "obs_frame": int(snapshot.frame),
                "slot_start_frame": int(slot_start),
                "obs_late_frames": int(obs_late),
                "obs_attempts": int(snapshot.attempts),
                "obs_completeness": snapshot.completeness,
                "hazards": int(snapshot.hazards),
                "send_frame_before": None if frame_before_send is None else int(frame_before_send),
                "send_frame_after": frame_after_send,
                "send_target_frame": int(send_target),
                "send_late_frames": int(send_late),
                "hold_frames": hold,
                "buttons": int(buttons),
                "buttons_changed": bool(buttons_changed),
                "action": list(decision.action),
                "intent": decision.intent,
                "route_id": decision.route_id,
                "reason": decision.reason[:200],
                "deadline_miss": bool(decision.deadline_miss),
                "plan_over_lag": bool(send_late > 0),
                "bombs": bombs,
                "read_ms": round((t_read1 - t_read0) * 1000.0, 4),
                "plan_ms": round((t_plan1 - t_plan0) * 1000.0, 4),
                "send_ms": round((t_send1 - t_send0) * 1000.0, 4),
                "obs_x": round(float(observation.player.x), 4),
                "obs_y": round(float(observation.player.y), 4),
                "deaths_at_send": int(getattr(snapshot.state, "deaths", 0) or 0),
                "alive": bool(getattr(observation.player, "alive", True)),
                "diagnostics": _json_safe(decision.diagnostics),
            }
            rows.append(row)
            slot += 1

            # -- hold until the next observation slot ---------------------
            # The next action is sent at ``slot_start + frameskip + lag``, so the
            # current one is exposed for exactly ``frameskip`` frames.  Waiting for
            # ``send + frameskip`` instead would make every decision cost
            # ``frameskip + lag`` frames and the schedule would slip one frame per
            # decision -- the same class of bug as v3.7.0, only quieter.
            hold_target = frame_after_send + int(config.frameskip) - int(config.actuation_lag_frames)
            hold_wait = clock.wait_until(
                hold_target,
                timeout_s=max(1.0, 6.0 * config.frameskip / 60.0),
                poll_s=config.poll_s,
            )
            note_samples(hold_wait)
            current = hold_wait.reading
            if hold_wait.reason in ("left_game", "rollback", "timeout", "stall"):
                fail(f"clock_{hold_wait.reason}")
                break

        # -- run out the remaining window so the boundary is observed ----
        # Not after a diagnostic ``stop_on_death``: that mode exists precisely to
        # freeze the frames around a death for forensics.
        if technical_failure is None and termination in ("window_complete", "running"):
            boundary = clock.wait_until(
                window.end_frame,
                timeout_s=max(2.0, 12.0 * config.frameskip / 60.0),
                poll_s=config.poll_s,
            )
            note_samples(boundary)
            if boundary.reason in ("timeout", "stall", "rollback", "left_game"):
                if boundary.reason != "left_game":
                    fail(f"clock_boundary_{boundary.reason}")
            end_frame = int(clock.last_frame if clock.last_frame is not None else end_frame)
    finally:
        try:
            port.set_keys(set())
        except Exception:  # noqa: BLE001 - releasing keys must never mask the run
            pass

    if probe is not None and pending_row is not None and pending_row < len(rows):
        rows[pending_row]["effective"] = probe.summary(config.effective_probe_frames)

    end_frame = int(clock.last_frame if clock.last_frame is not None else end_frame)
    exposure = window.exposure(end_frame)
    deaths = ledger.summary()
    holds = [row["hold_frames"] for row in rows if row.get("hold_frames") is not None]
    lateness = [row["send_late_frames"] for row in rows if "send_late_frames" in row]
    document = {
        "schema": HARNESS_SCHEMA,
        "episode_id": episode_id,
        "arm": arm_label,
        "config": _config_dict(config),
        "window": window.as_dict(),
        "start_state": first.as_dict(),
        "termination": termination,
        "technical_failure": technical_failure,
        "end_frame": end_frame,
        "exposure_frames": int(exposure),
        "window_frames": int(window.frames),
        "exposure_shortfall_frames": int(max(0, window.frames - exposure)),
        "exposure_overshoot_frames": int(max(0, end_frame - window.end_frame)),
        "observed_frames_seen": len(samples_seen),
        "frame_gaps": _gaps(sorted(samples_seen)),
        "decisions": len(rows),
        "rejected_snapshots": int(rejected),
        "late_slots": int(late_slots),
        "planned_slots": int(rule.slots),
        "deaths_in_window": int(deaths["deaths_in_window"]),
        "deaths_before_window": int(deaths["deaths_before_window"]),
        "deaths_after_window": int(deaths["deaths_after_window"]),
        "boundary_unknown": int(deaths["boundary_unknown"]),
        "death_events": deaths["deaths"],
        "bomb_events": deaths["bombs"],
        "bombs_in_window": int(deaths["bomb_count"]),
        "first_death_frame": _first_inside(deaths["deaths"]),
        "timing_ms": _timing(rows),
        "hold_frames": _describe(holds),
        "send_lateness_frames": _describe(lateness),
        "deadline_misses": int(sum(1 for row in rows if row["deadline_miss"])),
        "plan_over_lag": int(sum(1 for row in rows if row.get("plan_over_lag"))),
        "effective": _effective_summary(rows),
        "clock": clock.summary(),
        "rows": rows,
    }
    if config.record_frames:
        document["frames"] = frame_series
    return document


def _expected_displacement(action: tuple[int, int, int, int], frames: int) -> tuple[float, float]:
    """Where the calibrated movement model says one held action moves the player."""

    try:
        from . import geometry

        delta = geometry.MoveModel().displacement(int(action[2]), int(action[0]), frames=1)
        return (float(delta[0]) * int(frames), float(delta[1]) * int(frames))
    except Exception:  # noqa: BLE001 - a missing model must not fail a live run
        return (0.0, 0.0)


def _rejected_row(slot: int, snapshot: Snapshot, obs_late: int, read_s: float) -> dict:
    return {
        "decision": int(slot),
        "obs_frame": int(snapshot.frame),
        "obs_late_frames": int(obs_late),
        "obs_attempts": int(snapshot.attempts),
        "obs_completeness": snapshot.completeness,
        "status": "rejected",
        "reasons": list(snapshot.reasons),
        "read_ms": round(float(read_s) * 1000.0, 4),
    }


def _config_dict(config: HarnessConfig) -> dict:
    return {
        "frames": int(config.frames),
        "frameskip": int(config.frameskip),
        "actuation_lag_frames": int(config.actuation_lag_frames),
        "inject_delay_ms": float(config.inject_delay_ms),
        "effective_probe_frames": int(config.effective_probe_frames),
        "max_stale": int(config.max_stale),
        "poll_s": float(config.poll_s),
        "jump_threshold": int(config.jump_threshold),
        "stop_on_death": bool(config.stop_on_death),
        "start_frame": None if config.start_frame is None else int(config.start_frame),
    }


def _describe(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(int(value) for value in values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
        "mean": round(float(sum(ordered)) / len(ordered), 3),
        "distribution": {str(value): ordered.count(value) for value in sorted(set(ordered))},
    }


def _timing(rows: list[dict]) -> dict:
    out: dict = {}
    for key in ("read_ms", "plan_ms", "send_ms"):
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        ordered = sorted(values)
        out[key] = {
            "p50": round(ordered[len(ordered) // 2], 3),
            "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
            "max": round(ordered[-1], 3),
        }
    return out


def _effective_summary(rows: list[dict]) -> dict:
    entries = [row["effective"] for row in rows if isinstance(row.get("effective"), dict)]
    observed = [entry for entry in entries if entry.get("status") == "observed"]
    already = [entry for entry in entries if entry.get("status") == "already_active"]
    unknown = [entry for entry in entries if entry.get("status") == "unknown_interval"]
    deltas = [e["frames_after_send"] for e in observed if e.get("frames_after_send") is not None]
    motion = [e["motion_frames_after_send"] for e in entries if e.get("motion_frames_after_send") is not None]
    # The number that matters for C-03: of the decisions that actually *changed*
    # the held buttons, how many had their transition observed?  A repeated action
    # is trivially "already active" and must not inflate the resolution rate.
    changed = [
        row["effective"]
        for row in rows
        if row.get("buttons_changed") and isinstance(row.get("effective"), dict)
    ]
    changed_observed = [entry for entry in changed if entry.get("status") == "observed"]
    return {
        "entries": len(entries),
        "observed": len(observed),
        "already_active": len(already),
        "unknown_interval": len(unknown),
        "resolution_rate": round(len(observed) / len(entries), 4) if entries else None,
        "changed_entries": len(changed),
        "changed_resolved": len(changed_observed),
        "changed_resolution_rate": (
            round(len(changed_observed) / len(changed), 4) if changed else None
        ),
        "frames_after_send": _describe(deltas),
        "motion_frames_after_send": _describe(motion),
    }


def _gaps(frames: list[int]) -> dict:
    ordered = sorted(int(frame) for frame in frames)
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b - a > 1]
    return {
        "count": len(gaps),
        "max": max(gaps) if gaps else 0,
        "frames_unobserved": int(sum(gap - 1 for gap in gaps)),
    }


def _first_inside(events: list[dict]) -> int | None:
    for event in events:
        if event.get("attribution") == "inside":
            return int(event["frame_lower"])
    return None


def _json_safe(value: Any) -> Any:
    """Make numpy scalars/arrays and enums safe for ``json.dump``."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# AB/BA ordering (C-05)
# ---------------------------------------------------------------------------
def plan_order(
    pairs: int,
    *,
    scheme: str = "balanced_random",
    seed: int = 0,
    first: str = "baseline",
    second: str = "subject",
) -> list[tuple[str, ...]]:
    """The per-pair arm order, recorded before the run so it cannot be chosen after.

    ``scheme``:

    * ``ab`` -- ``first`` then ``second`` (what v3.7.0 did, and the reason session
      drift could not be separated from an arm effect);
    * ``ba`` -- the mirror;
    * ``balanced_random`` -- a seeded coin flip per pair, re-balanced so the two
      orders differ by at most one pair.  This is the default for C-05 because
      it removes the "first arm always sees the fresher session" bias.

    ``first``/``second`` are arm *names*, so the same runner can compare
    ``baseline`` vs ``hybrid_veto`` and ``delayed_baseline`` vs ``hybrid_veto``
    (the C-04 latency control) without a second implementation.
    """

    if pairs < 1:
        raise ValueError("pairs must be >= 1")
    if scheme == "ab":
        return [(first, second) for _ in range(pairs)]
    if scheme == "ba":
        return [(second, first) for _ in range(pairs)]
    if scheme != "balanced_random":
        raise ValueError(f"unknown order scheme {scheme}")
    rng = np.random.default_rng(int(seed))
    orders: list[tuple[str, ...]] = []
    ab = ba = 0
    for _ in range(pairs):
        if ab - ba >= 1:
            first_ab = False
        elif ba - ab >= 1:
            first_ab = True
        else:
            first_ab = bool(rng.integers(0, 2))
        if first_ab:
            orders.append((first, second))
            ab += 1
        else:
            orders.append((second, first))
            ba += 1
    return orders
