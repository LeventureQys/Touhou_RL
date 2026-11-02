"""v3.7.1 experiment driver: one episode/pair protocol over the shared harness.

This module is the formal tool that replaces the ``.tmp/live_pairs.py`` /
``.tmp/live_arm.py`` / ``.tmp/live_timing.py`` scripts.  It owns the things the
harness deliberately does not:

* the game session and the ``reset`` step, including the reset-phase record
  (start frame, stage, deaths, and a fingerprint of the initial bullet pool --
  TH08's RNG is not exposed by the reader, so "same session == same initial
  curtain" cannot be claimed and is not claimed here);
* AB/BA ordering, recorded before the run;
* the gate arithmetic for C-01..C-04, runnable **without a game** so that the
  clock fixes are verifiable on any machine;
* replay comparison ("same input -> same trajectory"), which is the only
  evidence that the pipeline itself does not inject state.

Nothing here is allowed to be the only copy of a decision: every number it emits
is recomputable from the raw JSON it writes (``tools/rl/v371_recompute.py``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import arms as arms_module
from . import clock as clock_module
from . import controller, fakegame, harness

EXPERIMENT_SCHEMA = "v371-experiment/1"
PAIR_SCHEMA = "v371-pair-session/1"
GATE_SCHEMA = "v371-gates/1"


# ---------------------------------------------------------------------------
# Live port
# ---------------------------------------------------------------------------
class LivePort:
    """Adapt a ``Th08Env`` session to the three calls the harness needs."""

    def __init__(self, env: Any) -> None:
        self.env = env

    @property
    def reader(self):
        assert self.env.session is not None and self.env.session.reader is not None
        return self.env.session.reader

    @property
    def keyboard(self):
        assert self.env.session is not None and self.env.session.keyboard is not None
        return self.env.session.keyboard

    def read_state(self, with_bullets: bool = False, with_title: bool = False):
        return self.reader.read_state(with_bullets=with_bullets, with_title=with_title)

    def read_bullets(self):
        return self.reader.read_bullets()

    def set_keys(self, keys: set[str]) -> None:
        self.keyboard.set_keys(keys)


def bullet_fingerprint(field: Any) -> dict:
    """A cheap, order-stable fingerprint of the bullet pool (world-state proxy)."""

    if field is None:
        return {"bullets": None}
    mask = np.asarray(field.active_mask())
    count = int(mask.sum())
    if count == 0:
        return {"bullets": 0, "sum_x": 0.0, "sum_y": 0.0}
    positions = np.asarray(field.position)[mask]
    return {
        "bullets": count,
        "sum_x": round(float(positions[:, 0].sum()), 2),
        "sum_y": round(float(positions[:, 1].sum()), 2),
    }


def reset_phase(env: Any, *, index: int, arm: str) -> dict:
    """Record everything observable about *where* an episode starts.

    TH08's RNG is not exposed, so the record says so explicitly instead of
    implying two arms shared a world.  The bullet fingerprint at the start frame
    is the closest available proxy, and it is only a proxy.
    """

    observation, info = env.reset()
    port = LivePort(env)
    state = port.read_state(with_bullets=False)
    try:
        field = port.read_bullets()
    except OSError:
        field = None
    return {
        "index": int(index),
        "arm": arm,
        "start_frame": int(state.gameplay_frame),
        "stage": int(state.stage),
        "run_frames": int(getattr(state, "run_frames", 0)),
        "deaths": int(getattr(state, "deaths", 0)),
        "bombs": float(getattr(state, "bombs", 0.0)),
        "lives": float(getattr(state, "lives", 0.0)),
        "supervisor_state": int(state.supervisor_state),
        "initial_pool": bullet_fingerprint(field),
        "rng_control": "unavailable: reader exposes no RNG state; same session is not the same curtain",
        "wall_s": round(time.time(), 3),
    }


# ---------------------------------------------------------------------------
# One arm, one episode
# ---------------------------------------------------------------------------
def run_live_episode(
    env: Any,
    arm,
    *,
    config: harness.HarnessConfig,
    episode_id: str,
    index: int = 0,
) -> dict:
    """Reset, run the window, and attach the reset-phase pair record."""

    phase = reset_phase(env, index=index, arm=getattr(arm, "name", "arm"))
    port = LivePort(env)
    document = harness.run_episode(
        port,
        arm,
        config=config,
        episode_id=episode_id,
        arm_name=getattr(arm, "name", None),
    )
    document["reset_phase"] = phase
    return document


# ---------------------------------------------------------------------------
# Paired session
# ---------------------------------------------------------------------------
def run_pair_session(
    env: Any,
    *,
    pairs: int,
    arm_names: tuple[str, str] = ("baseline", "subject"),
    order_scheme: str = "balanced_random",
    seed: int = 0,
    config: harness.HarnessConfig | None = None,
    arm_factory: Callable[[str], Any] | None = None,
    progress: Callable[[int, dict], None] | None = None,
    wall_budget_s: float = 120.0,
) -> dict:
    """Run ``pairs`` pairs in the preregistered order and return the raw record.

    ``arm_names`` is ``(first, second)`` and names the two conditions; the same
    runner therefore serves the P1 comparison (``baseline`` vs ``hybrid_veto``)
    and the C-04 latency control (``delayed_baseline`` vs ``hybrid_veto``)
    without a second implementation.

    Each pair runs both arms in the recorded order; a per-episode wall-clock
    overrun is a *technical failure* recorded on that episode, never a silent
    skip (v3.7.0 removed such pairs from the denominator).
    """

    config = config or harness.HarnessConfig()
    factory = arm_factory or (lambda name: arms_module.make_arm(name))
    if len(arm_names) != 2:
        raise ValueError("arm_names must name exactly two conditions")
    order = harness.plan_order(
        pairs, scheme=order_scheme, seed=seed, first=arm_names[0], second=arm_names[1]
    )
    entries: list[dict] = []
    for pair_index, sequence in enumerate(order):
        entry: dict = {"pair": pair_index + 1, "order": list(sequence)}
        for arm_name in sequence:
            arm = factory(arm_name)
            started = time.perf_counter()
            try:
                document = run_live_episode(
                    env,
                    arm,
                    config=config,
                    episode_id=f"p{pair_index + 1}-{arm_name}",
                    index=pair_index + 1,
                )
            except Exception as error:  # noqa: BLE001 - a failed episode is data
                entry[f"{arm_name}_error"] = f"{type(error).__name__}: {error}"
                continue
            elapsed = time.perf_counter() - started
            summary = _episode_summary(document)
            summary["elapsed_s"] = round(elapsed, 3)
            # Arms that can sit out decisions report their own counters, so the
            # report can say "the veto fired N times" instead of implying it ran.
            if hasattr(arm, "veto_fires"):
                summary["veto"] = {
                    "fired": int(getattr(arm, "veto_fires", 0)),
                    "decisions": int(getattr(arm, "decisions", 0)),
                    "rate": (
                        round(float(arm.veto_fires) / arm.decisions, 4)
                        if getattr(arm, "decisions", 0)
                        else None
                    ),
                }
            if elapsed > wall_budget_s:
                summary["technical_failure"] = summary["technical_failure"] or "wall_budget"
            entry[arm_name] = summary
        entries.append(entry)
        if progress is not None:
            progress(pair_index + 1, entry)

    return {
        "schema": PAIR_SCHEMA,
        "protocol": "preregistration/live-v1.md",
        "arm_names": list(arm_names),
        "order_scheme": order_scheme,
        "order_seed": int(seed),
        "order": [list(sequence) for sequence in order],
        "config": harness._config_dict(config),
        "pairs": entries,
        "pairs_complete": sum(
            1 for entry in entries if arm_names[0] in entry and arm_names[1] in entry
        ),
    }


def _episode_summary(document: dict) -> dict:
    """The fields a report needs, kept identical for both arms (C-05)."""

    return {
        "arm": document.get("arm"),
        "window": document["window"],
        "start_frame": document["window"]["start_frame"],
        "end_frame": document["end_frame"],
        "exposure_frames": document["exposure_frames"],
        "decisions": document["decisions"],
        "termination": document["termination"],
        "technical_failure": document["technical_failure"],
        "deaths_in_window": document["deaths_in_window"],
        "deaths_before_window": document["deaths_before_window"],
        "deaths_after_window": document["deaths_after_window"],
        "boundary_unknown": document["boundary_unknown"],
        "first_death_frame": document["first_death_frame"],
        "death_events": document["death_events"],
        "bombs_in_window": document["bombs_in_window"],
        "bomb_events": document["bomb_events"],
        "hold_frames": document["hold_frames"],
        "send_lateness_frames": document["send_lateness_frames"],
        "plan_over_lag": document["plan_over_lag"],
        "deadline_misses": document["deadline_misses"],
        "timing_ms": document["timing_ms"],
        "effective": document["effective"],
        "clock": document["clock"],
        "reset_phase": document.get("reset_phase"),
        "score": document.get("start_state", {}).get("frame"),
        "same_frame_pairs": document["observed_frames_seen"],
    }


# ---------------------------------------------------------------------------
# Replay check (same input -> same trajectory)
# ---------------------------------------------------------------------------
def replay_check(
    world_factory: Callable[[], tuple[Any, Any, Any]],
    actions,
    *,
    frames: int,
    frameskip: int = 2,
) -> dict:
    """Run the same recorded action sequence twice and compare the trajectories.

    ``world_factory()`` returns ``(port, monotonic, sleep)`` so a fake world can
    supply its own clock -- passing the real ``time.sleep`` with a fake game
    would spin forever, because the fake's frame counter only moves when its own
    clock is advanced.

    On the fake world equality must be exact (the C-01 hard form).  On the real
    game it may not be, because the curtain is not reproducible; the function
    therefore reports the divergence instead of asserting, and the caller decides
    what the numbers support.
    """

    action_list = tuple(tuple(int(v) for v in action) for action in actions)
    runs = []
    for _ in range(2):
        port, monotonic, sleep = world_factory()
        record = harness.run_episode(
            port,
            arms_module.ReplayArm(action_list),
            config=harness.HarnessConfig(frames=frames, frameskip=frameskip),
            episode_id="replay",
            monotonic=monotonic,
            sleep=sleep,
        )
        runs.append(record)
    first, second = runs
    positions_a = [(row["frame"], row["x"], row["y"]) for row in first.get("frames", [])]
    positions_b = [(row["frame"], row["x"], row["y"]) for row in second.get("frames", [])]
    common = min(len(positions_a), len(positions_b))
    deltas = [
        abs(positions_a[i][1] - positions_b[i][1]) + abs(positions_a[i][2] - positions_b[i][2])
        for i in range(common)
        if positions_a[i][0] == positions_b[i][0]
    ]
    return {
        "identical": positions_a == positions_b,
        "compared_frames": common,
        "max_abs_position_delta": round(max(deltas), 6) if deltas else None,
        "mean_abs_position_delta": round(float(np.mean(deltas)), 6) if deltas else None,
        "exposure_a": first["exposure_frames"],
        "exposure_b": second["exposure_frames"],
        "actions": len(action_list),
    }


# ---------------------------------------------------------------------------
# C-01..C-04 on the fake clock: the gate suite that needs no game
# ---------------------------------------------------------------------------
def _fake_episode(
    *,
    policy=None,
    frames: int = 240,
    game_kwargs: dict | None = None,
    config_kwargs: dict | None = None,
    arm=None,
) -> dict:
    fake_time = fakegame.FakeTime()
    game = fakegame.FakeGame(fake_time, **(game_kwargs or {}))
    if arm is None:
        work = {"frames": 0.0}
        if isinstance(policy, dict):
            work = policy
        work_s = float(work.get("frames", 0.0)) / 60.0

        def _policy(snapshot, context):
            if work_s:
                fake_time.advance(work_s)
            return (3, 1, 0, 0)

        arm = arms_module.ScriptedArm(_policy)
    config = harness.HarnessConfig(**{"frames": frames, **(config_kwargs or {})})
    document = harness.run_episode(
        game,
        arm,
        config=config,
        episode_id="fake-gate",
        monotonic=fake_time.monotonic,
        sleep=fake_time.sleep,
    )
    document["_game"] = game
    return document


def run_fake_gates() -> dict:
    """Execute C-01..C-04 against the deterministic fake clock.

    Every gate returns ``pass``/``fail`` plus the measurements it was decided on;
    nothing here is allowed to pass by absence of evidence.
    """

    gates: dict[str, dict] = {}

    # -- C-01: the clock is the termination basis ------------------------
    planning = {}
    for work in (0.0, 1.0, 3.0, 10.0):
        document = _fake_episode(policy={"frames": work}, frames=240)
        planning[f"{work:g}f"] = {
            "exposure_frames": document["exposure_frames"],
            "decisions": document["decisions"],
            "plan_over_lag": document["plan_over_lag"],
            "hold_max": document["hold_frames"].get("max"),
            "termination": document["termination"],
            "technical_failure": document["technical_failure"],
        }
    exposure_values = {entry["exposure_frames"] for entry in planning.values()}
    clean = all(entry["technical_failure"] is None for entry in planning.values())
    gates["C-01"] = {
        "status": "pass" if clean and exposure_values == {240} else "fail",
        "detail": "planning 0/1/3/10 frames all terminate on gameplay_frame",
        "planning": planning,
        "exposure_values": sorted(exposure_values),
    }

    # -- C-01b: rollback / stall / left game are classified --------------
    rollback = _fake_episode(game_kwargs={"rollback_at_read": 60}, frames=600)
    stall = _fake_episode(game_kwargs={"stall_from": 130}, frames=240)
    left = _fake_episode(game_kwargs={"supervisor_state": 3}, frames=240)
    rollback_ok = (
        rollback["technical_failure"] is not None
        and rollback["exposure_frames"] >= 0
        and rollback["clock"]["rollbacks"] >= 1
    )
    stall_ok = stall["technical_failure"] is not None and stall["exposure_frames"] >= 0
    left_ok = left["technical_failure"] is not None
    gates["C-01b"] = {
        "status": "pass" if (rollback_ok and stall_ok and left_ok) else "fail",
        "rollback": {
            "termination": rollback["termination"],
            "exposure_frames": rollback["exposure_frames"],
            "rollbacks": rollback["clock"]["rollbacks"],
        },
        "stall": {"termination": stall["termination"], "exposure_frames": stall["exposure_frames"]},
        "left_game": {"termination": left["termination"], "exposure_frames": left["exposure_frames"]},
    }

    # -- C-02: equal exposure for both arms ------------------------------
    # Both arms run the *same* config the live pair session uses, so this gate
    # measures the configuration that will actually be compared.
    live_controller = controller.ControllerConfig(
        planner=controller.planner.PlannerConfig(
            horizon_decisions=6,
            continuation_decisions=1,
            beam_width=1,
        )
    )
    baseline = _fake_episode(arm=arms_module.make_arm("baseline"), frames=240)
    subject = _fake_episode(
        arm=arms_module.make_arm("subject", controller_config=live_controller),
        frames=240,
    )
    exposure_equal = baseline["exposure_frames"] == subject["exposure_frames"] == 240
    windows_equal = baseline["window"] == subject["window"]
    hold_equal = baseline["hold_frames"] == subject["hold_frames"]
    gates["C-02"] = {
        "status": "pass" if (exposure_equal and windows_equal and hold_equal) else "fail",
        "baseline": {
            "exposure_frames": baseline["exposure_frames"],
            "decisions": baseline["decisions"],
            "deaths_in_window": baseline["deaths_in_window"],
            "hold_frames": baseline["hold_frames"].get("distribution"),
        },
        "subject": {
            "exposure_frames": subject["exposure_frames"],
            "decisions": subject["decisions"],
            "deaths_in_window": subject["deaths_in_window"],
            "hold_frames": subject["hold_frames"].get("distribution"),
        },
        "detail": "same window, same end frame, same hold distribution for both arms",
    }

    # -- C-02b: death attribution ---------------------------------------
    deaths = _fake_episode(game_kwargs={"deaths_at": (150, 170, 400)}, frames=240)
    inside = [event["frame_lower"] for event in deaths["death_events"] if event["attribution"] == "inside"]
    gates["C-02b"] = {
        "status": "pass" if inside == [150, 170] and deaths["boundary_unknown"] == 0 else "fail",
        "deaths_in_window": deaths["deaths_in_window"],
        "first_death_frame": deaths["first_death_frame"],
        "inside_frames": inside,
        "boundary_unknown": deaths["boundary_unknown"],
        "events": deaths["death_events"],
    }

    # -- C-03: send frame vs effective frame ----------------------------
    # The action must *change* for a transition to exist: a constant action makes
    # every frame "already active" and would let the gate pass vacuously.
    alternator = {"index": 0}

    def _alternate(snapshot, context):
        action = [(3, 1, 0, 0), (7, 1, 0, 0)][alternator["index"] % 2]
        alternator["index"] += 1
        return action

    timing = _fake_episode(arm=arms_module.ScriptedArm(_alternate), frames=240)
    effective = timing["effective"]
    resolved = effective["changed_entries"] > 0 and effective["changed_resolution_rate"] is not None
    gates["C-03"] = {
        "status": "pass" if resolved and effective["frames_after_send"].get("p50") is not None else "fail",
        "send_vs_effective": effective,
        "detail": (
            "engine g_CurFrameInput readback separates send_frame from effective_frame; "
            "only decisions that changed the held buttons count toward the resolution rate"
        ),
    }

    # -- C-03b: unresolved readback is an interval, not a point ---------
    blind = _fake_episode(game_kwargs={"input_latency_frames": 10_000}, frames=120)
    entries = [row["effective"] for row in blind["rows"] if isinstance(row.get("effective"), dict)]
    no_points = all(entry["effective_frame"] is None for entry in entries)
    gates["C-03b"] = {
        "status": "pass" if entries and no_points else "fail",
        "statuses": sorted({entry["status"] for entry in entries}),
        "detail": "an unobservable transition is reported as an interval with no point estimate",
    }

    # -- C-04: injected latency -----------------------------------------
    injected = {}
    for delay_ms in (0.0, 5.0, 15.0, 30.0, 50.0):
        document = _fake_episode(
            frames=240,
            config_kwargs={"inject_delay_ms": delay_ms},
        )
        injected[f"{delay_ms:g}ms"] = {
            "exposure_frames": document["exposure_frames"],
            "send_late_max": document["send_lateness_frames"].get("max"),
            "hold_max": document["hold_frames"].get("max"),
            "decisions": document["decisions"],
        }
    exposures = {entry["exposure_frames"] for entry in injected.values()}
    gates["C-04"] = {
        "status": "pass" if exposures == {240} else "fail",
        "injected": injected,
        "detail": "extra latency changes measured lateness, never the window length",
    }

    # -- C-01c: replay determinism --------------------------------------
    def _fake_world():
        fake_time = fakegame.FakeTime()
        return fakegame.FakeGame(fake_time), fake_time.monotonic, fake_time.sleep

    replay = replay_check(_fake_world, [(3, 1, 0, 0)] * 400, frames=180)
    gates["C-01c"] = {
        "status": "pass" if replay["identical"] else "fail",
        "replay": replay,
        "detail": "same action sequence through the same pipeline reproduces the trajectory",
    }

    return {
        "schema": GATE_SCHEMA,
        "version": "v3.7.1",
        "mode": "fake-clock",
        "note": (
            "All gates here run against the deterministic fake clock. They prove the "
            "clock/termination/attribution logic; they do NOT prove live input latency "
            "(that is C-03-live, which needs the game)."
        ),
        "gates": gates,
        "status": {
            name: entry["status"] for name, entry in sorted(gates.items())
        },
        "passed": all(entry["status"] == "pass" for entry in gates.values()),
    }


def write(document: dict, path: str | Path) -> Path:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
