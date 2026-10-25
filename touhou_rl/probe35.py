"""v3.5.0 S0 probe: state/event contract capture + P-point controlled experiment.

Three subcommands:

``collect``
    Run teacher episodes and write ``v350-trace/1`` decision records plus the
    bullet sidecars.  This is the S0 data used by ``verify``.

``p-contract``
    Same trace, but a controlled "move up + focus" override is applied whenever
    a P item is on screen and ``power < 128``.  The run is analysed against the
    pre-registered H-P1..H-P6 hypotheses (see
    ``Document/Update/Alpha/v3.5.0/contract/P点契约实验.md``).

``verify``
    Offline S0-1..S0-5 gate computation over an output directory.  Refuses to
    pass when the trace files do not match the manifest (extra/missing files are
    both failures, the v3.4.0 lesson).

The probe never writes game memory.  It only reads state, runs the existing
scripted pilot, and records what happened.  Teacher episodes are not a policy
result: they exist to exercise the contract, not to claim a milestone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import menu, spec, winapi
from .env import EnvConfig, Th08Env
from .pilot import AvoidPilot, PilotConfig
from .tactics import (
    choose_target,
    collect_power_override,
    describe_target,
    pilot_config_kwargs,
    target_as_boss_array,
    warmup_plan,
)
from .session import GameSession, SessionOptions, prepare_run_dir
from .state import BossInfo, GameState, ItemInfo
from .trace35 import (
    EPISODE_SCHEMA,
    MANIFEST_SCHEMA,
    REQUIRED_GROUPS,
    TRACE_SCHEMA,
    TRACEABILITY_GROUPS,
    Snapshot,
    TraceWriter,
    assess_complete,
    build_record,
    classify_terminal,
    group_status,
    read_episodes,
    read_trace,
    sha256_file,
    stable_split,
    state_snapshot,
    steady_exclusion,
    write_episode,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "Document/Update/Alpha/v3.5.0/probe/s0-teacher"
DEFAULT_RUN = ROOT / "third_party/th08/build/rl/v350-s0-session"
S0_THRESHOLD = 0.95
POWER_TYPES = set(spec.ITEM_POWER_TYPES)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _git_head(root: Path = ROOT) -> str:
    head = Path(root) / ".git" / "HEAD"
    try:
        text = head.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    if text.startswith("ref: "):
        ref = text[5:]
        try:
            return (Path(root) / ".git" / ref).read_text(encoding="utf-8").strip()
        except OSError:
            return text
    return text


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class ProbeConfig:
    out_dir: Path = DEFAULT_OUT
    run_dir: Path = DEFAULT_RUN
    mode: str = "collect"
    policy: str = "teacher"
    quit_after: int = 60
    episodes: int = 10
    max_steps: int = 3000
    frameskip: int = 2
    practice_stage: int | None = 0
    difficulty: int = spec.LUNATIC
    team: int = 0
    post_terminal_seconds: float = 2.0
    dialogue_taps: bool = True
    trace: bool = True
    replay_path: Path | None = None
    replay_stage: int = 0
    # v3.6.0 D2: collect several human replays in one run.  ``replay_path`` stays
    # the single-arm field the P-contract analysis uses; ``replays`` is the batch.
    replays: tuple[Path, ...] = ()
    # v3.6.1 T2: arms cycled per episode so machine drift hits both equally.
    # Only meaningful with ``policy="teacher"``; the replay arm ignores it.
    tactics: tuple[str, ...] = ("none",)
    # T2 pairing: pair k uses policy jitter derived from arm_seed + k in BOTH arms,
    # because the game is deterministic and identical arms would otherwise record
    # identical episodes (measured: byte-identical actions and player positions).
    arm_seed: int | None = None

    def validate(self) -> None:
        if self.mode not in ("collect", "p-contract"):
            raise ValueError(f"unknown mode {self.mode}")
        if self.policy not in ("teacher", "suicide", "quit"):
            raise ValueError(f"unknown policy {self.policy}")
        if self.episodes < 1:
            raise ValueError("episodes must be >= 1")
        if self.max_steps < 1 or self.frameskip < 1:
            raise ValueError("max_steps/frameskip must be >= 1")
        if self.practice_stage is not None and not 0 <= self.practice_stage <= 6:
            raise ValueError("practice_stage must be in 0..6 or None")
        if self.replay_path is not None and not Path(self.replay_path).exists():
            raise ValueError(f"replay not found: {self.replay_path}")
        for path in self.replays:
            if not Path(path).exists():
                raise ValueError(f"replay not found: {path}")
        if self.replays and self.mode == "p-contract" and len(self.replays) > 1:
            # The P-contract analysis compares one controlled arm with one replay
            # arm and matches episodes by index; a batch would silently pool
            # replays from different players.
            raise ValueError("p-contract accepts at most one replay")
        if self.replays and self.policy != "teacher":
            raise ValueError("replay collection does not take a policy")
        for tactic in self.tactics:
            if tactic not in ("none", "clear_enemy", "collect_power"):
                raise ValueError(f"unknown tactic {tactic!r}")
        if self.replays and any(tactic != "none" for tactic in self.tactics):
            # A replay plays the human's input; a tactic would be ignored and the
            # episode would silently be a baseline.  Refuse instead of pretending.
            raise ValueError("replay collection cannot take a tactic")

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "policy": self.policy,
            "quit_after": self.quit_after,
            "episodes": self.episodes,
            "max_steps": self.max_steps,
            "frameskip": self.frameskip,
            "practice_stage": self.practice_stage,
            "difficulty": self.difficulty,
            "team": self.team,
            "post_terminal_seconds": self.post_terminal_seconds,
            "dialogue_taps": self.dialogue_taps,
            "replay_path": None if self.replay_path is None else str(self.replay_path),
            "replay_stage": self.replay_stage,
            "replays": [str(path) for path in self.replays],
            "tactics": list(self.tactics),
            "arm_seed": self.arm_seed,
        }


def episode_plan(cfg: ProbeConfig, start: int) -> list[tuple[int, Path | None, str, int]]:
    """``(episode_index, replay_path_or_None, tactic)`` for every episode this run records.

    Pure so the batch wiring is unit-testable without a game: a replay batch
    yields one episode per replay, everything else falls back to the teacher loop
    (or the single ``replay_path`` arm used by the P contract).

    The tactic list is **cycled**, which is what makes the T2 comparison paired:
    with ``tactics=("none", "clear_enemy")`` episode 1 is baseline, episode 2 is
    the tactic, episode 3 is baseline again, and so on, so any machine or game
    drift between runs hits both arms in equal proportion.
    """
    tactics = cfg.tactics or ("none",)
    if cfg.replays:
        return [(index, Path(path), "none", 0) for index, path in enumerate(cfg.replays, start=start)]
    if cfg.replay_path is not None:
        return [(index, Path(cfg.replay_path), "none", 0) for index in range(start, start + cfg.episodes)]
    return [
        (index, None, tactics[offset % len(tactics)], offset // len(tactics))
        for offset, index in enumerate(range(start, start + cfg.episodes))
    ]


def _env_config(cfg: ProbeConfig) -> EnvConfig:
    return EnvConfig(
        team=cfg.team,
        difficulty=cfg.difficulty,
        practice_stage=cfg.practice_stage,
        frameskip=cfg.frameskip,
        max_steps=cfg.max_steps,
        reset_mode="relaunch",
        run_dir=cfg.run_dir,
        windowed=True,
        # DirectInput needs the game foreground; the probe runs in the interactive
        # session.  PostMessage was measured to leave the game's keyboard state at
        # zero on this build, so it is not an option for live collection.
        use_postmessage_input=False,
        expose_raw_bullets=True,
        expose_bosses=False,
        expose_boss_info=True,
        expose_frame_events=True,
    )


# ---------------------------------------------------------------------------
# snapshot capture
# ---------------------------------------------------------------------------


def _capture_snapshot(env: Th08Env, info: dict, wall_ms: int) -> Snapshot:
    """Read the S0 entity groups with explicit per-group status.

    A group read failure becomes ``status="error"`` and an empty list, never a
    silent zero.  ``build_record``'s completeness check turns that into a named
    incomplete reason; the verifier's S0-2 then sees it as a traceability miss.
    """
    state = env.last_state
    if state is None:  # defensive: env.step always sets it
        state = GameState()
    reader = env.session.reader if env.session is not None else None
    status = {name: "ok" for name in REQUIRED_GROUPS}

    enemies: list | None = None
    items: list | None = None
    bosses: list | None = None
    shot = None

    if reader is not None:
        try:
            enemies = reader.read_enemies()
            status["enemies"] = group_status(True, len(enemies))
        except OSError:
            enemies, status["enemies"] = [], "error"
        try:
            items = reader.read_items()
            status["items"] = group_status(True, len(items))
        except OSError:
            items, status["items"] = [], "error"
        try:
            shot = reader.read_shot_params()
            status["shot"] = "ok" if shot is not None else "error"
        except OSError:
            shot, status["shot"] = None, "error"
    else:
        enemies, items = [], []
        status["enemies"] = status["items"] = status["shot"] = "error"

    boss_info = info.get("boss_info")
    if boss_info is None:
        if reader is not None:
            try:
                bosses = reader.read_boss_info()
                status["boss"] = group_status(True, len(bosses))
            except OSError:
                bosses, status["boss"] = [], "error"
        else:
            bosses, status["boss"] = [], "error"
    else:
        bosses = list(boss_info)
        status["boss"] = group_status(True, len(bosses))

    bullets = info.get("bullets_raw")
    if bullets is None or not isinstance(bullets, dict) or "position" not in bullets:
        bullets = None
        status["bullets"] = "error"
    else:
        status["bullets"] = "ok"

    status["player"] = "ok"
    status["power"] = "ok"
    status["gui"] = "ok" if getattr(state, "gui_read_ok", True) else "error"
    frame_events = info.get("frame_events")
    if frame_events is None and reader is not None:
        try:
            frame_events = reader.read_frame_event_flags()
        except OSError:
            frame_events = None
    return Snapshot(
        state=state,
        wall_ms=int(wall_ms),
        enemies=enemies,
        items=items,
        bosses=bosses,
        shot=shot,
        bullets=bullets,
        frame_event_flags=frame_events,
        status=status,
    )


def _boss_array(bosses: list[BossInfo] | None) -> np.ndarray:
    return np.asarray([[b.x, b.y] for b in (bosses or [])], dtype=np.float32).reshape(-1, 2)


def _bullets_raw(reader) -> dict:
    """Read the full pool and return the active-bullet dict used by the trace."""
    field = reader.read_bullets()
    mask = field.active_mask()
    half = np.maximum(np.asarray(field.half_size[mask], dtype=np.float32).max(axis=1), 0.5)
    return {
        "position": np.array(field.position[mask], dtype=np.float32),
        "velocity": np.array(field.velocity[mask], dtype=np.float32),
        "half": half,
        "slot_id": np.flatnonzero(mask).astype(np.int32),
    }


def _capture_session_snapshot(reader, state: GameState, wall_ms: int) -> Snapshot:
    """Snapshot capture for the replay arm (no Th08Env/info object)."""
    status = {name: "ok" for name in REQUIRED_GROUPS}
    enemies = items = bosses = shot = None
    bullets = None
    try:
        bullets = _bullets_raw(reader)
        status["bullets"] = "ok"
    except OSError:
        status["bullets"] = "error"
    try:
        enemies = reader.read_enemies()
        status["enemies"] = group_status(True, len(enemies))
    except OSError:
        enemies, status["enemies"] = [], "error"
    try:
        items = reader.read_items()
        status["items"] = group_status(True, len(items))
    except OSError:
        items, status["items"] = [], "error"
    try:
        bosses = reader.read_boss_info()
        status["boss"] = group_status(True, len(bosses))
    except OSError:
        bosses, status["boss"] = [], "error"
    try:
        shot = reader.read_shot_params()
        status["shot"] = "ok" if shot is not None else "error"
    except OSError:
        shot, status["shot"] = None, "error"
    frame_events = None
    try:
        frame_events = reader.read_frame_event_flags()
    except OSError:
        frame_events = None
    status["player"] = "ok"
    status["power"] = "ok"
    status["gui"] = "ok" if getattr(state, "gui_read_ok", True) else "error"
    return Snapshot(
        state=state,
        wall_ms=int(wall_ms),
        enemies=enemies,
        items=items,
        bosses=bosses,
        shot=shot,
        bullets=bullets,
        frame_event_flags=frame_events,
        status=status,
    )


# ---------------------------------------------------------------------------
# policy: teacher and P-contract override
# ---------------------------------------------------------------------------


def _p_items(items: list[ItemInfo] | None) -> list[ItemInfo]:
    return [item for item in (items or []) if int(item.item_type) in POWER_TYPES]


def _choose_action(
    cfg: ProbeConfig,
    pilot: AvoidPilot,
    obs: dict,
    info: dict,
    pre_snapshot: Snapshot,
    dialogue_phase: int,
    tactic: str = "none",
    step: int = 0,
    warmup: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict | None, int]:
    """Teacher action plus the tactic / p-contract override; dialogue stays separate.

    ``tactic="clear_enemy"`` only substitutes the *aim target* handed to the
    pilot's existing aim term (its ``bosses`` argument), so the beam search,
    horizon, bomb logic and safety floor are untouched.  ``collect_power`` keeps
    the untouched pilot and may replace the action only when the pilot's own
    chosen action is safe, which is recorded in ``controller`` per decision.
    """
    state = pre_snapshot.state
    if warmup and 1 <= step <= len(warmup):
        # Seeded warm-up: random move, shoot on, no bomb, focus from the draw.
        move, focus = warmup[step - 1]
        controller = {"phase": "warmup", "move": move, "focus": focus}
        if cfg.dialogue_taps and state.gui_dialogue_present:
            action = np.array([move, int(dialogue_phase) & 1, focus, 0], dtype=np.int64)
            return action, controller, dialogue_phase ^ 1
        return np.array([move, 1, focus, 0], dtype=np.int64), controller, dialogue_phase
    bosses = _boss_array(pre_snapshot.bosses)
    controller: dict | None = None
    if tactic == "clear_enemy":
        target = choose_target(list(pre_snapshot.enemies or []), float(state.player_y))
        if target is not None:
            bosses = target_as_boss_array(target)
        controller = describe_target(target)
    teacher = pilot.act(
        obs,
        info.get("bullets_raw"),
        bombs=int(round(float(state.bombs))),
        bosses=bosses,
    )
    action = np.array(teacher, dtype=np.int64, copy=True)
    if cfg.policy == "suicide":
        # Stand still and shoot: a controlled death used only to exercise the
        # game-over event contract (S0-3), never as a policy result.
        action = np.array([0, 1, 0, 0], dtype=np.int64)
        controller = {"policy": "suicide"}
    elif cfg.mode == "p-contract":
        p_items = _p_items(pre_snapshot.items)
        clearance = float(pilot.debug.best_clearance)
        if state.power < spec.POWER_MAX and p_items and state.player_y > 110.0 and clearance >= 2.0:
            # Move up while focused; shoot stays on.  This is the controlled
            # collection route the experiment depends on, not a general policy.
            action = np.array([1, 1, 1, 0], dtype=np.int64)
            controller = {
                "override": "up_collect",
                "p_items": len(p_items),
                "player_y": round(float(state.player_y), 3),
                "clearance": round(clearance, 3),
            }
        elif state.power < spec.POWER_MAX and p_items and state.player_y <= 110.0:
            action = np.array([0, 1, 1, 0], dtype=np.int64)
            controller = {
                "override": "hold_collect",
                "p_items": len(p_items),
                "player_y": round(float(state.player_y), 3),
                "clearance": round(clearance, 3),
            }
    elif tactic == "collect_power":
        decision = collect_power_override(
            power=float(state.power),
            player_y=float(state.player_y),
            items=list(pre_snapshot.items or []),
            clearance=float(pilot.debug.best_clearance),
        )
        controller = {"tactic": "collect_power", "reason": decision["reason"]}
        if decision.get("p_items") is not None:
            controller["p_items"] = int(decision["p_items"])
        if decision.get("clearance") is not None:
            controller["clearance"] = decision["clearance"]
        if decision.get("override") is not None:
            action = np.array(decision["override"], dtype=np.int64)
            controller["override"] = decision["reason"]
    if cfg.dialogue_taps and state.gui_dialogue_present:
        # Gui advances dialogue on a shoot edge; holding Z forever stalls the run
        # after a midboss/boss phase (Gui.cpp:599).  Toggle only the shoot bit.
        action = np.array(action, dtype=np.int64, copy=True)
        action[1] = int(dialogue_phase) & 1
        controller = dict(controller or {})
        controller["dialogue_tap"] = True
        dialogue_phase ^= 1
    elif cfg.dialogue_taps:
        dialogue_phase = 0
    return action, controller, dialogue_phase


# ---------------------------------------------------------------------------
# post-terminal polling
# ---------------------------------------------------------------------------


def _recover_focus(env: Th08Env, timeout: float = 2.0) -> tuple[bool, dict | None]:
    """Try to resume a game whose frame clock stopped because focus was lost.

    TH08 pauses its simulation when the window is inactive.  A background probe
    can lose the foreground to the DSH/browser at any time; treating the first
    frozen frame as a game terminal destroys a 10-minute episode and mislabels
    the data.  Refocus once and wait for the clock; if it moves, re-capture the
    observation and continue.  Returns ``(recovered, observation)``.
    """
    if env.session is None or env.session.reader is None:
        return False, None
    state = env.last_state
    before = int(state.gameplay_frame) if state is not None else 0
    hwnd = env.session.refresh_window()
    if hwnd:
        winapi.force_foreground(hwnd)
    deadline = time.perf_counter() + max(0.0, timeout)
    while time.perf_counter() < deadline:
        try:
            current = env.session.reader.read_state(with_bullets=False, with_title=False)
        except OSError:
            time.sleep(0.05)
            continue
        if current.supervisor_state != spec.STATE_GAME_MANAGER or current.is_demo:
            return False, None
        if int(current.gameplay_frame) != before:
            observation, full = env._capture_observation()
            env._last_observation = observation
            env.last_state = full
            env._last_frame = full.gameplay_frame
            return True, observation
        hwnd = env.session.refresh_window()
        if hwnd:
            winapi.force_foreground(hwnd)
        time.sleep(0.05)
    return False, None


def _probe_post_terminal(env: Th08Env, seconds: float) -> list[dict]:
    reader = env.session.reader if env.session is not None else None
    if reader is None:
        return []
    samples: list[dict] = []
    deadline = time.perf_counter() + max(0.0, seconds)
    while len(samples) < 30 and time.perf_counter() < deadline:
        try:
            state = reader.read_state(with_bullets=False, with_title=False)
        except OSError:
            break
        samples.append(
            {
                "wall": round(time.perf_counter(), 3),
                "supervisor_state": int(state.supervisor_state),
                "stage": int(state.stage),
                "stage_frames": int(state.stage_frames),
                "gameplay_frame": int(state.gameplay_frame),
                "lives": float(state.lives),
                "deaths": int(state.deaths),
                "player_state": int(state.player_state),
                "gui_dialogue_present": bool(state.gui_dialogue_present),
                "is_demo": bool(state.is_demo),
                "is_replay": bool(state.is_replay),
                "in_game_menu": bool(state.in_game_menu),
                "title_screen_current": int(state.title_screen_current),
                "title_screen_state": int(state.title_screen_state),
                "flags": int(state.flags) & 0xFFFFFFFF,
                "game_cleared": bool(state.game_cleared),
                "stage_clear_sequence_active": bool(state.stage_clear_sequence_active),
                "stage_transition_state": int(state.stage_transition_state),
                "gui_stage_clear_screen_state": int(state.gui_stage_clear_screen_state),
            }
        )
        time.sleep(0.2)
    return samples


# ---------------------------------------------------------------------------
# episode
# ---------------------------------------------------------------------------


def run_episode(cfg: ProbeConfig, index: int, tactic: str = "none", pair: int = 0) -> dict:
    prefix = cfg.mode if cfg.policy == "teacher" else f"{cfg.mode}-{cfg.policy}"
    episode = f"{prefix}-ep{index:02d}"
    env = Th08Env(_env_config(cfg))
    # The ClearEnemy arm is defined *only* by these two pilot fields plus the aim
    # target substitution; every other arm keeps the baseline configuration.
    pilot = AvoidPilot(PilotConfig(frameskip=cfg.frameskip, **pilot_config_kwargs(tactic)))
    warmup = warmup_plan(cfg.arm_seed, pair)
    writer = TraceWriter(cfg.out_dir, episode) if cfg.trace else None
    started = time.perf_counter()
    ep: dict = {
        "schema": EPISODE_SCHEMA,
        "episode": episode,
        "mode": cfg.mode,
        "tactic": tactic,
        "warmup_decisions": len(warmup),
        "config": cfg.as_dict(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "start": {},
        "end": {},
        "termination": {},
        "stats": {},
        "split": stable_split(episode),
    }
    records = 0
    steady = 0
    complete = 0
    focus_recoveries = 0
    events: Counter[str] = Counter()
    max_power = 0.0
    max_enemies = 0
    max_items = 0
    try:
        obs, info = env.reset()
        start_state = env.last_state or GameState()
        ep["start"] = {
            "frame": int(start_state.gameplay_frame),
            "stage": int(start_state.stage),
            "power": float(start_state.power),
            "lives": float(start_state.lives),
            "bombs": float(start_state.bombs),
        }
        initial = _capture_snapshot(env, info, _now_ms())
        prev_snapshot = initial
        prev_scalar = state_snapshot(initial.state)
        prev_record: dict | None = None
        dialogue_phase = 0

        for step in range(1, cfg.max_steps + 1):
            pre_state = env.last_state or GameState()
            pre_frame = int(pre_state.gameplay_frame)
            if cfg.policy == "quit" and step == cfg.quit_after:
                # Controlled left-game evidence: quit through the pause menu,
                # then let one step observe supervisor != GAME_MANAGER.
                menu.exit_run_to_title(env.session)
                action = np.array([0, 0, 0, 0], dtype=np.int64)
                controller = {"policy": "quit"}
            else:
                action, controller, dialogue_phase = _choose_action(
                    cfg, pilot, obs, info, prev_snapshot, dialogue_phase, tactic,
                    step=step, warmup=warmup,
                )
            obs, _reward, terminated, truncated, info = env.step(action)
            post = _capture_snapshot(env, info, _now_ms())
            teacher = {
                "clearance": round(float(pilot.debug.best_clearance), 3),
                "bullets_used": int(pilot.debug.bullets_used),
            }
            record = build_record(
                episode=episode,
                decision=step,
                frameskip=cfg.frameskip,
                frame_pre=pre_frame,
                wall_ms=post.wall_ms,
                pre_state=prev_scalar,
                snapshot=post,
                action=action,
                teacher=teacher,
                controller=controller,
            )
            ok, reasons = assess_complete(record, prev_record)
            record["complete"] = bool(ok)
            record["incomplete_reasons"] = reasons
            records += 1
            if steady_exclusion(record) is None:
                steady += 1
                if ok:
                    complete += 1
            for event in record["events"]:
                events[event] += 1
            max_power = max(max_power, float(record.get("resources", {}).get("power") or 0.0))
            max_enemies = max(max_enemies, len(record.get("enemies", [])))
            max_items = max(max_items, len(record.get("items", [])))
            if writer is not None:
                writer.write(record, post)
            prev_record = record
            prev_scalar = state_snapshot(post.state)
            prev_snapshot = post
            if terminated and "frame stalled" in str(info.get("termination_reason", "")):
                recovered, fresh_obs = _recover_focus(env)
                if recovered:
                    focus_recoveries += 1
                    obs = fresh_obs if fresh_obs is not None else obs
                    terminated = False
                    truncated = False
                    info["termination_reason"] = "focus_recovered"
                    prev_scalar = state_snapshot(env.last_state or post.state)
            if terminated or truncated:
                break

        final_state = env.last_state or GameState()
        post = _probe_post_terminal(env, cfg.post_terminal_seconds)
        terminal = classify_terminal(info.get("termination_reason", ""), final_state, post)
        ep["termination"] = {
            "reason": terminal["classified"],
            "env_reason": terminal["reason"],
            "evidence": terminal,
            "post_terminal": post[-10:],
        }
        ep["end"] = {
            "frame": int(final_state.gameplay_frame),
            "run_frames": int(final_state.run_frames),
            "stage": int(final_state.stage),
            "power": float(final_state.power),
            "lives": float(final_state.lives),
            "bombs": float(final_state.bombs),
            "deaths": int(final_state.deaths),
            "score": int(final_state.score),
        }
        ep["terminated"] = bool(terminated)
        ep["truncated"] = bool(truncated)
    except Exception as exc:  # keep the batch alive; the report records the failure
        ep["error"] = f"{type(exc).__name__}: {exc}"
        ep["termination"] = {"reason": "probe_error", "env_reason": str(exc), "evidence": {}}
    finally:
        if writer is not None:
            writer.close()
        try:
            env.close()
        except Exception:
            pass
        ep["stats"] = {
            "decisions": records,
            "steady_decisions": steady,
            "complete_decisions": complete,
            "complete_rate": round(complete / steady, 4) if steady else None,
            "events": dict(events),
            "max_power": round(max_power, 3),
            "max_enemies": max_enemies,
            "max_items": max_items,
            "focus_recoveries": focus_recoveries,
            "wall_seconds": round(time.perf_counter() - started, 1),
        }
        write_episode(cfg.out_dir, ep)
    return ep


# ---------------------------------------------------------------------------
# replay arm (human trace, no policy)
# ---------------------------------------------------------------------------


def _wait_replay_frames(reader, start_frame: int, frames: int, timeout: float = 3.0) -> GameState:
    deadline = time.perf_counter() + timeout
    state = reader.read_state(with_bullets=False, with_title=False)
    while time.perf_counter() < deadline:
        state = reader.read_state(with_bullets=False, with_title=False)
        if state.supervisor_state != spec.STATE_GAME_MANAGER:
            return state
        if ((state.gameplay_frame - start_frame) & 0xFFFFFFFF) >= frames:
            return state
        time.sleep(0.001)
    return state


def _recover_focus_session(session: GameSession, before: int, timeout: float = 2.0) -> GameState | None:
    reader = session.reader
    if reader is None:
        return None
    hwnd = session.refresh_window()
    if hwnd:
        winapi.force_foreground(hwnd)
    deadline = time.perf_counter() + max(0.0, timeout)
    while time.perf_counter() < deadline:
        try:
            state = reader.read_state(with_bullets=False, with_title=False)
        except OSError:
            time.sleep(0.05)
            continue
        if state.supervisor_state != spec.STATE_GAME_MANAGER:
            return None
        if int(state.gameplay_frame) != before:
            return state
        hwnd = session.refresh_window()
        if hwnd:
            winapi.force_foreground(hwnd)
        time.sleep(0.05)
    return None


def _probe_post_terminal_session(session: GameSession, seconds: float) -> list[dict]:
    reader = session.reader
    if reader is None:
        return []
    samples: list[dict] = []
    deadline = time.perf_counter() + max(0.0, seconds)
    while len(samples) < 30 and time.perf_counter() < deadline:
        try:
            state = reader.read_state(with_bullets=False, with_title=False)
        except OSError:
            break
        samples.append(
            {
                "wall": round(time.perf_counter(), 3),
                "supervisor_state": int(state.supervisor_state),
                "stage": int(state.stage),
                "gameplay_frame": int(state.gameplay_frame),
                "lives": float(state.lives),
                "deaths": int(state.deaths),
                "player_state": int(state.player_state),
                "gui_dialogue_present": bool(state.gui_dialogue_present),
                "flags": int(state.flags) & 0xFFFFFFFF,
                "game_cleared": bool(state.game_cleared),
                "stage_clear_sequence_active": bool(state.stage_clear_sequence_active),
                "gui_stage_clear_screen_state": int(state.gui_stage_clear_screen_state),
            }
        )
        time.sleep(0.2)
    return samples


def run_replay_episode(cfg: ProbeConfig, index: int, replay_path: Path | None = None) -> dict:
    """Play one human replay and record the same v350 trace, with no policy."""
    source = Path(replay_path) if replay_path is not None else Path(cfg.replay_path)
    episode = f"replay-ep{index:02d}"
    prepare_run_dir(
        cfg.run_dir,
        ROOT / "third_party/th08/build/play",
        windowed=True,
        default_difficulty=cfg.difficulty,
        use_postmessage_input=False,
    )
    menu.install_replay(cfg.run_dir, source, index=1)
    session = GameSession(SessionOptions(run_dir=cfg.run_dir))
    writer = TraceWriter(cfg.out_dir, episode) if cfg.trace else None
    started = time.perf_counter()
    ep: dict = {
        "schema": EPISODE_SCHEMA,
        "episode": episode,
        "mode": "replay",
        "replay_file": source.name,
        "replay_sha256": sha256_file(source),
        "config": cfg.as_dict(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "start": {}, "end": {}, "termination": {}, "stats": {},
        "split": stable_split(episode),
    }
    records = steady = complete = 0
    focus_recoveries = 0
    events: Counter[str] = Counter()
    max_power = 0.0
    max_enemies = max_items = 0
    final_state: GameState | None = None
    try:
        session.start()
        menu.enter_replay(session, replay_index=0, stage=cfg.replay_stage)
        reader = session.reader
        if reader is None:
            raise RuntimeError("reader missing after replay start")
        state = reader.read_state(with_bullets=False, with_title=False)
        ep["start"] = {
            "frame": int(state.gameplay_frame),
            "stage": int(state.stage),
            "power": float(state.power),
            "lives": float(state.lives),
            "bombs": float(state.bombs),
            "is_replay": bool(state.is_replay),
        }
        if not state.is_replay:
            raise RuntimeError("enter_replay succeeded but is_replay is false")
        initial = _capture_session_snapshot(reader, state, _now_ms())
        prev_snapshot = initial
        prev_scalar = state_snapshot(state)
        prev_record: dict | None = None
        prev_frame = int(state.gameplay_frame)
        reason = ""
        for step in range(1, cfg.max_steps + 1):
            state = _wait_replay_frames(reader, prev_frame, cfg.frameskip)
            if ((int(state.gameplay_frame) - prev_frame) & 0xFFFFFFFF) == 0 and state.supervisor_state == spec.STATE_GAME_MANAGER:
                resumed = _recover_focus_session(session, prev_frame)
                if resumed is not None:
                    state = resumed
                    focus_recoveries += 1
            post = _capture_session_snapshot(reader, state, _now_ms())
            input_mask = int(state.gui_input_current)
            action = spec.buttons_to_action(input_mask)
            record = build_record(
                episode=episode,
                decision=step,
                frameskip=cfg.frameskip,
                frame_pre=prev_frame,
                wall_ms=post.wall_ms,
                pre_state=prev_scalar,
                snapshot=post,
                action=action,
                teacher=None,
                controller={"replay": True, "input_mask": input_mask},
            )
            ok, reasons = assess_complete(record, prev_record)
            record["complete"] = bool(ok)
            record["incomplete_reasons"] = reasons
            records += 1
            if steady_exclusion(record) is None:
                steady += 1
                complete += int(ok)
            for event in record["events"]:
                events[event] += 1
            max_power = max(max_power, float(record["resources"].get("power") or 0.0))
            max_enemies = max(max_enemies, len(record.get("enemies", [])))
            max_items = max(max_items, len(record.get("items", [])))
            if writer is not None:
                writer.write(record, post)
            prev_record = record
            prev_snapshot = post
            prev_scalar = state_snapshot(state)
            prev_frame = int(state.gameplay_frame)
            final_state = state
            if state.supervisor_state != spec.STATE_GAME_MANAGER:
                reason = f"left game (supervisor_state={state.supervisor_state})"
                break
            if state.lives < 0:
                reason = "game over"
                break
            if state.game_cleared:
                reason = "game cleared"
                break
            if state.stage_clear_sequence_active or state.gui_stage_clear_screen_state != 0:
                reason = "stage clear"
                break
        if final_state is None:
            final_state = state
        post_samples = _probe_post_terminal_session(session, cfg.post_terminal_seconds)
        terminal = classify_terminal(reason or ("max_steps" if records >= cfg.max_steps else "frame stalled"), final_state, post_samples)
        ep["termination"] = {"reason": terminal["classified"], "env_reason": terminal["reason"], "evidence": terminal}
        ep["end"] = {
            "frame": int(final_state.gameplay_frame),
            "run_frames": int(final_state.run_frames),
            "stage": int(final_state.stage),
            "power": float(final_state.power),
            "lives": float(final_state.lives),
            "bombs": float(final_state.bombs),
            "deaths": int(final_state.deaths),
            "score": int(final_state.score),
        }
        ep["terminated"] = reason not in ("", "max_steps")
        ep["truncated"] = reason in ("", "max_steps")
    except Exception as exc:
        ep["error"] = f"{type(exc).__name__}: {exc}"
        ep["termination"] = {"reason": "probe_error", "env_reason": str(exc), "evidence": {}}
    finally:
        if writer is not None:
            writer.close()
        try:
            session.close()
        except Exception:
            pass
        ep["stats"] = {
            "decisions": records,
            "steady_decisions": steady,
            "complete_decisions": complete,
            "complete_rate": round(complete / steady, 4) if steady else None,
            "events": dict(events),
            "max_power": round(max_power, 3),
            "max_enemies": max_enemies,
            "max_items": max_items,
            "focus_recoveries": focus_recoveries,
            "wall_seconds": round(time.perf_counter() - started, 1),
        }
        write_episode(cfg.out_dir, ep)
    return ep


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def write_manifest(out_dir: Path, cfg: ProbeConfig, episodes: list[dict]) -> dict:
    out_dir = Path(out_dir)
    files = []
    for path in sorted((out_dir / "trace").glob("*.jsonl")):
        lines = sum(1 for _ in path.open("r", encoding="utf-8"))
        files.append({"file": str(path.relative_to(out_dir)), "sha256": sha256_file(path),
                      "size": path.stat().st_size, "lines": lines})
    for path in sorted((out_dir / "bullets").glob("*.npz")):
        files.append({"file": str(path.relative_to(out_dir)), "sha256": sha256_file(path),
                      "size": path.stat().st_size, "lines": None})
    episodes_path = out_dir / "episodes.jsonl"
    if episodes_path.exists():
        files.append({"file": "episodes.jsonl", "sha256": sha256_file(episodes_path),
                      "size": episodes_path.stat().st_size,
                      "lines": sum(1 for _ in episodes_path.open("r", encoding="utf-8"))})
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _git_head(),
        "config": cfg.as_dict(),
        "episodes": [ep.get("episode") for ep in episodes],
        "files": files,
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# offline gates
# ---------------------------------------------------------------------------


def _load_records(out_dir: Path, episodes: list[dict]) -> dict[str, list[dict]]:
    traces = {}
    for ep in episodes:
        name = ep.get("episode")
        path = Path(out_dir) / "trace" / f"{name}.jsonl"
        traces[name] = read_trace(path) if path.exists() else []
    return traces


def verify(out_dir: Path, manifest_path: Path | None = None) -> dict:
    out_dir = Path(out_dir)
    episodes = read_episodes(out_dir)
    traces = _load_records(out_dir, episodes)
    report: dict[str, Any] = {
        "schema": "v350-s0-report/1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _git_head(),
        "out_dir": str(out_dir),
        "episodes": len(episodes),
        "gates": {},
        "diagnostics": {},
    }

    manifest_check: dict[str, Any] = {"checked": False}
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        listed = {item["file"]: item for item in manifest.get("files", [])}
        actual = set()
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.name not in ("manifest.json", "s0-report.json", "p-contract.json"):
                actual.add(str(path.relative_to(out_dir)))
        missing = sorted(set(listed) - actual)
        extra = sorted(actual - set(listed))
        hash_mismatch = []
        for name, item in listed.items():
            path = out_dir / name
            if path.exists() and sha256_file(path) != item["sha256"]:
                hash_mismatch.append(name)
        manifest_check = {
            "checked": True,
            "file_count": len(listed),
            "missing": missing,
            "extra": extra,
            "hash_mismatch": hash_mismatch,
            "pass": not (missing or extra or hash_mismatch),
        }
    report["manifest"] = manifest_check

    total_records = 0
    monotonic = 0
    steady_total = 0
    traceability = 0
    complete = 0
    episode_gate_rows = []
    split_errors = []
    missing_traces: list[str] = []
    record_mismatch: list[str] = []
    names = [ep.get("episode") for ep in episodes]
    duplicate_episodes = sorted(name for name, count in Counter(names).items() if count > 1 and name)
    for ep in episodes:
        name = ep.get("episode")
        records = traces.get(name, [])
        if not records:
            episode_gate_rows.append({"episode": name, "error": "missing_trace"})
            missing_traces.append(name)
            continue
        prev = None
        ep_steady = ep_traceable = ep_complete = 0
        for record in records:
            total_records += 1
            ok_monotonic = True
            if record.get("frame_pre") is None or record.get("frame_post") is None:
                ok_monotonic = False
            elif int(record["frame_post"]) < int(record["frame_pre"]):
                ok_monotonic = False
            if prev is not None:
                if int(record.get("decision", -1)) != int(prev.get("decision", -2)) + 1:
                    ok_monotonic = False
                if int(record.get("wall_ms", 0)) < int(prev.get("wall_ms", 0)):
                    ok_monotonic = False
            monotonic += int(ok_monotonic)
            excluded = steady_exclusion(record)
            if excluded is None:
                steady_total += 1
                ep_steady += 1
                status = record.get("schema_status", {})
                traceable = all(status.get(group, "error") != "error" for group in TRACEABILITY_GROUPS)
                traceability += int(traceable)
                ep_traceable += int(traceable)
                if bool(record.get("complete")):
                    complete += 1
                    ep_complete += 1
            prev = record
        expected = (ep.get("stats") or {}).get("decisions")
        episode_gate_rows.append(
            {
                "episode": name,
                "records": len(records),
                "expected_records": expected,
                "steady": ep_steady,
                "traceable": ep_traceable,
                "complete": ep_complete,
                "complete_rate": round(ep_complete / ep_steady, 4) if ep_steady else None,
                "termination": (ep.get("termination") or {}).get("reason"),
                "split": ep.get("split"),
                "split_consistent": ep.get("split") == stable_split(name),
                "records_match": expected is None or int(expected) == len(records),
            }
        )
        if ep.get("split") != stable_split(name):
            split_errors.append(name)
        if expected is not None and int(expected) != len(records):
            record_mismatch.append(name)

    # S0-3: only episodes with a real game terminal are testable.  A max_steps
    # truncation says nothing about distinguishing clear/game-over/left-game.
    known = {"stage_clear", "game_clear", "game_over", "left_game"}
    terminal_episodes = [
        ep for ep in episodes if (ep.get("termination") or {}).get("reason") in known | {"max_steps", "frame_stalled", "probe_error"}
    ]
    game_terminal = [ep for ep in terminal_episodes if (ep.get("termination") or {}).get("reason") in known]
    terminal_pass = 0
    ambiguous = []
    for ep in game_terminal:
        term = ep.get("termination") or {}
        evidence = term.get("evidence") or {}
        flags = [bool(evidence.get("game_clear")), bool(evidence.get("stage_advanced") or evidence.get("clear_sequence") or evidence.get("clear_screen")), bool(evidence.get("game_over"))]
        if sum(flags) <= 1:
            terminal_pass += 1
        else:
            ambiguous.append(ep.get("episode"))
    s0_3_value = (terminal_pass / len(game_terminal)) if game_terminal else None
    s0_3_pass = s0_3_value is not None and s0_3_value >= 1.0

    gates = {
        "S0-1": {
            "name": "逐决策时间戳单调",
            "value": round(monotonic / total_records, 6) if total_records else None,
            "threshold": 1.0,
            "numerator": monotonic,
            "denominator": total_records,
            "pass": total_records > 0 and monotonic == total_records,
        },
        "S0-2": {
            "name": "enemy/item/power 字段可追溯",
            "value": round(traceability / steady_total, 6) if steady_total else None,
            "threshold": 1.0,
            "numerator": traceability,
            "denominator": steady_total,
            "pass": steady_total > 0 and traceability == steady_total,
        },
        "S0-3": {
            "name": "clear/game-over/left-game 可区分",
            "value": round(s0_3_value, 6) if s0_3_value is not None else None,
            "threshold": 1.0,
            "numerator": terminal_pass,
            "denominator": len(game_terminal),
            "ambiguous": ambiguous,
            "pass": s0_3_pass,
        },
        "S0-4": {
            "name": "新采集 complete 有效率",
            "value": round(complete / steady_total, 6) if steady_total else None,
            "threshold": S0_THRESHOLD,
            "numerator": complete,
            "denominator": steady_total,
            "pass": steady_total > 0 and (complete / steady_total) >= S0_THRESHOLD,
        },
        "S0-5": {
            "name": "整局切分",
            "value": 1.0 - (len(split_errors) / len(episodes)) if episodes else None,
            "threshold": 1.0,
            "errors": split_errors,
            "duplicate_episodes": duplicate_episodes,
            "pass": bool(episodes) and not split_errors and not duplicate_episodes,
        },
    }
    if manifest_check.get("checked"):
        gates["S0-M"] = {
            "name": "manifest 与磁盘一致",
            "value": 1.0 if manifest_check["pass"] else 0.0,
            "threshold": 1.0,
            "pass": bool(manifest_check["pass"]),
        }
    failed = [name for name, gate in gates.items() if not gate.get("pass")]
    integrity_ok = not (missing_traces or record_mismatch or duplicate_episodes)
    report["gates"] = gates
    report["episode_rows"] = episode_gate_rows
    report["diagnostics"] = {
        "records": total_records,
        "steady": steady_total,
        "missing_traces": missing_traces,
        "record_mismatch": record_mismatch,
        "duplicate_episodes": duplicate_episodes,
        "schema_error_counts": _schema_error_counts(traces),
        "event_counts": _event_counts(traces),
        "termination_counts": dict(Counter((ep.get("termination") or {}).get("reason") for ep in episodes)),
        "complete_rate_by_episode": {row["episode"]: row.get("complete_rate") for row in episode_gate_rows},
        "episodes_with_errors": [ep.get("episode") for ep in episodes if ep.get("error")],
    }
    report["overall"] = {"pass": (not failed) and integrity_ok,
                         "failed": failed,
                         "integrity_ok": integrity_ok}
    _write_json(out_dir / "s0-report.json", report)
    return report


def _schema_error_counts(traces: dict[str, list[dict]]) -> dict:
    counts: Counter[str] = Counter()
    for records in traces.values():
        for record in records:
            for group, status in (record.get("schema_status") or {}).items():
                if status == "error":
                    counts[group] += 1
    return dict(counts)


def _event_counts(traces: dict[str, list[dict]]) -> dict:
    counts: Counter[str] = Counter()
    for records in traces.values():
        for record in records:
            for event in record.get("events") or []:
                counts[event] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# P contract analysis
# ---------------------------------------------------------------------------


def _p_items_in(record: dict) -> dict[int, dict]:
    return {
        int(item["slot"]): item
        for item in record.get("items", [])
        if int(item.get("type", -1)) in POWER_TYPES
    }


def analyse_p_contract(out_dir: Path, episodes: list[dict]) -> dict:
    """Compute H-P1..H-P6 from the recorded p-contract traces (pure offline)."""
    # ``--append`` can leave duplicate episode rows if an arm is re-run; the
    # trace file is authoritative, so analyse each episode name once.
    unique_episodes = []
    seen_episodes = set()
    for ep in episodes:
        name = ep.get("episode")
        if name in seen_episodes:
            continue
        seen_episodes.add(name)
        unique_episodes.append(ep)
    episodes = unique_episodes
    traces = _load_records(Path(out_dir), episodes)
    modes = {ep.get("episode"): ep.get("mode", "") for ep in episodes}
    per_episode = []
    all_max_power = 0.0
    over_max = 0
    focus_runs: list[int] = []
    focus_release_ok = False
    transitions: list[dict] = []
    crossings: list[dict] = []
    power_windows: list[dict] = []

    for ep in episodes:
        name = ep.get("episode")
        records = traces.get(name, [])
        if not records:
            per_episode.append({"episode": name, "error": "missing_trace"})
            continue
        line = None
        ep_max = 0.0
        ep_transitions = 0
        ep_crossings = 0
        current_run = 0
        for index, record in enumerate(records):
            shot = record.get("shot") or {}
            if line is None and shot.get("point_item_value_line") is not None:
                line = float(shot["point_item_value_line"])
            power = record.get("resources", {}).get("power")
            if power is not None:
                ep_max = max(ep_max, float(power))
                all_max_power = max(all_max_power, float(power))
                if float(power) > spec.POWER_MAX + 1e-6:
                    over_max += 1
            player = record.get("player") or {}
            action = record.get("action") or [0, 0, 0, 0]
            focused = int(player.get("focus_mode", 0)) == 1 and int(action[2]) == 1
            if focused:
                current_run += 1
            else:
                if current_run:
                    focus_runs.append(current_run)
                    if current_run >= 10:
                        tail = [
                            int((candidate.get("player") or {}).get("focus_mode", 0))
                            for candidate in records[index + 1: index + 3]
                        ]
                        if tail and all(value == 0 for value in tail):
                            focus_release_ok = True
                current_run = 0
            if index == 0:
                continue
            prev = records[index - 1]
            prev_y = (prev.get("player") or {}).get("y")
            cur_y = player.get("y")
            if line is not None and prev_y is not None and cur_y is not None:
                if float(prev_y) >= line > float(cur_y):
                    crossings.append(
                        {
                            "episode": name,
                            "decision": record.get("decision"),
                            "line": line,
                            "player_y": float(cur_y),
                            "prev_y": float(prev_y),
                            "focus_mode": int(player.get("focus_mode", 0)),
                            "player_state": int(player.get("state", -1)),
                            "power": power,
                            "triggered": int(player.get("focus_mode", 0)) != 0
                            or (power is not None and float(power) >= spec.POWER_MAX),
                        }
                    )
                    ep_crossings += 1
            prev_items = _p_items_in(prev)
            cur_items = _p_items_in(record)
            for slot, prev_item in prev_items.items():
                cur_item = cur_items.get(slot)
                if cur_item is None:
                    continue
                if int(prev_item.get("state", 0)) != int(spec.ITEM_STATE_AUTOCOLLECT) and int(cur_item.get("state", 0)) == int(spec.ITEM_STATE_AUTOCOLLECT):
                    ep_transitions += 1
                    transitions.append(
                        {
                            "episode": name,
                            "mode": modes.get(name, ""),
                            "decision": record.get("decision"),
                            "slot": slot,
                            "player_y": float(cur_y) if cur_y is not None else None,
                            "prev_player_y": float(prev_y) if prev_y is not None else None,
                            "line": line,
                            "focus_mode": int(player.get("focus_mode", 0)),
                            "power": power,
                            "state": int(cur_item.get("state", 0)),
                        }
                    )
            prev_power = (prev.get("resources") or {}).get("power")
            cur_power = (record.get("resources") or {}).get("power")
            disappeared = {slot: item for slot, item in prev_items.items() if slot not in cur_items}
            if disappeared:
                power_windows.append(
                    {
                        "episode": name,
                        "decision": record.get("decision"),
                        "items": len(disappeared),
                        "types": sorted({int(item.get("type", -1)) for item in disappeared.values()}),
                        "power_delta": None
                        if prev_power is None or cur_power is None
                        else float(cur_power) - float(prev_power),
                    }
                )
        if current_run:
            focus_runs.append(current_run)
        per_episode.append(
            {
                "episode": name,
                "records": len(records),
                "line": line,
                "max_power": round(ep_max, 3),
                "transitions": ep_transitions,
                "crossings": ep_crossings,
            }
        )

    # H-P1: max power observable and never above the cap.
    hp1 = {
        "max_power": round(all_max_power, 3),
        "over_max": over_max,
        "pass": all_max_power >= spec.POWER_MAX and over_max == 0,
        "not_tested": all_max_power < spec.POWER_MAX,
    }
    # H-P2: once at max, no positive power delta.
    hp2_violations = 0
    hp2_checked = 0
    for ep in episodes:
        records = traces.get(ep.get("episode"), [])
        at_max = False
        for record in records:
            power = (record.get("resources") or {}).get("power")
            if power is None:
                continue
            if power >= spec.POWER_MAX:
                at_max = True
            elif at_max:
                # after leaving max (a death?) a positive delta is only a violation
                # inside the same max window; keep a simple, conservative rule.
                pass
        powers = [(r.get("resources") or {}).get("power") for r in records]
        for prev, cur in zip(powers, powers[1:]):
            if prev is None or cur is None:
                continue
            if prev >= spec.POWER_MAX:
                hp2_checked += 1
                if cur > prev + 1e-6:
                    hp2_violations += 1
    hp2 = {"checked_windows": hp2_checked, "violations": hp2_violations,
           "pass": hp2_checked > 0 and hp2_violations == 0,
           "not_tested": hp2_checked == 0}

    # H-P3: the controlled arm holds focus before the line, so its first
    # auto-collect transition must sit on the shot-file line.  Replay arms are
    # not used for H-P3: a human may cross the line unfocused and only press
    # focus later, which is a legitimate but different observation.
    #
    # Sampling resolution: at frameskip=2 and focused speed 2 px/frame the
    # player moves 4 px per decision, so the first sampled y below the line can
    # be up to 4 px below it.  The line must lie inside the bracketing interval
    # [cur_y, prev_y], and the observed y must be within one step of the line.
    controlled_transitions = [t for t in transitions if t.get("mode") == "p-contract"]
    first_transition = controlled_transitions[0] if controlled_transitions else None
    hp3_diff = None
    hp3_interval = None
    if first_transition and first_transition.get("player_y") is not None and first_transition.get("line") is not None:
        line_value = float(first_transition["line"])
        cur_y = float(first_transition["player_y"])
        prev_y = first_transition.get("prev_player_y")
        hp3_diff = round(abs(cur_y - line_value), 3)
        if prev_y is not None:
            hp3_interval = bool(float(prev_y) >= line_value - 0.5 and cur_y <= line_value + 0.5)
    step_px = 4.0  # frameskip * focused axis speed; the control arm always focuses
    hp3 = {
        "first_transition": first_transition,
        "abs_diff": hp3_diff,
        "line": first_transition.get("line") if first_transition else None,
        "within_one_step": hp3_diff is not None and hp3_diff <= step_px + 0.5,
        "line_in_interval": hp3_interval,
        "pass": (
            hp3_diff is not None
            and hp3_diff <= step_px + 0.5
            and hp3_interval is not False
            and 120.0 <= float(first_transition["line"]) <= 136.0
        ),
        "not_tested": first_transition is None,
    }

    # H-P4: focus held for >=10 consecutive decisions and released within 2.
    hp4 = {
        "max_focus_run": max(focus_runs) if focus_runs else 0,
        "release_ok": focus_release_ok,
        "pass": bool(focus_runs) and max(focus_runs) >= 10 and focus_release_ok,
        "not_tested": not focus_runs,
    }

    # H-P5: at each crossing with the auto-collect condition satisfied (focus
    # held or power full), every in-use P enters auto-collect within 2 decisions.
    # Crossings while the player is dying/spawning are excluded: the source
    # explicitly skips auto-collect for PLAYER_STATE_DYING/SPAWNING
    # (ItemManager.cpp:273), so a missed transition there is not a contract miss.
    hp5_rows = []
    hp5_excluded = 0
    for crossing in crossings:
        if not crossing.get("triggered"):
            continue
        if int(crossing.get("player_state", 0)) not in (0, 3):
            hp5_excluded += 1
            continue
        name = crossing["episode"]
        records = traces.get(name, [])
        index = next((i for i, r in enumerate(records) if r.get("decision") == crossing["decision"]), None)
        if index is None:
            continue
        prev_items = _p_items_in(records[index - 1]) if index > 0 else {}
        missed = []
        for slot in prev_items:
            found = False
            for offset in (0, 1):
                candidate_index = index + offset
                if candidate_index >= len(records):
                    continue
                candidate = _p_items_in(records[candidate_index]).get(slot)
                if candidate is None:
                    found = True  # disappeared -> collected or offscreen, checked by P6
                    break
                if int(candidate.get("state", 0)) == int(spec.ITEM_STATE_AUTOCOLLECT):
                    found = True
                    break
            if not found:
                missed.append(slot)
        hp5_rows.append({"episode": name, "decision": crossing["decision"],
                         "p_items": len(prev_items), "missed": missed})
    missed_total = sum(len(row["missed"]) for row in hp5_rows)
    hp5 = {
        "crossings": len(hp5_rows),
        "excluded_player_dying": hp5_excluded,
        "missed_total": missed_total,
        "rows": hp5_rows,
        "pass": bool(hp5_rows) and missed_total <= 1,
        "not_tested": not hp5_rows,
    }

    # H-P6: power changes must be explained by disappearing P items, or by the
    # game's own per-frame "item collected" flag.  The flag matters because a P
    # item can spawn and be auto-collected inside one decision's blind window
    # (frameskip=2); requiring a visible disappearance would misclassify that
    # legitimate collection as a contract violation.
    positive_without_items = 0
    negative_with_items = 0
    unexplained_positive: list[dict] = []
    for ep in episodes:
        records = traces.get(ep.get("episode"), [])
        for index in range(1, len(records)):
            prev, cur = records[index - 1], records[index]
            prev_power = (prev.get("resources") or {}).get("power")
            cur_power = (cur.get("resources") or {}).get("power")
            if prev_power is None or cur_power is None:
                continue
            delta = float(cur_power) - float(prev_power)
            prev_slots = set(_p_items_in(prev))
            cur_slots = set(_p_items_in(cur))
            disappeared = len(prev_slots - cur_slots)
            following = records[index + 1] if index + 1 < len(records) else None
            # The collection flag and the power readout can fall on opposite
            # sides of a decision boundary (the flag is per simulated frame, the
            # power global is read after the wait).  Allow a +-1 decision window;
            # R5 in the P contract records this amendment.
            collected_flag = any(
                bool((candidate.get("frame_events") or {}).get("item_collected"))
                for candidate in (prev, cur, following)
                if candidate is not None
            )
            if delta > 1e-6 and disappeared == 0 and not collected_flag:
                positive_without_items += 1
                unexplained_positive.append(
                    {"episode": ep.get("episode"), "decision": cur.get("decision"), "power_delta": round(delta, 3)}
                )
            if delta < -1e-6 and disappeared > 0:
                negative_with_items += 1
    hp6 = {
        "power_windows": len(power_windows),
        "positive_without_disappearing_items": positive_without_items,
        "negative_with_disappearing_items": negative_with_items,
        "unexplained_positive": unexplained_positive[:20],
        "pass": len(power_windows) > 0 and positive_without_items == 0 and negative_with_items == 0,
        "not_tested": not power_windows,
    }

    hypotheses = {"H-P1": hp1, "H-P2": hp2, "H-P3": hp3, "H-P4": hp4, "H-P5": hp5, "H-P6": hp6}
    tested = [name for name, value in hypotheses.items() if not value.get("not_tested")]
    failed = [name for name, value in hypotheses.items() if not value.get("pass") and not value.get("not_tested")]
    not_tested = [name for name, value in hypotheses.items() if value.get("not_tested")]
    report = {
        "schema": "v350-p-contract/1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _git_head(),
        "episodes": per_episode,
        "hypotheses": hypotheses,
        "summary": {"tested": tested, "failed": failed, "not_tested": not_tested,
                    "pass": not failed and not not_tested},
    }
    _write_json(Path(out_dir) / "p-contract.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m touhou_rl.probe35")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
        p.add_argument("--episodes", type=int, default=10)
        p.add_argument("--max-steps", type=int, default=3000)
        p.add_argument("--frameskip", type=int, default=2)
        p.add_argument("--practice-stage", type=int, default=0)
        p.add_argument("--full", action="store_true",
                       help="play the full game from stage 1 (3 lives) instead of practice (8 lives)")
        p.add_argument("--policy", choices=["teacher", "suicide", "quit"], default="teacher",
                       help="controlled death/left-game arms for S0-3 event evidence")
        p.add_argument("--quit-after", type=int, default=60,
                       help="steps before the quit policy exits the run (only with --policy quit)")
        p.add_argument("--difficulty", type=int, default=spec.LUNATIC, choices=[0, 1, 2, 3, 4])
        p.add_argument("--team", type=int, default=0)
        p.add_argument("--post-terminal-seconds", type=float, default=2.0)
        p.add_argument("--no-dialogue-taps", action="store_true")
        p.add_argument("--replay", type=Path, action="append", default=None,
                       help="play these replay(s) instead of running the teacher; "
                            "repeat the flag to collect a batch (v3.6.0 D2)")
        p.add_argument("--replay-stage", type=int, default=0)
        p.add_argument("--tactic", action="append", choices=["none", "clear_enemy", "collect_power"],
                       default=None,
                       help="v3.6.1 T2 arm(s); repeat to cycle per episode "
                            "(e.g. --tactic none --tactic clear_enemy gives an interleaved "
                            "paired schedule)")
        p.add_argument("--arm-seed", type=int, default=None,
                       help="v3.6.1 T2 pairing seed; pair k jitters the pilot policy with "
                            "seed+k in BOTH arms (the game is deterministic, so without this "
                            "every episode in an arm would be identical)")
        p.add_argument("--append", action="store_true",
                       help="append to an existing output directory instead of starting a new collection")
        p.add_argument("--episode-start", type=int, default=1,
                       help="first episode index (useful when appending a longer arm)")

    collect = sub.add_parser("collect", help="teacher episodes -> v350-trace/1")
    common(collect)

    pcontract = sub.add_parser("p-contract", help="controlled P-item contract experiment")
    common(pcontract)

    verify_cmd = sub.add_parser("verify", help="compute S0-1..S0-5 from an output directory")
    verify_cmd.add_argument("--out", type=Path, default=DEFAULT_OUT)
    verify_cmd.add_argument("--manifest", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        manifest = args.manifest
        if manifest is None and (Path(args.out) / "manifest.json").exists():
            manifest = Path(args.out) / "manifest.json"
        report = verify(args.out, manifest)
        print(json.dumps(report["gates"], ensure_ascii=False, indent=2))
        print("overall:", report["overall"])
        return 0 if report["overall"]["pass"] else 1

    cfg = ProbeConfig(
        out_dir=args.out,
        run_dir=args.run_dir,
        mode=args.command,
        policy=args.policy,
        quit_after=args.quit_after,
        episodes=len(args.replay) if args.replay else args.episodes,
        max_steps=args.max_steps,
        frameskip=args.frameskip,
        practice_stage=None if args.full else args.practice_stage,
        difficulty=args.difficulty,
        team=args.team,
        post_terminal_seconds=args.post_terminal_seconds,
        dialogue_taps=not args.no_dialogue_taps,
        replay_path=args.replay[0] if args.replay and len(args.replay) == 1 else None,
        replay_stage=args.replay_stage,
        replays=tuple(args.replay) if args.replay and len(args.replay) > 1 else (),
        tactics=tuple(args.tactic) if args.tactic else ("none",),
        arm_seed=args.arm_seed,
    )
    cfg.validate()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("trace", "bullets"):
        (cfg.out_dir / name).mkdir(parents=True, exist_ok=True)
    # A new run must not append to an older episodes.jsonl; the manifest and
    # offline gates both assume one collection per directory.  ``--append`` is
    # the explicit exception used to add the replay arm to the controlled arm.
    episodes_path = cfg.out_dir / "episodes.jsonl"
    if args.append:
        episodes = read_episodes(cfg.out_dir)
    else:
        episodes = []
        if episodes_path.exists():
            episodes_path.unlink()
    plan = episode_plan(cfg, args.episode_start)
    started = time.perf_counter()
    for offset, (index, replay, tactic, pair) in enumerate(plan, start=1):
        print(
            f"[{offset}/{len(plan)}] {cfg.mode} episode {index} tactic={tactic} pair={pair} replay={replay} ...",
            flush=True,
        )
        if replay is not None:
            ep = run_replay_episode(cfg, index, replay)
        else:
            ep = run_episode(cfg, index, tactic, pair)
        episodes.append(ep)
        stats = ep.get("stats", {})
        print(
            f"  decisions={stats.get('decisions')} steady={stats.get('steady_decisions')} "
            f"complete={stats.get('complete_rate')} "
            f"term={(ep.get('termination') or {}).get('reason')} "
            f"wall={stats.get('wall_seconds')}s",
            flush=True,
        )
    manifest = write_manifest(cfg.out_dir, cfg, episodes)
    print(f"manifest: {len(manifest['files'])} files, elapsed {time.perf_counter() - started:.0f}s")
    if cfg.mode == "p-contract":
        result = analyse_p_contract(cfg.out_dir, episodes)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0 if result["summary"]["pass"] else 1
    report = verify(cfg.out_dir, cfg.out_dir / "manifest.json")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    return 0 if report["overall"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
