"""Stage 0.5 boss probe: can we see the boss, damage it, and record a clear?

This module deliberately does **not** train anything and does **not** touch the
Oracle.  It runs the existing scripted teacher (optionally with a safety-gated
boss-aim preference) and records, every decision:

* live boss slots, positions and HP (``Enemy.life``, i32 at ``0x2DFC``);
* ``GameManagerFlags`` bits (game cleared / stage-clear sequence / transition);
* current spell id, player position and death events;
* every termination reason, plus a post-termination poll that distinguishes
  "the stage-clear screen froze the frame counter" from "the run ended".

The point is to answer three questions before Stage B spends days on a
hierarchical Oracle:

1. Is boss HP readable and does it move when the teacher shoots?
2. Does the teacher ever damage the boss, with or without aiming?
3. Is a stage clear observable at all in the current harness?

Usage (20 episodes = 10 teacher + 10 teacher with safety-gated aim)::

    python -m touhou_rl.boss_probe --out Document/Update/Alpha/v3.4.0/boss-probe \
        --practice-stage 0 --episodes-per-arm 10 \
        --arm teacher:0.0 --arm teacher-boss:0.6

The report is ``boss-probe.json`` plus a human-readable ``boss-probe.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import spec
from .env import EnvConfig, Th08Env
from .pilot import AvoidPilot, PilotConfig
from .state import BossInfo, GameState

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "v340-boss-probe/1"
TEACHER_BASELINE_FRAMES = 6400.8  # v3.3.0 ladder, 8000-frame cap (censored lower bound)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _git_head(root: Path = ROOT) -> str:
    """Resolve ``.git/HEAD`` without spawning git (sandbox-safe)."""
    head = root / ".git" / "HEAD"
    try:
        text = head.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    if text.startswith("ref: "):
        ref = text[5:]
        try:
            return (root / ".git" / ref).read_text(encoding="utf-8").strip()
        except OSError:
            return text
    return text


def _mean(values: list[float]) -> float | None:
    return round(float(statistics.fmean(values)), 3) if values else None


def _median(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 3) if values else None


def _safe_ratio(num: float, den: float) -> float | None:
    return round(float(num) / float(den), 4) if den else None


def _boss_dict(bosses: list[BossInfo]) -> list[dict[str, Any]]:
    return [
        {"slot": int(b.slot), "pointer": int(b.pointer), "x": round(float(b.x), 3),
         "y": round(float(b.y), 3), "life": int(b.life)}
        for b in bosses
    ]


def _hp_sum(bosses: list[BossInfo]) -> int:
    """Sum of non-negative HP: a dying frame can report a negative life."""
    return int(sum(max(0, int(b.life)) for b in bosses))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class ProbeConfig:
    out_dir: Path
    run_dir: Path
    practice_stage: int | None = 0
    episodes_per_arm: int = 10
    arms: list[tuple[str, float]] = field(
        default_factory=lambda: [("teacher", 0.0), ("teacher-boss", 0.6)]
    )
    max_steps: int = 6000
    frameskip: int = 2
    aim_safe_margin: float = 12.0
    post_terminal_seconds: float = 5.0
    trace: bool = True
    # Toggle Z while a dialogue is present: the game advances dialogue on a
    # shoot *edge* (Gui.cpp:599), so a permanently-held shoot key stalls the run
    # after a boss phase.  This is a probe mechanism, not a policy change.
    dialogue_taps: bool = True

    def validate(self) -> None:
        if self.episodes_per_arm < 1:
            raise ValueError("episodes_per_arm must be >= 1")
        if self.max_steps < 1 or self.frameskip < 1:
            raise ValueError("max_steps/frameskip must be >= 1")
        if not self.arms:
            raise ValueError("at least one arm is required")
        names = [name for name, _ in self.arms]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate arm names: {names}")
        for name, weight in self.arms:
            if not name or weight < 0.0:
                raise ValueError(f"invalid arm {name!r}: aim weight must be >= 0")
        if self.practice_stage is not None and not 0 <= self.practice_stage <= 6:
            raise ValueError("practice_stage must be in 0..6 or None")


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------


def _env_config(cfg: ProbeConfig) -> EnvConfig:
    return EnvConfig(
        team=0,
        difficulty=spec.LUNATIC,
        practice_stage=cfg.practice_stage,
        frameskip=cfg.frameskip,
        max_steps=cfg.max_steps,
        reset_mode="relaunch",
        run_dir=cfg.run_dir,
        windowed=True,
        # SendInput (DirectInput) path: the probe runs in the foreground.  The
        # PostMessage/dontUseDirectInput path was tried first and measured to
        # leave g_CurFrameInput at 0 on this build, so it cannot drive the game.
        use_postmessage_input=False,
        expose_raw_bullets=True,
        expose_bosses=False,
        expose_boss_info=True,
    )


def _new_episode(arm: str, aim: float, index: int, cfg: ProbeConfig, start_state: GameState) -> dict:
    return {
        "schema": SCHEMA,
        "arm": arm,
        "aim_weight": float(aim),
        "episode": int(index),
        "practice_stage": cfg.practice_stage,
        "frameskip": cfg.frameskip,
        "max_steps": cfg.max_steps,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frames": 0,
        "steps": 0,
        "wall_seconds": 0.0,
        "decisions_per_s": None,
        "terminated": False,
        "truncated": False,
        "termination_reason": "",
        "stage_start": int(start_state.stage),
        "stage_end": int(start_state.stage),
        "stage_frames_end": int(start_state.stage_frames),
        "deaths": 0,
        "lives_left": float(start_state.lives),
        "bombs_used": 0,
        "score": 0,
        "is_practice": bool(start_state.is_practice),
        "boss_present_steps": 0,
        "first_boss_frame": None,
        "last_boss_frame": None,
        "boss_hp_first": None,
        "boss_hp_min": None,
        "boss_hp_last": None,
        "boss_hp_drop_fraction": None,
        "boss_kill": False,
        "boss_absent_after_present_steps": 0,
        "boss_kill_frame": None,
        "spells": [],
        "clear_sequence_steps": 0,
        "game_cleared_steps": 0,
        "dialogue_steps": 0,
        "dialogue_taps": 0,
        "stage_clear_screen_steps": 0,
        "stage_transitions": [],
        "death_events": [],
        "flags_transitions": [],
        "hp_nonfinite": 0,
        "hp_negative": 0,
        "hp_samples": 0,
        "hp_values": [],
        "hp_trace": [],
        "post_terminal": [],
        "error": None,
    }


def _step_record(step: int, action: np.ndarray, pilot: AvoidPilot, state: GameState,
                 bosses: list[BossInfo], info: dict, dialogue_tap: bool = False) -> dict:
    return {
        "decision": int(step),
        "frame": int(state.gameplay_frame),
        "stage": int(state.stage),
        "stage_frames": int(state.stage_frames),
        "flags": int(state.flags) & 0xFFFFFFFF,
        "game_cleared": bool(state.game_cleared),
        "stage_clear_sequence_active": bool(state.stage_clear_sequence_active),
        "stage_transition_state": int(state.stage_transition_state),
        "spell": int(state.current_spell),
        "lives": float(state.lives),
        "bombs": float(state.bombs),
        "deaths": int(state.deaths),
        "player_x": round(float(state.player_x), 3),
        "player_y": round(float(state.player_y), 3),
        "player_state": int(getattr(state, "player_state", -1)),
        "bullets": int(state.active_bullets),
        "action": [int(v) for v in np.asarray(action).reshape(-1)],
        "clearance": round(float(pilot.debug.best_clearance), 3),
        "bosses": _boss_dict(bosses),
        "boss_count": len(bosses),
        "boss_hp_sum": _hp_sum(bosses),
        "dialogue_present": bool(state.gui_dialogue_present),
        "dialogue_msg_idx": int(state.gui_dialogue_msg_idx),
        "stage_clear_screen_state": int(state.gui_stage_clear_screen_state),
        "gui_boss_present": bool(state.gui_boss_present),
        "dialogue_tap": bool(dialogue_tap),
    }


def _update_episode(ep: dict, rec: dict, prev: dict | None) -> None:
    state = rec
    bosses = state["bosses"]
    hp_sum = int(state["boss_hp_sum"])
    if bosses:
        ep["boss_present_steps"] += 1
        if ep["first_boss_frame"] is None:
            ep["first_boss_frame"] = state["frame"]
            if hp_sum > 0:
                ep["boss_hp_first"] = hp_sum
        ep["last_boss_frame"] = state["frame"]
        if hp_sum > 0:
            ep["boss_hp_last"] = hp_sum
            ep["boss_hp_min"] = hp_sum if ep["boss_hp_min"] is None else min(ep["boss_hp_min"], hp_sum)
        else:
            # A boss whose life has reached 0 must pin the minimum to 0, not
            # leave it at the last positive sample (3 HP was the observed trap).
            ep["boss_hp_min"] = 0 if ep["boss_hp_min"] is None else min(ep["boss_hp_min"], 0)
        if ep["boss_hp_first"] is not None and ep["boss_hp_min"] is not None:
            first = ep["boss_hp_first"]
            ep["boss_hp_drop_fraction"] = (
                round((first - ep["boss_hp_min"]) / first, 4) if first > 0 else None
            )
        if not ep["boss_kill"] and any(int(b["life"]) <= 0 for b in bosses):
            ep["boss_kill"] = True
            ep["boss_kill_frame"] = state["frame"]
        ep["hp_samples"] += 1
        for boss in bosses:
            life = boss["life"]
            if not _finite(life):
                ep["hp_nonfinite"] += 1
            elif life < 0:
                ep["hp_negative"] += 1
            else:
                ep["hp_values"].append(int(life))
    if prev is not None and prev["bosses"] and not bosses:
        ep["boss_absent_after_present_steps"] += 1
    elif bosses:
        ep["boss_absent_after_present_steps"] = 0

    if state["stage_clear_sequence_active"]:
        ep["clear_sequence_steps"] += 1
    if state["game_cleared"]:
        ep["game_cleared_steps"] += 1
    if state["dialogue_present"]:
        ep["dialogue_steps"] += 1
    if state["dialogue_tap"]:
        ep["dialogue_taps"] += 1
    if state["stage_clear_screen_state"] != 0:
        ep["stage_clear_screen_steps"] += 1
    if state["spell"] >= 0 and state["spell"] < 1000 and state["spell"] not in ep["spells"]:
        ep["spells"].append(int(state["spell"]))
    if prev is not None:
        if state["frame"] % 30 == 0:
            ep["hp_trace"].append(
                {
                    "frame": state["frame"],
                    "stage": state["stage"],
                    "boss_count": state["boss_count"],
                    "boss_hp_sum": hp_sum,
                    "spell": state["spell"],
                    "clear_sequence": state["stage_clear_sequence_active"],
                    "game_cleared": state["game_cleared"],
                }
            )
        if state["stage"] != prev["stage"]:
            ep["stage_transitions"].append(
                {
                    "frame": state["frame"],
                    "from": prev["stage"],
                    "to": state["stage"],
                    "flags": state["flags"],
                    "clear_sequence": state["stage_clear_sequence_active"],
                }
            )
        if state["deaths"] > prev["deaths"]:
            ep["death_events"].append(
                {
                    "frame": state["frame"],
                    "stage": state["stage"],
                    "player_x": state["player_x"],
                    "player_y": state["player_y"],
                    "boss_hp_sum": hp_sum,
                }
            )
        interesting = (
            state["flags"] != prev["flags"]
            and (
                state["game_cleared"] != prev["game_cleared"]
                or state["stage_clear_sequence_active"] != prev["stage_clear_sequence_active"]
                or state["stage_transition_state"] != prev["stage_transition_state"]
            )
        )
        if interesting:
            ep["flags_transitions"].append(
                {
                    "frame": state["frame"],
                    "flags": state["flags"],
                    "game_cleared": state["game_cleared"],
                    "clear_sequence": state["stage_clear_sequence_active"],
                    "transition_state": state["stage_transition_state"],
                }
            )


def _probe_post_terminal(env: Th08Env, seconds: float) -> list[dict]:
    """Poll after termination: a clear screen can freeze gameplayFrameCounter."""
    session = env.session
    if session is None or session.reader is None:
        return []
    reader = session.reader
    samples: list[dict] = []
    deadline = time.perf_counter() + max(0.0, seconds)
    while time.perf_counter() < deadline:
        try:
            state = reader.read_state(with_bullets=False, with_title=False)
        except OSError as exc:
            samples.append({"error": str(exc)})
            break
        samples.append(
            {
                "wall": round(time.perf_counter(), 3),
                "supervisor_state": int(state.supervisor_state),
                "stage": int(state.stage),
                "stage_frames": int(state.stage_frames),
                "gameplay_frame": int(state.gameplay_frame),
                "lives": float(state.lives),
                "flags": int(state.flags) & 0xFFFFFFFF,
                "game_cleared": bool(state.game_cleared),
                "stage_clear_sequence_active": bool(state.stage_clear_sequence_active),
                "stage_transition_state": int(state.stage_transition_state),
            }
        )
        if len(samples) >= 30:
            break
        time.sleep(0.2)
    return samples


def run_episode(cfg: ProbeConfig, arm: str, aim: float, index: int) -> dict:
    env = Th08Env(_env_config(cfg))
    pilot = AvoidPilot(
        PilotConfig(
            frameskip=cfg.frameskip,
            aim_weight=float(aim),
            aim_safe_margin=cfg.aim_safe_margin if aim else 0.0,
        )
    )
    started = time.perf_counter()
    ep: dict = {}
    trace_path = cfg.out_dir / "trace" / f"{arm}-ep{index:02d}.jsonl"
    trace_handle = None
    prev_rec: dict | None = None
    try:
        obs, info = env.reset()
        start_state = env.last_state or GameState()
        ep = _new_episode(arm, aim, index, cfg, start_state)
        if cfg.trace:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = trace_path.open("w", encoding="utf-8")
        reader = env.session.reader if env.session is not None else None
        bosses = reader.read_boss_info() if reader is not None else []
        dialogue_phase = 0

        for step in range(1, cfg.max_steps + 1):
            prev_state = env.last_state
            boss_array = np.array([[b.x, b.y] for b in bosses], dtype=np.float32).reshape(-1, 2)
            action = pilot.act(
                obs,
                info.get("bullets_raw"),
                bombs=int(round(float(info.get("bombs", 0)))),
                bosses=boss_array,
            )
            dialogue_tap = False
            if (
                cfg.dialogue_taps
                and prev_state is not None
                and prev_state.gui_dialogue_present
            ):
                action = np.array(action, dtype=np.int64, copy=True)
                action[1] = dialogue_phase
                dialogue_phase ^= 1
                dialogue_tap = True
            else:
                dialogue_phase = 0
            obs, _reward, terminated, truncated, info = env.step(action)
            state = env.last_state
            if state is None:  # pragma: no cover - env always sets it after step
                raise RuntimeError("env.last_state is None after step")
            bosses = info.get("boss_info") or []
            rec = _step_record(step, action, pilot, state, bosses, info, dialogue_tap=dialogue_tap)
            _update_episode(ep, rec, prev_rec)
            if trace_handle is not None:
                trace_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            prev_rec = rec
            ep["steps"] = step
            if terminated or truncated:
                episode_info = info.get("episode", {}) or {}
                ep["terminated"] = bool(terminated)
                ep["truncated"] = bool(truncated)
                ep["termination_reason"] = str(
                    episode_info.get("termination_reason", info.get("termination_reason", ""))
                )
                ep["frames"] = int(episode_info.get("frames", state.run_frames))
                ep["deaths"] = int(episode_info.get("deaths", state.deaths - start_state.deaths))
                ep["lives_left"] = float(episode_info.get("lives", state.lives))
                ep["bombs_used"] = max(
                    0,
                    int(state.bombs_used) - int(getattr(start_state, "bombs_used", 0)),
                )
                if ep["bombs_used"] == 0:
                    ep["bombs_used"] = int(
                        round(max(0.0, float(start_state.bombs) - float(state.bombs)))
                    )
                ep["score"] = int(episode_info.get("score", state.score - start_state.score))
                break

        ep["stage_end"] = int((env.last_state or start_state).stage)
        ep["stage_frames_end"] = int((env.last_state or start_state).stage_frames)
        if not ep.get("frames"):
            ep["frames"] = int((env.last_state or start_state).run_frames)
        ep["post_terminal"] = _probe_post_terminal(env, cfg.post_terminal_seconds)
        post = ep["post_terminal"]
        ep["post_terminal_clear"] = any(
            (s.get("stage_clear_sequence_active") or s.get("game_cleared")) for s in post
        )
        ep["post_terminal_stage_advanced"] = any(
            int(s.get("stage", ep["stage_end"])) > ep["stage_end"] for s in post
        )
    except Exception as exc:  # keep a 20-episode run alive; the report records the failure
        if not ep:
            ep = _new_episode(arm, aim, index, cfg, GameState())
        ep["error"] = f"{type(exc).__name__}: {exc}"
        ep["terminated"] = False
        ep["truncated"] = False
        ep["termination_reason"] = "probe_error"
    finally:
        if trace_handle is not None:
            trace_handle.close()
        try:
            env.close()
        except Exception:
            pass
        if ep:
            ep["wall_seconds"] = round(time.perf_counter() - started, 1)
            ep["decisions_per_s"] = _safe_ratio(ep.get("steps", 0), ep["wall_seconds"])
            # Keep the report small: the full per-step values live in trace/.
            ep["hp_values"] = sorted(set(ep.get("hp_values", [])))
    return ep


# ---------------------------------------------------------------------------
# aggregation and gates
# ---------------------------------------------------------------------------


def summarise(episodes: list[dict], arms: list[tuple[str, float]]) -> dict:
    by_arm: dict[str, list[dict]] = {name: [] for name, _ in arms}
    for ep in episodes:
        by_arm.setdefault(str(ep.get("arm", "unknown")), []).append(ep)

    arm_rows: dict[str, dict] = {}
    for name, _weight in arms:
        rows = by_arm.get(name, [])
        boss_rows = [r for r in rows if int(r.get("boss_present_steps", 0)) > 0]
        damaged = [r for r in boss_rows if (r.get("boss_hp_drop_fraction") or 0.0) >= 0.5]
        killed = [r for r in boss_rows if r.get("boss_kill")]
        cleared = [
            r for r in rows
            if int(r.get("clear_sequence_steps", 0)) > 0
            or int(r.get("stage_clear_screen_steps", 0)) > 0
            or bool(r.get("post_terminal_clear"))
            or bool(r.get("post_terminal_stage_advanced"))
        ]
        frames = [float(r["frames"]) for r in rows if r.get("frames")]
        deaths = [float(r.get("deaths", 0)) for r in rows]
        arm_rows[name] = {
            "episodes": len(rows),
            "aim_weight": float(dict(arms)[name]),
            "errors": sum(1 for r in rows if r.get("error")),
            "frames_mean": _mean(frames),
            "frames_median": _median(frames),
            "frames_min": min(frames) if frames else None,
            "frames_max": max(frames) if frames else None,
            "deaths_mean": _mean(deaths),
            "boss_episodes": len(boss_rows),
            "boss_present_rate": _safe_ratio(len(boss_rows), len(rows)),
            "boss_damage_50_episodes": len(damaged),
            "boss_damage_50_rate": _safe_ratio(len(damaged), len(boss_rows)),
            "boss_kill_episodes": len(killed),
            "boss_kill_rate": _safe_ratio(len(killed), len(boss_rows)),
            "clear_episodes": len(cleared),
            "clear_rate": _safe_ratio(len(cleared), len(rows)),
            "hp_first_mean": _mean([float(r["boss_hp_first"]) for r in boss_rows if r.get("boss_hp_first") is not None]),
            "hp_min_mean": _mean([float(r["boss_hp_min"]) for r in boss_rows if r.get("boss_hp_min") is not None]),
            "drop_fraction_mean": _mean([float(r["boss_hp_drop_fraction"]) for r in boss_rows if r.get("boss_hp_drop_fraction") is not None]),
            "spells": sorted({int(s) for r in boss_rows for s in r.get("spells", [])}),
            "termination_reasons": dict(Counter(str(r.get("termination_reason", "")) for r in rows)),
            "clear_sequence_steps": sum(int(r.get("clear_sequence_steps", 0)) for r in rows),
            "game_cleared_steps": sum(int(r.get("game_cleared_steps", 0)) for r in rows),
            "dialogue_episodes": sum(1 for r in rows if int(r.get("dialogue_steps", 0)) > 0),
            "dialogue_steps": sum(int(r.get("dialogue_steps", 0)) for r in rows),
            "dialogue_taps": sum(int(r.get("dialogue_taps", 0)) for r in rows),
            "stage_clear_screen_steps": sum(int(r.get("stage_clear_screen_steps", 0)) for r in rows),
        }

    all_boss = [r for r in episodes if int(r.get("boss_present_steps", 0)) > 0]
    hp_samples = sum(int(r.get("hp_samples", 0)) for r in all_boss)
    hp_nonfinite = sum(int(r.get("hp_nonfinite", 0)) for r in all_boss)
    hp_negative = sum(int(r.get("hp_negative", 0)) for r in all_boss)
    hp_values = sorted({int(v) for r in all_boss for v in r.get("hp_values", [])})
    clear_episodes = [
        r for r in episodes
        if int(r.get("clear_sequence_steps", 0)) > 0
        or int(r.get("stage_clear_screen_steps", 0)) > 0
        or bool(r.get("post_terminal_clear"))
        or bool(r.get("post_terminal_stage_advanced"))
    ]

    # The treatment arm is the one with the largest aim weight, the control the
    # one with the smallest.  With a single arm, G0.4 is informational.
    ordered = sorted(arms, key=lambda item: item[1])
    control_name = ordered[0][0] if ordered else None
    treatment_name = ordered[-1][0] if ordered else None
    treatment = arm_rows.get(treatment_name or "", {})
    control = arm_rows.get(control_name or "", {})
    treatment_weight = dict(arms).get(treatment_name or "", 0.0)

    g01 = {
        "name": "G0.1 boss HP readable",
        "pass": bool(hp_samples > 0 and hp_nonfinite == 0 and hp_negative == 0),
        "hp_samples": hp_samples,
        "hp_nonfinite": hp_nonfinite,
        "hp_negative": hp_negative,
        "distinct_hp_values": len(hp_values),
        "hp_value_range": [min(hp_values), max(hp_values)] if hp_values else None,
    }
    g02 = {
        "name": "G0.2 stage clear observable",
        "status": "pass" if clear_episodes else "not_tested",
        "clear_episodes": len(clear_episodes),
        "stage_clear_screen_steps": sum(int(r.get("stage_clear_screen_steps", 0)) for r in episodes),
        "detail": "no clear occurred; the probe cannot yet distinguish clear from game over"
        if not clear_episodes else "at least one clear sequence / stage advance was recorded",
    }
    if treatment_weight == 0.0:
        g03 = {
            "name": "G0.3 teacher can damage boss",
            "status": "not_tested",
            "detail": "the treatment arm has aim_weight=0; run a boss-aim arm",
        }
    elif not treatment.get("boss_episodes"):
        g03 = {
            "name": "G0.3 teacher can damage boss",
            "status": "not_tested",
            "detail": "no episode reached the boss",
        }
    else:
        rate = treatment.get("boss_damage_50_rate")
        g03 = {
            "name": "G0.3 teacher can damage boss",
            "status": "pass" if (rate or 0.0) >= 0.5 else "fail",
            "boss_damage_50_rate": rate,
            "boss_kill_rate": treatment.get("boss_kill_rate"),
            "threshold": ">= 0.5 of boss episodes with HP drop >= 50%",
        }
    if control_name == treatment_name or not control.get("frames_mean"):
        g04 = {
            "name": "G0.4 no catastrophic survival regression",
            "status": "informational",
            "detail": "no same-protocol control arm; compare with the v3.3 teacher baseline 6400.8",
            "treatment_frames_mean": treatment.get("frames_mean"),
            "teacher_baseline_frames": TEACHER_BASELINE_FRAMES,
        }
    else:
        ratio = (treatment.get("frames_mean") or 0.0) / float(control["frames_mean"])
        g04 = {
            "name": "G0.4 no catastrophic survival regression",
            "status": "pass" if ratio >= 0.9 else "warn",
            "ratio_vs_control": round(float(ratio), 4),
            "threshold": ">= 0.9",
            "control_frames_mean": control.get("frames_mean"),
            "treatment_frames_mean": treatment.get("frames_mean"),
        }

    if not g01["pass"]:
        verdict = "hp-unreadable"
    elif g03.get("status") == "pass":
        verdict = "damage-possible"
    elif g03.get("status") == "fail":
        verdict = "no-damage"
    else:
        verdict = "observability-only"
    return {
        "schema": SCHEMA,
        "arms": arm_rows,
        "overall": {
            "episodes": len(episodes),
            "errors": sum(1 for r in episodes if r.get("error")),
            "clear_episodes": len(clear_episodes),
            "hp_samples": hp_samples,
            "verdict": verdict,
        },
        "gates": {"G0.1": g01, "G0.2": g02, "G0.3": g03, "G0.4": g04},
    }


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict) -> str:
    cfg = report.get("config", {})
    arms = report.get("arms", {})
    lines = [
        "# v3.4.0 Stage 0.5 报告：Boss 探针",
        "",
        f"- 日期：{report.get('generated', '')}",
        f"- 代码版本：`{report.get('git_head', 'unknown')}`",
        f"- 模式：{'练习 Stage ' + str(cfg.get('practice_stage') + 1) if cfg.get('practice_stage') is not None else '完整游戏'}",
        f"- 每臂局数：{cfg.get('episodes_per_arm')}；决策上限：{cfg.get('max_steps')}（{int(cfg.get('max_steps', 0)) * int(cfg.get('frameskip', 1))} 帧）",
        f"- 判定：**{report.get('overall', {}).get('verdict', 'unknown')}**",
        "",
        "## 1 一页数字",
        "",
        "| 臂 | aim | 局数 | frames_mean | deaths_mean | 到 Boss | HP 降 ≥50% | 击杀 | clear |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in arms.items():
        lines.append(
            f"| {name} | {_fmt(row.get('aim_weight', cfg.get('aim_weight')))} | {row.get('episodes')} | "
            f"{_fmt(row.get('frames_mean'), 1)} | {_fmt(row.get('deaths_mean'))} | "
            f"{row.get('boss_episodes')} | {row.get('boss_damage_50_episodes')} | "
            f"{row.get('boss_kill_episodes')} | {row.get('clear_episodes')} |"
        )
    lines += [
        "",
        "## 2 逐局明细",
        "",
        "| 臂 | 局 | frames | 终止原因 | deaths | Boss 首帧 | HP 首/最低/末 | 降幅 | 击杀 | spells | clear |",
        "|---|---:|---:|---|---:|---:|---|---:|---|---:|---:|",
    ]
    for ep in report.get("episodes", []):
        hp = f"{_fmt(ep.get('boss_hp_first'))}/{_fmt(ep.get('boss_hp_min'))}/{_fmt(ep.get('boss_hp_last'))}"
        lines.append(
            f"| {ep.get('arm')} | {ep.get('episode')} | {ep.get('frames')} | "
            f"{ep.get('termination_reason') or '-'} | {ep.get('deaths')} | {_fmt(ep.get('first_boss_frame'))} | "
            f"{hp} | {_fmt(ep.get('boss_hp_drop_fraction'))} | {int(bool(ep.get('boss_kill')))} | "
            f"{','.join(str(s) for s in ep.get('spells', [])) or '-'} | "
            f"{int(bool(ep.get('clear_sequence_steps') or ep.get('post_terminal_clear')))} |"
        )
    lines += [
        "",
        "## 3 门禁",
        "",
        "| 编号 | 判据 | 结果 | 细节 |",
        "|---|---|---|---|",
    ]
    for key, gate in report.get("gates", {}).items():
        status = gate.get("status", "pass" if gate.get("pass") else "fail")
        detail = gate.get("detail") or gate.get("threshold") or ""
        lines.append(f"| {key} | {gate.get('name')} | **{status}** | {detail} |")
    lines += [
        "",
        "## 4 终止原因与 clear 证据",
        "",
    ]
    for name, row in arms.items():
        lines.append(f"- **{name}**：终止原因 `{row.get('termination_reasons')}`；"
                     f"clear_sequence 步数 {row.get('clear_sequence_steps')}；"
                     f"stage_clear_screen 步数 {row.get('stage_clear_screen_steps')}；"
                     f"game_cleared 步数 {row.get('game_cleared_steps')}；"
                     f"dialogue 步数 {row.get('dialogue_steps')}（tap {row.get('dialogue_taps')}）")
    lines += [
        "",
        "## 5 结论与下一步",
        "",
        f"- 总判定：**{report.get('overall', {}).get('verdict', 'unknown')}**",
        "- 如果 G0.1 不过：先查 `Enemy.life` 偏移/类型，不要写 Oracle。",
        "- 如果 G0.3 不过：先修 Boss 站位/DPS；Escape 已经由教师覆盖，不要重写 Escape。",
        "- 如果 G0.2 不过：先补 `GameManagerFlags` 的 clear 事件，再用 clear 作为 O4 的门。",
        "- 如果都过：按 Stage B Boss-first 设计写预注册，先用 `practice_stage=0` 调 Boss。",
        "",
        f"原始数据：`boss-probe.json`；逐步 trace：`trace/`（{report.get('trace_files', 0)} 个文件）。",
        "",
    ]
    return "\n".join(lines)


def write_report(cfg: ProbeConfig, episodes: list[dict], started: float) -> dict:
    summary = summarise(episodes, cfg.arms)
    report = {
        "schema": SCHEMA,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _git_head(),
        "host": {"python": __import__("sys").version.split()[0], "platform": __import__("sys").platform},
        "config": {
            "practice_stage": cfg.practice_stage,
            "episodes_per_arm": cfg.episodes_per_arm,
            "arms": [{"name": name, "aim_weight": weight} for name, weight in cfg.arms],
            "max_steps": cfg.max_steps,
            "frameskip": cfg.frameskip,
            "aim_safe_margin": cfg.aim_safe_margin,
            "post_terminal_seconds": cfg.post_terminal_seconds,
            "dialogue_taps": cfg.dialogue_taps,
            "teacher_baseline_frames": TEACHER_BASELINE_FRAMES,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "episodes": episodes,
        "trace_files": len(list((cfg.out_dir / "trace").glob("*.jsonl"))) if (cfg.out_dir / "trace").exists() else 0,
        **summary,
    }
    json_path = cfg.out_dir / "boss-probe.json"
    md_path = cfg.out_dir / "boss-probe.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    manifest = []
    for path in (json_path, md_path):
        manifest.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (cfg.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def load_episodes(path: Path) -> list[dict]:
    episodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            episodes.append(json.loads(line))
    return episodes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_arms(values: list[str] | None, default_weight: float) -> list[tuple[str, float]]:
    if not values:
        return [("teacher-boss", float(default_weight))]
    arms = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"--arm must be name:aim_weight, got {value!r}")
        name, weight = value.rsplit(":", 1)
        arms.append((name.strip(), float(weight)))
    return arms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 0.5 boss probe (no training)")
    parser.add_argument("--out", type=Path, default=ROOT / "Document/Update/Alpha/v3.4.0/boss-probe")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "third_party/th08/build/rl/boss-probe-session")
    parser.add_argument("--practice-stage", type=int, default=0)
    parser.add_argument("--full", action="store_true", help="run the full game instead of a practice stage")
    parser.add_argument("--episodes-per-arm", type=int, default=10)
    parser.add_argument("--arm", action="append", default=None, help="name:aim_weight (repeatable)")
    parser.add_argument("--aim-weight", type=float, default=0.6, help="default aim weight for one-arm runs")
    parser.add_argument("--aim-safe-margin", type=float, default=12.0)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--post-terminal-seconds", type=float, default=5.0)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--no-dialogue-taps", action="store_true", help="do not toggle Z to advance dialogue")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-episodes", type=Path, default=None, help="rebuild a report from episodes.jsonl")
    args = parser.parse_args(argv)

    cfg = ProbeConfig(
        out_dir=args.out,
        run_dir=args.run_dir,
        practice_stage=None if args.full else args.practice_stage,
        episodes_per_arm=args.episodes_per_arm,
        arms=_parse_arms(args.arm, args.aim_weight),
        max_steps=args.max_steps,
        frameskip=args.frameskip,
        aim_safe_margin=args.aim_safe_margin,
        post_terminal_seconds=args.post_terminal_seconds,
        trace=not args.no_trace,
        dialogue_taps=not args.no_dialogue_taps,
    )
    cfg.validate()
    started = time.perf_counter()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "trace").mkdir(parents=True, exist_ok=True)

    if args.from_episodes is not None:
        episodes = load_episodes(args.from_episodes)
        report = write_report(cfg, episodes, started)
        print(json.dumps({"verdict": report["overall"]["verdict"], "episodes": len(episodes)}, ensure_ascii=False))
        return 0

    plan = {
        "mode": "full" if cfg.practice_stage is None else f"practice-stage-{cfg.practice_stage + 1}",
        "arms": [{"name": name, "aim_weight": weight} for name, weight in cfg.arms],
        "episodes_per_arm": cfg.episodes_per_arm,
        "max_steps": cfg.max_steps,
        "frames": cfg.max_steps * cfg.frameskip,
        "out_dir": str(cfg.out_dir),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps({"started": plan}, ensure_ascii=False), flush=True)

    episodes: list[dict] = []
    episodes_path = cfg.out_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as handle:
        try:
            for arm_name, aim in cfg.arms:
                for index in range(1, cfg.episodes_per_arm + 1):
                    print(f"[{arm_name} {index}/{cfg.episodes_per_arm}] starting", flush=True)
                    ep = run_episode(cfg, arm_name, aim, index)
                    episodes.append(ep)
                    handle.write(json.dumps(ep, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                "arm": arm_name,
                                "episode": index,
                                "frames": ep.get("frames"),
                                "termination_reason": ep.get("termination_reason"),
                                "boss_present_steps": ep.get("boss_present_steps"),
                                "boss_hp_first": ep.get("boss_hp_first"),
                                "boss_hp_min": ep.get("boss_hp_min"),
                                "boss_hp_drop_fraction": ep.get("boss_hp_drop_fraction"),
                                "boss_kill": ep.get("boss_kill"),
                                "clear_sequence_steps": ep.get("clear_sequence_steps"),
                                "error": ep.get("error"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("interrupted; writing the partial report", flush=True)
    report = write_report(cfg, episodes, started)
    print(json.dumps({"verdict": report["overall"]["verdict"], "episodes": len(episodes)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
