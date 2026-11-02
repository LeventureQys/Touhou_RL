"""Phase 1 diagnostic pixel-student rollout; output is NOT approved training data.

Each decision uses the same initial double-render / FrameStack / act path as
``evaluate.run_evaluation``. Privileged observations only reach the sidecar teacher
and recorder, never the student. A fresh empty output directory is mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import spec, vision
from .env import EnvConfig, Th08Env
from .evaluate import _regions_from_screen, resolve_vision_depth
from .obs import Observation
from .pilot import AvoidPilot, PilotConfig
from .vision_data import META_FIELDS, VisionShardWriter
from .winapi import keep_awake, window_visible
from .vmodels import load_vision_checkpoint, vision_arch, VISION_ARCH_V3

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "v200-student-state-diagnostic/1"


@dataclass
class StudentRolloutConfig:
    checkpoint: Path
    out_dir: Path
    episodes: int = 2
    max_steps: int = 12000
    frameskip: int = 2
    difficulty: int = spec.LUNATIC
    team: int = 0
    practice_stage: int | None = None
    seed: int = 11
    device: str = "cuda"
    shard_size: int = 128
    max_seconds: float = 0.0
    run_dir: Path = ROOT / "third_party/th08/build/rl/v200-student-session"
    pilot: PilotConfig = field(default_factory=lambda: PilotConfig(switch_penalty=0.0))
    env_factory: Callable[[EnvConfig], object] | None = None
    behavior: str = "student"
    epsilon: float = 0.0
    record_input_evidence: bool = False
    protocol_sha256: str | None = None


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(info: dict) -> dict:
    return {**info.get("state", {}), **info}


def _teacher_query(teacher: AvoidPilot, observation: dict, info: dict) -> tuple[np.ndarray, float]:
    """Return the teacher label without mutating previous-action or bomb memory."""
    raw = info.get("bullets_raw")
    player, positions, velocities, half = teacher._prepare(observation, raw)
    from .pilot import _boss_x

    values, clearances = teacher._rollout_scores(
        player, positions, velocities, half, boss_x=_boss_x(info.get("bosses"))
    )
    candidate = int(np.argmax(values))
    move, focus = candidate % 9, candidate // 9
    clearance = float(clearances[candidate])
    bomb = int(
        teacher._bomb_cooldown <= 0
        and clearance < teacher.config.bomb_min_clearance
        and int(round(float(info.get("bombs", 0) or 0))) > 0
    )
    if bomb:
        focus = 1
    action = np.asarray([move, int(teacher.config.always_shoot), focus, bomb], dtype=np.int64)
    return action, clearance


def _capture_live_snapshot(env, fallback_observation: dict, fallback_info: dict):
    """Capture pixels' neighboring privileged snapshot without feeding it to student."""
    before = _frame_key(env)
    screen = env.render()
    after_render = _frame_key(env)
    if hasattr(env, "_capture_observation") and hasattr(env, "_state_dict"):
        observation_obj, state = env._capture_observation()
        observation = {
            "self_feat": observation_obj.self_feat.astype(np.float32),
            "bullets": observation_obj.bullets.astype(np.float32),
            "danger": observation_obj.danger_map.astype(np.float32),
        }
        info = env._state_dict(state)
        info["raw_player_position"] = [float(state.player_x), float(state.player_y)]
        info["raw_player_state"] = int(getattr(state, 'player_state', -1))
        info["raw_player_half"] = [float(state.hurtbox_half_x), float(state.hurtbox_half_y)]
        if getattr(env.config, "expose_raw_bullets", False) and getattr(env, "_last_bullets", None) is not None:
            info["bullets_raw"] = env._raw_bullet_dict()
        if getattr(env.config, "expose_bosses", False):
            try:
                info["bosses"] = env.session.reader.read_boss_positions()
            except (OSError, AttributeError):
                info["bosses"] = np.zeros((0, 2), dtype=np.float32)
    else:
        observation, info = fallback_observation, _state(fallback_info)
    stamp = _frame_key(env)
    return screen, observation, info, before, after_render, stamp


def _raw_safety_snapshot(info: dict, before, after, stamp) -> dict:
    """Persist all exposed active bullets, never reconstruct a pool from top-64.

    complete means complete *existing active bullet pool*, not a complete world.
    The surrounding stamps cover the sequential player/pool read, not atomic RAM.
    """
    raw = info.get("bullets_raw")
    result = {
        "schema": "v200-raw-safety-snapshot/1", "frame": int(info["frame"]),
        "stage": int(info["stage"]), "complete": False,
        "capture_stamps": [list(before), list(after), list(stamp)],
        "world_complete": False, "player_state": info.get('raw_player_state'),
        "unmodeled": ["births", "nonlinear-motion", "lasers", "enemy-collision", "invulnerability", "input-latency"],
    }
    if raw is None or "raw_player_position" not in info:
        result["invalid_reason"] = "raw-pool-or-player-missing"
        return result
    try:
        pos = np.asarray(raw["position"], dtype=np.float32).reshape(-1, 2)
        vel = np.asarray(raw["velocity"], dtype=np.float32).reshape(-1, 2)
        half = np.asarray(raw["half"], dtype=np.float32).reshape(-1)
        half_xy = np.asarray(raw.get("half_xy", np.repeat(half[:, None], 2, axis=1)), dtype=np.float32).reshape(-1, 2)
        player = np.asarray(info["raw_player_position"], dtype=np.float32)
        if player.shape != (2,) or pos.shape != vel.shape or len(pos) != len(half) or half_xy.shape != (len(pos), 2):
            raise ValueError("raw shape mismatch")
        if any(not np.isfinite(x).all() for x in (pos, vel, half, half_xy, player)) or np.any(half < 0) or np.any(half_xy < 0):
            raise ValueError("invalid raw geometry")
        result.update(player_position=player.tolist(), player_half=info.get("raw_player_half"),
                      bullets={"position": pos.tolist(), "velocity": vel.tolist(), "half": half.tolist(), "half_xy": half_xy.tolist(),
                               "slot_id": np.asarray(raw.get("slot_id", np.arange(len(pos))), dtype=np.int32).tolist(),
                               "state": np.asarray(raw.get("state", np.zeros(len(pos))), dtype=np.uint16).tolist(),
                               "collision_disabled": np.asarray(raw.get("collision_disabled", np.zeros(len(pos))), dtype=np.int8).tolist()},
                      generation_valid=False,
                      captured_count=len(pos), complete=before == after == stamp == (int(info["stage"]), int(info["frame"])))
    except (KeyError, ValueError, TypeError):
        result["invalid_reason"] = "malformed-raw-geometry"
    return result


def _label_observation(data: dict) -> Observation:
    return Observation(
        self_feat=np.asarray(data["self_feat"], dtype=np.float32),
        bullets=np.asarray(data["bullets"], dtype=np.float16),
        danger_map=np.asarray(data["danger"], dtype=np.uint8),
    )


def _frame_key(env) -> tuple[int, int]:
    """Independent live-frame stamp surrounding capture; fail closed without a reader."""
    session = getattr(env, "session", None)
    reader = getattr(session, "reader", None)
    if reader is None:
        raise RuntimeError("student rollout requires an independent env.session.reader frame stamp")
    state = reader.read_state(with_bullets=False, with_title=False)
    return int(state.stage), int(state.gameplay_frame)


def _patch_death_windows(
    out: Path, shard_names: list[str], flags: list[bool], valid: list[bool]
) -> None:
    cursor = 0
    for name in shard_names:
        path = out / name
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        count = int(payload["action"].shape[0])
        payload["death_window"] = np.asarray(flags[cursor:cursor + count], dtype=np.uint8)
        payload["death_window_valid"] = np.asarray(valid[cursor:cursor + count], dtype=np.uint8)
        cursor += count
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
        os.replace(temporary, path)
        sidecar = path.with_name(path.name + ".json")
        details = json.loads(sidecar.read_text(encoding="utf-8"))
        details["has_death_window"] = True
        details["death_window_horizon_decisions"] = 50
        details["bytes"] = path.stat().st_size
        sidecar.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    if cursor != len(flags):
        raise RuntimeError(f"death-window sample mapping mismatch: wrote {cursor}, expected {len(flags)}")


def collect_student_states(cfg: StudentRolloutConfig) -> dict:
    if cfg.behavior not in ("student", "teacher") or not 0 <= cfg.epsilon <= 1:
        raise ValueError("invalid behavior/epsilon")
    if cfg.episodes < 1 or cfg.max_steps < 1 or cfg.frameskip < 1 or cfg.shard_size < 1:
        raise ValueError("episodes/max_steps/frameskip/shard_size must be positive")
    out = Path(cfg.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"diagnostic round requires an empty directory: {out}")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    checkpoint = Path(cfg.checkpoint)
    model, extra = load_vision_checkpoint(checkpoint, map_location=device)
    if vision_arch(model) != VISION_ARCH_V3:
        raise ValueError("Phase 1 requires the frozen v3-st pixel policy")
    depth = resolve_vision_depth(model, 0)
    model.to(device).eval()
    if hasattr(model, "configure_temporal_policy"):
        model.configure_temporal_policy(ema=0.0, switch_penalty=0.0)
    if cfg.pilot.frameskip != cfg.frameskip:
        raise ValueError("teacher and environment frameskip disagree")
    if cfg.max_seconds < 0:
        raise ValueError("max_seconds must be >= 0")
    sha = _sha(checkpoint)
    run_id = uuid.uuid4().hex
    provenance = {
        "schema": SCHEMA, "training_approved": False,
        "source": "student-state-not-human" if cfg.behavior == "student" else "teacher-executed-not-student", "rollout_id": run_id,
        "behavior_policy": "v3-st-pixel-student" if cfg.behavior == "student" else "privileged-teacher",
        "epsilon": cfg.epsilon, "protocol_sha256": cfg.protocol_sha256,
        "collector_sha256": _sha(Path(__file__)), "input_evidence_enabled": cfg.record_input_evidence,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha,
        "teacher": "AvoidPilot", "teacher_source_sha256": _sha(Path(__file__).with_name("pilot.py")),
        "teacher_config": asdict(cfg.pilot),
        "difficulty": cfg.difficulty, "team": cfg.team,
        "practice_stage": cfg.practice_stage, "frameskip": cfg.frameskip,
        "max_steps": cfg.max_steps, "max_seconds": cfg.max_seconds,
        "seed": cfg.seed, "action_duration_decisions": 1,
    }
    env = (cfg.env_factory or Th08Env)(EnvConfig(
        difficulty=cfg.difficulty, team=cfg.team, practice_stage=cfg.practice_stage,
        frameskip=cfg.frameskip, max_steps=cfg.max_steps, reset_mode="relaunch",
        run_dir=cfg.run_dir, expose_raw_bullets=True, expose_bosses=True,
        record_input_evidence=cfg.record_input_evidence,
    ))
    writer = None
    sample_death_windows: list[bool] = []
    sample_death_valid: list[bool] = []
    report = {
        **provenance, "checkpoint_extra": extra, "episodes": [],
        "difficulty": cfg.difficulty, "team": cfg.team,
        "practice_stage": cfg.practice_stage, "max_steps": cfg.max_steps,
        "depth": depth, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "alignment": "reader stage/frame checked around decision render; unstable samples rejected",
        "deployment_temporal": {"ema": 0.0, "switch_penalty": 0.0},
    }
    try:
        if not all(hasattr(env, name) for name in ("reset", "render", "step", "close", "session")):
            raise TypeError("environment must implement reset/render/step/close and independent session reader")
        keep_awake(True)
        started_wall = time.time()
        for episode in range(1, cfg.episodes + 1):
            observation, info = env.reset(seed=cfg.seed + episode - 1)
            _frame_key(env)  # require an independent reader before creating output
            if writer is None:
                out.mkdir(parents=True, exist_ok=True)
                writer = VisionShardWriter(out, shard_size=cfg.shard_size, tag="student-state", provenance=provenance)
            teacher = AvoidPilot(PilotConfig(**asdict(cfg.pilot)))
            if hasattr(model, "reset_temporal_policy"):
                model.reset_temporal_policy()
            sizes = model.config.region_sizes
            stacks = {name: vision.FrameStack(depth=depth, shape=(*size, vision.VISION_CHANNELS))
                      for name, size in sizes.items()}
            first = _regions_from_screen(env.render(), sizes, None)
            last_render_digest = None
            last_render_key = None
            frozen_count = 0
            hwnd = getattr(getattr(env, "session", None), "hwnd", None)
            for name, stack in stacks.items():
                stack.reset()
                for _ in range(depth):
                    stack.push(first[name])
            rows = []
            skipped = 0
            terminated = False
            truncated = False
            next_info = info
            for decision in range(cfg.max_steps):
                if cfg.max_seconds > 0 and time.time() - started_wall >= cfg.max_seconds:
                    truncated = True
                    next_info = {**_state(info), "termination_reason": "max_seconds"}
                    break
                if hwnd and not window_visible(hwnd):
                    raise RuntimeError("game window became hidden or minimized during pixel rollout")
                screen, label_obs, previous, before, after, label_stamp = _capture_live_snapshot(
                    env, observation, info
                )
                native = vision.extract_regions(screen)
                regions = {name: vision.resize_area(native[name], *sizes[name]) for name in stacks}
                history = {name: stacks[name].push(regions[name]) for name in stacks}
                aligned = before == after == label_stamp == (int(previous["stage"]), int(previous["frame"]))
                started = time.perf_counter()
                action_started = time.perf_counter()
                with torch.no_grad():
                    action, _, _, _ = model.act(vision.regions_to_batch(history, device))
                student_latency_ms = (time.perf_counter() - action_started) * 1000.0
                executed = [int(v) for v in action[0].cpu().tolist()]
                teacher_pre = _teacher_query(teacher, label_obs, previous) if cfg.behavior == "teacher" else None
                if teacher_pre is not None:
                    executed = teacher_pre[0].astype(int).tolist()
                    if np.random.default_rng(cfg.seed + episode * 100000 + decision).random() < cfg.epsilon:
                        executed[0] = int(np.random.default_rng(cfg.seed + episode * 100000 + decision + 1).integers(9))
                stamp_before_step = _frame_key(env)
                step_started = time.perf_counter()
                next_obs, _, terminated, truncated, next_info = env.step(np.asarray(executed, dtype=np.int64))
                step_latency_ms = (time.perf_counter() - step_started) * 1000.0
                now = _state(next_info)
                teacher_started = time.perf_counter()
                teacher_action, teacher_clearance = teacher_pre if teacher_pre is not None else _teacher_query(teacher, label_obs, previous)
                teacher_latency_ms = (time.perf_counter() - teacher_started) * 1000.0
                # Commit teacher memory from the action actually executed, not its label.
                teacher._prev_move = executed[0]
                if executed[3]:
                    teacher._bomb_cooldown = teacher.config.bomb_cooldown
                elif teacher._bomb_cooldown > 0:
                    teacher._bomb_cooldown -= 1
                action_issue_stamp = stamp_before_step
                render_digest = hashlib.blake2b(native["playfield"].tobytes(), digest_size=16).hexdigest()
                same_pixels = last_render_digest is not None and render_digest == last_render_digest
                last_render_digest = render_digest
                pixel_stale = same_pixels and last_render_key is not None and last_render_key != (previous["stage"], previous["frame"])
                if pixel_stale:
                    frozen_count += 1
                else:
                    frozen_count = 0
                last_render_key = (previous["stage"], previous["frame"])
                frame_delta = int(now["frame"]) - int(previous["frame"]) if int(now["stage"]) == int(previous["stage"]) else 0
                row = {
                    "timing_contract": "v200-rollout-timing/1",
                    "timestamps": {
                        "capture_before": {"stage": int(before[0]), "frame": int(before[1])},
                        "capture_after": {"stage": int(after[0]), "frame": int(after[1])},
                        "action_issue": {"stage": int(action_issue_stamp[0]), "frame": int(action_issue_stamp[1])},
                        "action_effective": None,
                        "step_end": {"stage": int(now["stage"]), "frame": int(now["frame"])},
                    },
                    "timing": {
                        "capture_before": {"stage": int(before[0]), "frame": int(before[1])},
                        "capture_after": {"stage": int(after[0]), "frame": int(after[1])},
                        "action_issue": {"stage": int(action_issue_stamp[0]), "frame": int(action_issue_stamp[1])},
                        "action_effective": None,
                        "step_end": {"stage": int(now["stage"]), "frame": int(now["frame"])},
                    },
                    "input_mask_readback": None,
                    "action_hold_frames_valid": False,
                    "actual_hold_frames": None,
                    "raw_safety_snapshot": _raw_safety_snapshot(previous, before, after, label_stamp),
                    "action_duration_semantics": "snapshot-to-step-end frame delta, NOT measured input hold",
                    "rollout_id": run_id, "episode": episode, "decision": decision,
                    "stage": int(previous["stage"]), "frame": int(previous["frame"]),
                    "render_frame_before": before[1], "render_frame_after": after[1],
                    "action_issue_stage": action_issue_stamp[0], "action_issue_frame": action_issue_stamp[1],
                    "executed_action": executed, "teacher_action": teacher_action.astype(int).tolist(),
                    "teacher_clearance": teacher_clearance,
                    "action_duration_decisions": 1, "action_duration_game_frames": frame_delta,
                    "student_latency_ms": student_latency_ms,
                    "teacher_latency_ms": teacher_latency_ms,
                    "env_step_latency_ms": step_latency_ms,
                    "frame_delta": frame_delta,
                    "stage_delta": int(now["stage"]) - int(previous["stage"]),
                    "deaths_before": int(previous.get("deaths", 0)),
                    "deaths_after": int(now.get("deaths", 0)),
                    "aligned": aligned, "fresh_pixels": not pixel_stale,
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "termination_reason": next_info.get("termination_reason", ""),
                    "capture_to_step_seconds": time.perf_counter() - started,
                }
                row["input_timing"] = next_info.get("input_timing")
                evidence = row["input_timing"]
                if evidence is not None:
                    observed = evidence.get("effective_first_observed")
                    row["action_hold_frames_valid"] = bool(evidence.get("timing_valid"))
                    row["actual_hold_frames"] = (evidence["end"]["frame"] - observed["frame"]) if row["action_hold_frames_valid"] else None
                    row["input_observed_semantics"] = evidence["semantics"]
                    row["timestamps"]["action_effective"] = observed
                    row["timestamps"]["step_end"] = evidence["end"]
                if frozen_count >= 5:
                    row["termination_reason"] = "frozen-window-capture"
                    row["terminated"] = True
                    terminated = True
                if aligned and row["fresh_pixels"]:
                    blobs = {name: vision.encode_png(image) for name, image in native.items()}
                    meta = {name: previous.get(name, 0) for name in META_FIELDS}
                    if not all(np.isfinite(float(value)) for value in meta.values()):
                        raise ValueError("student-state metadata must be finite")
                    meta.update(episode=episode, decision=decision, frame_delta=frame_delta,
                                stage_delta=row["stage_delta"], executed_action=executed,
                                death_window=False, death_window_valid=False)
                    writer.add(blobs, _label_observation(label_obs), tuple(teacher_action), meta,
                               native_hw={name: image.shape[:2] for name, image in native.items()})
                    row["sample_index"] = writer.total - 1
                else:
                    skipped += 1
                    row["sample_index"] = None


                rows.append(row)
                observation, info = next_obs, next_info
                if terminated or truncated:
                    break
            # A death can occur after env.step returned but before the next render.
            # Attribute that event to the intervening decision, not to an
            # already-recorded label at the next state.
            for i in range(len(rows) - 1):
                if rows[i + 1]["deaths_before"] > rows[i]["deaths_after"]:
                    rows[i]["deaths_after"] = rows[i + 1]["deaths_before"]
                    rows[i]["death_evidence"] = "next-snapshot"
            # The future window is positive on a known death, negative only when
            # the full horizon was observed, and unknown near a truncation boundary.
            for i, row in enumerate(rows):
                future = rows[i:min(len(rows), i + 51)]
                positive = any(item["deaths_after"] > item["deaths_before"] for item in future)
                known = positive or i + 50 < len(rows)
                row["death_window"] = positive
                row["death_window_valid"] = known
                if row.get("sample_index") is not None:
                    sample_death_windows.append(positive)
                    sample_death_valid.append(known)
                    row["death_window_recorded"] = True
                else:
                    row["death_window_recorded"] = False
            trace = out / f"episode-{episode:03d}-decisions.jsonl"
            trace.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            terminal = rows[-1] if rows else {}
            report["episodes"].append({
                "episode": episode, "decisions": len(rows), "samples": len(rows) - skipped,
                "alignment_skipped": skipped, "trace": trace.name, "trace_sha256": _sha(trace),
                "frames": next_info.get("episode", {}).get("frames"),
                "deaths": next_info.get("episode", {}).get("deaths"),
                "termination_reason": terminal.get("termination_reason", "max_steps"),
                "hit_step_cap": bool(terminal.get("truncated")),
            })
    finally:
        try:
            if writer is not None:
                summary = writer.close()
                report["samples"] = summary["frames"]
                _patch_death_windows(out, summary["shards"], sample_death_windows, sample_death_valid)
                total_decisions = sum(ep["decisions"] for ep in report["episodes"])
                alignment_ratio = summary["frames"] / max(1, total_decisions)
                report["alignment_ratio"] = round(alignment_ratio, 6)
                report["phase1_collection_status"] = (
                    "collection-ready-for-distribution-audit"
                    if summary["frames"] >= 64 and alignment_ratio >= 0.8
                    and not any(ep["termination_reason"] == "frozen-window-capture" for ep in report["episodes"])
                    else "failed-insufficient-or-stale-samples"
                )
                report["death_window_horizon_decisions"] = 50
                report["death_window_unknown_count"] = int(sum(not value for value in sample_death_valid))
                report["shards"] = [{"name": name, "sha256": _sha(out / name)} for name in summary["shards"]]
                report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                (out / "student-rollout-report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
                )
        finally:
            keep_awake(False)
            env.close()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 diagnostic v3-st pixel student rollout (not training)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="new empty round directory")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument(
        "--behavior",
        default="student",
        choices=["student", "teacher"],
        help="who acts; the teacher is still the labeller either way. 'teacher' collects "
        "demonstrations from survivable states (step 2 warm start).",
    )
    args = parser.parse_args(argv)
    cfg = StudentRolloutConfig(
        checkpoint=Path(args.checkpoint), out_dir=Path(args.out), episodes=args.episodes,
        max_steps=args.max_steps, frameskip=args.frameskip, seed=args.seed, device=args.device,
        behavior=args.behavior,
        max_seconds=args.max_seconds,
        pilot=PilotConfig(frameskip=args.frameskip, switch_penalty=0.0),
    )
    report = collect_student_states(cfg)
    print(json.dumps({"samples": report["samples"], "episodes": report["episodes"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
