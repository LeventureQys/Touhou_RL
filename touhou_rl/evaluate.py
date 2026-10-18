"""Evaluate a checkpoint on the live game and write machine-readable evidence.

Each episode is a fresh menu start at the requested difficulty/team.  The report
records survival frames (the primary metric), stages reached, deaths, score and
graze, plus a screenshot and the full per-episode state trail so a reviewer can
reproduce the numbers without trusting prose.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import spec, vision
from .env import EnvConfig, Th08Env
from .models import load_checkpoint, observation_batch
from .pilot import AvoidPilot, PilotConfig
from .vmodels import load_vision_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EvalConfig:
    checkpoint: Path
    out_dir: Path
    episodes: int = 3
    difficulty: int = spec.LUNATIC
    team: int = 0
    practice_stage: int | None = None
    frameskip: int = 2
    max_steps: int = 12000
    deterministic: bool = False
    device: str = "cuda"
    screenshot: bool = True
    random_policy: bool = False
    reset_mode: str = "relaunch"
    run_dir: Path = ROOT / "third_party/th08/build/rl/eval-session"
    # Scripted avoidance teacher (no checkpoint needed); see touhou_rl/pilot.py.
    pilot: bool = False
    pilot_config: PilotConfig | None = None
    # Vision policy: the deployment path - frames only, no privileged reads.
    vision_checkpoint: Path | None = None
    vision_depth: int = 3
    vision_box: vision.PlayfieldBox | None = None


def build_pilot(cfg: EvalConfig) -> AvoidPilot | None:
    """Construct the scripted teacher, honouring the frameskip of the environment."""
    if not cfg.pilot:
        return None
    base = cfg.pilot_config or PilotConfig()
    base.frameskip = cfg.frameskip
    return AvoidPilot(base)


def _regions_from_screen(
    screen, region_sizes: dict[str, tuple[int, int]], box=None
) -> dict:
    """Native crops of every region, resized to the sizes the model expects."""
    native = vision.extract_regions(screen)
    out = {}
    for name, (height, width) in region_sizes.items():
        crop = native.get(name)
        if crop is None:  # region missing on this screen: feed zeros rather than fail
            out[name] = np.zeros((height, width, vision.VISION_CHANNELS), dtype=np.uint8)
        else:
            out[name] = vision.resize_area(crop, height, width)
    return out


def run_evaluation(cfg: EvalConfig) -> dict:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    pilot = build_pilot(cfg)
    vision_model = None
    vision_extra: dict = {}
    if cfg.random_policy:
        model = None
        extra: dict = {"random_policy": True}
    elif cfg.vision_checkpoint is not None:
        model = None
        vision_model, vision_extra = load_vision_checkpoint(cfg.vision_checkpoint, map_location=device)
        vision_model.to(device).eval()
        extra = {"vision_checkpoint": True, **vision_extra}
    elif pilot is not None:
        model = None
        extra = {"pilot_config": pilot.config.__dict__ if hasattr(pilot.config, "__dict__") else {
            name: getattr(pilot.config, name) for name in pilot.config.__dataclass_fields__
        }}
    else:
        model, extra = load_checkpoint(cfg.checkpoint, map_location=device)
        model.to(device).eval()
    env = Th08Env(
        EnvConfig(
            difficulty=cfg.difficulty,
            team=cfg.team,
            practice_stage=cfg.practice_stage,
            frameskip=cfg.frameskip,
            max_steps=cfg.max_steps,
            reset_mode=cfg.reset_mode,
            run_dir=cfg.run_dir,
            expose_raw_bullets=pilot is not None,
            expose_bosses=pilot is not None,
        )
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.random_policy:
        policy = "random"
    elif vision_model is not None:
        policy = "vision"
    elif pilot is not None:
        policy = "pilot"
    elif model is not None and model.__class__.__name__ == "ScorePolicy":
        policy = "scorer"
    else:
        policy = "network"
    report = {
        "checkpoint": str(cfg.checkpoint),
        "checkpoint_extra": extra,
        "policy": policy,
        "difficulty": cfg.difficulty,
        "team": cfg.team,
        "practice_stage": cfg.practice_stage,
        "frameskip": cfg.frameskip,
        "deterministic": cfg.deterministic,
        "device": str(device),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes": [],
    }
    try:
        for episode in range(1, cfg.episodes + 1):
            obs, info = env.reset()
            started = time.perf_counter()
            steps = 0
            episode_info = {}
            clearances: list[float] = []
            frame_stack = None
            if vision_model is not None:
                # Deployment path: the policy sees only pixels, as per-region
                # streams (playfield + HUD blocks).  Each stream is stacked so the
                # model gets motion without a recurrent state.
                region_sizes = vision_model.config.region_sizes
                frame_stack = {
                    name: vision.FrameStack(
                        depth=cfg.vision_depth, shape=(*size, vision.VISION_CHANNELS)
                    )
                    for name, size in region_sizes.items()
                }
                first = _regions_from_screen(env.render(), region_sizes, cfg.vision_box)
                for name, stack in frame_stack.items():
                    stack.reset()
                    for _ in range(cfg.vision_depth):
                        stack.push(first[name])
            # ``reset`` returns {"state": {...}} whereas ``step`` returns a flat
            # dict, so the bomb counter has to be read from both shapes.
            reset_state = info.get("state", {}) if isinstance(info, dict) else {}
            bombs_start = int(round(float(reset_state.get("bombs", info.get("bombs", 0)) or 0)))
            while True:
                if vision_model is not None:
                    regions = _regions_from_screen(env.render(), vision_model.config.region_sizes, cfg.vision_box)
                    stacks = {name: frame_stack[name].push(regions[name]) for name in frame_stack}
                    with torch.no_grad():
                        action, _logp, value, _extra = vision_model.act(vision.regions_to_batch(stacks, device))
                    obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                elif model is None:
                    if pilot is None:
                        action = env.action_space.sample()
                    else:
                        action = pilot.act(
                            obs,
                            info.get("bullets_raw"),
                            bombs=int(round(float(info.get("bombs", 0)))),
                            bosses=info.get("bosses"),
                        )
                        clearances.append(pilot.debug.best_clearance)
                    obs, reward, terminated, truncated, info = env.step(action)
                else:
                    batch = observation_batch(obs, device)
                    with torch.no_grad():
                        action, logp, value, _ = model.act(batch, deterministic=cfg.deterministic)
                    obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                steps += 1
                if cfg.screenshot and steps == 60:
                    try:
                        from PIL import Image

                        Image.fromarray(env.render()).save(cfg.out_dir / f"episode-{episode:02d}-mid.png")
                    except Exception:
                        pass
                if terminated or truncated:
                    episode_info = info.get("episode", {})
                    break
            frames = int(episode_info.get("frames", 0))
            elapsed = time.perf_counter() - started
            row = {
                "episode": episode,
                "steps": steps,
                "frames_reported": frames,
                "stages_reached": int(info.get("stages_reached", 0)),
                "deaths": int(episode_info.get("deaths", 0)),
                "lives_left": float(episode_info.get("lives", 0.0)),
                "score": int(episode_info.get("score", 0)),
                "graze": int(episode_info.get("graze", 0)),
                "episode_reward": round(float(episode_info.get("episode_reward", 0.0)), 3),
                "terminated": bool(episode_info.get("terminated", False)),
                "truncated": bool(episode_info.get("truncated", False)),
                "termination_reason": episode_info.get("termination_reason", ""),
                "wall_seconds": round(elapsed, 1),
                "decisions_per_s": round(steps / max(1e-6, elapsed), 2),
                "last_state": {
                    "stage": info.get("stage"),
                    "frame": info.get("frame"),
                    "bullets": info.get("bullets"),
                },
            }
            if pilot is not None:
                row["bombs_used"] = bombs_start - int(round(float(info.get("bombs", 0))))
                if clearances:
                    row["pilot_clearance"] = {
                        "mean": round(float(np.mean(clearances)), 2),
                        "min": round(float(np.min(clearances)), 2),
                    }
            report["episodes"].append(row)
            if cfg.screenshot:
                try:
                    from PIL import Image

                    frame = env.render()
                    path = cfg.out_dir / f"episode-{episode:02d}.png"
                    Image.fromarray(frame).save(path)
                    row["screenshot"] = str(path)
                except Exception as exc:  # diagnostics only
                    row["screenshot_error"] = repr(exc)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        env.close()
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        frames = [row["frames_reported"] for row in report["episodes"]]
        report["summary"] = {
            "episodes": len(frames),
            "frames_mean": round(float(np.mean(frames)), 1) if frames else 0.0,
            "frames_max": int(max(frames)) if frames else 0,
            "stages_max": max((row["stages_reached"] for row in report["episodes"]), default=0),
            "deaths_mean": round(float(np.mean([r["deaths"] for r in report["episodes"]])), 2)
            if report["episodes"]
            else 0.0,
        }
        (cfg.out_dir / "eval-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a TH08 policy on the live game")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--difficulty", default="lunatic", choices=["easy", "normal", "hard", "lunatic"])
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--practice-stage", type=int, default=-1)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--random-policy", action="store_true", help="uniform-random baseline")
    parser.add_argument(
        "--vision-checkpoint",
        default="",
        help="vision scorer checkpoint: the policy plays from a cropped+resized frame stack only",
    )
    parser.add_argument("--vision-depth", type=int, default=3, help="frame-stack depth for the vision policy")
    parser.add_argument(
        "--pilot", action="store_true", help="scripted avoidance teacher (no checkpoint needed)"
    )
    parser.add_argument("--pilot-horizon", type=int, default=0, help="planning steps (0 = default)")
    parser.add_argument("--pilot-beam", type=int, default=0, help="beam width (0 = default)")
    parser.add_argument("--pilot-speed", type=float, default=0.0, help="normal axis speed px/frame")
    parser.add_argument("--pilot-speed-diag", type=float, default=0.0, help="normal diagonal px/frame")
    parser.add_argument("--pilot-focus-speed", type=float, default=0.0, help="focused axis speed px/frame")
    parser.add_argument(
        "--pilot-focus-speed-diag", type=float, default=0.0, help="focused diagonal px/frame"
    )
    parser.add_argument("--pilot-player-half", type=float, default=0.0, help="player hurtbox half size px")
    parser.add_argument(
        "--pilot-max-bullets",
        type=int,
        default=0,
        help="cap the bullets the pilot may plan with (0 = default); set 64 to emulate the network observation",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not args.random_policy and not args.pilot and not args.checkpoint and not args.vision_checkpoint:
        parser.error("one of --checkpoint / --random-policy / --pilot / --vision-checkpoint is required")
    difficulty = {"easy": spec.EASY, "normal": spec.NORMAL, "hard": spec.HARD, "lunatic": spec.LUNATIC}[
        args.difficulty
    ]
    pilot_config = None
    if args.pilot:
        overrides = {
            "horizon_steps": args.pilot_horizon,
            "beam": args.pilot_beam,
            "speed": args.pilot_speed,
            "speed_diag": args.pilot_speed_diag,
            "focus_speed": args.pilot_focus_speed,
            "focus_speed_diag": args.pilot_focus_speed_diag,
            "player_half": args.pilot_player_half,
            "max_bullets": args.pilot_max_bullets,
        }
        pilot_config = PilotConfig(
            **{name: value for name, value in overrides.items() if value not in (0, 0.0)}
        )
    cfg = EvalConfig(
        checkpoint=Path(args.checkpoint) if args.checkpoint else Path("."),
        out_dir=Path(args.out),
        episodes=args.episodes,
        difficulty=difficulty,
        team=args.team,
        practice_stage=None if args.practice_stage < 0 else args.practice_stage,
        frameskip=args.frameskip,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        device=args.device,
        random_policy=args.random_policy,
        pilot=args.pilot,
        pilot_config=pilot_config,
        vision_checkpoint=Path(args.vision_checkpoint) if args.vision_checkpoint else None,
        vision_depth=args.vision_depth,
    )
    report = run_evaluation(cfg)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
