"""Play TH08 with a trained model -- the user-facing entry point.

The script does everything the game normally needs a human for: it launches the
rebuilt TH08, walks the title menu (Start -> difficulty -> team), plays with the
checkpoint, and keeps restarting after a game over until the time budget or the
episode count is exhausted.  No menu interaction is required from the user, which
is the v1.0.2 requirement "the model only needs to know how to play".

Examples::

    # Lunatic, Border Team, one continuous session of up to 20 minutes
    python -m touhou_rl.play --checkpoint runs/bc/best.pt --team 0 --minutes 20

    # One stage in practice mode (fast iteration)
    python -m touhou_rl.play --checkpoint runs/bc/best.pt --practice-stage 0 --episodes 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from . import spec
from .env import EnvConfig, Th08Env
from .models import load_checkpoint, observation_batch

ROOT = Path(__file__).resolve().parents[1]

DIFFICULTIES = {"easy": spec.EASY, "normal": spec.NORMAL, "hard": spec.HARD, "lunatic": spec.LUNATIC}


def play(
    checkpoint: Path,
    *,
    minutes: float = 20.0,
    episodes: int = 0,
    difficulty: int = spec.LUNATIC,
    team: int = 0,
    practice_stage: int | None = None,
    frameskip: int = 2,
    max_steps: int = 15000,
    deterministic: bool = False,
    device: str = "cuda",
    out_dir: Path | None = None,
) -> dict:
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, extra = load_checkpoint(checkpoint, map_location=torch_device)
    model.to(torch_device).eval()
    run_dir = ROOT / "third_party/th08/build/rl/play-session"
    env = Th08Env(
        EnvConfig(
            difficulty=difficulty,
            team=team,
            practice_stage=practice_stage,
            frameskip=frameskip,
            max_steps=max_steps,
            reset_mode="relaunch",
            run_dir=run_dir,
        )
    )
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_extra": extra,
        "difficulty": difficulty,
        "team": team,
        "practice_stage": practice_stage,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes": [],
    }
    deadline = time.time() + minutes * 60.0
    try:
        episode = 0
        while time.time() < deadline and (episodes <= 0 or episode < episodes):
            episode += 1
            obs, _info = env.reset()
            started = time.perf_counter()
            steps = 0
            while True:
                batch = observation_batch(obs, torch_device)
                with torch.no_grad():
                    action, _logp, _value, _ = model.act(batch, deterministic=deterministic)
                obs, _reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                steps += 1
                if terminated or truncated or time.time() >= deadline:
                    break
            row = {
                "episode": episode,
                "steps": steps,
                "wall_seconds": round(time.perf_counter() - started, 1),
                **info.get("episode", {}),
            }
            report["episodes"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        report["interrupted"] = True
    finally:
        env.close()
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "play-report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Let a trained model play TH08")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--difficulty", default="lunatic", choices=sorted(DIFFICULTIES))
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--practice-stage", type=int, default=-1)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = play(
        Path(args.checkpoint),
        minutes=args.minutes,
        episodes=args.episodes,
        difficulty=DIFFICULTIES[args.difficulty],
        team=args.team,
        practice_stage=None if args.practice_stage < 0 else args.practice_stage,
        frameskip=args.frameskip,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        device=args.device,
        out_dir=Path(args.out) if args.out else None,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
