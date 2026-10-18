"""Collect teacher-labelled frames on the live game (DAgger-style distillation data).

Why a separate collector
------------------------
Replay capture (``capture_dataset``) gives "what a human pressed on their own
trajectory".  It cannot give "what a human *would* press in the state the student
actually reaches", which is exactly the distribution where behaviour cloning
fails: v1.0.2's cloned policy died every seven seconds while its validation
accuracy looked fine.

This collector runs the game with the scripted avoidance teacher
(:mod:`touhou_rl.pilot`) in the loop and writes, for every decision, the
observation the *network* would see together with the action the *teacher*
chose.  Two behaviour policies are supported:

``--behaviour pilot`` (default)
    The teacher acts; a fraction of decisions are replaced by a random movement
    (``--epsilon``) so the recorded states include the states that no sane policy
    would visit.  Every one of those frames is still labelled by the teacher,
    which is what makes it DAgger rather than plain imitation.

``--behaviour checkpoint --checkpoint <path>``
    The student acts and the teacher labels.  This is full DAgger: the collected
    states are exactly the student's own failure modes.  It is the round that
    fixes the "off-trajectory state" problem that pure imitation cannot.

Output shards use the exact ``touhou_rl.dataset`` format, so training is the
unchanged ``python -m touhou_rl.bc``; the only difference from a replay dataset is
the provenance sidecar (``label_source: pilot``).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import spec
from .dataset import ShardWriter
from .env import EnvConfig, Th08Env
from .models import load_checkpoint, observation_batch
from .obs import Observation
from .pilot import AvoidPilot, PilotConfig

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CollectConfig:
    out_dir: Path
    episodes: int = 4
    max_steps: int = 20000
    frameskip: int = 2
    difficulty: int = spec.LUNATIC
    team: int = 0
    practice_stage: int | None = None
    tag: str = "pilot-dagger"
    shard_size: int = 2048
    behaviour: str = "pilot"  # "pilot" or "checkpoint"
    checkpoint: Path | None = None
    epsilon: float = 0.15  # probe-executed random movement (pilot behaviour only)
    device: str = "cuda"
    seed: int = 11
    max_minutes: float = 0.0  # 0 = no wall-clock budget
    run_dir: Path = ROOT / "third_party/th08/build/rl/dagger-session"
    pilot: PilotConfig = field(default_factory=PilotConfig)
    overwrite: bool = False


def _observation_from_dict(observation: dict) -> Observation:
    return Observation(
        self_feat=np.asarray(observation["self_feat"], dtype=np.float32),
        bullets=np.asarray(observation["bullets"], dtype=np.float16),
        danger_map=np.asarray(observation["danger"], dtype=np.uint8),
    )


def _meta(info: dict) -> dict:
    return {
        "frame": info.get("frame", 0),
        "stage": info.get("stage", 0),
        "score": info.get("score", 0),
        "graze": info.get("graze", 0),
        "lives": info.get("lives", 0.0),
        "bombs": info.get("bombs", 0.0),
        "power": info.get("power", 0.0),
        "deaths": info.get("deaths", 0),
    }


def collect(cfg: CollectConfig) -> dict:
    cfg.out_dir = Path(cfg.out_dir)
    if cfg.out_dir.exists() and any(cfg.out_dir.glob("*.npz")) and not cfg.overwrite:
        raise FileExistsError(f"{cfg.out_dir} already holds shards; pass --overwrite")
    rng = np.random.default_rng(cfg.seed)

    pilot_config = cfg.pilot
    pilot_config.frameskip = cfg.frameskip
    pilot = AvoidPilot(pilot_config)  # __post_init__ builds the frameskip table
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    student = None
    if cfg.behaviour == "checkpoint":
        if cfg.checkpoint is None:
            raise ValueError("--behaviour checkpoint needs --checkpoint")
        student, _extra = load_checkpoint(cfg.checkpoint, map_location=device)
        student.to(device).eval()

    env = Th08Env(
        EnvConfig(
            difficulty=cfg.difficulty,
            team=cfg.team,
            practice_stage=cfg.practice_stage,
            frameskip=cfg.frameskip,
            max_steps=cfg.max_steps,
            reset_mode="relaunch",
            run_dir=cfg.run_dir,
            expose_raw_bullets=True,
            expose_bosses=True,
        )
    )
    writer = ShardWriter(
        out_dir=cfg.out_dir,
        shard_size=cfg.shard_size,
        tag=cfg.tag,
        provenance={
            "label_source": "pilot",
            "behaviour": cfg.behaviour,
            "checkpoint": str(cfg.checkpoint or ""),
            "epsilon": cfg.epsilon,
            "difficulty": cfg.difficulty,
            "team": cfg.team,
            "practice_stage": cfg.practice_stage if cfg.practice_stage is not None else -1,
            "frameskip": cfg.frameskip,
        },
        allow_append=cfg.overwrite,
    )
    started = time.time()
    report = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "behaviour": cfg.behaviour,
        "checkpoint": str(cfg.checkpoint or ""),
        "epsilon": cfg.epsilon,
        "difficulty": cfg.difficulty,
        "team": cfg.team,
        "practice_stage": cfg.practice_stage,
        "frameskip": cfg.frameskip,
        "episodes": [],
    }
    try:
        for episode in range(1, cfg.episodes + 1):
            obs, info = env.reset()
            # ``reset`` nests the state under info["state"]; flatten it so the
            # first decision of an episode already sees bombs/lives/stage.
            info = {**info.get("state", {}), **info}
            steps = 0
            deaths_at_start = info.get("state", {}).get("deaths", 0)
            while True:
                bombs = int(round(float(info.get("bombs", 0) or 0)))
                teacher = pilot.act(
                    obs,
                    info.get("bullets_raw"),
                    bombs=bombs,
                    bosses=info.get("bosses"),
                )
                teacher_action = tuple(int(v) for v in teacher)

                if student is None:
                    if rng.random() < cfg.epsilon:
                        executed = np.array(
                            [
                                int(rng.integers(0, 9)),
                                1,
                                int(teacher[2]) if rng.random() > 0.2 else int(rng.integers(0, 2)),
                                int(teacher[3]),
                            ],
                            dtype=np.int64,
                        )
                    else:
                        executed = teacher.copy()
                else:
                    batch = observation_batch(obs, device)
                    with torch.no_grad():
                        action, _logp, _value, _aux = student.act(batch, deterministic=False)
                    executed = action[0].cpu().numpy().astype(np.int64)
                    # Keep the teacher's "did I just switch direction" state in
                    # sync with what the player is actually being told to do.
                    pilot._prev_move = int(executed[0])

                writer.add(_observation_from_dict(obs), teacher_action, _meta(info))
                obs, _reward, terminated, truncated, info = env.step(executed)
                steps += 1
                if terminated or truncated:
                    break
                if cfg.max_minutes and (time.time() - started) > cfg.max_minutes * 60.0:
                    truncated = True
                    break
            row = {
                "episode": episode,
                "steps": steps,
                "deaths": int(info.get("deaths", 0)) - int(deaths_at_start),
                "stage": int(info.get("stage", 0)),
                "frames": int(info.get("frame", 0)),
                "termination_reason": info.get("termination_reason", "budget" if truncated else ""),
            }
            report["episodes"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if cfg.max_minutes and (time.time() - started) > cfg.max_minutes * 60.0:
                break
    finally:
        summary = writer.close()
        env.close()
        report["frames"] = summary["frames"]
        report["shards"] = len(summary["shards"])
        report["out_dir"] = str(cfg.out_dir)
        report["elapsed_seconds"] = round(time.time() - started, 1)
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        (cfg.out_dir / "collect-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect teacher-labelled frames for distillation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--max-minutes", type=float, default=0.0)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--difficulty", default="lunatic", choices=["easy", "normal", "hard", "lunatic"])
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--practice-stage", type=int, default=-1)
    parser.add_argument("--tag", default="pilot-dagger")
    parser.add_argument("--behaviour", default="pilot", choices=["pilot", "checkpoint"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    difficulty = {"easy": spec.EASY, "normal": spec.NORMAL, "hard": spec.HARD, "lunatic": spec.LUNATIC}[
        args.difficulty
    ]
    cfg = CollectConfig(
        out_dir=Path(args.out),
        episodes=args.episodes,
        max_steps=args.max_steps,
        frameskip=args.frameskip,
        difficulty=difficulty,
        team=args.team,
        practice_stage=None if args.practice_stage < 0 else args.practice_stage,
        tag=args.tag,
        behaviour=args.behaviour,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        epsilon=args.epsilon,
        device=args.device,
        max_minutes=args.max_minutes,
        overwrite=args.overwrite,
    )
    report = collect(cfg)
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
