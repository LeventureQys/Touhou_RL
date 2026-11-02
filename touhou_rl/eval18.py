"""Closed-loop evaluation of the step-2 privileged entity policy.

Why a separate entry point
--------------------------
``evaluate.py`` loads the v1/v2/v3 *pixel* checkpoints or the four-head
``DanmakuPolicy``.  The step-2 policy reads the whole bullet pool from game memory
and emits an 18-way candidate, so it needs its own loop -- but the report has to
stay byte-compatible with ``tools/rl/compare_evals.py`` (same row keys), otherwise
the ladder cannot put the new model on the same table.

Protocol (identical to the ladder)
----------------------------------
Lunatic, team 0, ``frameskip=2``, Stage 1 start, one process per episode,
``termination_reason`` straight from ``env.step``, ``hit_step_cap`` left to
``compare_evals`` (it counts ``max_steps``).

    python -m touhou_rl.eval18 --checkpoint step2-privileged/bc/policy.pt \
        --episodes 20 --max-steps 12000 --decision-trace \
        --out third_party/th08/build/rl/eval/v311-step2-bc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import entity_obs, entity_policy, spec
from .env import EnvConfig, Th08Env
from .stage0_diagnostics import summarize_actions

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "v311-eval18-report/1"


@dataclass
class Eval18Config:
    checkpoint: Path
    out_dir: Path
    episodes: int = 3
    difficulty: int = spec.LUNATIC
    team: int = 0
    practice_stage: int | None = None
    frameskip: int = 2
    max_steps: int = 12000
    device: str = "cuda"
    reset_mode: str = "relaunch"
    run_dir: Path = ROOT / "third_party/th08/build/rl/eval-session"
    decision_trace: bool = False
    env_factory: Callable[[EnvConfig], object] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observation(observation: dict, info: dict) -> dict[str, np.ndarray]:
    """Entity observation for one decision, tolerating the reset() info shape."""
    if info.get("bullets_raw") is not None:
        return entity_obs.from_env(info, observation)
    # reset() has no bullets yet; the player position is already in self_feat.
    self_feat = np.asarray(observation["self_feat"], dtype=np.float32).reshape(-1)
    player_x = spec.PLAYFIELD_X + float(self_feat[0]) * spec.PLAYFIELD_W
    player_y = spec.PLAYFIELD_Y + float(self_feat[1]) * spec.PLAYFIELD_H
    empty = np.zeros((0, 2), dtype=np.float32)
    return entity_obs.build(player_x, player_y, empty, empty, np.zeros((0,), dtype=np.float32))


def run_evaluation(cfg: Eval18Config) -> dict:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model, extra = entity_policy.load_checkpoint(cfg.checkpoint, map_location=device)
    model.to(device).eval()
    env = (cfg.env_factory or Th08Env)(
        EnvConfig(
            difficulty=cfg.difficulty,
            team=cfg.team,
            practice_stage=cfg.practice_stage,
            frameskip=cfg.frameskip,
            max_steps=cfg.max_steps,
            reset_mode=cfg.reset_mode,
            run_dir=cfg.run_dir,
            expose_raw_bullets=True,
            expose_bosses=False,
        )
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": SCHEMA,
        "policy": "privileged-entity",
        "checkpoint": str(cfg.checkpoint),
        "checkpoint_sha256": _sha256(Path(cfg.checkpoint)),
        "checkpoint_extra": extra,
        "candidate_order": entity_policy.CANDIDATE_ORDER,
        "difficulty": cfg.difficulty,
        "team": cfg.team,
        "practice_stage": cfg.practice_stage,
        "frameskip": cfg.frameskip,
        "deterministic": True,
        "device": str(device),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes": [],
    }
    try:
        for episode in range(1, cfg.episodes + 1):
            observation, info = env.reset()
            started = time.perf_counter()
            steps = 0
            episode_info = {}
            actions: list[list[int]] = []
            active_counts: list[int] = []
            while True:
                observation_batch = entity_obs.batch([_observation(observation, info)])
                with torch.no_grad():
                    candidate, _logp, _value, _entropy = model.act(
                        entity_policy.to_tensors(observation_batch, device), deterministic=True
                    )
                decision = entity_obs.action_of_candidate(int(candidate[0]))
                actions.append(list(decision))
                raw = info.get("bullets_raw")
                if raw is not None:
                    active_counts.append(int(len(raw["position"])))
                observation, _reward, terminated, truncated, info = env.step(np.asarray(decision, dtype=np.int64))
                steps += 1
                if terminated or truncated:
                    episode_info = info.get("episode", {})
                    break
            elapsed = time.perf_counter() - started
            row = {
                "episode": episode,
                "steps": steps,
                "frames_reported": int(episode_info.get("frames", 0)),
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
                "active_bullets": {
                    "max": int(max(active_counts)) if active_counts else 0,
                    "p95": int(np.percentile(active_counts, 95)) if active_counts else 0,
                    "over_cap": int(sum(1 for value in active_counts if value > entity_obs.K_BULLETS)),
                    "decisions": len(active_counts),
                },
            }
            if cfg.decision_trace:
                row["decision_trace"] = {"schema": "v311-decision-trace/1", "actions": actions}
                row["action_diagnostics"] = summarize_actions(actions)
            report["episodes"].append(row)
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
            "deaths_mean": round(float(np.mean([row["deaths"] for row in report["episodes"]])), 2)
            if report["episodes"]
            else 0.0,
        }
        (cfg.out_dir / "eval-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the step-2 privileged entity policy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--difficulty", default="lunatic", choices=["easy", "normal", "hard", "lunatic"])
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--practice-stage", type=int, default=-1)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--decision-trace", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    difficulty = {"easy": spec.EASY, "normal": spec.NORMAL, "hard": spec.HARD, "lunatic": spec.LUNATIC}[
        args.difficulty
    ]
    report = run_evaluation(
        Eval18Config(
            checkpoint=Path(args.checkpoint),
            out_dir=Path(args.out),
            episodes=args.episodes,
            difficulty=difficulty,
            team=args.team,
            practice_stage=None if args.practice_stage < 0 else args.practice_stage,
            frameskip=args.frameskip,
            max_steps=args.max_steps,
            device=args.device,
            decision_trace=args.decision_trace,
        )
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
