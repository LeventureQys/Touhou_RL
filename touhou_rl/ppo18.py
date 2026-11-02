"""PPO fine-tuning of the step-2 privileged policy, anchored to its BC warm start.

Why this file exists
--------------------
``touhou_rl/ppo.py`` is the four-head ``DanmakuPolicy`` trainer from v1.0.2: it is
still the only runnable PPO in the repo, but it speaks the 72-way factorised action
space and the top-64 observation.  Step 2 needs the opposite: the 18-way candidate
head over the whole bullet pool, a KL anchor to the BC weights, and an evaluation
hook that reuses the *same* game process (the retail mutex forbids a second
instance).

Contract
--------
* env: ``Th08Env(EnvConfig(expose_raw_bullets=True, frameskip=2, reset_mode="relaunch"))``
* observation: ``entity_obs.from_env(info, obs)`` at every decision
* action: candidate ``focus*9+move`` -> ``(move, 1, focus, 0)``; bombs excluded
* reward: the environment's ``RewardEngine`` scalar, unchanged
* KL anchor: divergence from a frozen copy of the BC checkpoint, added with
  ``kl_coef`` (0 disables).  This is the "do not unlearn the prior" term from VPT.
* evaluation: every ``eval_every_episodes`` episodes, ``eval_episodes`` deterministic
  episodes on the same env, reported with the ladder's metric (``frames``)

The loop is deliberately single-environment: one training episode at a time, GAE over
the collected trajectory, then ``epochs`` passes of clipped PPO over it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from . import entity_obs, entity_policy, options, spec
from .env import EnvConfig, Th08Env
from .reward import RewardWeights

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PPO18Config:
    out_dir: Path
    init_checkpoint: Path | None = None
    anchor_checkpoint: Path | None = None
    total_steps: int = 400_000
    rollout_steps: int = 1024
    epochs: int = 4
    minibatch: int = 256
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    kl_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    episode_max_steps: int = 12_000
    eval_every_episodes: int = 20
    eval_episodes: int = 20
    save_every: int = 50_000
    device: str = "cuda"
    seed: int = 11
    #: Survive weight override.  The environment default is 0.01, which is
    #: ~1/100th of the critic's own error (value_loss ~1.0) over the GAE horizon,
    #: so the per-frame survival signal never reaches the policy (step-2 finding).
    survive_weight: float | None = None
    #: Potential-based shaping on the escape-option count (PBRS, state-only potential).
    #: 0 disables; ``pbrs_horizon`` is the lookahead in game frames.
    pbrs_weight: float = 0.0
    pbrs_horizon: int = options.HORIZON_FRAMES
    run_dir: Path = ROOT / "third_party/th08/build/rl/ppo18-session"
    env_factory: Callable[[EnvConfig], object] | None = None


def _gae(rewards, values, dones, gamma: float, lam: float):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for index in reversed(range(len(rewards))):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        nonterminal = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        last = delta + gamma * lam * nonterminal * last
        advantages[index] = last
    returns = advantages + np.asarray(values[: len(rewards)], dtype=np.float32)
    return advantages, returns


def _observation(observation: dict, info: dict) -> dict[str, np.ndarray]:
    if info.get("bullets_raw") is not None:
        return entity_obs.from_env(info, observation)
    self_feat = np.asarray(observation["self_feat"], dtype=np.float32).reshape(-1)
    player_x = spec.PLAYFIELD_X + float(self_feat[0]) * spec.PLAYFIELD_W
    player_y = spec.PLAYFIELD_Y + float(self_feat[1]) * spec.PLAYFIELD_H
    empty = np.zeros((0, 2), dtype=np.float32)
    return entity_obs.build(player_x, player_y, empty, empty, np.zeros((0,), dtype=np.float32))


def _evaluate(model, env, device, episodes: int, max_steps: int) -> dict:
    """Deterministic closed-loop evaluation on the *training* environment."""
    frames: list[int] = []
    reasons: dict[str, int] = {}
    for _ in range(episodes):
        observation, info = env.reset()
        steps = 0
        episode_info: dict = {}
        while True:
            batch = entity_obs.batch([_observation(observation, info)])
            with torch.no_grad():
                candidate, _logp, _value, _entropy = model.act(
                    entity_policy.to_tensors(batch, device), deterministic=True
                )
            action = np.asarray(entity_obs.action_of_candidate(int(candidate[0])), dtype=np.int64)
            observation, _reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                episode_info = info.get("episode", {})
                break
            if steps >= max_steps:
                break
        frame_count = int(episode_info.get("frames", 2 * steps))
        frames.append(frame_count)
        reason = str(episode_info.get("termination_reason", "cap"))
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "episodes": len(frames),
        "frames_mean": round(float(np.mean(frames)), 1) if frames else 0.0,
        "frames_median": round(float(np.median(frames)), 1) if frames else 0.0,
        "frames_min": int(min(frames)) if frames else 0,
        "frames_max": int(max(frames)) if frames else 0,
        "reasons": reasons,
        "frames": frames,
    }


def train(config: PPO18Config) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.out_dir.mkdir(parents=True, exist_ok=True)

    if config.init_checkpoint:
        model, _extra = entity_policy.load_checkpoint(config.init_checkpoint, map_location=device)
        model.to(device)
    else:
        model = entity_policy.EntityPolicy().to(device)
    anchor = None
    if config.anchor_checkpoint and config.kl_coef > 0:
        anchor, _extra = entity_policy.load_checkpoint(config.anchor_checkpoint, map_location=device)
        anchor.to(device).eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    reward_weights = RewardWeights()
    if config.survive_weight is not None:
        if not 0.0 < float(config.survive_weight) <= 10.0:
            raise ValueError(f"survive_weight must be in (0, 10], got {config.survive_weight}")
        reward_weights.survive = float(config.survive_weight)
    env = (config.env_factory or Th08Env)(
        EnvConfig(
            frameskip=2,
            max_steps=config.episode_max_steps,
            reset_mode="relaunch",
            run_dir=config.run_dir,
            expose_raw_bullets=True,
            expose_bosses=False,
            reward=reward_weights,
        )
    )
    summary: dict = {
        "schema": "v311-ppo18-summary/1",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            name: (str(value) if isinstance(value, Path) else value)
            for name, value in vars(config).items()
            if name != "env_factory"
        },
        "episodes": [],
        "evaluations": [],
        "updates": 0,
    }
    logger = (config.out_dir / "metrics.jsonl").open("w", encoding="utf-8")

    def log(row: dict) -> None:
        logger.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.flush()

    def _phi(info_now: dict, observation_now: dict) -> float:
        """Escape-option potential of one state, ``0`` when shaping is off."""
        if config.pbrs_weight <= 0.0:
            return 0.0
        raw = info_now.get("bullets_raw")
        if raw is None:
            return 0.0
        self_feat = np.asarray(observation_now["self_feat"], dtype=np.float32).reshape(-1)
        player_x = spec.PLAYFIELD_X + float(self_feat[0]) * spec.PLAYFIELD_W
        player_y = spec.PLAYFIELD_Y + float(self_feat[1]) * spec.PLAYFIELD_H
        return options.phi(
            (player_x, player_y), raw["position"], raw["velocity"], raw["half"], config.pbrs_horizon
        )

    observation, info = env.reset()
    episode = 0
    episode_steps = 0
    episode_return = 0.0
    episode_shaped_return = 0.0
    episode_x: list[float] = []
    episode_moves: list[int] = []
    episode_phi: list[float] = []
    phi_before = _phi(info, observation)
    steps_done = 0
    try:
        while steps_done < config.total_steps:
            batch_obs: list[dict] = []
            batch_actions: list[int] = []
            batch_logp: list[float] = []
            batch_values: list[float] = []
            batch_rewards: list[float] = []
            batch_dones: list[float] = []
            for _ in range(config.rollout_steps):
                observation_now = _observation(observation, info)
                batch = entity_obs.batch([observation_now])
                with torch.no_grad():
                    candidate, logp, value, _entropy = model.act(
                        entity_policy.to_tensors(batch, device), deterministic=False
                    )
                action = np.asarray(entity_obs.action_of_candidate(int(candidate[0])), dtype=np.int64)
                observation, reward, terminated, truncated, info = env.step(action)
                finished = bool(terminated or truncated)
                phi_after = _phi(info, observation)
                shaping = options.shaped_reward(
                    phi_before, phi_after, config.gamma, terminal=finished
                )
                step_reward = float(reward) + float(config.pbrs_weight) * shaping
                episode_shaped_return += float(config.pbrs_weight) * shaping
                episode_phi.append(phi_before)
                phi_before = 0.0 if finished else phi_after

                batch_obs.append(observation_now)
                batch_actions.append(int(candidate[0]))
                batch_logp.append(float(logp[0]))
                batch_values.append(float(value[0]))
                batch_rewards.append(float(step_reward))
                batch_dones.append(float(terminated or truncated))
                episode_steps += 1
                steps_done += 1
                episode_return += float(reward)
                episode_x.append(float(observation_now["meta"][0]))
                episode_moves.append(int(candidate[0]) % 9)

                if terminated or truncated:
                    episode += 1
                    row = {
                        "episode": episode,
                        "steps": episode_steps,
                        "frames": int(info.get("episode", {}).get("frames", 2 * episode_steps)),
                        "deaths": int(info.get("episode", {}).get("deaths", 0)),
                        "termination_reason": info.get("episode", {}).get("termination_reason", ""),
                        "return": round(episode_return, 3),
                        "shaped_return": round(episode_shaped_return, 3),
                        "phi_mean": round(float(np.mean(episode_phi)), 3) if episode_phi else 0.0,
                        "phi_last": round(float(episode_phi[-1]), 3) if episode_phi else 0.0,
                        "still_rate": round(float(np.mean([m == 0 for m in episode_moves])), 3),
                        "move_histogram": np.bincount(np.asarray(episode_moves), minlength=9).tolist(),
                        "x_p10": round(float(np.percentile(episode_x, 10)), 3),
                        "x_p50": round(float(np.percentile(episode_x, 50)), 3),
                        "x_p90": round(float(np.percentile(episode_x, 90)), 3),
                        "step": steps_done,
                    }
                    summary["episodes"].append(row)
                    print(f"[ppo18] {row}", flush=True)
                    log({"kind": "episode", **row})
                    episode_steps = 0
                    episode_return = 0.0
                    episode_shaped_return = 0.0
                    episode_x = []
                    episode_moves = []
                    episode_phi = []
                    observation, info = env.reset()
                    phi_before = _phi(info, observation)

                    if config.eval_every_episodes and episode % config.eval_every_episodes == 0:
                        model.eval()
                        evaluation = _evaluate(
                            model, env, device, config.eval_episodes, config.episode_max_steps
                        )
                        evaluation["episode"] = episode
                        evaluation["step"] = steps_done
                        summary["evaluations"].append(evaluation)
                        print(f"[ppo18] eval {evaluation}", flush=True)
                        log({"kind": "evaluation", **evaluation})
                        model.train()
                if steps_done >= config.total_steps:
                    break

            # --- update ---------------------------------------------------
            with torch.no_grad():
                last = _observation(observation, info)
                _c, _l, last_value, _e = model.act(
                    entity_policy.to_tensors(entity_obs.batch([last]), device), deterministic=True
                )
            values = batch_values + [float(last_value[0])]
            advantages, returns = _gae(
                np.asarray(batch_rewards), np.asarray(values), np.asarray(batch_dones), config.gamma, config.lam
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            tensors = entity_policy.to_tensors(entity_obs.batch(batch_obs), device)
            tensors["actions"] = torch.as_tensor(batch_actions, dtype=torch.int64, device=device)
            tensors["old_logp"] = torch.as_tensor(batch_logp, dtype=torch.float32, device=device)
            tensors["advantages"] = torch.as_tensor(advantages, dtype=torch.float32, device=device)
            tensors["returns"] = torch.as_tensor(returns, dtype=torch.float32, device=device)

            count = len(batch_actions)
            statistics: dict[str, float] = {}
            for _epoch in range(config.epochs):
                order = torch.randperm(count, device=device)
                for start in range(0, count, config.minibatch):
                    index = order[start : start + config.minibatch]
                    logits, value = model(
                        {
                            "bullets": tensors["bullets"][index],
                            "meta": tensors["meta"][index],
                            "grid": tensors["grid"][index],
                        }
                    )
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(tensors["actions"][index])
                    entropy = dist.entropy().mean()
                    ratio = (new_logp - tensors["old_logp"][index]).exp()
                    unclipped = ratio * tensors["advantages"][index]
                    clipped = (
                        ratio.clamp(1.0 - config.clip, 1.0 + config.clip) * tensors["advantages"][index]
                    )
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = F.mse_loss(value, tensors["returns"][index])
                    kl = torch.zeros((), device=device)
                    if anchor is not None:
                        with torch.no_grad():
                            anchor_logits, _anchor_value = anchor(
                                {
                                    "bullets": tensors["bullets"][index],
                                    "meta": tensors["meta"][index],
                                    "grid": tensors["grid"][index],
                                }
                            )
                        kl = F.kl_div(
                            F.log_softmax(logits, dim=-1),
                            F.softmax(anchor_logits, dim=-1),
                            reduction="batchmean",
                        )
                    loss = (
                        policy_loss
                        + config.vf_coef * value_loss
                        - config.ent_coef * entropy
                        + config.kl_coef * kl
                    )
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimiser.step()
                    statistics = {
                        "policy_loss": round(float(policy_loss.detach()), 5),
                        "value_loss": round(float(value_loss.detach()), 5),
                        "entropy": round(float(entropy.detach()), 5),
                        "kl_anchor": round(float(kl.detach()), 5),
                    }
                    summary["updates"] += 1

            update_row = {"step": steps_done, "update": summary["updates"], **statistics}
            log({"kind": "update", **update_row})
            if config.save_every and steps_done % config.save_every < config.rollout_steps:
                entity_policy.save_checkpoint(
                    config.out_dir / f"ppo18-{steps_done:07d}.pt",
                    model,
                    extra={"step": steps_done, "updates": summary["updates"]},
                )
                entity_policy.save_checkpoint(
                    config.out_dir / "last.pt", model, extra={"step": steps_done, "updates": summary["updates"]}
                )
    finally:
        env.close()
        logger.close()
        entity_policy.save_checkpoint(
            config.out_dir / "last.pt",
            model,
            extra={"step": steps_done, "updates": summary["updates"], "finished": True},
        )
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        summary["steps"] = steps_done
        summary["episodes_run"] = len(summary["episodes"])
        summary["last_evaluation"] = summary["evaluations"][-1] if summary["evaluations"] else None
        (config.out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 2 PPO fine-tuning (18-way privileged policy)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--anchor-checkpoint", default="")
    parser.add_argument("--total-steps", type=int, default=400_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--kl-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--episode-max-steps", type=int, default=12_000)
    parser.add_argument("--eval-every-episodes", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--pbrs-weight",
        type=float,
        default=0.0,
        help="coefficient on the escape-option PBRS potential (0 disables)",
    )
    parser.add_argument("--pbrs-horizon", type=int, default=options.HORIZON_FRAMES)
    parser.add_argument(
        "--survive-weight",
        type=float,
        default=None,
        help="override the environment survive weight (default 0.01); step-2 reward-scale test",
    )
    args = parser.parse_args(argv)
    summary = train(
        PPO18Config(
            out_dir=Path(args.out),
            init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
            anchor_checkpoint=Path(args.anchor_checkpoint) if args.anchor_checkpoint else None,
            total_steps=args.total_steps,
            rollout_steps=args.rollout_steps,
            epochs=args.epochs,
            minibatch=args.minibatch,
            kl_coef=args.kl_coef,
            ent_coef=args.ent_coef,
            lr=args.lr,
            episode_max_steps=args.episode_max_steps,
            eval_every_episodes=args.eval_every_episodes,
            eval_episodes=args.eval_episodes,
            save_every=args.save_every,
            device=args.device,
            seed=args.seed,
            survive_weight=args.survive_weight,
            pbrs_weight=args.pbrs_weight,
            pbrs_horizon=args.pbrs_horizon,
        )
    )
    print(json.dumps({"steps": summary["steps"], "episodes": summary["episodes_run"], "last_evaluation": summary["last_evaluation"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
