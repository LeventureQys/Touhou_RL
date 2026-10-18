"""PPO fine-tuning on the live environment, anchored to an imitation prior.

Why PPO here: the environment is a single 60 Hz game process, so throughput is
~15-30 decisions/s and every sample is expensive.  On-policy PPO with a short
horizon is the most robust way to improve a *pre-trained* policy without a replay
buffer; the imitation prior is kept alive by a KL anchor to the frozen BC model
(``kl_coef``), which is what prevents the classic collapse where RL instantly
unlearns a good supervised policy.

The scalar reward from ``RewardEngine`` is used for the advantage; GAE with
``gamma=0.995`` gives an effective horizon of ~200 decisions (≈7 s of play).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import spec
from .env import EnvConfig, Th08Env
from .models import DanmakuPolicy, NetConfig, load_checkpoint, observation_batch, save_checkpoint


@dataclass
class PPOConfig:
    out_dir: Path
    total_steps: int = 200_000
    rollout_steps: int = 1024
    epochs: int = 4
    minibatch: int = 256
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 1e-4
    max_grad_norm: float = 0.5
    kl_coef: float = 0.5
    bc_checkpoint: Path | None = None
    init_checkpoint: Path | None = None
    device: str = "cuda"
    seed: int = 11
    save_every: int = 20_000
    log_every: int = 1
    env: EnvConfig = field(default_factory=EnvConfig)


def _gae(rewards, values, dones, gamma: float, lam: float):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        advantages[t] = last
    returns = advantages + np.asarray(values[: len(rewards)], dtype=np.float32)
    return advantages, returns


def _to_batch(observations: list[dict], device) -> dict[str, torch.Tensor]:
    return {
        "self_feat": torch.as_tensor(np.stack([o["self_feat"] for o in observations]), device=device),
        "bullets": torch.as_tensor(np.stack([o["bullets"] for o in observations]), device=device),
        "danger": torch.as_tensor(np.stack([o["danger"] for o in observations]), device=device) / 8.0,
    }


def _jsonable(value):
    """Best-effort JSON encoding for dataclass/path fields in the summary."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(value).items()}
    return str(value)


def train_ppo(cfg: PPOConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.init_checkpoint:
        model, _ = load_checkpoint(cfg.init_checkpoint, map_location=device)
        model.to(device)
        print(f"[ppo] initialised from {cfg.init_checkpoint}", flush=True)
    elif cfg.bc_checkpoint and cfg.kl_coef > 0:
        # Fine-tuning starts from the imitation prior, not from noise: that is the
        # whole point of the KL-anchored pipeline.
        model, _ = load_checkpoint(cfg.bc_checkpoint, map_location=device)
        model.to(device)
        print(f"[ppo] initialised from the BC prior {cfg.bc_checkpoint}", flush=True)
    else:
        model = DanmakuPolicy(NetConfig()).to(device)
    bc_model: DanmakuPolicy | None = None
    if cfg.bc_checkpoint and cfg.kl_coef > 0:
        bc_model, _ = load_checkpoint(cfg.bc_checkpoint, map_location=device)
        bc_model.to(device).eval()
        for param in bc_model.parameters():
            param.requires_grad_(False)
        print(f"[ppo] KL anchor -> {cfg.bc_checkpoint}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    env = Th08Env(cfg.env)
    summary = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        "parameters": model.num_parameters(),
        "config": _jsonable({k: v for k, v in cfg.__dict__.items() if k != "env"}),
        "env": _jsonable(cfg.env.__dict__),
        "iterations": [],
        "episodes": [],
    }
    global_step = 0
    iteration = 0
    obs, _info = env.reset()
    started = time.time()
    try:
        while global_step < cfg.total_steps:
            iteration += 1
            observations, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
            ep_return = 0.0
            for _ in range(cfg.rollout_steps):
                observation_tensor = observation_batch(obs, device)
                with torch.no_grad():
                    action_t, logp_t, value_t, _ = model.act(observation_tensor, deterministic=False)
                action = action_t[0].cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                observations.append({k: v.copy() for k, v in obs.items()})
                actions.append(action)
                log_probs.append(float(logp_t[0]))
                rewards.append(float(reward))
                values.append(float(value_t[0]))
                dones.append(bool(terminated or truncated))
                ep_return += float(reward)
                global_step += 1
                if terminated or truncated:
                    summary["episodes"].append(info.get("episode", {}))
                    print(
                        f"[ppo] episode steps={info.get('step')} stage={info.get('stages_reached')} "
                        f"deaths={info.get('deaths')} lives={info.get('lives')} ret={round(ep_return, 2)}",
                        flush=True,
                    )
                    ep_return = 0.0
                    obs, _info = env.reset()
                if global_step >= cfg.total_steps:
                    break
            # bootstrap value for the last state
            with torch.no_grad():
                last_value = float(
                    model.value(observation_batch(obs, device))[0]
                )
            values.append(last_value)
            advantages, returns = _gae(rewards, values, dones, cfg.gamma, cfg.lam)
            batch_obs = _to_batch(observations, device)
            batch_actions = torch.as_tensor(np.stack(actions), device=device, dtype=torch.long)
            batch_old_logp = torch.as_tensor(log_probs, device=device, dtype=torch.float32)
            batch_adv = torch.as_tensor(advantages, device=device, dtype=torch.float32)
            batch_returns = torch.as_tensor(returns, device=device, dtype=torch.float32)
            batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

            n = len(rewards)
            losses = []
            for _epoch in range(cfg.epochs):
                order = torch.randperm(n, device=device)
                for start in range(0, n, cfg.minibatch):
                    idx = order[start : start + cfg.minibatch]
                    minibatch = {k: v[idx] for k, v in batch_obs.items()}
                    out = model(minibatch)
                    logp, entropy, value = model.evaluate_actions(minibatch, batch_actions[idx])
                    ratio = torch.exp(logp - batch_old_logp[idx])
                    adv = batch_adv[idx]
                    policy_loss = -torch.min(
                        ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv
                    ).mean()
                    value_loss = F.mse_loss(value, batch_returns[idx])
                    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy.mean()
                    if bc_model is not None:
                        with torch.no_grad():
                            bc_out = bc_model(minibatch)
                        kl = 0.0
                        for name in ("move_logits", "shoot_logits", "focus_logits", "bomb_logits"):
                            kl = kl + F.kl_div(
                                F.log_softmax(out[name], dim=-1),
                                F.softmax(bc_out[name], dim=-1),
                                reduction="batchmean",
                            )
                        loss = loss + cfg.kl_coef * kl
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optim.step()
                    losses.append(float(loss.detach()))
            row = {
                "iteration": iteration,
                "global_step": global_step,
                "reward_mean": round(float(np.mean(rewards)), 4),
                "value_loss": round(float(np.mean(losses)), 4),
                "elapsed_s": round(time.time() - started, 1),
                "steps_per_s": round(global_step / max(1e-6, time.time() - started), 2),
            }
            summary["iterations"].append(row)
            if iteration % cfg.log_every == 0:
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if global_step % cfg.save_every < cfg.rollout_steps:
                save_checkpoint(cfg.out_dir / f"ppo-{global_step:07d}.pt", model, {"step": global_step})
            save_checkpoint(cfg.out_dir / "last.pt", model, {"step": global_step})
    finally:
        env.close()
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        summary["global_step"] = global_step
        (cfg.out_dir / "ppo-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PPO fine-tuning on live TH08")
    parser.add_argument("--out", required=True)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--bc-checkpoint", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--kl-coef", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--practice-stage", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--run-dir", default=str(Path(__file__).resolve().parents[1] / "third_party/th08/build/rl/env-session"))
    args = parser.parse_args(argv)
    env_cfg = EnvConfig(
        frameskip=args.frameskip,
        practice_stage=None if args.practice_stage < 0 else args.practice_stage,
        max_steps=args.max_steps,
        team=args.team,
        reset_mode="relaunch",
        run_dir=Path(args.run_dir),
    )
    cfg = PPOConfig(
        out_dir=Path(args.out),
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        bc_checkpoint=Path(args.bc_checkpoint) if args.bc_checkpoint else None,
        init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
        kl_coef=args.kl_coef,
        lr=args.lr,
        device=args.device,
        env=env_cfg,
    )
    summary = train_ppo(cfg)
    print(json.dumps({k: v for k, v in summary.items() if k != "iterations"}, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
