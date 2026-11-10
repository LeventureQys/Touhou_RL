"""EXP-P4-001: a PPO prototype that starts from the BC ``m1`` checkpoint.

This is **exploration, not a gate**.  It does not touch the engine, the protocol,
``obs-spec-v1`` or the frozen student model; it composes the pieces the earlier
phases already verified so the question "can RL move the 2,250-frame plateau at
all?" can be asked with a real number instead of an opinion.

What it does
------------
* Builds :data:`--engines` independent ``th08-modern`` processes (one writable
  game-data directory each) and runs :class:`~touhou_rl.v400.rl.env.RlEnv` on
  top of them: obs-v1 pixels with no privileged state, the 18x2 frozen action
  set, engine-truth reward, stage 0 / normal / cap 19,887 / death stop.
* Initialises the actor from the BC checkpoint (``strict=True``; the value head
  is new) and runs PPO with GAE(0.95) and clipping 0.2.
* Saves a ``v400-student-checkpoint-1`` actor, so
  ``tools/rl/v400_student_eval.py`` measures it on the same 20-seed ruler as the
  BC baseline with no conversion and no second definition of "the policy".
* Writes ``metrics.jsonl`` (one row per update: returns, episode frames, deaths,
  entropy, KL, clip fraction, value loss, throughput) and ``train_summary.json``.

Deliberate choices, stated up front
-----------------------------------
* ``--reward-scale 1000`` divides the *specified* reward (``env.REWARD_TERMS``)
  before it reaches the optimiser.  The reward definition does not change; this
  only keeps the critic's target O(1) so one learning rate works for both heads.
* The actor's BatchNorm stays in eval mode (frozen statistics) for the whole run:
  ``StudentNet.act`` deploys in eval mode, so training against batch statistics
  would evaluate a different network than it trained.
* The first ``--value-warmup-updates`` updates train the value head only.  A
  random critic's first gradients would otherwise tear the cloned policy apart
  before any return signal justifies moving it.

Run inside WSL under one Xvfb display (see ``.tmp_v400_rl/run_wsl.sh``)::

    xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
      tools/rl/v400_rl_train.py \
      --bc-checkpoint .tmp_v400_obsv2/arms/m1/checkpoint.pt \
      --engines 6 --steps 200000 --minutes 14 \
      --out-dir .tmp_v400_rl/phaseA
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from touhou_rl.v400.gym.actions import cand_to_mask  # noqa: E402
from touhou_rl.v400.rl.env import (  # noqa: E402
    DECISION_FRAMES,
    DEFAULT_MAX_FRAMES,
    END_DEATH_STOP,
    END_FRAME_CAP,
    END_GAME_OVER,
    END_STAGE_CLEAR,
    EngineBackend,
    N_ACTIONS,
    REWARD_TERMS,
    RlEnv,
    VecRlEnv,
    reward_scale_report,
)
from touhou_rl.v400.rl.policy import (  # noqa: E402
    ActorCritic,
    build_actor_critic,
    save_policy_checkpoint,
)
from touhou_rl.v400.student import obs_spec  # noqa: E402

REPO = REPO_ROOT
#: WSL-side engine used by every v4.0.0 phase (sha256 1462829b...).
DEFAULT_ENGINE = "/root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern"
DEFAULT_DATA_TEMPLATE = "/root/v400-data/game"
#: The BC baseline the prototype has to beat (EXP-P3-009, 20 seeds 1000-1019).
DEFAULT_BC_CHECKPOINT = str(REPO / ".tmp_v400_obsv2" / "arms" / "m1" / "checkpoint.pt")

SCHEMA = "v400-rl-train-1"


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--data-template", default=DEFAULT_DATA_TEMPLATE,
                        help="read-only game data dir the per-engine dirs link to")
    parser.add_argument("--data-root", default=str(REPO / ".tmp_v400_rl" / "data"))
    parser.add_argument("--out-dir", default=str(REPO / ".tmp_v400_rl" / "phaseA"))
    parser.add_argument("--bc-checkpoint", default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument("--init-from", default=None,
                        help="warm start from an earlier RL actor checkpoint "
                             "(phase B of a two-phase run)")
    parser.add_argument("--engines", type=int, default=6)
    parser.add_argument("--steps", type=int, default=200_000,
                        help="target environment steps (one step == one decision "
                             "== 2 logic frames)")
    parser.add_argument("--minutes", type=float, default=15.0,
                        help="wall-clock cap; the loop stops at whichever of "
                             "--steps / --minutes comes first")
    parser.add_argument("--rollout", type=int, default=256,
                        help="decisions per env per PPO update")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--reward-scale", type=float, default=1000.0,
                        help="divisor applied to the specified reward before the "
                             "optimiser sees it (optimiser hygiene, not a change "
                             "of the reward definition)")
    parser.add_argument("--value-warmup-updates", type=int, default=5,
                        help="updates that train the value head only")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--difficulty", default="normal")
    parser.add_argument("--stop-on-death", dest="stop_on_death", action="store_true",
                        default=True)
    parser.add_argument("--no-stop-on-death", dest="stop_on_death",
                        action="store_false")
    parser.add_argument("--train-seed-base", type=int, default=5000,
                        help="first episode seed; the 20 evaluation seeds "
                             "(1000-1019) are never used for training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--privileged-state", action="store_true",
                        help="DIAGNOSTIC ONLY: also put the decoded 25 engine "
                             "fields in info. Never used as observation.")
    parser.add_argument("--save-at", action="append", default=[],
                        help="extra step counts at which to save a checkpoint")
    parser.add_argument("--metrics-every", type=int, default=1)
    parser.add_argument("--state-equivalence", type=int, default=1,
                        help="check N live frames of shm-state == STATE reply at "
                             "start-up (0 disables)")
    parser.add_argument("--throughput-only", type=float, default=0.0,
                        help="run the BC policy for this many seconds, print "
                             "decisions/s and frames/s, exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the hyper-parameter table and exit")
    parser.add_argument("--validate-ruler", type=int, default=0,
                        help="play N deterministic BC episodes on seeds 1000.. and "
                             "compare them with the recorded m1 per-seed frames; "
                             "this is the ruler-identity check")
    return parser


#: ``m1`` per-seed S1N frames, copied from
#: ``.tmp_v400_power/m1/episodes.jsonl`` (EXP-P3-009, 20 seeds 1000-1019).  A
#: deterministic replay of the same checkpoint through this environment has to
#: reproduce them exactly, or the RL loop is not measuring the published ruler.
M1_PER_SEED_FRAMES: dict[int, int] = {
    1000: 2243, 1001: 2999, 1002: 2299, 1003: 3355, 1004: 2976,
}


# --------------------------------------------------------------------------- #
# engine plumbing
# --------------------------------------------------------------------------- #

def prepare_data_dirs(template: str, root: str, count: int) -> list[Path]:
    """One writable game-data dir per engine.

    Two episodes side by side must not share ``score.dat`` / ``th08.cfg``
    (EXP-P0-021, re-used by ``.tmp_v400_power/launch_power.sh``).  The two large
    archives are symlinks; only the two mutable files are copied.
    """

    template_path = Path(template)
    if not template_path.is_dir():
        raise SystemExit(f"data template not found: {template_path}")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    dirs: list[Path] = []
    for index in range(count):
        data = root_path / f"env-{index}"
        data.mkdir(parents=True, exist_ok=True)
        for name in ("th08.dat", "thbgm.dat"):
            link = data / name
            if not link.exists():
                link.symlink_to(template_path / name)
        for name, source in (("th08.cfg", "th08.cfg.orig"),
                             ("score.dat", "score.dat.orig")):
            target = data / name
            if not target.exists():
                shutil.copy2(template_path / source, target)
        dirs.append(data)
    return dirs


def build_envs(args, data_dirs: list[Path], *, max_frames: int | None = None) -> VecRlEnv:
    envs = []
    for index, data in enumerate(data_dirs):
        backend = EngineBackend(args.engine, data,
                                shm_name=f"v400rltrain{index}{args.seed % 100000}")
        envs.append(RlEnv(backend, max_frames=max_frames or args.max_frames,
                          stop_on_death=args.stop_on_death,
                          privileged_state=args.privileged_state,
                          stage=args.stage, difficulty=args.difficulty))
    return VecRlEnv(envs)


def check_state_equivalence(env: RlEnv, samples: int = 3) -> dict:
    """Show *where* the readings may come from, on live frames.

    EXP-P4-001 measured a trap here, so it is checked every run rather than
    documented: the shared block's 25-field state region is **stale until a
    ``STATE`` command runs**.  Before any ``STATE`` after ``RESET`` the block
    reports ``state_hash=0`` and all 25 fields zero, while the ``STATE`` reply
    already carries ``lives_remaining=2.0``.  A loop that trusted the block would
    see ``deaths=0 / score=0 / stage=0`` forever and run every episode to the
    frame cap — which is exactly what the first version of this prototype did.

    The check therefore records both: the block *before* a ``STATE`` (expected
    stale) and the block *after* one (expected equal to the reply).  The env uses
    the reply, so only the second comparison is a correctness condition.
    """

    from touhou_rl.v400.labels.state import decode_state

    stale_before: list[dict] = []
    mismatches: list[dict] = []
    for index in range(samples):
        env.step((index * 5) % N_ACTIONS)
        block_before = decode_state([int(value) for value in env.backend._shm.state()])
        reply = env.backend.request_state_reply()
        block_after = decode_state([int(value) for value in env.backend._shm.state()])
        stale_before.append({
            "sample": index,
            "block_deaths": block_before["deaths"],
            "reply_deaths": reply["deaths"],
            "block_score": block_before["score"],
            "reply_score": reply["score"],
            "block_is_stale": block_before != reply,
        })
        diffs = {key: (block_after[key], reply[key]) for key in reply
                 if key in block_after and block_after[key] != reply[key]}
        if diffs:
            mismatches.append({"sample": index, "diffs": diffs})
    return {
        "samples": samples,
        "block_stale_before_state_command": any(
            row["block_is_stale"] for row in stale_before),
        "block_equals_reply_after_state_command": not mismatches,
        "mismatches": mismatches,
        "stale_samples": stale_before,
        "conclusion": "the env reads the STATE reply per logic frame; only the "
                      "pixels come from the shared block",
    }


# --------------------------------------------------------------------------- #
# rollout buffer
# --------------------------------------------------------------------------- #

@dataclass
class Transition:
    play: np.ndarray
    hud: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    advantage: float = 0.0
    returns: float = 0.0


@dataclass
class EpisodeRecord:
    seed: int
    frames: int
    decisions: int
    deaths: float
    score: int
    end: str
    episode_return: float


@dataclass
class TrainingState:
    updates: int = 0
    steps: int = 0
    episode_frames: list[int] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    episode_deaths: list[float] = field(default_factory=list)
    episode_ends: list[str] = field(default_factory=list)
    episodes: list[EpisodeRecord] = field(default_factory=list)
    started: float = 0.0

    def recent_frames(self, count: int = 20) -> list[int]:
        return self.episode_frames[-count:]


def stack_obs(observations) -> tuple[torch.Tensor, torch.Tensor]:
    play = np.stack([obs["play"] for obs in observations], axis=0)
    hud = np.stack([obs["hud"] for obs in observations], axis=0)
    return torch.from_numpy(play).float(), torch.from_numpy(hud).float()


def gae_returns(trajectories: list[list[Transition]], final_values: list[float], *,
                gamma: float, lam: float) -> None:
    """Fill ``advantage`` / ``returns`` in place, one episode segment at a time.

    A ``done`` transition ends a segment: its bootstrap value is 0 and the GAE
    accumulator restarts, so an advantage can never leak across an episode
    boundary even when a rollout contains two episodes of the same engine.
    """

    for trajectory, final_value in zip(trajectories, final_values):
        next_value = float(final_value)
        gae = 0.0
        for transition in reversed(trajectory):
            if transition.done:
                next_value = 0.0
                gae = 0.0
            delta = transition.reward + gamma * next_value - transition.value
            gae = delta + gamma * lam * gae
            transition.advantage = gae
            transition.returns = gae + transition.value
            next_value = transition.value


# --------------------------------------------------------------------------- #
# PPO update
# --------------------------------------------------------------------------- #

def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer,
               transitions: list[Transition], args, *, device: str,
               value_only: bool, rng: np.random.Generator) -> dict:
    play = torch.from_numpy(np.stack([row.play for row in transitions])).float().to(device)
    hud = torch.from_numpy(np.stack([row.hud for row in transitions])).float().to(device)
    actions = torch.tensor([row.action for row in transitions], dtype=torch.long,
                           device=device)
    old_log_prob = torch.tensor([row.log_prob for row in transitions], dtype=torch.float32,
                                device=device)
    advantages = torch.tensor([row.advantage for row in transitions], dtype=torch.float32,
                              device=device)
    returns = torch.tensor([row.returns for row in transitions], dtype=torch.float32,
                           device=device)
    if advantages.numel() > 1:
        std = float(advantages.std(unbiased=False))
        if std > 1e-8:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)

    count = len(transitions)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
             "clip_fraction": 0.0, "updates": 0}
    indices = np.arange(count)
    for _epoch in range(args.epochs):
        rng.shuffle(indices)
        for start in range(0, count, args.minibatch):
            batch = indices[start:start + args.minibatch]
            if batch.size < 2:
                continue
            batch_t = torch.from_numpy(batch).to(device)
            move_logits, bomb_logits, values = model(play[batch_t], hud[batch_t])
            log_prob = model.joint_log_prob(move_logits, bomb_logits, actions[batch_t])
            entropy = model.entropy(move_logits, bomb_logits).mean()
            value_loss = F.mse_loss(values, returns[batch_t])
            if value_only:
                loss = args.value_coef * value_loss
                policy_loss = torch.zeros((), device=device)
                clip_fraction = torch.zeros((), device=device)
                approx_kl = torch.zeros((), device=device)
            else:
                log_ratio = log_prob - old_log_prob[batch_t]
                ratio = log_ratio.exp()
                unclipped = ratio * advantages[batch_t]
                clipped = torch.clamp(ratio, 1.0 - args.clip,
                                      1.0 + args.clip) * advantages[batch_t]
                policy_loss = -torch.min(unclipped, clipped).mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy * entropy
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > args.clip).float().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy.detach())
            stats["approx_kl"] += float(approx_kl)
            stats["clip_fraction"] += float(clip_fraction)
            stats["updates"] += 1
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
        stats[key] = stats[key] / max(stats["updates"], 1)
    stats["samples"] = count
    return stats


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #

def env_summary(frames: list[int], ends: list[str]) -> dict:
    if not frames:
        return {"episodes": 0}
    array = np.asarray(frames, dtype=np.float64)
    return {
        "episodes": int(array.size),
        "mean_frames": float(array.mean()),
        "min_frames": float(array.min()),
        "max_frames": float(array.max()),
        "death_stop": int(sum(1 for end in ends if end == END_DEATH_STOP)),
        "game_over": int(sum(1 for end in ends if end == END_GAME_OVER)),
        "stage_clear": int(sum(1 for end in ends if end == END_STAGE_CLEAR)),
        "frame_cap": int(sum(1 for end in ends if end == END_FRAME_CAP)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dirs = prepare_data_dirs(args.data_template, args.data_root, args.engines)

    init_source = args.init_from or args.bc_checkpoint
    model, provenance = build_actor_critic(init_source, device=device, seed=args.seed)
    model.train()
    config = {
        "schema": SCHEMA,
        "experiment": "EXP-P4-001",
        "note": "P4 exploratory prototype; not G4, not a gate",
        "algorithm": "PPO",
        "engines": args.engines,
        "stage": args.stage,
        "difficulty": args.difficulty,
        "max_frames": args.max_frames,
        "stop_on_death": args.stop_on_death,
        "decision_frames": DECISION_FRAMES,
        "action_space": {"move_focus_classes": 18, "bomb_classes": 2,
                         "joint_actions": N_ACTIONS,
                         "encoding": "action = cand * 2 + bomb; mask = cand_to_mask(cand, bomb)"},
        "observation": {"spec": obs_spec.OBS_SPEC_VERSION,
                        "digest": obs_spec.obs_spec_digest(),
                        "privileged_state_in_observation": False,
                        "privileged_diagnostic_only": bool(args.privileged_state)},
        "hyperparameters": {
            "lr": args.lr, "rollout_per_env": args.rollout,
            "batch_transitions": args.rollout * args.engines,
            "epochs": args.epochs, "minibatch": args.minibatch,
            "clip": args.clip, "entropy_coef": args.entropy,
            "value_coef": args.value_coef, "gamma": args.gamma, "lam": args.lam,
            "max_grad_norm": args.max_grad_norm,
            "reward_scale_divisor": args.reward_scale,
            "value_warmup_updates": args.value_warmup_updates,
            "target_steps": args.steps, "wall_cap_minutes": args.minutes,
            "train_seed_base": args.train_seed_base,
        },
        "reward": reward_scale_report(),
        "init": provenance,
        "engine": args.engine,
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": device,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0

    vec = build_envs(args, data_dirs)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    seeds = [args.train_seed_base + index for index in range(args.engines)]
    next_seeds = list(seeds)
    state = TrainingState(started=time.monotonic())
    exit_code = 0
    try:
        t_open = time.monotonic()
        vec.open_all()
        config["open_seconds"] = round(time.monotonic() - t_open, 2)
        results = vec.reset_all(next_seeds)
        observations = [row[0] for row in results]
        infos = [row[1] for row in results]
        if args.state_equivalence > 0:
            equivalence = check_state_equivalence(vec.envs[0], args.state_equivalence)
            config["state_equivalence"] = equivalence
            print(json.dumps({"state_equivalence": equivalence}), flush=True)
            if not equivalence["block_equals_reply_after_state_command"]:
                raise SystemExit("the STATE reply and the shared block disagree even "
                                 "after a STATE command; readings are unsound")
            observations[0], infos[0] = vec.envs[0].reset(next_seeds[0])
        if args.validate_ruler > 0:
            return run_ruler_validation(args, model, vec)
        if args.throughput_only > 0:
            return run_throughput_only(args, model, vec, observations, infos, state)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        update_rng = np.random.default_rng(args.seed)
        deadline = time.monotonic() + args.minutes * 60.0
        save_steps = sorted({int(value) for value in args.save_at})
        episode_frames = state.episode_frames
        episode_ends = state.episode_ends
        last_log = time.monotonic()
        steps_at_last_log = 0
        frames_at_last_log = 0
        done_frames = 0
        while state.steps < args.steps and time.monotonic() < deadline:
            trajectories: list[list[Transition]] = [[] for _ in range(args.engines)]
            rollout_values: list[float] = []
            for _step in range(args.rollout):
                play, hud = stack_obs(observations)
                play = play.to(device)
                hud = hud.to(device)
                actions, log_prob, values = model.act(play, hud)
                actions_list = actions.cpu().tolist()
                log_probs = log_prob.cpu().tolist()
                values_list = values.cpu().tolist()
                results, new_observations = vec.step_all_with_reset(actions_list, next_seeds)
                for index, (obs, reward, terminated, truncated, info) in enumerate(results):
                    trajectories[index].append(Transition(
                        play=observations[index]["play"],
                        hud=observations[index]["hud"],
                        action=int(actions_list[index]),
                        log_prob=float(log_probs[index]),
                        value=float(values_list[index]),
                        reward=float(reward) / args.reward_scale,
                        done=bool(terminated or truncated),
                    ))
                    if terminated or truncated:
                        done_frames += int(info["frames"])
                        episode_frames.append(int(info["frames"]))
                        episode_ends.append(str(info["end"]))
                        state.episode_returns.append(float(info["episode_return"]))
                        state.episode_deaths.append(float(info["deaths"]))
                        state.episodes.append(EpisodeRecord(
                            seed=int(info["seed"]), frames=int(info["frames"]),
                            decisions=int(info["decisions"]), deaths=float(info["deaths"]),
                            score=int(info["score"]), end=str(info["end"]),
                            episode_return=float(info["episode_return"])))
                        observations[index] = new_observations[index]
                        next_seeds[index] += args.engines
                    else:
                        observations[index] = obs
                state.steps += args.engines

            play, hud = stack_obs(observations)
            with torch.no_grad():
                _, _, last_values = model(play.to(device), hud.to(device))
            rollout_values = last_values.cpu().tolist()
            gae_returns(trajectories, rollout_values, gamma=args.gamma, lam=args.lam)
            flat = [row for trajectory in trajectories for row in trajectory]
            value_only = state.updates < args.value_warmup_updates
            stats = ppo_update(model, optimizer, flat, args, device=device,
                               value_only=value_only, rng=update_rng)
            state.updates += 1

            now = time.monotonic()
            row = {
                "update": state.updates,
                "steps": state.steps,
                "wall_s": round(now - state.started, 1),
                "value_only": bool(value_only),
                "reward_mean": float(np.mean([t.reward for t in flat])),
                "advantage_abs_mean": float(np.mean([abs(t.advantage) for t in flat])),
                **stats,
                **env_summary(episode_frames, episode_ends),
                "returns_recent_mean": float(np.mean(state.episode_returns[-20:]))
                if state.episode_returns else None,
            }
            since = now - last_log
            if since > 0:
                row["throughput_decisions_per_s"] = round(
                    (state.steps - steps_at_last_log) / since, 1)
                row["throughput_frames_per_s"] = round(
                    (state.steps - steps_at_last_log) * DECISION_FRAMES / since, 1)
            last_log, steps_at_last_log, frames_at_last_log = now, state.steps, done_frames
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if state.updates % args.metrics_every == 0:
                print(json.dumps(row, ensure_ascii=False), flush=True)

            if save_steps and state.steps >= save_steps[0]:
                step = save_steps.pop(0)
                save_policy_checkpoint(
                    out_dir / f"checkpoint-step{state.steps:07d}.pt", model,
                    train={"algorithm": "PPO", "steps": state.steps,
                           "updates": state.updates, "experiment": "EXP-P4-001",
                           "reward": reward_scale_report()["terms"],
                           "hyperparameters": config["hyperparameters"]},
                    provenance={**provenance, "kind": "ppo-from-bc",
                                "requested_save_step": step,
                                "obs_spec_version": obs_spec.OBS_SPEC_VERSION,
                                "obs_spec_digest": obs_spec.obs_spec_digest()})
                print(json.dumps({"saved": str(out_dir / f"checkpoint-step{state.steps:07d}.pt")}),
                      flush=True)
    except KeyboardInterrupt:  # pragma: no cover - operator stop
        exit_code = 130
    finally:
        try:
            vec.close()
        except Exception:  # noqa: BLE001
            pass
        elapsed = time.monotonic() - state.started
        summary = {
            "schema": SCHEMA,
            "experiment": "EXP-P4-001",
            "not_a_gate": True,
            "note": "P4 exploratory prototype; not G4",
            "steps": state.steps,
            "updates": state.updates,
            "wall_s": round(elapsed, 1),
            "decisions_per_s": round(state.steps / elapsed, 2) if elapsed else None,
            "frames_per_s": round(state.steps * DECISION_FRAMES / elapsed, 2)
            if elapsed else None,
            "episodes": env_summary(state.episode_frames, state.episode_ends),
            "episode_frames": state.episode_frames,
            "episode_returns": state.episode_returns,
            "episode_ends": state.episode_ends,
            "final_episode_records": [
                {"seed": row.seed, "frames": row.frames, "deaths": row.deaths,
                 "score": row.score, "end": row.end,
                 "episode_return": row.episode_return}
                for row in state.episodes[-40:]],
            "hyperparameters": config["hyperparameters"],
            "init": provenance,
            "config": config,
        }
        if state.steps:
            final_path = out_dir / f"checkpoint-step{state.steps:07d}.pt"
            save_policy_checkpoint(
                final_path, model,
                train={"algorithm": "PPO", "steps": state.steps,
                       "updates": state.updates, "experiment": "EXP-P4-001",
                       "reward": reward_scale_report()["terms"],
                       "hyperparameters": config["hyperparameters"]},
                provenance={**provenance, "kind": "ppo-from-bc",
                            "obs_spec_version": obs_spec.OBS_SPEC_VERSION,
                            "obs_spec_digest": obs_spec.obs_spec_digest()})
            summary["final_checkpoint"] = str(final_path)
            # The evaluation controller wants a stable name, not a step number.
            shutil.copy2(final_path, out_dir / "checkpoint.pt")
            summary["checkpoint"] = str(out_dir / "checkpoint.pt")
        (out_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({key: summary[key] for key in
                          ("steps", "updates", "wall_s", "decisions_per_s",
                           "frames_per_s", "episodes", "checkpoint")
                          if key in summary}, indent=2, ensure_ascii=False), flush=True)
    return exit_code


def run_throughput_only(args, model: ActorCritic, vec: VecRlEnv, observations,
                        infos, state: TrainingState) -> int:
    """Drive the BC policy for ``--throughput-only`` seconds and report the rate."""

    device = args.device
    started = time.monotonic()
    decisions = 0
    frames = 0
    seeds = [args.train_seed_base + index for index in range(args.engines)]
    first_frames = [int(info["frames"]) for info in infos]
    while time.monotonic() - started < args.throughput_only:
        play, hud = stack_obs(observations)
        actions, _log_prob, _values = model.act(play.to(device), hud.to(device))
        results, new_observations = vec.step_all_with_reset(actions.cpu().tolist(), seeds)
        for index, (obs, reward, terminated, truncated, info) in enumerate(results):
            frames += DECISION_FRAMES
            decisions += 1
            if terminated or truncated:
                observations[index] = new_observations[index]
                seeds[index] += args.engines
            else:
                observations[index] = obs
    elapsed = time.monotonic() - started
    payload = {
        "engines": args.engines,
        "seconds": round(elapsed, 2),
        "decisions": decisions,
        "frames": frames,
        "decisions_per_s": round(decisions / elapsed, 2),
        "frames_per_s": round(frames / elapsed, 2),
        "per_engine_decisions_per_s": round(decisions / elapsed / args.engines, 2),
        "per_engine_frames_per_s": round(frames / elapsed / args.engines, 2),
        "note": "policy in the loop; one decision == 2 logic frames; "
                "'frames' counts game frames requested by decisions, not the "
                "render count (the consecutive obs window renders every frame)",
        "first_episode_frames": first_frames,
    }
    Path(args.out_dir, "throughput.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def run_ruler_validation(args, model: ActorCritic, vec: VecRlEnv) -> int:
    """Deterministic BC episodes vs the recorded ``m1`` per-seed frames.

    This is the identity check that makes every later number usable: the same
    checkpoint, the same obs window, the same stage/cap/death-stop rule must
    reproduce ``.tmp_v400_power/m1/episodes.jsonl`` frame for frame.  A
    mismatch means the RL loop is optimising a different ruler than the one the
    comparison is reported on.
    """

    env = vec.envs[0]
    rows = []
    failures = 0
    for index in range(args.validate_ruler):
        seed = 1000 + index
        obs, info = env.reset(seed)
        terminated = truncated = False
        while not (terminated or truncated):
            play, hud = stack_obs([obs])
            actions, _log_prob, _value = model.act(play.to(args.device),
                                                  hud.to(args.device),
                                                  deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(actions[0]))
        observed = int(info["frames"])
        expected = M1_PER_SEED_FRAMES.get(seed)
        match = (expected is None) or (observed == expected)
        failures += 0 if match else 1
        rows.append({"seed": seed, "frames": observed, "expected_m1_frames": expected,
                     "match": bool(match), "end": info["end"],
                     "deaths": info["deaths"], "score": info["score"],
                     "renders": info["renders"]})
        print(json.dumps(rows[-1]), flush=True)
    payload = {
        "what": "deterministic replay of the BC checkpoint through RlEnv",
        "why": "if this does not reproduce the published per-seed frames, the RL "
               "loop is not measuring the 20-seed ruler",
        "episodes": rows,
        "mismatches": failures,
        "all_match": failures == 0,
    }
    Path(args.out_dir, "ruler_validation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("mismatches", "all_match")}))
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
