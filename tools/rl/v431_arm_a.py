"""v4.3.1 SEG-001 arm A: D-state online learning on a dense window (A-HAZARDS).

What this implements
--------------------
Arm A of the frozen prereg (rev I), with the observation the R26 probe established is
actually available on an env-role session:

    player   : HAZARDS-independent subset of STATE (x, y, power, lives, bombs, graze)
    bullets  : the K=12 nearest LIVE bullets, 6 numbers each
               (dx, dy, vx, vy, half_width, time-to-closest-approach proxy)
    enemies  : the M=4 nearest collision-capable enemies, 5 numbers each
               (dx, dy, lethal half-x, lethal half-y, life)
    lasers   : count and the nearest laser's (dx, dy, width, damageable)
    globals  : bullet_count, laser_count, enemy_count, decision index

Red lines taken from R26 / 11-P1 §2.1:
  * `BulletRow.collision_size` is the FULL width; the lethal half-width is `half_width`.
  * `EnemyRow` lethal half extent is `hitbox_dimensions / 3`, via `lethal_half_extent`.
  * `HAZARDS` is queried once per DECISION (2 frames), not per frame: 12.8 KB per query.

Learning
--------
REINFORCE with an EMA baseline. Reward is dense: +1 for each decision survived, so the
return is the number of decisions outlived (0..decisions). Policy is a 2-layer tanh MLP
over the feature vector, sampling one of the 18 candidates per decision.

This is deliberately the *smallest* honest online learner: no value network, no PPO, no
replay. If this cannot beat `C-fair` (uniform random) the decision rule says stop and
inspect reward/exploration/optimisation before anything else.

Not in this file
----------------
* No evaluation on dev windows yet (that is a separate script; this one only trains).
* No pixel arms: `_pair_cand` refuses split-mask decisions (R22/R23).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import random
import sys
import time
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
import torch.nn as nn                                                # noqa: E402

from touhou_rl.v400.gym import protocol as P                          # noqa: E402
from touhou_rl.v400.teacher.engine import check_engine_binary         # noqa: E402
from touhou_rl.v420 import contracts as C                             # noqa: E402
from touhou_rl.v420 import counterfactual as cf                       # noqa: E402
from touhou_rl.v420.rl import check_data_dir                          # noqa: E402

N_CAND = int(C.N_MOVE_FOCUS)
K_BULLETS = 12
M_ENEMIES = 4
PLAY_W = 384.0
PLAY_H = 448.0
FEATURE_NAMES_HINT = ("player4+resources4+globals4+bullets6*12+enemies5*4+laser4 "
                      "-> fixed length")


def log(message: str) -> None:
    print(f"[armA] {message}", flush=True)


def load_dense_driver():
    path = REPO / "tools" / "rl" / "v431_dense_driver.py"
    spec = importlib.util.spec_from_file_location("v431_dense_driver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# observation
# --------------------------------------------------------------------------- #
def build_features(*, state: dict[str, Any], hazards: P.HazardFrame,
                   decision_index: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Fixed-length feature vector for arm A. Pure: no engine, no side effects."""

    px = float(state.get("player_x") or 0.0)
    py = float(state.get("player_y") or 0.0)
    feats: list[float] = [
        px / PLAY_W,
        py / PLAY_H,
        (float(state.get("player_power") or 0.0) - 64.0) / 64.0,
        float(state.get("lives_remaining") or 0.0) / 8.0,
        float(state.get("bombs_remaining") or 0.0) / 8.0,
        min(float(state.get("graze") or 0), 2000.0) / 2000.0,
        float(hazards.bullet_count) / 300.0,
        float(hazards.laser_count) / 10.0,
        float(hazards.enemy_count) / 20.0,
        decision_index / 300.0,
    ]

    def rel(pos: tuple[float, ...]) -> tuple[float, float]:
        return (float(pos[0]) - px) / PLAY_W, (float(pos[1]) - py) / PLAY_H

    live = [b for b in hazards.bullets if b.lethal]
    live.sort(key=lambda b: (float(b.position[0]) - px) ** 2 + (float(b.position[1]) - py) ** 2)
    detail: dict[str, Any] = {"live_bullets": len(live), "used_bullets": 0,
                             "used_enemies": 0}
    for b in live[:K_BULLETS]:
        dx, dy = rel(b.position)
        vx = float(b.velocity[0]) / 10.0
        vy = float(b.velocity[1]) / 10.0
        hw = (b.half_width or 0.0) / 16.0
        # closing speed toward the player, in playfield-normalised units per frame
        dist = max(math.hypot(dx, dy), 1e-6)
        closing = -(dx * vx + dy * vy) / dist
        feats.extend([dx, dy, vx, vy, hw, max(-4.0, min(4.0, closing))])
        detail["used_bullets"] += 1
    for _ in range(K_BULLETS - detail["used_bullets"]):
        feats.extend([0.0] * 6)

    ents = [e for e in hazards.enemies if e.has_collision]
    ents.sort(key=lambda e: (float(e.position[0]) - px) ** 2 + (float(e.position[1]) - py) ** 2)
    for e in ents[:M_ENEMIES]:
        dx, dy = rel(e.position)
        half = e.lethal_half_extent or (0.0, 0.0)
        feats.extend([dx, dy, half[0] / PLAY_W, half[1] / PLAY_H,
                      min(float(e.life), 5000.0) / 5000.0])
        detail["used_enemies"] += 1
    for _ in range(M_ENEMIES - detail["used_enemies"]):
        feats.extend([0.0] * 5)

    lasers = [lz for lz in hazards.lasers if getattr(lz, "in_use", False)]
    if lasers:
        lz = min(lasers, key=lambda l: (float(l.position[0]) - px) ** 2
                 + (float(l.position[1]) - py) ** 2)
        dx, dy = rel(lz.position)
        feats.extend([dx, dy, float(lz.current_width) / 64.0,
                      1.0 if getattr(lz, "damageable", False) else 0.0])
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.asarray(feats, dtype=np.float32), detail


def feature_dim() -> int:
    return 10 + 6 * K_BULLETS + 5 * M_ENEMIES + 4


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
class Policy(nn.Module):
    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, N_CAND))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--seeds", default="",
                    help="comma list of TRAIN windows to cycle per episode; blank = --seed "
                         "only. R34 showed a single-window policy overfits, so this is how "
                         "that hypothesis gets tested without touching the engine.")
    ap.add_argument("--frame", type=int, default=8664)
    ap.add_argument("--decisions", type=int, default=300, help="300 x 2 frames = 10 s")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--policy-seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--records", default="")
    ap.add_argument("--save-policy", default="", help="write the trained state_dict here")
    ap.add_argument("--load-policy", default="", help="load a state_dict before running")
    ap.add_argument("--eval-only", action="store_true",
                    help="run episodes without updating (use with --load-policy)")
    args = ap.parse_args(argv)

    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    train_seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
                   if args.seeds.strip() else [int(args.seed)])
    for s in train_seeds:
        C.assert_seed_allowed(s, "train")
    drv = load_dense_driver()
    windows: dict[int, tuple[list[int], dict[int, str], dict]] = {}
    for s in train_seeds:
        windows[s] = drv.build_masks(s, int(args.frame))
    masks, hashes, meta = windows[train_seeds[0]]

    work = pathlib.Path(str(config["work_dir"]))
    template = work / "data" / "template-env"
    data_dir = work / "run" / f"armA-{os.getpid()}"
    check_data_dir(str(template), role="engine data template")
    check_data_dir(str(data_dir), role="episode data dir")

    gate = dict(check_engine_binary(str(config["engine420"]),
                                   expected=config.get("engine_sha256") or None))
    bundle = cf.load_student_policy(config)

    torch.manual_seed(int(args.policy_seed))
    rng = random.Random(int(args.policy_seed))
    dim = feature_dim()
    policy = Policy(dim, args.hidden)
    if args.load_policy:
        state_dict = torch.load(args.load_policy, map_location="cpu")
        policy.load_state_dict(state_dict)
        log(f"loaded policy from {args.load_policy}")
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    rec_path = pathlib.Path(args.records) if args.records else None
    if rec_path:
        rec_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = 0.0
    session = cf._StudentSession(
        engine_path=str(config["engine420"]), data_dir=data_dir,
        source_env_dir=template, stage=int(config.get("stage", 0)),
        difficulty=int(config.get("difficulty", 1)), bundle=bundle, gate=gate, log=log)

    episode_records: list[dict[str, Any]] = []
    try:
        session.start()
        for ep in range(int(args.episodes)):
            t0 = time.perf_counter()
            ep_seed = train_seeds[ep % len(train_seeds)]
            masks, hashes, _meta = windows[ep_seed]
            session.reset_with_retry(ep_seed)
            mismatches = 0
            for i in range(int(args.frame) - 1):
                h, _r, _e, _g = session.step_frame(int(masks[i]))
                expected = hashes.get(i + 1)
                if expected is not None and h != expected:
                    mismatches += 1
                    break
            if mismatches:
                raise cf.CounterfactualError(f"episode {ep}: walk diverged")
            baseline_state = session.read_state()
            deaths0 = float(baseline_state.get("deaths") or 0.0)
            stage0 = int(baseline_state.get("stage") or session.stage)

            frame_now = int(args.frame) - 1
            logps: list[torch.Tensor] = []
            rewards: list[float] = []
            died_at = None
            for d in range(int(args.decisions)):
                state = session.read_state()
                hazards = P.parse_hazards_reply(session.client.request("HAZARDS"))
                feats, _detail = build_features(state=state, hazards=hazards,
                                               decision_index=d)
                logits = policy(torch.from_numpy(feats).unsqueeze(0))[0]
                dist = torch.distributions.Categorical(logits=logits)
                cand = int(dist.sample())
                logps.append(dist.log_prob(torch.tensor(cand)))
                mask = int(C.cand_to_mask(cand))
                reward = 0.0
                for _ in range(int(config["decision_frames"])):
                    h, reading, exited, game_over = session.step_frame(mask)
                    frame_now += 1
                    term = cf._terminal_of(reading, deaths_start=deaths0, stage0=stage0,
                                          hash_hex=h, frame=frame_now, exited=exited,
                                          game_over=game_over)
                    if term is not None:
                        if term.get("engine_dead"):
                            died_at = frame_now
                        break
                if died_at is not None:
                    rewards.append(0.0)
                    break
                rewards.append(1.0)

            ret = float(sum(rewards))
            episode_records.append({
                "episode": ep, "seed": int(ep_seed), "frame": int(args.frame),
                "return_decisions": ret, "died_at_offset":
                    None if died_at is None else int(died_at) - int(args.frame),
                "survived": len(rewards) >= int(args.decisions),
                "wall_s": round(time.perf_counter() - t0, 2),
            })

            if not args.eval_only and logps:
                # The naive estimator `sum(-logp) * adv` has gradient scale ~
                # decisions x advantage (300 x 300 = 9e4 here) and destroyed the policy in
                # the first 20-episode run (return fell 84 -> 38). Two fixes, both standard:
                #   * MEAN instead of SUM, so the scale does not depend on episode length;
                #   * advantage normalised by `decisions`, so it is O(1) not O(300).
                adv = (ret - baseline) / float(max(int(args.decisions), 1))
                baseline = ret if ep == 0 else 0.95 * baseline + 0.05 * ret
                loss = -(torch.stack(logps).mean() * adv)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
            if ep % 10 == 0 or ep == int(args.episodes) - 1:
                log(f"ep {ep}: return={ret:.0f} survived={len(rewards) >= int(args.decisions)} "
                    f"baseline={baseline:.1f} lr={args.lr:g} "
                    f"wall={episode_records[-1]['wall_s']}s")
            if rec_path:
                with open(rec_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(episode_records[-1], ensure_ascii=False) + "\n")
    except Exception as exc:                                          # noqa: BLE001
        log(f"FAILED: {type(exc).__name__}: {exc}")
    finally:
        try:
            session.close()
        except Exception:                                             # noqa: BLE001
            pass

    if args.save_policy and not args.eval_only:
        pathlib.Path(args.save_policy).parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), args.save_policy)
        log(f"saved policy to {args.save_policy}")

    if episode_records:
        survived = sum(1 for r in episode_records if r["survived"])
        mean_ret = sum(r["return_decisions"] for r in episode_records) / len(episode_records)
        n = len(episode_records)
        third = max(n // 3, 1)
        first = sum(r["return_decisions"] for r in episode_records[:third]) / third
        last = sum(r["return_decisions"] for r in episode_records[-third:]) / third
        log(f"episodes={n} survived={survived} "
            f"({survived / n:.1%}) mean_return_decisions={mean_ret:.1f}")
        log(f"trend: first {third} eps mean={first:.1f} -> last {third} mean={last:.1f} "
            f"({'+' if last > first else ''}{last - first:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
