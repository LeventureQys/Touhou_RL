"""Loop-level tests for the step-2 PPO trainer (stub environment, no game)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from touhou_rl import entity_policy, ppo18


class _StubEnv:
    """Minimal Th08Env stand-in: ends every episode after a few steps."""

    action_space = SimpleNamespace(sample=lambda: np.zeros(4, dtype=np.int64))

    def __init__(self, steps_to_end: int = 4, bullets: int = 3):
        self.steps_to_end = steps_to_end
        self.bullets = bullets
        self.actions: list[np.ndarray] = []
        self.closed = False
        self._step = 0

    def reset(self):
        self._step = 0
        observation = {"self_feat": np.zeros(24, dtype=np.float32)}
        observation["self_feat"][0] = 0.4
        observation["self_feat"][1] = 0.7
        return observation, {"state": {"bombs": 2}, "stage": 0, "stages_reached": 0}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.int64).copy())
        self._step += 1
        done = self._step >= self.steps_to_end
        positions = np.tile(np.array([[210.0, 210.0]], dtype=np.float32), (self.bullets, 1))
        info = {
            "episode": {
                "frames": 2 * self._step,
                "deaths": 0,
                "lives": 2.0,
                "terminated": done,
                "truncated": False,
                "termination_reason": "frame stalled" if done else "",
                "stages_reached": 0,
                "score": 10,
                "graze": 1,
                "episode_reward": 0.1,
            },
            "stages_reached": 0,
            "stage": 0,
            "frame": 2 * self._step,
            "bullets": self.bullets,
            "bullets_raw": {
                "position": positions,
                "velocity": np.zeros((self.bullets, 2), dtype=np.float32),
                "half": np.ones((self.bullets,), dtype=np.float32),
            },
        }
        observation = {"self_feat": np.zeros(24, dtype=np.float32)}
        return observation, 0.1, done, False, info

    def close(self):
        self.closed = True


def test_gae_matches_the_recursive_definition():
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.4, 0.3, 0.0], dtype=np.float32)
    dones = np.zeros(3, dtype=np.float32)
    advantages, returns = ppo18._gae(rewards, values, dones, gamma=0.5, lam=1.0)
    assert returns[2] == pytest.approx(1.0, rel=1e-6)
    assert returns[0] - values[0] == pytest.approx(advantages[0], rel=1e-6)
    assert advantages.shape == (3,) and returns.shape == (3,)


def test_a_done_step_does_not_bootstrap():
    rewards = np.array([1.0], dtype=np.float32)
    values = np.array([0.5, 100.0], dtype=np.float32)  # the 100 must be ignored
    advantages, returns = ppo18._gae(rewards, values, np.array([1.0], dtype=np.float32), gamma=0.9, lam=1.0)
    assert returns[0] == pytest.approx(1.0, rel=1e-6)


def test_ppo18_runs_a_full_cycle_on_a_stub_env(tmp_path):
    env = _StubEnv(steps_to_end=4, bullets=3)
    summary = ppo18.train(
        ppo18.PPO18Config(
            out_dir=tmp_path / "ppo",
            total_steps=24,
            rollout_steps=8,
            epochs=1,
            minibatch=4,
            eval_every_episodes=2,
            eval_episodes=1,
            episode_max_steps=8,
            save_every=16,
            device="cpu",
            env_factory=lambda config: env,
        )
    )
    assert summary["steps"] == 24
    assert summary["updates"] > 0
    assert summary["episodes_run"] >= 3
    assert summary["evaluations"], "the evaluation hook must fire"
    assert summary["evaluations"][0]["episodes"] == 1
    assert summary["last_evaluation"]["frames_mean"] > 0
    assert env.closed is True

    metrics = (tmp_path / "ppo" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    kinds = [json.loads(line)["kind"] for line in metrics]
    assert "episode" in kinds and "update" in kinds and "evaluation" in kinds
    episode_rows = [json.loads(line) for line in metrics if json.loads(line)["kind"] == "episode"]
    assert {"still_rate", "move_histogram", "x_p10", "x_p50", "x_p90"} <= set(episode_rows[0])
    assert len(episode_rows[0]["move_histogram"]) == 9
    assert (tmp_path / "ppo" / "last.pt").exists()
    written = json.loads((tmp_path / "ppo" / "summary.json").read_text(encoding="utf-8"))
    assert written["steps"] == 24


def test_ppo18_kl_anchor_runs_and_is_logged(tmp_path):
    model = entity_policy.EntityPolicy(entity_policy.EntityPolicyConfig(trunk_hidden=32, object_hidden=16))
    anchor = entity_policy.save_checkpoint(tmp_path / "anchor.pt", model)
    env = _StubEnv(steps_to_end=3, bullets=3)
    summary = ppo18.train(
        ppo18.PPO18Config(
            out_dir=tmp_path / "ppo",
            init_checkpoint=anchor,
            anchor_checkpoint=anchor,
            kl_coef=0.5,
            total_steps=12,
            rollout_steps=6,
            epochs=1,
            minibatch=3,
            eval_every_episodes=0,
            episode_max_steps=6,
            save_every=0,
            device="cpu",
            env_factory=lambda config: env,
        )
    )
    assert summary["updates"] > 0
    metrics = (tmp_path / "ppo" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    updates = [json.loads(line) for line in metrics if json.loads(line)["kind"] == "update"]
    assert updates and "kl_anchor" in updates[-1]
    assert updates[-1]["kl_anchor"] >= 0.0
