"""Reward-scale plumbing for the step-2 PPO (survive-weight override)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from touhou_rl import ppo18
from touhou_rl.reward import RewardWeights


class _CaptureEnv:
    """Stub env that records the config it was built with and ends episodes fast."""

    action_space = SimpleNamespace(sample=lambda: np.zeros(4, dtype=np.int64))

    def __init__(self):
        self.config = None
        self.closed = False
        self._step = 0

    def reset(self):
        self._step = 0
        observation = {"self_feat": np.zeros(24, dtype=np.float32)}
        observation["self_feat"][1] = 0.8
        return observation, {"state": {"bombs": 2}, "stage": 0}

    def step(self, action):
        self._step += 1
        done = self._step >= 2
        info = {
            "episode": {
                "frames": 2 * self._step,
                "deaths": 0,
                "lives": 2.0,
                "terminated": done,
                "truncated": False,
                "termination_reason": "frame stalled" if done else "",
                "stages_reached": 0,
                "score": 0,
                "graze": 0,
                "episode_reward": 0.0,
            },
            "stages_reached": 0,
            "stage": 0,
            "bullets": 2,
            "bullets_raw": {
                "position": np.array([[200.0, 200.0]], dtype=np.float32),
                "velocity": np.zeros((1, 2), dtype=np.float32),
                "half": np.ones((1,), dtype=np.float32),
            },
        }
        observation = {"self_feat": np.zeros(24, dtype=np.float32)}
        return observation, 0.0, done, False, info

    def close(self):
        self.closed = True


def _run(tmp_path, **overrides):
    captured = {}

    def factory(config):
        captured["config"] = config
        env = _CaptureEnv()
        env.config = config
        captured["env"] = env
        return env

    settings = dict(
        out_dir=tmp_path / "ppo",
        total_steps=6,
        rollout_steps=3,
        epochs=1,
        minibatch=2,
        eval_every_episodes=0,
        episode_max_steps=4,
        save_every=0,
        device="cpu",
        env_factory=factory,
    )
    settings.update(overrides)
    summary = ppo18.train(ppo18.PPO18Config(**settings))
    return captured["config"], summary


def test_survive_weight_override_reaches_the_environment(tmp_path):
    config, _summary = _run(tmp_path, survive_weight=0.1)
    assert isinstance(config.reward, RewardWeights)
    assert config.reward.survive == pytest.approx(0.1)
    # the other weights keep their defaults, so only one variable moves
    default = RewardWeights()
    assert config.reward.event == default.event
    assert config.reward.score == default.score


def test_default_run_leaves_the_reward_untouched(tmp_path):
    config, _summary = _run(tmp_path)
    assert config.reward.survive == pytest.approx(RewardWeights().survive)


def test_out_of_range_survive_weight_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="survive_weight"):
        _run(tmp_path, survive_weight=0.0)
    with pytest.raises(ValueError, match="survive_weight"):
        _run(tmp_path, survive_weight=11.0)
