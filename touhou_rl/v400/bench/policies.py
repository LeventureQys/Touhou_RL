"""Pixel-only P0 control policies; scores require real harness episodes."""
from __future__ import annotations
import numpy as np
from touhou_rl.v400.gym.actions import N_CAND


class RandomPolicy:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation):
        return int(self.rng.integers(N_CAND))


class StillPolicy:
    def __call__(self, observation):
        return 0
