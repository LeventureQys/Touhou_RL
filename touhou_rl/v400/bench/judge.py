"""Privileged referee isolated from the pixel-only policy interface."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from touhou_rl.v400.gym.env import F_DEATHS, F_SCORE, F_BOMBS, F_LIVES, _f32_bits


@dataclass(frozen=True)
class PixelObs:
    pixels: np.ndarray
    prev_actions: tuple[int, ...]


@dataclass(frozen=True)
class Verdict:
    deaths: float
    bombs: float
    lives: float
    score: int
    frame: int


def judge(env) -> Verdict:
    """Read referee state directly, never through the policy observation."""
    if env.shm is None:
        raise RuntimeError("referee requires a live shared block")
    fields = env.shm.state()
    frame = env.shm.header()["frame"]
    return Verdict(_f32_bits(int(fields[F_DEATHS])),
                   _f32_bits(int(fields[F_BOMBS])),
                   _f32_bits(int(fields[F_LIVES])),
                   int(fields[F_SCORE]), int(frame))


def policy_view(pixels: np.ndarray, previous: tuple[int, ...]) -> PixelObs:
    return PixelObs(pixels.copy(), tuple(previous))
