"""Multi-objective reward (contract §5).

Six components, each clipped, then a weighted sum:

===========  ==========================================================
component    definition (per decision step)
===========  ==========================================================
``survive``  +1.0 while alive (terminal step gets 0)
``progress`` +1.0 per new stage reached, plus a tiny per-frame term
``score``    Δscore / 1e6, clipped
``graze``    Δgraze / 100, clipped
``resource`` Δpower/8 + ΔtimeOrbs/4 − 2·ΔbombsUsed
``event``    −10·death, +100·stage_clear
===========  ==========================================================

Keeping the vector as well as the scalar is what makes multi-objective PPO
(MOPPO) possible later without a dataset format change; the scalar is the
weighted inner product with ``RewardWeights``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import GameState

COMPONENTS = ("survive", "progress", "score", "graze", "resource", "event")


@dataclass
class RewardWeights:
    survive: float = 0.01
    progress: float = 0.02
    score: float = 1.0
    graze: float = 0.1
    resource: float = 0.1
    event: float = 1.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.survive, self.progress, self.score, self.graze, self.resource, self.event],
            dtype=np.float32,
        )

    def validate(self) -> None:
        for name in COMPONENTS:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"reward weight {name}={value} must be finite and non-negative")


@dataclass
class RewardEngine:
    """Turns consecutive ``GameState`` samples into a 6-component reward vector."""

    weights: RewardWeights = field(default_factory=RewardWeights)
    score_scale: float = 1e6
    graze_scale: float = 100.0
    power_scale: float = 8.0
    time_orb_scale: float = 4.0
    bomb_penalty: float = 2.0
    death_penalty: float = 10.0
    stage_clear_bonus: float = 100.0
    clip: float = 10.0
    _prev: GameState | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.weights.validate()

    def reset(self, state: GameState) -> None:
        self._prev = state

    def compute(self, state: GameState, terminal: bool = False) -> tuple[np.ndarray, dict]:
        prev = self._prev
        self._prev = state
        if prev is None:
            return np.zeros(len(COMPONENTS), dtype=np.float32), {}
        delta_score = max(0, state.score - prev.score)
        delta_graze = max(0, state.graze - prev.graze)
        delta_power = max(0.0, state.power - prev.power)
        delta_orbs = max(0, state.time_orbs - prev.time_orbs)
        delta_bombs_used = max(0, state.bombs_used - prev.bombs_used)
        deaths = max(0, state.deaths - prev.deaths)
        stage_advanced = state.stage > prev.stage
        stage_cleared = stage_advanced or (state.stage_clear_flag and not prev.stage_clear_flag)

        vector = np.array(
            [
                0.0 if terminal else 1.0,
                1.0 if stage_advanced else (1.0 if state.gameplay_frame > prev.gameplay_frame else 0.0) * 0.01,
                min(self.clip, delta_score / self.score_scale),
                min(self.clip, delta_graze / self.graze_scale),
                min(
                    self.clip,
                    delta_power / self.power_scale
                    + delta_orbs / self.time_orb_scale
                    - self.bomb_penalty * delta_bombs_used,
                ),
                (-self.death_penalty if deaths else 0.0)
                + (self.stage_clear_bonus if stage_cleared else 0.0),
            ],
            dtype=np.float32,
        )
        info = {
            "delta_score": delta_score,
            "delta_graze": delta_graze,
            "delta_power": round(delta_power, 3),
            "delta_bombs_used": delta_bombs_used,
            "deaths": deaths,
            "stage_advanced": stage_advanced,
        }
        return vector, info

    def scalar(self, vector: np.ndarray) -> float:
        return float(np.dot(np.clip(vector, -self.clip, self.clip), self.weights.as_array()))
