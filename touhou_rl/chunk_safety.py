"""Offline conservative movement-chunk safety model.

This is a geometric *model prediction*, not a guarantee of game safety.  It
models only linear bullet motion and the player hurtbox; invincibility,
spawning, lasers, and other non-linear game effects are intentionally unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from . import spec

_SCHEMA = "v200-raw-safety-snapshot/1"
_DIRECTIONS = np.asarray(
    [[0, 0], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]],
    dtype=np.float64,
)
_DURATIONS = (1, 3, 5)


@dataclass(frozen=True)
class ChunkSafetyConfig:
    frameskip: int = 2
    speed: float = 4.0
    speed_diag: float = 2.8284
    focus_speed: float = 2.0
    focus_speed_diag: float = 1.4142
    player_half: float = 0.825
    bullet_pad: float = 0.5
    clearance_cap: float = 60.0

    def validate(self) -> None:
        if isinstance(self.frameskip, bool) or not isinstance(self.frameskip, (int, np.integer)) or self.frameskip < 1:
            raise ValueError("frameskip must be a positive integer")
        for name in ("speed", "speed_diag", "focus_speed", "focus_speed_diag", "player_half", "bullet_pad", "clearance_cap"):
            value = getattr(self, name)
            if not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if name in ("player_half", "bullet_pad") and float(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            if name in ("speed", "speed_diag", "focus_speed", "focus_speed_diag", "clearance_cap") and float(value) <= 0:
                raise ValueError(f"{name} must be positive")


def _invalid() -> dict[str, np.ndarray]:
    return {
        "clearance": np.full((18, 3), np.nan, dtype=np.float32),
        "model_valid": np.zeros((18, 3), dtype=np.uint8),
        "collision_predicted": np.zeros((18, 3), dtype=np.uint8),
        "world_risk_valid": np.zeros((18, 3), dtype=np.uint8),
        "duration_class": np.full(18, -1, dtype=np.int8),
        "duration_valid": np.zeros(18, dtype=np.uint8),
    }


def _array(value: Any, shape_tail: int, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.size == 0:
        return np.empty((0, shape_tail), dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != shape_tail:
        raise ValueError(f"{name} must have shape (N,{shape_tail})")
    return a


def _segment_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    """Minimum distance between two points moving linearly on one frame."""
    r0 = a0 - b0
    dr = (a1 - a0) - (b1 - b0)
    den = float(np.dot(dr, dr))
    if den > 0.0:
        t = float(np.clip(-np.dot(r0, dr) / den, 0.0, 1.0))
    else:
        t = 0.0
    return float(np.linalg.norm(r0 + t * dr))


def _valid_snapshot(snapshot: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not isinstance(snapshot, dict) or snapshot.get("schema") != _SCHEMA:
        return None
    if snapshot.get("complete") is not True:
        return None
    if snapshot.get('player_state') is not None and snapshot['player_state'] not in (0,3):
        return None
    try:
        player = np.asarray(snapshot["player_position"], dtype=np.float64)
        if player.shape != (2,):
            return None
        bullets = snapshot["bullets"]
        if not isinstance(bullets, dict):
            return None
        pos = _array(bullets["position"], 2, "position")
        vel = _array(bullets["velocity"], 2, "velocity")
        half = np.asarray(bullets["half"], dtype=np.float64)
        if half.ndim != 1 or len(half) != len(pos) or len(vel) != len(pos):
            raise ValueError("bullet arrays must have matching lengths")
        if np.any(half < 0):
            raise ValueError("bullet half extents must be non-negative")
        frame = np.asarray(snapshot["frame"])
        stage = np.asarray(snapshot["stage"])
        if frame.ndim != 0 or stage.ndim != 0:
            return None
        if isinstance(snapshot["frame"], (bool, np.bool_)) or isinstance(snapshot["stage"], (bool, np.bool_)):
            return None
        if not isinstance(snapshot["frame"], (int, np.integer)) or not isinstance(snapshot["stage"], (int, np.integer)):
            return None
        if int(snapshot["frame"]) < 0 or int(snapshot["stage"]) < 0:
            return None
        if not np.all(np.isfinite(player)) or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)) or not np.all(np.isfinite(half)):
            return None
        if not np.all(np.isfinite(frame)) or not np.all(np.isfinite(stage)):
            return None
        return player, pos, vel, half
    except (KeyError, TypeError, OverflowError):
        return None


def predict(snapshot: dict, config: ChunkSafetyConfig = ChunkSafetyConfig()) -> dict[str, np.ndarray]:
    """Predict safety for 18 ``focus*9`` movement candidates and durations 1,3,5.

    ``clearance`` is surface distance (positive means no modeled collision),
    capped at ``config.clearance_cap``.  Values are based on all supplied raw
    bullets and continuous relative linear motion, including time zero.
    """
    if not isinstance(config, ChunkSafetyConfig):
        raise TypeError("config must be ChunkSafetyConfig")
    config.validate()
    result = _invalid()
    parsed = _valid_snapshot(snapshot)
    if parsed is None:
        return result
    player, pos, vel, half = parsed
    # The movement area is explicitly inset by the player's half extent.
    lo = np.array([spec.MOVE_AREA_X + config.player_half, spec.MOVE_AREA_Y + config.player_half])
    hi = np.array([spec.MOVE_AREA_X + spec.MOVE_AREA_W - config.player_half, spec.MOVE_AREA_Y + spec.MOVE_AREA_H - config.player_half])
    origin = np.clip(player, lo, hi)
    # All 18 forced trajectories, one frame at a time; no beam continuation.
    axis = np.repeat([config.speed, config.focus_speed], 9)
    diag = np.repeat([config.speed_diag, config.focus_speed_diag], 9)
    dirs = np.tile(_DIRECTIONS, (2, 1))
    step = dirs * np.where(np.all(dirs != 0, axis=1), diag, axis)[:, None]
    p0 = np.tile(origin, (18, 1))
    minimum = np.full(18, config.clearance_cap, dtype=np.float64)
    ends = {d * config.frameskip: i for i, d in enumerate(_DURATIONS)}
    radius = half + config.player_half + config.bullet_pad
    for frame in range(1, 5 * config.frameskip + 1):
        p1 = np.clip(p0 + step, lo, hi)
        if len(pos):
            r0 = p0[:, None, :] - (pos + vel * (frame - 1))[None, :, :]
            dr = (p1 - p0)[:, None, :] - vel[None, :, :]
            den = np.sum(dr * dr, axis=-1)
            fraction = np.clip(np.divide(-np.sum(r0 * dr, axis=-1), den,
                                         out=np.zeros_like(den), where=den > 0), 0, 1)
            clear = np.linalg.norm(r0 + fraction[..., None] * dr, axis=-1) - radius[None, :]
            minimum = np.minimum(minimum, clear.min(axis=1))
        if frame in ends:
            col = ends[frame]
            result["clearance"][:, col] = minimum
            result["collision_predicted"][:, col] = minimum <= 0
        p0 = p1
    result["model_valid"][:] = 1
    result["duration_valid"][:] = 1
    result["duration_class"][:] = np.max(np.where(result["clearance"] > 0, np.arange(3), -1), axis=1)
    return result


__all__ = ["ChunkSafetyConfig", "predict"]
