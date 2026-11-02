"""Offline forensics: how many escape options did the player still have before dying?

Reads recorded decision traces (each row carries a ``raw_safety_snapshot`` with the
whole bullet pool), and for every decision evaluates the 18 movement candidates over
a **long** horizon ``H`` frames under constant-velocity bullet motion and a
constant-direction player.  The metric is "escape options": how many of the 18
directions keep the player clear of every modelled bullet for the whole horizon.

The point: a policy that "waits for the bullets to arrive" will show its option
count collapsing to 0 long before the death frame, i.e. the fatal decision was made
seconds earlier.  That is what makes late dodging unrecoverable, and it is visible
without the game.

Usage:
    python tools/rl/death_forensics.py --trace <episode-001-decisions.jsonl> \
        --horizon 20 --window 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import touhou_rl.spec as spec

DIRECTIONS = np.array(
    [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)],
    dtype=np.float64,
)
SPEED, SPEED_DIAG, FOCUS_SPEED, FOCUS_SPEED_DIAG = 4.0, 2.8284271, 2.0, 1.4142135
PLAYER_HALF = 0.825
BULLET_PAD = 0.5


def _candidate_steps() -> np.ndarray:
    axis = np.repeat([SPEED, FOCUS_SPEED], 9)
    diag = np.repeat([SPEED_DIAG, FOCUS_SPEED_DIAG], 9)
    dirs = np.tile(DIRECTIONS, (2, 1))
    return dirs * np.where(np.all(dirs != 0, axis=1), diag, axis)[:, None]


STEPS = _candidate_steps()


def option_count(snapshot: dict, horizon: int) -> int:
    """Number of the 18 directions that stay clear for ``horizon`` frames."""
    player = np.asarray(snapshot["player_position"], dtype=np.float64)
    bullets = snapshot["bullets"]
    pos = np.asarray(bullets["position"], dtype=np.float64).reshape(-1, 2)
    vel = np.asarray(bullets["velocity"], dtype=np.float64).reshape(-1, 2)
    half = np.asarray(bullets["half"], dtype=np.float64).reshape(-1)
    lo = np.array([spec.MOVE_AREA_X + PLAYER_HALF, spec.MOVE_AREA_Y + PLAYER_HALF])
    hi = np.array(
        [
            spec.MOVE_AREA_X + spec.MOVE_AREA_W - PLAYER_HALF,
            spec.MOVE_AREA_Y + spec.MOVE_AREA_H - PLAYER_HALF,
        ]
    )
    if pos.size == 0:
        return 18
    radius = half + PLAYER_HALF + BULLET_PAD
    clear = np.ones(18, dtype=bool)
    p = np.clip(np.tile(player, (18, 1)), lo, hi)
    for frame in range(horizon + 1):
        bullets_now = pos + vel * frame
        distance = np.linalg.norm(p[:, None, :] - bullets_now[None, :, :], axis=-1) - radius[None, :]
        clear &= distance.min(axis=1) > 0.0
        p = np.clip(p + STEPS, lo, hi)
    return int(clear.sum())


def time_to_collision(snapshot: dict) -> float:
    """Frames until the closest bullet would touch the hurtbox (inf if receding)."""
    player = np.asarray(snapshot["player_position"], dtype=np.float64)
    bullets = snapshot["bullets"]
    pos = np.asarray(bullets["position"], dtype=np.float64).reshape(-1, 2)
    vel = np.asarray(bullets["velocity"], dtype=np.float64).reshape(-1, 2)
    half = np.asarray(bullets["half"], dtype=np.float64).reshape(-1)
    if pos.size == 0:
        return float("inf")
    radius = half + PLAYER_HALF + BULLET_PAD
    offset = pos - player
    distance = np.linalg.norm(offset, axis=-1)
    gap = distance - radius
    closing = -(offset * vel).sum(axis=-1) / np.maximum(distance, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(closing > 1e-6, gap / np.maximum(closing, 1e-9), np.inf)
    ttc = np.where(gap <= 0.0, 0.0, ttc)
    return float(np.min(ttc))


def _rows(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            snapshot = row.get("raw_safety_snapshot") or {}
            if not (snapshot.get("bullets") or {}).get("position") and not (
                snapshot.get("bullets") or {}
            ).get("position"):
                continue
            yield row


def analyse(path: str, horizon: int, window: int) -> dict:
    rows = [row for row in _rows(path) if (row.get("raw_safety_snapshot") or {}).get("bullets")]
    if not rows:
        raise SystemExit(f"no snapshots in {path}")
    options = [option_count(row["raw_safety_snapshot"], horizon) for row in rows]
    ttcs = [time_to_collision(row["raw_safety_snapshot"]) for row in rows]
    tail = options[-window:]
    # first decision (counting back from the end) at which the player was already trapped
    trapped_at = None
    for depth, value in enumerate(reversed(options)):
        if value > 0:
            trapped_at = depth
            break
    return {
        "trace": Path(path).name,
        "decisions": len(rows),
        "horizon_frames": horizon,
        "options_mean": round(float(np.mean(options)), 2),
        "options_median": float(np.median(options)),
        "trapped_decisions": int(sum(1 for value in options if value == 0)),
        "trapped_share": round(float(np.mean([value == 0 for value in options])), 3),
        "tail_options": tail,
        "last_clear_option_at": trapped_at,
        "ttc_p10": round(float(np.percentile(ttcs, 10)), 1),
        "ttc_p50": round(float(np.percentile(ttcs, 50)), 1),
        "ttc_share_under_10f": round(float(np.mean([value < 10 for value in ttcs])), 3),
        "ttc_share_under_20f": round(float(np.mean([value < 20 for value in ttcs])), 3),
        "ttc_last20": [round(value, 1) for value in ttcs[-20:]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Escape-option forensics on recorded traces")
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--horizon", type=int, default=20, help="frames to look ahead (20 = 0.33 s)")
    parser.add_argument("--window", type=int, default=60, help="tail decisions to print")
    args = parser.parse_args(argv)
    results = [analyse(path, args.horizon, args.window) for path in args.trace]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
