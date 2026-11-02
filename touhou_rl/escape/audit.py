"""Offline audit of already-recorded traces: does the old contract hold on disk?

Why this exists
---------------
D0 must confirm the units, frame alignment and "first snapshot" facts *from real
recorded data*, not only from reading source.  The v3.5.0/v3.6.x probe traces are
already on disk (and are gitignored, so they are read, never rewritten).  This
module parses them read-only and reports:

* the complete/incomplete split with reasons (FR-01: a missing read is named),
* the real frame advance per decision (frameskip=2 must mean 2),
* the observed player velocity magnitudes, which must land on the measured
  4.0 / 2.8284 / 2.0 / 1.4142 px per frame,
* the bullet count vs the engine counter (pool integrity),
* whether the first decision already carried a full snapshot.

It never launches the game and never writes to the trace files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .contracts import MOVEMENT_MODEL

SCHEMA = "v370-trace-audit/1"

#: Magnitudes a legal movement frame can have, in px per frame.
LEGAL_MAGNITUDES = (
    MOVEMENT_MODEL["axis"],
    MOVEMENT_MODEL["diagonal"],
    MOVEMENT_MODEL["focus_axis"],
    MOVEMENT_MODEL["focus_diagonal"],
)


def classify_velocity(vx: float, vy: float) -> str:
    """Name the movement class from the velocity *components*.

    The total speed cannot separate axis from diagonal: ``4.0`` px/frame is both
    an axis move and the magnitude of ``(2.8284, 2.8284)``.  Only the component
    pattern can, which is why the trajectory audit classifies components rather
    than ``hypot``.
    """

    def near(value: float, target: float) -> bool:
        return abs(value - target) < 2e-3

    zero_x = abs(vx) < 1e-6
    zero_y = abs(vy) < 1e-6
    if zero_x and zero_y:
        return "idle"
    if zero_x or zero_y:
        other = vy if zero_x else vx
        if near(abs(other), MOVEMENT_MODEL["axis"]):
            return "axis"
        if near(abs(other), MOVEMENT_MODEL["focus_axis"]):
            return "focus_axis"
    if near(abs(vx), MOVEMENT_MODEL["diagonal"]) and near(abs(vy), MOVEMENT_MODEL["diagonal"]):
        return "diagonal"
    if near(abs(vx), MOVEMENT_MODEL["focus_diagonal"]) and near(abs(vy), MOVEMENT_MODEL["focus_diagonal"]):
        return "focus_diagonal"
    return f"other:({vx:.3f},{vy:.3f})"


def audit_episode(path: Path | str, max_decisions: int | None = None) -> dict:
    """Scan one JSONL trace file; return counters and the first decision."""
    path = Path(path)
    decisions = 0
    complete = 0
    reasons: Counter[str] = Counter()
    advances: Counter[int] = Counter()
    magnitudes: Counter[str] = Counter()
    schemas: Counter[str] = Counter()
    pool_counts: list[int] = []
    deltas: list[int] = []
    first: dict | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            decisions += 1
            if decisions == 1:
                first = record
            if record.get("complete"):
                complete += 1
            else:
                for reason in record.get("incomplete_reasons") or ["<unnamed>"]:
                    reasons[reason] += 1
            for group, status in (record.get("schema_status") or {}).items():
                if status != "ok":
                    schemas[f"{group}:{status}"] += 1
            frame_pre = record.get("frame_pre")
            frame_post = record.get("frame_post")
            if isinstance(frame_pre, int) and isinstance(frame_post, int):
                advances[frame_post - frame_pre] += 1
            player = record.get("player") or {}
            magnitudes[classify_velocity(float(player.get("vx", 0.0)), float(player.get("vy", 0.0)))] += 1
            bullets = record.get("bullets") or {}
            if isinstance(bullets.get("count"), int):
                pool_counts.append(bullets["count"])
            if isinstance(bullets.get("count"), int) and isinstance(bullets.get("counter"), int):
                deltas.append(bullets["count"] - bullets["counter"])
            if max_decisions is not None and decisions >= max_decisions:
                break
    first_summary = None
    if first is not None:
        groups = first.get("schema_status") or {}
        first_summary = {
            "complete": bool(first.get("complete")),
            "reasons": list(first.get("incomplete_reasons") or []),
            "groups_not_ok": {group: status for group, status in groups.items() if status != "ok"},
            "groups_error": sorted(group for group, status in groups.items() if status == "error"),
            "has_player": "player" in first,
            "has_clock": "clock" in first,
            "bullet_count": (first.get("bullets") or {}).get("count"),
        }
    delta_stats = None
    if deltas:
        array = np.asarray(deltas, dtype=np.float64)
        delta_stats = {
            "n": len(deltas),
            "exact_equal": int(np.sum(array == 0)),
            "within_2": int(np.sum(np.abs(array) <= 2)),
            "exact_ratio": round(float(np.mean(array == 0)), 6),
            "within_2_ratio": round(float(np.mean(np.abs(array) <= 2)), 6),
            "min": int(array.min()),
            "p5": float(np.percentile(array, 5)),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max": int(array.max()),
        }
    return {
        "episode": path.name,
        "bytes": path.stat().st_size,
        "decisions": decisions,
        "complete": complete,
        "complete_ratio": round(complete / decisions, 6) if decisions else None,
        "incomplete_reasons": dict(reasons),
        "schema_status_issues": dict(schemas),
        "frame_advance_histogram": {str(key): value for key, value in sorted(advances.items())},
        "speed_class_histogram": dict(magnitudes),
        "max_pool_count": max(pool_counts) if pool_counts else None,
        "mean_pool_count": round(float(np.mean(pool_counts)), 3) if pool_counts else None,
        "pool_delta": delta_stats,
        "first_decision": first_summary,
    }


def audit_paths(paths: list[Path | str], max_decisions: int | None = None) -> dict:
    episodes = [audit_episode(path, max_decisions=max_decisions) for path in paths]
    total = sum(episode["decisions"] for episode in episodes)
    complete = sum(episode["complete"] for episode in episodes)
    reasons: Counter[str] = Counter()
    advances: Counter[int] = Counter()
    for episode in episodes:
        reasons.update(episode["incomplete_reasons"])
        for key, value in episode["frame_advance_histogram"].items():
            advances[int(key)] += value
    legal_classes = {"idle", "axis", "diagonal", "focus_axis", "focus_diagonal"}
    illegal = 0
    frames = 0
    for episode in episodes:
        for key, value in episode["speed_class_histogram"].items():
            frames += value
            if key not in legal_classes:
                illegal += value
    pool_exact = sum(episode["pool_delta"]["exact_equal"] for episode in episodes if episode["pool_delta"])
    pool_n = sum(episode["pool_delta"]["n"] for episode in episodes if episode["pool_delta"])
    return {
        "schema": SCHEMA,
        "episodes": episodes,
        "totals": {
            "episodes": len(episodes),
            "decisions": total,
            "complete": complete,
            "complete_ratio": round(complete / total, 6) if total else None,
            "incomplete_reasons": dict(reasons),
            "frame_advance_histogram": {str(key): value for key, value in sorted(advances.items())},
            "speed_frames": frames,
            "speed_frames_off_model": illegal,
            "off_model_ratio": round(illegal / frames, 6) if frames else None,
            "pool_delta_exact_ratio": round(pool_exact / pool_n, 6) if pool_n else None,
            "pool_delta_samples": pool_n,
        },
    }
