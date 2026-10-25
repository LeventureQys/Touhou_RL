"""Measure the real per-frame player speed from recorded traces.

The pilot's constants (4.0 / 2.8284 / 2.0 / 1.4142 px per frame) come from a
team-0 (Border) calibration.  This script re-derives the speeds from any trace
directory, per (direction, focus) bucket, using the recorded velocity field —
which is the game's own number, so it measures the game rather than the pilot's
assumption about it.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def measure(trace_path: Path) -> dict:
    buckets: dict[tuple[int, int, int], list[tuple[float, float]]] = defaultdict(list)
    for line in trace_path.open(encoding="utf-8"):
        record = json.loads(line)
        action = record.get("action") or [0, 0, 0, 0]
        move, focus = int(action[0]), int(action[2])
        player = record.get("player") or {}
        vx, vy = player.get("vx"), player.get("vy")
        if vx is None or vy is None:
            continue
        from touhou_rl.pilot import _MOVE_VECTORS

        ux, uy = _MOVE_VECTORS[move]
        buckets[(int(ux), int(uy), focus)].append((float(vx), float(vy)))
    rows = {}
    for key, values in sorted(buckets.items()):
        if len(values) < 20:
            continue
        mx = sum(v[0] for v in values) / len(values)
        my = sum(v[1] for v in values) / len(values)
        rows[f"{key[0]},{key[1]},focus={key[2]}"] = {
            "n": len(values),
            "mean_vx": round(mx, 4),
            "mean_vy": round(my, 4),
            "mean_speed": round(math.hypot(mx, my), 4),
        }
    axis = {}
    for name, key in (("axis", (1, 0)), ("diag", (1, 1))):
        for focus in (0, 1):
            row = rows.get(f"{key[0]},{key[1]},focus={focus}")
            axis[f"{name}_focus{focus}"] = None if row is None else row["mean_speed"]
    return {"buckets": rows, "summary": axis}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.traces:
        result = measure(path)
        print(f"== {path.name}")
        for key, row in result["buckets"].items():
            print(f"   dir {key:18s} n={row['n']:6d} v=({row['mean_vx']:+.3f},{row['mean_vy']:+.3f}) |v|={row['mean_speed']:.4f}")
        print(f"   summary: {result['summary']}")


if __name__ == "__main__":
    main()
