"""Feasibility probe: can bullet spawns be attributed to enemies from the trace?

`struct Bullet` has no owner field, so v3.6.0 §2.3's "future bullet generation
removed by killing this enemy" needs a reconstructed attribution.  This probe
answers three questions on real traces before any label is built:

1. does linear-prediction matching separate spawns from survivors at a tolerance
   small enough to be believable?
2. how many spawns are there per decision (is the label dense enough)?
3. what fraction of spawns lands within R px of an enemy - i.e. how much of the
   bullet stream can attribution explain at all?

    python tools/rl/probe_bullet_attribution.py --probe <dir> [--radius 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from touhou_rl.bullet_track import track_episode
from touhou_rl.trace35 import read_episodes, read_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/rl/probe_bullet_attribution.py")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--radius", type=float, default=32.0)
    parser.add_argument("--tolerance", type=float, default=1.5)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=0, help="limit (0 = all)")
    args = parser.parse_args(argv)

    probe = Path(args.probe)
    names = [ep["episode"] for ep in read_episodes(probe)]
    if args.episodes:
        names = names[: args.episodes]

    started = time.perf_counter()
    rows = []
    for name in names:
        records = read_trace(probe / "trace" / f"{name}.jsonl")
        bullets = np.load(probe / "bullets" / f"{name}.npz")
        stats = track_episode(
            bullets,
            records,
            frameskip=args.frameskip,
            tolerance=args.tolerance,
            radius=args.radius,
        )
        stats["episode"] = name
        rows.append(stats)
        print(
            f"{name}: decisions={stats['decisions']} spawns/dec={stats['spawns_per_decision_mean']:.2f} "
            f"(p95={stats['spawns_per_decision_p95']:.0f}) attribution={stats['attribution_rate']} "
            f"mean_dist={stats['mean_attribution_distance']}"
        )

    spawns = sum(row["spawns"] for row in rows)
    attributed = sum(row["attributed"] for row in rows)
    decisions = sum(row["decisions"] for row in rows)
    matched = sum(row["matched_pairs"] for row in rows)
    summary = {
        "schema": "v360-bullet-attribution-probe/1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "probe": str(probe),
        "episodes": len(rows),
        "config": {
            "radius_px": args.radius,
            "tolerance_px": args.tolerance,
            "frameskip": args.frameskip,
        },
        "totals": {
            "decisions": decisions,
            "matched_pairs": matched,
            "spawns": spawns,
            "attributed": attributed,
            "attribution_rate": attributed / spawns if spawns else None,
            "spawns_per_decision": spawns / decisions if decisions else None,
        },
        "per_episode": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=1))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
