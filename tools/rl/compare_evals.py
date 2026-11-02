"""Collapse several ``eval-report.json`` files into one comparison table.

Why this exists
---------------
Every evaluation writes its own ``eval-report.json``; comparing policies by hand
across directories is how "it looks better" ends up in a report without a
comparable number behind it.  This tool enforces one table with one definition
per column:

* ``frames_mean`` - mean *game* frames survived (the primary metric);
* ``frames_median`` / ``frames_iqr`` - the ladder needs a robust centre and a
  spread that one lucky episode cannot move (v3.1.0 step 1 pre-registers them);
* ``deaths_mean`` - mean deaths per episode;
* ``stages_max``  - highest ``currentStage`` index reached (0 = Stage 1);
* ``terminated``  - how many episodes ended because the game ended them, versus
  hitting ``max_steps`` (a policy that merely survived the cap is not a policy
  that survived the stage).

Usage::

    python tools/rl/compare_evals.py \
        --run random=third_party/th08/build/rl/eval/random/eval-report.json \
        --run bc=third_party/th08/build/rl/eval/bc/eval-report.json \
        --run pilot=third_party/th08/build/rl/eval/pilot-v103/eval-report.json

The same numbers are printed as a Markdown table so they can be pasted into a
report without retyping.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, matching ``numpy.percentile``'s default.

    Kept here instead of importing numpy: this tool has to stay runnable on a
    bare interpreter (it is the only thing that turns reports into a table).
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower]) * (1.0 - weight) + float(ordered[upper]) * weight


def summarise(report: dict, label: str) -> dict:
    episodes = report.get("episodes", [])
    frames = [float(row.get("frames_reported", 0)) for row in episodes]
    deaths = [float(row.get("deaths", 0)) for row in episodes]
    stages = [int(row.get("stages_reached", 0)) for row in episodes]
    wall = [float(row.get("wall_seconds", 0.0)) for row in episodes]
    cut_by_cap = sum(1 for row in episodes if row.get("termination_reason") == "max_steps")
    policy = report.get("policy") or (
        "random" if report.get("checkpoint_extra", {}).get("random_policy") else "network"
    )
    return {
        "label": label,
        "policy": policy,
        "episodes": len(episodes),
        "frames_mean": round(sum(frames) / len(frames), 1) if frames else 0.0,
        "frames_median": round(percentile(frames, 50.0), 1) if frames else 0.0,
        "frames_p25": round(percentile(frames, 25.0), 1) if frames else 0.0,
        "frames_p75": round(percentile(frames, 75.0), 1) if frames else 0.0,
        "frames_iqr": round(percentile(frames, 75.0) - percentile(frames, 25.0), 1) if frames else 0.0,
        "frames_max": int(max(frames)) if frames else 0,
        "frames_min": int(min(frames)) if frames else 0,
        "deaths_mean": round(sum(deaths) / len(deaths), 2) if deaths else 0.0,
        "stages_max": max(stages) if stages else 0,
        "hit_step_cap": cut_by_cap,
        "wall_s_mean": round(sum(wall) / len(wall), 1) if wall else 0.0,
        "difficulty": report.get("difficulty"),
        "frameskip": report.get("frameskip"),
        "practice_stage": report.get("practice_stage"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare TH08 evaluation reports")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="one evaluation report; repeat for each policy",
    )
    parser.add_argument("--out", default="", help="optional JSON output path")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("at least one --run LABEL=PATH is required")
    rows = []
    for spec in args.run:
        if "=" not in spec:
            parser.error(f"--run expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append(summarise(report, label))

    header = (
        "| policy | episodes | frames_mean | frames_median | frames_iqr | frames_min-max "
        "| deaths_mean | stages_max | hit_step_cap |"
    )
    print(header)
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print(
            f"| {row['label']} | {row['episodes']} | {row['frames_mean']} | "
            f"{row['frames_median']} | {row['frames_iqr']} | "
            f"{row['frames_min']}-{row['frames_max']} | {row['deaths_mean']} | "
            f"{row['stages_max']} | {row['hit_step_cap']}/{row['episodes']} |"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
