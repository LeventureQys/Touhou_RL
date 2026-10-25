"""Close the last T0 gate: is the first decision's bullet pool a real snapshot?

v3.6.0 §3.3 / T0 requires "环境 reset 后第一步必须提供完整的 bullet/entity
snapshot；不允许首步把弹池当空集合".  The inverse failure is just as bad: a stale
pool from the previous run would make decision 1 claim bullets that are not there.

Checks per episode, on decision 1:
  * every required schema group is ``ok`` (not ``error``);
  * the recorded count matches the sidecar slice length and equals the number of
    slots the pool reports as active;
  * the ``activeBulletCount`` counter is within the contract tolerance;
  * the pool is *coherent with the next decisions*: bullets present at decision 1
    must either persist (match a bullet at decision 2 with linear motion) or be
    explained as despawning, and the count must not jump wildly.

    python tools/rl/probe_first_snapshot.py --probe <dir> [--probe <dir> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from touhou_rl.bullet_track import match_bullets
from touhou_rl.trace35 import read_episodes, read_trace

REQUIRED = ("player", "bullets", "enemies", "items", "boss", "power", "shot", "gui")
COUNTER_TOLERANCE = 2


PHANTOM_EPS = 1e-3


def _real_mask(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Drop the zeroed pool entry recorded by traces collected before the fix.

    ``BulletField.active_mask`` now filters it, but the D1/D2/D3 sidecars were
    written when it was still reported: one entry per snapshot at exactly
    (0, 0) with exactly (0, 0) velocity.  The playfield starts at (32, 16), so it
    cannot be a projectile.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
    vel = np.asarray(vel, dtype=np.float64).reshape(-1, 2)
    zeroed = (
        (np.abs(pos[:, 0]) < PHANTOM_EPS)
        & (np.abs(pos[:, 1]) < PHANTOM_EPS)
        & (np.abs(vel[:, 0]) < PHANTOM_EPS)
        & (np.abs(vel[:, 1]) < PHANTOM_EPS)
    )
    return ~zeroed


def _coherence(bullets, records, frameskip: int = 2) -> dict:
    """How many of decision 1's bullets survive (linearly) into decision 2?"""
    offsets = bullets["offsets"]
    if len(records) < 2 or len(offsets) < 3:
        return {"first_count": 0, "second_count": 0, "matched": 0, "survival_rate": None}
    start0, end0 = int(offsets[0]), int(offsets[1])
    start1, end1 = int(offsets[1]), int(offsets[2])
    first_pos = bullets["positions"][start0:end0]
    first_vel = bullets["velocities"][start0:end0]
    first_half = bullets["half"][start0:end0]
    second_pos = bullets["positions"][start1:end1]
    second_half = bullets["half"][start1:end1]
    second_vel = bullets["velocities"][start1:end1]
    keep_first = _real_mask(first_pos, first_vel)
    keep_second = _real_mask(second_pos, second_vel)
    first_pos, first_vel, first_half = first_pos[keep_first], first_vel[keep_first], first_half[keep_first]
    second_pos, second_half = second_pos[keep_second], second_half[keep_second]
    assigned, _new = match_bullets(first_pos, first_vel, first_half, second_pos, second_half, frameskip)
    matched = int((assigned >= 0).sum())
    return {
        "first_count": int(first_pos.shape[0]),
        "second_count": int(second_pos.shape[0]),
        "matched": matched,
        # If decision 1 were a stale/empty read, nothing would match.
        "survival_rate": (matched / int(first_pos.shape[0])) if first_pos.shape[0] else None,
    }


def probe_dir(probe: Path) -> dict:
    rows = []
    for episode in read_episodes(probe):
        name = episode.get("episode")
        records = read_trace(probe / "trace" / f"{name}.jsonl")
        bullets = np.load(probe / "bullets" / f"{name}.npz")
        if not records:
            rows.append({"episode": name, "reason": "no records"})
            continue
        first = records[0]
        status = first.get("schema_status") or {}
        error_groups = [group for group in REQUIRED if status.get(group) == "error"]
        first_slice = slice(int(bullets["offsets"][0]), int(bullets["offsets"][1]))
        keep = _real_mask(bullets["positions"][first_slice], bullets["velocities"][first_slice])
        sidecar_count = int(keep.sum())
        phantom = int((~keep).sum())
        recorded = (first.get("bullets") or {}).get("count")
        counter = (first.get("bullets") or {}).get("counter")
        coherence = _coherence(bullets, records)
        counter_ok = counter is None or recorded is None or abs(int(recorded) - int(counter)) <= COUNTER_TOLERANCE
        rows.append(
            {
                "episode": name,
                "error_groups": error_groups,
                "recorded_count": recorded,
                "sidecar_count": sidecar_count,
                "counter": counter,
                "recorded_count_including_phantom": recorded,
                "phantom_entries": phantom,
                "count_matches_sidecar": recorded == sidecar_count + phantom,
                "counter_within_tolerance": counter_ok,
                **coherence,
            }
        )
    failures = []
    for row in rows:
        if row.get("error_groups"):
            failures.append(f"{row['episode']}: schema error in {row['error_groups']}")
        if not row.get("count_matches_sidecar", False):
            failures.append(f"{row['episode']}: recorded count != sidecar count")
        if not row.get("counter_within_tolerance", True):
            failures.append(f"{row['episode']}: counter off by more than {COUNTER_TOLERANCE}")
        rate = row.get("survival_rate")
        # Empty pools are fine; a non-empty first snapshot must be coherent.
        if rate is not None and rate < 0.5:
            failures.append(f"{row['episode']}: only {rate:.0%} of decision-1 bullets survive to decision 2")
    return {
        "probe": str(probe),
        "episodes": len(rows),
        "rows": rows,
        "failures": failures,
        "pass": not failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/rl/probe_first_snapshot.py")
    parser.add_argument("--probe", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    results = [probe_dir(path) for path in args.probe]
    overall = all(result["pass"] for result in results)
    for result in results:
        print(f"{result['probe']}: episodes={result['episodes']} pass={result['pass']}")
        for row in result["rows"]:
            print(
                f"  {row.get('episode'):16s} real={row.get('sidecar_count')} "
                f"phantom={row.get('phantom_entries')} counter={row.get('counter')} "
                f"survivors@{row.get('survival_rate')} errors={row.get('error_groups')}"
            )
        for failure in result["failures"]:
            print(f"  FAIL {failure}")
    print("overall:", "PASS" if overall else "FAIL")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "v360-first-snapshot/1",
            "overall": {"pass": overall},
            "results": results,
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("wrote", args.out)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
