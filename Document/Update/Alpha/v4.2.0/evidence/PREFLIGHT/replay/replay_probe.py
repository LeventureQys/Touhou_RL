"""v4.2.0 preflight: replay a recorded S1 mask slice and compare per-decision HASH.

Read-only on the corpus: the engine runs against a *copy* of the per-shard data
directory so the corpus score.dat is never written.

Usage (inside WSL, under xvfb-run):
    python .tmp_v420_replay_probe.py <shard_dir> <work_data_dir> <decisions> <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO = "/mnt/d/WorkShop/Touhou_RL"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from touhou_rl.v400.gym.env import StageStart, Th08Env  # noqa: E402

ENGINE = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"


def main() -> int:
    shard = sys.argv[1]
    work_data = sys.argv[2]
    decisions = int(sys.argv[3])
    out_path = sys.argv[4]

    with open(os.path.join(shard, "shard.json"), "r", encoding="utf-8") as handle:
        sj = json.load(handle)
    rows = []
    with open(os.path.join(shard, "labels.jsonl"), "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    rows = rows[:decisions]
    seed = int(sj["seed"])
    stage = int(sj["stage"])
    difficulty_index = int(sj["difficulty"])
    difficulty = {0: "easy", 1: "normal", 2: "hard", 3: "lunatic"}[difficulty_index]

    report = {
        "schema": "v420-replay-probe/1",
        "shard": shard,
        "seed": seed,
        "stage": stage,
        "difficulty": difficulty,
        "decisions_requested": decisions,
        "data_dir": work_data,
        "engine": ENGINE,
    }

    started = time.perf_counter()
    shm_name = f"v420replay-{os.getpid()}"
    env = Th08Env(ENGINE, work_data, shm_name=shm_name, role="env",
                  decision_frames=2, max_frames=200_000)
    per_decision = []
    try:
        with env:
            env.reset(StageStart(stage=stage, difficulty=difficulty, team=int(sj["team"]),
                                 seed=seed, resources="standard"))
            report["reset_hash"] = env.last_reset_hash
            report["expected_first_pre"] = rows[0]["hash_pre"]
            report["reset_hash_matches_first_pre"] = env.last_reset_hash == rows[0]["hash_pre"]
            frame_hashes = []
            for index, row in enumerate(rows):
                step_frames = [int(v) for v in row["step_frames"]]
                if len(step_frames) != 4:
                    report["error"] = f"row {index}: step_frames length {len(step_frames)}"
                    break
                hashes = []
                for mask in step_frames:
                    reply = env.advance(mask, 1)
                    if reply.exit or reply.advanced != 1 or not reply.hash:
                        report["error"] = (
                            f"row {index}: exit={reply.exit} advanced={reply.advanced} "
                            f"hash={reply.hash!r}"
                        )
                        break
                    hashes.append(reply.hash.strip().lower())
                    frame_hashes.append(reply.hash.strip().lower())
                if "error" in report:
                    break
                post_expected = row["hash_post"].strip().lower()
                pre_expected = row["hash_pre"].strip().lower()
                per_decision.append(
                    {
                        "decision": index,
                        "pre_expected": pre_expected,
                        "pre_actual": (
                            frame_hashes[-5] if index > 0 else report["reset_hash"]
                        ),
                        "post_expected": post_expected,
                        "post_actual": hashes[-1],
                        "post_match": hashes[-1] == post_expected,
                        "frame_hashes": hashes,
                    }
                )
    except Exception as exc:  # noqa: BLE001 - the failure is the evidence
        report["exception"] = f"{type(exc).__name__}: {exc}"
    report["wall_s"] = round(time.perf_counter() - started, 3)
    report["decisions_replayed"] = len(per_decision)
    report["frames_replayed"] = len(per_decision) * 4
    report["post_mismatch"] = [d["decision"] for d in per_decision if not d["post_match"]]
    report["post_mismatch_count"] = len(report["post_mismatch"])
    report["first_mismatch"] = (
        report["post_mismatch"][0] if report["post_mismatch"] else None
    )
    report["per_decision"] = per_decision

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_decision"},
                     ensure_ascii=False, indent=2))
    return 0 if not report.get("exception") and report["post_mismatch_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
