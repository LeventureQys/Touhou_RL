"""v4.3.1 recheck R30: is there a free speed-up in the prefix replay?

Every episode pays ~70 s, and ~68 s of it is replaying 8,663 frames of the same prefix. Two
facts make a cheap win plausible:

  * the harness accepts `STEP <mask> <frames>`, so identical consecutive masks can be
    batched into one command;
  * `_StudentSession.replay` sends `STEP <mask> 1` once per frame.

This script only measures the tape's run structure offline (no engine): how many maximal
runs of identical masks exist in the first 8,663 frames, and what the command count would
drop to. Whether that actually saves wall time is a separate engine measurement, because
`replay` already pipelines its commands through `ask_many` -- if the 7 ms/frame is the
engine's own simulation cost, batching changes nothing.

Writes only Document/Update/Alpha/v4.3.1/evidence/R30/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
S1C = REPO / ".tmp_v410_s1c" / "d1" / "shards"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R30"
FRAMES = 8663
SEEDS = (11, 12, 13, 17)


def runs_for(seed: int, limit: int) -> dict:
    path = S1C / f"d1-s0d1-n3300-seed{seed}-shard" / "labels.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    masks: list[int] = []
    for r in rows:
        if int(r["decision_frame"]) >= limit:
            break
        masks.extend(int(x) for x in r["step_frames"])
    masks = masks[:limit]
    runs = 1
    lengths: list[int] = []
    current = 1
    for i in range(1, len(masks)):
        if masks[i] == masks[i - 1]:
            current += 1
        else:
            lengths.append(current)
            runs += 1
            current = 1
    lengths.append(current)
    hist: dict[int, int] = {}
    for n in lengths:
        hist[n] = hist.get(n, 0) + 1
    return {
        "seed": seed,
        "frames": len(masks),
        "runs": runs,
        "commands_per_frame": round(runs / len(masks), 4),
        "command_reduction_factor": round(len(masks) / runs, 2),
        "run_length_mean": round(sum(lengths) / len(lengths), 2),
        "run_length_max": max(lengths),
        "run_length_histogram": {str(k): hist[k] for k in sorted(hist)},
    }


def main() -> int:
    rows = [runs_for(s, FRAMES) for s in SEEDS]
    out = {
        "schema": "v431-recheck-r30/1",
        "question": ("can the 8,663-frame prefix be replayed with fewer commands by batching "
                     "identical consecutive masks via `STEP <mask> <frames>`?"),
        "offline": {"frames": FRAMES, "per_seed": rows},
        "summary": {
            "best_reduction_factor": max(r["command_reduction_factor"] for r in rows),
            "worst_reduction_factor": min(r["command_reduction_factor"] for r in rows),
            "mean_run_length": round(sum(r["run_length_mean"] for r in rows) / len(rows), 2),
        },
        "caveat": ("command count is not wall time: `replay` already batches through "
                   "ask_many, so if the 7 ms/frame is the engine simulating a frame, "
                   "fewer commands will not help. Needs an engine microbenchmark."),
        "engine_has_save_restore": False,
        "save_state_options": [
            {"option": "fork-server in rlharness", "how": ("hold one engine at the dense "
                                                           "frame and fork() a child per "
                                                           "episode, giving each child its own "
                                                           "socket + shm"),
             "touches": "third_party/th08-rl (C++, needs approval)", "speedup_estimate": "~25x"},
            {"option": "CRIU checkpoint/restore", "how": "external userspace checkpoint",
             "touches": "none, but needs CRIU + privileges", "speedup_estimate": "unknown"},
            {"option": "in-game save", "how": "ReplayManager save", "touches": "game code",
             "speedup_estimate": "not applicable: TH08 cannot restore mid-stage"},
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prefix_runs.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"prefix of {FRAMES} frames, maximal runs of identical masks:")
    print(f"{'seed':>5s} {'runs':>6s} {'mean run':>9s} {'max run':>8s} {'reduction':>10s}")
    print("-" * 44)
    for r in rows:
        print(f"{r['seed']:5d} {r['runs']:6d} {r['run_length_mean']:9.2f} "
              f"{r['run_length_max']:8d} {r['command_reduction_factor']:9.2f}x")
    print()
    print(f"best reduction {out['summary']['best_reduction_factor']}x, "
          f"mean run length {out['summary']['mean_run_length']} frames")
    print(f"engine has save/restore: {out['engine_has_save_restore']}")
    print()
    print(f"wrote: {OUT / 'prefix_runs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
