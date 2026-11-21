"""v4.3.1 recheck R30b: does batching `STEP <mask> <frames>` actually save wall time?

R30 found the prefix has ~2,230 maximal runs of identical masks in 8,663 frames (mean run
3.87 frames), so the command count could drop ~3.9x. But `_StudentSession.replay` already
pipelines single-frame commands through `ask_many`, so the 7 ms/frame may be the engine
simulating a frame rather than per-command overhead. Only a measurement can tell.

Method: one session from RESET, then 12 alternating blocks of 250 frames, where a block is
either 250 x `STEP m 1` or 50 x `STEP m 5`. Alternating keeps both granularities sampling
the same stretch of the stage, which matters because the bullet count (and so the frame
cost) grows over the stage. A constant mask is fine here: this is a throughput measurement,
not a tape reproduction.

Writes only Document/Update/Alpha/v4.3.1/evidence/R30/microbench.json.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from touhou_rl.v400.teacher.engine import check_engine_binary      # noqa: E402
from touhou_rl.v420 import contracts as C                          # noqa: E402
from touhou_rl.v420 import counterfactual as cf                    # noqa: E402
from touhou_rl.v420.rl import check_data_dir                       # noqa: E402

OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R30"
BLOCK_FRAMES = 250
BLOCKS = 12
GROUP_SIZES = (1, 5, 25)


def log(message: str) -> None:
    print(f"[bench] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=12)
    args = ap.parse_args(argv)

    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    work = pathlib.Path(str(config["work_dir"]))
    template = work / "data" / "template-env"
    data_dir = work / "run" / f"bench-{os.getpid()}"
    check_data_dir(str(template), role="engine data template")
    check_data_dir(str(data_dir), role="episode data dir")

    gate = dict(check_engine_binary(str(config["engine420"]),
                                   expected=config.get("engine_sha256") or None))
    bundle = cf.load_student_policy(config)
    session = cf._StudentSession(
        engine_path=str(config["engine420"]), data_dir=data_dir,
        source_env_dir=template, stage=int(config.get("stage", 0)),
        difficulty=int(config.get("difficulty", 1)), bundle=bundle, gate=gate, log=log)

    mask = int(C.cand_to_mask(7))
    results: dict[int, list[float]] = {g: [] for g in GROUP_SIZES}
    try:
        session.start()
        session.reset_with_retry(int(args.seed))
        # let the stage warm up so the first blocks are not all bullets-free
        for _ in range(60):
            session.step_frame(mask)
        for block in range(BLOCKS):
            group = GROUP_SIZES[block % len(GROUP_SIZES)]
            commands = BLOCK_FRAMES // group
            started = time.perf_counter()
            advanced_total = 0
            for _ in range(commands):
                reply = session.client.step(mask, group)
                advanced_total += int(reply.get("advanced", 0))
            seconds = time.perf_counter() - started
            if advanced_total != BLOCK_FRAMES:
                raise RuntimeError(f"block {block}: advanced {advanced_total} frames of "
                                   f"{BLOCK_FRAMES}")
            results[group].append(seconds)
            log(f"block {block:2d} group={group} cmds={commands:3d} "
                f"{seconds:.3f}s ({seconds / BLOCK_FRAMES * 1000:.2f} ms/frame)")
    finally:
        try:
            session.close()
        except Exception:                                        # noqa: BLE001
            pass

    stats = {}
    for group, times in results.items():
        per_frame = [t / BLOCK_FRAMES for t in times]
        stats[str(group)] = {
            "blocks": len(times),
            "commands_per_block": BLOCK_FRAMES // group,
            "mean_ms_per_frame": round(statistics.mean(per_frame) * 1000, 3),
            "median_ms_per_frame": round(statistics.median(per_frame) * 1000, 3),
            "min_ms_per_frame": round(min(per_frame) * 1000, 3),
            "max_ms_per_frame": round(max(per_frame) * 1000, 3),
            "per_block_s": [round(t, 4) for t in times],
        }
    one = stats["1"]["median_ms_per_frame"]
    best_group = min(("5", "25"), key=lambda g: stats[g]["median_ms_per_frame"])
    best = stats[best_group]["median_ms_per_frame"]
    speedup = round(one / best, 3) if best else None
    out_best_group = best_group

    out = {
        "schema": "v431-recheck-r30b/1",
        "question": "does batching identical masks in `STEP <mask> <frames>` save wall time?",
        "method": {"block_frames": BLOCK_FRAMES, "blocks": BLOCKS,
                   "alternate_group_sizes": list(GROUP_SIZES),
                   "note": "alternating keeps both granularities on the same stage stretch"},
        "stats": stats,
        "verdict": {
            "median_ms_per_frame_group1": one,
            "median_ms_per_frame_best_group": best,
            "best_group": out_best_group,
            "speedup_best_vs_group1": speedup,
            "batching_helps": bool(speedup and speedup > 1.15),
            "reading": None,
        },
    }
    v = out["verdict"]
    v["reading"] = (
        f"batching to group {out_best_group} ({speedup}x fewer commands) changes wall time by "
        f"{(speedup - 1) * 100:.1f}%, so the cost is dominated by the engine simulating each "
        f"frame, not by per-command overhead"
        if speedup is not None and abs(speedup - 1.0) < 0.15 else
        f"batching to group {out_best_group} ({speedup}x fewer commands) changes wall time by "
        f"{(speedup - 1) * 100:.1f}%, so per-command overhead IS worth attacking")

    (OUT / "microbench.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"{'group':>6s} {'cmds/block':>11s} {'median ms/frame':>17s} "
          f"{'min':>8s} {'max':>8s}")
    print("-" * 56)
    for g in ("1", "5", "25"):
        s = stats[g]
        print(f"{g:>6s} {s['commands_per_block']:11d} {s['median_ms_per_frame']:17.3f} "
              f"{s['min_ms_per_frame']:8.3f} {s['max_ms_per_frame']:8.3f}")
    print()
    print(f"speedup (group{out_best_group} vs group1): {speedup}x  "
          f"batching_helps={v['batching_helps']}")
    print(f"{v['reading']}")
    print()
    print(f"wrote: {OUT / 'microbench.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
