"""Acceptance driver for ``TrajectoryStart`` (task book T3 step 1, EXP-P3-012a).

It answers four questions with real engine output, and never invents a number:

1. **Does the take-over land on the recorded state?**  ``--points N`` random
   take-over points, each replayed from the recorded ``RESET`` line in its **own
   engine process**; the hash at every recorded checkpoint the tape passes, and
   at the take-over frame itself, must equal the recorded hash.
2. **Does one process support many take-overs?**  ``--consecutive M`` take-overs
   in **one** process, using the T7.1 in-process ``RESET``, each verified the same
   way, each timed.  The per-take-over ``RESET`` cost is compared with the
   process-restart cost measured in (1).
3. **Is a multi-frame ``STEP`` block the same as per-frame ``STEP``s?**  One point
   is replayed twice 鈥?default batching and ``--batch-frames 1`` 鈥?and the full
   checkpoint hash lists are compared.
4. **Does the rendered (env) role agree with the shadow role?**
   ``--env-cross-check K`` re-runs K of the points on an ``env``-role engine under
   Xvfb, where the D1/D0 production pixel path lives.
5. **What does one episode cost under each regime?**  ``--cold-restarts K``
   measures the "one process per episode" fixed cost (fresh data dir, spawn,
   ``RESET`` to a controllable frame, teardown) so the in-process ``RESET`` can be
   compared with the 6-7 s per-episode restart recorded in EXP-P4-001 §7.1.  It is
   measured here rather than quoted, and the historical figure is kept as a
   separate column.

Fail-closed: any hash mismatch, any abort, any engine that had to be restarted, a
short run, or a divergence between the two batching modes yields ``status !=
"pass"`` and exit code 3.  Nothing is written to a report path that already
exists, and no existing directory is reused.

Run inside WSL::

    /root/v400-venv/bin/python \\
        /mnt/d/WorkShop/Touhou_RL/touhou_rl/v400/gym/trajectory_verify.py \\
        --binary /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \\
        --data-template /root/v400-data/game \\
        --shards-root /mnt/d/WorkShop/Touhou_RL/.tmp_v400_d0/d0/shards \\
        --run-root /root/v400-traj/run1 \\
        --out /mnt/d/WorkShop/Touhou_RL/.tmp_v400_traj/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    # ``python path/to/trajectory_verify.py`` 鈥?make the repo importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from touhou_rl.v400.gym.env import Th08Env
from touhou_rl.v400.gym.inprocess_reset import create_data_dir
from touhou_rl.v400.gym.trajectory import (
    RecordedTrajectory,
    TakeoverReport,
    TrajectoryStartError,
    load_d0_shards,
    replay_to_frame,
    sample_target_frame,
)

SCHEMA = "v400-trajectory-start/1"
EXPERIMENT = "EXP-P3-012a"

#: T3 step 1 asks for 20 random take-over points, all hashes equal.
REQUIRED_POINTS = 20
#: T7.1/T3 ask for at least 10 take-overs inside one process.
REQUIRED_CONSECUTIVE = 10

#: Stage 1 Normal's boss section starts here, from the teacher run report
#: ``.tmp_v400_clear_stage0.jsonl`` (``report.boss_first.frame = 2936``).  The D0
#: shards do not store a boss marker, so this single measured boundary is used for
#: the sampling bias of every shard.  It only decides *where* the take-over points
#: are drawn; the verification itself is per-frame and independent of it.
BOSS_START_FRAME = 2936
BOSS_START_SOURCE = ".tmp_v400_clear_stage0.jsonl report.boss_first.frame"

#: Default sampling bias (T3: "鍋忓悜鍚庡崐娈典笌 Boss 娈?).
DEFAULT_BIAS = 0.65
DEFAULT_BOSS_WEIGHT = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def median(values: list[float]) -> float | None:
    """Median of ``values`` rounded to microseconds; ``None`` when empty."""
    return round(statistics.median(values), 6) if values else None


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _maybe_reexec_under_xvfb(needed: bool) -> None:
    """The env role needs a private Xvfb: WSLg's ``DISPLAY=:0`` hangs before HELLO."""
    if not needed or os.environ.get("V400_TRAJ_XVFB") == "1":
        return
    anchor = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path(__file__)
    argv = ["xvfb-run", "-a", sys.executable, str(anchor), *sys.argv[1:]]
    os.execvpe("xvfb-run", argv, dict(os.environ, V400_TRAJ_XVFB="1"))


def open_env(binary: Path, data_dir: Path, *, index: int, role: str) -> tuple[Th08Env, float]:
    """Start one engine process; return it and how long the start-up took."""
    shm = f"v400traj-{os.getpid()}-{role}-{index}"
    started = time.perf_counter()
    env = Th08Env(binary, data_dir, shm_name=shm, role=role)
    env.__enter__()
    return env, time.perf_counter() - started


def describe(report: TakeoverReport, *, role: str, index: int, traj: RecordedTrajectory,
             process_start_s: float | None, total_s: float,
             prepare_s: float | None = None) -> dict:
    """One take-over point, as stored in the report."""
    record = report.as_record()
    record.update({
        "index": index,
        "role": role,
        "corpus_source": traj.source,
        "frames_in_tape": traj.frames,
        "fraction_of_tape": round(report.target_frame / traj.frames, 4),
        "quartile": min(3, int(4 * report.target_frame / max(traj.frames, 1))),
        "late_half": report.target_frame >= (traj.frames + 1) // 2,
        "boss_section": report.target_frame >= BOSS_START_FRAME,
        "process_start_s": None if process_start_s is None else round(process_start_s, 6),
        "prepare_s": None if prepare_s is None else round(prepare_s, 6),
        "total_s": round(total_s, 6),
    })
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data-template", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="fresh directory for the new data dirs")
    parser.add_argument("--out", type=Path, required=True, help="JSON report (must not exist)")
    parser.add_argument("--points", type=int, default=REQUIRED_POINTS)
    parser.add_argument("--consecutive", type=int, default=REQUIRED_CONSECUTIVE)
    parser.add_argument("--env-cross-check", type=int, default=2,
                        help="how many of the points to re-run on an env-role engine")
    parser.add_argument("--sampling-seed", type=int, default=20261110)
    parser.add_argument("--bias", type=float, default=DEFAULT_BIAS)
    parser.add_argument("--boss-weight", type=float, default=DEFAULT_BOSS_WEIGHT)
    parser.add_argument("--boss-start", type=int, default=BOSS_START_FRAME)
    parser.add_argument("--batch-frames", type=int, default=64)
    parser.add_argument("--cold-restarts", type=int, default=3,
                        help="how many one-process-per-episode restarts to time (0 disables)")
    args = parser.parse_args(argv)

    if args.out.exists():
        print(f"refusing to overwrite an existing report: {args.out}", file=sys.stderr)
        return 2
    if sys.platform == "win32":
        print("blocked: the harness binary is a Linux ELF; run this inside WSL", file=sys.stderr)
        return 3
    if not args.binary.is_file():
        print(f"binary missing: {args.binary}", file=sys.stderr)
        return 2
    if args.run_root.exists() or args.run_root.is_symlink():
        print(f"refusing to reuse an existing run root: {args.run_root}", file=sys.stderr)
        return 2
    if args.points < 1 or args.consecutive < 1:
        print("points and consecutive must both be positive", file=sys.stderr)
        return 2

    _maybe_reexec_under_xvfb(args.env_cross_check > 0)

    template = args.data_template
    if not (template / "th08.dat").is_file():
        print(f"data template missing th08.dat: {template}", file=sys.stderr)
        return 2

    binary_info = {"path": str(args.binary), "sha256": sha256_file(args.binary),
                   "size": args.binary.stat().st_size}
    harness_info: dict[str, object] = {"repo": None, "head": None, "dirty": None}
    for candidate in args.binary.resolve().parents:
        if (candidate / "src" / "rlharness").is_dir():
            harness_info.update({"repo": str(candidate),
                                 "head": _git(candidate, "rev-parse", "HEAD"),
                                 "dirty": bool(_git(candidate, "status", "--porcelain",
                                                    "--untracked-files=no"))})
            break

    started = time.perf_counter()
    args.run_root.mkdir(parents=True)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "task": "T3 step 1: TrajectoryStart hands over at the recorded frame, verified",
        "binary": binary_info,
        "harness": harness_info,
        "corpus": {"root": str(args.shards_root), "boss_start": args.boss_start,
                   "boss_start_source": BOSS_START_SOURCE},
        "sampling": {"seed": args.sampling_seed, "bias": args.bias,
                     "boss_weight": args.boss_weight, "points": args.points,
                     "consecutive": args.consecutive, "batch_frames": args.batch_frames},
        "environment": {"host_load_1m": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
                        "nproc": os.cpu_count(), "xvfb": os.environ.get("V400_TRAJ_XVFB") == "1"},
        "phases": {},
        "reasons": [],
        "warnings": [],
    }

    try:
        trajectories = load_d0_shards(args.shards_root)
    except TrajectoryStartError as error:
        print(f"corpus: {error}", file=sys.stderr)
        return 3
    report["corpus"]["shards"] = [
        {"traj_id": traj.traj_id, "seed": traj.seed, "stage_index": traj.stage_index,
         "difficulty": traj.difficulty, "team": traj.team, "resources": traj.resources,
         "frames": traj.frames, "checkpoints": len(traj.hashes),
         "checkpoint_range": [min(traj.hashes), max(traj.hashes)],
         "masks_sha256": hashlib.sha256(
             ",".join(f"{mask:04x}" for mask in traj.masks).encode("ascii")).hexdigest()}
        for traj in trajectories
    ]

    rng = random.Random(args.sampling_seed)
    picks: list[tuple[RecordedTrajectory, int]] = []
    seen: set[tuple[str, int]] = set()
    while len(picks) < args.points:
        traj = trajectories[rng.randrange(len(trajectories))]
        target = sample_target_frame(rng, traj.checkpoint_frames, bias=args.bias,
                                     boss_start=args.boss_start,
                                     boss_weight=args.boss_weight)
        if (traj.traj_id, target) in seen:
            continue
        seen.add((traj.traj_id, target))
        picks.append((traj, target))

    # -- phase A: one fresh process per take-over -------------------------
    role = "shadow"
    phase_a: list[dict] = []
    reasons: list[str] = []
    for index, (traj, target) in enumerate(picks, start=1):
        data_dir = args.run_root / f"dt-a-{index:03d}"
        prepare_started = time.perf_counter()
        create_data_dir(template, data_dir)
        prepare_s = time.perf_counter() - prepare_started
        point_started = time.perf_counter()
        try:
            env, process_start_s = open_env(args.binary, data_dir, index=index, role=role)
        except Exception as error:                                 # noqa: BLE001
            phase_a.append({"index": index, "traj_id": traj.traj_id, "seed": traj.seed,
                            "target_frame": target, "error": f"{type(error).__name__}: {error}"})
            reasons.append(f"point {index}: engine did not start: {error}")
            continue
        try:
            takeover = replay_to_frame(env, traj.start(target), batch_frames=args.batch_frames)
            restarts = env.restart_count
            record = describe(takeover, role=role, index=index, traj=traj,
                              process_start_s=process_start_s,
                              total_s=time.perf_counter() - point_started,
                              prepare_s=prepare_s)
        except TrajectoryStartError as error:
            phase_a.append({"index": index, "traj_id": traj.traj_id, "seed": traj.seed,
                            "target_frame": target,
                            "error": f"{type(error).__name__}: {error}"})
            reasons.append(f"point {index} ({traj.traj_id} frame {target}): {error}")
            continue
        finally:
            env.close()
        record["engine_restarts"] = restarts
        phase_a.append(record)
        if not record["matches"]:
            reasons.append(f"point {index}: hash {record['target_hash']} != "
                           f"recorded {record['expected_target_hash']}")
        if restarts:
            reasons.append(f"point {index}: the harness was silently restarted {restarts} time(s)")

    matched = [row for row in phase_a if row.get("matches") is True]
    frames_replayed = sum(row.get("frames_advanced", 0) for row in matched)
    checkpoints = sum(row.get("checkpoints_checked", 0) for row in matched)
    report["phases"]["a_points"] = {
        "points": phase_a,
        "summary": {
            "requested": args.points,
            "recorded": len(phase_a),
            "matched": len(matched),
            "mismatched": len(phase_a) - len(matched),
            "frames_replayed": frames_replayed,
            "checkpoints_checked": checkpoints,
            "targets": [row["target_frame"] for row in matched],
            "seeds": sorted({row["seed"] for row in matched}),
            "stages": sorted({row["stage_index"] for row in matched}),
            "targets_sorted": sorted(row["target_frame"] for row in matched),
        },
    }
    if len(matched) != args.points:
        reasons.append(f"{len(matched)} of {args.points} points verified")

    # -- phase B: many take-overs inside one process ----------------------
    consecutive_targets = [picks[index % len(picks)] for index in range(args.consecutive)]
    data_dir_b = args.run_root / "dt-b"
    create_data_dir(template, data_dir_b)
    phase_b: list[dict] = []
    env, process_start_b = open_env(args.binary, data_dir_b, index=900, role=role)
    try:
        for index, (traj, target) in enumerate(consecutive_targets, start=1):
            takeover_started = time.perf_counter()
            try:
                takeover = replay_to_frame(env, traj.start(target),
                                           batch_frames=args.batch_frames)
            except TrajectoryStartError as error:
                phase_b.append({"index": index, "traj_id": traj.traj_id, "seed": traj.seed,
                                "target_frame": target,
                                "error": f"{type(error).__name__}: {error}"})
                reasons.append(f"consecutive {index} ({traj.traj_id} frame {target}): {error}")
                break
            record = describe(takeover, role=role, index=index, traj=traj,
                              process_start_s=None if index > 1 else process_start_b,
                              total_s=time.perf_counter() - takeover_started)
            record["mode"] = "in-process"
            phase_b.append(record)
            if not record["matches"]:
                reasons.append(f"consecutive {index}: hash {record['target_hash']} != "
                               f"recorded {record['expected_target_hash']}")
        restarts_b = env.restart_count
    finally:
        env.close()
    matched_b = [row for row in phase_b if row.get("matches") is True]
    report["phases"]["b_consecutive"] = {
        "takeovers": phase_b,
        "summary": {
            "requested": args.consecutive,
            "recorded": len(phase_b),
            "matched": len(matched_b),
            "frames_replayed": sum(row.get("frames_advanced", 0) for row in matched_b),
            "checkpoints_checked": sum(row.get("checkpoints_checked", 0) for row in matched_b),
            "reset_s": [row["reset_s"] for row in matched_b],
            "engine_restarts": restarts_b,
            "process_start_s": round(process_start_b, 6),
        },
    }
    if len(matched_b) != args.consecutive:
        reasons.append(f"{len(matched_b)} of {args.consecutive} consecutive take-overs verified")
    if restarts_b:
        reasons.append(f"the consecutive process restarted {restarts_b} time(s)")

    # -- phase C: block STEP vs per-frame STEP ----------------------------
    if picks:
        traj, target = picks[0]
        data_dir_c = args.run_root / "dt-c"
        create_data_dir(template, data_dir_c)
        env, _ = open_env(args.binary, data_dir_c, index=901, role=role)
        try:
            blocked = replay_to_frame(env, traj.start(target), batch_frames=args.batch_frames)
            single = replay_to_frame(env, traj.start(target), batch_frames=1)
        finally:
            env.close()
        phase_c = {
            "traj_id": traj.traj_id, "target_frame": target, "role": role,
            "blocked": {"steps": blocked.steps, "target_hash": blocked.target_hash,
                        "checkpoints": len(blocked.checkpoint_hashes)},
            "single_frame": {"steps": single.steps, "target_hash": single.target_hash,
                             "checkpoints": len(single.checkpoint_hashes)},
            "identical": blocked.checkpoint_hashes == single.checkpoint_hashes,
        }
        report["phases"]["c_batch_equivalence"] = phase_c
        if not phase_c["identical"]:
            reasons.append("per-frame replay and block replay produced different checkpoint hashes")

    # -- phase D: env-role cross-check ------------------------------------
    if args.env_cross_check > 0 and matched:
        cross = sorted(matched, key=lambda row: row["index"])[-args.env_cross_check:]
        phase_d: list[dict] = []
        for offset, row in enumerate(cross):
            traj = next(item for item in trajectories if item.traj_id == row["traj_id"])
            data_dir = args.run_root / f"dt-d-{offset:02d}"
            create_data_dir(template, data_dir)
            point_started = time.perf_counter()
            env, process_start_s = open_env(args.binary, data_dir, index=1000 + offset,
                                            role="env")
            try:
                takeover = replay_to_frame(env, traj.start(row["target_frame"]),
                                           batch_frames=args.batch_frames)
                restarts = env.restart_count
            except TrajectoryStartError as error:
                phase_d.append({"index": row["index"], "error": f"{type(error).__name__}: {error}"})
                reasons.append(f"env-role cross-check {row['index']}: {error}")
                continue
            finally:
                env.close()
            record = describe(takeover, role="env", index=row["index"], traj=traj,
                              process_start_s=process_start_s,
                              total_s=time.perf_counter() - point_started)
            record["engine_restarts"] = restarts
            phase_d.append(record)
            if not record["matches"]:
                reasons.append(f"env-role cross-check {row['index']}: hash {record['target_hash']} "
                               f"!= recorded {record['expected_target_hash']}")
            if record["target_hash"] != row["target_hash"]:
                reasons.append(f"env-role cross-check {row['index']}: env hash "
                               f"{record['target_hash']} != shadow hash {row['target_hash']}")
        report["phases"]["d_env_role"] = {"points": phase_d,
                                          "summary": {
                                              "requested": args.env_cross_check,
                                              "matched": sum(1 for row in phase_d
                                                             if row.get("matches") is True)}}

    # -- phase E: what one episode costs, one process per episode ----------
    if args.cold_restarts > 0:
        cold: list[dict] = []
        cold_reasons: list[str] = []
        for index in range(args.cold_restarts):
            data_dir = args.run_root / f"dt-e-{index:02d}"
            prepare_started = time.perf_counter()
            create_data_dir(template, data_dir)
            prepare_s = time.perf_counter() - prepare_started
            try:
                env, process_start_s = open_env(args.binary, data_dir, index=2000 + index,
                                                role=role)
            except Exception as error:                             # noqa: BLE001
                cold.append({"index": index, "error": f"{type(error).__name__}: {error}"})
                cold_reasons.append(f"cold restart {index}: engine did not start: {error}")
                continue
            try:
                reset_started = time.perf_counter()
                env.reset(trajectories[0].start(0).stage_start)
                reset_s = time.perf_counter() - reset_started
                reset_hash = getattr(env, "last_reset_hash", None)
                restarts = env.restart_count
            finally:
                env.close()
            record = {"index": index, "prepare_s": round(prepare_s, 6),
                      "process_start_s": round(process_start_s, 6),
                      "reset_s": round(reset_s, 6),
                      "total_s": round(prepare_s + process_start_s + reset_s, 6),
                      "reset_hash": reset_hash, "engine_restarts": restarts}
            cold.append(record)
            if restarts:
                cold_reasons.append(f"cold restart {index}: {restarts} silent restart(s)")
        totals = [row["total_s"] for row in cold if "error" not in row]
        report["phases"]["e_cold_restart"] = {
            "restarts": cold,
            "summary": {"requested": args.cold_restarts, "recorded": len(cold),
                        "median_s": median(totals) if totals else None,
                        "min_s": round(min(totals), 6) if totals else None,
                        "max_s": round(max(totals), 6) if totals else None},
        }
        reasons.extend(cold_reasons)

    restart_overhead = [row["process_start_s"] + row["reset_s"] for row in matched
                        if row.get("process_start_s") is not None]
    restart_with_prepare = [row["prepare_s"] + row["process_start_s"] + row["reset_s"]
                            for row in matched
                            if row.get("process_start_s") is not None
                            and row.get("prepare_s") is not None]
    inprocess_reset = [row["reset_s"] for row in matched_b]
    fast_forward = [row["fast_forward_s"] for row in matched] + [
        row["fast_forward_s"] for row in matched_b]
    cold_totals = [row["total_s"] for row in report["phases"].get("e_cold_restart", {})
                   .get("restarts", []) if "error" not in row] \
        if "e_cold_restart" in report["phases"] else []
    report["timing"] = {
        "phase_a_process_start_median_s": median([row["process_start_s"] for row in matched
                                                  if row.get("process_start_s") is not None]),
        "phase_a_prepare_median_s": median([row["prepare_s"] for row in matched
                                            if row.get("prepare_s") is not None]),
        "phase_a_total_median_s": median([row["total_s"] for row in matched]),
        "restart_per_point_median_s": median(restart_overhead),
        "restart_episode_overhead_median_s": median(restart_with_prepare),
        "cold_restart_median_s": median(cold_totals),
        "inprocess_takeover_median_s": median([row["total_s"] for row in matched_b]),
        "inprocess_reset_median_s": median(inprocess_reset),
        "inprocess_reset_max_s": round(max(inprocess_reset), 6) if inprocess_reset else None,
        "fast_forward_median_s": median(fast_forward),
        "frames_per_second_median": median([
            row["frames_advanced"] / row["fast_forward_s"]
            for row in matched + matched_b if row.get("fast_forward_s")]),
        "restart_vs_inprocess": (
            None if not restart_overhead or not inprocess_reset
            else round(statistics.median(restart_overhead) / statistics.median(inprocess_reset), 3)),
        "cold_restart_vs_inprocess": (
            None if not cold_totals or not inprocess_reset
            else round(statistics.median(cold_totals) / statistics.median(inprocess_reset), 3)),
    }
    #: The historical per-episode restart cost this comparison is measured against.
    report["reference"] = {
        "restart_per_episode_s": [6, 7],
        "source": "EXP-P4-001 §7.1 (旧构建：一局 = 一个进程)",
        "note": "历史值，不是本次测量；本次只测同一构建下的冷启动与进程内 RESET。",
        "inprocess_reset_vs_reference": (
            None if not inprocess_reset
            else [round(6.0 / statistics.median(inprocess_reset), 3),
                  round(7.0 / statistics.median(inprocess_reset), 3)]),
    }
    report["distribution"] = {
        "stages": sorted({row["stage_index"] for row in matched}),
        "seeds": sorted({row["seed"] for row in matched}),
        "frame_min": min((row["target_frame"] for row in matched), default=None),
        "frame_max": max((row["target_frame"] for row in matched), default=None),
        "quartiles": {str(quartile): sum(1 for row in matched if row["quartile"] == quartile)
                      for quartile in range(4)},
        "late_half": sum(1 for row in matched if row["late_half"]),
        "boss_section": sum(1 for row in matched if row["boss_section"]),
        "boss_start": args.boss_start,
    }

    report["wall_s"] = round(time.perf_counter() - started, 3)
    report["reasons"] = reasons
    report["status"] = "pass" if not reasons else "failed"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    summary = {
        "status": report["status"],
        "points": {"matched": len(matched), "requested": args.points,
                   "frames": frames_replayed, "checkpoints": checkpoints,
                   "targets": report["phases"]["a_points"]["summary"]["targets"]},
        "consecutive": {"matched": len(matched_b), "requested": args.consecutive,
                        "reset_s": report["phases"]["b_consecutive"]["summary"]["reset_s"]},
        "batch_equivalence": report["phases"].get("c_batch_equivalence", {}).get("identical"),
        "env_cross_check": report["phases"].get("d_env_role", {}).get("summary"),
        "cold_restart": report["phases"].get("e_cold_restart", {}).get("summary"),
        "distribution": report["distribution"],
        "timing": report["timing"],
        "reference": report["reference"],
        "binary_sha256": binary_info["sha256"],
        "wall_s": report["wall_s"],
        "reasons": reasons,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
