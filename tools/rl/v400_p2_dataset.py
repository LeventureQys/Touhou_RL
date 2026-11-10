#!/usr/bin/env python3
"""P2 v2 dataset generator (P2-WP2 minimal usable slice, EXP-P2-002).

Five subcommands:

``collect`` (WSL guest: the engine and its ``/dev/shm`` live there)
    Collect **one shard**: ``--source {random,still,teacher,replay}`` plus
    ``--stage/--difficulty/--seed/--frames``.  Writes into
    ``<out>/shards/<shard-key>/attempt-NN/``:

    * ``labels.jsonl`` — one line per frame, label set v2
      (``schema=v400-p2-dataset-label-2``: player / hud / events(v2) /
      occupancy / future_danger / decoded STATE);
    * ``pixels/frame-NNNNN.png`` — that frame's own 640x480 RGBA, lossless
      (``--pixel-format raw`` stores the raw RGBA bytes instead);
    * ``shard.json`` — the run's manifest (engine SHA, per-channel replies,
      availability counters, event counts, wall clock, ``complete``).

    Resume: an existing shard whose latest attempt is ``complete=true`` is
    skipped.  A shard is never overwritten — a new ``attempt-NN`` directory is
    used instead, so no file is ever deleted.

``suite``
    A list of shard specs (``source:stage:difficulty:seed:frames[:tag]``),
    collected one after another with the same resume rule.  Stops at the first
    ``--stop-after-failures`` failed shard (default 1) and keeps what it got.

``manifest``
    Rescan ``<out>/shards/*`` and rewrite ``<out>/manifest.json``.

``stats``
    Read every complete shard's labels and produce the P2-WP2 audit tables:
    event counts (real ``EVENTS`` bits), EVENTS-vs-proxy consistency, HUD value
    distributions, ``future_danger`` / occupancy coverage, and the **G2
    data-volume gap** (how many more positives/frames the gate would need —
    not a detector score).

``hud-lag``
    Per HUD row (``lives/bombs/score/graze/point/time/power``): pair each state
    change with the first band change in ``[k, k+max_lag]`` and report
    median/p95/max; rows with no attributable change are ``tested=false`` with
    the collection condition that would test them.

``teacher`` is refused fail-closed: see :func:`teacher_unavailable_reason`.

Examples::

    # WSL, one shard
    xvfb-run -a -s "-screen 0 1024x768x24" python tools/rl/v400_p2_dataset.py \\
      collect --engine /root/Touhou_RL/third_party/th08-rl/build/rl_hazard2/th08-modern \\
      --data-dir /root/v400-data/p2/game --out-dir .tmp_v400_p2v2/dataset \\
      --source random --stage 1 --difficulty 3 --seed 11 --frames 600

    # Windows, audit the result
    python tools/rl/v400_p2_dataset.py stats --out-dir .tmp_v400_p2v2/dataset \\
      --out .tmp_v400_p2v2/audit-v2.json
    python tools/rl/v400_p2_dataset.py hud-lag --out-dir .tmp_v400_p2v2/dataset \\
      --out .tmp_v400_p2v2/hud-lag-v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from touhou_rl.v400.labels.runner import (  # noqa: E402
    FUTURE_HORIZON,
    SOURCES,
    ShardSpec,
    SourceUnavailable,
    collect_shard,
    complete_label_files,
    latest_attempt,
    read_attempt_meta,
    scan_manifest,
    shard_complete,
)


def teacher_unavailable_reason() -> str:
    """Why ``--source teacher`` refuses to run instead of faking frames."""

    return (
        "teacher is not usable as a per-frame data source.  teacher-v1 "
        "(touhou_rl/v400/teacher/decide.py) is a *planner*, not a controller: one "
        "decision records 48 frames forward, beam-searches 18 first actions, forks "
        "the engine to verify plans, then restores the state with a fresh process + "
        "RESET + replay of the committed masks.  EXP-P1-003 measured 1744.15 ms per "
        "decision (0.5733 decisions/s) for R=2 sub-decisions = 4 executed frames, "
        "i.e. ~436 ms per collected frame, and the restore path needs one new engine "
        "process per decision, which cannot keep the SHM/READBACK pixel buffer or "
        "the ReplayManager event word across the restore.  Producing the requested "
        "frames would therefore take hours and would not be the same continuous "
        "trajectory the other sources record.  Use --source random/still/replay now "
        "and add teacher coverage after P1 exposes a state->action policy that does "
        "not move the authoritative process."
    )


def parse_shard_spec(text: str, *, replay_path: str | None = None) -> ShardSpec:
    """``source:stage:difficulty:seed:frames[:tag]`` -> :class:`ShardSpec`."""

    bits = text.split(":")
    if len(bits) not in (5, 6):
        raise SystemExit(
            f"bad --shard {text!r}; expected source:stage:difficulty:seed:frames[:tag]"
        )
    return ShardSpec(
        source=bits[0],
        stage=int(bits[1]),
        difficulty=int(bits[2]),
        seed=int(bits[3]),
        frames=int(bits[4]),
        tag=bits[5] if len(bits) == 6 else "",
        replay_path=replay_path,
    )


def _apply_common(spec: ShardSpec, args) -> ShardSpec:
    spec.role = args.role
    spec.hold = args.hold
    spec.bomb_prob = args.bomb_prob
    spec.warmup = args.warmup
    spec.timeout = args.timeout
    spec.replay_path = args.replay_path
    spec.pixel_format = args.pixel_format
    return spec


def _collect_one(args, spec: ShardSpec) -> dict:
    """Collect one shard unless a complete attempt already exists."""

    out_dir = Path(args.out_dir)
    shard_root = out_dir / "shards" / spec.key
    if args.resume and shard_root.is_dir():
        done = shard_complete(shard_root)
        if done is not None:
            print(
                json.dumps(
                    {
                        "shard": spec.key,
                        "skipped": True,
                        "reason": "complete attempt exists",
                        "attempt": done.get("attempt"),
                        "frames_written": done.get("frames_written"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return {"shard": spec.key, "skipped": True, "meta": done}
    log_path = out_dir / "logs" / f"{spec.key}-{int(time.time())}.engine.stderr.log"
    metrics = collect_shard(
        spec,
        engine=args.engine,
        data_dir=args.data_dir,
        shard_root=shard_root,
        log_path=log_path,
    )
    scan_manifest(out_dir)
    return {"shard": spec.key, "skipped": False, "meta": metrics}


def collect_command(args) -> int:
    if args.source == "teacher":
        print("ERR teacher-unavailable: " + teacher_unavailable_reason(), file=sys.stderr)
        return 3
    if args.source == "replay" and not args.replay_path:
        print("ERR usage: --source replay needs --replay-path <file.rpy>", file=sys.stderr)
        return 2
    spec = _apply_common(
        ShardSpec(
            source=args.source,
            stage=args.stage,
            difficulty=args.difficulty,
            seed=args.seed,
            frames=args.frames,
            tag=args.tag,
            replay_path=args.replay_path,
        ),
        args,
    )
    result = _collect_one(args, spec)
    meta = result["meta"] or {}
    print(
        json.dumps(
            {
                "shard": result["shard"],
                "skipped": result["skipped"],
                "complete": meta.get("complete"),
                "frames_written": meta.get("frames_written"),
                "counts": meta.get("counts"),
                "channel_errors": meta.get("channel_errors"),
                "error": meta.get("error"),
                "wall_seconds": meta.get("wall_seconds"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if meta.get("complete") else 4


def suite_command(args) -> int:
    specs = [
        parse_shard_spec(text, replay_path=args.replay_path) for text in args.shards
    ]
    for spec in specs:
        if spec.source == "teacher":
            print(
                "ERR teacher-unavailable: " + teacher_unavailable_reason(),
                file=sys.stderr,
            )
            return 3
    failures = 0
    results = []
    total_frames = 0
    for spec in specs:
        spec = _apply_common(spec, args)
        try:
            result = _collect_one(args, spec)
        except SourceUnavailable as exc:
            print(json.dumps({"shard": spec.key, "error": str(exc)}, ensure_ascii=False))
            failures += 1
            if failures >= args.stop_after_failures:
                break
            continue
        meta = result["meta"] or {}
        results.append(result)
        total_frames += int(meta.get("frames_written") or 0)
        if not meta.get("complete"):
            failures += 1
            if failures >= args.stop_after_failures:
                break
        if args.max_frames and total_frames >= args.max_frames:
            break
    manifest = scan_manifest(args.out_dir)
    print(
        json.dumps(
            {
                "shards_requested": len(specs),
                "shards_attempted": len(results),
                "frames_this_invocation": total_frames,
                "complete_shards": manifest["complete_shards"],
                "frames_written_complete_shards": manifest["frames_written_complete_shards"],
                "failures": failures,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if failures == 0 else 4


def manifest_command(args) -> int:
    manifest = scan_manifest(args.out_dir)
    print(
        json.dumps(
            {
                "shard_count": manifest["shard_count"],
                "complete_shards": manifest["complete_shards"],
                "frames_written_complete_shards": manifest[
                    "frames_written_complete_shards"
                ],
                "engine_sha256": manifest["engine_sha256"],
                "manifest": str(Path(args.out_dir) / "manifest.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def load_complete_records(out_dir: str | Path) -> list[dict]:
    records: list[dict] = []
    for path in complete_label_files(out_dir):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def stats_command(args) -> int:
    from touhou_rl.v400.labels import LABEL_VERSION
    from touhou_rl.v400.labels.audit import (
        event_consistency,
        event_summary,
        g2_gap_estimate,
        to_markdown_v2,
    )

    out_dir = Path(args.out_dir)
    records = load_complete_records(out_dir)
    if not records:
        raise SystemExit(f"no complete shard labels under {out_dir}")
    manifest = scan_manifest(out_dir)
    events = event_summary(records)
    consistency = event_consistency(records)
    gap = g2_gap_estimate(records, events, consistency=consistency)
    versions = sorted(
        {str((record.get("meta") or {}).get("label_version")) for record in records}
    )
    if versions != [LABEL_VERSION]:
        print(
            f"warning: label versions {versions} != {LABEL_VERSION}",
            file=sys.stderr,
        )
    report = {
        "schema": "v400-p2-dataset-stats-2",
        "out_dir": str(out_dir),
        "label_version": LABEL_VERSION,
        "label_versions_in_data": versions,
        "manifest": {
            "complete_shards": manifest["complete_shards"],
            "shard_count": manifest["shard_count"],
            "frames_written_complete_shards": manifest[
                "frames_written_complete_shards"
            ],
            "engine_sha256": manifest["engine_sha256"],
        },
        "frames": len(records),
        "shards": sorted(
            {
                f"{record.get('shard')}/{record.get('attempt')}"
                for record in records
            }
        ),
        "stages": _hist(record.get("stage") for record in records),
        "difficulties": _hist(record.get("difficulty") for record in records),
        "sources": _hist(record.get("source") for record in records),
        "events": events,
        "event_consistency": consistency,
        "future_danger": _danger_section(records),
        "occupancy": _occupancy_section(records),
        "hud": _hud_section(records),
        "g2_data_gap": gap,
    }
    out = Path(args.out) if args.out else out_dir / "stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = to_markdown_v2(report)
    if args.markdown:
        Path(args.markdown).write_text(markdown, encoding="utf-8")
    print(markdown)
    print(json.dumps({"stats": str(out)}, indent=2))
    return 0


def _hist(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _danger_section(records: list[dict]) -> dict:
    positive = negative = null = applicable = 0
    reasons: dict[str, int] = {}
    gaps: list[float] = []
    per_horizon = {"4": [0, 0, 0], "8": [0, 0, 0], "16": [0, 0, 0]}
    for record in records:
        danger = record.get("future_danger") or {}
        if danger.get("applicable"):
            applicable += 1
        value = danger.get("would_die")
        if value is True:
            positive += 1
        elif value is False:
            negative += 1
        else:
            null += 1
        reasons[str(danger.get("reason") or "ok")] = (
            reasons.get(str(danger.get("reason") or "ok"), 0) + 1
        )
        if danger.get("min_gap") is not None and value is not None:
            gaps.append(float(danger["min_gap"]))
        for key, horizon in (danger.get("per_horizon") or {}).items():
            bucket = per_horizon.get(str(key))
            if bucket is None:
                continue
            if horizon is True:
                bucket[0] += 1
            elif horizon is False:
                bucket[1] += 1
            else:
                bucket[2] += 1
    return {
        "applicable": applicable,
        "positive": positive,
        "negative": negative,
        "null": null,
        "missing_rate": (null / len(records)) if records else None,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "min_gap_px": _quantiles(gaps),
        "per_horizon": {
            "4": {"true": per_horizon["4"][0], "false": per_horizon["4"][1], "null": per_horizon["4"][2]},
            "8": {"true": per_horizon["8"][0], "false": per_horizon["8"][1], "null": per_horizon["8"][2]},
            "16": {"true": per_horizon["16"][0], "false": per_horizon["16"][1], "null": per_horizon["16"][2]},
        },
    }


def _occupancy_section(records: list[dict]) -> dict:
    nonempty = empty = missing = 0
    cells_hist: dict[str, int] = {}
    nearest: list[float] = []
    lasers = 0
    enemies = 0
    for record in records:
        occupancy = record.get("occupancy")
        if not occupancy:
            missing += 1
            continue
        cells = int(occupancy.get("occupied_cells", 0))
        if cells > 0:
            nonempty += 1
        else:
            empty += 1
        cells_hist[str(cells if cells < 50 else "50+")] = (
            cells_hist.get(str(cells if cells < 50 else "50+"), 0) + 1
        )
        if occupancy.get("nearest_distance") is not None:
            nearest.append(float(occupancy["nearest_distance"]))
        if int(occupancy.get("laser_cells", 0)) > 0:
            lasers += 1
        if int(occupancy.get("enemy_boxes", 0)) > 0:
            enemies += 1
    return {
        "nonempty": nonempty,
        "empty": empty,
        "missing": missing,
        "empty_rate": (empty / len(records)) if records else None,
        "occupied_cells_hist": cells_hist,
        "nearest_distance_px": _quantiles(nearest),
        "frames_with_laser_cells": lasers,
        "frames_with_enemy_boxes": enemies,
    }


def _hud_section(records: list[dict]) -> dict:
    def hist(key: str, *, cap: int = 20) -> dict[str, int]:
        out: dict[str, int] = {}
        for record in records:
            value = (record.get("hud") or {}).get(key)
            label = "missing" if value is None else str(value)
            out[label] = out.get(label, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (kv[0] == "missing", kv[0])))

    complete = sum(1 for record in records if (record.get("hud") or {}).get("complete"))
    scores = [
        int((record.get("hud") or {})["score"])
        for record in records
        if (record.get("hud") or {}).get("score") is not None
    ]
    graze = [
        int((record.get("hud") or {})["graze"])
        for record in records
        if (record.get("hud") or {}).get("graze") is not None
    ]
    return {
        "complete": complete,
        "complete_rate": (complete / len(records)) if records else None,
        "lives": hist("lives"),
        "bombs": hist("bombs"),
        "power": hist("power"),
        "point": hist("point"),
        "time": hist("time"),
        "score": {
            "present": len(scores),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        },
        "graze": {
            "present": len(graze),
            "min": min(graze) if graze else None,
            "max": max(graze) if graze else None,
        },
        "display_score_unavailable": "globals.displayScore is not exported",
    }


def _quantiles(values) -> dict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None}

    def at(frac: float) -> float:
        return ordered[min(len(ordered) - 1, int(len(ordered) * frac))]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": at(0.5),
        "p95": at(0.95),
        "max": ordered[-1],
    }


def hud_lag_command(args) -> int:
    from touhou_rl.v400.labels.audit import hud_lag_v2

    out_dir = Path(args.out_dir)
    records = load_complete_records(out_dir)
    if not records:
        raise SystemExit(f"no complete shard labels under {out_dir}")
    fields = tuple(args.fields)
    report = hud_lag_v2(records, out_dir, fields=fields, max_lag=args.max_lag)
    # record where each row's pixels live so the measurement is reproducible
    report["frames_root"] = str(out_dir)
    report["shards"] = sorted({f"{r.get('shard')}/{r.get('attempt')}" for r in records})
    out = Path(args.out) if args.out else out_dir / "hud-lag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"hud_lag": str(out)}, indent=2))
    for field in fields:
        entry = report[field]
        print(
            f"{field}: tested={entry['tested']} n={entry['n']} "
            f"median={entry['median']} p95={entry['p95']} max={entry['max']} "
            f"state_changes={entry['state_changes']} matched={entry['matched']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    # The reports are UTF-8 (Chinese table headers, "EVENTS <-> proxy"); a GBK
    # console must not turn a finished audit into an encoding crash.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_engine_args(target, *, frames_required: bool) -> None:
        target.add_argument("--engine", required=True)
        target.add_argument("--data-dir", required=True)
        target.add_argument("--out-dir", required=True, help="dataset root")
        target.add_argument("--role", default="env", choices=("env", "shadow"))
        target.add_argument("--hold", type=int, default=6)
        target.add_argument("--bomb-prob", type=float, default=0.0)
        target.add_argument("--warmup", type=int, default=700)
        target.add_argument("--timeout", type=float, default=300.0)
        target.add_argument("--replay-path", default=None)
        target.add_argument("--pixel-format", default="png", choices=("png", "raw"))
        target.add_argument("--resume", action="store_true", default=True)
        target.add_argument("--no-resume", dest="resume", action="store_false")

    collect_parser = sub.add_parser("collect", help="one shard, resumable")
    add_engine_args(collect_parser, frames_required=True)
    collect_parser.add_argument("--source", required=True, choices=("random", "still", "teacher", "replay"))
    collect_parser.add_argument("--stage", type=int, required=True)
    collect_parser.add_argument("--difficulty", type=int, required=True)
    collect_parser.add_argument("--seed", type=int, default=1)
    collect_parser.add_argument("--frames", type=int, required=True)
    collect_parser.add_argument("--tag", default="")
    collect_parser.set_defaults(func=collect_command)

    suite_parser = sub.add_parser("suite", help="many shards, resumable")
    add_engine_args(suite_parser, frames_required=True)
    suite_parser.add_argument("--shards", nargs="+", required=True)
    suite_parser.add_argument("--max-frames", type=int, default=0)
    suite_parser.add_argument("--stop-after-failures", type=int, default=1)
    suite_parser.add_argument("--source-teacher", action="store_true", help=argparse.SUPPRESS)
    suite_parser.set_defaults(func=suite_command)

    manifest_parser = sub.add_parser("manifest", help="rewrite manifest.json")
    manifest_parser.add_argument("--out-dir", required=True)
    manifest_parser.set_defaults(func=manifest_command)

    stats_parser = sub.add_parser("stats", help="audit tables + G2 data gap")
    stats_parser.add_argument("--out-dir", required=True)
    stats_parser.add_argument("--out", default=None)
    stats_parser.add_argument("--markdown", default=None)
    stats_parser.set_defaults(func=stats_command)

    lag_parser = sub.add_parser("hud-lag", help="per-row HUD lag")
    lag_parser.add_argument("--out-dir", required=True)
    lag_parser.add_argument("--out", default=None)
    lag_parser.add_argument("--max-lag", type=int, default=10)
    lag_parser.add_argument(
        "--fields",
        nargs="+",
        default=["lives", "bombs", "score", "graze", "point", "time", "power"],
    )
    lag_parser.set_defaults(func=hud_lag_command)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
