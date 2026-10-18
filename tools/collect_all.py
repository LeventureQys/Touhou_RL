"""Batch-collect every replay in a directory, resumably.

Why a driver instead of a loop in the shell
-------------------------------------------
A full corpus is ~4.58 M logic frames, i.e. **~22 hours** of real-time playback
(measured: TH08 replay playback is hard-capped at 60 logic fps and holding Ctrl
does not speed it up - see ``tools/probes/replay_speed.py``).  Nobody finishes that
in one sitting, so the driver must:

* keep a **registry** keyed by the replay's sha256, so a re-run skips what is done
  and re-does what failed, without duplicating shards;
* survive a bad file: a crashed or rejected replay is recorded as ``failed`` with
  the reason instead of aborting 20 hours of work;
* enforce a **wall-clock budget** so it can be run in bounded chunks;
* write one dataset directory with a **unique shard tag per replay**, so training
  can consume the whole corpus at once while provenance stays per-file.

Usage
-----
```powershell
# start / continue a batch, 90 minutes at a time
python tools/collect_all.py --dir th08_replay --out .../datasets/vision-all --max-minutes 90

# see progress without touching the game
python tools/collect_all.py --dir th08_replay --out .../datasets/vision-all --status
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY_NAME = "collection-registry.json"
# Measured after fixing two bugs in the capture path (modulo stride dropped
# frames; savez_compressed re-deflated already-compressed PNGs): native 640x480
# region capture sustains 29.97 samples per game-second against a 30/s supply, at
# ~229 KB/sample.  See tools/probes/capture_profile.py and grab_impact.py.
BYTES_PER_VISION_SAMPLE = 229_000
BYTES_PER_PRIVILEGED_SAMPLE = 1_030
VISION_SAMPLES_PER_SECOND = 29.97
PRIVILEGED_SAMPLES_PER_SECOND = 59.0


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 1, "files": {}}


def save_registry(path: Path, registry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def replay_frames(path: Path) -> int:
    """Input record count from the header (used for ETA only; 0 when unknown)."""
    try:
        scripts = ROOT / "third_party/th08/scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from replay_file import inspect as inspect_replay

        info = inspect_replay(path)
        return int(sum(stage["inputRecords"] for stage in info["stages"]))
    except Exception:
        return 0


@dataclass
class BatchSpec:
    replays: list[Path]
    out_dir: Path
    modality: str = "vision"
    every: int = 2
    max_minutes: float = 0.0
    per_file_seconds: float = 4200.0
    process_after: bool = True
    tag_prefix: str = ""
    dry_run: bool = False
    extra_labels: dict = field(default_factory=dict)


def _estimate_samples(frames: int, modality: str, every: int) -> int:
    return frames if modality == "privileged" else max(1, frames // every)


def _rate(modality: str) -> float:
    return VISION_SAMPLES_PER_SECOND if modality == "vision" else PRIVILEGED_SAMPLES_PER_SECOND


def run_batch(spec: BatchSpec) -> dict:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = spec.out_dir / REGISTRY_NAME
    registry = load_registry(registry_path)
    registry.setdefault("files", {})
    registry["out_dir"] = str(spec.out_dir)
    registry["modality"] = spec.modality
    budget_deadline = time.time() + spec.max_minutes * 60.0 if spec.max_minutes else None

    planned_seconds = 0.0
    done_now = 0
    for replay in spec.replays:
        frames = replay_frames(replay)
        planned_seconds += _estimate_samples(frames, spec.modality, spec.every) / _rate(spec.modality)
    started = time.time()

    for index, replay in enumerate(spec.replays, start=1):
        digest = sha256_of(replay)
        entry = registry["files"].get(replay.name) or {}
        if entry.get("sha256") == digest and entry.get("status") == "done":
            print(f"[batch] {index}/{len(spec.replays)} {replay.name}: already done, skip", flush=True)
            continue
        if budget_deadline is not None and time.time() >= budget_deadline:
            print(f"[batch] budget exhausted before {replay.name}; stopping (resume later)", flush=True)
            break
        if spec.dry_run:
            print(f"[batch] {index}/{len(spec.replays)} {replay.name}: would collect", flush=True)
            continue

        tag = f"{spec.tag_prefix}{replay.stem.lower().replace('th8_', '')}"
        print(f"[batch] {index}/{len(spec.replays)} {replay.name}: collecting (tag={tag})", flush=True)
        row = {
            "file": replay.name,
            "sha256": digest,
            "tag": tag,
            "status": "running",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frames_expected": frames,
        }
        registry["files"][replay.name] = row
        save_registry(registry_path, registry)
        file_started = time.time()
        try:
            from touhou_rl.pipeline.collect import CollectSpec, run_collect

            budget_left = (
                max(60.0, budget_deadline - time.time())
                if budget_deadline is not None
                else spec.per_file_seconds
            )
            payload = run_collect(
                CollectSpec(
                    modality=spec.modality,
                    out_dir=spec.out_dir,
                    source="replay",
                    replay=replay,
                    every=spec.every,
                    tag=tag,
                    max_seconds=min(spec.per_file_seconds, budget_left),
                    overwrite=True,  # append new shards to the shared dataset dir
                    verbose=True,
                    extra=spec.extra_labels,
                )
            )
            report = payload.get("report", {})
            row.update(
                {
                    "status": "done",
                    "frames": int(report.get("frames", 0)),
                    "shards": int(report.get("shards", 0)),
                    "elapsed_seconds": round(time.time() - file_started, 1),
                    "ended_reason": report.get("ended_reason", ""),
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            done_now += 1
        except Exception as exc:  # one bad replay must not kill the batch
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(time.time() - file_started, 1),
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            print(f"[batch] {replay.name}: FAILED - {row['error']}", flush=True)
        # Per-file report copy, so the shared directory keeps per-replay provenance.
        reports = spec.out_dir / "reports"
        reports.mkdir(exist_ok=True)
        (reports / f"{tag}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        save_registry(registry_path, registry)

    elapsed = time.time() - started
    done = sum(1 for row in registry["files"].values() if row.get("status") == "done")
    failed = sum(1 for row in registry["files"].values() if row.get("status") == "failed")
    remaining = [replay for replay in spec.replays if registry["files"].get(replay.name, {}).get("status") != "done"]
    remaining_seconds = sum(
        _estimate_samples(replay_frames(replay), spec.modality, spec.every) / _rate(spec.modality)
        for replay in remaining
    )
    summary = {
        "out_dir": str(spec.out_dir),
        "modality": spec.modality,
        "replays_total": len(spec.replays),
        "done": done,
        "failed": failed,
        "remaining": len(remaining),
        "collected_this_run": done_now,
        "elapsed_seconds": round(elapsed, 1),
        "estimated_total_seconds": round(planned_seconds, 1),
        "estimated_remaining_seconds": round(remaining_seconds, 1),
        "estimated_remaining_hours": round(remaining_seconds / 3600.0, 2),
        "registry": str(registry_path),
    }
    if spec.process_after and done_now and not spec.dry_run:
        try:
            from touhou_rl.pipeline.process import ProcessSpec, run_process

            print("[batch] processing the combined dataset (manifest + teacher labels)", flush=True)
            summary["process"] = run_process(
                ProcessSpec(raw_dir=spec.out_dir, modality=spec.modality, verbose=True)
            )
        except Exception as exc:
            summary["process_error"] = f"{type(exc).__name__}: {exc}"
    save_registry(registry_path, registry)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-collect replays, resumably")
    parser.add_argument("--dir", default="th08_replay", help="directory of .rpy files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--modality", default="vision", choices=["vision", "privileged"])
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--max-minutes", type=float, default=0.0, help="budget for this invocation")
    parser.add_argument("--per-file-seconds", type=float, default=4200.0)
    parser.add_argument("--tag-prefix", default="")
    parser.add_argument("--no-process", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true", help="print progress and exit")
    args = parser.parse_args(argv)

    replays = sorted(Path(args.dir).glob("*.rpy"))
    if not replays:
        parser.error(f"no .rpy files under {args.dir}")
    out_dir = Path(args.out)
    if args.status:
        registry = load_registry(out_dir / REGISTRY_NAME)
        rows = registry.get("files", {})
        done = [name for name, row in rows.items() if row.get("status") == "done"]
        failed = {name: row.get("error", "") for name, row in rows.items() if row.get("status") == "failed"}
        frames = sum(int(row.get("frames", 0)) for row in rows.values() if row.get("status") == "done")
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "collected": len(done),
                    "failed": failed,
                    "frames_collected": frames,
                    "not_yet_started": [replay.name for replay in replays if replay.name not in rows],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    spec = BatchSpec(
        replays=replays,
        out_dir=out_dir,
        modality=args.modality,
        every=args.every,
        max_minutes=args.max_minutes,
        per_file_seconds=args.per_file_seconds,
        process_after=not args.no_process,
        tag_prefix=args.tag_prefix,
        dry_run=args.dry_run,
    )
    summary = run_batch(spec)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
