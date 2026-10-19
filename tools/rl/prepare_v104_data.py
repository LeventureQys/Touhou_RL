#!/usr/bin/env python
"""Prepare the v1.0.4 data artefacts (S3-1): history audit, decode cache, B list.

Three jobs, one entry point, because all three answer "what is this dataset,
exactly?" and must be runnable by an independent reviewer without a chat log:

``audit``
    Recompute the frame-history audit (delta histogram, segment/reset counts,
    duplicate-history ratio) from ``meta`` only.  Reads no pixel and writes no
    data.

``cache``
    Build or repair the decoded-frame cache for the three fixed ROIs
    (playfield 224x192, hud_score 34x216, hud_status 126x216), then re-verify
    every cache file and its ``<cache>.json`` identity sidecar.  A free-space
    precheck runs first; nothing is deleted to make room.

``eval-manifest``
    Write a read-only evaluation manifest for an *external* dataset that has no
    manifest (``pilot-ud0063``).  It is refused for a dataset that already has
    ``manifest.json`` (that is the training source) and it refuses to write a
    dataset's default manifest path.  ``teacher={}``, ``has_teacher_values=false``,
    ``stats.evaluation_only=true``, ``provenance.purpose=external-evaluation-only``:
    the listed indices are for cross-replay evaluation, never a training licence.

Every failure that is a known contract/manifest/IO problem is printed to stderr
in the shared ``v104:<CODE>: <detail>`` format and exits 2; an unexpected failure
keeps its traceback and exits non-zero.

Usage::

    python tools/rl/prepare_v104_data.py --dataset <dir> --mode audit --depth 3 --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # allow "python tools/rl/prepare_v104_data.py" from anywhere
    sys.path.insert(0, str(REPO_ROOT))

from touhou_rl import vision_history  # noqa: E402
from touhou_rl.pipeline.manifest import (  # noqa: E402
    MANIFEST_NAME,
    DatasetManifest,
    ManifestError,
    ShardRecord,
    file_digest,
)
from touhou_rl.vision_data import (  # noqa: E402
    DECODE_CACHE_SCHEMA,
    ROI_SIZES,
    VisionDataset,
    build_decode_cache,
    check_decode_cache,
    decoded_cache_path,
    verify_decode_cache,
)

#: Bytes of the ``.npy`` header np.save writes for a 4-D uint8 array.
NPY_HEADER_BYTES = 128
#: Room left for the 1626 (and future) sidecars plus temporary files.
SIDECAR_ALLOWANCE_BYTES = 32 * 1024 * 1024
#: Never start a build that would consume the last GiB of the volume.
FREE_SPACE_HEADROOM_BYTES = 1024**3
#: The evaluation-manifest split contract (matches the training manifest / old tool).
EVAL_SEED = 7
EVAL_VAL_FRACTION = 0.1

KNOWN_ERRORS = (ValueError, KeyError, ManifestError, OSError)


class _Parser(argparse.ArgumentParser):
    """Argparse with a stable epilog; exit codes stay argparse's 2 for usage errors."""


# --- helpers ----------------------------------------------------------------
def _v104(code: str, detail: str) -> ValueError:
    return ValueError(f"v104:{code}: {detail}")


def _write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def _looks_like_manifest(path: Path) -> bool:
    """True when ``path`` already holds a ``DatasetManifest`` (never clobber one)."""
    path = Path(path)
    if not path.exists() or path.suffix != ".json":
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and {"version", "shards", "splits"} <= set(payload)
        and isinstance(payload.get("shards"), list)
    )


def _refuse_manifest_report(out: Path) -> None:
    if _looks_like_manifest(out):
        raise _v104(
            "EVIDENCE",
            f"refusing to overwrite a DatasetManifest with a report: {out}",
        )


def _manifest_context(dataset_dir: Path) -> dict:
    """Read-only summary of a dataset manifest (A's split proof), or ``{}``."""
    path = Path(dataset_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"path": str(path), "error": "unreadable"}
    splits = payload.get("splits", {}) or {}
    return {
        "path": str(path),
        "total_frames": payload.get("total_frames"),
        "shards": len(payload.get("shards", []) or []),
        "seed": splits.get("seed"),
        "val_fraction": splits.get("val_fraction"),
        "val_frames": splits.get("val_frames"),
        "val_index_sha256": splits.get("val_index_sha256"),
    }


def _manifest_content_digest(payload: dict) -> str:
    """Digest of a manifest ignoring ``created`` so a rebuild counts as identical."""
    clone = {key: value for key, value in payload.items() if key != "created"}
    encoded = json.dumps(clone, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _estimate_cache_need(dataset: VisionDataset, sizes: dict[str, tuple[int, int]]) -> dict:
    """Bytes a build would have to write, from ``stat`` plus an ``n x H x W x C`` payload."""
    missing_bytes = replace_bytes = present_bytes = 0
    files_missing = files_invalid = files_ok = 0
    largest = 0
    for path, _start, size in dataset.offsets:
        for region, (height, width) in sizes.items():
            if region not in dataset.region_hw:
                continue
            payload = NPY_HEADER_BYTES + int(size) * int(height) * int(width) * 3
            largest = max(largest, payload)
            target = decoded_cache_path(path, region, height, width)
            if not target.exists():
                files_missing += 1
                missing_bytes += payload
                continue
            check = check_decode_cache(
                path, region, height, width, frames=size, verify_source=False
            )
            if check.ok:
                files_ok += 1
                present_bytes += check.bytes
            else:
                files_invalid += 1
                replace_bytes += payload
                present_bytes += check.bytes
    return {
        "files_missing": files_missing,
        "files_invalid": files_invalid,
        "files_ok": files_ok,
        "missing_bytes": int(missing_bytes),
        "replace_bytes": int(replace_bytes),
        "present_bytes": int(present_bytes),
        "largest_file_bytes": int(largest),
        "sizes": {key: list(value) for key, value in sizes.items()},
        "cache_contract": DECODE_CACHE_SCHEMA,
        "note": (
            "estimate: replace_bytes is net new payload; the old file still exists while the "
            "temporary replacement is written, which is why it is charged again"
        ),
    }


# --- modes ------------------------------------------------------------------
def mode_audit(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset)
    if args.depth not in (1, 3):
        raise _v104("CONFIG", f"--depth must be 1 or 3, got {args.depth}")
    dataset = VisionDataset(dataset_dir, history_contract=args.history_contract)
    try:
        report = dataset.history_audit(int(args.depth))
        report["decode_cache"] = dataset.decode_cache_stats()
        report["cache_stats"] = dataset.cache_stats()
    finally:
        dataset.close()
    report = dict(report)
    report["mode"] = "audit"
    report["dataset"] = str(dataset_dir)
    report["manifest"] = _manifest_context(dataset_dir)
    report["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(Path(args.out), report)
    print(
        json.dumps(
            {
                "mode": "audit",
                "out": str(args.out),
                "history_contract": report["history_contract"],
                "depth": report["depth"],
                "frames": report["frames"],
                "resets": report["resets"],
                "repeated_history_samples": report["repeated_history_samples"],
                "repeated_history_ratio": report["repeated_history_ratio"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def mode_cache(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset)
    out = Path(args.out)
    _refuse_manifest_report(out)
    sizes = {name: tuple(size) for name, size in ROI_SIZES.items()}
    dataset = VisionDataset(dataset_dir, use_decode_cache=False)
    try:
        estimate = _estimate_cache_need(dataset, sizes)
    finally:
        dataset.close()
    free_before = int(shutil.disk_usage(dataset_dir).free)
    required = (
        estimate["missing_bytes"]
        + estimate["replace_bytes"]
        + estimate["largest_file_bytes"]
        + SIDECAR_ALLOWANCE_BYTES
        + FREE_SPACE_HEADROOM_BYTES
    )
    if free_before < required:
        raise _v104(
            "BUDGET",
            f"insufficient free space for the decode cache: {free_before} B free < {required} B "
            f"required (missing={estimate['missing_bytes']} B, replace={estimate['replace_bytes']} B); "
            "no user data was deleted",
        )
    build = build_decode_cache(
        dataset_dir,
        sizes,
        verbose=True,
        decode_threads=int(args.decode_threads),
        limit_shards=args.limit_shards,
    )
    verification = verify_decode_cache(
        dataset_dir,
        sizes,
        verify_payload=True,
        verbose=False,
        limit_shards=args.limit_shards,
    )
    report = {
        "mode": "cache",
        "dataset": str(dataset_dir),
        "sizes": {key: list(value) for key, value in sizes.items()},
        "cache_contract": DECODE_CACHE_SCHEMA,
        "limit_shards": args.limit_shards,
        "precheck": {
            **estimate,
            "free_bytes_before": free_before,
            "required_bytes": int(required),
            "headroom_bytes": FREE_SPACE_HEADROOM_BYTES,
            "sidecar_allowance_bytes": SIDECAR_ALLOWANCE_BYTES,
        },
        "build": build,
        "verification": verification,
        "free_bytes_after": int(shutil.disk_usage(dataset_dir).free),
        "manifest": _manifest_context(dataset_dir),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_json(out, report)
    print(
        json.dumps(
            {
                "mode": "cache",
                "out": str(out),
                "files_written": build["files_written"],
                "files_skipped": build["files_skipped"],
                "files_invalid": build["files_invalid"],
                "files_repaired": build["files_repaired"],
                "files_present": build["files_present"],
                "bytes_written": build["bytes_written"],
                "bytes_total": build["bytes_total"],
                "verified_files": verification["files"],
                "verified_ok": verification["files_ok"],
                "invalid_reasons": verification["invalid_reasons"],
                "free_bytes_before": free_before,
                "required_bytes": int(required),
                "elapsed_seconds": build["elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def mode_eval_manifest(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")
    default_manifest = dataset_dir / MANIFEST_NAME
    out = Path(args.out)
    if out.resolve() == default_manifest.resolve():
        raise _v104(
            "EVIDENCE",
            f"refusing to write the dataset's default manifest path: {out}",
        )
    if default_manifest.exists():
        raise _v104(
            "CONFIG",
            f"refusing --mode eval-manifest: {default_manifest} exists; a dataset with a "
            "manifest is a training source, not an external evaluation set",
        )
    dataset = VisionDataset(dataset_dir, use_decode_cache=False)
    try:
        records: list[ShardRecord] = []
        stage_counts: dict[int, int] = {}
        regions = {name: list(hw) for name, hw in dataset.region_hw.items()}
        for path, _start, size in dataset.offsets:
            with np.load(path) as data:
                stages = np.rint(np.asarray(data["meta"], dtype=np.float64)[:, 1])
            for value, count in zip(*np.unique(stages, return_counts=True)):
                key = int(value)
                stage_counts[key] = stage_counts.get(key, 0) + int(count)
            records.append(
                ShardRecord(
                    name=path.name,
                    frames=int(size),
                    bytes=int(path.stat().st_size),
                    sha256=file_digest(path),
                )
            )
        total = int(dataset.total)
        frame_shape = None
        playfield = dataset.region_hw.get("playfield")
        if playfield is not None:
            frame_shape = [int(playfield[0]), int(playfield[1]), 3]
    finally:
        dataset.close()
    records.sort(key=lambda record: record.name)
    manifest = DatasetManifest.build(
        modality="vision",
        root=str(dataset_dir),
        shards=records,
        frame_shape=frame_shape,
        has_teacher_values=False,
        teacher={},
        seed=EVAL_SEED,
        val_fraction=EVAL_VAL_FRACTION,
        stats={
            "evaluation_only": True,
            "frames": total,
            "shards": len(records),
            "stage_hist": {str(key): value for key, value in sorted(stage_counts.items())},
            "regions": regions,
        },
        provenance={
            "purpose": "external-evaluation-only",
            "tool": "tools/rl/prepare_v104_data.py",
            "mode": "eval-manifest",
            "source": "existing shards only; no capture, no process stage",
            "note": (
                "indices are for cross-replay evaluation only; the generic split is not a "
                "training authorisation for this dataset"
            ),
        },
    )
    payload = asdict(manifest)
    digest = _manifest_content_digest(payload)
    reused = False
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _v104("EVIDENCE", f"existing output {out} is not readable JSON: {exc}") from exc
        if not isinstance(existing, dict) or _manifest_content_digest(existing) != digest:
            raise _v104(
                "EVIDENCE",
                f"existing output {out} has different content than this manifest "
                f"(existing={_manifest_content_digest(existing) if isinstance(existing, dict) else 'n/a'} "
                f"new={digest}); remove it deliberately or use another --out",
            )
        reused = True
    else:
        _write_json(out, payload)
    summary = {
        "mode": "eval-manifest",
        "out": str(out),
        "reused": reused,
        "total_frames": payload["total_frames"],
        "shards": len(payload["shards"]),
        "frame_shape": payload["frame_shape"],
        "has_teacher_values": payload["has_teacher_values"],
        "teacher": payload["teacher"],
        "splits": payload["splits"],
        "stats": payload["stats"],
        "provenance": payload["provenance"],
        "content_sha256": digest,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def _classify(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return "CONFIG"
    if isinstance(exc, KeyError):
        return "CONFIG"
    if isinstance(exc, ManifestError):
        return "EVIDENCE"
    if isinstance(exc, OSError):
        return "EVIDENCE"
    return "CONFIG"


# --- entry point ------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        description="Prepare v1.0.4 data artefacts (S3-1): history audit, decode cache, B list.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes:\n"
            "  audit          recompute the history audit from meta only\n"
            "  cache          build/repair + re-verify the three-ROI decoded cache\n"
            "  eval-manifest  write a read-only manifest for an external dataset (no manifest)\n"
            "\n"
            "exit codes: 0 ok, 2 known contract/manifest/IO error (stderr 'v104:<CODE>: ...'),\n"
            "            other non-zero for unexpected failures (traceback kept)\n"
        ),
    )
    parser.add_argument("--dataset", required=True, help="dataset directory (vision shards)")
    parser.add_argument("--mode", required=True, choices=("cache", "audit", "eval-manifest"))
    parser.add_argument("--out", required=True, help="output JSON path (manifest path for eval-manifest)")
    parser.add_argument("--depth", type=int, default=3, help="history depth for --mode audit: 1 or 3")
    parser.add_argument(
        "--history-contract",
        default=vision_history.DEFAULT_HISTORY_CONTRACT,
        help="history contract for --mode audit (default: causal-segment-v1)",
    )
    parser.add_argument("--decode-threads", type=int, default=8, help="PNG decode workers")
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="smoke runs only: visit at most N shards (builds/verifies a partial cache)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "audit":
            return mode_audit(args)
        if args.mode == "cache":
            return mode_cache(args)
        return mode_eval_manifest(args)
    except KNOWN_ERRORS as exc:
        message = str(exc)
        if not message.startswith("v104:"):
            message = f"v104:{_classify(exc)}: {message}"
        print(message, file=sys.stderr, flush=True)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
