"""obs-v2 dataset root builder/loader (EXP-P3-008).

obs-v2 **adds** the 8-decision action history to a v1 corpus; it does not copy the
1.9 GiB of ``play64``/``hud`` planes.  A *new root* therefore stores only the added
fields, plus enough provenance to prove that the bytes it extends are the v1 bytes:

    <root>/manifest.json                      totals + obs-v2 digest + source roots
    <root>/shards/<key>/act_hist.npy           (N,8,18) uint8  the added field
    <root>/shards/<key>/stage.npy              (N,) int32      stage per decision
    <root>/shards/<key>/shard.json             provenance + checks for that shard

The new root is built by :func:`build_obs_v2_root`, which

1. validates every source shard through the **frozen v1 loader**
   (:func:`touhou_rl.v400.student.dataset.load_teacher_shard`: schema, obs-v1
   digest, alignment ``frames[d-4..d-1] -> mask[d]``, plane shapes/sizes and the
   recorded sha256 of both planes), then
2. reconstructs the executed action per decision from the **raw** label rows
   (DAgger: ``dagger.executed_cand``; D0: ``chosen_plan.first_action``) and builds
   the history with the same shard/stage-boundary rule the spec documents, then
3. writes the history and re-verifies the shard count, the per-row decision/frame
   alignment and the v1 plane sha256 **before** the shard is marked complete.

"Same samples/alignment as v1, only the added field differs" is therefore a checked
statement, not a description: :func:`verify_obs_v2_root` re-runs steps 1 and 3 against
the live sources and reports a per-shard verdict.

Run::

    python -m touhou_rl.v400.student.obs_v2_dataset build \
      --root .tmp_v400_d0/d0 --root .tmp_v400_dagger/d0_extra \
      --out .tmp_v400_obsv2/obsv2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from touhou_rl.v400.student import obs_spec, obs_spec_v2
from touhou_rl.v400.student.dataset import (
    DatasetError,
    load_teacher_manifest,
    load_teacher_shard,
)

MANIFEST_SCHEMA = "v400-obs-v2-manifest-1"
SHARD_SCHEMA = "v400-obs-v2-shard-1"
#: History arrays are written with an explicit name so a v1 plane cannot be mistaken
#: for the added field (fail closed on a missing/replaced file).
ACT_HIST_NAME = "act_hist.npy"
STAGE_NAME = "stage.npy"


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_raw_labels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_number}: bad JSON: {exc}") from exc
    return rows


def _complete_shards(root: Path) -> list[Path]:
    """``<root>/shards/<key>`` directories whose ``shard.json`` is complete."""

    shards_dir = Path(root) / "shards"
    if not shards_dir.is_dir():
        raise DatasetError(f"{root}: no shards/ directory")
    out: list[Path] = []
    for entry in sorted(shards_dir.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "shard.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("complete") is not True:
            print(f"skip shard {entry.name}: incomplete", flush=True)
            continue
        out.append(entry)
    if not out:
        raise DatasetError(f"{root}: no complete shard")
    return out


def _executed_actions(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(executed_cand[N], stage[N], provenance)`` from one shard's raw rows."""

    cand, stage = obs_spec_v2.reconstruct_from_label_rows(list(rows))
    provenance = {
        "executed_source": ("dagger.executed_cand"
                            if any((row.get("dagger") or {}).get("executed_cand") is not None
                                   for row in rows) else "chosen_plan.first_action"),
        "executed_cand_histogram": np.bincount(
            cand, minlength=obs_spec.N_MOVE_CLASSES).tolist() if len(cand) else [],
        "stage_values": sorted({int(value) for value in stage}),
        "stage_changes": int(np.sum(np.asarray(stage)[1:] != np.asarray(stage)[:-1]))
        if len(stage) > 1 else 0,
    }
    return cand, stage, provenance


def build_obs_v2_root(
    roots: Iterable[str | Path],
    out_root: str | Path,
    *,
    verify_sha: bool = True,
) -> dict[str, Any]:
    """Build the obs-v2 root for every complete source shard under ``roots``."""

    roots = [Path(root) for root in roots]
    out_root = Path(out_root)
    if out_root.exists():
        # Never silently merge into a previous build: a half-written root would let a
        # stale act_hist.npy pair with a fresh manifest.
        existing = list(out_root.iterdir()) if out_root.is_dir() else []
        if existing:
            raise DatasetError(
                f"{out_root} already exists and is not empty; remove it explicitly or "
                "pick a new --out (the builder never overwrites a root)")
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows_out: list[dict[str, Any]] = []
    totals = {"shards": 0, "decisions": 0, "act_hist_bytes": 0, "stage_bytes": 0}
    source_manifests: list[dict[str, Any]] = []
    seen_keys: dict[str, Path] = {}

    for root in roots:
        manifest = load_teacher_manifest(root)
        if manifest.get("obs_spec_version") != obs_spec.OBS_SPEC_VERSION:
            raise DatasetError(
                f"{root}: manifest obs spec {manifest.get('obs_spec_version')!r} is not "
                f"{obs_spec.OBS_SPEC_VERSION!r}; obs-v2 extends obs-v1 only")
        source_manifests.append({
            "root": str(root.resolve()),
            "manifest_sha256": _sha256_file(root / "manifest.json"),
            "totals": manifest.get("totals"),
        })
        shard_dirs = _complete_shards(root)
        for shard_dir in shard_dirs:
            key = shard_dir.name
            if key in seen_keys:
                raise DatasetError(
                    f"duplicate shard key {key!r} under {seen_keys[key]} and {shard_dir}")
            seen_keys[key] = shard_dir
            # 1) authoritative v1 validation (schema/digest/alignment/plane bytes)
            shard = load_teacher_shard(shard_dir, verify_sha=verify_sha)
            raw_rows = _read_raw_labels(shard_dir / "labels.jsonl")
            if len(raw_rows) != len(shard.records):
                raise DatasetError(
                    f"{shard_dir}: {len(raw_rows)} raw rows != {len(shard.records)} "
                    "validated rows")
            for index, row in enumerate(raw_rows):
                if row.get("kind") != "decision":
                    raise DatasetError(f"{shard_dir}: row {index} kind is not decision")
                if int(row.get("decision", -1)) != index:
                    raise DatasetError(
                        f"{shard_dir}: row {index} decision={row.get('decision')!r}")
                expected_frame = index * shard.frames_per_decision
                frame = row.get("decision_frame", row.get("frame"))
                if int(frame) != expected_frame:
                    raise DatasetError(
                        f"{shard_dir}: row {index} frame {frame!r} != {expected_frame} "
                        f"({shard.frames_per_decision} frames per decision)")
            # 2) reconstruct the added field
            cand, stage, provenance = _executed_actions(raw_rows)
            history = obs_spec_v2.build_action_history(cand, stage)
            obs_spec_v2.check_action_history(history)
            if history.shape[0] != len(shard.records):
                raise DatasetError(
                    f"{shard_dir}: history rows {history.shape[0]} != "
                    f"{len(shard.records)} decisions")
            # 3) write and re-verify
            target = out_root / "shards" / key
            (target).mkdir(parents=True, exist_ok=True)
            np.save(target / ACT_HIST_NAME, history)
            np.save(target / STAGE_NAME, stage.astype(np.int32))
            reloaded = np.load(target / ACT_HIST_NAME)
            if reloaded.shape != history.shape or reloaded.dtype != np.uint8:
                raise DatasetError(f"{target}: reloaded act_hist shape/dtype mismatch")
            if not np.array_equal(reloaded, history):
                raise DatasetError(f"{target}: reloaded act_hist bytes differ")
            zeros_per_row = (history.sum(axis=-1) == 0).sum(axis=1)
            first_full = int(np.argmax(zeros_per_row == 0)) if (zeros_per_row == 0).any() else -1
            record = {
                "schema": SHARD_SCHEMA,
                "shard": key,
                "source_root": str(root.resolve()),
                "source_dir": str(shard_dir.resolve()),
                "decisions": int(len(shard.records)),
                "frames_per_decision": int(shard.frames_per_decision),
                "decision_frames": int(shard.meta.get("decision_frames", obs_spec.DECISION_FRAMES)),
                "stage": shard.meta.get("stage"),
                "difficulty": shard.meta.get("difficulty"),
                "seed": shard.meta.get("seed"),
                "dagger_round": (shard.meta.get("dagger") or {}).get("round"),
                "obs_spec_version": obs_spec.OBS_SPEC_VERSION,
                "obs_spec_digest": obs_spec.obs_spec_digest(),
                "obs_v2_spec_version": obs_spec_v2.OBS_SPEC_VERSION,
                "obs_v2_spec_digest": obs_spec_v2.obs_spec_v2_digest(),
                "act_hist": {
                    "path": ACT_HIST_NAME,
                    "shape": list(history.shape),
                    "dtype": "uint8",
                    "bytes": int((target / ACT_HIST_NAME).stat().st_size),
                    "sha256": _sha256_file(target / ACT_HIST_NAME),
                },
                "stage_plane": {
                    "path": STAGE_NAME,
                    "shape": list(stage.shape),
                    "dtype": "int32",
                    "bytes": int((target / STAGE_NAME).stat().st_size),
                    "sha256": _sha256_file(target / STAGE_NAME),
                },
                "v1_planes": {
                    name: {
                        "sha256": meta.get("sha256"),
                        "bytes": meta.get("bytes"),
                    }
                    for name, meta in (shard.meta.get("obs") or {}).items()
                    if isinstance(meta, dict)
                },
                "v1_plane_sha256_verified_now": bool(verify_sha),
                "alignment": {
                    "decision_index": "row i has decision == i",
                    "frame": f"row i has decision_frame == i * {shard.frames_per_decision}",
                    "window": str((shard.meta.get("alignment") or {}).get("p3_window")),
                    "observation_time": str(
                        (shard.meta.get("alignment") or {}).get("observation_time")),
                },
                "history": {
                    "zero_rows_at_start": int(zeros_per_row[0]) if len(zeros_per_row) else 0,
                    "rows_with_padding": int((zeros_per_row > 0).sum()),
                    "first_unpadded_row": first_full,
                    "segments": 1 + provenance["stage_changes"],
                    **provenance,
                },
            }
            (target / "shard.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            rows_out.append({
                "shard": key,
                "dir": str(target.resolve()),
                "source_root": record["source_root"],
                "decisions": record["decisions"],
                "seed": record["seed"],
                "stage": record["stage"],
                "difficulty": record["difficulty"],
                "dagger_round": record["dagger_round"],
                "act_hist_sha256": record["act_hist"]["sha256"],
                "stage_sha256": record["stage_plane"]["sha256"],
            })
            totals["shards"] += 1
            totals["decisions"] += record["decisions"]
            totals["act_hist_bytes"] += record["act_hist"]["bytes"]
            totals["stage_bytes"] += record["stage_plane"]["bytes"]
            print(f"built {key}: {record['decisions']} decisions, "
                  f"{record['history']['segments']} segment(s), "
                  f"padding rows {record['history']['rows_with_padding']}", flush=True)

    # The v1 totals are the contract obs-v2 must not change.
    expected_decisions = sum(int((row.get("totals") or {}).get("decisions", 0))
                             for row in source_manifests)
    if expected_decisions != totals["decisions"]:
        raise DatasetError(
            f"obs-v2 root holds {totals['decisions']} decisions but the source "
            f"manifests total {expected_decisions}")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "kind": "obs-v2-action-history",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "obs_spec_version": obs_spec_v2.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec_v2.obs_spec_v2_digest(),
        "base_obs_spec_version": obs_spec.OBS_SPEC_VERSION,
        "base_obs_spec_digest": obs_spec.obs_spec_digest(),
        "base_obs_spec_digest_literal": obs_spec_v2.OBS_V1_DIGEST,
        "adds_only": [
            "act_hist (N,8,18) uint8: one-hot of the 8 actions executed *before* each "
            "decision, oldest first, cleared at shard/stage boundaries",
        ],
        "reserved_not_populated": {
            "self_state": obs_spec_v2.SELF_STATE_REASON,
            "danger_summary": obs_spec_v2.DANGER_SUMMARY_REASON,
        },
        "planes_copied": False,
        "planes_note": "the v1 planes stay in their own roots; this root adds the "
                       "history and records the sha256 of the v1 planes it verified",
        "sources": source_manifests,
        "totals": {**totals, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "shards": rows_out,
        "build_seconds": round(time.monotonic() - started, 3),
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_obs_v2_manifest(root: str | Path) -> dict[str, Any]:
    """Read and validate ``<root>/manifest.json``."""

    root = Path(root)
    path = root / "manifest.json"
    if not path.is_file():
        raise DatasetError(f"missing obs-v2 manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise DatasetError(
            f"{path}: schema {payload.get('schema')!r} != {MANIFEST_SCHEMA!r}")
    try:
        obs_spec_v2.check_obs_spec_v2(payload.get("obs_spec_version"),
                                      payload.get("obs_spec_digest"))
    except obs_spec_v2.ObsSpecV2Error as error:
        raise DatasetError(f"{path}: obs-v2 spec check failed: {error}") from error
    if payload.get("base_obs_spec_digest") != obs_spec_v2.OBS_V1_DIGEST:
        raise DatasetError(
            f"{path}: base obs-v1 digest {payload.get('base_obs_spec_digest')!r} is not "
            f"the frozen {obs_spec_v2.OBS_V1_DIGEST!r}")
    shards = payload.get("shards") or []
    if not shards:
        raise DatasetError(f"{path}: manifest lists no shards")
    if int((payload.get("totals") or {}).get("decisions", -1)) != sum(
            int(row.get("decisions", 0)) for row in shards):
        raise DatasetError(f"{path}: totals do not sum the shard rows")
    return payload


class ObsV2Shard:
    """One shard's added fields, memory-mapped."""

    def __init__(self, root: Path, row: dict[str, Any]) -> None:
        self.root = Path(root)
        self.row = dict(row)
        self.key = str(row["shard"])
        self.dir = Path(row["dir"]) if row.get("dir") else self.root / "shards" / self.key
        meta = json.loads((self.dir / "shard.json").read_text(encoding="utf-8"))
        if meta.get("schema") != SHARD_SCHEMA:
            raise DatasetError(f"{self.dir}: shard schema {meta.get('schema')!r}")
        if meta.get("obs_v2_spec_digest") != obs_spec_v2.obs_spec_v2_digest():
            raise DatasetError(f"{self.dir}: obs-v2 digest mismatch")
        self.meta = meta
        self.decisions = int(meta["decisions"])
        if self.decisions != int(row.get("decisions", -1)):
            raise DatasetError(f"{self.dir}: {self.decisions} decisions != manifest "
                               f"{row.get('decisions')!r}")
        self.act_hist = np.load(self.dir / ACT_HIST_NAME, mmap_mode="r")
        self.stage = np.load(self.dir / STAGE_NAME, mmap_mode="r")
        expected = (self.decisions, obs_spec_v2.ACTION_HISTORY_DECISIONS,
                    obs_spec.N_MOVE_CLASSES)
        if self.act_hist.shape != expected or self.act_hist.dtype != np.uint8:
            raise DatasetError(f"{self.dir}: act_hist {self.act_hist.shape} "
                               f"{self.act_hist.dtype} != {expected} uint8")
        if self.stage.shape != (self.decisions,):
            raise DatasetError(f"{self.dir}: stage {self.stage.shape} != ({self.decisions},)")

    def history(self) -> np.ndarray:
        return np.array(self.act_hist, dtype=np.uint8)

    def report(self) -> dict[str, Any]:
        return {
            "shard": self.key,
            "decisions": self.decisions,
            "source_root": self.meta.get("source_root"),
            "act_hist_sha256": (self.meta.get("act_hist") or {}).get("sha256"),
            "segments": (self.meta.get("history") or {}).get("segments"),
            "executed_source": (self.meta.get("history") or {}).get("executed_source"),
        }


def load_obs_v2_shards(root: str | Path) -> dict[str, ObsV2Shard]:
    root = Path(root)
    manifest = load_obs_v2_manifest(root)
    shards: dict[str, ObsV2Shard] = {}
    for row in manifest["shards"]:
        shard = ObsV2Shard(root, row)
        if shard.key in shards:
            raise DatasetError(f"duplicate shard {shard.key!r} in {root}")
        shards[shard.key] = shard
    return shards


def verify_obs_v2_root(
    root: str | Path,
    *,
    roots: Iterable[str | Path] | None = None,
    verify_sha: bool = True,
) -> dict[str, Any]:
    """Re-check every claim the builder made, against the live v1 sources.

    ``roots`` defaults to the ``sources`` recorded in the manifest.  For every
    shard this re-runs the frozen v1 loader (which re-hashes both planes), rebuilds
    the history from the raw labels and compares it byte for byte, and checks the
    sample count and the per-row decision/frame alignment again.
    """

    root = Path(root)
    manifest = load_obs_v2_manifest(root)
    source_roots = ([Path(item) for item in roots] if roots is not None
                    else [Path(item["root"]) for item in manifest.get("sources") or []])
    source_dirs: dict[str, Path] = {}
    for source_root in source_roots:
        for shard_dir in _complete_shards(source_root):
            source_dirs[shard_dir.name] = shard_dir
    by_key = {str(row["shard"]): row for row in manifest["shards"]}
    missing = sorted(set(by_key) - set(source_dirs))
    if missing:
        raise DatasetError(f"source root(s) no longer provide shards {missing[:3]}")
    verdicts: list[dict[str, Any]] = []
    rebuilt_total = 0
    for key in sorted(by_key):
        row = by_key[key]
        shard_dir = source_dirs[key]
        shard = load_teacher_shard(shard_dir, verify_sha=verify_sha)
        raw_rows = _read_raw_labels(shard_dir / "labels.jsonl")
        cand, stage, _ = _executed_actions(raw_rows)
        expected = obs_spec_v2.build_action_history(cand, stage)
        stored = np.load(Path(row["dir"]) / ACT_HIST_NAME)
        aligned = all(int(raw.get("decision_frame", raw.get("frame", -1)))
                      == index * shard.frames_per_decision
                      for index, raw in enumerate(raw_rows))
        count_equal = (len(shard.records) == int(row["decisions"]) == stored.shape[0])
        stage_equal = np.array_equal(np.load(Path(row["dir"]) / STAGE_NAME),
                                    stage.astype(np.int32))
        history_equal = np.array_equal(stored, expected)
        verdicts.append({
            "shard": key,
            "decisions": int(row["decisions"]),
            "v1_decisions": len(shard.records),
            "count_equal": bool(count_equal),
            "alignment_ok": bool(aligned),
            "v1_planes_sha256_ok": bool(verify_sha),
            "history_rebuild_equal": bool(history_equal),
            "stage_equal": bool(stage_equal),
        })
        rebuilt_total += int(row["decisions"])
        if not (count_equal and aligned and history_equal and stage_equal):
            raise DatasetError(f"{key}: obs-v2 root does not reproduce v1+history")
    returned = {
        "schema": "v400-obs-v2-verify-1",
        "root": str(root.resolve()),
        "obs_spec_version": obs_spec_v2.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec_v2.obs_spec_v2_digest(),
        "base_obs_spec_digest": obs_spec.obs_spec_digest(),
        "shards": len(verdicts),
        "decisions": rebuilt_total,
        "manifest_decisions": int((manifest.get("totals") or {}).get("decisions", -1)),
        "verified": True,
        "per_shard": verdicts,
    }
    return returned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root", action="append", required=True,
                       help="v1 D0-shaped root (repeatable, in split-independent order)")
    build.add_argument("--out", required=True)
    build.add_argument("--no-sha", action="store_true",
                       help="skip re-hashing the v1 planes (dev only; the report then "
                            "says v1_planes_sha256_ok=False)")
    verify = sub.add_parser("verify")
    verify.add_argument("--root", required=True)
    verify.add_argument("--source", action="append", default=None)
    verify.add_argument("--no-sha", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build_obs_v2_root(args.root, args.out, verify_sha=not args.no_sha)
        print(json.dumps({"out": args.out, "totals": manifest["totals"],
                          "obs_spec_digest": manifest["obs_spec_digest"]},
                         indent=2, ensure_ascii=False))
        return 0
    report = verify_obs_v2_root(args.root, roots=args.source,
                                verify_sha=not args.no_sha)
    print(json.dumps({key: report[key] for key in
                      ("root", "shards", "decisions", "manifest_decisions", "verified")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())


__all__ = [
    "ACT_HIST_NAME",
    "MANIFEST_SCHEMA",
    "ObsV2Shard",
    "SHARD_SCHEMA",
    "STAGE_NAME",
    "build_obs_v2_root",
    "load_obs_v2_manifest",
    "load_obs_v2_shards",
    "verify_obs_v2_root",
]
