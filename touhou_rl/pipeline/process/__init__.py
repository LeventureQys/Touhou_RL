"""Pipeline stage 2 - process: validate raw shards and write the dataset manifest.

What "process" means here
-------------------------
Not "make the data nicer" but "make the data *known*": every downstream training
run must be able to state exactly which frames, which labels and which teacher it
consumed.  Concretely this stage

1. **validates** every shard (required keys, dtypes, action ranges, emptiness);
2. **summarises** the dataset cheaply (frame counts, stage histogram, action
   distribution) without loading image tensors;
3. **computes teacher values** when the modality needs them and they are not
   already present - the labels are a pure function of the privileged arrays, so
   re-deriving them is free and keeps video out of the label path;
4. **pins the split** to ``(seed, val_fraction)`` plus a hash of the validation
   indices, so the train stage cannot silently train on different frames;
5. writes ``manifest.json``.

Idempotent: re-running on the same directory produces the same manifest apart
from the timestamp, and refuses to run until the shard set is stable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ... import obs
from ..manifest import DatasetManifest, ManifestError, ShardRecord, file_digest

REQUIRED_PRIVILEGED = {"self_feat", "bullets", "danger", "action", "meta"}
# Vision shards store native PNG blobs (frame_blob/frame_offset/frame_hw); the
# older 128x112 capture path stored a dense `frame` array.  Both are accepted.
REQUIRED_VISION = {"self_feat", "bullets", "danger", "action", "meta"}
FRAME_KEYS_BLOB = {"frame_blob", "frame_offset", "frame_hw"}
ACTION_DIMS = (9, 2, 2, 2)


@dataclass
class ProcessSpec:
    raw_dir: Path
    out_dir: Path | None = None  # None = write the manifest next to the shards
    modality: str | None = None  # inferred when omitted
    compute_teacher: bool = True
    seed: int = 7
    val_fraction: float = 0.1
    min_frames: int = 64
    min_free_bytes: int = 0
    verbose: bool = True
    teacher_name: str = "pilot"
    teacher_config: dict = field(default_factory=dict)


def _discover(raw_dir: Path, modality: str | None) -> tuple[str, list[Path]]:
    from ...dataset import discover_shards
    from ...vision_data import discover_vision_shards

    vision_paths = discover_vision_shards(raw_dir)
    privileged_paths = discover_shards(raw_dir)
    # vision shards also carry `action`, so they are a superset; decide by the
    # presence of `frame` rather than by which list is longer.
    if modality is None:
        modality = "vision" if vision_paths else "privileged"
    paths = vision_paths if modality == "vision" else privileged_paths
    if not paths:
        raise ManifestError(f"no {modality} shards under {raw_dir}")
    return modality, paths


def _validate_shard(path: Path, modality: str) -> dict:
    required = REQUIRED_VISION if modality == "vision" else REQUIRED_PRIVILEGED
    with np.load(path) as data:
        missing = required - set(data.files)
        if missing:
            raise ManifestError(f"{path.name}: missing keys {sorted(missing)}")
        frame_reference = None
        if modality == "vision":
            files = set(data.files)
            if "region_name" in files:
                names = [str(value) for value in data["region_name"]]
                shapes = data["region_hw"]
                for name, (height, width) in zip(names, shapes):
                    blob_key, offset_key = f"{name}_blob", f"{name}_offset"
                    if blob_key not in files or offset_key not in files:
                        raise ManifestError(f"{path.name}: missing payload for region {name!r}")
                    offsets = data[offset_key]
                    if offsets.shape[0] != data["action"].shape[0] + 1:
                        raise ManifestError(f"{path.name}: {offset_key} length mismatch")
                    if offsets[-1] != data[blob_key].shape[0]:
                        raise ManifestError(f"{path.name}: {offset_key} does not cover {blob_key}")
                frame_reference = tuple(int(v) for v in shapes[0])
            elif "frame" in files:
                frame_reference = tuple(int(v) for v in data["frame"].shape[1:3])
            else:
                raise ManifestError(
                    f"{path.name}: no frame payload (expected <region>_blob/<region>_offset or frame)"
                )
        action = data["action"]
        if action.ndim != 2 or action.shape[1] != 4:
            raise ManifestError(f"{path.name}: action shape {action.shape}, expected (n, 4)")
        for axis, limit in enumerate(ACTION_DIMS):
            column = action[:, axis]
            if column.size and int(column.max()) >= limit:
                raise ManifestError(
                    f"{path.name}: action[{axis}] max {int(column.max())} >= {limit}"
                )
        frames = int(action.shape[0])
        if frames == 0:
            raise ManifestError(f"{path.name}: empty shard")
        if data["self_feat"].shape != (frames, obs.N_SELF):
            raise ManifestError(f"{path.name}: self_feat shape {data['self_feat'].shape}")
        if data["bullets"].shape != (frames, obs.N_BULLETS, obs.N_BULLET_FEATURES):
            raise ManifestError(f"{path.name}: bullets shape {data['bullets'].shape}")
        if data["danger"].shape != (frames, obs.DANGER_ROWS, obs.DANGER_COLS):
            raise ManifestError(f"{path.name}: danger shape {data['danger'].shape}")
        if float(np.abs(data["self_feat"]).max(initial=0.0)) > 1e6:
            raise ManifestError(f"{path.name}: self_feat contains absurd values (wrong capture?)")
        row = {
            "frames": frames,
            "has_value": "value" in data.files,
            "frame_shape": [*frame_reference, 3] if modality == "vision" and frame_reference else None,
            "stage_hist": {},
            "move_hist": [0] * ACTION_DIMS[0],
            "focus_rate": 0.0,
            "bomb_frames": 0,
            "deaths": 0,
        }
        meta = data["meta"]
        if meta.size:
            stages, counts = np.unique(meta[:, 1].astype(int), return_counts=True)
            row["stage_hist"] = {int(stage): int(count) for stage, count in zip(stages, counts)}
            row["deaths"] = int(meta[:, 7].max() - meta[:, 7].min()) if frames > 1 else 0
        moves = np.bincount(action[:, 0].astype(np.int64), minlength=ACTION_DIMS[0])
        row["move_hist"] = [int(value) for value in moves[: ACTION_DIMS[0]]]
        row["focus_rate"] = round(float(action[:, 2].mean()), 4) if frames else 0.0
        row["bomb_frames"] = int(action[:, 3].sum())
    row["sha256"] = file_digest(path)
    row["bytes"] = path.stat().st_size
    return row


def _aggregate(rows: list[dict], modality: str) -> tuple[list[ShardRecord], dict]:
    records = []
    stage_hist: dict[int, int] = {}
    move_hist = [0] * ACTION_DIMS[0]
    focus_total = 0
    bomb_frames = 0
    has_value = False
    frame_shape = None
    total = 0
    for name, row in rows:
        records.append(ShardRecord(name=name, frames=row["frames"], bytes=row["bytes"], sha256=row["sha256"]))
        total += row["frames"]
        for stage, count in row["stage_hist"].items():
            stage_hist[stage] = stage_hist.get(stage, 0) + count
        move_hist = [a + b for a, b in zip(move_hist, row["move_hist"])]
        focus_total += int(round(row["focus_rate"] * row["frames"]))
        bomb_frames += row["bomb_frames"]
        has_value = has_value or row["has_value"]
        frame_shape = frame_shape or row["frame_shape"]
    stats = {
        "total_frames": total,
        "stage_hist": dict(sorted(stage_hist.items())),
        "move_hist": move_hist,
        "focus_rate": round(focus_total / total, 4) if total else 0.0,
        "bomb_frames": bomb_frames,
        "frame_shape": frame_shape if modality == "vision" else None,
    }
    return records, {**stats, "has_value": has_value}


def run_process(spec: ProcessSpec) -> dict:
    raw_dir = Path(spec.raw_dir)
    modality, paths = _discover(raw_dir, spec.modality)
    rows: list[tuple[str, dict]] = []
    for path in paths:
        if spec.verbose:
            print(f"[process] validating {path.name}", flush=True)
        rows.append((path.name, _validate_shard(path, modality)))
    records, stats = _aggregate(rows, modality)
    total = stats["total_frames"]
    if total < spec.min_frames:
        raise ManifestError(f"dataset has {total} frames, below the {spec.min_frames} minimum")

    teacher = {"name": "", "config": {}}
    has_values = stats["has_value"]
    if spec.compute_teacher and not has_values:
        teacher = {"name": spec.teacher_name, "config": spec.teacher_config}
        cache = raw_dir.parent / f"{raw_dir.name}-teacher-values.npz"
        started = time.time()
        if modality == "vision":
            from ...vision_data import VisionDataset
            from ...vision_train import teacher_values_for

            teacher_values_for(VisionDataset(raw_dir), cache, verbose=spec.verbose)
        else:
            from ...dataset import ReplayDataset
            from ...distill import label_dataset
            from ...pilot import AvoidPilot, PilotConfig

            label_dataset(
                ReplayDataset(raw_dir),
                cache,
                AvoidPilot(PilotConfig(switch_penalty=0.0)),
                verbose=spec.verbose,
            )
        has_values = True
        stats["teacher_label_seconds"] = round(time.time() - started, 1)
        stats["teacher_values"] = str(cache)
    elif has_values:
        teacher = {"name": "captured", "config": {}}

    provenance = {}
    collect_report = raw_dir / "collect-report.json"
    if collect_report.exists():
        try:
            provenance["collect"] = json.loads(collect_report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            provenance["collect"] = {"error": "unreadable collect-report.json"}

    manifest = DatasetManifest.build(
        modality=modality,
        root=raw_dir,
        shards=records,
        frame_shape=stats.get("frame_shape"),
        has_teacher_values=has_values,
        teacher=teacher,
        seed=spec.seed,
        val_fraction=spec.val_fraction,
        stats={key: value for key, value in stats.items() if key != "has_value"},
        provenance=provenance,
    )
    out_dir = Path(spec.out_dir) if spec.out_dir else raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = manifest.save(out_dir / "manifest.json")
    if spec.verbose:
        print(f"[process] manifest written: {path}", flush=True)
        print(json.dumps(manifest.summary(), ensure_ascii=False, indent=2), flush=True)
    return {"manifest": str(path), **manifest.summary()}


__all__ = ["ProcessSpec", "run_process"]
