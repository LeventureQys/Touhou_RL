"""Dataset manifest: the contract between the collect / process / train stages.

Why a manifest at all
---------------------
The three stages used to be three scripts that each re-derived the dataset's
shape by opening every shard.  That works until something drifts: a capture is
interrupted, a teacher config changes, a split is reseeded, and a training run
silently learns from a different dataset than the report claims.  The manifest
pins down *what the dataset is* so that:

* ``process`` writes it, ``train`` refuses to run without a matching one;
* the train/validation split is reproducible from ``(seed, val_fraction)`` and is
  additionally *verified* against a hash of the exact validation indices, so a
  changed dataset cannot silently re-shuffle the split;
* the teacher configuration that produced the labels is recorded, because a
  value label means nothing without the teacher that generated it.

The manifest is plain JSON so it can be diffed and reviewed.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"


class ManifestError(RuntimeError):
    """Raised when a manifest is missing, stale, or inconsistent with the data."""


def split_indices(total: int, seed: int, val_fraction: float) -> tuple[list[int], list[int]]:
    """Deterministic ``(train, val)`` index split.

    Shared by ``process`` (which records its hash) and ``train`` (which verifies
    it), so the two stages cannot disagree about which frames are held out.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(total))
    n_val = int(total * val_fraction)
    if val_fraction > 0 and n_val == 0:
        n_val = 1
    val = sorted(int(index) for index in perm[:n_val])
    train = sorted(int(index) for index in perm[n_val:])
    return train, val


def index_digest(indices: list[int]) -> str:
    payload = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ShardRecord:
    name: str
    frames: int
    bytes: int
    sha256: str


@dataclass
class DatasetManifest:
    version: int
    modality: str  # "privileged" | "vision"
    root: str
    created: str
    total_frames: int
    shards: list[ShardRecord] = field(default_factory=list)
    frame_shape: list[int] | None = None
    has_teacher_values: bool = False
    teacher: dict = field(default_factory=dict)
    splits: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        modality: str,
        root: Path,
        shards: list[ShardRecord],
        frame_shape: list[int] | None,
        has_teacher_values: bool,
        teacher: dict,
        seed: int,
        val_fraction: float,
        stats: dict,
        provenance: dict,
    ) -> "DatasetManifest":
        total = sum(shard.frames for shard in shards)
        _train, val = split_indices(total, seed, val_fraction)
        return cls(
            version=MANIFEST_VERSION,
            modality=modality,
            root=str(root),
            created=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_frames=total,
            shards=shards,
            frame_shape=frame_shape,
            has_teacher_values=has_teacher_values,
            teacher=teacher,
            splits={
                "seed": int(seed),
                "val_fraction": float(val_fraction),
                "val_frames": len(val),
                "val_index_sha256": index_digest(val),
            },
            stats=stats,
            provenance=provenance,
        )

    # -- io ----------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path is not None else Path(self.root) / MANIFEST_NAME
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "DatasetManifest":
        path = Path(path)
        if not path.exists():
            raise ManifestError(f"no manifest at {path}; run the process stage first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != MANIFEST_VERSION:
            raise ManifestError(
                f"manifest version {payload.get('version')} != {MANIFEST_VERSION}; re-run process"
            )
        shards = [ShardRecord(**row) for row in payload.get("shards", [])]
        return cls(
            version=payload["version"],
            modality=payload["modality"],
            root=payload["root"],
            created=payload.get("created", ""),
            total_frames=payload["total_frames"],
            shards=shards,
            frame_shape=payload.get("frame_shape"),
            has_teacher_values=bool(payload.get("has_teacher_values")),
            teacher=payload.get("teacher", {}),
            splits=payload.get("splits", {}),
            stats=payload.get("stats", {}),
            provenance=payload.get("provenance", {}),
        )

    # -- validation --------------------------------------------------------
    def verify(self, dataset_dir: Path, *, deep: bool = False) -> list[int]:
        """Check the directory still matches the manifest; return ``(train, val)``.

        ``deep`` re-hashes every shard, which is slow; the default checks presence
        and frame counts, which catches the realistic failures (a capture was
        deleted, truncated, or appended to).
        """
        dataset_dir = Path(dataset_dir)
        train, val = split_indices(self.total_frames, self.splits["seed"], self.splits["val_fraction"])
        if index_digest(val) != self.splits["val_index_sha256"]:
            raise ManifestError(
                "split reproducibility check failed: the recorded validation index hash "
                "does not match the split derived from (seed, val_fraction)"
            )
        total = 0
        for record in self.shards:
            path = dataset_dir / record.name
            if not path.exists():
                raise ManifestError(f"shard missing: {record.name}")
            actual = path.stat().st_size
            if actual != record.bytes:
                raise ManifestError(
                    f"shard size changed: {record.name} is {actual} B, manifest says {record.bytes} B"
                )
            if deep and file_digest(path) != record.sha256:
                raise ManifestError(f"shard hash changed: {record.name}")
            total += record.frames
        if total != self.total_frames:
            raise ManifestError(f"frame count changed: {total} on disk vs {self.total_frames} in manifest")
        # Files added *after* processing must also fail: the trainers scan the
        # directory, so an unlisted shard would be trained on while the manifest
        # (and therefore every report) claims otherwise.
        try:
            from ..dataset import discover_shards
            from ..vision_data import discover_vision_shards

            discovered = (
                discover_vision_shards(dataset_dir)
                if self.modality == "vision"
                else discover_shards(dataset_dir)
            )
            on_disk = {path.name for path in discovered}
        except Exception:  # discovery is best-effort; never mask the checks above
            on_disk = set()
        expected = {record.name for record in self.shards}
        extra = sorted(on_disk - expected)
        if extra:
            raise ManifestError(
                f"{len(extra)} shard(s) on disk are not in the manifest (re-run process): {extra[:3]}"
            )
        return train, val

    def summary(self) -> dict:
        return {
            "modality": self.modality,
            "root": self.root,
            "total_frames": self.total_frames,
            "shards": len(self.shards),
            "teacher": self.teacher.get("name", "") if self.teacher else "",
            "has_teacher_values": self.has_teacher_values,
            "frame_shape": self.frame_shape,
            "splits": {key: value for key, value in self.splits.items() if key != "val_index_sha256"},
            "stats": self.stats,
        }
