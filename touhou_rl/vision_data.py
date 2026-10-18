"""On-disk format for vision shards: frames plus privileged labels.

Why a second format instead of extending :mod:`touhou_rl.dataset`
-----------------------------------------------------------------
The privileged shards store ~500 B/frame; a vision shard stores 43 KB/frame.
Naively adding a ``frame`` key to the existing format would make the replay
dataset (112k frames) 4.8 GB and would force a full in-memory cache that no
longer fits.  So the vision format keeps the *same privileged arrays* (so labels
and provenance stay comparable) but is loaded per shard instead of cached whole.

Layout per shard
----------------
=====================  ====================  ==============================
key                    shape/dtype           meaning
=====================  ====================  ==============================
``frame``              u8  [n, H, W, 3]      :mod:`touhou_rl.vision` tensor
``self_feat``          f32 [n, 24]           privileged observation (label)
``bullets``            f16 [n, 64, 6]        privileged label
``danger``             u8  [n, 16, 8]        privileged label
``action``             u8  [n, 4]            expert action (human replay)
``value``              f16 [n, 18]           teacher value per candidate (optional)
``meta``               f32 [n, 8]            frame, stage, score, graze,
                                             lives, bombs, power, deaths
=====================  ====================  ==============================

``value`` is optional on purpose: human replay captures have expert *actions*
(what the human pressed), teacher captures have *values* (what each candidate
would be worth).  Both can train a policy; keeping them in different keys stops
the two label types from being silently confused.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import obs, vision

VISION_CORE_KEYS = ("frame", "self_feat", "bullets", "danger", "action", "meta")
VISION_OPTIONAL_KEYS = ("value",)
META_FIELDS = ("frame", "stage", "score", "graze", "lives", "bombs", "power", "deaths")


@dataclass
class VisionShardWriter:
    """Accumulates frames in memory and writes one ``.npz`` per shard."""

    out_dir: Path
    shard_size: int = 512
    tag: str = "vision"
    provenance: dict = field(default_factory=dict)
    allow_append: bool = False

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.out_dir.glob(f"{self.tag}-*.npz"))
        if existing and not self.allow_append:
            raise FileExistsError(
                f"{self.out_dir} already contains {len(existing)} '{self.tag}' shards; "
                "pass allow_append=True (or --overwrite) to append"
            )
        self._frames: list[np.ndarray] = []
        self._self = np.zeros((self.shard_size, obs.N_SELF), dtype=np.float32)
        self._bullets = np.zeros((self.shard_size, obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float16)
        self._danger = np.zeros((self.shard_size, obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8)
        self._action = np.zeros((self.shard_size, 4), dtype=np.uint8)
        self._value = np.zeros((self.shard_size, 18), dtype=np.float16)
        self._meta = np.zeros((self.shard_size, len(META_FIELDS)), dtype=np.float32)
        self._has_value = False
        self._n = 0
        self.total = 0
        self.shards: list[str] = []
        self._index = len(existing) if (existing and self.allow_append) else 0
        self.started = time.time()

    def add(
        self,
        frame: np.ndarray,
        observation: obs.Observation,
        action: tuple[int, int, int, int],
        meta: dict,
        value: np.ndarray | None = None,
    ) -> None:
        if frame.dtype != np.uint8 or frame.shape != (vision.VISION_H, vision.VISION_W, vision.VISION_CHANNELS):
            raise ValueError(f"unexpected frame tensor {frame.shape}/{frame.dtype}")
        index = self._n
        self._frames.append(np.ascontiguousarray(frame))
        self._self[index] = observation.self_feat
        self._bullets[index] = observation.bullets
        self._danger[index] = observation.danger_map
        self._action[index] = np.asarray(action, dtype=np.uint8)
        self._meta[index] = [float(meta[name]) for name in META_FIELDS]
        if value is not None:
            self._value[index] = np.asarray(value, dtype=np.float16).reshape(-1)[:18]
            self._has_value = True
        self._n += 1
        self.total += 1
        if self._n >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if self._n == 0:
            return
        n = self._n
        name = f"{self.tag}-{self._index:05d}.npz"
        self._index += 1
        path = self.out_dir / name
        payload = {
            "frame": np.stack(self._frames[:n]),
            "self_feat": self._self[:n],
            "bullets": self._bullets[:n],
            "danger": self._danger[:n],
            "action": self._action[:n],
            "meta": self._meta[:n],
        }
        if self._has_value:
            payload["value"] = self._value[:n]
        np.savez_compressed(path, **payload)
        self.shards.append(name)
        (self.out_dir / f"{name}.json").write_text(
            json.dumps(
                {
                    "shard": name,
                    "frames": int(n),
                    "total_frames": int(self.total),
                    "has_value": bool(self._has_value),
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    **self.provenance,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._frames = []
        self._has_value = False
        self._value[:] = 0
        self._n = 0

    def close(self) -> dict:
        self.flush()
        summary = {
            "shards": self.shards,
            "frames": int(self.total),
            "out_dir": str(self.out_dir),
            "elapsed_seconds": round(time.time() - self.started, 1),
            "frame_shape": [vision.VISION_H, vision.VISION_W, vision.VISION_CHANNELS],
            **self.provenance,
        }
        (self.out_dir / "vision-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary


def discover_vision_shards(dataset_dir: Path) -> list[Path]:
    """Every ``*.npz`` in ``dataset_dir`` that carries a ``frame`` array."""
    shards: list[Path] = []
    for path in sorted(Path(dataset_dir).glob("*.npz")):
        try:
            with np.load(path) as data:
                if "frame" in data.files and "action" in data.files:
                    shards.append(path)
        except Exception:
            continue
    return shards


class VisionDataset:
    """Per-shard cache over a directory of vision shards.

    Unlike :class:`touhou_rl.dataset.ReplayDataset` this does **not** load
    everything: at 43 KB/frame a 100k-frame corpus is 4.3 GB, so caching is
    bounded to one shard at a time.  Shuffling is done by index, which is enough
    for SGD as long as the shard order is randomised too.
    """

    def __init__(
        self,
        dataset_dir: Path,
        max_cached_shards: int | None = None,
        cache_budget_bytes: int = 2_000_000_000,
    ):
        self.paths = discover_vision_shards(dataset_dir)
        if not self.paths:
            raise FileNotFoundError(f"no vision shards under {dataset_dir}")
        # (path, first global index, size) so index -> shard resolution never
        # reopens a file; doing that inside a per-sample loop was measurably
        # slower than the training step itself.
        self.offsets: list[tuple[Path, int, int]] = []
        total = 0
        self.has_value = False
        for path in self.paths:
            with np.load(path) as data:
                n = int(data["action"].shape[0])
                self.has_value = self.has_value or ("value" in data.files)
            self.offsets.append((path, total, n))
            total += n
        self.total = total
        self._cache: dict[Path, dict[str, np.ndarray]] = {}
        self._order: list[Path] = []
        if max_cached_shards is None:
            # Shuffled access over more shards than the cache holds degenerates
            # into re-decompressing an npz per batch (measured 860 ms/batch versus
            # 3 ms when everything fits).  Cache the whole set when it fits in the
            # budget, otherwise keep a small window and accept the thrash.
            per_frame = vision.VISION_H * vision.VISION_W * vision.VISION_CHANNELS + 256
            max_cached_shards = (
                len(self.paths) if total * per_frame <= cache_budget_bytes else 4
            )
        self._max_cached = max(1, int(max_cached_shards))


    def __len__(self) -> int:
        return self.total

    def _shard_for(self, index: int) -> tuple[Path, int]:
        path, start, _size = self._locate(index)
        return path, index - start

    def _locate(self, index: int) -> tuple[Path, int, int]:
        for path, start, size in self.offsets:
            if index < start + size:
                return path, start, size
        raise IndexError(index)

    def stack_indices(self, indices: list[int], depth: int) -> np.ndarray:
        """Global indices of the ``depth``-frame history behind each index.

        Clamped **inside the sample's own shard** rather than globally: shards are
        written by different capture runs, so the frame before a shard's first
        sample belongs to a different episode (or a different stage) and would
        teach the model a fake motion cue.
        """
        depth = max(1, int(depth))
        out = np.empty((len(indices), depth), dtype=np.int64)
        for row, index in enumerate(indices):
            _path, start, size = self._locate(index)
            for slot in range(depth):
                candidate = index - (depth - 1 - slot)
                out[row, slot] = min(max(candidate, start), start + size - 1)
        return out

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        if path not in self._cache:
            with np.load(path) as data:
                self._cache[path] = {key: data[key] for key in data.files}
            self._order.append(path)
            while len(self._order) > self._max_cached:
                stale = self._order.pop(0)
                self._cache.pop(stale, None)
        return self._cache[path]

    def batch(self, indices: list[int]) -> dict[str, np.ndarray]:
        """Gather arbitrary indices (grouped per shard to keep reads contiguous)."""
        by_shard: dict[Path, list[tuple[int, int]]] = {}
        sizes: dict[Path, int] = {}
        for position, index in enumerate(indices):
            path, local = self._shard_for(index)
            by_shard.setdefault(path, []).append((position, local))
            sizes[path] = sizes.get(path, 0) + 1
        first = next(iter(by_shard))
        keys = list(self._load(first).keys())
        out: dict[str, np.ndarray] = {}
        for key in keys:
            sample = self._load(first)[key]
            shape = sample.shape[1:]
            dtype = sample.dtype
            out[key] = np.zeros((len(indices),) + shape, dtype=dtype)
        for path, entries in by_shard.items():
            shard = self._load(path)
            locals_ = [local for _, local in entries]
            positions = [position for position, _ in entries]
            for key in keys:
                out[key][positions] = shard[key][locals_]
        return out

    def label_stats(self) -> dict:
        """Cheap dataset description for reports (no frames loaded)."""
        stages: set[int] = set()
        frames = 0
        for path, _start, _size in self.offsets:
            with np.load(path) as data:
                meta = data["meta"]
            frames += meta.shape[0]
            stages.update(int(value) for value in np.unique(meta[:, 1]))
        return {
            "frames": frames,
            "shards": len(self.paths),
            "stages": sorted(stages),
            "has_teacher_values": self.has_value,
            "frame_shape": [vision.VISION_H, vision.VISION_W, vision.VISION_CHANNELS],
        }
