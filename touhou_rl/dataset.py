"""On-disk sample format for imitation / offline RL.

Each shard is one ``.npz`` holding aligned arrays for a contiguous run of game
frames, plus a JSON sidecar with provenance.  Storing the *encoded observation*
instead of raw memory snapshots keeps a full 6-stage Lunatic run around 90 MB
while remaining directly loadable by the trainer.

Layout per shard:

======================  ================  ==================================
key                     shape/dtype       meaning
======================  ================  ==================================
``self_feat``           f32 [n, 24]       observation contract §3
``bullets``             f16 [n, 64, 6]    observation contract §3
``danger``              u8  [n, 16, 8]    coarse occupancy
``action``              u8  [n, 4]        (move, shoot, focus, bomb) label
``meta``                f32 [n, 8]        frame, stage, score, graze, lives,
                                          bombs, power, deaths
======================  ================  ==================================

``meta.dtype == float32`` is deliberate: it keeps shards to a single numeric
kind per file while still allowing exact integer round-trips up to 2^24 frames.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import obs
from .obs import Observation

SHARD_KEYS = ("self_feat", "bullets", "danger", "action", "meta")
META_FIELDS = ("frame", "stage", "score", "graze", "lives", "bombs", "power", "deaths")


@dataclass
class ShardWriter:
    """Accumulates frames in memory and writes compressed ``.npz`` shards."""

    out_dir: Path
    shard_size: int = 2048
    tag: str = "replay"
    provenance: dict = field(default_factory=dict)
    allow_append: bool = False

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.out_dir.glob(f"{self.tag}-*.npz"))
        if existing and not self.allow_append:
            raise FileExistsError(
                f"{self.out_dir} already contains {len(existing)} '{self.tag}' shards; "
                "capturing into it would mix runs. Remove it or pass allow_append=True."
            )
        self._self = np.zeros((self.shard_size, obs.N_SELF), dtype=np.float32)
        self._bullets = np.zeros(
            (self.shard_size, obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float16
        )
        self._danger = np.zeros((self.shard_size, obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8)
        self._action = np.zeros((self.shard_size, 4), dtype=np.uint8)
        self._meta = np.zeros((self.shard_size, len(META_FIELDS)), dtype=np.float32)
        self._n = 0
        self.total = 0
        self.shards: list[str] = []
        self._index = 0
        if existing and self.allow_append:
            self._index = len(existing)
        self.started = time.time()

    def add(self, observation: Observation, action: tuple[int, int, int, int], meta: dict) -> None:
        i = self._n
        self._self[i] = observation.self_feat
        self._bullets[i] = observation.bullets
        self._danger[i] = observation.danger_map
        self._action[i] = np.asarray(action, dtype=np.uint8)
        self._meta[i] = [float(meta[name]) for name in META_FIELDS]
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
        np.savez_compressed(
            path,
            self_feat=self._self[:n],
            bullets=self._bullets[:n],
            danger=self._danger[:n],
            action=self._action[:n],
            meta=self._meta[:n],
        )
        self.shards.append(name)
        sidecar = {
            "shard": name,
            "frames": int(n),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_frames": int(self.total),
            **self.provenance,
        }
        (self.out_dir / f"{name}.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._n = 0

    def close(self) -> dict:
        self.flush()
        summary = {
            "shards": self.shards,
            "frames": int(self.total),
            "out_dir": str(self.out_dir),
            "elapsed_seconds": round(time.time() - self.started, 1),
            **self.provenance,
        }
        (self.out_dir / "dataset-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary


def discover_shards(dataset_dir: Path) -> list[Path]:
    """Every ``*.npz`` in ``dataset_dir`` that is actually an observation shard.

    The ``action`` key check matters: a stray array file in the directory (for
    example the ``score-labels.npz`` produced by ``touhou_rl.distill``) used to be
    picked up as a shard and fail later with an opaque ``KeyError``.
    """
    shards: list[Path] = []
    for path in sorted(Path(dataset_dir).glob("*.npz")):
        try:
            with np.load(path) as data:
                if "action" in data.files:
                    shards.append(path)
        except Exception:  # unreadable file: not a shard
            continue
    return shards


def load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in SHARD_KEYS}


def count_frames(dataset_dir: Path) -> int:
    total = 0
    for path in discover_shards(dataset_dir):
        with np.load(path) as data:
            total += int(data["action"].shape[0])
    return total


class ReplayDataset:
    """Torch-style dataset over a directory of shards (dependency-free import).

    ``cache_in_memory=True`` loads and concatenates every shard once.  A full
    Lunatic capture is only ~145 MB of arrays, and decompressing shards per batch
    dominated training time (measured: 78 s/epoch vs ~1 s cached), so caching is
    the default.
    """

    def __init__(self, dataset_dir: Path, mmap: bool = True, cache_in_memory: bool = True):
        import torch  # local import: dataset listing must work without torch

        self.paths = discover_shards(dataset_dir)
        if not self.paths:
            raise FileNotFoundError(f"no shards under {dataset_dir}")
        self.offsets: list[tuple[Path, int, int]] = []
        total = 0
        for path in self.paths:
            with np.load(path) as data:
                n = int(data["action"].shape[0])
            self.offsets.append((path, 0, n))
            total += n
        self.total = total
        self._torch = torch
        self._cache_path: Path | None = None
        self._cache: dict[str, np.ndarray] | None = None
        self._memory: dict[str, np.ndarray] | None = None
        if cache_in_memory:
            self._memory = self._load_all()

    def _load_all(self) -> dict[str, np.ndarray]:
        parts: dict[str, list[np.ndarray]] = {key: [] for key in SHARD_KEYS}
        for path in self.paths:
            shard = load_shard(path)
            for key in SHARD_KEYS:
                parts[key].append(shard[key])
        return {key: np.concatenate(parts[key], axis=0) for key in SHARD_KEYS}

    def __len__(self) -> int:
        return self.total

    def _shard_for(self, index: int) -> tuple[Path, int]:
        start = 0
        for path, _off, n in self.offsets:
            if index < start + n:
                return path, index - start
            start += n
        raise IndexError(index)

    def _get_shard(self, path: Path) -> dict[str, np.ndarray]:
        if self._memory is not None:
            return self._memory
        if self._cache_path != path:
            self._cache = load_shard(path)
            self._cache_path = path
        assert self._cache is not None
        return self._cache

    def _rows(self, key: str, indices: list[int]) -> np.ndarray:
        if self._memory is not None:
            return self._memory[key][indices]
        by_shard: dict[Path, list[tuple[int, int]]] = {}
        for position, index in enumerate(indices):
            path, local = self._shard_for(index)
            by_shard.setdefault(path, []).append((position, local))
        out = np.zeros((len(indices),) + self._memory_shape(key), dtype=self._memory_dtype(key))
        for path, entries in by_shard.items():
            shard = self._get_shard(path)
            positions = [p for p, _ in entries]
            locals_ = [l for _, l in entries]
            out[positions] = shard[key][locals_]
        return out

    def _memory_shape(self, key: str) -> tuple[int, ...]:
        return {
            "self_feat": (obs.N_SELF,),
            "bullets": (obs.N_BULLETS, obs.N_BULLET_FEATURES),
            "danger": (obs.DANGER_ROWS, obs.DANGER_COLS),
            "action": (4,),
            "meta": (len(META_FIELDS),),
        }[key]

    def _memory_dtype(self, key: str) -> np.dtype:
        return {
            "self_feat": np.float32,
            "bullets": np.float16,
            "danger": np.uint8,
            "action": np.uint8,
            "meta": np.float32,
        }[key]

    def __getitem__(self, index: int):
        torch = self._torch
        rows = {key: self._rows(key, [index])[0] for key in SHARD_KEYS}
        self_feat = rows["self_feat"].astype(np.float32)
        self_feat[2:4] = 0.0  # see stacked_batch(): velocity leaks the label
        return {
            "self_feat": torch.from_numpy(self_feat),
            "bullets": torch.from_numpy(rows["bullets"].astype(np.float32)),
            "danger": torch.from_numpy(rows["danger"].astype(np.float32) / 8.0),
            "action": torch.from_numpy(rows["action"].astype(np.int64)),
            "meta": torch.from_numpy(rows["meta"]),
        }


def stacked_batch(dataset: ReplayDataset, indices: list[int], masked_self_features: tuple[int, ...] = (2, 3)):
    """Fetch a batch with one shard read (or one slice when memory-cached).

    ``masked_self_features`` zeroes observation dimensions that would otherwise
    leak the label.  ``self_feat[2:4]`` is the player velocity, which the game
    sets *from the current movement input*: feeding it makes behaviour cloning
    look near-perfect (val move accuracy 0.9998) while teaching "keep the
    velocity you already have" instead of "dodge".  The raw values stay in the
    shards so the decision can be revisited without re-capturing.
    """
    torch = dataset._torch
    self_feat = dataset._rows("self_feat", indices).astype(np.float32)
    for column in masked_self_features:
        self_feat[:, column] = 0.0
    batch = {
        "self_feat": torch.from_numpy(self_feat),
        "bullets": torch.from_numpy(dataset._rows("bullets", indices).astype(np.float32)),
        "danger": torch.from_numpy(dataset._rows("danger", indices).astype(np.float32) / 8.0),
        "action": torch.from_numpy(dataset._rows("action", indices).astype(np.int64)),
        "meta": torch.from_numpy(dataset._rows("meta", indices).astype(np.float32)),
    }
    return batch
