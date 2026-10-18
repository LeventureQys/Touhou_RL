"""On-disk format for vision shards: **per-region** native PNG frames + labels.

Why regions instead of one frame
--------------------------------
A single 640x480 picture forces one bad trade: downscale it and the HUD digits
(lives, bombs, score, power) become unreadable; keep it native and the playfield
costs 900 KB/frame.  The HUD is not decoration - "bomb now or die" is exactly the
kind of decision that needs the bomb count - so each region is recorded on its own
at **native resolution**:

* ``playfield``  384x448  - bullets, player, enemies
* ``hud_score``  216x34   - HiScore + Score digits
* ``hud_status`` 216x126  - Player/Spell stars (lives/bombs), Power, Graze,
                            Point, Time

Layout per shard
----------------
========================  ======================  ==========================================
key                       shape/dtype             meaning
========================  ======================  ==========================================
``<region>_blob``         u8  [total_bytes]       concatenated PNGs of that region
``<region>_offset``       i64 [n + 1]             sample i is blob[off[i]:off[i+1]]
``region_name``           U24 [R]                 region order
``region_hw``             i32 [R, 2]              native (H, W) of each region
``self_feat``             f32 [n, 24]             privileged observation (label)
``bullets``               f16 [n, 64, 6]          privileged label
``danger``                u8  [n, 16, 8]          privileged label
``action``                u8  [n, 4]              expert action (human replay)
``value``                 f16 [n, 18]             teacher value (optional)
``meta``                  f32 [n, 8]              frame, stage, score, graze,
                                                   lives, bombs, power, deaths
========================  ======================  ==========================================

Shards written by the earlier single-frame capture path carry a plain ``frame``
array; :class:`VisionDataset` still reads those as a single ``playfield`` region.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import obs, vision

META_FIELDS = ("frame", "stage", "score", "graze", "lives", "bombs", "power", "deaths")


def _png_size(blob: bytes) -> tuple[int, int]:
    """(H, W) from a PNG header - 8-byte signature, then IHDR at offset 8.

    Parsed by hand rather than via PIL: capture calls this once per region per
    frame (~90 times/second) and no image decode is needed to read a header.
    """
    if len(blob) < 24 or blob[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG payload")
    width, height = struct.unpack(">II", blob[16:24])
    return int(height), int(width)


@dataclass
class VisionShardWriter:
    """Accumulates encoded region frames and writes one ``.npz`` per shard."""

    out_dir: Path
    shard_size: int = 64  # 64 samples x ~300 KB of playfield PNG ~= 20 MB per shard
    tag: str = "vision"
    provenance: dict = field(default_factory=dict)
    allow_append: bool = False
    regions: tuple[vision.RegionSpec, ...] = vision.TH08_REGIONS

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.out_dir.glob(f"{self.tag}-*.npz"))
        if existing and not self.allow_append:
            raise FileExistsError(
                f"{self.out_dir} already contains {len(existing)} '{self.tag}' shards; "
                "pass allow_append=True (or --overwrite) to append"
            )
        self._blobs: dict[str, list[bytes]] = {region.name: [] for region in self.regions}
        self._hw: dict[str, tuple[int, int]] = {
            region.name: (region.box.h, region.box.w) for region in self.regions
        }
        self._self = np.zeros((self.shard_size, obs.N_SELF), dtype=np.float32)
        self._bullets = np.zeros((self.shard_size, obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float16)
        self._danger = np.zeros((self.shard_size, obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8)
        self._action = np.zeros((self.shard_size, 4), dtype=np.uint8)
        self._value = np.zeros((self.shard_size, 18), dtype=np.float16)
        self._meta = np.zeros((self.shard_size, len(META_FIELDS)), dtype=np.float32)
        self._has_value = False
        self._n = 0
        self.total = 0
        self.bytes = 0
        self.shards: list[str] = []
        self._index = len(existing) if (existing and self.allow_append) else 0
        self.started = time.time()

    def add(
        self,
        regions: dict[str, bytes],
        observation: obs.Observation,
        action: tuple[int, int, int, int],
        meta: dict,
        value: np.ndarray | None = None,
        native_hw: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        if set(regions) != {region.name for region in self.regions}:
            raise ValueError(f"region keys {sorted(regions)} != {sorted(r.name for r in self.regions)}")
        index = self._n
        for name, blob in regions.items():
            if not isinstance(blob, (bytes, bytearray)):
                raise TypeError(f"region {name}: expected PNG bytes, got {type(blob)!r}")
            self._blobs[name].append(bytes(blob))
            self.bytes += len(blob)
        if native_hw:
            self._hw.update({name: (int(hw[0]), int(hw[1])) for name, hw in native_hw.items()})
        else:
            # Record what the blob actually contains: a declared-vs-actual mismatch
            # would silently resize every training sample and nobody would notice.
            for name, blob in regions.items():
                self._hw[name] = _png_size(blob)
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
            "region_name": np.asarray([region.name for region in self.regions], dtype="U24"),
            "region_hw": np.asarray([self._hw[region.name] for region in self.regions], dtype=np.int32),
            "self_feat": self._self[:n],
            "bullets": self._bullets[:n],
            "danger": self._danger[:n],
            "action": self._action[:n],
            "meta": self._meta[:n],
        }
        for region in self.regions:
            blobs = self._blobs[region.name]
            offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
            for position, blob in enumerate(blobs):
                offsets[position + 1] = offsets[position] + len(blob)
            payload[f"{region.name}_blob"] = np.frombuffer(b"".join(blobs), dtype=np.uint8).copy()
            payload[f"{region.name}_offset"] = offsets
        if self._has_value:
            payload["value"] = self._value[:n]
        # Plain ``savez``, not ``savez_compressed``: the payload is mostly PNG
        # blobs, which are already deflated, so compressing again costs ~4.5 ms per
        # sample (measured) and shrinks the file by well under 1%.
        np.savez(path, **payload)
        self.shards.append(name)
        (self.out_dir / f"{name}.json").write_text(
            json.dumps(
                {
                    "shard": name,
                    "frames": int(n),
                    "total_frames": int(self.total),
                    "regions": {region.name: list(self._hw[region.name]) for region in self.regions},
                    "bytes": int(path.stat().st_size),
                    "has_value": bool(self._has_value),
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    **self.provenance,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._blobs = {region.name: [] for region in self.regions}
        self._has_value = False
        self._value[:] = 0
        self._n = 0

    def close(self) -> dict:
        self.flush()
        summary = {
            "shards": self.shards,
            "frames": int(self.total),
            "out_dir": str(self.out_dir),
            "regions": {region.name: list(self._hw[region.name]) for region in self.regions},
            "frame_bytes": int(self.bytes),
            "elapsed_seconds": round(time.time() - self.started, 1),
            **self.provenance,
        }
        (self.out_dir / "vision-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary


def discover_vision_shards(dataset_dir: Path) -> list[Path]:
    """Every ``*.npz`` in ``dataset_dir`` that carries frames (region or legacy)."""
    shards: list[Path] = []
    for path in sorted(Path(dataset_dir).glob("*.npz")):
        try:
            with np.load(path) as data:
                files = set(data.files)
                has_frames = any(key.endswith("_blob") for key in files) or "frame" in files
                if has_frames and "action" in files:
                    shards.append(path)
        except Exception:
            continue
    return shards


class VisionDataset:
    """Per-shard cache over a directory of vision shards.

    Frames are decoded on demand per region and at a caller-chosen resolution, so
    the archive can stay native while training runs at whatever tensor size the
    model wants.
    """

    def __init__(
        self,
        dataset_dir: Path,
        max_cached_shards: int | None = None,
        cache_budget_bytes: int = 2_000_000_000,
        decode_threads: int = 8,
    ):
        self.paths = discover_vision_shards(dataset_dir)
        if not self.paths:
            raise FileNotFoundError(f"no vision shards under {dataset_dir}")
        self.offsets: list[tuple[Path, int, int]] = []
        total = 0
        self.has_value = False
        self.region_hw: dict[str, tuple[int, int]] = {}
        self.legacy_tensor = False
        for path in self.paths:
            with np.load(path) as data:
                n = int(data["action"].shape[0])
                self.has_value = self.has_value or ("value" in data.files)
                if "region_name" in data.files:
                    names = [str(value) for value in data["region_name"]]
                    shapes = data["region_hw"]
                    for name, (height, width) in zip(names, shapes):
                        self.region_hw.setdefault(name, (int(height), int(width)))
                elif "frame" in data.files:
                    self.region_hw.setdefault("playfield", tuple(int(v) for v in data["frame"].shape[1:3]))
                    self.legacy_tensor = True
            self.offsets.append((path, total, n))
            total += n
        self.total = total
        self._cache: dict[Path, dict[str, np.ndarray]] = {}
        self._order: list[Path] = []
        self.decode_threads = max(1, int(decode_threads))
        self._pool = None
        if max_cached_shards is None:
            per_frame = 1200
            max_cached_shards = len(self.paths) if total * per_frame <= cache_budget_bytes else 4
        self._max_cached = max(1, int(max_cached_shards))

    def __len__(self) -> int:
        return self.total

    @property
    def region_names(self) -> list[str]:
        return list(self.region_hw)

    def _locate(self, index: int) -> tuple[Path, int, int]:
        for path, start, size in self.offsets:
            if index < start + size:
                return path, start, size
        raise IndexError(index)

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        if path not in self._cache:
            with np.load(path) as data:
                self._cache[path] = {key: data[key] for key in data.files}
            self._order.append(path)
            while len(self._order) > self._max_cached:
                stale = self._order.pop(0)
                self._cache.pop(stale, None)
        return self._cache[path]

    def _group(self, indices: list[int]) -> dict[Path, list[tuple[int, int]]]:
        by_shard: dict[Path, list[tuple[int, int]]] = {}
        for position, index in enumerate(indices):
            path, _start, _size = self._locate(index)
            by_shard.setdefault(path, []).append((position, index - _start))
        return by_shard

    # -- labels ----------------------------------------------------------
    def labels(self, indices: list[int]) -> dict[str, np.ndarray]:
        """Privileged arrays for ``indices`` (never touches pixels)."""
        by_shard = self._group(indices)
        keys = ["self_feat", "bullets", "danger", "action", "meta"]
        first = next(iter(by_shard))
        sample = self._load(first)
        out = {
            key: np.zeros((len(indices),) + sample[key].shape[1:], dtype=sample[key].dtype)
            for key in keys
        }
        if "value" in sample:
            out["value"] = np.zeros((len(indices),) + sample["value"].shape[1:], dtype=sample["value"].dtype)
        for path, entries in by_shard.items():
            shard = self._load(path)
            positions = [position for position, _ in entries]
            locals_ = [local for _, local in entries]
            for key in list(out):
                if key in shard:
                    out[key][positions] = shard[key][locals_]
        return out

    # -- frames ----------------------------------------------------------
    def frames(
        self,
        indices: list[int],
        region: str = "playfield",
        out_h: int | None = None,
        out_w: int | None = None,
    ) -> np.ndarray:
        """Decode one region of ``indices`` to ``uint8[n, out_h, out_w, 3]``.

        Decoding runs on a thread pool: PNG inflation releases the GIL, and a single
        thread costs ~6 ms per sample, which made one epoch over 72k frames take 7
        minutes.  With ``decode_threads`` workers the data loader stops being the
        bottleneck of training.
        """
        native = self.region_hw.get(region)
        if native is None:
            raise KeyError(f"region {region!r} not present; dataset has {sorted(self.region_hw)}")
        out_h = out_h or native[0]
        out_w = out_w or native[1]
        blob_key, off_key = f"{region}_blob", f"{region}_offset"
        out = np.zeros((len(indices), out_h, out_w, vision.VISION_CHANNELS), dtype=np.uint8)

        jobs: list[tuple[int, bytes]] = []
        for path, entries in self._group(indices).items():
            shard = self._load(path)
            if blob_key in shard:
                blob = shard[blob_key]
                offsets = shard[off_key]
                for position, local in entries:
                    start, stop = int(offsets[local]), int(offsets[local + 1])
                    jobs.append((position, blob[start:stop].tobytes()))
            elif "frame" in shard:  # legacy single-frame shard: only "playfield" exists
                tiles = np.asarray(shard["frame"])[[local for _, local in entries]]
                for offset, (position, _local) in enumerate(entries):
                    out[position] = vision.resize_area(tiles[offset], out_h, out_w)
            else:
                raise KeyError(f"{path.name}: no payload for region {region!r}")

        if jobs:
            def work(job: tuple[int, bytes]) -> tuple[int, np.ndarray]:
                position, payload = job
                return position, vision.decode_png(payload, out_h, out_w)

            if self._pool is None and len(jobs) > 1:
                from concurrent.futures import ThreadPoolExecutor

                self._pool = ThreadPoolExecutor(max_workers=self.decode_threads)
            if self._pool is not None and len(jobs) > 1:
                for position, decoded in self._pool.map(work, jobs):
                    out[position] = decoded
            else:
                for job in jobs:
                    position, decoded = work(job)
                    out[position] = decoded
        return out

    def region_frames(
        self, indices: list[int], sizes: dict[str, tuple[int, int]]
    ) -> dict[str, np.ndarray]:
        """Decode several regions at once; ``sizes`` maps region -> (H, W)."""
        return {
            region: self.frames(indices, region, height, width)
            for region, (height, width) in sizes.items()
        }

    def stack_indices(self, indices: list[int], depth: int) -> np.ndarray:
        """Global indices of the ``depth``-frame history behind each index.

        Clamped **inside the sample's own shard**: shards come from different
        capture runs, so the frame before a shard's first sample belongs to a
        different episode (or a different stage) and would teach fake motion.
        """
        depth = max(1, int(depth))
        out = np.empty((len(indices), depth), dtype=np.int64)
        for row, index in enumerate(indices):
            _path, start, size = self._locate(index)
            for slot in range(depth):
                candidate = index - (depth - 1 - slot)
                out[row, slot] = min(max(candidate, start), start + size - 1)
        return out

    def label_stats(self) -> dict:
        """Cheap dataset description for reports (no pixels touched)."""
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
            "regions": {name: list(hw) for name, hw in self.region_hw.items()},
        }
