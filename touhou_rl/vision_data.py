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

import hashlib
import json
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import obs, vision, vision_history

META_FIELDS = ("frame", "stage", "score", "graze", "lives", "bombs", "power", "deaths")

#: v1.0.4 residency budget for the full-shard and label caches (512 MiB).  The
#: v1.0.3 default of 2 GB was paired with a ``per_frame=1200`` guess while the
#: real shard cost is ~253 KB/frame, so a "budgeted" loader could pin ~20 GB.
DEFAULT_CACHE_BUDGET_BYTES = 536_870_912

#: The three regions the v1.0.4 experiments consume, at the fixed input sizes of
#: contract C1 (playfield 224x192, hud_score 34x216, hud_status 126x216).
ROI_SIZES: dict[str, tuple[int, int]] = {
    "playfield": (224, 192),
    "hud_score": (34, 216),
    "hud_status": (126, 216),
}

#: Identity contract of a decoded-frame cache file and its ``<cache>.json``
#: sidecar.  A cache file without a matching sidecar is *not* trusted: it falls
#: back to PNG decoding and is only repaired by an explicit build.
DECODE_CACHE_SCHEMA = "v104-decode-cache/1"


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


def decoded_cache_path(shard: Path, region: str, height: int, width: int) -> Path:
    """Sibling file holding one region's frames already decoded at ``height x width``.

    Decoding a PNG costs ~4.4 ms of CPU per sample; a raw ``uint8`` frame at the
    model resolution is 233 KB versus 253 KB for the PNG, so storing the decoded
    frames costs ~8% more disk and turns the per-batch load into a memcpy.  On a
    96-sample batch that takes the load from 148 ms to ~15 ms (measured), which is
    what lets the GPU run at a useful duty cycle instead of waiting on inflate.
    """
    return shard.with_suffix(f".decoded-{region}-{height}x{width}.npy")


def decoded_sidecar_path(cache: Path) -> Path:
    """``<cache>.json`` - the identity file that must sit next to a cache file."""
    cache = Path(cache)
    return cache.with_name(cache.name + ".json")


def _sidecar_candidates(cache: Path) -> list[Path]:
    """Accepted sidecar names: ``<cache>.json`` first, ``<stem>.json`` as a legacy read."""
    cache = Path(cache)
    return [decoded_sidecar_path(cache), cache.with_suffix(".json")]


def _file_digest(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_decode_sidecar(cache: Path) -> dict | None:
    """Parse a decoded-cache sidecar; ``None`` when absent or unreadable."""
    for path in _sidecar_candidates(cache):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass
class DecodeCacheCheck:
    """Outcome of the full identity check of one decoded-cache file."""

    ok: bool
    reason: str
    frames: int = 0
    bytes: int = 0
    array: np.ndarray | None = None


def check_decode_cache(
    shard: Path,
    region: str,
    height: int,
    width: int,
    *,
    frames: int | None = None,
    verify_source: bool = True,
    source_digest: str | None = None,
    verify_payload: bool = False,
) -> DecodeCacheCheck:
    """Validate one decoded cache: n/H/W/C/dtype, sidecar identity and source digest.

    ``frames`` is the shard's frame count (``None`` skips the n checks).  The
    source digest is the expensive part (one SHA-256 over the ``.npz``), so the
    caller may pass a digest it already computed for this shard.
    """
    shard = Path(shard)
    cache = decoded_cache_path(shard, region, height, width)
    if not cache.exists():
        return DecodeCacheCheck(False, "missing")
    size = int(cache.stat().st_size)
    try:
        array = np.load(cache, mmap_mode="r")
    except Exception:
        return DecodeCacheCheck(False, "unreadable", bytes=size)
    if array.ndim != 4:
        return DecodeCacheCheck(False, f"ndim={array.ndim}", bytes=size)
    if array.dtype != np.dtype(np.uint8):
        return DecodeCacheCheck(False, f"dtype={array.dtype}", bytes=size)
    if int(array.shape[3]) != int(vision.VISION_CHANNELS):
        return DecodeCacheCheck(False, f"channels={int(array.shape[3])}", bytes=size)
    if tuple(int(v) for v in array.shape[1:3]) != (int(height), int(width)):
        return DecodeCacheCheck(False, f"hw={tuple(array.shape[1:3])}", bytes=size)
    if frames is not None and int(array.shape[0]) != int(frames):
        return DecodeCacheCheck(False, f"frames={int(array.shape[0])}!={int(frames)}", bytes=size)
    sidecar = read_decode_sidecar(cache)
    if sidecar is None:
        # An old file without identity: readable, but nothing proves which shard
        # or resolution produced it, so it must not be trusted.
        return DecodeCacheCheck(False, "no-identity", bytes=size)
    if str(sidecar.get("schema", "")) != DECODE_CACHE_SCHEMA:
        return DecodeCacheCheck(False, f"schema={sidecar.get('schema')!r}", bytes=size)
    if str(sidecar.get("region", "")) != str(region):
        return DecodeCacheCheck(False, f"sidecar-region={sidecar.get('region')!r}", bytes=size)
    try:
        recorded_shape = [int(value) for value in sidecar.get("shape", [])]
    except (TypeError, ValueError):
        return DecodeCacheCheck(False, "sidecar-shape", bytes=size)
    if recorded_shape != [int(value) for value in array.shape]:
        return DecodeCacheCheck(False, "sidecar-shape", bytes=size)
    if str(sidecar.get("dtype", "")) != "uint8":
        return DecodeCacheCheck(False, f"sidecar-dtype={sidecar.get('dtype')!r}", bytes=size)
    source = sidecar.get("source") or {}
    if not isinstance(source, dict):
        return DecodeCacheCheck(False, "sidecar-source", bytes=size)
    if str(source.get("name", "")) != shard.name:
        return DecodeCacheCheck(False, f"source-name={source.get('name')!r}", bytes=size)
    if frames is not None and int(source.get("frames", -1)) != int(frames):
        return DecodeCacheCheck(False, "source-frames", bytes=size)
    try:
        source_bytes = int(source.get("bytes", -1))
    except (TypeError, ValueError):
        return DecodeCacheCheck(False, "source-bytes", bytes=size)
    if source_bytes != int(shard.stat().st_size):
        return DecodeCacheCheck(False, "source-bytes", bytes=size)
    if verify_source:
        digest = source_digest if source_digest is not None else _file_digest(shard)
        if str(source.get("sha256", "")).lower() != str(digest).lower():
            return DecodeCacheCheck(False, "source-digest", bytes=size)
    if verify_payload:
        if str(sidecar.get("cache_sha256", "")).lower() != _file_digest(cache):
            return DecodeCacheCheck(False, "cache-digest", bytes=size)
    return DecodeCacheCheck(True, "ok", frames=int(array.shape[0]), bytes=size, array=array)


def _write_decode_cache(
    frames: np.ndarray,
    target: Path,
    shard: Path,
    region: str,
    *,
    source_digest: str,
    frames_total: int,
    verify_payload: bool = True,
) -> int:
    """Write one ``.npy`` cache plus its sidecar, atomically; return its size.

    The temporary file is validated (shape/dtype/size and, with
    ``verify_payload``, read back through mmap and compared value by value) before
    it replaces the target, so an interrupted build can never leave a truncated
    cache in place of a good one.
    """
    target = Path(target)
    array = np.ascontiguousarray(frames, dtype=np.uint8)
    if array.ndim != 4 or array.dtype != np.uint8:
        raise vision_history.v104_error("CACHE", f"refusing to cache {array.shape} {array.dtype}")
    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    sidecar = decoded_sidecar_path(target)
    temporary_sidecar = sidecar.with_name(f"{sidecar.name}.tmp-{os.getpid()}")
    try:
        with open(temporary, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
        written = np.load(temporary, mmap_mode="r")
        if [int(v) for v in written.shape] != [int(v) for v in array.shape]:
            raise vision_history.v104_error("CACHE", f"{target.name}: wrote {written.shape}")
        if written.dtype != np.dtype(np.uint8):
            raise vision_history.v104_error("CACHE", f"{target.name}: wrote {written.dtype}")
        if verify_payload and not np.array_equal(np.asarray(written), array):
            raise vision_history.v104_error("CACHE", f"{target.name}: payload differs after write")
        del written
        cache_digest = _file_digest(temporary)
        payload = {
            "schema": DECODE_CACHE_SCHEMA,
            "source": {
                "name": shard.name,
                "frames": int(frames_total),
                "bytes": int(shard.stat().st_size),
                "sha256": str(source_digest),
            },
            "region": str(region),
            "shape": [int(v) for v in array.shape],
            "dtype": "uint8",
            "cache_sha256": cache_digest,
            "payload_verified": bool(verify_payload),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        temporary_sidecar.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        size = int(temporary.stat().st_size)
        os.replace(temporary, target)
        os.replace(temporary_sidecar, sidecar)
        return size
    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        raise


def build_decode_cache(
    dataset_dir: Path,
    sizes: dict[str, tuple[int, int]],
    *,
    verbose: bool = True,
    verify_payload: bool = True,
    decode_threads: int = 8,
    limit_shards: int | None = None,
) -> dict:
    """Materialise the decoded-frame cache for ``sizes``. Idempotent and resumable.

    A cache file is only reused when its full identity checks out (n/H/W/C/dtype
    and the sidecar's shard name, frame count, byte size and SHA-256).  Anything
    else - including the v1.0.3 files that carry no sidecar at all - is counted in
    ``files_invalid`` and rewritten from the PNG source, which is the only way to
    earn the identity without asserting something unverified.

    Old report keys are kept: ``dataset``, ``sizes``, ``files_written``,
    ``files_skipped``, ``bytes_written``, ``elapsed_seconds``.  ``bytes_written``
    is what this run wrote; ``bytes_total`` is the ``stat`` size of every target
    file on disk afterwards.
    """
    dataset_dir = Path(dataset_dir)
    # A small residency budget keeps one shard's blobs in memory across its three
    # regions without letting the build pin the dataset.
    dataset = VisionDataset(
        dataset_dir,
        max_cached_shards=2,
        cache_budget_bytes=256 * 1024 * 1024,
        decode_threads=decode_threads,
        use_decode_cache=False,
    )
    started = time.time()
    written = skipped = invalid = repaired = missing = 0
    bytes_written = 0
    files_expected = 0
    processed = 0
    seen_shards = 0
    digests: dict[Path, str] = {}

    def source_digest(path: Path) -> str:
        """One SHA-256 per shard, reused by all of its regions (never per batch)."""
        digest = digests.get(path)
        if digest is None:
            digest = _file_digest(path)
            digests[path] = digest
        return digest

    try:
        for path, start, size in dataset.offsets:
            if limit_shards is not None and seen_shards >= int(limit_shards):
                break
            seen_shards += 1
            for region, (height, width) in sizes.items():
                if region not in dataset.region_hw:
                    continue
                files_expected += 1
                target = decoded_cache_path(path, region, height, width)
                existed = target.exists()
                check = check_decode_cache(
                    path,
                    region,
                    height,
                    width,
                    frames=size,
                    source_digest=source_digest(path),
                )
                if check.ok:
                    skipped += 1
                    processed += 1
                    continue
                if existed:
                    invalid += 1
                else:
                    missing += 1
                frames = dataset.frames(list(range(start, start + size)), region, height, width)
                bytes_written += _write_decode_cache(
                    frames,
                    target,
                    path,
                    region,
                    source_digest=source_digest(path),
                    frames_total=size,
                    verify_payload=verify_payload,
                )
                written += 1
                if existed:
                    repaired += 1
                processed += 1
                if verbose and processed % 500 == 0:
                    print(f"[decode-cache] {processed} region files done", flush=True)
        # The authoritative number is the on-disk size, not the sum of what we
        # think we wrote: a skipped file is only proven present by ``stat``.
        bytes_total = 0
        files_present = 0
        for path, _start, size in dataset.offsets[:seen_shards]:
            for region, (height, width) in sizes.items():
                if region not in dataset.region_hw:
                    continue
                target = decoded_cache_path(path, region, height, width)
                if target.exists():
                    bytes_total += int(target.stat().st_size)
                    files_present += 1
    finally:
        dataset.close()
    report = {
        "dataset": str(dataset_dir),
        "sizes": {key: list(value) for key, value in sizes.items()},
        "files_written": written,
        "files_skipped": skipped,
        "files_invalid": invalid,
        "files_repaired": repaired,
        "files_missing": missing,
        "files_expected": files_expected,
        "files_present": files_present,
        "bytes_written": int(bytes_written),
        "bytes_total": int(bytes_total),
        "cache_contract": DECODE_CACHE_SCHEMA,
        "shards_visited": seen_shards,
        "payload_verified": bool(verify_payload),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    if verbose:
        print(f"[decode-cache] {report}", flush=True)
    return report


def verify_decode_cache(
    dataset_dir: Path,
    sizes: dict[str, tuple[int, int]],
    *,
    verify_payload: bool = True,
    verbose: bool = False,
    limit_shards: int | None = None,
) -> dict:
    """Re-check every cache file (and its sidecar) without writing anything."""
    dataset_dir = Path(dataset_dir)
    dataset = VisionDataset(dataset_dir, use_decode_cache=False)
    checked = ok = 0
    bytes_total = 0
    invalid: list[dict] = []
    reasons: dict[str, int] = {}
    seen_shards = 0
    for path, _start, size in dataset.offsets:
        if limit_shards is not None and seen_shards >= int(limit_shards):
            break
        seen_shards += 1
        for region, (height, width) in sizes.items():
            if region not in dataset.region_hw:
                continue
            checked += 1
            check = check_decode_cache(
                path, region, height, width, frames=size, verify_payload=verify_payload
            )
            if check.ok:
                ok += 1
                bytes_total += check.bytes
                continue
            reasons[check.reason] = reasons.get(check.reason, 0) + 1
            if len(invalid) < 200:
                invalid.append(
                    {
                        "file": decoded_cache_path(path, region, height, width).name,
                        "reason": check.reason,
                    }
                )
        if verbose and checked % 500 == 0:
            print(f"[decode-cache] verified {checked} files", flush=True)
    dataset.close()
    return {
        "dataset": str(dataset_dir),
        "sizes": {key: list(value) for key, value in sizes.items()},
        "cache_contract": DECODE_CACHE_SCHEMA,
        "files": checked,
        "files_ok": ok,
        "files_invalid": checked - ok,
        "bytes_total": int(bytes_total),
        "payload_verified": bool(verify_payload),
        "invalid_reasons": reasons,
        "invalid": invalid,
        "shards_visited": seen_shards,
    }


class _Residency:
    """Shared-object residency accounting: every distinct ndarray is counted once.

    A full shard record and its label-only record can hold the *same* array
    objects when the loader reuses the full record (``_load_labels`` does), so a
    naive sum over both caches would double-count.  ``nbytes`` here is the sum over
    live objects; per-kind logical sums are reported separately by
    :meth:`VisionDataset.cache_stats`.
    """

    __slots__ = ("_refs", "nbytes")

    def __init__(self) -> None:
        self._refs: dict[int, list] = {}
        self.nbytes = 0

    def add(self, entries: dict) -> None:
        for value in entries.values():
            if not isinstance(value, np.ndarray):
                continue
            slot = self._refs.get(id(value))
            if slot is None:
                self._refs[id(value)] = [value, 1]
                self.nbytes += int(value.nbytes)
            else:
                slot[1] += 1

    def discard(self, entries: dict) -> None:
        for value in entries.values():
            if not isinstance(value, np.ndarray):
                continue
            slot = self._refs.get(id(value))
            if slot is None:
                continue
            slot[1] -= 1
            if slot[1] <= 0:
                self.nbytes -= int(value.nbytes)
                del self._refs[id(value)]

    def clear(self) -> None:
        self._refs.clear()
        self.nbytes = 0


def _dict_bytes(entries: dict) -> int:
    """Sum ``ndarray.nbytes`` of one record (a full or a label shard record)."""
    total = 0
    for value in entries.values():
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
    return total


def _entries_bytes(table: dict[Path, dict[str, np.ndarray]]) -> int:
    """Logical bytes of one cache kind (may overlap with the other kind)."""
    return sum(_dict_bytes(entries) for entries in table.values())


def _validated_indices(indices: list[int], total: int) -> list[int]:
    """``indices`` as a list of legal dataset indexes, else ``v104:DATA``."""
    try:
        array = np.asarray(indices)
    except Exception as exc:  # ragged input and similar
        raise vision_history.v104_error("DATA", f"indices are not array-like: {exc}") from exc
    if array.size == 0:
        return []
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)) or not np.all(array == np.rint(array)):
            raise vision_history.v104_error("DATA", "indices must be integers")
    elif array.dtype.kind not in "iu":
        raise vision_history.v104_error("DATA", f"indices must be integers, got {array.dtype}")
    values = [int(value) for value in array.reshape(-1)]
    for value in values:
        if value < 0 or value >= int(total):
            raise vision_history.v104_error(
                "DATA", f"index {value} outside dataset range [0, {int(total)})"
            )
    return values


class VisionDataset:
    """Per-shard cache over a directory of vision shards.

    Frames are decoded on demand per region and at a caller-chosen resolution, so
    the archive can stay native while training runs at whatever tensor size the
    model wants.
    """

    #: How many shards' derived segment tables stay in memory (a few hundred
    #: bytes each); this is an index, not cached pixels.
    _HISTORY_INDEX_SHARDS = 32

    def __init__(
        self,
        dataset_dir: Path,
        max_cached_shards: int | None = None,
        cache_budget_bytes: int = DEFAULT_CACHE_BUDGET_BYTES,
        decode_threads: int = 8,
        use_decode_cache: bool = True,
        *,
        history_contract: str = vision_history.HISTORY_LEGACY,
        verify_source_digest: bool = True,
    ):
        budget = int(cache_budget_bytes)
        if budget < 0:
            raise vision_history.v104_error("BUDGET", "cache_budget_bytes must be >= 0")
        self.cache_budget_bytes = budget
        self.history_contract = vision_history.normalize_history_contract(history_contract)
        self.verify_source_digest = bool(verify_source_digest)
        self.root = Path(dataset_dir)
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
        self._sizes = {path: n for path, _start, n in self.offsets}
        self._cache: dict[Path, dict[str, np.ndarray]] = {}
        self._label_cache: dict[Path, dict[str, np.ndarray]] = {}
        self._lru: OrderedDict[tuple[str, Path], None] = OrderedDict()
        self._resident = _Residency()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.decode_threads = max(1, int(decode_threads))
        self.use_decode_cache = bool(use_decode_cache)
        self._pool = None
        self._closed = False
        # ``max_cached_shards`` caps the number of *full* shard records; the byte
        # budget caps everything (full records + label-only records).
        self.max_cached_shards = None if max_cached_shards is None else max(1, int(max_cached_shards))
        self._max_cached = self.max_cached_shards if self.max_cached_shards is not None else len(self.paths)
        # Decoded-cache identity: one source SHA-256 per shard, computed at most
        # once, plus counters so a silent fallback to PNG decoding is visible.
        self._source_digests: dict[Path, str] = {}
        self._decode_counters = {"hits": 0, "missing": 0, "invalid": 0, "disabled": 0}
        self._decode_reasons: dict[str, int] = {}
        # Segment tables for ``causal-segment-v1``: 4 bytes + 1 bool per frame,
        # a derived index rather than cached pixels.  Bounded to a few shards.
        self._history_index: OrderedDict[Path, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    # -- lifecycle -------------------------------------------------------
    def __len__(self) -> int:
        return self.total

    @property
    def region_names(self) -> list[str]:
        return list(self.region_hw)

    def __enter__(self) -> "VisionDataset":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False

    def close(self) -> None:
        """Release cached arrays and the decode thread pool. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._cache.clear()
        self._label_cache.clear()
        self._lru.clear()
        self._resident.clear()
        self._history_index.clear()
        self._source_digests.clear()
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("v104:CACHE: VisionDataset is closed; create a new instance")

    # -- residency -------------------------------------------------------
    def _touch(self, kind: str, path: Path) -> None:
        """Every hit refreshes the entry: one unified LRU for full and label records."""
        key = (kind, path)
        self._lru[key] = None
        self._lru.move_to_end(key)

    def _evict(self, incoming_bytes: int = 0) -> None:
        """Evict in LRU order until the byte budget *and* the shard-count cap hold.

        ``incoming_bytes`` is the record about to be inserted: making room *before*
        the insert keeps ``resident_bytes`` inside the budget at every instant, not
        only after the fact.  A record that alone exceeds the budget is still
        evicted (it is returned to the caller without becoming resident).
        """
        while True:
            over_bytes = self._resident.nbytes + int(incoming_bytes) > self.cache_budget_bytes
            over_count = (
                self.max_cached_shards is not None
                and len(self._cache) > self.max_cached_shards
            )
            if not over_bytes and not over_count:
                return
            if not self._lru:
                return
            kind, path = next(iter(self._lru))
            del self._lru[(kind, path)]
            table = self._cache if kind == "full" else self._label_cache
            entries = table.pop(path, None)
            if entries is not None:
                self._resident.discard(entries)
            self._stats["evictions"] += 1

    def cache_stats(self) -> dict:
        """Residency accounting for the byte budget (D10).

        ``resident_bytes`` counts every distinct ``ndarray`` object once even when a
        full record and a label record share it; ``full.bytes``/``label.bytes`` are
        per-kind logical sums and may overlap.  mmap pages, temporary batches, the
        OS page cache and the thread pool are outside this budget - RSS is not
        promised.
        """
        return {
            "budget_bytes": int(self.cache_budget_bytes),
            "resident_bytes": int(self._resident.nbytes),
            "hits": int(self._stats["hits"]),
            "misses": int(self._stats["misses"]),
            "evictions": int(self._stats["evictions"]),
            "full": {"entries": len(self._cache), "bytes": _entries_bytes(self._cache)},
            "label": {"entries": len(self._label_cache), "bytes": _entries_bytes(self._label_cache)},
        }

    def decode_cache_stats(self) -> dict:
        """How often the identity-checked decoded cache was used versus PNG fallback."""
        return {
            "enabled": bool(self.use_decode_cache),
            "hits": int(self._decode_counters["hits"]),
            "missing": int(self._decode_counters["missing"]),
            "invalid": int(self._decode_counters["invalid"]),
            "disabled": int(self._decode_counters["disabled"]),
            "reasons": dict(self._decode_reasons),
            "verified_source_digests": len(self._source_digests),
            "contract": DECODE_CACHE_SCHEMA,
        }

    def _note_decode_cache(self, reason: str) -> None:
        if reason == "hit":
            self._decode_counters["hits"] += 1
        elif reason == "missing":
            self._decode_counters["missing"] += 1
        else:
            self._decode_counters["invalid"] += 1
        self._decode_reasons[reason] = self._decode_reasons.get(reason, 0) + 1

    def _shard_digest(self, path: Path) -> str:
        """One SHA-256 per shard, reused across regions and calls (not per batch)."""
        digest = self._source_digests.get(path)
        if digest is None:
            digest = _file_digest(path)
            self._source_digests[path] = digest
        return digest

    def _locate(self, index: int) -> tuple[Path, int, int]:
        for path, start, size in self.offsets:
            if index < start + size:
                return path, start, size
        raise IndexError(index)

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        """Full shard record (pixel blobs included), resident only while budgeted.

        A record that alone exceeds the budget is returned to the caller without
        being kept, which is what makes ``cache_budget_bytes=0`` a working mode.
        """
        self._ensure_open()
        record = self._cache.get(path)
        if record is not None:
            self._stats["hits"] += 1
            self._touch("full", path)
            return record
        self._stats["misses"] += 1
        with np.load(path) as data:
            record = {key: data[key] for key in data.files}
        if self.cache_budget_bytes > 0:
            self._evict(_dict_bytes(record))  # make room first: never over budget
            self._cache[path] = record
            self._resident.add(record)
            self._touch("full", path)
            self._evict()
        return record

    def _decoded_cache(self, shard: Path, region: str, out_h: int, out_w: int):
        """mmap the identity-checked decoded-frame cache, or None to decode PNGs.

        A cache file is only used when n/H/W/C/dtype, the sidecar schema, the
        region/shape, the source shard's name/frames/bytes and (once per shard) its
        SHA-256 all match.  Files without a sidecar - every v1.0.3 cache on disk -
        fall back to PNG decoding and are counted, never silently trusted.
        """
        if not self.use_decode_cache:
            self._decode_counters["disabled"] += 1
            return None
        path = decoded_cache_path(shard, region, out_h, out_w)
        if not path.exists():
            self._note_decode_cache("missing")
            return None
        try:
            array = np.load(path, mmap_mode="r")
        except Exception:
            self._note_decode_cache("unreadable")
            return None
        if (
            array.ndim != 4
            or array.shape[0] != self._sizes.get(shard)
            or tuple(int(v) for v in array.shape[1:3]) != (int(out_h), int(out_w))
            or int(array.shape[3]) != vision.VISION_CHANNELS
            or array.dtype != np.dtype(np.uint8)
        ):
            self._note_decode_cache("shape")
            return None
        check = check_decode_cache(
            shard,
            region,
            out_h,
            out_w,
            frames=self._sizes.get(shard),
            verify_source=self.verify_source_digest,
            source_digest=self._shard_digest(shard) if self.verify_source_digest else None,
        )
        if not check.ok:
            self._note_decode_cache(check.reason)
            return None
        self._note_decode_cache("hit")
        return check.array

    def shard_chunked_order(self, indices: list[int], rng: np.random.Generator) -> list[int]:
        """Reshuffle ``indices`` so each contiguous run comes from one shard.

        Reading 96 samples spread over 96 shards costs 90 ms on this machine; reading
        96 samples from 2 shards costs 21 ms (measured, decoded-cache files).  The
        sampler only needs "a random subset per epoch", not "every frame from a
        different file", so shuffling at *shard* granularity keeps the epoch's sample
        set identical while cutting the per-batch I/O by ~4x.
        """
        by_shard: dict[Path, list[int]] = {}
        for index in indices:
            path, _start, _size = self._locate(index)
            by_shard.setdefault(path, []).append(index)
        paths = list(by_shard)
        rng.shuffle(paths)
        order: list[int] = []
        for path in paths:
            rows = by_shard[path]
            rng.shuffle(rows)
            order.extend(rows)
        return order

    def _group(self, indices: list[int]) -> dict[Path, list[tuple[int, int]]]:
        by_shard: dict[Path, list[tuple[int, int]]] = {}
        for position, index in enumerate(indices):
            path, _start, _size = self._locate(index)
            by_shard.setdefault(path, []).append((position, index - _start))
        return by_shard

    # -- labels ----------------------------------------------------------
    def _load_labels(self, path: Path) -> dict[str, np.ndarray]:
        """Label members of one shard, cached separately from the pixel blobs.

        A shard's labels are ~140 KB while its three PNG blobs are ~30 MB.  Reading
        only the label members keeps ``labels()`` fast without pinning the pixel
        blobs in RAM - which matters now that frames come from the decode cache and
        the npz is otherwise never opened.

        When the full record of the same shard is already resident its label
        members are reused as-is, so the two caches share one array object instead
        of paying for a second copy (the residency tracker counts shared objects
        once).
        """
        self._ensure_open()
        cached = self._label_cache.get(path)
        if cached is not None:
            self._stats["hits"] += 1
            self._touch("label", path)
            return cached
        shared = self._cache.get(path)
        if shared is not None:
            self._stats["hits"] += 1
            self._touch("full", path)
            return shared
        self._stats["misses"] += 1
        with np.load(path) as data:
            cached = {key: data[key] for key in data.files if not key.endswith("_blob")}
        if self.cache_budget_bytes > 0:
            self._evict(_dict_bytes(cached))  # make room first: never over budget
            self._label_cache[path] = cached
            self._resident.add(cached)
            self._touch("label", path)
            self._evict()
        return cached

    def labels(self, indices: list[int]) -> dict[str, np.ndarray]:
        """Privileged arrays for ``indices`` (never touches pixels)."""
        self._ensure_open()
        by_shard = self._group(indices)
        keys = ["self_feat", "bullets", "danger", "action", "meta"]
        first = next(iter(by_shard))
        sample = self._load_labels(first)
        out = {
            key: np.zeros((len(indices),) + sample[key].shape[1:], dtype=sample[key].dtype)
            for key in keys
        }
        if "value" in sample:
            out["value"] = np.zeros((len(indices),) + sample["value"].shape[1:], dtype=sample["value"].dtype)
        for path, entries in by_shard.items():
            shard = self._load_labels(path)
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
        self._ensure_open()
        native = self.region_hw.get(region)
        if native is None:
            raise KeyError(f"region {region!r} not present; dataset has {sorted(self.region_hw)}")
        out_h = out_h or native[0]
        out_w = out_w or native[1]
        blob_key, off_key = f"{region}_blob", f"{region}_offset"
        out = np.zeros((len(indices), out_h, out_w, vision.VISION_CHANNELS), dtype=np.uint8)

        jobs: list[tuple[int, bytes]] = []
        for path, entries in self._group(indices).items():
            cached = self._decoded_cache(path, region, out_h, out_w)
            if cached is not None:
                rows = [local for _, local in entries]
                out[[position for position, _ in entries]] = cached[rows]
                continue
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

        The composition rule is the dataset's ``history_contract``:

        * ``legacy`` (the v1.0.3 default) clamps inside the sample's own shard.
          Shards come from different capture runs, so the frame before a shard's
          first sample would teach fake motion - but this is what every existing
          checkpoint was trained with, so it stays available unchanged.
        * ``causal-segment-v1`` clamps to the causal segment (same shard, same
          stage, frame delta 2, unchanged death counter) and repeats the sample
          itself when its meta is untrusted.  See :mod:`touhou_rl.vision_history`.

        The signature is unchanged; ``indices`` must be legal dataset indexes and
        ``depth`` a positive integer.
        """
        self._ensure_open()
        depth_value = vision_history.positive_int(depth)
        wanted = _validated_indices(indices, self.total)
        out = np.empty((len(wanted), depth_value), dtype=np.int64)
        if not wanted:
            return out
        grouped: dict[Path, list[tuple[int, int]]] = {}
        shard_of: dict[Path, tuple[int, int]] = {}
        for row, index in enumerate(wanted):
            path, start, size = self._locate(index)
            grouped.setdefault(path, []).append((row, index - start))
            shard_of[path] = (start, size)
        for path, entries in grouped.items():
            start, size = shard_of[path]
            rows = np.asarray([row for row, _local in entries], dtype=np.int64)
            local = np.asarray([local for _row, local in entries], dtype=np.int64)
            if self.history_contract == vision_history.HISTORY_LEGACY:
                history = vision_history.history_rows(local, depth_value, self.history_contract)
            else:
                starts, reliable = self._history_state(path, size)
                history = vision_history.history_rows(
                    local, depth_value, self.history_contract, starts, reliable
                )
            out[rows] = history + int(start)
        return out

    # -- history ---------------------------------------------------------
    def meta_of_shard(self, path: Path) -> np.ndarray:
        """The shard's ``meta`` block as float64 ``[n, 8]`` (labels only, no pixels)."""
        with np.load(Path(path)) as data:
            return np.asarray(data["meta"], dtype=np.float64)

    def _history_state(self, path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
        """Causal segment table of one shard, computed once and kept for a while."""
        cached = self._history_index.get(path)
        if cached is not None:
            self._history_index.move_to_end(path)
            return cached
        meta = self.meta_of_shard(path)
        if meta.shape[0] != int(size):
            raise vision_history.v104_error(
                "DATA", f"{path.name}: meta rows {meta.shape[0]} != frames {int(size)}"
            )
        starts, reliable = vision_history.segment_starts(meta)
        self._history_index[path] = (starts, reliable)
        while len(self._history_index) > self._HISTORY_INDEX_SHARDS:
            self._history_index.popitem(last=False)
        return starts, reliable

    def history_audit(self, depth: int = 3) -> dict:
        """Recomputable history audit (meta only; see :func:`vision_history.history_audit`)."""
        self._ensure_open()
        return vision_history.history_audit(self, depth)

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

    # -- metadata --------------------------------------------------------
    def stage_of(self, indices: list[int] | np.ndarray) -> np.ndarray:
        """每个索引所属 stage（来自分片 meta 的第 2 列）。返回 int64 数组，形状同 indices。
        indices 是**数据集全局索引**（0..total-1）；stage_of[i] 即全局索引 i 的 stage，
        因此调用方若需要按索引查表，应传 range(total)。"""
        requested = np.asarray(indices, dtype=np.int64)
        flat = requested.reshape(-1)
        stages = np.empty(flat.size, dtype=np.int64)
        cache: dict[Path, np.ndarray] = {}
        for position, index in enumerate(flat):
            path, start, _size = self._locate(int(index))
            column = cache.get(path)
            if column is None:
                # Only ``meta`` is opened: no PNG is decoded, so this costs a few
                # milliseconds per shard even on a 72k-frame dataset.
                with np.load(path) as data:
                    column = np.rint(data["meta"][:, 1]).astype(np.int64)
                cache[path] = column
            stages[position] = column[int(index) - start]
        return stages.reshape(requested.shape)
