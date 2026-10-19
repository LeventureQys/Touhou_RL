"""Tests for the data-cache layer (values, residency budget, cache identity).

The two original speedups
------------------------
Both exist because the GPU was idle 60%+ of every batch: PNG inflation cost
148 ms per 96-sample batch and the `uint8 -> float32` scaling another 38 ms,
while the GPU step itself takes 21 ms.  The fixes are:

* `build_decode_cache` / `VisionDataset(use_decode_cache=True)` - store each
  region already decoded at the model resolution as a memory-mappable `.npy`
  (233 KB/frame versus 253 KB for the PNG), so the hot path is a memcpy;
* `regions_to_batch(..., device=...)` - do the `/255` scaling on the device
  instead of writing a 4x larger float32 array on the CPU.

None of them may change the values the model sees, which the first half pins.

v1.0.4 adds three properties that are just as observable:

* **D10 residency budget** - the full-shard and label caches share one byte
  budget counted from real ``ndarray.nbytes`` (the v1.0.3 estimate of 1200 B per
  frame was ~200x low), one unified LRU refreshed on every hit, an optional
  shard-count cap, and ``cache_budget_bytes=0`` meaning "never retain".
* **D11 cache identity** - a decoded cache file is only used when its sidecar
  proves the source shard, shape, dtype and resolution; anything else falls back
  to PNG decoding and is counted, never silently trusted.
* **Lifecycle** - ``cache_stats()`` reports what is resident and ``close()``
  releases everything exactly once.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from touhou_rl import obs, vision
from touhou_rl.vision_data import (
    DECODE_CACHE_SCHEMA,
    VisionDataset,
    VisionShardWriter,
    build_decode_cache,
    check_decode_cache,
    decoded_cache_path,
    decoded_sidecar_path,
    verify_decode_cache,
)

SIZES = {"playfield": (24, 32), "hud_score": (8, 32), "hud_status": (16, 32)}


def _regions(value: int) -> dict[str, bytes]:
    sizes = {"playfield": (48, 64), "hud_score": (8, 32), "hud_status": (16, 32)}
    return {
        name: vision.encode_png(np.full((h, w, 3), (value * 7 + i) % 255, dtype=np.uint8))
        for i, (name, (h, w)) in enumerate(sizes.items())
    }


def _make_dataset(tmp_path, frames: int = 7) -> "VisionDataset":
    writer = VisionShardWriter(tmp_path, shard_size=3, tag="t")
    rng = np.random.default_rng(0)
    for index in range(frames):
        writer.add(
            _regions(index * 11),
            obs.Observation(
                self_feat=rng.random(obs.N_SELF, dtype=np.float32),
                bullets=rng.random((obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float32).astype(np.float16),
                danger_map=rng.integers(0, 5, size=(obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8),
            ),
            (0, 1, 0, 0),
            {name: index for name in
             ("frame", "stage", "score", "graze", "lives", "bombs", "power", "deaths")},
        )
    writer.close()
    return tmp_path


def test_decode_cache_is_value_identical_to_png_decoding(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    before = VisionDataset(dataset_dir, use_decode_cache=False)
    indices = [0, 3, 5]
    expected = {name: before.frames(indices, name, *size) for name, size in SIZES.items()}

    report = build_decode_cache(dataset_dir, SIZES, verbose=False)
    assert report["files_written"] == 3 * 3  # 3 shards x 3 regions

    after = VisionDataset(dataset_dir, use_decode_cache=True)
    for name, size in SIZES.items():
        got = after.frames(indices, name, *size)
        assert np.array_equal(got, expected[name]), name


def test_decode_cache_files_are_created_and_reused(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    first = build_decode_cache(dataset_dir, SIZES, verbose=False)
    second = build_decode_cache(dataset_dir, SIZES, verbose=False)
    assert first["files_written"] > 0
    assert second["files_written"] == 0  # idempotent
    assert second["files_skipped"] == first["files_written"]


def test_missing_cache_falls_back_to_png_decoding(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    dataset = VisionDataset(dataset_dir, use_decode_cache=True)
    # No cache built yet: values must still be right.
    frames = dataset.frames([0, 1], "playfield", *SIZES["playfield"])
    assert frames.shape == (2, SIZES["playfield"][0], SIZES["playfield"][1], 3)
    assert frames.dtype == np.uint8


def test_stale_cache_with_wrong_shape_is_ignored(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    shard = sorted(dataset_dir.glob("*.npz"))[0]
    stale = decoded_cache_path(shard, "playfield", *SIZES["playfield"])
    np.save(stale, np.zeros((999, SIZES["playfield"][0], SIZES["playfield"][1], 3), np.uint8))
    dataset = VisionDataset(dataset_dir, use_decode_cache=True)
    fresh = VisionDataset(dataset_dir, use_decode_cache=False)
    assert np.array_equal(
        dataset.frames([0, 1], "playfield", *SIZES["playfield"]),
        fresh.frames([0, 1], "playfield", *SIZES["playfield"]),
    )


def test_regions_to_batch_values_are_unchanged():
    """Scaling moved to the device, but the numbers must be bit-identical."""
    raw = {name: np.full((2, 1, 6, 8, 3), 51, np.uint8) for name in ("playfield", "hud_score", "hud_status")}
    batch = vision.regions_to_batch(raw)
    for name, value in batch.items():
        assert value.dtype == __import__("torch").float32
        assert tuple(value.shape) == (2, 1, 6, 8, 3)
        assert abs(float(value.max()) - 0.2) < 1e-6


def test_regions_to_batch_matches_on_cuda_when_available():
    torch = __import__("torch")
    raw = {name: np.arange(2 * 1 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 1, 4, 5, 3)
           for name in ("playfield", "hud_score", "hud_status")}
    cpu = vision.regions_to_batch(raw)
    if not torch.cuda.is_available():
        pytest.skip("no cuda")
    gpu = vision.regions_to_batch(raw, "cuda")
    for name in raw:
        assert torch.allclose(cpu[name], gpu[name].cpu(), atol=1e-6)

def test_shard_chunked_order_preserves_the_sample_set():
    """Same frames, different order - a chunked shuffle must not drop or duplicate."""
    import numpy as np
    from pathlib import Path
    from touhou_rl.vision_data import VisionDataset

    ds = VisionDataset(Path("third_party/th08/build/rl/datasets/pilot-ud0059-v2"))
    base = list(range(0, 1000, 7))
    order = ds.shard_chunked_order(list(base), np.random.default_rng(3))
    assert sorted(order) == sorted(base)
    assert order != base


def test_shard_chunked_order_keeps_batches_within_few_shards():
    """A batch of 96 must touch at most a couple of shard files."""
    import numpy as np
    from pathlib import Path
    from touhou_rl.vision_data import VisionDataset

    ds = VisionDataset(Path("third_party/th08/build/rl/datasets/pilot-ud0059-v2"))
    order = ds.shard_chunked_order(list(range(0, 72413, 3)), np.random.default_rng(7))
    files = {ds._locate(index)[0] for index in order[:96]}
    assert len(files) <= 3, len(files)


# --- D10: residency budget, LRU and lifecycle -------------------------------
def _shard_paths(dataset_dir) -> list:
    return sorted(dataset_dir.glob("*.npz"))


def _record_bytes(path) -> int:
    with np.load(path) as data:
        return sum(int(data[key].nbytes) for key in data.files)


def test_default_budget_is_512_mib(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    dataset = VisionDataset(dataset_dir)
    try:
        assert dataset.cache_budget_bytes == 536_870_912
        assert dataset.cache_stats()["budget_bytes"] == 536_870_912
    finally:
        dataset.close()


def test_negative_budget_is_rejected(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    with pytest.raises(ValueError, match=r"^v104:BUDGET: cache_budget_bytes must be >= 0$"):
        VisionDataset(dataset_dir, cache_budget_bytes=-1)


def test_zero_budget_retains_nothing_but_still_returns_values(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    plain = VisionDataset(dataset_dir, use_decode_cache=False)
    expected = plain.frames([0, 4], "playfield", *SIZES["playfield"])
    plain.close()
    dataset = VisionDataset(dataset_dir, cache_budget_bytes=0, use_decode_cache=False)
    try:
        got = dataset.frames([0, 4], "playfield", *SIZES["playfield"])
        assert np.array_equal(got, expected)
        stats = dataset.cache_stats()
        assert stats["resident_bytes"] == 0
        assert stats["full"] == {"entries": 0, "bytes": 0}
        assert stats["label"] == {"entries": 0, "bytes": 0}
        assert stats["evictions"] == 0
        assert stats["misses"] == 2  # shard 0 and shard 1 were loaded, then dropped
    finally:
        dataset.close()


def test_byte_budget_uses_real_ndarray_nbytes(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    shard_bytes = _record_bytes(_shard_paths(dataset_dir)[0])
    dataset = VisionDataset(dataset_dir, cache_budget_bytes=shard_bytes, use_decode_cache=False)
    try:
        for index in (0, 3, 6):  # one sample per shard
            dataset.frames([index], "playfield", *SIZES["playfield"])
        stats = dataset.cache_stats()
        assert stats["full"]["entries"] == 1
        assert stats["resident_bytes"] <= shard_bytes
        assert stats["resident_bytes"] == stats["full"]["bytes"]
        assert stats["evictions"] >= 1
    finally:
        dataset.close()


def test_single_entry_over_budget_is_returned_but_not_resident(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    plain = VisionDataset(dataset_dir, use_decode_cache=False)
    expected = plain.frames([0], "playfield", *SIZES["playfield"])
    plain.close()
    dataset = VisionDataset(dataset_dir, cache_budget_bytes=1, use_decode_cache=False)
    try:
        got = dataset.frames([0], "playfield", *SIZES["playfield"])
        assert np.array_equal(got, expected)
        stats = dataset.cache_stats()
        assert stats["resident_bytes"] == 0
        assert stats["full"]["entries"] == 0
    finally:
        dataset.close()


def test_max_cached_shards_caps_full_records(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    dataset = VisionDataset(
        dataset_dir, max_cached_shards=1, cache_budget_bytes=10**9, use_decode_cache=False
    )
    try:
        for index in (0, 3, 6):
            dataset.frames([index], "playfield", *SIZES["playfield"])
        stats = dataset.cache_stats()
        assert stats["full"]["entries"] == 1
        assert stats["evictions"] == 2
    finally:
        dataset.close()


def test_lru_hit_refreshes_the_entry_and_survives_eviction(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    size = SIZES["playfield"]
    dataset = VisionDataset(
        dataset_dir, max_cached_shards=2, cache_budget_bytes=10**9, use_decode_cache=False
    )
    try:
        dataset.frames([0], "playfield", *size)  # shard 0: miss
        dataset.frames([3], "playfield", *size)  # shard 1: miss
        dataset.frames([0], "playfield", *size)  # shard 0 again: hit, refreshed
        dataset.frames([6], "playfield", *size)  # shard 2: miss -> evicts shard 1
        dataset.frames([0], "playfield", *size)  # still resident: hit
        dataset.frames([3], "playfield", *size)  # was evicted: miss
        stats = dataset.cache_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 4
        assert stats["evictions"] == 2
        assert stats["full"]["entries"] == 2
    finally:
        dataset.close()


def test_max_cached_shards_counts_full_records_only(tmp_path):
    """The count cap is about complete shard records; labels are byte-bounded.

    This is the contracted reading ("max_cached_shards 完整片数量上限"): a label-only
    record is not a complete shard, so a labels()-only workload is bounded by the
    byte budget instead of thrashing to a handful of entries.
    """
    dataset_dir = _make_dataset(tmp_path)
    dataset = VisionDataset(
        dataset_dir, max_cached_shards=1, cache_budget_bytes=10**9, use_decode_cache=False
    )
    try:
        for index in (0, 3, 6):
            dataset.labels([index])
        stats = dataset.cache_stats()
        assert stats["full"]["entries"] == 0
        assert stats["label"]["entries"] == 3
        assert stats["evictions"] == 0
    finally:
        dataset.close()


def test_label_records_are_evicted_by_the_byte_budget(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    one_label = _record_bytes(_shard_paths(dataset_dir)[0])  # upper bound for a label record
    dataset = VisionDataset(
        dataset_dir, cache_budget_bytes=one_label, use_decode_cache=False
    )
    try:
        for index in (0, 3, 6):
            dataset.labels([index])
        stats = dataset.cache_stats()
        assert stats["label"]["entries"] <= 1
        assert stats["resident_bytes"] <= one_label
        assert stats["evictions"] >= 1
    finally:
        dataset.close()


def test_full_and_label_records_share_one_object(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    dataset = VisionDataset(dataset_dir, use_decode_cache=False)
    try:
        dataset.frames([0], "playfield", *SIZES["playfield"])  # full record of shard 0
        before = dataset.cache_stats()
        dataset.labels([0])  # must reuse the resident full record
        after = dataset.cache_stats()
        assert after["label"]["entries"] == 0
        assert after["hits"] > before["hits"]
        assert after["resident_bytes"] == before["resident_bytes"] == after["full"]["bytes"]
    finally:
        dataset.close()


def test_eviction_pressure_does_not_change_values(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    fresh = VisionDataset(dataset_dir, use_decode_cache=False)
    expected_frames = fresh.frames([0, 1, 2, 3, 4, 5, 6], "playfield", *SIZES["playfield"])
    expected_labels = fresh.labels([0, 3, 6])["self_feat"]
    fresh.close()
    dataset = VisionDataset(
        dataset_dir, max_cached_shards=1, cache_budget_bytes=1, use_decode_cache=False
    )
    try:
        frames = dataset.frames([0, 1, 2, 3, 4, 5, 6], "playfield", *SIZES["playfield"])
        assert np.array_equal(frames, expected_frames)
        assert np.array_equal(dataset.labels([0, 3, 6])["self_feat"], expected_labels)
        assert dataset.cache_stats()["resident_bytes"] == 0
    finally:
        dataset.close()


def test_cache_stats_has_the_contracted_keys(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    dataset = VisionDataset(dataset_dir)
    try:
        stats = dataset.cache_stats()
        assert set(stats) == {
            "budget_bytes",
            "resident_bytes",
            "hits",
            "misses",
            "evictions",
            "full",
            "label",
        }
        assert set(stats["full"]) == {"entries", "bytes"}
        assert set(stats["label"]) == {"entries", "bytes"}
    finally:
        dataset.close()


def test_close_is_idempotent_and_releases_everything(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    dataset = VisionDataset(dataset_dir, use_decode_cache=False)
    dataset.frames([0, 3], "playfield", *SIZES["playfield"])
    dataset.labels([0])
    assert dataset.cache_stats()["resident_bytes"] > 0
    dataset.close()
    dataset.close()  # idempotent
    stats = dataset.cache_stats()
    assert stats["resident_bytes"] == 0
    assert stats["full"] == {"entries": 0, "bytes": 0}
    assert stats["label"] == {"entries": 0, "bytes": 0}
    with pytest.raises(RuntimeError, match=r"^v104:CACHE: "):
        dataset.frames([0], "playfield", *SIZES["playfield"])


def test_context_manager_closes_the_dataset(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    with VisionDataset(dataset_dir, use_decode_cache=False) as dataset:
        assert len(dataset.frames([0], "playfield", *SIZES["playfield"])) == 1
    assert dataset.cache_stats()["resident_bytes"] == 0


# --- D11: decoded-cache identity -------------------------------------------
def test_sidecar_records_the_full_identity(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    report = build_decode_cache(dataset_dir, SIZES, verbose=False)
    assert report["cache_contract"] == DECODE_CACHE_SCHEMA
    shard = _shard_paths(dataset_dir)[0]
    cache = decoded_cache_path(shard, "playfield", *SIZES["playfield"])
    payload = json.loads(decoded_sidecar_path(cache).read_text(encoding="utf-8"))
    assert payload["schema"] == DECODE_CACHE_SCHEMA
    assert payload["region"] == "playfield"
    assert payload["dtype"] == "uint8"
    assert payload["shape"] == [3, *SIZES["playfield"], 3]
    assert payload["source"]["name"] == shard.name
    assert payload["source"]["frames"] == 3
    assert payload["source"]["bytes"] == shard.stat().st_size
    assert payload["source"]["sha256"] == hashlib.sha256(shard.read_bytes()).hexdigest()
    assert payload["cache_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert check_decode_cache(shard, "playfield", *SIZES["playfield"], frames=3).ok


def test_cache_without_sidecar_falls_back_to_png(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    build_decode_cache(dataset_dir, SIZES, verbose=False)
    shard = _shard_paths(dataset_dir)[0]
    cache = decoded_cache_path(shard, "playfield", *SIZES["playfield"])
    decoded_sidecar_path(cache).unlink()
    dataset = VisionDataset(dataset_dir)
    fresh = VisionDataset(dataset_dir, use_decode_cache=False)
    try:
        assert np.array_equal(
            dataset.frames([0, 1], "playfield", *SIZES["playfield"]),
            fresh.frames([0, 1], "playfield", *SIZES["playfield"]),
        )
        stats = dataset.decode_cache_stats()
        assert stats["hits"] == 0
        assert stats["invalid"] >= 1
        assert stats["reasons"].get("no-identity") == 1
    finally:
        dataset.close()
        fresh.close()


def test_cache_with_wrong_source_digest_falls_back(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    build_decode_cache(dataset_dir, SIZES, verbose=False)
    shard = _shard_paths(dataset_dir)[0]
    cache = decoded_cache_path(shard, "playfield", *SIZES["playfield"])
    sidecar = decoded_sidecar_path(cache)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["source"]["sha256"] = "0" * 64
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    dataset = VisionDataset(dataset_dir)
    fresh = VisionDataset(dataset_dir, use_decode_cache=False)
    try:
        assert np.array_equal(
            dataset.frames([0], "playfield", *SIZES["playfield"]),
            fresh.frames([0], "playfield", *SIZES["playfield"]),
        )
        stats = dataset.decode_cache_stats()
        assert stats["hits"] == 0
        assert stats["reasons"].get("source-digest") == 1
    finally:
        dataset.close()
        fresh.close()


@pytest.mark.parametrize(
    "mutation",
    ["frames", "height", "width", "channels", "dtype", "ndim"],
)
def test_structurally_wrong_cache_falls_back_to_png(tmp_path, mutation):
    dataset_dir = _make_dataset(tmp_path)
    build_decode_cache(dataset_dir, SIZES, verbose=False)
    shard = _shard_paths(dataset_dir)[0]
    height, width = SIZES["playfield"]
    cache = decoded_cache_path(shard, "playfield", height, width)
    good = np.load(cache)
    broken = good
    if mutation == "frames":
        broken = np.zeros((999, height, width, 3), np.uint8)
    elif mutation == "height":
        broken = np.zeros((3, height + 1, width, 3), np.uint8)
    elif mutation == "width":
        broken = np.zeros((3, height, width + 1, 3), np.uint8)
    elif mutation == "channels":
        broken = np.zeros((3, height, width, 1), np.uint8)
    elif mutation == "dtype":
        broken = good.astype(np.float32)
    elif mutation == "ndim":
        broken = good.reshape(3, height, width * 3)
    np.save(cache, broken, allow_pickle=False)
    dataset = VisionDataset(dataset_dir)
    fresh = VisionDataset(dataset_dir, use_decode_cache=False)
    try:
        got = dataset.frames([0, 1], "playfield", height, width)
        assert np.array_equal(got, fresh.frames([0, 1], "playfield", height, width))
        assert got.shape == (2, height, width, 3)
        assert got.dtype == np.uint8
        stats = dataset.decode_cache_stats()
        assert stats["hits"] == 0
        assert stats["invalid"] >= 1
    finally:
        dataset.close()
        fresh.close()


def test_rebuild_repairs_invalid_files_and_reports_bytes(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    first = build_decode_cache(dataset_dir, SIZES, verbose=False)
    assert first["files_written"] == 9
    assert first["files_invalid"] == 0 and first["files_repaired"] == 0
    shard = _shard_paths(dataset_dir)[0]
    decoded_sidecar_path(decoded_cache_path(shard, "playfield", *SIZES["playfield"])).unlink()
    second = build_decode_cache(dataset_dir, SIZES, verbose=False)
    assert second["files_skipped"] == 8
    assert second["files_invalid"] == 1
    assert second["files_repaired"] == 1
    assert second["files_written"] == 1
    assert second["files_present"] == 9
    expected = sum(
        decoded_cache_path(path, region, *size).stat().st_size
        for path in _shard_paths(dataset_dir)
        for region, size in SIZES.items()
    )
    assert second["bytes_total"] == expected
    assert second["bytes_written"] < second["bytes_total"]
    third = build_decode_cache(dataset_dir, SIZES, verbose=False)
    assert third["files_written"] == 0
    assert third["files_invalid"] == 0
    assert third["bytes_total"] == expected


def test_interrupted_build_keeps_the_previous_file_intact(tmp_path, monkeypatch):
    from touhou_rl import vision_data

    dataset_dir = _make_dataset(tmp_path)
    build_decode_cache(dataset_dir, SIZES, verbose=False)
    shard = _shard_paths(dataset_dir)[0]
    cache = decoded_cache_path(shard, "playfield", *SIZES["playfield"])
    before = cache.read_bytes()
    decoded_sidecar_path(cache).unlink()  # force a repair
    real_digest = vision_data._file_digest

    def exploding_digest(path, chunk=1 << 20):
        if ".tmp-" in str(path):
            raise RuntimeError("simulated crash while hashing the temporary cache")
        return real_digest(path, chunk)

    monkeypatch.setattr(vision_data, "_file_digest", exploding_digest)
    with pytest.raises(RuntimeError, match="simulated crash"):
        build_decode_cache(dataset_dir, SIZES, verbose=False)
    monkeypatch.setattr(vision_data, "_file_digest", real_digest)
    assert cache.read_bytes() == before  # never replaced by a partial file
    assert not list(dataset_dir.glob("*tmp-*"))
    dataset = VisionDataset(dataset_dir, use_decode_cache=False)
    try:
        assert dataset.frames([0], "playfield", *SIZES["playfield"]).shape == (
            1,
            *SIZES["playfield"],
            3,
        )
    finally:
        dataset.close()


def test_verify_decode_cache_reports_ok_and_counts_bytes(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    build_decode_cache(dataset_dir, SIZES, verbose=False)
    report = verify_decode_cache(dataset_dir, SIZES, verify_payload=True)
    assert report["files"] == 9
    assert report["files_ok"] == 9
    assert report["files_invalid"] == 0
    assert report["payload_verified"] is True
    expected = sum(
        decoded_cache_path(path, region, *size).stat().st_size
        for path in _shard_paths(dataset_dir)
        for region, size in SIZES.items()
    )
    assert report["bytes_total"] == expected


def test_verify_decode_cache_flags_missing_files(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    build_decode_cache(dataset_dir, SIZES, verbose=False)
    shard = _shard_paths(dataset_dir)[0]
    decoded_cache_path(shard, "hud_score", *SIZES["hud_score"]).unlink()
    report = verify_decode_cache(dataset_dir, SIZES)
    assert report["files_invalid"] == 1
    assert report["invalid_reasons"] == {"missing": 1}


def test_cache_of_another_resolution_is_not_used(tmp_path):
    dataset_dir = _make_dataset(tmp_path)
    build_decode_cache(dataset_dir, SIZES, verbose=False)
    dataset = VisionDataset(dataset_dir)
    fresh = VisionDataset(dataset_dir, use_decode_cache=False)
    try:
        other = (12, 16)
        assert np.array_equal(
            dataset.frames([0], "playfield", *other),
            fresh.frames([0], "playfield", *other),
        )
        assert dataset.decode_cache_stats()["reasons"].get("missing") == 1
    finally:
        dataset.close()
        fresh.close()
