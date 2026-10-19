"""Tests for the versioned history contracts and the history audit (v1.0.4 D11).

What is pinned here
--------------------
* ``legacy`` reproduces the v1.0.3 rule bit for bit (existing checkpoints keep
  their meaning), while ``causal-segment-v1`` never joins frames it cannot prove
  are continuous: same shard, same stage, finite strictly increasing frames with a
  delta of exactly 2, unchanged death counter.
* Every break flavour is covered: shard start, stage change, death change, frame
  delta 0 / 1 / 3 / negative, missing (NaN) meta and the "an untrusted sample also
  breaks the samples after it" rule.
* The audit is recomputable: a brute-force reimplementation in this file (plain
  Python loops, no shared code with the module) has to agree on the delta
  histogram, the reset counts and the repeated-history ratio.
"""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl import obs, vision, vision_history
from touhou_rl.vision_data import META_FIELDS, VisionDataset, VisionShardWriter


# --- synthetic shards -------------------------------------------------------
def _regions(value: int) -> dict[str, bytes]:
    sizes = {"playfield": (4, 4), "hud_score": (2, 4), "hud_status": (3, 4)}
    return {
        name: vision.encode_png(np.full((height, width, 3), value % 250, dtype=np.uint8))
        for name, (height, width) in sizes.items()
    }


def _row(frame: float, stage: float = 0, deaths: float = 0) -> dict:
    return {
        "frame": frame,
        "stage": stage,
        "score": 0,
        "graze": 0,
        "lives": 2,
        "bombs": 2,
        "power": 0.0,
        "deaths": deaths,
    }


def _observation(seed: int = 0) -> obs.Observation:
    rng = np.random.default_rng(seed)
    return obs.Observation(
        self_feat=rng.random(obs.N_SELF, dtype=np.float32),
        bullets=rng.random((obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float32).astype(np.float16),
        danger_map=rng.integers(0, 3, size=(obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8),
    )


def _write_dataset(root, shards: list[list[dict]], tag: str = "s") -> VisionDataset:
    """Write one shard file per ``shards`` entry; return the causal dataset over it."""
    for index, rows in enumerate(shards):
        writer = VisionShardWriter(root, shard_size=max(1, len(rows)), tag=f"{tag}{index:02d}")
        for row in rows:
            writer.add(_regions(index), _observation(index), (0, 1, 0, 0), row)
        writer.close()
    return VisionDataset(root, history_contract=vision_history.HISTORY_CAUSAL_SEGMENT_V1)


def _meta_rows(rows: list[dict]) -> np.ndarray:
    return np.asarray([[row[name] for name in META_FIELDS] for row in rows], dtype=np.float64)


def _rows_for(rows: list[dict], local: list[int], depth: int = 3) -> list[list[int]]:
    """History rows of one shard straight from the module's pure functions."""
    meta = _meta_rows(rows)
    starts, reliable = vision_history.segment_starts(meta)
    return (
        vision_history.history_rows(local, depth, vision_history.HISTORY_CAUSAL_SEGMENT_V1, starts, reliable)
        .tolist()
    )


# --- contract plumbing ------------------------------------------------------
def test_history_contract_names_and_metadata_columns():
    assert vision_history.HISTORY_CONTRACTS == ("legacy", "causal-segment-v1")
    assert vision_history.normalize_history_contract("causal-segment-v1") == "causal-segment-v1"
    assert vision_history.normalize_history_contract("legacy") == "legacy"
    with pytest.raises(ValueError, match=r"^v104:CONFIG: "):
        vision_history.normalize_history_contract("causal-segment-v2")
    # The meta column constants must not drift away from the shard format.
    assert META_FIELDS[vision_history.META_FRAME] == "frame"
    assert META_FIELDS[vision_history.META_STAGE] == "stage"
    assert META_FIELDS[vision_history.META_DEATHS] == "deaths"


def test_depth_validation():
    assert vision_history.positive_int(np.int64(3)) == 3
    for bad in (0, -1, 3.0, True, "3"):
        with pytest.raises(ValueError, match=r"^v104:CONFIG: "):
            vision_history.positive_int(bad)


def test_causal_rows_require_segment_inputs():
    with pytest.raises(ValueError, match=r"^v104:CONFIG: "):
        vision_history.history_rows([0, 1], 3, "causal-segment-v1")


def test_depth_one_repeats_nothing():
    rows = _rows_for([_row(value) for value in (0, 2, 4, 6)], [0, 1, 2, 3], depth=1)
    assert rows == [[0], [1], [2], [3]]


# --- the four segment rules -------------------------------------------------
def test_continuous_segment_is_oldest_to_newest():
    rows = _rows_for([_row(value) for value in (0, 2, 4, 6, 8)], [0, 1, 2, 3, 4])
    assert rows == [[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]]


def test_segment_start_repeats_the_segment_first_frame():
    # local 3 starts a new segment because the frame jumped from 4 to 100.
    rows = _rows_for([_row(value) for value in (0, 2, 4, 100, 102)], [3, 4])
    assert rows == [[3, 3, 3], [3, 3, 4]]


def test_stage_change_breaks_the_segment():
    rows = _rows_for(
        [_row(0, stage=0), _row(2, stage=0), _row(4, stage=1), _row(6, stage=1)], [2, 3]
    )
    assert rows == [[2, 2, 2], [2, 2, 3]]


def test_death_change_breaks_the_segment():
    rows = _rows_for(
        [_row(0, deaths=0), _row(2, deaths=0), _row(4, deaths=1), _row(6, deaths=1)], [2, 3]
    )
    assert rows == [[2, 2, 2], [2, 2, 3]]


@pytest.mark.parametrize("delta", [0, 1, 3, 4, -2])
def test_non_two_delta_breaks_the_segment(delta):
    rows = [_row(value) for value in (0, 2, 4, 4 + delta)]
    history = _rows_for(rows, [2, 3])
    # local 2 is still inside the first segment ...
    assert history[0] == [0, 1, 2]
    # ... but local 3 starts a new one, so all of its slots are itself.
    assert history[1] == [3, 3, 3]


def test_non_finite_meta_is_untrusted_and_breaks_the_next_sample():
    rows = [_row(0), _row(2), _row(np.nan), _row(6), _row(8)]
    history = _rows_for(rows, [2, 3, 4])
    # The untrusted sample repeats itself in every slot ...
    assert history[0] == [2, 2, 2]
    # ... and the samples after it may not reach back past it.
    assert history[1] == [3, 3, 3]
    assert history[2] == [3, 3, 4]


def test_non_finite_stage_or_deaths_is_untrusted():
    rows = [_row(0), _row(2, stage=np.inf), _row(4, deaths=np.nan), _row(6)]
    history = _rows_for(rows, [0, 1, 2, 3])
    assert history[1] == [1, 1, 1]
    assert history[2] == [2, 2, 2]
    assert history[3] == [3, 3, 3]


# --- dataset level ----------------------------------------------------------
def test_history_never_crosses_a_shard_boundary(tmp_path):
    first = [_row(value) for value in (0, 2, 4, 6)]
    second = [_row(value) for value in (8, 10, 12)]  # frame numbers continue, shard does not
    dataset = _write_dataset(tmp_path, [first, second])
    try:
        assert dataset.total == 7
        stack = dataset.stack_indices([3, 4, 5], depth=3)
        # index 3 is the last sample of shard 0 -> its own segment history only
        assert stack[0].tolist() == [1, 2, 3]
        # index 4 is the first sample of shard 1: no look-back into shard 0
        assert stack[1].tolist() == [4, 4, 4]
        assert stack[2].tolist() == [4, 4, 5]
    finally:
        dataset.close()


def test_legacy_contract_keeps_the_v103_clamp(tmp_path):
    first = [_row(value) for value in (0, 2, 4, 6)]
    second = [_row(value) for value in (8, 10, 12)]
    _write_dataset(tmp_path, [first, second])
    dataset = VisionDataset(tmp_path)  # default contract is legacy
    try:
        assert dataset.history_contract == "legacy"
        stack = dataset.stack_indices([3, 4, 5], depth=3)
        assert stack[0].tolist() == [1, 2, 3]
        assert stack[1].tolist() == [4, 4, 4]
        assert stack[2].tolist() == [4, 4, 5]
    finally:
        dataset.close()


def test_stack_indices_rejects_illegal_indexes_and_depth(tmp_path):
    dataset = _write_dataset(tmp_path, [[_row(value) for value in (0, 2, 4)]])
    try:
        with pytest.raises(ValueError, match=r"^v104:DATA: "):
            dataset.stack_indices([0, 3], depth=3)
        with pytest.raises(ValueError, match=r"^v104:DATA: "):
            dataset.stack_indices([-1], depth=3)
        with pytest.raises(ValueError, match=r"^v104:DATA: "):
            dataset.stack_indices([0.5], depth=3)
        with pytest.raises(ValueError, match=r"^v104:CONFIG: "):
            dataset.stack_indices([0], depth=0)
    finally:
        dataset.close()


def test_unknown_history_contract_is_rejected(tmp_path):
    _write_dataset(tmp_path, [[_row(value) for value in (0, 2)]])
    with pytest.raises(ValueError, match=r"^v104:CONFIG: "):
        VisionDataset(tmp_path, history_contract="causal-segment-v2")


# --- audit ------------------------------------------------------------------
def _bruteforce_audit(shards: list[list[dict]], depth: int) -> dict:
    """Independent reimplementation: plain loops, explicit slots, no shared code."""
    delta_histogram: dict[str, int] = {}
    resets = {"shard": 0, "stage": 0, "death": 0, "gap": 0, "untrusted": 0, "total": 0}
    repeated = 0
    frames = 0
    for shard_index, rows in enumerate(shards):
        meta = _meta_rows(rows)
        frame = meta[:, 0]
        stage = meta[:, 1]
        deaths = meta[:, 7]
        reliable = [bool(np.isfinite(frame[i]) and np.isfinite(stage[i]) and np.isfinite(deaths[i]))
                    for i in range(len(rows))]
        segment_start = [0] * len(rows)
        for i in range(1, len(rows)):
            continuous = (
                reliable[i]
                and reliable[i - 1]
                and stage[i] == stage[i - 1]
                and deaths[i] == deaths[i - 1]
                and np.isfinite(frame[i] - frame[i - 1])
                and frame[i] - frame[i - 1] == 2
            )
            segment_start[i] = segment_start[i - 1] if continuous else i
            if continuous:
                continue
            if not (reliable[i] and reliable[i - 1]):
                resets["untrusted"] += 1
            else:
                if stage[i] != stage[i - 1]:
                    resets["stage"] += 1
                if deaths[i] != deaths[i - 1]:
                    resets["death"] += 1
                if not (np.isfinite(frame[i] - frame[i - 1]) and frame[i] - frame[i - 1] == 2):
                    resets["gap"] += 1
            resets["total"] += 1
        if shard_index > 0:
            resets["shard"] += 1
            resets["total"] += 1
        for i in range(1, len(rows)):
            delta = frame[i] - frame[i - 1]
            key = "nonfinite" if not np.isfinite(delta) else str(int(round(float(delta))))
            delta_histogram[key] = delta_histogram.get(key, 0) + 1
        for i in range(len(rows)):
            slots = []
            for slot in range(depth):
                candidate = i - (depth - 1 - slot)
                slots.append(i if not reliable[i] else max(candidate, segment_start[i]))
            if depth > 1 and len(set(slots)) < depth:
                repeated += 1
        frames += len(rows)
    return {
        "delta_histogram": delta_histogram,
        "resets": resets,
        "repeated": repeated,
        "frames": frames,
        "ratio": repeated / frames if frames else 0.0,
    }


def _mixed_shards() -> list[list[dict]]:
    """A dataset exercising every reset flavour (frames chosen by hand)."""
    shard0 = [
        _row(0),
        _row(2),
        _row(4),
        _row(7),  # gap (delta 3)
        _row(9),
        _row(11, stage=1),  # stage change
        _row(13, stage=1, deaths=1),  # death change
        _row(15, stage=1, deaths=1),
    ]
    shard1 = [
        _row(200, stage=1, deaths=1),
        _row(202, stage=1, deaths=1),
        _row(np.nan, stage=1, deaths=1),  # untrusted
        _row(206, stage=1, deaths=1),
        _row(208, stage=1, deaths=1),
    ]
    shard2 = [_row(300 + 3 * i, stage=2, deaths=0) for i in range(6)]  # every delta is 3
    return [shard0, shard1, shard2]


@pytest.mark.parametrize("depth", [1, 3])
def test_audit_matches_bruteforce_recomputation(tmp_path, depth):
    shards = _mixed_shards()
    dataset = _write_dataset(tmp_path, shards)
    try:
        audit = dataset.history_audit(depth)
    finally:
        dataset.close()
    expected = _bruteforce_audit(shards, depth)
    assert audit["frames"] == expected["frames"] == 19
    assert audit["shards"] == 3
    assert audit["depth"] == depth
    assert audit["delta_pairs"] == sum(len(rows) - 1 for rows in shards)
    assert audit["delta_histogram"] == expected["delta_histogram"]
    assert audit["repeated_history_samples"] == expected["repeated"]
    assert audit["repeated_history_ratio"] == pytest.approx(expected["ratio"])
    for kind in ("shard", "stage", "death", "gap", "untrusted", "total"):
        assert audit["resets"][kind] == expected["resets"][kind], kind
    if depth == 1:
        assert audit["repeated_history_samples"] == 0


def test_audit_reports_expected_resets_and_overlap(tmp_path):
    shards = _mixed_shards()
    dataset = _write_dataset(tmp_path, shards)
    try:
        audit = dataset.history_audit(3)
    finally:
        dataset.close()
    resets = audit["resets"]
    assert resets["shard"] == 2  # the second and third shard start a new segment
    assert resets["stage"] == 1
    assert resets["death"] == 1
    assert resets["gap"] == 1 + 5  # one jump in shard 0, every sample of shard 2
    assert resets["untrusted"] == 2  # the NaN sample and the sample right after it
    assert resets["total"] == 2 + 1 + 1 + 6 + 2
    # Categories may overlap, so the naive sum is an upper bound, never a lower one.
    assert sum(value for key, value in resets.items() if key != "total") >= resets["total"]
    assert audit["segments"] == resets["total"] + 1


def test_audit_depth_one_is_never_repeated(tmp_path):
    dataset = _write_dataset(tmp_path, _mixed_shards())
    try:
        audit = dataset.history_audit(1)
    finally:
        dataset.close()
    assert audit["depth"] == 1
    assert audit["repeated_history_samples"] == 0
    assert audit["repeated_history_ratio"] == 0.0
    assert audit["history_delta_histogram"] == {}


def test_audit_reads_no_pixels(tmp_path):
    dataset = _write_dataset(tmp_path, _mixed_shards())
    try:
        before = dataset.decode_cache_stats()
        dataset.history_audit(3)
        after = dataset.decode_cache_stats()
    finally:
        dataset.close()
    assert after["hits"] == 0 and after["missing"] == 0
    assert after == before


def test_audit_ratio_denominator_is_frames(tmp_path):
    dataset = _write_dataset(tmp_path, _mixed_shards())
    try:
        audit = dataset.history_audit(3)
    finally:
        dataset.close()
    assert audit["repeated_history_ratio"] == pytest.approx(
        audit["repeated_history_samples"] / audit["frames"]
    )
    assert audit["repeated_history_ratio"] <= 1.0
