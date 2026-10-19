"""Versioned, causal frame-history contracts (v1.0.4 D11 / contract C2).

Why the old rule was wrong
--------------------------
The v1.0.3 loader built a depth-D stack by clamping *inside the sample's own
shard*.  A shard is a capture file cut every 128 samples, not an episode: inside
one shard the capture can skip frames (``every=2`` is the nominal rate, not a
guarantee - the measured deltas are 2, 3, 4, 5, 6, 7, 9, 10, 12, 14 and 15), the
stage can change and the player can lose a life.  Clamping therefore fabricates
motion: a "history" slot that is really the same picture twice, or a frame from
before a stage transition, teaches the model that nothing moved.

This module makes the history rule an explicit, versioned contract:

``legacy``
    The v1.0.3 behaviour: clamp to the shard's first frame.  Kept because every
    existing checkpoint, comparison and report was produced with it; changing it
    silently would invalidate those numbers.

``causal-segment-v1``
    A *continuous segment* is a run of samples inside one shard with the same
    stage, finite strictly increasing frame numbers, ``frame[i] - frame[i-1] == 2``
    and an unchanged death counter.  History never leaves the segment, never
    leaves the shard and never looks forward in time.  Where the segment is
    shorter than the requested depth the missing slots repeat the segment's first
    frame (so a stack still means "the oldest frame we can prove"), oldest slot
    first.  A sample whose meta is missing or non-finite repeats its own frame in
    every slot and also breaks the segment for the samples after it.

What the contract cannot do
---------------------------
There is no per-sample episode id on disk.  A same-stage run that restarts after
a death, a bomb or a menu pause is only detectable when the death counter or the
frame numbering changes; the residual risk is *reported* by
:func:`history_audit` (``resets`` / ``delta_histogram``) instead of hidden.  The
frame-level train/validation split is not touched: two centres on opposite sides
of the split can still share history *pixels* (never labels), which every report
must disclose.
"""

from __future__ import annotations

import numpy as np

# --- contracts --------------------------------------------------------------
HISTORY_LEGACY = "legacy"
HISTORY_CAUSAL_SEGMENT_V1 = "causal-segment-v1"
HISTORY_CONTRACTS: tuple[str, ...] = (HISTORY_LEGACY, HISTORY_CAUSAL_SEGMENT_V1)
DEFAULT_HISTORY_CONTRACT = HISTORY_CAUSAL_SEGMENT_V1

AUDIT_VERSION = "v104-history-audit/1"

#: Column indexes into a shard's ``meta`` block.  ``vision_data.META_FIELDS`` is
#: the canonical name list; ``test_vision_history`` pins the two together so the
#: two modules cannot drift.
META_FRAME = 0
META_STAGE = 1
META_DEATHS = 7

#: Consecutive samples of a continuous segment are 2 game frames apart (capture
#: ``every=2``).  Any other delta is a capture gap and starts a new segment.
SAMPLE_DELTA = 2

#: Reset categories counted by :func:`history_audit`.  ``shard`` marks the first
#: sample of every shard after the first one.
RESET_KINDS: tuple[str, ...] = ("shard", "stage", "death", "gap", "untrusted")


def v104_error(code: str, detail: str) -> ValueError:
    """A contract violation in the shared ``v104:<CODE>: <detail>`` format."""
    return ValueError(f"v104:{code}: {detail}")


def normalize_history_contract(contract: object) -> str:
    """Return the canonical contract name or raise ``v104:CONFIG``."""
    name = str(contract)
    if name not in HISTORY_CONTRACTS:
        raise v104_error(
            "CONFIG",
            f"unknown history_contract {name!r}; expected one of {list(HISTORY_CONTRACTS)}",
        )
    return name


def positive_int(value: object, *, name: str = "depth") -> int:
    """``value`` as an ``int >= 1``, else ``v104:CONFIG`` (``bool`` is rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise v104_error("CONFIG", f"{name} must be a positive integer, got {value!r}")
    number = int(value)
    if number < 1:
        raise v104_error("CONFIG", f"{name} must be >= 1, got {number}")
    return number


# --- meta analysis ----------------------------------------------------------
def meta_columns(meta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(frame, stage, deaths)`` as float64 copies of one shard's meta rows."""
    array = np.asarray(meta)
    if array.ndim != 2 or array.shape[1] <= META_DEATHS:
        raise v104_error("DATA", f"meta must be [n, >={META_DEATHS + 1}], got {array.shape}")
    return (
        array[:, META_FRAME].astype(np.float64),
        array[:, META_STAGE].astype(np.float64),
        array[:, META_DEATHS].astype(np.float64),
    )


def reliable_samples(meta: np.ndarray) -> np.ndarray:
    """``bool[n]`` - False where frame/stage/deaths is not finite (缺meta/不可信)."""
    frame, stage, deaths = meta_columns(meta)
    return np.isfinite(frame) & np.isfinite(stage) & np.isfinite(deaths)


def continuity(meta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(continuous[n-1], delta[n-1], reliable[n])`` for one shard's meta.

    ``continuous[i]`` is True when samples ``i`` and ``i+1`` may share a history:
    both meta rows finite, same stage, same death counter and a frame delta of
    exactly :data:`SAMPLE_DELTA` (which also implies strictly increasing).
    """
    frame, stage, deaths = meta_columns(meta)
    reliable = np.isfinite(frame) & np.isfinite(stage) & np.isfinite(deaths)
    delta = np.diff(frame)
    continuous = (
        reliable[1:]
        & reliable[:-1]
        & (stage[1:] == stage[:-1])
        & (deaths[1:] == deaths[:-1])
        & np.isfinite(delta)
        & (delta == SAMPLE_DELTA)
    )
    return continuous, delta, reliable


def segment_starts(meta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(segment_start_local[i], reliable[i])`` for every sample of one shard.

    ``segment_start_local[i]`` is the local index of the first sample of the
    causal segment that sample ``i`` belongs to; it is ``i`` itself whenever the
    sample starts a new segment (including every untrusted sample).
    """
    array = np.asarray(meta)
    count = int(array.shape[0])
    continuous, _delta, reliable = continuity(array)
    starts = np.arange(count, dtype=np.int64)
    if count == 0:
        return starts, reliable
    new_segment = np.ones(count, dtype=bool)
    new_segment[1:] = ~continuous
    return np.maximum.accumulate(np.where(new_segment, starts, 0)).astype(np.int64), reliable


def break_flags(meta: np.ndarray) -> dict[str, np.ndarray]:
    """Per-sample reset flags for one shard.

    The categories may **overlap**: a break where the stage changed *and* the
    frame delta is 3 sets both ``stage`` and ``gap``.  ``shard`` marks local
    index 0 only; the caller decides whether that shard's first sample is a real
    break (it is not for the dataset's very first sample).
    """
    array = np.asarray(meta)
    count = int(array.shape[0])
    flags = {kind: np.zeros(count, dtype=bool) for kind in RESET_KINDS}
    if count == 0:
        return flags
    flags["shard"][0] = True
    if count == 1:
        return flags
    _continuous, delta, reliable = continuity(array)
    frame, stage, deaths = meta_columns(array)  # noqa: F841 - kept for symmetry/readability
    both = reliable[1:] & reliable[:-1]
    flags["untrusted"][1:] = ~both
    flags["stage"][1:] = both & (stage[1:] != stage[:-1])
    flags["death"][1:] = both & (deaths[1:] != deaths[:-1])
    flags["gap"][1:] = both & ~(np.isfinite(delta) & (delta == SAMPLE_DELTA))
    return flags


# --- history rows -----------------------------------------------------------
def history_rows(
    local: np.ndarray | list[int],
    depth: int,
    contract: str,
    segment_start: np.ndarray | None = None,
    reliable: np.ndarray | None = None,
) -> np.ndarray:
    """In-shard history rows ``int64[n, depth]``, oldest slot first, no future.

    ``local`` are local (in-shard) sample indexes.  ``legacy`` clamps to the
    shard's first frame; ``causal-segment-v1`` clamps to the sample's causal
    segment and repeats the sample itself when its meta is untrusted.
    """
    name = normalize_history_contract(contract)
    depth_value = positive_int(depth)
    indexes = np.asarray(local, dtype=np.int64).reshape(-1)
    count = indexes.size
    offsets = np.arange(depth_value, dtype=np.int64)
    rows = indexes[:, None] - (depth_value - 1 - offsets)[None, :]
    if name == HISTORY_LEGACY:
        np.maximum(rows, 0, out=rows)
        np.minimum(rows, indexes[:, None], out=rows)
        return rows
    if segment_start is None or reliable is None:
        raise v104_error(
            "CONFIG", "causal-segment-v1 needs segment_start and reliable from the shard meta"
        )
    starts = np.asarray(segment_start, dtype=np.int64).reshape(-1)
    trusted = np.asarray(reliable, dtype=bool).reshape(-1)
    if count and (starts.size != trusted.size):
        raise v104_error(
            "DATA", f"segment_start/reliable length mismatch: {starts.size} != {trusted.size}"
        )
    if count and indexes.max() >= starts.size:
        raise v104_error(
            "DATA", f"local index {int(indexes.max())} outside shard of {starts.size} samples"
        )
    lower = starts[indexes]
    np.maximum(rows, lower[:, None], out=rows)
    np.minimum(rows, indexes[:, None], out=rows)
    untrusted = ~trusted[indexes]
    if untrusted.any():
        rows[untrusted] = indexes[untrusted][:, None]
    return rows


# --- audit ------------------------------------------------------------------
def _delta_key(value: float) -> str:
    """Stable histogram key for a measured delta (integers stay integers)."""
    number = float(value)
    if not np.isfinite(number):
        return "nonfinite"
    if number == float(round(number)):
        return str(int(round(number)))
    return repr(number)


def _sorted_histogram(histogram: dict[str, int]) -> dict[str, int]:
    def sort_key(item: tuple[str, int]) -> tuple[int, float]:
        key = item[0]
        try:
            return (0, float(key))
        except ValueError:
            return (1, 0.0)

    return {key: int(value) for key, value in sorted(histogram.items(), key=sort_key)}


def history_audit(dataset: object, depth: int = 3) -> dict:
    """Recomputable history audit over a dataset's meta only (no pixels).

    ``dataset`` must expose ``offsets`` (``[(path, start, frames)]``), ``total``,
    ``history_contract`` and ``meta_of_shard(path)`` - :class:`touhou_rl.
    vision_data.VisionDataset` does.
    """
    depth_value = positive_int(depth)
    contract = normalize_history_contract(getattr(dataset, "history_contract", HISTORY_LEGACY))
    offsets = list(getattr(dataset, "offsets"))
    total = int(getattr(dataset, "total", sum(size for _path, _start, size in offsets)))
    delta_histogram: dict[str, int] = {}
    history_delta_histogram: dict[str, int] = {}
    resets: dict[str, int] = {kind: 0 for kind in RESET_KINDS}
    resets["total"] = 0
    cross_shard = {"stage": 0, "death": 0, "delta_histogram": {}}
    repeated = 0
    segments_total = 0
    longest_segment = 0
    untrusted_frames = 0
    delta_pairs = 0
    frames_seen = 0
    previous: dict[str, float] | None = None

    for shard_index, (path, _start, size) in enumerate(offsets):
        meta = np.asarray(dataset.meta_of_shard(path), dtype=np.float64)
        if meta.shape[0] != int(size):
            raise v104_error(
                "DATA", f"{getattr(path, 'name', path)}: meta rows {meta.shape[0]} != frames {size}"
            )
        frames_seen += int(size)
        frame, stage, deaths = meta_columns(meta)
        trusted = np.isfinite(frame) & np.isfinite(stage) & np.isfinite(deaths)
        untrusted_frames += int((~trusted).sum())
        starts, _reliable = segment_starts(meta)
        flags = break_flags(meta)

        # Segment statistics.
        unique_starts = np.unique(starts)
        segments_total += int(unique_starts.size)
        if size:
            lengths = np.diff(np.append(unique_starts, size))
            longest_segment = max(longest_segment, int(lengths.max()))

        # Reset counts.  The dataset's first sample starts the stream, not a break.
        counted_shard = int(flags["shard"].sum()) if shard_index > 0 else 0
        internal_breaks = int(unique_starts.size) - 1
        resets["shard"] += counted_shard
        for kind in ("stage", "death", "gap", "untrusted"):
            resets[kind] += int(flags[kind].sum())
        resets["total"] += internal_breaks + counted_shard

        # Measured delta distribution between consecutive samples of a shard.
        if size > 1:
            delta = np.diff(frame)
            delta_pairs += int(delta.size)
            finite = np.isfinite(delta)
            for value, count in zip(*np.unique(delta[finite], return_counts=True)):
                key = _delta_key(float(value))
                delta_histogram[key] = delta_histogram.get(key, 0) + int(count)
            if int((~finite).sum()):
                key = "nonfinite"
                delta_histogram[key] = delta_histogram.get(key, 0) + int((~finite).sum())

        # History slots actually used by this contract at this depth.
        rows = history_rows(np.arange(size, dtype=np.int64), depth_value, contract, starts, trusted)
        if depth_value > 1 and size:
            # ``rows`` is [sample, slot]: a sample repeats history when two of its
            # own slots carry the same frame, i.e. the slot axis is not strictly
            # increasing.
            repeated += int((np.diff(rows, axis=1) <= 0).any(axis=1).sum())
            slot_delta = frame[rows[1:]] - frame[rows[:-1]]
            finite = np.isfinite(slot_delta)
            for value, count in zip(*np.unique(slot_delta[finite], return_counts=True)):
                key = _delta_key(float(value))
                history_delta_histogram[key] = history_delta_histogram.get(key, 0) + int(count)
            if int((~finite).sum()):
                key = "nonfinite"
                history_delta_histogram[key] = history_delta_histogram.get(key, 0) + int(
                    (~finite).sum()
                )

        # Cross-shard diagnostics only: the contract never joins two shards.
        if previous is not None and size:
            if stage[0] != previous["stage"]:
                cross_shard["stage"] += 1
            if deaths[0] != previous["deaths"]:
                cross_shard["death"] += 1
            gap = frame[0] - previous["frame"]
            key = _delta_key(float(gap))
            cross_shard["delta_histogram"][key] = (
                cross_shard["delta_histogram"].get(key, 0) + 1
            )
        if size:
            previous = {
                "frame": float(frame[-1]),
                "stage": float(stage[-1]),
                "deaths": float(deaths[-1]),
            }

    cross_shard["delta_histogram"] = _sorted_histogram(cross_shard["delta_histogram"])
    return {
        "contract_version": AUDIT_VERSION,
        "history_contract": contract,
        "dataset": str(getattr(dataset, "root", "")),
        "frames": total,
        "shards": len(offsets),
        "depth": depth_value,
        "delta_histogram": _sorted_histogram(delta_histogram),
        "delta_pairs": int(delta_pairs),
        "history_delta_histogram": _sorted_histogram(history_delta_histogram),
        "resets": resets,
        "segments": segments_total,
        "longest_segment_frames": longest_segment,
        "untrusted_frames": untrusted_frames,
        "cross_shard": cross_shard,
        "repeated_history_samples": int(repeated),
        "repeated_history_ratio": (float(repeated) / frames_seen) if frames_seen else 0.0,
        "resets_rule": (
            "resets counts break events; one break can satisfy several categories (for example "
            "a stage change that also skips frames sets stage and gap), so sum(resets[kind]) >= "
            "resets.total. The dataset's first sample starts the stream and is not counted."
        ),
        "repeated_history_rule": (
            "a centre sample counts once when any two of its depth slots carry the same frame, "
            "so depth 1 always reports 0 repeated samples"
        ),
        "repeated_history_denominator": "frames",
    }
