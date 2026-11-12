"""WP5 pixel-side death detector (EXP-P2-010): small causal CNN over ``play96``.

Why this module exists
----------------------
EXP-P2-008 closed four *hand-written* pixel criteria (v1..v4) and none of them
reached the G2 gate: the best one (v4, player-patch NCC collapse) scored event
recall 16.41 % / precision 17.43 % on the 908-event stage-3 held-out slice, and
0/9 in the live loop.  The separability probe in the same card showed the signal
is real (death-window NCC p50 0.377 vs non-death p01 0.383) but not threshold
separable, and the card's own conclusion was "must use a learned detector".

This module is that detector.  It is *pixel only*: the input is a short causal
window of the ``play96`` luminance grid (112 rows = height, 96 cols = width,
4x4 block means of the 384x448 playfield).  No HUD plane, no engine ``px``/``py``
label, no engine event bit enters the model; the engine labels are used as
*supervision and scoring only*.

Design
------
* Window: ``t-6 .. t`` (7 frames, causal).  Compressed into 5 channels so the
  convolution stack stays cheap (``CHANNELS``): current luminance, frame
  differences at lags 1, 3 and 6, and a signed motion-collapse contrast.  A
  7-frame stack of raw planes would cost 7x the input convolution for no extra
  temporal reach.
* Head: fully convolutional + ``(mean, max)`` spatial pooling.  The death
  dissolve is a *local* event (the player sprite vanishes for a few frames) on
  an otherwise busy screen, so a channel-wise max has to be available; a pure
  global average would dilute the event by 1/10752.
* Label: frame ``t`` is positive when ``0 <= t - d <= 4`` for a real death
  event ``d`` (the dissolve is visible 0..4 frames after the event, EXP-P2-008
  §14.2).  Frames closer than ``EDGE`` to a death are dropped from training
  instead of being labelled negative, so the model is never taught that the
  onset of a dissolve is "not a death".
* Event rule: rising edge above ``threshold`` confirmed one frame later
  (``EVENT_DELAY``) with a refractory gap.  The confirmation frame is part of
  the reported latency -- it is not a free shift, and it guarantees
  ``predicted >= d`` so the G2 ``match_events(before=0)`` window can accept it.

Everything above ``# --- torch model ---`` is pure numpy and importable without
torch (the Windows unit tests run without torch; training happens in the WSL
venv which has it).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# geometry / window
# ---------------------------------------------------------------------------
#: The play96 grid the collector stores: rows are height, columns are width.
PLAY96_SHAPE = (112, 96)
PLAY64_SHAPE = (64, 64)
#: Coarse trunk grid the auxiliary player-location head predicts over:
#: 112x96 play96 -> /2 /4 /8 == 14x12 cells.
AUX_GRID = (14, 12)

#: Causal window length: frames ``t - (WINDOW - 1) .. t``.
WINDOW = 7
#: Frame lags folded into the channel stack (0 = current frame).
LAGS = (0, 1, 3, 6)
#: Number of input channels produced by :func:`frame_channels`.
CHANNELS = 6

#: Positive frame window after a death event (inclusive), measured in EXP-P2-008:
#: the player sprite is gone 0..4 frames after the engine death bit.
POS_AFTER = 4
#: Frames with ``|t - d| < EDGE`` are dropped from training (ambiguous onset).
EDGE = 9
#: Hard-negative band after the dissolve: respawn blinking looks like a second
#: dissolve.  Kept out of the uniform negative pool so it can be labelled on
#: purpose instead of opportunistically.
RESPAWN_AFTER = (9, 20)

#: One alarm per death at most; the respawn blink is 90 frames in this engine
#: (``GameManager`` death/respawn animation), so a 60-frame refractory can never
#: swallow a second real death.
REFRACTORY = 60
#: Frames the rising edge is confirmed before an alarm is emitted.
EVENT_DELAY = 1
#: Matching window the G2 verdict uses (``match_events(after=...)``).
MATCH_AFTER = 10

INDEX_SCHEMA = "v400-wp5-pixel-index-1"
SPLIT_SCHEMA = "v400-wp5-pixel-split-1"
OFFLINE_SCHEMA = "v400-wp5-pixel-offline-1"
ONLINE_SCHEMA = "v400-wp5-pixel-online-1"
GATE_SCHEMA = "v400-wp5-pixel-gate-1"


class PixelDeathError(RuntimeError):
    """The corpus or a report cannot support a G2 statement (fail closed)."""


# ---------------------------------------------------------------------------
# statistics (delegated to the shared implementation)
# ---------------------------------------------------------------------------
def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """One-sided 95 % Wilson lower bound (0 when ``trials == 0``)."""

    if trials <= 0:
        return 0.0
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = phat + z * z / (2 * trials)
    margin = z * math.sqrt(
        max(0.0, phat * (1.0 - phat) / trials + z * z / (4 * trials * trials))
    )
    return max(0.0, (centre - margin) / denom)


def proportion(successes: int, trials: int) -> dict[str, Any]:
    value = (successes / trials) if trials else 0.0
    lower = wilson_lower_bound(successes, trials)
    return {
        "n": int(trials),
        "successes": int(successes),
        "value": value,
        "lower_bound_95": lower,
        # A Wilson bound cannot exceed 1 - 3/n at zero failures; reports the
        # ceiling so a small-n column is never read as "the detector failed"
        # when the sample size alone made 99 % unreachable.
        "lower_bound_ceiling": (1.0 - 3.0 / trials) if trials else 0.0,
        "pass_99": lower >= 0.99,
    }


def event_starts(flags: Sequence[bool] | Sequence[int]) -> list[int]:
    """Start index of every run of true values."""

    starts: list[int] = []
    previous = False
    for index, value in enumerate(flags):
        value = bool(value)
        if value and not previous:
            starts.append(index)
        previous = value
    return starts


def match_events(
    true_starts: Sequence[int],
    pred_starts: Sequence[int],
    *,
    before: int = 0,
    after: int = MATCH_AFTER,
) -> dict[str, Any]:
    """Greedy one-to-one event matching (same rule as the G2 verdict).

    ``before=0`` by default: a prediction that fires *before* the engine death
    bit does not count, because a gate that accepts early alarms cannot be used
    as a causal trigger.  ``match_events(before=2)`` is reported separately as
    the "±2 frame tolerance" diagnostic.
    """

    used: set[int] = set()
    latencies: list[int] = []
    matched_true = 0
    for true_frame in true_starts:
        best = None
        for index, pred_frame in enumerate(pred_starts):
            if index in used:
                continue
            if true_frame - before <= pred_frame <= true_frame + after:
                best = index
                break
        if best is None:
            continue
        used.add(best)
        matched_true += 1
        latencies.append(int(pred_starts[best]) - int(true_frame))
    return {
        "true_events": len(true_starts),
        "pred_events": len(pred_starts),
        "matched": matched_true,
        "matched_predictions": len(used),
        "latencies": latencies,
        "unmatched_predictions": len(pred_starts) - len(used),
    }


def latency_summary(latencies: Sequence[int], *, limit: int = MATCH_AFTER) -> dict[str, Any]:
    """median / p95 / max latency and whether every detection met the limit."""

    if not latencies:
        return {"n": 0, "median": None, "p95": None, "max": None,
                "within_limit": None, "limit": int(limit), "early_detections": 0}
    ordered = sorted(int(value) for value in latencies)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "n": len(ordered),
        "median": ordered[len(ordered) // 2],
        "p95": p95,
        "max": ordered[-1],
        "within_limit": all(value <= limit for value in ordered),
        "limit": int(limit),
        "early_detections": sum(1 for value in ordered if value < 0),
    }


# ---------------------------------------------------------------------------
# window construction
# ---------------------------------------------------------------------------
def window_indices(t: int, *, window: int = WINDOW, frames: int | None = None) -> list[int]:
    """Causal frame indices of the window ending at ``t`` (clipped, no future)."""

    t = int(t)
    indices = [t - (window - 1) + offset for offset in range(window)]
    if frames is not None:
        if frames <= 0:
            raise PixelDeathError("frames must be positive")
        if t < 0:
            raise PixelDeathError(f"window end {t} is negative")
        indices = [min(max(index, 0), frames - 1) for index in indices]
    return indices


def window_valid(t: int, *, window: int = WINDOW) -> bool:
    """True when the window fits without clipping (``t >= window - 1``)."""

    return int(t) >= window - 1


def stack_window(planes: Any, t: int, *, window: int = WINDOW) -> np.ndarray:
    """``(window, H, W)`` uint8 copy of the causal window ending at ``t``."""

    planes = np.asarray(planes)
    if planes.ndim != 3:
        raise PixelDeathError(f"planes must be (T, H, W), got {planes.shape}")
    frames = int(planes.shape[0])
    indices = window_indices(t, window=window, frames=frames)
    return np.asarray(planes[indices], dtype=np.uint8)


def frame_channels(planes: Any, t: int, *, window: int = WINDOW,
                   channels: int | None = None) -> np.ndarray:
    """``(CHANNELS, H, W)`` float32 feature stack for the window ending at ``t``.

    Channel order (see ``CHANNELS``/``LAGS``)::

        0  current luminance                      I[t]        / 255
        1  one-frame signed difference            (I[t]-I[t-1]) / 64
        2  three-frame signed difference          (I[t-1]-I[t-3]) / 64
        3  six-frame signed difference            (I[t-3]-I[t-6]) / 64
        4  motion-collapse contrast               (|I[t]-I[t-1]| - |I[t-3]-I[t-6]|) / 64
        5  appearance novelty                     (I[t] - mean(I[t-6..t-1])) / 64

    Leg 4 is the EXP-P2-008 signal (``mean |dI|`` collapse inside the player
    cell) expressed as a per-pixel map, so the convolution sees *where* the
    motion died instead of only that it did.  Leg 5 is what a sprite that
    vanishes (or explodes) looks like against its own recent history: it is the
    only leg that is non-zero for a *static* background with a disappearing
    object.

    ``channels`` trims the stack to the leading legs, so a checkpoint trained
    before a leg was added still loads and scores on exactly the legs it saw.
    """

    stack = stack_window(planes, t, window=window).astype(np.float32)
    current = stack[-1]
    prev1 = stack[-2]
    prev3 = stack[-4]
    prev6 = stack[-7]
    diff1 = current - prev1
    diff3 = prev1 - prev3
    diff6 = prev3 - prev6
    collapse = np.abs(diff1) - np.abs(diff3)
    novelty = current - stack[:-1].mean(axis=0)
    out = np.stack(
        [current / 255.0, diff1 / 64.0, diff3 / 64.0, diff6 / 64.0,
         collapse / 64.0, novelty / 64.0]
    ).astype(np.float32)
    if channels is not None:
        if not 1 <= int(channels) <= out.shape[0]:
            raise PixelDeathError(f"channels must be 1..{out.shape[0]}, got {channels}")
        out = out[: int(channels)]
    return out


def batch_channels(planes: Any, ends: Sequence[int], *, window: int = WINDOW,
                   channels: int | None = None) -> np.ndarray:
    """Vectorised :func:`frame_channels` over many window ends: ``(B, C, H, W)``."""

    planes = np.asarray(planes)
    if planes.ndim != 3:
        raise PixelDeathError(f"planes must be (T, H, W), got {planes.shape}")
    frames = int(planes.shape[0])
    ends = np.asarray([int(value) for value in ends], dtype=np.int64)
    if ends.size == 0:
        return np.zeros((0, CHANNELS) + tuple(planes.shape[1:]), dtype=np.float32)
    lag = np.stack(
        [np.clip(ends - offset, 0, frames - 1) for offset in LAGS], axis=0
    )                                                        # (4, B)
    gathered = planes[lag].astype(np.float32)                # (4, B, H, W)
    current, prev1, prev3, prev6 = gathered[0], gathered[1], gathered[2], gathered[3]
    diff1 = current - prev1
    collapse = np.abs(diff1) - np.abs(prev1 - prev3)
    # history mean over t-6 .. t-1 (frames the window already contains)
    history = np.stack(
        [np.clip(ends - offset, 0, frames - 1) for offset in range(1, window)],
        axis=0,
    )
    novelty = current - planes[history].astype(np.float32).mean(axis=0)
    out = np.stack(
        [
            current / 255.0,
            diff1 / 64.0,
            (prev1 - prev3) / 64.0,
            (prev3 - prev6) / 64.0,
            collapse / 64.0,
            novelty / 64.0,
        ],
        axis=1,
    )
    if channels is not None:
        if not 1 <= int(channels) <= out.shape[1]:
            raise PixelDeathError(f"channels must be 1..{out.shape[1]}, got {channels}")
        out = out[:, : int(channels)]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# corpus index
# ---------------------------------------------------------------------------
@dataclass
class Shard:
    """One stored attempt: planes + labels, as the death detector sees it."""

    dir: Path
    root: str
    shard: str
    attempt: str
    stage: int
    frames: int
    deaths: list[int] = field(default_factory=list)
    death_bombs: list[int] = field(default_factory=list)
    death_lives: list[int] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.root}/{self.shard}/{self.attempt}"

    def as_row(self) -> dict[str, Any]:
        return {
            "dir": str(self.dir),
            "root": self.root,
            "shard": self.shard,
            "attempt": self.attempt,
            "stage": int(self.stage),
            "frames": int(self.frames),
            "deaths": list(self.deaths),
            "death_bombs": list(self.death_bombs),
            "death_lives": list(self.death_lives),
        }


def stage_of_shard_name(name: str) -> int:
    match = re.search(r"-s(\d)d", name)
    return int(match.group(1)) if match else -1


def scan_root(root: str | Path) -> list[Shard]:
    """Every ``<root>/shards/<shard>/attempt-*/`` holding ``play96.u8``."""

    root = Path(root)
    out: list[Shard] = []
    for labels_path in sorted(root.glob("shards/*/attempt-*/labels.jsonl")):
        directory = labels_path.parent
        if not (directory / "play96.u8").is_file():
            continue
        deaths: list[int] = []
        bombs: list[int] = []
        lives: list[int] = []
        frames = 0
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index = int(row.get("i", frames))
            frames = max(frames, index + 1)
            if row.get("death"):
                deaths.append(index)
                bombs.append(int(row.get("bo", -1)))
                lives.append(int(row.get("li", -1)))
        if frames == 0:
            continue
        plane_bytes = (directory / "play96.u8").stat().st_size
        readable = plane_bytes // (PLAY96_SHAPE[0] * PLAY96_SHAPE[1])
        out.append(
            Shard(
                dir=directory,
                root=root.name,
                shard=directory.parent.name,
                attempt=directory.name,
                stage=stage_of_shard_name(directory.parent.name),
                frames=min(frames, readable),
                deaths=deaths,
                death_bombs=bombs,
                death_lives=lives,
            )
        )
    return out


def scan_roots(roots: Sequence[str | Path]) -> list[Shard]:
    shards: list[Shard] = []
    for root in roots:
        shards.extend(scan_root(root))
    if not shards:
        raise PixelDeathError(f"no shard under {list(roots)}")
    return shards


def death_events(deaths: Sequence[int], *, merge: int = 5) -> list[int]:
    """Distinct death events from the raw ``death`` flag frames."""

    events: list[int] = []
    for frame in sorted(int(value) for value in deaths):
        if not events or frame - events[-1] > merge:
            events.append(frame)
    return events


def bomb_empty_events(shard: Shard) -> list[int]:
    """Death events with ``bombs == 0`` -- the HUD route's +29-frame blind spot.

    ``GameManager.hpp:56 playerDeathDissolveMode`` picks the fast (+4 frame)
    dissolve while a bomb is stocked and the slow (+29 frame) dissolve once the
    bombs are gone; the HUD lives counter only moves after the dissolve, so the
    HUD route structurally cannot see these inside a 10-frame window.
    """

    events: list[int] = []
    for frame, bombs in zip(shard.deaths, shard.death_bombs):
        if int(bombs) != 0:
            continue
        if not events or frame - events[-1] > 5:
            events.append(int(frame))
    return events


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def shard_stratum(shard: Shard) -> str:
    """Stratum key: root + stage + whether the shard holds a bomb-empty death.

    The bomb-empty shards sit in six roots at most a few dozen runs wide, so a
    plain uniform split can put all of them on one side; stratifying by this
    key keeps a measurable +29 slice in ``val`` and ``test`` without ever
    putting the same run in two splits.
    """

    return f"{shard.root}|s{shard.stage}|{'bo0' if bomb_empty_events(shard) else 'boN'}"


def build_split(
    shards: Sequence[Shard],
    *,
    seed: int = 20261110,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    min_test_events: int = 100,
) -> dict[str, Any]:
    """Split **whole shards** into train/val/test, stratified, leakage-checked."""

    total = sum(fractions)
    if abs(total - 1.0) > 1e-9:
        raise PixelDeathError(f"fractions must sum to 1, got {fractions}")
    if any(value <= 0 for value in fractions):
        raise PixelDeathError(f"fractions must be positive, got {fractions}")
    if not shards:
        raise PixelDeathError("cannot split an empty corpus")
    keys = [shard.key for shard in shards]
    if len(set(keys)) != len(keys):
        raise PixelDeathError("duplicate shard key in corpus")

    buckets: dict[str, list[Shard]] = {}
    for shard in shards:
        buckets.setdefault(shard_stratum(shard), []).append(shard)
    generator = np.random.default_rng(seed)
    assign: dict[str, str] = {}
    for stratum in sorted(buckets):
        group = sorted(buckets[stratum], key=lambda item: item.key)
        order = generator.permutation(len(group))
        ordered = [group[int(index)] for index in order]
        n_train = int(round(fractions[0] * len(ordered)))
        n_val = int(round(fractions[1] * len(ordered)))
        # every stratum keeps at least one run in a non-train split when it can
        if len(ordered) >= 3:
            n_train = min(n_train, len(ordered) - 2)
            n_val = max(1, min(n_val, len(ordered) - n_train - 1))
        elif len(ordered) == 2:
            n_train = 1
            n_val = 0
        else:
            n_train = 1
            n_val = 0
        for position, shard in enumerate(ordered):
            if position < n_train:
                assign[shard.key] = "train"
            elif position < n_train + n_val:
                assign[shard.key] = "val"
            else:
                assign[shard.key] = "test"

    rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for shard in shards:
        split = assign[shard.key]
        previous = seen.get(shard.key)
        if previous is not None and previous != split:
            raise PixelDeathError(f"shard {shard.key} leaks across splits")
        seen[shard.key] = split
        rows.append(
            {
                **shard.as_row(),
                "split": split,
                "bomb_empty_events": bomb_empty_events(shard),
                "events": death_events(shard.deaths),
            }
        )

    summary: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        summary[split] = {
            "shards": len(selected),
            "frames": sum(row["frames"] for row in selected),
            "events": sum(len(row["events"]) for row in selected),
            "bomb_empty_events": sum(len(row["bomb_empty_events"]) for row in selected),
            "roots": sorted({row["root"] for row in selected}),
            "stages": sorted({row["stage"] for row in selected}),
        }
    if summary["test"]["events"] < min_test_events:
        raise PixelDeathError(
            f"test split holds {summary['test']['events']} death events "
            f"< min_test_events={min_test_events}"
        )
    if not summary["train"]["events"]:
        raise PixelDeathError("train split holds no death event")
    # Leakage machine check: a stored attempt is one run and may appear in one
    # split only.  A root or stage may span splits -- that is domain overlap
    # between *different runs*, which is what "split by game" means; the
    # per-run assertion below is the one the G2 table relies on.
    by_key: dict[str, set[str]] = {}
    for row in rows:
        key = f"{row['root']}/{row['shard']}/{row['attempt']}"
        by_key.setdefault(key, set()).add(row["split"])
    leaked = sorted(key for key, splits in by_key.items() if len(splits) > 1)
    if leaked:
        raise PixelDeathError(f"{len(leaked)} shards appear in more than one split")
    return {
        "schema": SPLIT_SCHEMA,
        "machine": "whole shard (one stored attempt = one run); stratified by "
                   "root|stage|bomb-empty",
        "seed": int(seed),
        "fractions": list(fractions),
        "shards": rows,
        "summary": summary,
    }


def split_rows(split: dict[str, Any], name: str) -> list[dict[str, Any]]:
    if name not in ("train", "val", "test"):
        raise PixelDeathError(f"unknown split {name!r}")
    rows = [row for row in split.get("shards") or [] if row.get("split") == name]
    if not rows:
        raise PixelDeathError(f"split {name!r} holds no shard")
    return rows


# ---------------------------------------------------------------------------
# training samples
# ---------------------------------------------------------------------------
def frame_labels(
    frames: int,
    events: Sequence[int],
    *,
    pos_after: int = POS_AFTER,
    edge: int = EDGE,
    slow_events: Sequence[int] = (),
    slow_after: int = 29,
) -> tuple[np.ndarray, np.ndarray]:
    """``(label, usable)`` for every frame of one shard.

    ``label[i] = 1`` for ``0 <= i - d <= pos_after``.  The bands whose class is
    genuinely *unknown* are marked unusable instead of negative:

    * ``[d - edge, d - 1]`` -- pre-onset; calling it negative would teach the
      model that a death is "not a death" two frames early, and a pre-event fire
      cannot pass the causal gate anyway.
    * ``[d + pos_after + 1, d + 8]`` -- end of the dissolve, class unknown.
    * ``[d + pos_after + 1, d + 29]`` for a bomb-empty (``+29``) event, whose
      dissolve is the long one (``GameManager.hpp:56``); the end of *that*
      animation was not measured, so the whole tail stays out of the loss.

    Frames after that (respawn blink, ``RESPAWN_AFTER``) are legitimate hard
    negatives and stay usable.
    """

    frames = int(frames)
    label = np.zeros(frames, dtype=np.int8)
    usable = np.ones(frames, dtype=bool)
    slow = {int(value) for value in slow_events}
    for event in sorted(int(value) for value in events):
        lo = max(0, event)
        hi = min(frames - 1, event + pos_after)
        if hi >= lo:
            label[lo : hi + 1] = 1
        tail = slow_after if event in slow else 8
        for start, stop in ((event - edge, event - 1), (event + pos_after + 1, event + tail)):
            start = max(0, start)
            stop = min(frames - 1, stop)
            if stop >= start:
                usable[start : stop + 1] = False
    usable[:WINDOW] = False          # causal window must exist unclipped
    usable[label.astype(bool)] = True  # a positive frame is never unusable
    return label, usable


def positive_sample_frames(
    frames: int, events: Sequence[int], *, pos_after: int = POS_AFTER
) -> list[int]:
    """Every labelled-positive frame of one shard (deduplicated, ordered)."""

    out: set[int] = set()
    for event in events:
        for offset in range(0, pos_after + 1):
            frame = int(event) + offset
            if 0 <= frame < frames and window_valid(frame):
                out.add(frame)
    return sorted(out)


def build_samples(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    negatives_per_positive: float = 2.0,
    respawn_negatives: bool = True,
    respawn_share: float = 0.5,
    jitter: int = 0,
) -> dict[str, Any]:
    """Deterministic training samples: all positives + sampled negatives.

    ``jitter`` (train only) shifts a positive window end by ``Uniform(-j, +j)``
    so the model cannot key on the exact onset frame; the label stays 1 inside
    the ``±2`` tolerance the task asks for.  Negatives are drawn from (a) the
    respawn band right after a dissolve and (b) frames elsewhere.

    ``respawn_share`` splits the negative budget between the two pools.  The
    shipped 2026-11-10 checkpoint was trained before this parameter existed:
    ``keep_respawn`` was taken first up to the *whole* budget, so the model saw
    25,291 respawn frames and only **57** ordinary gameplay frames — it learned
    "respawn looks like death" and fired on ordinary play (EXP-P2-011 §3).  Both
    pools are now sampled uniformly at random, never "the first N".
    """

    generator = np.random.default_rng(seed)
    positives: list[tuple[int, int]] = []
    respawn: list[tuple[int, int]] = []
    pool: list[tuple[int, int]] = []
    for shard_index, row in enumerate(rows):
        frames = int(row["frames"])
        events = [int(value) for value in row.get("events") or []]
        slow = [int(value) for value in row.get("bomb_empty_events") or []]
        label, usable = frame_labels(frames, events, slow_events=slow)
        for event in sorted(events):
            # jitter is clipped to the event's own positive window: an end before
            # the event would train a pre-event fire (the causal gate rejects it)
            # and an end past `d + pos_after` would label the dissolve tail.
            for offset in range(0, POS_AFTER + 1):
                base = event + offset
                for shift in range(-jitter, jitter + 1):
                    end = min(event + POS_AFTER, max(event, base + shift))
                    if window_valid(end) and 0 <= end < frames:
                        positives.append((shard_index, end))
        if respawn_negatives:
            for event in sorted(events):
                lo = event + RESPAWN_AFTER[0]
                hi = min(frames - 1, event + RESPAWN_AFTER[1])
                for frame in range(max(lo, WINDOW), hi + 1):
                    respawn.append((shard_index, frame))
        for frame in range(WINDOW, frames):
            if usable[frame] and not label[frame]:
                pool.append((shard_index, frame))
    dedup_positives = sorted(set(positives))
    budget = int(negatives_per_positive * max(1, len(dedup_positives)))
    respawn_unique = sorted(set(respawn) - set(dedup_positives))
    share = max(0.0, min(1.0, float(respawn_share)))
    respawn_budget = min(len(respawn_unique), int(round(budget * share)))
    if respawn_budget < len(respawn_unique):
        picks = generator.choice(len(respawn_unique), size=respawn_budget, replace=False)
        keep_respawn = [respawn_unique[int(index)] for index in sorted(picks)]
    else:
        keep_respawn = respawn_unique
    remaining = max(0, budget - len(keep_respawn))
    pool_set = set(pool) - set(dedup_positives) - set(keep_respawn)
    pool_list = sorted(pool_set)
    if len(pool_list) > remaining:
        picks = generator.choice(len(pool_list), size=remaining, replace=False)
        keep_uniform = [pool_list[int(index)] for index in sorted(picks)]
    else:
        keep_uniform = pool_list
    chosen = dedup_positives + keep_respawn + keep_uniform
    labels = [1] * len(dedup_positives) + [0] * (len(keep_respawn) + len(keep_uniform))
    # Sorting has to carry the label with the sample: sorting only ``chosen``
    # would leave ``labels`` in construction order and hand the trainer a
    # mislabelled corpus (this is the bug EXP-P2-010 §2.3 records).
    paired = sorted(zip(chosen, labels))
    chosen = [item[0] for item in paired]
    labels = [item[1] for item in paired]
    return {
        "samples": chosen,
        "labels": labels,
        "positives": len(dedup_positives),
        "respawn_negatives": len(keep_respawn),
        "uniform_negatives": len(keep_uniform),
        "usable_frames": len(pool),
        "jitter": int(jitter),
    }


def sample_arrays(
    rows: Sequence[dict[str, Any]],
    samples: Sequence[tuple[int, int]],
    *,
    resolution: str = "play96",
) -> np.ndarray:
    """``(N, C, H, W)`` float32 features for the given (shard, frame) samples."""

    by_shard: dict[int, list[int]] = {}
    for position, (shard_index, frame) in enumerate(samples):
        by_shard.setdefault(int(shard_index), []).append(int(frame))
    shape = PLAY96_SHAPE if resolution == "play96" else PLAY64_SHAPE
    out = np.zeros((len(samples), CHANNELS) + shape, dtype=np.float32)
    cursor_of: dict[int, int] = {}
    for shard_index, frames in by_shard.items():
        directory = Path(rows[shard_index]["dir"])
        total = int(rows[shard_index]["frames"])
        planes = np.memmap(directory / f"{resolution}.u8", dtype=np.uint8, mode="r",
                           shape=(total,) + shape)
        block = batch_channels(planes, frames)
        for offset, frame in enumerate(frames):
            cursor_of[(shard_index, frame)] = offset
        for position, (index, frame) in enumerate(samples):
            if int(index) != int(shard_index):
                continue
            out[position] = block[cursor_of[(shard_index, int(frame))]]
        del planes
    return out


# ---------------------------------------------------------------------------
# alarms
# ---------------------------------------------------------------------------
class AlarmRule:
    """Streaming rising-edge alarm rule -- the *one* rule offline and online use.

    ``push`` takes one probability and returns the alarm frame (``int``) or
    ``None``.  ``predict_events`` is a thin loop over this class, so an offline
    replay and the live loop cannot drift apart.
    """

    def __init__(
        self,
        *,
        threshold: float,
        refractory: int = REFRACTORY,
        delay: int = EVENT_DELAY,
        prominence: float = 0.0,
        lookback: int = 4,
    ) -> None:
        if not 0.0 <= float(threshold) <= 1.0:
            raise PixelDeathError(f"threshold {threshold} outside [0, 1]")
        if refractory < 0 or delay < 0:
            raise PixelDeathError("refractory/delay must be non-negative")
        if prominence < 0.0 or lookback < 1:
            raise PixelDeathError("prominence must be >= 0 and lookback >= 1")
        self.threshold = float(threshold)
        self.refractory = int(refractory)
        self.delay = int(delay)
        self.prominence = float(prominence)
        self.lookback = int(lookback)
        self.reset()

    def reset(self) -> None:
        self.index = -1
        self.history: list[float] = []
        self.last_alarm = -10 ** 9

    def push(self, prob: float) -> int | None:
        prob = float(prob)
        self.index += 1
        index = self.index
        self.history.append(prob)
        if len(self.history) > self.lookback + 1:
            self.history.pop(0)
        if index < max(1, self.lookback):
            return None
        if prob < self.threshold:
            return None
        previous = self.history[-2]
        if previous >= self.threshold:
            return None
        if self.prominence:
            floor = min(self.history[:-1][-self.lookback:])
            if prob - floor < self.prominence:
                return None
        frame = index + self.delay
        if frame - self.last_alarm <= self.refractory:
            return None
        self.last_alarm = frame
        return int(frame)


def predict_events(
    probs: Sequence[float],
    *,
    threshold: float,
    refractory: int = REFRACTORY,
    delay: int = EVENT_DELAY,
    min_index: int = 1,
    prominence: float = 0.0,
    lookback: int = 4,
) -> list[int]:
    """Rising-edge alarms from a per-frame probability series.

    The alarm frame is ``rise + delay``: one frame later the rise has been
    observed for two frames, which is what an online trigger can honestly
    claim.  ``delay`` is *added to the latency*, never subtracted, so a
    prediction cannot be moved in front of the engine event by this rule.

    ``prominence`` requires the crossing to clear the recent floor by that
    margin.  A soft ramp that hovers around the threshold would otherwise
    produce a crossing per wobble, which is the dominant false-alarm source on
    a busy screen.
    """

    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise PixelDeathError(f"probabilities must be 1-D, got {probs.shape}")
    rule = AlarmRule(threshold=threshold, refractory=refractory, delay=delay,
                     prominence=prominence, lookback=lookback)
    alarms: list[int] = []
    for prob in probs:
        frame = rule.push(prob)
        if frame is not None and frame < len(probs) and frame >= int(min_index):
            alarms.append(frame)
    return alarms


def event_metrics(
    events: Sequence[int],
    alarms: Sequence[int],
    *,
    before: int = 0,
    after: int = MATCH_AFTER,
) -> dict[str, Any]:
    """Event-level recall/precision with Wilson bounds and the latency table."""

    match = match_events(events, alarms, before=before, after=after)
    recall = proportion(match["matched"], match["true_events"])
    precision = proportion(match["matched_predictions"], match["pred_events"])
    return {
        "recall": recall,
        "precision": precision,
        "latency": latency_summary(match["latencies"], limit=after),
        "matched": match["matched"],
        "matched_predictions": match["matched_predictions"],
        "true_events": match["true_events"],
        "pred_events": match["pred_events"],
        "unmatched_predictions": match["unmatched_predictions"],
    }


def frame_metrics(
    label: Sequence[int], fired: Sequence[int], alarms: Sequence[int]
) -> dict[str, Any]:
    """Frame-level precision/recall over the alarm frames only.

    "Frame level" here means: was the alarm frame inside a labelled positive
    window (``0..POS_AFTER`` after a death)?  A per-frame confusion over all
    1.4 M frames would be dominated by the 99.7 % negative class and would say
    nothing about a detector that only emits ~2.6 k alarms.
    """

    label = np.asarray(label, dtype=bool)
    fired = np.asarray(fired, dtype=bool)
    alarms = list(int(value) for value in alarms)
    tp = sum(1 for frame in alarms if label[frame])
    fp = len(alarms) - tp
    fn = int(label.sum()) - tp if label.size else 0
    return {
        "tp": tp,
        "fp": fp,
        "fn": int(fn),
        "positive_frames": int(label.sum()),
        "precision": proportion(tp, tp + fp),
        "recall": proportion(tp, tp + fn) if (tp + fn) > 0 else proportion(0, 0),
        "alarm_frames": len(alarms),
    }


def subset_events(
    rows: Sequence[dict[str, Any]], *, bombs: str = "empty"
) -> list[tuple[int, int]]:
    """``(shard_index, event_frame)`` for bomb-empty (or bomb-stocked) events."""

    if bombs not in ("empty", "stocked"):
        raise PixelDeathError(f"unknown bombs selector {bombs!r}")
    out: list[tuple[int, int]] = []
    for shard_index, row in enumerate(rows):
        empty = {int(value) for value in row.get("bomb_empty_events") or []}
        for event in (int(value) for value in row.get("events") or []):
            keep = (event in empty) if bombs == "empty" else (event not in empty)
            if keep:
                out.append((shard_index, event))
    return out


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
def online_gate(
    payload: dict[str, Any],
    *,
    min_episodes: int = 5,
    min_frames: int = 2400,
    max_latency: int = MATCH_AFTER,
    min_recall_lb: float = 0.99,
    min_precision_lb: float = 0.99,
) -> dict[str, Any]:
    """Fail-closed verdict for the live loop (schema first, then the numbers).

    Every rule that can invalidate the measurement raises/records a failure:
    a missing episode, a short episode, an episode without any death event, an
    unknown schema, a latency worse than ``max_latency``, and both Wilson lower
    bounds.  A payload that does not satisfy the *shape* cannot be signed.
    """

    failures: list[str] = []
    if not isinstance(payload, dict):
        raise PixelDeathError("online payload must be an object")
    if payload.get("schema") != ONLINE_SCHEMA:
        raise PixelDeathError(
            f"online schema {payload.get('schema')!r} != {ONLINE_SCHEMA!r}"
        )
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise PixelDeathError("online payload carries no episode")
    deaths = 0
    alarms = 0
    frames = 0
    latencies: list[int] = []
    matched = 0
    for index, episode in enumerate(episodes):
        for key in ("frames", "deathEvents", "alarms", "latencies", "inferenceMs"):
            if key not in episode:
                raise PixelDeathError(f"episode {index} misses {key!r}")
        count = int(episode["frames"])
        if count < min_frames:
            failures.append(f"episode {index}: {count} frames < {min_frames}")
        if not episode["deathEvents"]:
            failures.append(f"episode {index}: no death event")
        frames += count
        deaths += len(episode["deathEvents"])
        alarms += len(episode["alarms"])
        latencies.extend(int(value) for value in episode["latencies"])
        matched += len(episode["latencies"])
    if len(episodes) < min_episodes:
        failures.append(f"{len(episodes)} episodes < {min_episodes}")
    recall = proportion(matched, deaths)
    precision = proportion(matched, alarms)
    latency = latency_summary(latencies, limit=max_latency)
    if latency["n"] and not latency["within_limit"]:
        failures.append(f"latency max {latency['max']} > {max_latency}")
    if latency["n"] and latency["early_detections"]:
        failures.append(f"{latency['early_detections']} alarm(s) fired before the event")
    if recall["lower_bound_95"] < min_recall_lb:
        failures.append(
            f"online recall LB {recall['lower_bound_95']:.4f} < {min_recall_lb} (n={deaths})"
        )
    if precision["lower_bound_95"] < min_precision_lb:
        failures.append(
            f"online precision LB {precision['lower_bound_95']:.4f} < {min_precision_lb} "
            f"(n={alarms})"
        )
    inference = [
        float(episode["inferenceMs"]) for episode in episodes
    ]
    return {
        "schema": GATE_SCHEMA,
        "episodes": len(episodes),
        "frames": frames,
        "deaths": deaths,
        "alarms": alarms,
        "matched": matched,
        "recall": recall,
        "precision": precision,
        "latency": latency,
        "inference_ms_mean": float(np.mean(inference)) if inference else None,
        "passed": not failures,
        "failures": failures,
        "limits": {
            "episodes": min_episodes,
            "frames_per_episode": min_frames,
            "latency_max": max_latency,
            "recall_lower_bound": min_recall_lb,
            "precision_lower_bound": min_precision_lb,
        },
    }


# ---------------------------------------------------------------------------
# offline report container
# ---------------------------------------------------------------------------
def offline_report(
    *,
    split: dict[str, Any],
    checkpoint: dict[str, Any],
    test: dict[str, Any],
    val: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the offline artifact the G2 section quotes."""

    if test.get("recall", {}).get("n", 0) <= 0:
        raise PixelDeathError("offline test report holds no event")
    return {
        "schema": OFFLINE_SCHEMA,
        "split": {
            "schema": split.get("schema"),
            "machine": split.get("machine"),
            "seed": split.get("seed"),
            "summary": split.get("summary"),
        },
        "checkpoint": checkpoint,
        "test": test,
        "val": val,
        **(extra or {}),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_ROOTS = (
    "/root/v400-data/p2det/out-main",
    "/root/v400-data/p2det/g2test",
    "/root/v400-data/p2det/g2test2",
    "/root/v400-data/p2det/g2c-go2",
    "/root/v400-data/p2det/g2c-go3",
    "/root/v400-data/p2det/g2c-go4",
    "/root/v400-data/p2det/g2c-go",
    "/root/v400-data/p2det/g2go",
    "/root/v400-data/p2det/g2c-bomb",
    "/root/v400-data/p2det/g2c-bomb2",
    "/root/v400-data/p2det/g2c-bomb-cover",
    "/root/v400-data/p2det/g2c-bomb-pilot",
    "/root/v400-data/p2det/g2c-bomb-pilot2",
    "/root/v400-data/p2det/out-probe-dive",
)


def index_command(args) -> int:
    roots = args.root or list(DEFAULT_ROOTS)
    shards = scan_roots(roots)
    payload = {
        "schema": INDEX_SCHEMA,
        "roots": [str(root) for root in roots],
        "shards": [shard.as_row() for shard in shards],
        "totals": {
            "shards": len(shards),
            "frames": sum(shard.frames for shard in shards),
            "events": sum(len(death_events(shard.deaths)) for shard in shards),
            "bomb_empty_events": sum(len(bomb_empty_events(shard)) for shard in shards),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload["totals"], indent=1))
    return 0


def _load_index(path: str | Path) -> list[Shard]:
    path = Path(path)
    if not path.is_file():
        raise PixelDeathError(f"index not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != INDEX_SCHEMA:
        raise PixelDeathError(f"index schema {payload.get('schema')!r} != {INDEX_SCHEMA!r}")
    shards = []
    for row in payload.get("shards") or []:
        shards.append(
            Shard(
                dir=Path(row["dir"]),
                root=str(row["root"]),
                shard=str(row["shard"]),
                attempt=str(row["attempt"]),
                stage=int(row["stage"]),
                frames=int(row["frames"]),
                deaths=[int(value) for value in row.get("deaths") or []],
                death_bombs=[int(value) for value in row.get("death_bombs") or []],
                death_lives=[int(value) for value in row.get("death_lives") or []],
            )
        )
    if not shards:
        raise PixelDeathError(f"index {path} holds no shard")
    return shards


def split_command(args) -> int:
    shards = _load_index(args.index)
    fractions = (float(args.train), float(args.val), float(args.test))
    payload = build_split(
        shards, seed=int(args.seed), fractions=fractions,
        min_test_events=int(args.min_test_events),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="scan the stored shards")
    index.add_argument("--root", action="append", default=None)
    index.add_argument("--out", required=True)
    index.set_defaults(func=index_command)

    split = sub.add_parser("split", help="whole-shard train/val/test split")
    split.add_argument("--index", required=True)
    split.add_argument("--out", required=True)
    split.add_argument("--seed", type=int, default=20261110)
    split.add_argument("--train", type=float, default=0.70)
    split.add_argument("--val", type=float, default=0.15)
    split.add_argument("--test", type=float, default=0.15)
    split.add_argument("--min-test-events", type=int, default=100)
    split.set_defaults(func=split_command)

    train = sub.add_parser("train", help="fit the pixel death CNN (needs torch)")
    train.add_argument("--index", required=True)
    train.add_argument("--split", required=True)
    train.add_argument("--out", required=True, help="checkpoint path (.pt)")
    train.add_argument("--report", default=None, help="training report (.json)")
    train.add_argument("--epochs", type=int, default=14)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=3e-3)
    train.add_argument("--width", type=int, default=32)
    train.add_argument("--aux-cells", type=int, default=0,
                       help="auxiliary player-location head (168 = 14x12 grid, 0 = off)")
    train.add_argument("--aux-weight", type=float, default=0.5)
    train.add_argument("--negatives-per-positive", type=float, default=6.0)
    train.add_argument("--respawn-share", type=float, default=0.5,
                       help="fraction of the negative budget drawn from the respawn band; "
                            "the rest is ordinary gameplay frames (see build_samples)")
    train.add_argument("--pos-weight", type=float, default=1.0,
                       help="loss re-weighting on top of the sampling ratio (1.0 = none)")
    train.add_argument("--channels", type=int, default=CHANNELS,
                       help="leading feature legs to train on (a checkpoint records its own)")
    train.add_argument("--jitter", type=int, default=2)
    train.add_argument("--seed", type=int, default=20261110)
    train.add_argument("--device", default="cuda")
    train.add_argument("--max-minutes", type=float, default=25.0)
    train.set_defaults(func=train_command)

    evaluate = sub.add_parser("eval", help="offline event/frame metrics on val+test")
    evaluate.add_argument("--index", required=True)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--splits", default="val,test")
    evaluate.add_argument("--thresholds", default=None,
                          help="comma list; default = sweep chosen on val")
    evaluate.add_argument("--prominence", type=float, default=0.0)
    evaluate.set_defaults(func=eval_command)

    bench = sub.add_parser("bench", help="CPU single-frame latency / throughput")
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--frames", type=int, default=200)
    bench.add_argument("--threads", type=int, default=0)
    bench.add_argument("--out", default=None)
    bench.set_defaults(func=bench_command)

    gate = sub.add_parser("gate", help="offline + online fail-closed verdict")
    gate.add_argument("--offline", required=True)
    gate.add_argument("--online", default=None)
    gate.add_argument("--out", required=True)
    gate.add_argument("--min-episodes", type=int, default=5)
    gate.add_argument("--min-frames", type=int, default=2400)
    gate.set_defaults(func=gate_command)

    args = parser.parse_args(argv)
    return int(args.func(args))


# ---------------------------------------------------------------------------
# torch side: model, training, inference, benchmarking
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.5


def build_model(*, width: int = 32, channels: int = CHANNELS, aux_cells: int = 0):
    """~0.1 M parameter fully-convolutional net with (mean, max) spatial pooling.

    `aux_cells > 0` adds an auxiliary player-location head: a 1x1 convolution
    that predicts which coarse cell of the 14x12 trunk grid holds the player.
    The labels come from the engine `px`/`py` *supervision* only -- nothing
    is read at inference -- and the head exists to force the shared trunk to
    keep *where* the player is, which is the information a purely pooled
    classifier throws away.  The head is dropped at inference (`aux=False`),
    so the deployed model is identical in cost to the aux-free one.
    """

    import torch
    from torch import nn

    class PixelDeathNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(channels, width, 3, stride=2, padding=1),      # /2
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width * 3 // 2, 3, stride=2, padding=1),  # /4
                nn.BatchNorm2d(width * 3 // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(width * 3 // 2, width * 2, 3, stride=2, padding=1),  # /8
                nn.BatchNorm2d(width * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(width * 2, width * 2, 3, stride=1, padding=1),
                nn.BatchNorm2d(width * 2),
                nn.ReLU(inplace=True),
            )
            features = width * 2 * 2                       # mean + max
            self.head = nn.Sequential(
                nn.Linear(features, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.aux = nn.Conv2d(width * 2, aux_cells, 1) if aux_cells else None
            self.aux_cells = int(aux_cells)

        def forward(self, x, *, aux: bool = False):
            body = self.body(x)
            average = self.pool(body).flatten(1)
            peak = body.amax(dim=(2, 3))
            logits = self.head(torch.cat([average, peak], dim=1)).squeeze(1)
            if aux and self.aux is not None:
                return logits, self.aux(body).flatten(2)
            return logits

    return PixelDeathNet()


def model_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _torch_device(prefer: str = "cuda"):
    import torch

    if prefer == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


class _PlaneCache:
    """LRU of open ``play96`` memmaps (one handle per stored run)."""

    def __init__(self, rows: Sequence[dict[str, Any]], *, resolution: str = "play96",
                 capacity: int = 96) -> None:
        self.rows = rows
        self.shape = PLAY96_SHAPE if resolution == "play96" else PLAY64_SHAPE
        self.resolution = resolution
        self.capacity = capacity
        self.handles: dict[int, Any] = {}
        self.order: list[int] = []

    def planes(self, shard_index: int):
        if shard_index in self.handles:
            self.order.remove(shard_index)
            self.order.append(shard_index)
            return self.handles[shard_index]
        row = self.rows[shard_index]
        handle = np.memmap(Path(row["dir"]) / f"{self.resolution}.u8", dtype=np.uint8,
                           mode="r", shape=(int(row["frames"]),) + self.shape)
        self.handles[shard_index] = handle
        self.order.append(shard_index)
        while len(self.order) > self.capacity:
            drop = self.order.pop(0)
            self.handles.pop(drop, None)
        return handle


def _batch_features(cache: _PlaneCache, samples: Sequence[tuple[int, int]],
                    *, channels: int | None = None) -> np.ndarray:
    """``(B, C, H, W)`` features for samples grouped by shard."""

    by_shard: dict[int, list[tuple[int, int]]] = {}
    for position, (shard_index, frame) in enumerate(samples):
        by_shard.setdefault(int(shard_index), []).append((position, int(frame)))
    out = None
    for shard_index, entries in by_shard.items():
        planes = cache.planes(shard_index)
        block = batch_channels(planes, [frame for _, frame in entries], channels=channels)
        if out is None:
            out = np.zeros((len(samples),) + block.shape[1:], dtype=np.float32)
        for offset, (position, _) in enumerate(entries):
            out[position] = block[offset]
    if out is None:
        raise PixelDeathError("empty batch")
    return out


class _PositionCache:
    """Player cell index per frame, parsed from `labels.jsonl` (supervision only).

    The auxiliary head learns *where* the player is; the deployed model never
    reads `px`/`py` -- this cache exists only while training.
    """

    def __init__(self, rows: Sequence[dict[str, Any]], *, capacity: int = 96) -> None:
        self.rows = rows
        self.capacity = capacity
        self.cache: dict[int, np.ndarray] = {}
        self.order: list[int] = []

    def cells(self, shard_index: int) -> np.ndarray:
        if shard_index in self.cache:
            self.order.remove(shard_index)
            self.order.append(shard_index)
            return self.cache[shard_index]
        row = self.rows[shard_index]
        frames = int(row["frames"])
        out = np.full(frames, -1, dtype=np.int16)
        path = Path(row["dir"]) / "labels.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                index = int(entry.get("i", -1))
                if not 0 <= index < frames:
                    continue
                px = float(entry.get("px", -1.0))
                py = float(entry.get("py", -1.0))
                if px < 0.0 or py < 0.0:
                    continue
                grid_row = min(AUX_GRID[0] - 1, max(0, int(py) // 32))
                grid_col = min(AUX_GRID[1] - 1, max(0, int(px) // 32))
                out[index] = grid_row * AUX_GRID[1] + grid_col
        self.cache[shard_index] = out
        self.order.append(shard_index)
        while len(self.order) > self.capacity:
            drop = self.order.pop(0)
            self.cache.pop(drop, None)
        return out


def _batch_cells(cache: "_PositionCache", samples: Sequence[tuple[int, int]]) -> np.ndarray:
    out = np.full(len(samples), -100, dtype=np.int64)
    for position, (shard_index, frame) in enumerate(samples):
        cells = cache.cells(int(shard_index))
        if 0 <= int(frame) < cells.size:
            out[position] = int(cells[int(frame)])
    return out


def _scores(model, features: np.ndarray, *, device, batch_size: int = 256) -> np.ndarray:
    import torch

    model.eval()
    out = np.empty(len(features), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            chunk = torch.from_numpy(features[start : start + batch_size]).to(device)
            with torch.autocast(device_type=device.type, enabled=False):
                out[start : start + len(chunk)] = (
                    model(chunk).float().sigmoid().cpu().numpy()
                )
    return out


def _mmap_planes(row: dict[str, Any], *, resolution: str = "play96"):
    shape = PLAY96_SHAPE if resolution == "play96" else PLAY64_SHAPE
    return np.memmap(Path(row["dir"]) / f"{resolution}.u8", dtype=np.uint8, mode="r",
                     shape=(int(row["frames"]),) + shape)


def shard_probabilities(model, row: dict[str, Any], *, device, resolution: str = "play96",
                        batch_size: int = 256, channels: int | None = None) -> np.ndarray:
    """Per-frame alarm probability for one stored run (rows ``< WINDOW-1`` = 0)."""

    frames = int(row["frames"])
    planes = _mmap_planes(row, resolution=resolution)
    probs = np.zeros(frames, dtype=np.float32)
    ends = np.arange(WINDOW - 1, frames)
    if ends.size == 0:
        return probs
    for start in range(0, ends.size, batch_size):
        chunk = ends[start : start + batch_size]
        features = batch_channels(planes, chunk, channels=channels)
        probs[chunk] = _scores(model, features, device=device, batch_size=batch_size)
    del planes
    return probs


def _select_threshold(
    model,
    rows: Sequence[dict[str, Any]],
    *,
    device,
    thresholds: Iterable[float],
    resolution: str,
    prominences: Iterable[float] = (0.0, 0.2, 0.4),
    channels: int | None = None,
) -> tuple[tuple[float, float], list[dict[str, Any]]]:
    """Sweep (threshold, prominence) on the **val** split, pick the best joint LB."""

    prob_map = {row["dir"]: shard_probabilities(model, row, device=device,
                                                resolution=resolution,
                                                channels=channels) for row in rows}
    table: list[dict[str, Any]] = []
    for threshold in thresholds:
        for prominence in prominences:
            events = alarms = matched = 0
            latencies: list[int] = []
            for row in rows:
                probs = prob_map[row["dir"]]
                found = predict_events(probs, threshold=float(threshold),
                                       prominence=float(prominence))
                stats = match_events([int(v) for v in row.get("events") or []], found)
                events += stats["true_events"]
                alarms += stats["pred_events"]
                matched += stats["matched"]
                latencies.extend(stats["latencies"])
            recall = proportion(matched, events)
            precision = proportion(matched, alarms)
            table.append(
                {
                    "threshold": float(threshold),
                    "prominence": float(prominence),
                    "events": events,
                    "alarms": alarms,
                    "matched": matched,
                    "recall": recall,
                    "precision": precision,
                    "latency": latency_summary(latencies),
                    "score": min(recall["lower_bound_95"], precision["lower_bound_95"]),
                }
            )
    best = max(table, key=lambda row: (row["score"], row["recall"]["value"]))
    return (float(best["threshold"]), float(best["prominence"])), table


def _evaluate_split(
    model,
    rows: Sequence[dict[str, Any]],
    *,
    device,
    threshold: float,
    prominence: float = 0.0,
    resolution: str = "play96",
    channels: int | None = None,
) -> dict[str, Any]:
    """Event + frame metrics for one split, including the bomb-empty slice."""

    events = alarms = matched = 0
    latencies: list[int] = []
    matched_before2 = 0
    frame_tp = frame_fp = frame_fn = 0
    bomb_empty = {"events": 0, "alarms": 0, "matched": 0, "latencies": []}
    bomb_stocked = {"events": 0, "matched": 0}
    per_shard: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for row in rows:
        probs = shard_probabilities(model, row, device=device, resolution=resolution,
                                    channels=channels)
        found = predict_events(probs, threshold=float(threshold),
                               prominence=float(prominence))
        row_events = [int(value) for value in row.get("events") or []]
        stats = match_events(row_events, found)
        stats_tol = match_events(row_events, found, before=2)
        events += stats["true_events"]
        alarms += stats["pred_events"]
        matched += stats["matched"]
        matched_before2 += stats_tol["matched"]
        latencies.extend(stats["latencies"])
        # frame level: alarm frame inside a labelled positive window?
        label, _usable = frame_labels(int(row["frames"]), row_events,
                                     slow_events=row.get("bomb_empty_events") or [])
        tp = int(sum(1 for frame in found if label[frame]))
        frame_tp += tp
        frame_fp += len(found) - tp
        frame_fn += int(label.sum()) - tp
        empty = {int(value) for value in row.get("bomb_empty_events") or []}
        empty_events = [event for event in row_events if event in empty]
        stocked_events = [event for event in row_events if event not in empty]
        empty_stats = match_events(empty_events, found)
        bomb_empty["events"] += empty_stats["true_events"]
        bomb_empty["alarms"] += empty_stats["pred_events"]
        bomb_empty["matched"] += empty_stats["matched"]
        bomb_empty["latencies"].extend(empty_stats["latencies"])
        bomb_stocked["events"] += len(stocked_events)
        bomb_stocked["matched"] += match_events(stocked_events, found)["matched"]
        event_rows.append(
            {
                "dir": row["dir"],
                "root": row.get("root"),
                "stage": row.get("stage"),
                "shard": row.get("shard"),
                "attempt": row.get("attempt"),
                "events": row_events,
                "bombEmptyEvents": sorted(empty),
                "alarms": found,
                "matched": stats["matched"],
                "latencies": stats["latencies"],
            }
        )
        per_shard.append(
            {
                "dir": row["dir"],
                "events": len(row_events),
                "alarms": len(found),
                "matched": stats["matched"],
            }
        )
    recall = proportion(matched, events)
    precision = proportion(matched, alarms)
    empty_recall = proportion(bomb_empty["matched"], bomb_empty["events"])
    stocked_recall = proportion(bomb_stocked["matched"], bomb_stocked["events"])
    return {
        "threshold": float(threshold),
        "prominence": float(prominence),
        "shards": len(rows),
        "frames": sum(int(row["frames"]) for row in rows),
        "events": events,
        "alarms": alarms,
        "matched": matched,
        "recall": recall,
        "precision": precision,
        "recall_tolerance_pm2": proportion(matched_before2, events),
        "latency": latency_summary(latencies),
        "frame": {
            "tp": frame_tp,
            "fp": frame_fp,
            "fn": frame_fn,
            "alarm_frames": alarms,
            "positive_frames": frame_tp + frame_fn,
            "precision": proportion(frame_tp, frame_tp + frame_fp),
            "recall": proportion(frame_tp, frame_tp + frame_fn),
        },
        "bombEmpty": {
            "events": bomb_empty["events"],
            "matched": bomb_empty["matched"],
            "recall": empty_recall,
            "latency": latency_summary(bomb_empty["latencies"]),
        },
        "bombStocked": {
            "events": bomb_stocked["events"],
            "matched": bomb_stocked["matched"],
            "recall": stocked_recall,
        },
        "perShard": per_shard,
        "eventRows": event_rows,
    }


def train_command(args) -> int:
    import time

    import torch
    import torch.nn.functional as F
    from torch import nn

    shards = _load_index(args.index)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    if split.get("schema") != SPLIT_SCHEMA:
        raise PixelDeathError(f"split schema {split.get('schema')!r} != {SPLIT_SCHEMA!r}")
    rows = split_rows(split, "train")
    val_rows = split_rows(split, "val")
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _torch_device(args.device)
    channels = int(getattr(args, "channels", CHANNELS) or CHANNELS)
    samples = build_samples(
        rows,
        seed=seed,
        negatives_per_positive=float(args.negatives_per_positive),
        respawn_share=float(getattr(args, "respawn_share", 0.5)),
        jitter=int(args.jitter),
    )
    labels = np.asarray(samples["labels"], dtype=np.float32)
    positions = np.asarray(samples["samples"], dtype=np.int64)
    cache = _PlaneCache(rows)
    position_cache = _PositionCache(rows) if int(getattr(args, "aux_cells", 0)) else None
    model = build_model(width=int(args.width), channels=channels,
                        aux_cells=int(getattr(args, "aux_cells", 0))).to(device)
    parameters = model_parameters(model)
    if parameters > 500_000:
        raise PixelDeathError(f"model has {parameters} parameters > 500k budget")
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    positives = max(1, int(labels.sum()))
    negatives = int(len(labels) - positives)
    # ``pos_weight = negatives/positives`` makes a 1:6 corpus behave like 1:1 for
    # the loss, which on this data produced a fire-on-everything detector (52
    # alarms for 7 deaths online, EXP-P2-011 §3).  The mix is handled by the
    # sampling ratio; the loss keeps the explicit knob.
    pos_weight = float(getattr(args, "pos_weight", 1.0))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight * negatives / positives], device=device)
    )
    aux_weight = float(getattr(args, "aux_weight", 0.5))
    generator = np.random.default_rng(seed)
    batch_size = int(args.batch_size)
    started = time.monotonic()
    deadline = started + 60.0 * float(args.max_minutes)
    history: list[dict[str, Any]] = []
    best = None
    for epoch in range(int(args.epochs)):
        model.train()
        order = generator.permutation(len(positions))
        total = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            if time.monotonic() > deadline:
                break
            chunk = order[start : start + batch_size]
            batch_samples = [tuple(int(v) for v in positions[index]) for index in chunk]
            features = _batch_features(cache, batch_samples, channels=channels)
            target = torch.from_numpy(labels[chunk]).to(device)
            inputs = torch.from_numpy(features).to(device)
            optimiser.zero_grad(set_to_none=True)
            if position_cache is not None:
                logits, aux_logits = model(inputs, aux=True)
                cells = torch.from_numpy(_batch_cells(position_cache, batch_samples)).to(device)
                loss = loss_fn(logits, target) + float(aux_weight) * F.cross_entropy(
                    aux_logits, cells, ignore_index=-100
                )
            else:
                logits = model(inputs)
                loss = loss_fn(logits, target)
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(chunk)
            seen += len(chunk)
        # validation on a fixed stride of the val frames for a cheap signal
        (threshold, prominence), table = _select_threshold(
            model, val_rows[:60], device=device,
            thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), resolution="play96",
            channels=channels,
        )
        entry = {
            "epoch": epoch,
            "loss": total / max(1, seen),
            "samples": seen,
            "seconds": round(time.monotonic() - started, 2),
            "valThreshold": threshold,
            "valProminence": prominence,
            "valBest": max(table, key=lambda row: row["score"]) if table else None,
        }
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        score = entry["valBest"]["score"] if entry["valBest"] else -1.0
        if best is None or score > best[0]:
            best = (score, {key: value for key, value in model.state_dict().items()},
                    {"threshold": threshold, "prominence": prominence, "epoch": epoch})
        if time.monotonic() > deadline:
            break
    if best is None:
        raise PixelDeathError("training produced no epoch")
    model.load_state_dict(best[1])
    best_rule = best[2]
    checkpoint = {
        "schema": "v400-wp5-pixel-ckpt-1",
        "state": model.state_dict(),
        "config": {
            "width": int(args.width),
            "channels": channels,
            "window": WINDOW,
            "lags": list(LAGS),
            "resolution": "play96",
            "posAfter": POS_AFTER,
            "refractory": REFRACTORY,
            "eventDelay": EVENT_DELAY,
            "jitter": int(args.jitter),
            "seed": seed,
            "epochs": int(args.epochs),
            "auxCells": int(getattr(args, "aux_cells", 0)),
            "auxWeight": float(getattr(args, "aux_weight", 0.5)),
        },
        "parameters": parameters,
        "samples": {
            "positives": samples["positives"],
            "respawn_negatives": samples["respawn_negatives"],
            "uniform_negatives": samples["uniform_negatives"],
        },
        "valThreshold": best_rule["threshold"],
        "valProminence": best_rule["prominence"],
        "valEpoch": best_rule["epoch"],
        "history": history,
        "trainSeconds": round(time.monotonic() - started, 2),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    if args.report:
        serialisable = {key: value for key, value in checkpoint.items() if key != "state"}
        Path(args.report).write_text(json.dumps(serialisable, indent=1) + "\n",
                                     encoding="utf-8")
    print(json.dumps({key: checkpoint[key] for key in
                      ("parameters", "trainSeconds", "device", "gpu", "samples")}, indent=1))
    return 0


def load_checkpoint(path: str | Path, *, device=None):
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != "v400-wp5-pixel-ckpt-1":
        raise PixelDeathError(f"checkpoint schema {payload.get('schema')!r} is not a WP5 model")
    config = payload.get("config") or {}
    state = payload.get("state") or {}
    weight = state.get("body.0.weight")
    channels = int(config.get("channels") or (weight.shape[1] if weight is not None
                                              else CHANNELS))
    model = build_model(width=int(config.get("width", 32)), channels=channels,
                        aux_cells=int(config.get("auxCells", 0)))
    model.load_state_dict(payload["state"])
    model.eval()
    payload.setdefault("config", {})["channels"] = channels
    if device is not None:
        model = model.to(device)
    return model, payload


def eval_command(args) -> int:
    shards = _load_index(args.index)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    if split.get("schema") != SPLIT_SCHEMA:
        raise PixelDeathError(f"split schema {split.get('schema')!r} is not {SPLIT_SCHEMA!r}")
    device = _torch_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device=device)
    channels = int((payload.get("config") or {}).get("channels") or CHANNELS)
    wanted = [name for name in str(args.splits).split(",") if name]
    rows_of = {name: split_rows(split, name) for name in wanted}
    if args.thresholds:
        candidates = [float(value) for value in str(args.thresholds).split(",")]
        threshold, _table = float(candidates[0]), None
        prominence = float(args.prominence)
        selected_on = "fixed"
    else:
        (threshold, prominence), table = _select_threshold(
            model, rows_of.get("val") or split_rows(split, "train")[:40], device=device,
            thresholds=(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                        0.6, 0.7, 0.8, 0.9), resolution="play96", channels=channels,
        )
        selected_on = "val"
    results = {
        name: _evaluate_split(model, rows_of[name], device=device, threshold=threshold,
                              prominence=prominence, channels=channels)
        for name in wanted
    }
    checkpoint = {
        "parameters": payload.get("parameters"),
        "config": payload.get("config"),
        "trainSeconds": payload.get("trainSeconds"),
        "samples": payload.get("samples"),
        "history": payload.get("history"),
    }
    report = offline_report(
        split=split,
        checkpoint=checkpoint,
        test=results.get("test") or results[wanted[-1]],
        val=results.get("val"),
        extra={
            "threshold": threshold,
            "prominence": prominence,
            "thresholdSelectedOn": selected_on,
            "thresholdSweep": table if not args.thresholds else None,
            "splits": results,
        },
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    summary = {
        name: {
            "events": results[name]["events"],
            "alarms": results[name]["alarms"],
            "recall": results[name]["recall"]["value"],
            "recallLB": results[name]["recall"]["lower_bound_95"],
            "precision": results[name]["precision"]["value"],
            "precisionLB": results[name]["precision"]["lower_bound_95"],
            "latency": results[name]["latency"],
            "bombEmpty": results[name]["bombEmpty"],
        }
        for name in wanted
    }
    print(json.dumps({"threshold": threshold, "prominence": prominence,
                      "selectedOn": selected_on,
                      "results": summary}, indent=1))
    return 0


def bench_command(args) -> int:
    import time

    import torch

    model, payload = load_checkpoint(args.checkpoint)
    channels = int((payload.get("config") or {}).get("channels") or CHANNELS)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    frames = int(args.frames)
    rng = np.random.default_rng(0)
    planes = rng.integers(0, 255, size=(frames + WINDOW, ) + PLAY96_SHAPE, dtype=np.uint8)
    # warm up
    for frame in range(WINDOW - 1, WINDOW + 8):
        features = frame_channels(planes, frame, channels=channels)
        with torch.no_grad():
            model(torch.from_numpy(features)[None])
    times: list[float] = []
    for frame in range(WINDOW - 1, WINDOW - 1 + frames):
        started = time.perf_counter()
        features = frame_channels(planes, frame, channels=channels)
        with torch.no_grad():
            model(torch.from_numpy(features)[None])
        times.append((time.perf_counter() - started) * 1000.0)
    times_arr = np.asarray(times)
    payload_out = {
        "schema": "v400-wp5-pixel-bench-1",
        "frames": frames,
        "threads": torch.get_num_threads(),
        "channels": channels,
        "parameters": payload.get("parameters"),
        "ms_per_frame_mean": float(times_arr.mean()),
        "ms_per_frame_median": float(np.median(times_arr)),
        "ms_per_frame_p95": float(np.percentile(times_arr, 95)),
        "ms_per_frame_max": float(times_arr.max()),
        "fps_mean": float(1000.0 / times_arr.mean()),
        "inference_only": "window features + forward pass, single frame, CPU",
    }
    print(json.dumps(payload_out, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload_out, indent=1) + "\n", encoding="utf-8")
    return 0


def gate_command(args) -> int:
    """Combine the offline and online artifacts into one fail-closed verdict."""

    offline = json.loads(Path(args.offline).read_text(encoding="utf-8"))
    if offline.get("schema") != OFFLINE_SCHEMA:
        raise PixelDeathError(f"offline schema {offline.get('schema')!r} != {OFFLINE_SCHEMA!r}")
    verdict: dict[str, Any] = {"schema": GATE_SCHEMA, "offline": None, "online": None}
    test = offline.get("test") or {}
    verdict["offline"] = {
        "recall": test.get("recall"),
        "precision": test.get("precision"),
        "latency": test.get("latency"),
        "bombEmpty": test.get("bombEmpty"),
        "gate": bool(
            (test.get("recall") or {}).get("lower_bound_95", 0.0) >= 0.99
            and (test.get("precision") or {}).get("lower_bound_95", 0.0) >= 0.99
            and (test.get("latency") or {}).get("within_limit")
        ),
    }
    online = None
    if args.online:
        payload = json.loads(Path(args.online).read_text(encoding="utf-8"))
        online = online_gate(payload, min_episodes=int(args.min_episodes),
                             min_frames=int(args.min_frames))
        verdict["online"] = online
    else:
        verdict["online"] = {"passed": False, "failures": ["no online artifact supplied"]}
    verdict["passed"] = bool(verdict["offline"]["gate"] and verdict["online"].get("passed"))
    verdict["row14"] = {
        "requirement": "online recall LB >= 0.99, precision LB >= 0.99, latency max <= 10",
        "passed": bool(verdict["online"].get("passed")),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: verdict[key] for key in ("passed", "row14", "online")},
                     ensure_ascii=False, indent=1))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
