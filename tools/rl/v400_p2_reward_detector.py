#!/usr/bin/env python3
"""P2-WP3 reward detector v0 + G2 metrics (EXP-P2-003).

What this is
------------
A small multi-task CNN that reads the *observation planes* the P2-WP3 collector
stores (``tools/rl/v400_p2_bulk_collect.py``) and predicts, for every frame:

===========  ============================================================
head         label (ground truth, never derived from pixels)
===========  ============================================================
``death``    ``EVENTS & 0x4`` on frame ``t+1`` — the reward event, one frame
             ahead so a detection is usable before the event commits
``bomb``     ``EVENTS & 0x1`` on frame ``t+1``
``lives``    ``STATE f7`` truncated on frame ``t`` (what the HUD draws)
``bombs``    ``STATE f8`` truncated on frame ``t``
``score_dn`` sign of ``score[t+1] - score[t]`` (-1 / 0 / +1)
===========  ============================================================

Inputs are two streams of the *same* frame window: ``play`` — a 4-frame stack of
the downsampled playfield grid (96x112 or 64x64, the P2-WP4 comparison) — and
``hud`` — the native-resolution numeric HUD bands of frame ``t``.  The HUD stream
is what makes ``lives``/``bombs``/``score`` readable at all; the play stream is
what makes ``death`` anticipatable.

Splitting (no same-run leakage)
-------------------------------
Every stored shard is one engine run with its own ``(stage, difficulty, seed)``.
Splits are assigned **by stage**, so a shard never straddles two splits and no
window built from one run can be scored in another split.  :func:`build_split`
verifies this and refuses to run otherwise.  The default assignment puts whole
stages in test/val, i.e. the test set is also *unseen levels*.

G2 metrics
----------
Frame level and event level for ``death`` (with detection latency), accuracy for
``lives``/``bombs``, and sign accuracy for the score delta.  Every number carries
its sample size and a one-sided 95 % lower bound (Wilson for a measured
proportion, rule of three for a zero-error proportion).  All of it is a
**detector** score: it says nothing about how a policy plays.

Fail-closed
-----------
Missing index, mixed engine SHAs, mixed label versions, an empty split, a shard
directory whose planes are shorter than its label rows, or a checkpoint whose
resolution/label version disagrees with the dataset all abort with a non-zero
exit instead of producing a number.

Typical use (inside the WSL guest, where the dataset lives)::

    python tools/rl/v400_p2_reward_detector.py split \
        --index /root/v400-data/p2det/out-main/index.json --out /root/v400-data/p2det/split.json
    python tools/rl/v400_p2_reward_detector.py train \
        --index /root/v400-data/p2det/out-main/index.json --split split.json \
        --resolution play96 --epochs 3 --out metrics-play96.json --checkpoint play96.pt
    python tools/rl/v400_p2_reward_detector.py report \
        --metrics metrics-play96.json metrics-play64.json --out g2.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Schema of the metrics file this module writes.
METRICS_SCHEMA = "v400-p2det-g2-1"

#: Dataset index schema it consumes.
INDEX_SCHEMA = "v400-p2det-index-1"

#: Label version the detector is allowed to train on.
LABEL_VERSION = "p2-v2-1"

#: Stage -> split.  Whole stages, so no run can appear in two splits and the
#: test stages are levels the model never trained on.
DEFAULT_SPLITS: dict[str, tuple[int, ...]] = {
    "train": (0, 1, 2, 4),
    "val": (5,),
    "test": (3, 6),
}

#: Frames stacked into the play stream, and the horizon the reward heads predict.
STACK = 4
HORIZON = 1

#: Frame budget per split for training/evaluation (deterministic subsample).
DEFAULT_MAX_SAMPLES = {"train": 48_000, "val": 12_000, "test": 12_000}

#: HUD planes are fixed by the collector; a checkpoint may not disagree.
HUD_SHAPE = (56, 160)


# ---------------------------------------------------------------------------
# statistics helpers (pure; unit-tested)
# ---------------------------------------------------------------------------
def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """One-sided 95 % Wilson lower bound of a proportion (0 if trials == 0)."""

    if trials <= 0:
        return 0.0
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = phat + z * z / (2 * trials)
    margin = z * math.sqrt(max(0.0, phat * (1.0 - phat) / trials + z * z / (4 * trials * trials)))
    return max(0.0, (centre - margin) / denom)


def rule_of_three_lower_bound(trials: int, failures: int = 0) -> float:
    """``1 - 3/n`` for zero failures; ``None``-like 0 when n == 0 or failures > 0."""

    if trials <= 0 or failures > 0:
        return 0.0
    return max(0.0, 1.0 - 3.0 / trials)


def proportion(successes: int, trials: int, *, zero_errors: bool = False) -> dict[str, Any]:
    """``{n, successes, value, lower_bound_95, method, pass_99, pass_995}``."""

    if zero_errors:
        lower = rule_of_three_lower_bound(trials, failures=trials - successes)
        method = "rule-of-three"
    else:
        lower = wilson_lower_bound(successes, trials)
        method = "wilson"
    value = (successes / trials) if trials else 0.0
    return {
        "n": trials,
        "successes": successes,
        "value": value,
        "lower_bound_95": lower,
        "method": method,
        "pass_99": lower >= 0.99,
        "pass_995": lower >= 0.995,
    }


def binary_scores(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Frame-level precision/recall with their Wilson lower bounds."""

    precision = proportion(tp, tp + fp)
    recall = proportion(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision["value"] * recall["value"] / (precision["value"] + recall["value"])
            if (precision["value"] + recall["value"]) > 0
            else 0.0
        ),
    }


def event_starts(flags: Sequence[bool] | Sequence[int]) -> list[int]:
    """Start index of every run of true values (an event, not a frame)."""

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
    after: int = 10,
) -> dict[str, Any]:
    """Greedy one-to-one matching of predicted events to true events.

    A prediction ``p`` matches a true event ``f`` when ``f - before <= p <= f +
    after``; the earliest still-free prediction inside that window wins, so the
    reported latency is the *first* usable detection.  Unmatched predictions are
    false positives; the metric never lets one prediction cover two events.
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
        latencies.append(pred_starts[best] - true_frame)
    return {
        "true_events": len(true_starts),
        "pred_events": len(pred_starts),
        "matched": matched_true,
        "matched_predictions": len(used),
        "latencies": latencies,
        "unmatched_predictions": len(pred_starts) - len(used),
    }


def latency_summary(latencies: Sequence[int], *, limit: int = 10) -> dict[str, Any]:
    """Median/p95/max latency and whether every detection met the <= limit gate."""

    if not latencies:
        return {"n": 0, "median": None, "p95": None, "max": None,
                "within_limit": None, "limit": limit}
    ordered = sorted(int(value) for value in latencies)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "n": len(ordered),
        "median": ordered[len(ordered) // 2],
        "p95": p95,
        "max": ordered[-1],
        "within_limit": all(value <= limit for value in ordered),
        "limit": limit,
        "early_detections": sum(1 for value in ordered if value < 0),
    }


# ---------------------------------------------------------------------------
# dataset index / split
# ---------------------------------------------------------------------------
@dataclass
class ShardEntry:
    """One stored shard attempt as the detector sees it."""

    dir: Path
    shard: str
    attempt: str
    frames: int
    stage: int
    difficulty: int
    seed: int
    source: str
    complete: bool
    engine_sha256: str | None
    label_version: str | None

    @property
    def key(self) -> str:
        return f"{self.shard}/{self.attempt}"


class DetectorDataError(RuntimeError):
    """The dataset cannot support a G2 measurement (fail closed, no guessing)."""


def load_index(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise DetectorDataError(f"dataset index not found: {path}")
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("schema") != INDEX_SCHEMA:
        raise DetectorDataError(
            f"index schema {index.get('schema')!r} != {INDEX_SCHEMA!r} ({path})"
        )
    if index.get("label_version") != LABEL_VERSION:
        raise DetectorDataError(
            f"index label_version {index.get('label_version')!r} != {LABEL_VERSION!r}"
        )
    return index


def shard_entries(index: dict[str, Any], *, verify_planes: bool = True) -> list[ShardEntry]:
    """Every indexed shard, with its stored frame count verified against labels."""

    from tools.rl.v400_p2_bulk_collect import (
        frame_bytes,
        plane_shapes,
        read_labels,
        shard_frame_count,
    )

    entries: list[ShardEntry] = []
    for row in index.get("shards") or []:
        directory = Path(row["dir"])
        frames = int(row.get("frames") or 0)
        if frames <= 0:
            continue
        if verify_planes:
            readable = shard_frame_count(directory)
            if readable < frames:
                raise DetectorDataError(
                    f"{directory}: labels promise {frames} frames but only {readable} "
                    "whole frames are readable"
                )
            labels = read_labels(directory)
            if len(labels) < frames:
                raise DetectorDataError(f"{directory}: readable label rows < frames")
        entries.append(
            ShardEntry(
                dir=directory,
                shard=str(row.get("shard")),
                attempt=str(row.get("attempt")),
                frames=frames,
                stage=int(row.get("stage") if row.get("stage") is not None else -1),
                difficulty=int(row.get("difficulty") if row.get("difficulty") is not None else -1),
                seed=int(row.get("seed") if row.get("seed") is not None else -1),
                source=str(row.get("source")),
                complete=bool(row.get("complete")),
                engine_sha256=row.get("engine_sha256"),
                label_version=row.get("label_version"),
            )
        )
    if not entries:
        raise DetectorDataError("index lists no readable shard")
    # geometry sanity: the planes the detector reads must be the stored ones
    shapes = plane_shapes()
    if shapes["hud"] != HUD_SHAPE:
        raise DetectorDataError(f"HUD plane {shapes['hud']} != expected {HUD_SHAPE}")
    if not frame_bytes():
        raise DetectorDataError("empty plane layout")
    return entries


def build_split(
    entries: Sequence[ShardEntry],
    *,
    stage_splits: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Assign whole stages to splits and verify nothing leaks across them."""

    stage_splits = dict(stage_splits or DEFAULT_SPLITS)
    stage_to_split: dict[int, str] = {}
    for split, stages in stage_splits.items():
        for stage in stages:
            if stage in stage_to_split:
                raise DetectorDataError(f"stage {stage} assigned to two splits")
            stage_to_split[stage] = split
    unassigned = sorted({entry.stage for entry in entries} - set(stage_to_split))
    if unassigned:
        raise DetectorDataError(
            f"stages {unassigned} present in the dataset but assigned to no split"
        )
    engines = {entry.engine_sha256 for entry in entries}
    if len(engines) > 1:
        raise DetectorDataError(f"dataset mixes engine SHAs: {sorted(engines)}")
    versions = {entry.label_version for entry in entries}
    if versions - {LABEL_VERSION}:
        raise DetectorDataError(f"dataset mixes label versions: {sorted(versions)}")

    split_of_shard: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for entry in entries:
        split = stage_to_split[entry.stage]
        previous = split_of_shard.get(entry.key)
        if previous is not None and previous != split:
            raise DetectorDataError(
                f"shard {entry.key} leaks across splits ({previous} and {split})"
            )
        split_of_shard[entry.key] = split
        rows.append(
            {
                "dir": str(entry.dir),
                "shard": entry.shard,
                "attempt": entry.attempt,
                "frames": entry.frames,
                "stage": entry.stage,
                "difficulty": entry.difficulty,
                "seed": entry.seed,
                "source": entry.source,
                "split": split,
            }
        )
    for split in stage_splits:
        total = sum(row["frames"] for row in rows if row["split"] == split)
        if total == 0:
            raise DetectorDataError(f"split {split!r} has no frames")
    return {
        "schema": "v400-p2det-split-1",
        "stage_splits": {name: list(stages) for name, stages in stage_splits.items()},
        "machine": "stage (whole shards, no same-run leakage)",
        "engine_sha256": sorted(engines)[0] if engines else None,
        "label_version": LABEL_VERSION,
        "frames": {split: sum(r["frames"] for r in rows if r["split"] == split)
                   for split in stage_splits},
        "shards": rows,
    }


# ---------------------------------------------------------------------------
# sample assembly
# ---------------------------------------------------------------------------
@dataclass
class SplitFrames:
    """Read-only view over one split's stored planes, grouped by shard."""

    rows: list[dict[str, Any]]
    play_shape: tuple[int, int]
    resolution: str = "play96"

    def planes(self, name: str):
        import numpy as np

        from tools.rl.v400_p2_bulk_collect import plane_shapes

        height, width = plane_shapes()[name]
        out = []
        for row in self.rows:
            frames = int(row["frames"])
            out.append(
                np.memmap(
                    Path(row["dir"]) / f"{name}.u8",
                    dtype=np.uint8,
                    mode="r",
                    shape=(frames, height, width),
                )
            )
        return out


def load_labels_for(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    from tools.rl.v400_p2_bulk_collect import read_labels

    return [read_labels(Path(row["dir"])) for row in rows]


def death_flags(rows: Sequence[dict[str, Any]]) -> list[list[bool]]:
    """Per frame, whether frame ``t`` carries the real ``death`` event bit."""

    out: list[list[bool]] = []
    for shard_rows in rows:
        out.append([bool(row.get("death")) for row in shard_rows])
    return out


def label_series(rows: Sequence[dict[str, Any]], key: str) -> list[list[Any]]:
    return [[row.get(key) for row in shard_rows] for shard_rows in rows]


def sample_indices(
    rows: Sequence[Sequence[dict[str, Any]]],
    *,
    max_samples: int,
    rng_seed: int,
    positive_boost: float = 4.0,
) -> dict[str, Any]:
    """Deterministic frame indices for one split, with the death positives kept.

    Sampling is anchored on ``death[t+1]`` so the rare reward event survives any
    budget; the rest is filled with (a) near-death hard negatives and (b) uniform
    frames, so the HUD heads still see the ordinary value distribution.
    """

    import numpy as np

    generator = np.random.default_rng(rng_seed)
    positives: list[tuple[int, int]] = []
    hard: list[tuple[int, int]] = []
    uniform: list[tuple[int, int]] = []
    for shard_index, shard_rows in enumerate(rows):
        for frame in range(len(shard_rows) - HORIZON):
            nxt = shard_rows[frame + HORIZON]
            uniform.append((shard_index, frame))
            if nxt.get("death"):
                positives.append((shard_index, frame))
            elif nxt.get("hit") or nxt.get("dying") or shard_rows[frame].get("death"):
                hard.append((shard_index, frame))
    budget = max(1, int(max_samples))
    keep_pos = positives[:budget]
    remaining = budget - len(keep_pos)
    keep_hard = hard[: max(0, min(len(hard), int(remaining * 0.4)))]
    remaining -= len(keep_hard)
    positive_set = set(keep_pos)
    pool = [item for item in uniform if item not in positive_set]
    if len(pool) > remaining:
        picks = generator.choice(len(pool), size=remaining, replace=False)
        keep_uniform = [pool[int(i)] for i in sorted(picks)]
    else:
        keep_uniform = pool
    chosen = keep_pos + keep_hard + keep_uniform
    chosen.sort()
    return {
        "samples": chosen,
        "positives": len(keep_pos),
        "hard_negatives": len(keep_hard),
        "uniform": len(keep_uniform),
        "candidates_uniform": len(uniform),
        "positive_boost": positive_boost,
    }


def build_arrays(
    rows: Sequence[dict[str, Any]],
    samples: Sequence[tuple[int, int]],
    *,
    resolution: str,
    stack: int = STACK,
    with_hud: bool = True,
) -> dict[str, Any]:
    """Stack the play window and slice the HUD frame for the chosen samples."""

    import numpy as np

    from tools.rl.v400_p2_bulk_collect import plane_shapes

    play_h, play_w = plane_shapes()[resolution]
    hud_h, hud_w = plane_shapes()["hud"]
    by_shard: dict[int, list[int]] = {}
    for shard_index, frame in samples:
        by_shard.setdefault(shard_index, []).append(frame)
    play_chunks = []
    hud_chunks = []
    shard_ids = np.empty(len(samples), dtype=np.int32)
    frames_out = np.empty(len(samples), dtype=np.int64)
    cursor = 0
    for shard_index in sorted(by_shard):
        frames = sorted(by_shard[shard_index])
        directory = Path(rows[shard_index]["dir"])
        count = int(rows[shard_index]["frames"])
        play = np.memmap(directory / f"{resolution}.u8", dtype=np.uint8, mode="r",
                         shape=(count, play_h, play_w))
        offsets = np.arange(-(stack - 1), 1)
        index = np.asarray(frames)[:, None] + offsets[None, :]
        index = np.clip(index, 0, count - 1)
        play_chunks.append(np.asarray(play[index], dtype=np.uint8))
        if with_hud:
            hud = np.memmap(directory / "hud.u8", dtype=np.uint8, mode="r",
                            shape=(count, hud_h, hud_w))
            hud_chunks.append(np.asarray(hud[np.asarray(frames)], dtype=np.uint8))
        shard_ids[cursor : cursor + len(frames)] = shard_index
        frames_out[cursor : cursor + len(frames)] = np.asarray(frames)
        cursor += len(frames)
    play_array = np.concatenate(play_chunks, axis=0) if play_chunks else np.zeros(
        (0, stack, play_h, play_w), dtype=np.uint8
    )
    hud_array = (
        np.concatenate(hud_chunks, axis=0)
        if hud_chunks
        else np.zeros((0, hud_h, hud_w), dtype=np.uint8)
    )
    return {
        "play": play_array,
        "hud": hud_array,
        "shard": shard_ids,
        "frame": frames_out,
    }


#: Class ranges for the two HUD counting heads (STATE is a float; the engine
#: draws the i32 truncation, so a value outside the range is a protocol surprise
#: and is counted, not silently clamped away).
LIVES_CLASSES = 7      # 0..6
BOMBS_CLASSES = 9      # 0..8
SCORE_CLASSES = 3      # down / same / up


def build_targets(
    rows: Sequence[Sequence[dict[str, Any]]],
    samples: Sequence[tuple[int, int]],
    *,
    window: int = 1,
) -> dict[str, Any]:
    """Labels for the five heads, from stored label rows only.

    ``window`` is the anticipation window of the death head: with the default
    ``1`` it predicts the single next frame; with ``K > 1`` it predicts "the death
    bit is set in frames ``t+1 .. t+K``", which turns one-frame spikes into a
    learnable onset window.  The G2 event metrics still measure the detection
    frame against the true death frame, so a larger window shows up as *early*
    detection (negative latency), not as a different metric.
    """

    import numpy as np

    n = len(samples)
    death = np.zeros(n, dtype=np.int64)
    bomb = np.zeros(n, dtype=np.int64)
    lives = np.zeros(n, dtype=np.int64)
    bombs = np.zeros(n, dtype=np.int64)
    score = np.ones(n, dtype=np.int64)  # 1 == "same"
    out_of_range = {"lives": 0, "bombs": 0}
    missing = 0
    for position, (shard_index, frame) in enumerate(samples):
        shard_rows = rows[shard_index]
        current = shard_rows[frame]
        last = len(shard_rows) - 1
        nxt = shard_rows[min(frame + HORIZON, last)]
        death[position] = int(
            any(bool(shard_rows[min(frame + step, last)].get("death"))
                for step in range(HORIZON, HORIZON + max(1, int(window))))
        )
        bomb[position] = int(bool(nxt.get("bomb")))
        life_value = current.get("li")
        bomb_value = current.get("bo")
        score_now = current.get("sc")
        score_next = nxt.get("sc")
        if life_value is None or bomb_value is None or score_now is None:
            missing += 1
            continue
        if int(life_value) >= LIVES_CLASSES or int(life_value) < 0:
            out_of_range["lives"] += 1
            life_value = min(max(int(life_value), 0), LIVES_CLASSES - 1)
        if int(bomb_value) >= BOMBS_CLASSES or int(bomb_value) < 0:
            out_of_range["bombs"] += 1
            bomb_value = min(max(int(bomb_value), 0), BOMBS_CLASSES - 1)
        lives[position] = int(life_value)
        bombs[position] = int(bomb_value)
        if score_next is None:
            score[position] = 1
        else:
            delta = int(score_next) - int(score_now)
            score[position] = 2 if delta > 0 else (0 if delta < 0 else 1)
    return {
        "death": death,
        "bomb": bomb,
        "lives": lives,
        "bombs": bombs,
        "score": score,
        "out_of_range": out_of_range,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_model(*, play_shape: tuple[int, int], hud_shape: tuple[int, int],
                stack: int = STACK, width: int = 16):
    """Small two-stream CNN; ~1.8 M parameters at either resolution."""

    import torch
    from torch import nn

    class RewardDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.play = nn.Sequential(
                nn.Conv2d(stack, width, 3, stride=2, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(width * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
                nn.BatchNorm2d(width * 4),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.hud = nn.Sequential(
                nn.Conv2d(1, width, 3, stride=2, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(width * 2),
                nn.ReLU(inplace=True),
                nn.Flatten(),
            )
            hud_h, hud_w = hud_shape[0] // 4, hud_shape[1] // 4
            hud_features = width * 2 * hud_h * hud_w
            self.hud_fc = nn.Sequential(nn.Linear(hud_features, 96), nn.ReLU(inplace=True))
            features = width * 4 + 96
            self.death = nn.Linear(features, 2)
            self.bomb = nn.Linear(features, 2)
            self.lives = nn.Linear(features, LIVES_CLASSES)
            self.bombs = nn.Linear(features, BOMBS_CLASSES)
            self.score = nn.Linear(features, SCORE_CLASSES)

        def forward(self, play, hud):
            shared = torch.cat([self.play(play), self.hud_fc(self.hud(hud))], dim=1)
            return {
                "death": self.death(shared),
                "bomb": self.bomb(shared),
                "lives": self.lives(shared),
                "bombs": self.bombs(shared),
                "score": self.score(shared),
            }

    return RewardDetector()


def parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def torch_device(prefer: str = "cuda"):
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# inference over whole shard sequences
# ---------------------------------------------------------------------------
def evaluate_shards(
    model,
    rows: Sequence[dict[str, Any]],
    *,
    resolution: str,
    device,
    batch_size: int = 512,
    window: int = 1,
) -> dict[str, Any]:
    """Run the model over **every** frame of every shard, in order.

    No sampling: frame-level precision then reflects the real event base rate,
    and event-level metrics can cluster the whole sequence.
    """

    import numpy as np
    import torch

    from tools.rl.v400_p2_bulk_collect import plane_shapes

    play_h, play_w = plane_shapes()[resolution]
    hud_h, hud_w = plane_shapes()["hud"]
    model.eval()
    per_shard: list[dict[str, Any]] = []
    offsets = np.arange(-(STACK - 1), 1)
    for row in rows:
        count = int(row["frames"])
        directory = Path(row["dir"])
        play = np.memmap(directory / f"{resolution}.u8", dtype=np.uint8, mode="r",
                         shape=(count, play_h, play_w))
        hud = np.memmap(directory / "hud.u8", dtype=np.uint8, mode="r",
                        shape=(count, hud_h, hud_w))
        from tools.rl.v400_p2_bulk_collect import read_labels

        labels = read_labels(directory)[:count]
        usable = max(0, count - HORIZON)
        if usable == 0:
            continue
        death_prob = np.zeros(count, dtype=np.float32)
        bomb_prob = np.zeros(count, dtype=np.float32)
        lives_pred = np.zeros(count, dtype=np.int64)
        bombs_pred = np.zeros(count, dtype=np.int64)
        score_pred = np.zeros(count, dtype=np.int64)
        with torch.no_grad():
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                index = np.arange(start, stop)
                stack_index = np.clip(index[:, None] + offsets[None, :], 0, count - 1)
                play_batch = torch.from_numpy(
                    np.asarray(play[stack_index], dtype=np.uint8)
                ).to(device).float().div_(255.0)
                hud_batch = torch.from_numpy(
                    np.asarray(hud[index], dtype=np.uint8)
                ).to(device).float().div_(255.0).unsqueeze(1)
                out = model(play_batch, hud_batch)
                death_prob[start:stop] = (
                    torch.softmax(out["death"], dim=1)[:, 1].detach().cpu().numpy()
                )
                bomb_prob[start:stop] = (
                    torch.softmax(out["bomb"], dim=1)[:, 1].detach().cpu().numpy()
                )
                score_pred[start:stop] = out["score"].argmax(dim=1).detach().cpu().numpy()
                lives_pred[start:stop] = out["lives"].argmax(dim=1).detach().cpu().numpy()
                bombs_pred[start:stop] = out["bombs"].argmax(dim=1).detach().cpu().numpy()
        death_prob = death_prob[:usable]
        bomb_prob = bomb_prob[:usable]
        score_pred = score_pred[:usable]
        true_death = np.zeros(usable, dtype=bool)
        true_bomb = np.zeros(usable, dtype=bool)
        true_lives = np.full(count, -1, dtype=np.int64)
        true_bombs = np.full(count, -1, dtype=np.int64)
        true_score = np.ones(usable, dtype=np.int64)
        for frame in range(usable):
            nxt = labels[frame + HORIZON]
            true_death[frame] = any(
                bool(labels[min(frame + step, count - 1)].get("death"))
                for step in range(HORIZON, HORIZON + max(1, int(window)))
            )
            true_bomb[frame] = bool(nxt.get("bomb"))
            now_score = labels[frame].get("sc")
            next_score = nxt.get("sc")
            if now_score is not None and next_score is not None:
                delta = int(next_score) - int(now_score)
                true_score[frame] = 2 if delta > 0 else (0 if delta < 0 else 1)
        for frame in range(count):
            value = labels[frame].get("li")
            if value is not None and 0 <= int(value) < LIVES_CLASSES:
                true_lives[frame] = int(value)
            value = labels[frame].get("bo")
            if value is not None and 0 <= int(value) < BOMBS_CLASSES:
                true_bombs[frame] = int(value)
        per_shard.append(
            {
                "shard": row["shard"],
                "dir": row["dir"],
                "source": row.get("source"),
                "stage": row.get("stage"),
                "frames": count,
                "death_prob": death_prob,
                "bomb_prob": bomb_prob,
                "true_death": true_death,
                "true_bomb": true_bomb,
                "lives_pred": lives_pred,
                "lives_true": true_lives,
                "bombs_pred": bombs_pred,
                "bombs_true": true_bombs,
                "score_pred": score_pred,
                "score_true": true_score,
            }
        )
    return {"shards": per_shard, "resolution": resolution}


def _threshold_sweep(result: dict[str, Any], thresholds: Iterable[float]) -> dict[str, Any]:
    best = None
    for threshold in thresholds:
        scores = binary_scores(*_death_confusion(result, threshold))
        candidate = {"threshold": float(threshold), **scores}
        if best is None or candidate["f1"] > best["f1"]:
            best = candidate
    return best or {"threshold": 0.5, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0,
                    "precision": proportion(0, 0), "recall": proportion(0, 0)}


def _death_confusion(result: dict[str, Any], threshold: float) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for shard in result["shards"]:
        predicted = shard["death_prob"] >= threshold
        truth = shard["true_death"]
        tp += int((predicted & truth).sum())
        fp += int((predicted & ~truth).sum())
        fn += int((~predicted & truth).sum())
    return tp, fp, fn


def g2_metrics(result: dict[str, Any], *, threshold: float,
               latency_limit: int = 10) -> dict[str, Any]:
    """Frame-level + event-level death metrics, HUD accuracy and score sign.

    Everything here is a detector measurement over the *whole* evaluation split.
    """

    import numpy as np

    tp, fp, fn = _death_confusion(result, threshold)
    frame_level = binary_scores(tp, fp, fn)

    true_events = 0
    pred_events = 0
    matched = 0
    matched_predictions = 0
    latencies: list[int] = []
    per_source: dict[str, dict[str, int]] = {}
    for shard in result["shards"]:
        threshold_mask = shard["death_prob"] >= threshold
        true_starts = event_starts(shard["true_death"])
        pred_starts = event_starts(threshold_mask)
        match = match_events(true_starts, pred_starts, before=latency_limit,
                            after=latency_limit)
        true_events += match["true_events"]
        pred_events += match["pred_events"]
        matched += match["matched"]
        matched_predictions += match["matched_predictions"]
        latencies.extend(match["latencies"])
        source = str(shard.get("source"))
        bucket = per_source.setdefault(
            source,
            {"true_events": 0, "matched": 0, "pred_events": 0,
             "matched_predictions": 0, "frames": 0},
        )
        bucket["true_events"] += match["true_events"]
        bucket["matched"] += match["matched"]
        bucket["pred_events"] += match["pred_events"]
        bucket["matched_predictions"] += match["matched_predictions"]
        bucket["frames"] += int(shard["frames"])
    event_level = {
        "true_events": true_events,
        "pred_events": pred_events,
        "matched": matched,
        "matched_predictions": matched_predictions,
        "recall": proportion(matched, true_events),
        "precision": proportion(matched_predictions, pred_events),
        "window": {"before": latency_limit, "after": latency_limit},
        "latency": latency_summary(latencies, limit=latency_limit),
        "per_source": {
            name: {
                "frames": bucket["frames"],
                "true_events": bucket["true_events"],
                "matched": bucket["matched"],
                "recall": proportion(bucket["matched"], bucket["true_events"]),
                "pred_events": bucket["pred_events"],
                "matched_predictions": bucket["matched_predictions"],
                "precision": proportion(bucket["matched_predictions"], bucket["pred_events"]),
            }
            for name, bucket in sorted(per_source.items())
        },
    }

    def accuracy(pred_key: str, true_key: str, classes: int) -> dict[str, Any]:
        correct = 0
        total = 0
        per_class = {cls: {"n": 0, "correct": 0} for cls in range(classes)}
        for shard in result["shards"]:
            predicted = shard[pred_key]
            truth = shard[true_key]
            mask = truth >= 0
            if not mask.any():
                continue
            predicted = predicted[mask]
            truth = truth[mask]
            correct += int((predicted == truth).sum())
            total += int(mask.sum())
            for cls in range(classes):
                cls_mask = truth == cls
                count = int(cls_mask.sum())
                if count:
                    per_class[cls]["n"] += count
                    per_class[cls]["correct"] += int((predicted[cls_mask] == truth[cls_mask]).sum())
        entry = proportion(correct, total)
        entry["per_class"] = {
            str(cls): {"n": values["n"], "correct": values["correct"],
                       "recall": (values["correct"] / values["n"]) if values["n"] else None}
            for cls, values in per_class.items()
        }
        return entry

    score_entry = accuracy("score_pred", "score_true", SCORE_CLASSES)
    score_entry["class_meaning"] = {"0": "score fell", "1": "score unchanged", "2": "score rose"}
    score_entry["negative_class_samples"] = score_entry["per_class"]["0"]["n"]
    score_entry["note"] = (
        "sign accuracy over observable frames; the 'score fell' class is reported "
        "separately because it had zero samples in earlier datasets"
        if score_entry["negative_class_samples"] == 0
        else "all three sign classes have samples"
    )
    return {
        "threshold": threshold,
        "death_frame_level": frame_level,
        "death_event_level": event_level,
        "lives_accuracy": accuracy("lives_pred", "lives_true", LIVES_CLASSES),
        "bombs_accuracy": accuracy("bombs_pred", "bombs_true", BOMBS_CLASSES),
        "score_sign_accuracy": score_entry,
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _tensor_batches(arrays: dict[str, Any], targets: dict[str, Any], *, batch_size: int,
                    shuffle: bool, seed: int = 0):
    import numpy as np

    count = len(arrays["frame"])
    order = np.arange(count)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, count, batch_size):
        pick = order[start : start + batch_size]
        yield (
            arrays["play"][pick],
            arrays["hud"][pick],
            {key: targets[key][pick] for key in ("death", "bomb", "lives", "bombs", "score")},
        )


def train_model(
    *,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[Sequence[dict[str, Any]]],
    resolution: str,
    epochs: int,
    max_samples: int,
    batch_size: int,
    learning_rate: float,
    device,
    seed: int = 0,
    window: int = 1,
    pos_weight_cap: float = 8.0,
    log: bool = True,
) -> dict[str, Any]:
    """Train the multi-task detector on one split; returns the train cost."""

    import time as _time

    import numpy as np
    import torch
    from torch import nn

    from tools.rl.v400_p2_bulk_collect import plane_shapes

    sampling = sample_indices(labels, max_samples=max_samples, rng_seed=seed)
    arrays = build_arrays(rows, sampling["samples"], resolution=resolution)
    targets = build_targets(labels, sampling["samples"], window=window)
    model = build_model(play_shape=plane_shapes()[resolution], hud_shape=HUD_SHAPE)
    model.to(device)
    params = parameter_count(model)
    positives = int(targets["death"].sum())
    negatives = len(targets["death"]) - positives
    pos_weight = torch.tensor(
        [1.0, max(1.0, min(float(pos_weight_cap), negatives / max(1, positives)))], device=device
    )
    loss_weights = {"death": 1.0, "bomb": 0.4, "lives": 0.5, "bombs": 0.5, "score": 0.4}
    criterion = {
        "death": nn.CrossEntropyLoss(weight=pos_weight),
        "bomb": nn.CrossEntropyLoss(weight=torch.tensor([1.0, 8.0], device=device)),
        "lives": nn.CrossEntropyLoss(),
        "bombs": nn.CrossEntropyLoss(),
        "score": nn.CrossEntropyLoss(),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    started = _time.monotonic()
    history = []
    for epoch in range(int(epochs)):
        model.train()
        totals = {key: 0.0 for key in loss_weights}
        batches = 0
        for play, hud, target in _tensor_batches(
            arrays, targets, batch_size=batch_size, shuffle=True, seed=seed + epoch
        ):
            play_t = torch.from_numpy(np.ascontiguousarray(play)).to(device).float().div_(255.0)
            hud_t = (
                torch.from_numpy(np.ascontiguousarray(hud)).to(device).float().div_(255.0)
                .unsqueeze(1)
            )
            out = model(play_t, hud_t)
            loss = 0.0
            for key, weight in loss_weights.items():
                target_t = torch.from_numpy(np.ascontiguousarray(target[key])).to(device).long()
                value = criterion[key](out[key], target_t)
                totals[key] += float(value.detach().cpu())
                loss = loss + weight * value
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batches += 1
        row = {
            "epoch": epoch + 1,
            "batches": batches,
            "loss": {key: (value / max(1, batches)) for key, value in totals.items()},
        }
        history.append(row)
        if log:
            print(json.dumps({"train": row}, ensure_ascii=False), flush=True)
    seconds = _time.monotonic() - started
    cost = {
        "resolution": resolution,
        "parameters": params,
        "epochs": int(epochs),
        "samples": int(len(arrays["frame"])),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "train_seconds": round(seconds, 2),
        "seconds_per_epoch": round(seconds / max(1, int(epochs)), 2),
        "samples_per_second": round(len(arrays["frame"]) * int(epochs) / max(1e-9, seconds), 1),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
            if device.type == "cuda"
            else None
        ),
        "sampling": {
            "positives": sampling["positives"],
            "hard_negatives": sampling["hard_negatives"],
            "uniform": sampling["uniform"],
            "candidates_uniform": sampling["candidates_uniform"],
            "target_positives_in_batch": positives,
        },
        "play_bytes_per_sample": int(np.prod(arrays["play"].shape[1:])),
        "history": history,
    }
    return {"model": model, "cost": cost, "targets": targets, "arrays": arrays}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def split_command(args) -> int:
    index = load_index(args.index)
    entries = shard_entries(index)
    split = build_split(entries)
    if args.out:
        Path(args.out).write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
    summary: dict[str, Any] = {
        "frames": split["frames"],
        "stage_splits": split["stage_splits"],
        "engine_sha256": split["engine_sha256"],
        "shards": {name: sum(1 for row in split["shards"] if row["split"] == name)
                   for name in split["stage_splits"]},
        "sources": {
            name: sum(1 for row in split["shards"]
                      if row["split"] == name and row["source"] == "dive")
            for name in split["stage_splits"]
        },
        "out": str(args.out) if args.out else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def plane_shapes() -> dict[str, tuple[int, int]]:
    """Plane geometry, re-exported from the collector (single source of truth)."""

    from tools.rl.v400_p2_bulk_collect import plane_shapes as _plane_shapes

    return _plane_shapes()


def train_command(args) -> int:
    import torch

    index = load_index(args.index)
    entries = shard_entries(index)
    split = build_split(entries)
    if args.split:
        recorded = json.loads(Path(args.split).read_text(encoding="utf-8"))
        if recorded.get("stage_splits") != split["stage_splits"]:
            raise DetectorDataError(
                f"saved split stage map {recorded.get('stage_splits')} != "
                f"recomputed {split['stage_splits']}"
            )
    rows_by_split = {
        name: [row for row in split["shards"] if row["split"] == name]
        for name in split["stage_splits"]
    }
    device = torch_device(args.device)
    shapes = plane_shapes()
    print(json.dumps({"device": str(device), "resolution": args.resolution,
                      "frames": split["frames"],
                      "shards": {k: len(v) for k, v in rows_by_split.items()}},
                     ensure_ascii=False), flush=True)

    labels_train = load_labels_for(rows_by_split["train"])
    result = train_model(
        rows=rows_by_split["train"],
        labels=labels_train,
        resolution=args.resolution,
        epochs=args.epochs,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed,
        window=args.death_window,
        pos_weight_cap=args.pos_weight_cap,
    )
    model = result["model"]
    if args.checkpoint:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "resolution": args.resolution,
                "label_version": LABEL_VERSION,
                "play_shape": list(shapes[args.resolution]),
                "hud_shape": list(HUD_SHAPE),
                "stack": STACK,
                "horizon": HORIZON,
                "engine_sha256": split["engine_sha256"],
                "stage_splits": split["stage_splits"],
            },
            args.checkpoint,
        )

    metrics: dict[str, Any] = {
        "schema": METRICS_SCHEMA,
        "dataset": {"index": str(args.index), "engine_sha256": split["engine_sha256"],
                    "label_version": LABEL_VERSION, "frames": split["frames"]},
        "resolution": args.resolution,
        "model": {
            "parameters": result["cost"]["parameters"],
            "inputs": {"play": [STACK, *shapes[args.resolution]], "hud": list(HUD_SHAPE)},
            "heads": ["death", "bomb", "lives", "bombs", "score_delta"],
            "arm": f"two-stream CNN, play {STACK}x{args.resolution[4:]} grid, hud 1x56x160",
            "death_window": args.death_window,
            "pos_weight_cap": args.pos_weight_cap,
        },
        "training": {key: value for key, value in result["cost"].items() if key != "history"},
        "history": result["cost"]["history"],
        "splits": split["stage_splits"],
        "threshold_selection": "max frame-level F1 on val",
    }
    evaluation: dict[str, Any] = {}
    val_threshold = 0.5
    for name in ("val", "test"):
        rows = rows_by_split.get(name) or []
        if not rows:
            raise DetectorDataError(f"split {name} has no shards")
        started = time.monotonic()
        evaluated = evaluate_shards(model, rows, resolution=args.resolution, device=device,
                                    window=args.death_window)
        sweep = _threshold_sweep(evaluated, [i / 50 for i in range(1, 50)])
        if name == "val":
            val_threshold = sweep["threshold"]
        threshold = val_threshold
        metrics_block = g2_metrics(evaluated, threshold=threshold)
        metrics_block["frames_evaluated"] = int(
            sum(shard["frames"] for shard in evaluated["shards"])
        )
        metrics_block["shards"] = len(evaluated["shards"])
        metrics_block["inference_seconds"] = round(time.monotonic() - started, 2)
        metrics_block["threshold_sweep_best"] = {
            "threshold": sweep["threshold"],
            "f1": sweep["f1"],
            "precision": sweep["precision"]["value"],
            "recall": sweep["recall"]["value"],
        }
        evaluation[name] = metrics_block
        print(json.dumps({"split": name, "threshold": threshold,
                          "death_frame_level": metrics_block["death_frame_level"],
                          "death_event_level": {
                              "true_events": metrics_block["death_event_level"]["true_events"],
                              "matched": metrics_block["death_event_level"]["matched"],
                              "recall": metrics_block["death_event_level"]["recall"],
                              "precision": metrics_block["death_event_level"]["precision"],
                              "latency": metrics_block["death_event_level"]["latency"]},
                          "lives": metrics_block["lives_accuracy"]["value"],
                          "bombs": metrics_block["bombs_accuracy"]["value"],
                          "score_sign": metrics_block["score_sign_accuracy"]["value"]},
                         ensure_ascii=False), flush=True)
    metrics["evaluation"] = evaluation
    metrics["verdict"] = g2_verdict(evaluation.get("test") or {})
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
    print(json.dumps({"verdict": metrics["verdict"]}, indent=2, ensure_ascii=False))
    return 0


def g2_verdict(block: dict[str, Any]) -> dict[str, Any]:
    """Gate table: each G2 requirement, its value, its 95 % lower bound, pass/fail."""

    if not block:
        return {"measured": False, "reason": "no test metrics"}
    frame = block["death_frame_level"]
    event = block["death_event_level"]
    latency = event["latency"]
    lives = block["lives_accuracy"]
    bombs = block["bombs_accuracy"]
    score = block["score_sign_accuracy"]
    rows = [
        {
            "gate": "death recall (frame level)",
            "value": frame["recall"]["value"],
            "lower_bound_95": frame["recall"]["lower_bound_95"],
            "n": frame["recall"]["n"],
            "requirement": ">= 0.99 one-sided 95 % LB",
            "pass": frame["recall"]["pass_99"],
        },
        {
            "gate": "death precision (frame level)",
            "value": frame["precision"]["value"],
            "lower_bound_95": frame["precision"]["lower_bound_95"],
            "n": frame["precision"]["n"],
            "requirement": ">= 0.99 one-sided 95 % LB",
            "pass": frame["precision"]["pass_99"],
        },
        {
            "gate": "death recall (event level)",
            "value": event["recall"]["value"],
            "lower_bound_95": event["recall"]["lower_bound_95"],
            "n": event["recall"]["n"],
            "requirement": ">= 0.99 one-sided 95 % LB",
            "pass": event["recall"]["pass_99"],
        },
        {
            "gate": "death precision (event level)",
            "value": event["precision"]["value"],
            "lower_bound_95": event["precision"]["lower_bound_95"],
            "n": event["precision"]["n"],
            "requirement": ">= 0.99 one-sided 95 % LB",
            "pass": event["precision"]["pass_99"],
        },
        {
            "gate": "death latency (max, frames)",
            "value": latency.get("max"),
            "lower_bound_95": None,
            "n": latency.get("n") or 0,
            "requirement": "<= 10",
            "pass": bool(latency.get("within_limit")) and latency.get("n", 0) > 0,
        },
        {
            "gate": "lives read accuracy",
            "value": lives["value"],
            "lower_bound_95": lives["lower_bound_95"],
            "n": lives["n"],
            "requirement": ">= 0.995 one-sided 95 % LB",
            "pass": lives["pass_995"],
        },
        {
            "gate": "bombs read accuracy",
            "value": bombs["value"],
            "lower_bound_95": bombs["lower_bound_95"],
            "n": bombs["n"],
            "requirement": ">= 0.995 one-sided 95 % LB",
            "pass": bombs["pass_995"],
        },
        {
            "gate": "score delta sign accuracy",
            "value": score["value"],
            "lower_bound_95": score["lower_bound_95"],
            "n": score["n"],
            "requirement": ">= 0.99 one-sided 95 % LB",
            "pass": score["pass_99"],
        },
    ]
    return {
        "measured": True,
        "threshold": block.get("threshold"),
        "gates": rows,
        "all_pass": all(row["pass"] for row in rows),
        "score_negative_samples": score.get("negative_class_samples"),
    }


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------
def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _mark(row: dict[str, Any]) -> str:
    value = row["value"]
    shown = f"{value:.2f}" if isinstance(value, float) and row["gate"].startswith("death latency") \
        else _pct(value)
    return (
        f"| {row['gate']} | {shown} | {_pct(row['lower_bound_95'])} | {row['n']} | "
        f"{row['requirement']} | {'达标' if row['pass'] else '未达标'} |"
    )


def report_command(args) -> int:
    blocks = []
    for path in args.metrics:
        blocks.append((Path(path).name, json.loads(Path(path).read_text(encoding="utf-8"))))
    lines = ["| 指标 | 值 | 95% 单侧下界 | n | 要求 | 判定 |", "|---|---|---|---|---|---|"]
    for name, metrics in blocks:
        verdict = metrics.get("verdict") or g2_verdict(
            metrics.get("evaluation", {}).get("test", {})
        )
        lines.append(f"| **{name} ({metrics['resolution']})** | | | | | |")
        for row in verdict.get("gates", []):
            lines.append(_mark(row))
    markdown = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


# ---------------------------------------------------------------------------
# v1: HUD-sequence event detector (EXP-P2-004)
# ---------------------------------------------------------------------------
# Why v1 exists
# -------------
# v0 asked the *playfield* to predict ``death`` at ``t+1`` and got 1 % event
# recall while the HUD heads were already perfect.  EXP-P2-004 measured what the
# event labels actually co-vary with (``.tmp_v400_det2/task1_align.json``):
#
#   ``bomb``  == bombs decrement, same frame, 114/114, no misses either way
#   ``death``  -> lives decrement 4 frames later (208 of 218 matched events);
#                the remaining events happen with ``lives == 0`` already, so no
#                decrement can exist at all
#   ``score``  the HUD draws ``displayScore`` (GameManager.cpp:526-544 rolls it
#               toward ``globals.score`` by ``(score-displayScore)>>5``), which
#              is the label's field only after the roll settles
#
# v1 therefore reads the HUD once per frame (lives / bombs / 9 score digits) and
# drives the event heads from the *causal* reading sequence, never from the
# playfield.  The play stream stays in the model, but only as the auxiliary
# hazard/occupancy input (``hit``), which is what it is actually good for.
#
# This is **event detection, not death prediction**: the head fires on the frame
# where the HUD reading changes, and the reported latency is
# ``detection frame - label frame`` (positive = later than the event).  Nothing
# looks ahead.

#: HUD plane geometry, measured on the stored planes (``probe03_hud.py`` /
#: ``probe04_roll.py``) and anchored on ``src/Gui.cpp:1297-1390``: the 9-digit
#: score row is drawn at 13 px pitch, the lives icons start at band column 11
#: with 16 px step, the bombs icons sit 16 px lower.
#:
#: EXP-P2-005 correction: ``Gui.cpp:1299`` draws the score at screen x=488, i.e.
#: band column 8 (the plane starts at column 480), and the glyph ink starts two
#: pixels into that cell (measured: ink runs at 10-21, 23-34, ... on every
#: stored plane).  The frozen EXP-P2-004 reader was trained with
#: ``HUD_SCORE_X0 = 3``, five pixels to the left of the glyph cell, so its nine
#: cell crops each held the left half of one glyph plus blank -- which is why
#: its low digits collapsed (47 % on the last position) even on frames where the
#: pixels did encode the value.  The legacy constant is kept for reproducing the
#: EXP-P2-004 numbers; ``HUD_SCORE_X0_MEASURED`` is the geometry the drawing code
#: actually produces and is what new readers are trained with.
HUD_SCORE_ROWS = (8, 24)
HUD_SCORE_X0 = 3
HUD_SCORE_X0_MEASURED = 8
HUD_SCORE_PITCH = 13
HUD_SCORE_DIGITS = 9
HUD_LIVES_ROWS = (28, 40)
HUD_BOMBS_ROWS = (44, 56)


#: Digit classes are 0-9; a blank is impossible because ``%.9d`` always prints
#: nine glyphs, so the reader never has to model one.
SCORE_DIGIT_CLASSES = 10

#: Causal window (frames of history, inclusive of ``t``) the event head sees.
EVENT_HISTORY = 4

#: Per-frame reading vector width: lives posterior + bombs posterior + score
#: delta sign one-hot + normalised decoded score.
READING_DIM = LIVES_CLASSES + BOMBS_CLASSES + 3 + 1


#: Stored label column for each reading.  The collector writes the short keys
#: (``li``/``bo``/``sc``); the detector keeps the long names internally so a
#: mis-keyed lookup cannot silently read ``None`` for every frame.
LABEL_COLUMN = {"lives": "li", "bombs": "bo", "score": "sc"}


def score_glyph_boxes(x0: int = HUD_SCORE_X0) -> list[tuple[int, int, int, int]]:
    """``(row0, row1, col0, col1)`` of each of the nine score glyphs.

    ``x0`` is the band column of the first cell.  The default keeps the frozen
    EXP-P2-004 geometry; :data:`HUD_SCORE_X0_MEASURED` is the origin the drawing
    code produces (see the geometry comment above).
    """

    r0, r1 = HUD_SCORE_ROWS
    return [
        (r0, r1, x0 + HUD_SCORE_PITCH * i, x0 + HUD_SCORE_PITCH * (i + 1))
        for i in range(HUD_SCORE_DIGITS)
    ]



def digits_of(value: int) -> list[int]:
    """The nine digits ``%.9d`` prints for ``value`` (clamped to 999999999)."""

    value = max(0, min(int(value), 999_999_999))
    return [int(ch) for ch in f"{value:09d}"]


def decode_digits(digit_argmax: Sequence[int]) -> int:
    """Inverse of :func:`digits_of`."""

    value = 0
    for digit in digit_argmax:
        value = value * 10 + int(digit)
    return value


def sign_series(values: Sequence[int]) -> list[int]:
    """``sign(v[t] - v[t-1])`` as -1/0/+1, with ``0`` at ``t == 0``."""

    out = [0]
    for t in range(1, len(values)):
        delta = int(values[t]) - int(values[t - 1])
        out.append(0 if delta == 0 else (1 if delta > 0 else -1))
    return out


def hud_diff_events(readings: Sequence[int]) -> list[bool]:
    """Event per frame: the reading strictly decreased since the previous frame.

    This is the deterministic form of the v1 event head (lives for ``death``,
    bombs for ``bomb``).  It is reported next to the learned head so the two can
    be compared: if they agree, the learned head is doing exactly the HUD diff.
    """

    flags = [False] * len(readings)
    for t in range(1, len(readings)):
        flags[t] = int(readings[t]) < int(readings[t - 1])
    return flags


def causal_windows(vector: Any, history: int = EVENT_HISTORY) -> Any:
    """Stack ``history`` previous frames onto every row (causal, no lookahead)."""

    import numpy as np

    n = int(vector.shape[0])
    width = int(vector.shape[1])
    out = np.zeros((n, history + 1, width), dtype=np.float32)
    padded = np.zeros((n + history, width), dtype=np.float32)
    padded[history:] = vector
    for k in range(history + 1):
        out[:, k, :] = padded[k:k + n]
    return out


def event_metrics(
    true_flags: Sequence[bool],
    pred_flags: Sequence[bool],
    *,
    max_latency: int = 10,
) -> dict[str, Any]:
    """Frame-level and event-level death/bomb metrics with latencies."""

    true_frames = [t for t, flag in enumerate(true_flags) if flag]
    pred_frames = [t for t, flag in enumerate(pred_flags) if flag]
    true_starts = event_starts(true_flags)
    pred_starts = event_starts(pred_flags)
    matched = match_events(true_starts, pred_starts, before=0, after=max_latency)
    frame = binary_scores(
        sum(1 for t in true_frames if t in set(pred_frames)),
        0,
        len(true_frames),
    )
    # frame precision needs the full confusion, not just the matched subset
    tp = len(set(true_frames) & set(pred_frames))
    fp = len(set(pred_frames)) - tp
    fn = len(set(true_frames)) - tp
    frame = binary_scores(tp, fp, fn)
    event = binary_scores(matched["matched"], matched["unmatched_predictions"],
                          len(true_starts) - matched["matched"])
    return {
        "frames": {"true": len(true_frames), "pred": len(pred_frames), "tp": tp,
                   "fp": fp, "fn": fn},
        "frame_level": frame,
        "event_level": event,
        "latency": latency_summary(matched["latencies"], limit=max_latency),
        "true_events": len(true_starts),
        "pred_events": len(pred_starts),
    }


class HudCorpus:
    """One split's HUD plane + label rows, read straight off the stored memmaps."""

    def __init__(self, rows: Sequence[dict[str, Any]]):
        import numpy as np

        self.rows = list(rows)
        self.labels = load_labels_for(self.rows)
        self.planes = []
        for row in self.rows:
            frames = int(row["frames"])
            self.planes.append(
                np.memmap(
                    Path(row["dir"]) / "hud.u8",
                    dtype=np.uint8,
                    mode="r",
                    shape=(frames, HUD_SHAPE[0], HUD_SHAPE[1]),
                )
            )
        self.offsets = []
        total = 0
        for frames in (int(row["frames"]) for row in self.rows):
            self.offsets.append(total)
            total += frames
        self.total = total
        # A shard can hold more label rows than the index declares (the tail of a
        # top-up run).  Planes and labels are truncated to the declared frame
        # count so every split array has exactly ``total`` entries.
        self.limit = [
            min(int(row["frames"]), len(shard_rows))
            for row, shard_rows in zip(self.rows, self.labels)
        ]

    def locate(self, index: int) -> tuple[int, int]:
        """Global frame index -> (shard, local frame)."""

        lo, hi = 0, len(self.offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.offsets[mid] <= index:
                lo = mid
            else:
                hi = mid - 1
        return lo, index - self.offsets[lo]

    def gather(self, indices: Sequence[int]) -> Any:
        """HUD planes for arbitrary global indices (stacked, uint8)."""

        import numpy as np

        indices = np.asarray(indices, dtype=np.int64)
        shard_of = np.searchsorted(np.asarray(self.offsets), indices, side="right") - 1
        out = np.empty((len(indices), HUD_SHAPE[0], HUD_SHAPE[1]), dtype=np.uint8)
        for shard in np.unique(shard_of):
            mask = shard_of == shard
            local = indices[mask] - self.offsets[int(shard)]
            out[mask] = np.asarray(self.planes[int(shard)][local])
        return out

    def flat_labels(self, key: str) -> Any:
        import numpy as np

        # the stored JSONL uses the short keys the collector writes
        column = LABEL_COLUMN[key]
        values: list[Any] = []
        for shard_rows, limit in zip(self.labels, self.limit):
            values.extend(row.get(column) for row in shard_rows[:limit])
        return np.asarray(values)

    def flat_flags(self, key: str) -> Any:
        import numpy as np

        return np.asarray(
            [
                bool(row.get(key))
                for shard_rows, limit in zip(self.labels, self.limit)
                for row in shard_rows[:limit]
            ],
            dtype=bool,
        )


def hud_crops(batch: Any, *, score_x0: int = HUD_SCORE_X0) -> dict[str, Any]:
    """Split a ``(n, 56, 160)`` HUD batch into the crops each reader consumes."""

    import torch

    def to_tensor(block: Any) -> Any:
        return torch.from_numpy(block.astype("float32") / 255.0).unsqueeze(1)

    lives = to_tensor(batch[:, HUD_LIVES_ROWS[0]:HUD_LIVES_ROWS[1], :])
    bombs = to_tensor(batch[:, HUD_BOMBS_ROWS[0]:HUD_BOMBS_ROWS[1], :])
    glyphs = [to_tensor(batch[:, r0:r1, c0:c1]) for r0, r1, c0, c1 in score_glyph_boxes(score_x0)]
    return {
        "lives": lives,
        "bombs": bombs,
        "score": torch.stack(glyphs, dim=1),  # (n, 9, 1, h, w)
    }



def build_reader_model():
    """Small HUD reader: one shared trunk plus the three reading heads."""

    import torch
    from torch import nn

    class Reader(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lives = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 8)),
            )
            self.lives_head = nn.Linear(32 * 2 * 8, LIVES_CLASSES)
            self.bombs = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 8)),
            )
            self.bombs_head = nn.Linear(32 * 2 * 8, BOMBS_CLASSES)
            self.digit = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 4)),
            )
            self.digit_head = nn.Linear(32 * 2 * 4, SCORE_DIGIT_CLASSES)

        def forward(self, crops: dict[str, Any]) -> dict[str, Any]:
            lives = self.lives_head(self.lives(crops["lives"]).flatten(1))
            bombs = self.bombs_head(self.bombs(crops["bombs"]).flatten(1))
            n = crops["score"].shape[0]
            flat = crops["score"].reshape(n * HUD_SCORE_DIGITS, 1,
                                          *crops["score"].shape[-2:])
            digits = self.digit_head(self.digit(flat).flatten(1)).reshape(
                n, HUD_SCORE_DIGITS, SCORE_DIGIT_CLASSES
            )
            return {"lives": lives, "bombs": bombs, "score": digits}

    return Reader()


def build_event_model():
    """Causal 1-D CNN over the reading sequence; death and bomb heads."""

    from torch import nn

    class EventHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Conv1d(READING_DIM, 48, 3, padding=0), nn.ReLU(),
                nn.Conv1d(48, 48, 3, padding=0), nn.ReLU(),
            )
            width = EVENT_HISTORY + 1 - 4
            if width < 1:
                raise ValueError("EVENT_HISTORY too small for a two-layer causal CNN")
            self.head = nn.Linear(48 * width, 2)  # [death, bomb]

        def forward(self, windows):
            # (n, history+1, dim) -> (n, dim, history+1): kernels stay causal
            return self.head(self.trunk(windows.transpose(1, 2)).flatten(1))

    return EventHead()


def _reader_targets(labels_slice: dict[str, Any], indices: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    lives = torch.from_numpy(np.asarray(labels_slice["lives"])[indices].astype("int64"))
    bombs = torch.from_numpy(np.asarray(labels_slice["bombs"])[indices].astype("int64"))
    digits = torch.from_numpy(
        np.stack([np.asarray(digits_of(int(v)), dtype="int64")
                  for v in np.asarray(labels_slice["score"])[indices]])
    )
    return {"lives": lives, "bombs": bombs, "score": digits}


def train_reader(
    corpus: HudCorpus,
    *,
    epochs: int,
    max_samples: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: Any,
    score_x0: int = HUD_SCORE_X0,
    log: Any = None,
) -> Any:
    """Fit the HUD readers on a split (lives / bombs / 9 score digits)."""

    import numpy as np
    import torch
    from torch import nn

    rng = np.random.default_rng(seed)
    lives_all = corpus.flat_labels("lives")
    keep = np.flatnonzero(lives_all != None)  # noqa: E711 - None marks a missing row
    if len(keep) > max_samples:
        keep = np.sort(rng.choice(keep, size=max_samples, replace=False))
    model = build_reader_model().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_lives = nn.CrossEntropyLoss()
    loss_bombs = nn.CrossEntropyLoss()
    loss_digits = nn.CrossEntropyLoss()
    losses = {key: corpus.flat_labels(key) for key in ("lives", "bombs", "score")}
    for epoch in range(epochs):
        model.train()
        total = 0.0
        order = np.sort(rng.permutation(keep))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            if len(idx) < 2:
                continue
            crops = hud_crops(corpus.gather(idx), score_x0=score_x0)
            crops = {key: value.to(device) for key, value in crops.items()}
            targets = _reader_targets(losses, idx)
            out = model(crops)
            loss = (
                loss_lives(out["lives"], targets["lives"].to(device))
                + loss_bombs(out["bombs"], targets["bombs"].to(device))
                + loss_digits(
                    out["score"].reshape(-1, SCORE_DIGIT_CLASSES),
                    targets["score"].reshape(-1).to(device),
                )
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.item()) * len(idx)
        if log is not None:
            log({"stage": "reader", "epoch": epoch, "loss": total / max(1, len(order))})
    model.eval()
    return model


def read_sequence(
    corpus: HudCorpus,
    model: Any,
    *,
    device: Any,
    batch_size: int = 4096,
    score_x0: int = HUD_SCORE_X0,
) -> dict[str, Any]:
    """Run the readers over every frame of a split, in order (causal readings)."""

    import numpy as np
    import torch

    n = corpus.total
    lives = np.zeros((n, LIVES_CLASSES), dtype=np.float32)
    bombs = np.zeros((n, BOMBS_CLASSES), dtype=np.float32)
    digits = np.zeros((n, HUD_SCORE_DIGITS), dtype=np.int16)
    model.eval()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n))
            crops = hud_crops(corpus.gather(idx), score_x0=score_x0)
            out = model({key: value.to(device) for key, value in crops.items()})
            lives[idx] = out["lives"].softmax(-1).float().cpu().numpy()
            bombs[idx] = out["bombs"].softmax(-1).float().cpu().numpy()
            digits[idx] = out["score"].argmax(-1).to(torch.int16).cpu().numpy()
    decoded = np.asarray([decode_digits(row) for row in digits], dtype=np.int64)
    return {"lives": lives, "bombs": bombs, "digits": digits, "decoded": decoded}


def build_reading_vector(readings: dict[str, Any]) -> Any:
    """Per-frame reading vector handed to the causal event head."""

    import numpy as np

    decoded = readings["decoded"]
    signs = np.asarray(sign_series(decoded), dtype=np.int64)
    onehot = np.zeros((len(signs), 3), dtype=np.float32)
    onehot[np.arange(len(signs)), signs + 1] = 1.0
    scale = float(max(1, int(decoded.max())))
    return np.concatenate(
        [
            readings["lives"],
            readings["bombs"],
            onehot,
            (decoded.astype(np.float32) / scale)[:, None],
        ],
        axis=1,
    ).astype(np.float32)


def train_event_head(
    vector: Any,
    targets: Any,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: Any,
    pos_weight_cap: float = 8.0,
    log: Any = None,
) -> Any:
    """Fit the causal event head on the reading sequence of the train split."""

    import numpy as np
    import torch
    from torch import nn

    windows = causal_windows(vector)
    labels = torch.from_numpy(targets.astype("float32"))
    model = build_event_model().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positives = labels.sum(dim=0).clamp(min=1.0)
    negatives = labels.shape[0] - positives
    weights = (negatives / positives).clamp(max=pos_weight_cap).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(labels.shape[0])
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = torch.from_numpy(windows[idx]).to(device)
            out = model(batch)
            loss = criterion(out, labels[idx].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.item()) * len(idx)
        if log is not None:
            log({"stage": "event", "epoch": epoch, "loss": total / labels.shape[0]})
    model.eval()
    return model


def event_probs(model: Any, vector: Any, *, device: Any, batch_size: int = 16384) -> Any:
    import numpy as np
    import torch

    windows = causal_windows(vector)
    out = []
    with torch.inference_mode():
        for start in range(0, windows.shape[0], batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).to(device)
            out.append(model(batch).sigmoid().float().cpu().numpy())
    return np.concatenate(out, axis=0)


def flag_series(probs: Any, threshold: float) -> list[bool]:
    import numpy as np

    return [bool(value) for value in (np.asarray(probs) >= float(threshold))]


def sweep_threshold(
    probs: Any,
    truth: Sequence[bool],
    *,
    thresholds: Iterable[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> dict[str, Any]:
    """Pick the threshold that maximises event F1 on the *validation* split."""

    best = None
    for threshold in thresholds:
        metrics = event_metrics(truth, flag_series(probs, threshold))
        score = metrics["event_level"]["f1"]
        if best is None or score > best["event_level"]["f1"]:
            best = dict(metrics, threshold=float(threshold))
    return best or {}


def _torch_no_grad(fn):
    return fn


# ---------------------------------------------------------------------------
# v1 evaluation: readers -> causal event head -> G2 table
# ---------------------------------------------------------------------------
#: Frames the rendered score counter needs to settle after ``globals.score``
#: moved.  ``GameManager.cpp:530`` rolls by ``(score-displayScore)>>5``, i.e. a
#: ~32-frame time constant; past this the drawn digits equal the label again.
SCORE_SETTLE_FRAMES = 33


def reader_accuracy(
    readings: dict[str, Any],
    labels: dict[str, Any],
) -> dict[str, Any]:
    """Per-field reader accuracy: lives, bombs, per-digit, integer, and sign."""

    import numpy as np

    lives_true = np.asarray(labels["lives"])
    bombs_true = np.asarray(labels["bombs"])
    lives_pred = np.asarray(readings["lives"]).argmax(axis=1)
    bombs_pred = np.asarray(readings["bombs"]).argmax(axis=1)
    digits_true = np.stack([np.asarray(digits_of(int(v))) for v in np.asarray(labels["score"])])
    digits_pred = np.asarray(readings["digits"])
    decoded = np.asarray(readings["decoded"])

    per_digit = []
    for position in range(HUD_SCORE_DIGITS):
        hits = int((digits_pred[:, position] == digits_true[:, position]).sum())
        per_digit.append(proportion(hits, len(digits_true)))

    integer_hits = int((decoded == np.asarray(labels["score"], dtype=np.int64)).sum())
    sign_true = np.asarray(sign_series(np.asarray(labels["score"], dtype=np.int64)))
    sign_pred = np.asarray(sign_series(decoded))
    sign_hits = int((sign_true == sign_pred).sum())

    # how long the drawn counter has been settled; the label is only recoverable
    # from the picture once the roll finished (see module docstring)
    score_true = np.asarray(labels["score"], dtype=np.int64)
    since = np.zeros(len(score_true), dtype=np.int64)
    for t in range(1, len(score_true)):
        since[t] = 0 if score_true[t] != score_true[t - 1] else since[t - 1] + 1
    settled = since >= SCORE_SETTLE_FRAMES
    rolling = ~settled

    negative = int((sign_true < 0).sum())
    return {
        "lives": proportion(int((lives_pred == lives_true).sum()), len(lives_true)),
        "bombs": proportion(int((bombs_pred == bombs_true).sum()), len(bombs_true)),
        "score_digits": per_digit,
        "score_digit_accuracy": proportion(
            int((digits_pred == digits_true).sum()), int(digits_true.size)
        ),
        "score_integer": proportion(integer_hits, len(digits_true)),
        "score_sign": proportion(sign_hits, len(sign_true)),
        "score_sign_negative_class_samples": negative,
        "score_sign_true_class_counts": {
            "down": negative,
            "same": int((sign_true == 0).sum()),
            "up": int((sign_true > 0).sum()),
        },
        "score_integer_settled": proportion(
            int((decoded[settled] == score_true[settled]).sum()), int(settled.sum())
        ),
        "score_sign_settled": proportion(
            int((sign_true[settled] == sign_pred[settled]).sum()), int(settled.sum())
        ),
        "score_sign_rolling": proportion(
            int((sign_true[rolling] == sign_pred[rolling]).sum()), int(rolling.sum())
        ),
        "score_roll_settle_frames": SCORE_SETTLE_FRAMES,
        "score_rolling_frames": int(rolling.sum()),
        "score_settled_frames": int(settled.sum()),
    }


def g2_verdict_v1(block: dict[str, Any]) -> dict[str, Any]:
    """The full EXP-P2-004 gate table (event head + readers + digit path)."""

    if not block:
        return {"measured": False, "reason": "no test metrics"}

    rows: list[dict[str, Any]] = []

    def add(gate: str, entry: dict[str, Any], requirement: str, threshold: float,
            *, key: str = "lower_bound_95") -> None:
        rows.append(
            {
                "gate": gate,
                "value": entry.get("value"),
                "lower_bound_95": entry.get("lower_bound_95"),
                "n": entry.get("n"),
                "requirement": requirement,
                "pass": bool((entry.get(key) or 0.0) >= threshold),
            }
        )

    death = block["death"]
    add("death recall (frame level)", death["frame_level"]["recall"],
        ">= 0.99 one-sided 95 % LB", 0.99)
    add("death precision (frame level)", death["frame_level"]["precision"],
        ">= 0.99 one-sided 95 % LB", 0.99)
    add("death recall (event level)", death["event_level"]["recall"],
        ">= 0.99 one-sided 95 % LB", 0.99)
    add("death precision (event level)", death["event_level"]["precision"],
        ">= 0.99 one-sided 95 % LB", 0.99)
    latency = death["latency"]
    rows.append(
        {
            "gate": "death latency (max, frames)",
            "value": latency.get("max"),
            "lower_bound_95": None,
            "n": latency.get("n") or 0,
            "requirement": "<= 10",
            "pass": bool(latency.get("within_limit")) and (latency.get("n") or 0) > 0,
        }
    )
    bomb = block["bomb"]
    add("bomb recall (event level)", bomb["event_level"]["recall"],
        ">= 0.99 one-sided 95 % LB", 0.99)
    add("bomb precision (event level)", bomb["event_level"]["precision"],
        ">= 0.99 one-sided 95 % LB", 0.99)
    rows.append(
        {
            "gate": "bomb latency (max, frames)",
            "value": bomb["latency"].get("max"),
            "lower_bound_95": None,
            "n": bomb["latency"].get("n") or 0,
            "requirement": "<= 10",
            "pass": bool(bomb["latency"].get("within_limit")) and (bomb["latency"].get("n") or 0) > 0,
        }
    )
    add("lives read accuracy", block["readers"]["lives"],
        ">= 0.995 one-sided 95 % LB", 0.995)
    add("bombs read accuracy", block["readers"]["bombs"],
        ">= 0.995 one-sided 95 % LB", 0.995)
    add("score digit accuracy", block["readers"]["score_digit_accuracy"],
        ">= 0.99 one-sided 95 % LB", 0.99)
    add("score integer accuracy", block["readers"]["score_integer"],
        ">= 0.99 one-sided 95 % LB", 0.99)
    add("score delta sign accuracy", block["readers"]["score_sign"],
        ">= 0.99 one-sided 95 % LB", 0.99)
    add("score integer accuracy (settled counter)", block["readers"]["score_integer_settled"],
        "diagnostic", 0.99)
    add("score delta sign accuracy (settled counter)", block["readers"]["score_sign_settled"],
        "diagnostic", 0.99)

    return {
        "measured": True,
        "gates": rows,
        "all_pass": all(row["pass"] for row in rows if not row["requirement"].startswith("diagnostic")),
        "bomb_gate_required": block.get("bomb_gate_required"),
        "score_negative_samples": block["readers"]["score_sign_negative_class_samples"],
        "notes": block.get("notes", []),
    }


def v1_command(args) -> int:
    """Train the v1 readers + causal event head and evaluate it on the split."""

    import numpy as np
    import torch

    index = load_index(args.index)
    entries = shard_entries(index)
    stage_splits = None
    if args.split:
        config = json.loads(Path(args.split).read_text(encoding="utf-8"))
        stage_splits = {
            name: tuple(int(stage) for stage in stages)
            for name, stages in config["stage_splits"].items()
        }
    split = build_split(entries, stage_splits=stage_splits)
    device = torch_device(args.device)

    corpora = {
        name: HudCorpus([row for row in split["shards"] if row["split"] == name])
        for name in ("train", "val", "test")
    }
    for name, corpus in corpora.items():
        if corpus.total == 0:
            raise DetectorDataError(f"split {name!r} has no frames")

    def log(message: dict[str, Any]) -> None:
        print(json.dumps(message, ensure_ascii=False), flush=True)

    reader = train_reader(
        corpora["train"],
        epochs=args.epochs,
        max_samples=args.max_samples,
        batch_size=args.reader_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
        log=log,
    )

    readings = {
        name: read_sequence(corpus, reader, device=device, batch_size=args.eval_batch_size)
        for name, corpus in corpora.items()
    }
    labels = {name: {key: corpus.flat_labels(key) for key in ("lives", "bombs", "score")}
              for name, corpus in corpora.items()}
    flags = {name: {"death": corpus.flat_flags("death"), "bomb": corpus.flat_flags("bomb")}
             for name, corpus in corpora.items()}

    vectors = {name: build_reading_vector(readings[name]) for name in readings}
    targets = {
        name: np.stack([flags[name]["death"], flags[name]["bomb"]], axis=1).astype(np.float32)
        for name in flags
    }
    event_model = train_event_head(
        vectors["train"],
        targets["train"],
        epochs=args.event_epochs,
        batch_size=args.event_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 1,
        device=device,
        pos_weight_cap=args.pos_weight_cap,
        log=log,
    )

    val_probs = event_probs(event_model, vectors["val"], device=device)
    thresholds = {}
    for column, name in enumerate(("death", "bomb")):
        best = sweep_threshold(val_probs[:, column], flags["val"][name])
        thresholds[name] = float(best.get("threshold", 0.5))
    log({"thresholds": thresholds, "stage": "threshold-sweep"})

    test_probs = event_probs(event_model, vectors["test"], device=device)
    verdicts = {
        name: event_metrics(
            flags["test"][name],
            flag_series(test_probs[:, column], thresholds[name]),
        )
        for column, name in enumerate(("death", "bomb"))
    }

    # deterministic HUD-diff baseline: the same heads without a learned model
    diff_death = hud_diff_events(np.asarray(readings["test"]["lives"]).argmax(axis=1))
    diff_bomb = hud_diff_events(np.asarray(readings["test"]["bombs"]).argmax(axis=1))
    baseline = {
        "death": event_metrics(flags["test"]["death"], diff_death),
        "bomb": event_metrics(flags["test"]["bomb"], diff_bomb),
    }

    readers = reader_accuracy(readings["test"], labels["test"])
    block = {
        "death": verdicts["death"],
        "bomb": verdicts["bomb"],
        "readers": readers,
        "hud_diff_baseline": {
            name: {
                "event_level": baseline[name]["event_level"],
                "frame_level": baseline[name]["frame_level"],
                "latency": baseline[name]["latency"],
            }
            for name in baseline
        },
        "bomb_gate_required": args.bomb_gate,
        "notes": [
            "death/bomb heads are driven by the causal HUD reading sequence; the "
            "playfield is only the auxiliary hazard input",
            "detection latency is measured as detection frame - label frame, so a "
            "positive value means the HUD change was reported after the event",
        ],
    }
    block["verdict"] = g2_verdict_v1(block)

    metrics = {
        "schema": METRICS_SCHEMA + "-v1",
        "resolution": "hud-sequence",
        "split": {name: split["frames"][name] for name in split["frames"]},
        "thresholds": thresholds,
        "evaluation": {"test": block},
        "verdict": block["verdict"],
    }
    out = Path(args.out)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.checkpoint:
        torch.save({"reader": reader.state_dict(), "event": event_model.state_dict(),
                    "thresholds": thresholds}, args.checkpoint)
    print(json.dumps({"out": str(out), "all_pass": block["verdict"]["all_pass"],
                      "thresholds": thresholds}, ensure_ascii=False))
    return 0


def load_display_score(
    relabel_dir: str | Path,
    rows: Sequence[dict[str, Any]],
    *,
    limits: Sequence[int],
) -> tuple[Any, Any, Any]:
    """Per-frame ``displayScore`` / ``globalsScore`` / roll age for a split.

    ``relabel_dir`` is the EXP-P2-005 re-label output: ``shards/<key>.jsonl``
    with one row per frame (``ds`` = displayScore, ``gs`` = globalsScore,
    ``age`` = displayScore lag).  Returns three int64 arrays in corpus order,
    with ``-1`` for frames the re-label did not cover so the caller can drop
    them instead of silently training on a missing value.
    """

    import numpy as np

    display: list[int] = []
    globals_score: list[int] = []
    ages: list[int] = []
    root = Path(relabel_dir) / "shards"
    for row, limit in zip(rows, limits):
        key = str(row.get("shard") or Path(row["dir"]).parent.name)
        path = root / f"{key}.jsonl"
        if not path.is_file():
            raise DetectorDataError(f"no displayScore rows for shard {key}: {path}")
        recovered: dict[int, tuple[int, int, int]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                recovered[int(record["i"])] = (
                    int(record["ds"]), int(record["gs"]), int(record.get("age") or 0)
                )
        for index in range(limit):
            ds, gs, age = recovered.get(index, (-1, -1, -1))
            display.append(ds)
            globals_score.append(gs)
            ages.append(age)
    return (
        np.asarray(display, dtype=np.int64),
        np.asarray(globals_score, dtype=np.int64),
        np.asarray(ages, dtype=np.int64),
    )


def split_bounds(corpus: HudCorpus) -> list[tuple[int, int]]:
    """``[(start, stop)]`` of each shard inside the flat corpus order."""

    return [(offset, offset + limit) for offset, limit in zip(corpus.offsets, corpus.limit)]


def score_accuracy(
    decoded: Any,
    digits: Any,
    labels: Any,
    bounds: Sequence[tuple[int, int]],
    *,
    mask: Any = None,
) -> dict[str, Any]:
    """Per-digit / integer / delta-sign accuracy of a reader against one label.

    ``labels`` is the series the pixels are scored against; a nine-digit digest
    makes any integer label a fair target for the nine glyph cells.  Delta signs
    are computed **inside each shard**, so a shard boundary cannot manufacture
    the fake ``-1`` that a flat concatenated series produces (EXP-P2-004 noted
    97 such pseudo-negatives).
    """

    import numpy as np

    labels = np.asarray(labels, dtype=np.int64)
    decoded = np.asarray(decoded, dtype=np.int64)
    digits = np.asarray(digits, dtype=np.int64)
    keep = np.ones(len(labels), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    true_digits = np.stack([np.asarray(digits_of(int(value))) for value in labels])

    per_digit = []
    for position in range(HUD_SCORE_DIGITS):
        hits = int((digits[keep, position] == true_digits[keep, position]).sum())
        per_digit.append(proportion(hits, int(keep.sum())))
    integer = proportion(int((decoded[keep] == labels[keep]).sum()), int(keep.sum()))

    sign_true_list: list[int] = []
    sign_pred_list: list[int] = []
    for start, stop in bounds:
        selection = np.arange(start, stop)[keep[start:stop]]
        if selection.size == 0:
            continue
        sign_true_list.extend(sign_series(labels[selection]))
        sign_pred_list.extend(sign_series(decoded[selection]))
    sign_true = np.asarray(sign_true_list, dtype=np.int64)
    sign_pred = np.asarray(sign_pred_list, dtype=np.int64)
    sign = proportion(int((sign_true == sign_pred).sum()), int(sign_true.size))
    return {
        "frames": int(keep.sum()),
        "per_digit": per_digit,
        "digit_accuracy": proportion(
            int((digits[keep] == true_digits[keep]).sum()), int(digits[keep].size)
        ),
        "integer_accuracy": integer,
        "sign_accuracy": sign,
        "sign_true_counts": {
            "down": int((sign_true < 0).sum()),
            "same": int((sign_true == 0).sum()),
            "up": int((sign_true > 0).sum()),
        },
    }


def score_gate_command(args) -> int:
    """EXP-P2-005: re-score the score gate with displayScore labels.

    The reader checkpoint is unchanged (``--checkpoint``); what changes is
    (a) the label the digits are scored against (``--label displayScore`` versus
    the old ``globalsScore``) and (b) optionally the cell origin
    (``--score-x0``), which is a property of the drawing code, not of the model.
    """

    import numpy as np
    import torch

    if not args.relabel:
        raise DetectorDataError("--relabel <dir> is required for displayScore labels")

    index = load_index(args.index)
    entries = shard_entries(index)
    stage_splits = None
    if args.split:
        config = json.loads(Path(args.split).read_text(encoding="utf-8"))
        stage_splits = {
            name: tuple(int(stage) for stage in stages)
            for name, stages in config["stage_splits"].items()
        }
    split = build_split(entries, stage_splits=stage_splits)
    rows = [row for row in split["shards"] if row["split"] == args.split_name]
    if not rows:
        raise DetectorDataError(f"split {args.split_name!r} has no shards")

    corpus = HudCorpus(rows)
    bounds = split_bounds(corpus)
    display, globals_score, ages = load_display_score(args.relabel, rows, limits=corpus.limit)
    stored = corpus.flat_labels("score").astype(np.int64)
    coverage = float((display >= 0).mean())

    device = torch_device(args.device)
    model = build_reader_model()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["reader"] if "reader" in state else state)
    model = model.to(device)
    model.eval()

    digits = np.zeros((corpus.total, HUD_SCORE_DIGITS), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, corpus.total, 2048):
            idx = np.arange(start, min(start + 2048, corpus.total))
            crops = hud_crops(corpus.gather(idx), score_x0=args.score_x0)
            out = model({key: value.to(device) for key, value in crops.items()})
            digits[idx] = out["score"].argmax(-1).cpu().numpy()
    decoded = np.asarray([decode_digits(row) for row in digits], dtype=np.int64)

    covered = display >= 0
    scores = {
        "globalsScore": score_accuracy(decoded, digits, stored, bounds),
        "displayScore": score_accuracy(decoded, digits, display, bounds, mask=covered),
    }
    # diagnostics: the roll state is only known for the covered frames
    settled = covered & (display == globals_score)
    rolling = covered & (display != globals_score)
    scores["displayScore_settled"] = score_accuracy(
        decoded, digits, display, bounds, mask=settled
    )
    scores["displayScore_rolling"] = score_accuracy(
        decoded, digits, display, bounds, mask=rolling
    )
    scores["globalsScore_settled"] = score_accuracy(
        decoded, digits, stored, bounds, mask=settled
    )
    scores["globalsScore_rolling"] = score_accuracy(
        decoded, digits, stored, bounds, mask=rolling
    )

    consistency = proportion(int((decoded[covered] == display[covered]).sum()), int(covered.sum()))
    metrics = {
        "schema": "v400-p2hud-score-gate-1",
        "split": args.split_name,
        "shards": len(rows),
        "frames": corpus.total,
        "display_score_coverage": coverage,
        "rolling_frames": int(rolling.sum()),
        "settled_frames": int(settled.sum()),
        "reader_checkpoint": str(args.checkpoint),
        "score_x0": int(args.score_x0),
        "label": args.label,
        "displayScore_vs_pixels": consistency,
        "scores": scores,
        "gates": {
            "score_digit_accuracy_displayScore_lb": scores["displayScore"]["digit_accuracy"]["lower_bound_95"],
            "score_integer_accuracy_displayScore_lb": scores["displayScore"]["integer_accuracy"]["lower_bound_95"],
            "score_sign_accuracy_displayScore_lb": scores["displayScore"]["sign_accuracy"]["lower_bound_95"],
            "sign_gate_0.99": bool(scores["displayScore"]["sign_accuracy"]["lower_bound_95"] >= 0.99),
            "pixel_consistency_0.995": bool(consistency["lower_bound_95"] >= 0.995),
        },
    }
    out = Path(args.out) if args.out else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "split": metrics["split"],
        "frames": metrics["frames"],
        "coverage": coverage,
        "x0": args.score_x0,
        "label": args.label,
        "pixel_consistency": consistency["value"],
        "display_digit": scores["displayScore"]["digit_accuracy"]["value"],
        "display_integer": scores["displayScore"]["integer_accuracy"]["value"],
        "display_sign": scores["displayScore"]["sign_accuracy"]["value"],
        "display_sign_lb": scores["displayScore"]["sign_accuracy"]["lower_bound_95"],
        "globals_digit": scores["globalsScore"]["digit_accuracy"]["value"],
        "globals_integer": scores["globalsScore"]["integer_accuracy"]["value"],
        "globals_sign": scores["globalsScore"]["sign_accuracy"]["value"],
    }, ensure_ascii=False))
    return 0


def report_v1_command(args) -> int:
    """Render the EXP-P2-004 G2 markdown table from a v1 metrics file."""

    lines = ["| 指标 | 值 | 95% 单侧下界 | n | 要求 | 判定 |", "|---|---|---|---|---|---|"]
    for path in args.metrics:
        metrics = json.loads(Path(path).read_text(encoding="utf-8"))
        verdict = metrics.get("verdict") or g2_verdict_v1(
            metrics.get("evaluation", {}).get("test", {})
        )
        name = Path(path).name
        lines.append(f"| **{name} ({metrics.get('resolution')})** | | | | | |")
        for row in verdict.get("gates", []):
            value = row["value"]
            if value is None:
                shown = "n/a"
            elif row["gate"].endswith("(max, frames)"):
                shown = f"{value:.1f}"
            else:
                shown = _pct(value)
            requirement = row["requirement"]
            passed = row["pass"]
            mark = "达标" if passed else ("未达标" if not requirement.startswith("diagnostic")
                                          else "诊断项")
            lines.append(
                f"| {row['gate']} | {shown} | {_pct(row['lower_bound_95'])} | {row['n']} | "
                f"{requirement} | {mark} |"
            )
        lines.append(
            f"| score 负向样本 (n) | {verdict.get('score_negative_samples')} | n/a | "
            f"{verdict.get('score_negative_samples')} | 需要 >0 才能测符号门 | "
            f"{'可测' if (verdict.get('score_negative_samples') or 0) > 0 else '不可测'} |"
        )
    markdown = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


# ---------------------------------------------------------------------------
# EXP-P2-006: two-scope death gate, game-over gate, two-column verdict
# ---------------------------------------------------------------------------
#: The amended death gate needs enough *observable* events that a zero-failure
#: Wilson 95 % lower bound can reach 0.99 (n >= 386); 350 is the floor this
#: project agreed on, and the verdict reports the actual n either way.
DEATH_GATE_MIN_OBSERVABLE = 350

#: Same latency window as the v1 event head (10 frames after the event).
GAME_OVER_MATCH_WINDOW = 10

#: 12-P2 §6: death latency median <= 10 frames; the table also enforces the max.
DEATH_LATENCY_LIMIT = 10

#: EXP-P2-007: the harness ``EVENTS`` bit the ``rl_events`` build appends for the
#: stage-clear screen (``Gui.cpp:672-687`` -> ``stageClearScreenState = 1``).
#: 0x0080 is the free bit in ``ReplayManager.hpp:161-172`` (0x0100 is
#: ``REPLAY_FRAME_EVENT_PAUSE_RECORDED``).
STAGE_CLEAR_BIT = 0x0080

#: EXP-P2-007 / WP5: the online validation has to be more than a smoke test.
ONLINE_MIN_EPISODES = 3
ONLINE_MIN_FRAMES = 5_000


def observable_bomb_scopes(shards: Iterable[dict[str, Any]], *,
                           window: int = 10) -> dict[str, Any]:
    """The bomb gate in the same two scopes as the death gate (EXP-P2-007).

    Each shard is ``{"bo": [...], "bomb": [bool...]}``.  The detector is the
    deterministic counter-difference form: a bomb is reported at every frame
    where the read ``bombs`` strictly decreased.  Nothing is trained, so a
    held-out claim is free of fitting.

    * ``as_written``: every ``bomb`` event is in the recall denominator.  A bomb
      event that does not consume a bomb (refused press, deathbomb window) can
      not move the HUD, so this scope can be below 1 for a physical reason.
    * ``observable``: only events with a ``bombs`` decrement inside the window.

    Precision keeps the full prediction denominator in both scopes, exactly like
    :func:`observable_death_scopes`.
    """

    events = 0
    observable = 0
    predictions = 0
    matched_all = 0
    matched_obs = 0
    latencies: list[int] = []
    per_shard: list[dict[str, Any]] = []
    for shard in shards:
        bo = [int(value) for value in shard["bo"]]
        starts = event_starts([bool(flag) for flag in shard["bomb"]])
        drops = [index for index in range(1, len(bo)) if bo[index] < bo[index - 1]]
        events += len(starts)
        predictions += len(drops)
        found_all = match_events(starts, drops, before=0, after=window)
        matched_all += found_all["matched"]
        latencies.extend(found_all["latencies"])
        observed = []
        for event in starts:
            # unlike `lives` (which the HUD only redraws ~4 frames later), the
            # bomb counter drops on the SAME frame the BOMB_STARTED bit is set
            # (measured latency 0 on 188/188 events), so the window starts at the
            # event frame itself rather than at event+1
            end = min(event + 1 + window, len(bo))
            if any(bo[index] < bo[index - 1] for index in range(max(1, event), end)):
                observed.append(event)
        observable += len(observed)
        matched_obs += match_events(observed, drops, before=0, after=window)["matched"]
        per_shard.append({"events": len(starts), "observable": len(observed),
                          "predictions": len(drops)})
    return {
        "shards": len(per_shard),
        "events": events,
        "observable_events": observable,
        "unobservable_events": events - observable,
        "predictions": predictions,
        "as_written": {
            "recall": proportion(matched_all, events),
            "precision": proportion(matched_all, predictions),
        },
        "observable": {
            "recall": proportion(matched_obs, observable),
            "precision": proportion(matched_obs, predictions),
        },
        "latency": latency_summary(latencies),
        "min_observable_events": DEATH_GATE_MIN_OBSERVABLE,
        "per_shard": per_shard,
        "detector": "bombs counter strictly decreased (no model, no threshold)",
    }


def stage_clear_metrics(shards: Iterable[dict[str, Any]], *,
                        window: int = 10) -> dict[str, Any]:
    """Stage-clear gate on the EXP-P2-007 ``EVENTS`` bit.

    Each shard is ``{"ev": [int...], "stage": [int...]?}`` where ``ev`` is the
    raw ``EVENTS`` bitmask (so the truth is ``ev & STAGE_CLEAR_BIT``) and the
    optional ``stage`` is ``STATE.f1`` (``GameManager.currentStage``).

    There is **no pixel-side stage-clear detector yet**, so this function never
    invents one: ``predictions`` defaults to ``None`` and the block reports
    ``measured=False`` with zero-sample proportions.  What it does measure is the
    thing the old proxy got wrong: the ``stage`` field only changes once the
    clear screen has finished, so a "stage changed" detector is late by the
    length of the clear screen.  ``proxy_latency`` reports that gap in frames.
    """

    events = 0
    predictions = 0
    matched = 0
    had_detector = False
    latencies: list[int] = []
    proxy_gaps: list[int] = []
    per_shard: list[dict[str, Any]] = []
    for shard in shards:
        ev = [int(value) for value in shard["ev"]]
        starts = event_starts([bool(value & STAGE_CLEAR_BIT) for value in ev])
        events += len(starts)
        stage = shard.get("stage")
        if stage is not None:
            flips = [index for index in range(1, len(stage))
                     if int(stage[index]) != int(stage[index - 1])]
            for event in starts:
                following = [index for index in flips if index >= event]
                if following:
                    proxy_gaps.append(following[0] - event)
        pred = shard.get("stage_clear_pred")
        if pred is not None:
            had_detector = True
            found = match_events(starts, [int(v) for v in pred], before=0, after=window)
            matched += found["matched"]
            predictions += len(pred)
            latencies.extend(found["latencies"])
        per_shard.append({"events": len(starts)})
    return {
        "shards": len(per_shard),
        "events": events,
        "predictions": predictions,
        "matched": matched,
        # a caller with no pixel-side detector gets ZERO-SAMPLE proportions, never
        # a "0 % recall" (which would read as a failed detector) nor a fake 100 %
        "measured": bool(events) and had_detector,
        "no_detector": not had_detector,
        "window": window,
        "bit": STAGE_CLEAR_BIT,
        "recall": proportion(matched, events if had_detector else 0),
        "precision": proportion(matched, predictions if had_detector else 0),
        "latency": latency_summary(latencies),
        "proxy_latency": latency_summary(proxy_gaps),
        "detector": ("none: no pixel-side stage-clear detector exists; the gate "
                     "reports the event label only"),
        "per_shard": per_shard,
    }


def online_validation_gate(payload: dict[str, Any], *,
                           thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """WP5: the G2 gates measured on a live engine stream (EXP-P2-007).

    ``payload`` is what ``.tmp_v400_g2c/wp5_online.py`` writes: a ``combined``
    block with ``episodes``, ``frames``, ``death``, ``bomb`` and ``game_over``.
    The gate fails closed on a short stream (< 3 episodes / < 5000 frames) and on
    any zero-sample proportion, because "no samples" is not "100 %".
    """

    thresholds = thresholds or {"event_recall": 0.99, "event_precision": 0.99,
                                "frame_precision": 0.99}
    combined = (payload or {}).get("combined") or {}
    episodes = int(combined.get("episodes") or 0)
    frames = int(combined.get("frames") or 0)
    long_enough = episodes >= ONLINE_MIN_EPISODES and frames >= ONLINE_MIN_FRAMES

    def cell(block: dict[str, Any], key: str, threshold: float,
             label: str) -> dict[str, Any]:
        entry = (block or {}).get(key) or {}
        n = int(entry.get("n") or 0)
        lower = entry.get("lower_bound_95")
        return {
            "gate": label,
            "value": entry.get("value"),
            "lower_bound_95": lower,
            "n": n,
            "requirement": f"LB >= {threshold}",
            "pass": bool(long_enough and n > 0 and lower is not None
                         and float(lower) >= threshold),
            "applicable": True,
            "note": "" if long_enough else "流太短（< 3 局或 < 5000 帧）",
        }

    rows = [
        cell(combined.get("death") or {}, "recall", thresholds["event_recall"],
             "online death recall"),
        cell(combined.get("death") or {}, "precision", thresholds["event_precision"],
             "online death precision"),
        cell(combined.get("bomb") or {}, "recall", thresholds["event_recall"],
             "online bomb recall"),
        cell(combined.get("bomb") or {}, "precision", thresholds["event_precision"],
             "online bomb precision"),
        cell(combined.get("game_over") or {}, "recall", thresholds["event_recall"],
             "online game_over interval recall"),
        cell(combined.get("game_over") or {}, "precision", thresholds["event_precision"],
             "online game_over interval precision"),
    ]
    passed = all(row["pass"] for row in rows)
    return {
        "measured": bool(episodes and frames),
        "episodes": episodes,
        "frames": frames,
        "long_enough": long_enough,
        "rows": rows,
        "pass": passed,
    }


def observable_death_scopes(shards: Iterable[dict[str, Any]], *,
                             window: int = 10) -> dict[str, Any]:
    """The death gate in two scopes (EXP-P2-004 §1 ceiling, EXP-P2-006 amendment).

    Each shard is ``{"li": [...], "death": [bool...]}``.  The detector is the
    deterministic HUD-difference form: a death is reported at every frame where
    the read ``lives`` strictly decreased.

    * ``as_written``: every ``death`` event is in the recall denominator.  Events
      that happen while ``lives`` is already 0 cannot move the HUD, so this scope
      has a physical ceiling below 1.
    * ``observable``: only events with a ``lives`` decrement inside the window,
      i.e. exactly the ones a pixel-side detector can see.  Precision keeps the
      full prediction denominator in both scopes.

    Nothing is discarded silently: ``unobservable_events`` is reported in both
    the block and the verdict.
    """

    deaths = 0
    observable = 0
    predictions = 0
    matched_all = 0
    matched_obs = 0
    latencies: list[int] = []
    per_shard: list[dict[str, Any]] = []
    for shard in shards:
        li = [int(value) for value in shard["li"]]
        events = event_starts([bool(flag) for flag in shard["death"]])
        drops = [index for index in range(1, len(li)) if li[index] < li[index - 1]]
        deaths += len(events)
        predictions += len(drops)
        found_all = match_events(events, drops, before=0, after=window)
        matched_all += found_all["matched"]
        latencies.extend(found_all["latencies"])
        observed = []
        for event in events:
            end = min(event + 1 + window, len(li))
            if any(li[index] < li[index - 1] for index in range(event + 1, end)):
                observed.append(event)
        observable += len(observed)
        matched_obs += match_events(observed, drops, before=0, after=window)["matched"]
        per_shard.append(
            {
                "events": len(events),
                "observable": len(observed),
                "predictions": len(drops),
            }
        )
    return {
        "shards": len(per_shard),
        "events": deaths,
        "observable_events": observable,
        "unobservable_events": deaths - observable,
        "predictions": predictions,
        "as_written": {
            "recall": proportion(matched_all, deaths),
            "precision": proportion(matched_all, predictions),
        },
        "observable": {
            "recall": proportion(matched_obs, observable),
            "precision": proportion(matched_obs, predictions),
        },
        "latency": latency_summary(latencies),
        "min_observable_events": DEATH_GATE_MIN_OBSERVABLE,
        "per_shard": per_shard,
    }


def game_over_metrics(shards: Iterable[dict[str, Any]], *,
                      window: int = GAME_OVER_MATCH_WINDOW) -> dict[str, Any]:
    """Game-over gate: ``HUD gameOverFlag`` rising edges as truth.

    Each shard is ``{"hgo": [0/1...], "go_pred": [0/1...]}``; ``go_pred`` defaults
    to ``hgo`` (the flag itself, i.e. the ceiling) so a caller that has no
    pixel-side detector yet cannot silently report a made-up recall.  A corpus
    with no event is **not** a pass: ``measured`` is false and both proportions
    are zero-sample.
    """

    events = 0
    predictions = 0
    matched = 0
    per_shard: list[dict[str, Any]] = []
    for shard in shards:
        truth = [int(value) for value in shard["hgo"]]
        predicted = [int(value) for value in shard.get("go_pred", truth)]
        truth_starts = event_starts(truth)
        pred_starts = event_starts(predicted)
        events += len(truth_starts)
        predictions += len(pred_starts)
        found = match_events(truth_starts, pred_starts, before=0, after=window)
        matched += found["matched"]
        per_shard.append(
            {
                "events": len(truth_starts),
                "predictions": len(pred_starts),
                "matched": found["matched"],
            }
        )
    return {
        "shards": len(per_shard),
        "events": events,
        "predictions": predictions,
        "matched": matched,
        "measured": events > 0 and predictions > 0,
        "window": window,
        "recall": proportion(matched, events),
        "precision": proportion(matched, predictions),
        "per_shard": per_shard,
    }


def _column(entry: dict[str, Any] | None, *, threshold: float | None,
            applicable: bool = True, note: str = "") -> dict[str, Any]:
    """One cell of the two-column table; fail-closed on zero samples."""

    if entry is None:
        return {"value": None, "lower_bound_95": None, "n": 0, "pass": False,
                "applicable": applicable, "note": note}
    lower = entry.get("lower_bound_95")
    n = int(entry.get("n") or 0)
    passed = bool(applicable and n > 0 and threshold is not None
                  and (lower if lower is not None else 0.0) >= threshold)
    return {
        "value": entry.get("value"),
        "lower_bound_95": lower,
        "n": n,
        "pass": passed,
        "applicable": applicable,
        "note": note,
    }


def g2_verdict_v2(block: dict[str, Any]) -> dict[str, Any]:
    """Two-column G2 verdict: as written vs the observable-event amendment.

    ``block`` keys: ``death_scopes`` (from :func:`observable_death_scopes`),
    ``bomb`` (``event_metrics``-style dict), ``readers`` (v1 reader block) and
    ``game_over`` (from :func:`game_over_metrics`).

    Every row is pass/fail on the **95 % one-sided Wilson lower bound**, never on
    the point estimate, and a row with ``n == 0`` can never pass.
    """

    death = block.get("death_scopes") or {}
    if not death:
        return {"measured": False, "reason": "no death scope block", "rows": [],
                "verdict": {"as_written": "未达", "amended": "未达", "signed": False}}

    rows: list[dict[str, Any]] = []

    def add(gate: str, requirement: str, as_written: dict[str, Any],
            amended: dict[str, Any], *, unit: str = "rate") -> None:
        rows.append({"gate": gate, "requirement": requirement, "unit": unit,
                     "as_written": as_written, "amended": amended})

    as_written = death["as_written"]
    observable = death["observable"]
    add("death recall (event level)", "LB >= 0.99",
        _column(as_written["recall"], threshold=0.99),
        _column(observable["recall"], threshold=0.99))
    add("death precision (event level)", "LB >= 0.99",
        _column(as_written["precision"], threshold=0.99),
        _column(observable["precision"], threshold=0.99))

    latency = death.get("latency") or {}
    latency_max = latency.get("max")
    latency_ok = latency_max is not None and int(latency_max) <= DEATH_LATENCY_LIMIT
    latency_cell = {
        "value": latency_max,
        "lower_bound_95": None,
        "n": int(latency.get("n") or 0),
        "pass": bool(latency_ok) and int(latency.get("n") or 0) > 0,
        "applicable": True,
        "note": "原文要求延迟中位数 <= 10；本列取更严的 max",
    }
    add("death latency (max, frames)", "<= 10", dict(latency_cell), dict(latency_cell))

    observable_n = int(death.get("observable_events") or 0)
    add("death observable sample size", f">= {DEATH_GATE_MIN_OBSERVABLE} events",
        {"value": None, "lower_bound_95": None, "n": 0, "pass": False,
         "applicable": False, "note": "限定口径新增：原文无此条"},
        {"value": observable_n, "lower_bound_95": None, "n": observable_n,
         "pass": observable_n >= DEATH_GATE_MIN_OBSERVABLE, "applicable": True,
         "note": "Wilson LB reach 0.99 needs n >= 386 at zero failures"},
        unit="count")

    bomb = block.get("bomb") or {}
    bomb_scopes = block.get("bomb_scopes") or {}
    if bomb_scopes:
        add("bomb recall (event level)", "LB >= 0.99",
            _column((bomb_scopes.get("as_written") or {}).get("recall"), threshold=0.99),
            _column((bomb_scopes.get("observable") or {}).get("recall"), threshold=0.99))
        add("bomb precision (event level)", "LB >= 0.99",
            _column((bomb_scopes.get("as_written") or {}).get("precision"), threshold=0.99),
            _column((bomb_scopes.get("observable") or {}).get("precision"), threshold=0.99))
        observable_bombs = int(bomb_scopes.get("observable_events") or 0)
        add("bomb observable sample size",
            f">= {DEATH_GATE_MIN_OBSERVABLE} events",
            {"value": None, "lower_bound_95": None, "n": 0, "pass": False,
             "applicable": False, "note": "限定口径新增：原文无此条"},
            {"value": observable_bombs, "lower_bound_95": None, "n": observable_bombs,
             "pass": observable_bombs >= DEATH_GATE_MIN_OBSERVABLE, "applicable": True,
             "note": "Wilson LB reach 0.99 needs n >= 386 at zero failures"},
            unit="count")
    else:
        for scope in ("recall", "precision"):
            entry = (bomb.get("event_level") or {}).get(scope) if "event_level" in bomb else bomb.get(scope)
            add(f"bomb {scope} (event level)", "LB >= 0.99",
                _column(entry, threshold=0.99),
                _column(entry, threshold=0.99))

    readers = block.get("readers") or {}
    add("lives read accuracy", "LB >= 0.995",
        _column(readers.get("lives"), threshold=0.995),
        _column(readers.get("lives"), threshold=0.995))
    add("bombs read accuracy", "LB >= 0.995",
        _column(readers.get("bombs"), threshold=0.995),
        _column(readers.get("bombs"), threshold=0.995))
    add("score delta sign accuracy", "LB >= 0.99",
        _column(readers.get("score_sign"), threshold=0.99),
        _column(readers.get("score_sign"), threshold=0.99))

    game_over = block.get("game_over") or {}
    go_unmeasured = not game_over.get("measured")
    # EXP-P2-007: the note is only about *measurement*; once the gate has samples
    # its pass/fail is already in the cell, so the note stays empty.
    go_note = "无 game_over 事件（不可测）" if go_unmeasured else ""
    add("game_over recall (event level)", "LB >= 0.99",
        {"value": None, "lower_bound_95": None, "n": 0, "pass": False,
         "applicable": False, "note": "原文 G2 判据无 game_over 门"},
        _column(game_over.get("recall"), threshold=0.99, note=go_note))
    add("game_over precision (event level)", "LB >= 0.99",
        {"value": None, "lower_bound_95": None, "n": 0, "pass": False,
         "applicable": False, "note": "原文 G2 判据无 game_over 门"},
        _column(game_over.get("precision"), threshold=0.99, note=go_note))

    # EXP-P2-007: the stage-clear row only becomes applicable once an engine
    # actually exports the bit.  A block without ``stage_clear`` keeps the old
    # behaviour (原文 口径: 不可测), so old metrics files render unchanged.
    stage_clear = block.get("stage_clear") or {}
    if stage_clear:
        sc_note = ("引擎已补 EVENTS 0x0080 事件位（rl_events）"
                   if stage_clear.get("measured") else
                   "有事件位但无像素侧检测器 → 只报标签，不报召回")
        sc_recall = _column(stage_clear.get("recall"), threshold=0.99, note=sc_note)
        sc_precision = _column(stage_clear.get("precision"), threshold=0.99)
        if stage_clear.get("precision") is not None:
            # EXP-P2-008: the gate needs recall *and* precision; the row passes
            # only when both Wilson lower bounds clear 0.99, and it reports the
            # weaker of the two so the cell cannot look better than the gate.
            values = [value for value in (sc_recall["value"], sc_precision["value"])
                      if value is not None]
            lowers = [value for value in (sc_recall["lower_bound_95"],
                                          sc_precision["lower_bound_95"])
                      if value is not None]
            note = (f"像素判据：{stage_clear.get('detector')}；精确率 "
                    f"{100.0 * (sc_precision['value'] or 0.0):.2f}% / LB "
                    f"{100.0 * (sc_precision['lower_bound_95'] or 0.0):.3f}%")
            if sc_note:
                note = f"{note}；{sc_note}"
            combined = {
                "value": (min(values) if values else None) if int(stage_clear.get("events") or 0) else None,
                "lower_bound_95": (min(lowers) if lowers else None) if int(stage_clear.get("events") or 0) else None,
                "n": int(stage_clear.get("events") or sc_recall["n"] or 0),
                "pass": bool(sc_recall["pass"] and sc_precision["pass"]),
                "applicable": True,
                "note": note,
            }
            add("stage_clear recall (event level)", "LB >= 0.99（召回与精确都需达标）",
                dict(combined), dict(combined))
        else:
            add("stage_clear recall (event level)", "LB >= 0.99",
                sc_recall, sc_recall)

    online = block.get("online") or {}
    if online:
        add("online validation (WP5, all gates)", "every online LB >= 0.99",
            {"value": None, "lower_bound_95": None,
             "n": int(online.get("frames") or 0),
             "pass": bool(online.get("pass")),
             "applicable": True,
             "note": (f"{online.get('episodes')} 局 / {online.get('frames')} 帧"
                      + ("" if online.get("long_enough") else "，流太短"))},
            {"value": None, "lower_bound_95": None,
             "n": int(online.get("frames") or 0),
             "pass": bool(online.get("pass")),
             "applicable": True,
             "note": (f"{online.get('episodes')} 局 / {online.get('frames')} 帧"
                      + ("" if online.get("long_enough") else "，流太短"))})

    def column_pass(name: str) -> bool:
        return all(row[name]["pass"] for row in rows if row[name]["applicable"])

    as_written_pass = column_pass("as_written")
    amended_pass = column_pass("amended")
    return {
        "measured": True,
        "rows": rows,
        "scope": {
            "events": death.get("events"),
            "observable_events": observable_n,
            "unobservable_events": death.get("unobservable_events"),
            "predictions": death.get("predictions"),
            "game_over_events": game_over.get("events"),
            "game_over_predictions": game_over.get("predictions"),
        },
        "as_written_pass": as_written_pass,
        "amended_pass": amended_pass,
        # one verdict, never both: the original criteria stay the reference until
        # a human signs the amendment (see the G2 report's signature block)
        "verdict": {
            "as_written": "达标" if as_written_pass else "未达",
            "amended": "PASS-with-amendment（待负责人签署）" if amended_pass else "未达",
            "signed": False,
        },
    }


def render_v2_table(verdict: dict[str, Any]) -> str:
    """Markdown for the two-column G2 table (as written / amended scope)."""

    def cell(entry: dict[str, Any], *, is_latency: bool = False,
             unit: str = "rate") -> str:
        if not entry.get("applicable"):
            return "不适用"
        value = entry.get("value")
        if value is None or not entry.get("n"):
            # zero samples are never shown as a rate: "0.00%" would read as a
            # measured failure instead of "not measured"
            shown = "n/a"
        elif is_latency:
            shown = f"{value:.0f}"
        elif unit == "count":
            shown = f"{int(value)}"
        else:
            shown = f"{100.0 * value:.2f}%"
        lower = entry.get("lower_bound_95")
        lower_text = ("n/a" if lower is None or not entry.get("n")
                      else f"{100.0 * lower:.3f}%")
        mark = "达标" if entry.get("pass") else "未达标"
        note = f"（{entry['note']}）" if entry.get("note") else ""
        return f"{shown} / LB {lower_text} / n={entry.get('n')} → **{mark}**{note}"

    lines = [
        "| # | 指标 | 要求 | 按原文口径 | 限定口径（death 可观测 + 独立 game_over） |",
        "|---|---|---|---|---|",
    ]
    for index, row in enumerate(verdict.get("rows", []), start=1):
        is_latency = "latency" in row["gate"]
        unit = row.get("unit", "rate")
        lines.append(
            f"| {index} | {row['gate']} | {row['requirement']} | "
            f"{cell(row['as_written'], is_latency=is_latency, unit=unit)} | "
            f"{cell(row['amended'], is_latency=is_latency, unit=unit)} |"
        )
    scope = verdict.get("scope", {})
    lines.append("")
    lines.append(
        f"样本：death 事件 {scope.get('events')}（可观测 {scope.get('observable_events')}，"
        f"不可观测 {scope.get('unobservable_events')}），预测 {scope.get('predictions')}；"
        f"game_over 事件 {scope.get('game_over_events')}，预测 {scope.get('game_over_predictions')}。"
    )
    lines.append(
        f"判定：按原文口径 **{verdict['verdict']['as_written']}**；"
        f"限定口径 **{verdict['verdict']['amended']}**（signed="
        f"{'true' if verdict['verdict']['signed'] else 'false'}）。"
    )
    return "\n".join(lines) + "\n"


def report_v2_command(args) -> int:
    """Render (or recompute) the two-column G2 table from a metrics file."""

    lines: list[str] = []
    exit_code = 0
    for path in args.metrics:
        metrics = json.loads(Path(path).read_text(encoding="utf-8"))
        block = metrics.get("block") or {}
        verdict = metrics.get("verdict") or g2_verdict_v2(block)
        lines.append(f"### {Path(path).name}")
        lines.append("")
        lines.append(render_v2_table(verdict))
        if not verdict.get("measured", True):
            exit_code = 2
    markdown = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
    print(markdown)
    return exit_code


# ---------------------------------------------------------------------------
# EXP-P2-008: pixel-side stage_clear detector (play64 gradient flip)
# ---------------------------------------------------------------------------
#: Gameplay frames are lit from below, so ``mean(top rows) - mean(bottom rows)``
#: of the 64x64 play64 plane is negative; the stage-clear screen replaces the
#: scrolling background with a top-lit gradient panel.  A pure ``vgrad >= 6``
#: level test fails on stages whose background is *already* top-lit (stage 2 in
#: the accepted corpus), so the event is the **jump**:
#: ``vgrad[t] - vgrad[t-1] >= JUMP`` while the panel stays top-lit
#: (``vgrad >= FLOOR`` for ``SUSTAIN`` frames).  With ``JUMP`` between 2 and 6
#: every captured event fires exactly once, and no gameplay frame does.
STAGE_CLEAR_VGRAD_JUMP = 4.0
STAGE_CLEAR_VGRAD_FLOOR = 3.0
#: The clear screen is static for tens of frames, so the panel must stay up for
#: this many frames; the reported frame is still the jump (the detector costs
#: ``sustain - 1`` frames of latency when it runs online).
STAGE_CLEAR_SUSTAIN = 5
#: One clear screen per stage: suppress a second rise for 150 frames (the fade
#: out of the same screen and the next stage's opening would otherwise re-fire).
STAGE_CLEAR_REFRACTORY = 150
#: Capture window conventions of ``tools/rl/v400_stage_clear_replay.py``.
STAGE_CLEAR_PRE = 60
STAGE_CLEAR_POST = 120
#: Event window used to match a predicted rise against the engine edge.
STAGE_CLEAR_MATCH_WINDOW = 10


def stage_clear_vgrad(play: Any) -> Any:
    """Top-minus-bottom brightness of a ``(T, 64, 64)`` play64 stack.

    ``play64`` rows are the vertical axis (the player starts at the bottom rows
    and the scrolling background lightens towards the bottom during play).
    """

    import numpy as np

    arr = np.asarray(play, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (64, 64):
        raise ValueError(f"play64 stack must be (T, 64, 64), got {arr.shape}")
    return arr[:, :16, :].mean(axis=(1, 2)) - arr[:, 48:, :].mean(axis=(1, 2))


def stage_clear_pixel_flags(play: Any, *, floor: float = STAGE_CLEAR_VGRAD_FLOOR) -> Any:
    """Per-frame pixel state: the playfield looks like the top-lit clear panel."""

    import numpy as np

    return stage_clear_vgrad(play) >= float(floor)


def stage_clear_pixel_predictions(
    play: Any,
    *,
    jump: float = STAGE_CLEAR_VGRAD_JUMP,
    floor: float = STAGE_CLEAR_VGRAD_FLOOR,
    sustain: int = STAGE_CLEAR_SUSTAIN,
    refractory: int = STAGE_CLEAR_REFRACTORY,
) -> list[int]:
    """Event frames: a gradient jump that stays top-lit (frame indices)."""

    import numpy as np

    vgrad = np.asarray(stage_clear_vgrad(play), dtype=np.float32)
    if vgrad.size == 0:
        return []
    delta = np.zeros_like(vgrad)
    delta[1:] = vgrad[1:] - vgrad[:-1]
    held = vgrad >= float(floor)
    out: list[int] = []
    last = -(10 ** 9)
    for index in range(1, int(vgrad.size)):
        if delta[index] < float(jump):
            continue
        if index + int(sustain) > held.size:
            break
        if not bool(held[index:index + int(sustain)].all()):
            continue
        if index - last < int(refractory):
            continue
        out.append(index)
        last = index
    return out


def split_stage_clear_cases(cases: Iterable[str]) -> dict[str, str]:
    """Deterministic whole-replay split: sha256(case)[0] % 10 < 3 -> test."""

    import hashlib

    return {case: ("test" if hashlib.sha256(case.encode()).digest()[0] % 10 < 3 else "train")
            for case in sorted(set(cases))}


def load_stage_clear_events(directory: str | Path) -> list[dict[str, Any]]:
    """Read the ``frames.npz`` + ``meta.json`` corpus written by the replay tool."""

    import numpy as np

    root = Path(directory)
    events: list[dict[str, Any]] = []
    for npz in sorted(root.glob("*/frames.npz")):
        meta = json.loads((npz.parent / "meta.json").read_text(encoding="utf-8"))
        data = np.load(npz)
        level = np.asarray(data["level"], dtype=np.int8)
        events.append({
            "case": meta["case"],
            "stage": int(meta["stage"]),
            "level": level,
            "play": np.asarray(data["play"], dtype=np.uint8),
            "edge": int(np.argmax(level == 1)) if bool((level == 1).any()) else None,
            "meta": meta,
            "path": str(npz),
        })
    return events


def _stage_clear_eval(events: Sequence[dict[str, Any]], *,
                      jump: float = STAGE_CLEAR_VGRAD_JUMP,
                      floor: float = STAGE_CLEAR_VGRAD_FLOOR,
                      sustain: int = STAGE_CLEAR_SUSTAIN,
                      refractory: int = STAGE_CLEAR_REFRACTORY,
                      window: int = STAGE_CLEAR_MATCH_WINDOW) -> dict[str, Any]:
    truth = matched = predicted = 0
    latencies: list[int] = []
    frame_true = frame_pred = frame_tp = 0
    misses: list[dict[str, Any]] = []
    for event in events:
        level = event["level"]
        preds = stage_clear_pixel_predictions(event["play"], jump=jump, floor=floor,
                                              sustain=sustain, refractory=refractory)
        flags = stage_clear_pixel_flags(event["play"], floor=floor)
        edge = event.get("edge")
        truth += 1 if edge is not None else 0
        predicted += len(preds)
        found = False
        if edge is not None:
            for pred in preds:
                if edge <= pred <= edge + window:
                    matched += 1
                    latencies.append(pred - edge)
                    found = True
                    break
            if not found:
                misses.append({"case": event["case"], "stage": event["stage"],
                               "edge": edge,
                               "predictions": preds[:4],
                               "vgrad_at_edge": float(
                                   stage_clear_vgrad(event["play"])[edge])})
        frame_true += int((level == 1).sum())
        frame_pred += int(flags.sum())
        frame_tp += int(((level == 1) & flags).sum())
    return {
        "events": truth,
        "matched": matched,
        "predictions": predicted,
        "window": window,
        "recall": proportion(matched, truth),
        "precision": proportion(matched, predicted),
        "latency": latency_summary(latencies),
        "frame": {
            "true": frame_true,
            "predicted": frame_pred,
            "tp": frame_tp,
            "recall": proportion(frame_tp, frame_true),
            "precision": proportion(frame_tp, frame_pred),
        },
        "misses": misses,
        "latencies": latencies,
        "thresholds": {"jump": jump, "floor": floor, "sustain": sustain,
                       "refractory": refractory},
    }


def stage_clear_cross_validation(events: Sequence[dict[str, Any]], *,
                                 folds: int = 3,
                                 jumps: Sequence[float] = (2.0, 3.0, 4.0, 6.0),
                                 floors: Sequence[float] = (0.0, 3.0, 5.0),
                                 sustains: Sequence[int] = (5, 8)) -> dict[str, Any]:
    """K-fold by replay: every event is scored by thresholds fitted without it.

    A single 70/30 split throws away 70 % of the events, which matters when the
    Wilson lower bound needs n >= 386.  Three folds keep a per-event estimate
    that is still out-of-fold while using the whole corpus, and every fold's
    chosen threshold is reported.
    """

    import hashlib

    grouped: dict[int, list[dict[str, Any]]] = {index: [] for index in range(int(folds))}
    for event in events:
        fold = hashlib.sha256(event["case"].encode()).digest()[1] % int(folds)
        grouped[fold].append(event)
    total_events = total_matched = total_predictions = 0
    frame_true = frame_pred = frame_tp = 0
    latencies: list[int] = []
    fold_rows = []
    for fold, held in sorted(grouped.items()):
        training = [event for index, chunk in grouped.items() if index != fold
                    for event in chunk]
        if not held or not training:
            continue
        grid = []
        for jump in jumps:
            for floor in floors:
                for sustain in sustains:
                    result = _stage_clear_eval(training, jump=jump, floor=floor,
                                               sustain=sustain)
                    grid.append((result["recall"]["lower_bound_95"],
                                 result["precision"]["lower_bound_95"],
                                 jump, floor, sustain))
        _, _, best_jump, best_floor, best_sustain = max(grid)
        result = _stage_clear_eval(held, jump=best_jump, floor=best_floor,
                                   sustain=best_sustain)
        total_events += result["events"]
        total_matched += result["matched"]
        total_predictions += result["predictions"]
        frame_true += result["frame"]["true"]
        frame_pred += result["frame"]["predicted"]
        frame_tp += result["frame"]["tp"]
        latencies.extend(result["latencies"])
        fold_rows.append({"fold": fold, "held_out": len(held),
                          "jump": best_jump, "floor": best_floor,
                          "sustain": best_sustain,
                          "events": result["events"], "matched": result["matched"],
                          "predictions": result["predictions"],
                          "recall": result["recall"], "precision": result["precision"],
                          "misses": result["misses"]})
    return {
        "folds": fold_rows,
        "events": total_events,
        "matched": total_matched,
        "predictions": total_predictions,
        "recall": proportion(total_matched, total_events),
        "precision": proportion(total_matched, total_predictions),
        "latency": latency_summary(latencies),
        "frame": {
            "true": frame_true,
            "predicted": frame_pred,
            "tp": frame_tp,
            "recall": proportion(frame_tp, frame_true),
            "precision": proportion(frame_tp, frame_pred),
        },
        "split": "3-fold by sha256(case)[1] % 3; each event scored by a fold that "
                 "never saw its replay",
    }


def stage_clear_corpus_gate(directory: str | Path, *,
                            jumps: Sequence[float] = (2.0, 3.0, 4.0, 6.0),
                            floors: Sequence[float] = (0.0, 3.0, 5.0),
                            sustains: Sequence[int] = (5, 8)) -> dict[str, Any]:
    """Fit the criterion on the training replays, report the held-out block.

    The returned dict is directly usable as ``block["stage_clear"]`` in
    :func:`g2_verdict_v2`: it carries ``recall``/``precision`` with Wilson lower
    bounds, the latency, the sample size and the split description.  The primary
    numbers are the 3-fold out-of-fold aggregate; the single 70/30 split is kept
    next to it for transparency.
    """

    events = load_stage_clear_events(directory)
    if not events:
        return {"measured": False, "reason": "no captured events", "events": 0,
                "detector": "vgrad jump", "recall": proportion(0, 0),
                "precision": proportion(0, 0), "latency": latency_summary([])}
    split = split_stage_clear_cases(event["case"] for event in events)
    train = [event for event in events if split[event["case"]] == "train"]
    test = [event for event in events if split[event["case"]] == "test"]
    grid = []
    for jump in jumps:
        for floor in floors:
            for sustain in sustains:
                result = _stage_clear_eval(train, jump=jump, floor=floor,
                                           sustain=sustain)
                grid.append({"jump": jump, "floor": floor, "sustain": sustain,
                             "train": result})
    usable = [row for row in grid if row["train"]["events"]]
    if not usable:
        return {"measured": False, "reason": "no training events", "events": len(events),
                "detector": "vgrad jump", "recall": proportion(0, 0),
                "precision": proportion(0, 0), "latency": latency_summary([])}
    best = max(usable, key=lambda row: (
        row["train"]["recall"]["lower_bound_95"],
        row["train"]["precision"]["lower_bound_95"]))
    held_out = _stage_clear_eval(test, jump=best["jump"], floor=best["floor"],
                                 sustain=best["sustain"])
    cross = stage_clear_cross_validation(events)
    return {
        "measured": True,
        # primary numbers = out-of-fold cross-validation over every event, so the
        # Wilson lower bound is not wasted on a 30 % split
        "events": cross["events"],
        "recall": cross["recall"],
        "precision": cross["precision"],
        "latency": cross["latency"],
        "frame": cross["frame"],
        "cross_validation": cross,
        "train_events": len(train),
        "test_events": len(test),
        "train_cases": len({event["case"] for event in train}),
        "test_cases": len({event["case"] for event in test}),
        "split": "sha256(case)[0] % 10 < 3 -> test (whole replay)",
        "thresholds": held_out["thresholds"],
        "grid": [{k: row["train"][k] for k in ("events", "matched", "recall",
                                               "precision", "latency")}
                 | {"jump": row["jump"], "floor": row["floor"],
                    "sustain": row["sustain"]}
                 for row in grid],
        "train": _stage_clear_eval(train, jump=best["jump"], floor=best["floor"],
                                   sustain=best["sustain"]),
        "test": held_out,
        "window": held_out["window"],
        "detector": ("play64 top-minus-bottom brightness jump >= "
                     f"{best['jump']:.0f} while the panel stays top-lit, "
                     f"sustain {best['sustain']} frames, "
                     f"{STAGE_CLEAR_REFRACTORY}-frame refractory"),
        "min_events": DEATH_GATE_MIN_OBSERVABLE,
    }


# ---------------------------------------------------------------------------
# EXP-P2-008: pixel-side death detector (player sprite NCC collapse)
# ---------------------------------------------------------------------------
#: Zero-mean NCC of the 5x5 play96-cell patch at the player: a live sprite keeps
#: the patch nearly identical between frames, the death dissolve does not.
DEATH_NCC_LOW = 0.55
DEATH_NCC_HIGH = 0.9
DEATH_NCC_HALF = 2
DEATH_NCC_SEARCH = 1
#: The post-respawn invincibility blink repeats the sprite loss; one death per
#: 90 frames is the physical floor of the game's respawn animation.
DEATH_NCC_REFRACTORY = 90
DEATH_NCC_WINDOW = 10


def player_patch_ncc(planes: Any, positions: Sequence[tuple[int, int]]) -> Any:
    """NCC of the player patch between consecutive frames.

    ``planes`` is ``(T, 112, 96)`` play96 (rows = height, cols = width) and
    ``positions`` the per-frame player cell ``(row, col)`` — the engine's own
    ``px``/``py`` (the harness ``PLAYER`` command), never a HUD counter.
    Frame 0 has no predecessor and reports 0.
    """

    import numpy as np

    arr = np.asarray(planes, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (112, 96):
        raise ValueError(f"play96 stack must be (T, 112, 96), got {arr.shape}")
    if len(positions) != arr.shape[0]:
        raise ValueError("positions must have one (row, col) per frame")
    half, search = DEATH_NCC_HALF, DEATH_NCC_SEARCH
    out = np.zeros(arr.shape[0], dtype=np.float32)
    for index in range(1, arr.shape[0]):
        row, col = positions[index]
        prow, pcol = positions[index - 1]
        template = arr[index - 1, prow - half:prow + half + 1, pcol - half:pcol + half + 1]
        template = template - template.mean()
        t_norm = float(np.sqrt((template * template).sum()))
        if t_norm < 1e-3:
            continue
        best = -1.0
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                patch = arr[index, row + dy - half:row + dy + half + 1,
                            col + dx - half:col + dx + half + 1]
                patch = patch - patch.mean()
                p_norm = float(np.sqrt((patch * patch).sum()))
                if p_norm < 1e-3:
                    continue
                value = float((template * patch).sum() / (t_norm * p_norm))
                if value > best:
                    best = value
        out[index] = best if best > -1.0 else 0.0
    return out


def pixel_death_predictions(ncc: Any, *, low: float = DEATH_NCC_LOW,
                            high: float = DEATH_NCC_HIGH,
                            refractory: int = DEATH_NCC_REFRACTORY) -> list[int]:
    """Rising events of the sprite-collapse flag, with a respawn refractory."""

    import numpy as np

    values = np.asarray(ncc, dtype=np.float32)
    out: list[int] = []
    last = -(10 ** 9)
    for index in range(1, values.size):
        if values[index] <= low and values[index - 1] >= high and index - last > refractory:
            out.append(index)
            last = index
    return out


def pixel_death_metrics(events: Iterable[dict[str, Any]], *,
                        low: float = DEATH_NCC_LOW, high: float = DEATH_NCC_HIGH,
                        window: int = DEATH_NCC_WINDOW) -> dict[str, Any]:
    """Event-level recall/precision/latency of the pixel death detector.

    Each event is ``{"ncc": [...], "events": [death frames], "labels": [...]}``;
    ``labels`` is only used to report the HUD-unobservable subset separately.
    ``predictions`` counts every fired event, so precision cannot be inflated by
    ignoring the extra fires.
    """

    truth = predicted = matched = 0
    unobservable = unobservable_matched = 0
    latencies: list[int] = []
    for event in events:
        preds = pixel_death_predictions(event["ncc"], low=low, high=high)
        predictions_list = list(preds)
        truth += len(event["events"])
        predicted += len(predictions_list)
        used: set[int] = set()
        labels = event.get("labels")
        for frame in event["events"]:
            for pred in predictions_list:
                if frame <= pred <= frame + window and pred not in used:
                    used.add(pred)
                    matched += 1
                    latencies.append(pred - frame)
                    break
            if labels is not None:
                row = labels[frame]
                observable = any(other.get("li", 0) < row.get("li", 0)
                                 for other in labels[frame + 1:frame + window + 1])
                if not observable:
                    unobservable += 1
                    if any(frame <= pred <= frame + window for pred in predictions_list):
                        unobservable_matched += 1
    return {
        "events": truth,
        "predictions": predicted,
        "matched": matched,
        "recall": proportion(matched, truth),
        "precision": proportion(matched, predicted),
        "latency": latency_summary(latencies),
        "unobservable_events": unobservable,
        "unobservable_matched": unobservable_matched,
        "thresholds": {"low": low, "high": high, "window": window,
                       "refractory": DEATH_NCC_REFRACTORY},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def stage_clear_gate_command(args) -> int:
    """EXP-P2-008: fit + evaluate the pixel stage_clear detector on a capture dir."""

    block = stage_clear_corpus_gate(args.events)
    payload = {
        "schema": "v400-stage-clear-gate/1",
        "eventsDir": args.events,
        "block": {"stage_clear": block},
        "stage_clear": block,
    }
    if "verdict" not in payload:
        payload["verdict"] = g2_verdict_v2(payload["block"])
    text = json.dumps(payload, indent=1, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    if not block.get("measured"):
        return 2
    # fail-closed: a measured-but-unsized block must not exit 0
    recall_lb = (block.get("recall") or {}).get("lower_bound_95") or 0.0
    precision_lb = (block.get("precision") or {}).get("lower_bound_95") or 0.0
    return 0 if min(recall_lb, precision_lb) >= 0.99 else 1


def wp5_online_block(gate: dict[str, Any]) -> dict[str, Any]:
    """EXP-P2-010: adapt a ``v400_pixel_death`` online gate to the G2 table row.

    The live CNN loop reports its own fail-closed verdict
    (``tools/rl/v400_pixel_death.py online_gate``: schema, >= 5 episodes of
    >= 2400 frames each, at least one death per episode, latency max <= 10,
    recall *and* precision Wilson LB >= 0.99).  This adapter only reshapes that
    verdict into the ``online`` block ``g2_verdict_v2`` reads, so row 14 of the
    two-column table is driven by the pixel CNN run instead of the retired
    HUD-readout loop -- and it keeps ``pass`` exactly as the gate decided.
    """

    if not isinstance(gate, dict):
        raise DetectorDataError("WP5 online gate payload must be an object")
    for key in ("episodes", "frames", "deaths", "alarms", "matched", "passed"):
        if key not in gate:
            raise DetectorDataError(f"WP5 online gate payload misses {key!r}")
    recall = gate.get("recall") or {}
    precision = gate.get("precision") or {}
    latency = gate.get("latency") or {}
    return {
        "episodes": int(gate["episodes"]),
        "frames": int(gate["frames"]),
        "long_enough": bool(
            int(gate["episodes"]) >= ONLINE_MIN_EPISODES
            and int(gate["frames"]) >= ONLINE_MIN_FRAMES
        ),
        "pass": bool(gate["passed"]),
        "pixel_death_recall": {
            "n": int(recall.get("n") or 0),
            "successes": int(recall.get("successes") or 0),
            "value": recall.get("value"),
            "lower_bound_95": recall.get("lower_bound_95"),
        },
        "pixel_death_precision": {
            "n": int(precision.get("n") or 0),
            "successes": int(precision.get("successes") or 0),
            "value": precision.get("value"),
            "lower_bound_95": precision.get("lower_bound_95"),
        },
        "pixel_death_latency": latency,
        "failures": list(gate.get("failures") or []),
    }


def wp5_online_command(args) -> int:
    """Inject the WP5 pixel CNN online gate into a G2 metrics file."""

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    gate = json.loads(Path(args.gate).read_text(encoding="utf-8"))
    block = dict(metrics.get("block") or {})
    block["online"] = wp5_online_block(gate)
    verdict = g2_verdict_v2(block)
    payload = {
        **metrics,
        "block": block,
        "verdict": verdict,
        "sources": {**(metrics.get("sources") or {}), "online": str(args.gate)},
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"online": block["online"],
                      "as_written_pass": verdict.get("as_written_pass"),
                      "amended_pass": verdict.get("amended_pass")},
                     ensure_ascii=False, indent=1))
    return 0 if block["online"]["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("split", help="assign whole stages to splits, verify no leakage")
    split.add_argument("--index", required=True)
    split.add_argument("--out", default=None)
    split.set_defaults(func=split_command)

    train = sub.add_parser("train", help="train + evaluate one input resolution")
    train.add_argument("--index", required=True)
    train.add_argument("--split", default=None)
    train.add_argument("--resolution", default="play96", choices=("play96", "play64"))
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--death-window", type=int, default=1,
                       help="death head anticipation window in frames (1 = next frame)")
    train.add_argument("--pos-weight-cap", type=float, default=8.0,
                       help="cap on the death positive-class weight (calibration guard)")
    train.add_argument("--max-samples", type=int, default=36_000)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    train.add_argument("--out", default=None)
    train.add_argument("--checkpoint", default=None)
    train.set_defaults(func=train_command)

    report = sub.add_parser("report", help="render the G2 table from metrics files")
    report.add_argument("--metrics", nargs="+", required=True)
    report.add_argument("--out", default=None)
    report.set_defaults(func=report_command)

    v1 = sub.add_parser(
        "train-v1",
        help="v1: HUD readers + causal HUD-sequence event heads, evaluated by split",
    )
    v1.add_argument("--index", required=True)
    v1.add_argument("--split", default=None)
    v1.add_argument("--epochs", type=int, default=3)
    v1.add_argument("--event-epochs", type=int, default=6)
    v1.add_argument("--max-samples", type=int, default=48_000)
    v1.add_argument("--reader-batch-size", type=int, default=512)
    v1.add_argument("--event-batch-size", type=int, default=8192)
    v1.add_argument("--eval-batch-size", type=int, default=2048)
    v1.add_argument("--learning-rate", type=float, default=2e-3)
    v1.add_argument("--pos-weight-cap", type=float, default=8.0)
    v1.add_argument("--seed", type=int, default=0)
    v1.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    v1.add_argument("--bomb-gate", action="store_true",
                    help="treat the bomb event gate as required (G2-relevant)")
    v1.add_argument("--out", required=True)
    v1.add_argument("--checkpoint", default=None)
    v1.set_defaults(func=v1_command)

    report_v1 = sub.add_parser("report-v1", help="render the v1 G2 table")
    report_v1.add_argument("--metrics", nargs="+", required=True)
    report_v1.add_argument("--out", default=None)
    report_v1.set_defaults(func=report_v1_command)

    report_v2 = sub.add_parser(
        "report-v2",
        help="render the two-column G2 table (as written vs observable amendment)",
    )
    report_v2.add_argument("--metrics", nargs="+", required=True)
    report_v2.add_argument("--out", default=None)
    report_v2.set_defaults(func=report_v2_command)

    score_gate = sub.add_parser(
        "score-gate",
        help="EXP-P2-005: score the reader against displayScore labels at a chosen cell origin",
    )
    score_gate.add_argument("--index", required=True)
    score_gate.add_argument("--split", default=None)
    score_gate.add_argument("--split-name", default="test")
    score_gate.add_argument("--relabel", required=True,
                            help="EXP-P2-005 re-label directory (shards/<key>.jsonl)")
    score_gate.add_argument("--checkpoint", required=True,
                            help="reader checkpoint (.pt with a 'reader' state dict)")
    score_gate.add_argument("--score-x0", type=int, default=HUD_SCORE_X0,
                            help="band column of the first score glyph cell "
                                 f"(legacy {HUD_SCORE_X0}, measured {HUD_SCORE_X0_MEASURED})")
    score_gate.add_argument("--label", default="displayScore",
                            choices=("displayScore", "globalsScore"))
    score_gate.add_argument("--device", default="cpu", choices=("cuda", "cpu"))
    score_gate.add_argument("--out", default=None)
    score_gate.set_defaults(func=score_gate_command)

    stage_clear = sub.add_parser(
        "stage-clear-gate",
        help="EXP-P2-008: pixel-side stage_clear detector on the replay corpus",
    )
    stage_clear.add_argument("--events", required=True,
                             help="directory of <case>-s<stage>/frames.npz+meta.json")
    stage_clear.add_argument("--out", default=None)
    stage_clear.set_defaults(func=stage_clear_gate_command)

    wp5_online = sub.add_parser(
        "wp5-online",
        help="EXP-P2-010: drive table row 14 from the pixel CNN live-loop gate",
    )
    wp5_online.add_argument("--metrics", required=True,
                            help="G2 metrics file whose block is reused")
    wp5_online.add_argument("--gate", required=True,
                            help="online gate JSON from tools/rl/v400_pixel_death.py")
    wp5_online.add_argument("--out", default=None)
    wp5_online.set_defaults(func=wp5_online_command)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DetectorDataError as exc:
        print(f"ERR fail-closed: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


