"""Stage 4 offline action-distribution diagnostics.

This module deliberately operates on ordered label rows only.  It never decodes
pixels, changes training data, or treats shuffled batches as temporal sequences.
Sequences are reset at causal-segment boundaries before continuity statistics are
computed.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

from .stage0_diagnostics import candidate_index, summarize_actions
from .pilot import AvoidPilot, PilotConfig
from . import vision_history
from .vision_data import VisionDataset

SCHEMA = "v105-stage4-action-distribution/1"


def _candidate_rows(actions: Iterable) -> list[int]:
    return [candidate_index(action) for action in actions]


def _stats(segments: list[list[int]]) -> dict:
    """Aggregate Stage 0 statistics over segments without cross-boundary pairs."""
    flat = [item for segment in segments for item in segment]
    actions = [[index % 9, 0, index // 9, 0] for index in flat]
    result = summarize_actions(actions)
    lengths: list[int] = []
    for segment in segments:
        if not segment:
            continue
        run = 1
        for previous, current in zip(segment, segment[1:]):
            if current == previous:
                run += 1
            else:
                lengths.append(run)
                run = 1
        lengths.append(run)
    result["action_run_length"] = {
        "p50": float(np.percentile(lengths, 50)) if lengths else None,
        "p90": float(np.percentile(lengths, 90)) if lengths else None,
        "max": max(lengths) if lengths else None,
    }
    transitions = sum(max(0, len(segment) - 1) for segment in segments)
    if transitions:
        moves = [index % 9 for index in flat]
        focuses = [index // 9 for index in flat]
        candidate_changes = sum(sum(a != b for a, b in zip(segment, segment[1:])) for segment in segments)
        move_changes = sum(sum((a % 9) != (b % 9) for a, b in zip(segment, segment[1:])) for segment in segments)
        focus_changes = sum(sum((a // 9) != (b // 9) for a, b in zip(segment, segment[1:])) for segment in segments)
        result["candidate_flip_rate"] = candidate_changes / transitions
        result["move_flip_rate"] = move_changes / transitions
        result["focus_flip_rate"] = focus_changes / transitions
    else:
        result["candidate_flip_rate"] = 0.0
        result["move_flip_rate"] = 0.0
        result["focus_flip_rate"] = 0.0
    return result


def _segments(indices: np.ndarray, starts: np.ndarray) -> list[list[int]]:
    """Split ordered candidate indices at explicit segment starts."""
    if len(indices) != len(starts):
        raise ValueError(f"indices/segment_starts length mismatch: {len(indices)} != {len(starts)}")
    out: list[list[int]] = []
    current: list[int] = []
    previous = None
    for value, start in zip(indices.tolist(), starts.tolist()):
        if previous is not None and int(start) != int(previous):
            if current:
                out.append(current)
            current = []
        current.append(int(value))
        previous = int(start)
    if current:
        out.append(current)
    return out


def analyze_action_sequences(
    human_candidates: Iterable[int],
    teacher_candidates: Iterable[int],
    segment_starts: Iterable[int],
) -> dict:
    """Compare human and teacher actions in causal order, resetting per segment.

    ``segment_starts`` contains the causal segment id for each row, not merely a
    boolean boundary.  Both streams must have identical ordered rows.
    """
    human = np.asarray(list(human_candidates), dtype=np.int64).reshape(-1)
    teacher = np.asarray(list(teacher_candidates), dtype=np.int64).reshape(-1)
    starts = np.asarray(list(segment_starts), dtype=np.int64).reshape(-1)
    if human.size != teacher.size or human.size != starts.size:
        raise ValueError("human, teacher, and segment_starts must have equal lengths")
    if np.any((human < 0) | (human >= 18) | (teacher < 0) | (teacher >= 18)):
        raise ValueError("candidate indices must be in [0, 18)")
    human_segments = _segments(human, starts)
    teacher_segments = _segments(teacher, starts)
    human_flat = [item for segment in human_segments for item in segment]
    teacher_flat = [item for segment in teacher_segments for item in segment]
    return {
        "schema": SCHEMA,
        "frames": int(human.size),
        "segments": len(human_segments),
        "human": _stats(human_segments),
        "teacher": _stats(teacher_segments),
        "human_teacher_agreement": float(np.mean(human == teacher)) if human.size else 0.0,
        "human_teacher_move_agreement": float(np.mean((human % 9) == (teacher % 9))) if human.size else 0.0,
        "human_teacher_focus_agreement": float(np.mean((human // 9) == (teacher // 9))) if human.size else 0.0,
        "segment_lengths": {
            "p50": float(np.percentile([len(s) for s in human_segments], 50)) if human_segments else None,
            "p90": float(np.percentile([len(s) for s in human_segments], 90)) if human_segments else None,
            "max": max((len(s) for s in human_segments), default=None),
        },
    }


def analyze_dataset(dataset_dir: Path, *, history_contract: str = "causal-segment-v1", teacher_values: Path | None = None, switch_penalty: float = 0.0) -> dict:
    """Run ordered human-vs-frozen-teacher diagnostics without decoding pixels.

    ``teacher_values`` is intentionally rejected: the A shards do not carry values
    and this audit must not mistake a cache for a recomputed teacher stream.
    """
    if teacher_values is not None:
        raise ValueError("teacher cache is not accepted; teacher is recomputed from A privileged labels")
    dataset_dir = Path(dataset_dir)
    dataset = VisionDataset(dataset_dir, history_contract=history_contract, decode_threads=1, use_decode_cache=False)
    pilot = AvoidPilot(PilotConfig(switch_penalty=float(switch_penalty)))
    try:
        human: list[int] = []
        teacher: list[int] = []
        starts: list[int] = []
        segment_offset = 0
        for shard_index, (path, _global_start, size) in enumerate(dataset.offsets):
            local_meta = dataset.meta_of_shard(path)
            local_starts, _reliable = vision_history.segment_starts(local_meta)
            labels = dataset.labels(list(range(segment_offset, segment_offset + size)))
            actions = labels["action"]
            human.extend((actions[:, 2].astype(np.int64) * 9 + actions[:, 0]).tolist())
            for row in range(size):
                if row == 0 or int(local_starts[row]) != int(local_starts[row - 1]):
                    pilot._prev_move = 0
                observation = {
                    "self_feat": labels["self_feat"][row].astype(np.float32),
                    "bullets": labels["bullets"][row].astype(np.float32),
                    "danger": labels["danger"][row],
                }
                scores = np.asarray(pilot.candidate_scores(observation), dtype=np.float32)
                choice = int(np.argmax(scores))
                teacher.append(choice)
                pilot._prev_move = choice % 9
            starts.extend(((shard_index << 32) + local_starts).tolist())
            segment_offset += size
        result = analyze_action_sequences(human, teacher, starts)
        result["dataset"] = str(dataset_dir)
        result["history_contract"] = history_contract
        result["teacher_source"] = "recomputed-from-A-privileged-labels"
        result["teacher_candidate_order"] = "focus*9+move; np.argmax lowest index on ties"
        result["teacher_config"] = {"switch_penalty": float(switch_penalty)}
        result["teacher_pilot_sha256"] = hashlib.sha256(Path(__file__).with_name("pilot.py").read_bytes()).hexdigest()
        result["teacher_values"] = "not-used"
        return result
    finally:
        dataset.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ordered Stage 4 action-distribution diagnostic")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history-contract", default="causal-segment-v1")
    parser.add_argument("--switch-penalty", type=float, default=0.0)
    args = parser.parse_args(argv)
    result = analyze_dataset(Path(args.dataset), history_contract=args.history_contract,
                             switch_penalty=args.switch_penalty)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
