"""Read-only Phase 1 distribution comparison; rejects failed or unapproved rounds.

Uses privilege labels for diagnosis only. It never builds training splits or weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .stage0_diagnostics import summarize_actions
from .stage4_diagnostics import _stats
from .vision_data import VisionDataset

SCHEMA = "v200-phase1-distribution-audit/1"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentiles(values: np.ndarray) -> dict:
    return {"p10": float(np.percentile(values, 10)), "p50": float(np.median(values)),
            "p90": float(np.percentile(values, 90))}


def audit_distribution(a_dir: Path, rollout_dir: Path) -> dict:
    report_path = rollout_dir / "student-rollout-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("phase1_collection_status") != "collection-ready-for-distribution-audit":
        raise ValueError(f"Phase 1 rollout not accepted: {report.get('phase1_collection_status')}")
    if report.get("training_approved") is not False:
        raise ValueError("Phase 1 data must be diagnostic-only")
    if report.get("checkpoint_sha256") != "7ebc32f61cd92d94fdcc56096d79fad5463d6f7f0cad68ca00b5aae703d3ad21":
        raise ValueError("unexpected student checkpoint; frozen v1.0.5 weight=0 required")
    for shard in report["shards"]:
        if _hash(rollout_dir / shard["name"]) != shard["sha256"]:
            raise ValueError(f"student-state shard hash mismatch: {shard['name']}")
    all_rows: list[dict] = []
    for episode in report["episodes"]:
        trace = rollout_dir / episode["trace"]
        if _hash(trace) != episode["trace_sha256"]:
            raise ValueError(f"trace hash mismatch: {trace}")
        all_rows.extend(json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines())
    student_segments: list[list[int]] = []
    current_segment: list[int] = []
    previous_row = None
    for row in all_rows:
        valid_pair = (
            row["sample_index"] is not None and previous_row is not None
            and previous_row["sample_index"] is not None
            and row["episode"] == previous_row["episode"]
            and row["stage"] == previous_row["stage"]
            and row["deaths_before"] == previous_row["deaths_after"]
            and row["frame"] - previous_row["frame"] in (1, 2)
        )
        if not valid_pair and current_segment:
            student_segments.append(current_segment)
            current_segment = []
        if row["sample_index"] is not None:
            action = row["executed_action"]
            current_segment.append(int(action[2]) * 9 + int(action[0]))
        previous_row = row
    if current_segment:
        student_segments.append(current_segment)
    aligned = [row for row in all_rows if row["sample_index"] is not None]
    if not aligned:
        raise ValueError("no aligned student-state samples")
    if len(aligned) != report["samples"] or sorted(row["sample_index"] for row in aligned) != list(range(len(aligned))):
        raise ValueError("trace-to-shard sample index mapping is not contiguous")
    obs_groups = []
    action_groups = []
    deaths = []
    for entry in report["shards"]:
        with np.load(rollout_dir / entry["name"], allow_pickle=False) as archive:
            n = int(archive["action"].shape[0])
            for key in ("executed_action", "decision_meta", "death_window", "death_window_valid"):
                if key not in archive.files or len(archive[key]) != n:
                    raise ValueError(f"missing or misaligned {key} in {entry['name']}")
            obs_groups.append((archive["self_feat"][:, :2].astype(np.float32),
                               archive["danger"].reshape(n, -1).mean(axis=1), archive["meta"].copy()))
            action_groups.append(archive["executed_action"].copy())
            deaths.append((archive["death_window"].copy(), archive["death_window_valid"].copy()))
    positions = np.concatenate([item[0] for item in obs_groups], axis=0)
    danger = np.concatenate([item[1] for item in obs_groups], axis=0)
    meta = np.concatenate([item[2] for item in obs_groups], axis=0)
    actions = np.concatenate(action_groups, axis=0)
    death_window = np.concatenate([item[0] for item in deaths])
    death_valid = np.concatenate([item[1] for item in deaths])
    for i, row in enumerate(aligned):
        if actions[i].tolist() != row["executed_action"]:
            raise ValueError(f"executed-action mismatch at sample {i}")
        if bool(death_window[i]) != row["death_window"] or bool(death_valid[i]) != row["death_window_valid"]:
            raise ValueError(f"death-window mismatch at sample {i}")
    a_dataset = VisionDataset(a_dir, history_contract="causal-segment-v1", decode_threads=1, use_decode_cache=False)
    try:
        a_meta = np.concatenate([a_dataset.meta_of_shard(path) for path, _, _ in a_dataset.offsets])
        a_positions = np.concatenate([
            a_dataset.labels(list(range(start, start + size)))["self_feat"][:, :2]
            for _path, start, size in a_dataset.offsets
        ], axis=0)
        a_danger = np.concatenate([
            a_dataset.labels(list(range(start, start + size)))["danger"].reshape(size, -1).mean(axis=1)
            for _path, start, size in a_dataset.offsets
        ], axis=0)
        a_actions = np.concatenate([
            a_dataset.labels(list(range(start, start + size)))["action"]
            for _path, start, size in a_dataset.offsets
        ], axis=0)
    finally:
        a_dataset.close()
    base_counts = np.bincount(a_meta[:, 1].astype(np.int64), minlength=8)
    student_counts = np.bincount(meta[:, 1].astype(np.int64), minlength=8)
    a_stage0 = a_meta[:, 1] == 0
    s_stage0 = meta[:, 1] == 0
    result = {
        "schema": SCHEMA, "training_allowed": False,
        "A_dataset": str(a_dir), "A_manifest_sha256": _hash(a_dir / "manifest.json"),
        "rollout": str(rollout_dir), "rollout_report_sha256": _hash(report_path),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "samples": {"A": len(a_meta), "student": len(meta), "decisions": len(all_rows),
                    "aligned_ratio": report["alignment_ratio"]},
        "position_x": {"A": _percentiles(a_positions[:, 0]), "student": _percentiles(positions[:, 0]),
                       "A_stage0": _percentiles(a_positions[a_stage0, 0])},
        "position_y": {"A": _percentiles(a_positions[:, 1]), "student": _percentiles(positions[:, 1]),
                       "A_stage0": _percentiles(a_positions[a_stage0, 1])},
        "danger_mean": {"A": _percentiles(a_danger), "student": _percentiles(danger),
                        "A_stage0": _percentiles(a_danger[a_stage0])},
        "stage_counts": {"A": base_counts.tolist(), "student": student_counts.tolist()},
        "actions": {"A_all_flat_diagnostic": summarize_actions(a_actions.tolist()),
                    "A_stage0_flat_diagnostic": summarize_actions(a_actions[a_stage0].tolist()),
                    "student_segmented": _stats(student_segments),
                    "student_segment_count": len(student_segments)},
        "death_window_50_decisions": {
            "known": int(death_valid.sum()), "positive": int((death_valid & death_window).sum()),
            "unknown": int((1 - death_valid).sum()),
        },
        "capture_latency_ms": {
            "student_inference_p50": float(np.median([r["student_latency_ms"] for r in all_rows])),
            "teacher_p50": float(np.median([r["teacher_latency_ms"] for r in all_rows])),
        },
        "caveat": "observational student-state distribution; not causal proof or an approved training set",
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only v2 Phase 1 distribution audit")
    parser.add_argument("--A", required=True)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = audit_distribution(Path(args.A), Path(args.rollout))
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
