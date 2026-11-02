"""Stage 0 ROI consistency evidence helpers.

This module compares uint8 RGB arrays produced by the capture, offline decode,
and deployment crop paths. It does not capture frames or alter the deployment
pipeline; it only creates reproducible evidence from supplied arrays/files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "v105-stage0-roi-consistency/1"


def _array_info(value: np.ndarray) -> dict:
    array = np.asarray(value)
    raw = np.ascontiguousarray(array).tobytes()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def compare_roi_paths(capture, offline, deployment, *, tolerance: int = 0) -> dict:
    """Compare three arrays and return machine-readable, auditable evidence."""
    if isinstance(tolerance, bool) or int(tolerance) < 0:
        raise ValueError(f"tolerance must be a non-negative integer, got {tolerance!r}")
    tolerance = int(tolerance)
    arrays = {
        "capture": np.asarray(capture),
        "offline": np.asarray(offline),
        "deployment": np.asarray(deployment),
    }
    if any(array.ndim != 3 for array in arrays.values()):
        raise ValueError("ROI arrays must be HWC")
    reference = arrays["capture"]
    rows = {name: _array_info(array) for name, array in arrays.items()}
    comparisons = {}
    for name, array in arrays.items():
        if name == "capture":
            continue
        if array.shape != reference.shape:
            comparisons[name] = {
                "shape_match": False,
                "equal": False,
                "max_abs": None,
                "mean_abs": None,
                "mismatch_ratio": None,
            }
            continue
        if not (np.issubdtype(reference.dtype, np.number) and np.issubdtype(array.dtype, np.number)):
            raise ValueError("ROI arrays must have numeric dtypes")
        diff = np.abs(array.astype(np.int64) - reference.astype(np.int64))
        comparisons[name] = {
            "shape_match": True,
            "equal": bool(np.all(diff == 0)),
            "max_abs": int(diff.max()) if diff.size else 0,
            "mean_abs": float(diff.mean()) if diff.size else 0.0,
            "mismatch_ratio": float(np.mean(diff > tolerance)) if diff.size else 0.0,
        }
    return {
        "schema": SCHEMA,
        "tolerance": tolerance,
        "arrays": rows,
        "comparisons_to_capture": comparisons,
        "all_equal": all(item["equal"] for item in comparisons.values()),
        "within_tolerance": all(item["shape_match"] and item["mismatch_ratio"] == 0.0 for item in comparisons.values()),
    }


def write_evidence(path: str | Path, capture, offline, deployment, *, tolerance: int = 0) -> dict:
    """Write comparison evidence as UTF-8 JSON and return the payload."""
    payload = compare_roi_paths(capture, offline, deployment, tolerance=tolerance)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


__all__ = ["SCHEMA", "compare_roi_paths", "write_evidence"]
