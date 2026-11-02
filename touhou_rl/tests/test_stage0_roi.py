"""Tests for Stage 0 ROI consistency evidence."""

import json

import numpy as np
import pytest

from touhou_rl.stage0_roi import compare_roi_paths, write_evidence


def test_equal_paths_are_reproducible(tmp_path):
    value = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    result = compare_roi_paths(value, value.copy(), value.copy())
    assert result["schema"] == "v105-stage0-roi-consistency/1"
    assert result["all_equal"] is True
    assert result["within_tolerance"] is True
    assert result["comparisons_to_capture"]["offline"]["mismatch_ratio"] == 0.0
    assert len(result["arrays"]["capture"]["sha256"]) == 64

    path = tmp_path / "roi.json"
    written = write_evidence(path, value, value, value)
    assert json.loads(path.read_text(encoding="utf-8")) == written


def test_difference_reports_pixel_metrics_and_tolerance():
    capture = np.zeros((2, 2, 3), dtype=np.uint8)
    other = capture.copy()
    other[0, 0, 0] = 2
    result = compare_roi_paths(capture, other, capture, tolerance=2)
    row = result["comparisons_to_capture"]["offline"]
    assert row["equal"] is False
    assert row["max_abs"] == 2
    assert row["mismatch_ratio"] == 0.0
    assert result["within_tolerance"] is True


def test_shape_mismatch_is_explicit():
    capture = np.zeros((2, 2, 3), dtype=np.uint8)
    other = np.zeros((3, 2, 3), dtype=np.uint8)
    result = compare_roi_paths(capture, other, capture)
    assert result["within_tolerance"] is False
    assert result["comparisons_to_capture"]["offline"]["shape_match"] is False
    with pytest.raises(ValueError):
        compare_roi_paths(capture, other, capture, tolerance=-1)
