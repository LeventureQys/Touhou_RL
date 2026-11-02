"""Tests for the v1.0.5 Stage 0 action diagnostics."""

import pytest

from touhou_rl.stage0_diagnostics import candidate_index, summarize_actions


def test_candidate_index_uses_frozen_mapping():
    assert candidate_index([4, 1, 0, 0]) == 4
    assert candidate_index([2, 1, 1, 0]) == 11


def test_candidate_index_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        candidate_index([9, 1, 0, 0])
    with pytest.raises(ValueError):
        candidate_index([0, 1, 2, 0])


def test_summary_reports_flips_rates_and_runs():
    stats = summarize_actions([
        [4, 1, 0, 0], [4, 1, 0, 0], [5, 1, 0, 0], [5, 1, 1, 0],
    ])
    assert stats["schema"] == "v105-stage0-action-diagnostics/1"
    assert stats["decisions"] == 4
    assert stats["candidate_flip_rate"] == pytest.approx(2 / 3)
    assert stats["move_flip_rate"] == pytest.approx(1 / 3)
    assert stats["focus_flip_rate"] == pytest.approx(1 / 3)
    assert stats["still_rate"] == 0.0  # move=4 is SE, not idle
    assert summarize_actions([[0, 1, 0, 0], [4, 1, 0, 0]])["still_rate"] == 0.5
    assert stats["focus_rate"] == pytest.approx(1 / 4)
    assert stats["action_run_length"]["max"] == 2
    assert stats["candidate_counts"] == {4: 2, 5: 1, 14: 1}


def test_empty_summary_is_finite_and_explicit():
    stats = summarize_actions([])
    assert stats["decisions"] == 0
    assert stats["action_run_length"] == {"p50": None, "p90": None, "max": None}
    assert stats["candidate_counts"] == {}
