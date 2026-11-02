from __future__ import annotations

import numpy as np
import pytest

from touhou_rl.stage4_diagnostics import analyze_action_sequences


def test_stage4_resets_continuity_at_segment_boundaries():
    result = analyze_action_sequences(
        [4, 4, 3, 3, 3],
        [4, 3, 3, 3, 4],
        [0, 0, 0, 3, 3],
    )
    assert result["segments"] == 2
    assert result["human"]["decisions"] == 5
    # The boundary between 3 and 3 is not counted as a flip across segments.
    assert result["human"]["candidate_flip_rate"] == pytest.approx(1 / 3)
    assert result["human_teacher_agreement"] == pytest.approx(3 / 5)


def test_stage4_rejects_mismatched_or_invalid_candidates():
    with pytest.raises(ValueError):
        analyze_action_sequences([0], [0, 1], [0])
    with pytest.raises(ValueError):
        analyze_action_sequences([18], [0], [0])


def test_stage4_empty_is_well_formed():
    result = analyze_action_sequences([], [], [])
    assert result["frames"] == 0
    assert result["segments"] == 0
    assert result["human"]["action_run_length"]["max"] is None
