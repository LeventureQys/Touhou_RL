"""Synthetic-only tests: no game process or collector imports."""
from copy import deepcopy
import pytest
from touhou_rl.rollout_contract import SCHEMA_VERSION, validate_timing_row, validate_trace


def row(decision=0):
    start = decision * 10
    return {"rollout_id": "synthetic", "episode": 1, "decision": decision,
            "stage": 1, "frame": start, "executed_action": [0, 1, 0, 0],
            "timestamps": {name: {"stage": 1, "frame": start + offset}
                           for name, offset in [("capture_before", 0), ("capture_after", 1),
                                                ("action_issue", 2), ("action_effective", 3), ("step_end", 7)]}}


def bullets(value, generation=1, state=1):
    value["raw_snapshot"] = {"bullets": {"slot_id": [3], "state": [state], "generation": [generation]}}
    return value


def test_normalization_is_independent_and_idempotent():
    original = row()
    normalized = validate_timing_row(original)
    assert normalized["schema"] == SCHEMA_VERSION
    assert normalized["actual_hold_frames"] == 4
    assert normalized["action_hold_frames_valid"] is True
    assert validate_timing_row(normalized) == normalized
    assert "schema" not in original


@pytest.mark.parametrize("field,value", [("episode", 0), ("decision", -1), ("stage", True),
                                        ("frame", 1.2), ("rollout_id", 4), ("executed_action", [0]*3),
                                        ("executed_action", [0, 0, 0, True])])
def test_invalid_required_fields(field, value):
    data = row()
    data[field] = value
    with pytest.raises(ValueError):
        validate_timing_row(data)


@pytest.mark.parametrize("field", ["rollout_id", "episode", "decision", "stage", "frame", "executed_action", "timestamps"])
def test_missing_fields(field):
    data = row()
    del data[field]
    with pytest.raises(ValueError):
        validate_timing_row(data)


def test_order_and_zero_hold():
    data = row()
    data["timestamps"]["action_issue"]["frame"] = 0
    with pytest.raises(ValueError):
        validate_timing_row(data)
    data = row()
    data["timestamps"]["step_end"]["frame"] = 3
    normalized = validate_timing_row(data)
    assert normalized["actual_hold_frames"] is None
    assert normalized["reasons"] == ["zero_frame_hold"]


def test_transition_and_cross_stage_duration():
    data = row()
    data["timestamps"]["step_end"] = {"stage": 2, "frame": 0}
    with pytest.raises(ValueError):
        validate_timing_row(data)
    data["transition"] = True
    assert validate_timing_row(data)["actual_hold_frames"] is None
    data["timestamps"]["capture_after"] = {"stage": 2, "frame": 0}
    with pytest.raises(ValueError):
        validate_timing_row(data)


def test_readback():
    data = row()
    data["input_mask_readback"] = {"mask": 1, "stage": 1, "frame": 3}
    assert validate_timing_row(data)["input_mask_readback_valid"] is True
    data["input_mask_readback"]["frame"] = 2
    assert validate_trace([data])["reason_counts"] == {"input_mask_readback_mismatch": 1}


@pytest.mark.parametrize("change", ["length", "duplicate", "negative", "generation"])
def test_malformed_bullets(change):
    data = bullets(row())
    arrays = data["raw_snapshot"]["bullets"]
    if change == "length":
        arrays["state"] = []
    elif change == "duplicate":
        arrays.update(slot_id=[3, 3], state=[1, 1], generation=[0, 0])
    elif change == "negative":
        arrays["slot_id"] = [-1]
    else:
        arrays["generation"] = [True]
    with pytest.raises(ValueError):
        validate_timing_row(data)


def test_trace_identity_and_reordering():
    a, b = row(), row(1)
    report = validate_trace([a, b])
    assert report["duration_valid_count"] == 2
    assert report["generation_issue_count"] == 0
    assert len(report["sha256"]) == 64
    assert report == validate_trace([dict(reversed(list(a.items()))), b])
    for rows in ([a, a], [b, a]):
        with pytest.raises(ValueError):
            validate_trace(rows)
    for field, value in [("episode", 2), ("rollout_id", "other"), ("frame", 0)]:
        bad = deepcopy(b)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_trace([a, bad])


def test_generation_regression_and_reuse():
    report = validate_trace([bullets(row(), 2), bullets(row(1), 1)])
    assert report["reason_counts"] == {"generation_regression": 1}
    report = validate_trace([bullets(row(), 2, 0), bullets(row(1), 2, 1)])
    assert report["reason_counts"] == {"slot_reuse_without_generation_increment": 1}
    assert validate_trace([bullets(row(), 2, 0), bullets(row(1), 3, 1)])["valid"]
    middle = row(1)
    middle["raw_snapshot"] = {"bullets": {"slot_id": [], "state": [], "generation": []}}
    assert validate_trace([bullets(row(), 2), middle, bullets(row(2), 2)])["generation_issue_count"] == 1


def test_collector_unknown_effective_stamp():
    data = row()
    data["timestamps"]["action_effective"] = None
    data["raw_safety_snapshot"] = {"generation_valid": False, "bullets": {"slot_id": [1], "state": [1]}}
    result = validate_timing_row(data)
    assert result["action_hold_frames_valid"] is False
    assert result["actual_hold_frames"] is None
    assert result["reasons"] == ["effective_stamp_missing"]


def test_bad_action_and_readback_invalidate_hold():
    data = row(); data["executed_action"][0] = 9
    with pytest.raises(ValueError): validate_timing_row(data)
    data = row(); data["input_mask_readback"] = {"mask": 1, "stage": 1, "frame": 2}
    assert validate_timing_row(data)["action_hold_frames_valid"] is False


def test_empty_trace():
    assert validate_trace([])["row_count"] == 0
