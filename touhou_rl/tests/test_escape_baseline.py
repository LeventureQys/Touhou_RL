"""Baseline fixture tests: the old pilot must not move, and its blind spots must stay visible.

``far-static-wall`` and ``crossing-between-samples`` are not "nice to have": they
are the two evidence cases for the v3.7.0 hypothesis (H1 short horizon, and
SC-08 between-sample crossing).  If a future change "fixes" the old pilot, these
tests fail on purpose, because the comparison arm would have moved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from touhou_rl.escape import baseline
from touhou_rl.pilot import PilotConfig

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "Document/Update/Alpha/v3.7.0/fixtures/baseline-golden.json"


def _records() -> dict[str, dict]:
    return {record["name"]: record for record in baseline.golden_document()["cases"]}


def test_baseline_config_is_the_dataclass_default():
    snapshot = baseline.config_snapshot()
    defaults = PilotConfig()
    assert snapshot["frameskip"] == defaults.frameskip == 2
    assert snapshot["horizon_steps"] == defaults.horizon_steps == 10
    assert snapshot["near_radius"] == defaults.near_radius == 240.0
    assert snapshot["max_bullets"] == defaults.max_bullets == 320
    assert snapshot["player_half"] == defaults.player_half == 0.825
    assert baseline.BASELINE_OVERRIDES == {}
    assert baseline.legacy_config() == PilotConfig()


def test_run_is_deterministic():
    first = baseline.golden_document()["cases"]
    second = baseline.golden_document()["cases"]
    assert first == second


def test_every_case_produces_a_legal_action():
    for record in baseline.golden_document()["cases"]:
        move, shoot, focus, bomb = record["action"]
        assert 0 <= move <= 8
        assert shoot in (0, 1) and focus in (0, 1) and bomb in (0, 1)
        assert len(record["values"]) == 18


def test_far_wall_is_outside_the_baseline_horizon():
    record = _records()["far-static-wall"]
    assert record["hazards_supplied"] == 24
    assert record["bullets_used"] == 0
    assert record["clearance"] == 60.0  # "nothing nearby" placeholder, not a measurement


def test_pool_truncation_is_reported_not_hidden():
    record = _records()["pool-truncation"]
    assert record["hazards_supplied"] == 400
    assert record["bullets_used"] == 320


def test_between_sample_crossing_is_invisible_to_the_baseline():
    record = _records()["crossing-between-samples"]
    analytic = record["analytic"]
    assert analytic["swept_min_clearance"] < 0.0
    assert analytic["sample_grid_min_clearance_x_axis"] > 5.0
    assert analytic["swept_first_segment"] >= 2


def test_mirror_pair_decisions_are_reflected():
    records = _records()
    right = records["mirror-right"]["action"]
    left = records["mirror-left"]["action"]
    mirror = {0: 0, 1: 1, 2: 8, 3: 7, 4: 6, 5: 5, 6: 4, 7: 3, 8: 2}
    assert mirror[right[0]] == left[0]
    assert right[1:] == left[1:]
    assert records["mirror-right"]["clearance"] == pytest.approx(records["mirror-left"]["clearance"])


def test_bomb_threshold_behaviour():
    with_bomb = _records()["dense-unavoidable-with-bomb"]
    without_bomb = _records()["dense-unavoidable-no-bomb"]
    assert with_bomb["action"][3] == 1
    assert with_bomb["action"][2] == 1  # the pilot focuses while bombing
    assert without_bomb["action"][3] == 0
    assert with_bomb["clearance"] < PilotConfig().bomb_min_clearance


def test_committed_fixture_matches_current_code():
    if not FIXTURE.is_file():
        pytest.skip("fixture not written yet")
    assert baseline.verify_golden(FIXTURE) == []


def test_fixture_tampering_is_detected(tmp_path):
    document = baseline.write_golden(tmp_path / "golden.json")
    assert baseline.verify_golden(tmp_path / "golden.json") == []
    document["cases"][0]["action"] = [0, 0, 0, 0]
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    issues = baseline.verify_golden(path)
    assert issues and any("action" in issue for issue in issues)


def test_pilot_code_hash_is_recorded():
    document = baseline.golden_document()
    assert len(document["baseline"]["pilot_sha256"]) == 64
    assert document["baseline"]["config"]["horizon_steps"] == 10
