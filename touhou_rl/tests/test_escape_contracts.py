"""Contract tests for ``touhou_rl.escape.contracts`` (no game, no training).

The point of these tests is that a *missing* read cannot be laundered into an
empty pool, that units cannot drift silently, and that the committed schema file
really is the one this code generates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from touhou_rl.escape import contracts as c

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "Document/Update/Alpha/v3.7.0/contract/escape-schema.json"


def observation_doc(**overrides) -> dict:
    document = c.to_json_dict(
        c.EscapeObservation(
            schema=c.OBSERVATION_SCHEMA,
            episode_id="synthetic-0",
            decision_id=3,
            frame_id=120,
            source=c.Source.SYNTHETIC,
            player=c.PlayerObs(x=192.0, y=384.0, focus=0, hurtbox_half=c.HURTBOX_HALF_PX),
            frameskip=2,
            observed_frame_age=0,
            action_latency_frames=1,
            completeness=c.Completeness.COMPLETE,
            coordinate_contract=c.COORDINATE_CONTRACT,
            hazards=(
                c.HazardObs(
                    position=(192.0, 300.0),
                    velocity=(0.0, 3.0),
                    half=(2.0, 2.0),
                    valid_until=180,
                ),
            ),
            previous_action=(5, 1, 0, 0),
            bullet_pool_count=1,
        )
    )
    document.update(overrides)
    return document


def test_generated_schema_covers_every_record():
    schema = c.contract_schema()
    assert schema["$id"] == c.CONTRACT_VERSION
    for record in c.RECORDS.values():
        assert record.__name__ in schema["$defs"]
    required = schema["$defs"]["EscapeObservation"]["required"]
    assert "coordinate_contract" in required and "hazards" not in required


def test_committed_schema_is_generated_not_hand_edited():
    if not SCHEMA_PATH.is_file():
        pytest.skip("schema file not written yet")
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(c.contract_schema()))


def test_valid_observation_passes():
    assert c.validate_document(observation_doc()) == []


def test_unreadable_pool_is_not_an_empty_pool():
    document = observation_doc()
    document["hazards"] = None
    issues = c.validate_document(document)
    assert any("cannot be 'complete'" in issue for issue in issues)
    assert any("missing_reasons" in issue for issue in issues)

    # Naming the reason and downgrading completeness makes it a legal record.
    document["completeness"] = c.Completeness.PARTIAL.value
    document["missing_reasons"] = ["bullets: read failed (OSError)"]
    assert c.validate_document(document) == []


def test_empty_pool_is_legal_and_distinct():
    document = observation_doc()
    document["hazards"] = []
    document["bullet_pool_count"] = 0
    document["previous_action"] = None
    assert c.validate_document(document) == []


def test_pool_count_smaller_than_hazard_list_is_rejected():
    document = observation_doc()
    document["bullet_pool_count"] = 0
    issues = c.validate_document(document)
    assert any("exceed the engine pool count" in issue for issue in issues)


def test_units_cannot_drift():
    document = observation_doc()
    document["coordinate_contract"] = dict(c.COORDINATE_CONTRACT, length_unit="normalised")
    issues = c.validate_document(document)
    assert any("differs from the frozen units" in issue for issue in issues)


def test_extra_field_is_rejected():
    document = observation_doc()
    document["future_truth"] = [1, 2, 3]
    issues = c.validate_document(document)
    assert any("unexpected field" in issue for issue in issues)


def test_bool_is_not_an_integer():
    document = observation_doc()
    document["frame_id"] = True
    issues = c.validate_document(document)
    assert any("expected an integer" in issue for issue in issues)


def test_missing_required_field_is_reported():
    document = observation_doc()
    del document["source"]
    issues = c.validate_document(document)
    assert any("source: missing required field" in issue for issue in issues)


def test_unknown_schema_is_rejected():
    with pytest.raises(c.ContractError):
        c.record_for({"schema": "v999-nope"})
    assert c.validate_document({"schema": "v999-nope"})


def test_forecast_time_axis_must_be_monotonic_and_per_node():
    forecast = c.to_json_dict(
        c.DangerForecast(
            schema=c.FORECAST_SCHEMA,
            model_version="forecast-0",
            time_nodes_frames=(2, 4, 6),
            coverage_shape=(2, 2),
            coverage=((0.0, 0.0, 0.0, 1.0),) * 3,
            uncertainty=(0.1, 0.2, 0.3),
            valid_until=6,
        )
    )
    assert c.validate_document(forecast) == []

    bad = json.loads(json.dumps(forecast))
    bad["time_nodes_frames"] = [2, 2, 6]
    assert any("strictly increasing" in issue for issue in c.validate_document(bad))

    bad = json.loads(json.dumps(forecast))
    bad["coverage"] = [[0.0, 0.0, 0.0, 1.0]]
    issues = c.validate_document(bad)
    assert any("one row per time node" in issue for issue in issues)

    bad = json.loads(json.dumps(forecast))
    bad["coverage"] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert any("expected 4" in issue for issue in c.validate_document(bad))


def test_route_requires_eighteen_first_action_statuses():
    plan = dict(
        schema=c.ROUTE_SCHEMA,
        route_id="r0",
        solution_status=c.SolutionStatus.FEASIBLE.value,
        first_action_status=[c.FirstActionStatus.UNKNOWN.value] * 18,
        action_sequence=[[5, 1, 0, 0]],
        min_clearance=4.0,
    )
    assert c.validate_document(plan) == []

    short = dict(plan, first_action_status=[c.FirstActionStatus.UNKNOWN.value] * 17)
    assert any("minimum 18" in issue for issue in c.validate_document(short))


def test_feasible_route_must_report_clearance_and_actions():
    plan = dict(
        schema=c.ROUTE_SCHEMA,
        route_id="r0",
        solution_status=c.SolutionStatus.FEASIBLE.value,
        first_action_status=[c.FirstActionStatus.UNKNOWN.value] * 18,
    )
    issues = c.validate_document(plan)
    assert any("action_sequence" in issue for issue in issues)
    assert any("min_clearance" in issue for issue in issues)

    # A failed search is a different status, not a fake feasible route.
    failed = dict(plan, solution_status=c.SolutionStatus.BUDGET_EXHAUSTED.value)
    assert c.validate_document(failed) == []


def test_decision_deadline_cannot_be_an_override():
    decision = dict(
        schema=c.DECISION_SCHEMA,
        decision_id=1,
        intent=c.Intent.EVACUATE.value,
        accepted_action=[7, 1, 0, 0],
        reason="pre-registered evacuation",
        deadline_miss=True,
        emergency_override=True,
    )
    assert any("must not be reported" in issue for issue in c.validate_document(decision))


def test_digest_is_canonical():
    first = observation_doc()
    second = json.loads(json.dumps(first))
    second = dict(reversed(list(second.items())))
    assert c.document_digest(first) == c.document_digest(second)
