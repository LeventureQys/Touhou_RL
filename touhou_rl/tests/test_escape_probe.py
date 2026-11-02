"""Offline tests for the D8 shadow probe (no game process involved)."""

from __future__ import annotations

import numpy as np

from touhou_rl.escape import contracts, families, probe, synth
from touhou_rl.escape.contracts import Completeness, EscapeObservation, Source


class _FakeSource:
    """Yields a fixed sequence of observations, then stops."""

    def __init__(self, observations):
        self._observations = list(observations)
        self.index = 0
        self.frame = 0

    def read(self):
        if self.index >= len(self._observations):
            return None
        observation = self._observations[self.index]
        self.index += 1
        return observation

    def advance(self):
        self.frame += 2

    def done(self):
        return self.index >= len(self._observations)


def _observation(frame: int, completeness: Completeness = Completeness.COMPLETE) -> EscapeObservation:
    partial = completeness is not Completeness.COMPLETE
    return EscapeObservation(
        schema="v370-escape/observation",
        episode_id="fake",
        decision_id=frame,
        frame_id=frame,
        source=Source.SYNTHETIC,
        player={"x": 192.0, "y": 384.0, "focus": 0, "hurtbox_half": contracts.HURTBOX_HALF_PX},
        frameskip=2,
        observed_frame_age=0,
        action_latency_frames=1,
        completeness=completeness,
        coordinate_contract=dict(contracts.COORDINATE_CONTRACT),
        hazards=None if partial else (),
        bullet_pool_count=0,
        missing_reasons=() if not partial else (),
    )


def test_probe_replays_a_scenario_and_pairs_each_decision():
    scenario = families.edge_instances()["E-01"]
    source = probe.ReplaySource(scenario)
    document = probe.run_probe(source, scenario, config=probe.ProbeConfig(decisions=4))
    assert document["schema"] == "v370-probe/1"
    assert document["live"] is False
    assert document["decisions"] == 4
    assert document["readable"] == 4
    for row in document["rows"]:
        assert row["status"] == "ok"
        assert row["contract_issues"] == []
        assert row["intent"] in {"local", "prepare_exit", "evacuate"}
        assert row["pair"] is not None and "action_survives_24" in row["pair"]
    assert document["chosen_action_survives_rate"] is not None


def test_probe_marks_unreadable_observations_without_crashing():
    scenario = families.edge_instances()["E-11"]  # over the entity cap -> no observation
    source = probe.ReplaySource(scenario)
    document = probe.run_probe(source, scenario, config=probe.ProbeConfig(decisions=3))
    assert document["decisions"] == 3
    assert document["readable"] == 0
    assert all(row["status"] == "unreadable" for row in document["rows"])
    assert all(row["intent"] == "unknown" for row in document["rows"])


def test_probe_never_claims_a_decision_on_an_invalid_observation():
    scenario = synth.Scenario("empty/0", "TEST", 0, (192.0, 384.0), (),
                              horizon_frames=8, hold_frames=0)
    source = _FakeSource([_observation(0, Completeness.PARTIAL)])
    document = probe.run_probe(source, scenario, config=probe.ProbeConfig(decisions=1))
    assert document["readable"] == 1  # the row is readable, the decision is not trusted
    row = document["rows"][0]
    assert row["intent"] == "unknown"
    assert row["contract_issues"]  # the contract validator flagged it


def test_probe_reports_its_own_offline_status():
    scenario = synth.Scenario("empty/0", "TEST", 0, (192.0, 384.0), (),
                              horizon_frames=4, hold_frames=0)
    source = probe.ReplaySource(scenario)
    document = probe.run_probe(source, scenario, config=probe.ProbeConfig(decisions=1))
    assert "live adapter path is not exercised" in document["note"]


def test_adapter_source_reports_the_same_interface_as_the_replay_source():
    """The live wrapper is testable with a fake read point, without the game."""

    calls = {"read": 0, "advance": 0}

    def read_fn():
        calls["read"] += 1
        return _observation(0)

    def advance_fn():
        calls["advance"] += 1

    source = probe.AdapterSource(read_fn=read_fn, advance_fn=advance_fn)
    scenario = synth.Scenario("empty/0", "TEST", 0, (192.0, 384.0), (),
                              horizon_frames=4, hold_frames=0)
    document = probe.run_probe(source, scenario, config=probe.ProbeConfig(decisions=2))
    assert calls["read"] == 2
    assert calls["advance"] == 2
    assert document["decisions"] == 2
    assert source.done() is False  # the live source decides when it is done


def test_replay_source_advances_by_one_decision():
    scenario = synth.Scenario("empty/0", "TEST", 0, (192.0, 384.0), (),
                              horizon_frames=10, hold_frames=0)
    source = probe.ReplaySource(scenario)
    assert source.read() is not None
    source.advance()
    assert source.frame == 2
    for _ in range(5):
        source.advance()
    assert source.done() is True
