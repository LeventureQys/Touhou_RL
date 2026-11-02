"""Live shadow harness tests: the same loop, driven by a fake adapter."""

from __future__ import annotations

from touhou_rl.escape import contracts, families, live


def _observation(scenario, frame: int, *, age: int = 0):
    hazards = tuple(
        contracts.HazardObs(
            position=hazard.at(frame),
            velocity=hazard.velocity,
            half=hazard.half,
            motion_model=contracts.MotionModel.LINEAR,
            confidence=1.0,
            valid_until=scenario.horizon_frames,
        )
        for hazard in scenario.hazards
        if hazard.active(frame)
    )
    return contracts.EscapeObservation(
        schema=contracts.OBSERVATION_SCHEMA,
        episode_id=scenario.scenario_id,
        decision_id=frame,
        frame_id=frame,
        source=contracts.Source.SYNTHETIC,
        player=contracts.PlayerObs(
            x=float(scenario.player_start[0]),
            y=float(scenario.player_start[1]),
            focus=0,
            hurtbox_half=contracts.HURTBOX_HALF_PX,
        ),
        frameskip=2,
        observed_frame_age=age,
        action_latency_frames=1,
        completeness=contracts.Completeness.COMPLETE,
        coordinate_contract=dict(contracts.COORDINATE_CONTRACT),
        hazards=hazards,
        bullet_pool_count=len(hazards),
    )


def test_live_loop_collects_decisions_and_verifies_the_frame_counter():
    scenario = families.build("SC-16", 0)
    state = {"frame": 100}

    def read():
        return _observation(scenario, state["frame"])

    def advance():
        state["frame"] += 1

    document = live.run_live(read, advance_fn=advance, decisions=8)
    assert document["schema"] == live.LIVE_SCHEMA
    assert document["live"] is True
    assert document["decisions"] == 8
    assert document["same_frame_verified"] is True
    assert document["complete_ratio"] == 1.0
    assert document["stale_reads"] == 0
    assert document["frame_span"] == [100, 107]
    assert live.b05_check(document)["status"] == "pass"
    assert live.b07_check(document)["status"] == "ready"


def test_live_loop_counts_stale_reads_instead_of_using_them():
    scenario = families.build("SC-16", 0)
    state = {"frame": 200, "calls": 0}

    def read():
        state["calls"] += 1
        age = 5 if state["calls"] % 3 == 0 else 0
        return _observation(scenario, state["frame"], age=age)

    document = live.run_live(read, decisions=6)
    assert document["decisions"] == 6
    assert document["stale_reads"] > 0
    assert document["complete_ratio"] < 1.0
    assert all(row["intent"] != "unknown" for row in document["rows"])


class _FakeReader:
    """Minimal SnapshotReader: two state reads bracket a bullet read per frame."""

    def __init__(self, frames=6, regress_at=None):
        self.calls = 0
        self.regress_at = regress_at

    def _frame(self) -> int:
        frame = 100 + (self.calls - 1) // 2
        if self.regress_at is not None and frame >= self.regress_at:
            frame -= 3
        return frame

    def read_state(self, with_bullets: bool = False, with_title: bool = False):
        from types import SimpleNamespace

        self.calls += 1
        return SimpleNamespace(
            gameplay_frame=self._frame(),
            player_state=0,
            player_x=192.0,
            player_y=384.0,
            focus_mode=0,
            hurtbox_half_x=0.825,
            hurtbox_half_y=0.825,
            bombs=2,
            lives=2,
        )

    def read_bullets(self):
        import numpy as np

        from types import SimpleNamespace

        return SimpleNamespace(
            active_mask=lambda: np.zeros((2,), dtype=bool),
            position=np.zeros((2, 2)),
            velocity=np.zeros((2, 2)),
            half_size=np.full((2, 2), 2.0),
        )


def test_game_source_binds_the_adapter_and_tracks_the_frame_counter():
    reader = _FakeReader()
    read, advance, state = live.game_source(reader, episode_id="fake")
    document = live.run_live(read, advance_fn=advance, decisions=4)
    assert document["decisions"] == 4
    assert state["reads"] >= 4
    assert state["last_frame"] is not None
    # Frames are non-decreasing by construction here; the check must be able to say so.
    assert document["same_frame_verified"] is True


def test_game_source_rejects_a_frame_regression():
    reader = _FakeReader(regress_at=103)
    read, advance, state = live.game_source(reader)
    for _ in range(6):
        observation = read()
        if observation is None:
            break
    assert state["rejected"] >= 1

    def read():
        return None

    try:
        live.run_live(read, decisions=4)
    except live.LiveError as error:
        assert "no observation" in str(error)
    else:  # pragma: no cover - must raise
        raise AssertionError("a read that never returns an observation must raise LiveError")
