"""Unit tests for the v3.5.0 S0 trace contract (no game process)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from touhou_rl import spec
from touhou_rl.state import GameState, ShotParams
from touhou_rl.trace35 import (
    REQUIRED_GROUPS,
    Snapshot,
    TraceWriter,
    assess_complete,
    build_record,
    classify_terminal,
    events_between,
    stable_split,
    state_snapshot,
    steady_exclusion,
)


def _state(**overrides) -> GameState:
    values = dict(
        supervisor_state=spec.STATE_GAME_MANAGER,
        stage=0,
        stage_frames=90,
        gameplay_frame=102,
        run_frames=102,
        flags=0,
        player_state=0,
        player_x=192.0,
        player_y=300.0,
        player_vx=0.0,
        player_vy=0.0,
        focus_mode=0,
        hurtbox_half_x=0.825,
        hurtbox_half_y=0.825,
        power=64.0,
        lives=2.0,
        bombs=3.0,
        bombs_used=0,
        deaths=0,
        score=100,
        graze=10,
        active_bullets=5,
        gui_read_ok=True,
        gui_dialogue_msg_idx=-1,
    )
    values.update(overrides)
    return GameState(**values)


def _snapshot(count: int = 5, **overrides) -> Snapshot:
    payload = {
        "position": np.zeros((count, 2), dtype=np.float32),
        "velocity": np.zeros((count, 2), dtype=np.float32),
        "half": np.ones((count,), dtype=np.float32),
        "counter": count,
    }
    return Snapshot(
        state=_state(active_bullets=count),
        wall_ms=1000,
        enemies=[],
        items=[],
        bosses=[],
        shot=ShotParams(10.0, 24.0, 128.0),
        bullets=payload,
        status={name: "ok" for name in REQUIRED_GROUPS},
    )


def _record(**overrides) -> dict:
    snapshot = overrides.pop("snapshot", None)
    if snapshot is None:
        snapshot = _snapshot()
    kwargs = dict(
        episode="ep01", decision=1, frameskip=2, frame_pre=100, wall_ms=1000,
        pre_state=None, snapshot=snapshot, action=[1, 1, 0, 0],
    )
    kwargs.update(overrides)
    return build_record(**kwargs)


def test_events_between_edges():
    prev = state_snapshot(_state(power=64.0, lives=2.0, deaths=0))
    cur = state_snapshot(
        _state(
            power=128.0,
            lives=-1.0,
            deaths=1,
            flags=spec.FLAG_STAGE_CLEAR_SEQUENCE,
        )
    )
    events = events_between(prev, cur)
    assert "death" in events
    assert "stage_clear" in events
    assert "game_over" in events
    assert "power_max" in events
    # No event should be emitted for the first decision.
    assert events_between(None, cur) == []


def test_events_between_dialogue_and_transition():
    prev = state_snapshot(_state(gui_dialogue_present=False))
    cur = state_snapshot(_state(gui_dialogue_present=True, gui_dialogue_msg_idx=3))
    assert "dialogue_start" in events_between(prev, cur)
    assert "dialogue_end" in events_between(cur, prev)


def test_build_record_complete():
    record = _record()
    assert record["schema"] == "v350-trace/1"
    assert record["complete"] is True
    assert record["incomplete_reasons"] == []
    assert record["bullets"]["count"] == 5
    assert record["bullets"]["counter"] == 5


def test_complete_rejects_frame_mismatch_and_bad_schema():
    record = _record(frame_pre=99)
    ok, reasons = assess_complete(record)
    assert ok is False and "frame_mismatch" in reasons

    snapshot = _snapshot()
    snapshot.status["enemies"] = "error"
    record = _record(snapshot=snapshot)
    ok, reasons = assess_complete(record)
    assert ok is False and "schema_enemies_error" in reasons


def test_complete_rejects_bullet_count_out_of_range():
    record = _record()
    record["bullets"]["count"] = spec.BULLET_COUNT + 1
    ok, reasons = assess_complete(record)
    assert ok is False and "bullet_count_out_of_range" in reasons


def test_complete_rejects_outside_move_area():
    snapshot = _snapshot()
    snapshot.state.player_x = -100.0
    record = _record(snapshot=snapshot)
    ok, reasons = assess_complete(record)
    assert ok is False and "player_outside_move_area" in reasons


def test_complete_continuity_between_records():
    first = _record()
    snap2 = _snapshot()
    snap2.state.gameplay_frame = 104
    second = _record(snapshot=snap2, decision=2, frame_pre=102, wall_ms=1001)
    assert assess_complete(second, first)[0] is True
    snap3 = _snapshot()
    snap3.state.gameplay_frame = 104
    broken = _record(snapshot=snap3, decision=4, frame_pre=104, wall_ms=1001)
    ok, reasons = assess_complete(broken, second)
    assert ok is False and "decision_not_contiguous" in reasons


def test_steady_exclusion():
    record = _record()
    assert steady_exclusion(record) is None
    dying = _record(snapshot=_snapshot())
    dying["player"]["state"] = 2
    assert steady_exclusion(dying) == "player_spawning_or_dying"
    clear = _record()
    clear["events"] = ["stage_clear"]
    assert steady_exclusion(clear) == "terminal_transition"
    paused = _record()
    paused["events"] = ["dialogue_start"]
    assert steady_exclusion(paused) == "paused_transition"


def test_classify_terminal_priority():
    final = _state(lives=-1.0, flags=spec.FLAG_GAME_CLEARED)
    assert classify_terminal("frame stalled", final, [])["classified"] == "game_clear"
    final = _state(lives=-1.0, flags=spec.FLAG_STAGE_CLEAR_SEQUENCE)
    assert classify_terminal("frame stalled", final, [])["classified"] == "stage_clear"
    final = _state(lives=-1.0)
    assert classify_terminal("frame stalled", final, [])["classified"] == "game_over"
    final = _state(supervisor_state=spec.STATE_TITLE_SCREEN)
    assert classify_terminal("left game", final, [])["classified"] == "left_game"
    assert classify_terminal("max_steps", _state(), [])["classified"] == "max_steps"
    assert classify_terminal("frame stalled", _state(), [])["classified"] == "frame_stalled"


def test_classify_terminal_frame_stall_with_zero_lives_is_game_over():
    # The observed TH08 death screen: frame counter frozen, lives exactly 0.0,
    # player_state 2 (dying) then 1 (spawning), no dialogue/clear flags.
    final = _state(lives=0.0, deaths=3, player_state=2)
    assert classify_terminal("frame stalled", final, [])["classified"] == "game_over"
    # A stall with lives > 0 is still unknown: it must not be called a game over.
    assert classify_terminal("frame stalled", _state(lives=2.0, player_state=0), [])["classified"] == "frame_stalled"


def test_classify_terminal_uses_post_samples():
    final = _state()
    samples = [{"stage_clear_sequence_active": True, "stage": 0, "lives": 2.0,
                "supervisor_state": spec.STATE_GAME_MANAGER}]
    assert classify_terminal("frame stalled", final, samples)["classified"] == "stage_clear"


def test_stable_split_is_deterministic_and_single_valued():
    first = stable_split("s0-teacher-ep01")
    assert first in ("train", "validation")
    assert stable_split("s0-teacher-ep01") == first
    assert stable_split("s0-teacher-ep02") != "unknown"


def test_trace_writer_roundtrip(tmp_path: Path):
    writer = TraceWriter(tmp_path, "ep01")
    snap1 = _snapshot(count=3)
    rec1 = _record(snapshot=snap1)
    writer.write(rec1, snap1)
    snap2 = _snapshot(count=5)
    snap2.state.gameplay_frame = 104
    rec2 = _record(decision=2, frame_pre=102, wall_ms=1001, snapshot=snap2)
    writer.write(rec2, snap2)
    info = writer.close()
    assert info["decisions"] == 2
    lines = (tmp_path / "trace" / "ep01.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    loaded = json.loads(lines[0])
    assert loaded["bullets"]["sidecar_index"] == 0
    data = np.load(tmp_path / "bullets" / "ep01.npz")
    assert data["offsets"].tolist() == [0, 3, 8]
    assert data["counters"].tolist() == [3, 5]


def test_trace_writer_without_snapshot_still_indexes(tmp_path: Path):
    writer = TraceWriter(tmp_path, "ep02")
    writer.write(_record(), None)
    writer.close()
    data = np.load(tmp_path / "bullets" / "ep02.npz")
    assert data["offsets"].tolist() == [0, 0]
