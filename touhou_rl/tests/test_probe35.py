"""Unit tests for the v3.5.0 S0 offline gates and P-contract analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from touhou_rl import spec
from touhou_rl.probe35 import ProbeConfig, analyse_p_contract, episode_plan, verify, write_manifest
from touhou_rl.trace35 import stable_split


def _record(decision: int, frame_pre: int, wall: int, events: list[str] | None = None) -> dict:
    return {
        "schema": "v350-trace/1",
        "episode": "placeholder",
        "decision": decision,
        "frameskip": 2,
        "frame_pre": frame_pre,
        "frame_post": frame_pre + 2,
        "wall_ms": wall,
        "complete": True,
        "incomplete_reasons": [],
        "schema_status": {
            "player": "ok", "bullets": "ok", "enemies": "ok", "items": "empty",
            "boss": "empty", "power": "ok", "shot": "ok", "gui": "ok",
        },
        "clock": {"flags": 0, "stage": 0},
        "player": {"state": 0, "x": 192.0, "y": 300.0, "focus_mode": 0},
        "resources": {"power": 10.0},
        "bullets": {"count": 1, "counter": 1},
        "events": events or [],
        "items": [],
        "enemies": [],
        "bosses": [],
        "shot": {"point_item_value_line": 128.0},
        "frame_events": {"bits": 0, "item_collected": False},
        "action": [0, 1, 0, 0],
    }


def _episode(name: str, reason: str, decisions: int, evidence: dict | None = None) -> dict:
    return {
        "schema": "v350-episode/1",
        "episode": name,
        "mode": "p-contract",
        "split": stable_split(name),
        "termination": {"reason": reason, "evidence": evidence or {}},
        "stats": {"decisions": decisions, "steady_decisions": decisions, "complete_decisions": decisions},
    }


def _write_trace(out: Path, name: str, records: list[dict]) -> None:
    path = out / "trace" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            record = dict(record)
            record["episode"] = name
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _make_out(tmp_path: Path) -> Path:
    out = tmp_path / "s0"
    (out / "trace").mkdir(parents=True)
    episodes = [
        _episode("ep01", "game_over", 3, {"game_over": True, "game_clear": False,
                                          "stage_advanced": False, "clear_sequence": False,
                                          "clear_screen": False, "post_samples": 2}),
        _episode("ep02", "max_steps", 2),
    ]
    (out / "episodes.jsonl").write_text(
        "\n".join(json.dumps(ep, ensure_ascii=False) for ep in episodes) + "\n", encoding="utf-8"
    )
    _write_trace(out, "ep01", [_record(1, 0, 100), _record(2, 2, 110), _record(3, 4, 120)])
    _write_trace(out, "ep02", [_record(1, 0, 100), _record(2, 2, 110)])
    cfg = ProbeConfig(out_dir=out, mode="collect", episodes=2, trace=True)
    write_manifest(out, cfg, episodes)
    return out


def test_verify_passes_on_clean_synthetic_trace(tmp_path: Path):
    out = _make_out(tmp_path)
    report = verify(out, out / "manifest.json")
    assert report["gates"]["S0-1"]["pass"] is True
    assert report["gates"]["S0-2"]["pass"] is True
    assert report["gates"]["S0-3"]["pass"] is True
    assert report["gates"]["S0-4"]["pass"] is True
    assert report["gates"]["S0-5"]["pass"] is True
    assert report["gates"]["S0-M"]["pass"] is True
    assert report["overall"]["pass"] is True


def test_verify_fails_when_trace_is_missing(tmp_path: Path):
    out = _make_out(tmp_path)
    (out / "trace" / "ep02.jsonl").unlink()
    report = verify(out, out / "manifest.json")
    assert report["overall"]["pass"] is False
    assert "ep02" in report["diagnostics"]["missing_traces"]
    assert report["manifest"]["pass"] is False


def test_verify_fails_when_completion_is_low(tmp_path: Path):
    out = _make_out(tmp_path)
    records = [_record(1, 0, 100), _record(2, 2, 110), _record(3, 4, 120)]
    for record in records:
        record["complete"] = False
    _write_trace(out, "ep01", records)
    cfg = ProbeConfig(out_dir=out, mode="collect", episodes=2)
    write_manifest(out, cfg, [
        _episode("ep01", "game_over", 3, {"game_over": True}),
        _episode("ep02", "max_steps", 2),
    ])
    report = verify(out, out / "manifest.json")
    assert report["gates"]["S0-4"]["pass"] is False
    assert report["overall"]["pass"] is False


def test_verify_fails_when_no_game_terminal(tmp_path: Path):
    out = _make_out(tmp_path)
    episodes = [_episode("ep01", "max_steps", 3), _episode("ep02", "max_steps", 2)]
    (out / "episodes.jsonl").write_text(
        "\n".join(json.dumps(ep, ensure_ascii=False) for ep in episodes) + "\n", encoding="utf-8"
    )
    cfg = ProbeConfig(out_dir=out, mode="collect", episodes=2)
    write_manifest(out, cfg, episodes)
    report = verify(out, out / "manifest.json")
    assert report["gates"]["S0-3"]["pass"] is False
    assert report["gates"]["S0-3"]["denominator"] == 0


# ---------------------------------------------------------------------------
# P contract analysis
# ---------------------------------------------------------------------------


def _p_record(decision: int, frame_pre: int, player_y: float, power: float,
              focus: int, items: list[dict], events: list[str] | None = None) -> dict:
    record = _record(decision, frame_pre, 100 + decision)
    record.update(
        {
            "player": {"state": 0, "x": 192.0, "y": player_y, "focus_mode": focus},
            "resources": {"power": power},
            "items": items,
            "events": events or [],
            "action": [0, 1, focus, 0],
            "shot": {"point_item_value_line": 128.0},
            "frame_events": {"bits": 0, "item_collected": False},
        }
    )
    return record


def test_p_contract_analysis_happy_path(tmp_path: Path):
    out = tmp_path / "p"
    (out / "trace").mkdir(parents=True)
    episodes = [_episode("p-contract-ep01", "max_steps", 14)]
    (out / "episodes.jsonl").write_text(json.dumps(episodes[0], ensure_ascii=False) + "\n", encoding="utf-8")
    p_item = {"slot": 1, "type": 0, "x": 100.0, "y": 300.0, "state": 0, "onscreen": True}
    records = []
    for decision in range(1, 15):
        if decision == 1:
            player_y, power, items, focus = 300.0, 0.0, [p_item], 1
        elif decision == 2:
            player_y, power, items, focus = 126.0, 0.0, [{**p_item, "state": 1}], 1
        elif decision < 10:
            player_y, power, items, focus = 120.0, 0.0, [{**p_item, "state": 1}], 1
        elif decision == 10:
            # The auto-collected item disappears and power rises by exactly 1.
            player_y, power, items, focus = 120.0, 1.0, [], 1
        elif decision <= 12:
            player_y, power, items, focus = 120.0, 1.0, [], 1
        else:
            player_y, power, items, focus = 120.0, 1.0, [], 0
        records.append(_p_record(decision, (decision - 1) * 2, player_y, power, focus, items))
    _write_trace(out, "p-contract-ep01", records)
    report = analyse_p_contract(out, episodes)
    assert report["hypotheses"]["H-P3"]["pass"] is True
    assert report["hypotheses"]["H-P4"]["pass"] is True
    assert report["hypotheses"]["H-P5"]["pass"] is True
    assert report["hypotheses"]["H-P6"]["pass"] is True
    # 1 power is below the cap: H-P1/H-P2 must be reported as not tested, not as pass.
    assert report["hypotheses"]["H-P1"]["not_tested"] is True
    assert report["summary"]["pass"] is False


# ---------------------------------------------------------------------------
# v3.6.0 D2: batch replay collection
# ---------------------------------------------------------------------------


def _fake_replay(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"T8RP" + b"\x00" * 32)
    return path


def test_episode_plan_teacher_loop():
    cfg = ProbeConfig(episodes=3)
    assert episode_plan(cfg, 1) == [(1, None), (2, None), (3, None)]
    assert episode_plan(cfg, 5) == [(5, None), (6, None), (7, None)]


def test_episode_plan_single_replay_arm(tmp_path):
    replay = _fake_replay(tmp_path, "a.rpy")
    cfg = ProbeConfig(mode="p-contract", episodes=2, replay_path=replay)
    assert episode_plan(cfg, 1) == [(1, replay), (2, replay)]


def test_episode_plan_replay_batch_one_episode_per_replay(tmp_path):
    first = _fake_replay(tmp_path, "a.rpy")
    second = _fake_replay(tmp_path, "b.rpy")
    cfg = ProbeConfig(mode="collect", episodes=len(cfg_episodes := [first, second]), replays=(first, second))
    plan = episode_plan(cfg, 1)
    assert plan == [(1, first), (2, second)]
    # The batch ignores ``episodes``: one episode per replay, never padded.
    assert episode_plan(ProbeConfig(mode="collect", episodes=9, replays=(first,)), 1) == [(1, first)]


def test_validate_rejects_multiple_replays_for_p_contract(tmp_path):
    first = _fake_replay(tmp_path, "a.rpy")
    second = _fake_replay(tmp_path, "b.rpy")
    cfg = ProbeConfig(mode="p-contract", replays=(first, second))
    with pytest.raises(ValueError, match="at most one replay"):
        cfg.validate()


def test_validate_rejects_replay_with_policy(tmp_path):
    replay = _fake_replay(tmp_path, "a.rpy")
    cfg = ProbeConfig(mode="collect", policy="suicide", replays=(replay,))
    with pytest.raises(ValueError, match="does not take a policy"):
        cfg.validate()


def test_validate_rejects_missing_replay(tmp_path):
    cfg = ProbeConfig(mode="collect", replays=(tmp_path / "missing.rpy",))
    with pytest.raises(ValueError, match="replay not found"):
        cfg.validate()


def test_validate_accepts_replay_batch_and_records_it(tmp_path):
    first = _fake_replay(tmp_path, "a.rpy")
    cfg = ProbeConfig(mode="collect", replays=(first,))
    cfg.validate()
    assert cfg.as_dict()["replays"] == [str(first)]
    assert cfg.as_dict()["replay_path"] is None
