"""Unit tests for the Stage 0.5 boss probe (no game process involved)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from touhou_rl import spec
from touhou_rl.boss_probe import (
    ProbeConfig,
    _new_episode,
    _parse_arms,
    _update_episode,
    render_markdown,
    summarise,
    write_report,
)
from touhou_rl.pilot import PilotConfig
from touhou_rl.state import GameState


def test_game_manager_clear_flags_decode():
    state = GameState(
        flags=spec.FLAG_GAME_CLEARED
        | (2 << spec.FLAG_STAGE_TRANSITION_SHIFT)
        | spec.FLAG_STAGE_CLEAR_SEQUENCE
    )
    assert state.game_cleared is True
    assert state.stage_clear_sequence_active is True
    assert state.stage_transition_state == 2


def test_clear_flags_are_off_by_default():
    state = GameState(flags=0)
    assert state.game_cleared is False
    assert state.stage_clear_sequence_active is False
    assert state.stage_transition_state == 0


def test_pilot_aim_safe_margin_validation():
    with pytest.raises(ValueError):
        PilotConfig(aim_safe_margin=-1.0).validate()
    with pytest.raises(ValueError):
        PilotConfig(aim_weight=-0.1).validate()
    PilotConfig(aim_weight=0.6, aim_safe_margin=12.0).validate()


def _episode(arm: str, aim: float, index: int, **overrides) -> dict:
    episode = {
        "schema": "v340-boss-probe/1",
        "arm": arm,
        "aim_weight": aim,
        "episode": index,
        "frames": 6000,
        "steps": 3000,
        "deaths": 1,
        "boss_present_steps": 100,
        "first_boss_frame": 3000,
        "boss_hp_first": 2000,
        "boss_hp_min": 2000,
        "boss_hp_last": 2000,
        "boss_hp_drop_fraction": 0.0,
        "boss_kill": False,
        "clear_sequence_steps": 0,
        "game_cleared_steps": 0,
        "hp_samples": 100,
        "hp_nonfinite": 0,
        "hp_negative": 0,
        "hp_values": [2000, 1999, 1998],
        "spells": [0, 1],
        "termination_reason": "frame stalled",
        "post_terminal_clear": False,
        "post_terminal_stage_advanced": False,
    }
    episode.update(overrides)
    return episode


def test_summarise_damage_and_clear():
    episodes = [
        _episode("teacher", 0.0, 1),
        _episode("teacher", 0.0, 2, frames=6100, deaths=2),
        _episode(
            "teacher-boss", 0.6, 1,
            frames=5500,
            boss_hp_min=500,
            boss_hp_last=500,
            boss_hp_drop_fraction=0.75,
            boss_kill=True,
            clear_sequence_steps=4,
            termination_reason="frame stalled",
            post_terminal_clear=True,
        ),
        _episode(
            "teacher-boss", 0.6, 2,
            frames=5600,
            boss_hp_min=400,
            boss_hp_last=400,
            boss_hp_drop_fraction=0.8,
            boss_kill=True,
            clear_sequence_steps=2,
        ),
    ]
    summary = summarise(episodes, [("teacher", 0.0), ("teacher-boss", 0.6)])
    treatment = summary["arms"]["teacher-boss"]
    assert treatment["episodes"] == 2
    assert treatment["frames_mean"] == 5550.0
    assert treatment["boss_damage_50_rate"] == 1.0
    assert treatment["boss_kill_rate"] == 1.0
    assert treatment["clear_episodes"] == 2
    assert summary["gates"]["G0.1"]["pass"] is True
    assert summary["gates"]["G0.2"]["status"] == "pass"
    assert summary["gates"]["G0.3"]["status"] == "pass"
    assert summary["gates"]["G0.4"]["status"] == "pass"
    assert summary["overall"]["verdict"] == "damage-possible"


def test_summarise_no_boss_is_not_tested():
    episodes = [
        _episode(
            "teacher-boss", 0.6, 1,
            boss_present_steps=0,
            first_boss_frame=None,
            boss_hp_first=None,
            boss_hp_min=None,
            boss_hp_last=None,
            boss_hp_drop_fraction=None,
            hp_samples=0,
            hp_values=[],
            frames=12000,
            termination_reason="max_steps",
        )
    ]
    summary = summarise(episodes, [("teacher-boss", 0.6)])
    assert summary["gates"]["G0.1"]["pass"] is False
    assert summary["gates"]["G0.3"]["status"] == "not_tested"
    assert summary["overall"]["verdict"] == "hp-unreadable"


def test_summarise_negative_hp_fails_g01():
    episodes = [
        _episode(
            "teacher-boss", 0.6, 1,
            hp_negative=3,
            hp_values=[100],
        )
    ]
    summary = summarise(episodes, [("teacher-boss", 0.6)])
    assert summary["gates"]["G0.1"]["pass"] is False
    assert summary["gates"]["G0.1"]["hp_negative"] == 3


def test_parse_arms():
    assert _parse_arms(None, 0.6) == [("teacher-boss", 0.6)]
    assert _parse_arms(["teacher:0.0", "teacher-boss:0.6"], 0.6) == [
        ("teacher", 0.0),
        ("teacher-boss", 0.6),
    ]
    with pytest.raises(ValueError):
        _parse_arms(["teacher"], 0.6)


def test_render_markdown_contains_gate_and_episode():
    episodes = [_episode("teacher-boss", 0.6, 1, boss_kill=True, boss_hp_drop_fraction=0.9)]
    summary = summarise(episodes, [("teacher-boss", 0.6)])
    report = {
        "generated": "2026-10-24T12:00:00",
        "git_head": "deadbeef",
        "config": {"practice_stage": 0, "episodes_per_arm": 1, "max_steps": 6000, "frameskip": 2},
        "episodes": episodes,
        "trace_files": 1,
        **summary,
    }
    text = render_markdown(report)
    assert "Boss 探针" in text
    assert "G0.1" in text
    assert "damage-possible" in text


def test_update_episode_pins_zero_hp_and_kill():
    cfg = ProbeConfig(out_dir=Path("."), run_dir=Path("."), max_steps=10, trace=False)
    ep = _new_episode("arm", 0.6, 1, cfg, GameState())

    def rec(frame, life):
        return {
            "frame": frame,
            "stage": 0,
            "stage_frames": frame,
            "flags": 0,
            "game_cleared": False,
            "stage_clear_sequence_active": False,
            "stage_transition_state": 0,
            "spell": 0,
            "lives": 2.0,
            "deaths": 0,
            "bosses": [{"slot": 0, "pointer": 1, "x": 192.0, "y": 100.0, "life": life}],
            "boss_count": 1,
            "boss_hp_sum": max(0, life),
            "dialogue_present": False,
            "dialogue_msg_idx": -1,
            "stage_clear_screen_state": 0,
            "dialogue_tap": False,
        }

    first = rec(100, 2000)
    zero = rec(200, 0)
    _update_episode(ep, first, None)
    _update_episode(ep, zero, first)
    assert ep["boss_hp_min"] == 0
    assert ep["boss_hp_drop_fraction"] == 1.0
    assert ep["boss_kill"] is True
    assert ep["boss_kill_frame"] == 200


def test_write_report_creates_json_markdown_and_manifest(tmp_path: Path):
    cfg = ProbeConfig(
        out_dir=tmp_path,
        run_dir=tmp_path / "run",
        practice_stage=0,
        episodes_per_arm=1,
        arms=[("teacher-boss", 0.6)],
        max_steps=10,
        trace=False,
    )
    (tmp_path / "trace").mkdir()
    episodes = [_episode("teacher-boss", 0.6, 1, boss_kill=True, boss_hp_drop_fraction=1.0)]
    report = write_report(cfg, episodes, started=0.0)
    assert report["schema"] == "v340-boss-probe/1"
    payload = json.loads((tmp_path / "boss-probe.json").read_text(encoding="utf-8"))
    assert payload["overall"]["verdict"] == "damage-possible"
    assert (tmp_path / "boss-probe.md").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {item["file"] for item in manifest} == {"boss-probe.json", "boss-probe.md"}
