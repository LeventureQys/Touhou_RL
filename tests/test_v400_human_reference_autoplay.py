"""Hermetic tests for the real-replay autoplay reference reader (EXP-P0-046).

These tests never start an engine: the protocol client is faked, so the checks
are about the decoding/accounting/fail-closed contract.  The real measurement
lives in the experiment document and its JSON evidence under
``.tmp_v400_human_autoplay*``.
"""
import struct

import pytest

from tools.rl.v400_human_reference_autoplay import (
    AutoplayUnavailable,
    aggregate,
    decode_state,
    main,
    play_stage,
    to_wsl_path,
)


def bits(value: float) -> int:
    return int.from_bytes(struct.pack("<f", value), "little")


def state_fields(*, frame, stage, gameplay, deaths=0.0, lives=2.0, bombs=3.0,
                 score=0, graze=0, power=0.0):
    return {
        "f0": frame,
        "f1": stage,
        "f2": gameplay,
        "f3": gameplay,
        "f4": score,
        "f5": graze,
        "f6": bits(deaths),
        "f7": bits(lives),
        "f8": bits(bombs),
        "f9": bits(power),
    }


def sample(frame, *, gameplay, stage=0, deaths=0.0, lives=2.0, bombs=3.0,
           cur_state=2, is_replay="1", setup_state="0"):
    return {
        "engine_frame": frame,
        "state": decode_state(state_fields(frame=frame, stage=stage, gameplay=gameplay,
                                           deaths=deaths, lives=lives, bombs=bombs)),
        "debug": {
            "curState": str(cur_state),
            "setupState": setup_state,
            "stage": str(stage),
            "isReplay": is_replay,
            "difficulty": "3",
            "shot": "10",
        },
        "note": "",
    }


def test_decode_state_reinterprets_only_float_fields():
    decoded = decode_state(state_fields(frame=7, stage=0, gameplay=3, deaths=1.0, bombs=2.0))
    assert decoded["f0"] == 7
    assert decoded["f1"] == 0
    assert decoded["f3"] == 3
    assert decoded["f6"] == 1.0
    assert decoded["f8"] == 2.0
    assert isinstance(decoded["f6"], float)


def test_aggregate_counts_frames_deaths_and_bombs_from_engine_counters():
    samples = [
        sample(300, gameplay=0, deaths=0.0, bombs=3.0, setup_state="1"),
        sample(316, gameplay=1, deaths=0.0, bombs=2.0),   # bomb used
        sample(400, gameplay=85, deaths=0.0, bombs=2.0),
        sample(500, gameplay=185, deaths=1.0, bombs=2.0),  # death
        sample(600, gameplay=285, deaths=1.0, bombs=3.0),  # bomb item gained
        sample(700, gameplay=385, deaths=1.0, bombs=2.0),  # bomb used
    ]
    row = aggregate(samples, stage=0)
    assert row["gameplay_frames"] == 385
    assert row["replay_input_frames"] == 385
    assert row["engine_frames"] == 700 - 316 + 1
    assert row["deaths"] == 1
    assert row["bombs_used"] == 2
    assert row["bombs_stock_spent"] == 2
    assert row["bombs_gained"] == 1
    assert row["bombs_start"] == 3 and row["bombs_end"] == 2
    assert [event["kind"] for event in row["events"]] == ["bomb", "death", "bomb_gained", "bomb"]


def test_death_bomb_reset_is_not_counted_as_a_bomb_use():
    samples = [
        sample(300, gameplay=0, bombs=7.0, lives=8.0),
        sample(400, gameplay=85, bombs=7.0, lives=8.0),
        sample(500, gameplay=185, deaths=1.0, bombs=3.0, lives=7.0),  # respawn resets the stock
        sample(600, gameplay=285, deaths=1.0, bombs=2.0, lives=7.0),  # a real bomb after respawn
    ]
    row = aggregate(samples, stage=0)
    assert row["deaths"] == 1
    assert row["bombs_used"] == 1
    assert row["bombs_stock_spent"] == 1
    assert row["bombs_reset_on_death"] == 4
    assert [event["kind"] for event in row["events"]] == ["death", "death_bomb_reset", "bomb"]


def test_deathbomb_costs_two_stock_points_but_is_one_activation():
    samples = [
        sample(300, gameplay=0, bombs=5.0),
        sample(400, gameplay=100, bombs=3.0),  # deathbomb: -2 stock, no death
    ]
    row = aggregate(samples, stage=0)
    assert row["deaths"] == 0
    assert row["bombs_used"] == 1
    assert row["bombs_stock_spent"] == 2


def test_aggregate_refuses_gameplay_that_is_not_replay_playback():
    samples = [sample(300, gameplay=0), sample(400, gameplay=100, is_replay="0")]
    with pytest.raises(AutoplayUnavailable, match="without replay playback"):
        aggregate(samples, stage=0)


def test_aggregate_refuses_a_stage_that_never_reached_gameplay():
    samples = [
        sample(60, gameplay=0, cur_state=1, is_replay="0"),
        sample(120, gameplay=0, cur_state=1, is_replay="0"),
    ]
    with pytest.raises(AutoplayUnavailable, match="never reached gameplay"):
        aggregate(samples, stage=0)


def test_aggregate_refuses_when_frame_zero_was_missed():
    samples = [sample(330, gameplay=15), sample(400, gameplay=85)]
    with pytest.raises(AutoplayUnavailable, match="frame zero"):
        aggregate(samples, stage=0)


def test_aggregate_refuses_a_counter_reset_inside_the_window():
    samples = [sample(300, gameplay=0), sample(400, gameplay=100), sample(500, gameplay=10)]
    with pytest.raises(AutoplayUnavailable, match="backwards"):
        aggregate(samples, stage=0)


class FakeClock:
    def __init__(self, step=0.5):
        self.now = 0.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


class FakeClient:
    """Scripted protocol client; each step yields the next canned timeline row."""

    def __init__(self, timeline, *, exit_after=None, game_over=True):
        self.timeline = list(timeline)
        self.index = -1
        self.exit_after = exit_after
        self.game_over = game_over
        self.commands = []
        self.closed = False

    def hello(self):
        self.commands.append("HELLO")
        return {"harness": "1", "proto": "1", "role": "shadow", "frames": "0"}

    def step(self, mask, frames):
        self.commands.append(f"STEP {mask} {frames}")
        if self.index >= len(self.timeline) - 1:
            return {"advanced": "0", "exit": "1", "game_over": "1"}
        self.index += 1
        row = self.timeline[self.index]
        reply = {"advanced": "1", "frame": str(row["engine_frame"]), "hash": "0"}
        if self.exit_after is not None and self.index + 1 >= self.exit_after:
            reply.update(exit="1", game_over="1" if self.game_over else "0")
        return reply

    def request(self, line):
        self.commands.append(line)
        if line != "STATE":
            raise AssertionError(f"unexpected request {line!r}")
        if self.index < 0:
            return "OK state=undefined frames=0"
        row = self.timeline[self.index]
        fields = state_fields(frame=row["engine_frame"], stage=row["stage"],
                              gameplay=row["gameplay"])
        parts = [f"OK frames={row['engine_frame']}"]
        for index in range(25):
            key = f"f{index}"
            value = fields.get(key, 0)
            parts.append(f"{key}={value:08x}" if index in (6, 7, 8, 9, 10, 11, 12)
                         else f"{key}={value}")
        return " ".join(parts)

    def debug(self):
        self.commands.append("DEBUG")
        if self.index < 0:
            return {"curState": "1", "setupState": "0", "stage": "0", "isReplay": "0"}
        row = self.timeline[self.index]
        return {"curState": str(row["cur_state"]), "setupState": row.get("setup_state", "0"),
                "stage": str(row["stage"]), "isReplay": row.get("is_replay", "1"),
                "difficulty": "3", "shot": "10"}

    def close(self):
        self.closed = True


def timeline_rows():
    rows = [
        {"engine_frame": 1, "stage": 0, "gameplay": 0, "cur_state": 1, "is_replay": "0", "setup_state": "0"},
        {"engine_frame": 2, "stage": 0, "gameplay": 0, "cur_state": 1, "is_replay": "0", "setup_state": "0"},
        {"engine_frame": 3, "stage": 0, "gameplay": 0, "cur_state": 2, "is_replay": "1", "setup_state": "0"},
        {"engine_frame": 4, "stage": 0, "gameplay": 1, "cur_state": 2, "is_replay": "1", "setup_state": "0"},
        {"engine_frame": 5, "stage": 0, "gameplay": 2, "cur_state": 2, "is_replay": "1", "setup_state": "0"},
    ]
    return rows


def test_play_stage_reports_measured_stage_on_game_over():
    client = FakeClient(timeline_rows(), exit_after=5)
    result = play_stage(client, stage=0, poll_frames=1, clock=FakeClock(), max_frames=100)
    assert result["status"] == "measured"
    assert result["stage"]["complete"] is True
    assert result["stage"]["end_reason"] == "game_over"
    assert result["stage"]["gameplay_frames"] == 2
    assert result["hello"]["role"] == "shadow"


def test_play_stage_fails_closed_when_gameplay_never_starts():
    rows = [{"engine_frame": i, "stage": 0, "gameplay": 0, "cur_state": 1,
             "is_replay": "0", "setup_state": "0"} for i in range(1, 60)]
    client = FakeClient(rows)
    with pytest.raises(AutoplayUnavailable, match="did not reach gameplay"):
        play_stage(client, stage=0, poll_frames=1, clock=FakeClock(step=10.0),
                   enter_gameplay_seconds=30.0, max_frames=1000)


def test_play_stage_marks_incomplete_when_the_frame_budget_runs_out():
    rows = [{"engine_frame": i, "stage": 0, "gameplay": i - 1, "cur_state": 2, "is_replay": "1"}
            for i in range(1, 6)]
    client = FakeClient(rows)
    result = play_stage(client, stage=0, poll_frames=1, clock=FakeClock(), max_frames=5)
    assert result["status"] == "incomplete"
    assert result["stage"]["complete"] is False
    assert result["stage"]["end_reason"] == "frame_budget"


def test_to_wsl_path_maps_windows_drives():
    assert to_wsl_path(r"D:\WorkShop\Touhou_RL\th08_replay\Th8_ud0059.rpy") == \
        "/mnt/d/WorkShop/Touhou_RL/th08_replay/Th8_ud0059.rpy"


def test_cli_fails_closed_without_a_replay_file(tmp_path, capsys):
    output = tmp_path / "evidence.json"
    code = main(["--replay", str(tmp_path / "missing.rpy"), "--output", str(output)])
    assert code == 3
    report = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "unavailable"
    assert report["human_reference_created"] is False
    assert "replay not found" in report["reason"]
