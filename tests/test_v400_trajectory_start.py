"""Tests for ``TrajectoryStart`` (03 §5, task book T3 step 1).

The engine is not available on every host these tests run on, so the replay
logic is exercised against a *fake* environment that implements the three things
the replay needs (``reset``, ``advance``, ``last_reset_hash``/``shm``) and can be
told to diverge, exit, or come up short.  That is also the only way to test the
fail-closed paths: a real engine does not produce a wrong hash on demand.

The one live test is skipped unless ``V400_TRAJ_LIVE_BINARY`` points at a harness
binary (inside WSL); the acceptance evidence for the real engine is produced by
``touhou_rl/v400/gym/trajectory_verify.py``, not here.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from touhou_rl.v400.gym.env import FrameReply, StageStart, Th08Env
from touhou_rl.v400.gym.trajectory import (
    RecordedTrajectory,
    TrajectoryAborted,
    TrajectoryCorpusError,
    TrajectoryHashMismatch,
    TrajectoryHashUnavailable,
    TrajectoryMaskError,
    TrajectoryStart,
    TrajectoryTargetError,
    load_d0_shard,
    load_d0_shards,
    parse_reset_command,
    replay_to_frame,
    sample_target_frame,
)

H0 = "7f04c7361749ed9f"
H4 = "e3b18537a740bc49"
H8 = "7002f5485e5f90ec"
H12 = "bbb0417a889e8d60"


# ---------------------------------------------------------------------------
# A fake environment: same three methods the replay uses
# ---------------------------------------------------------------------------
class _FakeBlock:
    def __init__(self, engine: "FakeEngine") -> None:
        self.engine = engine

    def header(self) -> dict[str, int]:
        return {"frame": self.engine.frame}


class FakeEngine:
    """Stands in for ``Th08Env`` and records every per-frame advance."""

    def __init__(self, hashes: dict[int, str], *, reset_hash: str | None = None,
                 wrong: dict[int, str] | None = None, exit_at: int | None = None,
                 game_over_at: int | None = None, short_at: int | None = None,
                 no_shm: bool = False) -> None:
        self.hashes = dict(hashes)
        self._reset_hash = reset_hash if reset_hash is not None else self.hashes.get(0)
        self.wrong = dict(wrong or {})
        self.exit_at = exit_at
        self.game_over_at = game_over_at
        self.short_at = short_at
        self.no_shm = no_shm
        self.frame = 0
        self.calls: list[tuple[int, int]] = []
        self.reset_start: StageStart | None = None

    # -- the contract replay_to_frame relies on ---------------------------
    def reset(self, start: StageStart):
        self.reset_start = start
        self.frame = 0
        self.calls = []
        return None, None

    @property
    def last_reset_hash(self) -> str | None:
        return self._reset_hash

    @property
    def shm(self):
        return None if self.no_shm else _FakeBlock(self)

    def advance(self, mask: int, frames: int = 1) -> FrameReply:
        self.calls.append((int(mask), int(frames)))
        target = self.frame + frames
        for boundary, game_over in ((self.exit_at, False), (self.game_over_at, True)):
            if boundary is not None and target >= boundary:
                advanced = boundary - self.frame
                self.frame = boundary
                return FrameReply(frame=boundary, advanced=advanced, hash=None,
                                  exit=not game_over, game_over=game_over)
        if self.short_at is not None and target >= self.short_at:
            self.frame = self.short_at - 1
            return FrameReply(frame=self.frame, advanced=frames - 1,
                              hash=self.hashes.get(self.frame, "0" * 16))
        self.frame = target
        value = self.wrong.get(target, self.hashes.get(target, "0" * 16))
        return FrameReply(frame=target, advanced=frames, hash=value)


def start_for(masks, target, hashes, **kwargs) -> TrajectoryStart:
    return TrajectoryStart(seed=1, stage_index=0, difficulty="normal", masks=list(masks),
                           target_frame=target, hashes=hashes, **kwargs)


# ---------------------------------------------------------------------------
# TrajectoryStart: construction is fail-closed
# ---------------------------------------------------------------------------
def test_start_normalises_masks_and_derives_stage_start():
    start = start_for([65, 65, 64, 64], 4, {0: H0, 4: H4})
    assert start.masks == (65, 65, 64, 64)
    assert start.frames == 4
    assert start.stage_start == StageStart(stage=0, difficulty="normal", team=0, seed=1)
    assert start.expected_target_hash == H4
    assert start.as_record()["masks"] == 4


def test_start_accepts_practice_resources_and_team():
    start = start_for([0, 0], 2, {0: H0, 2: H8}, resources="practice", team=3)
    assert start.stage_start.resources == "practice"
    assert start.stage_start.team == 3


def test_start_rejects_unknown_difficulty():
    with pytest.raises(ValueError):
        TrajectoryStart(seed=1, stage_index=0, difficulty="nightmare", masks=[0],
                        target_frame=0, hashes={0: H0})


def test_start_rejects_target_beyond_the_tape():
    with pytest.raises(TrajectoryMaskError):
        start_for([65, 65], 3, {0: H0, 3: H4})


def test_start_rejects_target_beyond_the_tape_with_a_counting_message():
    """The 'mask tape too short' failure names the tape length and the target."""
    with pytest.raises(TrajectoryMaskError) as error:
        start_for([65, 65], 3, {0: H0, 3: H4})
    message = str(error.value)
    assert "2 frames" in message and "target_frame is 3" in message


def test_start_accepts_the_final_frame_of_the_tape():
    """``target_frame == len(masks)`` is the last recorded frame, not an overflow."""
    start = start_for([65] * 8, 8, {0: H0, 8: H8})
    assert start.target_frame == 8 and start.frames == 8


def test_start_rejects_negative_target():
    with pytest.raises(TrajectoryTargetError):
        start_for([65, 65], -1, {0: H0})


def test_start_rejects_mask_out_of_range():
    for bad in (-1, 0x1_0000):
        with pytest.raises(TrajectoryMaskError):
            start_for([65, bad], 2, {0: H0, 2: H4})


def test_start_rejects_target_without_recorded_hash():
    """The fail-closed gate: a take-over that cannot be verified is refused."""
    with pytest.raises(TrajectoryHashUnavailable) as error:
        start_for([65] * 8, 8, {0: H0, 4: H4})
    assert "no recorded state hash" in str(error.value)


@pytest.mark.parametrize("bad", ["unknown", "", "0x7f04", "7f04c7361749ed9", "zzzzzzzzzzzzzzzz", 0])
def test_start_rejects_unusable_hash(bad):
    with pytest.raises(TrajectoryHashUnavailable):
        start_for([65] * 4, 4, {0: H0, 4: bad})


def test_start_accepts_uppercase_hash_by_normalising_it():
    start = start_for([65] * 4, 4, {0: H0, 4: H4.upper()})
    assert start.hashes[4] == H4


# ---------------------------------------------------------------------------
# replay_to_frame: the happy path
# ---------------------------------------------------------------------------
def test_replay_resets_with_the_recorded_stage_start_and_verifies_target():
    engine = FakeEngine({0: H0, 4: H4, 8: H8})
    masks = [65, 65, 65, 65, 64, 64, 64, 64]
    report = replay_to_frame(engine, start_for(masks, 8, {0: H0, 4: H4, 8: H8}))

    assert engine.reset_start == StageStart(stage=0, difficulty="normal", team=0, seed=1)
    assert engine.frame == 8
    assert sum(frames for _, frames in engine.calls) == 8
    assert report.matches is True
    assert report.target_hash == H8
    assert report.reset_hash == H0
    assert report.frames_advanced == 8
    assert report.checkpoints_checked == 3
    assert report.checkpoint_hashes == ((0, H0), (4, H4), (8, H8))


def test_replay_feeds_the_recorded_mask_of_every_frame():
    engine = FakeEngine({0: H0, 8: H8})
    masks = [65, 65, 65, 65, 64, 64, 64, 64]
    replay_to_frame(engine, start_for(masks, 8, {0: H0, 8: H8}))
    assert engine.calls == [(65, 4), (64, 4)]
    # Per-frame mode must send exactly one mask per frame, in order.
    engine = FakeEngine({0: H0, 8: H8})
    replay_to_frame(engine, start_for(masks, 8, {0: H0, 8: H8}), batch_frames=1)
    assert engine.calls == [(mask, 1) for mask in masks]


def test_replay_stops_exactly_on_target_and_breaks_at_checkpoints():
    engine = FakeEngine({0: H0, 4: H4, 6: "1" * 16})
    report = replay_to_frame(engine, start_for([65] * 8, 6, {0: H0, 4: H4, 6: "1" * 16}))
    assert engine.frame == 6
    assert sum(frames for _, frames in engine.calls) == 6
    assert engine.calls == [(65, 4), (65, 2)]
    assert report.frames_advanced == 6


def test_replay_with_target_zero_only_verifies_the_reset_hash():
    engine = FakeEngine({0: H0})
    report = replay_to_frame(engine, start_for([65] * 4, 0, {0: H0}))
    assert engine.calls == []
    assert report.frames_advanced == 0
    assert report.checkpoints_checked == 1
    assert report.target_hash == H0


def test_replay_batch_frames_must_be_positive():
    with pytest.raises(ValueError):
        replay_to_frame(FakeEngine({0: H0}), start_for([65], 0, {0: H0}), batch_frames=0)


def test_replay_tolerates_an_environment_without_a_shared_block():
    engine = FakeEngine({0: H0, 4: H4}, no_shm=True)
    assert replay_to_frame(engine, start_for([65] * 4, 4, {0: H0, 4: H4})).matches


# ---------------------------------------------------------------------------
# replay_to_frame: every divergence is fail-closed
# ---------------------------------------------------------------------------
def test_replay_reports_target_mismatch_with_frame_expected_and_observed():
    engine = FakeEngine({0: H0, 4: H4, 8: H8}, wrong={8: "dead" * 4})
    with pytest.raises(TrajectoryHashMismatch) as error:
        replay_to_frame(engine, start_for([65] * 8, 8, {0: H0, 4: H4, 8: H8}))
    record = error.value.as_record()
    assert record["trajectory_hash_mismatch"] is True
    assert record["frame"] == 8
    assert record["expected"] == H8
    assert record["observed"] == "dead" * 4
    message = str(error.value)
    assert "frame 8" in message and H8 in message and "dead" * 4 in message


def test_replay_reports_the_first_differing_checkpoint_not_only_the_target():
    engine = FakeEngine({0: H0, 4: H4, 8: H8}, wrong={4: "beef" * 4})
    with pytest.raises(TrajectoryHashMismatch) as error:
        replay_to_frame(engine, start_for([65] * 8, 8, {0: H0, 4: H4, 8: H8}))
    assert error.value.frame == 4
    assert error.value.where == "fast-forward checkpoint"


def test_replay_reports_a_reset_hash_mismatch_at_frame_zero():
    engine = FakeEngine({0: H0, 4: H4}, reset_hash="0" * 16)
    with pytest.raises(TrajectoryHashMismatch) as error:
        replay_to_frame(engine, start_for([65] * 4, 4, {0: H0, 4: H4}))
    assert error.value.frame == 0
    assert error.value.where == "RESET"
    assert error.value.expected == H0


def test_replay_refuses_a_reset_reply_without_a_usable_hash():
    engine = FakeEngine({0: H0, 4: H4}, reset_hash="unknown")
    with pytest.raises(TrajectoryHashUnavailable):
        replay_to_frame(engine, start_for([65] * 4, 4, {0: H0, 4: H4}))


def test_replay_aborts_when_the_engine_exits_midway():
    engine = FakeEngine({0: H0, 4: H4, 8: H8}, exit_at=6)
    with pytest.raises(TrajectoryAborted) as error:
        replay_to_frame(engine, start_for([65] * 8, 8, {0: H0, 4: H4, 8: H8}))
    assert "exit=True" in str(error.value) or "game_over" in str(error.value)


def test_replay_aborts_on_game_over_midway():
    engine = FakeEngine({0: H0, 4: H4, 8: H8}, game_over_at=6)
    with pytest.raises(TrajectoryAborted):
        replay_to_frame(engine, start_for([65] * 8, 8, {0: H0, 4: H4, 8: H8}))


def test_replay_aborts_when_the_engine_comes_up_short():
    engine = FakeEngine({0: H0, 8: H8}, short_at=8)
    with pytest.raises(TrajectoryAborted) as error:
        replay_to_frame(engine, start_for([65] * 8, 8, {0: H0, 8: H8}))
    assert "advanced" in str(error.value)


# ---------------------------------------------------------------------------
# sample_target_frame
# ---------------------------------------------------------------------------
FRAMES = tuple(range(0, 1000, 4))


def test_sample_target_frame_is_reproducible_for_a_seeded_rng():
    first_rng = random.Random(11)
    first = [sample_target_frame(first_rng, FRAMES) for _ in range(5)]
    second_rng = random.Random(11)
    again = [sample_target_frame(second_rng, FRAMES) for _ in range(5)]
    assert first == again
    assert all(frame in FRAMES for frame in first)


def test_sample_target_frame_with_full_bias_stays_in_the_late_half():
    rng = random.Random(3)
    draws = [sample_target_frame(rng, FRAMES, bias=1.0, boss_weight=0.0) for _ in range(200)]
    assert min(draws) >= 500


def test_sample_target_frame_with_full_boss_weight_stays_in_the_boss_section():
    rng = random.Random(3)
    draws = [sample_target_frame(rng, FRAMES, bias=0.0, boss_weight=1.0, boss_start=600)
             for _ in range(200)]
    assert min(draws) >= 600


def test_sample_target_frame_default_bias_favours_late_and_boss_frames():
    rng = random.Random(2026)
    draws = [sample_target_frame(rng, FRAMES, boss_start=600) for _ in range(2000)]
    late = sum(1 for frame in draws if frame >= 500) / len(draws)
    boss = sum(1 for frame in draws if frame >= 600) / len(draws)
    early = sum(1 for frame in draws if frame < 500) / len(draws)
    # bias=0.65 + boss_weight=0.25: at least 65% late, at least 25% boss, some early.
    assert late >= 0.80
    assert boss >= 0.25
    assert 0.0 < early < 0.20


def test_sample_target_frame_rejects_unusable_weights():
    rng = random.Random(1)
    with pytest.raises(ValueError):
        sample_target_frame(rng, FRAMES, bias=1.2)
    with pytest.raises(ValueError):
        sample_target_frame(rng, FRAMES, bias=0.6, boss_weight=0.6)
    with pytest.raises(ValueError):
        sample_target_frame(rng, FRAMES, boss_weight=-0.1)


def test_sample_target_frame_requires_candidates():
    with pytest.raises(ValueError):
        sample_target_frame(random.Random(1), [])


def test_sample_target_frame_returns_the_only_candidate():
    assert sample_target_frame(random.Random(1), [42]) == 42


# ---------------------------------------------------------------------------
# D0 corpus loader
# ---------------------------------------------------------------------------
def _write_shard(root: Path, *, seed: int = 7, unequal: bool = False,
                 break_chain: bool = False, drop_pair: bool = False,
                 short_masks: bool = False) -> Path:
    shard_dir = root / f"dual-s0d1seed{seed}"
    shard_dir.mkdir(parents=True)
    pairs = []
    for decision, (pre, post) in enumerate([(H0, H4), (H4, H8)]):
        pairs.append({"decision": decision, "boundary": "pre", "frame": decision * 4,
                      "shadow": pre, "env": pre, "equal": not unequal, "source": "A/B"})
        pairs.append({"decision": decision, "boundary": "post", "frame": decision * 4 + 4,
                      "shadow": post, "env": post, "equal": not unequal, "source": "A/B"})
    if drop_pair:
        pairs = pairs[:-1]
    shard = {"schema": "v400-teacher-obs-shard-1", "complete": True, "shard": shard_dir.name,
             "frames_per_decision": 4, "decision_frames": 2, "stage": 0, "difficulty": 1,
             "team": 0, "seed": seed, "reset_command": f"RESET 0 1 0 {seed} 0",
             "hash_checkpoint_pairs": pairs}
    (shard_dir / "shard.json").write_text(json.dumps(shard), encoding="utf-8")
    rows = [
        {"kind": "decision", "decision": 0, "frame": 0,
         "step_frames": [65, 65, 65] if short_masks else [65, 65, 65, 65],
         "hash_pre": H0, "hash_post": H4},
        {"kind": "decision", "decision": 1, "frame": 4, "step_frames": [64, 64, 64, 64],
         "hash_pre": H8 if break_chain else H4, "hash_post": H8},
    ]
    (shard_dir / "labels.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return shard_dir


def test_load_d0_shard_builds_the_per_frame_tape_and_hash_table(tmp_path):
    shard_dir = _write_shard(tmp_path)
    traj = load_d0_shard(shard_dir)
    assert traj.seed == 7 and traj.stage_index == 0 and traj.difficulty == "normal"
    assert traj.masks == (65, 65, 65, 65, 64, 64, 64, 64)
    assert traj.frames == 8
    assert traj.hashes == {0: H0, 4: H4, 8: H8}
    assert traj.checkpoint_frames == (0, 4, 8)
    start = traj.start(8)
    assert isinstance(start, TrajectoryStart)
    assert start.expected_target_hash == H8


def test_load_d0_shards_reads_every_shard(tmp_path):
    _write_shard(tmp_path, seed=5)
    _write_shard(tmp_path, seed=6)
    trajs = load_d0_shards(tmp_path)
    assert [traj.seed for traj in trajs] == [5, 6]


def test_load_d0_shard_refuses_an_unequal_checkpoint(tmp_path):
    with pytest.raises(TrajectoryCorpusError):
        load_d0_shard(_write_shard(tmp_path, unequal=True))


def test_load_d0_shard_refuses_a_broken_hash_chain(tmp_path):
    with pytest.raises(TrajectoryCorpusError):
        load_d0_shard(_write_shard(tmp_path, break_chain=True))


def test_load_d0_shard_refuses_a_label_checkpoint_without_an_ab_pair(tmp_path):
    with pytest.raises(TrajectoryCorpusError):
        load_d0_shard(_write_shard(tmp_path, drop_pair=True))


def test_load_d0_shard_refuses_a_decision_with_the_wrong_number_of_masks(tmp_path):
    """A short ``step_frames`` list is a hole in the tape: the replay would drift."""
    with pytest.raises(TrajectoryCorpusError) as error:
        load_d0_shard(_write_shard(tmp_path, short_masks=True))
    assert "masks" in str(error.value) and "expected 4" in str(error.value)


def test_load_d0_shard_refuses_a_missing_directory(tmp_path):
    with pytest.raises(TrajectoryCorpusError):
        load_d0_shard(tmp_path / "nope")


def test_recorded_trajectory_start_rejects_a_frame_past_the_tape(tmp_path):
    traj = load_d0_shard(_write_shard(tmp_path))
    with pytest.raises(TrajectoryMaskError):
        traj.start(12)


@pytest.mark.parametrize("text,expected", [
    ("RESET 0 1 0 1 0", {"stage_index": 0, "difficulty": "normal", "team": 0, "seed": 1,
                         "resources": "standard"}),
    ("RESET 4 2 3 65535 1", {"stage_index": 4, "difficulty": "hard", "team": 3, "seed": 65535,
                             "resources": "practice"}),
])
def test_parse_reset_command(text, expected):
    assert parse_reset_command(text) == expected


@pytest.mark.parametrize("text", ["", "RESET 0 1 0", "PLAY 0 1 0 1 0", "RESET 0 9 0 1 0",
                                  "RESET 0 1 0 1 2", None])
def test_parse_reset_command_rejects_bad_lines(text):
    with pytest.raises(TrajectoryCorpusError):
        parse_reset_command(text)


# ---------------------------------------------------------------------------
# Th08Env.advance: the additive raw-frame entry point
# ---------------------------------------------------------------------------
def _unstarted_env(tmp_path: Path) -> Th08Env:
    binary = tmp_path / "th08-modern"
    binary.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    return Th08Env(binary, data, shm_name="v400test", role="shadow")


def test_advance_validates_mask_and_frames_before_touching_the_engine(tmp_path):
    env = _unstarted_env(tmp_path)
    with pytest.raises(ValueError):
        env.advance(0x1_0000, 1)
    with pytest.raises(ValueError):
        env.advance(-1, 1)
    with pytest.raises(ValueError):
        env.advance(0x41, 0)
    with pytest.raises(RuntimeError):
        env.advance(0x41, 1)


def test_env_declares_the_role_it_launches(tmp_path):
    env = _unstarted_env(tmp_path)
    assert env.role == "shadow"
    assert env._client.role == "shadow"
    assert "DISPLAY" not in env._client._env


def test_env_rejects_an_unknown_role(tmp_path):
    binary = tmp_path / "th08-modern"
    binary.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    with pytest.raises(ValueError):
        Th08Env(binary, data, shm_name="x", role="replay")


# ---------------------------------------------------------------------------
# Live smoke (opt-in): the real acceptance run is trajectory_verify.py
# ---------------------------------------------------------------------------
LIVE_BINARY = os.environ.get("V400_TRAJ_LIVE_BINARY")
LIVE_DATA = os.environ.get("V400_TRAJ_LIVE_DATA")
LIVE_SHARD = os.environ.get("V400_TRAJ_LIVE_SHARD")


@pytest.mark.skipif(not (LIVE_BINARY and LIVE_DATA and LIVE_SHARD),
                    reason="set V400_TRAJ_LIVE_{BINARY,DATA,SHARD} to run against a real engine")
def test_live_takeover_matches_the_recorded_hash():                     # pragma: no cover
    traj = load_d0_shard(Path(LIVE_SHARD))
    env = Th08Env(Path(LIVE_BINARY), Path(LIVE_DATA), shm_name=f"v400pytest-{os.getpid()}",
                  role="shadow")
    with env:
        report = replay_to_frame(env, traj.start(traj.checkpoint_frames[-1]))
    assert report.matches is True
    assert report.checkpoints_checked == len(traj.hashes)
    assert env.restart_count == 0
