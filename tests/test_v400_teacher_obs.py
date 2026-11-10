"""Contract tests for the teacher **observation** capture (the D0 bridge).

Everything here runs without an engine and without ``/dev/shm``: the engine is a
small fake that advances a frame counter and publishes a synthetic 640x480 RGBA
frame into a fake shared block, so the *real* code path is exercised — the same
``ShmPixelSource``, the same obs-v1 encoder, the same streamed ``.npy`` writer and
the same ``dataset.load_teacher_shard`` the smoke run uses.

The three things this file exists to prove (EXP-P3-001 §3 hard blocker):

1. **Encoding** — what the shard stores is byte-for-byte
   ``obs_spec.encode_planes`` of the engine's raw readback, in oldest-first order,
   HUD from the newest render.
2. **Alignment** — a decision at frame ``d`` stores renders ``d-4..d-1`` (the
   frames the teacher was queried on) and labels the action applied at
   ``d .. d+1``; that is exactly P3's ``frames[i-3..i] -> mask[i+1]`` with
   ``i = d-1``.  The "after the decision" convention is rejected, not recorded.
3. **Fail-closed** — an obs-spec digest change, an unreadable/wrong frame, a
   padded frame in the middle of a shard, or a frame/decision mismatch stops the
   shard instead of writing half of it.

Run (Windows)::

    $env:PYTHONPATH='D:\\WorkShop\\Touhou_RL'
    python -m pytest -p no:cacheprovider --basetemp .tmp_v400_d0_pytest -q tests/test_v400_teacher_obs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from touhou_rl.v400.student import dataset, obs_spec
from touhou_rl.v400.teacher import obs
from touhou_rl.v400.teacher.label import TeacherLabel
from touhou_rl.v400.teacher.search import Plan
from touhou_rl.v400.teacher.shoot import commit_block, commands as step_commands

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "tools" / "rl" / "v400_teacher.py"


# ---------------------------------------------------------------------------
# fakes: one engine, one shared block, no /dev/shm
# ---------------------------------------------------------------------------
class FakeBlock:
    """What ``SharedBlock`` would return for a 640x480 RGBA readback.

    The frame's pixels are a deterministic function of the header frame, so a
    stored row can be checked against the frame it claims to be.
    """

    def __init__(self) -> None:
        self.frame = 0
        self.flags = 0
        self.width = 640
        self.height = 480
        self.header_calls = 0
        self.closed = False

    def publish(self, frame: int) -> None:
        self.frame = int(frame)
        self.flags = 3  # STATE_VALID | PIXELS_VALID

    def header(self) -> dict[str, int]:
        self.header_calls += 1
        return {
            "version": 1,
            "frame": int(self.frame),
            "state_hash": 0,
            "width": int(self.width),
            "height": int(self.height),
            "state_bytes": 100,
            "flags": int(self.flags),
        }

    def pixels(self) -> np.ndarray:
        return pixels_for(int(self.frame))

    def close(self) -> None:
        self.closed = True


def pixels_for(frame: int) -> np.ndarray:
    """A 640x480 RGBA image whose obs-v1 encoding is unique per ``frame``.

    The base level is kept well above zero: the capture path refuses an all-zero
    readback (a shadow null device publishes exactly that), so a fake frame must
    never be mistakable for "nothing was rendered".
    """

    image = np.zeros((480, 640, 4), dtype=np.uint8)
    image[:, :, 0] = 60 + frame % 100
    image[:, :, 1] = 90 + (frame * 3) % 100
    image[:, :, 2] = 7
    image[:, :, 3] = 255
    return image


class FakeEngine:
    """``STEP``/``READBACK``/``SHM``/``FRAME`` on a frame counter + a fake block."""

    def __init__(self, block: FakeBlock, *, start_frame: int = 0) -> None:
        self.block = block
        self.frame_count = int(start_frame)
        self.commands: list[str] = []
        self.readback = 0
        self.child_readbacks: list[int] = []

    def frame(self) -> int:
        return self.frame_count

    def ask(self, command: str) -> str:
        self.commands.append(command)
        parts = command.split()
        if parts[0] == "SHM":
            return "OK shm=fake active=1 bytes=1228992"
        if parts[0] == "READBACK":
            self.readback = int(parts[1])
            return f"OK readback={self.readback} frames=0 micros=0"
        if parts[0] == "FRAME":
            return f"OK frame={self.frame_count}"
        if parts[0] == "STEP":
            mask, count = int(parts[1]), int(parts[2])
            for _ in range(count):
                self.frame_count += 1
                self.block.publish(self.frame_count - 1)
            return f"OK advanced={count} frame={self.frame_count} hash=0"
        raise AssertionError(f"unexpected command {command!r}")


def make_label(*, frame: int, scores=None, bomb: int = 0, verified=None) -> TeacherLabel:
    values = np.full(obs_spec.N_MOVE_CLASSES, 100.0) if scores is None else np.asarray(scores)
    mask = (np.ones(obs_spec.N_MOVE_CLASSES, dtype=bool) if verified is None
            else np.asarray(verified, dtype=bool))
    plan = Plan(first_action=3, duration=1, slots=[3, 9], margin=12.5, death_frame=None,
                end_xy=(190.0, 380.0), score=113.5, verified=True)
    return TeacherLabel(
        scores=values,
        bomb=int(bomb),
        chosen_plan=plan,
        margin=12.5,
        verified_mask=mask,
        frame=int(frame),
        stage=1,
        difficulty=1,
    )


def make_source(engine: FakeEngine, block: FakeBlock, *, name: str = "fake") -> obs.ShmPixelSource:
    return obs.ShmPixelSource(engine.ask, name=name, block_factory=lambda _name: block)


def decisions_block(*, cand: int = 3, bomb: int = 0) -> list[tuple[int, int]]:
    return commit_block([cand, cand], bomb=bomb, decision_frames=obs_spec.DECISION_FRAMES)


# ---------------------------------------------------------------------------
# 1. frozen spec + the alignment arithmetic
# ---------------------------------------------------------------------------
def test_frozen_digest_is_the_documented_one():
    assert obs.FROZEN_OBS_SPEC_DIGEST == (
        "24f0ddace59104938dbe991c0d8efbb6a8cd21dfd5a29ebcb6d023b5b364a381")
    assert obs.check_frozen_spec() == obs.FROZEN_OBS_SPEC_DIGEST
    assert obs_spec.obs_spec_digest() == obs.FROZEN_OBS_SPEC_DIGEST


def test_frozen_gate_rejects_a_drifted_digest_or_version(monkeypatch):
    with pytest.raises(obs.ObservationError, match="version"):
        obs.check_frozen_spec("obs-v0", obs.FROZEN_OBS_SPEC_DIGEST)
    with pytest.raises(obs.ObservationError, match="frozen obs-spec"):
        obs.check_frozen_spec(obs_spec.OBS_SPEC_VERSION, "0" * 64)
    # A live module whose spec changed must be refused even if it is self-consistent.
    monkeypatch.setattr(obs_spec, "obs_spec_digest", lambda: "1" * 64)
    with pytest.raises(obs.ObservationError, match="drifted"):
        obs.check_frozen_spec()


def test_decision_window_is_the_p3_window():
    for decision_frame in (0, 4, 8, 400):
        window = obs.decision_window(decision_frame)
        assert window["render_frames"] == [
            decision_frame - 4, decision_frame - 3, decision_frame - 2, decision_frame - 1]
        assert window["newest_render_frame"] == decision_frame - 1
        # P3: frames[i-3..i] -> mask[i+1]
        assert window["p3_window_last_row"] == decision_frame - 1
        assert window["p3_label_frame"] == window["p3_window_last_row"] + 1
        assert window["p3_label_frame"] == decision_frame
        assert window["label_frames"] == [decision_frame, decision_frame + 1]


def test_after_the_decision_convention_is_a_different_window():
    """The rejected convention: newest render = d+3 labels a stale action."""

    decision_frame = 8
    obs_now = obs.decision_window(decision_frame)
    after_commit = decision_frame + 4  # the commit advanced 4 frames
    after = obs.decision_window(after_commit)
    assert after["newest_render_frame"] == decision_frame + 3
    assert after["newest_render_frame"] != obs_now["newest_render_frame"]
    # the label would no longer be "the action applied at i+1"
    assert after["p3_label_frame"] != decision_frame
    assert after["p3_label_frame"] - decision_frame == 4


# ---------------------------------------------------------------------------
# 2. encoding: shm -> encode_planes, byte for byte
# ---------------------------------------------------------------------------
def test_shm_source_encodes_exactly_like_obs_spec():
    block = FakeBlock()
    block.publish(17)
    engine = FakeEngine(block)
    source = make_source(engine, block)
    source.attach()
    source.readback(True)
    play, hud, header = source.read(17)
    reference_play, reference_hud = obs_spec.encode_planes(pixels_for(17))
    assert header["frame"] == 17
    assert np.array_equal(play, reference_play)
    assert np.array_equal(hud, reference_hud)
    # and the encoder really is content-dependent (the check is not vacuous)
    block.publish(18)
    play18, hud18, _ = source.read(18)
    assert not np.array_equal(play, play18)
    assert not np.array_equal(hud, hud18)


def test_shm_source_fails_closed_on_frame_or_pixels():
    block = FakeBlock()
    engine = FakeEngine(block)
    source = make_source(engine, block)
    source.attach()
    with pytest.raises(obs.ObservationError, match="no published pixels"):
        source.read(5)  # nothing was ever drawn with readback on
    block.publish(5)
    with pytest.raises(obs.ObservationError, match="shared-memory frame"):
        source.read(6)
    block.width = 640
    block.height = 240
    with pytest.raises(obs.ObservationError, match="640x480"):
        source.read(5)
    block.height = 480
    block.flags = 0
    with pytest.raises(obs.ObservationError, match="no published pixels"):
        source.read(5)
    block.flags = 3
    block.publish(5)
    assert source.read(5)[2]["frame"] == 5


def test_attach_requires_the_engine_to_confirm():
    engine = FakeEngine(FakeBlock())

    def refuse(command: str) -> str:
        return "ERR shm-attach-failed"

    source = obs.ShmPixelSource(refuse, name="fake", block_factory=lambda name: FakeBlock())
    with pytest.raises(obs.ObservationError, match="SHM attach refused"):
        source.attach()
    assert engine.frame() == 0


# ---------------------------------------------------------------------------
# 3. the rolling window
# ---------------------------------------------------------------------------
def buffer_with(frames) -> obs.DecisionObservationBuffer:
    buffer = obs.DecisionObservationBuffer()
    for frame in frames:
        play = np.full(obs_spec.PLAY_SHAPE, frame % 251, dtype=np.uint8)
        hud = np.full(obs_spec.HUD_SHAPE, frame % 251, dtype=np.uint8)
        buffer.push(obs.RenderFrame(frame=int(frame), play=play, hud=hud))
    return buffer


def test_buffer_keeps_the_last_four_and_orders_them_oldest_first():
    buffer = buffer_with(range(10))
    sample = buffer.stack_for(10)  # d = 10 -> renders 6,7,8,9
    assert sample["render_frames"] == [6, 7, 8, 9]
    assert sample["pad"] == 0
    assert [int(sample["play"][index].max()) for index in range(4)] == [6, 7, 8, 9]
    assert int(sample["hud"].max()) == 9  # HUD is the newest render only


def test_buffer_refuses_a_gap_or_a_short_window():
    buffer = buffer_with([0, 1, 2])  # 3 frames, no seed
    with pytest.raises(obs.ObservationError, match="need 4 consecutive"):
        buffer.stack_for(4)
    buffer = buffer_with([0, 1, 2, 4])  # missing frame 3
    with pytest.raises(obs.ObservationError, match=r"!= required window"):
        buffer.stack_for(5)
    buffer = buffer_with([1, 2, 3, 4])  # one decision late: d=6 needs 2..5
    with pytest.raises(obs.ObservationError, match=r"!= required window"):
        buffer.stack_for(6)


def test_buffer_rejects_a_duplicate_frame():
    buffer = buffer_with([0, 1, 2, 3])
    frame = obs.RenderFrame(3, np.zeros(obs_spec.PLAY_SHAPE, np.uint8),
                            np.zeros(obs_spec.HUD_SHAPE, np.uint8))
    with pytest.raises(obs.ObservationError, match="not after"):
        buffer.push(frame)


def test_episode_start_pads_with_the_reset_render():
    seed = obs.RenderFrame(
        41, np.full(obs_spec.PLAY_SHAPE, 9, np.uint8), np.full(obs_spec.HUD_SHAPE, 9, np.uint8))
    buffer = obs.DecisionObservationBuffer()
    buffer.reset(seed)
    sample = buffer.stack_for(0, episode_start=True)
    assert sample["pad"] == 3
    assert sample["render_frames"] == [41, 41, 41, 41]
    assert sample["play"].shape == (4, 64, 64)
    assert all(int(sample["play"][index].max()) == 9 for index in range(4))
    assert sample["hud"].shape == (56, 160)
    with pytest.raises(obs.ObservationError, match="only valid at decision 0"):
        buffer.stack_for(4, episode_start=True)
    empty = obs.DecisionObservationBuffer()
    with pytest.raises(obs.ObservationError, match="no render captured at RESET"):
        empty.stack_for(0, episode_start=True)


# ---------------------------------------------------------------------------
# 4. the streaming npy writer
# ---------------------------------------------------------------------------
def test_npy_append_writer_roundtrip(tmp_path):
    writer = obs.NpyAppendWriter(tmp_path / "a.npy", (2, 3), np.uint8)
    rows = [np.full((2, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
    for row in rows:
        writer.append(row)
    meta = writer.finalize()
    loaded = np.load(tmp_path / "a.npy")
    assert loaded.shape == (3, 2, 3)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, np.stack(rows))
    assert meta["shape"] == [3, 2, 3]
    assert meta["bytes"] == (tmp_path / "a.npy").stat().st_size
    mapped = np.load(tmp_path / "a.npy", mmap_mode="r")
    assert np.array_equal(mapped[2], rows[2])


def test_npy_append_writer_abort_declares_zero_rows(tmp_path):
    writer = obs.NpyAppendWriter(tmp_path / "b.npy", (4, 64, 64), np.uint8)
    writer.append(np.zeros((4, 64, 64), dtype=np.uint8))
    writer.abort()
    loaded = np.load(tmp_path / "b.npy")
    assert loaded.shape == (0, 4, 64, 64)


def test_npy_append_writer_rejects_a_bad_row(tmp_path):
    writer = obs.NpyAppendWriter(tmp_path / "c.npy", (4, 64, 64), np.uint8)
    with pytest.raises(obs.ObservationError, match="row shape"):
        writer.append(np.zeros((4, 64, 63), dtype=np.uint8))
    with pytest.raises(obs.ObservationError, match="dtype"):
        writer.append(np.zeros((4, 64, 64), dtype=np.float32))
    writer.abort()


# ---------------------------------------------------------------------------
# 5. the commit: one STEP per frame, state-identical split
# ---------------------------------------------------------------------------
def test_commit_splits_every_step_into_single_frames():
    block = decisions_block(cand=5, bomb=1)
    assert sum(count for _, count in block) == 4
    fake_block = FakeBlock()
    fake_block.publish(-1)
    engine = FakeEngine(fake_block)
    source = make_source(engine, fake_block)
    source.attach()
    run = obs.TeacherObservationRun(
        out_root=Path("unused"), key="k", meta={}, source=source,
        writer=_SpyWriter())
    result = run.commit(engine, block)
    assert result["complete"] and result["advanced"] == 4 and not result["exited"]
    step_commands_sent = [command for command in engine.commands if command.startswith("STEP")]
    # the split: one STEP per frame, same mask order as the batched form
    expected = [f"STEP {int(mask)} 1" for mask, count in block for _ in range(int(count))]
    assert step_commands_sent == expected
    assert step_commands(block) == [f"STEP {int(mask)} {int(count)}" for mask, count in block]
    # readback is on only for the commit
    assert engine.readback == 0
    assert [int(frame.frame) for frame in run.buffer.frames] == [0, 1, 2, 3]


class _SpyWriter:
    """Minimal stand-in so ``commit`` can run without touching the filesystem."""

    rows = 0
    dropped: list = []

    def __init__(self) -> None:
        self.render_stats: list = []


# ---------------------------------------------------------------------------
# 6. the full run -> shard -> dataset round trip
# ---------------------------------------------------------------------------
def run_decisions(
    tmp_path: Path,
    *,
    decisions: int = 5,
    seed_frame: int = 41,
    key: str = "teacher-fast-s1d1t0-seed1",
) -> tuple[Path, dict, FakeEngine, FakeBlock]:
    block = FakeBlock()
    block.publish(seed_frame)
    engine = FakeEngine(block)
    source = make_source(engine, block)
    run = obs.TeacherObservationRun(
        out_root=tmp_path,
        key=key,
        meta={"stage": 1, "difficulty": 1, "team": 0, "seed": 1, "mode": "fast"},
        source=source,
        frames_per_decision=4,
    )
    run.attach()
    run.seed_after_reset(state_frame=int(engine.frame()))
    for decision in range(decisions):
        decision_frame = decision * 4
        sample = run.begin_decision(engine, decision_frame)
        assert sample["pad"] == (3 if decision == 0 else 0)
        run.verify_frame_unchanged(engine)
        commit = run.commit(engine, decisions_block(cand=decision % 18))
        assert commit["complete"], commit
        run.add_sample(index=decision, label=make_label(frame=decision_frame))
    summary = run.finalize(report={"stop_reason": "decisions"})
    return tmp_path, summary, engine, block


def test_run_writes_a_shard_dataset_can_load(tmp_path):
    root, summary, engine, block = run_decisions(tmp_path, decisions=5)
    shard_dir = root / "shards" / "teacher-fast-s1d1t0-seed1"
    assert (shard_dir / "shard.json").is_file()
    assert (root / "manifest.json").is_file()
    assert not (root / "shards" / ".teacher-fast-s1d1t0-seed1.partial").exists()
    assert summary["decisions"] == 5
    per_decision = 4 * 64 * 64 + 56 * 160  # 25,344 B of obs planes per decision
    # two fixed 512 B npy headers (522 B each) are the only overhead per shard
    assert summary["obs_bytes"] == 5 * per_decision + 2 * 522
    assert summary["obs_bytes_per_decision"] == summary["obs_bytes"] // 5
    assert summary["obs_spec_digest"] == obs.FROZEN_OBS_SPEC_DIGEST
    assert summary["frames"] == [0, 20]

    shard = dataset.load_teacher_shard(shard_dir)
    assert len(shard) == 5
    assert shard.play.shape == (5, 4, 64, 64)
    assert shard.hud.shape == (5, 56, 160)
    first = shard[0]
    assert first["scores"].shape == (18,)
    assert first["play"].shape == (4, 64, 64)
    assert first["hud"].shape == (56, 160)
    assert first["decision"] == 0 and first["frame"] == 0
    # decision 0 is the episode-start padding: the RESET render four times
    seed_play = obs_spec.encode_planes(pixels_for(41))[0]
    assert np.array_equal(first["play"], np.repeat(seed_play[None, ...], 4, axis=0))
    assert np.array_equal(first["hud"], obs_spec.encode_planes(pixels_for(41))[1])
    # decision 1 stores the renders 0..3, oldest first, byte-exact
    expected = np.stack([obs_spec.encode_planes(pixels_for(frame))[0] for frame in range(4)])
    assert np.array_equal(shard[1]["play"], expected)
    assert np.array_equal(shard[1]["hud"], obs_spec.encode_planes(pixels_for(3))[1])
    normalized = shard.normalized(1)
    assert normalized["play"].dtype == np.float32
    assert normalized["play"].shape == (4, 64, 64)
    assert normalized["hud"].shape == (1, 56, 160)
    assert 0.0 <= float(normalized["play"].max()) <= 1.0
    report = shard.report()
    assert report["decisions"] == 5
    assert report["alignment"] == obs.OBSERVATION_TIME
    assert report["play_shape"] == [5, 4, 64, 64]


def test_run_shard_loads_through_the_teacher_dataset_api(tmp_path):
    root, summary, _engine, _block = run_decisions(tmp_path, decisions=5)
    labels = root / "shards" / "teacher-fast-s1d1t0-seed1" / "labels.jsonl"
    payload = dataset.load_teacher_samples(
        [labels], min_samples=5, obs_root=root, require_observations=True)
    assert payload["count"] == 5
    record = payload["samples"][0]
    assert record["scores"].shape == (18,)
    arrays = record["obs_arrays"]
    assert arrays["play"].shape == (4, 64, 64)
    assert arrays["hud"].shape == (56, 160)
    assert payload["obs_shapes"] == {"play": [64, 64], "hud": [56, 160], "stack": 4}
    assert payload["alignment"] == dataset.TEACHER_OBSERVATION_TIME
    assert payload["obs_spec_digest"] == obs.FROZEN_OBS_SPEC_DIGEST
    assert payload["obs_unresolved"] == []
    assert payload["obs_shards"], "the shard report must be part of the payload"
    manifest = dataset.load_teacher_manifest(root)
    assert manifest["totals"]["decisions"] == 5
    assert manifest["totals"]["shards"] == 1
    assert manifest["alignment"]["observation_time"] == obs.OBSERVATION_TIME


def test_decision_frames_and_labels_are_consistent(tmp_path):
    root, _summary, _engine, _block = run_decisions(tmp_path, decisions=3)
    shard_dir = root / "shards" / "teacher-fast-s1d1t0-seed1"
    rows = [json.loads(line) for line in
            (shard_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["decision"] for row in rows] == [0, 1, 2]
    assert [row["decision_frame"] for row in rows] == [0, 4, 8]
    assert [row["obs"]["render_frames"] for row in rows][1] == [0, 1, 2, 3]
    assert [row["obs"]["p3_label_frame"] for row in rows] == [0, 4, 8]
    assert rows[0]["obs"]["pad"] == 3
    assert rows[0]["obs"]["observation_time"] == dataset.TEACHER_OBSERVATION_TIME
    assert rows[0]["stage"] == 1 and rows[0]["difficulty"] == 1
    assert len(rows[0]["scores"]) == 18 and len(rows[0]["verified_mask"]) == 18


# ---------------------------------------------------------------------------
# 7. fail-closed behaviour
# ---------------------------------------------------------------------------
def test_begin_decision_rejects_a_frame_decision_mismatch(tmp_path):
    block = FakeBlock()
    block.publish(41)
    engine = FakeEngine(block)
    run = obs.TeacherObservationRun(
        out_root=tmp_path, key="k", meta={}, source=make_source(engine, block),
        frames_per_decision=4)
    run.attach()
    run.seed_after_reset(state_frame=0)
    with pytest.raises(obs.ObservationError, match="engine frame 0 != decision frame 4"):
        run.begin_decision(engine, 4)


def test_verify_frame_rejects_a_shadow_that_moved(tmp_path):
    block = FakeBlock()
    block.publish(41)
    engine = FakeEngine(block)
    run = obs.TeacherObservationRun(
        out_root=tmp_path, key="k", meta={}, source=make_source(engine, block),
        frames_per_decision=4)
    run.attach()
    run.seed_after_reset(state_frame=0)
    run.begin_decision(engine, 0)
    engine.frame_count += 2  # a recording path that moved the shadow
    with pytest.raises(obs.ObservationError, match="decision point moved"):
        run.verify_frame_unchanged(engine)


def test_commit_reports_a_short_advance_instead_of_writing_a_row(tmp_path):
    class StoppingEngine(FakeEngine):
        def ask(self, command: str) -> str:
            if command.startswith("STEP"):
                self.commands.append(command)
                self.frame_count += 1
                self.block.publish(self.frame_count - 1)
                return f"OK advanced=1 frame={self.frame_count} exit=1 game_over=1"
            return super().ask(command)

    block = FakeBlock()
    block.publish(41)
    engine = StoppingEngine(block)
    run = obs.TeacherObservationRun(
        out_root=tmp_path, key="k", meta={}, source=make_source(engine, block),
        frames_per_decision=4)
    run.attach()
    run.seed_after_reset(state_frame=0)
    run.begin_decision(engine, 0)
    result = run.commit(engine, decisions_block())
    assert result["exited"] is True
    assert result["complete"] is False
    run.drop_pending("commit_incomplete")
    with pytest.raises(obs.ObservationError, match="at least one decision"):
        run.finalize()
    detail = run.abort(obs.ObservationError("empty shard"))
    assert detail["complete"] is False
    partial = tmp_path / "shards" / ".k.partial"
    assert partial.is_dir() and (partial / "incomplete.json").is_file()
    assert not (tmp_path / "shards" / "k").exists()
    with pytest.raises(dataset.DatasetError, match="not found"):
        dataset.load_teacher_shard(partial)


def test_capture_refuses_a_blank_readback(tmp_path):
    """A shadow-role (null-device) run publishes an all-zero block: stop, don't fake.

    This is the measured capability of the teacher's role: ``main.cpp`` gives a
    shadow the "software D3D null device" and the Linux D3D8 layer never touches
    GL for it, so ``AfterDrawFrame``'s ``glReadPixels`` has no frame to read.  The
    seed gate must refuse before a single label row is written.
    """

    class BlankBlock(FakeBlock):
        def pixels(self) -> np.ndarray:
            return np.zeros((480, 640, 4), dtype=np.uint8)

    block = BlankBlock()
    block.publish(41)
    engine = FakeEngine(block)
    run = obs.TeacherObservationRun(
        out_root=tmp_path, key="k", meta={}, source=make_source(engine, block),
        frames_per_decision=4)
    run.attach()
    with pytest.raises(obs.ObservationError, match="entirely zero") as info:
        run.seed_after_reset(state_frame=0)
    assert "null device" in str(info.value)
    run.abort(info.value)
    partial = tmp_path / "shards" / ".k.partial"
    assert (partial / "incomplete.json").is_file()
    assert not (tmp_path / "shards" / "k").exists()
    with pytest.raises(dataset.DatasetError, match="not found"):
        dataset.load_teacher_shard(partial)


def test_commit_refuses_a_blank_frame_mid_run(tmp_path):
    class FadingBlock(FakeBlock):
        blank = False

        def pixels(self) -> np.ndarray:
            return (np.zeros((480, 640, 4), dtype=np.uint8) if self.blank
                    else pixels_for(int(self.frame)))

    block = FadingBlock()
    block.publish(41)
    engine = FakeEngine(block)
    run = obs.TeacherObservationRun(
        out_root=tmp_path, key="k", meta={}, source=make_source(engine, block),
        frames_per_decision=4)
    run.attach()
    run.seed_after_reset(state_frame=0)
    run.begin_decision(engine, 0)
    block.blank = True
    with pytest.raises(obs.ObservationError, match="entirely zero"):
        run.commit(engine, decisions_block())
    assert run.writer.rows == 0
    run.abort(obs.ObservationError("blank frame"))


def test_writer_refuses_to_overwrite_or_mix(tmp_path):
    root, _summary, _engine, _block = run_decisions(tmp_path, decisions=2)
    with pytest.raises(obs.ObservationError, match="already exists"):
        obs.TeacherObsShardWriter(root, key="teacher-fast-s1d1t0-seed1", meta={})
    stale = tmp_path / "shards" / ".other.partial"
    stale.mkdir(parents=True)
    with pytest.raises(obs.ObservationError, match="staging directory"):
        obs.TeacherObsShardWriter(root, key="other", meta={})


def test_dataset_refuses_a_wrong_alignment(tmp_path):
    root, _summary, _engine, _block = run_decisions(tmp_path, decisions=2)
    shard_dir = root / "shards" / "teacher-fast-s1d1t0-seed1"
    payload = json.loads((shard_dir / "shard.json").read_text(encoding="utf-8"))
    payload["alignment"]["observation_time"] = "after_commit"
    (shard_dir / "shard.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="observation_time"):
        dataset.load_teacher_shard(shard_dir, verify_sha=False)


def test_dataset_refuses_a_frame_decision_mismatch(tmp_path):
    root, _summary, _engine, _block = run_decisions(tmp_path, decisions=3)
    shard_dir = root / "shards" / "teacher-fast-s1d1t0-seed1"
    lines = (shard_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["frame"] = row["decision_frame"] = 5  # not 4
    row["obs"]["render_frames"] = [1, 2, 3, 4]
    row["obs"]["newest_render_frame"] = 4
    lines[1] = json.dumps(row)
    (shard_dir / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = json.loads((shard_dir / "shard.json").read_text(encoding="utf-8"))
    payload["labels"]["sha256"] = dataset._sha256_file(shard_dir / "labels.jsonl")
    (shard_dir / "shard.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="frames per decision"):
        dataset.load_teacher_shard(shard_dir)


def test_dataset_refuses_a_stale_observation_window(tmp_path):
    root, _summary, _engine, _block = run_decisions(tmp_path, decisions=3)
    shard_dir = root / "shards" / "teacher-fast-s1d1t0-seed1"
    lines = (shard_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    # the "after the decision" convention: newest render d+3 instead of d-1
    row["obs"]["render_frames"] = [4, 5, 6, 7]
    row["obs"]["newest_render_frame"] = 7
    lines[1] = json.dumps(row)
    (shard_dir / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = json.loads((shard_dir / "shard.json").read_text(encoding="utf-8"))
    payload["labels"]["sha256"] = dataset._sha256_file(shard_dir / "labels.jsonl")
    (shard_dir / "shard.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match=r"frames\[d-4\.\.d-1\]"):
        dataset.load_teacher_shard(shard_dir)


def test_dataset_refuses_a_digest_mismatch(tmp_path):
    root, _summary, _engine, _block = run_decisions(tmp_path, decisions=2)
    shard_dir = root / "shards" / "teacher-fast-s1d1t0-seed1"
    payload = json.loads((shard_dir / "shard.json").read_text(encoding="utf-8"))
    payload["obs_spec_digest"] = "0" * 64
    (shard_dir / "shard.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="obs spec check failed"):
        dataset.load_teacher_shard(shard_dir)


def test_dataset_refuses_a_tampered_plane(tmp_path):
    root, _summary, _engine, _block = run_decisions(tmp_path, decisions=2)
    shard_dir = root / "shards" / "teacher-fast-s1d1t0-seed1"
    plane = shard_dir / "obs" / "play64.npy"
    data = bytearray(plane.read_bytes())
    data[-1] ^= 0xFF
    plane.write_bytes(bytes(data))
    with pytest.raises(dataset.DatasetError, match="sha256"):
        dataset.load_teacher_shard(shard_dir)


def test_dataset_refuses_an_incomplete_shard(tmp_path):
    root, _summary, _engine, _block = run_decisions(tmp_path, decisions=2)
    shard_dir = root / "shards" / "teacher-fast-s1d1t0-seed1"
    payload = json.loads((shard_dir / "shard.json").read_text(encoding="utf-8"))
    payload["complete"] = False
    (shard_dir / "shard.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="not complete"):
        dataset.load_teacher_shard(shard_dir, verify_sha=False)


def test_teacher_samples_still_fail_closed_without_observations(tmp_path):
    """The EXP-P3-001 blocker stays fail-closed when no obs shard is given."""

    labels = tmp_path / "teacher.jsonl"
    labels.write_text(json.dumps({
        "kind": "decision", "frame": 0, "scores": [100.0] * 18,
        "verified_mask": [True] * 18, "bomb": 0}) + "\n", encoding="utf-8")
    with pytest.raises(dataset.DatasetInsufficient, match="no observation reference"):
        dataset.load_teacher_samples([labels], min_samples=1)
    # and an unresolvable reference is an error when observations are required
    reference = json.loads(labels.read_text(encoding="utf-8"))
    reference["obs"] = {"shard": "teacher-fast-s1d1t0-seed1", "row": 0}
    labels.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    with pytest.raises(dataset.DatasetInsufficient, match="no shard.json"):
        dataset.load_teacher_samples([labels], min_samples=1, obs_root=tmp_path,
                                     require_observations=True)


# ---------------------------------------------------------------------------
# 8. the CLI wiring
# ---------------------------------------------------------------------------
def test_cli_exposes_save_observations_and_refuses_unsupported_modes(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("v400_teacher_cli_test", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = str(tmp_path / "d0")
    args = module.parse_args(
        ["--decisions", "2", "--save-observations", out, "--obs-shard-key", "k",
         "--mode", "fast"])
    assert args.save_observations == tmp_path / "d0"
    assert args.obs_shard_key == "k"
    assert module.parse_args([]).save_observations is None
    # the two recording modes that cannot honour the window are refused before the
    # engine is even hashed (exit 3, not a silent recording)
    assert module.main(["--save-observations", out, "--follow-stage"]) == module.EXIT_FAIL_CLOSED
    assert module.main(["--save-observations", out,
                        "--allow-legacy-restore"]) == module.EXIT_FAIL_CLOSED
