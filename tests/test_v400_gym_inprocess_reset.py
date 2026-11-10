"""T7.1/T7.3 contracts: repeated in-process ``RESET`` and a fresh state block.

Nothing here starts an engine.  The pure half pins the acceptance driver's
identities (mask stream, comparison, verdict) against values another experiment
already published, and the gym half drives ``Th08Env`` with a protocol double to
pin the T7.3 guarantee: after ``RESET``/``STEP`` the shared block has to be the
state the reply describes, or the environment fails closed.

The real-engine measurements are EXP-P0-078 (``build/rl_v3``, 10 resets x 5,000
frames against three fresh reference processes).
"""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl.v400.gym import env, inprocess_reset as probe, shm

#: EXP-P0-049 published this digest for the documented SplitMix64 stream
#: (seed 400, 5,000 frames, whitelist 0x00F7).  Re-deriving it here is what makes
#: "the same input sequence as G0.2 item 1" a checked claim instead of a comment.
G02_MASK_SHA256 = "4173ff413d4bc18e27b771e86de16a441f765ac92b273c68ab7e32f5ee585e1f"

#: EXP-P0-049's frame-0 hash for ``RESET 0 1 0 1234 1``; EXP-P0-078 reproduced it
#: on ``build/rl_v3`` for both the cold and the warm path.
G02_FRAME0_HASH_SEED_1234 = "d97b43478cc24bae"


# ---------------------------------------------------------------------------
# Mask stream and comparison (pure)
# ---------------------------------------------------------------------------
def test_mask_stream_matches_the_published_g0_2_item_1_identity():
    masks = probe.mask_sequence(seed=400, frames=5000)

    assert len(masks) == 5000
    assert probe.mask_sequence_sha256(masks) == G02_MASK_SHA256


def test_mask_stream_head_and_tail_are_the_documented_ones():
    masks = probe.mask_sequence(400, 5000)

    assert [f"{mask:04x}" for mask in masks[:16]] == [
        "00b1", "00f6", "0040", "0073", "00a6", "0020", "00b4", "0094",
        "00b4", "00c7", "0050", "0046", "00b3", "0033", "00f3", "00e3",
    ]
    assert [f"{mask:04x}" for mask in masks[-8:]] == [
        "0081", "00b1", "00e6", "00f5", "00f1", "0003", "0036", "0001",
    ]


def test_every_mask_stays_inside_the_button_whitelist():
    masks = probe.mask_sequence(400, 5000)

    assert probe.MASK_WHITELIST == 0x00F7
    assert all(mask & ~probe.MASK_WHITELIST == 0 for mask in masks)
    assert len(set(masks)) == 128


def test_reset_command_matches_the_g0_2_item_1_stage_start():
    assert probe.reset_command(1234) == "RESET 0 1 0 1234 1"


def test_identical_sequences_compare_as_identical():
    sequence = ["d97b43478cc24bae", "46e2f54401a0fdb7"]

    result = probe.compare_hashes(sequence, list(sequence))

    assert result["identical"] is True
    assert result["frames_compared"] == 2
    assert result["different_frames"] == 0
    assert result["first_differing_frame"] is None


def test_a_single_differing_frame_is_located_and_counted():
    warm = ["d97b43478cc24bae", "46e2f54401a0fdb7", "4ede365c6cb537e1"]
    reference = ["d97b43478cc24bae", "0000000000000000", "4ede365c6cb537e1"]

    result = probe.compare_hashes(warm, reference)

    assert result["identical"] is False
    assert result["different_frames"] == 1
    assert result["first_differing_frame"] == 2
    assert result["differences"] == [
        {"frame": 2, "warm": "46e2f54401a0fdb7", "reference": "0000000000000000"}
    ]


def test_a_truncated_sequence_is_never_identical():
    result = probe.compare_hashes(["a"], ["a", "b"])

    assert result["identical"] is False
    assert result["lengths"] == [1, 2]


def test_hashes_sha256_is_the_newline_join():
    import hashlib

    hashes = ["d97b43478cc24bae", "46e2f54401a0fdb7"]

    assert probe.hashes_sha256(hashes) == hashlib.sha256(
        "d97b43478cc24bae\n46e2f54401a0fdb7".encode("ascii")).hexdigest()


def test_reset_wall_statistics_use_the_median_and_the_limit():
    within = probe.reset_wall_statistics([0.5, 0.6, 0.55])
    outside = probe.reset_wall_statistics([1.0, 1.1, 2.9])

    assert within["median_s"] == pytest.approx(0.55)
    assert within["within_limit"] is True
    assert outside["median_s"] == pytest.approx(1.1)
    assert outside["within_limit"] is True  # median, not max, is the criterion
    assert probe.reset_wall_statistics([2.0, 2.2, 9.0])["within_limit"] is False


def test_empty_wall_statistics_fail_closed():
    stats = probe.reset_wall_statistics([])

    assert stats["samples"] == 0
    assert stats["within_limit"] is False


# ---------------------------------------------------------------------------
# Verdict (pure)
# ---------------------------------------------------------------------------
def _reset(index: int, seed: int, *, ready: str = "1", identical: bool = True,
           frames: int = 5000, error: str | None = None, mode: str = "in-process"):
    compare = {"frames_compared": frames, "different_frames": 0 if identical else 1,
               "first_differing_frame": None if identical else 1, "identical": identical}
    return {"index": index, "seed": seed, "ready": ready, "mode": mode, "error": error,
            "frames_stepped": frames, "stop": None, "compare": compare,
            "debug_after_reset": {"subthread": "0", "threads": "1"},
            "reset_wall_s": 0.6}


def _reference(seed: int):
    return {"label": f"ref-{seed}", "seed": seed, "exit_code": 0, "frames_stepped": 5000,
            "error": None}


def _wall(within: bool = True):
    return {"samples": 10, "median_s": 0.6 if within else 2.0,
            "limit_s": probe.RESET_WALL_LIMIT_S, "within_limit": within}


def test_verdict_passes_on_ten_clean_resets():
    resets = [_reset(index + 1, (1234, 1235, 1236)[index % 3]) for index in range(10)]
    verdict = probe.evaluate(resets=resets, references=[_reference(seed) for seed in
                                                        (1234, 1235, 1236)],
                             required_resets=10, required_frames=5000, wall=_wall(),
                             warm_exit_code=0)

    assert verdict["pass"] is True
    assert verdict["reasons"] == []
    assert verdict["frames_compared"] == 50_000
    assert verdict["different_frames"] == 0


def test_verdict_rejects_a_not_ready_reset():
    resets = [_reset(1, 1234)] + [_reset(index, 1234, ready="0") for index in range(2, 11)]

    verdict = probe.evaluate(resets=resets, references=[_reference(1234)],
                             required_resets=10, required_frames=5000, wall=_wall(),
                             warm_exit_code=0)

    assert verdict["pass"] is False
    assert any("ready=" in reason for reason in verdict["reasons"])


def test_verdict_rejects_one_differing_frame_in_fifty_thousand():
    resets = [_reset(index + 1, 1234) for index in range(10)]
    resets[6]["compare"] = {"frames_compared": 5000, "different_frames": 1,
                            "first_differing_frame": 4999, "identical": False}

    verdict = probe.evaluate(resets=resets, references=[_reference(1234)],
                             required_resets=10, required_frames=5000, wall=_wall(),
                             warm_exit_code=0)

    assert verdict["pass"] is False
    assert any("differ from the fresh process" in reason for reason in verdict["reasons"])


def test_verdict_rejects_a_slow_reset_median():
    resets = [_reset(index + 1, 1234) for index in range(10)]

    verdict = probe.evaluate(resets=resets, references=[_reference(1234)],
                             required_resets=10, required_frames=5000,
                             wall=_wall(within=False), warm_exit_code=0)

    assert verdict["pass"] is False
    assert any("median" in reason for reason in verdict["reasons"])


def test_verdict_rejects_a_short_reset_count():
    resets = [_reset(index + 1, 1234) for index in range(3)]

    verdict = probe.evaluate(resets=resets, references=[_reference(1234)],
                             required_resets=10, required_frames=5000, wall=_wall(),
                             warm_exit_code=0)

    assert verdict["pass"] is False
    assert any("3 resets recorded" in reason for reason in verdict["reasons"])


def test_verdict_rejects_a_nonzero_warm_exit():
    resets = [_reset(index + 1, 1234) for index in range(10)]

    verdict = probe.evaluate(resets=resets, references=[_reference(1234)],
                             required_resets=10, required_frames=5000, wall=_wall(),
                             warm_exit_code=3)

    assert verdict["pass"] is False
    assert any("exit_code" in reason for reason in verdict["reasons"])


def test_verdict_counts_a_warm_reset_reported_as_cold_as_a_warning():
    resets = [_reset(index + 1, 1234, mode="cold") for index in range(10)]

    verdict = probe.evaluate(resets=resets, references=[_reference(1234)],
                             required_resets=10, required_frames=5000, wall=_wall(),
                             warm_exit_code=0)

    assert verdict["pass"] is True
    assert len(verdict["warnings"]) == 9


# ---------------------------------------------------------------------------
# Reply parsing and data-directory safety
# ---------------------------------------------------------------------------
def test_parse_reply_rejects_an_error_reply():
    with pytest.raises(probe.ResetProbeError):
        probe.parse_reply("ERR reset-usage")


def test_validate_hash_rejects_a_short_or_non_hex_value():
    for value in ("", "d97b434", "D97B43478CC24BAE", "d97b43478cc24bax", None):
        with pytest.raises(probe.ResetProbeError):
            probe.validate_hash(value, where="test")

    assert probe.validate_hash("d97b43478cc24bae", where="test") == "d97b43478cc24bae"


def test_create_data_dir_never_reuses_an_existing_path(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    for name in (*probe.LINKED_FILES, *probe.COPIED_FILES):
        (template / name).write_bytes(b"data")

    def fake_link(source, target):
        with open(target, "wb") as handle:
            handle.write(b"link")

    created = probe.create_data_dir(template, tmp_path / "fresh", link=fake_link)
    assert created["created"] is True
    assert set(created["symlinks"]) == set(probe.LINKED_FILES)
    assert set(created["copies"]) == set(probe.COPIED_FILES)

    with pytest.raises(probe.ResetProbeError):
        probe.create_data_dir(template, tmp_path / "fresh", link=fake_link)


# ---------------------------------------------------------------------------
# T7.3: the state block must be the one the reply describes
# ---------------------------------------------------------------------------
class _Block:
    """Stand-in for ``SharedBlock`` with a settable header frame."""

    def __init__(self, *, frame: int) -> None:
        self.frame = frame
        self.closed = 0

    def header(self) -> dict[str, int]:
        return {"version": shm.VERSION, "frame": self.frame, "state_hash": 0,
                "width": 640, "height": 480, "state_bytes": shm.STATE_BYTES,
                "flags": shm.FLAG_STATE_VALID | shm.FLAG_PIXELS_VALID}

    def state(self) -> np.ndarray:
        return np.arange(shm.STATE_FIELDS, dtype="<u4")

    def pixels(self) -> np.ndarray:
        return np.zeros((480, 640, 4), dtype=np.uint8)

    def close(self) -> None:
        self.closed += 1


class _Client:
    """Protocol double: canned RESET/STEP replies, no process."""

    def __init__(self, reset_reply, step_reply=None) -> None:
        self.reset_reply = dict(reset_reply)
        self.step_reply = dict(step_reply or {})
        self.closed = 0
        self.reset_calls: list[dict[str, int]] = []

    def reset(self, **kwargs):
        self.reset_calls.append(dict(kwargs))
        return dict(self.reset_reply)

    def step(self, mask, frames):
        return dict(self.step_reply)

    def close(self) -> None:
        self.closed += 1


def _attached(monkeypatch, tmp_path, *, block_frame: int, reset_reply, step_reply=None):
    monkeypatch.setattr(env, "HarnessClient", lambda *args, **kwargs: _Client(reset_reply,
                                                                             step_reply))
    instance = env.Th08Env(tmp_path / "harness", tmp_path, shm_name="block")
    client = _Client(reset_reply, step_reply)
    instance._client = client
    instance._shm = _Block(frame=block_frame)
    return instance, client


T73_RESET_REPLY = {"_verb": "OK", "reset": "1", "ready": "1", "setup_frames": "357",
                   "frames": "0", "hash": G02_FRAME0_HASH_SEED_1234,
                   "mode": "in-process", "state_published": "1"}
PRE_T73_RESET_REPLY = {"_verb": "OK", "reset": "1", "ready": "1", "setup_frames": "305",
                       "frames": "0", "hash": G02_FRAME0_HASH_SEED_1234}


def test_reset_accepts_a_block_published_for_frame_zero(monkeypatch, tmp_path):
    instance, client = _attached(monkeypatch, tmp_path, block_frame=0,
                                 reset_reply=T73_RESET_REPLY)

    _obs, info = instance.reset(env.StageStart(0, "normal", 0, 1234, "practice"))

    assert info.frame == 0
    assert instance.state()[env.F_DEATHS] == env.F_DEATHS
    assert client.reset_calls[0]["practice"] is True


def test_reset_fails_closed_when_the_block_still_holds_the_previous_episode(monkeypatch,
                                                                           tmp_path):
    instance, _ = _attached(monkeypatch, tmp_path, block_frame=5000,
                            reset_reply=T73_RESET_REPLY)

    with pytest.raises(env.StaleStateError) as caught:
        instance.reset(env.StageStart(0, "normal", 0, 1234, "practice"))

    assert "5000" in str(caught.value) and "0" in str(caught.value)


def test_step_fails_closed_when_the_block_did_not_advance(monkeypatch, tmp_path):
    step_reply = {"_verb": "OK", "advanced": "2", "frame": "2", "hash": "46e2f54401a0fdb7"}
    instance, _ = _attached(monkeypatch, tmp_path, block_frame=0,
                            reset_reply=T73_RESET_REPLY, step_reply=step_reply)

    instance._state_published = True

    with pytest.raises(env.StaleStateError):
        instance.step(0)


def test_step_accepts_a_block_at_the_reported_frame(monkeypatch, tmp_path):
    step_reply = {"_verb": "OK", "advanced": "2", "frame": "2", "hash": "46e2f54401a0fdb7"}
    instance, _ = _attached(monkeypatch, tmp_path, block_frame=2,
                            reset_reply=T73_RESET_REPLY, step_reply=step_reply)

    instance._state_published = True

    _obs, reward, terminated, truncated, info = instance.step(0)

    assert (terminated, truncated) == (False, False)
    assert info.frame == 2
    assert reward == 0.0


def test_a_pre_t7_3_engine_keeps_the_old_behaviour(monkeypatch, tmp_path):
    """No ``state_published`` in the reply: the engine cannot publish, so no guess."""
    instance, _ = _attached(monkeypatch, tmp_path, block_frame=4999,
                            reset_reply=PRE_T73_RESET_REPLY)

    obs, info = instance.reset(env.StageStart(0, "normal", 0, 1234, "practice"))

    assert instance._state_published is False
    assert info.frame == 4999
    assert obs.shape == (480, 640, 4)
