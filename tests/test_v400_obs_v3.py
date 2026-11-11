"""T2 closeout / T3 pre-work: the frozen **obs-v3** contract (D-14, 2026-11-10).

obs-v3 = obs-v1's play/HUD conventions with the playfield plane at 96x112 (EXP-P3-010
input B, 4x4 integer block mean) + obs-v2's ``act_hist`` (8x18, ``a[t-8]..a[t-1]``,
cleared at the shard start and at every stage change).

What this file locks, in order:

* **obs-v1 and obs-v2 do not move** — the sha256 of both *files* and both canonical
  spec digests are literals here, so a silent edit breaks a test instead of a run;
* **obs-v3's digest is frozen** and its payload states the T2a result, the D-14
  ruling, the known limitation (B's F1 0.866 < 0.90) and the policy-facing auxiliary
  label (the teacher's verified surviving action set);
* **the encoder is the probe's input B, byte for byte** — same luminance, same 4x4
  integer block mean, same HUD bytes as obs-v1, and deterministic across calls;
* **``act_hist`` is causal and boundary-cleared** — no current action, no action from
  another shard or another stage;
* **the collector writes v3 only when asked** — ``--obs`` defaults to v1, and
  ``--obs v3`` produces ``play96x112.npy`` + ``act_hist.npy`` with the v3 digest in
  ``shard.json``;
* **the v3 loader round-trips** — ``play (4,112,96) uint8``, ``act_hist (8,18)
  uint8``, ``require_observations=True`` refuses labels without readable pixels, and
  the v1 loader still refuses a v3 shard.

Everything here runs without an engine: the planes are synthetic.  The real
200-decision WSL smoke is the evidence for the collector, not this file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "tools" / "rl")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import v400_d0_dual as dual  # noqa: E402

from touhou_rl.v400.student import (  # noqa: E402
    dataset,
    obs_spec,
    obs_spec_v2,
    obs_spec_v3,
    obs_v3_dataset,
)
from touhou_rl.v400.student.probe import inputs as probe_inputs  # noqa: E402
from touhou_rl.v400.student.probe import native as probe_native  # noqa: E402
from touhou_rl.v400.teacher import obs as teacher_obs  # noqa: E402

# ---------------------------------------------------------------------------
# frozen literals (the whole point of the file: a change must break HERE)
# ---------------------------------------------------------------------------
OBS_V1_FILE = ROOT / "touhou_rl" / "v400" / "student" / "obs_spec.py"
OBS_V2_FILE = ROOT / "touhou_rl" / "v400" / "student" / "obs_spec_v2.py"
FROZEN_V1_FILE_SHA256 = (
    "0d47daf1bddc2b6fb8a33eb54a558f2d40f4a8c3b3d878155035d914d455578b")
FROZEN_V2_FILE_SHA256 = (
    "388b9bd86625200cda8355e4f2c3ddff8deaa993f5fab3a71946dd762a7b6370")
FROZEN_V1_DIGEST = "24f0ddace59104938dbe991c0d8efbb6a8cd21dfd5a29ebcb6d023b5b364a381"
FROZEN_V2_DIGEST = "95110743993c090e2b30fa40ac349a350e74021b35464e3204c784a69ae4dba3"
FROZEN_V3_DIGEST = "ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105"

FRAMES_PER_DECISION = dual.DEFAULT_FRAMES_PER_DECISION
DECISIONS = 6


# ---------------------------------------------------------------------------
# helpers (synthetic planes; every test that writes a shard says so)
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(index: int) -> str:
    return f"{index:016x}"


def readback(seed: int = 0) -> np.ndarray:
    """A deterministic pseudo-random 640x480 RGBA8 readback."""

    return np.random.default_rng(seed).integers(0, 256, (480, 640, 4), dtype=np.uint8)


def label_rows(count: int = DECISIONS) -> list[dict]:
    """``load_labels``-shaped rows with a different executed action per decision."""

    rows = []
    for index in range(count):
        pre = index * FRAMES_PER_DECISION
        rows.append({
            "decision": index,
            "frame": pre + FRAMES_PER_DECISION,
            "frame_pre": pre,
            "hash_pre": {"hash": _hash(1000 + pre), "frames": pre},
            "frame_post": pre + FRAMES_PER_DECISION,
            "hash_post": {"hash": _hash(1000 + pre + FRAMES_PER_DECISION),
                          "frames": pre + FRAMES_PER_DECISION},
            "step_frames": [1, 2, 4, 8],
            "scores": [float(value) for value in range(obs_spec.N_MOVE_CLASSES)],
            "verified_mask": [bool(value % 2) for value in range(obs_spec.N_MOVE_CLASSES)],
            "chosen_plan": {"first_action": (index * 3) % obs_spec.N_MOVE_CLASSES,
                            "slots": [1, 2]},
            "margin": 0.5,
            "bomb": 0,
            "stage": 0,
            "difficulty": 1,
            "frame_event_flags": 0,
            "shoot_release": False,
            "bomb_death_available": True,
        })
    return rows


def synthetic_pixels(count: int = DECISIONS,
                     shape: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    """SYNTHETIC planes with a distinct value per stored render frame.

    ``shape`` selects the play plane: obs-v3's ``(112, 96)`` by default, obs-v1's
    ``(64, 64)`` when the caller wants the v1 path.  Both versions share the 4-frame
    stack, so the render-frame arithmetic is the same.
    """

    play_shape = tuple(shape or obs_spec_v3.PLAY_SHAPE)
    play = np.zeros((count, obs_spec_v3.FRAME_STACK, *play_shape), dtype=np.uint8)
    hud = np.zeros((count, *obs_spec_v3.HUD_SHAPE), dtype=np.uint8)
    render_frames = np.zeros((count, obs_spec_v3.FRAME_STACK), dtype=np.int64)
    pads = np.zeros(count, dtype=np.int64)
    decision_frames = np.zeros(count, dtype=np.int64)
    for index in range(count):
        decision_frame = index * FRAMES_PER_DECISION
        decision_frames[index] = decision_frame
        if index == 0:
            pads[index] = obs_spec_v3.FRAME_STACK - 1
            render_frames[index] = [0] * obs_spec_v3.FRAME_STACK
        else:
            render_frames[index] = list(
                range(decision_frame - obs_spec_v3.FRAME_STACK, decision_frame))
        play[index] = np.arange(obs_spec_v3.FRAME_STACK, dtype=np.uint8)[:, None, None]
        hud[index] = np.uint8(index + 1)
    return {
        "play": play,
        "hud": hud,
        "render_frames": render_frames,
        "pads": pads,
        "decision_frames": decision_frames,
    }


def alignment_doc(rows: list[dict]) -> dict:
    pairs = []
    for row in rows:
        for boundary, frame, checkpoint in (
            ("pre", row["frame_pre"], row["hash_pre"]),
            ("post", row["frame_post"], row["hash_post"]),
        ):
            pairs.append({
                "decision": row["decision"],
                "boundary": boundary,
                "frame": frame,
                "shadow": checkpoint["hash"],
                "env": checkpoint["hash"],
                "env_frames": frame,
                "equal": True,
                "source": "A/B",
            })
    return {
        "schema": "v400-d0-dual-alignment-1",
        "method": "hash-checkpoint",
        "decisions": len(rows),
        "boundaries": len(pairs),
        "checkpoints_equal": len(pairs),
        "checkpoints_mismatched": 0,
        "first_mismatch": None,
        "pairs": pairs,
        "pairs_total": len(pairs),
        "all_equal": True,
        "sha256": dual._canonical_sha256(pairs),
    }


def args_for(tmp_path: Path, *, count: int = DECISIONS, obs_version: str = "v3"):
    return SimpleNamespace(
        decisions=count,
        decisions_requested=count,
        decisions_actual=None,
        stage=0,
        difficulty=1,
        team=0,
        seed=1,
        mode="fast",
        obs=obs_version,
        out_dir=str(tmp_path / "d0"),
        alignment_json=str(tmp_path / "alignment.json"),
        shard_key=f"dual-v3-{count}",
        max_pairs=512,
        trim_transition=False,
    )


def write_alignment(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def channel_reports() -> tuple[dict, dict]:
    a = {"decisions": DECISIONS, "labels_sha256": "a" * 64, "wall_s": 1.0,
         "exit_code": 0, "engine_binary": "/build/rl_hazard2/th08-modern",
         "engine_sha256": "e" * 64, "channel": "A", "role": "shadow"}
    b = {"decisions": DECISIONS, "frames_read": 12, "pixel_ms_per_frame": 1.0,
         "wall_s": 1.0, "engine_binary": "/build/rl_hazard2/th08-modern",
         "engine_sha256": "e" * 64, "channel": "B", "role": "env"}
    return a, b


def build_v3_shard(tmp_path: Path, *, count: int = DECISIONS):
    rows = label_rows(count)
    write_alignment(tmp_path / "alignment.json", alignment_doc(rows))
    args = args_for(tmp_path, count=count)
    a, b = channel_reports()
    summary = dual.assemble(args, rows, synthetic_pixels(count), a, b)
    return summary, rows, args


# ---------------------------------------------------------------------------
# 1. obs-v1 / obs-v2 are frozen: file bytes and canonical digests
# ---------------------------------------------------------------------------
def test_obs_v1_file_bytes_are_unchanged():
    assert _sha256_file(OBS_V1_FILE) == FROZEN_V1_FILE_SHA256
    assert obs_spec.obs_spec_digest() == FROZEN_V1_DIGEST
    assert obs_spec_v3.OBS_V1_FILE_SHA256 == FROZEN_V1_FILE_SHA256
    assert obs_spec_v3.OBS_V1_DIGEST == FROZEN_V1_DIGEST


def test_obs_v2_file_bytes_are_unchanged():
    assert _sha256_file(OBS_V2_FILE) == FROZEN_V2_FILE_SHA256
    assert obs_spec_v2.obs_spec_v2_digest() == FROZEN_V2_DIGEST
    assert obs_spec_v2.OBS_V1_DIGEST == FROZEN_V1_DIGEST
    assert obs_spec_v3.OBS_V2_FILE_SHA256 == FROZEN_V2_FILE_SHA256
    assert obs_spec_v3.OBS_V2_DIGEST == FROZEN_V2_DIGEST


def test_obs_v1_and_v2_payloads_do_not_mention_obs_v3():
    """A frozen spec must not silently absorb obs-v3's resolution or history."""

    v1 = obs_spec.obs_spec()
    v2 = obs_spec_v2.obs_spec_v2()
    assert v1["version"] == "obs-v1"
    assert v1["play"]["shape"] == list(obs_spec.PLAY_SHAPE) == [64, 64]
    assert "action_history" not in v1
    assert v2["version"] == "obs-v2"
    assert v2["play"]["shape"] == [64, 64]
    assert v2["action_history"]["decisions"] == 8


# ---------------------------------------------------------------------------
# 2. obs-v3 itself: digest, payload, encoder
# ---------------------------------------------------------------------------
def test_obs_v3_digest_is_frozen():
    assert obs_spec_v3.obs_spec_v3_digest() == FROZEN_V3_DIGEST
    assert teacher_obs.FROZEN_OBS_SPEC_V3_DIGEST == FROZEN_V3_DIGEST
    obs_spec_v3.check_obs_spec_v3("obs-v3", FROZEN_V3_DIGEST)
    with pytest.raises(obs_spec_v3.ObsSpecV3Mismatch):
        obs_spec_v3.check_obs_spec_v3("obs-v1", FROZEN_V1_DIGEST)
    with pytest.raises(obs_spec_v3.ObsSpecV3Mismatch):
        obs_spec_v3.check_obs_spec_v3("obs-v3", FROZEN_V2_DIGEST)


def test_obs_v3_payload_contract():
    spec = obs_spec_v3.obs_spec_v3()
    assert spec["version"] == "obs-v3"
    assert spec["supersedes"] == [
        {"version": "obs-v2", "digest": FROZEN_V2_DIGEST},
        {"version": "obs-v1", "digest": FROZEN_V1_DIGEST},
    ]
    play = spec["play"]
    assert play["plane"] == "play96x112"
    assert play["shape"] == [112, 96]
    assert play["extent_wh"] == [96, 112]
    assert play["block"] == [4, 4]
    assert play["stack"] == 4
    assert play["stack_order"] == obs_spec.STACK_ORDER
    assert play["pixels_per_frame"] == 10_752
    # obs-v2's action history is carried over verbatim (same contract, same order).
    assert spec["action_history"]["decisions"] == 8
    assert spec["action_history"]["order"] == "oldest-first a[t-8],...,a[t-1]"
    assert spec["action_history"]["excludes_current_action"] is True
    # HUD and normalisation are obs-v1's.
    assert spec["hud"]["shape"] == list(obs_spec.HUD_SHAPE) == [56, 160]
    assert spec["normalization"] == obs_spec.NORMALIZATION
    assert spec["decision_frames"] == obs_spec.DECISION_FRAMES
    # The ruling and the known limitation are part of the frozen bytes.
    resolution = spec["resolution"]
    assert resolution["ruling"]["id"] == "D-14"
    assert resolution["ruling"]["date"] == "2026-11-10"
    assert resolution["gate_f1"] == 0.90
    assert resolution["candidates"]["B"]["selected"] is True
    assert resolution["candidates"]["D"]["rejected_because"].startswith(
        "pixel player locator median error 11.59 px")
    assert "0.8661" in resolution["ruling"]["known_limitation"]
    assert "< 0.90" in resolution["ruling"]["known_limitation"]
    assert "occupancy F1" in resolution["ruling"]["decision"]
    assert "closed loop" in resolution["ruling"]["criterion"]
    assert spec["self_state"]["present"] is False
    assert spec["danger_summary"]["present"] is False
    assert spec["policy_aux_label"]["source"].startswith("TeacherLabel.verified_mask")
    assert spec["bytes_per_decision"]["total"] == 52_112


def test_encode_planes_v3_matches_probe_input_b_byte_for_byte():
    """The frozen encoder IS EXP-P3-010's input B (otherwise the ruling is moot)."""

    for seed in range(4):
        pixels = readback(seed)
        play, hud = obs_spec_v3.encode_planes_v3(pixels)
        assert play.shape == obs_spec_v3.PLAY_SHAPE == (112, 96)
        assert play.dtype == np.uint8
        fields = probe_native.encode_native(pixels)
        assert np.array_equal(play, fields["play96x112"])
        tensor = probe_inputs.build_input("B", fields)
        assert tensor.shape == (1, 112, 96)
        assert np.array_equal(np.rint(tensor[0] * 255.0).astype(np.uint8), play)
        # obs-v1's own plane is a different resolution but the same luminance rule.
        assert np.array_equal(fields["play64"], obs_spec.encode_planes(pixels)[0])


def test_encode_planes_v3_hud_is_obs_v1_hud():
    pixels = readback(11)
    _play, hud = obs_spec_v3.encode_planes_v3(pixels)
    _play64, hud_v1 = obs_spec.encode_planes(pixels)
    assert hud.shape == obs_spec.HUD_SHAPE
    assert np.array_equal(hud, hud_v1)


def test_encode_planes_v3_is_deterministic():
    pixels = readback(5)
    first = obs_spec_v3.encode_planes_v3(pixels)
    second = obs_spec_v3.encode_planes_v3(pixels.copy())
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert hashlib.sha256(first[0].tobytes()).hexdigest() == hashlib.sha256(
        second[0].tobytes()).hexdigest()


def test_encode_planes_v3_rejects_a_non_rgba8_readback():
    with pytest.raises(obs_spec_v3.ObsSpecV3Error):
        obs_spec_v3.encode_planes_v3(np.zeros((480, 640, 3), dtype=np.uint8))
    with pytest.raises(obs_spec_v3.ObsSpecV3Error):
        obs_spec_v3.encode_planes_v3(np.zeros((120, 160, 4), dtype=np.uint8))


def test_normalisation_is_div_255_on_the_new_shape():
    stack = np.full((obs_spec_v3.FRAME_STACK, *obs_spec_v3.PLAY_SHAPE), 255, np.uint8)
    normalized = obs_spec_v3.normalize_play(stack)
    assert normalized.shape == (4, 112, 96)
    assert normalized.dtype == np.float32
    assert normalized.max() == 1.0
    with pytest.raises(obs_spec_v3.ObsSpecV3Error):
        obs_spec_v3.normalize_play(np.zeros((4, 64, 64), dtype=np.uint8))


def test_observation_buffer_v3_arrays_are_model_ready():
    buffer = obs_spec_v3.ObservationBufferV3()
    play, hud = obs_spec_v3.encode_planes_v3(readback(3))
    buffer.push(play, hud)
    arrays = buffer.arrays()
    assert arrays["play"].shape == (4, 112, 96)
    assert arrays["hud"].shape == (1, 56, 160)
    assert arrays["act_hist"].shape == (144,)
    assert np.array_equal(arrays["play"][0], arrays["play"][3])  # padded, not invented
    history = obs_spec_v2.ActionHistoryBuffer()
    history.push(4)
    history.push(7)
    arrays = buffer.arrays(history)
    assert arrays["act_hist"].reshape(8, 18)[-1, 7] == 1.0
    assert arrays["act_hist"].reshape(8, 18)[-2, 4] == 1.0


# ---------------------------------------------------------------------------
# 3. act_hist: causality and boundary clearing
# ---------------------------------------------------------------------------
def test_action_history_is_causal_and_ordered():
    cand = np.array([2, 5, 9, 1, 17, 0, 3, 8, 6, 4], dtype=np.int16)
    stage = np.zeros(len(cand), dtype=np.int32)
    history = obs_spec_v2.build_action_history(cand, stage)
    assert history.shape == (len(cand), 8, 18)
    assert history.dtype == np.uint8
    # decision 0 has no past decision: the window is all-zero padding.
    assert history[0].sum() == 0
    for t in range(1, len(cand)):
        # row 7 is a[t-1] (the decision immediately before t), row 0 is a[t-8] once
        # eight past decisions exist.
        assert history[t, 7, int(cand[t - 1])] == 1
        assert history[t, 7].sum() == 1
        if t >= 8:
            assert int(np.argmax(history[t, 0])) == int(cand[t - 8])
        else:
            assert history[t, 0].sum() == 0  # shard start: zero padding
    # "no leak": a[t] is absent from row t whenever a[t] differs from every
    # a[t-8..t-1] (the synthetic sequence has no repeats inside the window).
    for t in range(1, len(cand)):
        window = [int(value) for value in cand[max(0, t - 8):t]]
        assert int(cand[t]) not in window
        assert all(int(np.argmax(row)) != int(cand[t])
                   for row in history[t] if row.sum() == 1)


def test_action_history_clears_at_stage_change():
    cand = np.arange(10, dtype=np.int16)
    stage = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    history = obs_spec_v2.build_action_history(cand, stage)
    # Before the boundary: shard-start padding then a real window.
    assert history[0].sum() == 0
    assert history[3, 7, int(cand[2])] == 1
    # The stage change at t=4 clears the window even though past actions exist.
    assert history[4].sum() == 0
    assert history[5, 7, int(cand[4])] == 1
    assert history[5, 6].sum() == 0  # the previous stage's actions are gone


def test_action_history_buffer_matches_the_reconstruction():
    cand = np.arange(12, dtype=np.int16)
    history = obs_spec_v2.build_action_history(cand, np.zeros(len(cand), dtype=np.int32))
    buffer = obs_spec_v2.ActionHistoryBuffer()
    for index, action in enumerate(cand):
        # The buffer is read *before* the decision's own action is pushed: the
        # window is strictly a[t-8..t-1], never a[t].
        assert np.array_equal(buffer.rows(), history[index])
        buffer.push(int(action))


def test_v3_action_history_helper_uses_the_executed_action():
    rows = label_rows(10)
    history, info = dual.v3_action_history(rows)
    executed = [row["chosen_plan"]["first_action"] for row in rows]
    assert history.shape == (10, 8, 18)
    for t in range(1, 10):
        assert int(np.argmax(history[t, 7])) == executed[t - 1]
    assert info["shape"] == [10, 8, 18]
    assert info["stage_changes"] == 0
    assert info["excludes_current_action"] is True
    assert info["zero_rows_at_start"] == 8
    assert info["sha256"] == obs_spec_v2.history_digest(history)


def test_v3_action_history_refuses_rows_without_an_executed_action():
    rows = label_rows(4)
    rows[2]["chosen_plan"] = None
    with pytest.raises(dual.DualError):
        dual.v3_action_history(rows)


# ---------------------------------------------------------------------------
# 4. the collector: default stays v1, --obs v3 writes the v3 planes
# ---------------------------------------------------------------------------
def test_collector_defaults_to_v1():
    args = dual.parse_args([])
    assert args.obs == "v1"
    assert dual.obs_version_spec(args.obs).short == "v1"
    assert dual.parse_args(["--obs", "v3"]).obs == "v3"
    with pytest.raises(SystemExit):
        dual.parse_args(["--obs", "v2"])


def test_collector_read_frame_returns_112x96_for_v3():
    from touhou_rl.v400.gym.shm import FLAG_PIXELS_VALID

    pixels = readback(8)

    class _Block:
        def header(self):
            return {"flags": FLAG_PIXELS_VALID, "width": 640, "height": 480, "frame": 7}

        def pixels(self):
            return pixels

    play, hud, header = dual._read_frame(_Block(), "test", "v3")
    assert play.shape == (112, 96)
    assert np.array_equal(play, obs_spec_v3.encode_planes_v3(pixels)[0])
    assert header["frame"] == 7
    play64, hud64, _ = dual._read_frame(_Block(), "test", "v1")
    assert play64.shape == (64, 64)
    assert np.array_equal(hud, hud64)


def test_collector_assemble_writes_obs_v3(tmp_path):
    summary, rows, args = build_v3_shard(tmp_path)
    shard_dir = Path(summary["dir"])
    assert summary["obs_spec_version"] == "obs-v3"
    assert summary["obs_spec_digest"] == FROZEN_V3_DIGEST
    assert summary["obs_version"] == "v3"
    assert (shard_dir / "obs" / "play96x112.npy").is_file()
    assert (shard_dir / "obs" / "act_hist.npy").is_file()
    assert (shard_dir / "obs" / "hud.npy").is_file()
    raw = json.loads((shard_dir / "shard.json").read_text(encoding="utf-8"))
    assert raw["obs_spec_version"] == "obs-v3"
    assert raw["obs_spec_digest"] == FROZEN_V3_DIGEST
    assert raw["obs_version"] == "v3"
    assert "play96x112" in raw["obs"] and "act_hist" in raw["obs"]
    assert raw["obs"]["bytes_per_decision"] == summary["obs_bytes"] // len(rows)
    assert raw["requested_obs"] if "requested_obs" in raw else True  # documented key
    assert raw["obs_requested"] == "v3"
    assert raw["alignment"]["action_history_excludes_current_action"] is True
    assert raw["alignment"]["observation_time"] == "decision_time_before_commit"
    expected = obs_spec_v2.build_action_history(
        np.array([row["chosen_plan"]["first_action"] for row in rows], dtype=np.int16),
        np.zeros(len(rows), dtype=np.int32))
    stored = np.load(shard_dir / "obs" / "act_hist.npy")
    assert np.array_equal(stored, expected)


def test_collector_assemble_v1_is_unchanged(tmp_path):
    rows = label_rows(DECISIONS)
    write_alignment(tmp_path / "alignment.json", alignment_doc(rows))
    args = args_for(tmp_path, obs_version="v1")
    a, b = channel_reports()
    pixels = synthetic_pixels(DECISIONS, shape=obs_spec.PLAY_SHAPE)
    summary = dual.assemble(args, rows, pixels, a, b)
    shard_dir = Path(summary["dir"])
    assert summary["obs_spec_version"] == "obs-v1"
    assert summary["obs_spec_digest"] == FROZEN_V1_DIGEST
    assert (shard_dir / "obs" / "play64.npy").is_file()
    assert (shard_dir / "obs" / "hud.npy").is_file()
    assert not (shard_dir / "obs" / "act_hist.npy").exists()
    raw = json.loads((shard_dir / "shard.json").read_text(encoding="utf-8"))
    assert raw["obs_spec_version"] == "obs-v1"
    assert "play64" in raw["obs"]
    assert dataset.load_teacher_shard(shard_dir).play.shape[2:] == obs_spec.PLAY_SHAPE


# ---------------------------------------------------------------------------
# 5. the loader round-trip and its gates
# ---------------------------------------------------------------------------
def test_v3_loader_round_trip(tmp_path):
    summary, rows, _args = build_v3_shard(tmp_path)
    shard = obs_v3_dataset.load_teacher_shard_v3(summary["dir"])
    assert len(shard) == len(rows)
    sample = shard[0]
    assert sample["play"].shape == (4, 112, 96)
    assert sample["play"].dtype == np.uint8
    assert sample["hud"].shape == (56, 160)
    assert sample["act_hist"].shape == (8, 18)
    assert sample["act_hist"].dtype == np.uint8
    assert sample["scores"].shape == (18,)
    assert sample["verified_mask"].shape == (18,)
    assert sample["verified_actions"] == [cand for cand in range(18) if cand % 2]
    assert sample["obs_spec_version"] == "obs-v3"
    assert sample["obs_spec_digest"] == FROZEN_V3_DIGEST
    normalized = shard.normalized(1)
    assert normalized["play"].shape == (4, 112, 96)
    assert normalized["play"].dtype == np.float32
    assert normalized["act_hist"].shape == (144,)
    report = shard.report()
    assert report["act_hist_shape"] == [len(rows), 8, 18]
    assert report["play_shape"] == [len(rows), 4, 112, 96]
    assert report["obs_spec_digest"] == FROZEN_V3_DIGEST


def test_v1_loader_still_refuses_a_v3_shard(tmp_path):
    summary, _rows, _args = build_v3_shard(tmp_path)
    with pytest.raises(dataset.DatasetError):
        dataset.load_teacher_shard(summary["dir"])


def test_v3_loader_refuses_a_tampered_act_hist(tmp_path):
    summary, _rows, _args = build_v3_shard(tmp_path)
    path = Path(summary["dir"]) / "obs" / "act_hist.npy"
    array = np.load(path)
    array[3, 0] = np.zeros(18, dtype=np.uint8)  # erase the oldest row of decision 3
    np.save(path, array)
    with pytest.raises(dataset.DatasetError, match="sha256"):
        obs_v3_dataset.load_teacher_shard_v3(summary["dir"])
    # ... and the history rebuild catches it even when the sha gate is skipped.
    with pytest.raises(dataset.DatasetError, match="act_hist"):
        obs_v3_dataset.load_teacher_shard_v3(summary["dir"], verify_sha=False)


def test_v3_loader_refuses_a_v1_digest(tmp_path):
    summary, _rows, _args = build_v3_shard(tmp_path)
    shard_json = Path(summary["dir"]) / "shard.json"
    payload = json.loads(shard_json.read_text(encoding="utf-8"))
    payload["obs_spec_digest"] = FROZEN_V1_DIGEST
    shard_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="obs spec"):
        obs_v3_dataset.load_teacher_shard_v3(summary["dir"])


def test_v3_loader_refuses_a_wrong_play_shape(tmp_path):
    summary, _rows, _args = build_v3_shard(tmp_path)
    shard_json = Path(summary["dir"]) / "shard.json"
    payload = json.loads(shard_json.read_text(encoding="utf-8"))
    payload["obs"]["play96x112"]["path"] = "obs/play64.npy"
    shard_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="play96x112"):
        obs_v3_dataset.load_teacher_shard_v3(summary["dir"])


def test_v3_samples_require_observations(tmp_path):
    summary, rows, _args = build_v3_shard(tmp_path)
    labels = Path(summary["dir"]) / "labels.jsonl"
    obs_root = Path(summary["dir"]).parents[1]  # <tmp>/d0 : the root holding shards/
    loaded = obs_v3_dataset.load_teacher_samples_v3(
        [labels], min_samples=1, obs_root=obs_root, require_observations=True)
    assert loaded["count"] == len(rows)
    assert loaded["obs_spec_digest"] == FROZEN_V3_DIGEST
    assert loaded["obs_shapes"]["play"] == [112, 96]
    arrays = loaded["samples"][0]["obs_arrays"]
    assert arrays["play"].shape == (4, 112, 96)
    assert arrays["act_hist"].shape == (8, 18)
    assert arrays["verified_actions"] == [cand for cand in range(18) if cand % 2]
    # A label file whose shard is not under obs_root cannot be resolved ...
    with pytest.raises(dataset.DatasetInsufficient, match="obs-v3 shard"):
        obs_v3_dataset.load_teacher_samples_v3(
            [labels], min_samples=1, obs_root=tmp_path / "elsewhere",
            require_observations=True)
    # ... and without the gate it is reported, not silently dropped.
    loose = obs_v3_dataset.load_teacher_samples_v3(
        [labels], min_samples=1, obs_root=tmp_path / "elsewhere")
    assert loose["obs_unresolved"]
    assert loose["obs_shapes"] is None


def test_v3_bytes_per_decision_and_two_million_frame_estimate():
    """The T3 disk estimate: 52,112 B of plane payload per decision."""

    assert obs_spec_v3.BYTES_PER_DECISION == (
        4 * 112 * 96 + 56 * 160 + 8 * 18)
    assert obs_spec_v3.BYTES_PER_DECISION == 52_112
    frames = 2_000_000
    decisions = frames // dual.DEFAULT_FRAMES_PER_DECISION
    assert decisions == 500_000
    total = decisions * obs_spec_v3.BYTES_PER_DECISION
    assert total == 26_056_000_000  # ~26.06 GB of plane payload
    # The v1 payload for the same frame count, for the comparison the card makes.
    v1 = 4 * 64 * 64 + 56 * 160
    assert v1 == 25_344
    assert total / (decisions * v1) == pytest.approx(52_112 / 25_344, rel=1e-12)
    assert total / (decisions * v1) == pytest.approx(2.0562, abs=1e-4)


def test_shard_bytes_per_decision_includes_only_the_npy_headers(tmp_path):
    summary, rows, _args = build_v3_shard(tmp_path)
    measured = summary["obs_bytes"] // len(rows)
    payload = obs_spec_v3.BYTES_PER_DECISION
    headers = 3 * (8 + 2 + 512)  # three planes, fixed 512-byte npy header each
    assert summary["obs_bytes"] == len(rows) * payload + headers
    assert payload <= measured <= payload + headers
