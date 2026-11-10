"""Tests for the P3 S9 slice: obs-v1 spec, dataset contract, model, eval schema.

Run (Windows)::

    $env:PYTHONPATH='D:\\WorkShop\\Touhou_RL'
    python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p3/pytest -q tests/test_v400_student.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from touhou_rl.v400 import gym as gym_pkg
from touhou_rl.v400.gym import actions as gym_actions
from touhou_rl.v400.gym.env import DECISION_FRAMES as GYM_DECISION_FRAMES
from touhou_rl.v400.student import dataset, model, obs_spec
from touhou_rl.v400.teacher import config as teacher_config

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = REPO_ROOT / "tools" / "rl" / "v400_p2_bulk_collect.py"


def _rng_pixels(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(480, 640, 4), dtype=np.uint8)


# ---------------------------------------------------------------------------
# obs spec
# ---------------------------------------------------------------------------
def test_obs_spec_frozen_constants():
    assert obs_spec.OBS_SPEC_VERSION == "obs-v1"
    assert obs_spec.PLAY_SHAPE == (64, 64)
    assert obs_spec.FRAME_STACK == 4
    assert obs_spec.HUD_SHAPE == (56, 160)
    assert obs_spec.N_MOVE_CLASSES == 18
    assert obs_spec.N_BOMB_CLASSES == 2
    assert obs_spec.DECISION_FRAMES == GYM_DECISION_FRAMES == teacher_config.DECISION_FRAMES == 2
    assert obs_spec.ACTION_DURATION_FRAMES == obs_spec.DECISION_FRAMES
    digest = obs_spec.obs_spec_digest()
    assert len(digest) == 64 and digest == obs_spec.obs_spec_digest()
    json.dumps(obs_spec.obs_spec())  # must stay serialisable for checkpoints


def test_obs_spec_guard_rejects_other_version():
    with pytest.raises(obs_spec.ObsSpecMismatch):
        obs_spec.check_obs_spec("obs-v0", obs_spec.obs_spec_digest())
    with pytest.raises(obs_spec.ObsSpecMismatch):
        obs_spec.check_obs_spec("obs-v1", "0" * 64)
    obs_spec.check_obs_spec(obs_spec.OBS_SPEC_VERSION, obs_spec.obs_spec_digest())


def test_encode_planes_is_byte_exact_with_p2_collector():
    import sys

    name = "v400_p2_bulk_collect_test"
    spec = importlib.util.spec_from_file_location(name, COLLECTOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    pixels = _rng_pixels(7)
    _play96_ref, play_ref, hud_ref = module.encode_frame(
        pixels, playfield_origin=(32, 16), playfield=(384, 448))
    play, hud = obs_spec.encode_planes(pixels)
    assert np.array_equal(play, play_ref)
    assert np.array_equal(hud, hud_ref)


def test_encode_planes_rejects_bad_input():
    with pytest.raises(obs_spec.ObsSpecError):
        obs_spec.encode_planes(np.zeros((480, 640, 3), dtype=np.uint8))
    with pytest.raises(obs_spec.ObsSpecError):
        obs_spec.encode_planes(np.zeros((100, 100, 4), dtype=np.uint8))


def test_normalize_and_buffer_padding():
    play = np.full((64, 64), 255, dtype=np.uint8)
    hud = np.full((56, 160), 128, dtype=np.uint8)
    buffer = obs_spec.ObservationBuffer()
    buffer.push(play, hud)
    stack = buffer.stack()
    assert stack.shape == (4, 64, 64)
    assert all(np.array_equal(stack[index], play) for index in range(4))
    arrays = buffer.arrays()
    assert arrays["play"].dtype == np.float32
    assert arrays["play"].shape == (4, 64, 64)
    assert arrays["play"].max() == pytest.approx(1.0)
    assert arrays["hud"].shape == (1, 56, 160)
    assert arrays["hud"].max() == pytest.approx(128 / 255.0)
    with pytest.raises(obs_spec.ObsSpecError):
        obs_spec.normalize_play(np.zeros((3, 64, 64), dtype=np.uint8))


# ---------------------------------------------------------------------------
# synthetic dataset
# ---------------------------------------------------------------------------
def _write_shard(root: Path, *, shard: str = "still-s0d1-seed1-n12-w00", frames: int = 12,
                 source: str = "still", stage: int = 0, seed: int = 1,
                 masks: list[int] | None = None) -> dict:
    shard_dir = root / "shards" / shard / "attempt-01"
    shard_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    play = rng.integers(0, 255, size=(frames, 64, 64), dtype=np.uint8)
    hud = rng.integers(0, 255, size=(frames, 56, 160), dtype=np.uint8)
    play.tofile(shard_dir / "play64.u8")
    hud.tofile(shard_dir / "hud.u8")
    if masks is None:
        masks = [gym_actions.cand_to_mask(index % 18, 0) for index in range(frames)]
    rows = [{"i": index, "f": index, "go": 0, "msk": int(mask)}
            for index, mask in enumerate(masks)]
    (shard_dir / "labels.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {
        "shard": shard, "attempt": "attempt-01", "dir": str(shard_dir),
        "frames": frames, "source": source, "stage": stage, "difficulty": 1,
        "seed": seed,
    }


def _write_index_split(root: Path, shard_rows: list[dict], stage: int = 0) -> tuple[Path, Path]:
    rows = list(shard_rows) + [
        {**shard_rows[0], "shard": "val-placeholder", "stage": stage + 5,
         "dir": str(root / "shards" / "val-placeholder")},
        {**shard_rows[0], "shard": "test-placeholder", "stage": stage + 3,
         "dir": str(root / "shards" / "test-placeholder")},
    ]
    frames_total = sum(row["frames"] for row in rows)
    index = {
        "schema": dataset.INDEX_SCHEMA,
        "out_dir": str(root),
        "frames_total": frames_total,
        "shards": rows,
        "plane_shapes": {name: list(shape) for name, shape in dataset.PLANE_SHAPES.items()},
        "label_version": dataset.LABEL_VERSION,
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    entries = []
    for row in rows:
        entry = dict(row)
        entry["split"] = "train" if row["stage"] == stage else "val"
        entries.append(entry)
    entries[-1]["split"] = "test"
    split = {
        "schema": dataset.SPLIT_SCHEMA,
        "stage_splits": {"train": [stage], "val": [stage + 5], "test": [stage + 3]},
        "machine": "stage (whole shards, no same-run leakage)",
        "engine_sha256": "f" * 64,
        "label_version": dataset.LABEL_VERSION,
        "frames": {"train": sum(e["frames"] for e in entries if e["split"] == "train"),
                   "val": sum(e["frames"] for e in entries if e["split"] == "val"),
                   "test": 0},
        "shards": entries,
    }
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return index_path, split_path


def test_dataset_loads_synthetic(tmp_path):
    rows = [_write_shard(tmp_path, frames=12)]
    index_path, split_path = _write_index_split(tmp_path, rows)
    ds = dataset.build_dataset(index_path, split_path, split="train")
    assert len(ds) == 12 - obs_spec.FRAME_STACK
    sample = ds[0]
    assert sample["schema"] == dataset.STUDENT_SAMPLE_SCHEMA
    assert sample["play"].shape == (4, 64, 64)
    assert sample["hud"].shape == (56, 160)
    assert sample["move"] == 4  # label comes from row frame+1 = 4
    assert sample["bomb"] == 0
    assert sample["frame"] == obs_spec.FRAME_STACK - 1
    report = ds.report()
    assert report["kind"] == "bc-v0-behaviour-cloning"
    assert report["disclosure"] == dataset.V0_DISCLOSURE
    assert report["actions_unmappable"] == 0
    assert report["windows"] == len(ds)


def test_dataset_fail_closed_on_label_version(tmp_path):
    rows = [_write_shard(tmp_path, frames=8)]
    index_path, _ = _write_index_split(tmp_path, rows)
    payload = json.loads(index_path.read_text())
    payload["label_version"] = "p2-s4-1"
    index_path.write_text(json.dumps(payload))
    with pytest.raises(dataset.DatasetError, match="label_version"):
        dataset.load_index(index_path)


def test_dataset_fail_closed_on_plane_size(tmp_path):
    rows = [_write_shard(tmp_path, frames=8)]
    index_path, split_path = _write_index_split(tmp_path, rows)
    plane = tmp_path / "shards" / rows[0]["shard"] / "attempt-01" / "play64.u8"
    plane.write_bytes(plane.read_bytes()[:-64])
    with pytest.raises(dataset.DatasetError, match="expected"):
        dataset.build_dataset(index_path, split_path, split="train")


def test_dataset_fail_closed_on_missing_hud(tmp_path):
    rows = [_write_shard(tmp_path, frames=8)]
    index_path, split_path = _write_index_split(tmp_path, rows)
    (tmp_path / "shards" / rows[0]["shard"] / "attempt-01" / "hud.u8").unlink()
    with pytest.raises(dataset.DatasetError, match="missing plane"):
        dataset.build_dataset(index_path, split_path, split="train")


def test_dataset_fail_closed_on_unmappable_mask(tmp_path):
    masks = [gym_actions.cand_to_mask(index % 18, 0) for index in range(8)]
    masks[5] = 0  # no shoot bit: not a student action
    rows = [_write_shard(tmp_path, frames=8, masks=masks)]
    index_path, split_path = _write_index_split(tmp_path, rows)
    ds = dataset.build_dataset(index_path, split_path, split="train")
    # windows are frames 3..6; the bad mask at row 5 is the label of frame 4
    with pytest.raises(dataset.DatasetError, match="not a student action"):
        _ = ds[1]
    assert ds.report()["actions_unmappable"] == 1


def test_split_stage_leakage_fails_closed(tmp_path):
    rows = [_write_shard(tmp_path, frames=8, stage=0),
            _write_shard(tmp_path, shard="val-s5d1-seed2-n8-w00", frames=8, stage=5)]
    index_path, split_path = _write_index_split(tmp_path, rows)
    split = json.loads(split_path.read_text())
    split["shards"][1]["split"] = "train"
    split_path.write_text(json.dumps(split))
    with pytest.raises(dataset.DatasetError, match="stage_splits"):
        dataset.load_split(split_path)


# ---------------------------------------------------------------------------
# teacher label contract
# ---------------------------------------------------------------------------
def _teacher_rows(with_obs: bool, count: int = 2) -> list[dict]:
    rows = [{"kind": "report", "status": "ok"}]
    for index in range(count):
        row = {
            "kind": "decision",
            "frame": index * 2,
            "stage": 1,
            "difficulty": 1,
            "scores": [100.0 + index] * 18,
            "verified_mask": [True] * 18,
            "bomb": 0,
            "margin": 16.0,
            "chosen_plan": {"first_action": 0, "duration": 1},
        }
        if with_obs:
            row["obs"] = {"shard": "teacher-s1", "frame": index * 2}
        rows.append(row)
    return rows


def test_teacher_labels_scan_counts(tmp_path):
    path = tmp_path / "teacher.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _teacher_rows(False)), encoding="utf-8")
    scan = dataset.scan_teacher_labels([path])
    assert scan["decisions"] == 2
    assert scan["with_observation_reference"] == 0
    assert scan["invalid"] == 0


def test_teacher_labels_fail_closed_without_observations(tmp_path):
    path = tmp_path / "teacher.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _teacher_rows(False)), encoding="utf-8")
    with pytest.raises(dataset.DatasetInsufficient) as info:
        dataset.load_teacher_samples([path])
    message = str(info.value)
    assert "no observation reference" in message
    assert "2 decisions" in message
    assert dataset.V0_DISCLOSURE in message


def test_teacher_labels_fail_closed_when_too_few(tmp_path):
    path = tmp_path / "teacher.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in _teacher_rows(True)), encoding="utf-8")
    with pytest.raises(dataset.DatasetInsufficient, match="insufficient"):
        dataset.load_teacher_samples([path], min_samples=dataset.MIN_TEACHER_SAMPLES)


def test_teacher_labels_load_when_schema_complete(tmp_path):
    rows = _teacher_rows(True)
    # the teacher child log spells the vector ``geo_score``; both spellings are accepted
    rows[1]["geo_score"] = rows[1].pop("scores")
    rows[2]["verified_mask"] = [False] * 18  # valid label, zero verified actions
    path = tmp_path / "teacher.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    payload = dataset.load_teacher_samples([path], min_samples=1)
    assert payload["count"] == 2
    record = payload["samples"][0]
    assert record["scores"].shape == (18,)
    assert record["verified_mask"].shape == (18,)
    assert record["verified_count"] == 18
    assert payload["samples"][1]["verified_count"] == 0
    assert record["observation"]["obs"]["shard"] == "teacher-s1"


# ---------------------------------------------------------------------------
# model / checkpoint
# ---------------------------------------------------------------------------
def test_model_forward_and_parameters():
    torch = pytest.importorskip("torch")
    net = model.StudentNet()
    net.eval()
    play = torch.zeros(2, obs_spec.FRAME_STACK, 64, 64)
    hud = torch.zeros(2, 1, 56, 160)
    move_logits, bomb_logits = net(play, hud)
    assert move_logits.shape == (2, 18)
    assert bomb_logits.shape == (2, 2)
    assert 50_000 < net.n_parameters() < 2_000_000
    with pytest.raises(ValueError):
        net(torch.zeros(2, 3, 64, 64), hud)


def test_checkpoint_roundtrip_and_guard(tmp_path):
    torch = pytest.importorskip("torch")
    net = model.StudentNet()
    net.eval()
    path = model.save_checkpoint(tmp_path / "ckpt.pt", net,
                                 train={"steps": 1}, provenance={"unit": "test"})
    loaded, payload = model.load_checkpoint(path)
    assert payload["obs_spec_version"] == obs_spec.OBS_SPEC_VERSION
    assert payload["obs_spec_digest"] == obs_spec.obs_spec_digest()
    assert loaded.n_parameters() == net.n_parameters()
    play = torch.zeros(1, 4, 64, 64)
    hud = torch.zeros(1, 1, 56, 160)
    with torch.no_grad():
        before = net(play, hud)[0]
        after = loaded(play, hud)[0]
    assert torch.allclose(before, after)
    payload["obs_spec_digest"] = "0" * 64
    torch.save(payload, path)
    with pytest.raises(obs_spec.ObsSpecMismatch):
        model.load_checkpoint(path)


# ---------------------------------------------------------------------------
# evaluation record schema
# ---------------------------------------------------------------------------
def _valid_record() -> dict:
    return {
        "schema": "v400-student-eval-episode-1",
        "checkpoint": ".tmp_v400_p3/checkpoint-v0.pt",
        "checkpoint_sha256": "a" * 64,
        "obs_spec_version": obs_spec.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec.obs_spec_digest(),
        "seed": 1000,
        "suite": "S1N",
        "stage": 1,
        "difficulty": "normal",
        "max_frames": 19887,
        "frames": 1234,
        "deaths": 3.0,
        "score": 4321,
        "end": "game_over",
        "privilege": "deploy_pixel",
        "wall_s": 1.5,
        "engine_sha256": "b" * 64,
        "policy_input": "play64 4-frame stack + hud strip; no privileged state",
        "not_a_gate": True,
    }


def test_eval_record_schema_accepts_valid_and_rejects_bad():
    import importlib.util as importlib_util

    path = REPO_ROOT / "tools" / "rl" / "v400_student_eval.py"
    spec = importlib_util.spec_from_file_location("v400_student_eval_test", path)
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_episode_record(_valid_record())
    for key in ("frames", "deaths", "end", "privilege", "obs_spec_digest"):
        broken = _valid_record()
        broken.pop(key)
        with pytest.raises(module.EvalError):
            module.validate_episode_record(broken)
    dirty = _valid_record()
    dirty["privilege"] = "privileged"
    with pytest.raises(module.EvalError, match="deploy_pixel"):
        module.validate_episode_record(dirty)
    gated = _valid_record()
    gated["not_a_gate"] = False
    with pytest.raises(module.EvalError, match="not_a_gate"):
        module.validate_episode_record(gated)
    bad_end = _valid_record()
    bad_end["end"] = "cleared"
    with pytest.raises(module.EvalError, match="end"):
        module.validate_episode_record(bad_end)


def test_action_space_matches_teacher_candidates():
    assert obs_spec.N_MOVE_CLASSES == teacher_config.N_CAND == gym_actions.N_CAND == 18
    assert len(gym_actions.STUDENT_ACTIONS) == 36
    for cand in range(obs_spec.N_MOVE_CLASSES):
        mask = gym_actions.cand_to_mask(cand, 0)
        assert gym_actions.mask_to_cand(mask) == (cand, 0)


# ---------------------------------------------------------------------------
# BC v1: teacher soft labels, tau calibration, shard split, checkpoint digest
# ---------------------------------------------------------------------------
def _scores_batch(n: int = 4) -> np.ndarray:
    """Teacher scores shaped like D0: ~5 point top-1/top-2 gap on a 111-118 band."""

    rng = np.random.default_rng(7)
    scores = rng.normal(112.0, 0.3, size=(n, obs_spec.N_MOVE_CLASSES))
    scores[:, 7] = scores[:, 7] + 5.0
    return scores


def test_teacher_soft_labels_and_kl():
    from touhou_rl.v400.student import teacher_bc

    scores = _scores_batch(4)
    for tau in (1.0, 2.0, 4.0):
        probs = teacher_bc.teacher_soft_labels(scores, tau)
        assert probs.shape == scores.shape
        assert np.allclose(probs.sum(axis=-1), 1.0)
        assert np.all(probs >= 0.0)
        # tau only reshapes the distribution: the argmax is tau-invariant
        assert np.array_equal(probs.argmax(axis=-1), scores.argmax(axis=-1))
        assert np.all(probs.max(axis=-1) > 1.0 / obs_spec.N_MOVE_CLASSES)
    flat = teacher_bc.teacher_soft_labels(scores, 1e6)
    assert np.allclose(flat, 1.0 / obs_spec.N_MOVE_CLASSES, atol=1e-9)
    with pytest.raises(ValueError):
        teacher_bc.teacher_soft_labels(scores, 0.0)
    with pytest.raises(ValueError):
        teacher_bc.teacher_soft_labels(np.zeros((2, 17)), 1.0)
    # the mask is diagnostics only: passing it may not change the target
    mask = np.zeros_like(scores, dtype=bool)
    mask[:, 7] = True
    assert np.allclose(teacher_bc.teacher_soft_labels(scores, 2.0, verified_mask=mask),
                       teacher_bc.teacher_soft_labels(scores, 2.0))
    with pytest.raises(ValueError):
        teacher_bc.teacher_soft_labels(scores, 2.0, verified_mask=mask[:, :5])

    probs = teacher_bc.teacher_soft_labels(scores, 2.0)
    logp = np.log(probs)
    assert np.allclose(teacher_bc.kl_teacher_student(probs, logp), 0.0, atol=1e-12)
    other = teacher_bc.teacher_soft_labels(scores, 1.0)
    forward = teacher_bc.kl_teacher_student(probs, np.log(other))
    backward = teacher_bc.kl_teacher_student(other, np.log(probs))
    assert np.all(forward > 0)
    assert not np.allclose(forward, backward)
    with pytest.raises(ValueError):
        teacher_bc.kl_teacher_student(probs, np.full_like(logp, 0.5))  # positive log-prob


def test_kl_matches_torch_kl_div():
    """The numpy teacher direction is exactly F.kl_div(log pi_s, pi_t)."""

    torch = pytest.importorskip("torch")
    from touhou_rl.v400.student import teacher_bc

    scores = _scores_batch(6)
    teacher = teacher_bc.teacher_soft_labels(scores, 2.0)
    student_probs = teacher_bc.teacher_soft_labels(scores, 0.5)
    student_log = np.log(student_probs)
    expected = teacher_bc.kl_teacher_student(teacher, student_log).mean()
    got = torch.nn.functional.kl_div(
        torch.from_numpy(student_log), torch.from_numpy(teacher), reduction="batchmean")
    assert np.isclose(float(got), expected, atol=1e-10)


def test_tau_selection_ranks_top1_not_kl():
    from touhou_rl.v400.student import teacher_bc

    candidates = {
        1.0: {"val_top1": 0.41, "val_kl": 0.20, "val_ce_argmax": 2.10, "val_bomb_acc": 1.0},
        2.0: {"val_top1": 0.47, "val_kl": 0.55, "val_ce_argmax": 1.95, "val_bomb_acc": 1.0},
        4.0: {"val_top1": 0.47, "val_kl": 0.90, "val_ce_argmax": 2.20, "val_bomb_acc": 1.0},
    }
    tau, rationale = teacher_bc.select_tau(candidates)
    # best top-1 wins; the tie is broken by the smaller NLL of the teacher argmax,
    # *not* by KL (tau=4 would win on KL for free because H(pi_t) grows with tau)
    assert tau == 2.0
    assert rationale["selected_tau"] == 2.0
    assert len(rationale["ranking"]) == 3
    assert {row["tau"] for row in rationale["candidates"]} == {1.0, 2.0, 4.0}
    assert "entropy" in rationale["kl_not_used_for_ranking"]
    assert teacher_bc.select_tau({4.0: candidates[1.0]})[0] == 4.0
    with pytest.raises(ValueError):
        teacher_bc.select_tau({})
    with pytest.raises(ValueError):
        teacher_bc.select_tau({1.0: {"val_top1": 0.4}})


def test_shard_split_is_episode_isolated():
    from touhou_rl.v400.student import teacher_bc

    keys = [f"dual-s0d1seed{i}" for i in range(1, 16)]
    assignments = teacher_bc.assign_shard_splits(keys, seed=0)
    assert set(assignments) == set(keys)
    counts = {name: sum(1 for v in assignments.values() if v == name)
              for name in teacher_bc.SPLIT_NAMES}
    assert counts == {"train": 11, "val": 2, "test": 2}
    assert teacher_bc.assign_shard_splits(keys, seed=0) == assignments
    assert teacher_bc.assign_shard_splits(keys, seed=1) != assignments
    seeds = {key: int(key.rsplit("seed", 1)[1]) for key in keys}
    decisions = {key: 3200 for key in keys}
    report = teacher_bc.split_report(assignments, decisions=decisions, seeds=seeds)
    seen: list[int] = []
    for name in teacher_bc.SPLIT_NAMES:
        seen.extend(report[name]["seeds"])
        assert report[name]["shard_count"] == counts[name]
        assert report[name]["decisions"] == 3200 * counts[name]
    assert len(seen) == len(set(seen)) == len(keys)
    assert report["total_decisions"] == 3200 * len(keys)
    assert "no shard and no seed crosses a split" in report["isolation"]
    assert teacher_bc.split_signature(assignments) == teacher_bc.split_signature(
        dict(reversed(list(assignments.items()))))

    with pytest.raises(teacher_bc.SplitError):
        teacher_bc.assign_shard_splits([])
    with pytest.raises(teacher_bc.SplitError):
        teacher_bc.assign_shard_splits(["a", "a", "b", "c"])
    with pytest.raises(teacher_bc.SplitError):
        teacher_bc.assign_shard_splits(["only-one", "two"])
    # one seed may not span two splits
    leaky = dict(assignments)
    train_key = next(k for k, v in leaky.items() if v == "train")
    test_key = next(k for k, v in leaky.items() if v == "test")
    seeds[test_key] = seeds[train_key]
    with pytest.raises(teacher_bc.SplitError):
        teacher_bc.split_report(leaky, decisions=decisions, seeds=seeds)


def test_bomb_pos_weight_and_unidentifiable_case():
    from touhou_rl.v400.student import teacher_bc

    none = teacher_bc.bomb_pos_weight(np.zeros(50, dtype=np.int8))
    assert none["positives"] == 0
    assert none["pos_weight"] == 1.0
    assert none["identifiable"] is False
    assert "no positive example" in none["note"]
    rare = teacher_bc.bomb_pos_weight(np.array([0] * 99 + [1], dtype=np.int8))
    assert rare["positives"] == 1
    assert np.isclose(rare["pos_weight"], 99.0)
    assert rare["identifiable"] is True
    clamped = teacher_bc.bomb_pos_weight(np.array([0] * 1000 + [1]), max_weight=100.0)
    assert clamped["pos_weight"] == 100.0
    with pytest.raises(ValueError):
        teacher_bc.bomb_pos_weight(np.array([0, 2]))


def _bc_train_block() -> dict:
    return {
        "bc_version": "bc-v1-teacher-soft",
        "tau": 2.0,
        "d0": {"samples": 50000, "obs_spec_digest": obs_spec.obs_spec_digest()},
        "splits": {"train": {"decisions": 35000}, "val": {"decisions": 7500},
                   "test": {"decisions": 7500}, "signature": "cafe"},
        "metrics": {"val": {"kl": 0.4, "top1": 0.5}, "test": {"kl": 0.42, "top1": 0.49}},
        "disclosure": "BC v1 is not a G3 pass",
    }


def test_checkpoint_summary_verifies_version_and_digest(tmp_path):
    torch = pytest.importorskip("torch")
    from touhou_rl.v400.student import teacher_bc

    net = model.StudentNet(width=16, hidden=32)
    path = model.save_checkpoint(tmp_path / "bc1.pt", net, train=_bc_train_block(),
                                 provenance={"unit": "test"})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    summary = teacher_bc.checkpoint_summary(payload)
    assert summary["parameters"] == net.n_parameters()
    assert summary["tau"] == 2.0
    assert summary["bc_version"] == "bc-v1-teacher-soft"
    assert summary["obs_spec_digest"] == obs_spec.obs_spec_digest()
    assert teacher_bc.checkpoint_summary(
        payload, expected_parameters=net.n_parameters())["parameters"] == net.n_parameters()

    with pytest.raises(ValueError):
        teacher_bc.checkpoint_summary(payload, expected_parameters=net.n_parameters() + 1)
    corrupt = dict(payload)
    corrupt["schema"] = "v400-student-checkpoint-0"
    with pytest.raises(ValueError, match="schema"):
        teacher_bc.checkpoint_summary(corrupt)
    corrupt = dict(payload)
    corrupt["parameters"] = payload["parameters"] + 1
    with pytest.raises(ValueError, match="parameters"):
        teacher_bc.checkpoint_summary(corrupt)
    corrupt = dict(payload)
    corrupt["obs_spec_digest"] = "0" * 64
    with pytest.raises(obs_spec.ObsSpecMismatch):
        teacher_bc.checkpoint_summary(corrupt)
    missing = dict(payload)
    train = dict(_bc_train_block())
    train.pop("tau")
    missing["train"] = train
    with pytest.raises(ValueError, match="tau"):
        teacher_bc.checkpoint_summary(missing)
    wrong_version = dict(payload)
    train = dict(_bc_train_block())
    train["bc_version"] = "bc-v0"
    wrong_version["train"] = train
    with pytest.raises(ValueError, match="bc_version"):
        teacher_bc.checkpoint_summary(wrong_version)


def test_eval_schema_death_stop_stage_clear_and_parser():
    path = REPO_ROOT / "tools" / "rl" / "v400_student_eval.py"
    spec = importlib.util.spec_from_file_location("v400_student_eval_schema", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # S1N is RESET stage 0: the harness stage argument is 0-based (Stage1).
    assert module.DEFAULT_STAGE == 0
    parser = module.build_parser()
    args = parser.parse_args(["--engine", "e", "--data-dir", "d", "--checkpoint", "c"])
    assert args.stage == 0
    assert args.stop_on_death is False
    assert args.seeds == "1000,1001,1002,1003,1004"
    flagged = parser.parse_args(["--engine", "e", "--data-dir", "d", "--checkpoint", "c",
                                "--stage", "0", "--stop-on-death",
                                "--max-frames", "19887"])
    assert flagged.stop_on_death is True
    assert flagged.max_frames == 19887
    assert "death_stop" in module.VALID_ENDS
    assert "stage_clear" in module.VALID_ENDS

    def record(end: str) -> dict:
        return {
            "schema": module.EPISODE_SCHEMA,
            "checkpoint": "c.pt",
            "checkpoint_sha256": "a" * 64,
            "obs_spec_version": obs_spec.OBS_SPEC_VERSION,
            "obs_spec_digest": obs_spec.obs_spec_digest(),
            "seed": 1000,
            "suite": "S1N",
            "stage": 0,
            "difficulty": "normal",
            "max_frames": 19887,
            "frames": 321,
            "deaths": 1.0,
            "score": 1234,
            "end": end,
            "privilege": "deploy_pixel",
            "wall_s": 1.0,
            "engine_sha256": "b" * 64,
            "policy_input": "play64 4-frame stack + hud strip; no privileged state",
            "not_a_gate": True,
            "stage_start": 0,
            "stage_end": 0,
            "cleared": False,
            "exit": "controller",
            "stop_on_death": True,
        }

    module.validate_episode_record(record("death_stop"))
    cleared = record("stage_clear")
    cleared.update(stage_end=1, cleared=True)
    module.validate_episode_record(cleared)
    with pytest.raises(module.EvalError, match="cleared"):
        bad = record("death_stop")
        bad["cleared"] = True
        module.validate_episode_record(bad)
    with pytest.raises(module.EvalError, match="exit"):
        bad = record("death_stop")
        bad["exit"] = "timeout"
        module.validate_episode_record(bad)
    with pytest.raises(module.EvalError, match="stage_start"):
        bad = record("death_stop")
        bad["stage_start"] = "zero"
        module.validate_episode_record(bad)
    with pytest.raises(module.EvalError, match="end"):
        module.validate_episode_record(record("cleared"))
