"""Contract tests for the step-2 privileged policy (entity observation + 18-way head).

Nothing here starts the game: the observation is built from synthetic snapshots,
and ``eval18.run_evaluation`` is exercised through its ``env_factory`` stub.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from touhou_rl import bc18, entity_obs, entity_policy, eval18, spec


# --------------------------------------------------------------------------- #
# entity observation
# --------------------------------------------------------------------------- #


def _world(bullets: int = 3) -> dict:
    positions = [[100.0 + 10.0 * i, 200.0 + 5.0 * i] for i in range(bullets)]
    velocities = [[0.0, 2.0] for _ in range(bullets)]
    halves = [2.0 for _ in range(bullets)]
    return {
        "player_position": [192.0, 384.0],
        "player_half": [0.825, 0.825],
        "stage": 0,
        "bullets": {"position": positions, "velocity": velocities, "half": halves},
    }


def test_observation_shapes_and_mask():
    observation = entity_obs.from_snapshot(_world(3))
    assert observation["bullets"].shape == (entity_obs.K_BULLETS, entity_obs.BULLET_FEATURES)
    assert observation["bullets"].dtype == np.float32
    assert observation["meta"].shape == (entity_obs.META_DIM,)
    # exactly three valid rows, the rest padded to zero
    assert observation["bullets"][:, 5].sum() == 3.0
    assert np.all(observation["bullets"][3:] == 0.0)
    # normalisation matches obs.py; slot 0 is the *nearest* bullet, (120, 210) here
    assert observation["bullets"][0, 0] == pytest.approx((120.0 - 192.0) / 192.0, rel=1e-6)
    assert observation["bullets"][0, 1] == pytest.approx((210.0 - 384.0) / 224.0, rel=1e-6)
    assert observation["bullets"][0, 2] == pytest.approx(0.0)
    assert observation["bullets"][0, 3] == pytest.approx(2.0 / 6.0, rel=1e-6)
    assert observation["bullets"][0, 4] == pytest.approx(2.0 / 8.0, rel=1e-6)
    assert observation["meta"][0] == pytest.approx((192.0 - 32.0) / 384.0, rel=1e-6)
    assert observation["meta"][1] == pytest.approx((384.0 - 16.0) / 448.0, rel=1e-6)


def test_observation_sorts_by_distance_and_truncates_at_k():
    positions = [[192.0 + 400.0, 384.0], [192.0 + 5.0, 384.0], [192.0 - 50.0, 384.0]]
    snapshot = _world(0)
    snapshot["bullets"] = {
        "position": positions,
        "velocity": [[0.0, 0.0]] * 3,
        "half": [1.0, 1.0, 1.0],
    }
    observation = entity_obs.from_snapshot(snapshot)
    dx = observation["bullets"][:3, 0] * 192.0
    assert list(np.round(dx, 3)) == [5.0, -50.0, 400.0]  # nearest first

    many = [[192.0 + float(i), 384.0] for i in range(entity_obs.K_BULLETS + 5)]
    snapshot["bullets"] = {"position": many, "velocity": [[0.0, 0.0]] * len(many), "half": [1.0] * len(many)}
    observation = entity_obs.from_snapshot(snapshot)
    assert observation["bullets"][:, 5].sum() == float(entity_obs.K_BULLETS)
    assert observation["meta"][3] == pytest.approx((entity_obs.K_BULLETS + 5) / 700.0, rel=1e-6)


def test_empty_pool_is_all_zero_but_keeps_the_player_context():
    snapshot = _world(0)
    snapshot["bullets"] = {"position": [], "velocity": [], "half": []}
    observation = entity_obs.from_snapshot(snapshot)
    assert np.all(observation["bullets"] == 0.0)
    assert observation["meta"][0] == pytest.approx((192.0 - 32.0) / 384.0, rel=1e-6)


def test_snapshot_and_env_paths_agree():
    snapshot = _world(4)
    from_snapshot = entity_obs.from_snapshot(snapshot)
    # The live path recovers the player position from self_feat[0:2].
    self_feat = np.zeros(24, dtype=np.float32)
    self_feat[0] = (192.0 - spec.PLAYFIELD_X) / spec.PLAYFIELD_W
    self_feat[1] = (384.0 - spec.PLAYFIELD_Y) / spec.PLAYFIELD_H
    info = {
        "bullets_raw": {
            "position": np.asarray(snapshot["bullets"]["position"], dtype=np.float32),
            "velocity": np.asarray(snapshot["bullets"]["velocity"], dtype=np.float32),
            "half": np.asarray(snapshot["bullets"]["half"], dtype=np.float32),
        },
        "stage": 0,
    }
    from_env = entity_obs.from_env(info, {"self_feat": self_feat})
    assert np.allclose(from_env["bullets"], from_snapshot["bullets"], atol=1e-5)
    assert np.allclose(from_env["meta"], from_snapshot["meta"], atol=1e-5)


def test_mismatched_bullet_arrays_are_rejected():
    with pytest.raises(ValueError, match="disagree"):
        entity_obs.build(
            0.0, 0.0, np.zeros((2, 2)), np.zeros((3, 2)), np.zeros((2,))
        )


def test_from_env_requires_the_raw_pool():
    with pytest.raises(ValueError, match="bullets_raw"):
        entity_obs.from_env({"stage": 0}, {"self_feat": np.zeros(24, dtype=np.float32)})


# --------------------------------------------------------------------------- #
# candidate mapping
# --------------------------------------------------------------------------- #


def test_candidate_round_trip_and_env_action():
    for focus in (0, 1):
        for move in range(9):
            candidate = entity_obs.candidate_index(move, focus)
            assert candidate == focus * 9 + move
            assert entity_obs.action_of_candidate(candidate) == (move, 1, focus, 0)
    assert entity_obs.action_of_candidate(0) == (0, 1, 0, 0)  # idle, fire, no focus
    assert entity_obs.action_of_candidate(17) == (8, 1, 1, 0)  # NW, fire, focused
    with pytest.raises(ValueError):
        entity_obs.action_of_candidate(18)
    with pytest.raises(ValueError):
        entity_obs.candidate_index(9, 0)


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #


def _batch(n: int = 2, valid: int = 5) -> dict[str, torch.Tensor]:
    bullets = torch.zeros(n, entity_obs.K_BULLETS, entity_obs.BULLET_FEATURES)
    bullets[:, :valid, 0] = torch.linspace(-0.5, 0.5, valid)
    bullets[:, :valid, 1] = torch.linspace(0.2, -0.2, valid)
    bullets[:, :valid, 5] = 1.0
    meta = torch.zeros(n, entity_obs.META_DIM)
    grid = torch.zeros(n, entity_obs.GRID_CHANNELS, entity_obs.GRID_ROWS, entity_obs.GRID_COLS)
    return {"bullets": bullets, "meta": meta, "grid": grid}


def test_policy_shapes_and_finite_logprobs():
    model = entity_policy.EntityPolicy(entity_policy.EntityPolicyConfig(trunk_hidden=32, object_hidden=16))
    logits, value = model(_batch())
    assert logits.shape == (2, entity_policy.N_CANDIDATES)
    assert value.shape == (2,)
    candidate, logp, value, entropy = model.act(_batch())
    assert candidate.shape == (2,) and candidate.dtype == torch.int64
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
    assert float(entropy[0].detach()) <= np.log(entity_policy.N_CANDIDATES) + 1e-5


def test_permuting_the_set_does_not_change_the_output():
    torch.manual_seed(3)
    model = entity_policy.EntityPolicy(entity_policy.EntityPolicyConfig(trunk_hidden=32, object_hidden=16))
    model.eval()
    batch = _batch(valid=6)
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    shuffled = dict(batch)
    shuffled["bullets"] = batch["bullets"].clone()
    shuffled["bullets"][:, :6] = batch["bullets"][:, permutation]
    with torch.no_grad():
        reference, _ = model(batch)
        reordered, _ = model(shuffled)
    assert torch.allclose(reference, reordered, atol=1e-6)


def test_padded_rows_are_ignored():
    model = entity_policy.EntityPolicy(entity_policy.EntityPolicyConfig(trunk_hidden=32, object_hidden=16))
    model.eval()
    batch = _batch(valid=4)
    padded = dict(batch)
    padded["bullets"] = batch["bullets"].clone()
    # Adding zeroed, invalid rows must not change anything (they are masked out).
    padded["bullets"][:, 10:40, :] = 0.0
    with torch.no_grad():
        reference, _ = model(batch)
        with_padding, _ = model(padded)
    assert torch.allclose(reference, with_padding, atol=1e-6)


def test_checkpoint_round_trip_and_candidate_order_guard(tmp_path):
    model = entity_policy.EntityPolicy(entity_policy.EntityPolicyConfig(trunk_hidden=32, object_hidden=16))
    path = entity_policy.save_checkpoint(tmp_path / "p.pt", model, extra={"note": "test"})
    restored, extra = entity_policy.load_checkpoint(path)
    assert extra == {"note": "test"}
    assert restored.config.trunk_hidden == 32
    with torch.no_grad():
        assert torch.allclose(model(_batch())[0], restored(_batch())[0], atol=1e-6)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["candidate_order"] = "move*2+focus"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="candidate order mismatch"):
        entity_policy.load_checkpoint(path)


# --------------------------------------------------------------------------- #
# behaviour cloning data path
# --------------------------------------------------------------------------- #


def _trace(root: Path, group: str, rows: int) -> Path:
    directory = root / group
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "episode-001-decisions.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for index in range(rows):
            snapshot = _world(2)
            snapshot["player_position"] = [100.0 + index, 300.0]
            handle.write(
                json.dumps(
                    {
                        "executed_action": [index % 9, 1, 0, 0],
                        "teacher_action": [index % 9, 1, (index // 9) % 2, 0],
                        "raw_safety_snapshot": snapshot,
                    }
                )
                + "\n"
            )
    return path


def test_build_cache_and_group_split(tmp_path):
    root = tmp_path / "rollouts"
    for group in ("a-student", "b-student", "c-student"):
        _trace(root, group, 4)
    cache = tmp_path / "cache.npz"
    summary = bc18.build_cache(root, cache, progress=lambda *_: None)
    assert summary["decisions"] == 12
    data = bc18.load_cache(cache)
    assert data["bullets"].shape == (12, entity_obs.K_BULLETS, entity_obs.BULLET_FEATURES)
    assert data["bullets"].dtype == np.float16
    assert data["labels"].min() >= 0 and data["labels"].max() < 18
    train_index, validation_index = bc18.split_by_group(data["groups"], validation_groups=1)
    assert train_index.size == 8 and validation_index.size == 4


def test_class_weights_damp_the_majority_class():
    counts = np.array([31735, 3011, 1843, 7123, 2761, 6730, 6603, 9218, 1800, 7, 163, 163, 206, 152, 118, 169, 178, 132])
    weights = bc18.class_weights(counts, "inv-sqrt")
    assert weights is not None and weights.shape == (18,)
    assert weights.mean() == pytest.approx(1.0, rel=1e-6)
    assert weights[0] < weights[3] < 1.0 < weights[9]
    assert bc18.class_weights(counts, "none") is None
    with pytest.raises(ValueError, match="unknown class weighting"):
        bc18.class_weights(counts, "sqrt")


def test_label_extraction_matches_the_candidate_order():
    actions = np.array([[3, 1, 0, 0], [3, 1, 1, 0], [0, 1, 0, 0]])
    assert bc18.candidate_labels_from_actions(actions).tolist() == [3, 12, 0]
    with pytest.raises(ValueError):
        bc18.candidate_labels_from_actions(np.zeros((2, 2)))


def test_training_reduces_loss_on_a_separable_toy_set(tmp_path):
    root = tmp_path / "rollouts"
    for group in ("a-student", "b-student", "c-student"):
        _trace(root, group, 6)
    cache = tmp_path / "cache.npz"
    bc18.build_cache(root, cache, progress=lambda *_: None)
    summary = bc18.train(
        cache,
        tmp_path / "bc",
        epochs=3,
        batch_size=4,
        learning_rate=3e-3,
        device="cpu",
        seed=1,
        progress=lambda *_: None,
    )
    assert summary["history"][-1]["train_loss"] <= summary["history"][0]["train_loss"]
    assert Path(summary["checkpoint"]).exists()
    written = json.loads((tmp_path / "bc" / "summary.json").read_text(encoding="utf-8"))
    assert written["best"]["accuracy"] >= 0.0


# --------------------------------------------------------------------------- #
# closed-loop evaluation (stub environment)
# --------------------------------------------------------------------------- #


class _StubEnv:
    action_space = SimpleNamespace(sample=lambda: np.zeros(4, dtype=np.int64))

    def __init__(self, steps_to_end: int = 3, bullets: int = 4):
        self.steps_to_end = steps_to_end
        self.bullets = bullets
        self.actions: list[np.ndarray] = []
        self.closed = False
        self._step = 0

    def reset(self):
        observation = {"self_feat": np.zeros(24, dtype=np.float32)}
        observation["self_feat"][0] = 0.5
        observation["self_feat"][1] = 0.8
        return observation, {"state": {"bombs": 2}, "stage": 0, "stages_reached": 0}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.int64).copy())
        self._step += 1
        done = self._step >= self.steps_to_end
        positions = np.tile(np.array([[200.0, 200.0]], dtype=np.float32), (self.bullets, 1))
        info = {
            "episode": {
                "frames": 2 * self._step,
                "deaths": 0,
                "lives": 2.0,
                "terminated": done,
                "truncated": False,
                "termination_reason": "frame stalled" if done else "",
                "stages_reached": 0,
                "score": 1234,
                "graze": 7,
                "episode_reward": 0.5,
            },
            "stages_reached": 0,
            "stage": 0,
            "frame": 2 * self._step,
            "bullets": self.bullets,
            "bullets_raw": {
                "position": positions,
                "velocity": np.zeros((self.bullets, 2), dtype=np.float32),
                "half": np.ones((self.bullets,), dtype=np.float32),
            },
        }
        observation = {"self_feat": np.zeros(24, dtype=np.float32)}
        return observation, 0.0, done, False, info

    def close(self):
        self.closed = True


def test_eval18_writes_compareable_rows(tmp_path):
    model = entity_policy.EntityPolicy(entity_policy.EntityPolicyConfig(trunk_hidden=32, object_hidden=16))
    checkpoint = entity_policy.save_checkpoint(tmp_path / "p.pt", model)
    env = _StubEnv(steps_to_end=3, bullets=entity_obs.K_BULLETS + 12)
    report = eval18.run_evaluation(
        eval18.Eval18Config(
            checkpoint=checkpoint,
            out_dir=tmp_path / "out",
            episodes=1,
            device="cpu",
            decision_trace=True,
            env_factory=lambda config: env,
        )
    )
    assert report["policy"] == "privileged-entity"
    assert report["summary"]["frames_mean"] == 6.0
    row = report["episodes"][0]
    assert row["termination_reason"] == "frame stalled"
    assert row["score"] == 1234
    # every action is a legal 18-way candidate mapping
    assert all(action[1] == 1 and action[3] == 0 for action in row["decision_trace"]["actions"])
    assert row["action_diagnostics"]["decisions"] == 3
    # the truncation evidence is recorded
    assert row["active_bullets"]["over_cap"] == 2  # decisions 1 and 2 saw the full pool
    assert env.closed is True
    written = json.loads((tmp_path / "out" / "eval-report.json").read_text(encoding="utf-8"))
    assert written["candidate_order"] == entity_policy.CANDIDATE_ORDER
    assert len(written["checkpoint_sha256"]) == 64


def test_multiple_roots_merge_with_prefixed_groups(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _trace(root_a, "shared-name", 2)
    _trace(root_b, "shared-name", 2)
    summary = bc18.build_cache([root_a, root_b], tmp_path / "cache.npz", progress=lambda *_: None)
    assert summary["decisions"] == 4
    assert summary["traces"] == 2
    data = bc18.load_cache(tmp_path / "cache.npz")
    assert sorted(set(int(g) for g in data["groups"])) == [0, 1]
    with pytest.raises(ValueError, match="does not exist"):
        bc18.build_cache([tmp_path / "missing"], tmp_path / "x.npz", progress=lambda *_: None)
