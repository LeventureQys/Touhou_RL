"""CPU-only Phase 1 diagnostic capture contract tests; never start the game."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from touhou_rl import obs, spec
from touhou_rl.collect_student_states import StudentRolloutConfig, collect_student_states
from touhou_rl.pipeline.collect import CollectSpec


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(depth=3, region_sizes={
            "playfield": (224, 192), "hud_score": (34, 216), "hud_status": (126, 216),
        })
        self.calls = []
        self.resets = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def configure_temporal_policy(self, **kwargs):
        assert kwargs == {"ema": 0.0, "switch_penalty": 0.0}

    def reset_temporal_policy(self):
        self.resets += 1

    def act(self, regions):
        assert set(regions) == set(self.config.region_sizes)
        assert all(value.shape[1] == 3 and value.dtype == torch.float32 for value in regions.values())
        self.calls.append(regions)
        return torch.tensor([[3, 1, 0, 0]]), None, None, None


class Reader:
    def __init__(self, env):
        self.env = env

    def read_state(self, **kwargs):
        return SimpleNamespace(stage=0, gameplay_frame=self.env.frame)


class FakeEnv:
    def __init__(self, config):
        self.config = config
        self.session = SimpleNamespace(reader=Reader(self))
        self.frame = 2
        self.actions = []
        self.render_count = 0
        self.closed = False

    def reset(self, *, seed=None):
        self.frame = 2
        return self.observation(), {"state": self.state()}

    def state(self):
        return dict(stage=0, frame=self.frame, score=0, graze=0, lives=3,
                    bombs=0, power=0, deaths=0)

    def observation(self):
        return dict(self_feat=np.zeros(obs.N_SELF, dtype=np.float32),
                    bullets=np.zeros((obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float32),
                    danger=np.zeros((obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8))

    def render(self):
        self.render_count += 1
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def step(self, action):
        self.actions.append(np.asarray(action).tolist())
        self.frame += 2
        return self.observation(), 0.0, True, False, {
            **self.state(), "termination_reason": "game over",
            "episode": {"frames": self.frame, "deaths": 1},
        }

    def close(self):
        self.closed = True


def test_student_round_pixel_isolation_and_auditable_shard(tmp_path, monkeypatch):
    from touhou_rl import collect_student_states as collector
    checkpoint = tmp_path / "frozen.pt"
    checkpoint.write_bytes(b"fake-checkpoint")
    model = FakeModel()
    monkeypatch.setattr(collector, "load_vision_checkpoint", lambda *a, **kw: (model, {}))
    monkeypatch.setattr(collector, "vision_arch", lambda _: "v3-st")
    envs = []
    def factory(config):
        envs.append(FakeEnv(config))
        return envs[-1]
    output = tmp_path / "r1-student-t0"
    report = collect_student_states(StudentRolloutConfig(
        checkpoint=checkpoint, out_dir=output, episodes=2, max_steps=1,
        shard_size=1, device="cpu", env_factory=factory,
    ))
    env = envs[0]
    assert env.closed and env.config.expose_raw_bullets and env.config.expose_bosses
    assert env.actions == [[3, 1, 0, 0]] * 2
    assert env.render_count == 4  # same initial double render as evaluate
    assert model.resets == 2  # one reset per episode
    assert len(model.calls) == 2
    assert report["training_approved"] is False and report["samples"] == 2
    with np.load(output / "student-state-00000.npz") as data:
        assert data["executed_action"].tolist() == [[3, 1, 0, 0]]
        assert data["decision_meta"].tolist() == [[1, 0, 2, 0]]
        assert data["action"].shape == (1, 4)
    trace = [json.loads(line) for line in (output / "episode-001-decisions.jsonl").read_text().splitlines()]
    assert trace[0]["sample_index"] == 0 and trace[0]["aligned"]
    snapshot = trace[0]["raw_safety_snapshot"]
    assert snapshot["schema"] == "v200-raw-safety-snapshot/1"
    assert snapshot["world_complete"] is False
    assert snapshot["invalid_reason"] == "raw-pool-or-player-missing"
    assert "bullets" not in snapshot
    assert trace[0]["action_hold_frames_valid"] is False
    assert trace[0]["timing_contract"] == "v200-rollout-timing/1"
    assert trace[0]["timing"]["action_effective"] is None
    assert trace[0]["actual_hold_frames"] is None
    assert trace[0]["death_window"] is False  # fake step does not increment state.deaths
    assert report["episodes"][0]["trace_sha256"]
    with pytest.raises(FileExistsError):
        collect_student_states(StudentRolloutConfig(checkpoint=checkpoint, out_dir=output,
                                                   device="cpu", env_factory=factory))


def test_death_window_patch_across_shards(tmp_path):
    from touhou_rl.collect_student_states import _patch_death_windows
    from touhou_rl.vision_data import VisionShardWriter, META_FIELDS
    from touhou_rl import vision
    out = tmp_path / "round"
    writer = VisionShardWriter(out, shard_size=1, tag="student-state")
    crops = vision.extract_regions(np.zeros((480, 640, 3), dtype=np.uint8))
    blobs = {name: vision.encode_png(image) for name, image in crops.items()}
    observation = FakeEnv(None).observation()
    from touhou_rl.collect_student_states import _label_observation
    for index in range(3):
        meta = {field: 0 for field in META_FIELDS}
        meta.update(episode=1, decision=index, frame_delta=2, stage_delta=0,
                    executed_action=[3, 1, 0, 0], death_window=False, death_window_valid=False)
        writer.add(blobs, _label_observation(observation), (3, 1, 0, 0), meta)
    summary = writer.close()
    _patch_death_windows(out, summary["shards"], [False, True, True], [False, True, True])
    assert len(summary["shards"]) == 3
    for index, name in enumerate(summary["shards"]):
        with np.load(out / name) as data:
            assert data["death_window"].tolist() == [[0], [1], [1]][index]
            assert data["death_window_valid"].tolist() == [[0], [1], [1]][index]


def test_optional_metadata_preflight_does_not_corrupt_shard(tmp_path):
    from touhou_rl.vision_data import VisionShardWriter, META_FIELDS
    from touhou_rl import vision
    from touhou_rl.collect_student_states import _label_observation
    writer = VisionShardWriter(tmp_path, shard_size=3, tag="t")
    crops = vision.extract_regions(np.zeros((480, 640, 3), dtype=np.uint8))
    blobs = {name: vision.encode_png(image) for name, image in crops.items()}
    obs_dict = FakeEnv(None).observation()
    basic = {field: 0 for field in META_FIELDS}
    writer.add(blobs, _label_observation(obs_dict), (0, 1, 0, 0), basic)
    with pytest.raises(ValueError, match="schema"):
        writer.add(blobs, _label_observation(obs_dict), (0, 1, 0, 0),
                   {**basic, "executed_action": [3, 1, 0, 0]})
    assert writer.total == 1
    with pytest.raises(ValueError, match="requires"):
        writer.add(blobs, _label_observation(obs_dict), (0, 1, 0, 0), {**basic, "episode": 1})
    summary = writer.close()
    with np.load(tmp_path / summary["shards"][0]) as data:
        assert len(data["playfield_offset"]) == 2
        assert data["action"].shape == (1, 4)


def test_pipeline_requires_checkpoint_and_never_uses_replay_fallback(tmp_path):
    spec = CollectSpec(modality="vision", source="teacher", out_dir=tmp_path / "r1")
    with pytest.raises(ValueError, match="extra.checkpoint"):
        spec.validate()
    spec.extra["checkpoint"] = "frozen.pt"
    spec.validate()
