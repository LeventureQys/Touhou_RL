"""Tests for the vision deployment path in ``touhou_rl/evaluate.py`` (S3-4).

Three things are pinned here, each of which was broken or unprovable before:

* **Architecture compatibility** - a v1 checkpoint (payload without ``arch``, as
  written before v1.0.3) and a v2 checkpoint both deploy through the same entry
  point, and the report labels them from the *loaded model type*, not from the
  payload.
* **Depth follows the checkpoint** - the first convolution takes ``3 * depth``
  channels, so the frame stack depth must be ``model.config.depth``.  The old
  default of 3 crashed on the depth=1 pilot model with a channel-count
  ``RuntimeError``; ``0`` now means "follow the checkpoint" and an explicit
  conflicting value exits instead of being silently overridden.
* **Isolation is testable** (design doc section 8.1) - I-1: the policy's only
  argument is ``regions_to_batch(env.render() crops)``; I-2: the vision branch
  touches no privileged field (the grep evidence lives in the completion report,
  the ``state`` counter below is the runtime counterpart); I-3: a poison object
  in ``info`` raises on any attribute access yet neither crashes the run nor
  changes a single action.

No test starts the game: ``EvalConfig.env_factory`` injects a stub environment,
so every run is CPU-only.  The models are tiny (width 4-8) because what is under
test is the plumbing, not the policy quality.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from touhou_rl import vision
from touhou_rl.evaluate import EvalConfig, run_evaluation
from touhou_rl.vmodels import (
    VisionNetConfig,
    VisionScorer,
    save_vision_checkpoint,
)
from touhou_rl.vmodels2 import VisionNetConfigV2, VisionScorerV2

ROOT = Path(__file__).resolve().parents[2]
CLIENT_H, CLIENT_W = 480, 640  # TH08 client area, the screen the regions are cut from


def _load_tool(name: str):
    """Import ``tools/rl/<name>.py`` as a module (``tools`` is not a package)."""
    module_name = f"s34_tools_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "tools" / "rl" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


COMPARE = _load_tool("compare_evals")


# --------------------------------------------------------------------------- #
# stub environment
# --------------------------------------------------------------------------- #


class Poison:
    """A stand-in for any privileged field: touching it fails the test."""

    def __getattr__(self, name):  # pragma: no cover - only the raise matters
        raise AssertionError(f"privileged field accessed: {name}")


def _screen() -> np.ndarray:
    """Deterministic 32 px block pattern, large enough to contain every region.

    Blocks rather than noise so the crop/resize path is exercised (each region
    gets a different subset of blocks) while the assertion stays exact: no
    dependence on how a resampler treats random data.
    """
    screen = np.zeros((CLIENT_H, CLIENT_W, 3), dtype=np.uint8)
    cells = itertools.product(range(0, CLIENT_H, 32), range(0, CLIENT_W, 32))
    for index, (row, column) in enumerate(cells):
        screen[row : row + 32, column : column + 32] = (
            (index * 37) % 256,
            (index * 91) % 256,
            (index * 53) % 256,
        )
    return screen


class StubEnv:
    """Minimal ``Th08Env``: fixed pixels out, fixed bookkeeping in.

    It exposes exactly what the deployment path is allowed to use
    (``reset``/``step``/``render``/``close``/``action_space``) plus a poisoned
    ``state`` that counts accesses, so a branch reaching for privileged data
    fails loudly instead of quietly working.
    """

    action_space = SimpleNamespace(sample=lambda: np.zeros(4, dtype=np.int64))

    def __init__(self, steps_to_end: int = 3, poison: bool = False, termination_reason: str = "game over"):
        self.steps_to_end = int(steps_to_end)
        self.poison = bool(poison)
        self.termination_reason = termination_reason
        self.env_config = None  # filled in by the factory
        self.actions: list[np.ndarray] = []
        self.renders = 0
        self.state_reads = 0
        self.closed = False
        self._step = 0
        self._screen = _screen()

    @property
    def state(self):
        """The game's memory view: the vision branch must never read it."""
        self.state_reads += 1
        raise AssertionError("vision branch read env.state")

    def reset(self):
        info = {"state": {"bombs": 2}, "stages_reached": 0, "stage": 0, "frame": 0, "bullets": 0}
        return {"state": {"bombs": 2}}, info

    def render(self):
        self.renders += 1
        return self._screen

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.int64).copy())
        self._step += 1
        done = self._step >= self.steps_to_end
        privileged = Poison() if self.poison else np.zeros((0, 4), dtype=np.float32)
        info = {
            "episode": {
                "frames": 2 * self._step,
                "deaths": 0,
                "lives": 2.0,
                "terminated": done,
                "truncated": False,
                "termination_reason": self.termination_reason if done else "",
                "stages_reached": 0,
                "score": 0,
                "graze": 0,
                "episode_reward": 0.0,
            },
            "stages_reached": 0,
            "stage": 0,
            "frame": 2 * self._step,
            "bullets": 0,
            "bullets_raw": privileged,
            "bosses": privileged,
            "bombs": 2,
        }
        return {"state": {"bombs": 2}}, 0.0, done, False, info

    def close(self):
        self.closed = True


class StubFactory:
    """``env_factory`` that keeps every stub it handed out, for assertions."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.envs: list[StubEnv] = []
        self.configs: list = []

    def __call__(self, env_config):
        self.configs.append(env_config)
        env = StubEnv(**self.kwargs)
        env.env_config = env_config
        self.envs.append(env)
        return env


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _write_v2(path: Path, depth: int) -> Path:
    model = VisionScorerV2(
        VisionNetConfigV2(depth=depth, pf_width=8, hud_width=4, trunk_hidden=32, state_hidden=32, candidate_hidden=16)
    )
    save_vision_checkpoint(path, model)
    return path


def _write_v1(path: Path, depth: int, keep_arch: bool = True) -> Path:
    model = VisionScorer(
        VisionNetConfig(
            depth=depth, width=8, trunk_hidden=32, candidate_hidden=16, state_hidden=32
        )
    )
    save_vision_checkpoint(path, model)
    if not keep_arch:
        # Pre-v1.0.3 checkpoints carry no arch key at all; strip it to reproduce one.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.pop("arch", None)
        torch.save(payload, path)
    return path


def _run(tmp_path, factory, **overrides) -> dict:
    """One ``run_evaluation`` call against a stub env; writes into ``tmp_path``."""
    settings = dict(
        checkpoint=Path("."),
        out_dir=tmp_path / "out",
        episodes=1,
        device="cpu",
        screenshot=False,
        max_steps=10,
        env_factory=factory,
    )
    settings.update(overrides)
    return run_evaluation(EvalConfig(**settings))


# --------------------------------------------------------------------------- #
# report node / architecture labelling
# --------------------------------------------------------------------------- #


def test_vision_report_node_and_policy(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    factory = StubFactory()
    report = _run(tmp_path, factory, vision_checkpoint=checkpoint)

    assert report["policy"] == "vision"
    node = report["vision"]
    assert node["checkpoint"] == str(checkpoint)
    assert node["arch"] == "v2"
    assert node["depth"] == 1
    assert node["region_sizes"] == {
        "playfield": [224, 192],
        "hud_score": [34, 216],
        "hud_status": [126, 216],
    }
    assert node["frame_shape"] == {
        "playfield": [224, 192, vision.VISION_CHANNELS],
        "hud_score": [34, 216, vision.VISION_CHANNELS],
        "hud_status": [126, 216, vision.VISION_CHANNELS],
    }
    assert isinstance(node["isolation"], str) and node["isolation"].startswith("I-1")

    # Existing top-level fields survive untouched.
    assert report["checkpoint_extra"]["vision_checkpoint"] is True
    assert report["frameskip"] == 2
    assert report["summary"]["episodes"] == 1
    assert len(report["episodes"]) == 1
    row = report["episodes"][0]
    assert row["steps"] == 3
    assert row["frames_reported"] == 6
    assert row["termination_reason"] == "game over"
    assert factory.envs[0].closed is True


def test_vision_node_is_written_to_eval_report_json(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    out = tmp_path / "out"
    _run(tmp_path, StubFactory(), vision_checkpoint=checkpoint, out_dir=out)

    written = json.loads((out / "eval-report.json").read_text(encoding="utf-8"))
    assert written["vision"]["arch"] == "v2"
    assert written["vision"]["depth"] == 1
    assert written["policy"] == "vision"


def test_v1_checkpoint_labels_arch_v1(tmp_path):
    # No arch key in the payload - the label has to come from the model type.
    checkpoint = _write_v1(tmp_path / "v1.pt", depth=1, keep_arch=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert "arch" not in payload

    report = _run(tmp_path, StubFactory(), vision_checkpoint=checkpoint)
    assert report["policy"] == "vision"
    assert report["vision"]["arch"] == "v1"
    assert report["vision"]["depth"] == 1


def test_v2_checkpoint_labels_arch_v2(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    report = _run(tmp_path, StubFactory(), vision_checkpoint=checkpoint)
    assert report["vision"]["arch"] == "v2"


def test_no_vision_node_without_vision_checkpoint(tmp_path):
    report = _run(tmp_path, StubFactory(), random_policy=True)
    assert report["policy"] == "random"
    assert "vision" not in report
    assert report["checkpoint_extra"] == {"random_policy": True}
    assert report["episodes"][0]["steps"] == 3

    # The non-vision report shape is frozen: adding the vision node must not add,
    # drop or rename a single key on this path.
    assert set(report) == {
        "checkpoint",
        "checkpoint_extra",
        "policy",
        "difficulty",
        "team",
        "practice_stage",
        "frameskip",
        "deterministic",
        "device",
        "started",
        "episodes",
        "finished",
        "summary",
    }
    assert set(report["summary"]) == {"episodes", "frames_mean", "frames_max", "stages_max", "deaths_mean"}
    assert set(report["episodes"][0]) == {
        "episode",
        "steps",
        "frames_reported",
        "stages_reached",
        "deaths",
        "lives_left",
        "score",
        "graze",
        "episode_reward",
        "terminated",
        "truncated",
        "termination_reason",
        "wall_seconds",
        "decisions_per_s",
        "last_state",
    }


# --------------------------------------------------------------------------- #
# isolation: I-1 / I-2 / I-3
# --------------------------------------------------------------------------- #


def test_vision_act_receives_pixels_only(tmp_path, monkeypatch):
    """I-1: the single argument is ``regions_to_batch`` of ``env.render()`` crops."""
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    calls = []
    real_act = VisionScorerV2.act

    def spy(self, *args, **kwargs):
        calls.append((args, kwargs))
        return real_act(self, *args, **kwargs)

    monkeypatch.setattr(VisionScorerV2, "act", spy)
    factory = StubFactory(steps_to_end=2)
    report = _run(tmp_path, factory, vision_checkpoint=checkpoint)
    env = factory.envs[0]
    depth = report["vision"]["depth"]

    # One call per decision, and each call gets exactly one positional argument:
    # the batched regions.  No observation, no info, no privileged tensor, no
    # keyword argument that could smuggle one in.
    assert len(calls) == 2
    for args, kwargs in calls:
        assert len(args) == 1 and kwargs == {}
        assert isinstance(args[0], dict)
        assert set(args[0]) == {"playfield", "hud_score", "hud_status"}
    regions_seen = [args[0] for args, _ in calls]

    # The values are the pixels the stub rendered, region by region: re-derive
    # the expected crop stack here and compare, so "only pixels" is measured,
    # not assumed.
    native = vision.extract_regions(env.render())
    for name, (height, width) in report["vision"]["region_sizes"].items():
        expected = vision.resize_area(native[name], height, width)
        for regions in regions_seen:
            tensor = regions[name]
            assert tensor.dtype == torch.float32
            assert tuple(tensor.shape) == (1, depth, height, width, vision.VISION_CHANNELS)
            assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0
            for frame in range(depth):
                decoded = np.rint(tensor[0, frame].numpy() * 255.0).astype(np.uint8)
                assert np.array_equal(decoded, expected), f"{name} frame {frame} is not the render crop"


def test_policy_never_reads_env_state(tmp_path):
    """I-2 (runtime half): the stub's ``state`` property is never touched."""
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    factory = StubFactory()
    report = _run(tmp_path, factory, vision_checkpoint=checkpoint)
    env = factory.envs[0]

    assert env.state_reads == 0
    assert len(env.actions) == 3
    assert env.renders >= 3  # every decision came from a rendered frame
    assert report["vision"]["isolation"].startswith("I-1")


def test_policy_gets_only_pixels(tmp_path):
    """I-3: poison privileged fields in ``info``; the run must not raise."""
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    factory = StubFactory(steps_to_end=3, poison=True)
    report = _run(tmp_path, factory, vision_checkpoint=checkpoint)

    env = factory.envs[0]
    assert len(env.actions) == 3
    assert env.state_reads == 0
    assert report["episodes"][0]["steps"] == 3


def test_actions_independent_of_privileged_fields(tmp_path):
    """I-3: same pixels, poison vs real privileged fields -> identical actions."""
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    poisoned = StubFactory(steps_to_end=3, poison=True)
    plain = StubFactory(steps_to_end=3, poison=False)
    _run(tmp_path, poisoned, vision_checkpoint=checkpoint, out_dir=tmp_path / "poison")
    _run(tmp_path, plain, vision_checkpoint=checkpoint, out_dir=tmp_path / "plain")

    poison_actions = np.stack(poisoned.envs[0].actions)
    plain_actions = np.stack(plain.envs[0].actions)
    assert poison_actions.shape == (3, 4)
    assert np.array_equal(poison_actions, plain_actions)


def test_vision_env_is_not_asked_for_raw_bullets(tmp_path):
    """I-2 (wiring half): the vision branch does not even expose privileged data."""
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    factory = StubFactory()
    _run(tmp_path, factory, vision_checkpoint=checkpoint, frameskip=4, max_steps=77)

    config = factory.configs[0]
    assert config.expose_raw_bullets is False
    assert config.expose_bosses is False
    assert config.frameskip == 4
    assert config.max_steps == 77


# --------------------------------------------------------------------------- #
# depth follows the checkpoint (the live crash this version fixes)
# --------------------------------------------------------------------------- #


def test_vision_depth_follows_checkpoint(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    factory = StubFactory()
    # vision_depth=0 is the new default; a depth of 3 here would raise inside the
    # model's first conv (9 input channels against a 3-channel weight).
    report = _run(tmp_path, factory, vision_checkpoint=checkpoint, vision_depth=0)
    assert report["vision"]["depth"] == 1
    assert report["episodes"][0]["steps"] == 3
    assert len(factory.envs[0].actions) == 3


def test_vision_depth_zero_means_follow_checkpoint_v1(tmp_path):
    checkpoint = _write_v1(tmp_path / "v1.pt", depth=2, keep_arch=False)
    report = _run(tmp_path, StubFactory(), vision_checkpoint=checkpoint, vision_depth=0)
    assert report["vision"]["arch"] == "v1"
    assert report["vision"]["depth"] == 2


def test_matching_explicit_vision_depth_is_accepted(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=2)
    report = _run(tmp_path, StubFactory(), vision_checkpoint=checkpoint, vision_depth=2)
    assert report["vision"]["depth"] == 2


def test_vision_depth_conflict_raises(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    factory = StubFactory()
    with pytest.raises(ValueError) as excinfo:
        _run(tmp_path, factory, vision_checkpoint=checkpoint, vision_depth=3)
    assert str(excinfo.value) == "v104:CHECKPOINT: --vision-depth 3 != checkpoint depth 1"
    # The mismatch is reported before the environment is touched at all.
    assert factory.envs == []


def test_negative_vision_depth_is_a_checkpoint_error(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    factory = StubFactory()
    with pytest.raises(ValueError, match="v104:CHECKPOINT: --vision-depth -1 is negative"):
        _run(tmp_path, factory, vision_checkpoint=checkpoint, vision_depth=-1)
    assert factory.envs == []


def test_vision_depth_conflict_cli_exits_2_with_the_prefix(tmp_path, capsys):
    """C6: the CLI prints ``v104:CHECKPOINT:`` to stderr and exits 2 (not 1)."""
    from touhou_rl.evaluate import main as evaluate_main

    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    code = evaluate_main(
        [
            "--vision-checkpoint",
            str(checkpoint),
            "--vision-depth",
            "3",
            "--device",
            "cpu",
            "--episodes",
            "1",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.startswith("v104:CHECKPOINT: ")
    assert "--vision-depth 3 != checkpoint depth 1" in captured.err
    # No report is written when the checkpoint cannot be deployed.
    assert not (tmp_path / "out" / "eval-report.json").exists()


def test_unknown_model_type_is_not_called_v1():
    """``vision_arch`` reports the actual type; anything else is an error (C6)."""
    from touhou_rl.evaluate import vision_arch

    class NotAVisionModel(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="v104:CHECKPOINT: unknown vision model type"):
        vision_arch(NotAVisionModel())

    legacy = VisionScorer(VisionNetConfig(depth=1, width=8, trunk_hidden=32))
    assert vision_arch(legacy) == "v1"
    assert vision_arch(VisionScorerV2(VisionNetConfigV2(depth=1))) == "v2"


def _write_v3(path: Path) -> Path:
    """A real v3-st checkpoint (S3-5 model) with complete C6 identity."""
    from touhou_rl.vmodels3 import VisionSpatialTemporalConfig, VisionSpatialTemporalPolicy

    model = VisionSpatialTemporalPolicy(VisionSpatialTemporalConfig())
    save_vision_checkpoint(
        path,
        model,
        {
            # Policy-stage checkpoint: C6 requires a real source-state sha here.
            "source_state_sha": "d" * 64,
            "teacher_fingerprint": "a" * 64,
            "manifest_sha": "b" * 64,
            "val_index_sha256": "c" * 64,
            "loss_config": {"objective": "rank", "rank_delta_px": 0.5},
            "stage": "policy",
            "epoch": 1,
            "depth": 3,
            "history_contract": "causal-segment-v1",
        },
    )
    return path


def test_v3_checkpoint_deploys_through_the_vision_branch(tmp_path):
    """G-v3 approved: the v3-st model dispatches through the same deployment path.

    The vision node must label the actual arch, carry the v3 ``contract`` metadata
    and record the A-04 step cap; the episode loop must run with the four-tuple
    action and the deployment diagnostics.  Still a stub environment: this is a
    plumbing test, not a real-game run.
    """
    checkpoint = _write_v3(tmp_path / "v3.pt")
    factory = StubFactory(steps_to_end=2)
    report = _run(tmp_path, factory, vision_checkpoint=checkpoint, max_steps=12000)

    assert report["policy"] == "vision"
    node = report["vision"]
    assert node["arch"] == "v3-st"
    assert node["depth"] == 3
    assert node["region_sizes"] == {
        "playfield": [224, 192],
        "hud_score": [34, 216],
        "hud_status": [126, 216],
    }
    assert node["frame_shape"]["playfield"] == [224, 192, vision.VISION_CHANNELS]
    assert node["max_steps"] == 12000
    assert node["isolation"].startswith("I-1")

    contract = node["contract"]
    assert contract["schema"] == "vision-checkpoint/3"
    assert contract["arch"] == "v3-st"
    assert contract["candidate_order_version"] == "focus9-move-v1"
    assert contract["score_semantics"] == "preference_logits"
    assert contract["history_contract"] == "causal-segment-v1"
    assert list(contract["roi_native_hw"]) == [128, 128]
    assert list(contract["local_grid_hw"]) == [16, 16]
    assert contract["align_corners"] is False
    assert contract["identity"]["source_state_sha"] == "d" * 64
    assert contract["identity"]["manifest_sha"] == "b" * 64

    row = report["episodes"][0]
    assert row["steps"] == 2
    assert row["vision_diagnostics"]["screenshot_to_action_ms"]["count"] == 2
    env = factory.envs[0]
    assert len(env.actions) == 2
    assert env.state_reads == 0
    for action in env.actions:
        assert int(action[1]) == 1 and int(action[3]) == 0
    assert env.closed is True


# --------------------------------------------------------------------------- #
# deployment diagnostics (C6): latency + observed frame delta
# --------------------------------------------------------------------------- #
class RegressingStubEnv(StubEnv):
    """Frame counter that goes backwards once, to exercise ``nonpositive``."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if info["frame"] == 4:
            info["frame"] = 2  # delta 2 -> 2 is non-monotonic
        return obs, reward, terminated, truncated, info


class RegressingFactory:
    """``env_factory`` returning a :class:`RegressingStubEnv`."""

    def __init__(self):
        self.envs = []
        self.configs = []

    def __call__(self, env_config):
        self.configs.append(env_config)
        env = RegressingStubEnv(steps_to_end=3)
        env.env_config = env_config
        self.envs.append(env)
        return env


def test_vision_diagnostics_contract(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    factory = StubFactory(steps_to_end=3)
    report = _run(tmp_path, factory, vision_checkpoint=checkpoint)
    row = report["episodes"][0]

    diagnostics = row["vision_diagnostics"]
    assert set(diagnostics) == {
        "bomb_actions_requested",
        "screenshot_to_action_ms",
        "observed_frame_delta",
    }
    assert diagnostics["bomb_actions_requested"] == 0  # no rule bomb, counted from the action
    timing = diagnostics["screenshot_to_action_ms"]
    assert set(timing) == {"count", "mean", "p50", "p95", "max"}
    assert timing["count"] == 3
    assert timing["mean"] > 0
    assert 0 < timing["p50"] <= timing["p95"] <= timing["max"]
    delta = diagnostics["observed_frame_delta"]
    assert set(delta) == {"count", "min", "p50", "p95", "max", "nonpositive"}
    # The stub reports frame=2,4,6 for stage 0: two same-stage positive deltas.
    assert delta["count"] == 2
    assert delta["min"] == delta["max"] == 2
    assert delta["nonpositive"] == 0


def test_vision_diagnostics_count_nonpositive_deltas_separately(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    factory = RegressingFactory()
    report = _run(tmp_path, factory, vision_checkpoint=checkpoint)
    diagnostics = report["episodes"][0]["vision_diagnostics"]
    # frames are 2, 4 -> 2, 6: one usable delta (2 -> 6) and one non-monotonic step.
    assert diagnostics["observed_frame_delta"]["count"] == 1
    assert diagnostics["observed_frame_delta"]["nonpositive"] == 1
    assert diagnostics["observed_frame_delta"]["min"] == 4
    assert len(factory.envs[0].actions) == 3


def test_vision_diagnostics_empty_statistics_are_null_not_zero(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=2)
    report = _run(tmp_path, StubFactory(steps_to_end=1), vision_checkpoint=checkpoint)
    diagnostics = report["episodes"][0]["vision_diagnostics"]
    timing = diagnostics["screenshot_to_action_ms"]
    assert timing["count"] == 1 and timing["mean"] is not None
    delta = diagnostics["observed_frame_delta"]
    assert delta["count"] == 0
    assert delta["min"] is None and delta["p50"] is None and delta["p95"] is None and delta["max"] is None
    assert delta["nonpositive"] == 0


def test_vision_diagnostics_raw_samples_are_written(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    out = tmp_path / "out"
    _run(tmp_path, StubFactory(steps_to_end=3), vision_checkpoint=checkpoint, out_dir=out)
    raw = json.loads((out / "vision-diagnostics-raw.json").read_text(encoding="utf-8"))
    assert raw["schema"] == "v104-vision-diagnostics-raw/1"
    assert raw["episodes"][0]["decisions"] == 3
    assert len(raw["episodes"][0]["timing_ms"]) == 3
    assert raw["episodes"][0]["frame_delta"] == [2, 2]
    assert "never used as a statistic" in raw["note"]


def test_non_vision_episodes_have_no_diagnostics_key(tmp_path):
    report = _run(tmp_path, StubFactory(), random_policy=True)
    assert "vision_diagnostics" not in report["episodes"][0]
    assert not (tmp_path / "out" / "vision-diagnostics-raw.json").exists()


def test_vision_node_records_the_step_cap_for_a04(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    report = _run(tmp_path, StubFactory(), vision_checkpoint=checkpoint, max_steps=12000)
    assert report["vision"]["max_steps"] == 12000


# --------------------------------------------------------------------------- #
# D6 judging chain: hit_step_cap belongs to the comparison tool
# --------------------------------------------------------------------------- #


def test_hit_step_cap_is_computed_from_termination_reason(tmp_path):
    checkpoint = _write_v2(tmp_path / "v2.pt", depth=1)
    capped = _run(
        tmp_path,
        StubFactory(steps_to_end=2, termination_reason="max_steps"),
        vision_checkpoint=checkpoint,
        out_dir=tmp_path / "cap",
    )
    ended = _run(
        tmp_path, StubFactory(steps_to_end=2), vision_checkpoint=checkpoint, out_dir=tmp_path / "end"
    )

    assert capped["episodes"][0]["termination_reason"] == "max_steps"
    assert ended["episodes"][0]["termination_reason"] == "game over"
    assert COMPARE.summarise(capped, "stub")["hit_step_cap"] == 1
    assert COMPARE.summarise(ended, "stub")["hit_step_cap"] == 0
    # evaluate.py records the reason; counting them is the tool's job.
    assert "hit_step_cap" not in capped


# --------------------------------------------------------------------------- #
# config defaults
# --------------------------------------------------------------------------- #


def test_eval_config_defaults(tmp_path):
    config = EvalConfig(checkpoint=Path("."), out_dir=tmp_path)
    assert config.env_factory is None  # None -> the real Th08Env
    assert config.vision_depth == 0  # follow the checkpoint
