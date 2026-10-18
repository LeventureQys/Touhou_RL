"""Unit tests for the vision layer (frame contract, shard format, models).

These run without the game.  They pin the properties whose silent breakage would
waste the most GPU time:

* the crop/resize keeps an isolated bullet visible (area averaging, not point
  sampling) - a bullet that vanishes from the tensor is a policy that cannot see
  the thing that kills it;
* frame stacks are ordered oldest-first, so "motion" has a consistent sign;
* shards round-trip exactly, and history indices never cross a shard boundary
  (a shard boundary is an episode boundary and would fabricate motion);
* the vision model's output shapes match the candidate set the teacher scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl import obs, spec, vision
from touhou_rl.vision_data import VisionDataset, VisionShardWriter
from touhou_rl.vmodels import (
    N_CANDIDATES,
    VisionNetConfig,
    VisionScorer,
    load_vision_checkpoint,
    save_vision_checkpoint,
)


def _client_frame(value: int = 24) -> np.ndarray:
    frame = np.full((480, 640, 3), value, dtype=np.uint8)
    return frame


def test_box_clamped_into_client_area():
    box = vision.PlayfieldBox(x=600, y=470, w=384, h=448)
    fitted = box.clamped(640, 480)
    assert fitted.x == 600 and fitted.y == 470
    assert fitted.x + fitted.w <= 640 and fitted.y + fitted.h <= 480
    assert fitted.w >= 1 and fitted.h >= 1


def test_th08_box_matches_spec():
    box = vision.PlayfieldBox.for_th08()
    assert (box.x, box.y, box.w, box.h) == (
        int(spec.PLAYFIELD_X),
        int(spec.PLAYFIELD_Y),
        int(spec.PLAYFIELD_W),
        int(spec.PLAYFIELD_H),
    )


def test_encode_frame_shape_and_dtype():
    frame = vision.encode_frame(_client_frame())
    assert frame.shape == (vision.VISION_H, vision.VISION_W, vision.VISION_CHANNELS)
    assert frame.dtype == np.uint8


def test_single_bullet_survives_downscale():
    """A 6x6 bright dot must leave energy in the tensor (area averaging)."""
    frame = _client_frame(0)
    box = vision.PlayfieldBox.for_th08()
    cx, cy = box.x + 200, box.y + 220
    frame[cy : cy + 6, cx : cx + 6] = 255
    encoded = vision.encode_frame(frame, box)
    assert int(encoded.max()) > 100, "bullet disappeared in the downscale"
    # The dot must land where the geometry says: playfield-relative 200/384, 220/448.
    ys, xs = np.unravel_index(np.argmax(encoded.max(axis=2)), encoded.shape[:2])
    assert abs(xs / vision.VISION_W - 200 / 384) < 0.06
    assert abs(ys / vision.VISION_H - 220 / 448) < 0.06


def test_frame_stack_is_oldest_first():
    stack = vision.FrameStack(depth=3)
    shape = (vision.VISION_H, vision.VISION_W, vision.VISION_CHANNELS)
    for value in (1, 2, 3):
        out = stack.push(np.full(shape, value, dtype=np.uint8))
    assert out.shape == (3, *shape)
    assert int(out[0, 0, 0, 0]) == 1 and int(out[-1, 0, 0, 0]) == 3


def test_frames_to_batch_scales_and_adds_batch_dim():
    stack = np.full((3, vision.VISION_H, vision.VISION_W, 3), 51, dtype=np.uint8)
    batch = vision.frames_to_batch(stack)
    assert tuple(batch.shape) == (1, 3, vision.VISION_H, vision.VISION_W, 3)
    assert pytest.approx(float(batch.max()), rel=1e-3) == 0.2


def _observation(seed: int = 0) -> obs.Observation:
    rng = np.random.default_rng(seed)
    return obs.Observation(
        self_feat=rng.random(obs.N_SELF, dtype=np.float32),
        bullets=rng.random((obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float32).astype(np.float16),
        danger_map=rng.integers(0, 8, size=(obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8),
    )


def _regions(value: int) -> dict[str, bytes]:
    """One PNG per region, distinct sizes so mix-ups are visible."""
    sizes = {"playfield": (48, 64), "hud_score": (8, 32), "hud_status": (16, 32)}
    return {
        name: vision.encode_png(np.full((h, w, 3), value, dtype=np.uint8))
        for name, (h, w) in sizes.items()
    }


def test_shard_roundtrip(tmp_path):
    writer = VisionShardWriter(tmp_path, shard_size=3, tag="t")
    for index in range(7):
        writer.add(
            _regions(index * 20),
            _observation(index),
            (index % 9, 1, 0, 0),
            {name: index for name in
             ("frame", "stage", "score", "graze", "lives", "bombs", "power", "deaths")},
        )
    summary = writer.close()
    assert summary["frames"] == 7
    dataset = VisionDataset(tmp_path)
    assert len(dataset) == 7
    assert set(dataset.region_names) == {"playfield", "hud_score", "hud_status"}
    frames = dataset.frames([0, 3, 6], "playfield")
    assert frames.shape == (3, 48, 64, 3)
    assert int(frames[0, 0, 0, 0]) == 0
    assert int(frames[2, 0, 0, 0]) == 120
    hud = dataset.frames([1], "hud_status")
    assert hud.shape == (1, 16, 32, 3)
    # decoding at a different tensor size must work without touching the archive
    small = dataset.frames([0], "playfield", 24, 32)
    assert small.shape == (1, 24, 32, 3)
    labels = dataset.labels([1, 4])
    assert labels["action"].shape == (2, 4)
    assert int(labels["action"][1, 0]) == 4


def test_stack_indices_never_cross_shard_boundary(tmp_path):
    writer = VisionShardWriter(tmp_path, shard_size=3, tag="t")
    for index in range(7):
        writer.add(
            _regions(index),
            _observation(index),
            (0, 1, 0, 0),
            {name: 0 for name in ("frame", "stage", "score", "graze", "lives", "bombs", "power", "deaths")},
        )
    writer.close()
    dataset = VisionDataset(tmp_path)
    stack = dataset.stack_indices([0, 2, 3], depth=3)
    # index 0 is the first frame of shard 0 -> all history clamps to 0
    assert list(stack[0]) == [0, 0, 0]
    # index 2 is the last frame of shard 0 -> history [0, 1, 2]
    assert list(stack[1]) == [0, 1, 2]
    # index 3 is the first frame of shard 1 -> history must NOT reach into shard 0
    assert list(stack[2]) == [3, 3, 3]


def _model_regions(config, batch: int = 4, depth: int = 2) -> dict:
    """One zero tensor per model region, at that region's configured size."""
    return {
        name: np.zeros((batch, depth, height, width, 3), dtype=np.uint8)
        for name, height, width in config.model_regions
    }


def test_vision_model_shapes():
    config = VisionNetConfig(depth=2, width=8, trunk_hidden=32)
    model = VisionScorer(config)
    batch = vision.regions_to_batch(_model_regions(config))
    values = model.score(batch)
    assert tuple(values.shape) == (4, N_CANDIDATES)
    state = model.decode_state(batch)
    assert tuple(state["self_feat"].shape) == (4, obs.N_SELF)
    assert tuple(state["danger_log"].shape) == (4, obs.DANGER_ROWS * obs.DANGER_COLS)
    actions, _logp, _value, _extra = model.act(batch)
    assert tuple(actions.shape) == (4, 4)
    assert int(actions[0, 1]) == 1  # shoot always on
    assert int(actions[0, 3]) == 0  # bomb is left to a rule


def test_vision_checkpoint_roundtrip(tmp_path):
    config = VisionNetConfig(depth=2, width=8, trunk_hidden=32)
    model = VisionScorer(config)
    path = tmp_path / "vision.pt"
    save_vision_checkpoint(path, model, {"stage": "state"})
    restored, extra = load_vision_checkpoint(path)
    assert extra["stage"] == "state"
    batch = vision.regions_to_batch(_model_regions(config, batch=1))
    assert np.allclose(
        model.score(batch).detach().numpy(), restored.score(batch).detach().numpy(), atol=1e-6
    )
