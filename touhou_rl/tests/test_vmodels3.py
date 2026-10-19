"""Unit tests for the v3 (``v3-st``) spatial model and its frozen interface.

``test_vmodels2.py`` pins v1/v2; these tests pin the C5 contract that S3-7's
training runs and S3-6's checkpoint dispatch depend on:

* the exact config surface and the ``v104:CONFIG`` / ``v104:DATA`` / ``KeyError``
  error behaviour,
* the layer geometry: F4 56x48, F8 28x24, heatmap 56x48, danger 128, and the
  56x48 -> 16x16 local crop in *native* pixels (8 px per bin, ``align_corners``
  false, centres at +-60 px),
* which pixels reach which output (playfield motion yes, old HUD frames no),
* which loss reaches which weights (heatmap loss -> localisation head; policy
  loss -> F4 and the local branch, never through the detached sampling centre),
* the 18-candidate action mapping.

No test asserts *quality*: random weights cannot localise a player or rank
movements, so only shapes, geometry, finiteness and gradient routing are pinned
here.  Localisation/ranking quality belongs to the Stage 3/4 experiments.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from touhou_rl import obs, spec, vision
from touhou_rl.vmodels3 import (
    FUSED_VEC,
    LOCAL_VEC,
    NATIVE_PLAYFIELD_HW,
    VisionSpatialTemporalConfig,
    VisionSpatialTemporalPolicy,
    decode_location,
    local_grid,
    sample_local,
)

REGION_NAMES = ("playfield", "hud_score", "hud_status")
FORWARD_KEYS = {
    "policy_logits",
    "heatmap_logits",
    "position",
    "danger_log",
    "self_pred",
    "features",
    "location_stats",
}
#: Frozen default parameter count; a change here changes the checkpoint identity
#: and has to be reported (v2, for reference, is 554_873).
PARAMETERS = 1_507_162


# --------------------------------------------------------------------------- #
# fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def config() -> VisionSpatialTemporalConfig:
    return VisionSpatialTemporalConfig()


@pytest.fixture(scope="module")
def model(config) -> VisionSpatialTemporalPolicy:
    torch.manual_seed(7)
    return VisionSpatialTemporalPolicy(config).eval()


def _regions(config, batch: int = 1, seed: int | None = None) -> dict:
    """Random uint8 frames through the *real* preprocessing (C1)."""
    array = np.zeros if seed is None else np.random.default_rng(seed).integers
    regions = {}
    for name, (height, width) in config.region_sizes.items():
        if seed is None:
            regions[name] = np.zeros((batch, config.depth, height, width, 3), dtype=np.uint8)
        else:
            regions[name] = array(
                0, 256, (batch, config.depth, height, width, 3), dtype=np.uint8
            )
    return vision.regions_to_batch(regions)


def _directed_model(config) -> VisionSpatialTemporalPolicy:
    """A model whose heatmap is a pure readout of F4 channel 0..63.

    The localisation conv becomes a centre-tap pass-through and the 1x1 head sums
    the channels, so a spike placed by hand in ``F4`` must appear at exactly that
    cell of the heatmap.  That turns the geometry into an assertion instead of a
    property of random weights.
    """
    torch.manual_seed(3)
    model = VisionSpatialTemporalPolicy(config).eval()
    with torch.no_grad():
        model.heat_conv[0].weight.zero_()
        model.heat_conv[0].bias.zero_()
        for channel in range(64):
            model.heat_conv[0].weight[channel, channel, 1, 1] = 1.0
        model.heatmap_head.weight.fill_(1.0)
        model.heatmap_head.bias.zero_()
    return model


# --------------------------------------------------------------------------- #
# config contract
# --------------------------------------------------------------------------- #
def test_config_defaults_match_c5(config):
    assert config.depth == 3
    assert config.region_sizes == {
        "playfield": (224, 192),
        "hud_score": (34, 216),
        "hud_status": (126, 216),
    }
    assert config.region_names == REGION_NAMES
    assert config.heatmap_hw == (56, 48)
    assert config.local_enabled is True
    assert config.roi_native_hw == (128, 128)
    assert config.local_grid_hw == (16, 16)
    assert config.align_corners is False
    assert config.center_detach is True
    assert config.heatmap_gauss_sigma == 2.0
    assert config.candidate_order_version == "focus9-move-v1"
    assert config.history_contract == "causal-segment-v1"
    assert config.score_semantics == "preference_logits"
    assert config.native_playfield_hw == (448, 384)
    # The fusion layer really consumes 128+128+64+64+16.
    assert FUSED_VEC == 400
    # ``state_head`` is deliberately not part of this architecture (C5 allows
    # leaving the switch out instead of pretending the state path can be disabled).
    assert not hasattr(config, "state_head")
    assert VisionSpatialTemporalPolicy.ARCH == "v3-st"


@pytest.mark.parametrize(
    "overrides",
    [
        {"depth": 1},
        {"depth": 4},
        {"playfield_hw": (225, 192)},
        {"playfield_hw": (224, 190)},
        {"playfield_hw": (0, 192)},
        {"playfield_hw": (224, -8)},
        {"hud_score_hw": (0, 216)},
        {"hud_status_hw": (126, 216.5)},
        {"roi_native_hw": (0, 128)},
        {"roi_native_hw": (128,)},
        {"local_grid_hw": (0, 16)},
        {"local_grid_hw": (5, 16)},  # 128 is not divisible by 5
        {"align_corners": True},
        {"local_enabled": "yes"},
        {"center_detach": 1},
        {"heatmap_gauss_sigma": 0.0},
        {"heatmap_gauss_sigma": float("nan")},
        {"candidate_order_version": "focus9"},
        {"score_semantics": "value"},
        {"history_contract": "unknown-history"},
        {"pf_width": 0},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError, match=r"v104:CONFIG: "):
        VisionSpatialTemporalConfig(**overrides)


def test_config_round_trips_through_a_plain_dict(config):
    """``__dict__`` is what S3-6's checkpoint writer stores; it must rebuild."""
    assert VisionSpatialTemporalConfig(**dict(config.__dict__)) == config


# --------------------------------------------------------------------------- #
# forward surface
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch", [1, 3])
def test_forward_shapes_dtype_and_finiteness(model, config, batch):
    regions = _regions(config, batch=batch, seed=batch)
    with torch.no_grad():
        out = model(regions)

    assert set(out) == FORWARD_KEYS
    expected = {
        "policy_logits": (batch, 18),
        "heatmap_logits": (batch, *config.heatmap_hw),
        "position": (batch, 2),
        "danger_log": (batch, obs.DANGER_ROWS * obs.DANGER_COLS),
        "self_pred": (batch, obs.N_SELF),
        "features": (batch, 256),
        "location_stats": (batch, 2),
    }
    for key, shape in expected.items():
        value = out[key]
        assert tuple(value.shape) == shape, (key, tuple(value.shape))
        assert value.dtype == torch.float32, key
        assert torch.isfinite(value).all(), key
    assert config.heatmap_hw == (56, 48)
    # Positions are normalised cell-centre expectations inside the field.
    assert torch.all(out["position"] > 0) and torch.all(out["position"] < 1)
    stats = out["location_stats"]
    assert torch.all(stats[:, 0] >= 0) and torch.all(stats[:, 0] <= 1.0001)
    assert torch.all(stats[:, 1] > 0) and torch.all(stats[:, 1] <= 1)


def test_forward_accepts_one_unbatched_region(model, config):
    regions = _regions(config, batch=1, seed=0)
    single = {name: value[0] for name, value in regions.items()}
    with torch.no_grad():
        out = model(single)
    assert tuple(out["policy_logits"].shape) == (1, 18)


def test_legacy_accessors_agree_with_forward(model, config):
    regions = _regions(config, batch=2, seed=11)
    with torch.no_grad():
        out = model(regions)
        scores = model.score(regions)
        features = model.features(regions)
        state = model.decode_state(regions)

    assert set(state) == {"self_feat", "danger_log", "heatmap_logits"}
    assert torch.equal(scores, out["policy_logits"])
    assert torch.equal(features, out["features"])
    assert torch.equal(state["heatmap_logits"], out["heatmap_logits"])
    assert torch.equal(state["danger_log"], out["danger_log"])
    assert torch.equal(state["self_feat"], out["self_pred"])


def test_external_features_only_replace_the_self_head(model, config):
    """C5: an external vector must never stand in for the spatial map."""
    regions = _regions(config, batch=2, seed=12)
    external = torch.randn(2, model.config.trunk_hidden)
    with torch.no_grad():
        out = model(regions)
        state = model.decode_state(regions, features=external)
        scores = model.score(regions, features=external)

    # Spatial outputs are recomputed from the pixels, byte for byte.
    assert torch.equal(state["heatmap_logits"], out["heatmap_logits"])
    assert torch.equal(state["danger_log"], out["danger_log"])
    # The self head (columns 2..23) does use the supplied vector; columns 0..1
    # stay the heatmap position.
    assert torch.allclose(state["self_feat"][:, 2:], model.self_head(external))
    assert torch.equal(state["self_feat"][:, :2], out["position"])
    assert torch.allclose(scores, model.policy_head(external))


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(2, 128),  # wrong width
        torch.zeros(3, 256),  # wrong batch
        torch.zeros(2, 256, dtype=torch.float64),  # wrong dtype
        torch.full((2, 256), float("nan")),  # non-finite
    ],
)
def test_features_argument_is_validated(model, config, bad):
    regions = _regions(config, batch=2, seed=13)
    with pytest.raises(ValueError, match=r"v104:DATA: "):
        model.decode_state(regions, features=bad)
    with pytest.raises(ValueError, match=r"v104:DATA: "):
        model.score(regions, features=bad)


def test_parameter_count(model):
    assert model.num_parameters() == PARAMETERS
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS
    # v2 is 554_873; v3 is bigger, and the report records why.
    assert PARAMETERS > 554_873


# --------------------------------------------------------------------------- #
# localisation geometry
# --------------------------------------------------------------------------- #
def test_directed_f4_spike_lands_on_the_declared_cell(config):
    """Centre, four corners and an off-centre cell -> exact soft-argmax position."""
    model = _directed_model(config)
    height, width = config.heatmap_hw  # 56 x 48
    f8 = torch.zeros(1, 96, height // 2, width // 2)
    for row, column in [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1), (28, 24), (10, 30)]:
        f4 = torch.zeros(1, 64, height, width)
        f4[0, :, row, column] = 50.0
        heatmap = model.locate(f4, f8)
        position, stats = decode_location(heatmap)
        assert tuple(torch.nonzero(heatmap[0] == heatmap[0].max())[0].tolist()) == (row, column)
        assert position[0, 0].item() == pytest.approx((column + 0.5) / width, abs=1e-6)
        assert position[0, 1].item() == pytest.approx((row + 0.5) / height, abs=1e-6)
        assert stats[0, 0].item() < 1e-6  # a single cell carries all the mass
        assert stats[0, 1].item() == pytest.approx(1.0, abs=1e-6)


def test_position_axes_have_screen_signs(config):
    """x grows to the right, y grows downwards (C1 screen coordinates)."""
    model = _directed_model(config)
    f8 = torch.zeros(1, 96, 28, 24)

    def position_at(row: int, column: int) -> list[float]:
        with torch.no_grad():
            f4 = torch.zeros(1, 64, 56, 48)
            f4[0, :, row, column] = 50.0
            return decode_location(model.locate(f4, f8))[0][0].tolist()

    left, right = position_at(28, 10), position_at(28, 40)
    up, down = position_at(10, 24), position_at(45, 24)
    assert right[0] > left[0] and right[1] == pytest.approx(left[1], abs=1e-6)
    assert down[1] > up[1] and down[0] == pytest.approx(up[0], abs=1e-6)


def test_point_image_lands_on_the_expected_f4_and_f8_cells(config):
    """A dot at native (y0, x0) must land on F4 (y0/4, x0/4) and F8 (y0/8, x0/8).

    The two strided convs are re-weighted as centre taps, so this pins the
    alignment of the trunk with the native playfield: cell (row, column) of F4
    owns native pixels ``[4*row, 4*row+4) x [4*column, 4*column+4)``.
    """
    torch.manual_seed(5)
    model = VisionSpatialTemporalPolicy(config).eval()
    with torch.no_grad():
        model.playfield.stem[0].weight.zero_()
        model.playfield.stem[0].bias.zero_()
        model.playfield.stem[0].weight[0, 0, 2, 2] = 1.0  # 9th channel is not needed
        model.playfield.stage4[0].weight.zero_()
        model.playfield.stage4[0].bias.zero_()
        for out_channel in range(64):
            model.playfield.stage4[0].weight[out_channel, 0, 1, 1] = 1.0
        model.playfield.stage8[0].weight.zero_()
        model.playfield.stage8[0].bias.zero_()
        for out_channel in range(96):
            model.playfield.stage8[0].weight[out_channel, 0, 1, 1] = 1.0

    y0, x0 = 24, 16  # divisible by 8 so the stride-8 cell is unambiguous
    pixels = torch.zeros(1, 9, 224, 192)
    pixels[0, 0, y0, x0] = 1.0
    f4 = model.playfield.res4(model.playfield.stage4(model.playfield.stem(pixels)))
    assert torch.nonzero(f4[0, 0] >= f4[0, 0].max() - 1e-6).tolist() == [[y0 // 4, x0 // 4]]

    # F8 reads the *residual* F4, so re-enter the trunk at the stage8 conv.
    f8 = model.playfield.res8(model.playfield.stage8(f4))
    assert torch.nonzero(f8[0, 0] >= f8[0, 0].max() - 1e-3).tolist() == [[y0 // 8, x0 // 8]]


def test_decode_location_entropy_and_peak(config):
    uniform = torch.zeros(2, 56, 48)
    position, stats = decode_location(uniform)
    cells = 56 * 48
    assert position[0].tolist() == pytest.approx([0.5, 0.5], abs=1e-3)
    assert stats[0, 0].item() == pytest.approx(1.0, abs=1e-6)  # uniform -> max entropy
    assert stats[0, 1].item() == pytest.approx(1.0 / cells, rel=1e-5)

    peaked = torch.full((1, 56, 48), -40.0)
    peaked[0, 5, 7] = 40.0
    position, stats = decode_location(peaked)
    assert stats[0, 0].item() < 1e-6
    assert stats[0, 1].item() == pytest.approx(1.0, abs=1e-6)
    assert position[0].tolist() == pytest.approx([7.5 / 48, 5.5 / 56], abs=1e-6)


# --------------------------------------------------------------------------- #
# local crop geometry
# --------------------------------------------------------------------------- #
def test_local_grid_is_the_literal_c5_formula(config):
    position = torch.tensor([[0.31, 0.72], [0.5, 0.5]])
    grid = local_grid(position, config)
    assert tuple(grid.shape) == (2, 16, 16, 2)
    native_h, native_w = NATIVE_PLAYFIELD_HW
    offsets_x = (torch.arange(16, dtype=torch.float32) + 0.5) * 8.0 - 64.0
    offsets_y = (torch.arange(16, dtype=torch.float32) + 0.5) * 8.0 - 64.0
    for index in range(2):
        rx, ry = position[index].tolist()
        expected_x = 2 * rx - 1 + 2 * offsets_x / native_w
        expected_y = 2 * ry - 1 + 2 * offsets_y / native_h
        assert torch.allclose(grid[index, 0, :, 0], expected_x, atol=1e-7)
        assert torch.allclose(grid[index, :, 0, 1], expected_y, atol=1e-7)
        assert grid[index, :, :, 0][0, 0].item() == pytest.approx(expected_x[0].item(), abs=1e-7)


def test_local_grid_covers_128_native_pixels_in_16_bins(config):
    grid = local_grid(torch.tensor([[0.5, 0.5]]), config)
    native_h, native_w = NATIVE_PLAYFIELD_HW
    offsets_x = (grid[0, 0, :, 0] - (2 * 0.5 - 1)) * native_w / 2
    offsets_y = (grid[0, :, 0, 1] - (2 * 0.5 - 1)) * native_h / 2
    assert offsets_x.tolist() == pytest.approx([-60.0 + 8.0 * i for i in range(16)])
    assert offsets_y.tolist() == pytest.approx([-60.0 + 8.0 * i for i in range(16)])
    # 16 bins x 8 native px == the declared 128 px window, centred on the player.
    assert (offsets_x[-1] + 4.0) - (offsets_x[0] - 4.0) == pytest.approx(128.0)
    assert config.roi_native_hw == (128, 128)


def test_local_validity_mask(config):
    ones = torch.ones(1, 1, 56, 48)
    centred = torch.tensor([[0.5, 0.5]])
    _patch, validity = sample_local(ones, centred, config)
    assert tuple(validity.shape) == (1, 1, 16, 16)
    assert validity.min().item() == pytest.approx(1.0)  # window fully inside

    corner = torch.tensor([[0.5 / 48, 0.5 / 56]])  # top-left cell centre
    _patch, validity = sample_local(ones, corner, config)
    assert validity.min().item() < 1.0  # edge bins are only partially covered
    assert validity.min().item() == pytest.approx(0.0)
    assert validity.max().item() == pytest.approx(1.0)
    assert 0.0 < validity.sum().item() < 16 * 16


def test_local_sampling_follows_align_corners_false(config):
    """Sampling a normalised-coordinate ramp returns ``rx + dx/384`` exactly."""
    ramp = ((torch.arange(48, dtype=torch.float32) + 0.5) / 48).view(1, 1, 1, 48)
    ramp = ramp.expand(2, 1, 56, 48).contiguous()
    # Keep the whole 128 px window inside the field (|dx| <= 60 px = 0.156 of 384).
    position = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
    patch, _validity = sample_local(ramp, position, config)
    offsets_x = (torch.arange(16, dtype=torch.float32) + 0.5) * 8.0 - 64.0
    for index in range(2):
        rx = position[index, 0].item()
        expected = rx + offsets_x / NATIVE_PLAYFIELD_HW[1]
        assert torch.allclose(
            patch[index, 0, 0, :], expected, atol=1e-6
        ), (rx, patch[index, 0, 0, :].tolist())


# --------------------------------------------------------------------------- #
# local ablation and input routing
# --------------------------------------------------------------------------- #
def test_local_disabled_feeds_a_zero_vector_with_identical_slots(config):
    torch.manual_seed(9)
    enabled = VisionSpatialTemporalPolicy(config).eval()
    disabled = VisionSpatialTemporalPolicy(
        VisionSpatialTemporalConfig(local_enabled=False)
    ).eval()
    disabled.load_state_dict(enabled.state_dict())

    assert enabled.num_parameters() == disabled.num_parameters()
    assert {key: tuple(v.shape) for key, v in enabled.state_dict().items()} == {
        key: tuple(v.shape) for key, v in disabled.state_dict().items()
    }
    f4 = torch.zeros(2, 64, 56, 48)
    position = torch.tensor([[0.5, 0.5], [0.2, 0.3]])
    assert torch.count_nonzero(disabled.local_features(f4, position)).item() == 0
    assert tuple(disabled.local_features(f4, position).shape) == (2, LOCAL_VEC)
    assert torch.count_nonzero(enabled.local_features(f4, position)).item() > 0

    regions = _regions(config, batch=1, seed=21)
    with torch.no_grad():
        assert not torch.equal(
            enabled(regions)["features"], disabled(regions)["features"]
        )


def test_old_playfield_frames_change_the_output(model, config):
    """The playfield sees the whole depth stack, so motion is visible (C1)."""
    regions = _regions(config, batch=2, seed=31)
    modified = {name: value.clone() for name, value in regions.items()}
    modified["playfield"][:, 0] = torch.flip(modified["playfield"][:, 0], dims=[-1])
    with torch.no_grad():
        base = model(regions)
        changed = model(modified)
    assert not torch.allclose(base["features"], changed["features"])
    assert not torch.allclose(base["heatmap_logits"], changed["heatmap_logits"])


def test_old_hud_frames_do_not_change_the_output(model, config):
    """Only the newest HUD frame is encoded; the older ones cannot leak in."""
    regions = _regions(config, batch=2, seed=32)
    modified = {name: value.clone() for name, value in regions.items()}
    for name in ("hud_score", "hud_status"):
        modified[name][:, 0] = torch.flip(modified[name][:, 0], dims=[-1])
        modified[name][:, 1] = 1.0 - modified[name][:, 1]
    with torch.no_grad():
        base = model(regions)
        changed = model(modified)
    for key in sorted(FORWARD_KEYS):
        assert torch.equal(base[key], changed[key]), key


def test_newest_hud_frame_does_change_the_output(model, config):
    """Guard against the previous test passing because the HUD is ignored."""
    regions = _regions(config, batch=2, seed=33)
    modified = {name: value.clone() for name, value in regions.items()}
    modified["hud_score"][:, -1] = 1.0 - modified["hud_score"][:, -1]
    with torch.no_grad():
        base = model(regions)
        changed = model(modified)
    assert not torch.allclose(base["features"], changed["features"])


# --------------------------------------------------------------------------- #
# action contract
# --------------------------------------------------------------------------- #
def test_act_candidate_mapping(model, config):
    """``index = focus*9 + move`` -> ``[move, 1, focus, 0]``, int64 (C1)."""
    regions = _regions(config, batch=3, seed=41)
    logits = torch.full((3, 18), -5.0)
    logits[0, 0] = 5.0  # index 0 -> idle, no focus
    logits[1, 17] = 5.0  # index 17 -> move 8, focus 1
    logits[2, 9] = 5.0  # index 9 -> move 0, focus 1
    original = model.score
    model.score = lambda regions, features=None: logits
    try:
        actions, log_probability, value, extra = model.act(regions)
    finally:
        model.score = original

    assert actions.dtype == torch.int64
    assert tuple(actions.shape) == (3, 4)
    assert actions.tolist() == [[0, 1, 0, 0], [8, 1, 1, 0], [0, 1, 1, 0]]
    assert torch.all(actions[:, 1] == 1)  # shooting always on
    assert torch.all(actions[:, 3] == 0)  # bombing is a rule
    assert torch.all(actions[:, 0] >= 0) and torch.all(actions[:, 0] < spec.ACTION_DIMS[0])
    assert torch.all(actions[:, 2] >= 0) and torch.all(actions[:, 2] < spec.ACTION_DIMS[2])
    assert log_probability.tolist() == [0.0, 0.0, 0.0]
    assert extra.tolist() == [0.0, 0.0, 0.0]
    assert value.tolist() == [5.0, 5.0, 5.0]
    assert log_probability.dtype == torch.float32
    assert value.dtype == torch.float32


def test_act_depends_only_on_pixels(model, config):
    """C1 isolation: relabelling/shuffling frames permutes actions, nothing else.

    The model has no label input at all; the strongest available check is that
    the same pixels in a different batch order produce the same per-frame
    actions, i.e. no cross-sample or metadata path exists.
    """
    regions = _regions(config, batch=3, seed=42)
    order = torch.tensor([2, 0, 1])
    permuted = {name: value[order] for name, value in regions.items()}
    with torch.no_grad():
        actions = model.act(regions)[0]
        again = model.act(regions)[0]
        permuted_actions = model.act(permuted)[0]
    assert torch.equal(actions, again)
    assert torch.equal(permuted_actions, actions[order])


# --------------------------------------------------------------------------- #
# gradient routing
# --------------------------------------------------------------------------- #
def test_localisation_loss_reaches_the_heatmap_only(config):
    torch.manual_seed(13)
    model = VisionSpatialTemporalPolicy(config).eval()
    out = model(_regions(config, batch=1, seed=51))
    out["heatmap_logits"].sum().backward()

    assert model.heat_conv[0].weight.grad is not None
    assert model.heat_conv[0].weight.grad.abs().sum().item() > 0
    assert model.heatmap_head.weight.grad.abs().sum().item() > 0
    # The trunk feeds the heatmap, so it must receive this loss too ...
    assert model.playfield.stem[0].weight.grad.abs().sum().item() > 0
    # ... but nothing downstream of the fused vector may.
    for name in ("trunk.0.weight", "policy_head.weight", "self_head.weight", "global_fc.0.weight"):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is None, name


def test_policy_loss_reaches_f4_and_the_local_branch(config):
    torch.manual_seed(14)
    model = VisionSpatialTemporalPolicy(config).eval()
    out = model(_regions(config, batch=1, seed=52))
    out["policy_logits"].sum().backward()

    gradients = dict(model.named_parameters())
    for name in (
        "playfield.stem.0.weight",
        "playfield.stage4.0.weight",
        "playfield.res4.0.body.0.weight",
        "playfield.res8.0.body.0.weight",  # F8 still feeds the heatmap path
        "local.conv1.weight",
        "local.conv2.weight",
        "local.fc.weight",
        "global_fc.0.weight",
        "hud_score.body.0.weight",
        "hud_status.body.0.weight",
        "trunk.0.weight",
        "policy_head.weight",
    ):
        grad = gradients[name].grad
        assert grad is not None and grad.abs().sum().item() > 0, name
    # The self head is not part of the policy objective.
    assert gradients["self_head.weight"].grad is None
    assert gradients["danger.reduce.weight"].grad is None


def test_policy_gradient_does_not_flow_through_the_sampling_centre(config):
    """``center_detach`` is the only thing keeping the crop centre a constant."""
    torch.manual_seed(15)
    detached = VisionSpatialTemporalPolicy(config).eval()
    attached = VisionSpatialTemporalPolicy(
        VisionSpatialTemporalConfig(center_detach=False)
    ).eval()
    attached.load_state_dict(detached.state_dict())
    # Non-zero pixels: the grid gradient is proportional to the sampled values,
    # so an all-zero F4 would report a real (but zero) gradient either way.
    f4 = torch.rand(1, 64, 56, 48)

    position = torch.tensor([[0.4, 0.6]], requires_grad=True)
    detached.local_features(f4, position).sum().backward()
    assert position.grad is None  # the grid is a constant

    position = torch.tensor([[0.4, 0.6]], requires_grad=True)
    attached.local_features(f4, position).sum().backward()
    assert position.grad is not None
    assert position.grad.abs().sum().item() > 0


# --------------------------------------------------------------------------- #
# error behaviour
# --------------------------------------------------------------------------- #
def test_missing_region_is_a_keyerror(model, config):
    regions = _regions(config, batch=1, seed=61)
    partial = {"playfield": regions["playfield"], "hud_score": regions["hud_score"]}
    with pytest.raises(KeyError) as excinfo:
        model(partial)
    assert excinfo.value.args[0] == "missing regions ['hud_status']; got ['hud_score', 'playfield']"


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda regions: regions.__setitem__("playfield", regions["playfield"][:, :1]), id="depth-1"),
        pytest.param(
            lambda regions: regions["playfield"].__setitem__((0, 0, 0, 0, 0), float("nan")),
            id="nan",
        ),
        pytest.param(
            lambda regions: regions.__setitem__("playfield", regions["playfield"].double()),
            id="float64",
        ),
        pytest.param(
            lambda regions: regions.__setitem__("hud_score", regions["hud_score"][:, :, :-1]),
            id="wrong-height",
        ),
        pytest.param(
            lambda regions: regions.__setitem__(
                "hud_status", torch.cat([regions["hud_status"], regions["hud_status"]], dim=-1)
            ),
            id="extra-channel",
        ),
        pytest.param(
            lambda regions: regions.__setitem__("hud_status", regions["hud_status"][:1]),
            id="batch-mismatch",
        ),
        pytest.param(
            lambda regions: regions.__setitem__("playfield", regions["playfield"][0]),
            id="unbatched-playfield-batch-mismatch",
        ),
        pytest.param(
            lambda regions: regions.__setitem__("playfield", regions["playfield"].reshape(2, -1)),
            id="wrong-ndim",
        ),
    ],
)
def test_data_errors(model, config, mutate):
    regions = _regions(config, batch=2, seed=62)
    mutate(regions)
    with pytest.raises(ValueError, match=r"v104:DATA: "):
        model(regions)


def test_missing_region_beats_the_data_checks(model, config):
    with pytest.raises(KeyError, match="missing regions"):
        model({})


def test_non_tensor_region_is_a_data_error(model, config):
    regions = _regions(config, batch=1, seed=63)
    regions["hud_score"] = np.zeros((1, 3, 34, 216, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=r"v104:DATA: "):
        model(regions)


# --------------------------------------------------------------------------- #
# checkpoint contract handoff (S3-6's frozen loader, no file I/O)
# --------------------------------------------------------------------------- #
def _v3_payload(model: VisionSpatialTemporalPolicy) -> dict:
    from touhou_rl import vmodels

    return {
        "schema": vmodels.V3_CHECKPOINT_SCHEMA,
        "kind": "vision_scorer",
        "arch": "v3-st",
        "config": dict(model.config.__dict__),
        "state_dict": model.state_dict(),
        # ``frame`` is the legacy H128/W112 marker, never the model input (C6).
        "frame": {"H": vision.VISION_H, "W": vision.VISION_W, "C": 3, "depth": 3},
        "extra": {
            "source_state_sha": "a" * 64,
            "teacher_fingerprint": "teacher-v1:deadbeef",
            "manifest_sha": "b" * 64,
            "val_index_sha256": "c" * 64,
            "loss_config": {"objective": "rank", "rank_delta_px": 0.5},
            "stage": "state",
            "depth": 3,
        },
    }


def test_s3_6_checkpoint_contract_accepts_this_config(model):
    """The interface S3-6 froze must accept the real config, not a fixture.

    ``dict(config.__dict__)`` is exactly what ``save_vision_checkpoint`` stores and
    what ``_v3_config_from_payload`` has to rebuild, so this locks the two
    modules together without a checkpoint file.
    """
    from touhou_rl import vmodels

    contract = vmodels.validate_v3_payload(_v3_payload(model))
    assert contract["schema"] == vmodels.V3_CHECKPOINT_SCHEMA
    assert contract["arch"] == "v3-st"
    assert contract["depth"] == 3
    assert contract["region_sizes"] == {
        "playfield": [224, 192],
        "hud_score": [34, 216],
        "hud_status": [126, 216],
    }
    assert contract["roi_native_hw"] == (128, 128)
    assert contract["local_grid_hw"] == (16, 16)
    assert contract["align_corners"] is False
    assert contract["history_contract"] == "causal-segment-v1"
    assert contract["legacy_frame"]["H"] == vision.VISION_H
    assert vmodels.vision_arch(model) == "v3-st"
