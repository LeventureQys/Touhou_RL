"""Unit tests for the v2 vision architecture and the checkpoint arch dispatch.

``test_vision.py`` pins the v1 frame/region contract; these tests pin the v2
contract that the training loop and deployment rely on: the exact ``decode_state``
keys and shapes, the soft-argmax decoder (range and sub-cell smoothness), the
gaussian heatmap targets, independent HUD encoders, the parameter budget, and the
checkpoint ``arch`` dispatch which must keep loading pre-v1.0.3 checkpoints.

No test asserts target quality: with random weights a heatmap or danger map
cannot be judged, so only shapes, finiteness and differentiability are pinned
here.  Localisation quality belongs to the Stage 3/4 experiments.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from touhou_rl import obs, vision
from touhou_rl.vmodels import (
    N_CANDIDATES,
    V3_CHECKPOINT_SCHEMA,
    V3_REQUIRED_EXTRA,
    VisionNetConfig,
    VisionScorer,
    load_vision_checkpoint,
    save_vision_checkpoint,
    validate_v3_payload,
    vision_arch,
    vision_contract_node,
)
from touhou_rl.vmodels2 import (
    VisionNetConfigV2,
    VisionScorerV2,
    decode_position,
    heatmap_target,
)
from touhou_rl.vmodels3 import VisionSpatialTemporalConfig, VisionSpatialTemporalPolicy

STATE_PATH_PREFIXES = (
    "playfield.body.",
    "playfield.heatmap_head.",
    "playfield.danger_head.",
    "playfield.vector.",
    "hud_score.",
    "hud_status.",
    "trunk.",
    "self_head.",
)
# The score path sees the pooled playfield vector only: the heatmap and danger
# heads hang off the feature map and are driven by the state losses alone.
POLICY_PATH_PREFIXES = (
    "playfield.body.",
    "playfield.vector.",
    "hud_score.",
    "hud_status.",
    "trunk.",
    "candidate_embed.",
    "score_head.",
)


def _regions(config, batch: int = 2, depth: int | None = None) -> dict:
    """Zeros at each configured region size, through the real preprocessing."""
    depth = config.depth if depth is None else depth
    return vision.regions_to_batch(
        {
            name: np.zeros((batch, depth, height, width, 3), dtype=np.uint8)
            for name, (height, width) in config.region_sizes.items()
        }
    )


def test_v2_shapes_and_keys():
    config = VisionNetConfigV2(depth=1)
    model = VisionScorerV2(config).eval()
    batch = _regions(config, batch=2)
    with torch.no_grad():
        state = model.decode_state(batch)

    assert set(state) == {"self_feat", "danger_log", "heatmap_logits"}
    self_feat = state["self_feat"]
    assert tuple(self_feat.shape) == (2, obs.N_SELF)
    assert self_feat.dtype == torch.float32
    assert torch.all(self_feat[:, 0] >= 0) and torch.all(self_feat[:, 0] <= 1)
    assert torch.all(self_feat[:, 1] >= 0) and torch.all(self_feat[:, 1] <= 1)

    assert tuple(state["danger_log"].shape) == (2, obs.DANGER_ROWS * obs.DANGER_COLS)
    assert state["danger_log"].dtype == torch.float32
    assert tuple(state["heatmap_logits"].shape) == (2, *config.heatmap_hw)
    assert config.heatmap_hw == (28, 24)
    assert config.region_names == ("playfield", "hud_score", "hud_status")


def test_v2_param_budget():
    model = VisionScorerV2()
    parameters = model.num_parameters()
    assert 450_000 <= parameters <= 650_000, parameters


def test_v2_act_contract():
    config = VisionNetConfigV2()
    model = VisionScorerV2(config).eval()
    batch = _regions(config, batch=3)
    with torch.no_grad():
        result = model.act(batch)

    assert isinstance(result, tuple) and len(result) == 4
    actions, logp, value, extra = result
    assert actions.dtype == torch.int64
    assert tuple(actions.shape) == (3, 4)
    assert torch.all(actions[:, 1] == 1)  # shooting always on
    assert torch.all(actions[:, 3] == 0)  # bombing left to a rule
    assert torch.all(actions[:, 0] >= 0) and torch.all(actions[:, 0] < 9)
    assert torch.all(actions[:, 2] >= 0) and torch.all(actions[:, 2] < 2)
    assert tuple(value.shape) == (3,)
    assert tuple(logp.shape) == (3,) and tuple(extra.shape) == (3,)


def test_hud_branches_are_independent():
    model = VisionScorerV2()
    assert model.hud_score is not model.hud_status

    score_pointers = {parameter.data_ptr() for parameter in model.hud_score.parameters()}
    status_pointers = {parameter.data_ptr() for parameter in model.hud_status.parameters()}
    assert score_pointers and status_pointers
    assert score_pointers.isdisjoint(status_pointers)

    keys = list(model.state_dict())
    assert any(key.startswith("hud_score.") for key in keys)
    assert any(key.startswith("hud_status.") for key in keys)
    assert any(key.startswith("playfield.") for key in keys)
    # Same conv architecture, but each exit is sized by its own pool cells.
    score_linear = next(module for module in model.hud_score.modules() if isinstance(module, torch.nn.Linear))
    status_linear = next(module for module in model.hud_status.modules() if isinstance(module, torch.nn.Linear))
    width = model.config.hud_width
    assert score_linear.in_features == width * 3 * 2 * 8  # (2, 8) pool
    assert status_linear.in_features == width * 3 * 8 * 4  # (8, 4) pool


def test_heatmap_target_peaks_at_label():
    target = heatmap_target(np.array([0.5]), np.array([0.5]), 28, 24)
    assert target.shape == (1, 28 * 24)
    assert target.dtype == np.float32
    row, column = np.unravel_index(int(target.argmax()), (28, 24))
    assert abs(row - 13.5) <= 1.0
    assert abs(column - 11.5) <= 1.0
    assert np.allclose(target.sum(axis=1), 1.0, atol=1e-5)

    # Positions on the border must not produce NaN or a degenerate distribution.
    edges = heatmap_target(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 1.0]), 28, 24)
    assert np.isfinite(edges).all()
    assert np.allclose(edges.sum(axis=1), 1.0, atol=1e-5)

    # Batch of many labels keeps the flat layout row-major over (grid_h, grid_w).
    single = heatmap_target(np.array([0.5]), np.array([0.5]), 28, 24)
    stacked = heatmap_target(np.array([0.5, 0.5]), np.array([0.5, 0.5]), 28, 24)
    assert np.allclose(stacked[0], single[0])
    assert np.allclose(stacked[1], single[0])


def test_decode_position_recovers_peak():
    logits = torch.full((1, 28, 24), -20.0)
    logits[0, 10, 7] = 20.0
    rx, ry = decode_position(logits)
    assert tuple(rx.shape) == (1,) and tuple(ry.shape) == (1,)
    assert abs(float(ry[0]) - (10.5) / 28) < 1e-4
    assert abs(float(rx[0]) - (7.5) / 24) < 1e-4


def test_decode_position_is_smooth():
    left = torch.full((1, 28, 24), -20.0)
    left[0, 10, 7] = 20.0
    right = torch.full((1, 28, 24), -20.0)
    right[0, 10, 8] = 20.0
    both = left.clone()
    both[0, 10, 8] = 20.0

    rx_left, _ = decode_position(left)
    rx_right, _ = decode_position(right)
    rx_both, ry_both = decode_position(both)

    expected = ((7 + 8) / 2 + 0.5) / 24
    assert abs(float(rx_both) - expected) < 1e-4
    assert float(rx_left) < float(rx_both) < float(rx_right)
    assert abs(float(ry_both) - (10.5) / 28) < 1e-4


def test_checkpoint_dispatch_and_backward_compat(tmp_path):
    config = VisionNetConfigV2()
    model = VisionScorerV2(config).eval()
    path = tmp_path / "vision-v2.pt"
    save_vision_checkpoint(path, model, {"stage": "state"})

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["kind"] == "vision_scorer"
    assert payload["arch"] == "v2"
    assert payload["frame"]["H"] == vision.VISION_H
    assert payload["frame"]["W"] == vision.VISION_W
    assert payload["frame"]["depth"] == config.depth

    restored, extra = load_vision_checkpoint(path)
    assert extra["stage"] == "state"
    assert isinstance(restored, VisionScorerV2)
    assert restored.config.playfield_hw == (224, 192)
    assert all(
        isinstance(getattr(restored.config, field), tuple)
        for field in ("playfield_hw", "hud_score_hw", "hud_status_hw")
    )
    batch = _regions(config, batch=1)
    with torch.no_grad():
        assert torch.allclose(model.score(batch), restored.score(batch), atol=1e-6)

    # A config that went through json (tuples come back as lists) must load too.
    json_path = tmp_path / "vision-v2-json.pt"
    json_payload = dict(payload)
    json_payload["config"] = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in payload["config"].items()
    }
    torch.save(json_payload, json_path)
    json_model, _ = load_vision_checkpoint(json_path)
    assert json_model.config.playfield_hw == (224, 192)
    with torch.no_grad():
        assert torch.allclose(model.score(batch), json_model.score(batch), atol=1e-6)

    # A pre-v1.0.3 v1 payload has no arch key and must still load as v1.
    legacy_config = VisionNetConfig(depth=1, width=8, trunk_hidden=32)
    legacy_path = tmp_path / "vision-legacy.pt"
    torch.save(
        {
            "kind": "vision_scorer",
            "state_dict": VisionScorer(legacy_config).state_dict(),
            "config": legacy_config.__dict__,
            "frame": {"H": vision.VISION_H, "W": vision.VISION_W, "C": 3, "depth": 1},
            "extra": {"stage": "state"},
        },
        legacy_path,
    )
    legacy_model, legacy_extra = load_vision_checkpoint(legacy_path)
    assert legacy_extra["stage"] == "state"
    assert isinstance(legacy_model, VisionScorer)
    assert not isinstance(legacy_model, VisionScorerV2)

    # Unknown arch must fail loudly instead of being coerced to v1.
    unknown_path = tmp_path / "vision-unknown.pt"
    torch.save(
        {
            "kind": "vision_scorer",
            "arch": "v9",
            "state_dict": VisionScorer(legacy_config).state_dict(),
            "config": legacy_config.__dict__,
            "frame": {"H": vision.VISION_H, "W": vision.VISION_W, "C": 3, "depth": 1},
            "extra": {},
        },
        unknown_path,
    )
    with pytest.raises(ValueError, match="unknown vision arch"):
        load_vision_checkpoint(unknown_path)


def test_depth3_input_channels():
    config = VisionNetConfigV2(depth=3)
    model = VisionScorerV2(config).eval()
    batch = _regions(config, batch=2)
    assert tuple(batch["playfield"].shape) == (2, 3, 224, 192, 3)
    with torch.no_grad():
        values = model.score(batch)
        state = model.decode_state(batch)
    assert tuple(values.shape) == (2, N_CANDIDATES)
    assert tuple(state["self_feat"].shape) == (2, obs.N_SELF)
    assert tuple(state["heatmap_logits"].shape) == (2, 28, 24)
    assert int(model.playfield.body[0].in_channels) == 3 * 3


def test_danger_head_regional_alignment():
    config = VisionNetConfigV2(depth=1)
    model = VisionScorerV2(config).eval()
    batch = _regions(config, batch=2)

    state = model.decode_state(batch)
    danger_log = state["danger_log"]
    assert tuple(danger_log.shape) == (2, config.danger_rows * config.danger_cols)
    assert torch.isfinite(danger_log).all()
    danger_map = danger_log.reshape(2, config.danger_rows, config.danger_cols)
    assert tuple(danger_map.shape) == (2, 16, 8)

    # The danger grid comes from AdaptiveAvgPool2d: its cell count follows the
    # config, so a change of playfield size cannot silently change the target.
    pooled = model.playfield.danger_pool(torch.zeros(1, 1, *config.heatmap_hw))
    assert tuple(pooled.shape) == (1, 1, config.danger_rows, config.danger_cols)

    # Both loss paths must be differentiable.  A single path cannot give every
    # parameter a gradient (the state loss never touches the score head), so the
    # combined loss is the case where "all parameters" is meaningful, and each
    # path is then checked on the parameters it actually uses.
    truth = np.zeros((2, config.danger_rows, config.danger_cols), dtype=np.float32)
    truth[:, 8, 4] = 8.0
    target = torch.as_tensor(np.log1p(truth / 8.0))
    state_loss = torch.nn.functional.smooth_l1_loss(danger_map, target)
    values = model.score(batch)
    policy_loss = torch.nn.functional.smooth_l1_loss(values, torch.zeros_like(values))
    # Cross-entropy against the normalised gaussian target, with the target built
    # by the training-side helper: this also pins that heatmap_target's row-major
    # layout matches heatmap_logits.flatten(1).
    heat_target = torch.as_tensor(
        np.repeat(heatmap_target(np.array([0.5]), np.array([0.5]), 28, 24), 2, axis=0)
    )
    heat_loss = -(
        heat_target * torch.log_softmax(state["heatmap_logits"].flatten(1), dim=-1)
    ).sum(-1).mean()
    # The self head regresses self_feat[2:24] only; 0:2 is driven by the heatmap.
    self_loss = torch.nn.functional.smooth_l1_loss(
        state["self_feat"][:, 2:], torch.zeros(2, obs.N_SELF - 2)
    )
    (state_loss + self_loss + policy_loss + heat_loss).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name

    model.zero_grad(set_to_none=True)
    state_out = model.decode_state(batch)
    (state_out["danger_log"].sum() + state_out["self_feat"].sum()).backward()
    for name, parameter in model.named_parameters():
        if name.startswith(STATE_PATH_PREFIXES):
            assert parameter.grad is not None, name

    model.zero_grad(set_to_none=True)
    model.score(batch).sum().backward()
    for name, parameter in model.named_parameters():
        if name.startswith(POLICY_PATH_PREFIXES):
            assert parameter.grad is not None, name


def test_v2_error_behaviour():
    config = VisionNetConfigV2()
    model = VisionScorerV2(config)
    batch = _regions(config, batch=1)
    partial = {"playfield": batch["playfield"], "hud_score": batch["hud_score"]}
    with pytest.raises(KeyError) as excinfo:
        model.decode_state(partial)
    assert excinfo.value.args[0] == "missing regions ['hud_status']; got ['hud_score', 'playfield']"

    no_state = VisionScorerV2(VisionNetConfigV2(depth=1, state_head=False))
    with pytest.raises(RuntimeError, match="this model was built with state_head=False"):
        no_state.decode_state(_regions(no_state.config, batch=1))
    with torch.no_grad():
        assert tuple(no_state.score(_regions(no_state.config, batch=1)).shape) == (1, N_CANDIDATES)


def test_v1_param_count_unchanged():
    """Guard: v2 must be an addition, not a change to v1 (design doc §3.6)."""
    assert VisionScorer(VisionNetConfig(depth=1)).num_parameters() == 707_129


# --------------------------------------------------------------------------- #
# v1.0.4 C6: checkpoint arch dispatch, v3 payload contract, deployment labels
# --------------------------------------------------------------------------- #
def _v3_extra(**overrides) -> dict:
    """The five C6 identity keys plus the loader's metadata."""
    extra = {
        "source_state_sha": "",
        "teacher_fingerprint": "a" * 64,
        "manifest_sha": "b" * 64,
        "val_index_sha256": "c" * 64,
        "loss_config": {"objective": "rank", "rank_delta_px": 0.5},
        "stage": "state",
        "epoch": 1,
        "depth": 3,
        "history_contract": "causal-segment-v1",
    }
    extra.update(overrides)
    return extra


def _valid_v3_payload() -> dict:
    """A payload built from the *real* S3-5 config layout, with dummy weights.

    S3-5 serialises ``playfield_hw``/``hud_score_hw``/``hud_status_hw`` (no
    ``region_sizes`` field, no ``state_head`` switch), which is what
    ``VisionSpatialTemporalConfig().__dict__`` contains; the validator derives the
    ROI sizes from those fields.
    """
    config = dict(VisionSpatialTemporalConfig().__dict__)
    return {
        "schema": "vision-checkpoint/3",
        "kind": "vision_scorer",
        "arch": "v3-st",
        "config": config,
        # ``frame`` is the legacy H128/W112 marker only (C6); the real ROI is config.
        "frame": {"H": vision.VISION_H, "W": vision.VISION_W, "C": 3, "depth": 3},
        "state_dict": {"weight": torch.zeros(1)},
        "extra": _v3_extra(),
    }


def _v3_regions(config, batch: int = 1) -> dict:
    return vision.regions_to_batch(
        {
            name: np.zeros((batch, config.depth, height, width, 3), dtype=np.uint8)
            for name, (height, width) in config.region_sizes.items()
        }
    )


def test_vision_arch_reports_the_actual_model_type():
    """C6: v1/v2 are named from the loaded type; anything else is an error."""
    assert vision_arch(VisionScorer(VisionNetConfig(depth=1, width=8, trunk_hidden=32))) == "v1"
    assert vision_arch(VisionScorerV2(VisionNetConfigV2(depth=1))) == "v2"

    class UnknownModel(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="v104:CHECKPOINT: unknown vision model type"):
        vision_arch(UnknownModel())


def test_v3_payload_contract_validation():
    payload = _valid_v3_payload()
    contract = validate_v3_payload(payload)
    assert contract["schema"] == V3_CHECKPOINT_SCHEMA
    assert contract["arch"] == "v3-st"
    assert contract["depth"] == 3
    assert contract["candidate_order_version"] == "focus9-move-v1"
    assert contract["score_semantics"] == "preference_logits"
    assert contract["roi_native_hw"] == (128, 128)
    assert contract["local_grid_hw"] == (16, 16)
    assert contract["align_corners"] is False
    assert contract["history_contract"] == "causal-segment-v1"
    assert contract["region_sizes"] == {
        "playfield": [224, 192],
        "hud_score": [34, 216],
        "hud_status": [126, 216],
    }
    # ``frame`` is recorded as a legacy marker, never as the model input.
    assert contract["legacy_frame"] == {"H": vision.VISION_H, "W": vision.VISION_W, "C": 3, "depth": 3}
    assert set(contract["identity"]) == set(V3_REQUIRED_EXTRA)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda payload: payload.pop("schema"), "missing the 'schema' key"),
        (lambda payload: payload.update(schema="vision-checkpoint/2"), "schema"),
        (lambda payload: payload.update(kind="state_dict_bundle"), "kind"),
        (lambda payload: payload.update(arch="v9"), "arch"),
        (lambda payload: payload["extra"].pop("teacher_fingerprint"), "missing \\['teacher_fingerprint'\\]"),
        (lambda payload: payload["extra"].update(source_state_sha="xyz"), "source_state_sha"),
        (lambda payload: payload["extra"].update(manifest_sha="not-a-sha"), "manifest_sha"),
        (lambda payload: payload["extra"].update(val_index_sha256=""), "val_index_sha256"),
        (lambda payload: payload["extra"].update(teacher_fingerprint=""), "teacher_fingerprint"),
        (lambda payload: payload["extra"].update(loss_config={}), "loss_config"),
        (lambda payload: payload["config"].update(depth=1), "depth"),
        (lambda payload: payload["config"].update(candidate_order_version="v1"), "candidate_order_version"),
        (lambda payload: payload["config"].update(score_semantics="value"), "score_semantics"),
        (lambda payload: payload["config"].update(align_corners=True), "align_corners"),
        (lambda payload: payload["config"].update(roi_native_hw=[64, 64]), "roi_native_hw"),
        (lambda payload: payload["config"].update(local_grid_hw=[8, 8]), "local_grid_hw"),
        (lambda payload: payload["config"].update(history_contract="unknown"), "history_contract"),
        (lambda payload: payload["config"].update(channels=4), "channels"),
        (lambda payload: payload["config"].update(state_head=False), "state_head"),
        (
            lambda payload: payload["config"].update(
                region_sizes={"playfield": [224, 192]}
            ),
            "region_sizes",
        ),
        (lambda payload: payload.update(state_dict={}), "state_dict"),
    ],
)
def test_v3_payload_rejects_broken_contract(mutation, expected):
    payload = _valid_v3_payload()
    mutation(payload)
    with pytest.raises(ValueError, match=f"v104:CHECKPOINT.*{expected}"):
        validate_v3_payload(payload)


def test_v3_policy_checkpoint_needs_a_real_source_state_sha():
    payload = _valid_v3_payload()
    payload["extra"]["stage"] = "policy"
    with pytest.raises(ValueError, match="policy-stage v3 checkpoint must record a real source_state_sha"):
        validate_v3_payload(payload)

    payload["extra"]["source_state_sha"] = "d" * 64
    contract = validate_v3_payload(payload)
    assert contract["identity"]["source_state_sha"] == "d" * 64


def test_v3_checkpoint_round_trip_and_arch_dispatch(tmp_path):
    """G-v3 approved: a v3 checkpoint saves, strictly loads and predicts identically.

    This replaces the previous "v3 must be blocked" assertion.  What it pins is the
    C6 file contract, not policy quality: explicit ``vision-checkpoint/3`` payload,
    the five identity keys, ``arch='v3-st'`` from the loaded type, strict weights
    and a bit-identical ``score``/working ``act`` after the round trip.
    """
    config = VisionSpatialTemporalConfig()
    model = VisionSpatialTemporalPolicy(config).eval()
    assert model.num_parameters() == 1_507_162
    assert config.depth == 3 and config.heatmap_hw == (56, 48)

    path = tmp_path / "v3.pt"
    extra = _v3_extra()
    save_vision_checkpoint(path, model, extra)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["schema"] == V3_CHECKPOINT_SCHEMA
    assert payload["kind"] == "vision_scorer"
    assert payload["arch"] == "v3-st"
    assert set(V3_REQUIRED_EXTRA) <= set(payload["extra"])
    # ``frame`` is the historical H128/W112 marker; the ROI comes from ``config``.
    assert payload["frame"] == {"H": vision.VISION_H, "W": vision.VISION_W, "C": 3, "depth": 3}
    assert payload["config"]["depth"] == 3
    assert payload["config"]["playfield_hw"] == (224, 192)
    contract = validate_v3_payload(payload)
    assert contract["roi_native_hw"] == (128, 128)
    assert contract["local_grid_hw"] == (16, 16)

    restored, restored_extra = load_vision_checkpoint(path)
    assert isinstance(restored, VisionSpatialTemporalPolicy)
    assert vision_arch(model) == "v3-st"
    assert vision_arch(restored) == "v3-st"
    assert restored_extra == extra
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name

    batch = _v3_regions(config, batch=1)
    with torch.no_grad():
        before = model.score(batch)
        after = restored.score(batch)
        actions, logp, value, placeholder = restored.act(batch)
    assert torch.equal(before, after)
    assert tuple(after.shape) == (1, N_CANDIDATES)
    assert actions.dtype == torch.int64 and tuple(actions.shape) == (1, 4)
    assert int(actions[0, 1]) == 1 and int(actions[0, 3]) == 0
    assert tuple(logp.shape) == (1,) and tuple(placeholder.shape) == (1,)
    assert tuple(value.shape) == (1,)


def test_v3_save_refuses_a_checkpoint_without_identity(tmp_path):
    """C6: a v3 file whose ``extra`` lacks the fingerprints must not be written."""
    model = VisionSpatialTemporalPolicy(VisionSpatialTemporalConfig())
    path = tmp_path / "bad-v3.pt"
    with pytest.raises(ValueError, match="v104:CHECKPOINT.*extra is missing"):
        save_vision_checkpoint(path, model, {"stage": "state"})
    assert not path.exists()


def test_v3_contract_node_is_v3_only():
    model = VisionScorerV2(VisionNetConfigV2(depth=1))
    with pytest.raises(ValueError, match="vision_contract_node is v3-only"):
        vision_contract_node(model, {})


def test_v2_depth3_checkpoint_roundtrip(tmp_path):
    config = VisionNetConfigV2(depth=3)
    model = VisionScorerV2(config).eval()
    path = tmp_path / "v2-depth3.pt"
    save_vision_checkpoint(path, model, {"stage": "state", "history_contract": "causal-segment-v1"})

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["arch"] == "v2"
    # v1/v2 keep the v1.0.3 payload shape; only v3 adds the explicit schema.
    assert "schema" not in payload
    assert payload["frame"] == {"H": vision.VISION_H, "W": vision.VISION_W, "C": 3, "depth": 3}

    restored, extra = load_vision_checkpoint(path)
    assert isinstance(restored, VisionScorerV2)
    assert restored.config.depth == 3
    assert extra["history_contract"] == "causal-segment-v1"
    batch = _regions(config, batch=1)
    with torch.no_grad():
        assert torch.allclose(model.score(batch), restored.score(batch), atol=1e-6)


def test_checkpoint_frame_is_a_legacy_marker_not_the_model_input(tmp_path):
    """C6: v2's real ROI is ``config.region_sizes``; ``frame`` stays H128/W112."""
    config = VisionNetConfigV2(depth=1, playfield_hw=(64, 48))
    model = VisionScorerV2(config)
    path = tmp_path / "v2-small.pt"
    save_vision_checkpoint(path, model)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["frame"]["H"] == vision.VISION_H
    assert payload["frame"]["W"] == vision.VISION_W
    assert payload["config"]["playfield_hw"] == (64, 48)
    restored, _extra = load_vision_checkpoint(path)
    assert restored.config.region_sizes["playfield"] == (64, 48)


def test_load_rejects_a_payload_without_a_state_dict(tmp_path):
    payload = {
        "kind": "vision_scorer",
        "arch": "v1",
        "config": VisionNetConfig(depth=1, width=8, trunk_hidden=32).__dict__,
        "frame": {"H": vision.VISION_H, "W": vision.VISION_W, "C": 3, "depth": 1},
        "extra": {},
    }
    path = tmp_path / "broken.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="v104:CHECKPOINT.*state_dict"):
        load_vision_checkpoint(path)


def test_load_rejects_a_frame_that_is_not_this_build(tmp_path):
    payload = {
        "kind": "vision_scorer",
        "arch": "v1",
        "state_dict": VisionScorer(VisionNetConfig(depth=1, width=8, trunk_hidden=32)).state_dict(),
        "config": VisionNetConfig(depth=1, width=8, trunk_hidden=32).__dict__,
        "frame": {"H": 64, "W": 64, "C": 3, "depth": 1},
        "extra": {},
    }
    path = tmp_path / "wrong-frame.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="v104:CHECKPOINT.*frame size 64x64"):
        load_vision_checkpoint(path)

