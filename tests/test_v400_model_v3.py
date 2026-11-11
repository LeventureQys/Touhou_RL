"""T4 tests: the obs-v3 student **v3** (architecture, losses, checkpoints, trainer).

The T4 acceptance list (tasks/2026-11-10-P3重启任务书.md §4 T4) is what this file
locks, in order:

* **shapes**: ``play (B,4,112,96)``, ``hud (B,56,160)`` (and the ``(B,1,56,160)``
  spelling obs-v1 uses), ``act_hist (B,8,18)`` (and its ``(B,144)`` flattening) go
  in; ``move (B,18)``, ``bomb (B,2)``, ``value (B,)`` and the aux grids come out;
* **parameter budget**: the canonical architecture is inside the pre-registered
  2-6 M band, and :func:`model_v3.check_parameter_budget` fails outside it;
* **no global average pooling**: no ``AdaptiveAvgPool2d``/``AdaptiveMaxPool2d``
  anywhere in the model tree (reflective traversal, not a source grep), and the
  play encoder's output really is spatial;
* **retained space works**: two inputs that a global average pool cannot tell
  apart (pooled-cosine 0.999) are separated by the spatial readout;
* **losses**: the aux term is skipped with weight 0.0 when its labels are absent,
  and the effective weights of KL/bomb/aux/value are exactly 13-P3 §2.4's
  ``1.0 / 1.0 / 0.5 / 0.25`` when every target is present;
* **checkpoints**: a v3 checkpoint round-trips, records ``arch="v3"`` and the
  frozen obs-v3 digest, and fails closed on another spec/arch; the v1 and v2
  checkpoints still load through their own loaders (no regression);
* **the trainer runs end to end** on a synthetic D1 corpus (4 contract-valid
  obs-v3 shards, built through the same collector the real D1 uses) with and
  without aux labels;
* **the two CLI tools route** ``--obs v3`` / ``--arch v3`` to the v3 path without
  changing the pre-T4 routes.

Everything here is synthetic: no engine, no real D1, no training run.
"""

from __future__ import annotations

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
    model,
    model_v2,
    model_v3,
    obs_spec,
    obs_spec_v2,
    obs_spec_v3,
    obs_v3_dataset,
    train_v3,
)
from touhou_rl.v400.labels import spec as label_spec  # noqa: E402

#: The frozen obs-v3 digest (T2 closeout, D-14), duplicated as a literal so an
#: accidental spec edit breaks a T4 test instead of a training run.
FROZEN_V3_DIGEST = "ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105"
FRAMES_PER_DECISION = dual.DEFAULT_FRAMES_PER_DECISION
DECISIONS = 6


# ---------------------------------------------------------------------------
# models under test
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def canonical() -> model_v3.StudentV3:
    """The pre-registered architecture (2.6 M parameters), in eval mode."""

    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    net = model_v3.StudentV3()
    net.eval()
    return net


def tiny_model() -> "model_v3.StudentV3":
    """A small variant with the same wiring, so tests do not pay 2.6 M parameters."""

    return model_v3.StudentV3(stage_channels=(4, 8, 8), trunk=32, readout_channels=2,
                              hud_width=2, hud_hidden=16, hist_hidden=8,
                              aux_channels=(8, 8, 8), residual_blocks=1)


def obs_batch(batch: int = 2, *, seed: int = 0) -> dict:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(seed)
    return {
        "play": torch.rand(batch, obs_spec_v3.FRAME_STACK, *obs_spec_v3.PLAY_SHAPE,
                           generator=generator),
        "hud": torch.rand(batch, obs_spec_v3.HUD_SHAPE[0], obs_spec_v3.HUD_SHAPE[1],
                          generator=generator),
        "act_hist": torch.zeros(batch, *obs_spec_v3.ACTION_HISTORY_SHAPE),
    }


# ---------------------------------------------------------------------------
# 1. shapes
# ---------------------------------------------------------------------------
def test_input_output_shapes(canonical):
    torch = pytest.importorskip("torch")
    batch = obs_batch(2)
    with torch.no_grad():
        out = canonical(batch["play"], batch["hud"], batch["act_hist"], aux=True)
    assert out.move.shape == (2, obs_spec.N_MOVE_CLASSES) == (2, 18)
    assert out.bomb.shape == (2, obs_spec.N_BOMB_CLASSES) == (2, 2)
    assert out.value.shape == (2,)
    # The 3-D HUD spelling of the T4 card and obs-v1's (B,1,56,160) both work.
    with torch.no_grad():
        four_d = canonical(batch["play"], batch["hud"].unsqueeze(1), batch["act_hist"])
        flat_hist = canonical(batch["play"], batch["hud"],
                              batch["act_hist"].reshape(2, -1))
    assert torch.allclose(out.move, four_d.move)
    assert torch.allclose(out.move, flat_hist.move)
    assert out.aux is not None
    assert out.aux["occupancy"].shape == (2, 1, *model_v3.AUX_SHAPE)
    assert out.aux["self_heatmap"].shape == (2, 1, *model_v3.AUX_SHAPE)
    assert out.aux["danger"].shape == (2, len(model_v3.AUX_HORIZONS), *model_v3.AUX_SHAPE)


def test_aux_head_grid_is_the_label_grid(canonical):
    """The aux output grid must be the label generator's 1/4-resolution grid."""

    assert model_v3.AUX_SHAPE == (label_spec.GRID_HEIGHT, label_spec.GRID_WIDTH)
    assert model_v3.AUX_SHAPE == obs_spec_v3.PLAY_SHAPE  # both are the 4x4 block mean
    assert tuple(model_v3.AUX_HORIZONS) == tuple(label_spec.HORIZON_FRAMES) == (4, 8, 16)
    assert canonical.perception.out_shape == model_v3.AUX_SHAPE
    assert canonical.config()["aux_shape"] == [112, 96]


def test_wrong_shapes_fail_closed(canonical):
    torch = pytest.importorskip("torch")
    batch = obs_batch(2)
    with pytest.raises(model_v3.ModelV3Error, match="play"):
        canonical(torch.zeros(2, 4, 64, 64), batch["hud"], batch["act_hist"])
    with pytest.raises(model_v3.ModelV3Error, match="hud"):
        canonical(batch["play"], torch.zeros(2, 56, 96), batch["act_hist"])
    with pytest.raises(model_v3.ModelV3Error, match="act_hist"):
        canonical(batch["play"], batch["hud"], torch.zeros(2, 4, 18))
    # A 4-frame stack is required: 1 frame would silently broadcast into the conv.
    with pytest.raises(model_v3.ModelV3Error, match="play"):
        canonical(torch.zeros(2, 1, 112, 96), batch["hud"], batch["act_hist"])


# ---------------------------------------------------------------------------
# 2. parameter budget
# ---------------------------------------------------------------------------
def test_parameter_count_is_inside_the_preregistered_band(canonical):
    count = canonical.n_parameters()
    assert model_v3.PARAMETER_BUDGET == (2_000_000, 6_000_000)
    assert 2_000_000 <= count <= 6_000_000, f"{count:,} parameters is outside 2-6 M"
    # The number the T4 card quotes, pinned so a change is a deliberate one.
    assert count == 2_602_490
    assert canonical.is_canonical() is True
    assert model_v3.check_parameter_budget(canonical) == count
    tiny = tiny_model()
    assert tiny.is_canonical() is False
    with pytest.raises(model_v3.ParameterBudgetError):
        model_v3.check_parameter_budget(tiny)


# ---------------------------------------------------------------------------
# 3. space is retained
# ---------------------------------------------------------------------------
def test_no_adaptive_pool_in_the_model(canonical):
    """Reflective check: a future submodule cannot reintroduce the defect."""

    import torch.nn as nn

    assert model_v3.adaptive_pool_layers(canonical) == []
    assert model_v3.adaptive_pool_layers(canonical.play) == []
    assert not any(isinstance(child, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d))
                   for child in canonical.modules())
    # The play stream really does end on a spatial map, not a 1x1 summary.
    torch = pytest.importorskip("torch")
    with torch.no_grad():
        features = canonical.play(torch.zeros(2, 4, 112, 96))
    assert features.shape == (2, 64, 14, 12)
    assert canonical.readout.flat_features == 16 * 14 * 12 == 2688
    # ... and the readout is a linear map over those 2688 cells, not an average.
    assert canonical.readout.linear[1].in_features == 2688


def test_spatial_readout_separates_what_average_pooling_cannot(canonical):
    """Move one bright square from the left to the right of the playfield.

    A global average pool sees almost the same vector in both cases (the cosine
    below), which is exactly why every pre-T4 student could not tell "bullet on
    the left" from "bullet on the right".  The spatial readout does separate them.
    """

    torch = pytest.importorskip("torch")
    play = torch.zeros(2, 4, 112, 96)
    play[0, :, 20:40, 10:30] = 1.0    # left
    play[1, :, 20:40, 66:86] = 1.0    # same square, right
    hud = torch.zeros(2, 56, 160)
    hist = torch.zeros(2, 8, 18)
    with torch.no_grad():
        features = canonical.play(play)
        pooled = features.mean(dim=(2, 3))
        readout = canonical.readout(features)
        out = canonical(play, hud, hist)
    cosine = torch.nn.functional.cosine_similarity(pooled[0], pooled[1], dim=0)
    relative = float((readout[0] - readout[1]).abs().mean()
                     / (readout[0].abs().mean() + 1e-9))
    assert float(cosine) > 0.99, "the pooled summary was expected to be blind here"
    assert relative > 0.10, (
        f"the spatial readout only moved {relative:.3f} between two positions; the "
        "position information is not being used")
    assert not torch.allclose(out.move[0], out.move[1], atol=1e-6)


def test_hud_and_action_history_are_independent_inputs():
    torch = pytest.importorskip("torch")
    net = tiny_model()
    net.eval()
    base = obs_batch(2)
    with torch.no_grad():
        reference = net(base["play"], base["hud"], base["act_hist"]).move
        hud_changed = net(base["play"], base["hud"] + 0.5, base["act_hist"]).move
        hist = base["act_hist"].clone()
        hist[:, 0, 7] = 1.0
        hist_changed = net(base["play"], base["hud"], hist).move
    assert not torch.allclose(reference, hud_changed)
    assert not torch.allclose(reference, hist_changed)
    # The HUD stream flattens its 32x7x20 map (4480 features), it does not pool it.
    assert net.hud.flat_features == net.hud.out_channels * 7 * 20
    assert net.hud.flat_features > net.hud.out_channels


# ---------------------------------------------------------------------------
# 4. losses
# ---------------------------------------------------------------------------
def test_teacher_soft_labels_are_tau_scaled_scores():
    torch = pytest.importorskip("torch")
    scores = torch.zeros(3, obs_spec.N_MOVE_CLASSES)
    scores[:, 5] = 4.0
    probs = model_v3.teacher_soft_labels(scores, 2.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3))
    assert probs.argmax(dim=-1).tolist() == [5, 5, 5]
    assert model_v3.teacher_soft_labels(scores, 1.0).max() > probs.max()
    with pytest.raises(model_v3.ModelV3Error, match="tau"):
        model_v3.teacher_soft_labels(scores, 0.0)
    with pytest.raises(model_v3.ModelV3Error, match="scores"):
        model_v3.teacher_soft_labels(torch.zeros(3, 17), 1.0)


def test_aux_loss_is_skipped_with_zero_weight_without_labels():
    torch = pytest.importorskip("torch")
    net = tiny_model()
    outputs = net(*obs_batch(2).values(), aux=True)
    for targets in (None, {}):
        loss, report = model_v3.perception_aux_loss(outputs.aux, targets)
        assert loss is None
        assert report["active"] == []
        assert set(report["effective_weights"]) == set(model_v3.AUX_COMPONENTS)
        assert all(value == 0.0 for value in report["effective_weights"].values())
        assert report["skipped"], "a skipped head must say why"
    # ... and only the supplied heads are trained.
    occupancy = torch.zeros(2, *model_v3.AUX_SHAPE)
    loss, report = model_v3.perception_aux_loss(outputs.aux, {"occupancy": occupancy})
    assert report["active"] == ["occupancy"]
    assert report["effective_weights"]["occupancy"] == 1.0
    assert report["effective_weights"]["self_heatmap"] == 0.0
    assert report["effective_weights"]["danger"] == 0.0
    assert torch.isfinite(loss)
    net.zero_grad()
    loss.backward()
    assert float(net.perception.occupancy.weight.grad.abs().sum()) > 0.0


def test_student_v3_loss_effective_weights_match_the_spec():
    torch = pytest.importorskip("torch")
    net = tiny_model()
    batch = obs_batch(3, seed=4)
    labels = {"scores": torch.randn(3, obs_spec.N_MOVE_CLASSES),
              "bomb": torch.tensor([0, 1, 0])}
    outputs = net(batch["play"], batch["hud"], batch["act_hist"], aux=True)
    loss, report = model_v3.student_v3_loss(outputs, labels, tau=2.0)
    assert report["effective_weights"] == {"kl": 1.0, "bomb": 1.0, "aux": 0.0,
                                           "value": 0.0}
    assert "aux" in report["skipped"] and "value" in report["skipped"]
    assert report["total"] == pytest.approx(
        report["terms"]["kl"] + report["terms"]["bomb"], rel=1e-5, abs=1e-6)
    with_labels = dict(labels)
    with_labels["aux"] = {"occupancy": torch.zeros(3, *model_v3.AUX_SHAPE),
                          "danger": torch.zeros(3, len(model_v3.AUX_HORIZONS),
                                                *model_v3.AUX_SHAPE)}
    with_labels["value"] = torch.rand(3)
    loss, report = model_v3.student_v3_loss(outputs, with_labels, tau=2.0)
    assert report["effective_weights"] == {"kl": 1.0, "bomb": 1.0, "aux": 0.5,
                                           "value": 0.25}
    assert report["aux"]["active"] == ["occupancy", "danger"]
    expected = (report["terms"]["kl"] + report["terms"]["bomb"]
                + 0.5 * report["aux"]["loss"] + 0.25 * report["terms"]["value"])
    assert report["total"] == pytest.approx(expected, rel=1e-5, abs=1e-6)
    with pytest.raises(model_v3.ModelV3Error, match="bomb"):
        model_v3.student_v3_loss(outputs, {"scores": labels["scores"]})
    with pytest.raises(model_v3.ModelV3Error, match="unknown aux"):
        model_v3.student_v3_loss(outputs, dict(labels, aux={"lives": torch.zeros(3)}))
    with pytest.raises(model_v3.ModelV3Error, match="channel"):
        model_v3.student_v3_loss(outputs, dict(labels, aux={
            "danger": torch.zeros(3, 1, *model_v3.AUX_SHAPE)}))


# ---------------------------------------------------------------------------
# 5. checkpoints
# ---------------------------------------------------------------------------
def test_checkpoint_round_trip_and_digest(tmp_path):
    torch = pytest.importorskip("torch")
    net = tiny_model()
    net.eval()
    path = model_v3.save_checkpoint_v3(tmp_path / "v3.pt", net,
                                       train={"tau": 2.0}, provenance={"unit": "t4"})
    assert model_v3.checkpoint_digest(path) == __import__("hashlib").sha256(
        path.read_bytes()).hexdigest()
    loaded, payload = model_v3.load_checkpoint_v3(path)
    assert payload["parameters"] == net.n_parameters()
    batch = obs_batch(2)
    with torch.no_grad():
        assert torch.allclose(net(*batch.values()).move, loaded(*batch.values()).move)


def test_checkpoint_records_arch_and_obs_spec(tmp_path):
    net = tiny_model()
    path = model_v3.save_checkpoint_v3(tmp_path / "v3.pt", net)
    payload = __import__("torch").load(path, map_location="cpu", weights_only=False)
    assert payload["schema"] == "v400-student-checkpoint-3"
    assert payload["arch"] == "v3"
    assert payload["model"]["arch"] == "v3"
    assert payload["obs_spec_version"] == "obs-v3"
    assert payload["obs_spec_digest"] == FROZEN_V3_DIGEST
    assert payload["obs_spec_digest"] == obs_spec_v3.obs_spec_v3_digest()
    assert payload["parameters"] == net.n_parameters()
    assert payload["parameter_budget"] == [2_000_000, 6_000_000]
    assert "trunk" in payload["model"] and "stage_channels" in payload["model"]
    assert "arch=v3" in model_v3.describe_checkpoint_v3(payload)


def test_checkpoint_fails_closed_on_another_obs_spec(tmp_path):
    torch = pytest.importorskip("torch")
    path = model_v3.save_checkpoint_v3(tmp_path / "v3.pt", tiny_model())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for field, value, match in (
        ("obs_spec_digest", "0" * 64, "digest"),
        ("obs_spec_version", "obs-v2", "version"),
    ):
        tampered = dict(payload)
        tampered[field] = value
        torch.save(tampered, path)
        with pytest.raises(obs_spec_v3.ObsSpecV3Mismatch, match=match):
            model_v3.load_checkpoint_v3(path)
    bad_arch = dict(payload, arch="mlp")
    torch.save(bad_arch, path)
    with pytest.raises(model_v3.ModelV3Error, match="arch"):
        model_v3.load_checkpoint_v3(path)
    bad_schema = dict(payload, schema="v400-student-checkpoint-2")
    torch.save(bad_schema, path)
    with pytest.raises(model_v3.ModelV3Error, match="schema"):
        model_v3.load_checkpoint_v3(path)
    # The parameter count recorded by the writer is re-derived, not trusted.
    lying = dict(payload, parameters=int(payload["parameters"]) + 1)
    torch.save(lying, path)
    with pytest.raises(model_v3.ModelV3Error, match="parameters"):
        model_v3.load_checkpoint_v3(path)


def test_v1_and_v2_checkpoints_still_load(tmp_path):
    """T4 may not change either frozen checkpoint contract (no regression)."""

    torch = pytest.importorskip("torch")
    v1 = model.StudentNet(width=16, hidden=32)
    path_v1 = model.save_checkpoint(tmp_path / "v1.pt", v1, train={"tau": 2.0})
    loaded_v1, payload_v1 = model.load_checkpoint(path_v1)
    assert payload_v1["obs_spec_version"] == "obs-v1"
    assert loaded_v1.n_parameters() == v1.n_parameters()
    assert payload_v1["obs_spec_digest"] == obs_spec.obs_spec_digest()

    v2 = model_v2.StudentNetV2(width=8, hidden=16, hist_hidden=8)
    path_v2 = model_v2.save_checkpoint_v2(tmp_path / "v2.pt", v2)
    loaded_v2, payload_v2 = model_v2.load_checkpoint_v2(path_v2)
    assert payload_v2["obs_spec_version"] == "obs-v2"
    assert loaded_v2.n_parameters() == v2.n_parameters()
    gru = model_v2.StudentGRU(width=8, hidden=16, hist_hidden=8, gru_input=16)
    path_gru = model_v2.save_checkpoint_v2(tmp_path / "gru.pt", gru)
    loaded_gru, payload_gru = model_v2.load_checkpoint_v2(path_gru)
    assert payload_gru["model"]["arch"] == "gru"
    assert loaded_gru.seq_len == gru.seq_len


def test_the_three_checkpoint_loaders_do_not_cross(tmp_path):
    torch = pytest.importorskip("torch")
    v3_path = model_v3.save_checkpoint_v3(tmp_path / "v3.pt", tiny_model())
    with pytest.raises((ValueError, model_v3.ModelV3Error)):
        model.load_checkpoint(v3_path)
    with pytest.raises((ValueError, model_v3.ModelV3Error)):
        model_v2.load_checkpoint_v2(v3_path)
    v1_path = model.save_checkpoint(tmp_path / "v1.pt", model.StudentNet(width=8, hidden=8))
    v2_path = model_v2.save_checkpoint_v2(
        tmp_path / "v2.pt", model_v2.StudentNetV2(width=8, hidden=8, hist_hidden=8))
    for other in (v1_path, v2_path):
        with pytest.raises((ValueError, model_v3.ModelV3Error)):
            model_v3.load_checkpoint_v3(other)


# ---------------------------------------------------------------------------
# 6. the trainer, end to end, on a synthetic D1
# ---------------------------------------------------------------------------
def _sha(index: int) -> str:
    return f"{index:016x}"


def _label_rows(count: int = DECISIONS) -> list[dict]:
    """``load_labels``-shaped rows, one distinct executed action per decision."""

    rows = []
    for index in range(count):
        pre = index * FRAMES_PER_DECISION
        rows.append({
            "decision": index,
            "frame": pre + FRAMES_PER_DECISION,
            "frame_pre": pre,
            "hash_pre": {"hash": _sha(1000 + pre), "frames": pre},
            "frame_post": pre + FRAMES_PER_DECISION,
            "hash_post": {"hash": _sha(1000 + pre + FRAMES_PER_DECISION),
                          "frames": pre + FRAMES_PER_DECISION},
            "step_frames": [1, 2, 4, 8],
            "scores": [float(value) for value in range(obs_spec.N_MOVE_CLASSES)],
            "verified_mask": [bool(value % 2) for value in range(obs_spec.N_MOVE_CLASSES)],
            "chosen_plan": {"first_action": (index * 3) % obs_spec.N_MOVE_CLASSES,
                            "slots": [1, 2]},
            "margin": 0.5 + index,
            "bomb": 0,
            "stage": 0,
            "difficulty": 1,
            "frame_event_flags": 0,
            "shoot_release": False,
            "bomb_death_available": True,
        })
    return rows


def _synthetic_pixels(count: int = DECISIONS) -> dict[str, np.ndarray]:
    play = np.zeros((count, obs_spec_v3.FRAME_STACK, *obs_spec_v3.PLAY_SHAPE), np.uint8)
    hud = np.zeros((count, *obs_spec_v3.HUD_SHAPE), np.uint8)
    render_frames = np.zeros((count, obs_spec_v3.FRAME_STACK), np.int64)
    pads = np.zeros(count, np.int64)
    decision_frames = np.zeros(count, np.int64)
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
    return {"play": play, "hud": hud, "render_frames": render_frames, "pads": pads,
            "decision_frames": decision_frames}


def _alignment_doc(rows: list[dict]) -> dict:
    pairs = []
    for row in rows:
        for boundary, frame, checkpoint in (("pre", row["frame_pre"], row["hash_pre"]),
                                            ("post", row["frame_post"], row["hash_post"])):
            pairs.append({"decision": row["decision"], "boundary": boundary,
                          "frame": frame, "shadow": checkpoint["hash"],
                          "env": checkpoint["hash"], "env_frames": frame,
                          "equal": True, "source": "A/B"})
    return {
        "schema": "v400-d0-dual-alignment-1", "method": "hash-checkpoint",
        "decisions": len(rows), "boundaries": len(pairs),
        "checkpoints_equal": len(pairs), "checkpoints_mismatched": 0,
        "first_mismatch": None, "pairs": pairs, "pairs_total": len(pairs),
        "all_equal": True, "sha256": dual._canonical_sha256(pairs),
    }


def _channel_reports() -> tuple[dict, dict]:
    a = {"decisions": DECISIONS, "labels_sha256": "a" * 64, "wall_s": 1.0,
         "exit_code": 0, "engine_binary": "/build/rl_v3/th08-modern",
         "engine_sha256": "e" * 64, "channel": "A", "role": "shadow"}
    b = {"decisions": DECISIONS, "frames_read": 12, "pixel_ms_per_frame": 1.0,
         "wall_s": 1.0, "engine_binary": "/build/rl_v3/th08-modern",
         "engine_sha256": "e" * 64, "channel": "B", "role": "env"}
    return a, b


def build_d1(root: Path, *, shards: int = 4, count: int = DECISIONS,
             with_aux: bool = False) -> Path:
    """Four contract-valid obs-v3 shards written through the real collector path.

    ``dual.assemble`` is what produced D0/D1: using it means the fixture is a
    genuine obs-v3 shard (digest, sha256s, label arithmetic, ``act_hist`` rebuild),
    not a hand-written JSON that could drift from the contract.
    """

    for index in range(shards):
        key = f"v3-fixture-s1seed{100 + index}"
        rows = _label_rows(count)
        alignment_path = root / f"alignment-{key}.json"
        alignment_path.write_text(json.dumps(_alignment_doc(rows)), encoding="utf-8")
        args = SimpleNamespace(
            decisions=count, decisions_requested=count, decisions_actual=None,
            stage=0, difficulty=1, team=0, seed=100 + index, mode="fast", obs="v3",
            out_dir=str(root), alignment_json=str(alignment_path), shard_key=key,
            max_pairs=512, trim_transition=False)
        channel_a, channel_b = _channel_reports()
        summary = dual.assemble(args, rows, _synthetic_pixels(count),
                                channel_a, channel_b)
        shard_dir = Path(summary["dir"])
        if with_aux:
            aux_dir = shard_dir / "aux"
            aux_dir.mkdir(exist_ok=True)
            rng = np.random.default_rng(index)
            np.save(aux_dir / "occupancy.npy",
                    (rng.random((count, *model_v3.AUX_SHAPE)) > 0.9).astype(np.uint8))
            np.save(aux_dir / "self_heatmap.npy",
                    np.zeros((count, *model_v3.AUX_SHAPE), np.uint8))
            np.save(aux_dir / "danger.npy",
                    np.zeros((count, len(model_v3.AUX_HORIZONS),
                              *model_v3.AUX_SHAPE), np.uint8))
    return root


def _train_argv(root: Path, tmp_path: Path, *, max_steps: int = 3) -> list[str]:
    return [
        "--d1-root", str(root),
        "--checkpoint", str(tmp_path / "checkpoint-v3.pt"),
        "--metrics", str(tmp_path / "train-metrics.json"),
        "--split-json", str(tmp_path / "split.json"),
        "--accept-partial-d1", "--device", "cpu", "--seed", "3", "--split-seed", "0",
        "--tau", "2.0", "--train-seconds", "1.5", "--max-total-seconds", "60",
        "--max-steps", str(max_steps), "--batch-size", "2", "--eval-batch", "4",
        "--eval-every", "1", "--val-eval-samples", "8",
        "--stage-channels", "4,4,4", "--residual-blocks", "1", "--trunk", "8",
        "--readout-channels", "2", "--hud-width", "2", "--hud-hidden", "8",
        "--hist-hidden", "4", "--aux-channels", "4,4,4",
    ]


@pytest.fixture(scope="module")
def d1_plain(tmp_path_factory) -> Path:
    return build_d1(tmp_path_factory.mktemp("d1_plain"), with_aux=False)


@pytest.fixture(scope="module")
def d1_aux(tmp_path_factory) -> Path:
    return build_d1(tmp_path_factory.mktemp("d1_aux"), with_aux=True)


def test_d1_fixture_is_a_gate_passing_v3_corpus(d1_plain):
    rows = train_v3.complete_shard_rows(d1_plain)
    assert len(rows) == 4
    shard = obs_v3_dataset.load_teacher_shard_v3(rows[0]["dir"])
    assert len(shard) == DECISIONS
    assert shard.meta["obs_spec_digest"] == FROZEN_V3_DIGEST
    assert shard.play.shape[1:] == (4, 112, 96)
    samples = obs_v3_dataset.load_teacher_samples_v3(
        [Path(row["dir"]) / "labels.jsonl" for row in rows],
        min_samples=1, obs_root=d1_plain, require_observations=True)
    assert samples["count"] == 4 * DECISIONS
    assert samples["obs_unresolved"] == []


def test_train_v3_runs_end_to_end_without_aux_labels(d1_plain, tmp_path):
    torch = pytest.importorskip("torch")
    code = train_v3.main(_train_argv(d1_plain, tmp_path))
    assert code == 0
    metrics = json.loads((tmp_path / "train-metrics.json").read_text(encoding="utf-8"))
    assert metrics["parameters"] > 0
    assert metrics["canonical_architecture"] is False  # the tiny variant
    assert metrics["obs_spec_digest"] == FROZEN_V3_DIGEST
    assert metrics["d1"]["gate_waived"] is True
    assert metrics["d1"]["decisions_in_complete_shards"] == 4 * DECISIONS
    assert metrics["d1"]["obs_spec_digest"] == FROZEN_V3_DIGEST
    assert {name: metrics["splits"][name]["shard_count"]
            for name in ("train", "val", "test")} == {"train": 2, "val": 1, "test": 1}
    assert metrics["loss_weights_effective"] == {"kl": 1.0, "bomb": 1.0, "aux": 0.0,
                                                 "value": 0.25}
    assert metrics["aux_labels"]["components_present"] == []
    assert "skipped" in metrics["loss_last_breakdown"]
    assert metrics["value_target"]["proxy"] is True
    loaded, payload = model_v3.load_checkpoint_v3(tmp_path / "checkpoint-v3.pt")
    assert payload["arch"] == "v3"
    assert payload["obs_spec_digest"] == FROZEN_V3_DIGEST
    assert metrics["checkpoint_sha256"] == model_v3.checkpoint_digest(
        tmp_path / "checkpoint-v3.pt")
    assert payload["train"]["tau"] == 2.0
    split = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
    assert split["signature"] == metrics["splits"]["signature"]
    assert len(split["assignments"]) == 4
    # The three splits are episode-disjoint and every seed appears once.
    seeds = [metrics["splits"][name]["seeds"] for name in ("train", "val", "test")]
    flat = [seed for group in seeds for seed in group]
    assert len(flat) == len(set(flat)) == 4


def test_train_v3_uses_aux_labels_when_a_shard_carries_them(d1_aux, tmp_path):
    code = train_v3.main(_train_argv(d1_aux, tmp_path, max_steps=2))
    assert code == 0
    metrics = json.loads((tmp_path / "train-metrics.json").read_text(encoding="utf-8"))
    assert metrics["aux_labels"]["components_present"] == ["danger", "occupancy",
                                                           "self_heatmap"]
    assert metrics["loss_weights_effective"]["aux"] == 0.5
    assert metrics["loss_last_breakdown"]["aux"]["active"] == ["occupancy",
                                                               "self_heatmap", "danger"]
    assert metrics["loss_last_breakdown"]["total"] > 0.0


def test_train_v3_refuses_a_mis_shaped_aux_label(d1_plain):
    rows = train_v3.complete_shard_rows(d1_plain)
    aux_dir = Path(rows[0]["dir"]) / "aux"
    aux_dir.mkdir(exist_ok=True)
    np.save(aux_dir / "occupancy.npy", np.zeros((DECISIONS, 56, 48), np.uint8))
    with pytest.raises(train_v3.TrainV3Error, match="aux label grid"):
        train_v3.load_shard_block(rows[0])
    (aux_dir / "occupancy.npy").unlink()


def test_train_v3_refuses_a_corpus_below_the_contract(d1_plain, tmp_path):
    argv = _train_argv(d1_plain, tmp_path)
    argv = [token for token in argv if token != "--accept-partial-d1"]
    with pytest.raises(SystemExit, match="accept-partial-d1"):
        train_v3.main(argv)


# ---------------------------------------------------------------------------
# 7. the two CLI tools route --obs v3 / --arch v3
# ---------------------------------------------------------------------------
def test_train_tool_routes_obs_v3_without_changing_the_old_routes():
    import v400_student_train_teacher as tool

    parser = tool.build_parser()
    base = ["--checkpoint", "c.pt", "--metrics", "m.json"]
    assert tool.route_obs(parser.parse_args(base)) == "v1"
    assert tool.route_obs(parser.parse_args(base + ["--obs", "v3"])) == "v3"
    assert tool.route_obs(parser.parse_args(base + ["--arch", "v3"])) == "v3"
    assert tool.route_obs(parser.parse_args(base + ["--obs-v2"])) == "v2"
    assert tool.route_obs(parser.parse_args(base + ["--arch", "gru"])) == "v2"
    assert tool.route_obs(parser.parse_args(base + ["--obs", "v2"])) == "v2"
    v3_args = parser.parse_args(base + ["--obs", "v3", "--taus", "2.0",
                                        "--d1-root", "/d1"])
    argv = tool._train_v3_argv(v3_args)
    assert argv[0] == "--d1-root" and argv[1] == "/d1"
    assert "--obsv2-root" not in argv and "--width" not in argv
    assert "--trunk" in argv and "--aux-weight" in argv
    # A sweep is refused by the v3 route exactly as it is by the obs-v2 route.
    with pytest.raises(SystemExit, match="tau"):
        tool._train_v3_argv(parser.parse_args(base + ["--obs", "v3"]))


def test_eval_tool_routes_obs_v3_and_stamps_the_record():
    import v400_student_eval as tool

    parser = tool.build_parser()
    base = ["--engine", "e", "--data-dir", "d", "--checkpoint", "c.pt"]
    assert tool.route_obs(parser.parse_args(base)) == "v1"
    assert tool.route_obs(parser.parse_args(base + ["--obs", "v3"])) == "v3"
    assert tool.route_obs(parser.parse_args(base + ["--arch", "v3"])) == "v3"
    assert tool.route_obs(parser.parse_args(base + ["--obs-v2"])) == "v2"
    disclosure, note = tool._policy_identity({
        "arch": "v3", "obs_spec_version": "obs-v3",
        "obs_spec_digest": FROZEN_V3_DIGEST, "parameters": 2_602_490,
        "model": {"arch": "v3", "trunk": 512}, "train": {"tau": 2.0}})
    assert "obs-v3" in disclosure and "arch=v3" in note
    record = {
        "schema": tool.EPISODE_SCHEMA, "checkpoint": "c.pt",
        "checkpoint_sha256": "a" * 64,
        "obs_spec_version": obs_spec_v3.OBS_SPEC_VERSION,
        "obs_spec_digest": FROZEN_V3_DIGEST,
        "seed": 1000, "suite": "S1N", "stage": 0, "difficulty": "normal",
        "max_frames": 19887, "frames": 321, "deaths": 1.0, "score": 1234,
        "end": "death_stop", "privilege": "deploy_pixel", "wall_s": 1.0,
        "engine_sha256": "b" * 64,
        "policy_input": "play96x112 4-frame stack + hud strip + act_hist",
        "not_a_gate": True, "stage_start": 0, "stage_end": 0, "cleared": False,
        "exit": "controller", "stop_on_death": True,
    }
    tool.validate_episode_record(
        record, expected_spec=(obs_spec_v3.OBS_SPEC_VERSION, FROZEN_V3_DIGEST))
    with pytest.raises(tool.EvalError, match="observed obs spec"):
        tool.validate_episode_record(
            dict(record, obs_spec_version="obs-v1"),
            expected_spec=(obs_spec_v3.OBS_SPEC_VERSION, FROZEN_V3_DIGEST))
    with pytest.raises(tool.EvalError, match="obs spec"):
        tool.validate_episode_record(record)  # obs-v1 is the default expectation
