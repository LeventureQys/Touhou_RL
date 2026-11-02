"""CPU stub tests: no game, memory reader, or real checkpoint is required."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from touhou_rl.macro_policy import (
    MacroConfig, MacroPolicy, load_macro_checkpoint, macro_loss,
    save_macro_checkpoint,
)


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(trunk_hidden=4, depth=3)
        self.trunk = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.8))
        self.policy_head = nn.Linear(4, 18)

    def forward_core(self, regions):
        features = self.trunk(regions["pixels"])
        return {"features": features, "policy_logits": self.policy_head(features),
                "position": features[:, :2]}

    def score(self, regions, features=None):
        return self.policy_head(features) if features is not None else self.forward_core(regions)["policy_logits"]

    def decode_state(self, regions, features=None):
        return {"self_feat": self.forward_core(regions)["features"]}


@pytest.fixture
def policy():
    torch.manual_seed(37)
    return MacroPolicy(StubBackbone(), MacroConfig(feature_dim=4, hidden_dim=8))


def inputs(batch=3):
    return torch.randn(batch, 4), torch.randn(batch, 18)


def test_initialization_exact_equivalence_and_legacy_surface(policy):
    policy.train()
    assert policy.config is policy.vision_model.config
    assert not policy.vision_model.training
    regions = {"pixels": torch.randn(3, 4)}
    baseline = policy.vision_model.forward_core(regions)
    actual = policy.forward_core(regions)
    assert torch.equal(actual["policy_logits"], baseline["policy_logits"])
    assert torch.equal(policy.score(regions, features=baseline["features"]), baseline["policy_logits"])
    assert torch.equal(policy.decode_state(regions)["self_feat"], baseline["features"])
    actions, logp, preference, extra = policy.act(regions)
    best = baseline["policy_logits"].argmax(-1)
    assert torch.equal(actions[:, 2] * 9 + actions[:, 0], best)
    assert (actions[:, 1] == 1).all() and (actions[:, 3] == 0).all()
    assert (logp == 0).all() and (extra == 0).all()
    assert torch.equal(preference, baseline["policy_logits"].max(-1).values)


def test_only_new_heads_receive_gradients_and_cached_inputs_detached(policy):
    features, base = inputs()
    features.requires_grad_()
    base.requires_grad_()
    positives = torch.zeros(3, 18, dtype=torch.bool)
    positives[:, 4] = True
    output = policy.forward_features(features, base)
    loss = macro_loss(output, positives, route_target=torch.ones(3, 2),
                      route_valid=torch.ones(3, dtype=torch.bool))["loss"]
    loss.backward()
    assert all(not p.requires_grad and p.grad is None for p in policy.vision_model.parameters())
    assert features.grad is None and base.grad is None
    assert all(p.grad is not None for p in policy.heads.parameters())
    assert policy.heads["residual"][-1].weight.grad.abs().sum() > 0
    assert policy.heads["route"][-2].weight.grad.abs().sum() > 0


def test_predicted_route_changes_final_action_and_gets_policy_gradient(policy):
    # Construct a transparent learned dependency: action 1 follows route x.
    with torch.no_grad():
        for p in policy.heads.parameters():
            p.zero_()
        policy.heads["residual"][0].weight[0, 4] = 1
        policy.heads["residual"][0].bias[0] = 1
        policy.heads["residual"][-1].weight[1, 0] = 1
        policy.heads["residual"][-1].bias[0] = 1
        policy.heads["route"][-2].bias[0] = -1
    features, _ = inputs(1)
    base = torch.zeros(1, 18)
    before = policy.forward_features(features, base)["policy_logits"].argmax(-1)
    with torch.no_grad():
        policy.heads["route"][-2].bias[0] = 1
    output = policy.forward_features(features, base)
    after = output["policy_logits"].argmax(-1)
    assert before.item() == 0 and after.item() == 1
    positives = torch.zeros(1, 18, dtype=torch.bool)
    positives[0, 0] = True
    macro_loss(output, positives)["loss"].backward()
    assert policy.heads["route"][-2].bias.grad[0].abs() > 0


def test_unknown_rows_skip_and_empty_valid_zero_gradients(policy):
    features, base = inputs()
    positives = torch.zeros(3, 18, dtype=torch.bool)
    positives[0, :2] = True
    output = policy.forward_features(features, base)
    output["policy_logits"].retain_grad()
    macro_loss(output, positives)["loss"].backward()
    assert torch.equal(output["policy_logits"].grad[1:], torch.zeros(2, 18))
    policy.zero_grad(set_to_none=True)
    output = policy.forward_features(features, base)
    loss = macro_loss(output, torch.zeros_like(positives))["loss"]
    assert loss.item() == 0 and loss.requires_grad
    loss.backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in policy.heads.parameters())
    with pytest.raises(ValueError, match="no positive"):
        macro_loss(output, torch.zeros_like(positives), torch.ones(3, dtype=torch.bool))


def test_positive_set_mass_not_bce_or_average_nll():
    logits = torch.zeros(1, 18, requires_grad=True)
    positives = torch.zeros(1, 18, dtype=torch.bool)
    positives[:, :3] = True
    output = {"policy_logits": logits}
    result = macro_loss(output, positives)["loss"]
    assert result.item() == pytest.approx(torch.log(torch.tensor(6.)).item())
    result.backward()
    assert torch.allclose(logits.grad[0, :3], torch.full((3,), -5 / 18))
    all_positive = macro_loss(output, torch.ones_like(positives))["loss"]
    assert all_positive.item() == pytest.approx(0, abs=1e-6)


def test_route_mask_and_local_kl_are_explicit_and_independent(policy):
    features, base = inputs()
    output = policy.forward_features(features, base)
    positives = torch.zeros(3, 18, dtype=torch.bool)
    target = output["route_target"].detach().clone()
    target[1:] += 100
    result = macro_loss(output, positives, route_target=target,
                        route_valid=torch.tensor([True, False, False]),
                        local_valid=torch.ones(3, dtype=torch.bool), kl_weight=1)
    assert result["route_loss"].item() == 0
    assert result["kl_loss"].item() == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError, match="explicit local_valid"):
        macro_loss(output, positives, kl_weight=1)
    with pytest.raises(ValueError, match="together"):
        macro_loss(output, positives, route_target=target)
    output["policy_logits"] = base + torch.arange(18).float()
    result = macro_loss(output, positives, local_valid=torch.ones(3, dtype=torch.bool), kl_weight=1)
    expected = torch.nn.functional.kl_div(output["policy_logits"].log_softmax(-1),
                                         base.softmax(-1), reduction="batchmean")
    assert torch.allclose(result["kl_loss"], expected)


@pytest.mark.parametrize("bad", ["features", "base", "mask", "route", "weight"])
def test_validation(policy, bad):
    features, base = inputs()
    if bad == "features":
        features[0, 0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            policy.forward_features(features, base)
        return
    if bad == "base":
        with pytest.raises(ValueError, match="shape"):
            policy.forward_features(features, base[:, :-1])
        return
    output = policy.forward_features(features, base)
    positive = torch.zeros(3, 18, dtype=torch.bool)
    kwargs = {}
    if bad == "mask":
        positive = positive.float()
    elif bad == "route":
        kwargs = {"route_target": torch.full((3, 2), float("inf")),
                  "route_valid": torch.zeros(3, dtype=torch.bool)}
    else:
        kwargs = {"kl_weight": float("nan")}
    with pytest.raises(ValueError):
        macro_loss(output, positive, **kwargs)


def test_zero_batch(policy):
    features, base = inputs(0)
    result = macro_loss(policy.forward_features(features, base), torch.zeros(0, 18, dtype=torch.bool))
    assert result["loss"].item() == 0
    result["loss"].backward()
    assert all(p.grad is not None and p.grad.count_nonzero() == 0 for p in policy.heads.parameters())


def test_sidecar_only_new_heads_roundtrip_and_sha_validation(policy, tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"frozen original baseline")
    sidecar = tmp_path / "macro.pt"
    original_state = {k: v.clone() for k, v in policy.vision_model.state_dict().items()}
    calls = []

    def loader(path, map_location):
        calls.append(path)
        model = StubBackbone()
        model.load_state_dict(original_state)
        return model, {"contract": "preserved"}

    monkeypatch.setattr("touhou_rl.vmodels.load_vision_checkpoint", loader)
    monkeypatch.setattr("touhou_rl.vmodels.vision_arch", lambda model: "v3-st")
    with torch.no_grad():
        policy.heads["residual"][-1].bias[7] = 2
    save_macro_checkpoint(sidecar, policy, baseline, metadata={"split": "train"})
    payload = torch.load(sidecar, weights_only=True)
    assert set(payload) == {"schema", "source", "config", "state_dict"}
    assert set(payload["state_dict"]) == set(policy.heads.state_dict())
    assert not any("vision_model" in key for key in payload["state_dict"])
    restored, extra = load_macro_checkpoint(sidecar)
    assert calls == [baseline, baseline]
    assert extra["contract"] == "preserved"
    assert extra["macro_source"]["metadata"] == {"split": "train"}
    features, base = inputs()
    assert torch.equal(policy.forward_features(features, base)["policy_logits"],
                       restored.forward_features(features, base)["policy_logits"])
    with pytest.raises(FileExistsError):
        save_macro_checkpoint(sidecar, policy, baseline)
    baseline.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        load_macro_checkpoint(sidecar)
    assert len(calls) == 3


def test_sidecar_save_rejects_wrong_baseline_and_load_rejects_non_v3(policy, tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    sidecar = tmp_path / "macro.pt"
    wrong = StubBackbone()
    with torch.no_grad():
        wrong.policy_head.bias.copy_(policy.vision_model.policy_head.bias + 1)
    monkeypatch.setattr("touhou_rl.vmodels.load_vision_checkpoint",
                        lambda *a, **k: (wrong, {}))
    with pytest.raises(ValueError, match="does not match baseline"):
        save_macro_checkpoint(sidecar, policy, baseline)
    assert not sidecar.exists()
    monkeypatch.setattr("touhou_rl.vmodels.load_vision_checkpoint",
                        lambda *a, **k: (policy.vision_model, {}))
    save_macro_checkpoint(sidecar, policy, baseline)
    monkeypatch.setattr("touhou_rl.vmodels.vision_arch", lambda model: "v2")
    with pytest.raises(ValueError, match="requires a v3-st"):
        load_macro_checkpoint(sidecar)


def test_sidecar_rejects_nonfinite_and_extra_backbone_state(policy, tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    sidecar = tmp_path / "macro.pt"
    monkeypatch.setattr("touhou_rl.vmodels.load_vision_checkpoint",
                        lambda *a, **k: (policy.vision_model, {}))
    monkeypatch.setattr("touhou_rl.vmodels.vision_arch", lambda model: "v3-st")
    save_macro_checkpoint(sidecar, policy, baseline)
    payload = torch.load(sidecar, weights_only=True)
    key = next(iter(payload["state_dict"]))
    payload["state_dict"][key].flatten()[0] = float("nan")
    torch.save(payload, sidecar)
    with pytest.raises(ValueError, match="finite"):
        load_macro_checkpoint(sidecar)
    payload["state_dict"][key].flatten()[0] = 0
    payload["state_dict"]["vision_model.forbidden"] = torch.zeros(1)
    torch.save(payload, sidecar)
    monkeypatch.setattr("touhou_rl.vmodels.load_vision_checkpoint",
                        lambda *a, **k: (StubBackbone(), {}))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_macro_checkpoint(sidecar)
    monkeypatch.setattr("touhou_rl.vmodels.load_vision_checkpoint",
                        lambda *a, **k: (policy.vision_model, {}))
    with torch.no_grad():
        policy.heads["residual"][-1].bias[0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        save_macro_checkpoint(tmp_path / "bad.pt", policy, baseline)
