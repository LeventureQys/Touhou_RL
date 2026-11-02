"""v3.7.2 pixel-only macro residual policy and positive-set supervision.

The backbone remains frozen/eval. Cached features and baseline logits must come
from that same backbone, not privileged game state. Unknown/beam-failed actions
are *not* negative labels: only verified positive sets supervise the policy.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

N_ACTIONS = 18
SCHEMA = "v3.7.2-macro-policy/1"


@dataclass
class MacroConfig:
    feature_dim: int = 256
    hidden_dim: int = 128
    route_dim: int = 2

    def __post_init__(self):
        for name in ("feature_dim", "hidden_dim", "route_dim"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.route_dim != 2:
            raise ValueError("route_dim must be 2 (image-derived target direction)")


def _finite(value, name, shape=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _mask(value, name, shape, device):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must be bool {tuple(shape)}")
    if value.device != device:
        raise ValueError(f"{name} device mismatch")
    return value


class MacroPolicy(nn.Module):
    """Continuous target-direction head conditions a trainable action residual.

    ``config`` intentionally remains the *backbone* config for evaluate's region
    and history contract; new-head architecture lives in ``macro_config``.
    """
    def __init__(self, vision_model: nn.Module, config: MacroConfig | None = None):
        super().__init__()
        self.vision_model = vision_model
        self.config = vision_model.config
        self.macro_config = config or MacroConfig(feature_dim=self.config.trunk_hidden)
        if self.macro_config.feature_dim != self.config.trunk_hidden:
            raise ValueError("macro feature_dim must match backbone trunk_hidden")
        c = self.macro_config
        self.heads = nn.ModuleDict({
            "route": nn.Sequential(nn.Linear(c.feature_dim, c.hidden_dim), nn.ReLU(),
                                   nn.Linear(c.hidden_dim, c.route_dim), nn.Tanh()),
            "residual": nn.Sequential(nn.Linear(c.feature_dim + c.route_dim, c.hidden_dim),
                                      nn.ReLU(), nn.Linear(c.hidden_dim, N_ACTIONS)),
        })
        nn.init.zeros_(self.heads["residual"][-1].weight)
        nn.init.zeros_(self.heads["residual"][-1].bias)
        for parameter in vision_model.parameters():
            parameter.requires_grad_(False)
        vision_model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.vision_model.eval()
        return self

    def forward_features(self, features, base_logits):
        """Train new heads from detached [B,F] features and [B,18] logits."""
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("features must be [B,feature_dim]")
        batch = features.shape[0]
        _finite(features, "features", (batch, self.macro_config.feature_dim))
        _finite(base_logits, "base_logits", (batch, N_ACTIONS))
        if features.device != base_logits.device or features.dtype != base_logits.dtype:
            raise ValueError("features/base_logits dtype or device mismatch")
        features, base_logits = features.detach(), base_logits.detach()
        route = self.heads["route"](features)
        # No detach here: policy gradients genuinely train the target direction.
        residual = self.heads["residual"](torch.cat((features, route), dim=-1))
        logits = base_logits + residual
        _finite(logits, "policy_logits")
        return {"features": features, "base_logits": base_logits,
                "residual_logits": residual, "policy_logits": logits,
                "route_target": route}

    def forward_core(self, regions):
        self.vision_model.eval()
        with torch.no_grad():
            core = self.vision_model.forward_core(regions)
        return {**core, **self.forward_features(core["features"], core["policy_logits"])}

    def forward(self, regions):
        return self.forward_core(regions)

    def features(self, regions):
        return self.forward_core(regions)["features"]

    def score(self, regions, features=None):
        if features is None:
            return self.forward_core(regions)["policy_logits"]
        with torch.no_grad():
            base = self.vision_model.score(regions, features=features)
        return self.forward_features(features, base)["policy_logits"]

    def decode_state(self, regions, features=None):
        with torch.no_grad():
            return self.vision_model.decode_state(regions, features=features)

    @torch.no_grad()
    def act(self, regions, deterministic=True):
        # v3's legacy signature is greedy even when deterministic=False.
        logits = self.score(regions)
        best = logits.argmax(-1)
        actions = torch.stack((best % 9, torch.ones_like(best), best // 9,
                               torch.zeros_like(best)), -1)
        zero = logits.new_zeros(logits.shape[0])
        return actions, zero, logits.max(-1).values, zero

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def macro_policy_loss(output, positive_mask, valid_mask=None, *, route_target=None,
                      route_valid=None, route_weight=1.0, local_valid=None,
                      kl_weight=0.0, base_logits=None):
    """Return loss components; all-empty masks yield differentiable exact zeros.

    Policy NLL is -log probability mass of the *verified positive set*, not
    per-action BCE nor mean positive NLL. No-positive rows are unknown by default;
    explicitly declaring one valid is a data error. Route MSE requires its own
    bool route_valid; local KL(base || policy) requires explicit local_valid.
    Those independent masks may supervise rows without a verified action set.
    """
    logits = output["policy_logits"]
    if logits.ndim != 2 or logits.shape[-1] != N_ACTIONS:
        raise ValueError("policy_logits must be [B,18]")
    _finite(logits, "policy_logits")
    batch = logits.shape[0]
    positive = _mask(positive_mask, "positive_mask", logits.shape, logits.device)
    has_positive = positive.any(-1)
    valid = has_positive if valid_mask is None else _mask(valid_mask, "valid_mask", (batch,), logits.device)
    if bool((valid & ~has_positive).any()):
        raise ValueError("valid policy sample has no positive actions")
    for name, weight in (("route_weight", route_weight), ("kl_weight", kl_weight)):
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    zero = logits.sum() * 0
    policy = zero
    if bool(valid.any()):
        logp = logits[valid].log_softmax(-1)
        policy = -logp.masked_fill(~positive[valid], -torch.inf).logsumexp(-1).mean()
    route = zero
    prediction = output.get("route_target")
    if prediction is not None:
        _finite(prediction, "predicted route_target", (batch, 2))
        route = route + prediction.sum() * 0
    if (route_target is None) != (route_valid is None):
        raise ValueError("route_target and route_valid must be supplied together")
    if route_target is not None:
        _finite(route_target, "route_target", (batch, 2))
        rv = _mask(route_valid, "route_valid", (batch,), logits.device)
        if prediction is None or route_target.device != logits.device:
            raise ValueError("missing prediction or route device mismatch")
        if bool(rv.any()):
            route = F.mse_loss(prediction[rv], route_target.detach()[rv])
    kl = zero
    if local_valid is not None:
        lv = _mask(local_valid, "local_valid", (batch,), logits.device)
        base = output["base_logits"] if base_logits is None else base_logits
        _finite(base, "base_logits", logits.shape)
        if base.device != logits.device:
            raise ValueError("base_logits device mismatch")
        if bool(lv.any()):
            kl = F.kl_div(logits[lv].log_softmax(-1),
                          base.detach()[lv].log_softmax(-1), log_target=True,
                          reduction="batchmean")
    elif kl_weight != 0:
        raise ValueError("nonzero kl_weight requires explicit local_valid")
    total = policy + route_weight * route + kl_weight * kl
    _finite(total, "loss")
    return {"loss": total, "policy_loss": policy, "route_loss": route, "kl_loss": kl,
            "valid_count": valid.sum()}


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


macro_loss = macro_policy_loss


def save_macro_checkpoint(path, policy, baseline_checkpoint, metadata=None):
    """Create a head-only sidecar; never overwrite an existing artifact."""
    from .vmodels import load_vision_checkpoint

    source = Path(baseline_checkpoint).resolve(strict=True)
    original, _ = load_vision_checkpoint(source, map_location="cpu")
    expected = original.state_dict()
    actual = policy.vision_model.state_dict()
    if set(expected) != set(actual) or any(
        expected[key].dtype != actual[key].dtype
        or expected[key].shape != actual[key].shape
        or not torch.equal(expected[key].detach().cpu(), actual[key].detach().cpu())
        for key in expected
    ):
        raise ValueError("policy backbone does not match baseline checkpoint state")
    state = {k: v.detach().cpu() for k, v in policy.heads.state_dict().items()}
    for name, value in state.items():
        _finite(value, name)
    payload = {"schema": SCHEMA, "config": asdict(policy.macro_config),
               "source": {"checkpoint": str(source), "sha256": checkpoint_sha256(source),
                          "metadata": dict(metadata or {})},
               "state_dict": state}
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)


def load_macro_checkpoint(path, baseline_checkpoint=None, map_location="cpu"):
    """Verify source SHA before loading it via the standard vision loader.

    Returns (MacroPolicy, extra), matching load_vision_checkpoint. The returned
    extra retains the baseline's evaluation contract and adds macro provenance.
    """
    from .vmodels import load_vision_checkpoint

    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"schema", "config", "source", "state_dict"} or payload["schema"] != SCHEMA:
        raise ValueError("invalid macro sidecar schema")
    source = payload["source"]
    if not isinstance(source, dict) or set(source) != {"checkpoint", "sha256", "metadata"} or not isinstance(source["metadata"], dict):
        raise ValueError("invalid macro sidecar source")
    baseline = Path(baseline_checkpoint or source["checkpoint"])
    if not baseline.is_file() or checkpoint_sha256(baseline) != source["sha256"]:
        raise ValueError("baseline checkpoint SHA256 mismatch")
    if not isinstance(payload["state_dict"], dict):
        raise ValueError("invalid macro head state")
    for name, value in payload["state_dict"].items():
        _finite(value, name)
    backbone, extra = load_vision_checkpoint(baseline, map_location=map_location)
    from .vmodels import vision_arch
    if vision_arch(backbone) != "v3-st":
        raise ValueError("macro sidecar requires a v3-st baseline")
    policy = MacroPolicy(backbone, MacroConfig(**payload["config"]))
    policy.heads.load_state_dict(payload["state_dict"], strict=True)
    # Backbone loader constructs CPU modules; preserve that established convention.
    policy.eval()
    return policy, {**extra, "macro_source": dict(source), "macro_schema": SCHEMA}
