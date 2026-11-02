"""P4 engineering interface only; no teacher, dataset, training or safety claim.

Cached visual inputs must originate from the SHA-bound frozen backbone. History
is causal executed history, not labels. N1 uses differentiable predicted option
probabilities and target in both train/eval; N0 omits those conditions; B1 is a
memoryless diagnostic with the same public input contract (history is ignored).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from .macro_policy import _finite, _mask, checkpoint_sha256

N_ACTIONS = 18
SCHEMA = "v3.7.3-tactical-policy/1"


@dataclass
class TacticalConfig:
    arm: str = "N1"
    feature_dim: int = 256
    hidden_size: int = 256
    target_dim: int = 2
    option_embed_dim: int = 16

    def __post_init__(self):
        if self.arm not in {"N0", "N1", "B1"}:
            raise ValueError("arm must be N0, N1 or B1")
        for name in ("feature_dim", "hidden_size", "target_dim", "option_embed_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.target_dim != 2:
            raise ValueError("target_dim=2 is normalized image-derived direction")


def bos_history(features, target_dim=2):
    shape = features.shape[:2]
    return {"previous_candidate": torch.full(shape, 18, device=features.device, dtype=torch.long),
            "previous_option": torch.full(shape, 3, device=features.device, dtype=torch.long),
            "previous_hold": features.new_zeros(shape), "frame_delta": features.new_zeros(shape),
            "previous_target": features.new_zeros((*shape, target_dim)),
            "target_valid": torch.zeros(shape, device=features.device, dtype=torch.bool),
            "history_valid": torch.zeros(shape, device=features.device, dtype=torch.bool)}


class TacticalPolicy(nn.Module):
    def __init__(self, vision_model, config=None):
        super().__init__()
        self.vision_model = vision_model
        self.config = vision_model.config
        self.tactical_config = config or TacticalConfig(feature_dim=self.config.trunk_hidden)
        c = self.tactical_config
        if c.feature_dim != self.config.trunk_hidden:
            raise ValueError("feature_dim must match backbone")
        # candidate19 + option4 + hold/delta2 + target2 + validity2
        self.heads = nn.ModuleDict({
            "gru": nn.GRU(c.feature_dim + 29, c.hidden_size, batch_first=True),
            "memoryless": nn.Linear(c.feature_dim, c.hidden_size),
            "option": nn.Linear(c.hidden_size, 3),
            "target": nn.Sequential(nn.Linear(c.hidden_size, c.target_dim), nn.Tanh()),
            "urgency": nn.Linear(c.hidden_size, 1),
            "termination": nn.Linear(c.hidden_size, 3),
            "slack": nn.Sequential(nn.Linear(c.hidden_size, 1), nn.Tanh()),
            "option_embedding": nn.Linear(3, c.option_embed_dim, bias=False),
            "residual": nn.Linear(c.feature_dim + c.hidden_size +
                                  (0 if c.arm == "N0" else c.option_embed_dim + c.target_dim), 18),
        })
        # Inactive diagnostic branches remain checkpoint-compatible but are not
        # trainable and must not inflate optimizer/parameter-budget reporting.
        inactive = ["gru"] if c.arm == "B1" else ["memoryless"]
        if c.arm == "N0":
            inactive += ["option_embedding"]
        for name in inactive:
            for p in self.heads[name].parameters():
                p.requires_grad_(False)
        nn.init.zeros_(self.heads["residual"].weight)
        nn.init.zeros_(self.heads["residual"].bias)
        for p in vision_model.parameters():
            p.requires_grad_(False)
        vision_model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.vision_model.eval()
        return self

    def forward_features(self, features, base_logits, history=None, hidden=None,
                         reset_mask=None, valid_mask=None):
        if features.ndim != 3 or min(features.shape[:2]) < 1:
            raise ValueError("features must be nonempty [B,T,F]")
        b, t, _ = features.shape
        c = self.tactical_config
        _finite(features, "features", (b, t, c.feature_dim))
        _finite(base_logits, "base_logits", (b, t, 18))
        if features.device != base_logits.device or features.dtype != base_logits.dtype:
            raise ValueError("features/base dtype or device mismatch")
        features, base_logits = features.detach(), base_logits.detach()
        hist = bos_history(features) if history is None else history
        if set(hist) != set(bos_history(features)):
            raise ValueError("history fields mismatch")
        for name, maximum in (("previous_candidate", 18), ("previous_option", 3)):
            x = hist[name]
            if x.shape != (b, t) or x.dtype != torch.long or x.device != features.device or bool(((x < 0) | (x > maximum)).any()):
                raise ValueError(f"invalid {name}")
        for name in ("target_valid", "history_valid"):
            _mask(hist[name], name, (b, t), features.device)
        for name in ("previous_hold", "frame_delta", "previous_target"):
            _finite(hist[name], name, (b, t, 2) if name == "previous_target" else (b, t))
            if hist[name].device != features.device or hist[name].dtype != features.dtype:
                raise ValueError("history dtype or device mismatch")
        if bool(((hist["previous_hold"] < 0) | (hist["frame_delta"] < 0)).any()):
            raise ValueError("hold and frame delta must be nonnegative")
        valid = torch.ones((b,t), dtype=torch.bool, device=features.device) if valid_mask is None else _mask(valid_mask, "valid_mask", (b,t), features.device)
        reset = torch.zeros_like(valid) if reset_mask is None else _mask(reset_mask, "reset_mask", (b,t), features.device)
        h = features.new_zeros((1,b,c.hidden_size)) if hidden is None else hidden
        _finite(h, "hidden", (1,b,c.hidden_size))
        if h.device != features.device or h.dtype != features.dtype:
            raise ValueError("hidden dtype or device mismatch")
        hv = hist["history_valid"]
        pc = torch.where(hv, hist["previous_candidate"], 18)
        po = torch.where(hv, hist["previous_option"], 3)
        tv = hv & hist["target_valid"]
        inp = torch.cat((features, F.one_hot(pc,19).to(features.dtype),
                         F.one_hot(po,4).to(features.dtype),
                         torch.log1p(hist["previous_hold"] * hv)[...,None],
                         torch.log1p(hist["frame_delta"] * hv)[...,None],
                         torch.where(tv[...,None], hist["previous_target"], 0).detach(),
                         tv[...,None].to(features.dtype), hv[...,None].to(features.dtype)), -1)
        states = []
        for i in range(t):
            # A padding position is inert, including reset flags on padding.
            start = torch.where((reset[:,i] & valid[:,i])[None,:,None], 0, h)
            if c.arm == "B1":
                candidate = torch.tanh(self.heads["memoryless"](features[:,i]))[None]
            else:
                _, candidate = self.heads["gru"](inp[:,i:i+1], start)
            h = torch.where(valid[:,i][None,:,None], candidate, h)
            states.append(h[0])
        state = torch.stack(states, 1)
        option = self.heads["option"](state)
        target = self.heads["target"](state)
        conditions = [features, state]
        if c.arm != "N0":
            conditions += [self.heads["option_embedding"](option.softmax(-1)), target]
        residual = self.heads["residual"](torch.cat(conditions, -1))
        return {"features": features, "base_logits": base_logits,
                "policy_logits": base_logits + residual, "candidate_logits": base_logits + residual,
                "residual_logits": residual, "option_logits": option, "target": target,
                "route_target": target, "urgency": self.heads["urgency"](state).sigmoid().squeeze(-1),
                "termination_logits": self.heads["termination"](state),
                "slack": self.heads["slack"](state).squeeze(-1), "hidden": h,
                "valid_mask": valid}

    def forward_regions(self, regions, history=None, hidden=None, reset_mask=None, valid_mask=None):
        """Pixel regions [B,T,depth,H,W,C], never privileged label fields."""
        if not isinstance(regions, dict) or not regions:
            raise ValueError("nonempty pixel-region mapping required")
        if any(x.ndim != 6 for x in regions.values()):
            raise ValueError("temporal regions must be 6D")
        features, base = [], []
        self.vision_model.eval()
        with torch.no_grad():
            for i in range(next(iter(regions.values())).shape[1]):
                core = self.vision_model.forward_core({k:v[:,i] for k,v in regions.items()})
                features.append(core["features"])
                base.append(core["policy_logits"])
        return self.forward_features(torch.stack(features,1), torch.stack(base,1), history,
                                     hidden, reset_mask, valid_mask)

    forward = forward_regions


def tactical_policy_loss(output, labels=None, *, weights=None):
    """Independent masks, each normalized by its own valid row count.

    labels: action_positive[...,18] bool; option_positive[...,3] bool OR
    option_target[...,3] soft distribution; target[...,2]; termination long;
    slack scalar OR slack_interval[...,2]. Each requires its own *_valid mask;
    local_valid enables KL(base || policy). All are intersected with seq valid.
    Unknown rows can carry NaN/sentinel label payloads: they are never indexed.
    Caller MUST skip optimizer.step when has_supervision is False.
    """
    labels = {} if labels is None else labels
    defaults = dict(action=1., option=.2, target=.5, termination=.2, slack=.1, local=1.)
    if weights is not None:
        if set(weights) - set(defaults):
            raise ValueError("unknown loss weight")
        defaults.update(weights)
    if any(not isinstance(w,(int,float)) or not math.isfinite(w) or w < 0 for w in defaults.values()):
        raise ValueError("weights must be finite nonnegative")
    logits = output["policy_logits"]
    shape = logits.shape[:-1]
    _finite(logits, "policy_logits")
    valid = output.get("valid_mask", torch.ones(shape, dtype=torch.bool, device=logits.device))
    zero = sum(output[k].sum()*0 for k in ("policy_logits", "option_logits", "target", "termination_logits", "slack"))
    losses, counts = {}, {}
    for name in defaults:
        mask = labels.get(name+"_valid")
        mask = torch.zeros_like(valid) if mask is None else _mask(mask, name+"_valid", shape, logits.device) & valid
        counts[name] = int(mask.sum())
        loss = zero
        if counts[name]:
            if name in {"action", "option"}:
                pred = logits if name == "action" else output["option_logits"]
                if name == "option" and "option_target" in labels:
                    target = labels["option_target"][mask].detach()
                    _finite(target, "option_target")
                    if target.shape != pred[mask].shape or bool((target < 0).any()) or not torch.allclose(target.sum(-1), torch.ones_like(target.sum(-1))):
                        raise ValueError("option_target must be a probability distribution")
                    loss = -(target * pred[mask].log_softmax(-1)).sum(-1).mean()
                else:
                    positive = labels[name+"_positive"]
                    _mask(positive, name+"_positive", pred.shape, logits.device)
                    if not bool(positive[mask].any(-1).all()):
                        raise ValueError("supervised set must contain a positive")
                    loss = -pred[mask].log_softmax(-1).masked_fill(~positive[mask], -torch.inf).logsumexp(-1).mean()
            elif name == "target":
                target = labels["target"][mask].detach()
                _finite(target, "target", output["target"][mask].shape)
                loss = F.mse_loss(output["target"][mask], target)
            elif name == "termination":
                loss = F.cross_entropy(output["termination_logits"][mask], labels["termination"][mask].detach())
            elif name == "slack":
                pred = output["slack"][mask]
                if "slack_interval" in labels:
                    interval = labels["slack_interval"][mask].detach()
                    _finite(interval, "slack_interval", (*pred.shape,2))
                    if bool((interval[...,0] > interval[...,1]).any()):
                        raise ValueError("invalid slack interval")
                    loss = ((interval[...,0]-pred).clamp_min(0).square() + (pred-interval[...,1]).clamp_min(0).square()).mean()
                else:
                    target = labels["slack"][mask].detach()
                    _finite(target, "slack", pred.shape)
                    loss = F.mse_loss(pred, target)
            else:
                loss = F.kl_div(logits[mask].log_softmax(-1), output["base_logits"][mask].detach().log_softmax(-1), log_target=True, reduction="batchmean")
        losses[name+"_loss"] = loss
    total = sum(defaults[name]*losses[name+"_loss"] for name in defaults)
    _finite(total, "loss")
    supervised = sum(counts[n] for n in defaults if defaults[n] > 0)
    return {"loss": total, **losses, "counts": counts, "supervised_count": supervised,
            "has_supervision": supervised > 0}


def save_tactical_checkpoint(path, policy, baseline_checkpoint, metadata=None):
    from .vmodels import load_vision_checkpoint
    source = Path(baseline_checkpoint).resolve(strict=True)
    backbone, _ = load_vision_checkpoint(source, map_location="cpu")
    expected, actual = backbone.state_dict(), policy.vision_model.state_dict()
    if set(expected) != set(actual) or any(not torch.equal(expected[k].cpu(), actual[k].detach().cpu()) for k in expected):
        raise ValueError("policy backbone differs from SHA-bound source")
    state = {k:v.detach().cpu() for k,v in policy.heads.state_dict().items()}
    for k,v in state.items():
        _finite(v,k)
    payload = dict(schema=SCHEMA, config=asdict(policy.tactical_config), state_dict=state,
                   source=dict(checkpoint=str(source), sha256=checkpoint_sha256(source)), metadata=dict(metadata or {}))
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)


def load_tactical_checkpoint(path, baseline_checkpoint=None, map_location="cpu"):
    from .vmodels import load_vision_checkpoint, vision_arch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"schema","config","state_dict","source","metadata"} or payload["schema"] != SCHEMA:
        raise ValueError("invalid tactical sidecar schema")
    source = payload["source"]
    base = Path(baseline_checkpoint or source["checkpoint"])
    if not base.is_file() or checkpoint_sha256(base) != source["sha256"]:
        raise ValueError("baseline checkpoint SHA256 mismatch")
    backbone, extra = load_vision_checkpoint(base, map_location="cpu")
    if vision_arch(backbone) != "v3-st":
        raise ValueError("tactical sidecar requires v3-st")
    policy = TacticalPolicy(backbone, TacticalConfig(**payload["config"]))
    for k,v in payload["state_dict"].items():
        _finite(v,k)
    policy.heads.load_state_dict(payload["state_dict"], strict=True)
    policy.to(map_location).eval()
    return policy, {**extra, "tactical_source": source, "tactical_schema": SCHEMA, "tactical_metadata":payload["metadata"]}
