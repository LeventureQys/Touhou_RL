"""Policy/value network for the privileged-state observation contract.

Architecture (contract §4): a permutation-invariant encoder over the 64 nearest
bullets, a coarse danger-map encoder, a self-feature MLP, then four action heads
(MultiDiscrete ``[9, 2, 2, 2]``) plus a value head and an auxiliary reward head.

Why a set encoder instead of a CNN over rasterised bullets: danmaku is a set of
objects, and a per-object MLP followed by masked pooling is invariant to bullet
ordering and to *count* (a pattern with 40 bullets and one with 400 share the
same representation path).  PIFE (IEEE CoG 2022) showed this is markedly more
sample-efficient than pixels for bullet-hell games, which matters because TH08
yields only ~30 decisions per second per instance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import obs, spec

N_MOVE, N_SHOOT, N_FOCUS, N_BOMB = spec.ACTION_DIMS
N_REWARD_COMPONENTS = 6
# Candidate actions for the value/scoring network: 9 directions x 2 focus states.
# Shooting is always on and bombing is handled by a rule, so the candidate set is
# exactly the part of the action space the avoidance teacher actually deliberates.
N_CANDIDATES = N_MOVE * N_FOCUS


@dataclass
class NetConfig:
    self_hidden: int = 128
    bullet_hidden: int = 64
    danger_hidden: int = 64
    trunk_hidden: int = 256
    reward_head: bool = True
    candidate_hidden: int = 64


class MaskedSetEncoder(nn.Module):
    """Per-bullet MLP + masked max/mean pooling (permutation invariant)."""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(nn.Linear(hidden * 2, out_features), nn.ReLU(inplace=True))

    def forward(self, bullets: torch.Tensor) -> torch.Tensor:
        # bullets: [B, K, F]; feature F-1 is the validity flag.
        valid = bullets[:, :, -1:].clamp(0.0, 1.0)
        mask = valid > 0.5
        feats = self.embed(bullets)
        neg_inf = torch.finfo(feats.dtype).min
        masked = feats.masked_fill(~mask, neg_inf)
        masked_max = masked.amax(dim=1)
        any_valid = mask.any(dim=1)  # [B, 1]
        masked_max = torch.where(any_valid, masked_max, torch.zeros_like(masked_max))
        denom = valid.sum(dim=1).clamp(min=1.0)
        masked_mean = (feats * valid).sum(dim=1) / denom
        return self.out(torch.cat([masked_max, masked_mean], dim=-1))


class DanmakuPolicy(nn.Module):
    """Shared trunk with multi-discrete action heads and a value head."""

    def __init__(self, config: NetConfig | None = None):
        super().__init__()
        self.config = config or NetConfig()
        cfg = self.config
        self.self_encoder = nn.Sequential(
            nn.Linear(obs.N_SELF, cfg.self_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.self_hidden, cfg.self_hidden),
            nn.ReLU(inplace=True),
        )
        self.bullet_encoder = MaskedSetEncoder(obs.N_BULLET_FEATURES, cfg.bullet_hidden, cfg.self_hidden)
        self.danger_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs.DANGER_ROWS * obs.DANGER_COLS, cfg.danger_hidden),
            nn.ReLU(inplace=True),
        )
        fused = cfg.self_hidden * 2 + cfg.danger_hidden
        self.trunk = nn.Sequential(
            nn.Linear(fused, cfg.trunk_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.trunk_hidden, cfg.trunk_hidden),
            nn.ReLU(inplace=True),
        )
        self.move_head = nn.Linear(cfg.trunk_hidden, N_MOVE)
        self.shoot_head = nn.Linear(cfg.trunk_hidden, N_SHOOT)
        self.focus_head = nn.Linear(cfg.trunk_hidden, N_FOCUS)
        self.bomb_head = nn.Linear(cfg.trunk_hidden, N_BOMB)
        self.value_head = nn.Linear(cfg.trunk_hidden, 1)
        self.reward_head = (
            nn.Linear(cfg.trunk_hidden, N_REWARD_COMPONENTS) if cfg.reward_head else None
        )

    # -- core -------------------------------------------------------------
    def encode(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        self_feat = observation["self_feat"].float()
        bullets = observation["bullets"].float()
        danger = observation["danger"].float()
        return self.trunk(
            torch.cat(
                [
                    self.self_encoder(self_feat),
                    self.bullet_encoder(bullets),
                    self.danger_encoder(danger),
                ],
                dim=-1,
            )
        )

    def forward(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.encode(observation)
        out = {
            "move_logits": self.move_head(features),
            "shoot_logits": self.shoot_head(features),
            "focus_logits": self.focus_head(features),
            "bomb_logits": self.bomb_head(features),
            "value": self.value_head(features).squeeze(-1),
        }
        if self.reward_head is not None:
            out["reward_pred"] = self.reward_head(features)
        return out

    # -- sampling ---------------------------------------------------------
    @staticmethod
    def _heads(out: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        return [out["move_logits"], out["shoot_logits"], out["focus_logits"], out["bomb_logits"]]

    def act(self, observation: dict[str, torch.Tensor], deterministic: bool = False):
        out = self.forward(observation)
        actions = []
        log_probs = []
        entropies = []
        for logits in self._heads(out):
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
        stacked = torch.stack(actions, dim=-1)  # [B, 4]
        return stacked, torch.stack(log_probs, dim=-1).sum(-1), out["value"], torch.stack(entropies, dim=-1).sum(-1)

    def evaluate_actions(self, observation: dict[str, torch.Tensor], actions: torch.Tensor):
        out = self.forward(observation)
        log_probs = []
        entropies = []
        for i, logits in enumerate(self._heads(out)):
            dist = torch.distributions.Categorical(logits=logits)
            log_probs.append(dist.log_prob(actions[:, i]))
            entropies.append(dist.entropy())
        return (
            torch.stack(log_probs, dim=-1).sum(-1),
            torch.stack(entropies, dim=-1).sum(-1),
            out["value"],
        )

    def value(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(observation)["value"]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ScorePolicy(DanmakuPolicy):
    """Candidate-conditioned value network: ``score(observation, action) -> value``.

    Why this exists
    ---------------
    Cloning the avoidance teacher's *argmax* action failed (measured: 1,336 frames
    of survival, no better than the v1.0.2 baseline) even though the teacher
    itself survives 6,502.  The diagnosis was label ambiguity: when several
    moves are equally safe, the teacher's tie-break is arbitrary, so identical
    states carry contradictory labels and the classifier learns their average -
    and the average of "left" and "right" is "stand still", the one action that
    certainly dies.

    This network instead regresses the teacher's *value* for every candidate
    action.  Ties stay ties (equal targets), and one frame yields 18 training
    signals instead of one.  Acting is ``argmax`` over the 18 candidates.

    The trunk is shared with :class:`DanmakuPolicy`, so a scorer checkpoint is
    still the same observation contract and can be compared head to head.
    """

    def __init__(self, config: NetConfig | None = None):
        super().__init__(config)
        cfg = self.config
        self.candidate_embed = nn.Embedding(N_CANDIDATES, cfg.candidate_hidden)
        self.score_head = nn.Sequential(
            nn.Linear(cfg.trunk_hidden + cfg.candidate_hidden, cfg.candidate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.candidate_hidden, 1),
        )

    def score(self, observation: dict[str, torch.Tensor], candidates: torch.Tensor | None = None):
        """[B, C] values. ``candidates`` [B] -> [B]; ``None`` -> all 18 as [B, 18]."""
        features = self.encode(observation)
        if candidates is None:
            index = torch.arange(N_CANDIDATES, device=features.device)
            embedded = self.candidate_embed(index)  # [18, C]
            fused = torch.cat(
                [
                    features.unsqueeze(1).expand(-1, N_CANDIDATES, -1),
                    embedded.unsqueeze(0).expand(features.size(0), -1, -1),
                ],
                dim=-1,
            )
        else:
            fused = torch.cat([features, self.candidate_embed(candidates)], dim=-1)
        return self.score_head(fused).squeeze(-1)

    def act(self, observation: dict[str, torch.Tensor], deterministic: bool = True):
        """Greedy over the candidate set; returns the same tuple shape as the policy."""
        values = self.score(observation)
        best = values.argmax(dim=-1)
        move = best % N_MOVE
        focus = best // N_MOVE
        ones = torch.ones_like(move)
        zeros = torch.zeros_like(move)
        actions = torch.stack([move, ones, focus, zeros], dim=-1)
        zero = torch.zeros(values.size(0), dtype=values.dtype, device=values.device)
        return actions, zero, values.max(dim=-1).values, zero


def save_checkpoint(path, model: DanmakuPolicy, extra: dict | None = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "config": model.config.__dict__,
        "kind": "scorer" if isinstance(model, ScorePolicy) else "policy",
        "obs": {
            "N_SELF": obs.N_SELF,
            "N_BULLETS": obs.N_BULLETS,
            "N_BULLET_FEATURES": obs.N_BULLET_FEATURES,
            "DANGER_ROWS": obs.DANGER_ROWS,
            "DANGER_COLS": obs.DANGER_COLS,
        },
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu") -> tuple[DanmakuPolicy, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload["obs"]["N_SELF"] != obs.N_SELF or payload["obs"]["N_BULLETS"] != obs.N_BULLETS:
        raise ValueError("checkpoint observation contract does not match this build")
    cls = ScorePolicy if payload.get("kind") == "scorer" else DanmakuPolicy
    model = cls(NetConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model, payload.get("extra", {})


def observation_from_numpy(batch: dict, device) -> dict[str, torch.Tensor]:
    """Convert one encoded observation (NumPy) into a batched torch dict.

    This is the *only* accepted way to feed a live observation to a model: the
    training loader scales the danger map by 1/8, and skipping that at inference
    feeds the network inputs eight times larger than it ever saw (a real bug
    found during v1.0.2 verification).
    """
    return {
        "self_feat": torch.as_tensor(batch["self_feat"], dtype=torch.float32, device=device).unsqueeze(0),
        "bullets": torch.as_tensor(batch["bullets"], dtype=torch.float32, device=device).unsqueeze(0),
        "danger": torch.as_tensor(batch["danger"], dtype=torch.float32, device=device).unsqueeze(0) / 8.0,
    }


def observation_batch(observation: dict, device) -> dict[str, torch.Tensor]:
    """Alias used by evaluate/play/ppo; see :func:`observation_from_numpy`."""
    return observation_from_numpy(observation, device)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None):
    return F.cross_entropy(logits, targets, weight=weight)
