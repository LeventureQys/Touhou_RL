"""18-way privileged policy for step 2 (entity set in, candidate out).

Relationship to the existing models
-----------------------------------
``models.DanmakuPolicy`` consumes the fixed ``[64,6]`` top-64 observation and emits
four factorised heads (move/shoot/focus/bomb = 72 combinations).  Step 2 needs a
*set* of up to 320 bullets and a **single 18-way candidate** head, so that the
policy speaks the same action language as the scripted teacher
(``focus*9 + move``) and as the step-3 distillation target.  The masked pooling
idea is the same one ``models.MaskedSetEncoder`` already uses.

Contract
--------
* input  ``bullets float32[B,K,6]`` (see :mod:`touhou_rl.entity_obs`), ``meta float32[B,4]``
* output ``logits float32[B,18]``, ``value float32[B]``
* permutation invariance: permuting the ``K`` axis must not change the output --
  that is what makes the encoder a set encoder, and it is unit-tested.
* checkpoint payload carries ``candidate_order`` so a checkpoint trained under a
  different action order can never be loaded silently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import entity_obs

CANDIDATE_ORDER = "focus*9+move"
CHECKPOINT_SCHEMA = "v311-entity-policy/1"
N_CANDIDATES = 18


@dataclass
class EntityPolicyConfig:
    k_bullets: int = entity_obs.K_BULLETS
    bullet_features: int = entity_obs.BULLET_FEATURES
    meta_dim: int = entity_obs.META_DIM
    grid_channels: int = entity_obs.GRID_CHANNELS
    grid_rows: int = entity_obs.GRID_ROWS
    grid_cols: int = entity_obs.GRID_COLS
    grid_hidden: int = 64
    object_hidden: int = 64
    trunk_hidden: int = 256
    n_candidates: int = N_CANDIDATES
    candidate_order: str = CANDIDATE_ORDER

    def validate(self) -> None:
        if self.candidate_order != CANDIDATE_ORDER:
            raise ValueError(
                f"candidate_order must be {CANDIDATE_ORDER!r}, got {self.candidate_order!r}"
            )
        if self.n_candidates != N_CANDIDATES:
            raise ValueError(f"n_candidates must be {N_CANDIDATES}, got {self.n_candidates}")
        for name in (
            "k_bullets",
            "bullet_features",
            "meta_dim",
            "grid_channels",
            "grid_rows",
            "grid_cols",
            "grid_hidden",
            "object_hidden",
            "trunk_hidden",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


def _masked_pool(features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Max + mean over the object axis, ignoring padded rows."""
    mask = valid.unsqueeze(-1)
    masked_max = features.masked_fill(~mask, float("-inf")).amax(dim=1)
    masked_max = torch.where(torch.isfinite(masked_max), masked_max, torch.zeros_like(masked_max))
    total = (features * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    masked_mean = total / counts
    return torch.cat([masked_max, masked_mean], dim=-1)


class EntityPolicy(nn.Module):
    """Permutation-invariant bullet encoder + 18-way candidate head."""

    def __init__(self, config: EntityPolicyConfig | None = None):
        super().__init__()
        self.config = config or EntityPolicyConfig()
        self.config.validate()
        self.object_mlp = nn.Sequential(
            nn.Linear(self.config.bullet_features, self.config.object_hidden),
            nn.ReLU(),
            nn.Linear(self.config.object_hidden, self.config.object_hidden),
            nn.ReLU(),
        )
        self.grid_conv = nn.Sequential(
            nn.Conv2d(self.config.grid_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, self.config.grid_hidden, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        pooled = 2 * self.config.object_hidden
        self.trunk = nn.Sequential(
            nn.Linear(
                pooled + self.config.meta_dim + self.config.grid_hidden, self.config.trunk_hidden
            ),
            nn.ReLU(),
            nn.Linear(self.config.trunk_hidden, self.config.trunk_hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(self.config.trunk_hidden, self.config.n_candidates)
        self.value_head = nn.Linear(self.config.trunk_hidden, 1)

    # -- forward ---------------------------------------------------------
    def features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        bullets = batch["bullets"]
        meta = batch["meta"]
        grid = batch["grid"]
        if bullets.ndim != 3 or bullets.shape[-1] != self.config.bullet_features:
            raise ValueError(f"bullets must be [B,K,{self.config.bullet_features}], got {tuple(bullets.shape)}")
        if bullets.shape[1] > self.config.k_bullets:
            raise ValueError(f"bullets K={bullets.shape[1]} > configured {self.config.k_bullets}")
        if meta.ndim != 2 or meta.shape[-1] != self.config.meta_dim:
            raise ValueError(f"meta must be [B,{self.config.meta_dim}], got {tuple(meta.shape)}")
        expected_grid = (self.config.grid_channels, self.config.grid_rows, self.config.grid_cols)
        if grid.ndim != 4 or tuple(grid.shape[1:]) != expected_grid:
            raise ValueError(f"grid must be [B,{expected_grid}], got {tuple(grid.shape)}")
        encoded = self.object_mlp(bullets)
        pooled = _masked_pool(encoded, bullets[..., -1] > 0.5)
        return self.trunk(torch.cat([pooled, meta, self.grid_conv(grid)], dim=-1))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.features(batch)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    # -- policy API ------------------------------------------------------
    def act(
        self, batch: dict[str, torch.Tensor], deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(batch)
        dist = torch.distributions.Categorical(logits=logits)
        candidate = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return candidate, dist.log_prob(candidate), value, dist.entropy()

    def evaluate_candidates(
        self, batch: dict[str, torch.Tensor], candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(batch)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(candidates), dist.entropy(), value


def to_tensors(batch: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {
        "bullets": torch.as_tensor(np.asarray(batch["bullets"], dtype=np.float32), device=device),
        "meta": torch.as_tensor(np.asarray(batch["meta"], dtype=np.float32), device=device),
        "grid": torch.as_tensor(np.asarray(batch["grid"], dtype=np.float32), device=device),
    }


def save_checkpoint(path: Path, model: EntityPolicy, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "candidate_order": CANDIDATE_ORDER,
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, map_location="cpu") -> tuple[EntityPolicy, dict]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"not an entity policy checkpoint: schema={payload.get('schema')!r}")
    if payload.get("candidate_order") != CANDIDATE_ORDER:
        raise ValueError(
            f"candidate order mismatch: {payload.get('candidate_order')!r} != {CANDIDATE_ORDER!r}"
        )
    config = EntityPolicyConfig(**payload["config"])
    model = EntityPolicy(config)
    model.load_state_dict(payload["state_dict"])
    return model, payload.get("extra", {})


__all__ = [
    "CANDIDATE_ORDER",
    "CHECKPOINT_SCHEMA",
    "EntityPolicy",
    "EntityPolicyConfig",
    "N_CANDIDATES",
    "load_checkpoint",
    "save_checkpoint",
    "to_tensors",
]
