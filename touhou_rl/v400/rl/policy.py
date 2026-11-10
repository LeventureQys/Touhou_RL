"""Actor-critic for the P4 RL prototype, initialised from the BC ``m1`` checkpoint.

Two heads, because obs-v1 already froze that shape: the 18 move-focus
candidates and the 2-way bomb dimension.  The joint action is
``action = cand * 2 + bomb`` (the order of ``STUDENT_ACTIONS``), so

    log pi(a) = log_softmax(move_logits)[cand] + log_softmax(bomb_logits)[bomb]
    H[pi]     = H[cand] + H[bomb]

which is exactly the factorisation ``StudentNet`` was trained under and exactly
the factorisation ``tools/rl/v400_student_eval.py`` reads back.  The trunk is the
BC network: :func:`build_actor_critic` loads the checkpoint with ``strict=True``,
so a shape or naming drift fails loudly instead of silently training a fresh
policy.

BatchNorm in the BC network is kept in **eval mode** for the whole RL run.
``StudentNet.act`` calls ``self.eval()`` at deployment, so the running statistics
are the normalisation a deployment episode actually uses; letting PPO overwrite
them would train against one normalisation and evaluate against another.  The
affine parameters still receive gradients (eval mode only freezes the statistics),
so the policy can adapt while every rollout and every evaluation see the same
pixels-to-features map.

The value head is new and randomly initialised (orthogonal, gain 1.0).  PPO
therefore starts with an uncalibrated critic; ``tools/rl/v400_rl_train.py`` spends
its first few updates on the value head alone so its large early gradients cannot
tear the cloned policy away from the BC solution before there is any return
signal to justify it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from touhou_rl.v400.rl.env import N_ACTIONS, N_BOMB_CLASSES
from touhou_rl.v400.student.model import (
    StudentNet,
    checkpoint_digest,
    load_checkpoint,
    save_checkpoint,
)
from touhou_rl.v400.student.obs_spec import N_MOVE_CLASSES


class ActorCritic(nn.Module):
    """``StudentNet`` trunk + two policy heads (from BC) + a new value head."""

    def __init__(self, actor: StudentNet) -> None:
        super().__init__()
        self.actor = actor
        self.hidden = int(actor.hidden)
        self.value_head = nn.Linear(self.hidden, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
        # Frozen normalisation statistics (see the module docstring).
        self._freeze_norm_stats()

    # -- normalisation ----------------------------------------------------
    def _freeze_norm_stats(self) -> None:
        for module in self.actor.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "ActorCritic":
        """``train()`` that never un-freezes the actor's BatchNorm statistics."""

        super().train(mode)
        self._freeze_norm_stats()
        return self

    # -- forward ----------------------------------------------------------
    def _features(self, play: torch.Tensor, hud: torch.Tensor) -> torch.Tensor:
        actor = self.actor
        if play.ndim != 4:
            raise ValueError(f"play batch must be (B,4,64,64), got {tuple(play.shape)}")
        features = torch.cat(
            [actor.play(play).flatten(1), actor.hud(hud).flatten(1)], dim=1)
        return actor.trunk(features)

    def forward(self, play: torch.Tensor, hud: torch.Tensor):
        hidden = self._features(play, hud)
        return (self.actor.move_head(hidden), self.actor.bomb_head(hidden),
                self.value_head(hidden).squeeze(-1))

    # -- distributions ----------------------------------------------------
    @staticmethod
    def joint_log_prob(move_logits: torch.Tensor, bomb_logits: torch.Tensor,
                       actions: torch.Tensor) -> torch.Tensor:
        cand = torch.div(actions, N_BOMB_CLASSES, rounding_mode="floor")
        bomb = actions.remainder(N_BOMB_CLASSES)
        move_logp = torch.log_softmax(move_logits, dim=-1).gather(
            1, cand.unsqueeze(1)).squeeze(1)
        bomb_logp = torch.log_softmax(bomb_logits, dim=-1).gather(
            1, bomb.unsqueeze(1)).squeeze(1)
        return move_logp + bomb_logp

    @staticmethod
    def entropy(move_logits: torch.Tensor, bomb_logits: torch.Tensor) -> torch.Tensor:
        return (Categorical(logits=move_logits).entropy()
                + Categorical(logits=bomb_logits).entropy())

    def distribution(self, move_logits: torch.Tensor, bomb_logits: torch.Tensor):
        return Categorical(logits=move_logits), Categorical(logits=bomb_logits)

    @torch.no_grad()
    def act(self, play: torch.Tensor, hud: torch.Tensor, *, deterministic: bool = False):
        """Sample (or argmax) one joint action for a batch; returns tensors."""

        move_logits, bomb_logits, value = self(play, hud)
        if deterministic:
            cand = move_logits.argmax(dim=-1)
            bomb = bomb_logits.argmax(dim=-1)
        else:
            cand = Categorical(logits=move_logits).sample()
            bomb = Categorical(logits=bomb_logits).sample()
        actions = cand * N_BOMB_CLASSES + bomb
        log_prob = self.joint_log_prob(move_logits, bomb_logits, actions)
        return actions, log_prob, value

    # -- introspection ----------------------------------------------------
    def n_parameters(self) -> dict[str, int]:
        actor = sum(parameter.numel() for parameter in self.actor.parameters())
        value = sum(parameter.numel() for parameter in self.value_head.parameters())
        return {"actor": actor, "value_head": value, "total": actor + value}

    @property
    def obs_spec(self):  # pragma: no cover - documented shortcut
        from touhou_rl.v400.student.obs_spec import obs_spec_digest, OBS_SPEC_VERSION

        return OBS_SPEC_VERSION, obs_spec_digest()


def build_actor_critic(bc_checkpoint: str | Path | None, *, device: str = "cpu",
                       seed: int | None = None) -> tuple[ActorCritic, dict[str, Any]]:
    """Load the BC actor (strictly) and attach a fresh value head.

    ``bc_checkpoint=None`` builds an untrained ``StudentNet(width=48, hidden=256)``
    so an ablation can be run from scratch; the returned metadata says which
    happened, and the card must print it.
    """

    if seed is not None:
        torch.manual_seed(int(seed))
    provenance: dict[str, Any] = {"init": "scratch", "bc_checkpoint": None}
    if bc_checkpoint is None:
        actor = StudentNet(width=48, hidden=256)
    else:
        actor, payload = load_checkpoint(bc_checkpoint, device=device)
        provenance = {
            "init": "bc-checkpoint",
            "bc_checkpoint": str(bc_checkpoint),
            "bc_checkpoint_sha256": checkpoint_digest(bc_checkpoint),
            "bc_parameters": int(payload.get("parameters") or 0),
            "bc_obs_spec_version": payload.get("obs_spec_version"),
            "bc_obs_spec_digest": payload.get("obs_spec_digest"),
            "bc_train": {key: payload.get("train", {}).get(key)
                         for key in ("bc_version", "tau")},
        }
    model = ActorCritic(actor).to(device)
    provenance.update(model.n_parameters())
    return model, provenance


def save_policy_checkpoint(
    path: str | Path,
    model: ActorCritic,
    *,
    train: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write the **actor** as a ``v400-student-checkpoint-1`` file.

    The evaluation controller (``tools/rl/v400_student_eval.py``) loads exactly
    this schema, so an RL checkpoint is measured on the same ruler as the BC
    baseline with no conversion step and no second definition of "the policy".
    The value head is deliberately not in ``state_dict``: it is training-only and
    its absence is recorded in ``provenance``.
    """

    train_payload = dict(train or {})
    provenance_payload = dict(provenance or {})
    provenance_payload.setdefault(
        "rl_note",
        "actor saved in v400-student-checkpoint-1 so the S1N evaluator reads it "
        "unchanged; the value head is a training-only module and is not saved")
    return save_checkpoint(path, model.actor, train=train_payload,
                           provenance=provenance_payload)


__all__ = [
    "ActorCritic",
    "N_ACTIONS",
    "N_MOVE_CLASSES",
    "build_actor_critic",
    "save_policy_checkpoint",
]
