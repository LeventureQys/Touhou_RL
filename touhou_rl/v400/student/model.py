"""Small pixel student for obs-v1: play64 4-frame stack + HUD strip.

This is the **V0 smoke** network, not the P3 target architecture.  13-P3 §2.3
targets 2–6 M parameters with an IMPALA-ResNet encoder and optional GRU; the V0
smoke is deliberately smaller so a five-minute behaviour-cloning baseline can be
trained and evaluated end to end.  Its job is to prove the observation, dataset
and evaluation contracts, not to reach G3.

Layout::

    play  (B,4,64,64)  -> conv 32 s2 -> conv 64 s2 -> conv 64 s2 -> GAP -> 64
    hud   (B,1,56,160) -> conv 16 s2 -> conv 32 s2 -> conv 32 s2 -> GAP -> 32
    concat(96) -> Linear 128 -> ReLU -> move_logits[18], bomb_logits[2]
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from touhou_rl.v400.student.obs_spec import (
    FRAME_STACK,
    HUD_FRAMES,
    HUD_SHAPE,
    N_BOMB_CLASSES,
    N_MOVE_CLASSES,
    OBS_SPEC_VERSION,
    PLAY_SHAPE,
    check_obs_spec,
    obs_spec_digest,
)

CHECKPOINT_SCHEMA = "v400-student-checkpoint-1"


class StudentNet(nn.Module):
    """Tiny twin-stream policy: pixels in, move-focus and bomb logits out."""

    def __init__(self, width: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.width = int(width)
        self.hidden = int(hidden)
        self.play = nn.Sequential(
            nn.Conv2d(FRAME_STACK, width, 5, stride=2, padding=2),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.hud = nn.Sequential(
            nn.Conv2d(HUD_FRAMES, width // 2, 5, stride=2, padding=2),
            nn.BatchNorm2d(width // 2), nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        features = width * 2 + width
        self.trunk = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(inplace=True))
        self.move_head = nn.Linear(hidden, N_MOVE_CLASSES)
        self.bomb_head = nn.Linear(hidden, N_BOMB_CLASSES)

    def forward(self, play: torch.Tensor, hud: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if play.ndim != 4 or play.shape[1:] != (FRAME_STACK, *PLAY_SHAPE):
            raise ValueError(f"play batch must be (B,{FRAME_STACK},{PLAY_SHAPE[0]},{PLAY_SHAPE[1]}), "
                             f"got {tuple(play.shape)}")
        if hud.ndim != 4 or hud.shape[1:] != (HUD_FRAMES, *HUD_SHAPE):
            raise ValueError(f"hud batch must be (B,{HUD_FRAMES},{HUD_SHAPE[0]},{HUD_SHAPE[1]}), "
                             f"got {tuple(hud.shape)}")
        features = torch.cat([self.play(play).flatten(1), self.hud(hud).flatten(1)], dim=1)
        hidden = self.trunk(features)
        return self.move_head(hidden), self.bomb_head(hidden)

    @torch.no_grad()
    def act(self, play: torch.Tensor, hud: torch.Tensor) -> tuple[int, int]:
        """Greedy ``(cand, bomb)`` for a single observation batch of size 1."""

        self.eval()
        move_logits, bomb_logits = self(play, hud)
        return int(move_logits.argmax(dim=1)[0]), int(bomb_logits.argmax(dim=1)[0])

    def config(self) -> dict[str, int]:
        return {"width": self.width, "hidden": self.hidden}

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def save_checkpoint(
    path: str | Path,
    model: StudentNet,
    *,
    train: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint that records the frozen obs spec it belongs to."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "obs_spec_version": OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec_digest(),
        "model": {"class": "StudentNet", **model.config()},
        "parameters": model.n_parameters(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "train": dict(train or {}),
        "provenance": dict(provenance or {}),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[StudentNet, dict[str, Any]]:
    """Load and verify a checkpoint; refuses a different obs spec."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"not a {CHECKPOINT_SCHEMA} checkpoint: {path}")
    check_obs_spec(payload.get("obs_spec_version"), payload.get("obs_spec_digest"))
    config = payload.get("model") or {}
    model = StudentNet(width=int(config.get("width", 32)), hidden=int(config.get("hidden", 128)))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def checkpoint_digest(path: str | Path) -> str:
    """sha256 of a checkpoint file, for evaluation provenance."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe_checkpoint(payload: dict[str, Any]) -> str:
    """One-line human summary used in reports."""

    return (f"{payload.get('schema')} obs={payload.get('obs_spec_version')} "
            f"params={payload.get('parameters')} " + json.dumps(payload.get("model", {}),
                                                                sort_keys=True))


__all__ = [
    "CHECKPOINT_SCHEMA",
    "StudentNet",
    "checkpoint_digest",
    "describe_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
]
