"""obs-v2 student networks: M2 (MLP) and M3 (small GRU) — EXP-P3-008.

Both arms share the **same per-decision encoder** as obs-v1's :class:`StudentNet`
(``play64`` 4-frame stack + HUD strip, width 48) and add the only new obs-v2 input,
``act_hist``:

    play (4,64,64) -> convs -> 96      }  obs-v1 trunk, unchanged
    hud  (1,56,160) -> convs -> 48     }
    act_hist (8,18) -> flatten 144 -> Linear 64 -> ReLU   (obs-v2 addition)
    features = 208

``StudentNetV2`` (M2, **M**ulti-**L**ayer **P**erceptron)
    ``features -> Linear 256 -> ReLU -> move_logits[18], bomb_logits[2]``
    (204k → 267k parameters: the +63k is exactly the history encoder, so the arm
    isolates "obs-v2 input" from "more capacity").

``StudentGRU`` (M3)
    ``features -> Linear 256 -> GRU(256, hidden=256) -> move_logits[18],
    bomb_logits[2]``.  The sequence is 8 consecutive **decisions**
    (``seq_len=8``), and training backpropagates through exactly those 8 steps
    (``bptt=8`` == seq_len, so the truncation is "the window", not a shorter prefix).
    At deployment the per-decision embeddings are cached, so each new decision costs
    one encoder pass plus one 8-step GRU pass — the same arithmetic as training.

Both classes expose the obs-v2 deployment API::

    act(play, hud, act_hist) -> (cand, bomb)          # MLP
    act(play, hud, act_hist, cache=cache) -> (...)    # GRU, cache carries the seq

Checkpoints record the obs-v2 version and digest and refuse to load under another
spec; an obs-v1 checkpoint still loads through
:func:`touhou_rl.v400.student.model.load_checkpoint` and is never readable here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from touhou_rl.v400.student import obs_spec
from touhou_rl.v400.student.obs_spec_v2 import (
    ACTION_HISTORY_DECISIONS,
    ACTION_HISTORY_DIM,
    OBS_SPEC_VERSION,
    check_obs_spec_v2,
    obs_spec_v2_digest,
)

CHECKPOINT_SCHEMA_V2 = "v400-student-checkpoint-2"
#: ``arch`` strings a v2 checkpoint may carry; the loader dispatches on it.
ARCH_MLP = "mlp"
ARCH_GRU = "gru"
ARCHS = (ARCH_MLP, ARCH_GRU)


class ObsEncoderV2(nn.Module):
    """Per-decision encoder: obs-v1 trunks + the obs-v2 action-history encoder."""

    def __init__(self, width: int = 48, hist_hidden: int = 64) -> None:
        super().__init__()
        self.width = int(width)
        self.hist_hidden = int(hist_hidden)
        self.play = nn.Sequential(
            nn.Conv2d(obs_spec.FRAME_STACK, width, 5, stride=2, padding=2),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.hud = nn.Sequential(
            nn.Conv2d(obs_spec.HUD_FRAMES, width // 2, 5, stride=2, padding=2),
            nn.BatchNorm2d(width // 2), nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.hist = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ACTION_HISTORY_DIM, hist_hidden), nn.ReLU(inplace=True),
        )

    @property
    def features(self) -> int:
        return self.width * 2 + self.width + self.hist_hidden

    def forward(self, play: torch.Tensor, hud: torch.Tensor,
                act_hist: torch.Tensor) -> torch.Tensor:
        if play.ndim != 5 or play.shape[2:] != (obs_spec.FRAME_STACK, *obs_spec.PLAY_SHAPE):
            raise ValueError(
                f"play must be (S,B,{obs_spec.FRAME_STACK},{obs_spec.PLAY_SHAPE[0]},"
                f"{obs_spec.PLAY_SHAPE[1]}), got {tuple(play.shape)}")
        if hud.shape[2:] != (obs_spec.HUD_FRAMES, *obs_spec.HUD_SHAPE):
            raise ValueError(f"hud must be (S,B,{obs_spec.HUD_FRAMES},"
                             f"{obs_spec.HUD_SHAPE[0]},{obs_spec.HUD_SHAPE[1]}), "
                             f"got {tuple(hud.shape)}")
        if act_hist.shape[2:] != (ACTION_HISTORY_DECISIONS, obs_spec.N_MOVE_CLASSES):
            raise ValueError(
                f"act_hist must be (S,B,{ACTION_HISTORY_DECISIONS},"
                f"{obs_spec.N_MOVE_CLASSES}), got {tuple(act_hist.shape)}")
        steps, batch = play.shape[0], play.shape[1]
        flat_play = play.reshape(steps * batch, *play.shape[2:])
        flat_hud = hud.reshape(steps * batch, *hud.shape[2:])
        flat_hist = act_hist.reshape(steps * batch, *act_hist.shape[2:])
        features = torch.cat([
            self.play(flat_play).flatten(1),
            self.hud(flat_hud).flatten(1),
            self.hist(flat_hist.float()),
        ], dim=1)
        return features.reshape(steps, batch, -1)


class StudentNetV2(nn.Module):
    """M2: obs-v2 + feed-forward MLP (no temporal memory beyond ``act_hist``)."""

    arch = ARCH_MLP

    def __init__(self, width: int = 48, hidden: int = 256, hist_hidden: int = 64) -> None:
        super().__init__()
        self.encoder = ObsEncoderV2(width=width, hist_hidden=hist_hidden)
        self.hidden = int(hidden)
        self.trunk = nn.Sequential(nn.Linear(self.encoder.features, hidden),
                                   nn.ReLU(inplace=True))
        self.move_head = nn.Linear(hidden, obs_spec.N_MOVE_CLASSES)
        self.bomb_head = nn.Linear(hidden, obs_spec.N_BOMB_CLASSES)

    def forward(self, play: torch.Tensor, hud: torch.Tensor,
                act_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (1,B,...) because a feed-forward arm has a sequence length of 1.
        features = self.encoder(play[None], hud[None], act_hist[None])[0]
        hidden = self.trunk(features)
        return self.move_head(hidden), self.bomb_head(hidden)

    @torch.no_grad()
    def act(self, play: torch.Tensor, hud: torch.Tensor,
            act_hist: torch.Tensor) -> tuple[int, int]:
        self.eval()
        move_logits, bomb_logits = self(play, hud, act_hist)
        return int(move_logits.argmax(dim=1)[0]), int(bomb_logits.argmax(dim=1)[0])

    def config(self) -> dict[str, int | str]:
        return {"arch": self.arch, "width": self.encoder.width,
                "hidden": self.hidden, "hist_hidden": self.encoder.hist_hidden}

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class GRUCache:
    """Per-episode cache of the last ``seq_len`` per-decision embeddings."""

    def __init__(self, seq_len: int = ACTION_HISTORY_DECISIONS) -> None:
        self.seq_len = int(seq_len)
        self._embeddings: list[torch.Tensor] = []
        self.steps = 0

    def reset(self) -> None:
        self._embeddings = []
        self.steps = 0

    def push(self, embedding: torch.Tensor) -> torch.Tensor:
        """Append one ``(1, features)`` embedding, return the ``(1,seq,features)`` window."""

        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError(f"embedding must be (1,F), got {tuple(embedding.shape)}")
        self._embeddings.append(embedding)
        if len(self._embeddings) > self.seq_len:
            del self._embeddings[0]
        self.steps += 1
        frames = list(self._embeddings)
        while len(frames) < self.seq_len:
            # Episode-start padding: repeat the first decision's embedding, exactly
            # the idea of obs-v1's frame padding (deterministic, recorded).
            frames.insert(0, frames[0])
        # (seq, batch, features): the GRU is batch_first=False, so stacking on dim 1
        # would feed a batch of 8 single-step sequences and silently delete the memory.
        return torch.stack(frames, dim=0)


class StudentGRU(nn.Module):
    """M3: obs-v2 + a single-layer GRU over 8 consecutive decisions."""

    arch = ARCH_GRU

    def __init__(self, width: int = 48, hidden: int = 256, hist_hidden: int = 64,
                 gru_input: int = 128, seq_len: int = ACTION_HISTORY_DECISIONS) -> None:
        super().__init__()
        self.encoder = ObsEncoderV2(width=width, hist_hidden=hist_hidden)
        self.hidden = int(hidden)
        self.seq_len = int(seq_len)
        self.input = nn.Sequential(nn.Linear(self.encoder.features, gru_input),
                                   nn.ReLU(inplace=True))
        self.gru = nn.GRU(input_size=gru_input, hidden_size=hidden, num_layers=1,
                          batch_first=False)
        self.move_head = nn.Linear(hidden, obs_spec.N_MOVE_CLASSES)
        self.bomb_head = nn.Linear(hidden, obs_spec.N_BOMB_CLASSES)

    def encode(self, play: torch.Tensor, hud: torch.Tensor,
               act_hist: torch.Tensor) -> torch.Tensor:
        """``(S,B,...)`` -> ``(S,B,gru_input)`` per-decision GRU inputs."""

        return self.input(self.encoder(play, hud, act_hist))

    def forward(self, play: torch.Tensor, hud: torch.Tensor,
                act_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``play (S,B,4,64,64)`` -> logits for **every** step ``(S,B,18)``."""

        steps = play.shape[0]
        if steps != self.seq_len:
            raise ValueError(f"GRU arm expects seq_len={self.seq_len} steps, got {steps}")
        sequence = self.encode(play, hud, act_hist)
        output, _ = self.gru(sequence)
        return self.move_head(output), self.bomb_head(output)

    @torch.no_grad()
    def act_step(self, play: torch.Tensor, hud: torch.Tensor, act_hist: torch.Tensor,
                 cache: GRUCache) -> tuple[int, int]:
        """One decision: encode the new observation, append to the cache, read the last step."""

        self.eval()
        embedding = self.input(self.encoder(play[None], hud[None], act_hist[None])[0])
        window = cache.push(embedding)
        output, _ = self.gru(window)
        move_logits = self.move_head(output[-1])
        bomb_logits = self.bomb_head(output[-1])
        return int(move_logits.argmax(dim=1)[0]), int(bomb_logits.argmax(dim=1)[0])

    @torch.no_grad()
    def act(self, play: torch.Tensor, hud: torch.Tensor, act_hist: torch.Tensor,
            cache: GRUCache | None = None) -> tuple[int, int]:
        if cache is None:
            raise ValueError("the GRU arm needs a per-episode GRUCache")
        return self.act_step(play, hud, act_hist, cache)

    def config(self) -> dict[str, int | str]:
        return {"arch": self.arch, "width": self.encoder.width,
                "hidden": self.hidden, "hist_hidden": self.encoder.hist_hidden,
                "gru_input": int(self.input[0].out_features), "seq_len": self.seq_len}

    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(config: dict[str, Any]) -> nn.Module:
    """Instantiate from a checkpoint's ``model`` block (fail closed on unknown arch)."""

    arch = str(config.get("arch", ""))
    if arch == ARCH_MLP:
        return StudentNetV2(width=int(config["width"]), hidden=int(config["hidden"]),
                            hist_hidden=int(config["hist_hidden"]))
    if arch == ARCH_GRU:
        return StudentGRU(width=int(config["width"]), hidden=int(config["hidden"]),
                          hist_hidden=int(config["hist_hidden"]),
                          gru_input=int(config["gru_input"]),
                          seq_len=int(config["seq_len"]))
    raise ValueError(f"unknown obs-v2 arch {arch!r}; expected one of {ARCHS}")


def save_checkpoint_v2(
    path: str | Path,
    model: nn.Module,
    *,
    train: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write an obs-v2 checkpoint (records the frozen obs-v2 spec + digest)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA_V2,
        "obs_spec_version": OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec_v2_digest(),
        "base_obs_spec_version": obs_spec.OBS_SPEC_VERSION,
        "base_obs_spec_digest": obs_spec.obs_spec_digest(),
        "model": {"class": type(model).__name__, **model.config()},
        "parameters": model.n_parameters(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "train": dict(train or {}),
        "provenance": dict(provenance or {}),
    }
    torch.save(payload, path)
    return path


def load_checkpoint_v2(path: str | Path, *, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """Load and verify an obs-v2 checkpoint; refuses another spec or another arch."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA_V2:
        raise ValueError(f"not a {CHECKPOINT_SCHEMA_V2} checkpoint: {path}")
    check_obs_spec_v2(payload.get("obs_spec_version"), payload.get("obs_spec_digest"))
    if payload.get("base_obs_spec_digest") != obs_spec.obs_spec_digest():
        raise ValueError(
            f"{path}: base obs-v1 digest {payload.get('base_obs_spec_digest')!r} is not "
            f"the frozen {obs_spec.obs_spec_digest()!r}")
    model = build_model(payload.get("model") or {})
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def describe_checkpoint_v2(payload: dict[str, Any]) -> str:
    return (f"{payload.get('schema')} obs={payload.get('obs_spec_version')} "
            f"params={payload.get('parameters')} arch={(payload.get('model') or {}).get('arch')}")


def checkpoint_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "ARCHS",
    "ARCH_GRU",
    "ARCH_MLP",
    "CHECKPOINT_SCHEMA_V2",
    "GRUCache",
    "ObsEncoderV2",
    "StudentGRU",
    "StudentNetV2",
    "build_model",
    "checkpoint_digest",
    "describe_checkpoint_v2",
    "load_checkpoint_v2",
    "save_checkpoint_v2",
]
