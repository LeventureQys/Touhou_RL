"""Vision models: one CNN trunk feeding a privileged-state decoder and a policy.

Design decisions worth defending
--------------------------------
**One trunk, two heads.**  The trunk must answer two questions about the same
picture: "what is on screen" (privileged state) and "what should I press"
(candidate values).  Training the state head first gives the policy head a trunk
that already sees bullets instead of one that has to invent geometry from a
sparse action signal - which matters because at 30 decisions/s the action signal
is the expensive one, while privileged labels are free (we read them).

**Candidate-value policy, not four action heads.**  This project measured that
cloning a planner's argmax collapses (ties become contradictory labels, the
average of "left" and "right" is "stand still").  The privileged teacher outputs
a value per candidate; the vision policy regresses those values and takes
``argmax``.  Ties stay ties.

**No recurrence, explicit stack.**  Motion has to come from somewhere.  A
``depth``-frame stack is passed as extra input channels; that keeps the model
stateless and the training/eval paths identical, at the cost of 3x input size for
depth 3.  It is a deliberate simplification - a GRU could be added behind the
same interface later.

Shapes (from :mod:`touhou_rl.vision`): input ``[B, depth, 128, 112, 3]`` uint8
scaled to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from . import obs, spec, vision

N_CANDIDATES = 18  # 9 directions x 2 focus states, same set the teacher scores


@dataclass
class VisionNetConfig:
    depth: int = 3
    width: int = 32  # channels of the first conv block
    trunk_hidden: int = 256
    candidate_hidden: int = 64
    state_hidden: int = 256
    state_head: bool = True
    danger_cells: int = obs.DANGER_ROWS * obs.DANGER_COLS
    # Model input size per screen region.  The archive keeps native resolution
    # (playfield 448x384, hud_score 34x216, hud_status 126x216); the model trades
    # that down for compute, but the *HUD stays legible* because those regions are
    # already small.  Tuple order is (H, W).
    model_regions: tuple[tuple[str, int, int], ...] = (
        ("playfield", 224, 192),
        ("hud_score", 34, 216),
        ("hud_status", 126, 216),
    )

    @property
    def region_names(self) -> tuple[str, ...]:
        return tuple(name for name, _h, _w in self.model_regions)

    @property
    def region_sizes(self) -> dict[str, tuple[int, int]]:
        return {name: (height, width) for name, height, width in self.model_regions}


class VisionEncoder(nn.Module):
    """Per-region CNN with shared weights, then a trunk.

    Weights are shared across regions on purpose: a region is just "an image", and
    sharing keeps the parameter count small enough that a handful of small HUD
    streams cost almost nothing.  Each region is pooled to a fixed 2x2 before
    concatenation so the trunk is independent of the input sizes.
    """

    def __init__(self, config: VisionNetConfig | None = None):
        super().__init__()
        self.config = config or VisionNetConfig()
        width = self.config.width
        self.body = nn.Sequential(
            nn.Conv2d(vision.VISION_CHANNELS * self.config.depth, width, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 3, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 3, width * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 4 * len(self.config.region_names), self.config.trunk_hidden),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _as_batched(region: torch.Tensor) -> tuple[torch.Tensor, int]:
        if region.ndim == 4:  # (depth, H, W, C) -> (1, depth, H, W, C)
            region = region.unsqueeze(0)
        batch, depth, height, width, channels = region.shape
        return region.permute(0, 1, 4, 2, 3).reshape(batch, depth * channels, height, width), batch

    def forward(self, regions: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = set(self.config.region_names) - set(regions)
        if missing:
            raise KeyError(f"missing regions {sorted(missing)}; got {sorted(regions)}")
        pooled = []
        for name in self.config.region_names:
            x, _batch = self._as_batched(regions[name])
            features = self.body(x)
            pooled.append(torch.nn.functional.adaptive_avg_pool2d(features, (2, 2)).flatten(1))
        return self.trunk(torch.cat(pooled, dim=-1))


class VisionStateDecoder(nn.Module):
    """Privileged-state head: features -> self_feat (24) + danger map (128)."""

    def __init__(self, config: VisionNetConfig | None = None):
        super().__init__()
        self.config = config or VisionNetConfig()
        self.net = nn.Sequential(
            nn.Linear(self.config.trunk_hidden, self.config.state_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.state_hidden, obs.N_SELF + self.config.danger_cells),
        )
        self.n_self = obs.N_SELF
        self.danger_cells = self.config.danger_cells

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(features)
        return {
            "self_feat": out[:, : self.n_self],
            # Danger is a *count* of bullets per cell (0..~30 after the /8 scaling
            # used elsewhere); predicting log1p keeps the loss from being
            # dominated by the few crowded cells.
            "danger_log": out[:, self.n_self :],
        }


class VisionScorer(nn.Module):
    """Candidate-conditioned policy over pixels.

    ``score(frames)`` -> [B, 18]; ``act`` takes the argmax.  The candidate set is
    the same 18 (focus, direction) pairs the privileged teacher scores, so a
    teacher value array can be used as the regression target verbatim.
    """

    def __init__(self, config: VisionNetConfig | None = None):
        super().__init__()
        self.config = config or VisionNetConfig()
        self.encoder = VisionEncoder(self.config)
        self.candidate_embed = nn.Embedding(N_CANDIDATES, self.config.candidate_hidden)
        self.score_head = nn.Sequential(
            nn.Linear(self.config.trunk_hidden + self.config.candidate_hidden, self.config.candidate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.candidate_hidden, 1),
        )
        self.state_decoder = VisionStateDecoder(self.config) if self.config.state_head else None

    def features(self, regions: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(regions)

    def score(self, regions: dict[str, torch.Tensor], features: torch.Tensor | None = None) -> torch.Tensor:
        features = self.features(regions) if features is None else features
        index = torch.arange(N_CANDIDATES, device=features.device)
        embedded = self.candidate_embed(index)
        fused = torch.cat(
            [
                features.unsqueeze(1).expand(-1, N_CANDIDATES, -1),
                embedded.unsqueeze(0).expand(features.size(0), -1, -1),
            ],
            dim=-1,
        )
        return self.score_head(fused).squeeze(-1)

    def decode_state(self, regions: dict[str, torch.Tensor], features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.state_decoder is None:
            raise RuntimeError("this model was built with state_head=False")
        features = self.features(regions) if features is None else features
        return self.state_decoder(features)

    def act(self, regions: dict[str, torch.Tensor], deterministic: bool = True):
        """Greedy action tuple ``[move, shoot, focus, bomb]`` from pixels.

        Shooting is always on and bombing is left to a rule (the teacher's value
        only ranks movement/focus), which mirrors how the privileged scorer acts.
        """
        values = self.score(regions)
        best = values.argmax(dim=-1)
        move = best % spec.ACTION_DIMS[0]
        focus = best // spec.ACTION_DIMS[0]
        ones = torch.ones_like(move)
        zeros = torch.zeros_like(move)
        actions = torch.stack([move, ones, focus, zeros], dim=-1)
        zero = torch.zeros(values.size(0), dtype=values.dtype, device=values.device)
        return actions, zero, values.max(dim=-1).values, zero

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def save_vision_checkpoint(path, model: VisionScorer, extra: dict | None = None) -> None:
    torch.save(
        {
            "kind": "vision_scorer",
            "state_dict": model.state_dict(),
            "config": model.config.__dict__,
            "frame": {
                "H": vision.VISION_H,
                "W": vision.VISION_W,
                "C": vision.VISION_CHANNELS,
                "depth": model.config.depth,
            },
            "extra": extra or {},
        },
        path,
    )


def load_vision_checkpoint(path, map_location: str = "cpu") -> tuple[VisionScorer, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("kind") != "vision_scorer":
        raise ValueError("not a vision checkpoint")
    frame = payload.get("frame", {})
    if frame.get("H") != vision.VISION_H or frame.get("W") != vision.VISION_W:
        raise ValueError(
            f"checkpoint frame size {frame.get('H')}x{frame.get('W')} does not match "
            f"this build's {vision.VISION_H}x{vision.VISION_W}"
        )
    model = VisionScorer(VisionNetConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model, payload.get("extra", {})
