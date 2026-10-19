"""Vision model v2: three independent encoders, player position from a heatmap.

This is a sibling of :mod:`touhou_rl.vmodels`, not a replacement: v1 checkpoints
and its tests must keep working verbatim.  Three changes against v1, one reason
each:

* **The playfield keeps its spatial resolution.**  v1 pooled every region to 2x2
  before a global linear layer, so the playfield reached the head as four
  numbers.  A ~16 px player sprite and 2-10 px bullets do not survive that - the
  measured v1 error (70.97 px) was quadrant level.  The v2 playfield trunk stops
  at stride 8 (28x24 for a 224x192 input) and exposes that grid through a
  heatmap head.
* **Position is a distribution, not two scalars.**  The player position is read
  by a soft-argmax over the heatmap logits: differentiable, sub-cell precise, and
  every cell contributes gradient.  :func:`decode_position` is the single
  implementation shared by the training metrics and by deployment - a second
  copy would drift.
* **The HUD regions get their own encoders.**  HUD pixels are text and icons at
  fixed positions; the playfield is sparse point targets.  Sharing one kernel
  set asks it to be good at both, and v1 additionally had to pool 34x216 and
  126x216 onto the same 2x2.  ``hud_score`` and ``hud_status`` here are separate
  modules with separate parameters.

The input contract is unchanged: ``{name: float32[B, depth, H, W, 3]}`` built by
:func:`touhou_rl.vision.regions_to_batch`; ``depth`` only changes the conv input
channels (``3 * depth``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import obs, spec, vision
from .vmodels import N_CANDIDATES

# Per-branch vector widths after pooling.  These are part of the checkpoint
# contract (section 3.3 of the design doc), not knobs: 128 + 64 + 64 = 256.
PLAYFIELD_VEC = 128
HUD_VEC = 64
PLAYFIELD_STRIDE = 8  # 224x192 -> 28x24 feature grid, i.e. heatmap_hw
PLAYFIELD_POOL = (2, 2)
HUD_SCORE_POOL = (2, 8)
HUD_STATUS_POOL = (8, 4)
SELF_HEAD_DIM = obs.N_SELF - 2  # self_feat[0:2] is filled from the heatmap

#: R3 (設計 §5.3): the danger head is the only thing that varies.  ``legacy`` is
#: the v1.0.3/v2.0 default (``AdaptiveAvgPool2d(16,8)`` then a 1x1 conv) and is
#: what every existing v2 checkpoint was trained with; a payload that does not
#: record the field is a ``legacy`` checkpoint.  ``learnable-v104`` adds one
#: small learnable conv stack before the fixed pool (D3 / R3):
#: ``96 -> 16 Conv1/ReLU -> depthwise Conv3 pad1/ReLU -> pool 16x8 -> Conv1
#: 16 -> 1 -> flatten 128``, with **no** output ReLU.
DANGER_HEAD_LEGACY = "legacy"
DANGER_HEAD_LEARNABLE_V104 = "learnable-v104"
DANGER_HEAD_VARIANTS: tuple[str, ...] = (DANGER_HEAD_LEGACY, DANGER_HEAD_LEARNABLE_V104)
#: Channels of the learnable bottleneck (D3 uses 16).
DANGER_HEAD_BOTTLENECK = 16


def _v104_error(code: str, detail: str) -> ValueError:
    return ValueError(f"v104:{code}: {detail}")


@dataclass
class VisionNetConfigV2:
    depth: int = 1
    playfield_hw: tuple[int, int] = (224, 192)
    hud_score_hw: tuple[int, int] = (34, 216)
    hud_status_hw: tuple[int, int] = (126, 216)
    pf_width: int = 32
    hud_width: int = 16
    trunk_hidden: int = 256
    state_hidden: int = 256
    candidate_hidden: int = 64
    heatmap_gauss_sigma: float = 1.5  # target gaussian sigma, in heatmap cells
    danger_rows: int = obs.DANGER_ROWS
    danger_cols: int = obs.DANGER_COLS
    state_head: bool = True
    #: R3 opt-in.  Missing in old payloads, so ``legacy`` is the compatibility
    #: default; an unknown value is rejected instead of being coerced.
    danger_head_variant: str = DANGER_HEAD_LEGACY

    def __post_init__(self) -> None:
        if self.danger_head_variant not in DANGER_HEAD_VARIANTS:
            raise _v104_error(
                "CONFIG",
                f"danger_head_variant must be one of {list(DANGER_HEAD_VARIANTS)}, "
                f"got {self.danger_head_variant!r}",
            )

    @property
    def region_names(self) -> tuple[str, ...]:
        return ("playfield", "hud_score", "hud_status")

    @property
    def region_sizes(self) -> dict[str, tuple[int, int]]:
        return {
            "playfield": self.playfield_hw,
            "hud_score": self.hud_score_hw,
            "hud_status": self.hud_status_hw,
        }

    @property
    def heatmap_hw(self) -> tuple[int, int]:
        return (self.playfield_hw[0] // PLAYFIELD_STRIDE, self.playfield_hw[1] // PLAYFIELD_STRIDE)


def decode_position(heatmap_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B, H, W]`` heatmap logits -> ``(rx[B], ry[B])`` in ``[0, 1]``.

    Soft-argmax: the softmax of all ``H*W`` logits weights each cell centre, so
    the result is the expectation of the position under the predicted
    distribution.  That is differentiable and interpolates between cells (an
    argmax would quantise to 16 native px per cell); it also means the output
    lives in ``[0.5/W, 1 - 0.5/W]``.
    """
    batch, height, width = heatmap_logits.shape
    probabilities = torch.softmax(heatmap_logits.flatten(1), dim=-1)
    columns = torch.arange(width, dtype=heatmap_logits.dtype, device=heatmap_logits.device).repeat(height)
    rows = torch.arange(height, dtype=heatmap_logits.dtype, device=heatmap_logits.device).repeat_interleave(width)
    column = (probabilities * columns).sum(-1)
    row = (probabilities * rows).sum(-1)
    return (column + 0.5) / width, (row + 0.5) / height


def heatmap_target(
    rx: np.ndarray,
    ry: np.ndarray,
    grid_h: int,
    grid_w: int,
    sigma: float = 1.5,
) -> np.ndarray:
    """``[N]`` normalised positions -> ``[N, grid_h*grid_w]`` gaussian targets.

    Row-major, matching ``heatmap_logits.flatten(1)``.  Targets are normalised to
    a probability distribution so the training loss is a cross-entropy against
    the heatmap; regression to the raw gaussian would be dominated by the peak
    and mismatch the softmax used in :func:`decode_position`.
    """
    rx = np.asarray(rx, dtype=np.float64).reshape(-1)
    ry = np.asarray(ry, dtype=np.float64).reshape(-1)
    if rx.shape != ry.shape:
        raise ValueError(f"rx and ry must have the same length, got {rx.shape} and {ry.shape}")
    gx = np.clip(rx * grid_w - 0.5, -1.0, grid_w)
    gy = np.clip(ry * grid_h - 0.5, -1.0, grid_h)
    xs = np.arange(grid_w, dtype=np.float64)
    ys = np.arange(grid_h, dtype=np.float64)
    squared_x = (xs[None, :] - gx[:, None]) ** 2  # [N, grid_w]
    squared_y = (ys[None, :] - gy[:, None]) ** 2  # [N, grid_h]
    target = np.exp(-(squared_y[:, :, None] + squared_x[:, None, :]) / (2.0 * sigma * sigma))
    target = target.reshape(rx.size, grid_h * grid_w)
    target /= target.sum(axis=1, keepdims=True)
    return target.astype(np.float32)


def _conv_block(in_channels: int, out_channels: int, kernel: int, stride: int, padding: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
    ]


def _as_batched(region: torch.Tensor) -> torch.Tensor:
    """``[B or (), depth, H, W, 3]`` -> ``[B, depth*3, H, W]`` conv input."""
    if region.ndim == 4:  # (depth, H, W, C) -> (1, depth, H, W, C)
        region = region.unsqueeze(0)
    batch, depth, height, width, channels = region.shape
    return region.permute(0, 1, 4, 2, 3).reshape(batch, depth * channels, height, width)


class PlayfieldBranch(nn.Module):
    """Stride-8 playfield CNN exporting a heatmap, a danger grid and a vector.

    The heatmap and the danger grid are the branch's spatial outputs; the pooled
    vector only carries the summary that the fusion trunk needs.  The danger head
    pools 28x24 -> 16x8 with ``AdaptiveAvgPool2d``, whose cell boundaries can be
    off by at most 0.875 feature cells (~7 native px against a 28 px danger cell).
    That approximation is accepted on purpose - cropping or interpolating to
    "fix" it would misalign the target grid instead of matching it.

    R3 adds ``danger_head_variant``: ``legacy`` keeps the exact v1.0.3 module
    names (``danger_head``), so every old checkpoint still loads strictly;
    ``learnable-v104`` adds ``danger_reduce`` / ``danger_depthwise`` /
    ``danger_out`` and leaves ``danger_head`` unregistered, so a variant
    checkpoint can never be mistaken for a legacy one (the state dict keys
    differ and ``strict=True`` refuses it).
    """

    def __init__(self, config: VisionNetConfigV2):
        super().__init__()
        self.config = config
        width = config.pf_width
        channels = vision.VISION_CHANNELS * config.depth
        self.body = nn.Sequential(
            *_conv_block(channels, width, 5, stride=2, padding=2),
            *_conv_block(width, width * 2, 3, stride=2, padding=1),
            *_conv_block(width * 2, width * 3, 3, stride=2, padding=1),
            *_conv_block(width * 3, width * 3, 3, stride=1, padding=1),
        )
        self.heatmap_head = nn.Conv2d(width * 3, 1, 1)
        self.danger_pool = nn.AdaptiveAvgPool2d((config.danger_rows, config.danger_cols))
        self.danger_variant = str(config.danger_head_variant)
        if self.danger_variant == DANGER_HEAD_LEGACY:
            self.danger_head = nn.Conv2d(width * 3, 1, 1)
        else:
            # Same learning target grid, one learnable local aggregation first.
            self.danger_head = None
            self.danger_reduce = nn.Conv2d(width * 3, DANGER_HEAD_BOTTLENECK, 1)
            self.danger_depthwise = nn.Conv2d(
                DANGER_HEAD_BOTTLENECK,
                DANGER_HEAD_BOTTLENECK,
                3,
                stride=1,
                padding=1,
                groups=DANGER_HEAD_BOTTLENECK,
            )
            self.danger_out = nn.Conv2d(DANGER_HEAD_BOTTLENECK, 1, 1)
        self.vector = nn.Sequential(
            nn.AdaptiveAvgPool2d(PLAYFIELD_POOL),
            nn.Flatten(),
            nn.Linear(width * 3 * PLAYFIELD_POOL[0] * PLAYFIELD_POOL[1], PLAYFIELD_VEC),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.body(x)
        heatmap_logits = self.heatmap_head(features).squeeze(1)  # [B, 28, 24]
        if self.danger_variant == DANGER_HEAD_LEGACY:
            danger_log = self.danger_head(self.danger_pool(features)).flatten(1)  # [B, 128]
        else:
            local = F.relu(self.danger_reduce(features))
            local = F.relu(self.danger_depthwise(local))
            # No output ReLU: the target is log1p(count/8) >= 0 but the loss owns
            # the lower bound, and a clamp would remove the gradient at 0.
            danger_log = self.danger_out(self.danger_pool(local)).flatten(1)
        return self.vector(features), heatmap_logits, danger_log



class HudBranch(nn.Module):
    """HUD CNN: stride-2 convs, a fixed adaptive pool, then one linear layer.

    ``pool_hw`` is the only thing that differs between ``hud_score`` and
    ``hud_status``; the two instances never share parameters.
    """

    def __init__(self, config: VisionNetConfigV2, pool_hw: tuple[int, int]):
        super().__init__()
        width = config.hud_width
        channels = vision.VISION_CHANNELS * config.depth
        self.pool_hw = pool_hw
        self.body = nn.Sequential(
            *_conv_block(channels, width, 5, stride=2, padding=2),
            *_conv_block(width, width * 2, 3, stride=2, padding=1),
            *_conv_block(width * 2, width * 3, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(pool_hw),
            nn.Flatten(),
            nn.Linear(width * 3 * pool_hw[0] * pool_hw[1], HUD_VEC),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class VisionScorerV2(nn.Module):
    """Candidate-conditioned vision policy with a heatmap position decoder.

    Same public surface as :class:`touhou_rl.vmodels.VisionScorer`
    (``features``/``score``/``decode_state``/``act``/``num_parameters``) so that
    ``evaluate.py`` and the training loop treat both architectures alike, plus a
    ``heatmap_logits`` entry in ``decode_state`` for the heatmap loss.
    """

    def __init__(self, config: VisionNetConfigV2 | None = None):
        super().__init__()
        self.config = config or VisionNetConfigV2()
        self.playfield = PlayfieldBranch(self.config)
        self.hud_score = HudBranch(self.config, HUD_SCORE_POOL)
        self.hud_status = HudBranch(self.config, HUD_STATUS_POOL)
        fused = PLAYFIELD_VEC + 2 * HUD_VEC
        self.trunk = nn.Sequential(nn.Linear(fused, self.config.trunk_hidden), nn.ReLU(inplace=True))
        self.self_head = (
            nn.Sequential(
                nn.Linear(self.config.trunk_hidden, self.config.state_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.state_hidden, SELF_HEAD_DIM),
            )
            if self.config.state_head
            else None
        )
        self.candidate_embed = nn.Embedding(N_CANDIDATES, self.config.candidate_hidden)
        self.score_head = nn.Sequential(
            nn.Linear(self.config.trunk_hidden + self.config.candidate_hidden, self.config.candidate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.candidate_hidden, 1),
        )

    def _encode(self, regions: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        missing = set(self.config.region_names) - set(regions)
        if missing:
            raise KeyError(f"missing regions {sorted(missing)}; got {sorted(regions)}")
        pf_vec, heatmap_logits, danger_log = self.playfield(_as_batched(regions["playfield"]))
        score_vec = self.hud_score(_as_batched(regions["hud_score"]))
        status_vec = self.hud_status(_as_batched(regions["hud_status"]))
        trunk = self.trunk(torch.cat([pf_vec, score_vec, status_vec], dim=-1))
        return trunk, heatmap_logits, danger_log

    def features(self, regions: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._encode(regions)[0]

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

    def decode_state(
        self, regions: dict[str, torch.Tensor], features: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if self.self_head is None:
            raise RuntimeError("this model was built with state_head=False")
        trunk, heatmap_logits, danger_log = self._encode(regions)
        if features is not None:
            trunk = features
        rx, ry = decode_position(heatmap_logits)
        self_pred = self.self_head(trunk)
        # Position is filled from the heatmap, never regressed: a second
        # regression path for the same two numbers would fight the heatmap loss.
        self_feat = torch.cat([rx.unsqueeze(-1), ry.unsqueeze(-1), self_pred], dim=-1)
        return {
            "self_feat": self_feat,
            "danger_log": danger_log,
            "heatmap_logits": heatmap_logits,
        }

    def act(self, regions: dict[str, torch.Tensor], deterministic: bool = True):
        """Greedy action tuple ``[move, shoot, focus, bomb]`` from pixels.

        Same contract as v1: shooting stays on and bombing is left to a rule
        (the teacher only ranks movement/focus).
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
