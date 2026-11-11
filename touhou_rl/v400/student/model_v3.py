"""Student **v3**: the P3 target architecture (IMPALA-ResNet, spatial readout).

13-P3 §2.3 asks for an IMPALA-ResNet encoder (3 stages, channels 32/64/64, each
stage = conv + max-pool + 2 residual blocks), a 512-wide trunk, the 18 x 2
factorised policy head, an auxiliary value head and a perception head group.
The two things this file adds beyond a first reading of that section are the ones
the P3 restart (EXP-P3-009, §0.2 reason 2) says were the actual defect:

**1. The playfield stream keeps its space.**  Every earlier student
(``model.py:55``, ``model_v2.py:75``, ``rl/policy.py:153``) ended the play
encoder with ``AdaptiveAvgPool2d(1)``, which throws away *where* a bullet is.
Here the encoder returns the full ``(B, 64, 14, 12)`` feature map and the trunk
reads it through a **learned spatial readout**::

    play (B,4,112,96) -> 3 x [conv -> maxpool -> 2 residual blocks] -> (B,64,14,12)
                      -> conv1x1(64->16) -> ReLU -> flatten (2688) -> Linear -> 512

``AdaptiveAvgPool2d(1)`` appears nowhere in the module tree
(:func:`adaptive_pool_layers` is what the tests reflect over).

Why a flattening readout and not the other "keep the space" option (a spatial
softmax over the final map, i.e. unsupervised keypoints whose coordinates become
the features):

* **the dense labels need the map anyway.**  The perception head group predicts
  1/4-resolution occupancy / self / danger grids, so the spatial map has to
  survive to the heads regardless.  A linear readout keeps one source of truth
  and makes the policy read the *same* map the perception heads are trained on;
  a keypoint readout would put a second, different summary on the policy path.
* **multi-modal states.**  A spatial softmax collapses each channel to one
  expectation, so two bullets far apart and two bullets close together can
  produce the same coordinates.  Dodging needs to see several threats at once;
  a linear readout over ``(row, col)`` cells can represent them independently.
* **it is the stronger test of the hypothesis.**  The restart hypothesis is
  "position information is what was missing".  A readout whose every weight is
  tied to one (row, col, channel) either learns position or provably cannot;
  an unsupervised-keypoint coordinate is a learned *shift-invariant* summary and
  would blur the result (EXP-P3-011 is what measures the effect).
* **budget.**  ``2688 -> 512`` costs 1.38 M parameters, which lands the whole
  network in the pre-registered 2-6 M band (2,565,404 with the default
  arguments) with room to spare, whereas the keypoint readout would leave the
  model at ~0.4 M and force unrelated capacity in elsewhere to reach the band.

**2. The observation is obs-v3**, not obs-v1: the play plane is 96x112 and the
8x18 ``act_hist`` of obs-v2 is a third input.  Shapes are asserted, never
guessed, and a wrong input raises instead of silently broadcasting.

Layout::

    play     (B,4,112,96) -> IMPALA stages -> (B,64,14,12) -> spatial readout -> 512
    hud      (B,1,56,160) -> small convs   -> (B,32,7,20)  -> flatten -> 96
    act_hist (B,8,18)     -> flatten 144   -> Linear      -> 64
    trunk: concat(672) -> Linear(672,512) -> ReLU
    heads: move_logits[18], bomb_logits[2], value[1]
    perception heads (training only): occupancy (B,1,112,96), self_heatmap
    (B,1,112,96), danger (B,3,112,96) for horizons 4/8/16

The perception outputs are exactly the label grid of ``touhou_rl/v400/labels``
(``GRID_HEIGHT x GRID_WIDTH = 112 x 96``, i.e. 1/4 of the 448x384 playfield),
which coincides with obs-v3's play plane because both are the same 4x4 integer
block mean.  A head whose labels are absent is **skipped, not zero-filled**: see
:func:`perception_aux_loss`, which returns ``None`` and a weight of 0.0 per
missing component, and :func:`student_v3_loss`, which records the *effective*
weight of every term in the returned breakdown.

Loss (13-P3 §2.4 default weights)::

    L = 1.0 * KL(pi_teacher || pi_student)          scores / tau
      + 1.0 * BCE(bomb; pos_weight)
      + 0.5 * perception aux (BCE over the labelled aux components)
      + 0.25 * Huber(value, target)

Checkpoint: schema ``v400-student-checkpoint-3``; it records ``arch="v3"``, the
obs-v3 version **and digest**, the full model config and the parameter count, and
:func:`load_checkpoint_v3` fails closed on any other spec or arch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from touhou_rl.v400.labels import spec as label_spec
from touhou_rl.v400.student import obs_spec
from touhou_rl.v400.student import obs_spec_v3

CHECKPOINT_SCHEMA = "v400-student-checkpoint-3"
#: ``arch`` string every v3 checkpoint carries (the loaders dispatch on it).
ARCH_V3 = "v3"

#: 13-P3 §2.3 / T4 card: the canonical architecture must sit in 2-6 M parameters.
PARAMETER_BUDGET: tuple[int, int] = (2_000_000, 6_000_000)

#: IMPALA stage widths (13-P3 §2.3).  Three stages, so the plane is /8.
STAGE_CHANNELS: tuple[int, ...] = (32, 64, 64)
RESIDUAL_BLOCKS = 2
#: Trunk width (13-P3 §2.3).
TRUNK = 512
#: 1x1 channel bottleneck of the spatial readout, before the flatten.
READOUT_CHANNELS = 16
HUD_HIDDEN = 96
HIST_HIDDEN = 64
#: Perception decoder widths, one per upsample step.
AUX_CHANNELS: tuple[int, int, int] = (32, 16, 16)
#: Future-danger horizons; identical to ``labels.spec.HORIZON_FRAMES``.
AUX_HORIZONS: tuple[int, ...] = (4, 8, 16)
#: Aux output grid (rows, cols) = the label grid = 1/4 of the 448x384 playfield.
AUX_SHAPE: tuple[int, int] = (label_spec.GRID_HEIGHT, label_spec.GRID_WIDTH)
#: The perception head group's members, in the order they are reported.
AUX_COMPONENTS: tuple[str, ...] = ("occupancy", "self_heatmap", "danger")

#: 13-P3 §2.4 default loss weights.  ``kl`` and ``bomb`` are normalised in
#: :func:`student_v3_loss`; ``aux`` and ``value`` are the lambdas of §2.4.
DEFAULT_LOSS_WEIGHTS: dict[str, float] = {"kl": 1.0, "bomb": 1.0, "aux": 0.5,
                                          "value": 0.25}


class ModelV3Error(ValueError):
    """An input, config or checkpoint does not match the v3 contract."""


class ParameterBudgetError(ModelV3Error):
    """The canonical architecture left the pre-registered 2-6 M parameter band."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _expected_play_shape() -> tuple[int, int, int]:
    return (obs_spec_v3.FRAME_STACK, *obs_spec_v3.PLAY_SHAPE)


def adaptive_pool_layers(module: nn.Module) -> list[nn.Module]:
    """Every ``AdaptiveAvgPool2d``/``AdaptiveMaxPool2d`` in ``module``'s tree.

    The tests (and the T4 acceptance) need a *reflective* check, not a grep of
    the source: a submodule added later must be caught too.  Returning the
    modules lets a caller print which one it was.
    """

    found: list[nn.Module] = []
    for child in module.modules():
        if isinstance(child, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
                              nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool1d)):
            found.append(child)
    return found


def n_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def check_parameter_budget(module: nn.Module, *,
                           budget: tuple[int, int] = PARAMETER_BUDGET) -> int:
    """Assert ``module`` sits inside the pre-registered parameter band."""

    count = n_parameters(module)
    low, high = int(budget[0]), int(budget[1])
    if not low <= count <= high:
        raise ParameterBudgetError(
            f"{type(module).__name__} has {count:,} parameters, outside the "
            f"pre-registered {low:,}-{high:,} band (13-P3 §2.3)")
    return count


# ---------------------------------------------------------------------------
# playfield stream: IMPALA-ResNet, space preserved
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """IMPALA residual block: ``relu(bn(conv(relu(bn(conv(x)))))) + x``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)


class ImpalaStage(nn.Module):
    """One IMPALA stage: conv -> max-pool -> ``RESIDUAL_BLOCKS`` residual blocks."""

    def __init__(self, in_channels: int, out_channels: int, *,
                 blocks: int = RESIDUAL_BLOCKS) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.blocks_per_stage = int(blocks)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.blocks = nn.Sequential(*[ResidualBlock(out_channels)
                                      for _ in range(int(blocks))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)), inplace=True)
        return self.blocks(self.pool(out))


class SpatialPlayEncoder(nn.Module):
    """obs-v3 play stack -> ``(B, C, H, W)`` feature map.  No global pooling."""

    def __init__(self, in_channels: int = obs_spec_v3.FRAME_STACK,
                 stage_channels: Sequence[int] = STAGE_CHANNELS,
                 blocks: int = RESIDUAL_BLOCKS) -> None:
        super().__init__()
        channels = int(in_channels)
        stages: list[nn.Module] = []
        for out_channels in stage_channels:
            stages.append(ImpalaStage(channels, int(out_channels), blocks=blocks))
            channels = int(out_channels)
        if not stages:  # pragma: no cover - the canonical config has three stages
            raise ModelV3Error("the play encoder needs at least one stage")
        self.in_channels = int(in_channels)
        self.stages = nn.ModuleList(stages)
        #: Stage widths as constructed (read back by ``config()``).
        self.stage_channels: tuple[int, ...] = tuple(int(value) for value in stage_channels)
        self.out_channels = channels
        downsample = 2 ** len(stages)
        height, width = obs_spec_v3.PLAY_SHAPE
        if height % downsample or width % downsample:
            raise ModelV3Error(
                f"{height}x{width} does not divide by {downsample} (2^{len(stages)} stages)")
        #: Spatial extent of the output map, derived -- never hard-coded.
        self.spatial_shape: tuple[int, int] = (height // downsample,
                                               width // downsample)

    @property
    def features(self) -> int:
        return self.out_channels * self.spatial_shape[0] * self.spatial_shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (self.in_channels, *obs_spec_v3.PLAY_SHAPE)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ModelV3Error(
                f"play must be (B,{expected[0]},{expected[1]},{expected[2]}), "
                f"got {tuple(x.shape)}")
        for stage in self.stages:
            x = stage(x)
        if tuple(x.shape[2:]) != self.spatial_shape:  # pragma: no cover - derived
            raise ModelV3Error(
                f"play encoder produced {tuple(x.shape[2:])}, "
                f"expected {self.spatial_shape}")
        return x


class SpatialReadout(nn.Module):
    """``(B,C,h,w)`` map -> ``(B,trunk)`` by flattening it, not by averaging it.

    The 1x1 convolution is a learned channel bottleneck (so the flatten stays
    affordable); the ``Linear`` weight matrix has one row per ``(row, col,
    channel)`` cell, which is exactly the position-coding a global average pool
    cannot express.
    """

    def __init__(self, in_channels: int, spatial_shape: tuple[int, int], *,
                 readout_channels: int = READOUT_CHANNELS, out_features: int = TRUNK,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.spatial_shape = (int(spatial_shape[0]), int(spatial_shape[1]))
        self.readout_channels = int(readout_channels)
        self.project = nn.Sequential(
            nn.Conv2d(self.in_channels, self.readout_channels, 1),
            nn.BatchNorm2d(self.readout_channels),
            nn.ReLU(inplace=True),
        )
        self.flat_features = (self.readout_channels * self.spatial_shape[0]
                              * self.spatial_shape[1])
        self.dropout = float(dropout)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, int(out_features)),
            nn.ReLU(inplace=True),
            *([nn.Dropout(self.dropout)] if self.dropout > 0 else []),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.in_channels:
            raise ModelV3Error(
                f"spatial features must be (B,{self.in_channels},H,W), "
                f"got {tuple(features.shape)}")
        return self.linear(self.project(features))


# ---------------------------------------------------------------------------
# HUD and action-history streams
# ---------------------------------------------------------------------------
class HudEncoder(nn.Module):
    """Independent small conv stream over the 56x160 HUD strip.

    Like the play stream it **flattens** its ``(32, 7, 20)`` map instead of
    averaging it: the HUD's meaning is tied to where a digit is (lives vs bombs
    vs power are three fixed bands), so a global pool would mix them.
    """

    def __init__(self, in_channels: int = obs_spec.HUD_FRAMES, width: int = 16,
                 hidden: int = HUD_HIDDEN) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, width, 5, stride=2, padding=2),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
        )
        self.width = int(width)
        self.hidden = int(hidden)
        height, width_px = obs_spec.HUD_SHAPE
        if height % 8 or width_px % 8:  # pragma: no cover - shapes are frozen
            raise ModelV3Error(f"HUD {height}x{width_px} does not divide by 8")
        self.spatial_shape = (height // 8, width_px // 8)
        self.out_channels = width * 2
        self.flat_features = self.out_channels * self.spatial_shape[0] * self.spatial_shape[1]
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_features, int(hidden)),
            nn.ReLU(inplace=True),
        )

    def forward(self, hud: torch.Tensor) -> torch.Tensor:
        if hud.ndim == 3:
            hud = hud.unsqueeze(1)
        expected = (self.in_channels, *obs_spec.HUD_SHAPE)
        if hud.ndim != 4 or tuple(hud.shape[1:]) != expected:
            raise ModelV3Error(
                f"hud must be (B,{expected[0]},{expected[1]},{expected[2]}) or "
                f"(B,{expected[1]},{expected[2]}), got {tuple(hud.shape)}")
        return self.project(self.features(hud))


class ActionHistoryEncoder(nn.Module):
    """``act_hist`` ``(B,8,18)`` (or flattened ``(B,144)``) -> ``(B,hist_hidden)``."""

    def __init__(self, decisions: int = obs_spec_v3.ACTION_HISTORY_DECISIONS,
                 classes: int = obs_spec.N_MOVE_CLASSES, hidden: int = HIST_HIDDEN) -> None:
        super().__init__()
        self.decisions = int(decisions)
        self.classes = int(classes)
        self.hidden = int(hidden)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.decisions * self.classes, self.hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, act_hist: torch.Tensor) -> torch.Tensor:
        if act_hist.ndim == 2 and act_hist.shape[1] == self.decisions * self.classes:
            act_hist = act_hist.reshape(-1, self.decisions, self.classes)
        if (act_hist.ndim != 3
                or tuple(act_hist.shape[1:]) != (self.decisions, self.classes)):
            raise ModelV3Error(
                f"act_hist must be (B,{self.decisions},{self.classes}) or "
                f"(B,{self.decisions * self.classes}), got {tuple(act_hist.shape)}")
        return self.encoder(act_hist.float())


# ---------------------------------------------------------------------------
# perception head group (training-time auxiliary)
# ---------------------------------------------------------------------------
class UpsampleBlock(nn.Module):
    """Nearest-neighbour upsample to a fixed shape + 3x3 conv + BN + ReLU."""

    def __init__(self, in_channels: int, out_channels: int,
                 target_shape: tuple[int, int]) -> None:
        super().__init__()
        self.target_shape = (int(target_shape[0]), int(target_shape[1]))
        self.out_channels = int(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.target_shape, mode="nearest")
        return F.relu(self.bn(self.conv(x)), inplace=True)


class PerceptionHeads(nn.Module):
    """Dense 1/4-resolution heads: occupancy, self heatmap, future danger.

    The decoder walks the play feature map ``(B,C,14,12)`` up to the label grid
    ``(B,*,112,96)`` with three nearest-neighbour upsample+conv steps, then three
    1x1 heads read it.  Every head is a *logit* map: the loss applies
    ``binary_cross_entropy_with_logits`` and deployment never needs them.
    """

    def __init__(self, in_channels: int, spatial_shape: tuple[int, int],
                 out_shape: tuple[int, int] = AUX_SHAPE,
                 aux_channels: Sequence[int] = AUX_CHANNELS,
                 horizons: Sequence[int] = AUX_HORIZONS) -> None:
        super().__init__()
        if len(aux_channels) != 3:  # pragma: no cover - canonical config
            raise ModelV3Error(f"aux_channels must have 3 steps, got {tuple(aux_channels)}")
        self.out_shape = (int(out_shape[0]), int(out_shape[1]))
        self.horizons = tuple(int(value) for value in horizons)
        height, width = (int(spatial_shape[0]), int(spatial_shape[1]))
        targets: list[tuple[int, int]] = []
        for _step in aux_channels:
            height, width = height * 2, width * 2
            targets.append((height, width))
        if targets[-1] != self.out_shape:
            raise ModelV3Error(
                f"three upsample steps from {tuple(spatial_shape)} reach {targets[-1]}, "
                f"not the label grid {self.out_shape}")
        self.targets = tuple(targets)
        #: Decoder widths as constructed (read back by ``config()``).
        self.aux_channels: tuple[int, ...] = tuple(int(value) for value in aux_channels)
        ups: list[nn.Module] = []
        channels = int(in_channels)
        for out_channels, target in zip(aux_channels, targets):
            ups.append(UpsampleBlock(channels, int(out_channels), target))
            channels = int(out_channels)
        self.ups = nn.ModuleList(ups)
        self.occupancy = nn.Conv2d(channels, 1, 1)
        self.self_heatmap = nn.Conv2d(channels, 1, 1)
        self.danger = nn.Conv2d(channels, len(self.horizons), 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        x = features
        for block in self.ups:
            x = block(x)
        return {
            "occupancy": self.occupancy(x),
            "self_heatmap": self.self_heatmap(x),
            "danger": self.danger(x),
        }


# ---------------------------------------------------------------------------
# the student
# ---------------------------------------------------------------------------
@dataclass
class StudentV3Output:
    """One forward pass: policy logits (+ value), and the aux maps when asked."""

    move: torch.Tensor
    bomb: torch.Tensor
    value: torch.Tensor
    aux: dict[str, torch.Tensor] | None = None


class StudentV3(nn.Module):
    """obs-v3 student: IMPALA-ResNet play + HUD conv + action history -> 512 trunk."""

    arch = ARCH_V3

    def __init__(
        self,
        *,
        stage_channels: Sequence[int] = STAGE_CHANNELS,
        trunk: int = TRUNK,
        readout_channels: int = READOUT_CHANNELS,
        hud_width: int = 16,
        hud_hidden: int = HUD_HIDDEN,
        hist_hidden: int = HIST_HIDDEN,
        aux_channels: Sequence[int] = AUX_CHANNELS,
        aux_horizons: Sequence[int] = AUX_HORIZONS,
        residual_blocks: int = RESIDUAL_BLOCKS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.trunk_width = int(trunk)
        self.dropout = float(dropout)
        self.play = SpatialPlayEncoder(stage_channels=stage_channels,
                                      blocks=residual_blocks)
        self.hud = HudEncoder(width=hud_width, hidden=hud_hidden)
        self.history = ActionHistoryEncoder(hidden=hist_hidden)
        self.readout = SpatialReadout(self.play.out_channels, self.play.spatial_shape,
                                      readout_channels=readout_channels,
                                      out_features=self.trunk_width,
                                      dropout=self.dropout)
        self.perception = PerceptionHeads(self.play.out_channels,
                                          self.play.spatial_shape,
                                          aux_channels=aux_channels,
                                          horizons=aux_horizons)
        self.trunk_input_width = (self.readout.linear[1].out_features
                                  + self.hud.hidden + self.history.hidden)
        self.trunk = nn.Sequential(nn.Linear(self.trunk_input_width, self.trunk_width),
                                   nn.ReLU(inplace=True))
        self.move_head = nn.Linear(self.trunk_width, obs_spec.N_MOVE_CLASSES)
        self.bomb_head = nn.Linear(self.trunk_width, obs_spec.N_BOMB_CLASSES)
        self.value_head = nn.Linear(self.trunk_width, 1)
        if self.is_canonical():
            # T4 acceptance: "parameter count 2-6 M, asserted".
            check_parameter_budget(self)

    # -- introspection ------------------------------------------------------
    def is_canonical(self) -> bool:
        """True when this is the pre-registered architecture, not a variant.

        The 2-6 M assertion applies to the canonical config; the tests build
        tiny variants (for speed) and the trainer builds a canonical one, so the
        distinction is derived from the arguments instead of a flag.
        """

        return (tuple(self.play.stage_channels) == tuple(STAGE_CHANNELS)
                and self.trunk_width == TRUNK
                and self.readout.readout_channels == READOUT_CHANNELS
                and self.hud.hidden == HUD_HIDDEN
                and self.history.hidden == HIST_HIDDEN
                and tuple(self.perception.aux_channels) == tuple(AUX_CHANNELS)
                and self.play.stages[0].blocks_per_stage == RESIDUAL_BLOCKS)

    def n_parameters(self) -> int:
        return n_parameters(self)

    def config(self) -> dict[str, Any]:
        return {
            "arch": self.arch,
            "stage_channels": list(self.play.stage_channels),
            "residual_blocks": self.play.stages[0].blocks_per_stage,
            "trunk": self.trunk_width,
            "readout_channels": self.readout.readout_channels,
            "hud_width": self.hud.width,
            "hud_hidden": self.hud.hidden,
            "hist_hidden": self.history.hidden,
            "aux_channels": list(self.perception.aux_channels),
            "aux_horizons": list(self.perception.horizons),
            "dropout": self.dropout,
            "play_spatial_shape": list(self.play.spatial_shape),
            "play_features": self.play.features,
            "readout_flat_features": self.readout.flat_features,
            "aux_shape": list(self.perception.out_shape),
            "trunk_input_features": self.trunk_input_width,
        }

    # -- forward ------------------------------------------------------------
    def forward(self, play: torch.Tensor, hud: torch.Tensor,
                act_hist: torch.Tensor, *, aux: bool = False) -> StudentV3Output:
        """Policy logits; with ``aux=True`` also the perception maps (training)."""

        features = self.play(play)
        trunk_input = torch.cat([
            self.readout(features),
            self.hud(hud),
            self.history(act_hist),
        ], dim=1)
        hidden = self.trunk(trunk_input)
        return StudentV3Output(
            move=self.move_head(hidden),
            bomb=self.bomb_head(hidden),
            value=self.value_head(hidden).squeeze(-1),
            aux=self.perception(features) if aux else None,
        )

    @torch.no_grad()
    def act(self, play: torch.Tensor, hud: torch.Tensor,
            act_hist: torch.Tensor) -> tuple[int, int]:
        """Greedy ``(cand, bomb)`` for a single observation (batch of 1)."""

        self.eval()
        out = self(play, hud, act_hist)
        return int(out.move.argmax(dim=1)[0]), int(out.bomb.argmax(dim=1)[0])


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def teacher_soft_labels(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """``softmax(scores / tau)`` (13-P3 §2.4).  ``tau`` must be positive."""

    if not isinstance(tau, (int, float)) or not float(tau) > 0:
        raise ModelV3Error(f"tau must be positive, got {tau!r}")
    if scores.ndim != 2 or scores.shape[1] != obs_spec.N_MOVE_CLASSES:
        raise ModelV3Error(
            f"scores must be (B,{obs_spec.N_MOVE_CLASSES}), got {tuple(scores.shape)}")
    return torch.softmax(scores / float(tau), dim=-1)


def _aux_target(name: str, target: torch.Tensor, out_shape: tuple[int, int],
                channels: int, batch: int) -> torch.Tensor:
    """Broadcast one aux label map to ``(B,C,H,W)``; fail closed on a mismatch."""

    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(1)
    if target.ndim != 4:
        raise ModelV3Error(f"aux target {name!r} must be (H,W), (B,H,W) or (B,C,H,W); "
                           f"got {tuple(target.shape)}")
    if int(target.shape[0]) != int(batch):
        raise ModelV3Error(f"aux target {name!r} has batch {target.shape[0]} != {batch}")
    if int(target.shape[1]) != int(channels):
        raise ModelV3Error(
            f"aux target {name!r} has {target.shape[1]} channel(s), expected {channels}")
    if tuple(target.shape[2:]) != tuple(out_shape):
        raise ModelV3Error(
            f"aux target {name!r} is {tuple(target.shape[2:])}, expected {tuple(out_shape)}")
    return target.float()


def perception_aux_loss(
    outputs: Mapping[str, torch.Tensor] | None,
    targets: Mapping[str, torch.Tensor] | None,
    *,
    pos_weight: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """BCE over the aux components that **have** labels; skip the rest.

    Returns ``(loss, report)``.  ``loss`` is ``None`` when no component has a
    label -- the caller must then add nothing (the T4 card's "skip gracefully,
    weight zero"), and ``report["effective_weights"]`` says so explicitly so a
    checkpoint can never look as if it was trained with an aux term it did not
    have.  ``report["active"]`` lists the components that did contribute.
    """

    weights = {name: 0.0 for name in AUX_COMPONENTS}
    report: dict[str, Any] = {
        "active": [], "effective_weights": weights, "components": {},
        "skipped": {}, "loss": None, "target_shapes": {},
    }
    if outputs is None:
        report["skipped"] = {name: "no perception output (model built without aux)"
                             for name in AUX_COMPONENTS}
        return None, report
    if not targets:
        report["skipped"] = {name: "no label supplied for this head"
                             for name in AUX_COMPONENTS}
        return None, report
    unknown = sorted(set(targets) - set(AUX_COMPONENTS))
    if unknown:
        raise ModelV3Error(f"unknown aux label(s) {unknown}; expected a subset of "
                           f"{list(AUX_COMPONENTS)}")
    reference = next(iter(outputs.values()))
    out_shape = tuple(int(value) for value in reference.shape[2:])
    batch = int(reference.shape[0])
    total: torch.Tensor | None = None
    for name in AUX_COMPONENTS:
        if name not in targets:
            report["skipped"][name] = "no label supplied for this head"
            continue
        if name not in outputs:
            report["skipped"][name] = "model has no such head"
            continue
        logits = outputs[name]
        target = _aux_target(name, targets[name], out_shape,
                            int(logits.shape[1]), batch)
        weight = torch.tensor(
            [float((pos_weight or {}).get(name, 1.0))], device=logits.device)
        component = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
        if not torch.isfinite(component):
            raise ModelV3Error(f"aux loss {name!r} is not finite")
        report["components"][name] = float(component.detach())
        report["target_shapes"][name] = list(target.shape)
        weights[name] = 1.0
        report["active"].append(name)
        total = component if total is None else total + component
    if total is None:
        report["skipped"].setdefault("all", "every head was missing its label")
        return None, report
    total = total / float(len(report["active"]))
    report["loss"] = float(total.detach())
    return total, report


def student_v3_loss(
    outputs: StudentV3Output,
    batch: Mapping[str, torch.Tensor],
    *,
    tau: float = 2.0,
    bomb_pos_weight: float = 1.0,
    weights: Mapping[str, float] = DEFAULT_LOSS_WEIGHTS,
    aux_pos_weight: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """13-P3 §2.4 loss with per-term *effective* weights recorded.

    ``batch`` must hold ``scores`` ``(B,18)`` and ``bomb`` ``(B,)``; ``aux`` (a
    mapping of label maps) and ``value`` (``(B,)``) are optional.  A term whose
    target is absent gets an effective weight of ``0.0`` and is not computed, so
    "the aux labels are not wired yet" cannot silently become "the aux heads are
    trained against zeros".
    """

    for key in ("scores", "bomb"):
        if key not in batch:
            raise ModelV3Error(f"batch is missing {key!r}")
    effective = {name: 0.0 for name in ("kl", "bomb", "aux", "value")}
    breakdown: dict[str, Any] = {"effective_weights": effective, "terms": {},
                                 "skipped": {}, "total": 0.0}

    teacher = teacher_soft_labels(batch["scores"], tau)
    kl = F.kl_div(F.log_softmax(outputs.move, dim=-1), teacher, reduction="batchmean")
    breakdown["terms"]["kl"] = float(kl.detach())
    effective["kl"] = float(weights.get("kl", 1.0))
    total = float(weights.get("kl", 1.0)) * kl

    bomb = batch["bomb"].long()
    if bomb.ndim != 1 or bomb.shape[0] != outputs.bomb.shape[0]:
        raise ModelV3Error(f"bomb target must be (B,), got {tuple(bomb.shape)}")
    targets = F.one_hot(bomb, num_classes=obs_spec.N_BOMB_CLASSES).float()
    pos = torch.tensor([1.0, float(bomb_pos_weight)], device=outputs.bomb.device)
    bce = F.binary_cross_entropy_with_logits(outputs.bomb, targets, pos_weight=pos)
    breakdown["terms"]["bomb"] = float(bce.detach())
    effective["bomb"] = float(weights.get("bomb", 1.0))
    total = total + float(weights.get("bomb", 1.0)) * bce

    aux_loss, aux_report = perception_aux_loss(outputs.aux, batch.get("aux"),
                                               pos_weight=aux_pos_weight)
    breakdown["aux"] = aux_report
    if aux_loss is None:
        breakdown["skipped"]["aux"] = "; ".join(
            f"{name}: {reason}" for name, reason in aux_report["skipped"].items())
    else:
        effective["aux"] = float(weights.get("aux", 0.5))
        total = total + effective["aux"] * aux_loss

    value_target = batch.get("value")
    if value_target is None:
        breakdown["skipped"]["value"] = "no value target in the batch"
    else:
        if value_target.ndim > 1:
            value_target = value_target.reshape(-1)
        if int(value_target.shape[0]) != int(outputs.value.shape[0]):
            raise ModelV3Error(
                f"value target has {value_target.shape[0]} rows, "
                f"the head produced {outputs.value.shape[0]}")
        huber = F.huber_loss(outputs.value, value_target.float(), delta=1.0)
        breakdown["terms"]["value"] = float(huber.detach())
        effective["value"] = float(weights.get("value", 0.25))
        total = total + effective["value"] * huber

    breakdown["total"] = float(total.detach())
    return total, breakdown


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def build_model_v3(config: Mapping[str, Any]) -> StudentV3:
    """Instantiate from a checkpoint's ``model`` block (fail closed on arch)."""

    arch = str(config.get("arch", ""))
    if arch != ARCH_V3:
        raise ModelV3Error(f"model config arch {arch!r} is not {ARCH_V3!r}")
    return StudentV3(
        stage_channels=tuple(config.get("stage_channels", STAGE_CHANNELS)),
        trunk=int(config.get("trunk", TRUNK)),
        readout_channels=int(config.get("readout_channels", READOUT_CHANNELS)),
        hud_width=int(config.get("hud_width", 16)),
        hud_hidden=int(config.get("hud_hidden", HUD_HIDDEN)),
        hist_hidden=int(config.get("hist_hidden", HIST_HIDDEN)),
        aux_channels=tuple(config.get("aux_channels", AUX_CHANNELS)),
        aux_horizons=tuple(config.get("aux_horizons", AUX_HORIZONS)),
        residual_blocks=int(config.get("residual_blocks", RESIDUAL_BLOCKS)),
        dropout=float(config.get("dropout", 0.0)),
    )


def save_checkpoint_v3(
    path: str | Path,
    model: StudentV3,
    *,
    train: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write a v3 checkpoint (records ``arch``, obs-v3 version **and digest**)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "arch": ARCH_V3,
        "obs_spec_version": obs_spec_v3.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec_v3.obs_spec_v3_digest(),
        "model": {"class": type(model).__name__, **model.config()},
        "parameters": model.n_parameters(),
        "parameter_budget": list(PARAMETER_BUDGET),
        "canonical_architecture": bool(model.is_canonical()),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "train": dict(train or {}),
        "provenance": dict(provenance or {}),
    }
    torch.save(payload, path)
    return path


def load_checkpoint_v3(path: str | Path, *,
                       device: str = "cpu") -> tuple[StudentV3, dict[str, Any]]:
    """Load a v3 checkpoint; refuses another schema, arch or obs spec."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ModelV3Error(
            f"{path}: checkpoint schema {payload.get('schema') if isinstance(payload, dict) else None!r} "
            f"is not {CHECKPOINT_SCHEMA!r}")
    if payload.get("arch") != ARCH_V3:
        raise ModelV3Error(
            f"{path}: arch {payload.get('arch')!r} is not {ARCH_V3!r}")
    obs_spec_v3.check_obs_spec_v3(payload.get("obs_spec_version"),
                                  payload.get("obs_spec_digest"))
    config = payload.get("model") or {}
    model = build_model_v3(config)
    recorded = payload.get("parameters")
    if recorded is not None and int(recorded) != model.n_parameters():
        raise ModelV3Error(
            f"{path}: checkpoint records {recorded} parameters, the config builds "
            f"{model.n_parameters()}")
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def checkpoint_digest(path: str | Path) -> str:
    """sha256 of the checkpoint file bytes (evaluation provenance)."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe_checkpoint_v3(payload: Mapping[str, Any]) -> str:
    model = payload.get("model") or {}
    return (f"{payload.get('schema')} arch={payload.get('arch')} "
            f"obs={payload.get('obs_spec_version')} "
            f"params={payload.get('parameters')} "
            f"trunk={model.get('trunk')} stages={model.get('stage_channels')}")


__all__ = [
    "ARCH_V3",
    "AUX_CHANNELS",
    "AUX_COMPONENTS",
    "AUX_HORIZONS",
    "AUX_SHAPE",
    "CHECKPOINT_SCHEMA",
    "DEFAULT_LOSS_WEIGHTS",
    "HIST_HIDDEN",
    "HUD_HIDDEN",
    "ModelV3Error",
    "PARAMETER_BUDGET",
    "ParameterBudgetError",
    "PerceptionHeads",
    "READOUT_CHANNELS",
    "RESIDUAL_BLOCKS",
    "STAGE_CHANNELS",
    "StudentV3",
    "StudentV3Output",
    "TRUNK",
    "adaptive_pool_layers",
    "build_model_v3",
    "check_parameter_budget",
    "checkpoint_digest",
    "describe_checkpoint_v3",
    "load_checkpoint_v3",
    "n_parameters",
    "perception_aux_loss",
    "save_checkpoint_v3",
    "student_v3_loss",
    "teacher_soft_labels",
]
