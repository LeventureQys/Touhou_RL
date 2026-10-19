"""Vision model v3 (``v3-st``): spatial-temporal playfield, local ROI, single heatmap.

Sibling of :mod:`touhou_rl.vmodels2`, not a replacement: v1/v2 checkpoints, their
tests and the legacy trainer paths keep working verbatim.  The four changes
against v2, one reason each (all from the v1.0.4 design doc C5):

* **The playfield keeps two resolutions, not one.**  v2 stopped at stride 8 and
  read the player position from that 28x24 grid, which is 8 native px per cell.
  v3 keeps a stride-4 map (``F4``, 56x48) for the local window and a stride-8 map
  (``F8``, 28x24) for the danger grid, and fuses both into one heatmap, so the
  heatmap can be sub-cell precise *and* still see the whole field.
* **A local native-resolution window around the predicted player.**  Bullets are
  2-10 native px; downsampling the whole 448x384 field to 56x48 destroys them.
  The local branch resamples a 128x128 *native* window around the predicted
  (detached) centre into a 16x16 grid of 8-px bins - one ``grid_sample`` on
  ``F4``, no second full-resolution CNN.
* **The HUD only contributes its newest frame.**  The playfield gets the whole
  ``depth`` stack (motion), the HUD encoders get the latest frame only; HUD digits
  are state, not velocity.
* **One spatial head.**  ``heatmap_logits`` (56x48) is the single localisation
  distribution; entropy and peak probability are handed to the trunk as
  ``location_stats``, so "how sure am I" is visible to the policy without a
  second regression path for the same two coordinates.

The input contract is unchanged from v1/v2: ``{name: float32[B, depth, H, W, 3]}``
built by :func:`touhou_rl.vision.regions_to_batch`; ``depth`` only changes the
playfield conv input channels (``3 * depth``).  ``depth`` is 3 in this round: C5
allows exactly one depth for v3, and every other value is a config error rather
than a silent fallback.

Nothing here is trained by this module.  It exposes the C5 forward/legacy surface
(``forward``/``features``/``score``/``decode_state``/``act``/``num_parameters``)
plus three *layer accessors* (``locate``/``local_features``/``global_features``)
so that the geometry below can be pinned by unit tests and instrumented by the
S3-7 ablations without re-deriving the shapes.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import obs, spec, vision
from .vision_history import HISTORY_CONTRACTS, v104_error
from .vmodels import N_CANDIDATES
from .vmodels2 import decode_position

#: Checkpoint ``arch`` string (C5/C6).  ``VisionScorer``/``VisionScorerV2`` keep
#: their own labels; :func:`touhou_rl.vmodels.vision_arch` reads the model type.
ARCH = "v3-st"

#: Playfield downsampling per stage.  ``F4`` is ``playfield_hw / 4`` (the heatmap
#: grid, 56x48 for a 224x192 input), ``F8`` is ``playfield_hw / 8`` (28x24).
PLAYFIELD_STRIDE = 8

#: Native (un-downsampled) playfield size in pixels, (H, W) - C1: 384x448.
#: The local grid formula ``2*rx - 1 + 2*dx/384`` / ``... + 2*dy/448`` is written
#: in *native* pixels against these two numbers.
NATIVE_PLAYFIELD_HW: tuple[int, int] = (448, 384)

#: Fusion widths (C5).  128 local + 128 global + 64 + 64 HUD + 16 location = 400.
LOCAL_VEC = 128
GLOBAL_VEC = 128
HUD_VEC = 64
LOCATION_VEC = 16
LOCATION_INPUT = 4  # rx, ry, normalised entropy, max probability
FUSED_VEC = LOCAL_VEC + GLOBAL_VEC + 2 * HUD_VEC + LOCATION_VEC  # 400
TRUNK_HIDDEN = 256

#: v2 parity for the two HUD pools (``vision_train``/checkpoints do not read them,
#: but the v2 comparison swaps one architecture for the other).
HUD_SCORE_POOL = (2, 8)
HUD_STATUS_POOL = (8, 4)

#: ``self_feat[0:2]`` stays (rel_x, rel_y) taken from the heatmap (C3: the self
#: auxiliary supervises columns 4..12, 14..23 and leaves 2,3 reserved / 13
#: constant), so the head regresses the remaining 22 columns.
SELF_HEAD_DIM = obs.N_SELF - 2

#: Global pooled cell-centre coordinates appended before the fusion linear layer.
GLOBAL_POOL_HW = (7, 6)

#: Fixed config enums (C5).  Anything else is ``v104:CONFIG``, never a fallback.
CANDIDATE_ORDER_VERSION = "focus9-move-v1"
SCORE_SEMANTICS = "preference_logits"


def _pair(value, field: str) -> tuple[int, int]:
    """``value`` as a positive ``(H, W)`` integer pair, or ``v104:CONFIG``."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise v104_error("CONFIG", f"{field} must be an (H, W) pair, got {value!r}")
    components = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, numbers.Integral):
            raise v104_error("CONFIG", f"{field} must be integer (H, W), got {value!r}")
        if int(component) <= 0:
            raise v104_error("CONFIG", f"{field} must be positive (H, W), got {value!r}")
        components.append(int(component))
    return components[0], components[1]


@dataclass
class VisionSpatialTemporalConfig:
    """C5 configuration for :class:`VisionSpatialTemporalPolicy`.

    ``region_names``/``region_sizes``/``heatmap_hw`` mirror
    :class:`touhou_rl.vmodels2.VisionNetConfigV2` (properties over the ``*_hw``
    fields), so the trainer, the evaluator and the checkpoint writer read one
    layout regardless of the architecture.  ``state_head`` is deliberately *not*
    exposed: v3 has a single spatial head and C5 allows leaving the switch out
    instead of pretending a disabled state path exists.
    """

    depth: int = 3
    playfield_hw: tuple[int, int] = (224, 192)
    hud_score_hw: tuple[int, int] = (34, 216)
    hud_status_hw: tuple[int, int] = (126, 216)
    #: ``False`` keeps every parameter slot but feeds an all-zero 128-vector into
    #: fusion (E4 ablation); it must not change the model's capacity.
    local_enabled: bool = True
    #: Native playfield pixels covered by the local crop (128x128), and the bin
    #: grid it is resampled onto (16x16 -> 8 native px per bin).
    roi_native_hw: tuple[int, int] = (128, 128)
    local_grid_hw: tuple[int, int] = (16, 16)
    #: Must stay ``False``: the grid formula below assumes half-pixel centres.
    align_corners: bool = False
    #: The local crop is centred on the *predicted* position, detached, so the
    #: policy loss never trains the heatmap through the sampling coordinates.
    center_detach: bool = True
    #: Gaussian sigma of the training target, in heatmap cells (C5: 2.0 keeps the
    #: physical width of v2's 1.5 cells at the coarser v2 stride).
    heatmap_gauss_sigma: float = 2.0
    candidate_order_version: str = CANDIDATE_ORDER_VERSION
    history_contract: str = "causal-segment-v1"
    score_semantics: str = SCORE_SEMANTICS
    #: Trunk widths, C5 defaults.  Kept as fields so a unit test can shrink them
    #: without touching the layer geometry.
    pf_width: int = 32
    hud_width: int = 16
    trunk_hidden: int = TRUNK_HIDDEN

    def __post_init__(self) -> None:
        if isinstance(self.depth, bool) or not isinstance(self.depth, numbers.Integral):
            raise v104_error("CONFIG", f"depth must be an integer, got {self.depth!r}")
        self.depth = int(self.depth)
        if self.depth != 3:
            raise v104_error(
                "CONFIG", f"depth must be 3 for v3-st (this round is depth 3 only), got {self.depth}"
            )
        self.playfield_hw = _pair(self.playfield_hw, "playfield_hw")
        self.hud_score_hw = _pair(self.hud_score_hw, "hud_score_hw")
        self.hud_status_hw = _pair(self.hud_status_hw, "hud_status_hw")
        self.roi_native_hw = _pair(self.roi_native_hw, "roi_native_hw")
        self.local_grid_hw = _pair(self.local_grid_hw, "local_grid_hw")
        height, width = self.playfield_hw
        if height % PLAYFIELD_STRIDE or width % PLAYFIELD_STRIDE:
            raise v104_error(
                "CONFIG",
                f"playfield H/W must be divisible by {PLAYFIELD_STRIDE}, got "
                f"{(height, width)}",
            )
        if self.roi_native_hw[0] % self.local_grid_hw[0] or self.roi_native_hw[1] % self.local_grid_hw[1]:
            raise v104_error(
                "CONFIG",
                f"roi_native_hw {self.roi_native_hw} must be divisible by local_grid_hw "
                f"{self.local_grid_hw}",
            )
        if self.align_corners is not False:
            raise v104_error("CONFIG", f"align_corners must be False, got {self.align_corners!r}")
        for field in ("local_enabled", "center_detach"):
            if not isinstance(getattr(self, field), bool):
                raise v104_error("CONFIG", f"{field} must be a bool, got {getattr(self, field)!r}")
        sigma = self.heatmap_gauss_sigma
        if isinstance(sigma, bool) or not isinstance(sigma, numbers.Real):
            raise v104_error("CONFIG", f"heatmap_gauss_sigma must be a number, got {sigma!r}")
        if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
            raise v104_error(
                "CONFIG", f"heatmap_gauss_sigma must be finite and > 0, got {sigma!r}"
            )
        self.heatmap_gauss_sigma = float(sigma)
        if self.candidate_order_version != CANDIDATE_ORDER_VERSION:
            raise v104_error(
                "CONFIG",
                f"candidate_order_version must be {CANDIDATE_ORDER_VERSION!r}, got "
                f"{self.candidate_order_version!r}",
            )
        if self.score_semantics != SCORE_SEMANTICS:
            raise v104_error(
                "CONFIG",
                f"score_semantics must be {SCORE_SEMANTICS!r}, got {self.score_semantics!r}",
            )
        if self.history_contract not in HISTORY_CONTRACTS:
            raise v104_error(
                "CONFIG",
                f"history_contract must be one of {list(HISTORY_CONTRACTS)}, got "
                f"{self.history_contract!r}",
            )
        for field in ("pf_width", "hud_width", "trunk_hidden"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0:
                raise v104_error("CONFIG", f"{field} must be a positive integer, got {value!r}")
            setattr(self, field, int(value))

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
        """(H, W) of the single heatmap = ``playfield_hw / 4`` (56x48)."""
        return (self.playfield_hw[0] // 4, self.playfield_hw[1] // 4)

    @property
    def native_playfield_hw(self) -> tuple[int, int]:
        return NATIVE_PLAYFIELD_HW


def decode_location(heatmap_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B, H, W]`` logits -> ``(position[B,2], location_stats[B,2])``.

    ``position`` is the shared :func:`touhou_rl.vmodels2.decode_position`
    soft-argmax result ``(rx, ry)`` in ``[0, 1]`` - reusing the v2 function keeps
    training metrics, deployment and both architectures on one definition.

    ``location_stats`` is ``[entropy / log(H*W), max_probability]``: the
    normalised entropy is in ``[0, 1]`` (1 = uniform, 0 = a single cell), so the
    two numbers are scale free and can be fed to the fusion trunk next to the
    coordinates.
    """
    if heatmap_logits.ndim != 3:
        raise v104_error(
            "DATA", f"heatmap_logits must be [B, H, W], got shape {tuple(heatmap_logits.shape)}"
        )
    rx, ry = decode_position(heatmap_logits)
    position = torch.stack([rx, ry], dim=-1)
    probabilities = torch.softmax(heatmap_logits.flatten(1), dim=-1)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum(-1)
    cells = heatmap_logits.size(-1) * heatmap_logits.size(-2)
    normalised = entropy / math.log(cells) if cells > 1 else torch.zeros_like(entropy)
    stats = torch.stack([normalised, probabilities.max(dim=-1).values], dim=-1)
    return position, stats


def local_grid(position: torch.Tensor, config: VisionSpatialTemporalConfig) -> torch.Tensor:
    """``[B,2]`` normalised (rx, ry) -> ``[B, grid_h, grid_w, 2]`` sample grid.

    C5, literally: ``grid_x = 2*rx - 1 + 2*dx/384`` and
    ``grid_y = 2*ry - 1 + 2*dy/448``, where ``dx``/``dy`` are the native-pixel
    offsets of the 16x16 bin centres from the window centre (``dx`` in
    ``{-60, -52, ..., 60}`` for a 128 px window) and 384/448 are the native
    playfield width/height.  ``align_corners=False`` is assumed, which is why the
    result is directly a ``grid_sample`` coordinate in ``[-1, 1]``.
    """
    if position.ndim != 2 or position.size(-1) != 2:
        raise v104_error("DATA", f"position must be [B, 2], got shape {tuple(position.shape)}")
    grid_h, grid_w = config.local_grid_hw
    roi_h, roi_w = config.roi_native_hw
    native_h, native_w = config.native_playfield_hw
    dtype, device = position.dtype, position.device
    # Bin centres in native pixels, relative to the window centre.
    offsets_x = (torch.arange(grid_w, dtype=dtype, device=device) + 0.5) * (
        roi_w / grid_w
    ) - roi_w / 2.0
    offsets_y = (torch.arange(grid_h, dtype=dtype, device=device) + 0.5) * (
        roi_h / grid_h
    ) - roi_h / 2.0
    rx = position[:, 0].view(-1, 1, 1)
    ry = position[:, 1].view(-1, 1, 1)
    grid_x = 2.0 * rx - 1.0 + 2.0 * offsets_x.view(1, 1, grid_w) / native_w
    grid_y = 2.0 * ry - 1.0 + 2.0 * offsets_y.view(1, grid_h, 1) / native_h
    return torch.stack(
        [grid_x.expand(-1, grid_h, grid_w), grid_y.expand(-1, grid_h, grid_w)], dim=-1
    )


def sample_local(
    pixels: torch.Tensor, position: torch.Tensor, config: VisionSpatialTemporalConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``pixels[B, C, Hf, Wf]`` at the local grid -> ``(patch, validity)``.

    ``validity`` is the same grid sampled on an all-ones ``[B, 1, Hf, Wf]`` map:
    1.0 for a bin fully inside the playfield, below 1 at the edge, 0 outside.  It
    is concatenated to ``F4`` as the 65th channel so the local CNN can tell real
    darkness from "outside the field".
    """
    if pixels.ndim != 4:
        raise v104_error(
            "DATA", f"local pixels must be [B, C, H, W], got shape {tuple(pixels.shape)}"
        )
    if pixels.size(0) != position.size(0):
        raise v104_error(
            "DATA",
            f"local pixels batch {pixels.size(0)} != position batch {position.size(0)}",
        )
    grid = local_grid(position, config)
    patch = F.grid_sample(
        pixels,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=config.align_corners,
    )
    ones = torch.ones(
        pixels.size(0), 1, pixels.size(-2), pixels.size(-1), dtype=pixels.dtype, device=pixels.device
    )
    validity = F.grid_sample(
        ones,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=config.align_corners,
    )
    return patch, validity


def _conv_block(
    in_channels: int, out_channels: int, kernel: int, stride: int, padding: int
) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
    ]


def _conv_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


class ResidualBlock(nn.Module):
    """Two 3x3 convs, ReLU, no BatchNorm, added to the input (C5).

    Post-activation residual: ``ReLU(x + conv(ReLU(conv(x))))``.  BatchNorm is
    excluded by C5 - the training batch is small and the danger/heatmap targets
    are sparse, so batch statistics would couple unrelated frames.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x))


class PlayfieldTrunk(nn.Module):
    """Conv5 s2 -> Conv3 s2 (F4) -> 2 residuals -> Conv3 s2 -> 2 residuals (F8)."""

    def __init__(self, config: VisionSpatialTemporalConfig):
        super().__init__()
        width = config.pf_width
        channels = vision.VISION_CHANNELS * config.depth
        self.stem = nn.Sequential(*_conv_block(channels, width, 5, stride=2, padding=2))
        self.stage4 = nn.Sequential(*_conv_block(width, width * 2, 3, stride=2, padding=1))
        self.res4 = nn.Sequential(*(ResidualBlock(width * 2) for _ in range(2)))
        self.stage8 = nn.Sequential(*_conv_block(width * 2, width * 3, 3, stride=2, padding=1))
        self.res8 = nn.Sequential(*(ResidualBlock(width * 3) for _ in range(2)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f4 = self.res4(self.stage4(self.stem(x)))
        f8 = self.res8(self.stage8(f4))
        return f4, f8


class LocalBranch(nn.Module):
    """65-channel 16x16 local map -> 64 -> 96 -> flatten 96x4x4 -> 128.

    Padding is 1 on both convs so the two stride-2 steps halve 16 -> 8 -> 4; the
    linear input size is derived from ``config.local_grid_hw`` instead of being
    hard-coded.
    """

    def __init__(self, config: VisionSpatialTemporalConfig):
        super().__init__()
        channels = config.pf_width * 2  # F4 channels
        width = config.pf_width * 2
        grid_h, grid_w = config.local_grid_hw
        out_h = _conv_out(_conv_out(grid_h, 3, 2, 1), 3, 2, 1)
        out_w = _conv_out(_conv_out(grid_w, 3, 2, 1), 3, 2, 1)
        self.conv1 = nn.Conv2d(channels + 1, width, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(width, config.pf_width * 3, 3, stride=2, padding=1)
        self.fc = nn.Linear(config.pf_width * 3 * out_h * out_w, LOCAL_VEC)

    def forward(self, local_input: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(local_input))
        x = F.relu(self.conv2(x))
        return self.fc(x.flatten(1))


class HudBranch(nn.Module):
    """HUD CNN, v2 structure, newest frame only.

    ``pool_hw`` is the only difference between the two HUD streams; they never
    share parameters.  The input channel count is 3 (one RGB frame), not
    ``3 * depth``: the playfield carries motion, the HUD carries state.
    """

    def __init__(self, config: VisionSpatialTemporalConfig, pool_hw: tuple[int, int]):
        super().__init__()
        width = config.hud_width
        channels = vision.VISION_CHANNELS
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


class DangerHead(nn.Module):
    """F8 -> 32ch 1x1 -> bilinear 32x16 -> Conv3/ReLU -> AvgPool2 -> Conv1 -> 128.

    Output is a log count per 16x8 danger cell, row-major, with **no** final ReLU:
    the target is ``log1p(count / 8)`` whose value is 0 for an empty cell, and a
    ReLU would clamp the prediction to the target's lower bound instead of
    letting the loss choose.
    """

    def __init__(self, config: VisionSpatialTemporalConfig):
        super().__init__()
        channels = config.pf_width * 3
        self.reduce = nn.Conv2d(channels, 32, 1)
        self.target_hw = (2 * obs.DANGER_ROWS, 2 * obs.DANGER_COLS)  # 32 x 16
        self.conv = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(2)  # 32x16 -> 16x8 == the danger grid
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, f8: torch.Tensor) -> torch.Tensor:
        x = self.reduce(f8)
        x = F.interpolate(x, size=self.target_hw, mode="bilinear", align_corners=False)
        x = F.relu(self.conv(x))
        x = self.pool(x)
        return self.head(x).flatten(1)  # [B, 128], row-major


class VisionSpatialTemporalPolicy(nn.Module):
    """C5 ``v3-st`` policy: one encode -> heatmap, danger, self and 18 preferences.

    Legacy surface (same call sites as v1/v2): ``features``, ``score``,
    ``decode_state``, ``act``, ``num_parameters``.  ``forward`` returns the seven
    C5 tensors in one pass.
    """

    ARCH = ARCH

    def __init__(self, config: VisionSpatialTemporalConfig | None = None):
        super().__init__()
        self.config = config or VisionSpatialTemporalConfig()
        self.playfield = PlayfieldTrunk(self.config)
        channels4 = self.config.pf_width * 2  # 64
        channels8 = self.config.pf_width * 3  # 96
        self.heat_conv = nn.Sequential(*(_conv_block(channels4 + channels8, channels4, 3, 1, 1)))
        self.heatmap_head = nn.Conv2d(channels4, 1, 1)
        self.location_mlp = nn.Sequential(
            nn.Linear(LOCATION_INPUT, LOCATION_VEC),
            nn.ReLU(inplace=True),
        )
        self.local = LocalBranch(self.config)
        self.global_conv = nn.Conv2d(channels8, 32, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(GLOBAL_POOL_HW)
        self.global_fc = nn.Sequential(
            nn.Linear((32 + 2) * GLOBAL_POOL_HW[0] * GLOBAL_POOL_HW[1], GLOBAL_VEC),
            nn.ReLU(inplace=True),
        )
        self.hud_score = HudBranch(self.config, HUD_SCORE_POOL)
        self.hud_status = HudBranch(self.config, HUD_STATUS_POOL)
        self.trunk = nn.Sequential(
            nn.Linear(FUSED_VEC, self.config.trunk_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.trunk_hidden, self.config.trunk_hidden),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(self.config.trunk_hidden, N_CANDIDATES)
        self.self_head = nn.Linear(self.config.trunk_hidden, SELF_HEAD_DIM)
        self.danger = DangerHead(self.config)

    # -- input contract ---------------------------------------------------- #
    def _checked_regions(self, regions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Validate the three ROI tensors; ``KeyError``/``v104:DATA`` on violation."""
        if not isinstance(regions, dict):
            raise v104_error(
                "DATA", f"regions must be a dict of tensors, got {type(regions).__name__}"
            )
        missing = set(self.config.region_names) - set(regions)
        if missing:
            raise KeyError(f"missing regions {sorted(missing)}; got {sorted(regions)}")
        checked: dict[str, torch.Tensor] = {}
        batch = None
        for name in self.config.region_names:
            tensor = regions[name]
            if not isinstance(tensor, torch.Tensor):
                raise v104_error(
                    "DATA", f"region {name!r} must be a torch.Tensor, got {type(tensor).__name__}"
                )
            if tensor.ndim == 4:  # (depth, H, W, C) -> (1, depth, H, W, C)
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 5:
                raise v104_error(
                    "DATA",
                    f"region {name!r} must be [B, depth, H, W, 3], got shape "
                    f"{tuple(tensor.shape)}",
                )
            height, width = self.config.region_sizes[name]
            this_batch, depth, got_h, got_w, channels = tensor.shape
            if this_batch < 1:
                raise v104_error("DATA", f"region {name!r} batch must be >= 1, got {this_batch}")
            if channels != vision.VISION_CHANNELS:
                raise v104_error(
                    "DATA",
                    f"region {name!r} must have {vision.VISION_CHANNELS} RGB channels, got "
                    f"{channels}",
                )
            if depth != self.config.depth:
                raise v104_error(
                    "DATA",
                    f"region {name!r} depth {depth} != config depth {self.config.depth}",
                )
            if (got_h, got_w) != (height, width):
                raise v104_error(
                    "DATA",
                    f"region {name!r} is {got_h}x{got_w}, config says {height}x{width}",
                )
            if tensor.dtype != torch.float32:
                raise v104_error(
                    "DATA", f"region {name!r} dtype {tensor.dtype} != torch.float32"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise v104_error("DATA", f"region {name!r} contains non-finite values")
            if batch is None:
                batch = this_batch
            elif this_batch != batch:
                raise v104_error(
                    "DATA",
                    f"region {name!r} batch {this_batch} != {batch} of the other regions",
                )
            checked[name] = tensor
        return checked

    @staticmethod
    def _playfield_input(region: torch.Tensor) -> torch.Tensor:
        """``[B, depth, H, W, 3]`` -> ``[B, 3*depth, H, W]`` (all frames)."""
        batch, depth, height, width, channels = region.shape
        return region.permute(0, 1, 4, 2, 3).reshape(batch, depth * channels, height, width)

    @staticmethod
    def _hud_input(region: torch.Tensor) -> torch.Tensor:
        """``[B, depth, H, W, 3]`` -> ``[B, 3, H, W]`` of the **newest** frame."""
        return region[:, -1].permute(0, 3, 1, 2)

    # -- layer accessors (used by forward and by the geometry tests) -------- #
    def locate(self, f4: torch.Tensor, f8: torch.Tensor) -> torch.Tensor:
        """F8 upsampled to the F4 grid, concatenated (160) -> 64 -> 1 heatmap."""
        upsampled = F.interpolate(
            f8, size=f4.shape[-2:], mode="bilinear", align_corners=self.config.align_corners
        )
        fused = torch.cat([f4, upsampled], dim=1)
        return self.heatmap_head(self.heat_conv(fused)).squeeze(1)  # [B, 56, 48]

    def local_features(self, f4: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """The 128-d local vector; exactly zeros when ``local_enabled`` is False.

        ``center_detach`` blocks the policy gradient **through the sampling
        coordinates only** (C5).  The four location features that enter fusion
        (rx, ry, entropy, peak) are not detached there, so the policy objective
        does reach the heatmap through ``location_mlp`` as well; the auxiliary
        heatmap loss remains its dominant gradient by orders of magnitude
        (measured in ``Stage3/S3-5/evidence/11-shape-grad-probe.log``).
        """
        if not self.config.local_enabled:
            return torch.zeros(
                f4.size(0), LOCAL_VEC, dtype=f4.dtype, device=f4.device
            )
        centre = position.detach() if self.config.center_detach else position
        patch, validity = sample_local(f4, centre, self.config)
        return self.local(torch.cat([patch, validity], dim=1))

    def global_features(self, f8: torch.Tensor) -> torch.Tensor:
        """F8 -> 32ch 1x1 -> pool 7x6 -> append cell-centre x/y -> 128."""
        pooled = self.global_pool(self.global_conv(f8))  # [B, 32, 7, 6]
        rows, columns = GLOBAL_POOL_HW
        ys = (torch.arange(rows, dtype=pooled.dtype, device=pooled.device) + 0.5) / rows
        xs = (torch.arange(columns, dtype=pooled.dtype, device=pooled.device) + 0.5) / columns
        grid_y = ys.view(1, 1, rows, 1).expand(pooled.size(0), 1, rows, columns)
        grid_x = xs.view(1, 1, 1, columns).expand(pooled.size(0), 1, rows, columns)
        with_coordinates = torch.cat([pooled, grid_y, grid_x], dim=1)  # [B, 34, 7, 6]
        return self.global_fc(with_coordinates.flatten(1))

    # -- core -------------------------------------------------------------- #
    def forward_core(self, regions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Validate, encode once, and return every C5 tensor."""
        checked = self._checked_regions(regions)
        f4, f8 = self.playfield(self._playfield_input(checked["playfield"]))
        heatmap_logits = self.locate(f4, f8)
        position, location_stats = decode_location(heatmap_logits)
        location = self.location_mlp(
            torch.cat([position, location_stats], dim=-1)
        )  # [B, 16]
        local_vec = self.local_features(f4, position)
        global_vec = self.global_features(f8)
        hud_score_vec = self.hud_score(self._hud_input(checked["hud_score"]))
        hud_status_vec = self.hud_status(self._hud_input(checked["hud_status"]))
        fused = torch.cat(
            [local_vec, global_vec, hud_score_vec, hud_status_vec, location], dim=-1
        )  # [B, 400]
        features = self.trunk(fused)
        self_pred = torch.cat([position, self.self_head(features)], dim=-1)  # [B, 24]
        return {
            "policy_logits": self.policy_head(features),
            "heatmap_logits": heatmap_logits,
            "position": position,
            "danger_log": self.danger(f8),
            "self_pred": self_pred,
            "features": features,
            "location_stats": location_stats,
        }

    # -- public contract ---------------------------------------------------- #
    def forward(self, regions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.forward_core(regions)

    def _encode(
        self, regions: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """v2-parity helper: ``(features, heatmap_logits, danger_log)``."""
        out = self.forward_core(regions)
        return out["features"], out["heatmap_logits"], out["danger_log"]

    def features(self, regions: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_core(regions)["features"]

    def _checked_features(self, features: torch.Tensor, batch: int) -> torch.Tensor:
        if features.ndim != 2 or features.size(0) != batch:
            raise v104_error(
                "DATA",
                f"features must be [B={batch}, {self.config.trunk_hidden}], got shape "
                f"{tuple(features.shape)}",
            )
        if features.size(1) != self.config.trunk_hidden:
            raise v104_error(
                "DATA",
                f"features width {features.size(1)} != {self.config.trunk_hidden}",
            )
        if features.dtype != torch.float32:
            raise v104_error("DATA", f"features dtype {features.dtype} != torch.float32")
        if not bool(torch.isfinite(features).all()):
            raise v104_error("DATA", "features contain non-finite values")
        return features

    def score(
        self, regions: dict[str, torch.Tensor], features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``[B, 18]`` preference logits (not values, not px clearance)."""
        if features is None:
            return self.forward_core(regions)["policy_logits"]
        return self.policy_head(self._checked_features(features, self._batch_of(regions)))

    def _batch_of(self, regions: dict[str, torch.Tensor]) -> int:
        checked = self._checked_regions(regions)
        return checked["playfield"].size(0)

    def decode_state(
        self, regions: dict[str, torch.Tensor], features: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """``{'self_feat', 'danger_log', 'heatmap_logits'}`` - v2 key set.

        The spatial tensors are **always** recomputed from ``regions``: an
        external ``features`` vector can only replace the trunk input of the self
        head, never stand in for the heatmap or the danger grid.  That is the C5
        rule that stops a caller (or a bug) from pairing a heatmap with the wrong
        pixels.
        """
        out = self.forward_core(regions)
        if features is None:
            self_feat = out["self_pred"]
        else:
            checked = self._checked_features(features, out["position"].size(0))
            self_feat = torch.cat([out["position"], self.self_head(checked)], dim=-1)
        return {
            "self_feat": self_feat,
            "danger_log": out["danger_log"],
            "heatmap_logits": out["heatmap_logits"],
        }

    def act(self, regions: dict[str, torch.Tensor], deterministic: bool = True):
        """Greedy ``[move, shoot, focus, bomb]`` from pixels (C1 mapping).

        ``index = focus*9 + move``; shooting stays on and bombing is left to a
        rule.  ``logp``/``extra`` are zero placeholders and the third element is
        the max *preference*, not a value estimate - this output must never be
        handed to PPO.  ``deterministic`` is accepted for the v1/v2 call
        signature; v3 only implements the greedy branch.
        """
        values = self.score(regions)
        best = values.argmax(dim=-1)
        move = best % spec.ACTION_DIMS[0]
        focus = best // spec.ACTION_DIMS[0]
        actions = torch.stack(
            [move, torch.ones_like(move), focus, torch.zeros_like(move)], dim=-1
        )
        zero = torch.zeros(values.size(0), dtype=values.dtype, device=values.device)
        return actions, zero, values.max(dim=-1).values, zero

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = [
    "ARCH",
    "CANDIDATE_ORDER_VERSION",
    "DangerHead",
    "FUSED_VEC",
    "GLOBAL_VEC",
    "HudBranch",
    "LOCAL_VEC",
    "LocalBranch",
    "NATIVE_PLAYFIELD_HW",
    "PLAYFIELD_STRIDE",
    "PlayfieldTrunk",
    "ResidualBlock",
    "SCORE_SEMANTICS",
    "VisionSpatialTemporalConfig",
    "VisionSpatialTemporalPolicy",
    "decode_location",
    "local_grid",
    "sample_local",
]
