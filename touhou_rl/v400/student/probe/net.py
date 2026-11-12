"""The shared perception head, the pixel player locator, and F1 accounting.

One head serves all four candidate inputs (EXP-P3-010 0.3):

    conv3x3(C_in->16) ReLU -> conv3x3(16->16) ReLU -> conv3x3(16->16) ReLU
    -> conv1x1(16->1) -> per-cell max-pool -> sigmoid -> 24x24 occupancy

There is deliberately **no** global pooling, no adaptive average pool to 1 and
no fully-connected layer anywhere: the point of the probe is to test how much
*position* survives, so the only aggregation allowed is "the maximum logit over
the input pixels that fall inside one 4-px label cell", which is what
"this cell contains a lethal object" means.

The locator is the deployment-legal piece input D depends on: it sees only the
64x64 playfield plane (the same plane obs-v1 gives a student) and returns a
sub-pixel ship position.  Its median error on the holdout seeds is the gate
that disqualifies D when it exceeds 4 px.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from touhou_rl.v400.labels.spec import (
    GRID_CELL,
    GRID_HEIGHT,
    GRID_WIDTH,
    PLAYFIELD_HEIGHT,
    PLAYFIELD_WIDTH,
)
from touhou_rl.v400.student.probe.inputs import (
    INPUT_SPECS,
    LOCAL_MARGIN_PX,
    LOCAL_SIDE_PX,
)
from touhou_rl.v400.student.probe.labels import WINDOW_CELLS, WINDOW_PX, cell_origin

#: Block sizes of the 64x64 plane in native playfield pixels (7 rows, 6 cols).
LOCATOR_BLOCK_Y = PLAYFIELD_HEIGHT // 64
LOCATOR_BLOCK_X = PLAYFIELD_WIDTH // 64
#: Native-pixel centre of plane cell 0, used by the locator's coordinate map.
LOCATOR_OFFSET_X = (LOCATOR_BLOCK_X - 1) / 2.0
LOCATOR_OFFSET_Y = (LOCATOR_BLOCK_Y - 1) / 2.0


class HeadShapeError(ValueError):
    """An input tensor does not match the pre-registered shapes."""


def _block(in_channels: int, out_channels: int, kernel: int = 3) -> nn.Conv2d:
    padding = kernel // 2
    return nn.Conv2d(in_channels, out_channels, kernel, padding=padding)


class OccupancyHead(nn.Module):
    """Small fully-convolutional occupancy head; identical for every input."""

    def __init__(self, in_channels: int = 1, width: int = 16) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.features = nn.Sequential(
            _block(self.in_channels, width), nn.ReLU(inplace=True),
            _block(width, width), nn.ReLU(inplace=True),
            _block(width, width), nn.ReLU(inplace=True),
        )
        self.score = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise HeadShapeError(
                f"expected (B,{self.in_channels},H,W), got {tuple(x.shape)}")
        return self.score(self.features(x))

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def window_logits(
    logits: torch.Tensor,
    *,
    name: str,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """Per-cell logits -> the 24x24 scored window, for any candidate input.

    ``logits`` is the head's output.  For the whole-field inputs it is max
    pooled to the 96x112 label grid and then cut at each sample's own anchor;
    for the local input D it is cut to ``[16:112]`` (constant) and max pooled
    straight to 24x24.  Both mean the same thing: max over the input pixels of
    a cell.
    """

    if logits.ndim != 4 or logits.shape[1] != 1:
        raise HeadShapeError(f"logits must be (B,1,H,W), got {tuple(logits.shape)}")
    if name == "D":
        if logits.shape[-1] != LOCAL_SIDE_PX or logits.shape[-2] != LOCAL_SIDE_PX:
            raise HeadShapeError(
                f"D's logits must be {LOCAL_SIDE_PX}x{LOCAL_SIDE_PX}, "
                f"got {tuple(logits.shape[-2:])}")
        inner = logits[
            :, :,
            LOCAL_MARGIN_PX:LOCAL_MARGIN_PX + WINDOW_PX,
            LOCAL_MARGIN_PX:LOCAL_MARGIN_PX + WINDOW_PX,
        ]
        return F.adaptive_max_pool2d(inner, (WINDOW_CELLS, WINDOW_CELLS))[:, 0]

    cells = F.adaptive_max_pool2d(logits, (GRID_HEIGHT, GRID_WIDTH))
    if anchors is None:
        raise HeadShapeError(f"input {name} needs per-sample window anchors")
    rows = anchors[:, 0].to(torch.long).view(-1, 1)
    cols = anchors[:, 1].to(torch.long).view(-1, 1)
    batch = cells.shape[0]
    if rows.shape[0] != batch or cols.shape[0] != batch:
        raise HeadShapeError(f"got {rows.shape[0]} anchors for a batch of {batch}")
    pad = WINDOW_CELLS
    padded = F.pad(cells, (pad, pad, pad, pad))
    offsets = torch.arange(WINDOW_CELLS, device=cells.device)
    row_index = (rows + offsets.view(1, -1) + pad).clamp(0, padded.shape[2] - 1)
    col_index = (cols + offsets.view(1, -1) + pad).clamp(0, padded.shape[3] - 1)
    # Flatten and gather instead of three-way advanced indexing: one explicit
    # linear index per output cell, no broadcast surprises.
    flat = padded.reshape(batch, -1)
    linear = (row_index.view(batch, -1, 1) * padded.shape[3]
              + col_index.view(batch, 1, -1)).reshape(batch, -1)
    return torch.gather(flat, 1, linear).reshape(batch, WINDOW_CELLS, WINDOW_CELLS)


class ProbeModel(nn.Module):
    """Head + window selection, so training code only sees ``(B,24,24)`` logits."""

    def __init__(self, name: str, width: int = 16) -> None:
        super().__init__()
        spec = INPUT_SPECS.get(name)
        if spec is None:
            raise KeyError(f"unknown input {name!r}; expected {sorted(INPUT_SPECS)}")
        self.name = name
        self.spec = spec
        self.head = OccupancyHead(in_channels=spec.channels, width=width)

    def forward(self, x: torch.Tensor, anchors: torch.Tensor | None = None) -> torch.Tensor:
        if tuple(x.shape[1:]) != self.spec.shape:
            raise HeadShapeError(
                f"input {self.name} must be {self.spec.shape}, got {tuple(x.shape[1:])}")
        logits = self.head(x)
        return window_logits(logits, name=self.name, anchors=anchors)


# ---------------------------------------------------------------------------
# pixel player locator (input D's dependency; deployment-legal)
# ---------------------------------------------------------------------------
class PlayerLocator(nn.Module):
    """64x64 playfield plane -> 64x64 ship heatmap (no state, pixels only)."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(1, width), nn.ReLU(inplace=True),
            _block(width, width), nn.ReLU(inplace=True),
            _block(width, width), nn.ReLU(inplace=True),
        )
        self.score = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise HeadShapeError(f"locator expects (B,1,64,64), got {tuple(x.shape)}")
        logits = self.score(self.features(x))
        return F.adaptive_max_pool2d(logits, (64, 64))


def player_cell_coords(player_x: float, player_y: float) -> tuple[float, float]:
    """Native playfield position -> continuous 64x64 plane coordinate."""

    u = (float(player_x) - LOCATOR_OFFSET_X) / LOCATOR_BLOCK_X
    v = (float(player_y) - LOCATOR_OFFSET_Y) / LOCATOR_BLOCK_Y
    return u, v


def cell_coords_to_playfield(u: float, v: float) -> tuple[float, float]:
    """Inverse of :func:`player_cell_coords`."""

    return (float(u) * LOCATOR_BLOCK_X + LOCATOR_OFFSET_X,
            float(v) * LOCATOR_BLOCK_Y + LOCATOR_OFFSET_Y)


def locator_target(player_x: float, player_y: float, *, sigma: float = 1.0,
                   size: int = 64) -> np.ndarray:
    """Gaussian heatmap whose centroid is the ship, in plane coordinates."""

    u, v = player_cell_coords(player_x, player_y)
    cols = np.arange(size, dtype=np.float32)
    rows = np.arange(size, dtype=np.float32)
    blob = np.exp(-(((cols[None, :] - u) ** 2 + (rows[:, None] - v) ** 2)
                    / (2.0 * sigma * sigma)))
    peak = float(blob.max())
    if peak > 0:
        blob = blob / peak
    return blob.astype(np.float32)


def decode_locator(heatmap: np.ndarray) -> tuple[float, float]:
    """Heatmap -> playfield position: argmax plus a 3x3 weighted centroid."""

    array = np.asarray(heatmap, dtype=np.float32)
    index = int(np.argmax(array))
    row, col = divmod(index, array.shape[1])
    row0, row1 = max(0, row - 1), min(array.shape[0], row + 2)
    col0, col1 = max(0, col - 1), min(array.shape[1], col + 2)
    patch = array[row0:row1, col0:col1].astype(np.float64)
    total = float(patch.sum())
    if total <= 0.0:
        return cell_coords_to_playfield(float(col), float(row))
    rows = np.arange(row0, row1, dtype=np.float64)
    cols = np.arange(col0, col1, dtype=np.float64)
    # Column centroid is the horizontal plane coordinate `u`, row centroid `v`.
    column_centroid = float((patch.sum(axis=0) * cols).sum() / total)
    row_centroid = float((patch.sum(axis=1) * rows).sum() / total)
    return cell_coords_to_playfield(column_centroid, row_centroid)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class F1Counts:
    """Cell-level confusion counts; F1 is always derived, never averaged."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, other: "F1Counts") -> "F1Counts":
        return F1Counts(self.tp + other.tp, self.fp + other.fp,
                        self.fn + other.fn, self.tn + other.tn)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp) / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp) / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return (2.0 * float(self.tp) / denom) if denom else 0.0

    @property
    def positives(self) -> int:
        return int(self.tp + self.fn)

    def as_dict(self) -> dict[str, object]:
        return {
            "tp": int(self.tp), "fp": int(self.fp), "fn": int(self.fn),
            "tn": int(self.tn), "f1": round(self.f1, 6),
            "precision": round(self.precision, 6), "recall": round(self.recall, 6),
            "positives": self.positives,
        }


def count_cells(predicted: np.ndarray, target: np.ndarray,
                valid: np.ndarray) -> F1Counts:
    """Confusion counts over valid cells only."""

    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if not (predicted.shape == target.shape == valid.shape):
        raise HeadShapeError(
            f"shape mismatch predicted={predicted.shape} target={target.shape} "
            f"valid={valid.shape}")
    tp = int(np.count_nonzero(predicted & target & valid))
    fp = int(np.count_nonzero(predicted & ~target & valid))
    fn = int(np.count_nonzero(~predicted & target & valid))
    tn = int(np.count_nonzero(~predicted & ~target & valid))
    return F1Counts(tp, fp, fn, tn)


def locator_error_px(predicted: tuple[float, float],
                     truth: tuple[float, float]) -> float:
    """Euclidean error in native playfield pixels."""

    return float(np.hypot(predicted[0] - truth[0], predicted[1] - truth[1]))


def anchor_offsets(player_x: float, player_y: float) -> tuple[int, int]:
    """Cell indices of the scored window's top-left inside the 96x112 grid."""

    return cell_origin(player_x, player_y)


__all__ = [
    "F1Counts",
    "HeadShapeError",
    "OccupancyHead",
    "PlayerLocator",
    "ProbeModel",
    "anchor_offsets",
    "cell_coords_to_playfield",
    "count_cells",
    "decode_locator",
    "locator_error_px",
    "locator_target",
    "player_cell_coords",
    "window_logits",
]
