"""Real-frame front end for D11 (pixels only, no memory reads).

Status: **extraction only, not calibrated**.  This module finds the playfield in a
640x480 game frame, extracts bright bullet cores and the red player point, and
reports what it found.  What it does *not* do yet is map pixels to game
coordinates: that needs a frame whose memory snapshot is known (same instant), so
the scale and offset can be measured instead of guessed.  Until then nothing here
may be used to drive the controller - the ``calibrated`` flag is False and the
adapter refuses pixel-only observations without an explicit opt-in.

The separation matters: a vision front end that silently guesses its scale looks
like it works (blobs are found) while producing wrong clearance values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: TH08 draws the playfield as a 384x448 rectangle.  In the reference frames the
#: window is larger than that and the field sits inside it, so the crop box is
#: searched rather than assumed.
PLAYFIELD_SIZE = (384, 448)


@dataclass(frozen=True)
class PlayfieldBox:
    left: int
    top: int
    width: int
    height: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)


def _array(frame) -> np.ndarray:
    """Accept a path, a PIL image or an array; return HxWx3 uint8."""

    if isinstance(frame, np.ndarray):
        array = frame
    elif isinstance(frame, (str, bytes)) or hasattr(frame, "read"):
        from PIL import Image

        array = np.asarray(Image.open(frame).convert("RGB"))
    else:  # PIL image
        array = np.asarray(frame.convert("RGB"))
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 frame, got {array.shape}")
    return array.astype(np.uint8)


def detect_playfield(frame, *, dark: int = 16) -> PlayfieldBox:
    """Find the dark playfield rectangle inside a windowed game frame.

    The desktop around the window is not uniformly dark, so the box is found by
    trimming fully dark rows/columns from the *inside* of the frame: the playfield
    border is the last band that is (almost) entirely dark, and the interior holds
    the content.  This is a heuristic and it is reported as such — the calibrated
    path must confirm it against a paired snapshot.
    """

    array = _array(frame)
    gray = array.mean(axis=2)
    height, width = gray.shape
    dark_mask = gray < dark

    # A column belongs to the field if it is mostly dark *and* the frame has
    # content around it; desktop icons break a pure "darkest rectangle" search.
    column_ratio = dark_mask.mean(axis=0)
    row_ratio = dark_mask.mean(axis=1)
    if not (column_ratio > 0.35).any() or not (row_ratio > 0.35).any():
        raise ValueError("playfield not found: no mostly-dark band in the frame")
    left = int(np.argmax(column_ratio > 0.35))
    right = int(width - np.argmax((column_ratio > 0.35)[::-1]))
    top = int(np.argmax(row_ratio > 0.35))
    bottom = int(height - np.argmax((row_ratio > 0.35)[::-1]))
    if right - left < PLAYFIELD_SIZE[0] // 2 or bottom - top < PLAYFIELD_SIZE[1] // 2:
        raise ValueError(
            f"playfield not found (box {left},{top},{right},{bottom}); the frame may be "
            "a menu rather than gameplay"
        )
    aspect = (right - left) / max(1, bottom - top)
    expected = PLAYFIELD_SIZE[0] / PLAYFIELD_SIZE[1]
    if abs(aspect - expected) > expected * 0.25:
        raise ValueError(
            f"playfield box aspect {aspect:.2f} does not match {expected:.2f}: the frame "
            "probably includes the desktop, so the box must be given explicitly"
        )
    return PlayfieldBox(left, top, right - left, bottom - top)


def crop(frame, box: PlayfieldBox) -> np.ndarray:
    array = _array(frame)
    left, top, right, bottom = box.box
    return array[top:bottom, left:right]


def bullet_mask(playfield: np.ndarray, *, brightness: int = 170, saturation: int = 40) -> np.ndarray:
    """Bright, low-saturation pixels: the white/pale cores of bullets and the player."""

    array = playfield.astype(np.int16)
    value = array.max(axis=2)
    minimum = array.min(axis=2)
    saturation_span = value - minimum
    return (value >= brightness) & (saturation_span <= saturation)


def player_mask(playfield: np.ndarray, *, red_min: int = 150) -> np.ndarray:
    """The player's red hitbox dot (strong red, weak green/blue)."""

    array = playfield.astype(np.int16)
    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    return (red >= red_min) & (red - green >= 80) & (red - blue >= 80)


def blobs(mask: np.ndarray, *, minimum_px: int = 3, maximum_px: int = 400) -> list[dict]:
    """Connected components of a mask with centroid and half extents in pixels."""

    from collections import deque

    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found: list[dict] = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            queue = deque([(row, col)])
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                for delta_row in (-1, 0, 1):
                    for delta_col in (-1, 0, 1):
                        next_row = current_row + delta_row
                        next_col = current_col + delta_col
                        if not (0 <= next_row < height and 0 <= next_col < width):
                            continue
                        if seen[next_row, next_col] or not mask[next_row, next_col]:
                            continue
                        seen[next_row, next_col] = True
                        queue.append((next_row, next_col))
            if not (minimum_px <= len(cells) <= maximum_px):
                continue
            rows = np.array([cell[0] for cell in cells], dtype=np.float64)
            cols = np.array([cell[1] for cell in cells], dtype=np.float64)
            found.append(
                {
                    "x": float(cols.mean()),
                    "y": float(rows.mean()),
                    "half_x": float((cols.max() - cols.min() + 1) / 2.0),
                    "half_y": float((rows.max() - rows.min() + 1) / 2.0),
                    "pixels": len(cells),
                }
            )
    return found


def extract(frame, *, box: PlayfieldBox | None = None) -> dict:
    """Extraction summary for one frame.  ``calibrated`` is always False for now."""

    array = _array(frame)
    playfield = crop(array, box) if box is not None else crop(array, detect_playfield(array))
    bright = bullet_mask(playfield)
    red = player_mask(playfield)
    bullets = blobs(bright, minimum_px=3, maximum_px=200)
    player_blobs = blobs(red, minimum_px=2, maximum_px=400)
    return {
        "schema": "v370-screen-extract/1",
        "calibrated": False,
        "box": list(box.box) if box is not None else list(detect_playfield(array).box),
        "shape": list(playfield.shape),
        "bullet_blobs": len(bullets),
        "player_blobs": len(player_blobs),
        "player": player_blobs[0] if player_blobs else None,
        "bullets": bullets[:12],
        "notes": (
            "pixel coordinates only; the pixel->game scale needs a frame paired with "
            "a memory snapshot before anything here can drive the controller"
        ),
    }
