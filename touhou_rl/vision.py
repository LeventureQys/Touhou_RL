"""Frame contract: user-selected box -> fixed-size RGB tensor.

Why this module exists
----------------------
The terminal goal of the project is *pixels*: the user selects a rectangle over
the game window and the agent learns to play from that image, ideally on other
Touhou games too.  Everything game-specific (window size, HUD placement,
playfield aspect ratio, sprite style) must therefore be confined to one small
layer: a box plus a resize.

Consequences that shaped the API
--------------------------------
* **The box is data, not a constant.**  :class:`PlayfieldBox` is a plain
  rectangle in client-area coordinates.  TH08's playfield is only its *default*
  (:meth:`PlayfieldBox.for_th08`); any other game supplies its own.
* **Output shape is fixed and small.**  ``VISION_H x VISION_W`` RGB at 1/3.5
  scale keeps ~43 KB per frame, which is what makes a 60 Hz capture stream
  affordable.  Aspect ratio is preserved by construction because the box is
  resized to a fixed tensor (any residual distortion is a property of the box
  the user chose, not of this code).
* **No normalisation here.**  Frames are stored as ``uint8`` (exact, compact);
  scaling to ``float32 / 255`` happens in :func:`frames_to_batch`, so the
  stored data never depends on a model's preprocessing choice.
* **Stacking is explicit.**  A single frame cannot distinguish "a bullet 10 px
  above me" from "a bullet 10 px below me moving up".  :class:`FrameStack`
  makes the history depth a visible parameter instead of a hidden assumption.

The privileged observation (``self_feat``/``bullets``/``danger``) is *not* part
of this contract; it is training-time label material and is defined in
:mod:`touhou_rl.obs`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import spec

# Output tensor shape (H, W, 3).  384x448 playfield -> 112x128 keeps the aspect
# ratio (0.857 vs 0.875) at 1/3.43 scale, which renders a 10 px bullet as ~3 px.
VISION_H = 128
VISION_W = 112
VISION_CHANNELS = 3


@dataclass(frozen=True)
class PlayfieldBox:
    """A rectangle in client-area pixel coordinates (top-left origin).

    ``x, y`` are inclusive offsets from the client area's top-left corner; the
    rectangle covers ``[x, x + w) x [y, y + h)``.
    """

    x: int
    y: int
    w: int
    h: int

    @property
    def aspect(self) -> float:
        return self.w / max(1, self.h)

    @classmethod
    def for_th08(cls) -> "PlayfieldBox":
        """TH08's playfield inside the 640x480 client area (values from spec)."""
        return cls(
            x=int(spec.PLAYFIELD_X),
            y=int(spec.PLAYFIELD_Y),
            w=int(spec.PLAYFIELD_W),
            h=int(spec.PLAYFIELD_H),
        )

    def clamped(self, client_w: int, client_h: int) -> "PlayfieldBox":
        """Fit the box inside a client area, never returning an empty rectangle.

        A stale or hand-written box is the most likely user error here, and a
        crop that silently returns an empty array would surface much later as a
        model-shape error, so the box is repaired in one place.
        """
        x = max(0, min(int(self.x), max(0, client_w - 1)))
        y = max(0, min(int(self.y), max(0, client_h - 1)))
        w = max(1, min(int(self.w), client_w - x))
        h = max(1, min(int(self.h), client_h - y))
        return replace(self, x=x, y=y, w=w, h=h)


def crop(frame: np.ndarray, box: PlayfieldBox) -> np.ndarray:
    """Slice ``frame`` (H, W, C) by ``box`` with clamping."""
    if frame.ndim != 3:
        raise ValueError(f"expected (H, W, C) frame, got shape {frame.shape}")
    height, width = frame.shape[:2]
    fitted = box.clamped(width, height)
    return frame[fitted.y : fitted.y + fitted.h, fitted.x : fitted.x + fitted.w]


def resize_area(image: np.ndarray, out_h: int = VISION_H, out_w: int = VISION_W) -> np.ndarray:
    """Area-average downscale to ``(out_h, out_w)``.

    Area averaging (``Image.BOX``) rather than bilinear on purpose: downscaling
    point-sized bullets with bilinear filtering can skip pixels entirely, so a
    4 px bullet could vanish from the tensor.  Averaging keeps its energy.
    """
    if image.shape[0] == out_h and image.shape[1] == out_w:
        return np.ascontiguousarray(image)
    from PIL import Image

    pil = Image.fromarray(np.ascontiguousarray(image))
    pil = pil.resize((out_w, out_h), Image.BOX)
    return np.asarray(pil)


def encode_frame(
    frame: np.ndarray,
    box: PlayfieldBox | None = None,
    *,
    out_h: int = VISION_H,
    out_w: int = VISION_W,
) -> np.ndarray:
    """Client-area RGB frame -> ``uint8[out_h, out_w, 3]`` observation."""
    if frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB frame, got dtype {frame.dtype}")
    if frame.shape[2] == 4:  # 32-bit DIB from GDI can carry a padding channel
        frame = frame[:, :, :3]
    box = box or PlayfieldBox.for_th08()
    return resize_area(crop(frame, box), out_h, out_w)


class FrameStack:
    """Fixed-depth frame history (oldest first), zero-initialised.

    History depth matters for a *vision* policy in a way it did not for the
    privileged one: the memory-reading observation carried bullet velocities
    explicitly, pixels do not, so motion has to come from consecutive frames.
    """

    def __init__(self, depth: int = 1, shape: tuple[int, int, int] = (VISION_H, VISION_W, VISION_CHANNELS)):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = int(depth)
        self.shape = shape
        self._frames = np.zeros((self.depth, *shape), dtype=np.uint8)

    def reset(self) -> None:
        self._frames[:] = 0

    def push(self, frame: np.ndarray) -> np.ndarray:
        """Append one frame; returns the whole stack ``uint8[depth, H, W, C]``."""
        if frame.shape != self.shape:
            raise ValueError(f"frame shape {frame.shape} != stack shape {self.shape}")
        if self.depth == 1:
            self._frames[0] = frame
        else:
            self._frames[:-1] = self._frames[1:]
            self._frames[-1] = frame
        return self._frames

    @property
    def current(self) -> np.ndarray:
        return self._frames


def frames_to_batch(stack: np.ndarray, device=None):
    """``uint8[B or (depth,H,W,3)]`` -> ``float32[B, depth, H, W, 3] / 255``.

    Kept as a function (not a transform object) so training and inference must
    call the same code - the v1.0.2 post-mortem has a bug caused by two copies of
    the preprocessing drifting apart.
    """
    import torch

    array = np.asarray(stack)
    if array.ndim == 4:  # (depth, H, W, C) -> (1, depth, H, W, C)
        array = array[None, ...]
    tensor = torch.as_tensor(array, dtype=torch.float32) / 255.0
    if device is not None:
        tensor = tensor.to(device)
    return tensor
