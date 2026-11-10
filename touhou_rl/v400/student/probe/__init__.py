"""T2a perception probe: which playfield resolution can see 48 px around the ship.

The package exists to answer one question with one number (EXP-P3-010):

    given a playfield picture at resolution X, can a small fully-convolutional
    head say which 4-px cells within 48 px of the player hold a lethal object,
    with F1 >= 0.90 ?

Everything here is *additive*.  It reads ``obs_spec`` (obs-v1) and the frozen
P2 label generators, and it rewrites none of them:

* :mod:`touhou_rl.v400.student.probe.native` -- 640x480 RGBA readback ->
  native 384x448 grayscale playfield, plus the exact obs-v1 block mean;
* :mod:`touhou_rl.v400.student.probe.labels` -- the 24x24 cell window around
  the player, the future-8 danger map, and the union of lethal families;
* :mod:`touhou_rl.v400.student.probe.inputs` -- the four pre-registered input
  tensors A/B/C/D built from one native frame;
* :mod:`touhou_rl.v400.student.probe.net` -- the shared fully-convolutional
  head (no global pooling anywhere), the pixel player locator, and the F1
  bookkeeping;
* :mod:`touhou_rl.v400.student.probe.collect` -- replay D0's recorded per-frame
  masks on an env-role engine and re-render the native field, gated on HASH.

The probe never feeds privileged state to a model: the player position is used
to *place the label window* (and, for input D, the crop anchor), and the pixel
locator's error is reported separately so D cannot be selected on a hidden
advantage (EXP-P3-010 0.5).
"""

from __future__ import annotations

from touhou_rl.v400.student.probe.inputs import (
    INPUT_AREAS,
    INPUT_SPECS,
    InputSpec,
    build_input,
    input_pixels,
    smallest_first,
)
from touhou_rl.v400.student.probe.labels import (
    CELL_PX,
    WINDOW_CELLS,
    WINDOW_PX,
    WindowLabel,
    window_label,
)
from touhou_rl.v400.student.probe.native import (
    PLAYFIELD_HEIGHT,
    PLAYFIELD_WIDTH,
    encode_native,
    gray_playfield,
    obs_v1_play64,
)

__all__ = [
    "CELL_PX",
    "INPUT_AREAS",
    "INPUT_SPECS",
    "InputSpec",
    "PLAYFIELD_HEIGHT",
    "PLAYFIELD_WIDTH",
    "WINDOW_CELLS",
    "WINDOW_PX",
    "WindowLabel",
    "build_input",
    "encode_native",
    "gray_playfield",
    "input_pixels",
    "obs_v1_play64",
    "smallest_first",
    "window_label",
]
