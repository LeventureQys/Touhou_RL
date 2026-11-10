"""Frozen constants for the P2 label generator (12-P2 §2.2, §3.2, §3.3).

Everything here is traceable to engine source or to the P2 specification; none
of it is fitted to the data:

* :data:`PLAYFIELD_WIDTH` / :data:`PLAYFIELD_HEIGHT` / :data:`SCREEN_ORIGIN` —
  the playfield viewport.  ``src/Background.cpp:740-743`` sets
  ``viewport = (32, 16, 384, 448)``, so a playfield coordinate becomes a screen
  coordinate by adding ``(32, 16)``.  :func:`playfield_to_screen` is the only
  place that addition happens.
* :data:`GRID_CELL` — P2 §3.3 asks for the lethal occupancy at native
  resolution, downsampled to 1/4, i.e. 384/4 x 448/4 = :data:`GRID_WIDTH` x
  :data:`GRID_HEIGHT` cells of 4x4 playfield pixels.
* :data:`HORIZON_FRAMES` — P2 §3.3's future-danger frames (t+4, t+8, t+16).
  The longest horizon is the one ``future_danger.would_die`` reports.
* :data:`HUD_LAYOUT` — HUD anchor positions and the ``%.9d`` score format from
  ``src/Gui.cpp:1297-1390`` (score (488,56); lives icons (488+16i,88); bombs
  (488+16i,104); power (488,136); graze (488,152); point (488,168); time
  (488,184)).
"""

from __future__ import annotations

from typing import Final

#: Bumped whenever a label's meaning changes; a dataset records the version it
#: was generated with so an audit can refuse to mix two meanings.
#: ``p2-s4-1`` is the EXP-P2-001 dataset (proxy events, ``build/rl_hazard``);
#: ``p2-v2-1`` is the EXP-P2-002 dataset (real ``EVENTS`` bits, ``build/rl_hazard2``).
LABEL_VERSION: Final = "p2-v2-1"

#: The previous label version, kept so an audit can name the old meaning.
LABEL_VERSION_V1: Final = "p2-s4-1"

#: Playfield size in playfield pixels (``Background.cpp:742-743``).
PLAYFIELD_WIDTH: Final = 384
PLAYFIELD_HEIGHT: Final = 448

#: Playfield origin in screen pixels (``Background.cpp:740-741``).
SCREEN_ORIGIN: Final[tuple[int, int]] = (32, 16)

#: Screen size of the readback frame (``rlharness.cpp`` ``PublishPixels``).
SCREEN_WIDTH: Final = 640
SCREEN_HEIGHT: Final = 480

#: P2 §3.3: rasterize at native playfield resolution, then downsample to 1/4.
GRID_CELL: Final = 4
GRID_WIDTH: Final = PLAYFIELD_WIDTH // GRID_CELL
GRID_HEIGHT: Final = PLAYFIELD_HEIGHT // GRID_CELL

#: P2 §3.3 future-danger horizons, plus the headline horizon.
HORIZON_FRAMES: Final[tuple[int, ...]] = (4, 8, 16)
HORIZON_HEADLINE: Final = 16

#: Distance channel saturation: the stored per-cell distance is a uint8 of
#: playfield pixels, clipped here.  255 px is more than half the playfield, so
#: "far" is unambiguous; the raw float value is kept as ``nearest_distance``.
DISTANCE_CLIP_PX: Final = 255

#: HUD rows: anchor (top-left of the text baseline cell, screen px), the band
#: used to detect redraws (x0, y0, x1, y1, screen px), and the text template.
#: Anchors and formats are read off ``src/Gui.cpp:1297-1390``.
HUD_LAYOUT: Final[dict[str, dict]] = {
    "score": {
        "x": 488, "y": 56,
        "format": "%.9d",
        "band": (480, 48, 640, 72),
        "source": "Gui.cpp:1298-1299 globals->displayScore",
    },
    "lives": {
        "x": 488, "y": 88, "icon_step": 16,
        "format": "%d",
        "band": (480, 80, 640, 96),
        "source": "Gui.cpp:1198-1203 frontVms[15] at (480, 88)",
    },
    "bombs": {
        "x": 488, "y": 104, "icon_step": 16,
        "format": "%d",
        "band": (480, 96, 640, 112),
        "source": "Gui.cpp:1204-1209 frontVms[15] at (480, 104)",
    },
    "power": {
        "x": 488, "y": 136,
        "format": "MAX" ,
        "band": (480, 128, 640, 150),
        "source": "Gui.cpp:1380-1390 GetPower() / MAX",
    },
    "graze": {
        "x": 488, "y": 152,
        "format": "%d",
        "band": (480, 148, 640, 166),
        "source": "Gui.cpp:1315-1319 globals->graze",
    },
    "point": {
        "x": 488, "y": 168,
        "format": "%d/%d",
        "band": (480, 164, 640, 182),
        "source": "Gui.cpp:1320-1329 pointItemsCollected / nextPointItemExtendThreshold",
    },
    "time": {
        "x": 488, "y": 184,
        "format": "%d/%d",
        "band": (480, 180, 640, 198),
        "source": "Gui.cpp:1330-1341 GetTimeOrbs() / lastSpellTimeOrbThreshold",
    },
}

#: P2 §3.2: HUD rows redraw only when their dirty counter is set
#: (``Gui.cpp:1166-1203``), so the picture can lag the memory.  The label is
#: the value the picture is expected to show once the measured lag is applied;
#: the lag itself is measured, not assumed, by ``audit.hud_lag``.
HUD_LAG_FRAMES_UNMEASURED: Final = None


def playfield_to_screen(
    x: float, y: float, origin: tuple[int, int] = SCREEN_ORIGIN
) -> tuple[float, float]:
    """Convert one playfield point to a screen point: ``p + (32, 16)``."""

    if not (isinstance(origin, tuple) and len(origin) == 2):
        raise ValueError(f"origin must be a 2-tuple, got {origin!r}")
    return (float(x) + float(origin[0]), float(y) + float(origin[1]))


def screen_to_playfield(
    x: float, y: float, origin: tuple[int, int] = SCREEN_ORIGIN
) -> tuple[float, float]:
    """Inverse of :func:`playfield_to_screen`."""

    if not (isinstance(origin, tuple) and len(origin) == 2):
        raise ValueError(f"origin must be a 2-tuple, got {origin!r}")
    return (float(x) - float(origin[0]), float(y) - float(origin[1]))


def playfield_rect_on_screen(
    origin: tuple[int, int] = SCREEN_ORIGIN,
) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` half-open screen rectangle the playfield occupies."""

    return (
        int(origin[0]),
        int(origin[1]),
        int(origin[0]) + PLAYFIELD_WIDTH,
        int(origin[1]) + PLAYFIELD_HEIGHT,
    )
