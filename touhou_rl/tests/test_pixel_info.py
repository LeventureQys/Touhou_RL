"""Unit tests for the round27 pixel-information probe — no game, no GPU.

The probe's own controls are what caught two invalid executions (a position
confound and a wrong label encoding), so they are exactly what must stay tested:
the reader must work on a hand-built image, must fail without the preview band,
and must fail when the band is occluded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools" / "rl") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "rl"))

import v373_pixel_info as probe  # noqa: E402

PLAYFIELD = (
    probe.PLAYFIELD_TOP,
    probe.PLAYFIELD_TOP + probe.PLAYFIELD_HEIGHT,
    probe.PLAYFIELD_LEFT,
    probe.PLAYFIELD_LEFT + probe.PLAYFIELD_WIDTH,
)


def _canvas() -> np.ndarray:
    image = np.full((480, 640, 3), (18, 20, 28), dtype=np.uint8)
    top, bottom, left, right = PLAYFIELD
    image[top:bottom, left:right, :] = probe.BACKGROUND_COLOUR
    return image


def _paint_band(image, span_left, span_right, gap_left, gap_right):
    """Mirror ``v373_telegraph.preview_columns`` exactly.

    The band covers the spans; the two edge markers sit *outside* the gap,
    ``[gap_left-4, gap_left-1]`` and ``[gap_right+1, gap_right+4]``.
    """
    top, bottom, left, _right = PLAYFIELD
    image[top:bottom, left + span_left : left + span_right, :] = probe.FILL_COLOUR
    image[top:bottom, left + gap_left - 4 : left + gap_left, :] = probe.EDGE_COLOUR
    image[top:bottom, left + gap_right + 1 : left + gap_right + 5, :] = probe.EDGE_COLOUR


def _paint_player(image, x):
    top, bottom, left, _right = PLAYFIELD
    image[bottom - 32 : bottom - 26, left + x - 5 : left + x + 6, :] = probe.PLAYER_COLOUR


def test_clusters_group_consecutive_columns():
    assert probe._clusters([1, 2, 3, 9, 10]) == [[1, 2, 3], [9, 10]]
    assert probe._clusters([]) == []


def test_reader_reads_gap_on_the_right():
    image = _canvas()
    _paint_band(image, 0, 200, 100, 119)
    _paint_player(image, 60)
    reading = probe.read_direction(image)
    assert reading["ok"] is True
    assert reading["gapLeft"] == 100 and reading["gapRight"] == 119
    assert reading["gapCenter"] == 109.5
    assert reading["playerX"] == 60.0
    assert reading["direction"] == 1


def test_reader_reads_gap_on_the_left():
    image = _canvas()
    _paint_band(image, 120, 383, 200, 219)
    _paint_player(image, 300)
    reading = probe.read_direction(image)
    assert reading["ok"] is True
    assert reading["gapCenter"] == 209.5
    assert reading["playerX"] == 300.0
    assert reading["direction"] == -1


def test_reader_fails_without_the_preview_band():
    image = _canvas()
    _paint_player(image, 200)
    reading = probe.read_direction(image)
    assert reading["ok"] is False
    assert reading["reason"] == "no preview band"


def test_reader_fails_when_the_band_is_occluded():
    image = _canvas()
    _paint_band(image, 0, 200, 100, 119)
    _paint_player(image, 60)
    assert probe.read_direction(image)["ok"] is True
    painted = np.all(image == probe.FILL_COLOUR, axis=-1) | np.all(
        image == probe.EDGE_COLOUR, axis=-1
    )
    image[painted] = probe.BACKGROUND_COLOUR
    assert probe.read_direction(image)["ok"] is False


def test_player_x_baseline_is_chance_when_positions_are_balanced():
    # Same x, both labels, equal counts: position cannot predict the label.
    worlds = []
    for x in (120.0, 160.0, 200.0, 240.0):
        for label in (-1, 1):
            worlds.append({"playerX": x, "label": label})
    baseline = probe.player_x_only_baseline(worlds)
    assert baseline["balanced"] is True
    assert 0.4 <= baseline["accuracy"] <= 0.6


def test_decision_frame_stays_inside_the_telegraph_window():
    class World:
        wall_frame = 76

    frame = probe.decision_frame(World())
    assert frame % 2 == 0
    assert World.wall_frame - probe.TELEGRAPH_LEAD <= frame < World.wall_frame
