"""Pixel front-end tests (D11 prototype)."""

from __future__ import annotations

import numpy as np

from touhou_rl.escape import screen


def _frame(height=480, width=640):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (60, 60, 70)  # desktop-ish grey
    frame[40:440, 40:424] = (0, 0, 0)  # playfield-ish dark rectangle
    return frame


def _dot(frame, x, y, colour=(255, 255, 255), radius=2):
    frame[y - radius : y + radius + 1, x - radius : x + radius + 1] = colour


def test_bullet_mask_finds_bright_low_saturation_pixels():
    playfield = np.zeros((60, 60, 3), dtype=np.uint8)
    _dot(playfield, 20, 20, (255, 255, 255))
    _dot(playfield, 40, 40, (255, 40, 40))  # saturated red is not a bullet core
    mask = screen.bullet_mask(playfield)
    assert mask[18:23, 18:23].any()
    assert not mask[38:43, 38:43].any()


def test_player_mask_requires_strong_red():
    playfield = np.zeros((40, 40, 3), dtype=np.uint8)
    _dot(playfield, 10, 10, (240, 30, 30))
    _dot(playfield, 30, 30, (255, 255, 255))
    mask = screen.player_mask(playfield)
    assert mask[8:13, 8:13].any()
    assert not mask[28:33, 28:33].any()


def test_blobs_report_centroid_and_half_extents():
    playfield = np.zeros((30, 30, 3), dtype=np.uint8)
    _dot(playfield, 15, 15, (255, 255, 255), radius=2)
    found = screen.blobs(screen.bullet_mask(playfield), minimum_px=3, maximum_px=200)
    assert len(found) == 1
    assert abs(found[0]["x"] - 15) < 0.6
    assert abs(found[0]["y"] - 15) < 0.6
    assert found[0]["half_x"] >= 1.5


def test_extract_reports_pixel_coordinates_and_refuses_to_claim_calibration():
    frame = _frame()
    _dot(frame, 100, 120, (255, 255, 255), radius=2)
    _dot(frame, 300, 300, (240, 30, 30), radius=2)
    box = screen.PlayfieldBox(left=40, top=40, width=384, height=400)
    summary = screen.extract(frame, box=box)
    assert summary["calibrated"] is False
    assert summary["box"] == [40, 40, 424, 440]
    assert summary["bullet_blobs"] >= 1
    assert summary["player_blobs"] >= 1
    player = summary["player"]
    assert abs(player["x"] - (300 - 40)) < 0.6
    assert abs(player["y"] - (300 - 40)) < 0.6


def test_detect_playfield_rejects_a_menu_like_frame():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)  # bright, no dark field
    try:
        screen.detect_playfield(frame)
    except ValueError as error:
        assert "playfield not found" in str(error)
    else:  # pragma: no cover - must raise
        raise AssertionError("a frame without a dark field must be rejected")
