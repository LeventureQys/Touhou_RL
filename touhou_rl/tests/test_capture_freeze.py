"""Tests for the capture freeze guard.

Why this exists
---------------
v1.0.3's vision dataset contained 52,888 identical frames (73% of it) because the
window DC stopped updating when the session locked while the game kept running.
`GetDC` + `BitBlt` still returns success in that state, so nothing in the old
recorder could tell a stale frame from a still scene.  These tests pin the guard
that closes that hole.
"""

from __future__ import annotations

import pytest

from touhou_rl import winapi
from touhou_rl.vision_capture import FrameStability, FrozenCapture


def _digest(value: int) -> bytes:
    return bytes([value]) * 16


def test_identical_frames_abort_at_the_limit():
    guard = FrameStability(limit=5)
    for _ in range(4):
        guard.update(_digest(1))
    guard.check()  # a run of 4 is below the limit
    guard.update(_digest(1))
    with pytest.raises(FrozenCapture) as exc:
        guard.check()
    assert "5 identical frames" in str(exc.value)
    assert guard.frozen is True
    assert guard.longest == 5


def test_changing_frames_never_abort():
    guard = FrameStability(limit=3)
    for value in range(50):
        guard.update(_digest(value % 7))
    guard.check()
    assert guard.frozen is False
    assert guard.longest == 1


def test_a_single_repeat_is_not_a_freeze():
    guard = FrameStability(limit=5)
    guard.update(_digest(1))
    guard.update(_digest(1))  # one repeat
    guard.update(_digest(2))
    guard.check()
    assert guard.longest == 2
    assert guard.frozen is False


def test_limit_zero_disables_the_guard():
    guard = FrameStability(limit=0)
    for _ in range(100):
        guard.update(_digest(9))
    guard.check()
    assert guard.frozen is False
    assert guard.summary()["freeze_limit"] == 0


def test_frozen_frames_keep_counting_after_the_abort_point():
    guard = FrameStability(limit=2)
    for _ in range(6):
        guard.update(_digest(4))
    assert guard.frozen_frames == 5  # every frame from the 2nd on
    assert guard.longest == 6
    assert guard.run_length == 6


def test_summary_reports_the_fields_the_capture_report_needs():
    guard = FrameStability(limit=4)
    for _ in range(4):
        guard.update(_digest(0))
    summary = guard.summary()
    assert summary == {"freeze_limit": 4, "frozen_frames": 1, "longest_identical_run": 4}


def test_keep_awake_sets_display_and_system_flags(monkeypatch):
    seen: list[int] = []

    def fake(state: int) -> int:
        seen.append(state)
        return 1

    monkeypatch.setattr(winapi.kernel32, "SetThreadExecutionState", fake)
    assert winapi.keep_awake(True) is True
    assert seen[0] == (winapi.ES_CONTINUOUS | winapi.ES_SYSTEM_REQUIRED | winapi.ES_DISPLAY_REQUIRED)
    assert winapi.keep_awake(False) is True
    assert seen[1] == winapi.ES_CONTINUOUS


def test_window_visible_rejects_minimised_and_cloaked(monkeypatch):
    monkeypatch.setattr(winapi, "is_iconic", lambda hwnd: True)
    monkeypatch.setattr(winapi, "is_cloaked", lambda hwnd: False)
    assert winapi.window_visible(1) is False

    monkeypatch.setattr(winapi, "is_iconic", lambda hwnd: False)
    monkeypatch.setattr(winapi, "is_cloaked", lambda hwnd: True)
    assert winapi.window_visible(1) is False

    monkeypatch.setattr(winapi, "is_cloaked", lambda hwnd: False)
    assert winapi.window_visible(1) is True
