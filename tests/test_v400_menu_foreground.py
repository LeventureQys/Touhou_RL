"""Unit checks for the WP8 foreground/input fix in ``touhou_rl.menu``.

Nothing here starts the game: these tests pin the *decision logic* the real run
depends on, because the failure they guard against is silent.

Background (EXP-P0-053): a ``dontUseDirectInput`` session gets a
``PostMessageKeyboard``, and TH08 on Windows reads ``GetKeyboardState``
(``Global.cpp:731``) while its message loop handles no ``WM_KEYDOWN`` at all
(``main.cpp:531`` handles only ``WM_ACTIVATEAPP``).  Posted keys therefore never
reach the engine, and because ``menu.tap`` only foregrounded the window for the
``SendInput`` backend, ``--mode practice`` sat in ``goto_start_menu_item`` until
its 5 s budget expired with the cursor frozen at item 0.
"""
from __future__ import annotations

import pytest

from touhou_rl import menu, spec, winapi


class FakeSession:
    """Minimal ``GameSession`` stand-in with a *modelled* Start-menu cursor."""

    def __init__(self, keyboard, cursor=0, screen=spec.SCREEN_START_MENU):
        self.keyboard = keyboard
        self.hwnd = 0x1234
        self.log: list[str] = []
        self.cursor = cursor
        self.screen = screen
        self.frozen = False

    def refresh_window(self) -> int:
        return self.hwnd

    def log_line(self, message: str) -> None:
        self.log.append(message)

    def read_state(self, with_bullets: bool = True, with_title: bool = True):
        class _State:
            pass

        state = _State()
        state.title_cursor = self.cursor
        state.title_screen_current = self.screen
        return state

    def press(self, key: str) -> None:
        if self.frozen:
            return
        self.cursor = (self.cursor + 1) % 9 if key == "down" else (self.cursor - 1) % 9


@pytest.fixture()
def patched_keyboards(monkeypatch):
    """Record taps instead of touching the real desktop."""
    sent: list[tuple[str, str]] = []
    released: list[str] = []

    def post_tap(self, name, hold_s=0.0):
        sent.append(("postmessage", name))

    def send_tap(self, name, hold_s=0.0):
        sent.append(("sendinput", name))

    def post_release(self):
        released.append("postmessage")

    monkeypatch.setattr(winapi.PostMessageKeyboard, "tap", post_tap)
    monkeypatch.setattr(winapi.SendInputKeyboard, "tap", send_tap)
    monkeypatch.setattr(winapi.PostMessageKeyboard, "release_all", post_release)
    monkeypatch.setattr(winapi.SendInputKeyboard, "release_all", lambda self: None)
    monkeypatch.setattr(winapi, "input_window_state", lambda hwnd: f"hwnd=0x{hwnd:X}")
    monkeypatch.setattr(winapi, "foreground_summary", lambda: "foreground=<none>")
    return sent, released


def test_tap_upgrades_postmessage_to_sendinput_when_foreground(monkeypatch, patched_keyboards):
    sent, released = patched_keyboards
    monkeypatch.setattr(winapi, "force_foreground", lambda hwnd, *a, **k: True)
    session = FakeSession(winapi.PostMessageKeyboard(hwnd=0x1234))

    menu.tap(session, "down", hold_s=0.0, settle_s=0.0)

    assert isinstance(session.keyboard, winapi.SendInputKeyboard)
    assert released == ["postmessage"], "held posted keys must be released before the switch"
    assert sent == [("sendinput", "down")], "the tap must go through the path the engine reads"
    assert any("upgraded PostMessage -> SendInput" in line for line in session.log)


def test_tap_keeps_postmessage_and_warns_when_foreground_is_unavailable(monkeypatch, patched_keyboards):
    sent, _released = patched_keyboards
    monkeypatch.setattr(winapi, "force_foreground", lambda hwnd, *a, **k: False)
    session = FakeSession(winapi.PostMessageKeyboard(hwnd=0x1234))

    menu.tap(session, "down", hold_s=0.0, settle_s=0.0)

    assert isinstance(session.keyboard, winapi.PostMessageKeyboard)
    assert sent == [("postmessage", "down")]
    assert any("could not take foreground" in line for line in session.log)


def test_tap_never_downgrades_a_sendinput_session(monkeypatch, patched_keyboards):
    sent, _released = patched_keyboards
    monkeypatch.setattr(winapi, "force_foreground", lambda hwnd, *a, **k: True)
    keyboard = winapi.SendInputKeyboard()
    session = FakeSession(keyboard)

    menu.tap(session, "z", hold_s=0.0, settle_s=0.0)

    assert session.keyboard is keyboard
    assert sent == [("sendinput", "z")]
    assert session.log == []


def test_goto_start_menu_item_walks_down_to_practice(monkeypatch, patched_keyboards):
    sent, _released = patched_keyboards
    monkeypatch.setattr(winapi, "force_foreground", lambda hwnd, *a, **k: True)
    session = FakeSession(winapi.PostMessageKeyboard(hwnd=0x1234))

    def _tap(self, name, hold_s=menu.DEFAULT_TAP_HOLD_S, settle_s=menu.DEFAULT_SETTLE_S):
        sent.append(("sendinput", name))
        session.press(name)

    monkeypatch.setattr(winapi.SendInputKeyboard, "tap", _tap)
    menu.goto_start_menu_item(session, spec.MENU_ITEM_PRACTICE_START, timeout=3.0)

    assert session.cursor == spec.MENU_ITEM_PRACTICE_START
    assert [name for _kind, name in sent] == ["down", "down", "down"]


def test_goto_start_menu_item_wraps_upwards(monkeypatch, patched_keyboards):
    sent, _released = patched_keyboards
    monkeypatch.setattr(winapi, "force_foreground", lambda hwnd, *a, **k: True)
    session = FakeSession(winapi.SendInputKeyboard())

    def _tap(self, name, hold_s=menu.DEFAULT_TAP_HOLD_S, settle_s=menu.DEFAULT_SETTLE_S):
        sent.append(("sendinput", name))
        session.press(name)

    monkeypatch.setattr(winapi.SendInputKeyboard, "tap", _tap)
    menu.goto_start_menu_item(session, spec.MENU_ITEM_QUIT, timeout=3.0)

    assert session.cursor == spec.MENU_ITEM_QUIT
    assert [name for _kind, name in sent] == ["up"]


def test_goto_start_menu_item_times_out_with_input_diagnostics(monkeypatch, patched_keyboards):
    monkeypatch.setattr(winapi, "force_foreground", lambda hwnd, *a, **k: False)
    session = FakeSession(winapi.PostMessageKeyboard(hwnd=0x1234))
    session.frozen = True

    with pytest.raises(TimeoutError) as excinfo:
        menu.goto_start_menu_item(session, spec.MENU_ITEM_PRACTICE_START, timeout=0.05)

    message = str(excinfo.value)
    assert "stuck at 0" in message
    assert "PostMessageKeyboard" in message
    assert "hwnd=0x1234" in message


def test_goto_start_menu_item_rejects_other_screens(patched_keyboards):
    session = FakeSession(winapi.SendInputKeyboard(), screen=spec.SCREEN_DIFFICULTY_SELECT)
    with pytest.raises(RuntimeError):
        menu.goto_start_menu_item(session, spec.MENU_ITEM_START, timeout=0.05)


def test_force_foreground_is_idempotent_when_already_foreground(monkeypatch):
    """The verification must not report failure for a window that is foreground."""
    monkeypatch.setattr(winapi.user32, "IsWindow", lambda hwnd: True)
    monkeypatch.setattr(winapi.user32, "GetForegroundWindow", lambda: 0x1234)
    assert winapi.force_foreground(0x1234) is True
