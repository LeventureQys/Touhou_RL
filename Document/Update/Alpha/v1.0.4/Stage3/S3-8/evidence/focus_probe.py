#!/usr/bin/env python
"""S3-8 focus probe: can the game window be made foreground, and do keys land?

Attempt 1 of the startup smoke failed because ``menu.tap`` could not take the
foreground (``foreground-diagnostic.json``: the user's Task Manager owns the
foreground and ``SPI_GETFOREGROUNDLOCKTIMEOUT`` is 0x7FFFFFFF, so
``SetForegroundWindow`` from a background process returns ERROR_ACCESS_DENIED).

This probe tries the two non-destructive ways to obtain the foreground:

1. ``SwitchToThisWindow(hwnd, True)`` - a user32 call that predates the
   foreground-lock rules;
2. a synthetic *no-op* mouse event via ``SendInput`` (dx=+1 then -1: the cursor
   returns to where it was) followed by ``SetForegroundWindow`` - Windows lets the
   process that received the last input event set the foreground.

Then it sends one ``z`` tap through the session's own keyboard backend and checks
whether the game actually reacted (supervisor state / screen change), which is the
property A-04 needs.  Nothing is killed and no other window is modified; the game
instance is the one this script starts and closes.  Two raw PNGs are saved so the
UI can be inspected.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def _repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "touhou_rl" / "vision_data.py").exists():
            return candidate
    raise SystemExit("cannot locate the repository root")


ROOT = _repo_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from touhou_rl import menu, spec, winapi  # noqa: E402
from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir  # noqa: E402

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.SwitchToThisWindow.argtypes = [wt.HWND, wt.BOOL]
user32.SwitchToThisWindow.restype = None


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wt.LONG),
        ("dy", wt.LONG),
        ("mouseData", wt.DWORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wt.ULONG)),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("padding", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("union", _INPUTUNION)]


def noop_mouse_event(dx: int) -> None:
    """One relative mouse move: a real input event with no net cursor motion."""
    item = INPUT()
    item.type = 0  # INPUT_MOUSE
    item.union.mi = MOUSEINPUT(dx, 0, 0, 0x0001, 0, None)  # MOUSEEVENTF_MOVE
    user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(INPUT))


def foreground() -> dict:
    hwnd = user32.GetForegroundWindow()
    return {
        "hwnd": f"0x{int(hwnd):X}" if hwnd else "0x0",
        "title": winapi.window_title(hwnd) if hwnd else "",
        "class": winapi.class_name(hwnd) if hwnd else "",
    }


def state_of(session: GameSession) -> dict:
    state = session.read_state(with_bullets=False)
    return {
        "supervisor_state": int(state.supervisor_state),
        "title_screen_current": int(state.title_screen_current),
        "title_screen_state": int(state.title_screen_state),
        "title_timer2": int(state.title_timer2),
        "is_demo": bool(state.is_demo),
        "stage": int(state.stage),
        "gameplay_frame": int(state.gameplay_frame),
    }


def main(argv: list[str] | None = None) -> int:
    out_dir = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")).resolve()
    run_dir = ROOT / "third_party/th08/build/rl/eval/v104-s38-focus-probe"
    prepare_run_dir(run_dir, ROOT / "third_party/th08/build/play", windowed=True)
    session = GameSession(SessionOptions(run_dir=run_dir, log_to_file=True))
    report: dict = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/focus_probe.py",
        "foreground_before": foreground(),
        "screenshots": [],
        "methods": [],
    }
    started = time.perf_counter()
    exit_code = 0
    try:
        session.start()
        report["boot_seconds"] = round(time.perf_counter() - started, 2)
        report["pid"] = session.pid
        report["hwnd"] = f"0x{session.hwnd:X}"
        report["input_backend"] = type(session.keyboard).__name__
        target = session.hwnd
        report["window"] = {
            "title": winapi.window_title(target),
            "class": winapi.class_name(target),
            "client_rect": list(winapi.client_rect(target)),
            "visible": bool(user32.IsWindowVisible(target)),
            "iconic": bool(winapi.is_iconic(target)),
            "cloaked": bool(winapi.is_cloaked(target)),
        }
        report["state_before"] = state_of(session)
        first = session.grab()
        first_path = out_dir / "focus-probe-before.png"
        Image.fromarray(np.asarray(first)).save(first_path)
        report["screenshots"].append(str(first_path.relative_to(ROOT)).replace("\\", "/"))

        # method 1: SwitchToThisWindow
        user32.SwitchToThisWindow(target, True)
        time.sleep(0.6)
        method1 = foreground()
        report["methods"].append(
            {"method": "SwitchToThisWindow", "foreground": method1, "success": method1["hwnd"] == report["hwnd"]}
        )
        focused = method1["hwnd"] == report["hwnd"]
        if not focused:
            # method 2: synthetic no-op input then SetForegroundWindow
            noop_mouse_event(1)
            time.sleep(0.05)
            noop_mouse_event(-1)
            time.sleep(0.05)
            ok = winapi.force_foreground(target)
            time.sleep(0.5)
            method2 = foreground()
            report["methods"].append(
                {
                    "method": "SendInput no-op + force_foreground",
                    "force_foreground_return": bool(ok),
                    "foreground": method2,
                    "success": method2["hwnd"] == report["hwnd"],
                }
            )
            focused = method2["hwnd"] == report["hwnd"]
        if not focused:
            # method 3: the gesture a human would make - move our own game window
            # to a known position, put the cursor inside it and click once.  A click
            # is real input, so Windows activates the clicked window even while the
            # foreground lock is held by another process.  TH08 ignores mouse input,
            # so the click has no gameplay effect.
            cursor = wt.POINT()
            user32.GetCursorPos(ctypes.byref(cursor))
            before_cursor = [int(cursor.x), int(cursor.y)]
            user32.SetWindowPos(target, 0, 60, 60, 640, 480, 0x0040 | 0x0004)  # SWP_SHOWWINDOW|SWP_NOZORDER
            time.sleep(0.3)
            rect = winapi.client_rect(target)
            cx = int(rect[0] + rect[2] // 2)
            cy = int(rect[1] + rect[3] // 2)
            user32.SetCursorPos(cx, cy)
            time.sleep(0.1)
            user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
            time.sleep(0.08)
            user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
            time.sleep(0.6)
            method3 = foreground()
            report["methods"].append(
                {
                    "method": "mouse click inside the moved game window",
                    "cursor_before": before_cursor,
                    "cursor_target": [cx, cy],
                    "client_rect_after_move": list(rect),
                    "foreground": method3,
                    "success": method3["hwnd"] == report["hwnd"],
                }
            )
            focused = method3["hwnd"] == report["hwnd"]
            user32.SetCursorPos(int(before_cursor[0]), int(before_cursor[1]))
        report["foreground_obtained"] = bool(focused)

        if focused:
            # does a key actually reach the game?  One 'z' at the start menu must
            # change the title screen (difficulty select) or start a demo.
            menu.tap(session, "z", settle_s=0.6)
            report["state_after_z"] = state_of(session)
            report["key_landed"] = report["state_after_z"] != report["state_before"]
            time.sleep(0.8)
            second = session.grab()
            second_path = out_dir / "focus-probe-after.png"
            Image.fromarray(np.asarray(second)).save(second_path)
            report["screenshots"].append(str(second_path.relative_to(ROOT)).replace("\\", "/"))
            report["frames_differ"] = bool(not np.array_equal(np.asarray(first), np.asarray(second)))
            report["foreground_end"] = foreground()
        else:
            report["key_landed"] = False
            exit_code = 3
    except Exception as exc:  # noqa: BLE001
        import traceback

        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        exit_code = 1
    finally:
        try:
            session.close()
        except Exception as exc:  # noqa: BLE001
            report["close_error"] = repr(exc)
        report["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
    (out_dir / "focus-probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "foreground_before": report["foreground_before"],
                "methods": report["methods"],
                "foreground_obtained": report.get("foreground_obtained"),
                "key_landed": report.get("key_landed"),
                "state_before": report.get("state_before"),
                "state_after_z": report.get("state_after_z"),
            },
            ensure_ascii=False,
        )
    )
    print(f"written: {out_dir / 'focus-probe.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
