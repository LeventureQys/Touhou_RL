#!/usr/bin/env python
"""S3-8 foreground/window diagnostic (read-only, one game boot, no menu navigation).

Attempt 1 of the startup smoke failed at ``menu.select_difficulty`` with
``screen 4 did not become ready`` and the session log said
``warning: could not take foreground for SendInput``.  The desktop itself is
interactive (WinSta0/Default, ``OpenInputDesktop`` succeeds, not locked), so this
script boots the game once and records *why* the foreground transfer fails:

* every top-level window of the game process (class/title/visible/iconic/cloaked/rect),
* the foreground window before and after each attempt,
* the SPI_GETFOREGROUNDLOCKTIMEOUT value,
* five ``winapi.force_foreground`` attempts with their return values,
* a GDI capture of the client area (mean/std) to prove the frame is live,
* an alternative path: ``ShowWindow(SW_SHOWNOACTIVATE)`` + ``SetWindowPos`` +
  ``SetForegroundWindow`` from a thread attached to the target window only.

No key is ever injected here, so nothing is played and nothing outside this
session is touched.  The game instance is the one this script starts and closes.
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

from touhou_rl import winapi  # noqa: E402
from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir  # noqa: E402

user32 = ctypes.WinDLL("user32", use_last_error=True)
EnumWindowsProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


def window_rows(pid: int) -> list[dict]:
    rows: list[dict] = []

    def callback(hwnd, _lparam):
        owner = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if int(owner.value) != int(pid):
            return True
        rows.append(
            {
                "hwnd": f"0x{int(hwnd):X}",
                "title": winapi.window_title(hwnd),
                "class": winapi.class_name(hwnd),
                "visible": bool(user32.IsWindowVisible(hwnd)),
                "iconic": bool(winapi.is_iconic(hwnd)),
                "cloaked": bool(winapi.is_cloaked(hwnd)),
                "rect": list(winapi.client_rect(hwnd)),
                "thread": int(owner.value),
            }
        )
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return rows


def foreground() -> dict:
    hwnd = user32.GetForegroundWindow()
    return {
        "hwnd": f"0x{int(hwnd):X}" if hwnd else "0x0",
        "title": winapi.window_title(hwnd) if hwnd else "",
        "class": winapi.class_name(hwnd) if hwnd else "",
    }


def lock_timeout() -> int | None:
    value = wt.DWORD()
    ok = user32.SystemParametersInfoW(0x2000, 0, ctypes.byref(value), 0)  # SPI_GETFOREGROUNDLOCKTIMEOUT
    return int(value.value) if ok else None


def main(argv: list[str] | None = None) -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("foreground-diagnostic.json")
    run_dir = ROOT / "third_party/th08/build/rl/eval/v104-s38-foreground-diag"
    prepare_run_dir(run_dir, ROOT / "third_party/th08/build/play", windowed=True)
    session = GameSession(SessionOptions(run_dir=run_dir, log_to_file=True))
    report: dict = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "foreground_lock_timeout_ms": lock_timeout(),
        "foreground_before_launch": foreground(),
        "attempts": [],
        "session_log": str((run_dir / "rl-session.log").relative_to(ROOT)).replace("\\", "/"),
    }
    started = time.perf_counter()
    try:
        session.start()
        report["boot_seconds"] = round(time.perf_counter() - started, 2)
        report["pid"] = session.pid
        report["hwnd"] = f"0x{session.hwnd:X}"
        report["client_size"] = list(session.client_size)
        report["input_backend"] = type(session.keyboard).__name__
        report["windows"] = window_rows(session.pid)
        target = session.hwnd
        for attempt in range(1, 6):
            before = foreground()
            ok = winapi.force_foreground(target)
            time.sleep(0.4)
            report["attempts"].append(
                {
                    "attempt": attempt,
                    "force_foreground": bool(ok),
                    "foreground_before": before,
                    "foreground_after": foreground(),
                    "is_window": bool(user32.IsWindow(target)),
                    "visible": bool(user32.IsWindowVisible(target)),
                    "iconic": bool(winapi.is_iconic(target)),
                    "cloaked": bool(winapi.is_cloaked(target)),
                    "last_error": int(ctypes.get_last_error()),
                }
            )
            if ok:
                break
        # alternative path: attach only to the target thread, show without activating,
        # raise it, then set foreground and focus.
        target_thread = user32.GetWindowThreadProcessId(target, None)
        current = ctypes.windll.kernel32.GetCurrentThreadId()
        attached = bool(user32.AttachThreadInput(current, target_thread, True))
        user32.ShowWindow(target, 4)  # SW_SHOWNOACTIVATE
        user32.SetWindowPos(target, 0, 0, 0, 0, 0, 0x0001 | 0x0002)  # SWP_NOSIZE|SWP_NOMOVE
        user32.BringWindowToTop(target)
        user32.SetForegroundWindow(target)
        user32.SetFocus(target)
        if attached:
            user32.AttachThreadInput(current, target_thread, False)
        time.sleep(0.4)
        report["alternative"] = {
            "attached": attached,
            "foreground": foreground(),
            "success": foreground()["hwnd"] == f"0x{target:X}",
        }
        frame = session.grab()
        report["capture"] = {
            "shape": list(frame.shape),
            "mean": round(float(frame.mean()), 3),
            "std": round(float(frame.std()), 3),
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        try:
            session.close()
        except Exception as exc:  # noqa: BLE001
            report["close_error"] = repr(exc)
        report["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
        report["session_log_tail"] = (
            (run_dir / "rl-session.log").read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            if (run_dir / "rl-session.log").is_file()
            else []
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("boot_seconds", "hwnd", "input_backend", "foreground_lock_timeout_ms", "alternative", "error")}, ensure_ascii=False))
    print(json.dumps(report.get("attempts"), ensure_ascii=False))
    print(f"written: {out}")
    return 0 if report.get("alternative", {}).get("success") else 3


if __name__ == "__main__":
    raise SystemExit(main())
