"""Diagnostic: enumerate every top-level window owned by a freshly launched th08.exe."""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from ctypes import wintypes as wt
from pathlib import Path

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM), wt.LPARAM]
user32.GetWindowTextLengthW.argtypes = [wt.HWND]
user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.IsWindowVisible.argtypes = [wt.HWND]
user32.IsWindowVisible.restype = wt.BOOL

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "third_party/th08/build/play").resolve()
proc = subprocess.Popen([str(run_dir / "th08.exe")], cwd=str(run_dir))
print("pid", proc.pid)
try:
    for tick in range(20):
        time.sleep(0.5)
        rows = []

        @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        def _enum(hwnd, _lp):
            owner = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value != proc.pid:
                return True
            title_len = user32.GetWindowTextLengthW(hwnd)
            title = ctypes.create_unicode_buffer(title_len + 1)
            user32.GetWindowTextW(hwnd, title, title_len + 1)
            cls = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls, 256)
            rows.append(
                (
                    int(hwnd),
                    cls.value,
                    title.value.strip().encode("unicode_escape").decode("ascii"),
                    [hex(ord(ch)) for ch in title.value[:8]],
                    bool(user32.IsWindowVisible(hwnd)),
                )
            )
            return True

        user32.EnumWindows(_enum, 0)
        print(f"t={0.5 * (tick + 1):4.1f}s alive={proc.poll() is None} windows={rows}")
        if rows:
            break
finally:
    if proc.poll() is None:
        proc.kill()
        print("killed")
