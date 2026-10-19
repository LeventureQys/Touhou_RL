"""Thin Win32 layer: window discovery, GDI capture, keyboard injection.

Design notes
------------
* ``find_window`` identifies the game by *process id + window class* (``BASE``).
  TH08 sets its title from a Shift-JIS ``ANSI`` string, which the Win32 ANSI
  entry points transcode to CP936 mojibake on this host, so title matching is
  unusable.
* Capture uses the window DC (``GetDC`` + ``BitBlt``).  Measured at 1.98 ms per
  640x480 frame and it keeps returning the real game frame while the window is
  occluded by other windows.
* Keyboard injection has two backends:
  - ``SendInputKeyboard``: scan-code ``SendInput``; the game's DirectInput
    keyboard uses ``DISCL_FOREGROUND`` so the window must be foreground.
  - ``PostMessageKeyboard``: posts ``WM_KEYDOWN``/``WM_KEYUP`` to the window.
    This only reaches the game when it reads input through ``GetKeyboardState``,
    i.e. when ``GameConfiguration.opts.dontUseDirectInput`` is set.  It does not
    require focus, which is what makes parallel game instances possible.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes as wt

from . import spec

# ---------------------------------------------------------------------------
# Libraries and prototypes
# ---------------------------------------------------------------------------
user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0
WM_CLOSE = 0x0010
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
SW_RESTORE = 9
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wt.DWORD),
        ("biWidth", wt.LONG),
        ("biHeight", wt.LONG),
        ("biPlanes", wt.WORD),
        ("biBitCount", wt.WORD),
        ("biCompression", wt.DWORD),
        ("biSizeImage", wt.DWORD),
        ("biXPelsPerMeter", wt.LONG),
        ("biYPelsPerMeter", wt.LONG),
        ("biClrUsed", wt.DWORD),
        ("biClrImportant", wt.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wt.DWORD * 3)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD),
        ("wScan", wt.WORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wt.ULONG)),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wt.LONG),
        ("dy", wt.LONG),
        ("mouseData", wt.DWORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wt.ULONG)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]


class RECT(ctypes.Structure):
    _fields_ = [("left", wt.LONG), ("top", wt.LONG), ("right", wt.LONG), ("bottom", wt.LONG)]


class POINT(ctypes.Structure):
    _fields_ = [("x", wt.LONG), ("y", wt.LONG)]


user32.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM), wt.LPARAM]
user32.GetWindowTextLengthW.argtypes = [wt.HWND]
user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(RECT)]
user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(RECT)]
user32.ClientToScreen.argtypes = [wt.HWND, ctypes.POINTER(POINT)]
user32.IsWindow.argtypes = [wt.HWND]
user32.IsWindow.restype = wt.BOOL
user32.IsWindowVisible.argtypes = [wt.HWND]
user32.IsWindowVisible.restype = wt.BOOL
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wt.HWND
user32.GetDC.argtypes = [wt.HWND]
user32.GetDC.restype = wt.HDC
user32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]
user32.SetForegroundWindow.argtypes = [wt.HWND]
user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
user32.BringWindowToTop.argtypes = [wt.HWND]
user32.SetFocus.argtypes = [wt.HWND]
user32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.GetWindowThreadProcessId.restype = wt.DWORD
user32.AttachThreadInput.argtypes = [wt.DWORD, wt.DWORD, wt.BOOL]
kernel32.GetCurrentThreadId.restype = wt.DWORD
gdi32.CreateCompatibleDC.argtypes = [wt.HDC]
gdi32.CreateCompatibleDC.restype = wt.HDC
gdi32.CreateDIBSection.argtypes = [
    wt.HDC,
    ctypes.POINTER(BITMAPINFO),
    wt.UINT,
    ctypes.POINTER(ctypes.c_void_p),
    wt.HANDLE,
    wt.DWORD,
]
gdi32.CreateDIBSection.restype = wt.HBITMAP
gdi32.SelectObject.argtypes = [wt.HDC, wt.HGDIOBJ]
gdi32.SelectObject.restype = wt.HGDIOBJ
gdi32.BitBlt.argtypes = [
    wt.HDC,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wt.HDC,
    ctypes.c_int,
    ctypes.c_int,
    wt.DWORD,
]
gdi32.DeleteObject.argtypes = [wt.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wt.HDC]


def dpi_aware() -> str:
    """Make our coordinates physical pixels (the host runs at 125% scaling)."""
    try:
        shcore = ctypes.WinDLL("shcore")
        shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return "SetProcessDpiAwareness(2)"
    except Exception:  # pragma: no cover - only on very old Windows
        user32.SetProcessDPIAware()
        return "SetProcessDPIAware"


# ---------------------------------------------------------------------------
# Window discovery
# ---------------------------------------------------------------------------
def _window_area(hwnd: int) -> int:
    rc = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rc))
    return max(0, rc.right - rc.left) * max(0, rc.bottom - rc.top)


def find_window(pid: int, class_name: str = "BASE") -> int:
    """Largest visible window owned by ``pid`` whose class is ``class_name``."""
    candidates: list[tuple[int, str, int]] = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def _enum(hwnd, _lparam):
        owner = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            cls = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls, 256)
            candidates.append((int(hwnd), cls.value, _window_area(hwnd)))
        return True

    user32.EnumWindows(_enum, 0)
    if not candidates:
        return 0
    preferred = [c for c in candidates if c[1] == class_name]
    pool = preferred or candidates
    pool.sort(key=lambda c: c[2], reverse=True)
    return pool[0][0]


def window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def class_name(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def client_rect(hwnd: int) -> tuple[int, int, int, int]:
    rc = RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rc))
    origin = POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return origin.x, origin.y, rc.right - rc.left, rc.bottom - rc.top


def close_window(hwnd: int) -> None:
    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)


def list_process_ids(image_name: str = "th08.exe") -> list[int]:
    """PIDs whose executable name matches ``image_name`` (Toolhelp32 snapshot).

    TH08 guards itself with the session-wide named mutex ``Touhou 08 App``
    (``main.cpp:994``), so a hard requirement of the runtime is that at most one
    instance exists; the launcher uses this to wait for a clean slate.
    """
    TH32CS_SNAPPROCESS = 0x00000002
    kernel32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wt.HANDLE
    kernel32.Process32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    pids: list[int] = []
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return pids
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            if entry.szExeFile.lower() == image_name.lower():
                pids.append(int(entry.th32ProcessID))
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return pids


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wt.DWORD),
        ("cntUsage", wt.DWORD),
        ("th32ProcessID", wt.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(wt.ULONG)),
        ("th32ModuleID", wt.DWORD),
        ("cntThreads", wt.DWORD),
        ("th32ParentProcessID", wt.DWORD),
        ("pcPriClassBase", wt.LONG),
        ("dwFlags", wt.DWORD),
        ("szExeFile", wt.WCHAR * 260),
    ]


def find_dialog(pid: int) -> int:
    """Return the HWND of a classic ``#32770`` dialog owned by ``pid`` (or 0).

    TH08 reports fatal start-up problems (notably ``TH_ERR_ALREADY_RUNNING`` from
    the session-wide ``Touhou 08 App`` mutex) as a modal dialog titled ``log``;
    the process then never reaches the title screen.
    """
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def _enum(hwnd, _lparam):
        owner = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid:
            cls = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, cls, 64)
            if cls.value == "#32770":
                found.append(int(hwnd))
        return True

    user32.EnumWindows(_enum, 0)
    return found[0] if found else 0


def dialog_text(hwnd: int) -> str:
    lines: list[str] = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def _child(child, _lparam):
        buf = ctypes.create_unicode_buffer(8192)
        user32.GetWindowTextW(child, buf, 8192)
        if buf.value.strip():
            lines.append(buf.value)
        return True

    user32.EnumChildWindows(hwnd, _child, 0)
    return "\n".join(lines)


def force_foreground(hwnd: int) -> bool:
    """Bring ``hwnd`` to the foreground from a background process.

    ``SetForegroundWindow`` is refused when the caller does not own the current
    foreground, so attach our input queue to both the foreground thread and the
    target thread for the duration of the call.
    """
    if not hwnd or not user32.IsWindow(hwnd):
        return False
    if user32.GetForegroundWindow() == hwnd:
        return True
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.BringWindowToTop(hwnd)
    fg = user32.GetForegroundWindow()
    current = kernel32.GetCurrentThreadId()
    fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    target_thread = user32.GetWindowThreadProcessId(hwnd, None)
    attached = []
    for thread in (fg_thread, target_thread):
        if thread and user32.AttachThreadInput(current, thread, True):
            attached.append(thread)
    try:
        user32.SetForegroundWindow(hwnd)
        user32.SetFocus(hwnd)
    finally:
        for thread in attached:
            user32.AttachThreadInput(current, thread, False)
    return user32.GetForegroundWindow() == hwnd


# ---------------------------------------------------------------------------
# Session / window state (guards against a frozen window DC)
# ---------------------------------------------------------------------------
# Why these exist: `GetDC` + `BitBlt` keeps returning the *last painted* client
# area when the window stops being composited (locked desktop, display off,
# minimised).  BitBlt still reports success, so a capture that only checks return
# values cannot tell a live frame from a stale one.  v1.0.3 shipped 52,888
# identical samples (73% of a dataset) because of exactly that.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
DWMWA_CLOAKED = 14

kernel32.SetThreadExecutionState.argtypes = [wt.DWORD]
kernel32.SetThreadExecutionState.restype = wt.DWORD
user32.IsIconic.argtypes = [wt.HWND]
user32.IsIconic.restype = wt.BOOL

try:  # pragma: no cover - depends on the host's DWM
    dwmapi = ctypes.WinDLL("dwmapi")
    dwmapi.DwmGetWindowAttribute.argtypes = [wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD]
    dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
except OSError:  # pragma: no cover
    dwmapi = None


def keep_awake(enable: bool = True) -> bool:
    """Ask Windows not to sleep and not to turn the display off.

    The power plan is a per-user setting the capture cannot rely on: on this host
    the display went off after 5 minutes and the session locked around minute 10,
    which froze the window DC mid-capture.  Calling this for the duration of a
    capture makes the guarantee local to the process instead of to the machine.
    """
    flags = ES_CONTINUOUS
    if enable:
        flags |= ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    return bool(kernel32.SetThreadExecutionState(flags))


def is_iconic(hwnd: int) -> bool:
    """True when the window is minimised."""
    return bool(user32.IsIconic(hwnd)) if hwnd else False


def is_cloaked(hwnd: int) -> bool:
    """True when DWM is not compositing the window (locked desktop, hidden shell)."""
    if not hwnd or dwmapi is None:
        return False
    value = wt.DWORD(0)
    status = dwmapi.DwmGetWindowAttribute(
        wt.HWND(hwnd), DWMWA_CLOAKED, ctypes.byref(value), ctypes.sizeof(value)
    )
    return status == 0 and bool(value.value)


def window_visible(hwnd: int) -> bool:
    """True when the window is neither minimised nor cloaked."""
    return not is_iconic(hwnd) and not is_cloaked(hwnd)


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------
class GdiCapture:
    """Reusable top-down 32-bit DIB that ``BitBlt`` writes into.

    Reusing the DIBSection/HBITMAP avoids allocating GDI objects every frame
    (the one-shot probe allocated per grab, which dominated its 1.98 ms).
    """

    def __init__(self, width: int, height: int):
        import numpy as np

        self.np = np
        self.width = width
        self.height = height
        self.hdc = gdi32.CreateCompatibleDC(0)
        if not self.hdc:
            raise ctypes.WinError(ctypes.get_last_error())
        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height  # top-down
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB
        bits = ctypes.c_void_p()
        self.hbitmap = gdi32.CreateDIBSection(
            self.hdc, ctypes.byref(info), DIB_RGB_COLORS, ctypes.byref(bits), None, 0
        )
        if not self.hbitmap:
            gdi32.DeleteDC(self.hdc)
            raise ctypes.WinError(ctypes.get_last_error())
        self._bits = bits
        self._old = gdi32.SelectObject(self.hdc, self.hbitmap)
        self._buffer = ctypes.string_at(bits, width * height * 4)

    def grab(self, hwnd: int):
        """Return the current client area as ``uint8[H, W, 3]`` RGB."""
        src = user32.GetDC(hwnd)
        if not src:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            ok = gdi32.BitBlt(self.hdc, 0, 0, self.width, self.height, src, 0, 0, SRCCOPY)
        finally:
            user32.ReleaseDC(hwnd, src)
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        raw = ctypes.string_at(self._bits, self.width * self.height * 4)
        bgra = self.np.frombuffer(raw, dtype=self.np.uint8).reshape(self.height, self.width, 4)
        return bgra[:, :, 2::-1]

    def close(self) -> None:
        if self._old:
            gdi32.SelectObject(self.hdc, self._old)
            self._old = None
        if self.hbitmap:
            gdi32.DeleteObject(self.hbitmap)
            self.hbitmap = None
        if self.hdc:
            gdi32.DeleteDC(self.hdc)
            self.hdc = None


# ---------------------------------------------------------------------------
# Keyboard injection
# ---------------------------------------------------------------------------
class SendInputKeyboard:
    """Scan-code ``SendInput``.  Requires the game window to be foreground."""

    def __init__(self) -> None:
        self._down: set[str] = set()

    def _send(self, name: str, key_up: bool) -> None:
        scan = spec.KEYS[name][1]
        inp = INPUT()
        inp.type = 1  # INPUT_KEYBOARD
        inp.ki = KEYBDINPUT(0, scan, KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if key_up else 0), 0, None)
        if user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def set_keys(self, keys: set[str]) -> None:
        for name in sorted(self._down - keys):
            self._send(name, True)
            self._down.discard(name)
        for name in sorted(keys - self._down):
            self._send(name, False)
            self._down.add(name)

    def tap(self, name: str, hold_s: float = 0.06) -> None:
        self.set_keys(self._down | {name})
        time.sleep(hold_s)
        self.set_keys(self._down - {name})

    def release_all(self) -> None:
        self.set_keys(set())


class PostMessageKeyboard:
    """``WM_KEYDOWN``/``WM_KEYUP`` injection; needs ``dontUseDirectInput``.

    ``GetKeyboardState`` (used by ``Controller::GetInput`` when the DirectInput
    keyboard object is absent) reflects the key state of the calling thread as
    updated by the messages it removes from its own queue, so posted messages
    are enough -- no foreground window required.
    """

    def __init__(self, hwnd: int) -> None:
        self.hwnd = hwnd
        self._down: set[str] = set()

    def _post(self, name: str, key_up: bool) -> None:
        vk, scan = spec.KEYS[name]
        lparam = 1 | (scan << 16) | (0xC0000000 if key_up else 0)
        if not user32.PostMessageW(self.hwnd, WM_KEYUP if key_up else WM_KEYDOWN, vk, lparam):
            raise ctypes.WinError(ctypes.get_last_error())

    def set_keys(self, keys: set[str]) -> None:
        for name in sorted(self._down - keys):
            self._post(name, True)
            self._down.discard(name)
        for name in sorted(keys - self._down):
            self._post(name, False)
            self._down.add(name)

    def tap(self, name: str, hold_s: float = 0.06) -> None:
        self.set_keys(self._down | {name})
        time.sleep(hold_s)
        self.set_keys(self._down - {name})

    def release_all(self) -> None:
        self.set_keys(set())
