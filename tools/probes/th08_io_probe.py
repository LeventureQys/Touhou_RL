"""TH08 I/O feasibility probe (one-shot measurement tool, not part of the runtime).

Purpose
-------
Establish measured facts for the v1.0.1 design before writing framework code:

1. Locate the game window and get a stable client rect (DPI-correct).
2. Compare Win32 capture backends on a real game frame (validity + cost).
3. Recover gameplay state through a strictly read-only 32-bit memory reader
   using linker-map addresses (g_Supervisor / g_GameManager / g_Player /
   g_BulletManager).
4. Drive the title menu from the outside into Stage 1, then verify that
   SendInput (scancode) reaches the game's DirectInput foreground device:
   observable through g_CurFrameInput, player position, focusMode, bullets.
5. Time a clean launch and a clean shutdown.

The probe never writes to game memory: TH08 has an anti-tamper checksum, so the
reader is strictly read-only.

Run (from repository root):
    python tools/probes/th08_io_probe.py --run-dir third_party/th08/build/play
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import struct
import subprocess
import sys
import time
from ctypes import wintypes as wt
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Win32 bindings
# ---------------------------------------------------------------------------

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0
PW_CLIENTONLY = 1
PW_RENDERFULLCONTENT = 2
WM_CLOSE = 0x0010
SW_RESTORE = 9
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400


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
    _fields_ = [
        ("left", wt.LONG),
        ("top", wt.LONG),
        ("right", wt.LONG),
        ("bottom", wt.LONG),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", wt.LONG), ("y", wt.LONG)]


user32.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM), wt.LPARAM]
user32.GetWindowTextLengthW.argtypes = [wt.HWND]
user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(RECT)]
user32.ClientToScreen.argtypes = [wt.HWND, ctypes.POINTER(POINT)]
user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(RECT)]
user32.IsWindowVisible.argtypes = [wt.HWND]
user32.IsWindowVisible.restype = wt.BOOL
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wt.HWND
user32.GetDC.argtypes = [wt.HWND]
user32.GetDC.restype = wt.HDC
user32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]
user32.PrintWindow.argtypes = [wt.HWND, wt.HDC, wt.UINT]
user32.SetForegroundWindow.argtypes = [wt.HWND]
user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
user32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
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
gdi32.BitBlt.argtypes = [wt.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wt.HDC, ctypes.c_int, ctypes.c_int, wt.DWORD]
gdi32.DeleteObject.argtypes = [wt.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wt.HDC]
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.ReadProcessMemory.argtypes = [wt.HANDLE, wt.LPCVOID, wt.LPVOID, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
kernel32.CloseHandle.argtypes = [wt.HANDLE]

# ---------------------------------------------------------------------------
# TH08 1.00d rebuilt bugfix build (Document/Update/Alpha/v1.0.0/问题清单.md
# F-06/F-07). Preferred image base 0x400000, DllCharacteristics has no
# DYNAMIC_BASE bit, so linker-map VA == runtime VA.
# ---------------------------------------------------------------------------
IMAGE_BASE_PREFERRED = 0x00400000
ADDR_SUPERVISOR = 0x017DC380       # g_Supervisor
ADDR_GAME_MANAGER = 0x0161C7A0     # g_GameManager
ADDR_PLAYER = 0x017E3120           # g_Player
ADDR_BULLET_MANAGER = 0x00F620B8   # g_BulletManager
ADDR_TITLE_SCREEN = 0x018CB030     # g_TitleScreen (TitleScreen *)
ADDR_CUR_FRAME_INPUT = 0x0165A808  # g_CurFrameInput (u16)
ADDR_LAST_FRAME_INPUT = 0x0165A804 # g_LastFrameInput (u16)

OFF_SUP_CUR_STATE = 0x15C          # Supervisor::curState (SupervisorState)
OFF_SUP_WANTED_STATE = 0x158
OFF_GM_GLOBALS = 0x8               # GameManager::globals
OFF_GM_CURRENT_STAGE = 0x3DDC4
OFF_GM_GAMEPLAY_FRAME = 0x3DDC0
OFF_GM_FLAGS = 0x3DBAC             # GameManager::flags (u32 bitfield)
OFF_GM_DEMO_REPLAY = 0x3DBB4
OFF_GM_REPLAY_MODE = 0x3DBB5
OFF_BM_ACTIVE_BULLETS = 0x6BA538
OFF_BM_BULLETS = 0x1A880
OFF_BULLET_POSITION = 0xD44
OFF_BULLET_SIZE = 0x10B8
OFF_PLAYER_POSITION = 0x2B4
OFF_PLAYER_FOCUS = 0x3
OFF_GLOBALS_LIVES = 0x74
OFF_GLOBALS_BOMBS = 0x80
OFF_GLOBALS_POWER = 0x98
OFF_GLOBALS_SCORE = 0x8
OFF_GLOBALS_DISPLAY_SCORE = 0x0
OFF_GLOBALS_GRAZE = 0xC
OFF_GLOBALS_CLOCK = 0x28
OFF_GLOBALS_TIMER_ORBS = 0x3C
OFF_GLOBALS_HIGH_SCORE = 0x14
OFF_TITLE_CURSOR = 0x0             # TitleScreen::cursor
OFF_TITLE_CURSOR2 = 0x4
OFF_TITLE_SCREEN_STATE = 0xC       # TitleScreen::currentScreenState
OFF_TITLE_STATE_TIMER2 = 0x14      # TitleScreen::stateTimer2 (unconsumedDword14 alias)

# Key bits: th08/src/Global.hpp:149-175.
BUTTON_SHOOT = 1 << 0
BUTTON_BOMB = 1 << 1
BUTTON_FOCUS = 1 << 2
BUTTON_MENU = 1 << 3
BUTTON_UP = 1 << 4
BUTTON_DOWN = 1 << 5
BUTTON_LEFT = 1 << 6
BUTTON_RIGHT = 1 << 7

# SupervisorState: th08/src/Supervisor.hpp:105-121.
STATE_TITLE = 1
STATE_GAMEMANAGER = 2
# TitleCurrentScreen: th08/src/TitleScreen.hpp:20-42.
SCREEN_START_MENU = 0
SCREEN_DIFFICULTY_SELECT = 4
SCREEN_CHARACTER_SELECT = 5
SCREEN_READY = 1

SCANCODE = {
    "up": 0x48,
    "down": 0x50,
    "left": 0x4B,
    "right": 0x4D,
    "z": 0x2C,
    "x": 0x2D,
    "shift": 0x2A,
    "esc": 0x01,
    "enter": 0x1C,
    "ctrl": 0x1D,
}

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def dpi_aware() -> str:
    """Make coordinates physical pixels; otherwise a scaled desktop reports lies."""
    try:
        shcore = ctypes.WinDLL("shcore")
        shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return "SetProcessDpiAwareness(2)"
    except Exception:
        user32.SetProcessDPIAware()
        return "SetProcessDPIAware"


def _window_area(hwnd: int) -> int:
    rc = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rc))
    return max(0, rc.right - rc.left) * max(0, rc.bottom - rc.top)


def find_window(pid: int) -> int:
    """Find the game window.

    TH08 sets its title from a Shift-JIS ANSI string, so on a CP936 host the
    Win32 title comes back as mojibake and cannot be matched by "永夜抄".
    Identify by process id + window class ("BASE") instead, then fall back to
    the largest visible window owned by the process.
    """
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
    bases = [c for c in candidates if c[1] == "BASE"]
    pool = bases or candidates
    pool.sort(key=lambda c: c[2], reverse=True)
    return pool[0][0]


def window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def client_rect(hwnd: int) -> tuple[int, int, int, int]:
    rc = RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rc))
    origin = POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return origin.x, origin.y, rc.right - rc.left, rc.bottom - rc.top


# ---------------------------------------------------------------------------
# Capture backends
# ---------------------------------------------------------------------------

class DibSurface:
    """A top-down 32bpp DIB section that BitBlt/PrintWindow can write into."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.hdc = gdi32.CreateCompatibleDC(0)
        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height  # negative => top-down rows
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB
        bits = ctypes.c_void_p()
        self.hbitmap = gdi32.CreateDIBSection(self.hdc, ctypes.byref(info), DIB_RGB_COLORS, ctypes.byref(bits), None, 0)
        if not self.hbitmap:
            raise ctypes.WinError(ctypes.get_last_error())
        self._bits = bits
        self._old = gdi32.SelectObject(self.hdc, self.hbitmap)

    def as_rgb(self) -> np.ndarray:
        raw = ctypes.string_at(self._bits, self.width * self.height * 4)
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
        return bgra[:, :, 2::-1].copy()  # BGRA -> RGB

    def close(self) -> None:
        if self._old:
            gdi32.SelectObject(self.hdc, self._old)
        gdi32.DeleteObject(self.hbitmap)
        gdi32.DeleteDC(self.hdc)


def grab_window_dc(hwnd: int, width: int, height: int) -> np.ndarray:
    surface = DibSurface(width, height)
    src = user32.GetDC(hwnd)
    if not src:
        surface.close()
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not gdi32.BitBlt(surface.hdc, 0, 0, width, height, src, 0, 0, SRCCOPY):
            raise ctypes.WinError(ctypes.get_last_error())
        return surface.as_rgb()
    finally:
        user32.ReleaseDC(hwnd, src)
        surface.close()


def grab_desktop(hwnd: int, width: int, height: int) -> np.ndarray:
    surface = DibSurface(width, height)
    src = user32.GetDC(0)
    ox, oy, _w, _h = client_rect(hwnd)
    try:
        if not gdi32.BitBlt(surface.hdc, 0, 0, width, height, src, ox, oy, SRCCOPY):
            raise ctypes.WinError(ctypes.get_last_error())
        return surface.as_rgb()
    finally:
        user32.ReleaseDC(0, src)
        surface.close()


def grab_printwindow(hwnd: int, width: int, height: int, flags: int = PW_RENDERFULLCONTENT) -> np.ndarray:
    surface = DibSurface(width, height)
    try:
        if not user32.PrintWindow(hwnd, surface.hdc, flags):
            raise OSError(f"PrintWindow failed (flags={flags})")
        return surface.as_rgb()
    finally:
        surface.close()


def frame_stats(rgb: np.ndarray) -> dict:
    gray = rgb.astype(np.float32).mean(axis=2)
    return {
        "shape": list(rgb.shape),
        "mean": round(float(gray.mean()), 2),
        "std": round(float(gray.std()), 2),
        "nonblack_ratio": round(float((gray > 8).mean()), 4),
        "unique_colors": int(len(np.unique(rgb.reshape(-1, 3), axis=0))),
    }


def measure_fps(fn, frames: int) -> dict:
    times = []
    for _ in range(frames):
        started = time.perf_counter()
        fn()
        times.append((time.perf_counter() - started) * 1000.0)
    times.sort()
    return {
        "frames": frames,
        "mean_ms": round(statistics.mean(times), 3),
        "p50_ms": round(statistics.median(times), 3),
        "p95_ms": round(times[int(len(times) * 0.95) - 1], 3),
        "max_ms": round(times[-1], 3),
        "fps_from_mean": round(1000.0 / statistics.mean(times), 1),
    }


# ---------------------------------------------------------------------------
# Read-only process memory access
# ---------------------------------------------------------------------------

class RemoteMemory:
    def __init__(self, pid: int):
        self.handle = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.pid = pid
        self.reads = 0
        self.failures = 0
        self.module_base = self._module_base()

    def _module_base(self) -> int:
        """Resolve th08.exe's load base in the target process.

        The 32-bit target is inspected from a 64-bit host, so Toolhelp32 with
        TH32CS_SNAPMODULE32 is the portable route (PSAPI's module enumeration
        needs extra flag handling for WOW64 targets).
        """
        class MODULEENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wt.DWORD),
                ("th32ModuleID", wt.DWORD),
                ("th32ProcessID", wt.DWORD),
                ("GlblcntUsage", wt.DWORD),
                ("ProccntUsage", wt.DWORD),
                ("modBaseAddr", ctypes.POINTER(wt.BYTE)),
                ("modBaseSize", wt.DWORD),
                ("hModule", wt.HMODULE),
                ("szModule", wt.WCHAR * 256),
                ("szExePath", wt.WCHAR * 260),
            ]

        TH32CS_SNAPMODULE = 0x00000008
        TH32CS_SNAPMODULE32 = 0x00000010
        kernel32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
        kernel32.CreateToolhelp32Snapshot.restype = wt.HANDLE
        kernel32.Module32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(MODULEENTRY32W)]
        kernel32.Module32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(MODULEENTRY32W)]

        snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, self.pid)
        if snapshot and snapshot != wt.HANDLE(-1).value:
            try:
                entry = MODULEENTRY32W()
                entry.dwSize = ctypes.sizeof(MODULEENTRY32W)
                ok = kernel32.Module32FirstW(snapshot, ctypes.byref(entry))
                while ok:
                    if entry.szModule.lower() == "th08.exe":
                        return ctypes.cast(entry.modBaseAddr, ctypes.c_void_p).value or 0
                    ok = kernel32.Module32NextW(snapshot, ctypes.byref(entry))
            finally:
                kernel32.CloseHandle(snapshot)
        # Fallback: no-ASLR image, accept 0x400000 only if an MZ header is there.
        try:
            if self.read(IMAGE_BASE_PREFERRED, 2) == b"MZ":
                return IMAGE_BASE_PREFERRED
        except OSError:
            pass
        return 0

    def read(self, address: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        got = ctypes.c_size_t()
        ok = kernel32.ReadProcessMemory(self.handle, ctypes.c_void_p(address), buf, size, ctypes.byref(got))
        self.reads += 1
        if not ok or got.value != size:
            self.failures += 1
            raise OSError(f"ReadProcessMemory failed at 0x{address:08X} size={size} err={ctypes.get_last_error()}")
        return buf.raw

    def u32(self, address: int) -> int:
        return struct.unpack("<I", self.read(address, 4))[0]

    def i32(self, address: int) -> int:
        return struct.unpack("<i", self.read(address, 4))[0]

    def f32(self, address: int) -> float:
        return struct.unpack("<f", self.read(address, 4))[0]

    def u16(self, address: int) -> int:
        return struct.unpack("<H", self.read(address, 2))[0]

    def u8(self, address: int) -> int:
        return struct.unpack("<B", self.read(address, 1))[0]

    def i8(self, address: int) -> int:
        return struct.unpack("<b", self.read(address, 1))[0]

    def close(self) -> None:
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = 0


def read_state(mem: RemoteMemory) -> dict:
    """Decode the small subset of state the design depends on."""
    delta = mem.module_base - IMAGE_BASE_PREFERRED
    sup = ADDR_SUPERVISOR + delta
    gm = ADDR_GAME_MANAGER + delta
    player = ADDR_PLAYER + delta
    bm = ADDR_BULLET_MANAGER + delta
    title_ptr = mem.u32(ADDR_TITLE_SCREEN + delta)
    # globals is a stored pointer: it is already an absolute address in the
    # target address space, so no rebase delta is applied to it.
    globals_ptr = mem.u32(gm + OFF_GM_GLOBALS)
    px, py, pz = (mem.f32(player + OFF_PLAYER_POSITION + 4 * i) for i in range(3))
    state = {
        "module_base": f"0x{mem.module_base:08X}",
        "globals_ptr": f"0x{globals_ptr:08X}",
        "lives": mem.f32(globals_ptr + OFF_GLOBALS_LIVES),
        "bombs": mem.f32(globals_ptr + OFF_GLOBALS_BOMBS),
        "power": mem.f32(globals_ptr + OFF_GLOBALS_POWER),
        "score": mem.u32(globals_ptr + OFF_GLOBALS_SCORE),
        "display_score": mem.u32(globals_ptr + OFF_GLOBALS_DISPLAY_SCORE),
        "high_score_display": mem.u32(globals_ptr + OFF_GLOBALS_HIGH_SCORE),
        "graze": mem.i32(globals_ptr + OFF_GLOBALS_GRAZE),
        "clock_time": mem.i8(globals_ptr + OFF_GLOBALS_CLOCK),
        "time_orbs": mem.i32(globals_ptr + OFF_GLOBALS_TIMER_ORBS),
        "supervisor_state": mem.i32(sup + OFF_SUP_CUR_STATE),
        "wanted_state": mem.i32(sup + OFF_SUP_WANTED_STATE),
        "current_stage": mem.i32(gm + OFF_GM_CURRENT_STAGE),
        "gameplay_frame": mem.u32(gm + OFF_GM_GAMEPLAY_FRAME),
        "flags": mem.u32(gm + OFF_GM_FLAGS),
        "demo_replay": mem.u8(gm + OFF_GM_DEMO_REPLAY),
        "replay_mode": mem.u8(gm + OFF_GM_REPLAY_MODE),
        "player_state": mem.i8(player),
        "player_focus": mem.u8(player + OFF_PLAYER_FOCUS),
        "player_pos": [round(px, 3), round(py, 3), round(pz, 3)],
        "input_mask": mem.u16(ADDR_CUR_FRAME_INPUT + delta),
        "last_input_mask": mem.u16(ADDR_LAST_FRAME_INPUT + delta),
        "active_bullets": mem.i32(bm + OFF_BM_ACTIVE_BULLETS),
    }
    # GameManagerFlags bitfield: th08/src/GameManager.hpp:27-54.
    state["is_demo"] = bool(state["flags"] & (1 << 1))
    state["is_replay"] = bool(state["flags"] & (1 << 3))
    state["is_practice"] = bool(state["flags"] & 1)
    state["is_spell_practice"] = bool(state["flags"] & (1 << 14))
    if title_ptr:
        state["title_screen"] = {
            "ptr": f"0x{title_ptr:08X}",
            "cursor": mem.i32(title_ptr + OFF_TITLE_CURSOR),
            "cursor2": mem.i32(title_ptr + OFF_TITLE_CURSOR2),
            "screen_state": mem.i32(title_ptr + OFF_TITLE_SCREEN_STATE),
            "state_timer2": mem.i32(title_ptr + OFF_TITLE_STATE_TIMER2),
        }
    return state


def read_bullets(mem: RemoteMemory, limit: int = 64) -> list[list[float]]:
    """Read the first `limit` bullet positions (privileged observation sample)."""
    delta = mem.module_base - IMAGE_BASE_PREFERRED
    base = ADDR_BULLET_MANAGER + delta + OFF_BM_BULLETS + OFF_BULLET_POSITION
    out = []
    for index in range(limit):
        address = base + index * OFF_BULLET_SIZE
        out.append(list(struct.unpack("<fff", mem.read(address, 12))))
    return out


# ---------------------------------------------------------------------------
# Input injection
# ---------------------------------------------------------------------------

class Keyboard:
    def __init__(self):
        self._down: set[str] = set()

    def _send(self, name: str, key_up: bool) -> None:
        scan = SCANCODE[name]
        inp = INPUT()
        inp.type = 1  # INPUT_KEYBOARD
        inp.ki = KEYBDINPUT(0, scan, KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if key_up else 0), 0, None)
        sent = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if sent != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def press(self, *names: str) -> None:
        for name in names:
            if name not in self._down:
                self._send(name, False)
                self._down.add(name)

    def release(self, *names: str) -> None:
        for name in names:
            if name in self._down:
                self._send(name, True)
                self._down.discard(name)

    def tap(self, name: str, hold_s: float = 0.08) -> None:
        self.press(name)
        time.sleep(hold_s)
        self.release(name)

    def release_all(self) -> None:
        for name in sorted(self._down):
            self._send(name, True)
        self._down.clear()


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def wait_until(predicate, timeout: float, interval: float = 0.05, description: str = ""):
    deadline = time.perf_counter() + timeout
    last_error = None
    while time.perf_counter() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except Exception as exc:  # memory may not be committed yet during boot
            last_error = exc
        time.sleep(interval)
    raise TimeoutError(f"timeout waiting for {description or predicate}: last_error={last_error!r}")


def describe(state: dict) -> dict:
    return {
        "supervisor_state": state.get("supervisor_state"),
        "title": state.get("title_screen"),
        "stage": state.get("current_stage"),
        "gameplay_frame": state.get("gameplay_frame"),
        "lives": state.get("lives"),
        "bombs": state.get("bombs"),
        "bullets": state.get("active_bullets"),
        "pos": state.get("player_pos"),
        "input_mask": state.get("input_mask"),
        "is_demo": state.get("is_demo"),
        "is_replay": state.get("is_replay"),
    }


def force_foreground(hwnd: int) -> bool:
    """Bring `hwnd` to the foreground from a background process.

    Windows refuses SetForegroundWindow from a process that does not own the
    foreground, so temporarily attach our input queue to the current foreground
    thread and to the target thread (the classic AttachThreadInput dance).
    """
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TH08 capture/state/input probe")
    parser.add_argument("--run-dir", default="third_party/th08/build/play")
    parser.add_argument("--evidence-dir", default="")
    parser.add_argument("--boot-timeout", type=float, default=40.0)
    parser.add_argument("--fps-frames", type=int, default=120)
    parser.add_argument("--hold-ms", type=int, default=450)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--keep-open", action="store_true", help="do not close the game at the end")
    parser.add_argument("--highdpi-shim", action="store_true", default=True,
                        help="launch with __COMPAT_LAYER=HIGHDPIAWARE so the game renders at 1:1 pixels")
    parser.add_argument("--no-highdpi-shim", dest="highdpi_shim", action="store_false")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    evidence_dir = Path(args.evidence_dir).resolve() if args.evidence_dir else run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    exe = run_dir / "th08.exe"
    if not exe.exists():
        print(json.dumps({"error": f"missing {exe}"}, ensure_ascii=False))
        return 2

    env = dict(os.environ)
    if args.highdpi_shim:
        # Without this the DPI-unaware game is virtualized at the host's scale
        # factor (125% here), so every captured frame is a resampled 800x600.
        env["__COMPAT_LAYER"] = "HIGHDPIAWARE"

    report: dict = {
        "run_dir": str(run_dir),
        "dpi_mode": dpi_aware(),
        "highdpi_shim": args.highdpi_shim,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    kb = Keyboard()
    proc = subprocess.Popen([str(exe)], cwd=str(run_dir), env=env)
    mem = None
    report["pid"] = proc.pid
    hwnd = 0
    try:
        boot_started = time.perf_counter()
        mem = RemoteMemory(proc.pid)
        report["module_base"] = f"0x{mem.module_base:08X}"

        def _booted():
            state = read_state(mem)
            return state if state["supervisor_state"] == STATE_TITLE else None

        state = wait_until(_booted, args.boot_timeout, 0.05, "title screen")
        report["boot_to_title_s"] = round(time.perf_counter() - boot_started, 2)

        # Wait for the window to exist and stop changing size (TH08 creates a
        # temporary window before the D3D device settles).
        def _stable_window():
            handle = find_window(proc.pid)
            if not handle:
                return 0
            rect = client_rect(handle)
            time.sleep(0.2)
            if find_window(proc.pid) != handle or client_rect(handle) != rect:
                return 0
            return handle

        hwnd = wait_until(_stable_window, 20.0, 0.2, "stable window")
        report["hwnd"] = hwnd
        report["title"] = window_title(hwnd).encode("unicode_escape").decode("ascii")
        report["focus_acquired"] = force_foreground(hwnd)
        time.sleep(0.4)
        report["foreground_matches"] = user32.GetForegroundWindow() == hwnd
        ox, oy, width, height = client_rect(hwnd)
        report["client_rect"] = {"screen_x": ox, "screen_y": oy, "width": width, "height": height}
        report["state_title"] = describe(read_state(mem))

        # --- capture backend comparison on the title screen -----------------
        backends = {
            "window_dc": lambda: grab_window_dc(hwnd, width, height),
            "desktop_dc": lambda: grab_desktop(hwnd, width, height),
            "printwindow_full": lambda: grab_printwindow(hwnd, width, height, PW_RENDERFULLCONTENT),
            "printwindow_client": lambda: grab_printwindow(hwnd, width, height, PW_CLIENTONLY),
        }
        capture_report: dict = {}
        for name, fn in backends.items():
            entry: dict = {}
            try:
                rgb = fn()
                entry["stats"] = frame_stats(rgb)
                try:
                    from PIL import Image

                    path = evidence_dir / f"probe_title_{name}.png"
                    Image.fromarray(rgb).save(path)
                    entry["png"] = str(path)
                except Exception as exc:  # pragma: no cover - diagnostics only
                    entry["png_error"] = repr(exc)
                entry["timing"] = measure_fps(fn, max(10, args.fps_frames // 3))
            except Exception as exc:
                entry["error"] = repr(exc)
            capture_report[name] = entry
        report["capture_backends"] = capture_report

        ranked = sorted(
            ((k, v["stats"]["nonblack_ratio"]) for k, v in capture_report.items() if "stats" in v),
            key=lambda kv: kv[1],
            reverse=True,
        )
        report["best_backend"] = ranked[0][0] if ranked else None
        if ranked:
            report["best_backend_timing"] = measure_fps(backends[ranked[0][0]], args.fps_frames)

        # --- reach a human-controlled game ----------------------------------
        # Menu layout verified in th08/src/TitleScreen.cpp:81-89 and :499-568:
        # StartMenu cursor 0 = Start -> DifficultySelect -> CharacterSelect ->
        # SupervisorState_GameManager. Pressing any key cancels the attract
        # demo, which TitleScreen.cpp:434 starts after 1500 idle frames and
        # which otherwise looks like "in game" except for GameManagerFlags.
        def in_human_game(state: dict) -> bool:
            return (
                state["supervisor_state"] == STATE_GAMEMANAGER
                and not state["is_demo"]
                and not state["is_replay"]
                and state["gameplay_frame"] > 5
            )

        nav_log: list[dict] = []
        game_reached = False
        for attempt in range(1, 4):
            if in_human_game(read_state(mem)):
                game_reached = True
                break
            state = read_state(mem)
            if state["supervisor_state"] == STATE_GAMEMANAGER and (state["is_demo"] or state["is_replay"]):
                nav_log.append({"attempt": attempt, "action": "abort attract demo", "state": describe(state)})
                kb.tap("z", 0.06)
                time.sleep(0.8)
            if not force_foreground(hwnd):
                nav_log.append({"attempt": attempt, "action": "focus lost"})
            time.sleep(1.0)
            for step, label in enumerate(("start", "difficulty", "character"), start=1):
                nav_log.append({"attempt": attempt, "step": step, "action": f"tap z ({label})",
                                "state": describe(read_state(mem))})
                kb.tap("z", 0.06)
                time.sleep(1.1)
                if in_human_game(read_state(mem)):
                    game_reached = True
                    break
            if game_reached:
                break
            time.sleep(1.0)
        report["menu_navigation"] = {"reached_human_game": game_reached, "attempts": nav_log[-12:]}
        report["state_after_navigation"] = describe(read_state(mem))

        if game_reached:
            # Wait until the stage actually runs and bullets exist, so the
            # frame we keep as evidence is a real danmaku frame.
            def _has_bullets():
                state = read_state(mem)
                return state if state["active_bullets"] > 5 else None

            try:
                wait_until(_has_bullets, 25.0, 0.1, "bullets in stage")
                bullets_seen = True
            except TimeoutError:
                bullets_seen = False
            report["bullets_seen"] = bullets_seen
            report["state_in_game"] = describe(read_state(mem))
            report["bullets_sample"] = read_bullets(mem, 8)

            # Capture gameplay frames + measure the state read cost.
            for name in ("window_dc", "desktop_dc", "printwindow_full"):
                try:
                    rgb = backends[name]()
                    from PIL import Image

                    path = evidence_dir / f"probe_game_{name}.png"
                    Image.fromarray(rgb).save(path)
                    capture_report[name + "_in_game"] = {"stats": frame_stats(rgb), "png": str(path)}
                except Exception as exc:
                    capture_report[name + "_in_game"] = {"error": repr(exc)}
            report["capture_backends"] = capture_report
            report["state_read_timing"] = measure_fps(lambda: read_state(mem), 30)

            # Liveness: consecutive captures of a live stage must differ.
            frames = [grab_window_dc(hwnd, width, height) for _ in range(12)]
            diffs = [float(np.abs(frames[i].astype(np.int16) - frames[i + 1].astype(np.int16)).mean())
                     for i in range(len(frames) - 1)]
            report["frame_liveness"] = {
                "samples": len(frames),
                "mean_abs_diff": round(float(np.mean(diffs)), 3),
                "min_abs_diff": round(float(np.min(diffs)), 3),
            }

            # --- input loopback measured in game (focus is mandatory:
            # Supervisor.cpp:937 uses DISCL_FOREGROUND) --------------------
            report["focus_before_input"] = force_foreground(hwnd)
            before = read_state(mem)
            press_started = time.perf_counter()
            kb.press("z")
            latency = None
            seen = []
            for _ in range(30):
                now = read_state(mem)
                seen.append(now["input_mask"])
                if latency is None and now["input_mask"] & BUTTON_SHOOT:
                    latency = round((time.perf_counter() - press_started) * 1000.0, 2)
                time.sleep(0.008)
            kb.release("z")
            report["shoot_loopback"] = {
                "latency_ms": latency,
                "saw_shoot_bit": any(m & BUTTON_SHOOT for m in seen),
                "masks": seen,
            }

            pos_before = read_state(mem)
            kb.press("left", "shift")
            time.sleep(args.hold_ms / 1000.0)
            pos_mid = read_state(mem)
            kb.release("left", "shift")
            time.sleep(0.25)
            pos_after = read_state(mem)
            report["move_loopback"] = {
                "before": pos_before["player_pos"],
                "during": pos_mid["player_pos"],
                "after": pos_after["player_pos"],
                "dx_during": round(pos_mid["player_pos"][0] - pos_before["player_pos"][0], 3),
                "focus_during": pos_mid["player_focus"],
                "mask_during": pos_mid["input_mask"],
                "lives_before": pos_before["lives"],
                "score_before": pos_before["score"],
            }

            # Focused movement is slower and shows the hitbox: a dodge policy
            # needs both modalities.
            kb.press("shift")
            time.sleep(0.4)
            focused = read_state(mem)
            kb.release("shift")
            time.sleep(0.2)
            report["focus_toggle"] = {
                "focused_focus_mode": focused["player_focus"],
                "unfocused_focus_mode": read_state(mem)["player_focus"],
            }

            # One more frame after the input test as final evidence.
            try:
                from PIL import Image

                final = grab_window_dc(hwnd, width, height)
                path = evidence_dir / "probe_game_final.png"
                Image.fromarray(final).save(path)
                report["final_png"] = str(path)
                report["final_state"] = describe(read_state(mem))
            except Exception as exc:
                report["final_png_error"] = repr(exc)
        else:
            report["game_state_probe"] = describe(read_state(mem))
    except Exception as exc:
        report["error"] = repr(exc)
        try:
            if mem:
                report["state_at_error"] = describe(read_state(mem))
        except Exception:
            pass
    finally:
        kb.release_all()
        if mem:
            mem.close()
        if not args.keep_open:
            if proc.poll() is None:
                if hwnd:
                    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    report["forced_kill"] = True
        report["proc_exit_code"] = proc.poll()

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    sys.exit(main())
