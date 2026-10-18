"""Read-only cross-process memory access for the 32-bit TH08 process.

The reader is strictly read-only by construction: it opens the target with
``PROCESS_VM_READ | PROCESS_QUERY_INFORMATION`` and never calls
``WriteProcessMemory``, ``CreateRemoteThread`` or ``VirtualProtectEx``.  TH08
validates lives/bombs/power against a per-frame anti-tamper checksum and calls
``CRASH_GAME()`` on a mismatch (``GameManager.hpp`` ``IsTampered``), so writing
is not an option anyway.

Module base resolution uses Toolhelp32 (``TH32CS_SNAPMODULE32``) because the
64-bit host inspects a 32-bit (WOW64) target.  The image has no ``DYNAMIC_BASE``
bit, so ``0x00400000`` is also accepted as a fallback after checking ``MZ``.
"""

from __future__ import annotations

import ctypes
import struct
from ctypes import wintypes as wt

from . import spec

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.ReadProcessMemory.argtypes = [
    wt.HANDLE,
    wt.LPCVOID,
    wt.LPVOID,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.ReadProcessMemory.restype = wt.BOOL
kernel32.CloseHandle.argtypes = [wt.HANDLE]
kernel32.GetExitCodeProcess.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
kernel32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
kernel32.CreateToolhelp32Snapshot.restype = wt.HANDLE
kernel32.Module32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_void_p)]
kernel32.Module32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_void_p)]


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


class ProcessMemory:
    """A read-only handle on the game process with typed and bulk reads."""

    def __init__(self, pid: int, module_name: str = "th08.exe"):
        self.pid = pid
        self.module_name = module_name
        # PROCESS_VM_READ | PROCESS_QUERY_INFORMATION -- never VM_WRITE/VM_OPERATION.
        self.handle = kernel32.OpenProcess(0x0010 | 0x0400, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.reads = 0
        self.failures = 0
        self.module_base = self._resolve_module_base()

    # -- setup ------------------------------------------------------------
    def _resolve_module_base(self) -> int:
        TH32CS_SNAPMODULE = 0x00000008
        TH32CS_SNAPMODULE32 = 0x00000010
        kernel32.Module32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(MODULEENTRY32W)]
        kernel32.Module32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(MODULEENTRY32W)]
        snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, self.pid)
        if snapshot and snapshot != ctypes.c_void_p(-1).value:
            try:
                entry = MODULEENTRY32W()
                entry.dwSize = ctypes.sizeof(MODULEENTRY32W)
                ok = kernel32.Module32FirstW(snapshot, ctypes.byref(entry))
                while ok:
                    if entry.szModule.lower() == self.module_name.lower():
                        return ctypes.cast(entry.modBaseAddr, ctypes.c_void_p).value or 0
                    ok = kernel32.Module32NextW(snapshot, ctypes.byref(entry))
            finally:
                kernel32.CloseHandle(snapshot)
        try:
            if self.read(spec.IMAGE_BASE, 2) == b"MZ":
                return spec.IMAGE_BASE
        except OSError:
            pass
        return 0

    # -- basic reads ------------------------------------------------------
    def read(self, address: int, size: int) -> bytes:
        buf = ctypes.create_string_buffer(size)
        got = ctypes.c_size_t()
        ok = kernel32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(address), buf, size, ctypes.byref(got)
        )
        self.reads += 1
        if not ok or got.value != size:
            self.failures += 1
            raise OSError(
                f"ReadProcessMemory 0x{address:08X}+{size} failed "
                f"(read {got.value}/{size}, err={ctypes.get_last_error()})"
            )
        return buf.raw

    def read_into(self, address: int, buffer: ctypes.Array) -> None:
        """Read into a caller-owned ctypes buffer (avoids per-call allocation)."""
        size = ctypes.sizeof(buffer)
        got = ctypes.c_size_t()
        ok = kernel32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(address), buffer, size, ctypes.byref(got)
        )
        self.reads += 1
        if not ok or got.value != size:
            self.failures += 1
            raise OSError(
                f"ReadProcessMemory 0x{address:08X}+{size} failed "
                f"(read {got.value}/{size}, err={ctypes.get_last_error()})"
            )

    def u8(self, address: int) -> int:
        return struct.unpack("<B", self.read(address, 1))[0]

    def i8(self, address: int) -> int:
        return struct.unpack("<b", self.read(address, 1))[0]

    def u16(self, address: int) -> int:
        return struct.unpack("<H", self.read(address, 2))[0]

    def u32(self, address: int) -> int:
        return struct.unpack("<I", self.read(address, 4))[0]

    def i32(self, address: int) -> int:
        return struct.unpack("<i", self.read(address, 4))[0]

    def f32(self, address: int) -> float:
        return struct.unpack("<f", self.read(address, 4))[0]

    # -- lifecycle --------------------------------------------------------
    def is_alive(self) -> bool:
        code = wt.DWORD()
        if not kernel32.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            return False
        return code.value == 259  # STILL_ACTIVE

    def close(self) -> None:
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = 0

    def __enter__(self) -> "ProcessMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
