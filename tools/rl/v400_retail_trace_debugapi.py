"""EXP-P0-068 (v4.0.0 WP1): native Windows Debug API replay observer for retail TH08.

The official capture pipeline (``third_party/th08/scripts/capture-replay.py`` +
``replay-external-observer.py``) is Linux + Wine + GDB: four x86 *hardware
execute* breakpoints installed with ``gdb.Breakpoint(..., BP_HARDWARE_BREAKPOINT)``.
There is no Windows code path and no ``cdb.exe`` on this host (EXP-P0-067), so
this tool rebuilds the observer on the only native facility that offers the same
per-thread hardware execute breakpoints: the Windows Debug API.

What it replicates, byte for byte, from ``replay-external-observer.py``:

* the same four retail RVAs (``0x452490`` frame boundary / ``0x441F52``
  completion / ``0x439BC7`` stage start / ``0x453080`` replay delete);
* the same mapped-bytes attestations before any breakpoint is armed
  (``attestBytes`` / ``completionAttestBytes``, plus 13 bytes at the two extra
  addresses), so a different image can never be mistaken for 1.00d;
* the same 25-field row, in the same order, from the same addresses, with the
  same ``json.dumps(row) + "\\n"`` line format;
* the same ``complete.json`` payload (``rows``/``ended``/``errors``/``stages``/
  ``stageFrames``/``endScores``/``rngOrigins``);
* the same menu state machine (``screen``/``substate``/``cursor``/``state``/
  ``sub_timer`` reads at ``g_TitleScreen`` + ``0x14428``), with ``SendInput``
  scan codes instead of ``xdotool``.

The one deliberate difference is the read channel: the official observer reads
through ``gdb.selected_inferior().read_memory`` while the target is *stopped at
the breakpoint*; here ``ReadProcessMemory`` runs inside the debug event, while
the process is frozen for the same reason, so the "one row per calculation
frame, read atomically inside the frame" property is preserved.

Hard rules encoded here (checked, not just documented):

* the retail image is identified by size + SHA-256 before anything runs (exit 2);
* exactly one ``th08.exe`` may exist -- the phase aborts otherwise (exit 5);
* nothing under ``third_party/th08`` is written (assets are copied to a private
  run dir; every watched source file is hashed before and after);
* the game is only ever opened with ``PROCESS_VM_READ|PROCESS_QUERY_INFORMATION``
  (``touhou_rl.memory.ProcessMemory``), never written;
* the debug registers are the *only* thing changed in the target, they are
  cleared again on detach, and every debug event is ended with
  ``ContinueDebugEvent`` so the game is never left frozen;
* every phase is bounded by ``--budget-seconds`` (default cap 180 s, raised to
  1800 s only for the authorized G0.1 single-run gate), and that budget is
  enforced by an *independent watchdog thread* as well as by the driver loop
  (EXP-P0-075): when it fires it writes the report, flushes the buffered rows,
  clears DR0..DR3, detaches and terminates the game, so a main thread wedged in
  a Win32 call can no longer hang the run;

Usage::

    python tools/rl/v400_retail_trace_debugapi.py --phase poc   --out .tmp_v400_trace_dbg_poc
    python tools/rl/v400_retail_trace_debugapi.py --phase four  --out .tmp_v400_trace_dbg_four
    python tools/rl/v400_retail_trace_debugapi.py --phase rows  --out .tmp_v400_trace_dbg_rows
"""

from __future__ import annotations

import argparse
import ctypes
import faulthandler
import hashlib
import json
import msvcrt
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
from ctypes import wintypes as wt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RETAIL_EXE = ROOT / "third_party/th08/resources/th08.exe"
ASSET_DIR = ROOT / "third_party/th08/build/play_fence"
FIXTURES_JSON = ROOT / "third_party/th08/config/replay-fixtures.json"
DEFAULT_REPLAY = (
    ROOT / "third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/th8_ud2cd2.rpy"
)
REPLAY_FILE_PY = ROOT / "third_party/th08/scripts/replay_file.py"
#: the official definition of ``retailTrace.sha256`` (see ``canonical_trace_sha256``).
OFFICIAL_COMPARISON_PY = ROOT / "third_party/th08/scripts/compare-replay-traces.py"
OFFICIAL_SCHEMA_JSON = ROOT / "third_party/th08/scripts/replay-external-schema.json"

RETAIL_SIZE = 840704
RETAIL_SHA256 = "330fbdbf58a710829d65277b4f312cfbb38d5448b3df523e79350b879213d924"

#: Retail 1.00d globals the official observer reads (capture-replay.py:37-48).
GM = 0x0160F508
PLAYER = 0x017D5EF8
RNG = 0x0164D520
INPUT = 0x0164D52C
REPLAY_MANAGER = 0x018B8A28
TITLE_SCREEN = 0x018BDE08

#: the four hardware execute breakpoints of the official observer.
BREAKPOINTS = {
    "frame": 0x00452490,
    "completion": 0x00441F52,
    "stageStart": 0x00439BC7,
    "deleteReplay": 0x00453080,
}
BREAKPOINT_ROLES = {
    "frame": "ReplayManager::ControlPlaybackFrameAdvance -- playback-control entry after the "
    "gameplay calculation; this is the frame fence that writes one row per calculation frame",
    "completion": "GameWindow::Render, 5 bytes past the `mov ecx,g_Chain / call RunCalcChain` "
    "site; the observer's menu/completion tick and end-of-playback detector",
    "stageStart": "GameManager::OnUpdate entry; records the RNG generation origin of a stage",
    "deleteReplay": "ReplayManager::DeleteReplayManager; end of playback",
}

#: mapped-byte attestations (observer.json in the official pipeline).
ATTEST = (
    ("attestAddress", 0x00452490, "attestBytes", "558bec51894dfca1b4d06401c1"),
    ("completionAttestAddress", 0x00441F48, "completionAttestBytes", "b948f56401e8feaaffff8945fc"),
    ("stageStartBoundary", 0x00439BC7, None, None),
    ("deleteReplay", 0x00453080, None, None),
)

#: 25 field names, replay-external-schema.json (version 4).
FIELDS = [
    "calc", "stage", "frame", "gameFrame", "score", "graze", "deaths", "lives",
    "bombs", "power", "x", "y", "z", "playerState", "rngSeed", "rngGeneration",
    "input", "pointItems", "timeOrbs", "gauge", "clock", "pointValue",
    "character", "shot", "difficulty",
]

ASSET_FILES = ("th08.dat", "thbgm.dat", "th08.cfg", "score.dat")

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BLOCKED = 3
EXIT_PROCESS_DIED = 4
EXIT_INSTANCE = 5
EXIT_STATE = 6
EXIT_TIMEOUT = 7

#: EXP-P0-068 measured ~106 rows/s, so the 99,113-row retail trace needs ~935 s
#: of uninterrupted playback.  A trace cannot be assembled from sharded runs
#: (a restart changes the stage entry state and the RNG origins), so the G0.1
#: gate authorizes one continuous run; the cap is still a cap, just a larger one.
#:
#: EXP-P0-076 (v4.0.0 WP1): raised from 1800 s to 5400 s because the batch driver
#: now passes an adaptive per-fixture budget,
#: ``max(1800, ceil(calculationFrames / 50) + 300)``.  The 12 fixtures above 135k
#: calculation frames need 2,195-2,439 s at the measured 61.5-100 rows/s, so
#: 1800 s truncated them by construction; the largest manifest fixture
#: (149,986 frames) gets 3300 s.  5400 s is the same hard cap the batch enforces
#: (``v400_retail_trace_batch.BUDGET_MAX_SECONDS``); the two are asserted equal by
#: the unit tests.  This constant only bounds the argument -- no capture semantics
#: changed.
MAX_BUDGET_SECONDS = 5400.0

#: EXP-P0-075 (v4.0.0 WP1): ``--budget-seconds`` used to be sampled *only* by the
#: phase driver loop, so an observer whose main thread blocked inside a Win32 call
#: (foreground transfer, ``WaitForDebugEvent``, ``ReadProcessMemory``, a replay
#: menu that never advances) ignored its own budget: fixture ``easy-2-b`` sat in
#: ``phase="attached"`` for 39 min with ``rows.jsonl`` at 0 bytes while the batch
#: waited.  The watchdog below is a daemon timer thread that fires no matter what
#: the main thread is doing.  It fires ``WATCHDOG_GRACE_SECONDS`` *after* the
#: budget so a healthy run can still leave through the graceful path (the driver
#: loop logs "budget exhausted" and grades the partial trace); the grace only
#: costs time when the main thread is actually wedged, which is the case the
#: watchdog exists for.
WATCHDOG_GRACE_SECONDS = 5.0

#: EXP-P0-075: playback must start within this many seconds or the run fails fast
#: (``status="failed"``, exit 6) instead of pumping a stuck title screen until the
#: budget runs out.  The navigator itself is bounded at ~80 s (40 s closed loop +
#: blind sequence + 35 s); this is the driver-side backstop for a navigation that
#: is alive but will never succeed.
NAV_DEADLINE_SECONDS = 90.0

#: EXP-P0-073 (v4.0.0 WP1): only the *menu* needs the foreground window.
#: ``SendInput`` scan codes reach TH08 only while its window is foreground
#: (``PostMessage`` is inert on this engine, EXP-P0-053), but replay playback and
#: the breakpoint reads never read the keyboard.  A second instance can therefore
#: boot while the first one plays, as long as the two menu navigations never
#: overlap.  ``--nav-lock`` is a cross-process byte-range lock that serialises
#: exactly that window-owning phase; ``--allow-existing-instances`` lifts the
#: single-instance refusal for the parallel pipeline only.
STATUS_NAME = "status.json"
DEFAULT_NAV_LOCK_TIMEOUT = 300.0

# ---------------------------------------------------------------------------
# EXP-P0-074 (v4.0.0 WP1): per-row read windows
# ---------------------------------------------------------------------------
#: The row needs 12 separate values and the official observer reads each one with
#: its own ``ReadProcessMemory``; here they are grouped into the smallest number
#: of *contiguous* windows that still yields byte-identical bytes.
#:
#:   GM+8        g_GameManager.globals (4 B)   \
#:   GM+0x30     difficulty (4 B)              /  -> GM+8 .. GM+0x34  (44 B)
#:
#:   GM+0x3DBA8  the 0x220-byte game-manager block used by the row  \
#:   RNG         rng seed + generation (8 B)                        |  -> one
#:   INPUT       input word (2 B)                                   /      1150 B
#:
#:   PLAYER      character/shot byte (PLAYER+0)                \
#:   PLAYER+0x2B4  x/y/z tail (12 B)                           / -> PLAYER .. +0x2C0
#:
#: The three windows are derived from the read addresses themselves (not written
#: as literals) so they can never drift from ``GM``/``RNG``/``INPUT``/``PLAYER``.
WINDOW_OFFSET = 0x3DBA8                 # replay-manager block inside the GM block
GM_HEAD_OFFSET = 8                      # g_GameManager.globals
GM_HEAD_BYTES = 0x30 - GM_HEAD_OFFSET + 4          # 0x2C: globals .. difficulty
RNG_IN_WINDOW = RNG - GM - WINDOW_OFFSET           # 0x470
INPUT_IN_WINDOW = INPUT - GM - WINDOW_OFFSET       # 0x47C
WINDOW_BYTES = INPUT_IN_WINDOW + 2                 # 0x47E
PLAYER_TAIL_OFFSET = 0x2B4
PLAYER_TAIL_BYTES = 12
PLAYER_BYTES = PLAYER_TAIL_OFFSET + PLAYER_TAIL_BYTES   # 0x2C0
GM_BLOCK_BYTES = 0x220

#: rows are joined in memory and written in batches so one row costs no syscall.
#: ``1024`` rows is ~160 KiB at the observed 154 B/row.
ROW_BATCH = 1024

# ---------------------------------------------------------------------------
# Win32 / Debug API bindings
# ---------------------------------------------------------------------------
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32 = ctypes.WinDLL("user32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

DWORD = wt.DWORD
HANDLE = wt.HANDLE
BOOL = wt.BOOL
ULONG_PTR = ctypes.c_size_t

#: EXP-P0-074: the buffered row reader calls ``ReadProcessMemory`` itself (the
#: same kernel32 entry ``touhou_rl.memory.ProcessMemory.read`` uses), so the
#: prototype is pinned here on this module's own ``WinDLL`` handle.
kernel32.ReadProcessMemory.argtypes = [
    HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.ReadProcessMemory.restype = BOOL


EXCEPTION_DEBUG_EVENT = 1
CREATE_THREAD_DEBUG_EVENT = 2
CREATE_PROCESS_DEBUG_EVENT = 3
EXIT_THREAD_DEBUG_EVENT = 4
EXIT_PROCESS_DEBUG_EVENT = 5
LOAD_DLL_DEBUG_EVENT = 6
UNLOAD_DLL_DEBUG_EVENT = 7
OUTPUT_DEBUG_STRING_EVENT = 8
RIP_EVENT = 9

EXCEPTION_BREAKPOINT = 0x80000003
EXCEPTION_SINGLE_STEP = 0x80000004
#: A 32-bit process under WOW64 reports its hardware-breakpoint trap to a 64-bit
#: debugger with the WOW64 status codes, not the native ones (EXP-P0-068: the
#: DR0 hit at 0x452490 arrived as 0x4000001E and killed the game for as long as
#: it was treated as an unhandled application exception).
STATUS_WX86_SINGLE_STEP = 0x4000001E
STATUS_WX86_BREAKPOINT = 0x4000001F
SINGLE_STEP_CODES = (EXCEPTION_SINGLE_STEP, STATUS_WX86_SINGLE_STEP)
BREAKPOINT_CODES = (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT)

DBG_CONTINUE = 0x00010002
DBG_EXCEPTION_NOT_HANDLED = 0x80010001

#: EFLAGS.RF.  A native hardware execute breakpoint is resumed past the trapping
#: instruction by the kernel setting RF; the WOW64 path does not, so the debugger
#: has to set it or the instruction re-traps forever (EXP-P0-068 POC run 2: five
#: hits at 0x452490 within 1 ms, all reporting gameFrame=1).
EFLAGS_RF = 0x00010000

WOW64_CONTEXT_i386 = 0x00010000
WOW64_CONTEXT_CONTROL = WOW64_CONTEXT_i386 | 0x00000001
WOW64_CONTEXT_DEBUG_REGISTERS = WOW64_CONTEXT_i386 | 0x00000010

#: DR7 bit 10 is documented "reserved, must be 1"; the local-enable bits of the
#: armed slots are OR-ed in by :func:`dr7_for`.
DR7_RESERVED = 0x00000400


class FLOATING_SAVE_AREA(ctypes.Structure):
    _fields_ = [
        ("ControlWord", DWORD),
        ("StatusWord", DWORD),
        ("TagWord", DWORD),
        ("ErrorOffset", DWORD),
        ("ErrorSelector", DWORD),
        ("DataOffset", DWORD),
        ("DataSelector", DWORD),
        ("RegisterArea", ctypes.c_ubyte * 80),
        ("Cr0NpxState", DWORD),
    ]


class WOW64_CONTEXT(ctypes.Structure):
    """x86 ``CONTEXT`` (COFF layout); 716 bytes, see csr CaptureContext."""

    _fields_ = [
        ("ContextFlags", DWORD),
        ("Dr0", DWORD),
        ("Dr1", DWORD),
        ("Dr2", DWORD),
        ("Dr3", DWORD),
        ("Dr6", DWORD),
        ("Dr7", DWORD),
        ("FloatSave", FLOATING_SAVE_AREA),
        ("SegGs", DWORD),
        ("SegFs", DWORD),
        ("SegEs", DWORD),
        ("SegDs", DWORD),
        ("Edi", DWORD),
        ("Esi", DWORD),
        ("Ebx", DWORD),
        ("Edx", DWORD),
        ("Ecx", DWORD),
        ("Eax", DWORD),
        ("Ebp", DWORD),
        ("Eip", DWORD),
        ("SegCs", DWORD),
        ("EFlags", DWORD),
        ("Esp", DWORD),
        ("SegSs", DWORD),
        ("ExtendedRegisters", ctypes.c_ubyte * 512),
    ]


class EXCEPTION_RECORD(ctypes.Structure):
    pass


EXCEPTION_RECORD._fields_ = [
    ("ExceptionCode", DWORD),
    ("ExceptionFlags", DWORD),
    ("ExceptionRecord", ctypes.POINTER(EXCEPTION_RECORD)),
    ("ExceptionAddress", ctypes.c_void_p),
    ("NumberParameters", DWORD),
    ("ExceptionInformation", ULONG_PTR * 15),
]


class EXCEPTION_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("ExceptionRecord", EXCEPTION_RECORD), ("dwFirstChance", DWORD)]


class CREATE_THREAD_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("hThread", HANDLE),
        ("lpThreadLocalBase", ctypes.c_void_p),
        ("lpStartAddress", ctypes.c_void_p),
    ]


class CREATE_PROCESS_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("hFile", HANDLE),
        ("hProcess", HANDLE),
        ("hThread", HANDLE),
        ("lpBaseOfImage", ctypes.c_void_p),
        ("dwDebugInfoFileOffset", DWORD),
        ("nDebugInfoSize", DWORD),
        ("lpThreadLocalBase", ctypes.c_void_p),
        ("lpStartAddress", ctypes.c_void_p),
        ("lpImageName", ctypes.c_void_p),
        ("fUnicode", wt.WORD),
    ]


class LOAD_DLL_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("hFile", HANDLE),
        ("lpBaseOfDll", ctypes.c_void_p),
        ("dwDebugInfoFileOffset", DWORD),
        ("nDebugInfoSize", DWORD),
        ("lpImageName", ctypes.c_void_p),
        ("fUnicode", wt.WORD),
    ]


class EXIT_THREAD_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("dwExitCode", DWORD)]


class EXIT_PROCESS_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("dwExitCode", DWORD)]


class UNLOAD_DLL_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("lpBaseOfDll", ctypes.c_void_p)]


class OUTPUT_DEBUG_STRING_INFO(ctypes.Structure):
    _fields_ = [
        ("lpDebugStringData", ctypes.c_void_p),
        ("fUnicode", wt.WORD),
        ("nDebugStringLength", wt.WORD),
    ]


class RIP_INFO(ctypes.Structure):
    _fields_ = [("dwError", DWORD), ("dwType", DWORD)]


class _DEBUG_EVENT_UNION(ctypes.Union):
    _fields_ = [
        ("Exception", EXCEPTION_DEBUG_INFO),
        ("CreateThread", CREATE_THREAD_DEBUG_INFO),
        ("CreateProcessInfo", CREATE_PROCESS_DEBUG_INFO),
        ("ExitThread", EXIT_THREAD_DEBUG_INFO),
        ("ExitProcess", EXIT_PROCESS_DEBUG_INFO),
        ("LoadDll", LOAD_DLL_DEBUG_INFO),
        ("UnloadDll", UNLOAD_DLL_DEBUG_INFO),
        ("DebugString", OUTPUT_DEBUG_STRING_INFO),
        ("RipInfo", RIP_INFO),
    ]


class DEBUG_EVENT(ctypes.Structure):
    _fields_ = [
        ("dwDebugEventCode", DWORD),
        ("dwProcessId", DWORD),
        ("dwThreadId", DWORD),
        ("u", _DEBUG_EVENT_UNION),
    ]


kernel32.DebugActiveProcess.argtypes = [DWORD]
kernel32.DebugActiveProcess.restype = BOOL
kernel32.DebugActiveProcessStop.argtypes = [DWORD]
kernel32.DebugActiveProcessStop.restype = BOOL
kernel32.WaitForDebugEvent.argtypes = [ctypes.POINTER(DEBUG_EVENT), DWORD]
kernel32.WaitForDebugEvent.restype = BOOL
kernel32.ContinueDebugEvent.argtypes = [DWORD, DWORD, DWORD]
kernel32.ContinueDebugEvent.restype = BOOL
kernel32.CloseHandle.argtypes = [HANDLE]
kernel32.CloseHandle.restype = BOOL
kernel32.GetLastError.restype = DWORD

_WOW64_CONTEXT_SIZE = ctypes.sizeof(WOW64_CONTEXT)


def _wow64_bindings() -> tuple[object, object]:
    """``(Wow64GetThreadContext, Wow64SetThreadContext)`` from kernel32 or wow64.dll."""
    for dll in (kernel32, ctypes.WinDLL("wow64", use_last_error=True)):
        getter = getattr(dll, "Wow64GetThreadContext", None)
        setter = getattr(dll, "Wow64SetThreadContext", None)
        if getter is not None and setter is not None:
            getter.argtypes = [HANDLE, ctypes.POINTER(WOW64_CONTEXT)]
            getter.restype = BOOL
            setter.argtypes = [HANDLE, ctypes.POINTER(WOW64_CONTEXT)]
            setter.restype = BOOL
            return getter, setter
    raise RuntimeError("Wow64GetThreadContext/Wow64SetThreadContext are unavailable")


def dr7_for(count: int) -> int:
    """DR7 for ``count`` execute breakpoints in DR0..DR3 (R/W=00, LEN=00)."""
    if not 0 <= count <= 4:
        raise ValueError(f"cannot arm {count} hardware breakpoints; x86 has four slots")
    value = DR7_RESERVED
    for index in range(count):
        value |= 1 << (2 * index)
    return value


def enable_debug_privilege() -> dict:
    """Best-effort ``SeDebugPrivilege``; the result is recorded, never assumed."""
    TOKEN_ADJUST_PRIVILEGES = 0x0020
    TOKEN_QUERY = 0x0008
    SE_PRIVILEGE_ENABLED = 0x00000002

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", DWORD), ("HighPart", wt.LONG)]

    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", DWORD)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [
            ("PrivilegeCount", DWORD),
            ("Privileges", LUID_AND_ATTRIBUTES * 1),
        ]

    kernel32.GetCurrentProcess.restype = HANDLE
    advapi32.OpenProcessToken.argtypes = [HANDLE, DWORD, ctypes.POINTER(HANDLE)]
    advapi32.OpenProcessToken.restype = BOOL
    advapi32.LookupPrivilegeValueW.argtypes = [wt.LPCWSTR, wt.LPCWSTR, ctypes.POINTER(LUID)]
    advapi32.LookupPrivilegeValueW.restype = BOOL
    advapi32.AdjustTokenPrivileges.argtypes = [
        HANDLE, BOOL, ctypes.POINTER(TOKEN_PRIVILEGES), DWORD, ctypes.c_void_p, ctypes.c_void_p,
    ]
    advapi32.AdjustTokenPrivileges.restype = BOOL

    token = HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, ctypes.byref(token)
    ):
        return {"ok": False, "error": f"OpenProcessToken err={ctypes.get_last_error()}"}
    try:
        luid = LUID()
        if not advapi32.LookupPrivilegeValueW(None, "SeDebugPrivilege", ctypes.byref(luid)):
            return {"ok": False, "error": f"LookupPrivilegeValue err={ctypes.get_last_error()}"}
        privileges = TOKEN_PRIVILEGES()
        privileges.PrivilegeCount = 1
        privileges.Privileges[0].Luid = luid
        privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED
        if not advapi32.AdjustTokenPrivileges(
            token, False, ctypes.byref(privileges), 0, None, None
        ):
            return {"ok": False, "error": f"AdjustTokenPrivileges err={ctypes.get_last_error()}"}
        error = ctypes.get_last_error()
        if error == 1300:  # ERROR_NOT_ALL_ASSIGNED
            return {"ok": False, "error": "SeDebugPrivilege not held by this token (err=1300)"}
        return {"ok": True, "enabled": "SeDebugPrivilege"}
    finally:
        kernel32.CloseHandle(token)


# ---------------------------------------------------------------------------
# pure reducers / identity helpers (unit-tested without a game)
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_comparison_module():
    """Import the official ``compare-replay-traces.py`` read-only.

    ``retailTrace.sha256`` is *not* the SHA-256 of ``rows.jsonl``: the official
    pipeline defines it in ``compare-replay-traces.trace_sha256`` as the hash of
    the compact schema JSON followed by every row packed as 25 little-endian
    uint32 values ("independent of JSON spacing").  Importing the official
    function is the only way to compare against that field without
    re-implementing (and possibly re-interpreting) it.
    """
    import importlib.util

    scripts = OFFICIAL_COMPARISON_PY.parent
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "th08_compare_replay_traces", OFFICIAL_COMPARISON_PY
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_trace_sha256(rows_path: Path) -> str:
    """The official ``retailTrace.sha256`` of a captured ``rows.jsonl``."""
    comparison = load_comparison_module()
    schema = json.loads(OFFICIAL_SCHEMA_JSON.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(rows_path).read_text(encoding="utf-8").splitlines()]
    return comparison.trace_sha256({"schema": schema}, rows)


def retail_identity(path: Path = RETAIL_EXE) -> dict:
    """Size + SHA-256 + fixed image base of the retail image (no launch)."""
    path = Path(path)
    info: dict = {"path": str(path), "present": path.is_file()}
    if not info["present"]:
        info.update(verified=False, reason="retail image missing")
        return info
    info["size"] = path.stat().st_size
    info["sha256"] = sha256_file(path)
    info["verified"] = info["size"] == RETAIL_SIZE and info["sha256"] == RETAIL_SHA256
    if not info["verified"]:
        info["reason"] = "size/SHA256 mismatch: refusing to run a reconstruction as retail"
    raw = path.read_bytes()[:0x400]
    offset = struct.unpack_from("<I", raw, 0x3C)[0]
    optional = offset + 24
    info["imageBase"] = hex(struct.unpack_from("<I", raw, optional + 28)[0])
    info["dllCharacteristics"] = hex(struct.unpack_from("<H", raw, optional + 70)[0])
    info["dynamicBase"] = bool(struct.unpack_from("<H", raw, optional + 70)[0] & 0x0040)
    return info


def load_replay_module():
    """Import ``third_party/th08/scripts/replay_file.py`` read-only."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("th08_replay_file", REPLAY_FILE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_expectations(replay_path: Path, replay_sha256: str | None = None) -> dict:
    """``retailTrace`` block of the matching fixture in ``replay-fixtures.json``."""
    replay_path = Path(replay_path)
    data = json.loads(FIXTURES_JSON.read_text(encoding="utf-8"))
    digest = replay_sha256 if replay_sha256 is not None else sha256_file(replay_path)
    for fixture in data.get("fixtures", []):
        if fixture.get("file") != replay_path.name:
            continue
        if fixture.get("sha256") and fixture["sha256"] != digest:
            continue
        trace = fixture.get("retailTrace") or {}
        return {
            "fixtureId": fixture.get("id"),
            "file": fixture.get("file"),
            "sha256": fixture.get("sha256"),
            "calculationFrames": trace.get("calculationFrames"),
            "retailTraceSha256": trace.get("sha256"),
            "schemaVersion": trace.get("schemaVersion"),
            "stages": [
                {key: stage.get(key) for key in ("index", "name", "inputRecords", "endScore")}
                for stage in fixture.get("stages", [])
            ],
        }
    return {"file": replay_path.name, "sha256": digest, "matched": False,
            "reason": "no fixture in replay-fixtures.json has this file + sha256"}


def compose_row(
    *,
    calc: int,
    stage: int,
    frame: int,
    gm: bytes,
    globals_data: bytes,
    player_tail: bytes,
    player_first: int,
    rng: bytes,
    rng_origin: int,
    input_value: int,
    gm_plus_30: int,
) -> list:
    """The official 25-field row, in the official order (observer:201-208)."""
    if len(gm) < 0x220:
        raise ValueError(f"game manager block is {len(gm)} B, need 0x220")
    if len(globals_data) < 0xA0:
        raise ValueError(f"globals block is {len(globals_data)} B, need 0xa0")
    if len(player_tail) < 12:
        raise ValueError(f"player tail is {len(player_tail)} B, need 12")
    if len(rng) < 8:
        raise ValueError(f"rng block is {len(rng)} B, need 8")

    def u32(offset: int) -> int:
        return struct.unpack_from("<I", globals_data, offset)[0]

    return [
        calc,
        stage,
        frame,
        struct.unpack_from("<I", gm, 0x218)[0],
        u32(8),
        u32(12),
        *[u32(offset) for offset in (0x64, 0x74, 0x80, 0x98)],
        *struct.unpack("<3I", player_tail),
        player_first,
        struct.unpack_from("<H", rng)[0],
        (struct.unpack_from("<I", rng, 4)[0] - rng_origin) & 0xFFFFFFFF,
        input_value,
        u32(0x30),
        u32(0x3C),
        struct.unpack_from("<h", globals_data, 0x22)[0] & 0xFFFFFFFF,
        globals_data[0x28],
        u32(0x24),
        gm[0],
        gm[1],
        gm_plus_30,
    ]


def row_line(row: list) -> str:
    """The exact line the official observer writes: ``json.dumps(row) + "\\n"``."""
    return json.dumps(row) + "\n"


def menu_key(screen: int, substate: int, cursor: int, state: int, sub_timer: int,
             start_stage: int, playback_mode: int) -> str | None:
    """The official ``Observer.menu`` decision, without the key injection."""
    if state or sub_timer < 12:
        return None
    if screen == 0 and substate == 1:
        return "z" if cursor == 4 else "down"
    if screen == 7:
        if substate == 1:
            return "z"
        if substate == 2:
            return "z" if cursor == start_stage else "down"
        if substate == 3:
            return "z" if cursor == playback_mode else "down"
    return None


def block_reader(mem) -> callable:  # noqa: ANN001 - duck-typed ProcessMemory
    """``block(address, size)`` over a ``ProcessMemory`` (mirrors the observer)."""

    def block(address: int, size: int) -> bytes:
        return mem.read(address, size)

    return block


def read32(block, address: int) -> int:  # noqa: ANN001
    return struct.unpack("<I", block(address, 4))[0]


def format_row_for_log(row: list) -> dict:
    return dict(zip(FIELDS, row))


class WindowReader:
    """``ReadProcessMemory`` over reused ctypes buffers (EXP-P0-074).

    ``touhou_rl.memory.ProcessMemory.read`` allocates a fresh
    ``create_string_buffer`` per call and returns ``buffer.raw``. The row path
    below issues the same reads every frame, so the buffers are allocated once
    per size and reused; the returned ``bytes`` copy keeps the previous read's
    contents intact, which is what the row builder slices.
    """

    __slots__ = ("handle", "read_process_memory", "buffers", "bytes_read", "reads")

    def __init__(self, handle: int, reader=None) -> None:  # noqa: ANN001
        self.handle = int(handle)
        if not self.handle:
            raise OSError("no process handle for a buffered row read")
        self.read_process_memory = reader or kernel32.ReadProcessMemory
        self.buffers: dict[int, ctypes.Array] = {}
        self.bytes_read = ctypes.c_size_t()
        self.reads = 0

    def buffer(self, size: int) -> ctypes.Array:
        buffer = self.buffers.get(size)
        if buffer is None:
            buffer = ctypes.create_string_buffer(size)
            self.buffers[size] = buffer
        return buffer

    def read(self, address: int, size: int) -> bytes:
        buffer = self.buffer(size)
        got = self.bytes_read
        ok = self.read_process_memory(
            self.handle, ctypes.c_void_p(address), buffer, size, ctypes.byref(got)
        )
        if not ok or got.value != size:
            raise OSError(
                f"ReadProcessMemory 0x{address:08X}+{size} failed "
                f"(read {got.value}/{size}, err={ctypes.get_last_error()})"
            )
        self.reads += 1
        return buffer.raw


# ---------------------------------------------------------------------------
# native debugger
# ---------------------------------------------------------------------------
class DebuggerError(RuntimeError):
    def __init__(self, message: str, exit_code: int = EXIT_BLOCKED):
        super().__init__(message)
        self.exit_code = exit_code


class NativeDebugger:
    """Attach with ``DebugActiveProcess`` and arm DR0..DR3 execute breakpoints."""

    def __init__(self, pid: int, breakpoints: dict[str, int], mem, log, on_hit=None,
                 dr7: int | None = None):
        self.pid = int(pid)
        self.breakpoints = dict(breakpoints)
        self.by_address = {address: name for name, address in self.breakpoints.items()}
        self.mem = mem
        self.log = log
        self.on_hit = on_hit
        order = list(self.breakpoints)
        self.dr_addresses = [self.breakpoints[name] for name in order]
        self.dr7 = dr7 if dr7 is not None else dr7_for(len(self.dr_addresses))
        self.hprocess = 0
        self.threads: dict[int, int] = {}
        self.hits = {name: 0 for name in self.breakpoints}
        self.first_hits: list[dict] = []
        self.hit_times: dict[str, float] = {}
        self.armed_threads = 0
        self.armed_details: list[dict] = []
        self.unexpected_steps = 0
        self.unexpected_exceptions: list[dict] = []
        self.resume_flags_set = 0
        self.resume_flag_failures = 0
        self.other_events = {"dll": 0, "thread": 0, "exception": 0, "rip": 0}
        self.attach_error: str | None = None
        self.exited: int | None = None
        self.attached = False
        self.last_event_at = 0.0
        self._getter = None
        self._setter = None
        self.events_handled = 0
        # EXP-P0-074: two breakpoint hits per row means the DEBUG_EVENT and the
        # WOW64_CONTEXT would otherwise be allocated ~400k times per fixture.
        # Both are pure out-parameters: the kernel overwrites the event, and the
        # context is refilled by the getter before every use.
        self._event = DEBUG_EVENT()
        self._hit_context = WOW64_CONTEXT()

    # -- attach / detach ---------------------------------------------------
    def attach(self) -> None:
        self._getter, self._setter = _wow64_bindings()
        if not kernel32.DebugActiveProcess(self.pid):
            error = ctypes.get_last_error()
            self.attach_error = f"DebugActiveProcess({self.pid}) failed err={error}"
            raise DebuggerError(self.attach_error)
        self.attached = True
        self.last_event_at = time.perf_counter()

    def detach(self) -> None:
        """Clear the debug registers and detach; never leave the game frozen."""
        self.clear_breakpoints()
        # A pending debug event would make DebugActiveProcessStop fail.
        drained = 0
        while drained < 64:
            event = self.pump(0)
            if event is None:
                break
            drained += 1
        if self.attached:
            ok = kernel32.DebugActiveProcessStop(self.pid)
            if not ok:
                self.log(f"DebugActiveProcessStop failed err={ctypes.get_last_error()}")
            self.attached = False
        for handle in list(self.threads.values()):
            if handle:
                kernel32.CloseHandle(handle)
        self.threads.clear()
        if self.hprocess:
            kernel32.CloseHandle(self.hprocess)
            self.hprocess = 0

    def hard_stop(self) -> dict:
        """Timeout path: clear DR0..DR3 and stop the debug session, no pumping.

        ``detach`` drains pending events through ``pump``, which is unsafe when
        the main thread may be *inside* ``pump`` -- exactly the situation this
        method exists for.  Handles are deliberately left open: the caller is
        the watchdog and it terminates the process right afterwards, so closing
        a handle the wedged main thread may still be using would only add a race.
        """
        cleared = self.clear_breakpoints()
        stopped = False
        error = None
        if self.attached:
            try:
                stopped = bool(kernel32.DebugActiveProcessStop(self.pid))
                if not stopped:
                    error = f"DebugActiveProcessStop err={ctypes.get_last_error()}"
            except OSError as exc:  # pragma: no cover - defensive
                error = str(exc)
            self.attached = False
        return {"clearedThreads": cleared, "debugActiveProcessStop": stopped,
                "error": error}

    # -- debug registers ---------------------------------------------------
    def _context(self, handle: int, flags: int,
                 context: WOW64_CONTEXT | None = None) -> WOW64_CONTEXT:
        if context is None:
            context = WOW64_CONTEXT()
        context.ContextFlags = flags
        if not self._getter(handle, ctypes.byref(context)):
            raise OSError(f"Wow64GetThreadContext failed err={ctypes.get_last_error()}")
        return context

    def arm_thread(self, tid: int, handle: int) -> dict | None:
        """Install the execute breakpoints into one thread's DR0..DR3."""
        try:
            context = self._context(handle, WOW64_CONTEXT_DEBUG_REGISTERS)
        except OSError as exc:
            return {"tid": tid, "ok": False, "error": str(exc)}
        values = list(self.dr_addresses) + [0] * (4 - len(self.dr_addresses))
        context.Dr0, context.Dr1, context.Dr2, context.Dr3 = values[:4]
        context.Dr6 = 0
        context.Dr7 = self.dr7
        ok = self._setter(handle, ctypes.byref(context))
        if not ok:
            # Some builds want CONTEXT_CONTROL as well; retry once with a full
            # control+debug-register context read/modify/write.
            try:
                context = self._context(handle, WOW64_CONTEXT_DEBUG_REGISTERS | WOW64_CONTEXT_CONTROL)
                context.Dr0, context.Dr1, context.Dr2, context.Dr3 = values[:4]
                context.Dr6 = 0
                context.Dr7 = self.dr7
                ok = self._setter(handle, ctypes.byref(context))
            except OSError as exc:
                return {"tid": tid, "ok": False, "error": str(exc)}
        if not ok:
            return {"tid": tid, "ok": False, "error": f"Wow64SetThreadContext err={ctypes.get_last_error()}"}
        back = self._context(handle, WOW64_CONTEXT_DEBUG_REGISTERS)
        detail = {
            "tid": tid,
            "ok": True,
            "dr0": hex(back.Dr0),
            "dr1": hex(back.Dr1),
            "dr2": hex(back.Dr2),
            "dr3": hex(back.Dr3),
            "dr6": hex(back.Dr6),
            "dr7": hex(back.Dr7),
            "dr7MatchesRequest": back.Dr7 == self.dr7,
            "addressesMatch": [back.Dr0, back.Dr1, back.Dr2, back.Dr3][: len(self.dr_addresses)]
            == self.dr_addresses[: len(self.dr_addresses)],
        }
        self.armed_threads += 1
        self.armed_details.append(detail)
        return detail

    def clear_breakpoints(self) -> int:
        if self._getter is None:
            return 0
        cleared = 0
        for tid, handle in list(self.threads.items()):
            try:
                context = self._context(handle, WOW64_CONTEXT_DEBUG_REGISTERS)
                context.Dr0 = context.Dr1 = context.Dr2 = context.Dr3 = 0
                context.Dr6 = 0
                context.Dr7 = 0
                if self._setter(handle, ctypes.byref(context)):
                    cleared += 1
            except OSError:
                continue
        return cleared

    # -- event loop --------------------------------------------------------
    def pump(self, timeout_ms: int) -> int | None:
        event = self._event
        if not kernel32.WaitForDebugEvent(ctypes.byref(event), int(timeout_ms)):
            return None
        status = DBG_CONTINUE
        try:
            status = self._handle(event)
        finally:
            if not kernel32.ContinueDebugEvent(event.dwProcessId, event.dwThreadId, status):
                self.log(f"ContinueDebugEvent failed err={ctypes.get_last_error()}")
        self.events_handled += 1
        self.last_event_at = time.perf_counter()
        return event.dwDebugEventCode

    def _handle(self, event: DEBUG_EVENT) -> int:
        code = event.dwDebugEventCode
        if code == CREATE_PROCESS_DEBUG_EVENT:
            info = event.u.CreateProcessInfo
            if info.hFile:
                kernel32.CloseHandle(info.hFile)
            self.hprocess = info.hProcess
            if info.hThread:
                self.threads[event.dwThreadId] = info.hThread
                self.arm_thread(event.dwThreadId, info.hThread)
            return DBG_CONTINUE
        if code == CREATE_THREAD_DEBUG_EVENT:
            info = event.u.CreateThread
            self.other_events["thread"] += 1
            if info.hThread:
                self.threads[event.dwThreadId] = info.hThread
                self.arm_thread(event.dwThreadId, info.hThread)
            return DBG_CONTINUE
        if code == EXIT_THREAD_DEBUG_EVENT:
            handle = self.threads.pop(event.dwThreadId, None)
            if handle:
                kernel32.CloseHandle(handle)
            return DBG_CONTINUE
        if code == EXIT_PROCESS_DEBUG_EVENT:
            self.exited = event.u.ExitProcess.dwExitCode
            return DBG_CONTINUE
        if code == LOAD_DLL_DEBUG_EVENT:
            self.other_events["dll"] += 1
            if event.u.LoadDll.hFile:
                kernel32.CloseHandle(event.u.LoadDll.hFile)
            return DBG_CONTINUE
        if code in (UNLOAD_DLL_DEBUG_EVENT, OUTPUT_DEBUG_STRING_EVENT):
            return DBG_CONTINUE
        if code == RIP_EVENT:
            self.other_events["rip"] += 1
            return DBG_CONTINUE
        if code == EXCEPTION_DEBUG_EVENT:
            return self._handle_exception(event)
        return DBG_CONTINUE

    def _handle_exception(self, event: DEBUG_EVENT) -> int:
        record = event.u.Exception.ExceptionRecord
        self.other_events["exception"] += 1
        code = record.ExceptionCode
        if code in BREAKPOINT_CODES:
            # The initial break-in from DebugActiveProcess (and any real int3).
            return DBG_CONTINUE
        if code not in SINGLE_STEP_CODES:
            if not event.u.Exception.dwFirstChance:
                self.log(f"second-chance exception 0x{code:08X} at 0x{(record.ExceptionAddress or 0):X}")
            if len(self.unexpected_exceptions) < 8:
                self.unexpected_exceptions.append(
                    {"code": f"0x{code:08X}",
                     "address": f"0x{(record.ExceptionAddress or 0):X}",
                     "firstChance": bool(event.u.Exception.dwFirstChance)}
                )
            return DBG_EXCEPTION_NOT_HANDLED
        handle = self.threads.get(event.dwThreadId, 0)
        if not handle:
            self.unexpected_steps += 1
            return DBG_CONTINUE
        try:
            context = self._context(handle, WOW64_CONTEXT_DEBUG_REGISTERS | WOW64_CONTEXT_CONTROL,
                                    self._hit_context)
        except OSError as exc:
            self.unexpected_steps += 1
            self.log(f"single-step without a context: {exc}")
            return DBG_CONTINUE
        eip = context.Eip
        name = self.by_address.get(eip)
        now = time.perf_counter()
        if name is None:
            self.unexpected_steps += 1
            if self.unexpected_steps <= 5:
                self.log(f"single-step at unarmed 0x{eip:08X} dr6=0x{context.Dr6:08X}")
            return DBG_CONTINUE
        # Resume past the trapping instruction exactly once.
        context.EFlags |= EFLAGS_RF
        if self._setter(handle, ctypes.byref(context)):
            self.resume_flags_set += 1
        else:
            self.resume_flag_failures += 1
        self.hits[name] += 1
        if name not in self.hit_times:
            self.hit_times[name] = now
        if len(self.first_hits) < 24:
            self.first_hits.append(
                {
                    "t": round(now, 4),
                    "breakpoint": name,
                    "address": f"0x{eip:08X}",
                    "expected": f"0x{self.breakpoints[name]:08X}",
                    "tid": event.dwThreadId,
                    "dr6": f"0x{context.Dr6:08X}",
                    "eip": f"0x{context.Eip:08X}",
                    "esp": f"0x{context.Esp:08X}",
                    "hitIndex": self.hits[name],
                }
            )
        if self.on_hit is not None:
            try:
                self.on_hit(name, context, event.dwThreadId)
            except Exception as exc:  # noqa: BLE001 - a handler bug must not freeze the game
                self.log(f"on_hit({name}) raised {type(exc).__name__}: {exc}")
        return DBG_CONTINUE

    # -- convenience -------------------------------------------------------
    def hit_summary(self) -> dict:
        order = sorted(self.hit_times.items(), key=lambda item: item[1])
        return {
            "hits": dict(self.hits),
            "totalHits": sum(self.hits.values()),
            "firstHitOrder": [name for name, _t in order],
            "hitTimesSeconds": {name: round(t - (order[0][1] if order else 0.0), 4)
                                for name, t in order},
            "armedThreads": self.armed_threads,
            "armedDetails": self.armed_details,
            "dr7Requested": hex(self.dr7),
            "singleStepUnarmed": self.unexpected_steps,
            "resumeFlagSet": self.resume_flags_set,
            "resumeFlagFailures": self.resume_flag_failures,
            "unexpectedExceptions": self.unexpected_exceptions,
            "eventsHandled": self.events_handled,
            "otherEvents": dict(self.other_events),
            "processExited": self.exited,
        }


# ---------------------------------------------------------------------------
# game session helpers
# ---------------------------------------------------------------------------
def _load_touhou_rl():
    """Import ``touhou_rl`` with the retail address table active."""
    addresses = ROOT / ".tmp_v400_retail_win_addresses.json"
    if addresses.is_file():
        os.environ.setdefault("TH08_ADDRESS_FILE", str(addresses))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from touhou_rl import memory, winapi

    return memory, winapi


def assemble_run_dir(run_dir: Path, skip_score_dat: bool = True) -> dict:
    """Copy the retail image plus its assets into a private directory."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "replay").mkdir(exist_ok=True)
    plan = [("th08.exe", RETAIL_EXE)]
    for name in ASSET_FILES:
        if name == "score.dat" and skip_score_dat:
            continue
        plan.append((name, ASSET_DIR / name))
    copied = []
    for name, source in plan:
        if not source.is_file():
            raise FileNotFoundError(f"missing source asset {source}")
        target = run_dir / name
        if target.exists() and target.stat().st_size == source.stat().st_size:
            copied.append({"name": name, "action": "reused", "bytes": target.stat().st_size})
            continue
        shutil.copy2(source, target)
        copied.append({"name": name, "action": "copied", "bytes": target.stat().st_size})
    return {"runDir": str(run_dir), "files": copied, "scoreDatSeeded": not skip_score_dat}


def install_replay(run_dir: Path, replay_path: Path, index: int = 1) -> Path:
    replay_dir = Path(run_dir) / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    for existing in replay_dir.glob("th8_*.rpy"):
        existing.unlink()
    target = replay_dir / f"th8_{index:02d}.rpy"
    target.write_bytes(Path(replay_path).read_bytes())
    return target


def wait_for_window(winapi, pid: int, timeout: float = 30.0) -> tuple[int, tuple[int, int, int, int]]:
    deadline = time.perf_counter() + timeout
    hwnd = 0
    rect = None
    stable = 0
    while time.perf_counter() < deadline:
        candidate = winapi.find_window(pid)
        if candidate:
            current = winapi.client_rect(candidate)
            if candidate == hwnd and current == rect and current[2] > 0 and current[3] > 0:
                stable += 1
                if stable >= 2:
                    return hwnd, current
            else:
                stable = 0
            hwnd, rect = candidate, current
        time.sleep(0.1)
    raise DebuggerError(f"no stable window for pid {pid} (hwnd=0x{hwnd:X} rect={rect})",
                        EXIT_TIMEOUT)


def read_attestation(mem, address: int, size: int) -> str | None:
    try:
        return mem.read(address, size).hex()
    except OSError:
        return None


def pe_rva_bytes(raw: bytes, rva: int, size: int) -> bytes | None:
    """``size`` bytes of a PE32 image at ``rva`` (section table walk, no loader)."""
    offset = struct.unpack_from("<I", raw, 0x3C)[0]
    if raw[offset:offset + 4] != b"PE\x00\x00":
        return None
    sections = struct.unpack_from("<H", raw, offset + 6)[0]
    optional = offset + 24
    size_of_headers = struct.unpack_from("<I", raw, optional + 60)[0]
    table = optional + struct.unpack_from("<H", raw, offset + 20)[0]
    for index in range(sections):
        base = table + index * 40
        virtual_size, virtual_address = struct.unpack_from("<II", raw, base + 8)
        raw_size, raw_pointer = struct.unpack_from("<II", raw, base + 16)
        extent = max(virtual_size, raw_size)
        if virtual_address <= rva < virtual_address + extent:
            start = raw_pointer + (rva - virtual_address)
            if start + size > len(raw):
                return None
            return raw[start:start + size]
    if rva + size <= size_of_headers:
        return raw[rva:rva + size]
    return None


def attest(mem, pe_path: Path) -> dict:
    """Compare the mapped bytes with the executable's bytes at the same RVAs."""
    raw = Path(pe_path).read_bytes()
    offset = struct.unpack_from("<I", raw, 0x3C)[0]
    optional = offset + 24
    image_base = struct.unpack_from("<I", raw, optional + 28)[0]
    checks = []
    ok = True
    for key, address, _bytes_key, _expected in ATTEST:
        expected_bytes = pe_rva_bytes(raw, address - image_base, 13)
        if expected_bytes is None or len(expected_bytes) != 13:
            checks.append({"address": f"0x{address:08X}", "ok": False, "reason": "unmapped RVA"})
            ok = False
            continue
        expected = expected_bytes.hex()
        seen = read_attestation(mem, address, 13)
        match = seen == expected
        ok = ok and match
        checks.append({"address": f"0x{address:08X}", "key": key, "expected": expected,
                       "mapped": seen, "ok": match})
    return {"ok": ok, "checks": checks}


# ---------------------------------------------------------------------------
# navigation (official menu state machine + blind fallback)
# ---------------------------------------------------------------------------
class Navigator(threading.Thread):
    """Drive Title -> Replay playback with the official read points."""

    def __init__(self, mem, keyboard, start_stage: int, playback_mode: int,
                 stop_event: threading.Event, log, replay_index: int = 0,
                 closed_timeout: float = 40.0):
        super().__init__(daemon=True)
        self.mem = mem
        self.keyboard = keyboard
        self.start_stage = int(start_stage)
        self.playback_mode = int(playback_mode)
        self.stop_event = stop_event
        self.log = log
        self.replay_index = int(replay_index)
        self.closed_timeout = float(closed_timeout)
        #: seconds granted to the blind key sequence before the navigator gives up
        #: (EXP-P0-075: exposed so a unit test can prove the thread always ends).
        self.blind_timeout = 35.0
        self.playback_started = False
        self.path: str | None = None
        self.menu_log: list[dict] = []
        self.key_taps = 0
        self.error: str | None = None
        self.first_flags = None
        self.stage_seen = None
        self.blind_at = None

    # -- reads -------------------------------------------------------------
    def title_state(self) -> dict | None:
        try:
            title = self.mem.u32(TITLE_SCREEN)
        except OSError:
            return None
        if not title:
            return None
        try:
            tail = self.mem.read(title + 0x14428, 16)
            head = self.mem.read(title, 20)
        except OSError:
            return None
        screen, timer, _unused, state = struct.unpack("<4I", tail)
        cursor, _a, _b, substate, sub_timer = struct.unpack("<5I", head)
        return {"pointer": title, "screen": screen, "timer": timer, "state": state,
                "cursor": cursor, "substate": substate, "subTimer": sub_timer}

    def playback_state(self) -> dict | None:
        try:
            gm = self.mem.read(GM + 0x3DBA8, 0x220)
            manager = self.mem.u32(REPLAY_MANAGER)
        except OSError:
            return None
        flags = struct.unpack_from("<I", gm, 4)[0]
        return {
            "flags": flags,
            "isReplay": bool(flags & 8),
            "isDemo": bool(flags & 2),
            "stage": struct.unpack_from("<I", gm, 0x21C)[0],
            "gameFrame": struct.unpack_from("<I", gm, 0x218)[0],
            "manager": manager,
            "frame": (self.mem.u32(manager) if manager else None),
        }

    def replay_count(self) -> int | None:
        state = self.title_state()
        if not state:
            return None
        try:
            return self.mem.u32(state["pointer"] + 0xC288)
        except OSError:
            return None

    # -- key injection -----------------------------------------------------
    def _tap(self, key: str, hold_s: float = 0.05) -> None:
        self.keyboard.set_keys({key})
        time.sleep(hold_s)
        self.keyboard.set_keys(set())
        self.key_taps += 1

    # -- thread body -------------------------------------------------------
    def run(self) -> None:
        started = time.perf_counter()
        try:
            closed_ok = self._closed_loop(self.closed_timeout)
            if not closed_ok and not self.stop_event.is_set():
                self.blind_at = round(time.perf_counter() - started, 3)
                self.log("navigation: falling back to the blind key sequence")
                self._blind()
                self._closed_loop(self.blind_timeout, allow_blind_success=True)
        except Exception as exc:  # noqa: BLE001 - navigation failure must not kill the run
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(f"navigation error: {self.error}")
        finally:
            try:
                self.keyboard.release_all()
            except OSError:
                pass

    def _closed_loop(self, timeout: float, allow_blind_success: bool = False) -> bool:
        deadline = time.perf_counter() + timeout
        last_log_signature = None
        idle_taps = 0
        while time.perf_counter() < deadline and not self.stop_event.is_set():
            playback = self.playback_state()
            if playback and playback["isReplay"] and not playback["isDemo"]:
                self.playback_started = True
                self.path = "blind+closed" if allow_blind_success and self.blind_at else "closed_loop"
                self.first_flags = playback
                self.stage_seen = playback["stage"]
                self.log(f"navigation: replay playback detected {json.dumps(playback)}")
                return True
            state = self.title_state()
            if state is None:
                # Not on a title screen: abort a running attract demo / sub-menu.
                idle_taps += 1
                if idle_taps % 40 == 0:
                    self._tap("esc", 0.05)
                    time.sleep(0.2)
                    self._tap("z", 0.05)
                time.sleep(0.05)
                continue
            signature = (state["screen"], state["substate"], state["cursor"], state["state"])
            if signature != last_log_signature:
                self.menu_log.append(
                    {
                        "t": round(time.perf_counter(), 3),
                        "screen": state["screen"],
                        "substate": state["substate"],
                        "cursor": state["cursor"],
                        "state": state["state"],
                        "timer": state["subTimer"],
                    }
                )
                last_log_signature = signature
            if state["screen"] == 7 and state["substate"] in (1, 2):
                if self.replay_count() == 0:
                    raise RuntimeError("Retail Replay menu rejected the supplied fixture")
            key = menu_key(
                state["screen"], state["substate"], state["cursor"], state["state"],
                state["subTimer"], self.start_stage, self.playback_mode,
            )
            if key:
                self._tap(key)
                time.sleep(0.2)
            else:
                time.sleep(0.03)
        return False

    def _blind(self) -> None:
        """Four Down to StartMenu/Replay, then list -> stage -> mode -> play."""
        for _ in range(4):
            if self.stop_event.is_set():
                return
            self._tap("down", 0.05)
            time.sleep(0.22)
        for _ in range(2):
            self._tap("z", 0.05)
            time.sleep(1.0)
        for _ in range(2):
            self._tap("down", 0.05)
            time.sleep(0.22)
        self._tap("z", 0.05)
        time.sleep(1.2)


# ---------------------------------------------------------------------------
# trace collector (official Observer semantics)
# ---------------------------------------------------------------------------
class TraceCollector:
    """Write one 25-field row per calculation frame, exactly like the official observer."""

    def __init__(self, mem, out_dir: Path, stages: list[dict], start_stage: int,
                 log, max_rows: int | None = None, merged_reads: bool | None = None):
        self.mem = mem
        self.out_dir = Path(out_dir)
        self.stages = {stage["index"]: stage for stage in stages}
        self.start_stage = int(start_stage)
        self.log = log
        self.max_rows = max_rows
        self.rows_path = self.out_dir / "rows.jsonl"
        self.stream = self.rows_path.open("w", encoding="utf-8", newline="\n")
        self.count = 0
        self.stage: int | None = None
        self.stage_sequence: list[int] = []
        self.stage_frames: dict[str, int] = {}
        self.stage_rows: dict[str, int] = {}
        self.end_scores: dict[str, int] = {}
        self.rng_origins: dict[str, int] = {}
        self.errors: list[str] = []
        self.finished = False
        self.stop = threading.Event()
        self.filtered = 0
        self.duplicate_frames = 0
        self.last_frame: dict[str, int] = {}
        self.sample_rows: list[dict] = []
        self.final_probe: dict | None = None
        # EXP-P0-074 throughput knobs. ``merged_reads=None`` means "decide once,
        # at the first frame, by comparing the merged windows against the
        # individual reads while the target is frozen"; ``False`` pins the
        # official per-address reads.
        self.merged_reads = merged_reads
        self.merged_checks: tuple[bool, ...] = ()
        self.merged_error: str | None = None
        self._reader: WindowReader | None = None
        self._merged: bool | None = None
        self._pending: list[str] = []
        self.rows_written = 0

    # -- official `stop(completion=False)` body ----------------------------
    def on_frame(self) -> None:
        if self.finished or self.stop.is_set():
            return
        try:
            self._on_frame()
        except Exception as exc:  # noqa: BLE001 - official `finish(str(error))`
            self.finish(f"{type(exc).__name__}: {exc}")

    def _on_frame(self) -> None:
        mem = self.mem
        manager = mem.u32(REPLAY_MANAGER)
        if not manager:
            return
        merged = self.use_merged_windows()
        window = None
        if merged:
            window = self._reader.read(GM + WINDOW_OFFSET, WINDOW_BYTES)
            gm = window[:GM_BLOCK_BYTES]
        else:
            gm = mem.read(GM + 0x3DBA8, GM_BLOCK_BYTES)
        flags = struct.unpack_from("<I", gm, 4)[0]
        if not flags & 8 or flags & 2:
            self.filtered += 1
            return
        stage = struct.unpack_from("<I", gm, 0x21C)[0]
        if stage != self.stage:
            if stage in self.stage_sequence:
                raise RuntimeError("Replay revisited an already captured stage")
            self.stage = stage
            self.stage_sequence.append(stage)
            if str(stage) not in self.rng_origins:
                raise RuntimeError("Stage began without its initialization callback")
            self.log(f"trace: stage {stage} began at frame {self.stage_frames.get(str(stage))}")
        frame = mem.u32(manager)
        recorded = self.stages.get(stage)
        if recorded is None:
            raise RuntimeError(f"Replay stage {stage} is absent from the fixture manifest")
        if frame > recorded["inputRecords"]:
            raise RuntimeError(
                f"Stage {stage} read beyond its recorded input at frame {frame}"
            )
        if merged:
            head = self._reader.read(GM + GM_HEAD_OFFSET, GM_HEAD_BYTES)
            globals_pointer = struct.unpack_from("<I", head, 0)[0]
        else:
            globals_pointer = mem.u32(GM + 8)
        if not globals_pointer:
            raise RuntimeError("g_GameManager globals pointer is null")
        if merged:
            globals_data = self._reader.read(globals_pointer, 0xA0)
            player = self._reader.read(PLAYER, PLAYER_BYTES)
            player_tail = player[PLAYER_TAIL_OFFSET:PLAYER_BYTES]
            rng = window[RNG_IN_WINDOW:RNG_IN_WINDOW + 8]
            player_first = player[0]
            input_value = struct.unpack_from("<H", window, INPUT_IN_WINDOW)[0]
            gm_plus_30 = struct.unpack_from(
                "<I", head, 0x30 - GM_HEAD_OFFSET)[0]
        else:
            globals_data = mem.read(globals_pointer, 0xA0)
            player_tail = mem.read(PLAYER + PLAYER_TAIL_OFFSET, PLAYER_TAIL_BYTES)
            rng = mem.read(RNG, 8)
            player_first = mem.read(PLAYER, 1)[0]
            input_value = struct.unpack("<H", mem.read(INPUT, 2))[0]
            gm_plus_30 = mem.u32(GM + 0x30)
        origin = self.rng_origins[str(stage)]
        row = compose_row(
            calc=self.count + 1,
            stage=stage,
            frame=frame,
            gm=gm,
            globals_data=globals_data,
            player_tail=player_tail,
            player_first=player_first,
            rng=rng,
            rng_origin=origin,
            input_value=input_value,
            gm_plus_30=gm_plus_30,
        )
        if len(row) != len(FIELDS):
            raise RuntimeError(f"composed {len(row)} fields, schema has {len(FIELDS)}")
        if self.last_frame.get(str(stage)) == frame:
            self.duplicate_frames += 1
        self._pending.append(row_line(row))
        self.count += 1
        self.stage_rows[str(stage)] = self.stage_rows.get(str(stage), 0) + 1
        self.stage_frames[str(stage)] = frame
        self.end_scores[str(stage)] = row[4]
        self.last_frame[str(stage)] = frame
        if len(self.sample_rows) < 5:
            self.sample_rows.append({"t": round(time.perf_counter(), 4),
                                     "row": format_row_for_log(row)})
        if len(self._pending) >= ROW_BATCH:
            self.write_pending()
        if self.max_rows is not None and self.count >= self.max_rows:
            self.finish(None, reason=f"row cap {self.max_rows} reached")

    # -- EXP-P0-074 row read windows ---------------------------------------
    def use_merged_windows(self) -> bool:
        """Decide (once) whether the merged row windows may be used."""
        if self._merged is None:
            self._merged = self.probe_merged_windows()
            mode = "merged" if self._merged else f"per-address ({self.merged_error})"
            self.log(f"trace: row read mode={mode}")
        return self._merged

    def probe_merged_windows(self) -> bool:
        """Prove the merged windows equal the individual reads, while frozen.

        Called from the first frame breakpoint, i.e. with the target stopped, so
        the two read sets must return identical bytes. Any doubt (no handle,
        unreadable window, mismatch) permanently pins the official per-address
        reads, which is the behaviour this tool had before EXP-P0-074.
        """
        if self.merged_reads is False:
            self.merged_error = "disabled by merged_reads=False"
            return False
        try:
            reader = WindowReader(getattr(self.mem, "handle", 0))
        except OSError as exc:
            self.merged_error = str(exc)
            return False
        try:
            window = reader.read(GM + WINDOW_OFFSET, WINDOW_BYTES)
            head = reader.read(GM + GM_HEAD_OFFSET, GM_HEAD_BYTES)
            player = reader.read(PLAYER, PLAYER_BYTES)
            gm_block = self.mem.read(GM + WINDOW_OFFSET, GM_BLOCK_BYTES)
            rng = self.mem.read(RNG, 8)
            input_word = self.mem.read(INPUT, 2)
            globals_pointer = self.mem.read(GM + GM_HEAD_OFFSET, 4)
            difficulty = self.mem.read(GM + 0x30, 4)
            tail = self.mem.read(PLAYER + PLAYER_TAIL_OFFSET, PLAYER_TAIL_BYTES)
            first = self.mem.read(PLAYER, 1)
        except OSError as exc:
            self.merged_error = str(exc)
            return False
        difficulty_at = 0x30 - GM_HEAD_OFFSET
        checks = (
            window[:GM_BLOCK_BYTES] == gm_block,
            head[:4] == globals_pointer,
            head[difficulty_at:difficulty_at + 4] == difficulty,
            window[RNG_IN_WINDOW:RNG_IN_WINDOW + 8] == rng,
            window[INPUT_IN_WINDOW:INPUT_IN_WINDOW + 2] == input_word,
            player[:1] == first,
            player[PLAYER_TAIL_OFFSET:PLAYER_BYTES] == tail,
        )
        self.merged_checks = checks
        if all(checks):
            self._reader = reader
            return True
        self.merged_error = f"window mismatch {checks}"
        return False

    def write_pending(self) -> None:
        """Flush the joined row buffer to ``rows.jsonl`` in one write."""
        if not self._pending:
            return
        self.stream.write("".join(self._pending))
        self.rows_written += len(self._pending)
        self._pending.clear()

    # -- official `finish(error)` -----------------------------------------
    def finish(self, error: str | None = None, reason: str | None = None,
               natural_exit: bool = False) -> None:
        """Official ``Observer.finish``: same checks, same payload.

        ``natural_exit`` is only set when ``0x453080`` (DeleteReplayManager)
        ended the playback.  The official observer calls ``finish()`` with no
        error there, so a natural end must not be recorded as an error; the
        extra key keeps "the replay reached its own end" auditable.
        """
        if self.finished:
            return
        self.finished = True
        self.stop.set()
        errors = list(self.errors)
        if error:
            errors.append(error)
        if reason:
            errors.append(reason)
        expected = [index for index in sorted(self.stages) if index >= self.start_stage]
        if self.stage_sequence != expected:
            errors.append(f"Stage sequence {self.stage_sequence} differs from {expected}")
        final = expected[-1] if expected else None
        for stage_index, stage in self.stages.items():
            key = str(stage_index)
            if stage_index not in self.stage_sequence or key not in self.stage_frames:
                continue
            remaining = stage["inputRecords"] - self.stage_frames[key]
            # replay_file.valid_input_tail: a stage-clear keeps 2/3 inputs, the
            # final stage's StopRecording extension leaves 6/7.
            tail = (6, 7) if stage_index == final else (2, 3)
            if remaining not in tail:
                errors.append(f"Stage {key} ended with {remaining} unconsumed input records")
            if self.end_scores.get(key) != stage["endScore"]:
                errors.append(f"Stage {key} score differs from the original recording")
        self.write_pending()
        self.stream.flush()
        self.stream.close()
        payload = dict(
            rows=self.count,
            ended=not errors,
            errors=errors,
            stages=self.stage_sequence,
            stageFrames=self.stage_frames,
            endScores=self.end_scores,
            rngOrigins=self.rng_origins,
            naturalReplayExit=bool(natural_exit),
        )
        with (self.out_dir / "complete.json").open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload) + "\n")
        self.payload = payload
        self.log(f"trace: finished rows={self.count} errors={errors}")

    def set_rng_origin(self, stage: int, value: int) -> None:
        key = str(stage)
        changed = self.rng_origins.get(key) != value
        self.rng_origins[key] = value
        # The stage-start breakpoint re-fires while the frame counter is still 0,
        # so this is called dozens of times per frame there; only a real change
        # is worth a log line (and a log line costs a print + write + flush).
        if changed:
            self.log(f"trace: rng origin of stage {stage} = {value}")

    def flush(self) -> None:
        """Push buffered rows to disk (the official observer flushes every 120)."""
        self.write_pending()
        if not self.stream.closed:
            self.stream.flush()


# ---------------------------------------------------------------------------
# phase driver
# ---------------------------------------------------------------------------
def _hash_watch(paths: list[Path]) -> dict:
    return {str(path): sha256_file(path) for path in paths if Path(path).is_file()}


def _running_instances() -> list[int]:
    output = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq th08.exe", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, timeout=30,
    )
    pids = []
    for line in (output.stdout or "").splitlines():
        parts = [part.strip().strip('"') for part in line.split('","')]
        if parts and parts[0].strip('"').lower() == "th08.exe" and len(parts) > 1:
            try:
                pids.append(int(parts[1]))
            except ValueError:
                continue
    return pids


def write_json_atomic(path: Path, payload: dict) -> None:
    """Replace ``path`` with ``payload`` so a reader never sees a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class NavigationLock:
    """Cross-process exclusive lock that serialises menu navigation only.

    Held from just before ``force_foreground`` until the replay is playing, so
    at most one observer in a parallel group owns the keyboard.  Playback and
    the trace reads need no input and therefore hold no lock.
    """

    def __init__(self, path: Path, timeout_s: float, log, poll_s: float = 0.25) -> None:
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self.log = log
        self.poll_s = float(poll_s)
        self.stream = None
        self.held = False
        self.waited = 0.0
        self.acquired_at: str | None = None
        self.released_at: str | None = None
        self._acquired_monotonic = 0.0

    def acquire(self, stop_event: threading.Event | None = None) -> dict:
        started = time.perf_counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            self.stream.seek(0)
            try:
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                self.held = True
                self._acquired_monotonic = time.perf_counter()
                self.waited = round(self._acquired_monotonic - started, 3)
                self.acquired_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                self.log(f"nav-lock: acquired {self.path} after {self.waited:.2f} s")
                return {"ok": True, "path": str(self.path), "waitedSeconds": self.waited,
                        "acquiredAt": self.acquired_at, "timeoutSeconds": self.timeout_s}
            except OSError:
                if time.perf_counter() - started >= self.timeout_s:
                    self.log(f"nav-lock: timeout after {self.timeout_s:g} s on {self.path}")
                    return {"ok": False, "path": str(self.path),
                            "waitedSeconds": round(time.perf_counter() - started, 3),
                            "timeoutSeconds": self.timeout_s}
                time.sleep(self.poll_s)

    def release(self) -> dict:
        info = {"held": self.held, "heldSeconds": None, "releasedAt": None}
        if not self.held or self.stream is None:
            return info
        held_seconds = round(time.perf_counter() - self._acquired_monotonic, 3)
        try:
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError as exc:
            self.log(f"nav-lock: unlock failed: {type(exc).__name__}: {exc}")
        finally:
            self.held = False
            try:
                self.stream.close()
            except OSError:
                pass
            self.stream = None
        self.released_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        info.update(held=True, heldSeconds=held_seconds, releasedAt=self.released_at)
        self.log(f"nav-lock: released {self.path} after {held_seconds:.2f} s "
                 f"at {self.released_at}")
        return info


class StatusWriter:
    """Milestone/progress file (``<out>/status.json``) for a parallel driver.

    Written atomically every few seconds so an external scheduler can (a) start
    the next instance only after this one finished its menu navigation and
    (b) notice a stalled or failed instance without reading the game's memory.
    """

    def __init__(self, path: Path, log, base: dict | None = None) -> None:
        self.path = Path(path)
        self.log = log
        self.payload = dict(base or {})
        self.payload.update({
            "schema": "v400-retail-trace-observer-status/1",
            "pid": os.getpid(),
            "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        self.write()

    def update(self, **fields) -> None:
        self.payload.update(fields)
        self.payload["updatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.write()

    def write(self) -> None:
        try:
            write_json_atomic(self.path, self.payload)
        except OSError as exc:  # a status file must never kill a capture
            self.log(f"status: write failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# EXP-P0-075: independent budget watchdog
# ---------------------------------------------------------------------------
def thread_stacks() -> dict:
    """Every thread's Python stack as text, for the watchdog's post-mortem.

    A wedged observer used to leave nothing behind but a frozen ``status.json``;
    the post-mortem is what turns the next hang into a diagnosis (the blocking
    call is the last frame of the ``main-*`` entry).
    """
    frames: dict[str, list[str]] = {}
    main_id = threading.main_thread().ident
    for thread_id, frame in sys._current_frames().items():
        label = f"{'main' if thread_id == main_id else 'thread'}-{thread_id}"
        try:
            frames[label] = [line.rstrip() for line in traceback.format_stack(frame)]
        except Exception as exc:  # pragma: no cover - defensive
            frames[label] = [f"<unavailable: {type(exc).__name__}: {exc}>"]
    return frames


def emit_timeout_state(*, out, report: dict, status=None, collector=None,
                       debugger=None, process=None, log=None, write_report=None,
                       exit_code: int = EXIT_TIMEOUT,
                       reason: str = "watchdog: --budget-seconds expired",
                       timer_seconds: float | None = None) -> dict:
    """Force the timeout ending from whatever state the run is in (EXP-P0-075).

    This runs on the watchdog thread, so every step is guarded on its own: a
    failing step must not stop the row flush, the detach, the game kill or the
    report.  Returns what each step did, so a unit test can assert the actions
    instead of trusting the ordering.
    """
    log = log or (lambda message: None)
    out = Path(out)
    started = time.perf_counter()
    result: dict = {
        "reason": reason,
        "exitCode": exit_code,
        "timerSeconds": timer_seconds,
        "graceSeconds": WATCHDOG_GRACE_SECONDS,
        "firedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "steps": {},
        "errors": [],
    }

    def step(name: str, action) -> None:  # noqa: ANN001 - tiny local helper
        try:
            result["steps"][name] = action()
        except Exception as exc:  # noqa: BLE001 - one step never sinks the rest
            result["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
            log(f"watchdog: {name} failed: {type(exc).__name__}: {exc}")

    def capture_stacks():
        snapshot = {"firedAt": result["firedAt"], "reason": reason,
                    "threads": thread_stacks()}
        path = out / "watchdog-stacks.json"
        write_json_atomic(path, snapshot)
        main_lines = next((lines for key, lines in snapshot["threads"].items()
                           if key.startswith("main-")), [])
        section = report.setdefault("watchdog", {})
        section["stacksPath"] = str(path)
        section["mainThreadStack"] = main_lines[-40:]
        return {"path": str(path), "threads": len(snapshot["threads"]),
                "mainThreadTop": (main_lines[-1].strip() if main_lines else None)}

    step("stacks", capture_stacks)

    def finish_rows():
        if collector is None:
            return None
        if not getattr(collector, "finished", False):
            collector.finish(None, reason=reason)
        collector.flush()
        return {"rows": getattr(collector, "count", None),
                "finished": bool(getattr(collector, "finished", False)),
                "rowsPath": str(getattr(collector, "rows_path", ""))}

    step("rows", finish_rows)
    step("debugger", lambda: None if debugger is None else debugger.hard_stop())

    def kill_game():
        if process is None:
            return None
        info: dict = {"pid": getattr(process, "pid", None)}
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except Exception:  # noqa: BLE001 - terminate refused; escalate once
                try:
                    process.kill()
                    process.wait(timeout=5.0)
                except Exception as exc:  # noqa: BLE001
                    info["killError"] = f"{type(exc).__name__}: {exc}"
        info["returnCode"] = process.poll()
        return info

    step("game", kill_game)

    def update_status():
        if status is None:
            return None
        status.update(phase="timeout", timeoutReason=reason,
                      watchdogExitCode=exit_code)
        return "timeout"

    step("status", update_status)

    report["status"] = "timeout"
    report.setdefault("result", {}).update(
        reason=reason, exitCode=exit_code,
        watchdog={"fired": True, "timerSeconds": timer_seconds,
                  "graceSeconds": WATCHDOG_GRACE_SECONDS},
    )
    report.setdefault("watchdog", {}).update(result)

    def write_the_report():
        if write_report is not None:
            write_report()
        else:
            write_json_atomic(out / "report.json", report)
        return str(out / "report.json")

    step("report", write_the_report)
    # the report was snapshotted before its own step was recorded; write it once
    # more so the file lists every step it performed (audit, EXP-P0-075)
    report.setdefault("watchdog", {}).update(result)
    step("report-final", write_the_report)
    log(f"watchdog: forced finish(status='timeout') in "
        f"{time.perf_counter() - started:.3f} s; steps={sorted(result['steps'])}")
    return result


class Watchdog:
    """Enforce ``--budget-seconds`` without depending on the main loop.

    ``threading.Timer`` semantics plus an explicit fired latch: the callback runs
    exactly once, on a daemon thread, and ``cancel()`` before the deadline
    prevents it.  The callback must assume *nothing* about the main thread -- it
    may be blocked inside a Win32 call (ctypes releases the GIL there, which is
    why this thread still gets to run) -- so it writes its own files, clears the
    debug registers, detaches, kills the game and then hard-exits the process
    with the timeout code.
    """

    def __init__(self, seconds: float, on_fire, log=None, name: str = "budget-watchdog"):
        self.seconds = max(0.0, float(seconds))
        self.on_fire = on_fire
        self.log = log or (lambda message: None)
        self.name = name
        self.fired = threading.Event()
        self._timer: threading.Timer | None = None
        self.started_at = time.perf_counter()

    @property
    def armed(self) -> bool:
        return self._timer is not None and self._timer.is_alive()

    def start(self) -> "Watchdog":
        if self._timer is not None:
            return self
        timer = threading.Timer(self.seconds, self._fire)
        timer.daemon = True
        timer.name = self.name
        self._timer = timer
        self.started_at = time.perf_counter()
        timer.start()
        return self

    def cancel(self) -> bool:
        """Disarm the watchdog; returns True when the deadline had not fired."""
        if self._timer is not None:
            self._timer.cancel()
        return not self.fired.is_set()

    def rearm(self, seconds: float) -> "Watchdog":
        """Arm one fresh timer after a fire (the run needed a longer grace).

        Used when the watchdog fires while the driver is already in its own
        shutdown sequence: the timeout still has to be enforced, but not over a
        run that is measurably finishing.
        """
        self.seconds = max(0.0, float(seconds))
        self._timer = None
        self.fired.clear()
        return self.start()

    def _fire(self) -> None:
        if self.fired.is_set():
            return
        self.fired.set()
        try:
            self.on_fire()
        except Exception as exc:  # noqa: BLE001 - a watchdog must never be the crash
            self.log(f"watchdog: callback failed: {type(exc).__name__}: {exc}")


def navigation_deadline_hit(playback_started: bool, elapsed: float,
                           deadline: float) -> bool:
    """EXP-P0-075: fail fast instead of pumping a title screen that never plays.

    ``deadline <= 0`` disables the check (``--nav-deadline 0``).
    """
    return deadline > 0 and (not playback_started) and elapsed >= deadline


#: EXP-P0-075 root cause #2: ``winapi.force_foreground`` calls
#: ``AttachThreadInput`` against the retail process's threads.  That call blocks
#: in the kernel until the target thread services its input queue, and the target
#: threads here are *also* frozen by our own hardware breakpoints, so a hit that
#: lands inside that window can leave the caller blocked with no upper bound (the
#: 39-minute ``easy-2-b`` freeze happened right after ``phase="attached"``, i.e.
#: exactly between the attach milestone and the ``navigating`` milestone, and the
#: only unbounded Win32 calls in that window are the foreground transfer).
#: The observer therefore runs the transfer on a throw-away daemon thread and
#: gives up after ``--foreground-timeout`` seconds instead of waiting forever:
#: a missed foreground costs a blind key sequence, a wedged main thread costs the
#: whole fixture.
def foreground_with_timeout(winapi, hwnd: int, timeout: float) -> tuple[bool, bool]:  # noqa: ANN001
    """``(foreground_ok, timed_out)`` for a foreground transfer bounded by ``timeout``."""
    box: dict[str, bool] = {"ok": False}

    def attempt() -> None:
        try:
            box["ok"] = bool(winapi.force_foreground(hwnd))
        except Exception:  # noqa: BLE001 - a failed transfer is not a failed run
            box["ok"] = False

    thread = threading.Thread(target=attempt, daemon=True, name="force-foreground")
    thread.start()
    thread.join(max(0.0, float(timeout)))
    return box["ok"], thread.is_alive()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("poc", "four", "rows"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--playback-mode", type=int, choices=(0, 2), default=2)
    parser.add_argument("--budget-seconds", type=float, default=175.0)
    parser.add_argument("--watchdog-grace", type=float, default=WATCHDOG_GRACE_SECONDS,
                        help="seconds after --budget-seconds before the independent "
                             "watchdog thread forces the timeout ending (EXP-P0-075); "
                             "the grace lets a healthy run leave through the graceful path")
    parser.add_argument("--nav-deadline", type=float, default=NAV_DEADLINE_SECONDS,
                        help="fail the run if replay playback has not started within this "
                             "many seconds (EXP-P0-075 fast-fail); 0 disables")
    parser.add_argument("--foreground-timeout", type=float, default=8.0,
                        help="bound the foreground transfer (EXP-P0-075: AttachThreadInput "
                             "can block forever on a thread frozen by our own breakpoints)")
    parser.add_argument("--max-hits", type=int, default=5,
                        help="poc phase: stop after this many frame hits")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--four-seconds", type=float, default=20.0,
                        help="four phase: seconds of playback to count after the first frame hit")
    parser.add_argument("--no-try-delete-breakpoint", dest="try_delete_breakpoint",
                        action="store_false",
                        help="four phase: do not quit the replay to exercise 0x453080")
    parser.add_argument("--breakpoints", default=None,
                        help="comma-separated subset of frame,completion,stageStart,deleteReplay")
    parser.add_argument("--nav", choices=("auto", "closed", "blind"), default="auto")
    parser.add_argument("--dr7", type=lambda text: int(text, 0), default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--quiet-settle", type=float, default=45.0,
                        help="how long to drain the attach burst before the reference clock starts")
    parser.add_argument("--with-score-dat", action="store_true",
                        help="seed the run dir with the template score.dat (default: do not)")
    parser.add_argument("--allow-existing-instances", action="store_true",
                        help="parallel pipeline: allow other th08.exe processes (the single-"
                             "instance refusal is for the serial mode)")
    parser.add_argument("--nav-lock", type=Path, default=None,
                        help="parallel pipeline: cross-process lock serialising menu navigation "
                             "(the only phase that needs the foreground window)")
    parser.add_argument("--nav-lock-timeout", type=float,
                        default=DEFAULT_NAV_LOCK_TIMEOUT,
                        help="how long to wait for --nav-lock before giving up")
    parser.add_argument("--status-json", type=Path, default=None,
                        help=f"progress file for an external scheduler (default <out>/{STATUS_NAME})")
    args = parser.parse_args(argv)

    if not 1.0 <= args.budget_seconds <= MAX_BUDGET_SECONDS:
        parser.error(f"--budget-seconds must be in [1, {MAX_BUDGET_SECONDS:g}]")
    if not 0.0 <= args.watchdog_grace <= 120.0:
        parser.error("--watchdog-grace must be in [0, 120]")
    if not 0.0 <= args.nav_deadline <= MAX_BUDGET_SECONDS:
        parser.error(f"--nav-deadline must be in [0, {MAX_BUDGET_SECONDS:g}]")
    if not 0.0 <= args.foreground_timeout <= 120.0:
        parser.error("--foreground-timeout must be in [0, 120]")

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"refusing to reuse non-empty {out}", file=sys.stderr)
        return EXIT_USAGE
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "debugger.log"
    log_stream = log_path.open("w", encoding="utf-8")
    started_at = time.perf_counter()

    def log(message: str) -> None:
        line = f"[{time.perf_counter() - started_at:8.3f}] {message}"
        print(line, flush=True)
        log_stream.write(line + "\n")
        log_stream.flush()

    status_path = Path(args.status_json) if args.status_json else out / STATUS_NAME
    status = StatusWriter(status_path, log, base={
        "out": str(out),
        "phase": "starting",
        "replay": str(args.replay),
        "budgetSeconds": args.budget_seconds,
        "allowExistingInstances": bool(args.allow_existing_instances),
        "navLockPath": str(args.nav_lock) if args.nav_lock else None,
    })

    report: dict = {
        "schema": "v400-retail-trace-debugapi/1",
        "phase": args.phase,
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "running",
        "arguments": {key: (str(value) if isinstance(value, Path) else value)
                      for key, value in vars(args).items()},
        "events": [],
        "result": {},
        "limitations": [
            "Reads use ReadProcessMemory inside the debug event, i.e. while the target is frozen "
            "exactly like a GDB stop; the official observer reads through the GDB inferior API.",
            "The retail image has no frame fence; the hardware execute breakpoint at 0x452490 is "
            "the fence, which is what the official observer uses too.",
        ],
    }

    def write_report() -> None:
        payload = {key: value for key, value in report.items() if not key.startswith("_")}
        (out / "report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    identity = retail_identity(RETAIL_EXE)
    report["retailImage"] = identity
    if not identity["verified"]:
        report["status"] = "blocked"
        report["result"].update(reason=identity.get("reason"), exitCode=EXIT_USAGE)
        write_report()
        log_stream.close()
        return EXIT_USAGE

    if not args.replay.is_file():
        report["status"] = "blocked"
        report["result"].update(reason=f"missing replay {args.replay}", exitCode=EXIT_USAGE)
        write_report()
        log_stream.close()
        return EXIT_USAGE

    replay_module = load_replay_module()
    replay_info = replay_module.inspect(args.replay)
    expectations = manifest_expectations(args.replay, replay_info["sha256"])
    report["replay"] = replay_info
    report["manifest"] = expectations
    log(f"replay {args.replay.name} sha256={replay_info['sha256']} "
        f"stages={[stage['index'] for stage in replay_info['stages']]} "
        f"calculationFrames={expectations.get('calculationFrames')} "
        f"retailTraceSha256={expectations.get('retailTraceSha256')}")

    running = _running_instances()
    report["runningInstancesBefore"] = running
    if running and not args.allow_existing_instances:
        report["status"] = "blocked"
        report["result"].update(
            reason=f"another th08.exe is running: pid {running}", exitCode=EXIT_INSTANCE
        )
        write_report()
        log_stream.close()
        return EXIT_INSTANCE
    if running:
        log(f"parallel mode: {len(running)} other th08.exe already running (allowed): {running}")
        status.update(phase="starting", otherInstances=running)

    privilege = enable_debug_privilege()
    report["seDebugPrivilege"] = privilege
    log(f"SeDebugPrivilege: {privilege}")

    if args.breakpoints:
        names = [name.strip() for name in args.breakpoints.split(",") if name.strip()]
        unknown = [name for name in names if name not in BREAKPOINTS]
        if unknown:
            parser.error(f"unknown breakpoint names {unknown}")
    elif args.phase == "poc":
        names = ["frame"]
    elif args.phase == "four":
        names = ["frame", "completion", "stageStart", "deleteReplay"]
    else:
        names = ["frame", "stageStart", "deleteReplay"]
    breakpoints = {name: BREAKPOINTS[name] for name in names}
    report["breakpoints"] = {name: f"0x{address:08X}" for name, address in breakpoints.items()}
    report["breakpointRoles"] = {name: BREAKPOINT_ROLES[name] for name in breakpoints}

    memory_mod, winapi = _load_touhou_rl()
    run_dir = Path(args.run_dir) if args.run_dir else out / "rundir"
    report["runDir"] = assemble_run_dir(run_dir, skip_score_dat=not args.with_score_dat)
    installed = install_replay(run_dir, args.replay, index=1)
    report["installedReplay"] = {"path": str(installed), "bytes": installed.stat().st_size}

    watched = [RETAIL_EXE, ASSET_DIR / "th08.dat", args.replay]
    before_hashes = _hash_watch(watched)

    exe = run_dir / "th08.exe"
    env = dict(os.environ)
    env["__COMPAT_LAYER"] = "HIGHDPIAWARE"
    env.pop("PYTHONPATH", None)
    winapi.dpi_aware()
    log(f"launching {exe}")
    process = subprocess.Popen(
        [str(exe)], cwd=str(run_dir), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    pid = process.pid
    report["pid"] = pid
    exit_code = EXIT_TIMEOUT
    debugger = None
    mem = None
    keyboard = None
    stop_event = threading.Event()
    navigator = None
    collector = None
    nav_lock = None
    deadline = started_at + args.budget_seconds

    # EXP-P0-075: the hard backstop.  Started before the window wait, so the
    # budget covers the whole run, and independent of every loop below, so a
    # main thread blocked in a Win32 call cannot outlive it.
    watchdog_timer_seconds = args.budget_seconds + args.watchdog_grace
    #: ``ending`` is set the moment the driver loop is done: a fire after that
    #: point means the graceful ending (finish + grade + close) is still running,
    #: not that the main thread is wedged, so the watchdog only re-arms once.
    run_state = {"ending": False, "rearms": 0}

    def on_budget_timeout() -> None:
        if run_state["ending"] and run_state["rearms"] < 1:
            run_state["rearms"] += 1
            log("watchdog: the driver loop already ended; re-arming once to let the "
                "graceful ending finish")
            watchdog.rearm(max(args.watchdog_grace, 15.0))
            return
        log(f"watchdog: budget of {args.budget_seconds:g} s + "
            f"{args.watchdog_grace:g} s grace expired; forcing the timeout ending")
        emit_timeout_state(
            out=out, report=report, status=status, collector=collector,
            debugger=debugger, process=process, log=log, write_report=write_report,
            exit_code=EXIT_TIMEOUT,
            reason=(f"watchdog: observer budget of {args.budget_seconds:g} s expired "
                    f"and the main loop never reached its own deadline check"),
            timer_seconds=round(watchdog_timer_seconds, 3),
        )
        try:
            log_stream.flush()
        except (OSError, ValueError):
            pass
        log(f"watchdog: forcing process exit code {EXIT_TIMEOUT}")
        os._exit(EXIT_TIMEOUT)

    watchdog = Watchdog(watchdog_timer_seconds, on_budget_timeout, log).start()
    report["watchdog"] = {
        "timerSeconds": round(watchdog_timer_seconds, 3),
        "budgetSeconds": args.budget_seconds,
        "graceSeconds": args.watchdog_grace,
        "started": True,
    }
    try:
        hwnd, rect = wait_for_window(winapi, pid, timeout=min(30.0, args.budget_seconds / 3))
        report["window"] = {
            "hwnd": hex(hwnd), "class": winapi.class_name(hwnd),
            "title": winapi.window_title(hwnd), "client": [rect[2], rect[3]],
            "titleSeconds": round(time.perf_counter() - started_at, 3),
        }
        log(f"window 0x{hwnd:X} client={rect[2]}x{rect[3]}")
        mem = memory_mod.ProcessMemory(pid)
        report["moduleBase"] = hex(mem.module_base)

        attestation = attest(mem, exe)
        report["attestation"] = attestation
        log(f"attestation ok={attestation['ok']} "
            f"{[check['ok'] for check in attestation['checks']]}")
        if not attestation["ok"]:
            report["status"] = "blocked"
            report["result"].update(
                reason="mapped instructions differ from the executable; refusing to arm",
                exitCode=EXIT_BLOCKED,
            )
            exit_code = EXIT_BLOCKED
            raise SystemExit
        log(f"timing anchors: gm+0x3DDC0={mem.u32(GM + 0x3DDC0)} "
            f"replayManager=0x{mem.u32(REPLAY_MANAGER):08X}")

        max_hits = args.max_hits if args.phase == "poc" else None
        hit_times: list[float] = []

        def on_hit(name: str, context, tid: int) -> None:  # noqa: ANN001
            hit_times.append(time.perf_counter())
            if name == "stageStart":
                manager = mem.u32(REPLAY_MANAGER)
                if manager and not mem.u32(manager):
                    stage = mem.u32(GM + 0x3DDC4)
                    if collector is not None:
                        collector.set_rng_origin(stage, mem.u32(RNG + 4))
                    if len(hit_times) <= 8:
                        log(f"stageStart: stage={stage} rngOrigin={mem.u32(RNG + 4)}")
            elif name == "frame":
                if collector is not None:
                    collector.on_frame()
                elif len(hit_times) <= (max_hits or 5):
                    manager = mem.u32(REPLAY_MANAGER)
                    gm = mem.read(GM + 0x3DBA8, 0x220)
                    log(
                        f"frame hit #{len(hit_times)} eip=0x{context.Eip:08X} "
                        f"stage={struct.unpack_from('<I', gm, 0x21C)[0]} "
                        f"gameFrame={struct.unpack_from('<I', gm, 0x218)[0]} "
                        f"flags=0x{struct.unpack_from('<I', gm, 4)[0]:08X} "
                        f"replayFrame={mem.u32(manager) if manager else None} "
                        f"rng={mem.read(RNG, 8).hex()} input={mem.read(INPUT, 2).hex()}"
                    )
            elif name == "deleteReplay":
                if collector is not None and not collector.finished:
                    log("deleteReplay: ReplayManager::DeleteReplayManager reached; "
                        "the replay ended on its own")
                    collector.finish(None, natural_exit=True)

        debugger = NativeDebugger(pid, breakpoints, mem, log, on_hit=on_hit, dr7=args.dr7)
        log(f"DebugActiveProcess({pid}) arming DR0..DR3 with dr7=0x{debugger.dr7:X}")
        debugger.attach()
        # Drain the attach burst: CREATE_PROCESS, one CREATE_THREAD per existing
        # thread, one LOAD_DLL per module, and the initial break-in.  The process
        # runs between events, so the game keeps booting while we drain.
        quiet_until = time.perf_counter() + args.quiet_settle
        idle_deadline = None
        while time.perf_counter() < quiet_until and process.poll() is None:
            code = debugger.pump(100)
            if code is None:
                if idle_deadline is None:
                    idle_deadline = time.perf_counter() + 0.4
                elif time.perf_counter() >= idle_deadline and debugger.armed_threads >= 1:
                    break
            else:
                idle_deadline = None
        report["attach"] = {
            "armedThreads": debugger.armed_threads,
            "armedDetails": debugger.armed_details,
            "eventsHandled": debugger.events_handled,
            "threadCount": len(debugger.threads),
            "dr7": hex(debugger.dr7),
        }
        log(f"armed {debugger.armed_threads} threads ({len(debugger.threads)} live); "
            f"dr7 readback={[detail.get('dr7') for detail in debugger.armed_details[:2]]}")
        if debugger.armed_threads == 0:
            report["status"] = "blocked"
            report["result"].update(reason="no thread could be armed with the debug registers",
                                    exitCode=EXIT_BLOCKED)
            exit_code = EXIT_BLOCKED
            raise SystemExit

        if args.phase == "rows":
            collector = TraceCollector(mem, out, replay_info["stages"], args.start_stage,
                                       log, max_rows=args.max_rows)
        status.update(phase="attached", gamePid=pid, armSeconds=round(
            time.perf_counter() - started_at, 3))
        keyboard = winapi.SendInputKeyboard()
        nav_lock_info = None
        if args.nav_lock:
            nav_lock = NavigationLock(args.nav_lock, args.nav_lock_timeout, log)
            nav_lock_info = nav_lock.acquire(stop_event)
            report["navLock"] = nav_lock_info
            status.update(phase="waiting-for-nav-lock", navLock=nav_lock_info)
            if not nav_lock_info["ok"]:
                raise DebuggerError(
                    f"navigation lock timeout after {args.nav_lock_timeout:g} s: "
                    f"{args.nav_lock}", EXIT_BLOCKED)
        foreground_ok, foreground_timed_out = foreground_with_timeout(
            winapi, hwnd, args.foreground_timeout)
        if foreground_timed_out:
            log(f"warning: foreground transfer still blocked after "
                f"{args.foreground_timeout:g} s (EXP-P0-075); continuing without it")
            report["foregroundTransfer"] = {
                "timedOut": True, "timeoutSeconds": args.foreground_timeout,
                "foregroundOk": foreground_ok,
            }
        elif not foreground_ok:
            log(f"warning: could not take foreground; {winapi.input_window_state(hwnd)}")
        status.update(phase="navigating", foregroundOk=foreground_ok,
                      foregroundTimedOut=foreground_timed_out,
                      navLockAcquiredAt=(nav_lock_info or {}).get("acquiredAt"),
                      navLockWaitSeconds=(nav_lock_info or {}).get("waitedSeconds"))
        navigator = Navigator(mem, keyboard, args.start_stage, args.playback_mode, stop_event,
                              log, replay_index=0)
        if args.nav == "blind":
            navigator.closed_timeout = 0.0
        navigator.start()
        report["navigation"] = None
        report["parallel"] = {
            "allowExistingInstances": bool(args.allow_existing_instances),
            "navLock": nav_lock_info,
            "foregroundOk": foreground_ok,
        }

        frame_timeouts = 0
        four_deadline = None
        quit_done = False
        playback_started_at = None
        last_status_at = 0.0
        navigation_started = time.perf_counter()
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                log(f"process exited with {process.returncode}")
                break
            if collector is not None and collector.finished:
                break
            # EXP-P0-075: a navigation that is alive but will never reach the replay
            # must fail the run instead of idling until the budget dies.
            if args.nav_deadline and navigation_deadline_hit(
                    navigator.playback_started,
                    time.perf_counter() - navigation_started, args.nav_deadline):
                navigator.error = (
                    f"navigation deadline: replay playback did not start within "
                    f"{args.nav_deadline:g} s (path={navigator.path}, "
                    f"keyTaps={navigator.key_taps}, "
                    f"lastMenuState={(navigator.menu_log[-1] if navigator.menu_log else None)})"
                )
                report["navigationDeadlineSeconds"] = args.nav_deadline
                log(f"navigation failed: {navigator.error}")
                break
            if args.phase == "poc" and debugger.hits["frame"] >= args.max_hits:
                log(f"poc: reached {debugger.hits['frame']} frame hits")
                break
            if args.phase == "four":
                if four_deadline is None and debugger.hits["frame"] > 0:
                    four_deadline = time.perf_counter() + args.four_seconds
                if four_deadline is not None and time.perf_counter() >= four_deadline:
                    if args.try_delete_breakpoint and not quit_done:
                        # 0x453080 (DeleteReplayManager) only runs when playback
                        # ends: leave the replay through the pause menu so the
                        # fourth slot is proven instead of merely armed.
                        log("four: quitting the replay to exercise 0x00453080")
                        quit_done = True
                        for key, settle in (("esc", 0.6), ("down", 0.18), ("down", 0.18),
                                            ("down", 0.18), ("z", 1.6)):
                            keyboard.tap(key, 0.05)
                            time.sleep(settle)
                        four_deadline = time.perf_counter() + 8.0
                        continue
                    log(f"four: {args.four_seconds:g} s of playback counted; stopping")
                    break
            if debugger.pump(20) is None:
                frame_timeouts += 1
            now = time.perf_counter()
            if navigator.playback_started and playback_started_at is None:
                playback_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                release_info = nav_lock.release() if nav_lock is not None else None
                if release_info is not None and nav_lock_info is not None:
                    nav_lock_info["release"] = release_info
                report.setdefault("parallel", {})["playbackStartedAt"] = playback_started_at
                report["parallel"]["navLockReleasedAt"] = (release_info or {}).get("releasedAt")
                status.update(phase="playing", playbackStarted=True,
                              playbackStartedAt=playback_started_at,
                              foregroundOk=foreground_ok,
                              navLockHeldSeconds=(release_info or {}).get("heldSeconds"))
                log(f"playback started at {playback_started_at}; navigation lock "
                    f"{'released' if nav_lock is not None else 'not used'}")
            if now - last_status_at >= 5.0:
                last_status_at = now
                status.update(
                    phase=("playing" if navigator.playback_started else "navigating"),
                    playbackStarted=bool(navigator.playback_started),
                    playbackStartedAt=playback_started_at,
                    rows=(collector.count if collector is not None else 0),
                    rowsAt=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    elapsedSeconds=round(now - started_at, 3),
                    navPath=navigator.path,
                    navigationError=navigator.error,
                    lastDebugEventAgeSeconds=round(now - debugger.last_event_at, 3),
                    foregroundOk=foreground_ok,
                )
            if navigator.error:
                log(f"navigation failed: {navigator.error}")
                break
            if (not navigator.is_alive() and not navigator.playback_started
                    and time.perf_counter() - started_at > 30.0):
                log("navigation thread ended without playback")
                break
        else:
            log("budget exhausted")
        run_state["ending"] = True

        stop_event.set()
        report["navigation"] = {
            "path": navigator.path,
            "playbackStarted": navigator.playback_started,
            "keyTaps": navigator.key_taps,
            "menuLog": navigator.menu_log[:40],
            "firstPlaybackState": navigator.first_flags,
            "blindFallbackAt": navigator.blind_at,
            "error": navigator.error,
        }
        report["hits"] = debugger.hit_summary()
        if collector is not None:
            if not collector.finished:
                collector.finish(None, reason="phase budget exhausted before the replay ended")
            report["trace"] = {
                "rows": collector.count,
                "rowsPath": str(collector.rows_path),
                "filteredHits": collector.filtered,
                "duplicateStageFrames": collector.duplicate_frames,
                "sampleRows": collector.sample_rows,
                "complete": collector.payload,
            }
        exit_code = _grade(args, report, debugger, collector, expectations, out, log)
    except DebuggerError as exc:
        report["status"] = "blocked"
        report["result"].update(reason=str(exc), exitCode=exc.exit_code)
        exit_code = exc.exit_code
    except SystemExit:
        exit_code = report["result"].get("exitCode", exit_code)
    except Exception as exc:  # noqa: BLE001 - every failure must be recorded, not raised
        import traceback

        report["status"] = "failed"
        report["result"].update(reason=f"{type(exc).__name__}: {exc}",
                                traceback=traceback.format_exc(), exitCode=EXIT_STATE)
        exit_code = EXIT_STATE
        log(f"failure: {type(exc).__name__}: {exc}")
    finally:
        # EXP-P0-075: a run that reached its own ending disarms the watchdog, so a
        # late fire can never turn a completed capture into a timeout.
        if watchdog.cancel():
            report.setdefault("watchdog", {})["armedAtExit"] = False
        stop_event.set()
        if nav_lock is not None and nav_lock.held:
            report.setdefault("navLock", {})["releasedInFinally"] = nav_lock.release()
        if keyboard is not None:
            try:
                keyboard.release_all()
            except OSError:
                pass
        if debugger is not None:
            cleared = debugger.clear_breakpoints()
            debugger.detach()
            report["detach"] = {"clearedThreads": cleared, "detached": not debugger.attached}
            log(f"detached; cleared debug registers on {cleared} threads")
        if process.poll() is None:
            try:
                if mem is not None:
                    report["finalProbe"] = {
                        "flags": mem.u32(GM + 0x3DBA8 + 4),
                        "stage": mem.u32(GM + 0x3DDC4),
                        "gameFrame": mem.u32(GM + 0x3DDC0),
                        "replayManager": mem.u32(REPLAY_MANAGER),
                    }
            except OSError as exc:
                report["finalProbeError"] = str(exc)
            try:
                if "hwnd" in report:
                    winapi.close_window(int(report["window"]["hwnd"], 16))
            except OSError:
                pass
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                log("game did not close in 8 s; terminating")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        report["gameExitCode"] = process.returncode
        if mem is not None:
            try:
                mem.close()
            except Exception:  # noqa: BLE001
                pass
        report["thirdPartyUnchanged"] = _hash_watch(watched) == before_hashes
        report["elapsedSeconds"] = round(time.perf_counter() - started_at, 3)
        report["result"]["exitCode"] = exit_code
        if report.get("status") == "running":
            report["status"] = "ok" if exit_code == EXIT_OK else (
                "partial" if exit_code == EXIT_STATE else "failed")
        write_report()
        status.update(phase="finished", finishedAt=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      exitCode=exit_code, status=report.get("status"),
                      rows=(collector.count if collector is not None else 0),
                      rowsAt=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      playbackStartedAt=(report.get("parallel") or {}).get("playbackStartedAt"),
                      navigationError=(report.get("navigation") or {}).get("error"),
                      error=(report.get("result") or {}).get("reason"),
                      elapsedSeconds=report["elapsedSeconds"])
        log(f"exit {exit_code} status={report.get('status')} "
            f"elapsed={report['elapsedSeconds']}s")
        log_stream.close()

    print(json.dumps({
        "phase": args.phase,
        "status": report.get("status"),
        "exitCode": exit_code,
        "elapsedSeconds": report.get("elapsedSeconds"),
        "report": str(out / "report.json"),
    }, ensure_ascii=False))
    return exit_code


def _grade(args, report: dict, debugger, collector, expectations: dict, out: Path, log) -> int:
    """Decide the phase exit code and write the trace verdict."""
    if args.phase == "poc":
        hits = debugger.hits["frame"]
        report["poc"] = {
            "address": f"0x{BREAKPOINTS['frame']:08X}",
            "expectedHits": args.max_hits,
            "hits": hits,
            "firstHits": debugger.first_hits[: args.max_hits],
            "hit": hits > 0,
        }
        if hits <= 0:
            report["status"] = "failed"
            report["result"].update(
                reason="hardware execute breakpoint never fired; see hits/navigation",
                exitCode=EXIT_STATE)
            return EXIT_STATE
        report["status"] = "ok"
        report["result"].update(exitCode=EXIT_OK)
        return EXIT_OK

    if args.phase == "four":
        summary = debugger.hit_summary()
        report["four"] = {
            "hits": summary["hits"],
            "firstHitOrder": summary["firstHitOrder"],
            "allFired": all(value > 0 for value in summary["hits"].values()),
            "firedBreakpoints": [name for name, value in summary["hits"].items() if value > 0],
        }
        if summary["hits"]["frame"] <= 0:
            report["status"] = "failed"
            report["result"].update(reason="frame boundary never fired", exitCode=EXIT_STATE)
            return EXIT_STATE
        report["status"] = "ok" if report["four"]["allFired"] else "partial"
        report["result"].update(exitCode=EXIT_OK)
        return EXIT_OK

    # rows phase: the whole point is an honest count + SHA comparison.
    rows = collector.count
    rows_path = collector.rows_path
    rows_file_sha = sha256_file(rows_path) if rows_path.is_file() else None
    canonical_sha = None
    canonical_error = None
    if rows_path.is_file():
        try:
            canonical_sha = canonical_trace_sha256(rows_path)
        except Exception as exc:  # noqa: BLE001 - a bad row must be reported, not hidden
            canonical_error = f"{type(exc).__name__}: {exc}"
    expected_rows = expectations.get("calculationFrames")
    expected_sha = expectations.get("retailTraceSha256")
    stages = getattr(collector, "stages", {}) or {}
    stage_rows = getattr(collector, "stage_rows", {}) or {}
    sequence = [index for index in sorted(stages) if index >= args.start_stage]
    final_stage = sequence[-1] if sequence else None
    stage_table = []
    for index in sorted(stages):
        stage = stages[index]
        key = str(index)
        last_frame = collector.stage_frames.get(key)
        remaining = stage["inputRecords"] - last_frame if last_frame is not None else None
        tail = (6, 7) if index == final_stage else (2, 3)
        stage_table.append({
            "stage": index,
            "name": stage.get("name"),
            "inputRecords": stage["inputRecords"],
            "rows": stage_rows.get(key, 0),
            "lastFrame": last_frame,
            "unconsumedInputRecords": remaining,
            "expectedTail": list(tail),
            "tailOk": (remaining in tail) if remaining is not None else False,
            "manifestEndScore": stage["endScore"],
            "observedEndScore": collector.end_scores.get(key),
            "endScoreMatch": collector.end_scores.get(key) == stage["endScore"],
        })
    complete = getattr(collector, "payload", {}) or {}
    verdict = {
        "rows": rows,
        "calculationFrames": expected_rows,
        "rowsMatch": (rows == expected_rows) if expected_rows is not None else None,
        "rowsFileSha256": rows_file_sha,
        "rowsFileEqualsRetailTraceField": (
            (rows_file_sha == expected_sha) if expected_sha else None),
        "canonicalTraceSha256": canonical_sha,
        "canonicalError": canonical_error,
        "retailTraceSha256": expected_sha,
        "shaMatch": (canonical_sha == expected_sha) if (expected_sha and canonical_sha) else None,
        "rowsBytes": rows_path.stat().st_size if rows_path.is_file() else 0,
        "rowsDifference": (expected_rows - rows) if expected_rows is not None else None,
        "ended": complete.get("ended"),
        "naturalReplayExit": complete.get("naturalReplayExit"),
        "completeErrors": complete.get("errors"),
        "stageSequence": complete.get("stages"),
        "stageTable": stage_table,
        "completeJson": str(out / "complete.json"),
        "note": "rows.jsonl is written in the official line format and never adjusted. "
                "retailTrace.sha256 is the official canonical hash (compact schema JSON + 25 "
                "little-endian uint32 per row, compare-replay-traces.trace_sha256), not the "
                "SHA-256 of the file bytes; both are reported",
    }
    report["verdict"] = verdict
    if verdict["rowsMatch"] and verdict["shaMatch"]:
        report["status"] = "ok"
        report["result"].update(
            reason="rows, complete.json and the canonical retail trace hash all match the manifest",
            exitCode=EXIT_OK)
        return EXIT_OK
    if rows == 0:
        navigation_error = (report.get("navigation") or {}).get("error")
        report["status"] = "failed"
        report["result"].update(
            reason="no row was captured" + (f": {navigation_error}" if navigation_error else ""),
            exitCode=EXIT_STATE)
        return EXIT_STATE
    if verdict["rowsMatch"]:
        report["status"] = "mismatch"
        report["result"].update(
            reason=(
                f"captured exactly {rows} rows (== calculationFrames) but the canonical trace "
                f"hash {canonical_sha} differs from the manifest {expected_sha}; the bytes were "
                f"not adjusted in any way"
            ),
            exitCode=EXIT_STATE,
        )
        return EXIT_STATE
    report["status"] = "partial"
    report["result"].update(
        reason=(
            f"captured {rows} of {expected_rows} calculation frames (ended={verdict['ended']}, "
            f"naturalReplayExit={verdict['naturalReplayExit']}); the canonical trace hash differs "
            f"from the manifest because the trace is incomplete"
        ),
        exitCode=EXIT_STATE,
    )
    return EXIT_STATE


if __name__ == "__main__":
    raise SystemExit(main())
