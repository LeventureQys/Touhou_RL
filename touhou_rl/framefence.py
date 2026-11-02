"""Frame-fence reader for the research-only TH08 instrumentation build.

The canonical game image cannot prove that a GDI ``BitBlt`` and a
``ReadProcessMemory`` describe the same drawn frame (see
``DATA-ACQUISITION.md`` §4).  The ``fence`` build
(``python third_party/th08/scripts/build.py --build-type fence``) adds a small
publisher at the end of ``GameWindow::Render`` that, for every *presented*
frame:

1. writes the state the frame was drawn from into a named shared-memory block
   (``Local\\th08.frame_fence.v1``),
2. bumps a monotonically increasing *ticket*,
3. optionally copies the presented back buffer into the block,
4. optionally **holds** the game inside the fence until this reader
   acknowledges the ticket.

While the game is held (``holdPhase == 1``) no logic frame can advance and no
new present can occur, so a capture and a memory read issued in that window are
provably the same frame.  The ticket's digests (FNV-1a over the bullet pool and
the enemy array) let the reader *verify* that its bulk ``ReadProcessMemory``
calls reproduced exactly the state the game published; the back-buffer copy lets
it verify that the pixels it captured are exactly the pixels the game drew.

Nothing here writes to the game process: the acknowledgement, the hold policy
and the enable flag travel through the shared section the game created.
"""

from __future__ import annotations

import ctypes
import struct
import time
from dataclasses import dataclass, field
from ctypes import wintypes as wt

import numpy as np

from . import spec

# ---------------------------------------------------------------------------
# ABI of the shared section (mirror of third_party/th08/src/framefence.cpp)
# ---------------------------------------------------------------------------
FENCE_MAPPING_NAME = r"Local\th08.frame_fence.v1"
FENCE_MAGIC = 0x38454E46  # 'FNE8'
FENCE_VERSION = 1

CONTROL_OFFSET = 0x000
TELEMETRY_OFFSET = 0x020
PIXELS_OFFSET = 0x1000

PIXEL_WIDTH = 640
PIXEL_HEIGHT = 480
PIXEL_STRIDE = PIXEL_WIDTH * 3
PIXEL_BYTES = PIXEL_STRIDE * PIXEL_HEIGHT
TOTAL_BYTES = PIXELS_OFFSET + PIXEL_BYTES

CTL_ENABLE = 0x0001
CTL_PIXELS = 0x0002
CTL_DIGEST_ALWAYS = 0x0004

TICKET_NONE = 0xFFFFFFFF
DIGEST_NONE = 0xFFFFFFFF

CONTROL_FIELDS = {
    "flags": 0x00,
    "hold_every": 0x04,
    "hold_min_us": 0x08,
    "hold_max_us": 0x0C,
    "ack_ticket": 0x10,
    "reset_request": 0x14,
    "reader_heartbeat": 0x18,
    "reserved0": 0x1C,
}

TELEMETRY_FIELDS = {
    "magic": 0x00,
    "version": 0x04,
    "block_size": 0x08,
    "pid": 0x0C,
    "flags": 0x10,
    "ticket": 0x14,
    "logic_frame": 0x18,
    "stage_frames": 0x1C,
    "run_frames": 0x20,
    "player_x_bits": 0x24,
    "player_y_bits": 0x28,
    "player_vx_bits": 0x2C,
    "player_vy_bits": 0x30,
    "player_state": 0x34,
    "focus_mode": 0x38,
    "input_mask": 0x3C,
    "gui_input_current": 0x40,
    "active_bullets": 0x44,
    "bullet_digest": 0x48,
    "enemy_digest": 0x4C,
    "digest_ticket": 0x50,
    "hold_phase": 0x54,
    "hold_count": 0x58,
    "hold_last_us": 0x5C,
    "hold_max_us": 0x60,
    "hold_budget_skips": 0x64,
    "pixel_ticket": 0x68,
    "pixel_width": 0x6C,
    "pixel_height": 0x70,
    "pixel_stride": 0x74,
    "pixel_format": 0x78,
    "pixel_offset": 0x7C,
    "qpc_high": 0x80,
    "qpc_low": 0x84,
}
TELEMETRY_SIZE = 0x88

FILE_MAP_WRITE = 0x0002
FILE_MAP_READ = 0x0004

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.OpenFileMappingW.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
kernel32.OpenFileMappingW.restype = wt.HANDLE
kernel32.MapViewOfFile.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.c_size_t]
kernel32.MapViewOfFile.restype = wt.LPVOID
kernel32.UnmapViewOfFile.argtypes = [wt.LPCVOID]
kernel32.CloseHandle.argtypes = [wt.HANDLE]


@dataclass(slots=True)
class FenceTelemetry:
    """One decoded telemetry block (raw integers plus convenience floats)."""

    ticket: int = 0
    logic_frame: int = 0
    stage_frames: int = 0
    run_frames: int = 0
    player_state: int = 0
    focus_mode: int = 0
    input_mask: int = 0
    gui_input_current: int = 0
    active_bullets: int = 0
    bullet_digest: int = DIGEST_NONE
    enemy_digest: int = DIGEST_NONE
    digest_ticket: int = DIGEST_NONE
    hold_phase: int = 0
    hold_count: int = 0
    hold_last_us: int = 0
    hold_max_us: int = 0
    hold_budget_skips: int = 0
    pixel_ticket: int = TICKET_NONE
    flags: int = 0
    qpc: int = 0
    _bits: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def player_x(self) -> float:
        return struct.unpack("<f", struct.pack("<I", self._bits[0]))[0]

    @property
    def player_y(self) -> float:
        return struct.unpack("<f", struct.pack("<I", self._bits[1]))[0]

    @property
    def player_vx(self) -> float:
        return struct.unpack("<f", struct.pack("<I", self._bits[2]))[0]

    @property
    def player_vy(self) -> float:
        return struct.unpack("<f", struct.pack("<I", self._bits[3]))[0]

    def player_bits(self) -> tuple[int, int, int, int]:
        return self._bits


class FrameFence:
    """Mapped view of the game's frame-fence shared section."""

    def __init__(self, handle: int, address: int, *, owns_view: bool = True):
        self._handle = handle
        self._address = address
        self._owns_view = owns_view
        self._u32 = (ctypes.c_uint32 * (TOTAL_BYTES // 4)).from_address(address)
        self.telemetry_reads = 0

    # -- lifecycle --------------------------------------------------------
    @classmethod
    def open(cls, timeout: float = 10.0, poll: float = 0.02) -> "FrameFence":
        """Open the section, waiting for the game to publish it."""
        deadline = time.perf_counter() + timeout
        last_error = None
        while time.perf_counter() < deadline:
            handle = kernel32.OpenFileMappingW(FILE_MAP_WRITE | FILE_MAP_READ, False, FENCE_MAPPING_NAME)
            if not handle:
                last_error = ctypes.get_last_error()
                time.sleep(poll)
                continue
            address = kernel32.MapViewOfFile(handle, FILE_MAP_WRITE | FILE_MAP_READ, 0, 0, TOTAL_BYTES)
            if not address:
                last_error = ctypes.get_last_error()
                kernel32.CloseHandle(handle)
                time.sleep(poll)
                continue
            fence = cls(handle, address)
            if fence.valid():
                return fence
            fence.close()
            last_error = "magic/version mismatch"
            time.sleep(poll)
        raise TimeoutError(f"frame fence {FENCE_MAPPING_NAME!r} not available (last error {last_error})")

    def close(self) -> None:
        if self._address and self._owns_view:
            kernel32.UnmapViewOfFile(ctypes.c_void_p(self._address))
        self._address = 0
        if self._handle:
            kernel32.CloseHandle(self._handle)
            self._handle = 0

    def __enter__(self) -> "FrameFence":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- raw access -------------------------------------------------------
    def _read(self, base: int, offset: int) -> int:
        return int(self._u32[(base + offset) // 4])

    def _write(self, base: int, offset: int, value: int) -> None:
        self._u32[(base + offset) // 4] = ctypes.c_uint32(value & 0xFFFFFFFF)

    @property
    def ticket(self) -> int:
        return self._read(TELEMETRY_OFFSET, TELEMETRY_FIELDS["ticket"])

    @property
    def hold_phase(self) -> int:
        return self._read(TELEMETRY_OFFSET, TELEMETRY_FIELDS["hold_phase"])

    # -- validation -------------------------------------------------------
    def valid(self) -> bool:
        return (
            self._read(TELEMETRY_OFFSET, TELEMETRY_FIELDS["magic"]) == FENCE_MAGIC
            and self._read(TELEMETRY_OFFSET, TELEMETRY_FIELDS["version"]) == FENCE_VERSION
            and self._read(TELEMETRY_OFFSET, TELEMETRY_FIELDS["block_size"]) == TOTAL_BYTES
        )

    def header(self) -> dict[str, int]:
        return {
            "pid": self._read(TELEMETRY_OFFSET, TELEMETRY_FIELDS["pid"]),
            "block_size": self._read(TELEMETRY_OFFSET, TELEMETRY_FIELDS["block_size"]),
            "magic": self._read(TELEMETRY_OFFSET, TELEMETRY_FIELDS["magic"]),
            "version": self._read(TELEMETRY_OFFSET, TELEMETRY_FIELDS["version"]),
        }

    # -- control ----------------------------------------------------------
    def control(self) -> dict[str, int]:
        return {
            name: self._read(CONTROL_OFFSET, offset) for name, offset in CONTROL_FIELDS.items()
        }

    def configure(
        self,
        *,
        enable: bool = True,
        pixels: bool = False,
        digest_always: bool = False,
        hold_every: int = 1,
        hold_min_us: int = 0,
        hold_max_us: int = 60_000,
        reset: bool = False,
    ) -> None:
        flags = 0
        if enable:
            flags |= CTL_ENABLE
        if pixels:
            flags |= CTL_PIXELS
        if digest_always:
            flags |= CTL_DIGEST_ALWAYS
        self._write(CONTROL_OFFSET, CONTROL_FIELDS["hold_every"], hold_every)
        self._write(CONTROL_OFFSET, CONTROL_FIELDS["hold_min_us"], hold_min_us)
        self._write(CONTROL_OFFSET, CONTROL_FIELDS["hold_max_us"], hold_max_us)
        self._write(CONTROL_OFFSET, CONTROL_FIELDS["flags"], flags)
        if reset:
            self._write(CONTROL_OFFSET, CONTROL_FIELDS["reset_request"], 1)

    def disable(self) -> None:
        self._write(CONTROL_OFFSET, CONTROL_FIELDS["flags"], 0)
        self._write(CONTROL_OFFSET, CONTROL_FIELDS["hold_every"], 0)

    def ack(self, ticket: int) -> None:
        self._write(CONTROL_OFFSET, CONTROL_FIELDS["ack_ticket"], ticket)

    def heartbeat(self, value: int) -> None:
        self._write(CONTROL_OFFSET, CONTROL_FIELDS["reader_heartbeat"], value)

    # -- telemetry --------------------------------------------------------
    def telemetry(self) -> FenceTelemetry:
        """Decode the telemetry block (two reads of `ticket` to flag changes)."""
        raw = ctypes.string_at(self._address + TELEMETRY_OFFSET, TELEMETRY_SIZE)
        words = struct.unpack("<34I", raw)
        self.telemetry_reads += 1
        get = lambda name: words[TELEMETRY_FIELDS[name] // 4]  # noqa: E731
        return FenceTelemetry(
            ticket=get("ticket"),
            logic_frame=get("logic_frame"),
            stage_frames=get("stage_frames"),
            run_frames=get("run_frames"),
            player_state=get("player_state"),
            focus_mode=get("focus_mode"),
            input_mask=get("input_mask"),
            gui_input_current=get("gui_input_current"),
            active_bullets=get("active_bullets"),
            bullet_digest=get("bullet_digest"),
            enemy_digest=get("enemy_digest"),
            digest_ticket=get("digest_ticket"),
            hold_phase=get("hold_phase"),
            hold_count=get("hold_count"),
            hold_last_us=get("hold_last_us"),
            hold_max_us=get("hold_max_us"),
            hold_budget_skips=get("hold_budget_skips"),
            pixel_ticket=get("pixel_ticket"),
            flags=get("flags"),
            qpc=(get("qpc_high") << 32) | get("qpc_low"),
            _bits=(
                get("player_x_bits"),
                get("player_y_bits"),
                get("player_vx_bits"),
                get("player_vy_bits"),
            ),
        )

    # -- pixels -----------------------------------------------------------
    def pixels(self) -> np.ndarray:
        """Copy of the published back buffer as ``uint8[480, 640, 3]`` RGB."""
        base = self._address + PIXELS_OFFSET
        array = np.ctypeslib.as_array((ctypes.c_uint8 * PIXEL_BYTES).from_address(base))
        return array.reshape(PIXEL_HEIGHT, PIXEL_WIDTH, 3).copy()


# ---------------------------------------------------------------------------
# Digests: the same FNV-1a the game computes inside the fence
# ---------------------------------------------------------------------------
def fnv1a32(data: bytes) -> int:
    hash_value = 2166136261
    for byte in data:
        hash_value = ((hash_value ^ byte) * 16777619) & 0xFFFFFFFF
    return hash_value


class FenceStateDigest:
    """Recomputes the game's bullet/enemy digests from our own memory reads."""

    def __init__(self, reader) -> None:
        self.reader = reader
        self._enemy_buf = ctypes.create_string_buffer(spec.ENEMY_SIZE * spec.ENEMY_SCAN_COUNT)

    def bullet_digest(self, field: "np.ndarray" = None) -> int:
        from .state import BulletField  # local import: avoid a cycle at import time

        bullets = field if field is not None else self.reader.read_bullets()
        raw = bullets.raw  # type: ignore[union-attr]
        records = np.zeros((raw.shape[0], 12), dtype=np.uint8)
        # u32 state (u16 zero-extended), f32 x bits, f32 y bits -- the exact
        # three little-endian words third_party/th08/src/framefence.cpp hashes.
        records[:, 0:2] = raw[:, spec.OFF_BULLET_STATE : spec.OFF_BULLET_STATE + 2]
        records[:, 4:8] = raw[:, spec.OFF_BULLET_POSITION : spec.OFF_BULLET_POSITION + 4]
        records[:, 8:12] = raw[:, spec.OFF_BULLET_POSITION + 4 : spec.OFF_BULLET_POSITION + 8]
        return fnv1a32(records.tobytes())

    def enemy_digest(self) -> int:
        mem = self.reader.mem
        mem.read_into(spec.ADDR_ENEMY_MANAGER + spec.OFF_EM_ENEMIES, self._enemy_buf)
        raw = np.frombuffer(self._enemy_buf, dtype=np.uint8).reshape(
            spec.ENEMY_SCAN_COUNT, spec.ENEMY_SIZE
        )
        records = np.zeros((raw.shape[0], 16), dtype=np.uint8)
        records[:, 0:4] = raw[:, spec.OFF_ENEMY_FLAGS1 : spec.OFF_ENEMY_FLAGS1 + 4]
        records[:, 4:8] = raw[:, spec.OFF_ENEMY_LIFE : spec.OFF_ENEMY_LIFE + 4]
        records[:, 8:12] = raw[:, spec.OFF_ENEMY_POSITION : spec.OFF_ENEMY_POSITION + 4]
        records[:, 12:16] = raw[:, spec.OFF_ENEMY_POSITION + 4 : spec.OFF_ENEMY_POSITION + 8]
        return fnv1a32(records.tobytes())


# ---------------------------------------------------------------------------
# Same-frame sampler
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FrameSample:
    """One capture that the fence pinned to a single drawn frame."""

    ticket: int = 0
    telemetry: FenceTelemetry | None = None
    state: object | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    timings_us: dict[str, float] = field(default_factory=dict)
    pixels: np.ndarray | None = None
    fence_pixels: np.ndarray | None = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return all(self.checks.values())


class FenceFrameSampler:
    """Implements the ticket -> capture -> verify -> ack protocol.

    ``next()`` returns a sample only when the whole capture happened inside one
    hold window; otherwise it retries (a hold that expires mid-read is reported
    in ``note``).
    """

    def __init__(
        self,
        session,
        fence: FrameFence,
        *,
        with_pixels: bool = True,
        with_enemies: bool = True,
        verify_pixels: bool = True,
        on_hold=None,
        accept_current: bool = False,
    ):
        self.session = session
        self.fence = fence
        self.reader = session.reader
        assert self.reader is not None
        self.digest = FenceStateDigest(self.reader)
        self.with_pixels = with_pixels
        self.with_enemies = with_enemies
        self.verify_pixels = verify_pixels
        # Optional hook, called while the game is still stopped inside the fence
        # and before the capture/state reads.  Used by the P0 re-run to type keys
        # whose effect is then attributable to a *known* logic frame instead of
        # to "somewhere between two presents".  It must not block: the game is
        # spinning on its hold budget meanwhile.
        self.on_hold = on_hold
        # By default a sample must be a ticket we have never seen, so the read
        # begins on a *change*.  When the caller pauses between samples (writing
        # a PNG, JSON, disk flush) the game has already presented, published and
        # entered the next hold; that hold then waits out its whole budget while
        # this sampler waits for a *newer* ticket.  `accept_current=True` consumes
        # the ticket that is holding right now instead, which removes that waste.
        self.accept_current = accept_current
        self.last_ticket = 0
        self.samples = 0
        self.retries = 0
        self.last_hold_us = 0
        self.on_hold_calls = 0
        self.on_hold_errors: list[str] = []

    def _still_held(self, ticket: int) -> bool:
        return self.fence.ticket == ticket and self.fence.hold_phase == 1

    def next(self, timeout: float = 2.0, attempts: int = 8) -> FrameSample | None:
        for _ in range(attempts):
            sample = self._attempt(timeout)
            if sample is not None:
                return sample
        return None

    def _attempt(self, timeout: float) -> FrameSample | None:
        wait_started = time.perf_counter()
        deadline = wait_started + timeout
        ticket = 0
        if self.accept_current:
            current = self.fence.ticket
            if current and current != self.last_ticket and self.fence.hold_phase == 1:
                ticket = current
        if not ticket:
            start_ticket = self.fence.ticket
            while time.perf_counter() < deadline:
                current = self.fence.ticket
                if current != start_ticket and self.fence.hold_phase == 1:
                    ticket = current
                    break
        if not ticket:
            return None

        t0 = wait_started
        t_hold = time.perf_counter()
        telemetry = self.fence.telemetry()
        if telemetry.ticket != ticket or telemetry.hold_phase != 1:
            self.retries += 1
            return None

        if self.on_hold is not None:
            # The tick the hook acts on is `telemetry.logic_frame`; anything it
            # sends can only be consumed by the *next* logic frame, because this
            # one has already been updated and presented.
            try:
                self.on_hold(telemetry)
                self.on_hold_calls += 1
            except Exception as exc:  # pragma: no cover - defensive
                self.on_hold_errors.append(repr(exc))

        if self.verify_pixels:
            blit_a = np.ascontiguousarray(self.session.grab())
            if not self._still_held(ticket):
                self.retries += 1
                return None
            blit_b = np.ascontiguousarray(self.session.grab())
            if not self._still_held(ticket):
                self.retries += 1
                return None
        else:
            blit_a = blit_b = None
        t_blit = time.perf_counter()

        state = self.reader.read_state(with_bullets=True)
        bullets = self.reader.read_bullets()
        t_state = time.perf_counter()
        bullet_digest = self.digest.bullet_digest(bullets)
        enemy_digest = None
        if self.with_enemies:
            enemy_digest = self.digest.enemy_digest()
        t_digest = time.perf_counter()

        if not self._still_held(ticket):
            # The hold budget expired while we were reading: the sample we just
            # took may straddle two presents, so it cannot be used.
            self.retries += 1
            self.last_ticket = ticket
            self.fence.ack(ticket)
            return None

        fence_pixels = None
        if self.with_pixels and telemetry.pixel_ticket == ticket:
            fence_pixels = self.fence.pixels()
        t_pixels = time.perf_counter()
        # The hold must still be in force for the ticket we just read: if the
        # budget expired, the game may already have presented a newer frame and
        # the sample would straddle two frames.
        hold_stable = self._still_held(ticket)
        self.fence.ack(ticket)
        self.last_ticket = ticket
        self.samples += 1
        self.last_hold_us = telemetry.hold_last_us

        checks = {
            "hold_stable": hold_stable,
            "player_position": state.player_x == telemetry.player_x
            and state.player_y == telemetry.player_y,
            "player_velocity": state.player_vx == telemetry.player_vx
            and state.player_vy == telemetry.player_vy,
            "player_state": int(state.player_state) == int(telemetry.player_state),
            "focus_mode": int(state.focus_mode) == int(telemetry.focus_mode),
            "gameplay_frame": int(state.gameplay_frame) == int(telemetry.logic_frame),
            "stage_frames": int(state.stage_frames) == int(telemetry.stage_frames),
            "input_mask": int(state.input_mask & 0xFFFF) == int(telemetry.input_mask & 0xFFFF),
            "active_bullets": int(state.active_bullets) == int(telemetry.active_bullets),
            "bullet_digest": bullet_digest == telemetry.bullet_digest
            and telemetry.digest_ticket == ticket,
        }
        if enemy_digest is not None:
            checks["enemy_digest"] = enemy_digest == telemetry.enemy_digest and (
                telemetry.digest_ticket == ticket
            )
        if blit_a is not None and blit_b is not None:
            checks["pixels_frozen"] = bool(np.array_equal(blit_a, blit_b))
        if fence_pixels is not None and blit_a is not None:
            checks["pixels_match_backbuffer"] = bool(np.array_equal(fence_pixels, blit_a))

        return FrameSample(
            ticket=ticket,
            telemetry=telemetry,
            state=state,
            checks=checks,
            timings_us={
                "wait": (t_hold - t0) * 1e6,
                "blit": (t_blit - t_hold) * 1e6,
                "state": (t_state - t_blit) * 1e6,
                "digest": (t_digest - t_state) * 1e6,
                "pixels": (t_pixels - t_digest) * 1e6,
                "total": (t_pixels - t_hold) * 1e6,
                "hold": float(telemetry.hold_last_us),
            },
            pixels=blit_a,
            fence_pixels=fence_pixels,
        )


def legacy_sample(session, fence: FrameFence, digest: FenceStateDigest) -> dict[str, object]:
    """One read in the *old* style: no hold, no ticket pinning.

    Reproduces what the existing pipeline does (``read_state`` + ``read_bullets``
    while the game free-runs) and reports whether the bytes it read equal any
    single published frame.  Used to quantify the problem the fence removes.
    """
    telemetry = fence.telemetry()
    state = session.reader.read_state(with_bullets=True)
    bullets = session.reader.read_bullets()
    bullet_digest = digest.bullet_digest(bullets)
    enemy_digest = digest.enemy_digest()
    after = fence.telemetry()
    return {
        "ticket_at_start": int(telemetry.ticket),
        "ticket_at_end": int(after.ticket),
        "ticket_unchanged": telemetry.ticket == after.ticket,
        "digest_ticket": int(telemetry.digest_ticket),
        "bullet_digest_match": bullet_digest == telemetry.bullet_digest,
        "enemy_digest_match": enemy_digest == telemetry.enemy_digest,
        "player_match": state.player_x == telemetry.player_x
        and state.player_y == telemetry.player_y,
        "frame_match": int(state.gameplay_frame) == int(telemetry.logic_frame),
    }
