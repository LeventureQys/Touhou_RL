"""Control-protocol client for the TH08 harness (03 §4.2, 10-P0 §3).

The engine speaks a line-oriented ASCII protocol over a pipe pair: one request
per line, one reply per line.  Bulk data does not travel here — it lands in the
shared block (see :mod:`touhou_rl.v400.gym.shm`).

Starting the process is deliberately explicit: this object owns the child's
lifetime and must be closed, so it doubles as a context manager.

Entity rows (v4.0.0 P1)
-----------------------
``BULLETS``, ``ENEMIES`` and ``LASERS`` kept their G0 prefixes and gained
appended fields for hazard geometry (11-P1 §2.1), and ``HAZARDS`` returns all
three families in one reply.  The decoders below are length-adaptive: they read
the prefix they know and take whatever appended fields are present, so a row
from a pre-P1 engine and a row from the current one both parse, and the missing
fields are simply ``None``.  Nothing here re-scales a value the engine already
scaled; the raw width is what ships.
"""

from __future__ import annotations

import os
import queue
import select
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT = 300.0

#: Capacity the engine gives one ``HAZARDS`` reply (rlharness.cpp
#: ``HAZARDS_REPLY_CAPACITY``).  A frame that would not fit fails closed with
#: ``ERR hazards-too-large`` rather than being truncated.
HAZARDS_REPLY_CAPACITY = 256 * 1024

#: Roles the engine accepts from ``--rl-role``.  ``env`` is the default (with
#: rendering); ``shadow`` is the null device, which needs no display and is what
#: the deterministic replay checks use.
ROLES = ("env", "shadow")


class HarnessError(RuntimeError):
    """The engine replied with ``ERR`` or died."""


# ---------------------------------------------------------------------------
# Entity rows: BULLETS / ENEMIES / LASERS / HAZARDS
#
# Field counts of the G0 prefix, for reference:
#   bullet  9  slot:state:color:posXyz:velXy:timer
#   enemy   7  slot:flags1:posXyz:life:timer3318
#   laser  11  slot:state:posXyz:angle:startOffset:endOffset:currentWidth:width:timer
# The P1 engine appends 2 / 5 / 1 fields respectively and is documented in
# rlharness.cpp; every decoder below tolerates any count at or above the prefix.
# EXP-P1-021 appends 6 more to the laser row (startTime, hitboxStartTime,
# duration, despawnDuration, hitboxEndDelay, damageable), so a laser row may
# carry 12 to 18 fields and every one of them is optional here.
# ---------------------------------------------------------------------------

BULLET_PREFIX_FIELDS = 9
ENEMY_PREFIX_FIELDS = 7
LASER_PREFIX_FIELDS = 11
#: Fields a laser row carries once the engine exports its judgement window.
LASER_WINDOW_FIELDS = 18

#: ``ENEMY_FLAG_COLLISION`` (EnemyManager.hpp:71) — enemy already freed by ``Deactivate()``.
ENEMY_FLAG_ACTIVE = 1 << 0
ENEMY_FLAG_COLLISION = 1 << 2

#: ``BULLET_STATE_FIRED``: the only bullet state that can kill (11-P1 §2.1).
BULLET_STATE_FIRED = 1

#: ``LaserState`` (BulletManager.hpp:306-311).  The hitbox windows are per state
#: (BulletManager.cpp:1096-1142) and EXP-P1-021 exports them; these names exist
#: so a fallback path can still say which state it saw.
LASER_STATE_STARTING = 0
LASER_STATE_ACTIVE = 1
LASER_STATE_DESPAWNING = 2


# ---------------------------------------------------------------------------
# Replay frame events (``EVENTS``)
#
# ``ReplayManager::frameEventFlags`` is a u16 that gameplay systems OR into
# during a frame and that ``ReplayManager::CaptureFrameSyncState`` clears at the
# start of the next one.  These are the engine's own names and values
# (ReplayManager.hpp:161-172); they are deliberately *not* part of the 25
# tracked ``HASH``/``STATE`` fields.
# ---------------------------------------------------------------------------
REPLAY_FRAME_EVENT_NONE = 0x0000
REPLAY_FRAME_EVENT_BOMB_STARTED = 0x0001
REPLAY_FRAME_EVENT_PLAYER_HIT = 0x0002
REPLAY_FRAME_EVENT_PLAYER_DEATH_COMMITTED = 0x0004
REPLAY_FRAME_EVENT_ENEMY_REMOVED = 0x0020
REPLAY_FRAME_EVENT_ITEM_COLLECTED = 0x0040
REPLAY_FRAME_EVENT_PAUSE_RECORDED = 0x0100
REPLAY_FRAME_EVENT_PLAYER_DYING_STARTED = 0x0200
REPLAY_FRAME_EVENT_EFFECT_SPAWNED = 0x0400
REPLAY_FRAME_EVENT_BULLET_PATTERN_SPAWNED = 0x0800
REPLAY_FRAME_EVENT_ENEMY_SPAWNED = 0x1000
REPLAY_FRAME_EVENT_PLAYER_GRAZED = 0x2000

#: Bit value -> engine enum name, for reports that must not decode by hand.
REPLAY_FRAME_EVENT_NAMES = {
    REPLAY_FRAME_EVENT_BOMB_STARTED: "BOMB_STARTED",
    REPLAY_FRAME_EVENT_PLAYER_HIT: "PLAYER_HIT",
    REPLAY_FRAME_EVENT_PLAYER_DEATH_COMMITTED: "PLAYER_DEATH_COMMITTED",
    REPLAY_FRAME_EVENT_ENEMY_REMOVED: "ENEMY_REMOVED",
    REPLAY_FRAME_EVENT_ITEM_COLLECTED: "ITEM_COLLECTED",
    REPLAY_FRAME_EVENT_PAUSE_RECORDED: "PAUSE_RECORDED",
    REPLAY_FRAME_EVENT_PLAYER_DYING_STARTED: "PLAYER_DYING_STARTED",
    REPLAY_FRAME_EVENT_EFFECT_SPAWNED: "EFFECT_SPAWNED",
    REPLAY_FRAME_EVENT_BULLET_PATTERN_SPAWNED: "BULLET_PATTERN_SPAWNED",
    REPLAY_FRAME_EVENT_ENEMY_SPAWNED: "ENEMY_SPAWNED",
    REPLAY_FRAME_EVENT_PLAYER_GRAZED: "PLAYER_GRAZED",
}


def _bits_float(token: str) -> float:
    """Decode one ``%08x`` field emitted by ``FloatBits()`` back to a float."""

    return struct.unpack("<f", struct.pack("<I", int(token, 16) & 0xFFFFFFFF))[0]


def _row_fields(row: str, minimum: int, command: str) -> list[str]:
    """Split one ``:``-separated row, rejecting a truncated prefix.

    A longer row is accepted: the engine is allowed to append more fields and
    the caller keeps whatever prefix it understands.
    """

    parts = row.split(":")
    if len(parts) < minimum:
        raise HarnessError(
            f"{command} row has {len(parts)} fields, needs at least {minimum}: {row!r}"
        )
    return parts


@dataclass(frozen=True)
class BulletRow:
    """One live bullet slot.

    ``collision_size`` is the **full width** in playfield pixels, exactly as the
    engine stores it (``Bullet::sprites.collisionSize.x``).  The engine's own
    collision test builds ``position ± size / 2`` (Player.cpp:345-348), so the
    lethal half-width is :attr:`half_width`.  Treating ``collision_size`` as a
    half-width is the specific error 11-P1 §2.1 calls out in the old teacher.
    """

    slot: int
    state: int
    color: int
    position: tuple[float, float, float]
    velocity: tuple[float, float]
    timer: int
    collision_size: float | None = None
    collision_disabled: int | None = None

    @property
    def half_width(self) -> float | None:
        return None if self.collision_size is None else self.collision_size / 2.0

    @property
    def lethal(self) -> bool:
        """``state == FIRED`` and no ECL transform disabled the hitbox."""

        return self.state == BULLET_STATE_FIRED and not self.collision_disabled


@dataclass(frozen=True)
class EnemyRow:
    """One active enemy slot.

    ``hitbox_width``/``hitbox_height`` are the raw ``Enemy::hitboxDimensions``.
    ``EnemyManager.cpp:852`` feeds ``hitboxDimensions / 1.5`` to
    ``Player::CheckLethalCollision``, which treats its argument as a **full**
    width, so the lethal half extent is ``hitboxDimensions / 3``
    (:attr:`lethal_half_extent`).
    """

    slot: int
    flags1: int
    position: tuple[float, float, float]
    life: int
    timer: int
    world_position: tuple[float, float, float] | None = None
    hitbox_width: float | None = None
    hitbox_height: float | None = None

    @property
    def has_collision(self) -> bool:
        return bool(self.flags1 & ENEMY_FLAG_COLLISION)

    @property
    def lethal_half_extent(self) -> tuple[float, float] | None:
        if self.hitbox_width is None or self.hitbox_height is None:
            return None
        return (self.hitbox_width / 3.0, self.hitbox_height / 3.0)


@dataclass(frozen=True)
class LaserRow:
    """One allocated laser slot.

    The hitbox interval and thickness are derived by the caller from
    ``state``/``timer``/``start_offset``/``end_offset``/``width``; this row is
    the engine's own numbers, never a rescaled copy.

    EXP-P1-021 adds the engine's **judgement window** (``Laser`` fields
    ``startTime``/``hitboxStartTime``/``duration``/``despawnDuration``/
    ``hitboxEndDelay``) and ``damageable``: the engine's own answer to "does the
    next simulated frame evaluate this laser's hitbox", derived from
    BulletManager.cpp:1096-1142.  A row from a pre-EXP-P1-021 engine leaves all
    six ``None`` and :attr:`carries_hitbox` falls back to the old state-only
    rule, so both engines parse.
    """

    slot: int
    state: int
    position: tuple[float, float, float]
    angle: float
    start_offset: float
    end_offset: float
    current_width: float
    width: float
    timer: int
    in_use: int | None = None
    start_time: int | None = None
    hitbox_start_time: int | None = None
    duration: int | None = None
    despawn_duration: int | None = None
    hitbox_end_delay: int | None = None
    damageable: int | None = None

    @property
    def windows_available(self) -> bool:
        """True when the row carries the EXP-P1-021 judgement window."""

        return self.damageable is not None

    @property
    def carries_hitbox(self) -> bool:
        """Does this laser's hitbox apply to the next simulated frame?

        Uses the engine's ``damageable`` when the row has it.  Otherwise (a
        pre-EXP-P1-021 engine) it degrades to the old rule — only
        ``LASER_STATE_ACTIVE`` counts — which misses a ``LASER_STATE_STARTING``
        beam whose ``timer`` already passed ``hitboxStartTime`` and a despawn
        tail still inside ``hitboxEndDelay``.
        """

        if self.damageable is not None:
            return bool(int(self.damageable))
        return int(self.state) == LASER_STATE_ACTIVE


def parse_bullet_row(row: str) -> BulletRow:
    parts = _row_fields(row, BULLET_PREFIX_FIELDS, "BULLETS")
    return BulletRow(
        slot=int(parts[0]),
        state=int(parts[1]),
        color=int(parts[2]),
        position=(_bits_float(parts[3]), _bits_float(parts[4]), _bits_float(parts[5])),
        velocity=(_bits_float(parts[6]), _bits_float(parts[7])),
        timer=int(parts[8]),
        collision_size=_bits_float(parts[9]) if len(parts) > 9 else None,
        collision_disabled=int(parts[10]) if len(parts) > 10 else None,
    )


def parse_enemy_row(row: str) -> EnemyRow:
    parts = _row_fields(row, ENEMY_PREFIX_FIELDS, "ENEMIES")
    return EnemyRow(
        slot=int(parts[0]),
        flags1=int(parts[1], 16),
        position=(_bits_float(parts[2]), _bits_float(parts[3]), _bits_float(parts[4])),
        life=int(parts[5], 16),
        timer=int(parts[6]),
        world_position=(
            (_bits_float(parts[7]), _bits_float(parts[8]), _bits_float(parts[9]))
            if len(parts) > 9
            else None
        ),
        hitbox_width=_bits_float(parts[10]) if len(parts) > 10 else None,
        hitbox_height=_bits_float(parts[11]) if len(parts) > 11 else None,
    )


def parse_laser_row(row: str) -> LaserRow:
    parts = _row_fields(row, LASER_PREFIX_FIELDS, "LASERS")
    return LaserRow(
        slot=int(parts[0]),
        state=int(parts[1]),
        position=(_bits_float(parts[2]), _bits_float(parts[3]), _bits_float(parts[4])),
        angle=_bits_float(parts[5]),
        start_offset=_bits_float(parts[6]),
        end_offset=_bits_float(parts[7]),
        current_width=_bits_float(parts[8]),
        width=_bits_float(parts[9]),
        timer=int(parts[10]),
        in_use=int(parts[11]) if len(parts) > 11 else None,
        start_time=int(parts[12]) if len(parts) > 12 else None,
        hitbox_start_time=int(parts[13]) if len(parts) > 13 else None,
        duration=int(parts[14]) if len(parts) > 14 else None,
        despawn_duration=int(parts[15]) if len(parts) > 15 else None,
        hitbox_end_delay=int(parts[16]) if len(parts) > 16 else None,
        damageable=int(parts[17]) if len(parts) > 17 else None,
    )


@dataclass(frozen=True)
class PlayerSnapshot:
    """``PLAYER`` reply, decoded.

    ``hp`` is TH08's death resource: the game has no separate HP bar and
    ``ZunGlobals::livesRemaining`` is what reaches zero before a death, so the
    wires carry that value both as raw float bits (``hp``) and as the integer a
    HUD would show (``lives``).

    The four speeds come from the loaded ``.sht`` files the engine itself reads
    (``primaryShtFile`` unfocused, ``secondaryShtFile`` focused; Player.cpp:
    783-806).  They are per-shot data, not constants, so different teams report
    different numbers.  ``speed_*`` are ``None`` when an older engine or an
    unloaded shot file left them out.
    """

    frame: int
    hp: float
    lives: int
    bombs: float
    power: float
    focus: int
    is_youkai: int
    player_state: int
    invuln_state: int
    position: tuple[float, float]
    hurtbox_min: tuple[float, float] | None
    hurtbox_max: tuple[float, float] | None
    speed_normal_axial: float | None = None
    speed_normal_diagonal: float | None = None
    speed_focused_axial: float | None = None
    speed_focused_diagonal: float | None = None

    @property
    def speeds_available(self) -> bool:
        return None not in (
            self.speed_normal_axial,
            self.speed_normal_diagonal,
            self.speed_focused_axial,
            self.speed_focused_diagonal,
        )

    def speed(self, *, focused: bool) -> tuple[float | None, float | None]:
        """``(axial, diagonal)`` for a focused or unfocused step."""

        if focused:
            return (self.speed_focused_axial, self.speed_focused_diagonal)
        return (self.speed_normal_axial, self.speed_normal_diagonal)


@dataclass(frozen=True)
class HazardFrame:
    """One ``HAZARDS`` reply: the whole current frame in a single round trip.

    ``bullet_count``/``laser_count``/``enemy_count`` are the engine's totals, so
    they stay comparable with the ``count=`` of the paged ``BULLETS`` /
    ``LASERS`` / ``ENEMIES`` replies even when a ``limit`` capped the rows.
    """

    frame: int
    bullet_count: int
    laser_count: int
    enemy_count: int
    bullets: tuple[BulletRow, ...] = ()
    lasers: tuple[LaserRow, ...] = ()
    enemies: tuple[EnemyRow, ...] = ()
    raw_bytes: int = 0


@dataclass(frozen=True)
class EventFrame:
    """One ``EVENTS`` reply: the replay frame-event bitmask of the last frame.

    The engine clears the word at the start of each frame and gameplay systems
    OR bits into it, so after ``STEP`` this is the events of the last advanced
    frame.  ``frame_event_flags`` is the raw u16; no bit is invented here.
    """

    frame: int
    frame_event_flags: int

    @property
    def player_hit(self) -> bool:
        """``REPLAY_FRAME_EVENT_PLAYER_HIT`` (0x2) was set."""

        return bool(self.frame_event_flags & REPLAY_FRAME_EVENT_PLAYER_HIT)

    @property
    def names(self) -> tuple[str, ...]:
        """Engine names of the set bits, in ascending bit order."""

        return tuple(
            name
            for bit, name in sorted(REPLAY_FRAME_EVENT_NAMES.items())
            if self.frame_event_flags & bit
        )


@dataclass(frozen=True)
class HudSnapshot:
    """One ``HUD`` reply (EXP-P2-005): the counters the HUD actually draws.

    ``display_score`` is the field the picture contains: ``Gui.cpp:1299`` draws
    ``globals->displayScore``, which ``GameManager.cpp:535-557`` scrolls toward
    ``globals->score`` by ``(gap >> 5)`` per frame.  ``globals_score`` is the
    authoritative total (the same value as ``STATE`` f4); on the frames where
    the counter is still rolling the two differ and only ``display_score`` is
    observable in pixels.

    ``display_age`` is ``max(0, globals_score - display_score)``, i.e. how far
    the drawn counter still lags in points.  ``display_stale`` is the number of
    consecutive ``STEP`` frames that left ``display_score`` unchanged (0 on the
    frame it moved); ``score_display_step`` is the per-frame roll step, which is
    0 once the counter has settled.  The last two are ``None`` only if an older
    engine omits the appended fields.
    """

    frame: int
    display_score: int
    globals_score: int
    lives: int
    bombs: int
    power: int
    graze: int
    point: int
    time: int
    game_over: int
    display_age: int
    display_stale: int | None = None
    score_display_step: int | None = None

    @property
    def settled(self) -> bool:
        """The drawn counter equals the authoritative score on this frame."""

        return self.display_score == self.globals_score


@dataclass
class ProtocolStats:
    """Cheap accounting so the budget ledger can be filled from real usage."""

    commands: int = 0
    frames: int = 0
    wall_seconds: float = 0.0
    replies: dict[str, int] = field(default_factory=dict)


class HarnessClient:
    """Owns one ``th08-modern --rl-control`` process."""

    def __init__(
        self,
        binary: str | os.PathLike[str],
        data_dir: str | os.PathLike[str],
        *,
        shm_name: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        role: str = "env",
    ) -> None:
        self.binary = Path(binary).expanduser()
        self.data_dir = Path(data_dir).expanduser()
        if not self.binary.is_file():
            raise FileNotFoundError(f"harness binary not found: {self.binary}")
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"game data dir not found: {self.data_dir}")
        if role not in ROLES:
            raise ValueError(f"unknown role: {role!r} (known: {ROLES})")

        self.role = role
        self.shm_name = shm_name
        self.timeout = timeout
        self.stats = ProtocolStats()
        self._proc: subprocess.Popen[bytes] | None = None
        self._buffer = b""
        self._reader_queue: queue.Queue[bytes | BaseException] | None = None
        self._reader_thread: threading.Thread | None = None
        self._env = dict(os.environ)
        # Software rendering keeps a single instance reproducible; the caller can
        # override for the GPU backend once that path is healthy again.
        self._env.update(
            SDL_AUDIODRIVER="dummy",
            LIBGL_ALWAYS_SOFTWARE="1",
            GALLIUM_DRIVER="llvmpipe",
            LP_NUM_THREADS="1",
        )
        if env:
            self._env.update(env)
        if self.role == "shadow":
            # The null device refuses a display on purpose (null_device.cpp
            # ``NullDevicePreflight``): a shadow process must not be able to
            # create a window, so an inherited ``DISPLAY`` is dropped here
            # instead of being left for the engine to trip over.
            self._env.pop("DISPLAY", None)

    # -- lifetime ---------------------------------------------------------
    def __enter__(self) -> "HarnessClient":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("client already started")
        argv = [str(self.binary), "--data-dir", str(self.data_dir), "--rl-control", "0,1"]
        if self.role != "env":
            # ``--rl-role shadow`` is the engine's null-device mode: no SDL window
            # and no GL.  The state export and the 25-field hash are the same; only
            # rendering is gone, which is why a shadow process needs no X server.
            argv += ["--rl-role", self.role]
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            bufsize=0,
        )
        if os.name == "nt":
            self._start_pipe_reader()

    def _start_pipe_reader(self) -> None:
        """Read the Windows anonymous pipe from a blocking helper thread.

        ``select.select`` only supports sockets on Windows, not subprocess
        anonymous pipes.  A daemon reader keeps the blocking operation off the
        request thread while the queue supplies a real timeout to ``_readline``.
        """
        assert self._proc is not None and self._proc.stdout is not None
        result: queue.Queue[bytes | BaseException] = queue.Queue()
        self._reader_queue = result

        def read_pipe() -> None:
            try:
                while True:
                    chunk = self._proc.stdout.read(8192) if self._proc is not None else b""
                    if not chunk:
                        result.put(b"")
                        return
                    result.put(chunk)
            except BaseException as exc:
                result.put(exc)

        self._reader_thread = threading.Thread(
            target=read_pipe, name="th08-control-reader", daemon=True
        )
        self._reader_thread.start()

    def close(self, *, quit_timeout: float = 30.0) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    self._write("QUIT")
                    self._readline(quit_timeout)
                except Exception:
                    pass
            try:
                proc.wait(timeout=quit_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
            self._proc = None
            self._reader_queue = None
            self._reader_thread = None

    # -- transport --------------------------------------------------------
    def _write(self, line: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((line + "\n").encode())
        self._proc.stdin.flush()

    def _readline(self, timeout: float) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if os.name == "nt" and self._reader_queue is not None:
                try:
                    item = self._reader_queue.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    if self._proc.poll() is not None:
                        raise HarnessError(
                            f"harness exited before replying (code {self._proc.returncode})"
                        )
                    continue
                if isinstance(item, BaseException):
                    raise HarnessError(f"control pipe reader failed: {item}") from item
                chunk = item
            else:
                ready, _, _ = select.select([self._proc.stdout], [], [], min(0.25, remaining))
                if not ready:
                    if self._proc.poll() is not None:
                        raise HarnessError(
                            f"harness exited before replying (code {self._proc.returncode})"
                        )
                    continue
                chunk = os.read(self._proc.stdout.fileno(), 8192)
            if not chunk:
                raise HarnessError("harness closed the control pipe")
            self._buffer += chunk
            if b"\n" in self._buffer:
                line, _, self._buffer = self._buffer.partition(b"\n")
                return line.decode("utf-8", "replace")
        raise HarnessError(f"no reply within {timeout:.1f}s")

    def request(self, line: str, timeout: float | None = None) -> str:
        """Send one command, return one reply, raise on ``ERR``."""
        if self._proc is None:
            raise RuntimeError("client is not running")
        started = time.monotonic()
        self._write(line)
        reply = self._readline(self.timeout if timeout is None else timeout)
        self.stats.commands += 1
        self.stats.wall_seconds += time.monotonic() - started
        verb = reply.split(" ", 1)[0]
        self.stats.replies[verb] = self.stats.replies.get(verb, 0) + 1
        if reply.startswith("ERR"):
            raise HarnessError(f"{line!r} -> {reply}")
        return reply

    # -- typed commands ---------------------------------------------------
    def hello(self) -> dict[str, str]:
        return _parse(self.request("HELLO"))

    def frame(self) -> int:
        return int(_parse(self.request("FRAME"))["frame"])

    def state_hash(self) -> tuple[str, int]:
        fields = _parse(self.request("HASH"))
        return fields["hash"], int(fields["frames"])

    def state(self) -> dict[str, int]:
        """The 25 tracked fields as ``{"f0": .., "f24": ..}``."""
        fields = _parse(self.request("STATE"))
        return {key: int(value) for key, value in fields.items() if key.startswith("f")}

    def debug(self) -> dict[str, str]:
        return _parse(self.request("DEBUG"))

    def attach_shm(self, name: str | None = None) -> dict[str, str]:
        name = name or self.shm_name
        if not name:
            raise ValueError("no shared-memory name given")
        reply = _parse(self.request(f"SHM /{name.lstrip('/')}"))
        self.shm_name = name.lstrip("/")
        return reply

    def reset(
        self,
        *,
        stage: int,
        difficulty: int,
        team: int,
        seed: int,
        practice: bool = False,
        timeout: float | None = None,
    ) -> dict[str, str]:
        cmd = f"RESET {stage} {difficulty} {team} {seed} {1 if practice else 0}"
        return _parse(self.request(cmd, timeout=timeout))

    def step(self, mask: int, frames: int) -> dict[str, str]:
        if not 0 <= mask <= 0xFFFF:
            raise ValueError(f"mask out of range: {mask}")
        if frames <= 0:
            raise ValueError(f"frames must be positive, got {frames}")
        fields = _parse(self.request(f"STEP {mask} {frames}"))
        self.stats.frames += int(fields.get("advanced", 0))
        return fields

    def readback(self, enabled: bool | None = None) -> dict[str, str]:
        if enabled is None:
            return _parse(self.request("READBACK"))
        return _parse(self.request(f"READBACK {1 if enabled else 0}"))

    def pixels(self, path: str | None = None) -> dict[str, str]:
        return _parse(self.request(f"PIXELS {path}" if path else "PIXELS"))

    # -- entity rows (v4.0.0 P1) ------------------------------------------
    def bullets(self, offset: int = 0, limit: int = 16) -> list[BulletRow]:
        """One page of live bullet slots (``BULLETS``)."""

        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if not 1 <= limit <= 16:
            raise ValueError(f"BULLETS limit is 1..16, got {limit}")
        fields = _parse(self.request(f"BULLETS {offset} {limit}"))
        return _rows(fields.get("bullets", ""), parse_bullet_row)

    def bullet_count(self) -> int:
        """Total live bullets this frame (the ``count=`` of ``BULLETS``)."""

        return int(_parse(self.request("BULLETS 0 1")).get("count", 0))

    def enemies(self, offset: int = 0, limit: int = 16) -> list[EnemyRow]:
        """One page of active enemy slots (``ENEMIES``)."""

        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if not 1 <= limit <= 16:
            raise ValueError(f"ENEMIES limit is 1..16, got {limit}")
        fields = _parse(self.request(f"ENEMIES {offset} {limit}"))
        return _rows(fields.get("enemies", ""), parse_enemy_row)

    def enemy_count(self) -> int:
        return int(_parse(self.request("ENEMIES 0 1")).get("count", 0))

    def lasers(self, offset: int = 0, limit: int = 16) -> list[LaserRow]:
        """One page of allocated laser slots (``LASERS``)."""

        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if not 1 <= limit <= 16:
            raise ValueError(f"LASERS limit is 1..16, got {limit}")
        fields = _parse(self.request(f"LASERS {offset} {limit}"))
        return _rows(fields.get("lasers", ""), parse_laser_row)

    def laser_count(self) -> int:
        return int(_parse(self.request("LASERS 0 1")).get("count", 0))

    def player(self) -> PlayerSnapshot:
        """The self ship (``PLAYER``).

        Raises :class:`HarnessError` unchanged when the engine refuses:
        ``ERR player-not-ready reset-required`` before a stage is running, or
        ``ERR player-speed-unavailable`` when the shot data is not loaded.  The
        speed table is never guessed here.
        """

        return parse_player_reply(self.request("PLAYER"))

    def hazards(self, limit: int = 0) -> HazardFrame:
        """Every bullet, laser and enemy body of this frame (``HAZARDS``).

        ``limit`` caps each of the three lists independently; ``0`` keeps the
        engine default, which is every slot.  One round trip per frame is what
        makes a 48-frame ``HazardTape`` affordable.
        """

        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        command = "HAZARDS" if limit == 0 else f"HAZARDS {limit}"
        return parse_hazards_reply(self.request(command))

    def events(self) -> EventFrame:
        """The last frame's replay event bitmask (``EVENTS``).

        Raises :class:`HarnessError` when the engine cannot answer: a
        pre-v4.0.0 engine replies ``ERR unknown-command EVENTS`` and a stage
        without a live ``ReplayManager`` replies ``ERR events-unavailable``.
        Neither case is turned into an all-zero bitmask here.
        """

        return parse_events_reply(self.request("EVENTS"))

    def hud(self) -> HudSnapshot:
        """The counters the HUD draws (``HUD``, EXP-P2-005).

        ``display_score`` is what the score digits on screen contain; the
        25-field ``STATE`` carries ``globals_score`` instead.  Raises
        :class:`HarnessError` on an engine that has no ``HUD`` command
        (``ERR unknown-command HUD``) or before a stage is running
        (``ERR hud-not-ready reset-required``) - neither is mapped to zeros.
        """

        return parse_hud_reply(self.request("HUD"))

    # -- child forwarding (EXP-P1-005) ------------------------------------
    def child(self, handle: int, command: str, timeout: float | None = None) -> str:
        """Send one command to a forked child (``CHILD <handle> <command>``).

        The raw reply is returned unchanged; a refusal such as
        ``ERR child-command``, ``ERR hazards-not-ready`` or
        ``ERR events-unavailable`` raises :class:`HarnessError` like every other
        error reply.  Use :meth:`child_snapshot` to decode typed replies.
        """

        handle = int(handle)
        if handle <= 0:
            raise ValueError(f"child handle must be positive, got {handle}")
        command = command.strip()
        if not command or "\n" in command or "\r" in command:
            raise ValueError(f"child command must be one non-empty line, got {command!r}")
        return self.request(f"CHILD {handle} {command}", timeout=timeout)

    def child_snapshot(
        self, handle: int, command: str, timeout: float | None = None
    ) -> HazardFrame | PlayerSnapshot | EventFrame | str:
        """``CHILD`` + the reply decoder that matches the forwarded verb.

        ``HAZARDS``, ``PLAYER`` and ``EVENTS`` decode to their dataclasses; every
        other allowed verb returns the raw reply string.  Backward compatible by
        construction: a caller that needs the raw line can still use
        :meth:`child`.
        """

        return parse_child_reply(command, self.child(handle, command, timeout=timeout))

    def child_hazards(self, handle: int, limit: int = 0) -> HazardFrame:
        """``CHILD <handle> HAZARDS`` — the child's own current frame."""

        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        command = "HAZARDS" if limit == 0 else f"HAZARDS {limit}"
        return parse_hazards_reply(self.child(handle, command))

    def child_player(self, handle: int) -> PlayerSnapshot:
        """``CHILD <handle> PLAYER`` — the child's own self ship."""

        return parse_player_reply(self.child(handle, "PLAYER"))

    def child_events(self, handle: int) -> EventFrame:
        """``CHILD <handle> EVENTS`` — the child's own last-frame event flags."""

        return parse_events_reply(self.child(handle, "EVENTS"))


def _rows(payload: str, decoder) -> list:
    return [decoder(row) for row in payload.split(",") if row]


def parse_player_reply(reply: str) -> PlayerSnapshot:
    """Decode one ``PLAYER`` reply.

    Fields an older engine does not send stay ``None``; the position, hitbox and
    resource fields are mandatory because the command would have failed instead
    of replying ``OK`` without them.
    """

    fields = _parse(reply)
    if fields.get("_verb") != "OK":
        raise HarnessError(f"PLAYER reply is not OK: {reply!r}")
    missing = [key for key in ("hp", "lives", "px", "py") if key not in fields]
    if missing:
        raise HarnessError(f"PLAYER reply missing {missing}: {reply!r}")

    def optional(name: str) -> float | None:
        return _bits_float(fields[name]) if name in fields else None

    def pair(low: str, high: str) -> tuple[float, float] | None:
        # A missing bound stays None: substituting a number here would hand the
        # geometry search a hitbox the engine never reported.
        if low not in fields or high not in fields:
            return None
        return (_bits_float(fields[low]), _bits_float(fields[high]))

    return PlayerSnapshot(
        frame=int(fields.get("frames", 0)),
        hp=_bits_float(fields["hp"]),
        lives=int(fields["lives"]),
        bombs=_bits_float(fields["bombs"]) if "bombs" in fields else 0.0,
        power=_bits_float(fields["power"]) if "power" in fields else 0.0,
        focus=int(fields.get("focus", 0)),
        is_youkai=int(fields.get("isYoukai", 0)),
        player_state=int(fields.get("playerState", 0)),
        invuln_state=int(fields.get("invulnState", 0)),
        position=(_bits_float(fields["px"]), _bits_float(fields["py"])),
        hurtbox_min=pair("hurtboxMinX", "hurtboxMinY"),
        hurtbox_max=pair("hurtboxMaxX", "hurtboxMaxY"),
        speed_normal_axial=optional("speedNormalAxial"),
        speed_normal_diagonal=optional("speedNormalDiagonal"),
        speed_focused_axial=optional("speedFocusedAxial"),
        speed_focused_diagonal=optional("speedFocusedDiagonal"),
    )


def parse_hazards_reply(reply: str) -> HazardFrame:
    """Decode one ``HAZARDS`` reply into a :class:`HazardFrame`."""

    fields = _parse(reply)
    if fields.get("_verb") != "OK":
        raise HarnessError(f"HAZARDS reply is not OK: {reply!r}")
    for key in ("nb", "nl", "ne"):
        if key not in fields:
            raise HarnessError(f"HAZARDS reply missing {key}: {reply!r}")
    bullets = _rows(fields.get("bullets", ""), parse_bullet_row)
    lasers = _rows(fields.get("lasers", ""), parse_laser_row)
    enemies = _rows(fields.get("enemies", ""), parse_enemy_row)
    return HazardFrame(
        frame=int(fields.get("frames", 0)),
        bullet_count=int(fields["nb"]),
        laser_count=int(fields["nl"]),
        enemy_count=int(fields["ne"]),
        bullets=tuple(bullets),
        lasers=tuple(lasers),
        enemies=tuple(enemies),
        raw_bytes=len(reply),
    )


def parse_events_reply(reply: str) -> EventFrame:
    """Decode one ``EVENTS`` reply.

    ``frameEventFlags`` is mandatory: an older engine answered ``ERR`` (raised
    by the client before this function is reached), so a reply that says ``OK``
    but lacks the word is a protocol error, not "zero events".
    """

    fields = _parse(reply)
    if fields.get("_verb") != "OK":
        raise HarnessError(f"EVENTS reply is not OK: {reply!r}")
    if "frameEventFlags" not in fields:
        raise HarnessError(f"EVENTS reply missing frameEventFlags: {reply!r}")
    return EventFrame(
        frame=int(fields.get("frames", 0)),
        frame_event_flags=int(fields["frameEventFlags"], 16),
    )


#: Fields a ``HUD`` reply must carry (EXP-P2-005).  ``displayStale`` and
#: ``scoreDisplayStep`` are appended extras, so they stay optional like every
#: other append-only field in this module.
HUD_REQUIRED_FIELDS = (
    "displayScore",
    "globalsScore",
    "lives",
    "bombs",
    "power",
    "graze",
    "point",
    "time",
    "gameOverFlag",
    "displayAge",
)


def parse_hud_reply(reply: str) -> HudSnapshot:
    """Decode one ``HUD`` reply (EXP-P2-005).

    Every required field is mandatory: an engine without the command answers
    ``ERR`` (raised by the client before this function is reached) and an
    engine before a stage answers ``ERR hud-not-ready``, so a reply that says
    ``OK`` but lacks a field is a protocol error, not a zero.
    """

    fields = _parse(reply)
    if fields.get("_verb") != "OK":
        raise HarnessError(f"HUD reply is not OK: {reply!r}")
    missing = [name for name in HUD_REQUIRED_FIELDS if name not in fields]
    if missing:
        raise HarnessError(f"HUD reply missing {','.join(missing)}: {reply!r}")

    def optional(name: str) -> int | None:
        return int(fields[name]) if name in fields else None

    return HudSnapshot(
        frame=int(fields.get("frames", 0)),
        display_score=int(fields["displayScore"]),
        globals_score=int(fields["globalsScore"]),
        lives=int(fields["lives"]),
        bombs=int(fields["bombs"]),
        power=int(fields["power"]),
        graze=int(fields["graze"]),
        point=int(fields["point"]),
        time=int(fields["time"]),
        game_over=int(fields["gameOverFlag"]),
        display_age=int(fields["displayAge"]),
        display_stale=optional("displayStale"),
        score_display_step=optional("scoreDisplayStep"),
    )


#: Verbs whose ``CHILD`` reply has a typed decoder.
CHILD_SNAPSHOT_VERBS = ("HAZARDS", "PLAYER", "EVENTS", "HUD")


def parse_child_reply(command: str, reply: str):
    """Decode a ``CHILD`` reply according to the forwarded command's verb.

    Unknown verbs (``HASH``/``STATE``/``FRAME``/``STEP``/...) return the raw
    reply so callers that forward their own command keep working.  Known verbs
    decode with the same functions the parent-side commands use, so a child
    snapshot and a parent snapshot cannot drift apart in meaning.
    """

    verb = command.strip().split(" ", 1)[0].upper()
    if verb == "HAZARDS":
        return parse_hazards_reply(reply)
    if verb == "PLAYER":
        return parse_player_reply(reply)
    if verb == "EVENTS":
        return parse_events_reply(reply)
    if verb == "HUD":
        return parse_hud_reply(reply)
    return reply


def _parse(reply: str) -> dict[str, str]:
    """``OK a=1 b=2`` -> ``{"_verb": "OK", "a": "1", "b": "2"}``."""
    parts = reply.split()
    out: dict[str, str] = {"_verb": parts[0] if parts else ""}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out
