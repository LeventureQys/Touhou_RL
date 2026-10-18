"""Decoding of TH08 game state from process memory.

``GameStateReader`` performs a fixed handful of ``ReadProcessMemory`` calls per
read (measured 0.12 ms for the scalar block).  ``read_bullets`` grabs the whole
bullet pool with a single call and exposes it as zero-copy NumPy strided views;
the pool is ``0x601`` bullets of ``0x10B8`` bytes (~6.3 MiB), which is the only
expensive read in the loop and is therefore done once per *decision*, not once
per rendered frame.

All fields are documented in ``框架接口契约.md`` §2; every offset comes from
``spec.py`` (each backed by a ``C_ASSERT(offsetof(...))`` in the sources).
"""

from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass, field

import numpy as np

from . import spec
from .memory import ProcessMemory

# Block reads: one RPM per contiguous region instead of one per field.
_GM_BLOCK_START = 0x3DBA8
_GM_BLOCK_SIZE = 0x3DE2C - _GM_BLOCK_START
_INPUT_BLOCK_SIZE = 8  # g_LastFrameInput(u16) + g_CurFrameInput(u16) + 4 spare
_PLAYER_BLOCK_SIZE = 0x410


@dataclass(slots=True)
class GameState:
    """Scalar state of one sampled frame (contract §2.1)."""

    # supervisor / mode
    supervisor_state: int = 0
    title_screen_current: int = -1
    title_screen_state: int = -1
    title_state: int = -1
    title_cursor: int = -1
    title_cursor2: int = -1
    title_practice_state: int = -1
    title_replay_count: int = 0
    title_selected_replay: int = -1
    title_selected_replay_stage: int = -1
    title_timer2: int = -1
    # run metadata
    stage: int = 0
    difficulty_hint: int = -1
    character: int = 0
    shot_type: int = 0
    flags: int = 0
    is_demo: bool = False
    is_replay: bool = False
    is_practice: bool = False
    is_spell_practice: bool = False
    replay_mode: int = 0
    in_game_menu: bool = False
    current_spell: int = -1
    # clocks
    gameplay_frame: int = 0
    stage_frames: int = 0
    run_frames: int = 0
    stage_clear_flag: int = 0
    # resources
    score: int = 0
    graze: int = 0
    clock_time: int = 0
    time_orbs: int = 0
    deaths: int = 0
    lives: float = 0.0
    bombs: float = 0.0
    bombs_used: int = 0
    power: float = 0.0
    # player
    player_x: float = 0.0
    player_y: float = 0.0
    player_vx: float = 0.0
    player_vy: float = 0.0
    focus_mode: int = 0
    hurtbox_half_x: float = 0.0
    hurtbox_half_y: float = 0.0
    # field geometry (from memory, cross-checked against spec constants)
    field_x: float = spec.PLAYFIELD_X
    field_y: float = spec.PLAYFIELD_Y
    field_w: float = spec.PLAYFIELD_W
    field_h: float = spec.PLAYFIELD_H
    # input
    input_mask: int = 0
    last_input_mask: int = 0
    gui_input_current: int = 0
    gui_input_previous: int = 0
    # bullets
    active_bullets: int = 0
    bullet_capacity: int = spec.BULLET_COUNT

    def is_playing(self) -> bool:
        """True when a human-controlled stage is running (not demo/replay/menu)."""
        return (
            self.supervisor_state == spec.STATE_GAME_MANAGER
            and not self.is_demo
            and not self.is_replay
            and self.gameplay_frame > 0
        )

    def is_run_over(self) -> bool:
        """True when the run can no longer continue (game over / back to title)."""
        return self.supervisor_state != spec.STATE_GAME_MANAGER or self.lives < 0


@dataclass(slots=True)
class BulletField:
    """Zero-copy views over one snapshot of ``g_BulletManager.bullets``."""

    raw: np.ndarray  # (BULLET_COUNT, BULLET_SIZE) uint8
    position: np.ndarray  # (N, 2) float32
    velocity: np.ndarray  # (N, 2) float32
    state: np.ndarray  # (N,) uint16
    disabled: np.ndarray  # (N,) int8
    half_size: np.ndarray  # (N, 2) float32

    def active_mask(self) -> np.ndarray:
        state = self.state
        return (state != spec.BULLET_UNUSED) & (state != spec.BULLET_DESPAWNING) & (self.disabled == 0)


def _strided(buffer: np.ndarray, offset: int, shape: tuple[int, ...], dtype, stride: int):
    return np.ndarray(
        shape=shape,
        dtype=dtype,
        buffer=buffer,
        offset=offset,
        strides=(stride,) + tuple(np.dtype(dtype).itemsize * k for k in range(1, len(shape))),
    )


class GameStateReader:
    """Reads scalars and the bullet pool from one TH08 process."""

    def __init__(self, mem: ProcessMemory):
        self.mem = mem
        self.base = mem.module_base
        if not self.base:
            raise RuntimeError("module base for th08.exe was not resolved")
        delta = self.base - spec.IMAGE_BASE
        self._sup = spec.ADDR_SUPERVISOR + delta
        self._gm = spec.ADDR_GAME_MANAGER + delta
        self._player = spec.ADDR_PLAYER + delta
        self._bm = spec.ADDR_BULLET_MANAGER + delta
        self._em = spec.ADDR_ENEMY_MANAGER + delta
        self._title = spec.ADDR_TITLE_SCREEN + delta
        self._input = spec.ADDR_LAST_FRAME_INPUT + delta
        self._gui_input = spec.ADDR_GUI_INPUT_CURRENT + delta

        self._gm_buf = ctypes.create_string_buffer(_GM_BLOCK_SIZE)
        self._input_buf = ctypes.create_string_buffer(_INPUT_BLOCK_SIZE)
        self._gui_buf = ctypes.create_string_buffer(4)
        self._player_buf = ctypes.create_string_buffer(_PLAYER_BLOCK_SIZE)
        self._sup_buf = ctypes.create_string_buffer(8)
        self._globals_buf = ctypes.create_string_buffer(spec.GLOBALS_READ_SIZE)
        self._ptr_buf = ctypes.create_string_buffer(4)
        self._title_head_buf = ctypes.create_string_buffer(0x70)
        self._title_replay_buf = ctypes.create_string_buffer(0x18)
        self._title_tail_buf = ctypes.create_string_buffer(0x18)
        self._counter_buf = ctypes.create_string_buffer(4)
        self._bullet_buf = ctypes.create_string_buffer(spec.BULLET_BLOCK_SIZE)
        self._boss_ptr_buf = ctypes.create_string_buffer(4 * spec.BOSS_COUNT)
        self._boss_pos_buf = ctypes.create_string_buffer(0xC)

    # -- scalars ----------------------------------------------------------
    def read_state(self, with_bullets: bool = True, with_title: bool = True) -> GameState:
        mem = self.mem
        state = GameState()
        state.bullet_capacity = spec.BULLET_COUNT

        mem.read_into(self._gm + _GM_BLOCK_START, self._gm_buf)
        gm = self._gm_buf.raw
        state.character = gm[spec.OFF_GM_CHARACTER - _GM_BLOCK_START]
        state.shot_type = gm[spec.OFF_GM_SHOT_TYPE - _GM_BLOCK_START]

        def gm_u8(offset: int) -> int:
            return gm[offset - _GM_BLOCK_START]

        def gm_u32(offset: int) -> int:
            return struct.unpack_from("<I", gm, offset - _GM_BLOCK_START)[0]

        state.flags = gm_u32(spec.OFF_GM_FLAGS)
        state.is_practice = bool(state.flags & spec.FLAG_PRACTICE)
        state.is_demo = bool(state.flags & spec.FLAG_DEMO)
        state.is_replay = bool(state.flags & spec.FLAG_REPLAY)
        state.is_spell_practice = bool(state.flags & spec.FLAG_SPELL_PRACTICE)
        state.in_game_menu = gm_u8(spec.OFF_GM_IS_IN_GAME_MENU) != 0
        state.replay_mode = gm_u8(spec.OFF_GM_REPLAY_MODE)
        state.current_spell = struct.unpack_from("<i", gm, spec.OFF_GM_CURRENT_SPELL - _GM_BLOCK_START)[0]
        state.gameplay_frame = gm_u32(spec.OFF_GM_GAMEPLAY_FRAME)
        state.stage = struct.unpack_from("<i", gm, spec.OFF_GM_CURRENT_STAGE - _GM_BLOCK_START)[0]
        state.stage_clear_flag = gm_u32(spec.OFF_GM_STAGE_CLEAR_FLAG)
        state.stage_frames = gm_u32(spec.OFF_GM_STAGE_ACTIVE_FRAMES)
        state.run_frames = gm_u32(spec.OFF_GM_RUN_ACTIVE_FRAMES)
        # Geometry: the game stores it per stage; the declared constants are the
        # fallback when no stage is loaded (all zeros).
        fx, fy, _fz = struct.unpack_from("<fff", gm, spec.OFF_GM_PLAYFIELD_ORIGIN - _GM_BLOCK_START)
        fw, fh, _fz2 = struct.unpack_from("<fff", gm, spec.OFF_GM_PLAYFIELD_SIZE - _GM_BLOCK_START)
        if fw > 1.0 and fh > 1.0:
            state.field_x, state.field_y, state.field_w, state.field_h = fx, fy, fw, fh

        mem.read_into(self._sup + spec.OFF_SUP_WANTED_STATE, self._sup_buf)
        wanted, cur = struct.unpack_from("<ii", self._sup_buf.raw, 0)
        state.supervisor_state = cur

        # globals is a pointer stored in GameManager; already absolute.
        mem.read_into(self._gm + spec.OFF_GM_GLOBALS, self._ptr_buf)
        globals_ptr = struct.unpack_from("<I", self._ptr_buf.raw, 0)[0]
        if globals_ptr:
            mem.read_into(globals_ptr, self._globals_buf)
            g = self._globals_buf.raw
            state.score = struct.unpack_from("<I", g, spec.OFF_GLOBALS_SCORE)[0]
            state.graze = struct.unpack_from("<i", g, spec.OFF_GLOBALS_GRAZE)[0]
            state.clock_time = struct.unpack_from("<b", g, spec.OFF_GLOBALS_CLOCK_TIME)[0]
            state.time_orbs = struct.unpack_from("<i", g, spec.OFF_GLOBALS_TIME_ORBS)[0]
            # ZunGlobals stores deaths/bombsUsed as f32 (Global.hpp:389-395); the
            # v1.0.1 offset table listed the offsets without the types, so this
            # must be decoded as float and rounded, never reinterpreted as i32.
            state.deaths = int(round(struct.unpack_from("<f", g, spec.OFF_GLOBALS_DEATHS)[0]))
            state.lives = struct.unpack_from("<f", g, spec.OFF_GLOBALS_LIVES)[0]
            state.bombs = struct.unpack_from("<f", g, spec.OFF_GLOBALS_BOMBS)[0]
            state.bombs_used = int(round(struct.unpack_from("<f", g, spec.OFF_GLOBALS_BOMBS_USED)[0]))
            state.power = struct.unpack_from("<f", g, spec.OFF_GLOBALS_POWER)[0]

        mem.read_into(self._player, self._player_buf)
        p = self._player_buf.raw
        state.focus_mode = p[spec.OFF_PLAYER_FOCUS_MODE]
        state.player_x, state.player_y, _pz = struct.unpack_from("<fff", p, spec.OFF_PLAYER_POSITION)
        state.player_vx, state.player_vy, _pvz = struct.unpack_from("<fff", p, spec.OFF_PLAYER_VELOCITY)
        state.hurtbox_half_x, state.hurtbox_half_y, _hz = struct.unpack_from(
            "<fff", p, spec.OFF_PLAYER_HURTBOX_HALF
        )

        mem.read_into(self._input, self._input_buf)
        state.last_input_mask, state.input_mask = struct.unpack_from("<HH", self._input_buf.raw, 0)
        mem.read_into(self._gui_input, self._gui_buf)
        state.gui_input_current, state.gui_input_previous = struct.unpack_from("<HH", self._gui_buf.raw, 0)

        if with_title:
            # The TitleScreen object is created and freed on every screen change;
            # a read that races the transition is expected, not fatal.
            try:
                title_ptr = mem.u32(self._title)
                if title_ptr:
                    mem.read_into(title_ptr, self._title_head_buf)
                    h = self._title_head_buf.raw
                    state.title_cursor = struct.unpack_from("<i", h, spec.OFF_TITLE_CURSOR)[0]
                    state.title_cursor2 = struct.unpack_from("<i", h, spec.OFF_TITLE_CURSOR2)[0]
                    state.title_screen_state = struct.unpack_from("<i", h, spec.OFF_TITLE_SCREEN_STATE)[0]
                    state.title_practice_state = struct.unpack_from("<i", h, spec.OFF_TITLE_PRACTICE_STATE)[0]
                    mem.read_into(title_ptr + spec.OFF_TITLE_REPLAY_COUNT - 4, self._title_replay_buf)
                    r = self._title_replay_buf.raw
                    state.title_replay_count = struct.unpack_from("<i", r, 4)[0]
                    state.title_selected_replay = struct.unpack_from("<i", r, 8)[0]
                    state.title_selected_replay_stage = struct.unpack_from("<i", r, 0xC)[0]
                    mem.read_into(title_ptr + spec.OFF_TITLE_CURRENT_SCREEN, self._title_tail_buf)
                    t = self._title_tail_buf.raw
                    state.title_screen_current = struct.unpack_from("<i", t, 0)[0]
                    state.title_timer2 = struct.unpack_from("<i", t, 4)[0]
                    state.title_state = struct.unpack_from(
                        "<i", t, spec.OFF_TITLE_STATE - spec.OFF_TITLE_CURRENT_SCREEN
                    )[0]
            except OSError:
                state.title_cursor = -1
                state.title_screen_current = -1
                state.title_screen_state = -1
                state.title_replay_count = 0

        if with_bullets:
            mem.read_into(self._bm + spec.OFF_BM_ACTIVE_BULLETS, self._counter_buf)
            state.active_bullets = struct.unpack_from("<i", self._counter_buf.raw, 0)[0]
        return state

    # -- bullet pool ------------------------------------------------------
    def read_bullets(self) -> BulletField:
        mem = self.mem
        mem.read_into(self._bm + spec.OFF_BM_BULLETS, self._bullet_buf)
        raw = np.frombuffer(self._bullet_buf, dtype=np.uint8).reshape(
            spec.BULLET_COUNT, spec.BULLET_SIZE
        )
        stride = spec.BULLET_SIZE
        return BulletField(
            raw=raw,
            position=_strided(raw, spec.OFF_BULLET_POSITION, (spec.BULLET_COUNT, 2), np.float32, stride),
            velocity=_strided(raw, spec.OFF_BULLET_VELOCITY, (spec.BULLET_COUNT, 2), np.float32, stride),
            state=_strided(raw, spec.OFF_BULLET_STATE, (spec.BULLET_COUNT,), np.uint16, stride),
            disabled=np.ndarray(
                shape=(spec.BULLET_COUNT,),
                dtype=np.int8,
                buffer=raw,
                offset=spec.OFF_BULLET_COLLISION_DISABLED,
                strides=(stride,),
            ),
            half_size=_strided(raw, spec.OFF_BULLET_HURTBOX_X, (spec.BULLET_COUNT, 2), np.float32, stride),
        )

    def read_lasers(self) -> np.ndarray:
        """Return raw laser pool bytes (``LASER_COUNT`` entries of ``LASER_SIZE``)."""
        return np.frombuffer(self.mem.read(self._bm + spec.OFF_BM_LASERS, spec.LASER_SIZE * spec.LASER_COUNT),
                             dtype=np.uint8).reshape(spec.LASER_COUNT, spec.LASER_SIZE)

    # -- bosses -----------------------------------------------------------
    def read_boss_positions(self) -> np.ndarray:
        """Positions of the live bosses, shape ``(n, 2)`` float32, ``n <= 8``.

        Cost: one 32-byte read for the pointer array plus 12 bytes per live boss
        (measured ~0.1 ms for a single boss).  ``bosses[]`` holds null pointers
        whenever no boss is present, which is most of a stage, so the common
        case is a single read.

        Why this exists: a policy that only dodges never deals damage - it can
        survive a stage forever without ever finishing it.  Aiming needs to know
        where the boss is, and ``g_EnemyManager`` is the only cheap source of
        that position (the ``enemies`` array is 10 MB of contiguous structs).
        """
        mem = self.mem
        mem.read_into(self._em + spec.OFF_EM_BOSSES, self._boss_ptr_buf)
        pointers = struct.unpack_from(f"<{spec.BOSS_COUNT}I", self._boss_ptr_buf.raw, 0)
        found: list[tuple[float, float]] = []
        for pointer in pointers:
            if not pointer:
                continue
            try:
                mem.read_into(pointer + spec.OFF_ENEMY_POSITION, self._boss_pos_buf)
            except OSError:
                continue
            x, y, _z = struct.unpack_from("<fff", self._boss_pos_buf.raw, 0)
            if not (0.0 < x < 640.0 and 0.0 < y < 480.0):
                continue
            found.append((x, y))
        return np.asarray(found, dtype=np.float32).reshape(-1, 2)

    # -- input ------------------------------------------------------------
    def read_input_mask(self) -> int:
        self.mem.read_into(self._input, self._input_buf)
        return struct.unpack_from("<H", self._input_buf.raw, 2)[0]
