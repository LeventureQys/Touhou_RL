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
class BossInfo:
    """One live boss slot: position plus current HP.

    ``Enemy.life`` is an ``i32`` at ``OFF_ENEMY_LIFE`` (EnemyManager.hpp:278 and
    the ``C_ASSERT`` at :395), not a float.  ``slot`` is the index in
    ``EnemyManager::bosses[8]``; ``pointer`` is kept so a trace can tell two
    successive bosses apart even when they share a slot.
    """

    slot: int
    pointer: int
    x: float
    y: float
    life: int

    @property
    def hp(self) -> int:
        return int(self.life)


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
    player_state: int = -1  # Player.hpp: first i8; alive=0, spawning=1, dying=2, invulnerable=3
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
    # g_Gui derived signals (see spec.ADDR_GUI)
    gui_dialogue_present: bool = False
    gui_dialogue_msg_idx: int = -1
    gui_stage_clear_screen_state: int = 0
    gui_boss_present: bool = False
    # False when the g_Gui reads hit a torn-down object; the fields above then
    # keep their defaults and must not be read as "no dialogue / no clear".
    gui_read_ok: bool = True
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

    @property
    def game_cleared(self) -> bool:
        """``GameManagerFlags::gameCleared`` (bit 4), the full-game ending flag."""
        return bool(self.flags & spec.FLAG_GAME_CLEARED)

    @property
    def stage_clear_sequence_active(self) -> bool:
        """``GameManagerFlags::stageClearSequenceActive`` (bit 9).

        This is the only reliable "the stage is being cleared right now" signal:
        ``currentStageClearFlag`` (``OFF_GM_STAGE_CLEAR_FLAG``) is a persistent
        per-run bitfield and reads 1 on ordinary gameplay frames too.
        """
        return bool(self.flags & spec.FLAG_STAGE_CLEAR_SEQUENCE)

    @property
    def stage_transition_state(self) -> int:
        """Two-bit ``stageTransitionState`` (bits 5..6), 0 when not transitioning."""
        return (self.flags & spec.FLAG_STAGE_TRANSITION_MASK) >> spec.FLAG_STAGE_TRANSITION_SHIFT


@dataclass(slots=True)
class EnemyInfo:
    """One live normal enemy or boss slot (``EnemyManager::enemies[]``).

    ``slot`` is the index in the contiguous 481-entry array (``Enemy.enemyIndex``
    is the same value).  ``sub_id`` is the ECL subroutine id taken from the
    enemy's active ECL context; it is the only stable "what kind of enemy is
    this" identifier available without reading the stage ECL tables, and is
    ``None`` when the context pointer is null or the read raced a despawn.

    The dataclass is deliberately explicit rather than a raw struct dump: the
    v3.5.0 S0 contract requires every field to be traceable (known value or
    ``None``/``unknown``), never a silent zero.
    """

    slot: int
    x: float
    y: float
    life: int
    max_life: int
    active: bool
    boss: bool
    sub_id: int | None
    item_drop: int
    boss_slot: int
    flags1: int
    # v3.6.0 damage/phase contract (all read from C_ASSERT-backed offsets):
    # ``damageable``/``accepts_damage`` are the game's own invulnerability gates,
    # ``phase_starting_life`` marks phase boundaries, ``enemy_index`` is stable
    # per-enemy identity, ``shoot_interval_frames`` is the current volley cadence.
    damageable: bool = False
    accepts_damage: bool = False
    phase_starting_life: int = 0
    enemy_index: int = -1
    shoot_interval_frames: int = 0
    flags2: int = 0


@dataclass(slots=True)
class ItemInfo:
    """One in-use item (``ItemManager::items[]``).

    ``item_type``/``state`` are raw enum values from ``ItemManager.hpp`` so an
    unknown future value stays visible instead of being mapped to a guess.
    """

    slot: int
    x: float
    y: float
    item_type: int
    state: int
    onscreen: bool
    is_max_value: bool


@dataclass(slots=True)
class ShotParams:
    """Per-shot-type item rules read from ``Player::primaryShtFile``.

    ``point_item_value_line`` is the y threshold under which items enter
    auto-collect (the player-facing "25% line"); ``item_collection_box_size``
    is the pickup box half-size used by ``Player::CalcItemBoxCollision``.
    """

    item_auto_collect_speed: float
    item_collection_box_size: float
    point_item_value_line: float


@dataclass(slots=True)
class BulletField:
    """Zero-copy views over one snapshot of ``g_BulletManager.bullets``."""

    raw: np.ndarray  # (BULLET_COUNT, BULLET_SIZE) uint8
    position: np.ndarray  # (N, 2) float32
    velocity: np.ndarray  # (N, 2) float32
    state: np.ndarray  # (N,) uint16
    disabled: np.ndarray  # (N,) int8
    half_size: np.ndarray  # (N, 2) float32

    # Position/velocity magnitude below which an entry is considered zeroed.
    _ZERO_EPS = 1e-3

    def active_mask(self) -> np.ndarray:
        """Bullets that are really in flight.

        ``state`` alone is not enough: the pool permanently holds one zeroed entry
        whose state is neither ``BULLET_UNUSED`` nor ``BULLET_DESPAWNING``.  Its
        position is exactly (0, 0) and its velocity exactly (0, 0), which cannot be
        a real projectile because the playfield origin is (32, 16).  Leaving it in
        made every snapshot claim one extra bullet, which also let the "first
        snapshot is complete" check pass vacuously (the phantom matched itself).
        """
        state = self.state
        live = (state != spec.BULLET_UNUSED) & (state != spec.BULLET_DESPAWNING) & (self.disabled == 0)
        zeroed = (
            (np.abs(self.position[:, 0]) < self._ZERO_EPS)
            & (np.abs(self.position[:, 1]) < self._ZERO_EPS)
            & (np.abs(self.velocity[:, 0]) < self._ZERO_EPS)
            & (np.abs(self.velocity[:, 1]) < self._ZERO_EPS)
        )
        return live & ~zeroed


def _strided(buffer: np.ndarray, offset: int, shape: tuple[int, ...], dtype, stride: int):
    return np.ndarray(
        shape=shape,
        dtype=dtype,
        buffer=buffer,
        offset=offset,
        strides=(stride,) + tuple(np.dtype(dtype).itemsize * k for k in range(1, len(shape))),
    )


def _read_current_spell(gm: bytes | bytearray) -> int:
    """Signed 16-bit ``GameManager::currentSpellCardNumber`` (0x3DBB0).

    The field is an i16 (``C_ASSERT(offsetof(GameManager, currentSpellCardNumber)
    == 0x3DBB0)``), but it used to be unpacked as a 4-byte int.  That pulled in
    ``isInGameMenu`` (0x3DBB2) and ``showRetryMenu`` (0x3DBB3), so a game-over
    retry menu read back as 0x01000000 and ``trace35.events_between`` recorded it
    as a ``spell`` event.  That is exactly why v3.5.0 S0's three "spell" events
    landed on the three game-over episodes.
    """
    return struct.unpack_from("<h", gm, spec.OFF_GM_CURRENT_SPELL - _GM_BLOCK_START)[0]


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
        self._im = spec.ADDR_ITEM_MANAGER + delta
        self._title = spec.ADDR_TITLE_SCREEN + delta
        self._input = spec.ADDR_LAST_FRAME_INPUT + delta
        self._gui_input = spec.ADDR_GUI_INPUT_CURRENT + delta
        self._gui_base = spec.ADDR_GUI + delta
        self._replay_manager = spec.ADDR_REPLAY_MANAGER + delta

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
        # v3.5.0 S0: one bulk read for the whole enemy array and one for the item
        # array.  Both buffers are allocated once so a per-decision scan does not
        # allocate 10 MB every call; NumPy strided views below never copy them.
        self._enemy_buf = ctypes.create_string_buffer(spec.ENEMY_SIZE * spec.ENEMY_SCAN_COUNT)
        self._item_buf = ctypes.create_string_buffer(spec.ITEM_SIZE * spec.ITEM_COUNT)
        self._sht_ptr_buf = ctypes.create_string_buffer(4)
        self._sht_buf = ctypes.create_string_buffer(spec.SHT_READ_SIZE)
        self._replay_ptr_buf = ctypes.create_string_buffer(4)
        self._replay_events_buf = ctypes.create_string_buffer(2)
        self._boss_ptr_buf = ctypes.create_string_buffer(4 * spec.BOSS_COUNT)
        self._boss_pos_buf = ctypes.create_string_buffer(0xC)
        self._boss_life_buf = ctypes.create_string_buffer(4)
        self._gui_impl_buf = ctypes.create_string_buffer(4)
        self._gui_msg_buf = ctypes.create_string_buffer(4)
        self._gui_clear_buf = ctypes.create_string_buffer(4)
        self._gui_boss_buf = ctypes.create_string_buffer(1)

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
        state.current_spell = _read_current_spell(gm)
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
        state.player_state = struct.unpack_from('<b', p, 0)[0]
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

        # g_Gui: dialogue presence and the stage-clear screen state.  These are
        # read-only decodes of the same global the game's IsDialoguePresent()
        # uses; a missing/zero impl pointer simply means the GUI is not up yet.
        state.gui_read_ok = True
        try:
            mem.read_into(self._gui_base + spec.OFF_GUI_IMPL, self._gui_impl_buf)
            gui_impl = struct.unpack_from("<I", self._gui_impl_buf.raw, 0)[0]
            mem.read_into(self._gui_base + spec.OFF_GUI_BOSS_PRESENT, self._gui_boss_buf)
            state.gui_boss_present = self._gui_boss_buf.raw[0] != 0
            if gui_impl:
                mem.read_into(
                    gui_impl + spec.OFF_GUI_IMPL_MESSAGE_CURRENT_MSG_IDX, self._gui_msg_buf
                )
                state.gui_dialogue_msg_idx = struct.unpack_from("<i", self._gui_msg_buf.raw, 0)[0]
                state.gui_dialogue_present = (
                    state.gui_dialogue_msg_idx >= 0 or state.gui_dialogue_msg_idx == -2
                )
                mem.read_into(
                    gui_impl + spec.OFF_GUI_IMPL_STAGE_CLEAR_SCREEN_STATE, self._gui_clear_buf
                )
                state.gui_stage_clear_screen_state = struct.unpack_from(
                    "<i", self._gui_clear_buf.raw, 0
                )[0]
        except OSError:
            # The GUI object is torn down between screens; record that the group
            # is unknown instead of leaving the defaults to look like real data.
            state.gui_read_ok = False

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
    def read_boss_info(self) -> list[BossInfo]:
        """Live bosses with position and HP, ordered by boss slot.

        Costs one 32-byte pointer read plus 16 bytes per live boss (12 position
        + 4 life).  The common case (no boss) is still a single read.  Dead or
        half-freed slots are skipped instead of raising: a boss can disappear
        between the pointer read and the field read, which is exactly the
        transition a stage-clear probe needs to observe.
        """
        mem = self.mem
        mem.read_into(self._em + spec.OFF_EM_BOSSES, self._boss_ptr_buf)
        pointers = struct.unpack_from(f"<{spec.BOSS_COUNT}I", self._boss_ptr_buf.raw, 0)
        found: list[BossInfo] = []
        for slot, pointer in enumerate(pointers):
            if not pointer:
                continue
            try:
                mem.read_into(pointer + spec.OFF_ENEMY_POSITION, self._boss_pos_buf)
                x, y, _z = struct.unpack_from("<fff", self._boss_pos_buf.raw, 0)
                if not (0.0 < x < 640.0 and 0.0 < y < 480.0):
                    continue
                mem.read_into(pointer + spec.OFF_ENEMY_LIFE, self._boss_life_buf)
                life = struct.unpack_from("<i", self._boss_life_buf.raw, 0)[0]
            except OSError:
                continue
            found.append(BossInfo(slot=slot, pointer=pointer, x=x, y=y, life=life))
        return found

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

    # -- normal enemies / items / shot params -----------------------------
    @staticmethod
    def _strided_view(buffer, offset: int, dtype, count: int, size: int) -> np.ndarray:
        """Read-only view of ``count`` records of ``size`` bytes at ``offset``."""
        return np.ndarray(
            shape=(count,),
            dtype=dtype,
            buffer=buffer,
            offset=offset,
            strides=(size,),
        )

    def read_enemies(self, active_only: bool = True, with_sub_id: bool = True) -> list[EnemyInfo]:
        """Live enemies from the contiguous ``EnemyManager::enemies`` array.

        Cost: one ~9.8 MB ``ReadProcessMemory`` plus one 2-byte read per live
        enemy when ``with_sub_id`` is true (typically < 50 reads).  The bulk read
        is what makes an all-enemy scan cheaper than 481 individual reads; the
        array is fixed-size and contiguous, so a single read is also a coherent
        enough snapshot for tracing.

        Sub-id reads can race a despawn; a failed dereference is recorded as
        ``None`` (unknown type) instead of aborting the whole scan.
        """
        mem = self.mem
        count = spec.ENEMY_SCAN_COUNT
        size = spec.ENEMY_SIZE
        mem.read_into(self._em + spec.OFF_EM_ENEMIES, self._enemy_buf)
        buf = self._enemy_buf
        flags = self._strided_view(buf, spec.OFF_ENEMY_FLAGS1, np.uint32, count, size)
        xs = self._strided_view(buf, spec.OFF_ENEMY_POSITION, np.float32, count, size)
        ys = self._strided_view(buf, spec.OFF_ENEMY_POSITION + 4, np.float32, count, size)
        lifes = self._strided_view(buf, spec.OFF_ENEMY_LIFE, np.int32, count, size)
        max_lifes = self._strided_view(buf, spec.OFF_ENEMY_MAX_LIFE, np.int32, count, size)
        item_drops = self._strided_view(buf, spec.OFF_ENEMY_ITEM_DROP, np.int8, count, size)
        boss_slots = self._strided_view(buf, spec.OFF_ENEMY_BOSS_SLOT, np.uint8, count, size)
        ecl_ptrs = self._strided_view(buf, spec.OFF_ENEMY_ACTIVE_ECL_CONTEXT, np.uint32, count, size)
        phase_lifes = self._strided_view(buf, spec.OFF_ENEMY_PHASE_STARTING_LIFE, np.int32, count, size)
        enemy_indices = self._strided_view(buf, spec.OFF_ENEMY_INDEX, np.int32, count, size)
        shoot_intervals = self._strided_view(buf, spec.OFF_ENEMY_SHOOT_INTERVAL, np.int32, count, size)
        flags2s = self._strided_view(buf, spec.OFF_ENEMY_FLAGS2, np.uint32, count, size)

        indices = np.flatnonzero(flags & spec.ENEMY_FLAG_ACTIVE) if active_only else np.arange(count)
        sub_id_cache: dict[int, int | None] = {}
        result: list[EnemyInfo] = []
        for index in indices:
            slot = int(index)
            sub_id: int | None = None
            if with_sub_id:
                pointer = int(ecl_ptrs[slot])
                if pointer:
                    if pointer in sub_id_cache:
                        sub_id = sub_id_cache[pointer]
                    else:
                        try:
                            raw = mem.read(pointer + spec.OFF_ECL_CONTEXT_SUB_ID, 2)
                            sub_id = int(struct.unpack("<h", raw)[0])
                        except OSError:
                            sub_id = None
                        sub_id_cache[pointer] = sub_id
            flags1 = int(flags[slot])
            result.append(
                EnemyInfo(
                    slot=slot,
                    x=float(xs[slot]),
                    y=float(ys[slot]),
                    life=int(lifes[slot]),
                    max_life=int(max_lifes[slot]),
                    active=bool(flags1 & spec.ENEMY_FLAG_ACTIVE),
                    boss=bool(flags1 & spec.ENEMY_FLAG_BOSS),
                    sub_id=sub_id,
                    item_drop=int(item_drops[slot]),
                    boss_slot=int(boss_slots[slot]),
                    flags1=flags1,
                    damageable=bool(flags1 & spec.ENEMY_FLAG_DAMAGEABLE),
                    accepts_damage=bool(flags1 & spec.ENEMY_FLAG_ACCEPTS_DAMAGE),
                    phase_starting_life=int(phase_lifes[slot]),
                    enemy_index=int(enemy_indices[slot]),
                    shoot_interval_frames=int(shoot_intervals[slot]),
                    flags2=int(flags2s[slot]),
                )
            )
        return result

    def read_items(self, active_only: bool = True) -> list[ItemInfo]:
        """In-use items from ``ItemManager::items`` (one ~1.5 MB bulk read)."""
        mem = self.mem
        count = spec.ITEM_COUNT
        size = spec.ITEM_SIZE
        mem.read_into(self._im + spec.OFF_IM_ITEMS, self._item_buf)
        buf = self._item_buf
        in_use = self._strided_view(buf, spec.OFF_ITEM_IN_USE, np.uint8, count, size)
        xs = self._strided_view(buf, spec.OFF_ITEM_CURRENT_POSITION, np.float32, count, size)
        ys = self._strided_view(buf, spec.OFF_ITEM_CURRENT_POSITION + 4, np.float32, count, size)
        types = self._strided_view(buf, spec.OFF_ITEM_TYPE, np.int8, count, size)
        onscreen = self._strided_view(buf, spec.OFF_ITEM_ONSCREEN, np.uint8, count, size)
        states = self._strided_view(buf, spec.OFF_ITEM_STATE, np.int8, count, size)
        max_values = self._strided_view(buf, spec.OFF_ITEM_MAX_VALUE, np.int8, count, size)
        indices = np.flatnonzero(in_use) if active_only else np.arange(count)
        return [
            ItemInfo(
                slot=int(index),
                x=float(xs[index]),
                y=float(ys[index]),
                item_type=int(types[index]),
                state=int(states[index]),
                onscreen=bool(onscreen[index]),
                is_max_value=bool(max_values[index]),
            )
            for index in indices
        ]

    def read_shot_params(self) -> ShotParams | None:
        """Read the active shot file's item rules, or ``None`` when not loaded.

        The player block read by :meth:`read_state` is only the first 0x410
        bytes; ``primaryShtFile`` lives at +0xE2A74, so this is a separate
        4-byte pointer read plus one 0x38-byte read.  It is not on the hot path
        (once per decision at most) and lets the trace record the *actual*
        auto-collect line instead of hard-coding 128.
        """
        mem = self.mem
        try:
            mem.read_into(self._player + spec.OFF_PLAYER_PRIMARY_SHT_FILE, self._sht_ptr_buf)
            pointer = struct.unpack_from("<I", self._sht_ptr_buf.raw, 0)[0]
            if not pointer:
                return None
            mem.read_into(pointer, self._sht_buf)
        except OSError:
            return None
        auto_speed, box, line = struct.unpack_from("<fff", self._sht_buf.raw, 0x14)
        if not all(np.isfinite([auto_speed, box, line])):
            return None
        return ShotParams(
            item_auto_collect_speed=float(auto_speed),
            item_collection_box_size=float(box),
            point_item_value_line=float(line),
        )

    def read_frame_event_flags(self) -> int | None:
        """Per-frame ``ReplayManager.frameEventFlags``, or ``None`` if unreadable.

        Bit ``FRAME_EVENT_ITEM_COLLECTED`` is set by ``Item::CollectPowerSmall``
        and friends during the simulated frame.  It is the only direct signal
        that a power increase came from an item pick-up rather than a spawn that
        was collected inside one decision's blind window.
        """
        mem = self.mem
        try:
            mem.read_into(self._replay_manager, self._replay_ptr_buf)
            pointer = struct.unpack_from("<I", self._replay_ptr_buf.raw, 0)[0]
            if not pointer:
                return None
            mem.read_into(pointer + spec.OFF_RM_FRAME_EVENT_FLAGS, self._replay_events_buf)
            return int(struct.unpack_from("<H", self._replay_events_buf.raw, 0)[0])
        except OSError:
            return None

    # -- input ------------------------------------------------------------
    def read_input_mask(self) -> int:
        self.mem.read_into(self._input, self._input_buf)
        return struct.unpack_from("<H", self._input_buf.raw, 2)[0]
