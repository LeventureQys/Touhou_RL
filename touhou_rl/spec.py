"""Static specification for TH08 1.00d (rebuilt bugfix image).

Every address/offset here is backed by a ``C_ASSERT(offsetof(...))`` in the
reconstruction sources (``third_party/th08/src``) or by a measured probe value
recorded in ``Document/Update/Alpha/v1.0.1/问题清单.md``.  The runtime binary is
the ``bugfix`` build produced by ``tools/th08/Build.ps1``: preferred image base
``0x00400000`` with no ``DYNAMIC_BASE``, so linker-map VAs equal runtime VAs.

The game is never written to: TH08's anti-tamper checksum aborts the process on
writes to lives/bombs/power (``GameManager.hpp`` ``IsTampered``).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Linker-map addresses (module base 0x00400000)
# ---------------------------------------------------------------------------
IMAGE_BASE = 0x00400000

ADDR_SUPERVISOR = 0x017DC380  # g_Supervisor
ADDR_GAME_MANAGER = 0x0161C7A0  # g_GameManager
ADDR_PLAYER = 0x017E3120  # g_Player
ADDR_BULLET_MANAGER = 0x00F620B8  # g_BulletManager
# g_EnemyManager (linker map: 00585148; the struct is ~10 MB, which is why the
# next global in the map, g_EnemyManagerCalcChain, starts at 0x00F62058).
ADDR_ENEMY_MANAGER = 0x00585148
ADDR_TITLE_SCREEN = 0x018CB030  # g_TitleScreen (pointer)
ADDR_CUR_FRAME_INPUT = 0x0165A808  # g_CurFrameInput (u16)
ADDR_LAST_FRAME_INPUT = 0x0165A804  # g_LastFrameInput (u16)
# g_GuiMessageInputCurrent / Previous (Gui.cpp:29-30).  Player::OnUpdate reads
# *this*, not g_CurFrameInput, so it is the input the game actually consumes --
# including replay-fed input during playback (ReplayManager.cpp:329).
ADDR_GUI_INPUT_CURRENT = 0x0161C674
ADDR_GUI_INPUT_PREVIOUS = 0x0161C678

# ---------------------------------------------------------------------------
# Structure offsets
# ---------------------------------------------------------------------------
# Supervisor (Supervisor.hpp:270-340)
OFF_SUP_CUR_STATE = 0x15C
OFF_SUP_WANTED_STATE = 0x158

# TitleScreen (TitleScreen.hpp:169-215 plus the computed tail offsets).
# ``cursor`` is the highlighted menu item; ``currentScreenState`` doubles as the
# replay sub-state while ``currentScreen == SCREEN_REPLAY``.
OFF_TITLE_CURSOR = 0x0
OFF_TITLE_CURSOR2 = 0x8
OFF_TITLE_SCREEN_STATE = 0xC
OFF_TITLE_STATE_TIMER = 0x10
OFF_TITLE_PREVIOUS_SCREEN = 0x68
OFF_TITLE_PRACTICE_STATE = 0x6C
OFF_TITLE_REPLAY_COUNT = 0xC288
OFF_TITLE_SELECTED_REPLAY = 0xC28C
OFF_TITLE_SELECTED_REPLAY_STAGE = 0xC290
# The tail offsets are measured (tools/probes/diag_screen_offset.py): pressing Z
# on the start menu flips 0x14428 from 0 (StartMenu) to 4 (DifficultySelect)
# while the start-menu idle counter at 0x14430 resets.  A pure struct-layout
# computation lands 0x20 bytes later, so the measured value is authoritative.
OFF_TITLE_CURRENT_SCREEN = 0x14428
OFF_TITLE_STATE_TIMER2 = 0x1442C
OFF_TITLE_START_MENU_IDLE = 0x14430
OFF_TITLE_STATE = 0x14434
TITLE_BLOCK_SIZE = 0x14440

# GameManager (GameManager.hpp:338-451)
OFF_GM_GLOBALS = 0x8  # pointer to ZunGlobals
OFF_GM_CHARACTER = 0x3DBA8
OFF_GM_SHOT_TYPE = 0x3DBA9
OFF_GM_FLAGS = 0x3DBAC  # GameManagerFlags bitfield
OFF_GM_CURRENT_SPELL = 0x3DBB0
OFF_GM_IS_IN_GAME_MENU = 0x3DBB2
OFF_GM_SHOW_RETRY_MENU = 0x3DBB3
OFF_GM_CURRENT_DEMO_REPLAY = 0x3DBB4
OFF_GM_REPLAY_MODE = 0x3DBB5
OFF_GM_GAMEPLAY_FRAME = 0x3DDC0
OFF_GM_CURRENT_STAGE = 0x3DDC4
OFF_GM_STAGE_AT_START = 0x3DDC8
OFF_GM_STAGE_CLEAR_FLAG = 0x3DDD0
OFF_GM_PLAYFIELD_ORIGIN = 0x3DDD4  # Float3 arcadeRegionTopLeftPos
OFF_GM_PLAYFIELD_SIZE = 0x3DDDC  # Float3 arcadeRegionSize
OFF_GM_MOVE_AREA_ORIGIN = 0x3DDE4  # Float3 playerMovementTopLeftPos
OFF_GM_MOVE_AREA_SIZE = 0x3DDEC  # Float3 playerMovementAreaSize
OFF_GM_RUN_ACTIVE_FRAMES = 0x3DE10
OFF_GM_STAGE_ACTIVE_FRAMES = 0x3DE14
OFF_GM_STAGE_STARTUP_MODE = 0x3DE28
OFF_GM_DIFFICULTY = None  # not a plain field: stored inside GameManager::cfg

# ZunGlobals (Global.hpp:364-433); reachable through GameManager::globals
OFF_GLOBALS_SCORE = 0x8
OFF_GLOBALS_GRAZE = 0xC
OFF_GLOBALS_CLOCK_TIME = 0x28
OFF_GLOBALS_TIME_ORBS = 0x3C
OFF_GLOBALS_DEATHS = 0x64
OFF_GLOBALS_LIVES = 0x74
OFF_GLOBALS_BOMBS = 0x80
OFF_GLOBALS_BOMBS_USED = 0x84
OFF_GLOBALS_POWER = 0x98
GLOBALS_READ_SIZE = 0x9C

# Player (Player.hpp:357-506)
OFF_PLAYER_FOCUS_MODE = 0x3
OFF_PLAYER_POSITION = 0x2B4
OFF_PLAYER_VELOCITY = 0x3F8
OFF_PLAYER_HURTBOX_HALF = 0x3D4

# BulletManager / Bullet (BulletManager.hpp:374-496)
OFF_BM_BULLETS = 0x1A880
OFF_BM_LASERS = 0x660938
OFF_BM_ACTIVE_BULLETS = 0x6BA538
BULLET_SIZE = 0x10B8
BULLET_COUNT = 0x601
OFF_BULLET_HURTBOX_X = 0xD34  # BulletTypeSprites::collisionSize.x
OFF_BULLET_POSITION = 0xD44
OFF_BULLET_VELOCITY = 0xD50
OFF_BULLET_STATE = 0xDB8
OFF_BULLET_COLLISION_DISABLED = 0x10B4
BULLET_BLOCK_SIZE = BULLET_SIZE * BULLET_COUNT

# EnemyManager / Enemy (EnemyManager.hpp).  Only the boss slots are read
# individually: ``enemies`` is 481 *contiguous* 0x53D0-byte structs (~10 MB), so
# scanning it would cost 481 reads per frame, while ``bosses[8]`` is a pointer
# array that is 8 reads plus one per live boss.
BOSS_COUNT = 8
OFF_EM_BOSSES = 0x9DCDA0
OFF_EM_ACTIVE_ENEMY_COUNT = 0x9DCDC4
ENEMY_SIZE = 0x53D0
OFF_ENEMY_POSITION = 0x2D34
OFF_ENEMY_LIFE = 0x2DFC

LASER_COUNT = 0x100
LASER_SIZE = 0x59C
OFF_LASER_POSITION = 0x548
OFF_LASER_ANGLE = 0x554
OFF_LASER_START_OFFSET = 0x558
OFF_LASER_END_OFFSET = 0x55C
OFF_LASER_WIDTH = 0x564
OFF_LASER_CURRENT_WIDTH = 0x568
OFF_LASER_IN_USE = 0x584
OFF_LASER_STATE = 0x598

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
# SupervisorState (Supervisor.hpp:105-121)
STATE_EXIT_GAME = -1
STATE_INIT = 0
STATE_TITLE_SCREEN = 1
STATE_GAME_MANAGER = 2
STATE_GAME_MANAGER_REINIT = 3
STATE_RESULT_SCREEN = 5
STATE_RESULT_SCREEN_FROM_GAME = 6
STATE_ENDING = 9
STATE_RESTART_FROM_BEGINNING = 10

# TitleCurrentScreen (TitleScreen.hpp:20-41)
SCREEN_START_MENU = 0
SCREEN_OPTION = 2
SCREEN_KEY_CONFIG = 3
SCREEN_DIFFICULTY_SELECT = 4
SCREEN_CHARACTER_SELECT = 5
SCREEN_REPLAY = 7
SCREEN_DIFFICULTY_SELECT_PRACTICE = 8
SCREEN_CHARACTER_SELECT_PRACTICE = 9
SCREEN_PRACTICE_STAGE_SELECT = 11
SCREEN_DIFFICULTY_SELECT_EXTRA = 12
SCREEN_CHARACTER_SELECT_EXTRA = 13

# TitleScreen menu items (TitleScreen.cpp:79-91)
MENU_ITEM_START = 0
MENU_ITEM_EXTRA_START = 1
MENU_ITEM_SPELL_PRACTICE = 2
MENU_ITEM_PRACTICE_START = 3
MENU_ITEM_REPLAY = 4
MENU_ITEM_RESULT = 5
MENU_ITEM_MUSIC_ROOM = 6
MENU_ITEM_OPTION = 7
MENU_ITEM_QUIT = 8

# Difficulty (Global.hpp)
EASY = 0
NORMAL = 1
HARD = 2
LUNATIC = 3
EXTRA = 4

# GameManagerFlags (GameManager.hpp:27-54)
FLAG_PRACTICE = 1 << 0
FLAG_DEMO = 1 << 1
FLAG_REPLAY = 1 << 3
FLAG_SPELL_PRACTICE = 1 << 14

# Gameplay buttons (Global.hpp:149-175)
BTN_SHOOT = 1 << 0
BTN_BOMB = 1 << 1
BTN_FOCUS = 1 << 2
BTN_MENU = 1 << 3
BTN_UP = 1 << 4
BTN_DOWN = 1 << 5
BTN_LEFT = 1 << 6
BTN_RIGHT = 1 << 7
BTN_SKIP = 1 << 8

# BulletState (BulletManager.hpp:363-372)
BULLET_UNUSED = 0
BULLET_DESPAWNING = 5

# ---------------------------------------------------------------------------
# Playfield geometry (GameManager.cpp:436-439, Player.cpp:1361-1362)
# ---------------------------------------------------------------------------
GAME_WIDTH = 640
GAME_HEIGHT = 480
PLAYFIELD_X = 32.0
PLAYFIELD_Y = 16.0
PLAYFIELD_W = 384.0
PLAYFIELD_H = 448.0
MOVE_AREA_X = 8.0
MOVE_AREA_Y = 16.0
MOVE_AREA_W = 368.0
MOVE_AREA_H = 416.0
PLAYER_START_X = 192.0
PLAYER_START_Y = 384.0

# ---------------------------------------------------------------------------
# Action space contract (MultiDiscrete([9, 2, 2, 2]))
# ---------------------------------------------------------------------------
# move: 0 = idle, 1 = N, 2 = NE, 3 = E, 4 = SE, 5 = S, 6 = SW, 7 = W, 8 = NW
# Screen coordinates: +x right, +y down.
MOVES: tuple[tuple[int, ...], ...] = (
    (),
    (BTN_UP,),
    (BTN_UP, BTN_RIGHT),
    (BTN_RIGHT,),
    (BTN_DOWN, BTN_RIGHT),
    (BTN_DOWN,),
    (BTN_DOWN, BTN_LEFT),
    (BTN_LEFT,),
    (BTN_UP, BTN_LEFT),
)
ACTION_DIMS = (9, 2, 2, 2)  # move, shoot, focus, bomb
N_ACTIONS = 1
for _dim in ACTION_DIMS:
    N_ACTIONS *= _dim  # 72


def encode_flat(move: int, shoot: int, focus: int, bomb: int) -> int:
    """Flatten MultiDiscrete action to ``[0, 72)`` (bijection, unit-tested)."""
    return move * 8 + shoot * 4 + focus * 2 + bomb


def decode_flat(flat: int) -> tuple[int, int, int, int]:
    """Inverse of :func:`encode_flat`."""
    if not 0 <= flat < N_ACTIONS:
        raise ValueError(f"flat action out of range: {flat}")
    move, rest = divmod(flat, 8)
    shoot, rest = divmod(rest, 4)
    focus, bomb = divmod(rest, 2)
    return move, shoot, focus, bomb


def action_to_buttons(action: tuple[int, int, int, int]) -> int:
    """Translate a multi-discrete action into the game's button bitmask."""
    move, shoot, focus, bomb = action
    if not 0 <= move < 9:
        raise ValueError(f"invalid move {move}")
    mask = 0
    for bit in MOVES[move]:
        mask |= bit
    if shoot:
        mask |= BTN_SHOOT
    if focus:
        mask |= BTN_FOCUS
    if bomb:
        mask |= BTN_BOMB
    return mask


def buttons_to_action(mask: int) -> tuple[int, int, int, int]:
    """Best-effort decode of a replay/button mask into a multi-discrete action."""
    up = bool(mask & BTN_UP)
    down = bool(mask & BTN_DOWN)
    left = bool(mask & BTN_LEFT)
    right = bool(mask & BTN_RIGHT)
    vertical = up != down
    horizontal = left != right
    if vertical and horizontal:
        move = 2 if (up and right) else 4 if (down and right) else 6 if (down and left) else 8
    elif up:
        move = 1
    elif right:
        move = 3
    elif down:
        move = 5
    elif left:
        move = 7
    else:
        move = 0
    return move, int(bool(mask & BTN_SHOOT)), int(bool(mask & BTN_FOCUS)), int(bool(mask & BTN_BOMB))


# ---------------------------------------------------------------------------
# Keyboard mapping for the two injection backends
# ---------------------------------------------------------------------------
# name -> (virtual key, scan code)
KEYS: dict[str, tuple[int, int]] = {
    "up": (0x26, 0x48),
    "down": (0x28, 0x50),
    "left": (0x25, 0x4B),
    "right": (0x27, 0x4D),
    "z": (0x5A, 0x2C),
    "x": (0x58, 0x2D),
    "shift": (0x10, 0x2A),
    "esc": (0x1B, 0x01),
    "enter": (0x0D, 0x1C),
    "ctrl": (0x11, 0x1D),
}
