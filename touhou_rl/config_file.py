"""TH08 ``th08.cfg`` handling.

``th08.cfg`` is exactly ``GameConfiguration`` (60 bytes, ``Supervisor.hpp:62-89``).
``Supervisor::LoadConfig`` rejects the file unless ``version == 0x80001``,
``fileSize == 60`` and every enum-ish field is in range, then honours
``opts.disableVsync`` / ``opts.dontUseDirectInput``.

Patching the config file (not process memory) is how the RL runtime selects
windowed mode, muted audio, the default difficulty and -- crucially -- the
``dontUseDirectInput`` input path that makes background/parallel instances
possible.  The file is re-read on every launch, so each run directory owns one.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

GAME_VERSION = 0x80001
CONFIG_SIZE = 60

# Field offsets (Supervisor.hpp C_ASSERTs plus the GameConfiguration layout).
OFF_VERSION = 0x14
OFF_LIFE_COUNT = 0x1C
OFF_BOMB_COUNT = 0x1D
OFF_COLOR_MODE_16BIT = 0x1E
OFF_MUSIC_MODE = 0x1F
OFF_PLAY_SOUNDS = 0x20
OFF_DEFAULT_DIFFICULTY = 0x21
OFF_WINDOWED = 0x22
OFF_FRAMESKIP = 0x23
OFF_EFFECT_QUALITY = 0x24
OFF_SLOW_MODE = 0x25
OFF_SHOT_SLOW = 0x26
OFF_MUSIC_VOLUME = 0x27
OFF_SFX_VOLUME = 0x28
OFF_OPTS = 0x38

# GameConfigOpts bit indices, least significant bit first.
OPT_USE_SW_TEXTURE_BLENDING = 0
OPT_DONT_USE_VERTEX_BUF = 1
OPT_FORCE_16BIT_TEXTURES = 2
OPT_CLEAR_BACKBUFFER = 3
OPT_DISPLAY_MINIMUM_GRAPHICS = 4
OPT_SUPPRESS_GOURAUD = 5
OPT_DISABLE_DEPTH_TEST = 6
OPT_FORCE_60FPS = 7
OPT_DISABLE_COLOR_COMPOSITING = 8
OPT_REFERENCE_RASTERIZER = 9
OPT_DISABLE_FOG = 10
OPT_DONT_USE_DIRECTINPUT = 11
OPT_REDRAW_HUD_EVERY_FRAME = 12
OPT_PRELOAD_MUSIC = 13
OPT_DISABLE_VSYNC = 14
OPT_DONT_DETECT_TEXT_BG = 15

MUSIC_OFF = 0
MUSIC_WAV = 1
MUSIC_MIDI = 2


@dataclass
class GameConfig:
    raw: bytearray

    @classmethod
    def default(cls) -> "GameConfig":
        raw = bytearray(CONFIG_SIZE)
        cfg = cls(raw)
        cfg.version = GAME_VERSION
        cfg.life_count = 2
        cfg.bomb_count = 3
        cfg.color_mode_16bit = 0
        cfg.music_mode = MUSIC_OFF
        cfg.play_sounds = 1
        cfg.default_difficulty = 3
        cfg.windowed = 1
        cfg.frameskip = 0
        cfg.effect_quality = 2
        cfg.music_volume = 0
        cfg.sfx_volume = 80
        cfg.opts = 0
        return cfg

    @classmethod
    def load(cls, path: Path) -> "GameConfig":
        data = path.read_bytes()
        if len(data) != CONFIG_SIZE:
            raise ValueError(f"{path} is {len(data)} bytes, expected {CONFIG_SIZE}")
        return cls(bytearray(data))

    def save(self, path: Path) -> None:
        path.write_bytes(bytes(self.raw))

    # -- typed accessors --------------------------------------------------
    @property
    def version(self) -> int:
        return struct.unpack_from("<i", self.raw, OFF_VERSION)[0]

    @version.setter
    def version(self, value: int) -> None:
        struct.pack_into("<i", self.raw, OFF_VERSION, value)

    def _get_u8(self, offset: int) -> int:
        return self.raw[offset]

    def _set_u8(self, offset: int, value: int) -> None:
        self.raw[offset] = value & 0xFF

    life_count = property(lambda s: s._get_u8(OFF_LIFE_COUNT), lambda s, v: s._set_u8(OFF_LIFE_COUNT, v))
    bomb_count = property(lambda s: s._get_u8(OFF_BOMB_COUNT), lambda s, v: s._set_u8(OFF_BOMB_COUNT, v))
    color_mode_16bit = property(
        lambda s: s._get_u8(OFF_COLOR_MODE_16BIT), lambda s, v: s._set_u8(OFF_COLOR_MODE_16BIT, v)
    )
    music_mode = property(lambda s: s._get_u8(OFF_MUSIC_MODE), lambda s, v: s._set_u8(OFF_MUSIC_MODE, v))
    play_sounds = property(lambda s: s._get_u8(OFF_PLAY_SOUNDS), lambda s, v: s._set_u8(OFF_PLAY_SOUNDS, v))
    default_difficulty = property(
        lambda s: s._get_u8(OFF_DEFAULT_DIFFICULTY), lambda s, v: s._set_u8(OFF_DEFAULT_DIFFICULTY, v)
    )
    windowed = property(lambda s: s._get_u8(OFF_WINDOWED), lambda s, v: s._set_u8(OFF_WINDOWED, v))
    frameskip = property(lambda s: s._get_u8(OFF_FRAMESKIP), lambda s, v: s._set_u8(OFF_FRAMESKIP, v))
    effect_quality = property(lambda s: s._get_u8(OFF_EFFECT_QUALITY), lambda s, v: s._set_u8(OFF_EFFECT_QUALITY, v))
    music_volume = property(
        lambda s: struct.unpack_from("<b", s.raw, OFF_MUSIC_VOLUME)[0],
        lambda s, v: struct.pack_into("<b", s.raw, OFF_MUSIC_VOLUME, v),
    )
    sfx_volume = property(
        lambda s: struct.unpack_from("<b", s.raw, OFF_SFX_VOLUME)[0],
        lambda s, v: struct.pack_into("<b", s.raw, OFF_SFX_VOLUME, v),
    )

    @property
    def opts(self) -> int:
        return struct.unpack_from("<I", self.raw, OFF_OPTS)[0]

    @opts.setter
    def opts(self, value: int) -> None:
        struct.pack_into("<I", self.raw, OFF_OPTS, value & 0xFFFFFFFF)

    def set_opt(self, bit: int, enabled: bool) -> None:
        opts = self.opts
        opts = (opts | (1 << bit)) if enabled else (opts & ~(1 << bit))
        self.opts = opts

    def get_opt(self, bit: int) -> bool:
        return bool(self.opts & (1 << bit))


def prepare_config(
    path: Path,
    *,
    windowed: bool,
    default_difficulty: int,
    use_postmessage_input: bool,
    muted: bool = True,
) -> GameConfig:
    """Load (or create) a config and apply the RL runtime profile."""
    cfg = GameConfig.load(path) if path.exists() else GameConfig.default()
    cfg.version = GAME_VERSION
    cfg.windowed = 1 if windowed else 0
    cfg.default_difficulty = default_difficulty
    if muted:
        cfg.music_mode = MUSIC_OFF
        cfg.play_sounds = 1
        cfg.music_volume = 0
    cfg.set_opt(OPT_DONT_USE_DIRECTINPUT, use_postmessage_input)
    cfg.set_opt(OPT_DISABLE_VSYNC, True)
    cfg.save(path)
    return cfg
