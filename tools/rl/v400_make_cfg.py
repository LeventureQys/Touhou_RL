"""Generate the harness ``th08.cfg`` template (v4.0.0 P0-WP0).

``th08.cfg`` is **not** a text file.  ``Supervisor::LoadConfig`` does

    g_Supervisor.cfg = *(GameConfiguration *)configFileBuffer;

on a fixed 60-byte (0x3C) buffer and rejects anything whose size is not 60 or
whose ``version`` is not ``GAME_VERSION``.  Writing ``key=value`` text produces a
file the game silently discards, so the template is built byte by byte here.

Field layout, from ``Supervisor.hpp`` on ``origin/port/portable-64bit``
(the ``C_ASSERT`` lines in that header pin every offset):

    0x00  ControllerMapping controllerMapping   0x14 bytes
    0x14  i32 version                           0x00080001 == GAME_VERSION
    0x18  i16 padXAxis
    0x1A  i16 padYAxis
    0x1C  u8  lifeCount           (default 2)
    0x1D  u8  bombCount           (default 3)
    0x1E  u8  colorMode16bit
    0x1F  u8  musicMode           0 = MIDI
    0x20  u8  playSounds
    0x21  u8  defaultDifficulty   1 = NORMAL
    0x22  u8  windowed
    0x23  u8  frameskipConfig     0 = full speed
    0x24  u8  effectQuality       2 = MAXIMUM
    0x25  u8  slowMode
    0x26  u8  shotSlow
    0x27  i8  musicVolume
    0x28  i8  sfxVolume
    0x29  i8  serializedReserved29[15]
    0x38  u32 opts               GameConfigOpts bitfield

The harness template turns the things that steal frames or spawn threads off:
windowed, vsync disabled, full speed, no music, no sound effects, music preload
enabled (see 10-P0 §4.7 - that combination is what keeps ``BGMPlayerThread`` from
being created, which is what makes ``fork()`` safe).

    python tools/rl/v400_make_cfg.py                       # write $V400_DATA/game/th08.cfg.rl
    python tools/rl/v400_make_cfg.py --dry-run             # print the bytes, write nothing
    python tools/rl/v400_make_cfg.py --inspect some.cfg    # decode an existing config
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

GAME_VERSION = 0x00080001
CONFIG_SIZE = 0x3C  # 60

# GameConfigOpts bit positions (Supervisor.hpp).
OPT_USE_SW_TEXTURE_BLENDING = 0
OPT_DONT_USE_VERTEX_BUF = 1
OPT_FORCE_16BIT_TEXTURES = 2
OPT_CLEAR_BACKBUFFER_ON_REFRESH = 3
OPT_DISPLAY_MINIMUM_GRAPHICS = 4
OPT_SUPPRESS_GOUROUD_SHADING = 5
OPT_DISABLE_DEPTH_TEST = 6
OPT_FORCE_60FPS = 7
OPT_DISABLE_COLOR_COMPOSITING = 8
OPT_REFERENCE_RASTERIZER_MODE = 9
OPT_DISABLE_FOG = 10
OPT_DONT_USE_DIRECTINPUT = 11
OPT_REDRAW_HUD_EVERY_FRAME = 12
OPT_PRELOAD_MUSIC = 13
OPT_DISABLE_VSYNC = 14
OPT_DONT_DETECT_TEXT_BACKGROUND = 15

# What the harness needs.  Every field is named so a reviewer can check it against
# 10-P0 §2 WP0 without decoding hex.
HARNESS_PROFILE = {
    "version": GAME_VERSION,
    "padXAxis": 600,
    "padYAxis": 600,
    "lifeCount": 2,  # th08.cfg default, per "lifeCount 取默认值"
    "bombCount": 3,
    "colorMode16bit": 0,
    "musicMode": 0,  # MIDI: no WAV streaming thread
    "playSounds": 0,
    "defaultDifficulty": 1,  # NORMAL
    "windowed": 1,
    "frameskipConfig": 0,
    "effectQuality": 2,  # MAXIMUM
    "slowMode": 0,
    "shotSlow": 0,
    "musicVolume": 0,
    "sfxVolume": 0,
    "opts": (1 << OPT_USE_SW_TEXTURE_BLENDING)
    | (1 << OPT_PRELOAD_MUSIC)
    | (1 << OPT_DISABLE_VSYNC),
}

# LoadConfig's own sanity bounds: a field at or above its bound makes the game
# discard the file and fall back to defaults.  Kept here so we can prove our
# template would be accepted.
FIELD_BOUNDS = {
    "lifeCount": 7,
    "bombCount": 4,
    "colorMode16bit": 2,
    "musicMode": 3,
    "defaultDifficulty": 6,
    "playSounds": 2,
    "windowed": 2,
    "frameskipConfig": 3,
    "effectQuality": 3,
    "slowMode": 2,
    "shotSlow": 2,
}


def build(controller_mapping: bytes = b"\x00" * 0x14) -> bytes:
    """Return the 60-byte configuration."""
    if len(controller_mapping) != 0x14:
        raise ValueError(f"controllerMapping must be 0x14 bytes, got {len(controller_mapping)}")
    profile = HARNESS_PROFILE
    buf = bytearray(CONFIG_SIZE)
    buf[0x00:0x14] = controller_mapping
    struct.pack_into("<i", buf, 0x14, profile["version"])
    struct.pack_into("<h", buf, 0x18, profile["padXAxis"])
    struct.pack_into("<h", buf, 0x1A, profile["padYAxis"])
    for offset, name in (
        (0x1C, "lifeCount"),
        (0x1D, "bombCount"),
        (0x1E, "colorMode16bit"),
        (0x1F, "musicMode"),
        (0x20, "playSounds"),
        (0x21, "defaultDifficulty"),
        (0x22, "windowed"),
        (0x23, "frameskipConfig"),
        (0x24, "effectQuality"),
        (0x25, "slowMode"),
        (0x26, "shotSlow"),
    ):
        buf[offset] = profile[name] & 0xFF
    struct.pack_into("<b", buf, 0x27, profile["musicVolume"])
    struct.pack_into("<b", buf, 0x28, profile["sfxVolume"])
    # 0x29..0x37 stays zero: serializedReserved29[15]
    struct.pack_into("<I", buf, 0x38, profile["opts"])
    return bytes(buf)


def decode(data: bytes) -> dict:
    """Decode a config buffer into named fields."""
    if len(data) != CONFIG_SIZE:
        raise ValueError(f"config must be exactly {CONFIG_SIZE} bytes, got {len(data)}")
    opts = struct.unpack_from("<I", data, 0x38)[0]
    out = {
        "controllerMapping": data[0x00:0x14].hex(),
        "version": hex(struct.unpack_from("<i", data, 0x14)[0]),
        "padXAxis": struct.unpack_from("<h", data, 0x18)[0],
        "padYAxis": struct.unpack_from("<h", data, 0x1A)[0],
        "reserved29": data[0x29:0x38].hex(),
        "opts": hex(opts),
        "optBits": {
            name: bool(opts & (1 << bit))
            for name, bit in (
                ("useSwTextureBlending", OPT_USE_SW_TEXTURE_BLENDING),
                ("disableFog", OPT_DISABLE_FOG),
                ("preloadMusic", OPT_PRELOAD_MUSIC),
                ("disableVsync", OPT_DISABLE_VSYNC),
                ("dontUseDirectInput", OPT_DONT_USE_DIRECTINPUT),
            )
        },
    }
    for offset, name in (
        (0x1C, "lifeCount"),
        (0x1D, "bombCount"),
        (0x1E, "colorMode16bit"),
        (0x1F, "musicMode"),
        (0x20, "playSounds"),
        (0x21, "defaultDifficulty"),
        (0x22, "windowed"),
        (0x23, "frameskipConfig"),
        (0x24, "effectQuality"),
        (0x25, "slowMode"),
        (0x26, "shotSlow"),
    ):
        out[name] = data[offset]
    out["musicVolume"] = struct.unpack_from("<b", data, 0x27)[0]
    out["sfxVolume"] = struct.unpack_from("<b", data, 0x28)[0]
    return out


def validate(data: bytes) -> list[str]:
    """Mirror the acceptance test inside Supervisor::LoadConfig. Empty list == accepted."""
    problems = []
    if len(data) != CONFIG_SIZE:
        problems.append(f"size {len(data)} != {CONFIG_SIZE}; LoadConfig discards the file")
        return problems
    version = struct.unpack_from("<i", data, 0x14)[0]
    if version != GAME_VERSION:
        problems.append(f"version {version:#x} != GAME_VERSION {GAME_VERSION:#x}")
    fields = decode(data)
    for name, bound in FIELD_BOUNDS.items():
        if fields[name] >= bound:
            problems.append(f"{name}={fields[name]} >= bound {bound}")
    return problems


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _default_data_dir() -> Path:
    env = os.environ.get("V400_DATA")
    return (Path(env) if env else Path.home() / "v400-data") / "game"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("v400-make-cfg", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="", help="output path (default: $V400_DATA/game/th08.cfg.rl)")
    parser.add_argument("--source-cfg", default="",
                        help="existing th08.cfg to copy controllerMapping from (optional)")
    parser.add_argument("--inspect", default="", help="decode an existing config and exit")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--dry-run", action="store_true", help="print the bytes, write nothing")
    args = parser.parse_args(argv)

    if args.inspect:
        data = Path(args.inspect).read_bytes()
        fields = decode(data)
        problems = validate(data)
        if args.json:
            print(json.dumps({"path": args.inspect, "fields": fields, "problems": problems}, indent=2))
        else:
            for key, value in fields.items():
                print(f"  {key:22s} {value}")
            print(f"  {'sha256':22s} {sha256(data)}")
            print(f"  {'accepted_by_game':22s} {not problems}")
            for problem in problems:
                print(f"  PROBLEM: {problem}")
        return 0 if not problems else 2

    mapping = b"\x00" * 0x14
    if args.source_cfg:
        source = Path(args.source_cfg).read_bytes()
        if len(source) >= 0x14:
            mapping = source[0x00:0x14]
    data = build(mapping)

    problems = validate(data)
    if problems:
        print("generated config would be REJECTED by LoadConfig:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 2

    target = Path(args.out) if args.out else _default_data_dir() / "th08.cfg.rl"
    if args.dry_run:
        print(f"would write {CONFIG_SIZE} bytes to {target}")
        print(f"  sha256 {sha256(data)}")
        print("  " + " ".join(f"{byte:02x}" for byte in data))
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    if args.json:
        print(json.dumps({"path": str(target), "bytes": len(data), "sha256": sha256(data),
                          "fields": decode(data)}, indent=2))
    else:
        print(f"wrote {target} ({len(data)} bytes)")
        print(f"  sha256 {sha256(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
