"""Decode the harness ``STATE`` export into named fields (P2 §3.2).

The 25 tracked fields are exactly the replay-trace row assembled by
``TrackedFields`` in ``src/rlharness/rlharness.cpp``.  The order below is that
row's order, so index ``i`` here is ``f<i>`` on the wire and offset
``64 + 4*i`` in the shared block.  The harness prints ``f6..f12`` as hex because
they are float bit patterns; they are decoded back to the ``float`` the engine
holds and never re-rounded by this module.
"""

from __future__ import annotations

import struct
from typing import Any, Sequence

#: ``f0..f24`` names, in wire order (``rlharness.cpp`` ``TrackedFields``).
FIELD_NAMES: tuple[str, ...] = (
    "frame",                  # f0  FrameCount()
    "stage",                  # f1  g_GameManager.currentStage
    "replay_frame",           # f2  ReplayManager->frameCounter (0 when not replaying)
    "gameplay_frame",         # f3  gameplayFrameCounter
    "score",                  # f4  globals.score
    "graze",                  # f5  globals.graze
    "deaths",                 # f6  FloatBits(globals.deaths)
    "lives_remaining",        # f7  FloatBits(globals.livesRemaining)
    "bombs_remaining",        # f8  FloatBits(globals.bombsRemaining)
    "player_power",           # f9  FloatBits(globals.playerPower)
    "player_x",               # f10 FloatBits(g_Player.position.x)
    "player_y",               # f11 FloatBits(g_Player.position.y)
    "player_z",               # f12 FloatBits(g_Player.position.z)
    "player_state",           # f13 u8(g_Player.playerState)
    "rng_seed",               # f14 g_Rng.GetSeed()
    "stage_generation",       # f15 generation - g_stageOrigin[stage]
    "gui_input",              # f16 g_GuiMessageInputCurrent
    "point_items_collected",  # f17 globals.pointItemsCollected
    "time_orbs",              # f18 globals.currentTimeOrbs (GameManager::GetTimeOrbs)
    "youkai_gauge",           # f19 globals.youkaiGauge
    "clock_time",             # f20 u8(globals.clockTime)
    "point_item_value",       # f21 globals.pointItemValue
    "character",              # f22 g_GameManager.character
    "shot_type",              # f23 g_GameManager.shotType
    "difficulty",             # f24 u32(g_GameManager.difficulty)
)

FIELD_COUNT = len(FIELD_NAMES)

#: Fields the harness emits as the raw float bit pattern (hex on the wire).
FLOAT_BIT_FIELDS: frozenset[str] = frozenset(
    {
        "deaths",
        "lives_remaining",
        "bombs_remaining",
        "player_power",
        "player_x",
        "player_y",
        "player_z",
    }
)

#: ``PlayerState`` (Player.hpp:86-92).
PLAYER_STATE_ALIVE = 0
PLAYER_STATE_SPAWNING = 1
PLAYER_STATE_DYING = 2
PLAYER_STATE_INVULNERABLE = 3
PLAYER_STATE_NAMES: dict[int, str] = {
    PLAYER_STATE_ALIVE: "ALIVE",
    PLAYER_STATE_SPAWNING: "SPAWNING",
    PLAYER_STATE_DYING: "DYING",
    PLAYER_STATE_INVULNERABLE: "INVULNERABLE",
}


def f32_from_bits(bits: int) -> float:
    """Reinterpret one ``%08x`` float bit pattern as the ``float32`` value."""

    return struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0]


def f32_bits(value: float) -> int:
    """Inverse of :func:`f32_from_bits`, for tests and round-trip assertions."""

    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def decode_state(values: Sequence[int]) -> dict[str, Any]:
    """Decode the 25 raw ``u32`` fields into named, typed values.

    A sequence shorter than 25 is a protocol error and raises: silently
    dropping a field would make a HUD label look like it had a value when the
    engine never sent one.  (Callers that *do* have a partial mapping — tests,
    or a future protocol that appends without renaming — should build a dict
    and pass it to the label functions, which turn a missing key into ``None``.)
    """

    if len(values) != FIELD_COUNT:
        raise ValueError(
            f"STATE has {len(values)} fields, expected {FIELD_COUNT}: {list(values)!r}"
        )
    out: dict[str, Any] = {}
    for name, raw in zip(FIELD_NAMES, values):
        raw = int(raw) & 0xFFFFFFFF
        out[name] = f32_from_bits(raw) if name in FLOAT_BIT_FIELDS else raw
    return out


def state_from_reply(reply: str) -> dict[str, Any]:
    """Decode a ``STATE`` reply line into the same dict :func:`decode_state` makes.

    The reply is ``OK frames=N f0=.. f1=.. ...``; ``f6..f12`` arrive as ``%08x``
    float bit patterns and the rest as decimals (``rlharness.cpp``
    ``HandleState``). Missing keys are an error rather than a zero.
    """

    parts = reply.split()
    verb = parts[0] if parts else ""
    if verb != "OK":
        raise ValueError(f"STATE reply is not OK: {reply!r}")
    fields: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    values: list[int] = []
    for index, name in enumerate(FIELD_NAMES):
        key = f"f{index}"
        if key not in fields:
            raise ValueError(f"STATE reply missing {key} ({name}): {reply!r}")
        token = fields[key]
        values.append(
            int(token, 16) if name in FLOAT_BIT_FIELDS else int(token, 10)
        )
    return decode_state(values)


def player_state_name(value: Any) -> str | None:
    """Human name of a ``playerState`` value, or ``None`` when unavailable."""

    if value is None:
        return None
    return PLAYER_STATE_NAMES.get(int(value), f"UNKNOWN({int(value)})")
