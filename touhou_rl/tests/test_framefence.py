"""Tests for the frame-fence reader and the variant address table.

These run without the game: the shared section is faked with a local buffer,
and the address table is checked against the canonical spec constants (which is
the property that makes the variant image usable at all).
"""

from __future__ import annotations

import ctypes
import importlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from touhou_rl import framefence, spec

ROOT = Path(__file__).resolve().parents[2]
TH08 = ROOT / "third_party/th08"
ALL_SYMBOLS = {
    "g_Supervisor": 0x017DC380,
    "g_GameManager": 0x0161C7A0,
    "g_Player": 0x017E3120,
    "g_BulletManager": 0x00F620B8,
    "g_EnemyManager": 0x00585148,
    "g_ItemManager": 0x01660880,
    "g_ReplayManager": 0x018C5C50,
    "g_TitleScreen": 0x018CB030,
    "g_CurFrameInput": 0x0165A808,
    "g_LastFrameInput": 0x0165A804,
    "g_GuiMessageInputCurrent": 0x0161C674,
    "g_GuiMessageInputPrevious": 0x0161C678,
    "g_Gui": 0x0161C680,
}


# ---------------------------------------------------------------------------
# ABI / hashing
# ---------------------------------------------------------------------------
def test_layout_constants_match_the_publisher():
    assert framefence.FENCE_MAGIC == 0x38454E46
    assert framefence.PIXEL_BYTES == 640 * 480 * 3
    assert framefence.TOTAL_BYTES == 0x1000 + framefence.PIXEL_BYTES
    # The C side C_ASSERTs sizeof(FenceTelemetry) == 0x88.
    assert framefence.TELEMETRY_SIZE == 0x88
    assert max(framefence.TELEMETRY_FIELDS.values()) + 4 == framefence.TELEMETRY_SIZE
    assert max(framefence.CONTROL_FIELDS.values()) + 4 == 0x20
    assert framefence.TELEMETRY_OFFSET >= 0x20


def test_fnv1a_matches_published_vectors():
    assert framefence.fnv1a32(b"") == 2166136261
    assert framefence.fnv1a32(b"hello") == 0x4F9F2CAB
    assert framefence.fnv1a32(b"a") == 0xE40C292C


def test_bullet_digest_hashes_state_then_x_then_y():
    from touhou_rl import state

    raw = bytearray(spec.BULLET_COUNT * spec.BULLET_SIZE)
    # slot 0: state=5, x=1.5, y=-2.25; slot 1: state=0, x=0, y=0
    struct.pack_into("<H", raw, 0 * spec.BULLET_SIZE + spec.OFF_BULLET_STATE, 5)
    struct.pack_into("<f", raw, 0 * spec.BULLET_SIZE + spec.OFF_BULLET_POSITION, 1.5)
    struct.pack_into("<f", raw, 0 * spec.BULLET_SIZE + spec.OFF_BULLET_POSITION + 4, -2.25)
    view = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(spec.BULLET_COUNT, spec.BULLET_SIZE)
    field = state.BulletField(
        raw=view,
        position=state._strided(view, spec.OFF_BULLET_POSITION, (spec.BULLET_COUNT, 2), np.float32, spec.BULLET_SIZE),
        velocity=state._strided(view, spec.OFF_BULLET_VELOCITY, (spec.BULLET_COUNT, 2), np.float32, spec.BULLET_SIZE),
        state=state._strided(view, spec.OFF_BULLET_STATE, (spec.BULLET_COUNT,), np.uint16, spec.BULLET_SIZE),
        disabled=np.ndarray(
            shape=(spec.BULLET_COUNT,),
            dtype=np.int8,
            buffer=view,
            offset=spec.OFF_BULLET_COLLISION_DISABLED,
            strides=(spec.BULLET_SIZE,),
        ),
        half_size=state._strided(view, spec.OFF_BULLET_HURTBOX_X, (spec.BULLET_COUNT, 2), np.float32, spec.BULLET_SIZE),
    )

    digest = framefence.FenceStateDigest(reader=None).bullet_digest(field)

    expected = 2166136261
    for slot in range(spec.BULLET_COUNT):
        words = [5, 0x3FC00000, 0xC0100000] if slot == 0 else [0, 0, 0]
        for word in words:
            for shift in (0, 8, 16, 24):
                expected = ((expected ^ ((word >> shift) & 0xFF)) * 16777619) & 0xFFFFFFFF
    assert digest == expected


# ---------------------------------------------------------------------------
# Section decoding (fake local buffer, no game)
# ---------------------------------------------------------------------------
def _fake_section() -> tuple[framefence.FrameFence, ctypes.Array]:
    buffer = ctypes.create_string_buffer(framefence.TOTAL_BYTES)
    fence = framefence.FrameFence(0, ctypes.addressof(buffer), owns_view=False)
    return fence, buffer


def test_telemetry_and_control_roundtrip():
    fence, _buffer = _fake_section()
    base = framefence.TELEMETRY_OFFSET
    for name, value in (
        ("magic", framefence.FENCE_MAGIC),
        ("version", framefence.FENCE_VERSION),
        ("block_size", framefence.TOTAL_BYTES),
        ("pid", 4242),
        ("ticket", 77),
        ("logic_frame", 9001),
        ("player_x_bits", struct.unpack("<I", struct.pack("<f", 192.5))[0]),
        ("player_y_bits", struct.unpack("<I", struct.pack("<f", 384.25))[0]),
        ("bullet_digest", 0xDEADBEEF),
        ("enemy_digest", 0xFEEDFACE),
        ("digest_ticket", 77),
        ("hold_phase", 1),
        ("pixel_ticket", 77),
        ("qpc_high", 0x1234),
        ("qpc_low", 0x5678),
    ):
        fence._write(base, framefence.TELEMETRY_FIELDS[name], value)

    assert fence.valid()
    assert fence.header()["pid"] == 4242
    telemetry = fence.telemetry()
    assert telemetry.ticket == 77
    assert telemetry.logic_frame == 9001
    assert telemetry.player_x == 192.5
    assert telemetry.player_y == 384.25
    assert telemetry.bullet_digest == 0xDEADBEEF
    assert telemetry.enemy_digest == 0xFEEDFACE
    assert telemetry.hold_phase == 1
    assert telemetry.qpc == (0x1234 << 32) | 0x5678
    assert fence.ticket == 77 and fence.hold_phase == 1

    fence.configure(enable=True, pixels=True, digest_always=True, hold_every=3, hold_min_us=11, hold_max_us=22)
    control = fence.control()
    assert control["flags"] == (
        framefence.CTL_ENABLE | framefence.CTL_PIXELS | framefence.CTL_DIGEST_ALWAYS
    )
    assert control["hold_every"] == 3
    assert control["hold_min_us"] == 11
    assert control["hold_max_us"] == 22

    fence.ack(76)
    assert fence.control()["ack_ticket"] == 76
    fence.disable()
    assert fence.control()["flags"] == 0 and fence.control()["hold_every"] == 0


def test_invalid_magic_is_rejected():
    fence, _buffer = _fake_section()
    assert not fence.valid()
    fence._write(framefence.TELEMETRY_OFFSET, framefence.TELEMETRY_FIELDS["magic"], 0x11111111)
    assert not fence.valid()


def test_pixels_view_shape_and_copy():
    fence, buffer = _fake_section()
    first = framefence.PIXELS_OFFSET
    for index in range(6):
        buffer[first + index] = index + 1
    pixels = fence.pixels()
    assert pixels.shape == (480, 640, 3) and pixels.dtype == np.uint8
    assert pixels.reshape(-1)[:6].tolist() == [1, 2, 3, 4, 5, 6]
    # pixels() must return a copy, not a view into the section
    buffer[first] = 0
    assert pixels.reshape(-1)[0] == 1


# ---------------------------------------------------------------------------
# Variant address table
# ---------------------------------------------------------------------------
def test_address_table_override_is_loaded_and_validated(tmp_path, monkeypatch):
    payload = {
        "schema": "th08-map-addresses/1",
        "imageBase": "0x00400000",
        "symbols": {
            name: {"mangled": f"?{name}@th08@@3XA", "address": f"0x{address + 0x3000:08X}"}
            for name, address in ALL_SYMBOLS.items()
        },
    }
    path = tmp_path / "addresses.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setenv(spec.ADDRESS_FILE_ENV, str(path))
    reloaded = importlib.reload(spec)
    try:
        assert reloaded.ADDR_SUPERVISOR == ALL_SYMBOLS["g_Supervisor"] + 0x3000
        assert reloaded.ADDR_GUI == ALL_SYMBOLS["g_Gui"] + 0x3000
        assert reloaded.IMAGE_BASE == 0x00400000
        assert reloaded.ADDRESS_SOURCE["addressFile"] == str(path)
    finally:
        monkeypatch.delenv(spec.ADDRESS_FILE_ENV, raising=False)
        importlib.reload(spec)

    assert spec.ADDR_SUPERVISOR == ALL_SYMBOLS["g_Supervisor"]
    assert spec.ADDRESS_SOURCE == {}


def test_address_table_rejects_wrong_schema(tmp_path, monkeypatch):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "something-else", "symbols": {}}), encoding="utf-8")
    monkeypatch.setenv(spec.ADDRESS_FILE_ENV, str(path))
    try:
        with pytest.raises(ValueError):
            importlib.reload(spec)
    finally:
        monkeypatch.delenv(spec.ADDRESS_FILE_ENV, raising=False)
        importlib.reload(spec)


def test_map_parser_reproduces_canonical_constants():
    """The generated table must match the hard-coded canonical addresses."""
    map_path = TH08 / "build" / "th08.map"
    if not map_path.is_file():
        pytest.skip("canonical linker map not present")
    import importlib.util

    script = TH08 / "scripts" / "fence-addresses.py"
    loader_spec = importlib.util.spec_from_file_location("fence_addresses", script)
    module = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(module)

    symbols, base = module.parse_map(map_path)
    found = module.friendly_lookup(symbols)
    assert base == 0x00400000
    for name, address in ALL_SYMBOLS.items():
        assert name in found, name
        assert found[name][1] == address, name
