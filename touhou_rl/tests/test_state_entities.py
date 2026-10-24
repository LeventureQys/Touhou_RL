"""Unit tests for the v3.5.0 S0 entity reads (no game process)."""

from __future__ import annotations

import ctypes
import struct

import pytest

from touhou_rl import spec
from touhou_rl.state import GameStateReader


class FakeMemory:
    """Sparse read-only memory: enough of ProcessMemory for GameStateReader."""

    def __init__(self, base: int = spec.IMAGE_BASE):
        self.module_base = base
        self.reads = 0
        self.failures = 0
        self.blocks: list[tuple[int, bytes]] = []

    def add(self, address: int, data: bytes | bytearray) -> None:
        self.blocks.append((int(address), bytes(data)))

    def read(self, address: int, size: int) -> bytes:
        for start, data in self.blocks:
            if start <= address and address + size <= start + len(data):
                self.reads += 1
                offset = address - start
                return data[offset: offset + size]
        self.failures += 1
        raise OSError(f"fake read 0x{address:X}+{size} failed")

    def read_into(self, address: int, buffer) -> None:
        size = ctypes.sizeof(buffer)
        data = self.read(address, size)
        ctypes.memmove(buffer, data, size)

    def is_alive(self) -> bool:
        return True


def _enemy_buffer() -> bytearray:
    buf = bytearray(spec.ENEMY_SIZE * spec.ENEMY_SCAN_COUNT)
    size = spec.ENEMY_SIZE

    def put(slot: int, offset: int, fmt: str, *values) -> None:
        struct.pack_into(fmt, buf, slot * size + offset, *values)

    # slot 5: live normal enemy with a readable ECL type.
    put(5, spec.OFF_ENEMY_FLAGS1, "<I", spec.ENEMY_FLAG_ACTIVE)
    put(5, spec.OFF_ENEMY_POSITION, "<fff", 100.0, 200.0, 0.0)
    put(5, spec.OFF_ENEMY_LIFE, "<i", 20)
    put(5, spec.OFF_ENEMY_MAX_LIFE, "<i", 20)
    put(5, spec.OFF_ENEMY_ITEM_DROP, "<b", 2)
    put(5, spec.OFF_ENEMY_BOSS_SLOT, "<B", 0)
    put(5, spec.OFF_ENEMY_ACTIVE_ECL_CONTEXT, "<I", 0x30000000)
    # slot 7: inactive (must be filtered out).
    # slot 9: live boss.
    put(9, spec.OFF_ENEMY_FLAGS1, "<I", spec.ENEMY_FLAG_ACTIVE | spec.ENEMY_FLAG_BOSS)
    put(9, spec.OFF_ENEMY_POSITION, "<fff", 192.0, 80.0, 0.0)
    put(9, spec.OFF_ENEMY_LIFE, "<i", 7300)
    put(9, spec.OFF_ENEMY_MAX_LIFE, "<i", 7300)
    put(9, spec.OFF_ENEMY_BOSS_SLOT, "<B", 0)
    return buf


def _item_buffer() -> bytearray:
    buf = bytearray(spec.ITEM_SIZE * spec.ITEM_COUNT)
    size = spec.ITEM_SIZE

    def put(slot: int, offset: int, fmt: str, *values) -> None:
        struct.pack_into(fmt, buf, slot * size + offset, *values)

    put(0, spec.OFF_ITEM_IN_USE, "<b", 1)
    put(0, spec.OFF_ITEM_ONSCREEN, "<b", 1)
    put(0, spec.OFF_ITEM_TYPE, "<b", spec.ITEM_POWER_SMALL)
    put(0, spec.OFF_ITEM_STATE, "<b", spec.ITEM_STATE_DEFAULT)
    put(0, spec.OFF_ITEM_CURRENT_POSITION, "<fff", 10.0, 300.0, 0.0)
    put(1, spec.OFF_ITEM_IN_USE, "<b", 0)
    put(2, spec.OFF_ITEM_IN_USE, "<b", 1)
    put(2, spec.OFF_ITEM_ONSCREEN, "<b", 1)
    put(2, spec.OFF_ITEM_TYPE, "<b", spec.ITEM_POWER_BIG)
    put(2, spec.OFF_ITEM_STATE, "<b", spec.ITEM_STATE_AUTOCOLLECT)
    put(2, spec.OFF_ITEM_CURRENT_POSITION, "<fff", 50.0, 120.0, 0.0)
    return buf


def _reader() -> tuple[GameStateReader, FakeMemory]:
    mem = FakeMemory()
    reader = GameStateReader(mem)  # type: ignore[arg-type]
    mem.add(spec.ADDR_ENEMY_MANAGER + spec.OFF_EM_ENEMIES, _enemy_buffer())
    mem.add(spec.ADDR_ITEM_MANAGER, _item_buffer())
    # Enemy 5's ECL context (pointer read by read_enemies).
    context = bytearray(spec.OFF_ECL_CONTEXT_SUB_ID + 2)
    struct.pack_into("<h", context, spec.OFF_ECL_CONTEXT_SUB_ID, 42)
    mem.add(0x30000000, context)
    # Shot file pointer + params.
    sht = bytearray(spec.SHT_READ_SIZE)
    struct.pack_into("<fff", sht, spec.OFF_SHT_ITEM_AUTOCOLLECT_SPEED, 10.0, 24.0, 128.0)
    mem.add(0x31000000, sht)
    player_ptr = bytearray(4)
    struct.pack_into("<I", player_ptr, 0, 0x31000000)
    mem.add(spec.ADDR_PLAYER + spec.OFF_PLAYER_PRIMARY_SHT_FILE, player_ptr)
    return reader, mem


def test_read_enemies_filters_and_parses():
    reader, _mem = _reader()
    enemies = reader.read_enemies()
    assert [enemy.slot for enemy in enemies] == [5, 9]
    normal = enemies[0]
    assert normal.active is True and normal.boss is False
    assert normal.x == pytest.approx(100.0)
    assert normal.y == pytest.approx(200.0)
    assert normal.life == 20 and normal.max_life == 20
    assert normal.item_drop == 2
    assert normal.sub_id == 42
    boss = enemies[1]
    assert boss.boss is True
    assert boss.life == 7300
    assert boss.sub_id is None  # null context pointer -> unknown type


def test_read_enemies_inactive_only_keeps_all_slots():
    reader, _mem = _reader()
    assert len(reader.read_enemies(active_only=False, with_sub_id=False)) == spec.ENEMY_SCAN_COUNT


def test_read_items_filters_and_parses():
    reader, _mem = _reader()
    items = reader.read_items()
    assert [item.slot for item in items] == [0, 2]
    assert items[0].item_type == spec.ITEM_POWER_SMALL
    assert items[0].state == spec.ITEM_STATE_DEFAULT
    assert items[0].y == pytest.approx(300.0)
    assert items[1].item_type == spec.ITEM_POWER_BIG
    assert items[1].state == spec.ITEM_STATE_AUTOCOLLECT


def test_read_shot_params_reads_the_line():
    reader, _mem = _reader()
    shot = reader.read_shot_params()
    assert shot is not None
    assert shot.point_item_value_line == pytest.approx(128.0)
    assert shot.item_collection_box_size == pytest.approx(24.0)
    assert shot.item_auto_collect_speed == pytest.approx(10.0)


def test_read_shot_params_missing_pointer_is_none():
    mem = FakeMemory()
    reader = GameStateReader(mem)  # type: ignore[arg-type]
    pointer = bytearray(4)  # zero
    mem.add(spec.ADDR_PLAYER + spec.OFF_PLAYER_PRIMARY_SHT_FILE, pointer)
    assert reader.read_shot_params() is None


def test_read_frame_event_flags():
    mem = FakeMemory()
    reader = GameStateReader(mem)  # type: ignore[arg-type]
    pointer = bytearray(4)
    struct.pack_into("<I", pointer, 0, 0x32000000)
    mem.add(spec.ADDR_REPLAY_MANAGER, pointer)
    events = bytearray(spec.OFF_RM_FRAME_EVENT_FLAGS + 2)
    struct.pack_into("<H", events, spec.OFF_RM_FRAME_EVENT_FLAGS, spec.FRAME_EVENT_ITEM_COLLECTED)
    mem.add(0x32000000, events)
    assert reader.read_frame_event_flags() == spec.FRAME_EVENT_ITEM_COLLECTED


def test_read_frame_event_flags_missing_manager_is_none():
    mem = FakeMemory()
    reader = GameStateReader(mem)  # type: ignore[arg-type]
    pointer = bytearray(4)  # null g_ReplayManager
    mem.add(spec.ADDR_REPLAY_MANAGER, pointer)
    assert reader.read_frame_event_flags() is None


def test_item_layout_matches_manager_size():
    # The whole S0 item contract rests on these layout identities:
    # items[2097] + nextIndex(4) + itemCount(4) + itemListHead(0x2E4) + tail(4).
    items = spec.ITEM_COUNT * spec.ITEM_SIZE
    assert items == 0x17ADA4
    assert items + 8 + spec.ITEM_SIZE + 4 == 0x17B094
    assert spec.OFF_IM_ITEM_COUNT == items + 4
