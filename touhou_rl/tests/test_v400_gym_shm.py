"""Unit tests for the shared-memory helpers (10-P0 §3). Pure CPU, no engine."""

from __future__ import annotations

import pytest

from touhou_rl.v400.gym import shm


def test_layout_constants_are_consistent():
    """The offsets must not overlap and the header must fit in 64 bytes."""
    assert shm.HEADER_SIZE == 64
    assert shm.STATE_OFFSET == shm.HEADER_SIZE
    assert shm.STATE_OFFSET + shm.STATE_BYTES <= shm.PIXELS_OFFSET
    # magic u32 + version u32 + frame u64 + hash u64 + 4 x u32 = 40 bytes,
    # which leaves 24 bytes of the 64-byte header area spare.
    assert shm.HEADER.size == 40
    assert shm.HEADER.size <= shm.HEADER_SIZE


def test_header_struct_layout():
    packed = shm.HEADER.pack(shm.MAGIC, shm.VERSION, 7, 0xDEADBEEF, 640, 480, 100, 3)
    assert len(packed) == 40
    unpacked = shm.HEADER.unpack(packed)
    assert unpacked == (shm.MAGIC, shm.VERSION, 7, 0xDEADBEEF, 640, 480, 100, 3)


@pytest.mark.parametrize("name", ["v400", "/v400", "abc123", "/a" * 1])
def test_segment_path_accepts_plain_names(name):
    path = shm.segment_path(name)
    assert path.name == name.lstrip("/")


@pytest.mark.parametrize("name", ["", "/", "..", ".", "a/b", "../etc/passwd", "/etc/passwd"])
def test_segment_path_rejects_escapes(name):
    with pytest.raises(ValueError):
        shm.segment_path(name)


def test_flag_bits_are_distinct():
    assert shm.FLAG_STATE_VALID != shm.FLAG_PIXELS_VALID
    assert shm.FLAG_STATE_VALID & shm.FLAG_PIXELS_VALID == 0
