"""P0 contract checks for the v400 shared-memory reader."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from touhou_rl.v400.gym import shm


def _write_block(path, *, width=2, height=2, magic=shm.MAGIC, payload_size=None):
    """Write only the bytes needed for a synthetic read-only shm block."""
    size = shm.PIXELS_OFFSET + width * height * 4
    if payload_size is not None:
        size = payload_size
    data = bytearray(size)
    shm.HEADER.pack_into(
        data,
        0,
        magic,
        shm.VERSION,
        17,
        0x1234,
        width,
        height,
        shm.STATE_BYTES,
        shm.FLAG_STATE_VALID | shm.FLAG_PIXELS_VALID,
    )
    if size >= shm.STATE_OFFSET + shm.STATE_BYTES:
        np.arange(shm.STATE_FIELDS, dtype="<u4").tofile  # keep dtype contract explicit
        data[shm.STATE_OFFSET : shm.STATE_OFFSET + shm.STATE_BYTES] = np.arange(
            shm.STATE_FIELDS, dtype="<u4"
        ).tobytes()
    if size >= shm.PIXELS_OFFSET + width * height * 4:
        # mmap stores bottom row first; pixels() must return top row first.
        rows = bytes(range(8, 16)) + bytes(range(0, 8))
        data[shm.PIXELS_OFFSET : shm.PIXELS_OFFSET + len(rows)] = rows
    path.write_bytes(data)


@pytest.fixture
def shm_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(shm, "SHM_DIR", tmp_path)
    return tmp_path


def test_open_map_close_and_name_normalization(shm_dir):
    path = shm_dir / "p0-block"
    _write_block(path)

    assert shm.segment_path("/p0-block") == shm.segment_path("p0-block") == path
    block = shm.SharedBlock("/p0-block")
    try:
        assert block.size == shm.PIXELS_OFFSET + 2 * 2 * 4
        assert block.header()["version"] == shm.VERSION
        assert block.state().shape == (shm.STATE_FIELDS,)
    finally:
        block.close()


def test_header_magic_is_validated(shm_dir):
    path = shm_dir / "bad-magic"
    _write_block(path, magic=0)

    with shm.SharedBlock("bad-magic") as block:
        with pytest.raises(ValueError, match="bad magic"):
            block.header()


def test_too_small_and_truncated_pixel_blocks_are_rejected(shm_dir):
    too_small = shm_dir / "too-small"
    _write_block(too_small, payload_size=shm.PIXELS_OFFSET - 1)
    with pytest.raises(ValueError, match="too small"):
        shm.SharedBlock("too-small")

    truncated = shm_dir / "truncated"
    _write_block(truncated, width=2, height=2, payload_size=shm.PIXELS_OFFSET + 15)
    with shm.SharedBlock("truncated") as block:
        with pytest.raises(ValueError, match="need 16"):
            block.pixels()


def test_state_and_pixels_have_contract_shape_and_orientation(shm_dir):
    path = shm_dir / "payload"
    _write_block(path)

    with shm.SharedBlock("payload") as block:
        np.testing.assert_array_equal(block.state(), np.arange(shm.STATE_FIELDS, dtype="<u4"))
        pixels = block.pixels()
        assert pixels.shape == (2, 2, 4)
        np.testing.assert_array_equal(pixels[0].reshape(-1), np.arange(0, 8, dtype=np.uint8))
        np.testing.assert_array_equal(pixels[1].reshape(-1), np.arange(8, 16, dtype=np.uint8))


def test_invalid_names_are_rejected(shm_dir):
    for name in ("", "/", "a/b", "..", "/../escape"):
        with pytest.raises(ValueError, match="invalid shared-memory name"):
            shm.segment_path(name)
