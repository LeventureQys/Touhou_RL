"""Fail-closed unit checks; never count synthetic fixtures as real WP5 evidence."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'rl' / 'v400_wp5_pixel_rebuild.py'
spec = importlib.util.spec_from_file_location('v400_wp5_pixel_rebuild', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_one_hundred_per_frame_inputs():
    assert module.FRAMES == 100
    assert len(module.MASKS) == 100
    assert all(type(mask) is int and mask == 0 for mask in module.MASKS)
    assert module.PIXEL_BYTES == 640 * 480 * 4
    assert module.HEADER.size == 40


def test_sha_is_raw_byte_hash_not_png_metadata():
    raw = bytes(range(256))
    assert module.sha(raw) == hashlib.sha256(raw).hexdigest()
    assert module.sha(raw) != module.sha(raw + b'\x00')


def test_new_data_dir_must_not_reuse_existing_directory(tmp_path):
    existing = tmp_path / 'exists'
    existing.mkdir()
    with pytest.raises(module.Blocked, match='already exists'):
        module.create_data(existing)


def test_new_data_dir_must_not_reuse_symlink(tmp_path):
    existing = tmp_path / 'already-a-link'
    try:
        existing.symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f'platform does not permit symlinks: {error}')
    with pytest.raises(module.Blocked, match='already exists'):
        module.create_data(existing)
