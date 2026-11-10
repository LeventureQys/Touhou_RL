"""Fail-closed unit checks for the LP_NUM_THREADS pixel-determinism tool.

Synthetic frames are used here only to exercise the comparison arithmetic; the
real evidence is the WSL run in EXP-P0-070.  Nothing in this file may be cited
as pixel evidence.
"""
import hashlib
import importlib.util
import struct
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'rl' / 'v400_render_pixel_determinism.py'
spec = importlib.util.spec_from_file_location('v400_render_pixel_determinism', SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module  # dataclasses needs the module registered
spec.loader.exec_module(module)

W, H = 640, 480


def header(frame=1, magic=module.SHM_MAGIC, version=1, width=W, height=H, flags=2):
    """The engine's 192-byte SHM header (40 used bytes + reserved padding)."""
    return module.HEADER.pack(magic, version, frame, 0, width, height, 100, flags) \
        + bytes(module.SHM_PIXELS_OFFSET - module.HEADER.size)


def record(index, raw=b'', checksum='00', state='aa'):
    return {'frame': index, 'sha256': hashlib.sha256(raw).hexdigest(),
            'checksum': checksum, 'nonzero': None, 'state_hash': state,
            'shm_state_hash': '0' * 16, 'raw': raw}


def test_constants_match_the_engine_protocol():
    assert module.PIXEL_WIDTH == 640 and module.PIXEL_HEIGHT == 480
    assert module.PIXEL_BYTES == 640 * 480 * 4
    assert module.SHM_PIXELS_OFFSET == 192
    assert module.SHM_TOTAL_BYTES == 192 + module.PIXEL_BYTES
    assert module.HEADER.size == 40
    assert module.FRAMES == 100
    assert module.START_ARGS == '0 1 0 1 0'


def test_valid_header_is_accepted_and_reports_frame():
    parsed = module.parse_header(header(frame=7) + bytes(module.PIXEL_BYTES))
    assert parsed['frame'] == 7
    assert (parsed['width'], parsed['height']) == (W, H)
    assert parsed['flags'] & module.SHM_FLAG_PIXELS_VALID


@pytest.mark.parametrize('kind', ['short', 'magic', 'version', 'width', 'flags'])
def test_bad_header_fails_closed(kind):
    raw = {
        'short': b'too short',
        'magic': header(magic=0xDEADBEEF) + bytes(module.PIXEL_BYTES),
        'version': header(version=2) + bytes(module.PIXEL_BYTES),
        'width': header(width=320) + bytes(module.PIXEL_BYTES),
        'flags': header(flags=0) + bytes(module.PIXEL_BYTES),
    }[kind]
    with pytest.raises(module.Blocked):
        module.parse_header(raw)


def test_diff_stats_counts_raw_bytes_and_bounding_box():
    a = bytes(module.PIXEL_BYTES)
    b = bytearray(a)
    b[0] = 1                            # pixel (0, 0)
    b[(H - 1) * W * 4 + (W - 1) * 4] = 2  # pixel (W-1, H-1)
    stats = module.diff_stats(a, bytes(b))
    assert stats['different_bytes'] == 2
    assert stats['different_pixels'] == 2
    assert stats['first_byte'] == 0
    assert stats['bbox'] == {'x0': 0, 'y0': 0, 'x1': W - 1, 'y1': H - 1}


def test_diff_stats_equal_frames_report_nothing():
    raw = bytes(range(256)) * 1
    stats = module.diff_stats(raw, raw)
    assert stats == {'different_bytes': 0, 'different_pixels': 0, 'first_byte': None,
                     'last_byte': None, 'bbox': None}


def test_identical_series_is_all_equal():
    left = [record(i + 1, bytes([i, 1, 2, 3] * 4)) for i in range(100)]
    right = [record(i + 1, bytes([i, 1, 2, 3] * 4)) for i in range(100)]
    result = module.compare_series(left, right)
    assert result['frames'] == 100
    assert result['equal_frames'] == 100
    assert result['different_frames'] == 0
    assert result['first_different_frame'] is None
    assert result['checksum_different_frames'] == 0


def test_divergent_series_reports_first_frame_and_byte_count():
    left = [record(i + 1, bytes([i, 1, 2, 3] * 4), checksum='aa') for i in range(100)]
    right = [record(i + 1, bytes([i, 1, 2, 3] * 4), checksum='aa') for i in range(100)]
    right[6]['raw'] = bytearray(right[6]['raw'])
    right[6]['raw'][9] ^= 0xFF
    right[6]['raw'] = bytes(right[6]['raw'])
    right[6]['sha256'] = hashlib.sha256(right[6]['raw']).hexdigest()
    right[6]['checksum'] = 'bb'
    result = module.compare_series(left, right)
    assert result['equal_frames'] == 99
    assert result['different_frames'] == 1
    assert result['first_different_frame'] == 7
    assert result['first_difference']['different_bytes'] == 1
    assert result['checksum_different_frames'] == 1


def test_checksum_difference_without_byte_difference_is_counted():
    left = [record(1, b'abcd', checksum='aa')]
    right = [record(1, b'abcd', checksum='bb')]
    result = module.compare_series(left, right)
    assert result['equal_frames'] == 1
    assert result['checksum_different_frames'] == 1


def test_sha_difference_with_equal_bytes_is_a_tool_defect():
    left = [record(1, b'abcd')]
    right = [record(1, b'abcd')]
    right[0]['sha256'] = 'f' * 64
    with pytest.raises(module.Blocked, match='sha differs but bytes are equal'):
        module.compare_series(left, right)


def test_series_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        module.compare_series([record(1, b'abcd')], [])


def test_verdict_requires_every_group_to_be_equal():
    equal = {'frames': 100, 'equal_frames': 100, 'different_frames': 0,
             'first_different_frame': None}
    report = {'groups': {'within': {'4': dict(equal)},
                         'cross': {'4': {'p0': dict(equal), 'p1': dict(equal)}}}}
    good = module.verdict(report, 1, 2)
    assert good['all_equal'] and good['usable_for_training']
    assert len(good['checks']) == 3

    bad_report = {'groups': {'within': {'4': dict(equal)},
                             'cross': {'4': {'p0': dict(equal, equal_frames=99,
                                                        different_frames=1,
                                                        first_different_frame=42),
                                             'p1': dict(equal)}}}}
    bad = module.verdict(bad_report, 1, 2)
    assert not bad['all_equal'] and not bad['usable_for_training']
    assert bad['failed_checks'] == ['1 vs 4 [p0]']


def test_child_env_forces_the_requested_lp_threads(monkeypatch):
    monkeypatch.setenv('LP_NUM_THREADS', '99')
    monkeypatch.setenv('GALLIUM_DRIVER', 'd3d12')
    env = module.build_env(4, ':98')
    assert env['LP_NUM_THREADS'] == '4'
    assert env['GALLIUM_DRIVER'] == 'llvmpipe'
    assert env['LIBGL_ALWAYS_SOFTWARE'] == '1'
    assert env['DISPLAY'] == ':98'
    assert env['SDL_AUDIODRIVER'] == 'dummy'
    with pytest.raises(ValueError):
        module.build_env(0, ':98')


def test_data_dir_refuses_reuse(tmp_path):
    template = tmp_path / 'template'
    template.mkdir()
    for name in ('th08.dat', 'thbgm.dat'):
        (template / name).write_bytes(b'x')
    for name in ('th08.cfg.orig', 'score.dat.orig'):
        (template / name).write_bytes(b'orig')
    target = tmp_path / 'data'
    module.create_data_dir(template, target)
    assert (target / 'th08.cfg').read_bytes() == b'orig'
    assert (target / 'score.dat').read_bytes() == b'orig'
    with pytest.raises(module.Blocked, match='refusing to reuse'):
        module.create_data_dir(template, target)


def test_data_dir_requires_orig_templates(tmp_path):
    template = tmp_path / 'template'
    template.mkdir()
    for name in ('th08.dat', 'thbgm.dat'):
        (template / name).write_bytes(b'x')
    with pytest.raises(module.Blocked, match='th08.cfg.orig'):
        module.create_data_dir(template, tmp_path / 'data')
