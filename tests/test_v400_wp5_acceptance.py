"""Fail-closed contract tests for the real WP5 acceptance runner."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/rl/v400_wp5_acceptance.py'
spec = importlib.util.spec_from_file_location('v400_wp5_acceptance', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FakeEngine:
    def __init__(self, role, *, mismatch=False):
        self.role, self.frame, self.mismatch = role, 0, mismatch

    def ask(self, command):
        op = command.split()[0]
        if op == 'HELLO':
            return {'role': self.role}
        if op == 'DEBUG':
            return {'threads': '1'}
        if op == 'RESET':
            self.frame = 0
            return {'ready': '1'}
        if op == 'STEP':
            n = int(command.split()[2])
            self.frame += n
            return {'advanced': str(n)}
        if op == 'HASH':
            return {'hash': str(self.frame + (self.mismatch and self.frame >= 60)), 'frames': str(self.frame)}
        if op == 'FRAME':
            return {'frame': str(self.frame)}
        if op == 'BRANCH':
            return {'branch': '1', 'advanced': '300', 'pid': '123', 'fork_us': '900', 'child_us': '600'}
        raise AssertionError(command)


def test_shadow_requires_all_10000_frames_and_all_167_checkpoints():
    result = runner.shadow_10000(FakeEngine('shadow'), FakeEngine('env'))
    assert result['frames'] == 10000
    assert result['checkpoints'] == 167
    assert result['hashes'][-1]['frame'] == 10000


def test_shadow_mismatch_blocks_at_first_checkpoint():
    with pytest.raises(runner.Blocked, match='divergence at 60'):
        runner.shadow_10000(FakeEngine('shadow'), FakeEngine('env', mismatch=True))


def test_branch_timing_covers_twenty_real_branch_replies():
    result = runner.branch_timing(FakeEngine('shadow'))
    assert result['samples'] == 20
    assert result['rows'][-1]['parent_frame'] == '19'
    assert result['mean_child_us_per_frame'] == 2
    assert 'not isolated' in result['timing_caveat']


def test_missing_original_pixels_never_pass(tmp_path):
    with pytest.raises(runner.Blocked, match='original pixel cache'):
        runner.pixel_gate(None, None, 'abc')
    cache = tmp_path / 'cache.bin'
    cache.write_bytes(b'not a genuine source')
    trajectory = tmp_path / 'trace.json'
    trajectory.write_text('{"harness_commit":"abc","source_verified":true}', encoding='utf-8')
    with pytest.raises(runner.Blocked, match='not implemented'):
        runner.pixel_gate(cache, trajectory, 'abc')
