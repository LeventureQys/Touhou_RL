"""Contract tests for the WP5 BRANCH cost decomposition tool.

No real engine is started here: the parsing, the least-squares fit, the probe
sequencing and the fail-closed paths are exercised against a fake shadow that
speaks the same line protocol.
"""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/rl/v400_wp5_branch_cost.py'
spec = importlib.util.spec_from_file_location('v400_wp5_branch_cost', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def branch_reply(*, frames, advanced=None, child_us=None, fork_us=None, parent_frame=0,
                 pid=4242, hash_override=None, trace_override=None):
    """Build a real-shaped BRANCH reply line."""
    advanced = frames if advanced is None else advanced
    child = int(round(10 + 48 * advanced)) if child_us is None else child_us
    fork = child + 1100 if fork_us is None else fork_us
    hashes = [f'{0x1000 + index:016x}' for index in range(advanced)]
    trace = ''.join(hashes) if trace_override is None else trace_override
    final_hash = hashes[-1] if hashes else f'{0:016x}'
    if hash_override is not None:
        final_hash = hash_override
    return (f'OK branch=1 pid={pid} parent_frame={parent_frame} advanced={advanced} '
            f'fork_us={fork} child_us={child} hash={final_hash} trace={trace}')


class FakeShadowEngine:
    """Minimal duck-typed engine: raw reply lines, protocol-shaped fields."""

    def __init__(self, *, role='shadow', threads='1', short_advance=None,
                 advance_parent=False, child_offset=10.0, child_slope=48.0, fork_overhead=1100):
        self.role = role
        self.threads = threads
        self.short_advance = short_advance
        self.advance_parent = advance_parent
        self.child_offset = child_offset
        self.child_slope = child_slope
        self.fork_overhead = fork_overhead
        self.frame = 0
        self.cur_state = 1
        self.branch_calls = []
        self.commands = []
        self.log = []
        self.argv = ['fake-th08-modern', '--rl-role', 'shadow']

    def ask(self, command, timeout=None):
        self.commands.append(command)
        op = command.split()[0]
        if op == 'HELLO':
            return f'OK role={self.role}'
        if op == 'DEBUG':
            return (f'OK curState={self.cur_state} setupState=0 subthread=0 globals=1 stage=0 '
                    f'difficulty=1 shot=0 isReplay=0 isDemo=0 threads={self.threads} '
                    f'frames={self.frame}')
        if op == 'RESET':
            self.frame = 0
            self.cur_state = 2
            return ('OK reset=1 ready=1 setup_frames=35 frames=0 stage=0 difficulty=1 '
                    'team=0 seed=1 practice=0')
        if op == 'BRANCH':
            frames = int(command.split()[2])
            self.branch_calls.append(frames)
            advanced = frames
            if self.short_advance is not None and frames == self.short_advance:
                advanced = frames - 1
            child = int(round(self.child_offset + self.child_slope * frames))
            reply = branch_reply(frames=frames, advanced=advanced, child_us=child,
                                 fork_us=child + self.fork_overhead,
                                 parent_frame=self.frame, pid=4242 + len(self.branch_calls))
            if self.advance_parent:
                self.frame += 1
            return reply
        if op == 'QUIT':
            return 'OK bye=1'
        raise AssertionError(f'unexpected command {command!r}')

    def close(self):
        return 0


# ---------------------------------------------------------------- parsing ---
def test_parse_reply_splits_pairs():
    fields = runner.parse_reply('OK branch=1 advanced=300 hash=00ff')
    assert fields == {'_verb': 'OK', 'branch': '1', 'advanced': '300', 'hash': '00ff'}


def test_parse_branch_accepts_a_real_shaped_reply():
    row = runner.parse_branch(branch_reply(frames=4), 4)
    assert row['advanced'] == 4
    assert row['frames'] == 4
    assert row['child_us'] == 202
    assert row['fork_us'] == 1302
    assert row['trace_entry_count'] == 4


def test_parse_branch_rejects_short_advance():
    with pytest.raises(runner.Unavailable, match='advanced=9 but 10 frames'):
        runner.parse_branch(branch_reply(frames=10, advanced=9), 10)


def test_parse_branch_rejects_missing_timing_field():
    reply = 'OK branch=1 pid=1 parent_frame=0 advanced=2 fork_us=100 hash=00 trace=000000000000000a000000000000000b'
    with pytest.raises(runner.Unavailable, match='lacks a timing field'):
        runner.parse_branch(reply, 2)


def test_parse_branch_rejects_trace_length_mismatch():
    with pytest.raises(runner.Unavailable, match='hex chars'):
        runner.parse_branch(branch_reply(frames=3, trace_override='ab' * 8), 3)


def test_parse_branch_rejects_hash_trace_disagreement():
    with pytest.raises(runner.Unavailable, match='disagrees with the last trace entry'):
        runner.parse_branch(branch_reply(frames=2, hash_override='f' * 16), 2)


def test_parse_branch_rejects_err_reply():
    with pytest.raises(runner.Unavailable, match='rejected by engine'):
        runner.parse_branch('ERR branch-usage BRANCH <mask> <frames<=300>', 2)


def test_parse_branch_rejects_fork_us_below_child_us():
    with pytest.raises(runner.Unavailable, match='parent span must contain'):
        runner.parse_branch(branch_reply(frames=2, child_us=900, fork_us=800), 2)


# -------------------------------------------------------------- fitting ----
def test_fit_linear_recovers_an_exact_line():
    fit = runner.fit_linear([1.0, 5.0, 10.0], [58.0, 250.0, 490.0])
    assert fit['slope_us_per_frame'] == pytest.approx(48.0)
    assert fit['intercept_us'] == pytest.approx(10.0)
    assert fit['r2'] == pytest.approx(1.0)
    assert fit['n'] == 3


def test_fit_linear_reports_imperfect_fit_and_uncertainty():
    fit = runner.fit_linear([1.0, 5.0, 10.0, 50.0], [60.0, 250.0, 500.0, 2410.0])
    assert 0.99 < fit['r2'] < 1.0
    assert fit['slope_us_per_frame'] == pytest.approx(48.0, abs=0.5)
    assert fit['se_slope_us_per_frame'] > 0


@pytest.mark.parametrize('xs, ys, message', [
    ([1.0, 2.0], [1.0, 2.0], 'at least 3 points'),
    ([5.0, 5.0, 5.0], [1.0, 2.0, 3.0], 'no variance'),
    ([1.0, 2.0, 3.0], [1.0, 2.0], 'length mismatch'),
])
def test_fit_linear_rejects_degenerate_input(xs, ys, message):
    with pytest.raises(ValueError, match=message):
        runner.fit_linear(xs, ys)


# ---------------------------------------------------------------- probe ----
def test_run_probe_collects_every_horizon_and_never_steps_the_parent():
    engine = FakeShadowEngine()
    body = runner.run_probe(engine, horizons=(1, 5, 10), repeats=3, warmup=1, passes=1, order_seed=7)
    assert len(body['samples']) == 9
    assert body['warmup']['discarded_branches'] == 1
    assert body['warmup']['rows'][0]['child_us'] == int(round(10 + 48 * 10))
    assert all(command.split()[0] != 'STEP' for command in engine.commands)
    assert {sample['parent_frame'] for sample in body['samples']} == {0}
    assert body['protocol']['parent_stepped_between_branches'] is False
    assert all(len(order) == 3 and sorted(order) == [1, 5, 10] for order in body['protocol']['orders'])
    assert int(body['protocol']['reset_reply']['ready']) == 1


def test_run_probe_order_is_reproducible_and_not_a_fixed_sequence():
    first = runner.run_probe(FakeShadowEngine(), horizons=(1, 5, 10, 50, 100),
                             repeats=5, warmup=0, passes=2, order_seed=20261109)
    second = runner.run_probe(FakeShadowEngine(), horizons=(1, 5, 10, 50, 100),
                              repeats=5, warmup=0, passes=2, order_seed=20261109)
    assert first['protocol']['orders'] == second['protocol']['orders']
    assert len({tuple(order) for order in first['protocol']['orders']}) > 1


def test_run_probe_recovers_the_planted_slope_and_intercept():
    body = runner.run_probe(FakeShadowEngine(child_offset=10.0, child_slope=48.0,
                                             fork_overhead=1100),
                            horizons=runner.DEFAULT_HORIZONS, repeats=5, warmup=2, passes=1)
    fit = body['fit']['child_us']
    assert fit['slope_us_per_frame'] == pytest.approx(48.0)
    assert fit['intercept_us'] == pytest.approx(10.0)
    assert fit['r2'] == pytest.approx(1.0)
    assert fit['n'] == 35
    cross = body['cross_check']
    assert cross['fork_minus_child_us']['mean_us'] == pytest.approx(1100.0)
    assert cross['fork_minus_child_us_fit']['slope_us_per_frame'] == pytest.approx(0.0, abs=1e-9)
    assert cross['fork_minus_child_us_fit']['intercept_us'] == pytest.approx(1100.0)
    assert cross['fork_fit']['intercept_us'] == pytest.approx(1110.0)
    assert cross['fork_fit']['slope_us_per_frame'] == pytest.approx(48.0)
    interpretation = body['interpretation']
    assert interpretation['fixed_total_us'] == pytest.approx(1110.0)
    assert interpretation['fixed_total_us_fork_fit'] == pytest.approx(1110.0)
    assert interpretation['fixed_total_relative_gap'] == pytest.approx(0.0, abs=1e-9)


def test_ols_additivity_is_an_identity_not_evidence():
    """fit(fork) == fit(child) + fit(fork - child) must hold exactly, by algebra."""
    body = runner.run_probe(FakeShadowEngine(child_offset=7.0, child_slope=41.5,
                                             fork_overhead=640),
                            horizons=(1, 5, 10, 50, 300), repeats=4, warmup=0, passes=1)
    child = body['fit']['child_us']
    delta = body['cross_check']['fork_minus_child_us_fit']
    fork = body['fit']['fork_us']
    assert fork['slope_us_per_frame'] == pytest.approx(
        child['slope_us_per_frame'] + delta['slope_us_per_frame'])
    assert fork['intercept_us'] == pytest.approx(
        child['intercept_us'] + delta['intercept_us'])
    assert 'algebra, not' in body['cross_check']['ols_additivity_identity']


def test_per_horizon_rows_keep_every_sample_count():
    body = runner.run_probe(FakeShadowEngine(), horizons=(1, 5, 10), repeats=4, warmup=0, passes=1)
    assert [row['n'] for row in body['per_horizon']] == [4, 4, 4]
    assert [row['frames'] for row in body['per_horizon']] == [1, 5, 10]
    assert all(row['distinct_hashes'] == 1 for row in body['per_horizon'])


def test_independent_passes_report_a_run_to_run_spread():
    body = runner.run_probe(FakeShadowEngine(), horizons=(1, 5, 10), repeats=3, warmup=0, passes=4)
    assert len(body['samples']) == 36
    assert sorted({row['pass'] for row in body['samples']}) == [0, 1, 2, 3]
    summary = body['passes']
    assert summary['count'] == 4
    assert summary['samples_per_pass'] == 9
    assert len(summary['per_pass']) == 4
    # The fake engine is exactly linear, so every pass must reproduce the pooled fit.
    assert summary['slope_mean_us_per_frame'] == pytest.approx(48.0)
    assert summary['slope_std_us_per_frame'] == pytest.approx(0.0, abs=1e-9)
    assert summary['intercept_mean_us'] == pytest.approx(10.0)
    assert summary['r2_min'] == pytest.approx(1.0)
    assert body['interpretation']['run_to_run_slope_relative_spread_percent'] == \
        pytest.approx(0.0, abs=1e-9)
    assert 'run-to-run' in summary['note']


def test_run_probe_blocks_on_wrong_role():
    with pytest.raises(runner.Unavailable, match='shadow role is mandatory'):
        runner.run_probe(FakeShadowEngine(role='env'), horizons=(1, 5, 10), repeats=3, warmup=0)


def test_run_probe_blocks_on_multithreaded_engine():
    with pytest.raises(runner.Unavailable, match='single-threaded shadow'):
        runner.run_probe(FakeShadowEngine(threads='2'), horizons=(1, 5, 10), repeats=3, warmup=0)


def test_run_probe_blocks_on_incomplete_branch():
    with pytest.raises(runner.Unavailable, match='advanced=9 but 10 frames'):
        runner.run_probe(FakeShadowEngine(short_advance=10), horizons=(1, 5, 10), repeats=3, warmup=0)


def test_run_probe_blocks_when_the_parent_advanced():
    with pytest.raises(runner.Unavailable, match='parent advanced between branches'):
        runner.run_probe(FakeShadowEngine(advance_parent=True), horizons=(1, 5, 10),
                         repeats=3, warmup=0)


@pytest.mark.parametrize('kwargs, message', [
    ({'horizons': (0, 5, 10)}, 'horizons must lie in 1..300'),
    ({'horizons': (1, 301)}, 'horizons must lie in 1..300'),
    ({'horizons': (1, 1, 5)}, 'horizons must be unique'),
    ({'horizons': (1, 5, 10), 'repeats': 2}, 'at least 3 repeats'),
    ({'horizons': (1, 5, 10), 'passes': 0}, 'at least 1 pass'),
])
def test_run_probe_rejects_impossible_designs(kwargs, message):
    design = {'horizons': (1, 5, 10), 'repeats': 3, 'warmup': 0, 'passes': 1}
    design.update(kwargs)
    with pytest.raises(runner.Unavailable, match=message):
        runner.run_probe(FakeShadowEngine(), **design)


# --------------------------------------------------------------- report ----
def test_build_report_carries_provenance_and_caveats(tmp_path):
    body = runner.run_probe(FakeShadowEngine(), horizons=(1, 5, 10), repeats=3, warmup=0, passes=1)
    report = runner.build_report(
        body, binary=tmp_path / 'th08-modern', data_dir=tmp_path,
        binary_sha256='deadbeef', engine_exit_code=0, host={'platform': 'linux'},
    )
    assert report['schema'] == runner.SCHEMA
    assert report['status'] == 'measured'
    assert report['tool_exit_code'] == 0
    assert report['engine']['role'] == 'shadow'
    assert report['engine']['binary_sha256'] == 'deadbeef'
    assert len(report['samples']) == 9
    assert len(report['caveats']) >= 5
    text = ' '.join(report['caveats']).lower()
    assert 'regression' in text and 'not the fork() system call' in text
    assert 'fork_us - child_us' in text


def test_build_report_marks_a_nonzero_engine_exit_as_unusable(tmp_path):
    body = runner.run_probe(FakeShadowEngine(), horizons=(1, 5, 10), repeats=3, warmup=0, passes=1)
    report = runner.build_report(body, binary=tmp_path / 'b', data_dir=tmp_path,
                                 binary_sha256=None, engine_exit_code=7)
    assert report['status'] == 'unavailable'
    assert report['tool_exit_code'] == 3
    assert 'exited 7' in report['reason']


def test_build_report_pins_the_running_image_and_flags_a_rebuild(tmp_path):
    body = runner.run_probe(FakeShadowEngine(), horizons=(1, 5, 10), repeats=3, warmup=0, passes=1)
    clean = runner.build_report(body, binary=tmp_path / 'b', data_dir=tmp_path,
                                binary_sha256='aaa', binary_sha256_after='aaa',
                                image_sha256='aaa', image_path='/proc/1/exe',
                                engine_exit_code=0)
    assert clean['engine']['running_image_matches_binary'] is True
    assert clean['engine']['binary_changed_during_run'] is False

    rebuilt = runner.build_report(body, binary=tmp_path / 'b', data_dir=tmp_path,
                                  binary_sha256='aaa', binary_sha256_after='bbb',
                                  image_sha256='ccc', image_path='/proc/1/exe (deleted)',
                                  engine_exit_code=0)
    assert rebuilt['engine']['running_image_matches_binary'] is False
    assert rebuilt['engine']['binary_changed_during_run'] is True
    assert rebuilt['engine']['running_image_sha256'] == 'ccc'


def test_running_image_hash_is_none_for_a_missing_process():
    assert runner.running_image_hash(999_999_999) is None
    assert runner.running_image_path(999_999_999) is None


def test_main_is_fail_closed_without_a_real_binary(tmp_path, capsys):
    out = tmp_path / 'report.json'
    code = runner.main(['--binary', str(tmp_path / 'absent'), '--data-dir', str(tmp_path),
                        '--out', str(out)])
    assert code == 3
    assert not out.exists()
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload['status'] == 'unavailable'
    assert 'binary not found' in payload['reason']


def test_main_rejects_unparseable_horizons(tmp_path, capsys):
    code = runner.main(['--binary', str(tmp_path), '--data-dir', str(tmp_path),
                        '--out', str(tmp_path / 'r.json'), '--horizons', '1,x,10'])
    assert code == 3
    assert 'comma separated integers' in capsys.readouterr().out


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    target = tmp_path / 'blob.bin'
    target.write_bytes(b'wp5-branch-cost')
    assert runner.sha256_file(target) == hashlib.sha256(b'wp5-branch-cost').hexdigest()
