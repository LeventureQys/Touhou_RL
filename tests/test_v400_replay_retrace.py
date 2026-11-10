"""Fail-closed unit checks for the WP1 differential replay retrace tool.

These checks never count synthetic row streams as real WP1 evidence: the real
differential run is recorded in EXP-P0-051.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'rl' / 'v400_replay_retrace.py'
spec = importlib.util.spec_from_file_location('v400_replay_retrace', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Byte-for-byte the upstream muted playback profile
# (third_party/th08/scripts/capture-replay.py -> muted_config()).
MUTED_CONFIG_HEX = '0000010002000400ffffffffffffffff0300000001000800580258020203000000010100020000000000000000000000000000000000000000000000'


def zero_row(**overrides):
    """One 25-field little-endian row, with named fields overridden."""
    values = [0] * module.ROW_FIELD_COUNT
    values[0] = 1
    for name, value in overrides.items():
        values[module.ROW_FIELDS.index(name)] = value
    return module.ROW_STRUCT.pack(*values)


def test_row_layout_matches_recorder_contract():
    assert module.ROW_FIELD_COUNT == 25
    assert len(module.ROW_FIELDS) == 25
    assert len(set(module.ROW_FIELDS)) == 25
    assert module.ROW_STRUCT.size == 100


def test_muted_config_is_the_upstream_sixty_byte_profile():
    assert module.muted_config() == bytes.fromhex(MUTED_CONFIG_HEX)
    assert len(module.muted_config()) == 60


def test_identical_streams_report_no_difference():
    data = zero_row() + zero_row(score=7)
    assert module.first_difference(data, data) is None


def test_first_difference_reports_first_row_and_named_fields():
    left = zero_row(score=1) + zero_row(score=2) + zero_row(score=3)
    right = zero_row(score=1) + zero_row(score=9) + zero_row(score=3)
    difference = module.first_difference(left, right)
    assert difference['kind'] == 'row'
    assert difference['rowIndex'] == 1
    assert difference['rowNumber'] == 2
    assert difference['differingFields'] == ['score']
    assert difference['on']['score'] == 2
    assert difference['off']['score'] == 9


def test_first_difference_reports_length_difference():
    left = zero_row() + zero_row()
    right = left + zero_row()
    difference = module.first_difference(left, right)
    assert difference['kind'] == 'length'
    assert difference['onRows'] == 2
    assert difference['offRows'] == 3


def test_first_difference_reports_trailing_bytes():
    difference = module.first_difference(zero_row(), zero_row() + b'\x01\x02')
    assert difference['kind'] == 'length'
    assert difference['rowsCompared'] == 1


def test_compare_rows_files_identical(tmp_path):
    rows = tmp_path / 'rows.bin'
    rows.write_bytes(zero_row() * 3)
    result = module.compare_rows_files(rows, rows)
    assert result['identical'] is True
    assert result['onRows'] == result['offRows'] == 3
    assert result['onSha256'] == result['offSha256'] == hashlib.sha256(rows.read_bytes()).hexdigest()
    assert result['firstDifference'] is None


def test_compare_rows_files_detects_difference(tmp_path):
    on, off = tmp_path / 'on.bin', tmp_path / 'off.bin'
    on.write_bytes(zero_row(score=1) + zero_row(score=2))
    off.write_bytes(zero_row(score=1) + zero_row(score=3))
    result = module.compare_rows_files(on, off)
    assert result['identical'] is False
    assert result['firstDifference']['rowIndex'] == 1
    assert result['firstDifference']['differingFields'] == ['score']


def test_compare_rows_files_missing_side_is_not_identical(tmp_path):
    on = tmp_path / 'on.bin'
    on.write_bytes(zero_row())
    result = module.compare_rows_files(on, tmp_path / 'missing.bin')
    assert result['identical'] is False
    assert result['firstDifference']['kind'] == 'missing'


def test_complete_problems_accepts_natural_record_only_exit(tmp_path):
    rows = tmp_path / 'rows.bin'
    rows.write_bytes(zero_row() * 2)
    complete = {'rows': 2, 'ended': True, 'diagnosticComplete': False,
                'naturalReplayExit': True, 'errors': [], 'stages': [0]}
    assert module.complete_problems(complete, rows) == []


def test_complete_problems_rejects_unnatural_exit(tmp_path):
    rows = tmp_path / 'rows.bin'
    rows.write_bytes(zero_row())
    complete = {'rows': 1, 'ended': False, 'naturalReplayExit': False, 'errors': []}
    problems = module.complete_problems(complete, rows)
    assert any('ended' in problem for problem in problems)
    assert any('naturalReplayExit' in problem for problem in problems)


def test_complete_problems_rejects_recorder_errors(tmp_path):
    rows = tmp_path / 'rows.bin'
    rows.write_bytes(zero_row())
    complete = {'rows': 1, 'ended': True, 'naturalReplayExit': True,
                'errors': ['Replay diverged; inspect first-difference.json']}
    assert module.complete_problems(complete, rows) == [
        "complete.json errors = ['Replay diverged; inspect first-difference.json']"
    ]


def test_complete_problems_rejects_truncated_rows_file(tmp_path):
    rows = tmp_path / 'rows.bin'
    rows.write_bytes(zero_row() * 2)
    complete = {'rows': 3, 'ended': True, 'naturalReplayExit': True, 'errors': []}
    problems = module.complete_problems(complete, rows)
    assert any('rows file size' in problem for problem in problems)


def test_complete_problems_rejects_zero_or_boolean_rows(tmp_path):
    rows = tmp_path / 'rows.bin'
    rows.write_bytes(b'')
    for value in (0, True):
        problems = module.complete_problems(
            {'rows': value, 'ended': True, 'naturalReplayExit': True, 'errors': []}, rows)
        assert any('positive integer' in problem for problem in problems)


def test_complete_problems_rejects_missing_rows_file(tmp_path):
    complete = {'rows': 1, 'ended': True, 'naturalReplayExit': True, 'errors': []}
    problems = module.complete_problems(complete, tmp_path / 'missing.bin')
    assert 'rows file missing' in problems


def test_capture_environment_enables_record_only_and_drops_reference(tmp_path):
    base = {
        'TH08_REPLAY_TRACE_REFERENCE': '/tmp/reference.bin',
        'TH08_REPLAY_TRACE_MAXIMUM': '5',
        'TH08_REPLAY_TRACE_ROWS': '/tmp/old-rows.bin',
        'KEEP': '1',
    }
    output = tmp_path / 'out'
    environment = module.capture_environment(base, tmp_path, output, 8)
    assert environment['TH08_REPLAY_TRACE_RECORD_ONLY'] == '1'
    assert 'TH08_REPLAY_TRACE_REFERENCE' not in environment
    assert 'TH08_REPLAY_TRACE_MAXIMUM' not in environment
    assert environment['TH08_REPLAY_TRACE_ROWS'] == str(tmp_path / 'rows.bin')
    assert environment['TH08_REPLAY_TRACE_COMPLETION'] == str(output / 'complete.json')
    assert environment['TH08_AUTOPLAY_REPLAY'] == './replay/audit.rpy'
    assert environment['TH08_AUTOPLAY_STAGE'] == '8'
    assert environment['SDL_AUDIODRIVER'] == 'dummy'
    assert environment['KEEP'] == '1'


def test_capture_command_is_bounded_and_uses_data_dir(tmp_path):
    command = module.capture_command(tmp_path / 'game', tmp_path / 'runtime', 180)
    assert command[0] == 'xvfb-run'
    assert '180s' in command
    assert command[-2:] == ['--data-dir', str(tmp_path / 'runtime')]
    assert str(tmp_path / 'game') in command


def test_run_capture_refuses_existing_directory(tmp_path):
    existing = tmp_path / 'exists'
    existing.mkdir()
    with pytest.raises(FileExistsError):
        module.run_capture(tmp_path / 'exe', tmp_path / 'replay.rpy', 0, tmp_path, existing, 180)


def test_inspect_build_flags_missing_executable(tmp_path):
    result = module.inspect_build(tmp_path / 'missing')
    assert result['problems'] == ['executable is missing']
    assert result['sha256'] is None


def test_inspect_build_flags_missing_recorder_symbol(tmp_path, monkeypatch):
    fake = tmp_path / 'th08-modern'
    fake.write_bytes(b'not-an-elf')

    class Result:
        stdout = ''

    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: Result())
    result = module.inspect_build(fake)
    assert result['recordSymbolPresent'] is False
    assert any('ReplayTraceRecord' in problem for problem in result['problems'])


def test_inspect_build_accepts_recorder_symbol(tmp_path, monkeypatch):
    fake = tmp_path / 'th08-modern'
    fake.write_bytes(b'not-an-elf')

    class Result:
        stdout = '0000000000000000 T th08::modern::ReplayTraceRecord(th08::ReplayManager*)\n'

    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: Result())
    result = module.inspect_build(fake)
    assert result['problems'] == []
    assert result['recordSymbolPresent'] is True


def test_cli_rejects_timeout_above_the_bound(tmp_path):
    with pytest.raises(SystemExit) as error:
        module.main(['--on-build', 'on', '--off-build', 'off', '--replay', 'replay.rpy',
                     '--output-dir', str(tmp_path / 'out'), '--timeout', '301'])
    assert error.value.code == 2


def test_cli_rejects_existing_output_dir(tmp_path):
    existing = tmp_path / 'exists'
    existing.mkdir()
    with pytest.raises(SystemExit) as error:
        module.main(['--on-build', 'on', '--off-build', 'off', '--replay', 'replay.rpy',
                     '--output-dir', str(existing), '--timeout', '60'])
    assert error.value.code == 2


def test_report_declares_differential_proof_strength():
    assert module.PROOF_STRENGTH.startswith('differential')
    assert 'not a bit-exact comparison against the retail executable' in module.PROOF_STRENGTH


# ---------------------------------------------------------------------------
# Sharded / resumable corpus mode (EXP-P0-055)
# ---------------------------------------------------------------------------

def test_default_bounds_are_three_hundred_seconds_and_one_retry():
    assert module.DEFAULT_TIMEOUT_SECONDS == 300
    assert module.MAXIMUM_TIMEOUT_SECONDS == 300
    assert module.DEFAULT_MAXIMUM_RETRIES == 1
    assert module.DEFAULT_SHARDS == 4
    assert module.DEFAULT_EXPECTED_TOTAL == 108


def test_compare_completion_files_reports_identical_bytes(tmp_path):
    on, off = tmp_path / 'on.json', tmp_path / 'off.json'
    on.write_text('{"rows": 2, "ended": true}')
    off.write_text('{"rows": 2, "ended": true}')
    result = module.compare_completion_files(on, off)
    assert result['identical'] is True
    assert result['onSha256'] == result['offSha256']
    assert result['firstDifference'] is None


def test_compare_completion_files_names_the_first_differing_key(tmp_path):
    on, off = tmp_path / 'on.json', tmp_path / 'off.json'
    on.write_text('{"rows": 2, "endScores": [1, 2], "ended": true}')
    off.write_text('{"rows": 2, "endScores": [1, 3], "ended": true}')
    result = module.compare_completion_files(on, off)
    assert result['identical'] is False
    assert result['firstDifference'] == {'kind': 'key', 'key': 'endScores',
                                         'on': [1, 2], 'off': [1, 3]}


def test_compare_completion_files_missing_side_is_not_identical(tmp_path):
    on = tmp_path / 'on.json'
    on.write_text('{}')
    result = module.compare_completion_files(on, tmp_path / 'missing.json')
    assert result['identical'] is False
    assert result['firstDifference']['kind'] == 'missing'


def test_side_status_classifies_timeout_before_failure():
    assert module.side_status({'timedOut': True, 'exitCode': 0, 'problems': []}) == 'timeout'
    assert module.side_status({'timedOut': False, 'exitCode': 124, 'problems': []}) == 'timeout'
    assert module.side_status({'timedOut': False, 'exitCode': 0, 'problems': ['x']}) == 'failed'
    assert module.side_status({'timedOut': False, 'exitCode': 0, 'problems': []}) == 'clean'


def test_case_status_is_fail_closed():
    clean = {'status': 'clean'}
    timeout = {'status': 'timeout'}
    failed = {'status': 'failed'}
    same = {'identical': True}
    differ = {'identical': False}
    assert module.case_status(clean, clean, same, same) == 'identical'
    assert module.case_status(clean, clean, differ, same) == 'different'
    assert module.case_status(clean, clean, same, differ) == 'different'
    assert module.case_status(timeout, clean, same, same) == 'timeout'
    assert module.case_status(clean, timeout, same, same) == 'timeout'
    assert module.case_status(failed, clean, same, same) == 'failed'
    assert module.case_status(clean, failed, same, same) == 'failed'


def _fake_corpus(tmp_path, count=3, mutate=None):
    fixtures_dir = tmp_path / 'accepted'
    fixtures_dir.mkdir()
    fixtures = []
    for index in range(count):
        name = f'th8_ud{index:04x}.rpy'
        path = fixtures_dir / name
        path.write_bytes(f'replay-{index}'.encode())
        fixtures.append({
            'id': f'case-{index}',
            'file': name,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size,
            'difficultyName': 'Easy',
            'shotName': 'Border Team',
            'stages': [{'index': 0, 'inputRecords': 10}],
            'retailTrace': {'calculationFrames': 1000 * (index + 1)},
        })
    if mutate is not None:
        mutate(fixtures, fixtures_dir)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'version': 1, 'fixtures': fixtures}))
    return fixtures_dir, manifest


def test_select_cases_verifies_the_manifest(tmp_path):
    fixtures_dir, manifest = _fake_corpus(tmp_path)
    cases = module.select_cases(fixtures_dir, manifest)
    assert len(cases) == 3
    assert cases[0]['case'] == 'th8_ud0000'
    assert cases[0]['fixtureId'] == 'case-0'
    assert cases[0]['manifestFrames'] == 1000
    assert cases[0]['sha256'] == cases[0]['manifestSha256']


def test_select_cases_rejects_missing_fixture(tmp_path):
    def drop(fixtures, fixtures_dir):
        fixtures.pop()
    fixtures_dir, manifest = _fake_corpus(tmp_path, mutate=drop)
    with pytest.raises(module.CorpusError):
        module.select_cases(fixtures_dir, manifest)


def test_select_cases_rejects_unmanifested_file(tmp_path):
    def add(fixtures, fixtures_dir):
        (fixtures_dir / 'th8_stranger.rpy').write_bytes(b'stranger')
    fixtures_dir, manifest = _fake_corpus(tmp_path, mutate=add)
    with pytest.raises(module.CorpusError):
        module.select_cases(fixtures_dir, manifest)


def test_select_cases_rejects_sha_mismatch(tmp_path):
    def corrupt(fixtures, fixtures_dir):
        fixtures[0]['sha256'] = '0' * 64
    fixtures_dir, manifest = _fake_corpus(tmp_path, mutate=corrupt)
    with pytest.raises(module.CorpusError):
        module.select_cases(fixtures_dir, manifest)


def test_select_cases_honours_only_filter(tmp_path):
    fixtures_dir, manifest = _fake_corpus(tmp_path)
    cases = module.select_cases(fixtures_dir, manifest, only='case-1')
    assert [case['case'] for case in cases] == ['th8_ud0001']
    with pytest.raises(module.CorpusError):
        module.select_cases(fixtures_dir, manifest, only='case-unknown')


def _fake_cases(count=108):
    return [{'case': f'case-{index:03d}', 'fixtureId': f'id-{index:03d}',
             'manifestFrames': 40_000 + index * 100} for index in range(count)]


def test_assign_shards_covers_every_case_once_in_four_shards_of_twenty_seven():
    cases = _fake_cases()
    shards = 4
    slices = [module.assign_shards(cases, shards, shard) for shard in range(1, shards + 1)]
    assert [len(entry) for entry in slices] == [27, 27, 27, 27]
    flat = [case['case'] for entry in slices for case in entry]
    assert sorted(flat) == sorted(case['case'] for case in cases)
    assert len(set(flat)) == len(flat)


def test_assign_shards_balances_predicted_frames():
    cases = _fake_cases()
    loads = [sum(case['manifestFrames'] for case in module.assign_shards(cases, 4, shard))
             for shard in range(1, 5)]
    assert max(loads) - min(loads) <= max(case['manifestFrames'] for case in cases)


def test_assign_shards_rejects_out_of_range_shard():
    with pytest.raises(ValueError):
        module.assign_shards(_fake_cases(), 4, 5)


def test_parse_shard_accepts_and_rejects():
    assert module.parse_shard('3/4') == (3, 4)
    with pytest.raises(module.argparse.ArgumentTypeError):
        module.parse_shard('5/4')
    with pytest.raises(module.argparse.ArgumentTypeError):
        module.parse_shard('nonsense')


def test_journal_round_trip_is_append_only_and_last_wins(tmp_path):
    journal = tmp_path / 'shard-1.jsonl'
    module.append_jsonl(journal, {'case': 'a', 'status': 'different'})
    module.append_jsonl(journal, {'case': 'b', 'status': 'identical'})
    module.append_jsonl(journal, {'case': 'a', 'status': 'identical'})
    records, corrupt = module.read_journal(journal)
    assert records['a']['status'] == 'identical'
    assert records['b']['status'] == 'identical'
    assert corrupt == 0
    assert len(journal.read_text().splitlines()) == 3


def test_read_journal_counts_corrupt_lines_and_tolerates_them(tmp_path):
    journal = tmp_path / 'shard-2.jsonl'
    journal.write_text('{"case": "a", "status": "identical"}\nnot-json\n')
    records, corrupt = module.read_journal(journal)
    assert list(records) == ['a']
    assert corrupt == 1


def test_run_side_retries_into_a_fresh_attempt_directory(tmp_path, monkeypatch):
    calls = []

    def fake_run_capture(executable, replay, stage, game_data, case_dir, timeout):
        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        calls.append(case_dir.name)
        clean = len(calls) >= 2
        return {
            'executable': str(executable), 'executableSha256': 'x', 'exitCode': 0 if clean else 1,
            'timedOut': False, 'seconds': 0.1, 'rowsPath': str(case_dir / 'runtime/rows.bin'),
            'rowsPresent': True, 'rowsBytes': 200, 'rowsSha256': 'y',
            'completePath': str(case_dir / 'complete.json'), 'completeSha256': 'z',
            'complete': {'rows': 2, 'ended': True, 'naturalReplayExit': True, 'errors': []},
            'runtimeScoreSha256': None, 'runtimeConfigSha256': None,
            'problems': [] if clean else ['process exit code 1'],
        }

    monkeypatch.setattr(module, 'run_capture', fake_run_capture)
    side_dir = tmp_path / 'on'
    first = module.run_side(tmp_path / 'exe', tmp_path / 'r.rpy', 0, tmp_path, side_dir, 300, 1)
    assert calls == ['attempt-1', 'attempt-2']
    assert first['attempts'] == 2
    assert first['status'] == 'clean'
    assert first['rows'] == 2
    assert first['totalSeconds'] == pytest.approx(0.2)
    assert first['attemptLog'][0]['problems'] == ['process exit code 1']
    (side_dir / 'side.json').exists()

    second = module.run_side(tmp_path / 'exe', tmp_path / 'r.rpy', 0, tmp_path, side_dir, 300, 1)
    assert calls[2:] == ['attempt-3']
    assert second['attempts'] == 1


def test_merge_report_counts_and_marks_partial(tmp_path):
    output = tmp_path / 'out'
    output.mkdir()
    module.append_jsonl(output / 'shard-1.jsonl', {'case': 'a', 'status': 'identical',
                                                   'on': {'rows': 10}, 'replay': {'manifestOrder': 0}})
    module.append_jsonl(output / 'shard-1.jsonl', {'case': 'b', 'status': 'different',
                                                   'on': {'rows': 20}, 'replay': {'manifestOrder': 1},
                                                   'comparison': {'firstDifference': {'kind': 'row'}}})
    module.append_jsonl(output / 'shard-2.jsonl', {'case': 'c', 'status': 'timeout',
                                                   'on': {'rows': 0}, 'replay': {'manifestOrder': 2}})
    module.append_jsonl(output / 'shard-2.jsonl', {'case': 'd', 'status': 'failed',
                                                   'on': {'rows': 0}, 'replay': {'manifestOrder': 3}})

    report = module.merge_report(output, 4)
    assert report['total'] == 4
    assert report['recorded'] == 4
    assert report['compared'] == 2
    assert report['identical'] == 1
    assert report['different'] == 1
    assert report['timeout'] == 1
    assert report['failed'] == 1
    assert report['partial'] is False
    assert report['allIdentical'] is False
    assert report['totalRowsPerSide'] == 30
    assert [case['case'] for case in report['cases']] == ['a', 'b', 'c', 'd']
    assert report['differences'][0]['case'] == 'b'
    assert {entry['case'] for entry in report['notIdentical']} == {'c', 'd'}
    assert (output / 'report.json').is_file()

    partial = module.merge_report(output, 6)
    assert partial['partial'] is True
    assert partial['remaining'] == 2


def test_merge_report_is_all_identical_only_when_every_case_is(tmp_path):
    output = tmp_path / 'out'
    output.mkdir()
    module.append_jsonl(output / 'shard-1.jsonl', {'case': 'a', 'status': 'identical', 'on': {'rows': 1}})
    report = module.merge_report(output, 1)
    assert report['allIdentical'] is True
    assert report['partial'] is False


def test_cli_requires_a_mode_and_ties_shard_to_all(tmp_path):
    base = ['--on-build', 'on', '--off-build', 'off', '--output-dir', str(tmp_path / 'out')]
    with pytest.raises(SystemExit) as error:
        module.main(base)
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        module.main(base + ['--shard', '1/4'])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        module.main(base + ['--all', '--replay', 'x.rpy'])
    assert error.value.code == 2


def test_cli_resume_allows_existing_output_dir(tmp_path):
    output = tmp_path / 'out'
    output.mkdir()
    game_data = tmp_path / 'game'
    game_data.mkdir()
    for name in ('th08.dat', 'thbgm.dat'):
        (game_data / name).write_bytes(b'x')
    helper = tmp_path / 'replay_file.py'
    helper.write_text('def inspect(path, require_retail=True):\n    return {}\n')
    try:
        code = module.main(['--on-build', 'on', '--off-build', 'off', '--all', '--resume',
                            '--fixtures-dir', str(tmp_path / 'missing'),
                            '--manifest', str(tmp_path / 'missing.json'),
                            '--game-data-dir', str(game_data), '--replay-helper', str(helper),
                            '--output-dir', str(output)])
    except SystemExit:
        pytest.fail('--resume must not reject an existing output directory')
    # Fake build paths then fail the fail-closed build inspection (exit 2).
    assert code == 2

