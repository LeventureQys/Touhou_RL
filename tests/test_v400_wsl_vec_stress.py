import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.rl import v400_wsl_vec_stress as stress

BINARY = "/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern"
ROOT = "/tmp/v400_wp4_diag_test"


def _proc(pid, *, argv0=BINARY, data_dir=None, state="S (sleeping)", threads=1,
          rss=1000, snippet=None):
    return {"pid": pid, "ppid": 1, "state": state, "threads": threads,
            "vm_rss_kb": rss, "vm_hwm_kb": rss, "starttime_ticks": pid * 10,
            "argv0": argv0, "argv_count": 7, "data_dir": data_dir,
            "cmdline_snippet": snippet if snippet is not None else
            f"{argv0} --data-dir {data_dir} --rl-control 0,1 --rl-role shadow"}


def _dump(processes, *, probe_pid=999):
    return {"ok": True, "probe_pid": probe_pid, "probe_argv0": "-", "engine": BINARY,
            "proc_scanned": len(processes) + 1, "proc_skipped": 0, "proc_truncated": 0,
            "engine_argv0_pids": [p["pid"] for p in processes if p["argv0"] == BINARY],
            "zombie_pids": [p["pid"] for p in processes
                            if (p.get("state") or "").startswith("Z")],
            "processes": processes}


def _full_dump(*, base=0, probe_pid=999, instances=16):
    return _dump([_proc(base + index, data_dir=f"{ROOT}/i-{index}")
                  for index in range(instances)], probe_pid=probe_pid)


class _FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_prepare_instance_dirs_constructs_safe_probe_and_captures_output(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "PREPARE_RESOLVED=/tmp/v400_wp4_diag_probe\\n", "")

    monkeypatch.setattr(stress.subprocess, "run", fake_run)
    stress._prepare_instance_dirs("/root/v400-data/game", "Ubuntu", "/tmp/v400_wp4_diag_probe", 1)
    argv, kwargs = calls[0]
    script = argv[argv.index("-lc") + 1]
    assert "realpath -e" in script
    assert "PREPARE_RESOLVED=" in script
    assert "find /root/v400-data/game" in script
    assert "ln -s" in script
    assert "cp -p" in script
    assert "-lt 1" in script
    assert kwargs["capture_output"] is True


def test_prepare_instance_dirs_reports_stdout_stderr_and_exit(monkeypatch):
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, "partial output", "real failure")

    monkeypatch.setattr(stress.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="exit=1") as exc:
        stress._prepare_instance_dirs("/root/v400-data/game", "Ubuntu", "/tmp/v400_wp4_diag_probe", 1)
    message = str(exc.value)
    assert "partial output" in message
    assert "real failure" in message


def test_exp032_long_stress_evidence_fails_closed():
    """EXP032 remains a negative control: its incomplete run must not pass."""
    p = Path('.tmp_v400_wp4_long.json')
    assert p.is_file()
    d = json.loads(p.read_text(encoding='utf-8'))
    assert d['instances'] == 16
    assert d['continuous'] is True
    assert d['requested_seconds'] == 600.0
    assert d['max_seconds'] == 600
    assert not d['all_instances_reached_deadline']
    assert not d['all_exit_zero']
    assert not d['all_error_free']
    assert not d['no_wsl_processes']
    assert len(d['results']) == 16
    assert all(r['error'] and r['exit_code'] != 0 and r['run_seconds'] < 600
               for r in d['results'])


def test_diag_039_600_stress_success_gate():
    p = Path('.tmp_v400_wp4_diag_039_600.json')
    assert p.is_file()
    d = json.loads(p.read_text(encoding='utf-8'))
    assert d['provider'] == 'DeepSeek'
    assert d['model'] == 'DeepSeek v4.1 flash'
    assert d['instances'] == 16
    assert d['continuous'] is True
    assert d['requested_seconds'] == 600.0
    assert d['max_seconds'] == 600
    assert d['all_instances_reached_deadline']
    assert d['all_exit_zero'] and d['all_error_free'] and d['no_wsl_processes']
    assert d['cleanup_exit_code'] == 0
    assert d['remaining_wsl_process_lines'] == []
    assert len(d['results']) == 16
    assert all(r['pid'] and r['run_seconds'] >= 600
               and r['frames'] > 0
               and isinstance(r['stderr_tail'], str)
               and len(r['stderr_tail'].encode('utf-8')) <= 16 * 1024
               and not r['error'] and r['exit_code'] == 0
               for r in d['results'])


def test_linux_probe_source_is_valid_python_and_never_shells_out():
    compile(stress._LINUX_PROBE_SOURCE, "<linux-probe>", "exec")
    for forbidden in ("subprocess", "pgrep", "grep", "awk", "ps -", "os.popen"):
        assert forbidden not in stress._LINUX_PROBE_SOURCE
    assert "/proc" in stress._LINUX_PROBE_SOURCE
    assert 'raw.split(b"\\x00")' in stress._LINUX_PROBE_SOURCE
    assert "--data-dir" in stress._LINUX_PROBE_SOURCE


def test_linux_probe_is_fed_on_stdin_not_as_a_shell_command(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, json.dumps(_full_dump()), "")

    monkeypatch.setattr(stress.subprocess, "run", fake_run)
    dump = stress._linux_proc_dump("Ubuntu", BINARY)
    argv, kwargs = calls[0]
    assert argv == ["wsl.exe", "-d", "Ubuntu", "--", "python3", "-", BINARY]
    assert kwargs["input"] == stress._LINUX_PROBE_SOURCE
    assert kwargs["capture_output"] is True and kwargs["text"] is True
    assert "bash" not in argv and "-lc" not in argv
    assert dump["ok"] is True
    assert dump["proc_scanned"] == 17


@pytest.mark.parametrize("completed", [
    subprocess.CompletedProcess(["wsl.exe"], 2, "", "python3: not found"),
    subprocess.CompletedProcess(["wsl.exe"], 0, "not json", ""),
    subprocess.CompletedProcess(["wsl.exe"], 0, json.dumps({"ok": True}), ""),
])
def test_linux_probe_reports_failure_instead_of_guessing(monkeypatch, completed):
    monkeypatch.setattr(stress.subprocess, "run", lambda argv, **kw: completed)
    dump = stress._linux_proc_dump("Ubuntu", BINARY)
    assert dump["ok"] is False
    assert dump["error"]


def test_linux_probe_maps_each_instance_to_its_own_pid():
    dump = _full_dump()
    mapping = stress._map_engines(dump, BINARY, ROOT, 16)
    records = stress._instance_records(mapping, 16, ROOT)
    assert mapping["ok"] is True and mapping["unattributed"] == []
    for index in range(16):
        record = records[str(index)]
        assert record["status"] == "ok"
        assert record["pid"] == index
        assert record["pids"] == [index]
        assert record["data_dir"] == f"{ROOT}/i-{index}"
        assert record["threads"] == 1 and record["vm_rss_kb"] == 1000
        assert record["state"] == "S (sleeping)"
        assert record["cmdline_snippet"]
    assert stress._record_counts(records) == {
        "total": 16, "ok": 16, "ambiguous": 0, "missing": 0}


def test_map_engines_requires_exact_argv0_and_exact_data_dir():
    dump = _dump([
        # argv0 is a symlink/wrapper path: not the engine, so no mapping.
        _proc(10, argv0="/usr/local/bin/th08-modern", data_dir=f"{ROOT}/i-0"),
        # exact "--data-dir=" form still maps.
        _proc(11, data_dir=f"{ROOT}/i-1"),
        # i-10 must not be treated as i-1 (exact equality, no prefix match).
        _proc(12, data_dir=f"{ROOT}/i-10"),
        # missing data-dir argument: unattributed, never guessed.
        _proc(13, data_dir=None),
    ])
    mapping = stress._map_engines(dump, BINARY, ROOT, 16)
    records = stress._instance_records(mapping, 16, ROOT)
    assert records["1"]["status"] == "ok" and records["1"]["pid"] == 11
    assert [row["pid"] for row in mapping["unattributed"]] == [13]
    assert records["0"]["status"] == "missing"
    # i-10 is its own instance (exact equality) and never folds onto i-1.
    assert records["10"]["status"] == "ok" and records["10"]["pid"] == 12
    assert records["1"]["pid"] != 12
    assert records["0"]["pid"] is None
    assert records["0"]["expected_data_dir"] == f"{ROOT}/i-0"


def test_map_engines_excludes_probe_pid_and_shell_lookalikes():
    """The old failure mode: the sampler's own command line looks like a run."""
    lookalike = _proc(4242, argv0="/bin/bash",
                      snippet=f"bash -lc python3 - {BINARY} --data-dir {ROOT}/i-3")
    self_match = _proc(777, data_dir=f"{ROOT}/i-0")
    dump = _dump([lookalike, self_match, _proc(20, data_dir=f"{ROOT}/i-3")],
                 probe_pid=777)
    mapping = stress._map_engines(dump, BINARY, ROOT, 16)
    records = stress._instance_records(mapping, 16, ROOT)
    assert records["3"]["status"] == "ok" and records["3"]["pid"] == 20
    assert records["0"]["status"] == "missing" and records["0"]["pid"] is None
    assert all(row["pid"] != 4242 for row in mapping["unattributed"])


def test_instance_records_marks_ambiguity_without_a_guessed_pid():
    dump = _dump([_proc(30, data_dir=f"{ROOT}/i-4"),
                  _proc(31, data_dir=f"{ROOT}/i-4")])
    records = stress._instance_records(stress._map_engines(dump, BINARY, ROOT, 16), 16, ROOT)
    record = records["4"]
    assert record["status"] == "ambiguous"
    assert record["pid"] is None
    assert record["pids"] == [30, 31]
    assert record["reason"] == "multiple_pid_match"
    assert len(record["rows"]) == 2


def test_instance_records_marks_probe_failure_as_missing_not_zero():
    failure = {"ok": False, "error": "probe timeout"}
    mapping = stress._map_engines(failure, BINARY, ROOT, 16)
    records = stress._instance_records(mapping, 16, ROOT)
    assert all(record["status"] == "missing" for record in records.values())
    assert all(record["pid"] is None for record in records.values())
    assert all(record["reason"] == "probe_unavailable" for record in records.values())


def test_sample_offsets_fit_the_requested_duration():
    assert stress._sample_offsets(60) == [0.0, 15.0, 30.0, 59.0]
    assert stress._sample_offsets(600) == [0.0, 15.0, 30.0, 59.0]
    assert stress._sample_offsets(10) == [0.0]


def test_collect_linux_samples_keeps_every_timepoint_independent():
    clock = _FakeClock()
    calls = []

    def dump_fn():
        calls.append(clock.now)
        return _full_dump(base=1000 * len(calls))

    sink = []
    samples = stress._collect_linux_samples(
        distro="Ubuntu", binary=BINARY, root=ROOT, instances=16,
        offsets=stress._sample_offsets(60), t0=clock.now, sink=sink,
        dump_fn=dump_fn, clock=clock, sleeper=clock.sleep)

    assert samples is sink and len(samples) == 4
    assert calls == [1000.0, 1015.0, 1030.0, 1059.0]
    assert [sample["offset_seconds"] for sample in samples] == [0.0, 15.0, 30.0, 59.0]
    assert all(sample["probe_ok"] for sample in samples)
    assert all(sample["counts"]["ok"] == 16 for sample in samples)
    assert all(sample["counts"]["ambiguous"] == 0 for sample in samples)
    assert all(abs(sample["drift_seconds"]) < 1e-9 for sample in samples)
    # Each timepoint keeps its own pids: no shared dict, so nothing is overwritten.
    assert [sample["instances"]["0"]["pid"] for sample in samples] == [1000, 2000, 3000, 4000]
    for timepoint, sample in enumerate(samples):
        for index in range(16):
            record = sample["instances"][str(index)]
            assert record["pid"] == 1000 * (timepoint + 1) + index
            assert record["status"] == "ok"


def test_collect_linux_samples_stops_when_the_stop_event_is_set():
    import threading

    clock = _FakeClock()
    stop = threading.Event()

    def dump_fn():
        stop.set()
        return _full_dump()

    samples = stress._collect_linux_samples(
        distro="Ubuntu", binary=BINARY, root=ROOT, instances=16,
        offsets=stress._sample_offsets(60), t0=clock.now, dump_fn=dump_fn,
        clock=clock, sleeper=clock.sleep, stop_event=stop)
    assert len(samples) == 1


def test_audit_verdict_pass_when_16_unique_and_clean_cleanup():
    clock = _FakeClock()
    samples = stress._collect_linux_samples(
        distro="Ubuntu", binary=BINARY, root=ROOT, instances=16,
        offsets=stress._sample_offsets(60), t0=clock.now,
        dump_fn=lambda: _full_dump(), clock=clock, sleeper=clock.sleep)
    clean = stress._residual_report(_dump([_proc(1, argv0="/sbin/init")]), BINARY, ROOT)
    audit = stress._audit_verdict(samples=samples, instances=16,
                                  expected_timepoints=4, post_run=clean,
                                  post_cleanup=clean)
    assert audit["verdict"] == "PASS"
    assert audit["reasons"] == []
    assert audit["unique_mapping_count"] == 16
    assert audit["strict_all_timepoints_instances"] == list(range(16))
    assert audit["unresolved_instances"] == [] and audit["pid_changes"] == []
    assert audit["post_cleanup"]["residual_this_run_count"] == 0
    assert audit["post_cleanup"]["zombie_count"] == 0


def test_audit_verdict_partial_on_ambiguous_or_missing_mapping():
    clock = _FakeClock()
    timepoints = []
    for offset in stress._sample_offsets(60):
        procs = [_proc(100 + index, data_dir=f"{ROOT}/i-{index}") for index in range(16)
                 if index != 2]
        procs.append(_proc(500, data_dir=f"{ROOT}/i-3"))
        procs.append(_proc(501, data_dir=f"{ROOT}/i-3"))
        timepoints.append(_dump(procs))
    counter = {"n": 0}

    def dump_fn():
        dump = timepoints[min(counter["n"], len(timepoints) - 1)]
        counter["n"] += 1
        return dump

    samples = stress._collect_linux_samples(
        distro="Ubuntu", binary=BINARY, root=ROOT, instances=16,
        offsets=stress._sample_offsets(60), t0=clock.now, dump_fn=dump_fn,
        clock=clock, sleeper=clock.sleep)
    clean = stress._residual_report(_dump([_proc(1, argv0="/sbin/init")]), BINARY, ROOT)
    audit = stress._audit_verdict(samples=samples, instances=16,
                                  expected_timepoints=4, post_run=clean,
                                  post_cleanup=clean)
    assert audit["verdict"] == "partial"
    assert audit["ambiguous_instances"] == [3]
    assert audit["never_mapped_instances"] == [2]
    assert audit["unresolved_instances"] == [2, 3]
    assert audit["unique_mapping_count"] == 14
    assert "ambiguous_instance_mapping" in audit["reasons"]
    assert "instance_never_mapped" in audit["reasons"]


def test_audit_verdict_fails_on_surviving_engine_or_zombie():
    clock = _FakeClock()
    samples = stress._collect_linux_samples(
        distro="Ubuntu", binary=BINARY, root=ROOT, instances=16,
        offsets=stress._sample_offsets(60), t0=clock.now,
        dump_fn=lambda: _full_dump(), clock=clock, sleeper=clock.sleep)
    survivor = stress._residual_report(
        _dump([_proc(900, data_dir=f"{ROOT}/i-7")]), BINARY, ROOT)
    assert survivor["residual_this_run_count"] == 1
    audit = stress._audit_verdict(samples=samples, instances=16, expected_timepoints=4,
                                  post_run=survivor, post_cleanup=survivor)
    assert audit["verdict"] == "FAIL"
    assert "engine_processes_survived_cleanup" in audit["reasons"]

    zombie = stress._residual_report(
        _dump([_proc(901, data_dir=f"{ROOT}/i-7", state="Z (zombie)")]), BINARY, ROOT)
    audit = stress._audit_verdict(samples=samples, instances=16, expected_timepoints=4,
                                  post_run=zombie, post_cleanup=zombie)
    assert audit["verdict"] == "FAIL"
    assert audit["post_cleanup"]["engine_zombie_pids"] == [901]


def test_audit_verdict_unknown_without_samples_or_probe():
    clean = stress._residual_report({"ok": False, "error": "boom"}, BINARY, ROOT)
    audit = stress._audit_verdict(samples=[], instances=16, expected_timepoints=4,
                                  post_run=clean, post_cleanup=clean)
    assert audit["verdict"] == "unknown"
    assert "no_linux_samples" in audit["reasons"]
    assert "post_cleanup_probe_unavailable" in audit["reasons"]


def test_main_wires_the_linux_audit_into_the_evidence_document(monkeypatch, tmp_path):
    monkeypatch.setattr(stress, "_prepare_instance_dirs", lambda *a, **k: None)
    monkeypatch.setattr(stress, "_cleanup_instance_dirs", lambda *a, **k: 0)
    monkeypatch.setattr(stress.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "", ""))
    monkeypatch.setattr(
        stress, "one",
        lambda i, binary, data, deadline, distro="Ubuntu": {
            "instance": i, "pid": 2000 + i, "frames": 10, "steps": 10, "error": None,
            "exit_code": 0, "stderr_tail": "", "started_at": 0.0, "ended_at": 60.0,
            "run_seconds": 60.0})
    monkeypatch.setattr(stress, "_safe_proc_dump",
                        lambda *a, **k: _dump([_proc(1, argv0="/sbin/init")]))

    def fake_collect(**kwargs):
        sink = kwargs["sink"]
        for offset in kwargs["offsets"]:
            sink.append({"offset_seconds": offset, "probe_ok": True, "probe_error": None,
                         "counts": {"total": 16, "ok": 16, "ambiguous": 0, "missing": 0},
                         "instances": {str(index): {"status": "ok", "pid": 2000 + index}
                                       for index in range(16)},
                         "zombie_pids": [], "processes": []})
        return sink

    monkeypatch.setattr(stress, "_collect_linux_samples", fake_collect)
    out = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["v400_wsl_vec_stress.py", "--binary", BINARY,
                                      "--data-dir", "/root/v400-data/game",
                                      "--instances", "16", "--seconds", "60",
                                      "--out", str(out)])
    code = stress.main()
    assert code == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["linux_sample_offsets"] == [0.0, 15.0, 30.0, 59.0]
    assert len(doc["linux_samples"]) == 4
    assert doc["linux_pid_audit"]["unique_mapping_count"] == 16
    assert doc["linux_pid_audit"]["verdict"] == "PASS"
    assert doc["linux_post_run"]["status"] == "ok"
    assert doc["linux_post_cleanup"]["zombie_count"] == 0
    assert doc["all_exit_zero"] and doc["all_error_free"] and doc["no_wsl_processes"]

