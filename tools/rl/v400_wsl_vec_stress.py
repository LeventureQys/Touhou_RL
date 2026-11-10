"""Bounded real Windows->WSL 16-instance WP4 long stress run."""
from __future__ import annotations
import argparse, json, os, shlex, subprocess, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tools.rl.v400_wsl_env import WslHarnessClient

_STDERR_TAIL_BYTES = 16 * 1024

# Offsets (seconds after the 16 workers are submitted) used for the Linux-side
# audit samples.  Fixed and small on purpose: cheap, reproducible, and enough to
# see engine establishment, steady state, late-run state and pre-exit state.
_SAMPLE_OFFSETS = (0, 15, 30, 59)
_PROBE_TIMEOUT_SECONDS = 30.0
_PROBE_PROCESS_LIMIT = 512
_PROBE_SNIPPET_LIMIT = 512

# Enumerates Linux processes *only* through /proc.  It never shells out to
# ps/pgrep/grep/awk, so no sampler can ever match its own command line: the
# caller-side matcher requires argv[0] to equal the engine path exactly plus an
# exact ``--data-dir <root>/i-N`` argument.  The program is fed on stdin
# (``python3 -``) so newlines survive wsl.exe argument flattening.
_LINUX_PROBE_SOURCE = r'''
import json, os, sys

SNIPPET_LIMIT = __SNIPPET_LIMIT__
PROCESS_LIMIT = __PROCESS_LIMIT__


def _read_status(pid):
    fields = {}
    try:
        with open("/proc/%d/status" % pid, "rb") as handle:
            for raw in handle:
                line = raw.decode("utf-8", "replace")
                key, sep, value = line.partition(":")
                if sep:
                    fields[key.strip()] = value.strip()
    except OSError:
        return fields
    return fields


def _read_stat(pid):
    try:
        with open("/proc/%d/stat" % pid, "rb") as handle:
            stat = handle.read().decode("utf-8", "replace")
    except OSError:
        return {}
    end = stat.rfind(")")
    if end < 0:
        return {}
    rest = stat[end + 2:].split()
    out = {}
    if rest:
        out["state"] = rest[0]
    if len(rest) > 1:
        try:
            out["ppid"] = int(rest[1])
        except ValueError:
            pass
    if len(rest) > 19:
        try:
            out["starttime_ticks"] = int(rest[19])
        except ValueError:
            pass
    return out


def _number(text):
    if not text:
        return None
    parts = text.split()
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def _data_dir(argv):
    for index, arg in enumerate(argv):
        if arg == "--data-dir" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--data-dir="):
            return arg.split("=", 1)[1]
    return None


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else ""
    processes = []
    scanned = 0
    skipped = 0
    truncated = 0
    names = [name for name in os.listdir("/proc") if name.isdigit()]
    names.sort(key=lambda item: int(item))
    for name in names:
        scanned += 1
        pid = int(name)
        try:
            with open("/proc/%d/cmdline" % pid, "rb") as handle:
                raw = handle.read()
        except OSError:
            skipped += 1
            continue
        argv = [chunk.decode("utf-8", "replace") for chunk in raw.split(b"\x00") if chunk]
        if not argv:
            # Kernel thread or a process already past its argv teardown.
            skipped += 1
            continue
        if len(processes) >= PROCESS_LIMIT:
            truncated += 1
            continue
        status = _read_status(pid)
        stat = _read_stat(pid)
        processes.append({
            "pid": pid,
            "ppid": stat.get("ppid"),
            "state": status.get("State") or stat.get("state"),
            "threads": _number(status.get("Threads")),
            "vm_rss_kb": _number(status.get("VmRSS")),
            "vm_hwm_kb": _number(status.get("VmHWM")),
            "starttime_ticks": stat.get("starttime_ticks"),
            "argv0": argv[0],
            "argv_count": len(argv),
            "data_dir": _data_dir(argv),
            "cmdline_snippet": " ".join(argv)[:SNIPPET_LIMIT],
        })
    print(json.dumps({
        "ok": True,
        "probe_pid": os.getpid(),
        "probe_argv0": sys.argv[0],
        "engine": engine,
        "proc_scanned": scanned,
        "proc_skipped": skipped,
        "proc_truncated": truncated,
        "engine_argv0_pids": [p["pid"] for p in processes if p["argv0"] == engine],
        "zombie_pids": [p["pid"] for p in processes
                        if (p.get("state") or "").startswith("Z")],
        "processes": processes,
    }))


main()
'''
_LINUX_PROBE_SOURCE = _LINUX_PROBE_SOURCE.replace(
    "__SNIPPET_LIMIT__", str(_PROBE_SNIPPET_LIMIT)
).replace("__PROCESS_LIMIT__", str(_PROBE_PROCESS_LIMIT))


def _prepare_instance_dirs(data: str, distro: str, root: str, instances: int) -> None:
    """Create isolated writable files with read-only data-file symlinks.

    Keep setup fail-closed: WSL reports the resolved diagnostic root and Python
    preserves stdout/stderr/exit status when setup fails.
    """
    if instances < 1:
        raise ValueError("instances must be positive")
    qdata = shlex.quote(data)
    qroot = shlex.quote(root)
    script = (
        "set -eu; "
        f"case {qroot} in /tmp/v400_wp4_diag_?*) ;; "
        "*) echo 'REFUSED_UNSAFE_ROOT' >&2; exit 2;; esac; "
        f"mkdir -p -- {qroot}; "
        f"resolved=\\$(realpath -e -- {qroot}); "
        "case \"\\$resolved\" in /tmp/v400_wp4_diag_?*) ;; "
        "*) echo 'REFUSED_UNSAFE_RESOLVED_ROOT' >&2; exit 3;; esac; "
        "printf 'PREPARE_RESOLVED=%s\\n' \"\\$resolved\"; "
        f"i=0; while [ \\$i -lt {instances} ]; do dst=\"\\$resolved/i-\\$i\"; mkdir -p -- \"\\$dst\"; "
        f"find {qdata} -maxdepth 1 -type f -name '*.dat' ! -name 'score.dat' ! -name 'th08.cfg' -exec ln -s {{}} \"\\$dst/\" \\; ; "
        f"for f in th08.cfg score.dat; do if [ -f {qdata}/\\$f ]; then cp -p {qdata}/\\$f \"\\$dst/\\$f\"; fi; done; "
        "i=\\$((i+1)); done"
    )
    completed = subprocess.run(
        ["wsl.exe", "-d", distro, "--", "bash", "-lc", script],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "instance directory preparation failed "
            f"(exit={completed.returncode})\\n"
            f"stdout:\\n{completed.stdout}\\n"
            f"stderr:\\n{completed.stderr}"
        )
    if f"PREPARE_RESOLVED={root}" not in completed.stdout:
        raise RuntimeError(
            "instance directory preparation did not report expected resolved root "
            f"(root={root!r}, stdout={completed.stdout!r}, stderr={completed.stderr!r})"
        )


def _cleanup_instance_dirs(distro: str, root: str) -> int:
    """Remove only a verified diagnostic root, and report resolved target."""
    qroot = shlex.quote(root)
    check = (
        f"set -eu; root={qroot}; case \"\\$root\" in /tmp/v400_wp4_diag_?*) ;; "
        "*) echo 'REFUSED_UNSAFE_ROOT' >&2; exit 2;; esac; "
        "resolved=\\$(realpath -e -- \"\\$root\"); "
        "case \"\\$resolved\" in /tmp/v400_wp4_diag_?*) ;; "
        "*) echo 'REFUSED_UNSAFE_RESOLVED_ROOT' >&2; exit 3;; esac; "
        "printf 'CLEANUP_RESOLVED=%s\\n' \"\\$resolved\"; rm -rf -- \"\\$resolved\""
    )
    completed = subprocess.run(
        ["wsl.exe", "-d", distro, "--", "bash", "-lc", check],
        check=False, capture_output=True, text=True,
    )
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=__import__("sys").stderr)
    return completed.returncode


def _linux_proc_dump(distro: str, binary: str, *, timeout: float = _PROBE_TIMEOUT_SECONDS) -> dict:
    """Enumerate Linux /proc from inside WSL and return raw per-process facts.

    The probe program is sent on stdin so that no shell, quoting layer or
    command-line flattening can turn it into something that matches itself.
    """
    argv = ["wsl.exe", "-d", distro, "--", "python3", "-", binary]
    completed = subprocess.run(
        argv, input=_LINUX_PROBE_SOURCE, capture_output=True, text=True, timeout=timeout,
    )
    if completed.returncode:
        return {
            "ok": False,
            "error": f"linux /proc probe exited {completed.returncode}",
            "stdout_tail": (completed.stdout or "")[-2000:],
            "stderr_tail": (completed.stderr or "")[-2000:],
        }
    try:
        dump = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error": f"linux /proc probe JSON decode failed: {exc}",
            "stdout_tail": (completed.stdout or "")[-2000:],
            "stderr_tail": (completed.stderr or "")[-2000:],
        }
    if not isinstance(dump, dict) or not isinstance(dump.get("processes"), list):
        return {"ok": False, "error": "linux /proc probe payload missing processes list",
                "stdout_tail": (completed.stdout or "")[-2000:],
                "stderr_tail": (completed.stderr or "")[-2000:]}
    dump["ok"] = True
    return dump


def _safe_proc_dump(distro: str, binary: str, *, timeout: float = _PROBE_TIMEOUT_SECONDS) -> dict:
    """Never raise: a failed sample is recorded as failed, not guessed."""
    try:
        return _linux_proc_dump(distro, binary, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"linux /proc probe timed out after {timeout}s"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "error": f"linux /proc probe {type(exc).__name__}: {exc}"}


def _sample_offsets(seconds: float) -> list[float]:
    """Fixed audit offsets that fit inside the requested duration."""
    return [float(offset) for offset in _SAMPLE_OFFSETS if float(offset) < float(seconds)] or [0.0]


def _instance_from_data_dir(data_dir, root: str, instances: int):
    """Exact ``<root>/i-N`` match; prefix matching would confuse i-1 and i-10."""
    if not isinstance(data_dir, str):
        return None
    for index in range(instances):
        if data_dir == f"{root}/i-{index}":
            return index
    return None


def _map_engines(dump: dict, binary: str, root: str, instances: int) -> dict:
    """Pure mapping: /proc dump -> per-instance engine rows, never a guessed value.

    A row is an engine iff ``argv[0]`` equals the absolute engine path exactly
    and its parsed ``--data-dir`` equals ``<root>/i-N`` exactly.  The probe's own
    pid is excluded by pid.
    """
    grouped = {index: [] for index in range(instances)}
    if not isinstance(dump, dict) or not dump.get("ok"):
        return {"ok": False,
                "error": (dump or {}).get("error", "linux /proc probe unavailable")
                if isinstance(dump, dict) else "linux /proc probe unavailable",
                "by_instance": grouped, "unattributed": []}
    probe_pid = dump.get("probe_pid")
    unattributed = []
    for proc in dump.get("processes") or []:
        if not isinstance(proc, dict):
            continue
        if probe_pid is not None and proc.get("pid") == probe_pid:
            continue
        if proc.get("argv0") != binary:
            continue
        row = {
            "pid": proc.get("pid"),
            "ppid": proc.get("ppid"),
            "state": proc.get("state"),
            "threads": proc.get("threads"),
            "vm_rss_kb": proc.get("vm_rss_kb"),
            "vm_hwm_kb": proc.get("vm_hwm_kb"),
            "starttime_ticks": proc.get("starttime_ticks"),
            "data_dir": proc.get("data_dir"),
            "cmdline_snippet": proc.get("cmdline_snippet"),
        }
        index = _instance_from_data_dir(proc.get("data_dir"), root, instances)
        if index is None:
            unattributed.append(row)
        else:
            grouped[index].append(row)
    return {"ok": True, "error": None, "by_instance": grouped, "unattributed": unattributed}


def _instance_records(mapping: dict, instances: int, root: str | None = None) -> dict:
    """One independent record per instance: ok / missing / ambiguous, no fill-in."""
    records = {}
    available = bool(mapping.get("ok"))
    reason = mapping.get("error") or "probe_unavailable"
    for index in range(instances):
        rows = (mapping.get("by_instance") or {}).get(index) or []
        key = str(index)
        expected = f"{root}/i-{index}" if root else None
        if not available:
            records[key] = {"status": "missing", "pid": None, "pids": [],
                            "reason": "probe_unavailable", "detail": reason,
                            "expected_data_dir": expected}
        elif len(rows) == 1:
            record = dict(rows[0])
            record["status"] = "ok"
            record["pids"] = [rows[0].get("pid")]
            record["reason"] = None
            records[key] = record
        elif not rows:
            records[key] = {"status": "missing", "pid": None, "pids": [],
                            "reason": "no_proc_match",
                            "expected_data_dir": expected}
        else:
            records[key] = {"status": "ambiguous", "pid": None,
                            "pids": [row.get("pid") for row in rows],
                            "reason": "multiple_pid_match", "rows": rows}
    return records


def _record_counts(records: dict) -> dict:
    values = list((records or {}).values())
    return {
        "total": len(values),
        "ok": sum(1 for value in values if value.get("status") == "ok"),
        "ambiguous": sum(1 for value in values if value.get("status") == "ambiguous"),
        "missing": sum(1 for value in values if value.get("status") == "missing"),
    }


def _residual_report(dump: dict, binary: str, root: str) -> dict:
    """Post-run / post-cleanup check: engine survivors and Z-state processes."""
    if not isinstance(dump, dict) or not dump.get("ok"):
        return {"status": "error",
                "error": (dump or {}).get("error", "linux /proc probe unavailable")
                if isinstance(dump, dict) else "linux /proc probe unavailable",
                "residual_this_run": [], "engine_processes_total": [],
                "engine_process_count": None, "zombie_count": None,
                "zombie_pids": [], "zombie_processes": [], "engine_zombie_pids": []}
    processes = [proc for proc in (dump.get("processes") or []) if isinstance(proc, dict)]
    engines = [proc for proc in processes if proc.get("argv0") == binary]
    prefix = f"{root}/i-"
    residual = [proc for proc in engines
                if isinstance(proc.get("data_dir"), str) and proc["data_dir"].startswith(prefix)]
    zombies = [proc for proc in processes if (proc.get("state") or "").startswith("Z")]
    engine_zombies = [proc for proc in zombies if proc.get("argv0") == binary]

    def brief(proc: dict) -> dict:
        return {"pid": proc.get("pid"), "ppid": proc.get("ppid"),
                "state": proc.get("state"), "data_dir": proc.get("data_dir"),
                "starttime_ticks": proc.get("starttime_ticks"),
                "cmdline_snippet": proc.get("cmdline_snippet")}

    return {
        "status": "ok",
        "proc_scanned": dump.get("proc_scanned"),
        "proc_skipped": dump.get("proc_skipped"),
        "residual_this_run": [brief(proc) for proc in residual],
        "residual_this_run_count": len(residual),
        "engine_processes_total": [brief(proc) for proc in engines],
        "engine_process_count": len(engines),
        "zombie_count": len(zombies),
        "zombie_pids": [proc.get("pid") for proc in zombies],
        "zombie_processes": [brief(proc) for proc in zombies],
        "engine_zombie_pids": [proc.get("pid") for proc in engine_zombies],
        "processes": processes,
    }


def _collect_linux_samples(*, distro: str, binary: str, root: str, instances: int,
                           offsets, t0: float, sink: list | None = None,
                           dump_fn=None, clock=time.monotonic, sleeper=time.sleep,
                           stop_event=None) -> list:
    """Sample /proc at each offset; every sample is an independent record.

    ``sink`` (when given) is appended to in place, so a caller can read partial
    progress without any shared per-instance dictionary that could overwrite.
    """
    collected = [] if sink is None else sink
    dump_fn = dump_fn or (lambda: _safe_proc_dump(distro, binary))
    for offset in offsets:
        target = t0 + float(offset)
        while True:
            if stop_event is not None and stop_event.is_set():
                return collected
            remaining = target - clock()
            if remaining <= 0:
                break
            sleeper(min(remaining, 0.25))
        dump = dump_fn()
        sampled_at = clock()
        mapping = _map_engines(dump, binary, root, instances)
        records = _instance_records(mapping, instances, root)
        collected.append({
            "offset_seconds": float(offset),
            "target_monotonic": target,
            "sampled_monotonic": sampled_at,
            "drift_seconds": sampled_at - target,
            "probe_ok": bool(isinstance(dump, dict) and dump.get("ok")),
            "probe_error": (dump.get("error") if isinstance(dump, dict) else "probe_not_dict"),
            "proc_scanned": dump.get("proc_scanned") if isinstance(dump, dict) else None,
            "proc_skipped": dump.get("proc_skipped") if isinstance(dump, dict) else None,
            "engine_process_count": (len(dump.get("engine_argv0_pids") or [])
                                     if isinstance(dump, dict) else None),
            "zombie_pids": (dump.get("zombie_pids") or []) if isinstance(dump, dict) else [],
            "instances": records,
            "counts": _record_counts(records),
            "unattributed_engine_processes": mapping.get("unattributed") or [],
            "processes": (dump.get("processes") or []) if isinstance(dump, dict) else [],
        })
    return collected


def _audit_verdict(*, samples, instances: int, expected_timepoints: int,
                   post_run: dict | None, post_cleanup: dict | None) -> dict:
    """Turn the raw samples into an explicit PASS / partial / FAIL / unknown call."""
    samples = list(samples or [])
    per_instance = {index: {"ok": 0, "ambiguous": 0, "missing": 0, "pids": []}
                    for index in range(instances)}
    coverage = []
    for sample in samples:
        counts = sample.get("counts") or {}
        coverage.append({
            "offset_seconds": sample.get("offset_seconds"),
            "probe_ok": sample.get("probe_ok"),
            "probe_error": sample.get("probe_error"),
            "drift_seconds": sample.get("drift_seconds"),
            "ok": counts.get("ok"),
            "ambiguous": counts.get("ambiguous"),
            "missing": counts.get("missing"),
            "engine_process_count": sample.get("engine_process_count"),
            "zombie_count": len(sample.get("zombie_pids") or []),
        })
        for key, record in (sample.get("instances") or {}).items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            slot = per_instance.setdefault(index, {"ok": 0, "ambiguous": 0, "missing": 0, "pids": []})
            status = record.get("status")
            if status == "ok":
                slot["ok"] += 1
                pid = record.get("pid")
                if pid is not None and pid not in slot["pids"]:
                    slot["pids"].append(pid)
            elif status == "ambiguous":
                slot["ambiguous"] += 1
            else:
                slot["missing"] += 1

    unique, ambiguous, never, strict, pid_changes = [], [], [], [], []
    for index in range(instances):
        slot = per_instance[index]
        if slot["ambiguous"]:
            ambiguous.append(index)
        if slot["ok"] == 0 and not slot["ambiguous"]:
            never.append(index)
        if slot["ok"] > 0 and not slot["ambiguous"]:
            unique.append(index)
        if len(samples) and slot["ok"] == len(samples) and not slot["ambiguous"]:
            strict.append(index)
        if len(slot["pids"]) > 1:
            pid_changes.append({"instance": index, "pids": slot["pids"]})

    post_cleanup = post_cleanup or {}
    post_run = post_run or {}
    residual = post_cleanup.get("residual_this_run") or []
    engine_zombies = post_cleanup.get("engine_zombie_pids") or []
    zombie_count = post_cleanup.get("zombie_count")

    verdict = "PASS"
    reasons = []
    if not samples:
        verdict = "unknown"
        reasons.append("no_linux_samples")
    elif not any(sample.get("probe_ok") for sample in samples):
        verdict = "unknown"
        reasons.append("all_linux_probes_failed")
    if post_cleanup.get("status") != "ok":
        reasons.append("post_cleanup_probe_unavailable")
        if verdict == "PASS":
            verdict = "unknown"
    if residual:
        reasons.append("engine_processes_survived_cleanup")
        verdict = "FAIL"
    if engine_zombies:
        reasons.append("engine_zombie_processes_after_cleanup")
        verdict = "FAIL"
    if zombie_count not in (None, 0):
        reasons.append("container_zombie_processes_present")
        if verdict == "PASS":
            verdict = "partial"
    if ambiguous:
        reasons.append("ambiguous_instance_mapping")
        if verdict == "PASS":
            verdict = "partial"
    if never:
        reasons.append("instance_never_mapped")
        if verdict == "PASS":
            verdict = "partial"
    if pid_changes:
        reasons.append("instance_pid_changed_between_timepoints")
        if verdict == "PASS":
            verdict = "partial"
    if len(samples) < expected_timepoints:
        reasons.append("missing_timepoints")
        if verdict == "PASS":
            verdict = "partial"
    if any(not sample.get("probe_ok") for sample in samples):
        reasons.append("timepoint_probe_failed")
        if verdict == "PASS":
            verdict = "partial"

    return {
        "verdict": verdict,
        "reasons": reasons,
        "expected_timepoints": expected_timepoints,
        "observed_timepoints": len(samples),
        "instances": instances,
        "unique_mapping_count": len(unique),
        "unique_mapping_instances": unique,
        "ambiguous_instances": ambiguous,
        "never_mapped_instances": never,
        "unresolved_instances": sorted(set(ambiguous) | set(never)),
        "strict_all_timepoints_instances": strict,
        "pid_changes": pid_changes,
        "timepoint_coverage": coverage,
        "post_run": {"status": post_run.get("status"),
                     "residual_this_run_count": post_run.get("residual_this_run_count"),
                     "engine_process_count": post_run.get("engine_process_count"),
                     "zombie_count": post_run.get("zombie_count"),
                     "zombie_pids": post_run.get("zombie_pids")},
        "post_cleanup": {"status": post_cleanup.get("status"),
                         "residual_this_run_count": post_cleanup.get("residual_this_run_count"),
                         "engine_process_count": post_cleanup.get("engine_process_count"),
                         "zombie_count": zombie_count,
                         "zombie_pids": post_cleanup.get("zombie_pids"),
                         "engine_zombie_pids": engine_zombies},
    }


def one(i: int, binary: str, data: str, deadline: float, distro: str = "Ubuntu") -> dict:
    c = WslHarnessClient(binary, data, distro=distro, timeout=5.0,
                         shm_name=f"wp4_{os.getpid()}_{i}")
    r = {"instance": i, "pid": None, "frames": 0, "steps": 0,
         "error": None, "exit_code": None, "stderr_tail": "",
         "started_at": time.time(), "ended_at": None}
    stderr_done = threading.Event()
    stderr_buf = bytearray()

    def drain_stderr() -> None:
        proc = c._proc
        stream = proc.stderr if proc is not None else None
        try:
            while stream is not None:
                chunk = stream.read(4096)
                if not chunk:
                    break
                stderr_buf.extend(chunk)
                if len(stderr_buf) > _STDERR_TAIL_BYTES:
                    del stderr_buf[:-_STDERR_TAIL_BYTES]
        finally:
            stderr_done.set()

    try:
        c.start()
        r["pid"] = c._proc.pid if c._proc else None
        stderr_thread = threading.Thread(target=drain_stderr,
                                          name=f"wsl-stderr-{i}", daemon=True)
        stderr_thread.start()
        c.hello()
        c.reset(stage=0, difficulty=1, team=0, seed=i + 1)
        while time.monotonic() < deadline:
            out = c.step(0, 1)
            r["steps"] += 1
            r["frames"] += int(out.get("advanced", 0))
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
    finally:
        proc = c._proc
        try:
            c.close(quit_timeout=3.0)
        except Exception as e:
            if not r["error"]:
                r["error"] = f"close {type(e).__name__}: {e}"
        if proc is not None:
            r["exit_code"] = proc.returncode
        stderr_done.wait(timeout=1.0)
        r["stderr_tail"] = bytes(stderr_buf).decode("utf-8", "replace")
        r["ended_at"] = time.time()
        r["run_seconds"] = r["ended_at"] - r["started_at"]
    return r


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--instances", type=int, default=16)
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--out", required=True)
    p.add_argument("--distro", default="Ubuntu")
    a = p.parse_args()
    if a.instances != 16 or a.seconds > 600 or a.seconds <= 0:
        p.error("WP4 long gate requires exactly 16 instances and 0 < seconds <= 600")
    started = time.time()
    root = f"/tmp/v400_wp4_diag_{os.getpid()}_{uuid.uuid4().hex}"
    setup_error = None
    samples: list = []
    offsets = _sample_offsets(a.seconds)
    sampler_thread = None
    stop_event = threading.Event()
    sampler_error = None
    post_run = None
    post_cleanup = None
    try:
        _prepare_instance_dirs(a.data_dir, a.distro, root, a.instances)
        deadline = time.monotonic() + a.seconds
        t0 = time.monotonic()
        sampler_thread = threading.Thread(
            target=_collect_linux_samples,
            kwargs={"distro": a.distro, "binary": a.binary, "root": root,
                    "instances": a.instances, "offsets": offsets, "t0": t0,
                    "sink": samples, "stop_event": stop_event},
            name="linux-proc-sampler", daemon=True,
        )
        sampler_thread.start()
        with ThreadPoolExecutor(max_workers=a.instances) as ex:
            fs = [ex.submit(one, i, a.binary, f"{root}/i-{i}", deadline, a.distro)
                  for i in range(a.instances)]
            results = [f.result() for f in as_completed(fs)]
        # The sampler's last offset (59s) always precedes the run deadline (60s),
        # so a normal join here completes the full schedule; stop_event only
        # matters if something hung.
        sampler_thread.join(timeout=15.0)
        if sampler_thread.is_alive():
            stop_event.set()
            sampler_error = "linux sampler thread still running after join timeout"
        post_run = _residual_report(_safe_proc_dump(a.distro, a.binary), a.binary, root)
    except Exception as e:
        setup_error = f"{type(e).__name__}: {e}"
        stop_event.set()
        results = []
        post_run = None
    if sampler_thread is not None and sampler_thread.is_alive():
        stop_event.set()
    results.sort(key=lambda x: x["instance"])
    ps = subprocess.run(["tasklist", "/FI", "IMAGENAME eq wsl.exe", "/FO", "CSV", "/NH"],
                        capture_output=True, text=True)
    remaining = [x for x in ps.stdout.splitlines() if x.strip() and "INFO:" not in x]
    cleanup_code = _cleanup_instance_dirs(a.distro, root)
    post_cleanup = _residual_report(_safe_proc_dump(a.distro, a.binary), a.binary, root)
    audit = _audit_verdict(samples=samples, instances=a.instances,
                           expected_timepoints=len(offsets),
                           post_run=post_run, post_cleanup=post_cleanup)
    wall = time.time() - started
    doc = {
        "provider": "DeepSeek", "model": "DeepSeek v4.1 flash",
        "instances": a.instances, "continuous": True,
        "requested_seconds": a.seconds, "max_seconds": 600,
        "wall_seconds": wall, "results": results,
        "setup_error": setup_error, "instance_data_root": root,
        "cleanup_exit_code": cleanup_code,
        "remaining_wsl_process_lines": remaining,
        "all_instances_reached_deadline": len(results) == a.instances and all(
            x["run_seconds"] >= a.seconds - 2 for x in results),
        "all_exit_zero": len(results) == a.instances and all(x["exit_code"] == 0 for x in results),
        "no_wsl_processes": not remaining,
        "all_error_free": len(results) == a.instances and all(x["error"] is None for x in results),
        "linux_sample_offsets": offsets,
        "linux_sampler_error": sampler_error,
        "linux_samples": samples,
        "linux_post_run": post_run,
        "linux_post_cleanup": post_cleanup,
        "linux_pid_audit": audit,
        "memory_audit_scope": "bounded_30s_only" if a.seconds <= 30 else "bounded_request_not_600s_proof",
    }
    Path(a.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps(doc, indent=2))
    return 0 if (doc["all_instances_reached_deadline"] and doc["all_exit_zero"]
                  and doc["no_wsl_processes"] and doc["all_error_free"]
                  and cleanup_code == 0
                  and audit["verdict"] != "FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
