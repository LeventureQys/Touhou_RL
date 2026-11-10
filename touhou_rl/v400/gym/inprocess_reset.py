"""T7.1 acceptance driver: repeated in-process ``RESET`` of one harness process.

The engine used to accept exactly one ``RESET`` per process (EXP-P4-001 §7.1:
the second one answered ``ready=0 setup_frames=1800`` with the frame counter
stuck at 0), so every episode needed a fresh process and 6-7 s of startup.  T7.1
removes that limit by driving the supervisor's own state machine back to the
title screen, whose Ready state is the only consumer of a pending start request.

This module is the *measurement*, not the fix, and it is deliberately split:

* the top half is pure (mask sequence, per-frame comparison, verdict) and is
  unit-tested on any OS;
* the bottom half talks to one real Linux ELF over the line protocol and only
  runs inside WSL, because the harness binary is an ELF.

What it measures (task book T7 验收):

1. ``--resets`` consecutive ``RESET`` commands in **one** process, every reply
   ``ready=1``;
2. for each of ``--seeds``, the 5,000-frame ``hash=`` sequence that follows every
   reset, compared **frame by frame** against a **fresh process** started with
   the same ``StageStart``;
3. the wall clock of each ``RESET`` command, with a median threshold.

Fail-closed: any ``ready=0``, any short run, any missing/invalid hash, any
non-zero engine exit, any per-frame difference, or a median over the threshold
yields ``status != "pass"`` and exit code 3.  It never invents a hash.

Run inside WSL::

    /root/v400-venv/bin/python \\
        /mnt/d/WorkShop/Touhou_RL/touhou_rl/v400/gym/inprocess_reset.py \\
        --binary /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \\
        --data-template /root/v400-data/game \\
        --run-root /root/v400-t7-reset-1 \\
        --out /mnt/d/WorkShop/Touhou_RL/.tmp_v400_t7/inprocess_reset.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

#: T7.1 asks for ten consecutive resets in one process.
REQUIRED_RESETS = 10

#: T7.1 asks for a 5,000-frame hash sequence after every reset.
REQUIRED_FRAMES = 5000

#: T7.1 asks for a median RESET wall clock of at most 1.5 s (a process restart
#: used to cost 6-7 s).
RESET_WALL_LIMIT_S = 1.5

#: Seeds the task book names for this check.
REQUIRED_SEEDS = (1234, 1235, 1236)

#: Per-frame mask sequence seed, whitelist and bit layout: identical to the
#: G0.2 item-1 sequence (EXP-P0-049), so the two experiments share one input.
MASK_SEED = 400
MASK_WHITELIST = 0x00F7
MASK_BITS = (0x0001, 0x0002, 0x0004, 0x0010, 0x0020, 0x0040, 0x0080)
MASK_BIT_NAMES = ("shoot", "bomb", "focus", "up", "down", "left", "right")

#: The same StageStart EXP-P0-049 used.  practice=1 is part of the start
#: configuration: the engine sets lifeCount=8 there, which is what lets one
#: episode cover 5,000 frames without reaching game over.
STAGE_START = {"stage": 0, "difficulty": 1, "team": 0, "practice": 1}

LINKED_FILES = ("th08.dat", "thbgm.dat")
COPIED_FILES = ("th08.cfg", "score.dat")

DEFAULT_TIMEOUT = 300.0
HASH_HEX_LENGTH = 16
HASH_DIGITS = frozenset("0123456789abcdef")

_U64 = (1 << 64) - 1


class ResetProbeError(RuntimeError):
    """A precondition, a protocol reply or an acceptance check failed."""


# ---------------------------------------------------------------------------
# Mask sequence: the documented SplitMix64 stream (pure)
# ---------------------------------------------------------------------------
def splitmix64_next(state: int) -> tuple[int, int]:
    """One SplitMix64 step: returns ``(new_state, output)``."""
    state = (state + 0x9E3779B97F4A7C15) & _U64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    return state, (z ^ (z >> 31)) & _U64


def mask_sequence(seed: int = MASK_SEED, frames: int = REQUIRED_FRAMES) -> list[int]:
    """One whitelisted mask per frame; every mask fits ``0x00F7``."""
    if not 0 <= seed <= _U64:
        raise ValueError(f"seed out of range: {seed}")
    if frames < 1:
        raise ValueError(f"frames must be positive: {frames}")
    state = seed
    masks: list[int] = []
    for _ in range(frames):
        state, value = splitmix64_next(state)
        mask = 0
        for bit_index, bit in enumerate(MASK_BITS):
            if (value >> bit_index) & 1:
                mask |= bit
        masks.append(mask)
    return masks


def mask_sequence_sha256(masks: list[int]) -> str:
    """SHA-256 of the documented textual form: comma-separated ``%04x``."""
    return hashlib.sha256(",".join(f"{mask:04x}" for mask in masks).encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# Comparison / verdict (pure)
# ---------------------------------------------------------------------------
def hashes_sha256(hashes: list[str]) -> str:
    """SHA-256 of a hash sequence, one hash per line."""
    return hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()


def compare_hashes(warm: list[str], reference: list[str], *, limit: int = 20) -> dict[str, object]:
    """Element-wise comparison of a warm sequence against a fresh-process one."""
    compared = min(len(warm), len(reference))
    differences: list[dict[str, object]] = []
    equal = 0
    first: int | None = None
    for index in range(compared):
        if warm[index] == reference[index]:
            equal += 1
            continue
        if first is None:
            first = index + 1
        if len(differences) < limit:
            differences.append({"frame": index + 1, "warm": warm[index],
                                "reference": reference[index]})
    return {
        "frames_compared": compared,
        "equal_frames": equal,
        "different_frames": compared - equal,
        "first_differing_frame": first,
        "differences": differences,
        "lengths": [len(warm), len(reference)],
        "identical": first is None and compared > 0 and len(warm) == len(reference),
    }


def reset_wall_statistics(samples: list[float]) -> dict[str, object]:
    """Median / max / min of the measured RESET wall clock, plus the verdict."""
    if not samples:
        return {"samples": 0, "median_s": None, "max_s": None, "min_s": None,
                "limit_s": RESET_WALL_LIMIT_S, "within_limit": False}
    median = statistics.median(samples)
    return {
        "samples": len(samples),
        "median_s": round(median, 6),
        "max_s": round(max(samples), 6),
        "min_s": round(min(samples), 6),
        "limit_s": RESET_WALL_LIMIT_S,
        "within_limit": median <= RESET_WALL_LIMIT_S,
    }


def evaluate(*, resets: list[dict[str, object]], references: list[dict[str, object]],
             required_resets: int, required_frames: int,
             wall: dict[str, object], warm_exit_code: object = 0) -> dict[str, object]:
    """Fail-closed acceptance decision over the resets, the references and the clock."""
    reasons: list[str] = []
    warnings: list[str] = []

    if len(resets) != required_resets:
        reasons.append(f"{len(resets)} resets recorded, require {required_resets}")
    if not references:
        reasons.append("no fresh-process reference run")
    if warm_exit_code != 0:
        reasons.append(f"warm process engine exit_code={warm_exit_code!r}, require 0")

    for run in references:
        label = run.get("label") or run.get("seed")
        if run.get("error"):
            reasons.append(f"reference {label}: {run['error']}")
        if run.get("exit_code") != 0:
            reasons.append(f"reference {label}: engine exit_code={run.get('exit_code')!r}, require 0")
        if run.get("frames_stepped") != required_frames:
            reasons.append(f"reference {label}: frames_stepped={run.get('frames_stepped')!r}, "
                           f"require {required_frames}")

    for index, run in enumerate(resets, start=1):
        label = f"reset {index} (seed {run.get('seed')})"
        if run.get("error"):
            reasons.append(f"{label}: {run['error']}")
            continue
        if run.get("ready") != "1":
            reasons.append(f"{label}: ready={run.get('ready')!r}, require 1 ({run.get('reply')})")
        if run.get("frames_stepped") != required_frames:
            reasons.append(f"{label}: frames_stepped={run.get('frames_stepped')!r}, "
                           f"require {required_frames}")
        if run.get("stop"):
            reasons.append(f"{label}: stopped early: {run['stop']}")
        compare = run.get("compare") or {}
        if compare.get("identical") is not True:
            reasons.append(
                f"{label}: {compare.get('different_frames')!r} of "
                f"{compare.get('frames_compared')!r} frames differ from the fresh process "
                f"(first differing frame {compare.get('first_differing_frame')!r})")
        debug = run.get("debug_after_reset") or {}
        if debug.get("subthread") not in (None, "0"):
            warnings.append(f"{label}: DEBUG subthread={debug.get('subthread')!r}, require 0")
        if debug.get("threads") not in (None, "1"):
            warnings.append(f"{label}: DEBUG threads={debug.get('threads')!r} "
                            "(single-thread proof not shown)")

    if not wall.get("within_limit"):
        reasons.append(f"RESET wall clock median {wall.get('median_s')!r}s exceeds "
                       f"{wall.get('limit_s')}s")
    if resets and resets[0].get("mode") not in (None, "cold"):
        warnings.append(f"the first reset reported mode={resets[0].get('mode')!r}, expected cold")
    for index, run in enumerate(resets[1:], start=2):
        if run.get("mode") not in (None, "in-process"):
            warnings.append(f"reset {index} reported mode={run.get('mode')!r}, expected in-process")

    return {
        "pass": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "required_resets": required_resets,
        "required_frames": required_frames,
        "resets_recorded": len(resets),
        "frames_compared": sum((run.get("compare") or {}).get("frames_compared") or 0
                               for run in resets),
        "different_frames": sum((run.get("compare") or {}).get("different_frames") or 0
                                for run in resets),
        "reset_wall": wall,
    }


# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
def create_data_dir(template: Path, target: Path, *, link=os.symlink) -> dict[str, object]:
    """Create a brand new data dir; never reuses or deletes an existing path."""
    template, target = Path(template), Path(target)
    if not template.is_dir():
        raise ResetProbeError(f"data template missing: {template}")
    if target.exists() or target.is_symlink():
        raise ResetProbeError(f"refusing to reuse an existing data dir: {target}")
    target.mkdir(parents=True)
    record: dict[str, object] = {"dir": str(target), "symlinks": {}, "copies": {}}
    for name in LINKED_FILES:
        source = template / name
        if not source.is_file():
            raise ResetProbeError(f"data template file missing: {source}")
        link(str(source), str(target / name))
        record["symlinks"][name] = {"source": str(source), "target": str(target / name)}
    for name in COPIED_FILES:
        source = template / name
        if not source.is_file():
            raise ResetProbeError(f"data template file missing: {source}")
        shutil.copyfile(source, target / name)
        record["copies"][name] = {
            "source": str(source),
            "target": str(target / name),
            "sha256": hashlib.sha256((target / name).read_bytes()).hexdigest(),
        }
    record["created"] = True
    return record


# ---------------------------------------------------------------------------
# Real engine process (line protocol over stdin/stdout)
# ---------------------------------------------------------------------------
def engine_argv(binary: Path, data_dir: Path) -> list[str]:
    return [str(binary), "--data-dir", str(data_dir), "--rl-control", "0,1",
            "--rl-role", "shadow"]


def child_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Shadow runs with the software null device: DISPLAY must be absent."""
    env = dict(os.environ if environ is None else environ)
    env.pop("DISPLAY", None)
    env["SDL_AUDIODRIVER"] = "dummy"
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["LP_NUM_THREADS"] = "1"
    return env


def parse_reply(line: str) -> dict[str, str]:
    """``OK reset=1 ready=1 ...`` -> ``{"_verb": "OK", ...}``."""
    parts = line.split()
    if not parts or parts[0] != "OK":
        raise ResetProbeError(f"unexpected reply: {line!r}")
    fields = {"_verb": "OK"}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def validate_hash(value: object, *, where: str) -> str:
    if not isinstance(value, str) or len(value) != HASH_HEX_LENGTH or any(
            character not in HASH_DIGITS for character in value):
        raise ResetProbeError(f"{where}: invalid hash={value!r}")
    return value


class EngineProcess:
    """One real shadow child process speaking the line protocol.

    Every ``ask`` records the wall clock of the round trip, which is what the
    RESET latency check has to be measured on: process start-up is exactly what
    T7.1 removes, so timing it would measure the wrong thing.
    """

    def __init__(self, argv: list[str], *, stderr_path: Path, env: dict[str, str],
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self.argv = list(argv)
        self.timeout = timeout
        self.stderr_path = Path(stderr_path)
        self._stderr = open(self.stderr_path, "wb")
        self.process = subprocess.Popen(
            self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr, env=env, bufsize=0,
        )
        self._buffer = b""
        self.log: list[dict[str, object]] = []

    def ask(self, command: str, timeout: float | None = None) -> dict[str, str]:
        limit = self.timeout if timeout is None else timeout
        record: dict[str, object] = {"command": command, "reply": None}
        self.log.append(record)
        started = time.perf_counter()
        try:
            assert self.process.stdin is not None
            self.process.stdin.write((command + "\n").encode("ascii"))
            self.process.stdin.flush()
        except OSError as error:
            raise ResetProbeError(
                f"pipe error on {command!r}: {error}; exit={self.process.poll()}") from error
        deadline = time.monotonic() + limit
        assert self.process.stdout is not None
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ResetProbeError(f"timeout after {limit:.1f}s on {command!r}")
            ready, _, _ = select.select([self.process.stdout], [], [], min(remaining, 0.25))
            if ready:
                chunk = os.read(self.process.stdout.fileno(), 8192)
                if not chunk:
                    raise ResetProbeError(
                        f"control pipe closed on {command!r}; exit={self.process.poll()}")
                self._buffer += chunk
            elif self.process.poll() is not None:
                raise ResetProbeError(
                    f"engine exited on {command!r}; exit={self.process.returncode}")
        line, _, self._buffer = self._buffer.partition(b"\n")
        reply = line.decode("utf-8", "replace").strip()
        record["reply"] = reply
        record["wall_s"] = round(time.perf_counter() - started, 6)
        return parse_reply(reply)

    def ask_timed(self, command: str, timeout: float | None = None) -> tuple[dict[str, str], float]:
        """Ask and return ``(reply, wall_seconds)`` for that one round trip."""
        started = time.perf_counter()
        reply = self.ask(command, timeout=timeout)
        return reply, time.perf_counter() - started

    def close(self, timeout: float = 30.0) -> dict[str, object]:
        quit_reply = None
        quit_error = None
        if self.process.poll() is None:
            try:
                quit_reply = self.ask("QUIT", timeout=timeout).get("_verb")
            except (ResetProbeError, OSError) as error:
                quit_error = str(error)
        try:
            exit_code = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            exit_code = self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        try:
            self._stderr.close()
        except OSError:
            pass
        stderr_text = ""
        if self.stderr_path.is_file():
            raw = self.stderr_path.read_bytes()
            stderr_text = (raw if len(raw) <= 65536 else raw[-65536:]).decode("utf-8", "replace")
        preflight = [line for line in stderr_text.splitlines()
                     if "shadow null-device preflight" in line]
        return {"exit_code": exit_code, "quit_reply": quit_reply, "quit_error": quit_error,
                "stderr_bytes": self.stderr_path.stat().st_size if self.stderr_path.is_file() else 0,
                "stderr_tail": stderr_text[-4096:],
                "null_device_preflight": preflight[-1] if preflight else None}


def reset_command(seed: int, stage_start: dict[str, int] = STAGE_START) -> str:
    """``RESET <stage> <difficulty> <team> <seed> <practice>``."""
    return (f"RESET {stage_start['stage']} {stage_start['difficulty']} "
            f"{stage_start['team']} {seed} {stage_start['practice']}")


# ---------------------------------------------------------------------------
# One episode: RESET (timed) then one STEP per frame
# ---------------------------------------------------------------------------
def step_frames(engine, masks: list[int], *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, object]:
    """One ``STEP <mask> 1`` per frame; return the hashes and any early stop."""
    hashes: list[str] = []
    stop: dict[str, object] | None = None
    started = time.perf_counter()
    for index, mask in enumerate(masks, start=1):
        reply = engine.ask(f"STEP {mask} 1", timeout=timeout)
        if reply.get("advanced") != "1" or reply.get("frame") != str(index) or "hash" not in reply:
            stop = {"frame": index, "mask": mask, "reply": reply,
                    "reason": "STEP did not advance exactly one frame with a hash"
                              + (" (engine reported exit=1)" if reply.get("exit") == "1" else "")}
            break
        hashes.append(validate_hash(reply.get("hash"), where=f"STEP frame {index}"))
    return {"hashes": hashes, "frames_stepped": len(hashes), "stop": stop,
            "step_wall_s": round(time.perf_counter() - started, 6)}


def run_reference(label: str, binary: Path, data_dir: Path, seed: int, masks: list[int], *,
                  run_root: Path, env: dict[str, str],
                  timeout: float = DEFAULT_TIMEOUT) -> dict[str, object]:
    """One fresh process: the sequence every warm reset must reproduce."""
    stderr_path = Path(run_root) / f"{label}.stderr.log"
    run: dict[str, object] = {"label": label, "seed": seed, "data_dir": str(data_dir),
                              "argv": engine_argv(binary, data_dir), "error": None,
                              "hashes": [], "frames_stepped": 0, "stop": None,
                              "exit_code": None, "role": None, "ready": None}
    engine = None
    started = time.perf_counter()
    try:
        engine = EngineProcess(run["argv"], stderr_path=stderr_path, env=env, timeout=timeout)
        hello = engine.ask("HELLO", timeout=timeout)
        run["role"] = hello.get("role")
        reply, reset_wall = engine.ask_timed(reset_command(seed), timeout=timeout)
        run["reply"] = reply
        run["ready"] = reply.get("ready")
        run["mode"] = reply.get("mode")
        run["setup_frames"] = reply.get("setup_frames")
        run["reset_wall_s"] = round(reset_wall, 6)
        run["frame0_hash"] = reply.get("hash")
        if reply.get("ready") != "1":
            raise ResetProbeError(f"reference RESET not ready: {reply}")
        # Freshness of the shared state block (T7.3): only meaningful with SHM.
        run.update(step_frames(engine, masks, timeout=timeout))
        if not run["stop"]:
            final = engine.ask("HASH", timeout=timeout)
            run["final_hash_crosscheck"] = (
                final.get("hash") == run["hashes"][-1] and final.get("frames") == str(len(masks)))
            run["debug_after"] = engine.ask("DEBUG", timeout=timeout)
    except (ResetProbeError, OSError, ValueError) as error:
        run["error"] = str(error)
    finally:
        if engine is not None:
            run.update(engine.close())
        run["wall_s"] = round(time.perf_counter() - started, 6)
    run["hashes_sha256"] = hashes_sha256(run["hashes"])
    return run


def run_warm(binary: Path, data_dir: Path, seeds: list[int], resets: int, masks: list[int], *,
             run_root: Path, env: dict[str, str], references: dict[int, list[str]],
             timeout: float = DEFAULT_TIMEOUT) -> dict[str, object]:
    """One process, ``resets`` consecutive RESETs, cycling through ``seeds``."""
    stderr_path = Path(run_root) / "warm.stderr.log"
    out: dict[str, object] = {"data_dir": str(data_dir), "argv": engine_argv(binary, data_dir),
                              "error": None, "resets": [], "exit_code": None, "role": None,
                              "hello": None}
    engine = None
    started = time.perf_counter()
    try:
        engine = EngineProcess(out["argv"], stderr_path=stderr_path, env=env, timeout=timeout)
        hello = engine.ask("HELLO", timeout=timeout)
        out["hello"] = hello
        out["role"] = hello.get("role")
        for index in range(1, resets + 1):
            seed = seeds[(index - 1) % len(seeds)]
            record: dict[str, object] = {"index": index, "seed": seed,
                                         "reset_command": reset_command(seed), "error": None}
            out["resets"].append(record)
            try:
                reply, reset_wall = engine.ask_timed(reset_command(seed), timeout=timeout)
            except ResetProbeError as error:
                record["error"] = str(error)
                break
            record["reply"] = reply
            record["ready"] = reply.get("ready")
            record["mode"] = reply.get("mode")
            record["setup_frames"] = reply.get("setup_frames")
            record["frame0_hash"] = reply.get("hash")
            record["reset_wall_s"] = round(reset_wall, 6)
            if reply.get("ready") != "1":
                # Keep going would measure a dead episode; record and stop.
                record["error"] = f"RESET not ready: {reply}"
                break
            record["debug_after_reset"] = engine.ask("DEBUG", timeout=timeout)
            record.update(step_frames(engine, masks, timeout=timeout))
            record["hashes_sha256"] = hashes_sha256(record["hashes"])
            record["compare"] = compare_hashes(record["hashes"], references.get(seed, []))
            if not record["stop"]:
                final = engine.ask("HASH", timeout=timeout)
                record["final_hash_crosscheck"] = (
                    final.get("hash") == record["hashes"][-1]
                    and final.get("frames") == str(len(masks)))
    except (ResetProbeError, OSError, ValueError) as error:
        out["error"] = str(error)
    finally:
        if engine is not None:
            out.update(engine.close())
        out["wall_s"] = round(time.perf_counter() - started, 6)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data-template", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="fresh directory for the new data dirs and stderr logs")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path (must not exist)")
    parser.add_argument("--resets", type=int, default=REQUIRED_RESETS)
    parser.add_argument("--frames", type=int, default=REQUIRED_FRAMES)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in REQUIRED_SEEDS))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)

    if args.out.exists():
        print(f"refusing to overwrite existing report: {args.out}", file=sys.stderr)
        return 2
    if sys.platform == "win32":
        print("blocked: the real shadow binary is a Linux ELF; run this tool inside WSL",
              file=sys.stderr)
        return 3
    if args.run_root.exists() or args.run_root.is_symlink():
        print(f"refusing to reuse an existing run root: {args.run_root}", file=sys.stderr)
        return 2
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    if not seeds:
        print("no seeds given", file=sys.stderr)
        return 2
    if args.resets < 1 or args.frames < 1:
        print(f"invalid counts: resets={args.resets} frames={args.frames}", file=sys.stderr)
        return 2

    binary_info: dict[str, object] = {"path": str(args.binary), "sha256": None, "size": None}
    harness_info: dict[str, object] = {"repo": None, "head": None, "dirty": None}
    data_info: dict[str, object] = {"template": str(args.data_template),
                                    "run_root": str(args.run_root), "created": []}
    environment: dict[str, object] = {"timeout_s": args.timeout}
    masks = mask_sequence(MASK_SEED, args.frames)
    references: list[dict[str, object]] = []
    warm: dict[str, object] = {"resets": []}
    started = time.perf_counter()
    report: dict[str, object] = {}
    invocation = {"argv": [sys.executable, *sys.argv], "cwd": os.getcwd(),
                  "python_version": sys.version.split()[0]}

    try:
        if not args.binary.is_file():
            raise ResetProbeError(f"binary missing: {args.binary}")
        binary_info.update({"sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                            "size": args.binary.stat().st_size})
        for candidate in args.binary.resolve().parents:
            if (candidate / "src" / "rlharness").is_dir():
                harness_info.update({"repo": str(candidate),
                                     "head": _git(candidate, "rev-parse", "HEAD"),
                                     "dirty": bool(_git(candidate, "status", "--porcelain",
                                                        "--untracked-files=no"))})
                break
        data_info["template_sha256"] = {
            name: hashlib.sha256((args.data_template / name).read_bytes()).hexdigest()
            for name in (*LINKED_FILES, *COPIED_FILES)
            if (args.data_template / name).is_file()
        }
        args.run_root.mkdir(parents=True)
        env = child_env()
        environment["variables"] = {key: env[key] for key in
                                    ("SDL_AUDIODRIVER", "LIBGL_ALWAYS_SOFTWARE",
                                     "LP_NUM_THREADS")}
        environment["display"] = "unset" if "DISPLAY" not in env else "present"

        # 1. one fresh process per seed: the reference sequence.
        reference_hashes: dict[int, list[str]] = {}
        for seed in seeds:
            directory = args.run_root / f"ref-{seed}"
            data_info["created"].append(create_data_dir(args.data_template, directory))
            run = run_reference(f"ref-{seed}", args.binary, directory, seed, masks,
                                run_root=args.run_root, env=env, timeout=args.timeout)
            references.append(run)
            reference_hashes[seed] = list(run["hashes"])

        # 2. one process, many resets, compared against those sequences.
        warm_directory = args.run_root / "warm"
        data_info["created"].append(create_data_dir(args.data_template, warm_directory))
        warm = run_warm(args.binary, warm_directory, seeds, args.resets, masks,
                        run_root=args.run_root, env=env, references=reference_hashes,
                        timeout=args.timeout)
    except (ResetProbeError, OSError, ValueError) as error:
        report = {"status": "blocked", "reason": str(error)}
    else:
        resets = list(warm.get("resets") or [])
        wall = reset_wall_statistics([float(run["reset_wall_s"]) for run in resets
                                      if run.get("reset_wall_s") is not None])
        verdict = evaluate(resets=resets, references=references,
                           required_resets=args.resets, required_frames=args.frames, wall=wall,
                           warm_exit_code=warm.get("exit_code"))
        status = "pass" if verdict["pass"] else "failed"
        if args.resets != REQUIRED_RESETS or args.frames != REQUIRED_FRAMES:
            # A short run is a preflight, never acceptance evidence.
            status = "preflight-only"
            verdict["pass"] = False
            verdict["reasons"].append(
                f"resets={args.resets} frames={args.frames} is not the acceptance shape "
                f"({REQUIRED_RESETS} x {REQUIRED_FRAMES})")
        report = {
            "schema": "v400-t7-inprocess-reset/1",
            "experiment": "EXP-P0-078",
            "task": "T7.1: repeated in-process RESET, identical to a fresh process",
            "status": status,
            "binary": binary_info,
            "harness": harness_info,
            "data": data_info,
            "environment": environment,
            "invocation": invocation,
            "stage_start": dict(STAGE_START, seeds=seeds,
                                command_template="RESET <stage> <difficulty> <team> <seed> <practice>"),
            "mask": {"seed": MASK_SEED, "length": len(masks),
                     "whitelist": f"0x{MASK_WHITELIST:04X}", "bit_names": list(MASK_BIT_NAMES),
                     "sha256": mask_sequence_sha256(masks), "masks": masks},
            "references": references,
            "warm": warm,
            "reset_wall": wall,
            "verdict": verdict,
            "wall_s": round(time.perf_counter() - started, 3),
            "limits": [
                "practice=1 StageStart: the engine sets lifeCount=8, which is what lets one "
                "5000-frame episode run without reaching game over; this is not a full-run claim.",
                "Per-frame hashes are the engine's FNV-1a digest over its 25 tracked fields, "
                "not a full-memory hash.",
                "Shadow role (null device, no GL): the reset path is the same state machine the "
                "env role uses, but these numbers are not a rendering claim.",
                "Data dirs are new paths created for this run and are never reused or deleted; "
                "th08.dat/thbgm.dat are read-only symlinks into the template.",
            ],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": report.get("status"),
        "resets": len((report.get("warm") or {}).get("resets") or []),
        "ready": [run.get("ready") for run in (report.get("warm") or {}).get("resets") or []],
        "modes": [run.get("mode") for run in (report.get("warm") or {}).get("resets") or []],
        "frames_per_reset": [run.get("frames_stepped") for run in
                             (report.get("warm") or {}).get("resets") or []],
        "different_frames": [((run.get("compare") or {}).get("different_frames"))
                             for run in (report.get("warm") or {}).get("resets") or []],
        "reset_wall": report.get("reset_wall"),
        "exit_codes": {"warm": (report.get("warm") or {}).get("exit_code"),
                       "references": [run.get("exit_code")
                                      for run in report.get("references") or []]},
        "mask_sha256": (report.get("mask") or {}).get("sha256"),
        "binary_sha256": (report.get("binary") or {}).get("sha256"),
        "wall_s": report.get("wall_s"),
        "out": str(args.out),
        "reasons": (report.get("verdict") or {}).get("reasons"),
        "warnings": (report.get("verdict") or {}).get("warnings"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report.get("status") == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
