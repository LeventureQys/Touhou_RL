"""WP5 / G0.2 item 1: real per-frame determinism of two independent shadow processes.

What this measures
------------------
Two **independent** real ``th08-modern --rl-role shadow`` processes get:

* their own **brand new** data directory (``th08.dat``/``thbgm.dat`` are
  read-only symlinks into the data template, ``th08.cfg``/``score.dat`` are
  byte copies) -- an existing path is never reused or deleted;
* the same ``RESET 0 1 0 1234 1`` (Stage 1 / Normal / team 0 / game seed 1234 /
  practice resources), i.e. the same ``StageStart``;
* the same fixed-seed mask sequence, one whitelisted mask per frame, one
  ``STEP <mask> 1`` per frame, 5,000 frames.

Every frame's ``hash=`` field from the ``STEP`` reply is recorded for both runs
and compared.  The tool is fail-closed: a short run, a missing/invalid hash, a
frame-number gap, a nonzero engine exit, a non-shadow role or any per-frame
mismatch yields ``status != "pass"`` and exit code 3.  It never invents a hash
and never edits the engine, the protocol or a dependency.

Mask sequence reproducibility
-----------------------------
``mask_sequence(seed=400)`` is a SplitMix64 stream: ``state = seed``; frame *i*
uses bits 0..6 of the ``i``-th 64-bit output, one bit per whitelist button
(``shoot|bomb|focus|up|down|left|right`` = ``0x00F7``).  ``BTN_MENU`` (``0x08``)
and every system bit stay clear.  The exact sequence is written to the report
together with its SHA-256, so an independent reader can regenerate it.

Run inside WSL (the binary is a Linux ELF)::

    env -u DISPLAY /root/v400-venv/bin/python tools/rl/v400_wp5_determinism.py \
        --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern \
        --data-template /root/v400-data/game \
        --run-root /root/v400-wp5-determinism-5000 \
        --out /mnt/d/WorkShop/Touhou_RL/.tmp_v400_wp5_determinism_5000.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: G0.2 item 1 asks for exactly 5,000 frames per run.
REQUIRED_FRAMES = 5000

#: Seed of the per-frame mask sequence (not the game RNG seed).
MASK_SEED = 400

#: The only bits a policy may emit: shoot|bomb|focus|up|down|left|right.
#: ``BTN_MENU`` (0x08) and all system bits are excluded.
MASK_WHITELIST = 0x00F7
MASK_BITS = (0x0001, 0x0002, 0x0004, 0x0010, 0x0020, 0x0040, 0x0080)
MASK_BIT_NAMES = ("shoot", "bomb", "focus", "up", "down", "left", "right")

#: The one StageStart both processes get.  practice=1 is part of the start
#: configuration (the engine sets lifeCount=8 there), exactly as in EXP-P0-005.
STAGE_START = {"stage": 0, "difficulty": 1, "team": 0, "seed": 1234, "practice": 1}
RESET_COMMAND = "RESET {stage} {difficulty} {team} {seed} {practice}".format(**STAGE_START)

#: Data files: the two archives are read-only symlinks, the two writable ones
#: are copies so the template is untouched.
LINKED_FILES = ("th08.dat", "thbgm.dat")
COPIED_FILES = ("th08.cfg", "score.dat")

DEFAULT_TIMEOUT = 120.0
HASH_HEX_LENGTH = 16


class DeterminismError(RuntimeError):
    """A real precondition, protocol reply or acceptance check failed."""


# ---------------------------------------------------------------------------
# Mask sequence (pure, reproducible, testable on any OS)
# ---------------------------------------------------------------------------
_U64 = (1 << 64) - 1


def splitmix64_next(state: int) -> tuple[int, int]:
    """One SplitMix64 step: returns ``(new_state, output)``."""
    state = (state + 0x9E3779B97F4A7C15) & _U64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    return state, (z ^ (z >> 31)) & _U64


def mask_sequence(seed: int = MASK_SEED, frames: int = REQUIRED_FRAMES) -> list[int]:
    """The reproducible one-mask-per-frame sequence; every mask fits 0x00F7."""
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
    text = ",".join(f"{mask:04x}" for mask in masks)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def mask_sequence_document(masks: list[int]) -> dict[str, object]:
    return {
        "seed": MASK_SEED,
        "length": len(masks),
        "whitelist": f"0x{MASK_WHITELIST:04X}",
        "bit_names": list(MASK_BIT_NAMES),
        "algorithm": (
            "state=seed; for each frame: state=(state+0x9E3779B97F4A7C15) mod 2^64; "
            "z=state; z=((z^(z>>30))*0xBF58476D1CE4E5B9) mod 2^64; "
            "z=((z^(z>>27))*0x94D049BB133111EB) mod 2^64; value=z^(z>>31); "
            "mask=OR of MASK_BITS[i] for i in 0..6 where bit i of value is 1"
        ),
        "digest_text_form": "comma separated lowercase %04x masks",
        "sha256": mask_sequence_sha256(masks),
        "head": masks[:16],
        "tail": masks[-8:],
        "unique_mask_count": len(set(masks)),
        "all_masks_whitelisted": all(mask & ~MASK_WHITELIST == 0 for mask in masks),
        "masks": masks,
    }


# ---------------------------------------------------------------------------
# Comparison / verdict (pure)
# ---------------------------------------------------------------------------
def compare_hashes(a: list[str], b: list[str], *, limit: int = 20) -> dict[str, object]:
    """Element-wise comparison of two per-frame hash sequences."""
    compared = min(len(a), len(b))
    differences: list[dict[str, object]] = []
    equal = 0
    first: int | None = None
    for index in range(compared):
        if a[index] == b[index]:
            equal += 1
            continue
        if first is None:
            first = index + 1  # 1-based frame number
        if len(differences) < limit:
            differences.append({"frame": index + 1, "a": a[index], "b": b[index]})
    return {
        "frames_compared": compared,
        "equal_frames": equal,
        "different_frames": compared - equal,
        "first_differing_frame": first,
        "differences": differences,
        "lengths": [len(a), len(b)],
        "all_equal": first is None and compared > 0 and len(a) == len(b),
    }


def run_failure_reasons(run: dict[str, object], required_frames: int) -> list[str]:
    """Every reason this single run cannot count as acceptance evidence."""
    label = run.get("label", "run")
    reasons: list[str] = []
    if run.get("error"):
        reasons.append(f"{label}: {run['error']}")
    if run.get("role") != "shadow":
        reasons.append(f"{label}: HELLO role={run.get('role')!r}, require shadow")
    if not run.get("reset_ready"):
        reasons.append(f"{label}: RESET did not report ready=1 ({run.get('reset')})")
    debug = run.get("debug_after_reset") or {}
    if debug.get("curState") != "2":
        reasons.append(f"{label}: after RESET curState={debug.get('curState')!r}, require 2")
    if run.get("stop"):
        reasons.append(f"{label}: stopped early: {run['stop']}")
    if run.get("frames_stepped") != required_frames:
        reasons.append(f"{label}: frames_stepped={run.get('frames_stepped')!r}, require {required_frames}")
    if len(run.get("hashes") or []) != required_frames:
        reasons.append(f"{label}: recorded {len(run.get('hashes') or [])} hashes, require {required_frames}")
    if run.get("exit_code") != 0:
        reasons.append(f"{label}: engine exit_code={run.get('exit_code')!r}, require 0")
    if run.get("final_hash_crosscheck") is False:
        reasons.append(f"{label}: final HASH disagrees with the last STEP hash")
    return reasons


def evaluate(runs: list[dict[str, object]], compare: dict[str, object],
             required_frames: int) -> dict[str, object]:
    """Fail-closed acceptance decision over both runs and the comparison."""
    reasons: list[str] = []
    if len(runs) != 2:
        reasons.append(f"need exactly 2 independent runs, got {len(runs)}")
    for run in runs:
        reasons.extend(run_failure_reasons(run, required_frames))
    if compare.get("frames_compared") != required_frames:
        reasons.append(f"frames compared={compare.get('frames_compared')!r}, require {required_frames}")
    if compare.get("different_frames") != 0:
        reasons.append(
            f"{compare.get('different_frames')} of {compare.get('frames_compared')} compared frames differ; "
            f"first differing frame={compare.get('first_differing_frame')}"
        )
    if compare.get("frame0_equal") is not True:
        reasons.append("frame-0 (RESET) hashes differ")
    warnings: list[str] = []
    for run in runs:
        debug = run.get("debug_after_reset") or {}
        if debug.get("threads") != "1":
            warnings.append(f"{run.get('label')}: DEBUG threads={debug.get('threads')!r} (single-thread proof not shown)")
        if debug.get("subthread") != "0":
            warnings.append(f"{run.get('label')}: DEBUG subthread={debug.get('subthread')!r}, require 0")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "required_frames": required_frames,
        "frames_compared": compare.get("frames_compared"),
        "equal_frames": compare.get("equal_frames"),
        "different_frames": compare.get("different_frames"),
        "first_differing_frame": compare.get("first_differing_frame"),
    }


# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
def create_data_dir(template: Path, target: Path, *, link=os.symlink) -> dict[str, object]:
    """Create a brand new data dir; never reuses or deletes an existing path."""
    template, target = Path(template), Path(target)
    if not template.is_dir():
        raise DeterminismError(f"data template missing: {template}")
    if target.exists() or target.is_symlink():
        raise DeterminismError(f"refusing to reuse an existing data dir: {target}")
    target.mkdir(parents=True)
    record: dict[str, object] = {"dir": str(target), "symlinks": {}, "copies": {}}
    for name in LINKED_FILES:
        source = template / name
        if not source.is_file():
            raise DeterminismError(f"data template file missing: {source}")
        link(str(source), str(target / name))
        record["symlinks"][name] = {"source": str(source), "target": str(target / name)}
    for name in COPIED_FILES:
        source = template / name
        if not source.is_file():
            raise DeterminismError(f"data template file missing: {source}")
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
    """``OK advanced=1 frame=1 hash=...`` -> ``{"_verb": "OK", ...}``."""
    parts = line.split()
    if not parts or parts[0] != "OK":
        raise DeterminismError(f"unexpected reply: {line!r}")
    fields = {"_verb": "OK"}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def validate_hash(value: object, *, where: str) -> str:
    if not isinstance(value, str) or len(value) != HASH_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in value):
        raise DeterminismError(f"{where}: invalid hash={value!r}")
    return value


class EngineProcess:
    """One real shadow child process speaking the line protocol."""

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
        self.log: list[dict[str, str | None]] = []

    def ask(self, command: str, timeout: float | None = None) -> dict[str, str]:
        limit = self.timeout if timeout is None else timeout
        record: dict[str, str | None] = {"command": command, "reply": None}
        self.log.append(record)
        try:
            assert self.process.stdin is not None
            self.process.stdin.write((command + "\n").encode("ascii"))
            self.process.stdin.flush()
        except OSError as error:
            raise DeterminismError(f"pipe error on {command!r}: {error}; exit={self.process.poll()}") from error
        deadline = time.monotonic() + limit
        assert self.process.stdout is not None
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DeterminismError(f"timeout after {limit:.1f}s on {command!r}")
            ready, _, _ = select.select([self.process.stdout], [], [], min(remaining, 0.25))
            if ready:
                chunk = os.read(self.process.stdout.fileno(), 8192)
                if not chunk:
                    raise DeterminismError(
                        f"control pipe closed on {command!r}; exit={self.process.poll()}")
                self._buffer += chunk
            elif self.process.poll() is not None:
                raise DeterminismError(
                    f"engine exited on {command!r}; exit={self.process.returncode}")
        line, _, self._buffer = self._buffer.partition(b"\n")
        reply = line.decode("utf-8", "replace").strip()
        record["reply"] = reply
        return parse_reply(reply)

    def close(self, timeout: float = 15.0) -> dict[str, object]:
        quit_reply = None
        quit_error = None
        if self.process.poll() is None:
            try:
                quit_reply = self.ask("QUIT", timeout=timeout).get("_verb")
            except (DeterminismError, OSError) as error:
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
        stderr_bytes = 0
        if self.stderr_path.is_file():
            raw = self.stderr_path.read_bytes()
            stderr_bytes = len(raw)
            stderr_text = (raw if len(raw) <= 65536 else raw[-65536:]).decode("utf-8", "replace")
        return {"exit_code": exit_code, "quit_reply": quit_reply,
                "quit_error": quit_error, "stderr_bytes": stderr_bytes,
                "stderr_tail": stderr_text[-4096:] if len(stderr_text) > 4096 else stderr_text,
                "stderr_text": stderr_text}


# ---------------------------------------------------------------------------
# One run: RESET then one STEP per frame
# ---------------------------------------------------------------------------
def collect_frames(engine, masks: list[int], *,
                   reset_command: str = RESET_COMMAND,
                   timeout: float = DEFAULT_TIMEOUT) -> dict[str, object]:
    """HELLO/DEBUG/RESET, then one ``STEP <mask> 1`` per frame; record every hash."""
    hello = engine.ask("HELLO", timeout=timeout)
    role = hello.get("role")
    debug_initial = engine.ask("DEBUG", timeout=timeout)
    reset = engine.ask(reset_command, timeout=timeout)
    reset_ready = reset.get("ready") == "1"
    if not reset_ready:
        raise DeterminismError(f"RESET not ready: {reset}")
    if reset.get("frames") not in (None, "0"):
        raise DeterminismError(f"RESET left a nonzero frame counter: {reset}")
    frame0_hash = validate_hash(reset.get("hash"), where="RESET")
    debug_after_reset = engine.ask("DEBUG", timeout=timeout)
    hashes: list[str] = []
    stop: dict[str, object] | None = None
    started = time.perf_counter()
    for index, mask in enumerate(masks, start=1):
        reply = engine.ask(f"STEP {mask} 1", timeout=timeout)
        advanced = reply.get("advanced")
        frame = reply.get("frame")
        if advanced != "1" or frame != str(index) or "hash" not in reply:
            stop = {
                "frame": index,
                "mask": mask,
                "reply": reply,
                "reason": "STEP did not advance exactly one frame with a hash"
                          + (" (engine reported exit=1)" if reply.get("exit") == "1" else ""),
            }
            break
        hashes.append(validate_hash(reply.get("hash"), where=f"STEP frame {index}"))
    step_wall_s = time.perf_counter() - started
    debug_after = None
    crosscheck = None
    if stop is None:
        debug_after = engine.ask("DEBUG", timeout=timeout)
        final = engine.ask("HASH", timeout=timeout)
        crosscheck = (final.get("hash") == hashes[-1]) and (final.get("frames") == str(len(masks)))
    return {
        "role": role,
        "hello": hello,
        "debug_initial": debug_initial,
        "reset_command": reset_command,
        "reset": reset,
        "reset_ready": reset_ready,
        "frame0_hash": frame0_hash,
        "debug_after_reset": debug_after_reset,
        "hashes": hashes,
        "frames_stepped": len(hashes),
        "stop": stop,
        "step_wall_s": step_wall_s,
        "final_hash_crosscheck": crosscheck,
        "debug_after": debug_after,
    }


def run_process(label: str, binary: Path, data_dir: Path, masks: list[int], *,
                run_root: Path, env: dict[str, str], timeout: float = DEFAULT_TIMEOUT) -> dict[str, object]:
    """Start a real shadow process, collect its per-frame hashes, close it."""
    argv = engine_argv(binary, data_dir)
    stderr_path = Path(run_root) / f"{label}.stderr.log"
    started = time.perf_counter()
    engine = None
    run: dict[str, object] = {
        "label": label, "data_dir": str(data_dir), "argv": argv,
        "stderr_log": str(stderr_path), "role": None, "error": None,
        "hashes": [], "frames_stepped": 0, "stop": None,
        "exit_code": None, "reset_ready": False,
    }
    try:
        engine = EngineProcess(argv, stderr_path=stderr_path, env=env, timeout=timeout)
        run.update(collect_frames(engine, masks, timeout=timeout))
    except (DeterminismError, OSError, ValueError) as error:
        run["error"] = str(error)
    finally:
        if engine is not None:
            run.update(engine.close())
        run["wall_s"] = time.perf_counter() - started
        stderr_text = str(run.pop("stderr_text", "") or "")
        preflight = [
            line for line in stderr_text.splitlines()
            if "shadow null-device preflight" in line
        ]
        run["null_device_preflight"] = preflight[-1] if preflight else None
    return run


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(*, runs: list[dict[str, object]], masks: list[int],
                 binary_info: dict[str, object], harness_info: dict[str, object],
                 data_info: dict[str, object], environment: dict[str, object],
                 required_frames: int = REQUIRED_FRAMES,
                 wall_s: float | None = None,
                 invocation: dict[str, object] | None = None) -> dict[str, object]:
    a = runs[0]["hashes"] if len(runs) > 0 else []
    b = runs[1]["hashes"] if len(runs) > 1 else []
    compare = compare_hashes(list(a), list(b))
    compare["frame0_equal"] = (
        len(runs) == 2
        and runs[0].get("frame0_hash") is not None
        and runs[0].get("frame0_hash") == runs[1].get("frame0_hash")
    )
    compare["hashes_sha256"] = [hashlib.sha256("\n".join(h).encode("ascii")).hexdigest() for h in (a, b)]
    verdict = evaluate(runs, compare, required_frames)
    debug = {
        run.get("label"): {
            "threads": (run.get("debug_after_reset") or {}).get("threads"),
            "subthread": (run.get("debug_after_reset") or {}).get("subthread"),
            "curState_after_reset": (run.get("debug_after_reset") or {}).get("curState"),
            "curState_after_steps": (run.get("debug_after") or {}).get("curState"),
            "frames_after_steps": (run.get("debug_after") or {}).get("frames"),
        }
        for run in runs
    }
    report = {
        "schema": "v400-wp5-determinism-5000/1",
        "experiment": "EXP-P0-049",
        "gate": "G0.2 item 1: same StageStart + same 5000-frame random input, two runs, every frame hash equal",
        "status": "pass" if verdict["pass"] else "failed",
        "required_frames": required_frames,
        "frames_per_run": [len(h) for h in (a, b)],
        "stage_start": dict(STAGE_START, command=RESET_COMMAND),
        "mask": mask_sequence_document(masks),
        "binary": binary_info,
        "harness": harness_info,
        "data": data_info,
        "environment": environment,
        "invocation": invocation,
        "wall_s": wall_s,
        "runs": runs,
        "compare": compare,
        "debug": debug,
        "verdict": verdict,
        "limits": [
            "Practice StageStart (practice=1): the engine sets lifeCount=8, which is what lets one "
            "episode cover 5,000 frames; this is not a full-run/no-death claim.",
            "Per-frame hashes are the engine's FNV-1a digest over its 25 tracked fields, not a full-memory hash.",
            "Data dirs are new paths created for this run and are never reused or deleted; "
            "th08.dat/thbgm.dat are read-only symlinks into the template.",
        ],
    }
    return report


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
                        help="fresh directory for the two new data dirs and stderr logs")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path (must not exist)")
    parser.add_argument("--frames", type=int, default=REQUIRED_FRAMES)
    parser.add_argument("--last-frame", type=int, default=None,
                        help="only use the first N masks (preflight); acceptance still needs --frames")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)

    if args.out.exists():
        print(f"refusing to overwrite existing report: {args.out}", file=sys.stderr)
        return 2
    if sys.platform == "win32":
        print("blocked: the real shadow binary is a Linux ELF; run this tool inside WSL", file=sys.stderr)
        return 3
    frames = args.frames if args.last_frame is None else min(args.frames, args.last_frame)
    if frames < 1:
        print(f"invalid frame count: {frames}", file=sys.stderr)
        return 2
    if args.run_root.exists() or args.run_root.is_symlink():
        print(f"refusing to reuse an existing run root: {args.run_root}", file=sys.stderr)
        return 2

    binary_info: dict[str, object] = {"path": str(args.binary), "sha256": None, "size": None}
    harness_info: dict[str, object] = {"repo": None, "head": None, "dirty": None}
    data_info: dict[str, object] = {"template": str(args.data_template), "run_root": str(args.run_root),
                                    "dirs": [], "created_fresh": False}
    environment: dict[str, object] = {"display": "unset", "timeout_s": args.timeout}
    runs: list[dict[str, object]] = []
    masks = mask_sequence(MASK_SEED, frames)
    started = time.perf_counter()
    report: dict[str, object] = {}
    invocation = {
        "argv": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "python_version": sys.version.split()[0],
    }

    try:
        if not args.binary.is_file():
            raise DeterminismError(f"binary missing: {args.binary}")
        binary_info.update({
            "sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            "size": args.binary.stat().st_size,
        })
        repo = None
        for candidate in args.binary.resolve().parents:
            if (candidate / "src" / "rlharness").is_dir():
                repo = candidate
                break
        if repo is not None:
            harness_info.update({
                "repo": str(repo),
                "head": _git(repo, "rev-parse", "HEAD"),
                "dirty": bool(_git(repo, "status", "--porcelain", "--untracked-files=no")),
            })
        data_info["template_sha256"] = {
            name: hashlib.sha256((args.data_template / name).read_bytes()).hexdigest()
            for name in (*LINKED_FILES, *COPIED_FILES)
            if (args.data_template / name).is_file()
        }
        args.run_root.mkdir(parents=True)
        data_dirs = [args.run_root / "data-a", args.run_root / "data-b"]
        data_info["dirs"] = [str(Path(directory)) for directory in data_dirs]
        data_info["created"] = [
            create_data_dir(args.data_template, directory) for directory in data_dirs
        ]
        data_info["created_fresh"] = True
        env = child_env()
        environment["variables"] = {key: env[key] for key in
                                    ("SDL_AUDIODRIVER", "LIBGL_ALWAYS_SOFTWARE", "LP_NUM_THREADS")}
        environment["display"] = "unset" if "DISPLAY" not in env else "present(empty)"
        for label, directory in zip(("run-a", "run-b"), data_dirs):
            runs.append(run_process(label, args.binary, directory, masks,
                                    run_root=args.run_root, env=env, timeout=args.timeout))
    except (DeterminismError, OSError, ValueError) as error:
        report = build_report(runs=runs, masks=masks, binary_info=binary_info,
                              harness_info=harness_info, data_info=data_info,
                              environment=environment, required_frames=args.frames,
                              wall_s=time.perf_counter() - started, invocation=invocation)
        report["status"] = "blocked"
        report["verdict"]["pass"] = False
        report["verdict"]["reasons"] = [*report["verdict"]["reasons"], f"precondition failed: {error}"]
        report["reason"] = str(error)
    else:
        report = build_report(runs=runs, masks=masks, binary_info=binary_info,
                              harness_info=harness_info, data_info=data_info,
                              environment=environment, required_frames=args.frames,
                              wall_s=time.perf_counter() - started, invocation=invocation)
        if args.frames != REQUIRED_FRAMES:
            report["status"] = "preflight-only"
            report["verdict"]["pass"] = False
            report["verdict"]["reasons"].append(
                f"frames={args.frames} != required {REQUIRED_FRAMES}: not acceptance evidence")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": report["status"],
        "frames_per_run": report["frames_per_run"],
        "equal_frames": report["compare"]["equal_frames"],
        "different_frames": report["compare"]["different_frames"],
        "first_differing_frame": report["compare"]["first_differing_frame"],
        "exit_codes": [run.get("exit_code") for run in runs],
        "debug": report["debug"],
        "mask_sha256": report["mask"]["sha256"],
        "wall_s": round(report["wall_s"], 3) if report.get("wall_s") is not None else None,
        "out": str(args.out),
        "reasons": report["verdict"]["reasons"],
        "warnings": report["verdict"]["warnings"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
