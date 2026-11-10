"""Bounded, real WP2 / G0.3 render+readback throughput probe.

The G0.3 hard gate asks for *whole-machine* ``>= 2000 frames/s`` for
"render + readback": the env-role harness draws each frame and then
``glReadPixels`` 640x480 RGBA back to the CPU, and the frame rate is what the
engine actually sustains.  This tool measures exactly that number and nothing
else:

* it launches the real ``th08-modern`` ELF with ``--rl-control 0,1 --rl-role env``
  (the env role is the only role that renders and reads pixels back);
* it enables per-frame readback with ``READBACK 1``;
* throughput is ``frames advanced / wall seconds spent inside STEP``, timed in
  this process, never estimated from a frame count alone;
* the engine's own ``READBACK`` counters are differenced to report the real
  ``glReadPixels`` cost per frame -- the evidence for *why* a number is low;
* the GL backend actually used by each child is proven from the child's own
  ``glGetString(GL_RENDERER)`` (captured in-process by an ``LD_PRELOAD``
  instrument) and from ``/proc/<pid>/environ``, instead of trusting the caller's
  environment.  ``/proc/<pid>/maps`` is recorded too but deliberately *not* used
  as proof: this Mesa build hardlinks every gallium driver to one file, so the
  mapped path can read ``swrast_dri.so`` while the context is really D3D12.

Runs inside WSL (Linux), where the harness binary lives::

    /root/v400-venv/bin/python tools/rl/v400_render_throughput.py \
        --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern \
        --data-template /root/v400-data/game \
        --run-root /root/v400-render-tp-a1 \
        --backend software --display :98 --instances 1 --seconds 10 \
        --gl-probe-shim /tmp/libgl_probe.so --expect-renderer llvmpipe \
        --out /mnt/d/WorkShop/Touhou_RL/.tmp_v400_render_tp/a1.json

Exit codes: ``0`` = valid measurement written, ``3`` = unavailable/invalid
(nothing is reported as engine throughput), ``2`` = bad usage.  A run that
cannot prove its GL backend fails closed with ``3``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

PIXEL_WIDTH = 640
PIXEL_HEIGHT = 480
PIXEL_BYTES = PIXEL_WIDTH * PIXEL_HEIGHT * 4
GATE_FPS = 2000.0
GATE_RANGE_LOW = 500.0

# Knobs that decide which Mesa driver a child uses.  They are stripped from the
# inherited environment before the backend overlay is applied so an ambient
# ``LIBGL_ALWAYS_SOFTWARE=1`` can never silently turn the hardware run into a
# software run.
GL_KNOBS = ("LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER", "LP_NUM_THREADS",
            "MESA_LOADER_DRIVER_OVERRIDE", "LIBGL_ALWAYS_INDIRECT", "DISPLAY",
            "WAYLAND_DISPLAY")

DATA_FILES_LINK = ("th08.dat", "thbgm.dat")
DATA_FILES_COPY = ("th08.cfg", "score.dat")


class ProbeUnavailable(RuntimeError):
    """The measurement cannot be made valid; never report a number instead."""


# ---------------------------------------------------------------- environment
def build_child_env(backend: str, *, display: str | None, lp_threads: int,
                    base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return the exact environment for one harness child.

    ``software`` forces Mesa llvmpipe; ``hardware`` removes every forcing knob
    so the WSLg D3D12 driver is used, and *refuses* to run when a forcing knob
    is still present afterwards.
    """
    if backend not in ("software", "hardware"):
        raise ValueError(f"unknown backend: {backend!r}")
    if lp_threads < 1:
        raise ValueError("lp_threads must be positive")
    env = {k: v for k, v in (base_env if base_env is not None else os.environ).items()
           if k not in GL_KNOBS}
    env["SDL_AUDIODRIVER"] = "dummy"
    if display:
        env["DISPLAY"] = display
    if backend == "software":
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"
        env["GALLIUM_DRIVER"] = "llvmpipe"
        env["LP_NUM_THREADS"] = str(lp_threads)
    for knob in GL_KNOBS:
        if knob in ("DISPLAY", "WAYLAND_DISPLAY"):
            continue
        if fingerprint_conflict(backend, knob, env.get(knob)):
            raise ProbeUnavailable(
                f"backend {backend!r} cannot be guaranteed: {knob}={env.get(knob)!r}"
            )
    return env


def parse_child_env(pairs: list[str] | None) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` experiment switches for the harness children.

    These carry build/feature switches that must reach the child *and* be
    recorded in the output document (e.g. ``TH08_ASYNC_READBACK=1`` for
    EXP-P0-061).  GL backend knobs are refused here: which Mesa driver runs is
    decided by ``--backend`` and proven from the child's own ``glGetString``, so
    an overlay must never be able to fake it.
    """
    overlay: dict[str, str] = {}
    for item in pairs or []:
        key, separator, value = item.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ProbeUnavailable(f"--child-env must be KEY=VALUE, got {item!r}")
        if key in GL_KNOBS:
            raise ProbeUnavailable(
                f"--child-env {key!r} is a GL backend knob; use --backend instead")
        overlay[key] = value
    return overlay


def fingerprint_conflict(backend: str, knob: str, value: str | None) -> bool:
    """True when a GL knob would invalidate the declared backend."""
    if value is None:
        return False
    if backend == "hardware":
        return knob in ("LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER",
                        "MESA_LOADER_DRIVER_OVERRIDE")
    return False


def classify_gl(maps_text: str) -> dict[str, bool]:
    """Which Mesa driver file names a process has mapped (informational only)."""
    return {
        "d3d12": "d3d12_dri.so" in maps_text,
        "swrast": "swrast_dri.so" in maps_text,
        "llvmpipe": "llvmpipe" in maps_text,
        "zink": "zink_dri.so" in maps_text,
    }


def parse_gl_fingerprint(text: str, pid: int | None = None) -> dict:
    """Parse the in-process shim log into one renderer fingerprint.

    The shim appends ``pid=/GL_*/`` lines on the first ``glReadPixels``.  When the
    engine ever emits more than one block, prefer the block for ``pid``.
    """
    blocks: list[dict] = []
    current: dict = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key == "pid":
            if current:
                blocks.append(current)
            current = {"pid": int(value) if value.isdigit() else None}
        else:
            current[key] = value
    if current:
        blocks.append(current)
    if not blocks:
        return {}
    if pid is not None:
        for block in blocks:
            if block.get("pid") == pid:
                return block
    return blocks[0]


def renderer_matches(expected: str | None, fingerprint: dict) -> bool:
    """True when the captured GL_RENDERER contains the expected token."""
    if not expected:
        return True
    renderer = (fingerprint or {}).get("GL_RENDERER") or ""
    return expected.lower() in renderer.lower()


def backend_matches(backend: str, seen: dict[str, bool]) -> bool:
    """Fallback check on mapped driver file names (used only without a shim)."""
    if backend == "software":
        return bool(seen.get("swrast")) and not seen.get("d3d12")
    return bool(seen.get("d3d12")) and not seen.get("swrast")


# ------------------------------------------------------------------ accounting
def read_loadavg(path: str = "/proc/loadavg") -> dict:
    """Sample the machine load so every point records what else was running.

    The G0.3 number is only comparable across runs when the machine is idle, so
    the probe records the real ``/proc/loadavg`` line instead of trusting the
    operator's memory.  A missing/unparseable file is reported, never invented.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        return {"available": False, "raw": None, "error": f"{type(exc).__name__}: {exc}"}
    parts = text.split()
    try:
        load1, load5, load15 = (float(parts[0]), float(parts[1]), float(parts[2]))
    except (IndexError, ValueError):
        return {"available": False, "raw": text, "error": "not a loadavg line"}
    runnable = parts[3] if len(parts) > 3 else None
    return {"available": True, "raw": text, "load1": load1, "load5": load5,
            "load15": load15, "runnable": runnable}


def compute_throughput(frames: int, seconds: float) -> float:
    """Frames per second; refuses to divide by a non-positive interval."""
    if frames < 0:
        raise ValueError("frames must not be negative")
    if not seconds > 0:
        raise ValueError("seconds must be positive")
    return frames / seconds


def aggregate_throughput(instances: list[dict], *, window_seconds: float) -> dict:
    """Whole-machine throughput: every instance's frames over one shared wall window."""
    if not instances:
        raise ProbeUnavailable("no instances")
    if not window_seconds > 0:
        raise ValueError("window_seconds must be positive")
    frames = sum(int(i["frames"]) for i in instances)
    step_seconds = sum(float(i["step_seconds"]) for i in instances)
    return {
        "instances": len(instances),
        "frames": frames,
        "window_seconds": window_seconds,
        "frames_per_second": compute_throughput(frames, window_seconds),
        "instance_fps_sum": sum(float(i["frames_per_second"]) for i in instances),
        "step_seconds_sum": step_seconds,
        "parallel_efficiency": (step_seconds / window_seconds) if window_seconds else 0.0,
    }


def gate_verdict(fps: float, *, readback: bool = True) -> dict:
    """Map a measured aggregate onto the ``10-P0`` WP2 decision table."""
    if not readback:
        return {"gate_applicable": False, "gate_fps": GATE_FPS, "passed": False,
                "decision": "diagnostic_only_readback_disabled"}
    if fps >= GATE_FPS:
        return {"gate_applicable": True, "gate_fps": GATE_FPS, "passed": True,
                "decision": "continue_as_planned"}
    if fps >= GATE_RANGE_LOW:
        return {"gate_applicable": True, "gate_fps": GATE_FPS, "passed": False,
                "decision": "continue_but_rebudget_by_measurement"}
    return {"gate_applicable": True, "gate_fps": GATE_FPS, "passed": False,
            "decision": "stop_and_ask_owner"}


# ------------------------------------------------------------------- data dirs
def _link_readonly_file(source: Path, target: Path) -> str:
    """Share a large read-only data file when the platform allows it.

    Symlink is what the WSL runs use; a hardlink or a copy keeps the tool usable
    on filesystems/platforms that refuse symlinks (e.g. a Windows checkout
    without developer mode).  The action taken is recorded, never guessed.
    """
    try:
        target.symlink_to(source)
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(source, target)
        return "hardlink"
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(source, target)
        return "copy"


def prepare_data_dirs(template: Path, run_root: Path, instances: int) -> dict:
    """Create one isolated writable data dir per instance; never overwrite.

    ``th08.dat``/``thbgm.dat`` are read-only links into the template (they are
    large and never written); ``th08.cfg``/``score.dat`` are per-instance copies
    because the engine writes them and a shared copy crashed N>=8 workers before
    (EXP-P0-003 finding 2).
    """
    if instances < 1:
        raise ValueError("instances must be positive")
    template = Path(template)
    if not (template / "th08.dat").is_file():
        raise ProbeUnavailable(f"template has no th08.dat: {template}")
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    dirs: list[Path] = []
    actions: list[dict] = []
    for index in range(instances):
        target = run_root / f"i-{index}"
        if target.exists():
            raise ProbeUnavailable(f"refusing to reuse existing instance dir: {target}")
        target.mkdir(parents=True)
        for name in DATA_FILES_LINK:
            source = template / name
            if source.is_file():
                action = _link_readonly_file(source, target / name)
                actions.append({"dir": str(target), "file": name, "action": action,
                                "source": str(source)})
        for name in DATA_FILES_COPY:
            source = template / name
            if source.is_file():
                shutil.copy2(source, target / name)
                actions.append({"dir": str(target), "file": name, "action": "copy",
                                "source": str(source)})
        dirs.append(target)
    return {"run_root": str(run_root), "dirs": [str(d) for d in dirs], "actions": actions}


# ----------------------------------------------------------------- protocol IO
@dataclass
class _Child:
    """One ``--rl-control`` child; line protocol over its stdin/stdout pipes."""

    binary: str
    data_dir: str
    env: dict[str, str]
    timeout: float = 120.0
    proc: subprocess.Popen | None = None
    buffer: bytes = b""
    stderr_tail: bytearray = field(default_factory=bytearray)
    _err_thread: threading.Thread | None = None

    def start(self) -> None:
        if self.proc is not None:
            raise RuntimeError("already started")
        self.proc = subprocess.Popen(
            [self.binary, "--data-dir", self.data_dir,
             "--rl-control", "0,1", "--rl-role", "env"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.env, cwd=self.data_dir, bufsize=0,
        )
        self._err_thread = threading.Thread(target=self._drain, name="stderr", daemon=True)
        self._err_thread.start()

    def _drain(self) -> None:
        stream = self.proc.stderr if self.proc is not None else None
        while stream is not None:
            chunk = stream.read(4096)
            if not chunk:
                return
            self.stderr_tail.extend(chunk)
            if len(self.stderr_tail) > 8192:
                del self.stderr_tail[:-8192]

    def _readline(self, timeout: float) -> str:
        assert self.proc is not None and self.proc.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if b"\n" in self.buffer:
                line, _, self.buffer = self.buffer.partition(b"\n")
                return line.decode("utf-8", "replace")
            ready, _, _ = select.select([self.proc.stdout], [], [], 0.25)
            if not ready:
                if self.proc.poll() is not None:
                    raise ProbeUnavailable(
                        f"harness exited before replying (code {self.proc.returncode})"
                    )
                continue
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise ProbeUnavailable("harness closed the control pipe")
            self.buffer += chunk
        raise ProbeUnavailable(f"no reply within {timeout:.1f}s")

    def request(self, line: str, timeout: float | None = None) -> dict[str, str]:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()
        reply = self._readline(self.timeout if timeout is None else timeout)
        parsed = parse_reply(reply)
        if parsed.get("_verb") != "OK":
            raise ProbeUnavailable(f"{line!r} -> {reply!r}")
        return parsed

    def maps_text(self) -> str:
        """Driver map for the live child; empty when it already exited."""
        if self.proc is None:
            return ""
        try:
            return Path(f"/proc/{self.proc.pid}/maps").read_text(errors="replace")
        except OSError:
            return ""

    def environ_subset(self) -> dict[str, str]:
        if self.proc is None:
            return {}
        try:
            raw = Path(f"/proc/{self.proc.pid}/environ").read_bytes()
        except OSError:
            return {}
        wanted = set(GL_KNOBS) | {"SDL_AUDIODRIVER"}
        out = {}
        for item in raw.split(b"\0"):
            if b"=" not in item:
                continue
            key, _, value = item.partition(b"=")
            k = key.decode("utf-8", "replace")
            if k in wanted:
                out[k] = value.decode("utf-8", "replace")
        return out

    def close(self, *, quit_timeout: float = 20.0) -> int | None:
        proc = self.proc
        if proc is None:
            return None
        try:
            if proc.poll() is None:
                try:
                    self.request("QUIT", timeout=quit_timeout)
                except Exception:
                    pass
            try:
                proc.wait(timeout=quit_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
        return proc.returncode


def parse_reply(reply: str) -> dict[str, str]:
    """``OK a=1 b=2`` -> ``{"_verb": "OK", "a": "1", "b": "2"}``."""
    parts = reply.split()
    out: dict[str, str] = {"_verb": parts[0] if parts else ""}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


# ------------------------------------------------------------------ one worker
def measure_instance(index: int, args, data_dir: Path, env: dict[str, str],
                     barrier: threading.Barrier, probe_path: Path | None = None) -> dict:
    """Drive one harness process through warmup + timed window."""
    child_env = dict(env)
    if probe_path is not None:
        child_env["LD_PRELOAD"] = args.gl_probe_shim
        child_env["TH08_GL_PROBE_OUT"] = str(probe_path)
    child = _Child(binary=args.binary, data_dir=str(data_dir), env=child_env,
                   timeout=args.timeout)
    record: dict = {
        "instance": index, "data_dir": str(data_dir), "pid": None,
        "role": None, "threads": None, "setup_frames": None,
        "frames": 0, "step_seconds": 0.0, "wall_seconds": 0.0,
        "frames_per_second": 0.0, "readback_frames": 0, "readback_micros": 0,
        "readback_us_per_frame": None, "resets": 0, "reset_seconds": 0.0,
        "game_over_events": 0, "exit_code": None, "teardown_crash": False,
        "maps": {}, "environ": {}, "stderr_tail": "", "error": None,
        "phase_start": None, "phase_end": None,
    }
    try:
        child.start()
        record["pid"] = child.proc.pid if child.proc else None
        hello = child.request("HELLO")
        record["role"] = hello.get("role")
        if hello.get("role") != "env":
            raise ProbeUnavailable(f"role is {hello.get('role')!r}, expected env")
        debug = child.request("DEBUG")
        record["threads"] = int(debug.get("threads", "-1"))
        reset = child.request(f"RESET 0 1 0 {args.seed_base + index} 0",
                              timeout=args.timeout)
        if reset.get("ready") != "1":
            raise ProbeUnavailable(f"RESET did not become ready: {reset}")
        record["setup_frames"] = int(reset.get("setup_frames", "0"))
        child.request(f"READBACK {1 if args.readback else 0}")
        if args.warmup_frames:
            child.request(f"STEP 0 {args.warmup_frames}", timeout=args.timeout)
        maps_text = child.maps_text()
        record["maps"] = classify_gl(maps_text)
        record["environ"] = child.environ_subset()

        before = child.request("READBACK")
        rb_frames_before = int(before.get("frames", "0"))
        rb_micros_before = int(before.get("micros", "0"))

        barrier.wait(timeout=args.timeout)
        start = time.perf_counter()
        record["phase_start"] = start
        deadline = start + args.seconds
        frames = 0
        step_seconds = 0.0
        resets = 0
        reset_seconds = 0.0
        game_over = 0
        chunk = args.chunk
        while time.perf_counter() < deadline:
            sent = time.perf_counter()
            reply = child.request(f"STEP 0 {chunk}", timeout=args.timeout)
            took = time.perf_counter() - sent
            step_seconds += took
            advanced = int(reply.get("advanced", "0"))
            frames += advanced
            # Keep each STEP near --chunk-seconds so the deadline is not
            # overshot by several seconds when an instance is slow (which would
            # stretch the shared window and understate the whole-machine rate).
            if advanced > 0 and took > 0:
                rate = advanced / took
                chunk = max(1, min(args.chunk, int(rate * args.chunk_seconds) or 1))
            if reply.get("exit") == "1":
                game_over += 1
                if time.perf_counter() >= deadline:
                    break
                # A death ends the run; re-arm and keep counting real frames.
                sent_reset = time.perf_counter()
                again = child.request(f"RESET 0 1 0 {args.seed_base + index + 1} 0",
                                      timeout=args.timeout)
                reset_seconds += time.perf_counter() - sent_reset
                resets += 1
                if again.get("ready") != "1":
                    break
            if advanced <= 0:
                break
        end = time.perf_counter()
        record["phase_end"] = end
        after = child.request("READBACK")
        rb_frames_after = int(after.get("frames", "0"))
        rb_micros_after = int(after.get("micros", "0"))
        record.update(
            frames=frames, step_seconds=step_seconds, wall_seconds=end - start,
            resets=resets, reset_seconds=reset_seconds, game_over_events=game_over,
            readback_frames=rb_frames_after - rb_frames_before,
            readback_micros=rb_micros_after - rb_micros_before,
        )
        if frames > 0 and step_seconds > 0:
            record["frames_per_second"] = compute_throughput(frames, step_seconds)
        if record["readback_frames"] > 0:
            record["readback_us_per_frame"] = (record["readback_micros"]
                                               / record["readback_frames"])
        # Loaded drivers can appear lazily; re-read after the timed window.
        record["maps"] = {k: (v or record["maps"].get(k, False))
                          for k, v in classify_gl(child.maps_text()).items()}
    except BaseException as exc:  # keep workers independent; report, never guess
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            code = child.close()
            record["exit_code"] = code
            record["teardown_crash"] = code not in (0, None)
            # SIGSEGV while the engine tears its GL context down is a known
            # WSLg/D3D12 defect; it is reported, never hidden.
            if code is not None and code < 0:
                record["teardown_signal"] = -code
        except Exception as exc:
            record["error"] = record["error"] or f"close {type(exc).__name__}: {exc}"
        record["stderr_tail"] = bytes(child.stderr_tail).decode("utf-8", "replace")
        if probe_path is not None and probe_path.is_file():
            record["gl"] = parse_gl_fingerprint(
                probe_path.read_text(errors="replace"), record["pid"])
            record["gl"]["source"] = "in-process glGetString via LD_PRELOAD shim"
    return record


# ------------------------------------------------------------------ GPU sample
class GpuSampler:
    """Cheap real ``nvidia-smi`` utilisation samples during the timed window."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = shutil.which("nvidia-smi") is not None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=False)
                if out.returncode == 0 and out.stdout.strip():
                    for line in out.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            self.samples.append({"util_pct": float(parts[0]),
                                                 "mem_mib": float(parts[1])})
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        if self.available:
            self._thread = threading.Thread(target=self._run, name="gpu", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        util = [s["util_pct"] for s in self.samples]
        mem = [s["mem_mib"] for s in self.samples]
        return {
            "available": self.available,
            "samples": len(util),
            "util_pct_mean": (sum(util) / len(util)) if util else None,
            "util_pct_max": max(util) if util else None,
            "mem_mib_max": max(mem) if mem else None,
        }


# ------------------------------------------------------------------- glx probe
_RENDERER_RE = re.compile(r"OpenGL renderer string:\s*(.+)")
_ACCEL_RE = re.compile(r"Accelerated:\s*(\w+)")


def parse_glxinfo(text: str) -> dict:
    """Extract the renderer and acceleration flag from ``glxinfo -B``."""
    renderer = _RENDERER_RE.search(text)
    accel = _ACCEL_RE.search(text)
    return {
        "renderer": renderer.group(1).strip() if renderer else None,
        "accelerated": (accel.group(1).strip().lower() == "yes") if accel else None,
        "d3d12": "D3D12" in text,
        "llvmpipe": "llvmpipe" in text.lower(),
    }


def probe_gl_backend(display: str | None, backend: str, lp_threads: int) -> dict:
    """Run ``glxinfo -B`` under the same environment the children would get."""
    if shutil.which("glxinfo") is None:
        return {"available": False, "reason": "glxinfo not installed"}
    env = build_child_env(backend, display=display, lp_threads=lp_threads)
    out = subprocess.run(["glxinfo", "-B"], capture_output=True, text=True,
                         env=env, timeout=120, check=False)
    parsed = parse_glxinfo(out.stdout)
    parsed.update({"available": True, "display": display, "backend": backend,
                   "exit_code": out.returncode, "stdout": out.stdout,
                   "stderr": out.stderr})
    return parsed


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, help="Linux ELF th08-modern")
    parser.add_argument("--data-template", required=True, help="dir with th08.dat")
    parser.add_argument("--run-root", required=True,
                        help="fresh root for per-instance data dirs (never reused)")
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--chunk", type=int, default=64, help="max frames per STEP")
    parser.add_argument("--chunk-seconds", type=float, default=0.25,
                        help="target duration per STEP; the chunk shrinks for "
                             "slow instances so the timed window stays tight")
    parser.add_argument("--seed-base", type=int, default=1)
    parser.add_argument("--backend", choices=("software", "hardware"), required=True)
    parser.add_argument("--display", default=None)
    parser.add_argument("--lp-threads", type=int, default=1)
    parser.add_argument("--readback", type=int, choices=(0, 1), default=1,
                        help="1 = per-frame glReadPixels 640x480 RGBA (the gate)")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--gl-probe-shim", default=None,
                        help="LD_PRELOAD .so that logs the engine's own "
                             "glGetString fingerprints (proves the backend)")
    parser.add_argument("--expect-renderer", default=None,
                        help="substring the captured GL_RENDERER must contain; "
                             "fails closed when it does not")
    parser.add_argument("--allow-teardown-crash", action="store_true",
                        help="keep the measurement valid when the engine only "
                             "crashes after QUIT (WSLg/D3D12 teardown defect)")
    parser.add_argument("--child-env", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="extra environment for every harness child, recorded "
                             "in the output document (e.g. TH08_ASYNC_READBACK=1); "
                             "GL backend knobs are refused")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.instances < 1:
        parser.error("--instances must be positive")
    if not args.seconds > 0:
        parser.error("--seconds must be positive")
    if args.chunk < 1:
        parser.error("--chunk must be positive")
    if not os.path.exists(args.binary):
        parser.error(f"binary not found: {args.binary}")

    started_at = time.time()
    doc: dict = {
        "label": args.label, "backend": args.backend, "display": args.display,
        "instances_requested": args.instances, "requested_seconds": args.seconds,
        "warmup_frames": args.warmup_frames, "chunk_frames": args.chunk,
        "chunk_seconds": args.chunk_seconds,
        "readback": bool(args.readback), "pixel_size": [PIXEL_WIDTH, PIXEL_HEIGHT],
        "pixel_bytes": PIXEL_BYTES, "binary": args.binary,
        "binary_sha256": sha256_of(args.binary),
        "data_template": args.data_template, "run_root": args.run_root,
        "lp_threads": args.lp_threads, "argv": sys.argv[1:],
        "python": sys.version.split()[0], "host": os.uname().nodename,
        "nproc": os.cpu_count(),
        "host_state_before": read_loadavg(),
    }
    status = "invalid"
    try:
        env = build_child_env(args.backend, display=args.display,
                              lp_threads=args.lp_threads)
        overlay = parse_child_env(args.child_env)
        env.update(overlay)
        doc["extra_child_env"] = overlay
        doc["child_env"] = {k: env.get(k) for k in
                            sorted(set(GL_KNOBS) | {"SDL_AUDIODRIVER"})}
        doc["glxinfo"] = probe_gl_backend(args.display, args.backend, args.lp_threads)
        prep = prepare_data_dirs(Path(args.data_template), Path(args.run_root),
                                args.instances)
        doc["data_dirs"] = prep
        barrier = threading.Barrier(args.instances)
        results: list[dict] = []
        errors: list[BaseException] = []

        def probe_path_for(index: int) -> Path | None:
            if not args.gl_probe_shim:
                return None
            return Path(prep["dirs"][index]) / "gl_probe.txt"

        def worker(index: int) -> None:
            try:
                results.append(measure_instance(index, args, Path(prep["dirs"][index]),
                                                env, barrier, probe_path_for(index)))
            except BaseException as exc:  # surface, do not abort siblings
                errors.append(exc)

        with GpuSampler() as sampler:
            threads = [threading.Thread(target=worker, args=(i,), name=f"inst-{i}")
                       for i in range(args.instances)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        doc["gpu"] = sampler.summary()
        doc["host_state_after"] = read_loadavg()
        results.sort(key=lambda r: r["instance"])
        doc["results"] = results
        if errors:
            doc["worker_errors"] = [f"{type(e).__name__}: {e}" for e in errors]

        broken = [r for r in results if r["error"] or r["frames"] <= 0
                  or r["frames_per_second"] <= 0]
        fingerprinted = [r for r in results if r.get("gl")]
        renderer_bad = [r["instance"] for r in results
                        if not renderer_matches(args.expect_renderer, r.get("gl") or {})]
        missing_fingerprint = [r["instance"] for r in results if not r.get("gl")]
        env_proof_bad = []
        for r in results:
            got = r.get("environ") or {}
            if args.backend == "software" and got.get("LIBGL_ALWAYS_SOFTWARE") != "1":
                env_proof_bad.append(r["instance"])
            if args.backend == "hardware" and got.get("LIBGL_ALWAYS_SOFTWARE") is not None:
                env_proof_bad.append(r["instance"])
        crash_before_measure = [r["instance"] for r in results
                                if r["teardown_crash"] and r["frames"] <= 0]
        doc["backend_proof"] = {
            "mode": "in_process_gl_renderer" if fingerprinted else "environment_only",
            "expect_renderer": args.expect_renderer,
            "fingerprints": [{"instance": r["instance"], "gl": r.get("gl")}
                             for r in results],
            "environ_check": "ok" if not env_proof_bad else f"bad:{env_proof_bad}",
            "note": "/proc/<pid>/maps is recorded but not used as proof: this Mesa "
                    "build hardlinks every gallium driver into one file, so a "
                    "D3D12 context can legitimately map swrast_dri.so.",
        }
        if broken:
            status = "invalid"
            doc["invalid_reason"] = f"instances without a valid measurement: " \
                                    f"{[(r['instance'], r['error']) for r in broken]}"
        elif env_proof_bad:
            status = "invalid"
            doc["invalid_reason"] = (
                f"child environment does not match backend {args.backend!r} for "
                f"instances {env_proof_bad}")
        elif args.expect_renderer and missing_fingerprint:
            status = "invalid"
            doc["invalid_reason"] = (
                f"no in-process GL fingerprint for instances {missing_fingerprint}; "
                f"the {args.backend!r} backend is unproven")
        elif renderer_bad:
            status = "invalid"
            doc["invalid_reason"] = (
                f"GL_RENDERER does not contain {args.expect_renderer!r} for "
                f"instances {renderer_bad}: "
                f"{[((r.get('gl') or {}).get('GL_RENDERER')) for r in results]}")
        elif crash_before_measure:
            status = "invalid"
            doc["invalid_reason"] = f"teardown crash before any frame: {crash_before_measure}"
        else:
            status = "measured"
            starts = [r["phase_start"] for r in results if r["phase_start"]]
            ends = [r["phase_end"] for r in results if r["phase_end"]]
            window = (max(ends) - min(starts)) if starts and ends else 0.0
            doc["window"] = {"start": min(starts) if starts else None,
                             "end": max(ends) if ends else None,
                             "seconds": window}
            doc["aggregate"] = aggregate_throughput(results, window_seconds=window)
            doc["gate"] = gate_verdict(doc["aggregate"]["frames_per_second"],
                                       readback=bool(args.readback))
            readbacks = [r["readback_us_per_frame"] for r in results
                         if r["readback_us_per_frame"] is not None]
            if readbacks:
                mean_us = sum(readbacks) / len(readbacks)
                doc["readback_cost"] = {
                    "us_per_frame_mean": mean_us,
                    "ms_per_frame_mean": mean_us / 1000.0,
                    "single_instance_ceiling_fps_by_readback": 1_000_000.0 / mean_us,
                    "source": "engine READBACK counters (glReadPixels only)",
                }
            teardown = [r["instance"] for r in results if r["teardown_crash"]]
            doc["teardown_crash_instances"] = teardown
            doc["exit_codes"] = [r["exit_code"] for r in results]
            if teardown and not args.allow_teardown_crash:
                doc["warnings"] = doc.get("warnings", []) + [
                    f"engine crashed after QUIT on instances {teardown} "
                    f"(exit codes {[r['exit_code'] for r in results]}); the timed "
                    f"window had already completed"]
    except ProbeUnavailable as exc:
        doc["invalid_reason"] = str(exc)
        status = "unavailable"
    except Exception as exc:
        doc["invalid_reason"] = f"{type(exc).__name__}: {exc}"
        status = "unavailable"
    doc["status"] = status
    doc["wall_seconds"] = time.time() - started_at
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    summary = {
        "status": status, "backend": args.backend, "display": args.display,
        "instances": args.instances, "readback": bool(args.readback),
        "seconds": args.seconds,
        "frames_per_second": (doc.get("aggregate") or {}).get("frames_per_second"),
        "per_instance_fps": [round(r["frames_per_second"], 2) for r in doc.get("results", [])],
        "readback_ms_per_frame": (doc.get("readback_cost") or {}).get("ms_per_frame_mean"),
        "gl_renderers": sorted({((r.get("gl") or {}).get("GL_RENDERER") or "unproven")
                                for r in doc.get("results", [])}),
        "gate_passed": (doc.get("gate") or {}).get("passed"),
        "invalid_reason": doc.get("invalid_reason"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if status == "measured" else 3


def sha256_of(path: str) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
