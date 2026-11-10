"""Byte-level pixel-consistency check across ``LP_NUM_THREADS`` values.

The G0.3 gate is met (or missed) by *rendering*; but the only reason a higher
``LP_NUM_THREADS`` is even admissible is that llvmpipe must still produce the
same pixels.  Multi-threaded rasterisation is allowed to change pixels in
principle, so "LP=4 is faster" is worthless for training until somebody proves
the framebuffer did not change.  This tool answers exactly that question on the
real engine and nothing else:

* it launches the real ``th08-modern`` ELF with ``--rl-control 0,1 --rl-role env``
  (the only role that renders and reads pixels back) under a fixed
  ``LP_NUM_THREADS`` per *arm*;
* every instance replays the identical deterministic trajectory
  (same ``RESET`` seed, same one-mask-per-frame list) for ``--frames`` frames;
* per frame it records the engine's own ``PIXELS`` reply (640x480, nonzero count,
  FNV-1a checksum) **and** the SHA-256 of the raw RGBA bytes taken from the
  POSIX shared-memory segment (``SHM``), i.e. the bytes themselves, not a PNG,
  not a checksum substituted for bytes;
* it then compares arms frame by frame **byte for byte**.  When bytes differ it
  reports the first differing frame, how many bytes/pixels differ there and the
  bounding box of the difference, so a HUD-only wall-clock artefact can be told
  apart from a real raster change.  Nothing is masked and nothing is excluded
  from the comparison.

``--instances`` per arm is deliberate: the same arm is run as N concurrent
processes, so run-to-run noise inside one configuration is measured too.  A
difference between arms is only meaningful if it exceeds that noise floor.

Runs inside WSL (Linux), where the harness binary lives::

    /root/v400-venv/bin/python tools/rl/v400_render_pixel_consistency.py \
        --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern \
        --data-template /root/v400-data/game \
        --run-root /root/v400-pix-1 --display :98 \
        --arm lp1:1 --arm lp4:4 --instances 2 --frames 100 \
        --gl-probe-shim /tmp/libgl_probe.so --expect-renderer llvmpipe \
        --out .tmp_v400_threads/pixel-consistency.json

Exit codes: ``0`` = valid measurement, pixels byte-identical everywhere;
``4`` = valid measurement but at least one comparison differs (a real result,
not an error); ``3`` = unavailable/invalid, nothing reported as evidence;
``2`` = bad usage.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

PIXEL_WIDTH = 640
PIXEL_HEIGHT = 480
PIXEL_BYTES = PIXEL_WIDTH * PIXEL_HEIGHT * 4
SHM_HEADER_SIZE = 192
SHM_BYTES = SHM_HEADER_SIZE + PIXEL_BYTES
SHM_MAGIC = 0x30303456
SHM_VERSION = 1
SHM_PIXEL_VALID = 2
HEADER = struct.Struct("<IIQQIIII")
ROW_BYTES = PIXEL_WIDTH * 4

GL_KNOBS = ("LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER", "LP_NUM_THREADS",
            "MESA_LOADER_DRIVER_OVERRIDE", "LIBGL_ALWAYS_INDIRECT", "DISPLAY",
            "WAYLAND_DISPLAY")
DATA_FILES_LINK = ("th08.dat", "thbgm.dat")
DATA_FILES_COPY = ("th08.cfg", "score.dat")

# Byte value -> "nonzero" flag table used to turn a difference mask into a
# bounding box without a Python-level loop over 1.2 million bytes.
_NONZERO_TABLE = bytes([0]) + bytes([255]) * 255


class PixelCheckUnavailable(RuntimeError):
    """The comparison cannot be made valid; never report a verdict instead."""


# ---------------------------------------------------------------- pure helpers
def hostname() -> str:
    """Host name on Linux (WSL) and on the Windows side where tests run."""
    try:
        return os.uname().nodename
    except AttributeError:
        return os.environ.get("COMPUTERNAME") or "unknown"


def sha256_bytes(raw: bytes) -> str:
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def parse_reply(reply: str) -> dict[str, str]:
    """``OK a=1 b=2`` -> ``{"_verb": "OK", "a": "1", "b": "2"}``."""
    parts = reply.split()
    out: dict[str, str] = {"_verb": parts[0] if parts else ""}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def parse_arm_specs(specs: list[str] | None) -> list[dict]:
    """Parse ``NAME:LP_THREADS`` into arms; names must be unique."""
    arms: list[dict] = []
    seen: set[str] = set()
    for spec in specs or []:
        name, separator, value = spec.partition(":")
        name = name.strip()
        if not separator or not name:
            raise PixelCheckUnavailable(f"--arm must be NAME:LP_THREADS, got {spec!r}")
        if name in seen:
            raise PixelCheckUnavailable(f"duplicate arm name: {name!r}")
        try:
            threads = int(value)
        except ValueError:
            raise PixelCheckUnavailable(
                f"--arm {name!r} has a non-integer thread count: {value!r}") from None
        if threads < 1:
            raise PixelCheckUnavailable(f"--arm {name!r} must have LP_THREADS >= 1")
        seen.add(name)
        arms.append({"name": name, "lp_threads": threads})
    if not arms:
        raise PixelCheckUnavailable("at least one --arm is required")
    return arms


def masks_for(frames: int, mask: int = 0) -> list[int]:
    """One input mask per frame; a constant mask is the reproducible trajectory."""
    if frames < 1:
        raise ValueError("frames must be positive")
    return [int(mask)] * frames


def validate_shm_payload(raw: bytes, *, expected_frame: int) -> dict:
    """Check the SHM header of one captured frame and return its fields.

    The raw RGBA bytes live at offset 192; a header that disagrees with the
    expected frame number means the segment was not published for this frame,
    which would silently compare the wrong picture -- so it fails closed.
    """
    if len(raw) != SHM_BYTES:
        raise PixelCheckUnavailable(
            f"pixel SHM size is {len(raw)}, expected {SHM_BYTES}")
    magic, version, frame, _reserved, width, height, _stride, flags = \
        HEADER.unpack_from(raw)
    if magic != SHM_MAGIC:
        raise PixelCheckUnavailable(f"pixel SHM magic is {magic:#x}, expected {SHM_MAGIC:#x}")
    if version != SHM_VERSION:
        raise PixelCheckUnavailable(f"pixel SHM version is {version}, expected {SHM_VERSION}")
    if (width, height) != (PIXEL_WIDTH, PIXEL_HEIGHT):
        raise PixelCheckUnavailable(
            f"pixel SHM is {width}x{height}, expected {PIXEL_WIDTH}x{PIXEL_HEIGHT}")
    if frame != expected_frame:
        raise PixelCheckUnavailable(
            f"pixel SHM header frame is {frame}, expected {expected_frame}")
    if not flags & SHM_PIXEL_VALID:
        raise PixelCheckUnavailable(f"pixel SHM pixel-valid flag is unset (flags={flags:#x})")
    return {"magic": magic, "version": version, "frame": frame,
            "width": width, "height": height, "flags": flags}


class ByteStore:
    """Content-addressed raw RGBA cache: identical frames are stored once."""

    def __init__(self) -> None:
        self._blobs: dict[str, bytes] = {}
        self.unique = 0
        self.puts = 0

    def put(self, raw: bytes) -> str:
        self.puts += 1
        digest = sha256_bytes(raw)
        if digest not in self._blobs:
            self._blobs[digest] = bytes(raw)
            self.unique += 1
        return digest

    def get(self, digest: str) -> bytes:
        try:
            return self._blobs[digest]
        except KeyError:
            raise PixelCheckUnavailable(
                f"raw RGBA for {digest} was never captured") from None

    def __len__(self) -> int:
        return len(self._blobs)


def diff_bbox(mask: bytes) -> tuple[int, int, int, int] | None:
    """Pixel bounding box (min_x, min_y, max_x, max_y) of nonzero RGBA bytes.

    ``mask`` is a difference mask with 640*480*4 bytes in row-major RGBA order.
    A differing byte makes its whole pixel "different".  Returns ``None`` when
    nothing differs.  Implemented with ``bytes`` primitives so a 1.2 MB frame
    costs milliseconds, not a Python loop over every byte.
    """
    if len(mask) != PIXEL_BYTES:
        raise ValueError(f"mask is {len(mask)} bytes, expected {PIXEL_BYTES}")
    nonzero = mask.translate(_NONZERO_TABLE)
    min_x = min_y = None
    max_x = max_y = -1
    for row in range(PIXEL_HEIGHT):
        chunk = nonzero[row * ROW_BYTES:(row + 1) * ROW_BYTES]
        if chunk.count(0) == ROW_BYTES:  # no differing byte on this row
            continue
        first = chunk.index(255) // 4
        last = chunk.rindex(255) // 4
        if min_x is None:
            min_x, min_y = first, row
        else:
            min_x = min(min_x, first)
        max_x = max(max_x, last)
        max_y = row
    if min_x is None:
        return None
    return (min_x, min_y, max_x, max_y)


def diff_rgba(left: bytes, right: bytes) -> dict:
    """Byte-exact difference between two raw RGBA frames.

    The fast path is a single C-level comparison; only genuinely different
    frames pay for the bounding box, and even then nothing is masked or
    excluded -- the caller sees the whole truth.
    """
    if len(left) != len(right):
        raise ValueError(f"frame sizes differ: {len(left)} vs {len(right)}")
    if len(left) != PIXEL_BYTES:
        raise ValueError(f"frame is {len(left)} bytes, expected {PIXEL_BYTES}")
    if left == right:
        return {"identical": True, "differing_bytes": 0, "differing_pixels": 0,
                "bbox": None, "sha256": sha256_bytes(left)}
    mask = (int.from_bytes(left, "big") ^ int.from_bytes(right, "big")).to_bytes(
        PIXEL_BYTES, "big")
    differing_bytes = PIXEL_BYTES - mask.count(0)
    bbox = diff_bbox(mask)
    differing_pixels = 0
    if bbox is not None:
        min_x, min_y, max_x, max_y = bbox
        for row in range(min_y, max_y + 1):
            chunk = mask[row * ROW_BYTES + min_x * 4:
                         row * ROW_BYTES + (max_x + 1) * 4]
            differing_pixels += _count_differing_pixels(chunk)
    return {"identical": False, "differing_bytes": differing_bytes,
            "differing_pixels": differing_pixels, "bbox": bbox,
            "sha256": sha256_bytes(right)}


def _count_differing_pixels(chunk: bytes) -> int:
    """Count changed pixels in a run of complete RGBA pixels."""
    if len(chunk) % 4:
        raise ValueError("pixel run is not a whole number of RGBA pixels")
    count = 0
    for index in range(0, len(chunk), 4):
        if chunk[index] or chunk[index + 1] or chunk[index + 2] or chunk[index + 3]:
            count += 1
    return count


def compare_captures(left: dict, right: dict, store: ByteStore,
                     *, max_detail: int = 10) -> dict:
    """Compare two captures frame by frame, byte for byte.

    Both captures carry per-frame ``rgba_sha256`` plus ``pixels_checksum`` from
    the engine.  Hashes decide identity cheaply; genuinely different frames are
    resolved down to bytes, first differing frame and difference size.
    """
    left_frames = list(left.get("frames") or [])
    right_frames = list(right.get("frames") or [])
    compared = min(len(left_frames), len(right_frames))
    checksum_mismatch = []
    differing = []
    union: list[int] | None = None
    for index in range(compared):
        a, b = left_frames[index], right_frames[index]
        if a.get("pixels_checksum") != b.get("pixels_checksum"):
            checksum_mismatch.append(a.get("frame", index + 1))
        if a["rgba_sha256"] == b["rgba_sha256"]:
            continue
        detail = diff_rgba(store.get(a["rgba_sha256"]), store.get(b["rgba_sha256"]))
        entry = {"frame": a.get("frame", index + 1),
                 "left_rgba_sha256": a["rgba_sha256"],
                 "right_rgba_sha256": b["rgba_sha256"],
                 "differing_bytes": detail["differing_bytes"],
                 "differing_pixels": detail["differing_pixels"],
                 "bbox": list(detail["bbox"]) if detail["bbox"] else None}
        differing.append(entry)
        if detail["bbox"] is not None:
            min_x, min_y, max_x, max_y = detail["bbox"]
            if union is None:
                union = [min_x, min_y, max_x, max_y]
            else:
                union = [min(union[0], min_x), min(union[1], min_y),
                         max(union[2], max_x), max(union[3], max_y)]
    first = differing[0] if differing else None
    return {
        "left": left.get("tag"), "right": right.get("tag"),
        "left_lp_threads": left.get("lp_threads"),
        "right_lp_threads": right.get("lp_threads"),
        "frames_compared": compared,
        "frames_missing": abs(len(left_frames) - len(right_frames)),
        "identical": not differing,
        "different_frame_count": len(differing),
        "first_different_frame": (first or {}).get("frame"),
        "first_different_bytes": (first or {}).get("differing_bytes"),
        "first_different_pixels": (first or {}).get("differing_pixels"),
        "first_bbox": (first or {}).get("bbox"),
        "union_bbox": union,
        "checksum_mismatch_frames": checksum_mismatch,
        "differing_frames": differing[:max_detail],
        "differing_frames_truncated": max(0, len(differing) - max_detail),
    }


def summarise(comparisons: list[dict]) -> dict:
    """Turn the comparison table into the one verdict the card needs."""
    cross = [c for c in comparisons if c.get("left_lp_threads") != c.get("right_lp_threads")]
    within = [c for c in comparisons if c.get("left_lp_threads") == c.get("right_lp_threads")]
    asymmetric = [c for c in comparisons if c.get("frames_missing")]
    return {
        "comparisons": len(comparisons),
        "cross_lp_comparisons": len(cross),
        "within_lp_comparisons": len(within),
        "cross_lp_identical": bool(cross) and all(c["identical"] for c in cross),
        "within_lp_identical": bool(within) and all(c["identical"] for c in within),
        "noise_floor_different_frames": sum(c["different_frame_count"] for c in within),
        "max_differing_bytes": max([c.get("first_different_bytes") or 0 for c in comparisons]
                                   or [0]),
        "union_bbox": next((c["union_bbox"] for c in comparisons if c.get("union_bbox")),
                           None),
        "frames_compared_per_pair": sorted({c["frames_compared"] for c in comparisons}),
        "incomplete_comparisons": len(asymmetric),
        "lp_invariant_bytes": (bool(cross) and all(c["identical"] for c in cross)
                               and not asymmetric),
    }


def exit_code_for(status: str, verdict: dict | None) -> int:
    """Map a finished report onto the documented exit codes."""
    if status != "measured":
        return 3
    return 0 if (verdict or {}).get("lp_invariant_bytes") else 4


def backend_conflict(backend: str, knob: str, value: str | None) -> bool:
    """True when a surviving GL knob would falsify the declared backend."""
    if value is None:
        return False
    if backend == "hardware":
        return knob in ("LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER",
                        "MESA_LOADER_DRIVER_OVERRIDE")
    return False


def build_arm_env(backend: str, *, display: str | None, lp_threads: int,
                  base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for one instance: backend overlay + this arm's thread count."""
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
        if backend_conflict(backend, knob, env.get(knob)):
            raise PixelCheckUnavailable(
                f"backend {backend!r} cannot be guaranteed: {knob}={env.get(knob)!r}")
    return env


def parse_gl_fingerprint(text: str, pid: int | None = None) -> dict:
    """Parse the LD_PRELOAD shim log into one renderer fingerprint."""
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
    if not expected:
        return True
    return expected.lower() in ((fingerprint or {}).get("GL_RENDERER") or "").lower()


def prepare_data_dirs(template: Path, run_root: Path, labels: list[str]) -> dict:
    """One isolated writable data dir per (arm, instance); never reuse one."""
    template = Path(template)
    if not (template / "th08.dat").is_file():
        raise PixelCheckUnavailable(f"template has no th08.dat: {template}")
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    dirs: list[str] = []
    actions: list[dict] = []
    for label in labels:
        target = run_root / label
        if target.exists():
            raise PixelCheckUnavailable(f"refusing to reuse existing instance dir: {target}")
        target.mkdir(parents=True)
        for name in DATA_FILES_LINK:
            source = template / name
            if source.is_file():
                try:
                    (target / name).symlink_to(source)
                    action = "symlink"
                except (OSError, NotImplementedError):
                    try:
                        os.link(source, target / name)
                        action = "hardlink"
                    except (OSError, NotImplementedError, AttributeError):
                        shutil.copy2(source, target / name)
                        action = "copy"
                actions.append({"dir": str(target), "file": name, "action": action})
        for name in DATA_FILES_COPY:
            source = template / name
            if source.is_file():
                shutil.copy2(source, target / name)
                actions.append({"dir": str(target), "file": name, "action": "copy"})
        dirs.append(str(target))
    return {"run_root": str(run_root), "dirs": dirs, "actions": actions}


# ----------------------------------------------------------------- protocol IO
class _Child:
    """One ``--rl-control`` child; line protocol over its stdin/stdout pipes."""

    def __init__(self, binary: str, data_dir: str, env: dict[str, str],
                 timeout: float = 60.0) -> None:
        self.binary = binary
        self.data_dir = data_dir
        self.env = env
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None
        self.buffer = b""
        self.stderr_tail = bytearray()
        self._err_thread: threading.Thread | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [self.binary, "--data-dir", self.data_dir,
             "--rl-control", "0,1", "--rl-role", "env"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.env, cwd=self.data_dir, bufsize=0)
        self._err_thread = threading.Thread(target=self._drain, daemon=True)
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
                    raise PixelCheckUnavailable(
                        f"harness exited before replying (code {self.proc.returncode})")
                continue
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise PixelCheckUnavailable("harness closed the control pipe")
            self.buffer += chunk
        raise PixelCheckUnavailable(f"no reply within {timeout:.1f}s")

    def request(self, line: str, timeout: float | None = None) -> dict[str, str]:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()
        reply = self._readline(self.timeout if timeout is None else timeout)
        parsed = parse_reply(reply)
        if parsed.get("_verb") != "OK":
            raise PixelCheckUnavailable(f"{line!r} -> {reply!r}")
        return parsed

    def close(self, *, quit_timeout: float = 30.0) -> int | None:
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


# ------------------------------------------------------------------ one worker
def capture_instance(arm: dict, index: int, args, data_dir: Path,
                     barrier: threading.Barrier, store: ByteStore,
                     probe_path: Path | None) -> dict:
    """Replay the same trajectory in one process and record 100 real frames."""
    env = build_arm_env(args.backend, display=args.display,
                        lp_threads=arm["lp_threads"])
    tag = f"{arm['name']}.i{index}"
    if probe_path is not None:
        env["LD_PRELOAD"] = args.gl_probe_shim
        env["TH08_GL_PROBE_OUT"] = str(probe_path)
    child = _Child(binary=args.binary, data_dir=str(data_dir), env=env,
                   timeout=args.timeout)
    record: dict = {"tag": tag, "arm": arm["name"], "lp_threads": arm["lp_threads"],
                    "instance": index, "data_dir": str(data_dir), "pid": None,
                    "role": None, "shm_name": None, "shm_bytes": None,
                    "setup_frames": None, "frames": [], "exit_code": None,
                    "error": None, "frames_captured": 0, "elapsed_seconds": None}
    shm_name = f"v400pix{os.getpid()}a{args.arm_index_of[arm['name']]}i{index}"
    try:
        child.start()
        record["pid"] = child.proc.pid if child.proc else None
        hello = child.request("HELLO")
        record["role"] = hello.get("role")
        if hello.get("role") != "env":
            raise PixelCheckUnavailable(f"role is {hello.get('role')!r}, expected env")
        reset = child.request(f"RESET 0 1 0 {args.seed} 0", timeout=args.timeout)
        if reset.get("ready") != "1":
            raise PixelCheckUnavailable(f"RESET did not become ready: {reset}")
        record["setup_frames"] = int(reset.get("setup_frames", "0"))
        if child.request("READBACK 1").get("readback") != "1":
            raise PixelCheckUnavailable("per-frame readback was refused")
        shm = child.request("SHM /" + shm_name)
        if shm.get("active") != "1" or int(shm.get("bytes", -1)) != SHM_BYTES:
            raise PixelCheckUnavailable(f"pixel SHM unavailable: {shm}")
        record["shm_name"] = shm_name
        record["shm_bytes"] = int(shm["bytes"])
        shm_path = Path("/dev/shm") / shm_name
        barrier.wait(timeout=args.timeout)
        started = time.perf_counter()
        for position, mask in enumerate(masks_for(args.frames, args.mask)):
            step = child.request(f"STEP {mask} 1", timeout=args.timeout)
            if int(step.get("advanced", -1)) != 1:
                raise PixelCheckUnavailable(
                    f"not exactly one real frame at {position + 1}: {step}")
            pixels = child.request("PIXELS")
            if pixels.get("pixels") != "640x480":
                raise PixelCheckUnavailable(
                    f"PIXELS did not return 640x480 at {position + 1}: {pixels}")
            if int(pixels.get("nonzero", "0")) <= 0:
                raise PixelCheckUnavailable(
                    f"PIXELS returned no nonzero byte at {position + 1}: {pixels}")
            raw = shm_path.read_bytes()
            header = validate_shm_payload(raw, expected_frame=position + 1)
            rgba = raw[SHM_HEADER_SIZE:]
            digest = store.put(rgba)
            record["frames"].append({
                "frame": header["frame"],
                "rgba_sha256": digest,
                "pixels_checksum": pixels.get("checksum"),
                "pixels_nonzero": int(pixels.get("nonzero", "0")),
                "state_hash": step.get("hash"),
            })
        record["frames_captured"] = len(record["frames"])
        record["elapsed_seconds"] = time.perf_counter() - started
    except BaseException as exc:  # report, never fabricate
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            record["exit_code"] = child.close()
        except Exception as exc:
            record["error"] = record["error"] or f"close {type(exc).__name__}: {exc}"
        record["stderr_tail"] = bytes(child.stderr_tail).decode("utf-8", "replace")
        if probe_path is not None and probe_path.is_file():
            record["gl"] = parse_gl_fingerprint(
                probe_path.read_text(errors="replace"), record["pid"])
            record["gl"]["source"] = "in-process glGetString via LD_PRELOAD shim"
    return record


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, help="Linux ELF th08-modern")
    parser.add_argument("--data-template", required=True, help="dir with th08.dat")
    parser.add_argument("--run-root", required=True,
                        help="fresh root for per-instance data dirs (never reused)")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME:LP",
                        help="one LP_NUM_THREADS arm, e.g. lp1:1 or lp4:4 "
                             "(repeatable; each arm runs --instances processes)")
    parser.add_argument("--instances", type=int, default=2,
                        help="concurrent processes per arm (default 2)")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1,
                        help="RESET seed; identical for every instance on purpose")
    parser.add_argument("--mask", type=int, default=0,
                        help="input mask applied to every frame")
    parser.add_argument("--backend", choices=("software", "hardware"),
                        default="software")
    parser.add_argument("--display", default=None)
    parser.add_argument("--gl-probe-shim", default=None)
    parser.add_argument("--expect-renderer", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.instances < 1:
        parser.error("--instances must be positive")
    if args.frames < 1:
        parser.error("--frames must be positive")
    if not os.path.exists(args.binary):
        parser.error(f"binary not found: {args.binary}")

    started_at = time.time()
    doc: dict = {
        "schema": "v400-render-pixel-consistency/1", "status": "invalid",
        "display": args.display, "backend": args.backend,
        "instances_per_arm": args.instances, "frames_requested": args.frames,
        "seed": args.seed, "mask": args.mask,
        "masks": masks_for(args.frames, args.mask),
        "binary": args.binary, "binary_sha256": None,
        "data_template": args.data_template, "run_root": args.run_root,
        "pixel_size": [PIXEL_WIDTH, PIXEL_HEIGHT], "pixel_bytes": PIXEL_BYTES,
        "shm_bytes": SHM_BYTES, "argv": sys.argv[1:],
        "python": sys.version.split()[0], "host": hostname(),
        "nproc": os.cpu_count(),
    }
    store = ByteStore()
    try:
        arms = parse_arm_specs(args.arm)
        doc["arms"] = arms
        args.arm_index_of = {arm["name"]: index for index, arm in enumerate(arms)}
        doc["binary_sha256"] = sha256_bytes(Path(args.binary).read_bytes())
        # Refuse a backend that cannot be proven where proof was requested.
        for arm in arms:
            build_arm_env(args.backend, display=args.display,
                          lp_threads=arm["lp_threads"])
        labels = [f"{arm['name']}-i{index}" for arm in arms
                  for index in range(args.instances)]
        prep = prepare_data_dirs(Path(args.data_template), Path(args.run_root), labels)
        doc["data_dirs"] = prep
        barrier = threading.Barrier(len(labels))
        captures: dict[str, dict] = {}
        errors: list[str] = []

        def worker(arm: dict, index: int, data_dir: Path, probe: Path | None) -> None:
            try:
                record = capture_instance(arm, index, args, data_dir, barrier,
                                          store, probe)
                captures[record["tag"]] = record
            except BaseException as exc:
                errors.append(f"{arm['name']}.i{index}: {type(exc).__name__}: {exc}")

        threads = []
        for arm in arms:
            for index in range(args.instances):
                label = f"{arm['name']}-i{index}"
                probe = (Path(prep["dirs"][labels.index(label)]) / "gl_probe.txt"
                         if args.gl_probe_shim else None)
                threads.append(threading.Thread(
                    target=worker,
                    args=(arm, index, Path(prep["dirs"][labels.index(label)]), probe),
                    name=label, daemon=True))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        doc["worker_errors"] = errors
        # Stable order: arm order, then instance index.
        ordered = [captures[f"{arm['name']}.i{index}"]
                   for arm in arms for index in range(args.instances)
                   if f"{arm['name']}.i{index}" in captures]
        doc["captures"] = ordered
        doc["raw_store"] = {"unique_frames": len(store), "puts": store.puts,
                            "bytes_per_frame": PIXEL_BYTES}

        broken = [c for c in ordered if c["error"] or c["frames_captured"] != args.frames]
        renderer_bad = [c["tag"] for c in ordered
                        if not renderer_matches(args.expect_renderer, c.get("gl") or {})]
        missing_fp = [c["tag"] for c in ordered if not c.get("gl")]
        if broken:
            doc["invalid_reason"] = (
                "captures without a complete frame series: "
                f"{[(c['tag'], c['frames_captured'], c['error']) for c in broken]}")
        elif len(ordered) != len(labels):
            doc["invalid_reason"] = (f"{len(ordered)} of {len(labels)} instances "
                                     f"produced a record; errors={errors}")
        elif args.expect_renderer and missing_fp:
            doc["invalid_reason"] = (
                f"no in-process GL fingerprint for {missing_fp}; the "
                f"{args.backend!r} backend is unproven")
        elif renderer_bad:
            doc["invalid_reason"] = (
                f"GL_RENDERER does not contain {args.expect_renderer!r} for {renderer_bad}")
        else:
            comparisons = []
            for i, left in enumerate(ordered):
                for right in ordered[i + 1:]:
                    comparisons.append(compare_captures(left, right, store))
            doc["comparisons"] = comparisons
            doc["verdict"] = summarise(comparisons)
            doc["status"] = "measured"
    except PixelCheckUnavailable as exc:
        doc["invalid_reason"] = str(exc)
    except Exception as exc:
        doc["invalid_reason"] = f"{type(exc).__name__}: {exc}"
    doc["wall_seconds"] = time.time() - started_at
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    verdict = doc.get("verdict") or {}
    summary = {
        "status": doc["status"],
        "arms": [(arm["name"], arm["lp_threads"]) for arm in doc.get("arms", [])],
        "instances_per_arm": args.instances, "frames": args.frames,
        "frames_captured": [c["frames_captured"] for c in doc.get("captures", [])],
        "lp_invariant_bytes": verdict.get("lp_invariant_bytes"),
        "within_lp_identical": verdict.get("within_lp_identical"),
        "cross_lp_identical": verdict.get("cross_lp_identical"),
        "noise_floor_different_frames": verdict.get("noise_floor_different_frames"),
        "comparisons": [{k: c[k] for k in
                         ("left", "right", "frames_compared", "identical",
                          "different_frame_count", "first_different_frame",
                          "first_different_bytes", "first_different_pixels",
                          "first_bbox", "checksum_mismatch_frames")}
                        for c in doc.get("comparisons", [])],
        "gl_renderers": sorted({((c.get("gl") or {}).get("GL_RENDERER") or "unproven")
                                for c in doc.get("captures", [])}),
        "invalid_reason": doc.get("invalid_reason"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return exit_code_for(doc["status"], doc.get("verdict"))


if __name__ == "__main__":
    raise SystemExit(main())
