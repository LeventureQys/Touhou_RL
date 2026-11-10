"""Multi-threaded llvmpipe pixel determinism: ``LP_NUM_THREADS`` > 1 vs 1.

EXP-P0-068 §3 and EXP-P0-061 §13 both leave the same hole open: every
multi-threaded llvmpipe throughput number (``LP=4`` reaching 961-1,097 fps,
against ``LP=1`` at ~130/800) is unusable for a training run until somebody
proves the *pixels* are unchanged.  Multi-threaded rasterisation is allowed to
change pixels in principle, so "faster" alone is not evidence.

This tool answers exactly one question, on the real engine:

    with the same StageStart and the same 100 all-zero masks, is the raw
    640x480 RGBA framebuffer byte-for-byte identical
      (a) between two independent processes at the same ``LP_NUM_THREADS``,
      (b) between ``LP=4`` (and optionally ``LP=8``) and ``LP=1``?

Protocol per ``env`` child, per frame:

* ``STEP 0 1`` advances exactly one real frame (the reply must say
  ``advanced=1 frame=<n>`` and carries the engine's own state hash);
* ``PIXELS`` makes the engine ``glReadPixels`` and publish into POSIX shared
  memory, and replies with ``nonzero`` plus the engine's FNV-1a checksum;
* the raw RGBA bytes are then read straight out of ``/dev/shm/<name>`` and
  SHA-256'd -- the bytes themselves, never a PNG and never a checksum standing
  in for bytes.

Runs inside WSL (Linux), where the harness binary lives::

    /root/v400-venv/bin/python tools/rl/v400_render_pixel_determinism.py \
        --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern \
        --data-template /root/v400-data/game \
        --run-root /root/v400-lpdet-1 --display :98 \
        --lp-threads 1 --lp-threads 4 --frames 100 \
        --out .tmp_v400_lpdet/report.json

Exit codes: ``0`` = every comparison is byte-identical (deterministic);
``4`` = a valid measurement that found real pixel differences;
``3`` = the measurement could not be made (nothing is reported as evidence);
``2`` = bad usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

PIXEL_WIDTH = 640
PIXEL_HEIGHT = 480
PIXEL_BYTES = PIXEL_WIDTH * PIXEL_HEIGHT * 4
SHM_PIXELS_OFFSET = 192
SHM_TOTAL_BYTES = SHM_PIXELS_OFFSET + PIXEL_BYTES
SHM_MAGIC = 0x30303456  # "V400"
SHM_VERSION = 1
SHM_FLAG_PIXELS_VALID = 1 << 1
HEADER = struct.Struct("<IIQQIIII")
FRAMES = 100
START_ARGS = "0 1 0 1 0"  # StageStart: stage 0, difficulty 1, team 0, seed 1, practice 0
DATA_COPY = (("th08.cfg", "th08.cfg.orig"), ("score.dat", "score.dat.orig"))
DATA_LINK = ("th08.dat", "thbgm.dat")


class Blocked(RuntimeError):
    """The comparison cannot be made; never report a verdict instead."""


# ----------------------------------------------------------------- primitives
def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_header(raw: bytes) -> dict:
    """Validate the 192-byte SHM header and return its fields."""
    if len(raw) != SHM_TOTAL_BYTES:
        raise Blocked(f"pixel SHM size {len(raw)}, expected {SHM_TOTAL_BYTES}")
    magic, version, frame, state_hash, width, height, state_bytes, flags = HEADER.unpack_from(raw)
    if magic != SHM_MAGIC or version != SHM_VERSION:
        raise Blocked(f"pixel SHM header magic/version {magic:x}/{version}")
    if (width, height) != (PIXEL_WIDTH, PIXEL_HEIGHT):
        raise Blocked(f"pixel SHM geometry {width}x{height}")
    if not flags & SHM_FLAG_PIXELS_VALID:
        raise Blocked("pixel SHM does not set PIXELS_VALID")
    return {"frame": frame, "state_hash": state_hash, "state_bytes": state_bytes,
            "width": width, "height": height, "flags": flags}


def diff_stats(a: bytes, b: bytes) -> dict:
    """Byte-level difference between two RGBA frames.

    Reports the number of differing bytes and the pixel bounding box so a
    HUD-only artefact can be told apart from a whole-frame raster change.
    """
    if len(a) != len(b):
        raise ValueError("frames have different lengths")
    different_bytes = 0
    first_byte = None
    last_byte = None
    pixels = set()
    min_x = min_y = None
    max_x = max_y = None
    for index in range(len(a)):
        if a[index] == b[index]:
            continue
        different_bytes += 1
        if first_byte is None:
            first_byte = index
        last_byte = index
        pixel = index // 4
        pixels.add(pixel)
        x = pixel % PIXEL_WIDTH
        y = pixel // PIXEL_WIDTH
        min_x = x if min_x is None else min(min_x, x)
        max_x = x if max_x is None else max(max_x, x)
        min_y = y if min_y is None else min(min_y, y)
        max_y = y if max_y is None else max(max_y, y)
    return {
        "different_bytes": different_bytes,
        "different_pixels": len(pixels),
        "first_byte": first_byte,
        "last_byte": last_byte,
        "bbox": (None if min_x is None
                 else {"x0": min_x, "y0": min_y, "x1": max_x, "y1": max_y}),
    }


def compare_series(left: list[dict], right: list[dict]) -> dict:
    """Frame-by-frame comparison of two recorded runs (``raw`` is optional).

    Each record must carry ``frame``, ``sha256`` and ``checksum``; when both
    sides still hold ``raw`` the exact differing-byte counts are computed too.
    """
    if len(left) != len(right):
        raise ValueError(f"different frame counts: {len(left)} vs {len(right)}")
    differing = []
    checksum_differing = []
    first_diff = None
    equal = 0
    for lrec, rrec in zip(left, right):
        same_bytes = lrec["sha256"] == rrec["sha256"]
        if not same_bytes and lrec.get("raw") is not None and rrec.get("raw") is not None:
            # A hash difference without a byte difference is a tool defect.
            same_bytes = lrec["raw"] == rrec["raw"]
            if same_bytes:
                raise Blocked(f"frame {lrec['frame']}: sha differs but bytes are equal")
        if lrec["checksum"] != rrec["checksum"]:
            checksum_differing.append(lrec["frame"])
        if same_bytes:
            equal += 1
            continue
        entry = {"frame": lrec["frame"],
                 "left_sha256": lrec["sha256"], "right_sha256": rrec["sha256"],
                 "left_checksum": lrec["checksum"], "right_checksum": rrec["checksum"],
                 "left_nonzero": lrec.get("nonzero"), "right_nonzero": rrec.get("nonzero")}
        if lrec.get("raw") is not None and rrec.get("raw") is not None:
            entry.update(diff_stats(lrec["raw"], rrec["raw"]))
        if first_diff is None:
            first_diff = entry
        differing.append(entry)
    return {
        "frames": len(left),
        "equal_frames": equal,
        "different_frames": len(differing),
        "first_different_frame": (first_diff["frame"] if first_diff else None),
        "first_difference": first_diff,
        "checksum_different_frames": len(checksum_differing),
        "checksum_different_frame_list": checksum_differing[:20],
        "max_different_bytes": max((e.get("different_bytes", 0) for e in differing), default=0),
        "different_frame_list": [e["frame"] for e in differing[:20]],
    }


def compare_groups(baseline: dict[str, list[dict]], others: dict[str, dict[str, list[dict]]],
                   instances: int) -> dict:
    """Build the three required groups: within-LP, baseline-within, cross-LP.

    ``baseline`` maps process name -> frame records; ``others`` maps LP tag ->
    process name -> frame records.
    """
    within = {}
    for tag, procs in others.items():
        if instances >= 2:
            names = sorted(procs)
            within[tag] = compare_series(procs[names[0]], procs[names[1]])
    cross = {}
    for tag, procs in others.items():
        per_process = {}
        for name in sorted(procs):
            per_process[name] = compare_series(baseline[name], procs[name])
        cross[tag] = per_process
    return {"within": within, "cross": cross}


def verdict(report: dict, baseline_lp: int, instances: int,
            state_divergence: list[dict] | None = None) -> dict:
    """Decide whether multi-threaded llvmpipe is usable for the training path.

    A process that reached StageStart in a *different* game state cannot say
    anything about rasterisation: its pixels are expected to differ because the
    scene differs.  Such a process is reported separately and blocks the
    training verdict on its own (a divergent multi-process reset is a real
    defect), but it is never counted as a pixel difference.
    """
    groups = report["groups"]
    divergent = {(row["lp"] if not isinstance(row["lp"], str) else int(row["lp"]),
                  row["process"]) for row in (state_divergence or [])}
    checks = []
    for tag, result in sorted(groups["within"].items()):
        arms = sorted({lp for lp, _ in divergent})
        state_bad = int(tag) in arms  # a within-arm difference is unattributable
        checks.append({"check": f"{tag} process-A vs process-B",
                       "frames": result["frames"],
                       "equal_frames": result["equal_frames"],
                       "different_frames": result["different_frames"],
                       "first_different_frame": result["first_different_frame"],
                       "state_divergent": state_bad})
    for tag, per_process in sorted(groups["cross"].items()):
        for name, result in sorted(per_process.items()):
            state_bad = (int(tag), name) in divergent
            checks.append({"check": f"{baseline_lp} vs {tag} [{name}]",
                           "frames": result["frames"],
                           "equal_frames": result["equal_frames"],
                           "different_frames": result["different_frames"],
                           "first_different_frame": result["first_different_frame"],
                           "state_divergent": state_bad})
    raster_checks = [c for c in checks if not c["state_divergent"]]
    bad = [c for c in checks if c["different_frames"]]
    raster_bad = [c for c in raster_checks if c["different_frames"]]
    blocking = [f"stage-start state divergence: LP={lp} process={name}"
                for lp, name in sorted(divergent)]
    return {
        "checks": checks,
        "all_equal": not bad,
        "pixel_checks_all_equal": not raster_bad,
        "state_divergent_processes": sorted(f"LP={lp} {name}" for lp, name in divergent),
        "raster_identical_for_state_matched": not raster_bad,
        "failed_checks": [c["check"] for c in bad],
        "blocking_reason": blocking or None,
        "usable_for_training": (not raster_bad) and not divergent,
    }


# ------------------------------------------------------------------ child I/O
@dataclass
class Child:
    """One ``--rl-control`` env child; line protocol over stdin/stdout."""

    binary: str
    data_dir: Path
    name: str
    env: dict[str, str]
    timeout: float = 120.0
    proc: subprocess.Popen | None = None
    buffer: bytes = b""
    pid: int | None = None
    exit_code: int | None = None
    stderr_tail: str = ""

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [self.binary, "--data-dir", str(self.data_dir),
             "--rl-control", "0,1", "--rl-role", "env"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.env, cwd=str(self.data_dir), bufsize=0)
        self.pid = self.proc.pid
        threading.Thread(target=self._drain, name=f"stderr-{self.name}",
                         daemon=True).start()

    def _drain(self) -> None:
        stream = self.proc.stderr if self.proc is not None else None
        chunks: list[bytes] = []
        while stream is not None:
            chunk = stream.read(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(len(c) for c in chunks) > 65536:
                chunks = chunks[-8:]
        self.stderr_tail = b"".join(chunks).decode("utf-8", "replace")

    def ask(self, line: str) -> dict[str, str]:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write((line + "\n").encode("ascii"))
        self.proc.stdin.flush()
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Blocked(f"{self.name}: timeout on {line!r}, exit={self.proc.poll()}")
            ready, _, _ = select.select([self.proc.stdout], [], [], min(remaining, 0.25))
            if ready:
                chunk = os.read(self.proc.stdout.fileno(), 1 << 16)
                if not chunk:
                    raise Blocked(f"{self.name}: pipe EOF on {line!r}, "
                                  f"exit={self.proc.poll()}")
                self.buffer += chunk
            elif self.proc.poll() is not None:
                raise Blocked(f"{self.name}: engine exit on {line!r}, "
                              f"exit={self.proc.returncode}")
        raw, _, self.buffer = self.buffer.partition(b"\n")
        reply = raw.decode("utf-8", "replace").strip()
        if not reply.startswith("OK"):
            raise Blocked(f"{self.name}: {line!r} -> {reply!r}")
        return dict(part.split("=", 1) for part in reply.split()[1:] if "=" in part)

    def environ_subset(self) -> dict[str, str]:
        if self.pid is None:
            return {}
        try:
            raw = Path(f"/proc/{self.pid}/environ").read_bytes()
        except OSError:
            return {}
        wanted = {"LP_NUM_THREADS", "LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER", "DISPLAY"}
        out = {}
        for item in raw.split(b"\0"):
            key, sep, value = item.partition(b"=")
            if not sep:
                continue
            name = key.decode("utf-8", "replace")
            if name in wanted:
                out[name] = value.decode("utf-8", "replace")
        return out

    def close(self) -> int | None:
        proc = self.proc
        if proc is None:
            return None
        try:
            if proc.poll() is None:
                try:
                    self.ask("QUIT")
                except (Blocked, OSError):
                    pass
            try:
                self.exit_code = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.exit_code = proc.wait(timeout=10)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
        return self.exit_code


# ------------------------------------------------------------------ data dirs
def create_data_dir(template: Path, target: Path) -> dict:
    """Fresh per-process data dir; never reuses an existing path.

    ``th08.cfg``/``score.dat`` are copied from the pristine ``*.orig`` template
    files (the engine writes both), ``th08.dat``/``thbgm.dat`` are read-only
    symlinks because they are large and never written.  The shared template
    directory is never used as ``--data-dir``.
    """
    if target.exists() or target.is_symlink():
        raise Blocked(f"refusing to reuse existing data dir: {target}")
    target.mkdir(parents=True)
    actions = []
    for name, source_name in DATA_COPY:
        source = template / source_name
        if not source.is_file():
            raise Blocked(f"template has no {source_name}: {template}")
        (target / name).write_bytes(source.read_bytes())
        actions.append({"file": name, "action": f"copy from {source_name}"})
    for name in DATA_LINK:
        source = template / name
        if not source.is_file():
            raise Blocked(f"template has no {name}: {template}")
        try:
            (target / name).symlink_to(source)
            action = f"symlink to {source}"
        except (OSError, NotImplementedError):
            # Only a platform without symlink support falls back; the action is
            # recorded so the evidence never claims a link that is not one.
            try:
                os.link(source, target / name)
                action = f"hardlink to {source}"
            except (OSError, NotImplementedError, AttributeError):
                import shutil
                shutil.copy2(source, target / name)
                action = f"copy of {source} (symlink unsupported)"
        actions.append({"file": name, "action": action})
    return {"dir": str(target), "actions": actions}


def build_env(lp_threads: int, display: str | None) -> dict[str, str]:
    """Child environment that forces software llvmpipe with ``lp_threads``."""
    if lp_threads < 1:
        raise ValueError("lp_threads must be positive")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LP_", "GALLIUM_", "LIBGL_"))}
    env.pop("MESA_LOADER_DRIVER_OVERRIDE", None)
    env["SDL_AUDIODRIVER"] = "dummy"
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["GALLIUM_DRIVER"] = "llvmpipe"
    env["LP_NUM_THREADS"] = str(lp_threads)
    if display:
        env["DISPLAY"] = display
    return env


# ------------------------------------------------------------------ one run
def record_run(child: Child, shm_name: str, frames: int, keep_raw: bool) -> dict:
    """Drive one child through the fixed 100-frame trajectory."""
    hello = child.ask("HELLO")
    if hello.get("role") != "env":
        raise Blocked(f"{child.name}: role is {hello.get('role')!r}, expected env")
    debug = child.ask("DEBUG")
    reset = child.ask(f"RESET {START_ARGS}")
    if reset.get("ready") != "1":
        raise Blocked(f"{child.name}: RESET did not reach StageStart: {reset}")
    readback = child.ask("READBACK 1")
    if readback.get("readback") != "1":
        raise Blocked(f"{child.name}: READBACK 1 refused: {readback}")
    shm = child.ask(f"SHM /{shm_name}")
    if shm.get("active") != "1" or int(shm.get("bytes", -1)) != SHM_TOTAL_BYTES:
        raise Blocked(f"{child.name}: pixel SHM unavailable: {shm}")
    shm_path = Path("/dev/shm") / shm_name
    if not shm_path.exists():
        raise Blocked(f"{child.name}: {shm_path} is missing")
    records: list[dict] = []
    for index in range(frames):
        step = child.ask("STEP 0 1")
        if int(step.get("advanced", -1)) != 1 or int(step.get("frame", -1)) != index + 1:
            raise Blocked(f"{child.name}: not exactly one real frame at {index + 1}: {step}")
        pixels = child.ask("PIXELS")
        if pixels.get("pixels") != "640x480":
            raise Blocked(f"{child.name}: pixel capability missing at {index + 1}: {pixels}")
        raw = shm_path.read_bytes()
        header = parse_header(raw)
        if header["frame"] != index + 1:
            raise Blocked(f"{child.name}: SHM frame {header['frame']} at step {index + 1}")
        rgba = raw[SHM_PIXELS_OFFSET:]
        record = {
            "frame": index + 1,
            "sha256": sha256_bytes(rgba),
            "checksum": pixels.get("checksum"),
            "nonzero": int(pixels.get("nonzero", -1)),
            "state_hash": step.get("hash"),
            "shm_state_hash": f"{header['state_hash']:016x}",
        }
        if keep_raw:
            record["raw"] = rgba
        records.append(record)
    return {"frames": records, "protocol": {"hello": hello, "reset": reset,
                                            "readback": readback, "shm": shm,
                                            "debug": debug}}


def record_config(binary: str, template: Path, run_root: Path, tag: str,
                  lp_threads: int, instances: int, frames: int, display: str | None,
                  seed_tag: str, keep_raw: bool, timeout: float) -> dict:
    """Run ``instances`` independent children concurrently under one LP value."""
    stem = run_root / f"{seed_tag}-{tag}"
    children = []
    for index in range(instances):
        data_dir = stem / f"p{index}"
        info = create_data_dir(template, data_dir)
        name = f"vpdet{os.getpid()}_{tag}_{index}"
        child = Child(binary=binary, data_dir=data_dir, name=name,
                      env=build_env(lp_threads, display), timeout=timeout)
        children.append((child, info, name, f"p{index}"))
    results: list[dict] = []
    errors: list[BaseException] = []

    def worker(child: Child, name: str, label: str) -> None:
        try:
            child.start()
            run = record_run(child, name, frames, keep_raw)
            results.append({"process": label, "shm": f"/{name}", "run": run,
                            "pid": child.pid,
                            "environ": child.environ_subset()})
        except BaseException as exc:  # surface, never abort siblings silently
            errors.append(exc)

    threads = []
    for child, _info, name, label in children:
        threads.append(threading.Thread(target=worker, args=(child, name, label),
                                        name=child.name, daemon=True))
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    exit_codes = [child.close() for child, _info, _name, _label in children]
    for result, (child, info, _name, _label) in zip(
            sorted(results, key=lambda r: r["process"]), children):
        result["data_dir"] = info["dir"]
        result["data_actions"] = info["actions"]
        result["exit_code"] = child.exit_code
        result["stderr_tail"] = child.stderr_tail[-2000:]
    if errors:
        raise Blocked(f"{tag}: {[f'{type(e).__name__}: {e}' for e in errors]}")
    if len(results) != instances:
        raise Blocked(f"{tag}: {len(results)} of {instances} processes reported")
    if any(code not in (0, None) for code in exit_codes):
        raise Blocked(f"{tag}: non-zero exit codes {exit_codes}")
    return {"tag": tag, "lp_threads": lp_threads, "instances": instances,
            "wall_seconds": wall, "exit_codes": exit_codes,
            "processes": sorted(results, key=lambda r: r["process"])}


def strip_raw(records: list[dict]) -> list[dict]:
    return [{k: v for k, v in record.items() if k != "raw"} for record in records]


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, help="Linux ELF th08-modern")
    parser.add_argument("--data-template", required=True,
                        help="dir holding th08.dat/thbgm.dat and *.orig copies")
    parser.add_argument("--run-root", required=True,
                        help="fresh root for private data dirs (never reused)")
    parser.add_argument("--lp-threads", type=int, action="append", required=True,
                        help="LP_NUM_THREADS arm; repeat for several arms "
                             "(first value is the baseline)")
    parser.add_argument("--instances", type=int, default=2,
                        help="independent processes per arm (>=2 for the "
                             "same-configuration check)")
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--display", default=None)
    parser.add_argument("--seed-tag", default=None,
                        help="subdirectory tag under --run-root (default: pid)")
    parser.add_argument("--keep-raw", action="store_true",
                        help="retain raw RGBA for the baseline arm only (memory "
                             "heavy; cross-arm byte counts need it)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.instances < 1:
        sys.exit("--instances must be positive")
    if args.frames < 1:
        sys.exit("--frames must be positive")
    lps = args.lp_threads
    if len(lps) < 2:
        sys.exit("need at least two --lp-threads arms to compare")
    if len(set(lps)) != len(lps):
        sys.exit("duplicate --lp-threads arms")
    if not Path(args.binary).is_file():
        sys.exit(f"binary not found: {args.binary}")
    baseline_lp, other_lps = lps[0], lps[1:]
    run_root = Path(args.run_root)
    if run_root.exists():
        sys.exit(f"refusing to reuse existing --run-root: {run_root}")
    run_root.mkdir(parents=True)
    seed_tag = args.seed_tag or f"s{os.getpid()}"
    doc: dict = {
        "schema": "v400-render-pixel-determinism/1",
        "status": "unavailable",
        "binary": args.binary,
        "binary_sha256": sha256_bytes(Path(args.binary).read_bytes()),
        "data_template": args.data_template,
        "run_root": str(run_root),
        "display": args.display,
        "baseline_lp": baseline_lp, "arm_lps": lps,
        "instances": args.instances, "frames": args.frames,
        "trajectory": {"stage_start": [0, 1, 0, 1, 0], "masks": [0] * args.frames},
        "pixel_bytes": PIXEL_BYTES, "shm_total_bytes": SHM_TOTAL_BYTES,
        "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME"),
        "nproc": os.cpu_count(),
        "loadavg_before": _loadavg(),
        "argv": sys.argv[1:],
    }
    try:
        template = Path(args.data_template)
        runs: dict[int, dict] = {}
        stripped: dict[int, dict[str, list[dict]]] = {}
        baseline_raw: dict[str, list[dict]] | None = None
        within: dict[str, dict] = {}
        cross: dict[str, dict[str, dict]] = {}
        # Arms run sequentially; the baseline's raw RGBA is kept in memory so
        # that a cross-arm difference can still be counted byte by byte, and
        # every other arm is reduced to hashes as soon as it is compared.
        for index, lp in enumerate(lps):
            run = record_config(args.binary, template, run_root, f"lp{lp}", lp,
                                args.instances, args.frames, args.display, seed_tag,
                                True, args.timeout)
            runs[lp] = run
            procs = {p["process"]: p["run"]["frames"] for p in run["processes"]}
            order = sorted(procs)
            if args.instances >= 2:
                within[str(lp)] = compare_series(procs[order[0]], procs[order[1]])
            if index == 0:
                baseline_raw = procs
            elif baseline_raw is not None:
                cross[str(lp)] = {name: compare_series(baseline_raw[name], procs[name])
                                  for name in order}
            stripped[lp] = {name: strip_raw(records) for name, records in procs.items()}
        doc["runs"] = {str(lp): {"lp_threads": run["lp_threads"],
                                 "wall_seconds": run["wall_seconds"],
                                 "exit_codes": run["exit_codes"],
                                 "processes": [{"process": p["process"], "pid": p["pid"],
                                                "shm": p["shm"], "environ": p["environ"],
                                                "data_dir": p["data_dir"],
                                                "data_actions": p["data_actions"],
                                                "exit_code": p["exit_code"],
                                                "stderr_tail": p["stderr_tail"],
                                                "protocol": p["run"]["protocol"],
                                                "frames": stripped[lp][p["process"]]}
                                               for p in run["processes"]]}
                         for lp, run in runs.items()}
        # Trajectory sanity: the engine's own state hash must agree everywhere.
        baseline = stripped[baseline_lp]
        names = sorted(baseline)
        state_divergence = []
        for lp, procs in stripped.items():
            for name in names:
                for lrec, rrec in zip(baseline[name], procs[name]):
                    if lrec["state_hash"] != rrec["state_hash"]:
                        state_divergence.append({"lp": lp, "process": name,
                                                 "frame": lrec["frame"]})
                        break
        doc["state_hash_divergence"] = state_divergence
        doc["groups"] = {"within": within, "cross": cross,
                         "baseline_lp": baseline_lp, "instances": args.instances}
        doc["comparison_scope"] = {
            "within": "two independent processes, same LP_NUM_THREADS, same "
                      "private data dir, own SHM name, same StageStart + masks",
            "cross": "each arm-process against the same-index baseline-process",
            "bytes": "raw 640x480 RGBA from /dev/shm, SHA-256 + bytewise compare",
        }
        doc["verdict"] = verdict(doc, baseline_lp, args.instances,
                                 doc["state_hash_divergence"])
        if doc["verdict"]["usable_for_training"]:
            doc["status"] = "deterministic"
        elif doc["verdict"]["pixel_checks_all_equal"]:
            # Pixels agree wherever the game state agrees; the blocker is a
            # multi-process StageStart race, reported separately.
            doc["status"] = "state_divergent"
        else:
            doc["status"] = "divergent"
        doc["loadavg_after"] = _loadavg()
    except (Blocked, OSError, ValueError) as error:
        doc["status"] = "unavailable"
        doc["reason"] = f"{type(error).__name__}: {error}"
        doc["loadavg_after"] = _loadavg()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    summary = {
        "status": doc["status"], "reason": doc.get("reason"),
        "arms": lps, "instances": args.instances, "frames": args.frames,
        "within": {tag: [r["equal_frames"], r["different_frames"],
                         r["first_different_frame"]]
                   for tag, r in (doc.get("groups") or {}).get("within", {}).items()},
        "cross": {tag: {name: [r["equal_frames"], r["different_frames"],
                               r["first_different_frame"]]
                        for name, r in per.items()}
                  for tag, per in (doc.get("groups") or {}).get("cross", {}).items()},
        "usable_for_training": (doc.get("verdict") or {}).get("usable_for_training"),
        "raster_identical_for_state_matched": (doc.get("verdict") or {}).get(
            "raster_identical_for_state_matched"),
        "blocking_reason": (doc.get("verdict") or {}).get("blocking_reason"),
        "state_hash_divergence": doc.get("state_hash_divergence"),
        "report": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if doc["status"] == "deterministic" else (
        4 if doc["status"] in ("divergent", "state_divergent") else 3)


def _loadavg() -> dict:
    try:
        parts = Path("/proc/loadavg").read_text().split()
        return {"raw": " ".join(parts), "load1": float(parts[0])}
    except (OSError, IndexError, ValueError) as error:
        return {"raw": None, "error": f"{type(error).__name__}: {error}"}


if __name__ == "__main__":
    raise SystemExit(main())
