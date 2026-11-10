"""WP5 / G0.2 item 5 — split real ``BRANCH`` cost into a fixed and a per-frame part.

Run this *inside* WSL against the real Linux engine; it drives the ``shadow``
role, whose ``BRANCH`` handler forks the live engine and advances the copy in
the child.

Why this tool exists
--------------------
The engine's ``fork_us`` reply field is a *parent-side* span: it starts just
before ``fork()`` and stops only after the child's whole advance has been read
back through the result pipe **and** ``waitpid`` returned.  That makes it a
mixed measure dominated by per-frame engine work, so quoting it as "branch
overhead" is wrong.  ``child_us`` — sampled inside the child from just after
``fork()`` until the frame loop ends — is the clean per-frame signal.

The tool takes ``repeats`` samples of ``BRANCH <mask> <frames>`` over a horizon
ladder, fits ``child_us = intercept + slope * frames`` by ordinary least
squares, and reports the slope as the per-frame cost and the intercept as the
fixed constant term of the child timing region.  ``fork_us - child_us`` is the
independent cross-check for everything outside the child's frame loop (fork,
result pipe, reap); a fit of ``fork_us`` itself supplies a second estimate of
the same total.

Honesty boundary
----------------
Every number produced here is *inferred from a regression over real BRANCH
replies*.  Nothing is a direct instrumentation of the ``fork()`` system call,
so no figure may be quoted as pure syscall time.  See the emitted ``caveats``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import select
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_HORIZONS = (1, 5, 10, 50, 100, 200, 300)
DEFAULT_REPEATS = 5
DEFAULT_PASSES = 3
DEFAULT_WARMUP = 2
DEFAULT_RESET_SEED = 1
DEFAULT_ORDER_SEED = 20261109
MAX_BRANCH_FRAMES = 300  # engine-side limit in HandleBranch()
SCHEMA = "v400-wp5-branch-cost/1"

CAVEATS = (
    "child_us is sampled inside the forked child, from just after fork() returns "
    "until the frame loop ends. It excludes fork() itself, the result-pipe write "
    "and the parent's read/waitpid.",
    "fork_us is the parent-side span from just before fork() until after the "
    "child's payload is read and waitpid returns; it therefore contains the "
    "child's entire advance and is NOT a fork overhead measure.",
    "The fitted intercept is the constant term of child_us, i.e. fixed "
    "per-BRANCH overhead inside the child timing region. It is not the fork() "
    "system call duration.",
    "fork_us - child_us is a difference of two independent real timers. It "
    "estimates fork + result pipe + reap overhead but is not a direct fork() "
    "instrument either.",
    "The per-frame slope includes the per-frame CurrentStateHash() call the "
    "child performs for every advanced frame, so it is engine-advance-plus-hash "
    "cost, not pure simulation cost.",
    "Both timers are CLOCK_MONOTONIC truncated to whole microseconds, so each "
    "sample carries up to 1 us truncation error.",
    "The fitted intercept absorbs every one-time cost the child pays before its "
    "frames run warm: for a freshly forked process that is dominated by cold "
    "caches and copy-on-write page faults on the first frame. The tool does not "
    "count page faults, so this attribution is an interpretation that the data "
    "supports (per-frame cost at horizon 1 is far above the steady-state slope), "
    "not an independently measured quantity.",
    "fit(fork_us) = fit(child_us) + fit(fork_us - child_us) holds exactly because "
    "OLS coefficients are linear in the response; it is an algebraic identity and "
    "is reported as a consistency check only.",
    "The intercept is the least reproducible of the two coefficients because it "
    "depends on cold-start effects; the per-pass refits under 'passes' report its "
    "run-to-run spread instead of hiding it behind a single number.",
    "This is a decomposition inferred by regression over real measurements, not "
    "an isolated syscall timing experiment.",
)


class Unavailable(RuntimeError):
    """The real environment, the protocol contract or the data were not usable."""


# --------------------------------------------------------------------------
# reply parsing / validation
# --------------------------------------------------------------------------
def parse_reply(reply: str) -> dict[str, str]:
    """``OK a=1 b=2`` -> ``{"_verb": "OK", "a": "1", "b": "2"}``."""
    parts = reply.split()
    fields: dict[str, str] = {"_verb": parts[0] if parts else ""}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def parse_branch(reply: str, requested_frames: int) -> dict:
    """Validate one real BRANCH reply and return its measured fields."""
    fields = parse_reply(reply)
    verb = fields.get("_verb")
    if verb != "OK":
        raise Unavailable(f"BRANCH rejected by engine: {reply[:200]!r}")
    if fields.get("branch") != "1":
        raise Unavailable(f"BRANCH did not report branch=1: {reply[:200]!r}")
    try:
        advanced = int(fields["advanced"])
        fork_us = int(fields["fork_us"])
        child_us = int(fields["child_us"])
        pid = int(fields["pid"])
        parent_frame = int(fields["parent_frame"])
        hash_value = fields["hash"]
    except (KeyError, ValueError) as error:
        raise Unavailable(f"BRANCH reply lacks a timing field ({error}): {reply[:200]!r}") from error
    if advanced != requested_frames:
        raise Unavailable(
            f"BRANCH advanced={advanced} but {requested_frames} frames were requested"
        )
    if fork_us < 0 or child_us <= 0:
        raise Unavailable(f"BRANCH reported non-positive timing fork_us={fork_us} child_us={child_us}")
    if fork_us < child_us:
        raise Unavailable(
            f"BRANCH fork_us={fork_us} < child_us={child_us}; the parent span must contain the child span"
        )
    trace = fields.get("trace", "")
    if len(trace) != 16 * advanced:
        raise Unavailable(
            f"BRANCH trace carries {len(trace)} hex chars; expected {16 * advanced} for {advanced} frames"
        )
    if trace[-16:] != hash_value:
        raise Unavailable(
            f"BRANCH hash={hash_value} disagrees with the last trace entry {trace[-16:]}"
        )
    return {
        "frames": requested_frames,
        "advanced": advanced,
        "pid": pid,
        "parent_frame": parent_frame,
        "fork_us": fork_us,
        "child_us": child_us,
        "hash": hash_value,
        "trace_entry_count": len(trace) // 16,
    }


# --------------------------------------------------------------------------
# least-squares fit
# --------------------------------------------------------------------------
def fit_linear(xs: list[float], ys: list[float]) -> dict:
    """Ordinary least squares ``y = intercept + slope * x``."""
    if len(xs) != len(ys):
        raise ValueError(f"xs/ys length mismatch: {len(xs)} vs {len(ys)}")
    count = len(xs)
    if count < 3:
        raise ValueError(f"a least-squares line needs at least 3 points, got {count}")
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        raise ValueError("x carries no variance; the slope is undefined")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    dof = count - 2
    sigma2 = ss_res / dof
    return {
        "n": count,
        "slope_us_per_frame": slope,
        "intercept_us": intercept,
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "ss_residual": ss_res,
        "ss_total": ss_tot,
        "residual_std_us": math.sqrt(sigma2),
        "se_slope_us_per_frame": math.sqrt(sigma2 / sxx),
        "se_intercept_us": math.sqrt(sigma2 * (1.0 / count + mean_x**2 / sxx)),
        "x_min": min(xs),
        "x_max": max(xs),
    }


def summarize(values: list[float]) -> dict:
    """Mean / sample std / min / max for one sample group."""
    count = len(values)
    if not count:
        raise ValueError("cannot summarize an empty sample")
    mean = sum(values) / count
    spread = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / (count - 1))
        if count > 1
        else 0.0
    )
    return {"n": count, "mean_us": mean, "std_us": spread, "min_us": min(values), "max_us": max(values)}


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def _require_shadow_threads(debug: dict[str, str], *, phase: str) -> None:
    if debug.get("threads") != "1" or debug.get("subthread") != "0":
        raise Unavailable(
            f"{phase}: DEBUG threads={debug.get('threads')!r} subthread={debug.get('subthread')!r}; "
            "BRANCH needs a single-threaded shadow"
        )


def _require_ready(debug: dict[str, str], *, phase: str) -> None:
    """Mirror of HandleBranch's Ready() preconditions."""
    expected = {"globals": "1", "setupState": "0", "curState": "2", "threads": "1", "subthread": "0"}
    wrong = {key: debug.get(key) for key, want in expected.items() if debug.get(key) != want}
    if wrong:
        raise Unavailable(f"{phase}: engine not branch-ready (Ready() preconditions): {wrong}")


def run_probe(
    engine,
    *,
    horizons=DEFAULT_HORIZONS,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
    passes: int = DEFAULT_PASSES,
    mask: int = 0,
    reset_seed: int = DEFAULT_RESET_SEED,
    order_seed: int = DEFAULT_ORDER_SEED,
) -> dict:
    """Drive one real shadow engine and return the measured body of the report."""
    horizons = list(horizons)
    if not horizons:
        raise Unavailable("at least one horizon is required")
    if len(set(horizons)) != len(horizons):
        raise Unavailable(f"horizons must be unique, got {horizons}")
    if any(not 1 <= horizon <= MAX_BRANCH_FRAMES for horizon in horizons):
        raise Unavailable(f"horizons must lie in 1..{MAX_BRANCH_FRAMES}, got {horizons}")
    if repeats < 3:
        raise Unavailable(f"at least 3 repeats per horizon are needed to fit a line, got {repeats}")
    if passes < 1:
        raise Unavailable(f"at least 1 pass is required, got {passes}")
    if warmup < 0:
        raise Unavailable("warmup cannot be negative")

    hello = parse_reply(engine.ask("HELLO"))
    if hello.get("role") != "shadow":
        raise Unavailable(
            f"HELLO role={hello.get('role')!r}; the shadow role is mandatory for BRANCH"
        )
    debug = parse_reply(engine.ask("DEBUG"))
    _require_shadow_threads(debug, phase="before RESET")

    reset_command = f"RESET 0 1 0 {reset_seed} 0"
    reset = parse_reply(engine.ask(reset_command))
    if reset.get("ready") != "1":
        raise Unavailable(f"{reset_command} -> ready={reset.get('ready')!r}; require 1")

    debug = parse_reply(engine.ask("DEBUG"))
    _require_ready(debug, phase="after RESET")

    warmup_rows = []
    if warmup:
        frames = max(horizons)
        for index in range(warmup):
            row = parse_branch(engine.ask(f"BRANCH {mask} {frames}"), frames)
            row["index"] = index
            warmup_rows.append(row)

    samples: list[dict] = []
    orders: list[list[int]] = []
    parent_frames: set[int] = set()
    rng = random.Random(order_seed)
    for pass_index in range(passes):
        for repeat in range(repeats):
            order = list(horizons)
            rng.shuffle(order)  # decouple horizon from position in the run
            orders.append(order)
            for frames in order:
                row = parse_branch(engine.ask(f"BRANCH {mask} {frames}"), frames)
                row["pass"] = pass_index
                row["repeat"] = repeat
                row["index"] = len(samples)
                parent_frames.add(row["parent_frame"])
                samples.append(row)
                if len(parent_frames) > 1:
                    raise Unavailable(
                        f"the parent advanced between branches: parent_frame values {sorted(parent_frames)}"
                    )

    debug_after = parse_reply(engine.ask("DEBUG"))
    _require_shadow_threads(debug_after, phase="after measurements")

    body = decompose(samples, horizons)
    body["warmup"] = {
        "discarded_branches": len(warmup_rows),
        "frames": max(horizons) if warmup_rows else 0,
        "rows": [
            {"index": row["index"], "pid": row["pid"], "fork_us": row["fork_us"],
             "child_us": row["child_us"]}
            for row in warmup_rows
        ],
    }
    body["protocol"] = {
        "argv": list(getattr(engine, "argv", [])),
        "environment": {
            "DISPLAY": "removed",
            "WAYLAND_DISPLAY": "removed",
            "SDL_AUDIODRIVER": "dummy",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "LP_NUM_THREADS": "1",
        },
        "commands": [
            "HELLO",
            "DEBUG",
            reset_command,
            "DEBUG",
            *[f"BRANCH {mask} {max(horizons)}" for _ in range(warmup)],
            f"BRANCH {mask} <frames> x{len(samples)} (horizons {horizons}, {repeats} repeats x "
            f"{passes} passes, order shuffled by seed {order_seed})",
            "DEBUG",
            "QUIT",
        ],
        "mask": mask,
        "horizons": horizons,
        "repeats": repeats,
        "passes": passes,
        "warmup": warmup,
        "reset_seed": reset_seed,
        "order_seed": order_seed,
        "orders": orders,
        "branch_requests": len(samples),
        "parent_stepped_between_branches": False,
        "debug_initial": debug,
        "debug_after": debug_after,
        "reset_reply": reset,
        "engine_log": list(getattr(engine, "log", [])),
    }
    return body


def decompose(samples: list[dict], horizons) -> dict:
    """Regress the raw BRANCH samples into fixed and per-frame components."""
    if not samples:
        raise Unavailable("no BRANCH samples were collected")
    frames = [float(row["frames"]) for row in samples]
    child = [float(row["child_us"]) for row in samples]
    fork = [float(row["fork_us"]) for row in samples]
    delta = [f - c for f, c in zip(fork, child)]

    child_fit = fit_linear(frames, child)
    fork_fit = fit_linear(frames, fork)
    delta_fit = fit_linear(frames, delta)

    per_horizon = []
    for horizon in sorted(set(int(value) for value in frames)):
        rows = [row for row in samples if row["frames"] == horizon]
        child_stats = summarize([float(row["child_us"]) for row in rows])
        delta_stats = summarize([float(row["fork_us"] - row["child_us"]) for row in rows])
        per_horizon.append(
            {
                "frames": horizon,
                "n": len(rows),
                "child_us": child_stats,
                "fork_us": summarize([float(row["fork_us"]) for row in rows]),
                "fork_minus_child_us": delta_stats,
                "child_us_per_frame_mean": child_stats["mean_us"] / horizon,
                "distinct_hashes": len({row["hash"] for row in rows}),
                "pids": sorted({row["pid"] for row in rows}),
            }
        )

    delta_stats = summarize(delta)
    horizon_means = [row["fork_minus_child_us"]["mean_us"] for row in per_horizon]
    drift = max(horizon_means) - min(horizon_means)
    # Explicit comparison rule so the boolean is reproducible, not a judgement.
    flat_rule = "max(per-horizon mean) - min(per-horizon mean) <= 50% of the overall mean"
    delta_flat = drift <= 0.5 * abs(delta_stats["mean_us"]) if delta_stats["mean_us"] else False

    cross_check = {
        "fork_minus_child_us": delta_stats,
        "fork_minus_child_us_per_horizon": [
            {"frames": row["frames"], "mean_us": row["fork_minus_child_us"]["mean_us"],
             "std_us": row["fork_minus_child_us"]["std_us"]}
            for row in per_horizon
        ],
        "fork_minus_child_us_fit": delta_fit,
        "fork_minus_child_us_horizon_drift_us": drift,
        "fork_minus_child_us_relative_drift": drift / abs(delta_stats["mean_us"]) if delta_stats["mean_us"] else None,
        "fork_minus_child_us_flat_across_horizons": bool(delta_flat),
        "fork_minus_child_us_flat_rule": flat_rule,
        "fork_minus_child_us_per_frame_share_of_child_slope": (
            delta_fit["slope_us_per_frame"] / child_fit["slope_us_per_frame"] * 100.0
            if child_fit["slope_us_per_frame"] else None
        ),
        "fork_fit": fork_fit,
        "slope_difference_us_per_frame": abs(
            fork_fit["slope_us_per_frame"] - child_fit["slope_us_per_frame"]
        ),
        "mixed_fork_us_mean_all_samples": summarize(fork)["mean_us"],
        "ols_additivity_identity": (
            "fit(fork_us) equals fit(child_us) + fit(fork_us - child_us) exactly, because "
            "OLS coefficients are linear in the response. That identity is algebra, not "
            "evidence. The independent comparison is the directly fitted fork_us intercept "
            "against (child_us intercept + observed fork_us - child_us mean)."
        ),
    }

    pass_ids = sorted({int(row.get("pass", 0)) for row in samples})
    per_pass = []
    for pass_id in pass_ids:
        rows = [row for row in samples if int(row.get("pass", 0)) == pass_id]
        if len(rows) < 3:
            continue
        pass_x = [float(row["frames"]) for row in rows]
        pass_child = [float(row["child_us"]) for row in rows]
        pass_fork = [float(row["fork_us"]) for row in rows]
        pass_delta = [f - c for f, c in zip(pass_fork, pass_child)]
        pass_child_fit = fit_linear(pass_x, pass_child)
        pass_delta_fit = fit_linear(pass_x, pass_delta)
        per_pass.append(
            {
                "pass": pass_id,
                "n": len(rows),
                "slope_us_per_frame": pass_child_fit["slope_us_per_frame"],
                "intercept_us": pass_child_fit["intercept_us"],
                "r2": pass_child_fit["r2"],
                "fork_minus_child_us_mean": sum(pass_delta) / len(pass_delta),
                "fork_minus_child_us_intercept": pass_delta_fit["intercept_us"],
            }
        )

    def _spread(values: list[float]) -> tuple[float, float, float | None]:
        mean = sum(values) / len(values)
        spread = (
            math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
            if len(values) > 1
            else 0.0
        )
        return mean, spread, (spread / abs(mean) * 100.0 if mean else None)

    slope_mean, slope_spread, slope_relative = _spread(
        [row["slope_us_per_frame"] for row in per_pass]
    )
    intercept_mean, intercept_spread, intercept_relative = _spread(
        [row["intercept_us"] for row in per_pass]
    )
    passes_summary = {
        "count": len(per_pass),
        "samples_per_pass": len(samples) // len(per_pass) if per_pass else 0,
        "per_pass": per_pass,
        "slope_mean_us_per_frame": slope_mean,
        "slope_std_us_per_frame": slope_spread,
        "slope_relative_spread_percent": slope_relative,
        "intercept_mean_us": intercept_mean,
        "intercept_std_us": intercept_spread,
        "intercept_relative_spread_percent": intercept_relative,
        "r2_min": min((row["r2"] for row in per_pass), default=None),
        "r2_max": max((row["r2"] for row in per_pass), default=None),
        "note": (
            "Each pass re-runs the complete horizon ladder with a fresh shuffle, so the "
            "spread measures run-to-run reproducibility inside one machine, not a "
            "cross-machine or cross-build study."
        ),
    }

    interpretation = {
        "per_frame_us": child_fit["slope_us_per_frame"],
        "per_frame_ms": child_fit["slope_us_per_frame"] / 1000.0,
        "fixed_us_child_region": child_fit["intercept_us"],
        "fork_ipc_reap_us": delta_stats["mean_us"],
        "fork_ipc_reap_constant_us": delta_fit["intercept_us"],
        "fork_ipc_reap_per_frame_us": delta_fit["slope_us_per_frame"],
        "fixed_total_us": child_fit["intercept_us"] + delta_stats["mean_us"],
        "fixed_total_us_fork_fit": fork_fit["intercept_us"],
        "fixed_total_relative_gap": (
            abs((child_fit["intercept_us"] + delta_stats["mean_us"]) - fork_fit["intercept_us"])
            / fork_fit["intercept_us"] * 100.0
            if fork_fit["intercept_us"] else None
        ),
        "run_to_run_slope_relative_spread_percent": slope_relative,
        "run_to_run_intercept_relative_spread_percent": intercept_relative,
        "old_mixed_fork_us_over_horizon": {
            row["frames"]: row["fork_us"]["mean_us"] for row in per_horizon
        },
        "statement": (
            "BRANCH cost is decomposed by ordinary least squares over real BRANCH replies: "
            "child_us = intercept + slope * frames. The slope is the per-frame cost (engine "
            "advance plus CurrentStateHash), the intercept is the fixed constant term of the "
            "child timing region. Everything outside that region (fork, result pipe, waitpid) "
            "is estimated by fork_us - child_us and, independently, by the intercept of the "
            "fork_us fit. All of these are regression inferences from real timings; none is a "
            "direct measurement of the fork() system call, and none may be quoted as pure "
            "syscall time."
        ),
    }

    return {
        "samples": samples,
        "per_horizon": per_horizon,
        "fit": {"child_us": child_fit, "fork_us": fork_fit},
        "cross_check": cross_check,
        "passes": passes_summary,
        "interpretation": interpretation,
        "caveats": list(CAVEATS),
    }


# --------------------------------------------------------------------------
# real engine process
# --------------------------------------------------------------------------
class ShadowEngine:
    """One real ``th08-modern --rl-control 0,1 --rl-role shadow`` process."""

    def __init__(self, binary: Path, data_dir: Path, *, timeout: float = 60.0) -> None:
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("DISPLAY", "WAYLAND_DISPLAY", "LP_NO_RAST")
        }
        env.update(SDL_AUDIODRIVER="dummy", LIBGL_ALWAYS_SOFTWARE="1", LP_NUM_THREADS="1")
        self.argv = [
            str(binary), "--data-dir", str(data_dir), "--rl-control", "0,1", "--rl-role", "shadow",
        ]
        self.timeout = timeout
        self.buffer = b""
        self.log: list[dict] = []
        self.process: subprocess.Popen[bytes] | None = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            bufsize=0,
        )
        # The shared WSL checkout may be rebuilt by concurrent work, so pin the
        # identity of the image this process actually mapped, not just the path.
        self.pid = self.process.pid
        self.image_sha256 = running_image_hash(self.pid)
        self.image_path = running_image_path(self.pid)

    def ask(self, command: str, timeout: float | None = None) -> str:
        """Send one command line and return the single reply line."""
        if self.process is None or self.process.stdin is None:
            raise Unavailable(f"engine is not running; cannot send {command!r}")
        try:
            self.process.stdin.write((command + "\n").encode("ascii"))
            self.process.stdin.flush()
        except OSError as error:
            raise Unavailable(
                f"control-pipe write failed on {command!r}: {error}; exit={self.process.poll()}"
            ) from error
        reply = self._readline(command, self.timeout if timeout is None else timeout)
        self.log.append({"command": command, "reply_head": reply[:200]})
        if reply.startswith("ERR"):
            raise Unavailable(f"{command!r} -> {reply[:200]}")
        return reply

    def _readline(self, command: str, limit: float) -> str:
        assert self.process is not None and self.process.stdout is not None
        deadline = time.monotonic() + limit
        while b"\n" not in self.buffer:
            left = deadline - time.monotonic()
            if left <= 0:
                raise Unavailable(f"no reply within {limit:.1f}s for {command!r}")
            ready, _, _ = select.select([self.process.stdout], [], [], min(left, 0.25))
            if ready:
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise Unavailable(
                        f"engine closed the control pipe during {command!r} "
                        f"(exit={self.process.poll()})"
                    )
                self.buffer += chunk
            elif self.process.poll() is not None:
                raise Unavailable(
                    f"engine exited ({self.process.returncode}) during {command!r}"
                )
        line, _, self.buffer = self.buffer.partition(b"\n")
        return line.decode("utf-8", "replace").strip()

    def close(self, *, quit_timeout: float = 15.0) -> int:
        process = self.process
        if process is None:
            return -1
        if process.poll() is None:
            try:
                self.ask("QUIT", timeout=quit_timeout)
            except (Unavailable, OSError):
                pass
        try:
            return process.wait(timeout=quit_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=5)
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
            self.process = None


# --------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def running_image_hash(pid: int) -> str | None:
    """SHA-256 of the executable image the live process actually mapped."""
    try:
        return sha256_file(Path(f"/proc/{pid}/exe"))
    except OSError:
        return None


def running_image_path(pid: int) -> str | None:
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return None


def host_info() -> dict:
    info = {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "kernel": platform.release(),
        "cpu_count": os.cpu_count(),
    }
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="replace")
        info["proc_version"] = version.strip()[:300]
        info["is_wsl"] = "microsoft" in version.lower()
    except OSError:
        info["proc_version"] = None
        info["is_wsl"] = False
    return info


def build_report(
    body: dict,
    *,
    binary: Path,
    data_dir: Path,
    binary_sha256: str | None,
    engine_exit_code: int,
    host: dict | None = None,
    image_sha256: str | None = None,
    image_path: str | None = None,
    binary_sha256_after: str | None = None,
) -> dict:
    """Wrap the measured body in provenance, status and exit codes."""
    status = "measured" if engine_exit_code == 0 else "unavailable"
    report = {
        "schema": SCHEMA,
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": host if host is not None else host_info(),
        "engine": {
            "binary": str(binary),
            "binary_sha256": binary_sha256,
            "binary_sha256_after_run": binary_sha256_after,
            "running_image_sha256": image_sha256,
            "running_image_path": image_path,
            "running_image_matches_binary": (
                image_sha256 == binary_sha256 if (image_sha256 and binary_sha256) else None
            ),
            "binary_changed_during_run": (
                binary_sha256 != binary_sha256_after
                if (binary_sha256 and binary_sha256_after)
                else None
            ),
            "data_dir": str(data_dir),
            "role": "shadow",
            "display_removed": True,
        },
        "protocol": body.get("protocol", {}),
        "warmup": body.get("warmup", {}),
        "samples": body["samples"],
        "per_horizon": body["per_horizon"],
        "fit": body["fit"],
        "cross_check": body["cross_check"],
        "passes": body.get("passes", {}),
        "interpretation": body["interpretation"],
        "caveats": body["caveats"],
        "engine_exit_code": engine_exit_code,
        "tool_exit_code": 0 if engine_exit_code == 0 else 3,
    }
    if engine_exit_code != 0:
        report["reason"] = f"the real engine exited {engine_exit_code} after the probe; measurements are not accepted"
    return report


def _require_real_environment(binary: Path, data_dir: Path) -> None:
    if not binary.is_file():
        raise Unavailable(f"engine binary not found: {binary}")
    if not data_dir.is_dir():
        raise Unavailable(f"game data dir not found: {data_dir}")
    if sys.platform != "linux":
        raise Unavailable(
            f"the real shadow engine needs Linux (WSL), host platform is {sys.platform!r}"
        )


def run(args) -> dict:
    """Execute the measurement and return the finished report."""
    binary = args.binary.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    _require_real_environment(binary, data_dir)
    if "LP_NO_RAST" in os.environ:
        raise Unavailable("LP_NO_RAST is set; that proxy is forbidden for real shadow measurements")

    binary_sha256_before = sha256_file(binary)
    engine = ShadowEngine(binary, data_dir, timeout=args.timeout)
    try:
        body = run_probe(
            engine,
            horizons=args.horizons,
            repeats=args.repeats,
            warmup=args.warmup,
            passes=args.passes,
            mask=args.mask,
            reset_seed=args.seed,
            order_seed=args.order_seed,
        )
    except BaseException:
        engine.close()
        raise
    image_sha256, image_path = engine.image_sha256, engine.image_path
    engine_exit_code = engine.close()
    # Refresh the log so the artefact also records the QUIT exchange.
    body.setdefault("protocol", {})["engine_log"] = list(engine.log)
    return build_report(
        body,
        binary=binary,
        data_dir=data_dir,
        binary_sha256=binary_sha256_before,
        binary_sha256_after=sha256_file(binary),
        image_sha256=image_sha256,
        image_path=image_path,
        engine_exit_code=engine_exit_code,
    )


def _parse_horizons(text: str) -> list[int]:
    try:
        values = [int(item) for item in text.replace(" ", "").split(",") if item]
    except ValueError as error:
        raise Unavailable(f"--horizons must be comma separated integers: {error}") from error
    if not values:
        raise Unavailable("--horizons must list at least one integer")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES,
                        help="independent re-runs of the whole ladder, used for the run-to-run spread")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--mask", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_RESET_SEED, help="RESET seed")
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    try:
        args.horizons = _parse_horizons(args.horizons)
        report = run(args)
    except (Unavailable, OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "unavailable", "reason": str(error)}, ensure_ascii=False))
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    child_fit = report["fit"]["child_us"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": str(args.out),
                "samples": len(report["samples"]),
                "per_frame_us": round(child_fit["slope_us_per_frame"], 4),
                "fixed_us_child_region": round(child_fit["intercept_us"], 4),
                "r2": round(child_fit["r2"], 6),
                "fork_ipc_reap_us": round(report["cross_check"]["fork_minus_child_us"]["mean_us"], 4),
                "passes": report.get("passes", {}).get("count", 0),
                "engine_exit_code": report["engine_exit_code"],
            },
            ensure_ascii=False,
        )
    )
    return report["tool_exit_code"]


if __name__ == "__main__":
    sys.exit(main())
