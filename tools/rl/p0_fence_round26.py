"""P0 re-run on the frame-fence build (round 26).

Why this file exists
--------------------
``Document/Update/Alpha/v3.7.3/reports/P0-exit-round15.md`` froze three P0
claims as ``unknown`` / ``partial`` / ``failed`` because plain GDI capture has no
present fence.  v3.7.4 delivered a present fence (research variant image), but
that round only demonstrated the mechanism on an attract demo: no keys, no
boundaries, no dynamics, no ``read_state`` comparisons against live play.

This script re-runs the frozen P0 gates *on that variant*, with the exact-frame
accounting the fence makes possible.  The pre-registered protocol, sample sizes
and pass criteria are in
``Document/Update/Alpha/v3.7.3/preregistration/p0-fence-round26.md``; nothing
here may be tuned after looking at the numbers.

Two sessions
------------
* demo session (attract demo, no player control): M1 clock trace, M5
  pixel/state samples, M6 legacy-read control.  A death cannot interrupt it.
* practice session (Lunatic stage 1, PostMessage input): M2/M3 input and
  dynamics trials, M4 boundary saturation plus reversal / axis-switch carry.

The fence build is **not** the canonical image: every number produced here is
scoped to ``third_party/th08/build_fence/th08.exe`` and its address table.

Usage
-----
    python tools/rl/p0_fence_round26.py --out-dir Document/Update/Alpha/v3.7.3/evidence/p0-fence-round26
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FENCE_EXE = ROOT / "third_party/th08/build_fence/th08.exe"
FENCE_ADDRESSES = ROOT / "third_party/th08/build_fence/frame-fence-addresses.json"
FENCE_MAP = ROOT / "third_party/th08/build_fence/th08.map"
FENCE_TEMPLATE = ROOT / "third_party/th08/build_fence_template"

# Input-mask bit layout, mirrored from touhou_rl/spec.py.  Duplicated (and
# asserted equal in the unit test) so that the pure analysis functions below can
# be imported without touching the address-table machinery in `spec`.
BTN_SHOOT = 1 << 0
BTN_FOCUS = 1 << 2
BTN_UP = 1 << 4
BTN_DOWN = 1 << 5
BTN_LEFT = 1 << 6
BTN_RIGHT = 1 << 7
DIRECTION_BITS = BTN_UP | BTN_DOWN | BTN_LEFT | BTN_RIGHT
COMPARED_BITS = DIRECTION_BITS | BTN_FOCUS

NEUTRAL = frozenset({"z"})
HOLD_MAX_US = 250_000

# Pre-registered trial classes (see the preregistration, §3).
TRIAL_CLASSES = (
    {
        "name": "C1_lateral_normal",
        "axis": "x",
        "keys": (frozenset({"z", "right"}), frozenset({"z", "left"})),
        "expected_axis_delta": 4.0,
        "note": "4 px/frame lateral, unfocused",
    },
    {
        "name": "C2_vertical_normal",
        "axis": "y",
        "keys": (frozenset({"z", "up"}), frozenset({"z", "down"})),
        "expected_axis_delta": 4.0,
        "note": "4 px/frame vertical, unfocused",
    },
    {
        "name": "C3_lateral_focus",
        "axis": "x",
        "keys": (frozenset({"z", "shift", "right"}), frozenset({"z", "shift", "left"})),
        "expected_axis_delta": 2.0,
        "note": "2 px/frame lateral, focused",
    },
    {
        "name": "C4_diagonal_normal",
        "axis": "x",
        "keys": (
            frozenset({"z", "right", "up"}),
            frozenset({"z", "left", "down"}),
        ),
        "expected_axis_delta": 4.0 / (2.0**0.5),
        "note": "4/sqrt(2) px/frame per axis, unfocused diagonal",
    },
)

# Boundary pushes: (name, axis, key set, expected clamped value from P0-round8).
BOUNDARY_PUSHES = (
    ("x_max", "x", frozenset({"z", "right"}), 376.0),
    ("x_min", "x", frozenset({"z", "left"}), 8.0),
    ("y_min", "y", frozenset({"z", "up"}), 16.0),
    ("y_max", "y", frozenset({"z", "down"}), 432.0),
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without the game)
# ---------------------------------------------------------------------------
def expected_mask(keys) -> int:
    """Engine input mask implied by a set of key names."""
    mask = 0
    if "z" in keys:
        mask |= BTN_SHOOT
    if "x" in keys:
        mask |= 1 << 1
    if "shift" in keys:
        mask |= BTN_FOCUS
    if "up" in keys:
        mask |= BTN_UP
    if "down" in keys:
        mask |= BTN_DOWN
    if "left" in keys:
        mask |= BTN_LEFT
    if "right" in keys:
        mask |= BTN_RIGHT
    return mask


def compared_mask(mask: int) -> int:
    """The bits a consumption check compares (directions + focus; never shoot)."""
    return int(mask) & COMPARED_BITS


def first_index(rows, predicate, start: int = 1):
    for index in range(start, len(rows)):
        if predicate(rows[index]):
            return index
    return None


def consumption_index(seq, mask: int, start: int = 1):
    """First frame at/after `start` whose engine input shows `mask`'s bits."""
    want = compared_mask(mask)
    return first_index(seq, lambda row: compared_mask(row["input_mask"]) == want, start)


def motion_index(seq, axis: str, anchor_value: float, start: int = 1, eps: float = 0.0):
    """First frame at/after `start` whose `axis` moved away from the anchor."""
    return first_index(seq, lambda row: abs(row[axis] - anchor_value) > eps, start)


def is_static(previous, current) -> bool:
    return (
        previous["x"] == current["x"]
        and previous["y"] == current["y"]
        and current["vx"] == 0.0
        and current["vy"] == 0.0
    )


def ticket_gaps(rows) -> list[int]:
    return [
        rows[i]["ticket"] - rows[i - 1]["ticket"] for i in range(1, len(rows))
    ]


def logic_gaps(rows) -> list[int]:
    return [
        rows[i]["logic_frame"] - rows[i - 1]["logic_frame"] for i in range(1, len(rows))
    ]


def gap_summary(gaps) -> dict:
    if not gaps:
        return {"n": 0, "allOne": False, "counts": {}}
    counts: dict[str, int] = {}
    for gap in gaps:
        counts[str(gap)] = counts.get(str(gap), 0) + 1
    return {
        "n": len(gaps),
        "allOne": all(gap == 1 for gap in gaps),
        "counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "max": max(gaps),
    }


def qpc_rate(rows, frequency: int) -> float | None:
    """Present rate in Hz from the game-side QPC stamps of consecutive frames."""
    if frequency <= 0 or len(rows) < 3:
        return None
    qpc = [row["qpc"] for row in rows]
    if any(later <= earlier for earlier, later in zip(qpc, qpc[1:])):
        return None
    seconds = (qpc[-1] - qpc[0]) / float(frequency)
    if seconds <= 0:
        return None
    return (len(rows) - 1) / seconds


def trial_metrics(seq, keys, axis: str) -> dict:
    """Derive lag_in / lag_mv / per-frame delta for one already-run trial."""
    anchor = seq[0]
    mask = expected_mask(keys)
    consumption = consumption_index(seq, mask, start=1)
    motion = motion_index(seq, axis, anchor[axis], start=1)
    anchor_logic = anchor["logic_frame"]
    metrics = {
        "consumption_index": consumption,
        "motion_index": motion,
        "lag_in": None if consumption is None else seq[consumption]["logic_frame"] - anchor_logic,
        "lag_mv": None if motion is None else seq[motion]["logic_frame"] - anchor_logic,
        "delta": None if motion is None else seq[motion][axis] - anchor[axis],
        "velocity_at_motion": None if motion is None else seq[motion]["v" + axis],
        "anchor": {"x": anchor["x"], "y": anchor["y"], "logic_frame": anchor_logic},
        "consumed_mask": None if consumption is None else seq[consumption]["input_mask"],
        "focus_at_motion": None if motion is None else seq[motion]["focus"],
    }
    if motion is not None:
        sign = 1.0 if seq[motion][axis] > anchor[axis] else -1.0
        metrics["direction"] = 1 if sign > 0 else -1
        # Does the displacement equal the reported velocity for that frame?
        velocity = metrics["velocity_at_motion"]
        metrics["delta_equals_velocity"] = (
            velocity is not None and abs(abs(metrics["delta"]) - abs(velocity)) <= 1e-6
        )
    else:
        metrics["direction"] = None
        metrics["delta_equals_velocity"] = False
    return metrics


def summarize_class(trials, spec) -> dict:
    """Aggregate one pre-registered class against its pass condition."""
    lags_in = [t["metrics"]["lag_in"] for t in trials if t["metrics"]["lag_in"] is not None]
    lags_mv = [t["metrics"]["lag_mv"] for t in trials if t["metrics"]["lag_mv"] is not None]
    deltas = [t["metrics"]["delta"] for t in trials if t["metrics"]["delta"] is not None]
    expected = spec["expected_axis_delta"]
    if spec["name"] == "C4_diagonal_normal":
        deltas_match = [d is not None and abs(abs(d) - expected) <= 1e-4 * expected for d in deltas]
    else:
        deltas_match = [d is not None and abs(abs(d) - expected) == 0.0 for d in deltas]
    lag_in_ok = [lag == 1 for lag in lags_in]
    lag_mv_ok = [
        t["metrics"]["lag_in"] is not None
        and t["metrics"]["lag_mv"] is not None
        and t["metrics"]["lag_mv"] >= t["metrics"]["lag_in"]
        and t["metrics"]["lag_mv"] <= t["metrics"]["lag_in"] + 1
        for t in trials
    ]
    return {
        "class": spec["name"],
        "note": spec["note"],
        "axis": spec["axis"],
        "expectedAxisDelta": expected,
        "trials": len(trials),
        "completed": len(lags_in),
        "lagInCounts": {str(v): lags_in.count(v) for v in sorted(set(lags_in))},
        "lagMvCounts": {str(v): lags_mv.count(v) for v in sorted(set(lags_mv))},
        "deltas": sorted(abs(d) for d in deltas),
        "deltaMatch": sum(deltas_match),
        "lagInMatch": sum(lag_in_ok),
        "lagMvOk": sum(lag_mv_ok),
        "pass_lag_in": bool(trials) and all(lag_in_ok) and len(lag_in_ok) == len(trials),
        "pass_lag_mv": bool(trials) and all(lag_mv_ok),
        "pass_delta": bool(trials) and all(deltas_match),
    }


def saturation_from_rows(rows, axis: str, direction: int):
    """Find the clamped value a saturating push settled on.

    Returns the value at the last frame that still moved, the number of static
    frames after it, and the observed extremes.  A push only counts as
    saturated when at least two consecutive trailing frames are static.
    """
    values = [row[axis] for row in rows]
    last_move = None
    for index in range(1, len(values)):
        if values[index] != values[index - 1]:
            last_move = index
    if last_move is None:
        return {
            "saturated": False,
            "reason": "never moved",
            "boundary": None,
            "staticTail": len(values),
            "observedMin": min(values),
            "observedMax": max(values),
        }
    static_tail = len(values) - 1 - last_move
    boundary = values[-1]
    monotone = all(
        (values[i] - values[i - 1]) * direction >= 0 for i in range(1, len(values))
    )
    return {
        "saturated": static_tail >= 2,
        "reason": "" if static_tail >= 2 else "did not settle",
        "boundary": boundary,
        "lastMoveValue": values[last_move],
        "staticTail": static_tail,
        "monotone": monotone,
        "observedMin": min(values),
        "observedMax": max(values),
    }


def carry_metrics(seq, axis: str, old_direction: int) -> dict:
    """Frames that still move in the *old* direction after the reversal frame.

    `seq[0]` is the last frame before the new keys were typed (still moving in
    the old direction); `seq[1:]` are the frames after the reversal.
    """
    deltas = [seq[i][axis] - seq[i - 1][axis] for i in range(1, len(seq))]
    carry_frames = 0
    carry_px = 0.0
    for delta in deltas:
        if delta * old_direction > 0:
            carry_frames += 1
            carry_px += delta
        else:
            break
    return {
        "deltas": deltas,
        "oldCarryFrames": carry_frames,
        "oldCarryPx": carry_px,
        "reversedOnNextFrame": len(deltas) > carry_frames
        and deltas[carry_frames] * old_direction < 0,
    }


def check_summary(samples) -> dict:
    """Per-check pass counts over a list of fence samples."""
    names: dict[str, list[bool]] = {}
    for sample in samples:
        for name, value in sample["checks"].items():
            names.setdefault(name, []).append(bool(value))
    return {
        name: {"passed": sum(values), "total": len(values)}
        for name, values in sorted(names.items())
    }


def sample_all_ok(sample) -> bool:
    return bool(sample["checks"]) and all(sample["checks"].values())


# ---------------------------------------------------------------------------
# Windows helpers
# ---------------------------------------------------------------------------
def qpc_frequency() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    value = ctypes.c_int64()
    if not kernel32.QueryPerformanceFrequency(ctypes.byref(value)):
        return 0
    return int(value.value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def text_section_sha256(path: Path) -> str:
    """SHA-256 of the PE .text section (the part that must stay canonical)."""
    import struct

    raw = Path(path).read_bytes()
    if raw[:2] != b"MZ":
        raise ValueError(f"not a PE file: {path}")
    pe_offset = struct.unpack_from("<I", raw, 0x3C)[0]
    if raw[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ValueError(f"missing PE signature: {path}")
    section_count = struct.unpack_from("<H", raw, pe_offset + 6)[0]
    optional_size = struct.unpack_from("<H", raw, pe_offset + 20)[0]
    table = pe_offset + 24 + optional_size
    for index in range(section_count):
        entry = table + index * 40
        name = raw[entry : entry + 8].rstrip(b"\0").decode("ascii", "replace")
        if name == ".text":
            virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from(
                "<IIII", raw, entry + 8
            )
            del virtual_size, virtual_address
            return hashlib.sha256(raw[raw_pointer : raw_pointer + raw_size]).hexdigest().upper()
    raise ValueError(f"no .text section in {path}")


def environment_fingerprint() -> dict:
    fingerprint = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            timeout=10,
            text=True,
        )
        fingerprint["gpu"] = output.stdout.strip().splitlines()[:2]
    except Exception as exc:  # pragma: no cover - diagnostics only
        fingerprint["gpu"] = f"unavailable: {exc!r}"
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, timeout=10, text=True
        )
        fingerprint["gitHead"] = output.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, timeout=10, text=True
        )
        fingerprint["gitDirtyFiles"] = len(
            [line for line in dirty.stdout.splitlines() if line.strip()]
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        fingerprint["gitHead"] = f"unavailable: {exc!r}"
    return fingerprint


# ---------------------------------------------------------------------------
# Fence clock: one held frame at a time, with an optional key change
# ---------------------------------------------------------------------------
class ProbeAborted(RuntimeError):
    """The live session can no longer produce valid evidence."""


class HitInterrupted(ProbeAborted):
    """The player was hit (dying/spawning/invulnerable) inside a measurement.

    The interrupted trial is not a data point: it is discarded and retried after
    the player is alive again.  Every hit is logged and counted.
    """

    def __init__(self, row, phase: str = ""):
        super().__init__(
            f"player_state={row.get('player_state')} at frame {row.get('logic_frame')}"
        )
        self.row = row
        self.phase = phase


class FenceClock:
    """Consume held frames one by one; optionally type keys inside the hold."""

    def __init__(
        self, session, fence, *, hold_max_us: int = HOLD_MAX_US, require_foreground: bool = False
    ):
        self.session = session
        self.fence = fence
        self.hold_max_us = hold_max_us
        self.require_foreground = require_foreground
        self.last_ticket = 0
        self.rows: list[dict] = []
        self.missed = 0
        self.budget_expired = 0
        self.foreground_reclaims = 0

    def _ensure_foreground(self) -> None:
        """Keys only reach the game through SendInput while it is foreground."""
        from touhou_rl import winapi

        if winapi.user32.GetForegroundWindow() == self.session.hwnd:
            return
        if winapi.force_foreground(self.session.hwnd):
            self.foreground_reclaims += 1
            return
        raise ProbeAborted("game window lost foreground; refusing to type blind")

    def wait_ticket(self, timeout: float = 3.0):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            ticket = self.fence.ticket
            if ticket and ticket != self.last_ticket and self.fence.hold_phase == 1:
                return ticket
        return 0

    def step(self, keys=None, timeout: float = 3.0):
        if self.require_foreground:
            self._ensure_foreground()
        ticket = self.wait_ticket(timeout)
        if not ticket:
            self.missed += 1
            return None
        telemetry = self.fence.telemetry()
        if telemetry.ticket != ticket or telemetry.hold_phase != 1:
            self.missed += 1
            self.fence.ack(ticket)
            return None
        if keys is not None:
            self.session.keyboard.set_keys(set(keys))
        held = self.fence.ticket == ticket and self.fence.hold_phase == 1
        if held:
            self.fence.ack(ticket)
        else:
            # The hold budget expired while we worked: release whatever hold is
            # in force and drop the frame instead of pretending it is valid.
            self.budget_expired += 1
            self.fence.ack(self.fence.ticket)
            self.last_ticket = ticket
            return None
        self.last_ticket = ticket
        row = {
            "ticket": int(telemetry.ticket),
            "logic_frame": int(telemetry.logic_frame),
            "stage_frames": int(telemetry.stage_frames),
            "run_frames": int(telemetry.run_frames),
            "x": float(telemetry.player_x),
            "y": float(telemetry.player_y),
            "vx": float(telemetry.player_vx),
            "vy": float(telemetry.player_vy),
            "player_state": int(telemetry.player_state),
            "focus": int(telemetry.focus_mode),
            "input_mask": int(telemetry.input_mask),
            "gui_input_current": int(telemetry.gui_input_current),
            "active_bullets": int(telemetry.active_bullets),
            "hold_last_us": int(telemetry.hold_last_us),
            "hold_budget_skips": int(telemetry.hold_budget_skips),
            "qpc": int(telemetry.qpc),
        }
        self.rows.append(row)
        return row


# ---------------------------------------------------------------------------
# Evidence bookkeeping
# ---------------------------------------------------------------------------
class Evidence:
    def __init__(self, out_dir: Path):
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.data: dict = {
            "schema": "v373-p0-fence-round26/1",
            "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "preregistration": "Document/Update/Alpha/v3.7.3/preregistration/p0-fence-round26.md",
            "build": self._build_identity(),
            "environment": environment_fingerprint(),
            "sessions": {},
        }

    @staticmethod
    def _build_identity() -> dict:
        identity = {
            "image": "third_party/th08/build_fence/th08.exe",
            "canonical": False,
            "note": (
                "research-only fence variant; every global is +0x3000 relative to "
                "the canonical image, see third_party/th08/docs/FRAME_FENCE.md"
            ),
        }
        for key, path in (
            ("exeBytes", FENCE_EXE),
            ("addressTableBytes", FENCE_ADDRESSES),
            ("mapBytes", FENCE_MAP),
        ):
            identity[key] = path.stat().st_size if path.is_file() else None
        if FENCE_EXE.is_file():
            identity["exeSha256"] = sha256_file(FENCE_EXE)
            identity["textSha256"] = text_section_sha256(FENCE_EXE)
        if FENCE_ADDRESSES.is_file():
            identity["addressTableSha256"] = sha256_file(FENCE_ADDRESSES)
        if FENCE_MAP.is_file():
            identity["mapSha256"] = sha256_file(FENCE_MAP)
        return identity

    def write_json(self, name: str, payload) -> Path:
        path = self.out / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        return path

    def write_jsonl(self, name: str, rows) -> Path:
        path = self.out / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return path


# ---------------------------------------------------------------------------
# Boot helpers
# ---------------------------------------------------------------------------
def run_dir_for(name: str) -> Path:
    """Session directories live under ``.tmp``: they hold the game assets."""
    directory = ROOT / ".tmp" / "p0-fence-round26" / f"run-{name}"
    directory.parent.mkdir(parents=True, exist_ok=True)
    return directory


def _boot_session(run_dir: Path, template: Path, difficulty: int, *, postmessage: bool):
    from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir

    prepare_run_dir(
        run_dir,
        template,
        windowed=True,
        default_difficulty=difficulty,
        use_postmessage_input=postmessage,
        force=True,
    )
    session = GameSession(SessionOptions(run_dir=run_dir, windowed=True, boot_timeout_s=45.0))
    session.start()
    return session


def _open_fence(session):
    from touhou_rl.framefence import FrameFence

    fence = FrameFence.open(timeout=20.0)
    if fence.header()["pid"] != session.pid:
        fence.close()
        raise ProbeAborted("stale fence section: pid mismatch")
    fence.configure(enable=True, hold_every=0)
    return fence


def _arm_holds(fence, *, pixels: bool) -> int:
    """Acknowledge one ordinary ticket, then start holding every drawn frame."""
    start = fence.ticket
    deadline = time.perf_counter() + 5.0
    while fence.ticket == start:
        if time.perf_counter() > deadline:
            raise ProbeAborted("fence ticket did not advance while arming holds")
        time.sleep(0.0005)
    fence.ack(fence.ticket)
    time.sleep(0.01)
    fence.configure(
        enable=True,
        pixels=pixels,
        hold_every=1,
        hold_min_us=0,
        hold_max_us=HOLD_MAX_US,
    )
    return fence.ticket


def _assert_alive(state) -> None:
    if int(state.player_state) != 0:
        raise ProbeAborted(
            f"player_state={state.player_state} at frame {state.gameplay_frame}"
        )


def _assert_playing(state) -> None:
    if not state.is_playing():
        raise ProbeAborted(
            f"not playing at frame {state.gameplay_frame} "
            f"(supervisor={state.supervisor_state})"
        )


# ---------------------------------------------------------------------------
# Demo session: M1 clock, M5 pixel/state samples, M6 legacy control
# ---------------------------------------------------------------------------
def run_demo_session(evidence: Evidence, args) -> dict:
    from frame_fence_round1 import prepare_fence_template
    from touhou_rl import spec
    from touhou_rl.framefence import FenceFrameSampler, FenceStateDigest, legacy_sample

    result: dict = {"mode": "demo", "phases": {}}
    template = prepare_fence_template()
    session = None
    fence = None
    try:
        session = _boot_session(run_dir_for("demo"), template, spec.LUNATIC, postmessage=True)
        fence = _open_fence(session)
        result["inputBackend"] = type(session.keyboard).__name__
        result["clientSize"] = list(session.client_size)

        deadline = time.perf_counter() + 45.0
        while time.perf_counter() < deadline:
            state = session.read_state(with_bullets=False)
            if state.supervisor_state == spec.STATE_GAME_MANAGER:
                break
            time.sleep(0.05)
        else:
            raise ProbeAborted("attract demo never started")
        time.sleep(args.demo_warmup)
        state = session.read_state(with_bullets=False)
        result["demoState"] = {
            "stage": int(state.stage),
            "isDemo": bool(state.is_demo),
            "isReplay": bool(state.is_replay),
            "activeBullets": int(state.active_bullets),
            "playerState": int(state.player_state),
        }

        # -- M1: exact clock, no keys, telemetry only ----------------------
        # The attract demo may lose a life; that does not weaken G1/G2/G3 (which
        # are about ticket/logic-frame/present-rate) or the pixel proof, so the
        # trace is not gated on `player_state`.  Every value is recorded.
        _arm_holds(fence, pixels=False)
        clock = FenceClock(session, fence)
        trace = []
        for _ in range(args.frames):
            row = clock.step()
            if row is None:
                continue
            trace.append(row)
        frequency = qpc_frequency()
        rate = qpc_rate(trace, frequency)
        result["phases"]["M1_clock"] = {
            "frames": len(trace),
            "missing": clock.missed,
            "budgetExpired": clock.budget_expired,
            "qpcFrequency": frequency,
            "presentHz": rate,
            "ticketGaps": gap_summary(ticket_gaps(trace)),
            "logicGaps": gap_summary(logic_gaps(trace)),
            "holdUs": {
                "median": statistics.median([r["hold_last_us"] for r in trace]) if trace else None,
                "max": max((r["hold_last_us"] for r in trace), default=None),
            },
            "holdBudgetSkips": trace[-1]["hold_budget_skips"] if trace else None,
            "distinctTickets": len({r["ticket"] for r in trace}),
            "playerStates": {str(v): [r["player_state"] for r in trace].count(v)
                             for v in sorted({r["player_state"] for r in trace})},
        }
        evidence.write_jsonl("trace.jsonl", trace)

        # -- M5: pixel/state samples with the 13 checks --------------------
        fence.configure(
            enable=True, pixels=True, hold_every=1, hold_min_us=0, hold_max_us=HOLD_MAX_US
        )
        sampler = FenceFrameSampler(session, fence, with_pixels=True, with_enemies=True)
        samples = []
        png_dir = evidence.out / "samples"
        png_dir.mkdir(parents=True, exist_ok=True)
        attempts = args.samples * 3
        while len(samples) < args.samples and attempts > 0:
            attempts -= 1
            sample = sampler.next(timeout=3.0, attempts=12)
            if sample is None or not sample.ok:
                result["discardedSamples"] = result.get("discardedSamples", 0) + 1
                continue
            telemetry = sample.telemetry
            assert telemetry is not None and sample.pixels is not None
            index = len(samples)
            image_name = f"{index:04d}-ticket-{sample.ticket:06d}.png"
            image_path = png_dir / image_name
            try:
                from PIL import Image

                Image.fromarray(sample.pixels).save(image_path, compress_level=6)
            except Exception as exc:  # pragma: no cover - diagnostics only
                result.setdefault("imageErrors", []).append(repr(exc))
                continue
            samples.append(
                {
                    "index": index,
                    "ticket": int(sample.ticket),
                    "logicFrame": int(telemetry.logic_frame),
                    "stageFrames": int(telemetry.stage_frames),
                    "player": {
                        "x": float(telemetry.player_x),
                        "y": float(telemetry.player_y),
                        "vx": float(telemetry.player_vx),
                        "vy": float(telemetry.player_vy),
                        "state": int(telemetry.player_state),
                        "focus": int(telemetry.focus_mode),
                    },
                    "inputMask": int(telemetry.input_mask),
                    "activeBullets": int(telemetry.active_bullets),
                    "bulletDigest": int(telemetry.bullet_digest),
                    "enemyDigest": int(telemetry.enemy_digest),
                    "digestTicket": int(telemetry.digest_ticket),
                    "pixelTicket": int(telemetry.pixel_ticket),
                    "holdUs": int(telemetry.hold_last_us),
                    "image": f"samples/{image_name}",
                    "imageSha256": hashlib.sha256(sample.pixels.tobytes()).hexdigest().upper(),
                    "pngSha256": sha256_file(image_path),
                    "checks": dict(sample.checks),
                    "timingsUs": {k: round(v, 1) for k, v in sample.timings_us.items()},
                }
            )
        result["phases"]["M5_samples"] = {
            "requested": args.samples,
            "accepted": len(samples),
            "discarded": result.get("discardedSamples", 0),
            "samplerRetries": sampler.retries,
            "checkCounts": check_summary(samples),
            "allChecksPassed": sum(1 for s in samples if sample_all_ok(s)),
            "distinctTickets": len({s["ticket"] for s in samples}),
            "distinctImages": len({s["imageSha256"] for s in samples}),
        }
        evidence.write_jsonl("samples.jsonl", samples)

        # -- M6: the old read style, no hold -------------------------------
        fence.configure(enable=True, hold_every=0)
        time.sleep(0.05)
        digest = FenceStateDigest(session.reader)
        legacy = []
        for _ in range(args.legacy):
            legacy.append(legacy_sample(session, fence, digest))
        result["phases"]["M6_legacy"] = {
            "reads": len(legacy),
            "ticketChanged": sum(1 for row in legacy if not row["ticket_unchanged"]),
            "bulletDigestMismatch": sum(1 for row in legacy if not row["bullet_digest_match"]),
            "enemyDigestMismatch": sum(1 for row in legacy if not row["enemy_digest_match"]),
            "playerMismatch": sum(1 for row in legacy if not row["player_match"]),
            "frameMismatch": sum(1 for row in legacy if not row["frame_match"]),
        }
        evidence.write_jsonl("legacy.jsonl", legacy)
        result["fenceFinal"] = {
            "ticket": fence.ticket,
            "holdCount": fence.telemetry().hold_count,
            "holdBudgetSkips": fence.telemetry().hold_budget_skips,
        }
        result["status"] = "completed"
    except ProbeAborted as exc:
        result["status"] = "aborted"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if fence is not None:
            try:
                fence.disable()
            except Exception:
                pass
            fence.close()
        if session is not None:
            session.close()
    return result


def _assert_alive_state(row, phase: str = "") -> None:
    if int(row["player_state"]) != 0:
        raise HitInterrupted(row, phase)


def _recover_after_hit(clock, session, timeout: float = 12.0):
    """Wait for the player to be alive again (respawn invulnerability helps)."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        row = clock.step(keys=NEUTRAL)
        if row is None:
            continue
        state = session.read_state(with_bullets=False, with_title=False)
        if not state.is_playing():
            raise ProbeAborted("game left the playing state after a hit")
        if row["player_state"] == 0:
            return row
    raise ProbeAborted("player did not become alive again after a hit")


def _with_hit_retry(run, clock, session, hits: list, max_hits: int, label: str):
    """Run one measurement; a player hit discards and retries it."""
    while True:
        try:
            return run()
        except HitInterrupted as hit:
            hits.append(
                {
                    "phase": label,
                    "logicFrame": hit.row.get("logic_frame"),
                    "playerState": hit.row.get("player_state"),
                    "ticket": hit.row.get("ticket"),
                }
            )
            if len(hits) > max_hits:
                raise ProbeAborted(
                    f"hit budget exhausted ({len(hits)} hits, max {max_hits}) in {label}"
                ) from None
            _recover_after_hit(clock, session)


# ---------------------------------------------------------------------------
# Practice session: M2/M3 trials, M4 boundaries / carry
# ---------------------------------------------------------------------------
def _wait_controllable(session, clock, probe_timeout: float = 25.0) -> dict:
    """Wait until typed keys actually move the player (stage intro skipped)."""
    from touhou_rl import spec, winapi

    started = time.perf_counter()
    attempts = 0
    if not winapi.force_foreground(session.hwnd):
        raise ProbeAborted("cannot bring the game window to the foreground")
    while time.perf_counter() - started < probe_timeout:
        attempts += 1
        clock._ensure_foreground()
        before = session.read_state(with_bullets=False, with_title=False)
        _assert_alive(before)
        session.keyboard.set_keys({"z", "left"})
        time.sleep(0.07)
        after = session.read_state(with_bullets=False, with_title=False)
        session.keyboard.set_keys(set())
        if not after.is_playing():
            raise ProbeAborted(f"left playing state {after.supervisor_state}")
        if after.player_x != before.player_x:
            session.keyboard.set_keys({"z"})
            return {
                "attempts": attempts,
                "waitedSeconds": round(time.perf_counter() - started, 3),
                "x": float(after.player_x),
                "y": float(after.player_y),
            }
        time.sleep(0.1)
    raise ProbeAborted("player never responded to keys")


def _settle(clock, wanted: int = 2, limit: int = 12):
    """Step neutral frames until `wanted` consecutive frames are fully static."""
    rows = []
    static = 0
    for _ in range(limit):
        row = clock.step(keys=NEUTRAL)
        if row is None:
            continue
        _assert_alive_state(row)
        rows.append(row)
        if len(rows) >= 2 and is_static(rows[-2], rows[-1]):
            static += 1
            if static >= wanted:
                return rows, True
        else:
            static = 0
    return rows, static >= wanted


def _run_trial(clock, spec, keys) -> dict:
    rows, settled = _settle(clock)
    trial = {"class": spec["name"], "settled": settled, "keys": sorted(keys)}
    if not settled:
        trial["error"] = "no static anchor"
        return trial
    anchor = clock.step(keys=keys)
    if anchor is None:
        trial["error"] = "anchor frame lost"
        return trial
    _assert_alive_state(anchor)
    after = []
    for _ in range(3):
        row = clock.step(keys=keys)
        if row is None:
            break
        _assert_alive_state(row)
        after.append(row)
    release = []
    for _ in range(3):
        row = clock.step(keys=NEUTRAL)
        if row is None:
            break
        _assert_alive_state(row)
        release.append(row)
    seq = [anchor] + after
    trial["sequence"] = seq
    trial["release"] = release
    trial["metrics"] = trial_metrics(seq, keys, spec["axis"])
    trial["release_static"] = (
        len(release) >= 2
        and release[-1]["x"] == release[-2]["x"]
        and release[-1]["y"] == release[-2]["y"]
    )
    return trial


def _boundary_push(clock, axis: str, keys, limit: int = 170) -> dict:
    rows = []
    row = clock.step(keys=keys)
    if row is None:
        return {"error": "first frame lost"}
    _assert_alive_state(row)
    rows.append(row)
    for _ in range(limit):
        row = clock.step(keys=keys)
        if row is None:
            continue
        _assert_alive_state(row)
        rows.append(row)
        if len(rows) >= 4:
            tail = rows[-3:]
            if all(tail[i][axis] == tail[i - 1][axis] for i in range(1, len(tail))):
                break
    return {"rows": rows}


def _ensure_room(clock, axis: str, center: float, margin: float = 40.0, limit: int = 150) -> dict:
    """Move away from a wall before a carry trial.

    A reversal that starts against a clamped edge produces a zero delta and looks
    like "no carry at all"; recentring is what makes the four directions
    comparable.
    """
    row = clock.step(keys=NEUTRAL)
    if row is None:
        raise ProbeAborted("recentre: no frame available")
    _assert_alive_state(row)
    value = row[axis]
    steps = 0
    while abs(value - center) > margin and steps < limit:
        if axis == "x":
            keys = {"z", "right"} if value < center else {"z", "left"}
        else:
            keys = {"z", "down"} if value < center else {"z", "up"}
        row = clock.step(keys=frozenset(keys))
        steps += 1
        if row is None:
            continue
        _assert_alive_state(row)
        value = row[axis]
    clock.step(keys=NEUTRAL)
    return {"axis": axis, "value": value, "steps": steps}


def _reversal_trial(clock, axis: str, push_keys, reverse_keys, old_direction: int) -> dict:
    center = 192.0 if axis == "x" else 224.0
    recentre = _ensure_room(clock, axis, center)
    # Push a few frames away from a wall so the old-direction carry is visible.
    rows = []
    row = clock.step(keys=push_keys)
    if row is None:
        return {"error": "first frame lost", "recentre": recentre}
    _assert_alive_state(row)
    rows.append(row)
    for _ in range(4):
        row = clock.step(keys=push_keys)
        if row is None:
            continue
        _assert_alive_state(row)
        rows.append(row)
    # The reversal frame: telemetry still describes the old direction (this frame
    # was updated before the new keys could be read), the keys typed during its
    # hold are the new ones.  The carry is measured from THIS frame onwards, so
    # the delta that belongs to the old direction before the reversal is not
    # counted as a carry frame.
    anchor = clock.step(keys=reverse_keys)
    if anchor is None:
        return {"error": "reversal anchor lost"}
    _assert_alive_state(anchor)
    seq = [anchor]
    for _ in range(2):
        row = clock.step(keys=reverse_keys)
        if row is None:
            break
        _assert_alive_state(row)
        seq.append(row)
    # Restore neutral so the next trial starts from rest.
    for _ in range(2):
        clock.step(keys=NEUTRAL)
    return {
        "approach": rows,
        "sequence": seq,
        "recentre": recentre,
        "carry": carry_metrics(seq, axis, old_direction),
    }


def _axis_switch_trial(clock, axis: str, old_keys, new_keys, old_direction: int,
                       new_axis: str) -> dict:
    center = 192.0 if axis == "x" else 224.0
    recentre = _ensure_room(clock, axis, center)
    new_center = 192.0 if new_axis == "x" else 224.0
    recentre_new = _ensure_room(clock, new_axis, new_center)
    rows = []
    row = clock.step(keys=old_keys)
    if row is None:
        return {"error": "first frame lost", "recentre": recentre}
    _assert_alive_state(row)
    rows.append(row)
    for _ in range(4):
        row = clock.step(keys=old_keys)
        if row is None:
            continue
        _assert_alive_state(row)
        rows.append(row)
    anchor = clock.step(keys=new_keys)
    if anchor is None:
        return {"error": "switch anchor lost"}
    _assert_alive_state(anchor)
    seq = [anchor]
    for _ in range(2):
        row = clock.step(keys=new_keys)
        if row is None:
            break
        _assert_alive_state(row)
        seq.append(row)
    for _ in range(2):
        clock.step(keys=NEUTRAL)
    return {
        "approach": rows,
        "sequence": seq,
        "recentre": recentre,
        "recentreNewAxis": recentre_new,
        "carryOldAxis": carry_metrics(seq, axis, old_direction),
        "newAxisMovement": [
            seq[i][new_axis] - seq[i - 1][new_axis] for i in range(1, len(seq))
        ],
    }


def _task_queue(args) -> list[dict]:
    """Every pre-registered measurement as an independently resumable task."""
    tasks: list[dict] = []
    for trial_spec in TRIAL_CLASSES:
        for index in range(args.per_class):
            tasks.append(
                {
                    "kind": "trial",
                    "label": trial_spec["name"],
                    "spec": trial_spec,
                    "keys": trial_spec["keys"][index % 2],
                }
            )
    for name, axis, keys, expected in BOUNDARY_PUSHES:
        tasks.append(
            {
                "kind": "boundary",
                "label": name,
                "name": name,
                "axis": axis,
                "keys": keys,
                "expected": expected,
            }
        )
    reversal_plan = (
        ("x_right_to_left", "x", frozenset({"z", "right"}), frozenset({"z", "left"}), 1),
        ("x_left_to_right", "x", frozenset({"z", "left"}), frozenset({"z", "right"}), -1),
        ("y_down_to_up", "y", frozenset({"z", "down"}), frozenset({"z", "up"}), 1),
        ("y_up_to_down", "y", frozenset({"z", "up"}), frozenset({"z", "down"}), -1),
    )
    for name, axis, push_keys, reverse_keys, old_direction in reversal_plan:
        for _ in range(args.carry_repeats):
            tasks.append(
                {
                    "kind": "reversal",
                    "label": name,
                    "name": name,
                    "axis": axis,
                    "pushKeys": push_keys,
                    "reverseKeys": reverse_keys,
                    "oldDirection": old_direction,
                }
            )
    switch_plan = (
        ("right_to_up", "x", frozenset({"z", "right"}), frozenset({"z", "up"}), 1, "y"),
        ("down_to_left", "y", frozenset({"z", "down"}), frozenset({"z", "left"}), 1, "x"),
    )
    for name, axis, old_keys, new_keys, old_direction, new_axis in switch_plan:
        for _ in range(args.carry_repeats):
            tasks.append(
                {
                    "kind": "switch",
                    "label": name,
                    "name": name,
                    "axis": axis,
                    "oldKeys": old_keys,
                    "newKeys": new_keys,
                    "oldDirection": old_direction,
                    "newAxis": new_axis,
                }
            )
    return tasks


def _execute_task(clock, task: dict):
    kind = task["kind"]
    if kind == "trial":
        return _run_trial(clock, task["spec"], task["keys"])
    if kind == "boundary":
        return _boundary_push(clock, task["axis"], task["keys"])
    if kind == "reversal":
        return _reversal_trial(
            clock, task["axis"], task["pushKeys"], task["reverseKeys"], task["oldDirection"]
        )
    if kind == "switch":
        return _axis_switch_trial(
            clock,
            task["axis"],
            task["oldKeys"],
            task["newKeys"],
            task["oldDirection"],
            task["newAxis"],
        )
    raise ProbeAborted(f"unknown task kind {kind!r}")


def _merge_gap_summaries(summaries: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for summary in summaries:
        for key, value in summary.get("counts", {}).items():
            counts[key] = counts.get(key, 0) + value
    return {
        "n": sum(summary.get("n", 0) for summary in summaries),
        "allOne": bool(summaries) and all(summary.get("allOne") for summary in summaries),
        "counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "max": max((summary.get("max", 0) for summary in summaries), default=None),
    }


def _start_practice_game(attempt: int, args) -> tuple:
    """Boot one practice game and return (session, fence, clock, info)."""
    from frame_fence_round1 import prepare_fence_template
    from touhou_rl import menu, spec, winapi

    template = prepare_fence_template()
    # SendInput + foreground: the title menu only reacts to the focused window
    # (the previous P0 probes used the same path).
    session = _boot_session(
        run_dir_for(f"practice-{attempt}"), template, spec.LUNATIC, postmessage=False
    )
    fence = _open_fence(session)
    if not winapi.force_foreground(session.hwnd):
        raise ProbeAborted("cannot bring the game window to the foreground")
    menu.enter_game(session, difficulty=spec.LUNATIC, practice_stage=0)
    # player_state: alive=0, spawning=1, dying=2, invulnerable=3 (state.py:98).
    try:
        state = session.wait_for(
            lambda current: current.player_state == 0, timeout=20.0, description="player alive"
        )
    except TimeoutError:
        raise ProbeAborted("player never reached the alive state after practice start") from None
    _assert_playing(state)
    info = {
        "startStage": int(state.stage),
        "startLives": int(state.lives),
        "startDeaths": int(state.deaths),
        "startGameplayFrame": int(state.gameplay_frame),
    }
    clock = FenceClock(session, fence, require_foreground=True)
    info["controllable"] = _wait_controllable(session, clock)
    # Holds must be armed *after* the menu work: the title/menu loops would
    # otherwise be held frame by frame, and a session without holds makes every
    # measurement wait for a `hold_phase` that never appears.
    _arm_holds(fence, pixels=False)
    clock.last_ticket = 0
    return session, fence, clock, info


def run_practice_session(evidence: Evidence, args) -> dict:
    """Practice measurements, resumable across games.

    Stage 1 Lunatic kills a nearly-static player: the first attempt lost all
    eight lives before the boundary phase.  Instead of aborting the round (and
    instead of retrying until the numbers look good), the task queue is resumed
    in a fresh game: every trial is either complete or absent, never patched.
    """
    result: dict = {"mode": "practice", "phases": {}}
    tasks = _task_queue(args)
    total_tasks = len(tasks)
    collected: dict = {
        "trials": {spec["name"]: [] for spec in TRIAL_CLASSES},
        "boundaries": [],
        "carry": {"reversals": [], "axisSwitches": []},
    }
    hits: list[dict] = []
    sessions_log: list[dict] = []
    gap_summaries: list[dict] = []
    attempt = 0
    while tasks and attempt < args.max_sessions:
        attempt += 1
        log = {"attempt": attempt, "tasksAtStart": len(tasks)}
        session = None
        fence = None
        clock = None
        try:
            session, fence, clock, info = _start_practice_game(attempt, args)
            log.update(info)
            session_deadline = time.perf_counter() + args.session_timeout
            consecutive_empty = 0
            while tasks:
                if time.perf_counter() > session_deadline:
                    raise ProbeAborted(
                        f"session time budget ({args.session_timeout:.0f}s) exhausted"
                    )
                task = tasks[0]
                try:
                    outcome = _execute_task(clock, task)
                except HitInterrupted as hit:
                    hits.append(
                        {
                            "attempt": attempt,
                            "phase": task["label"],
                            "kind": task["kind"],
                            "logicFrame": hit.row.get("logic_frame"),
                            "playerState": hit.row.get("player_state"),
                            "ticket": hit.row.get("ticket"),
                            "tasksRemaining": len(tasks),
                        }
                    )
                    if len(hits) > args.max_hits:
                        raise ProbeAborted(
                            f"hit budget exhausted ({len(hits)} > {args.max_hits})"
                        ) from None
                    _recover_after_hit(clock, session)
                    continue
                tasks.pop(0)
                if task["kind"] == "trial" and outcome.get("error"):
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        raise ProbeAborted(
                            "three consecutive trials produced no measurement "
                            "(are the holds armed?)"
                        ) from None
                else:
                    consecutive_empty = 0
                if task["kind"] == "trial":
                    collected["trials"][task["spec"]["name"]].append(outcome)
                elif task["kind"] == "boundary":
                    rows = outcome.get("rows", [])
                    if not rows:
                        collected["boundaries"].append(
                            {"name": task["name"], "error": outcome.get("error")}
                        )
                    else:
                        direction = 1 if task["keys"] & {"right", "down"} else -1
                        saturation = saturation_from_rows(rows, task["axis"], direction)
                        saturation.update(
                            {
                                "name": task["name"],
                                "axis": task["axis"],
                                "expected": task["expected"],
                                "match": saturation["boundary"] is not None
                                and saturation["boundary"] == task["expected"],
                                "frames": len(rows),
                                "attempt": attempt,
                                "rows": rows,
                            }
                        )
                        collected["boundaries"].append(saturation)
                    evidence.write_json("boundaries.json", collected["boundaries"])
                else:
                    block_key = "reversals" if task["kind"] == "reversal" else "axisSwitches"
                    blocks = collected["carry"][block_key]
                    block = next((b for b in blocks if b["name"] == task["name"]), None)
                    if block is None:
                        block = {"name": task["name"], "trials": []}
                        blocks.append(block)
                    outcome["attempt"] = attempt
                    block["trials"].append(outcome)
                    evidence.write_json("carry.json", collected["carry"])
                evidence.write_json("trials.json", result["phases"].get("M2_M3_trials", {}))
            log["end"] = "all tasks completed"
        except Exception as exc:  # noqa: BLE001 - a dead attempt must not erase the rest
            log["end"] = f"{type(exc).__name__}: {exc}"
            log["tasksRemaining"] = len(tasks)
        finally:
            if clock is not None:
                gap_summaries.append(
                    {
                        "ticket": gap_summary(ticket_gaps(clock.rows)),
                        "logic": gap_summary(logic_gaps(clock.rows)),
                        "missed": clock.missed,
                        "budgetExpired": clock.budget_expired,
                        "frames": len(clock.rows),
                    }
                )
            if fence is not None:
                try:
                    fence.disable()
                except Exception:
                    pass
                try:
                    fence.close()
                except Exception:
                    pass
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
        sessions_log.append(log)
        if getattr(args, "session_pause", 0):
            time.sleep(args.session_pause)

    classes = {
        "classes": [],
        "raw": collected["trials"],
        "tasksTotal": total_tasks,
        "tasksRemaining": len(tasks),
        "sessions": sessions_log,
        "interruptions": hits,
    }
    for trial_spec in TRIAL_CLASSES:
        classes["classes"].append(
            summarize_class(collected["trials"][trial_spec["name"]], trial_spec)
        )
    result["phases"]["M2_M3_trials"] = classes
    result["phases"]["M4_boundaries"] = collected["boundaries"]
    result["phases"]["M4_carry"] = collected["carry"]
    result["interruptions"] = hits
    result["sessions"] = sessions_log
    result["inputBackend"] = "SendInputKeyboard"
    result["clockAfterTrials"] = {
        "ticketGaps": _merge_gap_summaries([gap["ticket"] for gap in gap_summaries]),
        "logicGaps": _merge_gap_summaries([gap["logic"] for gap in gap_summaries]),
        "sessions": gap_summaries,
    }
    result["status"] = "completed" if not tasks else "partial"
    if tasks:
        result["error"] = (
            f"stopped after {attempt} game sessions with {len(tasks)} of {total_tasks} "
            "measurements still missing"
        )
    evidence.write_json("trials.json", classes)
    return result


# ---------------------------------------------------------------------------
# Verdict (pre-registered gates G1..G11)
# ---------------------------------------------------------------------------
def build_verdict(data: dict) -> dict:
    demo = data["sessions"].get("demo", {})
    practice = data["sessions"].get("practice", {})
    demo_phases = demo.get("phases", {})
    practice_phases = practice.get("phases", {})
    clock = demo_phases.get("M1_clock", {})
    trials_block = practice_phases.get("M2_M3_trials", {})
    class_summaries = trials_block.get("classes", [])
    boundaries = practice_phases.get("M4_boundaries", [])
    carry = practice_phases.get("M4_carry", {})
    samples = demo_phases.get("M5_samples", {})
    legacy = demo_phases.get("M6_legacy", {})
    practice_clock = practice.get("clockAfterTrials", {})

    by_class = {item["class"]: item for item in class_summaries}
    c1 = by_class.get("C1_lateral_normal", {})
    c2 = by_class.get("C2_vertical_normal", {})
    c3 = by_class.get("C3_lateral_focus", {})
    c4 = by_class.get("C4_diagonal_normal", {})

    ticket_ok = bool(clock.get("ticketGaps", {}).get("allOne")) and bool(
        practice_clock.get("ticketGaps", {}).get("allOne")
    )
    logic_ok = bool(clock.get("logicGaps", {}).get("allOne")) and bool(
        practice_clock.get("logicGaps", {}).get("allOne")
    )
    rate = clock.get("presentHz")
    rate_ok = rate is not None and rate >= 55.0

    lag_in_strict = bool(c1.get("pass_lag_in")) and bool(c2.get("pass_lag_in"))
    lag_in_flex = bool(c3.get("pass_lag_in")) and bool(c4.get("pass_lag_in"))
    lag_mv_all = all(
        item.get("pass_lag_mv")
        for item in (c1, c2, c3, c4)
        if item.get("trials")
    ) and all(item.get("trials") for item in (c1, c2, c3, c4))
    delta_all = all(
        item.get("pass_delta") for item in (c1, c2, c3, c4) if item.get("trials")
    ) and all(item.get("trials") for item in (c1, c2, c3, c4))

    boundary_names = {item.get("name") for item in boundaries if item.get("saturated")}
    boundary_match = all(item.get("match") for item in boundaries) and len(boundaries) == 4

    reversal_trials = [
        trial
        for block in carry.get("reversals", [])
        for trial in block.get("trials", [])
        if trial.get("carry")
    ]
    reversal_ok_frames = [t["carry"]["oldCarryFrames"] for t in reversal_trials]
    reversal_ok_px = [t["carry"]["oldCarryPx"] for t in reversal_trials]
    reversal_pass = bool(reversal_trials) and all(
        frames == 1 and abs(px) == 4.0
        for frames, px in zip(reversal_ok_frames, reversal_ok_px)
    )
    reversal_blocks_ok = sum(
        1
        for block in carry.get("reversals", [])
        for trial in block.get("trials", [])
        if trial.get("carry")
        and trial["carry"]["oldCarryFrames"] == 1
        and abs(trial["carry"]["oldCarryPx"]) == 4.0
    )

    sample_checks = samples.get("checkCounts", {})
    samples_all_pass = (
        samples.get("accepted", 0) >= 60
        and samples.get("allChecksPassed", 0) == samples.get("accepted", -1)
        and sample_checks.get("pixels_match_backbuffer", {}).get("passed", 0)
        == sample_checks.get("pixels_match_backbuffer", {}).get("total", -1)
    )
    legacy_measured = legacy.get("reads", 0) > 0

    gates = {
        "G1_ticket_step": {
            "passed": ticket_ok,
            "detail": {
                "demo": clock.get("ticketGaps"),
                "practice": practice_clock.get("ticketGaps"),
            },
        },
        "G2_logic_step": {
            "passed": logic_ok,
            "detail": {
                "demo": clock.get("logicGaps"),
                "practice": practice_clock.get("logicGaps"),
            },
        },
        "G3_present_rate": {"passed": rate_ok, "detail": {"presentHz": rate}},
        "G4_lag_in_strict": {
            "passed": lag_in_strict,
            "detail": {
                "C1": {"match": c1.get("lagInMatch"), "of": c1.get("trials")},
                "C2": {"match": c2.get("lagInMatch"), "of": c2.get("trials")},
            },
        },
        "G5_lag_in_flex": {
            "passed": lag_in_flex,
            "detail": {
                "C3": {"match": c3.get("lagInMatch"), "of": c3.get("trials")},
                "C4": {"match": c4.get("lagInMatch"), "of": c4.get("trials")},
            },
        },
        "G6_motion_lag": {
            "passed": lag_mv_all,
            "detail": {
                name: {"ok": item.get("lagMvOk"), "of": item.get("trials")}
                for name, item in by_class.items()
            },
        },
        "G7_dynamics": {
            "passed": delta_all,
            "detail": {
                name: {
                    "deltaMatch": item.get("deltaMatch"),
                    "of": item.get("trials"),
                    "expected": item.get("expectedAxisDelta"),
                }
                for name, item in by_class.items()
            },
        },
        "G8_boundaries": {
            "passed": boundary_match,
            "detail": {
                item.get("name"): {
                    "boundary": item.get("boundary"),
                    "expected": item.get("expected"),
                    "saturated": item.get("saturated"),
                    "match": item.get("match"),
                }
                for item in boundaries
            },
            "saturatedDirections": sorted(boundary_names),
        },
        "G9_reversal_carry": {
            "passed": reversal_pass,
            "detail": {
                "trials": len(reversal_trials),
                "exactlyOneFrameOneStep": reversal_blocks_ok,
                "carryFrames": reversal_ok_frames,
                "carryPx": reversal_ok_px,
            },
        },
        "G10_pixel_state": {
            "passed": samples_all_pass,
            "detail": {
                "accepted": samples.get("accepted"),
                "discarded": samples.get("discarded"),
                "allChecksPassed": samples.get("allChecksPassed"),
                "pixelsMatchBackbuffer": sample_checks.get("pixels_match_backbuffer"),
                "checkCounts": sample_checks,
            },
        },
        "G11_legacy_still_unknown": {
            "passed": None,
            "detail": {
                "reads": legacy.get("reads"),
                "ticketChanged": legacy.get("ticketChanged"),
                "bulletDigestMismatch": legacy.get("bulletDigestMismatch"),
                "enemyDigestMismatch": legacy.get("enemyDigestMismatch"),
                "measured": legacy_measured,
            },
        },
    }
    gate_names = [name for name in gates if not name.startswith("G11")]
    unfreeze = all(bool(gates[name]["passed"]) for name in gate_names)
    return {
        "gates": gates,
        "p0UnfreezePassed": unfreeze,
        "interruptions": {
            "count": len(practice.get("interruptions", [])),
            "log": practice.get("interruptions", []),
        },
        "decisions": {
            "exactLabelsAllowed": unfreeze,
            "scope": "fence build only; canonical image stays alignment_unknown",
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("p0-fence-round26")
    parser.add_argument(
        "--out-dir",
        default="Document/Update/Alpha/v3.7.3/evidence/p0-fence-round26",
    )
    parser.add_argument("--session", choices=["both", "demo", "practice"], default="both")
    parser.add_argument("--frames", type=int, default=240, help="M1 trace frames")
    parser.add_argument("--samples", type=int, default=60, help="M5 samples")
    parser.add_argument("--legacy", type=int, default=60, help="M6 legacy reads")
    parser.add_argument("--per-class", type=int, default=20, help="trials per class")
    parser.add_argument("--carry-repeats", type=int, default=5)
    parser.add_argument(
        "--max-hits",
        type=int,
        default=24,
        help="player hits tolerated before the practice session is given up; every "
        "interrupted trial is discarded, retried and reported",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=5,
        help="practice games to use before reporting the remaining measurements as "
        "missing (stage 1 Lunatic ends a nearly-static player's run in ~10 s)",
    )
    parser.add_argument("--session-pause", type=float, default=2.0)
    parser.add_argument(
        "--session-timeout",
        type=float,
        default=240.0,
        help="wall-clock budget for one practice game before it is closed and the "
        "queue continues in a new one",
    )
    parser.add_argument("--demo-warmup", type=float, default=8.0)
    args = parser.parse_args(argv)

    os.environ["TH08_ADDRESS_FILE"] = str(FENCE_ADDRESSES)
    for entry in (str(ROOT), str(ROOT / "tools" / "rl")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    evidence = Evidence(Path(args.out_dir))
    # A crash in one session must not destroy the evidence of the other one:
    # every failure is recorded, the summary is always written.
    for name, runner in (
        ("demo", run_demo_session),
        ("practice", run_practice_session),
    ):
        if args.session not in ("both", name):
            continue
        try:
            evidence.data["sessions"][name] = runner(evidence, args)
        except Exception as exc:  # pragma: no cover - live-machine diagnostics
            evidence.data["sessions"][name] = {
                "mode": name,
                "status": "crashed",
                "error": f"{type(exc).__name__}: {exc}",
                "phases": {},
            }

    evidence.data["verdict"] = build_verdict(evidence.data)
    evidence.data["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = evidence.write_json("summary.json", evidence.data)
    print(json.dumps(evidence.data["verdict"], indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
