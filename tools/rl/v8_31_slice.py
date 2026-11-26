#!/usr/bin/env python3
"""V831-SLICE-01: the two real-machine trajectories of the v8.31.0 short slice.

This is a **new instrument**, not a change to anything frozen.  It reuses, by
import only, the harness client (``touhou_rl.v400.gym.protocol``), the shared
block reader (``...gym.shm``), the action table (``...gym.actions``), the SHOOT
pulse rule (``touhou_rl.v400.teacher.shoot``), the frozen observation contract
(``...student.obs_spec_v3`` / ``obs_spec_v2``), the ``STATE`` decoder
(``...labels.state``), the difficulty table (``...rl.env``) and 閳?for the
teacher schedule only 閳?the frozen tape loader
(``touhou_rl.v420.rl.load_clear_schedule``).  No existing file, protocol, ABI,
dependency or third-party source is touched.

Two subcommands, one job each
------------------------------
``pixel-smoke``
    One natural-start trajectory with a **constant** candidate (``--constant-cand
    12``), 2 game frames per decision, 600 controlled frames.  Its acceptance is
    the *wiring*: real RGBA readback reaches the actor input, the frame order and
    the per-decision action segments are recorded, and the action history only
    grows after an action actually took effect.

``teacher-calibration``
    The same loop and the same pulse, driven by the one recorded teacher tape
    that really cleared Stage 1.  Its acceptance is a **real** stage clear:
    ``EVENTS``'s ``stageClear`` / ``stageClearEdgeFrame`` and ``STATE.stage``'s
    first advance are both recorded, with their order and interval.

Both runs share the same preflight: engine sha256, obs-v3 digest, private game
directory, muted ``th08.cfg`` copy, private Xvfb, and ``SDL_AUDIODRIVER=dummy``.

What is deliberately *not* here
-------------------------------
* No training, no checkpoint, no selection, no prefix re-recording, no WP05
  resume, no git operation, no upload.  No existing evidence file is written.
* No audio: the game data directory is a **private copy** (so the read-only
  ``--data-source`` is never written) and only the copy's ``th08.cfg`` has
  ``0x1F``/``0x20``/``0x27``/``0x28`` zeroed.  Both the source and the copy are
  re-decoded and written to ``<out>/mute.json`` before the first start and again
  after the process exits.
* No pixel dumps: ``decisions.jsonl`` carries digests and the local FNV-1a-64 of
  the captured RGBA (``pixels_checksum_local``), never an image.

The one input boundary that matters
-----------------------------------
:func:`build_actor_inputs` takes **exactly two** arguments 閳?the RGBA readback
and the action-history buffer.  Referee fields (lives, bombs, power, stage,
deaths, frame-event flags, stage-clear) are not parameters, not globals it reads
and not keys it returns; :data:`ACTOR_FIELDS` is the whitelist and
:data:`REFEREE_FIELDS` is the denial list, both asserted by the offline tests and
re-checked by :func:`actor_boundary_problems` on every run.

Known limitations (recorded, not hidden)
----------------------------------------
* ``EVENTS`` reports the frame-event word of the **last** frame a ``STEP``
  advanced.  A multi-frame ``STEP`` therefore has one observation for its last
  frame only; this tool never invents a value for the frames it did not see.
  Both runs use one-frame ``STEP`` units (``obs_window="consecutive"``), so in
  practice every advanced frame is observed.
* The shared block's ``frame`` field is **overloaded** and is never the pixel
  frame after a ``PublishState``.  ``PublishPixels`` stamps it with the frame it
  just published, and ``AfterDrawFrame`` runs *before* ``++g_frameCount``
  (``third_party/th08-rl/src/main.cpp:143-146``), so the RGBA bytes always
  belong to frame ``EVENTS.frames - 1``.  ``PublishState`` instead stamps the
  *current* ``FrameCount()`` (``rlharness.cpp:1362``), and on ``rl_v3`` the
  ``STEP`` command itself ends with that publish (``rlharness.cpp:901``), so a
  read taken after a step sees either ``EVENTS.frames`` (the state publish, the
  normal path) or ``EVENTS.frames - 1`` (a ``STEP`` that returned before its
  publish, e.g. the exit path).  Both are legal, nothing else is; the recorded
  ``pixel_frame`` is the frame of the captured bytes (``EVENTS.frames - 1``)
  and the raw header value is kept beside it.  A mismatch is re-read exactly
  once -- before any ``STATE``/``PIXELS`` is sent -- and then reported as a
  technical conflict (never silently accepted).  ``STATE``/``PIXELS`` can only
  make the header less trustworthy, which is why the shared block is read
  *first* (:data:`AUDIT_COMMAND_SEQUENCE`).
* A decision whose remaining budget is smaller than ``--decision-frames`` is
  **not** issued by default: a decision's mask sequence is defined for the full
  width, and the ticket's truncation rule is "never exceed the controlled-frame
  cap".  The run ends as ``truncated`` with the unspent frames recorded.
  ``--allow-partial-tail`` restores the frozen v4.2.0 behaviour (the remaining
  ``< --decision-frames`` units go out as one partial decision).  Neither
  acceptance run can reach that tail: the 600-frame budget is 300 whole
  decisions and the teacher run stops at its stage clear (~13,195 frames), far
  short of its 19,887-frame cap.
* The teacher run stops as soon as ``STATE.stage`` advances, which is what makes
  ``frames`` the in-stage frame count and what the frozen evidence compares.
  Signals that only ``EVENTS`` carries are recorded up to that stop; if the
  second signal is missing at the stop, ``missing_second_signal`` is set and the
  audited window around the stop is kept verbatim.
* ``time.monotonic()`` wall-clock buckets are machine load dependent; they are
  reported as measured and never used as a gate.

Audit volume and cost attribution (V831-SLICE-01b / 01c)
--------------------------------------------------------
The first teacher run was stopped by its 600 s deadline at frame 9,582 of 13,195.
The cause was *not* the protocol traffic: a ``STEP`` unit cost 7.4 ms of engine
time and 1.5 ms of ``EVENTS``/``STATE``/``HUD``/``PIXELS``, while ~60 ms per unit
went into the **local** digest of the captured RGBA (a pure-Python FNV-1a-64 over
1.2 MB) and was measured by no bucket at all -- 83% ``unaccounted``.

* ``capture`` now measures that local digest and ``unit_total`` wraps every unit,
  so ``cost.unaccounted_share`` is the only place a hidden cost can still hide.
* The per-unit audit is ``step -> shm -> events -> state``.  ``HUD`` and
  ``PIXELS`` are read-only tails that fire on the first unit, once per SHOOT
  pulse period (``HUD`` only) and once after the last unit
  (:data:`AUDIT_COMMAND_GATED`); the resource columns come from ``STATE``
  (``hud_source="state_derived"``).
* **Digest scopes (01c).**  The ticket records the raw-picture checksum once per
  *decision*, so ``pixels_checksum_local`` (the engine-comparable FNV-1a-64,
  raw bottom-up) is now computed on the decision's **last** unit -- 6,598
  computations instead of 13,195, and the recorded value is exactly the one
  ``decisions.jsonl`` always carried.  Every unit still gets a recomputable
  per-frame identity, ``pixel_frame_blake2b128`` (blake2b-128, measured 1.10 ms
  per 1,228,800 B against 54.02 ms for the FNV).  The scopes are recorded in
  ``result.audit`` as ``checksum_scope`` / ``frame_digest_scope`` with the
  ticket quote in ``checksum_scope_rationale``.
* ``--audit-stride N`` (default 1, i.e. off) is the explicit, recorded escape
  hatch: it thins ``EVENTS``/``STATE`` but never touches ``STEP``, the mask
  trace, the action sequence, the observation digest or the per-frame blake2b.
* ``--timeout-s`` keeps its defaults (120 / 600).  Any positive number is
  accepted and recorded verbatim (``result.timeout_s``), so an acceptance
  script can pass a larger budget and report the measured wall clock separately.

Run (inside WSL; ``xvfb-run`` is applied automatically unless ``--no-xvfb``)::

    python tools/rl/v8_31_slice.py pixel-smoke \\
      --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \\
      --data-source /root/v400-data/game --out .tmp_v831_slice_01_pixel \\
      --stage 0 --difficulty normal --shot-type 0 --seed 1000 --constant-cand 12 \\
      --decision-frames 2 --shoot-pulse-period 48 --max-controlled-frames 600 \\
      --timeout-s 120

Exit codes: ``0`` the subcommand met its own acceptance rule (pixel-smoke: the
full controlled-frame budget, a legal first death, or a real clear; teacher: a
real stage clear), ``1`` the run happened but did not meet it (truncated short of
the budget, game over, engine exit, technical conflict, timeout), ``2`` the
invocation was refused before any engine was started.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import struct
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from touhou_rl.spec import BTN_BOMB, BTN_SHOOT  # noqa: E402
from touhou_rl.v400.gym import actions as gym_actions  # noqa: E402
from touhou_rl.v400.gym import protocol  # noqa: E402
from touhou_rl.v400.gym.shm import (  # noqa: E402
    FLAG_PIXELS_VALID,
    PIXELS_OFFSET,
    SharedBlock,
)
from touhou_rl.v400.labels.state import decode_state, state_from_reply  # noqa: E402
from touhou_rl.v400.rl.env import DIFFICULTIES  # noqa: E402
from touhou_rl.v400.student import obs_spec_v3  # noqa: E402
from touhou_rl.v400.student.obs_spec_v2 import ActionHistoryBuffer  # noqa: E402
from touhou_rl.v400.teacher.shoot import commit_frames, release_due  # noqa: E402

#: This file, as written into every artefact it produces.
CONTROLLER = "tools/rl/v8_31_slice.py"
#: Schema tags.  A reader can tell a v8.31.0 slice run from any earlier one.
RESULT_SCHEMA = "v831-slice-result/1"
DECISION_SCHEMA = "v831-slice-decision/1"
MUTE_SCHEMA = "v831-slice-mute/1"

#: Registered identities (v8.30 reply / v4.3.1 R14 / ASSET_SYNC.md).  Each one is
#: re-computed at run time and compared; nothing is trusted from memory.
REGISTERED_ENGINE_SHA256 = (
    "691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6")
REGISTERED_SOURCE_CFG_SHA256 = (
    "48bf7621da5879ef5ec1392fcac2cc0a85c313912a24fb33b4cddccf0d28f07a")
EXPECTED_MUTED_CFG_SHA256 = (
    "cfa292ae10ee1907df7c637d013386851486dadf7a97bb5adf12bcf21dd14638")
REGISTERED_OBS_V3_DIGEST = (
    "ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105")
REGISTERED_TAPE_SHA256 = (
    "f759de7dacc78343dec8825da5de14942e749b8a5a2c5319304e513d1618b7b9")

#: ``th08.cfg`` is a fixed 60-byte ``GameConfiguration`` struct, not text
#: (``Supervisor.hpp`` layout; see ``tools/rl/v400_make_cfg.py``).  These four
#: bytes are the whole audio switch.
CFG_SIZE = 0x3C
MUTE_OFFSETS: tuple[int, ...] = (0x1F, 0x20, 0x27, 0x28)
MUTE_FIELD_OFFSETS: tuple[tuple[str, int], ...] = (
    ("musicMode", 0x1F),
    ("playSounds", 0x20),
    ("musicVolume", 0x27),
    ("sfxVolume", 0x28),
)

#: ``ReplayManager::frameEventFlags`` ``STAGE_CLEAR`` level bit.  It is a
#: *level*, not an edge: the edge is ``stageClearEdge``/``stageClearEdgeFrame``.
EVENT_FLAG_STAGE_CLEAR = 0x0080
#: Fields every audited ``EVENTS`` reply must carry.  A reply that says ``OK``
#: but lacks one is a protocol conflict, never a zero.
EVENTS_REQUIRED_FIELDS: tuple[str, ...] = (
    "frames",
    "frameEventFlags",
    "stageClear",
    "stageClearEdge",
    "stageClearEdgeFrame",
)

#: One decision is 2 game frames (``obs_spec_v3.DECISION_FRAMES``).
DECISION_FRAMES = int(obs_spec_v3.DECISION_FRAMES)
#: 48 decisions x 2 frames = 96 frames between SHOOT release frames.
SHOOT_PULSE_PERIOD = 48
#: The pixels the engine publishes: 640x480 RGBA8 at ``PIXELS_OFFSET``.
PIXEL_WIDTH = 640
PIXEL_HEIGHT = 480
SHM_EXPECTED_BYTES = PIXELS_OFFSET + PIXEL_WIDTH * PIXEL_HEIGHT * 4

#: The frozen v4.3.1 ``teacher-calibration`` expectation (T08
#: ``EVAL_SHOOT_PULSE.md`` 鎼? and ``.tmp_v420_evalpulse/positive_control.json``,
#: sha256 ``9cdd9033閳ヮ泦`: same tape, same pulse, same loader).  It is **not** a
#: gate: the only exit-0 condition is a real stage clear.  The ``match`` block is
#: reported item by item so an equivalence claim is a measured one, and the two
#: digest items are reported as measured even when they differ (the observation
#: digest depends on push timing; the mask digest is a pure action sequence).
EXPECTED_TEACHER_CALIBRATION: dict[str, Any] = {
    "stage_clear": True,
    "frames": 13195,
    "score": 2631430,
    "decisions": 6598,
    "state_frame": 13195,
    "gameplay_frame": 13196,
    "lives_end": 2.0,
    "deaths": 0.0,
    "renders": 13195,
    "pixels_warmup_frames": 0,
    "shoot": {
        "shoot_frames": 13058,
        "no_shoot_frames": 137,
        "releases_planned": 137,
        "releases_applied": 137,
        "shoot_rising_edges": 138,
        "distinct_masks": 30,
        "mask_trace_sha256": (
            "66c81344b00f8766f7d9e860d1874b1141c699122f491ed1a8ed0aea9f179924"),
    },
    "action_sequence_sha256": (
        "5e9d66c4f5685da9f0ea56f3105656b7210721124f8b74a12813b49089995cd3"),
    "observed_input_sha256": (
        "315a2ab6552edcba23a40f49069150e52eac67f034407e9480f48a66f47c6212"),
    "harness_role": "env",
    "obs_window": "consecutive",
}

#: How many audited frame rows around the stop are kept when a second clear
#: signal is missing (a "small window", never the full 13k-row log).
SIGNAL_WINDOW_FRAMES = 32

#: Terminal kinds.  ``terminal_reason`` carries the fine-grained detail; the kind
#: is the six-way classification the ticket asks for.
TERMINAL_STAGE_CLEAR = "stage_clear"
TERMINAL_LEGAL_DEATH = "legal_death"
TERMINAL_TRUNCATED = "truncated"
TERMINAL_ENVIRONMENT_EXIT = "environment_exit"
TERMINAL_TECHNICAL_INVALID = "technical_invalid"
TERMINAL_UNKNOWN = "unknown"
TERMINAL_KINDS: tuple[str, ...] = (
    TERMINAL_STAGE_CLEAR,
    TERMINAL_LEGAL_DEATH,
    TERMINAL_TRUNCATED,
    TERMINAL_ENVIRONMENT_EXIT,
    TERMINAL_TECHNICAL_INVALID,
    TERMINAL_UNKNOWN,
)

#: Terminal reasons, one constant per branch.  ``_accepted`` reads these, so a
#: "legal death" can never be spelled two different ways and quietly fail.
REASON_FIRST_DEATH = "first_death"
REASON_GAME_OVER = "game_over"
REASON_GAME_OVER_NO_LIVES = "game_over_no_lives"
REASON_FRAME_CAP = "frame_cap"
REASON_TIMEOUT = "timeout"
REASON_ENGINE_EXIT = "engine_exit"
REASON_SHORT_ADVANCE = "short_advance"
REASON_STATE_STAGE_ADVANCED = "state_stage_advanced"
REASON_EVENTS_CLEAR_ONLY = "events_clear_without_state_advance"
REASON_BUDGET_EXHAUSTED = "budget_exhausted"

#: The reasons that count as a *legal* death for the pixel-wiring acceptance:
#: a death with lives left is the documented 0; a game over is not.
LEGAL_DEATH_ACCEPTED_REASONS: tuple[str, ...] = (REASON_FIRST_DEATH,)

#: The actor input boundary, as data: exactly these two arguments may reach
#: :func:`build_actor_inputs`.  Nothing else is a parameter or a return key.
ACTOR_FIELDS: tuple[str, str] = ("pixels_rgba", "history")
#: Referee-only fields.  They may be *reported* by this tool and may sit in a
#: side-channel record, but they must never enter the actor input path.
REFEREE_FIELDS: tuple[str, ...] = (
    "lives",
    "lives_remaining",
    "bombs",
    "bombs_remaining",
    "power",
    "player_power",
    "stage",
    "deaths",
    "score",
    "frame_event_flags",
    "stage_clear",
    "stage_clear_edge",
    "stage_clear_edge_frame",
    "game_over",
    "frameEventFlags",
    "stageClear",
)


class SliceError(RuntimeError):
    """Base class for every refusal and conflict this tool can report."""


class Refusal(SliceError):
    """The invocation cannot run at all (bad input, moved identity).  Exit 2."""


class TechnicalInvalid(SliceError):
    """A protocol conflict at a known position.  Exit 1, never mapped to 0."""

    def __init__(self, reason: str, *, decision_index: int | None = None,
                 frame: int | None = None, mask: int | None = None) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.position: dict[str, Any] = {
            "decision_index": decision_index,
            "frame": frame,
            "mask": mask,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    """Local wall clock with its timezone, the project's report format."""

    return datetime.now().astimezone().isoformat()


def sha256_bytes(data: bytes) -> str:
    """sha256 of a byte string."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """sha256 of a file's bytes, streamed so a 474 MB archive is cheap."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    """sha256 of one ndarray's bytes (C order, dtype included by the bytes)."""

    return sha256_bytes(np.ascontiguousarray(np.asarray(array)).tobytes())


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON atomically (``<path>.tmp`` then replace)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(target)
    return target


def write_jsonl_row(handle: Any, row: Mapping[str, Any]) -> None:
    """Append one compact JSON row; the caller owns the handle's lifetime."""

    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_kv(reply: str) -> dict[str, str]:
    """``OK a=1 b=2`` -> ``{"_verb": "OK", "a": "1", "b": "2"}``.

    Same wire shape as ``touhou_rl.v400.gym.protocol._parse``; spelled out here so
    this tool does not depend on a private name of the frozen client.
    """

    parts = reply.split()
    out: dict[str, str] = {"_verb": parts[0] if parts else ""}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


class CostMeter:
    """Named ``time.monotonic()`` accumulators, one per reported bucket.

    **Leaf buckets are mutually exclusive** -- no leaf is ever measured inside
    another leaf -- so ``accounted_seconds`` is their plain sum and nothing is
    counted twice.  ``unit_total`` is the **one parent bucket**: it wraps the
    whole body of each ``STEP`` unit and therefore *contains* that unit's
    ``step``, ``audit``, ``capture`` and ``actor`` time.  It is reported under
    ``parent_seconds`` and is deliberately **excluded** from
    ``accounted_seconds``; ``accounted_inside_parent_seconds`` says how much leaf
    time was measured inside it, and ``unit_internal_seconds`` is the remainder
    (row building, digest bookkeeping, the shoot counters) that no leaf covers.

    Buckets: ``startup`` (process + HELLO + RESET), ``resume`` (always 0 in this
    ticket: nothing is resumed), ``readback`` (``READBACK``/``SHM``/block map),
    ``actor`` (encode + digest), ``step`` (``STEP``), ``audit`` (``EVENTS`` /
    ``STATE`` / ``HUD`` / ``PIXELS`` / shared-block reads), ``io`` (files),
    ``capture`` (the *local* digest of the captured RGBA: FNV-1a-64, sha256,
    non-zero count, the reversed raw-order copy), ``session_close``
    (``EngineHost.close``), ``copytree`` (the private game directory copy) and
    ``preflight`` (engine/obs/cfg identities, disk, git).

    ``unaccounted`` is ``total - accounted``, reported per unit and as a share,
    never hidden.  V831-SLICE-01's teacher run had 83% unaccounted because the
    per-unit capture digest was measured by nothing; ``capture`` and
    ``unit_total`` are the two buckets that close that hole.
    """

    #: Mutually exclusive leaves: their plain sum is ``accounted_seconds``.
    LEAF_BUCKETS: tuple[str, ...] = (
        "startup", "resume", "readback", "actor", "step", "audit", "io",
        "capture", "session_close", "copytree", "preflight")
    #: Parent buckets: they contain leaves and are never added to ``accounted``.
    PARENT_BUCKETS: tuple[str, ...] = ("unit_total",)
    #: Everything that is reported under ``payload()["seconds"]``.
    BUCKETS: tuple[str, ...] = LEAF_BUCKETS + PARENT_BUCKETS

    NESTED_MEASURE_NOTE = (
        "parent bucket 'unit_total' wraps every STEP unit and CONTAINS its "
        "'step', 'audit', 'capture' and in-unit 'actor' leaves; the leaf buckets "
        "are mutually exclusive and accounted_seconds sums ONLY them, so the "
        "parent and its children are never added together. "
        "accounted_inside_parent_seconds is the leaf time measured inside the "
        "parent, unit_internal_seconds is the parent minus that (row building, "
        "digests, shoot counters), and unaccounted_seconds is "
        "total_seconds - accounted_seconds")

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {name: 0.0 for name in self.BUCKETS}
        self._counts: dict[str, int] = {name: 0 for name in self.BUCKETS}
        self._started: float | None = None
        self._finished: float | None = None
        self._parent_depth = 0
        self._child_seconds = 0.0
        self._child_calls = 0

    def begin(self) -> None:
        if self._started is None:
            self._started = time.monotonic()

    def finish(self) -> None:
        if self._finished is None:
            self._finished = time.monotonic()

    @contextmanager
    def measure(self, bucket: str) -> Iterator[None]:
        if bucket not in self._seconds:
            raise KeyError(f"unknown cost bucket {bucket!r}")
        parent = bucket in self.PARENT_BUCKETS
        started = time.monotonic()
        if parent:
            self._parent_depth += 1
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            self._seconds[bucket] += elapsed
            self._counts[bucket] += 1
            if self._parent_depth > 0 and not parent:
                # Leaf time that ran inside a parent: reported separately so the
                # parent's own total can be split without double counting.
                self._child_seconds += elapsed
                self._child_calls += 1
            if parent:
                self._parent_depth -= 1

    def add(self, bucket: str, seconds: float) -> None:
        if bucket not in self._seconds:
            raise KeyError(f"unknown cost bucket {bucket!r}")
        elapsed = float(seconds)
        self._seconds[bucket] += elapsed
        self._counts[bucket] += 1
        if self._parent_depth > 0 and bucket not in self.PARENT_BUCKETS:
            self._child_seconds += elapsed
            self._child_calls += 1

    def payload(self, *, step_units: int | None = None) -> dict[str, Any]:
        total = None
        if self._started is not None:
            end = self._finished if self._finished is not None else time.monotonic()
            total = end - self._started
        accounted = sum(self._seconds[name] for name in self.LEAF_BUCKETS)
        parent = sum(self._seconds[name] for name in self.PARENT_BUCKETS)
        units = step_units if step_units else self._counts.get("unit_total", 0)
        units = max(1, int(units))
        unaccounted = None if total is None else total - accounted
        return {
            "seconds": {name: round(value, 6)
                        for name, value in self._seconds.items()},
            "calls": dict(self._counts),
            "leaf_buckets": list(self.LEAF_BUCKETS),
            "parent_buckets": list(self.PARENT_BUCKETS),
            "accounted_seconds": round(accounted, 6),
            "parent_seconds": {name: round(self._seconds[name], 6)
                               for name in self.PARENT_BUCKETS},
            "parent_calls": {name: int(self._counts[name])
                             for name in self.PARENT_BUCKETS},
            "accounted_inside_parent_seconds": round(self._child_seconds, 6),
            "accounted_inside_parent_calls": int(self._child_calls),
            "unit_internal_seconds": round(parent - self._child_seconds, 6),
            "unit_internal_by_unit": round(
                (parent - self._child_seconds) / units, 6),
            "total_seconds": None if total is None else round(total, 6),
            "unaccounted_seconds": (None if unaccounted is None
                                    else round(unaccounted, 6)),
            "unaccounted_by_unit": (None if unaccounted is None
                                    else round(unaccounted / units, 6)),
            "unaccounted_share": (None if unaccounted is None or not total
                                  else round(unaccounted / total, 6)),
            "unaccounted_by_unit_rule": (
                "unaccounted_seconds / max(1, step_units)"),
            "nested_measure_note": self.NESTED_MEASURE_NOTE,
            "bucket_definitions": {
                "startup": "engine process start, HELLO and RESET",
                "resume": "always 0 in this ticket: nothing is resumed",
                "readback": "READBACK, SHM attach and the shared-block map",
                "actor": "encode_planes_v3 + ObservationBufferV3 + digest",
                "step": "every STEP command",
                "audit": "EVENTS / STATE / HUD / PIXELS / shared-block reads",
                "io": "JSON/JSONL writes",
                "capture": ("the local digest of one captured RGBA block: "
                            "FNV-1a-64, sha256, non-zero count and the reversed "
                            "raw-order copy (never a command)"),
                "session_close": "EngineHost.close: shm unmap, client close, /dev/shm cleanup",
                "copytree": "shutil.copytree of --data-source into <out>/game",
                "preflight": "engine sha256, obs-v3 digest, cfg decode, disk, git head",
                "unit_total": ("PARENT: the whole body of one STEP unit, from "
                               "before host.step to the end of that unit; it "
                               "contains step/audit/capture/in-unit actor"),
            },
        }


# ---------------------------------------------------------------------------
# th08.cfg: decode, mute, and the byte-level evidence
# ---------------------------------------------------------------------------
def decode_cfg_mute_fields(raw: bytes, *, path: str = "") -> dict[str, Any]:
    """Decode the four audio fields of one 60-byte ``th08.cfg``.

    Same semantics as ``tools/rl/v431_audio_cfg_audit.py``'s ``decode``: a size
    other than 60 is an error (never a legal config), ``musicMode`` and
    ``playSounds`` are the raw bytes at ``0x1F``/``0x20``, and the two volumes are
    little-endian ``i8`` at ``0x27``/``0x28``.  Two verdicts are recorded because
    two definitions exist in the tree:

    ``muted_legacy_three_field``
        ``playSounds == 0 and musicVolume <= 0 and sfxVolume <= 0`` 閳?the v4.3.1
        audit's rule, which does not look at ``musicMode``.
    ``muted_strict_four_field``
        all four exactly zero 閳?the rule this ticket requires.
    """

    out: dict[str, Any] = {
        "path": str(path),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "size_ok": len(raw) == CFG_SIZE,
        "expected_bytes": CFG_SIZE,
        "fields": None,
        "muted_strict_four_field": False,
        "muted_legacy_three_field": False,
        "error": None,
    }
    if len(raw) != CFG_SIZE:
        out["error"] = f"size {len(raw)} != {CFG_SIZE}"
        return out
    fields = {
        "musicMode": raw[0x1F],
        "playSounds": raw[0x20],
        "musicVolume": struct.unpack_from("<b", raw, 0x27)[0],
        "sfxVolume": struct.unpack_from("<b", raw, 0x28)[0],
    }
    out["fields"] = fields
    out["field_offsets"] = {name: offset for name, offset in MUTE_FIELD_OFFSETS}
    out["muted_strict_four_field"] = bool(
        fields["musicMode"] == 0 and fields["playSounds"] == 0
        and fields["musicVolume"] == 0 and fields["sfxVolume"] == 0)
    out["muted_legacy_three_field"] = bool(
        fields["playSounds"] == 0 and fields["musicVolume"] <= 0
        and fields["sfxVolume"] <= 0)
    return out


def decode_cfg_file(path: str | Path) -> dict[str, Any]:
    """Decode the four audio fields of a ``th08.cfg`` on disk (any path)."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        return {
            "path": str(source), "bytes": None, "sha256": None, "size_ok": False,
            "expected_bytes": CFG_SIZE, "fields": None,
            "muted_strict_four_field": False, "muted_legacy_three_field": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return decode_cfg_mute_fields(raw, path=str(source))


def zero_mute_offsets(raw: bytes) -> bytes:
    """Return the same 60-byte config with ``0x1F/0x20/0x27/0x28`` set to zero.

    Nothing else is touched: the caller's diff proves it byte by byte.
    """

    if len(raw) != CFG_SIZE:
        raise Refusal(f"th08.cfg must be {CFG_SIZE} bytes, got {len(raw)}")
    buffer = bytearray(raw)
    for offset in MUTE_OFFSETS:
        buffer[offset] = 0
    return bytes(buffer)


def cfg_copy_diff(source_raw: bytes, copy_raw: bytes) -> dict[str, Any]:
    """Byte-level difference between the source cfg and its private copy."""

    diff: dict[str, Any] = {
        "comparable": len(source_raw) == len(copy_raw),
        "source_bytes": len(source_raw),
        "copy_bytes": len(copy_raw),
        "diff_byte_offsets": [],
        "diff_bytes": [],
        "expected_diff_byte_offsets": list(MUTE_OFFSETS),
        "diff_offsets_match_expected": False,
        "other_bytes_identical": False,
        "other_byte_count": None,
        "unexpected_offsets": [],
    }
    if not diff["comparable"]:
        return diff
    offsets = [index for index in range(len(source_raw))
               if source_raw[index] != copy_raw[index]]
    other = [index for index in offsets if index not in MUTE_OFFSETS]
    diff["diff_byte_offsets"] = offsets
    diff["diff_bytes"] = [
        {"offset": index, "source": source_raw[index], "copy": copy_raw[index]}
        for index in offsets
    ]
    diff["unexpected_offsets"] = other
    diff["diff_offsets_match_expected"] = offsets == list(MUTE_OFFSETS)
    diff["other_bytes_identical"] = not other
    diff["other_byte_count"] = len(source_raw) - len(offsets)
    return diff


def build_mute_report(
    *,
    source_path: Path,
    copy_path: Path,
    source_raw: bytes,
    copy_before_raw: bytes,
    copy_after_raw: bytes | None,
    engine_env: Mapping[str, Any],
    checked_at: Sequence[str],
    problems: Sequence[str],
) -> dict[str, Any]:
    """Assemble ``mute.json``: identity, per-byte diff and both decodes.

    ``copy_after_raw`` is ``None`` until the engine process has exited; the file
    is rewritten after the exit so the same document carries both checks.
    """

    source = decode_cfg_mute_fields(source_raw, path=str(source_path))
    source["matches_registered_source_sha256"] = bool(
        source["sha256"] == REGISTERED_SOURCE_CFG_SHA256)
    source["registered_source_sha256"] = REGISTERED_SOURCE_CFG_SHA256
    before = decode_cfg_mute_fields(copy_before_raw, path=str(copy_path))
    before.update(cfg_copy_diff(source_raw, copy_before_raw))
    before["matches_expected_muted_sha256"] = bool(
        before["sha256"] == EXPECTED_MUTED_CFG_SHA256)
    before["expected_muted_sha256"] = EXPECTED_MUTED_CFG_SHA256
    after: dict[str, Any] | None = None
    if copy_after_raw is not None:
        after = decode_cfg_mute_fields(copy_after_raw, path=str(copy_path))
        after.update(cfg_copy_diff(source_raw, copy_after_raw))
        after["matches_expected_muted_sha256"] = bool(
            after["sha256"] == EXPECTED_MUTED_CFG_SHA256)
        after["expected_muted_sha256"] = EXPECTED_MUTED_CFG_SHA256
    return {
        "schema": MUTE_SCHEMA,
        "controller": CONTROLLER,
        "checked_at": list(checked_at),
        "mute_offsets": list(MUTE_OFFSETS),
        "mute_field_offsets": {name: offset for name, offset in MUTE_FIELD_OFFSETS},
        "source": source,
        "copy_before_start": before,
        "copy_after_exit": after,
        "engine_env": dict(engine_env),
        "problems": list(problems),
        "notes": [
            "the source data directory is never written; the engine only ever "
            "sees the private copy",
            "diff_byte_offsets is computed byte by byte, not assumed",
            "muted_strict_four_field is this ticket's rule; "
            "muted_legacy_three_field is the v4.3.1 audit's three-field rule, "
            "kept for comparison",
        ],
    }


def mute_report_problems(report: Mapping[str, Any]) -> list[str]:
    """Everything the mute document proves to be wrong, as strings."""

    problems: list[str] = []
    source = report.get("source") or {}
    before = report.get("copy_before_start") or {}
    after = report.get("copy_after_exit")
    if source.get("error"):
        problems.append(f"source cfg undecodable: {source['error']}")
    if not source.get("matches_registered_source_sha256"):
        problems.append(
            "source cfg sha256 "
            f"{source.get('sha256')} != registered {REGISTERED_SOURCE_CFG_SHA256} "
            "(recorded as a difference; the run is not refused on this alone)")
    if before.get("error"):
        problems.append(f"copy cfg undecodable before start: {before['error']}")
    if not before.get("muted_strict_four_field"):
        problems.append(
            f"copy cfg is not strictly muted before start: {before.get('fields')}")
    if not before.get("diff_offsets_match_expected"):
        problems.append(
            "copy-before-start diff offsets "
            f"{before.get('diff_byte_offsets')} != {list(MUTE_OFFSETS)}")
    if not before.get("other_bytes_identical"):
        problems.append(
            "copy-before-start changed bytes outside the four mute offsets: "
            f"{before.get('unexpected_offsets')}")
    if before.get("sha256") and not before.get("matches_expected_muted_sha256"):
        problems.append(
            "copy-before-start sha256 "
            f"{before.get('sha256')} != expected {EXPECTED_MUTED_CFG_SHA256}")
    if after is None:
        problems.append("copy cfg was not re-decoded after the process exited")
    else:
        if after.get("error"):
            problems.append(f"copy cfg undecodable after exit: {after['error']}")
        if not after.get("muted_strict_four_field"):
            problems.append(
                "copy cfg is NOT strictly muted after exit "
                f"({after.get('fields')}); recorded as found, not patched")
        if not after.get("other_bytes_identical"):
            problems.append(
                "copy-after-exit changed bytes outside the four mute offsets: "
                f"{after.get('unexpected_offsets')}")
    return problems


# ---------------------------------------------------------------------------
# the actor input boundary (pixels + action history, nothing else)
# ---------------------------------------------------------------------------
def _actor_planes(buffer: obs_spec_v3.ObservationBufferV3,
                  history: Any) -> dict[str, np.ndarray]:
    """The three **uint8** actor planes, in the frozen digest order.

    ``play (4,112,96)`` -> ``hud (56,160)`` -> ``act_hist (8,18)``.  These are the
    raw bytes of ``ObservationBufferV3.stack()`` / ``.hud()`` and
    ``ActionHistoryBuffer.rows()`` 閳?exactly the arrays
    ``tools/rl/v420_eval_shoot.py::_run_loop`` hashes, so a digest produced here
    is directly comparable with the frozen ``observed_input_sha256``.  The
    normalised tensors a policy consumes are a *derived* view and are not what the
    digest covers.
    """

    rows = (history.rows() if history is not None
            else np.zeros(obs_spec_v3.ACTION_HISTORY_SHAPE, dtype=np.uint8))
    return {
        "play": np.ascontiguousarray(buffer.stack()),
        "hud": np.ascontiguousarray(buffer.hud()),
        "act_hist": np.ascontiguousarray(rows),
    }


def digest_actor_arrays(planes: Mapping[str, np.ndarray]) -> dict[str, str]:
    """sha256 of each actor plane plus one digest over all three, in order."""

    digests = {name: array_sha256(planes[name])
               for name in ("play", "hud", "act_hist")}
    combined = hashlib.sha256()
    for name in ("play", "hud", "act_hist"):
        combined.update(np.ascontiguousarray(np.asarray(planes[name])).tobytes())
    digests["actor_input_digest"] = combined.hexdigest()
    return digests


def build_actor_inputs(pixels_rgba: Any, history: Any) -> dict[str, Any]:
    """The actor's input, built from **only** pixels and action history.

    Exactly two parameters: the 640x480 RGBA8 readback and an
    ``ActionHistoryBuffer`` (or ``None``).  Liveness, bombs, power, stage,
    deaths, frame-event flags and stage-clear are not arguments, not globals this
    function reads, and not keys it returns 閳?:data:`ACTOR_FIELDS` is that
    whitelist and :func:`actor_boundary_problems` checks it at run time.

    Returns the three uint8 planes the digest covers (``play``/``hud``/
    ``act_hist``), their per-plane and combined digests, and ``normalized`` 閳?the
    ``float32`` tensors a policy actually consumes.
    """

    buffer = obs_spec_v3.ObservationBufferV3()
    play, hud = obs_spec_v3.encode_planes_v3(np.asarray(pixels_rgba))
    buffer.push(play, hud)
    planes = _actor_planes(buffer, history)
    arrays = buffer.arrays(history)
    return {
        "play": planes["play"],
        "hud": planes["hud"],
        "act_hist": planes["act_hist"],
        "digests": digest_actor_arrays(planes),
        "normalized": {
            "play": arrays["play"],
            "hud": arrays["hud"],
            "act_hist": arrays["act_hist"],
        },
    }


class ActorInputTap:
    """The live actor input: a persistent obs-v3 stack fed one frame at a time.

    :meth:`pump` takes a readback and pushes its planes; :meth:`inputs` takes the
    action history and returns the uint8 planes plus digests.  Neither method
    accepts or reads a referee field, so the boundary of
    :func:`build_actor_inputs` holds on the live path too.
    """

    def __init__(self) -> None:
        self._buffer = obs_spec_v3.ObservationBufferV3()
        self.frames_pumped = 0

    def reset(self) -> None:
        self._buffer.reset()
        self.frames_pumped = 0

    def pump(self, pixels_rgba: Any) -> int:
        """Encode one readback and push it onto the 4-frame stack."""

        play, hud = obs_spec_v3.encode_planes_v3(np.asarray(pixels_rgba))
        self._buffer.push(play, hud)
        self.frames_pumped += 1
        return self.frames_pumped

    def inputs(self, history: Any) -> dict[str, Any]:
        """The three uint8 actor planes and their digests for the current stack."""

        planes = _actor_planes(self._buffer, history)
        return {
            "play": planes["play"],
            "hud": planes["hud"],
            "act_hist": planes["act_hist"],
            "digests": digest_actor_arrays(planes),
        }

    def stack(self) -> np.ndarray:
        return self._buffer.stack()

    def hud_plane(self) -> np.ndarray:
        return self._buffer.hud()


def actor_boundary_problems() -> list[str]:
    """Re-check the input-isolation boundary structurally (empty == clean)."""

    problems: list[str] = []
    try:
        parameters = list(inspect.signature(build_actor_inputs).parameters)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        return [f"cannot introspect build_actor_inputs: {exc}"]
    if parameters != list(ACTOR_FIELDS):
        problems.append(
            f"build_actor_inputs parameters {parameters} != {list(ACTOR_FIELDS)}")
    for name in parameters:
        for referee in REFEREE_FIELDS:
            if referee.lower() == name.lower():
                problems.append(
                    f"actor input parameter {name!r} is a referee field")
    overlap = sorted(set(ACTOR_FIELDS) & set(REFEREE_FIELDS))
    if overlap:
        problems.append(f"ACTOR_FIELDS overlaps REFEREE_FIELDS: {overlap}")
    for method_name in ("pump", "inputs"):
        for name in inspect.signature(
                getattr(ActorInputTap, method_name)).parameters:
            if name == "self":
                continue
            if name in REFEREE_FIELDS:
                problems.append(
                    f"ActorInputTap.{method_name} takes the referee field {name!r}")
    return problems


# ---------------------------------------------------------------------------
# EVENTS: the raw reply, with the stage-clear edge kept distinct from the level
# ---------------------------------------------------------------------------
def parse_events_reply_full(reply: str) -> dict[str, Any]:
    """Decode one raw ``EVENTS`` reply without inventing a missing field.

    ``frameEventFlags`` bit ``0x0080`` is the **level** (it follows the clear
    screen); the rising edge is ``stageClearEdge`` and its frame is
    ``stageClearEdgeFrame``.  The two are kept separate here because the whole
    point of the double-signal record is that they are not the same field.

    Raises :class:`TechnicalInvalid` when the reply is not ``OK`` or when a
    required field is missing 閳?a missing ``stageClearEdgeFrame`` must never be
    read as "0", i.e. as "no clear".
    """

    fields = _parse_kv(reply)
    if fields.get("_verb") != "OK":
        raise TechnicalInvalid(f"EVENTS reply is not OK: {reply!r}")
    missing = [name for name in EVENTS_REQUIRED_FIELDS if name not in fields]
    if missing:
        raise TechnicalInvalid(
            f"EVENTS reply missing {','.join(missing)}: {reply!r}")
    flags = int(fields["frameEventFlags"], 16)
    event = protocol.parse_events_reply(reply)
    return {
        "raw": reply,
        "frames": int(fields["frames"]),
        "frame_event_flags": flags,
        "frame_event_names": list(event.names),
        "player_hit": bool(event.player_hit),
        "stage_clear": int(fields["stageClear"]),
        "stage_clear_edge": int(fields["stageClearEdge"]),
        "stage_clear_edge_frame": int(fields["stageClearEdgeFrame"]),
        "stage_clear_screen_state": (
            int(fields["stageClearScreenState"])
            if "stageClearScreenState" in fields else None),
        "stage_clear_seq": (
            int(fields["stageClearSeq"]) if "stageClearSeq" in fields else None),
        "level_stage_clear_bit": int(flags & EVENT_FLAG_STAGE_CLEAR),
        "has_clear_signal": bool(
            int(fields["stageClear"]) == 1
            or int(fields["stageClearEdge"]) == 1
            or flags & EVENT_FLAG_STAGE_CLEAR),
    }


def parse_pixels_reply(reply: str) -> dict[str, Any]:
    """Decode one ``PIXELS`` reply: size, nonzero count and the RGBA checksum."""

    fields = _parse_kv(reply)
    if fields.get("_verb") != "OK":
        raise TechnicalInvalid(f"PIXELS reply is not OK: {reply!r}")
    checksum = fields.get("checksum")
    if not checksum:
        raise TechnicalInvalid(f"PIXELS reply has no checksum: {reply!r}")
    width = height = None
    size = fields.get("pixels")
    if size and "x" in size:
        try:
            width, height = (int(value) for value in size.split("x", 1))
        except ValueError:
            width = height = None
    return {
        "raw": reply,
        "pixels": size,
        "width": width,
        "height": height,
        "nonzero": int(fields["nonzero"]) if "nonzero" in fields else None,
        "checksum": checksum,
    }


# ---------------------------------------------------------------------------
# the teacher tape: the frozen loader, read-only
# ---------------------------------------------------------------------------
def load_teacher_tape(path: str | Path) -> dict[str, Any]:
    """Rebuild the recorded clear tape with the training code's own loader.

    ``touhou_rl.v420.rl.load_clear_schedule`` is imported read-only and called,
    never copied: it concatenates slot 0's eight leading decisions
    (``history_frames``/``history``), the 6,588 ``executed_cand`` rows and
    ``tail_decisions`` repeats of the last one.  The shard sha256 is pinned to the
    registered value, so replaying a different tape is a refusal.
    """

    from touhou_rl.v420.rl import load_clear_schedule  # noqa: PLC0415 - heavy import

    shard = Path(path)
    if not shard.is_absolute():
        shard = REPO_ROOT / shard
    if not shard.is_file():
        raise Refusal(f"teacher tape is missing: {shard}")
    info = load_clear_schedule(shard, tail_decisions=TEACHER_TAIL_DECISIONS)
    if info["shard_sha256"] != REGISTERED_TAPE_SHA256:
        raise Refusal(
            f"tape sha256 {info['shard_sha256']} != registered "
            f"{REGISTERED_TAPE_SHA256}; this subcommand replays the one audited "
            "clear tape")
    schedule = [(int(frame), int(cand)) for frame, cand in info["schedule"]]
    if not schedule:
        raise Refusal(f"the recording {shard} has no decisions")
    bombs = tape_bomb_values(shard)
    if not bombs or any(bombs):
        raise Refusal(
            f"the tape carries non-zero bombs {sorted(set(bombs))}; this ticket "
            "runs with bomb=0 for every decision")
    payload = {
        "shard": str(shard.resolve()),
        "shard_sha256": str(info["shard_sha256"]),
        "seed": int(info["seed"]),
        "slots": int(info["slots"]),
        "leading_decisions": int(info["leading_decisions"]),
        "leading_candidates": [int(value) for value in info["leading_candidates"]],
        "tail_decisions": int(info["tail_decisions"]),
        "schedule_decisions": len(schedule),
        "last_recorded_decision_frame": int(info["last_recorded_decision_frame"]),
        "first_schedule_frame": int(schedule[0][0]),
        "first_schedule_cand": int(schedule[0][1]),
        "loader": "touhou_rl.v420.rl.load_clear_schedule",
        "bomb_rows": len(bombs),
        "bomb_values_unique": sorted(set(bombs)),
        "bomb_all_zero": bool(bombs and not any(bombs)),
        "candidates": [cand for _frame, cand in schedule],
    }
    if not payload["bomb_all_zero"]:
        raise Refusal(
            f"the tape carries non-zero bombs {sorted(set(bombs))}; this ticket "
            "runs with bomb=0 for every decision")
    return payload


#: The frozen loader's default tail; the v8.30 registration used the same value
#: (8 leading + 6,588 recorded + 64 repeats; the clear is reached inside the
#: first two repeats, which is why the recorded decision count is 6,598).
TEACHER_TAIL_DECISIONS = 64


def tape_bomb_values(path: str | Path, *, limit: int | None = None
                     ) -> list[int]:
    """Every ``bomb`` value in a ``slots.jsonl`` tape (``limit`` for a prefix)."""

    values: list[int] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            values.append(int(row.get("bomb", 0)))
            if limit is not None and len(values) >= int(limit):
                break
    return values


def tape_is_bomb_free(path: str | Path) -> bool:
    """True when every recorded decision carries ``bomb = 0``."""

    values = tape_bomb_values(path)
    return bool(values) and not any(values)


# ---------------------------------------------------------------------------
# action sources
# ---------------------------------------------------------------------------
class ConstantActions:
    """The pixel-smoke action source: one fixed candidate, every decision."""

    kind = "constant"

    def __init__(self, cand: int) -> None:
        value = int(cand)
        if not 0 <= value < gym_actions.N_CAND:
            raise Refusal(
                f"--constant-cand {cand} is outside the {gym_actions.N_CAND} "
                "move-focus candidates")
        self.cand = value
        self.decisions = 0

    def decision(self, index: int) -> tuple[int, str]:
        self.decisions += 1
        return self.cand, "constant"


class TapeActions:
    """The teacher-calibration action source: the recorded schedule, in order.

    Decision ``i`` takes ``schedule[i]``; the schedule already contains the eight
    leading decisions and the frozen tail repeats, so a decision past its end can
    only mean the engine ran further than the recording 閳?it repeats the last
    candidate and says so in the row's ``source`` label.
    """

    kind = "tape"

    def __init__(self, tape: Mapping[str, Any]) -> None:
        candidates = [int(value) for value in tape["candidates"]]
        if not candidates:
            raise Refusal("the tape schedule is empty")
        self.candidates = candidates
        self.leading = int(tape["leading_decisions"])
        self.slots = int(tape["slots"])
        self.decisions = 0

    def label(self, index: int) -> str:
        if index < self.leading:
            return "leading"
        if index < self.leading + self.slots:
            return "tape"
        return "tail"

    def decision(self, index: int) -> tuple[int, str]:
        position = int(index)
        self.decisions += 1
        if position < len(self.candidates):
            return self.candidates[position], self.label(position)
        return self.candidates[-1], "tail_past_schedule"


# ---------------------------------------------------------------------------
# the engine host: one process, one shared block, all protocol traffic
# ---------------------------------------------------------------------------
class EngineHost:
    """One ``th08-modern --rl-control`` process in the ``env`` role.

    Every method is a thin, timed wrapper around the frozen
    :class:`touhou_rl.v400.gym.protocol.HarnessClient` plus the shared block.  The
    only thing this class adds is ``EVENTS``/``PIXELS`` raw-reply capture, which
    the frozen client decodes into dataclasses that drop those fields.
    """

    def __init__(
        self,
        *,
        engine: str | Path,
        data_dir: str | Path,
        shm_name: str,
        timeout: float,
        env: Mapping[str, str],
        meter: CostMeter,
    ) -> None:
        self.engine = Path(engine)
        self.data_dir = Path(data_dir)
        self.shm_name = str(shm_name).lstrip("/")
        self.timeout = float(timeout)
        self.env = dict(env)
        self.meter = meter
        self.client: protocol.HarnessClient | None = None
        self.shm: SharedBlock | None = None
        self.hello: dict[str, str] = {}
        self.shm_reply: dict[str, str] = {}
        self.readback_reply: dict[str, str] = {}
        self.reset_reply: dict[str, str] = {}

    # -- lifetime ---------------------------------------------------------
    def open(self, *, stage: int, difficulty_index: int, shot_type: int,
             seed: int) -> dict[str, Any]:
        """Start the engine, attach the block and RESET one stage."""

        with self.meter.measure("startup"):
            client = protocol.HarnessClient(
                self.engine, self.data_dir, shm_name=self.shm_name,
                env=self.env, timeout=self.timeout, role="env")
            client.start()
            self.hello = client.hello()
        self.client = client
        with self.meter.measure("readback"):
            self.shm_reply = client.attach_shm()
            self.readback_reply = client.readback(True)
        if self.shm_reply.get("active") != "1":
            raise Refusal(f"shared block is not active: {self.shm_reply}")
        size = int(self.shm_reply.get("bytes", -1))
        if size != SHM_EXPECTED_BYTES:
            raise Refusal(
                f"shared block is {size} bytes, expected {SHM_EXPECTED_BYTES} "
                f"({PIXELS_OFFSET} + {PIXEL_WIDTH}x{PIXEL_HEIGHT}x4)")
        with self.meter.measure("readback"):
            self.shm = SharedBlock(client.shm_name or self.shm_name)
        with self.meter.measure("startup"):
            self.reset_reply = client.reset(
                stage=int(stage), difficulty=int(difficulty_index),
                team=int(shot_type), seed=int(seed), practice=False)
        if self.reset_reply.get("ready") != "1":
            raise Refusal(
                f"RESET did not reach a playable state: {self.reset_reply}")
        return {
            "hello": dict(self.hello),
            "shm": dict(self.shm_reply),
            "readback": dict(self.readback_reply),
            "reset": dict(self.reset_reply),
        }

    def close(self) -> None:
        shm, self.shm = self.shm, None
        client, self.client = self.client, None
        try:
            if shm is not None:
                shm.close()
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                for path in Path("/dev/shm").glob(f"{self.shm_name}*"):
                    try:
                        path.unlink()
                    except OSError:
                        pass

    # -- protocol ---------------------------------------------------------
    def _require(self) -> protocol.HarnessClient:
        if self.client is None:
            raise SliceError("engine is not running")
        return self.client

    def step(self, mask: int, frames: int) -> dict[str, str]:
        with self.meter.measure("step"):
            return self._require().step(int(mask), int(frames))

    def events_raw(self) -> str:
        with self.meter.measure("audit"):
            return self._require().request("EVENTS")

    def state(self) -> dict[str, Any]:
        with self.meter.measure("audit"):
            return state_from_reply(self._require().request("STATE"))

    def hud(self) -> dict[str, Any]:
        with self.meter.measure("audit"):
            snapshot = self._require().hud()
        return {
            "frame": int(snapshot.frame),
            "display_score": int(snapshot.display_score),
            "globals_score": int(snapshot.globals_score),
            "lives": int(snapshot.lives),
            "bombs": int(snapshot.bombs),
            "power": int(snapshot.power),
            "graze": int(snapshot.graze),
            "point": int(snapshot.point),
            "time": int(snapshot.time),
            "game_over": int(snapshot.game_over),
        }

    def pixels(self) -> tuple[dict[str, int], np.ndarray]:
        with self.meter.measure("audit"):
            block = self.shm
            if block is None:
                raise SliceError("shared block is not mapped")
            header = block.header()
            rgba = block.pixels()
        return header, rgba

    def pixels_checksum(self) -> dict[str, Any]:
        with self.meter.measure("audit"):
            reply = self._require().request("PIXELS")
        return parse_pixels_reply(reply)


# ---------------------------------------------------------------------------
# the shared block: one read, immediately after STEP, and the frame relation
# ---------------------------------------------------------------------------
#: FNV-1a-64, the digest the engine's ``PIXELS`` reply carries: offset basis
#: ``14695981039346656037``, prime ``1099511628211``
#: (``rlharness.cpp`` ``HandlePixels``).  Computed here over the captured bytes
#: so the checksum never depends on a second readback.
FNV1A64_OFFSET_BASIS = 14695981039346656037
FNV1A64_PRIME = 1099511628211

#: The **one** audited command order for a *sampled* ``STEP`` unit.  ``shm``
#: must come second: it is the only read that can still see the frame the
#: ``STEP`` just drew.  ``EVENTS`` only reads the event word, so it may sit
#: between the block read and the frame check: that is what makes the single
#: allowed re-read legal without any command that could ``PublishState``/
#: ``PublishPixels`` in between.  ``STATE`` (``rlharness.cpp:1362``) and
#: ``PIXELS`` (``:1190`` -> ``PublishPixels`` at ``:1372``) both stamp the header
#: with the *current* ``FrameCount()``; issuing either before the block read is
#: the V831-SLICE-01 run-2 failure (decision 0 / frame 1).
#:
#: V831-SLICE-01b slimmed this from six commands to four.  The two that left the
#: per-unit path are *read-only* (``HUD`` and ``PIXELS``): neither one is an
#: input to ``STEP``, the mask trace, the action sequence or the observation
#: digest, and both are still issued on the units named in
#: :data:`AUDIT_COMMAND_GATED`.  Nothing else about the unit changed: one
#: ``STEP``, one shared-block capture (plus the single allowed re-read), one
#: ``EVENTS`` and one ``STATE``.
AUDIT_COMMAND_SEQUENCE: tuple[str, ...] = ("step", "shm", "events", "state")

#: One SHOOT pulse period in *units*: 48 decisions x 2 units.  This is the
#: default cadence of the gated ``HUD`` cross-check; :func:`gated_audit_plan`
#: derives the live value from the effective ``--shoot-pulse-period`` and
#: ``--decision-frames`` so a non-default pulse keeps "once per period".
AUDIT_HUD_PERIOD_UNITS = SHOOT_PULSE_PERIOD * DECISION_FRAMES

#: The gated audit commands and exactly when they are allowed to fire.
#:
#: Why dropping them from the per-unit path loses **no ticket field**:
#:
#: ``hud``
#:     Every resource the ticket asks for is a ``STATE`` field on the same unit:
#:     ``lives_remaining``, ``bombs_remaining``, ``player_power``, ``score``,
#:     ``graze`` and ``deaths`` (``game_over`` comes from the ``STEP`` reply).
#:     The rows therefore keep the same columns, filled from ``STATE`` and
#:     tagged ``hud_source="state_derived"``; ``HUD`` is a *second* opinion and
#:     is sampled on the first unit, once per pulse period and on the last unit
#:     so the two sources are cross-checked and any difference is a warning.
#: ``pixels``
#:     The ticket's raw-picture checksum is ``pixels_checksum_local``, the local
#:     FNV-1a-64 of the *captured* RGBA, and it is still computed on every unit.
#:     The ``PIXELS`` reply is an independent engine-side re-readback
#:     (``HandlePixels`` runs a fresh ``glReadPixels`` and republishes), so it can
#:     never replace the local value; it is sampled on the first and the last
#:     unit, and a difference is recorded in ``warnings``, never used as a gate.
AUDIT_COMMAND_GATED: dict[str, dict[str, Any]] = {
    "hud": {
        "when": ("the first unit, every AUDIT_HUD_PERIOD_UNITS-th unit (one "
                 "SHOOT pulse period), and once after the last unit"),
        "first_unit": True,
        "every_units": AUDIT_HUD_PERIOD_UNITS,
        "last_unit": ("issued by run_subcommand right after the loop, before "
                      "any further STEP (result.audit.final_crosscheck)"),
        "replaced_by": ("hud_source=state_derived: lives/bombs/power/score/graze/"
                        "deaths from STATE, game_over from the STEP reply"),
        "difference_policy": "recorded in warnings and hud_mismatches, never a gate",
    },
    "pixels": {
        "when": ("the first unit and once after the last unit; the local "
                 "pixels_checksum_local is computed once per decision "
                 "(last unit) and pixel_frame_blake2b128 on every unit"),
        "first_unit": True,
        "every_units": None,
        "last_unit": ("issued by run_subcommand right after the loop, before "
                      "any further STEP (result.audit.final_crosscheck)"),
        "replaced_by": ("pixels_checksum_local (per decision: the local "
                        "FNV-1a-64 of the last unit's captured RGBA) and "
                        "pixel_frame_blake2b128 (per unit)"),
        "difference_policy": "recorded in warnings, never a gate",
    },
}

#: Where the recorded resource columns (``lives``/``bombs``/``power``/``score``/
#: ``graze``/``deaths``) come from now that ``HUD`` is not read on every unit:
#: ``STATE`` (+ the ``STEP`` reply for ``game_over``).  It is a constant because
#: it is a *fact about the instrument*, not a per-row label -- a gated ``HUD``
#: read is recorded separately in ``hud_crosscheck`` so the two are never mixed.
HUD_SOURCE_STATE = "state_derived"

#: The per-frame pixel identity added in V831-SLICE-01c: ``blake2b`` with a
#: 128-bit digest over the captured RGBA in the engine's raw bottom-up order.
#: Measured 1.10 ms per 1,228,800 B on this host (60-run median) against
#: 54.02 ms for the FNV-1a-64, so it can run on **every** unit while the FNV
#: runs once per decision.
PIXEL_FRAME_DIGEST_ALGORITHM = "blake2b-128"
PIXEL_FRAME_DIGEST_FIELD = "pixel_frame_blake2b128"

#: The scope of each local digest.  The ticket records the raw-picture checksum
#: **per decision** ("每次决策记录逻辑帧、像素帧、原始画面校验值、预处理后的
#: actor 输入摘要、请求动作、每段实际 STEP mask 和生效帧数"): one row per
#: decision, not one row per frame.  V831-SLICE-01b computed the FNV per *unit*
#: (13,195 times) to fill a ``frame_rows`` list that is never written to
#: ``result.json`` beyond the last :data:`SIGNAL_WINDOW_FRAMES` rows; 01c keeps
#: the recorded value identical (the decision's **last** unit's captured bytes,
#: which is what ``decisions.jsonl`` already carried) and gives every unit a
#: cheap per-frame digest instead.
CHECKSUM_SCOPE_PER_DECISION = "per_decision"
FRAME_DIGEST_SCOPE_PER_UNIT = "per_unit_blake2b128"
CHECKSUM_SCOPE_RATIONALE = (
    "工单原文（V831-SLICE-01 字段表）：'每次决策记录逻辑帧、像素帧、原始画面校验值、"
    "预处理后的 actor 输入摘要、请求动作、每段实际 STEP mask 和生效帧数' -- the "
    "ticket records ONE ROW PER DECISION, not one row per frame.  So "
    "pixels_checksum_local is measured on the decision's last STEP unit (the "
    "exact bytes decisions.jsonl already carried) and compared with the engine "
    "PIXELS reply on the gated units.  Every unit still carries a recomputable "
    "per-frame pixel identity, pixel_frame_blake2b128 (blake2b-128, 1.10 ms per "
    "1,228,800 B), so no frame is left without a content digest")

#: The wall-clock defaults, unchanged by this ticket: the parent agent passes a
#: larger ``--timeout-s`` in its own acceptance script and reports the measured
#: wall clock separately from "does it fit 600 s".  The CLI accepts any positive
#: number and records it verbatim (``result.timeout_s`` and
#: ``result.preflight.timeout_s``).
DEFAULT_TIMEOUT_S: dict[str, float] = {
    "pixel-smoke": 120.0,
    "teacher-calibration": 600.0,
}

#: ``header->frame`` after a ``STEP`` whose own ``PublishState`` ran: the
#: post-increment ``FrameCount()``, i.e. ``EVENTS.frames`` (rl_v3, T7.3).
RELATION_STATE_PUBLISH = "state_publish"
#: ``header->frame`` after a ``STEP`` that returned before that publish (the
#: exit path) or on a pre-T7.3 build: the frame ``AfterDrawFrame`` published,
#: i.e. ``EVENTS.frames - 1``.
RELATION_DRAW_PUBLISH = "draw_publish"

#: The frame relation, spelled once so the row, the conflict record, the
#: docstring and the tests cannot drift apart.
PIXEL_FRAME_RULE = (
    "the captured RGBA belongs to frame EVENTS.frames - 1 (AfterDrawFrame "
    "publishes before ++g_frameCount, main.cpp:143-146); header->frame reads "
    "EVENTS.frames when a PublishState ran after that draw (rl_v3 STEP's own, "
    "rlharness.cpp:901/1362) and EVENTS.frames - 1 when none did")


def fnv1a64(data: bytes) -> int:
    """FNV-1a-64 of ``data`` as an unsigned 64-bit integer.

    The engine computes the same digest in ``HandlePixels`` (offset basis
    ``14695981039346656037``, prime ``1099511628211``) over a **fresh**
    ``glReadPixels`` buffer; this one is computed over the captured shared-block
    bytes, so the two are comparable but not the same measurement.
    """

    value = FNV1A64_OFFSET_BASIS
    for byte in bytes(data):
        value = ((value ^ byte) * FNV1A64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def fnv1a64_hex(data: bytes) -> str:
    """FNV-1a-64 as the 16 lowercase hex digits a ``PIXELS`` reply carries."""

    return f"{fnv1a64(data):016x}"


def gated_audit_plan(*, unit_number: int, hud_period: int) -> dict[str, bool]:
    """Which gated audit commands this unit must issue.

    ``unit_number`` is 0-based over the whole run.  ``hud`` fires on the first
    unit and on every ``hud_period``-th unit (the default period is one SHOOT
    pulse period, :data:`AUDIT_HUD_PERIOD_UNITS` units); ``pixels`` fires on the
    first unit only.  The **last** unit is not knowable here -- a unit only
    learns it is the last one after the termination check -- so its ``hud`` and
    ``pixels`` are issued by :func:`run_subcommand` immediately after the loop,
    with no ``STEP`` in between (``result.audit.final_crosscheck``).  The
    per-unit rule is the whole contract, so it is a function and not an
    ``if`` buried in the loop.
    """

    unit_number = int(unit_number)
    period = max(1, int(hud_period))
    return {
        "hud": bool(unit_number == 0 or unit_number % period == 0),
        "pixels": bool(unit_number == 0),
    }


def pixel_frame_blake2b128(raw_bytes: bytes) -> str:
    """The per-frame C-level pixel identity: 32 hex digits of blake2b-128.

    Computed over the captured RGBA in the engine's raw bottom-up byte order,
    i.e. the exact bytes :func:`fnv1a64_hex` hashes, so a frame can be
    re-identified cheaply (measured 1.10 ms over 1,228,800 B, against 54.02 ms
    for FNV-1a-64 on the same buffer -- see :data:`PIXEL_FRAME_DIGEST_ALGORITHM`).
    It is a *content identity*, not a replacement for the engine-comparable FNV:
    the two algorithms are different and only the FNV can be compared with a
    ``PIXELS`` reply.
    """

    return hashlib.blake2b(bytes(raw_bytes), digest_size=16).hexdigest()


def capture_shared_block(host: Any, *, meter: CostMeter | None = None,
                         checksum: bool = True) -> dict[str, Any]:
    """Read the shared block **once**, immediately after a ``STEP``.

    This is the only read whose bytes are still the frame that ``STEP`` drew:
    every later command that publishes (``STATE``, ``PIXELS``) rewrites
    ``header->frame`` with the current ``FrameCount()`` *and* ``PIXELS`` may
    return a different buffer altogether.  Call it as the first thing after
    :meth:`EngineHost.step` and do not reorder it
    (:data:`AUDIT_COMMAND_SEQUENCE`).

    The returned ``header_frame`` is recorded, not trusted: on rl_v3 it is the
    post-increment count after the step's own ``PublishState`` while ``rgba`` is
    still one frame older.  :func:`classify_captured_frame` turns the pair
    ``(header_frame, EVENTS.frames)`` into the frame the bytes belong to.

    Two local digests are produced, and they have different scopes
    (V831-SLICE-01c):

    ``pixels_checksum_local`` (``checksum=True``)
        The engine-comparable FNV-1a-64 (raw bottom-up order).  The ticket asks
        for the raw-picture checksum **per decision** ("每次决策记录...原始画面
        校验值"), so :func:`run_decisions` asks for it on the last unit of each
        decision -- 54.02 ms per call, ~6,598 calls instead of 13,195.
    ``pixel_frame_blake2b128`` (always)
        A 128-bit blake2b over the same bytes, 1.10 ms, so every unit still has
        a recomputable pixel identity (``fnv_units`` vs ``blake2b_units`` in
        ``result.audit``).

    ``meter`` (optional) wraps the **local** digest work -- the reversed
    raw-order copy, the sha256, the blake2b, the optional FNV and the non-zero
    count -- in the ``capture`` bucket.  The shared-block read itself is already
    inside the ``audit`` bucket (``EngineHost.pixels``), so the two never
    overlap.  This is the measurement whose absence left V831-SLICE-01's teacher
    run 83% unaccounted: ~60 ms/unit of pure-Python hashing, not protocol
    traffic.
    """

    header, rgba = host.pixels()
    with (nullcontext() if meter is None else meter.measure("capture")):
        header = {str(key): int(value) for key, value in dict(header).items()}
        pixels = np.ascontiguousarray(np.asarray(rgba, dtype=np.uint8))
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise TechnicalInvalid(
                f"shared block pixels are not RGBA8 HxWx4: shape {pixels.shape}")
        height, width = int(pixels.shape[0]), int(pixels.shape[1])
        # ``SharedBlock.pixels`` flips glReadPixels' bottom-up rows for the actor;
        # the engine's own checksum covers the unflipped buffer, so hash the raw
        # order to keep the two numbers comparable.
        raw_order = np.ascontiguousarray(pixels[::-1]).tobytes()
        return {
            "header": header,
            "header_frame": int(header["frame"]),
            "flags": int(header["flags"]),
            "width": width,
            "height": height,
            "state_hash": int(header.get("state_hash", 0)),
            "pixels_valid": bool(int(header["flags"]) & FLAG_PIXELS_VALID),
            "rgba": pixels,
            "rgba_sha256": array_sha256(pixels),
            "rgba_bytes": int(pixels.size),
            PIXEL_FRAME_DIGEST_FIELD: pixel_frame_blake2b128(raw_order),
            "pixels_checksum_local": (fnv1a64_hex(raw_order) if checksum
                                      else None),
            "checksum_computed": bool(checksum),
            "pixels_nonzero": int(np.count_nonzero(pixels)),
            "pixels_size": f"{width}x{height}",
            "checksum_byte_order": "raw bottom-up, the engine PIXELS order",
        }


def classify_captured_frame(header_frame: int, events_frames: int
                            ) -> dict[str, Any] | None:
    """Which frame the captured RGBA belongs to, or ``None`` when illegal.

    Two header values are legal, and only these two:

    ``RELATION_STATE_PUBLISH`` (``header_frame == events_frames``)
        A ``PublishState`` ran after the last draw and stamped the
        post-increment count.  This is the normal rl_v3 path: ``STEP`` ends with
        its own ``PublishState`` (``rlharness.cpp:901`` -> ``:1362``).
    ``RELATION_DRAW_PUBLISH`` (``header_frame == events_frames - 1``)
        No publish ran after the last draw: a ``STEP`` that returned early (the
        ``exit=1`` path) or a build without T7.3.  The header still carries the
        frame ``AfterDrawFrame`` published.

    In both branches the bytes are ``EVENTS.frames - 1``, which is what the run
    records as ``pixel_frame``.  Anything else means the capture cannot be tied
    to a frame this ``STEP`` drew, so it is a technical conflict.
    """

    header_frame = int(header_frame)
    events_frames = int(events_frames)
    legal = [events_frames - 1, events_frames]
    common = {
        "header_frame": header_frame,
        "events_frames": events_frames,
        "legal_header_frames": legal,
        "pixel_frame": events_frames - 1,
    }
    if header_frame == events_frames:
        return dict(common, relation=RELATION_STATE_PUBLISH, why=(
            "a PublishState ran after the draw (rl_v3 STEP publishes itself, "
            "rlharness.cpp:901/1362) and stamped the post-increment FrameCount"))
    if header_frame == events_frames - 1:
        return dict(common, relation=RELATION_DRAW_PUBLISH, why=(
            "no publish after the draw (STEP returned early, e.g. exit=1, or a "
            "pre-T7.3 build); the header still carries AfterDrawFrame's frame"))
    return None


def pixel_relation_reason(header_frame: int, events_frames: int) -> str:
    """The conflict text for an illegal ``(header_frame, EVENTS.frames)`` pair."""

    return (
        f"pixel frame {int(header_frame)} is neither "
        f"{int(events_frames) - 1} nor {int(events_frames)} for EVENTS frames "
        f"{int(events_frames)}; the captured RGBA cannot be tied to a frame "
        "this STEP drew")


# ---------------------------------------------------------------------------
# the decision loop (shared by both subcommands)
# ---------------------------------------------------------------------------
def _readings(state: Mapping[str, Any]) -> dict[str, Any]:
    """The four referee scalars this tool reports (never an actor input)."""

    return {
        "deaths": float(state["deaths"]),
        "lives": float(state["lives_remaining"]),
        "bombs": float(state["bombs_remaining"]),
        "score": int(state["score"]),
        "stage": int(state["stage"]),
    }


def _state_resources(state: Mapping[str, Any]) -> dict[str, Any]:
    """The resource columns ``HUD`` used to supply, taken from ``STATE``.

    ``STATE`` carries all of them (``lives_remaining``, ``bombs_remaining``,
    ``player_power``, ``score``, ``graze``, ``deaths``), which is why the
    per-unit ``HUD`` command is not needed to keep a single recorded column.
    ``game_over`` is the only one that is not a ``STATE`` field; the loop takes
    it from the ``STEP`` reply instead.
    """

    return {
        "lives": int(float(state["lives_remaining"])),
        "bombs": int(float(state["bombs_remaining"])),
        "player_power": float(state["player_power"]),
        "score": int(state["score"]),
        "graze": int(state["graze"]),
        "deaths": float(state["deaths"]),
    }


def run_decisions(
    *,
    host: EngineHost,
    actions: Any,
    tap: ActorInputTap,
    history: ActionHistoryBuffer,
    meter: CostMeter,
    decision_frames: int,
    shoot_pulse_period: int,
    max_controlled_frames: int,
    start_frames: int,
    initial_state: Mapping[str, Any],
    timeout_s: float,
    stop_on_first_death: bool,
    allow_partial_tail: bool = False,
    start_renders: int = 0,
    audit_stride: int = 1,
    on_decision: Callable[[dict[str, Any]], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Drive one trajectory: one decision, one audit, one row -- in that order.

    The loop is bit-for-bit the frozen v4.2.0 ``_run_loop`` skeleton: 2 one-frame
    ``STEP`` units per decision (``obs_window="consecutive"``), the release
    decision's **last** unit drops ``SHOOT|BOMB``, and ``frames`` counts advanced
    game frames.  What it adds is a per-unit audit and a per-decision row.

    The per-unit audit issues exactly :data:`AUDIT_COMMAND_SEQUENCE`::

        step -> shm -> events -> state

    and, on the units named by :func:`gated_audit_plan`, the read-only tail
    ``hud`` (first unit, every pulse period, plus the post-loop last-unit check)
    and ``pixels`` (first unit, plus the post-loop last-unit check) -- the
    order is always ``step -> shm -> events -> state -> [hud] -> [pixels]``.
    ``shm`` is the :func:`capture_shared_block` read and it comes **immediately
    after** the ``STEP``, before any command that can ``PublishState``
    (``STATE``) or ``PublishPixels`` (``PIXELS``).  This is the
    V831-SLICE-01 run-2 fix: the old block read ``EVENTS``, ``STATE``, ``HUD``
    and then the block, so ``STATE`` had already rewritten ``header->frame`` to
    the post-increment count while the bytes were the pre-increment draw, and
    ``header.frame != EVENTS.frames - 1`` was guaranteed.  Never reorder it; no
    later read can recover the frame the ``STEP`` drew.

    Digest scopes (V831-SLICE-01c): ``pixels_checksum_local`` -- the
    engine-comparable FNV-1a-64 -- is the ticket's **per-decision** value
    (:data:`CHECKSUM_SCOPE_PER_DECISION`), so it is computed on the decision's
    last unit (+ once on the first unit so the gated first ``PIXELS`` keeps a
    local counterpart, counted in ``fnv_crosscheck_units``); ``fnv_units``
    therefore equals the decision count.  Every unit still computes
    ``pixel_frame_blake2b128`` (blake2b-128, ~1.1 ms), so ``blake2b_units``
    equals ``step_units`` even under ``--audit-stride``.

    ``audit_stride`` (default 1) is the V831-SLICE-01b *time* escape hatch and is
    **off by default**: with ``N > 1`` only every N-th unit issues
    ``EVENTS``/``STATE``; the other units are ``STEP`` + capture only.  ``STEP``,
    the mask trace and the action sequence are untouched by it.  A skipped unit
    carries the previous ``STATE`` readings forward (``state_sampled=False`` and
    ``readings_source="carried"`` in its row), so a stage advance or a death can
    only be noticed at the next sampled unit -- which is recorded, not hidden.

    Termination order: the audit's protocol conflict first, then
    ``STATE.stage > stage_start`` (a real clear, the only exit-0 condition of the
    teacher run), then ``exit``/``game_over``/``deaths``, then a short
    ``advanced``, and the controlled-frame cap last.
    """

    if decision_frames <= 0:
        raise Refusal(f"decision_frames must be positive, got {decision_frames}")
    if max_controlled_frames < 0:
        raise Refusal(
            f"max_controlled_frames must be >= 0, got {max_controlled_frames}")
    if int(audit_stride) <= 0:
        raise Refusal(f"audit_stride must be >= 1, got {audit_stride}")
    audit_stride = int(audit_stride)
    # One SHOOT pulse period, in units: the HUD cross-check cadence.
    hud_period = max(1, int(shoot_pulse_period) * int(decision_frames))
    deadline = clock() + float(timeout_s)
    frames = int(start_frames)
    warmup_frames = int(start_frames)
    start = _readings(initial_state)
    stage_start = int(initial_state["stage"])
    previous = dict(start)

    mask_digest = hashlib.sha256()
    action_digest = hashlib.sha256()
    obs_digest = hashlib.sha256()
    mask_counts: dict[int, int] = {}
    action_counts = np.zeros(gym_actions.N_CAND, dtype=np.int64)
    release_decisions: list[int] = []
    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    problems: list[str] = []
    warnings: list[str] = []
    if audit_stride > 1:
        warnings.append(
            f"audit_stride={audit_stride}: only every {audit_stride}-th STEP "
            "unit issues EVENTS/STATE; the others are STEP + shared-block "
            "capture, so a stage advance or a death can be noticed up to "
            f"{audit_stride - 1} frame(s) late.  STEP order/masks/frames and "
            "the three digests are unchanged (reported, not a gate)")

    steps = 0
    units_sent = 0
    checksum_mismatches = 0
    renders = int(start_renders)
    shoot_frames = 0
    no_shoot_frames = 0
    rising_edges = 0
    releases_applied = 0
    zero_advances = 0
    previous_shoot = 0
    decisions_without_effect = 0
    history_pushes = 0
    audit_sampled_units = 0
    audit_commands_sent: dict[str, int] = {
        name: 0 for name in ("step", "shm", "events", "state", "hud", "pixels")}
    hud_measured_units = 0
    hud_crosschecks = 0
    hud_mismatches = 0
    pixels_crosschecks = 0
    # Digest scopes (V831-SLICE-01c): the FNV is the per-decision value, the
    # blake2b is per unit.  These three counters are what ``result.audit``
    # reports, and the offline tests assert fnv_units == decisions and
    # blake2b_units == step_units.
    fnv_units = 0
    fnv_crosscheck_units = 0
    blake2b_units = 0
    first_conflict: dict[str, Any] | None = None
    terminal: dict[str, Any] = {"kind": TERMINAL_UNKNOWN, "reason": REASON_BUDGET_EXHAUSTED,
                                "frame": None, "decision_index": None}
    signals = {
        "first_clear_event_frame": None,
        "first_clear_event_source": None,
        "first_clear_edge_frame": None,
        "first_clear_edge_frame_reported": None,
        "first_level_bit_frame": None,
        "first_stage_advance_frame": None,
        "stage_advanced": False,
        "events_clear_seen": False,
        "order": [],
        "interval_frames": None,
        "missing_second_signal": False,
        "player_hit_frames": [],
    }
    signal_window: list[dict[str, Any]] = []

    def note_clear(name: str) -> None:
        if name not in signals["order"]:
            signals["order"].append(name)

    try:
        while True:
            controlled = frames - warmup_frames
            remaining = int(max_controlled_frames) - controlled
            if remaining <= 0 or (
                    remaining < decision_frames and not allow_partial_tail):
                terminal = {
                    "kind": TERMINAL_TRUNCATED, "reason": REASON_FRAME_CAP,
                    "frame": frames, "decision_index": steps,
                    "unspent_frames": int(max(remaining, 0)),
                    "note": ("a decision shorter than --decision-frames is not "
                             "issued, so the controlled-frame cap is never "
                             "exceeded" if not allow_partial_tail else
                             "no controlled frame left"),
                }
                break
            if clock() > deadline:
                terminal = {
                    "kind": TERMINAL_UNKNOWN, "reason": REASON_TIMEOUT,
                    "frame": frames, "decision_index": steps,
                    "timeout_s": float(timeout_s),
                }
                warnings.append(
                    f"wall-clock timeout {timeout_s}s reached at frame {frames}")
                break
            units_this = int(min(int(decision_frames), int(remaining)))

            decision_index = steps
            logical_frame = frames
            with meter.measure("actor"):
                # No shared-block read here on purpose: after the previous
                # unit's STATE/HUD/PIXELS the header no longer describes the
                # frame its bytes belong to.  The audit reads the block once per
                # STEP unit (see the loop below), and that capture feeds both
                # the digest and the actor's stack ("the picture this frame
                # drew", requirement 3 of the fix).
                hist_pushes_at_input = int(history.pushes)
                actor = tap.inputs(history)
                actor_digests = actor["digests"]
                # The frozen v4.2.0 observation digest: raw uint8 bytes of
                # play(4,112,96) -> hud(56,160) -> act_hist(8,18), once per
                # decision.  Never the normalised tensors.
                for plane in ("play", "hud", "act_hist"):
                    obs_digest.update(
                        np.ascontiguousarray(actor[plane]).tobytes())
            # Per-decision values, taken from the *last* audited unit.
            last_capture: dict[str, Any] | None = None
            unit_audit: list[dict[str, Any]] = []

            cand, source_label = actions.decision(decision_index)
            requested_mask = gym_actions.cand_to_mask(int(cand), 0)
            # One STEP per game frame (obs_window="consecutive"): a non-release
            # decision is TWO one-frame STEPs, never one two-frame STEP.  That is
            # what the frozen v4.2.0 loop does, what puts 2 x 2 bytes into
            # ``mask_digest`` per decision, and what keeps one render per frame.
            # The release decision's LAST unit carries the reduced mask; a
            # single-unit decision never releases (the frozen ``units > 1`` rule).
            release = bool(units_this > 1 and release_due(
                decision_index, shoot_pulse_period))
            segments: list[tuple[int, int]] = []
            for unit in range(int(units_this)):
                release_this = bool(release) and unit == int(units_this) - 1
                segments.extend(commit_frames(
                    int(cand), bomb=0, decision_frames=1,
                    release_shoot=release_this))
            if release:
                release_decisions.append(decision_index)
            action_counts[int(cand)] += 1
            action_digest.update(np.asarray([cand], dtype=np.uint8).tobytes())
            steps += 1

            advanced_total = 0
            segment_rows: list[list[int]] = []
            ended = False
            for unit, (step_mask, want) in enumerate(segments):
                # Task A: one *parent* measurement around the whole unit body,
                # from before ``host.step`` to the end of the unit.  The local
                # capture digest (~60 ms/unit of pure-Python FNV-1a-64 over
                # 1.2 MB) used to sit in no bucket at all -- that is the 83%
                # ``unaccounted`` of the teacher run -- and is now inside
                # ``unit_total`` and split out again as ``capture``.
                with meter.measure("unit_total"):
                    unit_number = int(units_sent)
                    reply = host.step(step_mask, want)
                    units_sent += 1
                    advanced = int(reply.get("advanced", 0))
                    advanced_total += advanced
                    frames += advanced
                    mask_digest.update(int(step_mask).to_bytes(2, "little"))
                    mask_counts[int(step_mask)] = (
                        mask_counts.get(int(step_mask), 0) + 1)
                    if int(step_mask) != int(requested_mask):
                        releases_applied += 1
                    shoot = 1 if int(step_mask) & BTN_SHOOT else 0
                    if shoot:
                        shoot_frames += advanced
                    elif advanced:
                        no_shoot_frames += advanced
                    if shoot and not previous_shoot:
                        rising_edges += 1
                    previous_shoot = shoot
                    if advanced <= 0:
                        zero_advances += 1
                    segment_rows.append(
                        [int(step_mask), int(want), int(advanced)])

                    # -- audit: step -> shm -> events -> state -> [hud] -> [pixels]
                    # :data:`AUDIT_COMMAND_SEQUENCE` is the root-cause ordering of
                    # this ticket.  The shared block is read FIRST: on rl_v3 the
                    # last writer of header->frame before STEP returns is the
                    # STEP's own PublishState (rlharness.cpp:901 -> :1362), and
                    # both STATE (:1362) and PIXELS (HandlePixels -> PublishPixels
                    # at :1190/:1372) republish as well.  The RGBA bytes are older
                    # than any of those publishes: AfterDrawFrame runs before
                    # ++g_frameCount (main.cpp:143-146), so they are the frame
                    # EVENTS.frames - 1 while the header may already read
                    # EVENTS.frames.  Read the block after STATE and the two facts
                    # can no longer be separated - that was the run-2 bug.  Never
                    # reorder this block.
                    #
                    # ``hud`` and ``pixels`` are gate-only tails
                    # (:data:`AUDIT_COMMAND_GATED`): when they are not issued the
                    # resource columns come from STATE
                    # (``hud_source="state_derived"``) and the raw-picture
                    # checksum from the local FNV of the captured bytes, and both
                    # are recorded on every unit regardless.
                    #
                    # Digest scopes (V831-SLICE-01c): the engine-comparable FNV
                    # is the ticket's **per-decision** value
                    # (:data:`CHECKSUM_SCOPE_PER_DECISION`), so it is computed on
                    # this decision's last unit -- plus, once per run, on the
                    # first unit so the gated first ``PIXELS`` still has a local
                    # counterpart.  Every unit gets ``pixel_frame_blake2b128``.
                    need_fnv_record = bool(unit == len(segments) - 1)
                    need_fnv_crosscheck = bool(unit_number == 0)
                    need_fnv = bool(need_fnv_record or need_fnv_crosscheck)
                    audit_commands: list[str] = ["step"]
                    capture_attempts: list[dict[str, Any]] = []
                    capture: dict[str, Any] | None = None
                    classification: dict[str, Any] | None = None
                    events: dict[str, Any] | None = None
                    state: dict[str, Any] | None = None
                    hud: dict[str, Any] | None = None
                    hud_crosscheck: dict[str, Any] | None = None
                    events_frames: int | None = None
                    pixel_frame: int | None = None
                    unit_checksum: dict[str, Any] | None = None
                    unit_checksum_error: str | None = None
                    # audit_stride=1 (the default) samples every unit; N>1 keeps
                    # STEP and the capture on every unit but issues EVENTS/STATE
                    # only every N-th one.
                    sampled = bool(
                        audit_stride <= 1 or unit_number % audit_stride == 0)
                    readings = dict(previous)
                    stage_advanced = False
                    try:
                        capture = capture_shared_block(
                            host, meter=meter, checksum=need_fnv)
                        if need_fnv_record:
                            fnv_units += 1
                        if need_fnv_crosscheck:
                            fnv_crosscheck_units += 1
                        blake2b_units += 1
                        audit_commands.append("shm")
                        capture_attempts.append({
                            "read": 1,
                            "header_frame": int(capture["header_frame"]),
                            "rgba_sha256": capture["rgba_sha256"],
                        })
                        if sampled:
                            raw_events = host.events_raw()
                            audit_commands.append("events")
                            events = parse_events_reply_full(raw_events)
                            events_frames = int(events["frames"])
                            classification = classify_captured_frame(
                                capture["header_frame"], events_frames)
                            if classification is None:
                                # Exactly ONE re-read, and it is still before any
                                # command that could PublishState/PublishPixels:
                                # EVENTS only reads the event word.  Anything
                                # else would be a second bug.
                                capture = capture_shared_block(
                                    host, meter=meter, checksum=need_fnv)
                                audit_commands.append("shm")
                                capture_attempts.append({
                                    "read": 2,
                                    "header_frame": int(capture["header_frame"]),
                                    "rgba_sha256": capture["rgba_sha256"],
                                })
                                classification = classify_captured_frame(
                                    capture["header_frame"], events_frames)
                                if classification is None:
                                    raise TechnicalInvalid(
                                        pixel_relation_reason(
                                            capture["header_frame"],
                                            events_frames),
                                        decision_index=decision_index,
                                        frame=frames, mask=int(step_mask))
                            pixel_frame = int(classification["pixel_frame"])
                            if pixel_frame != events_frames - 1:
                                # Unreachable by construction; kept because the
                                # recorded value, not the overloaded header, is
                                # what the row asserts: pixel_frame ==
                                # EVENTS.frames - 1.
                                raise TechnicalInvalid(
                                    f"captured pixel_frame {pixel_frame} != "
                                    f"EVENTS frames {events_frames} - 1",
                                    decision_index=decision_index, frame=frames,
                                    mask=int(step_mask))
                            state = host.state()
                            audit_commands.append("state")
                            readings = _readings(state)
                            stage_advanced = (
                                int(readings["stage"]) > stage_start)
                            gate = gated_audit_plan(
                                unit_number=unit_number, hud_period=hud_period)
                            if gate["hud"]:
                                hud = host.hud()
                                audit_commands.append("hud")
                                resources = _state_resources(state)
                                hud_crosscheck = {
                                    "unit": int(unit),
                                    "engine": {
                                        "lives": int(hud["lives"]),
                                        "bombs": int(hud["bombs"]),
                                        "score": int(hud["display_score"]),
                                        "graze": int(hud["graze"]),
                                        "power": int(hud["power"]),
                                        "game_over": int(hud["game_over"]),
                                    },
                                    "state": {
                                        "lives": int(resources["lives"]),
                                        "bombs": int(resources["bombs"]),
                                        "score": int(resources["score"]),
                                        "graze": int(resources["graze"]),
                                        "power": int(resources["player_power"]),
                                        "game_over": int(
                                            str(reply.get("game_over", "0"))
                                            == "1"),
                                    },
                                }
                                hud_crosscheck["matches"] = {
                                    key: bool(hud_crosscheck["engine"][key]
                                              == hud_crosscheck["state"][key])
                                    for key in hud_crosscheck["engine"]}
                                hud_crosscheck["lives_bombs_match"] = bool(
                                    hud_crosscheck["matches"]["lives"]
                                    and hud_crosscheck["matches"]["bombs"])
                                hud_crosschecks += 1
                                if not hud_crosscheck["lives_bombs_match"]:
                                    hud_mismatches += 1
                                    if hud_mismatches == 1:
                                        warnings.append(
                                            "HUD disagrees with STATE on "
                                            f"lives/bombs at decision "
                                            f"{decision_index} frame {frames}: "
                                            "engine "
                                            f"{hud_crosscheck['engine']} vs "
                                            f"state {hud_crosscheck['state']} "
                                            "(recorded here and in "
                                            "hud_crosscheck; STATE stays the "
                                            "recorded value, never a gate)")
                            if gate["pixels"]:
                                try:
                                    unit_checksum = host.pixels_checksum()
                                except Exception as exc:  # noqa: BLE001 - recorded
                                    unit_checksum_error = (
                                        f"{type(exc).__name__}: {exc}")
                                audit_commands.append("pixels")
                                pixels_crosschecks += 1
                    except TechnicalInvalid as exc:
                        position = dict(exc.position)
                        # The decoder cannot know where it was called from; fill
                        # the loop's position in, so "the first conflict" is a
                        # location.
                        position["decision_index"] = decision_index
                        position["frame"] = frames
                        position["mask"] = int(step_mask)
                        # The capture order, both numbers and the command
                        # sequence are part of the evidence, not a later
                        # reconstruction.
                        position.update({
                            "audit_sequence": list(audit_commands),
                            "capture_reads": len(capture_attempts),
                            "captures": list(capture_attempts),
                            "events_frames": events_frames,
                            "captured_header_frame": (
                                None if capture is None
                                else int(capture["header_frame"])),
                            "legal_header_frames": (
                                None if events_frames is None
                                else [events_frames - 1, events_frames]),
                            "pixel_frame_rule": PIXEL_FRAME_RULE,
                        })
                        first_conflict = position
                        terminal = {
                            "kind": TERMINAL_TECHNICAL_INVALID,
                            "reason": exc.reason,
                            "frame": frames, "decision_index": decision_index,
                        }
                        ended = True
                        break
                    except protocol.HarnessError as exc:
                        reason = f"engine error: {exc}"
                        first_conflict = {
                            "decision_index": decision_index, "frame": frames,
                            "mask": int(step_mask), "reason": reason,
                            "audit_sequence": list(audit_commands),
                            "capture_reads": len(capture_attempts),
                        }
                        terminal = {
                            "kind": TERMINAL_TECHNICAL_INVALID, "reason": reason,
                            "frame": frames, "decision_index": decision_index,
                        }
                        ended = True
                        break

                    audit_sampled_units += int(sampled)
                    audit_commands_sent["step"] += 1
                    for issued in audit_commands[1:]:
                        audit_commands_sent[issued] = (
                            audit_commands_sent.get(issued, 0) + 1)

                    # The local digest is the measurement; the PIXELS reply is a
                    # second, independent readback (it re-runs glReadPixels and
                    # republishes, so it may see another buffer).  A difference
                    # is a warning, never a gate.
                    engine_checksum = (unit_checksum or {}).get("checksum")
                    local_checksum = capture["pixels_checksum_local"]
                    checksum_matches: bool | None = None
                    if unit_checksum is not None:
                        checksum_matches = bool(engine_checksum == local_checksum)
                        if not checksum_matches:
                            checksum_mismatches += 1
                            if checksum_mismatches == 1:
                                warnings.append(
                                    "PIXELS checksum differs from the local "
                                    "FNV-1a-64 of the captured RGBA at decision "
                                    f"{decision_index} frame {frames}: engine "
                                    f"{engine_checksum} != local {local_checksum} "
                                    "(recorded as a fact; the PIXELS command "
                                    "re-reads the framebuffer, so this is not a "
                                    "gate)")
                    hud_source = HUD_SOURCE_STATE
                    # A gated HUD is a *second opinion*: the recorded resource
                    # columns stay STATE-derived so one column always has one
                    # source, and the engine values live in hud_crosscheck.  A
                    # disagreement warns; it never changes a value.
                    hud_measured = hud is not None
                    hud_measured_units += int(hud_measured)
                    hud_lives = int(readings["lives"])
                    hud_bombs = int(readings["bombs"])
                    hud_game_over = int(
                        str(reply.get("game_over", "0")) == "1")
                    last_capture = capture
                    unit_audit.append({
                        "unit": int(unit),
                        "pixel_frame": (None if pixel_frame is None
                                        else int(pixel_frame)),
                        "header_frame": int(capture["header_frame"]),
                        "header_frame_relation": (
                            None if classification is None
                            else str(classification["relation"])),
                        "capture_reads": len(capture_attempts),
                        "audit_sequence": list(audit_commands),
                        "pixels_checksum_local": local_checksum,
                        PIXEL_FRAME_DIGEST_FIELD: capture[
                            PIXEL_FRAME_DIGEST_FIELD],
                        "pixels_checksum_engine": engine_checksum,
                        "pixels_checksum_matches": checksum_matches,
                        "pixels_checksum_error": unit_checksum_error,
                        "hud_source": hud_source,
                        "hud_measured": hud_measured,
                        "hud_crosscheck": hud_crosscheck,
                        "state_sampled": sampled,
                        "readings_source": ("state" if sampled
                                            else "carried"),
                        "audit_gated": [name for name in ("hud", "pixels")
                                        if name in audit_commands],
                    })

                    if events is None:
                        # audit_stride>1 skipped EVENTS/STATE on this unit: the
                        # fields are recorded as None, never invented.
                        event_values: dict[str, Any] = {
                            "pixel_frame": None,
                            "pixel_frame_ok": None,
                            "header_frame_relation": None,
                            "events_frames": None,
                            "frame_event_flags": None,
                            "frame_event_names": None,
                            "stage_clear": None,
                            "stage_clear_edge": None,
                            "stage_clear_edge_frame": None,
                            "stage_clear_screen_state": None,
                            "stage_clear_seq": None,
                            "level_stage_clear_bit": None,
                            "has_clear_signal": None,
                            "player_hit": None,
                        }
                    else:
                        event_values = {
                            "pixel_frame": int(pixel_frame),
                            "pixel_frame_ok": True,
                            "header_frame_relation": str(
                                classification["relation"]),
                            "events_frames": int(events["frames"]),
                            "frame_event_flags": int(
                                events["frame_event_flags"]),
                            "frame_event_names": list(
                                events["frame_event_names"]),
                            "stage_clear": int(events["stage_clear"]),
                            "stage_clear_edge": int(events["stage_clear_edge"]),
                            "stage_clear_edge_frame": int(
                                events["stage_clear_edge_frame"]),
                            "stage_clear_screen_state": (
                                events["stage_clear_screen_state"]),
                            "stage_clear_seq": events["stage_clear_seq"],
                            "level_stage_clear_bit": int(
                                events["level_stage_clear_bit"]),
                            "has_clear_signal": bool(
                                events["has_clear_signal"]),
                            "player_hit": bool(events["player_hit"]),
                        }
                    frame_row = {
                        "frame": frames,
                        "decision_index": decision_index,
                        "unit": unit,
                        "mask": int(step_mask),
                        "requested_frames": int(want),
                        "advanced": advanced,
                        "pixel_flags": int(capture["flags"]),
                        "pixel_flags_valid": bool(capture["pixels_valid"]),
                        "header_frame": int(capture["header_frame"]),
                        "capture_reads": len(capture_attempts),
                        "audit_sequence": list(audit_commands),
                        "pixels_checksum_local": local_checksum,
                        PIXEL_FRAME_DIGEST_FIELD: capture[
                            PIXEL_FRAME_DIGEST_FIELD],
                        "pixels_checksum_engine": engine_checksum,
                        "pixels_checksum_matches": checksum_matches,
                        "state_sampled": sampled,
                        "state_stage": int(readings["stage"]),
                        "stage_advanced": bool(stage_advanced),
                        "deaths": readings["deaths"],
                        "lives_remaining": readings["lives"],
                        "bombs_remaining": readings["bombs"],
                        "score": readings["score"],
                        "player_power": (None if state is None
                                         else float(state["player_power"])),
                        "graze": None if state is None else int(state["graze"]),
                        "hud_source": hud_source,
                        "hud_measured": hud_measured,
                        "hud_crosscheck": hud_crosscheck,
                        "hud_lives": hud_lives,
                        "hud_bombs": hud_bombs,
                        "hud_game_over": hud_game_over,
                        **event_values,
                    }
                    frame_rows.append(frame_row)
                    signal_window.append(frame_row)
                    if len(signal_window) > SIGNAL_WINDOW_FRAMES:
                        signal_window.pop(0)

                    # -- clear signals (only when EVENTS was read) ------------
                    if events is not None:
                        if events["has_clear_signal"]:
                            signals["events_clear_seen"] = True
                            if signals["first_clear_event_frame"] is None:
                                signals["first_clear_event_frame"] = frames
                                signals["first_clear_event_source"] = (
                                    "stageClear" if events["stage_clear"]
                                    else "stageClearEdge"
                                    if events["stage_clear_edge"]
                                    else "frameEventFlags&0x0080")
                                note_clear("events_clear")
                        if int(events["stage_clear_edge"]) == 1 and (
                                signals["first_clear_edge_frame"] is None):
                            signals["first_clear_edge_frame"] = frames
                            signals["first_clear_edge_frame_reported"] = int(
                                events["stage_clear_edge_frame"])
                            note_clear("events_clear")
                        if int(events["level_stage_clear_bit"]) and (
                                signals["first_level_bit_frame"] is None):
                            signals["first_level_bit_frame"] = frames
                            note_clear("events_clear")
                    if stage_advanced and (
                            signals["first_stage_advance_frame"] is None):
                        signals["first_stage_advance_frame"] = frames
                        signals["stage_advanced"] = True
                        note_clear("state_stage_advance")
                    elif stage_advanced:
                        signals["stage_advanced"] = True
                    if events is not None and events["player_hit"]:
                        signals["player_hit_frames"].append(frames)

                    # -- termination (the frozen v4.2.0 order) ---------------
                    # exit -> STATE.stage -> game_over -> deaths -> short
                    # advance.  A unit that terminates anything does **not**
                    # push its frame: that is what makes ``renders == frames``
                    # on a clean run.
                    if reply.get("exit") == "1":
                        if reply.get("game_over") == "1":
                            terminal = {
                                "kind": TERMINAL_LEGAL_DEATH,
                                "reason": REASON_GAME_OVER,
                                "frame": frames,
                                "decision_index": decision_index,
                            }
                        else:
                            terminal = {
                                "kind": TERMINAL_ENVIRONMENT_EXIT,
                                "reason": REASON_ENGINE_EXIT,
                                "frame": frames,
                                "decision_index": decision_index,
                            }
                        ended = True
                    elif stage_advanced:
                        terminal = {
                            "kind": TERMINAL_STAGE_CLEAR,
                            "reason": REASON_STATE_STAGE_ADVANCED,
                            "frame": frames,
                            "decision_index": decision_index,
                            "stage_start": stage_start,
                            "stage_now": int(readings["stage"]),
                        }
                        ended = True
                    elif reply.get("game_over") == "1":
                        terminal = {
                            "kind": TERMINAL_LEGAL_DEATH,
                            "reason": REASON_GAME_OVER,
                            "frame": frames,
                            "decision_index": decision_index,
                        }
                        ended = True
                    else:
                        if readings["deaths"] > float(previous["deaths"]):
                            death = {
                                "frame": frames,
                                "decision_index": decision_index,
                                "deaths": readings["deaths"],
                                "lives_remaining": readings["lives"],
                            }
                            warnings.append(f"death recorded: {death}")
                            if readings["lives"] <= 0:
                                terminal = {
                                    "kind": TERMINAL_LEGAL_DEATH,
                                    "reason": REASON_GAME_OVER_NO_LIVES,
                                    "frame": frames,
                                    "decision_index": decision_index,
                                }
                                ended = True
                            elif stop_on_first_death:
                                terminal = {
                                    "kind": TERMINAL_LEGAL_DEATH,
                                    "reason": REASON_FIRST_DEATH,
                                    "frame": frames,
                                    "decision_index": decision_index,
                                }
                                ended = True
                            else:
                                previous = dict(readings)
                        if not ended and advanced < int(want):
                            terminal = {
                                "kind": TERMINAL_ENVIRONMENT_EXIT,
                                "reason": REASON_SHORT_ADVANCE,
                                "frame": frames,
                                "decision_index": decision_index,
                                "requested_frames": int(want),
                                "advanced": int(advanced),
                            }
                            ended = True
                    if ended:
                        break

                    # -- push the frame the actor will see next --------------
                    if advanced > 0:
                        with meter.measure("actor"):
                            # The captured RGBA, i.e. "the picture this frame
                            # drew" (requirement 3).  A later read could be a
                            # different buffer after PIXELS' re-readback.
                            tap.pump(capture["rgba"])
                        renders += 1
                    previous = dict(readings)
            # end of unit loop

            if advanced_total > 0:
                history.push(int(cand))
                history_pushes += 1
            else:
                decisions_without_effect += 1
                warnings.append(
                    f"decision {decision_index} advanced no frame; its candidate "
                    "was NOT pushed into the action history")

            last = unit_audit[-1] if unit_audit else {}
            # ``pixels_checksum_local`` is computed on every unit, so ``last``
            # always has one.  The frame relation needs EVENTS, which
            # ``--audit-stride`` may skip on the last unit: fall back to the last
            # *sampled* unit and say which unit the frame came from.
            audited = next((item for item in reversed(unit_audit)
                            if item.get("pixel_frame") is not None), None)
            frame_source = audited if audited is not None else last
            checksum_units = [item for item in unit_audit
                              if item.get("pixels_checksum_engine") is not None
                              or item.get("pixels_checksum_error") is not None]
            pixels_crosscheck = None
            if checksum_units:
                item = checksum_units[-1]
                pixels_crosscheck = {
                    "unit": int(item["unit"]),
                    "local": item["pixels_checksum_local"],
                    "engine": item["pixels_checksum_engine"],
                    "matches": item["pixels_checksum_matches"],
                    "error": item["pixels_checksum_error"],
                }
            hud_units = [item for item in unit_audit
                         if item.get("hud_measured")]
            row = {
                "schema": DECISION_SCHEMA,
                "decision_index": decision_index,
                "logical_frame": logical_frame,
                # The frame the captured RGBA belongs to (== EVENTS.frames - 1
                # of the last audited unit), never the overloaded header field.
                "pixel_frame": frame_source.get("pixel_frame"),
                "pixel_frames": [item["pixel_frame"] for item in unit_audit],
                "pixel_frame_header": frame_source.get("header_frame"),
                "pixel_frame_relation": frame_source.get("header_frame_relation"),
                "pixel_frame_unit": frame_source.get("unit"),
                "pixels_checksum_local": last.get("pixels_checksum_local"),
                PIXEL_FRAME_DIGEST_FIELD: last.get(PIXEL_FRAME_DIGEST_FIELD),
                "pixels_checksum_engine": last.get("pixels_checksum_engine"),
                "pixels_checksum_matches": last.get("pixels_checksum_matches"),
                "pixels_crosscheck": pixels_crosscheck,
                "pixels_checksum_engine_units": [
                    int(item["unit"]) for item in checksum_units],
                "pixels_nonzero": (
                    None if last_capture is None
                    else last_capture["pixels_nonzero"]),
                "pixels_size": (
                    None if last_capture is None
                    else last_capture["pixels_size"]),
                "pixels_checksum_error": last.get("pixels_checksum_error"),
                # The resource columns are STATE-derived; the gated HUD read is
                # a cross-check, never the source.
                "hud_source": HUD_SOURCE_STATE,
                "hud_measured_units": [int(item["unit"]) for item in hud_units],
                "hud_crosscheck": (
                    hud_units[-1].get("hud_crosscheck") if hud_units else None),
                "state_sampled": bool(last.get("state_sampled", True)),
                "capture_reads": sum(int(item["capture_reads"])
                                     for item in unit_audit),
                # What the last unit actually issued (an extra ``shm`` appears
                # when the single retry fired, and ``hud``/``pixels`` only on a
                # gated unit); the declared order is the fallback for a decision
                # whose audit never completed.
                "audit_sequence": list(last.get(
                    "audit_sequence", AUDIT_COMMAND_SEQUENCE)),
                "audit_gated": list(last.get("audit_gated", [])),
                "actor_input_digest": actor_digests["actor_input_digest"],
                "actor_play_sha256": actor_digests["play"],
                "actor_hud_sha256": actor_digests["hud"],
                "actor_hist_sha256": actor_digests["act_hist"],
                "actor_hist_pushes": int(hist_pushes_at_input),
                "cand": int(cand),
                "bomb": 0,
                "action_source": source_label,
                "requested_mask": int(requested_mask),
                "release": bool(release),
                "segments": segment_rows,
                "advanced_total": int(advanced_total),
                "history_pushed": bool(advanced_total > 0),
                "not_executed": bool(advanced_total == 0),
                "ended_after": bool(ended),
            }
            rows.append(row)
            if on_decision is not None:
                on_decision(row)
            if ended:
                break
    except Exception as exc:  # noqa: BLE001 - never let a run vanish
        reason = f"{type(exc).__name__}: {exc}"
        first_conflict = first_conflict or {
            "decision_index": steps, "frame": frames, "mask": None,
            "reason": reason}
        terminal = {"kind": TERMINAL_TECHNICAL_INVALID, "reason": reason,
                    "frame": frames, "decision_index": steps}
        problems.append(f"unhandled during the decision loop: {reason}")

    controlled = frames - warmup_frames
    signal_frames = {
        "events_clear": signals["first_clear_event_frame"],
        "state_stage_advance": signals["first_stage_advance_frame"],
    }
    if len(signals["order"]) >= 2:
        first_name, second_name = signals["order"][0], signals["order"][1]
        first_frame = signal_frames.get(first_name)
        second_frame = signal_frames.get(second_name)
        if first_frame is not None and second_frame is not None:
            signals["interval_frames"] = int(second_frame) - int(first_frame)
    if terminal["kind"] != "stage_clear" and signals["events_clear_seen"]:
        # The engine's own clear signal was seen but ``STATE.stage`` had not
        # advanced at the stop.  That is a real clear by the engine's judgement;
        # the missing second signal is reported, not smoothed over.
        if signals["first_stage_advance_frame"] is None:
            terminal["kind"] = TERMINAL_STAGE_CLEAR
            terminal["reason"] = REASON_EVENTS_CLEAR_ONLY
            warnings.append(
                "EVENTS reported a stage clear but STATE.stage never advanced in "
                "the observed window")
    signals["missing_second_signal"] = bool(
        (signals["first_clear_event_frame"] is None)
        != (signals["first_stage_advance_frame"] is None))
    if terminal.get("frame") is None:
        terminal["frame"] = frames
    if terminal.get("decision_index") is None:
        terminal["decision_index"] = steps

    shoot_accounted = shoot_frames + no_shoot_frames
    if checksum_mismatches:
        warnings.append(
            f"PIXELS checksum differs from the local FNV-1a-64 in "
            f"{checksum_mismatches} of {units_sent} audited units; recorded, "
            "never a gate")
    if shoot_accounted != controlled:
        problems.append(
            f"SHOOT frame accounting is incomplete: {shoot_frames} + "
            f"{no_shoot_frames} != {controlled} controlled frames")

    return {
        "rows": rows,
        "frame_rows": frame_rows,
        "frames": int(frames),
        "controlled_frames": int(controlled),
        "pixels_warmup_frames": int(warmup_frames),
        "steps": int(steps),
        "step_units": int(units_sent),
        "audit_command_sequence": list(AUDIT_COMMAND_SEQUENCE),
        "audit_command_gated": {name: dict(rule)
                                for name, rule in AUDIT_COMMAND_GATED.items()},
        "audit_stride": int(audit_stride),
        "audit_hud_period_units": int(hud_period),
        "audit_sampled_units": int(audit_sampled_units),
        "checksum_scope": CHECKSUM_SCOPE_PER_DECISION,
        "frame_digest_scope": FRAME_DIGEST_SCOPE_PER_UNIT,
        "checksum_scope_rationale": CHECKSUM_SCOPE_RATIONALE,
        "pixel_frame_digest_algorithm": PIXEL_FRAME_DIGEST_ALGORITHM,
        "fnv_units": int(fnv_units),
        "fnv_crosscheck_units": int(fnv_crosscheck_units),
        "blake2b_units": int(blake2b_units),
        "audit_commands_sent": {name: int(count)
                                for name, count in audit_commands_sent.items()},
        "hud_source": HUD_SOURCE_STATE,
        "hud_measured_units": int(hud_measured_units),
        "hud_crosschecks": int(hud_crosschecks),
        "hud_mismatches": int(hud_mismatches),
        "pixels_crosschecks": int(pixels_crosschecks),
        "pixel_frame_rule": PIXEL_FRAME_RULE,
        "pixels_checksum_mismatches": int(checksum_mismatches),
        "mask_trace_sha256": mask_digest.hexdigest(),
        "action_sequence_sha256": action_digest.hexdigest(),
        "observed_input_sha256": obs_digest.hexdigest(),
        "renders": int(renders),
        "action_counts": [int(value) for value in action_counts],
        "shoot_frames": int(shoot_frames),
        "no_shoot_frames": int(no_shoot_frames),
        "shoot_rising_edges": int(rising_edges),
        "releases_planned": int(len(release_decisions)),
        "release_decision_indices_head": release_decisions[:8],
        "releases_applied": int(releases_applied),
        "distinct_masks": int(len(mask_counts)),
        "mask_counts": {str(key): int(value) for key, value in
                        sorted(mask_counts.items())},
        "zero_advances": int(zero_advances),
        "decisions_without_effect": int(decisions_without_effect),
        "history_pushes": int(history_pushes),
        "start": start,
        "previous": previous,
        "terminal": terminal,
        "signals": signals,
        "signal_window": signal_window,
        "first_conflict": first_conflict,
        "problems": problems,
        "warnings": warnings,
    }


def prime_first_frame(
    *,
    host: EngineHost,
    tap: ActorInputTap,
    meter: CostMeter,
) -> dict[str, Any]:
    """Take the first usable readback and make it the stack's earliest frame.

    ``FLAG_PIXELS_VALID`` is polled; while it is clear one ``STEP
    cand_to_mask(0, 0) 1`` is sent (the frozen v4.2.0 warmup) and the frames it
    cost are counted in ``pixels_warmup_frames``.  The *fill* itself advances no
    game frame: the same picture is repeated by ``ObservationBufferV3.stack``.

    The reported ``header_frame`` is the raw overloaded field, **not** a
    relation-checked pixel frame: no ``EVENTS`` is read here, and the value may
    be a ``PublishState`` stamp (``FrameCount()`` after the warmup step) just as
    well as an ``AfterDrawFrame`` stamp.  The first audited unit in
    :func:`run_decisions` is what establishes the relation.
    """

    frames_used = 0
    # No FNV here: this capture only fills the stack and reports ``rgba_sha256``;
    # the per-decision checksum belongs to run_decisions' last unit.
    capture = capture_shared_block(host, meter=meter, checksum=False)
    flags = int(capture["flags"])
    attempts = 0
    while not flags & FLAG_PIXELS_VALID:
        if attempts >= 64:
            raise Refusal(
                "FLAG_PIXELS_VALID never became set after 64 warmup frames")
        reply = host.step(gym_actions.cand_to_mask(0, 0), 1)
        frames_used += int(reply.get("advanced", 0))
        attempts += 1
        capture = capture_shared_block(host, meter=meter, checksum=False)
        flags = int(capture["flags"])
    with meter.measure("actor"):
        tap.pump(capture["rgba"])
    return {
        "header_frame": int(capture["header_frame"]),
        "pixel_frame": (int(capture["header_frame"]) - 1 if attempts else None),
        "pixel_frame_basis": (
            "header_frame - 1: the warmup STEP's AfterDrawFrame ran before "
            "++g_frameCount" if attempts else
            "no warmup STEP was needed; no drawn frame is attributable here and "
            "the first audited unit establishes the relation"),
        "flags": flags,
        "pixels_valid": bool(flags & FLAG_PIXELS_VALID),
        "warmup_frames": int(frames_used),
        "warmup_attempts": int(attempts),
        "width": int(capture["width"]),
        "height": int(capture["height"]),
        "state_hash": int(capture["state_hash"]),
        "rgba_sha256": capture["rgba_sha256"],
    }


# ---------------------------------------------------------------------------
# preflight: identities, space, the private copy and the mute evidence
# ---------------------------------------------------------------------------
#: Environment guard against an ``xvfb-run`` re-exec loop.
XVFB_GUARD_ENV = "V831_SLICE_XVFB"
#: The child-process audio/GL environment this tool forces (recorded verbatim in
#: ``mute.json``; the frozen client adds its own software-GL defaults).
FORCED_ENGINE_ENV: dict[str, str] = {"SDL_AUDIODRIVER": "dummy"}


def engine_env_record() -> dict[str, Any]:
    """The env values the engine child actually receives, as observed here."""

    client_defaults = {
        "SDL_AUDIODRIVER": "dummy",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "GALLIUM_DRIVER": "llvmpipe",
        "LP_NUM_THREADS": "1",
    }
    merged = dict(client_defaults)
    merged.update(FORCED_ENGINE_ENV)
    return {
        "SDL_AUDIODRIVER": merged["SDL_AUDIODRIVER"],
        "LIBGL_ALWAYS_SOFTWARE": merged["LIBGL_ALWAYS_SOFTWARE"],
        "GALLIUM_DRIVER": merged["GALLIUM_DRIVER"],
        "LP_NUM_THREADS": merged["LP_NUM_THREADS"],
        "DISPLAY": os.environ.get("DISPLAY") or None,
        XVFB_GUARD_ENV: os.environ.get(XVFB_GUARD_ENV) or None,
        "forced_by": "tools/rl/v8_31_slice.py FORCED_ENGINE_ENV",
        "client_defaults": "touhou_rl.v400.gym.protocol.HarnessClient.__init__",
    }


def maybe_reexec_under_xvfb(*, enabled: bool) -> None:
    """The ``env`` role needs a private X server (WSLg's ``DISPLAY=:0`` hangs).

    A one-shot ``os.execvpe`` into ``xvfb-run -a <python> <this file> <argv>``,
    guarded by :data:`XVFB_GUARD_ENV` so it cannot loop, and skipped when the
    process already runs under a non-default display or ``--no-xvfb`` was given.
    """

    if not enabled:
        return
    if os.environ.get(XVFB_GUARD_ENV) == "1":
        return
    display = os.environ.get("DISPLAY", "")
    if display and display != ":0":
        return
    executable = shutil.which("xvfb-run")
    if not executable:
        raise Refusal(
            "xvfb-run is not installed, and the env role needs a private X "
            "server (WSLg's DISPLAY=:0 hangs before HELLO); pass --no-xvfb only "
            "if you know the current display is usable")
    script = str(Path(__file__).resolve())
    argv = [executable, "-a", sys.executable, script, *sys.argv[1:]]
    env = dict(os.environ)
    env[XVFB_GUARD_ENV] = "1"
    env.update(FORCED_ENGINE_ENV)
    os.execvpe(executable, argv, env)


def make_shm_name(mode: str) -> str:
    """A shared-block name unique to this process and run."""

    token = hashlib.sha1(
        f"{mode}-{os.getpid()}-{time.time_ns()}".encode()).hexdigest()[:12]
    return f"v831slice_{mode.replace('-', '')}_{token}"


def git_head() -> str | None:
    """The checkout's HEAD, or ``None`` when git cannot answer."""

    try:
        import subprocess  # noqa: PLC0415 - only needed for this one call

        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=10)
        if completed.returncode == 0:
            return completed.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None
    return None


def preflight(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """Every identity check that must pass before a frame is ever stepped."""

    engine = Path(args.engine).expanduser()
    if not engine.is_file():
        raise Refusal(f"engine binary not found: {engine}")
    engine_sha = sha256_file(engine)
    if engine_sha != REGISTERED_ENGINE_SHA256:
        raise Refusal(
            f"engine sha256 {engine_sha} != registered "
            f"{REGISTERED_ENGINE_SHA256}; this ticket runs the registered rl_v3 "
            "build only")
    obs_digest = obs_spec_v3.obs_spec_v3_digest()
    if obs_digest != REGISTERED_OBS_V3_DIGEST:
        raise Refusal(
            f"obs-v3 digest {obs_digest} != registered "
            f"{REGISTERED_OBS_V3_DIGEST}; the observation contract moved and a "
            "calibration against it would be meaningless")

    data_source = Path(args.data_source).expanduser().resolve()
    if not data_source.is_dir():
        raise Refusal(f"--data-source is not a directory: {data_source}")
    if out_dir == data_source or data_source.is_relative_to(out_dir / "game"):
        raise Refusal(
            f"--data-source {data_source} points inside the private run "
            "directory; the source must be a separate read-only input")
    source_cfg = data_source / "th08.cfg"
    if not source_cfg.is_file():
        raise Refusal(f"--data-source has no th08.cfg: {source_cfg}")
    for required in ("th08.dat", "thbgm.dat"):
        if not (data_source / required).exists():
            raise Refusal(
                f"--data-source {data_source} lacks {required}; the private copy "
                "must reproduce the whole game directory")

    source_report = decode_cfg_file(source_cfg)
    if source_report.get("error"):
        raise Refusal(
            f"source th08.cfg cannot be decoded: {source_report['error']}")

    difficulty = str(args.difficulty)
    if difficulty not in DIFFICULTIES:
        raise Refusal(f"--difficulty {difficulty!r} not in {list(DIFFICULTIES)}")
    difficulty_index = DIFFICULTIES.index(difficulty)

    if int(args.decision_frames) <= 0:
        raise Refusal(
            f"--decision-frames must be positive, got {args.decision_frames}")
    if int(args.shoot_pulse_period) <= 0:
        raise Refusal(
            f"--shoot-pulse-period must be positive, got "
            f"{args.shoot_pulse_period}: 0 would silently disable the release "
            "frame")
    if int(args.max_controlled_frames) < 0:
        raise Refusal(
            f"--max-controlled-frames must be >= 0, got "
            f"{args.max_controlled_frames}")
    if float(args.timeout_s) <= 0:
        raise Refusal(f"--timeout-s must be positive, got {args.timeout_s}")
    audit_stride = int(getattr(args, "audit_stride", 1) or 1)
    if audit_stride < 1:
        raise Refusal(
            f"--audit-stride must be >= 1, got {args.audit_stride}: 1 audits "
            "every STEP unit and is the default; N>1 trades audit sampling for "
            "wall clock and is recorded in result.json")

    space = shutil.disk_usage(str(out_dir.parent if out_dir.parent.exists()
                                  else REPO_ROOT))
    return {
        "engine": str(engine),
        "engine_sha256": engine_sha,
        "engine_sha256_registered": REGISTERED_ENGINE_SHA256,
        "engine_sha256_matches": True,
        "obs_spec_version": obs_spec_v3.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_digest,
        "obs_spec_digest_registered": REGISTERED_OBS_V3_DIGEST,
        "obs_spec_digest_matches": True,
        "data_source": str(data_source),
        "data_source_cfg": source_report,
        "source_cfg_registered_sha256": REGISTERED_SOURCE_CFG_SHA256,
        "source_cfg_matches_registered": bool(
            source_report.get("sha256") == REGISTERED_SOURCE_CFG_SHA256),
        "difficulty": difficulty,
        "difficulty_index": difficulty_index,
        "difficulty_table": list(DIFFICULTIES),
        "stage": int(args.stage),
        "shot_type": int(args.shot_type),
        "seed": int(args.seed),
        "decision_frames": int(args.decision_frames),
        "shoot_pulse_period": int(args.shoot_pulse_period),
        "max_controlled_frames": int(args.max_controlled_frames),
        "timeout_s": float(args.timeout_s),
        "audit_stride": audit_stride,
        "disk": {
            "total_bytes": int(space.total),
            "used_bytes": int(space.used),
            "free_bytes": int(space.free),
            "checked_at": now_iso(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "git_head": git_head(),
        "controller": CONTROLLER,
    }


def prepare_private_game_dir(
    *,
    out_dir: Path,
    data_source: Path,
    meter: CostMeter,
) -> dict[str, Any]:
    """Copy the whole game data directory and mute the **copy's** ``th08.cfg``.

    The source directory is only ever read.  ``shutil.copytree`` is given the
    whole tree 閳?``th08.dat`` and ``thbgm.dat`` included 閳?so the engine sees a
    complete private game directory, exactly what the ticket asks for.
    """

    target = out_dir / "game"
    if target.exists():
        raise Refusal(
            f"private game directory already exists: {target}; every run needs "
            "its own fresh .tmp_v831_slice_01_* directory")
    with meter.measure("copytree"):
        shutil.copytree(data_source, target)
    cfg = target / "th08.cfg"
    source_raw = (data_source / "th08.cfg").read_bytes()
    copied_raw = cfg.read_bytes()
    muted_raw = zero_mute_offsets(copied_raw)
    if muted_raw != copied_raw:
        cfg.write_bytes(muted_raw)
    if cfg.read_bytes() != muted_raw:
        raise Refusal(f"could not write the muted copy {cfg}")
    return {
        "source_dir": str(data_source),
        "copy_dir": str(target),
        "copied_cfg_sha256_before_mute": sha256_bytes(copied_raw),
        "copied_cfg_sha256_after_mute": sha256_bytes(muted_raw),
        "source_raw": source_raw,
        "copy_raw": muted_raw,
        "copied_entries": sorted(
            entry.name for entry in target.iterdir()),
    }


def write_mute_json(
    *,
    out_dir: Path,
    paths: Mapping[str, Path],
    source_raw: bytes,
    copy_before_raw: bytes,
    copy_after_raw: bytes | None,
    checked_at: Sequence[str],
    problems: Sequence[str],
    meter: CostMeter,
) -> tuple[Path, dict[str, Any]]:
    """Write ``<out>/mute.json`` (before start and again after the exit)."""

    report = build_mute_report(
        source_path=paths["source"], copy_path=paths["copy"],
        source_raw=source_raw, copy_before_raw=copy_before_raw,
        copy_after_raw=copy_after_raw, engine_env=engine_env_record(),
        checked_at=checked_at, problems=problems)
    with meter.measure("io"):
        target = write_json(out_dir / "mute.json", report)
    return target, report


# ---------------------------------------------------------------------------
# the two subcommands
# ---------------------------------------------------------------------------
def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"shoot": {"a": 1}}`` -> ``{"shoot.a": 1}``; leaf values only."""

    out: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flatten(value, path + "."))
        else:
            out[path] = value
    return out


def _accepted(mode: str, terminal: Mapping[str, Any], *,
              controlled_frames: int, max_controlled_frames: int,
              decision_frames: int) -> bool:
    """Does this run meet its own subcommand's acceptance rule?

    ``pixel-smoke`` accepts the full controlled-frame budget (the cap may not be
    a multiple of the decision width, so "full" means "no whole decision fits
    any more"), a legal first death, or a real stage clear.  A game over, an
    engine exit, a technical conflict or a timeout is not accepted.
    """

    kind = str(terminal.get("kind"))
    reason = str(terminal.get("reason"))
    if mode == "pixel-smoke":
        budget_spent = (
            controlled_frames >= int(decision_frames)
            and controlled_frames + int(decision_frames) > int(
                max_controlled_frames))
        if kind == TERMINAL_TRUNCATED and reason == REASON_FRAME_CAP and budget_spent:
            return True
        if kind == TERMINAL_LEGAL_DEATH and reason in LEGAL_DEATH_ACCEPTED_REASONS:
            return True
        if kind == TERMINAL_STAGE_CLEAR:
            return True
        return False
    return kind == TERMINAL_STAGE_CLEAR


def build_final_crosscheck(
    *,
    summary: Mapping[str, Any],
    hud: Mapping[str, Any],
    engine_checksum: Mapping[str, Any],
    final_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The last-unit half of :data:`AUDIT_COMMAND_GATED`, as one record.

    ``HUD`` and ``PIXELS`` also fire once **after** the last unit.  A unit cannot
    know it is the last one until its termination check has run, and nothing
    publishes or advances between that check and this read, so the two see the
    same frame the last unit captured.  The comparison is recorded here and in
    ``warnings``; it never changes a terminal kind, an exit code or a recorded
    value.
    """

    rows = list(summary.get("rows") or [])
    last_local = rows[-1].get("pixels_checksum_local") if rows else None
    hud_error = hud.get("error")
    state = None
    if final_state is not None:
        state = _state_resources(final_state)
    engine_values = None
    if hud_error is None:
        engine_values = {
            "lives": int(hud["lives"]),
            "bombs": int(hud["bombs"]),
            "score": int(hud["display_score"]),
            "graze": int(hud["graze"]),
            "power": int(hud["power"]),
            "game_over": int(hud["game_over"]),
        }
    matches_lives_bombs: bool | None = None
    if engine_values is not None and state is not None:
        matches_lives_bombs = bool(
            engine_values["lives"] == int(state["lives"])
            and engine_values["bombs"] == int(state["bombs"]))
    engine_pixels = engine_checksum.get("checksum")
    return {
        "when": ("after the last STEP unit, before any further STEP or RESET; "
                 "this is the 'last unit' trigger of AUDIT_COMMAND_GATED"),
        "hud": {
            "engine": engine_values,
            "state": state,
            "error": hud_error,
            "matches_lives_bombs": matches_lives_bombs,
            "score_power_graze_note": (
                "recorded side by side, never compared: HUD.score is the "
                "animated display counter and HUD.power is in hundredths while "
                "STATE.player_power is a float"),
        },
        "pixels": {
            "engine": engine_pixels,
            "local": last_local,
            "error": engine_checksum.get("error"),
            "matches": (None if engine_pixels is None or last_local is None
                        else bool(engine_pixels == last_local)),
            "kind": ("engine PIXELS reply vs the last recorded "
                     "pixels_checksum_local (the local FNV-1a-64 of the "
                     "captured RGBA is the value this tool reports)"),
        },
    }


def run_subcommand(args: argparse.Namespace, *, mode: str,
                   tape: Mapping[str, Any] | None = None) -> int:
    """Run one trajectory end to end and write its artefacts.

    The engine is started exactly once and closed in a ``finally``; the mute
    document is written before the first start **and** after the exit, so the
    audio claim is checked on both sides of the process.
    """

    meter = CostMeter()
    meter.begin()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    warnings: list[str] = []
    boundary = actor_boundary_problems()
    problems.extend(boundary)

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "controller": CONTROLLER,
        "mode": mode,
        "run_id": f"{mode}-{datetime.now().strftime('%Y%m%dT%H%M%S')}-"
                  f"{os.getpid()}",
        "created": now_iso(),
        "argv": list(sys.argv),
        "out_dir": str(out_dir),
        "actor_boundary": {
            "fields": list(ACTOR_FIELDS),
            "referee_fields": list(REFEREE_FIELDS),
            "problems": boundary,
            "function": "build_actor_inputs(pixels_rgba, history)",
        },
        "exit_code": None,
        "terminal_kind": None,
        "terminal_reason": None,
        "audit_stride": int(getattr(args, "audit_stride", 1) or 1),
        # Recorded verbatim so an acceptance script that passes a larger value
        # can show it: the default itself is never changed here.
        "timeout_s": float(getattr(args, "timeout_s", 0.0) or 0.0),
        "problems": problems,
        "warnings": warnings,
    }
    if tape is not None:
        result["tape"] = {key: value for key, value in tape.items()
                          if key != "candidates"}
        result["tape_sha256"] = tape.get("shard_sha256")

    host: EngineHost | None = None
    source_raw: bytes | None = None
    copy_before_raw: bytes | None = None
    copy_after_raw: bytes | None = None
    mute_paths: dict[str, Path] | None = None
    checked_at: list[str] = []
    summary: dict[str, Any] | None = None
    initial_state: dict[str, Any] | None = None
    prime: dict[str, Any] | None = None
    final_state: dict[str, Any] | None = None
    decisions_path = out_dir / "decisions.jsonl"
    counts = {
        "planned": 1, "started": 0, "effective": 0, "legal_failure": 0,
        "technical_invalid": 0, "not_started": 1, "auto_resume": 0,
        "resume": 0,
    }
    exit_code = 1
    final_crosscheck: dict[str, Any] | None = None
    try:
        with meter.measure("preflight"):
            info = preflight(args, out_dir)
        result["preflight"] = info
        if not info["source_cfg_matches_registered"]:
            warnings.append(
                "source th08.cfg sha256 "
                f"{info['data_source_cfg'].get('sha256')} != registered "
                f"{REGISTERED_SOURCE_CFG_SHA256}: recorded as a difference; the "
                "copy's own sha256 is the acceptance value")

        private = prepare_private_game_dir(
            out_dir=out_dir, data_source=Path(info["data_source"]), meter=meter)
        result["private_game_dir"] = {
            key: value for key, value in private.items()
            if key not in ("source_raw", "copy_raw")}
        source_raw = private["source_raw"]
        copy_before_raw = private["copy_raw"]
        mute_paths = {"source": Path(info["data_source"]) / "th08.cfg",
                      "copy": Path(private["copy_dir"]) / "th08.cfg"}
        checked_at.append(now_iso())
        _, mute = write_mute_json(
            out_dir=out_dir, paths=mute_paths, source_raw=source_raw,
            copy_before_raw=copy_before_raw, copy_after_raw=None,
            checked_at=checked_at, problems=[], meter=meter)
        for message in mute_report_problems(mute):
            if "after the process exited" in message:
                continue  # not yet knowable before the first start
            problems.append(message)
        if not mute["copy_before_start"]["muted_strict_four_field"]:
            raise Refusal(
                "the private th08.cfg copy is not strictly muted before start: "
                f"{mute['copy_before_start'].get('fields')}")

        # -- the engine session -------------------------------------------
        shm_name = make_shm_name(mode)
        host = EngineHost(
            engine=info["engine"],
            data_dir=private["copy_dir"],
            shm_name=shm_name,
            timeout=max(60.0, float(args.timeout_s)),
            env=dict(FORCED_ENGINE_ENV),
            meter=meter,
        )
        session = host.open(
            stage=info["stage"], difficulty_index=info["difficulty_index"],
            shot_type=info["shot_type"], seed=info["seed"])
        counts["started"] = 1
        counts["not_started"] = 0
        result["engine_session"] = session
        result["engine_env"] = engine_env_record()

        try:
            preflight_events: dict[str, Any]
            try:
                preflight_events = parse_events_reply_full(host.events_raw())
            except Exception as exc:  # noqa: BLE001 - the loop will classify it
                preflight_events = {"error": f"{type(exc).__name__}: {exc}"}
                warnings.append(
                    "the preflight EVENTS read failed; the decision loop will "
                    "report the conflict at its first frame")
            result["preflight_events"] = preflight_events
            initial_state = host.state()
            result["initial_state"] = {
                key: initial_state[key] for key in (
                    "frame", "stage", "deaths", "lives_remaining",
                    "bombs_remaining", "player_power", "score", "graze",
                    "gameplay_frame")}
            tap = ActorInputTap()
            history = ActionHistoryBuffer()
            if mode == "pixel-smoke":
                actions: Any = ConstantActions(int(args.constant_cand))
                stop_on_first_death = True
            else:
                if tape is None:
                    raise Refusal("teacher-calibration needs a tape")
                actions = TapeActions(tape)
                stop_on_first_death = False

            # The first usable readback becomes the stack's earliest frame; the
            # fill advances no game frame, so the warmup count is the only cost.
            result["first_frame"] = prime_first_frame(
                host=host, tap=tap, meter=meter)
            initial_frames = int(result["first_frame"]["warmup_frames"])

            with meter.measure("io"):
                handle = decisions_path.open("w", encoding="utf-8")

            def _write_row(row: dict[str, Any]) -> None:
                # The decision row's JSONL write is an ``io`` cost; it happens
                # outside every ``unit_total``, so it is a leaf and cannot be
                # double counted with the unit body.
                with meter.measure("io"):
                    write_jsonl_row(handle, row)

            try:
                summary = run_decisions(
                    host=host, actions=actions, tap=tap, history=history,
                    meter=meter, decision_frames=int(args.decision_frames),
                    shoot_pulse_period=int(args.shoot_pulse_period),
                    max_controlled_frames=int(args.max_controlled_frames),
                    start_frames=initial_frames, initial_state=initial_state,
                    timeout_s=float(args.timeout_s),
                    stop_on_first_death=stop_on_first_death,
                    allow_partial_tail=bool(
                        getattr(args, "allow_partial_tail", False)),
                    start_renders=int(tap.frames_pumped),
                    audit_stride=int(getattr(args, "audit_stride", 1) or 1),
                    on_decision=_write_row,
                )
            finally:
                with meter.measure("io"):
                    handle.close()
            with meter.measure("io"):
                result["decisions_jsonl"] = {
                    "path": str(decisions_path),
                    "rows": len(summary["rows"]),
                    "sha256": (sha256_file(decisions_path)
                               if decisions_path.is_file() else None),
                    "bytes": (decisions_path.stat().st_size
                              if decisions_path.is_file() else None),
                }
            try:
                final_state = host.state()
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"final STATE read failed: {type(exc).__name__}: {exc}")
            try:
                final_hud = host.hud()
            except Exception as exc:  # noqa: BLE001
                final_hud = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                last_checksum = host.pixels_checksum()
            except Exception as exc:  # noqa: BLE001
                last_checksum = {"error": f"{type(exc).__name__}: {exc}"}
            result["last_pixels_checksum"] = last_checksum
            result["last_pixels_checksum_kind"] = (
                "engine PIXELS reply (a second readback; the recorded "
                "pixels_checksum_local is the local FNV-1a-64 of the captured "
                "RGBA and is the value this tool reports)")
            # -- the last-unit half of AUDIT_COMMAND_GATED --------------------
            # ``hud`` and ``pixels`` also fire once here: the last unit cannot
            # know it is the last one until its termination check has run, and
            # nothing publishes or advances in between, so this is the same
            # frame the last unit captured.  Differences are warnings, never
            # gates.
            final_crosscheck = build_final_crosscheck(
                summary=summary, hud=final_hud, engine_checksum=last_checksum,
                final_state=final_state)
            result["audit_final_crosscheck"] = final_crosscheck
            if final_crosscheck["hud"]["matches_lives_bombs"] is False:
                warnings.append(
                    "the last-unit HUD cross-check disagrees with the final "
                    f"STATE on lives/bombs: engine {final_crosscheck['hud']['engine']} "
                    f"vs state {final_crosscheck['hud']['state']} (recorded; "
                    "STATE stays the recorded value, not a gate)")
            if final_crosscheck["pixels"]["matches"] is False:
                warnings.append(
                    "the last-unit PIXELS checksum differs from the last "
                    f"recorded pixels_checksum_local: engine "
                    f"{final_crosscheck['pixels']['engine']} != local "
                    f"{final_crosscheck['pixels']['local']} (recorded; the "
                    "local FNV of the captured bytes stays the value this tool "
                    "reports, not a gate)")
        finally:
            with meter.measure("session_close"):
                host.close()
                host = None
            copy_after_raw = mute_paths["copy"].read_bytes() if (
                mute_paths is not None and mute_paths["copy"].is_file()) else None
            checked_at.append(now_iso())

        if source_raw is not None and copy_before_raw is not None:
            _, mute = write_mute_json(
                out_dir=out_dir, paths=mute_paths, source_raw=source_raw,
                copy_before_raw=copy_before_raw, copy_after_raw=copy_after_raw,
                checked_at=checked_at, problems=[], meter=meter)
            result["mute"] = {
                "path": str(out_dir / "mute.json"),
                "source_sha256": mute["source"]["sha256"],
                "source_matches_registered": mute["source"][
                    "matches_registered_source_sha256"],
                "copy_before": {
                    "sha256": mute["copy_before_start"]["sha256"],
                    "fields": mute["copy_before_start"]["fields"],
                    "muted_strict_four_field": mute["copy_before_start"][
                        "muted_strict_four_field"],
                    "diff_byte_offsets": mute["copy_before_start"][
                        "diff_byte_offsets"],
                    "other_bytes_identical": mute["copy_before_start"][
                        "other_bytes_identical"],
                    "matches_expected_muted_sha256": mute["copy_before_start"][
                        "matches_expected_muted_sha256"],
                },
                "copy_after_exit": (
                    None if mute["copy_after_exit"] is None else {
                        "sha256": mute["copy_after_exit"]["sha256"],
                        "fields": mute["copy_after_exit"]["fields"],
                        "muted_strict_four_field": mute["copy_after_exit"][
                            "muted_strict_four_field"],
                        "other_bytes_identical": mute["copy_after_exit"][
                            "other_bytes_identical"],
                        "matches_expected_muted_sha256": mute["copy_after_exit"][
                            "matches_expected_muted_sha256"],
                    }),
                "engine_env": mute["engine_env"],
            }
            for message in mute_report_problems(mute):
                problems.append(message)
            after = mute.get("copy_after_exit") or {}
            if after and not after.get("muted_strict_four_field"):
                warnings.append(
                    "the private th08.cfg came back NOT muted after the engine "
                    "exited; recorded as a fact (not patched) in mute.json")
    except Refusal as exc:
        problems.append(f"refused: {exc}")
        exit_code = 2
    except Exception as exc:  # noqa: BLE001 - a failed run is a recorded run
        problems.append(f"{type(exc).__name__}: {exc}")
        exit_code = 1
    finally:
        if host is not None:
            try:
                with meter.measure("session_close"):
                    host.close()
            except Exception:  # noqa: BLE001
                pass

    # -- assemble the record ---------------------------------------------
    if summary is not None:
        terminal = summary["terminal"]
        result.update({
            "terminal_kind": terminal["kind"],
            "terminal_reason": terminal["reason"],
            "terminal": terminal,
            "frames": summary["frames"],
            "controlled_frames": summary["controlled_frames"],
            "pixels_warmup_frames": summary["pixels_warmup_frames"],
            "decisions": summary["steps"],
            "step_units": summary["step_units"],
            "shoot": {
                "pulse_period": int(args.shoot_pulse_period),
                "release_rule": (
                    "release_due(i, period): the decision's last STEP frame drops "
                    "BTN_SHOOT|BTN_BOMB"),
                "releases_planned": summary["releases_planned"],
                "releases_applied": summary["releases_applied"],
                "release_decision_indices_head": summary[
                    "release_decision_indices_head"],
                "shoot_frames": summary["shoot_frames"],
                "no_shoot_frames": summary["no_shoot_frames"],
                "shoot_rising_edges": summary["shoot_rising_edges"],
                "distinct_masks": summary["distinct_masks"],
                "mask_counts": summary["mask_counts"],
            },
            "renders": summary["renders"],
            "mask_trace_sha256": summary["mask_trace_sha256"],
            "action_sequence_sha256": summary["action_sequence_sha256"],
            "observed_input_sha256": summary["observed_input_sha256"],
            "obs_window": "consecutive",
            "harness_role": (result.get("engine_session") or {}).get(
                "hello", {}).get("role"),
            "action_counts": summary["action_counts"],
            "zero_advances": summary["zero_advances"],
            "decisions_without_effect": summary["decisions_without_effect"],
            "history_pushes": summary["history_pushes"],
            "start_readings": summary["start"],
            "end_readings": summary["previous"],
            "signals": summary["signals"],
            "signal_window": summary["signal_window"],
            "first_conflict": summary["first_conflict"],
            "audit": {
                "command_sequence": summary["audit_command_sequence"],
                "command_gated": summary["audit_command_gated"],
                "stride": summary["audit_stride"],
                "hud_period_units": summary["audit_hud_period_units"],
                "sampled_units": summary["audit_sampled_units"],
                "commands_sent": summary["audit_commands_sent"],
                "checksum_scope": summary["checksum_scope"],
                "frame_digest_scope": summary["frame_digest_scope"],
                "checksum_scope_rationale": summary[
                    "checksum_scope_rationale"],
                "pixel_frame_digest_algorithm": summary[
                    "pixel_frame_digest_algorithm"],
                "fnv_units": summary["fnv_units"],
                "fnv_crosscheck_units": summary["fnv_crosscheck_units"],
                "blake2b_units": summary["blake2b_units"],
                "pixel_frame_rule": summary["pixel_frame_rule"],
                "pixels_checksum_mismatches": summary[
                    "pixels_checksum_mismatches"],
                "pixels_crosschecks": summary["pixels_crosschecks"],
                "hud_source": summary["hud_source"],
                "hud_measured_units": summary["hud_measured_units"],
                "hud_crosschecks": summary["hud_crosschecks"],
                "hud_mismatches": summary["hud_mismatches"],
                "final_crosscheck": final_crosscheck,
                "capture_reads": (summary["rows"][-1].get("capture_reads")
                                  if summary["rows"] else 0),
                "note": (
                    "step -> shm -> events -> state per STEP unit"
                    " (+ hud on the first unit, every hud_period_units units and "
                    "once after the last unit; + pixels on the first unit and "
                    "once after the last unit).  The shared block is captured "
                    "before any command that can PublishState/PublishPixels, and "
                    "one re-read is allowed while no such command has been sent.  "
                    "hud_source=state_derived means the resource columns came from "
                    "STATE and the STEP reply; pixels_checksum_local is the local "
                    "FNV-1a-64 of the captured RGBA and is recorded on every unit"),
            },
            "decisions_head": summary["rows"][:2],
            "decisions_tail": summary["rows"][-2:],
        })
        problems.extend(summary["problems"])
        warnings.extend(summary["warnings"])
        result["problems"] = problems
        result["warnings"] = warnings
        if final_state is not None:
            result["final_state"] = {
                key: final_state[key] for key in (
                    "frame", "stage", "deaths", "lives_remaining",
                    "bombs_remaining", "player_power", "score", "graze",
                    "gameplay_frame")}
        if summary["rows"]:
            result["first_pixels_checksum_local"] = summary["rows"][0].get(
                "pixels_checksum_local")
            result["last_decision_pixels_checksum_local"] = summary["rows"][
                -1].get("pixels_checksum_local")
        counts["effective"] = 0 if terminal["kind"] == TERMINAL_TECHNICAL_INVALID else (
            1 if counts["started"] else 0)
        counts["legal_failure"] = (1 if terminal["kind"] == TERMINAL_LEGAL_DEATH else 0)
        counts["technical_invalid"] = (
            1 if terminal["kind"] == TERMINAL_TECHNICAL_INVALID else 0)
        if mode == "teacher-calibration":
            observed = {
                "stage_clear": terminal["kind"] == TERMINAL_STAGE_CLEAR,
                "frames": summary["frames"],
                "score": (final_state or summary["previous"])["score"],
                "decisions": summary["steps"],
                "state_frame": ((final_state or {}).get("frame")
                                if final_state else None),
                "gameplay_frame": ((final_state or {}).get("gameplay_frame")
                                   if final_state else None),
                "lives_end": (final_state or summary["previous"])[
                    "lives_remaining"],
                "deaths": (final_state or summary["previous"])["deaths"],
                "renders": summary["renders"],
                "pixels_warmup_frames": summary["pixels_warmup_frames"],
                "shoot": {
                    "shoot_frames": summary["shoot_frames"],
                    "no_shoot_frames": summary["no_shoot_frames"],
                    "releases_planned": summary["releases_planned"],
                    "releases_applied": summary["releases_applied"],
                    "shoot_rising_edges": summary["shoot_rising_edges"],
                    "distinct_masks": summary["distinct_masks"],
                    "mask_trace_sha256": summary["mask_trace_sha256"],
                },
                "action_sequence_sha256": summary["action_sequence_sha256"],
                "observed_input_sha256": summary["observed_input_sha256"],
                "harness_role": (result.get("engine_session") or {}).get(
                    "hello", {}).get("role"),
                "obs_window": "consecutive",
            }
            expected = dict(EXPECTED_TEACHER_CALIBRATION)
            result["expected"] = expected
            result["observed_for_expected"] = observed
            if terminal["kind"] == TERMINAL_STAGE_CLEAR:
                flat_expected = _flatten(expected)
                flat_observed = _flatten(observed)
                result["match"] = {
                    key: bool(flat_observed.get(key) == value)
                    for key, value in flat_expected.items()}
            else:
                result["match"] = {key: False
                                   for key in _flatten(expected)}
                result["not_compared_reason"] = (
                    "run did not reach stage_clear")
            result["expected_is_not_a_gate"] = (
                "the only exit-0 condition of teacher-calibration is a real stage "
                "clear; this block reports equivalence, it does not gate")
            mismatched = [key for key, ok in result["match"].items() if not ok]
            if mismatched:
                warnings.append(
                    "teacher-calibration differs from the frozen v4.2.0 values: "
                    f"{mismatched} (reported as measured, never patched)")
        accepted = _accepted(
            mode, terminal, controlled_frames=summary["controlled_frames"],
            max_controlled_frames=int(args.max_controlled_frames),
            decision_frames=int(args.decision_frames))
        if summary["first_conflict"] is not None:
            accepted = False
        exit_code = 0 if accepted else 1
        result["accepted"] = bool(accepted)
        result["acceptance_rule"] = (
            "pixel-smoke: the full controlled-frame budget, a legal first death "
            "or a real stage clear; teacher-calibration: a real stage clear"
            if mode == "pixel-smoke" else
            "teacher-calibration: a real stage clear (EVENTS clear or STATE.stage "
            "advance)")
    else:
        result.setdefault("accepted", False)
        result["terminal_kind"] = None
        result["terminal_reason"] = (
            "refused-before-start" if exit_code == 2 else "no-run")
        result["acceptance_rule"] = "a run that never started accepts nothing"
        counts["effective"] = 0
        if exit_code == 0:
            exit_code = 1

    result["counts"] = counts
    result["exit_code"] = exit_code
    result["problems"] = list(dict.fromkeys(problems))
    result["warnings"] = list(dict.fromkeys(warnings))
    result["finished"] = now_iso()
    meter.finish()
    result["cost"] = meter.payload(
        step_units=(None if summary is None else summary["step_units"]))
    with meter.measure("io"):
        result["result_path"] = str(write_json(out_dir / "result.json", result))
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    """The CLI the ticket's acceptance commands call verbatim."""

    parser = argparse.ArgumentParser(
        prog="v8_31_slice.py",
        description=(
            "V831-SLICE-01: the pixel-wiring trajectory and the teacher "
            "stage-clear calibration of the v8.31.0 short slice."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Both subcommands copy --data-source into <out>/game, mute only the "
            "copy's th08.cfg (0x1F/0x20/0x27/0x28), set SDL_AUDIODRIVER=dummy "
            "and re-exec under a private xvfb-run unless --no-xvfb is given.\n"
            "Exit codes: 0 acceptance met, 1 run not accepted, 2 refused before "
            "start."))
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--engine", required=True,
                         help="path to the rl_v3 th08-modern harness binary")
        sub.add_argument("--data-source", required=True, dest="data_source",
                         help="read-only game data directory to copy")
        sub.add_argument("--out", required=True,
                         help="fresh run directory (result.json, mute.json, "
                              "decisions.jsonl, game/)")
        sub.add_argument("--stage", type=int, default=0,
                         help="RESET stage argument (0-based; default 0)")
        sub.add_argument("--difficulty", default="normal",
                         help=f"one of {list(DIFFICULTIES)} (default normal)")
        sub.add_argument("--shot-type", type=int, default=0, dest="shot_type",
                         help="RESET team/shotType argument (default 0)")
        sub.add_argument("--seed", type=int, required=True,
                         help="RESET seed")
        sub.add_argument("--decision-frames", type=int, default=DECISION_FRAMES,
                         dest="decision_frames",
                         help=f"game frames per decision (default "
                              f"{DECISION_FRAMES})")
        sub.add_argument("--shoot-pulse-period", type=int,
                         default=SHOOT_PULSE_PERIOD, dest="shoot_pulse_period",
                         help="decisions between SHOOT release frames (default "
                              f"{SHOOT_PULSE_PERIOD}; 0 is refused)")
        sub.add_argument("--max-controlled-frames", type=int,
                         required=True, dest="max_controlled_frames",
                         help="cap on decision-driven game frames")
        sub.add_argument("--timeout-s", type=float, default=None,
                         dest="timeout_s",
                         help="wall-clock deadline for the decision loop")
        sub.add_argument("--no-xvfb", action="store_true", dest="no_xvfb",
                         help="do not re-exec under xvfb-run (debug only)")
        sub.add_argument("--allow-partial-tail", action="store_true",
                         dest="allow_partial_tail",
                         help=("issue the remaining < --decision-frames units as "
                               "a partial decision at the controlled-frame cap "
                               "(the frozen v4.2.0 tail); off by default so the "
                               "cap rounds down to whole decisions"))
        sub.add_argument("--audit-stride", type=int, default=1,
                         dest="audit_stride",
                         help=("audit sampling stride over STEP units (default "
                               "1 = every unit).  N>1 issues EVENTS/STATE only "
                               "on every N-th unit; STEP and the shared-block "
                               "capture still happen on every unit, so STEP "
                               "order/masks/frames and the three digests are "
                               "unchanged.  A skipped unit carries the previous "
                               "STATE readings forward and is recorded as "
                               "state_sampled=false"))

    pixel = subparsers.add_parser(
        "pixel-smoke",
        help="one natural-start trajectory with a constant candidate",
        description=("Natural start (stage/difficulty/shotType/seed as given), a "
                     "constant candidate every decision, 2 game frames per "
                     "decision and one SHOOT release frame every "
                     "--shoot-pulse-period decisions.  Acceptance: the full "
                     "controlled-frame budget, a legal first death, or a real "
                     "stage clear."))
    common(pixel)
    pixel.add_argument("--constant-cand", type=int, required=True,
                       dest="constant_cand",
                       help=f"the fixed candidate, 0..{gym_actions.N_CAND - 1}")

    teacher = subparsers.add_parser(
        "teacher-calibration",
        help="the recorded teacher tape replayed for a real stage clear",
        description=("The same loop and pulse driven by the recorded teacher tape "
                     "through the frozen loader.  Acceptance: a real stage clear "
                     "reported by EVENTS and/or STATE.stage."))
    common(teacher)
    teacher.add_argument("--tape", required=True,
                         help="the registered slots.jsonl teacher tape")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, apply the private-display rule, dispatch."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timeout_s is None:
        # Unchanged defaults (120 / 600).  The parent agent's acceptance script
        # passes its own --timeout-s and reports the measured wall clock
        # separately from "does it fit 600 s"; any positive number is accepted
        # and recorded in result.json (result.timeout_s, result.preflight).
        args.timeout_s = DEFAULT_TIMEOUT_S[args.command]
    try:
        maybe_reexec_under_xvfb(enabled=not bool(args.no_xvfb))
    except Refusal as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    if args.command == "pixel-smoke":
        return run_subcommand(args, mode="pixel-smoke")
    tape = None
    try:
        tape = load_teacher_tape(args.tape)
    except Refusal as exc:
        out_dir = Path(args.out).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": RESULT_SCHEMA, "controller": CONTROLLER,
            "mode": "teacher-calibration", "created": now_iso(),
            "argv": list(sys.argv), "exit_code": 2, "accepted": False,
            "terminal_kind": None, "terminal_reason": "refused-before-start",
            "problems": [f"refused: {exc}"], "warnings": [],
            "counts": {"planned": 1, "started": 0, "effective": 0,
                       "legal_failure": 0, "technical_invalid": 0,
                       "not_started": 1, "auto_resume": 0, "resume": 0},
        }
        write_json(out_dir / "result.json", payload)
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return run_subcommand(args, mode="teacher-calibration", tape=tape)


if __name__ == "__main__":
    raise SystemExit(main())

