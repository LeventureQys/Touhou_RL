"""Teacher **observation** capture: the D0 bridge between P1 labels and P3 samples.

EXP-P3-001 §3 reported the P3 hard blocker in one sentence: the teacher JSONL has
1500 decisions and **zero** observation references, so
:func:`touhou_rl.v400.student.dataset.load_teacher_samples` fails closed and the
D0 dataset (``(obs, TeacherLabel)``) does not exist.  This module writes the
missing half: for every decision it stores

* ``play64`` — the obs-v1 4-frame stack (``t-3 .. t``, oldest first, uint8), and
* ``hud`` — the obs-v1 56x160 green HUD strip of the newest frame,

next to the teacher's own label (``scores[18]``/``verified_mask``/``bomb``/
``chosen_plan``/``margin``), in a shard that
:mod:`touhou_rl.v400.student.dataset` can load in its ``teacher`` mode.

**Alignment (frozen here, written into ``shard.json`` and re-checked on load).**

1. *One CLI decision commits* ``replan_decisions x DECISION_FRAMES = 4`` logic
   frames, i.e. two student decisions of 2 frames each.  ``scores[18]`` ranks the
   **first** student decision — the plan's slot 0, applied to frames
   ``d .. d+DECISION_FRAMES-1`` where ``d`` is the decision frame.  The second
   committed slot is the plan's continuation and is *not* a label of its own.
2. *The observation is decision-time, before the commit* ("pre-action"), which is
   what ``obs-spec-v1.md`` §3 requires: "``obs`` 必须是教师被查询的那个状态，不是
   教师执行后到达的状态".  The four stored render frames are therefore
   ``d-4 .. d-1``: after ``STEP`` reaches frame ``d`` the shared-memory block holds
   the render of frame ``d-1`` (``AfterDrawFrame`` publishes *before*
   ``++g_frameCount``, ``main.cpp:140-147``).
3. *The two conventions are the same window.*  P3's loader stacks
   ``frames[i-3..i]`` and labels it with ``mask[i+1]``; with ``i = d-1`` the label
   frame is ``i+1 = d``.  :func:`decision_window` makes that identity explicit and
   :func:`touhou_rl.v400.student.dataset.load_teacher_shard` re-checks it, so a
   shard recorded "after the decision" (newest render ``d+3``, label one to two
   student decisions stale, action effect leaked into its own input) is refused
   rather than silently trained on.
4. *Episode start* pads with the earliest available render (obs-spec-v1 §2: repeat
   the first frame until the stack is full).  After ``RESET`` the last setup frame
   is the only render there is, so decision 0 stores it four times with
   ``pad=3``; every later decision must have four *distinct, consecutive* renders
   ``d-4..d-1`` or the shard fails closed.

**Fail-closed.**  :class:`ObservationError` is raised — and the shard is left
unfinalised (a hidden ``.<key>.partial`` directory with an ``incomplete.json``,
never a loadable ``<key>``) — when the frozen obs-spec digest does not match the
live module, when a shared-memory frame cannot be read (no pixel flag, bad magic,
wrong size), when the header frame is not ``decided_frame - 1``, or when the
engine's frame counter disagrees with the decision index (``d != 4*k``).
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from touhou_rl.v400.student import obs_spec

#: Label-row schema written into ``labels.jsonl`` (one row per decision).
TEACHER_OBS_SCHEMA = "v400-teacher-obs-1"
#: Shard contract: ``<out>/shards/<key>/shard.json``.
SHARD_SCHEMA = "v400-teacher-obs-shard-1"
#: Top level ``<out>/manifest.json``.
MANIFEST_SCHEMA = "v400-teacher-obs-manifest-1"

#: ``obs_spec_digest()`` of the frozen obs-v1 spec (``obs-spec-v1.md`` line 5).
#: Kept as a literal *on purpose*: if ``touhou_rl/v400/student/obs_spec.py`` ever
#: changes meaning, this module and the loader must refuse the shard instead of
#: writing observations under a spec nobody agreed to.
FROZEN_OBS_SPEC_DIGEST = "24f0ddace59104938dbe991c0d8efbb6a8cd21dfd5a29ebcb6d023b5b364a381"

#: The one supported observation instant.  A loader that sees anything else
#: refuses the shard (see the module docstring, point 2).
OBSERVATION_TIME = "decision_time_before_commit"

#: Everything the alignment contract asserts, verbatim, into ``shard.json``.
ALIGNMENT: dict[str, Any] = {
    "observation_time": OBSERVATION_TIME,
    "observation_time_note": (
        "the 4 most recent rendered frames at the instant the teacher is queried, "
        "before the decision's STEP; newest render = decision_frame - 1 because "
        "AfterDrawFrame publishes before ++g_frameCount"
    ),
    "label_time": (
        "scores[18] ranks the plan's first student decision (slot 0), applied to "
        "frames [decision_frame, decision_frame + decision_frames)"
    ),
    "label_is_first_slot_not_continuation": True,
    "stack_order": obs_spec.STACK_ORDER,
    "stack": obs_spec.FRAME_STACK,
    "hud": "green channel of the score/lives/bombs bands of the newest render (t)",
    "hud_frames": obs_spec.HUD_FRAMES,
    "decision_frames": obs_spec.DECISION_FRAMES,
    "teacher_duration_units": obs_spec.TEACHER_DURATION_UNITS,
    "p3_window": (
        "frames[i-3..i] -> mask[i+1] with i = decision_frame - 1; the stored label "
        "is the action applied after i, so a teacher sample IS a P3 window"
    ),
    "p3_window_last_row": "i = decision_frame - 1",
    "p3_label_frame": "i + 1 = decision_frame",
    "episode_start_padding": (
        "repeat the earliest available render until the stack is full "
        "(obs-spec-v1 §2); decision 0 stores the last RESET setup render 4x with pad=3"
    ),
    "frame_unit": "engine logic frames (FrameCount, reset to 0 by RESET)",
}


class ObservationError(RuntimeError):
    """An observation cannot be captured or written as obs-v1; never guess."""


def frame_content(play: np.ndarray, hud: np.ndarray) -> dict[str, float]:
    """Cheap content statistics of one encoded observation (evidence, not policy)."""

    play = np.asarray(play)
    hud = np.asarray(hud)
    return {
        "play_mean": float(play.mean()),
        "play_std": float(play.std()),
        "play_max": int(play.max()),
        "hud_mean": float(hud.mean()),
        "hud_std": float(hud.std()),
        "hud_max": int(hud.max()),
        "nonzero": int(np.count_nonzero(play)) + int(np.count_nonzero(hud)),
    }


def require_rendered(play: np.ndarray, hud: np.ndarray, *, where: str) -> dict[str, float]:
    """Fail closed when the readback contains no rendered pixel at all.

    Measured capability, not a guess: the teacher runs the ``shadow`` role, and
    the Linux D3D8 layer gives a shadow a **CPU-only null device** that never
    touches SDL or GL (``d3d8_compat.cpp: "Shadow owns a CPU-only D3D slice: do
    not touch SDL or any GL entry point"``, ``main.cpp: "Shadow uses the software
    D3D null device and never creates SDL/GL"``).  ``AfterDrawFrame`` reads the
    render with ``glReadPixels``, so in a shadow process there is no render to
    read: the block is published with an all-zero buffer.  Recording that would
    be a fabricated observation, so the shard stops instead.
    """

    stats = frame_content(play, hud)
    if stats["nonzero"] == 0:
        raise ObservationError(
            f"{where}: the shared-memory frame is entirely zero — the engine did not "
            "render, so there is no observation to store. The teacher runs the shadow "
            "role, whose D3D8 device is the CPU-only null device (no SDL/GL), and "
            "glReadPixels therefore has no frame to read; this needs a rendering "
            "source (env role under a display, paired with the shadow by replay) "
            "before D0 observations can be produced. Refusing to write blank "
            "observations")
    return stats


# ---------------------------------------------------------------------------
# frozen-spec gate + the decision/window arithmetic
# ---------------------------------------------------------------------------
def check_frozen_spec(version: str | None = None, digest: str | None = None) -> str:
    """Refuse to record unless the live encoder is the frozen obs-v1 one.

    ``version``/``digest`` default to the live module's values; passing a stored
    pair (a checkpoint or an existing shard) checks *that* against the freeze.
    """

    live_version = obs_spec.OBS_SPEC_VERSION if version is None else version
    live_digest = obs_spec.obs_spec_digest() if digest is None else digest
    if live_version != obs_spec.OBS_SPEC_VERSION:
        raise ObservationError(
            f"observation spec version {live_version!r} is not "
            f"{obs_spec.OBS_SPEC_VERSION!r}; refusing to record observations under "
            "a spec the P3 loader does not know")
    try:
        obs_spec.check_obs_spec(live_version, live_digest)
    except obs_spec.ObsSpecError as error:  # ObsSpecMismatch is a ValueError
        raise ObservationError(f"frozen obs-spec check failed: {error}") from error
    if obs_spec.obs_spec_digest() != FROZEN_OBS_SPEC_DIGEST:
        raise ObservationError(
            "obs_spec_digest() drifted from the frozen obs-v1 digest "
            f"{FROZEN_OBS_SPEC_DIGEST}: a spec change needs obs-v2 + a new "
            "document, not a silent re-record"
        )
    return FROZEN_OBS_SPEC_DIGEST


def decision_window(decision_frame: int, *, stack: int = obs_spec.FRAME_STACK) -> dict[str, Any]:
    """The P3 window a teacher decision at ``decision_frame`` corresponds to.

    Returns the render frames that must be stored (oldest first), the P3 window
    index ``i`` and the frame the label action is applied to.  The last two are
    asserted to be ``decision_frame - 1`` and ``decision_frame`` — i.e. exactly
    ``frames[i-3..i] -> mask[i+1]`` — so an off-by-one becomes an exception here
    instead of a training set with the action's own effect in its input.
    """

    frame = int(decision_frame)
    if frame < 0:
        raise ObservationError(f"decision frame must be >= 0, got {frame}")
    newest = frame - 1
    frames = [newest - stack + 1 + offset for offset in range(stack)]
    if frames[-1] != newest or frames[0] != newest - stack + 1:
        raise ObservationError(f"bad render window for decision frame {frame}")
    p3_i = newest
    p3_label = p3_i + 1
    if p3_label != frame:
        raise ObservationError(
            f"P3 mapping broken: frames[{p3_i}-3..{p3_i}] -> mask[{p3_label}] but the "
            f"decision frame is {frame}")
    return {
        "decision_frame": frame,
        "render_frames": frames,
        "newest_render_frame": newest,
        "p3_window_last_row": p3_i,
        "p3_label_frame": p3_label,
        "label_frame": frame,
        "label_frames": [frame + offset for offset in range(obs_spec.DECISION_FRAMES)],
    }


# ---------------------------------------------------------------------------
# streaming .npy writer (fixed header, shape patched at finalise)
# ---------------------------------------------------------------------------
_NPY_MAGIC = b"\x93NUMPY\x01\x00"
#: Fixed header length so the data offset never moves while rows are appended.
_HEADER_LEN = 512


def _npy_header(shape: Sequence[int], dtype, header_len: int = _HEADER_LEN) -> bytes:
    descr = np.lib.format.dtype_to_descr(np.dtype(dtype))
    text = ("{'descr': %r, 'fortran_order': False, 'shape': %s, }"
            % (descr, tuple(int(value) for value in shape)))
    raw = text.encode("latin1")
    pad = header_len - len(raw) - 1
    if pad < 0:
        raise ObservationError(f"npy header needs {len(raw) + 1} B > {header_len} B")
    return _NPY_MAGIC + struct.pack("<H", header_len) + raw + b" " * pad + b"\n"


class NpyAppendWriter:
    """Append rows to a ``.npy`` file without knowing the row count up front.

    The header is written once with shape ``(0, *tail)`` (fixed 512-byte length)
    and patched at :meth:`finalize`; the data offset is unchanged, so the result
    is a plain ``.npy`` that ``np.load(..., mmap_mode="r")`` reads.  A writer that
    is aborted keeps the ``(0, *tail)`` header, i.e. the file *declares zero rows*
    and cannot be mistaken for a complete plane.
    """

    def __init__(self, path: str | Path, shape_tail: Sequence[int], dtype) -> None:
        self.path = Path(path)
        self.shape_tail = tuple(int(value) for value in shape_tail)
        self.dtype = np.dtype(dtype)
        self.count = 0
        self.finalized = False
        self.closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "wb")
        self._handle.write(_npy_header((0, *self.shape_tail), self.dtype))
        self.data_offset = len(_NPY_MAGIC) + 2 + _HEADER_LEN

    # -- writing ----------------------------------------------------------
    def append(self, array: np.ndarray) -> None:
        if self.closed:
            raise ObservationError(f"{self.path}: append after close")
        array = np.ascontiguousarray(array)
        if array.dtype != self.dtype:
            raise ObservationError(f"{self.path}: dtype {array.dtype} != {self.dtype}")
        if array.shape != self.shape_tail:
            raise ObservationError(
                f"{self.path}: row shape {array.shape} != {self.shape_tail}")
        self._handle.write(array.tobytes(order="C"))
        self.count += 1

    def finalize(self) -> dict[str, Any]:
        if self.closed:
            raise ObservationError(f"{self.path}: finalize after close")
        self._handle.flush()
        # Same header length, so patching in place cannot move the data offset.
        self._handle.seek(0)
        self._handle.write(_npy_header((self.count, *self.shape_tail), self.dtype))
        self._handle.close()
        self.closed = True
        self.finalized = True
        return {
            "path": self.path.name,
            "shape": [int(self.count), *self.shape_tail],
            "dtype": self.dtype.str,
            "bytes": self.path.stat().st_size,
        }

    def abort(self) -> None:
        if self.closed:
            return
        try:
            self._handle.flush()
        finally:
            self._handle.close()
            self.closed = True
            self.finalized = False

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown only
        try:
            self.abort()
        except Exception:
            pass


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in bounded chunks."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_render(stats: Sequence[dict[str, float]]) -> dict[str, Any] | None:
    """Per-shard summary of the readback content that was actually observed."""

    if not stats:
        return None
    count = len(stats)
    play_means = [float(stat["play_mean"]) for stat in stats]
    return {
        "frames_sampled": count,
        "play_mean": round(sum(play_means) / count, 6),
        "play_mean_min": round(min(play_means), 6),
        "play_mean_max": round(max(play_means), 6),
        "play_std_mean": round(sum(float(stat["play_std"]) for stat in stats) / count, 6),
        "hud_mean": round(sum(float(stat["hud_mean"]) for stat in stats) / count, 6),
        "hud_mean_max": round(max(float(stat["hud_mean"]) for stat in stats), 6),
        "zero_frames": sum(1 for stat in stats if int(stat["nonzero"]) == 0),
        "note": "content of the frames actually read; an all-zero block (shadow null "
                "device / readback off) fails closed before a shard exists",
    }


# ---------------------------------------------------------------------------
# the rolling 4-frame window
# ---------------------------------------------------------------------------
@dataclass
class RenderFrame:
    """One rendered frame as the obs-v1 planes (plus the frame it came from)."""

    frame: int
    play: np.ndarray
    hud: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {"frame": int(self.frame)}


class DecisionObservationBuffer:
    """The last ``stack`` rendered frames, and the sample they form.

    ``stack_for`` is the *only* place a decision's observation is assembled, so
    the window arithmetic exists once.  It refuses anything but
    ``d-4 .. d-1`` (or the documented episode-start padding).
    """

    def __init__(self, *, stack: int = obs_spec.FRAME_STACK) -> None:
        if stack != obs_spec.FRAME_STACK:
            raise ObservationError(
                f"frame stack is frozen at {obs_spec.FRAME_STACK}, got {stack}")
        self.stack = int(stack)
        self.frames: list[RenderFrame] = []
        self.seed: RenderFrame | None = None

    def reset(self, seed: RenderFrame | None = None) -> None:
        """Start a new episode; ``seed`` is the only padding row there is.

        The seed is deliberately **not** pushed into ``frames``: its frame index
        comes from the pre-``RESET`` counter, so it cannot satisfy the consecutive
        ``d-4 .. d-1`` rule; it is used exactly once, as the padding row of
        decision 0.
        """

        self.seed = seed
        self.frames = []

    def push(self, frame: RenderFrame) -> None:
        if frame.play.shape != obs_spec.PLAY_SHAPE:
            raise ObservationError(f"play frame shape {frame.play.shape} != {obs_spec.PLAY_SHAPE}")
        if frame.hud.shape != obs_spec.HUD_SHAPE:
            raise ObservationError(f"hud frame shape {frame.hud.shape} != {obs_spec.HUD_SHAPE}")
        if frame.play.dtype != np.uint8 or frame.hud.dtype != np.uint8:
            raise ObservationError("play/hud must be uint8")
        if self.frames and frame.frame <= self.frames[-1].frame:
            raise ObservationError(
                f"render frame {frame.frame} is not after {self.frames[-1].frame}")
        self.frames.append(frame)
        if len(self.frames) > self.stack:
            del self.frames[0]

    def stack_for(self, decision_frame: int, *, episode_start: bool = False) -> dict[str, Any]:
        window = decision_window(decision_frame, stack=self.stack)
        expected = list(window["render_frames"])
        if episode_start:
            if int(decision_frame) != 0:
                raise ObservationError(
                    f"episode-start padding is only valid at decision 0, got frame "
                    f"{decision_frame}")
            if self.seed is None:
                raise ObservationError(
                    "no render captured at RESET: the episode-start padding needs at "
                    "least one frame (is READBACK enabled before RESET?)")
            seed = self.seed
            pad = self.stack - 1
            rows = [seed] * self.stack
            return {
                "play": np.stack([row.play for row in rows]),
                "hud": rows[-1].hud.copy(),
                "render_frames": [int(seed.frame)] * self.stack,
                "window": window,
                "pad": pad,
                "episode_start": True,
                "seed_frame": int(seed.frame),
            }
        if len(self.frames) != self.stack:
            raise ObservationError(
                f"decision frame {decision_frame}: {len(self.frames)} render frame(s) "
                f"buffered, need {self.stack} consecutive frames ending at "
                f"{window['newest_render_frame']}")
        actual = [int(row.frame) for row in self.frames]
        if actual != expected:
            raise ObservationError(
                f"decision frame {decision_frame}: buffered renders {actual} != "
                f"required window {expected}")
        return {
            "play": np.stack([row.play for row in self.frames]),
            "hud": self.frames[-1].hud.copy(),
            "render_frames": actual,
            "window": window,
            "pad": 0,
            "episode_start": False,
            "seed_frame": None,
        }


# ---------------------------------------------------------------------------
# shared-memory pixel source (Linux only; the engine is the only writer)
# ---------------------------------------------------------------------------
def default_shm_name(prefix: str = "v400d0") -> str:
    import os

    return f"{prefix}-{os.getpid()}-{int(time.time() * 1000) % 1_000_000}"


class ShmPixelSource:
    """``SHM`` + ``READBACK`` on a running engine, read frame by frame.

    ``block_factory`` exists so tests can inject a fake block: the real one is
    :class:`touhou_rl.v400.gym.shm.SharedBlock`, which maps
    ``/dev/shm/<name>`` read-only.  ``read`` never accepts a frame it did not ask
    for: the header's frame counter is the seqlock, and a mismatch is an error,
    not a retry.
    """

    def __init__(
        self,
        ask: Callable[[str], str],
        *,
        name: str | None = None,
        block_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._ask = ask
        self.name = name or default_shm_name()
        self._factory = block_factory
        self.block: Any = None
        self.frames_read = 0
        self.readback_ms = 0.0

    # -- attach / readback ------------------------------------------------
    def attach(self) -> dict[str, Any]:
        reply = self._ask(f"SHM /{self.name}")
        if "active=1" not in reply:
            raise ObservationError(f"SHM attach refused: {reply[:200]}")
        factory = self._factory
        if factory is None:
            from touhou_rl.v400.gym.shm import SharedBlock

            factory = SharedBlock
        try:
            self.block = factory(self.name)
        except (OSError, ValueError) as error:
            raise ObservationError(
                f"cannot map shared block {self.name!r} ({error}); pixel readback "
                "needs the engine's /dev/shm segment") from error
        return {"shm": self.name, "reply": reply[:160]}

    def readback(self, enabled: bool) -> dict[str, Any]:
        reply = self._ask(f"READBACK {1 if enabled else 0}")
        fields = _parse_tokens(reply)
        expected = "1" if enabled else "0"
        if fields.get("readback") != expected:
            raise ObservationError(f"READBACK {expected} refused: {reply[:200]}")
        return {
            "readback": int(fields.get("readback", 0)),
            "frames": int(fields.get("frames", 0)),
            "micros": int(fields.get("micros", 0)),
        }

    # -- pixels -----------------------------------------------------------
    def read(self, expected_frame: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """``(play64, hud, header)``; ``expected_frame=None`` skips the frame gate."""

        if self.block is None:
            raise ObservationError("shared block is not attached")
        started = time.perf_counter()
        try:
            header = self.block.header()
        except ValueError as error:  # bad magic
            raise ObservationError(f"shared block is not an engine block: {error}") from error
        from touhou_rl.v400.gym.shm import FLAG_PIXELS_VALID

        if not int(header.get("flags", 0)) & FLAG_PIXELS_VALID:
            raise ObservationError(
                "shared block has no published pixels (READBACK was off when the "
                "frame was drawn)")
        if int(header.get("width", 0)) != 640 or int(header.get("height", 0)) != 480:
            raise ObservationError(
                f"readback is {header.get('width')}x{header.get('height')}, the "
                "obs-v1 encoder needs 640x480")
        if expected_frame is not None and int(header["frame"]) != int(expected_frame):
            raise ObservationError(
                f"shared-memory frame {header['frame']} != expected {expected_frame}: "
                "the observation would not be the frame the label was taken at")
        try:
            pixels = self.block.pixels()
        except ValueError as error:
            raise ObservationError(f"cannot read pixels: {error}") from error
        play, hud = obs_spec.encode_planes(pixels)
        self.frames_read += 1
        self.readback_ms += (time.perf_counter() - started) * 1000.0
        return play, hud, header

    def close(self, *, unlink: bool = True) -> None:
        block = self.block
        self.block = None
        if block is not None:
            try:
                block.close()
            except Exception:  # pragma: no cover - closing must not mask errors
                pass
        if unlink:
            try:
                from touhou_rl.v400.gym.shm import segment_path

                segment_path(self.name).unlink()
            except (OSError, ValueError):  # pragma: no cover - platform/race dependent
                pass


def _parse_tokens(reply: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in reply.split():
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# the per-run observation state machine
# ---------------------------------------------------------------------------
class TeacherObservationRun:
    """One run's observation side: SHM pixels in, P3 shard rows out.

    The CLI owns the engine and the decisions; this class owns *when* a frame may
    be read (only inside a commit, only one frame per ``STEP``) and *what* the
    stored sample is.  Every method either proves its frame arithmetic or raises
    :class:`ObservationError`.
    """

    def __init__(
        self,
        *,
        out_root: str | Path,
        key: str,
        meta: dict[str, Any],
        source: ShmPixelSource,
        writer: "TeacherObsShardWriter | None" = None,
        buffer: DecisionObservationBuffer | None = None,
        frames_per_decision: int = 2 * obs_spec.DECISION_FRAMES,
    ) -> None:
        self.out_root = Path(out_root)
        self.key = str(key)
        self.meta = dict(meta)
        self.source = source
        self.buffer = buffer or DecisionObservationBuffer()
        self.frames_per_decision = int(frames_per_decision)
        self.writer = writer or TeacherObsShardWriter(
            self.out_root,
            key=self.key,
            meta={**self.meta, "frames_per_decision": self.frames_per_decision},
        )
        self.pending: dict[str, Any] | None = None
        self.decision_frame: int | None = None
        self.seed_info: dict[str, Any] | None = None
        self.attach_info: dict[str, Any] | None = None
        self.readback_off_info: dict[str, Any] | None = None

    # -- lifecycle --------------------------------------------------------
    def attach(self) -> dict[str, Any]:
        """Attach the shared block and enable readback (before ``RESET``)."""

        check_frozen_spec()
        self.attach_info = self.source.attach()
        self.source.readback(True)
        return self.attach_info

    def seed_after_reset(self, *, state_frame: int | None = None) -> dict[str, Any]:
        """Capture the post-``RESET`` render, then stop readback.

        ``RESET`` zeroes the engine's frame counter, so this render carries the
        pre-reset setup number; it is recorded as ``seed_frame`` and used only as
        the episode-start padding row.
        """

        play, hud, header = self.source.read(None)
        if state_frame is not None and int(state_frame) != 0:
            raise ObservationError(
                f"RESET left the engine frame counter at {state_frame}, expected 0; the "
                "frame/decision arithmetic of the shard would be meaningless")
        seed = RenderFrame(frame=int(header["frame"]), play=play, hud=hud)
        stats = require_rendered(play, hud, where="post-RESET seed render")
        self.buffer.reset(seed)
        self.readback_off_info = self.source.readback(False)
        self.seed_info = {
            "seed_render_frame": int(header["frame"]),
            "flags": int(header.get("flags", 0)),
            "state_frame_at_reset": None if state_frame is None else int(state_frame),
            "readback_frames": self.readback_off_info["frames"],
            "readback_micros": self.readback_off_info["micros"],
            "content": stats,
        }
        self.writer.render_stats.append(stats)
        return self.seed_info

    # -- per decision -----------------------------------------------------
    def begin_decision(self, engine, decision_frame: int) -> dict[str, Any]:
        """Frame gate + snapshot the stack the teacher is about to be queried on."""

        frame_now = int(engine.frame())
        if frame_now != int(decision_frame):
            raise ObservationError(
                f"engine frame {frame_now} != decision frame {decision_frame}: the "
                "frame counter and the decision index disagree")
        expected_step = self.writer.rows * self.frames_per_decision
        if frame_now != expected_step:
            raise ObservationError(
                f"decision {self.writer.rows} starts at frame {frame_now}, expected "
                f"{expected_step} ({self.frames_per_decision} frames per decision)")
        sample = self.buffer.stack_for(
            int(decision_frame), episode_start=(self.writer.rows == 0))
        self.pending = sample
        self.decision_frame = int(frame_now)
        return sample

    def verify_frame_unchanged(self, engine) -> None:
        """The record/verify phases must not move the shadow (child path)."""

        frame_now = int(engine.frame())
        if self.decision_frame is not None and frame_now != self.decision_frame:
            raise ObservationError(
                f"the decision point moved from frame {self.decision_frame} to "
                f"{frame_now} during recording/verification; the observation would "
                "no longer match the label")

    def commit(self, engine, block: Iterable[tuple[int, int]], *, ask=None) -> dict[str, Any]:
        """Commit one decision **one frame at a time**, reading every frame.

        ``block`` is :func:`touhou_rl.v400.teacher.shoot.commit_block`'s
        ``(mask, frames)`` list.  Splitting ``STEP mask 2`` into two
        ``STEP mask 1`` is state-identical: ``HandleStep`` sets ``g_mask`` once and
        then pumps frames (``rlharness.cpp:654-694``), and the mask is the same for
        both frames of a slot.
        """

        ask = ask if ask is not None else engine.ask
        steps: list[str] = []
        advanced = 0
        exited = False
        self.source.readback(True)
        try:
            for mask, count in block:
                for _ in range(int(count)):
                    reply = ask(f"STEP {int(mask)} 1")
                    steps.append(reply)
                    fields = _parse_tokens(reply)
                    step_advanced = int(fields.get("advanced", 0))
                    advanced += step_advanced
                    if fields.get("exit") == "1":
                        exited = True
                    if step_advanced != 1:
                        return {"steps": steps, "advanced": advanced,
                                "exited": exited, "complete": False,
                                "reason": f"advanced={step_advanced} on frame {len(steps)}"}
                    frame_now = int(fields["frame"])
                    play, hud, header = self.source.read(frame_now - 1)
                    self.writer.render_stats.append(
                        require_rendered(play, hud, where=f"frame {frame_now - 1}"))
                    self.buffer.push(RenderFrame(frame=int(header["frame"]),
                                                 play=play, hud=hud))
                    if exited:
                        break
        finally:
            self.source.readback(False)
        complete = advanced == int(sum(int(count) for _, count in block)) and not exited
        return {"steps": steps, "advanced": advanced, "exited": exited,
                "complete": complete,
                "reason": None if complete else "engine stopped inside the commit"}

    # -- sample -----------------------------------------------------------
    def add_sample(self, *, index: int, label) -> dict[str, Any]:
        """Write the decision's obs planes and its ``labels.jsonl`` row."""

        if self.pending is None:
            raise ObservationError(f"decision {index}: no pending observation")
        row = self.writer.add(
            decision=int(index),
            decision_frame=int(self.decision_frame if self.decision_frame is not None else -1),
            label=label,
            sample=self.pending,
            extra={
                "frame_event_flags": label.frame_event_flags,
                "shoot_release": bool(getattr(label, "shoot_release", False)),
                "bomb_death_available": bool(
                    getattr(label, "bomb_death_available", False)),
            },
        )
        self.pending = None
        return row

    def drop_pending(self, reason: str) -> None:
        if self.pending is not None:
            self.writer.dropped.append({"decision": self.writer.rows, "reason": reason})
        self.pending = None

    # -- finish -----------------------------------------------------------
    def finalize(self, *, report: dict[str, Any] | None = None) -> dict[str, Any]:
        summary = self.writer.finalize(report=report)
        summary["seed"] = self.seed_info
        summary["attach"] = self.attach_info
        summary["frames_read"] = self.source.frames_read
        summary["pixel_ms"] = round(self.source.readback_ms, 3)
        return summary

    def abort(self, error: BaseException) -> dict[str, Any] | None:
        self.pending = None
        try:
            return self.writer.abort(error)
        finally:
            self.source.close()

    def close(self) -> None:
        self.source.close()


# ---------------------------------------------------------------------------
# the shard writer (streamed, staged, atomic rename)
# ---------------------------------------------------------------------------
class TeacherObsShardWriter:
    """``shards/<key>/{labels.jsonl, obs/play64.npy, obs/hud.npy, shard.json}``.

    Rows are streamed (a 200k-decision shard is ~5 GB and is never held in RAM).
    Everything is written under ``shards/.<key>.partial/`` and only renamed to
    ``shards/<key>/`` by :meth:`finalize`, so a crashed or failed run leaves an
    explicitly incomplete directory that neither the manifest nor
    ``dataset.load_teacher_shard`` will accept.
    """

    def __init__(
        self,
        out_root: str | Path,
        *,
        key: str,
        meta: dict[str, Any],
    ) -> None:
        self.out_root = Path(out_root)
        self.key = str(key)
        if not self.key or "/" in self.key or self.key.startswith("."):
            raise ObservationError(f"invalid shard key {key!r}")
        self.final_dir = self.out_root / "shards" / self.key
        self.partial_dir = self.out_root / "shards" / f".{self.key}.partial"
        if self.final_dir.exists():
            raise ObservationError(
                f"shard {self.final_dir} already exists; refusing to overwrite a "
                "recorded shard")
        if self.partial_dir.exists():
            raise ObservationError(
                f"staging directory {self.partial_dir} already exists from an earlier "
                "unfinished run; move it aside instead of mixing shards")
        (self.partial_dir / "obs").mkdir(parents=True)
        self.meta = dict(meta)
        self.rows = 0
        self.dropped: list[dict[str, Any]] = []
        self.frames: list[tuple[int, int]] = []
        #: ``require_rendered`` stats of every frame actually read (evidence).
        self.render_stats: list[dict[str, float]] = []
        self._labels_path = self.partial_dir / "labels.jsonl"
        self._labels = open(self._labels_path, "w", encoding="utf-8")
        self._play = NpyAppendWriter(self.partial_dir / "obs" / "play64.npy",
                                     (obs_spec.FRAME_STACK, *obs_spec.PLAY_SHAPE), np.uint8)
        self._hud = NpyAppendWriter(self.partial_dir / "obs" / "hud.npy",
                                    obs_spec.HUD_SHAPE, np.uint8)
        self._closed = False
        self._finalized = False
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    # -- rows -------------------------------------------------------------
    def add(
        self,
        *,
        decision: int,
        decision_frame: int,
        label,
        sample: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise ObservationError(f"{self.key}: add after close")
        if int(decision) != self.rows:
            raise ObservationError(
                f"{self.key}: decision {decision} is not the next row ({self.rows})")
        expected_frame = int(sample["window"]["decision_frame"])
        if int(decision_frame) != expected_frame:
            raise ObservationError(
                f"{self.key}: decision {decision} frame {decision_frame} != window "
                f"{expected_frame}")
        play = np.asarray(sample["play"], dtype=np.uint8)
        hud = np.asarray(sample["hud"], dtype=np.uint8)
        if play.shape != (obs_spec.FRAME_STACK, *obs_spec.PLAY_SHAPE):
            raise ObservationError(f"{self.key}: play stack {play.shape} is not obs-v1")
        if hud.shape != obs_spec.HUD_SHAPE:
            raise ObservationError(f"{self.key}: hud {hud.shape} is not obs-v1")
        self._play.append(play)
        self._hud.append(hud)
        row = self._label_row(int(decision), int(decision_frame), label, sample, extra or {})
        self._labels.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows += 1
        self.frames.append((int(decision_frame),
                            int(sample["render_frames"][-1])))
        return row

    def _label_row(
        self,
        decision: int,
        decision_frame: int,
        label,
        sample: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        scores = np.asarray(label.scores, dtype=np.float64)
        mask = np.asarray(label.verified_mask, dtype=bool)
        chosen = label.chosen_plan
        row = {
            "kind": "decision",
            "schema": TEACHER_OBS_SCHEMA,
            "decision": decision,
            # ``frame`` is the decision point (pre-commit) in this artifact; the
            # teacher CLI's JSONL uses the post-commit counter, so the name is
            # spelled out twice rather than left to convention.
            "frame": decision_frame,
            "decision_frame": decision_frame,
            "frames_per_decision": int(self.meta.get("frames_per_decision", 4)),
            "decision_frames": obs_spec.DECISION_FRAMES,
            "scores": [round(float(value), 6) for value in scores],
            "verified_mask": [bool(value) for value in mask],
            "verified_count": int(mask.sum()),
            "bomb": int(label.bomb),
            "margin": None if chosen is None else float(label.margin),
            "chosen_plan": None if chosen is None else chosen.as_dict(),
            "stage": int(label.stage),
            "difficulty": int(label.difficulty),
            "obs": {
                "shard": self.key,
                "shard_dir": str(self.final_dir),
                "play": "obs/play64.npy",
                "hud": "obs/hud.npy",
                "row": decision,
                "observation_time": OBSERVATION_TIME,
                "stack": obs_spec.FRAME_STACK,
                "stack_order": obs_spec.STACK_ORDER,
                "render_frames": [int(value) for value in sample["render_frames"]],
                "newest_render_frame": int(sample["window"]["newest_render_frame"]),
                "label_frame": decision_frame,
                "label_frames": list(sample["window"]["label_frames"]),
                "p3_window_last_row": int(sample["window"]["p3_window_last_row"]),
                "p3_label_frame": int(sample["window"]["p3_label_frame"]),
                "pad": int(sample["pad"]),
            },
        }
        row.update(extra)
        return row

    # -- finish -----------------------------------------------------------
    def finalize(self, *, report: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._closed:
            raise ObservationError(f"{self.key}: finalize after close")
        if self.rows == 0:
            raise ObservationError(f"{self.key}: a shard needs at least one decision")
        check_frozen_spec()
        self._labels.flush()
        self._labels.close()
        self._closed = True
        play_meta = self._play.finalize()
        hud_meta = self._hud.finalize()
        # Shard-relative paths (the writer only knows the file names).
        play_meta["path"] = "obs/play64.npy"
        hud_meta["path"] = "obs/hud.npy"
        play_meta["sha256"] = sha256_file(self.partial_dir / play_meta["path"])
        hud_meta["sha256"] = sha256_file(self.partial_dir / hud_meta["path"])
        labels_meta = {
            "path": "labels.jsonl",
            "rows": int(self.rows),
            "bytes": self._labels_path.stat().st_size,
            "sha256": sha256_file(self._labels_path),
        }
        first_frame = int(self.frames[0][0])
        last_frame = int(self.frames[-1][0])
        obs_bytes = int(play_meta["bytes"]) + int(hud_meta["bytes"])
        # EXP-P1-014 dual channel: the pixels can come from an independent env-role
        # replay instead of this process.  ``alignment_extra`` is the only hook the
        # dual tool needs; without it the frozen ALIGNMENT is written verbatim.
        alignment = dict(ALIGNMENT)
        alignment.update({str(key): value
                          for key, value in (self.meta.get("alignment_extra") or {}).items()})
        shard: dict[str, Any] = {
            "schema": SHARD_SCHEMA,
            "shard": self.key,
            "complete": True,
            "created_at": self.started_at,
            "obs_spec_version": obs_spec.OBS_SPEC_VERSION,
            "obs_spec_digest": obs_spec.obs_spec_digest(),
            "encoder": "touhou_rl.v400.student.obs_spec.encode_planes",
            "alignment": alignment,
            "decision_count": int(self.rows),
            "decision_frames": obs_spec.DECISION_FRAMES,
            "frames_per_decision": int(self.meta.get("frames_per_decision", 4)),
            "first_decision_frame": first_frame,
            "last_decision_frame": last_frame,
            "stage_frames": [first_frame, last_frame + int(self.meta.get("frames_per_decision", 4))],
            "labels": labels_meta,
            "obs": {
                "play64": {**play_meta, "shape": play_meta["shape"],
                           "stack": obs_spec.FRAME_STACK,
                           "stack_order": obs_spec.STACK_ORDER},
                "hud": {**hud_meta, "shape": hud_meta["shape"]},
                "bytes_per_decision": obs_bytes // int(self.rows),
                "bytes_total": obs_bytes,
            },
            "rows": self.rows,
            "dropped": list(self.dropped),
            # What the readback actually contained: a run that observed nothing is
            # refused before this point, so these are real rendered frames.
            "render": _aggregate_render(self.render_stats),
            "report": report or {},
        }
        shard.update({key: value for key, value in self.meta.items() if key not in shard})
        (self.partial_dir / "shard.json").write_text(
            json.dumps(shard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if self.final_dir.exists():  # pragma: no cover - raced by another writer
            raise ObservationError(f"{self.final_dir} appeared during finalize")
        self.partial_dir.rename(self.final_dir)
        manifest = self._update_manifest(shard)
        return {
            "shard": self.key,
            "dir": str(self.final_dir),
            "decisions": int(self.rows),
            "frames": [first_frame, last_frame + int(self.meta.get("frames_per_decision", 4))],
            "obs_bytes": obs_bytes,
            "obs_bytes_per_decision": obs_bytes // int(self.rows),
            "labels_bytes": labels_meta["bytes"],
            "labels_sha256": labels_meta["sha256"],
            "play_sha256": play_meta["sha256"],
            "hud_sha256": hud_meta["sha256"],
            "obs_spec_version": shard["obs_spec_version"],
            "obs_spec_digest": shard["obs_spec_digest"],
            "render": shard["render"],
            "manifest": manifest,
            "dropped": list(self.dropped),
        }

    def _update_manifest(self, shard: dict[str, Any]) -> dict[str, Any]:
        path = self.out_root / "manifest.json"
        payload: dict[str, Any]
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != MANIFEST_SCHEMA:
                raise ObservationError(
                    f"{path} schema {payload.get('schema')!r} != {MANIFEST_SCHEMA!r}")
            if payload.get("obs_spec_digest") != shard["obs_spec_digest"]:
                raise ObservationError(
                    f"{path} was written under obs digest "
                    f"{payload.get('obs_spec_digest')!r}, this shard "
                    f"{shard['obs_spec_digest']!r}; mixing specs in one dataset")
            if payload.get("obs_spec_version") != shard["obs_spec_version"]:
                raise ObservationError(f"{path} obs spec version differs")
        else:
            payload = {
                "schema": MANIFEST_SCHEMA,
                "kind": "teacher-observation-dataset",
                "created_at": shard["created_at"],
                "obs_spec_version": shard["obs_spec_version"],
                "obs_spec_digest": shard["obs_spec_digest"],
                "alignment": dict(ALIGNMENT),
                "shards": [],
            }
        entries = [row for row in payload.get("shards") or [] if row.get("shard") != self.key]
        entries.append({
            "shard": self.key,
            "dir": str(self.final_dir),
            "decisions": int(self.rows),
            "stage_frames": list(shard["stage_frames"]),
            "obs_bytes": int(shard["obs"]["bytes_total"]),
            "stage": self.meta.get("stage"),
            "difficulty": self.meta.get("difficulty"),
            "seed": self.meta.get("seed"),
            "mode": self.meta.get("mode"),
            "labels_sha256": shard["labels"]["sha256"],
            "play_sha256": shard["obs"]["play64"]["sha256"],
            "hud_sha256": shard["obs"]["hud"]["sha256"],
        })
        entries.sort(key=lambda row: str(row["shard"]))
        payload["shards"] = entries
        payload["totals"] = {
            "shards": len(entries),
            "decisions": sum(int(row["decisions"]) for row in entries),
            "obs_bytes": sum(int(row["obs_bytes"]) for row in entries),
            "stage_frames": sum(int(row["stage_frames"][1]) - int(row["stage_frames"][0])
                                for row in entries),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
        return payload["totals"]

    def abort(self, error: BaseException) -> dict[str, Any]:
        """Leave an explicitly incomplete staging directory; never a shard."""

        detail = {"schema": "v400-teacher-obs-incomplete-1", "shard": self.key,
                  "complete": False, "rows": int(self.rows),
                  "error": f"{type(error).__name__}: {error}",
                  "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            if not self._closed:
                self._labels.flush()
                self._labels.close()
                self._closed = True
            self._play.abort()
            self._hud.abort()
            (self.partial_dir / "incomplete.json").write_text(
                json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception:  # pragma: no cover - evidence is best effort on a bad path
            pass
        return detail

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown only
        try:
            if not self._closed:
                self.abort(ObservationError("writer garbage-collected without finalize"))
        except Exception:
            pass


__all__ = [
    "ALIGNMENT",
    "DecisionObservationBuffer",
    "FROZEN_OBS_SPEC_DIGEST",
    "MANIFEST_SCHEMA",
    "NpyAppendWriter",
    "OBSERVATION_TIME",
    "ObservationError",
    "RenderFrame",
    "SHARD_SCHEMA",
    "ShmPixelSource",
    "TEACHER_OBS_SCHEMA",
    "TeacherObservationRun",
    "TeacherObsShardWriter",
    "check_frozen_spec",
    "decision_window",
    "default_shm_name",
    "sha256_file",
]
