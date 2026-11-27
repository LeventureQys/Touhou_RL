#!/usr/bin/env python3
"""V8.32 conditional takeover entry: replay the tape, verify it, then hand over.

What this tool is for
---------------------
S1N-MB2 needs to start an arm at a *natural* entry point, not from ``RESET`` and
not from a saved state file: ``RESET 0 1 0 <seed>``, then the recorded tape frame
by frame up to ``--entry-frame F``, a hash check at **every** slot boundary, and
only then control passes to one arm.  The three arms are:

``tape``       continue the recorded mask schedule from the slot at ``F`` (the
               teacher continuation);
``constant``   emit one fixed candidate for every decision of the window, with
               the same SHOOT release pulse the executor uses;
``checkpoint`` run an **obs-v3 actor** loaded from a checkpoint file: the last
               four prefix frames are rendered into a 4x112x96 play stack, the
               HUD strip comes from the same render, the 8x18 action history is
               seeded from the tape's ``executed_cand`` for decisions
               ``0..entry-1``, and each policy decision advances two frames with
               one render per frame.  The policy sees **pixels and its own
               action history only**; the SHOOT release pulse stays exogenous.

What the checkpoint arm refuses (fail-closed, never a substitute)
-----------------------------------------------------------------
* a missing ``--checkpoint``, a sha256 that is not the file's sha256, or a
  checkpoint whose ``obs_spec_version``/``obs_spec_digest`` is not the frozen
  obs-v3 pair (``ef2fbd15…a105``): refused **before** an engine is started;
* pixels that cannot be read: the shared block must be attached with
  ``READBACK 1`` **before** ``RESET`` (so every frame of the episode, prefix
  included, republishes it), ``FLAG_PIXELS_VALID`` must be set on every read, and
  the block's frame counter must strictly advance between reads -- a zero image
  or a previous frame is never used to keep a run alive;
* a policy output outside ``cand 0..17`` / ``bomb 0..1``: the window stops
  instead of being logged as a policy decision.

The stack window
----------------
``ObservationBufferV3`` keeps the newest four frames, so the prefix renders
exactly the last four frames (``F-3 .. F``); every earlier render would be
evicted before the entry and is not read.  That is the same stack content
``tools/rl/v400_student_eval.py`` has at its next decision after the same frames,
and ``tests/test_v8_32_takeover.py`` proves it by running both paths on the same
synthetic pixels and comparing the three tensors ``model.act`` receives.

Why a new tool instead of ``v8_31_followup.py intervene``
--------------------------------------------------------
That command cannot express this contract.  Measured on this checkout:

* its flags are ``--repo/--out/--engine/--data-source/--max-starts/
  --max-engine-frames/--max-wall-s/--max-write-gib/--timeout-s/--no-xvfb`` --
  ``--seed``, ``--entry-frame``, ``--arm``, ``--cap-frames``, ``--constant-cand``,
  ``--tape-slots``, ``--decision-log`` and ``--expected-sha`` are all rejected by
  ``argparse`` with ``unrecognized arguments: ...``;
* the frozen card is read from ``<out>/02b/task_card.json`` (no ``--card`` flag,
  refusal text ``the frozen task card is missing: <path>``); the card's
  ``content_sha256`` is a *self* digest, so a hand-synthesized card is accepted --
  the blocker is the CLI and the hardwired inputs, not the card;
* it always runs **two** arms from two engines, hardwires ``RESET ... 12``
  (``RESET_SEED``), hardwires ``CONTROL_CAND = 12``, reads the tape from a
  hardwired T04 shard path *plus* a declared 64-decision tail, and needs the
  D1 canonical corpus for its anchors;
* its telemetry is ``02c/arm_{T,C}/decisions.jsonl`` with schema
  ``v831-02c-decision/1`` and no header line, and ``02c/intervene.json`` has no
  ``entry_verified``/``mismatch_slot``: an entry hash mismatch is raised as
  ``EngineRefused`` and the arm loop still starts the other arm.

What this runner guarantees
---------------------------
1. ``RESET <stage 0> <difficulty 1> <team 0> <seed>``; the reply is stored
   verbatim in ``result.json`` (``reset``) and in the decision-log header.
2. The tape prefix is replayed from the *tape itself*: the first slot's
   ``history_frames``/``history`` (the 8 lead-in decisions that build frames
   0..15) and then every slot with ``decision_frame < F``, two ``STEP <mask> 1``
   per decision, one ``HASH`` after every slot.  A slot whose ``hash_post`` does
   not come back fails the run **closed**: ``entry_verified=false``,
   ``mismatch_slot`` named, non-zero exit, and no control is ever handed over.
3. At ``F`` the tool records the full ``STATE`` (25 named fields), ``PLAYER``,
   ``HUD`` and an ``ENEMIES`` summary, plus ``entry_frame``/``entry_hash``.
4. Hand-over runs until ``--cap-frames`` (spent in whole 2-frame decisions) or
   until a death / game over / stage change / engine exit / tape end.
5. ``--decision-log`` (optional) gets one header line, then one line per
   hand-over decision; the row schema is ``tools/rl/v400_student_eval.py``'s
   ``v400-student-eval-decision-1`` with the same field names, plus ``masks`` and
   the referee-side ``enemies`` reading described below.
6. ``result.json`` carries ``entry_verified``, ``entry_frame``, ``entry_hash``,
   ``mismatch_slot``, ``frames``, ``stop``, ``deaths``, ``wall_s``, the engine
   sha256, the four mute bytes before and after the engine, the failure text and
   ``boss_summary`` (the aggregate of those per-decision readings).

The per-decision ``enemies`` reading (referee side, never a policy input)
-------------------------------------------------------------------------
With ``--decision-log`` every hand-over decision also reads one ``ENEMIES`` page
(``ENEMIES 0 16``, the same page ``v400_student_eval.py`` reads) and writes its
summary into the row:

``{"available": true, "count": N, "boss_count": N, "boss_life": N,
"boss_slots": [...], "truncated": bool}``

``count`` is how many rows the page returned, ``boss_count``/``boss_slots`` the
rows that carry ``ENEMY_FLAG_BOSS`` (``EnemyManager.hpp:69-71``), ``boss_life``
the largest ``life`` among those rows (``0`` when the page lists no boss) and
``truncated`` says the page is smaller than the engine's own active-enemy count,
so a ``truncated: true`` page is not a census and a boss can sit off-page.  A
refused or undecodable page becomes
``{"available": false, "reason": <the engine's own text>}`` and the row is still
written: a fabricated ``0`` would be indistinguishable from a measured one.
Without ``--decision-log`` nothing is probed, so a run that writes no log sends
exactly the protocol traffic it always did.

This reading is **referee-side telemetry**.  It is never an input to a policy
(the checkpoint arm's ``policy_input`` stays "pixels and its own action history
only"), it never chooses a candidate, and it never stops or fails a window.
Judging uses ``HAZARDS`` row identity (slot + ``flags1`` + signed life);
``ENEMIES`` pagination is diagnostic only.  ``result.json``'s ``boss_summary``
aggregates the per-decision readings with the row's own ``frame``: the first row
whose ``boss_count > 0`` (``first_boss_frame``/``boss_life_first``), the first
row that lists a boss **and** reads ``boss_life == 0``
(``boss_zero_life_frame``), and the last available reading's ``boss_life``.  With
no ``--decision-log`` (or with every probe refused) it is ``available: false``,
``samples: 0`` and every frame/life field is ``null`` -- never ``0``.

口径（中文）：逐决策 ``enemies`` 是裁判侧读数，不进策略输入；判分以 ``HAZARDS``
行身份为准，``ENEMIES`` 分页只作诊断。

``--save-obs DIR``: the student's own observation planes
--------------------------------------------------------
The checkpoint arm's obs-v3 tensors used to live in memory only, so the states a
student *visits* could not be re-labelled by a teacher -- which is exactly what the
WP05 targeted re-collection (DAgger) needs.  ``--save-obs DIR`` (checkpoint arm
only; any other arm is refused before a client exists) writes:

``DIR/slots.jsonl``
    one row per completed decision, flushed as it is written, aligned with the T04
    corpus: ``decision_frame``/``commit_frames``/``duration_frames``/
    ``obs_render_frames``/``history_frames``/``history``/``executed_cand``/``mask``/
    ``hash_pre``/``hash_post``/``episode_seed``/``slot_id``/``label_source``/
    ``obs_source``, plus the measured ``state``/``events``/``end``.  The fields a
    student rollout cannot fill (``slot_id``, ``scores``, ``chosen``, ``margin``, the
    verification fields, ...) are written as ``null`` -- never as 0 -- and listed in
    ``DIR/README.md`` with the reason.  ``label_source`` is
    ``dagger_student_rollout``: no teacher label exists yet, and none of the T04
    (``teacher_2f``) or D1 (``shadow_teacher``) words would be true.
``DIR/obs/play96x112.npy``, ``hud.npy``, ``act_hist.npy``
    ``N x 4x112x96`` / ``N x 56x160`` / ``N x 8x18`` **uint8**, the raw 0..255
    planes ``tools/rl/v400_student_eval.py`` feeds its actor (the ``/255`` happens
    at the tensor boundary).  They are copied off the same
    ``ObservationBufferV3``/``ActionHistoryBuffer`` state ``model.act``'s tensors
    were built from: never re-rendered, never re-normalized, and the test pins
    ``torch.equal`` against the tensors the actor actually received.
``DIR/meta.json``
    schema, runner sha256, checkpoint path+sha256, obs-spec version/digest,
    ``entry_frame``, tape, seed, decisions, frames, the stop word, and each plane's
    shape/dtype/sha256.
``DIR/README.md``
    the field table, every ``null`` field and why, and the frame-ruler note.

Fail-closed: a directory that is not empty is **refused** (never merged, never
overwritten), a row that cannot be written stops the window, and a plane/meta
write failure leaves the rows already written, records the reason and exits
non-zero.  Without ``--save-obs`` nothing is created and not one extra protocol
command is sent.

口径（中文）：``--save-obs`` 只属于 checkpoint 臂，落盘的是**学生自己看到的**三张
obs-v3 平面（原始 0–255，不归一化）与逐决策行；``label_source`` 写
``dagger_student_rollout``，因为这一行还没有教师标签；拿不到的字段一律 ``null`` 并在
``README.md`` 里说明，不许编造。

The mask rule (why two masks are not recorded)
----------------------------------------------
T04 stores **one** mask per slot (the decision's first frame).  The executor's
two-frame commit is ``commit_frames(cand, decision_frames=2,
release_shoot=release_due(decision_index, period))``, i.e. ``[m, m]`` or, on a
pulse decision, ``[m, m & ~SHOOT]``.  This tool derives the pair with that frozen
function and requires its first element to equal the recorded ``mask``, so
``--shoot-pulse-period`` (default 48) must stay at the value the tape was
recorded with or the hash chain breaks at the first pulse decision (137 releases
inside the seed-12 shard).

Safety
------
No audio: the four mute bytes of ``<data-dir>/th08.cfg`` are decoded before the
start and the run is **refused** unless ``musicMode``/``playSounds`` and both
volumes are 0 (``AGENTS.md`` section 9).  The same bytes are read again after the
engine exits and a change is reported.  ``SDL_AUDIODRIVER=dummy`` is set by
``HarnessClient`` itself.  This tool never starts an engine unless it is asked to
run the ``run`` subcommand.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOOL_PATH = Path(__file__).resolve()

#: One decision is two game frames (``touhou_rl.v420.contracts.DECISION_FRAMES``).
DECISION_FRAMES = 2
#: The first ``decision_frame`` a T04 shard records; frames 0..15 are the 8 lead-in
#: decisions the shard keeps in slot 0's ``history_frames``/``history``.
TAPE_FIRST_FRAME = 16
#: ``(decision_index + 1) % period == 0`` -- the recorded SHOOT release phase.
DEFAULT_PULSE_PERIOD = 48
DEFAULT_TIMEOUT_S = 120.0
#: ``RESET <stage> <difficulty> <team> <seed>``: stage 0, difficulty 1 = normal.
RESET_STAGE = 0
RESET_DIFFICULTY = 1
RESET_TEAM = 0
#: ``--arm`` values.  ``checkpoint`` runs an obs-v3 actor from a file.
ARMS = ("tape", "constant", "checkpoint")
#: ``cand = focus * 9 + move``: 18 candidates.
N_CAND = 18

SCHEMA_RESULT = "v832-takeover-result/1"
SCHEMA_LOG_HEADER = "v832-takeover-decision-log/1"
#: The row schema of ``tools/rl/v400_student_eval.py --decision-log``.
SCHEMA_LOG_ROW = "v400-student-eval-decision-1"

#: ``--save-obs DIR`` schemas: one row per decision, one identity for the window.
SCHEMA_OBS_SLOT = "v832-takeover-obs-slot/1"
SCHEMA_OBS_META = "v832-takeover-obs/1"
#: ``label_source`` of a student rollout.  A T04 slot says ``teacher_2f`` (or
#: ``teacher_slot0`` for the shard's first slot) and the D1/DAgger collectors say
#: ``shadow_teacher``/``shadow_teacher_lockstep``; none of them is true for a row
#: **the policy itself produced**, and there is no teacher label yet -- filling the
#: teacher's value in would claim a label that was never taken.  So the value names
#: exactly what the row is: a student rollout that the WP05 DAgger pass still has
#: to label.  ``DIR/README.md`` spells this out next to every other vocabulary.
OBS_LABEL_SOURCE = "dagger_student_rollout"
#: Where the planes came from: this runner's readback, never a re-render.
OBS_SOURCE = "takeover-runner"
OBS_SLOTS_FILE = "slots.jsonl"
OBS_META_FILE = "meta.json"
OBS_README_FILE = "README.md"
OBS_PLANE_DIR = "obs"
#: Slot fields a student rollout cannot fill.  They are written as ``null`` -- never
#: 0, never a guess -- and ``DIR/README.md`` lists each one with its reason.  The
#: DAgger pass that re-labels these states is what fills them.
OBS_SLOT_NULL_FIELDS: tuple[tuple[str, str], ...] = (
    ("slot_id", "a student decision is not a T04 slot: the student's state at frame "
                "d is not the tape's slot at frame d, so a tape slot_id would name "
                "the wrong state (join on (episode_seed, decision_frame, hash_pre))"),
    ("shard_key", "this window is not a T04 shard; the tape's shard path is in "
                  "meta.json under tape.path"),
    ("provenance", "there is no T04 corpus/slot/block to point at: the row's own "
                   "identity is (episode_seed, decision_frame, hash_pre/hash_post)"),
    ("scores", "the 18 teacher candidates' scores are not computed by this runner"),
    ("verified_mask", "no per-candidate teacher verification was run"),
    ("verified_count", "no per-candidate teacher verification was run"),
    ("chosen", "no teacher choice exists: the row records the policy's own action"),
    ("chosen_verified", "no teacher choice exists to verify"),
    ("chosen_score", "no teacher score exists for a student state yet"),
    ("teacher_bomb", "no teacher label was taken"),
    ("margin", "a teacher margin needs a teacher label"),
    ("label_verified", "the row carries no label yet (label_source is "
                       "dagger_student_rollout)"),
)

#: The obs-v3 contract the checkpoint arm feeds the policy.  Both values are the
#: frozen ones from :mod:`touhou_rl.v400.student.obs_spec_v3`; the literal digest is
#: pinned here so a moved spec is a refusal instead of a quiet re-interpretation.
OBS_SPEC_V3_VERSION = "obs-v3"
OBS_SPEC_V3_DIGEST = (
    "ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105")
#: What the tape/constant arms "observe" (nothing) and what the checkpoint arm's
#: policy is allowed to see -- written to ``result.json`` and the log header.
OBS_SPEC_NONE = "none (a tape/constant takeover builds no observation)"
POLICY_INPUT_NONE = "none (the tape and constant arms evaluate no policy)"
POLICY_INPUT_CHECKPOINT = (
    "play 4x112x96 (4 consecutive frames, /255) + hud 1x56x160 + act_hist 8x18 "
    "(one-hot of the policy's own last 8 decisions); pixels and its own action "
    "history only, no privileged state")
#: ``ObservationBufferV3`` depth: the number of consecutive frames one decision sees.
OBS_STACK_FRAMES = 4
#: ``ActionHistoryBuffer`` depth: the number of decisions ``act_hist`` carries.
OBS_HISTORY_DECISIONS = 8
#: ``name -> (file, per-decision shape, dtype)`` for ``--save-obs``.  The dtype is
#: the *raw* octet plane the encoder produced: ``/255`` happens at the tensor
#: boundary in :meth:`ObservationStream.tensors`, never here (no re-normalization).
OBS_PLANES: tuple[tuple[str, str, tuple[int, ...], str], ...] = (
    ("play", "play96x112.npy", (OBS_STACK_FRAMES, 112, 96), "uint8"),
    ("hud", "hud.npy", (56, 160), "uint8"),
    ("act_hist", "act_hist.npy", (OBS_HISTORY_DECISIONS, N_CAND), "uint8"),
)
#: The pixel readback the obs-v3 encoder expects (``SharedBlock``/``PIXELS``).
READBACK_SHAPE = (480, 640, 4)
#: ``EnemyManager.hpp:69-71``: the BOSS bit (the same flag
#: ``touhou_rl.v400.teacher.engine`` exports; the gym protocol only exports
#: ``ENEMY_FLAG_ACTIVE``/``_COLLISION``).
ENEMY_FLAG_BOSS = 1 << 1
#: The largest page the frozen client accepts (``HarnessClient.enemies``: 1..16),
#: i.e. the same page ``v400_student_eval.py`` reads for its diagnostics.
ENEMIES_PAGE_LIMIT = 16

EXIT_OK = 0
EXIT_ENTRY_FAILED = 1
EXIT_TECHNICAL = 2
#: Exit 3 was "``--arm checkpoint`` is not implemented".  The arm is implemented
#: now, so nothing returns 3; the value is kept so an old caller can still name it.
EXIT_RETIRED_NOT_IMPLEMENTED = 3
EXIT_REFUSED = 4
EXIT_MUTE = 5

#: ``th08.cfg`` is a fixed 60-byte ``GameConfiguration`` (``Supervisor.hpp``).
CONFIG_SIZE = 0x3C
#: The four audio bytes ``AGENTS.md`` section 9 pins; 0 = silent.
MUTE_BYTES: tuple[tuple[str, int, bool], ...] = (
    ("musicMode", 0x1F, False),
    ("playSounds", 0x20, False),
    ("musicVolume", 0x27, True),
    ("sfxVolume", 0x28, True),
)

#: Stop reasons ``result.json`` may carry (a closed vocabulary).
STOP_REASONS = (
    "frame_cap",
    "legal_death",
    "game_over",
    "stage_clear",
    "engine_exit",
    "frame_shortfall",
    "tape_exhausted",
    "entry_hash_mismatch",
    "frame_relation_conflict",
    "entry_frame_invalid",
    "entry_frame_uncovered",
    "tape_invalid",
    "engine_sha_mismatch",
    "engine_missing",
    "mute_refused",
    "data_dir_missing",
    "decision_log_parent_missing",
    "decision_log_error",
    "arm_invalid",
    "constant_cand_invalid",
    "save_obs_invalid",
    "save_obs_dir_not_empty",
    "save_obs_error",
    "checkpoint_missing",
    "checkpoint_sha_mismatch",
    "obs_spec_mismatch",
    "policy_load_failed",
    "pixels_unavailable",
    "policy_output_invalid",
    "seed_invalid",
    "cap_frames_invalid",
    "reset_not_ready",
    "protocol_error",
    "not_implemented",
    "technical_error",
)

#: The stop reasons that mean "the window ran as asked".  Everything else
#: (engine exit, frame shortfall, tape end) is a technical failure of the window.
HANDOVER_OK_STOPS = ("frame_cap", "legal_death", "game_over", "stage_clear")

#: Keys every ``result.json`` carries, whatever happened (refusals included).
RESULT_REQUIRED_KEYS = (
    "schema",
    "arm",
    "seed",
    "entry_frame",
    "entry_frame_requested",
    "entry_hash",
    "entry_verified",
    "mismatch_slot",
    "mismatch",
    "prefix",
    "entry_state",
    "frames",
    "frames_prefix",
    "frames_takeover",
    "cap_frames",
    "decisions",
    "stop",
    "deaths",
    "wall_s",
    "engine",
    "tape",
    "data_dir",
    "mute",
    "reset",
    "decision_log",
    "save_obs",
    "boss_summary",
    "failure",
    "exit_code",
    "checkpoint",
    "checkpoint_sha256",
    "obs_spec_version",
    "obs_spec_digest",
    "policy_input",
    "frames_per_decision",
    "checkpoint_arm",
    "pixels",
    "renders",
    "policy_calls",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class TakeoverError(RuntimeError):
    """Base class of every expected failure of this tool."""


class PreflightRefused(TakeoverError):
    """The run was refused before an engine was started."""

    def __init__(self, message: str, *, stop: str = "technical_error") -> None:
        super().__init__(message)
        self.stop = str(stop)


class TapeError(PreflightRefused):
    """The tape does not satisfy the 2-frame contract this runner replays."""

    def __init__(self, message: str, *, stop: str = "tape_invalid") -> None:
        super().__init__(message, stop=stop)


class DecisionLogError(TakeoverError):
    """The ``--decision-log`` telemetry could not be written."""


class ObsSaveError(TakeoverError):
    """The ``--save-obs`` corpus could not be written.

    Raised **after** the files are opened and while the window runs: the run stops
    with a non-zero exit instead of continuing into a corpus that quietly lost a
    decision (a hole in ``slots.jsonl`` would misalign the planes).
    """


class EngineProtocolError(TakeoverError):
    """The engine replied something a takeover cannot interpret."""


class StepFailed(EngineProtocolError):
    """One ``STEP`` did not advance exactly one frame."""

    def __init__(self, *, mask: int, frame: int, advanced: int, exit_flag: bool,
                 game_over: bool) -> None:
        self.mask = int(mask)
        self.frame = int(frame)
        self.advanced = int(advanced)
        self.exit_flag = bool(exit_flag)
        self.game_over = bool(game_over)
        self.stop = "engine_exit" if self.exit_flag else "frame_shortfall"
        super().__init__(
            f"STEP {self.mask} 1 into frame {frame}: advanced={advanced}, "
            f"exit={exit_flag}, game_over={game_over}")


class EntryMismatch(TakeoverError):
    """A state hash did not come back as the tape recorded it."""

    def __init__(self, *, slot: Mapping[str, Any], expected: str, actual: str,
                 frame: int, frames_report: int | None, kind: str = "slot") -> None:
        self.slot = dict(slot)
        self.expected = str(expected)
        self.actual = str(actual)
        self.frame = int(frame)
        self.frames_report = frames_report
        self.kind = str(kind)
        super().__init__(
            f"slot {self.slot.get('slot_id')} (decision_frame "
            f"{self.slot.get('decision_frame')}, frame {frame}): the engine's "
            f"HASH is {self.actual!r}, the tape records {self.expected!r}")

    def record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "slot": self.slot.get("slot_id"),
            "decision_frame": self.slot.get("decision_frame"),
            "frame": self.frame,
            "expected": self.expected,
            "actual": self.actual,
            "hash_frames": self.frames_report,
        }


class PixelsUnavailable(TakeoverError):
    """No pixel bytes, or bytes that are not a fresh obs-v3 readback.

    Raised instead of substituting a zero image, a checksum reply or the previous
    frame: a policy fed any of those would be measured against an observation the
    contract does not describe.
    """


class PolicyOutputError(TakeoverError):
    """The policy emitted something the 18-candidate action contract cannot express."""



# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    """sha256 of one file, read in 1 MiB blocks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                      default=str)
    target.write_text(text + "\n", encoding="utf-8")
    return target


def expand_commit(parts: Iterable[tuple[int, int]]) -> list[int]:
    """``[(mask, frames)]`` -> one mask per frame (never a local rule)."""

    masks: list[int] = []
    for mask, frames in parts:
        masks.extend([int(mask)] * int(frames))
    return masks


def _payload(reading: Any) -> dict[str, Any]:
    """A dataclass reading (``PlayerSnapshot``/``HudSnapshot``) as a JSON dict."""

    if isinstance(reading, Mapping):
        return dict(reading)
    if dataclasses.is_dataclass(reading) and not isinstance(reading, type):
        return dataclasses.asdict(reading)
    raise EngineProtocolError(
        f"reading {type(reading).__name__} is neither a mapping nor a dataclass")


def enemies_summary(rows: Sequence[Any]) -> dict[str, Any]:
    """Diagnostics-only summary of one ``ENEMIES`` page (never a scoring source)."""

    decoded = [_payload(row) for row in rows]
    lives = [int(row.get("life", 0)) for row in decoded]
    bosses = [row for row in decoded
              if int(row.get("flags1", 0)) & ENEMY_FLAG_BOSS]
    return {
        "count": len(decoded),
        "slots": [int(row.get("slot", -1)) for row in decoded],
        "boss_count": len(bosses),
        "boss_slots": [int(row.get("slot", -1)) for row in bosses],
        "life_min": min(lives) if lives else None,
        "life_max": max(lives) if lives else None,
        "rows": decoded,
    }


def enemies_reading(rows: Sequence[Any], *,
                    total: int | None = None,
                    limit: int = ENEMIES_PAGE_LIMIT) -> dict[str, Any]:
    """The per-decision ``enemies`` field of one log row (referee side only).

    ``count`` is the rows this page returned, ``boss_count``/``boss_slots`` the
    rows carrying ``ENEMY_FLAG_BOSS``, and ``boss_life`` the largest ``life``
    among them (``0`` when the page lists no boss, the same convention
    ``tools/rl/v400_student_eval.py`` uses).  ``truncated`` is ``True`` when the
    engine reports more active enemies than the page returned -- with the
    engine's own total unknown, a full page is still marked truncated, because a
    page that *might* hide a row must never claim to be a census.

    This is telemetry, not an input: no arm reads it, and judging uses ``HAZARDS``
    row identity (slot + ``flags1`` + signed life) precisely because ``ENEMIES``
    is paginated.
    """

    decoded = [_payload(row) for row in rows]
    bosses = [row for row in decoded
              if int(row.get("flags1", 0)) & ENEMY_FLAG_BOSS]
    lives = [int(row.get("life", 0)) for row in bosses]
    if total is None:
        truncated = len(decoded) >= int(limit)
    else:
        truncated = int(total) > len(decoded)
    return {
        "available": True,
        "count": len(decoded),
        "boss_count": len(bosses),
        "boss_life": int(max(lives, default=0)),
        "boss_slots": [int(row.get("slot", -1)) for row in bosses],
        "truncated": bool(truncated),
    }


def probe_enemies(client: Any, *, limit: int = ENEMIES_PAGE_LIMIT) -> dict[str, Any]:
    """One ``ENEMIES`` page as a decision-row reading -- never raises.

    A refused page, an engine whose ``enemy_count`` is refused, and a page that
    cannot be decoded all still produce a dict: the first two are recorded as
    ``{"available": false, "reason": <the engine's text>}`` so the row can be
    written and the refusal stays visible; an undecodable page is recorded the
    same way with the decoding error.  Nothing is ever filled in with a zero.
    """

    try:
        rows = list(client.enemies(0, int(limit)))
    except Exception as exc:  # noqa: BLE001 - a refused probe is recorded, not raised
        return {"available": False, "reason": str(exc) or type(exc).__name__}
    total: int | None = None
    counter = getattr(client, "enemy_count", None)
    if callable(counter):
        try:
            total = int(counter())
        except Exception:  # noqa: BLE001 - the page is still a measurement
            total = None
    try:
        return enemies_reading(rows, total=total, limit=int(limit))
    except Exception as exc:  # noqa: BLE001 - an undecodable page is a refusal too
        return {"available": False,
                "reason": f"{type(exc).__name__}: {exc}"}


def boss_summary(available: bool = False, first_boss_frame: int | None = None,
                 boss_life_first: int | None = None,
                 boss_zero_life_frame: int | None = None,
                 boss_life_last: int | None = None,
                 samples: int = 0) -> dict[str, Any]:
    """The ``result.json`` shape of the aggregate of the per-decision readings."""

    return {
        "available": bool(available),
        "first_boss_frame": first_boss_frame,
        "boss_life_first": boss_life_first,
        "boss_zero_life_frame": boss_zero_life_frame,
        "boss_life_last": boss_life_last,
        "samples": int(samples),
    }


class BossTracker:
    """Aggregate the per-decision ``enemies`` readings into ``boss_summary``.

    The ruler is frozen here, not inferred at report time:

    ``first_boss_frame`` / ``boss_life_first``
        the first logged decision whose reading lists a boss
        (``boss_count > 0``) and that reading's ``boss_life``;
    ``boss_zero_life_frame``
        the first reading that lists a boss **and** reads ``boss_life == 0`` -- a
        listed boss with no life left.  A page with no boss at all is *not* a
        zero-life observation, even though ``boss_life`` is ``0`` there only
        because the maximum over an empty list is 0;
    ``boss_life_last``
        the last *available* reading's ``boss_life`` (so a window can be read
        without replaying the log), ``None`` when no reading was available;
    ``samples``
        how many readings were available.

    ``frame`` is the row's own ``frame`` (the decision's start frame; the reading
    is taken after its two frames, exactly like the row's ``deaths``/``lives``),
    so ``first_boss_frame`` joins straight onto the log.
    """

    def __init__(self) -> None:
        self.samples = 0
        self.unavailable = 0
        self.first_boss_frame: int | None = None
        self.boss_life_first: int | None = None
        self.boss_zero_life_frame: int | None = None
        self.boss_life_last: int | None = None

    def observe(self, frame: int, reading: Mapping[str, Any] | None) -> None:
        """Fold one decision's reading in (a refused reading is counted, not used)."""

        if not reading or reading.get("available") is not True:
            self.unavailable += 1
            return
        self.samples += 1
        life = int(reading.get("boss_life", 0))
        bosses = int(reading.get("boss_count", 0))
        self.boss_life_last = life
        if bosses <= 0:
            return
        if self.first_boss_frame is None:
            self.first_boss_frame = int(frame)
            self.boss_life_first = life
        if life == 0 and self.boss_zero_life_frame is None:
            self.boss_zero_life_frame = int(frame)

    def summary(self) -> dict[str, Any]:
        return boss_summary(available=self.samples > 0,
                            first_boss_frame=self.first_boss_frame,
                            boss_life_first=self.boss_life_first,
                            boss_zero_life_frame=self.boss_zero_life_frame,
                            boss_life_last=self.boss_life_last,
                            samples=self.samples)


def readings(state: Mapping[str, Any]) -> dict[str, Any]:
    """The counters a decision row carries, from the decoded ``STATE``."""

    return {
        "deaths": float(state["deaths"]),
        "lives": float(state["lives_remaining"]),
        "bombs": float(state["bombs_remaining"]),
        "score": int(state["score"]),
        "stage": int(state["stage"]),
        "frame": int(state["frame"]),
    }


def read_state(client: Any) -> dict[str, Any]:
    """One ``STATE`` reply decoded to the 25 named fields (f0..f24)."""

    from touhou_rl.v400.labels.state import state_from_reply  # noqa: PLC0415

    reply = client.request("STATE")
    try:
        return state_from_reply(reply)
    except ValueError as exc:
        raise EngineProtocolError(f"STATE reply is unusable: {exc}") from exc


def read_events(client: Any) -> tuple[list[str], str | None]:
    """``EVENTS`` names, or ``([], reason)`` -- telemetry never aborts a run."""

    try:
        frame = client.events()
    except Exception as exc:  # noqa: BLE001 - a refused probe is recorded
        return [], f"{type(exc).__name__}: {exc}"
    names = getattr(frame, "names", None)
    if names is None:
        return [], f"EVENTS reply has no bit names: {type(frame).__name__}"
    return [str(name) for name in names], None


# ---------------------------------------------------------------------------
# The four mute bytes (AGENTS.md section 9)
# ---------------------------------------------------------------------------
def read_mute(data_dir: str | Path) -> dict[str, Any]:
    """Decode ``<data-dir>/th08.cfg``'s four audio bytes.

    ``th08.cfg`` is a fixed 60-byte ``GameConfiguration``: text is silently
    discarded by the game, so the only honest reading is the struct.  A missing
    or wrongly sized file is reported as ``available: false`` -- never as "silent".
    """

    path = Path(data_dir) / "th08.cfg"
    record: dict[str, Any] = {"path": path.as_posix(), "available": False,
                              "muted": None, "size": None, "sha256": None,
                              "bytes": {}}
    if not path.is_file():
        record["reason"] = "th08.cfg is missing (run tools/rl/v400_make_cfg.py)"
        return record
    data = path.read_bytes()
    record["size"] = len(data)
    record["sha256"] = sha256_bytes(data)
    if len(data) != CONFIG_SIZE:
        record["reason"] = (f"th08.cfg is {len(data)} bytes, the game reads "
                            f"exactly {CONFIG_SIZE}")
        return record
    values: dict[str, int] = {}
    for name, offset, signed in MUTE_BYTES:
        values[name] = (struct.unpack_from("<b", data, offset)[0] if signed
                        else data[offset])
    record["bytes"] = values
    record["available"] = True
    record["muted"] = all(value == 0 for value in values.values())
    if not record["muted"]:
        record["reason"] = (f"audible configuration: {values} (musicMode/"
                            "playSounds/both volumes must all be 0)")
    return record


# ---------------------------------------------------------------------------
# The tape
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TakeoverSlot:
    """One ``slots.jsonl`` row, as the replay needs it."""

    slot_id: int
    decision_frame: int
    mask: int
    cand: int
    bomb: int
    hash_pre: str
    hash_post: str
    duration_frames: int = DECISION_FRAMES

    @property
    def decision_index(self) -> int:
        """Tape decision index: ``decision i`` owns STEP frames ``2i+1``/``2i+2``."""

        return self.decision_frame // DECISION_FRAMES

    def as_dict(self) -> dict[str, Any]:
        return {"slot_id": self.slot_id, "decision_frame": self.decision_frame,
                "mask": self.mask, "cand": self.cand, "hash_pre": self.hash_pre,
                "hash_post": self.hash_post}


@dataclass(frozen=True)
class LeadInDecision:
    """One of the 8 decisions that build frames 0..15 (slot 0's ``history``)."""

    decision_index: int
    frame: int
    cand: int


def _hash_token(row: Mapping[str, Any], key: str, where: str) -> str:
    value = str(row.get(key) or "").strip().lower()
    if not value:
        raise TapeError(f"{where}: {key} is missing")
    return value


def load_tape(path: str | Path) -> dict[str, Any]:
    """Read and validate one ``slots.jsonl``; refuse anything off the contract.

    Every rejection names the slot: this file is the *only* source of the entry
    frame, so a silently accepted anomaly would corrupt every later comparison.
    """

    from touhou_rl.v400.gym.actions import cand_to_mask  # noqa: PLC0415
    from touhou_rl.v400.teacher.shoot import release_due  # noqa: PLC0415

    target = Path(path).expanduser()
    if not target.is_file():
        raise TapeError(f"the tape is missing: {target.as_posix()}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
            target.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError as exc:
            raise TapeError(
                f"{target.as_posix()}: line {line_number} is not JSON: {exc}",
                stop="tape_invalid") from exc
    if not rows:
        raise TapeError(f"{target.as_posix()} has no slots")
    for index, row in enumerate(rows):
        if "decision_frame" not in row:
            raise TapeError(f"{target.as_posix()}: row {index} has no decision_frame")
    rows.sort(key=lambda row: int(row["decision_frame"]))

    seeds = sorted({int(row["episode_seed"]) for row in rows
                    if row.get("episode_seed") is not None})
    slots: list[TakeoverSlot] = []
    for index, row in enumerate(rows):
        where = f"{target.name}: row {index}"
        try:
            frame = int(row["decision_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TapeError(f"{where}: decision_frame is not an integer") from exc
        duration = int(row.get("duration_frames", DECISION_FRAMES)
                       or DECISION_FRAMES)
        where = f"{target.name}: slot {row.get('slot_id', index)} (frame {frame})"
        if duration != DECISION_FRAMES:
            raise TapeError(f"{where}: duration_frames is {duration}, the replay "
                            f"contract is {DECISION_FRAMES}")
        if frame % DECISION_FRAMES:
            raise TapeError(f"{where}: decision_frame is odd, the tape is on a "
                            f"{DECISION_FRAMES}-frame grid")
        expected = TAPE_FIRST_FRAME + DECISION_FRAMES * index
        if frame != expected:
            raise TapeError(f"{where}: the 2-frame grid is broken, expected frame "
                            f"{expected} at row {index}")
        try:
            mask = int(row["mask"])
            executed = row.get("executed") or {}
            cand = int(row["executed_cand"] if "executed_cand" in row
                       else executed["cand"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TapeError(f"{where}: mask/executed_cand is not an integer") from exc
        bomb = int(row.get("bomb", 0) or 0)
        if int(cand_to_mask(cand, bomb)) != mask:
            raise TapeError(
                f"{where}: recorded mask {mask} is not cand_to_mask({cand}, bomb="
                f"{bomb}) = {int(cand_to_mask(cand, bomb))}, so the row cannot be "
                "replayed")
        slots.append(TakeoverSlot(
            slot_id=int(row.get("slot_id", index)), decision_frame=frame, mask=mask,
            cand=cand, bomb=bomb,
            hash_pre=_hash_token(row, "hash_pre", where),
            hash_post=_hash_token(row, "hash_post", where),
            duration_frames=duration))

    chain_breaks: list[dict[str, Any]] = []
    for left, right in zip(slots, slots[1:]):
        if left.hash_post != right.hash_pre:
            chain_breaks.append({"slot": left.slot_id, "next_slot": right.slot_id,
                                 "left_hash_post": left.hash_post,
                                 "right_hash_pre": right.hash_pre})
    if chain_breaks:
        first = chain_breaks[0]
        raise TapeError(
            f"{target.as_posix()}: the hash chain is broken at slot {first['slot']} "
            f"(hash_post {first['left_hash_post']} != slot {first['next_slot']}'s "
            f"hash_pre {first['right_hash_pre']}); {len(chain_breaks)} break(s)")

    lead_in: list[LeadInDecision] = []
    first = rows[0]
    history_frames = list(first.get("history_frames") or [])
    history = list(first.get("history") or [])
    if history_frames or history:
        if len(history_frames) != len(history):
            raise TapeError(
                f"{target.as_posix()}: slot 0 has {len(history_frames)} "
                f"history_frames but {len(history)} history entries")
        for position, (frame, cand) in enumerate(zip(history_frames, history)):
            if int(frame) != DECISION_FRAMES * position:
                raise TapeError(
                    f"{target.as_posix()}: lead-in decision {position} sits at frame "
                    f"{frame}, expected {DECISION_FRAMES * position}")
            lead_in.append(LeadInDecision(decision_index=position, frame=int(frame),
                                          cand=int(cand)))
        if DECISION_FRAMES * len(lead_in) != TAPE_FIRST_FRAME:
            raise TapeError(
                f"{target.as_posix()}: the lead-in covers "
                f"{DECISION_FRAMES * len(lead_in)} frames, but the first slot is at "
                f"frame {TAPE_FIRST_FRAME}")

    releases = sum(1 for slot in slots
                   if release_due(slot.decision_index, DEFAULT_PULSE_PERIOD))
    return {
        "path": target.as_posix(),
        "sha256": sha256_file(target),
        "slots": slots,
        "lead_in": lead_in,
        "count": len(slots),
        "first_frame": slots[0].decision_frame,
        "last_frame": slots[-1].decision_frame,
        "last_decision_index": slots[-1].decision_index,
        "episode_seeds": seeds,
        "chain_breaks": 0,
        "pulse_releases_at_48": releases,
        "classification": "measured",
    }


def slot_masks(slot: TakeoverSlot, period: int) -> list[int]:
    """The two STEP masks the recorded executor sent for this slot."""

    from touhou_rl.v400.teacher.shoot import commit_frames, release_due  # noqa: PLC0415

    release = bool(release_due(slot.decision_index, int(period)))
    masks = expand_commit(commit_frames(slot.cand, bomb=slot.bomb,
                                        decision_frames=DECISION_FRAMES,
                                        release_shoot=release))
    if len(masks) != DECISION_FRAMES:
        raise TapeError(f"slot {slot.slot_id}: commit_frames expanded to "
                        f"{len(masks)} masks, expected {DECISION_FRAMES}")
    if masks[0] != slot.mask:
        raise TapeError(
            f"slot {slot.slot_id}: the derived first mask {masks[0]} is not the "
            f"recorded mask {slot.mask}; --shoot-pulse-period {period} does not "
            "match the phase the tape was recorded with")
    return masks


def lead_in_masks(decision: LeadInDecision, period: int) -> list[int]:
    """The two STEP masks for one lead-in decision (frames 0..15)."""

    from touhou_rl.v400.teacher.shoot import commit_frames, release_due  # noqa: PLC0415

    release = bool(release_due(decision.decision_index, int(period)))
    masks = expand_commit(commit_frames(decision.cand,
                                        decision_frames=DECISION_FRAMES,
                                        release_shoot=release))
    if len(masks) != DECISION_FRAMES:
        raise TapeError(f"lead-in decision {decision.decision_index}: "
                        f"commit_frames expanded to {len(masks)} masks")
    return masks


def constant_masks(cand: int, decision_index: int, period: int) -> list[int]:
    """The two STEP masks the constant arm emits for one decision."""

    from touhou_rl.v400.teacher.shoot import commit_frames, release_due  # noqa: PLC0415

    release = bool(release_due(int(decision_index), int(period)))
    masks = expand_commit(commit_frames(int(cand), decision_frames=DECISION_FRAMES,
                                        release_shoot=release))
    if len(masks) != DECISION_FRAMES:
        raise PreflightRefused(
            f"constant cand {cand}: commit_frames expanded to {len(masks)} masks",
            stop="constant_cand_invalid")
    return masks


def checkpoint_masks(cand: int, bomb: int, decision_index: int,
                     period: int) -> list[int]:
    """The two STEP masks the checkpoint arm emits for one decision.

    The policy chooses only ``(cand, bomb)``.  The SHOOT release stays exogenous
    and identical to the other two arms: :func:`release_due` decides the phase and
    :func:`commit_frames` drops ``BTN_SHOOT`` from the decision's last frame.
    """

    from touhou_rl.v400.teacher.shoot import commit_frames, release_due  # noqa: PLC0415

    release = bool(release_due(int(decision_index), int(period)))
    masks = expand_commit(commit_frames(int(cand), bomb=int(bomb),
                                        decision_frames=DECISION_FRAMES,
                                        release_shoot=release))
    if len(masks) != DECISION_FRAMES:
        raise PolicyOutputError(
            f"cand {cand}/bomb {bomb}: commit_frames expanded to {len(masks)} masks, "
            f"expected {DECISION_FRAMES}")
    return masks


def entry_plan(tape: Mapping[str, Any], entry_frame: int) -> dict[str, Any]:
    """Everything the run does before control is handed over.

    ``F`` must be a slot boundary, so ``(F - 16) / 2`` tape slots are replayed
    **after** the 8 lead-in decisions.  The plan also carries the tape's own
    expectation for the entry hash (``hash_pre`` of the slot at ``F``), which is
    the same value as the last prefix slot's ``hash_post``.
    """

    frame = int(entry_frame)
    if frame % DECISION_FRAMES:
        raise PreflightRefused(
            f"--entry-frame {frame} is odd; the tape is on a {DECISION_FRAMES}-frame "
            "grid, so an entry frame is even", stop="entry_frame_invalid")
    if frame < TAPE_FIRST_FRAME:
        raise PreflightRefused(
            f"--entry-frame {frame} is before the tape's first recorded slot frame "
            f"{TAPE_FIRST_FRAME}", stop="entry_frame_invalid")
    slots: list[TakeoverSlot] = list(tape["slots"])
    by_frame = {slot.decision_frame: slot for slot in slots}
    if frame not in by_frame:
        raise PreflightRefused(
            f"--entry-frame {frame} is not a slot boundary of {tape['path']}: the "
            f"tape starts at {tape['first_frame']} and steps {DECISION_FRAMES}",
            stop="entry_frame_uncovered")
    if not tape["lead_in"] and frame == TAPE_FIRST_FRAME:
        raise PreflightRefused(
            f"--entry-frame {frame}: the tape records no lead-in decisions "
            "(slot 0 has no history_frames/history), so frame 16 is not reachable "
            "from RESET", stop="entry_frame_uncovered")
    entry_slot = by_frame[frame]
    prefix_slots = [slot for slot in slots if slot.decision_frame < frame]
    takeover_slots = [slot for slot in slots if slot.decision_frame >= frame]
    if entry_slot.decision_frame != TAPE_FIRST_FRAME + DECISION_FRAMES * len(prefix_slots):
        raise TapeError(
            "--entry-frame mapping is inconsistent: "
            f"{len(prefix_slots)} prefix slots but the entry slot is at frame "
            f"{entry_slot.decision_frame}")
    if not takeover_slots:
        raise PreflightRefused(
            f"--entry-frame {frame}: the tape has no slot at or after the entry, so "
            "there is nothing to hand over to", stop="entry_frame_uncovered")
    return {
        "entry_frame": frame,
        "entry_index": entry_slot.decision_index,
        "entry_slot": entry_slot,
        "prefix_slots": prefix_slots,
        "takeover_slots": takeover_slots,
        "lead_in": list(tape["lead_in"]),
        "anchor_hash": entry_slot.hash_pre,
        "lead_in_anchor_hash": slots[0].hash_pre,
        "decision_count": len(prefix_slots) + len(tape["lead_in"]),
    }


# ---------------------------------------------------------------------------
# The checkpoint arm: action history, pixels, obs-v3 actor
#
# Every construction below is the *same call the eval makes*, in the same order,
# so the tensors cannot drift away from ``tools/rl/v400_student_eval.py``:
#
#   buffer  = obs_spec_v3.ObservationBufferV3()
#   history = obs_spec_v2.ActionHistoryBuffer()
#   play, hud = obs_spec_v3.encode_planes_v3(pixels); buffer.push(play, hud)
#   arrays = buffer.arrays()                  # play (4,112,96), hud (1,56,160)
#   play_t = torch.from_numpy(arrays["play"][None]).float()
#   hud_t  = torch.from_numpy(arrays["hud"][None]).float()
#   hist_t = torch.from_numpy(history.rows()[None]).float()
#   cand, bomb = model.act(play_t, hud_t, hist_t)
#
# ``tests/test_v8_32_takeover.py`` pins that by running both paths on the same
# synthetic pixels and comparing the three tensors bit for bit.
# ---------------------------------------------------------------------------
def prefix_action_cands(plan: Mapping[str, Any]) -> list[int]:
    """The candidates the prefix executes, decisions ``0..entry-1``, tape order.

    The 8 lead-in decisions come from slot 0's ``history`` and every later one from
    a slot's ``executed_cand``; :func:`load_tape` has already checked each slot's
    recorded ``mask`` is ``cand_to_mask(executed_cand, bomb)``, so these are the
    actions the replayed frames actually took.
    """

    return ([int(decision.cand) for decision in plan["lead_in"]]
            + [int(slot.cand) for slot in plan["prefix_slots"]])


def seed_action_history(plan: Mapping[str, Any]) -> tuple[Any, list[int]]:
    """Seed the obs-v2 ``8x18`` history from the tape's own prefix actions.

    ``ActionHistoryBuffer`` keeps the newest eight decisions, so the first policy
    decision sees exactly the last eight candidates the tape executed.  Nothing
    after the entry is pushed here: the history at decision ``t`` must hold
    ``a[0..t-1]`` and can never contain the policy's own current action.
    """

    from touhou_rl.v400.student.obs_spec_v2 import ActionHistoryBuffer  # noqa: PLC0415

    cands = prefix_action_cands(plan)
    history = ActionHistoryBuffer()
    for cand in cands:
        if not 0 <= int(cand) < N_CAND:
            raise TapeError(
                f"the prefix's executed cand {cand} is outside 0..{N_CAND - 1}, so "
                "it cannot seed the 8x18 action history")
        history.push(int(cand))
    if history.pushes != len(cands):  # pragma: no cover - the buffer counts pushes
        raise TakeoverError(
            f"the action history took {history.pushes} pushes, expected {len(cands)}")
    return history, cands


def default_shm_name(prefix: str = "v832takeover") -> str:
    """A per-process shared-block name (the engine creates, we only attach)."""

    return f"{prefix}-{os.getpid()}-{int(time.time() * 1000) % 1_000_000}"


def pixel_array(value: Any) -> Any | None:
    """The RGBA array inside one pixel reading, or ``None`` when there are no bytes.

    Accepted forms: the array itself, a ``(header, array)`` pair, a mapping with
    ``"rgba"``/``"pixels"``, or an object with a ``.rgba`` array.  A checksum reply
    (what the frozen client's ``PIXELS`` command returns) matches none of them and
    is therefore *not* pixels.
    """

    import numpy as np  # noqa: PLC0415

    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        for key in ("rgba", "pixels"):
            candidate = value.get(key)
            if isinstance(candidate, np.ndarray):
                return candidate
        return None
    if isinstance(value, (tuple, list)) and value:
        last = value[-1]
        return last if isinstance(last, np.ndarray) else None
    candidate = getattr(value, "rgba", None)
    return candidate if isinstance(candidate, np.ndarray) else None


class PixelSource:
    """Fail-closed pixel readings for the checkpoint arm.

    Two sources, tried in this order; which one was used is written to
    ``result.json`` (``pixels.source``) so a run can never claim the shared block
    when it read something else:

    ``shm``
        ``client.attach_shm(name)`` + ``client.readback(1)`` + a mapped
        :class:`touhou_rl.v400.gym.shm.SharedBlock`.  This is the only path the
        frozen client has to pixel *bytes*: its ``PIXELS`` command returns an
        FNV-1a-64 checksum of a fresh readback, not an image.  ``READBACK 1`` is
        sent **before** ``RESET`` so every frame of the episode republishes the
        block.  Two gates run on every read: ``FLAG_PIXELS_VALID`` must be set and
        the header's frame counter must strictly advance since the previous read
        (a block that did not publish a new frame is refused, never reused).
    ``pixels()``
        a client whose ``pixels()`` returns the RGBA array -- the scripted clients
        the unit tests use.  A checksum reply is refused at :meth:`open`, before
        anything is stepped.
    """

    def __init__(self, client: Any, *, name: str | None = None,
                 block_factory: Callable[[str], Any] | None = None) -> None:
        self.client = client
        self.name = str(name or default_shm_name()).lstrip("/")
        self.block_factory = block_factory
        self.block: Any = None
        self.source: str | None = None
        self.problems: list[str] = []
        self.attach_reply: Any = None
        self.readback_reply: Any = None
        self.readback_error: str | None = None
        self.reads = 0
        self.last_frame: int | None = None
        self.probe_frames: list[int] = []

    # -- setup ----------------------------------------------------------
    def open(self) -> dict[str, Any]:
        """Attach the pixel source; raise :class:`PixelsUnavailable` if there is none."""

        self._open_shm()
        if self.block is None:
            self._probe_pixels_command()
        if self.block is None and self.source is None:
            raise PixelsUnavailable(
                "no pixels are available for the obs-v3 stack: "
                + "; ".join(self.problems or ["the client exposes no pixel source"]))
        return self.identity()

    def _open_shm(self) -> None:
        existing = getattr(self.client, "shm", None)
        if existing is not None:
            self.block = existing
            self.source = "shm"
            return
        attach = getattr(self.client, "attach_shm", None)
        if attach is None:
            self.problems.append("the client has no attach_shm()")
            return
        try:
            self.attach_reply = attach(self.name)
        except Exception as exc:  # noqa: BLE001 - no pixels is a refusal, not a crash
            self.problems.append(f"SHM attach failed: {type(exc).__name__}: {exc}")
            return
        reply = self.attach_reply
        if isinstance(reply, Mapping) and str(reply.get("active")) != "1":
            self.problems.append(f"SHM /{self.name} was refused: {dict(reply)}")
            return
        readback = getattr(self.client, "readback", None)
        if readback is not None:
            try:
                self.readback_reply = readback(True)
            except Exception as exc:  # noqa: BLE001
                self.readback_error = f"{type(exc).__name__}: {exc}"
                self.problems.append(f"READBACK 1 failed: {self.readback_error}")
                return
            if (isinstance(self.readback_reply, Mapping)
                    and str(self.readback_reply.get("readback")) != "1"):
                self.problems.append(
                    f"READBACK 1 was refused: {dict(self.readback_reply)}")
                return
        factory = self.block_factory
        if factory is None:
            from touhou_rl.v400.gym.shm import SharedBlock  # noqa: PLC0415

            factory = SharedBlock
        name = getattr(self.client, "shm_name", None) or self.name
        try:
            self.block = factory(name)
        except Exception as exc:  # noqa: BLE001
            self.problems.append(
                f"cannot map shared block {name!r}: {type(exc).__name__}: {exc}")
            return
        self.source = "shm"

    def _probe_pixels_command(self) -> None:
        reader = getattr(self.client, "pixels", None)
        if reader is None:
            self.problems.append("the client has no pixels()")
            return
        try:
            reply = reader()
        except Exception as exc:  # noqa: BLE001
            self.problems.append(f"PIXELS failed: {type(exc).__name__}: {exc}")
            return
        if pixel_array(reply) is None:
            self.problems.append(
                "client.pixels() returned no pixel bytes "
                f"({type(reply).__name__}: {reply!s:.160}); the frozen client's PIXELS "
                "command is a checksum of a fresh readback, so the checkpoint arm "
                "needs SHM + READBACK 1")
            return
        self.source = "pixels()"

    # -- payload --------------------------------------------------------
    def identity(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "shm_name": (None if self.source != "shm"
                         else str(getattr(self.block, "name", None) or self.name)),
            "attach": (dict(self.attach_reply)
                       if isinstance(self.attach_reply, Mapping) else None),
            "readback": (dict(self.readback_reply)
                         if isinstance(self.readback_reply, Mapping) else None),
            "readback_error": self.readback_error,
            "reads": int(self.reads),
            "probe_frames": list(self.probe_frames),
            "last_frame": self.last_frame,
            "problems": list(self.problems),
            "readback_shape": list(READBACK_SHAPE),
            "encoder": "touhou_rl.v400.student.obs_spec_v3.encode_planes_v3",
        }

    def _require_array(self, value: Any, *, where: str) -> Any:
        import numpy as np  # noqa: PLC0415

        if not isinstance(value, np.ndarray):
            raise PixelsUnavailable(
                f"{where}: the pixel reading is {type(value).__name__}, not an RGBA "
                "array")
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 4:
            raise PixelsUnavailable(
                f"{where}: the readback is {value.dtype} {value.shape}, obs-v3 needs "
                "uint8 HxWx4")
        if tuple(int(dim) for dim in value.shape) != READBACK_SHAPE:
            raise PixelsUnavailable(
                f"{where}: the readback is {value.shape}, the engine's readback is "
                f"{READBACK_SHAPE}; the obs-v3 encoder and the eval read the same "
                "640x480 frame")
        return value

    def read(self, *, elapsed_frames: int | None = None) -> tuple[Any, dict[str, Any]]:
        """One fresh pixel reading, or :class:`PixelsUnavailable`."""

        if self.block is not None:
            return self._read_shm(elapsed_frames=elapsed_frames)
        if self.source == "pixels()":
            return self._read_pixels_command(elapsed_frames=elapsed_frames)
        raise PixelsUnavailable(
            "the pixel source is not open: " + "; ".join(self.problems or ["no source"]))

    def _read_shm(self, *, elapsed_frames: int | None) -> tuple[Any, dict[str, Any]]:
        from touhou_rl.v400.gym.shm import FLAG_PIXELS_VALID  # noqa: PLC0415

        try:
            header = dict(self.block.header())
        except Exception as exc:  # noqa: BLE001
            raise PixelsUnavailable(
                f"cannot read the shared-block header: {type(exc).__name__}: {exc}"
            ) from exc
        flags = int(header.get("flags", 0) or 0)
        if not flags & FLAG_PIXELS_VALID:
            raise PixelsUnavailable(
                "the shared block holds no published pixels "
                f"(FLAG_PIXELS_VALID clear, flags=0x{flags:04x}, frame "
                f"{header.get('frame')}); READBACK 1 must be on before RESET so every "
                "STEP republishes the block")
        frame = header.get("frame")
        frame = None if frame is None else int(frame)
        if frame is not None and self.last_frame is not None and frame <= self.last_frame:
            raise PixelsUnavailable(
                f"the shared block still reports frame {frame} after a STEP whose "
                f"previous read was frame {self.last_frame}: the arm refuses to feed "
                "the policy a frame it has already seen")
        try:
            pixels = self.block.pixels()
        except Exception as exc:  # noqa: BLE001
            raise PixelsUnavailable(
                f"cannot read the shared-block pixels: {type(exc).__name__}: {exc}"
            ) from exc
        array = self._require_array(pixels, where=f"shared block frame {frame}")
        if frame is not None:
            self.last_frame = frame
        self.reads += 1
        return array, {"source": "shm", "frame": frame, "flags": flags,
                       "elapsed_frames": elapsed_frames}

    def _read_pixels_command(self, *,
                             elapsed_frames: int | None) -> tuple[Any, dict[str, Any]]:
        try:
            reply = self.client.pixels()
        except Exception as exc:  # noqa: BLE001
            raise PixelsUnavailable(
                f"client.pixels() failed at frame {elapsed_frames}: "
                f"{type(exc).__name__}: {exc}") from exc
        pixels = pixel_array(reply)
        if pixels is None:
            raise PixelsUnavailable(
                f"client.pixels() returned no pixel bytes at frame {elapsed_frames} "
                f"({type(reply).__name__}: {reply!s:.160})")
        self.reads += 1
        return self._require_array(pixels, where="client.pixels()"), {
            "source": "pixels()", "frame": None, "flags": None,
            "elapsed_frames": elapsed_frames}

    def probe(self, *, elapsed_frames: int) -> dict[str, Any]:
        """One gated read that is **discarded** -- no observation is built from it.

        The checkpoint arm probes after the prefix's first ``STEP`` so a broken
        readback fails within one frame instead of after thousands of replayed
        frames, and so the first observation is never a leftover buffer.
        """

        _pixels, header = self.read(elapsed_frames=elapsed_frames)
        self.probe_frames.append(int(elapsed_frames))
        return header

    def close(self, *, unlink: bool = True) -> None:
        block = self.block
        self.block = None
        if block is not None:
            try:
                block.close()
            except Exception:  # pragma: no cover - closing never masks a result
                pass
        if unlink and self.source == "shm" and self.attach_reply is not None:
            # Only the segment this run attached (its name carries pid + time).
            try:
                from touhou_rl.v400.gym.shm import SHM_DIR  # noqa: PLC0415

                for path in SHM_DIR.glob(f"{self.name}*"):
                    try:
                        path.unlink()
                    except OSError:
                        pass
            except Exception:  # pragma: no cover - cleanup never masks a result
                pass


class ObservationStream:
    """The obs-v3 play/HUD stack plus the obs-v2 action history for one window.

    ``push``/``arrays``/``rows`` are the frozen classes; this object only keeps the
    order the eval keeps (one push per rendered frame, one history push per
    decision, after the action was applied) and counts the renders.
    """

    def __init__(self, *, pixels: PixelSource, history: Any) -> None:
        from touhou_rl.v400.student import obs_spec_v3  # noqa: PLC0415

        self.pixels = pixels
        self.history = history
        self.buffer = obs_spec_v3.ObservationBufferV3()
        self.renders = 0
        self.frames: list[int] = []
        self.last_header: dict[str, Any] | None = None

    def render(self, elapsed_frames: int) -> None:
        """Read the frame the last ``STEP`` drew and push it into the stack."""

        from touhou_rl.v400.student import obs_spec_v3  # noqa: PLC0415

        pixels, header = self.pixels.read(elapsed_frames=elapsed_frames)
        try:
            play, hud = obs_spec_v3.encode_planes_v3(pixels)
        except Exception as exc:  # noqa: BLE001 - an unencodable readback is no pixels
            raise PixelsUnavailable(
                f"the readback at frame {elapsed_frames} cannot be encoded as obs-v3: "
                f"{type(exc).__name__}: {exc}") from exc
        self.buffer.push(play, hud)
        self.renders += 1
        self.frames.append(int(elapsed_frames))
        self.last_header = dict(header)

    def arrays(self) -> dict[str, Any]:
        return self.buffer.arrays()

    def planes(self) -> dict[str, Any]:
        """The raw ``0..255`` planes behind :meth:`tensors`, for ``--save-obs``.

        They are read from the same buffer state ``tensors()`` used -- the play
        stack, the HUD strip and the action history -- so nothing is re-rendered
        (the pixel source is not touched) and nothing is re-normalized (the frozen
        ``normalize_play``/``normalize_hud`` run in :meth:`tensors`, not here).
        """

        import numpy as np  # noqa: PLC0415

        return {"play": self.buffer.stack(), "hud": self.buffer.hud(),
                "act_hist": np.asarray(self.history_rows(), dtype=np.uint8)}

    def history_rows(self) -> Any:
        return self.history.rows()

    def push_action(self, cand: int) -> None:
        """Push one decision's candidate *after* its frames were applied."""

        self.history.push(int(cand))

    def tensors(self) -> tuple[Any, Any, Any]:
        """The three tensors ``model.act`` takes, spelled exactly as the eval does."""

        import torch  # noqa: PLC0415

        arrays = self.arrays()
        play_t = torch.from_numpy(arrays["play"][None]).float()
        hud_t = torch.from_numpy(arrays["hud"][None]).float()
        hist_t = torch.from_numpy(self.history_rows()[None]).float()
        return play_t, hud_t, hist_t


@dataclass
class LoadedPolicy:
    """The obs-v3 actor and the identity ``result.json`` must carry."""

    model: Any
    payload: Mapping[str, Any]
    path: Path
    sha256: str
    sha_expected: str
    sha_matches: bool
    obs_spec_version: str
    obs_spec_digest: str
    device: str
    arch: str
    parameters: int | None

    def identity(self) -> dict[str, Any]:
        return {
            "arm": "checkpoint",
            "checkpoint": self.path.as_posix(),
            "checkpoint_sha256": self.sha256,
            "checkpoint_sha256_expected": self.sha_expected,
            "checkpoint_sha_matches": self.sha_matches,
            "checkpoint_bytes": int(self.path.stat().st_size),
            "obs_spec_version": self.obs_spec_version,
            "obs_spec_digest": self.obs_spec_digest,
            "obs_spec_digest_required": OBS_SPEC_V3_DIGEST,
            "policy_input": POLICY_INPUT_CHECKPOINT,
            "frames_per_decision": DECISION_FRAMES,
            "obs_stack_frames": OBS_STACK_FRAMES,
            "history_shape": [8, N_CAND],
            "history_seed_source": "the tape's executed_cand for decisions 0..entry-1",
            "device": self.device,
            "arch": self.arch,
            "parameters": self.parameters,
            "loader": "touhou_rl.v400.student.model_v3.load_checkpoint_v3",
            "action_contract": "cand 0..17 (focus*9+move) + bomb 0..1",
            "shoot_pulse": "exogenous: release_due(decision_index, period)",
        }


def default_policy_loader(config: "TakeoverConfig") -> tuple[Any, Mapping[str, Any]]:
    """``load_checkpoint_v3`` on the configured path/device (the eval's loader)."""

    from touhou_rl.v400.student.model_v3 import load_checkpoint_v3  # noqa: PLC0415

    model, payload = load_checkpoint_v3(config.checkpoint, device=str(config.device))
    return model, payload


def load_policy(config: "TakeoverConfig", *,
                loader: Callable[["TakeoverConfig"], Any] | None = None
                ) -> "LoadedPolicy":
    """Resolve, sha-gate, load and obs-spec-gate the checkpoint arm's actor.

    Nothing is started and no result is written before this returns, so a wrong
    path, a wrong sha or another observation contract is refused before an engine
    exists.  ``loader`` exists so the offline tests can inject a scripted actor;
    the sha gate and the obs-spec gate run in front of it and cannot be skipped.
    """

    from touhou_rl.v400.student.obs_spec_v3 import (  # noqa: PLC0415
        ObsSpecV3Mismatch,
        obs_spec_v3_digest,
    )

    if config.checkpoint is None:
        raise PreflightRefused(
            "--arm checkpoint needs --checkpoint PATH (the obs-v3 actor to run)",
            stop="checkpoint_missing")
    path = Path(config.checkpoint).expanduser()
    if not path.is_file():
        raise PreflightRefused(
            f"--checkpoint {path.as_posix()} is not a file", stop="checkpoint_missing")
    digest = sha256_file(path)
    expected = str(config.checkpoint_sha256 or "").strip().lower()
    if expected and digest != expected:
        raise PreflightRefused(
            f"--checkpoint {path.as_posix()} has sha256 {digest}, the expected value "
            f"is {expected}", stop="checkpoint_sha_mismatch")
    try:
        model, payload = (loader or default_policy_loader)(config)
    except ObsSpecV3Mismatch as exc:
        raise PreflightRefused(
            f"--checkpoint {path.as_posix()} was written under another observation "
            f"contract: {exc}", stop="obs_spec_mismatch") from exc
    except Exception as exc:  # noqa: BLE001 - a wrong file is a refusal
        raise PreflightRefused(
            f"cannot load --checkpoint {path.as_posix()}: {type(exc).__name__}: {exc}",
            stop="policy_load_failed") from exc
    if not isinstance(payload, Mapping):
        raise PreflightRefused(
            f"--checkpoint {path.as_posix()} loaded without a payload mapping "
            f"({type(payload).__name__})", stop="policy_load_failed")
    live = obs_spec_v3_digest()
    if live != OBS_SPEC_V3_DIGEST:
        raise PreflightRefused(
            f"this checkout's obs-v3 digest is {live}, the takeover arm is pinned to "
            f"{OBS_SPEC_V3_DIGEST}; refusing to feed a policy an observation contract "
            "that moved under it", stop="obs_spec_mismatch")
    version = str(payload.get("obs_spec_version") or "")
    spec_digest = str(payload.get("obs_spec_digest") or "")
    if version != OBS_SPEC_V3_VERSION:
        raise PreflightRefused(
            f"--checkpoint {path.as_posix()} declares obs spec version {version!r}; "
            f"the checkpoint arm requires {OBS_SPEC_V3_VERSION!r}",
            stop="obs_spec_mismatch")
    if spec_digest != OBS_SPEC_V3_DIGEST:
        raise PreflightRefused(
            f"--checkpoint {path.as_posix()} declares obs spec digest {spec_digest!r}; "
            f"the checkpoint arm requires {OBS_SPEC_V3_DIGEST!r}",
            stop="obs_spec_mismatch")
    if not callable(getattr(model, "act", None)):
        raise PreflightRefused(
            f"the object loaded from {path.as_posix()} has no act(play, hud, "
            "act_hist); the checkpoint arm needs an obs-v3 actor",
            stop="policy_load_failed")
    model_config = payload.get("model")
    arch = str(model_config.get("arch", "")) if isinstance(model_config, Mapping) else ""
    recorded = payload.get("parameters")
    return LoadedPolicy(
        model=model,
        payload=payload,
        path=path,
        sha256=digest,
        sha_expected=expected,
        sha_matches=(not expected) or digest == expected,
        obs_spec_version=version,
        obs_spec_digest=spec_digest,
        device=str(config.device),
        arch=arch,
        parameters=(None if recorded is None else int(recorded)),
    )


def checkpoint_frame_sink(*, stream: ObservationStream, pixel_source: PixelSource,
                          entry_frame: int) -> Callable[[int], None]:
    """The prefix observer: gate on frame 1, render the last ``OBS_STACK`` frames.

    ``ObservationBufferV3`` keeps only the newest four frames, so frames
    ``F-3 .. F`` are the ones the first policy decision sees; rendering an earlier
    frame would be evicted before the entry and is not read.
    """

    first = int(entry_frame) - OBS_STACK_FRAMES + 1

    def sink(frames: int) -> None:
        if int(frames) == 1:
            pixel_source.probe(elapsed_frames=int(frames))
        if int(frames) >= first:
            stream.render(int(frames))

    return sink


# ---------------------------------------------------------------------------
# The decision log (aligned with tools/rl/v400_student_eval.py)
# ---------------------------------------------------------------------------
class DecisionLog:
    """Append-only JSONL sink; every line is flushed as it is written."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        parent = self.path.parent
        if not parent.is_dir():
            raise DecisionLogError(
                f"--decision-log parent directory does not exist: {parent} (it is "
                "never created implicitly)")
        try:
            self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise DecisionLogError(
                f"cannot open decision log {self.path}: {exc}") from exc
        self.rows = 0

    def write(self, row: Mapping[str, Any]) -> int:
        line = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)
        try:
            self._handle.write(line + "\n")
            self._handle.flush()
        except (OSError, ValueError) as exc:
            raise DecisionLogError(
                f"decision log write failed for {self.path}: {exc}") from exc
        self.rows += 1
        return self.rows

    def close(self) -> None:
        if self._handle.closed:
            return
        try:
            self._handle.close()
        except OSError as exc:
            raise DecisionLogError(
                f"decision log close failed for {self.path}: {exc}") from exc


def decision_header(*, arm: str, seed: int, entry_frame: int, entry_hash: str,
                    reset: Mapping[str, Any], tape: Mapping[str, Any],
                    engine: Mapping[str, Any],
                    config: "TakeoverConfig",
                    checkpoint: str = "",
                    checkpoint_sha256: str = "",
                    obs_spec_version: str = OBS_SPEC_NONE,
                    obs_spec_digest: str = "",
                    policy_input: str = POLICY_INPUT_NONE) -> dict[str, Any]:
    """First line of the log: the arm, the seed **and** the entry identity.

    The key set is a superset of ``v400-student-eval-decision-log/1`` (``arm``,
    ``seed``, ``repeat``, ``frames_per_decision``, ``checkpoint``,
    ``checkpoint_sha256``, ``obs_spec_version``, ``obs_spec_digest``, ``reset``),
    plus ``entry_frame`` / ``entry_hash`` / ``tape``, which is what makes a
    takeover entry auditable from the log alone.  The checkpoint arm fills the
    identity arguments; the tape and constant arms leave the defaults (no
    checkpoint is loaded and no observation is built).
    """

    return {
        "schema": SCHEMA_LOG_HEADER,
        "arm": str(arm),
        "seed": int(seed),
        "repeat": 0,
        "frames_per_decision": DECISION_FRAMES,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": str(checkpoint_sha256),
        "obs_spec_version": str(obs_spec_version),
        "obs_spec_digest": str(obs_spec_digest),
        "policy_input": str(policy_input),
        "reset": dict(reset),
        "entry_frame": int(entry_frame),
        "entry_hash": str(entry_hash),
        "entry_verified": True,
        "constant_cand": (None if config.constant_cand is None
                          else int(config.constant_cand)),
        "shoot_pulse_period": int(config.shoot_pulse_period),
        "cap_frames": int(config.cap_frames),
        "engine": {"path": engine.get("resolved_path"),
                   "sha256": engine.get("sha256")},
        "tape": {"path": tape.get("path"), "sha256": tape.get("sha256"),
                 "slots": tape.get("count")},
        "classification": "measured",
    }


def decision_row(*, seed: int, decision: int, frame: int, cand: int,
                 masks: Sequence[int], current: Mapping[str, Any],
                 events: Sequence[str], end: str,
                 enemies: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One log row, field names identical to ``v400-student-eval-decision-1``.

    ``frame`` is the game frame the decision was made at (frames elapsed
    *before* it) and ``frames_per_decision`` how many frames it advanced, so
    ``frame + frames_per_decision`` is the next row's frame.  ``masks`` is the
    only addition: one mask per frame, because a pulse decision legitimately
    sends two different masks and one integer cannot describe it.

    ``enemies`` is :func:`probe_enemies`' reading, taken **after** the decision's
    frames and written only when ``--decision-log`` was given.  It is
    referee-side telemetry: nothing on this row feeds a policy, and a refused
    reading (``available: false``) is written as-is rather than forged into 0s.
    """

    row = {
        "schema": SCHEMA_LOG_ROW,
        "seed": int(seed),
        "repeat": 0,
        "decision": int(decision),
        "frame": int(frame),
        "frames_per_decision": int(len(masks)) or DECISION_FRAMES,
        "cand": int(cand),
        "mask": int(masks[0]),
        "masks": [int(mask) for mask in masks],
        "deaths": float(current["deaths"]),
        "lives": float(current["lives"]),
        "bombs": float(current["bombs"]),
        "score": int(current["score"]),
        "stage": int(current["stage"]),
        "events": [str(name) for name in events],
        "end": str(end),
    }
    if enemies is not None:
        row["enemies"] = dict(enemies)
    return row


# ---------------------------------------------------------------------------
# --save-obs: the student's own observation planes, on disk
# ---------------------------------------------------------------------------
def _save_npy(path: str | Path, array: Any) -> None:
    """``np.save`` without pickle -- one seam, so a write failure is testable."""

    import numpy as np  # noqa: PLC0415

    np.save(Path(path), array, allow_pickle=False)


def history_cands(rows: Any) -> list[int | None]:
    """Decode the ``8x18`` one-hot ``act_hist`` rows back into candidates.

    An all-zero row is the frozen "no recorded action yet" padding of
    :class:`touhou_rl.v400.student.obs_spec_v2.ActionHistoryBuffer`, so it decodes
    to ``None`` -- never to 0, which would be a real candidate.
    """

    import numpy as np  # noqa: PLC0415

    array = np.asarray(rows)
    shape = (OBS_HISTORY_DECISIONS, N_CAND)
    if tuple(array.shape) != shape:
        raise ObsSaveError(
            f"the act_hist plane is {tuple(array.shape)}, obs-v3 says {shape}")
    out: list[int | None] = []
    for index, one_hot in enumerate(array):
        positions = np.flatnonzero(np.asarray(one_hot, dtype=np.uint8))
        if positions.size == 0:
            out.append(None)
        elif positions.size == 1:
            out.append(int(positions[0]))
        else:
            raise ObsSaveError(
                f"act_hist row {index} is not one-hot ({positions.size} set columns); "
                "the frozen buffer only writes action_one_hot/zero rows")
    return out


def check_obs_dir(path: str | Path) -> Path:
    """Refuse to write into anything but a **new or empty** directory.

    Called twice: at preflight (before an engine exists) and again when the
    recorder is created, so a directory that filled up in between is still refused.
    """

    target = Path(path)
    if target.exists() and not target.is_dir():
        raise PreflightRefused(
            f"--save-obs {target.as_posix()} exists and is not a directory",
            stop="save_obs_invalid")
    if target.is_dir():
        entries = sorted(target.iterdir(), key=lambda entry: entry.name)
        if entries:
            names = ", ".join(entry.name for entry in entries[:5])
            more = ", ..." if len(entries) > 5 else ""
            raise PreflightRefused(
                f"--save-obs {target.as_posix()} is not empty ({len(entries)} "
                f"entries: {names}{more}); a window is never merged into an existing "
                "directory -- name a new one", stop="save_obs_dir_not_empty")
    return target


def obs_meta_identity(*, policy: "LoadedPolicy", tape: Mapping[str, Any],
                      config: "TakeoverConfig", entry_frame: int) -> dict[str, Any]:
    """The ``meta.json`` identity every row of the corpus belongs to."""

    return {
        "schema": SCHEMA_OBS_META,
        "created_at": now_iso(),
        "runner": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "checkpoint": {
            "path": policy.path.as_posix(),
            "sha256": policy.sha256,
            "sha256_expected": policy.sha_expected,
            "sha_matches": bool(policy.sha_matches),
            "bytes": int(policy.path.stat().st_size),
            "arch": policy.arch,
            "parameters": policy.parameters,
        },
        "obs_spec_version": policy.obs_spec_version,
        "obs_spec_digest": policy.obs_spec_digest,
        "obs_spec_digest_required": OBS_SPEC_V3_DIGEST,
        "policy_input": POLICY_INPUT_CHECKPOINT,
        "obs_source": OBS_SOURCE,
        "label_source": OBS_LABEL_SOURCE,
        "normalization": ("none: the three .npy files hold the raw 0..255 uint8 "
                          "planes the encoder produced; the /255 the actor sees "
                          "happens at the tensor boundary"),
        "frames_per_decision": DECISION_FRAMES,
        "obs_stack_frames": OBS_STACK_FRAMES,
        "history_decisions": OBS_HISTORY_DECISIONS,
        "entry_frame": int(entry_frame),
        "seed": int(config.seed),
        "cap_frames": int(config.cap_frames),
        "shoot_pulse_period": int(config.shoot_pulse_period),
        "tape": {"path": tape.get("path"), "sha256": tape.get("sha256"),
                 "slots": tape.get("count"), "first_frame": tape.get("first_frame"),
                 "last_frame": tape.get("last_frame")},
        "argv": list(config.argv),
        "classification": "measured",
    }


def obs_readme(meta: Mapping[str, Any]) -> str:
    """``DIR/README.md``: the field table, every ``null`` and why, and the ruler.

    Written by the runner (not by hand) so the table cannot drift away from the
    fields it documents.
    """

    nulls = "\n".join(f"| `{name}` | **null** | {why} |"
                      for name, why in OBS_SLOT_NULL_FIELDS)
    lines = [
        f"# `--save-obs` corpus ({SCHEMA_OBS_META})",
        "",
        "The observation planes **a student policy actually visited**, written by",
        "`tools/rl/v8_32_takeover.py run --arm checkpoint --save-obs DIR`.",
        "It is the prerequisite of the WP05 targeted re-collection (DAgger): the",
        "states below are the student's own, so a teacher can be asked for labels",
        "on them.  **This directory contains no teacher label.**",
        "",
        "## Identity",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| runner | `{meta.get('runner', {}).get('path')}` |",
        f"| runner sha256 | `{meta.get('runner', {}).get('sha256')}` |",
        f"| checkpoint | `{meta.get('checkpoint', {}).get('path')}` |",
        f"| checkpoint sha256 | `{meta.get('checkpoint', {}).get('sha256')}` |",
        f"| obs spec | `{meta.get('obs_spec_version')}` / `{meta.get('obs_spec_digest')}` |",
        f"| seed | `{meta.get('seed')}` |",
        f"| entry frame | `{meta.get('entry_frame')}` |",
        f"| tape | `{meta.get('tape', {}).get('path')}` |",
        f"| decisions | `{meta.get('decisions')}` |",
        f"| frames | `{meta.get('frames')}` (entry `{meta.get('frames_prefix')}` + "
        f"hand-over `{meta.get('frames_takeover')}`) |",
        f"| stop | `{meta.get('stop')}` |",
        f"| complete | `{meta.get('complete')}` |",
        f"| created at | `{meta.get('created_at')}` |",
        "",
        "## Layout",
        "",
        "| path | content |",
        "| --- | --- |",
        f"| `{OBS_SLOTS_FILE}` | one JSON row per completed decision, flushed as written |",
    ]
    for name, file_name, shape, dtype in OBS_PLANES:
        planes = (meta.get("planes") or {}).get(name) or {}
        lines.append(
            f"| `{OBS_PLANE_DIR}/{file_name}` | `{dtype}` `{planes.get('shape')}` "
            f"(`N` x `{list(shape)}`), sha256 `{planes.get('sha256')}` |")
    lines += [
        f"| `{OBS_META_FILE}` | this identity, the plane shapes/dtypes/sha256 |",
        f"| `{OBS_README_FILE}` | this file |",
        "",
        "Row `i` of every plane is decision `i` of `slots.jsonl`: the three planes",
        "are written in the same order the rows are, so `slots[i].obs.row == i`.",
        "",
        "**The planes are the observation the policy saw, byte for byte.**  They are",
        "read from the same `ObservationBufferV3`/`ActionHistoryBuffer` state",
        "`ObservationStream.tensors()` turns into `model.act`'s input, never",
        "re-rendered and never re-normalized.  `tests/test_v8_32_takeover.py` pins",
        "`normalize_play(play[i]) == model.act's play tensor[i]` with `torch.equal`.",
        "",
        "## Row fields",
        "",
        "| field | meaning | relation to a T04 `slots.jsonl` row |",
        "| --- | --- | --- |",
        "| `schema` | `" + SCHEMA_OBS_SLOT + "` | T04: `v420-dataset-2f/1/slot` |",
        "| `dataset_version` / `contract` | `" + SCHEMA_OBS_SLOT + "` | this corpus' "
        "own id, deliberately **not** T04's `v420-2f-dataset/1`: a reader must not "
        "mistake a student rollout for a T04 shard |",
        "| `obs_contract` | the obs spec the planes obey (`obs-v3`) | T04: "
        "`obs-v420-2f` |",
        "| `action_contract` | `cand 0..17 (focus*9+move) + bomb 0..1, 2 frames per "
        "decision` | the same frozen `cand_to_mask`/`commit_frames`; T04 names its "
        "own `action-v420-2f` |",
        "| `episode_seed` | the `RESET` seed | same |",
        "| `decision` | absolute decision index (`decision_frame / 2`) | T04 stores "
        "none; `slot_id` plays that role |",
        "| `window_decision` | decisions since the entry (0-based) | T04 stores none |",
        "| `decision_frame` | game frames elapsed before the decision | same value, "
        "same ruler |",
        "| `commit_frames` | `[d, d + 2]`, the span the decision covers | identical "
        "to `touhou_rl.v420.contracts.decision_span(d)` |",
        "| `duration_frames` | `2` | same |",
        "| `obs_render_frames` | `[d-4, d-3, d-2, d-1]` (0-based frame indices) | "
        "identical to `v420.dataset_2f.obs_window(d)` |",
        "| `runner_frames.obs_render` | the same four **images** labelled by this "
        "runner's clock (`d-3..d`, frames elapsed) | T04 labels an image by its "
        "0-based index, this runner by elapsed count: `runner = t04 + 1` |",
        "| `runner_frames.commit` | `[d+1, d+2]`, the two `STEP` target frames | T04 "
        "records the span, not the targets |",
        "| `history_frames` | the eight decision frames the history came from, "
        "`d-16..d-2` | identical to `v420.dataset_2f.history_window(d)` |",
        "| `history` | eight candidates, oldest first; `null` where the buffer had no "
        "action yet | same (T04 rows are never padded, this runner's "
        "`entry_frame >= 16` seeds all eight) |",
        "| `executed_cand` | the candidate the **policy** emitted | T04: the "
        "teacher's executed candidate |",
        "| `cand` / `bomb` / `executed` | the same action, spelled as T04's "
        "`executed` block | same keys, `source` is `policy_act` |",
        "| `mask` / `masks` | the first `STEP` mask / both, from "
        "`commit_frames(..., release_shoot=release_due(...))` | T04 stores only "
        "`mask` |",
        "| `hash_pre` / `hash_post` | the engine `HASH` before the first decision and "
        "after each decision's frames | same names; **these are the student's own "
        "chain** |",
        "| `state` | `deaths`/`lives`/`bombs`/`score`/`stage`/`frame`, one `STATE` "
        "read per decision | T04 stores none of them per slot |",
        "| `events` / `end` | the `EVENTS` names of the decision's last frame, and "
        "the stop word | T04 stores none |",
        "| `obs` | `{play, hud, act_hist, row, hud_frame}` | same shape of mapping; "
        "`row` is this row's index in the three planes, `hud_frame` is `d-1` (the "
        "HUD image's 0-based frame) |",
        "| `label_source` | `" + OBS_LABEL_SOURCE + "` | T04 says `teacher_2f` / "
        "`teacher_slot0`; the D1/DAgger collectors say `shadow_teacher` / "
        "`shadow_teacher_lockstep`. **This row has no teacher label**, so it says so |",
        "| `obs_source` | `" + OBS_SOURCE + "` | T04 has no such field (its planes "
        "are the D1 render) |",
        "| `classification` | `measured` | same |",
        "",
        "## Fields written as `null` (and why)",
        "",
        "| field | value | reason |",
        "| --- | --- | --- |",
        nulls,
        "",
        "Nothing above is `0` as a stand-in: a `0` would be indistinguishable from a",
        "measured teacher choice.  The WP05 DAgger pass fills `chosen`, the `scores`",
        "vector, `margin` and the verification fields when it asks the teacher about",
        "these very states; until then they stay `null`.",
        "",
        "## Frame ruler (the one thing that differs from T04)",
        "",
        "T04 labels a rendered image by its **0-based frame index** and a decision by",
        "the **frames elapsed** at it; this runner's clock counts elapsed frames for",
        "both.  The same four images therefore read `[d-4 .. d-1]` in T04 and",
        "`[d-3 .. d]` here -- that is why both spellings are stored:",
        "`obs_render_frames` (T04-aligned) and `runner_frames.obs_render` (measured).",
        "Both fields are cross-checked at run time: a stack that is not the four",
        "consecutive frames ending at `d` stops the run instead of being relabelled.",
        "",
        "## Loading",
        "",
        "```python",
        "import json, numpy as np",
        "from touhou_rl.v400.student import obs_spec_v3",
        "slot = json.loads(open('slots.jsonl').readline())",
        "play = np.load('obs/play96x112.npy', mmap_mode='r')[slot['obs']['row']]",
        "hud = np.load('obs/hud.npy', mmap_mode='r')[slot['obs']['row']]",
        "hist = np.load('obs/act_hist.npy', mmap_mode='r')[slot['obs']['row']]",
        "play_t = obs_spec_v3.normalize_play(play)   # what model.act received",
        "```",
        "",
    ]
    return "\n".join(lines)


class ObsRecorder:
    """``--save-obs DIR``: the window's obs-v3 planes and its per-decision rows.

    ``slots.jsonl`` is appended and flushed per decision; the three planes are
    collected in memory (they are a few MiB per thousand decisions) and written by
    :meth:`finalize`, together with ``meta.json`` and ``README.md``.  Both steps
    fail closed: a row that cannot be written stops the window
    (:class:`ObsSaveError`), and a plane/meta write failure leaves the rows that
    were written, reports the reason in ``meta.json`` and ``result.json`` and makes
    the run exit non-zero.
    """

    def __init__(self, root: str | Path, *, identity: Mapping[str, Any]) -> None:
        self.root = check_obs_dir(root)
        self.identity = dict(identity)
        self.slots_path = self.root / OBS_SLOTS_FILE
        self.obs_dir = self.root / OBS_PLANE_DIR
        self.planes: dict[str, list[Any]] = {name: [] for name, *_ in OBS_PLANES}
        self.rows: int = 0
        self.error: str | None = None
        self.finalized: dict[str, Any] | None = None
        self._handle: Any = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.obs_dir.mkdir(parents=True, exist_ok=True)
            self._handle = self.slots_path.open("x", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise ObsSaveError(
                f"cannot prepare the --save-obs directory {self.root.as_posix()}: "
                f"{exc}") from exc

    # -- the running window ---------------------------------------------
    def summary(self) -> dict[str, Any]:
        """The identity part ``result.json`` carries while the window runs."""

        return {
            "root": self.root.as_posix(),
            "slots": self.slots_path.as_posix(),
            "obs_dir": self.obs_dir.as_posix(),
            "meta": (self.root / OBS_META_FILE).as_posix(),
            "readme": (self.root / OBS_README_FILE).as_posix(),
            "rows": int(self.rows),
            "planes": {name: f"{OBS_PLANE_DIR}/{file_name}"
                       for name, file_name, _shape, _dtype in OBS_PLANES},
            "complete": False,
            "error": self.error,
        }

    def record(self, *, decision: int, window_decision: int, decision_frame: int,
               cand: int, bomb: int, masks: Sequence[int],
               planes: Mapping[str, Any], hash_pre: str | None,
               hash_post: str | None, current: Mapping[str, Any],
               events: Sequence[str], end: str,
               runner_frames: Mapping[str, Sequence[int]]) -> int:
        """Append one decision: its ``slots.jsonl`` row and its three planes."""

        import numpy as np  # noqa: PLC0415

        row_index = int(self.rows)
        arrays = self._checked_planes(planes)
        duration = int(len(masks))
        t04_obs = [int(decision_frame) - OBS_STACK_FRAMES + offset
                   for offset in range(OBS_STACK_FRAMES)]
        expect_runner_obs = [int(decision_frame) - OBS_STACK_FRAMES + 1 + offset
                             for offset in range(OBS_STACK_FRAMES)]
        observed = [int(value) for value in runner_frames.get("obs_render") or ()]
        if observed != expect_runner_obs:
            raise ObsSaveError(
                f"decision {decision}: the stack was rendered at {observed}, but a "
                f"policy decision at frame {decision_frame} must see "
                f"{expect_runner_obs}; refusing to relabel an observation the policy "
                "did not look at")
        commits = [int(value) for value in runner_frames.get("commit") or ()]
        if commits != [int(decision_frame) + offset + 1 for offset in range(duration)]:
            raise ObsSaveError(
                f"decision {decision}: the STEP targets {commits} are not the "
                f"{duration} frames after {decision_frame}")
        history = history_cands(arrays["act_hist"])
        history_frames: list[int | None] = [
            (int(decision_frame) - DECISION_FRAMES * (OBS_HISTORY_DECISIONS - position)
             if cand_value is not None else None)
            for position, cand_value in enumerate(history)]
        row: dict[str, Any] = {
            "schema": SCHEMA_OBS_SLOT,
            "dataset_version": SCHEMA_OBS_SLOT,
            "contract": SCHEMA_OBS_SLOT,
            "obs_contract": self.identity.get("obs_spec_version") or "obs-v3",
            "action_contract": ("cand 0..17 (focus*9+move) + bomb 0..1, "
                                f"{DECISION_FRAMES} frames per decision"),
            "episode_seed": int(self.identity["seed"]),
            "slot_id": None,
            "shard_key": None,
            "provenance": None,
            "decision": int(decision),
            "window_decision": int(window_decision),
            "decision_frame": int(decision_frame),
            "commit_frames": [int(decision_frame),
                              int(decision_frame) + DECISION_FRAMES],
            "duration_frames": duration,
            "obs_render_frames": t04_obs,
            "history_frames": history_frames,
            "history": history,
            "executed_cand": int(cand),
            "executed": {"cand": int(cand), "bomb": int(bomb),
                         "mask": int(masks[0]), "source": "policy_act"},
            "cand": int(cand),
            "bomb": int(bomb),
            "mask": int(masks[0]),
            "masks": [int(mask) for mask in masks],
            "hash_pre": hash_pre,
            "hash_post": hash_post,
            "state": {"deaths": float(current["deaths"]),
                      "lives": float(current["lives"]),
                      "bombs": float(current["bombs"]),
                      "score": int(current["score"]),
                      "stage": int(current["stage"]),
                      "frame": int(current["frame"])},
            "events": [str(name) for name in events],
            "end": str(end),
            "runner_frames": {"obs_render": observed, "commit": commits},
            "obs": {"play": f"{OBS_PLANE_DIR}/play96x112.npy",
                    "hud": f"{OBS_PLANE_DIR}/hud.npy",
                    "act_hist": f"{OBS_PLANE_DIR}/act_hist.npy",
                    "row": row_index,
                    "hud_frame": int(decision_frame) - 1},
            "label_source": OBS_LABEL_SOURCE,
            "label_verified": None,
            "scores": None,
            "verified_mask": None,
            "verified_count": None,
            "chosen": None,
            "chosen_verified": None,
            "chosen_score": None,
            "teacher_bomb": None,
            "margin": None,
            "obs_source": OBS_SOURCE,
            "classification": "measured",
        }
        line = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        try:
            self._handle.write(line + "\n")
            self._handle.flush()
        except (OSError, ValueError) as exc:
            raise ObsSaveError(
                f"cannot append decision {decision} to {self.slots_path.as_posix()}: "
                f"{exc}") from exc
        # Only now is the row durable: the planes stay index-aligned with it.
        for name, _file_name, _shape, _dtype in OBS_PLANES:
            self.planes[name].append(arrays[name])
        self.rows = row_index + 1
        return self.rows

    def _checked_planes(self, planes: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and copy the three raw planes (shape/dtype are not negotiable)."""

        import numpy as np  # noqa: PLC0415

        checked: dict[str, Any] = {}
        for name, _file_name, shape, dtype in OBS_PLANES:
            if name not in planes:
                raise ObsSaveError(f"--save-obs: the {name} plane was not captured")
            array = np.asarray(planes[name])
            if tuple(array.shape) != shape:
                raise ObsSaveError(
                    f"--save-obs: the {name} plane is {tuple(array.shape)}, obs-v3 "
                    f"says {shape}")
            if array.dtype != np.dtype(dtype):
                raise ObsSaveError(
                    f"--save-obs: the {name} plane is {array.dtype}, obs-v3 says "
                    f"{dtype} (the planes are stored unnormalized)")
            copy = np.array(array, dtype=dtype, copy=True)
            copy.setflags(write=False)
            checked[name] = copy
        return checked

    # -- closing the window ---------------------------------------------
    def finalize(self, *, result: Mapping[str, Any]) -> dict[str, Any]:
        """Write the three planes, ``meta.json`` and ``README.md``; never raises."""

        import numpy as np  # noqa: PLC0415

        if self.finalized is not None:
            return self.finalized
        problems: list[str] = []
        plane_records: dict[str, Any] = {}
        for name, file_name, shape, dtype in OBS_PLANES:
            arrays = self.planes[name]
            path = self.obs_dir / file_name
            try:
                stacked = (np.stack(arrays, axis=0) if arrays
                           else np.zeros((0, *shape), dtype=np.dtype(dtype)))
                _save_npy(path, stacked)
                plane_records[name] = {
                    "file": f"{OBS_PLANE_DIR}/{file_name}",
                    "shape": [int(dim) for dim in stacked.shape],
                    "dtype": str(stacked.dtype),
                    "rows": int(stacked.shape[0]),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                    "classification": "measured",
                }
            except Exception as exc:  # noqa: BLE001 - a broken sink is recorded, not raised
                problems.append(
                    f"cannot write {path.as_posix()}: {type(exc).__name__}: {exc}")
        if self._handle is not None and not self._handle.closed:
            try:
                self._handle.close()
            except OSError as exc:
                problems.append(
                    f"cannot close {self.slots_path.as_posix()}: {exc}")
        slots_record: dict[str, Any] = {
            "file": OBS_SLOTS_FILE,
            "rows": int(self.rows),
            "bytes": None,
            "sha256": None,
        }
        try:
            slots_record["bytes"] = int(self.slots_path.stat().st_size)
            slots_record["sha256"] = sha256_file(self.slots_path)
        except OSError as exc:
            problems.append(
                f"cannot read back {self.slots_path.as_posix()}: {exc}")
        decisions = int(result.get("decisions") or 0)

        def _flags() -> tuple[bool, bool]:
            """``(ok, complete)`` for the problems known *right now*.

            ``ok``: every decision the window ran is in the corpus and every
            artifact was written.  ``complete``: that, and the window itself did not
            fail.  A window that stopped on a technical failure still leaves a valid
            corpus (that is what a failure dump is for) but is never called complete.
            """

            rows_ok = (not problems and int(self.rows) == decisions)
            return rows_ok, bool(rows_ok and result.get("failure") is None)

        ok, complete = _flags()
        meta: dict[str, Any] = dict(self.identity)
        meta.update({
            "created_at": self.identity.get("created_at") or now_iso(),
            "decisions": int(self.rows),
            "decisions_reported": decisions,
            "frames": int(result.get("frames") or 0),
            "frames_prefix": int(result.get("frames_prefix") or 0),
            "frames_takeover": int(result.get("frames_takeover") or 0),
            "stop": result.get("stop"),
            "decision_log": result.get("decision_log"),
            "save_obs": {"root": self.root.as_posix(), "source": OBS_SOURCE,
                         "schema": SCHEMA_OBS_SLOT},
            "slots": slots_record,
            "planes": plane_records,
            "null_fields": {name: why for name, why in OBS_SLOT_NULL_FIELDS},
            "complete": complete,
            "problems": problems,
            "failure": result.get("failure"),
        })
        meta_path = self.root / OBS_META_FILE
        readme_path = self.root / OBS_README_FILE
        try:
            write_json(meta_path, meta)
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            problems.append(f"cannot write {meta_path.as_posix()}: {exc}")
        try:
            readme_path.write_text(obs_readme(meta), encoding="utf-8", newline="\n")
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            problems.append(f"cannot write {readme_path.as_posix()}: {exc}")
        # a write above may have added a problem: restate the flags once
        ok, complete = _flags()
        meta["complete"] = complete
        meta["problems"] = problems
        if problems and not self.error:
            self.error = "; ".join(problems)
        try:
            write_json(meta_path, meta)
        except Exception as exc:  # noqa: BLE001 - the reason is kept in the result
            if not self.error:
                self.error = f"cannot rewrite {meta_path.as_posix()}: {exc}"
        self.finalized = {
            "root": self.root.as_posix(),
            "slots": slots_record,
            "obs_dir": self.obs_dir.as_posix(),
            "meta": meta_path.as_posix(),
            "readme": readme_path.as_posix(),
            "planes": plane_records,
            "rows": int(self.rows),
            "decisions": decisions,
            "frames": int(result.get("frames") or 0),
            "complete": complete,
            "ok": ok,
            "error": self.error,
            "problems": problems,
        }
        return self.finalized


# ---------------------------------------------------------------------------
# Configuration and the result skeleton
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TakeoverConfig:
    """One takeover run, fully resolved (the CLI is a thin wrapper around this)."""

    engine: Path
    expected_sha: str
    data_dir: Path
    tape_slots: Path
    seed: int
    entry_frame: int
    arm: str
    cap_frames: int
    out: Path
    constant_cand: int | None = None
    decision_log: Path | None = None
    save_obs: Path | None = None
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None
    device: str = "cpu"
    shoot_pulse_period: int = DEFAULT_PULSE_PERIOD
    timeout_s: float = DEFAULT_TIMEOUT_S
    no_xvfb: bool = False
    argv: tuple[str, ...] = ()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TakeoverConfig":
        return cls(
            engine=Path(args.engine).expanduser(),
            expected_sha=str(args.expected_sha),
            data_dir=Path(args.data_dir).expanduser(),
            tape_slots=Path(args.tape_slots).expanduser(),
            seed=int(args.seed),
            entry_frame=int(args.entry_frame),
            arm=str(args.arm),
            cap_frames=int(args.cap_frames),
            out=Path(args.out).expanduser(),
            constant_cand=(None if args.constant_cand is None
                           else int(args.constant_cand)),
            decision_log=(None if not args.decision_log
                          else Path(args.decision_log).expanduser()),
            save_obs=(None if not getattr(args, "save_obs", None)
                      else Path(args.save_obs).expanduser()),
            checkpoint=(None if not getattr(args, "checkpoint", None)
                        else Path(args.checkpoint).expanduser()),
            checkpoint_sha256=(None if not getattr(args, "checkpoint_sha256", None)
                               else str(args.checkpoint_sha256)),
            device=str(getattr(args, "device", "cpu") or "cpu"),
            shoot_pulse_period=int(args.shoot_pulse_period),
            timeout_s=float(args.timeout_s),
            no_xvfb=bool(getattr(args, "no_xvfb", False)),
            argv=tuple(getattr(args, "argv", ()) or ()),
        )


def base_result(config: "TakeoverConfig") -> dict[str, Any]:
    """The result skeleton: every required key exists even for a refusal."""

    return {
        "schema": SCHEMA_RESULT,
        "generated_by": f"{TOOL_PATH.as_posix()} run",
        "created_at": now_iso(),
        "argv": list(config.argv),
        "classification": "measured",
        "arm": str(config.arm),
        "seed": int(config.seed),
        "entry_frame": None,
        "entry_frame_requested": int(config.entry_frame),
        "entry_hash": None,
        "entry_verified": False,
        "entry_ready": False,
        "entry_anchor_hash": None,
        "entry_hash_matches_tape_anchor": None,
        "mismatch_slot": None,
        "mismatch": None,
        "frame_relation": None,
        "prefix": {"lead_in_decisions": None, "slots": None, "decisions": None,
                   "frames": None, "verified_slots": None,
                   "lead_in_verified": None},
        "entry_state": None,
        "frames": 0,
        "frames_prefix": 0,
        "frames_takeover": 0,
        "cap_frames": int(config.cap_frames),
        "decisions": 0,
        "stop": None,
        "stop_detail": None,
        "deaths": None,
        "deaths_at_entry": None,
        "deaths_delta": None,
        "lives_at_entry": None,
        "lives_final": None,
        "score_final": None,
        "stage_final": None,
        "wall_s": 0.0,
        "engine": {"requested": config.engine.as_posix(), "resolved_path": None,
                   "sha256": None, "expected_sha": str(config.expected_sha),
                   "sha_matches": None},
        "tape": None,
        "data_dir": config.data_dir.as_posix(),
        "mute": {"before": None, "after": None, "ok": None, "problems": []},
        "reset": None,
        "decision_log": None,
        # The --save-obs corpus of this window (None unless --save-obs was given, so
        # a result can never claim a corpus it did not write).
        "save_obs": None,
        # The aggregate of the per-decision ``enemies`` readings.  ``available``
        # stays false until a reading was taken, so a window that never logged a
        # decision cannot look like "no boss".
        "boss_summary": boss_summary(),
        "client": {"kind": None, "started": False, "closed": False},
        # The checkpoint arm's identity.  The tape/constant arms keep the empty
        # defaults, so a result can never claim a policy it did not run.
        "checkpoint": "",
        "checkpoint_sha256": "",
        "obs_spec_version": OBS_SPEC_NONE,
        "obs_spec_digest": "",
        "policy_input": POLICY_INPUT_NONE,
        "frames_per_decision": DECISION_FRAMES,
        "checkpoint_arm": None,
        "pixels": None,
        "renders": None,
        "policy_calls": None,
        "failure": None,
        "exit_code": None,
    }


def _fail(result: dict[str, Any], *, kind: str, stop: str, exit_code: int,
          detail: Any = None, trace: str | None = None) -> dict[str, Any]:
    result["stop"] = stop
    result["exit_code"] = int(exit_code)
    failure: dict[str, Any] = {"kind": kind, "stop": stop}
    if detail is not None:
        failure["detail"] = detail
    if trace:
        failure["traceback"] = trace
    result["failure"] = failure
    return result


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def resolve_engine(engine: str | Path) -> Path:
    """Accept a harness binary or a build directory and return the binary."""

    path = Path(engine).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = [path / "th08-modern", path / "rl_v3" / "th08-modern",
                      path / "build" / "rl_v3" / "th08-modern",
                      path / "build" / "th08-modern"]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise PreflightRefused(
            f"--engine {path.as_posix()} is a directory with no th08-modern; looked "
            f"for {[c.as_posix() for c in candidates]}", stop="engine_missing")
    raise PreflightRefused(f"--engine {path.as_posix()} does not exist",
                           stop="engine_missing")


def default_client_factory(config: "TakeoverConfig") -> Any:
    """The real engine, through the frozen ``HarnessClient`` (env role)."""

    from touhou_rl.v400.gym.protocol import HarnessClient  # noqa: PLC0415

    return HarnessClient(config.engine, config.data_dir,
                         timeout=float(config.timeout_s), role="env")


def _validate_arm(config: "TakeoverConfig") -> None:
    """Arm selector and its arguments, before anything is read or started."""

    arm = str(config.arm)
    if arm not in ARMS:
        raise PreflightRefused(f"--arm {arm!r} not in {list(ARMS)}",
                               stop="arm_invalid")
    if arm == "checkpoint":
        if config.checkpoint is None:
            raise PreflightRefused(
                "--arm checkpoint needs --checkpoint PATH (the obs-v3 actor to run; "
                "the arm cannot be measured without one)", stop="checkpoint_missing")
        if not str(config.checkpoint):
            raise PreflightRefused("--checkpoint is empty", stop="checkpoint_missing")
    else:
        if config.checkpoint is not None or config.checkpoint_sha256:
            raise PreflightRefused(
                f"--arm {arm} must not be given --checkpoint/--checkpoint-sha256: no "
                "policy is evaluated, and a record that mixes the arms cannot say "
                "what produced the frames", stop="arm_invalid")
    if arm == "constant":
        if config.constant_cand is None:
            raise PreflightRefused(
                "--arm constant needs --constant-cand N (the candidate every "
                "decision of the window emits)", stop="constant_cand_invalid")
        if not 0 <= int(config.constant_cand) < N_CAND:
            raise PreflightRefused(
                f"--constant-cand {config.constant_cand} is outside 0.."
                f"{N_CAND - 1} (cand = focus * 9 + move)", stop="constant_cand_invalid")
    elif config.constant_cand is not None:
        raise PreflightRefused(
            f"--arm {arm} must not be given --constant-cand "
            f"({config.constant_cand}): no fixed candidate is emitted",
            stop="constant_cand_invalid")


def _validate_save_obs(config: "TakeoverConfig") -> None:
    """``--save-obs``: the checkpoint arm only, into a new or empty directory."""

    if config.save_obs is None:
        return
    if str(config.arm) != "checkpoint":
        raise PreflightRefused(
            f"--save-obs {Path(config.save_obs).as_posix()} needs --arm checkpoint: "
            "only that arm builds the obs-v3 planes (the tape and constant arms build "
            "no observation at all, so there would be nothing to save)",
            stop="save_obs_invalid")
    if not str(config.save_obs):
        raise PreflightRefused("--save-obs is empty", stop="save_obs_invalid")
    try:
        check_obs_dir(config.save_obs)
    except PreflightRefused:
        raise
    except OSError as exc:
        raise PreflightRefused(
            f"--save-obs {Path(config.save_obs).as_posix()} cannot be inspected: {exc}",
            stop="save_obs_invalid") from exc


def _validate_numbers(config: "TakeoverConfig") -> None:
    """Numbers the CLI accepts but the contract can still refuse."""

    if int(config.cap_frames) < 0:
        raise PreflightRefused(
            f"--cap-frames {config.cap_frames} is negative; the cap counts frames "
            "after the entry and 0 means 'verify the entry and hand over nothing'",
            stop="cap_frames_invalid")
    if int(config.shoot_pulse_period) <= 0:
        raise PreflightRefused(
            f"--shoot-pulse-period {config.shoot_pulse_period} must be positive",
            stop="cap_frames_invalid")
    if int(config.seed) < 0:
        raise PreflightRefused(f"--seed {config.seed} is negative",
                               stop="seed_invalid")
    frame = int(config.entry_frame)
    if frame % DECISION_FRAMES:
        raise PreflightRefused(
            f"--entry-frame {frame} is odd; the tape is on a {DECISION_FRAMES}-frame "
            "grid, so an entry frame is even", stop="entry_frame_invalid")
    if frame < TAPE_FIRST_FRAME:
        raise PreflightRefused(
            f"--entry-frame {frame} is before the tape's first recorded slot frame "
            f"{TAPE_FIRST_FRAME}", stop="entry_frame_invalid")


# ---------------------------------------------------------------------------
# The engine, step by step
# ---------------------------------------------------------------------------
def _check_step(client: Any, mask: int, frame: int) -> dict[str, Any]:
    """One ``STEP <mask> 1``; the reply must advance exactly one frame."""

    reply = client.step(int(mask), 1)
    advanced = int(reply.get("advanced", 0) or 0)
    exit_flag = reply.get("exit") == "1"
    game_over = reply.get("game_over") == "1"
    if advanced != 1 or exit_flag:
        raise StepFailed(mask=int(mask), frame=int(frame), advanced=advanced,
                         exit_flag=exit_flag, game_over=game_over)
    return reply


def replay_prefix(client: Any, plan: Mapping[str, Any], *,
                  period: int,
                  frame_sink: Callable[[int], None] | None = None
                  ) -> dict[str, Any]:
    """Walk the tape up to the entry, checking every anchor the tape records.

    Order: the 8 lead-in decisions, then one ``HASH`` at frame 16 against slot
    0's ``hash_pre`` (skipped when the entry itself is frame 16, where the entry
    read *is* that anchor), then every slot with ``decision_frame < F`` with a
    ``HASH`` after its two frames against that slot's ``hash_post``.  The first
    disagreement raises :class:`EntryMismatch`, which the caller turns into a
    fail-closed result.

    ``frame_sink`` is called with the frames elapsed **after** every successful
    ``STEP``.  The checkpoint arm uses it to render the last four prefix frames;
    the tape and constant arms pass nothing and the protocol traffic is unchanged.
    """

    frames = 0
    lead_in = list(plan["lead_in"])
    for decision in lead_in:
        for mask in lead_in_masks(decision, period):
            frames += 1
            _check_step(client, mask, frames)
            if frame_sink is not None:
                frame_sink(frames)
    lead_in_verified: bool | str | None
    entry_frame = int(plan["entry_frame"])
    if not lead_in:
        lead_in_verified = None
    elif entry_frame == TAPE_FIRST_FRAME:
        lead_in_verified = "covered by the entry hash"
    else:
        value, reported = client.state_hash()
        actual = str(value).strip().lower()
        expected = str(plan["lead_in_anchor_hash"]).strip().lower()
        if actual != expected:
            slot = (plan["prefix_slots"][0].as_dict() if plan["prefix_slots"]
                    else {"slot_id": None, "decision_frame": TAPE_FIRST_FRAME})
            raise EntryMismatch(slot=slot, expected=expected, actual=actual,
                                frame=frames, frames_report=reported,
                                kind="lead_in")
        lead_in_verified = True

    verified = 0
    for slot in plan["prefix_slots"]:
        for mask in slot_masks(slot, period):
            frames += 1
            _check_step(client, mask, frames)
            if frame_sink is not None:
                frame_sink(frames)
        value, reported = client.state_hash()
        actual = str(value).strip().lower()
        if actual != slot.hash_post:
            raise EntryMismatch(slot=slot.as_dict(), expected=slot.hash_post,
                                actual=actual, frame=frames,
                                frames_report=reported, kind="slot")
        verified += 1
    return {"frames": frames, "verified_slots": verified,
            "lead_in_decisions": len(lead_in),
            "lead_in_verified": lead_in_verified}


def _entry_readings(client: Any) -> dict[str, Any]:
    """``entry_state``: the full STATE, PLAYER, HUD and an ENEMIES summary."""

    state = read_state(client)
    player = _payload(client.player())
    hud = _payload(client.hud())
    enemies = enemies_summary(list(client.enemies(0, 16)))
    return {"state": state, "player": player, "hud": hud, "enemies": enemies}


def _run_handover(*, client: Any, config: "TakeoverConfig",
                  plan: Mapping[str, Any], result: dict[str, Any],
                  log: DecisionLog | None,
                  tracker: "BossTracker | None" = None) -> dict[str, Any]:
    """Execute the arm's window and return the counters to merge into ``result``.

    Stops on the first of: the frame cap (spent in whole 2-frame decisions), a
    death, a game over, a stage change, an engine exit, or the tape running out.

    The per-decision ``ENEMIES`` probe runs **only** when ``log`` is not ``None``:
    a run without ``--decision-log`` sends exactly the traffic it always did, and
    ``tracker`` is ``None`` then too (nothing to aggregate).
    """

    arm = str(config.arm)
    slots: list[TakeoverSlot] = list(plan["takeover_slots"])
    period = int(config.shoot_pulse_period)
    cap = int(config.cap_frames)
    entry_state = result["entry_state"]["state"]
    deaths_at_entry = float(entry_state["deaths"])
    stage_at_entry = int(entry_state["stage"])
    frames_takeover = 0
    decisions = 0
    position = 0
    index = int(plan["entry_index"])
    current = readings(entry_state)
    stop = "frame_cap"
    stop_detail: dict[str, Any] = {}
    events_missing: str | None = None
    while True:
        if frames_takeover + DECISION_FRAMES > cap:
            stop = "frame_cap"
            stop_detail = {"cap_frames": cap, "frames_takeover": frames_takeover,
                           "note": ("the cap is spent in whole 2-frame decisions, an "
                                    "odd cap leaves the last frame unused")}
            break
        if arm == "tape":
            if position >= len(slots):
                stop = "tape_exhausted"
                stop_detail = {"slots_available": len(slots),
                               "frames_takeover": frames_takeover}
                break
            slot = slots[position]
            position += 1
            cand = int(slot.cand)
            masks = slot_masks(slot, period)
            index = int(slot.decision_index)
        else:
            cand = int(config.constant_cand if config.constant_cand is not None else 0)
            masks = constant_masks(cand, index, period)
        decision_frame = index * DECISION_FRAMES
        end = ""
        try:
            for offset, mask in enumerate(masks):
                _check_step(client, mask, decision_frame + offset + 1)
                frames_takeover += 1
        except StepFailed as exc:
            stop = exc.stop
            stop_detail = {"decision": index, "mask": exc.mask,
                           "frame": decision_frame, "advanced": exc.advanced,
                           "exit": exc.exit_flag, "game_over": exc.game_over,
                           "frames_takeover": frames_takeover}
            break
        current = readings(read_state(client))
        names, events_missing = read_events(client)
        enemies = probe_enemies(client) if log is not None else None
        if current["deaths"] > deaths_at_entry:
            end = "game_over" if current["lives"] <= 0 else "legal_death"
        elif int(current["stage"]) > stage_at_entry:
            end = "stage_clear"
        if enemies is not None and tracker is not None:
            tracker.observe(decision_frame, enemies)
        if log is not None:
            log.write(decision_row(seed=int(config.seed), decision=index,
                                   frame=decision_frame, cand=cand, masks=masks,
                                   current=current, events=names, end=end,
                                   enemies=enemies))
        decisions += 1
        if end:
            stop = end
            stop_detail = {"decision": index, "frame": int(current["frame"]),
                           "deaths": current["deaths"], "lives": current["lives"],
                           "stage": current["stage"]}
            break
        index += 1
    return {
        "frames_takeover": int(frames_takeover),
        "frames": int(result["frames_prefix"]) + int(frames_takeover),
        "decisions": int(decisions),
        "stop": stop,
        "stop_detail": stop_detail,
        "deaths": float(current["deaths"]),
        "deaths_delta": float(current["deaths"]) - deaths_at_entry,
        "lives_final": float(current["lives"]),
        "bombs_final": float(current["bombs"]),
        "score_final": int(current["score"]),
        "stage_final": int(current["stage"]),
        "events_unavailable": events_missing,
    }


# ---------------------------------------------------------------------------
# The checkpoint arm's window
# ---------------------------------------------------------------------------
def _run_checkpoint_handover(*, client: Any, config: "TakeoverConfig",
                             plan: Mapping[str, Any], result: dict[str, Any],
                             log: DecisionLog | None,
                             policy: "LoadedPolicy",
                             stream: ObservationStream,
                             tracker: "BossTracker | None" = None,
                             recorder: "ObsRecorder | None" = None
                             ) -> dict[str, Any]:
    """Run the obs-v3 actor's window: act, commit two frames, render each, log.

    The cadence is the eval's ``obs_window="consecutive"`` ruler: one decision is
    ``DECISION_FRAMES`` frames, every frame is rendered once into the 4-frame
    stack, and the action history is pushed **after** the frames were applied so
    ``hist`` at decision ``t`` holds ``a[0..t-1]``.

    A policy that raises, emits a candidate outside ``0..17`` or a bomb outside
    ``0..1``, or a readback that stops being readable ends the window with that
    stop reason; nothing is invented to keep going.

    ``recorder`` (``--save-obs``) sees the **same** planes the policy was fed: they
    are copied off ``stream.planes()`` right after ``stream.tensors()`` built the
    tensors, and each decision also reads one ``HASH`` so the corpus carries the
    student's own ``hash_pre``/``hash_post`` chain.  With no recorder not one extra
    protocol command is sent and nothing is copied.
    """

    model = policy.model
    period = int(config.shoot_pulse_period)
    cap = int(config.cap_frames)
    entry_state = result["entry_state"]["state"]
    deaths_at_entry = float(entry_state["deaths"])
    stage_at_entry = int(entry_state["stage"])
    frames_prefix = int(result["frames_prefix"])
    frames_takeover = 0
    decisions = 0
    policy_calls = 0
    history_pushes = 0
    index = int(plan["entry_index"])
    entry_index = int(plan["entry_index"])
    current = readings(entry_state)
    stop = "frame_cap"
    stop_detail: dict[str, Any] = {}
    events_missing: str | None = None
    hash_pre: str | None = None
    if recorder is not None:
        # decision 0's pre-state *is* the entry state the run already hashed.
        hash_pre = str(result.get("entry_hash") or "").strip().lower() or None
    while True:
        if frames_takeover + DECISION_FRAMES > cap:
            stop = "frame_cap"
            stop_detail = {"cap_frames": cap, "frames_takeover": frames_takeover,
                           "note": ("the cap is spent in whole 2-frame decisions, an "
                                    "odd cap leaves the last frame unused")}
            break
        decision_frame = index * DECISION_FRAMES
        try:
            play_t, hud_t, hist_t = stream.tensors()
        except Exception as exc:  # noqa: BLE001 - no observation, no decision
            stop = "pixels_unavailable"
            stop_detail = {"decision": index, "frame": decision_frame,
                           "frames_takeover": frames_takeover,
                           "reason": f"cannot build the obs-v3 tensors: "
                                     f"{type(exc).__name__}: {exc}"}
            break
        planes = stream.planes() if recorder is not None else None
        obs_frames = (list(stream.frames[-OBS_STACK_FRAMES:])
                      if recorder is not None else [])
        try:
            raw_cand, raw_bomb = model.act(play_t, hud_t, hist_t)
        except Exception as exc:  # noqa: BLE001 - a failed forward is a stop
            stop = "policy_output_invalid"
            stop_detail = {"decision": index, "frame": decision_frame,
                           "frames_takeover": frames_takeover,
                           "reason": f"model.act raised {type(exc).__name__}: {exc}"}
            break
        policy_calls += 1
        try:
            cand, bomb = int(raw_cand), int(raw_bomb)
        except (TypeError, ValueError) as exc:
            stop = "policy_output_invalid"
            stop_detail = {"decision": index, "frame": decision_frame,
                           "frames_takeover": frames_takeover,
                           "reason": f"model.act returned {raw_cand!r}/{raw_bomb!r}, "
                                     f"which is not an integer pair: {exc}"}
            break
        if not 0 <= cand < N_CAND or bomb not in (0, 1):
            stop = "policy_output_invalid"
            stop_detail = {"decision": index, "frame": decision_frame,
                           "frames_takeover": frames_takeover,
                           "cand": cand, "bomb": bomb,
                           "reason": (f"the policy emitted cand {cand}/bomb {bomb}; the "
                                      f"action contract is cand 0..{N_CAND - 1}, bomb "
                                      "0..1")}
            break
        masks = checkpoint_masks(cand, bomb, index, period)
        end = ""
        try:
            for offset, mask in enumerate(masks):
                _check_step(client, mask, decision_frame + offset + 1)
                frames_takeover += 1
                stream.render(frames_prefix + frames_takeover)
        except StepFailed as exc:
            stop = exc.stop
            stop_detail = {"decision": index, "mask": exc.mask,
                           "frame": decision_frame, "advanced": exc.advanced,
                           "exit": exc.exit_flag, "game_over": exc.game_over,
                           "frames_takeover": frames_takeover}
            break
        except PixelsUnavailable as exc:
            stop = "pixels_unavailable"
            stop_detail = {"decision": index, "frame": decision_frame,
                           "frames_takeover": frames_takeover, "reason": str(exc)}
            break
        current = readings(read_state(client))
        names, events_missing = read_events(client)
        enemies = probe_enemies(client) if log is not None else None
        hash_post: str | None = None
        if recorder is not None:
            # The student's own chain, one HASH per decision: without it the corpus
            # could not prove which state a re-collected label belongs to.
            hashed, _hash_frames = client.state_hash()
            hash_post = str(hashed).strip().lower() or None
        stream.push_action(cand)
        history_pushes += 1
        if current["deaths"] > deaths_at_entry:
            end = "game_over" if current["lives"] <= 0 else "legal_death"
        elif int(current["stage"]) > stage_at_entry:
            end = "stage_clear"
        if enemies is not None and tracker is not None:
            tracker.observe(decision_frame, enemies)
        if log is not None:
            log.write(decision_row(seed=int(config.seed), decision=index,
                                   frame=decision_frame, cand=cand, masks=masks,
                                   current=current, events=names, end=end,
                                   enemies=enemies))
        if recorder is not None:
            try:
                recorder.record(
                    decision=index, window_decision=index - entry_index,
                    decision_frame=decision_frame, cand=cand, bomb=bomb, masks=masks,
                    planes=planes or {}, hash_pre=hash_pre, hash_post=hash_post,
                    current=current, events=names, end=end,
                    runner_frames={"obs_render": obs_frames,
                                   "commit": [decision_frame + offset + 1
                                              for offset in range(len(masks))]})
            except ObsSaveError as exc:
                # Stop the window instead of running on: a corpus with a hole in it
                # would misalign the planes.  The rows already written stay, and the
                # counters below are still merged into result.json.
                stop = "save_obs_error"
                stop_detail = {"decision": index, "frame": decision_frame,
                               "frames_takeover": frames_takeover,
                               "corpus_rows": int(recorder.rows),
                               "reason": str(exc)}
                break
            hash_pre = hash_post
        decisions += 1
        if end:
            stop = end
            stop_detail = {"decision": index, "frame": int(current["frame"]),
                           "deaths": current["deaths"], "lives": current["lives"],
                           "stage": current["stage"]}
            break
        index += 1
    return {
        "frames_takeover": int(frames_takeover),
        "frames": frames_prefix + int(frames_takeover),
        "decisions": int(decisions),
        "stop": stop,
        "stop_detail": stop_detail,
        "deaths": float(current["deaths"]),
        "deaths_delta": float(current["deaths"]) - deaths_at_entry,
        "lives_final": float(current["lives"]),
        "bombs_final": float(current["bombs"]),
        "score_final": int(current["score"]),
        "stage_final": int(current["stage"]),
        "events_unavailable": events_missing,
        "policy_calls": int(policy_calls),
        "history_pushes": int(history_pushes),
        "renders": int(stream.renders),
        "rendered_frames": list(stream.frames),
    }


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------
def run_takeover(config: "TakeoverConfig", *,
                 client_factory: Callable[["TakeoverConfig"], Any] | None = None,
                 block_factory: Callable[[str], Any] | None = None,
                 policy_loader: Callable[["TakeoverConfig"], Any] | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """The whole run: preflight, RESET, prefix, entry, hand-over, result.json.

    Every expected failure is recorded in the returned result dict (and written to
    ``result.json``) with a non-zero ``exit_code``; nothing raises for a refusal.

    ``block_factory`` (the shared-block mapper) and ``policy_loader`` (the torch
    checkpoint loader) exist so the offline tests can run the real code path with a
    scripted block and actor.  The sha gate and the obs-spec gate run in front of
    ``policy_loader`` and cannot be bypassed.
    """

    started = clock()
    config.out.mkdir(parents=True, exist_ok=True)
    result = base_result(config)
    client: Any = None
    log: DecisionLog | None = None
    tracker: BossTracker | None = None
    policy: LoadedPolicy | None = None
    pixel_source: PixelSource | None = None
    stream: ObservationStream | None = None
    recorder: ObsRecorder | None = None
    try:
        _validate_arm(config)
        _validate_save_obs(config)
        _validate_numbers(config)

        if str(config.arm) == "checkpoint":
            # The requested path is identity; the sha256/spec fields are filled only
            # after a successful load, so a refusal cannot look like an identity.
            result["checkpoint"] = str(config.checkpoint or "")
            policy = load_policy(config, loader=policy_loader)
            result.update({
                "checkpoint": policy.path.as_posix(),
                "checkpoint_sha256": policy.sha256,
                "obs_spec_version": policy.obs_spec_version,
                "obs_spec_digest": policy.obs_spec_digest,
                "policy_input": POLICY_INPUT_CHECKPOINT,
                "frames_per_decision": DECISION_FRAMES,
                "checkpoint_arm": policy.identity(),
            })

        engine_path = resolve_engine(config.engine)
        result["engine"]["resolved_path"] = engine_path.as_posix()
        digest = sha256_file(engine_path)
        result["engine"]["sha256"] = digest
        expected = str(config.expected_sha).strip().lower()
        result["engine"]["sha_matches"] = bool(expected) and digest == expected
        if not result["engine"]["sha_matches"]:
            raise PreflightRefused(
                f"--engine {engine_path.as_posix()} has sha256 {digest}, the "
                f"expected value is {expected or '(empty)'}",
                stop="engine_sha_mismatch")

        if not config.data_dir.is_dir():
            raise PreflightRefused(
                f"--data-dir {config.data_dir.as_posix()} is not a directory",
                stop="data_dir_missing")
        before = read_mute(config.data_dir)
        result["mute"]["before"] = before
        if before.get("muted") is not True:
            raise PreflightRefused(
                f"{before['path']}: {before.get('reason', 'not muted')}; a takeover "
                "run refuses to start an audible engine (AGENTS.md section 9)",
                stop="mute_refused")

        tape = load_tape(config.tape_slots)
        result["tape"] = {key: value for key, value in tape.items()
                          if key not in ("slots", "lead_in")}
        seeds = list(tape.get("episode_seeds") or [])
        if seeds and int(config.seed) not in seeds:
            raise PreflightRefused(
                f"--seed {config.seed} but the tape records episode_seed {seeds}: a "
                "RESET with another seed cannot reproduce the hash chain",
                stop="seed_invalid")
        plan = entry_plan(tape, config.entry_frame)
        result["entry_frame"] = int(plan["entry_frame"])
        result["entry_anchor_hash"] = plan["anchor_hash"]
        result["prefix"]["lead_in_decisions"] = len(plan["lead_in"])
        result["prefix"]["slots"] = len(plan["prefix_slots"])
        result["prefix"]["decisions"] = int(plan["decision_count"])
        result["tape"]["cap_slots"] = len(plan["takeover_slots"])
        if str(config.arm) == "tape":
            available = len(plan["takeover_slots"]) * DECISION_FRAMES
            if available < int(config.cap_frames):
                raise PreflightRefused(
                    f"--cap-frames {config.cap_frames} exceeds the tape: after the "
                    f"entry at frame {plan['entry_frame']} only "
                    f"{len(plan['takeover_slots'])} slots ({available} frames) remain",
                    stop="cap_frames_invalid")

        if config.decision_log is not None:
            parent = config.decision_log.parent
            if not parent.is_dir():
                raise PreflightRefused(
                    f"--decision-log parent directory does not exist: {parent} (it "
                    "is never created implicitly)",
                    stop="decision_log_parent_missing")

        if config.save_obs is not None:
            # Created **before** the engine exists, so a directory that cannot be
            # used is a refusal instead of a window's worth of lost decisions.
            if policy is None:  # pragma: no cover - _validate_save_obs gates the arm
                raise TakeoverError("--save-obs needs a loaded checkpoint arm")
            recorder = ObsRecorder(
                config.save_obs,
                identity=obs_meta_identity(policy=policy, tape=tape, config=config,
                                           entry_frame=int(plan["entry_frame"])))
            result["save_obs"] = recorder.summary()

        factory = client_factory or default_client_factory
        client = factory(config)
        result["client"]["kind"] = type(client).__name__
        client.start()
        result["client"]["started"] = True

        if policy is not None:
            # SHM + READBACK 1 must be on **before** RESET (the eval's order), so
            # every frame of the episode -- the prefix included -- republishes the
            # shared block and the entry's stack is a real readback.
            pixel_source = PixelSource(client, name=default_shm_name(),
                                       block_factory=block_factory)
            result["pixels"] = pixel_source.open()
            history, seed_cands = seed_action_history(plan)
            stream = ObservationStream(pixels=pixel_source, history=history)
            result["checkpoint_arm"]["history_seed"] = {
                "decisions": len(seed_cands),
                "source": "the tape's executed_cand for decisions 0..entry-1",
                "cands": list(seed_cands),
                "buffer_rows": [int(value) for value in history.rows().reshape(-1)],
                "buffer_shape": [int(dim) for dim in history.rows().shape],
            }

        reset = client.reset(stage=RESET_STAGE, difficulty=RESET_DIFFICULTY,
                             team=RESET_TEAM, seed=int(config.seed))
        result["reset"] = dict(reset)
        if str(reset.get("ready")) != "1":
            raise EngineProtocolError(
                f"RESET {RESET_STAGE} {RESET_DIFFICULTY} {RESET_TEAM} "
                f"{config.seed} did not reach a playable state: {dict(reset)}")

        frame_sink = (None if stream is None or pixel_source is None
                      else checkpoint_frame_sink(stream=stream,
                                                 pixel_source=pixel_source,
                                                 entry_frame=int(plan["entry_frame"])))
        prefix = replay_prefix(client, plan, period=int(config.shoot_pulse_period),
                               frame_sink=frame_sink)
        result["prefix"].update({"frames": int(prefix["frames"]),
                                 "verified_slots": int(prefix["verified_slots"]),
                                 "lead_in_verified": prefix["lead_in_verified"]})
        result["frames_prefix"] = int(prefix["frames"])
        if int(prefix["frames"]) != int(plan["entry_frame"]):
            raise EngineProtocolError(
                f"the prefix advanced {prefix['frames']} frames, the entry frame is "
                f"{plan['entry_frame']}")

        entry_hash, entry_hash_frames = client.state_hash()
        entry_hash = str(entry_hash).strip().lower()
        entry = _entry_readings(client)
        result["entry_hash"] = entry_hash
        result["entry_state"] = {
            "hash": entry_hash,
            "hash_frames": int(entry_hash_frames),
            "state": entry["state"],
            "player": entry["player"],
            "hud": entry["hud"],
            "enemies": entry["enemies"],
            "classification": "measured",
        }
        anchor = str(plan["anchor_hash"]).strip().lower()
        result["entry_hash_matches_tape_anchor"] = entry_hash == anchor
        state_frame = int(entry["state"]["frame"])
        result["frame_relation"] = {
            "expected": int(plan["entry_frame"]),
            "state_frame": state_frame,
            "hash_frames": int(entry_hash_frames),
            "ok": state_frame == int(plan["entry_frame"]),
        }
        result["entry_verified"] = bool(entry_hash == anchor)
        if not result["entry_verified"]:
            result["mismatch_slot"] = plan["entry_slot"].slot_id
            result["mismatch"] = {
                "kind": "entry", "slot": plan["entry_slot"].slot_id,
                "decision_frame": plan["entry_slot"].decision_frame,
                "frame": int(plan["entry_frame"]), "expected": anchor,
                "actual": entry_hash, "hash_frames": int(entry_hash_frames)}
            _fail(result, kind="entry_hash_mismatch",
                  stop="entry_hash_mismatch", exit_code=EXIT_ENTRY_FAILED,
                  detail=(f"the entry hash {entry_hash!r} is not the tape's {anchor!r} "
                          f"at frame {plan['entry_frame']}"))
            return result
        if not result["frame_relation"]["ok"]:
            _fail(result, kind="frame_relation_conflict",
                  stop="frame_relation_conflict", exit_code=EXIT_ENTRY_FAILED,
                  detail=(f"the engine's STATE reports frame {state_frame} at the "
                          f"entry, the tape's entry frame is "
                          f"{plan['entry_frame']}"))
            return result
        result["entry_ready"] = True

        current = readings(entry["state"])
        result["deaths_at_entry"] = current["deaths"]
        result["deaths"] = current["deaths"]
        result["lives_at_entry"] = current["lives"]
        result["lives_final"] = current["lives"]
        result["score_final"] = current["score"]
        result["stage_final"] = current["stage"]

        if config.decision_log is not None:
            log = DecisionLog(config.decision_log)
            tracker = BossTracker()
            log.write(decision_header(
                arm=config.arm, seed=int(config.seed),
                entry_frame=int(plan["entry_frame"]),
                entry_hash=entry_hash,
                reset=result["reset"], tape=tape,
                engine=result["engine"], config=config,
                checkpoint=(policy.path.as_posix() if policy is not None else ""),
                checkpoint_sha256=(policy.sha256 if policy is not None else ""),
                obs_spec_version=(policy.obs_spec_version if policy is not None
                                  else OBS_SPEC_NONE),
                obs_spec_digest=(policy.obs_spec_digest if policy is not None else ""),
                policy_input=(POLICY_INPUT_CHECKPOINT if policy is not None
                              else POLICY_INPUT_NONE)))
            result["decision_log"] = {"path": log.path.as_posix(), "rows": 0,
                                      "header_rows": 1}

        if policy is not None and stream is not None:
            result.update(_run_checkpoint_handover(
                client=client, config=config, plan=plan, result=result, log=log,
                policy=policy, stream=stream, tracker=tracker, recorder=recorder))
            window = {key: result.get(key) for key in
                      ("decisions", "frames_takeover", "renders", "policy_calls",
                       "history_pushes", "rendered_frames")}
            window["history_seed_decisions"] = len(prefix_action_cands(plan))
            result["checkpoint_arm"]["window"] = window
            if pixel_source is not None:
                result["pixels"] = pixel_source.identity()
        else:
            result.update(_run_handover(client=client, config=config, plan=plan,
                                        result=result, log=log, tracker=tracker))
        if tracker is not None:
            result["boss_summary"] = tracker.summary()
        stop = str(result.get("stop"))
        if stop in HANDOVER_OK_STOPS:
            result["exit_code"] = EXIT_OK
        else:
            _fail(result, kind=stop, stop=stop, exit_code=EXIT_TECHNICAL,
                  detail=result.get("stop_detail"))
    except EntryMismatch as exc:
        result["entry_verified"] = False
        result["mismatch_slot"] = exc.slot.get("slot_id")
        result["mismatch"] = exc.record()
        _fail(result, kind="entry_hash_mismatch", stop="entry_hash_mismatch",
              exit_code=EXIT_ENTRY_FAILED, detail=str(exc))
    except PreflightRefused as exc:
        _fail(result, kind=type(exc).__name__, stop=exc.stop,
              exit_code=EXIT_REFUSED, detail=str(exc))
    except DecisionLogError as exc:
        _fail(result, kind="decision_log", stop="decision_log_error",
              exit_code=EXIT_TECHNICAL, detail=str(exc))
    except ObsSaveError as exc:
        # A corpus that lost a decision is not a corpus: the window stops here and
        # the rows/planes written so far are kept (meta.json says it is incomplete).
        _fail(result, kind="save_obs", stop="save_obs_error",
              exit_code=EXIT_TECHNICAL, detail=str(exc))
    except PixelsUnavailable as exc:
        _fail(result, kind="pixels_unavailable", stop="pixels_unavailable",
              exit_code=EXIT_TECHNICAL, detail=str(exc))
    except PolicyOutputError as exc:
        _fail(result, kind="policy_output_invalid", stop="policy_output_invalid",
              exit_code=EXIT_TECHNICAL, detail=str(exc))
    except EngineProtocolError as exc:
        _fail(result, kind="engine_protocol", stop="protocol_error",
              exit_code=EXIT_TECHNICAL, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - a technical failure is evidence
        _fail(result, kind="technical", stop="technical_error",
              exit_code=EXIT_TECHNICAL, detail=f"{type(exc).__name__}: {exc}",
              trace=traceback.format_exc())
    finally:
        _close_and_settle(config=config, result=result, client=client, log=log,
                          pixel_source=pixel_source, recorder=recorder)
        result["wall_s"] = round(clock() - started, 3)
        if result.get("exit_code") is None:
            result["exit_code"] = (EXIT_TECHNICAL if result.get("failure")
                                   else EXIT_OK)
        write_json(config.out / "result.json", result)
    return result


def _close_and_settle(*, config: "TakeoverConfig", result: dict[str, Any],
                      client: Any, log: DecisionLog | None,
                      pixel_source: "PixelSource | None" = None,
                      recorder: "ObsRecorder | None" = None) -> None:
    """Close the client and the log, and settle the four mute bytes afterwards."""

    if recorder is not None:
        # First: the obs corpus is pure disk I/O on data already measured, and a
        # failure here must survive even if the engine's close goes wrong.
        settled = recorder.finalize(result=result)
        result["save_obs"] = settled
        if not settled.get("ok"):
            reason = settled.get("error") or (
                f"the corpus holds {settled.get('rows')} rows for "
                f"{settled.get('decisions')} decisions")
            if result.get("failure") is None:
                result["failure"] = {"kind": "save_obs", "stop": "save_obs_error",
                                     "detail": reason}
            if result.get("exit_code") in (None, EXIT_OK):
                result["exit_code"] = EXIT_TECHNICAL
    if pixel_source is not None:
        # Before the client: the block is a mapping of a segment the engine owns.
        try:
            result["pixels"] = pixel_source.identity()
        finally:
            pixel_source.close()
    if client is not None:
        try:
            client.close()
            result["client"]["closed"] = True
        except Exception as exc:  # noqa: BLE001 - recorded, never raised
            result["client"]["close_error"] = f"{type(exc).__name__}: {exc}"
        if result["client"]["started"] and result["mute"].get("before") is not None:
            after = read_mute(config.data_dir)
            result["mute"]["after"] = after
            problems: list[str] = list(result["mute"].get("problems") or [])
            if after.get("available") is not True:
                problems.append(f"after the exit: {after.get('reason')}")
            elif after.get("muted") is not True:
                problems.append(
                    f"after the exit the cfg became audible: {after.get('bytes')}")
            result["mute"]["problems"] = problems
            result["mute"]["ok"] = (result["mute"]["before"].get("muted") is True
                                    and not problems)
            if problems and result.get("exit_code") in (None, EXIT_OK):
                result["exit_code"] = EXIT_MUTE
                if result.get("failure") is None:
                    result["failure"] = {"kind": "mute_violation",
                                         "stop": result.get("stop"),
                                         "detail": "; ".join(problems)}
    if log is not None:
        try:
            log.close()
        except DecisionLogError as exc:
            if result.get("failure") is None:
                result["failure"] = {"kind": "decision_log",
                                     "detail": str(exc)}
        if result.get("decision_log") is not None:
            result["decision_log"]["rows"] = log.rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def maybe_reexec_under_xvfb(enabled: bool) -> None:
    """Re-exec under ``xvfb-run`` on POSIX when there is no ``DISPLAY``."""

    if not enabled or os.name == "nt":
        return
    if os.environ.get("DISPLAY") or os.environ.get("V832_TAKEOVER_XVFB") == "1":
        return
    xvfb = shutil.which("xvfb-run")
    if not xvfb:
        return
    os.environ["V832_TAKEOVER_XVFB"] = "1"
    os.execvp(xvfb, [xvfb, "-a", "-s", "-screen 0 1280x1024x24",
                     sys.executable, *sys.argv])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v8_32_takeover.py",
        description=("Conditional takeover entry: RESET, replay the tape frame by "
                     "frame to --entry-frame, verify the state hash at every slot, "
                     "then hand control to one arm.  Starts no engine unless the "
                     "`run` subcommand is used."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Exit codes: 0 the window ran, 1 the entry or a hash check failed "
                "(fail-closed, no control handed over), 2 technical failure (no "
                "pixels, a policy outside the action contract), 4 refused before the "
                "start, 5 the four mute bytes changed."))
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run", help="RESET, replay the tape to the entry, verify it, hand over",
        description=("Replay the tape from RESET to --entry-frame, check every "
                     "slot's hash_post, then run the selected arm until "
                     "--cap-frames or a death.  Writes <out>/result.json and, with "
                     "--decision-log, the per-decision JSONL."))
    run.add_argument("--engine", required=True,
                     help="the rl_v3 th08-modern binary, or a build directory "
                          "containing it")
    run.add_argument("--expected-sha", required=True, dest="expected_sha",
                     help="the sha256 the engine binary must have")
    run.add_argument("--data-dir", required=True, dest="data_dir",
                     help="the game data directory the engine is started in "
                          "(th08.cfg must be muted)")
    run.add_argument("--tape-slots", required=True, dest="tape_slots",
                     help="the slots.jsonl of one T04 2-frame shard")
    run.add_argument("--seed", required=True, type=int,
                     help="the RESET seed; must be the tape's episode_seed")
    run.add_argument("--entry-frame", required=True, dest="entry_frame", type=int,
                     help="the frame control is handed over at (even, >= 16, a slot "
                          "boundary of the tape)")
    run.add_argument("--arm", required=True, choices=ARMS,
                     help="tape = continue the recorded masks, constant = one fixed "
                          "candidate, checkpoint = run an obs-v3 actor loaded from "
                          "--checkpoint")
    run.add_argument("--constant-cand", dest="constant_cand", type=int, default=None,
                     help="the candidate --arm constant emits (0..17)")
    run.add_argument("--checkpoint", default=None,
                     help="the obs-v3 actor checkpoint --arm checkpoint runs")
    run.add_argument("--checkpoint-sha256", dest="checkpoint_sha256", default=None,
                     help="the sha256 the checkpoint file must have (a mismatch is "
                          "refused before anything is started)")
    run.add_argument("--device", default="cpu",
                     help="the torch device the checkpoint is loaded on (default cpu)")
    run.add_argument("--cap-frames", required=True, dest="cap_frames", type=int,
                     help="frames the arm may advance after the entry")
    run.add_argument("--out", required=True,
                     help="run directory (result.json goes here)")
    run.add_argument("--decision-log", dest="decision_log", default=None,
                     help="per-decision JSONL path; the parent directory must exist")
    run.add_argument("--save-obs", dest="save_obs", default=None,
                     help="write the student's own obs-v3 planes and one slot row "
                          "per decision into this NEW (or empty) directory; "
                          "--arm checkpoint only")
    run.add_argument("--shoot-pulse-period", dest="shoot_pulse_period", type=int,
                     default=DEFAULT_PULSE_PERIOD,
                     help="the SHOOT release phase the tape was recorded with "
                          f"(default {DEFAULT_PULSE_PERIOD})")
    run.add_argument("--timeout-s", dest="timeout_s", type=float,
                     default=DEFAULT_TIMEOUT_S,
                     help=f"per-command protocol timeout (default {DEFAULT_TIMEOUT_S})")
    run.add_argument("--no-xvfb", action="store_true", dest="no_xvfb",
                     help="do not re-exec under xvfb-run (POSIX only)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    maybe_reexec_under_xvfb(enabled=not bool(args.no_xvfb))
    args.argv = list(sys.argv if argv is None else ["v8_32_takeover.py", *argv])
    config = TakeoverConfig.from_args(args)
    try:
        result = run_takeover(config)
    except OSError as exc:
        print(f"run: cannot write under --out {config.out.as_posix()}: {exc}")
        return EXIT_TECHNICAL
    if result.get("failure"):
        print(f"run: FAILED stop={result.get('stop')} rc={result.get('exit_code')}")
        print(f"  {result['failure'].get('detail')}")
    else:
        print(f"run: ok arm={result.get('arm')} "
              f"entry_frame={result.get('entry_frame')} "
              f"entry_hash={result.get('entry_hash')} "
              f"frames={result.get('frames')} decisions={result.get('decisions')} "
              f"stop={result.get('stop')} rc={result.get('exit_code')}")
    print(f"wrote: {(config.out / 'result.json').as_posix()}")
    return int(result.get("exit_code", EXIT_TECHNICAL))


if __name__ == "__main__":
    raise SystemExit(main())
