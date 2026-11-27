#!/usr/bin/env python3
"""V8.32 DAgger relabel: ask the teacher for labels on the **student's own states**.

What this tool is, in one sentence
----------------------------------
This is **not** a replay and **not** a re-collection.  It takes a rollout corpus
that ``tools/rl/v8_32_takeover.py run --arm checkpoint --save-obs DIR`` wrote
(``DIR/slots.jsonl`` + ``DIR/obs/*.npy`` + ``DIR/meta.json``), replays that
rollout's *own* frames frame by frame to every decision boundary, and at each of
those boundaries asks the teacher a **fresh question**::

    decide(engine, config, record_source="child", frame=<the student's frame>)

The teacher's 18-way score vector, its per-candidate verification mask and its
verdict are written next to the row it was asked about.  The frames the student
executed are replayed with the student's own masks -- the teacher never drives
this process, never chooses an action that is committed, and never moves the
shadow (``record_source="child"``; ``touhou_rl/v420/reslot.py`` asserts the same
thing at its own query point).  That is what makes the labels *DAgger* labels:
the state distribution is the student's, the label is the teacher's answer on it.

口径（中文）：本工具产出的是 **DAgger 标签**，不是重放、不是重新采集。状态来自学生
自己的 rollout（``--save-obs`` 落盘的 ``slots.jsonl``），标签来自在这些状态上**重新
查询教师**（``decide(record_source="child")``）；学生执行的帧由学生的 mask 重放，
教师不推进主进程、不提交任何动作。

Why a new file instead of the v4.2.0 ``reslot``
-----------------------------------------------
``touhou_rl/v420/reslot.py`` relabels a **legacy shard** (``shard.json`` +
``labels.jsonl`` + four-frame blocks).  A ``--save-obs`` directory is not one:
measured on this checkout, ``reslot``/``dataset_2f`` fail closed on every captured
row -- the shard triple is missing and fifteen row-level keys are absent -- while
the observation planes and the frame ruler are clean.  Forging a legacy shard
around a student rollout would claim a provenance that does not exist, so this
tool writes its own format and says so.

What it guarantees (fail-closed everywhere)
-------------------------------------------
1. **Source self-check before anything starts** (:func:`load_source`): the row
   count must be the one ``meta.json`` declares, every row must sit on the
   two-frame grid, every ``episode_seed`` must be the requested one, the
   ``hash_pre -> hash_post -> hash_pre`` chain must be unbroken, every row must
   carry ``mask``/``masks``, the ``(cand, bomb) -> masks`` round trip must hold
   *and* equal the masks this checkout's :func:`v8_32_takeover.checkpoint_masks`
   derives for that decision index, and no row may already carry a teacher label.
   Any violation refuses the run (exit 4) with the row named.  Nothing is written.
2. **The prefix is the tape's, verified per slot** (:func:`replay_prefix` from
   ``v8_32_takeover``): ``RESET``, the 8 lead-in decisions from the tape's slot 0
   ``history``, then every tape slot with ``decision_frame < --entry-frame``, two
   ``STEP``s each, one ``HASH`` after every slot against that slot's recorded
   ``hash_post``.  A mismatch is fail-closed.
3. **The entry is proven three ways** before a single label is written: the
   engine's frame is ``--entry-frame``, its hash equals the tape slot's
   ``hash_pre``, and that same hash equals **row 0's ``hash_pre``** -- i.e. the
   state the teacher is about to be asked about is the state the student actually
   stood on.
4. **Every decision boundary is re-verified**: the engine's hash must equal that
   row's ``hash_pre`` *before* the query, and its ``hash_post`` *after* the
   student's masks were stepped.  A divergence abandons the rest of the run
   instead of writing labels from a different trajectory.
5. **One query per decision**, ``record_source="child"``, and the process must
   still sit on the same frame afterwards (a teacher that moved the shadow would
   make every later label belong to a different state).
6. **Write-time validation** (:func:`check_label_fields`) runs on every row
   before it reaches the file; a record that violates it aborts the run.
7. **A failure keeps what was written**: rows already flushed stay, ``meta.json``
   records the stop word, the reason and the counters, and the exit code is
   non-zero.

The label fields (and the one place this schema differs from ``reslot``)
-----------------------------------------------------------------------
``scores`` / ``verified_mask`` / ``verified_count`` / ``chosen`` /
``chosen_verified`` / ``chosen_score`` / ``margin`` / ``bomb`` / ``teacher_bomb``
/ ``label_source="teacher_2f"`` / ``label_verified`` are the
``touhou_rl/v420/reslot.py`` label fields, produced by the **same**
:func:`touhou_rl.v420.reslot.decision_view` that module uses, so a label taken
here is interchangeable with a T04 one.

* ``chosen`` is derived here, not copied: the highest teacher score **inside**
  ``verified_mask`` (ties -> the lower candidate index), and -- only when nothing
  verified -- the highest finite score overall with ``chosen_verified=false``.
  The teacher's own ``chosen`` is kept in ``relabel.teacher.chosen`` for audit.
* ``label_verified`` is ``verified_count > 0``, the rule
  :func:`touhou_rl.v420.contracts.check_label_provenance` states for
  ``teacher_2f`` ("verified only when at least one branch really verified").
* ``bomb`` is the **teacher's** bomb channel and ``teacher_bomb`` its twin (the
  two are the same byte here; ``touhou_rl/v420/dataset_2f.py`` reads
  ``label["bomb"]`` as ``teacher_bomb``).  The **student's** executed bomb is
  preserved verbatim in ``executed.bomb``, next to ``executed.cand``/
  ``executed.mask``: after relabelling, the top-level ``(cand, bomb, mask)``
  triple is deliberately *not* a round trip, exactly as in a T04 slot record.
  A reader that wants what was executed reads ``executed``.
* ``scores`` entries are numbers -- except when the teacher produced a non-finite
  score for a candidate (``beam_search`` starts at ``-np.inf`` and a degenerate
  tape keeps it).  Such an entry is written as ``null`` and listed in
  ``relabel.nonfinite_scores``; fabricating a floor value would be a measurement
  that was never taken.  ``scores_length``/``scores_finite`` accept that case
  only when the list agrees with ``relabel.nonfinite_scores``.
* ``slot_id`` stays ``null``: a student decision at frame ``d`` is **not** the
  tape's slot at frame ``d`` (``touhou_rl/v420/reslot.py``'s
  ``check_label_record`` refuses a ``slot_id`` that is not
  ``(d - 16) // 2``; here such a slot id would name a different state).  Join on
  ``(episode_seed, decision_frame, hash_pre)`` instead.

The player coordinate read (this format's addition)
---------------------------------------------------
Every label row also carries::

    "player": {"px": 192.0, "py": 384.0, "read_phase": "before_action", ...}

The value is the ``px``/``py`` pair of the engine's ``PLAYER`` reply
(``third_party/th08-rl/src/rlharness/rlharness.cpp::HandlePlayer``:
``"OK frames=%llu ... px=%08x py=%08x ..."``, i.e. two float32 bit patterns) as
decoded by the shipped decoder
``touhou_rl/v400/gym/protocol.py::parse_player_reply`` -- the tool never splits
the wire string itself, and it never re-derives a coordinate from pixels.  The
read happens on the **authoritative** process at each decision boundary, in this
order:

1. ``FRAME``/``HASH``: the engine sits on the row's ``decision_frame`` and its
   hash equals the row's ``hash_pre`` (the state the label belongs to);
2. ``PLAYER``: one read, decoded to ``px``/``py``, stamped
   ``read_phase="before_action"``;
3. ``decide(...)``: the teacher is asked about that state;
4. the student's own masks are ``STEP``ped.

So ``player.px``/``player.py`` are the student's position **before** the action
this row is labelled on, on the same frame and the same state hash as the label.
``PLAYER`` is a read-only command (``HandlePlayer`` only reads ``g_Player`` /
``g_GameManager.globals``), and the run's own per-decision hash checks prove the
replay never left the student's trajectory.  ``player.frame`` is the ``frames=``
field of the same reply, and a reply whose frame is not the decision frame is
refused (``player_frame_mismatch``): a coordinate from another frame would
silently corrupt any position-conditioned reading of the label.

``--require-player`` makes the coordinates part of the **contract**: the run is
refused (``player_unavailable``, exit 2) if a boundary cannot read them, instead
of writing a row whose ``px``/``py`` are ``null``.  It is off by default, so the
v8.32 label contract (which had no coordinate at all) is unchanged for existing
callers, and the field table below says which of the two a given file is.

Outputs
-------
``DIR2/labels.jsonl``
    one row per labelled decision: the source row **verbatim** plus the label
    fields, a ``teacher`` provenance block and a ``relabel`` block (engine sha256,
    tape sha256, entry hash, tolerance, timestamp, the teacher's own verdict).
    Started only after the entry is verified; flushed per row.
``DIR2/meta.json``
    schema, source identity and sha256, engine/tape/teacher identity, the entry
    proof, the counters (rows/replays/queries/frames), the ``chosen`` histogram,
    the ``executed == chosen`` agreement rate, the failure record if any, and the
    wall clock.
``DIR2/README.md``
    the field table, the ``bomb`` rule above, and every ``null`` and why.

Safety
------
No audio: the four mute bytes of ``<data-dir>/th08.cfg`` are decoded before the
engine is started and the run is **refused** unless ``musicMode``/``playSounds``
and both volumes are 0 (``AGENTS.md`` section 9); the same bytes are read again
after the engine exits.  ``TeacherEngine`` sets ``SDL_AUDIODRIVER=dummy`` itself.
This tool starts no engine unless the ``run`` subcommand is used.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = Path(__file__).resolve().parent
for _entry in (str(REPO_ROOT), str(TOOL_DIR)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import v8_32_takeover as tk  # noqa: E402  (read-only reuse of the tape replayer)

TOOL_PATH = Path(__file__).resolve()

#: One decision is two game frames (``touhou_rl.v420.contracts.DECISION_FRAMES``).
DECISION_FRAMES = 2
#: ``cand = focus * 9 + move``: 18 candidates, the teacher's score vector length.
N_CAND = 18
#: The SHOOT release phase every T04 tape and every student rollout uses.
DEFAULT_PULSE_PERIOD = 48
#: ``TeacherEngine`` timeout per protocol command: the teacher's decision is the
#: slow command and it forks children, so the value is the same one
#: ``touhou_rl/v420/counterfactual.py`` gives its shadow sessions.
DEFAULT_TIMEOUT_S = 240.0
#: The teacher profile the two-frame corpus was labelled with.
DEFAULT_PROFILE = "fast"
#: ``RESET <stage> <difficulty> <team> <seed>`` defaults, identical to the runner's.
RESET_STAGE = 0
RESET_DIFFICULTY = 1
RESET_TEAM = 0
#: The label this tool takes: a fresh per-slot two-frame teacher query
#: (``touhou_rl.v420.contracts.LABEL_SOURCE_RULES["teacher_2f"]``).
LABEL_SOURCE = "teacher_2f"
#: The word ``tools/rl/v8_32_takeover.py --save-obs`` writes into a student row;
#: a source that says anything else is either not a student rollout or already
#: labelled, and both are refused.
STUDENT_LABEL_SOURCE = tk.OBS_LABEL_SOURCE
#: ``verify.verify_plans``' frozen tolerance.  ``decide()`` does not expose it, so
#: the tool *reads it off the function* at run time (:func:`teacher_tolerance`) and
#: records the value it read; a changed default would make the record a lie.
PINNED_TOLERANCE = 0.001

SCHEMA_META = "v832-dagger-relabel/1"
SCHEMA_LABEL = "v832-dagger-relabel-label/1"
SCHEMA_RELABEL_BLOCK = "v832-dagger-relabel-block/1"
SCHEMA_PLAYER_BLOCK = "v832-dagger-relabel-player/1"
LABELS_FILE = "labels.jsonl"
META_FILE = "meta.json"
README_FILE = "README.md"
#: The ``PLAYER`` request verb (``rlharness.cpp``, ``HandlePlayer``).
PLAYER_COMMAND = "PLAYER"
#: The two reply fields this tool records, spelled exactly as the engine spells
#: them (``"px=%08x py=%08x"``) and as ``protocol.parse_player_reply`` reads them.
PLAYER_FIELDS = ("px", "py")
#: When the read happens, stamped into every row.
PLAYER_READ_PHASE = "before_action"
#: What the read precedes on that decision -- also stamped into every row, so a
#: reader can check the ordering claim without opening this file.
PLAYER_PRECEDES = ("decide", "student_step")
#: The player coordinate checks :func:`check_player_fields` reports.  They are
#: deliberately **not** in :data:`CHECK_NAMES`: those fifteen names are the
#: ``reslot`` label checks plus this format's own, and ``tests/test_v8_32_relabel.py``
#: pins the exact set difference (``test_our_checks_cover_every_label_check_reslot_reports``).
PLAYER_CHECK_NAMES = ("player_present", "player_phase", "player_fields",
                      "player_range")
#: How ``chosen`` is picked, quoted into every row so the rule cannot drift.
CHOSEN_RULE = (
    "argmax of the teacher's score vector over the verified candidates "
    "(verified_mask[i] == 1), ties broken towards the lower candidate index; "
    "when nothing verified, the argmax over every finite score with "
    "chosen_verified=false; chosen=null only when no candidate has a finite score")

EXIT_OK = 0
#: The replay disagreed with the corpus (a state hash, the entry, a frame).
EXIT_STATE_FAILED = 1
#: The teacher or the filesystem failed; the run stopped, the labels written stay.
EXIT_TECHNICAL = 2
#: Refused before an engine was started (source self-check, engine sha, paths).
EXIT_REFUSED = 4
#: The four mute bytes were not 0 (``AGENTS.md`` section 9).
EXIT_MUTE = 5

#: Stop word -> exit code.  Every stop a ``meta.json`` may carry is in here; an
#: unlisted one is a technical failure, never a success.
STOP_EXIT: dict[str, int] = {
    "complete": EXIT_OK,
    "limit_reached": EXIT_OK,
    # -- the replay disagreed with the corpus --------------------------------
    "entry_hash_mismatch": EXIT_STATE_FAILED,
    "student_hash_mismatch": EXIT_STATE_FAILED,
    "frame_mismatch": EXIT_STATE_FAILED,
    # -- refused before the engine start -------------------------------------
    "source_missing": EXIT_REFUSED,
    "source_invalid": EXIT_REFUSED,
    "source_already_labelled": EXIT_REFUSED,
    "tape_invalid": EXIT_REFUSED,
    "entry_frame_invalid": EXIT_REFUSED,
    "entry_frame_uncovered": EXIT_REFUSED,
    "engine_missing": EXIT_REFUSED,
    "engine_sha_mismatch": EXIT_REFUSED,
    "data_dir_missing": EXIT_REFUSED,
    "out_dir_invalid": EXIT_REFUSED,
    "out_dir_not_empty": EXIT_REFUSED,
    "config_invalid": EXIT_REFUSED,
    "mute_refused": EXIT_MUTE,
    # -- the run started and then failed -------------------------------------
    "reset_not_ready": EXIT_TECHNICAL,
    "child_hazards_unavailable": EXIT_TECHNICAL,
    "player_unavailable": EXIT_TECHNICAL,
    "player_frame_mismatch": EXIT_TECHNICAL,
    "protocol_error": EXIT_TECHNICAL,
    "step_failed": EXIT_TECHNICAL,
    "teacher_query_failed": EXIT_TECHNICAL,
    "teacher_output_invalid": EXIT_TECHNICAL,
    "label_invalid": EXIT_TECHNICAL,
    "write_failed": EXIT_TECHNICAL,
    "technical_error": EXIT_TECHNICAL,
}

#: Every per-row check :func:`check_label_fields` reports, in report order.  The
#: first eight are ``touhou_rl/v420/reslot.py``'s label checks under the same
#: names (see ``reslot.CHECK_NAMES``); the rest are this format's additions.
CHECK_NAMES = (
    "scores_length",
    "scores_finite",
    "verified_mask",
    "verified_count",
    "verified_argmax",
    "chosen_range",
    "bomb",
    "margin",
    "label_source",
    "label_verified",
    "teacher_bomb",
    "teacher_profile",
    "executed_preserved",
    "nonfinite_consistent",
    "relabel_block",
)

#: The ``null`` fields a *source* row legitimately carries, and what fills them.
SOURCE_NULL_FIELDS: tuple[tuple[str, str], ...] = (
    ("slot_id", "a student decision is not a T04 slot; this tool does not "
                "invent one (join on episode_seed/decision_frame/hash_pre)"),
    ("shard_key", "the source window is not a T04 shard"),
    ("provenance", "there is no T04 corpus slot to point at"),
    ("scores", "filled by this tool: the teacher's 18-way score vector"),
    ("verified_mask", "filled by this tool: the per-candidate verification mask"),
    ("verified_count", "filled by this tool"),
    ("chosen", "filled by this tool (see relabel.chosen_rule)"),
    ("chosen_verified", "filled by this tool"),
    ("chosen_score", "filled by this tool"),
    ("teacher_bomb", "filled by this tool: the teacher's bomb channel"),
    ("margin", "filled by this tool"),
    ("label_verified", "filled by this tool: verified_count > 0"),
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class RelabelError(RuntimeError):
    """An expected failure of the relabel, with the stop word it exits on."""

    def __init__(self, message: str, *, stop: str = "technical_error",
                 detail: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.stop = str(stop)
        self.detail = dict(detail or {})


class SourceRefused(RelabelError):
    """The ``--save-obs`` corpus does not satisfy the contract this tool needs."""

    def __init__(self, message: str, *, stop: str = "source_invalid",
                 detail: Mapping[str, Any] | None = None) -> None:
        super().__init__(message, stop=stop, detail=detail)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    """UTC timestamp with a ``Z`` suffix -- the tool's only clock format."""

    return datetime.now(tz=None).astimezone().isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file (the ``teacher.engine`` implementation, re-used)."""

    from touhou_rl.v400.teacher.engine import sha256_file as engine_sha256  # noqa: PLC0415

    return engine_sha256(path)


_TOOL_SHA: str | None = None


def tool_sha() -> str:
    """This file's sha256, read once (it is stamped into every label row)."""

    global _TOOL_SHA
    if _TOOL_SHA is None:
        _TOOL_SHA = sha256_file(TOOL_PATH)
    return _TOOL_SHA


def _is_hash(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 16 and all(char in "0123456789abcdef" for char in text)


def hash_token(value: Any, where: str) -> str:
    """A 16-hex state hash or a refusal naming the row."""

    text = str(value or "").strip().lower()
    if not _is_hash(text):
        raise SourceRefused(
            f"{where}: the state hash is {value!r}, not 16 lowercase hex digits; the "
            "row cannot prove which state its label belongs to", stop="source_invalid")
    return text


def _int_field(row: Mapping[str, Any], key: str, where: str) -> int:
    if key not in row:
        raise SourceRefused(f"{where}: {key} is missing from the row")
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceRefused(f"{where}: {key} is {value!r}, not an integer")
    return int(value)


# ---------------------------------------------------------------------------
# The source corpus (a ``--save-obs`` directory)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StudentRow:
    """One ``slots.jsonl`` row, validated, plus the raw payload it came from."""

    index: int
    decision_frame: int
    decision_index: int
    cand: int
    bomb: int
    masks: tuple[int, ...]
    hash_pre: str
    hash_post: str
    episode_seed: int
    payload: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "decision_frame": self.decision_frame,
                "cand": self.cand, "bomb": self.bomb,
                "masks": list(self.masks), "hash_pre": self.hash_pre,
                "hash_post": self.hash_post}


@dataclass(frozen=True)
class SourceCorpus:
    """A validated student rollout: the states the teacher will be asked about."""

    root: Path
    slots_path: Path
    slots_sha256: str
    meta_path: Path
    meta_sha256: str
    meta: Mapping[str, Any]
    rows: tuple[StudentRow, ...]
    seed: int
    entry_frame: int
    shoot_pulse_period: int
    stage: int | None
    difficulty: int | None

    @property
    def count(self) -> int:
        return len(self.rows)

    def identity(self) -> dict[str, Any]:
        return {
            "root": self.root.as_posix(),
            "slots": {"file": self.slots_path.name, "path": self.slots_path.as_posix(),
                      "sha256": self.slots_sha256, "rows": int(self.count)},
            "meta": {"file": self.meta_path.name, "path": self.meta_path.as_posix(),
                     "sha256": self.meta_sha256},
            "seed": int(self.seed),
            "entry_frame": int(self.entry_frame),
            "shoot_pulse_period": int(self.shoot_pulse_period),
            "first_decision_frame": int(self.rows[0].decision_frame),
            "last_decision_frame": int(self.rows[-1].decision_frame),
            "schema": self.meta.get("schema"),
            "source_label_source": self.rows[0].payload.get("label_source"),
            "source_stop": self.meta.get("stop"),
            "source_complete": self.meta.get("complete"),
            "classification": "measured",
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every non-blank line of ``path`` as a JSON object (a refusal, not a skip)."""

    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SourceRefused(f"cannot read {path.as_posix()}: {exc}",
                            stop="source_missing") from exc
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SourceRefused(f"{path.as_posix()}: line {number} is not JSON: {exc}")
        if not isinstance(payload, dict):
            raise SourceRefused(
                f"{path.as_posix()}: line {number} is a {type(payload).__name__}, not a "
                "JSON object")
        rows.append(payload)
    return rows


def _check_row(index: int, row: Mapping[str, Any], *, first_frame: int,
               seed: int, period: int) -> StudentRow:
    """One row against the two-frame roster; every refusal names the row."""

    where = f"{tk.OBS_SLOTS_FILE} row {index}"
    frame = _int_field(row, "decision_frame", where)
    where = f"{tk.OBS_SLOTS_FILE} row {index} (decision_frame {frame})"
    if frame % DECISION_FRAMES:
        raise SourceRefused(
            f"{where}: decision_frame is odd; a student decision is on the "
            f"{DECISION_FRAMES}-frame grid")
    expected = first_frame + DECISION_FRAMES * index
    if frame != expected:
        raise SourceRefused(
            f"{where}: the two-frame grid is broken, expected frame {expected} at "
            f"row {index}")
    commit = row.get("commit_frames")
    if list(commit or []) != [frame, frame + DECISION_FRAMES]:
        raise SourceRefused(
            f"{where}: commit_frames {commit!r} is not [{frame}, "
            f"{frame + DECISION_FRAMES}]")
    duration = row.get("duration_frames", row.get("frames_per_decision"))
    if duration is not None and int(duration) != DECISION_FRAMES:
        raise SourceRefused(
            f"{where}: duration_frames is {duration!r}, the contract is "
            f"{DECISION_FRAMES}")
    row_seed = _int_field(row, "episode_seed", where)
    if row_seed != seed:
        raise SourceRefused(
            f"{where}: episode_seed {row_seed} is not the requested seed {seed}; a "
            "label taken on another episode's state would be attached to the wrong "
            "trajectory")
    cand = _int_field(row, "executed_cand", where)
    if not 0 <= cand < N_CAND:
        raise SourceRefused(f"{where}: executed_cand {cand} outside 0..{N_CAND - 1}")
    bomb = _int_field(row, "bomb", where)
    if bomb not in (0, 1):
        raise SourceRefused(f"{where}: bomb {bomb} is not the binary channel")
    mask = _int_field(row, "mask", where)
    masks = row.get("masks")
    if not isinstance(masks, list) or not masks:
        raise SourceRefused(
            f"{where}: masks is {masks!r}; the replay needs the exact per-frame "
            "masks the student sent")
    derived = tk.checkpoint_masks(cand, bomb, frame // DECISION_FRAMES, period)
    observed = [int(value) for value in masks]
    if observed != derived:
        raise SourceRefused(
            f"{where}: the recorded masks {observed} are not the ones "
            f"checkpoint_masks(cand={cand}, bomb={bomb}, decision="
            f"{frame // DECISION_FRAMES}, period={period}) derives {derived}; "
            "--shoot-pulse-period or the row's action cannot reproduce the "
            "student's frames")
    if observed[0] != mask:
        raise SourceRefused(
            f"{where}: masks[0] {observed[0]} is not the row's mask {mask}")
    executed = row.get("executed")
    if not isinstance(executed, Mapping):
        raise SourceRefused(
            f"{where}: executed is {executed!r}; the student's own action must be on "
            "the row (it is what the label is taken next to)")
    try:
        executed_cand = int(executed.get("cand", cand))
        executed_bomb = int(executed.get("bomb", bomb))
        executed_mask = int(executed.get("mask", mask))
    except (TypeError, ValueError) as exc:
        raise SourceRefused(
            f"{where}: executed {executed!r} does not hold integers: {exc}") from exc
    if executed_cand != cand or executed_bomb != bomb or executed_mask != mask:
        raise SourceRefused(
            f"{where}: executed {(executed_cand, executed_bomb, executed_mask)} is not "
            f"the row's (cand, bomb, mask) {(cand, bomb, mask)}")
    label_source = row.get("label_source")
    if label_source != STUDENT_LABEL_SOURCE:
        raise SourceRefused(
            f"{where}: label_source is {label_source!r}, not {STUDENT_LABEL_SOURCE!r}; "
            "this tool relabels a student rollout, and re-labelling an already "
            "labelled row would overwrite a measurement",
            stop="source_already_labelled")
    if row.get("scores") is not None:
        raise SourceRefused(
            f"{where}: the row already carries a score vector; relabelling it would "
            "overwrite a measurement", stop="source_already_labelled")
    return StudentRow(index=index, decision_frame=frame,
                      decision_index=frame // DECISION_FRAMES, cand=cand, bomb=bomb,
                      masks=tuple(observed),
                      hash_pre=hash_token(row.get("hash_pre"), where),
                      hash_post=hash_token(row.get("hash_post"), where),
                      episode_seed=row_seed, payload=dict(row))


def load_source(root: str | Path, *, seed: int, entry_frame: int,
                period: int = DEFAULT_PULSE_PERIOD) -> SourceCorpus:
    """Read and validate one ``--save-obs`` directory (the source self-check).

    Every check names the row it refused, because this directory is the only
    evidence of which state a label belongs to: a silently accepted anomaly would
    attach every later label to a state the student never stood on.
    """

    target = Path(root)
    if not target.is_dir():
        raise SourceRefused(f"--save-obs directory does not exist: {target.as_posix()}",
                            stop="source_missing")
    slots_path = target / tk.OBS_SLOTS_FILE
    meta_path = target / tk.OBS_META_FILE
    for path, label in ((slots_path, "slots.jsonl"), (meta_path, "meta.json")):
        if not path.is_file():
            raise SourceRefused(
                f"{target.as_posix()} has no {label}; the directory is not a "
                "--save-obs corpus", stop="source_missing")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceRefused(f"cannot read {meta_path.as_posix()}: {exc}",
                            stop="source_missing") from exc
    if not isinstance(meta, Mapping):
        raise SourceRefused(f"{meta_path.as_posix()} does not hold a JSON object")
    if meta.get("schema") != tk.SCHEMA_OBS_META:
        raise SourceRefused(
            f"{meta_path.as_posix()}: schema {meta.get('schema')!r} is not "
            f"{tk.SCHEMA_OBS_META!r}; this tool only relabels a --save-obs corpus")
    meta_seed = meta.get("seed")
    if meta_seed is None:
        raise SourceRefused(f"{meta_path.as_posix()} records no seed")
    if int(meta_seed) != int(seed):
        raise SourceRefused(
            f"--seed {seed} is not the corpus's seed {int(meta_seed)}; RESET would "
            "replay a different episode")
    meta_entry = meta.get("entry_frame")
    if meta_entry is None:
        raise SourceRefused(f"{meta_path.as_posix()} records no entry_frame")
    if int(meta_entry) != int(entry_frame):
        raise SourceRefused(
            f"--entry-frame {entry_frame} is not the entry the corpus was recorded "
            f"at ({int(meta_entry)}); the prefix would land on another state")
    if meta.get("frames_per_decision") not in (None, DECISION_FRAMES):
        raise SourceRefused(
            f"{meta_path.as_posix()}: frames_per_decision "
            f"{meta.get('frames_per_decision')!r} is not {DECISION_FRAMES}")
    meta_period = int(meta.get("shoot_pulse_period") or period)
    if meta_period != int(period):
        raise SourceRefused(
            f"--shoot-pulse-period {period} is not the corpus's {meta_period}; the "
            "student's masks were derived with the corpus's phase")
    raw = read_jsonl(slots_path)
    if not raw:
        raise SourceRefused(f"{slots_path.as_posix()} has no rows")
    declared = meta.get("decisions")
    if declared is None:
        raise SourceRefused(f"{meta_path.as_posix()} records no decision count")
    if int(declared) != len(raw):
        raise SourceRefused(
            f"{slots_path.as_posix()} has {len(raw)} rows but {meta_path.name} "
            f"declares {int(declared)} decisions; the corpus is incomplete or was "
            "edited")
    first_frame = _int_field(raw[0], "decision_frame", f"{slots_path.name} row 0")
    if first_frame != int(entry_frame):
        raise SourceRefused(
            f"{slots_path.name} row 0 sits on frame {first_frame}, not the entry "
            f"frame {entry_frame}; the first labelled decision must be the entry")
    rows = tuple(_check_row(index, row, first_frame=first_frame, seed=int(seed),
                            period=int(period))
                 for index, row in enumerate(raw))
    breaks: list[dict[str, Any]] = []
    for left, right in zip(rows, rows[1:]):
        if left.hash_post != right.hash_pre:
            breaks.append({"row": left.index, "next_row": right.index,
                           "hash_post": left.hash_post,
                           "next_hash_pre": right.hash_pre})
    if breaks:
        first = breaks[0]
        raise SourceRefused(
            f"{slots_path.as_posix()}: the hash chain breaks at row {first['row']}: "
            f"hash_post {first['hash_post']} is not row {first['next_row']}'s hash_pre "
            f"{first['next_hash_pre']}; {len(breaks)} break(s). The rows would not "
            "describe one trajectory")
    return SourceCorpus(
        root=target, slots_path=slots_path, slots_sha256=sha256_file(slots_path),
        meta_path=meta_path, meta_sha256=sha256_file(meta_path), meta=dict(meta),
        rows=rows, seed=int(seed), entry_frame=int(entry_frame),
        shoot_pulse_period=int(period),
        stage=(None if meta.get("stage") is None else int(meta["stage"])),
        difficulty=(None if meta.get("difficulty") is None
                    else int(meta["difficulty"])))


# ---------------------------------------------------------------------------
# Configuration and the injectable seams
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RelabelConfig:
    """Where the states come from, which teacher asks, and where labels go."""

    save_obs: Path
    tape_slots: Path
    seed: int
    entry_frame: int
    engine: Path
    expected_sha: str
    data_dir: Path
    out: Path
    stage: int = RESET_STAGE
    difficulty: int = RESET_DIFFICULTY
    team: int = RESET_TEAM
    limit: int | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    shoot_pulse_period: int = DEFAULT_PULSE_PERIOD
    profile: str = DEFAULT_PROFILE
    #: Make the ``PLAYER`` coordinate part of the contract: a boundary whose
    #: ``px``/``py`` cannot be read refuses the run instead of writing ``null``.
    require_player: bool = False
    argv: tuple[str, ...] = ()

    def validate(self) -> None:
        """Refuse the numbers argparse happily accepts but the contract cannot."""

        if int(self.seed) < 0:
            raise RelabelError(f"--seed {self.seed} is negative", stop="config_invalid")
        if int(self.entry_frame) % DECISION_FRAMES:
            raise RelabelError(
                f"--entry-frame {self.entry_frame} is odd; the tape and the student "
                "rollout are on a two-frame grid", stop="entry_frame_invalid")
        if int(self.entry_frame) < tk.TAPE_FIRST_FRAME:
            raise RelabelError(
                f"--entry-frame {self.entry_frame} is before the tape's first slot "
                f"frame {tk.TAPE_FIRST_FRAME}", stop="entry_frame_invalid")
        if self.limit is not None:
            if isinstance(self.limit, bool) or not isinstance(self.limit, int):
                raise RelabelError(f"--limit {self.limit!r} is not an integer",
                                   stop="config_invalid")
            if self.limit <= 0:
                raise RelabelError(
                    f"--limit {self.limit} must be positive (it is a smoke-run cap)",
                    stop="config_invalid")
        if int(self.shoot_pulse_period) <= 0:
            raise RelabelError(
                f"--shoot-pulse-period {self.shoot_pulse_period} must be positive",
                stop="config_invalid")
        if float(self.timeout_s) <= 0:
            raise RelabelError(f"--timeout-s {self.timeout_s} must be positive",
                               stop="config_invalid")
        if not str(self.expected_sha).strip():
            raise RelabelError("--expected-sha is empty", stop="config_invalid")


@dataclass(frozen=True)
class RelabelDeps:
    """Every seam that would otherwise need a real engine or a real teacher.

    ``None`` means "the shipped implementation" (:meth:`resolve` fills it in).
    ``tests/test_v8_32_relabel.py`` drives all of them with fakes, so the
    orchestration is pinned without a process: the engine client, the teacher
    call, the teacher profile, the mute reader, the engine gate, the label sink
    and the clock.

    ``player_reader`` is the ``PLAYER`` seam: it takes the engine and returns the
    read record (:func:`default_player_reader` is the shipped one).  A caller that
    leaves it ``None`` gets that default, which reports ``available=False`` for an
    engine object with no protocol surface -- so a fake engine still runs the
    orchestration, and ``--require-player`` is what turns a missing read into a
    refusal.
    """

    client_factory: Callable[[RelabelConfig], Any] | None = None
    decide_call: Callable[..., Any] | None = None
    teacher_config: Callable[[RelabelConfig], Any] | None = None
    mute_reader: Callable[[Path], Mapping[str, Any]] | None = None
    engine_gate: Callable[[RelabelConfig], Mapping[str, Any]] | None = None
    writer_factory: Callable[[Path], Any] | None = None
    clock: Callable[[], float] | None = None
    tolerance: Callable[[], float] | None = None
    player_reader: Callable[[Any], Mapping[str, Any]] | None = None

    def resolve(self) -> "RelabelDeps":
        return RelabelDeps(
            client_factory=self.client_factory or default_client_factory,
            decide_call=self.decide_call or default_decide_call,
            teacher_config=self.teacher_config or default_teacher_config,
            mute_reader=self.mute_reader or tk.read_mute,
            engine_gate=self.engine_gate or default_engine_gate,
            writer_factory=self.writer_factory or default_writer_factory,
            clock=self.clock or time.perf_counter,
            tolerance=self.tolerance or teacher_tolerance,
            player_reader=self.player_reader or default_player_reader,
        )


# ---------------------------------------------------------------------------
# The plan: where the prefix ends and the student's states begin
# ---------------------------------------------------------------------------
def build_plan(tape: Mapping[str, Any], source: SourceCorpus
               ) -> dict[str, Any]:
    """The takeover's entry plan plus the two anchors this tool must also match."""

    plan = tk.entry_plan(tape, int(source.entry_frame))
    row0 = source.rows[0]
    if int(row0.decision_frame) != int(plan["entry_frame"]):
        raise RelabelError(
            f"row 0 sits on frame {row0.decision_frame}, the plan enters at "
            f"{plan['entry_frame']}", stop="entry_frame_invalid")
    if str(plan["anchor_hash"]) != row0.hash_pre:
        raise RelabelError(
            f"the tape's entry hash {plan['anchor_hash']} is not row 0's hash_pre "
            f"{row0.hash_pre}; the student rollout was not recorded from this tape's "
            "entry, so no label taken here would describe its states",
            stop="entry_hash_mismatch",
            detail={"tape_anchor_hash": str(plan["anchor_hash"]),
                    "student_hash_pre": row0.hash_pre})
    return plan


# ---------------------------------------------------------------------------
# Engine seams
# ---------------------------------------------------------------------------
class PrefixClient:
    """The takeover replayer's client shape, over a teacher engine.

    :func:`v8_32_takeover.replay_prefix` speaks ``client.step(mask, 1)`` and
    ``client.state_hash() -> (hash, frame)``; ``TeacherEngine`` speaks ``step``/
    ``hash``/``frame``.  This adapter is the single place the two shapes meet, so
    the prefix replayer itself is re-used unchanged (its per-slot ``HASH`` check
    included) instead of being re-implemented here.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def step(self, mask: int, frames: int = 1) -> Any:
        return self.engine.step(int(mask), int(frames))

    def state_hash(self) -> tuple[str, int]:
        return (str(self.engine.hash()).strip().lower(), int(self.engine.frame()))

    def frame(self) -> int:
        return int(self.engine.frame())


def default_engine_gate(config: RelabelConfig) -> Mapping[str, Any]:
    """Resolve ``--engine`` and enforce ``--expected-sha`` before anything starts."""

    from touhou_rl.v400.teacher.engine import (  # noqa: PLC0415
        EngineShaMismatch,
        check_engine_binary,
    )

    try:
        path = tk.resolve_engine(config.engine)
    except tk.PreflightRefused as exc:
        raise RelabelError(str(exc), stop="engine_missing") from exc
    try:
        gate = check_engine_binary(path, expected=str(config.expected_sha))
    except EngineShaMismatch as exc:
        raise RelabelError(
            str(exc), stop="engine_sha_mismatch",
            detail={"engine": {"path": Path(path).as_posix(),
                               "binary": Path(path).as_posix(),
                               "sha256": sha256_file(path),
                               "expected_sha256": str(config.expected_sha),
                               "matches": False, "role": "shadow"}}) from exc
    return {"path": str(gate.get("path") or path), "sha256": str(gate.get("sha256")),
            "expected_sha256": str(config.expected_sha),
            "matches": bool(gate.get("match")), "role": "shadow",
            "binary": Path(path).as_posix()}


def default_client_factory(config: RelabelConfig) -> Any:
    """The real teacher engine: one shadow process, no pixels, no GL, no audio.

    Mirrors ``touhou_rl/v420/counterfactual.py::_Session.start`` -- the shadow role
    is what serves ``PARK``/``SPAWN``/``CHILD`` and therefore ``decide()``.  The
    binary is resolved through :func:`v8_32_takeover.resolve_engine` so
    ``--engine <build-dir>`` reaches the engine as a file, never as a directory.
    """

    from touhou_rl.v400.teacher.engine import TeacherEngine  # noqa: PLC0415

    binary = tk.resolve_engine(config.engine)
    engine = TeacherEngine(binary, config.data_dir, role="shadow",
                           timeout=float(config.timeout_s))
    engine.start()
    engine.hello()
    return engine


def default_teacher_config(config: RelabelConfig) -> Any:
    """The measured two-frame profile: ``fast`` with ``replan_decisions=1``."""

    from touhou_rl.v400.teacher.config import get_config  # noqa: PLC0415

    base = get_config(str(config.profile))
    return dataclasses.replace(base, replan_decisions=1,
                               shoot_pulse_period=int(config.shoot_pulse_period))


def default_decide_call(engine: Any, config: Any, **kwargs: Any) -> Any:
    """``touhou_rl.v400.teacher.decide.decide`` (imported, never re-implemented)."""

    from touhou_rl.v400.teacher.decide import decide  # noqa: PLC0415

    return decide(engine, config, **kwargs)


def teacher_tolerance() -> float:
    """``verify.verify_plans``' effective ``tolerance``, read off the function.

    ``decide()`` calls ``verify_plans`` with its default and exposes no switch, so
    the only honest way to *record* the tolerance is to read the default that is
    really in force.  A drifted default is recorded as drifted.
    """

    import inspect  # noqa: PLC0415

    from touhou_rl.v400.teacher.verify import verify_plans  # noqa: PLC0415

    parameter = inspect.signature(verify_plans).parameters.get("tolerance")
    if parameter is None or parameter.default is inspect.Parameter.empty:
        raise RelabelError("verify_plans has no tolerance parameter",
                           stop="teacher_output_invalid")
    return float(parameter.default)


def probe_engine(engine: Any) -> dict[str, Any]:
    """Capability + child-hazard probe, once per process, right after ``RESET``.

    ``record_child_tape`` calls ``probe_child_hazards`` itself and that probe
    ``PARK``s the shadow, releasing the template ``decide`` just took -- the
    recorded failure is ``ERR spawn-template expected=2``.  Probing once here is
    what makes the first ``decide()`` work (the same order
    ``touhou_rl/v420/counterfactual.py::_Session.probe_once`` and
    ``reslot._reslot_one`` use).
    """

    try:
        capabilities = engine.probe_capabilities()
        child_hazards = engine.probe_child_hazards()
    except Exception as exc:  # noqa: BLE001 - a refused probe is a stop
        raise RelabelError(f"the engine probe failed: {type(exc).__name__}: {exc}",
                           stop="protocol_error") from exc
    if not child_hazards.get("available"):
        raise RelabelError(
            f"the engine cannot serve CHILD <h> HAZARDS "
            f"({child_hazards.get('reason')!r}); the teacher's child recording would "
            "fail on every query", stop="child_hazards_unavailable")
    return {"capabilities": dict(capabilities or {}),
            "child_hazards": dict(child_hazards or {})}


def close_engine(engine: Any) -> dict[str, Any]:
    """Close the engine; a dead pipe is recorded, never raised."""

    if engine is None:
        return {"closed": False, "exit_code": None, "error": None}
    try:
        code = engine.close()
        return {"closed": True, "exit_code": (-1 if code is None else int(code)),
                "error": None}
    except Exception as exc:  # noqa: BLE001 - a dead pipe is not a result
        return {"closed": False, "exit_code": None,
                "error": f"{type(exc).__name__}: {exc}"}


def reset_engine(engine: Any, config: RelabelConfig) -> dict[str, Any]:
    """``RESET`` and the frame-0 / playable checks, then the probe."""

    try:
        reply = engine.reset(stage=int(config.stage), difficulty=int(config.difficulty),
                             team=int(config.team), seed=int(config.seed))
    except Exception as exc:  # noqa: BLE001 - a refused RESET is a stop
        raise RelabelError(f"RESET failed: {type(exc).__name__}: {exc}",
                           stop="protocol_error") from exc
    info = dict(reply) if isinstance(reply, Mapping) else {"reply": reply}
    if str(info.get("ready")) != "1":
        raise RelabelError(f"RESET did not reach a playable state: {info}",
                           stop="reset_not_ready")
    frame = int(engine.frame())
    if frame != 0:
        raise RelabelError(
            f"the engine sits on frame {frame} after RESET, not 0; the prefix replay "
            "would start from the wrong frame", stop="protocol_error")
    return {"reply": info, "hash": str(engine.hash()).strip().lower(), "frame": frame}


# ---------------------------------------------------------------------------
# The player coordinate: one ``PLAYER`` read per decision boundary
# ---------------------------------------------------------------------------
def play_area() -> dict[str, tuple[float, float]]:
    """The player-centre clamp, read off ``touhou_rl.spec`` (never a local copy)."""

    from touhou_rl import spec  # noqa: PLC0415

    return {"x": (float(spec.MOVE_AREA_X),
                  float(spec.MOVE_AREA_X + spec.MOVE_AREA_W)),
            "y": (float(spec.MOVE_AREA_Y),
                  float(spec.MOVE_AREA_Y + spec.MOVE_AREA_H))}


def reply_token(reply: str, name: str) -> str | None:
    """The raw ``name=<token>`` of one protocol reply -- **provenance only**.

    The value written to a row is the one
    :func:`touhou_rl.v400.gym.protocol.parse_player_reply` decoded; this helper
    only keeps the wire token next to it so a reader can see the bytes the
    coordinate came from (``px=%08x`` is the float32 bit pattern).
    """

    for token in str(reply).split():
        if token.startswith(f"{name}="):
            return token[len(name) + 1:]
    return None


def default_player_reader(engine: Any) -> Mapping[str, Any]:
    """Read ``PLAYER`` and decode ``px``/``py`` with the shipped decoder.

    ``ask`` + ``parse_player_reply`` is the same pair
    ``touhou_rl/v400/teacher/engine.py::TeacherEngine.player`` uses, so the field
    names recorded here are the ones on the wire.  An engine object with no
    ``ask`` (a test double) reports ``available=False`` instead of inventing a
    coordinate; ``--require-player`` is what turns that into a refusal.
    """

    ask = getattr(engine, "ask", None)
    if not callable(ask):
        return {"available": False, "command": PLAYER_COMMAND,
                "fields": list(PLAYER_FIELDS), "kind": "no_player_command",
                "reason": ("the engine object exposes no ask()/PLAYER command, so no "
                           "player coordinate can be read from it")}
    reply = str(ask(PLAYER_COMMAND))
    from touhou_rl.v400.gym.protocol import parse_player_reply  # noqa: PLC0415

    snapshot = parse_player_reply(reply)
    px, py = (float(snapshot.position[0]), float(snapshot.position[1]))
    return {"available": True, "command": PLAYER_COMMAND,
            "fields": list(PLAYER_FIELDS), "kind": PLAYER_COMMAND, "reason": None,
            "px": px, "py": py,
            "raw": {name: reply_token(reply, name) for name in PLAYER_FIELDS},
            "frame": int(snapshot.frame)}


def read_player_at_boundary(reader: Callable[[Any], Mapping[str, Any]], engine: Any, *,
                            row: StudentRow, index: int, hash_pre: str) -> dict[str, Any]:
    """One boundary ``PLAYER`` read, normalised to the per-row ``player`` block.

    Never raises: a refused read becomes ``available=False`` with the reason it
    carries, and :func:`player_gate` decides whether the run may go on.  The
    block is stamped with *when* the read happened (``read_phase``/``precedes``)
    and with the state it was taken on (``decision_frame``/``hash_pre``), so the
    timing claim travels with the number.
    """

    block: dict[str, Any] = {
        "schema": SCHEMA_PLAYER_BLOCK,
        "read_phase": PLAYER_READ_PHASE,
        "precedes": list(PLAYER_PRECEDES),
        "command": PLAYER_COMMAND,
        "fields": list(PLAYER_FIELDS),
        "read_index": int(index),
        "decision_frame": int(row.decision_frame),
        "hash_pre": str(hash_pre),
        "px": None, "py": None, "raw": None,
        "frame": None, "frame_matches": None,
        "available": False, "kind": None, "reason": None,
    }
    try:
        record = dict(reader(engine) or {})
    except Exception as exc:  # noqa: BLE001 - a refused read is recorded, not raised
        block["kind"] = "read_failed"
        block["reason"] = f"{type(exc).__name__}: {exc}"
        return block
    for key in ("available", "kind", "reason", "px", "py", "raw", "frame"):
        if key in record:
            block[key] = record[key]
    block["available"] = bool(block.get("available"))
    for name in PLAYER_FIELDS:
        value = block.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            block[name] = round(float(value), 6)
    frame = block.get("frame")
    if isinstance(frame, int) and not isinstance(frame, bool):
        block["frame_matches"] = int(frame) == int(row.decision_frame)
    return block


def player_gate(block: Mapping[str, Any], *, require: bool, where: str) -> None:
    """Fail closed on a coordinate that would be missing or from another frame.

    A ``PLAYER`` reply carries the engine's own ``frames=`` counter, so a read
    whose frame is not the decision frame is refused **always** -- that
    coordinate belongs to another state and any position-conditioned reading of
    the label would be wrong.  ``require`` (``--require-player``) additionally
    refuses a read that simply did not happen.
    """

    if block.get("frame_matches") is False:
        raise RelabelError(
            f"{where}: the PLAYER reply reports frame {block.get('frame')}, the "
            f"decision sits on frame {block.get('decision_frame')}; the coordinate "
            "would belong to another state",
            stop="player_frame_mismatch",
            detail={"player": dict(block)})
    if require and not block.get("available"):
        raise RelabelError(
            f"{where}: --require-player is set but the PLAYER coordinate could not be "
            f"read ({block.get('reason')!r}); a label without the player's position "
            "would not answer the question this corpus exists for",
            stop="player_unavailable", detail={"player": dict(block)})


def player_block_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The ``meta.json`` player block: what was read, how often, and its spread."""

    px = [float(row["px"]) for row in rows if isinstance(row.get("px"), (int, float))]
    py = [float(row["py"]) for row in rows if isinstance(row.get("py"), (int, float))]
    area = play_area()
    out_of_range = [row for row in rows
                    if isinstance(row.get("px"), (int, float))
                    and not (area["x"][0] <= float(row["px"]) <= area["x"][1]
                             and area["y"][0] <= float(row["py"]) <= area["y"][1])]
    mismatched = [row for row in rows if row.get("frame_matches") is False]

    def spread(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"min": None, "median": None, "max": None, "count": 0}
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (ordered[middle] if len(ordered) % 2
                  else (ordered[middle - 1] + ordered[middle]) / 2.0)
        return {"min": round(ordered[0], 6), "median": round(median, 6),
                "max": round(ordered[-1], 6), "count": len(ordered)}

    return {
        "rows_scanned": len(rows),
        "rows_with_coordinates": len(px),
        "rows_missing": len(rows) - len(px),
        "available": bool(px) and len(px) == len(rows),
        "reason": next((row.get("reason") for row in rows
                        if not isinstance(row.get("px"), (int, float))), None),
        "px": spread(px), "py": spread(py),
        "play_area": {axis: [lo, hi] for axis, (lo, hi) in area.items()},
        "range_violations": len(out_of_range),
        "frame_mismatches": len(mismatched),
        "read_phases": sorted({str(row.get("read_phase")) for row in rows}),
    }


def player_meta_block(config: "RelabelConfig") -> dict[str, Any]:
    """The ``meta.json`` record of *how* the coordinate is read (command/fields/when)."""

    return {
        "require_player": bool(config.require_player),
        "read": {
            "command": PLAYER_COMMAND,
            "fields": list(PLAYER_FIELDS),
            "read_phase": PLAYER_READ_PHASE,
            "precedes": list(PLAYER_PRECEDES),
            "decoder": "touhou_rl.v400.gym.protocol.parse_player_reply",
            "timing": ("one read per decision boundary on the authoritative process, "
                       "after the engine's hash was checked against the row's hash_pre "
                       "and before decide() and before every STEP of that decision"),
            "protocol_reply": ("OK frames=<n> hp=<f32bits> lives=<n> bombs=<f32bits> "
                               "power=<f32bits> focus=<n> isYoukai=<n> playerState=<n> "
                               "invulnState=<n> px=<f32bits> py=<f32bits> ..."),
            "units": "px in the engine's own coordinate space; the play area is the "
                     "player-centre clamp from touhou_rl/spec.py",
            "read_only": ("HandlePlayer only reads g_Player/g_GameManager.globals; it "
                          "advances nothing"),
        },
        "rows_scanned": 0, "rows_with_coordinates": 0, "rows_missing": 0,
        "available": None, "reason": None,
        "px": {"min": None, "median": None, "max": None, "count": 0},
        "py": {"min": None, "median": None, "max": None, "count": 0},
        "play_area": {axis: list(bounds) for axis, bounds in play_area().items()},
        "range_violations": 0, "frame_mismatches": 0, "read_phases": [],
    }


def check_player_fields(payload: Mapping[str, Any]) -> dict[str, str]:
    """Run the four player-coordinate checks; return ``{check: error}`` (empty = valid).

    Separate from :func:`check_label_fields` on purpose: those fifteen names are
    the ``reslot`` label checks plus this format's, and a caller (or
    ``tests/test_v8_32_relabel.py``) compares the two rosters exactly.  The names
    reported here are :data:`PLAYER_CHECK_NAMES`.
    """

    errors: dict[str, str] = {}
    player = payload.get("player")

    def run(name: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except RelabelError as exc:
            errors.setdefault(name, str(exc))
        except (KeyError, TypeError, ValueError) as exc:
            errors.setdefault(name, f"{type(exc).__name__}: {exc}")

    def expect(condition: Any, message: str) -> None:
        if not condition:
            raise RelabelError(message)

    def present() -> None:
        expect(isinstance(player, Mapping),
               f"player is {player!r}, not the coordinate block")
        for key in ("px", "py", "read_phase", "decision_frame"):
            expect(key in player, f"player.{key} is missing from the row")

    def phase() -> None:
        expect((player or {}).get("read_phase") == PLAYER_READ_PHASE,
               f"player.read_phase {(player or {}).get('read_phase')!r} is not "
               f"{PLAYER_READ_PHASE!r}; the coordinate must be read before the action")
        frame_matches = (player or {}).get("frame_matches")
        expect(frame_matches is not False,
               "player.frame_matches is false: the PLAYER reply was taken on another "
               f"frame ({player.get('frame')} vs {player.get('decision_frame')})")

    def fields() -> None:
        expect((player or {}).get("command") == PLAYER_COMMAND,
               f"player.command {(player or {}).get('command')!r} is not "
               f"{PLAYER_COMMAND!r}")
        expect(list((player or {}).get("fields") or []) == list(PLAYER_FIELDS),
               f"player.fields {(player or {}).get('fields')!r} is not "
               f"{list(PLAYER_FIELDS)!r}; the names must be the protocol's")

    def coordinate_range() -> None:
        block = player or {}
        px, py = block.get("px"), block.get("py")
        if px is None or py is None:
            expect(px is None and py is None,
                   f"half a coordinate: px={px!r} py={py!r}")
            expect(block.get("available") is not True,
                   "player.available is true but the coordinate is null")
            return
        area = play_area()
        for name, value, bounds in (("px", px, area["x"]), ("py", py, area["y"])):
            expect(isinstance(value, (int, float)) and not isinstance(value, bool),
                   f"player.{name} is {value!r}, not a number")
            expect(math.isfinite(float(value)),
                   f"player.{name} is not finite: {value!r}")
            expect(bounds[0] <= float(value) <= bounds[1],
                   f"player.{name} {value!r} is outside the play area {list(bounds)}")

    run("player_present", present)
    run("player_phase", phase)
    run("player_fields", fields)
    run("player_range", coordinate_range)
    return errors


# ---------------------------------------------------------------------------
# The label: the teacher's answer, derived by this tool's documented rule
# ---------------------------------------------------------------------------
def derive_label(decision: Any) -> dict[str, Any]:
    """Normalise one ``Decision`` into the 18-way label + this tool's ``chosen``.

    The vector itself comes from
    :func:`touhou_rl.v420.reslot.decision_view` -- the *same* normaliser the T04
    reslot writes with -- so the two label sources cannot drift.  ``chosen`` is
    then re-derived here (see :data:`CHOSEN_RULE`), never copied, so that
    ``chosen`` is inside ``verified_mask`` by construction.
    """

    from touhou_rl.v420.reslot import decision_view  # noqa: PLC0415

    try:
        view = decision_view(decision)
    except Exception as exc:  # noqa: BLE001 - the teacher's shape is a contract
        raise RelabelError(
            f"the teacher's Decision cannot be normalised: {type(exc).__name__}: {exc}",
            stop="teacher_output_invalid") from exc
    if len(view.scores) != N_CAND:
        raise RelabelError(
            f"the teacher returned {len(view.scores)} scores, the action contract has "
            f"{N_CAND}", stop="teacher_output_invalid")
    if len(view.verified_mask) != N_CAND:
        raise RelabelError(
            f"the teacher returned a {len(view.verified_mask)}-entry verified_mask, "
            f"the action contract has {N_CAND}", stop="teacher_output_invalid")
    raw = [float(value) for value in view.scores]
    finite = [index for index, value in enumerate(raw) if math.isfinite(value)]
    finite_set = set(finite)
    nonfinite = [index for index in range(N_CAND) if index not in finite_set]
    mask = [1 if flag else 0 for flag in view.verified_mask]
    verified = [index for index, flag in enumerate(mask) if flag]
    for index in range(N_CAND):
        if mask[index] not in (0, 1):  # pragma: no cover - decision_view normalises
            raise RelabelError(
                f"verified_mask[{index}] is {mask[index]!r}, not a flag",
                stop="teacher_output_invalid")
    candidates = [index for index in verified if index in finite_set] if verified else []
    if verified and not candidates:
        raise RelabelError(
            f"the teacher verified candidates {verified} but none carries a finite "
            "score; the label could not name a chosen action",
            stop="teacher_output_invalid")
    chosen: int | None = None
    chosen_verified = False
    if verified:
        chosen = max(candidates, key=lambda index: (raw[index], -index))
        chosen_verified = bool(mask[chosen])
    elif finite:
        chosen = max(finite, key=lambda index: (raw[index], -index))
    chosen_score = None if chosen is None else round(float(raw[chosen]), 6)
    margin = float(view.margin)
    bomb = int(view.bomb)
    if bomb not in (0, 1):
        raise RelabelError(f"the teacher's bomb is {bomb}, not the binary channel",
                           stop="teacher_output_invalid")
    chosen_plan = getattr(decision, "chosen", None)
    teacher_chosen = getattr(chosen_plan, "first_action", None)
    teacher_chosen = None if teacher_chosen is None else int(teacher_chosen)
    teacher_score = getattr(chosen_plan, "score", None) if chosen_plan is not None else None
    label = getattr(decision, "label", None)
    events = list(getattr(label, "events", None) or [])
    timing = getattr(decision, "timing_ms", None)
    return {
        # -- the eleven fields a T04 slot record also carries -----------------
        "scores": [round(float(raw[index]), 6) if index in finite_set else None
                   for index in range(N_CAND)],
        "verified_mask": mask,
        "verified_count": int(sum(mask)),
        "chosen": chosen,
        "chosen_verified": bool(chosen_verified),
        "chosen_score": chosen_score,
        "margin": None if not math.isfinite(margin) else round(margin, 6),
        "bomb": bomb,
        "teacher_bomb": bomb,
        "label_source": LABEL_SOURCE,
        "label_verified": bool(sum(mask)),
        # -- this tool's own audit trail (never a policy input) ---------------
        "relabel_extra": {
            "nonfinite_scores": nonfinite,
            "chosen_rule": CHOSEN_RULE,
            "verified_indices": verified,
            "teacher": {
                "chosen": teacher_chosen,
                "chosen_score": (None if teacher_score is None
                                 else round(float(teacher_score), 6)),
                "chosen_verified": bool(view.chosen_verified),
                "chosen_duration": view.chosen_duration,
                "chosen_death_frame": view.chosen_death_frame,
                "margin": None if not math.isfinite(margin) else round(margin, 6),
                "bomb": bomb,
                "events": events,
                "timing_ms": ({key: round(float(value), 3)
                               for key, value in timing.items()}
                              if isinstance(timing, Mapping) else {}),
                "agrees_with_chosen": (teacher_chosen is not None
                                       and chosen is not None
                                       and int(teacher_chosen) == int(chosen)),
            },
        },
    }


def teacher_row_block(profile: Mapping[str, Any]) -> dict[str, Any]:
    """The compact per-row teacher identity (the full snapshot stays in meta.json).

    ``touhou_rl/v420/reslot.py::check_label_record``'s ``teacher_profile`` check
    reads ``replan_decisions`` and ``decision_frames`` off this block, so the two
    formats agree on what "a two-frame teacher profile" means.
    """

    return {"profile": profile.get("mode"), "mode": profile.get("mode"),
            "horizon": profile.get("horizon"), "beam": profile.get("beam"),
            "verify_k": profile.get("verify_k"),
            "max_iterations": profile.get("max_iterations"),
            "replan_decisions": profile.get("replan_decisions"),
            "decision_frames": profile.get("decision_frames"),
            "shoot_pulse_period": profile.get("shoot_pulse_period"),
            "profile_sha256": profile.get("sha256")}


def build_label_row(*, row: StudentRow, label: Mapping[str, Any],
                    teacher_profile: Mapping[str, Any], context: Mapping[str, Any],
                    player: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The source row verbatim + the label fields + provenance.

    The student's own measurement stays exactly where it was; only the fields a
    student rollout writes as ``null`` (and ``label_source``) are replaced, plus
    the top-level ``bomb``, which becomes the teacher's channel (reslot's
    convention -- ``executed.bomb`` keeps the student's).

    ``player`` is this format's own addition and the **only** key this function
    creates that the source row did not have: the student's position on the
    decision frame, read before the action (see the module docstring).  Nothing
    the source wrote is replaced by it.
    """

    payload = dict(row.payload)
    payload["schema"] = SCHEMA_LABEL
    payload["teacher"] = teacher_row_block(teacher_profile)
    for key in ("scores", "verified_mask", "verified_count", "chosen",
                "chosen_verified", "chosen_score", "margin", "bomb",
                "teacher_bomb", "label_source", "label_verified"):
        payload[key] = label[key]
    if player is not None:
        payload["player"] = dict(player)
    payload["relabel"] = dict(context)
    return payload


def check_label_fields(payload: Mapping[str, Any]) -> dict[str, str]:
    """Run every per-row label check; return ``{check: error}`` (empty = valid).

    The first eight entries of :data:`CHECK_NAMES` are lifted from
    ``touhou_rl/v420/reslot.py::check_label_record`` with the same names and the
    same rules -- including its ``verified_argmax`` and ``chosen_range`` rules --
    so a label taken here and a T04 label are audited by the same arithmetic.
    ``scores_length``/``scores_finite`` additionally accept a ``null`` entry when
    ``relabel.nonfinite_scores`` lists exactly that index (the documented
    difference: ``beam_search`` may leave ``-inf`` for a candidate it could not
    plan, and this format says so instead of inventing a floor).
    """

    errors: dict[str, str] = {}

    def run(name: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except RelabelError as exc:
            errors.setdefault(name, str(exc))
        except (KeyError, TypeError, ValueError) as exc:
            errors.setdefault(name, f"{type(exc).__name__}: {exc}")

    def expect(condition: Any, message: str) -> None:
        if not condition:
            raise RelabelError(message)

    relabel = payload.get("relabel")
    nonfinite = (list(relabel.get("nonfinite_scores") or [])
                 if isinstance(relabel, Mapping) else [])

    def scores_length() -> None:
        values = payload.get("scores")
        expect(isinstance(values, list) and len(values) == N_CAND,
               f"scores must be {N_CAND} entries, got "
               f"{None if values is None else len(values)}")

    run("scores_length", scores_length)

    def scores_finite() -> None:
        values = payload.get("scores") or []
        nulls: list[int] = []
        for index, value in enumerate(values):
            if value is None:
                nulls.append(index)
                continue
            expect(isinstance(value, (int, float)) and not isinstance(value, bool),
                   f"scores[{index}] is not a number: {value!r}")
            expect(math.isfinite(float(value)),
                   f"scores[{index}] is not finite: {value!r}; only a null entry is "
                   "allowed, and only when relabel.nonfinite_scores lists it")
        expect(nulls == sorted(int(index) for index in nonfinite),
               f"the null scores {nulls} are not relabel.nonfinite_scores {nonfinite}")

    run("scores_finite", scores_finite)

    def verified_mask() -> None:
        values = payload.get("verified_mask")
        expect(isinstance(values, list) and len(values) == N_CAND,
               f"verified_mask must be {N_CAND} flags, got "
               f"{None if values is None else len(values)}")
        for index, value in enumerate(values):
            expect(value in (0, 1, False, True),
                   f"verified_mask[{index}] is not a 0/1 flag: {value!r}")

    run("verified_mask", verified_mask)

    def verified_count() -> None:
        flags = payload.get("verified_mask") or ()
        expect(payload.get("verified_count") == sum(1 for value in flags if value),
               f"verified_count {payload.get('verified_count')!r} != the mask's "
               f"{sum(1 for value in flags if value)}")

    run("verified_count", verified_count)

    def argmax() -> None:
        flags = payload.get("verified_mask") or ()
        values = payload.get("scores") or ()
        verified = [index for index, flag in enumerate(flags) if flag]
        if not verified:
            return
        chosen = payload.get("chosen")
        expect(chosen is not None,
               "a slot with a verified candidate must have a chosen action")
        expect(isinstance(chosen, int) and not isinstance(chosen, bool),
               f"chosen {chosen!r} is not an int")
        expect(0 <= int(chosen) < len(flags),
               f"chosen {chosen!r} outside [0, {len(flags)})")
        expect(0 <= int(chosen) < len(values),
               f"chosen {chosen!r} outside the {len(values)}-entry score vector")
        expect(bool(flags[int(chosen)]),
               f"chosen {chosen} is not one of the verified candidates {verified}")
        finite = [index for index in verified if values[index] is not None]
        expect(bool(finite),
               f"none of the verified candidates {verified} carries a finite score")
        best = max(float(values[index]) for index in finite)
        expect(float(values[int(chosen)]) >= best - 1e-9,
               f"scores[chosen={chosen}] = {values[int(chosen)]} is below the best "
               f"verified score {best}")

    run("verified_argmax", argmax)

    def chosen_range() -> None:
        chosen = payload.get("chosen")
        expect(chosen is None
               or (isinstance(chosen, int) and not isinstance(chosen, bool)
                   and 0 <= int(chosen) < N_CAND),
               f"chosen {chosen!r} outside [0, {N_CAND})")

    run("chosen_range", chosen_range)

    run("bomb", lambda: expect(payload.get("bomb") in (0, 1),
                               f"bomb {payload.get('bomb')!r} is not a binary channel"))
    run("margin", lambda: expect(
        payload.get("margin") is None
        or (isinstance(payload.get("margin"), (int, float))
            and not isinstance(payload.get("margin"), bool)
            and math.isfinite(float(payload.get("margin")))),
        f"margin {payload.get('margin')!r} is neither a finite number nor null"))
    run("label_source", lambda: expect(
        payload.get("label_source") == LABEL_SOURCE,
        f"label_source {payload.get('label_source')!r} is not {LABEL_SOURCE!r}"))

    def label_verified() -> None:
        flags = payload.get("verified_mask") or ()
        expected = bool(sum(1 for value in flags if value))
        expect(payload.get("label_verified") is expected,
               f"label_verified {payload.get('label_verified')!r} is not "
               f"verified_count > 0 ({expected!r})")
        from touhou_rl.v420 import contracts as C  # noqa: PLC0415

        try:
            C.check_label_provenance(payload.get("label_source"),
                                     payload.get("label_verified"))
        except C.ContractError as exc:
            raise RelabelError(str(exc)) from exc

    run("label_verified", label_verified)
    run("teacher_bomb", lambda: expect(
        payload.get("teacher_bomb") == payload.get("bomb"),
        f"teacher_bomb {payload.get('teacher_bomb')!r} is not the label's bomb "
        f"{payload.get('bomb')!r}"))
    run("teacher_profile", lambda: expect(
        (payload.get("teacher") or {}).get("replan_decisions") == 1
        and (payload.get("teacher") or {}).get("decision_frames") == DECISION_FRAMES,
        "the row does not carry a two-frame teacher profile"))

    def executed_preserved() -> None:
        executed = payload.get("executed")
        expect(isinstance(executed, Mapping),
               f"executed is {executed!r}; the student's own action must stay on the "
               "row")
        from touhou_rl.v420 import contracts as C  # noqa: PLC0415

        expect(C.cand_to_mask(executed.get("cand"), executed.get("bomb", 0))
               == executed.get("mask"),
               f"executed mask {executed.get('mask')!r} is not "
               f"cand_to_mask({executed.get('cand')!r}, "
               f"{executed.get('bomb', 0)!r})")
        masks = payload.get("masks")
        expect(isinstance(masks, list) and masks
               and int(masks[0]) == int(executed.get("mask")),
               f"masks {masks!r} does not start with the executed mask "
               f"{executed.get('mask')!r}")
        expect(int(payload.get("executed_cand", executed.get("cand")))
               == int(executed.get("cand")),
               "executed_cand and executed.cand disagree")

    run("executed_preserved", executed_preserved)

    def nonfinite_consistent() -> None:
        expect(isinstance(relabel, Mapping), f"relabel is {relabel!r}, not a block")
        values = payload.get("scores") or []
        nulls = [index for index, value in enumerate(values) if value is None]
        expect(nulls == sorted(int(index) for index in nonfinite),
               f"relabel.nonfinite_scores {nonfinite} is not the null entries {nulls}")

    run("nonfinite_consistent", nonfinite_consistent)

    def relabel_block() -> None:
        expect(isinstance(relabel, Mapping), "the row has no relabel block")
        for key in ("engine_sha256", "tape_sha256", "entry_hash", "tolerance", "at",
                    "query_index", "decision_frame", "hash_pre", "hash_post"):
            expect(relabel.get(key) not in (None, ""),
                   f"relabel.{key} is missing")
        expect(int(relabel["decision_frame"]) == int(payload.get("decision_frame")),
               f"relabel.decision_frame {relabel['decision_frame']} is not the row's "
               f"{payload.get('decision_frame')}")
        expect(str(relabel["hash_pre"]).strip().lower()
               == str(payload.get("hash_pre")).strip().lower(),
               "relabel.hash_pre is not the row's hash_pre")
        expect(float(relabel["tolerance"]) > 0,
               f"relabel.tolerance {relabel['tolerance']!r} is not positive")

    run("relabel_block", relabel_block)
    return errors


# ---------------------------------------------------------------------------
# The label sink
# ---------------------------------------------------------------------------
class LabelSink:
    """Append-only JSONL sink for ``labels.jsonl``; every row is flushed.

    Opened with ``"x"``: a ``labels.jsonl`` that already exists is refused rather
    than appended to, so two runs can never interleave into one file.  A failed
    write raises :class:`RelabelError` after the rows already written were
    flushed, which is what "keep what was written and say why" needs.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows = 0
        try:
            self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise RelabelError(
                f"cannot create the label file {self.path.as_posix()}: "
                f"{type(exc).__name__}: {exc}", stop="write_failed") from exc

    def write(self, row: Mapping[str, Any]) -> int:
        line = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)
        try:
            self._handle.write(line + "\n")
            self._handle.flush()
        except (OSError, ValueError) as exc:
            raise RelabelError(
                f"cannot append row {self.rows} to {self.path.as_posix()}: "
                f"{type(exc).__name__}: {exc}", stop="write_failed") from exc
        self.rows += 1
        return self.rows

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is None or handle.closed:
            return
        try:
            handle.close()
        except OSError as exc:
            raise RelabelError(f"cannot close {self.path.as_posix()}: {exc}",
                               stop="write_failed") from exc


def default_writer_factory(path: Path) -> LabelSink:
    """The shipped sink (the seam tests replace)."""

    return LabelSink(path)


def check_out_dir(path: str | Path, *, source: Path | None = None) -> Path:
    """Refuse anything but a **new or empty** directory, distinct from the source."""

    target = Path(path)
    if target.exists() and not target.is_dir():
        raise RelabelError(f"--out {target.as_posix()} exists and is not a directory",
                           stop="out_dir_invalid")
    if source is not None:
        try:
            same = target.resolve() == Path(source).resolve()
        except OSError:  # pragma: no cover - an unresolvable path is refused below
            same = False
        if same:
            raise RelabelError(
                f"--out {target.as_posix()} is the --save-obs directory; the labels "
                "are a new corpus and must not land on top of the source",
                stop="out_dir_invalid")
    if target.is_dir():
        entries = sorted(target.iterdir(), key=lambda entry: entry.name)
        if entries:
            names = ", ".join(entry.name for entry in entries[:5])
            more = ", ..." if len(entries) > 5 else ""
            raise RelabelError(
                f"--out {target.as_posix()} is not empty ({len(entries)} entries: "
                f"{names}{more}); a relabel run is never merged into an existing "
                "directory -- name a new one", stop="out_dir_not_empty")
    return target


# ---------------------------------------------------------------------------
# The result skeleton
# ---------------------------------------------------------------------------
def base_result(config: RelabelConfig) -> dict[str, Any]:
    """Every key ``meta.json`` carries, whatever happened (refusals included)."""

    return {
        "schema": SCHEMA_META,
        "created_at": now_iso(),
        "runner": {"path": TOOL_PATH.as_posix(), "sha256": tool_sha()},
        "argv": list(config.argv),
        "source": {"root": Path(config.save_obs).as_posix(), "rows": None,
                   "slots_sha256": None, "meta_sha256": None},
        "tape": {"path": Path(config.tape_slots).as_posix(), "sha256": None,
                 "slots": None},
        "entry": {"frame": int(config.entry_frame), "hash": None,
                  "tape_anchor_hash": None, "student_hash_pre": None,
                  "verified": False},
        "seed": int(config.seed),
        "stage": int(config.stage),
        "difficulty": int(config.difficulty),
        "team": int(config.team),
        "engine": {"path": Path(config.engine).as_posix(), "sha256": None,
                   "expected_sha256": str(config.expected_sha), "matches": None},
        "data_dir": Path(config.data_dir).as_posix(),
        "mute": {"before": None, "after": None},
        "teacher": {"profile": str(config.profile), "tolerance": None,
                    "queries": 0},
        "player": player_meta_block(config),
        "reset": None,
        "probes": None,
        "counts": {"source_rows": None, "rows_requested": None, "rows_labelled": 0,
                   "limit": config.limit, "truncated": False,
                   "frames_prefix": 0, "frames_student": 0,
                   "decisions_replayed": 0, "teacher_queries": 0,
                   "verified_slots": 0, "verified_candidates": 0,
                   "nonfinite_slots": 0, "failures": 0},
        "prefix": None,
        "labels": {"file": LABELS_FILE, "path": None, "rows": 0, "bytes": None,
                   "sha256": None},
        "distribution": {"chosen": {}, "chosen_null": 0, "chosen_verified": 0,
                         "chosen_unverified_fallback": 0, "bomb": {}, "label_verified": 0},
        "agreement": {"comparable": 0, "executed_equals_chosen": 0, "ratio": None,
                      "verified_only": {"comparable": 0, "equal": 0, "ratio": None}},
        "stop": None,
        "failure": None,
        "complete": False,
        "exit_code": EXIT_TECHNICAL,
        "wall_s": None,
        "field_notes": {
            "bomb": ("the teacher's bomb channel (reslot's convention); the student's "
                     "executed bomb is preserved in executed.bomb"),
            "slot_id": ("null: a student decision at frame d is not the tape's slot at "
                        "frame d; join on (episode_seed, decision_frame, hash_pre)"),
            "scores_null": ("a null score means the teacher produced a non-finite "
                            "score for that candidate; the index is listed in "
                            "relabel.nonfinite_scores and never replaced by a floor"),
            "not_a_replay": ("DAgger labels: the states are the student's own, the "
                             "frames were replayed with the student's masks, and the "
                             "teacher was re-queried on them; no teacher action was "
                             "committed"),
        },
        "classification": "measured",
    }


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
def _require_mute(record: Mapping[str, Any], where: str) -> None:
    if record.get("available") and record.get("muted"):
        return
    raise RelabelError(
        f"{where}: th08.cfg is not a muted 60-byte configuration "
        f"(available={record.get('available')!r}, muted={record.get('muted')!r}, "
        f"reason={record.get('reason')!r}); AGENTS.md section 9 forbids starting an "
        "audible engine", stop="mute_refused", detail={"mute": dict(record)})


def _label_context(*, row: StudentRow, index: int, gate: Mapping[str, Any],
                   tape: Mapping[str, Any], entry: Mapping[str, Any],
                   tolerance: float, release: bool, executed: Mapping[str, Any],
                   extra: Mapping[str, Any]) -> dict[str, Any]:
    """The ``relabel`` block: what produced this label and where it was asked."""

    block = {
        "schema": SCHEMA_RELABEL_BLOCK,
        "at": now_iso(),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": tool_sha()},
        "query_index": int(index),
        "decision_index": int(row.decision_index),
        "decision_frame": int(row.decision_frame),
        "hash_pre": row.hash_pre,
        "hash_post": row.hash_post,
        "engine_sha256": str(gate.get("sha256") or ""),
        "engine_expected_sha256": str(gate.get("expected_sha256") or ""),
        "tape_sha256": str(tape.get("sha256") or ""),
        "tape_path": str(tape.get("path") or ""),
        "entry_frame": int(entry["frame"]),
        "entry_hash": str(entry["hash"]),
        "tolerance": float(tolerance),
        "record_source": "child",
        "committed": [],
        "extra_actions": [],
        "shoot_release": bool(release),
        "executed": dict(executed),
        "nonfinite_scores": list(extra.get("nonfinite_scores") or []),
        "chosen_rule": CHOSEN_RULE,
        "teacher": dict(extra.get("teacher") or {}),
    }
    return block


def _label_row_for(*, row: StudentRow, decision: Any, index: int,
                   gate: Mapping[str, Any], tape: Mapping[str, Any],
                   entry: Mapping[str, Any], tolerance: float, release: bool,
                   teacher_profile: Mapping[str, Any],
                   player: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Derive the label and assemble the row this run will write."""

    label = derive_label(decision)
    extra = label.pop("relabel_extra")
    context = _label_context(
        row=row, index=index, gate=gate, tape=tape, entry=entry,
        tolerance=tolerance, release=release,
        executed=dict(row.payload["executed"]), extra=extra)
    payload = build_label_row(row=row, label=label, teacher_profile=teacher_profile,
                             context=context, player=player)
    errors = check_label_fields(payload)
    if player is not None:
        # The coordinate checks live in their own roster (PLAYER_CHECK_NAMES) so the
        # fifteen label-check names keep matching reslot's; the refusal is the same.
        errors.update(check_player_fields(payload))
    if errors:
        raise RelabelError(
            f"decision frame {row.decision_frame}: the label record violates the "
            f"contract before writing: {errors}", stop="label_invalid",
            detail={"check_errors": errors, "decision_frame": row.decision_frame})
    return payload


def _accumulate(result: dict[str, Any], payload: Mapping[str, Any]) -> None:
    """Update the counters and the distributions from one written label."""

    counts = result["counts"]
    dist = result["distribution"]
    agreement = result["agreement"]
    counts["verified_slots"] += 1 if payload.get("verified_count") else 0
    counts["verified_candidates"] += int(payload.get("verified_count") or 0)
    counts["nonfinite_slots"] += 1 if payload["relabel"]["nonfinite_scores"] else 0
    chosen = payload.get("chosen")
    if chosen is None:
        dist["chosen_null"] += 1
    else:
        key = str(int(chosen))
        dist["chosen"][key] = int(dist["chosen"].get(key, 0)) + 1
    dist["bomb"][str(int(payload["bomb"]))] = int(
        dist["bomb"].get(str(int(payload["bomb"])), 0)) + 1
    if payload.get("label_verified"):
        dist["label_verified"] += 1
    if payload.get("chosen_verified"):
        dist["chosen_verified"] += 1
    elif chosen is not None:
        dist["chosen_unverified_fallback"] += 1
    executed = int(payload["executed"]["cand"])
    if chosen is not None:
        agreement["comparable"] += 1
        agreement["executed_equals_chosen"] += 1 if executed == int(chosen) else 0
    if payload.get("chosen_verified"):
        agreement["verified_only"]["comparable"] += 1
        agreement["verified_only"]["equal"] += 1 if executed == int(chosen) else 0


def _finalise_ratios(result: dict[str, Any]) -> None:
    agreement = result["agreement"]
    comparable = int(agreement["comparable"])
    agreement["ratio"] = (round(agreement["executed_equals_chosen"] / comparable, 6)
                          if comparable else None)
    verified = agreement["verified_only"]
    verified["ratio"] = (round(verified["equal"] / verified["comparable"], 6)
                         if verified["comparable"] else None)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8", newline="\n")


def relabel_readme(meta: Mapping[str, Any]) -> str:
    """``DIR2/README.md``: what the labels are, and every field a reader must know."""

    nulls = "\n".join(f"| `{name}` | **null** | {why} |"
                      for name, why in SOURCE_NULL_FIELDS)
    return "\n".join([
        f"# DAgger relabel ({SCHEMA_META})",
        "",
        "The **teacher's labels on the states a student policy itself visited**.",
        "Written by `tools/rl/v8_32_relabel.py run` from a",
        "`v8_32_takeover.py --arm checkpoint --save-obs DIR` rollout.",
        "",
        "**This is not a replay and not a re-collection.**  The frames were replayed",
        "with the student's own masks to prove the states, and at every decision",
        "boundary the teacher was asked a *fresh* question",
        '(`decide(record_source="child")`).  The teacher never chose an action that',
        "was committed; every `executed` field is the student's own.",
        "",
        "## Identity",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| source | `{meta.get('source', {}).get('root')}` |",
        f"| source slots sha256 | `{meta.get('source', {}).get('slots_sha256')}` |",
        f"| seed | `{meta.get('seed')}` |",
        f"| entry frame / hash | `{meta.get('entry', {}).get('frame')}` / "
        f"`{meta.get('entry', {}).get('hash')}` |",
        f"| tape sha256 | `{meta.get('tape', {}).get('sha256')}` |",
        f"| engine sha256 | `{meta.get('engine', {}).get('sha256')}` |",
        f"| teacher profile | `{meta.get('teacher', {}).get('profile')}` "
        f"(tolerance `{meta.get('teacher', {}).get('tolerance')}`) |",
        f"| rows labelled | `{meta.get('counts', {}).get('rows_labelled')}` of "
        f"`{meta.get('counts', {}).get('source_rows')}` |",
        f"| stop | `{meta.get('stop')}` |",
        f"| complete | `{meta.get('complete')}` |",
        f"| created at | `{meta.get('created_at')}` |",
        "",
        "## Layout",
        "",
        "| path | content |",
        "| --- | --- |",
        f"| `{LABELS_FILE}` | one JSON row per labelled decision, flushed as written |",
        f"| `{META_FILE}` | this identity, the counters, the failure record |",
        f"| `{README_FILE}` | this file |",
        "",
        "## The label fields",
        "",
        "| field | meaning |",
        "| --- | --- |",
        "| `scores` | the teacher's 18-way score vector (`cand = focus * 9 + move`) |",
        "| `verified_mask` | 18 flags: which candidates were engine-verified |",
        "| `verified_count` | how many candidates the mask marks |",
        "| `chosen` | the label (see `relabel.chosen_rule`) |",
        "| `chosen_verified` | whether `chosen` is inside `verified_mask` |",
        "| `chosen_score` | `scores[chosen]` |",
        "| `margin` | the teacher's margin (null when it is not finite) |",
        "| `bomb` | the teacher's bomb channel -- **not** the student's |",
        "| `teacher_bomb` | the same byte as `bomb` |",
        "| `label_source` | `teacher_2f` |",
        "| `label_verified` | `verified_count > 0` |",
        "| `teacher` | the teacher profile identity (never a policy input) |",
        "| `relabel` | provenance: engine/tape sha256, entry hash, tolerance, time |",
        "| `player` | the student's `px`/`py` on this decision frame, read from "
        "`PLAYER` **before** the action (this format's addition) |",
        "",
        "## The player coordinate",
        "",
        "`player.px`/`player.py` are the `px`/`py` fields of the engine's `PLAYER`",
        "reply, decoded by `touhou_rl.v400.gym.protocol.parse_player_reply`.  The read",
        "happens once per decision boundary, **after** the engine's hash was checked",
        "against the row's `hash_pre` and **before** `decide()` and before any",
        "`STEP` of that decision (`player.read_phase = \"before_action\"`,",
        "`player.precedes = [\"decide\", \"student_step\"]`).  `player.raw` keeps the",
        "wire tokens (`px=%08x` is a float32 bit pattern).",
        "",
        f"`--require-player` was {'**set**' if (meta.get('player') or {}).get('require_player') else '**not** set'} for this",
        "run: when it is set, a boundary whose coordinate cannot be read refuses the",
        "run (`player_unavailable`) instead of writing a `null`, and a `PLAYER` reply",
        "taken on another frame is always refused (`player_frame_mismatch`).  When it",
        "is not set, `--require-player`-less rows may carry `player.px = null` with",
        "`player.available = false` and `player.reason` explaining why.",
        "",
        "**The student's own action is under `executed`** (`cand`/`bomb`/`mask`) and in",
        "`masks`.  After relabelling the top-level `(cand, bomb, mask)` triple is",
        "deliberately **not** a round trip, because the top-level `bomb` is the",
        "teacher's channel -- exactly as in a T04 slot record.",
        "",
        "## Fields the source wrote as null",
        "",
        "| field | value | why |",
        "| --- | --- | --- |",
        nulls,
        "",
    ])


def run_relabel(config: RelabelConfig, *, deps: RelabelDeps | None = None,
                log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Replay the student's rollout, ask the teacher at every boundary, write labels.

    Returns the ``meta.json`` payload (which is written to ``<out>/meta.json``
    before returning) with ``exit_code`` set.  Every failure is recorded rather
    than raised: the caller decides the process exit code from ``exit_code``.
    """

    resolved = (deps or RelabelDeps()).resolve()
    clock = resolved.clock
    say = log if log is not None else (lambda message: print(message))
    started = float(clock())
    result = base_result(config)
    #: every written row's player block, for the meta.json summary (a refusal
    #: before the loop simply leaves it empty)
    player_blocks: list[dict[str, Any]] = []
    engine: Any = None
    sink: Any = None
    statement = "technical_error"

    try:
        config.validate()
        # -- 1. everything that needs no engine ---------------------------
        try:
            gate = dict(resolved.engine_gate(config))
        except RelabelError as exc:
            # a refused engine gate still reports the binary it looked at
            if isinstance(exc.detail.get("engine"), Mapping):
                result["engine"].update(dict(exc.detail["engine"]))
            raise
        result["engine"].update(dict(gate))
        if not result["engine"].get("matches"):
            raise RelabelError(
                f"the engine sha256 {gate.get('sha256')!r} is not --expected-sha "
                f"{config.expected_sha!r}", stop="engine_sha_mismatch")
        data_dir = Path(config.data_dir)
        if not data_dir.is_dir():
            raise RelabelError(f"--data-dir does not exist: {data_dir.as_posix()}",
                               stop="data_dir_missing")
        mute_before = dict(resolved.mute_reader(data_dir))
        result["mute"]["before"] = mute_before
        _require_mute(mute_before, f"--data-dir {data_dir.as_posix()}")
        source = load_source(config.save_obs, seed=int(config.seed),
                             entry_frame=int(config.entry_frame),
                             period=int(config.shoot_pulse_period))
        result["source"].update({
            "root": source.root.as_posix(), "rows": int(source.count),
            "slots_sha256": source.slots_sha256,
            "meta_sha256": source.meta_sha256, **source.identity()})
        result["counts"]["source_rows"] = int(source.count)
        try:
            tape = tk.load_tape(config.tape_slots)
        except tk.PreflightRefused as exc:
            raise RelabelError(str(exc), stop="tape_invalid") from exc
        plan = build_plan(tape, source)
        result["tape"] = {"path": tape["path"], "sha256": tape["sha256"],
                          "slots": int(tape["count"]),
                          "first_frame": int(tape["first_frame"]),
                          "last_frame": int(tape["last_frame"])}
        result["entry"]["tape_anchor_hash"] = str(plan["anchor_hash"])
        result["entry"]["student_hash_pre"] = source.rows[0].hash_pre
        out_dir = check_out_dir(config.out, source=source.root)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RelabelError(
                f"cannot create --out {out_dir.as_posix()}: {type(exc).__name__}: {exc}",
                stop="out_dir_invalid") from exc
        teacher_cfg = None
        try:
            teacher_cfg = resolved.teacher_config(config)
            from touhou_rl.v420.reslot import profile_snapshot  # noqa: PLC0415

            teacher_profile = profile_snapshot(teacher_cfg)
        except RelabelError:
            raise
        except Exception as exc:  # noqa: BLE001 - a bad profile is a refusal
            raise RelabelError(
                f"cannot build the teacher profile {config.profile!r}: "
                f"{type(exc).__name__}: {exc}", stop="config_invalid") from exc
        tolerance = float(resolved.tolerance())
        result["teacher"] = {
            "profile": str(config.profile), "tolerance": tolerance,
            "replan_decisions": int(teacher_profile["replan_decisions"]),
            "decision_frames": int(teacher_profile["decision_frames"]),
            "horizon": int(teacher_profile["horizon"]),
            "shoot_pulse_period": int(teacher_profile["shoot_pulse_period"]),
            "snapshot": teacher_profile, "queries": 0}
        if int(teacher_profile["replan_decisions"]) != 1:
            raise RelabelError(
                f"the teacher config replans {teacher_profile['replan_decisions']} "
                "decisions per query; a two-frame label needs 1",
                stop="config_invalid")
        if float(tolerance) != PINNED_TOLERANCE:
            say(f"relabel: WARNING verify.verify_plans tolerance is {tolerance}, the "
                f"pinned value is {PINNED_TOLERANCE}; the record carries the value in "
                "force")
        rows = source.rows
        requested = len(rows)
        truncated = False
        if config.limit is not None and int(config.limit) < requested:
            requested = int(config.limit)
            truncated = True
        result["counts"]["rows_requested"] = int(requested)
        result["counts"]["truncated"] = bool(truncated)
        say(f"relabel: source {source.root.as_posix()} rows={source.count} "
            f"entry={source.entry_frame} seed={source.seed} "
            f"tape_slots={tape['count']} labelling {requested}")

        # -- 2. the engine, RESET, the probe ------------------------------
        engine = resolved.client_factory(config)
        reset = reset_engine(engine, config)
        result["reset"] = {"reply": reset["reply"], "hash": reset["hash"]}
        result["probes"] = probe_engine(engine)
        say(f"relabel: reset hash={reset['hash']} probes="
            f"{result['probes']['child_hazards'].get('available')}")

        # -- 3. the tape prefix, every slot's hash verified ---------------
        prefix_client = PrefixClient(engine)
        try:
            prefix = tk.replay_prefix(prefix_client, plan,
                                      period=int(config.shoot_pulse_period))
        except tk.EntryMismatch as exc:
            raise RelabelError(
                f"the prefix replay diverged: {exc}", stop="entry_hash_mismatch",
                detail=exc.record()) from exc
        except tk.StepFailed as exc:
            raise RelabelError(f"the prefix replay stopped: {exc}",
                               stop="step_failed") from exc
        except tk.TakeoverError as exc:
            raise RelabelError(f"the prefix replay failed: {exc}",
                               stop="protocol_error") from exc
        result["prefix"] = dict(prefix)
        result["counts"]["frames_prefix"] = int(prefix["frames"])

        # -- 4. the entry gate -------------------------------------------
        frame_now = int(engine.frame())
        if frame_now != int(source.entry_frame):
            raise RelabelError(
                f"the prefix landed on frame {frame_now}, the entry is "
                f"{source.entry_frame}", stop="frame_mismatch",
                detail={"frame": frame_now, "entry_frame": int(source.entry_frame)})
        entry_hash = str(engine.hash()).strip().lower()
        result["entry"]["hash"] = entry_hash
        if entry_hash != str(plan["anchor_hash"]) or entry_hash != rows[0].hash_pre:
            raise RelabelError(
                f"the entry hash {entry_hash} is not the tape's "
                f"{plan['anchor_hash']} / row 0's {rows[0].hash_pre}; the teacher "
                "would be asked about a state the student never stood on",
                stop="entry_hash_mismatch",
                detail={"entry_hash": entry_hash,
                        "tape_anchor_hash": str(plan["anchor_hash"]),
                        "student_hash_pre": rows[0].hash_pre})
        result["entry"]["verified"] = True
        say(f"relabel: entry verified frame={frame_now} hash={entry_hash}")

        # -- 5. the label sink (only now: a failed entry writes nothing) --
        sink = resolved.writer_factory(out_dir / LABELS_FILE)
        result["labels"]["path"] = (out_dir / LABELS_FILE).as_posix()

        # -- 6. one fresh teacher query per student decision --------------
        # No ``consistency`` object is shared across the queries: every state is an
        # independent question, so each one gets its own bookkeeping (the same
        # choice ``touhou_rl/v420/reslot.py::_reslot_one`` makes, i.e. the measured
        # T04 path, rather than ``counterfactual``'s per-session accumulator).
        from touhou_rl.v400.teacher.shoot import release_due  # noqa: PLC0415

        current_frame = int(source.entry_frame)
        for row in rows[:requested]:
            if int(engine.frame()) != int(row.decision_frame):
                raise RelabelError(
                    f"decision frame {row.decision_frame}: the engine is on frame "
                    f"{int(engine.frame())}; the replay and the corpus disagree",
                    stop="frame_mismatch",
                    detail={"decision_frame": row.decision_frame,
                            "frame": int(engine.frame())})
            hash_now = str(engine.hash()).strip().lower()
            if hash_now != row.hash_pre:
                raise RelabelError(
                    f"decision frame {row.decision_frame}: the engine's hash "
                    f"{hash_now} is not the row's hash_pre {row.hash_pre}; the replay "
                    "left the student's trajectory, so the label would describe "
                    "another state", stop="student_hash_mismatch",
                    detail={"decision_frame": row.decision_frame,
                            "expected": row.hash_pre, "actual": hash_now})
            # The player coordinate, read here on purpose: the state is already
            # proven (frame + hash above) and nothing of this decision has been
            # stepped or committed yet -- so px/py is the position *before* the
            # action the row is labelled on.  PLAYER is read-only.
            player = read_player_at_boundary(
                resolved.player_reader, engine, row=row, index=int(row.index),
                hash_pre=hash_now)
            player_gate(player, require=bool(config.require_player),
                        where=f"decision frame {row.decision_frame}")
            release = bool(release_due(int(row.decision_index),
                                       int(config.shoot_pulse_period)))
            try:
                decision = resolved.decide_call(
                    engine, teacher_cfg, committed=(), extra_actions=(),
                    frame=int(row.decision_frame), stage=int(config.stage),
                    difficulty=int(config.difficulty),
                    record_source="child", shoot_release=release)
            except RelabelError:
                raise
            except Exception as exc:  # noqa: BLE001 - a refused query is a stop
                raise RelabelError(
                    f"decision frame {row.decision_frame}: the teacher query failed: "
                    f"{type(exc).__name__}: {exc}", stop="teacher_query_failed",
                    detail={"decision_frame": row.decision_frame,
                            "frames": current_frame}) from exc
            result["teacher"]["queries"] += 1
            result["counts"]["teacher_queries"] += 1
            moved = int(engine.frame())
            if moved != int(row.decision_frame):
                raise RelabelError(
                    f"decision frame {row.decision_frame}: record_source=\"child\" moved "
                    f"the shadow to frame {moved}; a query must not advance the "
                    "authoritative process", stop="protocol_error",
                    detail={"decision_frame": row.decision_frame, "frame": moved})
            payload = _label_row_for(
                row=row, decision=decision, index=int(row.index), gate=gate,
                tape=tape, entry={"frame": int(source.entry_frame),
                                  "hash": entry_hash},
                tolerance=tolerance, release=release,
                teacher_profile=teacher_profile, player=player)
            # The student's own masks are what advances the replay: the teacher
            # never drives this process.
            for offset, mask in enumerate(row.masks):
                reply = None
                try:
                    reply = engine.step(int(mask), 1)
                except Exception as exc:  # noqa: BLE001 - a refused STEP is a stop
                    raise RelabelError(
                        f"decision frame {row.decision_frame}: STEP {int(mask)} 1 "
                        f"failed: {type(exc).__name__}: {exc}", stop="step_failed",
                        detail={"decision_frame": row.decision_frame,
                                "mask": int(mask)}) from exc
                fields = dict(reply) if isinstance(reply, Mapping) else {}
                # The shadow build answers STEP without an ``advanced`` field on some
                # revisions; the two-frame hash check below is the authority, so a
                # missing field defaults to the documented one-frame step (the same
                # default ``reslot`` and ``counterfactual`` use).
                advanced = int(fields.get("advanced", 1) or 1)
                if advanced != 1 or fields.get("exit") == "1":
                    raise RelabelError(
                        f"decision frame {row.decision_frame}: STEP {int(mask)} 1 "
                        f"advanced={advanced}, exit={fields.get('exit')!r}",
                        stop="step_failed",
                        detail={"decision_frame": row.decision_frame,
                                "mask": int(mask), "advanced": advanced,
                                "offset": offset})
                current_frame += 1
                result["counts"]["frames_student"] += 1
            hash_after = str(engine.hash()).strip().lower()
            if hash_after != row.hash_post:
                raise RelabelError(
                    f"decision frame {row.decision_frame}: the hash after the "
                    f"student's masks is {hash_after}, the row records "
                    f"{row.hash_post}; the replay diverged inside the decision",
                    stop="student_hash_mismatch",
                    detail={"decision_frame": row.decision_frame,
                            "expected": row.hash_post, "actual": hash_after})
            sink.write(payload)
            player_blocks.append(dict(payload["player"]))
            _accumulate(result, payload)
            result["counts"]["rows_labelled"] = int(sink.rows)
            result["counts"]["decisions_replayed"] += 1
            if sink.rows % 250 == 0:
                say(f"relabel: {sink.rows}/{requested} labelled "
                    f"(frame {row.decision_frame})")
        statement = "limit_reached" if truncated else "complete"
    except RelabelError as exc:
        statement = exc.stop
        result["failure"] = {"stop": exc.stop, "detail": str(exc),
                            "at": now_iso(), "error_kind": type(exc).__name__,
                            **{key: value for key, value in exc.detail.items()
                               if key != "stop"}}
    except tk.TakeoverError as exc:
        statement = getattr(exc, "stop", "protocol_error")
        result["failure"] = {"stop": statement, "detail": str(exc), "at": now_iso(),
                            "error_kind": type(exc).__name__}
    except OSError as exc:
        statement = "write_failed"
        result["failure"] = {"stop": statement,
                            "detail": f"{type(exc).__name__}: {exc}",
                            "at": now_iso(), "error_kind": type(exc).__name__}
    except Exception as exc:  # noqa: BLE001 - an unexpected failure is still reported
        statement = "technical_error"
        result["failure"] = {"stop": statement,
                            "detail": f"{type(exc).__name__}: {exc}",
                            "at": now_iso(), "error_kind": type(exc).__name__,
                            "traceback": traceback.format_exc()}
    finally:
        result["engine"]["close"] = close_engine(engine)
        engine = None
        if sink is not None:
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001 - a refused close is a stop
                if result["failure"] is None:
                    statement = getattr(exc, "stop", "write_failed")
                    result["failure"] = {"stop": statement, "detail": str(exc),
                                        "at": now_iso(),
                                        "error_kind": type(exc).__name__}
        try:
            result["mute"]["after"] = dict(resolved.mute_reader(Path(config.data_dir)))
        except Exception as exc:  # noqa: BLE001 - a read-back failure is recorded
            result["mute"]["after"] = {"available": False,
                                       "reason": f"{type(exc).__name__}: {exc}"}

    # -- 7. finalise: the counters, the sink identity, meta.json ----------
    result["stop"] = statement
    result["counts"]["rows_labelled"] = int(getattr(sink, "rows", 0))
    result["labels"]["rows"] = int(getattr(sink, "rows", 0))
    # The player block keeps the *how* that base_result wrote (command/fields/
    # timing) and gains the *what* of this run: the spread and the misses.
    result["player"].update(player_block_summary(player_blocks))
    label_path = getattr(sink, "path", None)
    if label_path is not None and Path(label_path).is_file():
        result["labels"]["bytes"] = int(Path(label_path).stat().st_size)
        result["labels"]["sha256"] = sha256_file(label_path)
    _finalise_ratios(result)
    result["counts"]["teacher_queries"] = int(result["counts"]["teacher_queries"])
    # The run stops at the first failure, so the count is 0 or 1 -- but it is a
    # counter a reader can sum over runs, not a boolean to interpret.
    result["counts"]["failures"] = 0 if result["failure"] is None else 1
    result["complete"] = bool(statement in ("complete", "limit_reached")
                             and result["failure"] is None
                             and result["counts"]["rows_labelled"]
                             == (result["counts"]["rows_requested"] or 0))
    result["exit_code"] = int(STOP_EXIT.get(statement, EXIT_TECHNICAL))
    result["wall_s"] = round(float(clock()) - started, 3)

    out = Path(config.out)
    meta_path = out / META_FILE
    try:
        out.mkdir(parents=True, exist_ok=True)
        _write_json(meta_path, result)
        (out / README_FILE).write_text(relabel_readme(result), encoding="utf-8",
                                       newline="\n")
    except OSError as exc:
        say(f"relabel: cannot write {meta_path.as_posix()}: "
            f"{type(exc).__name__}: {exc}")
        result["exit_code"] = EXIT_TECHNICAL
        result["failure"] = result["failure"] or {
            "stop": "write_failed", "at": now_iso(),
            "detail": f"cannot write {meta_path.as_posix()}: {exc}"}
    result["meta_path"] = meta_path.as_posix()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v8_32_relabel.py",
        description=(
            "DAgger relabel: replay a student's --save-obs rollout to every decision "
            "boundary, ask the teacher there (record_source=\"child\") and write the "
            "labels next to the student's own rows.  This is a teacher query on the "
            "student's states, not a replay of the teacher.  Starts no engine unless "
            "the `run` subcommand is used."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Exit codes: 0 every requested row was labelled, 1 the replay "
                "disagreed with the corpus (entry/state hash), 2 technical failure "
                "(teacher, engine protocol, write), 4 refused before the start, "
                "5 the four mute bytes were not 0."))
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run", help="replay the student's rollout and label every decision",
        description=("RESET, replay the tape to --entry-frame verifying every slot, "
                     "prove the entry hash equals row 0's hash_pre, then replay the "
                     "student's decisions and query the teacher at each boundary."))
    run.add_argument("--save-obs", required=True, dest="save_obs",
                     help="the student rollout directory (slots.jsonl + obs/ + "
                          "meta.json) written by v8_32_takeover.py --save-obs")
    run.add_argument("--tape-slots", required=True, dest="tape_slots",
                     help="the slots.jsonl of the T04 tape the rollout entered from")
    run.add_argument("--seed", required=True, type=int,
                     help="the RESET seed; must be the corpus's episode_seed")
    run.add_argument("--entry-frame", required=True, dest="entry_frame", type=int,
                     help="the frame the student rollout entered at (even, >= 16, a "
                          "slot boundary of the tape)")
    run.add_argument("--engine", required=True,
                     help="the rl_v3 th08-modern binary, or a build directory "
                          "containing it")
    run.add_argument("--expected-sha", required=True, dest="expected_sha",
                     help="the sha256 the engine binary must have")
    run.add_argument("--data-dir", required=True, dest="data_dir",
                     help="the game data directory the engine is started in "
                          "(th08.cfg must be muted)")
    run.add_argument("--out", required=True,
                     help="the output directory for labels.jsonl/meta.json (must be "
                          "new or empty)")
    run.add_argument("--stage", type=int, default=RESET_STAGE,
                     help=f"RESET stage (default {RESET_STAGE})")
    run.add_argument("--difficulty", type=int, default=RESET_DIFFICULTY,
                     help=f"RESET difficulty (default {RESET_DIFFICULTY})")
    run.add_argument("--team", type=int, default=RESET_TEAM,
                     help=f"RESET team (default {RESET_TEAM})")
    run.add_argument("--limit", type=int, default=None,
                     help="label only the first N decisions (smoke run)")
    run.add_argument("--profile", default=DEFAULT_PROFILE,
                     help=f"the teacher profile (default {DEFAULT_PROFILE})")
    run.add_argument("--shoot-pulse-period", dest="shoot_pulse_period", type=int,
                     default=DEFAULT_PULSE_PERIOD,
                     help="the SHOOT release phase the tape/corpus was recorded with "
                          f"(default {DEFAULT_PULSE_PERIOD})")
    run.add_argument("--timeout-s", dest="timeout_s", type=float,
                     default=DEFAULT_TIMEOUT_S,
                     help=f"per-command protocol timeout (default {DEFAULT_TIMEOUT_S})")
    run.add_argument("--require-player", dest="require_player", action="store_true",
                     help="refuse a run whose rows would carry no PLAYER px/py "
                          "(player_unavailable); off by default so the v8.32 label "
                          "contract is unchanged")
    run.add_argument("--no-xvfb", action="store_true", dest="no_xvfb",
                     help="do not re-exec under xvfb-run (POSIX only)")
    return parser


def config_from_args(args: argparse.Namespace) -> RelabelConfig:
    return RelabelConfig(
        save_obs=Path(args.save_obs), tape_slots=Path(args.tape_slots),
        seed=int(args.seed), entry_frame=int(args.entry_frame),
        engine=Path(args.engine), expected_sha=str(args.expected_sha),
        data_dir=Path(args.data_dir), out=Path(args.out), stage=int(args.stage),
        difficulty=int(args.difficulty), team=int(args.team),
        limit=(None if args.limit is None else int(args.limit)),
        timeout_s=float(args.timeout_s),
        shoot_pulse_period=int(args.shoot_pulse_period), profile=str(args.profile),
        require_player=bool(getattr(args, "require_player", False)),
        argv=tuple(getattr(args, "argv", ())))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    tk.maybe_reexec_under_xvfb(enabled=not bool(args.no_xvfb))
    args.argv = list(sys.argv if argv is None else ["v8_32_relabel.py", *argv])
    config = config_from_args(args)
    result = run_relabel(config)
    stop = result.get("stop")
    if result.get("failure"):
        print(f"run: FAILED stop={stop} rc={result.get('exit_code')}")
        print(f"  {result['failure'].get('detail')}")
        print(f"  labels written: {result['labels'].get('rows')} "
              f"({result['labels'].get('path')})")
    else:
        print(f"run: ok rows={result['labels'].get('rows')} of "
              f"{result['counts'].get('source_rows')} stop={stop} "
              f"entry_hash={result['entry'].get('hash')} "
              f"agreement={result['agreement'].get('ratio')} "
              f"wall_s={result.get('wall_s')} rc={result.get('exit_code')}")
    print(f"wrote: {result.get('meta_path')}")
    return int(result.get("exit_code", EXIT_TECHNICAL))


if __name__ == "__main__":
    raise SystemExit(main())
