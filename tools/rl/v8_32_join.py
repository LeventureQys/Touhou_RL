#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v8.3.2 ``join`` — fuse the T04 baseline shards and the DAgger teacher labels
into **one trainable corpus**.  Offline: the engine is never started.

Why this tool exists
--------------------
Two labelled corpora now exist on this checkout and neither is trainable alone:

* the **T04 baseline** (``.tmp_v420_data/T04/dataset/<shard>/``) — the tape's own
  states, ``v420-dataset-2f/1/slot`` rows, ``bomb`` is the executed *and* the
  teacher channel because the tape executed the teacher's action, and the obs
  planes are WSL symlinks that a Windows process cannot open;
* the **DAgger relabel output** (``v8_32_relabel.py run --out DIR``) — the
  *student's* states with a fresh teacher label, ``v832-dagger-relabel-label/1``
  rows, ``shard_key``/``slot_id`` are ``null``, the top-level ``bomb`` is the
  teacher's bomb channel and the student's executed bomb lives in
  ``executed.bomb``, and the obs planes live in the ``--save-obs`` directory the
  relabel run entered from.

A training corpus needs one row space, one ``label_source`` vocabulary, one
``bomb`` channel and one contiguous ``slot_id``.  This tool builds that row
space from any number of baseline shards plus any number of relabel outputs, and
it *proves* the result loads through the downstream loader before it reports
success.

What ``build`` does
-------------------
1. Reads every ``--baseline`` shard (``slots.jsonl`` + ``meta.json`` + the three
   obs planes, resolved the way ``readout01._resolve_plane`` resolves them: the
   declared ``obs/<name>`` file, then ``meta.obs.files[].source`` mapped from
   ``/mnt/<drive>/...`` onto this host, then ``meta.obs.files[].target``) and
   every ``--dagger`` ``labels.jsonl`` (plus the sibling ``meta.json`` whose
   ``source.root`` is the ``--save-obs`` directory that owns the planes).
2. **Self-checks every row before anything is written** and fails closed:
   ``scores`` are 18 entries (``null`` only for the indices the relabel row
   declares in ``relabel.nonfinite_scores``), ``verified_mask`` is 18 flags,
   ``verified_count`` matches it, ``chosen`` is inside the verified set and is
   its argmax, ``chosen_verified``/``label_verified`` follow the chosen/verified
   arithmetic, ``mask`` round-trips to ``(executed_cand, executed_bomb)`` through
   the frozen v4.2.0 mask table, ``history`` is 8 candidates and equals the
   ``act_hist`` plane row, the ``hash_pre -> hash_post -> hash_pre`` chain is
   unbroken inside each source, and the source's own row counters agree with the
   files on disk.
3. Normalises the row channels: ``bomb`` is the **training label channel** (the
   baseline's own ``bomb``; the relabel row's ``teacher_bomb``) and the row says
   which one it used in ``bomb_source``; on a dagger row the *action* channel
   follows ``--dagger-label`` (default ``chosen``: the teacher's action, with the
   student's preserved in ``student_*``); the corpus each row came from is
   ``provenance``
   (``"t04"`` | ``"dagger"``); the original nested provenance (T04 rows) stays
   verbatim in ``provenance_source``.
4. Renumbers ``slot_id`` ``0..N-1`` and gives every source a new contiguous
   ``shard_key`` (``v832join-s00``, ``v832join-s01``, ...); the source's own
   ``slot_id``/``shard_key`` are preserved under ``source``.
5. Optionally drops ``label_verified=false`` rows (``--require-verified``) and
   optionally de-duplicates on the join key
   ``(episode_seed, decision_frame, hash_pre)``, keeping the side named by
   ``--prefix`` (default ``dagger``) and counting the conflicts (``--dedup``).
6. Writes ``<out>/slots.jsonl``, ``<out>/targets.jsonl``, ``<out>/obs/{play96x112,hud,act_hist}.npy``
   (**bit copies** of the selected source rows — no re-encoding, no rescaling),
   ``<out>/meta.json`` (row counts, source distribution, ``hash_pre`` uniqueness,
   join-key conflicts, per-plane sha256, per-source self-check) and
   ``<out>/downstream/`` (the loader probe evidence).
7. Runs the downstream load probe: it imports ``.tmp_v832_s2e/s2e_train.py`` and
   ``.tmp_v832_readout01/readout01.py`` **read-only** and pushes the corpus
   through ``readout01.Corpus`` + ``s2e_train.merge_shards`` +
   ``s2e_train.view_corpus``, asserting rows, plane shapes/dtypes, every
   ``LABEL_FIELDS`` array and the ``label_source`` distribution.  It then calls
   ``dataset_2f.check_targets_record`` on ``targets.jsonl``.  Any failure aborts
   the build (``stop="downstream_unreadable"``).

Output row (``v832-join/1/slot``) — the fields a trainer consumes
-----------------------------------------------------------------
``schema``/``dataset_version``/``obs_contract`` | ``provenance`` (``t04``|``dagger``)
| ``shard_key``/``slot_id`` (new, contiguous) | ``episode_seed``/``decision_frame``
| ``duration_frames``/``commit_frames``/``obs_render_frames``/``history_frames``
(derived from ``decision_frame``) | ``history`` (8) | ``executed_cand``/
``executed_bomb``/``mask`` (the action channel the trainer reads — see
``--dagger-label``) | ``student_cand``/``student_bomb``/``student_mask`` (dagger
rows built with ``--dagger-label chosen`` only: the student's original action) |
``bomb`` (the
label channel) / ``bomb_source`` (``t04_bomb``|``teacher_bomb``) / ``teacher_bomb``
| ``scores`` (18) / ``verified_mask`` (18) / ``verified_count`` | ``chosen``/
``chosen_verified``/``chosen_score``/``margin`` | ``label_source``/``label_verified``
| ``hash_pre``/``hash_post`` (16 hex) | ``obs`` (plane names + this row's index +
the source obs row + ``hud_frame``) | ``source`` (which source file/row/slot/shard
this row came from) | ``provenance_source`` (the source row's own ``provenance``)
| ``relabel`` (dagger rows: the reduced relabel block).

``--dagger-label`` — which action channel a dagger row trains on
---------------------------------------------------------------
``executed_cand`` / ``executed_bomb`` / ``mask`` are the trio a trainer reads as the
target, and on a relabel row the *raw* trio is the **student's** action.  Only 331
of the 3,116 rows in ``.tmp_v832_wp05`` agree with the teacher's ``chosen``, so a
corpus built with the raw trio teaches the student to imitate itself.

* ``--dagger-label chosen`` (**default**) — a dagger row writes the teacher's
  verified action into the trio: ``executed_cand := chosen``,
  ``executed_bomb := teacher_bomb``, ``mask := cand_to_mask(chosen, teacher_bomb)``
  (the frozen v4.2.0 table), and preserves the student's own action verbatim in
  ``student_cand`` / ``student_bomb`` / ``student_mask`` (traceability only — no
  trainer reads those as a label).
* ``--dagger-label executed`` — the pre-flag behaviour, byte for byte: the trio is
  the student's action and no ``student_*`` key is written.

A **T04 row never changes**: there the tape executed the teacher's action, so its
trio already is the label channel.  The key order of a T04 row is identical under
both modes, which is what makes the T04 block of a corpus built before this flag
and after it byte-identical.  The choice is recorded in
``meta.label_channel`` / ``meta.counts.dagger_label`` / ``meta.field_notes`` and in
each dagger source's ``selfcheck.label_channel``.

Known gap (measured, recorded in ``meta.field_notes`` and
``meta.downstream.readout01_gate_corpus``)
-----------------------------------------
``readout01.gate_corpus`` is a **single-episode T04 shard** gate: it requires
``record["decision_frame"] == 16 + 2 * index`` and an unbroken hash chain over
the whole file, and its ``check_dataset_record`` needs the nested
``provenance`` dict and a ``mask`` that round-trips with the *label* ``bomb``.
A joined corpus of several episodes can satisfy none of those by construction, so
the joined corpus is consumed through ``s2e_train.merge_shards`` (the tensors the
trainer actually reads), not through that gate.  The probe runs the gate anyway
and records its exact refusal, so the gap is evidence, not an assumption.

Usage::

    python tools/rl/v8_32_join.py build \\
        --baseline .tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed10-shard \\
        --dagger .tmp_v832_relabel/manual/out/labels.jsonl \\
        --out .tmp_v832_join/corpus --dedup --require-verified

    python tools/rl/v8_32_join.py verify --out .tmp_v832_join/corpus
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
PROG = "v8_32_join.py"

SCHEMA = "v832-join/1"
SCHEMA_ROW = "v832-join/1/slot"
SCHEMA_TARGET = "v832-join/1/target"
SCHEMA_META = "v832-join/1/meta"

N_CAND = 18
HISTORY_LEN = 8
DECISION_FRAMES = 2
FRAME_STACK = 4
HISTORY_SPAN_FRAMES = 16
MIN_DECISION_FRAME = HISTORY_SPAN_FRAMES
HASH_CHARS = 16

#: ``(key, file name, per-row shape)`` — the frozen obs-v3 planes both corpora
#: write; a source whose plane leaves this table is refused (a bit copy cannot
#: mix shapes or dtypes).
PLANE_SPEC: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("play", "play96x112.npy", (4, 112, 96)),
    ("hud", "hud.npy", (56, 160)),
    ("act_hist", "act_hist.npy", (8, 18)),
)
PLANE_DTYPE = np.dtype(np.uint8)
PLANE_DIR = "obs"
OBS_CONTRACT = "obs-v3"

SLOT_FILE = "slots.jsonl"
TARGETS_FILE = "targets.jsonl"
META_FILE = "meta.json"
DOWNSTREAM_DIR = "downstream"

SIDES = ("baseline", "dagger")
PROVENANCE = {"baseline": "t04", "dagger": "dagger"}
BOMB_SOURCE = {"baseline": "t04_bomb", "dagger": "teacher_bomb"}
DEFAULT_PRECEDENCE = "dagger"
JOIN_KEY = ("episode_seed", "decision_frame", "hash_pre")

#: which action channel a dagger row's ``executed_cand`` / ``executed_bomb`` /
#: ``mask`` trio carries.  ``chosen`` (the default) writes the **teacher's**
#: verified action there and preserves the student's own action in
#: ``student_cand`` / ``student_bomb`` / ``student_mask``; ``executed`` keeps the
#: pre-flag behaviour (the student's action in the executed trio, nothing in the
#: student trio).  T04 rows are teacher-committed in both modes and never change.
DAGGER_LABELS = ("chosen", "executed")
DEFAULT_DAGGER_LABEL = "chosen"
STUDENT_FIELDS = ("student_cand", "student_bomb", "student_mask")

#: the only top-level label kind a trainable row may carry
LABEL_SOURCE = "teacher_2f"

BASELINE_META_SCHEMA = "v420-dataset-2f"
BASELINE_ROW_SCHEMA = "v420-dataset-2f"
DAGGER_META_SCHEMA = "v832-dagger-relabel"
DAGGER_ROW_SCHEMA = "v832-dagger-relabel-label"

#: rows copied per ``npy`` block write
COPY_CHUNK = 512

DEFAULT_READOUT01 = ".tmp_v832_readout01/readout01.py"
DEFAULT_S2E_TRAIN = ".tmp_v832_s2e/s2e_train.py"

_REPO = Path(__file__).resolve().parents[2]
_TOOL_PATH = Path(__file__).resolve()


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class JoinError(RuntimeError):
    """The corpus, a source or the output violates the contract; nothing is kept."""

    def __init__(self, message: str, *, stop: str = "contract", detail: Any = None):
        super().__init__(message)
        self.stop = str(stop)
        self.detail = detail if detail is not None else {}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    return _REPO


def rel(path: str | Path) -> str:
    """Repository-relative POSIX path (an absolute path when outside the repo)."""

    text = Path(path)
    try:
        common = Path(os.path.commonpath([str(text.resolve()), str(_REPO)]))
    except ValueError:                                  # different drives
        return text.as_posix()
    if common != _REPO:
        return text.as_posix()
    return text.resolve().relative_to(_REPO).as_posix()


def host_path(declared: str | Path) -> Path:
    """Map a WSL ``/mnt/<drive>/...`` path onto this host; passthrough otherwise.

    Same translation ``readout01.host_path`` performs, and for the same reason:
    the T04 baseline's obs planes are WSL symlinks that a Windows process cannot
    open, so the row metadata names them by their ``/mnt/d/...`` target.
    """

    text = str(declared).replace("\\", "/")
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        return Path(f"{text[5].upper()}:{text[6:]}")
    return Path(text)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise JoinError(f"{rel(resolved)} does not exist", stop="source_missing")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JoinError(f"{rel(resolved)} is not readable JSON: {exc}",
                        stop="source_invalid") from exc
    if not isinstance(payload, dict):
        raise JoinError(f"{rel(resolved)} must hold a JSON object", stop="source_invalid")
    return payload


def iter_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Stream ``(1-based line number, object)``; a malformed line is refused."""

    resolved = Path(path)
    if not resolved.is_file():
        raise JoinError(f"{rel(resolved)} does not exist", stop="source_missing")
    with resolved.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise JoinError(f"{rel(resolved)}:{number} is not valid JSON: {exc}",
                                stop="source_invalid") from exc
            if not isinstance(payload, dict):
                raise JoinError(f"{rel(resolved)}:{number} is not a JSON object",
                                stop="source_invalid")
            yield number, payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")


def _repo_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        return importlib.import_module(name)


def contracts() -> Any:
    """The frozen v4.2.0 action table — the single source of the mask arithmetic."""

    try:
        return _repo_module("touhou_rl.v420.contracts")
    except ModuleNotFoundError as exc:              # pragma: no cover - env problem
        raise JoinError(f"cannot import touhou_rl.v420.contracts: {exc}",
                        stop="dependency_missing") from exc


def _int_field(value: Any, name: str, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JoinError(f"{where}: {name} {value!r} is not an int")
    return int(value)


def _flag_field(value: Any, name: str, *, where: str) -> int:
    if value in (0, 1) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, bool):
        return int(value)
    raise JoinError(f"{where}: {name} {value!r} is not a 0/1 flag")


def _hash_field(value: Any, name: str, *, where: str) -> str:
    text = str(value or "")
    if len(text) != HASH_CHARS or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise JoinError(f"{where}: {name} {value!r} is not {HASH_CHARS} hex characters")
    return text.lower()


# ---------------------------------------------------------------------------
# the normalised source row
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SourceRow:
    """One labelled decision, normalised to the joined-corpus channels."""

    side: str
    source_index: int
    source_path: str
    row_index: int
    obs_row: int
    hud_frame: int | None
    episode_seed: int
    decision_frame: int
    history: tuple[int, ...]
    executed_cand: int
    executed_bomb: int
    mask: int
    bomb: int                     # the training label channel
    declared_bomb: int            # the source row's own top-level bomb, verbatim
    teacher_bomb: int
    hash_pre: str
    hash_post: str
    label_source: str
    label_verified: bool
    scores: tuple[float | None, ...]
    null_score_indices: tuple[int, ...]
    declared_nonfinite: tuple[int, ...]
    verified_mask: tuple[int, ...]
    verified_count: int
    chosen: int | None
    chosen_verified: bool
    chosen_score: float | None
    margin: float | None
    source_slot_id: int | None
    source_shard_key: str | None
    source_provenance: Any
    relabel: Mapping[str, Any] | None
    history_frames: tuple[int, ...]
    obs_render_frames: tuple[int, ...]
    #: ``t04`` | ``chosen`` | ``executed`` — what the executed trio holds
    label_channel: str = "t04"
    #: the student's own action, kept verbatim on a ``--dagger-label chosen`` row
    #: (``None`` on a T04 row and on an ``executed``-mode dagger row, where the
    #: executed trio already *is* the student's action)
    student_cand: int | None = None
    student_bomb: int | None = None
    student_mask: int | None = None

    @property
    def join_key(self) -> tuple[int, int, str]:
        return (self.episode_seed, self.decision_frame, self.hash_pre)


@dataclasses.dataclass
class Source:
    """One input corpus: its rows, its planes and its self-check result."""

    side: str
    ordinal: int
    rows_path: Path
    meta_path: Path
    meta: Mapping[str, Any]
    rows: list[SourceRow]
    plane_paths: dict[str, Path]
    plane_arrays: dict[str, np.ndarray]
    plane_rows: int
    selfcheck: dict[str, Any]
    hashes: dict[str, Any]

    @property
    def shard_key(self) -> str:
        return f"v832join-s{self.ordinal:02d}"


# ---------------------------------------------------------------------------
# self-check bookkeeping
# ---------------------------------------------------------------------------
class Violations:
    """Named counters with the first message of each — the fail-closed report."""

    def __init__(self, label: str):
        self.label = label
        self.counts: dict[str, int] = {}
        self.first: dict[str, str] = {}

    def add(self, name: str, detail: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1
        self.first.setdefault(name, detail)

    def run(self, name: str, fn: Any) -> None:
        try:
            fn()
        except (JoinError, KeyError, TypeError, ValueError, IndexError) as exc:
            self.add(name, str(exc))

    @property
    def bad(self) -> bool:
        return bool(self.counts)

    def report(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "label": self.label,
            "violations": dict(sorted(self.counts.items())),
            "violations_total": int(sum(self.counts.values())),
            "first": dict(sorted(self.first.items())),
            "passed": not self.bad,
        }
        if extra:
            payload.update(extra)
        return payload

    def raise_if_bad(self, *, stop: str = "selfcheck_failed") -> None:
        if not self.bad:
            return
        raise JoinError(
            f"{self.label}: self-check failed on {len(self.counts)} named check(s): "
            f"{dict(sorted(self.counts.items()))}",
            stop=stop,
            detail=self.report())


# ---------------------------------------------------------------------------
# plane resolution (the readout01 rule, so a WSL-symlinked T04 shard still reads)
# ---------------------------------------------------------------------------
def _meta_plane_entries(meta: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """``{file name: entry}`` from any of the meta shapes the two corpora use."""

    found: dict[str, Mapping[str, Any]] = {}
    obs = meta.get("obs")
    if isinstance(obs, Mapping):
        for entry in obs.get("files") or []:
            if isinstance(entry, Mapping) and entry.get("name"):
                found[os.path.basename(str(entry["name"]))] = entry
    planes = meta.get("planes")
    if isinstance(planes, Mapping):
        for entry in planes.values():
            if not isinstance(entry, Mapping):
                continue
            name = os.path.basename(str(entry.get("file") or entry.get("name") or ""))
            if name and name not in found:
                found[name] = {**entry, "name": name}
    return found


def resolve_planes(root: Path, meta: Mapping[str, Any] | None, *,
                   label: str) -> tuple[dict[str, Path], dict[str, str]]:
    """Find a readable copy of each obs-v3 plane of one source, on this host.

    Returns ``({key: path}, {key: how})`` where ``how`` names the candidate that
    resolved (the T04 baseline's planes are WSL symlinks a Windows process cannot
    open, so it is ``meta.source`` in practice — recorded, not assumed).
    """

    entries = _meta_plane_entries(meta or {})
    resolved: dict[str, Path] = {}
    kinds: dict[str, str] = {}
    for key, name, _shape in PLANE_SPEC:
        candidates: list[tuple[str, Path]] = [("obs/", root / PLANE_DIR / name)]
        entry = entries.get(name) or {}
        source = entry.get("source")
        if isinstance(source, str) and source:
            candidates.append(("meta.source", host_path(source)))
        target = entry.get("target")
        if isinstance(target, str) and target:
            candidates.append(("meta.target",
                               Path(os.path.normpath(
                                   os.path.join(str(root), PLANE_DIR, target)))))
        file_name = entry.get("file")
        if isinstance(file_name, str) and file_name:
            candidates.append(("meta.file", root / file_name))
        errors: list[str] = []
        for kind, path in candidates:
            try:
                if path.is_file():
                    resolved[key] = path
                    kinds[key] = kind
                    break
            except OSError as exc:                      # WSL symlink on Windows
                errors.append(f"{kind}: {rel(path)}: {type(exc).__name__}: {exc}")
        else:
            raise JoinError(
                f"{label}: no readable copy of {PLANE_DIR}/{name}; tried "
                f"{[kind for kind, _ in candidates]} ({errors})",
                stop="plane_missing")
    return resolved, kinds


def open_planes(paths: Mapping[str, Path], *, label: str,
                resolved_via: Mapping[str, str] | None = None
                ) -> tuple[dict[str, np.ndarray], int, dict[str, Any]]:
    """Open (mem-mapped) the three planes; refuse any shape/dtype/row disagreement."""

    arrays: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {}
    rows: int | None = None
    for key, name, tail in PLANE_SPEC:
        path = paths[key]
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise JoinError(f"{label}: cannot read {rel(path)}: {type(exc).__name__}: "
                            f"{exc}", stop="plane_missing") from exc
        if array.dtype != PLANE_DTYPE:
            raise JoinError(
                f"{label}: {rel(path)} dtype {array.dtype} != {PLANE_DTYPE}; a bit "
                f"copy cannot mix dtypes", stop="plane_invalid")
        if tuple(int(dim) for dim in array.shape[1:]) != tail:
            raise JoinError(
                f"{label}: {rel(path)} per-row shape {tuple(array.shape[1:])} != "
                f"{tail} (obs-v3); a bit copy cannot mix shapes",
                stop="plane_invalid")
        count = int(array.shape[0])
        if rows is None:
            rows = count
        elif count != rows:
            raise JoinError(
                f"{label}: {rel(path)} has {count} rows but the other planes have "
                f"{rows}; the three planes must be stacked in the same order",
                stop="plane_invalid")
        arrays[key] = array
        info[name] = {"path": rel(path), "rows": count,
                      "resolved_via": (resolved_via or {}).get(key),
                      "shape": [count, *tail], "dtype": str(array.dtype)}
    assert rows is not None
    return arrays, rows, info


def verify_plane_hashes(paths: Mapping[str, Path], meta: Mapping[str, Any] | None,
                        *, label: str) -> dict[str, Any]:
    """Re-hash each plane and require the value the source meta recorded."""

    entries = _meta_plane_entries(meta or {})
    out: dict[str, Any] = {}
    for key, name, _shape in PLANE_SPEC:
        entry = entries.get(name) or {}
        recorded = entry.get("sha256")
        actual = sha256_file(paths[key])
        if recorded:
            if str(recorded) != actual:
                raise JoinError(
                    f"{label}: {rel(paths[key])} sha256 {actual} != the source meta's "
                    f"{recorded}; the plane on disk is not the one the labels were "
                    f"taken on", stop="plane_invalid")
            out[name] = {"sha256": actual, "recorded": str(recorded), "match": True}
        else:
            out[name] = {"sha256": actual, "recorded": None, "match": None}
    return out


def check_act_hist(rows: Sequence[SourceRow], act_hist: np.ndarray, *,
                   label: str) -> tuple[int, int]:
    """``act_hist[row] == one_hot(history)`` for every row (vectorised).

    This is the check that proves the obs row and the slot row belong together —
    a silently shifted plane would train the model on another state's history.
    """

    total = len(rows)
    bad = 0
    first = ""
    for start in range(0, total, COPY_CHUNK):
        block = np.asarray(act_hist[start:start + COPY_CHUNK])
        expected = np.asarray([row.history for row in rows[start:start + block.shape[0]]],
                              dtype=np.int64)
        ones = block.sum(axis=2)
        good = (ones == 1).all(axis=1) & (block.argmax(axis=2) == expected).all(axis=1)
        bad += int((~good).sum())
        if not good.all() and not first:
            index = int(np.flatnonzero(~good)[0]) + start
            first = (f"{label}: act_hist row {index} is not one_hot(history)="
                     f"{list(rows[index].history)}")
    if bad:
        raise JoinError(first, stop="selfcheck_failed")
    return total, 0


# ---------------------------------------------------------------------------
# row normalisation — baseline (T04 dataset-2f rows)
# ---------------------------------------------------------------------------
def _windows(decision_frame: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    history = tuple(decision_frame - HISTORY_SPAN_FRAMES + 2 * step
                    for step in range(HISTORY_LEN))
    render = tuple(decision_frame - FRAME_STACK + step for step in range(FRAME_STACK))
    return history, render


def _normalise_baseline_row(payload: Mapping[str, Any], *, index: int,
                            source_path: Path, ordinal: int) -> SourceRow:
    where = f"{rel(source_path)}:{index + 1}"
    if not str(payload.get("schema", "")).startswith(BASELINE_ROW_SCHEMA):
        raise JoinError(f"{where}: schema {payload.get('schema')!r} is not a "
                        f"{BASELINE_ROW_SCHEMA}/... row; this is not a T04 baseline "
                        f"shard", stop="source_invalid")
    obs = payload.get("obs") or {}
    executed = payload.get("executed") or {}
    episode_seed = _int_field(payload.get("episode_seed"), "episode_seed", where=where)
    decision_frame = _int_field(payload.get("decision_frame"), "decision_frame",
                                where=where)
    executed_cand = _int_field(payload.get("executed_cand"), "executed_cand", where=where)
    top_bomb = _flag_field(payload.get("bomb"), "bomb", where=where)
    executed_bomb = _flag_field(executed.get("bomb", top_bomb), "executed.bomb",
                                where=where)
    teacher_bomb = _flag_field(payload.get("teacher_bomb", top_bomb), "teacher_bomb",
                               where=where)
    verified_mask = tuple(_flag_field(value, "verified_mask[]", where=where)
                          for value in payload.get("verified_mask") or ())
    scores_raw = list(payload.get("scores") or ())
    scores: list[float | None] = []
    for position, value in enumerate(scores_raw):
        if value is None:
            scores.append(None)
        else:
            scores.append(float(value))
    chosen = payload.get("chosen")
    chosen = None if chosen is None else _int_field(chosen, "chosen", where=where)
    relabel = payload.get("relabel")
    history_frames, render_frames = _windows(decision_frame)
    return SourceRow(
        side="baseline",
        source_index=int(ordinal),
        source_path=rel(source_path),
        row_index=int(index),
        obs_row=int(obs.get("row", index)),
        hud_frame=None if obs.get("hud_frame") is None else int(obs["hud_frame"]),
        episode_seed=episode_seed,
        decision_frame=decision_frame,
        history=tuple(_int_field(value, "history[]", where=where)
                      for value in payload.get("history") or ()),
        executed_cand=executed_cand,
        executed_bomb=executed_bomb,
        mask=_int_field(payload.get("mask"), "mask", where=where),
        # the T04 baseline's own bomb *is* the label channel (the tape executed
        # the teacher's action, so executed == teacher there)
        bomb=top_bomb,
        declared_bomb=top_bomb,
        teacher_bomb=teacher_bomb,
        hash_pre=_hash_field(payload.get("hash_pre"), "hash_pre", where=where),
        hash_post=_hash_field(payload.get("hash_post"), "hash_post", where=where),
        label_source=str(payload.get("label_source")),
        label_verified=payload.get("label_verified") is True,
        scores=tuple(scores),
        null_score_indices=tuple(position for position, value in enumerate(scores)
                                 if value is None),
        declared_nonfinite=(),
        verified_mask=verified_mask,
        verified_count=_int_field(payload.get("verified_count", -1), "verified_count",
                                  where=where),
        chosen=chosen,
        chosen_verified=payload.get("chosen_verified") is True,
        chosen_score=None if payload.get("chosen_score") is None
        else float(payload["chosen_score"]),
        margin=None if payload.get("margin") is None else float(payload["margin"]),
        source_slot_id=payload.get("slot_id"),
        source_shard_key=payload.get("shard_key"),
        source_provenance=payload.get("provenance"),
        relabel=relabel if isinstance(relabel, Mapping) else None,
        history_frames=tuple(int(value) for value in
                             (payload.get("history_frames") or history_frames)),
        obs_render_frames=tuple(int(value) for value in
                                (payload.get("obs_render_frames") or render_frames)),
    )


def load_baseline(path: str | Path, *, ordinal: int) -> Source:
    """Read and self-check one T04 baseline shard directory."""

    root = Path(path)
    if not root.is_dir():
        raise JoinError(f"--baseline {rel(root)} is not a directory",
                        stop="source_missing")
    meta_path = root / META_FILE
    slots_path = root / SLOT_FILE
    meta = read_json(meta_path)
    if not str(meta.get("schema", "")).startswith(BASELINE_META_SCHEMA):
        raise JoinError(
            f"{rel(meta_path)}: schema {meta.get('schema')!r} is not a "
            f"{BASELINE_META_SCHEMA}/... meta; this is not a T04 baseline shard",
            stop="source_invalid")

    rows: list[SourceRow] = []
    for _number, payload in iter_jsonl(slots_path):
        rows.append(_normalise_baseline_row(payload, index=len(rows),
                                            source_path=slots_path, ordinal=ordinal))
    if not rows:
        raise JoinError(f"{rel(slots_path)} holds no rows", stop="empty_input")

    plane_paths, plane_kinds = resolve_planes(root, meta,
                                              label=f"baseline s{ordinal:02d}")
    arrays, plane_rows, plane_info = open_planes(plane_paths,
                                                 label=f"baseline s{ordinal:02d}",
                                                 resolved_via=plane_kinds)
    hashes = verify_plane_hashes(plane_paths, meta,
                                 label=f"baseline s{ordinal:02d}")

    violations = Violations(f"baseline s{ordinal:02d} ({rel(root)})")
    _self_check_rows(rows, violations)
    violations.run("rows_vs_planes", lambda: _expect_equal(
        len(rows), plane_rows,
        f"{len(rows)} slots rows but the obs planes hold {plane_rows} rows "
        f"(a truncated or extended plane is refused: the labels would not belong "
        f"to the images)"))
    counts = meta.get("counts") or {}
    if counts.get("slots") is not None:
        violations.run("rows_vs_meta", lambda: _expect_equal(
            len(rows), int(counts["slots"]),
            f"{len(rows)} slots rows but meta.counts.slots says {counts['slots']}"))
    if counts.get("seed") is not None:
        violations.run("seed_vs_meta", lambda: _check_all_equal(
            [row.episode_seed for row in rows], int(counts["seed"]),
            "episode_seed does not equal meta.counts.seed"))
    violations.run("act_hist_history", lambda: check_act_hist(
        rows, arrays["act_hist"], label=f"baseline s{ordinal:02d}"))
    violations.raise_if_bad()

    chain = _hash_chain(rows, f"baseline s{ordinal:02d}")
    selfcheck = violations.report({
        "rows": len(rows),
        "plane_rows": plane_rows,
        "planes": plane_info,
        "hash": chain,
        "label_source_counts": _counts([row.label_source for row in rows]),
        "provenance": PROVENANCE["baseline"],
    })
    return Source(side="baseline", ordinal=ordinal, rows_path=slots_path,
                  meta_path=meta_path, meta=meta, rows=rows,
                  plane_paths=plane_paths, plane_arrays=arrays,
                  plane_rows=plane_rows, selfcheck=selfcheck, hashes=hashes)


# ---------------------------------------------------------------------------
# row normalisation — DAgger relabel output
# ---------------------------------------------------------------------------
def _normalise_dagger_row(payload: Mapping[str, Any], *, index: int,
                          source_path: Path, ordinal: int,
                          dagger_label: str = DEFAULT_DAGGER_LABEL) -> SourceRow:
    where = f"{rel(source_path)}:{index + 1}"
    if dagger_label not in DAGGER_LABELS:
        raise JoinError(f"dagger_label {dagger_label!r} is not one of "
                        f"{list(DAGGER_LABELS)}")
    if not str(payload.get("schema", "")).startswith(DAGGER_ROW_SCHEMA):
        raise JoinError(f"{where}: schema {payload.get('schema')!r} is not a "
                        f"{DAGGER_ROW_SCHEMA}/... label; a student row carries no "
                        f"teacher label", stop="source_invalid")
    relabel = payload.get("relabel")
    relabel = relabel if isinstance(relabel, Mapping) else {}
    obs = payload.get("obs") or {}
    executed = payload.get("executed") or {}
    episode_seed = _int_field(payload.get("episode_seed"), "episode_seed", where=where)
    decision_frame = _int_field(payload.get("decision_frame"), "decision_frame",
                                where=where)
    executed_cand = _int_field(
        payload.get("executed_cand", executed.get("cand")), "executed_cand", where=where)
    executed_bomb = _flag_field(executed.get("bomb"), "executed.bomb", where=where)
    # the known pit: a relabel row's *top-level* bomb is the teacher's channel
    teacher_bomb = _flag_field(payload.get("teacher_bomb", payload.get("bomb")),
                               "teacher_bomb", where=where)
    declared_nulls = tuple(sorted(int(value) for value in
                                  relabel.get("nonfinite_scores") or ()))
    verified_mask = tuple(_flag_field(value, "verified_mask[]", where=where)
                          for value in payload.get("verified_mask") or ())
    scores_raw = list(payload.get("scores") or ())
    scores: list[float | None] = []
    for value in scores_raw:
        scores.append(None if value is None else float(value))
    chosen = payload.get("chosen")
    chosen = None if chosen is None else _int_field(chosen, "chosen", where=where)
    student_mask_raw = _int_field(payload.get("mask"), "mask", where=where)
    # -- the label channel --------------------------------------------------
    # ``executed_cand`` / ``executed_bomb`` / ``mask`` are the trio a trainer reads
    # as the target.  On a relabel row the *raw* trio is the student's own action
    # (``executed.cand`` / ``executed.bomb`` / ``mask``), which is NOT the teacher
    # label: only 331 of the 3,116 rows in .tmp_v832_wp05 agree with ``chosen``.
    # ``chosen`` mode therefore puts the teacher's action in the trio and keeps the
    # student's action verbatim in ``student_*``.
    if dagger_label == "chosen":
        if chosen is None:
            raise JoinError(
                f"{where}: --dagger-label chosen needs the teacher's verified "
                f"action, but this row carries chosen=null (label_verified="
                f"{payload.get('label_verified')!r}); build with --require-verified "
                f"or fall back to --dagger-label executed", stop="source_invalid")
        label_mask = int(contracts().cand_to_mask(int(chosen), int(teacher_bomb)))
        label_cand, label_bomb = int(chosen), int(teacher_bomb)
        student = (executed_cand, executed_bomb, student_mask_raw)
    else:
        label_cand, label_bomb, label_mask = (executed_cand, executed_bomb,
                                              student_mask_raw)
        student = (None, None, None)
    history_frames, render_frames = _windows(decision_frame)
    reduced = {key: relabel.get(key) for key in
               ("schema", "at", "entry_frame", "decision_index", "entry_hash",
                "query_index", "record_source", "nonfinite_scores", "tolerance",
                "hash_pre", "hash_post", "executed")}
    reduced["tape_path"] = relabel.get("tape_path")
    reduced["tape_sha256"] = relabel.get("tape_sha256")
    reduced["engine_sha256"] = relabel.get("engine_sha256")
    reduced["teacher"] = ({"bomb": (relabel.get("teacher") or {}).get("bomb"),
                           "chosen": (relabel.get("teacher") or {}).get("chosen"),
                           "chosen_score": (relabel.get("teacher") or {}).get("chosen_score"),
                           "margin": (relabel.get("teacher") or {}).get("margin"),
                           "agrees_with_chosen": (relabel.get("teacher") or {})
                           .get("agrees_with_chosen")}
                          if isinstance(relabel.get("teacher"), Mapping) else None)
    reduced["tool_sha256"] = ((relabel.get("tool") or {}).get("sha256")
                              if isinstance(relabel.get("tool"), Mapping) else None)
    return SourceRow(
        side="dagger",
        source_index=int(ordinal),
        source_path=rel(source_path),
        row_index=int(index),
        obs_row=int(obs.get("row", index)),
        hud_frame=None if obs.get("hud_frame") is None else int(obs["hud_frame"]),
        episode_seed=episode_seed,
        decision_frame=decision_frame,
        history=tuple(_int_field(value, "history[]", where=where)
                      for value in payload.get("history") or ()),
        executed_cand=label_cand,
        executed_bomb=label_bomb,
        mask=label_mask,
        bomb=teacher_bomb,
        declared_bomb=_flag_field(payload.get("bomb"), "bomb", where=where),
        teacher_bomb=teacher_bomb,
        hash_pre=_hash_field(payload.get("hash_pre"), "hash_pre", where=where),
        hash_post=_hash_field(payload.get("hash_post"), "hash_post", where=where),
        label_source=str(payload.get("label_source")),
        label_verified=payload.get("label_verified") is True,
        scores=tuple(scores),
        null_score_indices=tuple(position for position, value in enumerate(scores)
                                 if value is None),
        declared_nonfinite=declared_nulls,
        verified_mask=verified_mask,
        verified_count=_int_field(payload.get("verified_count", -1), "verified_count",
                                  where=where),
        chosen=chosen,
        chosen_verified=payload.get("chosen_verified") is True,
        chosen_score=None if payload.get("chosen_score") is None
        else float(payload["chosen_score"]),
        margin=None if payload.get("margin") is None else float(payload["margin"]),
        source_slot_id=payload.get("slot_id"),
        source_shard_key=payload.get("shard_key"),
        source_provenance=payload.get("provenance"),
        relabel=reduced,
        history_frames=tuple(int(value) for value in
                             (payload.get("history_frames") or history_frames)),
        obs_render_frames=tuple(int(value) for value in
                                (payload.get("obs_render_frames") or render_frames)),
        label_channel=str(dagger_label),
        student_cand=student[0],
        student_bomb=student[1],
        student_mask=student[2],
    )


def _dagger_obs_root(labels_dir: Path, meta: Mapping[str, Any]) -> Path:
    """The ``--save-obs`` directory a relabel output took its states from."""

    candidates: list[tuple[str, Path]] = [("labels dir", labels_dir)]
    source = meta.get("source")
    if isinstance(source, Mapping) and source.get("root"):
        declared = host_path(str(source["root"]))
        candidates.append(("meta.source.root", declared if declared.is_absolute()
                           else (_REPO / declared)))
    for kind, candidate in candidates:
        try:
            if all((candidate / PLANE_DIR / name).is_file()
                   for _key, name, _shape in PLANE_SPEC):
                return candidate
        except OSError:
            continue
    wanted = ", ".join(f"{PLANE_DIR}/{name}" for _key, name, _shape in PLANE_SPEC)
    raise JoinError(
        f"{rel(labels_dir)}: cannot find the obs planes of this relabel output; "
        f"tried {[kind for kind, _ in candidates]} (each needs {wanted})",
        stop="plane_missing")


def load_dagger(path: str | Path, *, ordinal: int,
                dagger_label: str = DEFAULT_DAGGER_LABEL) -> Source:
    """Read and self-check one ``v8_32_relabel.py run`` output."""

    if dagger_label not in DAGGER_LABELS:
        raise JoinError(f"--dagger-label {dagger_label!r} is not one of "
                        f"{list(DAGGER_LABELS)}")
    given = Path(path)
    labels_path = given / "labels.jsonl" if given.is_dir() else given
    if not labels_path.is_file():
        raise JoinError(f"--dagger {rel(given)}: no labels.jsonl there",
                        stop="source_missing")
    labels_dir = labels_path.parent
    meta_path = labels_dir / META_FILE
    meta = read_json(meta_path)
    if not str(meta.get("schema", "")).startswith(DAGGER_META_SCHEMA):
        raise JoinError(
            f"{rel(meta_path)}: schema {meta.get('schema')!r} is not a "
            f"{DAGGER_META_SCHEMA}/... meta; this is not a relabel output",
            stop="source_invalid")
    if meta.get("complete") is False:
        raise JoinError(f"{rel(meta_path)}: the relabel run is incomplete "
                        f"(complete=false); its labels are not a corpus",
                        stop="source_invalid")

    rows: list[SourceRow] = []
    for _number, payload in iter_jsonl(labels_path):
        rows.append(_normalise_dagger_row(payload, index=len(rows),
                                          source_path=labels_path, ordinal=ordinal,
                                          dagger_label=dagger_label))
    if not rows:
        raise JoinError(f"{rel(labels_path)} holds no rows", stop="empty_input")

    obs_root = _dagger_obs_root(labels_dir, meta)
    obs_meta: Mapping[str, Any] | None = None
    if obs_root != labels_dir:
        candidate = obs_root / META_FILE
        if candidate.is_file():
            obs_meta = read_json(candidate)
            if not str(obs_meta.get("schema", "")).startswith("v832-takeover-obs"):
                raise JoinError(
                    f"{rel(candidate)}: schema {obs_meta.get('schema')!r} is not a "
                    f"v832-takeover-obs/... meta; this is not a --save-obs corpus",
                    stop="source_invalid")
    plane_paths, plane_kinds = resolve_planes(obs_root, obs_meta,
                                              label=f"dagger s{ordinal:02d}")
    arrays, plane_rows, plane_info = open_planes(plane_paths,
                                                 label=f"dagger s{ordinal:02d}",
                                                 resolved_via=plane_kinds)
    hashes = verify_plane_hashes(plane_paths, obs_meta,
                                 label=f"dagger s{ordinal:02d}")

    violations = Violations(f"dagger s{ordinal:02d} ({rel(labels_path)})")
    _self_check_rows(rows, violations)

    counts = meta.get("counts") or {}
    truncated = bool(counts.get("truncated")) or (counts.get("limit") is not None)
    source_rows: int | None = None
    source_slots = obs_root / SLOT_FILE
    if source_slots.is_file():
        source_rows = sum(1 for _number, _payload in iter_jsonl(source_slots))
        violations.run("save_obs_rows_vs_planes", lambda: _expect_equal(
            source_rows, plane_rows,
            f"the --save-obs slots.jsonl holds {source_rows} rows but the planes "
            f"hold {plane_rows}; the saved corpus is truncated"))
    if truncated:
        violations.run("rows_vs_planes", lambda: _expect_ge(
            plane_rows, len(rows),
            f"{len(rows)} label rows but only {plane_rows} obs rows; --limit must "
            f"not exceed the saved corpus"))
        violations.run("obs_row_range", lambda: _check_obs_rows(
            rows, plane_rows,
            f"a label's obs.row is outside the {plane_rows}-row plane"))
    else:
        violations.run("rows_vs_planes", lambda: _expect_equal(
            len(rows), plane_rows,
            f"{len(rows)} label rows but {plane_rows} obs rows; a relabel output "
            f"writes one label per saved decision, so the counts must be equal "
            f"(the only exception is a run whose meta records a --limit)"))
    if counts.get("rows_labelled") is not None:
        violations.run("rows_vs_meta", lambda: _expect_equal(
            len(rows), int(counts["rows_labelled"]),
            f"{len(rows)} label rows but meta.counts.rows_labelled says "
            f"{counts['rows_labelled']}"))
    if meta.get("seed") is not None:
        violations.run("seed_vs_meta", lambda: _check_all_equal(
            [row.episode_seed for row in rows], int(meta["seed"]),
            f"episode_seed does not equal meta.seed {meta['seed']}"))
    violations.run("act_hist_history", lambda: check_act_hist(
        rows, arrays["act_hist"], label=f"dagger s{ordinal:02d}"))
    violations.raise_if_bad()

    chain = _hash_chain(rows, f"dagger s{ordinal:02d}")
    with_student = [row for row in rows if row.student_cand is not None]
    selfcheck = violations.report({
        "rows": len(rows),
        "plane_rows": plane_rows,
        "save_obs_root": rel(obs_root),
        "save_obs_rows": source_rows,
        "truncated": truncated,
        "planes": plane_info,
        "hash": chain,
        "label_source_counts": _counts([row.label_source for row in rows]),
        "provenance": PROVENANCE["dagger"],
        "label_channel": {
            "dagger_label": dagger_label,
            "executed_cand_is": ("chosen (the teacher's verified action)"
                                 if dagger_label == "chosen"
                                 else "executed (the student's own action)"),
            "student_channel_rows": len(with_student),
            "student_cand_differs_from_executed_cand": int(sum(
                1 for row in with_student
                if int(row.student_cand) != int(row.executed_cand))),
        },
    })
    return Source(side="dagger", ordinal=ordinal, rows_path=labels_path,
                  meta_path=meta_path, meta=meta, rows=rows,
                  plane_paths=plane_paths, plane_arrays=arrays,
                  plane_rows=plane_rows, selfcheck=selfcheck, hashes=hashes)


# ---------------------------------------------------------------------------
# the per-row self-check both sides share
# ---------------------------------------------------------------------------
def _expect_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise JoinError(message)


def _expect_ge(actual: Any, floor: Any, message: str) -> None:
    if actual < floor:
        raise JoinError(message)


def _check_all_equal(values: Sequence[int], expected: int, message: str) -> None:
    bad = sorted({int(value) for value in values if int(value) != int(expected)})
    if bad:
        raise JoinError(f"{message}: {len(bad)} other value(s) {bad[:8]}")


def _check_obs_rows(rows: Sequence[SourceRow], plane_rows: int, message: str) -> None:
    bad = [row.row_index for row in rows if not 0 <= row.obs_row < plane_rows]
    if bad:
        raise JoinError(f"{message}: first at label row {bad[0]}")


def _counts(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _hash_chain(rows: Sequence[SourceRow], label: str) -> dict[str, Any]:
    breaks = 0
    first = ""
    for position in range(1, len(rows)):
        if rows[position].hash_pre != rows[position - 1].hash_post:
            breaks += 1
            if not first:
                first = (f"{label}: hash chain breaks at row {position}: "
                         f"hash_pre {rows[position].hash_pre} != the previous row's "
                         f"hash_post {rows[position - 1].hash_post}")
    if breaks:
        raise JoinError(first, stop="selfcheck_failed")
    return {"pairs": int(len(rows)), "breaks": 0,
            "first_hash_pre": rows[0].hash_pre,
            "last_hash_post": rows[-1].hash_post}


def _self_check_rows(rows: Sequence[SourceRow], violations: Violations) -> None:
    """Every per-row rule, counted by name; the caller decides when to raise."""

    contract = contracts()
    for row in rows:
        where = f"{row.source_path} row {row.row_index}"
        violations.run("hash_format", lambda: _check_hashes(row, where))
        violations.run("decision_frame", lambda: _check_frame(row, where))
        violations.run("history", lambda: _check_history(row, where))
        violations.run("scores_length", lambda: _expect_equal(
            len(row.scores), N_CAND,
            f"{where}: scores must be {N_CAND} entries, got {len(row.scores)}"))
        violations.run("scores_finite", lambda: _check_scores(row, where))
        violations.run("verified_mask", lambda: _check_mask(row, where))
        violations.run("verified_count", lambda: _expect_equal(
            row.verified_count, sum(row.verified_mask),
            f"{where}: verified_count {row.verified_count} != the mask's "
            f"{sum(row.verified_mask)}"))
        violations.run("verified_argmax", lambda: _check_chosen(row, where))
        violations.run("chosen_range", lambda: _check_chosen_range(row, where))
        violations.run("chosen_verified", lambda: _check_chosen_verified(row, where))
        violations.run("label_verified", lambda: _check_label_verified(row, where))
        violations.run("label_source", lambda: _expect_equal(
            row.label_source, LABEL_SOURCE,
            f"{where}: label_source {row.label_source!r} != {LABEL_SOURCE!r}; a "
            f"trainable row carries a teacher label"))
        violations.run("executed_cand", lambda: _check_cand(
            row.executed_cand, f"{where}: executed_cand"))
        violations.run("bomb", lambda: _check_bomb(row, where))
        violations.run("mask_roundtrip", lambda: _check_mask_roundtrip(
            row, where, contract))
        violations.run("student_channel", lambda: _check_student_channel(
            row, where, contract))
        violations.run("windows", lambda: _check_windows(row, where))


def _check_hashes(row: SourceRow, where: str) -> None:
    for name in ("hash_pre", "hash_post"):
        value = getattr(row, name)
        if len(value) != HASH_CHARS or any(char not in "0123456789abcdef" for char in value):
            raise JoinError(f"{where}: {name} {value!r} is not {HASH_CHARS} lower-case "
                            f"hex characters")


def _check_frame(row: SourceRow, where: str) -> None:
    if row.decision_frame % DECISION_FRAMES:
        raise JoinError(f"{where}: decision_frame {row.decision_frame} is not a "
                        f"{DECISION_FRAMES}-frame slot boundary")
    if row.decision_frame < MIN_DECISION_FRAME:
        raise JoinError(f"{where}: decision_frame {row.decision_frame} < "
                        f"{MIN_DECISION_FRAME}; a shorter episode has no 16-frame "
                        f"history window")
    if row.episode_seed < 0:
        raise JoinError(f"{where}: episode_seed {row.episode_seed} is negative")


def _check_history(row: SourceRow, where: str) -> None:
    if len(row.history) != HISTORY_LEN:
        raise JoinError(f"{where}: history must be {HISTORY_LEN} candidates, got "
                        f"{len(row.history)}")
    for value in row.history:
        _check_cand(value, f"{where}: history[]", allow_none=False)


def _check_scores(row: SourceRow, where: str) -> None:
    declared = tuple(sorted(row.declared_nonfinite))
    if tuple(sorted(set(row.null_score_indices))) != declared:
        raise JoinError(
            f"{where}: the null scores {list(row.null_score_indices)} are not the "
            f"declared relabel.nonfinite_scores {list(declared)}")
    for position, value in enumerate(row.scores):
        if value is None:
            continue
        if not np.isfinite(float(value)):
            raise JoinError(f"{where}: scores[{position}] = {value!r} is not finite; "
                            f"only a declared null may stand for a non-finite score")


def _check_mask(row: SourceRow, where: str) -> None:
    if len(row.verified_mask) != N_CAND:
        raise JoinError(f"{where}: verified_mask must be {N_CAND} flags, got "
                        f"{len(row.verified_mask)}")
    for value in row.verified_mask:
        if value not in (0, 1):
            raise JoinError(f"{where}: verified_mask holds {value!r}, not a 0/1 flag")


def _check_chosen(row: SourceRow, where: str) -> None:
    verified = [index for index, flag in enumerate(row.verified_mask) if flag]
    if not verified:
        if row.chosen is not None:
            raise JoinError(f"{where}: chosen {row.chosen} on a row with no verified "
                            f"candidate")
        return
    if row.chosen is None:
        raise JoinError(f"{where}: a row with verified candidates {verified} carries "
                        f"no chosen action")
    if not 0 <= row.chosen < N_CAND:
        raise JoinError(f"{where}: chosen {row.chosen} outside [0, {N_CAND})")
    if not row.verified_mask[row.chosen]:
        raise JoinError(f"{where}: chosen {row.chosen} is not one of the verified "
                        f"candidates {verified}")
    finite = [index for index in verified if row.scores[index] is not None]
    if not finite:
        raise JoinError(f"{where}: none of the verified candidates {verified} carries "
                        f"a finite score")
    best = max(float(row.scores[index]) for index in finite)
    value = row.scores[row.chosen]
    if value is None or float(value) < best - 1e-9:
        raise JoinError(f"{where}: scores[chosen={row.chosen}] = {value!r} is below "
                        f"the best verified score {best}")


def _check_chosen_range(row: SourceRow, where: str) -> None:
    if row.chosen is None:
        return
    _check_cand(row.chosen, f"{where}: chosen")


def _check_chosen_verified(row: SourceRow, where: str) -> None:
    if bool(row.chosen_verified) is not bool(row.chosen is not None):
        raise JoinError(f"{where}: chosen_verified {row.chosen_verified!r} disagrees "
                        f"with chosen {row.chosen!r}")


def _check_label_verified(row: SourceRow, where: str) -> None:
    if bool(row.label_verified) is not bool(sum(row.verified_mask) >= 1):
        raise JoinError(f"{where}: label_verified {row.label_verified!r} is not "
                        f"exactly (verified_count >= 1)")


def _check_cand(value: Any, where: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise JoinError(f"{where} {value!r} is not an int")
    if not 0 <= int(value) < N_CAND:
        raise JoinError(f"{where} {value!r} outside [0, {N_CAND})")


def _check_bomb(row: SourceRow, where: str) -> None:
    for name in ("bomb", "teacher_bomb", "executed_bomb"):
        value = getattr(row, name)
        if value not in (0, 1):
            raise JoinError(f"{where}: {name} {value!r} is not a binary channel")
    if row.side == "baseline":
        # T04 executed the teacher's action, so the three channels coincide; a
        # row that disagrees is not a T04 baseline row
        if not row.bomb == row.teacher_bomb == row.executed_bomb == row.declared_bomb:
            raise JoinError(f"{where}: T04 bomb {row.bomb} / teacher_bomb "
                            f"{row.teacher_bomb} / executed.bomb {row.executed_bomb} "
                            f"must coincide")
    else:
        # the relabel row writes the same teacher bomb twice (top-level bomb and
        # teacher_bomb); the student's channel is executed.bomb and may differ
        if row.bomb != row.teacher_bomb:
            raise JoinError(f"{where}: the normalised bomb {row.bomb} is not the "
                            f"teacher_bomb {row.teacher_bomb}")
        if row.declared_bomb != row.teacher_bomb:
            raise JoinError(f"{where}: the row's top-level bomb {row.declared_bomb} "
                            f"is not its teacher_bomb {row.teacher_bomb}; the two "
                            f"teacher channels must agree")


def _check_mask_roundtrip(row: SourceRow, where: str, contract: Any) -> None:
    try:
        pair = contract.mask_to_cand(int(row.mask))
    except Exception as exc:                        # noqa: BLE001 - a refusal is data
        raise JoinError(f"{where}: mask {row.mask} is not a candidate mask: {exc}")
    if tuple(int(value) for value in pair) != (row.executed_cand, row.executed_bomb):
        raise JoinError(f"{where}: mask {row.mask} decodes to {tuple(pair)}, not the "
                        f"executed (cand={row.executed_cand}, "
                        f"bomb={row.executed_bomb})")
    if int(contract.cand_to_mask(row.executed_cand, row.executed_bomb)) != int(row.mask):
        raise JoinError(f"{where}: cand_to_mask({row.executed_cand}, "
                        f"{row.executed_bomb}) != mask {row.mask}")


def _check_student_channel(row: SourceRow, where: str, contract: Any) -> None:
    """The preserved student channel must be the *raw* action, and be valid.

    It never enters the label channel, so the only rules are: all three fields are
    present or all three are absent, and the mask is ``cand_to_mask(cand, bomb)``.
    """

    trio = (row.student_cand, row.student_bomb, row.student_mask)
    if all(value is None for value in trio):
        if row.label_channel == "chosen":
            raise JoinError(f"{where}: a --dagger-label chosen row lost its student "
                            f"channel (student_cand/bomb/mask are all null)")
        return
    if any(value is None for value in trio):
        raise JoinError(f"{where}: the student channel is half written: "
                        f"cand={row.student_cand!r} bomb={row.student_bomb!r} "
                        f"mask={row.student_mask!r}")
    _check_cand(row.student_cand, f"{where}: student_cand")
    if int(row.student_bomb) not in (0, 1):
        raise JoinError(f"{where}: student_bomb {row.student_bomb!r} is not a binary "
                        f"channel")
    expected = int(contract.cand_to_mask(int(row.student_cand), int(row.student_bomb)))
    if expected != int(row.student_mask):
        raise JoinError(f"{where}: student_mask {row.student_mask} is not "
                        f"cand_to_mask({row.student_cand}, {row.student_bomb}) = "
                        f"{expected}")
    if row.label_channel != "chosen":
        raise JoinError(f"{where}: a student channel is only legal on a "
                        f"--dagger-label chosen row, not on {row.label_channel!r}")


def _check_windows(row: SourceRow, where: str) -> None:
    history_frames, render_frames = _windows(row.decision_frame)
    if tuple(row.history_frames) != history_frames:
        raise JoinError(f"{where}: history_frames {list(row.history_frames)} != "
                        f"{list(history_frames)} for decision_frame "
                        f"{row.decision_frame}")
    if tuple(row.obs_render_frames) != render_frames:
        raise JoinError(f"{where}: obs_render_frames {list(row.obs_render_frames)} != "
                        f"{list(render_frames)} for decision_frame "
                        f"{row.decision_frame}")


# ---------------------------------------------------------------------------
# filters: verified-only and the join-key de-duplication
# ---------------------------------------------------------------------------
def filter_verified(items: Sequence[tuple[Source, SourceRow]]
                    ) -> tuple[list[tuple[Source, SourceRow]], dict[str, Any]]:
    kept = [item for item in items if item[1].label_verified]
    dropped = len(items) - len(kept)
    return kept, {
        "require_verified": True,
        "rows_in": len(items),
        "rows_kept": len(kept),
        "dropped_unverified": dropped,
        "dropped_by_side": _counts(item[1].side for item in items
                                   if not item[1].label_verified),
    }


def dedup(items: Sequence[tuple[Source, SourceRow]], *,
          precedence: str) -> tuple[list[tuple[Source, SourceRow]], dict[str, Any]]:
    """Keep one row per join key, preferring ``precedence``; count the conflicts."""

    groups: dict[tuple[int, int, str], list[int]] = {}
    for index, item in enumerate(items):
        groups.setdefault(item[1].join_key, []).append(index)
    winner: dict[int, int] = {}
    conflict_keys = 0
    for key, indices in groups.items():
        if len(indices) > 1:
            conflict_keys += 1
        preferred = [index for index in indices
                     if items[index][1].side == precedence] or indices
        for index in indices:
            winner[index] = preferred[0]
    kept: list[tuple[Source, SourceRow]] = []
    dropped = 0
    for index, item in enumerate(items):
        if winner[index] == index:
            kept.append(item)
        else:
            dropped += 1
    conflicts = [key for key, indices in groups.items() if len(indices) > 1]
    kept_conflicts = [key for key in conflicts if winner[groups[key][0]] is not None]
    return kept, {
        "enabled": True,
        "precedence": precedence,
        "join_key": list(JOIN_KEY),
        "keys": len(groups),
        "dedup_conflicts": conflict_keys,
        "conflict_keys": conflict_keys,
        "conflict_rows": int(sum(len(groups[key]) - 1 for key in conflicts)),
        "dedup_dropped": dropped,
        "rows_in": len(items),
        "rows_kept": len(kept),
        "conflict_rows_kept_by_side": _counts(
            items[winner[groups[key][0]]][1].side for key in kept_conflicts),
        "conflict_keys_sample": [list(key) for key in conflicts[:5]],
    }


def join_key_stats(items: Sequence[tuple[Source, SourceRow]]) -> dict[str, Any]:
    keys = [item[1].join_key for item in items]
    groups: dict[tuple[int, int, str], int] = {}
    for key in keys:
        groups[key] = groups.get(key, 0) + 1
    conflicts = {key: count for key, count in groups.items() if count > 1}
    return {
        "join_key": list(JOIN_KEY),
        "rows": len(keys),
        "unique_keys": len(groups),
        "duplicate_keys": len(conflicts),
        "duplicate_rows": int(sum(count - 1 for count in conflicts.values())),
        "duplicate_keys_sample": [list(key) for key in list(conflicts)[:5]],
        "key_sha256": sha256_text("\n".join(
            f"{seed}\t{frame}\t{pre}" for seed, frame, pre in keys)),
    }


# ---------------------------------------------------------------------------
# writing the output
# ---------------------------------------------------------------------------
def _output_row(source: Source, row: SourceRow, index: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA_ROW,
        "dataset_version": SCHEMA,
        "obs_contract": OBS_CONTRACT,
        "provenance": PROVENANCE[row.side],
        "shard_key": source.shard_key,
        "slot_id": int(index),
        "episode_seed": int(row.episode_seed),
        "decision_frame": int(row.decision_frame),
        "duration_frames": DECISION_FRAMES,
        "commit_frames": [int(row.decision_frame),
                          int(row.decision_frame) + DECISION_FRAMES],
        "obs_render_frames": [int(value) for value in row.obs_render_frames],
        "history_frames": [int(value) for value in row.history_frames],
        "history": [int(value) for value in row.history],
        "executed_cand": int(row.executed_cand),
        "executed_bomb": int(row.executed_bomb),
        "mask": int(row.mask),
        "bomb": int(row.bomb),
        "bomb_source": BOMB_SOURCE[row.side],
        "teacher_bomb": int(row.teacher_bomb),
        "scores": [None if value is None else float(value) for value in row.scores],
        "verified_mask": [int(value) for value in row.verified_mask],
        "verified_count": int(row.verified_count),
        "chosen": None if row.chosen is None else int(row.chosen),
        "chosen_verified": bool(row.chosen_verified),
        "chosen_score": None if row.chosen_score is None else float(row.chosen_score),
        "margin": None if row.margin is None else float(row.margin),
        "label_source": str(row.label_source),
        "label_verified": bool(row.label_verified),
        "hash_pre": str(row.hash_pre),
        "hash_post": str(row.hash_post),
        "obs": {"play": f"{PLANE_DIR}/{PLANE_SPEC[0][1]}",
                "hud": f"{PLANE_DIR}/{PLANE_SPEC[1][1]}",
                "act_hist": f"{PLANE_DIR}/{PLANE_SPEC[2][1]}",
                "row": int(index),
                "source_row": int(row.obs_row),
                "hud_frame": row.hud_frame},
        "source": {"side": row.side,
                   "ordinal": int(row.source_index),
                   "shard_key": source.shard_key,
                   "rows_file": row.source_path,
                   "row": int(row.row_index),
                   "slot_id": row.source_slot_id,
                   "shard_key_source": row.source_shard_key},
        "provenance_source": row.source_provenance,
    }
    if row.relabel is not None:
        payload["relabel"] = dict(row.relabel)
    #: appended last (never inserted) so a T04 row keeps the byte-for-byte key
    #: order it had before ``--dagger-label`` existed
    if row.student_cand is not None:
        payload["student_cand"] = int(row.student_cand)
        payload["student_bomb"] = int(row.student_bomb)
        payload["student_mask"] = int(row.student_mask)
    return payload


def _target_row(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA_TARGET,
        "dataset_version": SCHEMA,
        "provenance": record["provenance"],
        "shard_key": record["shard_key"],
        "slot_id": record["slot_id"],
        "episode_seed": record["episode_seed"],
        "decision_frame": record["decision_frame"],
        "executed_cand": record["executed_cand"],
        "chosen_verified_cand": record["chosen"],
        "chosen_verified": record["chosen_verified"],
        "bomb": record["bomb"],
    }


def _write_planes(out_dir: Path, items: Sequence[tuple[Source, SourceRow]]) -> dict[str, Any]:
    """Copy the selected source rows row by row into the output planes."""

    plane_dir = out_dir / PLANE_DIR
    plane_dir.mkdir(parents=True, exist_ok=True)
    total = len(items)
    info: dict[str, Any] = {}
    for key, name, tail in PLANE_SPEC:
        path = plane_dir / name
        array = np.lib.format.open_memmap(path, mode="w+", dtype=PLANE_DTYPE,
                                          shape=(total, *tail))
        cursor = 0
        index = 0
        try:
            while index < total:
                source = items[index][0]
                obs_rows: list[int] = []
                while index < total and items[index][0] is source:
                    obs_rows.append(int(items[index][1].obs_row))
                    index += 1
                plane = source.plane_arrays[key]
                for start in range(0, len(obs_rows), COPY_CHUNK):
                    selection = np.asarray(obs_rows[start:start + COPY_CHUNK],
                                           dtype=np.int64)
                    array[cursor:cursor + selection.size] = plane[selection]
                    cursor += int(selection.size)
            if cursor != total:
                raise JoinError(f"internal: copied {cursor} rows, expected {total}",
                                stop="internal")
            array.flush()
        finally:
            del array
        info[name] = {"path": f"{PLANE_DIR}/{name}", "rows": int(total),
                      "shape": [int(total), *tail], "dtype": str(PLANE_DTYPE),
                      "bytes": int(path.stat().st_size),
                      "sha256": sha256_file(path)}
    return info


def _write_jsonl(path: Path, payloads: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False,
                                    separators=(",", ":")))
            handle.write("\n")
            rows += 1
    return {"path": path.name, "rows": rows, "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path)}


# ---------------------------------------------------------------------------
# the downstream load probe (the part that must not "look readable")
# ---------------------------------------------------------------------------
def read_joined(out_dir: str | Path, *, mmap: bool = True
                ) -> dict[str, Any]:
    """Re-read an output corpus with the primitives the downstream loader uses."""

    root = Path(out_dir)
    slots = list(payload for _number, payload in
                 iter_jsonl(root / SLOT_FILE))
    if not slots:
        raise JoinError(f"{rel(root / SLOT_FILE)} holds no rows", stop="empty_input")
    planes: dict[str, np.ndarray] = {}
    for _key, name, tail in PLANE_SPEC:
        path = root / PLANE_DIR / name
        try:
            array = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise JoinError(f"{rel(path)} is not loadable: {type(exc).__name__}: {exc}",
                            stop="output_invalid") from exc
        if array.dtype != PLANE_DTYPE or tuple(int(dim) for dim in array.shape[1:]) != tail:
            raise JoinError(f"{rel(path)}: dtype/shape {array.dtype} {array.shape} is "
                            f"not uint8 (*, {tail})", stop="output_invalid")
        if int(array.shape[0]) != len(slots):
            raise JoinError(f"{rel(path)}: {array.shape[0]} rows != {len(slots)} "
                            f"slots rows", stop="output_invalid")
        planes[name] = array
    return {"dir": root, "rows": slots, "planes": planes}


def _load_script(path: str | Path, module_name: str) -> Any:
    target = Path(path)
    if not target.is_absolute():
        target = _REPO / target
    if not target.is_file():
        raise JoinError(f"downstream script {rel(target)} does not exist",
                        stop="downstream_missing")
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    spec = importlib.util.spec_from_file_location(module_name, str(target))
    if spec is None or spec.loader is None:          # pragma: no cover - env problem
        raise JoinError(f"cannot import {rel(target)}", stop="downstream_missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _t04_gate_diagnosis(rows: Sequence[Mapping[str, Any]], dataset_2f: Any) -> dict[str, Any]:
    """Which single-episode T04 rules a joined corpus breaks, by count.

    ``readout01.gate_corpus`` cannot get past the joined ``provenance`` marker
    (it expects the nested v420 dict and raises ``AttributeError`` on a string),
    so its refusal alone is not evidence of *what* is incompatible.  This re-runs
    the same per-record rule set on a **shadow copy** of each row that carries a
    nested provenance dict; every other field is the corpus's own.  Nothing here
    touches the corpus — it only names the gap in the gate's own terms.
    """

    frame_for = dataset_2f.decision_frame_for
    frame_breaks = sum(1 for index, row in enumerate(rows)
                       if int(row["decision_frame"]) != int(frame_for(index)))
    chain_breaks = sum(1 for index in range(1, len(rows))
                       if rows[index]["hash_pre"] != rows[index - 1]["hash_post"])
    slot_breaks = sum(1 for index, row in enumerate(rows)
                      if int(row["slot_id"]) != index)
    violations: dict[str, int] = {}
    first: dict[str, str] = {}
    for row in rows:
        shadow = dict(row)
        shadow["provenance"] = {
            "join_key": {"shard_key": row["shard_key"], "slot_id": row["slot_id"]},
            "corpus": {"decision_frame": row["decision_frame"]},
            "label": {"decision_frame": row["decision_frame"],
                      "teacher_chosen": row["chosen"]},
        }
        try:
            errors = dataset_2f.check_dataset_record(shadow)
        except Exception as exc:                    # noqa: BLE001 - a refusal is data
            errors = {"check_dataset_record_raised":
                      f"{type(exc).__name__}: {exc}"[:240]}
        for name, message in errors.items():
            violations[name] = violations.get(name, 0) + 1
            first.setdefault(name, str(message)[:240])
    return {
        "decision_frame_rule_breaks": int(frame_breaks),
        "hash_chain_breaks": int(chain_breaks),
        "slot_id_order_breaks": int(slot_breaks),
        "check_dataset_record_violations": dict(sorted(violations.items())),
        "check_dataset_record_first": dict(sorted(first.items())),
        "note": ("the shadow copy only swaps the provenance marker for the nested "
                 "dict the v420 checker dereferences; these are the rules a joined "
                 "corpus still breaks against a single T04 episode"),
    }


def load_corpora(out_dir: str | Path, *,
                 readout01: str | Path = DEFAULT_READOUT01) -> list[Any]:
    """The joined corpus as ``readout01.Corpus`` objects, ready for ``s2e_train``.

    The whole integration for the T04 multi-shard trainer is::

        import readout01, s2e_train, v8_32_join
        merged, info = s2e_train.merge_shards(
            readout01, v8_32_join.load_corpora("<out>"))

    One ``Corpus`` comes back because a joined corpus is **one** row space; the
    per-source partition lives in ``meta.shards`` / the contiguous ``slot_id``
    ranges, not in separate corpora (``merge_shards`` concatenates, it does not
    split).  Every array is built with ``readout01.load_corpus``'s own
    expressions; the one documented difference is ``label_source``, read from
    the row's top-level field because a joined corpus replaces the v420
    slot-level vocabulary (``teacher_slot0`` / ``executed_slot1``) with the
    corpus provenance marker.
    """

    ro = _load_script(readout01, "v832_join_readout01")
    corpus, _payload = _corpus_from_rows(ro, Path(out_dir))
    return [corpus]


def _corpus_from_rows(ro: Any, root: Path) -> tuple[Any, dict[str, Any]]:
    """One ``readout01.Corpus`` over the joined rows, with the loader's arithmetic."""

    data = read_joined(root)
    rows = data["rows"]
    total = len(rows)

    def expect(condition: Any, message: str) -> None:
        if not condition:
            raise JoinError(f"downstream loader cannot read {rel(root)}: {message}",
                            stop="downstream_unreadable")

    null_chosen = [row["slot_id"] for row in rows if row.get("chosen") is None]
    expect(not null_chosen,
           f"{len(null_chosen)} row(s) carry chosen=null (first slot "
           f"{null_chosen[0] if null_chosen else '-'}); readout01.load_corpus does "
           f"int(record['chosen']), so an unverified row is not loadable — rebuild "
           f"with --require-verified")
    null_scores = [row["slot_id"] for row in rows
                   if any(value is None for value in row.get("scores") or ())]
    expect(not null_scores,
           f"{len(null_scores)} row(s) carry a null score (first slot "
           f"{null_scores[0] if null_scores else '-'}); a float32 score vector "
           f"cannot hold it")
    chosen = np.asarray([int(row["chosen"]) for row in rows], dtype=np.int64)
    executed = np.asarray([int(row["executed_cand"]) for row in rows], dtype=np.int64)
    verified = np.asarray([[bool(value) for value in row["verified_mask"]]
                           for row in rows])
    history = np.asarray([list(row["history"]) for row in rows], dtype=np.int64)
    scores = np.asarray([list(row["scores"]) for row in rows], dtype=np.float32)
    #: the one documented difference: the joined corpus stores the corpus
    #: provenance in ``provenance`` and the label kind in the top-level
    #: ``label_source``; readout01 reads the *slot-level* v420 vocabulary out of
    #: ``provenance.corpus.label_source``, which a joined corpus does not carry
    label_source = np.asarray([str(row["label_source"]) for row in rows])
    slot_id = np.asarray([int(row["slot_id"]) for row in rows], dtype=np.int64)

    expect(np.isfinite(scores).all(), "a score vector is not finite")
    expect(np.isfinite(history).all(), "a history entry is not an integer")
    expect(sorted(set(slot_id.tolist())) == list(range(total)),
           "slot_id is not the contiguous range 0..N-1")
    expect(int(verified.sum(axis=1).min()) >= 1,
           "a row has no verified candidate; its chosen target is not a label")

    label_counts = _counts(label_source.tolist())
    gate = {"key": "joined", "rows": total, "passed": True,
            "violations_total": 0, "violations": {},
            "obs_planes_all_match": True, "hash_pairs": total,
            "executed_history_match_rows": int(total),
            "corpus_label_source_counts": label_counts,
            "source": "v8_32_join.py self-check (see meta.sources[].selfcheck)"}
    corpus = ro.Corpus(
        key="joined", seed=int(rows[0]["episode_seed"]), dir=str(root),
        planes=dict(data["planes"]), chosen=chosen, executed=executed,
        verified=verified, history=history, scores=scores,
        label_source=label_source, slot_id=slot_id, gate=gate)
    expect(corpus.n == total, f"Corpus.n {corpus.n} != {total}")
    return corpus, {"rows": rows, "total": total, "label_counts": label_counts,
                    "gate": gate}


def downstream_probe(out_dir: str | Path, *, readout01: str | Path,
                     s2e_train: str | Path) -> dict[str, Any]:
    """Push the corpus through the downstream loader's own code; fail closed.

    The probe imports ``readout01`` and ``s2e_train`` read-only (no engine, no
    training), builds ``readout01.Corpus`` objects from the joined rows with the
    same expressions ``readout01.load_corpus`` uses, and calls
    ``s2e_train.merge_shards`` / ``s2e_train.view_corpus`` — the functions the
    training run itself calls.  Everything it asserts is recorded in
    ``meta.downstream``.
    """

    root = Path(out_dir)
    started = time.perf_counter()
    ro = _load_script(readout01, "v832_join_readout01")
    s2e = _load_script(s2e_train, "v832_join_s2e_train")
    dataset_2f = ro.dataset_2f
    corpus, loaded = _corpus_from_rows(ro, root)
    rows = loaded["rows"]
    total = loaded["total"]
    label_counts = loaded["label_counts"]
    gate = loaded["gate"]

    def expect(condition: Any, message: str) -> None:
        if not condition:
            raise JoinError(f"downstream loader cannot read {rel(root)}: {message}",
                            stop="downstream_unreadable")

    # -- the functions the training run calls -------------------------------
    merged, info = s2e.merge_shards(ro, [corpus])
    expect(int(info["rows_total"]) == total,
           f"merge_shards rows_total {info['rows_total']} != {total}")
    for name in s2e.PLANE_NAMES:
        expect(name in merged.planes, f"the merged corpus lost plane {name}")
        expect(int(merged.planes[name].shape[0]) == total,
               f"plane {name} has {merged.planes[name].shape[0]} rows != {total}")
        expect(tuple(int(dim) for dim in merged.planes[name].shape[1:])
               == dict((spec[1], spec[2]) for spec in PLANE_SPEC)[name],
               f"plane {name} shape {merged.planes[name].shape[1:]} is not obs-v3")
        expect(str(merged.planes[name].dtype) == "uint8",
               f"plane {name} dtype {merged.planes[name].dtype} is not uint8")
        expect(int(info["plane_rows"][name]) == total,
               f"merge_shards counted {info['plane_rows'][name]} rows for {name}")
    for attr in s2e.LABEL_FIELDS:
        value = getattr(merged, attr)
        expect(len(value) == total,
               f"label field {attr} has {len(value)} rows != {total}")
    merged_sources = _counts(str(value) for value in merged.label_source)
    expect(merged_sources == label_counts,
           f"merged label_source {merged_sources} != {label_counts}")
    view = s2e.view_corpus(merged, ro, 0, total, key="joined-view",
                           seed=int(rows[0]["episode_seed"]), gate=gate,
                           directory=str(root))
    expect(view.n == total, f"view_corpus returned {view.n} rows != {total}")
    expect(set(view.planes) == set(merged.planes),
           "view_corpus lost a plane")

    # -- the target file against the frozen v4.2.0 target contract ----------
    targets = list(payload for _number, payload in iter_jsonl(root / TARGETS_FILE))
    expect(len(targets) == total,
           f"{TARGETS_FILE} holds {len(targets)} rows != {total}")
    target_errors: dict[str, int] = {}
    for record, target in zip(rows, targets):
        for name in dataset_2f.check_targets_record(target, record):
            target_errors[name] = target_errors.get(name, 0) + 1
    expect(not target_errors,
           f"{TARGETS_FILE} violates dataset_2f.check_targets_record: {target_errors}")

    # -- informational: the single-episode T04 gate's own verdict -----------
    gate_probe: dict[str, Any] = {"called": False}
    scratch = root / DOWNSTREAM_DIR
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        ro.OUT_DIR = str(scratch)
        result = ro.gate_corpus(str(root), "joined")
    except Exception as exc:                        # noqa: BLE001 - the refusal is data
        gate_probe = {"called": True, "passed": False,
                      "exception": type(exc).__name__,
                      "message": str(exc)[:1200]}
    else:
        gate_probe = {"called": True, "passed": True,
                      "violations": result.get("violations"),
                      "violations_total": result.get("violations_total")}
    gate_probe["diagnosis"] = _t04_gate_diagnosis(rows, dataset_2f)

    payload = {
        "called": ["readout01.Corpus", "s2e_train.merge_shards",
                   "s2e_train.view_corpus", "dataset_2f.check_targets_record",
                   "readout01.gate_corpus"],
        "readout01": {"path": rel(readout01), "sha256": sha256_file(readout01)},
        "s2e_train": {"path": rel(s2e_train), "sha256": sha256_file(s2e_train)},
        "rows": int(total),
        "rows_via_merge_shards": int(info["rows_total"]),
        "plane_rows": {name: int(value) for name, value in info["plane_rows"].items()},
        "plane_dtype": {name: str(value) for name, value in info["plane_dtype"].items()},
        "plane_shapes": {name: [int(dim) for dim in merged.planes[name].shape]
                         for name in s2e.PLANE_NAMES},
        "label_fields": list(s2e.LABEL_FIELDS),
        "label_source_counts": label_counts,
        "label_digest_executed": s2e.label_digest(merged.executed),
        "targets_rows": int(len(targets)),
        "targets_contract_errors": target_errors,
        "readout01_gate_corpus": gate_probe,
        "field_notes": {
            "label_source": ("read out of the row's top-level label_source; "
                             "readout01.load_corpus instead reads the v420 "
                             "slot-level vocabulary from provenance.corpus."
                             "label_source, which a joined corpus replaces with the "
                             "corpus provenance marker (t04|dagger)"),
            "gate_corpus": ("informational: gate_corpus audits ONE T04 episode "
                            "(decision_frame == 16 + 2*index, one unbroken hash "
                            "chain, nested provenance, mask round-tripping with the "
                            "label bomb). A join of several episodes cannot satisfy "
                            "it by construction; the tensors the training run reads "
                            "come through merge_shards, which is asserted above."),
        },
        "passed": True,
        "seconds": round(time.perf_counter() - started, 3),
    }
    return payload


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class BuildConfig:
    baselines: tuple[str, ...]
    daggers: tuple[str, ...]
    out: str
    prefix: str = DEFAULT_PRECEDENCE
    dedup: bool = False
    require_verified: bool = False
    readout01: str = DEFAULT_READOUT01
    s2e_train: str = DEFAULT_S2E_TRAIN
    dagger_label: str = DEFAULT_DAGGER_LABEL


FIELD_NOTES: dict[str, str] = {
    "bomb": ("the training label channel. A baseline row uses the T04 bomb (there "
             "the tape executed the teacher's action, so bomb == executed.bomb == "
             "teacher_bomb); a dagger row uses the relabel row's teacher_bomb, "
             "because the relabel row's top-level bomb is the *teacher* channel and "
             "the student's own bomb lives in executed.bomb. bomb_source says which "
             "one was used."),
    "bomb_source": "t04_bomb (baseline) | teacher_bomb (dagger) — provenance of the bomb channel.",
    "executed_bomb": ("the student's executed bomb channel, kept next to the label "
                      "channel so a dagger row can differ from its label."),
    "mask": ("the executed (student) mask, copied verbatim. It round-trips with "
             "(executed_cand, executed_bomb), NOT necessarily with bomb: a dagger "
             "row can execute bomb=0 while the teacher labels bomb=1. This is exactly "
             "why dataset_2f.check_dataset_record's executed_mask_roundtrip cannot "
             "accept a joined corpus."),
    "provenance": ("the corpus the row came from: t04 (T04 baseline shard) | dagger "
                   "(v8_32_relabel.py label). It replaces the v420 nested provenance "
                   "dict, which is not a corpus marker."),
    "provenance_source": ("the source row's own provenance value, verbatim: the T04 "
                          "nested dict (join_key/corpus/label) for baseline rows, "
                          "null for dagger rows (the relabel output writes null). "
                          "Read the source ids from here or from source.*, never from "
                          "the primary slot_id/shard_key."),
    "slot_id": ("renumbered 0..N-1 over the joined corpus (the source slot is "
                "source.slot_id). A T04 slot id is not unique across episodes, so it "
                "cannot be reused."),
    "shard_key": ("new per-source key v832join-sNN in source order (the source shard "
                  "is source.shard_key_source); never null, never a T04 key."),
    "decision_frame": ("preserved verbatim. After joining, decision_frame is no longer "
                       "16 + 2*slot_id: several episodes reuse the same frames, which "
                       "is why readout01.gate_corpus's decision_frame_rule cannot hold "
                       "on a joined corpus."),
    "hash_pre/hash_post": ("preserved verbatim. The chain is unbroken *inside* each "
                           "source and is counted per source in meta.sources[].selfcheck; "
                           "a joined corpus has one chain break per source after the "
                           "first, by construction."),
    "scores": ("verbatim, 18 entries. A null entry is only legal when the relabel row "
               "declares the index in relabel.nonfinite_scores; such a corpus is "
               "refused by the downstream probe because a float32 score vector cannot "
               "carry null (and a CE/soft target needs 18 finite floats)."),
    "history": ("verbatim executed-action history (8 candidates). It is checked against "
                "the act_hist plane (act_hist[row] == one_hot(history)) so a shifted "
                "plane cannot silently train on another state's history."),
    "chosen": ("the teacher's verified argmax. chosen=null only for an unverified row; "
               "readout01.load_corpus does int(record['chosen']) and therefore cannot "
               "load such a row — build with --require-verified to drop them."),
    "label_verified": "verbatim; --require-verified drops the false ones and counts them.",
    "obs": ("the output plane names + this row's index (row) + the source obs row "
            "(source_row) + hud_frame. The planes are bit copies of the selected "
            "source rows: no re-encoding, no rescaling."),
    "targets.jsonl": ("written from the joined rows with the v420 target field names "
                      "(executed_cand / chosen_verified_cand / chosen_verified / bomb) "
                      "and this corpus's schema tag; dataset_2f.check_targets_record "
                      "passes on every row (asserted at build time)."),
    "why not the v420 dataset schema": ("a joined corpus cannot pass "
                                        "dataset_2f.check_dataset_record: provenance "
                                        "is a corpus marker here, the decision_frame "
                                        "grid restarts per episode, the hash chain "
                                        "breaks per source and the label bomb need not "
                                        "equal the executed mask. The record says so "
                                        "instead of pretending to be a T04 shard."),
    "dedup": ("--dedup groups on (episode_seed, decision_frame, hash_pre) and keeps the "
              "side named by --prefix (default dagger); the conflicts are counted in "
              "meta.dedup and meta.join."),
}


def field_notes(dagger_label: str = DEFAULT_DAGGER_LABEL) -> dict[str, str]:
    """``FIELD_NOTES`` specialised to the label channel this corpus was built with."""

    notes = dict(FIELD_NOTES)
    if dagger_label == "chosen":
        notes["executed_cand"] = (
            "the training label candidate channel. A baseline row keeps the T04 "
            "action (the tape executed the teacher's action, so executed_cand == "
            "the committed action == chosen on that row); a dagger row carries the "
            "teacher's verified *chosen* action, i.e. executed_cand == chosen.")
        notes["executed_bomb"] = (
            "the training label bomb channel. A baseline row keeps the T04 bomb; a "
            "dagger row carries the teacher's bomb, i.e. executed_bomb == bomb == "
            "teacher_bomb. A dagger row's own student bomb is in student_bomb.")
        notes["mask"] = (
            "the label mask, recomputed with the frozen cand_to_mask as "
            "cand_to_mask(executed_cand, executed_bomb). On a dagger row that is "
            "cand_to_mask(chosen, teacher_bomb) -- NOT the mask the student "
            "transmitted, which is preserved in student_mask. It round-trips with "
            "(executed_cand, executed_bomb) on every row.")
        notes["student_cand"] = (
            "a dagger row's *original* student action (the relabel row's "
            "executed_cand / executed.cand), kept verbatim for traceability. It is "
            "NOT the label channel: it moved out of executed_cand when "
            "--dagger-label chosen rewrote that field. Absent on a T04 row, whose "
            "executed action already is the labelled one.")
        notes["student_bomb"] = (
            "a dagger row's original student bomb (executed.bomb of the relabel row). "
            "Not a label; may differ from executed_bomb/bomb/teacher_bomb.")
        notes["student_mask"] = (
            "a dagger row's original transmitted mask (the relabel row's own mask), "
            "which round-trips with (student_cand, student_bomb). Not a label.")
        notes["dagger_label"] = (
            "the label channel this corpus was built with: 'chosen' (this corpus) "
            "writes the teacher's action into (executed_cand, executed_bomb, mask) "
            "and the student's action into (student_cand, student_bomb, "
            "student_mask); 'executed' is the pre-flag behaviour and writes the "
            "student's action into the executed trio. T04 rows are identical in "
            "both modes.")
    else:
        notes["executed_cand"] = (
            "the executed (student) candidate channel, copied verbatim from the "
            "source row. --dagger-label executed was used, so a dagger row's "
            "executed_cand is the action the STUDENT took, which generally is not "
            "the teacher's chosen action; no student_* channel is written.")
        notes["dagger_label"] = (
            "the label channel this corpus was built with: 'executed'. The executed "
            "trio is the student's own action on a dagger row, so training on it "
            "fits the student, not the teacher.")
    return notes


def build(config: BuildConfig) -> dict[str, Any]:
    started = time.perf_counter()
    if config.dagger_label not in DAGGER_LABELS:
        raise JoinError(f"--dagger-label {config.dagger_label!r} is not one of "
                        f"{list(DAGGER_LABELS)}", stop="config_invalid")
    sources: list[Source] = []
    for ordinal, path in enumerate(config.baselines):
        sources.append(load_baseline(path, ordinal=ordinal))
    for offset, path in enumerate(config.daggers):
        sources.append(load_dagger(path, ordinal=len(config.baselines) + offset,
                                   dagger_label=config.dagger_label))
    if not sources:
        raise JoinError("no --baseline and no --dagger was given: there is nothing "
                        "to join", stop="empty_input")

    items: list[tuple[Source, SourceRow]] = [(source, row) for source in sources
                                            for row in source.rows]
    if not items:
        raise JoinError("every source holds zero rows: there is nothing to join",
                        stop="empty_input")
    rows_in = len(items)
    join_in = join_key_stats(items)

    filters: dict[str, Any] = {"require_verified": False, "rows_in": rows_in}
    if config.require_verified:
        items, filters = filter_verified(items)
        if not items:
            raise JoinError("every row was dropped by --require-verified: there is "
                            "nothing to join", stop="empty_input")
    dedup_info: dict[str, Any] = {"enabled": False}
    if config.dedup:
        items, dedup_info = dedup(items, precedence=config.prefix)
        if not items:
            raise JoinError("every row was dropped by --dedup", stop="empty_input")
    join_out = join_key_stats(items)

    out = Path(config.out)
    if out.exists() and not out.is_dir():
        raise JoinError(f"--out {rel(out)} exists and is not a directory",
                        stop="out_dir_invalid")
    if out.is_dir() and any(out.iterdir()):
        raise JoinError(f"--out {rel(out)} is not empty; refusing to overwrite a "
                        f"corpus", stop="out_dir_in_use")
    out.mkdir(parents=True, exist_ok=True)

    total = len(items)
    obs_info = _write_planes(out, items)
    records = [_output_row(source, row, index)
               for index, (source, row) in enumerate(items)]
    slots = _write_jsonl(out / SLOT_FILE, records)
    targets = _write_jsonl(out / TARGETS_FILE, (_target_row(record)
                                                for record in records))

    per_shard: list[dict[str, Any]] = []
    cursor = 0
    for source in sources:
        count = sum(1 for item in items if item[0] is source)
        if not count:
            continue
        entry: dict[str, Any] = {
            "shard_key": source.shard_key,
            "provenance": PROVENANCE[source.side],
            "side": source.side,
            "rows": int(count),
            "slot_id_range": [cursor, cursor + count - 1],
            "episode_seeds": sorted({item[1].episode_seed for item in items
                                     if item[0] is source}),
            "bomb_sources": _counts(item[1].bomb for item in items
                                    if item[0] is source),
            "rows_file": rel(source.rows_path),
            "meta_file": rel(source.meta_path),
        }
        if source.side == "dagger":
            entry["label_channel"] = config.dagger_label
            entry["executed_cand_equals_chosen"] = int(sum(
                1 for item in items if item[0] is source
                and item[1].chosen is not None
                and int(item[1].executed_cand) == int(item[1].chosen)))
            entry["student_cand_differs_from_chosen"] = int(sum(
                1 for item in items if item[0] is source
                and item[1].student_cand is not None
                and item[1].chosen is not None
                and int(item[1].student_cand) != int(item[1].chosen)))
        per_shard.append(entry)
        cursor += count

    all_pre = [record["hash_pre"] for record in records]
    all_post = [record["hash_post"] for record in records]
    joined_chain_breaks = sum(1 for index in range(1, total)
                              if all_pre[index] != all_post[index - 1])
    dagger_records = [record for record in records
                      if record["provenance"] == PROVENANCE["dagger"]]
    label_channel = {
        "dagger_label": config.dagger_label,
        "dagger_rows": len(dagger_records),
        "dagger_executed_cand_is": ("chosen (teacher's verified action)"
                                    if config.dagger_label == "chosen"
                                    else "executed (student's own action)"),
        "dagger_executed_cand_equals_chosen": int(sum(
            1 for record in dagger_records
            if record["executed_cand"] == record["chosen"])),
        "dagger_student_cand_present": int(sum(
            1 for record in dagger_records if "student_cand" in record)),
        "dagger_student_cand_differs_from_chosen": int(sum(
            1 for record in dagger_records if "student_cand" in record
            and record["student_cand"] != record["chosen"])),
        "dagger_mask_equals_cand_to_mask_label": int(sum(
            1 for record in dagger_records
            if int(record["mask"]) == int(contracts().cand_to_mask(
                int(record["executed_cand"]), int(record["executed_bomb"]))))),
        "student_channel_fields": list(STUDENT_FIELDS),
        "t04_rows": int(total - len(dagger_records)),
        "t04_unchanged_by_the_flag": True,
        "t04_note": ("a T04 row's executed_cand/executed_bomb/mask are the action the "
                     "tape actually committed (the teacher's), so they are the label "
                     "channel already and --dagger-label never touches them; the key "
                     "order of a T04 row is identical in both modes"),
        "bomb_channel_unchanged": True,
        "bomb_note": ("the bomb channel already used the teacher side before this "
                      "flag (bomb_source=teacher_bomb on a dagger row); "
                      "--dagger-label chosen only moves the *action* channel to the "
                      "same side and replaces the transmitted mask with the label "
                      "mask"),
    }
    meta: dict[str, Any] = {
        "schema": SCHEMA_META,
        "dataset_version": SCHEMA,
        "obs_contract": OBS_CONTRACT,
        "created_at": now_iso(),
        "tool": {"path": rel(_TOOL_PATH), "sha256": sha256_file(_TOOL_PATH)},
        "argv": list(sys.argv),
        "out": rel(out),
        "complete": False,
        "counts": {
            "rows": int(total),
            "rows_in": int(rows_in),
            "sources": len(sources),
            "source_rows": {f"{source.side}_s{source.ordinal:02d}": len(source.rows)
                            for source in sources},
            "rows_by_provenance": _counts(record["provenance"] for record in records),
            "rows_by_episode_seed": _counts(record["episode_seed"]
                                            for record in records),
            "label_source_counts": _counts(record["label_source"]
                                           for record in records),
            "bomb_counts": _counts(record["bomb"] for record in records),
            "bomb_source_counts": _counts(record["bomb_source"] for record in records),
            "dagger_label": config.dagger_label,
            "dagger_executed_cand_equals_chosen":
                label_channel["dagger_executed_cand_equals_chosen"],
            "dagger_student_cand_differs_from_chosen":
                label_channel["dagger_student_cand_differs_from_chosen"],
            "dagger_executed_bomb_equals_teacher_bomb": int(sum(
                1 for record in dagger_records
                if record["executed_bomb"] == record["teacher_bomb"])),
            "rows_with_student_channel":
                label_channel["dagger_student_cand_present"],
            "chosen_null": int(sum(1 for record in records
                                   if record["chosen"] is None)),
            "rows_with_null_scores": int(sum(
                1 for record in records
                if any(value is None for value in record["scores"]))),
            "label_verified": int(sum(1 for record in records
                                      if record["label_verified"])),
            "verified_count_total": int(sum(record["verified_count"]
                                            for record in records)),
            "episode_seeds": sorted({record["episode_seed"] for record in records}),
        },
        "shards": per_shard,
        "sources": [
            {
                "side": source.side,
                "ordinal": source.ordinal,
                "shard_key": source.shard_key,
                "provenance": PROVENANCE[source.side],
                "rows": len(source.rows),
                "rows_file": rel(source.rows_path),
                "rows_sha256": sha256_file(source.rows_path),
                "meta_file": rel(source.meta_path),
                "meta_sha256": sha256_file(source.meta_path),
                "meta_schema": source.meta.get("schema"),
                "obs": {key: {"path": rel(source.plane_paths[key]),
                              "sha256": source.hashes[
                                  dict((spec[0], spec[1]) for spec in PLANE_SPEC)[key]
                              ]["sha256"],
                              "recorded_sha256": source.hashes[
                                  dict((spec[0], spec[1]) for spec in PLANE_SPEC)[key]
                              ]["recorded"],
                              "rows": source.plane_rows}
                        for key, _name, _shape in PLANE_SPEC},
                "selfcheck": source.selfcheck,
            }
            for source in sources
        ],
        "join": {
            **join_in,
            "duplicate_keys_after_filters": join_out["duplicate_keys"],
            "duplicate_rows_after_filters": join_out["duplicate_rows"],
            "rows_after_filters": int(total),
        },
        "hash": {
            "hash_pre_unique": len(set(all_pre)),
            "hash_pre_duplicate_rows": int(total - len(set(all_pre))),
            "hash_post_unique": len(set(all_post)),
            "unique_ratio": round(len(set(all_pre)) / float(total), 6),
            "hash_chain_breaks_per_source": {
                f"{source.side}_s{source.ordinal:02d}":
                    int(source.selfcheck["hash"]["breaks"]) for source in sources},
            "hash_chain_breaks_joined": int(joined_chain_breaks),
            "note": ("the chain is unbroken inside every source (checked) and breaks "
                     "once per source after the first when the corpora are "
                     "concatenated; a joined corpus cannot carry one global chain"),
        },
        "obs": {
            "mode": "copy",
            "files": [{"name": f"{PLANE_DIR}/{name}",
                       "path": f"{PLANE_DIR}/{name}",
                       "source": str((out / PLANE_DIR / name).resolve()),
                       "target": None,
                       "kind": "bit-copy",
                       "sha256": obs_info[name]["sha256"],
                       "size": obs_info[name]["bytes"],
                       "rows": obs_info[name]["rows"],
                       "shape": obs_info[name]["shape"],
                       "dtype": obs_info[name]["dtype"]}
                      for _key, name, _shape in PLANE_SPEC],
            "note": ("each row is the source plane's row, copied by value into a "
                     "uint8 array of the same per-row shape; no re-encoding, no "
                     "normalisation, no rescaling"),
        },
        "slots": slots,
        "targets": targets,
        "filters": filters,
        "dedup": dedup_info,
        "precedence": config.prefix,
        "label_channel": label_channel,
        "field_notes": field_notes(config.dagger_label),
    }
    write_json(out / META_FILE, meta)

    downstream = downstream_probe(out, readout01=config.readout01,
                                  s2e_train=config.s2e_train)
    meta["downstream"] = downstream
    meta["complete"] = bool(downstream.get("passed"))
    meta["wall_s"] = round(time.perf_counter() - started, 3)
    write_json(out / META_FILE, meta)

    return {
        "ok": True,
        "out": rel(out),
        "rows": int(total),
        "rows_in": int(rows_in),
        "sources": len(sources),
        "rows_by_provenance": meta["counts"]["rows_by_provenance"],
        "label_source_counts": meta["counts"]["label_source_counts"],
        "hash_pre_unique": meta["hash"]["hash_pre_unique"],
        "join_duplicate_keys": meta["join"]["duplicate_keys"],
        "dedup": dedup_info,
        "filters": filters,
        "dagger_label": config.dagger_label,
        "label_channel": label_channel,
        "obs_sha256": {name: obs_info[name]["sha256"]
                       for _key, name, _shape in PLANE_SPEC},
        "slots_sha256": slots["sha256"],
        "targets_sha256": targets["sha256"],
        "downstream_passed": downstream["passed"],
        "downstream_rows": downstream["rows_via_merge_shards"],
        "readout01_gate_corpus": downstream["readout01_gate_corpus"].get("passed"),
        "wall_s": meta["wall_s"],
    }


# ---------------------------------------------------------------------------
# verify (re-audit an existing output)
# ---------------------------------------------------------------------------
def verify(out_dir: str | Path, *, readout01: str = DEFAULT_READOUT01,
           s2e_train: str = DEFAULT_S2E_TRAIN) -> dict[str, Any]:
    root = Path(out_dir)
    meta = read_json(root / META_FILE)
    if str(meta.get("schema", "")) != SCHEMA_META:
        raise JoinError(f"{rel(root / META_FILE)}: schema {meta.get('schema')!r} != "
                        f"{SCHEMA_META!r}", stop="output_invalid")
    data = read_joined(root)
    records = data["rows"]
    total = len(records)
    problems: list[str] = []
    if total != int(meta["counts"]["rows"]):
        problems.append(f"rows {total} != meta.counts.rows {meta['counts']['rows']}")
    pre = [record["hash_pre"] for record in records]
    unique = len(set(pre))
    if unique != int(meta["hash"]["hash_pre_unique"]):
        problems.append(f"hash_pre unique {unique} != meta {meta['hash']['hash_pre_unique']}")
    provenance = _counts(record["provenance"] for record in records)
    if provenance != meta["counts"]["rows_by_provenance"]:
        problems.append(f"provenance {provenance} != meta {meta['counts']['rows_by_provenance']}")
    labels = _counts(record["label_source"] for record in records)
    if labels != meta["counts"]["label_source_counts"]:
        problems.append(f"label_source {labels} != meta {meta['counts']['label_source_counts']}")
    ids = [int(record["slot_id"]) for record in records]
    if ids != list(range(total)):
        problems.append("slot_id is not the contiguous range 0..N-1")
    plane_sha: dict[str, Any] = {}
    for _key, name, _shape in PLANE_SPEC:
        path = root / PLANE_DIR / name
        actual = sha256_file(path)
        recorded = next(entry["sha256"] for entry in meta["obs"]["files"]
                        if entry["name"] == f"{PLANE_DIR}/{name}")
        plane_sha[name] = {"sha256": actual, "recorded": recorded,
                           "match": actual == recorded}
        if actual != recorded:
            problems.append(f"{name} sha256 {actual} != meta {recorded}")
    if problems:
        raise JoinError(f"{rel(root)}: re-audit failed: {problems}",
                        stop="verify_failed", detail={"problems": problems})
    downstream = downstream_probe(root, readout01=readout01, s2e_train=s2e_train)
    return {"ok": True, "out": rel(root), "rows": total,
            "rows_by_provenance": provenance,
            "label_source_counts": labels,
            "hash_pre_unique": unique,
            "planes": plane_sha,
            "downstream_passed": downstream["passed"],
            "downstream_rows": downstream["rows_via_merge_shards"],
            "readout01_gate_corpus": downstream["readout01_gate_corpus"].get("passed")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=("Join T04 baseline shards and v8_32_relabel.py teacher labels "
                     "into one trainable corpus (offline; no engine)."))
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd = sub.add_parser(
        "build", help="write <out>/slots.jsonl + obs/ + targets.jsonl + meta.json")
    build_cmd.add_argument("--baseline", action="append", default=[], metavar="DIR",
                           help="a T04 baseline shard (slots.jsonl + obs/ + meta.json); "
                                "repeatable")
    build_cmd.add_argument("--dagger", action="append", default=[], metavar="LABELS",
                           help="a v8_32_relabel.py output (labels.jsonl + meta.json); "
                                "repeatable")
    build_cmd.add_argument("--out", required=True, metavar="DIR",
                           help="output corpus directory (must be absent or empty)")
    build_cmd.add_argument("--prefix", choices=SIDES, default=DEFAULT_PRECEDENCE,
                           help="the side --dedup keeps on a join-key conflict "
                                "(default: dagger)")
    build_cmd.add_argument("--dagger-label", choices=DAGGER_LABELS,
                           default=DEFAULT_DAGGER_LABEL, dest="dagger_label",
                           metavar="{%s}" % ",".join(DAGGER_LABELS),
                           help="which action channel a dagger row's executed_cand/"
                                "executed_bomb/mask carries: 'chosen' (default) writes "
                                "the teacher's verified action and preserves the "
                                "student's action in student_cand/student_bomb/"
                                "student_mask; 'executed' is the pre-flag behaviour "
                                "(the student's own action, no student_* channel). "
                                "T04 rows are identical in both modes.")
    build_cmd.add_argument("--dedup", action="store_true",
                           help="drop duplicate (episode_seed, decision_frame, "
                                "hash_pre) rows, keeping --prefix's side")
    build_cmd.add_argument("--require-verified", action="store_true",
                           dest="require_verified",
                           help="drop label_verified=false rows (counted in meta)")
    build_cmd.add_argument("--readout01", default=DEFAULT_READOUT01, metavar="PATH",
                           help=f"the downstream gate module (default: "
                                f"{DEFAULT_READOUT01})")
    build_cmd.add_argument("--s2e-train", default=DEFAULT_S2E_TRAIN, metavar="PATH",
                           dest="s2e_train",
                           help=f"the downstream training script (default: "
                                f"{DEFAULT_S2E_TRAIN})")

    verify_cmd = sub.add_parser("verify", help="re-audit an existing output corpus")
    verify_cmd.add_argument("--out", required=True, metavar="DIR")
    verify_cmd.add_argument("--readout01", default=DEFAULT_READOUT01, metavar="PATH")
    verify_cmd.add_argument("--s2e-train", default=DEFAULT_S2E_TRAIN, metavar="PATH",
                            dest="s2e_train")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build(BuildConfig(
                baselines=tuple(args.baseline), daggers=tuple(args.dagger),
                out=args.out, prefix=str(args.prefix), dedup=bool(args.dedup),
                require_verified=bool(args.require_verified),
                readout01=str(args.readout01), s2e_train=str(args.s2e_train),
                dagger_label=str(args.dagger_label)))
        elif args.command == "verify":                 # pragma: no cover - thin
            result = verify(args.out, readout01=str(args.readout01),
                            s2e_train=str(args.s2e_train))
        else:                                          # pragma: no cover - argparse
            parser.error(f"unknown command {args.command!r}")
            return 2
    except JoinError as error:
        payload = {"ok": False, "stop": error.stop, "error": str(error),
                   "detail": error.detail}
        print(json.dumps(payload, ensure_ascii=False, indent=1), file=sys.stderr)
        return 4
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":                             # pragma: no cover - entry
    raise SystemExit(main())
