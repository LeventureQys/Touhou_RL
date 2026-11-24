"""Build a ``v829-pair-manifest/1`` from a research-side ``v829-sources/1`` sheet.

Why this file exists
--------------------
The run table `v829_guard.py` admits is full of numbers nobody can *derive*:
`takeover_frame`, `total_frames`, `parent_prefix_digest` and `verify_anchor` are
required per root and deliberately never inherited (`v829_guard.py` docstring; the
v8.29 INFO-01 inventory found no field for any of them in
`configs/v430/splits.json`).  So the research side cannot hand over a manifest
directly -- and if it tried, it would be retyping a 64-character digest next to a
frame count that has to satisfy `takeover % 4 == 3` and
`takeover + budget <= total_frames`.

This tool is the missing segment: the research side supplies only what it
actually knows (which recording, at which 4k-1 frame, how long, which checkpoint,
whether that recording is already exposed), and every derived number is computed
from the tape's **bytes** on disk.  The finished manifest is then handed to
`tools/rl/v829_guard.py` **in a subprocess of this same interpreter**, so "the
builder said it is fine" is never the last word.

Input (schema ``v829-sources/1``)
---------------------------------
::

    {
      "schema": "v829-sources/1",
      "experiment_id": "v829-pilot-001",
      "engine": "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern",
      "data_source": "/root/v400-data/game",
      "budget_frames": 1800,
      "ckpt": {"path": ".tmp_v860_bombdag/ckpt/seed0/bombdag.pt",
               "sha256": "9c6d5d4a... (64 lowercase hex)"},
      "sources": [
        {
          "ancestor_id": "anon-01",
          "stage": "th08-stage1-normal-team0",
          "tape": ".tmp_v400_dagger2/r2/shards/dagger-s0d1seed3006/labels.jsonl",
          "takeover_frame": 2999,
          "pair_rng": 20261201,
          "used_for_training": false,
          "used_for_selection": false,
          "lineage_new": false
        }
      ]
    }

Optional per source: ``window`` (int), ``lineage_ancestor_ids`` (list of strings,
passed through for ``--lineage-config``).  Optional top level: ``purpose``,
``manifest_id``, ``engine_sha256`` (a claim that is *verified*, never copied).
`build --help` lists the CLI overrides.

One ancestor, 1-3 checkpoints: the top-level ``ckpts`` array
------------------------------------------------------------
``PREREGISTRATION_PAIR_001`` §2 evaluates **3 checkpoints x 2 pair RNGs per
ancestor**, and an *ancestor* is an independent play-root -- the checkpoint is a
repeat-measurement axis inside it.  A top-level ``ckpt`` object binds exactly one
checkpoint and therefore cannot express that shape.  The optional ``ckpts`` array
can::

    {
      "schema": "v829-sources/1",
      "experiment_id": "v829-rehearsal-8x3-001",
      "engine": "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern",
      "data_source": "/root/v400-data/game",
      "budget_frames": 1800,
      "ckpts": [
        {"path": ".tmp_v860_bombdag/ckpt/seed0/bombdag.pt",
         "sha256": "9c6d5d4a...", "seed": 0},
        {"path": ".tmp_v860_bombdag/ckpt/seed1/bombdag.pt",
         "sha256": "36a51090...", "seed": 1},
        {"path": ".tmp_v860_bombdag/ckpt/seed2/bombdag.pt",
         "sha256": "885dde70...", "seed": 2}
      ],
      "sources": [
        {"ancestor_id": "reh-seed3201", "stage": "th08-stage1-normal-team0",
         "window": 3201, "tape": ".tmp_v829_collect/seed3201/labels.jsonl",
         "takeover_frame": 2999,
         "used_for_training": false, "used_for_selection": false,
         "lineage_new": true}
      ]
    }

Rules:

1. ``ckpts`` holds **1 to 3** entries; each needs ``path`` and ``sha256``
   (verified against the bytes on disk, exactly like ``ckpt``).  ``seed`` and
   ``pair_index`` are optional integers.
2. ``ckpt`` and ``ckpts`` are **mutually exclusive**: giving both is
   ``SOURCES_INVALID``, because "which one binds the root" has no safe answer.
   ``--ckpt`` / ``--ckpt-sha256`` are refused in this mode for the same reason
   (they can only override one of several checkpoints).
3. Every source is expanded once **per checkpoint**: ``n`` sources x ``m``
   checkpoints = ``n * m`` roots, and all ``m`` roots of one source share its
   ``ancestor_id``.  The roots differ only in ``ckpt_sha256`` / ``ckpt_path`` /
   ``pair_rng`` / ``pair_index`` / ``ckpt_seed`` -- everything that describes the
   play-root itself (stage, window, tape, takeover frame, total frames, budget,
   prefix digest) is derived from the one source row and is therefore identical by
   construction, so the guard's ``ANCESTOR_INCONSISTENT`` gate is satisfied by
   construction rather than by luck.
4. ``pair_rng`` is **allocated from the indices**, not copied from the sheet::

       pair_rng = PAIR_RNG_BASE
                + rng_index    * PAIR_RNG_RNG_STRIDE
                + source_index * PAIR_RNG_SOURCE_STRIDE
                + ckpt_index
       PAIR_RNG_BASE = 20265000   PAIR_RNG_RNG_STRIDE = 1000
       PAIR_RNG_SOURCE_STRIDE = 10

   so source 0 gets 20265000/01/02, source 1 gets 20265010/11/12, and an 8-source
   x 3-checkpoint sheet lands on 20265000-20265072: unique by construction, and
   outside the allocations already on disk (20261601-20261603 park-reuse,
   20263001-20263008 rehearsal).  A repeated measurement moves a whole
   1,000-wide layer (``rng_index`` 1 = 20266000-20266072) rather than shifting the
   checkpoint digits, so repeat 0 keeps exactly the numbers the 24-root rehearsal
   already uses.  A source-level ``pair_rng`` is **refused** whenever this tool
   allocates (``ckpts`` or ``rngs > 1``; ``SOURCES_INVALID``) instead of being
   interpreted: one seed on the row cannot address several cells, and picking one
   for it silently would be an inference.  In the single-``ckpt`` mode with
   ``rngs: 1`` ``pair_rng`` stays required, unchanged.
5. Each generated root records ``pair_index`` and ``ckpt_seed``.  ``pair_index``
   is the repeat number the guard keys a cell on: with ``rngs > 1`` it is the
   repeat index ``0 .. rngs - 1`` inside the cell, counted per checkpoint; with
   ``rngs == 1`` it keeps the earlier meaning -- ``ckpts[i].pair_index`` when
   given, else the checkpoint index.  A ``ckpts[i].pair_index`` is refused when
   ``rngs > 1`` (one repeat number per checkpoint cannot address N repeats, and
   silently reinterpreting it would be an inference).  ``ckpt_seed`` is
   ``ckpts[i].seed`` when given.

The second §2 axis: the top-level ``rngs`` count
------------------------------------------------
§2 evaluates **3 checkpoints x 2 pair RNGs per ancestor**, so ``ckpts`` alone is
only half the grid.  The optional top-level ``rngs`` (an integer ``1..16``,
default 1) says how many repeat measurements every ``(source, checkpoint)`` cell
gets::

    "rngs": 2

Every source then expands once per checkpoint *and* once per repeat, with
``pair_index`` ``0 .. rngs - 1`` inside each cell; all of them still share the
source's single ``ancestor_id``.  8 sources x 3 checkpoints x 2 repeats = 48
roots, which is the pre-registered shape.  ``rngs`` also works without ``ckpts``
(a single ``ckpt``: n sources x 1 checkpoint x N repeats), and a repeated
measurement is not a new ancestor -- it is the same cell measured again, which is
what the guard's ``(ancestor_id, ckpt_sha256, pair_index)`` key reads.

Output (schema ``v829-pair-manifest/1``): one root per source (single ``ckpt``),
per (source, checkpoint) (``ckpts``), or per (source, checkpoint, repeat) when
``rngs > 1``.  Derivation table
---------------------------------------------------------------------------
==============================  ===============================================
manifest field                  derived from
==============================  ===============================================
``parent_prefix_digest``        ``sha256`` of the tape's raw bytes
``verify_anchor.prefix_digest`` same value (the guard's ``DIGEST_MISMATCH``
                                redundancy is kept on purpose)
``verify_anchor.window_recording_sha256``  same value
``verify_anchor.prefix_record``  the tape path as the sheet spells it
                                (repo-relative; an absolute path is kept
                                verbatim and flagged ``absolute: true``)
``verify_anchor.recorded_frames``  ``takeover_frame``
``verify_anchor.hash_checkpoints`` ``takeover_frame // 4`` (= ``N // 4``)
``total_frames``                ``rows * 4`` = the tape's ``frames_built``
``total_frames_source``         one generated sentence: the row count, the tape
                                it was counted from, and the budget headroom
``prefix_tape`` / ``_sha256`` / ``_rows`` / ``_frames``  the source tape
``window``                      ``sources[i].window``, else ``--window``, else a
                                stable integer derived from ``ancestor_id``
                                (``window_derived: true``); ``--window`` only
                                selects a root and names the private data dir
``budget_frames``               the sheet, or ``--budget`` (recorded as
                                ``budget_source``)
``ckpt_sha256``                 ``sha256`` measured on disk, compared with the
                                sheet's claim (``CKPT_SHA_MISMATCH``); with
                                ``ckpts`` one root per entry
``ckpt_path``                   the sheet's path (guard ignores it; it lets
                                ``v829_pair.py --mode check`` print an argv)
``pair_rng``                    single ``ckpt``: ``sources[i].pair_rng``; ``ckpts``:
                                ``PAIR_RNG_BASE + i * PAIR_RNG_SOURCE_STRIDE + j``
``pair_index`` / ``ckpt_seed``  ``ckpts`` only: the repeat number and the seed that
                                produced that checkpoint (``ckpts[j].seed``)
``engine_sha256``               ``sha256`` measured on disk -- the sheet's
                                ``engine_sha256``, if present, is verified and
                                never trusted (``ENGINE_SHA_MISMATCH``)
``used_for_training`` / ``used_for_selection``  the sheet, both required: a
                                missing exposure is ``EXPOSURE_REQUIRED``, never
                                a silent ``false``
==============================  ===============================================

The tape reader in this file, and its proof of equivalence
----------------------------------------------------------
:func:`scan_tape` is a port of the row rules `v829_pair.read_tape` enforces
(`tools/rl/v829_pair.py`, ``def read_tape`` / ``ANCHOR_STRIDE = 4``); it is longer
than a bare counter only because every refusal names the offending row and frame:

1. every row carries exactly 4 ``step_frames``;
2. ``decision_frame`` is ``rows_so_far * 4`` (start at 0, step 4, no gap /
   overlap / duplicate), so every anchor is a multiple of 4;
3. every row carries both ``hash_pre`` and ``hash_post``, which (1)+(2) turns
   into the complete anchor set ``{0, 4, ..., frames_built}`` that
   ``walk_prefix`` requires;
4. ``frames_built = rows * 4`` and at least one row must exist.

Importing `v829_pair.py` is not expensive (measured 0.01 s, no engine import at
module level), but the build path deliberately stays independent of the run path:
the manifest must be derivable on any host, and a builder that imports the runner
cannot be used to check the runner.  The equivalence is therefore *tested*, not
assumed::

    python -B tools/rl/v829_manifest_build.py equivalence --tape T --takeover N

imports `tools/rl/v829_pair.py` **only** in that subcommand, calls
``read_tape(T, N + 1)``, and prints both sides field by field --
``sha256``, ``rows``, ``frames_built`` and ``hash_checkpoints`` (= ``N // 4``,
which `v829_pair.py` also calls ``expected_checkpoints``).  Any disagreement is a
non-zero exit.  `REPORT.md` in the delivery directory records the run.

No timestamp: byte-identical reruns
-----------------------------------
This file writes **no** ``generated_at`` and does not read ``SOURCE_DATE_EPOCH``.
Every field is a pure function of (a) the sources JSON bytes and (b) the tape,
checkpoint and engine bytes on disk, so the same input yields the same output
file **byte for byte**.  A timestamp would be the only non-reproducible field and
it would make the manifest digest useless as an identity: the SHA of the file is
what "the manifest did not change" means downstream (`SCHEMA_CLOSURE` /
``MANIFEST_CHANGED`` in `v829_pair.py`).  The check is: build twice, compare
``sha256``.  `REPORT.md` records the pair.

Error codes (every one is a refusal; nothing is written when one fires)
---------------------------------------------------------------------
==========================  ====  ===============================================
``SOURCES_UNREADABLE``        2   sheet missing / not UTF-8 / not a JSON object /
                                  an object that repeats a key (``json.loads``
                                  would keep the last value, so the sheet being
                                  judged is already ambiguous)
``SOURCES_INVALID``           1   schema is not ``v829-sources/1``, or a required
                                  field is missing or mistyped, or the same
                                  ``(ancestor_id, ckpt)`` appears twice, or
                                  ``ckpt`` and ``ckpts`` are both given, or
                                  ``ckpts`` is not a 1-3 array of
                                  ``{path, sha256[, seed][, pair_index]}``
                                  objects, or a source declares ``pair_rng`` in
                                  ``ckpts`` mode (see the allocation rule above)
``EXPOSURE_REQUIRED``         1   ``used_for_training`` / ``used_for_selection``
                                  absent or not a boolean -- an unknown exposure
                                  is a refusal, not a default
``TAKEOVER_NOT_ANCHOR``       1   ``takeover_frame % 4 != 3`` (the tape anchors
                                  every 4 frames and the runner walks ``N + 1``)
``TAKEOVER_BEYOND_TAPE``      1   ``takeover_frame + 1 > frames_built``: target
                                  ``N + 1`` is not a recorded anchor of this tape
``BUDGET_EXCEEDS_TOTAL``      1   ``takeover_frame + budget_frames > total_frames``
``BUDGET_NOT_A_TIER``         1   ``budget_frames`` is not one of the
                                  pre-registered tiers (1800, 2400), so the guard
                                  could not prove the requested tier
``TAPE_MISSING``              2   the tape is not a readable file on this host
``TAPE_MALFORMED``            1   a row is not 4 frames, or ``decision_frame`` is
                                  not 0/4/8..., or ``hash_pre``/``hash_post`` is
                                  missing (this is also how an empty tape is
                                  refused: 0 rows cannot carry a recorded anchor)
``RNG_DUPLICATE``             1   two roots share a ``pair_rng`` under the same
                                  ``(ancestor_id, budget_frames, ckpt)``
``WINDOW_DUPLICATE``          1   two roots name the same window identity
                                  ``(ckpt sha256, tape sha256, takeover_frame,
                                  budget_frames)`` under different labels -- one
                                  physical window counted twice would move the
                                  section 4 average.  The checkpoint is inside the
                                  key because 3 checkpoints over one tape are 3
                                  different windows, which is the section 2 axis
``CKPT_SHA_MISMATCH``         1   the measured checkpoint digest is not the one the
                                  sheet claims (reported per ``ckpts[j]``)
``CKPT_MISSING``              2   the checkpoint is not a readable file
``ENGINE_MISSING``            2   the engine is not a readable file, so
                                  ``engine_sha256`` cannot be measured at all
``ENGINE_SHA_MISMATCH``       1   the sheet declares an ``engine_sha256`` that the
                                  file on disk contradicts
``OUT_UNWRITABLE``            2   the manifest (or its parent directory) could not
                                  be written
``GUARD_UNRUNNABLE``          2   the guard subprocess could not be started
==========================  ====  ===============================================

The guard tail (mandatory)
--------------------------
After the manifest is written, this tool runs::

    <this interpreter> -B tools/rl/v829_guard.py check <out> --budget <budget> \
        [--allow-exposed]

prints the guard's **exit code and full stdout**, writes the same run to
``<out>.guard.json`` (argv, exit, stdout, stderr, manifest sha256) and adopts the
guard's exit code as its own when the guard refuses.  ``--allow-exposed`` is
forwarded **only** when the operator passed it, so an exposed root still needs
the explicit acknowledgement the guard demands.

If the guard refuses, the manifest is **kept at the exact ``--out`` path** (never
renamed, never deleted) and the build report plus the sidecar are clearly stamped
``UNVALIDATED``: a human needs to see the file that was refused.  The exit code is
the guard's own (1 = refused, 2 = could not evaluate, e.g. unreadable bytes).

Exit codes
----------
``0``  manifest written and the guard exited 0
``1``  refused: at least one finding, or the guard refused with 1
``2``  could not evaluate: an input/output/environment input was unreadable, or
       the guard exited 2 (``argparse`` usage errors also exit 2)

Usage
-----
::

    python -B tools/rl/v829_manifest_build.py build --sources S.json --out M.json \\
        [--window W] [--budget 1800|2400] [--allow-exposed] [--report R.json] \\
        [--ckpt PATH] [--ckpt-sha256 HEX] [--engine PATH] [--guard PATH]
    python -B tools/rl/v829_manifest_build.py equivalence --tape T --takeover N
    python -B tools/rl/v829_manifest_build.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TOOL = "tools/rl/v829_manifest_build.py"
#: v8.29.2: added the top-level ``rngs`` count (1-16 repeat measurements per
#: (source, checkpoint) cell), the repeat layer of the ``pair_rng`` formula and the
#: ``pair_index`` repeat number, so the second §2 axis -- 3 checkpoints x 2 pair
#: RNGs -- is writable in one manifest.
TOOL_VERSION = "v8.29.2"

SOURCES_SCHEMA = "v829-sources/1"
MANIFEST_SCHEMA = "v829-pair-manifest/1"

#: One recorded hash anchor every 4 frames (`v829_pair.ANCHOR_STRIDE`); a legal
#: takeover frame is ``4k - 1`` because the runner walks ``target = N + 1``.
ANCHOR_STRIDE = 4
TAKEOVER_MOD = ANCHOR_STRIDE - 1

#: The two pre-registered budget tiers.  `v829_guard.py` freezes them as
#: ``--budget`` choices and checks the requested tier against every root, so a
#: manifest built at any other budget could not be validated by the guard.
BUDGET_TIERS = (1800, 2400)
DEFAULT_GUARD = "tools/rl/v829_guard.py"

#: ``ckpts`` (one ancestor, several checkpoints): 1-3 entries, and the seed of each
#: cell is allocated from the indices so it is unique by construction::
#:
#:     pair_rng = PAIR_RNG_BASE
#:              + rng_index * PAIR_RNG_RNG_STRIDE
#:              + source_index * PAIR_RNG_SOURCE_STRIDE
#:              + ckpt_index
#:
#: 20265000 sits outside the allocations already on disk -- 20261601-20261603
#: (park-reuse) and 20263001-20263008 (rehearsal) -- and the stride of 10 leaves
#: room for the three checkpoints before the next source starts.  The repeat layer
#: is 1,000 wide, so ``rng_index`` 0 keeps the 24-root rehearsal's numbers exactly
#: and repeat 1 starts at 20266000: 8 sources x 3 checkpoints x 16 repeats still
#: fit in one layer of 1,000 (the largest offset is 7 * 10 + 2 = 72).
MAX_CKPTS = 3
#: §2's second axis: each (source, checkpoint) cell may be measured 1..16 times
#: with different pair RNGs.  Above 16 is outside the pre-registration and refused.
MAX_RNGS = 16
PAIR_RNG_BASE = 20265000
PAIR_RNG_SOURCE_STRIDE = 10
PAIR_RNG_RNG_STRIDE = 1000

#: `--window` is only a label (it selects the root and names the data dir), so a
#: sheet without one still needs a stable integer.  It is derived from the
#: ``ancestor_id`` and kept in a 6-digit range that cannot collide with the four
#: frozen S1c windows ``(11, 12, 17, 18)``.
WINDOW_SALT = "v829-manifest-build/window/1"
WINDOW_LOW = 100000
WINDOW_SPAN = 900000

EXPOSURE_FIELDS = ("used_for_training", "used_for_selection")
SOURCE_TEXT_FIELDS = ("ancestor_id", "stage", "tape")

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_FATAL = 2

#: code -> exit code.  Definition order is the reporting order.
CODE_EXIT = {
    "SOURCES_UNREADABLE": EXIT_FATAL,
    "SOURCES_INVALID": EXIT_REFUSED,
    "EXPOSURE_REQUIRED": EXIT_REFUSED,
    "TAKEOVER_NOT_ANCHOR": EXIT_REFUSED,
    "TAKEOVER_BEYOND_TAPE": EXIT_REFUSED,
    "BUDGET_EXCEEDS_TOTAL": EXIT_REFUSED,
    "BUDGET_NOT_A_TIER": EXIT_REFUSED,
    "TAPE_MISSING": EXIT_FATAL,
    "TAPE_MALFORMED": EXIT_REFUSED,
    "RNG_DUPLICATE": EXIT_REFUSED,
    "WINDOW_DUPLICATE": EXIT_REFUSED,
    "CKPT_SHA_MISMATCH": EXIT_REFUSED,
    "CKPT_MISSING": EXIT_FATAL,
    "ENGINE_MISSING": EXIT_FATAL,
    "ENGINE_SHA_MISMATCH": EXIT_REFUSED,
    "OUT_UNWRITABLE": EXIT_FATAL,
    "GUARD_UNRUNNABLE": EXIT_FATAL,
}
CODE_ORDER = tuple(CODE_EXIT)

_HEX = frozenset("0123456789abcdef")


class BuildError(Exception):
    """One refusal with a code from :data:`CODE_EXIT`."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------------- #
# typed predicates -- the same ones v829_guard.py uses, so a value this tool
# accepts is a value the guard accepts
# --------------------------------------------------------------------------- #
def is_int(value: object) -> bool:
    """True for a real integer; ``True``/``False`` are not integers here."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_text(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def is_digest(value: object) -> bool:
    """True for a bare 64-character lowercase SHA-256 hex digest."""
    return isinstance(value, str) and len(value) == 64 and all(ch in _HEX for ch in value)


def type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


# --------------------------------------------------------------------------- #
# hashing and path resolution
# --------------------------------------------------------------------------- #
def sha256_file(path: pathlib.Path) -> str:
    """SHA-256 of the raw file bytes (no line-ending or text normalisation)."""

    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def looks_absolute(text: str) -> bool:
    """True for ``/x``, ``\\x`` and ``C:\\x`` / ``C:/x`` on either host."""

    if text.startswith(("/", "\\")):
        return True
    return len(text) >= 3 and text[1] == ":" and text[2] in "\\/"


def declared_path(value: str) -> tuple[str, bool]:
    """The spelling to record in the manifest, plus "was it absolute?".

    Repo-relative paths are normalised to ``/`` separators so the same sheet
    produces the same bytes on Windows and in WSL; an absolute path is kept
    verbatim (it names a location on *this* host) and flagged ``absolute``.
    """

    text = value.strip()
    if looks_absolute(text):
        return text, True
    return text.replace("\\", "/"), False


def filesystem_path(value: str) -> pathlib.Path:
    """Resolve a recorded path for reading, without inventing a location.

    ``/mnt/<d>/...`` maps to ``<D>:\\...`` on Windows and ``<D>:\\...`` maps back
    to ``/mnt/<d>/...`` in WSL -- the two spellings of the same file the v8.29
    evidence uses.  Anything this host cannot map stays as-is and is reported as
    missing by the caller rather than silently re-rooted elsewhere.
    """

    text = value.strip()
    windows = sys.platform == "win32"
    if windows and text.startswith("/mnt/") and len(text) > 7:
        head, rest = text[5:6], text[7:]
        if head.isalpha() and rest:
            return pathlib.Path(f"{head.upper()}:\\" + rest.replace("/", "\\"))
    if not windows and len(text) >= 3 and text[1] == ":" and text[2] in "\\/":
        rest = text[2:].lstrip("\\/").replace("\\", "/")
        return pathlib.Path(f"/mnt/{text[0].lower()}/{rest}")
    path = pathlib.Path(text.replace("\\", "/")).expanduser()
    return path if path.is_absolute() else (REPO / path)


def repo_relative(path: pathlib.Path) -> str:
    """A repo-relative POSIX spelling when possible, else an absolute one."""

    try:
        resolved = pathlib.Path(path).resolve()
    except OSError:
        return str(path)
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError:
        return resolved.as_posix()


# --------------------------------------------------------------------------- #
# the tape reader: the row rules of v829_pair.read_tape, ported
# --------------------------------------------------------------------------- #
def scan_tape(path: pathlib.Path) -> dict:
    """Count and validate a 4-frame prefix tape; ``dict`` of the derivable span.

    The four rules are `v829_pair.read_tape`'s (that docstring calls them "the
    four extra refusals that makes an *arbitrary* tape safe"):

    1. exactly :data:`ANCHOR_STRIDE` ``step_frames`` per row;
    2. ``decision_frame == rows_so_far * 4`` -- contiguous 4-frame blocks from 0;
    3. ``hash_pre`` and ``hash_post`` present on every row;
    4. at least one row, and ``frames_built == rows * 4``.

    Rule 3 combined with rule 2 is what makes the recorded anchor set complete
    (``{0, 4, ..., frames_built}``), which is the invariant ``walk_prefix``
    counts hits against.  Returns ``sha256``, ``bytes``, ``rows``,
    ``frames_built`` and ``recorded_hash_frames`` -- the fields this tool derives
    anything from, and the fields the equivalence subcommand compares.
    """

    tape = pathlib.Path(path)
    if not tape.is_file():
        raise BuildError("TAPE_MISSING",
                         f"{tape} is not a readable file on this host, so neither "
                         f"the rows nor the takeover anchor can be checked")
    try:
        digest = sha256_file(tape)
        size = tape.stat().st_size
    except OSError as error:
        raise BuildError("TAPE_MISSING",
                         f"{tape}: cannot read ({error.strerror or error})") from error
    rows = 0
    hashes: dict[int, str] = {}
    try:
        handle = tape.open("r", encoding="utf-8")
    except OSError as error:
        raise BuildError("TAPE_MISSING",
                         f"{tape}: cannot open ({error.strerror or error})") from error
    with handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise BuildError(
                    "TAPE_MALFORMED",
                    f"{tape}: row {index} is not JSON ({error}); the frames it "
                    f"stands for cannot be known") from error
            if not isinstance(row, dict):
                raise BuildError("TAPE_MALFORMED",
                                 f"{tape}: row {index} is not a JSON object "
                                 f"({type_name(row)})")
            frame = row.get("decision_frame")
            if not is_int(frame):
                raise BuildError(
                    "TAPE_MALFORMED",
                    f"{tape}: row {index} has decision_frame {frame!r}, which is "
                    f"not an integer (expected {rows * ANCHOR_STRIDE})")
            if frame != rows * ANCHOR_STRIDE:
                raise BuildError(
                    "TAPE_MALFORMED",
                    f"{tape}: row {index} has decision_frame {frame} but a "
                    f"contiguous 4-frame recording needs {rows * ANCHOR_STRIDE} "
                    f"(every anchor is a multiple of {ANCHOR_STRIDE}; a gap, an "
                    f"overlap or a duplicate is refused, not replayed out of order)")
            steps = row.get("step_frames")
            steps = steps if isinstance(steps, list) else []
            if len(steps) != ANCHOR_STRIDE:
                raise BuildError(
                    "TAPE_MALFORMED",
                    f"{tape}: row {index} (frame {frame}) carries {len(steps)} "
                    f"step_frames, expected {ANCHOR_STRIDE}")
            try:
                [int(value) for value in steps]
            except (TypeError, ValueError) as error:
                raise BuildError(
                    "TAPE_MALFORMED",
                    f"{tape}: row {index} (frame {frame}) has non-integer "
                    f"step_frames {steps!r}: {error}") from error
            if not row.get("hash_pre") or not row.get("hash_post"):
                raise BuildError(
                    "TAPE_MALFORMED",
                    f"{tape}: row {index} (frame {frame}) is missing "
                    f"hash_pre/hash_post; walk_prefix compares hashes at every "
                    f"anchor, so a row without them cannot be walked")
            hashes[frame] = str(row["hash_pre"])
            hashes[frame + ANCHOR_STRIDE] = str(row["hash_post"])
            rows += 1
    frames_built = rows * ANCHOR_STRIDE
    if rows == 0:
        raise BuildError("TAPE_MALFORMED",
                         f"{tape}: 0 rows; an empty recording carries no anchor at "
                         f"all, so no takeover frame can be admitted")
    recorded = len(hashes)
    complete = bool(
        frames_built % ANCHOR_STRIDE == 0
        and recorded == frames_built // ANCHOR_STRIDE + 1
        and all(frame in hashes
                for frame in range(0, frames_built + 1, ANCHOR_STRIDE)))
    if not complete:
        missing = [frame for frame in range(0, frames_built + 1, ANCHOR_STRIDE)
                   if frame not in hashes][:8]
        raise BuildError(
            "TAPE_MALFORMED",
            f"{tape}: the anchor set is incomplete ({recorded} recorded hash "
            f"frames for {frames_built} frames, expected "
            f"{frames_built // ANCHOR_STRIDE + 1}; first missing {missing}); "
            f"read_tape and walk_prefix both refuse an incomplete prefix")
    return {"path": str(tape), "sha256": digest, "bytes": size, "rows": rows,
            "frames_built": frames_built, "recorded_hash_frames": recorded,
            "anchors_complete": complete, "anchor_stride": ANCHOR_STRIDE}


def derived_window(ancestor_id: str) -> int:
    """A stable 6-digit window label derived from ``ancestor_id``.

    Stable across hosts and runs (plain SHA-256 of a namespaced string, no
    ``hash()``), and outside ``(11, 12, 17, 18)`` so a derived label can never be
    mistaken for one of the four frozen S1c windows.
    """

    seed = f"{WINDOW_SALT}|{ancestor_id.strip()}"
    value = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12], 16)
    return WINDOW_LOW + value % WINDOW_SPAN


def rng_slot(source_index: int, ckpt_index: int, rng_index: int = 0) -> int:
    """The ``pair_rng`` of one (source, checkpoint, repeat) cell.

    §1 allocates one seed per (ancestor, budget, checkpoint, repeat), so the seed is
    a function of *which* cell this is, and an index-derived formula is the only
    way to keep it unique without asking the sheet to retype two thirds of the
    allocation.  Two strides keep the layers apart::

        PAIR_RNG_BASE
          + rng_index    * PAIR_RNG_RNG_STRIDE      (1000)
          + source_index * PAIR_RNG_SOURCE_STRIDE   (10)
          + ckpt_index

    ``rng_index`` 0 is the earlier (24-root) allocation, so adding the second §2
    axis does not move one existing seed; repeat 1 starts 1,000 above it, and the
    largest offset inside a layer is ``(sources - 1) * 10 + (ckpts - 1)``, which
    :func:`build_manifest` refuses to let reach :data:`PAIR_RNG_RNG_STRIDE`.  See
    the module docstring (rule 4) for the exact rule.
    """

    return (PAIR_RNG_BASE + rng_index * PAIR_RNG_RNG_STRIDE
            + source_index * PAIR_RNG_SOURCE_STRIDE + ckpt_index)


# --------------------------------------------------------------------------- #
# findings
# --------------------------------------------------------------------------- #
def finding(code: str, detail: str, source_index: int | None = None,
            ancestor_id: str | None = None,
            ckpt_index: int | None = None) -> dict:
    entry = {"code": code, "detail": detail}
    if source_index is not None:
        entry["source_index"] = source_index
    if ancestor_id is not None:
        entry["ancestor_id"] = ancestor_id
    if ckpt_index is not None:
        entry["ckpt_index"] = ckpt_index
    return entry


def sort_findings(findings: list[dict]) -> list[dict]:
    rank = {code: index for index, code in enumerate(CODE_ORDER)}

    def key(entry: dict) -> tuple:
        return (rank.get(entry["code"], len(CODE_ORDER)),
                entry.get("source_index", -1),
                entry.get("ckpt_index", -1),
                str(entry.get("ancestor_id", "")),
                entry["detail"])

    return sorted(findings, key=key)


def codes_of(findings: list[dict]) -> list[str]:
    present = {entry["code"] for entry in findings}
    ordered = [code for code in CODE_ORDER if code in present]
    return ordered + sorted(present.difference(CODE_ORDER))


def exit_for(findings: list[dict]) -> int:
    codes = set(codes_of(findings))
    fatal = any(CODE_EXIT.get(code, EXIT_REFUSED) == EXIT_FATAL for code in codes)
    return EXIT_FATAL if fatal else EXIT_REFUSED


# --------------------------------------------------------------------------- #
# the builder
# --------------------------------------------------------------------------- #
def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """Refuse a repeated JSON key instead of letting ``json.loads`` keep the last.

    The same reasoning as `v829_guard._object_without_duplicate_keys`: a sheet
    saying ``"takeover_frame": 999999`` and later ``"takeover_frame": 2999`` is
    already ambiguous, so no field-level verdict is meaningful.
    """

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"duplicate JSON key {key!r}: one object cannot carry two values "
                f"for the same field, and picking one would be an inference")
        result[key] = value
    return result


def load_sources(path: pathlib.Path) -> tuple[dict, str, int]:
    """Read + digest + parse the sources sheet, refusing ambiguity."""

    try:
        raw = pathlib.Path(path).read_bytes()
    except OSError as error:
        raise BuildError("SOURCES_UNREADABLE",
                         f"cannot read {path}: {error.strerror or error}") from error
    try:
        parsed = json.loads(raw.decode("utf-8-sig"),
                            object_pairs_hook=_object_without_duplicate_keys)
    except (UnicodeDecodeError, ValueError) as error:
        raise BuildError(
            "SOURCES_UNREADABLE",
            f"{path} is not a JSON document this tool can trust: {error}") from error
    if not isinstance(parsed, dict):
        raise BuildError("SOURCES_UNREADABLE",
                         f"{path} must be a JSON object, got {type_name(parsed)}")
    return parsed, hashlib.sha256(raw).hexdigest(), len(raw)


def build_manifest(doc: dict, *, sources_rel: str, sources_sha256: str,
                   sources_bytes: int, window_cli: int | None,
                   budget_cli: int | None, ckpt_cli: pathlib.Path | None,
                   ckpt_sha_cli: str | None, engine_cli: str | None
                   ) -> dict:
    """Derive the manifest.  Returns a build document (see :func:`command_build`)."""

    findings: list[dict] = []
    notes: list[str] = []

    def add(code: str, detail: str, source_index: int | None = None,
            ancestor_id: str | None = None) -> None:
        findings.append(finding(code, detail, source_index, ancestor_id))

    build: dict = {
        "tool": TOOL,
        "version": TOOL_VERSION,
        "sources": {"path": sources_rel, "sha256": sources_sha256,
                    "bytes": sources_bytes},
        "status": "refused",
        "options": {"window": window_cli, "budget": budget_cli,
                    "ckpt": None if ckpt_cli is None else str(ckpt_cli),
                    "ckpt_sha256": ckpt_sha_cli,
                    "engine": engine_cli, "rngs": None},
        "manifest_id": None,
        "engine": None,
        "ckpt": None,
        "rngs": None,
        "budget_frames": None,
        "budget_source": None,
        "roots": [],
        "findings": findings,
        "codes": [],
        "notes": notes,
        "guard": None,
    }

    # ---- 1. the sheet itself ------------------------------------------------
    schema = doc.get("schema")
    if schema != SOURCES_SCHEMA:
        add("SOURCES_INVALID",
            f"schema is {schema!r}; this tool only reads {SOURCES_SCHEMA!r}, "
            f"because every field below is interpreted against that contract")
        return build

    experiment_id = doc.get("experiment_id")
    if not is_text(experiment_id):
        add("SOURCES_INVALID",
            f"experiment_id must be a non-empty string, got "
            f"{type_name(experiment_id)} ({experiment_id!r}); it names the "
            f"manifest and cannot be defaulted")

    data_source = doc.get("data_source")
    if not is_text(data_source):
        add("SOURCES_INVALID",
            f"data_source must be a non-empty string, got {type_name(data_source)} "
            f"({data_source!r})")

    engine_value = engine_cli if engine_cli is not None else doc.get("engine")
    if not is_text(engine_value):
        add("SOURCES_INVALID",
            f"engine must be a non-empty path, got {type_name(engine_value)} "
            f"({engine_value!r}); engine_sha256 is measured from that file")

    budget = budget_cli if budget_cli is not None else doc.get("budget_frames")
    budget_source = "cli" if budget_cli is not None else "sources"
    if not is_int(budget) or budget <= 0:
        add("SOURCES_INVALID",
            f"budget_frames must be a positive integer, got {type_name(budget)} "
            f"({budget!r}); a budget is not a number to be guessed")
    elif budget not in BUDGET_TIERS:
        add("BUDGET_NOT_A_TIER",
            f"budget_frames {budget} is not one of the pre-registered tiers "
            f"{BUDGET_TIERS}; v829_guard.py only proves those two, so a run at "
            f"{budget} frames could not be validated after the fact")
    elif budget_cli is not None and doc.get("budget_frames") not in (None, budget):
        notes.append(
            f"--budget {budget} overrides sources.budget_frames "
            f"{doc.get('budget_frames')!r}; the manifest records {budget} "
            f"(budget_source=cli)")

    # ---- 1a. the repeat count: `rngs` (paired RNGs per cell) ----------------
    # §2's second axis: how many times each (source, checkpoint) cell is measured
    # with a different pair RNG.  Default 1 keeps every earlier sheet the same
    # shape, and above MAX_RNGS there is no pre-registered cell to point at.
    rngs = doc.get("rngs", 1)
    if not is_int(rngs) or not 1 <= rngs <= MAX_RNGS:
        add("SOURCES_INVALID",
            f"'rngs' must be an integer in 1-{MAX_RNGS} (how many paired RNGs each "
            f"(source, checkpoint) cell is measured with), got {type_name(rngs)} "
            f"({rngs!r}); zero repeats would measure nothing and a "
            f"{MAX_RNGS + 1}th is outside the pre-registration")
        rngs = 1
    build["options"]["rngs"] = rngs

    # ---- 1b. the checkpoint(s): `ckpt` (one root per source) XOR `ckpts` (one
    #          root per source *per checkpoint*, all under one ancestor) --------
    ckpt_doc = doc.get("ckpt")
    ckpts_doc = doc.get("ckpts")
    ckpts_mode = ckpts_doc is not None
    # `ckpts` and `rngs` both make this tool the author of the seed allocation, so
    # a source-level `pair_rng` cannot be honoured in either mode.
    allocating = ckpts_mode or rngs > 1
    # Normalised specs, in the order the manifest will emit them.  `pair_index` /
    # `seed` stay None in single-`ckpt` mode (the guard treats "absent" as "one
    # repeat", which is exactly what one checkpoint per source is).
    ckpt_specs: list[dict] = []

    if ckpts_mode and ckpt_doc is not None:
        add("SOURCES_INVALID",
            "'ckpt' and 'ckpts' are both declared: they are mutually exclusive, "
            "because 'ckpt' binds exactly one checkpoint per root while 'ckpts' "
            "expands every source over 1-3 of them, and there is no safe answer to "
            "'which one authors the root'. Keep 'ckpt' for a single-checkpoint "
            f"manifest or 'ckpts' for the section 2 3-checkpoint shape (add 'rngs' "
            f"for the second axis, 1-{MAX_RNGS} paired RNGs per cell)")
    elif ckpts_mode:
        if ckpt_cli is not None or ckpt_sha_cli is not None:
            add("SOURCES_INVALID",
                "--ckpt / --ckpt-sha256 cannot be combined with 'ckpts': they can "
                f"override at most one of the {len(ckpts_doc) if isinstance(ckpts_doc, list) else '1-3'} "
                f"checkpoints the sheet declares, so one of them would silently win")
        elif not isinstance(ckpts_doc, list):
            add("SOURCES_INVALID",
                f"'ckpts' must be an array of 1-{MAX_CKPTS} objects, got "
                f"{type_name(ckpts_doc)} ({ckpts_doc!r})")
        elif not 1 <= len(ckpts_doc) <= MAX_CKPTS:
            add("SOURCES_INVALID",
                f"'ckpts' holds {len(ckpts_doc)} entries; the section 2 axis is "
                f"1-{MAX_CKPTS} checkpoints per ancestor. Zero checkpoints would "
                f"bind no policy, and a fourth is outside the pre-registration")
        else:
            seen_pair_index: dict[int, int] = {}
            for position, entry in enumerate(ckpts_doc):
                label = f"ckpts[{position}]"
                if not isinstance(entry, dict):
                    add("SOURCES_INVALID",
                        f"{label} must be an object with 'path' and 'sha256', got "
                        f"{type_name(entry)} ({entry!r})")
                    continue
                path_text = entry.get("path")
                if not is_text(path_text):
                    add("SOURCES_INVALID",
                        f"{label}.path must be a non-empty string, got "
                        f"{type_name(path_text)} ({path_text!r})")
                    path_text = None
                expected = entry.get("sha256")
                if not is_text(expected):
                    add("SOURCES_INVALID",
                        f"{label}.sha256 is missing: the digest is what makes the "
                        f"checkpoint a registered input, not a filename")
                    expected = None
                elif not is_digest(expected.strip().lower()):
                    add("SOURCES_INVALID",
                        f"{label}.sha256 {expected!r} is not a 64-character hex "
                        f"digest; a truncated digest cannot prove which file is "
                        f"loaded")
                    expected = None
                else:
                    expected = expected.strip()
                if "pair_index" in entry and rngs > 1:
                    add("SOURCES_INVALID",
                        f"{label}.pair_index {entry.get('pair_index')!r} cannot be "
                        f"used with 'rngs' {rngs}: the repeat number is the RNG "
                        f"index 0..{rngs - 1} inside each (source, checkpoint) "
                        f"cell, so a per-checkpoint repeat number could address "
                        f"only one of the {rngs} repeats of every checkpoint")
                    pair_index = None
                else:
                    pair_index = entry.get("pair_index", position)
                    if not is_int(pair_index) or pair_index < 0:
                        add("SOURCES_INVALID",
                            f"{label}.pair_index must be a non-negative integer, got "
                            f"{type_name(pair_index)} ({pair_index!r})")
                        pair_index = None
                    elif pair_index in seen_pair_index:
                        add("SOURCES_INVALID",
                            f"{label}.pair_index {pair_index} repeats "
                            f"ckpts[{seen_pair_index[pair_index]}]; a repeat number "
                            f"names one cell of this ancestor, so two checkpoints "
                            f"cannot share it")
                        pair_index = None
                    else:
                        seen_pair_index[pair_index] = position
                seed = entry.get("seed")
                if seed is not None and (not is_int(seed) or seed < 0):
                    add("SOURCES_INVALID",
                        f"{label}.seed must be a non-negative integer when present, "
                        f"got {type_name(seed)} ({seed!r})")
                    seed = None
                ckpt_specs.append({"label": label, "path": path_text,
                                   "expected": expected, "pair_index": pair_index,
                                   "seed": seed})
    else:
        ckpt_path_text = None
        ckpt_expected = None
        if isinstance(ckpt_doc, dict):
            if is_text(ckpt_doc.get("path")):
                ckpt_path_text = ckpt_doc["path"]
            if is_text(ckpt_doc.get("sha256")):
                ckpt_expected = ckpt_doc["sha256"].strip()
            declared_ckpt_sha = ckpt_doc.get("sha256")
            if declared_ckpt_sha is not None and not is_text(declared_ckpt_sha):
                add("SOURCES_INVALID",
                    f"ckpt.sha256 must be a 64-character lowercase hex string, got "
                    f"{type_name(declared_ckpt_sha)} ({declared_ckpt_sha!r})")
        elif ckpt_doc is not None:
            add("SOURCES_INVALID",
                f"ckpt must be an object with 'path' and 'sha256', got "
                f"{type_name(ckpt_doc)}")
        if ckpt_cli is not None:
            ckpt_path_text = str(ckpt_cli)
        if ckpt_sha_cli is not None:
            ckpt_expected = ckpt_sha_cli.strip()
        if ckpt_path_text is None:
            add("SOURCES_INVALID",
                "ckpt.path is missing: the manifest binds one checkpoint per root "
                f"and cannot be built without it (use --ckpt to point at a copy, or "
                f"the 'ckpts' array for 1-{MAX_CKPTS} of them)")
        if ckpt_expected is None:
            add("SOURCES_INVALID",
                "ckpt.sha256 is missing: the digest is what makes the checkpoint a "
                "registered input (use --ckpt-sha256 to supply it)")
        elif not is_digest(ckpt_expected.lower()):
            add("SOURCES_INVALID",
                f"ckpt.sha256 {ckpt_expected!r} is not a 64-character hex digest; "
                f"a truncated digest cannot prove which file is loaded")
        ckpt_specs.append({"label": "ckpt", "path": ckpt_path_text,
                           "expected": ckpt_expected, "pair_index": None,
                           "seed": None})

    sources = doc.get("sources")
    if not isinstance(sources, list) or not sources:
        add("SOURCES_INVALID",
            f"sources must be a non-empty array, got {type_name(sources)} "
            f"({sources!r}); a manifest with no root admits no window")
    elif rngs > 1 and ckpt_specs:
        # One repeat layer is PAIR_RNG_RNG_STRIDE wide, so the (source, checkpoint)
        # offsets have to fit inside it; otherwise two repeats could collide and
        # "unique by construction" would be a claim, not a fact.
        span = ((len(sources) - 1) * PAIR_RNG_SOURCE_STRIDE
                + (len(ckpt_specs) - 1))
        if span >= PAIR_RNG_RNG_STRIDE:
            add("SOURCES_INVALID",
                f"{len(sources)} sources x {len(ckpt_specs)} checkpoint(s) span "
                f"{span} seed offsets, but one repeat layer is only "
                f"{PAIR_RNG_RNG_STRIDE} wide (PAIR_RNG_RNG_STRIDE); the allocation "
                f"of rngs {rngs} would overlap the next repeat's layer, so the "
                f"manifest could not claim unique seeds")
    if findings:
        return build

    # ---- 2. the two measured assets ----------------------------------------
    engine_declared = str(engine_value).strip()
    engine_fs = filesystem_path(engine_declared)
    engine_sha = None
    engine_bytes = None
    if not engine_fs.is_file():
        add("ENGINE_MISSING",
            f"engine {engine_declared} -> {engine_fs} is not a readable file on "
            f"this host, so engine_sha256 cannot be measured; refusing to copy a "
            f"declared digest instead of measuring one")
    else:
        try:
            engine_sha = sha256_file(engine_fs)
            engine_bytes = engine_fs.stat().st_size
        except OSError as error:
            add("ENGINE_MISSING",
                f"engine {engine_fs}: cannot read ({error.strerror or error})")
            engine_bytes = None
    declared_engine_sha = doc.get("engine_sha256")
    if declared_engine_sha is not None and not is_text(declared_engine_sha):
        add("SOURCES_INVALID",
            f"engine_sha256 must be a string when present, got "
            f"{type_name(declared_engine_sha)} ({declared_engine_sha!r})")
    elif is_text(declared_engine_sha) and engine_sha is not None:
        claimed = declared_engine_sha.strip().lower()
        if claimed != engine_sha:
            add("ENGINE_SHA_MISMATCH",
                f"engine_sha256 {claimed[:16]}... was declared but "
                f"sha256({engine_fs}) is {engine_sha[:16]}...; the sheet names a "
                f"different binary than the one on disk")
        elif declared_engine_sha.strip() != engine_sha:
            notes.append("engine_sha256 in the sheet was upper/mixed case; it "
                         "matched and the manifest records the measured lowercase "
                         "digest")

    measured: list[dict] = []
    for position, spec in enumerate(ckpt_specs):
        declared, is_absolute = declared_path(spec["path"])
        fs = filesystem_path(declared)
        sha = None
        nbytes = None
        if not fs.is_file():
            add("CKPT_MISSING",
                f"checkpoint {declared} -> {fs} is not a readable file on this "
                f"host; CKPT_SHA_MISMATCH cannot be judged on a missing file",
                ckpt_index=position if ckpts_mode else None)
        else:
            try:
                sha = sha256_file(fs)
                nbytes = fs.stat().st_size
            except OSError as error:
                add("CKPT_MISSING",
                    f"checkpoint {fs}: cannot read ({error.strerror or error})",
                    ckpt_index=position if ckpts_mode else None)
                nbytes = None
        expected = spec["expected"]
        if sha is not None and expected is not None and is_digest(expected.lower()):
            if sha != expected.lower():
                add("CKPT_SHA_MISMATCH",
                    f"sha256({declared}) is {sha} ({nbytes} B) but "
                    f"{spec['label']}.sha256 is {expected.lower()}; the manifest "
                    f"must register the checkpoint that is actually loaded",
                    ckpt_index=position if ckpts_mode else None)
            elif expected != expected.lower():
                notes.append(f"{spec['label']}.sha256 in the sheet was upper/mixed "
                             f"case; it matched and the manifest records the "
                             f"measured lowercase digest")
        measured.append({"label": spec["label"], "declared": declared,
                         "absolute": is_absolute, "path": str(fs), "sha256": sha,
                         "bytes": nbytes,
                         "expected": None if expected is None else expected.lower(),
                         "pair_index": spec["pair_index"], "seed": spec["seed"],
                         "source": "measured"})
    if findings:
        return build

    build["engine"] = {"declared": engine_declared, "path": str(engine_fs),
                       "sha256": engine_sha, "bytes": engine_bytes,
                       "source": "measured"}
    build["ckpts"] = measured
    # Single-`ckpt` mode keeps the v8.29.0 report shape (`build["ckpt"]` is the one
    # measured checkpoint); `ckpts` mode reports the list and leaves `ckpt` null.
    build["ckpt"] = None if ckpts_mode else measured[0]
    build["budget_frames"] = budget
    build["budget_source"] = budget_source

    # ---- 3. per-source derivation ------------------------------------------
    roots: list[dict] = []
    for index, source in enumerate(sources):
        label = f"sources[{index}]"
        if not isinstance(source, dict):
            add("SOURCES_INVALID",
                f"{label} must be a JSON object, got {type_name(source)}",
                index)
            continue
        ancestor = source.get("ancestor_id")
        ancestor_label = ancestor.strip() if is_text(ancestor) else f"<{label}>"
        usable = True

        for name in SOURCE_TEXT_FIELDS:
            value = source.get(name)
            if not is_text(value):
                add("SOURCES_INVALID",
                    f"{label}.{name} must be a non-empty string, got "
                    f"{type_name(value)} ({value!r})", index, ancestor_label)
                usable = False

        for name in EXPOSURE_FIELDS:
            if name not in source:
                add("EXPOSURE_REQUIRED",
                    f"{label}.{name} is missing; every source must declare both "
                    f"{' and '.join(EXPOSURE_FIELDS)} as booleans, because an "
                    f"undeclared exposure cannot be ruled out and this sheet is "
                    f"the only place it is stated", index, ancestor_label)
                usable = False
            elif not isinstance(source[name], bool):
                add("EXPOSURE_REQUIRED",
                    f"{label}.{name} is {type_name(source[name])} "
                    f"({source[name]!r}); it must be a JSON boolean so that 'no "
                    f"exposure' is a positive claim", index, ancestor_label)
                usable = False
            elif source[name] is True:
                notes.append(f"{label} ({ancestor_label}) declares {name}=true: "
                             f"building is allowed, but v829_guard.py refuses the "
                             f"manifest without --allow-exposed")

        rng = source.get("pair_rng")
        if allocating:
            # When this tool owns the allocation (`ckpts` or `rngs > 1`) the seed is
            # derived per (source, checkpoint, repeat) from the indices: one number
            # on the row cannot address several cells, and choosing one of them for
            # the row would be an inference.
            if "pair_rng" in source:
                first = rng_slot(index, 0, 0)
                last = (rng_slot(index, len(ckpt_specs) - 1, 0) if ckpts_mode
                        else rng_slot(index, 0, rngs - 1))
                cells = (f"{len(ckpt_specs)} checkpoint(s) x {rngs} repeat(s)"
                         if ckpts_mode else f"1 checkpoint x {rngs} repeat(s)")
                add("SOURCES_INVALID",
                    f"{label}.pair_rng {source.get('pair_rng')!r} cannot be used: "
                    f"this sheet declares "
                    f"{'ckpts' if ckpts_mode else 'a single ckpt'} and rngs {rngs}, "
                    f"so every cell names its own seed and the manifest allocates "
                    f"PAIR_RNG_BASE + rng_index * {PAIR_RNG_RNG_STRIDE} + "
                    f"source_index * {PAIR_RNG_SOURCE_STRIDE} + ckpt_index = "
                    f"{first}..{last} for this source. One seed on the row could "
                    f"only cover one of its {cells}", index, ancestor_label)
                usable = False
            rng = None
        elif not is_int(rng) or rng < 0:
            add("SOURCES_INVALID",
                f"{label}.pair_rng must be a non-negative integer, got "
                f"{type_name(rng)} ({rng!r}); one seed per (ancestor, budget, "
                f"checkpoint, repeat) is pre-allocated", index, ancestor_label)
            usable = False

        takeover = source.get("takeover_frame")
        if not is_int(takeover) or takeover < 0:
            add("SOURCES_INVALID",
                f"{label}.takeover_frame must be a non-negative integer, got "
                f"{type_name(takeover)} ({takeover!r})", index, ancestor_label)
            usable = False
        elif takeover % ANCHOR_STRIDE != TAKEOVER_MOD:
            lower = (takeover // ANCHOR_STRIDE) * ANCHOR_STRIDE - 1
            upper = lower + ANCHOR_STRIDE
            add("TAKEOVER_NOT_ANCHOR",
                f"{label}.takeover_frame {takeover}: {takeover} % "
                f"{ANCHOR_STRIDE} = {takeover % ANCHOR_STRIDE}, but the runner "
                f"walks target = N + 1 and the tape records an anchor every "
                f"{ANCHOR_STRIDE} frames, so N must be 4k - 1 (nearest legal: "
                f"{lower} or {upper}). {takeover + 1} is not a recorded anchor",
                index, ancestor_label)
            usable = False

        lineage_new = source.get("lineage_new")
        if lineage_new is not None and not isinstance(lineage_new, bool):
            add("SOURCES_INVALID",
                f"{label}.lineage_new must be a JSON boolean, got "
                f"{type_name(lineage_new)} ({lineage_new!r})", index, ancestor_label)
            usable = False
        lineage_ids = source.get("lineage_ancestor_ids")
        if lineage_ids is not None:
            if not isinstance(lineage_ids, list) or not all(
                    is_text(item) for item in lineage_ids):
                add("SOURCES_INVALID",
                    f"{label}.lineage_ancestor_ids must be an array of non-empty "
                    f"strings, got {type_name(lineage_ids)} ({lineage_ids!r})",
                    index, ancestor_label)
                usable = False

        declared_window = source.get("window")
        window = None
        window_source = None
        if declared_window is not None:
            if not is_int(declared_window) or declared_window < 0:
                add("SOURCES_INVALID",
                    f"{label}.window must be a non-negative integer, got "
                    f"{type_name(declared_window)} ({declared_window!r})",
                    index, ancestor_label)
                usable = False
            elif window_cli is not None and int(window_cli) != declared_window:
                add("SOURCES_INVALID",
                    f"{label}.window {declared_window} contradicts --window "
                    f"{window_cli}: the window labels the root and names its "
                    f"private data dir, so a conflict cannot be resolved by "
                    f"preference", index, ancestor_label)
                usable = False
            else:
                window, window_source = declared_window, "source"
        elif window_cli is not None:
            window, window_source = int(window_cli), "cli"
        elif is_text(ancestor):
            window, window_source = derived_window(ancestor), "derived"

        if not usable:
            continue

        tape_declared, tape_absolute = declared_path(source["tape"])
        tape_fs = filesystem_path(tape_declared)
        try:
            meta = scan_tape(tape_fs)
        except BuildError as error:
            add(error.code, f"{label}: {error.detail}", index, ancestor_label)
            continue
        frames_built = meta["frames_built"]
        checkpoints = takeover // ANCHOR_STRIDE
        if takeover + 1 > frames_built:
            add("TAKEOVER_BEYOND_TAPE",
                f"{label}: takeover_frame {takeover} needs target = N + 1 = "
                f"{takeover + 1} to be a recorded anchor, but the tape holds "
                f"frames_built = {frames_built} ({meta['rows']} rows x "
                f"{ANCHOR_STRIDE}); the last admissible takeover frame is "
                f"{frames_built - 1}",
                index, ancestor_label)
            continue
        if takeover + budget > frames_built:
            add("BUDGET_EXCEEDS_TOTAL",
                f"{label}: takeover_frame {takeover} + budget_frames {budget} = "
                f"{takeover + budget} > total_frames {frames_built} (the tape's "
                f"frames_built), short by {takeover + budget - frames_built} "
                f"frames", index, ancestor_label)
            continue

        total_source = (
            f"total_frames {frames_built} = prefix tape rows {meta['rows']} x "
            f"{ANCHOR_STRIDE} frames (frames_built counted by {TOOL} from "
            f"{tape_declared}, sha256 {meta['sha256'][:16]}...); takeover "
            f"{takeover} + budget {budget} = {takeover + budget} <= "
            f"{frames_built}, headroom {frames_built - takeover - budget} frames. "
            f"The tape is the recorded span, not a hard end: the live engine keeps "
            f"running past the recording.")
        anchor_note = (
            f"N = {takeover} = 4 x {checkpoints} + {takeover % ANCHOR_STRIDE}, so "
            f"target = N + 1 = {takeover + 1} is a recorded "
            f"{ANCHOR_STRIDE}-frame anchor and expected checkpoints = N // "
            f"{ANCHOR_STRIDE} = {checkpoints}. v431_dense_driver.build_masks "
            f"refuses a target with no hash, which is why an admissible takeover "
            f"frame is 4k - 1. All three digest fields are this tape's own sha256 "
            f"({meta['bytes']} B) over the raw file bytes.")
        # One root per checkpoint -- and, with `rngs > 1`, one more per paired RNG.
        # Everything above this line describes the play-root and is therefore shared
        # by every root of this source; only the checkpoint, the repeat number, the
        # seed and the checkpoint's own seed change between them.
        for ckpt_position, ckpt_item in enumerate(measured):
            for rng_index in range(rngs):
                if rngs > 1:
                    cell = (f"{label}/ckpts[{ckpt_position}]/rng[{rng_index}]"
                            if ckpts_mode else f"{label}/rng[{rng_index}]")
                else:
                    cell = (f"{label}/ckpts[{ckpt_position}]" if ckpts_mode else label)
                cell_rng = (rng_slot(index, ckpt_position, rng_index)
                            if allocating else rng)
                root = {
                    "ancestor_id": ancestor.strip(),
                    "stage": source["stage"].strip(),
                    "window": window,
                    "window_source": window_source,
                    "budget_frames": budget,
                    "ckpt_sha256": ckpt_item["sha256"],
                    "ckpt_path": ckpt_item["declared"],
                    "engine_sha256": engine_sha,
                    "pair_rng": cell_rng,
                    "takeover_frame": takeover,
                    "total_frames": frames_built,
                    "total_frames_source": total_source,
                    "parent_prefix_digest": meta["sha256"],
                    "prefix_tape": tape_declared,
                    "prefix_tape_sha256": meta["sha256"],
                    "prefix_tape_rows": meta["rows"],
                    "prefix_tape_frames": frames_built,
                    "verify_anchor": {
                        "prefix_digest": meta["sha256"],
                        "prefix_record": tape_declared,
                        "window_recording_sha256": meta["sha256"],
                        "recording_bytes": meta["bytes"],
                        "recorded_frames": takeover,
                        "hash_checkpoints": checkpoints,
                        "note": anchor_note,
                    },
                    "used_for_training": source["used_for_training"],
                    "used_for_selection": source["used_for_selection"],
                }
                if ckpts_mode or rngs > 1:
                    # The repeat number the guard keys the cell on.  With `rngs > 1`
                    # it is the RNG index inside the (source, checkpoint) cell; with
                    # a single repeat it keeps the earlier per-checkpoint meaning.
                    # `ckpt_seed` is the seed that produced the checkpoint.
                    if rngs > 1:
                        root["pair_index"] = rng_index
                    else:
                        root["pair_index"] = (ckpt_position
                                              if ckpt_item["pair_index"] is None
                                              else ckpt_item["pair_index"])
                    if ckpt_item["seed"] is not None:
                        root["ckpt_seed"] = ckpt_item["seed"]
                if window_source == "derived":
                    root["window_derived"] = True
                if tape_absolute:
                    root["prefix_tape_absolute"] = True
                    root["verify_anchor"]["prefix_record_absolute"] = True
                if lineage_new is not None:
                    root["lineage_new"] = lineage_new
                if lineage_ids is not None:
                    root["lineage_ancestor_ids"] = list(lineage_ids)
                roots.append({
                    "index": index,
                    "ckpt_index": ckpt_position if ckpts_mode else None,
                    "rng_index": rng_index if rngs > 1 else None,
                    "cell": cell,
                    "record": root,
                    "window": window,
                    "window_source": window_source,
                    "tape_sha256": meta["sha256"],
                    "takeover_frame": takeover,
                    "budget_frames": budget,
                    "total_frames": frames_built,
                    "hash_checkpoints": checkpoints,
                    "pair_rng": cell_rng,
                    "pair_index": root.get("pair_index"),
                    "ckpt_sha256": ckpt_item["sha256"],
                    "ancestor_id": root["ancestor_id"],
                })

    # ---- 4. cross-root gates ------------------------------------------------
    # One ancestor owns several roots -- one per checkpoint, and one more per paired
    # RNG -- so the duplicate key is the cell (ancestor, checkpoint, repeat), not
    # the label.  A repeat is the same cell twice.
    def cell_repeat(entry: dict) -> int:
        """The effective repeat number: an absent ``pair_index`` means repeat 0."""
        return 0 if entry["pair_index"] is None else entry["pair_index"]

    by_ancestor: dict[tuple, list[str]] = {}
    for entry in roots:
        by_ancestor.setdefault((entry["ancestor_id"], entry["ckpt_sha256"],
                                cell_repeat(entry)), []).append(entry["cell"])
    for key in sorted(by_ancestor):
        cells = by_ancestor[key]
        if len(cells) > 1:
            add("SOURCES_INVALID",
                f"ancestor_id {key[0]!r} with ckpt {str(key[1])[:12]}... at repeat "
                f"pair_index {key[2]} appears {len(cells)} times ({cells}); one cell "
                f"is one measurement, and a repeat is the same cell counted twice. "
                f"A *different* checkpoint or a *different* pair_index under this "
                f"ancestor is a section 2 repeat axis, not a second source")

    rng_scope: dict[tuple, list[tuple[str, object]]] = {}
    for entry in roots:
        key = (entry["ancestor_id"], budget, entry["ckpt_sha256"])
        rng_scope.setdefault(key, []).append((entry["cell"], entry["pair_rng"]))
    for key in sorted(rng_scope, key=repr):
        grouped: dict[str, tuple[object, list[str]]] = {}
        for cell, value in rng_scope[key]:
            slot = grouped.get(repr(value))
            if slot is None:
                slot = (value, [])
                grouped[repr(value)] = slot
            slot[1].append(cell)
        for marker in sorted(grouped):
            value, cells = grouped[marker]
            if len(cells) < 2:
                continue
            add("RNG_DUPLICATE",
                f"{cells} share pair_rng {value!r} under ancestor "
                f"{key[0]!r}, budget {budget}, ckpt {str(key[2])[:12]}...; one seed "
                f"per (ancestor, budget, checkpoint, repeat) is pre-allocated, so "
                f"a repeat silently correlates two cells")

    # The allocation is unique by construction, but "by construction" is exactly the
    # claim that should be verified: in `ckpts` / `rngs` mode every seed this tool
    # hands out has to be distinct across the WHOLE manifest, not just inside one
    # (ancestor, budget, checkpoint) family.
    if allocating:
        seen_rng: dict[int, list[str]] = {}
        for entry in roots:
            seen_rng.setdefault(entry["pair_rng"], []).append(entry["cell"])
        for value in sorted(seen_rng):
            cells = seen_rng[value]
            if len(cells) > 1:
                add("RNG_DUPLICATE",
                    f"{cells} were allocated the same pair_rng {value}; the "
                    f"allocation (PAIR_RNG_BASE + rng_index * {PAIR_RNG_RNG_STRIDE} "
                    f"+ source_index * {PAIR_RNG_SOURCE_STRIDE} + ckpt_index) is "
                    f"unique by construction, so a collision means the sheet "
                    f"exhausted a repeat layer and two cells overlapped")

    # The checkpoint and the repeat number are part of the window identity: the
    # three checkpoints of one ancestor are three *different* physical windows over
    # the same tape, and the two paired RNGs of one cell are two different
    # measurements of that window (the guard's window key carries both too).
    window_scope: dict[tuple, list[str]] = {}
    label_of = {entry["cell"]: entry["ancestor_id"] for entry in roots}
    for entry in roots:
        key = (entry["ckpt_sha256"], entry["tape_sha256"], entry["takeover_frame"],
               entry["budget_frames"], cell_repeat(entry))
        window_scope.setdefault(key, []).append(entry["cell"])
    for key in sorted(window_scope, key=repr):
        cells = window_scope[key]
        if len(cells) < 2:
            continue
        labels = sorted({label_of[cell] for cell in cells})
        add("WINDOW_DUPLICATE",
            f"{cells} derive the same window identity (ckpt {str(key[0])[:12]}..., "
            f"tape {key[1][:12]}..., takeover_frame {key[2]}, budget_frames "
            f"{key[3]}, pair_index {key[4]}) under labels {labels}; a different "
            f"ancestor_id does not make it a second window, and section 4 would "
            f"average the same window twice")

    # ---- 5. assemble --------------------------------------------------------
    if findings:
        return build

    windows: dict[int, list[str]] = {}
    for entry in roots:
        windows.setdefault(entry["window"], []).append(entry["ancestor_id"])
    for window in sorted(windows):
        if len(windows[window]) > 1:
            notes.append(f"window {window} labels {len(windows[window])} roots "
                         f"({sorted(windows[window])}); that is legal (the window "
                         f"only selects roots and names data dirs), but "
                         f"`v829_pair.py --window {window}` will select all of "
                         f"them at once -- with 'ckpts' that is every checkpoint of "
                         f"that ancestor in one invocation, which is the section 2 "
                         f"shape")

    purpose = doc.get("purpose")
    if not is_text(purpose):
        if ckpts_mode and rngs == 1:
            # The v8.29.1 wording, kept byte for byte so a rebuild of a 24-root
            # sheet is identical to the manifest already on disk.
            checkpoint_text = (
                f"{len(measured)} checkpoints ("
                + ", ".join(str(item["sha256"])[:8] + "..." for item in measured)
                + f") across {len(sources)} ancestor(s); every root of one source "
                  f"shares that ancestor_id and differs only in its checkpoint, its "
                  f"seed and its pair_index/ckpt_seed")
        elif ckpts_mode:
            checkpoint_text = (
                f"{len(measured)} checkpoints ("
                + ", ".join(str(item["sha256"])[:8] + "..." for item in measured)
                + f") across {len(sources)} ancestor(s) measured at {rngs} paired "
                  f"RNG(s) each; every root of one source shares that ancestor_id "
                  f"and differs only in its checkpoint, its repeat number "
                  f"pair_index, its seed and its ckpt_seed")
        elif rngs > 1:
            checkpoint_text = (
                f"the checkpoint is {str(measured[0]['sha256'])[:16]}... as measured "
                f"on disk, measured {rngs} times per source with {rngs} paired RNGs")
        else:
            checkpoint_text = (
                f"the checkpoint is {str(measured[0]['sha256'])[:16]}... as measured "
                f"on disk")
        purpose = (
            f"{len(roots)} root(s) built from {sources_rel} (experiment "
            f"{experiment_id}): each root replays its own 4-frame prefix tape up "
            f"to a 4k - 1 takeover frame, then runs {budget} frames per arm; "
            f"{checkpoint_text}.")
    manifest_id = doc.get("manifest_id")
    if not is_text(manifest_id):
        manifest_id = f"{experiment_id}-manifest"

    builder_doc = {
        "tool": TOOL,
        "version": TOOL_VERSION,
        "sources_schema": SOURCES_SCHEMA,
        "sources_path": sources_rel,
        "sources_sha256": sources_sha256,
        "sources_bytes": sources_bytes,
        "experiment_id": experiment_id,
        "budget_frames": budget,
        "budget_source": budget_source,
        "note": ("every derived field comes from the tape / checkpoint / "
                 "engine bytes on disk; there is no generated_at, so the same "
                 "sources file yields this file byte for byte"),
    }
    if rngs > 1:
        # Only recorded when the second §2 axis is actually used, so a single-RNG
        # manifest keeps the shape (and the bytes) it had before `rngs` existed.
        builder_doc["rngs"] = rngs

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "manifest_id": manifest_id,
        "purpose": purpose,
        "engine": engine_declared,
        "data_source": data_source,
        "builder": builder_doc,
        "roots": [entry["record"] for entry in roots],
    }
    build["manifest"] = manifest
    build["manifest_id"] = manifest_id
    build["status"] = "built"
    build["rngs"] = rngs
    build["roots"] = [
        {"source_index": entry["index"],
         "ckpt_index": entry["ckpt_index"],
         "rng_index": entry["rng_index"],
         "cell": entry["cell"],
         "ancestor_id": entry["ancestor_id"],
         "window": entry["window"],
         "window_source": entry["window_source"],
         "tape": entry["record"]["prefix_tape"],
         "tape_sha256": entry["tape_sha256"],
         "tape_rows": entry["record"]["prefix_tape_rows"],
         "frames_built": entry["total_frames"],
         "takeover_frame": entry["takeover_frame"],
         "hash_checkpoints": entry["hash_checkpoints"],
         "budget_frames": entry["budget_frames"],
         "total_frames": entry["total_frames"],
         "headroom_frames": (entry["total_frames"] - entry["takeover_frame"]
                             - entry["budget_frames"]),
         "pair_rng": entry["pair_rng"],
         "pair_index": entry["pair_index"],
         "ckpt_sha256": entry["ckpt_sha256"],
         "ckpt_path": entry["record"]["ckpt_path"],
         "parent_prefix_digest": entry["tape_sha256"]}
        for entry in roots
    ]
    return build


# --------------------------------------------------------------------------- #
# writing, then the mandatory guard run
# --------------------------------------------------------------------------- #
def write_manifest(path: pathlib.Path, manifest: dict) -> tuple[str, int]:
    """Serialise deterministically and write UTF-8 bytes with LF endings."""

    text = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    data = text.encode("utf-8")
    try:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_bytes(data)
    except OSError as error:
        raise BuildError("OUT_UNWRITABLE",
                         f"cannot write {path}: {error.strerror or error}") from error
    return hashlib.sha256(data).hexdigest(), len(data)


def run_guard(manifest_path: pathlib.Path, *, budget: int, allow_exposed: bool,
              guard_path: pathlib.Path) -> dict:
    """Run ``v829_guard.py check`` in a subprocess of *this* interpreter."""

    argv = [sys.executable, "-B", str(guard_path), "check", str(manifest_path),
            "--budget", str(budget)]
    if allow_exposed:
        argv.append("--allow-exposed")
    entry = {"interpreter": sys.executable, "interpreter_version": sys.version.split()[0],
             "argv": argv, "exit": None, "stdout": "", "stderr": "",
             "verdict": "UNRUNNABLE"}
    try:
        completed = subprocess.run(argv, capture_output=True, text=True)
    except OSError as error:
        entry["stderr"] = f"{type(error).__name__}: {error}"
        entry["error_code"] = "GUARD_UNRUNNABLE"
        return entry
    entry["exit"] = completed.returncode
    entry["stdout"] = completed.stdout
    entry["stderr"] = completed.stderr
    entry["verdict"] = ("PASS" if completed.returncode == 0
                        else "REFUSED" if completed.returncode == 1 else "UNEVALUABLE")
    return entry


def print_guard(entry: dict) -> None:
    print(f"=== guard: exit {entry['exit']} ({entry['verdict']}) ===")
    print(f"    interpreter {entry['interpreter_version']} {entry['interpreter']}")
    print("    argv " + " ".join(entry["argv"]))
    if entry["stdout"]:
        print(entry["stdout"].rstrip("\n"))
    if entry["stderr"]:
        print("--- guard stderr ---")
        print(entry["stderr"].rstrip("\n"))


# --------------------------------------------------------------------------- #
# the equivalence check against v829_pair.read_tape
# --------------------------------------------------------------------------- #
def load_pair_module():
    """Import ``tools/rl/v829_pair.py`` by path (only the equivalence command)."""

    path = REPO / "tools/rl/v829_pair.py"
    if not path.is_file():
        raise RuntimeError(f"{path} is missing; there is nothing to compare to")
    spec = importlib.util.spec_from_file_location("v829_pair_equivalence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command_equivalence(args: argparse.Namespace) -> int:
    """Print this tool's tape fields next to ``v829_pair.read_tape``'s."""

    takeover = int(args.takeover)
    if takeover < 0 or takeover % ANCHOR_STRIDE != TAKEOVER_MOD:
        print(f"equivalence: takeover {takeover} is not 4k - 1; read_tape needs "
              f"target N + 1 to be a {ANCHOR_STRIDE}-frame anchor", file=sys.stderr)
        return EXIT_REFUSED
    print(f"interpreter: {sys.executable} ({sys.version.split()[0]})")
    print(f"takeover_frame {takeover} -> target {takeover + 1} "
          f"(expected checkpoints N // {ANCHOR_STRIDE} = "
          f"{takeover // ANCHOR_STRIDE})")
    try:
        pair = load_pair_module()
    except Exception as error:                          # noqa: BLE001 -- report it
        print(f"equivalence: cannot load tools/rl/v829_pair.py: {error}",
              file=sys.stderr)
        return EXIT_FATAL
    failures = 0
    for tape in args.tape:
        path = filesystem_path(str(tape))
        print("")
        print(f"tape {path}")
        try:
            masks, hashes, meta = pair.read_tape(str(path), takeover + 1)
        except Exception as error:                      # noqa: BLE001 -- report it
            code = getattr(error, "code", type(error).__name__)
            print(f"  v829_pair.read_tape refused: {code}: {error}", file=sys.stderr)
            failures += 1
            continue
        try:
            mine = scan_tape(path)
        except BuildError as error:
            print(f"  this tool refused: {error.code}: {error.detail}",
                  file=sys.stderr)
            failures += 1
            continue
        theirs = {
            "sha256": meta["sha256"],
            "rows": meta["rows"],
            "frames_built": meta["frames_built"],
            "hash_checkpoints": (int(meta["target"]) - 1) // ANCHOR_STRIDE,
        }
        ours = {
            "sha256": mine["sha256"],
            "rows": mine["rows"],
            "frames_built": mine["frames_built"],
            "hash_checkpoints": takeover // ANCHOR_STRIDE,
        }
        tape_failures = 0
        print(f"  {'field':18s} {'this tool':>66s}  {'v829_pair.read_tape':>66s}  equal")
        for field in ("sha256", "rows", "frames_built", "hash_checkpoints"):
            same = ours[field] == theirs[field]
            tape_failures += 0 if same else 1
            print(f"  {field:18s} {str(ours[field]):>66s}  "
                  f"{str(theirs[field]):>66s}  {same}")
        extra = {
            "recorded_hash_frames": (mine["recorded_hash_frames"],
                                     meta["recorded_hash_frames"]),
            "anchors_complete": (mine["anchors_complete"],
                                 meta["anchors_complete"]),
            "bytes": (mine["bytes"], meta["bytes"]),
            "masks_len": (len(masks), mine["frames_built"]),
            "hashes_len": (len(hashes), mine["recorded_hash_frames"]),
            "target": (takeover + 1, meta["target"]),
        }
        for field in ("recorded_hash_frames", "anchors_complete", "bytes",
                      "masks_len", "hashes_len", "target"):
            left, right = extra[field]
            same = left == right
            tape_failures += 0 if same else 1
            print(f"  {field:18s} {str(left):>66s}  {str(right):>66s}  {same}")
        print(f"  => {'IDENTICAL' if tape_failures == 0 else 'MISMATCH'}")
        failures += tape_failures
    print("")
    print(f"EQUIVALENCE: {'ok' if failures == 0 else 'failed'} "
          f"({len(args.tape)} tape(s), {failures} disagreement(s))")
    return EXIT_OK if failures == 0 else EXIT_REFUSED


# --------------------------------------------------------------------------- #
# the build command
# --------------------------------------------------------------------------- #
def command_build(args: argparse.Namespace) -> int:
    sources_path = pathlib.Path(args.sources)
    out_path = pathlib.Path(args.out)
    guard_path = (pathlib.Path(args.guard) if args.guard is not None
                  else (REPO / DEFAULT_GUARD))
    sources_rel = repo_relative(sources_path)
    report: dict = {}
    try:
        doc, sources_sha, sources_bytes = load_sources(sources_path)
    except BuildError as error:
        print(f"REFUSED [{error.code}] {error.detail}", file=sys.stderr)
        report = {
            "tool": TOOL, "version": TOOL_VERSION,
            "sources": {"path": sources_rel},
            "status": "refused",
            "findings": [finding(error.code, error.detail)],
            "codes": [error.code],
            "guard": None,
        }
        if args.report is not None:
            write_report(args.report, report)
        return CODE_EXIT.get(error.code, EXIT_REFUSED)

    build = build_manifest(
        doc, sources_rel=sources_rel, sources_sha256=sources_sha,
        sources_bytes=sources_bytes, window_cli=args.window,
        budget_cli=args.budget, ckpt_cli=args.ckpt,
        ckpt_sha_cli=args.ckpt_sha256, engine_cli=args.engine)
    findings = build["findings"]
    report.update(build)
    report["options"]["allow_exposed"] = bool(args.allow_exposed)
    report["options"]["guard"] = str(guard_path)
    report["out"] = str(out_path)

    print(f"v829 manifest build: {sources_rel} -> {out_path}")
    print(f"  interpreter {sys.version.split()[0]} {sys.executable}")
    print(f"  sources sha256 {sources_sha} ({sources_bytes} B)")
    if build["budget_frames"] is not None:
        print(f"  budget_frames {build['budget_frames']} "
              f"({build['budget_source']})")
    if build["rngs"] is not None:
        repeat_note = ("paired RNGs per (source, checkpoint) cell" if build["rngs"] > 1
                       else "one measurement per cell")
        print(f"  rngs {build['rngs']} ({repeat_note})")
    for note in build["notes"]:
        print(f"  note {note}")

    if findings:
        ordered = sort_findings(findings)
        build["findings"] = ordered
        build["codes"] = codes_of(ordered)
        report["findings"] = ordered
        report["codes"] = build["codes"]
        report["status"] = "refused"
        print(f"REFUSED: {len(ordered)} finding(s), no manifest written")
        for entry in ordered:
            where = []
            if "source_index" in entry:
                where.append(f"sources[{entry['source_index']}]")
            if "ancestor_id" in entry:
                where.append(str(entry["ancestor_id"]))
            print(f"  [{entry['code']}]" + (f" {' '.join(where)}" if where else ""))
            print(f"      {entry['detail']}")
        if args.report is not None:
            write_report(args.report, report)
        return exit_for(ordered)

    manifest = build["manifest"]
    try:
        manifest_sha, manifest_bytes = write_manifest(out_path, manifest)
    except BuildError as error:
        print(f"REFUSED [{error.code}] {error.detail}", file=sys.stderr)
        report["status"] = "refused"
        report["findings"] = [finding(error.code, error.detail)]
        report["codes"] = [error.code]
        if args.report is not None:
            write_report(args.report, report)
        return CODE_EXIT.get(error.code, EXIT_FATAL)

    print(f"  roots {len(manifest['roots'])}")
    for entry in build["roots"]:
        repeat_text = ("" if entry["pair_index"] is None
                       else f"pair_index {entry['pair_index']} ")
        print(f"  root {entry['cell']:22s} {entry['ancestor_id']} "
              f"ckpt {str(entry['ckpt_sha256'])[:12]}... "
              f"{repeat_text}rng {entry['pair_rng']} "
              f"window {entry['window']} ({entry['window_source']}) "
              f"tape {entry['tape_sha256'][:16]}... rows {entry['tape_rows']} "
              f"frames {entry['frames_built']} takeover {entry['takeover_frame']} "
              f"checkpoints {entry['hash_checkpoints']} "
              f"budget {entry['budget_frames']} total {entry['total_frames']} "
              f"headroom {entry['headroom_frames']}")
    print(f"  wrote {out_path}: {manifest_bytes} B sha256 {manifest_sha}")

    guard = run_guard(out_path, budget=int(manifest["roots"][0]["budget_frames"]),
                      allow_exposed=bool(args.allow_exposed),
                      guard_path=guard_path)
    report["guard"] = guard
    report["manifest"] = {"path": str(out_path), "sha256": manifest_sha,
                          "bytes": manifest_bytes}
    report["status"] = "validated" if guard["exit"] == 0 else "unvalidated"
    print_guard(guard)

    sidecar = pathlib.Path(str(out_path) + ".guard.json")
    write_report(sidecar, {"manifest": str(out_path), "manifest_sha256": manifest_sha,
                           "allow_exposed": bool(args.allow_exposed),
                           "budget": int(manifest["roots"][0]["budget_frames"]),
                           "guard": guard, "status": report["status"]})

    if args.report is not None:
        write_report(args.report, report)

    if guard["exit"] is None:
        print("REFUSED [GUARD_UNRUNNABLE] the guard subprocess could not start; the "
              "manifest is kept for inspection and is UNVALIDATED", file=sys.stderr)
        return EXIT_FATAL
    if guard["exit"] != 0:
        print(f"UNVALIDATED: v829_guard.py exited {guard['exit']}; the manifest is "
              f"kept at {out_path} for human review (sidecar {sidecar.name})",
              file=sys.stderr)
        return int(guard["exit"])
    print(f"VALIDATED: guard exit 0; manifest {out_path} sha256 {manifest_sha}")
    first = manifest["roots"][0]
    print(f"next: python -B tools/rl/v829_pair.py --mode check --manifest {out_path} "
          f"--window {first['window']} "
          f"--budget-frames {first['budget_frames']}"
          + (" --allow-exposed" if args.allow_exposed else ""))
    same_window = [root for root in manifest["roots"]
                   if root.get("window") == first.get("window")]
    if len(same_window) > 1:
        print(f"      --window {first['window']} selects {len(same_window)} roots "
              f"(every checkpoint x paired RNG of that ancestor) -- the section 2 "
              f"shape; omit --window to select all {len(manifest['roots'])} roots")
    return EXIT_OK


def write_report(path: pathlib.Path, payload: dict) -> None:
    """Write a JSON evidence document (never fatal: it is not the deliverable)."""

    try:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        pathlib.Path(path).write_bytes(text.encode("utf-8"))
    except OSError as error:
        print(f"warning: cannot write report {path}: {error}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
EPILOG = """
the derivation in one screen
----------------------------
  sha256(tape)               -> parent_prefix_digest, verify_anchor.prefix_digest,
                                verify_anchor.window_recording_sha256,
                                prefix_tape_sha256
  rows * 4                   -> total_frames (= prefix_tape_frames); the row count
                                and the tape path go into total_frames_source
  takeover_frame // 4        -> verify_anchor.hash_checkpoints (N // 4)
  takeover_frame             -> verify_anchor.recorded_frames
  takeover_frame % 4 == 3    -> required (4k - 1), else TAKEOVER_NOT_ANCHOR
  takeover + 1 <= total      -> else TAKEOVER_BEYOND_TAPE
  takeover + budget <= total -> else BUDGET_EXCEEDS_TOTAL
  sha256(ckpt)               -> ckpt_sha256, compared with the sheet's claim
  sha256(engine)             -> engine_sha256, measured (the sheet's claim is
                                verified, never copied)

one ancestor, 1-3 checkpoints x 1-16 paired RNGs
------------------------------------------------
  ckpts: [{path, sha256[, seed][, pair_index]}, ...]   1-3 entries
  rngs: 1..16 (optional, default 1)  -> measurements per (source, checkpt):
                              n sources x m checkpoints x r rngs = n * m * r roots,
                              all of one source SHARING its ancestor_id -- the
                              section 2 "3 checkpoints x 2 pair RNGs per ancestor"
                              grid, in one manifest
  pair_rng                 -> PAIR_RNG_BASE + rng_index * PAIR_RNG_RNG_STRIDE
                              + source_index * PAIR_RNG_SOURCE_STRIDE + ckpt_index
                              (20265000/01/02, ..., repeat 1: 20266000/01/02, ...)
  pair_index               -> rngs > 1: the RNG index 0..rngs-1 inside the cell;
                              rngs == 1: ckpts[j].pair_index, else the ckpt index
  ckpt_seed                -> recorded on the root (ckpts[j].seed)
  ckpt + ckpts together    -> SOURCES_INVALID (mutually exclusive)
  sources[i].pair_rng      -> SOURCES_INVALID when ckpts or rngs > 1: one seed on
                              the row cannot address several cells
  ckpts[j].pair_index      -> SOURCES_INVALID with rngs > 1: the repeat number is
                              the RNG index, so a per-checkpoint number would
                              address only one of the repeats

the 4 and the 3 above are ANCHOR_STRIDE and TAKEOVER_MOD in this file, the seed
constants are PAIR_RNG_BASE / PAIR_RNG_SOURCE_STRIDE / PAIR_RNG_RNG_STRIDE; the
columns are aligned by hand for --help, the code always uses the constants.

codes (exit): SOURCES_UNREADABLE(2) SOURCES_INVALID(1) EXPOSURE_REQUIRED(1)
  TAKEOVER_NOT_ANCHOR(1) TAKEOVER_BEYOND_TAPE(1) BUDGET_EXCEEDS_TOTAL(1)
  BUDGET_NOT_A_TIER(1) TAPE_MISSING(2) TAPE_MALFORMED(1) RNG_DUPLICATE(1)
  WINDOW_DUPLICATE(1) CKPT_SHA_MISMATCH(1) CKPT_MISSING(2) ENGINE_MISSING(2)
  ENGINE_SHA_MISMATCH(1) OUT_UNWRITABLE(2) GUARD_UNRUNNABLE(2)

after writing, the guard always runs:
  <this python> -B tools/rl/v829_guard.py check <out> --budget <budget> [--allow-exposed]
and its exit code becomes this tool's exit code (1 refused / 2 unevaluable).
No manifest is written at all when a builder check fails.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v829_manifest_build.py",
        description=(f"build a {MANIFEST_SCHEMA} from a research-side "
                     f"{SOURCES_SCHEMA} sheet and validate it with v829_guard.py"),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    build_cmd = sub.add_parser(
        "build",
        help="derive the manifest, write it, then run the guard",
        description="derive every manifest field from the tape / ckpt / engine "
                    "bytes, write the manifest, then run v829_guard.py check on it. "
                    "The sheet binds ONE checkpoint per root through 'ckpt', or 1-3 "
                    "checkpoints per ancestor through the optional 'ckpts' array "
                    "(one root per source x checkpoint, all under the source's "
                    "ancestor_id).  The optional top-level 'rngs' (1-16) measures "
                    "every (source, checkpoint) cell that many times with different "
                    "pair RNGs -- 8 sources x 3 checkpoints x 2 rngs = 48 roots, the "
                    "pre-registered section 2 grid -- see the epilog for the seed "
                    "allocation rule.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    build_cmd.add_argument("--sources", required=True, type=pathlib.Path,
                           help=f"the {SOURCES_SCHEMA} sheet to read; it declares "
                                f"either 'ckpt' (one checkpoint) or 'ckpts' "
                                f"(1-{MAX_CKPTS} checkpoints per ancestor), never "
                                f"both, and optionally 'rngs' (1-{MAX_RNGS} paired "
                                f"RNGs per (source, checkpoint) cell)")
    build_cmd.add_argument("--out", required=True, type=pathlib.Path,
                           help="where to write the manifest (parent dirs created)")
    build_cmd.add_argument("--window", type=int, default=None,
                           help="default window label for sources that declare "
                                "none; it selects the root and names the private "
                                "data dir, so a missing one is derived from "
                                "ancestor_id instead of guessed")
    build_cmd.add_argument("--budget", type=int, default=None,
                           help=f"override sources.budget_frames; must be one of "
                                f"{BUDGET_TIERS} (the tiers the guard can prove)")
    build_cmd.add_argument("--ckpt", type=pathlib.Path, default=None,
                           help="measure this checkpoint instead of sources.ckpt.path "
                                "(the sheet's sha256 must still match); refused when "
                                "the sheet uses 'ckpts'")
    build_cmd.add_argument("--ckpt-sha256", default=None,
                           help="expected checkpoint digest when the sheet omits it; "
                                "refused when the sheet uses 'ckpts'")
    build_cmd.add_argument("--engine", default=None,
                           help="engine binary to measure instead of sources.engine")
    build_cmd.add_argument("--guard", type=pathlib.Path, default=None,
                           help=f"guard to run (default {DEFAULT_GUARD})")
    build_cmd.add_argument("--allow-exposed", action="store_true",
                           help="forward --allow-exposed to the guard: acknowledge "
                                "that a root was already used for training/selection "
                                "(without it an exposed root is refused)")
    build_cmd.add_argument("--report", type=pathlib.Path, default=None,
                           help="also write the JSON build report (derivation "
                                "evidence) here")

    eq_cmd = sub.add_parser(
        "equivalence",
        help="prove this tool's tape reader equals v829_pair.read_tape",
        description="import tools/rl/v829_pair.py, call read_tape(tape, N + 1) and "
                    "print its fields next to this tool's scan_tape() output",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    eq_cmd.add_argument("--tape", required=True, action="append",
                        type=pathlib.Path,
                        help="prefix tape to compare (repeatable)")
    eq_cmd.add_argument("--takeover", required=True, type=int,
                        help="takeover frame N; read_tape is called with N + 1")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "equivalence":
        return command_equivalence(args)
    if args.command == "build":
        return command_build(args)
    parser.print_help(sys.stderr)
    return EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())
