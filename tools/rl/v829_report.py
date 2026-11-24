#!/usr/bin/env python3
"""v829_report.py -- V829-PAIR-01 sec. 4 scoring and reporting, fully offline.

What this file is
=================
`tools/rl/v829_pair.py` *runs* pairs and writes one JSON line per single-arm
episode (``schema = "v829-pair-episode/1"``).  Nothing in the repository scored
those lines: the pair runner records, it does not aggregate.  This file is the
missing sec. 4 layer -- it reads ``*.jsonl`` pair rows (or a glob of them) and
produces the quantities the pre-registration names, with every number traceable
to the exact file/line it came from.

It starts no engine, imports nothing from ``touhou_rl``/``torch``/``numpy`` and
uses only the Python standard library (3.10+).  It writes nothing except the
``--json``/``--markdown`` files it is asked for.

Pre-registration sec. 4, mapped to code
=======================================
For every ancestor and every budget (1,800 frames = 30 s, 2,400 = 40 s) the
complete pairs are tallied into the 2x2 table

===========================  =============================================
cell                          condition
===========================  =============================================
``both_full``                 G1 full and T2 full
``only_G1``                   G1 full and T2 not full
``only_T2``                   T2 full and G1 not full
``neither``                   neither full
===========================  =============================================

and the ancestor's difference is

    diff(ancestor) = (only_G1 - only_T2) / n_valid_pairs

``n_valid_pairs`` is exactly ``both_full + only_G1 + only_T2 + neither``, i.e.
the pairs whose two arms both carry a usable score.  The endpoint effect is the
**equal-weight mean over the ancestors** (never pooled over rounds):

    Delta = mean over ancestors of diff(ancestor)

plus the two absolute per-arm full-segment rates, the positive/tie/negative
ancestor counts, and a real **leave-one-out** interval: drop ancestor k, take
the mean of the remaining ones, for every k; report ``min``/``max`` over those
k runs (and which ancestor produced each extreme).  That is not a standard
deviation and is not advertised as one.

Ties in the leave-one-out extremes (field contract)
===================================================
Dropping different ancestors can reach *the same* extreme (e.g. four ancestors
at +0.5 and four at -0.5: dropping any negative one gives the high end, dropping
any positive one gives the low end).  A single name silently answers "which one"
with "whichever the loop reached first", which is an accident of arithmetic, not
a result.  This file therefore keeps the historical string fields and adds the
full tie set:

============================  ==========================================
``loo.min`` / ``loo.max``      the extreme value (float; see exactness below)
``loo.min_dropped``            **string, unchanged meaning**: deterministic
``loo.max_dropped``            primary name for the extreme
``loo.*_dropped_primary``      the same primary name, spelled out (new)
``loo.*_dropped_ties``         **sorted list** of every ancestor whose run
                               reaches that extreme (new)
``loo.values[i]``              per-run number: ``mean_remaining`` (float),
                               ``mean_remaining_exact`` ("num/den"), which
                               ancestors are marked ``下界``/``上界`` in the
                               markdown table
============================  ==========================================

The primary name is defined as **the lexicographically first ancestor id among
the tied runs**, so it is a documented function of the data, not of iteration
order.  ``min_dropped``/``max_dropped`` were strings before this change and stay
strings: turning a documented string field into a list would break every reader
that formats or compares it (silently printing ``['anc4', ...]`` where a name is
expected), so the list lives in a new, explicitly named field instead.

Exactness: every mean published here (``delta``, the leave-one-out runs, the
unresolved-pair bounds, the per-checkpoint delta/bounds) is computed as an exact
rational (:class:`fractions.Fraction`) from the integer 2x2 cells and rounded to
float **once**, at the end.  Two mathematically equal means are then bit-equal,
so a tie is detected as a tie and not as a 1-ULP accident.  This matters across
interpreters: CPython 3.12 changed the builtin ``sum()`` for floats
(gh-100425/gh-105901, Neumaier compensation), so ``sum([...])/k`` over the same
numbers is not guaranteed to be bit-identical on 3.10 and 3.12.

"Full segment" is recomputed here from the raw fields, not trusted:

    full = (frames == budget_frames
            and decisions_executed == budget_decisions
            and first_death_offset is None
            and terminal is None)

A last-frame death therefore scores 0, and a legal stage clear scores 0 as
well while being listed separately as an event (pre-registration sec. 4).
When a row also carries ``score.full_segment`` (``v829_pair.score``), the two
must agree or the report refuses: silently preferring one of two
disagreements would be inventing a number.

Fail-closed gates
=================
The pre-registration's fourth decision row is "not enough ancestors / identity
mismatch / missing key scoring evidence / main batch incomplete -> report the
usable denominator and bounds".  Implemented as three levels:

* level 1 ``delta`` (a point estimate) is emitted only if the data gates pass
  **and** every pre-registered ancestor has at least one complete pair at that
  budget **and** ``--require-complete`` (default on).  Otherwise it is
  ``null`` with a ``withheld_reason`` and the denominators are still printed.
* level 2 ``bounds`` (all-G1 / all-T2 extremes over the unresolved pairs) is
  always printed: each unresolved pair may resolve to ``only_G1`` or
  ``only_T2``, so the ancestor difference can move by at most +/-1 per
  unresolved pair, and the endpoint bound is the mean of the ancestor bounds.
* level 3 ``verdict`` (the pre-registration's decision table) is emitted only
  when the main batch is complete (every expected pair present, both arms,
  three checkpoints).  Otherwise the verdict is
  ``REFUSED_INSUFFICIENT_DENOMINATOR`` with the concrete reasons.

Hard refusals (duplicate row, arm/ancestor conflict, manifest disagreement,
score disagreement, ...) set exit code 1 and null every ``delta``, but the
per-ancestor denominators and the bounds are still written -- refusing to
*conclude* is not refusing to *count*.

Retries.  Pre-registration sec. 4 allows exactly one re-run of both arms of a
failed pair, keeping both records.  A retry therefore collides on
(ancestor, budget, checkpoint, pair_rng, episode, arm) with the original.  The
rule here: two rows for the same cell are accepted only when their ``tag``
fields differ (a retry runs under its own tag); the later one is used and the
earlier is recorded as ``superseded``.  Two rows with the *same* tag, or three
rows for one cell, are ``DUPLICATE_ROW`` / ``RETRY_LIMIT`` and refuse.

Inventory events (sec. 4, last paragraph)
========================================
Per the caliper in TRAINING_HOST_REPLY_02 sec. 3, ``emit_decrement_at`` is the
frame **inside the emitting decision** at which the inventory drop is first
observed (``= 2`` for a 2-frame decision), so a request is honoured only if the
inventory actually fell, and the honest reading of "which request was
honoured" is ``emit_decrement_at_list[i] is not None`` -- not "the last
request".  This file reports, per arm and split by survived/death:

* every actual inventory deduction with its decision step, its absolute frame
  (``bomb_frames[i] + emit_decrement_at_list[i] - 1``), the inventory before
  and after, and the gap to the previous deduction;
* frames alive after the inventory reached 0 (``bombs_end == 0``); a surviving
  episode is truncated at the segment end and says so;
* deaths with a bomb still in stock, derived from ``bombs_end`` when the death
  is the segment's last frame, otherwise derived only if the inventory series
  is monotone and ``bombs_used`` equals the honoured count -- and reported as
  "insufficient" (``不足``) instead of guessed when it is not derivable.

Path form of the document (cross-platform comparability)
=======================================================
s12 made the *numbers* byte-identical on 3.10 and 3.12.  The path strings were
still whatever the runtime saw, so the same batch produced
``D:\\WorkShop\\Touhou_RL\\.tmp_v829_pair\\eval\\x.jsonl`` on Windows and
``/mnt/d/WorkShop/Touhou_RL/.tmp_v829_pair/eval/x.jsonl`` on WSL: identical
numbers, incomparable files.  Every path field is now written as a
**repository-relative POSIX** path (``.tmp_v829_pair/eval/x.jsonl``), computed by
``path_form()`` against ``REPO`` -- i.e. ``REPO = __file__.resolve().parents[2]``,
which resolves to the same tree whether it is reached as ``D:\\...`` or as
``/mnt/d/...``.  Fields covered: ``inputs.files``, ``inputs.pairs``,
``inputs.manifest``, ``provenance.files[].path``, ``provenance.manifest.path``,
``rows[].ref`` and every ``row_refs``/``files`` list built from them.

* a file **outside** the repository keeps an absolute path and is additionally
  flagged, so a reader can tell "comparable" from "host-specific": it appears in
  ``inputs.files_external`` and as ``{"path": ..., "external": true}`` in
  ``inputs.files_detail`` (and ``provenance.files[].external``).
* ``--keep-native-paths`` additionally records the raw runtime spelling in
  ``inputs.files_native`` / ``inputs.manifest_native`` / ``inputs.pairs_native``,
  so a human on either host can still open the files.  It is OFF by default
  because a native spelling is host-specific by definition: emitting it
  unconditionally would recreate exactly the defect above.

Usage
=====
::

    python -B tools/rl/v829_report.py \
        --pairs ".tmp_v829_pair/eval/*.jsonl" \
        --manifest .tmp_v829_pair/manifest/smoke-old-window.json \
        --checkpoint-seeds 0,1,2 \
        --json out.json --markdown out.md

Exit codes: 0 = every gate passed and every endpoint delta emitted; 1 = refused
(some gate failed; the report still carries denominators and bounds); 2 =
usage/IO error (unreadable input, bad JSON line, bad arguments).
"""

from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import time
from fractions import Fraction

TOOL = "tools/rl/v829_report.py"
TOOL_VERSION = "1"
REPORT_SCHEMA = "v829-pair-report/1"

# The schema `v829_pair.py` really writes (SCHEMA_EPISODE, v829_pair.py:320).
# The work order guessed "v829-pair-row/1"; this is what the rows contain.
ROW_SCHEMA = "v829-pair-episode/1"
ACCEPTED_ROW_SCHEMAS = (ROW_SCHEMA,)
MANIFEST_SCHEMA = "v829-pair-manifest/1"

ARMS = ("G1", "T2")

# Pre-registration sec. 2: "8 个新开发祖先", "3 个 checkpoint x 2 个配对 RNG",
# sec. 4 table: "5 个百分点" and "至少 5/8".
PREREG_ANCESTORS = 8
PREREG_CKPTS = 3
PREREG_RNGS = 2
PREREG_THRESHOLD_PP = 5.0
PREREG_MIN_POSITIVE = (5, 8)

PRIMARY_BUDGET = 1800      # 30 s, the primary endpoint (sec. 2: "30 秒主比较")
SECONDARY_BUDGET = 2400    # 40 s, reported with the same formula

BUDGET_NAMES = {1800: "30s", 2400: "40s"}
BUDGET_CN = {1800: "30 秒（1,800 帧 / 900 决策）",
             2400: "40 秒（2,400 帧 / 1,200 决策）"}

EXIT_OK, EXIT_REFUSED, EXIT_ERROR = 0, 1, 2

DECISION_FRAMES_DEFAULT = 2
LEGAL_TERMINALS = (None, "first_death", "game_over", "stage_clear")

_SEED_RE = re.compile(r"seed[-_ ]?(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class ReportError(Exception):
    """A refusal or a usage/IO error.  ``code`` is stable, ``hard`` means "no delta"."""

    def __init__(self, code, message, *, hard=True, exit_code=EXIT_REFUSED,
                 detail=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hard = hard
        self.exit_code = exit_code
        self.detail = detail or {}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def flag(value) -> bool:
    """'invalid'/'error' 非空: None, '', False, 0 and empty containers are empty."""
    if value is None or value is False:
        return False
    if isinstance(value, (str, bytes)) and not value.strip():
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
        return False
    if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
        return False
    return True


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def fmt_frac(value) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def fmt_pp(value) -> str:
    """A fraction -> '+12.50 pp' (multiplies by 100)."""
    return "n/a" if value is None else f"{value * 100:+.2f} pp"


def fmt_ppv(value) -> str:
    """A number already in percentage points -> '+12.50 pp' (no rescaling)."""
    return "n/a" if value is None else f"{value:+.2f} pp"


def fmt_rate(value) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def fmt_ratio(num, den) -> str:
    return "n/a" if den in (None, 0) else f"{num}/{den}"


def fmt_int(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def as_dict_of_counts() -> dict:
    return {"both_full": 0, "only_G1": 0, "only_T2": 0, "neither": 0}


# --------------------------------------------------------------------------- #
# exact arithmetic for every published mean
# --------------------------------------------------------------------------- #
# Rationale: a mean of k ratios of small integers is a rational number.  Doing it
# in float means the answer depends on the summation order *and* on the
# interpreter: CPython 3.12 changed the builtin sum() for floats (gh-100425,
# Neumaier compensation), so sum([...]) / k over the same numbers can differ in
# the last bit between 3.10 and 3.12.  That 1 ULP was enough to turn an exact tie
# (all four ancestors at -0.5 dropped from a +0.5/4x(-0.5) batch) into a
# "unique" extreme.  Keeping the arithmetic exact and rounding once removes both
# the order dependence and the version dependence, and it makes "tied" mean
# mathematically equal.
def exact_diff(item):
    """``diff = (only_G1 - only_T2) / n_valid_pairs`` as an exact Fraction."""
    n = item["n_valid_pairs"]
    if not n:
        return None
    cells = item["cells"]
    return Fraction(cells["only_G1"] - cells["only_T2"], n)


def exact_bounds(item):
    """(lo, hi, lo_pending_only, hi_pending_only) as exact Fractions.

    Same formulas as ``tally_ancestor``: one unresolved pair can move an
    ancestor's difference by at most 1/n towards either arm.
    """
    n = item["n_valid_pairs"]
    if not n:
        return None
    cells = item["cells"]
    numerator = cells["only_G1"] - cells["only_T2"]
    unresolved = item["unresolved"]["total"]
    pending = item["unresolved"]["pending"]
    return (Fraction(numerator - unresolved, n),
            Fraction(numerator + unresolved, n),
            Fraction(numerator - pending, n),
            Fraction(numerator + pending, n))


def exact_mean(values):
    """Mean of exact rationals -> float, rounded exactly once (None if empty)."""
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values, Fraction(0)) / len(values))


def fmt_exact(value) -> str:
    """A Fraction as ``num/den`` (evidence for the JSON, no float rounding)."""
    return "n/a" if value is None else f"{value.numerator}/{value.denominator}"


def fmt_tie_names(names) -> str:
    """`['anc4','anc7']` -> ``'`anc4`, `anc7`（并列）'``; a single name as-is."""
    if not names:
        return "n/a"
    text = ", ".join(f"`{name}`" for name in names)
    return f"{text}（并列）" if len(names) > 1 else text


# --------------------------------------------------------------------------- #
# path form of the document (cross-platform comparability)
# --------------------------------------------------------------------------- #
#: The checkout this script lives in: ``tools/rl/v829_report.py`` -> parents[2].
#: Both the report and its inputs are inside it, so "repo-relative POSIX" is
#: well defined on Windows (``D:\WorkShop\Touhou_RL``) and on WSL
#: (``/mnt/d/WorkShop/Touhou_RL``) and yields the SAME string for the same file.
#:
#: Every path field of the document goes through ``path_form()`` and is therefore
#: normalised unconditionally.  ``--keep-native-paths`` does not undo that: it
#: *adds* the raw runtime spellings as ``inputs.files_native`` /
#: ``inputs.manifest_native`` so a human can open the files on either host, while
#: the comparable fields stay comparable.
REPO = pathlib.Path(__file__).resolve().parents[2]

#: Set from ``--keep-native-paths`` in ``main()`` *before* ``build_document()``.
#: While it is False (the default) the native spellings are not written into the
#: document at all -- that is what keeps two hosts' ``--json`` byte-identical.
#: One module-level switch, so ``load_rows``/``load_manifest``/``build_document``
#: cannot disagree about it.
KEEP_NATIVE_PATHS = False


def native_or_none(value):
    """The raw runtime spelling, or None when ``--keep-native-paths`` is off."""

    return str(value) if KEEP_NATIVE_PATHS else None


def repo_relative_parts(resolved: pathlib.Path):
    """``(parts, external)``: this path's components below the checkout root.

    ``external`` is True for a path outside the repository.  On Windows the
    comparison is retried case-insensitively, because NTFS preserves the case a
    path was *written* with while ``Path.resolve()`` does not canonicalise it --
    without that retry a lowercase ``--pairs`` argument would be misfiled as
    external and would silently leave the comparable form.
    """

    try:
        return list(resolved.relative_to(REPO).parts), False
    except ValueError:
        pass
    if os.name == "nt":
        want = [part.lower() for part in REPO.parts]
        have = [part.lower() for part in resolved.parts]
        if len(have) >= len(want) and have[:len(want)] == want:
            return list(resolved.parts[len(want):]), False
    return None, True


def path_form(value):
    """``(rendered, external, native)`` for one filesystem path.

    * ``rendered`` -- repository-relative POSIX (``a/b/c.jsonl``) when the file is
      inside this checkout; the absolute POSIX path when it is not.
    * ``external`` -- True iff the file is outside the repository, i.e. a path two
      platforms cannot be expected to spell the same way (``external: true``).
    * ``native``  -- exactly what the runtime saw, for human cross-check.
    """

    raw = str(value)
    path = pathlib.Path(raw).expanduser()
    try:
        resolved = path.resolve()
    except OSError:                                        # pragma: no cover
        resolved = path
    parts, external = repo_relative_parts(resolved)
    if external:
        return resolved.as_posix(), True, raw
    return "/".join(parts) if parts else ".", False, raw


def pattern_form(text) -> str:
    """``path_form`` for a ``--pairs`` entry that may be a glob.

    ``glob.has_magic`` decides: a literal path goes through ``path_form`` whole,
    a pattern has only its *directory* part normalised and its magic tail kept, so
    ``D:\\...\\.tmp\\*.jsonl`` and ``/mnt/d/.../.tmp/*.jsonl`` both become
    ``.tmp/*.jsonl``.
    """

    raw = str(text)
    if not globmod.has_magic(raw):
        return path_form(raw)[0]
    marks = [raw.index(char) for char in "*?[" if char in raw]
    magic = min(marks) if marks else len(raw)
    cut = max(raw.rfind("/", 0, magic),
              raw.rfind(os.sep, 0, magic) if os.sep != "/" else -1)
    if cut <= 0:
        return raw.replace(os.sep, "/")
    head, tail = raw[:cut], raw[cut:]
    return path_form(head)[0] + tail.replace(os.sep, "/")


# --------------------------------------------------------------------------- #
# input: pair rows
# --------------------------------------------------------------------------- #
class RowRef:
    """One JSONL line, identified by file + line + digest of that very line."""

    __slots__ = ("path", "line", "line_sha256_16", "row", "order")

    def __init__(self, path: str, line: int, line_sha256_16: str, row: dict,
                 order: int):
        self.path = path
        self.line = line
        self.line_sha256_16 = line_sha256_16
        self.row = row
        self.order = order

    @property
    def ref(self) -> str:
        return f"{self.path}#L{self.line}"

    @property
    def tag(self):
        return self.row.get("tag")


def expand_patterns(patterns):
    """Expand every --pairs entry (glob or literal path) into sorted unique paths."""
    files: list[pathlib.Path] = []
    seen = set()
    for pattern in patterns:
        text = str(pattern)
        # an absolute glob or a relative one both work; glob() is not recursive so
        # a typo can never silently walk the repository root.
        matches = sorted(globmod.glob(text)) if globmod.has_magic(text) else [text]
        if not matches:
            raise ReportError(
                "INPUT_PATTERN_EMPTY",
                f"--pairs {text!r} matched no file; refusing to report on an empty "
                f"selection (a zero-row report would look like a data result)",
                exit_code=EXIT_ERROR)
        for match in matches:
            path = pathlib.Path(match)
            if path.is_dir():
                raise ReportError(
                    "INPUT_IS_DIRECTORY",
                    f"--pairs {match!r} is a directory; pass a *.jsonl glob or file",
                    exit_code=EXIT_ERROR)
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    if not files:
        raise ReportError("INPUT_EMPTY", "no pair files after expansion",
                          exit_code=EXIT_ERROR)
    return files


def load_rows(files):
    """Parse every file as JSONL.  Returns (rows, files_meta).  Raises on bad lines."""
    rows: list[RowRef] = []
    files_meta: list[dict] = []
    order = 0
    for path in files:
        shaped = dict(zip(("path", "external", "native"), path_form(path)))
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ReportError("INPUT_UNREADABLE", f"{path}: {exc}",
                              exit_code=EXIT_ERROR)
        digest = sha256_bytes(raw)
        count = 0
        for number, blob in enumerate(raw.split(b"\n"), 1):
            if not blob.strip():
                continue
            try:
                text = blob.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ReportError(
                    "ROW_UNDECODABLE",
                    f"{path}:{number}: not UTF-8 ({exc})", exit_code=EXIT_ERROR)
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ReportError(
                    "ROW_UNREADABLE",
                    f"{path}:{number}: not JSON ({exc})", exit_code=EXIT_ERROR)
            if not isinstance(row, dict):
                raise ReportError(
                    "ROW_NOT_OBJECT",
                    f"{path}:{number}: a pair row must be a JSON object, got "
                    f"{type(row).__name__}", exit_code=EXIT_ERROR)
            rows.append(RowRef(str(shaped["path"]), number,
                               sha256_bytes(blob)[:16], row, order))
            order += 1
            count += 1
        files_meta.append({
            "path": shaped["path"],
            "external": shaped["external"],
            "path_native": native_or_none(shaped["native"]),
            "bytes": len(raw),
            "sha256": digest,
            "rows_read": count,
            "manifest_sha256_values": sorted({
                str(r.row.get("manifest_sha256")) for r in rows[-count:]
                if count}),
        })
    if not rows:
        raise ReportError(
            "INPUT_NO_ROWS",
            f"{len(files)} file(s) parsed but no pair row was found; refusing to "
            f"report an empty batch", exit_code=EXIT_ERROR)
    return rows, files_meta


# --------------------------------------------------------------------------- #
# row-level validation (identity + score)
# --------------------------------------------------------------------------- #
class Issues:
    """Collects hard refusals and soft warnings."""

    def __init__(self):
        self.hard: list[dict] = []
        self.soft: list[dict] = []

    def hard_error(self, code, message, ref=None, detail=None):
        self.hard.append({"code": code, "message": message, "ref": ref,
                          "detail": detail or {}})

    def warn(self, code, message, ref=None, detail=None):
        self.soft.append({"code": code, "message": message, "ref": ref,
                          "detail": detail or {}})

    @property
    def ok(self) -> bool:
        return not self.hard


def check_identity(ref: RowRef, issues: Issues) -> dict:
    """Per-line structural checks.  Returns the normalised cell identity."""
    row = ref.row
    schema = row.get("schema")
    if schema not in ACCEPTED_ROW_SCHEMAS:
        issues.hard_error(
            "ROW_SCHEMA",
            f"schema {schema!r} is not one of {list(ACCEPTED_ROW_SCHEMAS)}; the "
            f"report layer only scores rows the pair runner wrote", ref.ref)
    arm = row.get("arm")
    if arm not in ARMS:
        issues.hard_error(
            "ARM_UNKNOWN", f"arm {arm!r} is neither 'G1' nor 'T2'", ref.ref)
    ancestor = row.get("ancestor_id")
    if not isinstance(ancestor, str) or not ancestor.strip():
        issues.hard_error("ANCESTOR_MISSING",
                          f"ancestor_id {ancestor!r} is not a non-empty string",
                          ref.ref)
        ancestor = None
    pair_id = row.get("pair_id")
    if isinstance(pair_id, str) and isinstance(ancestor, str) and "@" in pair_id:
        prefix = pair_id.split("@", 1)[0]
        if prefix != ancestor:
            issues.hard_error(
                "ARM_ANCESTOR_CONFLICT",
                f"pair_id {pair_id!r} names ancestor {prefix!r} but ancestor_id is "
                f"{ancestor!r}; the arm cannot be attributed to two ancestors",
                ref.ref)
    budget_frames = row.get("budget_frames")
    if not is_int(budget_frames):
        issues.hard_error(
            "BUDGET_MISSING", f"budget_frames {budget_frames!r} is not an int",
            ref.ref)
        budget_frames = None
    budget_decisions = row.get("budget_decisions")
    decision_frames = row.get("decision_frames", DECISION_FRAMES_DEFAULT)
    if not is_int(decision_frames) or decision_frames <= 0:
        issues.hard_error(
            "DECISION_FRAMES_INVALID",
            f"decision_frames {decision_frames!r} is not a positive int", ref.ref)
        decision_frames = DECISION_FRAMES_DEFAULT
    if not is_int(budget_decisions):
        issues.hard_error(
            "BUDGET_DECISIONS_MISSING",
            f"budget_decisions {budget_decisions!r} is not an int", ref.ref)
    elif is_int(budget_frames) and budget_frames != decision_frames * budget_decisions:
        issues.hard_error(
            "BUDGET_INCOHERENT",
            f"budget_frames {budget_frames} != decision_frames {decision_frames} x "
            f"budget_decisions {budget_decisions}", ref.ref)
    ckpt = row.get("ckpt_sha256")
    declared = row.get("ckpt_sha256_declared")
    if not isinstance(ckpt, str) or len(ckpt) != 64:
        issues.hard_error(
            "CKPT_SHA_MISSING", f"ckpt_sha256 {ckpt!r} is not a 64-char digest",
            ref.ref)
    elif isinstance(declared, str) and len(declared) == 64 and declared.lower() != ckpt.lower():
        issues.hard_error(
            "CKPT_SHA_CONFLICT",
            f"ckpt_sha256 {ckpt} != ckpt_sha256_declared {declared} in the same row",
            ref.ref)
    pair_rng = row.get("pair_rng")
    if not is_int(pair_rng):
        issues.hard_error(
            "PAIR_RNG_MISSING", f"pair_rng {pair_rng!r} is not an int", ref.ref)
    episode = row.get("episode", 0)
    if not is_int(episode):
        issues.hard_error(
            "EPISODE_MISSING", f"episode {episode!r} is not an int", ref.ref)
        episode = 0
    terminal = row.get("terminal")
    if terminal not in LEGAL_TERMINALS:
        issues.hard_error(
            "TERMINAL_UNKNOWN",
            f"terminal {terminal!r} is not one of "
            f"{[t for t in LEGAL_TERMINALS if t]} or null", ref.ref)
    frames = row.get("frames")
    death = row.get("first_death_offset")
    if is_int(frames) and is_int(death) and terminal is not None and death != frames - 1:
        issues.hard_error(
            "DEATH_OFFSET_INCONSISTENT",
            f"terminal {terminal!r} with first_death_offset {death} but frames "
            f"{frames}: a terminal episode must end at the death frame", ref.ref)
    if terminal is not None and death is None:
        issues.hard_error(
            "DEATH_OFFSET_MISSING",
            f"terminal {terminal!r} but first_death_offset is null; the score "
            f"cannot be anchored to a frame", ref.ref)
    if is_int(death) and is_int(frames) and not 0 <= death < frames:
        issues.hard_error(
            "DEATH_OFFSET_OUT_OF_RANGE",
            f"first_death_offset {death} outside [0, frames={frames})", ref.ref)
    return {
        "arm": arm,
        "ancestor_id": ancestor,
        "budget_frames": budget_frames,
        "budget_decisions": budget_decisions,
        "decision_frames": decision_frames,
        "ckpt_sha256": ckpt.lower() if isinstance(ckpt, str) else None,
        "pair_rng": pair_rng,
        "episode": episode,
    }


def classify_row(ref: RowRef, cell: dict):
    """Score one row.  Returns {'status', 'full_segment', 'reasons', ...}.

    status: 'scored' | 'technical_invalid' | 'pending'
    """
    row = ref.row
    if flag(row.get("invalid")) or flag(row.get("error")):
        reasons = []
        if flag(row.get("invalid")):
            reasons.append(f"invalid={row.get('invalid')!r}")
        if flag(row.get("error")):
            reasons.append(f"error={row.get('error')!r}")
        return {"status": "technical_invalid", "full_segment": None,
                "reasons": reasons, "terminal": row.get("terminal")}
    missing = [key for key in ("frames", "decisions_executed", "terminal",
                               "first_death_offset") if key not in row]
    if missing:
        return {"status": "pending", "full_segment": None,
                "reasons": [f"missing_field:{key}" for key in missing],
                "terminal": row.get("terminal")}
    frames = row.get("frames")
    decisions = row.get("decisions_executed")
    death = row.get("first_death_offset")
    terminal = row.get("terminal")
    if not is_int(frames) or not is_int(decisions):
        return {"status": "pending", "full_segment": None,
                "reasons": [f"frames/decisions_executed not int: "
                            f"{frames!r}/{decisions!r}"], "terminal": terminal}
    if terminal is None and decisions != cell["budget_decisions"]:
        # stopped early with no recorded terminal: the pre-registration calls this
        # "终态不明".  The row does not set invalid, so this layer refuses to
        # invent that flag; it is kept as a pending pair and counted in the bounds.
        return {"status": "pending", "full_segment": None,
                "reasons": ["end_without_terminal (终态不明; sec.4 列为技术无效，"
                            "但行内 invalid/error 为空，本层不代填)"],
                "terminal": terminal}
    full = (frames == cell["budget_frames"]
            and decisions == cell["budget_decisions"]
            and death is None
            and terminal is None)
    score = row.get("score")
    if isinstance(score, dict) and isinstance(score.get("full_segment"), bool):
        if bool(score["full_segment"]) != bool(full):
            return {"status": "conflict", "full_segment": None,
                    "reasons": [f"score.full_segment={score['full_segment']} but "
                                f"the raw fields give {full}"],
                    "terminal": terminal, "computed": bool(full)}
    if isinstance(score, dict) and isinstance(score.get("technical_invalid"), bool):
        if bool(score["technical_invalid"]) != flag(row.get("invalid")):
            return {"status": "conflict", "full_segment": None,
                    "reasons": [f"score.technical_invalid="
                                f"{score['technical_invalid']} but invalid="
                                f"{row.get('invalid')!r}"],
                    "terminal": terminal, "computed": bool(full)}
    return {"status": "scored", "full_segment": bool(full), "reasons": [],
            "terminal": terminal,
            "legal_clear": terminal == "stage_clear",
            "legal_death": terminal in ("first_death", "game_over")}


# --------------------------------------------------------------------------- #
# grouping rows into pairs
# --------------------------------------------------------------------------- #
def pair_key_of(cell: dict):
    return (cell["ancestor_id"], cell["budget_frames"], cell["ckpt_sha256"],
            cell["pair_rng"], cell["episode"])


def build_pairs(rows, issues: Issues):
    """Group rows into pair cells; apply the one-retry rule; classify each row."""
    cells: dict = {}
    identities: dict = {}
    for ref in rows:
        before = len(issues.hard)
        cell = check_identity(ref, issues)
        identities[ref.ref] = cell
        if len(issues.hard) > before:
            # identity is broken; keep the row for the denominator but never pair it
            key = ("<invalid>", ref.ref)
        else:
            key = pair_key_of(cell)
        cells.setdefault(key, []).append(ref)

    pairs: dict = {}
    for key, refs in cells.items():
        if key[0] == "<invalid>":
            continue
        by_arm: dict = {arm: [] for arm in ARMS}
        for ref in refs:
            arm = identities[ref.ref]["arm"]
            if arm in by_arm:
                by_arm[arm].append(ref)
        for arm, group in by_arm.items():
            group.sort(key=lambda r: r.order)
            if len(group) > 2:
                issues.hard_error(
                    "RETRY_LIMIT",
                    f"pair {key} arm {arm} has {len(group)} rows; sec.4 allows one "
                    f"re-run, so at most two records", group[0].ref,
                    {"refs": [r.ref for r in group]})
            if len(group) == 2:
                if group[0].tag == group[1].tag:
                    issues.hard_error(
                        "DUPLICATE_ROW",
                        f"pair {key} arm {arm} appears twice under the same tag "
                        f"{group[0].tag!r}: that is a duplicated line, not a retry",
                        group[1].ref, {"refs": [r.ref for r in group]})
                else:
                    issues.warn(
                        "RETRY_USED",
                        f"pair {key} arm {arm} has two rows with different tags; the "
                        f"later one is scored and the earlier kept as superseded "
                        f"(sec.4 one-retry rule)",
                        group[1].ref, {"used": group[1].ref,
                                       "superseded": group[0].ref})
        record = {
            "key": key,
            "ancestor_id": key[0], "budget_frames": key[1],
            "ckpt_sha256": key[2], "pair_rng": key[3], "episode": key[4],
            "rows": {}, "superseded": [], "status": None, "reasons": [],
            "full": {}, "refs": [],
        }
        for arm in ARMS:
            group = by_arm.get(arm) or []
            if not group:
                record["reasons"].append(f"missing_arm:{arm}")
                continue
            used = group[-1]
            record["rows"][arm] = used
            record["refs"].append(used.ref)
            if len(group) == 2:
                record["superseded"].append(group[0].ref)
            cell = identities[used.ref]
            verdict = classify_row(used, cell)
            record[("class_" + arm)] = verdict
            if verdict["status"] == "scored":
                record["full"][arm] = verdict["full_segment"]
            else:
                record["reasons"].append(f"{arm}:{verdict['status']}:"
                                         + ";".join(verdict["reasons"]))
        statuses = [record["class_" + arm]["status"]
                    for arm in ARMS if ("class_" + arm) in record]
        if len(record["rows"]) < 2:
            record["status"] = "pending"
            if not record["reasons"]:
                record["reasons"].append("missing_arm")
        elif any(s == "conflict" for s in statuses):
            record["status"] = "conflict"
        elif any(s == "technical_invalid" for s in statuses):
            record["status"] = "technical_invalid"
        elif any(s == "pending" for s in statuses):
            record["status"] = "pending"
        else:
            record["status"] = "complete"
        pairs[record["key"]] = record
    return pairs, identities


# --------------------------------------------------------------------------- #
# manifest cross-check
# --------------------------------------------------------------------------- #
def load_manifest(path: pathlib.Path):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReportError("MANIFEST_UNREADABLE", f"{path}: {exc}",
                          exit_code=EXIT_ERROR)
    digest = sha256_bytes(raw)
    try:
        doc = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError("MANIFEST_UNREADABLE", f"{path}: {exc}",
                          exit_code=EXIT_ERROR)
    if not isinstance(doc, dict):
        raise ReportError("MANIFEST_UNREADABLE",
                          f"{path}: a manifest must be a JSON object",
                          exit_code=EXIT_ERROR)
    if doc.get("schema") != MANIFEST_SCHEMA:
        raise ReportError(
            "MANIFEST_SCHEMA",
            f"{path}: schema {doc.get('schema')!r} != {MANIFEST_SCHEMA!r}",
            exit_code=EXIT_ERROR)
    roots = doc.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ReportError("MANIFEST_ROOTS", f"{path}: roots must be a non-empty list",
                          exit_code=EXIT_ERROR)
    return {
        "path": path_form(path)[0], "path_native": native_or_none(path),
        "external": path_form(path)[1],
        "bytes": len(raw), "sha256": digest,
        "schema": doc.get("schema"), "manifest_id": doc.get("manifest_id"),
        "roots": roots,
    }


def cross_check_manifest(manifest, pairs, rows, identities, issues: Issues):
    """Rows must agree with the manifest and with each other on manifest_sha256."""
    info = {"path": manifest["path"], "sha256": manifest["sha256"],
            "bytes": manifest["bytes"], "manifest_id": manifest.get("manifest_id"),
            "roots": manifest["roots"],
            "root_count": len(manifest["roots"]),
            "row_manifest_sha256_values": [], "root_table": []}
    expected = {}
    for index, root in enumerate(manifest["roots"]):
        if not isinstance(root, dict):
            issues.hard_error("MANIFEST_ROOT", f"root #{index} is not an object",
                              manifest["path"])
            continue
        ancestor = root.get("ancestor_id")
        budget = root.get("budget_frames")
        ckpt = root.get("ckpt_sha256")
        rng = root.get("pair_rng")
        if not isinstance(ckpt, str):
            issues.hard_error(
                "MANIFEST_ROOT",
                f"root #{index} has no ckpt_sha256; the checkpoint identity is "
                f"what groups the three expected checkpoints", manifest["path"])
            ckpt = None
        if not is_int(rng):
            issues.hard_error(
                "MANIFEST_ROOT",
                f"root #{index} has no integer pair_rng; pre-registration sec. 2 "
                f"freezes one RNG per pair", manifest["path"])
        expected[(ancestor, budget,
                  ckpt.lower() if isinstance(ckpt, str) else ckpt, rng)] = index
        info["root_table"].append({
            "root_index": index, "ancestor_id": ancestor,
            "budget_frames": budget, "window": root.get("window"),
            "ckpt_sha256": ckpt, "pair_rng": root.get("pair_rng"),
            "ckpt_path": root.get("ckpt_path"),
        })
    values = {}
    for ref in rows:
        value = ref.row.get("manifest_sha256")
        values.setdefault(str(value), []).append(ref.ref)
    info["row_manifest_sha256_values"] = [
        {"value": key, "rows": len(refs), "examples": refs[:3]}
        for key, refs in sorted(values.items())]
    if len(values) > 1:
        issues.hard_error(
            "MANIFEST_MIXED",
            f"rows came from {len(values)} different manifest digests "
            f"{sorted(values)}; a pre-registered batch is one frozen run table, so "
            f"these cells cannot be pooled into one delta", None,
            {"values": {k: len(v) for k, v in values.items()}})
    for key, refs in values.items():
        if key != manifest["sha256"] and key not in ("None", "null"):
            issues.hard_error(
                "MANIFEST_MISMATCH",
                f"{len(refs)} row(s) record manifest_sha256 {key} but "
                f"{manifest['path']} digests to {manifest['sha256']}; the rows and "
                f"the manifest are not the same frozen table", refs[0],
                {"rows": len(refs)})
    for pair in pairs.values():
        for arm in ARMS:
            ref = pair["rows"].get(arm)
            if ref is None:
                continue
            row = ref.row
            match = None
            for (ancestor, budget, ckpt, rng), index in expected.items():
                if (row.get("ancestor_id") == ancestor
                        and row.get("budget_frames") == budget
                        and (not isinstance(ckpt, str)
                             or str(row.get("ckpt_sha256", "")).lower() == ckpt)
                        and (rng is None or row.get("pair_rng") == rng)):
                    match = index
                    break
            if match is None:
                issues.hard_error(
                    "MANIFEST_ROOT_MISSING",
                    f"row {ref.ref} ({row.get('ancestor_id')!r}, "
                    f"budget_frames={row.get('budget_frames')}, ckpt "
                    f"{str(row.get('ckpt_sha256'))[:12]}, pair_rng "
                    f"{row.get('pair_rng')}) has no matching manifest root; the "
                    f"manifest does not bind this cell", ref.ref)
                continue
            root = manifest["roots"][match]
            window = root.get("window")
            if window is not None and row.get("window") is not None \
                    and int(row["window"]) != int(window):
                issues.hard_error(
                    "MANIFEST_WINDOW_MISMATCH",
                    f"row {ref.ref} window {row.get('window')} != manifest root "
                    f"#{match} window {window}", ref.ref)
    info["expected_pairs_per_ancestor_budget"] = {}
    for (ancestor, budget, _ckpt, _rng) in expected:
        key = f"{ancestor}|{budget}"
        info["expected_pairs_per_ancestor_budget"][key] = (
            info["expected_pairs_per_ancestor_budget"].get(key, 0) + 1)
    return info, expected


# --------------------------------------------------------------------------- #
# anchors: which ancestors / budgets / checkpoints are expected
# --------------------------------------------------------------------------- #
def expected_ancestors_from(manifest_info, pairs):
    if manifest_info is not None:
        found = {str(root.get("ancestor_id")) for root in manifest_info["roots"]
                 if isinstance(root, dict) and root.get("ancestor_id")}
        return sorted(found), "manifest"
    found = {p["ancestor_id"] for p in pairs.values()
             if isinstance(p["ancestor_id"], str)}
    return sorted(found), "rows"


def expected_pairs_table(args, manifest_info, expected_roots, pairs):
    """(ancestor, budget) -> expected number of pairs in the frozen design."""
    table: dict = {}
    source = None
    if manifest_info is not None:
        source = "manifest-roots"
        for (ancestor, budget, _ckpt, _rng), _index in expected_roots.items():
            key = (ancestor, budget)
            table[key] = table.get(key, 0) + int(args.episodes)
        return table, source
    source = (f"prereg-default({PREREG_CKPTS} checkpoints x {PREREG_RNGS} pair RNGs)")
    budgets = {p["budget_frames"] for p in pairs.values()
               if is_int(p["budget_frames"])}
    ancestors = {p["ancestor_id"] for p in pairs.values()
                 if isinstance(p["ancestor_id"], str)}
    for ancestor in ancestors:
        for budget in budgets:
            table[(ancestor, budget)] = PREREG_CKPTS * PREREG_RNGS
    return table, source


# --------------------------------------------------------------------------- #
# per-endpoint scoring
# --------------------------------------------------------------------------- #
def tally_ancestor(pairs, ancestor, budget):
    cells = as_dict_of_counts()
    complete = []
    unresolved = {"pending": [], "technical_invalid": [], "conflict": []}
    refs = []
    files = set()
    for pair in pairs.values():
        if pair["ancestor_id"] != ancestor or pair["budget_frames"] != budget:
            continue
        for ref in pair["rows"].values():
            files.add(ref.path)
        if pair["status"] == "complete":
            complete.append(pair)
            refs.extend(pair["refs"])
            g1 = pair["full"]["G1"]
            t2 = pair["full"]["T2"]
            if g1 and t2:
                cells["both_full"] += 1
            elif g1 and not t2:
                cells["only_G1"] += 1
            elif t2 and not g1:
                cells["only_T2"] += 1
            else:
                cells["neither"] += 1
        else:
            bucket = unresolved.get(pair["status"])
            if bucket is not None:
                bucket.append(pair["key"])
    n = sum(cells.values())
    diff = None if n == 0 else (cells["only_G1"] - cells["only_T2"]) / n
    n_unresolved = len(unresolved["pending"]) + len(unresolved["technical_invalid"]) \
        + len(unresolved["conflict"])
    n_pending_only = len(unresolved["pending"])
    lo = hi = lo_pending = hi_pending = None
    if n:
        lo = (cells["only_G1"] - cells["only_T2"] - n_unresolved) / n
        hi = (cells["only_G1"] - cells["only_T2"] + n_unresolved) / n
        lo_pending = (cells["only_G1"] - cells["only_T2"] - n_pending_only) / n
        hi_pending = (cells["only_G1"] - cells["only_T2"] + n_pending_only) / n
    return {
        "ancestor_id": ancestor,
        "budget_frames": budget,
        "cells": cells,
        "n_valid_pairs": n,
        "diff": diff,
        "diff_pp": None if diff is None else diff * 100.0,
        "diff_fraction": fmt_ratio(cells["only_G1"] - cells["only_T2"], n),
        "sign": (None if diff is None else
                 (1 if diff > 0 else (-1 if diff < 0 else 0))),
        "unresolved": {
            "total": n_unresolved,
            "pending": len(unresolved["pending"]),
            "technical_invalid": len(unresolved["technical_invalid"]),
            "conflict": len(unresolved["conflict"]),
            "items": [{"key": list(key), "status": status}
                      for status, keys in unresolved.items() for key in keys],
        },
        "bounds": {
            "lo": lo, "hi": hi,
            "formula": ("lo = 全部未决对都判给 T2 的极值 = "
                        "(only_G1 - only_T2 - u) / n_valid_pairs；hi = 全部未决对都判给 "
                        "G1 的极值 = (only_G1 - only_T2 + u) / n_valid_pairs"),
            "lo_pending_only": lo_pending, "hi_pending_only": hi_pending,
        },
        "pairs_complete": len(complete),
        "pairs_observed": len(complete) + n_unresolved,
        "row_refs": sorted(refs),
        "files": sorted(files),
    }


def arm_absolute_rate(pairs, ancestor, budget, arm):
    full = 0
    n = 0
    refs = []
    files = set()
    for pair in pairs.values():
        if pair["ancestor_id"] != ancestor or pair["budget_frames"] != budget:
            continue
        ref = pair["rows"].get(arm)
        if ref is None:
            continue
        status = pair["class_" + arm]["status"]
        if status != "scored":
            continue
        n += 1
        refs.append(ref.ref)
        files.add(ref.path)
        if pair["full"].get(arm):
            full += 1
    return {"full": full, "n": n, "rate": (full / n if n else None),
            "row_refs": sorted(refs), "files": sorted(files)}


def build_endpoint(pairs, budget, expected_ancestors, expected_pairs):
    """All sec.4 numbers for one budget (1,800 or 2,400 frames)."""
    per_ancestor = []
    for ancestor in expected_ancestors:
        per_ancestor.append(tally_ancestor(pairs, ancestor, budget))
    available = [item for item in per_ancestor if item["n_valid_pairs"] > 0]
    missing = [item["ancestor_id"] for item in per_ancestor
               if item["n_valid_pairs"] == 0]
    diffs = [item["diff"] for item in per_ancestor if item["diff"] is not None]
    positive = [item["ancestor_id"] for item in per_ancestor
                if item["diff"] is not None and item["diff"] > 0]
    negative = [item["ancestor_id"] for item in per_ancestor
                if item["diff"] is not None and item["diff"] < 0]
    tie = [item["ancestor_id"] for item in per_ancestor
           if item["diff"] is not None and item["diff"] == 0]

    delta = None
    delta_note = None
    if diffs:
        # exact rational mean of the per-ancestor rational differences, rounded
        # once: sum(diffs)/len(diffs) is 1-ULP dependent on the interpreter.
        delta = exact_mean([exact_diff(item) for item in per_ancestor])
        delta_note = (f"equal-weight mean of {len(diffs)} ancestor difference(s)"
                      + (f"; 只有 {len(diffs)}/{len(expected_ancestors)} 个祖先在该档"
                         f"有完整配对，属阶段性数字" if missing else
                         " (sec.4: 完整清单上 8 个祖先差值的等权平均)"))
    elif not diffs:
        delta_note = (f"no ancestor has a complete pair at {budget} frames")

    loo = None
    if delta is not None and len(diffs) >= 2:
        exact_runs = [exact_diff(item) for item in per_ancestor]
        values = []
        means = []
        for index, item in enumerate(per_ancestor):
            if exact_runs[index] is None:
                continue
            rest = [value for pos, value in enumerate(exact_runs)
                    if pos != index and value is not None]
            mean = sum(rest, Fraction(0)) / len(rest)
            means.append(mean)
            values.append({"dropped": item["ancestor_id"],
                           "n_remaining": len(rest),
                           "mean_remaining": float(mean),
                           "mean_remaining_exact": fmt_exact(mean)})
        low = min(means)
        high = max(means)
        # every ancestor whose run reaches the extreme, sorted; the primary name
        # is the lexicographically first of that set (documented in the module
        # docstring, --help and loo["tie_policy"]).
        min_dropped = sorted(value["dropped"] for value, mean in zip(values, means)
                             if mean == low)
        max_dropped = sorted(value["dropped"] for value, mean in zip(values, means)
                             if mean == high)
        loo = {
            "method": ("leave-one-out: drop one ancestor, average the remaining "
                       "k-1; min/max over those k runs (not a standard deviation); "
                       "exact rational means, every tied run listed"),
            "k": len(values),
            "min": float(low), "min_dropped": min_dropped[0],
            "min_dropped_primary": min_dropped[0],
            "min_dropped_ties": min_dropped,
            "max": float(high), "max_dropped": max_dropped[0],
            "max_dropped_primary": max_dropped[0],
            "max_dropped_ties": max_dropped,
            "span": float(high - low),
            "tie_policy": (
                "min_dropped/max_dropped keep their historical meaning: the "
                "deterministic primary name = lexicographically first ancestor id "
                "among the runs that reach the extreme.  The full tie set is "
                "min_dropped_ties/max_dropped_ties (sorted); "
                "min_dropped_primary/max_dropped_primary are the same primary "
                "spelled out.  Ties are exact rational equalities, never float "
                "coincidences."),
            "values": values,
        }

    rates = {}
    rows_used = 0
    files = set()
    # absolute per-arm full-segment rate over the whole endpoint (all ancestors),
    # keeping provenance for every numerator and denominator.
    for arm in ARMS:
        full = n = 0
        refs = []
        arm_files = set()
        for pair in pairs.values():
            if pair["budget_frames"] != budget:
                continue
            if pair["ancestor_id"] not in expected_ancestors:
                continue
            ref = pair["rows"].get(arm)
            if ref is None:
                continue
            if pair["class_" + arm]["status"] != "scored":
                continue
            n += 1
            refs.append(ref.ref)
            arm_files.add(ref.path)
            if pair["full"].get(arm):
                full += 1
        rates[arm] = {"full": full, "n": n,
                      "rate": (full / n if n else None),
                      "row_refs": sorted(refs), "files": sorted(arm_files),
                      "definition": ("绝对满段率 = full_segment 行数 / 该臂在该档"
                                     "可计分行数（含未配成对的行，排除技术无效与"
                                     "待判行）")}
        rows_used += n
        files |= arm_files

    bounds = {
        "lo": None, "hi": None,
        "formula": ("逐祖先 lo = (only_G1 - only_T2 - u)/n（未决对全判给 T2），"
                    "hi = (only_G1 - only_T2 + u)/n（未决对全判给 G1），u 为该祖先的"
                    "未决对数；端点界限 = 各祖先界限的等权平均"),
        "unresolved_pairs": sum(item["unresolved"]["total"] for item in per_ancestor),
        "pending_pairs": sum(item["unresolved"]["pending"] for item in per_ancestor),
        "technical_invalid_pairs": sum(item["unresolved"]["technical_invalid"]
                                       for item in per_ancestor),
        "conflict_pairs": sum(item["unresolved"]["conflict"] for item in per_ancestor),
        "lo_pending_only": None, "hi_pending_only": None,
        "per_ancestor": {item["ancestor_id"]: item["bounds"]
                         for item in per_ancestor},
    }
    with_n = [item for item in per_ancestor if item["n_valid_pairs"] > 0]
    if with_n:
        # exact rational means (one rounding at the end), see exact_mean()
        exact = [exact_bounds(item) for item in with_n]
        bounds["lo"] = exact_mean([value[0] for value in exact])
        bounds["hi"] = exact_mean([value[1] for value in exact])
        bounds["lo_pending_only"] = exact_mean([value[2] for value in exact])
        bounds["hi_pending_only"] = exact_mean([value[3] for value in exact])
        bounds["n_ancestors"] = len(with_n)

    # design completeness against the frozen expectation
    observed_pairs = sum(item["pairs_observed"] for item in per_ancestor)
    complete_pairs = sum(item["pairs_complete"] for item in per_ancestor)
    expected_total = sum(expected_pairs.get((ancestor, budget), 0)
                         for ancestor in expected_ancestors)
    ckpt_groups = sorted({p["ckpt_sha256"] for p in pairs.values()
                          if p["budget_frames"] == budget
                          and p["ancestor_id"] in expected_ancestors
                          and isinstance(p["ckpt_sha256"], str)})
    design = {
        "expected_pairs": expected_total,
        "complete_pairs": complete_pairs,
        "observed_pairs": observed_pairs,
        "shortfall": max(0, expected_total - complete_pairs),
        "unresolved_pairs": sum(item["unresolved"]["total"]
                                for item in per_ancestor),
        "complete": (bool(expected_total)
                     and complete_pairs >= expected_total
                     and sum(item["unresolved"]["total"]
                             for item in per_ancestor) == 0),
        "checkpoint_groups": len(ckpt_groups),
        "checkpoint_shas": ckpt_groups,
    }

    outside = sum(1 for pair in pairs.values()
                  if pair["budget_frames"] == budget
                  and pair["ancestor_id"] not in expected_ancestors)
    return {
        "budget_frames": budget,
        "budget_decisions": budget // DECISION_FRAMES_DEFAULT,
        "name": BUDGET_NAMES.get(budget, f"{budget}fr"),
        "title_cn": BUDGET_CN.get(budget, f"{budget} 帧"),
        "per_ancestor": per_ancestor,
        "available": [item["ancestor_id"] for item in available],
        "missing": missing,
        "rows_used": rows_used,
        "files": sorted(files),
        "pairs_outside_expected": outside,
        "delta": delta,
        "delta_pp": None if delta is None else delta * 100.0,
        "delta_note": delta_note,
        "counts": {"positive": len(positive), "tie": len(tie),
                   "negative": len(negative),
                   "positive_ancestors": positive, "tie_ancestors": tie,
                   "negative_ancestors": negative},
        "loo": loo,
        "rates": rates,
        "bounds": bounds,
        "design": design,
        "coverage": {
            "expected_ancestors": len(expected_ancestors),
            "available_ancestors": len(available),
            "missing_ancestors": missing,
            "complete": not missing and bool(expected_ancestors),
        },
        "emitted": False,
        "withheld_reason": None,
    }


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def build_checkpoints(pairs, expected_ancestors, seed_hints, args):
    groups: dict = {}
    for pair in pairs.values():
        sha = pair["ckpt_sha256"]
        if not isinstance(sha, str):
            continue
        groups.setdefault(sha, []).append(pair)
    out = {"groups": [], "expected": len(seed_hints) if seed_hints else PREREG_CKPTS,
           "observed": len(groups),
           "seed_hints": list(seed_hints or []),
           "complete": None,
           "note": ("按 ckpt_sha256 分组；每组用与主表相同的 2x2 与等权平均，"
                    "分母是该组实际覆盖的祖先数")}
    ordered = sorted(groups.items(), key=lambda item: item[0])
    for index, (sha, members) in enumerate(ordered):
        paths = sorted({str(pair["rows"][arm].row.get("ckpt_path"))
                        for pair in members for arm in pair["rows"]
                        if pair["rows"][arm].row.get("ckpt_path")})
        hint = None
        for path in paths:
            found = _SEED_RE.search(path)
            if found:
                hint = int(found.group(1))
                break
        ancestors = sorted({pair["ancestor_id"] for pair in members})
        entry = {"index": index, "ckpt_sha256": sha, "ckpt_paths": paths,
                 "seed_hint": hint, "ancestors": ancestors, "endpoints": {}}
        for budget in (PRIMARY_BUDGET, SECONDARY_BUDGET):
            subset = {key: pair for key, pair in pairs.items()
                      if pair["ckpt_sha256"] == sha
                      and pair["budget_frames"] == budget}
            per = []
            for ancestor in expected_ancestors:
                per.append(tally_ancestor(subset, ancestor, budget))
            diffs = [item["diff"] for item in per if item["diff"] is not None]
            missing = [item["ancestor_id"] for item in per
                       if item["n_valid_pairs"] == 0]
            covered = [item for item in per if item["n_valid_pairs"] > 0]
            # exact rational means, same as build_endpoint (one rounding at the end)
            delta = None if (missing or not diffs) else exact_mean(
                [exact_diff(item) for item in per])
            exact = [exact_bounds(item) for item in covered]
            rates = {}
            for arm in ARMS:
                full = n = 0
                refs = []
                arm_files = set()
                for pair in subset.values():
                    ref = pair["rows"].get(arm)
                    if ref is None or pair["class_" + arm]["status"] != "scored":
                        continue
                    if pair["ancestor_id"] not in expected_ancestors:
                        continue
                    n += 1
                    refs.append(ref.ref)
                    arm_files.add(ref.path)
                    if pair["full"].get(arm):
                        full += 1
                rates[arm] = {"full": full, "n": n,
                              "rate": (full / n if n else None),
                              "row_refs": sorted(refs), "files": sorted(arm_files)}
            entry["endpoints"][str(budget)] = {
                "budget_frames": budget,
                "per_ancestor": per,
                "delta": delta,
                "delta_pp": None if delta is None else delta * 100.0,
                "available_ancestors": len([p for p in per if p["n_valid_pairs"] > 0]),
                "missing_ancestors": missing,
                "counts": {
                    "positive": sum(1 for d in diffs if d > 0),
                    "tie": sum(1 for d in diffs if d == 0),
                    "negative": sum(1 for d in diffs if d < 0)},
                "rates": rates,
                "bounds": {
                    "lo": None if not diffs or missing else
                    exact_mean([value[0] for value in exact]),
                    "hi": None if not diffs or missing else
                    exact_mean([value[1] for value in exact]),
                },
                "rows_used": sum(rate["n"] for rate in rates.values()),
                "files": sorted({f for rate in rates.values()
                                 for f in rate["files"]}),
            }
        out["groups"].append(entry)
    out["complete"] = (out["observed"] == out["expected"])
    return out


# --------------------------------------------------------------------------- #
# inventory events (sec. 4 last paragraph)
# --------------------------------------------------------------------------- #
def inventory_row(ref: RowRef, cell: dict, score: dict):
    row = ref.row
    out = {
        "row_ref": ref.ref, "arm": cell["arm"],
        "ancestor_id": cell["ancestor_id"], "budget_frames": cell["budget_frames"],
        "pair_rng": cell["pair_rng"], "episode": cell["episode"],
        "terminal": row.get("terminal"),
        "first_death_offset": row.get("first_death_offset"),
        "frames": row.get("frames"),
        "survived": row.get("survived"),
        "status": "ok", "issues": [],
        "population": ("death" if row.get("first_death_offset") is not None
                       else ("survived" if score.get("full_segment") else "unknown")),
        "deductions": [], "exhaustion": None, "death_with_bomb": None,
    }
    issues = out["issues"]

    emits = row.get("bomb_emits")
    lists = {
        "bomb_decisions": row.get("bomb_decisions"),
        "bomb_frames": row.get("bomb_frames"),
        "bombs_remaining_at_emit": row.get("bombs_remaining_at_emit"),
        "bomb_available_at_emit": row.get("bomb_available_at_emit"),
        "emit_decrement_at_list": row.get("emit_decrement_at_list"),
    }
    for name, value in lists.items():
        if not isinstance(value, list):
            issues.append(f"missing:{name}")
    if issues:
        out["insufficient_because"] = list(issues)
        out["status"] = "insufficient"
        return out
    lengths = {name: len(value) for name, value in lists.items()}
    if len(set(lengths.values())) != 1:
        issues.append(f"list_length_mismatch:{lengths}")
        out["insufficient_because"] = list(issues)
        out["status"] = "insufficient"
        return out
    count = lengths["emit_decrement_at_list"]
    if is_int(emits) and emits != count:
        issues.append(f"bomb_emits={emits} != len(bomb_decisions)={count}")
    decisions = lists["bomb_decisions"]
    frames = lists["bomb_frames"]
    remaining = lists["bombs_remaining_at_emit"]
    available = lists["bomb_available_at_emit"]
    decrements = lists["emit_decrement_at_list"]
    decision_frames = cell["decision_frames"]
    previous = None
    for index in range(count):
        mark = decrements[index]
        if mark is None:
            continue
        if not is_int(mark) or mark < 1:
            issues.append(f"emit_decrement_at_list[{index}]={mark!r} is not a "
                          f"positive int")
            continue
        if not is_int(frames[index]) or not is_int(decisions[index]):
            issues.append(f"bomb_frames/bomb_decisions[{index}] not int")
            continue
        absolute = frames[index] + (mark - 1)
        step = decisions[index] + (mark - 1) // decision_frames
        before = remaining[index] if is_int(remaining[index]) else None
        out["deductions"].append({
            "index": index,
            "step": step,
            "decision_index": decisions[index],
            "decrement_frame_in_decision": mark,
            "absolute_frame": absolute,
            "inventory_before": before,
            "inventory_after": None if before is None else before - 1,
            "available_at_emit": available[index],
            "gap_frames_from_previous": (None if previous is None
                                         else absolute - previous),
            "formula": ("absolute_frame = bomb_frames[i] + "
                        "emit_decrement_at_list[i] - 1; step = bomb_decisions[i] "
                        "+ (emit_decrement_at_list[i]-1)//decision_frames"),
        })
        previous = absolute
    bombs_used = row.get("bombs_used")
    if is_int(bombs_used) and len(out["deductions"]) != bombs_used:
        issues.append(f"bombs_used={bombs_used} != honoured "
                      f"emit_decrement_at_list entries={len(out['deductions'])}")
    bomb_ignored = row.get("bomb_ignored")
    if is_int(bomb_ignored) and is_int(emits) and is_int(bombs_used) \
            and bomb_ignored != emits - bombs_used:
        issues.append(f"bomb_ignored={bomb_ignored} != bomb_emits {emits} - "
                      f"bombs_used {bombs_used}")
    if issues:
        out["status"] = "insufficient"

    bombs_start = row.get("bombs_start")
    bombs_end = row.get("bombs_end")
    out["bombs_start"] = bombs_start
    out["bombs_end"] = bombs_end
    out["bombs_used"] = bombs_used
    out["bomb_emits"] = emits
    out["honoured"] = len(out["deductions"])
    out["emits_available_true"] = sum(1 for value in available if value is True)

    # frames alive after the inventory reached 0
    if bombs_end is None:
        out["exhaustion"] = {"status": "insufficient",
                             "reason": "bombs_end is missing: whether the inventory "
                                       "ran out cannot be decided"}
    elif bombs_end == 0:
        if not out["deductions"]:
            out["exhaustion"] = {"status": "insufficient",
                                 "reason": "bombs_end==0 but no honoured deduction"}
        else:
            last = out["deductions"][-1]
            zero_frame = last["absolute_frame"]
            confirmed = (is_int(last["inventory_before"])
                         and last["inventory_before"] - 1 == 0)
            death = row.get("first_death_offset")
            frames_total = row.get("frames")
            if not confirmed:
                out["exhaustion"] = {
                    "status": "insufficient",
                    "reason": (f"inventory after the last honoured deduction is "
                               f"{None if last['inventory_before'] is None else last['inventory_before'] - 1}"
                               f" but bombs_end==0: the drop to 0 has no recorded "
                               f"event"),
                    "zero_frame": zero_frame,
                }
            elif is_int(death):
                out["exhaustion"] = {
                    "status": "ok", "zero_frame": zero_frame,
                    "alive_after_zero": death - zero_frame,
                    "truncated": False, "end_offset": death,
                    "formula": ("first_death_offset - zero_frame; zero_frame = "
                                "absolute frame of the deduction that took inventory "
                                "to 0"),
                }
            elif is_int(frames_total):
                out["exhaustion"] = {
                    "status": "ok", "zero_frame": zero_frame,
                    "alive_after_zero": (frames_total - 1) - zero_frame,
                    "truncated": True, "end_offset": frames_total - 1,
                    "truncation_note": ("survived: the segment ends at frame "
                                        f"{frames_total - 1} while the budget is "
                                        f"{cell['budget_frames']}; the tail after "
                                        f"inventory 0 is truncated there, not "
                                        f"observed to death"),
                    "formula": ("(frames - 1) - zero_frame, truncated at the "
                                "segment end of a surviving episode"),
                }
            else:
                out["exhaustion"] = {"status": "insufficient",
                                     "reason": "no death offset and no frames"}
    else:
        out["exhaustion"] = {"status": "not_exhausted", "bombs_end": bombs_end}

    # deaths with a bomb still in stock
    death = row.get("first_death_offset")
    if is_int(death):
        frames_total = row.get("frames")
        method = None
        value = None
        if not is_int(bombs_end):
            method = "insufficient: bombs_end is missing"
        elif is_int(frames_total) and death == frames_total - 1:
            value = bombs_end
            method = "bombs_end_at_death_frame (bombs_end is measured at the " \
                     "episode's last frame, which is the death frame)"
        elif is_int(bombs_start) and is_int(bombs_used) \
                and bombs_used == len(out["deductions"]):
            monotone = all(out["deductions"][i]["inventory_before"] is not None
                           and out["deductions"][i + 1]["inventory_before"]
                           is not None
                           and out["deductions"][i]["inventory_before"]
                           >= out["deductions"][i + 1]["inventory_before"]
                           for i in range(len(out["deductions"]) - 1))
            if monotone:
                upto = sum(1 for item in out["deductions"]
                           if item["absolute_frame"] <= death)
                value = bombs_start - upto
                method = ("derive: bombs_start - honoured deductions with "
                          "absolute_frame <= first_death_offset (inventory series "
                          "monotone, no pick-up evidence)")
            else:
                method = ("insufficient: death is not the segment's last frame and "
                          "the inventory series is not monotone (pick-up possible)")
        else:
            method = ("insufficient: death is not the segment's last frame and "
                      "bombs_start/bombs_used do not allow a derivation")
        out["death_with_bomb"] = {
            "status": "ok" if value is not None else "insufficient",
            "bombs_at_death": value,
            "with_bomb": None if value is None else value > 0,
            "method": method,
        }
    # a row whose inventory bookkeeping is inconsistent, or whose exhaustion /
    # death-with-bomb quantity cannot be derived, is reported as "insufficient"
    # (不足) rather than being silently reduced to what happens to be computable.
    sub = []
    if isinstance(out["exhaustion"], dict) and out["exhaustion"].get("status") == "insufficient":
        sub.append("exhaustion: " + str(out["exhaustion"].get("reason")))
    if isinstance(out["death_with_bomb"], dict) \
            and out["death_with_bomb"].get("status") == "insufficient":
        sub.append("death_with_bomb: " + str(out["death_with_bomb"].get("method")))
    out["insufficient_because"] = list(out["issues"]) + sub
    if out["insufficient_because"]:
        out["status"] = "insufficient"
    return out


def build_inventory(rows, identities, pairs):
    details = []
    used_files = set()
    for pair in pairs.values():
        if pair["status"] == "conflict":
            continue
        for arm in ARMS:
            ref = pair["rows"].get(arm)
            if ref is None:
                continue
            cell = identities[ref.ref]
            score = classify_row(ref, cell)
            detail = inventory_row(ref, cell, score)
            detail["pair_status"] = pair["status"]
            detail["row_status"] = score["status"]
            details.append(detail)
            used_files.add(ref.path)
    groups = {}
    for name in ("survived", "death", "unknown"):
        members = [item for item in details if item["population"] == name]
        deductions = [d for item in members for d in item["deductions"]]
        exhausted = [item["exhaustion"] for item in members
                     if isinstance(item["exhaustion"], dict)
                     and item["exhaustion"].get("status") == "ok"]
        with_bomb = [item for item in members
                     if isinstance(item["death_with_bomb"], dict)
                     and item["death_with_bomb"].get("with_bomb")]
        groups[name] = {
            "rows": len(members),
            "rows_insufficient": sum(1 for item in members
                                     if item["status"] == "insufficient"),
            "deductions": len(deductions),
            "deductions_per_row": (len(deductions) / len(members) if members else None),
            "exhausted_rows": len(exhausted),
            "alive_after_zero": [item["alive_after_zero"] for item in exhausted],
            "alive_after_zero_mean": (sum(item["alive_after_zero"]
                                          for item in exhausted) / len(exhausted)
                                      if exhausted else None),
            "alive_after_zero_truncated": sum(1 for item in exhausted
                                              if item.get("truncated")),
            "death_with_bomb": len(with_bomb),
            "death_with_bomb_refs": [item["row_ref"] for item in with_bomb],
            "bombs_at_death_insufficient": sum(
                1 for item in members
                if isinstance(item.get("death_with_bomb"), dict)
                and item["death_with_bomb"].get("status") == "insufficient"),
        }
    totals = {
        "rows": len(details),
        "deductions": sum(len(item["deductions"]) for item in details),
        "exhausted_rows": sum(1 for item in details
                              if isinstance(item["exhaustion"], dict)
                              and item["exhaustion"].get("status") == "ok"),
        "death_with_bomb": sum(1 for item in details
                               if isinstance(item["death_with_bomb"], dict)
                               and item["death_with_bomb"].get("with_bomb")),
        "rows_insufficient": sum(1 for item in details
                                 if item["status"] == "insufficient"),
        "emits": sum(item.get("bomb_emits") or 0 for item in details),
        "honoured": sum(item.get("honoured") or 0 for item in details),
        "emits_available_true": sum(item.get("emits_available_true") or 0
                                    for item in details),
        "available_vs_honoured_note": (
            "bomb_available_at_emit 只说明请求帧上雷是否可用，不说明是否兑现；"
            "按 REPLY_02 §3 的扣减口径，honoured = emit_decrement_at_list 非 None 的个数。"
            "两者在本批数据里数值相同，但这只是巧合，不能互相替代（一个决策内敌人"
            "可能进入无敌帧，或雷在同一决策内变得可用）。"),
    }
    return {
        "definition": {
            "deduction": ("每次库存真的下降的请求：emit_decrement_at_list[i] 非 None；"
                          "绝对帧 = bomb_frames[i] + emit_decrement_at_list[i] - 1"
                          "（REPLY_02 §3: emit_decrement_at=2，扣减在决策第 2 帧观测），"
                          "步 = bomb_decisions[i] + (mark-1)//decision_frames"),
            "exhaustion": ("bombs_end==0 时，最后一颗实际扣减的绝对帧到 first_death_offset"
                           "（死亡局）或到段尾 frames-1（幸存局，截断并注明）的帧数"),
            "death_with_bomb": ("first_death_offset 非空且死亡时库存 >0；库存取 "
                                "bombs_end（死亡即段末帧）或按单调序列推导，推不出记"
                                " insufficient（不足）"),
        },
        "groups": groups,
        "totals": totals,
        "files": sorted(used_files),
        "detail": details,
    }


# --------------------------------------------------------------------------- #
# verdict (sec.4 decision table)
# --------------------------------------------------------------------------- #
def decide(entry, args, gate_reasons, *, role):
    """role is 'primary' (Delta30) or 'secondary' (Delta40)."""
    n_expected = max(1, int(args.expected_ancestors))
    need_positive = max(1, math.ceil(PREREG_MIN_POSITIVE[0] * n_expected
                                     / PREREG_MIN_POSITIVE[1]))
    threshold = PREREG_THRESHOLD_PP / 100.0
    available = len(entry["available"]) if entry else 0
    role_cn = "主要终点 Δ30" if role == "primary" else "次要终点 Δ40"
    base = {
        "role": role,
        "role_cn": role_cn,
        "budget_frames": entry["budget_frames"] if entry else None,
        "threshold_pp": PREREG_THRESHOLD_PP, "min_positive": need_positive,
        "n_ancestors_expected": n_expected,
        "available_denominator": available,
        "delta": entry["delta"] if entry else None,
        "delta_pp": entry["delta_pp"] if entry else None,
        "counts": entry["counts"] if entry else None,
        "bounds": (None if entry is None else {"lo": entry["bounds"]["lo"],
                                               "hi": entry["bounds"]["hi"]}),
        "reasons": list(gate_reasons),
        "decision": None,
        "rule": ("sec.4 表：Δ ≥ +5 pp 且 ≥5/8 正 → 提名 G1；Δ ≤ −5 pp 且 ≥5/8 负 "
                 "→ 保留 T2；其余完整结果 → 差异未定；祖先不足/身份不符/关键计分缺证/"
                 "主配对不完整 → 报可用分母及界限"),
    }
    if gate_reasons:
        base["decision"] = "REFUSED_INSUFFICIENT_DENOMINATOR"
        base["decision_cn"] = "拒绝出判定：按预注册 §4 第四行只报可用分母及界限"
        return base
    if entry is None or entry["delta"] is None:
        base["decision"] = "REFUSED_INSUFFICIENT_DENOMINATOR"
        base["decision_cn"] = "拒绝出判定：该档 Δ 未出（见 withheld_reason）"
        base["reasons"] = base["reasons"] + [entry["withheld_reason"]
                                            if entry else "no endpoint"]
        return base
    delta = entry["delta"]
    positive = entry["counts"]["positive"]
    negative = entry["counts"]["negative"]
    if delta >= threshold and positive >= need_positive:
        base["decision"] = "NOMINATE_G1"
        base["decision_cn"] = ("提名 G1 进入另一批独立祖先的 30 秒复验"
                               if role == "primary" else
                               "仅当 40 秒批完整且达到同一门槛时建议同时复验 40 秒"
                               "（次要终点）")
    elif delta <= -threshold and negative >= need_positive:
        base["decision"] = "KEEP_T2"
        base["decision_cn"] = ("保留 T2 作 30 秒默认参照，复查 G1 在哪些阶段消耗过早"
                               "或带雷死亡")
    else:
        base["decision"] = "UNDETERMINED"
        base["decision_cn"] = ("两者差异未定，T2 保留为简单对照：先看逐祖先、逐阶段"
                               "和库存事件")
    return base


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def md_table(headers, rows, align=None):
    align = align or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(align) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def render_markdown(document) -> str:
    lines = []
    add = lines.append
    prov = document["provenance"]
    add(f"# V829-PAIR-01 §4 计分报告（{document['generated_at']}）")
    add("")
    add(f"> 由 `{TOOL}` v{TOOL_VERSION}（SHA-256 `{document['tool_sha256']}`）"
        f"离线生成；不启动引擎。报告 schema `{document['schema']}`。")
    add("")
    add("## 0. 结论")
    add("")
    verdict = document["verdict"]
    add(f"- 退出码：**{document['exit_code']}**"
        f"（0=全部通过；1=有门未过，仍出分母与界限；2=用法/输入错误）")
    add(f"- Δ30 判定：**{verdict['primary_30s']['decision']}** —— "
        f"{verdict['primary_30s']['decision_cn']}")
    add(f"- Δ40 判定：**{verdict['secondary_40s']['decision']}** —— "
        f"{verdict['secondary_40s']['decision_cn']}")
    add(f"- 作用范围：{document['scope']['note']}")
    if document["gates"]["refusals"]:
        add("")
        add("**拒绝项（fail-closed）**")
        add("")
        for item in document["gates"]["refusals"]:
            add(f"- `{item['code']}` {item['message']}")
    add("")
    add("## 1. 输入与可追溯性")
    add("")
    add(md_table(
        ["文件", "字节", "SHA-256", "行数"],
        [[f"`{item['path']}`", fmt_int(item["bytes"]),
          f"`{item['sha256']}`", fmt_int(item["rows_read"])]
         for item in prov["files"]],
        ["---", "---:", "---", "---:"]))
    add("")
    add(f"- 读入行数 **{fmt_int(prov['rows_total'])}**；参与计分 "
        f"{fmt_int(prov['rows_scored'])}；技术无效 {fmt_int(prov['rows_technical_invalid'])}；"
        f"待判 {fmt_int(prov['rows_pending'])}；被后一次重跑取代 "
        f"{fmt_int(prov['rows_superseded'])}。")
    add(f"- 配对单元（祖先×档×ckpt×pair_rng×episode）**{fmt_int(prov['pairs_total'])}** 个："
        f"完整 {fmt_int(prov['pairs_complete'])}，技术无效 {fmt_int(prov['pairs_technical_invalid'])}，"
        f"待判 {fmt_int(prov['pairs_pending'])}，冲突 {fmt_int(prov['pairs_conflict'])}。")
    add(f"- 逐行证据索引见 `--json` 的 `rows[]`（每行含 `ref`=文件#行号、"
        f"`line_sha256_16`、臂/祖先/档/ckpt/pair_rng、计分状态与 full_segment，"
        f"共 {len(document['rows'])} 条）。")
    if prov.get("manifest"):
        man = prov["manifest"]
        values = ", ".join(f"`{item['value'][:16]}…`×{item['rows']}"
                           for item in man["row_manifest_sha256_values"])
        add(f"- 清单 `{man['path']}`：{man['bytes']} B，SHA-256 `{man['sha256']}`，"
            f"`manifest_id={man.get('manifest_id')}`，roots={man.get('root_count')}；"
            f"行内 manifest_sha256 取值 {values}")
    add("")
    add("## 2. 逐祖先 2×2")
    for key in document["endpoint_order"]:
        entry = document["endpoints"][key]
        add("")
        add(f"### {entry['title_cn']}（budget_frames={entry['budget_frames']}）")
        add("")
        rows = []
        for item in entry["per_ancestor"]:
            cells = item["cells"]
            rows.append([
                f"`{item['ancestor_id']}`", fmt_int(item["n_valid_pairs"]),
                fmt_int(cells["both_full"]), fmt_int(cells["only_G1"]),
                fmt_int(cells["only_T2"]), fmt_int(cells["neither"]),
                fmt_frac(item["diff"]), fmt_ppv(item["diff_pp"]),
                fmt_int(item["unresolved"]["total"]),
                fmt_frac(item["bounds"]["lo"]), fmt_frac(item["bounds"]["hi"]),
            ])
        add(md_table(
            ["祖先", "有效配对 n", "同满", "仅 G1 满", "仅 T2 满", "同未满",
             "差值", "差值(pp)", "未决对", "下界", "上界"],
            rows, ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:",
                   "---:", "---:", "---:"]))
        add("")
        add(f"差值公式：`(only_G1 - only_T2) / n_valid_pairs`；本表用 "
            f"{fmt_int(sum(i['n_valid_pairs'] for i in entry['per_ancestor']))} 个有效配对，"
            f"来自 {len(entry['files'])} 个文件：{', '.join('`'+f+'`' for f in entry['files'])}")
        add("")
        add(f"Δ = **{fmt_frac(entry['delta'])}**（{fmt_ppv(entry['delta_pp'])}）；"
            f"{entry['delta_note']}")
        add(f"正/平/负祖先：{entry['counts']['positive']}/"
            f"{entry['counts']['tie']}/{entry['counts']['negative']}"
            f"（正：{entry['counts']['positive_ancestors'] or '无'}；"
            f"负：{entry['counts']['negative_ancestors'] or '无'}）")
        if entry["loo"]:
            loo = entry["loo"]
            add(f"去一法区间：[{fmt_frac(loo['min'])}, {fmt_frac(loo['max'])}]"
                f"（{fmt_pp(loo['min'])} … {fmt_pp(loo['max'])}；跨度 "
                f"{fmt_pp(loo['span'])}；去掉 "
                f"{fmt_tie_names(loo['min_dropped_ties'])} 得下界、去掉 "
                f"{fmt_tie_names(loo['max_dropped_ties'])} 得上界；"
                f"k={loo['k']} 次留一，非标准差）")
            add("")
            add(md_table(
                ["极值", "值", "达到该极值的祖先（全部并列者）", "确定性主名"],
                [["下界", fmt_frac(loo["min"]),
                  fmt_tie_names(loo["min_dropped_ties"]),
                  f"`{loo['min_dropped_primary']}`"],
                 ["上界", fmt_frac(loo["max"]),
                  fmt_tie_names(loo["max_dropped_ties"]),
                  f"`{loo['max_dropped_primary']}`"]],
                ["---", "---:", "---", "---"]))
            add("")
            extreme_flags: dict = {}
            for name in loo["min_dropped_ties"]:
                extreme_flags.setdefault(name, []).append("下界")
            for name in loo["max_dropped_ties"]:
                extreme_flags.setdefault(name, []).append("上界")
            add(md_table(
                ["去掉的祖先", "剩余 k-1 个祖先的等权平均", "精确分数", "极值"],
                [[f"`{v['dropped']}`", fmt_frac(v["mean_remaining"]),
                  f"`{v.get('mean_remaining_exact', 'n/a')}`",
                  "、".join(extreme_flags.get(v["dropped"], [])) or "—"]
                 for v in loo["values"]], ["---", "---:", "---:", "---"]))
            add("")
            add(f"并列口径：`min_dropped`/`max_dropped` 是**确定性主名**（并列集合中"
                f"祖先 id 字典序第一个），`*_dropped_ties` 是按 id 排序的**全部并列者**，"
                f"`*_dropped_primary` 与主名相同。本次下界 "
                f"{len(loo['min_dropped_ties'])} 个并列、上界 "
                f"{len(loo['max_dropped_ties'])} 个并列；均值按精确有理数计算、末尾"
                f"只舍入一次，所以并列是数学相等而不是浮点巧合。")
        add("")
        rates = entry["rates"]
        add(md_table(
            ["臂", "满段行数", "可计分行数", "绝对满段率", "来源"],
            [[arm, fmt_int(rates[arm]["full"]), fmt_int(rates[arm]["n"]),
              fmt_rate(rates[arm]["rate"]),
              f"{len(rates[arm]['files'])} 文件"]
             for arm in ARMS], ["---", "---:", "---:", "---:", "---"]))
        add("")
        add(f"界限（全偏向一侧的极值，未决对 "
            f"{entry['bounds']['unresolved_pairs']} 个：待判 "
            f"{entry['bounds']['pending_pairs']}、技术无效 "
            f"{entry['bounds']['technical_invalid_pairs']}、冲突 "
            f"{entry['bounds']['conflict_pairs']}；"
            f"基于 {fmt_int(sum(i['pairs_complete'] for i in entry['per_ancestor']))} "
            f"个完整配对，来自 {len(entry['files'])} 个文件）："
            f"[{fmt_frac(entry['bounds']['lo'])}, {fmt_frac(entry['bounds']['hi'])}]；"
            f"只算待判对时 "
            f"[{fmt_frac(entry['bounds']['lo_pending_only'])}, "
            f"{fmt_frac(entry['bounds']['hi_pending_only'])}]")
        add("")
        design = entry["design"]
        add(f"设计完整度：应有 {design['expected_pairs']} 对、完整 {design['complete_pairs']} 对"
            f"（缺 {design['shortfall']}），checkpoint 组 {design['checkpoint_groups']} 个；"
            f"覆盖 {entry['coverage']['available_ancestors']}/"
            f"{entry['coverage']['expected_ancestors']} 个祖先"
            + (f"，缺祖先 {', '.join(entry['coverage']['missing_ancestors'])}"
               if entry["coverage"]["missing_ancestors"] else "")
            + (f"，另有 {entry['pairs_outside_expected']} 个配对单元因祖先不在"
               f"预期集合内被排除"
               if entry.get("pairs_outside_expected") else ""))
        if not entry["emitted"]:
            add("")
            add(f"**Δ 未出（fail-closed）**：{entry['withheld_reason']}")
    add("")
    add("## 3. 三份 checkpoint")
    add("")
    ck = document["checkpoints"]
    add(f"按 `ckpt_sha256` 分组：应有 {ck['expected']} 组，实测 {ck['observed']} 组"
        f"（seed hints：{ck['seed_hints'] or '未提供'}）。")
    add("")
    for group in ck["groups"]:
        add(f"### ckpt#{group['index']} `{group['ckpt_sha256']}`"
            + (f"（seed {group['seed_hint']}）" if group["seed_hint"] is not None else ""))
        add("")
        add(f"路径：{', '.join('`'+p+'`' for p in group['ckpt_paths']) or 'n/a'}；"
            f"覆盖祖先 {len(group['ancestors'])} 个。")
        add("")
        rows = []
        for key in document["endpoint_order"]:
            entry = group["endpoints"][key]
            rows.append([entry["budget_frames"], fmt_int(entry["rows_used"]),
                         fmt_frac(entry["delta"]), fmt_ppv(entry["delta_pp"]),
                         entry["counts"]["positive"], entry["counts"]["tie"],
                         entry["counts"]["negative"],
                         f"{entry['available_ancestors']}",
                         fmt_frac(entry["bounds"]["lo"]),
                         fmt_frac(entry["bounds"]["hi"])])
        add(md_table(["档", "行数", "Δ", "Δ(pp)", "正", "平", "负",
                      "可用祖先", "下界", "上界"], rows,
                     ["---:", "---:", "---:", "---:", "---:", "---:", "---:",
                      "---:", "---:", "---:"]))
        add("")
    add("## 4. 技术无效与待判")
    add("")
    add(md_table(["类别", "行数", "配对单元数"],
                 [["技术无效（invalid/error 非空）",
                   fmt_int(prov["rows_technical_invalid"]),
                   fmt_int(prov["pairs_technical_invalid"])],
                  ["待判（缺臂/不可计分/终态不明）",
                   fmt_int(prov["rows_pending"]), fmt_int(prov["pairs_pending"])],
                  ["冲突（同 pair 同 arm / 计分不一致）",
                   fmt_int(prov["rows_conflict"]), fmt_int(prov["pairs_conflict"])]],
                 ["---", "---:", "---:"]))
    add("")
    if document["technical_invalid"]:
        add(md_table(["行", "臂", "祖先", "档", "原因"],
                     [[f"`{item['ref']}`", item["arm"], f"`{item['ancestor_id']}`",
                       item["budget_frames"], item["reason"]]
                      for item in document["technical_invalid"]],
                     ["---", "---", "---", "---:", "---"]))
        add("")
    if document["pending"]:
        add(md_table(["配对单元", "状态", "原因"],
                     [[f"`{item['key']}`", item["status"], item["reason"]]
                      for item in document["pending"]],
                     ["---", "---", "---"]))
        add("")
    add("## 5. 库存事件（预注册 §4 末段）")
    add("")
    inv = document["inventory"]
    add(f"口径：{inv['definition']['deduction']}；{inv['definition']['exhaustion']}；"
        f"{inv['definition']['death_with_bomb']}")
    add("")
    rows = []
    for name, label in (("survived", "存活局"), ("death", "死亡局"),
                        ("unknown", "终态不明")):
        group = inv["groups"][name]
        rows.append([label, fmt_int(group["rows"]), fmt_int(group["deductions"]),
                     fmt_int(group["exhausted_rows"]),
                     fmt_int(group["alive_after_zero_truncated"]),
                     fmt_int(group["death_with_bomb"]),
                     fmt_int(group["bombs_at_death_insufficient"]),
                     fmt_int(group["rows_insufficient"])])
    add(md_table(["分组", "行数", "实际扣库存次数", "耗尽局数", "其中截断",
                  "带雷死亡数", "带雷推不出", "口径不足行"], rows,
                 ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    add("")
    # one compact line per row: the step list is the traceable form
    compact = []
    for item in inv["detail"]:
        steps = ",".join(str(drop["step"]) for drop in item["deductions"]) or "-"
        frames = ",".join(str(drop["absolute_frame"])
                          for drop in item["deductions"]) or "-"
        exhaustion = item["exhaustion"] if isinstance(item["exhaustion"], dict) else {}
        bomb = item["death_with_bomb"] if isinstance(item["death_with_bomb"], dict) else {}
        compact.append([
            f"`{item['row_ref']}`", item["arm"], item["budget_frames"],
            fmt_int(len(item["deductions"])), steps, frames,
            fmt_int(exhaustion.get("alive_after_zero")),
            ("截断" if exhaustion.get("truncated") else
             ("—" if exhaustion.get("status") != "ok" else "至死亡")),
            fmt_int(bomb.get("bombs_at_death")),
            ("是" if bomb.get("with_bomb") else
             ("不足" if bomb.get("status") == "insufficient" else "否")),
        ])
    if compact:
        add(f"逐行库存事件（{len(compact)} 行；步=决策下标，帧=绝对帧，"
            f"帧 = bomb_frames[i] + mark - 1）：")
        add("")
        add(md_table(["行", "臂", "档", "扣库存次数", "步", "绝对帧",
                      "耗尽后存活帧", "耗尽后口径", "死亡时库存", "带雷"],
                     compact, ["---", "---", "---:", "---:", "---", "---",
                               "---:", "---", "---:", "---"]))
        add("")
    detail_rows = []
    for item in inv["detail"]:
        for drop in item["deductions"]:
            detail_rows.append([
                f"`{item['row_ref']}`", item["arm"], item["budget_frames"],
                fmt_int(drop["index"]), fmt_int(drop["step"]),
                fmt_int(drop["absolute_frame"]),
                fmt_int(drop["decrement_frame_in_decision"]),
                fmt_int(drop["inventory_before"]),
                fmt_int(drop["inventory_after"]),
                fmt_int(drop["gap_frames_from_previous"])])
    if detail_rows and len(detail_rows) <= 150:
        add("每颗实际扣库存的步（全量，可核对）：")
        add("")
        add(md_table(["行", "臂", "档", "第几颗", "决策步", "绝对帧", "决策内帧",
                      "扣前", "扣后", "距上一颗帧数"], detail_rows,
                     ["---", "---", "---:", "---:", "---:", "---:", "---:",
                      "---:", "---:", "---:"]))
        add("")
    elif detail_rows:
        add(f"逐颗明细共 {len(detail_rows)} 条，已全量写入 `--json` 的 "
            f"`inventory.detail[].deductions`（上表按行汇总，未丢信息）。")
        add("")
    ex_rows = [[f"`{item['row_ref']}`", item["arm"], item["budget_frames"],
                fmt_int(item["exhaustion"].get("zero_frame")),
                fmt_int(item["exhaustion"].get("alive_after_zero")),
                "是" if item["exhaustion"].get("truncated") else "否",
                item["exhaustion"].get("truncation_note", "")]
               for item in inv["detail"]
               if isinstance(item["exhaustion"], dict)
               and item["exhaustion"].get("status") == "ok"]
    if ex_rows:
        add("耗尽后存活帧数（段尾处截断的已注明）：")
        add("")
        add(md_table(["行", "臂", "档", "库存归零帧", "耗尽后存活帧", "截断", "注"],
                     ex_rows, ["---", "---", "---:", "---:", "---:", "---", "---"]))
        add("")
    bomb_rows = [[f"`{item['row_ref']}`", item["arm"], item["budget_frames"],
                  fmt_int(item["death_with_bomb"].get("bombs_at_death")),
                  "是" if item["death_with_bomb"].get("with_bomb") else "否",
                  item["death_with_bomb"].get("method")]
                 for item in inv["detail"]
                 if isinstance(item["death_with_bomb"], dict)]
    if bomb_rows:
        add("带雷死亡判定：")
        add("")
        add(md_table(["行", "臂", "档", "死亡时库存", "带雷", "依据"],
                     bomb_rows, ["---", "---", "---:", "---:", "---", "---"]))
        add("")
    insufficient = [item for item in inv["detail"]
                    if item["status"] == "insufficient"]
    if insufficient:
        add(f"**库存口径不足的行（{len(insufficient)}，如实报「不足」而非估算）：**")
        add("")
        for item in insufficient:
            reasons = item.get("insufficient_because") or item["issues"] or ["见细节"]
            add(f"- `{item['row_ref']}` {item['arm']}：{'; '.join(reasons)}")
        add("")
    add("## 6. 判定表与门槛")
    add("")
    for key in ("primary_30s", "secondary_40s"):
        item = verdict[key]
        add(f"**{item['role_cn']}**（budget_frames={item['budget_frames']}）："
            f"Δ={fmt_frac(item['delta'])}（{fmt_ppv(item['delta_pp'])}），"
            f"正/平/负={item['counts']['positive'] if item['counts'] else 'n/a'}/"
            f"{item['counts']['tie'] if item['counts'] else 'n/a'}/"
            f"{item['counts']['negative'] if item['counts'] else 'n/a'}，"
            f"门槛 Δ≥{item['threshold_pp']} pp 且 ≥{item['min_positive']}/"
            f"{item['n_ancestors_expected']} 正（或对称的负），"
            f"可用分母 {item['available_denominator']} 个祖先 → **{item['decision']}**")
        if item["reasons"]:
            add(f"  - 原因：{'; '.join(str(r) for r in item['reasons'])}")
    add("")
    add(f"作用范围（预注册 §4 末段逐字）：{document['scope']['note']}")
    add("")
    add("## 7. 方法、公式与口径")
    add("")
    for line in document["methods"]:
        add(f"- {line}")
    add("")
    add("## 8. 门（gates）")
    add("")
    add(md_table(["门", "通过", "说明"],
                 [[f"`{item['code']}`", "✅" if item["ok"] else "❌",
                   item["detail"]] for item in document["gates"]["checks"]],
                 ["---", "---", "---"]))
    add("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# text summary (stdout)
# --------------------------------------------------------------------------- #
def render_text(document) -> str:
    out = []
    add = out.append
    prov = document["provenance"]
    add(f"v829 report (schema {document['schema']}, tool {TOOL_VERSION} "
        f"sha256 {document['tool_sha256'][:16]})")
    add(f"inputs: {len(prov['files'])} file(s), {prov['rows_total']} row(s), "
        f"{prov['pairs_total']} pair cell(s)")
    for item in prov["files"]:
        add(f"  - {item['path']}  {item['rows_read']} row(s)  "
            f"sha256 {item['sha256'][:16]}")
    if prov.get("manifest"):
        man = prov["manifest"]
        add(f"manifest: {man['path']}  sha256 {man['sha256'][:16]}  "
            f"roots={man.get('root_count')}")
    add(f"GATES ok={document['gates']['ok']}")
    for item in document["gates"]["checks"]:
        if not item["ok"]:
            add(f"  [FAIL] {item['code']}: {item['detail']}")
    for key in document["endpoint_order"]:
        entry = document["endpoints"][key]
        add("")
        add(f"[{entry['name']} / {entry['budget_frames']} frames] "
            f"rows={entry['rows_used']} complete_pairs="
            f"{sum(i['pairs_complete'] for i in entry['per_ancestor'])}")
        add(f"  可用分母 = {entry['coverage']['available_ancestors']} 个祖先"
            f"（预注册 §2 要求 {document['prereg']['expected_ancestors']}）")
        if entry["emitted"]:
            add(f"  delta_{entry['name']} = {fmt_frac(entry['delta'])} "
                f"({fmt_ppv(entry['delta_pp'])})  "
                f"pos/tie/neg={entry['counts']['positive']}/"
                f"{entry['counts']['tie']}/{entry['counts']['negative']}")
            if entry["loo"]:
                loo = entry["loo"]
                add(f"  leave-one-out: [{fmt_frac(loo['min'])}, "
                    f"{fmt_frac(loo['max'])}] (drop "
                    f"{', '.join(loo['min_dropped_ties'])} for the low end, "
                    f"{', '.join(loo['max_dropped_ties'])} for the high end; "
                    f"primary {loo['min_dropped_primary']}/"
                    f"{loo['max_dropped_primary']})")
        else:
            add(f"  delta_{entry['name']} = WITHHELD: {entry['withheld_reason']}")
        for arm in ARMS:
            rate = entry["rates"][arm]
            add(f"  full-rate {arm}: {rate['full']}/{rate['n']} = "
                f"{fmt_rate(rate['rate'])}")
        add(f"  bounds: [{fmt_frac(entry['bounds']['lo'])}, "
            f"{fmt_frac(entry['bounds']['hi'])}] over "
            f"{entry['bounds'].get('n_ancestors', 0)} ancestor(s), "
            f"{entry['bounds']['unresolved_pairs']} unresolved pair(s) "
            f"(pending {entry['bounds']['pending_pairs']}, "
            f"technical-invalid {entry['bounds']['technical_invalid_pairs']})")
    add("")
    ck = document["checkpoints"]
    add(f"checkpoints: {ck['observed']} group(s), expected {ck['expected']}")
    for group in ck["groups"]:
        parts = []
        for key in document["endpoint_order"]:
            entry = group["endpoints"][key]
            parts.append(f"{entry['budget_frames']}f={fmt_frac(entry['delta'])}")
        add(f"  ckpt#{group['index']} {group['ckpt_sha256'][:16]} "
            f"(seed {group['seed_hint']}) " + " ".join(parts))
    add("")
    inv = document["inventory"]
    add(f"inventory: {inv['totals']['rows']} row(s), "
        f"{inv['totals']['deductions']} deduction(s), "
        f"{inv['totals']['exhausted_rows']} exhausted, "
        f"{inv['totals']['death_with_bomb']} death-with-bomb, "
        f"{inv['totals']['rows_insufficient']} row(s) insufficient")
    add("")
    verdict = document["verdict"]
    for key in ("primary_30s", "secondary_40s"):
        item = verdict[key]
        add(f"verdict {key}: {item['decision']}  ({item['decision_cn']})")
        for reason in item["reasons"]:
            add(f"    reason: {reason}")
    add("")
    add(f"scope: {document['scope']['note']}")
    add(f"exit code: {document['exit_code']}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def build_document(args, files):
    rows, files_meta = load_rows(files)
    issues = Issues()
    pairs, identities = build_pairs(rows, issues)
    manifest_info = None
    expected_roots = {}
    if args.manifest:
        manifest = load_manifest(pathlib.Path(args.manifest))
        manifest_info, expected_roots = cross_check_manifest(
            manifest, pairs, rows, identities, issues)

    expected_ancestors, expected_source = expected_ancestors_from(
        manifest_info, pairs)
    pair_expectation, pair_expectation_source = expected_pairs_table(
        args, manifest_info, expected_roots, pairs)

    # ---- row-level status counts -----------------------------------------
    row_counts = {"scored": 0, "technical_invalid": 0, "pending": 0, "conflict": 0,
                  "superseded": 0}
    technical_invalid = []
    pending = []
    for pair in pairs.values():
        row_counts["superseded"] += len(pair["superseded"])
        for arm in ARMS:
            ref = pair["rows"].get(arm)
            if ref is None:
                continue
            cell = identities[ref.ref]
            row_state = classify_row(ref, cell)
            row_counts[row_state["status"]] = (row_counts.get(row_state["status"], 0)
                                               + 1)
            if row_state["status"] == "technical_invalid":
                technical_invalid.append({
                    "ref": ref.ref, "arm": arm,
                    "ancestor_id": pair["ancestor_id"],
                    "budget_frames": pair["budget_frames"],
                    "ckpt_sha256": pair["ckpt_sha256"],
                    "pair_rng": pair["pair_rng"], "episode": pair["episode"],
                    "reason": "; ".join(row_state["reasons"])})
            elif row_state["status"] in ("pending", "conflict"):
                pending.append({
                    "key": ";".join(str(part) for part in pair["key"]),
                    "status": row_state["status"], "arm": arm,
                    "ref": ref.ref,
                    "reason": "; ".join(row_state["reasons"]),
                    "budget_frames": pair["budget_frames"]})
        if pair["status"] != "complete":
            pending.append({
                "key": ";".join(str(part) for part in pair["key"]),
                "status": pair["status"], "arm": None,
                "ref": pair["refs"][0] if pair["refs"] else None,
                "reason": "; ".join(pair["reasons"]),
                "budget_frames": pair["budget_frames"]})

    pair_stats = {"complete": 0, "technical_invalid": 0, "pending": 0, "conflict": 0}
    for pair in pairs.values():
        pair_stats[pair["status"]] = pair_stats.get(pair["status"], 0) + 1

    # ---- gates ------------------------------------------------------------
    checks = []

    def gate(code, ok, detail):
        checks.append({"code": code, "ok": bool(ok), "detail": detail})
        return bool(ok)

    data_ok = gate(
        "DATA_INTEGRITY", issues.ok,
        ("no hard row-level refusal" if issues.ok else
         f"{len(issues.hard)} hard refusal(s): "
         + "; ".join(f"{item['code']} @ {item['ref']}" for item in issues.hard[:6])))
    count_ok = gate(
        "ANCESTOR_COUNT", len(expected_ancestors) == int(args.expected_ancestors),
        f"{len(expected_ancestors)} ancestor(s) in the {'manifest' if manifest_info else 'rows'}"
        f" ({', '.join(expected_ancestors) or 'none'}) vs pre-registered "
        f"{args.expected_ancestors} (PREREGISTRATION_PAIR_001 sec. 2: 8 个新开发祖先)")
    mixed_ok = gate(
        "MANIFEST_SINGLE_TABLE",
        len({str(row.row.get("manifest_sha256")) for row in rows}) == 1,
        "one frozen manifest digests every row"
        if len({str(row.row.get("manifest_sha256")) for row in rows}) == 1 else
        f"{len({str(row.row.get('manifest_sha256')) for row in rows})} distinct "
        f"manifest_sha256 values across rows: "
        f"{sorted({str(row.row.get('manifest_sha256')) for row in rows})}")
    scoring_ok = gate(
        "SCORING_EVIDENCE", pair_stats.get("conflict", 0) == 0,
        ("no pair whose score disagrees with its raw fields" if
         pair_stats.get("conflict", 0) == 0 else
         f"{pair_stats.get('conflict', 0)} pair cell(s) have a row whose "
         f"score.full_segment / score.technical_invalid disagrees with the raw "
         f"fields: 关键计分缺证（该对按保守方式记入上下界）"))

    checkpoint_seeds = args.checkpoint_seeds_list
    endpoints = {}
    for budget in sorted({PRIMARY_BUDGET, SECONDARY_BUDGET}
                         | {p["budget_frames"] for p in pairs.values()
                            if is_int(p["budget_frames"])}):
        entry = build_endpoint(pairs, budget, expected_ancestors, pair_expectation)
        endpoints[str(budget)] = entry

    checkpoint_axis = {budget: (len(endpoints[str(budget)]["design"]["checkpoint_shas"]))
                       for budget in (PRIMARY_BUDGET, SECONDARY_BUDGET)}
    expected_groups = len(checkpoint_seeds) if checkpoint_seeds else PREREG_CKPTS
    ckpt_ok = gate(
        "CHECKPOINT_AXIS",
        all(count == expected_groups for count in checkpoint_axis.values()),
        f"checkpoint groups per budget {checkpoint_axis} vs expected "
        f"{expected_groups}"
        + (" (--checkpoint-seeds)" if checkpoint_seeds else
           f" (PREREG sec. 2: {PREREG_CKPTS} 个 checkpoint)"))

    # ---- endpoint emission gate ------------------------------------------
    withheld_all = False
    for key, entry in endpoints.items():
        reasons = []
        if not data_ok:
            reasons.append("DATA_INTEGRITY: 行级硬拒绝未清（见 gates）")
        if not count_ok:
            reasons.append(
                f"ANCESTOR_COUNT: 清单/行内只有 {len(expected_ancestors)} 个祖先，"
                f"预注册要求 {args.expected_ancestors}")
        if entry["coverage"]["missing_ancestors"]:
            reasons.append(
                f"ANCESTOR_COVERAGE: {len(entry['coverage']['missing_ancestors'])} "
                f"个祖先在该档没有有效配对（"
                f"{', '.join(entry['coverage']['missing_ancestors'])}）")
        if entry["delta"] is None:
            reasons.append(f"NO_DELTA: {entry['delta_note']}")
        # the endpoint delta itself only needs ancestor coverage + clean rows; the
        # pre-registered 8-ancestor count is part of "祖先不足".
        refuse = bool(reasons)
        if refuse and args.require_complete:
            entry["emitted"] = False
            entry["withheld_reason"] = "；".join(reasons)
            # a refused endpoint must not carry a point estimate at all: the
            # per-ancestor rows above still let anyone recompute it, but the
            # report itself does not publish a Delta it refused to stand behind.
            entry["delta"] = None
            entry["delta_pp"] = None
            entry["loo"] = None
            withheld_all = True
        elif refuse:
            entry["emitted"] = entry["delta"] is not None
            entry["withheld_reason"] = "；".join(reasons)
            entry["stage_only"] = True
            withheld_all = True
        else:
            entry["emitted"] = entry["delta"] is not None
            entry["withheld_reason"] = None

    # ---- per-ancestor "no complete pair" note ----------------------------
    for key, entry in endpoints.items():
        entry["withheld"] = not entry["emitted"]

    # ---- verdicts ---------------------------------------------------------
    gate_reasons_common = []
    if not data_ok:
        gate_reasons_common.append("关键计分缺证：DATA_INTEGRITY 未通过")
    if not count_ok:
        gate_reasons_common.append(
            f"祖先不足：只有 {len(expected_ancestors)} 个祖先（预注册要求 "
            f"{args.expected_ancestors}）")
    if not ckpt_ok:
        gate_reasons_common.append(
            f"缺档：checkpoint 组数 {checkpoint_axis} != {expected_groups}")
    if not mixed_ok:
        gate_reasons_common.append("身份不符：行内 manifest_sha256 不唯一")
    if not scoring_ok:
        gate_reasons_common.append(
            f"关键计分缺证：{pair_stats.get('conflict', 0)} 个配对单元的 score 与原始"
            f"字段不一致")

    verdicts = {}
    for role, key in (("primary", str(PRIMARY_BUDGET)),
                      ("secondary", str(SECONDARY_BUDGET))):
        entry = endpoints.get(key)
        reasons = list(gate_reasons_common)
        if entry is None:
            reasons.append(f"没有 {key} 帧档的任何行")
        else:
            if entry["coverage"]["missing_ancestors"]:
                reasons.append(
                    f"祖先覆盖不足：{entry['coverage']['available_ancestors']}/"
                    f"{entry['coverage']['expected_ancestors']}（缺 "
                    f"{', '.join(entry['coverage']['missing_ancestors'])}）")
            if not entry["design"]["complete"]:
                reasons.append(
                    f"主配对不完整：应有 {entry['design']['expected_pairs']} 对、"
                    f"完整 {entry['design']['complete_pairs']} 对")
            if not entry["emitted"]:
                reasons.append(f"Δ 未出：{entry['withheld_reason']}")
        verdicts[role + ("_30s" if key == str(PRIMARY_BUDGET) else "_40s")] = decide(
            entry, args, reasons, role=role)
    verdict = {
        "primary_30s": verdicts["primary_30s"],
        "secondary_40s": verdicts["secondary_40s"],
        "rule_table": document_rule_table(),
        "scope": "试点不用于宣称泛化",
    }

    # ---- provenance -------------------------------------------------------
    provenance = {
        "files": files_meta,
        "rows_total": len(rows),
        "rows_scored": row_counts["scored"],
        "rows_technical_invalid": row_counts["technical_invalid"],
        "rows_pending": row_counts["pending"],
        "rows_conflict": row_counts["conflict"],
        "rows_superseded": row_counts["superseded"],
        "pairs_total": len(pairs),
        "pairs_complete": pair_stats.get("complete", 0),
        "pairs_technical_invalid": pair_stats.get("technical_invalid", 0),
        "pairs_pending": pair_stats.get("pending", 0),
        "pairs_conflict": pair_stats.get("conflict", 0),
        "manifest": (None if manifest_info is None else {
            "path": manifest_info["path"], "sha256": manifest_info["sha256"],
            "bytes": manifest_info["bytes"],
            "manifest_id": manifest_info.get("manifest_id"),
            "root_count": manifest_info["root_count"],
            "expected_pairs_per_ancestor_budget": manifest_info[
                "expected_pairs_per_ancestor_budget"],
            "row_manifest_sha256_values": manifest_info[
                "row_manifest_sha256_values"]}),
        "expected_pairs_source": pair_expectation_source,
        "expected_ancestors_source": expected_source,
        "wall_s_sum": math.fsum(float(ref.row.get("wall_s") or 0.0)
                                for ref in rows),
    }

    # ---- per-row evidence index: every number above can be walked back to a
    #      file + line + the digest of that very line ------------------------
    row_index = []
    used_refs = set()
    for pair in pairs.values():
        for arm in ARMS:
            ref = pair["rows"].get(arm)
            if ref is None:
                continue
            used_refs.add(ref.ref)
            cell = identities[ref.ref]
            row_state = classify_row(ref, cell)
            row_index.append({
                "ref": ref.ref, "line_sha256_16": ref.line_sha256_16,
                "arm": arm, "ancestor_id": pair["ancestor_id"],
                "budget_frames": pair["budget_frames"],
                "ckpt_sha256": pair["ckpt_sha256"],
                "pair_rng": pair["pair_rng"], "episode": pair["episode"],
                "tag": ref.tag, "status": row_state["status"],
                "full_segment": row_state.get("full_segment"),
                "pair_status": pair["status"],
                "superseded_rows": len(pair["superseded"]),
            })
    for ref in rows:
        if ref.ref in used_refs:
            continue
        cell = identities[ref.ref]
        row_index.append({
            "ref": ref.ref, "line_sha256_16": ref.line_sha256_16,
            "arm": cell["arm"], "ancestor_id": cell["ancestor_id"],
            "budget_frames": cell["budget_frames"],
            "ckpt_sha256": cell["ckpt_sha256"], "pair_rng": cell["pair_rng"],
            "episode": cell["episode"], "tag": ref.tag,
            "status": "identity_invalid", "full_segment": None,
            "pair_status": "unpaired",
            "superseded_rows": 0,
        })
    row_index.sort(key=lambda item: (str(item["ancestor_id"]),
                                     item["budget_frames"] or 0,
                                     str(item["ckpt_sha256"]),
                                     item["pair_rng"] or 0,
                                     item["arm"], item["ref"]))

    refusals = [{"code": item["code"], "message": item["message"],
                 "ref": item["ref"]} for item in issues.hard]
    for item in checks:
        if not item["ok"]:
            refusals.append({"code": item["code"], "message": item["detail"],
                             "ref": None})

    verdict_refused = any(
        item["decision"] == "REFUSED_INSUFFICIENT_DENOMINATOR"
        for item in (verdict["primary_30s"], verdict["secondary_40s"]))
    exit_code = EXIT_OK
    if refusals or withheld_all or verdict_refused:
        exit_code = EXIT_REFUSED

    order = [str(PRIMARY_BUDGET), str(SECONDARY_BUDGET)]
    for key in endpoints:
        if key not in order:
            order.append(key)
    # keep only the endpoints we actually built, in order
    order = [key for key in order if key in endpoints]

    document = {
        "schema": REPORT_SCHEMA,
        "tool": TOOL,
        "tool_version": TOOL_VERSION,
        "tool_sha256": (sha256_file(pathlib.Path(__file__))
                        if pathlib.Path(__file__).is_file() else "unknown"),
        "generated_at": report_timestamp(),
        "inputs": {
            "pairs": [pattern_form(pattern) for pattern in args.pairs],
            # the raw runtime spelling of every file, same order, so a human on
            # either host can still open them.  OFF by default on purpose: a
            # native spelling is host-specific (D:\... vs /mnt/d/...) and would
            # make two hosts' --json differ, which is exactly the defect gap C
            # fixes.  --keep-native-paths puts it back, side by side.
            "files_native": ([path_form(path)[2] for path in files]
                             if KEEP_NATIVE_PATHS else None),
            "files": [path_form(path)[0] for path in files],
            # the subset of inputs.files that is NOT inside this checkout: those
            # keep an absolute path, so their spelling is host-specific by nature
            "files_external": [path_form(path)[0] for path in files
                               if path_form(path)[1]],
            # one object per file: the machine-checkable form of the same fact
            "files_detail": [{"path": path_form(path)[0],
                              "external": path_form(path)[1]}
                             for path in files],
            "path_form": "repo-relative-posix",
            "manifest": None if args.manifest is None else path_form(
                args.manifest)[0],
            # native spellings, side by side, only when --keep-native-paths asked
            "manifest_native": (None if args.manifest is None
                                else native_or_none(args.manifest)),
            "pairs_native": ([str(pattern) for pattern in args.pairs]
                             if KEEP_NATIVE_PATHS else None),
            "checkpoint_seeds": checkpoint_seeds,
            "require_complete": bool(args.require_complete),
            "episodes": int(args.episodes),
            "expected_ancestors": int(args.expected_ancestors),
        },
        "prereg": {
            "document": "Document/Update/Alpha/v8.29.0/PREREGISTRATION_PAIR_001.md",
            "section": "§4 计分与验收",
            "expected_ancestors": int(args.expected_ancestors),
            "expected_checkpoints": expected_groups,
            "expected_pair_rngs": PREREG_RNGS,
            "threshold_pp": PREREG_THRESHOLD_PP,
            "min_positive": list(PREREG_MIN_POSITIVE),
            "budgets": {str(PRIMARY_BUDGET): "30 秒主比较（900 决策 / 1,800 帧）",
                        str(SECONDARY_BUDGET): "40 秒（1,200 决策 / 2,400 帧）"},
        },
        "provenance": provenance,
        "rows": row_index,
        "gates": {"ok": all(item["ok"] for item in checks), "checks": checks,
                  "refusals": refusals, "warnings": issues.soft,
                  "require_complete": bool(args.require_complete)},
        "endpoints": endpoints,
        "endpoint_order": order,
        "checkpoints": build_checkpoints(pairs, expected_ancestors,
                                         checkpoint_seeds, args),
        "technical_invalid": technical_invalid,
        "pending": pending,
        "inventory": build_inventory(rows, identities, pairs),
        "verdict": verdict,
        "scope": {
            "note": ("试点不用于宣称泛化：只有 8 个独立祖先，研究端不依据回合池化 "
                     "p 值宣称泛化或等效（预注册 §4 末段）；本报告不做任何显著性"
                     "检验，去一法区间只是敏感度范围，不是置信区间。"),
            "not_claimed": ["泛化", "等效性", "统计显著性", "回合池化 p 值"],
        },
        "limits": [
            "库存只在请求事件（bombs_remaining_at_emit）与段末（bombs_end）有读数，"
            "没有逐帧库存序列：任意帧的库存只能由 honoured 事件推导。",
            "补给（合法拾取导致 B>3）没有独立事件记录。bombs_end 与推导值不一致时"
            "只能判 insufficient，无法区分「补给」与「异常」；按 §4 该对仍保留在原位"
            "并进入待判上下界。",
            "bomb_available_at_emit 只表示请求帧上雷是否可用，不等于兑现； honoured "
            "判据只能是 emit_decrement_at_list 非 None（REPLY_02 §3）。",
            "若死亡发生在发出请求的那个决策内部，决策末帧的扣减可能观测不到"
            "（mark=None），该次请求是否兑现无法判定。",
            "某祖先缺档 → 该档 Δ 与判定不出，只给分母与界限；三份 checkpoint 缺一份"
            "同理不出主效应判定。",
            "本层不实现 §4 的「技术故障按原 pair 重跑双方一次」自动化：重跑记录由 "
            "runner 侧产生，本层只按规定接受同一 cell 的第二行（tag 必须不同）。",
            "谱系与曝光不在计分范围：本层只能核 ancestor_id 与清单/行内的身份一致，"
            "不能证明该祖先未被训练过。",
            "没有逐阶段拆分：行内 stage 是单值；若一批混了道中与 Boss，本层只能整体"
            "报，除非调用方按 stage 拆成不同文件。",
        ],
        "determinism": {
            "timestamp": ("generated_at 是本报告唯一随运行变化的字段；设置 "
                          "SOURCE_DATE_EPOCH=<unix 秒> 让它取该时刻（UTC，固定 "
                          "`+0000`），同一输入的 --json/--markdown 即可逐字节相同。"),
            "means": ("所有均值按 fractions.Fraction 精确计算、末尾只转一次 float，"
                      "因此不受 CPython 版本（3.12 改了 sum() 的浮点累加）与求和顺序"
                      "影响；并列按精确相等判定。"),
            "ordering": ("所有列表字段显式 sorted()；set 只做成员测试与计数，从不用于"
                         "决定输出顺序；dict 只按插入顺序读取，插入顺序由已排序的行"
                         "决定。"),
            "paths": ("所有路径字段都是仓库相对的 POSIX 形式（a/b/c.jsonl），由 "
                      "path_form() 相对 REPO = __file__.resolve().parents[2] 计算；"
                      "同一批数据在 Windows（D:\\...）与 WSL（/mnt/d/...）上得到同一个"
                      "字符串，所以 --json 的 sha256 相同（s12 只保证了数值相同）。"
                      "仓库外的文件保留绝对路径并标 external: true（列在 "
                      "inputs.files_external / inputs.files_detail 里），那种路径本来"
                      "就是主机特有的。--keep-native-paths 另存原始写法到 "
                      "inputs.files_native / manifest_native / pairs_native 供人核对，"
                      "默认关，否则两台主机的输出必然不同。"),
        },
        "methods": [
            "full_segment = (frames == budget_frames) and (decisions_executed == "
            "budget_decisions) and (first_death_offset is None) and (terminal is "
            "None)；末帧死亡记 0；合法转场/过面单列事件且满段记 0（§4）。",
            "2×2 = 同满 both_full / 仅 G1 only_G1 / 仅 T2 only_T2 / 同未满 neither。",
            "祖先差值 diff = (only_G1 - only_T2) / n_valid_pairs，"
            "n_valid_pairs = both_full + only_G1 + only_T2 + neither。",
            "Δ = 各祖先 diff 的等权平均（不按回合池化；祖先数即独立样本量）。",
            "绝对满段率 = 该臂 full_segment 行数 / 该臂在该档可计分行数（排除技术无效与"
            "待判行），两臂各算各的分母。",
            "正/平/负祖先 = diff > 0 / diff == 0 / diff < 0 的祖先个数。",
            "去一法区间：依次去掉一个祖先，对剩余 k-1 个 diff 取等权平均，共 k 次，取 "
            "min/max；不是标准差，不是置信区间。极值可能并列：min_dropped_ties / "
            "max_dropped_ties 列出全部并列祖先（按 id 排序），min_dropped / max_dropped "
            "保持旧字符串语义 = 确定性主名（并列集合中 id 字典序第一个），"
            "min_dropped_primary / max_dropped_primary 为同值显式名。",
            "所有均值（Δ、去一法每次留一、未决对上下界、每个 ckpt 组）都用精确有理数"
            "（fractions.Fraction）计算、末尾只舍入一次：并列按数学相等判定，且同一输入"
            "在 Python 3.10 与 3.12 上给出同一个浮点数（CPython 3.12 改了内置 sum() 的"
            "浮点累加，gh-100425）。",
            "Δ40 与 Δ30 同法；三份 checkpoint 各自按 ckpt_sha256 分组后同法重算。",
            "未决对上下界：每个未决对最多把该祖先差值向一侧推动 1/n，故上界 = "
            "(only_G1 - only_T2 + u)/n、下界 = (only_G1 - only_T2 - u)/n，"
            "端点界限 = 各祖先界限的等权平均；另给只计待判对（不含技术无效）的一组。",
            "技术无效 = invalid 或 error 非空；单列，不进 Δ 的分母，但进上下界（其得分"
            "未知，属于待判）。",
            "重跑：同 (祖先, 档, ckpt, pair_rng, episode, arm) 的第二行只有在 tag 不同"
            "时才被接受为 §4 的一次重跑，取较晚的一行为准并保留并被取代的行；同 tag "
            "重复行直接拒绝。",
            "库存：honoured 判定以 emit_decrement_at_list[i] 非 None 为准（REPLY_02 §3: "
            "emit_decrement_at=2，扣减在决策第 2 帧观测）；绝对帧 = bomb_frames[i] + "
            "mark - 1；耗尽后存活帧数 = 死亡帧或段尾 − 归零帧（幸存局截断并注明）。",
            "所有统计量附带 row_refs / files：每个数字都能回到 JSONL 的具体文件、行号"
            "与行 SHA-256 前 16 位。",
        ],
        "exit_code": exit_code,
    }
    document["verdict"]["primary_30s"]["available_denominator"] = (
        len(endpoints.get(str(PRIMARY_BUDGET), {}).get("available", [])))
    return document


def document_rule_table():
    return [
        {"condition": "Δ30 ≥ +5 个百分点且至少 5/8 个祖先为正",
         "decision": "NOMINATE_G1",
         "text": "提名 G1 进入另一批独立祖先的 30 秒复验；仅当 40 秒批完整、"
                 "Δ40 ≥ +5 pp 且 ≥5/8 正时同时建议复验 40 秒（次要终点）"},
        {"condition": "Δ30 ≤ −5 个百分点且至少 5/8 个祖先为负",
         "decision": "KEEP_T2",
         "text": "保留 T2 作 30 秒默认参照，复查 G1 在哪些阶段消耗过早或带雷死亡"},
        {"condition": "其余完整结果", "decision": "UNDETERMINED",
         "text": "两者差异未定，T2 保留为简单对照；先看逐祖先、逐阶段和库存事件"},
        {"condition": "祖先不足、身份不符、关键计分缺证或主配对不完整",
         "decision": "REFUSED_INSUFFICIENT_DENOMINATOR",
         "text": "如实报可用分母及界限，补来源、恢复或计分证据"},
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
EPILOG = """\
levels (why a delta can be missing while numbers are still printed)
  delta   point estimate; needs clean rows, every pre-registered ancestor present
          with >= 1 complete pair at that budget, and --require-complete
  bounds  always printed: each unresolved pair can move an ancestor's difference
          by at most +/-1/n, so [lo, hi] spans every resolution of the pending and
          technical-invalid pairs
  verdict the sec.4 decision table; needs a complete main batch (all expected
          pairs, both arms, three checkpoints), otherwise it says
          REFUSED_INSUFFICIENT_DENOMINATOR and names the usable denominator

leave-one-out ties (field contract)
  dropping different ancestors can reach the same extreme; a single name would
  then depend on loop order, i.e. on float noise.  The historical string fields
  keep their exact old meaning, and every added field is additive:
    loo.min_dropped / loo.max_dropped        string: deterministic PRIMARY name =
                                             lexicographically first ancestor id
                                             among the tied runs (unchanged)
    loo.min_dropped_primary / ..._primary    same primary name, spelled out
    loo.min_dropped_ties / loo.max_dropped_ties
                                             sorted list of EVERY tied ancestor
  A tie here is an exact rational equality (every published mean is computed as
  a fractions.Fraction and rounded to float once), not a float coincidence.
  The markdown table marks every tied run with 下界/上界 and prints them as
  `anc4`, `anc5`（并列 N 个，主名 `anc4`）.

determinism
  every mean is an exact rational rounded once, all list fields are sorted, and
  the only per-run field is generated_at.  Export SOURCE_DATE_EPOCH=<unix
  seconds> to pin it (UTC, "+0000"); two runs on the same input are then
  byte-identical (sha256 of --json).

path form (cross-platform comparability)
  every path field is a repository-relative POSIX path (a/b/c.jsonl) computed by
  path_form() against REPO = __file__.resolve().parents[2], so the same batch
  gives the same string on Windows (D:\\...) and on WSL (/mnt/d/...) and the
  --json sha256 matches across hosts -- s12 only made the numbers match.
  Covered: inputs.files, inputs.pairs, inputs.manifest, provenance.files[].path,
  provenance.manifest.path, rows[].ref and every row_refs/files list.
  A file OUTSIDE the repository keeps its absolute path and is flagged:
  inputs.files_external, inputs.files_detail[].external, provenance.files[]
  .external.  Such a path is host-specific by nature and is reported as such
  instead of being silently "comparable".
  --keep-native-paths ALSO writes the raw runtime spelling to
  inputs.files_native / inputs.manifest_native / inputs.pairs_native, for a human
  on this host.  Off by default: a native spelling is host-specific, so emitting
  it unconditionally would recreate the defect this exists to fix.

exit codes
  0  every gate passed, every endpoint delta emitted
  1  refused: some gate failed (duplicate row, arm/ancestor conflict, manifest
     disagreement, ancestor shortfall).  The report is still written and still
     carries denominators and bounds.
  2  usage/IO error (unreadable file, bad JSON line, empty --pairs pattern)
"""


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="v829_report.py",
        description=(
            "V829-PAIR-01 sec. 4 scoring: 2x2 per ancestor, equal-weight delta, "
            "absolute full-segment rates, positive/tie/negative ancestor counts, "
            "true leave-one-out interval, per-checkpoint delta, all-G1/all-T2 "
            "bounds for unresolved pairs, inventory events and the pre-registered "
            "verdict.  Offline; standard library only."),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", action="append", metavar="GLOB_OR_PATH",
                        required=True,
                        help="pair rows (*.jsonl), repeatable; a glob or a literal "
                             "path.  Every row must be a v829-pair-episode/1 line.")
    parser.add_argument("--manifest", metavar="PATH", default=None,
                        help="optional frozen run table (v829-pair-manifest/1): "
                             "cross-checks ancestor / budget / checkpoint / pair_rng "
                             "/ manifest_sha256 against the rows and supplies the "
                             "expected pair count per (ancestor, budget).")
    parser.add_argument("--checkpoint-seeds", metavar="0,1,2", default=None,
                        help="optional checkpoint seed list; used to label groups "
                             "(and to check the group count).  Grouping itself is "
                             "always by ckpt_sha256.")
    parser.add_argument("--json", metavar="OUT", default=None,
                        help="write the machine-readable report here")
    parser.add_argument("--markdown", metavar="OUT", default=None,
                        help="write the paste-ready markdown report here")
    parser.add_argument("--require-complete", dest="require_complete",
                        action="store_true", default=True,
                        help="default ON: an ancestor short of 2 arms x both "
                             "budgets refuses the delta and only denominators and "
                             "bounds are printed")
    parser.add_argument("--no-require-complete", dest="require_complete",
                        action="store_false",
                        help="emit the delta even when the batch is incomplete, "
                             "flagged stage_only (never a verdict)")
    parser.add_argument("--expected-ancestors", type=int,
                        default=PREREG_ANCESTORS, metavar="N",
                        help=f"pre-registered number of independent ancestors "
                             f"(default {PREREG_ANCESTORS}, PREREGISTRATION_PAIR_001 "
                             f"sec. 2)")
    parser.add_argument("--episodes", type=int, default=1, metavar="N",
                        help="episodes per manifest root (default 1; multiplies the "
                             "expected pair count when --manifest is given)")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print the text summary to stdout")
    parser.add_argument("--keep-native-paths", dest="keep_native_paths",
                        action="store_true", default=False,
                        help="also record the RAW runtime spelling of every input "
                             "path in inputs.files_native / inputs.manifest_native "
                             "/ inputs.pairs_native, next to the normalised ones, "
                             "for a human on this host to open. Off by default: a "
                             "native spelling is host-specific (D:\\... on Windows "
                             "vs /mnt/d/... on WSL), so emitting it unconditionally "
                             "would make two hosts' --json differ byte for byte "
                             "over the same data. The comparable fields "
                             "(inputs.files, provenance.files[].path, rows[].ref, "
                             "provenance.manifest.path) are repo-relative POSIX "
                             "either way; a file outside the repository keeps an "
                             "absolute path and is flagged external: true")
    args = parser.parse_args(argv)
    args.checkpoint_seeds_list = None
    if args.checkpoint_seeds:
        seeds = []
        for part in str(args.checkpoint_seeds).split(","):
            part = part.strip()
            if not part:
                continue
            if not part.lstrip("+-").isdigit():
                parser.error(f"--checkpoint-seeds: {part!r} is not an integer")
            seeds.append(int(part))
        if len(set(seeds)) != len(seeds):
            parser.error("--checkpoint-seeds: duplicate seed")
        args.checkpoint_seeds_list = seeds
    if args.expected_ancestors < 1:
        parser.error("--expected-ancestors must be >= 1")
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    return args


def report_timestamp() -> str:
    """``generated_at``: wall clock, or a pinned instant from SOURCE_DATE_EPOCH.

    The reproducible-builds convention (SOURCE_DATE_EPOCH = unix seconds) replaces
    "now" with that instant, formatted in UTC with a fixed ``+0000``, so that two
    runs over the same inputs produce byte-identical --json/--markdown files.
    Without it this field is the only one expected to differ between runs.
    """
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is None:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")
    text = epoch.strip()
    if not text.lstrip("+-").isdigit():
        raise ReportError(
            "SOURCE_DATE_EPOCH",
            f"SOURCE_DATE_EPOCH={epoch!r} is not an integer number of seconds",
            exit_code=EXIT_ERROR)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(int(text))) + "+0000"


def write_text(path, text):
    target = pathlib.Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")


def main(argv=None):
    global KEEP_NATIVE_PATHS
    args = parse_args(sys.argv[1:] if argv is None else argv)
    # one switch, read by load_rows/load_manifest/build_document: the native
    # spellings either appear everywhere or nowhere, never half-way
    KEEP_NATIVE_PATHS = bool(getattr(args, "keep_native_paths", False))
    try:
        files = expand_patterns(args.pairs)
        document = build_document(args, files)
    except ReportError as error:
        print(f"REFUSED [{error.code}] {error.message}", file=sys.stderr)
        return error.exit_code
    if args.json:
        write_text(args.json, json.dumps(document, ensure_ascii=False, indent=2,
                                         default=str) + "\n")
    markdown = render_markdown(document)
    if args.markdown:
        write_text(args.markdown, markdown)
    if not args.quiet:
        print(render_text(document))
    return int(document["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
