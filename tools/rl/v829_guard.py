"""v8.29 pair-manifest guard: refuse to run unless the manifest can prove its frames.

Why this file exists
--------------------
`PREREGISTRATION_PAIR_001.md` §2 only admits a new development ancestor if the
existing scripts can hold **2,400 frames** from a verifiable takeover state, and
§3.1 requires the last executed frame, the first controlled frame and the verify
anchors to be recorded per start.  The v8.29 INFO-01 inventory then found that
`configs/v430/splits.json` carries **no** `takeover_frame`, `total_frames`,
`parent_prefix_digest` or `verify_anchor` anywhere -- `frame_count` and `bbox` are
absent too, and the four letters `frame` do not occur in its 118 KB at all.  So
nothing in the current indexes can be used to *derive* those numbers.

A guard that filled them in from a default would silently bless a window nobody
verified, and the failure would only surface after the run had already produced
policy results.  This file therefore refuses instead: every root must carry its
own takeover frame, total frame count, parent prefix digest and verify anchor,
and a missing, mistyped or contradictory value is a finding -- never an
inference.  There is no fallback path and no default value anywhere below: the
top level is *not* consulted for `takeover_frame` / `total_frames` /
`parent_prefix_digest` / `verify_anchor`, and a top-level copy is reported as
ignored rather than inherited (`--allow-exposed` is the only opt-out and it
relaxes exactly one check).

Frozen manifest contract (``"v829-pair-manifest/1"``)
-----------------------------------------------------
::

    {
      "schema": "v829-pair-manifest/1",
      "manifest_id": "optional free text",
      "roots": [
        {
          "ancestor_id": "text; one independent play-root, NOT unique per root",
          "stage": "text; the guard does NOT judge stage admissibility",
          "budget_frames": 2400,
          "ckpt_sha256": "<64 lowercase hex>",
          "engine_sha256": "<64 lowercase hex>",
          "pair_rng": 20261123,
          "takeover_frame": 8663,
          "total_frames": 20000,
          "parent_prefix_digest": "<64 lowercase hex>",
          "verify_anchor": {"prefix_digest": "<64 lowercase hex>"},
          "used_for_training": false,
          "used_for_selection": false,
          "pair_index": "OPTIONAL integer 0..15, the pair-RNG repeat number
                         inside one (ancestor, checkpoint) cell; absent = 0",
          "ckpt_seed": "OPTIONAL integer, the seed that produced this checkpoint"
        }
      ]
    }

`takeover_frame`, `total_frames`, `parent_prefix_digest` and `verify_anchor` are
the four classes §2 needs and are required **per root**.  `verify_anchor` is an
object because §3.1 records more than one number against it; only
`prefix_digest` is load-bearing here, and it must equal the root's
`parent_prefix_digest` (that redundancy is the point of `DIGEST_MISMATCH`: a typo
in either copy is caught before the engine starts).

Identity keys
-------------
``ancestor_id`` is free text, so it is normalised with ``str.strip()`` before it
is used as a duplicate key; the raw value is still printed in every finding.  It
is deliberately **not** case-folded: the ids come from
``configs/v430/splits.json``, where case is significant, so folding would merge
two genuinely distinct ids into a false ``DUPLICATE_ANCESTOR``; and the bypass a
fold would chase -- the same window under a different label -- is closed by the
window key below, which never looks at the label.  ``strip()`` costs nothing
because a leading or trailing space is never meaningful in an id.

An **ancestor** is an independent play-root, not a row.  §2 evaluates "3
checkpoints x 2 pair RNGs **per ancestor**", so both of those are
repeat-measurement axes *inside* one ancestor: a *cell* is one point of that grid,
``(ancestor_id, ckpt_sha256, pair_index)``, where ``pair_index`` says which of the
two (up to :data:`MAX_PAIR_INDEX`) pair RNGs this root is and defaults to **0**
when the root does not say.  One ancestor therefore legitimately owns several
roots that differ only in their checkpoint, their repeat number and their seed.
An earlier version keyed on ``ancestor_id`` alone, which made the pre-registered
design unwritable: expressing 8 ancestors x 3 checkpoints honestly -- 24 roots,
8 distinct ids -- was refused as 8 ``DUPLICATE_ANCESTOR`` findings, while the only
file the guard accepted spelled the same 24 cells as 24 distinct ids and was then
refused by `tools/rl/v829_report.py` with ``ANCESTOR_COUNT: 24 vs 8``.  The
version after that fixed the checkpoint axis but still keyed the cell on
``(ancestor_id, ckpt_sha256)``, so the second §2 axis -- one checkpoint measured
at two pair RNGs, 48 cells -- was refused as ``DUPLICATE_ANCESTOR`` +
``DUPLICATE_WINDOW``.  Both halves of the squeeze and the second axis are in
``.tmp_v829_info/s18/REPORT.md`` and ``.tmp_v829_info/s19/REPORT.md``.

One ancestor, several roots, therefore also needs a consistency gate of its own:
the fields that describe the play-root itself -- ``stage``, ``window``, the tape
(``prefix_tape`` or an S1c ``s1c_window``), ``takeover_frame``, ``total_frames``
and ``parent_prefix_digest`` -- must agree across every root carrying that
``ancestor_id``, or the file is describing two different roots under one label
(``ANCESTOR_INCONSISTENT``).  Only the checkpoint, the seed, the repeat number
and the **budget tier** may differ: ``pair_index`` is deliberately **not** one of
the invariant fields, because it is the repeat axis and *must* vary between the
two pair-RNG runs of one cell, and ``budget_frames`` is not one either -- see the
next section, which is why v8.30.0 removed it.

The budget tier is §2's endpoint axis (30 s = 1,800 frames, 40 s = 2,400)
------------------------------------------------------------------------
§2 evaluates the *same take-over state* at two budgets: "30 秒与 40 秒分别从同一
接管状态开跑".  A manifest that can only carry one ``budget_frames`` per ancestor
therefore cannot express the primary/secondary endpoint pair at all, and until
v8.30.0 the consistency gate made it worse than unwritable: ``budget_frames`` was
an invariant, so the pre-registered 2-root shape (one ancestor, one tape, one
``takeover_frame``, one ``total_frames``, budgets 1,800 + 2,400) was refused as
``ANCESTOR_INCONSISTENT`` *and* ``DUPLICATE_ANCESTOR``.  The reproduction is
``.tmp_v829_info/s20/repro-before.log`` (exit 1, both codes).

The shape is now carried **one tier per root** (a scalar ``budget_frames``), not
as a ``budgets: [1800, 2400]`` array on one root, for three reasons:

1. every downstream reader already treats the root as one arm: `v829_pair.py`
   refuses a root whose declared ``budget_frames`` is not the ``--budget-frames``
   of the invocation (``BUDGET_ROOT_MISMATCH``) and `v829_report.py` averages
   "for every ancestor **and every budget**", so one root *is* one
   (ancestor, budget) arm -- an array would have to be expanded downstream, and
   that expansion is exactly what this guard exists to make impossible to do
   silently;
2. the window identity, the seed slot and the per-root ceiling
   (``takeover_frame + budget_frames <= total_frames``) are all defined on a
   scalar, so a scalar keeps one meaning for "the frames this root proves";
3. an array would need a "which value binds this root" rule -- the same
   ambiguous-precedence trap ``ckpt`` vs ``ckpts`` has -- while two roots need
   no rule at all.

What replaces the old invariant is :data:`ANCESTOR_BUDGET_SET`: across one
``ancestor_id`` the declared tiers must be a **subset of** :data:`BUDGET_TIERS`
and **no (checkpoint, repeat) cell may be measured twice at the same tier**.
That second rule is the only reading of "no repeats" that is compatible with the
pre-registered grid: one ancestor legitimately owns 3 checkpoints x 2 pair RNGs =
6 roots and every one of them declares the *same* tier (the 24- and 48-root
manifests on disk do exactly that and must stay exit 0), so the tier is a set per
*cell*, not per ancestor.  ``ANCESTOR_BUDGET_SET`` therefore names the budget
axis in its own words; the same 2-root path is also a duplicate cell, which
``DUPLICATE_ANCESTOR`` reports with ``budget_frames`` now in its key.

A label is not an identity, so every root is also keyed on the window itself:
``(ckpt_sha256, parent_prefix_digest, takeover_frame, total_frames,
budget_frames, pair_index)``.  Two roots that agree on all six are the same
physical measurement whatever ``ancestor_id`` they carry, and section 4 averages
the eight ancestors equally -- counting one window twice moves the primary
endpoint.  ``budget_frames`` stays inside the key because the same takeover window
measured at 1,800 and at 2,400 frames is a different section 2 pair and the two
must not be merged.  ``ckpt_sha256`` is inside the key on purpose: the three
checkpoints of one ancestor are three *different* windows over the same tape, so
they pass ``DUPLICATE_WINDOW`` untouched, and the identical (tape, takeover,
budget, checkpoint) under two labels is still refused.  ``pair_index`` is inside
the key for the same reason: the same window measured twice with two pair RNGs is
the second §2 axis, while the same window at the *same* repeat number is one cell
whose second copy section 4 would average twice.  The residue is honest -- a
re-labelled copy that also changes its repeat number looks exactly like a
legitimate second RNG run, which is why section 1 pre-allocates one seed per
(ancestor, budget, checkpoint, repeat) and why ``RNG_REUSED`` still names the
reused seed.

Repeat number and checkpoint seed (``pair_index``, ``ckpt_seed``)
-----------------------------------------------------------------
``pair_index`` is an optional integer, default **0**, and must satisfy
``0 <= pair_index < 16`` (:data:`MAX_PAIR_INDEX`); a boolean, a float, a string or
a number outside that range is ``FIELD_INVALID`` *and* the root seeds no
cross-root key, so the broken value cannot manufacture a duplicate finding either.
It is the second §2 axis: two roots of one (ancestor, checkpoint) cell that differ
only in ``pair_index`` and ``pair_rng`` are two legitimate repeat measurements, and
``v829_manifest_build.py``'s ``rngs`` option writes exactly that.  A root without
``pair_index`` means repeat 0, so a root that omits it and a root that says
``"pair_index": 0`` describe the same cell and one of the two is a duplicate.

``ckpt_seed`` is still accepted, typed and range-checked, and is recorded as the
seed that produced the checkpoint.  It is **not** part of any identity key: the
repeat number, not the checkpoint's own seed, decides which measurement a root is.

Duplicate JSON keys
-------------------
A file that repeats a key inside one object is refused with
``MANIFEST_UNREADABLE`` / exit 2 instead of a field-level code.  No field-level
verdict would be meaningful: ``json.loads`` keeps the last value, so the object
being judged is already ambiguous, which is exactly the "could not be evaluated
at all" class that exit 2 exists for.  A separate code would be an alias of that
class and would still have to exit 2.  Note that :func:`audit` cannot perform
this check -- a Python ``dict`` has already collapsed the duplicate -- so
byte-level callers must go through :func:`evaluate` or ``check``.

Error codes (every one is a refusal)
------------------------------------
==========================  ==========================================================
``MANIFEST_UNREADABLE``     path missing, not UTF-8 JSON, not a JSON object, or an
                            object that repeats a key (first value would be lost)
``BUDGET_INVALID``          the ``budget`` gate parameter is not one of
                            ``BUDGET_TIERS``: ``2400.0``, ``0``, ``-1``, ``None``,
                            ``"2400"`` and ``True``/``False`` are refusals, never a
                            silent "skip the requested tier"
``SCHEMA_MISSING``          no ``schema`` key
``SCHEMA_UNSUPPORTED``      ``schema`` is not ``v829-pair-manifest/1``
``FIELD_MISSING``           a required field is absent (named one per finding)
``FIELD_INVALID``           present but unusable (wrong type, bool-as-int, bad hex)
``BUDGET_EXCEEDS_FRAMES``   ``takeover_frame + budget > total_frames``
``EXPOSURE_UNKNOWN``        ``used_for_training`` / ``used_for_selection`` absent or
                            not a boolean -- an unknown exposure is a refusal
``EXPOSURE_CONFLICT``       one of them is true and ``--allow-exposed`` was not given
``DIGEST_MISMATCH``         ``parent_prefix_digest`` != ``verify_anchor.prefix_digest``
``ANCESTOR_INCONSISTENT``   two roots share an ``ancestor_id`` but disagree on what
                            that play-root *is*: ``stage`` / ``window`` / the tape
                            (``prefix_tape`` or ``s1c_window``) / ``takeover_frame``
                            / ``total_frames`` / ``parent_prefix_digest``, or one of
                            them declares such a field while its sibling does not.
                            ``pair_index`` is not one of them (it is the repeat
                            axis and must vary) and ``budget_frames`` is not one
                            either (it is the endpoint axis -- ``ANCESTOR_BUDGET_SET``
                            judges it)
``ANCESTOR_BUDGET_SET``     the endpoint axis of one ``ancestor_id`` is malformed:
                            a declared ``budget_frames`` is outside
                            :data:`BUDGET_TIERS`, or one ``(ckpt_sha256,
                            pair_index)`` cell declares the same tier twice (a tier
                            is one endpoint, so the same tier twice is one
                            measurement counted twice, not a third endpoint).  Two
                            *different* tiers of one cell are the pre-registered
                            30 s/40 s pair and pass
``DUPLICATE_ANCESTOR``      the same ``(ancestor_id, ckpt_sha256, pair_index,
                            budget_frames)`` appears twice (``ancestor_id`` after
                            ``strip()``, ``pair_index`` default 0): one cell is one
                            measurement, however it is labelled.  A *different*
                            checkpoint, a *different* repeat number or a *different*
                            budget tier under one ``ancestor_id`` is a §2 repeat /
                            endpoint axis and is legal
``DUPLICATE_WINDOW``        two roots agree on ``(ckpt_sha256,
                            parent_prefix_digest, takeover_frame, total_frames,
                            budget_frames, pair_index)``: one measurement under
                            two labels
``RNG_REUSED``              two roots share a ``pair_rng`` under one of the seed
                            identity keys; a diagnosis with one legitimate
                            standalone case (see below)
==========================  ==========================================================

``RNG_REUSED`` can now stand alone, and that is a property of the two-axis grid,
not an accident: with the repeat number legal the two runs of one cell differ only
in their seed, so two of them sharing one ``pair_rng`` are a genuine reuse of an
allocation section 1 reserves per (ancestor, budget, checkpoint, repeat) -- and no
other code fires, because checkpoint *and* repeat number do name two different
cells.  Three key families are checked so the diagnosis is not blind to any route:

* the **cell slot** -- ``(ancestor, budget, checkpoint, pair_index)``: one cell,
  one seed; this is the key the ticket freezes;
* the **ancestor family** -- ``(ancestor, budget, checkpoint)``: a declared repeat
  number is not a licence, so the two repeats of one checkpoint sharing a seed are
  still a reuse (the one case where ``RNG_REUSED`` refuses on its own);
* the **window family** -- ``(checkpoint, prefix, takeover, total, budget,
  pair_index)``: catches a seed reused by a re-labelled copy of one measurement.

So a repeated seed either repeats a cell (which also trips
``DUPLICATE_ANCESTOR`` / ``DUPLICATE_WINDOW``) or repeats a seed across the two
repeat numbers of one cell (``RNG_REUSED`` alone).  ``ANCESTOR_BUDGET_SET`` (the
endpoint axis) is reported before ``DUPLICATE_ANCESTOR``, which is reported before
``DUPLICATE_WINDOW``, and all three before ``RNG_REUSED`` (:data:`CODE_ORDER`), so
when a cell is duplicated the report already says which cell, which tier and which
seed it is.

Exit codes
----------
``0``  every check passed -- the manifest may be used
``1``  at least one finding -- the manifest must be refused
``2``  the manifest was never judged: the bytes are unreadable or ambiguous, or
       the gate parameter is unusable (``BUDGET_INVALID``).  This is also what
       ``argparse`` returns for a usage error, so a library caller passing
       ``budget=2400.0`` and ``--budget 2400.0`` on the command line agree on the
       same exit code

The `--budget` default is **2400**, the pre-registration's admission bar, not the
secondary 1,800 endpoint: checking only the declared tier by default would let a
manifest that declares 1,800 frames pass a gate that requires 2,400.

Usage
-----
::

    python -B tools/rl/v829_guard.py check <manifest.json> [--allow-exposed]
                                          [--budget 1800|2400] [--json]
    python -B tools/rl/v829_guard.py selftest [--work <dir>]

Self-test: ``python -B tools/rl/v829_guard.py selftest`` -- synthetic manifests
covering every declared code plus the semantics the codes stand for: the default
2,400 bar, ``takeover_frame + budget_frames == total_frames`` (a pass), bool not
being an integer, an uppercase digest, a non-boolean exposure, a relabelled
window, a label carrying a trailing space, duplicate JSON keys, the seven
unusable ``budget`` values, "the top level is never inherited", "unknown fields
are ignored", and the ancestor/repeat-axis semantics this file owns: 3
checkpoints under one ancestor pass, the same ancestor at the same checkpoint
(and repeat number) is ``DUPLICATE_ANCESTOR``, a checkpoint whose
``takeover_frame`` (or ``stage``) contradicts its siblings is
``ANCESTOR_INCONSISTENT``, 3 checkpoints sharing one ``pair_rng`` pass (the
checkpoint already separates them), the pre-registered **two-RNG** shape passes
when ``pair_index`` 0/1 carry different seeds, a duplicated
``(ancestor, checkpoint, pair_index)`` is refused, the same cell at the same
repeat number with one seed reports ``RNG_REUSED``, an absent ``pair_index`` and
an explicit ``0`` are one repeat number, ``pair_index`` 16 is
``FIELD_INVALID``, and the two repeats of one cell sharing one seed is the case
where ``RNG_REUSED`` stands alone.  The endpoint axis has its own four cases: the
same ancestor / tape / takeover / total at budgets 1,800 **and** 2,400 passes (the
pre-registered 30 s + 40 s pair), the same cell at 1,800 twice is
``ANCESTOR_BUDGET_SET``, two tiers whose ``takeover_frame`` disagrees is
``ANCESTOR_INCONSISTENT``, and a 2,400 tier that does not fit
(``takeover + 2400 > total``) is ``BUDGET_EXCEEDS_FRAMES``.  Both the exit code
and the exact finding-code set of every case are asserted, and the battery prints
``SELFTEST: ok N/N`` with the count it actually ran (there is no hard-coded N).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

SCHEMA = "v829-pair-manifest/1"
GUARD = "tools/rl/v829_guard.py"
#: v8.29.2: ``pair_index`` became part of both cell identity keys --
#: ``(ancestor_id, ckpt_sha256, pair_index)`` for ``DUPLICATE_ANCESTOR`` and the
#: six-field window (five + ``pair_index``) for ``DUPLICATE_WINDOW`` -- so the
#: second §2 axis (3 checkpoints x **2 pair RNGs** = 48 cells) is writable in one
#: manifest.  The document keys did not change (``v829_pair.py`` reads them).
#: v8.30.0: ``budget_frames`` left the ancestor-invariant set (it is §2's endpoint
#: axis: one take-over state at 1,800 and at 2,400 frames), joined the
#: ``DUPLICATE_ANCESTOR`` cell key, and got its own gate
#: (``ANCESTOR_BUDGET_SET``).  Keys and codes are additions only.
GUARD_VERSION = "v8.30.0"

#: The two pre-registered budget tiers (1,800 = secondary endpoint, 2,400 = the
#: §2 admission bar).  Frozen as choices so a typo cannot silently widen a gate.
BUDGET_TIERS = (1800, 2400)
DEFAULT_BUDGET = 2400

#: The four classes §2 needs.  Required per root; never read from the top level.
FRAME_FIELDS = ("takeover_frame", "total_frames", "parent_prefix_digest", "verify_anchor")

#: The per-root identity fields §3.1/§2 freeze alongside the frames.
IDENTITY_FIELDS = ("ancestor_id", "stage", "budget_frames",
                   "ckpt_sha256", "engine_sha256", "pair_rng")

REQUIRED_ROOT_FIELDS = IDENTITY_FIELDS + FRAME_FIELDS
EXPOSURE_FIELDS = ("used_for_training", "used_for_selection")
ANCHOR_KEY = "prefix_digest"
#: ``pair_index`` / ``ckpt_seed`` are optional: typed and range-checked when
#: present, and absent from :data:`REQUIRED_ROOT_FIELDS`.  ``pair_index`` names
#: which repeat of one (ancestor, checkpoint) cell a root is, so it joins both cell
#: identity keys and the seed slot; ``ckpt_seed`` is recorded, never keyed on.
REPEAT_AXIS_FIELDS = ("pair_index", "ckpt_seed")
#: The repeat axis is an index, not a free integer: 0 is the first pair RNG, 15 the
#: last.  A number above this names no pre-registered measurement, so it is
#: refused instead of being wrapped or silently ignored.
MAX_PAIR_INDEX = 16
INT_FIELDS = ("budget_frames", "pair_rng", "takeover_frame", "total_frames") \
    + REPEAT_AXIS_FIELDS
TEXT_FIELDS = ("ancestor_id", "stage")
DIGEST_FIELDS = ("ckpt_sha256", "engine_sha256", "parent_prefix_digest")

#: Keys a caller might expect the guard to inherit from the top level.  They are
#: reported as ignored so nobody mistakes a template for a checked value.
NOT_INHERITED = ("takeover_frame", "total_frames", "parent_prefix_digest", "verify_anchor")

#: The five fields that name a physical window.  Two roots that agree on all five
#: are the same window no matter which ``ancestor_id`` they carry; the repeat
#: number joins them in :func:`window_identity`, so one window at two pair RNGs is
#: two legal measurements while one window at the same repeat number is not.
WINDOW_FIELDS = ("ckpt_sha256", "parent_prefix_digest", "takeover_frame",
                 "total_frames", "budget_frames")

#: The spellings that can name a root's recording.  The manifest builder writes
#: ``prefix_tape``; an S1c row names the same thing as ``s1c_window``, so either
#: one is the tape descriptor when ``ANCESTOR_INCONSISTENT`` compares siblings.
TAPE_FIELDS = ("prefix_tape", "s1c_window")

#: The play-root fields that must agree across every root of one ``ancestor_id``.
#: ``tuple`` order is the order the disagreement list is printed in.
#: ``budget_frames`` is deliberately **absent**: §2 measures one take-over state at
#: two budgets, so it is the endpoint axis and must be free to differ across the
#: roots of one ancestor (``ANCESTOR_BUDGET_SET`` judges the set instead).
ANCESTOR_INVARIANT_FIELDS = ("stage", "window", "tape", "takeover_frame",
                             "total_frames", "parent_prefix_digest")

#: The tiers a declared ``budget_frames`` may name.  §2 pre-registers exactly two
#: endpoints (30 s = 1,800 frames, 40 s = 2,400), so anything else is outside the
#: contract even if it is a positive integer.
ANCESTOR_BUDGET_TIERS = BUDGET_TIERS

#: Printed for a field that a root does not declare, so "one root names the tape
#: and its sibling does not" is a visible disagreement instead of a silent pass.
_ABSENT = "<absent>"

#: Codes that mean "this manifest was never judged": the bytes are unusable, or
#: the gate parameter is.  ``evaluate`` maps them to exit 2 (``argparse``'s own
#: usage-error code) instead of 1, which is a verdict on the manifest itself.
FATAL_CODES = ("MANIFEST_UNREADABLE", "BUDGET_INVALID")

#: Reporting order.  A finding's position never depends on dict iteration order.
CODE_ORDER = (
    "MANIFEST_UNREADABLE",
    "BUDGET_INVALID",
    "SCHEMA_MISSING",
    "SCHEMA_UNSUPPORTED",
    "FIELD_MISSING",
    "FIELD_INVALID",
    "BUDGET_EXCEEDS_FRAMES",
    "EXPOSURE_UNKNOWN",
    "EXPOSURE_CONFLICT",
    "DIGEST_MISMATCH",
    "ANCESTOR_INCONSISTENT",
    "ANCESTOR_BUDGET_SET",
    "DUPLICATE_ANCESTOR",
    "DUPLICATE_WINDOW",
    "RNG_REUSED",
)
_HEX = frozenset("0123456789abcdef")


# --------------------------------------------------------------------------- #
# typed predicates -- bool is an int subclass, so it is rejected explicitly
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
# gate parameter and window identity
# --------------------------------------------------------------------------- #
def budget_problem(budget: object) -> str | None:
    """Return the refusal detail for an unusable ``budget``, else ``None``.

    The requested tier used to be guarded by ``if is_int(budget) and budget > 0``
    inside the per-root loop, so *any* unusable value (``2400.0``, ``0``, ``-1``,
    ``None``, ``"2400"``, ``True``) silently turned that check off and let the
    manifest through.  A gate parameter that cannot be honoured is now a refusal,
    the same way ``argparse`` refuses ``--budget 2400.0``.
    """
    if not is_int(budget):
        return (f"budget must be one of the pre-registered tiers {BUDGET_TIERS}; got "
                f"{type_name(budget)} ({budget!r}), which would silently disable the "
                f"requested-tier check instead of proving it")
    if budget not in BUDGET_TIERS:
        return (f"budget {budget} is not one of the pre-registered tiers "
                f"{BUDGET_TIERS}; a tier outside the contract cannot be proven")
    return None


def pair_index_of(root: dict) -> int | None:
    """The root's repeat number: 0 when undeclared, ``None`` when unusable.

    Absent means "the first repeat", so a root that omits ``pair_index`` and one
    that says ``0`` describe the same cell -- that equality is why the two
    spellings cannot be counted as two axes.  A value that is not an integer, is a
    boolean, or falls outside ``0 <= pair_index < MAX_PAIR_INDEX`` has already been
    reported ``FIELD_INVALID``; returning ``None`` keeps it from seeding a
    cross-root key that nobody could compare (the same rule the window fields use).
    """
    if "pair_index" not in root:
        return 0
    value = root["pair_index"]
    if not is_int(value) or not 0 <= value < MAX_PAIR_INDEX:
        return None
    return value


def window_identity(root: dict) -> tuple | None:
    """The five window fields plus the repeat number, or ``None`` if unusable.

    A root whose window fields are invalid is already reported field by field and
    must not seed a cross-root key, so every field has to pass its own type and
    range check first.  ``pair_index`` is the sixth component because the second §2
    axis measures one window twice: two roots that agree on the five window fields
    *and* on the repeat number are one measurement counted twice, while the same
    five fields at repeat 0 and repeat 1 are the pre-registered two-RNG pair.
    """
    ckpt = root.get("ckpt_sha256")
    prefix = root.get("parent_prefix_digest")
    takeover = root.get("takeover_frame")
    total = root.get("total_frames")
    declared = root.get("budget_frames")
    repeat = pair_index_of(root)
    if not is_digest(ckpt) or not is_digest(prefix):
        return None
    if not is_int(takeover) or takeover < 0:
        return None
    if not is_int(total) or total <= 0:
        return None
    if not is_int(declared) or declared <= 0:
        return None
    if repeat is None:
        return None
    return (ckpt, prefix, takeover, total, declared, repeat)


def ancestor_descriptor(root: dict) -> dict:
    """The play-root fields of one root, ``<absent>`` for anything undeclared.

    ``ANCESTOR_INCONSISTENT`` compares these across every root that carries one
    ``ancestor_id``: a checkpoint, a repeat number, a seed, the checkpoint's own
    seed and the **budget tier** may differ inside one ancestor (§2's
    3-checkpoint x 2-RNG grid, measured at 1,800 *and* 2,400 frames), the
    play-root itself may not.  ``stage`` / ``takeover_frame`` / ``total_frames`` /
    ``parent_prefix_digest`` are required, so they are always present (possibly
    with an invalid value, which is already reported field by field); ``window``
    and the tape are optional in this contract -- ``v829_pair.py`` accepts a window
    from the root or from ``--window`` -- so a field is compared only as "what
    this root says", and a sibling that says nothing prints ``<absent>``.  That is
    deliberate: two roots claiming to describe the same play-root must agree on
    *whether* they name the tape as well as on what they name, otherwise the guard
    could not tell "same root, one label omits the tape" from "two different
    roots".  ``pair_index`` is absent from this descriptor on purpose (it is the
    repeat axis and must differ between the two RNG runs) and so is
    ``budget_frames`` (the endpoint axis; ``ANCESTOR_BUDGET_SET`` judges it as a
    set over the (checkpoint, repeat) cells).
    """
    tape = _ABSENT
    for name in TAPE_FIELDS:
        if name in root:
            tape = root[name]
            break
    described = {"tape": tape}
    for name in ("stage", "window", "takeover_frame", "total_frames",
                 "parent_prefix_digest"):
        described[name] = root[name] if name in root else _ABSENT
    return described


def _repeated(values: list) -> list:
    """Distinct values occurring more than once, in first-appearance order.

    De-duplicated by ``repr`` so an unhashable ``pair_rng`` cannot crash the
    cross-root check before it gets the chance to refuse the manifest.
    """
    repeated: list = []
    seen: set[str] = set()
    for value in values:
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        if values.count(value) > 1:
            repeated.append(value)
    return repeated


def _rng_scope_slot(key: tuple) -> str:
    ancestor, declared_budget, ckpt, repeat = key
    return (f"seed slot of ancestor {ancestor!r} budget {declared_budget} "
            f"ckpt {ckpt[:12]}... repeat {repeat}")


def _rng_scope_ancestor(key: tuple) -> str:
    ancestor, declared_budget, ckpt = key
    return f"ancestor {ancestor!r} budget {declared_budget} ckpt {ckpt[:12]}..."


def _rng_scope_window(key: tuple) -> str:
    ckpt, prefix, takeover, total, declared_budget, repeat = key
    return (f"window ckpt {ckpt[:12]}... prefix {prefix[:12]}... takeover {takeover} "
            f"total {total} budget {declared_budget} repeat {repeat}")


# --------------------------------------------------------------------------- #
# findings
# --------------------------------------------------------------------------- #
def finding(code: str, detail: str, ancestor_id: str | None = None,
            root_index: int | None = None) -> dict:
    entry = {"code": code, "detail": detail}
    if ancestor_id is not None:
        entry["ancestor_id"] = ancestor_id
    if root_index is not None:
        entry["root_index"] = root_index
    return entry


def sort_findings(findings: list[dict]) -> list[dict]:
    rank = {code: index for index, code in enumerate(CODE_ORDER)}

    def key(entry: dict) -> tuple:
        return (rank.get(entry["code"], len(CODE_ORDER)),
                str(entry.get("ancestor_id", "")),
                entry.get("root_index", -1),
                entry["detail"])

    return sorted(findings, key=key)


def code_set(findings: list[dict]) -> list[str]:
    present = {entry["code"] for entry in findings}
    ordered = [code for code in CODE_ORDER if code in present]
    return ordered + sorted(present.difference(CODE_ORDER))


# --------------------------------------------------------------------------- #
# the audit itself -- pure, no I/O, no printing
# --------------------------------------------------------------------------- #
def audit(manifest: object, *, budget: int = DEFAULT_BUDGET,
          allow_exposed: bool = False) -> dict:
    """Return the audit document for an already-parsed manifest.

    ``budget`` must be one of :data:`BUDGET_TIERS`; anything else yields a
    ``BUDGET_INVALID`` finding and no manifest verdict at all (``evaluate`` maps
    it to exit 2).  ``audit`` cannot detect duplicate JSON keys -- a ``dict`` has
    already collapsed them -- so byte-level callers must use ``evaluate``.
    """
    findings: list[dict] = []
    checks: list[str] = []
    ignored: list[str] = []
    roots_count = 0

    # ---- gate parameter: an unusable budget is a refusal, never a no-op -----
    problem = budget_problem(budget)
    if problem is not None:
        checks.append("budget.tier")
        findings.append(finding("BUDGET_INVALID", problem))
        return _document(findings, checks, ignored, 0, budget, allow_exposed)

    if not isinstance(manifest, dict):
        findings.append(finding(
            "MANIFEST_UNREADABLE",
            f"manifest must be a JSON object, got {type_name(manifest)}"))
        checks.append("manifest.is_object")
        return _document(findings, checks, ignored, 0, budget, allow_exposed)

    # ---- schema gate: if the contract is unknown, per-field semantics are too --
    checks.append("schema.present")
    if "schema" not in manifest:
        findings.append(finding(
            "SCHEMA_MISSING",
            f"manifest has no 'schema' key; expected {SCHEMA!r}"))
        return _document(findings, checks, ignored, 0, budget, allow_exposed)
    checks.append("schema.supported")
    if manifest["schema"] != SCHEMA:
        findings.append(finding(
            "SCHEMA_UNSUPPORTED",
            f"schema is {manifest['schema']!r}; this guard only reads {SCHEMA!r}"))
        return _document(findings, checks, ignored, 0, budget, allow_exposed)

    for key in NOT_INHERITED:
        if key in manifest:
            ignored.append(key)

    # ---- roots container ---------------------------------------------------
    checks.append("roots.present")
    roots = manifest.get("roots")
    if roots is None:
        findings.append(finding(
            "FIELD_MISSING", "manifest has no 'roots' array; there is nothing to admit"))
        return _document(findings, checks, ignored, 0, budget, allow_exposed)
    if not isinstance(roots, list):
        findings.append(finding(
            "FIELD_INVALID", f"'roots' must be an array, got {type_name(roots)}"))
        return _document(findings, checks, ignored, 0, budget, allow_exposed)
    if not roots:
        findings.append(finding(
            "FIELD_INVALID",
            "'roots' is empty; an empty manifest admits no window and must not pass"))
        return _document(findings, checks, ignored, 0, budget, allow_exposed)
    roots_count = len(roots)

    # ---- per-root checks ---------------------------------------------------
    # ``seen_ids``       (ancestor_id, ckpt_sha256, pair_index, budget_frames) ->
    #                    roots: one cell at one tier once
    # ``ancestor_groups`` ancestor_id -> roots: the play-root consistency check
    # ``budget_axis``    ancestor_id -> (root index, usable tier, ckpt, repeat):
    #                    the endpoint axis, judged as a set per cell
    seen_ids: dict[tuple, list[int]] = {}
    ancestor_groups: dict[str, list[int]] = {}
    budget_axis: dict[str, list[tuple[int, int | None, str | None, int | None]]] = {}
    descriptors: dict[int, dict] = {}
    labels: dict[int, str] = {}
    windows: dict[tuple, list[int]] = {}
    rng_by_slot: dict[tuple, list[tuple[int, object]]] = {}
    rng_by_ancestor: dict[tuple, list[tuple[int, object]]] = {}
    rng_by_window: dict[tuple, list[tuple[int, object]]] = {}

    for index, root in enumerate(roots):
        label = f"roots[{index}]"
        if not isinstance(root, dict):
            findings.append(finding(
                "FIELD_INVALID", f"{label} must be an object, got {type_name(root)}",
                root_index=index))
            continue

        ancestor = root.get("ancestor_id")
        ancestor_label = ancestor if is_text(ancestor) else f"<{label}>"
        labels[index] = ancestor_label

        # ---- 1. required fields, named one per finding ---------------------
        checks.append(f"{label}.required_fields")
        missing = [name for name in REQUIRED_ROOT_FIELDS if name not in root]
        for name in missing:
            findings.append(finding(
                "FIELD_MISSING",
                f"{label} is missing required field {name!r}",
                ancestor_label, index))

        # ---- 2. the verify anchor is an object with a prefix digest ---------
        checks.append(f"{label}.anchor_fields")
        anchor = root.get("verify_anchor")
        anchor_ok = False
        if "verify_anchor" in root and isinstance(anchor, dict):
            anchor_ok = True
            if ANCHOR_KEY not in anchor:
                findings.append(finding(
                    "FIELD_MISSING",
                    f"{label}.verify_anchor is missing required field {ANCHOR_KEY!r}",
                    ancestor_label, index))
        elif "verify_anchor" in root:
            findings.append(finding(
                "FIELD_INVALID",
                f"{label}.verify_anchor must be an object, got {type_name(anchor)}",
                ancestor_label, index))
            anchor_ok = True  # the object-type error is already reported

        # ---- 3. types ------------------------------------------------------
        checks.append(f"{label}.field_types")
        for name in INT_FIELDS:
            if name not in root:
                continue
            value = root[name]
            if not is_int(value):
                findings.append(finding(
                    "FIELD_INVALID",
                    f"{label}.{name} must be an integer, got {type_name(value)} "
                    f"({value!r})", ancestor_label, index))
            elif name == "takeover_frame" and value < 0:
                findings.append(finding(
                    "FIELD_INVALID",
                    f"{label}.takeover_frame must be >= 0, got {value}",
                    ancestor_label, index))
            elif name in ("total_frames", "budget_frames") and value <= 0:
                findings.append(finding(
                    "FIELD_INVALID",
                    f"{label}.{name} must be > 0, got {value}", ancestor_label, index))
            elif name == "pair_index" and not 0 <= value < MAX_PAIR_INDEX:
                findings.append(finding(
                    "FIELD_INVALID",
                    f"{label}.pair_index must satisfy 0 <= pair_index < "
                    f"{MAX_PAIR_INDEX} (the pair-RNG repeat axis has at most "
                    f"{MAX_PAIR_INDEX} slots), got {value}",
                    ancestor_label, index))
            elif name in ("pair_rng", "ckpt_seed") and value < 0:
                findings.append(finding(
                    "FIELD_INVALID",
                    f"{label}.{name} must be >= 0, got {value}",
                    ancestor_label, index))

        for name in TEXT_FIELDS:
            if name in root and not is_text(root[name]):
                findings.append(finding(
                    "FIELD_INVALID",
                    f"{label}.{name} must be a non-empty string, "
                    f"got {type_name(root[name])} ({root[name]!r})",
                    ancestor_label, index))

        for name in DIGEST_FIELDS:
            if name in root and not is_digest(root[name]):
                findings.append(finding(
                    "FIELD_INVALID",
                    f"{label}.{name} must be a bare 64-character lowercase sha256 hex "
                    f"digest, got {type_name(root[name])} ({root[name]!r})",
                    ancestor_label, index))
        if anchor_ok and ANCHOR_KEY in (anchor if isinstance(anchor, dict) else {}):
            value = anchor[ANCHOR_KEY]
            if not is_digest(value):
                findings.append(finding(
                    "FIELD_INVALID",
                    f"{label}.verify_anchor.{ANCHOR_KEY} must be a bare 64-character "
                    f"lowercase sha256 hex digest, got {type_name(value)} ({value!r})",
                    ancestor_label, index))

        # ---- 4. exposure: unknown is refused, exposed is refused -----------
        checks.append(f"{label}.exposure")
        exposed_to = []
        for name in EXPOSURE_FIELDS:
            if name not in root:
                findings.append(finding(
                    "EXPOSURE_UNKNOWN",
                    f"{label}.{name} is missing; every root must declare "
                    f"{' and '.join(EXPOSURE_FIELDS)} as booleans, because an "
                    f"undeclared exposure cannot be ruled out",
                    ancestor_label, index))
            elif not isinstance(root[name], bool):
                findings.append(finding(
                    "EXPOSURE_UNKNOWN",
                    f"{label}.{name} is {type_name(root[name])} ({root[name]!r}); it "
                    f"must be a boolean so 'no exposure' is a positive claim",
                    ancestor_label, index))
            elif root[name]:
                exposed_to.append(name)
        if exposed_to and not allow_exposed:
            findings.append(finding(
                "EXPOSURE_CONFLICT",
                f"{label} declares {' and '.join(exposed_to)} = true: this ancestor "
                f"has already been seen, so it cannot be a new development root; "
                f"re-run with --allow-exposed only if that exposure is accepted",
                ancestor_label, index))

        # ---- 5. budget ceilings, declared tier and requested tier ----------
        checks.append(f"{label}.budget_ceiling")
        takeover = root.get("takeover_frame")
        total = root.get("total_frames")
        declared = root.get("budget_frames")
        if is_int(takeover) and takeover >= 0 and is_int(total) and total > 0:
            tiers: dict[int, list[str]] = {}
            if is_int(declared) and declared > 0:
                tiers.setdefault(declared, []).append("declared")
            # `budget` passed budget_problem() at the top of audit() and is always
            # one of BUDGET_TIERS, so the requested tier is always checked: there
            # is no "if the budget looks odd, quietly skip it" branch left.
            tiers.setdefault(budget, []).append("requested")
            for tier in sorted(tiers):
                needed = takeover + tier
                if needed > total:
                    sources = "+".join(sorted(set(tiers[tier])))
                    findings.append(finding(
                        "BUDGET_EXCEEDS_FRAMES",
                        f"{label}: takeover_frame {takeover} + {tier} frames "
                        f"({sources}) = {needed} > total_frames {total}, short by "
                        f"{needed - total} frames",
                        ancestor_label, index))

        # ---- 6. digest chain ------------------------------------------------
        checks.append(f"{label}.digest_chain")
        parent = root.get("parent_prefix_digest")
        if is_digest(parent) and isinstance(anchor, dict) and is_digest(anchor.get(ANCHOR_KEY)):
            if parent != anchor[ANCHOR_KEY]:
                findings.append(finding(
                    "DIGEST_MISMATCH",
                    f"{label}: parent_prefix_digest {parent} != "
                    f"verify_anchor.{ANCHOR_KEY} {anchor[ANCHOR_KEY]}; the recovery "
                    f"anchor would restore a different prefix than the one declared",
                    ancestor_label, index))

        # ---- collect for the cross-root checks -----------------------------
        # `ancestor_id` is free text, so it is stripped before it is used as a
        # key; every finding still prints the raw value (see `labels`).
        if is_text(ancestor):
            ancestor_key = ancestor.strip()
            # Every root of one ancestor, for ANCESTOR_INCONSISTENT: that check
            # must see the checkpoints, repeats and budget tiers of one ancestor as
            # siblings, so it groups by the label alone.
            ancestor_groups.setdefault(ancestor_key, []).append(index)
            descriptors[index] = ancestor_descriptor(root)
            # The endpoint axis.  A tier that is not a usable positive integer is
            # already FIELD_INVALID, so it is recorded as None and seeds no
            # cross-root key; the checkpoint and the repeat number are recorded the
            # same way, because a cell can only be named when both are comparable.
            repeat = pair_index_of(root)
            budget_axis.setdefault(ancestor_key, []).append((
                index,
                declared if is_int(declared) and declared > 0 else None,
                root.get("ckpt_sha256") if is_digest(root.get("ckpt_sha256")) else None,
                repeat))
            # The identity DUPLICATE_ANCESTOR refuses on is the cell
            # (label, checkpoint, repeat number, budget tier): one measurement is
            # one cell at one tier.  The key is only built when the checkpoint is a
            # real digest and the repeat number and the tier are usable -- a root
            # with a broken ckpt_sha256, an out-of-range pair_index or a zero
            # budget is already reported FIELD_INVALID and must not seed a
            # cross-root key from a value nobody can compare.
            if (is_digest(root.get("ckpt_sha256")) and repeat is not None
                    and is_int(declared) and declared > 0):
                seen_ids.setdefault(
                    (ancestor_key, root["ckpt_sha256"], repeat, declared),
                    []).append(index)
                if "pair_rng" in root:
                    # Two seed families: the cell slot (label, budget, checkpoint,
                    # repeat) and the coarser ancestor family (the same minus the
                    # repeat) that makes a repeat number no licence for a reused
                    # seed.  See the module docstring.
                    key = (ancestor_key, declared, root["ckpt_sha256"])
                    rng_by_ancestor.setdefault(key, []).append(
                        (index, root.get("pair_rng")))
                    slot = key + (repeat,)
                    rng_by_slot.setdefault(slot, []).append((index, root.get("pair_rng")))
        window = window_identity(root)
        if window is not None:
            windows.setdefault(window, []).append(index)
            if "pair_rng" in root:
                rng_by_window.setdefault(window, []).append((index, root.get("pair_rng")))

    # ---- cross-root: every root of one ancestor describes one play-root ----
    checks.append("roots.ancestor_consistency")
    for ancestor in sorted(ancestor_groups):
        indices = ancestor_groups[ancestor]
        if len(indices) < 2:
            continue
        disagreements = []
        for name in ANCESTOR_INVARIANT_FIELDS:
            values = [(index, descriptors[index][name]) for index in indices]
            distinct = {repr(value) for _, value in values}
            if len(distinct) > 1:
                shown = ", ".join(f"roots[{index}]={value!r}" for index, value in values)
                disagreements.append(f"{name} ({shown})")
        if not disagreements:
            continue
        findings.append(finding(
            "ANCESTOR_INCONSISTENT",
            f"ancestor_id {ancestor!r} appears on roots{sorted(indices)}, but those "
            f"roots disagree about the play-root itself: {'; '.join(disagreements)}. "
            f"One ancestor is one play-root, so its checkpoint, its pair_rng, its "
            f"repeat number pair_index and its budget tier may differ (those are the "
            f"section 2 three-checkpoint x two-RNG axes and the 30 s/40 s endpoint "
            f"axis) while the tape, takeover frame, total frames, stage, window and "
            f"prefix digest may not",
            labels[indices[0]]))

    # ---- cross-root: the budget tiers of one ancestor form a set -----------
    # `budget_frames` is §2's endpoint axis (1,800 = 30 s, 2,400 = 40 s), so it is
    # free to differ across the roots of one ancestor -- but the tiers it names
    # have to be pre-registered, and a tier is one endpoint: the same tier twice on
    # one (checkpoint, repeat) cell is one measurement counted twice, not a third
    # endpoint.  Per *cell*, not per ancestor: the pre-registered grid already
    # gives one ancestor 3 checkpoints x 2 pair RNGs = 6 roots, and every one of
    # them declares the same tier (the 24- and 48-root manifests on disk), which is
    # legal and must stay legal.
    checks.append("roots.ancestor_budget_set")
    for ancestor in sorted(budget_axis):
        entries = budget_axis[ancestor]
        outside = [(index, value) for index, value, _ckpt, _repeat in entries
                   if value is not None and value not in ANCESTOR_BUDGET_TIERS]
        if outside:
            shown = ", ".join(f"roots[{index}]={value!r}" for index, value in outside)
            findings.append(finding(
                "ANCESTOR_BUDGET_SET",
                f"ancestor_id {ancestor!r}: {shown} declare(s) a budget_frames "
                f"outside the pre-registered tiers {ANCESTOR_BUDGET_TIERS}. The "
                f"budget is section 2's endpoint axis (1,800 frames = 30 s, 2,400 = "
                f"40 s), so a third value names no endpoint the report can average",
                labels[outside[0][0]]))
        cells: dict[tuple, list[tuple[int, int]]] = {}
        for index, value, ckpt, repeat in entries:
            if value is None or ckpt is None or repeat is None:
                continue
            cells.setdefault((ckpt, repeat), []).append((index, value))
        for cell in sorted(cells):
            pairs = cells[cell]
            for value in _repeated([item for _index, item in pairs]):
                holders = [index for index, item in pairs if item == value]
                findings.append(finding(
                    "ANCESTOR_BUDGET_SET",
                    f"ancestor_id {ancestor!r} measures the cell (ckpt "
                    f"{cell[0][:12]}..., pair_index {cell[1]}) at budget_frames "
                    f"{value} on roots{holders}: the tiers are a SET per "
                    f"(checkpoint, repeat) cell -- budget {value} is one endpoint "
                    f"(section 2), so the same tier twice is one measurement counted "
                    f"twice rather than a second endpoint. Two *different* tiers of "
                    f"one cell (1,800 + 2,400 from one take-over state) are the "
                    f"pre-registered 30 s/40 s pair and pass",
                    labels[holders[0]]))

    # ---- cross-root: one cell per (ancestor, checkpoint, repeat, tier) ------
    checks.append("roots.unique_ancestor")
    for key in sorted(seen_ids):
        indices = seen_ids[key]
        if len(indices) < 2:
            continue
        ancestor, ckpt, repeat, tier = key
        raw_labels = sorted({labels[index] for index in indices})
        tail = ("; the raw labels "
                f"{raw_labels} collapse to that key after str.strip()"
                if len(raw_labels) > 1 else "")
        findings.append(finding(
            "DUPLICATE_ANCESTOR",
            f"ancestor_id {ancestor!r} with ckpt {ckpt[:12]}... at repeat "
            f"pair_index {repeat} and budget_frames {tier} appears {len(indices)} "
            f"times (roots{sorted(indices)}){tail}; section 2 of the pre-registration "
            f"admits one cell per (independent ancestor, checkpoint, pair RNG) at "
            f"each budget, and a repeat is the same cell counted twice. A *different* "
            f"checkpoint, a *different* pair_index or a *different* budget tier "
            f"under this ancestor is a section 2 repeat/endpoint axis, not a second "
            f"ancestor"))

    # ---- cross-root: one physical measurement per manifest ------------------
    checks.append("roots.unique_window")
    for window in sorted(windows):
        indices = windows[window]
        if len(indices) < 2:
            continue
        ckpt, prefix, takeover, total, declared_budget, repeat = window
        raw_labels = sorted({labels[index] for index in indices})
        findings.append(finding(
            "DUPLICATE_WINDOW",
            f"the measurement (ckpt {ckpt[:12]}..., parent_prefix {prefix[:12]}..., "
            f"takeover_frame {takeover}, total_frames {total}, budget_frames "
            f"{declared_budget}, pair_index {repeat}) appears {len(indices)} times "
            f"(roots{sorted(indices)}) under labels {raw_labels}; a different "
            f"ancestor_id does not make it a second measurement, and section 4 would "
            f"average the same measurement twice. The same window at a *different* "
            f"pair_index is the section 2 second axis and is not a duplicate"))

    # ---- cross-root: one RNG per cell, addressed three ways ----------------
    # The same cell can be named three ways: by its seed slot (ancestor, budget,
    # checkpoint, repeat number), by the coarser ancestor family (the same minus the
    # repeat number), and by the six window fields.  The slot family names "this
    # allocation was entered twice"; the ancestor family makes a declared repeat
    # number no licence for a reused seed; the window family makes a re-labelled
    # copy with a reused seed visible.  A hit from several families on the same
    # roots and value is merged into a single finding.
    checks.append("roots.unique_rng")
    merged: dict[tuple, dict] = {}
    for scope_of, groups in ((_rng_scope_slot, rng_by_slot),
                             (_rng_scope_ancestor, rng_by_ancestor),
                             (_rng_scope_window, rng_by_window)):
        for key in sorted(groups, key=repr):
            entries = groups[key]
            if len(entries) < 2:
                continue
            values = [value for _, value in entries]
            for value in _repeated(values):
                hit = tuple(index for index, item in entries if item == value)
                slot = merged.setdefault((repr(value), frozenset(hit)),
                                         {"value": value, "indices": hit, "scopes": []})
                scope = scope_of(key)
                if scope not in slot["scopes"]:
                    slot["scopes"].append(scope)
    for slot_key in sorted(merged, key=repr):
        slot = merged[slot_key]
        placements = ", ".join(f"roots[{index}]={slot['value']!r}"
                               for index in slot["indices"])
        scopes = " and ".join(slot["scopes"])
        findings.append(finding(
            "RNG_REUSED",
            f"{placements} share pair_rng {slot['value']!r} ({scopes}); section 1 of "
            f"the pre-registration pre-allocates a unique seed per "
            f"(ancestor, budget, checkpoint, repeat)"))

    return _document(findings, checks, ignored, roots_count, budget, allow_exposed)


def _document(findings: list[dict], checks: list[str], ignored: list[str],
              roots_count: int, budget: int, allow_exposed: bool) -> dict:
    ordered = sort_findings(findings)
    return {
        "guard": GUARD,
        "guard_version": GUARD_VERSION,
        "schema": SCHEMA,
        "options": {
            "budget": budget,
            "allow_exposed": allow_exposed,
            "budget_note": (
                "the requested tier is checked against every root in addition to the "
                "tier it declares; the default is the 2,400-frame admission bar"
            ),
        },
        "ok": not ordered,
        "roots": roots_count,
        "checks_run": sorted(set(checks)),
        "check_count": len(set(checks)),
        "ignored_top_level": sorted(set(ignored)),
        "findings": ordered,
        "codes": code_set(ordered),
    }


# --------------------------------------------------------------------------- #
# loading and one check run
# --------------------------------------------------------------------------- #
def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """``object_pairs_hook`` that refuses a repeated key instead of last-wins.

    ``json.loads`` keeps the last value of a duplicate key, so a manifest saying
    ``"takeover_frame": 999999`` and later ``"takeover_frame": 8663`` would be
    judged on 8663 alone -- precisely the "contradictory value" this module
    refuses to resolve by inference.  The hook runs for every object in the file
    (top level, each root, ``verify_anchor``), so no level is left ambiguous.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"duplicate JSON key {key!r}: one object cannot carry two values "
                f"for the same field, and picking one would be an inference")
        result[key] = value
    return result


def evaluate(path: pathlib.Path, *, budget: int = DEFAULT_BUDGET,
             allow_exposed: bool = False) -> tuple[dict, int]:
    """Audit ``path`` and return ``(document, exit_code)`` without printing."""
    document = {"guard": GUARD, "guard_version": GUARD_VERSION, "schema": SCHEMA,
                "options": {"budget": budget, "allow_exposed": allow_exposed},
                "manifest_path": str(path)}
    try:
        raw = path.read_bytes()
    except OSError as error:
        document.update({
            "ok": False, "roots": 0, "checks_run": ["manifest.readable"],
            "check_count": 1, "ignored_top_level": [],
            "findings": [finding("MANIFEST_UNREADABLE",
                                 f"cannot read {path}: {error.strerror or error}")],
            "codes": ["MANIFEST_UNREADABLE"]})
        return document, 2

    document["manifest_bytes"] = len(raw)
    document["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    document["manifest_sha256_scope"] = "raw file bytes, no normalisation"
    try:
        parsed = json.loads(raw.decode("utf-8-sig"),
                            object_pairs_hook=_object_without_duplicate_keys)
    except (UnicodeDecodeError, ValueError) as error:
        # json.JSONDecodeError is a ValueError subclass, so this one clause covers
        # undecodable bytes, malformed JSON and a repeated key alike.
        document.update({
            "ok": False, "roots": 0, "checks_run": ["manifest.parses"],
            "check_count": 1, "ignored_top_level": [],
            "findings": [finding("MANIFEST_UNREADABLE",
                                 f"{path} is not a JSON document this guard can "
                                 f"trust: {error}")],
            "codes": ["MANIFEST_UNREADABLE"]})
        return document, 2

    report = audit(parsed, budget=budget, allow_exposed=allow_exposed)
    document.update({key: value for key, value in report.items()
                     if key not in ("guard", "guard_version", "schema", "options")})
    if document["ok"]:
        return document, 0
    fatal = any(entry["code"] in FATAL_CODES for entry in document["findings"])
    return document, 2 if fatal else 1


def render(document: dict) -> str:
    """Human-readable refusal report; the JSON summary is printed after it."""
    lines = [f"v829 guard: {document.get('manifest_path', '<memory>')}",
             f"  sha256  {document.get('manifest_sha256', '<none>')}",
             f"  options budget={document['options']['budget']} "
             f"allow_exposed={document['options']['allow_exposed']}",
             f"  schema  {document['schema']}",
             f"  roots   {document['roots']}",
             f"  checks  {document['check_count']}"]
    if document["ignored_top_level"]:
        lines.append(f"  ignored top-level keys (NOT inherited by any root): "
                     f"{', '.join(document['ignored_top_level'])}")
    if document["ok"]:
        lines.append("PASS: every check passed")
    else:
        lines.append(f"REFUSED: {len(document['findings'])} finding(s)")
        for entry in document["findings"]:
            where = entry.get("ancestor_id", "")
            if where:
                lines.append(f"  [{entry['code']}] ancestor {where}")
            else:
                lines.append(f"  [{entry['code']}]")
            lines.append(f"      {entry['detail']}")
    return "\n".join(lines)


def run_check(path: pathlib.Path, *, budget: int, allow_exposed: bool, as_json: bool) -> int:
    document, code = evaluate(path, budget=budget, allow_exposed=allow_exposed)
    if as_json:
        print(json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False))
    else:
        print(render(document))
        print("--- json summary ---")
        print(json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False))
    if code == 2:
        print(f"cannot evaluate {path}: see the findings above (exit 2 = the bytes "
              f"or the gate options were refused, not a manifest verdict)",
              file=sys.stderr)
    return code


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _root(ancestor: str, *, takeover: int = 8663, total: int = 36000,
          budget: int = 2400, rng: int = 20261123, stage: str = "stage1-normal",
          training: object = False, selection: object = False,
          parent: str | None = None, anchor: str | None = None,
          ckpt: str | None = None, window: int | None = None,
          tape: str | None = None, pair_index: int | None = None,
          ckpt_seed: int | None = None) -> dict:
    """One synthetic root.  ``ckpt`` names a checkpoint *inside* ``ancestor``.

    The digests are salted with ``ancestor.strip()``, not with the raw label, so a
    label carrying a trailing space still describes the same play-root and the same
    tape -- which is what lets the ``label_space`` case test ``str.strip()`` instead
    of accidentally testing two different recordings.
    """
    key = ancestor.strip()
    shared = _digest(f"prefix|{key}")
    root = {
        "ancestor_id": ancestor,
        "stage": stage,
        "budget_frames": budget,
        "ckpt_sha256": _digest(f"ckpt|{key if ckpt is None else ckpt}"),
        "engine_sha256": _digest("engine|th08-modern"),
        "pair_rng": rng,
        "takeover_frame": takeover,
        "total_frames": total,
        "parent_prefix_digest": shared if parent is None else parent,
        "verify_anchor": {"prefix_digest": shared if anchor is None else anchor},
        "used_for_training": training,
        "used_for_selection": selection,
    }
    if window is not None:
        root["window"] = window
    if tape is not None:
        root["prefix_tape"] = tape
    if pair_index is not None:
        root["pair_index"] = pair_index
    if ckpt_seed is not None:
        root["ckpt_seed"] = ckpt_seed
    return root


def _manifest(roots: list) -> dict:
    return {"schema": SCHEMA, "manifest_id": "v829-pair-001-selftest", "roots": roots}


def _selftest_cases() -> list[tuple[str, dict, dict, set, int]]:
    """(name, manifest, options, expected code set, expected exit code)."""
    cases: list[tuple[str, dict, dict, set, int]] = []

    # 1. legal manifest: two roots, both fit 2,400, no exposure, unique rng
    cases.append((
        "valid", _manifest([_root("anc-a", takeover=8663, total=36000),
                            _root("anc-b", takeover=3000, total=20000, rng=7)]),
        {}, set(), 0))

    # 2. schema key absent
    legal = _manifest([_root("anc-a")])
    no_schema = {"manifest_id": "no-schema", "roots": legal["roots"]}
    cases.append(("schema_missing", no_schema, {}, {"SCHEMA_MISSING"}, 1))

    # 3. schema present but a different version
    wrong = _manifest([_root("anc-a")])
    wrong["schema"] = "v829-pair-manifest/2"
    cases.append(("schema_unsupported", wrong, {}, {"SCHEMA_UNSUPPORTED"}, 1))

    # 4. field errors: one root missing six fields, one root with wrong types
    thin = _root("anc-thin")
    for name in ("stage", "pair_rng", "takeover_frame", "total_frames",
                 "parent_prefix_digest", "verify_anchor"):
        del thin[name]
    mistyped = _root("anc-typed")
    mistyped["takeover_frame"] = "8663"          # string, not integer
    mistyped["total_frames"] = True              # bool sneaking in as int
    mistyped["budget_frames"] = 0                # not > 0
    mistyped["ckpt_sha256"] = "deadbeef"         # not a 64-hex digest
    cases.append(("fields", _manifest([thin, mistyped]), {},
                  {"FIELD_MISSING", "FIELD_INVALID"}, 1))

    # 5. budget: anc-a fits 1,800 but not 2,400; anc-b overflows its own 2,400
    cases.append((
        "budget", _manifest([_root("anc-a", takeover=16000, total=18000, budget=1800),
                             _root("anc-b", takeover=17000, total=18000, budget=2400)]),
        {}, {"BUDGET_EXCEEDS_FRAMES"}, 1))
    cases.append((
        "budget@1800", _manifest([_root("anc-a", takeover=16000, total=18000, budget=1800),
                                  _root("anc-b", takeover=17000, total=18000, budget=2400)]),
        {"budget": 1800}, {"BUDGET_EXCEEDS_FRAMES"}, 1))

    # 6. exposure: unknown for one root, exposed for the other
    exposed = _root("anc-exposed", training=True, selection=False)
    unknown = _root("anc-unknown")
    del unknown["used_for_training"]
    unknown["used_for_selection"] = "false"       # a string is not a boolean
    cases.append(("exposure", _manifest([exposed, unknown]), {},
                  {"EXPOSURE_CONFLICT", "EXPOSURE_UNKNOWN"}, 1))
    cases.append(("exposure@allow", _manifest([exposed, unknown]),
                  {"allow_exposed": True}, {"EXPOSURE_UNKNOWN"}, 1))

    # 7. digest chain: parent digest disagrees with the verify anchor
    broken = _root("anc-digest")
    broken["parent_prefix_digest"] = _digest("some-other-prefix")
    cases.append(("digest", _manifest([broken]), {}, {"DIGEST_MISMATCH"}, 1))

    # 8. uniqueness: a repeated ancestor that repeats its window *and* its rng --
    #    identical copies now trip every cross-root gate, which is the point.  The
    #    two copies also share the tier of one cell, so ANCESTOR_BUDGET_SET reports
    #    the endpoint axis too (a tier is one endpoint, not two).
    first = _root("anc-dup", rng=99)
    second = _root("anc-dup", rng=99)
    third = _root("anc-dup", rng=100)             # different rng: only the id repeats
    cases.append(("uniqueness", _manifest([first, second, third]), {},
                  {"ANCESTOR_BUDGET_SET", "DUPLICATE_ANCESTOR", "DUPLICATE_WINDOW",
                   "RNG_REUSED"}, 1))

    # 9. the 2,400 admission bar is the default: this window fits the secondary
    #    1,800 endpoint but not 2,400, so a default of anything else would let a
    #    manifest through that section 2 refuses
    cases.append((
        "default_bar", _manifest([_root("anc-bar", takeover=16000, total=18000,
                                        budget=1800)]),
        {}, {"BUDGET_EXCEEDS_FRAMES"}, 1))

    # 10. the boundary must pass: takeover_frame + budget_frames == total_frames
    cases.append((
        "boundary_equal", _manifest([_root("anc-edge", takeover=9600, total=12000,
                                           budget=2400)]),
        {}, set(), 0))

    # 11. bool is not an integer: each root fails only because of a bool (the
    #     chosen values are silent if a bool is accepted, so the code set would
    #     collapse to [] and this case is what notices)
    bools = []
    for name, field, value in (("anc-bool-takeover", "takeover_frame", True),
                               ("anc-bool-rng", "pair_rng", True),
                               ("anc-bool-budget", "budget_frames", True)):
        item = _root(name)
        item[field] = value
        bools.append(item)
    cases.append(("bool_is_not_int", _manifest(bools), {}, {"FIELD_INVALID"}, 1))

    # 12. an uppercase digest is not a digest (PowerShell Get-FileHash prints
    #     uppercase; the frozen contract is lowercase, so this must be refused)
    upper = _root("anc-upper")
    upper["ckpt_sha256"] = upper["ckpt_sha256"].upper()
    cases.append(("digest_upper", _manifest([upper]), {}, {"FIELD_INVALID"}, 1))

    # 13. exposure must be a real boolean: "false" and 0 are claims, not "no"
    sneaky = _root("anc-sneaky")
    sneaky["used_for_training"] = "false"
    sneaky["used_for_selection"] = 0
    cases.append(("exposure_not_bool", _manifest([sneaky]), {},
                  {"EXPOSURE_UNKNOWN"}, 1))

    # 14. a relabelled copy of one window: all five window fields and the seed
    #     are identical, only `ancestor_id` changes
    window = _root("anc-x", takeover=9600, total=12000, rng=99)
    relabelled = dict(window)
    relabelled["ancestor_id"] = "anc-y"
    cases.append(("window_relabel", _manifest([window, relabelled]), {},
                  {"DUPLICATE_WINDOW", "RNG_REUSED"}, 1))

    # 15. the same window with a different seed is still one window
    reseeded = dict(window)
    reseeded["ancestor_id"] = "anc-z"
    reseeded["pair_rng"] = 100
    cases.append(("window_other_seed", _manifest([window, reseeded]), {},
                  {"DUPLICATE_WINDOW"}, 1))

    # 16. a trailing space in `ancestor_id` is not a second ancestor
    spaced = _root("anc-sp")
    padded = dict(spaced)
    padded["ancestor_id"] = "anc-sp "
    cases.append(("label_space", _manifest([spaced, padded]), {},
                  {"ANCESTOR_BUDGET_SET", "DUPLICATE_ANCESTOR", "DUPLICATE_WINDOW",
                   "RNG_REUSED"}, 1))

    # 17. strip() must not need the window to match: same label modulo a
    #     trailing space, two genuinely different windows -- the same play-root is
    #     then described two different ways, which is what ANCESTOR_INCONSISTENT is
    #     for (the different seed keeps RNG_REUSED out of this case on purpose)
    left = _root("anc-w", takeover=8663, rng=20261125)
    right = _root("anc-w ", takeover=3000, rng=20261126)
    cases.append(("label_space_distinct_window", _manifest([left, right]), {},
                  {"ANCESTOR_BUDGET_SET", "DUPLICATE_ANCESTOR",
                   "ANCESTOR_INCONSISTENT"}, 1))

    # 18. one ancestor, two windows, one seed: RNG_REUSED has to be able to fire
    #     without DUPLICATE_WINDOW (the ancestor key family still matters).  Two
    #     different takeover frames of one play-root are also the inconsistency
    #     gate's business, and that is the invariant: RNG_REUSED still never
    #     appears alone -- a repeated seed is either the same window
    #     (DUPLICATE_WINDOW) or a different window of one root (this code).
    twin_a = _root("anc-twin", takeover=8663, rng=5)
    twin_b = _root("anc-twin", takeover=9000, rng=5)
    cases.append(("rng_across_windows", _manifest([twin_a, twin_b]), {},
                  {"ANCESTOR_BUDGET_SET", "DUPLICATE_ANCESTOR",
                   "ANCESTOR_INCONSISTENT", "RNG_REUSED"}, 1))

    # 19. the top level is never inherited: the four frame fields live only at
    #     the top, and the root is complete otherwise
    hollow = _root("anc-hollow")
    orphans = {name: hollow[name] for name in FRAME_FIELDS}
    for name in FRAME_FIELDS:
        del hollow[name]
    inherited = _manifest([hollow])
    inherited.update(orphans)
    cases.append(("top_level_not_inherited", inherited, {},
                  {"FIELD_MISSING"}, 1))

    # 20. unknown fields (root level, verify_anchor, top level) are ignored --
    #     forward compatibility is deliberate; silence is the price
    extra = _root("anc-extra")
    extra["window"] = 11
    extra["verify_anchor"] = dict(extra["verify_anchor"], anchor_count=7)
    extras = _manifest([extra])
    extras["unknown_top"] = {"note": "ignored on purpose"}
    cases.append(("unknown_fields", extras, {}, set(), 0))

    # 21.-27. an unusable gate parameter is refused, not skipped.  The manifest
    #     fits 1,800 but not 2,400, so "silently ignoring the budget" would
    #     return exit 0 / ok=True for every one of these.
    gated = _manifest([_root("anc-gate", takeover=16000, total=18000, budget=1800)])
    for name, value in (("budget_float", 2400.0), ("budget_zero", 0),
                        ("budget_negative", -1), ("budget_none", None),
                        ("budget_string", "2400"), ("budget_true", True),
                        ("budget_false", False)):
        cases.append((name, gated, {"budget": value}, {"BUDGET_INVALID"}, 2))

    # 28.-36. the ancestor / repeat-axis semantics: §2 evaluates "3 checkpoints x 2
    #     pair RNGs per ancestor", so the checkpoint is an axis *inside* one
    #     ancestor.  These are the cases the old `DUPLICATE_ANCESTOR` made
    #     inexpressible -- 3 checkpoints under one ancestor_id are one ancestor.
    #
    # 28. the pre-registered shape, one RNG: 3 checkpoints, one ancestor -> passes
    three = [_root("anc-3ckpt", ckpt=f"ckpt|anc-3ckpt|{index}", rng=20262 + index)
             for index in range(3)]
    cases.append(("ancestor_three_ckpt", _manifest(three), {}, set(), 0))

    # 29. the same ancestor at the same checkpoint is one cell, not a second
    #     ancestor: refused whatever the seed says
    same_ckpt = [_root("anc-sameckpt", ckpt="ckpt|anc-sameckpt", rng=20262001),
                 _root("anc-sameckpt", ckpt="ckpt|anc-sameckpt", rng=20262002)]
    cases.append(("ancestor_same_ckpt", _manifest(same_ckpt), {},
                  {"ANCESTOR_BUDGET_SET", "DUPLICATE_ANCESTOR",
                   "DUPLICATE_WINDOW"}, 1))

    # 30. one ancestor cannot be two different roots: a sibling whose takeover
    #     frame contradicts the other two is ANCESTOR_INCONSISTENT (and is NOT
    #     silently treated as a second ancestor)
    skewed = [_root("anc-skew", ckpt=f"ckpt|anc-skew|{index}", rng=20262100 + index)
              for index in range(3)]
    skewed[2]["takeover_frame"] = 9000
    cases.append(("ancestor_takeover_differs", _manifest(skewed), {},
                  {"ANCESTOR_INCONSISTENT"}, 1))

    # 31. one seed for three checkpoints is legal: the checkpoint separates the
    #     cells, so each is its own seed slot
    shared_seed = [_root("anc-shared", ckpt=f"ckpt|anc-shared|{index}",
                         rng=20261123) for index in range(3)]
    cases.append(("ancestor_three_ckpt_one_seed", _manifest(shared_seed), {}, set(), 0))

    # 32. `stage` is part of the play-root too
    staged = [_root("anc-staged", ckpt=f"ckpt|anc-staged|{index}", rng=20262200 + index)
              for index in range(3)]
    staged[1]["stage"] = "stage1-hard-team0"
    cases.append(("ancestor_stage_differs", _manifest(staged), {},
                  {"ANCESTOR_INCONSISTENT"}, 1))

    # 33. the shape `v829_manifest_build.py` now writes: one ancestor, 3
    #     checkpoints, a declared pair_index/ckpt_seed per checkpoint, one window
    built = [_root("anc-built", ckpt=f"ckpt|anc-built|{index}", rng=20265000 + index,
                   window=3201, tape=".tmp_v829_collect/seed3201/labels.jsonl",
                   pair_index=index, ckpt_seed=index) for index in range(3)]
    cases.append(("ancestor_three_ckpt_repeat_axes", _manifest(built), {}, set(), 0))

    # 34. a repeat number is not a licence to reuse a seed: repeat 0 and repeat 1
    #     of one checkpoint are two legal cells, but they still may not share one
    #     allocation.  This is the one shape where RNG_REUSED stands alone -- the
    #     cells differ (so no DUPLICATE_* fires) and the seed is the same.
    axis = [_root("anc-axis", ckpt="ckpt|anc-axis", rng=99, pair_index=0),
            _root("anc-axis", ckpt="ckpt|anc-axis", rng=99, pair_index=1)]
    cases.append(("repeat_axis_not_a_licence", _manifest(axis), {},
                  {"RNG_REUSED"}, 1))

    # 35. the tape must be named by every root of one ancestor or by none: one
    #     sibling declaring `prefix_tape` and the other not is a disagreement about
    #     what the play-root is, not a silent "no tape"
    tape_left = _root("anc-tape", ckpt="ckpt|anc-tape|0", rng=20262300,
                      tape=".tmp_v829_collect/seed3201/labels.jsonl")
    tape_right = _root("anc-tape", ckpt="ckpt|anc-tape|1", rng=20262301)
    cases.append(("ancestor_tape_half_declared", _manifest([tape_left, tape_right]),
                  {}, {"ANCESTOR_INCONSISTENT"}, 1))

    # 36. the two optional repeat axes are integers, not booleans or negatives
    axis_types = _root("anc-axis-types", ckpt="ckpt|anc-axis-types", rng=20262302)
    axis_types["pair_index"] = True                 # bool sneaking in as int
    axis_types["ckpt_seed"] = -1                    # negative
    cases.append(("repeat_axes_types", _manifest([axis_types]), {},
                  {"FIELD_INVALID"}, 1))

    # 37.-44. the second §2 axis: one (ancestor, checkpoint) cell measured at two
    #     pair RNGs.  The cell key is (ancestor, checkpoint, pair_index) and the
    #     window key adds pair_index, so repeat 0 and repeat 1 are two legal cells
    #     while a repeated repeat number is one cell counted twice.
    #
    # 37. the pre-registered shape, two RNGs: 3 checkpoints x repeat 0/1 = 6 roots
    #     under one ancestor, every (checkpoint, repeat) its own seed -> passes.
    #     Before this version all three repeat-1 roots were refused.
    grid = [_root("anc-2rng", ckpt=f"ckpt|anc-2rng|{ckpt_index}",
                  pair_index=rng_index,
                  rng=20265000 + rng_index * 1000 + ckpt_index)
            for ckpt_index in range(3) for rng_index in range(2)]
    cases.append(("pair_rng_two", _manifest(grid), {}, set(), 0))

    # 38. the same cell twice: same ancestor, checkpoint *and* repeat number, two
    #     different seeds -> still one cell, whatever the seed says
    dup_cell = [_root("anc-dup-cell", ckpt="ckpt|anc-dup-cell", pair_index=0,
                      rng=20264001),
                _root("anc-dup-cell", ckpt="ckpt|anc-dup-cell", pair_index=0,
                      rng=20264002)]
    cases.append(("pair_rng_same_cell", _manifest(dup_cell), {},
                  {"ANCESTOR_BUDGET_SET", "DUPLICATE_ANCESTOR",
                   "DUPLICATE_WINDOW"}, 1))

    # 39. the same cell twice *and* one seed: RNG_REUSED joins the two.  Which code
    #     is printed first is asserted on the ordered finding list below
    #     (DUPLICATE_ANCESTOR, per CODE_ORDER).
    dup_seed = [_root("anc-dup-seed", ckpt="ckpt|anc-dup-seed", pair_index=0,
                      rng=20264003),
                _root("anc-dup-seed", ckpt="ckpt|anc-dup-seed", pair_index=0,
                      rng=20264003)]
    cases.append(("pair_rng_same_cell_same_seed", _manifest(dup_seed), {},
                  {"ANCESTOR_BUDGET_SET", "DUPLICATE_ANCESTOR", "DUPLICATE_WINDOW",
                   "RNG_REUSED"}, 1))

    # 40. an absent pair_index means repeat 0: it cannot be counted as a second
    #     axis next to an explicit 0 -- the two rows are one cell
    implicit = [_root("anc-implicit", ckpt="ckpt|anc-implicit", rng=20264004),
                _root("anc-implicit", ckpt="ckpt|anc-implicit", rng=20264005,
                      pair_index=0)]
    cases.append(("pair_index_default_zero", _manifest(implicit), {},
                  {"ANCESTOR_BUDGET_SET", "DUPLICATE_ANCESTOR",
                   "DUPLICATE_WINDOW"}, 1))

    # 41. the repeat axis is an index below MAX_PAIR_INDEX: 16 names no
    #     pre-registered measurement and is refused instead of wrapped
    out_of_range = _root("anc-range", ckpt="ckpt|anc-range", pair_index=16)
    cases.append(("pair_index_out_of_range", _manifest([out_of_range]), {},
                  {"FIELD_INVALID"}, 1))

    # 42. ... and it is an integer, not a boolean
    axis_bool = _root("anc-axis-bool", ckpt="ckpt|anc-axis-bool", pair_index=True)
    cases.append(("pair_index_bool", _manifest([axis_bool]), {},
                  {"FIELD_INVALID"}, 1))

    # 43. the manifest `v829_manifest_build.py --rngs 2` writes: 3 checkpoints x
    #     pair_index 0/1, one shared window label and tape, a ckpt_seed per
    #     checkpoint and one seed per cell -> passes
    built_two = [_root("anc-built-two", ckpt=f"ckpt|anc-built-two|{ckpt_index}",
                       pair_index=rng_index, window=3201,
                       tape=".tmp_v829_collect/seed3201/labels.jsonl",
                       ckpt_seed=ckpt_index,
                       rng=20265000 + rng_index * 1000 + ckpt_index)
                 for ckpt_index in range(3) for rng_index in range(2)]
    cases.append(("pair_rng_built_shape", _manifest(built_two), {}, set(), 0))

    # 44. one cell can be repeated more than twice: pair_index 0/1/2 with three
    #     seeds pass, so the axis is not hard-wired to a pair
    three_repeats = [_root("anc-3rng", ckpt="ckpt|anc-3rng", pair_index=index,
                           rng=20265000 + index) for index in range(3)]
    cases.append(("pair_rng_three", _manifest(three_repeats), {}, set(), 0))

    # 45.-49. §2's ENDPOINT axis: the same take-over state measured at 30 s
    #     (1,800 frames) and at 40 s (2,400).  Until v8.30.0 `budget_frames` was an
    #     ancestor invariant, so the pre-registered 2-root shape below was refused
    #     as ANCESTOR_INCONSISTENT + DUPLICATE_ANCESTOR (repro:
    #     .tmp_v829_info/s20/repro-before.log, exit 1).  The take-over frame is
    #     2,399 because 2399 + 2,400 = 4,799 and the tape holds 4,800 frames.
    #
    # 45. one ancestor, one tape, one takeover_frame, one total_frames, one
    #     checkpoint -- two budget tiers, two seeds -> the pre-registered pair
    endpoint = [_root("anc-endpoint", takeover=2399, total=4800, budget=1800,
                      ckpt="ckpt|anc-endpoint|0", rng=20265000),
                _root("anc-endpoint", takeover=2399, total=4800, budget=2400,
                      ckpt="ckpt|anc-endpoint|0", rng=20265001)]
    cases.append(("budget_endpoint_pair", _manifest(endpoint), {}, set(), 0))

    # 46. the same tier twice on one cell is not a second endpoint: the tier set of
    #     that cell has a repeat (and, budget now being in the cell key, the two
    #     roots are also one duplicate cell)
    twice = [_root("anc-tier-twice", takeover=2399, total=4800, budget=1800,
                   ckpt="ckpt|anc-tier-twice|0", rng=20265002),
             _root("anc-tier-twice", takeover=2399, total=4800, budget=1800,
                   ckpt="ckpt|anc-tier-twice|0", rng=20265003)]
    cases.append(("budget_same_tier_twice", _manifest(twice), {},
                  {"ANCESTOR_BUDGET_SET", "DUPLICATE_ANCESTOR",
                   "DUPLICATE_WINDOW"}, 1))

    # 47. both tiers must come from ONE take-over frame: a 2,400 root whose
    #     takeover_frame contradicts its 1,800 sibling describes a different
    #     play-root, so the endpoint pair is not comparable (and this is NOT
    #     silently read as two ancestors)
    tier_skew = [_root("anc-tier-skew", takeover=2399, total=4800, budget=1800,
                       ckpt="ckpt|anc-tier-skew|0", rng=20265004),
                 _root("anc-tier-skew", takeover=2299, total=4800, budget=2400,
                       ckpt="ckpt|anc-tier-skew|0", rng=20265005)]
    cases.append(("budget_takeover_differs", _manifest(tier_skew), {},
                  {"ANCESTOR_INCONSISTENT"}, 1))

    # 48. the ceiling is judged per root AND per tier: 2,399 + 1,800 = 4,199 fits
    #     while 2,399 + 2,400 = 4,799 still fits, but a takeover of 2,999 leaves
    #     4,800 - 2,999 = 1,801 frames, so only the 2,400 tier overflows
    tier_long = [_root("anc-tier-long", takeover=2999, total=4800, budget=1800,
                       ckpt="ckpt|anc-tier-long|0", rng=20265006),
                 _root("anc-tier-long", takeover=2999, total=4800, budget=2400,
                       ckpt="ckpt|anc-tier-long|0", rng=20265007)]
    cases.append(("budget_2400_does_not_fit", _manifest(tier_long), {},
                  {"BUDGET_EXCEEDS_FRAMES"}, 1))

    # 49. only the two pre-registered tiers name an endpoint: a third value is
    #     refused as a malformed set even though it is a positive integer
    third = [_root("anc-tier-third", takeover=2399, total=4800, budget=1800,
                   ckpt="ckpt|anc-tier-third|0", rng=20265008),
             _root("anc-tier-third", takeover=2399, total=4800, budget=3600,
                   ckpt="ckpt|anc-tier-third|0", rng=20265009)]
    cases.append(("budget_third_tier", _manifest(third), {},
                  {"ANCESTOR_BUDGET_SET", "BUDGET_EXCEEDS_FRAMES"}, 1))

    return cases


def selftest(work: pathlib.Path) -> int:
    work.mkdir(parents=True, exist_ok=True)
    cases = _selftest_cases()
    passed = 0
    failures: list[str] = []

    def check(name: str, document: dict, exit_code: int,
              expected_codes: set, expected_exit: int) -> None:
        nonlocal passed
        got = set(document.get("codes", []))
        problems = []
        if exit_code != expected_exit:
            problems.append(f"exit {exit_code} != {expected_exit}")
        if got != expected_codes:
            problems.append(f"codes {sorted(got)} != {sorted(expected_codes)}")
        if expected_exit == 0 and not document.get("ok"):
            problems.append("document says not ok on a passing case")
        if expected_exit != 0 and document.get("ok"):
            problems.append("document says ok on a failing case")
        if problems:
            failures.append(f"{name}: " + "; ".join(problems))
        else:
            passed += 1
            print(f"  ok  {name:20s} exit={exit_code} codes={sorted(got)}")

    # ---- 8 synthetic manifests on disk, through the same path `check` uses --
    for name, manifest, options, expected_codes, expected_exit in cases:
        path = work / f"{name.replace('@', '-')}.json"
        path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        document, exit_code = evaluate(
            path,
            budget=options.get("budget", DEFAULT_BUDGET),
            allow_exposed=options.get("allow_exposed", False))
        check(name, document, exit_code, expected_codes, expected_exit)

    # ---- structure case 1: the path does not exist at all -------------------
    missing_path = work / "no-such-manifest.json"
    document, exit_code = evaluate(missing_path)
    check("unreadable", document, exit_code, {"MANIFEST_UNREADABLE"}, 2)

    # ---- structure case 2: an empty root list must not pass vacuously -------
    empty = {"schema": SCHEMA, "roots": []}
    document = audit(empty)
    check("empty_roots", document, 0 if document["ok"] else 1, {"FIELD_INVALID"}, 1)

    # ---- structure case 3: a repeated JSON key must not be resolved last-wins
    #      (the injected first value, 999999, could not fit 36,000 frames while
    #      the second one, 8663, fits: last-wins would wash the refusal away)
    seeded = _root("anc-dup-keys")
    seeded["takeover_frame"] = "DUP_KEY_HOLE"
    text = json.dumps(_manifest([seeded]), indent=1, ensure_ascii=False) + "\n"
    needle = '"takeover_frame": "DUP_KEY_HOLE"'
    if needle not in text:
        failures.append("duplicate_keys: could not build the fixture")
    else:
        dup_path = work / "duplicate-keys.json"
        dup_path.write_text(
            text.replace(needle,
                         '"takeover_frame": 999999, "takeover_frame": 8663', 1),
            encoding="utf-8")
        document, exit_code = evaluate(dup_path)
        check("duplicate_keys", document, exit_code, {"MANIFEST_UNREADABLE"}, 2)

    # ---- structure case 4: the frozen admission bar --------------------------
    if DEFAULT_BUDGET == 2400:
        passed += 1
        print("  ok  default_budget     DEFAULT_BUDGET == 2400 (the admission bar)")
    else:
        failures.append(f"default_budget: DEFAULT_BUDGET is {DEFAULT_BUDGET!r}, not 2400")

    # ---- structure case 5: on a duplicated cell the report names that cell first.
    #      ANCESTOR_BUDGET_SET / DUPLICATE_ANCESTOR / DUPLICATE_WINDOW / RNG_REUSED
    #      have to be printed in CODE_ORDER (so "which code reports first" is not
    #      left to dict order), and the duplicate-cell finding must already say which
    #      repeat number and which tier it is.
    duplicated = audit(next(case[1] for case in cases
                            if case[0] == "pair_rng_same_cell_same_seed"))
    ordered_codes = [entry["code"] for entry in duplicated["findings"]]
    expected_order = ["ANCESTOR_BUDGET_SET", "DUPLICATE_ANCESTOR",
                      "DUPLICATE_WINDOW", "RNG_REUSED"]
    duplicate_detail = next((entry["detail"] for entry in duplicated["findings"]
                             if entry["code"] == "DUPLICATE_ANCESTOR"), "")
    if ordered_codes != expected_order:
        order_problem = f"ordered codes {ordered_codes} != {expected_order}"
    elif "pair_index 0" not in duplicate_detail or "budget_frames 2400" not in duplicate_detail:
        order_problem = (f"the DUPLICATE_ANCESTOR finding does not name the repeat "
                         f"number and the tier: {duplicate_detail!r}")
    else:
        order_problem = None
    if order_problem:
        failures.append(f"duplicate_cell_order: {order_problem}")
    else:
        passed += 1
        print("  ok  duplicate_cell_order DUPLICATE_ANCESTOR names pair_index 0 + "
              "budget_frames 2400")

    # ---- the detail of the thin root must name every missing field ---------
    thin_report = audit(next(case[1] for case in cases if case[0] == "fields"))
    details = [entry["detail"] for entry in thin_report["findings"]
               if entry["code"] == "FIELD_MISSING"]
    expected_missing = ["stage", "pair_rng", "takeover_frame", "total_frames",
                        "parent_prefix_digest", "verify_anchor"]
    unnamed = [field for field in expected_missing
               if not any(f"missing required field {field!r}" in detail
                          for detail in details)]
    if unnamed:
        named_problem = f"FIELD_MISSING did not name {unnamed}"
    else:
        named_problem = None
    if named_problem:
        failures.append(f"missing_names: {named_problem}")
    else:
        passed += 1
        print("  ok  missing_names    FIELD_MISSING names all 6 absent fields")

    total = passed + len(failures)   # no hard-coded count: print what was run
    if failures:
        for problem in failures:
            print(f"  FAIL {problem}", file=sys.stderr)
        print(f"SELFTEST: failed {passed}/{total}", file=sys.stderr)
        return 1
    print(f"SELFTEST: ok {passed}/{total}")
    return 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v829_guard.py",
        description="fail-closed guard for a v8.29 pair manifest (schema "
                    f"{SCHEMA}); exit 0 only when every check passes")
    sub = parser.add_subparsers(dest="command", required=True)

    check_cmd = sub.add_parser(
        "check", help="validate one manifest and print a JSON summary")
    check_cmd.add_argument("manifest", type=pathlib.Path,
                           help="path to the pair manifest")
    check_cmd.add_argument("--allow-exposed", action="store_true",
                           help="accept a root that is already used for training "
                                "or selection (default: refuse)")
    check_cmd.add_argument("--budget", type=int, choices=BUDGET_TIERS,
                           default=DEFAULT_BUDGET,
                           help=f"budget tier to prove on top of each root's declared "
                                f"tier (default {DEFAULT_BUDGET}, the section 2 "
                                f"admission bar)")
    check_cmd.add_argument("--json", action="store_true",
                           help="print only the JSON summary, no human report")

    selftest_cmd = sub.add_parser(
        "selftest", help="run the built-in synthetic-manifest battery")
    selftest_cmd.add_argument("--work", type=pathlib.Path,
                              default=pathlib.Path(".tmp_v829_selftest"),
                              help="scratch directory for the synthetic manifests")

    args = parser.parse_args(argv)

    if args.command == "selftest":
        return selftest(args.work)
    return run_check(args.manifest, budget=args.budget,
                     allow_exposed=args.allow_exposed, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
