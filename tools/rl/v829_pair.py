#!/usr/bin/env python3
"""v8.29 paired G1/T2 evaluation entry point -- old windows only, v1.

Why this file exists
====================
`PREREGISTRATION_PAIR_001.md` sec. 1 asks for one thing the v8 runners cannot do: the
**same parked parent**, the **same start frame**, **two rules (G1 / T2)**, and a
**fresh random stream per arm** ("one fresh same-state generator per arm,
never consumed across episodes").
`v8r0_urgency.py` and `v8o0_ruletight.py` each own a four-window x twenty-episode
driver, one rule, one process-wide `torch.Generator` (`v8r0:1290`), and a dataset
projection that silently drops `bomb_frames` / `bomb_available_at_emit`
(`v8r0:841-900`, `v8o0:694-741`).  None of those files may be edited this round,
so this file is the missing orchestration layer and *only* that layer:

* the decision loop is `v830_bombpol.run_episode36` (never copied);
* the candidate sampler is `v8r0_urgency.candidate_sampled` (byte-identical in
  `v8o0:558`, MAP sec. 2);
* the bomb rule is `v8r0_urgency.rule_decision` (it differs from `v8o0:588` by
  one no-op line, MAP sec. 2);
* the **only** per-arm difference is the threshold table:
  `v8r0_urgency.thresholds_for(...)` for G1, `v8o0_ruletight.thresholds("T2")`
  for T2;
* the parked parent is `v450_cf_label.start_parent` (MAP sec. 8 "preferred"), i.e. HELLO
  -> 8663-frame prefix walk -> 2165 hash checks -> `PARK` at 8663 -> `DEBUG`
  `threads=1`; the private data dir is `v450_cf_label.prepare_data_dir`, never
  hand-built (MAP sec. 9-8: a hand-built dir loses the muted `th08.cfg`);
* the manifest is validated by `tools/rl/v829_guard.py` and this file does not
  re-implement a single guard rule (MAP sec. 7.0).

Modes
=====
``check``    **No engine, no engine-side import, no subprocess.**  Runs
             `v829_guard.evaluate`, resolves every root's window / takeover /
             budget / engine digest, re-runs every preflight that does not need
             the engine, and prints the exact argv that the other modes would
             execute.  It never spawns anything; its only write is
             ``<work>/runs/<tag>/run.json`` (the invocation record: manifest
             byte closure, lineage verdict, ``engine_started: false``) -- a
             re-run of the same cell overwrites **only** that record, never a
             real run's ``runs/<tag>`` -- plus ``--out`` when given.
``trace``    One arm (``--arm G1|T2``) on one parked parent.
``pair``     ``--arm-order G1,T2`` (or ``T2,G1``): both arms from **one** PARK,
             so decision 0 of both arms starts at the same engine frame and the
             same inventory.  That is the definition of a pair here.
``compare``  Two trace documents, G1 vs T2.  Pure stdlib; starts nothing.

Selectors vs registered values (sec. 2's two tiers in ONE manifest)
==================================================================
Two options choose **which** roots of the manifest this invocation runs.  They
are *selectors over the manifest subset*, applied in this order:

1. ``--window N``         keeps only the roots that declare window ``N``;
2. ``--budget-frames F``  keeps only the roots that declare budget ``F``
                          (``F`` in ``BUDGET_TIERS``; the default is unchanged,
                          2400 = sec. 2's admission bar).

After both, what is left is this invocation's root set.  So one manifest may
register both tiers of sec. 2's pilot shape ("8 ancestors x 3 ckpt x 2 RNG x
2 budgets" = 96 roots) and the two tiers are run as **two invocations** against
that one file.  An **empty set is refused** -- ``BUDGET_TIER_ABSENT`` (exit 1),
never a silent run of zero cells: asking for a tier the manifest does not
register is a usage error, not a verdict on a cell.  When every root already
declares the requested tier the selection keeps all of them and nothing else
changes: no new plan key, no new stdout line.  That is exactly what the
byte-identical regression on the single-tier manifests compares.

Every other pinned option is a **registered value, not a selector**: it must
agree with what the root declares, and a disagreement is refused, because the
run table is the manifest --

* ``--pair-rng``       vs ``roots[i].pair_rng``        (``PAIR_RNG_CONFLICT``; a
                        different seed is a different registered pair);
* ``--takeover-frame`` vs ``roots[i].takeover_frame``  (``TAKEOVER_ROOT_MISMATCH``);
* ``--prefix-tape``    vs ``roots[i].prefix_tape``     (``PREFIX_TAPE_NOT_IN_MANIFEST``
                        / ``PREFIX_TAPE_CLI_MISMATCH``);
* ``--ckpt-sha256``    vs ``roots[i].ckpt_sha256``     (``CKPT_SHA_MISMATCH``).

A selector that matches nothing has a refusal code of its own
(``WINDOW_NOT_IN_MANIFEST`` / ``BUDGET_TIER_ABSENT``); a registered value that
disagrees is a refusal of that *cell*.  Neither edits the manifest and neither
relaxes a ``v829_guard`` rule: the guard still judges the whole file at the
requested tier **before** any root is selected, so whether a manifest may be run
at a tier at all stays the guard's decision (e.g. ``BUDGET_EXCEEDS_FRAMES`` is
still raised by the guard, not by this selector).

Where the selection is recorded
-------------------------------
``budget_selection {requested, matched, skipped_by_budget, matched_roots}`` is
written to the plan document, to ``run.json`` and to the stdout line
``budget <F>: selected N root(s), skipped M (other tiers)`` -- but only when at
least one root was skipped (a mixed-tier manifest).  That is the same
present-only-when-needed discipline ``takeover_override`` / ``prefix_tape`` /
``lineage_new_roots`` already follow, and it is what keeps a single-tier
invocation byte-identical.

Recovery: windows and the takeover frame
=======================================
The four old windows ``(11, 12, 17, 18)`` are the only ones v1 recovers, and a
window outside that set is still ``WINDOW_NOT_SUPPORTED`` when the mode needs the
engine.

The takeover frame is now a **declared, validated parameter** (``--takeover-frame
N``, default: the root's own ``roots[i].takeover_frame``, which is 8663 for every
pre-v8.30 root).  Three pinned modules hard-code 8663 for it:

* `v432_fork_arm.walk_prefix` -- ``PREFIX_FRAMES = 8663`` / ``CHECKPOINTS_IN_PREFIX
  = 2165`` (`v432:189-191`, read at `:618`, `:625`, `:644`, `:653`);
* `v450_cf_label.masks_for` -- ``TARGET_FRAME = 8664`` (`v450:128`, `:425-427`),
  plus its own ``PREFIX_FRAMES`` / ``CHECKPOINTS_IN_PREFIX`` at `:460`, `:466`,
  `:468`, `:725`;
* `v830_bombpol.run_episode36` -- asserts the spawned child starts at
  ``PREFIX_FRAMES`` (`v830:789-791`), and ``first_death_frame =
  PREFIX_FRAMES + died_at`` (`:886`).

Those files are **pinned and are never edited** (MAP sec. 9-1).  What changes is
the *module object*: every ``tools/rl`` script here is loaded by path with
``importlib`` and is deliberately **not** registered in ``sys.modules``
(`load_module`, MAP sec. 9-12), so each load is an independent copy living only in
this process.  ``apply_prefix_override()`` re-points the globals of **this
process's copies** -- and of the copies those copies load for themselves
(``v450._M432``, ``v830._M450``, ``v830._M432``) -- then reads them back.  No
byte on disk changes and no other process or round is affected; the three files'
``sha256`` are compared before and after the patch and recorded in
``prefix_constants.files_on_disk_unmodified``.  For the frozen cell (``N ==
8663``) nothing is imported and nothing is assigned: the old read-only
``assert_prefix_constants()`` path runs unchanged.

Admissible takeover frames are ``N >= 4``, ``N % 4 == 3`` (one recorded anchor
every 4 frames, `v431_dense_driver.py:81-82`: the hash keys are the multiples of
4, and ``build_masks`` *refuses* a target with no hash, `:88-89`, so the
``target = N + 1`` used by `masks_for` must be a multiple of 4) and ``N + 1 <=
frames_built`` of this window's S1c recording.  Each refusal has its own code:
``TAKEOVER_FRAME_NOT_INTEGER`` / ``TAKEOVER_FRAME_TOO_SMALL`` /
``TAKEOVER_FRAME_NOT_ANCHOR`` / ``TAKEOVER_BEYOND_RECORDING`` /
``TAKEOVER_ROOT_MISMATCH``.  Expected checkpoints are ``N // 4`` (8663 -> 2165,
3999 -> 999).

RNG derivation (pre-registration sec. 1)
====================================
::

    pair_rng                 int, unique per (ancestor, budget, checkpoint,
                             replicate); the manifest root declares it
                             (``roots[i].pair_rng``) and ``v829_guard`` refuses
                             a reuse (``RNG_REUSED``)
    generator_seed           = pair_rng * 1000 + 7
                             = ``v8r0_urgency.generator_seed`` / ``v8o0:343`` /
                               ``v830:453`` (all three are ``seed*1000 + 7``,
                               GEN_CONST = 7)
    per arm, per episode     torch.Generator(); .manual_seed(generator_seed)
                             -> both arms start from the *same* state, and no
                                episode ever inherits the stream of another

The generator is created **after** `v860_bombdag.load_policy36`, because that
loader calls `v830.policy36` which runs `torch.manual_seed(seed)` (`v830:722`,
MAP sec. 9-2).  Ordering it the other way would still work by accident; doing it in
the documented order means the recorded digest is the digest that ran.

Every episode row records ``torch.__version__``, ``device``, the initial RNG
digest ``sha256(generator.get_state().numpy().tobytes())[:16]`` (with an explicit
fallback if ``get_state`` is unavailable), the final digest, and the number of
``torch.multinomial`` draws consumed.  ``--episodes N`` replays the *same*
``pair_rng`` N times (the row says so in ``rng_stream_note``): a new replicate
needs a new ``pair_rng``, which needs a new manifest root, which is what sec. 1's
"written into the pair table" means.

Bomb accounting (MAP sec. 4, corrected)
===================================
`run_episode36` **does** export ``bomb_frames`` (`v830:832`) and
``bomb_available_at_emit`` (`v830:873`); only the v8 runners' projection loses
them.  ``emit_decrement_at`` is the one quantity the record never had
(`v830:846`, `:862-863`), so it is taken through a **parent proxy**
(`TracedChild`), exactly like the precedent in `v800_bomb.py:703-738`:

* one entry per emit, ``frame_index + 1`` within that decision (so 1 or 2 with
  ``decision_frames == 2``), ``None`` when the inventory never dropped;
* the list is length-aligned with ``bomb_decisions`` / ``bomb_frames`` /
  ``bombs_remaining_at_emit`` / ``bomb_available_at_emit``; a misalignment is
  reported as ``trace_alignment_ok = false`` instead of being hidden;
* the row carries both ``emit_decrement_at`` (the first non-``None`` value, for
  compact aggregation) and ``emit_decrement_at_list`` (the v800-shaped list).

The proxy also records the masks that were **actually sent** (the ``STEP <mask>
1`` commands), so the trace compares wire bytes, not a recomputation.

``arm_label`` is deliberately **not** written (MAP sec. 9-4): `v8r0.arm_label` reads
``getattr(parent, "arm", "G2")`` and no code ever sets ``parent.arm``, so every
row would claim ``URG-a1.00``.  Rows carry ``rule_label`` (derived from the real
producer: ``alpha_of("G1")`` / ``v8o0.arm_spec("T2")["label"]``) plus
``threshold_table`` / ``threshold_table_last``.

Compare mode: criteria and output format
=======================================
Inputs: two trace documents written by ``mode=trace``/``mode=pair`` with
``--keep-trace`` (``<out-stem>.<ARM>.trace.json``).

Per decision index ``i`` (0-based, the same numbering as
``run_episode36``'s ``trace[].decision``) exactly these seven values are
compared, by strict equality (``None == None`` counts as equal, floats are
compared bit-exactly because both arms run the same formula from the same
source):

===========================  ==========================  ====================
trace key                    report name                 type
===========================  ==========================  ====================
``threshold_px``             ``threshold``               float | null
``nearest_px``               ``nearest_px``              float | null
``cand``                     ``cand``                    int 0..17
``masks``                    ``mask``                    list[int] (wire order)
``bomb_request``             ``bomb_request``            0 | 1
``inventory_before``         ``inventory_before``        int
``inventory_after``          ``inventory_after``         int
===========================  ==========================  ====================

* ``same_prefix_decisions`` = the number of leading decisions for which all
  seven are equal in both traces (``0`` when decision 0 already differs).
* ``first_diff_decision`` = that prefix length, i.e. the 0-based index of the
  first differing decision; ``first_diff_decision_1based`` is the human
  numbering.  ``first_diff_frame`` = ``decisions[i].frame``, the **absolute**
  engine frame of that decision (8663 at decision 0 for the old windows).
* a differing decision count is itself a difference, reported at index
  ``min(len(g1), len(t2))`` with ``field == "decision_count"``.
* ``verdict`` is ``IDENTICAL`` (exit 0) or ``DIFFER`` (exit 1); unreadable or
  structurally incompatible traces exit 2.
* ``first_action_difference`` is a **secondary** answer over the behaviour
  subset only (``cand``, ``mask``, ``bomb_request``, ``inventory_before``,
  ``inventory_after``).  G1 and T2 have different threshold tables *by
  construction*, so ``first_diff_decision`` is normally ``0`` on a real pair;
  the behaviour difference is the number that says when the two arms actually
  acted differently.  It never changes the verdict.
* headers must agree on ``prefix_frame``, ``decision_frames``,
  ``budget_decisions`` and ``window``; if they do not, ``TRACE_INCOMPATIBLE``
  (exit 2) -- comparing two different start frames would be meaningless.

Printed form (stdout), followed by ``--- json summary ---`` and the document::

    v829 pair compare: G1 <path>  vs  T2 <path>
      arms        G1=URG-a0.50  T2=40/20/5
      start       prefix_frame=8663 decision_frames=2 budget_decisions=1200
      decisions   G1=1200  T2=1200
      compared    threshold, nearest_px, cand, mask, bomb_request,
                  inventory_before, inventory_after
      same prefix length: 37
      first difference:   decision 37 (0-based)  absolute frame 8737
        cand             G1=4  T2=7
      first behaviour difference: decision 37 (0-based)  absolute frame 8737
        fields=['cand']
    VERDICT: DIFFER

Manifest
========
``schema = "v829-pair-manifest/1"`` exactly as `tools/rl/v829_guard.py` freezes
it.  This file adds **only** optional, additive root keys that the guard
ignores: ``window`` (which S1c recording and which ``w<NN>`` data dir),
``ckpt_path`` (so a multi-root manifest can bind one checkpoint per root),
``lineage_ancestor_ids`` (see B below).
A root without ``window`` is still runnable if ``--window`` is given, and the
row records ``window_source = "cli"`` so the provenance is visible.

A. Manifest byte closure (one read, digest asserted)
===================================================
`v829_guard.evaluate` reads the manifest, digests it and judges **those** bytes.
The runner then needs the same bytes, not "whatever is at that path now": a
second ``json.loads(path.read_text())`` is a TOCTOU window in which an edit
lands between the checked bytes and the used bytes (r1 REPORT sec. M4).

``manifest_from_single_read()`` therefore does exactly one ``read_bytes()``,
asserts ``sha256(raw) == guard_doc["manifest_sha256"]`` *and*
``len(raw) == guard_doc["manifest_bytes"]``, and parses **that same buffer**.
Any disagreement -> ``MANIFEST_CHANGED`` (exit 1) before a single field is read.
``plan.json``/``run.json`` record ``manifest_sha256`` (the digest of the bytes
actually used), ``manifest_reads: 1``, ``manifest_sha256_source`` (``guard``,
or ``self`` when the guard document carries no digest -- then the one buffer is
digested and parsed in the same step and the weaker provenance is recorded as a
residual risk) and ``manifest_sha256_guard`` for the comparison itself.

B. Lineage cross-check (``--lineage-config``, off by default)
============================================================
``used_for_training`` / ``used_for_selection`` are declared **by the manifest
itself**, so on their own they are self-certification, not evidence.  With
``--lineage-config configs/v430/splits.json`` every selected root must declare
``lineage_ancestor_ids`` (array of strings) and every id is looked up in

* ``splits.{train,dev,confirm,adapt,quarantine}.ancestor_ids`` of that file, and
* the ``pretraining_sources.json`` registry named by that file's
  ``registries[]`` (role ``pretraining_sources``): ``sources[].status``.

Verdicts, per root:

* ``lineage_ancestor_ids`` missing / not a non-empty string array ->
  ``LINEAGE_MISSING`` (refused: the switch asked for a check that cannot run);
* an id in ``train`` or ``dev`` -- or in a source whose status is
  ``used_for_training`` / ``used_for_selection`` -- while the root declares
  ``used_for_training: false`` **and** ``used_for_selection: false`` ->
  ``EXPOSURE_CONTRADICTS_LINEAGE`` (refused: the declaration understates a
  documented consumption);
* declared exposed but the id appears in no split and no source ->
  ``LINEAGE_UNKNOWN``: recorded in ``lineage_unmatched`` and ``lineage_codes``
  but **not** refused, because an old window can predate the frozen genealogy
  and refusing there would manufacture a contradiction out of missing evidence.

The verdict is written to ``run.json`` (``lineage_checked``,
``lineage_config_sha256``, ``lineage_sources_sha256``, ``lineage_unmatched``
and, per root, ``lineage_split`` / ``lineage_status`` /
``lineage_unmatched``).  Without the switch ``run.json`` states
``lineage_checked: false`` plus a ``residual_risk`` entry that exposure is
self-declared and was not cross-checked.

New ancestors (``roots[i].lineage_new: true``)
----------------------------------------------
A research-side **new** ancestor is by definition not in ``splits.json``, so it
cannot declare a non-empty ``lineage_ancestor_ids`` and the switch above would be
unusable for it.  Such a root may instead declare the JSON boolean
``lineage_new: true``, and then:

* empty / absent ``lineage_ancestor_ids`` is *expected*, not a gap: the verdict is
  the recorded code ``LINEAGE_NEW`` (``verdict: RECORDED``, **exit 0**, never
  refused), the root is named in ``run.json.lineage_new_roots`` and the
  ``residual_risk`` list gains exactly
  ``"new ancestor: exposure is declared by the run table, not provable against
  the frozen lineage"``;
* ``lineage_new: true`` **together with** an id that is in ``train``/``dev`` (or a
  source consumed for training/selection) is self-contradictory and is refused
  ``EXPOSURE_CONTRADICTS_LINEAGE``;
* a root with neither ``lineage_new`` nor ``lineage_ancestor_ids`` is still
  ``LINEAGE_MISSING`` -- the old behaviour, unchanged.

``lineage_new`` is judged only as a boolean (a string ``"true"`` is
``MANIFEST_INVALID``): the flag decides between "refused" and "recorded", so it
cannot be inferred from a truthy-looking string.

Technical-fault re-run (pre-registration sec. 4)
===============================================
"One technical fault re-runs both arms once with the same pair, start frame,
random stream and configuration, and both records are kept."  This file reads
sec. 4's list literally -- "无法恢复、帧丢失、引擎报错或终态不明记技术无效"
(unrecoverable, lost frame, engine error, unknown terminal state) -- so an
attempt is a technical fault when **either** of these holds:

* **recovery (PARK) fault**: `v450_cf_label.start_parent` raises at any point of
  HELLO -> ``takeover``-frame prefix -> ``takeover // 4`` hash checks -> PARK ->
  DEBUG ``threads=1`` (8663 / 2165 for the frozen cell).
  No parent exists, so no arm can start, and sec. 4's "无法恢复" is exactly
  this.  `park_parent()` turns the engine-side exception into `RecoveryFault`;
  the attempt record carries ``phase: "recovery"``, the attempt number, the
  exception and a ``kind`` (``engine_crash`` / ``prefix_hash_mismatch`` /
  ``recovery_error``), e.g. ``"attempt 1 engine_crash: HarnessError: harness
  closed the control pipe"``;
* **row fault**: after a PARK, **any** arm's row has ``invalid: true`` or a
  non-empty ``error``: `technical_invalid()` decides one row and
  `should_retry()` decides the attempt.  A legal result -- full segment
  (``survived: true``), death, ``stage_clear`` -- is *never* a fault and never
  re-runs (the decision looks at ``invalid``/``error`` only, never at
  ``terminal``/``frames``/``survived``).

Never a technical fault, therefore never re-run: a legal death, a full segment,
a ``stage_clear``, and every **guard/preflight refusal** (`PairError` raised
before the first PARK: engine / data-source / checkpoint / tag / pair-rng /
audio / lineage / prefix-recording / manifest).  Those are input problems, not
technical faults, and they do not consume a second attempt.

Prefix hash mismatch vs engine crash: a mismatch *may* be a data/tape problem,
but sec. 4 makes *unrecoverability* the fault, and one bounded retry is the
cheapest way to tell a flaky engine start from a real divergence.  Both are
therefore retried once, and the record keeps the distinction in ``kind``: a
second identical ``prefix_hash_mismatch`` is visible in ``run.json`` as a data
problem instead of being laundered into "engine flake".

`run_pair_with_retry(runner, ...)` owns the decision *and* the persistence, with
the attempt body injected as ``runner(attempt)``:

* at most ``RETRY_ATTEMPT_LIMIT = 2`` attempts (1 original + 1 re-run) -- a third
  is impossible, and a caller asking for more is clamped.  A recovery fault
  consumes one of those two attempts exactly like a row fault does;
* attempt 1's rows are **kept**: they stay in the same jsonl, marked
  ``attempt: 1``, and attempt 2 is appended with ``attempt: 2`` and
  ``supersedes: <sha256 of the attempt-1 row it replaces>`` (a recovery fault
  produced no row, so there is nothing to supersede; its record lives in
  ``run.json.attempts[0]`` and the fresh PARK in ``gate-w<w>-a2.json``);
* if **both** attempts fail in recovery, the process still exits
  ``RECOVERY_FAILED`` (1), ``run.json`` is written with
  ``outcome: "recovery_failed"`` and **two** ``phase: "recovery"`` records whose
  ``error`` strings name attempt 1 and attempt 2, and no third attempt is made;
* ``--mode trace`` runs one arm, so a *row* fault is not retried there (there is
  no pair to re-run); a *recovery* fault is retried once, because no row exists
  yet and the re-run repeats the same PARK / single arm / configuration;
* ``run.json`` gets ``attempts: [{attempt, phase, error, kind, tag, rows,
  invalid_rows, wall_s, attempt_fingerprint, supersedes, root_index, window}]``,
  plus ``rows_final`` / ``superseded_rows``; ``attempts[i].tag`` is the same cell
  tag by design (the re-run is the *same* cell, which is why the row-level
  fingerprint, not the tag, is the supersede reference);
* the final attempt is authoritative: ``all_rows_valid``, ``invalid_rows`` and
  ``full_segments`` in ``run.json`` are computed from it only;
* ``--keep-trace`` writes attempt 1's trace under its old name and attempt 2's as
  ``<stem>.<ARM>[.ep<N>].a2.trace.json``, so neither overwrites the other.

Concurrency (``--workers N``, pre-registration sec. 3.4)
=======================================================
``--workers N`` (default 1, max 4) spreads the selected **pair cells** over up to
N worker *processes*: one cell = one root x its budget x its checkpoint (the
manifest pins one budget and one checkpoint per root, so a cell is a root, and
two roots that differ only by window or ckpt are still distinct cells), one
worker = one cell, ``ceil(cells / N)`` batches run in order.

* every worker owns a **unique tag** ``<tag>-w<i>`` and therefore its own
  ``runs/<tag>-w<i>/`` and ``data/<tag>-w<i>/``; `assert_unique_worker_tags()`
  refuses a duplicate tag / runs dir / data dir as ``TAG_NOT_UNIQUE`` **before**
  the first spawn (MAP sec. 9-3 is exactly the accident this prevents);
* **merge choice**: every worker writes its own jsonl
  ``<work>/eval/<tag>-w<i>.jsonl`` and the coordinator concatenates them into the
  single ``--out`` in cell order; ``run.json.merge_sources`` and
  ``runs/<tag>/workers.json`` list every worker jsonl with its row count and
  digest, so the merge can be re-derived.  Nothing is deleted or moved after the
  merge;
* interruption safety: the workers of the current batch are allowed to finish,
  batches that had not started are recorded as ``skipped`` instead of being
  launched, the aggregate exit code is non-zero (2 outranks 1), and every
  finished worker's artifacts stay on disk;
* the workers are spawned with ``sys.executable`` (this very interpreter, i.e.
  Python 3.10.12 under ``/root/v400-venv/bin/python``), never with a bare
  ``python`` from ``PATH``, and ``--workers`` is not propagated to them;
* ``--workers 1`` is the old single-process path, byte for byte: `mode_run()`
  runs every root in this process exactly as before.

Prefix digest (``parent_prefix_digest``)
=======================================
The four old windows were measured on **four different** S1c recordings, so no
single digest describes them all: ``PREFIX_RECORD_SHA256`` is the *seed12*
recording (MAP sec. 1, the D3 gate) and is not a per-window value.
``parent_prefix_digest`` is the root's own window recording digest and it is
**compared**, not merely recorded: `prefix_digest_gate()` hashes
``s1c_recording_path(window)`` and refuses ``PREFIX_RECORDING_MISMATCH`` when it
differs from ``parent_prefix_digest`` (or from
``verify_anchor.window_recording_sha256`` when that key is present).  When the
recording is not visible from this host the comparison is deferred to the run
host and recorded as ``gate: "deferred-to-run"``, like the engine/checkpoint
digests.

Exit codes
==========
``0``  ok: guard passed, preflight coherent (``check``), or every row valid and
       the post-run audio audit muted, or the two traces are identical
``1``  refused: a fail-closed finding (guard refusal, window/takeover/engine/
       checkpoint/tag/pair-rng/audio problem, an invalid episode row, a PARK that
       failed twice = ``RECOVERY_FAILED``, or a trace difference in ``compare``)
``2``  cannot evaluate: unreadable manifest/trace/lineage config, missing input
       file, argparse usage error

Codes this file owns (none of them comes from the guard): ``MANIFEST_CHANGED``
(A), ``PREFIX_RECORDING_MISMATCH`` (C), ``LINEAGE_MISSING`` /
``EXPOSURE_CONTRADICTS_LINEAGE`` / ``LINEAGE_CONFIG_UNREADABLE`` (B, exit 1 or
2), ``LINEAGE_NEW`` (B, recorded only -- exit 0) and ``LINEAGE_UNKNOWN`` (B,
recorded only -- the invocation still exits 0 unless something else refuses).
``WORKERS_OUT_OF_RANGE`` / ``WORKERS_MODE_UNSUPPORTED`` are usage refusals
(exit 2) of the ``--workers`` knob.

Audio (AGENTS.md sec. 9)
====================
Three independent layers, all of them recorded in ``plan.json``/``run.json``:
(1) `v450_cf_label.prepare_env()` sets ``SDL_AUDIODRIVER=dummy`` and drops
``DISPLAY``/``WAYLAND_DISPLAY``; (2) `HarnessClient.__init__` sets
``SDL_AUDIODRIVER=dummy`` again and pops ``DISPLAY`` for ``role="shadow"``
(`protocol.py:497-510`); (3) `v432.build_private_data_dir` regenerates the
60-byte ``th08.cfg`` on every start and re-inspects four fields for zero
(`v432:416-446`), which `prepare_data_dir` surfaces as
``info["cfg"]["muted_four_zero"]``.  After the engine exits, this file decodes
every ``th08.cfg`` under ``<work>/data/<tag>`` with
`v431_audio_cfg_audit.decode()` and refuses (exit 1) unless
``cfg_files == muted``, ``audible == 0`` and the four fields (musicMode,
playSounds, musicVolume, sfxVolume) are all zero in every file.

Usage
=====
::

    python -B tools/rl/v829_pair.py --mode check --manifest M.json --window 11 \
        --budget-frames 2400 --allow-exposed
    python -B tools/rl/v829_pair.py --mode check --manifest M.json --window 11 \
        --budget-frames 2400 --allow-exposed \
        --lineage-config configs/v430/splits.json
    python -B tools/rl/v829_pair.py --mode pair --manifest M.json --window 11 \
        --budget-frames 2400 --ckpt C.pt --ckpt-sha256 <64hex> --pair-rng 20261124 \
        --tag v829-smoke-w11 --allow-exposed --allow-legacy-window --keep-trace
    python -B tools/rl/v829_pair.py --mode compare \
        --trace-g1 eval/x.G1.trace.json --trace-t2 eval/x.T2.trace.json

Nothing here writes outside ``--work-dir`` (default ``.tmp_v829_pair``), the
explicit ``--out``/``--log``, or a path the ordered ``WRITE_FENCE`` rule table
allows.  The fence allows the whole ``.tmp_v829_*`` namespace this round owns
(``.tmp_v829_pair``, ``.tmp_v829_info``, ``.tmp_v829_collect``, and any new
``.tmp_v829_<x>``); every other existing ``.tmp_*`` directory (``.tmp_v400_*``,
``.tmp_v410_*``, ``.tmp_v420_*``, ``.tmp_v430_*``, ``.tmp_v431_*``, ``.tmp_v44*``,
``.tmp_v8[0-9a-z]*``, catch-all) and ``Document/`` / ``data/`` / the source trees
are refused as historical evidence (``HISTORICAL_PATH``), and the refusal names
the rule that fired.  ``--explain-path PATH`` prints the same judgement offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:                                     # noqa: SIM103
    sys.path.insert(0, str(REPO))

TOOL = "tools/rl/v829_pair.py"
TOOL_VERSION = "v8.29.0"

# -- schemas this file produces ---------------------------------------------
SCHEMA_EPISODE = "v829-pair-episode/1"
SCHEMA_TRACE = "v829-pair-trace/1"
SCHEMA_COMPARE = "v829-pair-compare/1"
SCHEMA_PLAN = "v829-pair-plan/1"
SCHEMA_RUN = "v829-pair-run/1"
SCHEMA_AUDIO = "v829-pair-audio-audit/1"
SCHEMA_LINEAGE = "v829-pair-lineage/1"
SCHEMA_CLOSURE = "v829-pair-manifest-bytes/1"
#: the ``budget_selection`` block of a mixed-tier invocation (a selector, see the
#: module docstring "Selectors vs registered values"); a single-tier manifest
#: never produces it, so its plan/run.json stays byte-identical
SCHEMA_BUDGET_SELECTION = "v829-pair-budget-selection/1"
SCHEMA_GUARD = "v829-pair-manifest/1"          # owned by tools/rl/v829_guard.py

# -- the frozen inputs this file obeys --------------------------------------
GUARD_REL = "tools/rl/v829_guard.py"
WORK_DEFAULT = REPO / ".tmp_v829_pair"
ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"
#: `v432:185`, `v450:123`, `v8r0:188`, `v8o0:170` -- all four agree.
ENGINE_SHA256 = "691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6"
#: The **seed12** recording digest (5,739,622 B), i.e. the value MAP sec. 1 names
#: for the D3 gate.  It is *not* a per-window constant: the four old windows were
#: measured on four different S1c recordings (window 11 is
#: ``bc663759...`` / 5,744,217 B).  The per-window value is the manifest root's
#: ``parent_prefix_digest`` and it is compared against
#: ``sha256(s1c_recording_path(window))`` by `prefix_digest_gate()`
#: (``PREFIX_RECORDING_MISMATCH``), never assumed to be this one.
PREFIX_RECORD_SHA256 = ("f79f8ab6cef1ca4ac5c9489e4eeca6f70b55a97d4de847bca66a923"
                        "b0033d028")

# -- lineage cross-check (B) -------------------------------------------------
#: Frozen genealogy: `configs/v430/splits.json`, read in this precedence order.
LINEAGE_SPLIT_ORDER: tuple[str, ...] = ("train", "dev", "confirm", "adapt",
                                        "quarantine")
#: Splits whose membership means "already consumed": a false/false declaration is
#: then a contradiction, not a preference.
LINEAGE_EXPOSED_SPLITS: tuple[str, ...] = ("train", "dev")
#: ``splits["registries"]`` role that names the pretraining-source registry.
LINEAGE_SOURCES_ROLE = "pretraining_sources"
#: Fallback if the config names no registry (relative to the repo root).
LINEAGE_SOURCES_DEFAULT = ("Document/Update/Alpha/v4.3.0/registries/"
                           "pretraining_sources.json")
#: ``sources[].status`` values that claim training / selection consumption.
LINEAGE_STATUS_EXPOSED: tuple[str, ...] = ("used_for_training",
                                           "used_for_selection")
#: Registry statuses in precedence order (for the single-valued ``lineage_status``).
LINEAGE_STATUS_ORDER: tuple[str, ...] = ("used_for_training",
                                         "used_for_selection", "evaluated_only",
                                         "rejected_not_used",
                                         "rule_based_no_training_data",
                                         "executed_instrument_only")
#: The exact residual risk recorded when ``--lineage-config`` is not given.
RESIDUAL_EXPOSURE = ("exposure is self-declared: roots[].used_for_training / "
                     "used_for_selection come from the manifest itself and were "
                     "NOT cross-checked against configs/v430/splits.json or the "
                     "pretraining_sources registry; pass "
                     "--lineage-config configs/v430/splits.json to check them")
#: The exact residual risk recorded for a root that declares ``lineage_new``.
#: The frozen genealogy cannot confirm or refute such an ancestor by
#: construction (it is not in it), so the run table is the only declaration.
RESIDUAL_LINEAGE_NEW = ("new ancestor: exposure is declared by the run table, "
                        "not provable against the frozen lineage")
#: ``roots[i].lineage_new`` -- the opt-in that lets a research-side NEW ancestor
#: (one the frozen genealogy does not enumerate) use ``--lineage-config`` at all.
LINEAGE_NEW_FIELD = "lineage_new"

#: v1 supports these four windows and nothing else (MAP sec. 9-1).
OLD_WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
#: `v432:189`, `v450:127`, `v8r0:153`, `v8o0:143`, `v830` (via import) -- the
#: PARK frame of the *frozen* cell.  `assert_prefix_constants()` still reads it
#: back from the pinned files and refuses if anyone moved it on disk; a non-8663
#: takeover is applied to this process's module objects only (see
#: `apply_prefix_override`), never to the files and never to this constant.
PREFIX_FRAMES = 8663
DECISION_FRAMES = 2
N_CAND = 18

#: One recorded hash anchor every 4 frames: `v431_dense_driver.py:72` takes the
#: row's `decision_frame` (a multiple of 4) and `:81-82` writes `hashes[f]` and
#: `hashes[f + 4]`, so every key is a multiple of 4.  `:88-89` refuses a target
#: with no hash, and `v450.masks_for` passes `TARGET_FRAME = N + 1` -- therefore
#: an admissible takeover frame is `4k - 1`.
ANCHOR_STRIDE = 4
#: `N % 4` of every admissible takeover frame (8663 % 4 == 3).
TAKEOVER_MOD = ANCHOR_STRIDE - 1
#: Smallest takeover frame the prefix walk can start from.
MIN_TAKEOVER_FRAME = 4
#: Optional per-root **prefix tape** (additive).  A root may name its own 4-frame
#: ``labels.jsonl`` instead of the window's S1c shard that
#: `v431_dense_driver.build_masks` hard-codes
#: (``.tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed<window>-shard/labels.jsonl``,
#: `v431:64-99`).  Without the field every path below is exactly the pre-tape one:
#: the S1c shard, `prefix_digest_gate` and ``PREFIX_RECORDING_MISMATCH`` stay in
#: charge, and `mode=check` output is byte-identical.
PREFIX_TAPE_FIELDS: tuple[str, ...] = ("prefix_tape", "prefix_tape_sha256",
                                       "prefix_tape_rows", "prefix_tape_frames")
#: The three pinned files whose *module globals* a non-8663 takeover re-points,
#: and the globals to set in each.  `v432`/`v450`/`v830` are never edited: only
#: this process's by-path copies are mutated (`load_module`, MAP sec. 9-12).
PREFIX_TARGETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("v432_fork_arm", "tools/rl/v432_fork_arm.py",
     ("PREFIX_FRAMES", "CHECKPOINTS_IN_PREFIX")),
    ("v450_cf_label", "tools/rl/v450_cf_label.py",
     ("PREFIX_FRAMES", "TARGET_FRAME", "CHECKPOINTS_IN_PREFIX")),
    ("v830_bombpol", "tools/rl/v830_bombpol.py",
     ("PREFIX_FRAMES",)),
)

BUDGET_TIERS: tuple[int, ...] = (1800, 2400)
DEFAULT_BUDGET_FRAMES = 2400
ARMS: tuple[str, ...] = ("G1", "T2")
DEFAULT_ARM_ORDER = "G1,T2"
#: `v8r0:157` GEN_CONST; the same 7 in `v8o0:147` and `v830:259`.
GEN_CONST = 7

#: Pre-registration sec. 4: "一次技术故障只按原 pair、起点、随机流和配置重跑双方
#: 一次，保留两次记录."  So a pair is allowed exactly TWO attempts (1 original +
#: 1 re-run) and never a third; a legal result (full segment / death / clear) is
#: never a technical fault and never triggers the re-run.
RETRY_ATTEMPT_LIMIT = 2
#: Pre-registration sec. 3.4 ("each work process has a unique tag") -- the pair
#: cells of one invocation, at most this many processes at a time.
MAX_WORKERS = 4
DEFAULT_WORKERS = 1

#: ``--reuse-park`` (OFF by default).  One PARK is ~1.8 s of wall clock (8,663
#: prefix frames + 2,165 hash checks) and it depends on exactly three things: the
#: prefix the parent replays, the frame it PARKs at and the budget the episode is
#: written against -- **not** on the checkpoint, the ``pair_rng`` or the arm.
#: Cells that agree on those three can therefore share one parent and each spawn
#: its own child from it (`ForkParent.spawn`, `v432_fork_arm.py:580`, which is
#: what ``v830_bombpol.run_episode36`` already uses for every arm).
#:
#: The default is off because the flag changes the row ORDER of the jsonl (the
#: cells are visited group by group, so one parent is alive at a time) and adds
#: the ``park_reuse`` key to every row; with the flag absent not one byte of the
#: old output changes.
PARK_REUSE_SCHEMA = "v829-pair-park-reuse/1"
DEFAULT_REUSE_PARK = False
#: The three fields the work order names as the grouping key.  They are the ones
#: a PARK result actually depends on; `park_reuse_identity` adds every remaining
#: PARK input as a *fail-closed* discriminator (a group is split, never mixed).
PARK_REUSE_KEY_FIELDS = ("prefix", "takeover_frame", "budget_frames")

#: Written trees: everything else under ``.tmp_v*`` and all of ``Document/`` is
#: somebody's historical evidence and must not receive a byte from this tool.
ALLOWED_TMP = (".tmp_v829_pair", ".tmp_v829_info", ".tmp_v829_collect")
FORBIDDEN_TREES = ("Document", ".git", "tools", "touhou_rl", "configs", "tests",
                   "third_party", "third_party_data", "data")

#: The whole ``.tmp_v829_*`` namespace is THIS round's scratch space, not a
#: historical run directory.  s11 measured the failure mode the old fence had:
#: ``--out .tmp_v829_collect/eval/<tag>.jsonl`` (the tape tree
#: ``tools/rl/v829_collect_source.py`` writes) was refused as HISTORICAL_PATH
#: because the fence was a *whitelist* -- "every .tmp_* except these two names is
#: historical" -- so any new directory this round invents was historical by
#: default.  The fence is now an explicit, ordered RULE TABLE: a path is allowed
#: only by a rule that says so, and a refusal names the rule that refused it.
TMP_NAMESPACE = ".tmp_v829_"

#: Ordered write fence for repository-relative first components.  Each entry is
#: ``(rule_id, kind, pattern, verdict, reason)``; ``kind`` is ``exact``,
#: ``prefix`` (first component starts with ``pattern``) or ``glob``
#: (``fnmatch`` over the first component).  The FIRST match wins, so the allow
#: rules sit above the deny rules and the catch-all ``.tmp_`` rule sits last among
#: them.  Adding a new tree to this round means adding an allow rule here; it no
#: longer happens implicitly.
WRITE_FENCE: tuple[tuple[str, str, str, str, str], ...] = (
    ("tmp_v829_namespace", "prefix", TMP_NAMESPACE, "allow",
     "the .tmp_v829_* namespace belongs to this round (runs/, data/, eval/, "
     "tape collection, evidence); listing it here is what makes a NEW "
     ".tmp_v829_<something> writable without editing the fence"),
    ("tmp_v829_pair", "exact", ".tmp_v829_pair", "allow",
     "default --work-dir of this round's pair/trace/check runs"),
    ("tmp_v829_info", "exact", ".tmp_v829_info", "allow",
     "this round's evidence/report tree"),
    ("tmp_v829_collect", "exact", ".tmp_v829_collect", "allow",
     "this round's tape collection tree (tools/rl/v829_collect_source.py)"),
    ("tmp_v400", "prefix", ".tmp_v400_", "deny",
     "historical v4.0 run tree (D0/D1 canonical data: .tmp_v400_d1 is 28.8 GB "
     "of frozen evidence)"),
    ("tmp_v410", "prefix", ".tmp_v410_", "deny",
     "historical v4.1 run tree (S1 66,143 + WP05 DAgger round1/2/3)"),
    ("tmp_v420", "prefix", ".tmp_v420_", "deny",
     "historical v4.2 run tree (expansion / private data dirs)"),
    ("tmp_v430", "prefix", ".tmp_v430_", "deny",
     "historical v4.3 run tree"),
    ("tmp_v431", "prefix", ".tmp_v431_", "deny",
     "historical v4.31 run tree"),
    ("tmp_v44", "prefix", ".tmp_v44", "deny",
     "historical v4.4x run tree"),
    ("tmp_v45", "prefix", ".tmp_v45", "deny",
     "historical v4.5x run tree"),
    ("tmp_v8x", "glob", ".tmp_v8[0-9a-z]*", "deny",
     "historical v8.x run tree (v8.29/s11/s12/... rounds; glob covers .tmp_v829* "
     "as well, which is why the namespace allow rule has to come first)"),
    ("tmp_other", "prefix", ".tmp_", "deny",
     "any other existing .tmp_* tree is somebody's historical evidence: the "
     "default for an unlisted .tmp_* is DENY, never 'probably scratch'"),
    ("tree_document", "exact", "Document", "deny",
     "Document/ is the research-side record (pre-registration, evidence, "
     "reports); this tool writes none of it"),
    ("tree_data", "exact", "data", "deny",
     "data/ is a source tree, not a run directory"),
    ("tree_third_party", "exact", "third_party", "deny",
     "third_party/ holds the engine build (rl_v3 / rl_laser) and is not ours"),
    ("tree_third_party_data", "exact", "third_party_data", "deny",
     "third_party_data/ holds vendor data"),
    ("tree_tools", "exact", "tools", "deny",
     "tools/ is source code; a run never writes next to its own scripts"),
    ("tree_touhou_rl", "exact", "touhou_rl", "deny",
     "touhou_rl/ is the package source"),
    ("tree_configs", "exact", "configs", "deny",
     "configs/ is frozen configuration (splits.json is a lineage input)"),
    ("tree_tests", "exact", "tests", "deny",
     "tests/ is source code"),
    ("tree_git", "exact", ".git", "deny",
     "the repository metadata is never a write target"),
)


def fence_rule_for(head: str) -> dict:
    """Which write-fence rule decides a repository-relative first component.

    Returns ``{"rule", "kind", "pattern", "verdict", "reason"}``; a first
    component no rule names is allowed, because the fence guards *historical
    trees inside the repository*, not arbitrary destinations (``--out
    /tmp/x.jsonl`` stays legal, as it always was).
    """

    import fnmatch

    for rule_id, kind, pattern, verdict, reason in WRITE_FENCE:
        if kind == "exact":
            hit = head == pattern
        elif kind == "prefix":
            hit = head.startswith(pattern)
        elif kind == "glob":
            hit = fnmatch.fnmatchcase(head, pattern)
        else:                                             # pragma: no cover
            raise PairError("INTERNAL", f"unknown fence rule kind {kind!r}")
        if hit:
            return {"rule": rule_id, "kind": kind, "pattern": pattern,
                    "verdict": verdict, "reason": reason}
    return {"rule": "unlisted_top_level", "kind": "none", "pattern": None,
            "verdict": "allow",
            "reason": (f"{head!r} is not named by any fence rule and is not a "
                       f".tmp_* tree, so it cannot be historical evidence")}


def fence_verdict(path: pathlib.Path, *, work_dir: pathlib.Path,
                  what: str = "path") -> dict:
    """Judge one write target and return the whole decision (never raises).

    The returned block is what ``assert_writable`` puts into its refusal (the
    "which rule decided this, and why" requirement) and what the unit tests
    assert on, so an allowed and a refused path are judged by the same function.
    """

    target = pathlib.Path(path).expanduser().resolve()
    root = pathlib.Path(work_dir).expanduser().resolve()
    block = {"what": str(what), "target": str(target), "work_dir": str(root),
             "top_level": None, "verdict": "allow", "rule": "work_dir",
             "pattern": None,
             "reason": None, "basis": None}
    if target == root or root in target.parents:
        block["reason"] = (f"inside --work-dir {root}, which this invocation "
                           f"owns")
        block["basis"] = "under --work-dir"
        return block
    try:
        rel = target.relative_to(REPO.resolve())
    except ValueError:
        block["rule"] = "outside_repository"
        block["reason"] = (f"outside the repository ({REPO.resolve()}), so it "
                           f"cannot be inside a historical run directory")
        block["basis"] = "not under the repository root"
        return block
    if not rel.parts:
        block.update({"rule": "repo_root", "verdict": "deny",
                      "reason": "the repository root itself is not a write target",
                      "basis": "relative_to(repo) has no parts"})
        return block
    head = rel.parts[0]
    rule = fence_rule_for(head)
    block.update({"top_level": head, "verdict": rule["verdict"],
                  "rule": rule["rule"], "pattern": rule["pattern"],
                  "reason": rule["reason"],
                  "basis": (f"top-level component {head!r} matched rule "
                            f"{rule['rule']} ({rule['kind']} {rule['pattern']!r}"
                            f" -> {rule['verdict']})")})
    return block

#: Compare criteria (trace key, report name), in report order.  Frozen: the
#: pre-registration sec. 3.2 names exactly these comparisons.
COMPARE_FIELDS: tuple[tuple[str, str], ...] = (
    ("threshold_px", "threshold"),
    ("nearest_px", "nearest_px"),
    ("cand", "cand"),
    ("masks", "mask"),
    ("bomb_request", "bomb_request"),
    ("inventory_before", "inventory_before"),
    ("inventory_after", "inventory_after"),
)
#: Trace header keys the two documents must agree on before a comparison means
#: anything.
COMPARE_HEADER_KEYS = ("prefix_frame", "decision_frames", "budget_decisions", "window")
#: The subset of the compared fields that is *behaviour* rather than rule input.
ACTION_FIELDS = ("cand", "masks", "bomb_request", "inventory_before",
                 "inventory_after")

# -- exit codes --------------------------------------------------------------
EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_UNUSABLE = 2

#: Codes that mean "the input could not be evaluated at all" (exit 2).
UNUSABLE_CODES = frozenset({
    "MANIFEST_UNREADABLE", "TRACE_UNREADABLE", "TRACE_INCOMPATIBLE",
    "ENGINE_MISSING", "DATA_SOURCE_MISSING", "CKPT_MISSING",
    "S1C_RECORDING_MISSING", "LINEAGE_CONFIG_UNREADABLE",
    # a non-integer --takeover-frame is a usage error, like argparse's own exit 2;
    # a recording that is not visible at run time is a missing input, not a finding
    "TAKEOVER_FRAME_NOT_INTEGER", "TAKEOVER_RECORDING_MISSING",
    # same rule for a declared prefix_tape: a tape this host cannot read is a
    # missing input (exit 2), while "the tape is not the digest the manifest pins"
    # is a verdict on the cell and stays exit 1
    "PREFIX_TAPE_MISSING",
    # --workers is a usage knob, not a verdict on the cell: a value outside
    # 1..MAX_WORKERS, or one asked of a mode that has no cells to spread, cannot
    # be evaluated at all (exit 2), like argparse's own usage failures
    "WORKERS_OUT_OF_RANGE", "WORKERS_MODE_UNSUPPORTED",
    # --reuse-park is likewise a usage knob: asking for it in a mode that has no
    # cells to share (trace/compare) or next to --workers N (one cell per process,
    # so two processes can never share one PARK) is a usage error, not a verdict
    "PARK_REUSE_MODE_UNSUPPORTED", "PARK_REUSE_WORKERS_UNSUPPORTED",
    "PARK_REUSE_NO_PARK_FN",
})

_LOG_FH = None
_CACHE: dict[str, object] = {}


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class PairError(RuntimeError):
    """A fail-closed refusal.  ``code`` is stable and greppable."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"[{code}] {detail}")
        self.code = str(code)
        self.detail = str(detail)

    @property
    def exit_code(self) -> int:
        return EXIT_UNUSABLE if self.code in UNUSABLE_CODES else EXIT_REFUSED


def log(message: str) -> None:
    line = f"[v829] {message}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, payload: object) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def load_json(path: pathlib.Path) -> object:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_module(name: str, path: pathlib.Path):
    """Import a repo script by path (the ``tools/rl`` modules are scripts)."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(name, pathlib.Path(path))
    if spec is None or spec.loader is None:
        raise PairError("MODULE_UNLOADABLE", f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # NB: not registered in sys.modules -- MAP sec. 9-12: every load is a new copy.
    spec.loader.exec_module(module)
    return module


def _mod(key: str, rel: str):
    """Cached by-path import of one ``tools/rl`` script, log handle attached."""

    module = _CACHE.get(key)
    if module is None:
        module = load_module(key, REPO / rel)
        if hasattr(module, "_LOG_FH"):
            module._LOG_FH = _LOG_FH
        _CACHE[key] = module
    elif hasattr(module, "_LOG_FH"):
        module._LOG_FH = _LOG_FH
    return module


def m829():
    """``tools/rl/v829_guard.py`` -- the manifest guard (never re-implemented)."""

    return _mod("v829_guard", GUARD_REL)


def m432():
    return _mod("v432_fork_arm", "tools/rl/v432_fork_arm.py")


def m450():
    return _mod("v450_cf_label", "tools/rl/v450_cf_label.py")


def m830():
    return _mod("v830_bombpol", "tools/rl/v830_bombpol.py")


def m860():
    return _mod("v860_bombdag", "tools/rl/v860_bombdag.py")


def m8r0():
    return _mod("v8r0_urgency", "tools/rl/v8r0_urgency.py")


def m8o0():
    return _mod("v8o0_ruletight", "tools/rl/v8o0_ruletight.py")


def arm_a():
    return _mod("v431_arm_a", "tools/rl/v431_arm_a.py")


def m_audio():
    return _mod("v431_audio_cfg_audit", "tools/rl/v431_audio_cfg_audit.py")


# --------------------------------------------------------------------------- #
# write confinement (requirement: never write a historical evidence tree)
# --------------------------------------------------------------------------- #
def assert_writable(path: pathlib.Path, *, work_dir: pathlib.Path,
                    what: str) -> pathlib.Path:
    """Refuse any target the explicit write fence denies; return absolute path.

    The decision itself is ``fence_verdict()``: the fence is an ordered rule
    table (``WRITE_FENCE``), not a whitelist of .tmp_* names, so ".tmp_v829_*"
    (this round, including new directories such as ``.tmp_v829_collect``) is
    allowed by rule while every other existing ``.tmp_*`` tree stays denied.
    The refusal names the rule that fired and why it fired; nothing is judged by
    "it starts with .tmp_ and is not one of two constants".
    """

    block = fence_verdict(path, work_dir=work_dir, what=what)
    if block["verdict"] == "deny":
        raise PairError(
            "HISTORICAL_PATH",
            f"{what} {block['target']} is REFUSED by write-fence rule "
            f"{block['rule']} [{block['basis']}]: {block['reason']}. "
            f"Allowed: --work-dir ({block['work_dir']}), the whole "
            f"{TMP_NAMESPACE}* namespace "
            f"({', '.join(ALLOWED_TMP)} are this round's named trees), and any "
            f"path outside the repository. Denied: every other existing .tmp_* "
            f"tree, Document/, data/, and the source trees "
            f"({', '.join(FORBIDDEN_TREES)}). The fence and its rules are "
            f"WRITE_FENCE in this file; run "
            f"`python tools/rl/v829_pair.py --explain-path PATH` to see the "
            f"verdict for any path without writing anything.")
    return pathlib.Path(block["target"])


def ensure_parent_dir(path: pathlib.Path) -> pathlib.Path:
    """Create ``path``'s parent so opening the file cannot fail *after* PARK.

    v8.29 opened ``--out``/``--log`` without creating their directory: on a
    fresh work dir the parent process parked, an arm ran, and only then
    ``out.open("a")`` raised FileNotFoundError (exit 2, "REFUSED [UNEXPECTED]";
    r1 evidence under .tmp_v829_pair/runs/v829-smoke-w11-c0-b2400-r1).  The
    mkdir belongs *before* the engine can spawn.  OSError is deliberately not
    swallowed -- main() still fails closed on it, so an unwritable --out stays a
    non-zero exit (2), exactly the code the late failure produced before.
    """

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def check_record_only(path: pathlib.Path) -> bool:
    """True when ``path`` holds nothing but a previous ``mode=check`` record.

    ``mode=check`` now writes ``runs/<tag>/run.json`` (the invocation record), so
    a second identical check must be allowed to overwrite **its own** record --
    and still must not be allowed to touch a real run (``runs/<tag>`` of a
    pair/trace run holds guard.json, plan.json, run.log, gate-w*.json, ...).
    Anything that is not exactly ``["run.json"]`` with ``mode == "check"`` is a
    collision, i.e. a refusal, exactly as before.
    """

    path = pathlib.Path(path)
    if not path.exists():
        return True
    if not path.is_dir():
        return False
    try:
        names = sorted(entry.name for entry in path.iterdir())
    except OSError:
        return False
    if not names:
        return True
    if names != ["run.json"]:
        return False
    try:
        document = load_json(path / "run.json")
    except (OSError, ValueError):
        return False
    return isinstance(document, dict) and document.get("mode") == "check"


# --------------------------------------------------------------------------- #
# arm identity: the only difference between the two arms is the table
# --------------------------------------------------------------------------- #
def check_arm(arm: str) -> str:
    key = str(arm).strip().upper()
    if key not in ARMS:
        raise PairError("ARM_UNKNOWN", f"--arm must be one of {list(ARMS)}, "
                                      f"got {arm!r}")
    return key


def parse_arm_order(text: str) -> list[str]:
    items = [part.strip().upper() for part in str(text).replace(" ", "").split(",")
             if part.strip()]
    if not items:
        raise PairError("ARM_ORDER_INVALID",
                        f"--arm-order must list {list(ARMS)}, got {text!r}")
    for arm in items:
        if arm not in ARMS:
            raise PairError("ARM_ORDER_INVALID",
                            f"--arm-order entry {arm!r} is not one of {list(ARMS)}")
    if len(set(items)) != len(items):
        raise PairError("ARM_ORDER_INVALID",
                        f"--arm-order repeats an arm: {items}")
    return items


def table_for(arm: str, *, budget_decisions: int, decisions_executed: int,
              bombs_left: int):
    """The **only** per-arm difference (MAP sec. 2.3).

    Returns ``(table, urgency, scale)``; ``urgency``/``scale`` are ``None`` for
    the fixed triple because T2 has no urgency term.
    """

    arm = check_arm(arm)
    if arm == "G1":
        urgency, scale, table = m8r0().thresholds_for(
            budget_decisions=int(budget_decisions),
            decisions_executed=int(decisions_executed),
            bombs_left=int(bombs_left),
            alpha=m8r0().alpha_of("G1"))
        return dict(table), float(urgency), float(scale)
    return dict(m8o0().thresholds("T2")), None, None


def rule_label_for(arm: str) -> tuple[str, str]:
    """``(label, source)``.  Never `v8r0.arm_label` (MAP sec. 9-4)."""

    arm = check_arm(arm)
    if arm == "G1":
        alpha = float(m8r0().alpha_of("G1"))
        return (f"URG-a{alpha:.2f}",
                "v8r0_urgency.alpha_of('G1') -> f'URG-a{alpha:.2f}' (arm_label "
                "NOT used: it reads getattr(parent,'arm','G2'), so it always "
                "reports URG-a1.00 -- MAP sec. 9-4)")
    return (str(m8o0().rule_label("T2")),
            "v8o0_ruletight.arm_spec('T2')['label'] (MAP sec. 9-4)")


def generator_seed(pair_rng: int) -> int:
    """``pair_rng * 1000 + 7`` -- the frozen family formula (docstring RNG sec. )."""

    value = int(pair_rng) * 1000 + GEN_CONST
    if not 0 <= value < (1 << 63):
        raise PairError("PAIR_RNG_INVALID",
                        f"pair_rng {pair_rng} derives seed {value}, outside "
                        f"torch's manual_seed range [0, 2**63)")
    return value


def generator_digest(generator) -> dict[str, object]:
    """Initial/current RNG state as a short digest, with a declared fallback."""

    try:
        state = generator.get_state()
        raw = state.numpy().tobytes()
        return {"method": "sha256(generator.get_state().numpy().tobytes())[:16]",
                "digest": hashlib.sha256(raw).hexdigest()[:16],
                "bytes": len(raw)}
    except Exception as exc:                                      # noqa: BLE001
        seed = int(generator.initial_seed())
        return {"method": ("fallback: sha256(str(generator.initial_seed()))[:16] "
                           "-- get_state() unavailable"),
                "digest": hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:16],
                "fallback_reason": f"{type(exc).__name__}: {exc}",
                "initial_seed": seed}


# --------------------------------------------------------------------------- #
# manifest -> roots (guard already validated the four frame fields)
# --------------------------------------------------------------------------- #
def manifest_document(args: argparse.Namespace) -> tuple[dict, int]:
    """Run the guard; return ``(guard_document, exit_code)``."""

    return m829().evaluate(pathlib.Path(args.manifest),
                           budget=int(args.budget_frames),
                           allow_exposed=bool(args.allow_exposed))


def manifest_from_single_read(args: argparse.Namespace,
                              guard_doc: dict) -> tuple[dict, dict]:
    """Read the manifest **once** and parse that exact buffer (A: byte closure).

    The guard has already read the file and judged *those* bytes; re-opening the
    path for the fields is a TOCTOU window (r1 REPORT sec. M4).  So:

    1. ``raw = path.read_bytes()`` -- the one and only read this file does;
    2. ``sha256(raw)`` must equal ``guard_doc["manifest_sha256"]`` (and
       ``len(raw)`` must equal ``guard_doc["manifest_bytes"]`` when the guard
       reported it), else ``MANIFEST_CHANGED`` -- nothing is parsed, nothing is
       used;
    3. ``json.loads(raw.decode("utf-8-sig"))`` -- the fields come from the very
       buffer whose digest was just compared, so there is no second window.

    If the guard document carries no ``manifest_sha256`` at all, the digest is
    computed here from the same buffer and ``manifest_sha256_source`` says
    ``"self"`` (weaker provenance: we cannot prove these are the bytes the guard
    judged, only that the bytes we use are the bytes we digested).
    """

    path = pathlib.Path(args.manifest)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PairError("MANIFEST_UNREADABLE",
                        f"cannot read {path}: {error.strerror or error}") from error
    digest = hashlib.sha256(raw).hexdigest()
    guard_digest = guard_doc.get("manifest_sha256")
    guard_bytes = guard_doc.get("manifest_bytes")
    source = "guard"
    if not (isinstance(guard_digest, str) and len(guard_digest) == 64):
        source = "self"
        guard_digest = digest
    if digest != str(guard_digest).lower():
        raise PairError(
            "MANIFEST_CHANGED",
            f"{path} changed between the guard's read and this file's read: the "
            f"guard verified sha256 {str(guard_digest).lower()} but the bytes read "
            f"now digest to {digest} ({len(raw)} B). The guard verdict belongs to "
            f"the earlier bytes, so this run would use data no rule checked; "
            f"refusing before reading a single field (r1 REPORT sec. M4). Rerun "
            f"against a manifest that is not being edited")
    if (isinstance(guard_bytes, int) and not isinstance(guard_bytes, bool)
            and int(guard_bytes) != len(raw)):
        raise PairError(
            "MANIFEST_CHANGED",
            f"{path}: the guard measured {guard_bytes} B but the file is "
            f"{len(raw)} B now (digest {digest}); the size the guard judged is not "
            f"the size in hand, so the manifest was replaced between the two reads")
    try:
        manifest = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as error:
        # The guard parsed these bytes, so with an equal digest this cannot happen
        # -- which is exactly why it must be a hard refusal if it ever does.
        raise PairError(
            "MANIFEST_CHANGED",
            f"{path}: the bytes digest to the verified {digest} but do not parse "
            f"({type(error).__name__}: {error}); treating this as a substituted "
            f"manifest instead of guessing") from error
    closure = {
        "schema": SCHEMA_CLOSURE,
        "manifest_path": str(path),
        "manifest_sha256": digest,
        "manifest_sha256_source": source,
        "manifest_sha256_guard": guard_doc.get("manifest_sha256"),
        "manifest_sha256_scope": "raw file bytes, no normalisation (same scope as "
                                 "v829_guard.evaluate: read_bytes -> sha256)",
        "manifest_bytes": len(raw),
        "manifest_bytes_guard": guard_bytes,
        "manifest_reads": 1,
        "manifest_read_scope": (
            "one path.read_bytes(); the digest is compared to the guard's before "
            "json.loads of that same buffer, so the checked bytes and the used "
            "bytes are the same object and no second read exists to be replaced"),
        "manifest_changed": False,
    }
    if source == "self":
        closure["residual_risk"] = [
            "the guard document carried no manifest_sha256, so the digest proves "
            "only that this run used the bytes it digested -- not that the guard "
            "judged those bytes (MANIFEST_CHANGED cannot be detected in this mode)"]
    return manifest, closure


def budget_selection_line(block: dict) -> str:
    """The one stdout line a mixed-tier selection prints (the selector's record)."""

    return (f"budget {int(block['requested'])}: selected {int(block['matched'])} "
            f"root(s), skipped {int(block['skipped_by_budget'])} (other tiers)")


def select_budget_tier(args: argparse.Namespace, candidates: list[dict], *,
                       manifest_indices: list[int]) -> tuple[list[dict], dict]:
    """Second selection pass: keep the roots declared at ``--budget-frames``.

    ``--budget-frames`` is a **selector over the manifest subset**, exactly like
    ``--window`` (module docstring, "Selectors vs registered values").  One
    manifest can register both tiers of pre-registration sec. 2's pilot shape
    ("8 ancestors x 3 ckpt x 2 RNG x 2 budgets"), and each tier is then run as
    its own invocation against that one file.  The option is deliberately *not*
    an assertion that every root is at that tier: that assertion is what this
    function replaces, and it is why a 96-root two-tier manifest used to be
    refused (``BUDGET_ROOT_MISMATCH``) at **both** tiers before a single root
    could be selected.

    Ordering matters for the refusal codes: ``--window`` runs first
    (``select_roots``), so an invocation that names an absent window still gets
    ``WINDOW_NOT_IN_MANIFEST`` verbatim.  Only then is the budget applied, and an
    empty result is ``BUDGET_TIER_ABSENT`` (exit 1) -- a tier the manifest does
    not register is a usage error, never a silent no-op run of zero cells.

    A root whose ``budget_frames`` is missing or not an integer is **kept** here
    and refused by ``check_root`` with ``MANIFEST_INVALID`` and its own label, so
    the more precise code keeps winning; it is not this selector's job to
    re-judge a field the guard and ``check_root`` already own.

    The counters are named so that two invocations of the *same* mixed-tier
    manifest can be compared without arithmetic:

    * ``matched``             -- roots this invocation runs (``kept``);
    * ``skipped_by_budget``   -- every manifest root the selection leaves out,
                                 i.e. ``manifest_roots - matched`` (a window
                                 exclusion is an exclusion from the same run);
    * ``skipped_by_window``   -- how many of those the window step removed
                                 (``manifest_roots - roots_after_window``);
    * ``dropped_by_budget_step`` -- how many the budget step removed from the
                                 roots the window step left (the strictly
                                 "other tier" part).

    ``candidates`` are the roots the window step kept (with ``_index`` already
    set), ``manifest_indices`` is every manifest root index in file order.
    Returns ``(kept, block)`` where ``block`` is the ``budget_selection`` record.
    """

    requested = int(args.budget_frames)
    source = str(getattr(args, "budget_frames_source", "cli"))
    manifest_roots = len(manifest_indices)
    kept: list[dict] = []
    skipped: list[dict] = []
    tiers: dict[int, int] = {}
    for root in candidates:
        declared = root.get("budget_frames")
        if isinstance(declared, bool) or not isinstance(declared, int):
            kept.append(root)               # check_root -> MANIFEST_INVALID
            continue
        tiers[int(declared)] = tiers.get(int(declared), 0) + 1
        if int(declared) == requested:
            kept.append(root)
        else:
            skipped.append(root)
    if not kept:
        window_note = ("" if args.window is None
                       else f" within --window {int(args.window)}")
        raise PairError(
            "BUDGET_TIER_ABSENT",
            f"no root in the manifest declares budget_frames {requested}"
            f"{window_note}; --budget-frames is a SELECTOR over the manifest "
            f"subset, and it selected 0 of the {len(candidates)} root(s) the "
            f"window step left, so this invocation would run nothing -- refusing "
            f"instead of exiting 0 on an empty plan. The manifest declares "
            f"tier(s) {sorted(tiers)} (counts "
            f"{ {str(tier): count for tier, count in sorted(tiers.items())} }); "
            f"pass one of them, or build a new root/manifest version for a new "
            f"tier")
    matched_roots = [int(root["_index"]) for root in kept]
    matched_set = set(matched_roots)
    skipped_roots = [int(index) for index in manifest_indices
                     if int(index) not in matched_set]
    block = {
        "schema": SCHEMA_BUDGET_SELECTION,
        "requested": requested,
        "requested_source": source,
        "matched": len(kept),
        "skipped_by_budget": len(skipped_roots),
        "matched_roots": matched_roots,
        "skipped_roots": skipped_roots,
        "skipped_by_window": manifest_roots - len(candidates),
        "dropped_by_budget_step": len(skipped),
        "declared_tiers": {str(tier): count
                           for tier, count in sorted(tiers.items())},
        "manifest_roots": manifest_roots,
        "roots_after_window": len(candidates),
        "window": (None if args.window is None else int(args.window)),
        "scope": ("--budget-frames is a SELECTOR over the manifest subset, "
                  "applied after --window: matched_roots are the roots this "
                  "invocation runs, skipped_roots are the manifest roots it "
                  "leaves out (skipped_by_window of them went to another window, "
                  "dropped_by_budget_step to another tier). The other pinned "
                  "options (--pair-rng / --takeover-frame / --prefix-tape / "
                  "--ckpt-sha256) are REGISTERED VALUES, not selectors: each "
                  "must equal the root's declared value or the cell is refused"),
        "line": budget_selection_line({"requested": requested,
                                       "matched": len(kept),
                                       "skipped_by_budget": len(skipped_roots)}),
    }
    return kept, block


def select_roots(manifest: object, args: argparse.Namespace) -> list[dict]:
    """Resolve every root this invocation would touch, failing closed on any gap.

    Two selectors, in this order: ``--window`` (below) and ``--budget-frames``
    (``select_budget_tier``).  The result is the root set this invocation runs,
    and the ``budget_selection`` record is stored on ``args`` so the plan
    document, ``run.json`` and the stdout line all report the same object
    instead of re-deriving it from the root list.
    """

    if not isinstance(manifest, dict):
        raise PairError("MANIFEST_UNREADABLE", "manifest is not a JSON object")
    roots = manifest.get("roots")
    if not isinstance(roots, list) or not roots:
        raise PairError("MANIFEST_INVALID", "manifest has no non-empty 'roots' array")
    selected: list[dict] = []
    for index, root in enumerate(roots):
        if not isinstance(root, dict):
            raise PairError("MANIFEST_INVALID", f"roots[{index}] is not an object")
        declared = root.get("window")
        if declared is None:
            if args.window is None:
                raise PairError(
                    "WINDOW_NOT_DECLARED",
                    f"roots[{index}] has no 'window' and --window was not given: "
                    f"the window picks the S1c prefix recording and the "
                    f"w<NN> data dir, so it cannot be guessed")
            window, window_source = int(args.window), "cli"
        else:
            if isinstance(declared, bool) or not isinstance(declared, int):
                raise PairError("MANIFEST_INVALID",
                                f"roots[{index}].window must be an integer, got "
                                f"{declared!r}")
            window, window_source = int(declared), "manifest"
        if args.window is not None and window != int(args.window):
            continue
        resolved = dict(root)
        resolved["_index"] = index
        resolved["window"] = window
        resolved["window_source"] = window_source
        selected.append(resolved)
    if not selected:
        raise PairError(
            "WINDOW_NOT_IN_MANIFEST",
            f"no root in the manifest matches --window {args.window}; the "
            f"manifest declares windows "
            f"{sorted({r.get('window') for r in roots if isinstance(r, dict)})}")
    selected, selection = select_budget_tier(
        args, selected, manifest_indices=list(range(len(roots))))
    args.budget_selection = selection
    return selected


def check_root(root: dict, args: argparse.Namespace, *, needs_engine: bool) -> list[str]:
    """Every per-root gate that needs no engine.  Returns notes, raises on refusal."""

    notes: list[str] = []
    label = f"roots[{root['_index']}] {root.get('ancestor_id')!r}"
    window = int(root["window"])
    # 0a. `lineage_new` is the opt-in that lets a research-side NEW ancestor (not
    #     in configs/v430/splits.json) use --lineage-config at all.  v829_guard
    #     ignores unknown root fields on purpose, so the *type* is judged here:
    #     a string "true" would otherwise read as absent and silently fall back
    #     to LINEAGE_MISSING.
    if root.get(LINEAGE_NEW_FIELD) is not None and not isinstance(
            root.get(LINEAGE_NEW_FIELD), bool):
        raise PairError(
            "MANIFEST_INVALID",
            f"{label}: '{LINEAGE_NEW_FIELD}' must be a JSON boolean, got "
            f"{root.get(LINEAGE_NEW_FIELD)!r}; the flag decides whether a root "
            f"without lineage_ancestor_ids is refused (LINEAGE_MISSING) or "
            f"recorded (LINEAGE_NEW), so it cannot be judged from a string")
    # 0. optional prefix_tape: identity first (path + digests), before anything
    #    that needs to know how long the tape is.  Absent -> None and every branch
    #    below is the pre-tape one.
    tape = resolve_prefix_tape(root, args)
    root["_prefix_tape"] = tape
    root["_prefix_tape_state"] = None
    if window not in OLD_WINDOWS:
        if needs_engine and tape is None:
            raise PairError(
                "WINDOW_NOT_SUPPORTED",
                f"{label}: window {window} is not one of {list(OLD_WINDOWS)}. "
                f"v1 can only recover the four old windows; a new takeover frame "
                f"needs m432.PREFIX_FRAMES / m450.TARGET_FRAME / m830.PREFIX_FRAMES "
                f"re-pointed, which is a separate authorisation (MAP sec. 9-1). "
                f"Refusing rather than patching a module global.")
        if tape is None:
            notes.append(f"{label}: window {window} outside {list(OLD_WINDOWS)} "
                         f"-> would be WINDOW_NOT_SUPPORTED in pair/trace")
        else:
            notes.append(
                f"{label}: window {window} is outside {list(OLD_WINDOWS)} but the "
                f"root declares prefix_tape, so the prefix does not come from this "
                f"window's S1c recording; the window now only labels the private "
                f"data dir w{window} and the takeover frame's anchor rule is judged "
                f"against the tape")
    takeover = root.get("takeover_frame")
    if isinstance(takeover, bool) or not isinstance(takeover, int):
        raise PairError("MANIFEST_INVALID",
                        f"{label}: takeover_frame must be an integer, got {takeover!r}")
    tape_scan = scan_tape(tape["path"]) if tape is not None else None
    resolution = resolve_takeover_frame(root, args, needs_engine=needs_engine,
                                        tape=tape_scan)
    root["_takeover_resolution"] = resolution
    if tape is not None:
        # 0b. the strict reader a run uses: it is also what mode=check validates
        #     against, so a tape no run could walk is refused offline too
        masks, hashes, meta = read_tape(tape["path"], resolution["target_frame"])
        gate = tape_meta_gate(root, meta)
        root["_prefix_tape_state"] = {"meta": meta, "masks": masks,
                                      "hashes": hashes, "gate": gate,
                                      "scan": tape_scan}
        notes.append(
            f"{label}: prefix_tape {tape['path']} sha256 {meta['sha256']} "
            f"({meta['bytes']} B) -> {meta['rows']} rows x {ANCHOR_STRIDE} = "
            f"{meta['frames_built']} frames_built, {meta['recorded_hash_frames']} "
            f"recorded hash frames, anchors_complete={meta['anchors_complete']}; "
            f"target {meta['target']} is a recorded anchor; v450.masks_for is "
            f"re-pointed at this tuple instead of build_masks' S1c shard")
    if resolution["override_active"]:
        tape_rec = resolution["recording"]
        notes.append(
            f"{label}: takeover_frame {takeover} is not the frozen {PREFIX_FRAMES} "
            f"-> prefix constants re-pointed in-process (PREFIX_FRAMES="
            f"{takeover}, CHECKPOINTS_IN_PREFIX="
            f"{resolution['expected_checkpoints']}, TARGET_FRAME="
            f"{resolution['target_frame']}); target {resolution['target_frame']} is "
            f"a recorded anchor (N % 4 == 3) and the tape covers "
            f"{tape_rec.get('frames_built')} frames")
    declared_budget = root.get("budget_frames")
    if isinstance(declared_budget, bool) or not isinstance(declared_budget, int):
        raise PairError("MANIFEST_INVALID",
                        f"{label}: budget_frames must be an integer, "
                        f"got {declared_budget!r}")
    if int(declared_budget) != int(args.budget_frames):
        # POST-SELECTION INVARIANT, not the selection rule.  `--budget-frames` is
        # a selector (select_roots -> select_budget_tier drops every root of
        # another tier, BUDGET_TIER_ABSENT when that empties the set), so a
        # selected root that still disagrees means the selector was bypassed or
        # the root set was built by hand.  Refusing keeps the old code and the old
        # guarantee: a run never executes a root at a tier the manifest did not
        # register for it.
        raise PairError(
            "BUDGET_ROOT_MISMATCH",
            f"{label}: the manifest declares budget_frames {declared_budget} but "
            f"--budget-frames is {args.budget_frames} -- this root should have "
            f"been filtered out by the --budget-frames selector before any mode "
            f"reached it (defensive assertion: the run table is the manifest, so a "
            f"run at a different tier would not be the registered one)")
    engine_declared = check_engine_declaration(root)
    pair_rng = root.get("pair_rng")
    if isinstance(pair_rng, bool) or not isinstance(pair_rng, int) or pair_rng < 0:
        raise PairError("PAIR_RNG_ABSENT",
                        f"{label}: pair_rng must be a non-negative integer, "
                        f"got {pair_rng!r} (pre-registration sec. 1 pre-allocates it)")
    prefix_gate(root)      # C: parent_prefix_digest must be *this* window's tape
    return notes


def prefix_digest_gate(root: dict) -> dict:
    """C: compare ``parent_prefix_digest`` with this window's recording digest.

    ``parent_prefix_digest`` used to be recorded only, so a manifest could carry
    another window's recording digest (the smoke manifest carried the seed12 one,
    ``f79f8ab6...``, for window 11, whose own recording is ``bc663759...``) and
    the run would still pass.  Here it is **compared**:

    * ``sha256(s1c_recording_path(window))`` must equal ``parent_prefix_digest``;
    * ``verify_anchor.window_recording_sha256``, when present, must equal the same
      digest (that field names the window's own recording, so a disagreement is a
      second, independent mismatch);
    * a mismatch -> ``PREFIX_RECORDING_MISMATCH`` (fail-closed, exit 1);
    * when the recording is not readable from this host the comparison is
      deferred to the run host (``gate = "deferred-to-run"``), exactly like the
      engine/checkpoint digests: ``mode=check`` must stay usable on a host that
      cannot see ``.tmp_v410_s1c``, and refusing there would report a missing
      asset as a wrong digest.
    """

    label = f"roots[{root.get('_index')}] {root.get('ancestor_id')!r}"
    window = int(root["window"])
    recording = s1c_recording_path(window)
    declared = root.get("parent_prefix_digest")
    anchor = root.get("verify_anchor")
    anchor_declared = (anchor.get("window_recording_sha256")
                       if isinstance(anchor, dict) else None)
    entry: dict = {
        "window": window, "recording": str(recording),
        "declared": declared, "anchor_declared": anchor_declared,
        "actual": None, "bytes": None, "match": None, "anchor_match": None,
        "gate": "deferred-to-run",
        "scope": ("sha256 of the whole S1c labels.jsonl for this window; the "
                  "per-window prefix digest is the recording digest"),
    }
    if not recording.is_file():
        entry["note"] = (
            "the S1c recording of this window is not visible from this host, so "
            "parent_prefix_digest cannot be compared here; the comparison stays "
            "fail-closed on the run host (PREFIX_RECORDING_MISMATCH)")
        return entry
    actual = recording_sha256(recording)
    size = recording.stat().st_size
    entry.update({"actual": actual, "bytes": size, "gate": "checked",
                  "match": bool(isinstance(declared, str)
                                and declared.lower() == actual)})
    if not entry["match"]:
        raise PairError(
            "PREFIX_RECORDING_MISMATCH",
            f"{label}: parent_prefix_digest {declared!r} is not the digest of this "
            f"window's recording {recording} ({actual}, {size} B). The digest is a "
            f"per-window value: the four old windows were measured on four "
            f"different S1c recordings, and {PREFIX_RECORD_SHA256[:16]}... is the "
            f"*seed12* recording (MAP sec. 1 / D3 gate), not a shared constant. "
            f"Recording the digest of another window is the error this gate "
            f"refuses")
    if anchor_declared is not None:
        entry["anchor_match"] = bool(isinstance(anchor_declared, str)
                                     and anchor_declared.lower() == actual)
        if not entry["anchor_match"]:
            raise PairError(
                "PREFIX_RECORDING_MISMATCH",
                f"{label}: verify_anchor.window_recording_sha256 "
                f"{anchor_declared!r} != sha256({recording}) {actual} "
                f"({size} B); the field names this window's own recording, so it "
                f"must be that recording's digest")
    return entry


def recording_sha256(path: pathlib.Path) -> str:
    """``sha256_file`` with a small cache (a run checks the same 5.7 MB recording)."""

    path = pathlib.Path(path)
    try:
        stat = path.stat()
    except OSError:
        return sha256_file(path)
    key = f"recording-sha256:{path}:{stat.st_size}:{stat.st_mtime_ns}"
    cached = _CACHE.get(key)
    if cached is None:
        cached = sha256_file(path)
        _CACHE[key] = cached
    return str(cached)


# --------------------------------------------------------------------------- #
# B: exposure cross-check against the frozen genealogy
# --------------------------------------------------------------------------- #
def _read_json_bytes(path: pathlib.Path, *, code: str, what: str
                     ) -> tuple[object, str, int]:
    """Read + digest + parse one file, once (the same one-read discipline as A)."""

    try:
        raw = pathlib.Path(path).read_bytes()
    except OSError as error:
        raise PairError(code, f"cannot read {what} {path}: "
                              f"{error.strerror or error}") from error
    digest = hashlib.sha256(raw).hexdigest()
    try:
        parsed = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as error:
        raise PairError(code, f"{what} {path} is not JSON this tool can read: "
                              f"{type(error).__name__}: {error}") from error
    return parsed, digest, len(raw)


def lineage_document(config: pathlib.Path) -> dict:
    """Load ``splits.json`` + the pretraining-source registry into lookup maps."""

    document, digest, nbytes = _read_json_bytes(
        config, code="LINEAGE_CONFIG_UNREADABLE", what="lineage config")
    if not isinstance(document, dict):
        raise PairError("LINEAGE_CONFIG_UNREADABLE",
                        f"{config}: a lineage config must be a JSON object, got "
                        f"{type(document).__name__}")
    splits = document.get("splits")
    if not isinstance(splits, dict):
        raise PairError("LINEAGE_CONFIG_UNREADABLE",
                        f"{config} has no 'splits' object; the exposure genealogy of "
                        f"configs/v430/splits.json is splits.<name>.ancestor_ids")
    membership: dict[str, list[str]] = {}
    split_counts: dict[str, int] = {}
    for name in LINEAGE_SPLIT_ORDER:
        block = splits.get(name)
        ids = block.get("ancestor_ids") if isinstance(block, dict) else None
        if ids is None:
            ids = []
        if not isinstance(ids, list):
            raise PairError("LINEAGE_CONFIG_UNREADABLE",
                            f"{config}: splits.{name}.ancestor_ids must be an array, "
                            f"got {type(ids).__name__}")
        clean = [str(entry) for entry in ids]
        split_counts[name] = len(clean)
        for ancestor in clean:
            membership.setdefault(ancestor, []).append(name)

    # -- the registry named by this very config (never a hand-typed path) -----
    sources_path: pathlib.Path | None = None
    sources_declared: str | None = None
    registries = document.get("registries")
    if isinstance(registries, list):
        for entry in registries:
            if isinstance(entry, dict) and entry.get("role") == LINEAGE_SOURCES_ROLE:
                sources_path = repo_path(entry.get("path"))
                declared = entry.get("sha256")
                sources_declared = str(declared) if isinstance(declared, str) else None
    if sources_path is None:
        fallback = repo_path(LINEAGE_SOURCES_DEFAULT)
        if fallback.is_file():
            sources_path = fallback
    status: dict[str, list[dict]] = {}
    sources_info: dict = {"path": None if sources_path is None else str(sources_path),
                          "checked": False, "sha256": None, "bytes": None,
                          "sha256_declared": sources_declared, "sha256_match": None,
                          "source_count": 0, "status_counts": {}}
    residual: list[str] = []
    if sources_path is None or not sources_path.is_file():
        residual.append(
            "pretraining-sources registry not readable from this host, so only "
            "splits.<name>.ancestor_ids were cross-checked; sources[].status was "
            "not consulted")
    else:
        sources_doc, sources_digest, sources_bytes = _read_json_bytes(
            sources_path, code="LINEAGE_CONFIG_UNREADABLE",
            what="pretraining-sources registry")
        raw_sources = (sources_doc.get("sources")
                       if isinstance(sources_doc, dict) else None)
        if not isinstance(raw_sources, list):
            raise PairError("LINEAGE_CONFIG_UNREADABLE",
                            f"{sources_path} has no 'sources' array")
        counts: dict[str, int] = {}
        for source in raw_sources:
            if not isinstance(source, dict):
                continue
            source_id = source.get("source_id")
            source_status = source.get("status")
            if isinstance(source_status, str):
                counts[source_status] = counts.get(source_status, 0) + 1
            ids = source.get("ancestor_ids")
            if not isinstance(ids, list):
                continue
            for ancestor in ids:
                status.setdefault(str(ancestor), []).append(
                    {"source_id": source_id, "status": source_status})
        sources_info.update({"checked": True, "sha256": sources_digest,
                             "bytes": sources_bytes, "source_count":
                             len(raw_sources), "status_counts": counts})
        if sources_declared is not None:
            sources_info["sha256_match"] = bool(
                sources_declared.lower() == sources_digest)
            if not sources_info["sha256_match"]:
                residual.append(
                    f"{sources_path} digest {sources_digest[:16]}... does not match "
                    f"the digest {sources_declared[:16]}... that "
                    f"{pathlib.Path(config).name} freezes for it; the status "
                    f"cross-check used the file on disk")
    return {
        "schema": SCHEMA_LINEAGE,
        "config_path": str(config),
        "config_sha256": digest,
        "config_bytes": nbytes,
        "split_order": list(LINEAGE_SPLIT_ORDER),
        "split_counts": split_counts,
        "exposed_splits": list(LINEAGE_EXPOSED_SPLITS),
        "splits_declared": sorted(str(name) for name in splits),
        "sources": sources_info,
        "membership": membership,
        "status": status,
        "residual_risk": residual,
    }


def _split_of(lineage: dict, ancestor_ids: list[str]) -> str | None:
    """Highest-precedence split any of ``ancestor_ids`` belongs to (or ``None``)."""

    found = {name for ancestor in ancestor_ids
             for name in lineage["membership"].get(ancestor, [])}
    for name in LINEAGE_SPLIT_ORDER:
        if name in found:
            return name
    return None


def _status_of(lineage: dict, ancestor_ids: list[str]) -> tuple[str | None,
                                                               list[dict]]:
    """Highest-precedence registry status (and the raw ``(source_id, status)`` list)."""

    rows: list[dict] = []
    seen: set[tuple] = set()
    for ancestor in ancestor_ids:
        for row in lineage["status"].get(ancestor, []):
            key = (row.get("source_id"), row.get("status"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(row))
    statuses = {row.get("status") for row in rows}
    for name in LINEAGE_STATUS_ORDER:
        if name in statuses:
            return name, rows
    return (rows[0].get("status") if rows else None), rows


def lineage_root_entry(root: dict, lineage: dict) -> dict:
    """One root's B verdict, returned (never raised: the caller records them all)."""

    label = f"roots[{root.get('_index')}] {root.get('ancestor_id')!r}"
    declared_ids = root.get("lineage_ancestor_ids")
    # A research-side NEW ancestor is not in the frozen genealogy at all, so a
    # non-empty lineage_ancestor_ids is impossible by construction.  The root may
    # say so explicitly (`lineage_new: true`) and then be RECORDED instead of
    # refused; see the module docstring ("exposure lineage").
    lineage_new = root.get(LINEAGE_NEW_FIELD) is True
    entry: dict = {
        "root_index": root.get("_index"),
        "ancestor_id": root.get("ancestor_id"),
        "window": int(root.get("window")) if root.get("window") is not None else None,
        "manifest_ancestor_id": root.get("ancestor_id"),
        "lineage_ancestor_ids": declared_ids,
        "lineage_new": lineage_new,
        "used_for_training": root.get("used_for_training") is True,
        "used_for_selection": root.get("used_for_selection") is True,
        "lineage_split": None,
        "lineage_status": None,
        "lineage_statuses": [],
        "lineage_matched_ids": [],
        "lineage_unmatched": [],
        "lineage_exposed": False,
        "declared_unexposed": (root.get("used_for_training") is False
                               and root.get("used_for_selection") is False),
        "code": None, "verdict": "OK", "detail": None,
    }
    if (not isinstance(declared_ids, list) or not declared_ids
            or not all(isinstance(item, str) and item for item in declared_ids)):
        if lineage_new:
            entry.update({
                "code": "LINEAGE_NEW", "verdict": "RECORDED",
                "detail": (f"{label}: 'lineage_new': true, so the missing/empty "
                           f"lineage_ancestor_ids {declared_ids!r} is expected "
                           f"rather than a gap: this ancestor is newer than the "
                           f"frozen genealogy and there is nothing in "
                           f"configs/v430/splits.json to cross-check it against. "
                           f"Recorded as LINEAGE_NEW (exit 0) and named in "
                           f"run.json.lineage_new_roots; exposure stays whatever "
                           f"the run table declares "
                           f"(training={entry['used_for_training']}, "
                           f"selection={entry['used_for_selection']})")})
            return entry
        entry.update({
            "code": "LINEAGE_MISSING", "verdict": "REFUSED",
            "detail": (f"{label}: --lineage-config was given, so the root must "
                       f"declare 'lineage_ancestor_ids' as a non-empty array of "
                       f"strings (or 'lineage_new': true for an ancestor the "
                       f"frozen genealogy does not have); got {declared_ids!r}. "
                       f"Without it the declared exposure cannot be cross-checked "
                       f"at all")})
        return entry
    ids = [str(item) for item in declared_ids]
    matched = [item for item in ids
               if item in lineage["membership"] or item in lineage["status"]]
    unmatched = [item for item in ids if item not in matched]
    split = _split_of(lineage, ids)
    status_value, status_rows = _status_of(lineage, ids)
    exposed_statuses = sorted({str(row.get("status")) for row in status_rows
                               if row.get("status") in LINEAGE_STATUS_EXPOSED})
    split_exposed = split in LINEAGE_EXPOSED_SPLITS
    entry.update({
        "lineage_split": split,
        "lineage_status": status_value,
        "lineage_statuses": status_rows,
        "lineage_matched_ids": matched,
        "lineage_unmatched": unmatched,
        "lineage_splits": sorted({name for ancestor in ids
                                  for name in lineage["membership"].get(ancestor, [])}),
        "lineage_exposed_statuses": exposed_statuses,
        "lineage_exposed": bool(split_exposed or exposed_statuses),
        "lineage_evidence": (
            f"split={split}" if split_exposed else
            (f"status={','.join(exposed_statuses)}" if exposed_statuses else None)),
    })
    declared_exposed = bool(entry["used_for_training"] or entry["used_for_selection"])
    if lineage_new and entry["lineage_exposed"]:
        # `lineage_new: true` says "the frozen genealogy does not contain this
        # ancestor"; an id that *does* hit train/dev says the opposite.  Both at
        # once is self-contradictory, whatever the exposure flags claim.
        where = []
        if split_exposed:
            where.append(f"splits.{split}.ancestor_ids")
        if exposed_statuses:
            where.append(f"pretraining_sources.status in {exposed_statuses}")
        entry.update({
            "code": "EXPOSURE_CONTRADICTS_LINEAGE", "verdict": "REFUSED",
            "detail": (f"{label}: declares 'lineage_new': true (a NEW ancestor "
                       f"the frozen genealogy does not enumerate) but also "
                       f"declares lineage_ancestor_ids {matched}, which ARE "
                       f"consumed by " + " and ".join(where)
                       + f". 'New' and 'already in the frozen genealogy' cannot "
                         f"both be true; refusing instead of picking one")})
        return entry
    if entry["declared_unexposed"] and entry["lineage_exposed"]:
        where = []
        if split_exposed:
            where.append(f"splits.{split}.ancestor_ids")
        if exposed_statuses:
            where.append(f"pretraining_sources.status in {exposed_statuses}")
        entry.update({
            "code": "EXPOSURE_CONTRADICTS_LINEAGE", "verdict": "REFUSED",
            "detail": (f"{label}: declares used_for_training=false and "
                       f"used_for_selection=false, but lineage_ancestor_ids "
                       f"{matched} are consumed by "
                       + " and ".join(where)
                       + f". A false/false declaration routes an already-consumed "
                         f"window into a fresh-sample slot; refusing instead of "
                         f"trusting the self-declaration")})
        return entry
    if lineage_new:
        # new ancestor, declared ids that the genealogy happens not to know: the
        # verdict is the same LINEAGE_NEW, but the match/unmatch evidence above
        # is kept so the record shows exactly what was looked up
        entry.update({
            "code": "LINEAGE_NEW", "verdict": "RECORDED",
            "detail": (f"{label}: 'lineage_new': true -- {len(matched)} of "
                       f"{len(ids)} declared lineage_ancestor_ids are already in "
                       f"the frozen genealogy (split={split}, "
                       f"status={status_value}) but none of them is train/dev or a "
                       f"consumed source, so the NEW-ancestor declaration does not "
                       f"contradict it. Recorded as LINEAGE_NEW; exposure stays "
                       f"training={entry['used_for_training']}, "
                       f"selection={entry['used_for_selection']}")})
        return entry
    if declared_exposed and not matched:
        entry.update({
            "code": "LINEAGE_UNKNOWN", "verdict": "UNMATCHED-RECORDED",
            "detail": (f"{label}: declares an exposed root "
                       f"(used_for_training={entry['used_for_training']}, "
                       f"used_for_selection={entry['used_for_selection']}) but none "
                       f"of lineage_ancestor_ids {ids} appears in "
                       f"splits.{{{','.join(LINEAGE_SPLIT_ORDER)}}}.ancestor_ids or "
                       f"in pretraining_sources.sources[].ancestor_ids. Recorded in "
                       f"run.json.lineage_unmatched and NOT refused: an old window "
                       f"may predate the frozen genealogy, and refusing would turn "
                       f"missing evidence into a contradiction")})
        return entry
    entry["detail"] = (
        f"{label}: consistent -- declares training={entry['used_for_training']}, "
        f"selection={entry['used_for_selection']}; lineage says split={split}, "
        f"status={status_value}, unmatched={unmatched}")
    return entry


def lineage_audit(*, roots: list[dict], args: argparse.Namespace) -> dict:
    """The whole B block: ``lineage_checked``, per-root verdicts, residual risk.

    Refusals are recorded **before** they are raised, so a rejected invocation
    still leaves ``<work>/runs/<tag>/lineage.json`` behind (the same evidence
    discipline as the audio audit).
    """

    if not getattr(args, "lineage_config", None):
        return {
            "schema": SCHEMA_LINEAGE, "lineage_checked": False,
            "lineage_config_path": None, "lineage_config_sha256": None,
            "lineage_config_bytes": None, "lineage_sources_path": None,
            "lineage_sources_sha256": None, "lineage_sources_checked": False,
            "lineage_codes": [], "lineage_unmatched": [],
            "residual_risk": [RESIDUAL_EXPOSURE],
            "roots": [],
            "note": ("--lineage-config was not given: exposure stayed "
                     "self-declared; run.json records lineage_checked=false and "
                     "this residual risk instead of a verdict"),
        }
    config = repo_path(args.lineage_config)
    lineage = lineage_document(config)
    entries = [lineage_root_entry(root, lineage) for root in roots]
    unmatched = [{"root_index": entry["root_index"],
                  "ancestor_id": entry["ancestor_id"],
                  "lineage_ancestor_ids": entry["lineage_ancestor_ids"],
                  "unmatched": entry["lineage_unmatched"],
                  "code": entry["code"]}
                 for entry in entries if entry["lineage_unmatched"]]
    codes = sorted({entry["code"] for entry in entries if entry["code"]})
    # `lineage_new: true` roots: recorded, named and counted, never refused.  A
    # run made only of them still exits 0 -- the point of the flag is that the
    # switch stays usable for an ancestor the frozen genealogy cannot contain.
    new_roots = sorted({int(entry["root_index"]) for entry in entries
                        if entry.get("lineage_new")})
    residual = list(lineage["residual_risk"])
    if new_roots:
        residual.append(RESIDUAL_LINEAGE_NEW)
    for entry in entries:
        if (entry["lineage_unmatched"] and not entry["lineage_exposed"]
                and not entry.get("lineage_new")):
            residual.append(
                f"roots[{entry['root_index']}] {entry['ancestor_id']!r} declares "
                f"used_for_training={entry['used_for_training']} / "
                f"used_for_selection={entry['used_for_selection']} and "
                f"{len(entry['lineage_unmatched'])} of its lineage_ancestor_ids are "
                f"in no split and no source: the genealogy neither confirms nor "
                f"refutes the declaration (absence of evidence is not evidence of "
                f"absence)")
    fatal = [entry for entry in entries
             if entry["code"] in ("LINEAGE_MISSING",
                                 "EXPOSURE_CONTRADICTS_LINEAGE")]
    document = {
        "schema": SCHEMA_LINEAGE,
        "lineage_checked": True,
        "lineage_config_path": lineage["config_path"],
        "lineage_config_sha256": lineage["config_sha256"],
        "lineage_config_bytes": lineage["config_bytes"],
        "lineage_sources_path": lineage["sources"]["path"],
        "lineage_sources_sha256": lineage["sources"]["sha256"],
        "lineage_sources_checked": lineage["sources"]["checked"],
        "lineage_splits": lineage["split_counts"],
        "lineage_split_order": lineage["split_order"],
        "lineage_exposed_splits": lineage["exposed_splits"],
        "lineage_sources_sha256_declared": lineage["sources"]["sha256_declared"],
        "lineage_sources_sha256_match": lineage["sources"]["sha256_match"],
        "lineage_codes": codes,
        "lineage_unmatched": unmatched,
        "residual_risk": residual,
        "roots": entries,
        "refused": bool(fatal),
    }
    if new_roots:
        # present ONLY when a root opted in, so every pre-`lineage_new`
        # invocation keeps the exact document (and the exact stdout) it had
        document["lineage_new_roots"] = new_roots
    if fatal:

        _write_lineage_refusal(roots=roots, args=args, document=document)
        first = fatal[0]
        others = [entry["code"] for entry in fatal[1:]]
        raise PairError(
            first["code"],
            first["detail"]
            + (f" (also refused: {others})" if others else "")
            + f"; the record is in <work>/runs/<tag>/lineage.json")
    return document


def _write_lineage_refusal(*, roots: list[dict], args: argparse.Namespace,
                           document: dict) -> None:
    """Best-effort evidence write; a write failure never masks the refusal."""

    try:
        work = pathlib.Path(args.work_dir)
        tag = default_tag(args, roots[0])
        target = assert_writable(work / "runs" / tag / "lineage.json",
                                 work_dir=work, what="lineage refusal record")
        write_json(target, document)
        log(f"lineage refusal recorded: {target}")
    except Exception as error:                                    # noqa: BLE001
        log(f"could not record the lineage refusal: {type(error).__name__}: {error}")


def check_engine_declaration(root: dict) -> str:
    """The manifest must declare the one frozen engine (fail-closed, no default)."""

    label = f"roots[{root['_index']}] {root.get('ancestor_id')!r}"
    engine_declared = root.get("engine_sha256")
    if not isinstance(engine_declared, str) or len(engine_declared) != 64:
        raise PairError("MANIFEST_INVALID",
                        f"{label}: engine_sha256 must be a 64-hex digest")
    if engine_declared != ENGINE_SHA256:
        raise PairError(
            "ENGINE_DECLARED_MISMATCH",
            f"{label}: manifest declares engine {engine_declared[:16]}... but the "
            f"frozen engine of this round is {ENGINE_SHA256[:16]}... "
            f"(v432:185 / v450:123 / v8r0:188 / v8o0:170)")
    return engine_declared


def resolve_pair_rng(root: dict, args: argparse.Namespace,
                     roots: list[dict]) -> tuple[int, str]:
    """``--pair-rng`` must not collide with any other root, and not contradict this one."""

    declared = int(root["pair_rng"])
    if args.pair_rng is None:
        return declared, "manifest"
    value = int(args.pair_rng)
    if value < 0:
        raise PairError("PAIR_RNG_INVALID", f"--pair-rng must be >= 0, got {value}")
    if value != declared:
        holders = [r.get("ancestor_id") for r in roots
                   if int(r.get("pair_rng", -1)) == value]
        raise PairError(
            "PAIR_RNG_CONFLICT",
            f"--pair-rng {value} differs from roots[{root['_index']}]."
            f"pair_rng {declared} (ancestor {root.get('ancestor_id')!r}); the "
            f"pair table is the manifest, so a different seed is a different "
            f"registered pair -- add a root instead of overriding it"
            + (f"; value {value} is already allocated to {holders}" if holders else ""))
    for other in roots:
        if other is root:
            continue
        if int(other.get("pair_rng", -1)) != value:
            continue
        same_cell = (other.get("ancestor_id") == root.get("ancestor_id")
                     and int(other.get("budget_frames", -1)) == int(root["budget_frames"])
                     and other.get("ckpt_sha256") == root.get("ckpt_sha256"))
        raise PairError(
            "PAIR_RNG_REUSED",
            f"pair_rng {value} is already used by roots[{other['_index']}] "
            f"(ancestor {other.get('ancestor_id')!r}); pre-registration sec. 1 "
            f"pre-allocates a unique seed per (ancestor, budget, checkpoint, "
            f"repeat)" + (" -- and that is the same cell" if same_cell else ""))
    return value, "cli"


# --------------------------------------------------------------------------- #
# engine / data source / checkpoints
# --------------------------------------------------------------------------- #
def resolve_engine(args: argparse.Namespace, manifest: dict) -> tuple[str, str]:
    for value, source in ((args.engine, "cli"),
                          (manifest.get("engine"), "manifest"),
                          (ENGINE_DEFAULT, "frozen default")):
        if value:
            return str(value), source
    raise PairError("ENGINE_MISSING", "no engine path anywhere")


def resolve_data_source(args: argparse.Namespace, manifest: dict) -> tuple[str, str]:
    for value, source in ((args.data_source, "cli"),
                          (manifest.get("data_source"), "manifest"),
                          (DATA_SOURCE_DEFAULT, "frozen default")):
        if value:
            return str(value), source
    raise PairError("DATA_SOURCE_MISSING", "no data source path anywhere")


def engine_soft_check(engine: str, declared: str) -> dict:
    """Digest a local engine binary if it is visible; never spawn anything.

    ``mode=check`` must work on a host where the engine lives inside another
    filesystem view (this repo's Windows host cannot see the WSL build), so a
    missing path is recorded as ``gate="deferred-to-run"`` instead of passing
    silently -- ``mode=pair``/``trace`` still refuse ``ENGINE_MISSING``.
    """

    path = pathlib.Path(engine)
    if not path.is_file():
        return {"path": str(engine), "present": False, "gate": "deferred-to-run",
                "declared": declared, "actual": None, "match": None,
                "note": ("the engine binary is not visible from this host; the "
                         "digest gate stays fail-closed in mode=pair/trace, "
                         "which refuses ENGINE_MISSING")}
    actual = sha256_file(path)
    return {"path": str(engine), "present": True, "gate": "checked",
            "declared": declared, "actual": actual,
            "bytes": path.stat().st_size, "match": bool(actual == declared)}


def resolve_ckpt(root: dict, args: argparse.Namespace) -> tuple[pathlib.Path | None,
                                                                str | None]:
    """``--ckpt``/``--ckpt-sha256`` are a pair; the digest must be the root's."""

    wanted = root.get("ckpt_sha256")
    if args.ckpt is None:
        if args.ckpt_sha256 is not None:
            raise PairError("CKPT_PAIR_INCOMPLETE",
                            "--ckpt-sha256 without --ckpt: the two are a pair")
        fallback = root.get("ckpt_path")
        if not fallback:
            return None, None
        path = repo_path(fallback)
    else:
        if args.ckpt_sha256 is None:
            raise PairError("CKPT_PAIR_INCOMPLETE",
                            "--ckpt without --ckpt-sha256: the two are a pair, "
                            "and the digest must equal roots[i].ckpt_sha256")
        path = repo_path(args.ckpt)
        if str(args.ckpt_sha256).lower() != str(wanted).lower():
            raise PairError(
                "CKPT_SHA_MISMATCH",
                f"--ckpt-sha256 {str(args.ckpt_sha256)[:16]}... != roots["
                f"{root['_index']}].ckpt_sha256 {str(wanted)[:16]}...; the "
                f"manifest binds one checkpoint per root (MAP sec. 9-5)")
    if not path.is_file():
        raise PairError("CKPT_MISSING", f"checkpoint not found: {path}")
    actual = sha256_file(path)
    if actual != str(wanted).lower():
        raise PairError(
            "CKPT_SHA_MISMATCH",
            f"{path} sha256 {actual[:16]}... != roots[{root['_index']}]."
            f"ckpt_sha256 {str(wanted)[:16]}...; refusing to evaluate a "
            f"checkpoint the manifest did not register")
    return path, actual


def repo_path(value: object) -> pathlib.Path:
    """Resolve a path from the manifest/CLI; relative ones are repo-relative."""

    path = pathlib.Path(str(value)).expanduser()
    return path if path.is_absolute() else (REPO / path)


def s1c_recording_path(window: int) -> pathlib.Path:
    """The prefix recording v431_dense_driver reads (MAP sec. 1.2)."""

    return REPO / ".tmp_v410_s1c" / "d1" / "shards" / (
        f"d1-s0d1-n3300-seed{int(window)}-shard") / "labels.jsonl"


# --------------------------------------------------------------------------- #
# optional prefix tape: any legal 4-frame recording replaces the S1c shard
# --------------------------------------------------------------------------- #
def tape_path(value: object) -> pathlib.Path:
    """Resolve a declared tape path: repo-relative, host-absolute, WSL ``/mnt/<d>/``.

    `v431_dense_driver.build_masks` can only name a repo-relative S1c shard
    (`v431:65`), so the S1c path needs no resolver.  A tape comes from wherever the
    research side approved it, and the recordings themselves carry the WSL form in
    ``obs.shard_dir`` (e.g. ``/mnt/d/WorkShop/Touhou_RL/.tmp_v400_dagger2/r2/
    shards/dagger-s0d1seed3006``), so both spellings of the *same* file must
    resolve to the same path.  Nothing is guessed: a spelling this host cannot map
    stays unresolved and `resolve_prefix_tape` reports ``PREFIX_TAPE_MISSING``.
    """

    text = str(value).strip()
    if sys.platform == "win32" and text.startswith("/mnt/") and len(text) > 7:
        head, rest = text[5:6], text[7:]
        if head.isalpha() and rest:
            return pathlib.Path(f"{head.upper()}:\\" + rest.replace("/", "\\"))
    path = pathlib.Path(text).expanduser()
    if path.is_absolute():
        return path
    if text.startswith(("/", "\\")):
        # rooted but driveless on Windows: returned as-is, never re-rooted under
        # REPO (which would silently point at a different file)
        return path
    return REPO / path


def scan_tape(path: pathlib.Path) -> dict:
    """Rows / ``frames_built`` of a declared prefix tape, counted, never imported.

    The lenient pass ``mode=check`` needs: it derives the same span
    `read_tape`/`build_masks` do (``frames_built = 4 * rows``, one anchor every 4
    frames) and lists every invariant break in ``integrity`` **without** refusing,
    so a plan can still be printed for a tape a run would reject.  The strict
    reader a run uses is `read_tape`; both encode the same four rules.
    """

    path = pathlib.Path(path)
    entry: dict = {
        "path": str(path), "present": False, "rows": None, "frames_built": None,
        "gate": "deferred-to-run", "integrity": [], "bytes": None, "sha256": None,
        "scope": ("prefix tape: 4 step_frames per row, one hash anchor every 4 "
                  "frames; the same span build_masks derives, counted without "
                  "importing it"),
    }
    if not path.is_file():
        entry["note"] = ("the declared prefix_tape is not visible from this host; "
                         "mode=pair/trace refuses PREFIX_TAPE_MISSING")
        return entry
    try:
        stat = path.stat()
    except OSError as error:
        entry["note"] = f"cannot stat {path}: {error}"
        return entry
    key = f"tape-scan:{path}:{stat.st_size}:{stat.st_mtime_ns}"
    cached = _CACHE.get(key)
    if isinstance(cached, dict):
        return dict(cached)
    rows = 0
    integrity: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    integrity.append(f"row {index}: not JSON")
                    continue
                if not isinstance(row, dict):
                    integrity.append(f"row {index}: not a JSON object")
                    continue
                frame = row.get("decision_frame")
                if isinstance(frame, bool) or not isinstance(frame, int) \
                        or frame != rows * ANCHOR_STRIDE:
                    integrity.append(
                        f"row {index}: decision_frame {frame!r} != "
                        f"{rows * ANCHOR_STRIDE} (read_tape needs one contiguous "
                        f"{ANCHOR_STRIDE}-frame block per row, starting at 0)")
                if len(row.get("step_frames") or []) != ANCHOR_STRIDE:
                    integrity.append(
                        f"row {index}: {len(row.get('step_frames') or [])} "
                        f"step_frames, expected {ANCHOR_STRIDE}")
                if not row.get("hash_pre") or not row.get("hash_post"):
                    integrity.append(f"row {index}: hash_pre/hash_post missing")
                rows += 1
    except OSError as error:
        entry["note"] = f"cannot read {path}: {error}"
        return entry
    entry.update({"present": True, "rows": rows,
                  "frames_built": rows * ANCHOR_STRIDE, "gate": "checked",
                  "integrity": integrity, "bytes": stat.st_size,
                  "sha256": recording_sha256(path)})
    _CACHE[key] = dict(entry)
    return entry


def resolve_prefix_tape(root: dict, args: argparse.Namespace) -> dict | None:
    """Resolve + verify the root's **optional** ``prefix_tape``; ``None`` if absent.

    Same contract shape as ``--pair-rng`` / ``--takeover-frame``:

    * the manifest is the run table -- a ``--prefix-tape`` that differs from
      ``roots[i].prefix_tape`` is ``PREFIX_TAPE_CLI_MISMATCH``, and a
      ``--prefix-tape`` on a root that declares none is
      ``PREFIX_TAPE_NOT_IN_MANIFEST``: the flag may *confirm* the registered
      source, never introduce one nobody registered;
    * a declared tape this host cannot read is ``PREFIX_TAPE_MISSING`` (exit 2 --
      a missing input, not a verdict on the manifest);
    * identity is bound once: ``sha256(tape)`` must equal ``parent_prefix_digest``
      (the value the guard already proved equal to
      ``verify_anchor.prefix_digest``), ``prefix_tape_sha256`` when declared must
      equal the same value, and ``verify_anchor.window_recording_sha256`` when
      present must too.  The
      ``parent_prefix_digest != prefix_tape_sha256`` case is
      ``PREFIX_TAPE_DIGEST_MISMATCH`` -- the tape the manifest names and the
      digest the manifest pins must be one file, or the cell is a different cell.

    Without the field all of this is skipped and `prefix_digest_gate` (the S1c
    gate) stays the only digest authority, which is what keeps pre-tape cells
    byte-identical.
    """

    label = f"roots[{root.get('_index')}] {root.get('ancestor_id')!r}"
    declared = root.get("prefix_tape")
    cli = getattr(args, "prefix_tape", None)
    if declared is None:
        if cli is not None:
            raise PairError(
                "PREFIX_TAPE_NOT_IN_MANIFEST",
                f"{label}: --prefix-tape {cli!r} was given but roots["
                f"{root.get('_index')}] declares no 'prefix_tape'. The manifest is "
                f"the run table: this flag may only *confirm* the tape a root "
                f"already names, so a source nobody registered cannot be run by "
                f"adding a flag (same rule as --pair-rng / --takeover-frame)")
        return None
    if not isinstance(declared, str) or not declared.strip():
        raise PairError(
            "PREFIX_TAPE_INVALID",
            f"{label}: prefix_tape must be a non-empty string (repo-relative path, "
            f"host-absolute path, or the WSL /mnt/<drive>/... form), got "
            f"{declared!r}")
    path = tape_path(declared)
    source = "manifest"
    if cli is not None:
        if _norm_path(tape_path(cli)) != _norm_path(path):
            raise PairError(
                "PREFIX_TAPE_CLI_MISMATCH",
                f"{label}: --prefix-tape {cli!r} (-> {tape_path(cli)}) is not the "
                f"tape the root declares ({declared!r} -> {path}); a different "
                f"recording is a different registered cell, so add a root instead "
                f"of overriding the source")
        source = "cli"
    if not path.is_file():
        raise PairError(
            "PREFIX_TAPE_MISSING",
            f"{label}: the declared prefix_tape {declared!r} resolves to {path}, "
            f"which is not a readable file on this host. A tape is an input, not a "
            f"finding: refusing as 'cannot evaluate' (exit 2) rather than running "
            f"another recording's prefix")
    actual = recording_sha256(path)
    size = path.stat().st_size
    declared_sha = root.get("prefix_tape_sha256")
    parent = root.get("parent_prefix_digest")
    anchor = root.get("verify_anchor")
    anchor_sha = (anchor.get("window_recording_sha256")
                  if isinstance(anchor, dict) else None)
    if declared_sha is not None and (not isinstance(declared_sha, str)
                                     or len(declared_sha) != 64):
        raise PairError(
            "PREFIX_TAPE_INVALID",
            f"{label}: prefix_tape_sha256 must be a bare 64-character hex digest, "
            f"got {declared_sha!r}")
    problems: list[tuple[str, str]] = []
    if declared_sha is not None and isinstance(parent, str) \
            and str(declared_sha).lower() != parent.lower():
        problems.append((
            "prefix_tape_sha256 != parent_prefix_digest",
            f"the manifest pins two different tapes: prefix_tape_sha256 "
            f"{declared_sha} but parent_prefix_digest {parent}"))
    if isinstance(parent, str) and parent.lower() != actual:
        problems.append((
            "sha256(prefix_tape) != parent_prefix_digest",
            f"sha256({path}) is {actual} ({size} B) but parent_prefix_digest is "
            f"{parent}"))
    if declared_sha is not None and str(declared_sha).lower() != actual:
        problems.append((
            "sha256(prefix_tape) != prefix_tape_sha256",
            f"sha256({path}) is {actual} ({size} B) but prefix_tape_sha256 is "
            f"{declared_sha}"))
    if anchor_sha is not None and str(anchor_sha).lower() != actual:
        problems.append((
            "sha256(prefix_tape) != verify_anchor.window_recording_sha256",
            f"sha256({path}) is {actual} ({size} B) but "
            f"verify_anchor.window_recording_sha256 is {anchor_sha}"))
    if problems:
        raise PairError(
            "PREFIX_TAPE_DIGEST_MISMATCH",
            f"{label}: " + " | ".join(f"[{name}] {text}" for name, text in problems)
            + ". A tape's digest is its identity: parent_prefix_digest, "
              "prefix_tape_sha256 and verify_anchor must all name the file that is "
              "on disk (a wrong digest would replay one recording under another "
              "recording's name)")
    return {"path": str(path), "declared": declared, "source": source,
            "sha256": actual, "bytes": size,
            "declared_sha256": declared_sha, "parent_prefix_digest": parent,
            "anchor_declared": anchor_sha, "gate": "checked",
            "scope": ("sha256 of the whole prefix_tape labels.jsonl; the tape is "
                      "the prefix, so its digest is the parent prefix digest")}


def read_tape(path: object, target: int) -> tuple[list[int], dict[int, str], dict]:
    """Read a 4-frame prefix tape exactly as ``build_masks`` would, but stricter.

    `v431_dense_driver.build_masks` (`:64-99`) is the semantics being reproduced:
    ``masks`` is the concatenation of every row's 4 ``step_frames`` (never
    truncated at ``target`` -- a ``recorded`` policy keeps following the tape),
    ``hashes[decision_frame]`` is ``hash_pre`` and ``hashes[decision_frame + 4]``
    is ``hash_post``, so every key is a multiple of 4.

    The four extra refusals are what makes an *arbitrary* tape safe:

    1. each row carries exactly 4 ``step_frames`` (build_masks: same);
    2. ``decision_frame`` must be ``rows_so_far * 4`` -- i.e. start at 0, advance
       by 4, no gap, no overlap, no duplicate (build_masks refuses gaps/overlaps
       with a message; here it is one code and the offending row is named);
    3. ``hash_pre`` and ``hash_post`` must be present on every row, and the anchor
       set must be complete (``frames_built // 4 + 1`` keys, every multiple of 4);
    4. ``target`` must be a multiple of 4 (an anchor) *and* recorded, *and*
       ``target <= frames_built`` (build_masks: `:88-89` plus its coverage check).

    Returns ``(masks, hashes, meta)``; ``meta`` carries at least ``path``,
    ``sha256``, ``rows``, ``frames_built``, ``recorded_hash_frames``, ``target``
    and ``anchors_complete``, so ``run.json`` can prove which file was replayed.
    """

    tape = tape_path(path)
    target = int(target)
    label = str(tape)
    if target % ANCHOR_STRIDE != 0:
        raise PairError(
            "PREFIX_TAPE_INVALID",
            f"{label}: target {target} is not a {ANCHOR_STRIDE}-frame anchor. "
            f"v450.masks_for passes target = N + 1, so N must be 4k - 1 and the "
            f"anchor set is {{0, 4, 8, ...}}")
    if not tape.is_file():
        raise PairError(
            "PREFIX_TAPE_MISSING",
            f"{label}: not a readable file on this host, so neither the masks nor "
            f"target {target} can be checked")
    digest = recording_sha256(tape)
    masks: list[int] = []
    hashes: dict[int, str] = {}
    rows = 0
    try:
        handle = tape.open("r", encoding="utf-8")
    except OSError as error:
        raise PairError("PREFIX_TAPE_MISSING",
                        f"{label}: cannot open ({error.strerror or error})") from error
    with handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise PairError(
                    "PREFIX_TAPE_INVALID",
                    f"{label}: row {index} is not JSON ({error}); read_tape cannot "
                    f"tell which frames it stands for") from error
            if not isinstance(row, dict):
                raise PairError("PREFIX_TAPE_INVALID",
                                f"{label}: row {index} is not a JSON object")
            frame = row.get("decision_frame")
            if isinstance(frame, bool) or not isinstance(frame, int):
                raise PairError(
                    "PREFIX_TAPE_INVALID",
                    f"{label}: row {index} has decision_frame {frame!r}, which is "
                    f"not an integer (expected {rows * ANCHOR_STRIDE})")
            if frame != rows * ANCHOR_STRIDE:
                raise PairError(
                    "PREFIX_TAPE_INVALID",
                    f"{label}: row {index} has decision_frame {frame} but this "
                    f"reader expects {rows * ANCHOR_STRIDE} (contiguous 4-frame "
                    f"blocks from 0: a gap, an overlap or a duplicate is refused "
                    f"rather than replayed out of order)")
            steps = row.get("step_frames")
            steps = steps if isinstance(steps, list) else []
            if len(steps) != ANCHOR_STRIDE:
                raise PairError(
                    "PREFIX_TAPE_INVALID",
                    f"{label}: row {index} (frame {frame}) carries "
                    f"{len(steps)} step_frames, expected {ANCHOR_STRIDE}")
            try:
                ints = [int(value) for value in steps]
            except (TypeError, ValueError) as error:
                raise PairError(
                    "PREFIX_TAPE_INVALID",
                    f"{label}: row {index} (frame {frame}) has non-integer "
                    f"step_frames {steps!r}: {error}") from error
            if not row.get("hash_pre") or not row.get("hash_post"):
                raise PairError(
                    "PREFIX_TAPE_INVALID",
                    f"{label}: row {index} (frame {frame}) is missing "
                    f"hash_pre/hash_post; walk_prefix compares hashes at every "
                    f"anchor, so a row without them cannot be walked")
            masks.extend(ints)
            hashes[frame] = str(row["hash_pre"])
            hashes[frame + ANCHOR_STRIDE] = str(row["hash_post"])
            rows += 1
    frames_built = len(masks)
    if frames_built != rows * ANCHOR_STRIDE:
        raise PairError(
            "PREFIX_TAPE_INVALID",
            f"{label}: built {frames_built} masks from {rows} rows; the 4-frames-"
            f"per-row invariant is broken")
    recorded_hash_frames = len(hashes)
    anchors_complete = bool(
        frames_built % ANCHOR_STRIDE == 0
        and recorded_hash_frames == frames_built // ANCHOR_STRIDE + 1
        and all(frame in hashes
                for frame in range(0, frames_built + 1, ANCHOR_STRIDE)))
    if not anchors_complete:
        missing = [frame for frame in range(0, frames_built + 1, ANCHOR_STRIDE)
                   if frame not in hashes][:8]
        raise PairError(
            "PREFIX_TAPE_INVALID",
            f"{label}: the anchor set is incomplete: {recorded_hash_frames} "
            f"recorded hash frames for {frames_built} frames, expected "
            f"{frames_built // ANCHOR_STRIDE + 1}; first missing anchors {missing}. "
            f"walk_prefix counts hash hits and refuses unless every anchor of the "
            f"prefix is recorded")
    if target > frames_built or target not in hashes:
        raise PairError(
            "PREFIX_TAPE_TARGET_MISSING",
            f"{label}: target {target} is not a recorded anchor of this tape "
            f"({frames_built} frames = {rows} rows x {ANCHOR_STRIDE}, "
            f"{recorded_hash_frames} recorded hash frames, "
            f"target {target} <= frames_built "
            f"{target <= frames_built}); build_masks refuses a target with no hash "
            f"(:88-89) and masks_for would pass N + 1")
    meta = {
        "path": str(tape), "sha256": digest, "bytes": tape.stat().st_size,
        "rows": rows, "frames_built": frames_built,
        "recorded_hash_frames": recorded_hash_frames, "target": target,
        "anchors_complete": anchors_complete, "anchor_stride": ANCHOR_STRIDE,
        "source_kind": "prefix_tape", "masks_len": len(masks),
        "scope": ("read by v829_pair.read_tape: the same (masks, hashes) "
                  "build_masks would build from this file, plus the four strict "
                  "checks"),
        "note": ("v450.masks_for is re-pointed at this tuple in this process only; "
                 "v431_dense_driver.build_masks (and the S1c shard it hard-codes) "
                 "is not used for this root"),
    }
    return masks, hashes, meta


def tape_meta_gate(root: dict, meta: dict) -> dict:
    """Cross-check the optional ``prefix_tape_rows`` / ``prefix_tape_frames``.

    Both are optional; when declared they must match what `read_tape` counted, so
    a manifest cannot round ``frames_built`` or disagree with its own tape.
    """

    label = f"roots[{root.get('_index')}] {root.get('ancestor_id')!r}"
    entry = {"rows_declared": root.get("prefix_tape_rows"),
             "rows": int(meta["rows"]),
             "frames_declared": root.get("prefix_tape_frames"),
             "frames_built": int(meta["frames_built"]), "match": True}
    problems: list[tuple[str, str]] = []
    declared_rows = root.get("prefix_tape_rows")
    if declared_rows is not None:
        if isinstance(declared_rows, bool) or not isinstance(declared_rows, int):
            problems.append(("PREFIX_TAPE_INVALID",
                             f"prefix_tape_rows must be an integer, got "
                             f"{declared_rows!r}"))
        elif int(declared_rows) != int(meta["rows"]):
            problems.append((
                "PREFIX_TAPE_ROW_COUNT_MISMATCH",
                f"prefix_tape_rows {declared_rows} != the {meta['rows']} rows read "
                f"from {meta['path']}"))
    declared_frames = root.get("prefix_tape_frames")
    if declared_frames is not None:
        if isinstance(declared_frames, bool) or not isinstance(declared_frames, int):
            problems.append(("PREFIX_TAPE_INVALID",
                             f"prefix_tape_frames must be an integer, got "
                             f"{declared_frames!r}"))
        elif int(declared_frames) != int(meta["frames_built"]):
            problems.append((
                "PREFIX_TAPE_FRAME_COUNT_MISMATCH",
                f"prefix_tape_frames {declared_frames} != frames_built "
                f"{meta['frames_built']} (= {meta['rows']} rows x "
                f"{ANCHOR_STRIDE})"))
    if problems:
        raise PairError(problems[0][0],
                        f"{label}: " + " | ".join(f"[{name}] {text}"
                                                  for name, text in problems))
    return entry


def prefix_gate(root: dict) -> dict:
    """The digest gate of this root: the tape's when declared, else the S1c one."""

    resolve = root.get("_prefix_tape")
    if not resolve:
        return prefix_digest_gate(root)
    state = root.get("_prefix_tape_state") or {}
    meta = dict(state.get("meta") or {})
    return {
        "window": int(root["window"]),
        "recording": resolve["path"],
        "declared": resolve.get("parent_prefix_digest"),
        "anchor_declared": resolve.get("anchor_declared"),
        "actual": resolve.get("sha256"), "bytes": resolve.get("bytes"),
        "match": True, "anchor_match": None, "gate": "checked",
        "tape": True, "path": resolve["path"], "sha256": resolve.get("sha256"),
        "rows": meta.get("rows"), "frames_built": meta.get("frames_built"),
        "target": meta.get("target"),
        "declared_sha256": resolve.get("declared_sha256"),
        "source": resolve.get("source"),
        "scope": resolve.get("scope"),
    }


def _tape_masks_for(masks: list[int], hashes: dict[int, str], meta: dict):
    """Build the ``masks_for`` replacement that hands `start_parent` the tape."""

    def masks_for(window: int) -> tuple[list[int], dict[int, str], dict]:
        # fresh containers per call, exactly like build_masks: start_parent keeps
        # the reference and walk_prefix reads it once, but a shared mutable list
        # would make a second root's PARK depend on the first one's
        return list(masks), dict(hashes), dict(meta)

    masks_for.__name__ = "masks_for"
    masks_for.__qualname__ = "masks_for"
    masks_for.__doc__ = (
        "v829 prefix-tape injection: returns the (masks, hashes, meta) read by "
        "v829_pair.read_tape(); the pinned v450.masks_for would have called "
        "v431_dense_driver.build_masks on the S1c shard instead (v450:425-427)")
    masks_for.__v829_prefix_tape__ = True                        # type: ignore[attr-defined]
    return masks_for


def prefix_tape_module_objects() -> list[dict]:
    """Every in-process module object that could carry a ``masks_for`` reference.

    Same reachable graph `prefix_module_objects` walks (``v829``'s own copies plus
    every copy v450/v830 load for themselves), because `start_parent` resolves
    ``masks_for`` as a *module global* of whichever copy of v450 it runs in.
    """

    roots = [("v829_pair.m450()", m450()), ("v829_pair.m432()", m432()),
             ("v829_pair.m830()", m830())]
    m450().m432()          # v450._M432 is lazy; force it before it is inspected
    m830().v450()
    m830().m432()
    by_path = {_norm_path(REPO / rel): (name, rel) for name, rel, _globals in PREFIX_TARGETS}
    found: list[dict] = []
    seen: set[int] = set()

    def walk(module, via: str) -> None:
        if module is None or id(module) in seen:
            return
        seen.add(id(module))
        info = by_path.get(_norm_path(getattr(module, "__file__", "")))
        if info is not None:
            found.append({"module": module, "name": info[0], "file": info[1],
                          "role": via, "object_id": id(module),
                          "has_masks_for": callable(
                              getattr(module, "masks_for", None))})
        for attr in ("_M432", "_M450", "_M830"):
            walk(getattr(module, attr, None), f"{via}.{attr}")

    for via, module in roots:
        walk(module, via)
    return found


def inject_prefix_tape(root: dict, *, mode: str = "applied") -> dict:
    """Point this process's ``masks_for`` at the tape's ``(masks, hashes, meta)``.

    `v450_cf_label.start_parent` gets its prefix from the module-global
    ``masks_for(window)`` (`v450:425-427`, `:450`), which `build_masks` answers
    from the S1c shard `v431:64-99` hard-codes.  The tape is injected the way a
    non-frozen takeover is (`apply_prefix_override`): only the module objects this
    process loaded by path are assigned to, and the three pinned files are
    sha256-compared before and after the assignment
    (``PREFIX_TAPE_TOUCHED_DISK`` if a single byte moved), so "the pin is
    untouched" is measured rather than promised.
    """

    state = root.get("_prefix_tape_state") or {}
    meta = dict(state.get("meta") or {})
    masks = list(state.get("masks") or [])
    hashes = dict(state.get("hashes") or {})
    if not masks or not hashes or not meta:
        raise PairError(
            "PREFIX_TAPE_INJECT_FAILED",
            f"roots[{root.get('_index')}]: the tape was not read before injection "
            f"(check_root stores (masks, hashes, meta) on the root); refusing "
            f"rather than letting build_masks read the S1c shard")
    label = f"roots[{root.get('_index')}] {root.get('ancestor_id')!r}"
    before = pinned_file_state()
    injected: list[str] = []
    checked: list[dict] = []
    for entry in prefix_tape_module_objects():
        checked.append({"module": entry["name"], "file": entry["file"],
                        "role": entry["role"], "object_id": entry["object_id"],
                        "has_masks_for": bool(entry["has_masks_for"]),
                        "patched": False})
        if not entry["has_masks_for"]:
            continue
        module = entry["module"]
        replacement = _tape_masks_for(masks, hashes, meta)
        setattr(module, "masks_for", replacement)
        if getattr(module, "masks_for", None) is not replacement:
            raise PairError(
                "PREFIX_TAPE_INJECT_FAILED",
                f"{entry['file']} (loaded by {entry['role']}) did not keep the "
                f"injected masks_for")
        sample_masks, sample_hashes, sample_meta = replacement(int(root["window"]))
        if (len(sample_masks) != len(masks)
                or len(sample_hashes) != len(hashes)
                or str(sample_meta.get("path")) != str(meta.get("path"))
                or int(sample_meta.get("target")) != int(meta.get("target"))):
            raise PairError(
                "PREFIX_TAPE_INJECT_FAILED",
                f"{entry['file']} (loaded by {entry['role']}) reads back as "
                f"{len(sample_masks)} masks / {len(sample_hashes)} hashes / "
                f"target {sample_meta.get('target')}, expected {len(masks)} / "
                f"{len(hashes)} / {meta.get('target')}")
        checked[-1].update({"patched": True,
                            "read_back": {"masks": len(sample_masks),
                                          "hashes": len(sample_hashes),
                                          "target": sample_meta.get("target"),
                                          "path": sample_meta.get("path")}})
        injected.append(
            f"{entry['name']}.masks_for@{entry['file']} (loaded by {entry['role']}, "
            f"object id {entry['object_id']})")
    if not injected:
        raise PairError(
            "PREFIX_TAPE_INJECT_NO_TARGET",
            f"{label}: no in-process copy of v450_cf_label exposes masks_for, so "
            f"the tape cannot be injected; refusing rather than letting "
            f"start_parent call build_masks on the S1c shard")
    after = pinned_file_state()
    if before != after:
        raise PairError(
            "PREFIX_TAPE_TOUCHED_DISK",
            f"{label}: the sha256 of a pinned file changed while masks_for was "
            f"being re-pointed ({before} -> {after}); the injection must be "
            f"memory-only")
    return prefix_tape_block(root=root, meta=meta, mode=mode,
                             injected_into=injected, checked_targets=checked,
                             before=before, after=after)


def prefix_tape_block(*, root: dict, meta: dict, mode: str,
                      injected_into: list[str] | None = None,
                      checked_targets: list[dict] | None = None,
                      before: dict | None = None, after: dict | None = None,
                      cross_check: dict | None = None) -> dict:
    """The ``prefix_tape`` block: the same keys in the plan and in ``run.json``."""

    before = dict(before or {})
    after = dict(after or before)
    resolve = root.get("_prefix_tape") or {}
    takeover = root.get("_takeover_resolution") or {}
    takeover_frame = int(takeover.get("takeover_frame", root["takeover_frame"]))
    target = int(meta.get("target", takeover.get("target_frame", takeover_frame + 1)))
    return {
        # -- the seven keys the acceptance names, in this order ------------------
        "path": meta.get("path"), "sha256": meta.get("sha256"),
        "rows": meta.get("rows"), "frames_built": meta.get("frames_built"),
        "target": target, "injected_into": list(injected_into or []),
        "files_on_disk_unmodified": bool(before == after),
        # -- context (additive: nothing consumed these before) -------------------
        "mode": mode, "applied": bool(mode == "applied"),
        "window": int(root["window"]),
        "takeover_frame": takeover_frame,
        "expected_checkpoints": takeover_frame // ANCHOR_STRIDE,
        "target_frame": takeover_frame + 1,
        "recorded_hash_frames": meta.get("recorded_hash_frames"),
        "anchors_complete": bool(meta.get("anchors_complete")),
        "anchor_stride": ANCHOR_STRIDE,
        "bytes": meta.get("bytes"),
        "masks_len": meta.get("masks_len", meta.get("frames_built")),
        "source": resolve.get("source"),
        "declared": {"prefix_tape": resolve.get("declared"),
                     "prefix_tape_sha256": resolve.get("declared_sha256"),
                     "parent_prefix_digest": resolve.get("parent_prefix_digest")},
        "cross_check": dict(cross_check or {}),
        "checked_targets": list(checked_targets or []),
        "pinned_files": {
            rel: {"path": str(REPO / rel),
                  "sha256": (after.get(rel) or {}).get("sha256"),
                  "bytes": (after.get(rel) or {}).get("bytes"),
                  "sha256_before_injection": (before.get(rel) or {}).get("sha256")}
            for _name, rel, _globals in PREFIX_TARGETS},
        "injection_note": (
            "in-process only: v450_cf_label.masks_for (and every copy that carries "
            "the name) returns read_tape()'s tuple; the three pinned files keep "
            "their bytes on disk and no other process sees this"
            if mode == "applied" else
            "planned only: mode=check imports none of the run-side modules (the "
            "ENGINE_SIDE_IMPORT gate), so pair/trace perform the assignment and "
            "record the object ids in run.json.prefix_tape.injected_into"),
    }


def planned_prefix_tape_block(root: dict) -> dict:
    """The block for ``mode=check``: planned, proven not to have touched disk."""

    state = root.get("_prefix_tape_state") or {}
    before = pinned_file_state()
    return prefix_tape_block(root=root, meta=dict(state.get("meta") or {}),
                             mode="planned", injected_into=[],
                             checked_targets=[], before=before,
                             after=pinned_file_state(),
                             cross_check=dict(state.get("gate") or {}))


# --------------------------------------------------------------------------- #
# takeover frame: fail-closed validation, then an in-process module override
# --------------------------------------------------------------------------- #
def recording_tape(window: int) -> dict:
    """Rows / ``frames_built`` of this window's S1c recording -- no import.

    `v431_dense_driver.build_masks` extends 4 step_frames per row and keys
    ``hashes`` at ``decision_frame`` and ``decision_frame + 4`` (`:72`, `:81-82`),
    so the recorded span is exactly ``4 * rows`` and every anchor is a multiple of
    4.  Counting the rows here reproduces that number **without importing**
    ``v431_dense_driver`` (which would drag nothing dangerous in, but the rule of
    ``mode=check`` is that it imports none of the run-side modules at all).

    The scan is lenient on purpose: a mismatch against `build_masks`' own
    invariants is reported in ``integrity`` and only refused by a mode that would
    actually walk the tape, so ``mode=check`` stays usable as an inspection tool.
    """

    path = s1c_recording_path(window)
    entry: dict = {
        "path": str(path), "present": False, "rows": None, "frames_built": None,
        "gate": "deferred-to-run", "integrity": [], "bytes": None,
        "scope": ("4 step_frames per row, one hash anchor every 4 frames; the same "
                  "span build_masks derives, counted without importing it"),
    }
    if not path.is_file():
        entry["note"] = ("this window's S1c recording is not visible from this "
                         "host, so frames_built cannot be evaluated here; "
                         "mode=pair/trace refuses TAKEOVER_RECORDING_MISSING")
        return entry
    try:
        stat = path.stat()
    except OSError as error:
        entry["note"] = f"cannot stat {path}: {error}"
        return entry
    key = f"recording-tape:{path}:{stat.st_size}:{stat.st_mtime_ns}"
    cached = _CACHE.get(key)
    if isinstance(cached, dict):
        return dict(cached)
    rows = 0
    integrity: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    integrity.append(f"row {index}: not JSON")
                    continue
                if not isinstance(row, dict):
                    integrity.append(f"row {index}: not a JSON object")
                    continue
                frame = row.get("decision_frame")
                if isinstance(frame, bool) or not isinstance(frame, int) \
                        or frame != rows * ANCHOR_STRIDE:
                    integrity.append(
                        f"row {index}: decision_frame {frame!r} != "
                        f"{rows * ANCHOR_STRIDE} (build_masks needs one contiguous "
                        f"{ANCHOR_STRIDE}-frame block per row)")
                if len(row.get("step_frames") or []) != ANCHOR_STRIDE:
                    integrity.append(
                        f"row {index}: {len(row.get('step_frames') or [])} "
                        f"step_frames, expected {ANCHOR_STRIDE}")
                if not row.get("hash_pre") or not row.get("hash_post"):
                    integrity.append(f"row {index}: hash_pre/hash_post missing")
                rows += 1
    except OSError as error:
        entry["note"] = f"cannot read {path}: {error}"
        return entry
    entry.update({"present": True, "rows": rows,
                  "frames_built": rows * ANCHOR_STRIDE, "gate": "checked",
                  "integrity": integrity, "bytes": stat.st_size})
    _CACHE[key] = dict(entry)
    return entry


def pinned_file_state() -> dict:
    """``sha256`` of the three pinned files, read from disk and never imported."""

    state: dict = {}
    for _name, rel, _globals in PREFIX_TARGETS:
        path = REPO / rel
        try:
            state[rel] = {"present": True, "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)}
        except OSError as error:                                  # noqa: PERF203
            state[rel] = {"present": False, "bytes": None, "sha256": None,
                          "error": f"{type(error).__name__}: {error}"}
    return state


def _prefix_globals_of(module_name: str) -> tuple[str, ...]:
    for name, _rel, globals_ in PREFIX_TARGETS:
        if name == module_name:
            return globals_
    return ()


def _override_values(module_name: str, takeover: int) -> dict[str, int]:
    """The exact globals a non-8663 takeover writes into one pinned module."""

    table = {"PREFIX_FRAMES": int(takeover),
             "CHECKPOINTS_IN_PREFIX": int(takeover) // ANCHOR_STRIDE,
             "TARGET_FRAME": int(takeover) + 1}
    return {attr: table[attr] for attr in _prefix_globals_of(module_name)}


def prefix_constants(takeover: int, *, window: int | None = None,
                     source: str = "manifest", mode: str = "planned",
                     overridden: list[str] | None = None,
                     targets: list[dict] | None = None,
                     before: dict | None = None,
                     after: dict | None = None,
                     tape: dict | None = None) -> dict:
    """The ``prefix_constants`` block: same keys in ``run.json`` and in every row.

    ``tape`` is the `scan_tape` result of a root that declares ``prefix_tape``;
    when given, the ``recording`` sub-block describes the tape that actually feeds
    the prefix instead of the window's S1c shard.  ``None`` reproduces the pre-tape
    block exactly (the byte-identical regression depends on it).
    """

    takeover = int(takeover)
    before = dict(before or {})
    after = dict(after or before)
    return {
        # -- the six keys the acceptance names, in this order -------------------
        "takeover_frame": takeover,
        "expected_checkpoints": takeover // ANCHOR_STRIDE,
        "target_frame": takeover + 1,
        "override_applied": bool(mode == "applied"),
        "overridden_modules": list(overridden or []),
        "files_on_disk_unmodified": bool(before == after),
        # -- context (additive: this block is new, nothing consumed it before) --
        "mode": mode,
        "source": source,
        "window": None if window is None else int(window),
        # the pre-v8.30 block was ``{"read_only": True, "values": {...}}``; the two
        # keys are kept, now carrying the *effective* values the pinned modules
        # hold in this process (== the frozen ones whenever no override is active)
        "read_only": bool(mode != "applied"),
        "values": {"v432.PREFIX_FRAMES": takeover,
                   "v432.CHECKPOINTS_IN_PREFIX": takeover // ANCHOR_STRIDE,
                   "v450.PREFIX_FRAMES": takeover,
                   "v450.TARGET_FRAME": takeover + 1,
                   "v830.PREFIX_FRAMES": takeover,
                   "v830.DECISION_FRAMES": DECISION_FRAMES},
        "frozen": {"prefix_frames": PREFIX_FRAMES,
                   "checkpoints_in_prefix": PREFIX_FRAMES // ANCHOR_STRIDE,
                   "target_frame": PREFIX_FRAMES + 1},
        "anchor_stride": ANCHOR_STRIDE,
        "anchor_rule": ("N % 4 == 3: target = N + 1 must be a recorded anchor "
                        "(v431_dense_driver.py:81-82, :88-89)"),
        "targets": list(targets or [
            {"module": name, "file": rel,
             "globals": _override_values(name, takeover)}
            for name, rel, _globals in PREFIX_TARGETS]),
        "pinned_files": {
            rel: {"path": str(REPO / rel),
                  "sha256": (after.get(rel) or {}).get("sha256"),
                  "bytes": (after.get(rel) or {}).get("bytes"),
                  "sha256_before_patch": (before.get(rel) or {}).get("sha256")}
            for _name, rel, _globals in PREFIX_TARGETS},
        "recording": ((tape if tape is not None else recording_tape(window))
                      if window is not None else None),
    }


def planned_prefix_constants(takeover: int, *, window: int | None = None,
                             source: str = "manifest",
                             targets: list[dict] | None = None,
                             tape: dict | None = None) -> dict:
    """The block for a cell whose override is **not** applied in this process.

    ``mode=check`` imports none of the run-side modules, so it can only report
    what would be re-pointed and prove that the files on disk did not move; the
    frozen cell (``takeover == 8663``) is exactly this case even during a run,
    because there is nothing to override.
    """

    before = pinned_file_state()
    mode = "frozen" if int(takeover) == PREFIX_FRAMES else "planned"
    block = prefix_constants(int(takeover), window=window, source=source,
                             mode=mode, overridden=[], targets=targets,
                             before=before, after=pinned_file_state(), tape=tape)
    block["applied_note"] = (
        "no module global was assigned in this process: the frozen cell needs no "
        "override" if mode == "frozen" else
        "mode=check imports none of the run-side modules (the ENGINE_SIDE_IMPORT "
        "gate), so the override is planned here and applied by pair/trace")
    return block


def prefix_module_objects(takeover: int) -> list[dict]:
    """Every **in-process** copy of the three pinned files, nested copies included.

    ``load_module`` here does not register anything in ``sys.modules`` (MAP sec.
    9-12), and each ``tools/rl`` script that needs another one loads *its own*
    copy: ``v450.m432()`` (`v450:198-205`), ``v830.v450()`` / ``v830.m432()``
    (`v830:342-358`).  Patching only the copies this file holds would leave the
    copy that actually runs ``walk_prefix`` (``v450`` -> ``_M432``) at 8663, so the
    reachable graph is walked and every copy is listed.
    """

    roots = [("v829_pair.m432()", m432()), ("v829_pair.m450()", m450()),
             ("v829_pair.m830()", m830())]
    # the nested copies are lazy: force them now so they exist before they are
    # patched (v830.v450() also forces v830._M450._M432 -- v830:349)
    m450().m432()
    m830().v450()
    m830().m432()
    by_path: dict[str, tuple[str, str]] = {}
    for name, rel, _globals in PREFIX_TARGETS:
        by_path[_norm_path(REPO / rel)] = (name, rel)
    found: list[dict] = []
    seen: set[int] = set()

    def walk(module, via: str) -> None:
        if module is None or id(module) in seen:
            return
        seen.add(id(module))
        info = by_path.get(_norm_path(getattr(module, "__file__", "")))
        if info is not None:
            found.append({"module": module, "name": info[0], "file": info[1],
                          "role": via, "object_id": id(module),
                          "globals": _override_values(info[0], takeover)})
        for attr in ("_M432", "_M450", "_M830"):
            walk(getattr(module, attr, None), f"{via}.{attr}")

    for via, module in roots:
        walk(module, via)
    return found


def _norm_path(value: object) -> str:
    try:
        return str(pathlib.Path(str(value)).resolve()).lower()
    except (OSError, ValueError):
        return str(value).lower()


def apply_prefix_override(takeover: int, *, window: int | None = None,
                          source: str = "manifest",
                          tape: dict | None = None) -> dict:
    """Re-point this process's pinned module globals at ``takeover``; return the block.

    Only the module objects ``load_module`` created **in this process** are
    assigned to.  The three files are hashed immediately before and immediately
    after the assignment and the block refuses (``PREFIX_OVERRIDE_TOUCHED_DISK``)
    if a single byte moved, so "the pin is untouched" is measured, not promised.
    """

    takeover = int(takeover)
    before = pinned_file_state()
    if takeover == PREFIX_FRAMES:
        return planned_prefix_constants(takeover, window=window, source=source,
                                        tape=tape)
    objects = prefix_module_objects(takeover)
    if not objects:
        raise PairError(
            "PREFIX_OVERRIDE_NO_TARGET",
            f"takeover_frame {takeover}: no in-process copy of "
            f"{[name for name, _rel, _g in PREFIX_TARGETS]} could be found, so the "
            f"override cannot be applied; refusing rather than running the pinned "
            f"8663 constants against a {takeover}-frame PARK")
    overridden: list[str] = []
    targets: list[dict] = []
    for entry in objects:
        module = entry["module"]
        for attr, value in entry["globals"].items():
            if not hasattr(module, attr):
                raise PairError(
                    "PREFIX_OVERRIDE_NO_TARGET",
                    f"{entry['file']} (loaded by {entry['role']}) has no global "
                    f"{attr}; the pinned module changed shape, so the override "
                    f"cannot be applied")
            setattr(module, attr, int(value))
        values = {attr: getattr(module, attr) for attr in entry["globals"]}
        if values != entry["globals"]:
            raise PairError(
                "PREFIX_OVERRIDE_FAILED",
                f"{entry['file']} (loaded by {entry['role']}) read back as {values} "
                f"after assigning {entry['globals']}")
        label = (f"{entry['name']}@{entry['file']} (loaded by {entry['role']}, "
                 f"object id {entry['object_id']})")
        overridden.append(label)
        targets.append({"module": entry["name"], "file": entry["file"],
                        "role": entry["role"], "object_id": entry["object_id"],
                        "globals": dict(entry["globals"]),
                        "read_back": values, "applied": True})
    after = pinned_file_state()
    if before != after:
        raise PairError(
            "PREFIX_OVERRIDE_TOUCHED_DISK",
            f"takeover_frame {takeover}: the sha256 of a pinned file changed while "
            f"the module globals were being re-pointed ({before} -> {after}); the "
            f"override must be memory-only")
    block = prefix_constants(takeover, window=window, source=source,
                             mode="applied", overridden=overridden,
                             targets=targets, before=before, after=after,
                             tape=tape)
    block["applied_note"] = (
        "in-process only: the three pinned files keep their bytes on disk (sha256 "
        "compared before and after the assignment) and no other process sees this")
    return block


def coerce_takeover_frame(raw: object) -> int | None:
    """``--takeover-frame`` is parsed as text so a non-integer has its own code."""

    if raw is None:
        return None
    try:
        return int(str(raw).strip(), 10)
    except (TypeError, ValueError):
        raise PairError(
            "TAKEOVER_FRAME_NOT_INTEGER",
            f"--takeover-frame {raw!r} is not an integer; the prefix walk needs a "
            f"frame number (every admissible value is 4k - 1, e.g. 8663)") from None


def resolve_takeover_frame(root: dict, args: argparse.Namespace, *,
                           needs_engine: bool = False,
                           tape: dict | None = None) -> dict:
    """Validate ``--takeover-frame`` against the anchor rule, the tape and the root.

    Fail-closed and per rule, in this precedence (the first violation is the
    error code, every violation is spelled out in the detail):

    1. ``TAKEOVER_FRAME_TOO_SMALL``      ``N < 4``
    2. ``TAKEOVER_FRAME_NOT_ANCHOR``     ``N % 4 != 3`` -- ``target = N + 1`` is
       not a recorded anchor, and `masks_for` would hand `build_masks` a target
       it refuses (`v431_dense_driver.py:88-89`)
    3. ``TAKEOVER_BEYOND_RECORDING``     ``N + 1 > frames_built`` of this window
    4. ``TAKEOVER_ROOT_MISMATCH``        ``N != roots[i].takeover_frame``; the
       manifest is the run table, so a new frame is a new root (the same
       philosophy as ``--pair-rng``)

    ``tape`` is the `scan_tape` result of a root that declares ``prefix_tape``.
    When it is given, rules 3 and the ``needs_engine`` refusals are judged against
    **that** file instead of this window's S1c recording -- the tape *is* the
    prefix, so its ``frames_built`` is the bound that matters.  ``None`` keeps the
    pre-tape behaviour and messages word for word, which is what the byte-identical
    regression for the S1c cells rests on.
    """

    label = f"roots[{root['_index']}] {root.get('ancestor_id')!r}"
    declared = int(root["takeover_frame"])
    window = int(root["window"])
    raw = getattr(args, "takeover_frame", None)
    if raw is None:
        value, source = declared, "manifest"
    else:
        value, source = int(raw), "cli"
    problems: list[tuple[str, str]] = []
    if value < MIN_TAKEOVER_FRAME:
        problems.append((
            "TAKEOVER_FRAME_TOO_SMALL",
            f"takeover frame {value} < {MIN_TAKEOVER_FRAME}: the prefix walk needs "
            f"at least {MIN_TAKEOVER_FRAME} frames before the PARK, and target = "
            f"N + 1 = {value + 1} must exist as an anchor"))
    elif value % ANCHOR_STRIDE != TAKEOVER_MOD:
        problems.append((
            "TAKEOVER_FRAME_NOT_ANCHOR",
            f"takeover frame {value} % 4 = {value % ANCHOR_STRIDE}, but a takeover "
            f"frame must be 4k - 1: the recording has one anchor every 4 frames "
            f"(v431_dense_driver.py:81-82) and build_masks refuses a target with no "
            f"hash (:88-89), while v450.masks_for passes target = N + 1 = "
            f"{value + 1}, which is not a multiple of {ANCHOR_STRIDE}"))
    recording = tape if tape is not None else recording_tape(window)
    where = ("this window's recording" if tape is None
             else "the declared prefix_tape")
    if recording["gate"] == "checked":
        frames = int(recording["frames_built"])
        if value + 1 > frames:
            problems.append((
                "TAKEOVER_BEYOND_RECORDING",
                f"takeover frame {value} needs target {value + 1} <= frames_built "
                f"{frames}, but {where} covers {frames} frames "
                f"({recording['rows']} rows x {ANCHOR_STRIDE} frames, "
                f"{recording['path']}); admissible takeover frames for w{window} "
                f"are 4k - 1 <= {frames - 1}"))
    if value != declared:
        problems.append((
            "TAKEOVER_ROOT_MISMATCH",
            f"--takeover-frame {value} differs from roots[{root['_index']}]."
            f"takeover_frame {declared} (ancestor {root.get('ancestor_id')!r}); the "
            f"run table is the manifest, so a different takeover frame is a "
            f"different registered root -- add a root instead of overriding it"))
    if problems:
        raise PairError(
            problems[0][0],
            f"{label}: takeover frame {value} ({source}) refused. "
            + " | ".join(f"[{code}] {text}" for code, text in problems))
    if needs_engine and recording["gate"] != "checked":
        raise PairError(
            "TAKEOVER_RECORDING_MISSING",
            f"{label}: {recording['path']} is not readable from this host, so "
            f"neither frames_built nor target {value + 1} can be checked and "
            f"masks_for cannot build the tape")
    if needs_engine and recording["integrity"]:
        raise PairError(
            "TAKEOVER_RECORDING_MALFORMED",
            f"{label}: {where} breaks build_masks' own invariants "
            f"({len(recording['integrity'])} problem(s), first: "
            f"{recording['integrity'][0]}); build_masks would abort on it, so this "
            f"refuses first")
    return {"takeover_frame": value, "source": source, "declared": declared,
            "override_active": bool(value != PREFIX_FRAMES),
            "expected_checkpoints": value // ANCHOR_STRIDE,
            "target_frame": value + 1, "window": window, "recording": recording,
            "recording_kind": ("prefix_tape" if tape is not None else "s1c"),
            "frozen": bool(value == PREFIX_FRAMES)}


# --------------------------------------------------------------------------- #
# prefix-constant drift guard (v1 never moves them -- MAP sec. 9-1)
# --------------------------------------------------------------------------- #
def assert_prefix_constants() -> dict:
    """Read the frozen constants back **before any override** and refuse on drift.

    This is the on-disk pin gate, and it is deliberately blind to
    ``--takeover-frame``: it runs on the pristine modules first, so a byte edited
    in one of the three pinned files is still ``PREFIX_CONSTANT_DRIFT`` rather
    than being silently masked by a later in-process assignment.
    """

    seen = {"v432.PREFIX_FRAMES": int(m432().PREFIX_FRAMES),
            "v432.CHECKPOINTS_IN_PREFIX": int(m432().CHECKPOINTS_IN_PREFIX),
            "v450.PREFIX_FRAMES": int(m450().PREFIX_FRAMES),
            "v450.TARGET_FRAME": int(m450().TARGET_FRAME),
            "v830.PREFIX_FRAMES": int(m830().PREFIX_FRAMES),
            "v830.DECISION_FRAMES": int(m830().DECISION_FRAMES)}
    problems = []
    for name in ("v432.PREFIX_FRAMES", "v450.PREFIX_FRAMES", "v830.PREFIX_FRAMES"):
        if seen[name] != PREFIX_FRAMES:
            problems.append(f"{name}={seen[name]} != {PREFIX_FRAMES}")
    if seen["v432.CHECKPOINTS_IN_PREFIX"] != PREFIX_FRAMES // 4:
        problems.append(f"v432.CHECKPOINTS_IN_PREFIX="
                        f"{seen['v432.CHECKPOINTS_IN_PREFIX']} != "
                        f"{PREFIX_FRAMES // 4} (one anchor every 4 frames)")
    if seen["v450.TARGET_FRAME"] != PREFIX_FRAMES + 1:
        problems.append(f"v450.TARGET_FRAME={seen['v450.TARGET_FRAME']} != "
                        f"{PREFIX_FRAMES + 1}")
    if seen["v830.DECISION_FRAMES"] != DECISION_FRAMES:
        problems.append(f"v830.DECISION_FRAMES={seen['v830.DECISION_FRAMES']} != "
                        f"{DECISION_FRAMES}")
    if problems:
        raise PairError(
            "PREFIX_CONSTANT_DRIFT",
            "the frozen prefix constants are not the v1 values: "
            + "; ".join(problems)
            + " -- v1 must not patch a module global (MAP sec. 9-1); rerun on the "
              "unmodified tree or get a separate authorisation")
    return {"read_only": True, "values": seen}


# --------------------------------------------------------------------------- #
# the parent proxy: masks on the wire + frame-level bomb decrement (Q4)
# --------------------------------------------------------------------------- #
class TracedChild:
    """Wrap a ``ForkParent`` for ``run_episode36`` without changing either.

    `v830_bombpol.run_episode36` only uses ``spawn`` / ``child_state`` /
    ``child_hazards`` / ``child`` / ``quit_child`` / ``alive`` / ``window`` /
    ``template`` (MAP sec. 4.3), so a proxy is enough:

    * ``child(handle, "STEP <mask> 1")`` -> record the mask that went on the
      wire (the trace's ``masks``) and advance the per-decision frame counter;
    * ``child_state(handle)`` -> record the inventory and, on the first drop
      below the emit's ``bombs_before``, ``emit_decrement_at = frame_index + 1``
      (the exact formula of `v800_bomb.py:724-725`, precedent sec. 4.3(b)).

    Everything else (``PARK``-time invariants, ``HELLO``, ``DEBUG``) is the real
    parent through ``__getattr__``.
    """

    def __init__(self, parent, ctx: dict) -> None:
        self._p = parent
        self._c = ctx

    def child(self, handle: int, command: str) -> str:
        reply = self._p.child(handle, command)
        text = str(command)
        if text.startswith("STEP "):
            ctx = self._c
            ctx["step_count"] = int(ctx.get("step_count", 0)) + 1
            try:
                mask = int(text.split()[1])
            except (IndexError, ValueError):                       # pragma: no cover
                mask = None
            ctx.setdefault("masks", []).append(mask)
        return reply

    def child_state(self, handle: int):
        reading, reply = self._p.child_state(handle)
        ctx = self._c
        try:
            now = int(reading.get("bombs_remaining") or 0)
        except (TypeError, ValueError):                            # pragma: no cover
            now = None
        if now is not None:
            ctx["last_bombs"] = now
            if (ctx.get("emit") and ctx.get("decrement_at") is None
                    and now < int(ctx.get("emit_bombs_before", 0))):
                ctx["decrement_at"] = int(ctx.get("step_count", 0))
        return reading, reply

    def __getattr__(self, name: str):
        return getattr(self._p, name)


def close_decision(ctx: dict, audit: dict) -> None:
    """Finish the pending decision entry once its frames have been executed."""

    pending = ctx.get("pending")
    if pending is None:
        return
    pending["masks"] = list(ctx.get("masks") or [])
    pending["inventory_after"] = ctx.get("last_bombs")
    pending["emit_decrement_at"] = ctx.get("decrement_at")
    audit["decisions"].append(pending)
    if pending.get("bomb_request"):
        audit["decrement_list"].append(ctx.get("decrement_at"))
    ctx["pending"] = None


def make_decide(*, policy, generator, audit: dict, ctx: dict, arm: str,
                budget_decisions: int):
    """``decide(state, hazards, decision) -> (cand, bomb)`` for one arm.

    One closure, one code path: `arm_a.build_features` -> policy -> `candidate_
    sampled` -> `rule_decision`, and the *only* arm-dependent input is
    ``table_for(arm, ...)`` (MAP sec. 2.3).
    """

    import torch

    A = arm_a()
    R8 = m8r0()                                    # shared sampler + rule
    arm = check_arm(arm)

    def decide(state: dict, hazards, decision: int) -> tuple[int, int]:
        close_decision(ctx, audit)
        executed = int(decision)
        bombs_before = int(state["bombs_remaining"] or 0)
        table, urgency, scale = table_for(
            arm, budget_decisions=int(budget_decisions),
            decisions_executed=executed, bombs_left=bombs_before)
        feats, _detail = A.build_features(state=state, hazards=hazards,
                                          decision_index=decision)
        with torch.no_grad():
            logits = policy(torch.from_numpy(feats).unsqueeze(0))[0]
        cand, best, margin, prob = R8.candidate_sampled(logits, generator)
        audit["rng_draws"] = int(audit["rng_draws"]) + 1
        rule = R8.rule_decision(state, hazards, table)
        audit["cand_counts"][int(cand)] += 1

        ctx["pending"] = {
            "decision": executed,
            "frame": int(state["frame"]),
            "bombs_before": bombs_before,
            "threshold_px": (None if rule["threshold_px"] is None
                             else float(rule["threshold_px"])),
            "threshold_table": {str(k): float(v) for k, v in table.items()},
            "nearest_px": (None if rule["nearest_all_px"] is None
                           else float(rule["nearest_all_px"])),
            "cand": int(cand),
            "cand_prob": float(prob),
            "cand_best": float(best),
            "cand_margin": float(margin),
            "action": int(cand) * 2 + int(rule["bomb"]),
            "bomb_request": int(rule["bomb"]),
            "distance_ok": bool(rule["distance_ok"]),
            "urgency": urgency,
            "scale": scale,
            "inventory_before": bombs_before,
        }
        # the proxy state for THIS decision's frames
        ctx["emit"] = bool(rule["bomb"])
        ctx["emit_bombs_before"] = bombs_before
        ctx["decrement_at"] = None
        ctx["step_count"] = 0
        ctx["masks"] = []
        ctx["last_bombs"] = bombs_before
        return int(cand), int(rule["bomb"])

    return decide


# --------------------------------------------------------------------------- #
# one arm on one parked parent
# --------------------------------------------------------------------------- #
def run_one_arm(*, parent, arm: str, pair: dict, policy, root: dict,
                args: argparse.Namespace, episode: int, arm_order: list[str],
                guard_doc: dict, engine: str, engine_source: str,
                engine_gate: dict, ckpt_path: pathlib.Path | None,
                ckpt_sha: str | None, data_dir: pathlib.Path,
                cfg_info: dict, parent_gate: dict,
                constants: dict,
                manifest_closure: dict | None = None,
                attempt: int = 1,
                supersedes: str | None = None) -> tuple[dict, dict | None]:
    """One episode of one arm; returns ``(row, trace_document_or_None)``.

    ``manifest_closure`` is the A block: the digest/size of the bytes this
    invocation actually parsed.  It -- not the guard document -- is what the row
    records, so a row can never name a manifest digest the runner did not use.

    ``attempt`` / ``supersedes`` are the pre-registration sec. 4 re-run keys.  They
    are written into the row **here**, before ``score()`` and before the trace's
    ``row_sha256``, so that digest still describes the row exactly as it is
    persisted (attempt 1 passes the defaults and therefore adds ``attempt: 1``
    only -- no ``supersedes``, because nothing was superseded).
    """

    import torch

    closure = dict(manifest_closure or {})
    manifest_sha256 = closure.get("manifest_sha256", guard_doc.get("manifest_sha256"))
    manifest_bytes = closure.get("manifest_bytes", guard_doc.get("manifest_bytes"))

    budget_frames = int(root["budget_frames"])
    budget_decisions = budget_frames // DECISION_FRAMES
    window = int(root["window"])
    seed = generator_seed(int(pair["pair_rng"]))

    # -- RNG: a brand-new generator per arm per episode (pre-registration sec. 1) --
    generator = torch.Generator()
    generator.manual_seed(seed)
    initial = generator_digest(generator)

    ctx: dict = {"pending": None}
    audit = {"cand_counts": [0] * N_CAND, "rng_draws": 0, "decisions": [],
             "decrement_list": []}
    decide = make_decide(policy=policy, generator=generator, audit=audit,
                         ctx=ctx, arm=arm, budget_decisions=budget_decisions)
    traced = TracedChild(parent, ctx)

    record = m830().run_episode36(
        traced, episode=int(episode), policy_seed=int(pair["policy_seed"]),
        decisions=int(budget_decisions), decide=decide,
        budget=int(budget_decisions), decision_frames=DECISION_FRAMES,
        keep_trace=False)
    close_decision(ctx, audit)                     # the last decision's frames
    final = generator_digest(generator)

    decisions = audit["decisions"]
    first_table = (decisions[0]["threshold_table"] if decisions else None)
    last_table = (decisions[-1]["threshold_table"] if decisions else None)
    label, label_source = rule_label_for(arm)

    alignment = {
        "bomb_emits": int(record.get("bomb_emits") or 0),
        "len_bomb_decisions": len(record.get("bomb_decisions") or []),
        "len_bomb_frames": len(record.get("bomb_frames") or []),
        "len_bombs_remaining_at_emit": len(record.get("bombs_remaining_at_emit") or []),
        "len_bomb_available_at_emit": len(record.get("bomb_available_at_emit") or []),
        "len_emit_decrement_at_list": len(audit["decrement_list"]),
    }
    aligned = (alignment["bomb_emits"] == alignment["len_bomb_decisions"]
               == alignment["len_bomb_frames"]
               == alignment["len_bombs_remaining_at_emit"]
               == alignment["len_bomb_available_at_emit"]
               == alignment["len_emit_decrement_at_list"])
    first_decrement = next((value for value in audit["decrement_list"]
                            if value is not None), None)

    row: dict = {
        "schema": SCHEMA_EPISODE, "kind": "pair-episode", "tool": TOOL,
        "tool_version": TOOL_VERSION,
        # -- identity of this cell -------------------------------------------
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "manifest_bytes": manifest_bytes,
        "manifest_sha256_source": closure.get("manifest_sha256_source"),
        "manifest_reads": closure.get("manifest_reads"),
        "guard": guard_doc.get("guard"),
        "guard_version": guard_doc.get("guard_version"),
        "guard_ok": guard_doc.get("ok"),
        "guard_codes": guard_doc.get("codes"),
        "ancestor_id": root.get("ancestor_id"), "stage": root.get("stage"),
        "window": window, "window_source": root.get("window_source"),
        "root_index": root["_index"],
        "takeover_frame": int(root["takeover_frame"]),
        # the frame this episode's decisions start from; equals
        # prefix_constants.takeover_frame (a mismatch means the pinned modules were
        # not re-pointed, which v830.run_episode36 would refuse at spawn)
        "prefix_frame": int(root["takeover_frame"]),
        "total_frames": int(root["total_frames"]),
        "parent_prefix_digest": root.get("parent_prefix_digest"),
        "verify_anchor": root.get("verify_anchor"),
        # -- the two budget units, side by side (MAP sec. 9-10) ------------------
        "budget_frames": budget_frames,
        "budget_decisions": budget_decisions,
        "decision_frames": DECISION_FRAMES,
        # -- arm --------------------------------------------------------------
        "arm": arm, "rule_label": label, "rule_label_source": label_source,
        "arm_label_excluded": (
            "MAP sec. 9-4: v8r0.arm_label reads getattr(parent,'arm','G2') and no "
            "code sets parent.arm, so it always reports URG-a1.00; this row "
            "writes rule_label + threshold_table instead"),
        "threshold_table": first_table,
        "threshold_table_last": last_table,
        "threshold_table_source": (
            "decision 0 of this episode (G1 rescales per decision; the table "
            "of every decision is in the --keep-trace file)"
            if arm == "G1" else
            "constant triple from v8o0_ruletight.arm_spec('T2') (decision 0)"),
        "pair_id": pair["pair_id"], "pair_rng": int(pair["pair_rng"]),
        "pair_rng_source": pair["pair_rng_source"],
        "episode": int(episode), "episodes_planned": int(args.episodes),
        "arm_order": list(arm_order), "arm_index": int(arm_order.index(arm)),
        # -- RNG (pre-registration sec. 1) ---------------------------------------
        "generator_seed": seed,
        "generator_seed_formula": "pair_rng * 1000 + 7 (GEN_CONST = 7)",
        "generator_creation": ("torch.Generator() created per arm per episode, "
                               "after load_policy36 (v830.policy36 calls "
                               "torch.manual_seed -- MAP sec. 9-2)"),
        "generator_initial": initial, "generator_final": final,
        "generator_initial_digest": initial["digest"],
        "generator_final_digest": final["digest"],
        "rng_draws": int(audit["rng_draws"]),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "device": "cpu",
        "device_note": ("Generator + Policy are CPU-only here: "
                        "v860.load_policy36 loads with map_location='cpu'"),
        "rng_stream_note": (
            f"episode {episode} of {args.episodes}: every episode starts a new "
            f"generator at seed {seed}, so nothing is consumed across episodes; "
            f"a *new replicate* needs a new pair_rng (manifest sec. 1)"),
        # -- checkpoint / engine ---------------------------------------------
        "ckpt_path": None if ckpt_path is None else str(ckpt_path),
        "ckpt_sha256": ckpt_sha, "ckpt_sha256_declared": root.get("ckpt_sha256"),
        "engine_path": str(engine), "engine_source": engine_source,
        "engine_sha256_declared": root.get("engine_sha256"),
        "engine_gate": engine_gate,
        "prefix_constants": constants,
        # -- data dir / audio -------------------------------------------------
        "tag": args.tag, "data_dir": str(data_dir),
        "data_cfg": cfg_info.get("cfg"),
        "audio_layers": {
            "env": "v450_cf_label.prepare_env() -> SDL_AUDIODRIVER=dummy",
            "client": "HarnessClient.__init__ -> SDL_AUDIODRIVER=dummy, "
                      "DISPLAY popped for role=shadow",
            "cfg": "v432.build_private_data_dir -> regenerated 60-byte th08.cfg, "
                   "four fields re-inspected",
        },
        # -- recovery gate ----------------------------------------------------
        "parent_gate": {key: parent_gate.get(key) for key in (
            "window", "pid", "park_reply", "park_frame", "park_ok", "threads_1",
            "hello_role_shadow", "all_pass")},
        "parent_prefix": parent_gate.get("prefix"),
        # -- the frozen loop's own record -------------------------------------
        "survived": record.get("survived"),
        "frames": record.get("frames"),
        "alive_frames": record.get("alive_frames"),
        "decisions_executed": record.get("decisions_executed"),
        "decisions_measured": record.get("decisions_measured"),
        "decisions_cap": record.get("decisions_cap"),
        "return_decisions": record.get("return_decisions"),
        "terminal": record.get("terminal"),
        "first_death_offset": record.get("first_death_offset"),
        "first_death_frame": record.get("first_death_frame"),
        "invalid": record.get("invalid"), "error": record.get("error"),
        "template": record.get("template"), "handle": record.get("handle"),
        "spawn_ms": record.get("spawn_ms"),
        "baseline": record.get("baseline"),
        # -- bomb audit (MAP sec. 4): every quantity the pre-registration names ---
        "bombs_start": record.get("bombs_start"),
        "bombs_end": record.get("bombs_end"),
        "bombs_used": record.get("bombs_used"),
        "bomb_emits": record.get("bomb_emits"),
        "bomb_decisions": record.get("bomb_decisions"),
        "bomb_frames": record.get("bomb_frames"),
        "bombs_remaining_at_emit": record.get("bombs_remaining_at_emit"),
        "bomb_available_at_emit": record.get("bomb_available_at_emit"),
        "emit_decrement_at": first_decrement,
        "emit_decrement_at_list": list(audit["decrement_list"]),
        "emit_decrement_at_note": (
            "frame_index + 1 inside the emitting decision (1 or 2 with "
            "decision_frames=2), None when the inventory never dropped; taken "
            "through the parent proxy, precedent v800_bomb.py:724-725/738"),
        "bomb_ignored": record.get("bomb_ignored"),
        "bomb_cands": record.get("bomb_cands"),
        "nearest_px_at_emit": record.get("nearest_px_at_emit"),
        "min_nearest_px": record.get("min_nearest_px"),
        "actions_sampled": record.get("actions_sampled"),
        "cand_counts": list(audit["cand_counts"]),
        "distinct_cands": int(sum(1 for c in audit["cand_counts"] if c)),
        "trace_alignment": alignment,
        "trace_alignment_ok": bool(aligned),
        "wall_s": record.get("wall_s"),
        "parent_alive": record.get("parent_alive"),
        "keep_trace": bool(args.keep_trace),
    }
    # pre-registration sec. 4: which attempt this row belongs to, and (for a
    # re-run) the sha256 of the attempt-1 row it replaces.  Set before score() and
    # before the trace's row_sha256 so both describe the persisted row.
    row["attempt"] = int(attempt)
    if supersedes:
        row["supersedes"] = str(supersedes)
    row["score"] = score(row)
    trace = None
    if args.keep_trace:
        trace = {
            "schema": SCHEMA_TRACE, "kind": "trace", "tool": TOOL,
            "tool_version": TOOL_VERSION,
            "arm": arm, "rule_label": label, "rule_label_source": label_source,
            "tag": args.tag, "pair_id": pair["pair_id"],
            "pair_rng": int(pair["pair_rng"]), "episode": int(episode),
            "ancestor_id": root.get("ancestor_id"),
            "manifest_sha256": manifest_sha256,
            "manifest_sha256_source": closure.get("manifest_sha256_source"),
            "window": window, "prefix_frame": int(root["takeover_frame"]),
            "decision_frames": DECISION_FRAMES,
            "budget_frames": budget_frames, "budget_decisions": budget_decisions,
            "policy_seed": int(pair["policy_seed"]),
            "generator_seed": seed,
            "generator_initial": initial, "generator_final": final,
            "generator_initial_digest": initial["digest"],
            "rng_draws": int(audit["rng_draws"]),
            "torch_version": str(getattr(torch, "__version__", "unknown")),
            "device": "cpu",
            "ckpt_sha256": ckpt_sha,
            "trace_alignment": alignment,
            "trace_alignment_ok": bool(aligned),
            "decisions_total": len(decisions),
            "row_sha256": hashlib.sha256(
                json.dumps(row, ensure_ascii=False, sort_keys=True,
                           default=str).encode("utf-8")).hexdigest(),
            "decisions": decisions,
        }
    return row, trace


def score(row: dict) -> dict:
    """Pre-registration sec. 4, with the MAP sec. 6.3 acceptance formula spelled out."""

    budget_frames = int(row["budget_frames"])
    budget_decisions = int(row["budget_decisions"])
    full = (row.get("frames") == budget_frames
            and row.get("decisions_executed") == budget_decisions
            and row.get("first_death_offset") is None
            and row.get("terminal") is None)
    return {
        "full_segment": bool(full),
        "legal_death": row.get("terminal") in ("first_death", "game_over"),
        "legal_clear": row.get("terminal") == "stage_clear",
        "ended_before_budget": (None if row.get("terminal") is not None
                                else row.get("decisions_executed") != budget_decisions),
        "technical_invalid": bool(row.get("invalid")),
        "retryable": bool(row.get("invalid")),
        "frames_expected": budget_frames,
        "frames_actual": row.get("frames"),
        "decisions_expected": budget_decisions,
        "decisions_actual": row.get("decisions_executed"),
        "formula": ("frames == budget_frames (= 2 * budget_decisions) and "
                    "decisions_executed == budget_decisions and "
                    "first_death_offset is None and terminal is None "
                    "(MAP sec. 6.3, TRAINING_HOST_REQUEST:69-72)"),
        "stage_clear_note": ("a legal clear is a single event, not a full "
                             "segment (pre-registration sec. 4; MAP sec. 9-7)"),
    }


# --------------------------------------------------------------------------- #
# post-run audio audit (AGENTS.md sec. 9.3 #3)
# --------------------------------------------------------------------------- #
def audio_audit(data_root: pathlib.Path, *, tag: str,
                out_path: pathlib.Path) -> dict:
    """Decode every ``th08.cfg`` under this tag's data root and refuse audible."""

    decode = m_audio().decode
    paths = sorted({path.resolve() for path in pathlib.Path(data_root).rglob("th08.cfg")})
    files: list[dict] = []
    problem_count = 0
    muted_count = 0
    four_zero_count = 0
    for path in paths:
        try:
            decoded = decode(path)
        except OSError as exc:
            decoded = {"path": str(path), "error": str(exc)}
        files.append(decoded)
        if "error" in decoded:
            problem_count += 1
            continue
        if decoded.get("muted"):
            muted_count += 1
        wanted = ("musicMode", "playSounds", "musicVolume", "sfxVolume")
        if all(int(decoded.get(key) or 0) == 0 for key in wanted):
            four_zero_count += 1
    cfg_files = len(paths)
    audible = cfg_files - problem_count - muted_count
    document = {
        "schema": SCHEMA_AUDIO, "tool": TOOL, "tag": str(tag),
        "data_root": str(data_root),
        "scan": f"{data_root}/**/th08.cfg",
        "decode_source": (f"{GUARD_REL.rsplit('/', 1)[0]}/v431_audio_cfg_audit.py"
                          f"::decode(), loaded by path with importlib"),
        "counts": {"cfg_files": cfg_files, "decoded": cfg_files - problem_count,
                   "errors": problem_count, "muted": muted_count,
                   "audible": audible, "four_zero": four_zero_count},
        "criteria": ("cfg_files >= 1 and errors == 0 and muted == cfg_files and "
                     "audible == 0 and four_zero == cfg_files "
                     "(AGENTS.md sec. 9.3: all four fields zero)"),
        "four_zero_note": ("v431 decode()['muted'] checks playSounds/volumes but "
                           "not musicMode; four_zero re-checks all four fields, "
                           "matching v432.build_private_data_dir's "
                           "muted_four_zero"),
        "files": files,
    }
    ok = (cfg_files >= 1 and problem_count == 0 and muted_count == cfg_files
          and audible == 0 and four_zero_count == cfg_files)
    document["ok"] = bool(ok)
    document["outcome"] = "muted" if ok else ("empty" if cfg_files == 0 else "AUDIBLE")
    write_json(out_path, document)
    return document


# --------------------------------------------------------------------------- #
# compare mode (pure stdlib, starts nothing)
# --------------------------------------------------------------------------- #
def load_trace(path: str) -> dict:
    target = pathlib.Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise PairError("TRACE_UNREADABLE",
                        f"cannot read {target}: {exc.strerror or exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PairError("TRACE_UNREADABLE",
                        f"{target} is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise PairError("TRACE_UNREADABLE", f"{target} is not a JSON object")
    if not isinstance(document.get("decisions"), list):
        raise PairError("TRACE_UNREADABLE",
                        f"{target} has no 'decisions' array -- is it a "
                        f"{SCHEMA_TRACE} document written with --keep-trace?")
    return document


def _identity(trace: dict, path: str) -> dict:
    return {
        "path": str(path), "arm": trace.get("arm"),
        "rule_label": trace.get("rule_label"),
        "decisions": len(trace.get("decisions") or []),
        "window": trace.get("window"),
        "prefix_frame": trace.get("prefix_frame"),
        "decision_frames": trace.get("decision_frames"),
        "budget_frames": trace.get("budget_frames"),
        "budget_decisions": trace.get("budget_decisions"),
        "pair_rng": trace.get("pair_rng"),
        "generator_seed": trace.get("generator_seed"),
        "generator_initial_digest": trace.get("generator_initial_digest"),
        "tag": trace.get("tag"), "ancestor_id": trace.get("ancestor_id"),
        "schema": trace.get("schema"),
    }


def compare_traces(path_g1: str, path_t2: str) -> dict:
    """Compare the seven frozen fields decision by decision (docstring sec. )."""

    g1 = load_trace(path_g1)
    t2 = load_trace(path_t2)
    for key in COMPARE_HEADER_KEYS:
        left, right = g1.get(key), t2.get(key)
        if left is not None and right is not None and left != right:
            raise PairError(
                "TRACE_INCOMPATIBLE",
                f"{key}: G1 trace says {left!r}, T2 trace says {right!r}; two "
                f"runs with different start conditions cannot be compared "
                f"decision by decision")
    da = g1["decisions"]
    db = t2["decisions"]
    shared = min(len(da), len(db))
    prefix = 0
    first: dict | None = None
    for index in range(shared):
        left, right = da[index], db[index]
        differing = [(key, name) for key, name in COMPARE_FIELDS
                     if left.get(key) != right.get(key)]
        if differing:
            first = {"index": index, "reason": "field",
                     "fields": [name for _key, name in differing],
                     "detail": [{"field": name, "g1": left.get(key),
                                 "t2": right.get(key)} for key, name in differing]}
            break
        prefix += 1
    if first is None and len(da) != len(db):
        index = shared
        longer = "G1" if len(da) > len(db) else "T2"
        first = {"index": index, "reason": "decision_count",
                 "fields": ["decision_count"],
                 "detail": [{"field": "decision_count", "g1": len(da),
                             "t2": len(db),
                             "note": f"{longer} trace continues past decision "
                                     f"{index - 1}"}]}
    frame = None
    if first is not None:
        for source in (da, db):
            if first["index"] < len(source):
                frame = source[first["index"]].get("frame")
                break
    # The thresholds differ by construction (that *is* the experiment), so the
    # first difference is normally decision 0.  The behaviour-only difference is
    # therefore reported as a clearly separate second answer.
    action = None
    for index in range(shared):
        left, right = da[index], db[index]
        differing = [name for key, name in COMPARE_FIELDS
                     if key in ACTION_FIELDS and left.get(key) != right.get(key)]
        if differing:
            action = {"decision": index, "fields": differing,
                      "frame": left.get("frame", right.get("frame"))}
            break
    if action is None and len(da) != len(db):
        action = {"decision": shared, "fields": ["decision_count"],
                  "frame": (da[shared].get("frame") if shared < len(da) else None)}
    return {
        "schema": SCHEMA_COMPARE, "kind": "compare", "tool": TOOL,
        "tool_version": TOOL_VERSION,
        "compared_fields": [name for _key, name in COMPARE_FIELDS],
        "action_fields": [name for key, name in COMPARE_FIELDS
                          if key in ACTION_FIELDS],
        "compare_scope": ("strict equality of "
                          + ", ".join(name for _key, name in COMPARE_FIELDS)
                          + "; decision indices are 0-based, the same numbering "
                            "as run_episode36's trace[].decision"),
        "g1": _identity(g1, path_g1), "t2": _identity(t2, path_t2),
        "same_prefix_decisions": int(prefix),
        "verdict": "DIFFER" if first is not None else "IDENTICAL",
        "first_diff_decision": None if first is None else int(first["index"]),
        "first_diff_decision_1based": None if first is None else int(first["index"]) + 1,
        "first_diff_frame": frame,
        "first_diff_fields": [] if first is None else first["fields"],
        "first_diff_reason": None if first is None else first["reason"],
        "first_diff_detail": [] if first is None else first["detail"],
        "first_action_difference": action,
        "first_action_difference_note": (
            "secondary answer: first decision where the *behaviour* fields ("
            + ", ".join(name for key, name in COMPARE_FIELDS
                        if key in ACTION_FIELDS)
            + ") differ.  G1 and T2 have different threshold tables by "
              "construction, so first_diff_decision is normally 0 and this is "
              "the number that says when the two arms actually acted "
              "differently"),
        "warnings": ([] if g1.get("arm") != t2.get("arm")
                     else [f"both traces declare arm {g1.get('arm')!r}; a pair "
                           f"should be G1 vs T2"]),
    }


def render_compare(document: dict) -> str:
    lines = [
        f"v829 pair compare: G1 {document['g1']['path']}  vs  T2 {document['t2']['path']}",
        f"  arms        G1={document['g1']['rule_label']}  "
        f"T2={document['t2']['rule_label']}",
        f"  start       prefix_frame={document['g1']['prefix_frame']} "
        f"decision_frames={document['g1']['decision_frames']} "
        f"budget_decisions={document['g1']['budget_decisions']}",
        f"  decisions   G1={document['g1']['decisions']}  "
        f"T2={document['t2']['decisions']}",
        "  compared    " + ", ".join(document["compared_fields"]),
        f"  same prefix length: {document['same_prefix_decisions']}",
    ]
    if document["verdict"] == "IDENTICAL":
        lines.append("VERDICT: IDENTICAL")
    else:
        lines.append(f"  first difference:   decision "
                     f"{document['first_diff_decision']} (0-based)  "
                     f"absolute frame {document['first_diff_frame']}")
        for entry in document["first_diff_detail"]:
            if "note" in entry:
                lines.append(f"    {entry['field']:16s} {entry['note']}")
            else:
                lines.append(f"    {entry['field']:16s} G1={entry['g1']!r}  "
                             f"T2={entry['t2']!r}")
        action = document.get("first_action_difference")
        if action:
            lines.append(f"  first behaviour difference: decision "
                         f"{action['decision']} (0-based)  absolute frame "
                         f"{action['frame']}  fields={action['fields']}")
        else:
            lines.append("  first behaviour difference: none (same actions all "
                         "the way)")
        lines.append("VERDICT: DIFFER")
    for warning in document.get("warnings") or []:
        lines.append(f"  WARNING: {warning}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# plan: what check prints, and what pair/trace then execute
# --------------------------------------------------------------------------- #
def default_tag(args: argparse.Namespace, root: dict) -> str:
    if args.tag:
        return str(args.tag)
    stem = pathlib.Path(args.manifest).stem or "manifest"
    return (f"{stem}-w{int(root['window'])}-b{int(root['budget_frames'])}"
            f"-{str(args.mode)}")


def out_for(args: argparse.Namespace, tag: str) -> pathlib.Path:
    if args.out:
        return pathlib.Path(args.out)
    return pathlib.Path(args.work_dir) / "eval" / f"{tag}.jsonl"


def log_for(args: argparse.Namespace, tag: str) -> pathlib.Path:
    if args.log:
        return pathlib.Path(args.log)
    return pathlib.Path(args.work_dir) / "runs" / tag / "run.log"


def trace_path_for(out: pathlib.Path, arm: str, *, episode: int,
                   episodes: int, attempt: int = 1) -> pathlib.Path:
    """Trace file next to ``--out``; ``attempt >= 2`` gets its own name.

    Attempt 1's name is exactly what it always was (``<stem>.<ARM>[.ep<N>]
    .trace.json``), so a cell that is never re-run keeps the byte-identical name
    the pre-registration's compare step expects.  A re-run (pre-registration
    sec. 4, "keep both records") is written as ``<stem>.<ARM>[.ep<N>].a<N>
    .trace.json`` instead of overwriting attempt 1's file.
    """

    suffix = f".ep{episode}" if int(episodes) > 1 else ""
    extra = "" if int(attempt) <= 1 else f".a{int(attempt)}"
    return out.with_name(f"{out.stem}.{arm}{suffix}{extra}.trace.json")


def quote_argv(argv: list[str]) -> str:
    return " ".join(f'"{item}"' if " " in str(item) else str(item) for item in argv)


def build_argv(*, mode: str, args: argparse.Namespace, root: dict, tag: str,
               out: pathlib.Path, log_path: pathlib.Path, arm: str | None,
               ckpt: pathlib.Path | None, arm_order: list[str],
               takeover: int | None = None,
               extra: list[str] | None = None) -> list[str]:
    """The exact command line that would run this cell.

    ``--takeover-frame`` is appended **only** when this cell overrides the frozen
    8663: the frozen cell's argv stays character for character what it was, and a
    non-frozen cell's argv is complete enough to be pasted and re-run (the flag
    must equal the root's declared value there, so it can never be dropped).

    ``extra`` is appended last and is empty for every pre-existing caller, so the
    argv printed by ``mode=check`` is byte-identical unless a caller opts in (the
    ``--workers`` coordinator passes ``--lineage-config`` so a worker re-derives
    the same exposure verdict).
    """

    argv = [sys.executable, "-B", str(REPO / "tools" / "rl" / "v829_pair.py"),
            "--mode", mode, "--manifest", str(args.manifest),
            "--window", str(int(root["window"])),
            "--budget-frames", str(int(root["budget_frames"])),
            "--episodes", str(int(args.episodes))]
    if takeover is not None:
        argv += ["--takeover-frame", str(int(takeover))]
    tape = root.get("_prefix_tape") or {}
    if tape:
        # printed for every tape cell (the value the manifest declares), because
        # the flag may only ever *equal* roots[i].prefix_tape: a pasted argv must
        # carry the registered source so it cannot be run against another one
        argv += ["--prefix-tape", str(tape.get("declared") or tape.get("path"))]
    if mode == "pair":
        argv += ["--arm-order", ",".join(arm_order)]
    else:
        argv += ["--arm", str(arm)]
    if ckpt is not None:
        argv += ["--ckpt", str(ckpt),
                 "--ckpt-sha256", str(root.get("ckpt_sha256"))]
    argv += ["--pair-rng", str(int(root["pair_rng"])),
             "--work-dir", str(args.work_dir), "--tag", tag,
             "--out", str(out), "--log", str(log_path)]
    if args.allow_exposed:
        argv.append("--allow-exposed")
    argv.append("--allow-legacy-window")
    if args.keep_trace:
        argv.append("--keep-trace")
    if extra:
        argv += [str(item) for item in extra]
    return argv


def preflight(args: argparse.Namespace, *, guard_doc: dict, manifest: dict,
              roots: list[dict], needs_engine: bool,
              closure: dict | None = None,
              selection: dict | None = None) -> dict:
    """Every check that needs no engine; returns the plan document.

    ``closure`` is the A block (``manifest_from_single_read``): the digest of the
    manifest bytes this invocation actually parsed.  It is echoed into the plan
    and, for a run, into ``run.json`` -- the guard's own document stays verbatim
    in ``guard.json``, so the two provenances remain distinguishable.

    ``selection`` is the ``budget_selection`` record of ``select_budget_tier``
    (the ``--budget-frames`` selector).  It is echoed into the plan **only when it
    actually skipped a root**: a single-tier manifest keeps exactly the plan
    document -- and therefore the exact stdout -- it had, which is what the
    byte-identical regression compares.  The same discipline gates
    ``takeover_override`` / ``prefix_tape`` / ``lineage_new_roots`` below.
    """

    closure = dict(closure or {})
    if selection is None:
        selection = getattr(args, "budget_selection", None)
    budget_block = (dict(selection)
                    if isinstance(selection, dict)
                    and int(selection.get("skipped_by_budget") or 0) > 0 else None)
    notes: list[str] = []
    checks: list[dict] = []
    for root in roots:
        check_engine_declaration(root)
    engine, engine_source = resolve_engine(args, manifest)
    data_source, data_source_source = resolve_data_source(args, manifest)
    engine_declared = str(roots[0].get("engine_sha256"))
    engine_info = engine_soft_check(engine, engine_declared)
    if engine_info["gate"] == "checked" and not engine_info["match"]:
        raise PairError(
            "ENGINE_SHA_MISMATCH",
            f"{engine} sha256 {str(engine_info['actual'])[:16]}... != manifest "
            f"declared {engine_declared[:16]}... (fail-closed: the run would be "
            f"unattributable)")
    # NB: a *missing* engine is not raised here even for pair/trace.  The
    # preflight is an unordered set of checks, and mode_run raises
    # ENGINE_MISSING last, after every protocol refusal (window, takeover,
    # budget, exposure, legacy opt-in, tag, pair_rng, checkpoint), so a
    # mis-registered cell is reported as itself rather than as "engine missing".
    checks.append({"check": "engine.sha256", "status": engine_info["gate"],
                   "engine": engine, "source": engine_source,
                   "detail": engine_info.get("note")})

    source_path = pathlib.Path(data_source)
    source_present = source_path.is_dir()
    # NB: like the engine digest, a missing data source is recorded here and
    # refused by mode_run (after every protocol refusal) instead of preempting
    # them -- see the note above the engine check.
    checks.append({"check": "data_source.present",
                   "status": "checked" if source_present else "deferred-to-run",
                   "path": data_source, "source": data_source_source,
                   "detail": (None if source_present else
                              "not visible from this host; mode=pair/trace "
                              "refuses DATA_SOURCE_MISSING")})

    plans = []
    arm_order = (list(args.arm_order) if isinstance(args.arm_order, list)
                 else parse_arm_order(args.arm_order))
    for root in roots:
        notes.extend(check_root(root, args, needs_engine=needs_engine))
        pair_rng, pair_rng_source = resolve_pair_rng(root, args, roots)
        tag = default_tag(args, root)
        if not str(tag).strip():
            raise PairError("TAG_REQUIRED",
                            "--tag is mandatory for a run: v8.17.0's 18-way "
                            "launch shared one timestamp data dir and corrupted "
                            "10 of 18 cells (MAP sec. 9-3)")
        out = out_for(args, tag)
        log_path = log_for(args, tag)
        work = pathlib.Path(args.work_dir)
        for target, what in ((work, "--work-dir"), (out, "--out"),
                             (log_path, "--log"),
                             (work / "runs" / tag, "runs/<tag>"),
                             (work / "data" / tag, "data/<tag>"),
                             (trace_path_for(out, "G1", episode=0,
                                             episodes=args.episodes),
                              "trace G1"),
                             (trace_path_for(out, "T2", episode=0,
                                             episodes=args.episodes),
                              "trace T2")):
            assert_writable(target, work_dir=work, what=what)
        collisions = [str(path) for path in (work / "runs" / tag,
                                             work / "data" / tag) if path.exists()]
        # mode=check writes its own invocation record to runs/<tag>/run.json, so a
        # repeated check must be able to overwrite *that* and nothing else.
        check_own_record = (args.mode == "check"
                            and all(check_record_only(pathlib.Path(entry))
                                    for entry in collisions))
        if collisions and not check_own_record:
            raise PairError(
                "TAG_NOT_UNIQUE",
                f"--tag {tag!r} already exists ({', '.join(collisions)}); the tag "
                f"owns data/<tag> and runs/<tag> and a reused tag silently "
                f"inherits leftover score.dat/th08.cfg (MAP sec. 9-3). Pick a new "
                f"tag or a new --work-dir")
        if args.out and pathlib.Path(args.out).exists():
            raise PairError("OUT_EXISTS",
                            f"--out {args.out} already exists; every run writes a "
                            f"fresh file (a retry after a technical fault uses a "
                            f"new tag, pre-registration sec. 4)")
        checks.append({"check": f"tag.unique[{root['_index']}]", "status": "checked",
                       "tag": tag, "runs": str(work / "runs" / tag),
                       "data": str(work / "data" / tag),
                       "overwrites_check_record_only": bool(check_own_record)})
        tape_resolve = root.get("_prefix_tape")
        tape_state = root.get("_prefix_tape_state") or {}
        tape_meta = dict(tape_state.get("meta") or {})
        if tape_resolve is None:
            recording_gate = prefix_digest_gate(root)
            checks.append({"check": f"prefix_recording[{root['_index']}]",
                           "status": recording_gate["gate"],
                           "recording": recording_gate["recording"],
                           "declared": recording_gate["declared"],
                           "actual": recording_gate["actual"],
                           "bytes": recording_gate["bytes"],
                           "match": recording_gate["match"],
                           "anchor_declared": recording_gate["anchor_declared"],
                           "anchor_match": recording_gate["anchor_match"],
                           "detail": recording_gate.get("note")})
        else:
            checks.append({
                "check": f"prefix_tape[{root['_index']}]", "status": "checked",
                "path": tape_resolve["path"], "sha256": tape_meta.get("sha256"),
                "bytes": tape_meta.get("bytes"), "rows": tape_meta.get("rows"),
                "frames_built": tape_meta.get("frames_built"),
                "target": tape_meta.get("target"),
                "recorded_hash_frames": tape_meta.get("recorded_hash_frames"),
                "anchors_complete": tape_meta.get("anchors_complete"),
                "declared": tape_resolve.get("declared"),
                "declared_sha256": tape_resolve.get("declared_sha256"),
                "source": tape_resolve.get("source"),
                "detail": ("the root declares its own 4-frame tape; pair/trace "
                           "re-point v450.masks_for at read_tape()'s "
                           "(masks, hashes, meta), so build_masks' hard-coded S1c "
                           "shard is not read for this root")})

        ckpt, ckpt_sha = resolve_ckpt(root, args) if needs_engine else (None, None)
        if needs_engine and ckpt is None:
            raise PairError(
                "CKPT_REQUIRED",
                f"roots[{root['_index']}]: --ckpt and --ckpt-sha256 are mandatory "
                f"for a run (or the root must carry an optional 'ckpt_path'); the "
                f"manifest pins the digest per root (MAP sec. 9-5)")
        argv_ckpt = ckpt or (pathlib.Path(args.ckpt) if args.ckpt
                             else (repo_path(root["ckpt_path"])
                                   if root.get("ckpt_path") else None))
        if not needs_engine:
            declared_ckpt = str(root.get("ckpt_sha256"))
            entry = {"check": f"ckpt[{root['_index']}]", "path":
                     None if argv_ckpt is None else str(argv_ckpt),
                     "declared": declared_ckpt}
            if argv_ckpt is None:
                entry.update({"status": "deferred-to-run", "detail":
                              "--ckpt/--ckpt-sha256 not supplied and the root "
                              "carries no 'ckpt_path'; required for pair/trace"})
            elif not argv_ckpt.is_file():
                entry.update({"status": "deferred-to-run", "actual": None,
                              "detail": "the checkpoint is not visible from this "
                                        "host; pair/trace refuse CKPT_MISSING"})
            else:
                actual_ckpt = sha256_file(argv_ckpt)
                if actual_ckpt != declared_ckpt:
                    raise PairError(
                        "CKPT_SHA_MISMATCH",
                        f"{argv_ckpt} sha256 {actual_ckpt[:16]}... != "
                        f"roots[{root['_index']}].ckpt_sha256 "
                        f"{declared_ckpt[:16]}...")
                entry.update({"status": "checked", "actual": actual_ckpt,
                              "bytes": argv_ckpt.stat().st_size, "match": True})
            checks.append(entry)
        recording = s1c_recording_path(int(root["window"]))
        if tape_resolve is None:
            # with a declared tape the S1c shard is not the prefix at all, so the
            # line would report an irrelevant missing asset; the prefix_tape line
            # above takes its place
            checks.append({"check": f"s1c_recording[{root['_index']}]",
                           "status": ("checked" if recording.is_file()
                                      else "deferred-to-run"),
                           "path": str(recording),
                           "bytes": (recording.stat().st_size
                                     if recording.is_file() else None)})
        # the takeover resolution `check_root` stored above; the extra check line
        # is emitted ONLY on the override path, so a frozen cell's plan document
        # stays byte-identical to the pre-v8.30 tool's
        takeover_plan = root["_takeover_resolution"]
        takeover_arg: int | None = None
        takeover_block: dict | None = None
        if takeover_plan["override_active"]:
            takeover_arg = int(takeover_plan["takeover_frame"])
            takeover_block = planned_prefix_constants(
                takeover_arg, window=int(root["window"]),
                source=takeover_plan["source"],
                tape=(tape_state.get("scan") if tape_resolve is not None else None))
            checks.append({
                "check": f"takeover[{root['_index']}]",
                "status": "planned",
                "takeover_frame": takeover_arg,
                "source": takeover_plan["source"],
                "expected_checkpoints": takeover_plan["expected_checkpoints"],
                "target_frame": takeover_plan["target_frame"],
                "recording_frames_built":
                    takeover_plan["recording"]["frames_built"],
                "files_on_disk_unmodified":
                    takeover_block["files_on_disk_unmodified"],
                "detail": ("non-frozen takeover: the pinned modules' globals are "
                           "re-pointed in this process at run time (mode=check "
                           "imports none of them, so nothing is applied here)")})
        plan = {
            "root_index": root["_index"],
            "ancestor_id": root.get("ancestor_id"), "stage": root.get("stage"),
            "window": int(root["window"]),
            "window_source": root.get("window_source"),
            "takeover_frame": int(root["takeover_frame"]),
            "total_frames": int(root["total_frames"]),
            "budget_frames": int(root["budget_frames"]),
            "budget_decisions": int(root["budget_frames"]) // DECISION_FRAMES,
            "pair_rng": int(pair_rng), "pair_rng_source": pair_rng_source,
            "tag": tag, "out": str(out), "log": str(log_path),
            "ckpt_path": None if argv_ckpt is None else str(argv_ckpt),
            "ckpt_sha256": ckpt_sha or root.get("ckpt_sha256"),
            "argv_pair": quote_argv(build_argv(
                mode="pair", args=args, root=root, tag=tag, out=out,
                log_path=log_path, arm=None, arm_order=arm_order,
                ckpt=argv_ckpt, takeover=takeover_arg)),
            "argv_trace": {arm: quote_argv(build_argv(
                mode="trace", args=args, root=root, tag=tag, out=out,
                log_path=log_path, arm=arm, arm_order=arm_order,
                ckpt=argv_ckpt, takeover=takeover_arg))
                for arm in ARMS},
            "guard_argv": quote_argv(
                [sys.executable, "-B", str(REPO / GUARD_REL), "check",
                 str(args.manifest), "--budget", str(int(args.budget_frames))]
                + (["--allow-exposed"] if args.allow_exposed else [])),
            "arm_order": arm_order,
            "episodes": int(args.episodes),
            "decision_frames": DECISION_FRAMES,
            "trace_files": {arm: str(trace_path_for(
                out, arm, episode=0, episodes=args.episodes)) for arm in ARMS},
            "rule_labels": {arm: rule_label_for(arm)[0] for arm in ARMS}
            if needs_engine else "computed at run time (needs v8r0/v8o0)",
        }
        if takeover_block is not None:
            plan["takeover"] = takeover_block
        if tape_resolve is not None:
            plan["prefix_tape"] = planned_prefix_tape_block(root)
        plans.append(plan)

    # B: the exposure cross-check runs over every selected root at once, so the
    # "same root twice" and "two roots, one exposed" cases cannot be judged apart.
    lineage = lineage_audit(roots=roots, args=args)
    lineage_by_root = {entry["root_index"]: entry for entry in lineage["roots"]}
    for plan in plans:
        entry = lineage_by_root.get(plan["root_index"], {})
        plan.update({
            "lineage_split": entry.get("lineage_split"),
            "lineage_status": entry.get("lineage_status"),
            "lineage_unmatched": list(entry.get("lineage_unmatched") or []),
            "lineage_code": entry.get("code"),
        })
    for entry in lineage["roots"]:
        checks.append({
            "check": f"lineage[{entry['root_index']}]",
            "status": ("checked" if entry["verdict"] == "OK" else entry["verdict"]),
            "lineage_split": entry["lineage_split"],
            "lineage_status": entry["lineage_status"],
            "lineage_unmatched": entry["lineage_unmatched"],
            "code": entry["code"],
            "detail": entry["detail"]})

    plan_document = {
        "schema": SCHEMA_PLAN, "tool": TOOL, "tool_version": TOOL_VERSION,
        "kind": "preflight-plan", "mode_requested": args.mode,
        "manifest_path": str(args.manifest),
        "manifest_sha256": closure.get("manifest_sha256",
                                       guard_doc.get("manifest_sha256")),
        "manifest_bytes": closure.get("manifest_bytes",
                                      guard_doc.get("manifest_bytes")),
        "manifest_closure": closure,
        "guard": {"path": GUARD_REL, "version": guard_doc.get("guard_version"),
                  "schema": guard_doc.get("schema"), "ok": guard_doc.get("ok"),
                  "codes": guard_doc.get("codes"),
                  "checks_run": guard_doc.get("checks_run"),
                  "check_count": guard_doc.get("check_count"),
                  "options": guard_doc.get("options"),
                  "findings": guard_doc.get("findings")},
        "engine": engine_info, "engine_source": engine_source,
        "data_source": {"path": data_source, "source": data_source_source,
                        "present": source_present},
        "lineage_checked": bool(lineage["lineage_checked"]),
        "lineage_config_path": lineage.get("lineage_config_path"),
        "lineage_config_sha256": lineage.get("lineage_config_sha256"),
        "lineage_sources_path": lineage.get("lineage_sources_path"),
        "lineage_sources_sha256": lineage.get("lineage_sources_sha256"),
        "lineage_codes": list(lineage.get("lineage_codes") or []),
        "lineage_unmatched": list(lineage.get("lineage_unmatched") or []),
        "lineage": lineage,
        "residual_risk": (list(closure.get("residual_risk") or [])
                          + list(lineage.get("residual_risk") or [])),
        "audio": {
            "env": "v450_cf_label.prepare_env() -> SDL_AUDIODRIVER=dummy, "
                   "DISPLAY/WAYLAND_DISPLAY popped",
            "client": "HarnessClient.__init__ -> SDL_AUDIODRIVER=dummy; "
                      "DISPLAY popped for role=shadow (protocol.py:497-510)",
            "cfg": "v432.build_private_data_dir -> regenerated 60-byte th08.cfg, "
                   "four fields re-inspected (v432:416-446)",
            "post_run_audit": "v431_audio_cfg_audit.decode() over "
                              "<work>/data/<tag>/**/th08.cfg -> "
                              "runs/<tag>/audio_audit.json; refuses when "
                              "muted != cfg_files or audible != 0",
        },
        "checks": checks,
        "notes": notes,
        "roots_selected": len(roots),
        "plans": plans,
        "would_start_engine": bool(needs_engine),
        "engine_side_modules_imported": engine_modules_loaded(),
    }
    if budget_block is not None:
        # The --budget-frames selector recorded its own result.  Only a mixed-tier
        # manifest gets this key: with selected == every root and skipped == 0 the
        # plan document, run.json and stdout stay byte-identical (requirement of
        # the same name), exactly like takeover_override / prefix_tape below.
        plan_document["budget_selection"] = budget_block
    # Present ONLY when some selected root is not the frozen 8663 cell.  Adding
    # the key unconditionally would change mode=check's output -- and its own
    # ``run.json`` -- for every pre-v8.30 cell, and the regression for the four
    # old windows at --takeover-frame 8663 is defined as byte-identical output.
    overrides = [entry["takeover"] for entry in plans if entry.get("takeover")]
    if overrides:
        plan_document["takeover_override"] = {
            "schema": SCHEMA_PLAN + ":takeover-override",
            "present_for": [int(entry["takeover_frame"]) for entry in overrides],
            "frozen_takeover_frame": PREFIX_FRAMES,
            "applied_by_check": False,
            "reason": ("mode=check imports none of the run-side modules, so it "
                       "reports what pair/trace would re-point instead of doing it"),
            "blocks": overrides,
        }
    # Same discipline for the optional prefix tape: absent unless a selected root
    # declares one, so every pre-tape cell keeps the exact plan document (and the
    # exact stdout) it had.
    tape_blocks = [entry["prefix_tape"] for entry in plans
                   if entry.get("prefix_tape")]
    if tape_blocks:
        plan_document["prefix_tape"] = {
            "schema": SCHEMA_PLAN + ":prefix-tape",
            "present_for_windows": [int(block["window"]) for block in tape_blocks],
            "applied_by_check": False,
            "reason": ("mode=check imports none of the run-side modules (the "
                       "ENGINE_SIDE_IMPORT gate), so it validates the tape with "
                       "read_tape() and reports the masks_for injection pair/trace "
                       "will perform; run.json records injected_into with the "
                       "object ids and files_on_disk_unmodified"),
            "blocks": tape_blocks,
        }
    # Same discipline for `lineage_new`: the key appears only when a selected root
    # opted in, so every pre-`lineage_new` invocation keeps its exact plan document
    # (and therefore its exact stdout) -- which is what the byte-identical
    # regression for the three old manifests compares.
    new_roots = list(lineage.get("lineage_new_roots") or [])
    if new_roots:
        plan_document["lineage_new_roots"] = new_roots
    return plan_document


def engine_modules_loaded() -> list[str]:
    """Proof that a dry run never touched the engine side (requirement 10)."""

    banned_roots = {"touhou_rl", "torch", "numpy", "subprocess"}
    found = []
    for name in sorted(sys.modules):
        head = name.split(".")[0]
        if head in banned_roots:
            found.append(name)
    return found


def render_plan(plan: dict) -> str:
    closure = plan.get("manifest_closure") or {}
    lineage = plan.get("lineage") or {}
    lines = [
        f"v829 pair: mode={plan['mode_requested']}  (check starts nothing)",
        f"  tool        {TOOL}  {plan['tool_version']}",
        f"  manifest    {plan['manifest_path']}",
        f"  sha256      {plan['manifest_sha256']}  "
        f"({plan['manifest_bytes']} B, source={closure.get('manifest_sha256_source')},"
        f" reads={closure.get('manifest_reads')})",
        f"  guard       {GUARD_REL}  {plan['guard']['version']}  "
        f"-> {'PASS' if plan['guard']['ok'] else 'REFUSED'} "
        f"({plan['guard']['check_count']} checks, codes={plan['guard']['codes']})",
        f"  roots       {plan['roots_selected']} selected",
    ]
    budget_selection = plan.get("budget_selection") or {}
    if budget_selection:
        # the selector's own line, printed only when it skipped a root (a
        # single-tier selection prints nothing new: byte-identical output)
        lines.append(budget_selection_line(budget_selection))
    if lineage.get("lineage_checked"):
        lines.append(f"  lineage     checked against {lineage['lineage_config_path']} "
                     f"sha256={str(lineage['lineage_config_sha256'])[:16]}...  "
                     f"codes={lineage['lineage_codes']}  "
                     f"unmatched={len(lineage['lineage_unmatched'])}")
        lines.append(f"  lineage src {lineage.get('lineage_sources_path')}  "
                     f"checked={lineage.get('lineage_sources_checked')}")
    else:
        lines.append("  lineage     NOT checked (--lineage-config absent): exposure "
                     "is self-declared")
    for entry in plan["plans"]:
        lines.append(
            f"  root[{entry['root_index']}] ancestor={entry['ancestor_id']} "
            f"stage={entry['stage']}")
        lines.append(
            f"      window {entry['window']} (source: {entry['window_source']})  "
            f"takeover {entry['takeover_frame']}  total {entry['total_frames']}  "
            f"budget {entry['budget_frames']} frames = "
            f"{entry['budget_decisions']} decisions")
        lines.append(
            f"      pair_rng {entry['pair_rng']} ({entry['pair_rng_source']})  "
            f"ckpt {entry['ckpt_path']}  tag {entry['tag']}")
        lines.append(f"      trace files "
                     f"{entry['trace_files']['G1']} / "
                     f"{entry['trace_files']['T2']}")
        lines.append(f"      lineage split={entry.get('lineage_split')} "
                     f"status={entry.get('lineage_status')} "
                     f"unmatched={entry.get('lineage_unmatched')} "
                     f"code={entry.get('lineage_code')}")
    for check in plan["checks"]:
        detail = check.get("detail") or ""
        lines.append(f"  check  {check['check']:24s} {check['status']:16s} {detail}")
    for note in plan["notes"]:
        lines.append(f"  note   {note}")
    override = plan.get("takeover_override") or {}
    if override:
        lines.append(f"  takeover    non-frozen takeover requested: N in "
                     f"{override['present_for']}  (frozen is {PREFIX_FRAMES})")
        for block in override["blocks"]:
            recording = block.get("recording") or {}
            lines.append(
                f"      root[{block['takeover_frame']}] window {block['window']}  "
                f"N={block['takeover_frame']} ({block['source']})  "
                f"expected checkpoints {block['expected_checkpoints']} "
                f"(= N // {ANCHOR_STRIDE})  target frame {block['target_frame']} "
                f"(= N + 1, a {ANCHOR_STRIDE}-frame anchor)")
            lines.append(
                f"      recording {recording.get('path')}  "
                f"{recording.get('rows')} rows x {ANCHOR_STRIDE} = "
                f"{recording.get('frames_built')} frames_built  "
                f"target {block['target_frame']} <= "
                f"{recording.get('frames_built')}  "
                f"{'OK' if recording.get('gate') == 'checked' else 'DEFERRED'}")
            lines.append("      module globals to re-point (this process's copies "
                         "only; the files on disk are not touched):")
            for target in block["targets"]:
                rendered = "  ".join(f"{key}={value}"
                                     for key, value in target["globals"].items())
                lines.append(f"          {target['file']:30s} {rendered}")
            lines.append(
                f"      module objects: mode=check imports none of them, so the object "
                f"ids are discovered at run time -- v829's own 3 copies plus every "
                f"copy v450/v830 load for themselves (v450._M432, v830._M450, "
                f"v830._M432, v830._M450._M432); run.json records the authoritative "
                f"list with object ids")
            lines.append(
                f"      files on disk unmodified: "
                f"{block['files_on_disk_unmodified']}  "
                + "  ".join(f"{rel} {info['sha256'][:16]}..."
                            for rel, info in block["pinned_files"].items()
                            if info.get("sha256")))
    tapes = [entry["prefix_tape"] for entry in plan["plans"]
             if entry.get("prefix_tape")]
    if tapes:
        lines.append("  prefix_tape  a root declares its own 4-frame tape; "
                     "v450_cf_label.masks_for is re-pointed in-process, so "
                     "v431_dense_driver.build_masks (and the S1c shard it "
                     "hard-codes) is NOT read for that root")
        for block in tapes:
            lines.append(
                f"      window {block['window']}  takeover "
                f"{block['takeover_frame']} (= 4k - 1)  ->  expected checkpoints "
                f"{block['expected_checkpoints']} (N // {ANCHOR_STRIDE})  "
                f"target {block['target_frame']} (N + 1, a recorded anchor)")
            lines.append(f"      tape {block['path']}  ({block['bytes']} B)")
            declared_tape = block['declared']['prefix_tape_sha256'] or ''
            declared_parent = block['declared']['parent_prefix_digest'] or ''
            digest_note = ('all equal'
                           if block['sha256'] in (declared_tape, declared_parent)
                           else 'checked')
            lines.append(
                f"      sha256 {block['sha256']}  declared prefix_tape_sha256 "
                f"{block['declared']['prefix_tape_sha256']}  parent_prefix_digest "
                f"{block['declared']['parent_prefix_digest']}  ({digest_note})")
            lines.append(
                f"      rows {block['rows']} x {ANCHOR_STRIDE} = "
                f"{block['frames_built']} frames_built  recorded hash frames "
                f"{block['recorded_hash_frames']}  anchors_complete "
                f"{block['anchors_complete']}  target {block['target']} <= "
                f"{block['frames_built']}  OK")
            lines.append(
                f"      read_tape(path, {block['target']}) -> "
                f"(masks[{block['masks_len']}], hashes["
                f"{block['recorded_hash_frames']}], meta); the list is not "
                f"truncated at the target (a recorded policy keeps following it)")
            lines.append(
                "      injection: (planned) v450_cf_label.masks_for = read_tape()'s "
                "tuple, plus every in-process copy that carries the name "
                "(v450._M432 / v830._M450 copies are inspected and reported) -- "
                "mode=check imports none of the run-side modules, so nothing is "
                "assigned here; run.json records injected_into with object ids")
            lines.append(
                f"      files on disk unmodified: "
                f"{block['files_on_disk_unmodified']}  "
                + "  ".join(f"{rel} {info['sha256'][:16]}..."
                            for rel, info in block["pinned_files"].items()
                            if info.get("sha256")))
    lines.append(f"  engine-side modules imported: "
                 f"{plan['engine_side_modules_imported'] or 'none'}")
    lines.append(f"  would start engine: {plan['would_start_engine']}")
    lines.append("--- would_execute (nothing below has been run) ---")
    for entry in plan["plans"]:
        lines.append(f"  [pair  ] {entry['argv_pair']}")
        lines.append(f"  [trace G1] {entry['argv_trace']['G1']}")
        lines.append(f"  [trace T2] {entry['argv_trace']['T2']}")
        lines.append(f"  [guard ] {entry['guard_argv']}")
    lines.append("  note: the printed argv carries --allow-legacy-window because "
                 "v1 supports only windows (11, 12, 17, 18); pair/trace refuse "
                 "LEGACY_WINDOW_NOT_ALLOWED without it")
    if override:
        lines.append("  note: the printed argv carries "
                     f"--takeover-frame {override['blocks'][0]['takeover_frame']} "
                     "because that cell is not the frozen 8663 one; the flag must "
                     "equal roots[i].takeover_frame")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# modes that start the engine
# --------------------------------------------------------------------------- #
def require_legacy_opt_in(args: argparse.Namespace, roots: list[dict]) -> None:
    if args.allow_legacy_window:
        return
    windows = sorted({int(root["window"]) for root in roots})
    raise PairError(
        "LEGACY_WINDOW_NOT_ALLOWED",
        f"windows {windows} are the historical development windows that "
        f"pre-registration sec. 2 excludes from new samples; a run against them is "
        f"a smoke/bridge run and needs the explicit --allow-legacy-window "
        f"(it is not implied by --allow-exposed)")


# --------------------------------------------------------------------------- #
# technical-fault re-run of one pair (pre-registration sec. 4)
# --------------------------------------------------------------------------- #
#: "一次技术故障只按原 pair、起点、随机流和配置重跑双方一次，保留两次记录."
#:
#: A technical fault is either a **row fault** -- ``invalid`` or a non-empty
#: ``error`` on **any** arm's row -- or a **recovery fault** -- ``start_parent``
#: raised while building HELLO -> prefix -> PARK -> DEBUG (sec. 4's "无法恢复").
#: A legal death, a full segment and a ``stage_clear`` are results, not faults:
#: they are exactly what the run was supposed to measure, so they never re-run
#: (`should_retry` below never looks at ``terminal`` / ``frames`` / ``score``).
#:
#: Both records are kept: attempt 1 stays in the jsonl (nothing is deleted or
#: overwritten) and attempt 2 is appended with ``attempt: 2`` plus
#: ``supersedes: <sha256 of the attempt-1 row it replaces>``.  ``run.json`` gets
#: ``attempts: [{attempt, phase, error, tag, invalid_rows, wall_s, ...}]``.
def technical_invalid(row: object) -> bool:
    """True when one row records a *technical* fault (never a legal result)."""

    if not isinstance(row, dict):
        return True
    if bool(row.get("invalid")):
        return True
    return bool(str(row.get("error") or "").strip())


def should_retry(rows: object) -> bool:
    """The *row* re-run decision: does this attempt hold a technical fault?

    Injected-free and total, so a unit test can drive it directly.  An empty
    attempt is **not** retried: a runner that produced no row at all has failed
    in a way a second identical attempt cannot diagnose, and pre-registration
    sec. 4 counts rows, not absence of rows.  A recovery fault is the other kind
    of technical fault and is decided by `run_pair_with_retry` from a
    `RecoveryFault`, never through this function (it has no row to look at).
    """

    rows_list = list(rows or [])
    if not rows_list:
        return False
    return any(technical_invalid(row) for row in rows_list)


def row_fingerprint(row: dict) -> str:
    """Stable identity of one row (the ``supersedes`` value of a re-run)."""

    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True,
                   default=str).encode("utf-8")).hexdigest()


def attempt_fingerprint(fingerprints: list[str]) -> str | None:
    """One digest over a whole attempt's rows (``None`` for an empty attempt)."""

    if not fingerprints:
        return None
    return hashlib.sha256(("\n".join(fingerprints)).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# recovery (PARK) faults: pre-registration sec. 4's "无法恢复"
# --------------------------------------------------------------------------- #
#: ``phase`` of one ``run.json.attempts[i]`` record: a normal attempt that could
#: produce rows (``"play"``) or an attempt that died inside ``start_parent``
#: before any row could exist (``"recovery"``).
PLAY_PHASE = "play"
RECOVERY_PHASE = "recovery"

#: Values of ``run.json.attempts[i].kind`` for a recovery fault.  Both kinds are
#: retried once -- sec. 4 makes *unrecoverability* the technical fault -- but the
#: label keeps a flaky engine start (`engine_crash`) distinguishable from a
#: possible data/tape problem (`prefix_hash_mismatch`) in the record.
RECOVERY_KINDS = ("engine_crash", "prefix_hash_mismatch", "recovery_error")


class RecoveryFault(RuntimeError):
    """One failed PARK: technical invalid, re-run at most once (sec. 4).

    Raised by `park_parent`; `run_pair_with_retry` catches it, writes one
    ``phase: "recovery"`` attempt record and -- if an attempt is left -- re-runs
    the *whole* pair with the same manifest / window / takeover / tape / ckpt /
    ``pair_rng`` (they are closed over by the runner).  A `PairError` is never
    wrapped: those are pre-spawn refusals (missing engine/data/checkpoint, a bad
    path, a bad manifest), i.e. input problems, and they must not retry.
    """

    def __init__(self, *, window: int, attempt: int, error: str,
                 error_type: str, kind: str, stage: str) -> None:
        super().__init__(error)
        self.window = int(window)
        self.attempt = int(attempt)
        self.error = str(error)
        self.error_type = str(error_type)
        self.kind = str(kind)
        self.stage = str(stage)

    def record(self, *, attempt: int | None = None) -> dict:
        """The ``run.json.attempts[i]`` half of this fault.

        ``error`` is ``"attempt <n> <kind>: <ExceptionClass>: <message>"``: the
        class and message make it greppable, the attempt number is what tells the
        two records of a doubly-failed recovery apart.
        """

        number = self.attempt if attempt is None else int(attempt)
        return {
            "phase": RECOVERY_PHASE,
            "stage": self.stage,
            "kind": self.kind,
            "window": self.window,
            "attempt": number,
            "error_type": self.error_type,
            "error": f"attempt {number} {self.kind}: {self.error}",
        }


def recovery_kind(exc: BaseException) -> str:
    """Classify one ``start_parent`` exception (never imports the pinned modules)."""

    name = type(exc).__name__
    text = str(exc)
    if name == "CfError" and ("fidelity gate failed" in text
                              or "hash_mismatch" in text
                              or "mismatches" in text):
        # `v450_cf_label.start_parent:469-472` dumps the whole gate; the failing
        # assertion may be the prefix hashes, HELLO role, PARK frame or DEBUG.
        return "prefix_hash_mismatch"
    if name in ("HarnessError", "ProtocolError", "HarnessClosed",
                "BrokenPipeError", "EOFError", "TimeoutError",
                "ConnectionError", "ConnectionResetError"):
        return "engine_crash"
    if "control pipe" in text or "harness" in text.lower():
        return "engine_crash"
    return "recovery_error"


def recovery_stage(exc: BaseException) -> str:
    """Best-effort sub-stage of the failed PARK, for the attempt record."""

    text = str(exc)
    if "fidelity gate failed" in text:
        # the gate JSON names the assertion; the distinct code above is the
        # machine-readable half, this is only a human hint
        return "fidelity_gate"
    lowered = text.lower()
    if "hello" in lowered:
        return "hello"
    if "park" in lowered:
        return "park"
    if "debug" in lowered or "thread" in lowered:
        return "debug"
    return "start_parent"


#: Module-level replaceable hook for the PARK call.  ``None`` (the default) means
#: `v450_cf_label.start_parent`; a test or a wrapper sets it to a callable with
#: the same keyword signature to drive the recovery path without an engine.
START_PARENT_HOOK = None


def default_start_parent(*, slot: int, window: int, engine: str,
                         data_dir: pathlib.Path,
                         log_dir: pathlib.Path):
    """The production PARK call: `v450_cf_label.start_parent` (pinned file)."""

    return m450().start_parent(slot=slot, window=window, engine=engine,
                               data_dir=data_dir, log_dir=log_dir)


def gate_name_for(window: int, attempt: int) -> str:
    """``gate-w<w>.json`` for attempt 1, ``gate-w<w>-a<attempt>.json`` after that.

    A recovery re-run therefore never overwrites attempt 1's gate: if attempt 1
    died inside `start_parent` there is no attempt-1 gate at all, and the fresh
    PARK lands in ``gate-w<w>-a2.json`` -- the same a2 naming an in-play re-run
    uses, so the two kinds of re-run stay comparable.
    """

    if int(attempt) <= 1:
        return f"gate-w{int(window)}.json"
    return f"gate-w{int(window)}-a{int(attempt)}.json"


def park_parent(*, window: int, engine: str, data_dir: pathlib.Path,
                log_dir: pathlib.Path, attempt: int, run_dir: pathlib.Path,
                start_parent_fn=None) -> dict:
    """One PARK (HELLO -> prefix -> PARK -> DEBUG), injected for testability.

    ``start_parent_fn`` (or the module-level `START_PARENT_HOOK`) defaults to
    `default_start_parent`; a test passes a fake and a wrapper may replace the
    module global, so the whole recovery path is drivable offline.

    Any exception from the PARK call other than `PairError` becomes a
    `RecoveryFault`: pre-registration sec. 4 counts "无法恢复" as a technical
    fault, so the attempt is retried once instead of aborting the run -- the gap
    this function closes.  On success the gate document is written immediately
    under its attempt-specific name (`gate_name_for`) and returned; nothing else
    writes ``gate-w*.json``.
    """

    fn = start_parent_fn or START_PARENT_HOOK or default_start_parent
    try:
        parent, gate = fn(slot=int(window), window=int(window), engine=engine,
                          data_dir=data_dir, log_dir=log_dir)
    except PairError:
        # a pre-spawn refusal is an input problem, not a technical fault
        raise
    except Exception as exc:                                      # noqa: BLE001
        error_type = type(exc).__name__
        raise RecoveryFault(
            window=int(window), attempt=int(attempt),
            error=f"{error_type}: {exc}", error_type=error_type,
            kind=recovery_kind(exc), stage=recovery_stage(exc)) from exc
    gate_name = gate_name_for(window, attempt)
    write_json(pathlib.Path(run_dir) / gate_name, gate)
    return {"parent": parent, "gate": gate, "gate_name": gate_name,
            "attempt": int(attempt)}


def recovery_failure(retry: dict, *, window: int,
                     takeover_frame: int | None = None,
                     expected_checkpoints: int | None = None) -> PairError:
    """The ``RECOVERY_FAILED`` refusal for a pair that could not PARK twice.

    ``run.json`` is written by the caller before this error is raised, so both
    ``phase: "recovery"`` records survive; the message repeats both errors (each
    naming its attempt number) and says no third attempt was made.
    """

    records = [record for record in (retry.get("attempts") or [])
               if record.get("phase") == RECOVERY_PHASE]
    detail = "; ".join(str(record.get("error")) for record in records) \
        or "no recovery record"
    used = len(retry.get("attempts") or [])
    expected = ("HELLO -> prefix walk -> PARK -> DEBUG threads=1"
                if takeover_frame is None or expected_checkpoints is None else
                f"HELLO -> {int(takeover_frame)}-frame prefix -> "
                f"{int(expected_checkpoints)} hash checks -> PARK -> "
                f"DEBUG threads=1")
    return PairError(
        "RECOVERY_FAILED",
        f"w{int(window)}: v450.start_parent failed on every attempt "
        f"({used} attempt(s); {expected}): {detail}; no third attempt "
        f"(pre-registration sec. 4 allows exactly one re-run)")


def retry_exit_code(retry: dict) -> int:
    """0 when the final attempt is clean, ``EXIT_REFUSED`` when it is a fault."""

    if retry.get("fault") is not None or retry.get("invalid"):
        return EXIT_REFUSED
    return EXIT_OK


def run_pair_with_retry(runner, *, tag: str, out: pathlib.Path | None = None,
                        max_attempts: int | None = None,
                        on_attempt=None, context: dict | None = None,
                        clock=time.time, retry_rows: bool = True) -> dict:
    """Run one pair, re-running the **whole** pair at most once on a fault.

    ``runner(attempt) -> rows`` (or ``{"rows": [...]}``) is injected: the real
    ``mode_run`` passes a closure that prepares the data dir, starts one parked
    parent and runs every episode/arm, and a unit test passes a fake.  The
    re-run is therefore literally the same call with ``attempt=2`` and **no**
    parameter changed -- same manifest, window, takeover frame, tape,
    checkpoint and ``pair_rng`` (they are closed over by the runner).

    Two kinds of technical fault are retried, both under the same two-attempt
    cap (pre-registration sec. 4):

    * a `RecoveryFault` raised by the runner (PARK died at HELLO / prefix / PARK
      / DEBUG) is caught here; the attempt gets a ``phase: "recovery"`` record
      with its ``error`` and no rows;
    * a *row* fault (``invalid`` / non-empty ``error``, `should_retry`) when
      ``retry_rows`` is true.  ``--mode trace`` passes ``retry_rows=False``: one
      arm has no pair to re-run, but a recovery fault still retries once.

    Persistence lives here, after the decoration, so both records are kept and
    the written row and its ``attempt``/``supersedes`` keys can never disagree:
    every row is appended to ``out`` as one JSON line (the same serialisation
    ``mode_run`` always used).  ``context`` (a dict) is filled in before each
    attempt with ``attempt``, ``previous_fingerprints`` and
    ``previous_attempt_fingerprint``, so a real runner can decorate its rows
    *before* it hashes them into a trace.

    Returns ``{"attempts": [...], "rows": <final attempt>, "all_rows": <both>,
    "attempts_used", "retried", "invalid", "fault", "recovery_failed",
    "superseded_rows", "last"}``.  ``fault`` is the final attempt's recovery
    record (``None`` when the final attempt produced rows), so the caller can
    tell the two failure modes apart and fail closed with `recovery_failure`.
    """

    limit = RETRY_ATTEMPT_LIMIT if max_attempts is None else int(max_attempts)
    # the pre-registered cap wins over any caller: never a third attempt
    limit = max(1, min(limit, RETRY_ATTEMPT_LIMIT))
    state = context if context is not None else {}
    attempts: list[dict] = []
    all_rows: list[dict] = []
    final_rows: list[dict] = []
    previous_fingerprints: list[str] = []
    previous_attempt: str | None = None
    last_result: object = None
    for attempt in range(1, limit + 1):
        state.update({"attempt": attempt,
                      "previous_fingerprints": list(previous_fingerprints),
                      "previous_attempt_fingerprint": previous_attempt})
        started = clock()
        try:
            result = runner(attempt)
        except RecoveryFault as fault:
            # sec. 4's "无法恢复": no parent exists, so this attempt has no row;
            # it still gets a record and the loop's cap decides whether a second
            # PARK is allowed (never a third).
            result = {"rows": [], "fault": fault.record(attempt=attempt),
                      "recovery": True}
        last_result = result
        call_source = (result.get("rows") if isinstance(result, dict)
                       else result)
        rows = [dict(row) for row in (call_source or [])]
        fault_record = (result.get("fault") if isinstance(result, dict)
                        else None)
        if fault_record is not None and not isinstance(fault_record, dict):
            raise TypeError(f"runner returned a non-dict fault: {fault_record!r}")
        wall_s = max(0.0, float(clock()) - float(started))
        # decorate FIRST, then fingerprint: a `supersedes` value must be the
        # sha256 of the row exactly as it was written (with its `attempt` key),
        # otherwise the two records could not be matched up again.
        for index, row in enumerate(rows):
            row.setdefault("attempt", attempt)
            if attempt > 1:
                if index < len(previous_fingerprints):
                    row.setdefault("supersedes", previous_fingerprints[index])
                elif previous_attempt is not None:
                    row.setdefault("supersedes", previous_attempt)
        fingerprints = [row_fingerprint(row) for row in rows]
        record = {
            "attempt": attempt, "tag": tag,
            "phase": RECOVERY_PHASE if fault_record is not None else PLAY_PHASE,
            "rows": len(rows),
            "invalid_rows": sum(1 for row in rows if technical_invalid(row)),
            "error": None if fault_record is None else fault_record.get("error"),
            "kind": None if fault_record is None else fault_record.get("kind"),
            "stage": None if fault_record is None else fault_record.get("stage"),
            "fault": fault_record,
            "wall_s": round(wall_s, 3),
            "attempt_fingerprint": attempt_fingerprint(fingerprints),
            "supersedes": previous_attempt,
            "out": None if out is None else str(out),
        }
        attempts.append(record)
        if out is not None:
            with pathlib.Path(out).open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False,
                                            default=str) + "\n")
                handle.flush()
        all_rows.extend(rows)
        final_rows = rows
        if on_attempt is not None:
            on_attempt(record, rows)
        row_fault = bool(should_retry(rows)) if retry_rows else False
        if fault_record is None and not row_fault:
            break
        previous_fingerprints = fingerprints
        previous_attempt = record["attempt_fingerprint"]
    retried = len(attempts) > 1
    final_fault = attempts[-1].get("fault") if attempts else None
    return {
        "attempts": attempts,
        "rows": final_rows,
        "all_rows": all_rows,
        "attempts_used": len(attempts),
        "max_attempts": limit,
        "retried": bool(retried),
        "invalid": bool(final_fault is not None or should_retry(final_rows)),
        "fault": final_fault,
        "recovery_failed": bool(final_fault is not None
                                and final_fault.get("phase") == RECOVERY_PHASE),
        "superseded_rows": (sum(record["rows"] for record in attempts[:-1])
                            if retried else 0),
        "last": last_result,
        "tag": tag,
        "out": None if out is None else str(out),
    }


# --------------------------------------------------------------------------- #
# --reuse-park: one PARK per (prefix, takeover, budget) group, one spawned child
#               per (cell, arm) -- OFF BY DEFAULT
# --------------------------------------------------------------------------- #
#: Measured cost of one PARK on this host: ~1.8 s (8,663-frame prefix replay +
#: 2,165 hash checks), of which **none** depends on the checkpoint, the pair_rng
#: or the arm -- so the 96 cells of the sec. 2 pilot would spend 96 x 1.8 s ~
#: 2.9 min re-deriving the same parked state.  The three things a PARK result
#: *does* depend on are `PARK_REUSE_KEY_FIELDS`; everything else that can reach
#: the parent is added by `park_reuse_identity` as a fail-closed discriminator.
def park_reuse_enabled(args: argparse.Namespace) -> bool:
    """``--reuse-park`` as a pure predicate, defaulting to off."""

    return bool(getattr(args, "reuse_park", DEFAULT_REUSE_PARK))


def park_reuse_tape_key(root: dict) -> str:
    """The prefix half of the group key: the tape's sha256, or ``s1c-seed<w>``.

    A root that declares ``prefix_tape`` replays **that file**; `resolve_prefix_tape`
    proved its sha256 is the cell's ``parent_prefix_digest``, so the digest *is*
    the prefix identity.  A root without one replays the window's hard-coded S1c
    shard (`v431_dense_driver.build_masks`), whose identity is its seed -- two
    different windows are two different prefixes and never share a PARK.
    """

    resolve = root.get("_prefix_tape") or {}
    state = root.get("_prefix_tape_state") or {}
    meta = dict(state.get("meta") or {})
    digest = resolve.get("sha256") or meta.get("sha256")
    if digest:
        return str(digest)
    return "s1c-seed" + str(int(root["window"]))


def park_reuse_group_key(root: dict) -> tuple:
    """``(prefix_tape sha256 | "s1c-seed<window>", takeover_frame, budget_frames)``.

    Exactly the key the work order names (`PARK_REUSE_KEY_FIELDS`), as a hashable
    tuple.  A different takeover frame or a different budget is a different group
    even on the same tape: the parent PARKS at the first and the episode is
    written against the second.
    """

    return (park_reuse_tape_key(root), int(root["takeover_frame"]),
            int(root["budget_frames"]))


def park_reuse_identity(root: dict, *, work, tag: str) -> dict:
    """Every **other** input a PARK result depends on (the fail-closed half).

    The group key above is the three fields the work order names; this tuple is
    what makes "never a silent mix" measurable.  Two roots that agree on the key
    but disagree here -- a different window, private data dir, declared engine
    digest, tape target/frames or pinned prefix digest -- get **separate** groups
    (each its own PARK), because one parked parent cannot serve both.

    It is computed once when the plan is built and re-computed per group member by
    `park_reuse_verify`, so a future change to the key can only ever split a
    group, never merge two different cells into one prefix.
    """

    resolve = root.get("_prefix_tape") or {}
    state = root.get("_prefix_tape_state") or {}
    meta = dict(state.get("meta") or {})
    takeover = int(root["takeover_frame"])
    window = int(root["window"])
    engine_sha = root.get("engine_sha256")
    parent_digest = root.get("parent_prefix_digest")
    return {
        "window": window,
        "takeover_frame": takeover,
        "budget_frames": int(root["budget_frames"]),
        "expected_checkpoints": takeover // ANCHOR_STRIDE,
        "target_frame": takeover + 1,
        # `v450.prepare_data_dir(work/data/<tag>, window, source)` is exactly this
        # path (`v450_cf_label.py:414-415`); it is spelled out here because
        # mode=check must not import the engine side to draw the same plan.
        "data_dir": str(pathlib.Path(work) / "data" / str(tag) / ("w" + str(window))),
        "engine_sha256": None if engine_sha is None else str(engine_sha),
        "prefix_tape_path": (None if resolve.get("path") is None
                             else str(resolve.get("path"))),
        "prefix_tape_sha256": (None if resolve.get("sha256") is None
                               else str(resolve.get("sha256"))),
        "prefix_tape_target": (None if meta.get("target") is None
                               else int(meta.get("target"))),
        "prefix_tape_frames_built": (None if meta.get("frames_built") is None
                                     else int(meta.get("frames_built"))),
        "parent_prefix_digest": (None if parent_digest is None
                                 else str(parent_digest)),
    }


def park_reuse_key_digest(key: tuple, identity: dict) -> str:
    """Stable id of one group: sha256 over its key **and** its identity."""

    payload = json.dumps({"key": list(key), "identity": identity},
                         ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def park_reuse_plan(roots: list[dict], *, work, tag: str,
                    enabled: bool = True) -> dict:
    """Group the selected roots; **one PARK per group** when ``enabled``.

    ``enabled=False`` builds the very same document with one singleton group per
    cell, so the caller has exactly ONE code path: the legacy behaviour is the
    degenerate plan (one PARK per cell, no proxy, no ``park_reuse`` row key, and
    therefore not one byte of changed output).

    ``only``: a root whose group key *and* full PARK identity match a group
    already in the list joins it; anything else opens a new group.  Nothing is
    ever merged on a partial match -- that is hard constraint #1 of the work
    order, and `park_reuse_verify` re-derives it from the roots afterwards.

    ``order`` is the execution order: groups by first appearance and, inside a
    group, manifest order.  A group's cells are therefore **contiguous**, which
    is what keeps at most one parent alive at a time.  ``--reuse-park`` changes
    the row ORDER of the jsonl (group order instead of manifest order); the
    content of every row is unchanged (see the report's equivalence table).
    """

    groups: list[dict] = []
    by_root: dict[str, str] = {}
    for root in roots:
        key = park_reuse_group_key(root)
        identity = park_reuse_identity(root, work=work, tag=tag)
        digest = park_reuse_key_digest(key, identity)
        group = None
        if enabled:
            for candidate in groups:
                if candidate["key_digest"] == digest:
                    group = candidate
                    break
        if group is None:
            index = len(groups)
            group = {
                "index": index,
                "group": "g" + str(index + 1),
                "park_id": "g" + str(index + 1) + "-" + digest[:12],
                "key": key,
                "key_digest": digest,
                "identity": identity,
                "root_indices": [], "roots": [],
                # -- live state, filled in by ParkReuseSession -----------------
                "park_calls": 0, "park": None, "park_pid": None,
                "park_gate_name": None, "children_spawned": 0,
                "children_failed": 0, "rows": 0, "cell_faults": [],
                "skipped": [], "void": False, "void_reason": None,
                "closed": False, "closed_reason": None, "close_error": None,
            }
            groups.append(group)
        group["roots"].append(root)
        group["root_indices"].append(int(root["_index"]))
        by_root[str(int(root["_index"]))] = group["group"]
    order: list[int] = []
    for group in groups:
        order.extend(group["root_indices"])
    document = {
        "schema": PARK_REUSE_SCHEMA,
        "enabled": bool(enabled),
        "key_fields": list(PARK_REUSE_KEY_FIELDS),
        "work": str(work), "tag": str(tag),
        "cells": len(roots), "groups": groups,
        "group_count": len(groups),
        "by_root": by_root, "order": order,
        # what the flag is worth: one PARK per cell without it, one per group with
        "park_calls_planned": len(groups),
        "park_calls_without_reuse": len(roots),
        "park_saved_planned": 0 if not enabled else len(roots) - len(groups),
        "group_note": (
            "one PARK per (prefix, takeover, budget) group; every other PARK "
            "input is a fail-closed discriminator, so a partial match opens a "
            "new group instead of mixing two cells' prefixes"),
        "verified": False,
    }
    park_reuse_verify(document, work=work, tag=tag)
    return document


def park_reuse_verify(plan: dict, *, work, tag: str) -> dict:
    """Fail closed: no group may hold two roots with a different PARK identity.

    Hard constraint #1 of the work order ("组内任一根的 takeover_frame / tape /
    budget 不同 -> 必须不分组,不能静默混用").  Every group member is re-derived
    from its root and compared field by field; a disagreement is
    ``PARK_REUSE_GROUP_MISMATCH`` (exit 1) instead of a shared PARK.
    """

    seen: dict[str, dict] = {}
    for group in plan.get("groups") or []:
        for root in group.get("roots") or []:
            key = park_reuse_group_key(root)
            identity = park_reuse_identity(root, work=work, tag=tag)
            if key != group["key"] or identity != group["identity"]:
                raise PairError(
                    "PARK_REUSE_GROUP_MISMATCH",
                    "group " + str(group["group"]) + " would mix roots["
                    + str(root.get("_index")) + "] with key " + repr(key)
                    + " / identity " + repr(identity) + " against the group's "
                    + repr(group["key"]) + " / " + repr(group["identity"])
                    + "; one parked parent cannot serve two different prefixes, "
                      "so the group must be split (each cell its own PARK)")
            seen[str(int(root["_index"]))] = group["group"]
    if seen != {str(key): value for key, value in (plan.get("by_root") or {}).items()}:
        raise PairError(
            "PARK_REUSE_GROUP_MISMATCH",
            "the by_root map does not agree with the group membership: "
            + repr(plan.get("by_root")) + " vs " + repr(seen))
    plan["verified"] = True
    return plan


def park_reuse_document(plan: dict) -> dict:
    """The ``park_reuse`` block ``run.json`` records (JSON-safe, all live state)."""

    groups: list[dict] = []
    for group in plan.get("groups") or []:
        groups.append({
            "group": group["group"],
            "park_id": group["park_id"],
            "key": {"prefix": group["key"][0],
                    "takeover_frame": group["key"][1],
                    "budget_frames": group["key"][2]},
            "key_digest": group["key_digest"],
            "identity": group["identity"],
            "root_indices": list(group["root_indices"]),
            "cells": len(group["root_indices"]),
            "park_calls": int(group["park_calls"]),
            "park_pid": group.get("park_pid"),
            "park_gate_name": group.get("park_gate_name"),
            "rows": int(group.get("rows") or 0),
            "children_spawned": int(group.get("children_spawned") or 0),
            "children_failed": int(group.get("children_failed") or 0),
            "cell_faults": list(group.get("cell_faults") or []),
            "void": bool(group.get("void")),
            "void_reason": group.get("void_reason"),
            "skipped_root_indices": list(group.get("skipped") or []),
            "closed": bool(group.get("closed")),
            "closed_reason": group.get("closed_reason"),
            "close_error": group.get("close_error"),
        })
    return {
        "schema": PARK_REUSE_SCHEMA,
        "enabled": bool(plan.get("enabled")),
        "key_fields": list(plan.get("key_fields") or PARK_REUSE_KEY_FIELDS),
        "cells": int(plan.get("cells") or 0),
        "group_count": int(plan.get("group_count") or 0),
        "groups": groups,
        "order": list(plan.get("order") or []),
        # ``park_calls`` is what actually happened (0 in a --mode check plan);
        # ``park_calls_planned`` is what this plan will do, and
        # ``park_calls_without_reuse`` is what the same cells cost without the flag
        "park_calls": sum(int(group["park_calls"]) for group in groups),
        "park_calls_planned": int(plan.get("park_calls_planned") or 0),
        "park_calls_without_reuse": int(plan.get("park_calls_without_reuse") or 0),
        "park_saved": (int(plan.get("park_calls_without_reuse") or 0)
                       - sum(int(group["park_calls"]) for group in groups)),
        "park_saved_planned": int(plan.get("park_saved_planned") or 0),
        "children_spawned": sum(int(group["children_spawned"]) for group in groups),
        "children_failed": sum(int(group["children_failed"]) for group in groups),
        "void_groups": [group["group"] for group in groups if group["void"]],
        "any_cell_fault": bool(any(group["cell_faults"] for group in groups)),
        "verified": bool(plan.get("verified")),
        "note": (
            "one start_parent per group; every (cell, arm) spawns its own child "
            "from that one parked parent and QUITs it. A spawn/QUIT fault makes "
            "the cell invalid and never re-PARKs; a dead parent voids the whole "
            "group and the run exits non-zero"),
    }


def render_park_reuse(plan: dict) -> str:
    """The ``--reuse-park`` section of a ``--mode check`` rehearsal (no engine)."""

    lines = [
        "park reuse  --reuse-park: one PARK per (prefix, takeover, budget) group",
        "  key       (" + ", ".join(plan.get("key_fields") or ()) + ")",
        "  cells     " + str(plan.get("cells")) + " -> "
        + str(plan.get("group_count")) + " PARK(s); without --reuse-park "
        + str(plan.get("park_calls_without_reuse"))
        + " (saved " + str(plan.get("park_saved_planned")) + ")",
        "  verified  " + str(bool(plan.get("verified")))
        + "  (every group member re-derived from its root)",
    ]
    for group in plan.get("groups") or []:
        index = group["index"]
        lines.append(
            "  group[" + str(index) + "] " + str(group["group"]) + "  "
            + str(len(group["root_indices"])) + " cell(s)  roots="
            + str(group["root_indices"]) + "  park_id=" + str(group["park_id"]))
        lines.append(
            "      prefix " + str(group["key"][0]) + "  takeover "
            + str(group["key"][1]) + "  budget " + str(group["key"][2])
            + "  window " + str(group["identity"]["window"])
            + "  expected_checkpoints "
            + str(group["identity"]["expected_checkpoints"])
            + "  target_frame " + str(group["identity"]["target_frame"]))
        tape_path = group["identity"].get("prefix_tape_path")
        if tape_path:
            lines.append("      tape " + str(tape_path))
        lines.append(
            "      data_dir " + str(group["identity"]["data_dir"])
            + "  engine_sha256 " + str(group["identity"]["engine_sha256"]))
    return "\n".join(lines)


class ParkReuseParent:
    """Transparent parent proxy: counts SPAWN/QUIT and records their failures.

    `v830_bombpol.run_episode36` (`:774-901`) already owns the protocol -- SPAWN,
    the decisions, and a mandatory QUIT in its ``finally`` -- so this proxy must
    not change one step of it.  It only

    * counts ``spawn()`` / ``quit_child()`` for the row's ``park_reuse`` block;
    * turns a SPAWN refusal / a QUIT error into a recorded fault, because
      `ForkParent.quit_child` (`v432_fork_arm.py:588-592`) **swallows** the
      exception and returns ``"ERR <type>: <msg>"`` -- without this the row would
      look clean and the work order's "quit_child 报错 -> invalid" could not hold.

    Everything else (``child``, ``child_state``, ``child_hazards``, ``alive``,
    ``window``, ``template``) is the real parent through ``__getattr__``, so the
    frozen loop sees exactly the object it always saw.
    """

    def __init__(self, parent) -> None:
        self.__dict__["_proxy_parent"] = parent
        self.__dict__["spawned"] = 0
        self.__dict__["spawn_failed"] = 0
        self.__dict__["quit_failed"] = 0
        self.__dict__["faults"] = []

    def spawn(self):
        try:
            handle, reply = self._proxy_parent.spawn()
        except BaseException as exc:                              # noqa: BLE001
            self.spawn_failed += 1
            self.faults.append("spawn failed: " + type(exc).__name__ + ": "
                               + str(exc))
            raise
        self.spawned += 1
        return handle, reply

    def quit_child(self, handle: int):
        try:
            reply = self._proxy_parent.quit_child(handle)
        except BaseException as exc:                              # noqa: BLE001
            self.quit_failed += 1
            text = "quit_child failed: " + type(exc).__name__ + ": " + str(exc)
            self.faults.append(text)
            return "ERR " + type(exc).__name__ + ": " + str(exc)
        if str(reply).startswith("ERR"):
            self.quit_failed += 1
            self.faults.append("quit_child failed: " + str(reply))
        return reply

    @property
    def children_failed(self) -> int:
        """SPAWN refusals plus QUIT errors -- the row's ``children_failed``."""

        return int(self.spawn_failed) + int(self.quit_failed)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_proxy_parent"], name)


class ParkReuseSession:
    """Group -> **one** PARK -> one spawned child per (cell, arm).

    ``enabled=False`` is the legacy path expressed in the same code: the plan has
    one singleton group per cell, `acquire` parks that cell, `end_cell` closes it,
    no proxy is built and no ``park_reuse`` key reaches a row -- which is why the
    default run stays byte-identical.

    The work order's hard constraints, all enforced here:

    * **one PARK per group**: `acquire` calls the injected ``park_fn`` only while
      ``group["park"] is None``.  A SPAWN refusal / QUIT error is recorded and
      returned; nothing in this class ever parks a second time (`run_arms`);
    * **no cross-group reuse**: `acquire` closes the parent of the previously open
      group before a new group can park, and a group id that is not the plan's is
      refused, so a spawned child can never leak into another group;
    * **a group is all-or-nothing**: `void_group` marks the whole group void, the
      remaining cells are skipped through `skip`, and ``mode_run`` exits non-zero
      -- never "half a group";
    * **at most one parent alive**: group-contiguous execution (`plan["order"]`)
      plus `close_group` at the last cell of a group.
    """

    def __init__(self, *, plan: dict, park_fn=None, log_fn=log) -> None:
        self.plan = plan
        self.enabled = bool(plan.get("enabled"))
        self.park_fn = park_fn
        self.log = log_fn
        self.groups = {group["group"]: group for group in plan.get("groups") or []}
        self.open_group_id: str | None = None
        self.park_calls = 0
        self.close_calls = 0
        self.cell_faults: list[dict] = []

    # -- grouping ---------------------------------------------------------
    def group_for(self, root: dict) -> dict:
        identifier = self.plan["by_root"][str(int(root["_index"]))]
        return self.groups[identifier]

    def is_last_cell(self, root: dict) -> bool:
        indices = self.group_for(root)["root_indices"]
        return bool(indices) and indices[-1] == int(root["_index"])

    # -- one PARK per group ------------------------------------------------
    def acquire(self, root: dict, *, attempt: int = 1, park_fn=None) -> dict:
        """The parent of this cell's group, starting it only on the first cell.

        The returned dict is ``park_parent()``'s own block plus ``_owns``: ``True``
        for the cell that actually called ``start_parent``, ``False`` for every
        cell that joined it.  The block is shared, so ``parent_gate`` /
        ``gate_name`` are literally the same PARK for the whole group.
        """

        fn = park_fn or self.park_fn
        if fn is None:
            raise PairError(
                "PARK_REUSE_NO_PARK_FN",
                "ParkReuseSession.acquire needs a park callable (a start_parent "
                "wrapper); refusing rather than silently starting nothing")
        group = self.group_for(root)
        if group["void"]:
            raise PairError(
                "PARK_REUSE_GROUP_VOID",
                "group " + str(group["group"]) + " is void ("
                + str(group.get("void_reason")) + "); roots"
                + str(group["root_indices"]) + " share one PARK, so no cell of "
                "it may run -- refusing instead of PARKing it again")
        if self.enabled and group["park"] is not None:
            # exactly the reuse: this cell joins the group's PARK, nothing starts
            return dict(group["park"], _owns=False)
        if self.open_group_id is not None and self.open_group_id != group["group"]:
            self.close_group(self.open_group_id, reason="a new group parks")
        # NB: with the flag OFF every acquire parks, even a second time for the
        # same cell -- that is the pre-registered re-run (`run_pair_with_retry`
        # calls attempt_once again with attempt=2), and reusing the attempt-1
        # parent there would hand a CLOSED parent to the re-run.
        block = dict(fn(int(attempt)))
        block["_owns"] = True
        group["park"] = block
        group["park_calls"] = int(group["park_calls"]) + 1
        group["park_pid"] = (block.get("gate") or {}).get("pid")
        group["park_gate_name"] = block.get("gate_name")
        self.park_calls += 1
        self.open_group_id = group["group"]
        return block

    def close_group(self, group_id: str, *, reason: str) -> None:
        """Close the group's parent (once); called at its last cell or on void."""

        group = self.groups.get(group_id)
        if group is None:
            return
        if group.get("park") is None or group.get("closed"):
            return
        group["closed"] = True
        group["closed_reason"] = str(reason)
        parent = group["park"].get("parent")
        try:
            parent.close()
        except Exception as exc:                                  # noqa: BLE001
            group["close_error"] = type(exc).__name__ + ": " + str(exc)
            self.log("park reuse: group " + str(group_id)
                     + ": parent.close() failed: " + str(group["close_error"]))
        self.close_calls += 1
        if self.open_group_id == group_id:
            self.open_group_id = None

    def end_cell(self, root: dict) -> bool:
        """Close the group after its **last** cell; True when this closed it."""

        if not self.is_last_cell(root):
            return False
        group = self.group_for(root)
        self.close_group(group["group"], reason="last cell of the group")
        return True

    # -- void / skip -------------------------------------------------------
    def void_group(self, root: dict, *, reason: str) -> dict:
        """Void the whole group: its remaining cells are skipped, exit non-zero."""

        group = self.group_for(root)
        group["void"] = True
        group["void_reason"] = str(reason)
        self.log("park reuse: group " + str(group["group"]) + " VOID ("
                 + str(reason) + "); roots" + str(group["root_indices"])
                 + " are one PARK and are voided together")
        self.close_group(group["group"], reason="void: " + str(reason))
        return group

    def skip(self, root: dict) -> str | None:
        """Non-``None`` when this cell must not run because its group is void."""

        group = self.group_for(root)
        if not group["void"]:
            return None
        index = int(root["_index"])
        if index not in group["skipped"]:
            group["skipped"].append(index)
            self.log("park reuse: roots[" + str(index) + "] SKIPPED -- group "
                     + str(group["group"]) + " is void ("
                     + str(group.get("void_reason")) + ")")
        return str(group.get("void_reason"))

    # -- the arms of one cell ---------------------------------------------
    def run_arms(self, root: dict, park: dict, *, episodes: int, arms: list,
                 run_arm, write_trace=None) -> dict:
        """Run every (episode, arm) of one cell on the group's parked parent.

        ``run_arm(episode, arm, parent) -> (row, trace_document_or_None)`` is
        injected: production passes a closure over `run_one_arm`, a unit test a
        fake that drives a fake parent's ``spawn``/``child``/``step``.  The parent
        handed over is a `ParkReuseParent` proxy when the flag is on (so SPAWN /
        QUIT are counted and their failures recorded) and the raw parent when it
        is off.

        A fault never re-PARKs: a SPAWN refusal / QUIT error is written into the
        row (``invalid``/``error``/``score`` re-derived) and the cell keeps its
        group; an exception that escapes ``run_arm`` means the child table can no
        longer be assumed clean, so the whole group is voided instead.

        With the flag **off** nothing is caught here: an exception propagates
        exactly as it did before, and the caller keeps the legacy
        ``parent.close()`` in its own ``finally``.
        """

        parent = park["parent"]
        group = self.group_for(root)
        rows: list[dict] = []
        traces: list[str] = []
        faults: list[dict] = []
        spawned = 0
        failed = 0

        def seal(row, trace, episode, arm):
            """Add the ``park_reuse`` block, then re-derive the trace's digest."""

            if trace is None:
                return None
            if self.enabled:
                trace["row_sha256"] = hashlib.sha256(
                    json.dumps(row, ensure_ascii=False, sort_keys=True,
                               default=str).encode("utf-8")).hexdigest()
                trace["park_reuse"] = dict(row.get("park_reuse") or {})
            if write_trace is None:
                return None
            return write_trace(row, trace, episode, arm)

        for episode in range(int(episodes)):
            for arm in arms:
                if not self.enabled:
                    # the legacy arm body, untouched: same call, same object,
                    # same exception surface, no counters, no extra row key
                    row, trace = run_arm(int(episode), arm, parent)
                    path = seal(row, trace, int(episode), arm)
                    if path is not None:
                        traces.append(str(path))
                    rows.append(row)
                    continue
                proxy = ParkReuseParent(parent)
                try:
                    row, trace = run_arm(int(episode), arm, proxy)
                except BaseException as exc:                      # noqa: BLE001
                    text = type(exc).__name__ + ": " + str(exc)
                    faults.append({"episode": int(episode), "arm": arm,
                                   "phase": "arm", "error": text})
                    self.cell_faults.append({"root_index": int(root["_index"]),
                                             "episode": int(episode),
                                             "arm": arm, "error": text})
                    group["cell_faults"] = list(group.get("cell_faults") or [])
                    group["cell_faults"].append(
                        {"episode": int(episode), "arm": arm, "phase": "arm",
                         "error": text})
                    spawned += int(proxy.spawned)
                    failed += int(proxy.children_failed)
                    self.void_group(
                        root, reason="arm " + str(arm) + " raised: " + text)
                    group["children_spawned"] = (int(group["children_spawned"])
                                                 + spawned)
                    group["children_failed"] = (int(group["children_failed"])
                                                + failed)
                    return {"rows": rows, "traces": traces, "faults": faults,
                            "children_spawned": spawned,
                            "children_failed": failed, "invalid": True,
                            "error": text, "voided": True}
                spawned += int(proxy.spawned)
                failed += int(proxy.children_failed)
                if proxy.faults:
                    row["invalid"] = True
                    row["error"] = "; ".join(proxy.faults)
                    row["score"] = score(row)
                    faults.append({"episode": int(episode), "arm": arm,
                                   "phase": "spawn/quit",
                                   "error": row["error"]})
                    self.cell_faults.append(
                        {"root_index": int(root["_index"]),
                         "episode": int(episode), "arm": arm,
                         "error": row["error"]})
                    group["cell_faults"] = list(group.get("cell_faults") or [])
                    group["cell_faults"].append(
                        {"episode": int(episode), "arm": arm,
                         "phase": "spawn/quit", "error": row["error"]})
                row["park_reuse"] = {
                    "group": group["group"],
                    "park_id": group["park_id"],
                    "children_spawned": int(proxy.spawned),
                    "children_failed": int(proxy.children_failed),
                }
                path = seal(row, trace, int(episode), arm)
                if path is not None:
                    traces.append(str(path))
                rows.append(row)
                group["rows"] = int(group["rows"]) + 1
        group["children_spawned"] = int(group["children_spawned"]) + spawned
        group["children_failed"] = int(group["children_failed"]) + failed
        return {"rows": rows, "traces": traces, "faults": faults,
                "children_spawned": spawned, "children_failed": failed,
                "invalid": bool(faults), "error": None, "voided": False}

    # -- record ------------------------------------------------------------
    def document(self) -> dict:
        return park_reuse_document(self.plan)


def mode_run(args: argparse.Namespace, *, guard_doc: dict, manifest: dict,
             roots: list[dict], plan: dict, single_arm: str | None) -> int:
    """``pair`` and ``trace``: one PARK, then 1 or 2 arms per episode."""

    global _LOG_FH

    # 1. audio layer 1 (AGENTS.md sec. 9.3 #2) before anything can spawn
    env = dict(m450().prepare_env())
    if env.get("SDL_AUDIODRIVER") != "dummy":
        raise PairError("AUDIO_ENV_NOT_MUTED",
                        f"v450.prepare_env() did not pin SDL_AUDIODRIVER=dummy: "
                        f"{env}")

    # 2. every protocol refusal first (window / takeover / budget / exposure /
    #    legacy opt-in / tag / rng / checkpoint): none of these needs the engine,
    #    so a mis-registered cell is reported as itself instead of as
    #    "engine missing" on a host that cannot see the binary.
    work = pathlib.Path(args.work_dir)
    tag = str(args.tag or "").strip()
    if not tag:
        raise PairError("TAG_REQUIRED", "--tag is mandatory for a run (MAP sec. 9-3)")
    require_legacy_opt_in(args, roots)
    for root in roots:
        check_root(root, args, needs_engine=True)
        resolve_pair_rng(root, args, roots)
        resolve_ckpt(root, args)
    run_dir = work / "runs" / tag
    if run_dir.exists() or (work / "data" / tag).exists():
        raise PairError("TAG_NOT_UNIQUE",
                        f"--tag {tag!r} already exists under {work}")
    out = out_for(args, tag)
    log_path = log_for(args, tag)
    assert_writable(out, work_dir=work, what="--out")
    assert_writable(log_path, work_dir=work, what="--log")
    if out.exists():
        raise PairError("OUT_EXISTS", f"--out {out} already exists")

    # 2a. --reuse-park: group the cells **before** anything is started.  With the
    #     flag off the plan is the degenerate one (a singleton group per cell), so
    #     the session below is the same code with sharing disabled: one PARK per
    #     cell, no proxy, no `park_reuse` key -- byte-identical to the old run.
    reuse_plan = park_reuse_plan(roots, work=work, tag=tag,
                                 enabled=park_reuse_enabled(args))
    session = ParkReuseSession(plan=reuse_plan, log_fn=log)
    by_root_index = {int(root["_index"]): root for root in roots}
    ordered_roots = (list(roots) if not session.enabled else
                     [by_root_index[index] for index in reuse_plan["order"]])

    # 2b. --out/--log are files this run creates, but their directories are not:
    #     a fresh work dir has no eval/ yet.  r1 parked the parent, ran an arm,
    #     and only then out.open("a") raised FileNotFoundError (exit 2,
    #     UNEXPECTED).  Create the parents here, before any spawn; an OSError is
    #     not swallowed (main() still fails closed, exit 2 as before).
    ensure_parent_dir(out)
    ensure_parent_dir(log_path)

    # 3. engine digest (never relaxed), then the frozen prefix constants **as they
    #    are on disk** (this is the drift gate; it must run before any override
    #    below, otherwise a patched global would mask an edited file), then the
    #    override itself for the first root (the loop re-applies per root).
    engine, engine_source = resolve_engine(args, manifest)
    if not pathlib.Path(engine).is_file():
        raise PairError("ENGINE_MISSING",
                        f"engine binary not found: {engine} (source: "
                        f"{engine_source}); refusing before any spawn")
    try:
        gate = dict(m450().check_engine(engine))
    except Exception as exc:                                      # noqa: BLE001
        raise PairError("ENGINE_SHA_MISMATCH",
                        f"{engine}: {type(exc).__name__}: {exc}") from exc
    model_engine = str(roots[0].get("engine_sha256"))
    if str(gate.get("sha256")) != model_engine:
        raise PairError(
            "ENGINE_SHA_MISMATCH",
            f"{engine} sha256 {str(gate.get('sha256'))[:16]}... != manifest "
            f"declared {model_engine[:16]}...")
    pristine = assert_prefix_constants()
    data_source, data_source_source = resolve_data_source(args, manifest)
    if not pathlib.Path(data_source).is_dir():
        raise PairError("DATA_SOURCE_MISSING",
                        f"engine data source not found: {data_source}")

    run_dir.mkdir(parents=True, exist_ok=True)
    _LOG_FH = log_path.open("a", encoding="utf-8")
    try:
        log(f"mode={args.mode} tag={tag} engine={engine} ({engine_source})")
        # the --budget-frames selector's result, once, on stdout: only a mixed-tier
        # invocation prints it (see preflight's docstring), so a single-tier run's
        # output is byte-identical
        budget_block = dict(plan.get("budget_selection") or {})
        if budget_block:
            print(budget_selection_line(budget_block), flush=True)
        # The override is applied **after** the log handle exists and before the
        # first spawn: `prefix_module_objects()` loads the nested copies lazily, and
        # they capture their `_LOG_FH` at load time, so applying it earlier would
        # silently drop their lines from run.log.
        constants = apply_prefix_override(
            int(roots[0]["takeover_frame"]), window=int(roots[0]["window"]),
            source=(roots[0].get("_takeover_resolution") or {}).get("source",
                                                                    "manifest"))
        log(f"prefix constants: takeover={constants['takeover_frame']} "
            f"expected_checkpoints={constants['expected_checkpoints']} "
            f"target_frame={constants['target_frame']} "
            f"override_applied={constants['override_applied']} "
            f"modules={len(constants['overridden_modules'])} "
            f"files_on_disk_unmodified={constants['files_on_disk_unmodified']}")
        closure = dict(plan.get("manifest_closure") or {})
        log(f"manifest {args.manifest} sha256 "
            f"{closure.get('manifest_sha256', guard_doc.get('manifest_sha256'))} "
            f"({closure.get('manifest_bytes')} B, "
            f"source={closure.get('manifest_sha256_source')}, "
            f"reads={closure.get('manifest_reads')})")
        write_json(run_dir / "guard.json", guard_doc)
        write_json(run_dir / "manifest-bytes.json", closure)
        write_json(run_dir / "plan.json", plan)
        write_json(run_dir / "audio-env.json", {
            "schema": SCHEMA_AUDIO + ":env", "prepare_env": env,
            "sdl_audio_driver": "dummy",
            "layers": plan["audio"],
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")})

        arm_order = ([single_arm] if single_arm else list(args.arm_order))
        budget_frames = int(roots[0]["budget_frames"])
        if session.enabled:
            log(f"park reuse: {reuse_plan['cells']} cell(s) -> "
                f"{reuse_plan['group_count']} PARK(s), one per (prefix, "
                f"takeover, budget) group; without the flag it would be "
                f"{reuse_plan['park_calls_without_reuse']} (saved "
                f"{reuse_plan['park_saved_planned']}); rows are written in "
                f"group order")
            for group in reuse_plan["groups"]:
                log(f"park reuse: group {group['group']} roots"
                    f"{group['root_indices']} prefix={group['key'][0]} "
                    f"takeover={group['key'][1]} budget={group['key'][2]} "
                    f"park_id={group['park_id']}")
        lineage = dict(plan.get("lineage") or {})
        lineage_by_root = {entry["root_index"]: entry
                           for entry in lineage.get("roots") or []}
        summary: dict = {
            "schema": SCHEMA_RUN, "tool": TOOL, "tool_version": TOOL_VERSION,
            "mode": args.mode, "tag": tag, "manifest_path": str(args.manifest),
            "manifest_sha256": closure.get("manifest_sha256",
                                           guard_doc.get("manifest_sha256")),
            "manifest_bytes": closure.get("manifest_bytes",
                                          guard_doc.get("manifest_bytes")),
            "manifest_sha256_source": closure.get("manifest_sha256_source"),
            "manifest_sha256_guard": closure.get("manifest_sha256_guard"),
            "manifest_reads": closure.get("manifest_reads"),
            "manifest_read_scope": closure.get("manifest_read_scope"),
            "manifest_closure": closure,
            "lineage_checked": bool(lineage.get("lineage_checked")),
            "lineage_config_path": lineage.get("lineage_config_path"),
            "lineage_config_sha256": lineage.get("lineage_config_sha256"),
            "lineage_config_bytes": lineage.get("lineage_config_bytes"),
            "lineage_sources_path": lineage.get("lineage_sources_path"),
            "lineage_sources_sha256": lineage.get("lineage_sources_sha256"),
            "lineage_sources_checked": lineage.get("lineage_sources_checked"),
            "lineage_splits": lineage.get("lineage_splits"),
            "lineage_codes": list(lineage.get("lineage_codes") or []),
            "lineage_unmatched": list(lineage.get("lineage_unmatched") or []),
            "lineage": lineage,
            "residual_risk": (list(closure.get("residual_risk") or [])
                              + list(lineage.get("residual_risk") or [])),
            "engine": {"path": engine, "source": engine_source,
                       "gate": gate, "declared": model_engine},
            "data_source": {"path": data_source, "source": data_source_source},
            "prefix_constants": constants,
            "prefix_constants_pristine": pristine,
            "prefix_constants_by_root": {},
            "budget_frames": budget_frames,
            "budget_decisions": budget_frames // DECISION_FRAMES,
            "decision_frames": DECISION_FRAMES,
            "arm_order": arm_order, "episodes": int(args.episodes),
            "allow_exposed": bool(args.allow_exposed),
            "allow_legacy_window": bool(args.allow_legacy_window),
            "out": str(out), "log": str(log_path),
            "roots": [], "rows": 0, "rng": [],
            # pre-registration sec. 4: one technical fault re-runs the whole pair
            # once, so a pair is 1 or 2 attempts; both stay in the jsonl
            "attempts": [], "rows_final": 0, "superseded_rows": 0,
        }
        new_roots = list(lineage.get("lineage_new_roots") or [])
        if new_roots:
            # only a `lineage_new` invocation gets this key, and it is the
            # machine-readable half of residual_risk RESIDUAL_LINEAGE_NEW
            summary["lineage_new_roots"] = new_roots
        if budget_block:
            # only a mixed-tier invocation gets this key (selected>0, skipped>0),
            # so a single-tier run.json keeps exactly the shape it had; the roots
            # this run did NOT touch are named in skipped_roots
            summary["budget_selection"] = budget_block
        if session.enabled:
            # only an --reuse-park run gets this key, so a default run.json keeps
            # exactly the shape it had
            summary["park_reuse"] = session.document()

        exit_code = EXIT_OK
        all_rows: list[dict] = []
        final_rows_all: list[dict] = []
        for root in ordered_roots:
            if session.enabled:
                voided = session.skip(root)
                if voided is not None:
                    # hard constraint #3: a void group is void as a whole, so its
                    # remaining cells are named in run.json and never run
                    summary["park_reuse"] = session.document()
                    exit_code = EXIT_REFUSED
                    continue
            window = int(root["window"])
            budget = int(root["budget_frames"]) // DECISION_FRAMES
            ckpt_path, ckpt_sha = resolve_ckpt(root, args)
            pair_rng, pair_rng_source = resolve_pair_rng(root, args, roots)
            label_of = {arm: rule_label_for(arm)[0] for arm in arm_order}
            # re-applied per root (idempotent): the pinned globals are
            # process-wide, so a second root with a different takeover frame must
            # re-point them before *its* PARK, not inherit the previous root's
            constants = apply_prefix_override(
                int(root["takeover_frame"]), window=window,
                source=(root.get("_takeover_resolution") or {}).get("source",
                                                                    "manifest"),
                tape=((root.get("_prefix_tape_state") or {}).get("scan")))
            summary["prefix_constants_by_root"][str(root["_index"])] = constants
            tape_block: dict | None = None
            if root.get("_prefix_tape") is not None:
                # 3b. the tape itself, after the log handle exists (load_module
                #     captures _LOG_FH at load time) and before the first spawn:
                #     start_parent must receive read_tape()'s tuple, not
                #     build_masks' S1c shard
                tape_block = inject_prefix_tape(root)
                log(f"w{window}: prefix_tape {tape_block['path']} sha256 "
                    f"{tape_block['sha256']} rows={tape_block['rows']} "
                    f"frames_built={tape_block['frames_built']} "
                    f"target={tape_block['target']} "
                    f"injected_into={len(tape_block['injected_into'])} object(s) "
                    f"files_on_disk_unmodified="
                    f"{tape_block['files_on_disk_unmodified']}")
                summary.setdefault("prefix_tape_by_root",
                                   {})[str(root["_index"])] = tape_block
                summary["prefix_tape"] = tape_block
            # One private data dir per GROUP under --reuse-park: the cells of a
            # group share the window, hence the same w<window> dir, and
            # `build_private_data_dir` regenerates th08.cfg on every call
            # (`v432:378-447`, "called before every engine start").  A reused
            # cell starts no engine, so calling it again would rewrite the cfg
            # of the parent that is still running; the generation is
            # deterministic (same 60 bytes / same sha256 on every call, measured),
            # so the row's data_cfg is the same either way.
            cached_dir = (session.group_for(root).get("data_dir_info")
                          if session.enabled else None)
            if cached_dir is not None:
                data_dir, cfg_info = cached_dir
                log(f"w{window}: data dir {data_dir} cfg "
                    f"{cfg_info['cfg']['sha256'][:16]} muted_four_zero=True "
                    f"(reused from group "
                    f"{session.group_for(root)['group']}; no second engine "
                    f"start, so the cfg is not regenerated under a live parent)")
            else:
                try:
                    data_dir, cfg_info = m450().prepare_data_dir(
                        work / "data" / tag, window, data_source)
                except PairError:
                    raise
                except Exception as exc:                          # noqa: BLE001
                    raise PairError(
                        "DATA_DIR_FAILED",
                        f"w{window}: v450.prepare_data_dir({work / 'data' / tag}, "
                        f"{window}, {data_source}) failed: {type(exc).__name__}: {exc} "
                        f"-- the private data dir is never hand-built (MAP sec. 9-8) "
                        f"because that is how the 105 audible th08.cfg files appeared"
                    ) from exc
                if not cfg_info["cfg"]["muted_four_zero"]:
                    raise PairError(
                        "AUDIO_CFG_NOT_MUTED",
                        f"w{window}: the generated th08.cfg is not muted: "
                        f"{cfg_info['cfg']} (AGENTS.md sec. 9.3 #1)")
                log(f"w{window}: data dir {data_dir} cfg "
                    f"{cfg_info['cfg']['sha256'][:16]} muted_four_zero=True")
                if session.enabled:
                    session.group_for(root)["data_dir_info"] = (data_dir, cfg_info)

            # checkpoint first, generator second (MAP sec. 9-2 ordering)
            try:
                policy = m860().load_policy36(pathlib.Path(ckpt_path),
                                              seed=int(pair_rng))
            except PairError:
                raise
            except Exception as exc:                              # noqa: BLE001
                raise PairError(
                    "CKPT_LOAD_FAILED",
                    f"{ckpt_path}: v860.load_policy36 failed: "
                    f"{type(exc).__name__}: {exc}") from exc
            log(f"w{window}: loaded {ckpt_path} sha {str(ckpt_sha)[:16]} "
                f"(policy_seed=pair_rng={pair_rng})")

            retry_state: dict = {}

            def attempt_once(attempt: int, *, _window=window, _pair_rng=pair_rng,
                             _ckpt_path=ckpt_path, _ckpt_sha=ckpt_sha,
                             _policy=policy, _data_dir=data_dir,
                             _cfg_info=cfg_info,
                             _budget=budget) -> dict:
                """One *complete* pair attempt: data dir, PARK, every arm, close.

                This is the injected ``runner`` of ``run_pair_with_retry``.  A
                re-run calls it again with ``attempt=2`` and not one parameter
                changed (window / takeover / tape / ckpt / pair_rng are closed
                over), which is exactly pre-registration sec. 4's "same pair,
                start frame, random stream and configuration".
                """

                # One PARK per attempt, injected through `park_parent`: a failure
                # at any point of HELLO -> prefix -> PARK -> DEBUG raises a
                # `RecoveryFault`, which `run_pair_with_retry` records as a
                # `phase: "recovery"` attempt and retries once (sec. 4's
                # "无法恢复").  The gate is written under its attempt-specific
                # name by `park_parent`, never here.
                #
                # With --reuse-park the PARK is the *group's*: `session.acquire`
                # starts it for the first cell of the group and hands the same
                # `{parent, gate, gate_name}` block to every later cell, so the
                # second cell's rows carry the very same parent_gate/prefix
                # constants without a second prefix walk.  `_owns` says which cell
                # actually parked (it owns the gate and the close).
                park = session.acquire(
                    root, attempt=attempt,
                    park_fn=lambda _attempt: park_parent(
                        window=_window, engine=engine, data_dir=_data_dir,
                        log_dir=run_dir, attempt=_attempt, run_dir=run_dir))
                parent = park["parent"]
                parent_gate = park["gate"]
                log(f"w{_window}: parent pid={parent.pid} PARK frame="
                    f"{parent_gate.get('park_frame')} prefix="
                    f"{parent_gate['prefix']['hash_checks']}/"
                    f"{constants['values']['v432.CHECKPOINTS_IN_PREFIX']}"
                    f" attempt={attempt} gate={park['gate_name']}"
                    + (f" reused_by_group={session.group_for(root)['group']}"
                       if session.enabled else ""))
                rows: list[dict] = []
                traces: list[str] = []
                previous = list(retry_state.get("previous_fingerprints") or [])

                def write_trace(row: dict, trace: dict, episode: int,
                                arm: str) -> str:
                    # attempt 2 gets its own name, so attempt 1's trace is kept as
                    # well ("保留两次记录")
                    path = trace_path_for(
                        out, arm, episode=episode,
                        episodes=args.episodes, attempt=attempt)
                    assert_writable(path, work_dir=work, what="trace")
                    write_json(path, trace)
                    return str(path)

                def run_arm(episode: int, arm: str, target_parent):
                    """One (episode, arm) on the group's parked parent.

                    ``target_parent`` is a `ParkReuseParent` proxy under
                    --reuse-park and the raw parent otherwise; `run_one_arm` (and
                    therefore `run_episode36`, which SPAWNs and QUITs inside it)
                    cannot tell the difference.
                    """

                    pair = {"pair_id": (f"{root.get('ancestor_id')}"
                                        f"@w{_window}"
                                        f"-b{budget_frames}-r{_pair_rng}"
                                        f"-ep{episode}"),
                            "pair_rng": int(_pair_rng),
                            "pair_rng_source": pair_rng_source,
                            "policy_seed": int(_pair_rng)}
                    supersedes: str | None = None
                    if attempt > 1:
                        index = len(rows)
                        if index < len(previous):
                            supersedes = previous[index]
                        else:
                            supersedes = retry_state.get(
                                "previous_attempt_fingerprint")
                    row, trace = run_one_arm(
                        parent=target_parent, arm=arm, pair=pair,
                        policy=_policy, root=root, args=args,
                        episode=episode, arm_order=arm_order,
                        guard_doc=guard_doc, engine=engine,
                        engine_source=engine_source,
                        engine_gate=gate, ckpt_path=pathlib.Path(
                            _ckpt_path),
                        ckpt_sha=_ckpt_sha, data_dir=_data_dir,
                        cfg_info=_cfg_info, parent_gate=parent_gate,
                        constants=constants,
                        manifest_closure=closure,
                        attempt=attempt, supersedes=supersedes)
                    row["engine_source"] = engine_source
                    log(f"{pair['pair_id']} {arm} attempt={attempt} "
                        f"full_segment={row['score']['full_segment']} "
                        f"frames={row['frames']}/{budget_frames} "
                        f"decisions={row['decisions_executed']}/{_budget}"
                        f" terminal={row['terminal']} "
                        f"emits={row['bomb_emits']} "
                        f"decrement_at={row['emit_decrement_at']} "
                        f"rng_draws={row['rng_draws']} "
                        f"invalid={row['invalid']} "
                        f"error={row['error']}")
                    return row, trace

                outcome: dict = {}
                try:
                    outcome = session.run_arms(
                        root, park, episodes=int(args.episodes),
                        arms=list(arm_order), run_arm=run_arm,
                        write_trace=write_trace)
                    rows = list(outcome.get("rows") or [])
                    traces = list(outcome.get("traces") or [])
                except BaseException as exc:                      # noqa: BLE001
                    if session.enabled:
                        # an exception that escapes the cell (a refused trace
                        # write, a torch failure inside run_one_arm, ...) leaves
                        # the child table unprovable: void the group and close its
                        # one parent before the failure goes up to main()
                        session.void_group(
                            root, reason="an exception escaped the cell: "
                                         + type(exc).__name__ + ": " + str(exc))
                    raise
                finally:
                    if not session.enabled:
                        # the legacy lifetime, unchanged: the cell's parent is
                        # closed here, before its audio audit, even if an arm
                        # raised (run_arms re-raises when the flag is off)
                        parent.close()
                if session.enabled:
                    # a parent that died after the PARK voids the whole group
                    # (hard constraint #3) -- never a silent partial group
                    group = session.group_for(root)
                    if not group["void"] and not parent.alive():
                        session.void_group(
                            root,
                            reason="the parked parent died (alive() == False) "
                                   "after the PARK")
                    session.end_cell(root)
                return {"rows": rows, "parent_gate": parent_gate,
                        "data_dir": str(_data_dir), "cfg_info": _cfg_info,
                        "traces": traces, "attempt": attempt,
                        "parked_here": bool(park.get("_owns")),
                        "park_reuse": outcome}

            if args.mode == "pair":
                # --reuse-park: a re-run would need a SECOND PARK of the same
                # group, which hard constraint #2 forbids ("spawn/quit 失败 ...
                # 不得退化为重 PARK"), so the group gets exactly one attempt.  Any
                # fault of a reused cell is therefore reported instead of retried;
                # with the flag off this is still the pre-registered two attempts.
                retry = run_pair_with_retry(
                    attempt_once, tag=tag, out=out,
                    max_attempts=(1 if session.enabled else RETRY_ATTEMPT_LIMIT),
                    context=retry_state,
                    on_attempt=lambda record, _rows: log(
                        f"w{window}: attempt {record['attempt']} finished: "
                        f"rows={record['rows']} "
                        f"invalid_rows={record['invalid_rows']} "
                        f"wall_s={record['wall_s']} "
                        f"phase={record.get('phase')}"
                        + (f" error={record['error']}"
                           if record.get("error") else "")
                        + ("  -> re-running the whole pair once "
                           "(pre-registration sec. 4)"
                           if record["attempt"] < RETRY_ATTEMPT_LIMIT
                           and (should_retry(_rows)
                                or record.get("phase") == RECOVERY_PHASE)
                           else "")))
            else:
                # `trace` runs ONE arm: a row fault has no pair to re-run, so
                # only a recovery fault retries (it produced no row yet and the
                # re-run repeats the same PARK / arm / configuration).
                retry = run_pair_with_retry(
                    attempt_once, tag=tag, out=out,
                    max_attempts=RETRY_ATTEMPT_LIMIT, context=retry_state,
                    retry_rows=False,
                    on_attempt=lambda record, _rows: log(
                        f"w{window}: attempt {record['attempt']} finished: "
                        f"rows={record['rows']} invalid_rows="
                        f"{record['invalid_rows']} wall_s={record['wall_s']} "
                        f"phase={record.get('phase')}"
                        + (f" error={record['error']}"
                           if record.get("error") else "")))
            if retry["recovery_failed"]:
                # Both attempts died inside start_parent: sec. 4 keeps both
                # records, so write run.json with the two phase="recovery"
                # entries and then fail closed (RECOVERY_FAILED, exit 1).  No
                # row exists for this root, so --out stays empty for it.
                if session.enabled:
                    # the group's ONE PARK is the one that died: the whole group
                    # is void (its later cells are skipped below by `skip`)
                    session.void_group(
                        root, reason="the group's PARK failed: "
                                     + str(retry["attempts"][-1].get("error")))
                    summary["park_reuse"] = session.document()
                for record in retry["attempts"]:
                    summary["attempts"].append(dict(
                        record, root_index=root["_index"], window=window,
                        retried=bool(retry["retried"]), traces=[],
                        **({"park_reuse_group": session.group_for(root)["group"]}
                           if session.enabled else {})))
                failure = recovery_failure(
                    retry, window=window,
                    takeover_frame=int(constants["takeover_frame"]),
                    expected_checkpoints=int(constants["expected_checkpoints"]))
                summary["outcome"] = "recovery_failed"
                summary["recovery_failure"] = {
                    "phase": RECOVERY_PHASE, "window": window,
                    "root_index": root["_index"],
                    "attempts": len(retry["attempts"]),
                    "errors": [record.get("error")
                               for record in retry["attempts"]],
                    "error": str(failure)}
                summary["all_rows_valid"] = False
                summary["rng_same_initial_state"] = None
                # attempt 1 may already have written rows to --out (a row fault
                # followed by a failed recovery); those lines stay on disk, so the
                # count must not pretend the jsonl is empty.
                summary["rows"] += len(retry["all_rows"])
                summary["invalid_rows"] = sum(1 for row in final_rows_all
                                              if row["invalid"])
                summary["full_segments"] = sum(1 for row in final_rows_all
                                               if row["score"]["full_segment"])
                summary["rows_final"] = len(final_rows_all)
                summary["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                write_json(run_dir / "run.json", summary)
                log(f"w{window}: recovery failed on all "
                    f"{len(retry['attempts'])} attempt(s) -> "
                    f"REFUSED [RECOVERY_FAILED]; {run_dir / 'run.json'} keeps "
                    f"all {len(retry['attempts'])} record(s)")
                raise failure
            rows = list(retry["rows"])
            meta = retry["last"] if isinstance(retry["last"], dict) else {}
            parent_gate = dict(meta.get("parent_gate") or {})
            data_dir = meta.get("data_dir", str(data_dir))
            outcome = dict(meta.get("park_reuse") or {})
            if session.enabled and outcome.get("invalid"):
                # a cell fault that produced no invalid ROW (a SPAWN refusal, a
                # QUIT error, or an arm that raised) must still fail the run
                exit_code = EXIT_REFUSED
            all_rows.extend(retry["all_rows"])
            final_rows_all.extend(rows)
            summary["rows"] += len(retry["all_rows"])
            summary["rows_final"] += len(rows)
            for record in retry["attempts"]:
                summary["attempts"].append(dict(
                    record, root_index=root["_index"], window=window,
                    retried=bool(retry["retried"]),
                    traces=list(meta.get("traces") or [])
                    if record["attempt"] == len(retry["attempts"]) else [],
                    **({"park_reuse_group": session.group_for(root)["group"]}
                       if session.enabled else {})))
            if retry["retried"]:
                first = retry["attempts"][0]
                if first.get("phase") == RECOVERY_PHASE:
                    log(f"w{window}: attempt 1 failed in recovery "
                        f"({first.get('error')}) -> the same PARK / pair / "
                        f"window / takeover / tape / ckpt / pair_rng was re-run "
                        f"once; the new PARK is "
                        f"{gate_name_for(window, 2)}")
                else:
                    log(f"w{window}: technical fault on attempt 1 -> the whole "
                        f"pair was re-run once with the same manifest/window/"
                        f"takeover/tape/ckpt/pair_rng; rows of attempt 1 are "
                        f"kept in {out}")
            if retry["invalid"]:
                exit_code = EXIT_REFUSED
            for episode_row in rows:
                summary["rng"].append({
                    "pair_id": episode_row.get("pair_id"),
                    "arm": episode_row.get("arm"),
                    "episode": episode_row.get("episode"),
                    "attempt": episode_row.get("attempt"),
                    "generator_seed": episode_row.get("generator_seed"),
                    "generator_initial_digest":
                        episode_row.get("generator_initial_digest"),
                    "generator_final_digest":
                        episode_row.get("generator_final_digest"),
                    "rng_draws": episode_row.get("rng_draws")})

            audit = audio_audit(work / "data" / tag, tag=tag,
                                out_path=run_dir / f"audio_audit-w{window}.json")
            log(f"w{window}: audio audit {audit['outcome']} "
                f"cfg_files={audit['counts']['cfg_files']} "
                f"muted={audit['counts']['muted']} "
                f"audible={audit['counts']['audible']}")
            write_json(run_dir / "audio_audit.json", audit)
            if not audit["ok"]:
                exit_code = EXIT_REFUSED
            root_summary = {
                "root_index": root["_index"],
                "ancestor_id": root.get("ancestor_id"), "window": window,
                "budget_frames": budget_frames,
                "budget_decisions": budget,
                "pair_rng": int(pair_rng), "pair_rng_source": pair_rng_source,
                "ckpt_path": str(ckpt_path), "ckpt_sha256": ckpt_sha,
                "data_dir": str(data_dir),
                "park": {key: parent_gate.get(key) for key in
                         ("park_frame", "park_ok", "threads_1",
                          "hello_role_shadow", "all_pass")},
                "prefix": parent_gate.get("prefix"),
                "prefix_constants": constants,
                "prefix_recording": prefix_gate(root),
                "lineage_split": lineage_by_root.get(root["_index"], {})
                .get("lineage_split"),
                "lineage_status": lineage_by_root.get(root["_index"], {})
                .get("lineage_status"),
                "lineage_unmatched": list(lineage_by_root.get(root["_index"], {})
                                          .get("lineage_unmatched") or []),
                "lineage_code": lineage_by_root.get(root["_index"], {}).get("code"),
                "audio_audit": {key: audit[key] for key in
                                ("ok", "outcome", "counts")},
                "rule_labels": label_of,
                "rows": len(rows),
                "attempt": (int(rows[0]["attempt"])
                            if rows and rows[0].get("attempt") is not None
                            else None),
                "attempts_used": len(retry["attempts"]),
                "retried": bool(retry["retried"]),
                # True when attempt 1 died in start_parent and attempt 2 was the
                # fresh PARK (its gate is gate-w<w>-a2.json)
                "recovery_retried": bool(
                    retry["attempts"][0].get("phase") == RECOVERY_PHASE),
            }
            if retry["retried"]:
                root_summary["superseded_rows"] = int(retry["superseded_rows"])
            if tape_block is not None:
                # what actually fed the parent's prefix: read_tape()'s meta is
                # also echoed by v450.start_parent into gate["source"] (gate-w<w>
                # .json), so the two records can be compared
                root_summary["prefix_tape"] = tape_block
                root_summary["prefix_tape_source_meta"] = parent_gate.get("source")
            if session.enabled:
                # the cell's own view of the shared PARK (run.json keeps the
                # authoritative per-group block below)
                root_summary["park_reuse"] = {
                    "group": session.group_for(root)["group"],
                    "park_id": session.group_for(root)["park_id"],
                    "parked_here": bool(meta.get("parked_here")),
                    "children_spawned": int(outcome.get("children_spawned") or 0),
                    "children_failed": int(outcome.get("children_failed") or 0),
                    "void": bool(session.group_for(root)["void"]),
                }
            summary["roots"].append(root_summary)

        initial = [entry["generator_initial_digest"] for entry in summary["rng"]]
        summary["rng_same_initial_state"] = bool(initial and len(set(initial)) == 1)
        # The authoritative rows are the LAST attempt of every root: a superseded
        # attempt-1 row is kept in the jsonl (and in summary["attempts"]) but it
        # must not decide all_rows_valid / full_segments.
        summary["all_rows_valid"] = all(not row["invalid"]
                                        for row in final_rows_all)
        summary["invalid_rows"] = sum(1 for row in final_rows_all if row["invalid"])
        summary["full_segments"] = sum(1 for row in final_rows_all
                                      if row["score"]["full_segment"])
        summary["superseded_rows"] = len(all_rows) - len(final_rows_all)
        if session.enabled:
            # the authoritative per-group block: one PARK per group, how many
            # children each group spawned / failed, and which groups are void
            summary["park_reuse"] = session.document()
            if summary["park_reuse"]["void_groups"]:
                exit_code = EXIT_REFUSED
        summary["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json(run_dir / "run.json", summary)
        log(f"done: {summary['rows']} row(s) in {out} "
            f"({summary['rows_final']} in the final attempt), "
            f"full_segments={summary['full_segments']}, "
            f"invalid={summary['invalid_rows']}, "
            f"attempts={len(summary['attempts'])}, "
            f"same_initial_rng={summary['rng_same_initial_state']} -> "
            f"{run_dir}")
        if session.enabled:
            log(f"park reuse: {session.park_calls} PARK(s) for "
                f"{reuse_plan['cells']} cell(s) (one per group), "
                f"children_spawned="
                f"{summary['park_reuse']['children_spawned']} "
                f"children_failed="
                f"{summary['park_reuse']['children_failed']} "
                f"void_groups={summary['park_reuse']['void_groups']}")
        return exit_code
    finally:
        if _LOG_FH is not None:
            _LOG_FH.flush()
            _LOG_FH.close()
            _LOG_FH = None


# --------------------------------------------------------------------------- #
# --workers: one process per pair cell, one unique tag per process
#            (pre-registration sec. 3.4)
# --------------------------------------------------------------------------- #
#: "各工作进程唯一 tag": every worker of one invocation owns a tag of its own,
#: and therefore its own ``runs/<tag>-w<i>/`` and ``data/<tag>-w<i>/``.  Two
#: workers must never share either -- that is precisely the v8.17.0 accident
#: (18 launches, one timestamp data dir, 10 of 18 cells corrupted, MAP sec. 9-3).
#:
#: Merge choice (written here, in ``--help`` and in the module docstring, because
#: the two are both defensible): the workers write **one jsonl per worker**
#: (``<work>/eval/<tag>-w<i>.jsonl``) and the coordinator concatenates them into
#: the single ``--out`` file in cell order.  ``run.json`` lists every worker jsonl
#: next to the merged one, so the merge can be re-derived (and re-checked) from
#: the per-worker files: nothing is deleted after the merge.
def work_cells(roots: list[dict]) -> list[dict]:
    """One cell per selected root: ``(root, budget_frames, ckpt_sha256)``.

    Pre-registration sec. 3.4 names a cell as "one root x one budget x one ckpt".
    ``v829_guard`` pins exactly one ``budget_frames`` and one ``ckpt_sha256`` per
    root, and ``--budget-frames`` is a **selector** that keeps only the roots
    already declared at that tier, so the cartesian product collapses to one cell
    per root (a root of another tier is not in ``roots`` at all -- it was skipped
    by ``select_budget_tier`` and is named in ``budget_selection.skipped_roots``).
    ``check_root`` still keeps ``BUDGET_ROOT_MISMATCH`` as a defensive assertion
    for a root set built by hand.  The key is spelled out in full, which is what
    makes two roots that differ only by window (or only by ckpt) provably
    different cells.
    """

    cells: list[dict] = []
    for index, root in enumerate(roots):
        window = int(root["window"])
        budget = int(root["budget_frames"])
        ckpt = str(root.get("ckpt_sha256"))
        cells.append({
            "cell_index": index,
            "cell_key": (f"w{window}-b{budget}-ckpt{ckpt[:12]}"
                         f"-root{int(root['_index'])}"),
            "root_index": int(root["_index"]),
            "ancestor_id": root.get("ancestor_id"),
            "window": window,
            "budget_frames": budget,
            "ckpt_sha256": ckpt,
            "_root": root,
        })
    return cells


def worker_tag(tag: str, cell: dict) -> str:
    """``<tag>-w<i>`` -- the 1-based worker/cell suffix (sec. 3.4 uniqueness)."""

    return f"{tag}-w{int(cell['cell_index']) + 1}"


def plan_workers(cells: list[dict], *, workers: int, tag: str,
                 work_dir: pathlib.Path) -> list[dict]:
    """Partition the cells into batches of at most ``workers`` workers.

    One cell = one worker, so ``ceil(len(cells) / workers)`` batches: four cells
    at ``--workers 2`` are exactly two batches of two.  A worker never handles two
    cells (that would put two cells in one ``data/<tag>``), and a batch never
    exceeds ``workers`` processes.
    """

    workers = int(workers)
    if workers < 1 or workers > MAX_WORKERS:
        raise PairError("WORKERS_OUT_OF_RANGE",
                        f"--workers must be 1..{MAX_WORKERS}, got {workers}")
    work = pathlib.Path(work_dir)
    specs: list[dict] = []
    for cell in cells:
        index = int(cell["cell_index"])
        worker = f"{tag}-w{index + 1}"
        specs.append({
            "worker_index": index + 1,
            "batch": index // workers,
            "tag": worker,
            "cell_key": cell["cell_key"],
            "root_index": cell["root_index"],
            "ancestor_id": cell["ancestor_id"],
            "window": cell["window"],
            "budget_frames": cell["budget_frames"],
            "ckpt_sha256": cell["ckpt_sha256"],
            "runs_dir": str(work / "runs" / worker),
            "data_dir": str(work / "data" / worker),
            "out": str(work / "eval" / f"{worker}.jsonl"),
            "log": str(work / "runs" / worker / "run.log"),
            # NB: the worker's OWN stdout/stderr log must live OUTSIDE
            # runs/<worker_tag>/.  default_spawn_worker() has to create the file's
            # parent before the child starts, and a pre-created runs/<worker_tag>/
            # makes the child refuse its own tag with TAG_NOT_UNIQUE (measured on
            # the real coordinator: .tmp_v829_info/s11/work-gate/runs/
            # gate-pair-w*).  The base runs/<tag>/ belongs to the coordinator, so
            # the worker logs go there.
            "worker_log": str(work / "runs" / tag / f"worker-w{index + 1}.log"),
            "_root": cell["_root"],
        })
    return specs


def assert_unique_worker_tags(specs: list[dict]) -> dict:
    """Refuse before spawning: two workers may not share a tag / runs / data dir.

    A unique tag implies unique dirs by construction, so both are checked
    explicitly -- the dirs are the thing that actually gets corrupted, and a
    future change that decouples them must not be able to slip through here.
    """

    seen: dict[str, dict[str, str]] = {"tag": {}, "runs_dir": {}, "data_dir": {}}
    for spec in specs:
        for kind in ("tag", "runs_dir", "data_dir"):
            value = str(spec[kind])
            if value in seen[kind]:
                raise PairError(
                    "TAG_NOT_UNIQUE",
                    f"--workers: worker {spec['worker_index']} and worker "
                    f"{seen[kind][value]} would both use {kind} {value!r}; every "
                    f"work process needs its own tag and its own data dir "
                    f"(pre-registration sec. 3.4, MAP sec. 9-3)")
            seen[kind][value] = str(spec["worker_index"])
    return {"workers": len(specs),
            "tags": [str(spec["tag"]) for spec in specs],
            "batches": sorted({int(spec["batch"]) for spec in specs})}


def default_spawn_worker(argv: list[str], spec: dict, *,
                         env: dict | None = None):
    """Start one worker with **this** interpreter; returns a handle with ``wait()``.

    ``sys.executable`` in ``argv`` (``build_argv``) is the running interpreter, so
    the workers are the same Python 3.10.12 of ``/root/v400-venv/bin/python`` and
    not whatever ``python`` happens to be on ``PATH``.  The child's stdout/stderr
    go to ``runs/<worker_tag>/worker.log`` -- never to a pipe the coordinator
    would have to drain (a full pipe would deadlock a real pair run).
    """

    import subprocess

    log_path = pathlib.Path(spec["worker_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    try:
        return subprocess.Popen(
            [str(item) for item in argv], cwd=str(REPO),
            env=dict(env) if env is not None else None,
            stdout=handle, stderr=subprocess.STDOUT, text=True)
    finally:
        handle.close()


def run_worker_batches(*, specs: list[dict], spawn, argv_for: callable,
                       on_start=None, on_finish=None) -> dict:
    """Run the batches in order; a failing worker stops the *later* batches only.

    Interrupt safety: every worker of the current batch is allowed to finish (its
    artifacts are the evidence of what it did), the batches that have not started
    are recorded as ``skipped`` instead of being launched, and the aggregate exit
    code is non-zero.  Nothing already written is deleted.
    """

    if not specs:
        return {"batch_count": 0, "batches": [], "results": [], "failed": False,
                "returncode": EXIT_OK, "started": 0, "completed": 0}
    order = sorted({int(spec["batch"]) for spec in specs})
    results: list[dict] = []
    failed = False
    started = 0
    completed = 0
    batches: list[list[int]] = []
    for batch in order:
        current = [spec for spec in specs if int(spec["batch"]) == batch]
        batches.append([int(spec["worker_index"]) for spec in current])
        if failed:
            for spec in current:
                results.append({"worker_index": spec["worker_index"],
                                "tag": spec["tag"], "batch": batch,
                                "out": spec["out"], "status": "skipped",
                                "returncode": None,
                                "detail": ("an earlier worker exited non-zero, so "
                                           "this batch was not started; its "
                                           "jsonl stays absent rather than empty")})
            continue
        handles: list[tuple[dict, object, str | None]] = []
        for spec in current:
            argv = [str(item) for item in argv_for(spec)]
            if on_start is not None:
                on_start(spec, argv)
            try:
                handle = spawn(argv, spec)
            except Exception as exc:                              # noqa: BLE001
                handles.append((spec, None, f"{type(exc).__name__}: {exc}"))
                continue
            started += 1
            handles.append((spec, handle, None))
        for spec, handle, spawn_error in handles:
            if handle is None:
                failed = True
                results.append({"worker_index": spec["worker_index"],
                                "tag": spec["tag"], "batch": batch,
                                "out": spec["out"], "status": "spawn-failed",
                                "returncode": None,
                                "detail": spawn_error})
                if on_finish is not None:
                    on_finish(spec, None, spawn_error)
                continue
            wait_error = None
            try:
                returncode = int(handle.wait())
            except Exception as exc:                              # noqa: BLE001
                returncode = None
                wait_error = f"{type(exc).__name__}: {exc}"
            completed += 1
            if returncode != 0:
                failed = True
            results.append({"worker_index": spec["worker_index"],
                            "tag": spec["tag"], "batch": batch,
                            "out": spec["out"],
                            "status": ("ok" if returncode == 0 else "failed"),
                            "returncode": returncode,
                            "detail": wait_error})
            if on_finish is not None:
                on_finish(spec, returncode, wait_error)
    # aggregate exit code: 2 ("cannot evaluate") outranks 1 ("refused"), and a
    # worker that never produced a code (spawn failure) also means "cannot
    # evaluate" -- a non-zero coordinator exit is the contract either way
    attempted = [entry["returncode"] for entry in results
                 if entry["status"] != "skipped"]
    if not failed:
        aggregate = EXIT_OK
    elif any(code is None or code == EXIT_UNUSABLE for code in attempted):
        aggregate = EXIT_UNUSABLE
    else:
        aggregate = EXIT_REFUSED
    return {"batch_count": len(batches), "batches": batches,
            "results": results, "failed": bool(failed),
            "returncode": aggregate, "started": started,
            "completed": completed}


def merge_worker_rows(*, specs: list[dict], merge_out: pathlib.Path) -> dict:
    """Concatenate every worker jsonl into ``merge_out``, in cell order.

    Only completed workers have a file, and a missing worker file is recorded as
    ``present: false`` instead of being invented: after a failure the merge is a
    record of what actually exists, not a full 4-cell table.
    """

    lines: list[str] = []
    sources: list[dict] = []
    for spec in specs:
        path = pathlib.Path(spec["out"])
        present = path.is_file()
        rows = 0
        if present:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                lines.append(line)
                rows += 1
        sources.append({"worker_index": spec["worker_index"],
                        "tag": spec["tag"], "out": str(path),
                        "present": present, "rows": rows})
    target = pathlib.Path(merge_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(line + "\n" for line in lines)
    target.write_text(payload, encoding="utf-8")
    return {"out": str(target), "rows": len(lines),
            "bytes": len(payload.encode("utf-8")),
            "sha256": sha256_file(target), "sources": sources,
            "empty_workers": [entry["worker_index"] for entry in sources
                              if not entry["present"]]}


def mode_workers(args: argparse.Namespace, *, guard_doc: dict, manifest: dict,
                 roots: list[dict], plan: dict) -> int:
    """``--mode pair --workers N``: one process per cell, merged ``--out``.

    The coordinator runs **no** episode itself.  It validates the cell table,
    asserts tag uniqueness before the first spawn, launches the batches, merges
    the per-worker jsonl into ``--out`` and writes ``runs/<tag>/run.json`` with
    the whole worker table -- including for a run that a failed worker cut short.
    """

    workers = int(getattr(args, "workers", DEFAULT_WORKERS))
    if args.mode != "pair":
        raise PairError("WORKERS_MODE_UNSUPPORTED",
                        f"--workers {workers} is only defined for --mode pair (a "
                        f"pair cell is what fans out); got --mode {args.mode}")
    work = pathlib.Path(args.work_dir)
    tag = str(args.tag or "").strip()
    if not tag:
        raise PairError("TAG_REQUIRED", "--tag is mandatory for a run (MAP sec. 9-3)")
    require_legacy_opt_in(args, roots)
    # NB: every per-root protocol gate (window / takeover / budget / exposure /
    # pair_rng / ckpt) already ran in preflight() with needs_engine=True before
    # this function was reached; it is deliberately not re-run here, so the
    # coordinator cannot mutate the roots twice.
    cells = work_cells(roots)
    specs = plan_workers(cells, workers=workers, tag=tag, work_dir=work)
    uniqueness = assert_unique_worker_tags(specs)
    merge_out = out_for(args, tag)
    if pathlib.Path(merge_out).exists():
        raise PairError("OUT_EXISTS", f"--out {merge_out} already exists")
    assert_writable(pathlib.Path(merge_out), work_dir=work, what="--out")
    base_dir = work / "runs" / tag
    assert_writable(base_dir / "run.json", work_dir=work,
                    what="coordinator run.json")
    for spec in specs:
        for target, what in ((pathlib.Path(spec["out"]), "worker --out"),
                             (pathlib.Path(spec["log"]), "worker --log"),
                             (pathlib.Path(spec["runs_dir"]), "worker runs dir"),
                             (pathlib.Path(spec["data_dir"]),
                              "worker data dir")):
            assert_writable(target, work_dir=work, what=what)
    # engine digest first (never relaxed), exactly as mode_run does: a missing or
    # drifted engine must be reported once, before four workers each discover it
    engine, engine_source = resolve_engine(args, manifest)
    if not pathlib.Path(engine).is_file():
        raise PairError("ENGINE_MISSING",
                        f"engine binary not found: {engine} (source: "
                        f"{engine_source}); refusing before any spawn")
    data_source, data_source_source = resolve_data_source(args, manifest)
    if not pathlib.Path(data_source).is_dir():
        raise PairError("DATA_SOURCE_MISSING",
                        f"engine data source not found: {data_source}")
    env = dict(m450().prepare_env())
    if env.get("SDL_AUDIODRIVER") != "dummy":
        raise PairError("AUDIO_ENV_NOT_MUTED",
                        f"v450.prepare_env() did not pin SDL_AUDIODRIVER=dummy: "
                        f"{env}")
    arm_order = (list(args.arm_order) if isinstance(args.arm_order, list)
                 else parse_arm_order(args.arm_order))
    extra: list[str] = []
    if getattr(args, "lineage_config", None):
        extra += ["--lineage-config", str(args.lineage_config)]

    def argv_for(spec: dict) -> list[str]:
        root = spec["_root"]
        ckpt_path, _ckpt_sha = resolve_ckpt(root, args)
        return build_argv(
            mode="pair", args=args, root=root, tag=spec["tag"],
            out=pathlib.Path(spec["out"]), log_path=pathlib.Path(spec["log"]),
            arm=None, arm_order=arm_order, ckpt=ckpt_path,
            takeover=(int(root["takeover_frame"])
                      if (root.get("_takeover_resolution") or {}).get(
                          "override_active") else None),
            extra=extra)

    base_dir.mkdir(parents=True, exist_ok=True)
    log(f"mode=pair --workers {workers}: {len(specs)} cell(s) -> "
        f"{len(uniqueness['batches'])} batch(es); tags "
        f"{[spec['tag'] for spec in specs]}")
    run = run_worker_batches(
        specs=specs, spawn=default_spawn_worker, argv_for=argv_for,
        on_start=lambda spec, argv: log(
            f"worker {spec['worker_index']}/{len(specs)} start tag={spec['tag']} "
            f"batch={int(spec['batch']) + 1} argv={quote_argv(argv)}"),
        on_finish=lambda spec, rc, error: log(
            f"worker {spec['worker_index']}/{len(specs)} done tag={spec['tag']} "
            f"rc={rc}" + (f" ({error})" if error else "")))
    merge = merge_worker_rows(specs=specs, merge_out=merge_out)
    log(f"merged {merge['rows']} row(s) from {len(specs)} worker jsonl into "
        f"{merge['out']} sha256={merge['sha256'][:16]}...")
    summary = {
        "schema": SCHEMA_RUN, "tool": TOOL, "tool_version": TOOL_VERSION,
        "mode": args.mode, "tag": tag,
        "kind": "workers-coordinator",
        "manifest_path": str(args.manifest),
        "manifest_sha256": plan.get("manifest_sha256"),
        "manifest_bytes": plan.get("manifest_bytes"),
        "manifest_sha256_source": (plan.get("manifest_closure") or {})
        .get("manifest_sha256_source"),
        "manifest_closure": plan.get("manifest_closure"),
        "guard_ok": plan.get("guard", {}).get("ok"),
        "guard_codes": plan.get("guard", {}).get("codes"),
        "lineage_checked": bool(plan.get("lineage_checked")),
        "lineage_codes": list(plan.get("lineage_codes") or []),
        "lineage": plan.get("lineage"),
        "residual_risk": list(plan.get("residual_risk") or []),
        "engine": {"path": engine, "source": engine_source},
        "data_source": {"path": data_source, "source": data_source_source},
        "workers_requested": workers, "max_workers": MAX_WORKERS,
        "cells": [{key: value for key, value in cell.items()
                   if not key.startswith("_")} for cell in cells],
        "workers": [{key: value for key, value in spec.items()
                     if not key.startswith("_")} for spec in specs],
        "uniqueness": uniqueness,
        "batch_count": run["batch_count"], "batches": run["batches"],
        "results": run["results"], "failed": bool(run["failed"]),
        "started": run["started"], "completed": run["completed"],
        "merge": {key: value for key, value in merge.items() if key != "sources"},
        "merge_sources": merge["sources"],
        "merge_note": ("each worker writes its own jsonl and the coordinator "
                       "concatenates them into --out in cell order; nothing is "
                       "moved or deleted, so the merge can be re-derived"),
        "out": str(merge_out),
        "exit_code": int(run["returncode"]),
        "engine_started": bool(run["started"]),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    new_roots = list((plan.get("lineage") or {}).get("lineage_new_roots") or [])
    if new_roots:
        summary["lineage_new_roots"] = new_roots
    budget_block = dict(plan.get("budget_selection") or {})
    if budget_block:
        # only a mixed-tier invocation gets this key (see mode_run): the
        # --budget-frames selector's result, so the coordinator record names the
        # roots these cells came from and the ones left for the other tier
        summary["budget_selection"] = budget_block
    write_json(base_dir / "run.json", summary)
    write_json(base_dir / "workers.json", {
        "schema": SCHEMA_RUN + ":workers", "tag": tag,
        "workers": summary["workers"], "uniqueness": uniqueness,
        "batches": run["batches"], "results": run["results"],
        "merge": summary["merge"], "merge_sources": merge["sources"]})
    log(f"done: workers={len(specs)} batches={run['batch_count']} "
        f"completed={run['completed']} failed={run['failed']} "
        f"rows={merge['rows']} -> {base_dir}")
    return int(run["returncode"])


# --------------------------------------------------------------------------- #
# --mode check --workers N: the offline rehearsal of mode_workers' plan
# --------------------------------------------------------------------------- #
#: What a real `--mode pair --workers N` additionally insists on, judged by
#: ``mode_check_workers`` and printed as ``preconditions`` instead of refused --
#: check mode is the *plan*, so it reports what the run would demand.
WORKERS_RUN_ONLY_REQUIREMENTS = (
    ("tag", "--tag", "mode_workers refuses TAG_REQUIRED without it: the "
                     "coordinator tag names runs/<tag>/ and every worker tag "
                     "<tag>-w<i>; check derives a placeholder from the manifest "
                     "stem so the plan is still printable"),
    ("allow-legacy-window", "--allow-legacy-window",
     "require_legacy_opt_in() refuses LEGACY_WINDOW_NOT_ALLOWED for the four "
     "historical windows unless the run opts in explicitly"),
)


def check_worker_specs(args: argparse.Namespace, *, roots: list[dict],
                       plan: dict, workers: int) -> dict:
    """The concurrency plan ``--mode pair --workers N`` would execute, offline.

    Pure stdlib and pure planning: it calls ``work_cells``/``plan_workers``/
    ``assert_unique_worker_tags`` -- the very functions the coordinator uses --
    and ``build_argv`` (which takes the checkpoint path from the preflight plan,
    so ``resolve_ckpt``'s engine-side neighbours are never imported).  Nothing is
    created, nothing is spawned, nothing is deleted.
    """

    work = pathlib.Path(args.work_dir)
    tag_given = bool(str(args.tag or "").strip())
    tag = (str(args.tag).strip() if tag_given
           else default_tag(args, roots[0]))
    cells = work_cells(roots)
    specs = plan_workers(cells, workers=workers, tag=tag, work_dir=work)
    uniqueness = assert_unique_worker_tags(specs)
    merge_out = out_for(args, tag)
    arm_order = (list(args.arm_order) if isinstance(args.arm_order, list)
                 else parse_arm_order(args.arm_order))
    extra: list[str] = []
    if getattr(args, "lineage_config", None):
        extra += ["--lineage-config", str(args.lineage_config)]
    # plan["plans"] is index-aligned with roots (both iterate the selected roots
    # in order), so plans[i]["ckpt_path"] is exactly the --ckpt that root's argv
    # must carry -- preflight already resolved repo-relative paths for us.
    by_root = {int(entry["root_index"]): entry for entry in plan["plans"]}
    table: list[dict] = []
    for cell, spec in zip(cells, specs):
        root = cell["_root"]
        entry = by_root[int(root["_index"])]
        resolution = root.get("_takeover_resolution") or {}
        takeover = (int(root["takeover_frame"])
                    if resolution.get("override_active") else None)
        ckpt = (pathlib.Path(entry["ckpt_path"])
                if entry.get("ckpt_path") else None)
        argv = build_argv(
            mode="pair", args=args, root=root, tag=spec["tag"],
            out=pathlib.Path(spec["out"]), log_path=pathlib.Path(spec["log"]),
            arm=None, arm_order=arm_order, ckpt=ckpt, takeover=takeover,
            extra=extra)
        plan_entry = {key: value for key, value in spec.items()
                      if not key.startswith("_")}
        plan_entry.update({
            "window": int(root["window"]),
            "pair_rng": int(entry["pair_rng"]),
            "ancestor_id": entry.get("ancestor_id"),
            "takeover_frame": int(root["takeover_frame"]),
            "ckpt_path": entry.get("ckpt_path"),
            "argv": [str(item) for item in argv],
            "argv_text": quote_argv(argv),
            "command": quote_argv(argv),
        })
        table.append(plan_entry)
    # the fence every one of those writes would go through: dry, but the same
    # verdict, so a plan that could not be written cannot look executable
    fence: list[dict] = []
    for target, what in (
            (merge_out, "merge --out"),
            (work / "runs" / tag / "run.json", "coordinator run.json"),
            (work / "runs" / tag / "workers.json", "coordinator workers.json")):
        fence.append(fence_verdict(target, work_dir=work, what=what))
    for spec in table:
        for key, what in (("out", "worker --out"), ("log", "worker --log"),
                          ("worker_log", "worker stdout log"),
                          ("runs_dir", "worker runs dir"),
                          ("data_dir", "worker data dir")):
            fence.append(fence_verdict(pathlib.Path(str(spec[key])),
                                       work_dir=work, what=what))
    denied = [item for item in fence if item["verdict"] == "deny"]
    existing = sorted({str(pathlib.Path(str(spec[key])))
                       for spec in table
                       for key in ("runs_dir", "data_dir", "out")
                       if pathlib.Path(str(spec[key])).exists()}
                      | {str(path) for path in (work / "runs" / tag,
                                                work / "data" / tag,
                                                merge_out) if path.exists()})
    batches = uniqueness["batches"]
    batches_by_index = [
        [int(spec["worker_index"]) for spec in table
         if int(spec["batch"]) == int(batch)] for batch in batches]
    document = {
        "schema": SCHEMA_PLAN + ":workers",
        "tool": TOOL, "tool_version": TOOL_VERSION,
        "kind": "workers-plan", "mode_requested": args.mode,
        "check_only": True, "would_start_engine": False,
        "would_spawn": 0, "engine_side_modules_imported": engine_modules_loaded(),
        "manifest_path": str(args.manifest),
        "manifest_sha256": plan.get("manifest_sha256"),
        "manifest_bytes": plan.get("manifest_bytes"),
        "guard_ok": (plan.get("guard") or {}).get("ok"),
        "guard_codes": (plan.get("guard") or {}).get("codes"),
        "work_dir": str(pathlib.Path(work).expanduser().resolve()),
        "tag": tag,
        "tag_source": ("--tag" if tag_given else
                       "derived from the manifest stem (placeholder: a real "
                       "--mode pair --workers run refuses TAG_REQUIRED)"),
        "workers_requested": int(workers), "max_workers": MAX_WORKERS,
        "cells": [{key: value for key, value in cell.items()
                   if not key.startswith("_")} for cell in cells],
        "cell_count": len(cells),
        "batch_count": len(batches),
        "batches": batches_by_index,
        "batch_sizes": [len(entry) for entry in batches_by_index],
        "workers": table,
        "uniqueness": uniqueness,
        "merge_out": str(merge_out),
        "merge_note": ("mode=pair concatenates the per-worker jsonl into "
                       "merge_out in cell order; check does NOT create it, "
                       "because mode_workers refuses OUT_EXISTS"),
        "fence": fence,
        "fence_denied": denied,
        "targets_existing": existing,
        "preconditions": [
            {"key": key, "value_now": getattr(args, key.replace("-", "_"), None),
             "run_only_flag": flag, "why": why}
            for key, flag, why in WORKERS_RUN_ONLY_REQUIREMENTS],
        "notes": [
            "check --workers N prints the plan mode=pair --workers N would run: "
            "one cell per root, one worker per cell, ceil(cells/N) batches of at "
            "most N processes, each worker with its own tag <tag>-w<i> and its "
            "own runs/data/eval paths",
            "nothing is spawned, no engine-side module is imported and no file "
            "is written by this plan; a real run additionally requires --tag and "
            "--allow-legacy-window (see preconditions)",
            "the per-worker argv is the same build_argv() the coordinator uses, "
            "with --ckpt taken from the preflight plan, so it can be pasted and "
            "re-run as-is",
        ],
    }
    budget_block = dict(plan.get("budget_selection") or {})
    if budget_block:
        # the --budget-frames selector of the plan these cells came from (only a
        # mixed-tier manifest produces it, so a single-tier document keeps shape)
        document["budget_selection"] = budget_block
    return document


def render_workers_plan(document: dict) -> str:
    """Human-readable form of ``check_worker_specs`` (the plan, not a run)."""

    lines = [
        f"v829 pair: mode=check --workers {document['workers_requested']} "
        f"(check only: prints the plan, starts nothing)",
        f"  work-dir    {document['work_dir']}",
        f"  tag         {document['tag']}  ({document['tag_source']})",
        f"  cells       {document['cell_count']}",
        f"  batches     {document['batch_count']}  sizes={document['batch_sizes']} "
        f"-> {document['batches']}",
        f"  merge --out {document['merge_out']}  (not created by check)",
        f"  would_start_engine {str(document['would_start_engine']).lower()}   "
        f"would_spawn {document['would_spawn']}   "
        f"engine_side_modules_imported {document['engine_side_modules_imported']}",
    ]
    budget_selection = document.get("budget_selection") or {}
    if budget_selection:
        # only a mixed-tier plan prints it (single-tier output stays identical)
        lines.append(budget_selection_line(budget_selection))
    for spec in document["workers"]:
        lines.append(
            f"  worker {spec['worker_index']}/{len(document['workers'])}  "
            f"batch {int(spec['batch']) + 1}/{document['batch_count']}  "
            f"tag {spec['tag']}")
        lines.append(
            f"      cell {spec['cell_key']}")
        lines.append(
            f"      window {spec['window']}  pair_rng {spec['pair_rng']}  "
            f"budget {spec['budget_frames']} frames  "
            f"takeover {spec['takeover_frame']}  ancestor {spec['ancestor_id']}")
        lines.append(f"      out  {spec['out']}")
        lines.append(f"      argv {spec['argv_text']}")
    for item in document["fence"]:
        lines.append(f"  fence {'ALLOW' if item['verdict'] != 'deny' else 'DENY '}"
                     f"  {item['what']:<20} rule={item['rule']}  "
                     f"{item['target']}")
    if document["targets_existing"]:
        lines.append("  WARNING these worker targets already exist, so "
                     "mode=pair would refuse TAG_NOT_UNIQUE:")
        for path in document["targets_existing"]:
            lines.append(f"      {path}")
    for item in document["preconditions"]:
        lines.append(f"  run-only    {item['run_only_flag']}: {item['why']}")
    return "\n".join(lines)


def write_workers_plan_record(*, args: argparse.Namespace,
                              document: dict) -> pathlib.Path:
    """Persist the check-mode concurrency plan as ``runs/<tag>/run.json``.

    Same place and the same ``v829-pair-run/1`` envelope a coordinator run uses,
    with ``engine_started: false`` and ``kind: workers-plan``, so the record is
    obviously a rehearsal.  ``check_record_only`` keeps a repeated check
    idempotent (it may overwrite this one file and nothing else).
    """

    work = pathlib.Path(args.work_dir)
    target = assert_writable(work / "runs" / str(document["tag"]) / "run.json",
                             work_dir=work, what="check workers run.json")
    record = {
        "schema": SCHEMA_RUN, "tool": TOOL, "tool_version": TOOL_VERSION,
        "mode": "check", "tag": document["tag"], "kind": "workers-plan",
        "manifest_path": document["manifest_path"],
        "manifest_sha256": document["manifest_sha256"],
        "manifest_bytes": document["manifest_bytes"],
        "guard_ok": document["guard_ok"], "guard_codes": document["guard_codes"],
        "engine_started": False, "would_start_engine": False,
        "workers_requested": document["workers_requested"],
        "max_workers": MAX_WORKERS,
        "cells": document["cells"], "workers": document["workers"],
        "uniqueness": document["uniqueness"],
        "batch_count": document["batch_count"], "batches": document["batches"],
        "merge_out": document["merge_out"],
        "fence": document["fence"], "preconditions": document["preconditions"],
        "notes": document["notes"],
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if document.get("budget_selection"):
        # only a mixed-tier plan gets it, so a single-tier workers record keeps
        # exactly the shape it had
        record["budget_selection"] = document["budget_selection"]
    write_json(target, record)
    return target


def mode_check_workers(args: argparse.Namespace, *, plan: dict,
                       roots: list[dict], workers: int) -> int:
    """``--mode check --workers N``: print the concurrency plan and exit 0.

    s11 refused this combination with ``WORKERS_MODE_UNSUPPORTED``, so the only
    way to find out which cell lands in which batch (and what each worker's tag
    and argv would be) was to start four engines.  Nothing here starts anything:
    the plan is computed by the coordinator's own ``plan_workers`` and the
    engine-side import gate is re-checked before the document is printed.
    """

    document = check_worker_specs(args, roots=roots, plan=plan, workers=workers)
    print(render_workers_plan(document))
    loaded = document["engine_side_modules_imported"]
    if loaded:
        print(f"REFUSED [ENGINE_SIDE_IMPORT] check mode must not touch the "
              f"engine side, but these modules are loaded: {loaded}",
              file=sys.stderr)
        return EXIT_REFUSED
    if document["fence_denied"]:
        block = document["fence_denied"][0]
        raise PairError(
            "HISTORICAL_PATH",
            f"{block['what']} {block['target']} is REFUSED by write-fence rule "
            f"{block['rule']} [{block['basis']}]: {block['reason']}. The plan "
            f"cannot be executed, so --mode pair --workers "
            f"{document['workers_requested']} would refuse it too "
            f"({len(document['fence_denied'])} denied target(s))")
    print("--- json summary ---")
    print(json.dumps(document, indent=1, ensure_ascii=False))
    written = write_workers_plan_record(args=args, document=document)
    print(f"wrote: {written}   (mode=check --workers invocation record)")
    print(f"plan: cells={document['cell_count']} "
          f"batches={document['batch_count']} {document['batch_sizes']} "
          f"workers={[spec['tag'] for spec in document['workers']]} "
          f"would_start_engine=false")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
EPILOG = """\
modes
  check    manifest guard + every preflight that needs no engine + the exact
           argv the other modes would run.  Starts nothing and imports nothing
           from the engine side (touhou_rl / torch / numpy / subprocess are
           checked to be absent from sys.modules at the end).  Its only write is
           <work>/runs/<tag>/run.json (manifest byte closure + lineage verdict +
           engine_started: false), which a repeated check overwrites in place.
           With --workers N > 1 it additionally prints the concurrency plan
           (see "--workers N" below) and writes that same record with
           kind: workers-plan; it does NOT write --out there, because --out is
           the merge target and mode=pair refuses OUT_EXISTS.

manifest bytes (TOCTOU closure)
  The guard reads the manifest and judges those bytes; this file then does
  exactly ONE read_bytes(), asserts sha256(raw) == guard.manifest_sha256 (and
  len(raw) == guard.manifest_bytes), and parses that same buffer.  A mismatch is
  MANIFEST_CHANGED (exit 1) before any field is read, so an edit between the two
  reads can never be used.  plan.json/run.json record manifest_sha256,
  manifest_reads: 1 and manifest_sha256_source (guard, or self when the guard
  document carries no digest -- then the weaker provenance is a residual risk).

exposure lineage (--lineage-config PATH, off by default)
  used_for_training / used_for_selection are the manifest's own declaration.  With
  --lineage-config configs/v430/splits.json every root must declare
  lineage_ancestor_ids (array of strings) and every id is looked up in
  splits.{train,dev,confirm,adapt,quarantine}.ancestor_ids plus the
  pretraining_sources registry named by that config's registries[]:
    LINEAGE_MISSING               no lineage_ancestor_ids on the root (refused)
    LINEAGE_NEW                   the root declares "lineage_new": true, i.e. an
                                  ancestor the frozen genealogy does not have:
                                  recorded, exit 0 (see the section below)
    EXPOSURE_CONTRADICTS_LINEAGE  id in train/dev (or a used_for_training /
                                  used_for_selection source) while the root says
                                  used_for_training=false AND
                                  used_for_selection=false (refused), or
                                  "lineage_new": true plus such an id (refused)
    LINEAGE_UNKNOWN               declared exposed but in no split and no source:
                                  recorded in run.json.lineage_unmatched, NOT
                                  refused (an old window may predate the freeze)
  Results land in run.json: lineage_checked, lineage_config_sha256,
  lineage_sources_sha256, lineage_unmatched and per root lineage_split /
  lineage_status / lineage_unmatched.  Without the switch run.json states
  lineage_checked: false plus residual_risk "exposure is self-declared".

prefix digest
  parent_prefix_digest is compared, not recorded: prefix_digest_gate() hashes
  s1c_recording_path(window) and refuses PREFIX_RECORDING_MISMATCH when it differs
  (or when verify_anchor.window_recording_sha256 differs).  The four old windows
  were measured on four different recordings; f79f8ab6... is the seed12 one (MAP
  sec. 1 / D3 gate) and is not a per-window constant.

prefix tape (optional per root, additive)
  build_masks hard-codes one recording: .tmp_v410_s1c/d1/shards/
  d1-s0d1-n3300-seed<window>-shard/labels.jsonl (v431:64-99).  A root may instead
  declare its own 4-frame labels.jsonl:
      "prefix_tape":          repo-relative path, host-absolute path, or the WSL
                             /mnt/<drive>/... form the recordings carry
      "prefix_tape_sha256":   optional; when present it must equal
                              parent_prefix_digest AND sha256(the file)
      "prefix_tape_rows":     optional; when present it must equal the row count
      "prefix_tape_frames":   optional; when present it must equal rows * 4
  read_tape(path, target) is the reader: masks = every row's 4 step_frames (never
  truncated -- a recorded policy keeps following the tape), hashes[frame] =
  hash_pre and hashes[frame + 4] = hash_post, exactly build_masks' semantics plus
  four strict refusals (decision_frame must be rows_so_far * 4 -- 0, 4, 8, ... no
  gap/overlap/duplicate; every row needs hash_pre and hash_post; the anchor set
  must be complete; target must be a multiple of 4, recorded, and <=
  frames_built).  pair/trace then re-point v450_cf_label.masks_for (and every
  in-process copy that carries the name) at read_tape()'s (masks, hashes, meta) --
  the same memory-only discipline as the takeover override, with the three pinned
  files sha256-compared before and after (PREFIX_TAPE_TOUCHED_DISK).
  --prefix-tape PATH may only CONFIRM the root's declared tape (a different path
  is PREFIX_TAPE_CLI_MISMATCH, a root without prefix_tape is
  PREFIX_TAPE_NOT_IN_MANIFEST).  A missing tape is PREFIX_TAPE_MISSING (exit 2);
  a digest disagreement is PREFIX_TAPE_DIGEST_MISMATCH (exit 1), including the
  parent_prefix_digest != prefix_tape_sha256 case: the tape the cell names and the
  digest it pins must be one file.  Without prefix_tape none of this runs and the
  S1c path above is byte-for-byte the old one.  run.json records the block
  {path, sha256, rows, frames_built, target, injected_into,
  files_on_disk_unmodified} plus anchors_complete / recorded_hash_frames; a tape
  root whose window is outside (11, 12, 17, 18) is allowed, because its prefix no
  longer comes from that window's S1c recording (the window then only labels the
  private data dir w<window>).

  trace    one arm (--arm) on one parked parent.
  pair     both arms (--arm-order) from ONE PARK, i.e. the same start frame and
           the same inventory at decision 0.
  compare  two --keep-trace documents, G1 vs T2; pure stdlib.

recovery (windows, and the takeover frame since v8.30-dev)
  Windows: only (11, 12, 17, 18); anything else is WINDOW_NOT_SUPPORTED when the
  mode needs the engine.
  Takeover frame: --takeover-frame N (default: the root's own
  roots[i].takeover_frame, 8663 for every pre-v8.30 root).  The frozen cell
  (N == 8663) only reads the pinned constants back and refuses on drift
  (PREFIX_CONSTANT_DRIFT) -- nothing is assigned.  A non-frozen cell re-points the
  *module objects this process loaded by path* (never the files, which are
  sha256-compared before and after the assignment; never another process):
    PREFIX_FRAMES=N, CHECKPOINTS_IN_PREFIX=N//4 on v432_fork_arm,
    PREFIX_FRAMES=N, TARGET_FRAME=N+1, CHECKPOINTS_IN_PREFIX=N//4 on v450_cf_label,
    PREFIX_FRAMES=N on v830_bombpol,
  including the copies those modules load for themselves (v450._M432,
  v830._M450, v830._M432) -- patching only v829's own copies would leave the copy
  that actually runs walk_prefix at 8663.  Admissible N: N >= 4, N % 4 == 3
  (target = N+1 must be a recorded anchor), N + 1 <= frames_built; refusals are
  TAKEOVER_FRAME_NOT_INTEGER / TAKEOVER_FRAME_TOO_SMALL /
  TAKEOVER_FRAME_NOT_ANCHOR / TAKEOVER_BEYOND_RECORDING /
  TAKEOVER_ROOT_MISMATCH.  The block is recorded as prefix_constants in run.json
  and in every episode row.

park reuse (--reuse-park, OFF by default)
  One PARK costs ~1.8 s (8,663 prefix frames replayed + 2,165 hash checks) and
  its result depends on exactly three things: the prefix that is replayed, the
  frame the parent PARKs at, and the budget the episode is written against --
  NOT on the checkpoint, the pair_rng or the arm.  --reuse-park therefore parks
  once per GROUP and spawns one child per (cell, arm) from that one parent
  (ForkParent.spawn, v432_fork_arm.py:580 -- the call v830_bombpol
  .run_episode36 already makes for every arm; nothing else changes, and QUIT
  still ends every episode).
    group key   (prefix, takeover_frame, budget_frames) where prefix is the
                declared prefix_tape's sha256, or "s1c-seed<window>" for a root
                that replays its window's hard-coded S1c shard
    fail-closed a different takeover frame / tape / budget is a DIFFERENT group;
                every other PARK input (window, private data dir, declared
                engine digest, tape target/frames, parent_prefix_digest) is an
                extra discriminator, so a partial match opens a new group and
                park_reuse_verify() re-derives every member from its root before
                the first PARK.  Two cells are never mixed into one prefix.
    rows        every row of a reused cell carries the SAME parent_gate /
                prefix_constants / trace metadata as the cell that parked (it is
                the same gate document and the same constants block) and gains
                  park_reuse {group, park_id, children_spawned, children_failed}
                run.json gains one park_reuse block (per group: park_calls,
                park_pid, gate_name, roots, rows, children_spawned,
                children_failed, void, skipped_root_indices) and every attempt
                record names its park_reuse_group.
    faults      a SPAWN refusal or a QUIT error (ForkParent.quit_child returns
                "ERR ..." instead of raising, v432:588-592) makes THAT CELL
                invalid (invalid=true + error, score re-derived) and never
                re-PARKs; an exception escaping an arm voids the group, and a
                parent that is no longer alive after the PARK voids the whole
                group -- the run exits non-zero and run.json names the void group
                and every skipped cell.  Because a re-run would need a second
                PARK, a reuse cell gets ONE attempt (--reuse-park trades the
                sec. 4 re-run for "never re-PARK"; with the flag off the two
                attempts are unchanged).
    order       groups are run in order of first appearance and cells in manifest
                order, so a group is contiguous and at most one parent is alive;
                the jsonl is therefore written in group order (row content
                unchanged, file order not).  --mode check --reuse-park prints the
                same grouping without starting or importing anything.
    refused     --mode trace / compare (no cell to share) and --workers N > 1
                (one cell per worker process, so no PARK can be shared).

rng (pre-registration sec. 1)
  generator_seed = pair_rng * 1000 + 7   (= v8r0_urgency.generator_seed)
  one torch.Generator() per arm per episode, created AFTER load_policy36; both
  arms therefore start from the same state, and no episode inherits another's
  stream.  Each row records torch.__version__, device, the initial and final
  state digests (sha256(state.numpy().tobytes())[:16]) and the draw count.

bomb accounting
  bombs_start / bombs_end / bombs_used / bomb_emits / bomb_decisions /
  bomb_frames / bombs_remaining_at_emit / bomb_available_at_emit / bomb_ignored
  come from run_episode36; emit_decrement_at (frame_index + 1 inside the emitting
  decision, None if the inventory never dropped) comes from a parent proxy,
  precedent v800_bomb.py:724-725/738.  arm_label is deliberately not written
  (MAP sec. 9-4); rows carry rule_label + threshold_table instead.

compare criteria (frozen)
  Per 0-based decision, strict equality of: threshold (threshold_px),
  nearest_px, cand, mask (the masks actually sent on the wire), bomb_request,
  inventory_before, inventory_after.  A differing decision count is a difference
  at index min(len(G1), len(T2)).  Output: same_prefix_decisions,
  first_diff_decision (0-based, with a 1-based echo), first_diff_frame (absolute
  engine frame), first_diff_fields, verdict IDENTICAL (exit 0) / DIFFER (exit 1).
  first_action_difference is a secondary answer over the behaviour subset only
  (cand, mask, bomb_request, inventory_before, inventory_after): the threshold
  tables differ by construction, so on a real pair first_diff_decision is
  normally 0 and the behaviour difference is when the arms actually diverge.
  Headers must agree on prefix_frame / decision_frames / budget_decisions /
  window, else TRACE_INCOMPATIBLE (exit 2).

exit codes
  0 ok        1 refused (a finding, an invalid row, an audible cfg, or a trace
              difference)        2 cannot evaluate (unreadable input, missing
              file, usage error)

new ancestors in the lineage cross-check (--lineage-config)
  a root may declare "lineage_new": true when its ancestor is NEWER than the
  frozen genealogy (configs/v430/splits.json), where a non-empty
  lineage_ancestor_ids is impossible by construction.  Then:
    empty/absent lineage_ancestor_ids   -> LINEAGE_NEW, verdict RECORDED, exit 0
                                           (named in run.json.lineage_new_roots,
                                           residual_risk gains "new ancestor:
                                           exposure is declared by the run table,
                                           not provable against the frozen
                                           lineage")
    lineage_new + an id in train/dev    -> EXPOSURE_CONTRADICTS_LINEAGE, exit 1
    no lineage_new and no ids            -> LINEAGE_MISSING, exit 1 (unchanged)

technical-fault re-run (pre-registration sec. 4)
  What is technical invalid (triggers the one re-run):
    - RECOVERY (PARK) fault: v450.start_parent raises at ANY point of
      HELLO -> prefix walk -> prefix hash checks -> PARK -> DEBUG threads=1.
      sec. 4's "无法恢复" is exactly this: no parent exists, so no arm can start.
      The attempt record says phase: "recovery" (not "play") and carries
      kind: engine_crash | prefix_hash_mismatch | recovery_error, the exception
      class + message, and the attempt number, e.g.
      error: "attempt 1 engine_crash: HarnessError: harness closed the control pipe".
    - ROW fault: ANY arm's row has invalid=true or a non-empty error.
  What is NEVER technical invalid (never re-runs):
    - a legal death (terminal=first_death/game_over), a full segment
      (survived=true), a stage_clear: those are the results being measured;
    - every guard/preflight refusal (manifest/window/takeover/engine/data/
      checkpoint/tag/pair_rng/audio/lineage/prefix-recording).  That is an input
      problem: the process refuses and does NOT consume a second attempt.
  Re-run rules:
    mode pair: the WHOLE pair is re-run ONCE with the same manifest / window /
    takeover / tape / ckpt / pair_rng (should_retry / run_pair_with_retry, the
    attempt body injected).  Both records are kept in the SAME jsonl: attempt 1
    stays with attempt: 1 and attempt 2 is appended with attempt: 2 and
    supersedes: <sha256 of the attempt-1 row>.  At most 2 attempts, never a
    third; if attempt 2 is still technically invalid the pair is invalid and the
    process exits 1.
    Recovery re-run: attempt 1 failing in start_parent leaves NO row and NO gate;
    attempt 2 makes a fresh PARK written as gate-w<w>-a2.json and its rows carry
    attempt: 2.  If BOTH attempts fail in recovery the process exits
    RECOVERY_FAILED (1), and run.json is still written with
    outcome: "recovery_failed" and TWO phase: "recovery" attempt records whose
    errors name attempt 1 and attempt 2; no third attempt is made.
    mode trace: one arm, so a row fault is not retried (no pair to re-run); a
    recovery fault is retried once, because no row exists yet and the re-run
    repeats the same PARK / arm / configuration.
    Prefix hash mismatch is retried like any other recovery fault (sec. 4 makes
    unrecoverability the fault); kind: "prefix_hash_mismatch" keeps a possible
    data/tape problem distinguishable from an engine crash in the record.
  run.json records attempts: [{attempt, phase, error, tag, rows, invalid_rows,
  wall_s, attempt_fingerprint, supersedes, root_index, window}] plus rows_final
  and superseded_rows, and all_rows_valid/invalid_rows/full_segments describe the
  LAST attempt only.  --keep-trace keeps both: attempt 2's trace is
  <stem>.<ARM>[.ep<N>].a2.trace.json.

--workers N (pre-registration sec. 3.4: one unique tag per work process)
  mode pair and mode check, 1..4, default 1.  Cells fan out over worker
  PROCESSES: one cell = one root x its budget x its ckpt (the manifest pins one
  budget and one ckpt per root), one worker = one cell, ceil(cells/N) batches in
  order.  Every worker owns the unique tag <tag>-w<i> and therefore its own
  runs/<tag>-w<i>/ and data/<tag>-w<i>/; a duplicate tag / runs dir / data dir is
  refused TAG_NOT_UNIQUE before the first spawn.
  REHEARSAL: `--mode check --workers N` prints that plan and exits 0 without
  spawning or importing anything -- cells, batches (and their sizes), every
  worker's tag / window / pair_rng / full argv, the merge --out, the write-fence
  verdict for each target and whether any target already exists -- then writes
  <work>/runs/<tag>/run.json as a `kind: workers-plan` record with
  engine_started: false.  It is computed by the coordinator's own
  work_cells()/plan_workers()/assert_unique_worker_tags()/build_argv(), so the
  plan cannot drift from the run.  --tag is optional here (a placeholder is
  derived from the manifest stem) and --allow-legacy-window is reported as a
  `preconditions` entry rather than demanded, because both are *run* gates.
  MERGE: each worker writes its own jsonl <work>/eval/<tag>-w<i>.jsonl and the
  coordinator concatenates them into the single --out (default
  <work>/eval/<tag>.jsonl) in cell order; run.json.merge_sources and
  runs/<tag>/workers.json list every worker jsonl with its rows and digest, so
  the merge is re-derivable and nothing is deleted after it.
  Interruption: any worker exiting non-zero makes the coordinator exit non-zero
  (2 outranks 1), the finished workers' artifacts are kept, and batches that had
  not started are recorded as skipped instead of launched.  Workers are spawned
  with sys.executable (this Python 3.10.12), and --workers is not propagated.
  --workers 1 is the old single-process path, byte for byte.

audio (AGENTS.md sec. 9)
  prepare_env() pins SDL_AUDIODRIVER=dummy, HarnessClient pins it again and pops
  DISPLAY for role=shadow, build_private_data_dir regenerates the 60-byte
  th08.cfg and re-inspects four zero fields.  After the engine exits every
  th08.cfg under <work>/data/<tag> is decoded with v431_audio_cfg_audit.decode()
  and the run exits non-zero unless muted == cfg_files, audible == 0 and all
  four fields are zero.

writes (the fence is an ordered rule table, not a .tmp_* whitelist)
  Only under --work-dir (default .tmp_v829_pair) plus the explicit --out/--log,
  and only where WRITE_FENCE allows it: the whole .tmp_v829_* namespace this
  round owns (.tmp_v829_pair, .tmp_v829_info, .tmp_v829_collect, and any new
  .tmp_v829_<x> a later work order adds), plus anything outside the repository.
  Everything else existing under .tmp_* is DENIED -- .tmp_v400_* (D0/D1),
  .tmp_v410_* (S1 + WP05), .tmp_v420_*, .tmp_v430_*, .tmp_v431_*, .tmp_v44*,
  .tmp_v45*, .tmp_v8[0-9a-z]*, and the catch-all "any other .tmp_*" -- as are
  Document/, data/, third_party/, third_party_data/, tools/, touhou_rl/,
  configs/, tests/ and .git/ (HISTORICAL_PATH).  A refusal prints the rule id,
  the top-level component it matched and that rule's reason; the same judgement
  is available offline for any path via
  `python tools/rl/v829_pair.py --explain-path PATH` (reads nothing, writes
  nothing).
"""


def write_check_record(*, args: argparse.Namespace, plan: dict) -> list[pathlib.Path]:
    """Persist the ``mode=check`` invocation record as ``runs/<tag>/run.json``.

    A check produces no episode rows, but it does produce the two facts the r1
    report asked to be auditable -- which manifest bytes were used (A) and what
    the exposure cross-check said (B) -- so they are written in the same
    ``v829-pair-run/1`` shape a real run uses, one file per selected tag, with
    ``engine_started: false``.  A repeated check overwrites **only** these
    records (``check_record_only``), so ``mode=check`` stays idempotent while a
    real ``runs/<tag>`` is never touched.

    A **mixed-tier** selection adds ``budget_selection`` -- the ``--budget-frames``
    selector's result, the same block ``mode=pair``'s ``run.json`` carries -- to
    every record it writes.  A single-tier selection adds nothing at all, so a
    single-tier record keeps exactly the shape (and bytes) it had.
    """

    lineage = dict(plan.get("lineage") or {})
    lineage_by_root = {entry["root_index"]: entry
                       for entry in lineage.get("roots") or []}
    closure = dict(plan.get("manifest_closure") or {})
    work = pathlib.Path(args.work_dir)
    written: list[pathlib.Path] = []
    for entry in plan["plans"]:
        tag = str(entry["tag"])
        lineage_entry = lineage_by_root.get(entry["root_index"], {})
        record = {
            "schema": SCHEMA_RUN, "tool": TOOL, "tool_version": TOOL_VERSION,
            "mode": "check", "tag": tag,
            "manifest_path": plan["manifest_path"],
            "manifest_sha256": plan["manifest_sha256"],
            "manifest_bytes": plan["manifest_bytes"],
            "manifest_sha256_source": closure.get("manifest_sha256_source"),
            "manifest_sha256_guard": closure.get("manifest_sha256_guard"),
            "manifest_reads": closure.get("manifest_reads"),
            "manifest_read_scope": closure.get("manifest_read_scope"),
            "manifest_closure": closure,
            "guard_ok": plan["guard"]["ok"], "guard_codes": plan["guard"]["codes"],
            "guard_check_count": plan["guard"]["check_count"],
            "lineage_checked": bool(lineage.get("lineage_checked")),
            "lineage_config_path": lineage.get("lineage_config_path"),
            "lineage_config_sha256": lineage.get("lineage_config_sha256"),
            "lineage_config_bytes": lineage.get("lineage_config_bytes"),
            "lineage_sources_path": lineage.get("lineage_sources_path"),
            "lineage_sources_sha256": lineage.get("lineage_sources_sha256"),
            "lineage_sources_checked": lineage.get("lineage_sources_checked"),
            "lineage_splits": lineage.get("lineage_splits"),
            "lineage_codes": list(lineage.get("lineage_codes") or []),
            "lineage_unmatched": list(lineage.get("lineage_unmatched") or []),
            "lineage": lineage,
            "residual_risk": list(plan.get("residual_risk") or []),
            "engine_started": False,
            "would_start_engine": bool(plan.get("would_start_engine")),
            # the same block a real run writes, so a check already shows which
            # takeover frame this cell means and that the pinned files are intact;
            # `override_applied` stays false here by construction (check imports
            # none of the run-side modules)
            "prefix_constants": (
                entry.get("takeover")
                or planned_prefix_constants(
                    int(entry["takeover_frame"]), window=int(entry["window"]))),
            "argv_pair": entry["argv_pair"], "argv_trace": entry["argv_trace"],
            "prefix_recording": next(
                (check for check in plan["checks"]
                 if check["check"] == f"prefix_recording[{entry['root_index']}]"),
                None),
            "roots": [{
                "root_index": entry["root_index"],
                "ancestor_id": entry["ancestor_id"], "window": entry["window"],
                "budget_frames": entry["budget_frames"],
                "budget_decisions": entry["budget_decisions"],
                "pair_rng": entry["pair_rng"],
                "lineage_split": lineage_entry.get("lineage_split"),
                "lineage_status": lineage_entry.get("lineage_status"),
                "lineage_unmatched": list(lineage_entry.get("lineage_unmatched")
                                          or []),
                "lineage_code": lineage_entry.get("code"),
                "rows": 0,
            }],
            "rows": 0,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tape_check = next(
            (check for check in plan["checks"]
             if check["check"] == f"prefix_tape[{entry['root_index']}]"), None)
        if tape_check is not None:
            # only a tape cell gets these keys, so a pre-tape check record keeps
            # exactly the shape it had
            record["prefix_tape"] = tape_check
            record["prefix_tape_block"] = entry.get("prefix_tape")
        new_roots = list(lineage.get("lineage_new_roots") or [])
        if entry["root_index"] in new_roots:
            # only a `lineage_new` cell gets this key (same discipline): the
            # LINEAGE_NEW verdict is also visible in lineage_codes and in
            # roots[0].lineage_code, and this names the root that carried it
            record["lineage_new_roots"] = [entry["root_index"]]
        budget_block = dict(plan.get("budget_selection") or {})
        if budget_block:
            # only a mixed-tier check gets this key: it is the --budget-frames
            # selector's result (requested / matched / skipped_by_budget /
            # matched_roots), which names the roots the matching pair run would
            # execute and the ones it would leave for the other tier.  A
            # single-tier check record keeps exactly the shape it had.
            record["budget_selection"] = budget_block
        target = assert_writable(work / "runs" / tag / "run.json",
                                 work_dir=work, what="check-mode run.json")
        written.append(write_json(target, record))
    return written


def build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="v829_pair.py",
        description=(
            "v8.29 paired G1/T2 evaluation (schema v829-pair-manifest/1). "
            "One PARK at frame 8663, two rules, one fresh torch.Generator per "
            "arm. Old windows only; the decision loop is "
            "v830_bombpol.run_episode36 and the manifest is validated by "
            "tools/rl/v829_guard.py -- neither is re-implemented here."),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True,
                        choices=("check", "trace", "pair", "compare"),
                        help="check = no engine at all (guard + preflight + "
                             "argv); trace = one arm; pair = both arms from one "
                             "PARK; compare = two --keep-trace files")
    parser.add_argument("--manifest", type=pathlib.Path,
                        help="v829-pair-manifest/1 document; validated by "
                             "tools/rl/v829_guard.py before anything else "
                             "(required for check/trace/pair)")
    parser.add_argument("--window", type=int, default=None,
                        help="SELECTOR (not an assertion): keep only the manifest "
                             "roots that declare this window, so one manifest can "
                             "register several windows. v1 only supports 11, 12, "
                             "17, 18 for a run (it is also the S1c recording seed "
                             "and the w<NN> data dir); an empty selection is "
                             "WINDOW_NOT_IN_MANIFEST")
    parser.add_argument("--budget-frames", type=int, choices=BUDGET_TIERS,
                        default=DEFAULT_BUDGET_FRAMES,
                        help="selector over the manifest subset (NOT an assertion "
                             "on every root): keep only the manifest roots whose "
                             "budget_frames equals this tier, applied after "
                             "--window. A "
                             "manifest may therefore register pre-registration "
                             "sec. 2's two tiers (1800 and 2400) side by side and "
                             "each tier is run as its own invocation. Default "
                             "2400 = sec. 2's admission bar, unchanged. 1800/2400 "
                             "frames = 900/1200 decisions (frames / 2); an empty "
                             "selection is BUDGET_TIER_ABSENT (exit 1), and the "
                             "selection is recorded as budget_selection "
                             "{requested, matched, skipped_by_budget, "
                             "matched_roots} in the plan and in run.json when any "
                             "root was skipped. --pair-rng / --takeover-frame / "
                             "--prefix-tape / --ckpt-sha256 are NOT selectors: "
                             "they are registered values and must equal the "
                             "root's declared value")
    parser.add_argument("--arm", choices=ARMS, default=None,
                        help="trace mode: the single arm to run")
    parser.add_argument("--arm-order", default=DEFAULT_ARM_ORDER,
                        help="pair mode: comma-separated balanced order, both "
                             f"arms, e.g. {DEFAULT_ARM_ORDER} or T2,G1 "
                             f"(default {DEFAULT_ARM_ORDER})")
    parser.add_argument("--ckpt", type=pathlib.Path, default=None,
                        help="policy checkpoint (.pt); pair/trace require it "
                             "together with --ckpt-sha256 unless the root "
                             "declares an optional 'ckpt_path'")
    parser.add_argument("--ckpt-sha256", default=None,
                        help="expected sha256 of --ckpt; must equal "
                             "roots[i].ckpt_sha256 (the manifest binds one "
                             "checkpoint per root)")
    parser.add_argument("--takeover-frame", default=None, metavar="N",
                        help="the engine frame the parent PARKs at and the two arms "
                             "start from. Default: the root's own "
                             "roots[i].takeover_frame (8663 for every pre-v8.30 "
                             "root, i.e. the frozen cell). Must be 4k - 1 (the "
                             "recording has one hash anchor every 4 frames, and "
                             "v450.masks_for passes target = N + 1 to "
                             "v431_dense_driver.build_masks, which refuses a target "
                             "with no hash), >= 4, <= frames_built - 1 of this "
                             "window's S1c recording, and equal to the root's "
                             "declared value (the manifest is the run table, so a "
                             "new frame is a new root). A non-8663 value re-points "
                             "the pinned modules' globals in this process only: the "
                             "three files are never edited and their sha256 is "
                             "compared before and after. Refusals: "
                             "TAKEOVER_FRAME_NOT_INTEGER / _TOO_SMALL / "
                             "_NOT_ANCHOR / TAKEOVER_BEYOND_RECORDING / "
                             "TAKEOVER_ROOT_MISMATCH")
    parser.add_argument("--pair-rng", type=int, default=None,
                        help="pre-registration sec. 1 seed of this pair; must equal "
                             "the root's pair_rng (a different seed is a "
                             "different registered pair). Defaults to the "
                             "root's declared value")
    parser.add_argument("--prefix-tape", default=None, metavar="PATH",
                        help="the 4-frame labels.jsonl this cell replays as its "
                             "prefix; must equal the root's optional 'prefix_tape' "
                             "(a repo-relative path or the WSL /mnt/<drive>/... "
                             "form). The manifest is the run table, so this flag "
                             "can only CONFIRM the tape a root already declares: "
                             "given on a root without 'prefix_tape' it is refused "
                             "(PREFIX_TAPE_NOT_IN_MANIFEST), and a different path "
                             "is refused (PREFIX_TAPE_CLI_MISMATCH). With it the "
                             "root ignores the window's S1c shard and re-points "
                             "v450.masks_for at read_tape()'s (masks, hashes, "
                             "meta) in this process; the three pinned files are "
                             "sha256-compared before and after")
    parser.add_argument("--episodes", type=int, default=1,
                        help="episodes per (root, arm); every episode restarts "
                             "the generator at the same seed (default 1)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        metavar="N",
                        help=f"--mode pair: run the pair cells in up to N "
                             f"concurrent worker PROCESSES (1..{MAX_WORKERS}, "
                             f"default {DEFAULT_WORKERS}). --mode check: print "
                             f"exactly that plan and exit 0 without spawning or "
                             f"importing anything (cells, batches, each worker's "
                             f"tag / window / pair_rng / full argv, the merge "
                             f"--out, the write-fence verdict per target), which "
                             f"is the offline rehearsal s11 could not do. "
                             f"One cell = one root x "
                             f"its budget x its ckpt (the manifest pins one "
                             f"budget and one ckpt per root, so a cell is a root), "
                             f"one worker = one cell, and ceil(cells/N) batches "
                             f"are run in order. Every worker gets its OWN tag "
                             f"<tag>-w<i> and therefore its own "
                             f"runs/<tag>-w<i>/ and data/<tag>-w<i>/ "
                             f"(pre-registration sec. 3.4; two workers sharing a "
                             f"tag is refused TAG_NOT_UNIQUE before the first "
                             f"spawn). MERGE: each worker writes its own jsonl "
                             f"<work>/eval/<tag>-w<i>.jsonl and the coordinator "
                             f"concatenates them into --out in cell order, "
                             f"listing every worker jsonl in runs/<tag>/"
                             f"workers.json and in run.json.merge_sources; "
                             f"nothing is deleted after the merge. Any worker "
                             f"exiting non-zero makes the coordinator return "
                             f"non-zero, keeps the finished workers' artifacts "
                             f"and skips only the batches that had not started. "
                             f"--workers 1 (the default) is the old single-process "
                             f"path, byte for byte")
    parser.add_argument("--reuse-park", action="store_true",
                        default=DEFAULT_REUSE_PARK,
                        help="OFF by default.  --mode pair: PARK once per "
                             "(prefix, takeover, budget) group instead of once "
                             "per cell, and spawn one child per (cell, arm) from "
                             "that one parked parent (ForkParent.spawn, the same "
                             "call v830_bombpol.run_episode36 already makes for "
                             "every arm).  One PARK is ~1.8 s (8,663 prefix "
                             "frames + 2,165 hash checks) and depends on exactly "
                             "those three fields -- not on the checkpoint, the "
                             "pair_rng or the arm -- so a 96-cell pilot drops "
                             "from 96 PARKs (~2.9 min) to one per group.  "
                             "Grouping is fail-closed: a different takeover / "
                             "tape / budget, and any other PARK input (window, "
                             "data dir, engine digest, tape target), opens a NEW "
                             "group instead of being mixed, and every group is "
                             "re-verified from its roots before anything starts.  "
                             "A SPAWN/QUIT failure makes that cell invalid "
                             "(invalid=true + error) and NEVER re-PARKs; a parent "
                             "that dies after the PARK voids the WHOLE group and "
                             "the run exits non-zero.  With the flag on the rows "
                             "gain park_reuse {group, park_id, "
                             "children_spawned, children_failed}, run.json gains "
                             "a park_reuse block, and the jsonl is written in "
                             "group order (group by first appearance, cells in "
                             "manifest order).  Refused with --mode trace / "
                             "compare and with --workers N > 1 (there one cell = "
                             "one process, so no PARK can be shared).  With the "
                             "flag absent nothing changes, byte for byte")
    parser.add_argument("--engine", default=None,
                        help="engine binary; defaults to the manifest's "
                             "'engine' key, then to the frozen WSL path. The "
                             "sha256 gate never relaxes")
    parser.add_argument("--data-source", default=None,
                        help="read-only game data template; defaults to the "
                             "manifest's 'data_source' key, then to the frozen "
                             "WSL path. v450.prepare_data_dir mirrors it")
    parser.add_argument("--work-dir", type=pathlib.Path,
                        default=WORK_DEFAULT,
                        help=f"root for runs/<tag> and data/<tag> "
                             f"(default {WORK_DEFAULT})")
    parser.add_argument("--tag", default=None,
                        help="unique name owning data/<tag> and runs/<tag>; "
                             "mandatory for pair/trace, refused if it already "
                             "exists (MAP sec. 9-3). Defaults to a derived name in "
                             "check mode")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="episode jsonl (default <work>/eval/<tag>.jsonl); "
                             "must not exist yet. In --mode compare it is the "
                             "compare document; in --mode check it is the plan "
                             "document; with --mode check --workers N it is "
                             "NOT written, because there it names mode=pair's "
                             "merge target and a dry run must not create it")
    parser.add_argument("--log", type=pathlib.Path, default=None,
                        help="run log (default <work>/runs/<tag>/run.log)")
    parser.add_argument("--allow-exposed", action="store_true",
                        help="pass --allow-exposed to the guard: accept a root "
                             "whose used_for_training/used_for_selection is "
                             "true (old dev windows are exposed)")
    parser.add_argument("--lineage-config", type=pathlib.Path, default=None,
                        metavar="PATH",
                        help="cross-check the declared exposure against the frozen "
                             "genealogy (e.g. configs/v430/splits.json): every root "
                             "must declare lineage_ancestor_ids, each id is looked "
                             "up in splits.{train,dev,confirm,adapt,quarantine}"
                             ".ancestor_ids plus the pretraining_sources registry "
                             "that the config's registries[] names. Missing field "
                             "-> LINEAGE_MISSING; declared false/false while the id "
                             "is in train/dev -> EXPOSURE_CONTRADICTS_LINEAGE "
                             "(both refused); declared exposed but unknown -> "
                             "LINEAGE_UNKNOWN (recorded in "
                             "run.json.lineage_unmatched, not refused). A root may "
                             "instead declare 'lineage_new': true for an ancestor "
                             "the frozen genealogy does not enumerate at all: empty "
                             "lineage_ancestor_ids is then expected and the verdict "
                             "is the recorded code LINEAGE_NEW (exit 0, named in "
                             "run.json.lineage_new_roots, with residual_risk 'new "
                             "ancestor: ...'); 'lineage_new': true plus an id that "
                             "IS in train/dev is EXPOSURE_CONTRADICTS_LINEAGE. Off "
                             "by default: without it run.json records "
                             "lineage_checked=false and that exposure is "
                             "self-declared")

    parser.add_argument("--allow-legacy-window", action="store_true",
                        help="required by pair/trace: these four windows are the "
                             "historical ones pre-registration sec. 2 excludes from "
                             "new samples, so a run on them is a smoke/bridge run")
    parser.add_argument("--keep-trace", action="store_true",
                        help="write <out-stem>.<ARM>[.ep<N>].trace.json with one "
                             "entry per decision (frame, threshold, nearest_px, "
                             "cand, masks, bomb_request, inventories); required "
                             "input for --mode compare")
    parser.add_argument("--trace-g1", type=pathlib.Path, default=None,
                        help="compare: the G1 trace document")
    parser.add_argument("--trace-t2", type=pathlib.Path, default=None,
                        help="compare: the T2 trace document")
    parser.add_argument("--explain-path", metavar="PATH", default=None,
                        help="print the write-fence verdict for PATH and exit 0 "
                             "(allowed) / 1 (denied), without --mode, without "
                             "reading the manifest and without writing anything. "
                             "It is the same fence_verdict() the write targets go "
                             "through, so it answers 'which rule decided my --out'")
    args = parser.parse_args(argv)
    # Whether the tier was requested explicitly or is the 2400 default: recorded in
    # budget_selection.requested_source so a reader can tell "the manifest's only
    # tier happened to be the default" from "the default actually selected it".
    # Matched by prefix because argparse accepts --budget / --budget=1800 too.
    args.budget_frames_source = ("cli" if any(
        str(item).split("=", 1)[0].startswith("--budget")
        for item in (sys.argv[1:] if argv is None else argv)) else "default")
    return args


def mode_explain_path(raw: list[str]) -> int:
    """``--explain-path PATH``: the write fence, offline, for any path.

    Handled before ``build_args`` so it needs no ``--mode``/``--manifest`` and
    cannot mutate anything: it resolves the path, judges it with the same
    ``fence_verdict()`` every write target uses, and prints the rule that fired.
    Exit 0 = allowed, 1 = denied (the codes a write would produce).
    """

    path_value = None
    work_value = None
    index = 0
    while index < len(raw):
        item = str(raw[index])
        if item == "--explain-path" and index + 1 < len(raw):
            path_value = str(raw[index + 1])
            index += 2
            continue
        if item.startswith("--explain-path="):
            path_value = item.split("=", 1)[1]
            index += 1
            continue
        if item == "--work-dir" and index + 1 < len(raw):
            work_value = str(raw[index + 1])
            index += 2
            continue
        if item.startswith("--work-dir="):
            work_value = item.split("=", 1)[1]
            index += 1
            continue
        index += 1
    if not path_value:
        print("--explain-path needs a PATH", file=sys.stderr)
        return EXIT_UNUSABLE
    work = pathlib.Path(work_value) if work_value else WORK_DEFAULT
    block = fence_verdict(pathlib.Path(path_value), work_dir=work,
                          what="path")
    print(f"path        {block['target']}")
    print(f"top-level   {block['top_level']}")
    print(f"work-dir    {block['work_dir']}")
    print(f"verdict     {block['verdict'].upper()}")
    print(f"rule        {block['rule']}")
    print(f"basis       {block['basis']}")
    print(f"reason      {block['reason']}")
    print("--- json summary ---")
    print(json.dumps(block, indent=1, ensure_ascii=False))
    return EXIT_OK if block["verdict"] != "deny" else EXIT_REFUSED


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # the fence explainer is answered before argparse so it needs no --mode and
    # cannot be combined with a mode that would start something
    if any(str(item) == "--explain-path"
           or str(item).startswith("--explain-path=") for item in raw):
        return mode_explain_path(raw)
    args = build_args(argv)

    workers = int(getattr(args, "workers", DEFAULT_WORKERS))
    if workers < 1 or workers > MAX_WORKERS:
        print(f"REFUSED [WORKERS_OUT_OF_RANGE] --workers must be 1..{MAX_WORKERS}, "
              f"got {workers}", file=sys.stderr)
        return EXIT_UNUSABLE
    if workers > 1 and args.mode not in ("pair", "check"):
        # check is included on purpose: `--mode check --workers N` is the offline
        # rehearsal of the very plan mode=pair would execute (see mode_check_workers)
        print(f"REFUSED [WORKERS_MODE_UNSUPPORTED] --workers {workers} is only "
              f"defined for --mode pair (a pair cell is what fans out) and for "
              f"--mode check (which only *prints* that plan); got "
              f"--mode {args.mode}", file=sys.stderr)
        return EXIT_UNUSABLE
    if park_reuse_enabled(args):
        # --reuse-park is only meaningful where cells can share one process's
        # PARK: one PARK per cell process cannot be shared across processes, so
        # --workers N > 1 would silently hand back only 1/N of the saving
        if args.mode not in ("pair", "check"):
            print(f"REFUSED [PARK_REUSE_MODE_UNSUPPORTED] --reuse-park is "
                  f"defined for --mode pair (many cells, one PARK per group) and "
                  f"for --mode check (which only *prints* that grouping); got "
                  f"--mode {args.mode}", file=sys.stderr)
            return EXIT_UNUSABLE
        if workers > 1:
            print(f"REFUSED [PARK_REUSE_WORKERS_UNSUPPORTED] --reuse-park with "
                  f"--workers {workers}: one worker process runs exactly one cell "
                  f"(pre-registration sec. 3.4), so no two cells can ever share "
                  f"one PARK and the flag would silently save nothing; run "
                  f"--workers 1 --reuse-park instead", file=sys.stderr)
            return EXIT_UNUSABLE

    if args.mode == "compare":
        if args.trace_g1 is None or args.trace_t2 is None:
            raise SystemExit("--mode compare needs --trace-g1 and --trace-t2 "
                             "(two --keep-trace documents)")
        try:
            document = compare_traces(str(args.trace_g1), str(args.trace_t2))
            print(render_compare(document))
            print("--- json summary ---")
            print(json.dumps(document, indent=1, ensure_ascii=False))
            if args.out:
                write_json(assert_writable(args.out,
                                           work_dir=pathlib.Path(args.work_dir),
                                           what="--out"), document)
                print(f"wrote: {args.out}")
        except PairError as error:
            print(f"REFUSED {error}", file=sys.stderr)
            return error.exit_code
        return EXIT_OK if document["verdict"] == "IDENTICAL" else EXIT_REFUSED

    if args.manifest is None:
        raise SystemExit(f"--mode {args.mode} needs --manifest")

    needs_engine = args.mode in ("trace", "pair")
    try:
        # the takeover frame is parsed as *text* above so that a non-integer has a
        # code of its own instead of argparse's bare usage error
        args.takeover_frame = coerce_takeover_frame(args.takeover_frame)
        guard_doc, guard_rc = manifest_document(args)
        print(m829().render(guard_doc))
        if guard_rc != 0:
            print(f"REFUSED [MANIFEST_REFUSED] the guard refused this manifest "
                  f"(exit {guard_rc}); v829_pair never re-implements or relaxes "
                  f"a guard rule", file=sys.stderr)
            return guard_rc
        # A: one read, digest asserted against the guard's, then parsed from the
        #    very same buffer -- never a second load_json(args.manifest).
        manifest, closure = manifest_from_single_read(args, guard_doc)
        roots = select_roots(manifest, args)
        if args.mode == "pair":
            args.arm_order = parse_arm_order(args.arm_order)
            if set(args.arm_order) != set(ARMS):
                raise PairError(
                    "ARM_ORDER_INCOMPLETE",
                    f"pair mode needs both arms, got {args.arm_order}; a pair is "
                    f"G1 and T2 from the same PARK (use --mode trace for one arm)")
        single_arm = None
        if args.mode == "trace":
            if args.arm is None:
                raise PairError("ARM_UNKNOWN", "--mode trace needs --arm G1|T2")
            single_arm = check_arm(args.arm)
        if int(args.episodes) < 1:
            raise PairError("EPISODES_INVALID",
                            f"--episodes must be >= 1, got {args.episodes}")
        plan = preflight(args, guard_doc=guard_doc, manifest=manifest,
                         roots=roots, needs_engine=needs_engine, closure=closure)
        if args.mode == "check":
            print(render_plan(plan))
            if workers > 1:
                # s11 refused this combination; it is the offline rehearsal of
                # exactly what mode=pair --workers N would do (no spawn, no
                # engine-side import, no write beyond runs/<tag>/run.json)
                return mode_check_workers(args, plan=plan, roots=roots,
                                          workers=workers)
            if park_reuse_enabled(args):
                # the offline rehearsal of the grouping itself: the same
                # park_reuse_plan()/park_reuse_verify() a real run uses, drawn
                # without importing one engine-side module (check mode asserts
                # that afterwards)
                reuse_tag = str(args.tag or (plan["plans"][0]["tag"]
                                             if plan.get("plans") else "check"))
                reuse_plan = park_reuse_plan(
                    roots, work=pathlib.Path(args.work_dir), tag=reuse_tag,
                    enabled=True)
                plan["park_reuse"] = park_reuse_document(reuse_plan)
                print(render_park_reuse(reuse_plan))
            loaded = plan["engine_side_modules_imported"]
            if loaded:
                print(f"REFUSED [ENGINE_SIDE_IMPORT] check mode must not touch "
                      f"the engine side, but these modules are loaded: {loaded}",
                      file=sys.stderr)
                return EXIT_REFUSED
            print("--- json summary ---")
            print(json.dumps(plan, indent=1, ensure_ascii=False))
            if args.out:
                write_json(args.out, plan)
                print(f"wrote: {args.out}")
            written = write_check_record(args=args, plan=plan)
            for path in written:
                print(f"wrote: {path}   (mode=check invocation record)")
            return EXIT_OK
        if args.mode == "pair" and workers > 1:
            # one process per cell, each with its own tag/runs/data (sec. 3.4)
            return mode_workers(args, guard_doc=guard_doc, manifest=manifest,
                                roots=roots, plan=plan)
        return mode_run(args, guard_doc=guard_doc, manifest=manifest,
                        roots=roots, plan=plan, single_arm=single_arm)
    except PairError as error:
        print(f"REFUSED {error}", file=sys.stderr)
        return error.exit_code
    except Exception as error:                                    # noqa: BLE001
        # fail closed on anything unexpected: never a zero exit, never a silent
        # partial run.  The traceback stays on stderr for the post-mortem.
        import traceback

        traceback.print_exc()
        print(f"REFUSED [UNEXPECTED] {type(error).__name__}: {error} "
              f"(treated as a technical failure; see the traceback above)",
              file=sys.stderr)
        return EXIT_UNUSABLE


if __name__ == "__main__":
    raise SystemExit(main())
