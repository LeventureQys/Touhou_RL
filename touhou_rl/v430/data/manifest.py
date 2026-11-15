"""v4.3.0 W03 — dataset manifest: load, fail-closed validation, partial view.

``DEVELOPMENT_GUIDE.md`` §6.1 (Manifest) fixes the minimum content of a dataset
manifest and — in one sentence — the whole job of this module:

    loader 拒绝缺 shard、SHA 不符、schema/digest 错配、额外未登记 shard 及
    train/dev 祖先交叉。状态为 partial 时只可明确读取已完成、不可变的 shard 作
    开发，不能把整个 partial 数据集判通过。

So this module answers exactly two questions and refuses to blur them:

* ``validate_manifest`` — "may I believe this manifest, byte for byte, **and** is
  it a *complete* dataset?"  The answer is a report with ``valid`` **and**
  ``complete`` as two different booleans, because a partial-but-consistent
  dataset is a real, usable thing that must never be able to masquerade as a
  finished one;
* ``partial_view`` — "which shards may I actually read as development data?"
  Only ``status == "completed"`` **and** ``immutable is True`` shards, explicitly
  labelled as not representing the whole dataset.

Design rules (the same ones :mod:`touhou_rl.v430.data.lineage` follows):

* **pure stdlib** — no numpy, no torch, no engine, no pixel I/O.  The only
  imports are ``dataclasses``/``json``/``pathlib``/``typing`` plus the sibling
  :mod:`~touhou_rl.v430.data.lineage` and
  :mod:`~touhou_rl.v430.contracts` (for the frozen exit codes);
* **one digest implementation** — every sha256 in this module comes from
  :func:`lineage.text_lf_digest` / :func:`lineage.binary_raw_digest` /
  :func:`lineage.digest_bytes` / :func:`lineage.load_json_text`.  A text
  artefact (a manifest JSON) is hashed on its ``CRLF -> LF`` normalised bytes and
  labelled ``text/lf-normalized``; a shard data file is hashed on its raw bytes
  and labelled ``binary/raw``.  There is no second implementation and no
  unlabelled digest anywhere: an identity field carrying ``sha256`` without a
  ``digest_scope`` is a hard rejection (``P_DIGEST_SCOPE_MISSING``);
* **fail-closed** — :func:`validate_manifest` never returns a sentinel that a
  caller could mistake for "passed", and the *absence* of a check is listed in
  ``not_checked`` instead of being silently skipped;
* **no truth, no fitting** — nothing here decides what data to train on.  It
  decides whether a recorded dataset identity is internally consistent and
  whether its shard bytes are the ones the manifest names.

Fail-closed rejections (stable ``P_*`` codes, one per §6.1 rule):

======================================  ====================================
code                                    rejects
======================================  ====================================
``P_SCHEMA``                            an unknown ``schema``
``P_MISSING_FIELD``                     a missing required field (named)
``P_FIELD_TYPE``                        a field of the wrong JSON type
``P_STATUS_UNKNOWN``                    an unknown manifest/shard status
``P_SHARD_MISSING``                     a shard file that is not on disk
``P_SHARD_SIZE_MISMATCH``               ``size_bytes`` != bytes on disk
``P_SHARD_SHA_MISMATCH``                ``sha256`` != digest of bytes on disk
``P_SHARD_UNREGISTERED``                a file under an explicit ``roots`` that
                                        the manifest does not list
``P_ANCESTOR_CROSSING``                 one ancestor group in two splits
                                        (``lineage.group_key`` semantics)
``P_OBS_DIGEST_MISMATCH``               ``obs_digest`` != the expected value
``P_ACTION_DIGEST_MISMATCH``            ``action_digest`` != the expected value
``P_LABEL_DIGEST_MISMATCH``             ``label_digest`` != the expected value
``P_SPLIT_REGISTRY_SHA_MISMATCH``       ``split_registry_sha`` != the sha256 of
                                        the supplied split config
``P_PARTIAL_INCOMPLETE``                a ``status="partial"`` manifest offered
                                        as a complete dataset
``P_DIGEST_SCOPE_MISSING``              a ``sha256`` without a ``digest_scope``
``P_DIGEST_SCOPE_UNKNOWN``              a ``digest_scope`` outside
                                        :data:`lineage.DIGEST_SCOPES`
``P_DIGEST_NOT_HEX``                    a digest that is not 64 lowercase hex
``P_SHARD_DUPLICATE``                   the same shard path listed twice
``P_SHARD_PATH_INVALID``                an absolute/escaping shard path
``P_SHARD_NOT_FILE``                    a registered path that is not a file
``P_SHARD_UNREADABLE``                  a registered shard that cannot be read
``P_SHARD_SPLIT_UNKNOWN``               a shard ``split`` outside
                                        :data:`lineage.ALL_SPLIT_NAMES`
``P_SHARD_STATUS_UNKNOWN``              a shard ``status`` outside
                                        :data:`KNOWN_SHARD_STATUSES`
``P_SHARD_NOT_COMPLETED``               a ``status="complete"`` manifest lists
                                        a shard that is not completed and
                                        immutable
``P_ANCESTOR_INVALID``                  an ancestor record that cannot be turned
                                        into ``(task_family, episode)``
``P_LABELS_EXCEED_DECISIONS``           ``labels_valid > decisions``
``P_FRAME_RANGE_INVALID``               a frame range that is not
                                        ``[start, end)`` with ``start <= end``
``P_DUPLICATES_INVALID``                a ``duplicates`` field that is neither a
                                        non-negative int nor a mapping
``P_COUNTS_MISMATCH``                   ``counts`` != the sum over the shards
``P_ROOTS_INVALID``                     ``roots`` is not ``{name: directory}``
``P_SPLIT_CONFIG_UNREADABLE``           the supplied split config cannot be
                                        loaded by ``lineage.validate_splits``
``P_OBS_CONTRACT_UNKNOWN``              ``obs_contract`` is not one of
                                        :data:`KNOWN_OBS_CONTRACTS`
``P_OBS_SHAPE_CONTRACT_MISMATCH``       a manifest labelled ``obs-v430-o2``
                                        declares ``obs_shapes`` that disagree
                                        with ``observation.O2_SHAPES``
======================================  ====================================

Observation contract (the one rule this module adds on top of §6.1): a manifest
must **say which observation contract its pixels were recorded under**
(``obs_contract``, one of :data:`KNOWN_OBS_CONTRACTS`) and
:func:`assert_o2_compatible` refuses to hand anything but ``obs-v430-o2`` to the
O2 model.  A corpus captured under the v4.2.0 downsampled observation
(``obs-v420-2f``: no ``global_rgb``, ``play_gray`` at ``(4,112,96)`` = 2x
downsampled, ``hud_gray`` at ``(56,160)`` = 8x vertically downsampled) is a
legitimate dataset and still *validates*, but it is **not** O2 training data and
there is deliberately no conversion path: nothing in this module resamples, pads
or synthesises a missing channel.  Re-capture under ``obs-v430-o2`` instead.

``P_PARTIAL_INCOMPLETE`` is the only *advisory* code: in ``strict=False`` mode it
is reported under ``degradations`` (the completed shards are still checked and
``complete`` stays ``False``), which is what :func:`partial_view` uses.
Everything else is fatal in both modes.

A synthetic test fixture lives in ``tests/test_v430_data.py``; it is **not**
evidence that any real dataset, capability or gate passed.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..contracts import EXIT_INCOMPLETE, EXIT_INVALID, EXIT_OK
from . import lineage
from .lineage import (
    DIGEST_SCOPES,
    DIGEST_SCOPE_TEXT,
    LineageError,
    ancestor_id,
    digest_bytes,
    group_key,
    load_json_text,
    text_lf_digest,
)

__all__ = [
    # identity
    "MANIFEST_SCHEMA",
    "REPORT_SCHEMA",
    "MANIFEST_STATUS_COMPLETE",
    "MANIFEST_STATUS_PARTIAL",
    "SHARD_STATUS_COMPLETED",
    "KNOWN_MANIFEST_STATUSES",
    "KNOWN_SHARD_STATUSES",
    "REQUIRED_MANIFEST_FIELDS",
    "REQUIRED_SHARD_FIELDS",
    "IDENTITY_FIELDS",
    "ADVISORY_CODES",
    "PROBLEM_CODES",
    "PARTIAL_DISCLAIMER",
    # observation contract
    "OBS_CONTRACT_O2",
    "LEGACY_OBS_CONTRACTS",
    "KNOWN_OBS_CONTRACTS",
    "O2_OBSERVATION_CHANNELS",
    # types
    "ManifestError",
    "Shard",
    "Manifest",
    "PartialView",
    # entry points
    "load_manifest",
    "validate_manifest",
    "assert_o2_compatible",
    "o2_compatible",
    "partial_view",
    "exit_code_for",
]

# ---------------------------------------------------------------------------
# 1. Schemas, statuses and the field lists of §6.1
# ---------------------------------------------------------------------------
#: Schema of a dataset manifest.  Anything else is refused: the field semantics
#: differ per schema, so "close enough" is not a thing this loader accepts.
MANIFEST_SCHEMA = "v430-manifest/1"
#: Schema of the report returned by :func:`validate_manifest`.
REPORT_SCHEMA = "v430-manifest-validation/1"

#: The dataset is finished: every shard is completed and immutable.
MANIFEST_STATUS_COMPLETE = "complete"
#: The dataset is still being written.  Only explicitly completed, immutable
#: shards may be read as development data (see :func:`partial_view`).
MANIFEST_STATUS_PARTIAL = "partial"
KNOWN_MANIFEST_STATUSES: tuple[str, ...] = (
    MANIFEST_STATUS_COMPLETE, MANIFEST_STATUS_PARTIAL)

#: A shard that is closed: no byte may change any more.
SHARD_STATUS_COMPLETED = "completed"
#: ``writing`` is still being appended to; ``quarantined`` holds barred records;
#: ``failed`` is a known-bad shard kept only as evidence.  None of the three is
#: readable as training data, and none of them is "completed".
KNOWN_SHARD_STATUSES: tuple[str, ...] = (
    SHARD_STATUS_COMPLETED, "writing", "quarantined", "failed")

#: The §6.1 minimum, in the order the guide lists it.  ``counts`` is last only
#: because the guide lists it last; it is still mandatory.
REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "schema",
    "dataset_id",
    "parent_ids",
    "code_snapshot",
    "engine_sha",
    "game_data_manifest_sha",
    "obs_digest",
    "action_digest",
    "label_digest",
    "split_registry_sha",
    "shards",
    "counts",
    "status",
    #: Which observation contract the recorded pixels obey.  Required, because a
    #: manifest that does not name its contract cannot be told apart from one
    #: recorded under a *different* encoder (``obs_digest`` binds the parameter
    #: block, not the shape of the arrays that were actually written).
    "obs_contract",
)

#: The **only** observation contract an O2 model may be trained or evaluated on:
#: ``global_rgb``/``play_gray``/``hud_gray``/``executed_key_history`` at
#: ``observation.O2_SHAPES``.
OBS_CONTRACT_O2 = "obs-v430-o2"

#: Contracts that exist on disk and are *named*, but are **not** O2.  The value is
#: the exact reason it cannot be promoted, quoted back by
#: :func:`assert_o2_compatible`.  ``obs-v420-2f`` is the real value read from
#: ``.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed10-shard/meta.json``
#: (``contract.obs_contract``): every corpus in this repository stores it.
LEGACY_OBS_CONTRACTS: dict[str, str] = {
    "obs-v420-2f": (
        "it stores the v4.2.0 downsampled observation, not O2: global_rgb is "
        "absent from it entirely, play_gray is (4,112,96) = exactly 2x "
        "downsampled from O2's (4,224,192), and hud_gray is (56,160) = 8x "
        "vertically downsampled from O2's (1,448,160)"
    ),
}

#: Every ``obs_contract`` value this module will accept on a manifest.  An
#: unregistered value is ``P_OBS_CONTRACT_UNKNOWN`` — a corpus cannot be
#: *relabelled* into the contract it was not captured under.
KNOWN_OBS_CONTRACTS: tuple[str, ...] = (OBS_CONTRACT_O2,) + tuple(LEGACY_OBS_CONTRACTS)

#: The per-frame channels an O2 corpus must record, and the only ones whose
#: geometry :func:`assert_o2_compatible` will compare.  ``memory`` is the model's
#: own recurrent state and is never stored in a corpus; ``history_valid`` and
#: ``elapsed_capture_s`` are derived at encode time.
O2_OBSERVATION_CHANNELS: tuple[str, ...] = (
    "global_rgb",
    "play_gray",
    "hud_gray",
    "executed_key_history",
)

#: The §6.1 per-shard minimum ("每个 shard 有相对路径、字节数、SHA、episode/
#: ancestor 列表、帧区间、实际决策数、标签有效数、重复情况").  ``split`` is
#: required as well: without it the train/dev ancestor check has nothing to test.
#: ``status``/``immutable`` are required *for a shard to be readable* — a partial
#: manifest that does not say a shard is finished and immutable cannot have it
#: include the shard, and :func:`partial_view` excludes it by name.
REQUIRED_SHARD_FIELDS: tuple[str, ...] = (
    "path",
    "size_bytes",
    "sha256",
    "digest_scope",
    "episodes",
    "ancestors",
    "frame_range",
    "decisions",
    "labels_valid",
    "duplicates",
    "split",
)

#: Fields that carry an identity digest.  Each one must present **both** its
#: sha256 and its ``digest_scope`` (``P_DIGEST_SCOPE_MISSING`` otherwise).
IDENTITY_FIELDS: tuple[str, ...] = (
    "engine_sha",
    "game_data_manifest_sha",
    "obs_digest",
    "action_digest",
    "label_digest",
    "split_registry_sha",
)

#: Codes that are fatal in strict mode and reported as degradations when the
#: caller accepts a partial-but-consistent dataset (``strict=False``).
ADVISORY_CODES = frozenset({"P_PARTIAL_INCOMPLETE"})

#: Every code this module can emit, with the one-line meaning.  A test asserts
#: that no unregistered code can escape, so a new rejection cannot be added
#: without naming it.
PROBLEM_CODES: dict[str, str] = {
    "P_SCHEMA": "unknown manifest schema",
    "P_MISSING_FIELD": "a required field is absent",
    "P_FIELD_TYPE": "a field has the wrong JSON type",
    "P_STATUS_UNKNOWN": "unknown manifest or shard status",
    "P_SHARD_MISSING": "a registered shard file does not exist",
    "P_SHARD_SIZE_MISMATCH": "size_bytes disagrees with the bytes on disk",
    "P_SHARD_SHA_MISMATCH": "sha256 disagrees with the bytes on disk",
    "P_SHARD_UNREGISTERED": "a file under an explicit root is not registered",
    "P_ANCESTOR_CROSSING": "one ancestor group appears in two splits",
    "P_OBS_DIGEST_MISMATCH": "obs_digest disagrees with the expected value",
    "P_ACTION_DIGEST_MISMATCH": "action_digest disagrees with the expected value",
    "P_LABEL_DIGEST_MISMATCH": "label_digest disagrees with the expected value",
    "P_SPLIT_REGISTRY_SHA_MISMATCH": "split_registry_sha disagrees with the config",
    "P_SPLIT_CONFIG_UNREADABLE": "the supplied split config cannot be loaded",
    "P_PARTIAL_INCOMPLETE": "a partial manifest is not a complete dataset",
    "P_DIGEST_SCOPE_MISSING": "a sha256 is not labelled with its digest_scope",
    "P_DIGEST_SCOPE_UNKNOWN": "digest_scope is not text/lf-normalized or binary/raw",
    "P_DIGEST_NOT_HEX": "a digest is not 64 lowercase hex characters",
    "P_SHARD_DUPLICATE": "the same shard path is listed twice",
    "P_SHARD_PATH_INVALID": "a shard path is absolute or escapes its root",
    "P_SHARD_NOT_FILE": "a registered shard path is not a regular file",
    "P_SHARD_UNREADABLE": "a registered shard cannot be read",
    "P_SHARD_SPLIT_UNKNOWN": "a shard split is not one of the four splits",
    "P_SHARD_STATUS_UNKNOWN": "a shard status is not a known status",
    "P_SHARD_NOT_COMPLETED": "a complete manifest lists a shard that is not "
                             "completed and immutable",
    "P_ANCESTOR_INVALID": "an ancestor record cannot yield (task_family, episode)",
    "P_LABELS_EXCEED_DECISIONS": "labels_valid exceeds decisions",
    "P_FRAME_RANGE_INVALID": "frame_range is not a valid [start, end) pair",
    "P_DUPLICATES_INVALID": "duplicates is neither an int nor a mapping",
    "P_COUNTS_MISMATCH": "counts disagrees with the sum over the shards",
    "P_ROOTS_INVALID": "roots is not a {name: directory} mapping",
    "P_OBS_CONTRACT_UNKNOWN": "obs_contract is not a registered observation contract",
    "P_OBS_SHAPE_CONTRACT_MISMATCH": "a manifest labelled obs-v430-o2 declares "
                                     "obs_shapes that disagree with O2_SHAPES",
}

#: Wording that must accompany any partial view.  Kept as one constant so a
#: caller cannot paraphrase it into a completeness claim.
PARTIAL_DISCLAIMER = (
    "partial view: this selection of completed, immutable shards is NOT the "
    "complete dataset; it must not be cited as coverage, as an evaluation set "
    "boundary, or as evidence that the dataset finished"
)

#: Names of files that are never shard data.  Scanning an explicit root treats
#: every other regular file as a shard that the manifest must list; dotfiles are
#: skipped because version control and editors create them.
_SCAN_IGNORED_NAMES = frozenset({".gitkeep", "Thumbs.db", ".DS_Store"})


class ManifestError(RuntimeError):
    """Raised by :func:`load_manifest` when there is nothing to parse at all.

    A missing file, invalid JSON or a JSON document that is not an object cannot
    become a :class:`Manifest`, so this raises instead of returning a half-built
    one.  **Every** other rejection is a problem inside the report returned by
    :func:`validate_manifest` — a caller who only catches this exception has not
    validated anything.
    """


# ---------------------------------------------------------------------------
# 2. Small helpers (problems, digests, types)
# ---------------------------------------------------------------------------


def _problem(code: str, message: str, **extra: Any) -> dict[str, Any]:
    """One problem record.  ``code`` is stable; ``message`` is for a human."""

    if code not in PROBLEM_CODES:  # pragma: no cover - guarded by the code table
        raise ManifestError(f"unregistered problem code {code!r}")
    item: dict[str, Any] = {"code": code, "message": message}
    item.update({k: v for k, v in extra.items() if v is not None})
    return item


def _is_int(value: Any) -> bool:
    """A real ``int``; ``bool`` is refused (``True`` is not a byte count)."""

    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex64(value: Any) -> bool:
    """64 lowercase hex characters — the only shape a sha256 is accepted in."""

    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


def _posix(text: str) -> str:
    return text.replace("\\", "/").strip()


def _iter_sha_nodes(obj: Any, *, path: str = "", depth: int = 0) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Yield every mapping that carries a ``sha256`` key, with its dotted path."""

    if depth > 12:
        return
    if isinstance(obj, Mapping):
        if "sha256" in obj:
            yield path or "<root>", obj
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _iter_sha_nodes(value, path=child, depth=depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            yield from _iter_sha_nodes(item, path=f"{path}[{index}]", depth=depth + 1)


def _identity_record(
    parent: Mapping[str, Any], field: str, *, where: str
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    """Read one identity field as ``(sha256, digest_scope, problems)``.

    Two encodings are accepted, both of which keep the scope *next to* the
    digest: the canonical record ``{"sha256": ..., "digest_scope": ...}`` and the
    flat form ``"<hex64>"`` with a sibling ``<field>_scope`` (the shape
    ``lineage.validate_splits`` reports).  A bare sha256 with neither is refused:
    an unlabelled digest cannot be reproduced, so it is not an identity.
    """

    problems: list[dict[str, Any]] = []
    value = parent[field]
    sibling = parent.get(f"{field}_scope", parent.get("digest_scope"))
    sha: Any = None
    scope: Any = sibling
    if isinstance(value, Mapping):
        sha = value.get("sha256")
        if value.get("digest_scope") is not None:
            scope = value.get("digest_scope")
    elif isinstance(value, str):
        sha = value
    else:
        problems.append(_problem(
            "P_FIELD_TYPE",
            f"{where} must be a sha256 string or a {{sha256, digest_scope}} "
            f"record, got {type(value).__name__}",
            field=where))
        return None, None, problems

    if sha is None:
        problems.append(_problem(
            "P_FIELD_TYPE",
            f"{where} carries no sha256",
            field=where))
        return None, None, problems
    if not _is_hex64(sha):
        problems.append(_problem(
            "P_DIGEST_NOT_HEX",
            f"{where}.sha256 {sha!r} is not 64 lowercase hex characters",
            field=where, sha256=sha if isinstance(sha, str) else repr(sha)))
        sha = None
    if scope is None:
        problems.append(_problem(
            "P_DIGEST_SCOPE_MISSING",
            f"{where} carries sha256 without digest_scope; every identity must be "
            f"labelled {DIGEST_SCOPES[0]!r} or {DIGEST_SCOPES[1]!r}",
            field=where, sha256=sha))
    elif not isinstance(scope, str) or scope not in DIGEST_SCOPES:
        problems.append(_problem(
            "P_DIGEST_SCOPE_UNKNOWN",
            f"{where}.digest_scope {scope!r} is not one of {list(DIGEST_SCOPES)}",
            field=where))
    return (sha if isinstance(sha, str) else None), scope, problems


def _expected_sha(value: Any, name: str) -> str:
    """Normalise a caller-supplied expected digest; a bad one is a *caller* bug.

    This is the only place in the module that raises for a bad digest: an
    expectation is program code, not dataset content, so it cannot be reported as
    "the data is wrong".
    """

    if isinstance(value, Mapping):
        value = value.get("sha256")
    if not _is_hex64(value):
        raise ManifestError(
            f"expected {name} must be 64 lowercase hex characters or a record "
            f"carrying one, got {value!r}")
    return value


def _as_string_list(value: Any) -> list[str] | None:
    """``None`` unless ``value`` is a list/tuple of non-empty strings."""

    if not isinstance(value, (list, tuple)):
        return None
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        out.append(item.strip())
    return out


# ---------------------------------------------------------------------------
# 3. Manifest / shard records
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Shard:
    """One manifest shard entry, parsed — **not** yet validated.

    Every field defaults to ``None`` when the entry did not carry it, so a
    missing field is reported by name (:data:`REQUIRED_SHARD_FIELDS`) instead of
    aborting the load halfway.  Reading these values without calling
    :func:`validate_manifest` first is meaningless: nothing here has been checked
    against the disk.
    """

    index: int
    path: Any = None
    size_bytes: Any = None
    sha256: Any = None
    digest_scope: Any = None
    episodes: Any = None
    ancestors: Any = None
    frame_range: Any = None
    decisions: Any = None
    labels_valid: Any = None
    duplicates: Any = None
    split: Any = None
    status: Any = None
    immutable: Any = None
    raw: Any = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def identifier(self) -> str:
        """Stable human identifier for messages: the path, else the index."""

        if isinstance(self.path, str) and self.path.strip():
            return self.path.strip()
        return f"shards[{self.index}]"

    @classmethod
    def from_mapping(cls, index: int, item: Any) -> "Shard":
        if not isinstance(item, Mapping):
            return cls(index=index, raw=None)
        return cls(
            index=index,
            path=item.get("path"),
            size_bytes=item.get("size_bytes"),
            sha256=item.get("sha256"),
            digest_scope=item.get("digest_scope"),
            episodes=item.get("episodes"),
            ancestors=item.get("ancestors"),
            frame_range=item.get("frame_range"),
            decisions=item.get("decisions"),
            labels_valid=item.get("labels_valid"),
            duplicates=item.get("duplicates"),
            split=item.get("split"),
            status=item.get("status"),
            immutable=item.get("immutable"),
            raw=dict(item),
        )

    @property
    def is_readable(self) -> bool:
        """``status == "completed"`` **and** ``immutable is True``.

        Both halves are required: a completed shard that may still be rewritten
        is not immutable, and an immutable shard that is not completed is not
        data.  A shard that does not *say* it is immutable is not treated as
        immutable (fail closed).
        """

        return self.status == SHARD_STATUS_COMPLETED and self.immutable is True

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw) if isinstance(self.raw, Mapping) else {}


@dataclasses.dataclass(frozen=True)
class Manifest:
    """A parsed dataset manifest (§6.1), **not** a validation result.

    :func:`load_manifest` only decides "is this a JSON object I can address by
    field".  Nothing is checked against the disk, no digest is compared and no
    ancestry is walked here; :func:`validate_manifest` is the only function whose
    answer may be cited.  ``raw`` keeps the decoded document verbatim (including
    fields this module does not interpret) and ``source`` is the file identity of
    the manifest JSON itself — a ``text/lf-normalized`` digest, because a manifest
    is a text artefact whose CRLF/LF copy must have the same identity.
    """

    schema: Any = None
    dataset_id: Any = None
    parent_ids: Any = None
    code_snapshot: Any = None
    engine_sha: Any = None
    game_data_manifest_sha: Any = None
    obs_digest: Any = None
    action_digest: Any = None
    label_digest: Any = None
    split_registry_sha: Any = None
    shards: tuple[Shard, ...] = ()
    counts: Any = None
    status: Any = None
    #: Which observation contract the pixels were recorded under (one of
    #: :data:`KNOWN_OBS_CONTRACTS`).  Declared, never inferred: see
    #: :func:`assert_o2_compatible`.
    obs_contract: Any = None
    #: Optional ``{channel: [dim, ...]}`` claim.  When present on an
    #: ``obs-v430-o2`` manifest it must agree with ``observation.O2_SHAPES``;
    #: when absent no shape claim was made (that fact is reported, not hidden).
    obs_shapes: Any = None
    #: Optional explicit data roots: ``{name: directory}``.  When present (or
    #: passed to :func:`validate_manifest`) the unregistered-shard scan runs.
    roots: Any = None
    raw: Mapping[str, Any] = dataclasses.field(default_factory=dict, repr=False, compare=False)
    source: Mapping[str, Any] | None = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def dataset(self) -> str:
        """Name used in report messages when ``dataset_id`` is unusable."""

        if isinstance(self.dataset_id, str) and self.dataset_id.strip():
            return self.dataset_id.strip()
        return "<unnamed dataset>"

    def to_dict(self) -> dict[str, Any]:
        """The manifest document as it was read (a shallow copy of ``raw``)."""

        return dict(self.raw)

    def shard_by_path(self) -> dict[str, Shard]:
        """``{posix path: shard}`` for every shard that names a path."""

        out: dict[str, Shard] = {}
        for shard in self.shards:
            if isinstance(shard.path, str) and shard.path.strip():
                out.setdefault(_posix(shard.path), shard)
        return out

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *,
                     source: Mapping[str, Any] | None = None) -> "Manifest":
        shards_value = payload.get("shards")
        shards: tuple[Shard, ...] = ()
        if isinstance(shards_value, (list, tuple)):
            shards = tuple(
                Shard.from_mapping(index, item)
                for index, item in enumerate(shards_value))
        return cls(
            schema=payload.get("schema"),
            dataset_id=payload.get("dataset_id"),
            parent_ids=payload.get("parent_ids"),
            code_snapshot=payload.get("code_snapshot"),
            engine_sha=payload.get("engine_sha"),
            game_data_manifest_sha=payload.get("game_data_manifest_sha"),
            obs_digest=payload.get("obs_digest"),
            action_digest=payload.get("action_digest"),
            label_digest=payload.get("label_digest"),
            split_registry_sha=payload.get("split_registry_sha"),
            shards=shards,
            counts=payload.get("counts"),
            status=payload.get("status"),
            obs_contract=payload.get("obs_contract"),
            obs_shapes=payload.get("obs_shapes"),
            roots=payload.get("roots"),
            raw=dict(payload),
            source=dict(source) if isinstance(source, Mapping) else None,
        )


@dataclasses.dataclass(frozen=True)
class PartialView:
    """The readable subset of a partial manifest, explicitly not the dataset.

    ``complete`` is always ``False`` — there is deliberately no parameter that
    could set it: a view of some shards is not a coverage claim about the
    dataset.  ``excluded`` names every shard that was left out and why, so the
    omission is auditable instead of silent.
    """

    manifest: Manifest
    source_status: str
    excluded: tuple[Mapping[str, Any], ...] = ()
    report: Mapping[str, Any] | None = None
    disclaimer: str = PARTIAL_DISCLAIMER
    complete: bool = False

    @property
    def shards(self) -> tuple[Shard, ...]:
        return self.manifest.shards

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "v430-manifest-partial-view/1",
            "dataset_id": self.manifest.dataset_id,
            "source_status": self.source_status,
            "complete": False,
            "disclaimer": self.disclaimer,
            "included": [shard.to_dict() for shard in self.shards],
            "excluded": [dict(item) for item in self.excluded],
            "report": dict(self.report) if isinstance(self.report, Mapping) else None,
        }


# ---------------------------------------------------------------------------
# 4. Loading
# ---------------------------------------------------------------------------


def load_manifest(path: str | Path, *, roots: Any = None) -> Manifest:
    """Read and parse a manifest JSON; raise :class:`ManifestError` if unreadable.

    The manifest is a **text** artefact, so its own identity is
    :func:`lineage.text_lf_digest` over the raw bytes (``text/lf-normalized``):
    the same document checked out with CRLF and with LF must load to the same
    ``source["sha256"]``.  Parsing goes through :func:`lineage.load_json_text`,
    which folds CRLF and tolerates a UTF-8 BOM for exactly the same reason.

    Parameters
    ----------
    path:
        The manifest file.  A relative path is resolved against the current
        working directory.
    roots:
        Optional explicit data roots ``{name: Path}`` (a sequence of directories
        is accepted and keyed by its own name).  They are stored on
        :attr:`Manifest.roots` and used by :func:`validate_manifest` for the
        unregistered-shard scan; a root passed there wins over this one.

    Returns
    -------
    Manifest
        The parsed document.  **It is not validated**; call
        :func:`validate_manifest` before believing anything in it.
    """

    manifest_path = Path(path)
    try:
        raw_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc
    try:
        payload = load_json_text(raw_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ManifestError(f"manifest {manifest_path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError(
            f"manifest {manifest_path} must be a JSON object, got "
            f"{type(payload).__name__}")

    source = {
        "path": _posix(str(manifest_path)),
        "sha256": text_lf_digest(raw_bytes),
        "digest_scope": DIGEST_SCOPE_TEXT,
        "size_bytes": len(raw_bytes),
        "size_bytes_raw": len(raw_bytes),
        "canonicalization": lineage.TEXT_CANONICALIZATION,
    }
    manifest = Manifest.from_mapping(payload, source=source)
    parsed_roots = _parse_roots(roots if roots is not None else manifest.roots)
    if parsed_roots is not None:
        manifest = dataclasses.replace(manifest, roots=parsed_roots)
    return manifest


def _parse_roots(roots: Any) -> dict[str, str] | None:
    """Normalise ``roots`` to ``{name: posix path}``; ``None`` when absent.

    Malformed roots are **not** repaired here: they are returned as ``{"<invalid>":
    repr}`` so the validator reports ``P_ROOTS_INVALID`` instead of the loader
    silently dropping the scan that the caller asked for.
    """

    if roots is None:
        return None
    if isinstance(roots, Mapping):
        out: dict[str, str] = {}
        for name, value in roots.items():
            out[str(name)] = _posix(str(value))
        return out
    if isinstance(roots, (str, Path)):
        return {"<invalid>": repr(roots)}
    if isinstance(roots, (list, tuple)):
        out = {}
        for value in roots:
            text = _posix(str(value))
            out[Path(text).name or text] = text
        return out
    return {"<invalid>": repr(roots)}


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------


def validate_manifest(
    manifest: Manifest | Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    strict: bool = True,
    roots: Any = None,
    expected_obs_digest: Any = None,
    expected_action_digest: Any = None,
    expected_label_digest: Any = None,
    split_config: Any = None,
    split_registry_sha: Any = None,
) -> dict[str, Any]:
    """Validate a manifest against the disk and against caller expectations.

    Never returns a silent pass: the report carries ``valid`` **and**
    ``complete`` as two separate booleans, ``problems`` (fatal), ``degradations``
    (advisory, only in ``strict=False``), ``checks_performed`` and the honest
    ``not_checked`` list.

    Parameters
    ----------
    manifest:
        A :class:`Manifest` from :func:`load_manifest`, or the raw mapping (it is
        parsed first, so a caller cannot skip the load step by accident).
    base_dir:
        Directory that relative shard paths resolve against.  Defaults to the
        manifest file's own directory when it was loaded from a path.
    strict:
        ``True`` (default): the caller wants a **complete** dataset, so a
        ``status="partial"`` manifest fails with ``P_PARTIAL_INCOMPLETE``.
        ``False``: the completed part is validated and may be ``valid`` while
        ``complete`` stays ``False``; every other rejection is still fatal.
    roots:
        Explicit data roots ``{name: directory}`` (or a sequence of directories).
        This is what enables the *unregistered shard* scan: every regular file
        under a root that no shard entry names is a ``P_SHARD_UNREGISTERED``
        problem.  Without roots the scan **does not run** and that fact is listed
        in ``not_checked`` — absence of the check is never reported as a pass.
    expected_obs_digest / expected_action_digest / expected_label_digest:
        Optional expected digests (64 lowercase hex, or a ``{sha256: ...}``
        record).  When supplied, a mismatch is ``P_OBS_DIGEST_MISMATCH`` /
        ``P_ACTION_DIGEST_MISMATCH`` / ``P_LABEL_DIGEST_MISMATCH``.
        When absent, the comparison is listed in ``not_checked``.
    split_config:
        The frozen split config (path or mapping) whose
        ``lineage.validate_splits`` sha256 must equal ``split_registry_sha``.
    split_registry_sha:
        An expected sha256 for ``split_registry_sha`` (used instead of
        ``split_config`` when both are absent from the caller's side).

    Returns
    -------
    dict
        ``{schema, dataset_id, status, mode, valid, complete, problems,
        degradations, checks_performed, not_checked, statistics, manifest,
        split_registry, exit_code}``.
    """

    if isinstance(manifest, Mapping):
        manifest = Manifest.from_mapping(manifest)
    if not isinstance(manifest, Manifest):
        raise ManifestError(
            f"validate_manifest needs a Manifest or a mapping, got "
            f"{type(manifest).__name__}")

    checks: list[str] = []
    problems: list[dict[str, Any]] = []
    not_checked: list[str] = [
        "shard payload bytes are hashed, never interpreted: this validator says "
        "the bytes are the ones the manifest names, not that they are good data",
        "counts['ancestors'] is not cross-checked: a distinct-ancestor count is "
        "not derivable from per-shard ancestor lists alone",
        "the split registry's own lineage rules belong to lineage.validate_splits; "
        "this validator only binds its sha256",
    ]
    if "obs_shapes" not in manifest.raw:
        not_checked.append(
            "obs_contract was bound by name only: the manifest declares no "
            "obs_shapes, so its per-channel geometry against "
            "observation.O2_SHAPES is unverified — a manifest may only claim the "
            "contract it was actually recorded under")

    # --- 1. identity --------------------------------------------------------
    checks.append("identity")
    problems.extend(_check_identity(manifest))

    # --- 2. shards: declared vs on disk ------------------------------------
    checks.append("shards")
    if base_dir is None and isinstance(manifest.source, Mapping):
        source_path = manifest.source.get("path")
        if isinstance(source_path, str) and source_path.strip():
            base_dir = Path(source_path).parent
    roots_eff = _parse_roots(roots) if roots is not None else manifest.roots
    root_paths: dict[str, Path] = {}
    if roots_eff is not None:
        root_paths, root_problems = _root_paths(roots_eff)
        problems.extend(root_problems)
    resolve = _resolver(base_dir, roots_eff)
    problems.extend(_check_shards(manifest, resolve))

    # --- 3. counts ----------------------------------------------------------
    checks.append("counts")
    problems.extend(_check_counts(manifest))

    # --- 4. ancestry --------------------------------------------------------
    checks.append("ancestors")
    groups, ancestor_problems = _ancestor_index(manifest)
    problems.extend(ancestor_problems)
    problems.extend(_crossing_problems(groups))

    # --- 5. one digest vocabulary ------------------------------------------
    checks.append("digest_scope")
    problems.extend(_check_digest_scopes(manifest))

    # --- 6. caller expectations --------------------------------------------
    # TODO(W03B): the manifest binds obs/action/label digests only at dataset
    # level, so a single shard recorded under a *different* observation encoder
    # is invisible here unless the caller also holds that shard's own digest.
    # Per-shard digests (or a cross-manifest lineage lookup) belong to the
    # `index` module of this package, which does not exist yet.
    expectations = (
        ("obs_digest", expected_obs_digest, "P_OBS_DIGEST_MISMATCH"),
        ("action_digest", expected_action_digest, "P_ACTION_DIGEST_MISMATCH"),
        ("label_digest", expected_label_digest, "P_LABEL_DIGEST_MISMATCH"),
    )
    supplied = [name for name, value, _code in expectations if value is not None]
    if supplied:
        checks.append("expected_digests")
    for field, expected, code in expectations:
        if expected is None:
            not_checked.append(
                f"{field} was not compared: no expected value was supplied")
            continue
        want = _expected_sha(expected, field)
        if field not in manifest.raw:
            continue  # already P_MISSING_FIELD from the identity check
        got, scope, _own = _identity_record(manifest.raw, field, where=field)
        if got is None:
            continue  # already P_DIGEST_* from the identity check
        if got != want:
            problems.append(_problem(
                code,
                f"{manifest.dataset}: {field} is {got} but the expected value is "
                f"{want} ({scope})",
                field=field, expected=want, actual=got, digest_scope=scope))

    # --- 7. split registry binding -----------------------------------------
    split_binding: dict[str, Any] | None = None
    if split_config is None and split_registry_sha is None:
        not_checked.append(
            "split_registry_sha was not compared: no split config or expected "
            "sha256 was supplied")
    else:
        checks.append("split_registry")
        split_binding, split_problems = _check_split_registry(
            manifest, split_config, split_registry_sha)
        problems.extend(split_problems)

    # --- 8. unregistered shards --------------------------------------------
    if roots_eff is None:
        not_checked.append(
            "the unregistered-shard scan did not run: it needs explicit roots "
            "(validate_manifest(..., roots={name: directory}))")
    else:
        checks.append("roots")
        problems.extend(_check_unregistered(manifest, root_paths, resolve))

    # --- 9. completeness / status ------------------------------------------
    checks.append("status")
    problems.extend(_check_status(manifest))

    degradations: list[dict[str, Any]] = []
    if not strict:
        kept: list[dict[str, Any]] = []
        for item in problems:
            if item["code"] in ADVISORY_CODES:
                degradations.append(item)
            else:
                kept.append(item)
        problems = kept

    complete = _is_complete(manifest) and not problems
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "dataset_id": manifest.dataset_id if isinstance(manifest.dataset_id, str) else None,
        "manifest_schema": manifest.schema if isinstance(manifest.schema, str) else None,
        "status": manifest.status if isinstance(manifest.status, str) else None,
        "mode": "strict" if strict else "degraded",
        "valid": not problems,
        "complete": complete,
        "problems": problems,
        "degradations": degradations,
        "checks_performed": checks,
        "not_checked": not_checked,
        "statistics": _statistics(manifest, groups),
        "manifest": dict(manifest.source) if isinstance(manifest.source, Mapping) else None,
        "split_registry": split_binding,
        # Declared, and separately whether the declaration lets an O2 model read
        # this corpus.  ``valid`` says the manifest is internally true; it does
        # *not* say the pixels are O2.
        "obs_contract": (
            manifest.raw.get("obs_contract")
            if isinstance(manifest.raw.get("obs_contract"), str) else None),
        "o2_compatible": o2_compatible(manifest),
    }
    report["exit_code"] = exit_code_for(report)
    return report


def exit_code_for(report: Mapping[str, Any]) -> int:
    """CLI exit code for a manifest report (``contracts.EXIT_*``).

    ``0`` — valid **and** complete; ``4`` — valid but explicitly not complete
    (the only way a partial dataset may ever be "ok" to read); ``2`` —
    invalid.  A report is never both: ``complete`` already implies ``valid``.
    """

    if report.get("valid") is not True:
        return EXIT_INVALID
    if report.get("complete") is not True:
        return EXIT_INCOMPLETE
    return EXIT_OK


# ---------------------------------------------------------------------------
# 5b. The O2 observation-contract gate: declare or refuse, never convert
# ---------------------------------------------------------------------------


def _not_o2_reason(obs_contract: Any) -> str:
    """The concrete, per-contract reason a manifest is not O2 training data."""

    if isinstance(obs_contract, str) and obs_contract in LEGACY_OBS_CONTRACTS:
        return LEGACY_OBS_CONTRACTS[obs_contract]
    return (
        "it does not record the O2 channels: global_rgb ((3,240,320)) is absent, "
        "play_gray ((4,224,192)) and hud_gray ((1,448,160)) have no declared "
        "geometry this module could trust"
    )


def assert_o2_compatible(manifest: "Manifest | Mapping[str, Any]") -> "Manifest":
    """Refuse to hand ``manifest`` to an O2 model unless it *is* an O2 manifest.

    This is a declaration check, **not** an adapter: it never resamples, pads,
    concatenates or synthesises a channel.  A corpus recorded under another
    contract (all of this repository's corpora are ``obs-v420-2f``) is refused by
    name, because ``observation.o2_digest()`` can only compare the contract
    *version*/parameter block — it cannot tell that an array's shape was produced
    by downsampling.

    Parameters
    ----------
    manifest:
        A :class:`Manifest` from :func:`load_manifest`, or the raw mapping.

    Returns
    -------
    Manifest
        The same manifest, unchanged — returned so a call site can read
        ``with_o2(assert_o2_compatible(m))``.

    Raises
    ------
    ManifestError
        If ``obs_contract`` is missing, is not :data:`OBS_CONTRACT_O2`, or names
        O2 while declaring channel geometry that disagrees with
        ``observation.O2_SHAPES``.  The message always names the actual contract
        and at least one concrete shape difference.
    """

    if isinstance(manifest, Mapping):
        manifest = Manifest.from_mapping(manifest)
    if not isinstance(manifest, Manifest):
        raise ManifestError(
            f"assert_o2_compatible needs a Manifest or a mapping, got "
            f"{type(manifest).__name__}")

    if "obs_contract" not in manifest.raw:
        raise ManifestError(
            f"manifest {manifest.dataset}: no obs_contract field, so it cannot be "
            f"asserted O2-compatible; the required value is "
            f"{OBS_CONTRACT_O2!r} and a manifest that does not name its "
            f"observation contract could have been recorded under any encoder")
    obs_contract = manifest.raw.get("obs_contract")
    if obs_contract != OBS_CONTRACT_O2:
        raise ManifestError(
            f"manifest {manifest.dataset}: obs_contract is {obs_contract!r}, but "
            f"the O2 model requires {OBS_CONTRACT_O2!r}; "
            f"{_not_o2_reason(obs_contract)}. There is no upgrade path in this "
            f"module (no resampling, no zero-fill, no channel synthesis), so the "
            f"corpus must be re-captured under {OBS_CONTRACT_O2!r}")

    shape_problems = _check_obs_shapes(manifest, OBS_CONTRACT_O2)
    if shape_problems:
        first = shape_problems[0]
        raise ManifestError(
            f"{first['message']} (problem code {first['code']}); a manifest that "
            f"says {OBS_CONTRACT_O2!r} while describing other geometry is exactly "
            f"the 'right shape, wrong pixels' case this guard exists for")
    return manifest


def o2_compatible(manifest: "Manifest | Mapping[str, Any]") -> bool:
    """``True`` only when :func:`assert_o2_compatible` would not raise.

    The non-raising form for callers that route corpora instead of asserting.
    It never returns ``True`` for a legacy contract and never raises: an
    unreadable or malformed input is simply not O2-compatible.
    """

    try:
        assert_o2_compatible(manifest)
    except ManifestError:
        return False
    return True


# ---------------------------------------------------------------------------
# 6. Identity / structure checks
# ---------------------------------------------------------------------------


def _check_identity(manifest: Manifest) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    raw = manifest.raw

    if manifest.schema != MANIFEST_SCHEMA:
        problems.append(_problem(
            "P_SCHEMA",
            f"schema must be {MANIFEST_SCHEMA!r}, got {manifest.schema!r}; an "
            f"unknown schema is not 'close enough' — the field semantics differ",
            field="schema", expected=MANIFEST_SCHEMA,
            actual=manifest.schema if isinstance(manifest.schema, str) else repr(manifest.schema)))

    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in raw:
            problems.append(_problem(
                "P_MISSING_FIELD",
                f"manifest is missing the required field {field!r} "
                f"(DEVELOPMENT_GUIDE §6.1)",
                field=field, where="manifest"))

    if "dataset_id" in raw:
        dataset_id = raw.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            problems.append(_problem(
                "P_FIELD_TYPE",
                f"dataset_id must be a non-empty string, got {dataset_id!r}",
                field="dataset_id"))
    if "parent_ids" in raw:
        if _as_string_list(raw.get("parent_ids")) is None:
            problems.append(_problem(
                "P_FIELD_TYPE",
                "parent_ids must be a list of non-empty strings (an empty list "
                f"for a root dataset), got {raw.get('parent_ids')!r}",
                field="parent_ids"))
    if "code_snapshot" in raw:
        snapshot = raw.get("code_snapshot")
        if not isinstance(snapshot, Mapping) or not snapshot:
            problems.append(_problem(
                "P_FIELD_TYPE",
                "code_snapshot must be a non-empty mapping (commit/dirty/ref), "
                f"got {snapshot!r}",
                field="code_snapshot"))
    for field in IDENTITY_FIELDS:
        if field not in raw:
            continue
        _sha, _scope, identity_problems = _identity_record(raw, field, where=field)
        problems.extend(identity_problems)

    if "status" in raw:
        status = raw.get("status")
        if not isinstance(status, str) or status not in KNOWN_MANIFEST_STATUSES:
            problems.append(_problem(
                "P_STATUS_UNKNOWN",
                f"status {status!r} is not one of {list(KNOWN_MANIFEST_STATUSES)}",
                field="status"))
    # The declared observation contract is an identity claim like the digests
    # above: it says *what was recorded*, so it is checked here.
    problems.extend(_check_obs_contract(manifest))
    return problems


# ---------------------------------------------------------------------------
# 6b. The observation contract: declaration only, never conversion
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _o2_shapes() -> dict[str, tuple[int, ...]]:
    """``observation.O2_SHAPES``, imported lazily so this module stays stdlib.

    ``manifest`` is pure stdlib by design; ``observation`` pulls in numpy.  The
    table is only needed when a manifest actually *claims* channel geometry, so
    the import happens on first such claim and never on a plain manifest.
    """

    from ..observation import O2_SHAPES  # noqa: PLC0415 - deliberately lazy

    return {name: tuple(shape) for name, shape in O2_SHAPES.items()}


def _check_obs_contract(manifest: Manifest) -> list[dict[str, Any]]:
    """``obs_contract`` must be present, known, and (for O2) geometrically true."""

    raw = manifest.raw
    if "obs_contract" not in raw:
        return []  # P_MISSING_FIELD already reported from REQUIRED_MANIFEST_FIELDS
    value = raw.get("obs_contract")
    if not isinstance(value, str) or not value.strip():
        return [_problem(
            "P_FIELD_TYPE",
            f"obs_contract must be a non-empty string naming the observation "
            f"contract the pixels were recorded under, got {value!r}",
            field="obs_contract")]
    if value not in KNOWN_OBS_CONTRACTS:
        return [_problem(
            "P_OBS_CONTRACT_UNKNOWN",
            f"{manifest.dataset}: obs_contract {value!r} is not a registered "
            f"observation contract (known: {list(KNOWN_OBS_CONTRACTS)}). A corpus "
            f"cannot be relabelled into the contract it was not captured under; "
            f"the name is the only record of which pixels exist",
            field="obs_contract", actual=value,
            known=list(KNOWN_OBS_CONTRACTS))]
    return _check_obs_shapes(manifest, value)


def _check_obs_shapes(manifest: Manifest, obs_contract: str) -> list[dict[str, Any]]:
    """When geometry is claimed, it is compared; a legacy contract claims nothing
    about O2 geometry."""

    raw = manifest.raw
    if "obs_shapes" not in raw:
        return []  # no claim to verify; validate_manifest reports the omission
    declared = raw.get("obs_shapes")
    if not isinstance(declared, Mapping):
        return [_problem(
            "P_FIELD_TYPE",
            f"obs_shapes must be a {{channel: [dim, ...]}} mapping, got "
            f"{type(declared).__name__}",
            field="obs_shapes")]
    shapes: dict[str, tuple[int, ...]] = {}
    for name, shape in declared.items():
        if not isinstance(name, str) or not name.strip():
            return [_problem(
                "P_FIELD_TYPE",
                f"obs_shapes keys must be non-empty strings, got {name!r}",
                field="obs_shapes")]
        if (not isinstance(shape, (list, tuple))
                or not all(_is_int(dim) for dim in shape)):
            return [_problem(
                "P_FIELD_TYPE",
                f"obs_shapes[{name!r}] must be a list of ints, got {shape!r}",
                field=f"obs_shapes.{name}")]
        shapes[name] = tuple(shape)

    if obs_contract != OBS_CONTRACT_O2:
        return []  # a legacy contract's geometry is recorded, not compared to O2

    try:
        expected = _o2_shapes()
    except Exception as exc:  # pragma: no cover - observation is in this repo
        return [_problem(
            "P_OBS_SHAPE_CONTRACT_MISMATCH",
            f"{manifest.dataset}: obs_contract is {OBS_CONTRACT_O2!r} but "
            f"observation.O2_SHAPES cannot be loaded ({exc}), so the declared "
            f"obs_shapes are unverifiable; fail-closed",
            field="obs_shapes")]

    differences = _o2_shape_differences(shapes, expected)
    if not differences:
        return []
    return [_problem(
        "P_OBS_SHAPE_CONTRACT_MISMATCH",
        f"{manifest.dataset}: obs_contract is {OBS_CONTRACT_O2!r} but the "
        f"declared obs_shapes do not match observation.O2_SHAPES: "
        + "; ".join(differences),
        field="obs_shapes", differences=list(differences))]


def _o2_shape_differences(declared: Mapping[str, tuple[int, ...]],
                          expected: Mapping[str, tuple[int, ...]]) -> list[str]:
    """Every way ``declared`` fails to describe an O2 corpus, as human text."""

    out: list[str] = []
    for name in O2_OBSERVATION_CHANNELS:
        want = expected.get(name)
        if want is None:  # pragma: no cover - the contract always has these
            out.append(f"{name} is not in observation.O2_SHAPES at all")
            continue
        if name not in declared:
            out.append(f"{name} is not declared, but O2 requires {want}")
            continue
        if declared[name] != want:
            out.append(f"{name} is declared {declared[name]} but O2 requires {want}")
    for name, shape in declared.items():
        if name in O2_OBSERVATION_CHANNELS:
            continue
        if name not in expected:
            out.append(f"{name} is declared {shape} but is not an O2 field")
        elif shape != expected[name]:
            out.append(f"{name} is declared {shape} but O2 requires {expected[name]}")
    return out


def _check_shards(manifest: Manifest, resolve: Any) -> list[dict[str, Any]]:
    raw = manifest.raw
    if "shards" not in raw:
        return []  # P_MISSING_FIELD already reported by the identity check
    value = raw.get("shards")
    if not isinstance(value, (list, tuple)):
        return [_problem(
            "P_FIELD_TYPE",
            f"shards must be a list of shard objects, got {type(value).__name__}",
            field="shards")]
    if not value:
        return [_problem(
            "P_MISSING_FIELD",
            "shards is empty: a dataset manifest must list at least one shard",
            field="shards", where="manifest")]

    problems: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for shard in manifest.shards:
        problems.extend(_check_shard(shard, resolve))
        if isinstance(shard.path, str) and shard.path.strip():
            key = _posix(shard.path)
            if key in seen:
                problems.append(_problem(
                    "P_SHARD_DUPLICATE",
                    f"shard path {key!r} is listed twice (shards[{seen[key]}] and "
                    f"shards[{shard.index}])",
                    path=key, first=seen[key], second=shard.index))
            else:
                seen[key] = shard.index
    return problems


def _check_shard(shard: Shard, resolve: Any) -> list[dict[str, Any]]:
    """One shard: required fields, domains, then the bytes on disk."""

    where = f"shards[{shard.index}]"
    if not isinstance(shard.raw, Mapping):
        return [_problem(
            "P_FIELD_TYPE", f"{where} must be a shard object", field=where)]

    problems: list[dict[str, Any]] = []
    for field in REQUIRED_SHARD_FIELDS:
        if field not in shard.raw:
            problems.append(_problem(
                "P_MISSING_FIELD",
                f"{where} is missing the required field {field!r} "
                f"(DEVELOPMENT_GUIDE §6.1)",
                field=f"{where}.{field}", where=where))

    # --- path --------------------------------------------------------------
    disk_path: Path | None = None
    if "path" in shard.raw:
        text = _posix(shard.path) if isinstance(shard.path, str) else ""
        if not text:
            problems.append(_problem(
                "P_FIELD_TYPE",
                f"{where}.path must be a non-empty relative path, got "
                f"{shard.path!r}",
                field=f"{where}.path"))
        elif (Path(text).is_absolute()
              or (len(text) > 1 and text[1] == ":")
              or text.startswith("//")):
            problems.append(_problem(
                "P_SHARD_PATH_INVALID",
                f"{where}.path {text!r} is absolute; a shard path must be "
                f"relative to its root so the manifest is portable",
                path=text, field=f"{where}.path"))
        elif ".." in Path(text).parts:
            problems.append(_problem(
                "P_SHARD_PATH_INVALID",
                f"{where}.path {text!r} escapes its root with '..'",
                path=text, field=f"{where}.path"))
        else:
            disk_path = resolve(shard)

    # --- size and digest ---------------------------------------------------
    if "size_bytes" in shard.raw and not _is_int(shard.size_bytes):
        problems.append(_problem(
            "P_FIELD_TYPE",
            f"{where}.size_bytes must be an int, got {shard.size_bytes!r}",
            field=f"{where}.size_bytes"))
    sha: str | None = None
    scope: str | None = None
    if "sha256" in shard.raw:
        sha, scope, digest_problems = _identity_record(shard.raw, "sha256", where=where)
        problems.extend(digest_problems)

    if disk_path is not None:
        problems.extend(_check_shard_bytes(shard, disk_path, sha, scope))

    # --- episodes / frame range / counts / duplicates / split --------------
    if "episodes" in shard.raw:
        episodes = shard.episodes
        if not isinstance(episodes, (list, tuple)) or not episodes:
            problems.append(_problem(
                "P_FIELD_TYPE",
                f"{where}.episodes must be a non-empty list, got {episodes!r}",
                field=f"{where}.episodes"))
        else:
            for index, episode in enumerate(episodes):
                if isinstance(episode, str):
                    if not episode.strip():
                        problems.append(_problem(
                            "P_FIELD_TYPE",
                            f"{where}.episodes[{index}] must not be blank",
                            field=f"{where}.episodes[{index}]"))
                elif not _is_int(episode):
                    problems.append(_problem(
                        "P_FIELD_TYPE",
                        f"{where}.episodes[{index}] must be a string or an int, "
                        f"got {episode!r}",
                        field=f"{where}.episodes[{index}]"))

    if "frame_range" in shard.raw:
        frame_range = shard.frame_range
        if (not isinstance(frame_range, (list, tuple)) or len(frame_range) != 2
                or not all(_is_int(part) for part in frame_range)):
            problems.append(_problem(
                "P_FRAME_RANGE_INVALID",
                f"{where}.frame_range must be [start, end) with two integers, got "
                f"{frame_range!r}",
                field=f"{where}.frame_range"))
        elif frame_range[0] < 0 or frame_range[1] < frame_range[0]:
            problems.append(_problem(
                "P_FRAME_RANGE_INVALID",
                f"{where}.frame_range {list(frame_range)} is not a valid "
                f"[start, end) interval",
                field=f"{where}.frame_range"))

    decisions = shard.decisions
    labels_valid = shard.labels_valid
    for field, value in (("decisions", decisions), ("labels_valid", labels_valid)):
        if field in shard.raw and (not _is_int(value) or value < 0):
            problems.append(_problem(
                "P_FIELD_TYPE",
                f"{where}.{field} must be a non-negative int, got {value!r}",
                field=f"{where}.{field}"))
            if field == "decisions":
                decisions = None
            else:
                labels_valid = None
    if _is_int(decisions) and _is_int(labels_valid) and labels_valid > decisions:
        problems.append(_problem(
            "P_LABELS_EXCEED_DECISIONS",
            f"{where}.labels_valid {labels_valid} > decisions {decisions}; a shard "
            f"cannot label more decisions than it contains",
            field=f"{where}.labels_valid", decisions=decisions,
            labels_valid=labels_valid))

    if "duplicates" in shard.raw:
        duplicates = shard.duplicates
        if not (isinstance(duplicates, Mapping)
                or (_is_int(duplicates) and duplicates >= 0)):
            problems.append(_problem(
                "P_DUPLICATES_INVALID",
                f"{where}.duplicates must be a mapping or a non-negative int, got "
                f"{duplicates!r}",
                field=f"{where}.duplicates"))

    if "split" in shard.raw:
        split = shard.split
        if not isinstance(split, str):
            problems.append(_problem(
                "P_FIELD_TYPE",
                f"{where}.split must be a string, got {split!r}",
                field=f"{where}.split"))
        elif split not in lineage.ALL_SPLIT_NAMES:
            problems.append(_problem(
                "P_SHARD_SPLIT_UNKNOWN",
                f"{where}.split {split!r} is not one of "
                f"{list(lineage.ALL_SPLIT_NAMES)}",
                field=f"{where}.split", split=split))

    if "status" in shard.raw:
        status = shard.status
        if not isinstance(status, str) or status not in KNOWN_SHARD_STATUSES:
            problems.append(_problem(
                "P_SHARD_STATUS_UNKNOWN",
                f"{where}.status {status!r} is not one of "
                f"{list(KNOWN_SHARD_STATUSES)}",
                field=f"{where}.status", status=status))
    if "immutable" in shard.raw and not isinstance(shard.immutable, bool):
        problems.append(_problem(
            "P_FIELD_TYPE",
            f"{where}.immutable must be true/false, got {shard.immutable!r}",
            field=f"{where}.immutable"))
    return problems


def _check_shard_bytes(shard: Shard, disk_path: Path, sha: str | None,
                       scope: str | None) -> list[dict[str, Any]]:
    """The core check: the bytes on disk against ``size_bytes`` and ``sha256``."""

    path_text = _posix(str(disk_path))
    if not disk_path.exists():
        return [_problem(
            "P_SHARD_MISSING",
            f"shard {shard.identifier!r} does not exist at {path_text}",
            path=shard.identifier, resolved=path_text)]
    if not disk_path.is_file():
        return [_problem(
            "P_SHARD_NOT_FILE",
            f"shard {shard.identifier!r} is not a regular file ({path_text})",
            path=shard.identifier, resolved=path_text)]
    try:
        raw_bytes = disk_path.read_bytes()
    except OSError as exc:
        return [_problem(
            "P_SHARD_UNREADABLE",
            f"shard {shard.identifier!r} cannot be read ({path_text}): {exc}",
            path=shard.identifier, resolved=path_text)]

    problems: list[dict[str, Any]] = []
    actual_size = len(raw_bytes)
    if _is_int(shard.size_bytes) and shard.size_bytes != actual_size:
        problems.append(_problem(
            "P_SHARD_SIZE_MISMATCH",
            f"shard {shard.identifier!r} size_bytes is {shard.size_bytes} but the "
            f"file at {path_text} has {actual_size} bytes",
            path=shard.identifier, expected=shard.size_bytes, actual=actual_size,
            resolved=path_text))
    if sha is not None and scope in DIGEST_SCOPES:
        actual = digest_bytes(raw_bytes, scope)
        if actual != sha:
            problems.append(_problem(
                "P_SHARD_SHA_MISMATCH",
                f"shard {shard.identifier!r} sha256 mismatch ({scope}): manifest "
                f"says {sha}, the file at {path_text} hashes to {actual}",
                path=shard.identifier, expected=sha, actual=actual,
                digest_scope=scope, resolved=path_text, size_bytes=actual_size))
    return problems


def _check_counts(manifest: Manifest) -> list[dict[str, Any]]:
    """``counts`` must equal the sum over the listed shards."""

    if "counts" not in manifest.raw:
        return []  # P_MISSING_FIELD already reported
    counts = manifest.counts
    if not isinstance(counts, Mapping):
        return [_problem(
            "P_FIELD_TYPE",
            f"counts must be a mapping, got {type(counts).__name__}",
            field="counts")]

    expected = {
        "shards": len(manifest.shards),
        "decisions": sum(s.decisions for s in manifest.shards if _is_int(s.decisions)),
        "labels_valid": sum(
            s.labels_valid for s in manifest.shards if _is_int(s.labels_valid)),
        "episodes": sum(
            len(s.episodes) for s in manifest.shards
            if isinstance(s.episodes, (list, tuple))),
    }
    problems: list[dict[str, Any]] = []
    for key, value in expected.items():
        if key not in counts:
            problems.append(_problem(
                "P_MISSING_FIELD",
                f"counts.{key} is missing; a manifest must record counts.shards, "
                f"counts.decisions, counts.episodes and counts.labels_valid",
                field=f"counts.{key}"))
            continue
        recorded = counts.get(key)
        if not _is_int(recorded) or recorded < 0:
            problems.append(_problem(
                "P_FIELD_TYPE",
                f"counts.{key} must be a non-negative int, got {recorded!r}",
                field=f"counts.{key}"))
            continue
        if recorded != value:
            problems.append(_problem(
                "P_COUNTS_MISMATCH",
                f"counts.{key} is {recorded} but the shards sum to {value}",
                field=f"counts.{key}", expected=recorded, actual=value))
    return problems


def _check_shard_status(manifest: Manifest) -> list[dict[str, Any]]:
    """``status="complete"`` may not list a shard that is not readable."""

    problems: list[dict[str, Any]] = []
    if manifest.status != MANIFEST_STATUS_COMPLETE:
        return problems
    for shard in manifest.shards:
        if shard.is_readable:
            continue
        problems.append(_problem(
            "P_SHARD_NOT_COMPLETED",
            f"shard {shard.identifier!r} has status {shard.status!r} and immutable "
            f"{shard.immutable!r}; a 'complete' manifest may only list completed, "
            f"immutable shards",
            path=shard.identifier,
            status=shard.status if isinstance(shard.status, str) else None,
            immutable=shard.immutable if isinstance(shard.immutable, bool) else None))
    return problems


def _check_status(manifest: Manifest) -> list[dict[str, Any]]:
    """Completeness: a partial manifest is never a complete dataset."""

    problems: list[dict[str, Any]] = []
    if manifest.status == MANIFEST_STATUS_PARTIAL:
        problems.append(_problem(
            "P_PARTIAL_INCOMPLETE",
            f"{manifest.dataset} has status 'partial': it is not a complete "
            f"dataset and must not be validated as one. Read only completed, "
            f"immutable shards via partial_view(manifest) and cite it as a "
            f"partial view, never as coverage",
            status=manifest.status))
    problems.extend(_check_shard_status(manifest))
    return problems


def _is_complete(manifest: Manifest) -> bool:
    """Whether the manifest itself claims a finished, fully readable dataset."""

    return (manifest.status == MANIFEST_STATUS_COMPLETE
            and bool(manifest.shards)
            and all(shard.is_readable for shard in manifest.shards))


# ---------------------------------------------------------------------------
# 7. Ancestry (lineage.group_key semantics, never a second definition)
# ---------------------------------------------------------------------------


def _ancestor_parts(item: Any) -> tuple[str | None, str | None, str | None, dict[str, Any] | None]:
    """``(ancestor_id, task_family, episode, problem)`` for one ancestor record.

    Two recorded forms are understood, and both are reduced to
    :func:`lineage.group_key`'s ``(task_family, episode)`` pair — this module
    never invents its own notion of "the same ancestor":

    * the canonical id string of :func:`lineage.ancestor_id`,
      ``kind|source|episode|task_family``;
    * a mapping with ``task_family`` + ``episode`` (plus optional
      ``source_kind``/``source_id``, which are fed back through
      :func:`lineage.ancestor_id` so the id is derived, not typed twice).
    """

    if isinstance(item, str):
        text = item.strip()
        parts = text.split("|")
        if len(parts) != 4 or not all(part.strip() for part in parts):
            return None, None, None, _problem(
                "P_ANCESTOR_INVALID",
                f"ancestor {item!r} is not a lineage.ancestor_id "
                f"'kind|source|episode|task_family'",
                ancestor=item)
        return text, parts[3].strip(), parts[2].strip(), None
    if isinstance(item, Mapping):
        family = item.get("task_family")
        episode = item.get("episode")
        if not isinstance(family, str) or not family.strip():
            return None, None, None, _problem(
                "P_ANCESTOR_INVALID",
                f"ancestor record {dict(item)!r} has no task_family",
                ancestor=item.get("ancestor_id"))
        if not (isinstance(episode, str) or _is_int(episode)):
            return None, None, None, _problem(
                "P_ANCESTOR_INVALID",
                f"ancestor record {dict(item)!r} has no episode id",
                ancestor=item.get("ancestor_id"))
        derived = item.get("ancestor_id")
        if derived is None and item.get("source_kind") and item.get("source_id"):
            try:
                derived = ancestor_id(item["source_kind"], item["source_id"],
                                      episode, family)
            except LineageError as exc:
                return None, None, None, _problem(
                    "P_ANCESTOR_INVALID", str(exc), ancestor=None)
        if derived is not None and not isinstance(derived, str):
            return None, None, None, _problem(
                "P_ANCESTOR_INVALID",
                f"ancestor_id must be a string, got {derived!r}",
                ancestor=None)
        aid = derived or f"{family}|{episode}"
        return aid, family.strip(), str(episode).strip(), None
    return None, None, None, _problem(
        "P_ANCESTOR_INVALID",
        f"ancestor entry {item!r} is neither an id string nor a mapping",
        ancestor=None)


def _ancestor_index(manifest: Manifest) -> tuple[dict[str, dict[str, set[str]]],
                                                 list[dict[str, Any]]]:
    """``{group_key: {split: {ancestor_id}}}`` plus the per-shard problems."""

    problems: list[dict[str, Any]] = []
    groups: dict[str, dict[str, set[str]]] = {}
    for shard in manifest.shards:
        where = f"shards[{shard.index}].ancestors"
        if not isinstance(shard.raw, Mapping) or "ancestors" not in shard.raw:
            continue  # missing-field / non-object already reported
        value = shard.ancestors
        if not isinstance(value, (list, tuple)):
            problems.append(_problem(
                "P_FIELD_TYPE",
                f"{where} must be a list of ancestor ids or records, got "
                f"{value!r}",
                field=where))
            continue
        if not value:
            problems.append(_problem(
                "P_ANCESTOR_INVALID",
                f"{where} is empty: a shard with no recorded ancestor cannot be "
                f"checked for a train/dev crossing",
                field=where))
            continue
        if isinstance(shard.split, str) and shard.split in lineage.ALL_SPLIT_NAMES:
            split_name = shard.split
        else:
            split_name = "<unknown split>"
        for item in value:
            aid, family, episode, problem = _ancestor_parts(item)
            if problem is not None:
                problem.setdefault("field", where)
                problems.append(problem)
                continue
            try:
                key = group_key(family, episode)
            except LineageError as exc:
                problems.append(_problem(
                    "P_ANCESTOR_INVALID", str(exc), field=where, ancestor=aid))
                continue
            groups.setdefault(key, {}).setdefault(split_name, set()).add(aid)
    return groups, problems


def _crossing_problems(groups: Mapping[str, Mapping[str, set[str]]]) -> list[dict[str, Any]]:
    """One ancestor group in two splits is a lineage violation, not a new task."""

    problems: list[dict[str, Any]] = []
    for key, by_split in sorted(groups.items()):
        # ``quarantine`` is excluded on purpose: it exists to *name* barred
        # records, which may legitimately duplicate a train episode.  Nothing may
        # be fitted from it, and that is the registry's job to enforce.
        splits = sorted(split for split in by_split if split != "quarantine")
        if len(splits) < 2:
            continue
        named = sorted({aid for split in splits for aid in by_split[split]})
        problems.append(_problem(
            "P_ANCESTOR_CROSSING",
            f"ancestor group {key!r} appears in splits {splits} "
            f"(ancestors {named}); one task ancestor belongs to exactly one split, "
            f"whatever the seeds say (lineage.group_key collapses them)",
            ancestor_group=key, ancestor_ids=named, splits=splits))
    return problems


# ---------------------------------------------------------------------------
# 8. Digest vocabulary, split binding, roots, statistics
# ---------------------------------------------------------------------------


def _check_digest_scopes(manifest: Manifest) -> list[dict[str, Any]]:
    """No identity anywhere in the document may carry a bare ``sha256``.

    The six manifest-level identity fields and the per-shard ``sha256`` are
    reported by their own checks; this walk catches everything else (a nested
    ``code_snapshot`` or an auxiliary record), so a new field cannot smuggle an
    unreproducible digest in.
    """

    problems: list[dict[str, Any]] = []
    for path, node in _iter_sha_nodes(manifest.raw):
        if "digest_scope" in node:
            continue
        leaf = path.split(".")[-1]
        if leaf in IDENTITY_FIELDS:
            continue
        if path.startswith("shards[") and path.endswith("]"):
            continue
        problems.append(_problem(
            "P_DIGEST_SCOPE_MISSING",
            f"identity {path!r} carries sha256 without digest_scope; every digest "
            f"must be labelled {DIGEST_SCOPES[0]!r} or {DIGEST_SCOPES[1]!r}",
            field=path))
    return problems


def _check_split_registry(
    manifest: Manifest, split_config: Any, split_registry_sha: Any
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Bind ``split_registry_sha`` to the frozen split config's own sha256.

    The split config is hashed by :func:`lineage.validate_splits` (the W09
    owner of that number); this function never re-derives it.
    """

    binding: dict[str, Any] = {"source": None, "sha256": None, "split_config_valid": None}
    # TODO(W03B): a split config that fails `lineage.validate_splits` still yields
    # a sha256, and its failure is recorded (`split_config_valid`) rather than
    # made fatal here.  A future "manifests only bind frozen registries" tier
    # should reject `split_config_valid is False` outright instead of recording
    # it; that tier needs the W09 verdict vocabulary, so it is not guessed here.
    if split_registry_sha is not None:
        want = _expected_sha(split_registry_sha, "split_registry_sha")
        binding.update({"source": "expected_value", "sha256": want})
        problems: list[dict[str, Any]] = []
    else:
        try:
            split_report = lineage.validate_splits(split_config)
        except Exception as exc:  # noqa: BLE001 - fail closed, never raise here
            return binding, [_problem(
                "P_SPLIT_CONFIG_UNREADABLE",
                f"the supplied split config cannot be validated: {exc}")]
        want = split_report.get("split_registry_sha")
        binding.update({
            "source": "lineage.validate_splits",
            "sha256": want,
            "split_config_valid": bool(split_report.get("passed")),
            "split_config_problems": [p.get("code") for p in
                                     split_report.get("problems", [])],
        })
        if not _is_hex64(want):
            return binding, [_problem(
                "P_SPLIT_CONFIG_UNREADABLE",
                "lineage.validate_splits returned no split_registry_sha for the "
                f"supplied config (problems: {binding['split_config_problems']})")]
        problems = []

    if "split_registry_sha" not in manifest.raw:
        return binding, problems  # P_MISSING_FIELD already reported
    got, scope, _own = _identity_record(
        manifest.raw, "split_registry_sha", where="split_registry_sha")
    if got is not None and got != want:
        problems.append(_problem(
            "P_SPLIT_REGISTRY_SHA_MISMATCH",
            f"{manifest.dataset}: split_registry_sha is {got} but the supplied "
            f"split config hashes to {want}",
            field="split_registry_sha", expected=want, actual=got,
            digest_scope=scope))
    return binding, problems


def _root_paths(roots: Mapping[str, str]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Check that every declared root is a directory; ``{name: Path}`` output."""

    if "<invalid>" in roots:
        return {}, [_problem(
            "P_ROOTS_INVALID",
            f"roots must be a mapping {{name: directory}} or a sequence of "
            f"directories, got {roots['<invalid>']}",
            field="roots")]
    problems: list[dict[str, Any]] = []
    out: dict[str, Path] = {}
    for name, text in roots.items():
        path = Path(text)
        if not name or not str(name).strip():
            problems.append(_problem(
                "P_ROOTS_INVALID", f"root {text!r} has no name", field="roots"))
            continue
        if not path.is_dir():
            problems.append(_problem(
                "P_ROOTS_INVALID",
                f"root {name!r} is not a directory: {text}",
                root=name, field="roots"))
            continue
        out[name] = path
    return out, problems


def _resolver(base_dir: str | Path | None, roots: Any) -> Any:
    """Build ``shard -> Path`` with the shard's optional ``root`` name honoured."""

    root_map: dict[str, str] = dict(roots) if isinstance(roots, Mapping) else {}
    base = Path(base_dir) if base_dir is not None else None

    def resolve(shard: Shard) -> Path:
        text = _posix(str(shard.path))
        name = shard.raw.get("root") if isinstance(shard.raw, Mapping) else None
        if isinstance(name, str) and name in root_map:
            return Path(root_map[name]) / text
        if base is not None:
            return base / text
        return Path(text)

    return resolve


def _check_unregistered(manifest: Manifest, root_paths: Mapping[str, Path],
                        resolve: Any) -> list[dict[str, Any]]:
    """Every file under an explicit root must be listed; nothing hides in a root."""

    registered: set[str] = set()
    for shard in manifest.shards:
        if isinstance(shard.path, str) and shard.path.strip():
            registered.add(_path_key(resolve(shard)))
    if isinstance(manifest.source, Mapping):
        source_path = manifest.source.get("path")
        if isinstance(source_path, str) and source_path.strip():
            registered.add(_path_key(Path(source_path)))

    problems: list[dict[str, Any]] = []
    for name, root in sorted(root_paths.items()):
        if not root.is_dir():
            continue  # already P_ROOTS_INVALID
        for file_path in _walk_files(root):
            if _path_key(file_path) in registered:
                continue
            try:
                rel = _posix(str(file_path.relative_to(root)))
            except ValueError:  # pragma: no cover - walk yields children of root
                rel = _posix(str(file_path))
            problems.append(_problem(
                "P_SHARD_UNREGISTERED",
                f"file {rel!r} under root {name!r} is not listed by the manifest; "
                f"an unregistered shard is data outside the recorded dataset",
                path=rel, root=name, resolved=_posix(str(file_path))))
    return problems


def _walk_files(root: Path) -> Iterator[Path]:
    """Regular files under ``root``, skipping VCS/cache directories and dotfiles."""

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in {".git", ".pytest_cache", "__pycache__"}
            and not name.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or name in _SCAN_IGNORED_NAMES:
                continue
            yield Path(dirpath) / name


def _path_key(path: Path) -> str:
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - a delete race between two calls
        resolved = path
    return os.path.normcase(str(resolved))


def _statistics(manifest: Manifest, groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Counts a reader may quote, computed from the listed shards only."""

    by_split: dict[str, int] = {}
    for shard in manifest.shards:
        if isinstance(shard.split, str):
            by_split[shard.split] = by_split.get(shard.split, 0) + 1
    return {
        "shards_listed": len(manifest.shards),
        "shards_readable": sum(1 for shard in manifest.shards if shard.is_readable),
        "shards_by_split": dict(sorted(by_split.items())),
        "bytes_listed": sum(shard.size_bytes for shard in manifest.shards
                            if _is_int(shard.size_bytes)),
        "decisions": sum(shard.decisions for shard in manifest.shards
                         if _is_int(shard.decisions)),
        "labels_valid": sum(shard.labels_valid for shard in manifest.shards
                            if _is_int(shard.labels_valid)),
        "episodes": sum(len(shard.episodes) for shard in manifest.shards
                        if isinstance(shard.episodes, (list, tuple))),
        "ancestor_ids": sum(len(shard.ancestors) for shard in manifest.shards
                            if isinstance(shard.ancestors, (list, tuple))),
        "ancestor_groups": len(groups),
    }


# ---------------------------------------------------------------------------
# 9. Partial view
# ---------------------------------------------------------------------------


def partial_view(
    manifest: Manifest | Mapping[str, Any],
    *,
    completed_only: bool = True,
    base_dir: str | Path | None = None,
    strict: bool = False,
) -> PartialView:
    """The explicitly readable subset of a (partial) manifest.

    ``completed_only=True`` (default) keeps a shard only when it is **both**
    ``status == "completed"`` **and** ``immutable is True``; everything else is
    listed in :attr:`PartialView.excluded` with the reason, so a caller can see
    what was left out instead of inferring it.  ``completed_only=False`` keeps
    every structurally addressable shard (for diagnosing a manifest that is still
    being written) — it makes no immutability claim at all.

    ``complete`` on the returned view is always ``False``: a view never states
    coverage.  The view's own ``counts`` are recomputed over the included shards
    so the sub-manifest is internally consistent, and ``report`` validates that
    sub-manifest (``strict`` defaults to ``False`` here — completeness is the one
    thing a partial view cannot claim).

    There is deliberately **no** ``roots`` parameter: "is every file under a data
    root registered?" is a property of the *whole* dataset, so it belongs to
    :func:`validate_manifest` on the source manifest.  Running that scan on a
    filtered view would report every excluded shard as "unregistered data", which
    is the opposite of what it means.
    """

    if isinstance(manifest, Mapping):
        manifest = Manifest.from_mapping(manifest)
    if not isinstance(manifest, Manifest):
        raise ManifestError(
            f"partial_view needs a Manifest or a mapping, got "
            f"{type(manifest).__name__}")

    included: list[Shard] = []
    excluded: list[dict[str, Any]] = []
    for shard in manifest.shards:
        reason = _exclusion_reason(shard, completed_only=completed_only)
        if reason is None:
            included.append(shard)
        else:
            excluded.append({
                "path": shard.identifier,
                "status": shard.status if isinstance(shard.status, str) else None,
                "immutable": shard.immutable if isinstance(shard.immutable, bool) else None,
                "reason": reason,
            })

    counts = _recompute_counts(included)
    raw = dict(manifest.raw)
    raw["shards"] = [shard.to_dict() for shard in included]
    raw["counts"] = dict(counts)
    # TODO(W03B): the view is not yet bound to the source manifest's identity
    # (``parent_manifest_sha``), so a view file saved on its own cannot be proven
    # to have come from this manifest.  That binding needs the `index` module.
    # ``roots`` is dropped on purpose: see the docstring — the unregistered-shard
    # scan belongs to the source manifest, not to a filtered view of it.
    view = dataclasses.replace(
        manifest, shards=tuple(included), counts=counts, raw=raw, roots=None)
    report = validate_manifest(view, base_dir=base_dir, strict=strict)
    return PartialView(
        manifest=view,
        source_status=str(manifest.status),
        excluded=tuple(excluded),
        report=report,
    )


def _exclusion_reason(shard: Shard, *, completed_only: bool) -> str | None:
    """Why a shard may not be read; ``None`` means it may be."""

    if not isinstance(shard.raw, Mapping):
        return "not_a_shard_object"
    if not (isinstance(shard.path, str) and shard.path.strip()):
        return "no_path"
    if not completed_only:
        return None
    if shard.status is None:
        return "status_not_recorded"
    if shard.status != SHARD_STATUS_COMPLETED:
        return f"status_{shard.status}"
    if shard.immutable is not True:
        return "mutable"
    return None


def _recompute_counts(shards: Sequence[Shard]) -> dict[str, Any]:
    """Counts that describe the included shards only (never the whole dataset)."""

    return {
        "shards": len(shards),
        "decisions": sum(s.decisions for s in shards if _is_int(s.decisions)),
        "labels_valid": sum(s.labels_valid for s in shards if _is_int(s.labels_valid)),
        "episodes": sum(len(s.episodes) for s in shards
                        if isinstance(s.episodes, (list, tuple))),
        "derived_from": "included shards only (partial view)",
    }
