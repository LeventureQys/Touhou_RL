"""v4.3.0 W09 — clean hold-out task lineage and the fail-closed split validator.

Why this module exists
----------------------
``Document/Update/Alpha/v4.3.0/README.md`` §9.2 freezes the task lineage **before**
the corpus grows: G410-6 (clean hold-out adaptation) is only interpretable if the
two hold-out task families were never touched by *any* pretraining component —
policy RL, perception pretraining, the reward/outcome detector, human replays,
teacher queries, hyper-parameter selection, model selection, behaviour-cloning
corpora and DAgger rounds.  V4.1.0 ``README.md`` §4.3/§WP10 says the same thing in
one line: "训练谱系包含教师数据、感知预训练、回报检测器、模型选择和适应数据，
不能只查最终策略的 seed 字段。"

This module is the machine-checkable half of that governance:

* a **task ancestor** is uniquely identified by ``(source_kind, source_id,
  episode, task_family)`` — see :func:`ancestor_id`.  The ``seed`` is an
  *attribute* of the ancestor, never part of its identity, because "same
  ``(task_family, episode)`` recorded under a fresh seed" is the same task, not a
  new one (``TRAINING_PLAN.md`` §4: "不可由换 seed 伪造未见任务");
* a **pretraining source** is a ``(kind, source_id)`` pair that records which
  ancestors it touched (``pretraining source`` register);
* :func:`validate_splits` is fail-closed: it *rejects* rather than warns, and
  returns ``passed``/``problems``/``statistics`` plus the stable
  ``split_registry_sha`` a dataset manifest binds (``DEVELOPMENT_GUIDE.md`` §6.1);
* every identity field is ``sha256`` **plus** ``digest_scope``; a bare digest
  without a scope is itself a validation failure.  Text artefacts are normalised
  ``CRLF -> LF`` first (``digest_scope="text/lf-normalized"``), binaries are
  hashed raw (``digest_scope="binary/raw"``).

Scope of the checks — and what this module cannot prove
------------------------------------------------------
The validator checks *records*, not game bytes.  It cannot prove that an evidence
file says what the registry claims, it cannot see the pixels of a sealed asset, and
it can only recognise "same trajectory, new seed" when a ``trajectory_sha256`` is
supplied.  Those limits are returned in the report's ``not_checked`` list instead
of being silently assumed to hold.

Standard library only: no numpy, no torch, no engine, no pixel I/O.  Nothing here
starts the game, trains, or opens any ``.tmp_*`` payload.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "SCHEMA",
    "REPORT_SCHEMA",
    "FAMILY_REGISTRY_SCHEMA",
    "SOURCE_REGISTRY_SCHEMA",
    "HOLDOUT_REGISTRY_SCHEMA",
    "RUNS_REGISTRY_SCHEMA",
    "SPEC_REVISION",
    "PRETRAINING_SOURCE_KINDS",
    "EVALUATION_SOURCE_KIND",
    "KNOWN_SOURCE_KINDS",
    "SPLIT_NAMES",
    "OPTIONAL_SPLIT_NAMES",
    "BANNED_SEED_MIN",
    "BANNED_SEED_MAX",
    "DIGEST_SCOPE_TEXT",
    "DIGEST_SCOPE_BINARY",
    "DIGEST_SCOPES",
    "TEXT_CANONICALIZATION",
    "normalise_line_endings",
    "text_lf_digest",
    "binary_raw_digest",
    "digest_bytes",
    "file_identity",
    "canonical_json",
    "stable_registry_digest",
    "ancestor_id",
    "group_key",
    "load_json_text",
    "validate_splits",
    "exit_code_for",
]

# ---------------------------------------------------------------------------
# 1. Schemas, enums and constants
# ---------------------------------------------------------------------------

#: Schema of the frozen split config (``configs/v430/splits.json``).
SCHEMA = "v430-splits/1"
#: Schema of the report returned by :func:`validate_splits`.
REPORT_SCHEMA = "v430-splits-validation/1"
FAMILY_REGISTRY_SCHEMA = "v430-task-families/1"
SOURCE_REGISTRY_SCHEMA = "v430-pretraining-sources/1"
HOLDOUT_REGISTRY_SCHEMA = "v430-holdout-families/1"
RUNS_REGISTRY_SCHEMA = "v430-runs/1"

#: Plan revision this validator belongs to.  A config bound to another revision is
#: rejected: the split semantics differ per revision.
SPEC_REVISION = "v430-plan-r2"

#: Every pretraining source category that **must** be registered.  Checking only
#: the final RL seed is explicitly forbidden (README §9.2).
PRETRAINING_SOURCE_KINDS: tuple[str, ...] = (
    "policy_rl",
    "perception_pretrain",
    "reward_detector",
    "human_replay",
    "teacher_query",
    "hyperparameter_selection",
    "model_selection",
    "bc_dataset",
    "dagger_round",
)

#: Non-training records (formal confirmation / adaptation evaluation) still need a
#: kind so that they cannot be smuggled into a "pretraining" claim.  This kind is
#: *not* a pretraining source and must not appear in the source register.
EVALUATION_SOURCE_KIND = "evaluation"

KNOWN_SOURCE_KINDS: tuple[str, ...] = PRETRAINING_SOURCE_KINDS + (EVALUATION_SOURCE_KIND,)

#: The four splits the plan names (``README.md`` §9.2, ``TRAINING_PLAN.md`` §4).
SPLIT_NAMES: tuple[str, ...] = ("train", "dev", "confirm", "adapt")
#: Additional split that only ever holds explicitly barred records.  Nothing may
#: be fitted from it; it exists so quarantined data is named instead of hidden.
OPTIONAL_SPLIT_NAMES: tuple[str, ...] = ("quarantine",)
ALL_SPLIT_NAMES: tuple[str, ...] = SPLIT_NAMES + OPTIONAL_SPLIT_NAMES

#: Permanently banned training seeds (``EXECUTION_PLAN.md`` §6).
BANNED_SEED_MIN = 60000
BANNED_SEED_MAX = 60199

DIGEST_SCOPE_TEXT = "text/lf-normalized"
DIGEST_SCOPE_BINARY = "binary/raw"
DIGEST_SCOPES: tuple[str, ...] = (DIGEST_SCOPE_TEXT, DIGEST_SCOPE_BINARY)

#: How the canonical bytes of a registry are produced.  Recorded next to every
#: registry digest so a verifier can reproduce it.
TEXT_CANONICALIZATION = "json/sorted-keys-compact-utf8-lf"

_HEX64 = set("0123456789abcdef")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,199}$")
_FOREIGN_HOST_RE = re.compile(r"^/(root|mnt|home|tmp|opt|usr|srv|var)/")

#: Fields that are excluded from the registry digest because they would make the
#: digest self-referential.
#: Problem codes that are fatal in strict mode and reported as degradations when the
#: caller asks for the minimal trustworthy subset (``strict=False``).  They never
#: describe a lineage violation — only the *granularity* of the evidence.
ADVISORY_CODES = frozenset({"P_SOURCE_ENUMERATION_LIMIT"})

_DIGEST_EXCLUDED_KEYS = frozenset({"split_registry_sha", "split_registry_sha_scope", "validation"})


class LineageError(RuntimeError):
    """Raised only by the low-level helpers (bad bytes/scope), never by the validator."""


# ---------------------------------------------------------------------------
# 2. Line endings and stable digests
# ---------------------------------------------------------------------------


def _as_bytes(data: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    raise LineageError(f"digest input must be bytes or str, got {type(data).__name__}")


def normalise_line_endings(data: bytes | bytearray | memoryview | str) -> bytes:
    """Return ``data`` with ``CRLF`` and lone ``CR`` folded to ``LF``.

    Text artefacts are compared/hashed on this form so that a checkout with
    ``core.autocrlf=true`` and one without produce the *same* identity.  This is
    exactly the ``digest_scope="text/lf-normalized"`` scope.
    """

    return _as_bytes(data).replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def text_lf_digest(data: bytes | bytearray | memoryview | str) -> str:
    """``sha256`` of the LF-normalised text form (``text/lf-normalized``)."""

    return hashlib.sha256(normalise_line_endings(data)).hexdigest()


def binary_raw_digest(data: bytes | bytearray | memoryview | str) -> str:
    """``sha256`` of the raw bytes (``binary/raw``) — no normalisation at all."""

    return hashlib.sha256(_as_bytes(data)).hexdigest()


def digest_bytes(data: bytes | bytearray | memoryview | str, digest_scope: str) -> str:
    """Dispatch on an explicit scope.  An unlabelled digest is not allowed."""

    if digest_scope == DIGEST_SCOPE_TEXT:
        return text_lf_digest(data)
    if digest_scope == DIGEST_SCOPE_BINARY:
        return binary_raw_digest(data)
    raise LineageError(
        f"unknown digest_scope {digest_scope!r}; expected one of {list(DIGEST_SCOPES)}")


def file_identity(path: str | Path, digest_scope: str) -> dict[str, Any]:
    """Identity record for a file: ``sha256`` **and** its scope, never one alone."""

    p = Path(path)
    raw = p.read_bytes()
    return {
        "path": str(path).replace("\\", "/"),
        "sha256": digest_bytes(raw, digest_scope),
        "digest_scope": digest_scope,
        "size_bytes": len(raw),
    }


def canonical_json(payload: Any) -> str:
    """Compact, key-sorted JSON used as the canonical byte form of a registry."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def stable_registry_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Stable digest of all registered content (sorted keys + compact JSON).

    The returned mapping always carries *both* the digest and the scope, because
    an identity field with a bare ``sha256`` is rejected elsewhere in this module.
    """

    if not isinstance(payload, Mapping):
        raise LineageError(f"payload must be a mapping, got {type(payload).__name__}")
    pruned = {k: v for k, v in payload.items() if k not in _DIGEST_EXCLUDED_KEYS}
    text = canonical_json(pruned)
    normalised = normalise_line_endings(text)
    return {
        "sha256": hashlib.sha256(normalised).hexdigest(),
        "digest_scope": DIGEST_SCOPE_TEXT,
        "canonicalization": TEXT_CANONICALIZATION,
        "canonical_bytes": len(normalised),
    }


# ---------------------------------------------------------------------------
# 3. Ancestor identity and grouping
# ---------------------------------------------------------------------------


def _clean_part(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise LineageError(f"{name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise LineageError(f"{name} must not be empty")
    if not _ID_RE.match(text):
        raise LineageError(f"{name} contains characters outside [A-Za-z0-9._:/+-]: {text!r}")
    return text


def ancestor_id(source_kind: str, source_id: str, episode: Any, task_family: str) -> str:
    """Canonical ancestor id: ``kind|source|episode|family``.

    ``seed`` is deliberately absent: a re-run of the same episode under a new seed
    is the same task ancestor (and is collapsed by :func:`group_key`).
    """

    kind = _clean_part(source_kind, "source_kind")
    src = _clean_part(source_id, "source_id")
    fam = _clean_part(task_family, "task_family")
    ep = episode if isinstance(episode, str) else str(episode)
    ep = _clean_part(ep, "episode")
    return f"{kind}|{src}|{ep}|{fam}"


def group_key(task_family: str, episode: Any) -> str:
    """Independence group of an ancestor: ``(task_family, episode)``.

    ``TRAINING_PLAN.md`` §4: "相邻帧、裁剪、重标、重复重放、由同一起点生成的多个
    分支属于同组。不同 seed 也要检查是否实际重复同轨迹。"  Two records that share
    this key are **one** independent sample, whatever their seeds say.
    """

    fam = _clean_part(task_family, "task_family")
    ep = episode if isinstance(episode, str) else str(episode)
    return f"{fam}|{_clean_part(ep, 'episode')}"


class _UnionFind:
    """Minimal union-find used to merge groups that share a trajectory hash."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            # Deterministic representative: lexicographic minimum.
            lo, hi = (a, b) if a < b else (b, a)
            self._parent[hi] = lo


# ---------------------------------------------------------------------------
# 4. Small validation helpers
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX64 for c in value)


def _iter_dicts(obj: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        yield obj
        for value in obj.values():
            yield from _iter_dicts(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_dicts(value)


def _iter_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            if isinstance(key, str):
                yield key
            yield from _iter_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_strings(value)


def _normalise_path_text(text: str) -> str:
    return text.replace("\\", "/").strip()


def _unwrap(payload: Any, key: str, default: Any) -> Any:
    """Accept either the bare block or a registry document wrapping ``key``."""

    if isinstance(payload, Mapping):
        wrapped = payload.get(key)
        return wrapped if wrapped is not None else default
    return payload


def load_json_text(data: bytes) -> Any:
    """Parse JSON from bytes while tolerating CRLF/UTF-8 BOM."""

    text = normalise_line_endings(data).decode("utf-8-sig")
    return json.loads(text)


def _problem(code: str, message: str, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "message": message}
    item.update({k: v for k, v in extra.items() if v is not None})
    return item


# ---------------------------------------------------------------------------
# 5. Config loading
# ---------------------------------------------------------------------------


def _load_config(config: Any, base_dir: str | Path | None) -> tuple[dict[str, Any], Path | None]:
    if isinstance(config, Mapping):
        return dict(config), None
    if isinstance(config, (str, Path)):
        path = Path(config)
        raw = path.read_bytes()
        payload = load_json_text(raw)
        if not isinstance(payload, dict):
            raise LineageError(f"config {path} must contain a JSON object")
        return payload, path
    raise LineageError(f"config must be a path or a mapping, got {type(config).__name__}")


def _resolve(path_text: str, base_dir: Path | None) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    if base_dir is not None:
        candidate = base_dir / p
        if candidate.exists():
            return candidate
        # configs/v430/splits.json binds repo-root-relative registry paths.
        for parent in base_dir.parents:
            candidate = parent / p
            if candidate.exists():
                return candidate
        return base_dir / p
    return p


# ---------------------------------------------------------------------------
# 6. The validator
# ---------------------------------------------------------------------------

_SEALED_PLACEHOLDER = "\u0000"  # never present in real config text


def validate_splits(
    config: Any,
    *,
    base_dir: str | Path | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate a frozen split config; **never** returns a silent pass.

    Parameters
    ----------
    config:
        A path to ``configs/v430/splits.json`` (preferred) or an already-loaded
        mapping.  A mapping may inline ``families`` / ``holdout_families`` /
        ``pretraining_sources``; a path normally binds those registries through
        ``registries`` (relative paths resolve against the config's directory and
        then its ancestors, so repo-root-relative bindings work).
    base_dir:
        Override for resolving relative registry paths (tests use this).
    strict:
        When true, a missing/unreadable registry binding is a failure; when false
        the missing registry is reported as ``not_checked`` instead.  The CLI must
        use the default.

    Returns
    -------
    dict with ``passed`` (bool), ``problems`` (list), ``statistics`` (dict),
    ``split_registry_sha`` + ``split_registry_sha_scope``, ``exit_code`` and the
    honest ``not_checked`` list.
    """

    problems: list[dict[str, Any]] = []
    checks: list[str] = []
    not_checked: list[str] = [
        "records only: this validator cannot verify that the cited evidence files "
        "say what the registry claims (no payload/asset bytes are read)",
        "no sealed asset exists on this host, so sealed protection is reference-based "
        "(a sealed path appearing anywhere in the config is rejected)",
        "repeat trajectories are only recognised when an ancestor supplies "
        "trajectory_sha256; seed-only re-runs are collapsed by (task_family, episode)",
        "a family listed as untouched is not re-derived from pixels; it relies on the "
        "recorded training set membership of the registered sources",
    ]

    cfg: dict[str, Any] = {}
    config_path: Path | None = None
    try:
        cfg, config_path = _load_config(config, base_dir)
    except Exception as exc:  # noqa: BLE001 - fail closed, never raise at the CLI
        return _report(
            passed=False,
            problems=[_problem("P_CONFIG_UNREADABLE", f"cannot load split config: {exc}")],
            statistics={},
            config={},
            checks=checks,
            not_checked=not_checked,
        )

    if base_dir is None:
        base_dir = config_path.parent if config_path is not None else None

    # --- 6.1 top-level identity -------------------------------------------------
    checks.append("identity")
    schema = cfg.get("schema")
    if schema != SCHEMA:
        problems.append(_problem("P_SCHEMA", f"schema must be {SCHEMA!r}, got {schema!r}"))
    if cfg.get("spec_revision") != SPEC_REVISION:
        problems.append(_problem(
            "P_SPEC_REVISION",
            f"spec_revision must be {SPEC_REVISION!r}, got {cfg.get('spec_revision')!r}"))
    experiment_id = cfg.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        problems.append(_problem("P_EXPERIMENT_ID", "experiment_id must be a non-empty string"))
    frozen_at = cfg.get("frozen_at")
    if not isinstance(frozen_at, str) or not _has_timezone(frozen_at):
        problems.append(_problem(
            "P_FROZEN_AT",
            "frozen_at must be an ISO-8601 timestamp with an explicit UTC offset "
            f"(e.g. 2026-11-15T18:00:00+08:00), got {frozen_at!r}"))
    timezone = cfg.get("timezone")
    if not isinstance(timezone, str) or not timezone.strip():
        problems.append(_problem("P_TIMEZONE", "timezone must be a non-empty string"))

    problems.extend(_check_roots(cfg.get("roots")))
    checks.append("roots")

    # --- 6.2 digest discipline --------------------------------------------------
    checks.append("digest_scope")
    problems.extend(_check_digest_scopes(cfg))

    # --- 6.3 registries (bindings + content) ------------------------------------
    checks.append("registries")
    registries, binding_problems = _load_registries(cfg, base_dir, strict)
    problems.extend(binding_problems)

    families = cfg.get("families")
    if families is None:
        families = registries.get("task_families")
        if families is None:
            problems.append(_problem(
                "P_FAMILIES_MISSING",
                "no family registry: inline 'families' or a frozen binding with "
                "role='task_families' is required"))
            families = []
    families = _unwrap(families, "families", [])

    holdout_block = cfg.get("holdout_families")
    if holdout_block is None:
        holdout_block = registries.get("holdout_families")
        if holdout_block is None:
            problems.append(_problem(
                "P_HOLDOUT_MISSING",
                "no hold-out registry: inline 'holdout_families' or a frozen binding "
                "with role='holdout_families' is required"))
            holdout_block = {}
    holdout_block = _unwrap(holdout_block, "holdout_families", holdout_block)
    if isinstance(holdout_block, list):
        holdout_block = {"frozen": holdout_block}
    elif not isinstance(holdout_block, Mapping):
        holdout_block = {}
    elif "frozen" not in holdout_block and "reserved" not in holdout_block:
        holdout_block = dict(holdout_block)

    sources = cfg.get("pretraining_sources")
    if sources is None:
        sources = registries.get("pretraining_sources")
        if sources is None:
            problems.append(_problem(
                "P_SOURCES_MISSING",
                "no pretraining-source register: inline 'pretraining_sources' or a "
                "frozen binding with role='pretraining_sources' is required"))
            sources = []
    sources = _unwrap(sources, "sources", [])

    family_map, family_problems = _check_families(families)
    problems.extend(family_problems)
    checks.append("families")

    source_map, source_kind_seen, source_problems = _check_sources(sources, family_map)
    problems.extend(source_problems)
    checks.append("pretraining_sources")

    missing_kinds = [k for k in PRETRAINING_SOURCE_KINDS if k not in source_kind_seen]
    if missing_kinds:
        problems.append(_problem(
            "P_SOURCE_KIND_UNREGISTERED",
            "every pretraining source category must be registered, including the ones "
            f"with no fitted data yet; missing: {missing_kinds}",
            missing=missing_kinds))

    holdouts, holdout_problems = _check_holdout_registry(holdout_block, family_map)
    problems.extend(holdout_problems)
    checks.append("holdout_families")

    # --- 6.4 the ancestor table -------------------------------------------------
    checks.append("ancestors")
    ancestors_raw = cfg.get("ancestors")
    if not isinstance(ancestors_raw, list):
        problems.append(_problem("P_ANCESTORS_MISSING", "ancestors must be a list"))
        ancestors_raw = []
    ancestor_map, ancestor_problems, ancestor_groups = _check_ancestors(ancestors_raw, family_map)
    problems.extend(ancestor_problems)

    # --- 6.5 splits -------------------------------------------------------------
    checks.append("splits")
    splits_raw = cfg.get("splits")
    split_members: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(splits_raw, Mapping):
        problems.append(_problem("P_SPLITS_MISSING", "splits must be an object with "
                                                    f"train/dev/confirm/adapt, got {type(splits_raw).__name__}"))
        splits_raw = {}
        for name in ALL_SPLIT_NAMES:
            split_members[name] = []
    else:
        for name in SPLIT_NAMES:
            if name not in splits_raw:
                problems.append(_problem("P_SPLIT_MISSING", f"split {name!r} is required",
                                         split=name))
        for name, block in splits_raw.items():
            if name not in ALL_SPLIT_NAMES:
                problems.append(_problem(
                    "P_SPLIT_UNKNOWN",
                    f"unknown split {name!r}; allowed: {list(ALL_SPLIT_NAMES)}"))
                continue
            members, member_problems = _check_split_block(
                name, block, ancestor_map, family_map, holdouts)
            problems.extend(member_problems)
            split_members[name] = members
        for name in ALL_SPLIT_NAMES:
            split_members.setdefault(name, [])

    if not split_members.get("train"):
        problems.append(_problem("P_TRAIN_EMPTY", "the train split must not be empty"))

    # --- 6.6 ancestor bookkeeping: assigned once, attributed to a source --------
    checks.append("ancestor_bookkeeping")
    problems.extend(_check_ancestor_bookkeeping(ancestor_map, split_members, source_map))

    # --- 6.7 cross-split independence ------------------------------------------
    checks.append("cross_split")
    group_members = _build_groups(split_members, ancestor_groups)
    problems.extend(_check_cross_split(group_members))

    # --- 6.8 hold-out protection ------------------------------------------------
    checks.append("holdout_protection")
    problems.extend(_check_holdout_protection(
        holdouts, family_map, split_members, source_map, ancestor_map, splits_raw))

    # --- 6.9 reserved / banned seeds -------------------------------------------
    checks.append("reserved_seeds")
    reserved = cfg.get("reserved_training_exclusion")
    reserved_ranges, reserved_problems = _check_reserved_block(reserved)
    problems.extend(reserved_problems)
    problems.extend(_check_seed_ranges(split_members, source_map, reserved_ranges))

    # --- 6.10 sealed ------------------------------------------------------------
    checks.append("sealed")
    sealed_patterns, sealed_problems = _check_sealed_block(cfg.get("sealed"))
    problems.extend(sealed_problems)
    problems.extend(_check_sealed_references(cfg, sealed_patterns))

    # --- 6.11 confirm batches (spent seed ranges) -------------------------------
    checks.append("confirm_batches")
    problems.extend(_check_confirm_batches(cfg.get("confirm_batches"), reserved_ranges))

    # --- statistics and digest --------------------------------------------------
    statistics = _statistics(
        families=family_map,
        sources=source_map,
        source_kinds=source_kind_seen,
        holdouts=holdouts,
        ancestors=ancestor_map,
        ancestor_groups=ancestor_groups,
        split_members=split_members,
        problems=problems,
        reserved_ranges=reserved_ranges,
    )

    # --- strict / degraded tiering ---------------------------------------------
    # A code in ADVISORY_CODES is fatal in strict mode (the default) and a readable
    # *degradation* otherwise.  Nothing is ever silently dropped: the report always
    # lists every finding once, under either "problems" or "degradations".
    degradations = [p for p in problems
                    if not strict and p["code"] in ADVISORY_CODES]
    fatal = [p for p in problems if p not in degradations]
    statistics["degradation_count"] = len(degradations)

    digest = stable_registry_digest(cfg)

    return _report(
        passed=not fatal,
        problems=fatal,
        degradations=degradations,
        statistics=statistics,
        config=cfg,
        checks=checks,
        not_checked=not_checked,
        digest=digest,
        mode="degraded" if not strict else "strict",
    )


# ---------------------------------------------------------------------------
# 7. Sub-checks
# ---------------------------------------------------------------------------


def _has_timezone(text: str) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if stripped.endswith("Z"):
        return True
    return bool(re.search(r"[+-]\d{2}:?\d{2}$", stripped))


def _check_roots(roots: Any) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    if not isinstance(roots, Mapping) or not roots:
        return [_problem("P_ROOTS", "roots must be a non-empty object of explicit paths")]
    for name, value in roots.items():
        if not isinstance(value, str) or not value.strip():
            problems.append(_problem("P_ROOTS", f"roots.{name} must be a non-empty string"))
            continue
        text = _normalise_path_text(value)
        if _FOREIGN_HOST_RE.match(text):
            problems.append(_problem(
                "P_ROOT_FOREIGN_PATH",
                f"roots.{name} hard-codes an execution-host POSIX path ({value!r}); "
                "bind paths per host instead (EXECUTION_PLAN §4)",
                root=name))
    return problems


def _check_digest_scopes(obj: Any) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for node in _iter_dicts(obj):
        if "sha256" not in node:
            continue
        scope = node.get("digest_scope")
        label = node.get("path") or node.get("id") or node.get("role") or "<anonymous>"
        if scope is None:
            problems.append(_problem(
                "P_DIGEST_SCOPE_MISSING",
                f"identity {label!r} carries sha256 without digest_scope; every digest "
                f"must be labelled {DIGEST_SCOPE_TEXT!r} or {DIGEST_SCOPE_BINARY!r}",
                identity=str(label)))
        elif scope not in DIGEST_SCOPES:
            problems.append(_problem(
                "P_DIGEST_SCOPE_UNKNOWN",
                f"identity {label!r} has unknown digest_scope {scope!r}",
                identity=str(label)))
    return problems


def _load_registries(
    cfg: Mapping[str, Any], base_dir: Path | None, strict: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    loaded: dict[str, Any] = {}
    bindings = cfg.get("registries")
    if bindings is None:
        return loaded, problems
    if not isinstance(bindings, list):
        return loaded, [_problem("P_BINDING_SHAPE", "registries must be a list")]

    for entry in bindings:
        if not isinstance(entry, Mapping):
            problems.append(_problem("P_BINDING_SHAPE", "each registry binding must be an object"))
            continue
        role = entry.get("role")
        path_text = entry.get("path")
        expected = entry.get("sha256")
        scope = entry.get("digest_scope")
        frozen = entry.get("frozen", True)
        if not isinstance(role, str) or not role:
            problems.append(_problem("P_BINDING_SHAPE", "registry binding needs a non-empty role"))
            continue
        if not isinstance(path_text, str) or not path_text:
            problems.append(_problem("P_BINDING_SHAPE", f"registry {role!r} needs a path"))
            continue
        if not _is_hex64(expected):
            problems.append(_problem(
                "P_BINDING_SHAPE",
                f"registry {role!r} needs a lowercase 64-hex sha256", role=role))
            continue
        if scope not in DIGEST_SCOPES:
            problems.append(_problem(
                "P_BINDING_SHAPE",
                f"registry {role!r} needs an explicit digest_scope", role=role))
            continue

        resolved = _resolve(path_text, base_dir)
        try:
            raw = resolved.read_bytes()
        except OSError as exc:
            if strict and frozen:
                problems.append(_problem(
                    "P_BINDING_MISSING",
                    f"bound registry {path_text} cannot be read: {exc}", role=role))
            continue
        if frozen:
            actual = digest_bytes(raw, scope)
            if actual != expected:
                problems.append(_problem(
                    "P_BINDING_SHA_MISMATCH",
                    f"bound registry {path_text} does not match its declared {scope} "
                    f"sha256 (declared {expected}, actual {actual})",
                    role=role, path=path_text))
                continue
        try:
            payload = load_json_text(raw)
        except Exception as exc:  # noqa: BLE001
            problems.append(_problem(
                "P_BINDING_LOAD", f"bound registry {path_text} is not valid JSON: {exc}",
                role=role))
            continue
        loaded[role] = payload

    for role, schema in (
        ("task_families", FAMILY_REGISTRY_SCHEMA),
        ("pretraining_sources", SOURCE_REGISTRY_SCHEMA),
        ("holdout_families", HOLDOUT_REGISTRY_SCHEMA),
        ("runs", RUNS_REGISTRY_SCHEMA),
    ):
        payload = loaded.get(role)
        if payload is None:
            continue
        if isinstance(payload, Mapping) and payload.get("schema") not in (schema, None):
            problems.append(_problem(
                "P_REGISTRY_SCHEMA",
                f"registry {role!r} declares schema {payload.get('schema')!r}, "
                f"expected {schema!r}", role=role))
    return loaded, problems


def _check_families(families: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    family_map: dict[str, dict[str, Any]] = {}
    if not isinstance(families, list):
        return family_map, [_problem("F_NOT_LIST", "the family registry must be a list")]

    for index, entry in enumerate(families):
        if not isinstance(entry, Mapping):
            problems.append(_problem("F_FIELD", f"family #{index} must be an object"))
            continue
        fid = entry.get("family_id")
        if not isinstance(fid, str) or not _ID_RE.match(fid):
            problems.append(_problem("F_FIELD", f"family #{index} has a bad family_id: {fid!r}"))
            continue
        if fid in family_map:
            problems.append(_problem("F_DUPLICATE_ID", f"family {fid!r} is declared twice",
                                     family=fid))
            continue
        missing = [k for k in ("game", "stage_index", "stage_name", "difficulty",
                               "difficulty_index", "team", "status", "route_family")
                   if k not in entry]
        if missing:
            problems.append(_problem("F_FIELD", f"family {fid!r} missing fields {missing}",
                                     family=fid))
            continue
        if not _is_int(entry["stage_index"]) or not _is_int(entry["difficulty_index"]) \
                or not _is_int(entry["team"]):
            problems.append(_problem(
                "F_FIELD", f"family {fid!r} needs integer stage_index/difficulty_index/team",
                family=fid))
            continue
        status = entry.get("status")
        if status not in ("touched_by_training", "instrument_only", "measurement_only",
                          "untouched", "unknown"):
            problems.append(_problem("F_BAD_STATUS", f"family {fid!r} has status {status!r}",
                                     family=fid))
            continue
        evidence = entry.get("clean_evidence")
        if status in ("untouched", "instrument_only", "measurement_only"):
            if not isinstance(evidence, list) or not evidence:
                problems.append(_problem(
                    "F_CLEAN_NO_EVIDENCE",
                    f"family {fid!r} claims status {status!r} without clean_evidence; "
                    "a cleanliness claim needs citable records (fail-closed)",
                    family=fid))
        if status == "touched_by_training":
            touched_by = entry.get("touched_by")
            if not isinstance(touched_by, list) or not any(
                    k in PRETRAINING_SOURCE_KINDS for k in touched_by):
                problems.append(_problem(
                    "F_STATUS_EXPOSURE",
                    f"family {fid!r} is touched_by_training but lists no pretraining "
                    "source kind in touched_by", family=fid))
            if not isinstance(entry.get("training_evidence"), list) or not entry["training_evidence"]:
                problems.append(_problem(
                    "F_STATUS_EXPOSURE",
                    f"family {fid!r} is touched_by_training without training_evidence",
                    family=fid))
        family_map[fid] = dict(entry)
    return family_map, problems


def _check_sources(
    sources: Any, family_map: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[tuple[str, str], dict[str, Any]], set[str], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    source_map: dict[tuple[str, str], dict[str, Any]] = {}
    kinds_seen: set[str] = set()
    if not isinstance(sources, list):
        return source_map, kinds_seen, [_problem("P_SOURCES_SHAPE",
                                                 "pretraining_sources must be a list")]

    for index, entry in enumerate(sources):
        if not isinstance(entry, Mapping):
            problems.append(_problem("P_SOURCES_SHAPE", f"source #{index} must be an object"))
            continue
        kind = entry.get("kind")
        sid = entry.get("source_id")
        if kind not in KNOWN_SOURCE_KINDS:
            problems.append(_problem(
                "P_SOURCE_KIND_UNKNOWN",
                f"source {sid!r} has unknown kind {kind!r}; known kinds: "
                f"{list(KNOWN_SOURCE_KINDS)}", source_id=str(sid)))
            continue
        if not isinstance(sid, str) or not _ID_RE.match(sid):
            problems.append(_problem("P_SOURCES_SHAPE", f"source #{index} has a bad source_id"))
            continue
        if kind == EVALUATION_SOURCE_KIND:
            problems.append(_problem(
                "P_SOURCE_IS_EVALUATION",
                f"source {sid!r} uses the evaluation kind, which is not a pretraining "
                "source and must not be registered as one", source_id=sid))
            continue
        if (kind, sid) in source_map:
            problems.append(_problem("P_SOURCE_DUPLICATE",
                                     f"source {kind}/{sid} is declared twice", source_id=sid))
            continue
        kinds_seen.add(kind)

        status = entry.get("status")
        ancestor_ids = entry.get("ancestor_ids", [])
        if not isinstance(ancestor_ids, list):
            problems.append(_problem("P_SOURCES_SHAPE",
                                     f"source {sid!r} ancestor_ids must be a list",
                                     source_id=sid))
            continue
        for aid in ancestor_ids:
            if not isinstance(aid, str):
                problems.append(_problem("P_SOURCES_SHAPE",
                                         f"source {sid!r} has a non-string ancestor id",
                                         source_id=sid))
        enumeration = entry.get("ancestor_enumeration", "complete")
        if enumeration not in ("complete", "family_level"):
            problems.append(_problem(
                "P_SOURCE_ENUMERATION",
                f"source {kind}/{sid} needs ancestor_enumeration in "
                "{'complete','family_level'}", source_id=sid))
            continue
        touched_families = entry.get("touched_families")
        if enumeration == "family_level":
            if not isinstance(touched_families, list) or not touched_families or not all(
                    isinstance(f, str) for f in touched_families):
                problems.append(_problem(
                    "P_SOURCE_FAMILY_COVERAGE",
                    f"source {kind}/{sid} enumerates at family level but declares no "
                    "touched_families", source_id=sid))
            else:
                unknown = [f for f in touched_families if f not in family_map]
                if unknown:
                    problems.append(_problem(
                        "P_SOURCE_FAMILY_COVERAGE",
                        f"source {kind}/{sid} touches unregistered families {unknown}",
                        source_id=sid, families=unknown))
            limit = entry.get("enumeration_limit")
            if not isinstance(limit, str) or not limit.strip():
                problems.append(_problem(
                    "P_SOURCE_ENUMERATION_LIMIT",
                    f"source {kind}/{sid} enumerates at family level without an "
                    "enumeration_limit; the registry must say what cannot be enumerated "
                    "(strict mode fails, non-strict reports it as a degradation)",
                    source_id=sid))
        needs_ancestors = (
            enumeration == "complete"
            and status not in ("not_started", "not_used_for_pretraining",
                               "rule_based_no_training_data", "rejected_not_used",
                               "executed_instrument_only")
        )
        if needs_ancestors and not ancestor_ids:
            problems.append(_problem(
                "P_SOURCE_NO_ANCESTORS",
                f"source {kind}/{sid} has status {status!r} but registers no ancestors; "
                "declare an empty-data status instead of an unsupported training claim",
                source_id=sid))
        evidence = entry.get("evidence_paths")
        if not isinstance(evidence, list) or not evidence or not all(
                isinstance(p, str) and p.strip() for p in evidence):
            problems.append(_problem(
                "P_SOURCE_NO_EVIDENCE",
                f"source {kind}/{sid} needs non-empty evidence_paths", source_id=sid))
        uses_fixtures = entry.get("uses_retail_replay_fixtures")
        if not isinstance(uses_fixtures, bool):
            problems.append(_problem(
                "P_SOURCE_FIXTURE_FLAG",
                f"source {kind}/{sid} must declare uses_retail_replay_fixtures (bool) so the "
                "retail-replay leak rule can be evaluated", source_id=sid))
        source_map[(kind, sid)] = dict(entry)
    return source_map, kinds_seen, problems


def _check_holdout_registry(
    block: Any, family_map: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    if not isinstance(block, Mapping):
        return {"frozen": [], "reserved": []}, [
            _problem("H_SHAPE", "holdout_families must be an object with 'frozen'")]
    frozen = block.get("frozen")
    reserved = block.get("reserved", [])
    if not isinstance(frozen, list):
        problems.append(_problem("H_SHAPE", "holdout_families.frozen must be a list"))
        frozen = []
    if not isinstance(reserved, list):
        problems.append(_problem("H_SHAPE", "holdout_families.reserved must be a list"))
        reserved = []
    if len(frozen) < 2:
        problems.append(_problem(
            "H_TOO_FEW",
            f"at least two frozen clean hold-out families are required (G410-6), got {len(frozen)}"))

    cleaned: dict[str, list[dict[str, Any]]] = {"frozen": [], "reserved": []}
    for bucket, entries in (("frozen", frozen), ("reserved", reserved)):
        for entry in entries:
            if not isinstance(entry, Mapping):
                problems.append(_problem("H_SHAPE", f"{bucket} hold-out entry must be an object"))
                continue
            fid = entry.get("family_id")
            if not isinstance(fid, str) or fid not in family_map:
                problems.append(_problem(
                    "H_UNKNOWN_FAMILY",
                    f"hold-out family {fid!r} is not in the family registry "
                    "(renaming a label is not a new family)", family=str(fid)))
                continue
            status = family_map[fid].get("status")
            if status not in ("untouched", "instrument_only", "measurement_only"):
                problems.append(_problem(
                    "H_DIRTY_FAMILY",
                    f"hold-out family {fid!r} has registry status {status!r}; only a family "
                    "with no fitting source may be declared clean", family=fid))
            claims = entry.get("clean_evidence")
            if not isinstance(claims, list) or not claims:
                problems.append(_problem(
                    "H_NO_EVIDENCE",
                    f"hold-out family {fid!r} declares no clean_evidence", family=fid))
            else:
                for claim in claims:
                    if not isinstance(claim, Mapping):
                        problems.append(_problem(
                            "H_NO_EVIDENCE", f"hold-out family {fid!r} has a malformed claim",
                            family=fid))
                        continue
                    if not claim.get("claim") or not isinstance(claim.get("claim"), str):
                        problems.append(_problem(
                            "H_NO_EVIDENCE",
                            f"hold-out family {fid!r} has a claim without text", family=fid))
                    paths = claim.get("records_absent_from", [])
                    if not isinstance(paths, list) or not paths:
                        problems.append(_problem(
                            "H_NO_EVIDENCE",
                            f"hold-out family {fid!r} claim {claim.get('claim')!r} does not "
                            "name the records it is absent from (records_absent_from)",
                            family=fid))
            cleaned[bucket].append(dict(entry, family_id=fid, status=status))
    return cleaned, problems


def _check_ancestors(
    ancestors: Sequence[Any], family_map: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    problems: list[dict[str, Any]] = []
    ancestor_map: dict[str, dict[str, Any]] = {}
    groups: dict[str, str] = {}

    for index, entry in enumerate(ancestors):
        if not isinstance(entry, Mapping):
            problems.append(_problem("A_MISSING_FIELD", f"ancestor #{index} must be an object"))
            continue
        required = ("source_kind", "source_id", "episode", "seed", "task_family")
        missing = [k for k in required if k not in entry]
        if missing:
            problems.append(_problem(
                "A_MISSING_FIELD", f"ancestor #{index} missing fields {missing}",
                ancestor_index=index))
            continue
        kind = entry["source_kind"]
        if kind not in KNOWN_SOURCE_KINDS:
            problems.append(_problem(
                "A_UNKNOWN_SOURCE_KIND",
                f"ancestor #{index} has unknown source_kind {kind!r}; known kinds: "
                f"{list(KNOWN_SOURCE_KINDS)}", ancestor_index=index))
            continue
        seed = entry["seed"]
        if not _is_int(seed) or seed < 0:
            problems.append(_problem(
                "A_BAD_SEED", f"ancestor #{index} seed must be a non-negative integer",
                ancestor_index=index))
            continue
        family = entry["task_family"]
        if not isinstance(family, str) or family not in family_map:
            problems.append(_problem(
                "A_UNKNOWN_FAMILY",
                f"ancestor #{index} names task_family {family!r}, which is not in the "
                "family registry (a renamed label is not a new family)",
                family=str(family)))
            continue
        episode = entry["episode"]
        if isinstance(episode, str):
            if not episode.strip():
                problems.append(_problem("A_BAD_EPISODE",
                                         f"ancestor #{index} episode is empty",
                                         ancestor_index=index))
                continue
        elif not _is_int(episode) or episode < 0:
            problems.append(_problem(
                "A_BAD_EPISODE",
                f"ancestor #{index} episode must be a non-empty string or a non-negative int",
                ancestor_index=index))
            continue
        try:
            aid = ancestor_id(kind, entry["source_id"], episode, family)
        except LineageError as exc:
            problems.append(_problem("A_MISSING_FIELD", f"ancestor #{index}: {exc}",
                                     ancestor_index=index))
            continue
        traj = entry.get("trajectory_sha256")
        if traj is not None and not _is_hex64(traj):
            problems.append(_problem(
                "A_BAD_TRAJECTORY",
                f"ancestor {aid} trajectory_sha256 must be a lowercase 64-hex digest",
                ancestor_id=aid))
        if aid in ancestor_map:
            problems.append(_problem(
                "A_DUPLICATE", f"ancestor {aid} is declared twice in the ancestor table",
                ancestor_id=aid))
            continue
        record = dict(entry)
        record["ancestor_id"] = aid
        ancestor_map[aid] = record
        groups[aid] = group_key(family, episode)
    return ancestor_map, problems, groups


def _check_split_block(
    name: str,
    block: Any,
    ancestor_map: Mapping[str, Mapping[str, Any]],
    family_map: Mapping[str, Mapping[str, Any]],
    holdouts: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    if not isinstance(block, Mapping):
        return members, [_problem("X_SPLIT_SHAPE", f"split {name!r} must be an object",
                                  split=name)]

    ids = block.get("ancestor_ids", [])
    if not isinstance(ids, list):
        return members, [_problem("X_SPLIT_SHAPE",
                                  f"split {name!r} ancestor_ids must be a list", split=name)]

    seen: set[str] = set()
    for aid in ids:
        if not isinstance(aid, str):
            problems.append(_problem("X_SPLIT_SHAPE", f"split {name!r} has a non-string id",
                                     split=name))
            continue
        record = ancestor_map.get(aid)
        if record is None:
            problems.append(_problem(
                "X_UNKNOWN_ANCESTOR",
                f"split {name!r} references ancestor {aid!r}, which is not in the "
                "ancestor table", split=name, ancestor_id=aid))
            continue
        if aid in seen:
            problems.append(_problem(
                "A_DUPLICATE",
                f"split {name!r} lists ancestor {aid!r} twice (duplicate counting)",
                split=name, ancestor_id=aid))
            continue
        seen.add(aid)
        members.append(record)

    if not members:
        reason = block.get("empty_reason")
        if not isinstance(reason, str) or not reason.strip():
            problems.append(_problem(
                "P_SPLIT_EMPTY",
                f"split {name!r} has no ancestors and no empty_reason", split=name))

    declared_families = block.get("target_families")
    if declared_families is not None:
        if not isinstance(declared_families, list) or not all(
                isinstance(f, str) for f in declared_families):
            problems.append(_problem("X_SPLIT_SHAPE",
                                     f"split {name!r} target_families must be a list of ids",
                                     split=name))
        else:
            for fid in declared_families:
                if fid not in family_map:
                    problems.append(_problem(
                        "X_UNKNOWN_FAMILY",
                        f"split {name!r} targets unknown family {fid!r}",
                        split=name, family=fid))
    return members, problems


def _check_ancestor_bookkeeping(
    ancestor_map: Mapping[str, Mapping[str, Any]],
    split_members: Mapping[str, Sequence[Mapping[str, Any]]],
    source_map: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    assignment: dict[str, list[str]] = {}
    for split, records in split_members.items():
        for record in records:
            assignment.setdefault(record["ancestor_id"], []).append(split)

    for key, source in source_map.items():
        for aid in source.get("ancestor_ids", []) or []:
            if aid not in ancestor_map:
                problems.append(_problem(
                    "P_SOURCE_UNKNOWN_ANCESTOR",
                    f"source {key[0]}/{key[1]} claims ancestor {aid!r}, which is not in the "
                    "ancestor table", source_id=key[1]))

    for aid, record in ancestor_map.items():
        where = assignment.get(aid, [])
        if not where:
            problems.append(_problem(
                "A_UNASSIGNED",
                f"ancestor {aid} belongs to no split; every registered record must be "
                "named by train/dev/confirm/adapt (or quarantine)", ancestor_id=aid))
        elif len(where) > 1:
            problems.append(_problem(
                "A_MULTI_SPLIT", f"ancestor {aid} is assigned to several splits {where}",
                ancestor_id=aid))
        if record["source_kind"] in PRETRAINING_SOURCE_KINDS:
            key = (record["source_kind"], record["source_id"])
            source = source_map.get(key)
            if source is None:
                problems.append(_problem(
                    "A_UNATTRIBUTED",
                    f"ancestor {aid} has pretraining kind {record['source_kind']!r} but no "
                    f"registered source {key[0]}/{key[1]} claims it",
                    ancestor_id=aid))
            elif aid not in set(source.get("ancestor_ids", [])):
                problems.append(_problem(
                    "A_UNATTRIBUTED",
                    f"source {key[0]}/{key[1]} does not list ancestor {aid} although it "
                    "produced it", ancestor_id=aid))
    return problems


def _build_groups(
    split_members: Mapping[str, Sequence[Mapping[str, Any]]],
    ancestor_groups: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Merge ``(family, episode)`` groups that share a trajectory digest.

    The base group is ``(task_family, episode)`` so a fresh seed on the same
    episode collapses; two episodes that expose the same ``trajectory_sha256`` are
    merged on top of that, which is the only machine-checkable form of "same
    trajectory under a different name".
    """

    uf = _UnionFind()
    for gkey in ancestor_groups.values():
        uf.find(gkey)
    by_traj: dict[str, str] = {}
    groups: dict[str, dict[str, Any]] = {}
    for split, records in split_members.items():
        for record in records:
            aid = record["ancestor_id"]
            gkey = ancestor_groups[aid]
            traj = record.get("trajectory_sha256")
            if isinstance(traj, str):
                other = by_traj.get(traj)
                if other is not None:
                    uf.union(gkey, other)
                else:
                    by_traj[traj] = gkey
    for split, records in split_members.items():
        for record in records:
            aid = record["ancestor_id"]
            root = uf.find(ancestor_groups[aid])
            slot = groups.setdefault(root, {"group": root, "ancestors": [], "splits": set(),
                                            "seeds": set(), "families": set(),
                                            "members_by_split": {}})
            slot["ancestors"].append(aid)
            slot["splits"].add(split)
            slot["seeds"].add(record["seed"])
            slot["families"].add(record["task_family"])
            slot["members_by_split"].setdefault(split, []).append(aid)
    for slot in groups.values():
        slot["splits"] = sorted(slot["splits"])
        slot["seeds"] = sorted(slot["seeds"])
        slot["families"] = sorted(slot["families"])
    return groups


def _check_cross_split(groups: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for group in groups.values():
        splits = group["splits"]

        # Two ids from the same (task_family, episode) inside one split are one task
        # recorded twice: that is double counting, not two independent samples.
        for split, members in group["members_by_split"].items():
            if len(members) > 1:
                problems.append(_problem(
                    "A_DUPLICATE_GROUP",
                    f"{len(members)} ancestors of split {split!r} belong to the same "
                    f"(task_family, episode) group {group['group']} with seeds "
                    f"{group['seeds']}; a fresh seed does not create a new task",
                    split=split, group=group["group"], seeds=list(group["seeds"]),
                    conflicting_ancestors=list(members)))

        if len(splits) < 2:
            continue
        for i, left in enumerate(splits):
            for right in splits[i + 1:]:
                code = "X_TRAIN_OVERLAP" if "train" in (left, right) else "X_SPLIT_OVERLAP"
                problems.append(_problem(
                    code,
                    f"ancestor group {group['group']} appears in both {left!r} and {right!r} "
                    f"(ancestors: {group['ancestors']}, seeds: {group['seeds']})",
                    split=f"{left}|{right}",
                    ancestor_id=group["ancestors"][0],
                    group=group["group"],
                    conflicting_ancestors=list(group["ancestors"])))
    return problems


def _check_holdout_protection(
    holdouts: Mapping[str, Any],
    family_map: Mapping[str, Mapping[str, Any]],
    split_members: Mapping[str, Sequence[Mapping[str, Any]]],
    source_map: Mapping[tuple[str, str], Mapping[str, Any]],
    ancestor_map: Mapping[str, Mapping[str, Any]],
    splits_raw: Mapping[str, Any],
) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    frozen_ids = [entry["family_id"] for entry in holdouts.get("frozen", [])]
    reserved_ids = [entry["family_id"] for entry in holdouts.get("reserved", [])]
    protected = frozen_ids + reserved_ids

    # (a) a clean family must not appear in an ordinary split
    for split, records in split_members.items():
        if split == "adapt":
            continue
        for record in records:
            if record["task_family"] in protected:
                problems.append(_problem(
                    "H_HOLDOUT_IN_SPLIT",
                    f"hold-out family {record['task_family']!r} appears in split {split!r} "
                    f"via ancestor {record['ancestor_id']}",
                    split=split, family=record["task_family"],
                    ancestor_id=record["ancestor_id"]))

    # (b) a clean family must not appear in *any* pretraining source's lineage
    for (kind, sid), source in source_map.items():
        families_hit: set[str] = set()
        declared = source.get("touched_families")
        if isinstance(declared, list):
            families_hit.update(f for f in declared if isinstance(f, str))
        for aid in source.get("ancestor_ids", []) or []:
            record = ancestor_map.get(aid)
            if record is not None:
                families_hit.add(record["task_family"])
        hit = sorted(f for f in families_hit if f in protected)
        if hit:
            problems.append(_problem(
                "H_TOUCHED_BY_SOURCE",
                f"pretraining source {kind}/{sid} declares it touched clean hold-out "
                f"families {hit}", source_id=sid, kind=kind, families=hit))

    # (c) the retail-replay leak rule
    fixture_users = [f"{k}/{sid}" for (k, sid), s in source_map.items()
                     if s.get("uses_retail_replay_fixtures") is True]
    for fid in protected:
        fixtures = family_map[fid].get("retail_replay_fixture_ids") or []
        if fixture_users and fixtures:
            problems.append(_problem(
                "H_INSTRUMENT_FIXTURE_LEAK",
                f"hold-out family {fid!r} has {len(fixtures)} retail replay fixture(s) and "
                f"source(s) {fixture_users} declare uses_retail_replay_fixtures=true; the "
                "retail corpus covers every TH08 (difficulty, stage) pair, so consuming it "
                "touches this hold-out family",
                family=fid, sources=fixture_users))

    # (d) the adapt split must target every frozen family
    adapt_block = splits_raw.get("adapt") if isinstance(splits_raw, Mapping) else None
    targets: list[str] = []
    if isinstance(adapt_block, Mapping) and isinstance(adapt_block.get("target_families"), list):
        targets = [f for f in adapt_block["target_families"] if isinstance(f, str)]
    elif isinstance(adapt_block, Mapping):
        targets = sorted({r["task_family"] for r in split_members.get("adapt", [])})
    missing = [fid for fid in frozen_ids if fid not in targets]
    if missing:
        problems.append(_problem(
            "H_NOT_TARGETED",
            f"the adapt split does not target frozen hold-out families {missing}",
            families=missing))
    extra = [fid for fid in targets if fid not in frozen_ids]
    if extra:
        problems.append(_problem(
            "H_EXTRA_TARGET",
            f"the adapt split targets {extra}, which are not frozen hold-out families",
            families=extra))

    problems.extend(_check_holdout_claims_vs_registry(holdouts, family_map))
    return problems


def _check_holdout_claims_vs_registry(
    holdouts: Mapping[str, Any], family_map: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    for bucket in ("frozen", "reserved"):
        for entry in holdouts.get(bucket, []):
            fid = entry["family_id"]
            registry_evidence = family_map.get(fid, {}).get("clean_evidence") or []
            if not registry_evidence:
                problems.append(_problem(
                    "H_REGISTRY_EVIDENCE_MISSING",
                    f"hold-out family {fid!r} has no clean_evidence in the family registry",
                    family=fid))
    return problems


def _check_reserved_block(
    reserved: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    ranges: list[dict[str, Any]] = []
    if not isinstance(reserved, Mapping):
        return ranges, [_problem(
            "P_RESERVED_MISSING",
            "reserved_training_exclusion must be an object declaring the banned seed "
            "range (60000-60199) and the protected ranges")]
    if reserved.get("read_sealed_results") is not False:
        problems.append(_problem(
            "P_SEALED_READ_ALLOWED",
            "reserved_training_exclusion.read_sealed_results must be false"))
    lo, hi = reserved.get("seed_min"), reserved.get("seed_max")
    if not _is_int(lo) or not _is_int(hi) or lo > hi:
        problems.append(_problem(
            "P_RESERVED_MISSING",
            "reserved_training_exclusion needs integer seed_min/seed_max"))
    elif (lo, hi) != (BANNED_SEED_MIN, BANNED_SEED_MAX):
        problems.append(_problem(
            "P_RESERVED_RANGE",
            f"the banned training range must be exactly {BANNED_SEED_MIN}-{BANNED_SEED_MAX}, "
            f"got {lo}-{hi}"))
    else:
        ranges.append({"name": "banned_training", "seed_min": lo, "seed_max": hi,
                       "role": "banned", "usage": "forbidden"})
    protected = reserved.get("protected_ranges")
    if not isinstance(protected, list) or not protected:
        problems.append(_problem(
            "P_RESERVED_MISSING",
            "reserved_training_exclusion.protected_ranges must list the historical "
            "protected confirmation ranges"))
        protected = []
    for entry in protected:
        if not isinstance(entry, Mapping):
            problems.append(_problem("P_RESERVED_RANGE", "protected range must be an object"))
            continue
        name = entry.get("name")
        rlo, rhi = entry.get("seed_min"), entry.get("seed_max")
        usage = entry.get("usage")
        if not isinstance(name, str) or not name:
            problems.append(_problem("P_RESERVED_RANGE", "protected range needs a name"))
            continue
        if not _is_int(rlo) or not _is_int(rhi) or rlo > rhi:
            problems.append(_problem("P_RESERVED_RANGE",
                                     f"protected range {name!r} needs integer seed_min/seed_max"))
            continue
        if usage not in ("unused", "spent", "reserved_for_v430"):
            problems.append(_problem(
                "P_RESERVED_RANGE",
                f"protected range {name!r} needs usage in "
                "{'unused','spent','reserved_for_v430'}, got {usage!r}"))
            continue
        if not isinstance(entry.get("evidence"), str) or not entry["evidence"].strip():
            problems.append(_problem(
                "P_RESERVED_EVIDENCE",
                f"protected range {name!r} needs an evidence path for its usage status"))
        ranges.append({"name": name, "seed_min": rlo, "seed_max": rhi,
                       "role": "protected", "usage": usage})
    for i, left in enumerate(ranges):
        for right in ranges[i + 1:]:
            if left["role"] == "banned" or right["role"] == "banned":
                continue
            if left["seed_min"] <= right["seed_max"] and right["seed_min"] <= left["seed_max"]:
                problems.append(_problem(
                    "S_PROTECTED_OVERLAP",
                    f"protected ranges {left['name']} and {right['name']} overlap"))
    return ranges, problems


def _check_seed_ranges(
    split_members: Mapping[str, Sequence[Mapping[str, Any]]],
    source_map: Mapping[tuple[str, str], Mapping[str, Any]],
    reserved_ranges: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []

    def banned(seed: int) -> bool:
        return BANNED_SEED_MIN <= seed <= BANNED_SEED_MAX

    def protected(seed: int) -> Mapping[str, Any] | None:
        for rng in reserved_ranges:
            if rng["role"] == "protected" and rng["seed_min"] <= seed <= rng["seed_max"]:
                return rng
        return None

    for split, records in split_members.items():
        for record in records:
            seed = record["seed"]
            if banned(seed):
                problems.append(_problem(
                    "S_BANNED_SEED",
                    f"split {split!r} ancestor {record['ancestor_id']} uses seed {seed}, which "
                    f"is inside the permanently banned training range "
                    f"{BANNED_SEED_MIN}-{BANNED_SEED_MAX}",
                    split=split, ancestor_id=record["ancestor_id"], seed=seed))
                continue
            rng = protected(seed)
            if rng is None:
                continue
            if split == "train":
                problems.append(_problem(
                    "S_RESERVED_IN_TRAIN",
                    f"train ancestor {record['ancestor_id']} uses seed {seed} from protected "
                    f"range {rng['name']} ({rng['seed_min']}-{rng['seed_max']}, "
                    f"usage={rng['usage']})",
                    split=split, ancestor_id=record["ancestor_id"], seed=seed,
                    range=rng["name"]))
    return problems


def _check_sealed_block(sealed: Any) -> tuple[list[str], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    patterns: list[str] = []
    if not isinstance(sealed, Mapping):
        return patterns, [_problem(
            "P_SEALED_MISSING",
            "sealed must be an object declaring the sealed path patterns and the "
            "sealed-range policy (absence must be explicit, not implied)")]
    raw_patterns = sealed.get("path_patterns")
    if not isinstance(raw_patterns, list) or not raw_patterns or not all(
            isinstance(p, str) and p.strip() for p in raw_patterns):
        problems.append(_problem(
            "P_SEALED_MISSING", "sealed.path_patterns must be a non-empty list of strings"))
        raw_patterns = []
    for pattern in raw_patterns:
        patterns.append(_normalise_path_text(pattern))
    if sealed.get("read_sealed_results") is not False:
        problems.append(_problem(
            "P_SEALED_READ_ALLOWED", "sealed.read_sealed_results must be false"))
    if not isinstance(sealed.get("present_on_host"), bool):
        problems.append(_problem(
            "P_SEALED_SHAPE",
            "sealed.present_on_host must state whether any sealed asset exists here"))
    if sealed.get("present_on_host") is True:
        problems.append(_problem(
            "SEALED_PRESENT",
            "sealed assets are declared present on this host; the split validator refuses "
            "to certify a registry while sealed material is reachable"))
    absent = sealed.get("verified_absent")
    if not isinstance(absent, Mapping) or not isinstance(absent.get("evidence_paths"), list) \
            or not absent["evidence_paths"]:
        problems.append(_problem(
            "P_SEALED_EVIDENCE",
            "sealed.verified_absent.evidence_paths must record how absence was verified"))
    return patterns, problems


def _check_sealed_references(cfg: Mapping[str, Any], patterns: Sequence[str]) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    if not patterns:
        return problems
    sealed_block = cfg.get("sealed") if isinstance(cfg, Mapping) else None
    for text in _iter_strings({k: v for k, v in cfg.items() if k != "sealed"}):
        normalised = _normalise_path_text(text)
        if not normalised:
            continue
        for pattern in patterns:
            if pattern and pattern in normalised:
                problems.append(_problem(
                    "SEALED_REFERENCED",
                    f"the split config references sealed material ({pattern!r} in {text!r}); "
                    "sealed assets and sealed results must never appear in a split registry",
                    pattern=pattern, reference=text))
    return problems


def _check_confirm_batches(
    batches: Any, reserved_ranges: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    if batches is None:
        return problems
    if not isinstance(batches, list):
        return [_problem("P_CONFIRM_BATCHES", "confirm_batches must be a list")]
    protected = {(r["seed_min"], r["seed_max"]): r for r in reserved_ranges
                 if r["role"] == "protected"}
    for entry in batches:
        if not isinstance(entry, Mapping):
            problems.append(_problem("P_CONFIRM_BATCHES", "confirm batch must be an object"))
            continue
        lo, hi = entry.get("seed_min"), entry.get("seed_max")
        status = entry.get("status")
        if not _is_int(lo) or not _is_int(hi) or lo > hi:
            problems.append(_problem("P_CONFIRM_BATCHES",
                                     "confirm batch needs integer seed_min/seed_max"))
            continue
        if status not in ("spent", "unused"):
            problems.append(_problem("P_CONFIRM_BATCHES",
                                     f"confirm batch {lo}-{hi} needs status spent|unused"))
            continue
        if BANNED_SEED_MIN <= lo <= BANNED_SEED_MAX or BANNED_SEED_MIN <= hi <= BANNED_SEED_MAX:
            problems.append(_problem(
                "S_BANNED_SEED",
                f"confirm batch {lo}-{hi} overlaps the banned training range"))
        matching = protected.get((lo, hi))
        if matching is None:
            problems.append(_problem(
                "P_CONFIRM_BATCHES",
                f"confirm batch {lo}-{hi} is not one of the declared protected ranges"))
        elif matching["usage"] == "unused" and status == "spent":
            problems.append(_problem(
                "P_CONFIRM_BATCHES",
                f"confirm batch {lo}-{hi} claims spent while the protected range "
                f"{matching['name']} is declared unused"))
        elif matching["usage"] == "spent" and status == "unused":
            problems.append(_problem(
                "P_CONFIRM_BATCHES",
                f"confirm batch {lo}-{hi} claims unused while the protected range "
                f"{matching['name']} is declared spent"))
    return problems


# ---------------------------------------------------------------------------
# 8. Report assembly
# ---------------------------------------------------------------------------


def _statistics(
    *,
    families: Mapping[str, Mapping[str, Any]],
    sources: Mapping[tuple[str, str], Mapping[str, Any]],
    source_kinds: set[str],
    holdouts: Mapping[str, Any],
    ancestors: Mapping[str, Mapping[str, Any]],
    ancestor_groups: Mapping[str, str],
    split_members: Mapping[str, Sequence[Mapping[str, Any]]],
    problems: Sequence[Mapping[str, Any]],
    reserved_ranges: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    per_split: dict[str, Any] = {}
    all_seeds: set[int] = set()
    for split, records in split_members.items():
        groups: dict[str, list[str]] = {}
        seeds: set[int] = set()
        fams: set[str] = set()
        for record in records:
            gkey = ancestor_groups.get(record["ancestor_id"], record["ancestor_id"])
            groups.setdefault(gkey, []).append(record["ancestor_id"])
            seeds.add(record["seed"])
            fams.add(record["task_family"])
        collapsed = {g: ids for g, ids in groups.items() if len(ids) > 1}
        per_split[split] = {
            "declared_ancestors": sum(len(ids) for ids in groups.values()),
            "independent_ancestors": len(groups),
            "collapsed_groups": sorted(collapsed),
            "seeds": sorted(seeds),
            "seed_count": len(seeds),
            "families": sorted(fams),
        }
        if split == "train":
            all_seeds |= seeds
    codes = sorted({p["code"] for p in problems})
    touched_links = sum(len(s.get("ancestor_ids", []) or []) for s in sources.values())
    touched_unique: set[str] = set()
    for source in sources.values():
        touched_unique.update(source.get("ancestor_ids", []) or [])
    banned_hits = sorted(s for s in all_seeds if BANNED_SEED_MIN <= s <= BANNED_SEED_MAX)
    protected_hits: list[int] = []
    for rng in reserved_ranges:
        if rng.get("role") != "protected":
            continue
        protected_hits.extend(s for s in all_seeds
                              if rng["seed_min"] <= s <= rng["seed_max"])
    return {
        "splits": per_split,
        "ancestors_total": len(ancestors),
        "families_registered": len(families),
        "families_touched_by_training": sorted(
            fid for fid, fam in families.items() if fam.get("status") == "touched_by_training"),
        "families_instrument_only": sorted(
            fid for fid, fam in families.items() if fam.get("status") == "instrument_only"),
        "holdout_families_frozen": [e["family_id"] for e in holdouts.get("frozen", [])],
        "holdout_families_reserved": [e["family_id"] for e in holdouts.get("reserved", [])],
        "pretraining_sources": len(sources),
        "pretraining_sources_family_level": sorted(
            f"{k}/{sid}" for (k, sid), s in sources.items()
            if s.get("ancestor_enumeration") == "family_level"),
        "pretraining_kinds_registered": sorted(source_kinds),
        "pretraining_kinds_missing": [k for k in PRETRAINING_SOURCE_KINDS
                                      if k not in source_kinds],
        "pretraining_touched_ancestor_links": touched_links,
        "pretraining_touched_ancestors_unique": len(touched_unique),
        "seed_range_check": {
            "banned_seed_min": BANNED_SEED_MIN,
            "banned_seed_max": BANNED_SEED_MAX,
            "train_seed_min": min(all_seeds) if all_seeds else None,
            "train_seed_max": max(all_seeds) if all_seeds else None,
            "train_seed_count": len(all_seeds),
            "banned_hits": banned_hits,
            "protected_hits": sorted(set(protected_hits)),
            "protected_ranges": [r["name"] for r in reserved_ranges if r.get("role") == "protected"],
        },
        "problem_count": len(problems),
        "problem_codes": codes,
    }


def _report(
    *,
    passed: bool,
    problems: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    config: Mapping[str, Any],
    checks: Sequence[str],
    not_checked: Sequence[str],
    digest: Mapping[str, Any] | None = None,
    degradations: Sequence[Mapping[str, Any]] = (),
    mode: str = "strict",
) -> dict[str, Any]:
    if digest is None:
        digest = stable_registry_digest(dict(config)) if config else {
            "sha256": None,
            "digest_scope": DIGEST_SCOPE_TEXT,
            "canonicalization": TEXT_CANONICALIZATION,
            "canonical_bytes": 0,
        }
    report = {
        "schema": REPORT_SCHEMA,
        "mode": mode,
        "config_schema": config.get("schema") if isinstance(config, Mapping) else None,
        "experiment_id": config.get("experiment_id") if isinstance(config, Mapping) else None,
        "spec_revision": config.get("spec_revision") if isinstance(config, Mapping) else None,
        "frozen_at": config.get("frozen_at") if isinstance(config, Mapping) else None,
        "passed": bool(passed),
        "problems": list(problems),
        "degradations": list(degradations),
        "statistics": dict(statistics),
        "checks_performed": list(checks),
        "not_checked": list(not_checked),
        "split_registry_sha": digest["sha256"],
        "split_registry_sha_scope": digest["digest_scope"],
        "split_registry_sha_canonicalization": digest["canonicalization"],
    }
    report["exit_code"] = exit_code_for(report)
    return report


def exit_code_for(report: Mapping[str, Any]) -> int:
    """CLI exit code for a validation report.

    ``0`` — the requested check is satisfied; ``2`` — the input/protocol is invalid
    (``EXECUTION_PLAN.md`` §4).  A split registry never returns ``3``: it measures
    nothing, it only decides whether the frozen lineage is admissible.
    """

    return 0 if report.get("passed") is True else 2
