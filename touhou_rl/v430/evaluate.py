"""Unified evaluation entry and isolated judge (v4.3.0 work package W01, slice B).

``DEVELOPMENT_GUIDE`` §5.2/§10, ``EXECUTION_PLAN.md`` §4 and
``PLAN_SPEC.evaluation`` pin five properties this module exists to enforce:

* **one entry, sealed input.**  The only authoritative input is a per-episode
  record file (or its content hash).  A hand-typed ``--mean`` / ``--clears`` is
  *refused*, not ignored: :data:`MANUAL_AGGREGATE_KEYS` are rejected at the top
  level of a batch document and inside every episode record, because a number
  somebody typed cannot be re-derived from the raw runs and a capability claim
  has to be.
* **fail-closed rejections.**  Empty result set, missing field, zero
  denominator, digest mismatch and any record flagged invalid all produce
  ``execution_status='invalid'`` with exit 2 -- never a "gate passed".  There is
  no skip path and no partial-success path.
* **Wilson interval, point estimate and denominator.**  :func:`wilson_interval`
  is the v4.2.0 formula (Wilson score, not the normal approximation), reported
  together with the point estimate and the *actual* number of episodes, so a
  reader cannot see a rate without its denominator.
* **four result fields, four exit codes.**  Every result carries
  ``execution_status / protocol_validity / research_result / capability_gate``
  with ``capability_gate`` defaulting to :data:`not_run`, and the process exit
  code is strictly ``0 / 2 / 3 / 4``.  :func:`write_result` refuses to write a
  result that fails :func:`touhou_rl.v430.contracts.assert_result_schema`, so a
  caller cannot construct a result, skip the check and write it anyway (the
  ``capability_gate`` guard lives in ``assert_result_schema``; a write path that
  skips it would bypass the guard).
* **judge isolation.**  Statistics are computed from the sealed bytes only, and
  this module's dependency graph is asserted: it imports the standard library
  plus :mod:`touhou_rl.v430.contracts` and **nothing else** -- no training, no
  teacher, no engine, no action compiler.  :func:`assert_judge_isolation`
  re-derives that from this file's own source, so a future import of
  ``touhou_rl.v430.rl`` fails the check instead of quietly coupling the judge to
  what it is judging.

Digest scopes (``evidence/W00/SHA_LINE_ENDING_FINDING.md`` §6)
-------------------------------------------------------------
Every identity this module produces carries an explicit ``digest_scope``:

* :data:`DIGEST_SCOPE_TEXT` (``"text/lf-normalized"``) for text products
  (configs, JSON evidence, script source).  The bytes are ``CRLF -> LF``
  normalised **before** hashing, so the digest is independent of
  ``core.autocrlf`` and of the checkout's line endings.
* :data:`DIGEST_SCOPE_BINARY` (``"binary/raw"``) for binary products
  (checkpoints, ``.dat``, PNG, engine images): raw bytes, no normalisation.

:func:`check_identity` refuses an identity field that has a ``sha256`` but no
``digest_scope`` (:class:`MissingDigestScopeError`), and :func:`verify_identity`
raises :class:`DigestScopeMismatch` when the content is identical but the scope
differs -- a different failure *message* from :class:`DigestMismatch` (the content
really changed), because conflating the two is exactly the defect that made 96 of
the 101 historical source SHAs unverifiable.

Everything here is standard-library only and reads only the paths it is given.
It starts no engine, loads no model and writes nothing except through
:func:`write_result`.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import touhou_rl.v430.contracts as C

__all__ = [
    # schema / vocabularies
    "RESULT_SCHEMA",
    "BATCH_SCHEMA",
    "OUTCOMES",
    "SUCCESS_OUTCOMES",
    "SPLITS",
    "EPISODE_REQUIRED_FIELDS",
    "EPISODE_OPTIONAL_FIELDS",
    "MANUAL_AGGREGATE_KEYS",
    "BATCH_REQUIRED_FIELDS",
    # digest scopes
    "DIGEST_SCOPE_TEXT",
    "DIGEST_SCOPE_BINARY",
    "DIGEST_SCOPES",
    "normalize_lf",
    "digest_bytes",
    "describe_bytes",
    "identify_bytes",
    "identify_file",
    "check_identity",
    "verify_identity",
    "assert_identity_scopes",
    "identity_fields",
    # statistics
    "DEFAULT_WILSON_Z",
    "wilson_interval",
    "z_for_confidence",
    "GateRequest",
    # judge isolation
    "JUDGE_IMPORTS_ALLOWED",
    "JUDGE_IMPORTS_FORBIDDEN",
    "module_imports",
    "assert_judge_isolation",
    # errors
    "EvaluationError",
    "InvalidInputError",
    "MissingFieldError",
    "EmptyResultSetError",
    "ZeroDenominatorError",
    "InvalidRecordError",
    "DigestMismatch",
    "DigestScopeMismatch",
    "MissingDigestScopeError",
    "ManualAggregateError",
    "ResultSchemaError",
    "OutputExistsError",
    # the entry points
    "EpisodeSource",
    "load_batch",
    "validate_episode",
    "evaluate",
    "write_result",
]


# ---------------------------------------------------------------------------
# schemas and closed vocabularies
# ---------------------------------------------------------------------------
#: Identity of a result written by this module.
RESULT_SCHEMA = "v430-evaluation/1"
#: Identity of the sealed per-episode input document.
BATCH_SCHEMA = "v430-episode-batch/1"

#: ``outcome`` of one episode.  ``invalid`` means the run itself is not
#: admissible evidence (a protocol fault); one such record poisons the batch.
OUTCOMES: tuple[str, ...] = (
    "clear", "game_over", "task_failure", "aborted", "invalid")
#: The only outcome that counts as one success.
SUCCESS_OUTCOMES: tuple[str, ...] = ("clear",)
#: Which split an episode belongs to.  A batch is one split (see
#: :func:`evaluate`): mixing dev and confirm episodes in one denominator would
#: silently turn a development number into a confirmation number.
SPLITS: tuple[str, ...] = ("train", "dev", "confirm", "probe")

EPISODE_REQUIRED_FIELDS: tuple[str, ...] = (
    "episode_id", "ancestor_id", "seed", "split", "outcome",
    "protocol_validity", "policy_sha256", "frames",
)
EPISODE_OPTIONAL_FIELDS: tuple[str, ...] = (
    "success", "invalid", "checkpoint_sha256", "started_at", "finished_at",
    "notes", "stop_reason", "wall_s", "deaths", "stage_reached",
)
EPISODE_FIELDS: tuple[str, ...] = EPISODE_REQUIRED_FIELDS + EPISODE_OPTIONAL_FIELDS

BATCH_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema", "experiment_id", "run_id", "policy_sha256", "split", "episodes")
BATCH_OPTIONAL_FIELDS: tuple[str, ...] = (
    "records", "checkpoint_sha256", "frozen_at", "started_at", "finished_at",
    "planned_episodes", "incomplete", "notes", "mode", "spec_revision",
    "evaluation_episodes",
)

#: Names a *hand-written* aggregate would use.  They are refused in the input:
#: the judge recomputes every count from the episode records, so a typed ``mean``
#: or ``clears`` is at best redundant and at worst a fabricated capability claim.
MANUAL_AGGREGATE_KEYS: tuple[str, ...] = (
    "mean", "clears", "successes", "success_rate", "clear_rate", "rate",
    "point_estimate", "total", "n_episodes", "total_episodes", "passed",
    "pass", "capability_gate", "research_result", "gate_passed",
)


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
class EvaluationError(RuntimeError):
    """Base class of every refusal raised by this module.

    Each subclass below is a *different* reason, so a caller (and an evidence
    reader) can tell "the content changed" from "the scope changed" from "the
    batch is empty".  None of them is caught inside :func:`evaluate` and turned
    into a pass.
    """

    kind = "evaluation_error"


class InvalidInputError(EvaluationError):
    """The input document is not a usable per-episode batch."""

    kind = "invalid_input"


class MissingFieldError(InvalidInputError):
    """A required field is absent."""

    kind = "missing_field"


class EmptyResultSetError(InvalidInputError):
    """The batch contains no episode record at all."""

    kind = "empty_result_set"


class ZeroDenominatorError(InvalidInputError):
    """The denominator of the requested statistic is zero."""

    kind = "zero_denominator"


class InvalidRecordError(InvalidInputError):
    """One episode (or the whole batch) is flagged invalid."""

    kind = "invalid_record"


class ManualAggregateError(InvalidInputError):
    """A hand-written aggregate was supplied as if it were authoritative data."""

    kind = "manual_aggregate"


class MissingDigestScopeError(EvaluationError):
    """An identity field carries a ``sha256`` but no ``digest_scope``."""

    kind = "missing_digest_scope"


class DigestMismatch(EvaluationError):
    """The content under the declared scope really changed."""

    kind = "content_mismatch"


class DigestScopeMismatch(EvaluationError):
    """The bytes are identical, but the digest was taken under another scope."""

    kind = "scope_mismatch"


class ResultSchemaError(EvaluationError):
    """The result would not pass ``contracts.assert_result_schema``; not written."""

    kind = "result_schema"


class OutputExistsError(EvaluationError):
    """The output path already exists; results are never overwritten."""

    kind = "output_exists"


# ---------------------------------------------------------------------------
# digest scopes (CRLF -> LF for text, raw for binary)
# ---------------------------------------------------------------------------
#: Text products are hashed after ``CRLF -> LF`` normalisation.
DIGEST_SCOPE_TEXT = "text/lf-normalized"
#: Binary products are hashed exactly as stored.
DIGEST_SCOPE_BINARY = "binary/raw"
#: Every identity must declare one of these.  There is no default.
DIGEST_SCOPES: tuple[str, ...] = (DIGEST_SCOPE_TEXT, DIGEST_SCOPE_BINARY)


def normalize_lf(payload: bytes) -> bytes:
    """``CRLF -> LF``.  The only normalisation v4.3.0 identities apply."""

    if not isinstance(payload, (bytes, bytearray)):
        raise EvaluationError(
            f"normalize_lf expects bytes, got {type(payload).__name__}")
    return bytes(payload).replace(b"\r\n", b"\n")


def digest_bytes(payload: bytes, *, scope: str = DIGEST_SCOPE_TEXT) -> str:
    """sha256 of ``payload`` under an explicit ``scope``."""

    if scope not in DIGEST_SCOPES:
        raise EvaluationError(
            f"digest scope {scope!r} not in {list(DIGEST_SCOPES)}; an identity "
            f"without a scope cannot be verified across machines")
    if scope == DIGEST_SCOPE_TEXT:
        return hashlib.sha256(normalize_lf(payload)).hexdigest()
    return hashlib.sha256(bytes(payload)).hexdigest()


def describe_bytes(payload: bytes) -> dict[str, Any]:
    """Both scope digests plus the line-ending census of one byte string."""

    crlf = bytes(payload).count(b"\r\n")
    lf = bytes(payload).count(b"\n") - crlf
    return {
        "bytes": len(payload),
        "sha256_text": digest_bytes(payload, scope=DIGEST_SCOPE_TEXT),
        "sha256_binary": digest_bytes(payload, scope=DIGEST_SCOPE_BINARY),
        "line_endings": {"crlf": crlf, "lf_only": lf,
                         "has_crlf": crlf > 0, "mixed": crlf > 0 and lf > 0},
    }


def identify_bytes(payload: bytes, *, path: str | None = None,
                   scope: str = DIGEST_SCOPE_TEXT) -> dict[str, Any]:
    """One identity record: path, sha256, scope, size and line-ending census.

    ``digest_scope`` is always written.  There is no code path here that emits a
    ``sha256`` without it.
    """

    described = describe_bytes(payload)
    identity: dict[str, Any] = {
        "path": path,
        "sha256": (described["sha256_text"] if scope == DIGEST_SCOPE_TEXT
                   else described["sha256_binary"]),
        "digest_scope": scope,
        "bytes": described["bytes"],
        "line_endings": described["line_endings"],
    }
    return identity


def identify_file(path: str | Path, *, scope: str = DIGEST_SCOPE_TEXT
                  ) -> dict[str, Any]:
    """Identity of one file on disk, hashed under ``scope``."""

    resolved = Path(path)
    if not resolved.is_file():
        raise InvalidInputError(f"file does not exist: {resolved}")
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise InvalidInputError(f"cannot read {resolved}: {exc}") from exc
    return identify_bytes(payload, path=str(resolved), scope=scope)


def check_identity(record: Any) -> dict[str, Any]:
    """Validate one identity record; refuse a digest without a scope.

    This is the enforcement point for the line-ending finding: the *shape* of the
    record is checked before any digest is compared, so "sha256 but no口径"
    cannot reach a comparison at all.
    """

    if not isinstance(record, Mapping):
        raise MissingDigestScopeError(
            f"identity must be a mapping, got {type(record).__name__}")
    if "sha256" not in record:
        raise MissingDigestScopeError(
            f"identity {dict(record)!r} carries no sha256; there is nothing to "
            f"verify")
    digest = record["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest):
        raise MissingDigestScopeError(
            f"identity sha256 {digest!r} is not 64 lowercase hex characters")
    scope = record.get("digest_scope")
    if scope is None:
        raise MissingDigestScopeError(
            f"identity for {record.get('path')!r} carries sha256 {digest} without "
            f"a digest_scope; every v4.3.0 identity must declare "
            f"{list(DIGEST_SCOPES)} (see SHA_LINE_ENDING_FINDING.md §6.3)")
    if scope not in DIGEST_SCOPES:
        raise MissingDigestScopeError(
            f"identity for {record.get('path')!r} declares digest_scope={scope!r}, "
            f"not one of {list(DIGEST_SCOPES)}")
    return dict(record)


def verify_identity(record: Any, payload: bytes, *,
                    what: str = "input") -> dict[str, Any]:
    """Compare ``record``'s digest with ``payload``, distinguishing the failures.

    * match under the declared scope -> returns the record;
    * match only under the *other* scope -> :class:`DigestScopeMismatch`
      ("the bytes are identical, the scope differs");
    * no match under either scope -> :class:`DigestMismatch` ("the content
      differs"), with both computed digests in the message so the reader can see
      that the line endings are not the explanation.

    The two cases must never be collapsed: the first says "re-pin the scope", the
    second says "the file changed".
    """

    checked = check_identity(record)
    declared = checked["digest_scope"]
    actual = digest_bytes(payload, scope=declared)
    if actual == checked["sha256"]:
        return checked
    other_scope = (DIGEST_SCOPE_BINARY if declared == DIGEST_SCOPE_TEXT
                   else DIGEST_SCOPE_TEXT)
    other = digest_bytes(payload, scope=other_scope)
    described = describe_bytes(payload)
    if other == checked["sha256"]:
        raise DigestScopeMismatch(
            f"{what} {checked.get('path')!r}: the bytes are identical but the "
            f"digest was taken under a different scope. The identity declares "
            f"digest_scope={declared!r} and its sha256 {checked['sha256']} is the "
            f"{other_scope!r} digest of this file; the same content hashed under "
            f"{declared!r} is {actual}. This is a 口径 (line-ending) difference, "
            f"not a content change: re-pin the identity under {declared!r} "
            f"(crlf={described['line_endings']['crlf']}, "
            f"lf_only={described['line_endings']['lf_only']}, "
            f"mixed={described['line_endings']['mixed']})")
    raise DigestMismatch(
        f"{what} {checked.get('path')!r}: the content differs. The identity "
        f"declares digest_scope={declared!r} with sha256 {checked['sha256']}; the "
        f"file hashes to {actual} under {declared!r} and to {other} under "
        f"{other_scope!r}, so neither scope reproduces the pin "
        f"(crlf={described['line_endings']['crlf']}, "
        f"lf_only={described['line_endings']['lf_only']}, mixed="
        f"{described['line_endings']['mixed']}). A line-ending change alone would "
        f"have matched under {other_scope!r}; it did not.")


def identity_fields(obj: Any, path: str = "", depth: int = 0
                    ) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Every mapping reachable from ``obj`` that looks like an identity record."""

    if depth > 8:
        return
    if isinstance(obj, Mapping):
        if "sha256" in obj:
            yield (path or "<root>", obj)
            return
        for name, child in obj.items():
            child_path = f"{path}.{name}" if path else str(name)
            yield from identity_fields(child, child_path, depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            yield from identity_fields(item, child_path, depth + 1)


def assert_identity_scopes(obj: Any) -> list[str]:
    """Every ``sha256`` in ``obj`` must sit next to a ``digest_scope``.

    Returns the list of identity paths checked (so a caller can report how many
    identities it actually validated instead of claiming compliance).
    """

    paths: list[str] = []
    for where, record in identity_fields(obj):
        check_identity(record)
        paths.append(where)
    return paths


# ---------------------------------------------------------------------------
# Wilson interval (the v4.2.0 formula, same口径)
# ---------------------------------------------------------------------------
#: ``PLAN_SPEC.evaluation.formal.wilson_confidence`` = 0.95; the v4.2.0 code used
#: the fixed normal quantile 1.96 for it and this module keeps that number so the
#: two implementations agree to the last digit.
DEFAULT_WILSON_Z = 1.96


def z_for_confidence(confidence: float) -> float:
    """The two-sided normal quantile for ``confidence`` (0.95 -> 1.959964...)."""

    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise EvaluationError(f"confidence must be a number, got {confidence!r}")
    if not 0.0 < float(confidence) < 1.0:
        raise EvaluationError(f"confidence {confidence} outside (0, 1)")
    return statistics.NormalDist().inv_cdf(0.5 + float(confidence) / 2.0)


def wilson_interval(successes: int, total: int, *,
                    z: float = DEFAULT_WILSON_Z) -> tuple[float, float]:
    """The Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because a small development run
    is exactly where the normal interval produces impossible bounds (negative
    lower limits, upper limits above 1) and, at ``0/n`` or ``n/n``, a zero-width
    interval that claims certainty it does not have.  The formula is the one
    :func:`touhou_rl.v420.eval.wilson_interval` uses, so a v4.3.0 result and a
    v4.2.0 result are comparable.

    ``total == 0`` raises :class:`ZeroDenominatorError` instead of returning
    ``(0.0, 0.0)``: an interval over no episodes is not an interval.
    """

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise EvaluationError(f"successes must be an int, got {successes!r}")
    if isinstance(total, bool) or not isinstance(total, int):
        raise EvaluationError(f"total must be an int, got {total!r}")
    if total <= 0:
        raise ZeroDenominatorError(
            f"total must be >= 1 to form an interval, got {total}; a rate with a "
            f"zero denominator is not a measurement")
    if not 0 <= successes <= total:
        raise EvaluationError(f"successes {successes} outside [0, {total}]")
    if z <= 0:
        raise EvaluationError(f"z must be > 0, got {z}")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = (z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
            / denominator)
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# the requested gate
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class GateRequest:
    """The measurement gate a batch is being judged against.

    Defaults are ``PLAN_SPEC.evaluation.formal``: 100 episodes, at least 50
    successes, 95% Wilson interval whose lower bound must clear 0.35.  Passing
    this gate is a *research result*; it never sets ``capability_gate``.
    """

    episodes: int
    min_successes: int
    wilson_confidence: float = 0.95
    wilson_min_lower_bound: float = 0.35
    wilson_z: float = DEFAULT_WILSON_Z
    require_exact_denominator: bool = True

    def __post_init__(self) -> None:
        for name, value in (("episodes", self.episodes),
                            ("min_successes", self.min_successes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise EvaluationError(f"gate.{name} must be a positive int, got {value!r}")
        if self.min_successes > self.episodes:
            raise EvaluationError(
                f"gate.min_successes {self.min_successes} exceeds "
                f"gate.episodes {self.episodes}")
        z_for_confidence(self.wilson_confidence)
        if abs(z_for_confidence(self.wilson_confidence) - self.wilson_z) > 0.01:
            raise EvaluationError(
                f"gate.wilson_confidence {self.wilson_confidence} implies "
                f"z={z_for_confidence(self.wilson_confidence):.6f} but "
                f"gate.wilson_z={self.wilson_z}; the declared confidence and the "
                f"quantile disagree")
        if not 0.0 <= self.wilson_min_lower_bound <= 1.0:
            raise EvaluationError(
                f"gate.wilson_min_lower_bound {self.wilson_min_lower_bound} "
                f"outside [0, 1]")

    @classmethod
    def from_plan_spec(cls, spec: Mapping[str, Any]) -> "GateRequest":
        """Read ``PLAN_SPEC.evaluation.formal`` into a gate request."""

        formal = (spec.get("evaluation") or {}).get("formal")
        if not isinstance(formal, Mapping):
            raise EvaluationError("PLAN_SPEC carries no evaluation.formal block")
        for key in ("episodes", "min_successes", "wilson_confidence",
                    "wilson_min_lower_bound"):
            if key not in formal:
                raise EvaluationError(f"PLAN_SPEC.evaluation.formal is missing {key!r}")
        return cls(episodes=int(formal["episodes"]),
                   min_successes=int(formal["min_successes"]),
                   wilson_confidence=float(formal["wilson_confidence"]),
                   wilson_min_lower_bound=float(formal["wilson_min_lower_bound"]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "min_successes": self.min_successes,
            "wilson_confidence": self.wilson_confidence,
            "wilson_z": self.wilson_z,
            "wilson_min_lower_bound": self.wilson_min_lower_bound,
            "require_exact_denominator": self.require_exact_denominator,
        }


# ---------------------------------------------------------------------------
# judge isolation
# ---------------------------------------------------------------------------
#: The only non-standard-library module this judge may import.
JUDGE_IMPORTS_ALLOWED: tuple[str, ...] = (
    "__future__", "ast", "dataclasses", "datetime", "hashlib", "json", "math",
    "pathlib", "statistics", "typing", "touhou_rl.v430.contracts",
)
#: Modules that would couple the judge to what it judges.
JUDGE_IMPORTS_FORBIDDEN: tuple[str, ...] = (
    "torch", "numpy", "touhou_rl.v430.actions", "touhou_rl.v430.session",
    "touhou_rl.v430.rl", "touhou_rl.v430.teacher", "touhou_rl.v430.curriculum",
    "touhou_rl.v430.learn", "touhou_rl.v430.models", "touhou_rl.v430.data",
    "touhou_rl.v430.observation", "touhou_rl.v430.tactics",
    "touhou_rl.v430.feedback", "touhou_rl.v430.deploy", "touhou_rl.v430.adapt",
    "touhou_rl.v430.transfer", "touhou_rl.v400.teacher", "touhou_rl.v400.student",
    "touhou_rl.v400.rl", "touhou_rl.v400.gym", "touhou_rl.v420",
)


def module_imports(path: str | Path | None = None) -> list[str]:
    """Every module named by an ``import`` statement in ``path`` (this file)."""

    source_path = Path(path) if path is not None else Path(__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
        elif isinstance(node, ast.Call):
            # ``importlib.import_module("...")`` / ``__import__("...")`` would
            # smuggle a dependency past the static scan; refuse the pattern.
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in ("__import__", "import_module"):
                raise EvaluationError(
                    f"{source_path} uses {name}(), a dynamic import the judge "
                    f"census cannot verify; the judge must have a static "
                    f"dependency graph")
    return sorted(set(modules))


def assert_judge_isolation(path: str | Path | None = None) -> dict[str, Any]:
    """Prove the judge imports no training/teacher/engine module.

    Two independent claims, both re-derived from this file's own source:

    * no import matches :data:`JUDGE_IMPORTS_FORBIDDEN` (nor a submodule of one);
    * every import is in :data:`JUDGE_IMPORTS_ALLOWED`, so a new dependency has
      to be added to the allow-list on purpose rather than appear by accident.
    """

    source_path = Path(path) if path is not None else Path(__file__)
    modules = module_imports(source_path)
    forbidden = [name for name in modules
                 if any(name == bad or name.startswith(bad + ".")
                        for bad in JUDGE_IMPORTS_FORBIDDEN)]
    unknown = [name for name in modules if name not in JUDGE_IMPORTS_ALLOWED]
    if forbidden or unknown:
        raise EvaluationError(
            f"{source_path} breaks judge isolation: forbidden={forbidden}, "
            f"outside the allow-list={unknown}. The judge may import the "
            f"standard library and touhou_rl.v430.contracts only; statistics "
            f"must be computable from the sealed input alone.")
    identity = identify_file(source_path, scope=DIGEST_SCOPE_TEXT)
    return {
        "isolated": True,
        "path": str(source_path),
        "imports": modules,
        "allowed": list(JUDGE_IMPORTS_ALLOWED),
        "forbidden_checked": list(JUDGE_IMPORTS_FORBIDDEN),
        "sha256": identity["sha256"],
        "digest_scope": identity["digest_scope"],
        "claim": (
            "the evaluation entry point does not import training, teacher, "
            "engine or action modules; it reads only the sealed per-episode "
            "records it is given"),
    }


# ---------------------------------------------------------------------------
# loading and validating the sealed input
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class EpisodeSource:
    """One sealed per-episode batch, optionally pinned by content hash."""

    path: Path
    expected: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path),
                "pinned": None if self.expected is None else dict(self.expected)}


def _coerce_source(value: Any) -> EpisodeSource:
    if isinstance(value, EpisodeSource):
        return value
    if isinstance(value, (str, Path)):
        return EpisodeSource(path=Path(value))
    raise InvalidInputError(
        f"a source must be a path or an EpisodeSource, got {type(value).__name__}")


def _coerce_sources(sources: Any) -> list[EpisodeSource]:
    """Normalise the ``sources`` argument into a list of sealed batches."""

    if sources is None:
        return []
    if isinstance(sources, (str, Path, EpisodeSource)):
        return [_coerce_source(sources)]
    if isinstance(sources, (list, tuple)):
        return [_coerce_source(item) for item in sources]
    raise InvalidInputError(
        f"sources must be a path, an EpisodeSource, or a list/tuple of them, got "
        f"{type(sources).__name__}; a hand-written aggregate is not a source")


def _pin_identity(pin: Any, *, path: Path) -> dict[str, Any] | None:
    """Normalise a pin argument into an identity record (or ``None``).

    A bare 64-hex string is accepted for a *text* product with the scope recorded
    explicitly as an assumption, because the alternative -- silently guessing the
    scope -- is the defect this module exists to prevent.
    """

    if pin is None:
        return None
    if isinstance(pin, str):
        return {"sha256": pin, "digest_scope": DIGEST_SCOPE_TEXT,
                "path": str(path),
                "scope_source": "assumed_text_scope_from_bare_digest"}
    if isinstance(pin, Mapping):
        record = dict(pin)
        record.setdefault("path", str(path))
        record.setdefault("scope_source", "declared")
        return check_identity(record)
    raise InvalidInputError(
        f"a pin must be a 64-hex string or an identity mapping, got "
        f"{type(pin).__name__}")


def _refuse_manual_aggregates(document: Mapping[str, Any], *, where: str) -> None:
    hits = [key for key in document if key in MANUAL_AGGREGATE_KEYS]
    if hits:
        raise ManualAggregateError(
            f"{where} carries hand-written aggregate fields {hits}; the judge "
            f"recomputes every count from the per-episode records and refuses a "
            f"typed mean/clears as authoritative data")


def validate_episode(record: Any, *, index: int) -> dict[str, Any]:
    """Validate one episode record and derive its ``success`` flag.

    ``success`` is *derived* from ``outcome`` (``outcome == 'clear'``).  If the
    record also states ``success``, the two must agree -- a record that says
    ``outcome='game_over', success=True`` is a hand edit, not a measurement.
    """

    if not isinstance(record, Mapping):
        raise InvalidRecordError(
            f"episode[{index}] must be a mapping, got {type(record).__name__}")
    _refuse_manual_aggregates(record, where=f"episode[{index}]")
    missing = [name for name in EPISODE_REQUIRED_FIELDS if name not in record]
    if missing:
        raise MissingFieldError(
            f"episode[{index}] is missing required fields {missing}; the "
            f"denominator cannot be audited without them")
    unknown = [name for name in record if name not in EPISODE_FIELDS]
    if unknown:
        raise InvalidRecordError(
            f"episode[{index}] carries unregistered fields {unknown}; known "
            f"fields are {list(EPISODE_FIELDS)}")

    episode_id = record["episode_id"]
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise InvalidRecordError(
            f"episode[{index}].episode_id must be a non-empty string")
    ancestor = record["ancestor_id"]
    if not isinstance(ancestor, str) or not ancestor.strip():
        raise InvalidRecordError(
            f"episode[{index}].ancestor_id must be a non-empty string; pooling "
            f"episodes without their ancestor is not allowed")
    seed = record["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InvalidRecordError(f"episode[{index}].seed must be an int, got {seed!r}")
    split = record["split"]
    if split not in SPLITS:
        raise InvalidRecordError(
            f"episode[{index}].split {split!r} not in {list(SPLITS)}")
    outcome = record["outcome"]
    if outcome not in OUTCOMES:
        raise InvalidRecordError(
            f"episode[{index}].outcome {outcome!r} not in {list(OUTCOMES)}")
    validity = record["protocol_validity"]
    if validity not in C.PROTOCOL_VALIDITY_VALUES:
        raise InvalidRecordError(
            f"episode[{index}].protocol_validity {validity!r} not in "
            f"{list(C.PROTOCOL_VALIDITY_VALUES)}")
    check_identity({"sha256": record["policy_sha256"],
                    "digest_scope": DIGEST_SCOPE_TEXT,
                    "path": f"episode[{index}].policy_sha256"})
    frames = record["frames"]
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
        raise InvalidRecordError(
            f"episode[{index}].frames must be a non-negative int, got {frames!r}")
    if "checkpoint_sha256" in record and record["checkpoint_sha256"] is not None:
        check_identity({"sha256": record["checkpoint_sha256"],
                        "digest_scope": DIGEST_SCOPE_BINARY,
                        "path": f"episode[{index}].checkpoint_sha256"})

    success = outcome in SUCCESS_OUTCOMES
    if "success" in record:
        declared = record["success"]
        if not isinstance(declared, bool):
            raise InvalidRecordError(
                f"episode[{index}].success must be a bool, got {declared!r}")
        if declared != success:
            raise InvalidRecordError(
                f"episode[{index}] declares success={declared} but "
                f"outcome={outcome!r}; success is derived from the outcome and a "
                f"disagreeing flag is a hand edit")
    invalid_flag = record.get("invalid", False)
    if not isinstance(invalid_flag, bool):
        raise InvalidRecordError(
            f"episode[{index}].invalid must be a bool, got {invalid_flag!r}")

    normalised = dict(record)
    normalised["success"] = success
    normalised["invalid"] = (invalid_flag or validity == "invalid"
                             or outcome == "invalid")
    return normalised


def load_batch(path: str | Path, *, expected: Any = None
               ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one sealed batch, verify its identity and validate its episodes.

    Returns ``(batch, identity)``.  Raises on a missing file, a wrong schema, a
    hand-written aggregate, a digest mismatch (content **or** scope) and on any
    invalid record -- there is no "best effort" mode.
    """

    resolved = Path(path)
    if not resolved.is_file():
        raise InvalidInputError(f"episode batch does not exist: {resolved}")
    try:
        payload = resolved.read_bytes()
    except OSError as exc:
        raise InvalidInputError(f"cannot read {resolved}: {exc}") from exc

    identity = identify_bytes(payload, path=str(resolved),
                              scope=DIGEST_SCOPE_TEXT)
    pin = _pin_identity(expected, path=resolved)
    if pin is not None:
        verify_identity(pin, payload, what="episode batch")
    try:
        document = json.loads(normalize_lf(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidInputError(f"{resolved} is not UTF-8 JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise InvalidInputError(f"{resolved} root must be a JSON object")

    _refuse_manual_aggregates(document, where=f"batch {resolved.name}")
    if document.get("schema") != BATCH_SCHEMA:
        raise InvalidInputError(
            f"{resolved} schema {document.get('schema')!r} is not "
            f"{BATCH_SCHEMA!r}; the judge reads the sealed per-episode format "
            f"only (a reused summary is a different schema and is not a "
            f"denominator)")
    unknown = [key for key in document
               if key not in BATCH_REQUIRED_FIELDS + BATCH_OPTIONAL_FIELDS]
    if unknown:
        raise InvalidInputError(
            f"{resolved} carries unregistered fields {unknown}; known fields are "
            f"{list(BATCH_REQUIRED_FIELDS + BATCH_OPTIONAL_FIELDS)}")
    missing = [key for key in BATCH_REQUIRED_FIELDS if key not in document]
    if missing:
        raise MissingFieldError(f"{resolved} is missing fields {missing}")

    if "records" in document and document["records"] is not None:
        declared = check_identity(document["records"])
        verify_identity(declared, payload,
                        what=f"batch {resolved.name} self-identity")

    if document["split"] not in SPLITS:
        raise InvalidInputError(
            f"{resolved} split {document['split']!r} not in {list(SPLITS)}")
    check_identity({"sha256": document["policy_sha256"],
                    "digest_scope": DIGEST_SCOPE_TEXT,
                    "path": f"{resolved.name}.policy_sha256"})

    episodes = document["episodes"]
    if not isinstance(episodes, list):
        raise InvalidInputError(
            f"{resolved} episodes must be a list, got {type(episodes).__name__}")
    if not episodes:
        raise EmptyResultSetError(
            f"{resolved} contains no episode record; an empty result set has no "
            f"denominator and cannot pass a gate")

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(episodes):
        episode = validate_episode(record, index=index)
        if episode["episode_id"] in seen:
            raise InvalidRecordError(
                f"{resolved} contains episode_id {episode['episode_id']!r} twice; "
                f"a duplicated episode would inflate the denominator")
        seen.add(episode["episode_id"])
        if episode["split"] != document["split"]:
            raise InvalidInputError(
                f"{resolved} is a {document['split']!r} batch but "
                f"episode {episode['episode_id']!r} claims split "
                f"{episode['split']!r}; a batch is one split, otherwise a "
                f"development number could be reported as a confirmation")
        validated.append(episode)

    incomplete = document.get("incomplete", False)
    if not isinstance(incomplete, bool):
        raise InvalidInputError(
            f"{resolved}.incomplete must be a bool, got {incomplete!r}")
    planned = document.get("planned_episodes")
    if planned is not None:
        if isinstance(planned, bool) or not isinstance(planned, int) or planned < 0:
            raise InvalidInputError(
                f"{resolved}.planned_episodes must be a non-negative int")

    batch = dict(document)
    batch["episodes"] = validated
    batch["_path"] = str(resolved)
    batch["_incomplete"] = incomplete or (
        planned is not None and planned > len(validated))
    batch["_planned"] = planned
    return batch, identity


# ---------------------------------------------------------------------------
# the judge
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _statistics(episodes: Sequence[Mapping[str, Any]], *, z: float,
                confidence: float) -> dict[str, Any]:
    denominator = len(episodes)
    if denominator <= 0:
        raise ZeroDenominatorError(
            "the denominator is zero; a batch with no episode cannot produce a "
            "rate or an interval")
    successes = sum(1 for episode in episodes if episode["success"])
    low, high = wilson_interval(successes, denominator, z=z)
    by_outcome: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_ancestor: dict[str, int] = {}
    for episode in episodes:
        by_outcome[episode["outcome"]] = by_outcome.get(episode["outcome"], 0) + 1
        by_split[episode["split"]] = by_split.get(episode["split"], 0) + 1
        by_ancestor[episode["ancestor_id"]] = (
            by_ancestor.get(episode["ancestor_id"], 0) + 1)
    return {
        "denominator": denominator,
        "denominator_note": (
            "every episode record in the sealed batch, including game_over, "
            "task_failure and aborted runs; invalid records invalidate the batch "
            "instead of being dropped"),
        "successes": successes,
        "failures": denominator - successes,
        "point_estimate": successes / denominator,
        "wilson": {
            "low": low,
            "high": high,
            "z": z,
            "confidence": confidence,
            "method": "wilson-score-interval",
            "note": "the v4.2.0 formula; no normal approximation",
        },
        "by_outcome": dict(sorted(by_outcome.items())),
        "by_split": dict(sorted(by_split.items())),
        "ancestors": len(by_ancestor),
        "episodes_per_ancestor": dict(sorted(by_ancestor.items())),
        "invalid_records": sum(1 for episode in episodes if episode["invalid"]),
        "frames": sum(int(episode["frames"]) for episode in episodes),
    }


def _gate_verdict(statistics: Mapping[str, Any], gate: GateRequest
                  ) -> tuple[str, list[str]]:
    """``(status, failures)`` for the requested gate; never promotes a capability."""

    failures: list[str] = []
    denominator = statistics["denominator"]
    if gate.require_exact_denominator and denominator != gate.episodes:
        failures.append(
            f"denominator {denominator} != requested {gate.episodes}")
    if statistics["successes"] < gate.min_successes:
        failures.append(
            f"successes {statistics['successes']} < required "
            f"{gate.min_successes}")
    low = statistics["wilson"]["low"]
    if low < gate.wilson_min_lower_bound:
        failures.append(
            f"wilson lower bound {low:.6f} < required "
            f"{gate.wilson_min_lower_bound}")
    return ("fail" if failures else "pass"), failures


def _empty_statistics() -> dict[str, Any]:
    return {
        "denominator": 0, "successes": 0, "failures": 0,
        "point_estimate": None, "wilson": None, "by_outcome": {},
        "by_split": {}, "ancestors": 0, "episodes_per_ancestor": {},
        "invalid_records": 0, "frames": 0,
        "denominator_note": "no valid measurement was taken",
    }


def evaluate(sources: Any, *, gate: GateRequest | Mapping[str, Any] | None = None,
             pins: Mapping[str, Any] | None = None,
             experiment_id: str | None = None,
             argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Judge a set of sealed per-episode batches and return a result document.

    The returned mapping always carries the four contract fields and an
    ``exit_code`` consistent with :func:`touhou_rl.v430.contracts.exit_code_for`;
    :func:`write_result` re-validates it with ``assert_result_schema`` before
    anything touches the filesystem.

    Input problems do **not** raise: they produce ``execution_status='invalid'``
    with exit 2 and ``research_result.passed = False``, so a caller that ignores
    the return value still cannot read a gate as passed.  ``capability_gate`` is
    always ``not_run`` -- this judge measures, it does not promote a capability.
    """

    started = _now()
    pin_map = dict(pins or {})

    isolation: dict[str, Any]
    try:
        isolation = assert_judge_isolation()
    except EvaluationError as exc:  # pragma: no cover - only if the file is edited
        isolation = {"isolated": False, "reason": str(exc)}

    batches: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    gate_request: GateRequest | None = None
    try:
        gate_request = _coerce_gate(gate)
    except EvaluationError as exc:
        rejections.append({"kind": exc.kind, "message": str(exc)})

    source_list: list[EpisodeSource] = []
    try:
        source_list = _coerce_sources(sources)
    except EvaluationError as exc:
        rejections.append({"kind": exc.kind, "message": str(exc)})

    if not source_list and not rejections:
        rejections.append({"kind": EmptyResultSetError.kind,
                           "message": "no episode batch was supplied; an empty "
                                      "result set has no denominator"})
    for source in source_list:
        pin = source.expected
        if pin is None and str(source.path) in pin_map:
            pin = pin_map[str(source.path)]
        try:
            batch, identity = load_batch(source.path, expected=pin)
        except EvaluationError as exc:
            rejections.append({"kind": exc.kind, "message": str(exc),
                               "path": str(source.path)})
            continue
        invalid = [episode for episode in batch["episodes"] if episode["invalid"]]
        if invalid:
            rejections.append({
                "kind": InvalidRecordError.kind,
                "path": str(source.path),
                "message": (
                    f"{len(invalid)} episode(s) are flagged invalid "
                    f"({[episode['episode_id'] for episode in invalid[:5]]}); "
                    f"invalid_policy is 'invalidate full batch; preserve records; "
                    f"fresh batch after repair'"),
            })
            continue
        identities.append({**identity, "status": "accepted"})
        batches.append(batch)

    if rejections:
        result = _result(
            execution_status="invalid", protocol_validity="invalid",
            research_result={
                "status": "invalid",
                "kind": "sealed_episode_batch",
                "requested_gate": None if gate_request is None
                else gate_request.as_dict(),
                "passed": False,
                "failures": [row["kind"] for row in rejections],
                "note": ("no gate is passed on rejected input; the records are "
                         "preserved in `rejections` for the repair record"),
            },
            statistics=_empty_statistics(), identities=identities,
            rejections=rejections, isolation=isolation,
            experiment_id=experiment_id, argv=argv, started_at=started)
        return _finish(result)

    episodes = [episode for batch in batches for episode in batch["episodes"]]
    splits = {batch["split"] for batch in batches}
    if len(splits) > 1:
        result = _result(
            execution_status="invalid", protocol_validity="invalid",
            research_result={
                "status": "invalid", "kind": "sealed_episode_batch",
                "requested_gate": gate_request.as_dict() if gate_request else None,
                "passed": False,
                "failures": [f"mixed splits {sorted(splits)} in one denominator"],
                "note": "one judgement, one split; dev and confirm may not be pooled",
            },
            statistics=_empty_statistics(), identities=identities,
            rejections=[{"kind": "mixed_split",
                         "message": f"batches carry splits {sorted(splits)}"}],
            isolation=isolation, experiment_id=experiment_id, argv=argv,
            started_at=started)
        return _finish(result)

    incomplete = any(batch["_incomplete"] for batch in batches)
    statistics_block = _statistics(
        episodes, z=(gate_request.wilson_z if gate_request else DEFAULT_WILSON_Z),
        confidence=(gate_request.wilson_confidence if gate_request else 0.95))

    if incomplete:
        result = _result(
            execution_status="incomplete", protocol_validity="valid",
            research_result={
                "status": "incomplete", "kind": "sealed_episode_batch",
                "requested_gate": gate_request.as_dict() if gate_request else None,
                "passed": False,
                "failures": ["the batch declares fewer planned episodes than it "
                             "contains; the measurement is incomplete"],
                "note": "exit 4: resources or interruption, not a gate verdict",
            },
            statistics=statistics_block, identities=identities,
            rejections=[], isolation=isolation, experiment_id=experiment_id,
            argv=argv, started_at=started)
        return _finish(result)

    if gate_request is None:
        result = _result(
            execution_status="ok", protocol_validity="valid",
            research_result={
                "status": "inconclusive", "kind": "sealed_episode_batch",
                "requested_gate": None, "passed": False,
                "failures": ["no gate was requested"],
                "note": ("a summary is not a gate verdict; supply a GateRequest "
                         "to judge the batch"),
            },
            statistics=statistics_block, identities=identities,
            rejections=[], isolation=isolation, experiment_id=experiment_id,
            argv=argv, started_at=started)
        return _finish(result)

    status, failures = _gate_verdict(statistics_block, gate_request)
    result = _result(
        execution_status=("ok" if status == "pass" else "gate_fail"),
        protocol_validity="valid",
        research_result={
            "status": status,
            "kind": "sealed_episode_batch",
            "requested_gate": gate_request.as_dict(),
            "passed": status == "pass",
            "failures": failures,
            "note": ("a passed research gate is a measurement, not a capability; "
                     "capability_gate stays not_run until an independent judge "
                     "settles it"),
        },
        statistics=statistics_block, identities=identities, rejections=[],
        isolation=isolation, experiment_id=experiment_id, argv=argv,
        started_at=started)
    return _finish(result)


def _coerce_gate(gate: Any) -> GateRequest | None:
    if gate is None:
        return None
    if isinstance(gate, GateRequest):
        return gate
    if isinstance(gate, Mapping):
        return GateRequest(**gate)
    raise EvaluationError(
        f"gate must be a GateRequest or a mapping, got {type(gate).__name__}")


def _result(*, execution_status: str, protocol_validity: str,
            research_result: Mapping[str, Any], statistics: Mapping[str, Any],
            identities: Sequence[Mapping[str, Any]],
            rejections: Sequence[Mapping[str, Any]], isolation: Mapping[str, Any],
            experiment_id: str | None, argv: Sequence[str] | None,
            started_at: str) -> dict[str, Any]:
    result = C.new_result(
        execution_status=execution_status,
        protocol_validity=protocol_validity,
        research_result=research_result,
        command_class="evaluation",
        schema=RESULT_SCHEMA,
        experiment_id=experiment_id,
        judge="touhou_rl.v430.evaluate",
        judge_isolation=dict(isolation),
        started_at=started_at,
        finished_at=_now(),
        argv=list(argv) if argv is not None else None,
        statistics=dict(statistics),
        inputs=[dict(record) for record in identities],
        rejections=[dict(row) for row in rejections],
    )
    result["remaining_gaps"] = _gaps(result)
    return result


def _gaps(result: Mapping[str, Any]) -> list[str]:
    gaps = [
        "capability_gate stays not_run: an evaluation measures a candidate, it "
        "does not settle a capability gate (that is W12/settlement).",
        "the sealed input is synthetic unless the batch names a real run; a unit "
        "test that drives this judge is not game evidence.",
    ]
    if result.get("execution_status") == "invalid":
        gaps.append("no measurement was taken: the inputs were rejected.")
    if result.get("execution_status") == "incomplete":
        gaps.append("the batch is incomplete; the measurement is not a verdict.")
    isolation = result.get("judge_isolation") or {}
    if not isolation.get("isolated", False):
        gaps.append("judge isolation could not be proven for this run.")
    return gaps


def _finish(result: dict[str, Any]) -> dict[str, Any]:
    """Set ``exit_code`` from the execution status and self-validate."""

    result["exit_code"] = C.exit_code_for(result)
    # Every identity anywhere in the result must carry a digest_scope; scanning
    # the whole document also covers ``judge_isolation.sha256``.
    result["identity_count"] = len(assert_identity_scopes(result))
    # Self-check: the same validation the write path enforces. Building a result
    # and writing it without this call is refused by write_result().
    C.assert_result_schema(result)
    return result


# ---------------------------------------------------------------------------
# writing (validated first, exclusive, never overwritten)
# ---------------------------------------------------------------------------
def write_result(result: Mapping[str, Any], path: str | Path) -> Path:
    """Validate ``result``, then create ``path`` exclusively.

    Order is the whole point of this function:

    1. :func:`touhou_rl.v430.contracts.assert_result_schema` -- a result that
       cannot prove its ``capability_gate`` (e.g. ``command_class='training'``
       with ``capability_gate='pass'``) raises :class:`ResultSchemaError` here and
       **no file and no directory is created**;
    2. only then is the target opened with mode ``"x"``, so an existing result is
       never overwritten.

    There is deliberately no ``force`` argument: a rejected result cannot be
    written by any caller.
    """

    try:
        C.assert_result_schema(dict(result))
    except C.ContractError as exc:
        raise ResultSchemaError(
            f"refusing to write a result that fails assert_result_schema: {exc}; "
            f"nothing was written and no gate is reported") from exc
    payload = dict(result)
    assert_identity_scopes(payload.get("inputs") or [])
    target = Path(path)
    if target.exists():
        raise OutputExistsError(f"refusing to overwrite existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    return target
