"""Fail-closed ``RESET`` state-hash guard (EXP-P0-071).

EXP-P0-070 run c measured a real, silent divergence: 1 of 18 engine launches
(≈5.6 %) answered ``RESET 0 1 0 1 0`` with hash ``795b9441ed0c8056`` while the
other five replied ``7f04c7361749ed9f``, and the engine still reported
``ready=1``.  That divergence is visible *before any* ``STEP``, so no downstream
check can see it; a training or benchmark launch must therefore verify the hash
the engine itself reports and refuse to continue when it does not match the
expected start state.

This module is deliberately narrow: it compares the ``hash`` field of one real
``RESET`` reply against an expected value and raises on any difference (or when
the reply carries no verifiable hash at all).  It does **not** explain why a
launch diverges, restart the engine, or relax any other check — a guard that
detects and rejects an unverified start state is not a fix for the root cause.

Expected-hash precedence (highest first), exactly as documented in the CLI:

1. an explicit ``--expect-reset-hash <hex>`` argument (``source="cli"``);
2. the ``V400_EXPECT_RESET_HASH`` environment variable (``source="env"``);
3. the first successfully verified reset hash of the same run
   (``source="first-observed"``), which makes several instances of one batch
   agree with each other.  With this source alone the guard cannot detect a
   divergence of the *first* instance; pinning a value with (1) or (2) is what
   makes a launch fail closed against an externally known start state.
"""
from __future__ import annotations

import os
from typing import Mapping

#: Environment variable that supplies the expected reset hash (precedence 2).
RESET_HASH_ENV = "V400_EXPECT_RESET_HASH"

SOURCE_CLI = "cli"
SOURCE_ENV = "env"
SOURCE_FIRST = "first-observed"

#: One retry is the default: a divergence is worth one repeat, never silent luck.
DEFAULT_RESET_RETRIES = 1
#: Upper bound on retries so a guard can never turn into an unbounded relaunch loop.
MAX_RESET_RETRIES = 5

_HEX = frozenset("0123456789abcdef")


class ResetHashError(RuntimeError):
    """The start state could not be verified; the episode must not be recorded."""


class ResetHashUnavailable(ResetHashError):
    """The ``RESET`` reply carries no usable ``hash`` field to verify."""

    def __init__(self, message: str, *, stage_start: str, raw=None) -> None:
        self.stage_start = stage_start
        self.raw = raw
        super().__init__(message)

    def as_record(self) -> dict:
        """The unverifiable reply as JSON-safe fields; nothing is measured."""
        return {
            "reset_guard_failed": True,
            "reset_hash_mismatch": False,
            "reset_hash_observed": None if self.raw is None else str(self.raw),
            "reset_stage_start": self.stage_start,
        }


class ResetHashMismatch(ResetHashError):
    """The engine's own reset hash disagrees with the expected start state."""

    def __init__(self, *, expected: str, observed: str, stage_start: str,
                 source: str, attempts: int, observed_hashes: list[str]) -> None:
        self.expected = expected
        self.observed = observed
        self.stage_start = stage_start
        self.source = source
        self.attempts = int(attempts)
        self.observed_hashes = list(observed_hashes)
        message = (
            f"RESET state-hash guard diverged after {self.attempts} attempt(s): "
            f"expected hash={expected} (source={source}), observed hash={observed}, "
            f"stage_start={stage_start}; the engine reported ready=1 from an "
            "unverified start state, so no bench row is written and the episode is refused"
        )
        if len(self.observed_hashes) > 1:
            message += f"; observed hashes across attempts={self.observed_hashes}"
        super().__init__(message)

    def as_record(self) -> dict:
        """The mismatch as JSON-safe fields; no measured number is invented."""
        return {
            "reset_guard_failed": True,
            "reset_hash_mismatch": True,
            "reset_hash_expected": self.expected,
            "reset_hash_observed": self.observed,
            "reset_hash_source": self.source,
            "reset_stage_start": self.stage_start,
            "reset_attempts": self.attempts,
            "reset_observed_hashes": self.observed_hashes,
        }


def normalize_reset_hash(value) -> str:
    """Return *value* as a lowercase hex string, or raise ``ValueError``.

    The engine prints the 64-bit state hash as lowercase hex (``%016llx``).  Any
    other content is rejected instead of coerced: a guard that accepts a
    non-hex placeholder such as ``"unknown"`` or ``"ready"`` could compare two
    unverifiable values and call them equal.
    """
    if not isinstance(value, str):
        raise ValueError(f"reset hash must be a hex string, got {type(value).__name__}")
    text = value.strip().lower()
    if not text:
        raise ValueError("reset hash is empty")
    if any(character not in _HEX for character in text):
        raise ValueError(f"reset hash is not hexadecimal: {value!r}")
    return text


def stage_start_command(*, stage: int, difficulty: int, team: int, seed: int,
                        practice: bool) -> str:
    """The exact ``RESET`` line a stage start was requested with.

    Kept identical to ``HarnessClient.reset``'s wire format so an error message
    names the parameters that were actually sent, never a reconstruction.
    """
    return f"RESET {int(stage)} {int(difficulty)} {int(team)} {int(seed)} {1 if practice else 0}"


def resolve_expected_reset_hash(explicit: str | None = None,
                                environ: Mapping[str, str] | None = None
                                ) -> tuple[str | None, str | None]:
    """Resolve the expected hash and where it came from (CLI > env > none).

    Returns ``(None, None)`` when neither source is configured; the guard then
    pins the first successful reset it verifies (precedence 3).  A value that is
    present but not hexadecimal raises ``ValueError``: a malformed expectation
    must fail the launch instead of being silently ignored.
    """
    if explicit is not None:
        return normalize_reset_hash(explicit), SOURCE_CLI
    env = os.environ if environ is None else environ
    raw = env.get(RESET_HASH_ENV)
    if raw is None or not str(raw).strip():
        return None, None
    try:
        return normalize_reset_hash(raw), SOURCE_ENV
    except ValueError as exc:
        raise ValueError(f"{RESET_HASH_ENV}={raw!r} is not a usable reset hash: {exc}") from exc


def guard_from_config(*, explicit: str | None = None,
                      environ: Mapping[str, str] | None = None,
                      retries: int | None = None) -> "ResetHashGuard":
    """Build the guard the way the CLI does (explicit argument then environment)."""
    expected, source = resolve_expected_reset_hash(explicit, environ)
    return ResetHashGuard(expected=expected, source=source,
                          retries=DEFAULT_RESET_RETRIES if retries is None else retries)


class ResetHashGuard:
    """Verify the engine's own ``RESET`` hash against one expected start state.

    One guard instance is meant to be shared by every episode of a run, which is
    what makes precedence 3 ("the first successful reset of this run") apply to a
    whole batch instead of to a single process.
    """

    def __init__(self, *, expected: str | None = None, source: str | None = None,
                 retries: int = DEFAULT_RESET_RETRIES) -> None:
        if expected is None and source not in (None, SOURCE_FIRST):
            raise ValueError(f"an expected hash source needs an expected hash: {source!r}")
        self._expected = normalize_reset_hash(expected) if expected is not None else None
        self._source = source if self._expected is not None else None
        self.retries = _check_retries(retries)
        self.observed: list[str] = []
        self.attempts = 0

    # -- inspection ------------------------------------------------------
    @property
    def expected(self) -> str | None:
        return self._expected

    @property
    def source(self) -> str | None:
        return self._source

    @property
    def pinned(self) -> bool:
        return self._expected is not None

    @property
    def attempts_allowed(self) -> int:
        return 1 + self.retries

    def as_record(self) -> dict:
        """Provenance worth storing next to a measured episode."""
        return {
            "reset_hash_expected": self._expected,
            "reset_hash_source": self._source,
            "reset_attempts": self.attempts,
            "reset_observed_hashes": list(self.observed),
        }

    # -- verification ----------------------------------------------------
    def check(self, reply: Mapping[str, str], *, stage_start: str) -> str:
        """Verify one real ``RESET`` reply and return its normalized hash.

        Raises ``ResetHashUnavailable`` when the reply has no verifiable hash and
        ``ResetHashMismatch`` when it disagrees with the expected start state.
        Both are fail-closed: the caller must not run or record the episode.
        """
        raw = reply.get("hash")
        if raw is None:
            raise ResetHashUnavailable(
                f"RESET reply carries no hash to verify (fields={sorted(reply)}), "
                f"stage_start={stage_start}; refusing an unverified start state",
                stage_start=stage_start,
            )
        try:
            observed = normalize_reset_hash(raw)
        except ValueError as exc:
            raise ResetHashUnavailable(
                f"RESET reply hash is unusable ({exc}), stage_start={stage_start}; "
                "refusing an unverified start state",
                stage_start=stage_start, raw=raw,
            ) from exc
        self.attempts += 1
        self.observed.append(observed)
        if self._expected is None:
            # Precedence 3: the first successful reset of this run becomes the
            # expectation for every later reset of the same run.
            self._expected = observed
            self._source = SOURCE_FIRST
            return observed
        if observed != self._expected:
            raise ResetHashMismatch(
                expected=self._expected, observed=observed, stage_start=stage_start,
                source=self._source or "unknown", attempts=self.attempts,
                observed_hashes=list(self.observed),
            )
        return observed


def _check_retries(retries: int) -> int:
    try:
        value = int(retries)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reset retries must be an integer, got {retries!r}") from exc
    if not 0 <= value <= MAX_RESET_RETRIES:
        raise ValueError(f"reset retries must be in [0, {MAX_RESET_RETRIES}], got {value}")
    return value
