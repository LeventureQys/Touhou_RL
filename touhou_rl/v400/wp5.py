"""Fake/in-process WP5 contract harness.

This module is a deterministic, single-threaded state machine for contract tests.
It does not provide a real game process, rendering device, OS fork, cross-process
isolation, or pixel-level consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import multiprocessing
from typing import Any


class InvalidStateError(RuntimeError):
    """Raised when an operation is not valid for the process lifecycle state."""


@dataclass(frozen=True)
class ShadowState:
    """Immutable state exposed by the fake harness."""

    seed: int
    frame: int
    state_digest: str


class ShadowProcess:
    """A deterministic in-process stand-in for the WP5 process contract."""

    _CREATED = "created"
    _RUNNING = "running"
    _PARKED = "parked"
    _CLOSED = "closed"

    def __init__(self, seed: int):
        self._seed = seed
        self._state = ShadowState(seed, 0, self._digest(seed, 0, None, None))
        self._status = self._CREATED

    @staticmethod
    def _canonical(value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, default=repr)
        except (TypeError, ValueError):
            return repr(value)

    @classmethod
    def _digest(cls, seed: int, frame: int, previous: str | None,
                action: Any) -> str:
        payload = cls._canonical({"seed": seed, "frame": frame,
                                  "previous": previous, "action": action})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def state(self) -> ShadowState:
        return self.snapshot()

    @property
    def status(self) -> str:
        return self._status

    def _require(self, *allowed: str) -> None:
        if self._status not in allowed:
            raise InvalidStateError(
                f"operation is invalid while process is {self._status}")

    def start(self) -> ShadowState:
        self._require(self._CREATED)
        self._status = self._RUNNING
        return self._state

    def step(self, action: Any) -> ShadowState:
        self._require(self._RUNNING)
        frame = self._state.frame + 1
        self._state = ShadowState(
            self._seed, frame,
            self._digest(self._seed, frame, self._state.state_digest, action),
        )
        return self._state

    def snapshot(self) -> ShadowState:
        self._require(self._CREATED, self._RUNNING, self._PARKED)
        return self._state

    def branch(self, snapshot: ShadowState | None = None) -> "ShadowProcess":
        self._require(self._CREATED, self._RUNNING, self._PARKED)
        source = self._state if snapshot is None else snapshot
        if not isinstance(source, ShadowState):
            raise TypeError("snapshot must be a ShadowState")
        child = object.__new__(ShadowProcess)
        child._seed = source.seed
        child._state = ShadowState(source.seed, source.frame, source.state_digest)
        child._status = self._status if self._status != self._CREATED else self._CREATED
        return child

    def spawn(self, snapshot: ShadowState | None = None) -> "ShadowProcess":
        """Create an independent in-process branch from this process."""
        return self.branch(snapshot)

    def park(self) -> ShadowState:
        self._require(self._RUNNING)
        self._status = self._PARKED
        return self._state

    def resume(self) -> ShadowState:
        self._require(self._PARKED)
        self._status = self._RUNNING
        return self._state

    def close(self) -> None:
        """Close the process; repeated close calls are intentionally harmless."""
        if self._status == self._CLOSED:
            return
        self._status = self._CLOSED




def shadow_worker(connection: Any) -> None:
    """Run a deterministic, device-free worker for multiprocessing smoke tests.

    This is only an in-process state machine hosted in a real Python child process;
    it is not a game fork and provides no rendering or pixel-equivalence guarantee.
    """
    try:
        request = connection.recv()
        seed = request["seed"]
        actions = request["actions"]
        process = ShadowProcess(seed)
        process.start()
        digests = [process.step(action).state_digest for action in actions]
        connection.send({"ok": True, "state_digests": digests})
    except Exception as exc:
        try:
            connection.send({"ok": False, "error": type(exc).__name__})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


__all__ = ["InvalidStateError", "ShadowProcess", "ShadowState", "shadow_worker"]
