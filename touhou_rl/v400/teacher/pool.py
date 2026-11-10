"""Reusable recording templates: what ``PARK``/``SPAWN`` actually allow (P1-WP?).

The expensive phase of a teacher decision is the *restore*: recording walks the
authoritative shadow ``H`` frames forward, and the only way back to the decision
point is a new process plus ``RESET`` (once per process, EXP-P1-003 §2) plus a
replay of the committed masks.  The obvious fix is a reusable template: ``PARK``
one state and ``SPAWN`` a copy per decision.  This module is that template, with
its one hard constraint measured rather than assumed (EXP-P1-006 §4).

What ``PARK`` really is
-----------------------
``HandlePark`` forks the *parent* into a template child that sits in
``TemplateLoop`` (``fork.cpp:264``), and ``SPAWN`` forks *that template* into a
grandchild which re-enters the command loop with private pipes
(``fork.cpp:188``).  Two consequences, both read out of the harness source and
observed on a real process:

* the template is a **one-way snapshot**: the parent keeps running forward and
  there is no protocol command that adopts the template's state back into the
  parent (no ``RESTORE``/``SWAP``), so a parked decision point cannot become the
  next decision's main process;
* a spawned copy is a **child**: on ``build/rl_hazard`` it answers
  ``HELLO/HASH/QUIT/DEBUG/FRAME/PING/STATE/STEP`` and refuses ``HAZARDS`` with
  ``ERR child-command``.  A child can therefore *reach* the next decision point
  (measured: 4 commit frames, then ``STATE`` reports the expected position) but
  it cannot record it.

So a recorder pool is **only** possible when the engine can answer
``CHILD <h> HAZARDS``.  This module refuses to pretend otherwise
(:class:`~touhou_rl.v400.teacher.hazard_tape.ChildHazardsUnavailable`) and
exposes the measured budget so the saving is a number, not a hope.

Measured on stage 1 normal seed 1 (EXP-P1-006, ``build/rl_hazard``): process
start + ``HELLO`` 1180.8 ms, ``RESET`` 415.3 ms, replay 19.9 ms per 200 frames
(100 decision slots), ``PARK`` + ``SPAWN`` 0.6 ms.  The replay is the only part
a template can remove, and it is ~0.5% of the phase — the 1.6 s floor is process
start + ``RESET``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from touhou_rl.v400.teacher.hazard_tape import ChildHazardsUnavailable


@dataclass
class RestoreBudget:
    """One measured restore, split into the parts a pool could and could not remove."""

    start_ms: float = 0.0
    reset_ms: float = 0.0
    replay_ms: float = 0.0
    park_spawn_ms: float = 0.0
    frames_replayed: int = 0

    @property
    def fixed_ms(self) -> float:
        """What every decision must pay no matter what: process start + ``RESET``."""
        return float(self.start_ms + self.reset_ms)

    @property
    def poolable_ms(self) -> float:
        """What a recorder template would remove from the critical path."""
        return float(self.replay_ms)

    @property
    def total_ms(self) -> float:
        return float(self.start_ms + self.reset_ms + self.replay_ms)

    def as_dict(self) -> dict[str, float]:
        return {
            "start_ms": round(self.start_ms, 3),
            "reset_ms": round(self.reset_ms, 3),
            "replay_ms": round(self.replay_ms, 3),
            "park_spawn_ms": round(self.park_spawn_ms, 3),
            "frames_replayed": int(self.frames_replayed),
            "fixed_ms": round(self.fixed_ms, 3),
            "poolable_ms": round(self.poolable_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


@dataclass
class RecorderPool:
    """A decision-point template that can hand out recorders — when the engine allows it.

    ``status`` is the honest answer for the current build: the probe's verdict
    plus the measured budget of the phase the pool is meant to replace.
    ``acquire`` raises instead of returning a main-process fallback, because that
    fallback would silently move the shadow and hide the ~1.6 s restore.
    """

    template: int | None = None
    budget: RestoreBudget = field(default_factory=RestoreBudget)
    _spawned: int = 0

    def status(self, engine) -> dict[str, object]:
        probe = getattr(engine, "probe_child_hazards", None)
        capability = probe() if callable(probe) else {
            "available": False,
            "reason": "engine has no probe_child_hazards; treating child hazards as unavailable",
        }
        return {
            "available": bool(capability.get("available")),
            "reason": str(capability.get("reason", "")),
            "template": None if self.template is None else int(self.template),
            "spawned": int(self._spawned),
            "budget": self.budget.as_dict(),
        }

    def acquire(self, engine) -> int:
        """A handle that can record the next decision; fail closed when it cannot."""
        status = self.status(engine)
        if not status["available"]:
            raise ChildHazardsUnavailable(
                "a PARKed template cannot produce a recorder on this engine build: "
                f"{status['reason']!r}; the pool would need CHILD <h> HAZARDS"
            )
        if self.template is None:
            self.template = engine.park()
        handle = engine.spawn(self.template)
        self._spawned += 1
        return handle


@dataclass
class RecordingPool(RecorderPool):
    """Alias kept readable at call sites: the pool's job is recording, not forking."""
