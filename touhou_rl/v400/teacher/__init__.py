"""P1 teacher package: hazard recording, geometric search, engine verification.

11-P1's teacher is the only consumer of ``BRANCH``/``PARK``/``SPAWN``/``CHILD`` in
the repository.  The public surface is small on purpose:

* :func:`touhou_rl.v400.teacher.decide.decide` — one decision, timed in four
  phases, returning a :class:`~touhou_rl.v400.teacher.label.TeacherLabel`;
* :func:`touhou_rl.v400.teacher.child_record.record_child_tape` — record in a fork
  so the authoritative process never moves (EXP-P1-007);
* :func:`touhou_rl.v400.teacher.hazard_tape.record` — one forward recording
  (``source="main"`` is the legacy path);
* :func:`touhou_rl.v400.teacher.search.beam_search` — ``geo_score[18]``;
* :func:`touhou_rl.v400.teacher.verify.verify_plans` — the engine's verdict.

The teacher is privileged and never ships: it reads the state export and forks
the engine, both of which are refused in zero-privilege mode.
"""

from __future__ import annotations

from touhou_rl.v400.teacher.child_record import ChildRecord, record_child_tape
from touhou_rl.v400.teacher.config import CONFIGS, FAST, FULL, TeacherConfig, Weights, get_config
from touhou_rl.v400.teacher.decide import death_bomb_feasible, death_bomb_window, decide
from touhou_rl.v400.teacher.engine import (
    EXPECTED_ENGINE_SHA256,
    EngineShaMismatch,
    check_engine_binary,
)
from touhou_rl.v400.teacher.hazard_tape import (
    ChildHazardsUnavailable,
    HazardExportUnavailable,
    HazardTape,
    record,
)
from touhou_rl.v400.teacher.label import TeacherLabel
from touhou_rl.v400.teacher.pool import RecorderPool, RestoreBudget
from touhou_rl.v400.teacher.search import Plan, SearchResult, beam_search, score_plan
from touhou_rl.v400.teacher.verify import Consistency, VerifiedPlan, verify_plans

__all__ = [
    "CONFIGS",
    "ChildHazardsUnavailable",
    "ChildRecord",
    "Consistency",
    "EXPECTED_ENGINE_SHA256",
    "EngineShaMismatch",
    "FAST",
    "FULL",
    "HazardExportUnavailable",
    "HazardTape",
    "Plan",
    "RecorderPool",
    "RestoreBudget",
    "SearchResult",
    "TeacherConfig",
    "TeacherLabel",
    "VerifiedPlan",
    "Weights",
    "beam_search",
    "check_engine_binary",
    "death_bomb_feasible",
    "death_bomb_window",
    "decide",
    "get_config",
    "record",
    "record_child_tape",
    "score_plan",
    "verify_plans",
]
