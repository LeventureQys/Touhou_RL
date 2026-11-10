"""Declared suites and disjoint seed partitions; no unmeasured frame caps.

Stage numbering (EXP-P0-077)
----------------------------
Every stage number in this module is the engine's own **0-based** ``stage_index``:
the value is passed verbatim as the ``RESET <stage> ...`` argument and it is what
``STATE.f1`` reports back.  It is the ``Stage`` enum order of the retail executable
(``third_party/th08/src/ScoreDat.hpp``: ``STAGE1 = 0`` ... ``STAGE6B = 7``,
``EXTRASTAGE = 8``) and the ``stage_index`` numbering of
``bench/human_reference.json``, so ``stage_index 0`` (human-visible **Stage 1**) is
the first stage.

Concretely: ``S1N`` is ``stage_index 0``, normal, one stage.  ``stage_index 1``
would be the real Stage 2 (this was the pre-EXP-P0-077 defect: ``S1N`` declared
``1``, so the engine started Stage 2 while the cap was looked up as Stage 1).

The human-visible names never appear on the wire: use :func:`stage_name` for the
``bench/human_reference.json`` name (``"Stage1"`` ... ``"Stage6B"``) and
:func:`stage_ordinal` for the ordinal label (``"Stage 1"`` ... ``"Stage 8"``).
"""
from __future__ import annotations
from dataclasses import dataclass

DEV_SEEDS = tuple(range(1000, 1100))
CONFIRM_SEEDS = tuple(range(60000, 60200))

#: Human-visible stage names by 0-based ``stage_index``; identical to
#: ``human_reference.STAGE_NAMES`` and to the ``stage_name`` values measured in
#: ``bench/human_reference.json``.
STAGE_NAMES: tuple[str, ...] = ("Stage1", "Stage2", "Stage3", "Stage4A", "Stage4B",
                                "Stage5", "Stage6A", "Stage6B", "Extra")
#: Ordinal label for each ``stage_index``, i.e. ``stage_index + 1``.  This is the
#: human-facing "Stage N" wording and is never sent to the engine.
STAGE_ORDINALS: tuple[str, ...] = tuple(f"Stage {index + 1}" for index in range(8))
#: The eight gameplay stages as 0-based indices: Stage 1 .. Stage 6B.
STAGE_INDEXES: tuple[int, ...] = tuple(range(8))


def stage_name(stage_index: int) -> str:
    """Return the measured reference name for a 0-based *stage_index*."""
    if not isinstance(stage_index, int) or isinstance(stage_index, bool):
        raise TypeError(f"stage_index must be an int, got {stage_index!r}")
    if not 0 <= stage_index < len(STAGE_NAMES):
        raise ValueError(f"stage_index out of range 0..{len(STAGE_NAMES) - 1}: {stage_index}")
    return STAGE_NAMES[stage_index]


def stage_ordinal(stage_index: int) -> str:
    """Return ``"Stage N"`` for a 0-based *stage_index* (``0`` -> ``"Stage 1"``)."""
    stage_name(stage_index)
    return STAGE_ORDINALS[stage_index] if stage_index < len(STAGE_ORDINALS) else "Extra"


@dataclass(frozen=True)
class Suite:
    """A declared suite; ``stage_indexes`` are 0-based engine stage indices."""

    name: str
    stage_indexes: tuple[int, ...]
    difficulty: str
    resources: str
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if not self.stage_indexes:
            raise ValueError(f"suite {self.name} declares no stage")
        for stage_index in self.stage_indexes:
            stage_name(stage_index)  # validates type and 0..8 range

    @property
    def stages(self) -> tuple[int, ...]:
        """Alias of :attr:`stage_indexes`; **0-based**, never a 1-based number."""
        return self.stage_indexes

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(STAGE_NAMES[index] for index in self.stage_indexes)

    @property
    def stage_index(self) -> int | None:
        """The single 0-based stage index, or ``None`` for a multi-stage suite."""
        return self.stage_indexes[0] if len(self.stage_indexes) == 1 else None

    @property
    def stage_name(self) -> str | None:
        """The single stage's measured reference name, or ``None`` when multi-stage."""
        return STAGE_NAMES[self.stage_indexes[0]] if len(self.stage_indexes) == 1 else None


SUITES = {
    "S1N": Suite("S1N", (0,), "normal", "standard"),
    "S1L": Suite("S1L", (0,), "lunatic", "standard"),
    "PSN": Suite("PSN", STAGE_INDEXES, "normal", "practice"),
    "PSL": Suite("PSL", STAGE_INDEXES, "lunatic", "practice"),
    "RUNN": Suite("RUNN", STAGE_INDEXES, "normal", "standard"),
}


def seeds(split: str, count: int) -> tuple[int, ...]:
    pool = {"dev": DEV_SEEDS, "confirm": CONFIRM_SEEDS}[split]
    if not 1 <= count <= len(pool):
        raise ValueError("count exceeds registered seed split")
    return pool[:count]
