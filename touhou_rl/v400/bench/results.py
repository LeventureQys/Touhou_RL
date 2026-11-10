"""Strict handling of measured benchmark JSONL; does not invent scores.

``results.jsonl`` is append-only, so superseded attempts stay on disk as audit
records (``status="invalid"``/``"error"`` and rows named by
``invalidated_started_at``).  Those lines are not episodes: they carry no outcome
numbers and are skipped here instead of being validated as measurements.  Every
line that is not explicitly voided is still validated strictly.
"""
from __future__ import annotations
import json
from pathlib import Path
from .stats import bootstrap
from .suites import SUITES, DEV_SEEDS, CONFIRM_SEEDS, STAGE_NAMES

REQUIRED = ("ckpt", "agent", "suite", "seed", "harness_commit", "privilege",
            "frames", "stages", "deaths", "bombs_used", "lives_end", "score",
            "cleared", "end", "wall_s")

#: ``status`` values that explicitly mean "this line is not a measured episode".
VOID_STATUSES = frozenset({"error", "invalid", "invalidated", "void", "voided",
                           "retracted", "superseded", "corrected"})
#: Boolean fields that void a row even when its ``status`` looks measured.
VOID_FLAGS = ("invalid", "invalidated", "void", "voided", "retracted", "superseded")


def row_stage(row) -> tuple[int | None, str | None, str]:
    """``(stage_index, stage_name, source)`` for one row, never guessed.

    ``stage_index`` is 0-based (``RESET <stage>`` / ``STATE.f1``).  Rows recorded
    before EXP-P0-077 carry no ``stage_index`` but do carry the engine's own
    ``stage_observed``, which is the same 0-based index; such a row is labelled
    ``source="stage_observed"`` so a derived table never passes an inferred stage
    off as a declared one.
    """
    if not isinstance(row, dict):
        return None, None, "unrecorded"
    declared = row.get("stage_index")
    if isinstance(declared, int) and not isinstance(declared, bool):
        if 0 <= declared < len(STAGE_NAMES):
            return declared, STAGE_NAMES[declared], "declared"
        return None, None, "unrecorded"
    observed = row.get("stage_observed")
    if isinstance(observed, int) and not isinstance(observed, bool):
        if 0 <= observed < len(STAGE_NAMES):
            return observed, STAGE_NAMES[observed], "stage_observed"
    return None, None, "unrecorded"


def is_voided(row) -> bool:
    """True when *row* explicitly labels itself as a non-measured record."""
    if not isinstance(row, dict):
        return False
    status = row.get("status")
    if isinstance(status, str) and status.strip().lower() in VOID_STATUSES:
        return True
    if row.get("measured") is False:
        return True
    return any(row.get(flag) is True for flag in VOID_FLAGS)


def invalidated_stamps(rows) -> set[str]:
    """Every ``started_at`` value an explicit ``status="invalid"`` audit row names.

    Kept in sync with ``tools/rl/v400_bench_run.py:invalidated_stamps`` so this
    reader and the scoreboard generator exclude the same superseded episodes.
    """
    stamps: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "invalid":
            continue
        for stamp in row.get("invalidated_started_at") or ():
            if isinstance(stamp, str):
                stamps.add(stamp)
    return stamps


def exclude_voided(rows: list[dict]) -> list[dict]:
    """Drop explicitly voided/invalidated lines; survivors are still validated."""
    stamps = invalidated_stamps(rows)
    return [row for row in rows
            if not is_voided(row) and row.get("started_at") not in stamps]


def validate(row: dict) -> dict:
    if not isinstance(row, dict) or any(key not in row for key in REQUIRED):
        raise ValueError("incomplete measured episode record")
    if row["suite"] not in SUITES or row["seed"] not in (*DEV_SEEDS, *CONFIRM_SEEDS):
        raise ValueError("unregistered suite or seed")
    if row["privilege"] != "deploy_pixel" or not row["harness_commit"]:
        raise ValueError("benchmark requires pixel privilege and harness provenance")
    if not isinstance(row["cleared"], bool) or not isinstance(row["stages"], list):
        raise ValueError("malformed episode verdict")
    if not row["stages"] or row["frames"] < 0 or row["wall_s"] < 0:
        raise ValueError("missing stages or negative duration")
    # EXP-P0-077: a row may declare the stage it ran.  It is a 0-based engine
    # stage index, so it must agree with the measured reference name; a 1-based
    # value here would silently relabel the episode.
    if "stage_name" in row and "stage_index" not in row:
        raise ValueError("stage_name without a stage_index")
    if "stage_index" in row:
        index = row["stage_index"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(STAGE_NAMES):
            raise ValueError(f"stage_index is not a 0-based engine stage index: {index!r}")
        if "stage_name" in row and row["stage_name"] != STAGE_NAMES[index]:
            raise ValueError(
                f"stage_name {row['stage_name']!r} does not match 0-based stage_index {index} "
                f"({STAGE_NAMES[index]!r})")
    return row


def load_results(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"no measured results: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [validate(row) for row in exclude_voided(rows)]


def summarize(rows: list[dict]) -> str:
    """Per-cell summary; the cell key includes the 0-based ``stage_index``.

    Stage is part of the key because a suite name alone does not identify what was
    played: before EXP-P0-077 ``S1N`` reset engine stage ``1`` (the real Stage 2).
    Grouping such rows with a Stage-1 row would average two different stages into
    one cell, so every row is labelled with the stage it declares (``stage_index``)
    or, for a pre-EXP-P0-077 row, the stage the engine itself reported
    (``stage_observed``).
    """
    groups: dict[tuple[str, str, int | None, str, str, str], list[dict]] = {}
    for row in exclude_voided(rows):
        validate(row)
        index, name, source = row_stage(row)
        # Cap provenance and the stage *source* are part of the group key: a
        # provisional-cap episode and a human-referenced episode are not the same
        # measurement, and a declared stage must not be averaged into a cell whose
        # stage was only observed.
        key = (row["agent"], row["suite"], index, name or "unrecorded",
               row.get("cap_source", "unrecorded"), source)
        groups.setdefault(key, []).append(row)
    out = ["# TH08-Bench measured results", "",
           "Stage indices are 0-based engine stages (`RESET <stage>` / `STATE.f1`); "
           "`stage_index 0` is the human-visible Stage 1.", "",
           "| Agent | Suite | Stage idx | Stage | Cap source | Stage source | Episodes | "
           "Clear rate [95% CI] | Deaths [95% CI] | Score [95% CI] |",
           "|---|---|---:|---|---|---|---:|---|---|---|"]
    for (agent, suite, index, name, cap_source, source), episodes in sorted(groups.items()):
        def cell(key):
            r = bootstrap([float(row[key]) for row in episodes])
            return f'{r["mean"]:.3f} [{r["lo"]:.3f}, {r["hi"]:.3f}]'
        index_text = "unrecorded" if index is None else str(index)
        out.append(f'| {agent} | {suite} | {index_text} | {name} | {cap_source} | {source} | '
                   f'{len(episodes)} | {cell("cleared")} | {cell("deaths")} | {cell("score")} |')
    if not groups:
        out.append("\nNo measured episodes available; baseline scores are pending.")
    return "\n".join(out) + "\n"
