"""Stage 0 deployment behavior diagnostics.

The functions here are deliberately independent of the game and model.  They turn
recorded four-tuples into auditable candidate/action continuity statistics without
changing the v1.0.4 policy contract.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


DIAGNOSTICS_SCHEMA = "v105-stage0-action-diagnostics/1"


def candidate_index(action) -> int:
    """Decode the frozen ``focus*9 + move`` candidate index from an action tuple."""
    values = np.asarray(action).reshape(-1)
    if values.size < 3:
        raise ValueError(f"action must contain move and focus, got shape {values.shape}")
    move, focus = int(values[0]), int(values[2])
    if not 0 <= move < 9 or not 0 <= focus < 2:
        raise ValueError(f"action out of frozen candidate range: move={move}, focus={focus}")
    return focus * 9 + move


def summarize_actions(actions: Iterable) -> dict:
    """Return deterministic continuity statistics for a sequence of actions."""
    indices = [candidate_index(action) for action in actions]
    moves = [index % 9 for index in indices]
    focuses = [index // 9 for index in indices]
    changes = lambda values: sum(a != b for a, b in zip(values, values[1:]))
    lengths: list[int] = []
    if indices:
        run = 1
        for previous, current in zip(indices, indices[1:]):
            if current == previous:
                run += 1
            else:
                lengths.append(run)
                run = 1
        lengths.append(run)
    percentile = lambda q: float(np.percentile(np.asarray(lengths, dtype=np.float64), q)) if lengths else None
    return {
        "schema": DIAGNOSTICS_SCHEMA,
        "decisions": len(indices),
        "candidate_flip_rate": changes(indices) / max(1, len(indices) - 1),
        "move_flip_rate": changes(moves) / max(1, len(moves) - 1),
        "focus_flip_rate": changes(focuses) / max(1, len(focuses) - 1),
        "still_rate": sum(move == 0 for move in moves) / max(1, len(moves)),
        "focus_rate": sum(focuses) / max(1, len(focuses)),
        "action_run_length": {
            "p50": percentile(50), "p90": percentile(90), "max": max(lengths) if lengths else None,
        },
        "candidate_counts": dict(sorted(Counter(indices).items())),
    }


def empty_action_trace() -> dict:
    return {"actions": [], "candidate_indices": []}


__all__ = ["DIAGNOSTICS_SCHEMA", "candidate_index", "summarize_actions", "empty_action_trace"]
