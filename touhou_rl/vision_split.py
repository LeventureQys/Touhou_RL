"""Stratified, nested training-set sub-sampling for the R3 data-scaling curve.

Why "nested prefix" instead of independent random subsets
--------------------------------------------------------
R3 compares 25% / 50% / 100% of the training set on one fixed validation set.  If
each fraction were sampled independently, the 25% subset would contain frames the
50% subset does not, and a difference between the two curves could come from the
sampling noise instead of from the data volume.  Taking a prefix of one
permutation per stage makes the subsets literally nested
(``0.25 < 0.5 < 1.0``), so the curve measures data volume and nothing else.

Why stratify by stage
---------------------
A uniform prefix over all stages would hand the small stages a wildly different
fraction than the large ones, and stage coverage is exactly what the student must
generalise across.  Stratifying keeps each stage's share constant.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def stratified_prefix(
    train_indices: Sequence[int],
    stage_of: np.ndarray,
    fraction: float,
    seed: int = 7,
) -> list[int]:
    """First ``round(fraction * n)`` frames of a per-stage permutation.

    ``stage_of`` is indexed by **dataset-global** index (``stage_of[i]`` is the
    stage of frame ``i``, not of the ``i``-th entry of ``train_indices``); build it
    with ``VisionDataset.stage_of(list(range(total)))``.

    One generator is consumed stage by stage in ascending stage order, and each
    stage's permutation is drawn from the full stage before any truncation, so
    ``stratified_prefix(x, f1)`` is a subset of ``stratified_prefix(x, f2)`` for
    every ``f1 <= f2`` under the same ``seed``.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    ordered = sorted(int(index) for index in train_indices)
    if fraction == 1.0:
        return ordered

    stages = np.asarray(stage_of)
    groups: dict[int, list[int]] = {}
    for index in ordered:
        groups.setdefault(int(stages[index]), []).append(index)

    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for stage in sorted(groups):
        members = np.asarray(groups[stage], dtype=np.int64)
        order = rng.permutation(members.size)
        take = min(round(fraction * members.size), members.size)
        picked.extend(int(value) for value in members[order[:take]])
    return sorted(picked)
