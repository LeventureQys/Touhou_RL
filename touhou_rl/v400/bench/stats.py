"""Episode-level confidence intervals and paired comparisons."""
from __future__ import annotations
import numpy as np


def bootstrap(values, *, samples: int = 10_000, seed: int = 0) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("nonempty finite episode values required")
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(samples, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    return {"mean": float(values.mean()), "lo": float(low), "hi": float(high), "n": len(values)}


def paired(a, b, *, samples: int = 10_000, seed: int = 0) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("equal nonempty paired episode arrays required")
    diff = a - b
    interval = bootstrap(diff, samples=samples, seed=seed)
    rng = np.random.default_rng(seed + 1)
    signs = rng.choice((-1, 1), size=(samples, len(diff)))
    null = (signs * diff).mean(axis=1)
    interval["permutation_p"] = float((1 + np.count_nonzero(np.abs(null) >= abs(diff.mean()))) / (samples + 1))
    return interval
