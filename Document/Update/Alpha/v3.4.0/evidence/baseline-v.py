"""Trivial baselines for the V gate: how much does phi (or bullet count) alone explain?"""
import json

import numpy as np

from touhou_rl import stage0_offline as s0


def safe(value):
    return None if value is None else round(float(value), 4)


data = s0.load_cache(r"Document/Update/Alpha/v3.4.0/stage0/cache.npz")
val = np.flatnonzero(data["split"].astype(np.int64) == 1)
groups = data["group"][val].astype(np.int64)
remaining = data["remaining"][val].astype(np.float64)
phi = data["meta"][val][:, 4].astype(np.float64)
bullets = data["bullets"][val][:, :, 5].astype(np.float32).sum(axis=1).astype(np.float64)
stage = data["meta"][val][:, 2].astype(np.float64) * 6.0

out = {"stage_unique": np.unique(stage).tolist(), "signals": {}}
for label, signal, sign in (
    ("phi", phi, +1.0),
    ("bullet_count", bullets, -1.0),
    ("stage", stage, -1.0),
):
    per_episode = []
    for group in sorted(set(int(item) for item in groups.tolist())):
        mask = groups == group
        value = s0.spearman(sign * signal[mask], remaining[mask])
        if value is not None:
            per_episode.append(value)
    out["signals"][label] = {
        "spearman_per_episode_mean": safe(np.mean(per_episode)) if per_episode else None,
        "spearman_pooled": safe(s0.spearman(sign * signal, remaining)),
        "episodes": len(per_episode),
    }
print(json.dumps(out, ensure_ascii=False, indent=2))
open(r".tmp\v340-baseline-v.json", "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))
