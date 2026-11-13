"""Read-only standard-resource seed probe: fixed action after first death."""

from __future__ import annotations

import json

from touhou_rl.v400.gym.env import Th08Env, StageStart
from touhou_rl.v400.labels.state import FIELD_NAMES

ENGINE = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA = "/root/v400-data/game"
DECISIONS = 3000


def run(seed: int) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    with Th08Env(
        ENGINE,
        DATA,
        shm_name=f"v410seedstandard-{seed}",
        privileged=True,
        decision_frames=2,
        max_frames=6000,
        role="shadow",
    ) as env:
        env.reset(
            StageStart(stage=0, difficulty="normal", team=0, seed=seed, resources="standard")
        )
        for _ in range(DECISIONS):
            _obs, _reward, terminated, truncated, info = env.step(0, 0)
            rows.append(tuple(int(value) for value in info.state.values()))
            if terminated or truncated:
                break
    return rows


def main() -> int:
    runs = {seed: run(seed) for seed in (1, 2, 3)}
    names = list(FIELD_NAMES)
    base = runs[1]
    payload = {
        "schema": "v410-seed-standard-probe/1",
        "engine": ENGINE,
        "decisions": DECISIONS,
        "rows": {str(seed): len(rows) for seed, rows in runs.items()},
        "first_non_seed_divergence": {},
        "field_diff_counts": {},
    }
    for seed in (2, 3):
        other = runs[seed]
        first = None
        diff_counts: dict[str, int] = {}
        for index, (left, right) in enumerate(zip(base, other)):
            for field_index, name in enumerate(names):
                if name == "rng_seed":
                    continue
                if left[field_index] != right[field_index]:
                    diff_counts[name] = diff_counts.get(name, 0) + 1
                    if first is None:
                        first = {"decision": index, "frame": index * 2, "field": name}
        payload["first_non_seed_divergence"][f"1_vs_{seed}"] = first
        payload["field_diff_counts"][f"1_vs_{seed}"] = diff_counts
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
