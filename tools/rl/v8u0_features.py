#!/usr/bin/env python3
"""v8.28.0 FEATURES — the v8.24.0 sparse-reward recipe with **3 explicit
features appended to the bomb head's input**, to test whether the *input
representation* is what kept the bomb head from learning *when* to bomb.

What this round asks
--------------------
v8.27.0 showed that *neither* the sparse (v8.24.0, 4.08 % at 30 s) *nor* the
dense reward (0.61 % / 0.78 %) teaches 30 s bomb timing, and that **both arms
converge to "first bomb at decision 0"** (surviving episodes use all 3 bombs).
That leaves one hypothesis standing (prereg §0): the bottleneck is the
**representation** — ``BombHead(106 -> 64 -> 2)`` reads the raw 106-dim state,
which does *not* expose "how many bombs are left / how much of the episode is
left / how close is the nearest bullet", so the only robust policy it can learn
is "bomb immediately".

This round keeps **everything** from v8.24.0's published sparse arm (D0) and
changes exactly one thing: the bomb head's input dimension.

===================  =========================================================
``obs``              106-dim (``v431_arm_a.build_features``) — **unchanged**
``f1``               ``bombs_left / 3.0``
``f2``               ``max(budget - decisions_executed, 0) / budget``
``f3``               ``clip(nearest_all_px, 0, 200) / 200.0``
===================  =========================================================

``obs_aug = concat(obs_106, [f1, f2, f3])`` -> **109 dims**, and
``BombHead = Linear(109, 64) -> Tanh -> Linear(64, 2)``.

The reward is **v8.24.0's sparse form** ``R = survived(900) + 0.5 *
alive/1800`` so the 30 s result is directly comparable with D0's 4.08 %.

Scope (``PREREGISTRATION_FEATURES_001.md`` §1)
-----------------------------------------------
* **1 arm** (``F``, the explicit-feature head).  No second arm: the prereg's
  single factor is the input dimension, and the 106-dim counterparts are the
  already-published v8.24.0 (sparse) and v8.27.0 (dense) numbers, never re-run.
* **3 seeds x 300 episodes x 900 decisions**.
* **evaluation: the 900-decision tier only**, 3 seeds x 4 windows x **150**
  episodes = **1,800** episodes.  There is **no 20 s tier**, so nothing may be
  inferred about 20 s / 40 s.
* **P3 is now a behaviour metric** (prereg §1.2): the *median* of
  ``first_bomb_decision`` must be **> 10** decisions, i.e. the head must stop
  bombing at decision 0.

Frozen plumbing (identical to v8.21.0 / v8.24.0 / v8.27.0)
---------------------------------------------------------
* candidate head frozen: ``.tmp_v860_bombdag/ckpt/seed{0,1,2}/bombdag.pt``
  (``Policy(106, 64, 36)``) read out by **argmax**;
* 200 BCE warm-start steps on the 60/40/20 rule's own decisions (4 windows x 1
  episode, 900 decisions each) — **now on the 109-dim input** (the features are
  computed on every decision by the same code path as training);
* REINFORCE, Adam 3e-4, entropy 0.01, moving-average baseline 0.95, grad-clip
  1.0, 300 episodes per seed;
* the episode loop, the masks, the death rule and the engine plumbing are the
  frozen ones (``v830_bombpol.run_episode36`` / ``v450_cf_label``), imported.

Modes
-----
``--mode preflight``      engine census before any launch.
``--mode assert-ckpt``    engine ``691bfd77``, ``bombdag.pt`` (``9c6d5d4a`` /
                          ``36a51090`` / ``885dde70``), ``round2.pt``
                          (``4abee069`` / ``a93746bb`` / ``c69960fc``), the
                          protected sources, this file, the prereg.
``--mode d3``             the D3 fidelity gate (5 assertions); a non-5/5 gate
                          voids the round.  **Must be written before the
                          summary/verdict is generated** (v8.27.0 was once
                          misread as ``S-fork`` because a summary predated the
                          gate file by 12 s).
``--mode pretrain``       BCE warm start on the **109-dim** input.
``--mode train``          REINFORCE for ``--episodes`` (300) x 900 decisions
                          with the sparse v8.24.0 reward.
``--mode train-smoke``    one training episode + the feature-range assertion
                          (every ``f1/f2/f3`` in ``[0, 1]``) + the R magnitude
                          assertion (``R = survived + 0.5*alive/1800 in [0, 1.5]``)
                          + the first 5 bomb emissions' ``f1/f2/f3``/nearest.
``--mode eval``           one seed x a window subset x 150 episodes.
``--mode verdict``        pooled rate + exact CP95 + per-window/per-seed/
                          per-cell ``distinct`` + bomb usage split by outcome +
                          the ``first_bomb_decision`` distribution (P3) + the
                          side-by-side with 4.08 % / 24.17 % / 60.83 %.
``--mode summary``        alias of ``verdict`` (one code path, so the published
                          summary can never disagree with the verdict).
``--mode audio-audit``    AGENTS.md §9.3 on this round's own data dirs.
``--mode process-audit``  this round's own engine processes must be 0.
``--mode publish``        copy the raw artifacts into
                          ``Document/Update/Alpha/v8.28.0/evidence/FEATURES/``
                          and write ``manifest.json``.

The ``--tag`` rule
------------------
Every process **must** pass its own unique ``--tag``: the private data dir is
``<work>/data/<tag>/w<window>`` and the engine log dir is ``<work>/runs/<tag>``.
This round's tags all start with ``v8u0-``.

Not in this file
----------------
* No change to ``touhou_rl/**``, ``third_party/**``, any existing runner or any
  existing evidence directory; ``tools/rl/v8t0_densereward.py`` and
  ``tools/rl/v8p0_bombrl900.py`` are only read.
* No claim about resource *allocation*; no 20 s / 40 s inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASE_WORK = REPO / ".tmp_v8u0_features"
ARM = "F"                       # the single arm of this round (--arm F)
WORK = BASE_WORK                # single arm: no per-arm sub directory
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.28.0" / "evidence"
           / "FEATURES")

SCHEMA_PREFLIGHT = "v8u0-features-preflight/1"
SCHEMA_ASSERT = "v8u0-features-ckpt-assertions/1"
SCHEMA_D3 = "v8u0-features-d3-gate/1"
SCHEMA_PRETRAIN = "v8u0-features-pretrain/1"
SCHEMA_TRAIN = "v8u0-features-train/1"
SCHEMA_TRAIN_SMOKE = "v8u0-features-train-smoke/1"
SCHEMA_EVAL = "v8u0-features-eval/1"
SCHEMA_EVAL_EPISODE = "v8u0-features-episode/1"
SCHEMA_EVAL_SMOKE = "v8u0-features-eval-smoke/1"
SCHEMA_SUMMARY = "v8u0-features-summary/1"
SCHEMA_AUDIO = "v8u0-features-audio-audit/1"
SCHEMA_PROC = "v8u0-features-process-audit/1"
SCHEMA_PUBLISH = "v8u0-features-publish/1"
SCHEMA_MANIFEST = "v8u0-features-manifest/1"

PRODUCER = ("tools/rl/v8u0_features.py (live engine, HarnessClient role=shadow, "
            "episode loop = tools/rl/v830_bombpol.py::run_episode36)")
PREREG_REL = "Document/Update/Alpha/v8.28.0/PREREGISTRATION_FEATURES_001.md"
ERRATA_REL = ""

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

#: **one** arm: the explicit-feature bomb head (prereg §1's single factor).
#: The 106-dim counterparts are the published v8.24.0 / v8.27.0 numbers and are
#: never re-run here.
ARMS: tuple[str, ...] = ("F",)

# -- the frozen protocol (prereg §1) -----------------------------------------
SEEDS: tuple[int, ...] = (0, 1, 2)
WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
DECISION_FRAMES = 2
PREFIX_FRAMES = 8663
N_CAND = 18
N_ACTIONS = 36
#: the frozen 106-dim observation (``v431_arm_a.build_features``), unchanged
OBS_DIM = 106
#: prereg §1.1: three explicit features appended to the observation
EXPLICIT_DIM = 3
FEATURE_DIM = OBS_DIM + EXPLICIT_DIM          # 109
FEATURE_NAMES = ("bombs_left/3", "decisions_left/budget",
                 "clip(nearest_all_px,0,200)/200")
BOMBS_MAX = 3.0               # f1's denominator (prereg §1.1)
NEAREST_CLIP_PX = 200.0       # f3's clip and denominator (prereg §1.1)
BOMB_HIDDEN = 64
N_BOMB = 2
GEN_CONST = 7

TRAIN_EPISODES = 300
TRAIN_DECISIONS = 900        # v8.24.0's single change: 600 -> 900 (prereg §1)
TRAIN_FRAMES = TRAIN_DECISIONS * DECISION_FRAMES          # 1,800 frames = 30 s

EVAL_BUDGETS: tuple[int, ...] = (900,)   # prereg §1: the 30 s tier only
EVAL_EPISODES = 150                      # prereg §3: 4 windows x 150 per cell

#: the reward (prereg §1: **unchanged** from v8.24.0's D0, so the 30 s column is
#: comparable with the published 4.08 %):
#:   R = survived(900) + 0.5 * alive_frames / 1800
REWARD_BY_ARM: dict[str, tuple[float, float]] = {
    "F": (1.0, 0.5),
}
REWARD_SURVIVED = 1.0        # per-arm, see reward_of()
REWARD_ALIVE_WEIGHT = 0.5    # per-arm, see reward_of()
ALIVE_FRAMES_NORM = 1800.0

BASELINE_EMA = 0.95          # moving-average baseline (v431_arm_a's 0.95/0.05)
ENTROPY_COEF = 0.01          # prereg §1
LR = 3e-4                    # prereg §1
GRAD_CLIP = 1.0

BC_STEPS = 200               # prereg §1
BC_BATCH = 256
BC_LR = 1e-2                 # not fixed by the prereg; see RUN.md §deviations
#: one rule episode per dev window: with the frozen candidate head read out by
#: argmax the rule's trajectory is deterministic, so extra episodes are byte
#: duplicates of the same 900 decisions (v8.21.0 verified ep1 == ep0 on every
#: window; prereg §1 keeps the same 4-episode warm start).
BC_EPISODES_PER_WINDOW = 1
BC_POS_FRACTION = 0.5        # balanced batches; see RUN.md §deviations

#: the warm-start rule: v8.20.0 U-*'s published 60/40/20 (read through v8l0).
#: NB: the warm start is *unchanged* from v8.21.0 (prereg §1); the P1 reference
#: is the separate published 40/20/5 table below, which is never re-run here.
RULE_NAME = "S1"
RULE_LABEL = "60/40/20"

#: identities this round must find (fail closed)
V860_ROOT = REPO / ".tmp_v860_bombdag" / "ckpt"
V860_SHA256: dict[int, str] = {
    0: "9c6d5d4ac7c9a6120ecd33aa9de7a0762c916413cc8f52e0c82f8ac41f8a483b",
    1: "36a51090d56149dc185d86e5b386f89304e5522c3612d9e919a4092e99f8e84c",
    2: "885dde70c57f18bb7f1e3d0351737d239a52ce064aeb47fda025866bd6e45943",
}
V450_ROOT = REPO / ".tmp_v450_cf" / "ckpt"
V450_SHA256: dict[int, str] = {
    0: "4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc",
    1: "a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4",
    2: "c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01",
}
ENGINE_SHA256 = "691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6"
PROTECTED_FILES = (
    "tools/rl/v432_fork_arm.py",
    "tools/rl/v450_cf_label.py",
    "tools/rl/v431_arm_a.py",
    "tools/rl/v800_bomb.py",
    "tools/rl/v830_bombpol.py",
    "tools/rl/v860_bombdag.py",
    "tools/rl/v8l0_unified.py",
    "tools/rl/v8p0_bombrl900.py",
    "tools/rl/v8q0_t33tiers.py",
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)

# --------------------------------------------------------------------------- #
# published reference tables — **numbers only, never re-run** (prereg §2)
# --------------------------------------------------------------------------- #
#: v8.23.0 RULETIGHT T2 = 40/20/5 at 900 decisions (30 s) — **P1's threshold**
#: and the deployment config ``S-no`` falls back to.  Per-window n from the
#: published table: w11/w12/w17 = 60 each, w18 = 14 (that cell lost rows);
#: 40 + 39 + 26 + 7 = 112 of 194 = 57.73 %.
REF_30S_40_20_5: dict[str, Any] = {
    "label": "v8.23.0 T2 = 40/20/5, 900 decisions (30 s)",
    "pooled": (112, 194),
    "per_window": {11: (40, 60), 12: (39, 60), 17: (26, 60), 18: (7, 14)},
    "source": ("Document/Update/Alpha/v8.13.0/SYNTHESIS_BOMBLINE_2026-11-22.md"
               " §31.1 and §31.5 (v8.23.0 RULETIGHT, frozen 2026-11-22)"),
}

#: v8.20.0 U-900 = 60/40/20 at 900 decisions (30 s), re-derived from
#: ``.tmp_v8l0_unified/eval/eval_U900_d900_w*_seed*.jsonl`` (12 x 20 rows).
REF_30S_60_40_20: dict[str, Any] = {
    "label": "v8.20.0 U-900 = 60/40/20, 900 decisions (30 s)",
    "pooled": (58, 240),
    "per_window": {11: (16, 60), 12: (11, 60), 17: (11, 60), 18: (20, 60)},
    "source": ".tmp_v8l0_unified/eval/eval_U900_d900_w*_seed*.jsonl",
}

#: v8.21.0 BOMBRL's published cells (synthesis §31.2) — numbers only.
#: 10 s 99.86 % (n = 1,417, 1.315 bombs), 20 s 90.26 % (n = 729, 2.964 bombs),
#: 30 s 4.79 % (n = 585, 2.962 bombs).
REF_RL821: dict[str, Any] = {
    "label": "v8.21.0 BOMBRL (600-decision training episodes)",
    "10s": {"pooled": (1415, 1417), "rate": 1415 / 1417, "bombs_mean": 1.315},
    "20s": {"pooled": (658, 729), "rate": 658 / 729, "bombs_mean": 2.964},
    "30s": {"pooled": (28, 585), "rate": 28 / 585, "bombs_mean": 2.962},
    "source": ("Document/Update/Alpha/v8.13.0/SYNTHESIS_BOMBLINE_2026-11-22.md"
               " §31.2"),
}

#: v8.20.0 U-600 = 60/40/20 at 600 decisions (20 s), re-derived from
#: ``.tmp_v8l0_unified/eval/eval_U600_d600_w*_seed*.jsonl`` (12 x 20 rows).
REF_20S_60_40_20: dict[str, Any] = {
    "label": "v8.20.0 U-600 = 60/40/20, 600 decisions (20 s)",
    "pooled": (237, 240),
    "per_window": {11: (60, 60), 12: (59, 60), 17: (58, 60), 18: (60, 60)},
    "source": ".tmp_v8l0_unified/eval/eval_U600_d600_w*_seed*.jsonl",
}

#: **D0** — v8.24.0's published sparse-reward arm: 147/3,600 = 4.08 % at
#: 900 decisions, 1,250/3,600 = 34.72 % at 600 (numbers only, never re-run).
REF_D0_824: dict[str, Any] = {
    "label": "v8.24.0 D0 sparse reward R = survived(900) + 0.5 x alive/1800",
    "600": {"pooled": (1250, 3600), "rate": 1250 / 3600},
    "900": {"pooled": (147, 3600), "rate": 147 / 3600},
    "source": ("Document/Update/Alpha/v8.24.0/evidence/DENSEREWARD/summary.json"
               " (arms.600 / arms.900, published 2026-11-22)"),
}

#: **G1** — v8.26.0 URGENCY's adaptive rule (a = 0.50), the current best of
#: this line at 30 s: 146/240 = 60.83 % (own-window n = 4 windows x 60).
REF_G1: dict[str, Any] = {
    "label": "v8.26.0 URGENCY G1 adaptive (a = 0.50), n = 240",
    "600": {"pooled": (239, 240), "rate": 239 / 240},
    "900": {"pooled": (146, 240), "rate": 146 / 240},
    "1200": {"pooled": (34, 240), "rate": 34 / 240},
    "source": ("Document/Update/Alpha/v8.26.0/evidence/URGENCY/RUN.md"
               " (G1 row; frozen 2026-11-22)"),
}

#: prereg §1.2 thresholds
P1_THRESHOLD = 58.0 / 240.0     # 24.1667 %, v8.20.0 U-900 = 60/40/20
P1_ALPHA = 0.05                 # one-sided Fisher
P1_REFERENCE = REF_30S_60_40_20
P2_THRESHOLD = 147.0 / 3600.0   # 4.0833 %, v8.24.0 D0 (same framework, sparse)
P2_ALPHA = 0.05
P2_REFERENCE = REF_D0_824["900"]
P2_REFERENCE_MEAN = 2.962       # v8.21.0's 30 s bombs/episode
#: P3 (prereg §1.2): the **median** of ``first_bomb_decision`` must exceed 10 —
#: the head must stop "bombing at decision 0".
P3_MEDIAN_THRESHOLD = 10.0
P3_TESTED = True
#: prereg §1.2 P5: a (seed, window) cell with fewer distinct outcomes is void
DISTINCT_MIN = 10

_LOG_FH: Any = None
_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# errors / logging / hashing / module loading
# --------------------------------------------------------------------------- #
class BombRLError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v8u0] {message}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BombRLError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cached(key: str, rel: str):
    if key not in _CACHE:
        module = load_module(key, REPO / rel)
        if hasattr(module, "_LOG_FH"):
            module._LOG_FH = _LOG_FH
        _CACHE[key] = module
    return _CACHE[key]


def v450():
    return _cached("v450_cf_label", "tools/rl/v450_cf_label.py")


def arm_a():
    return _cached("v431_arm_a", "tools/rl/v431_arm_a.py")


def m800():
    return _cached("v800_bomb", "tools/rl/v800_bomb.py")


def v830():
    return _cached("v830_bombpol", "tools/rl/v830_bombpol.py")


def v860():
    return _cached("v860_bombdag", "tools/rl/v860_bombdag.py")


def v8l0():
    """The frozen v8.20.0 runner: the 60/40/20 rule and the exact CP95/Fisher."""

    return _cached("v8l0_unified", "tools/rl/v8l0_unified.py")


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> dict[str, Any]:
    return v8l0().clopper_pearson(int(k), int(n), alpha)


def fisher_greater(a: int, b: int, c: int, d: int) -> dict[str, Any]:
    return v8l0().fisher_greater(int(a), int(b), int(c), int(d))


def fisher_less(a: int, b: int, c: int, d: int) -> dict[str, Any]:
    return v8l0().fisher_less(int(a), int(b), int(c), int(d))


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


def read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def quantiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(float(v) for v in values)

    def pick(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = q * (len(ordered) - 1)
        low = int(math.floor(index))
        high = int(math.ceil(index))
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return {"n": len(ordered), "mean": round(statistics.fmean(ordered), 4),
            "median": round(statistics.median(ordered), 4),
            "q1": round(pick(0.25), 4), "q3": round(pick(0.75), 4),
            "min": round(ordered[0], 4), "max": round(ordered[-1], 4),
            "stdev": (round(statistics.stdev(ordered), 4)
                      if len(ordered) > 1 else None)}


def generator_seed(seed: int) -> int:
    """v4.5.0 §2.3 / v8.20.0: ``torch.Generator().manual_seed(seed * 1000 + 7)``."""

    return int(seed) * 1000 + GEN_CONST


def reward_of(survived: bool, alive_frames: int,
              arm: str | None = None) -> float:
    """The arm's reward: ``R = survived(900) + 0.5 * alive/1800`` (v8.24.0 D0)."""

    weight_survived, weight_alive = REWARD_BY_ARM[arm or ARM]
    return (weight_survived * (1.0 if survived else 0.0)
            + weight_alive * float(alive_frames) / ALIVE_FRAMES_NORM)


def explicit_features(state: dict[str, Any], hazards: Any, decision: int,
                      budget: int) -> dict[str, Any]:
    """prereg §1.1: the three explicit features, each in ``[0, 1]``.

    ``f1 = bombs_left / 3.0``
        ``bombs_remaining`` is the engine's own counter, read from the state the
        decision is made on (the same value the 60/40/20 rule thresholds on).

    ``f2 = max(budget_decisions - decisions_executed, 0) / budget_decisions``
        ``decisions_executed`` at the *start* of decision ``d`` is exactly ``d``
        (the loop increments it after a decision survives its 2 frames), so the
        decision index **is** the number of decisions already executed.

    ``f3 = clip(nearest_all_px, 0, 200) / 200``
        ``nearest_all_px`` is ``v800_bomb.nearest_bullet_distances`` — **the very
        measurement the 60/40/20 rule thresholds on** — so the feature and the
        rule cannot disagree about the geometry.  When the bullet list is empty
        the minimum is undefined; the far end of the clip (``1.0``) is used and
        ``no_bullets`` is flagged, which is the only case that is not a literal
        reading of the formula.

    Every value is clipped into ``[0, 1]`` and the caller asserts the range.
    """

    bombs_left = float(state.get("bombs_remaining") or 0.0)
    budget = max(int(budget), 1)
    decisions_left = max(int(budget) - int(decision), 0)
    distances = m800().nearest_bullet_distances(state, hazards)
    nearest = distances["nearest_all_px"]
    no_bullets = nearest is None
    nearest_value = NEAREST_CLIP_PX if no_bullets else float(nearest)
    f1 = min(max(bombs_left / BOMBS_MAX, 0.0), 1.0)
    f2 = min(max(float(decisions_left) / float(budget), 0.0), 1.0)
    f3 = min(max(nearest_value / NEAREST_CLIP_PX, 0.0), 1.0)
    return {"f1": f1, "f2": f2, "f3": f3,
            "bombs_left": bombs_left, "decisions_left": int(decisions_left),
            "budget_decisions": int(budget),
            "nearest_all_px": (None if no_bullets else round(nearest_value, 4)),
            "no_bullets": bool(no_bullets),
            "features": (f1, f2, f3)}


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def ckpt_dir() -> pathlib.Path:
    return WORK / "ckpt"


def bc_ckpt_path(seed: int) -> pathlib.Path:
    return ckpt_dir() / f"seed{int(seed)}" / f"bomb_head_seed{int(seed)}_bc.pt"


def head_ckpt_path(seed: int) -> pathlib.Path:
    return ckpt_dir() / f"seed{int(seed)}" / f"bomb_head_seed{int(seed)}.pt"


def train_rows_path(seed: int) -> pathlib.Path:
    return WORK / "train" / f"seed{int(seed)}.jsonl"


def eval_path(arm: str, budget: int, window: int, seed: int) -> pathlib.Path:
    return WORK / "eval" / (f"eval_{arm}_d{int(budget)}_w{int(window)}"
                            f"_seed{int(seed)}.jsonl")


def progress_path(arm: str, budget: int, seed: int,
                  windows: Sequence[int] | None = None) -> pathlib.Path:
    """The eval progress file; the window set is part of the name.

    v8.27.0 shards one eval cell per process (one window each) so that 12 cells
    can run at once; without the window suffix the four windows of one
    (arm, seed) pair would overwrite each other's progress file.
    """

    suffix = ("_w" + "-".join(str(int(w)) for w in windows)) if windows else ""
    return WORK / "eval" / (f"progress_{arm}_d{int(budget)}_s{int(seed)}"
                            f"{suffix}.json")


# --------------------------------------------------------------------------- #
# the frozen candidate head, the bomb head, the observation
# --------------------------------------------------------------------------- #
def load_policy(seed: int) -> tuple[Any, pathlib.Path, str]:
    path = V860_ROOT / f"seed{int(seed)}" / "bombdag.pt"
    if not path.is_file():
        raise BombRLError(f"seed {seed}: checkpoint not found: {path}")
    digest = sha256_file(path)
    expected = V860_SHA256[int(seed)]
    if digest != expected:
        raise BombRLError(f"seed {seed}: {path} sha256 {digest[:16]} != "
                          f"{expected[:16]}")
    return v860().load_policy36(path, seed=seed), path, digest


def make_bomb_head(seed: int) -> Any:
    """``BombHead(109 -> 64 -> 2)``: ``Linear -> Tanh -> Linear(64, 2)``."""

    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))
    return nn.Sequential(nn.Linear(FEATURE_DIM, BOMB_HIDDEN), nn.Tanh(),
                         nn.Linear(BOMB_HIDDEN, N_BOMB))


def load_bomb_head(seed: int, path: pathlib.Path) -> Any:
    import torch

    head = make_bomb_head(seed)
    if not path.is_file():
        raise BombRLError(f"bomb head checkpoint not found: {path}")
    head.load_state_dict(torch.load(str(path), map_location="cpu",
                                    weights_only=True))
    head.eval()
    return head


def features_of(state: dict[str, Any], hazards: Any, decision: int,
                budget: int) -> tuple[Any, Any, dict[str, Any]]:
    """``obs_aug = concat(obs_106, [f1, f2, f3])`` -> **109 dims** (prereg §1.1).

    Returns ``(obs_106, obs_109, explicit)``.  The 106-dim block is the frozen
    ``v431_arm_a.build_features`` (byte-identical to every previous round) and is
    what the **candidate** head consumes; the three explicit features are
    appended for the **bomb** head only, so the frozen candidate policy still
    sees exactly the observation it was trained on.
    """

    import numpy as np

    feats, _detail = arm_a().build_features(state=state, hazards=hazards,
                                            decision_index=int(decision))
    obs = np.asarray(feats, dtype=np.float32)
    if obs.shape[0] != OBS_DIM:
        raise BombRLError(f"the frozen observation has {obs.shape[0]} dims, "
                          f"expected {OBS_DIM}")
    explicit = explicit_features(state, hazards, decision, budget)
    augmented = np.concatenate([obs, np.asarray(explicit["features"],
                                                dtype=np.float32)])
    if augmented.shape[0] != FEATURE_DIM:
        raise BombRLError(f"augmented observation has {augmented.shape[0]} "
                          f"dims, expected {FEATURE_DIM}")
    return obs, augmented, explicit


def candidate_argmax(policy: Any, feats: Any) -> tuple[int, float, float]:
    """The frozen head read out deterministically (prereg §1)."""

    import torch

    with torch.no_grad():
        logits = policy(torch.from_numpy(feats).unsqueeze(0))[0]
        scores = logits.view(N_CAND, 2).max(dim=1).values
        cand = int(torch.argmax(scores).item())
        ordered = torch.sort(scores, descending=True).values
        best = float(ordered[0].item())
        second = float(ordered[1].item()) if ordered.numel() > 1 else best
    return cand, best, round(best - second, 6)


def rule_decision(state: dict[str, Any], hazards: Any) -> dict[str, Any]:
    """v8.20.0's published rule, called through the frozen runner."""

    return v8l0().rule_decision(state, hazards)


def new_audit() -> dict[str, Any]:
    return {"cand_counts": [0] * N_CAND, "bombs": 0, "cand_scores": [],
            "margins": [], "rule": {"distance_true_bombs_positive": 0,
                                    "distance_true_bombs_zero": 0,
                                    "distance_false": 0},
            "emit_nearest_px": [], "emit_threshold_px": [],
            "bomb_probs": [],
            # -- v8.28.0: the three explicit features (prereg §1.1).  The range
            # is tracked on **every** decision (not just the bombs) so the
            # [0, 1] assertion covers the whole episode.
            "feature_min": [float("inf")] * EXPLICIT_DIM,
            "feature_max": [float("-inf")] * EXPLICIT_DIM,
            "feature_decisions": 0, "no_bullets_decisions": 0,
            "first_bombs": [], "first_bombs_limit": 5}


def feature_range(audit: dict[str, Any]) -> dict[str, Any]:
    """The observed ``[min, max]`` of ``f1/f2/f3`` over one episode."""

    lo = [None if value == float("inf") else round(float(value), 6)
          for value in audit["feature_min"]]
    hi = [None if value == float("-inf") else round(float(value), 6)
          for value in audit["feature_max"]]
    inside = bool(audit["feature_decisions"] > 0
                  and all(value is not None and 0.0 <= value <= 1.0 for value in lo)
                  and all(value is not None and 0.0 <= value <= 1.0 for value in hi))
    return {"names": list(FEATURE_NAMES), "decisions": int(audit["feature_decisions"]),
            "min": dict(zip(("f1", "f2", "f3"), lo)),
            "max": dict(zip(("f1", "f2", "f3"), hi)),
            "all_in_unit_interval": inside,
            "no_bullets_decisions": int(audit["no_bullets_decisions"])}


def make_decide(*, policy: Any, head: Any, source: str, mode: str,
                budget: int,
                generator: Any = None, audit: dict[str, Any] | None = None,
                collect: list[tuple[Any, int]] | None = None) -> Any:
    """``decide(state, hazards, decision) -> (cand, bomb)``.

    ``source`` is ``"bomb_head"`` (the learned 2-action head) or ``"rule"`` (the
    published 60/40/20 rule, used by the BCE warm start only).  ``mode`` is
    ``"sample"`` (softmax sampling) or ``"argmax"`` (warm start / clone probe).
    ``collect`` records ``(features_109, bomb)`` for one REINFORCE/BC update;
    ``audit`` records the explicit features' range and the first 5 emissions.
    """

    import torch

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        obs, feats, explicit = features_of(state, hazards, decision, budget)
        cand, best, margin = candidate_argmax(policy, obs)
        if audit is not None:
            audit["feature_decisions"] += 1
            audit["no_bullets_decisions"] += int(explicit["no_bullets"])
            for index, value in enumerate(explicit["features"]):
                audit["feature_min"][index] = min(audit["feature_min"][index],
                                                  float(value))
                audit["feature_max"][index] = max(audit["feature_max"][index],
                                                  float(value))
        bomb_prob = None
        if source == "rule":
            rule = rule_decision(state, hazards)
            bomb = int(rule["bomb"])
            if audit is not None:
                key = int(rule["bombs_remaining"])
                if rule["distance_ok"]:
                    if key > 0:
                        audit["rule"]["distance_true_bombs_positive"] += 1
                        audit["emit_threshold_px"].append(
                            float(rule["threshold_px"]))
                        audit["emit_nearest_px"].append(
                            None if rule["nearest_all_px"] is None
                            else float(rule["nearest_all_px"]))
                    else:
                        audit["rule"]["distance_true_bombs_zero"] += 1
                else:
                    audit["rule"]["distance_false"] += 1
                audit["bomb_probs"].append(
                    None if rule["distance_ok"] else 0.0)
        else:
            with torch.no_grad():
                blogits = head(torch.from_numpy(feats).unsqueeze(0))[0]
                probs = torch.softmax(blogits, -1)
                bomb_prob = float(probs[1].item())
                if mode == "argmax":
                    bomb = int(torch.argmax(blogits).item())
                elif mode == "sample":
                    if generator is None:
                        bomb = int(torch.multinomial(probs, 1).item())
                    else:
                        bomb = int(torch.multinomial(probs, 1,
                                                     generator=generator).item())
                else:
                    raise BombRLError(f"unknown bomb mode {mode!r}")
            if audit is not None:
                audit["bomb_probs"].append(bomb_prob)
        if audit is not None and bomb and (len(audit["first_bombs"])
                                           < audit["first_bombs_limit"]):
            # the first 5 emissions: the feature triple *and* the raw nearest
            audit["first_bombs"].append({
                "decision": int(decision), "f1": round(explicit["f1"], 6),
                "f2": round(explicit["f2"], 6), "f3": round(explicit["f3"], 6),
                "nearest_all_px": explicit["nearest_all_px"],
                "bombs_left": explicit["bombs_left"],
                "decisions_left": explicit["decisions_left"]})
        if audit is not None:
            audit["cand_counts"][int(cand)] += 1
            audit["bombs"] += int(bomb)
            audit["cand_scores"].append(float(best))
            audit["margins"].append(float(margin))
        if collect is not None:
            collect.append((feats, int(bomb)))
        return int(cand), int(bomb)

    return decide


# --------------------------------------------------------------------------- #
# engine plumbing (delegated to the frozen modules)
# --------------------------------------------------------------------------- #
def engine_gate(path: str) -> dict[str, Any]:
    return dict(v450().check_engine(path))


def prepare_env() -> dict[str, Any]:
    return dict(v450().prepare_env())


def start_parent(*, slot: int, window: int, engine: str,
                 data_dir: pathlib.Path, log_dir: pathlib.Path
                 ) -> tuple[Any, dict[str, Any]]:
    return v450().start_parent(slot=int(slot), window=int(window),
                               engine=engine, data_dir=data_dir,
                               log_dir=log_dir)


def prepare_window_data(tag: str, window: int, source: str
                        ) -> tuple[pathlib.Path, dict[str, Any]]:
    data_dir, info = v450().prepare_data_dir(WORK / "data" / tag, int(window),
                                             source)
    if not info["cfg"]["muted_four_zero"]:
        raise BombRLError(f"w{window}: the generated th08.cfg is not muted: "
                          f"{info['cfg']}")
    return data_dir, info


def ensure_parent(*, parents: dict[int, Any], window: int, engine: str,
                  data_dir: pathlib.Path, log_dir: pathlib.Path,
                  slot: int) -> tuple[Any, dict[str, Any] | None]:
    """Health check: a parked parent may be restarted, never silently reused."""

    parent = parents.get(int(window))
    if parent is not None:
        try:
            if parent.alive():
                return parent, None
        except Exception:                                             # noqa: BLE001
            pass
        try:
            parent.close()
        except Exception:                                             # noqa: BLE001
            pass
        parents.pop(int(window), None)
    parent, gate = start_parent(slot=slot, window=int(window), engine=engine,
                                data_dir=data_dir, log_dir=log_dir)
    parents[int(window)] = parent
    return parent, gate


def close_parents(parents: dict[int, Any]) -> None:
    for window, parent in list(parents.items()):
        try:
            parent.close()
        except Exception:                                             # noqa: BLE001
            pass
        parents.pop(window, None)


# --------------------------------------------------------------------------- #
# one episode
# --------------------------------------------------------------------------- #
def run_episode(*, parent: Any, episode: int, seed: int, budget: int,
                policy: Any, head: Any, source: str, mode: str,
                generator: Any, tag: str,
                collect: list[tuple[Any, int]] | None = None
                ) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = new_audit()
    decide = make_decide(policy=policy, head=head, source=source, mode=mode,
                         budget=int(budget), generator=generator, audit=audit,
                         collect=collect)
    record = v830().run_episode36(parent, episode=int(episode),
                                  policy_seed=int(seed), decisions=int(budget),
                                  decide=decide, budget=int(budget),
                                  decision_frames=DECISION_FRAMES)
    probs = [p for p in audit["bomb_probs"] if p is not None] or [0.0]
    emit_ok = [bool(nearest is not None and threshold is not None
                    and float(nearest) < float(threshold))
               for nearest, threshold in zip(audit["emit_nearest_px"],
                                             audit["emit_threshold_px"])]
    feature_audit = feature_range(audit)
    row = {
        "schema": SCHEMA_EVAL_EPISODE, "producer": PRODUCER, "tag": tag,
        "arm": (ARM if source == "bomb_head" else "RULE-WARMSTART"),
        "budget": int(budget), "budget_decisions": int(budget),
        "bomb_source": source, "bomb_mode": mode,
        "candidate_readout": "argmax",
        "policy_seed": int(seed), "window": int(parent.window),
        "episode": int(episode), "decision_frames": DECISION_FRAMES,
        "prefix_frame": PREFIX_FRAMES,
        "survived": bool(record["survived"]),
        "alive_frames": record["alive_frames"], "frames": record["frames"],
        "decisions_executed": record["decisions_executed"],
        "decisions_measured": record["decisions_measured"],
        "return_decisions": record["return_decisions"],
        "terminal": record["terminal"],
        "first_death_frame": record["first_death_frame"],
        "first_death_offset": record["first_death_offset"],
        "bombs_start": record["bombs_start"], "bombs_end": record["bombs_end"],
        "bombs_used": record["bombs_used"], "bomb_emits": record["bomb_emits"],
        "bomb_decisions": record["bomb_decisions"],
        "first_bomb_decision": record["first_bomb_decision"],
        "last_bomb_decision": (max(record["bomb_decisions"])
                               if record["bomb_decisions"] else None),
        "bombs_remaining_at_emit": record["bombs_remaining_at_emit"],
        "nearest_px_at_emit": record["nearest_px_at_emit"],
        "bomb_cands": record["bomb_cands"], "bomb_ignored": record["bomb_ignored"],
        "min_nearest_px": record["min_nearest_px"],
        "actions_sampled": record["actions_sampled"],
        "invalid": bool(record["invalid"]), "error": record["error"],
        "wall_s": record["wall_s"], "spawn_ms": record["spawn_ms"],
        "parent_alive": record["parent_alive"],
        "cand_counts": audit["cand_counts"],
        "distinct_cands": sum(1 for count in audit["cand_counts"] if count),
        "cand_margins": quantiles(audit["margins"]),
        "bomb_prob": {"mean": round(statistics.fmean(probs), 6),
                      "min": round(min(probs), 6),
                      "max": round(max(probs), 6)},
        "decisions": sum(audit["cand_counts"]),
        "rule_counts": audit["rule"],
        "emit_threshold_px": audit["emit_threshold_px"],
        # -- v8.28.0: the explicit features (prereg §1.1 / §1.2 P3)
        "explicit_features": feature_audit,
        "first_bombs": audit["first_bombs"],
    }
    row["reward"] = round(reward_of(row["survived"],
                                    int(row["alive_frames"] or 0)), 6)
    row["checks"] = {
        "one_decision_per_measurement": bool(
            row["decisions"] == row["decisions_measured"]
            and row["decisions_measured"] in (row["decisions_executed"],
                                              row["decisions_executed"] + 1)),
        "counter_bookkeeping_closes": bool(
            int(row["bombs_start"]) - int(row["bombs_end"])
            == int(row["bombs_used"] or 0)),
        "frames_consistent": bool(
            int(row["frames"]) - int(row["decisions_executed"]) * DECISION_FRAMES
            in ((0,) if row["survived"] else (1, 2))),
        "bomb_emits_recorded": bool(
            len(row["bomb_decisions"]) == int(row["bomb_emits"])
            and int(row["bombs_used"] or 0) <= int(row["bomb_emits"])),
        "threshold_respected": bool(
            (not emit_ok) if source != "rule"
            else (len(emit_ok) == int(row["bomb_emits"]) and all(emit_ok))),
        # prereg §1.1: every f1/f2/f3 must be in [0, 1]
        "explicit_features_in_unit_interval": bool(
            feature_audit["all_in_unit_interval"]),
    }
    if not all(row["checks"].values()):
        row["invalid"] = True
        row["error"] = row["error"] or f"audit check failed: {row['checks']}"
    return row, record


# --------------------------------------------------------------------------- #
# mode: preflight
# --------------------------------------------------------------------------- #
def engine_census() -> dict[str, Any]:
    proc = subprocess.run(["pgrep", "-af", "th08-modern"],
                          capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    engines = [line for line in lines
               if "th08-modern" in line and "pgrep" not in line
               and "rl-role" in line]
    mine = [line for line in engines if ".tmp_v8u0_features" in line]
    return {"pgrep_returncode": proc.returncode,
            "engine_processes_total": len(engines),
            "other_round_processes": len(engines) - len(mine),
            "matching": len(mine), "lines_mine": mine,
            "lines_sample": engines[:3]}


def mode_preflight(args: argparse.Namespace) -> int:
    out_path = pathlib.Path(args.out) if args.out else WORK / "preflight.json"
    census = engine_census()
    previous = read_json(out_path)
    history = list(previous.get("history") or [])
    if previous.get("census"):
        history.append({"created_at": previous.get("created_at"),
                        "phase": previous.get("phase"),
                        "census": previous["census"],
                        "loadavg": previous.get("loadavg")})
    payload: dict[str, Any] = {
        "schema": SCHEMA_PREFLIGHT, "producer": PRODUCER,
        "status": "complete", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("count the engine processes before launching; the rule "
                        "is '> 6 -> wait 5 minutes' and 'never pkill -f another "
                        "round's processes'"),
        "phase": args.phase, "nproc": os.cpu_count(), "census": census,
        "history": history, "waited_s": float(args.waited_s or 0.0),
    }
    try:
        payload["loadavg"] = pathlib.Path("/proc/loadavg").read_text(
            encoding="utf-8").split()[:3]
    except OSError:
        payload["loadavg"] = None
    write_json(out_path, payload)
    log(f"preflight {args.phase}: engines={census['engine_processes_total']} "
        f"(own={census['matching']}, other={census['other_round_processes']}) "
        f"waited={payload['waited_s']}s -> {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# mode: assert-ckpt
# --------------------------------------------------------------------------- #
def mode_assert(args: argparse.Namespace) -> int:
    out_path = (pathlib.Path(args.out) if args.out
                else WORK / f"ckpt_assertions_{args.phase}.json")
    payload: dict[str, Any] = {
        "schema": SCHEMA_ASSERT, "producer": PRODUCER, "status": "failed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": args.phase,
        "requirement": ("the candidate head is the frozen v8.6.0 `bombdag.pt` "
                        "Policy(106,64,36) (9c6d5d4a / 36a51090 / 885dde70); the "
                        "behaviour policy is the frozen v4.5.0 student "
                        "(round2.pt = 4abee069 / a93746bb / c69960fc); the "
                        "engine (691bfd77) and the protected sources must not "
                        "have moved.  The **bomb** head is this round's own "
                        "Linear(109,64)->Tanh->Linear(64,2) and is recorded, not "
                        "asserted"),
        "engine": None, "v860_bombdag": {}, "v450_round2": {}, "protected": {},
        "bomb_heads": {}, "self": {}, "prereg": {}, "all_pass": False,
        "error": None,
    }
    try:
        payload["engine"] = engine_gate(args.engine)
        ok = bool(payload["engine"].get("match"))
        for seed in SEEDS:
            path = V860_ROOT / f"seed{seed}" / "bombdag.pt"
            expected = V860_SHA256[seed]
            digest = sha256_file(path) if path.is_file() else None
            match = bool(digest == expected)
            ok = ok and match
            payload["v860_bombdag"][str(seed)] = {
                "path": str(path), "exists": path.is_file(), "sha256": digest,
                "expected": expected, "expected_prefix": expected[:8],
                "match": match,
                "bytes": path.stat().st_size if path.is_file() else None}
        for seed in SEEDS:
            path = V450_ROOT / f"seed{seed}" / "round2.pt"
            expected = V450_SHA256[seed]
            digest = sha256_file(path) if path.is_file() else None
            match = bool(digest == expected)
            ok = ok and match
            payload["v450_round2"][str(seed)] = {
                "path": str(path), "exists": path.is_file(), "sha256": digest,
                "expected": expected, "expected_prefix": expected[:8],
                "match": match,
                "bytes": path.stat().st_size if path.is_file() else None}
        for rel in PROTECTED_FILES:
            path = REPO / rel
            payload["protected"][rel] = {
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None}
            ok = ok and path.is_file()
        for seed in SEEDS:
            path = head_ckpt_path(seed)
            payload["bomb_heads"][str(seed)] = {
                "path": str(path), "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None}
        me = pathlib.Path(__file__).resolve()
        payload["self"] = {"path": str(me), "sha256": sha256_file(me)}
        prereg = REPO / PREREG_REL
        payload["prereg"] = {"path": PREREG_REL,
                             "sha256": (sha256_file(prereg)
                                        if prereg.is_file() else None)}
        payload["all_pass"] = bool(ok)
        payload["status"] = "complete" if ok else "failed"
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
    write_json(out_path, payload)
    log(f"assert-ckpt {args.phase}: all_pass={payload['all_pass']} -> {out_path}")
    return 0 if payload["all_pass"] else 2


# --------------------------------------------------------------------------- #
# mode: d3 (the 5-assertion fidelity gate, through the frozen v432 script)
# --------------------------------------------------------------------------- #
def mode_d3(args: argparse.Namespace) -> int:
    out_path = pathlib.Path(args.out) if args.out else WORK / "d3_gate.json"
    raw_path = WORK / "fidelity_d3.json"
    tag = args.tag or "v8u0-fidelity"
    argv = [sys.executable, str(REPO / "tools" / "rl" / "v432_fork_arm.py"),
            "--mode", "fidelity", "--seeds", "12", "--learner", "frozen",
            "--work-dir", str(WORK / "work"), "--out", str(raw_path),
            "--tag", tag]
    started = time.perf_counter()
    proc = subprocess.run(argv, capture_output=True, text=True)
    wall = round(time.perf_counter() - started, 3)
    raw = read_json(raw_path)
    assertions = raw.get("assertions") or {}
    checks = {key: {"passed": bool(value.get("passed")),
                    "requirement": value.get("requirement")}
              for key, value in assertions.items()}
    payload: dict[str, Any] = {
        "schema": SCHEMA_D3, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": tag,
        "requirement": ("D3 fidelity, 5 assertions (FOR-001 §2.1-§2.5): HELLO "
                        "role=shadow, DEBUG threads=1, 2,165 prefix checkpoints "
                        "to frame 8663 with 0 mismatches, R24's 18 constants "
                        "18/18, PARK frame=8663.  A non-5/5 gate voids the round."),
        "argv": argv, "exit": proc.returncode, "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip()[-4000:], "wall_s": wall,
        "raw": str(raw_path),
        "raw_sha256": (sha256_file(raw_path) if raw_path.is_file() else None),
        "assertions_present": raw.get("assertions_present"), "checks": checks,
        "passed": sum(1 for value in checks.values() if value["passed"]),
        "total": len(checks), "verdict": raw.get("verdict"),
        "engine": raw.get("engine"), "data_dir": raw.get("data_dir"),
        "status": "failed",
    }
    payload["all_pass"] = bool(proc.returncode == 0
                               and payload["passed"] == 5
                               and payload["total"] == 5
                               and "PASS" in str(payload["verdict"]))
    payload["status"] = "complete" if payload["all_pass"] else "failed"
    write_json(out_path, payload)
    for key in sorted(checks):
        log(f"d3 {key}: {'PASS' if checks[key]['passed'] else 'FAIL'}")
    log(f"d3: {payload['passed']}/{payload['total']} PASS "
        f"all_pass={payload['all_pass']} wall={wall}s -> {out_path}")
    return 0 if payload["all_pass"] else 2


# --------------------------------------------------------------------------- #
# mode: pretrain (200 BCE steps on the 60/40/20 rule's own decisions)
# --------------------------------------------------------------------------- #
def collect_rule_dataset(*, seed: int, policy: Any, engine: str, tag: str,
                         data_source: str, episodes: int
                         ) -> tuple[list[tuple[Any, int]], list[dict[str, Any]],
                                    dict[int, dict[str, Any]]]:
    import numpy as np

    collected: list[tuple[Any, int]] = []
    outcomes: list[dict[str, Any]] = []
    gates: dict[int, dict[str, Any]] = {}
    parents: dict[int, Any] = {}
    try:
        for window in WINDOWS:
            data_dir, info = prepare_window_data(tag, window, data_source)
            parent, gate = ensure_parent(parents=parents, window=window,
                                         engine=engine, data_dir=data_dir,
                                         log_dir=WORK / "runs" / tag,
                                         slot=int(window))
            gates[int(window)] = {
                "hello_role_shadow": gate["hello_role_shadow"],
                "park_ok": gate["park_ok"], "park_frame": gate["park_frame"],
                "threads_1": gate["threads_1"], "all_pass": gate["all_pass"],
                "cfg_muted_four_zero": info["cfg"]["muted_four_zero"],
                "cfg_sha256": info["cfg"]["sha256"]}
            for episode in range(int(episodes)):
                record: list[tuple[Any, int]] = []
                row, raw = run_episode(parent=parent, episode=episode,
                                       seed=seed, budget=TRAIN_DECISIONS,
                                       policy=policy, head=None, source="rule",
                                       mode="argmax", generator=None, tag=tag,
                                       collect=record)
                collected.extend(record)
                outcomes.append({
                    "window": int(window), "episode": int(episode),
                    "survived": row["survived"],
                    "alive_frames": row["alive_frames"],
                    "decisions": len(record),
                    "bombs_used": row["bombs_used"],
                    "bomb_emits": row["bomb_emits"],
                    "positives": int(sum(label for _f, label in record)),
                    "wall_s": row["wall_s"], "invalid": row["invalid"]})
                log(f"  BC w{window} ep{episode}: states={len(record)} "
                    f"positives={outcomes[-1]['positives']} "
                    f"bombs_used={row['bombs_used']} wall={row['wall_s']}s")
    finally:
        close_parents(parents)
    return collected, outcomes, gates


def mode_pretrain(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import torch.nn as nn

    seed = int(args.policy_seed)
    tag = str(args.tag or "").strip()
    if not tag:
        raise BombRLError("--tag is mandatory in every mode that starts an engine")
    out_path = (pathlib.Path(args.out) if args.out
                else WORK / "pretrain" / f"seed{seed}.json")
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    payload: dict[str, Any] = {
        "schema": SCHEMA_PRETRAIN, "producer": PRODUCER, "status": "failed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": tag,
        "requirement": ("warm-start BombHead(109->64->2) [obs_106 + f1/f2/f3] "
                        "from the 60/40/20 "
                        "rule's own decisions (1 episode per window x 4 windows, "
                        "900 decisions each) with 200 BCE steps (prereg §1)"),
        "policy_seed": seed, "decisions": TRAIN_DECISIONS,
        "ckpt": {"path": str(ckpt), "sha256": digest,
                 "bytes": ckpt.stat().st_size},
        "engine": engine, "launch_env": launch_env,
        "bc": {"steps": BC_STEPS, "batch": BC_BATCH, "lr": BC_LR,
               "pos_fraction": BC_POS_FRACTION,
               "episodes_per_window": BC_EPISODES_PER_WINDOW},
        "dataset": None, "outcomes": [], "gates": {}, "bce": None,
        "clone_probe": None, "head": None, "error": None, "wall_s": None,
    }
    started = time.perf_counter()
    parents: dict[int, Any] = {}
    try:
        collected, outcomes, gates = collect_rule_dataset(
            seed=seed, policy=policy, engine=args.engine, tag=tag,
            data_source=args.data_source,
            episodes=BC_EPISODES_PER_WINDOW)
        payload["outcomes"] = outcomes
        payload["gates"] = {str(k): v for k, v in gates.items()}
        if not all(value["all_pass"] for value in gates.values()):
            raise BombRLError(f"fidelity gate failed on this tag: {gates}")
        features = np.stack([feat for feat, _label in collected])
        labels = np.asarray([label for _feat, label in collected], dtype=np.float32)
        positives = int(labels.sum())
        payload["dataset"] = {
            "states": int(labels.size), "positives": positives,
            "negatives": int(labels.size) - positives,
            "positive_rate": round(positives / max(int(labels.size), 1), 6),
            "features_shape": list(features.shape),
            "episodes": len(outcomes),
            "survived_episodes": sum(1 for o in outcomes if o["survived"]),
            "windows": sorted({o["window"] for o in outcomes})}
        if labels.size < 500 or positives < 10:
            raise BombRLError(f"BC dataset too small / too imbalanced: "
                              f"{payload['dataset']}")
        head = make_bomb_head(seed)
        opt = torch.optim.Adam(head.parameters(), lr=BC_LR)
        loss_fn = nn.BCEWithLogitsLoss()
        pos_idx = np.flatnonzero(labels > 0.5)
        neg_idx = np.flatnonzero(labels <= 0.5)
        rng = random.Random(int(seed) * 1000 + GEN_CONST)
        n_pos = int(round(BC_BATCH * BC_POS_FRACTION))
        n_neg = BC_BATCH - n_pos
        feature_t = torch.from_numpy(features)
        label_flat = torch.from_numpy(labels)
        label_t = label_flat.unsqueeze(1)
        history: list[dict[str, Any]] = []
        head.train()
        for step in range(BC_STEPS):
            batch_pos = [int(pos_idx[rng.randrange(len(pos_idx))])
                         for _ in range(n_pos)]
            batch_neg = [int(neg_idx[rng.randrange(len(neg_idx))])
                         for _ in range(n_neg)]
            index = torch.tensor(batch_pos + batch_neg, dtype=torch.long)
            # two logits == one logit for a binary target: the "bomb" logit
            logits = head(feature_t.index_select(0, index))[:, 1]
            loss = loss_fn(logits, label_flat.index_select(0, index))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
            opt.step()
            if step % 20 == 0 or step == BC_STEPS - 1:
                history.append({"step": step, "loss": round(float(loss.item()), 6)})
        head.eval()
        with torch.no_grad():
            full_logits = head(feature_t)
            probs = torch.softmax(full_logits, -1)[:, 1]
            pred = (probs > 0.5).float()
            accuracy = float((pred == label_flat).float().mean().item())
            argmax_bomb = float((full_logits.argmax(dim=1) == 1).float().mean()
                                .item())
            agreement = float((pred == label_flat).float().mean().item())
        payload["bce"] = {
            "history": history, "final_loss": history[-1]["loss"],
            "dataset_accuracy_at_0.5": round(accuracy, 6),
            "argmax_bomb_rate": round(argmax_bomb, 6),
            "rule_positive_rate": payload["dataset"]["positive_rate"],
            "mean_predicted_p": round(float(probs.mean().item()), 6),
            "agreement_with_rule": round(agreement, 6)}
        path = bc_ckpt_path(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), str(path))
        payload["head"] = {"path": str(path), "sha256": sha256_file(path),
                           "bytes": path.stat().st_size,
                           "shape": "Linear(109,64)->Tanh->Linear(64,2)"}
        log(f"BC seed{seed}: steps={BC_STEPS} loss={history[-1]['loss']} "
            f"argmax_bomb_rate={argmax_bomb:.4f} "
            f"rule_pos_rate={payload['dataset']['positive_rate']:.4f} -> {path}")
        # -- clone probe: does the warm start behave like the rule?
        probe: list[dict[str, Any]] = []
        for window in WINDOWS:
            data_dir, _info = prepare_window_data(tag, window, args.data_source)
            parent, gate = ensure_parent(parents=parents, window=window,
                                         engine=args.engine, data_dir=data_dir,
                                         log_dir=WORK / "runs" / tag,
                                         slot=int(window))
            row, _raw = run_episode(parent=parent, episode=0, seed=seed,
                                    budget=TRAIN_DECISIONS, policy=policy,
                                    head=head, source="bomb_head", mode="argmax",
                                    generator=None, tag=tag)
            probe.append({"window": int(window), "survived": row["survived"],
                          "alive_frames": row["alive_frames"],
                          "bombs_used": row["bombs_used"],
                          "bomb_emits": row["bomb_emits"],
                          "first_bomb_decision": row["first_bomb_decision"],
                          "invalid": row["invalid"],
                          "gate_all_pass": gate["all_pass"] if gate else True})
            log(f"  probe w{window}: survived={row['survived']} "
                f"bombs_used={row['bombs_used']} alive={row['alive_frames']}")
        payload["clone_probe"] = {
            "episodes": probe,
            "survived": sum(1 for p in probe if p["survived"]),
            "bombs_used_mean": round(statistics.fmean(
                [p["bombs_used"] or 0 for p in probe]), 4) if probe else None}
        payload["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
        log(f"PRETRAIN FAILED seed{seed}: {payload['error']}")
    finally:
        close_parents(parents)
        payload["wall_s"] = round(time.perf_counter() - started, 3)
    write_json(out_path, payload)
    log(f"pretrain seed{seed}: status={payload['status']} "
        f"wall={payload['wall_s']}s -> {out_path}")
    return 0 if payload["status"] == "complete" else 4


# --------------------------------------------------------------------------- #
# mode: train (REINFORCE on the bomb head, terminal return only)
# --------------------------------------------------------------------------- #
def reinforce_update(*, head: Any, opt: Any, record: list[tuple[Any, int]],
                     reward: float, baseline: float | None
                     ) -> dict[str, Any]:
    """One REINFORCE update; ``loss = -(mean_t log pi) * adv - 0.01 * mean H``."""

    import numpy as np
    import torch

    feats = torch.from_numpy(np.stack([f for f, _b in record]))
    actions = torch.tensor([int(b) for _f, b in record], dtype=torch.long)
    logits = head(feats)
    logp = torch.log_softmax(logits, dim=-1)
    picked = logp.gather(1, actions.view(-1, 1)).squeeze(1)
    entropy = -(logp.exp() * logp).sum(dim=-1)
    adv = float(reward - (baseline if baseline is not None else 0.0))
    loss = -(picked.mean() * adv) - ENTROPY_COEF * entropy.mean()
    opt.zero_grad()
    loss.backward()
    grad_sq = 0.0
    finite = True
    max_grad = 0.0
    for param in head.parameters():
        if param.grad is None:
            continue
        value = param.grad.detach()
        if not bool(torch.isfinite(value).all().item()):
            finite = False
        grad_sq += float((value * value).sum().item())
        max_grad = max(max_grad, float(value.abs().max().item()))
    torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
    opt.step()
    return {"loss": round(float(loss.item()), 6), "adv": round(adv, 6),
            "logp_mean": round(float(picked.mean().item()), 6),
            "entropy_mean": round(float(entropy.mean().item()), 6),
            "grad_norm": round(math.sqrt(grad_sq), 6),
            "grad_max_abs": round(max_grad, 6), "grad_finite": bool(finite)}


def mode_train(args: argparse.Namespace) -> int:
    import torch

    seed = int(args.policy_seed)
    episodes = int(args.episodes)
    tag = str(args.tag or "").strip()
    if not tag:
        raise BombRLError("--tag is mandatory in every mode that starts an engine")
    if seed not in SEEDS:
        raise BombRLError(f"--policy-seed must be one of {SEEDS}, got {seed}")
    smoke = bool(getattr(args, "smoke", False))
    full_run = bool(episodes == TRAIN_EPISODES)
    suffix = "" if full_run else f"_probe{episodes}"
    out_path = (pathlib.Path(args.out) if args.out
                else (WORK / "train_smoke.json" if smoke
                      else WORK / "train" / f"seed{seed}{suffix}.json"))
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    init_path = bc_ckpt_path(seed)
    head = make_bomb_head(seed)
    init_used = None
    if init_path.is_file():
        head.load_state_dict(torch.load(str(init_path), map_location="cpu",
                                        weights_only=True))
        init_used = {"path": str(init_path), "sha256": sha256_file(init_path)}
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    payload: dict[str, Any] = {
        "schema": SCHEMA_TRAIN_SMOKE if smoke else SCHEMA_TRAIN,
        "producer": PRODUCER, "status": "failed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": tag,
        "requirement": ("REINFORCE, terminal return only: R = survived(900) + "
                        "0.5 * alive_frames / 1800; moving-average baseline; "
                        "entropy 0.01; Adam 3e-4; 300 episodes x 900 decisions"),
        "policy_seed": seed, "episodes": episodes,
        "decisions": TRAIN_DECISIONS, "frames": TRAIN_FRAMES,
        "windows": list(WINDOWS), "loop": "window = WINDOWS[episode % 4]",
        "reward": {"survived": REWARD_SURVIVED,
                   "alive_weight": REWARD_ALIVE_WEIGHT,
                   "alive_frames_norm": ALIVE_FRAMES_NORM},
        "optimizer": {"name": "Adam", "lr": LR,
                      "baseline_ema": BASELINE_EMA,
                      "entropy_coef": ENTROPY_COEF, "grad_clip": GRAD_CLIP},
        "candidate": {"readout": "argmax", "frozen": True},
        "ckpt": {"path": str(ckpt), "sha256": digest},
        "engine": engine, "launch_env": launch_env, "init": init_used,
        "head": None, "episodes_run": [], "baseline_final": None,
        "gradients": {}, "error": None, "wall_s": None,
    }
    started = time.perf_counter()
    parents: dict[int, Any] = {}
    gates: dict[int, dict[str, Any]] = {}
    baseline: float | None = None
    rows_path = (train_rows_path(seed) if full_run
                 else WORK / "train" / f"seed{seed}{suffix}.jsonl")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    if rows_path.exists():
        rows_path.unlink()
    data_dirs: dict[int, pathlib.Path] = {}
    try:
        for window in WINDOWS:
            data_dir, info = prepare_window_data(tag, window, args.data_source)
            data_dirs[int(window)] = data_dir
            parent, gate = ensure_parent(parents=parents, window=window,
                                         engine=args.engine, data_dir=data_dir,
                                         log_dir=WORK / "runs" / tag,
                                         slot=int(window))
            gates[int(window)] = {
                "hello_role_shadow": gate["hello_role_shadow"],
                "park_ok": gate["park_ok"], "park_frame": gate["park_frame"],
                "threads_1": gate["threads_1"], "all_pass": gate["all_pass"],
                "cfg_muted_four_zero": info["cfg"]["muted_four_zero"]}
        payload["gates"] = {str(k): v for k, v in gates.items()}
        if not all(value["all_pass"] for value in gates.values()):
            raise BombRLError(f"fidelity gate failed on this tag: {gates}")
        finite_all = True
        for episode in range(episodes):
            window = int(WINDOWS[episode % len(WINDOWS)])
            parent, gate = ensure_parent(parents=parents, window=window,
                                         engine=args.engine,
                                         data_dir=data_dirs[window],
                                         log_dir=WORK / "runs" / tag,
                                         slot=window)
            if gate is not None:
                gates[window] = {
                    "hello_role_shadow": gate["hello_role_shadow"],
                    "park_ok": gate["park_ok"], "park_frame": gate["park_frame"],
                    "threads_1": gate["threads_1"], "all_pass": gate["all_pass"]}
            record: list[tuple[Any, int]] = []
            row, raw = run_episode(parent=parent, episode=episode, seed=seed,
                                   budget=TRAIN_DECISIONS, policy=policy,
                                   head=head, source="bomb_head", mode="sample",
                                   generator=generator, tag=tag, collect=record)
            if row["invalid"]:
                raise BombRLError(f"episode {episode} invalid: {row['error']}")
            reward = reward_of(row["survived"],
                               int(row["alive_frames"] or 0))
            update = (reinforce_update(head=head, opt=opt, record=record,
                                       reward=reward, baseline=baseline)
                      if record else {"loss": None})
            finite_all = finite_all and bool(update.get("grad_finite", True))
            baseline = (reward if baseline is None
                        else BASELINE_EMA * baseline
                        + (1.0 - BASELINE_EMA) * reward)
            entry = {
                "episode": episode, "window": window,
                "survived": row["survived"],
                "alive_frames": row["alive_frames"],
                "decisions_executed": row["decisions_executed"],
                "bombs_used": row["bombs_used"],
                "bomb_emits": row["bomb_emits"],
                "bombs_remaining_at_emit": row["bombs_remaining_at_emit"],
                "reward": round(reward, 6),
                "advantage": update.get("adv"),
                "loss": update.get("loss"),
                "logp_mean": update.get("logp_mean"),
                "entropy_mean": update.get("entropy_mean"),
                "grad_norm": update.get("grad_norm"),
                "grad_max_abs": update.get("grad_max_abs"),
                "grad_finite": update.get("grad_finite"),
                "baseline_after": round(float(baseline), 6),
                "wall_s": row["wall_s"], "tag": tag}
            if smoke:
                # the smoke row carries the feature audit (prereg §1.1) and the
                # first 5 emissions' f1/f2/f3 + nearest (the work order's smoke)
                entry["explicit_features"] = row["explicit_features"]
                entry["first_bombs"] = row["first_bombs"]
                entry["head_input_dim"] = FEATURE_DIM
            payload["episodes_run"].append(entry)
            if not smoke:
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if episode % 10 == 0 or episode == episodes - 1:
                log(f"  train ep{episode} w{window}: R={reward:.4f} "
                    f"survived={row['survived']} bombs_used={row['bombs_used']} "
                    f"alive={row['alive_frames']} adv={update.get('adv')} "
                    f"entropy={update.get('entropy_mean')} "
                    f"grad={update.get('grad_norm')} wall={row['wall_s']}s")
        payload["baseline_final"] = (round(float(baseline), 6)
                                     if baseline is not None else None)
        rewards = [e["reward"] for e in payload["episodes_run"]]
        payload["summary"] = {
            "episodes": len(rewards),
            "reward_mean": round(statistics.fmean(rewards), 6) if rewards else None,
            "survived": sum(1 for e in payload["episodes_run"] if e["survived"]),
            "bombs_used_mean": round(statistics.fmean(
                [e["bombs_used"] or 0 for e in payload["episodes_run"]]), 4)
            if rewards else None,
            "entropy_first": payload["episodes_run"][0]["entropy_mean"]
            if payload["episodes_run"] else None,
            "entropy_last": payload["episodes_run"][-1]["entropy_mean"]
            if payload["episodes_run"] else None,
            "grad_finite_all": bool(finite_all),
            "grad_norm_mean": round(statistics.fmean(
                [e["grad_norm"] for e in payload["episodes_run"]
                 if e["grad_norm"] is not None]), 6)
            if payload["episodes_run"] else None}
        payload["gradients"] = {
            "finite_all": bool(finite_all),
            "nan_or_inf_episodes": [e["episode"] for e in payload["episodes_run"]
                                    if not e["grad_finite"]]}
        path = (head_ckpt_path(seed) if full_run else
                ckpt_dir() / f"seed{seed}" / f"bomb_head_seed{seed}{suffix}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), str(path))
        payload["head"] = {"path": str(path), "sha256": sha256_file(path),
                           "bytes": path.stat().st_size,
                           "shape": "Linear(109,64)->Tanh->Linear(64,2)"}
        payload["status"] = "complete"
        log(f"train seed{seed}: episodes={len(rewards)} "
            f"reward_mean={payload['summary']['reward_mean']} "
            f"survived={payload['summary']['survived']} "
            f"grad_finite={finite_all} -> {path}")
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
        log(f"TRAIN FAILED seed{seed}: {payload['error']}")
    finally:
        close_parents(parents)
        payload["wall_s"] = round(time.perf_counter() - started, 3)
    write_json(out_path, payload)
    ok = bool(payload["status"] == "complete")
    if smoke:
        first = payload["episodes_run"][0] if payload["episodes_run"] else {}
        features = first.get("explicit_features") or {}
        first_bombs = first.get("first_bombs") or []
        head_in = None
        try:
            saved = (payload.get("head") or {}).get("path")
            if saved and pathlib.Path(saved).is_file():
                head_in = int(torch.load(saved, map_location="cpu",
                                         weights_only=True)["0.weight"].shape[1])
        except Exception:                                             # noqa: BLE001
            head_in = None
        checks = {
            "episodes_expected": len(payload["episodes_run"]) == episodes,
            "return_nonzero": bool((first.get("reward") or 0.0) > 0.0),
            # prereg §1: R = survived(900) + 0.5 * alive/1800 in [0, 1.5],
            # and the row's own reward must be exactly the arm's formula
            "return_in_arm_range": bool(
                0.0 <= float(first.get("reward") or -1.0)
                <= REWARD_BY_ARM[ARM][0] + REWARD_BY_ARM[ARM][1] + 1e-9),
            "return_matches_arm_formula": bool(
                abs(float(first.get("reward") or -1.0)
                    - reward_of(bool(first.get("survived")),
                                int(first.get("alive_frames") or 0))) < 1e-6),
            "alive_frames_positive": bool((first.get("alive_frames") or 0) > 0),
            "grad_finite": bool(payload["gradients"].get("finite_all")),
            "grad_nonzero": bool((first.get("grad_norm") or 0.0) > 0.0),
            # -- v8.28.0: the explicit features (prereg §1.1) -------------------
            "head_input_is_109": bool(head_in == FEATURE_DIM == 109),
            "features_audited": bool(int(features.get("decisions") or 0) > 0),
            "features_cover_every_decision": bool(
                int(features.get("decisions") or 0)
                >= int(first.get("decisions_executed") or 0)),
            "features_in_unit_interval": bool(
                features.get("all_in_unit_interval")),
            "first_bombs_logged": bool(len(first_bombs) >= 1),
            "cfg_muted_four_zero": all(
                value.get("cfg_muted_four_zero") for value in
                payload.get("gates", {}).values()) if payload.get("gates") else False,
            "gate_all_pass": bool(payload.get("gates")) and all(
                value.get("all_pass") for value in payload["gates"].values()),
            "head_saved": payload["head"] is not None,
        }
        payload["checks"] = checks
        payload["first_bombs"] = first_bombs
        payload["explicit_features"] = features
        payload["head_input_dim"] = head_in
        payload["all_pass"] = bool(all(checks.values()))
        write_json(out_path, payload)
        log(f"train-smoke: all_pass={payload['all_pass']} checks={checks}")
        log(f"  smoke R={first.get('reward')} adv={first.get('advantage')} "
            f"grad_norm={first.get('grad_norm')} "
            f"entropy={first.get('entropy_mean')} "
            f"bombs_used={first.get('bombs_used')} "
            f"alive={first.get('alive_frames')} "
            f"survived={first.get('survived')} head_in={head_in}")
        log(f"  feature range f1={features.get('min', {}).get('f1')}"
            f"..{features.get('max', {}).get('f1')} "
            f"f2={features.get('min', {}).get('f2')}"
            f"..{features.get('max', {}).get('f2')} "
            f"f3={features.get('min', {}).get('f3')}"
            f"..{features.get('max', {}).get('f3')} "
            f"decisions={features.get('decisions')}")
        for index, bomb in enumerate(first_bombs, start=1):
            log(f"  bomb#{index}: decision={bomb['decision']} "
                f"f1={bomb['f1']} f2={bomb['f2']} f3={bomb['f3']} "
                f"nearest={bomb['nearest_all_px']} "
                f"bombs_left={bomb['bombs_left']} "
                f"decisions_left={bomb['decisions_left']}")
        ok = payload["all_pass"]
    log(f"train seed{seed}: status={payload['status']} "
        f"wall={payload['wall_s']}s -> {out_path}")
    return 0 if ok else 4


# --------------------------------------------------------------------------- #
# mode: eval (one budget x one seed: 4 windows x --episodes)
# --------------------------------------------------------------------------- #
def mode_eval(args: argparse.Namespace) -> int:
    import torch

    arm = str(args.arm)
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    episodes = int(args.episodes)
    tag = str(args.tag or "").strip()
    if not tag:
        raise BombRLError("--tag is mandatory: every eval process needs its own "
                          "data directory (v8.17.0 lost 10 of 18 cells to a "
                          "shared one)")
    if arm != ARM:
        raise BombRLError(f"this round runs only the {ARM} arm (softmax sampling "
                          f"of the trained bomb head); there is no deterministic "
                          f"arm, got --arm {arm!r}")
    if seed not in SEEDS:
        raise BombRLError(f"--policy-seed must be one of {SEEDS}, got {seed}")
    if budget not in EVAL_BUDGETS:
        raise BombRLError(f"this round fixes --decisions to {EVAL_BUDGETS}, got "
                          f"{budget}")
    if episodes != EVAL_EPISODES:
        raise BombRLError(f"prereg §1 fixes {arm} to {EVAL_EPISODES} episodes per "
                          f"window, got {episodes}")
    windows = tuple(int(part) for part in str(args.windows).split(",")
                    if part.strip())
    for window in windows:
        if window not in WINDOWS:
            raise BombRLError(f"window {window} is not one of {WINDOWS}")
    source = "bomb_head"
    mode = "sample"
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    head = None
    head_info = None
    path = head_ckpt_path(seed)
    head = load_bomb_head(seed, path)
    head_info = {"path": str(path), "sha256": sha256_file(path)}
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    out_dir = WORK / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    progress: dict[str, Any] = {
        "schema": SCHEMA_EVAL, "producer": PRODUCER, "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "arm": arm,
        "bomb_source": source, "bomb_mode": mode, "candidate_readout": "argmax",
        "budget": budget, "budget_decisions": budget,
        "episodes_per_window": episodes, "episodes_total": episodes,
        "episodes_grand_total": episodes * len(windows),
        "windows": list(windows), "policy_seed": seed, "tag": tag,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest}, "bomb_head": head_info,
        "rule": {"name": RULE_NAME, "label": RULE_LABEL,
                 "thresholds": {str(k): float(v)
                                for k, v in v8l0().thresholds().items()}},
        "generator": ({"seed": generator_seed(seed),
                       "kind": "one torch.Generator per process, consumed once "
                               "per decision across every window and episode"}
                      if generator is not None else None),
        "cells": {}, "error": None, "wall_s": None,
    }
    progress_path_ = progress_path(arm, budget, seed, windows)
    started = time.perf_counter()
    parents: dict[int, Any] = {}
    try:
        for window in windows:
            out_path = eval_path(arm, budget, window, seed)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data_dir, info = prepare_window_data(tag, window, args.data_source)
            parent, gate = ensure_parent(parents=parents, window=window,
                                         engine=args.engine, data_dir=data_dir,
                                         log_dir=WORK / "runs" / tag,
                                         slot=int(window))
            log(f"{arm} d{budget} s{seed} w{window}: cfg muted="
                f"{info['cfg']['muted_four_zero']} tag={tag} data_dir={data_dir}")
            rows: list[dict[str, Any]] = []
            started_window = time.perf_counter()
            with out_path.open("w", encoding="utf-8") as handle:
                for episode in range(episodes):
                    parent, restart_gate = ensure_parent(
                        parents=parents, window=window, engine=args.engine,
                        data_dir=data_dir, log_dir=WORK / "runs" / tag,
                        slot=int(window))
                    row, _raw = run_episode(parent=parent, episode=episode,
                                            seed=seed, budget=budget,
                                            policy=policy, head=head,
                                            source=source, mode=mode,
                                            generator=generator, tag=tag)
                    row["ckpt_sha256"] = digest
                    row["engine_sha256"] = engine.get("sha256")
                    row["cfg_sha256"] = info["cfg"]["sha256"]
                    row["cfg_muted_four_zero"] = info["cfg"]["muted_four_zero"]
                    row["bomb_head_sha256"] = (head_info or {}).get("sha256")
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    rows.append(row)
                    if row["invalid"]:
                        raise BombRLError(f"{arm} s{seed} w{window} ep{episode} "
                                          f"invalid: {row['error']}")
                    if episode % 25 == 0 or episode == episodes - 1:
                        log(f"  {arm} d{budget} s{seed} w{window} ep{episode}: "
                            f"survived={row['survived']} "
                            f"bombs_used={row['bombs_used']} "
                            f"alive={row['alive_frames']} "
                            f"wall={row['wall_s']}s")
            survived = sum(1 for row in rows if row["survived"])
            progress["cells"][str(window)] = {
                "window": int(window), "episodes": len(rows),
                "survived": survived,
                "rate": round(survived / len(rows), 6) if rows else None,
                "bombs_used_total": sum(int(row["bombs_used"] or 0)
                                        for row in rows),
                "bombs_used_mean": round(statistics.fmean(
                    [int(row["bombs_used"] or 0) for row in rows]), 4)
                if rows else None,
                "cfg_muted_four_zero": info["cfg"]["muted_four_zero"],
                "data_dir": str(data_dir),
                "gate": {key: gate[key] for key in (
                    "hello_role_shadow", "park_ok", "park_frame", "threads_1",
                    "all_pass")},
                "gate_prefix": {key: gate["prefix"][key] for key in (
                    "hash_checks", "hash_mismatches", "final_frame")},
                "wall_s": round(time.perf_counter() - started_window, 3),
                "out": str(out_path), "out_sha256": sha256_file(out_path),
                "out_bytes": out_path.stat().st_size}
            progress["status"] = "partial"
            progress["wall_s"] = round(time.perf_counter() - started, 3)
            write_json(progress_path_, progress)
            log(f"{arm} d{budget} s{seed} w{window}: {survived}/{len(rows)} "
                f"survived ({progress['cells'][str(window)]['rate']})")
        progress["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        progress["error"] = f"{type(exc).__name__}: {exc}"
        progress["status"] = "failed"
        log(f"EVAL FAILED {arm} d{budget} s{seed}: {progress['error']}")
    finally:
        close_parents(parents)
        progress["wall_s"] = round(time.perf_counter() - started, 3)
        write_json(progress_path_, progress)
    log(f"eval {arm} d{budget} s{seed}: status={progress['status']} wall="
        f"{progress['wall_s']}s")
    return 0 if progress["status"] == "complete" else 4


# --------------------------------------------------------------------------- #
# mode: eval-smoke (one episode; asserts the configured total is 300)
# --------------------------------------------------------------------------- #
def mode_eval_smoke(args: argparse.Namespace) -> int:
    import torch

    arm = str(args.arm or ARM)
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    window = 11
    tag = str(args.tag or "").strip()
    if not tag:
        raise BombRLError("--tag is mandatory in every mode that starts an engine")
    if arm != ARM:
        raise BombRLError(f"this round runs only the {ARM} arm, got {arm!r}")
    source = "bomb_head"
    mode = "sample"
    out_path = pathlib.Path(args.out) if args.out else WORK / "eval_smoke.json"
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    path = head_ckpt_path(seed)
    head = load_bomb_head(seed, path)
    head_info = {"path": str(path), "sha256": sha256_file(path)}
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    configured_total = EVAL_EPISODES
    payload: dict[str, Any] = {
        "schema": SCHEMA_EVAL_SMOKE, "producer": PRODUCER, "status": "failed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": tag,
        "requirement": ("one episode of the arm + confirmation that the eval "
                        "plan asks for 150 episodes per window"),
        "arm": arm, "bomb_source": source, "bomb_mode": mode,
        "candidate_readout": "argmax", "policy_seed": seed, "window": window,
        "budget": budget, "windows_planned": list(WINDOWS),
        "episodes_total": configured_total,
        "episodes_grand_total": configured_total * len(WINDOWS),
        "episodes_run": 1, "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest}, "bomb_head": head_info,
        "row": None, "checks": {}, "all_pass": False, "cfg": None, "gate": None,
        "inspect": None, "error": None, "wall_s": None}
    parent: Any = None
    started = time.perf_counter()
    try:
        data_dir, info = prepare_window_data(tag, window, args.data_source)
        payload["cfg"] = info["cfg"]
        payload["data_dir"] = str(data_dir)
        maker = REPO / "tools" / "rl" / "v400_make_cfg.py"
        inspect = subprocess.run([sys.executable, str(maker), "--inspect",
                                  str(data_dir / "th08.cfg"), "--json"],
                                 capture_output=True, text=True)
        payload["inspect"] = {"exit": inspect.returncode,
                              "stdout": inspect.stdout.strip()[:2000],
                              "stderr": inspect.stderr.strip()[-2000:]}
        parent, gate = start_parent(slot=0, window=window, engine=args.engine,
                                    data_dir=data_dir,
                                    log_dir=WORK / "runs" / tag)
        payload["gate"] = {key: gate[key] for key in (
            "hello_role_shadow", "park_ok", "park_frame", "threads_1",
            "all_pass")}
        payload["gate_prefix"] = {key: gate["prefix"][key] for key in (
            "hash_checks", "hash_mismatches", "final_frame")}
        row, _raw = run_episode(parent=parent, episode=0, seed=seed,
                                budget=budget, policy=policy, head=head,
                                source=source, mode=mode, generator=generator,
                                tag=tag)
        payload["row"] = row
        decoded = json.loads(payload["inspect"]["stdout"]) if (
            payload["inspect"]["exit"] == 0
            and payload["inspect"]["stdout"].strip()) else {}
        inspected = decoded.get("fields", decoded) if decoded else {}
        payload["inspect"]["four_fields"] = {
            key: inspected.get(key)
            for key in ("musicMode", "playSounds", "musicVolume", "sfxVolume")}
        payload["checks"] = {
            "episodes_total_is_150": bool(configured_total == EVAL_EPISODES),
            "episodes_total_configured": configured_total,
            "cfg_muted_four_zero": bool(info["cfg"]["muted_four_zero"]),
            "inspect_four_zero": bool(
                inspected.get("musicMode") == 0
                and inspected.get("playSounds") == 0
                and inspected.get("musicVolume") == 0
                and inspected.get("sfxVolume") == 0),
            "gate_all_pass": bool(payload["gate"]["all_pass"]),
            "not_invalid": bool(not row["invalid"]),
            "decisions_measured": int(row["decisions_measured"]),
            "audit_checks_all_pass": bool(all(row["checks"].values())),
        }
        payload["all_pass"] = bool(
            payload["checks"]["episodes_total_is_150"]
            and payload["checks"]["cfg_muted_four_zero"]
            and payload["checks"]["inspect_four_zero"]
            and payload["checks"]["gate_all_pass"]
            and payload["checks"]["not_invalid"]
            and payload["checks"]["audit_checks_all_pass"]
            and int(row["decisions_measured"]) >= 1)
        payload["status"] = "complete" if payload["all_pass"] else "failed"
        log(f"EVAL-SMOKE {arm} d{budget} s{seed} w{window}: "
            f"survived={row['survived']} bombs_used={row['bombs_used']} "
            f"alive={row['alive_frames']} decisions={row['decisions_executed']} "
            f"episodes_total={configured_total} "
            f"checks={payload['checks']}")
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
        log(f"EVAL-SMOKE FAILED: {payload['error']}")
    finally:
        if parent is not None:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        payload["wall_s"] = round(time.perf_counter() - started, 3)
    write_json(out_path, payload)
    log(f"eval-smoke: all_pass={payload['all_pass']} wall={payload['wall_s']}s "
        f"-> {out_path}")
    return 0 if payload["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: summary
# --------------------------------------------------------------------------- #
def distinct_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """prereg 搂2 P4: an outcome is distinct on exactly this 4-tuple.

    ``(alive_frames, bombs_used, bomb_emits, tuple(bomb_decisions))`` 鈥?the
    window and the seed are **not** part of the key, so a value above one means
    the arm really produced different rollouts inside the same cell.
    """

    return (int(row["alive_frames"] or 0), int(row["bombs_used"] or 0),
            int(row["bomb_emits"] or 0),
            tuple(int(v) for v in (row.get("bomb_decisions") or [])))


def distinct_count(rows: Sequence[dict[str, Any]]) -> int:
    return len({distinct_key(row) for row in rows})


def distinct_with_window(rows: Sequence[dict[str, Any]]) -> int:
    return len({(int(row["window"]), int(row["policy_seed"])) + distinct_key(row)
                for row in rows})


def _block(budget: int) -> dict[str, Any]:
    arm = ARM
    episodes_expected = EVAL_EPISODES
    expected = len(SEEDS) * len(WINDOWS) * EVAL_EPISODES
    rows: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    for seed in SEEDS:
        for window in WINDOWS:
            path = eval_path(arm, budget, window, seed)
            cell = read_jsonl(path)
            files[f"w{window}_seed{seed}"] = {
                "name": path.name, "exists": path.is_file(),
                "rows": len(cell),
                "full": sum(1 for row in cell if row.get("survived")),
                "distinct": distinct_count(cell),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None}
            rows.extend(cell)
    per_window: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    per_cell: list[dict[str, Any]] = []
    for window in WINDOWS:
        cell_rows = [row for row in rows if int(row["window"]) == int(window)]
        if not cell_rows:
            continue
        survived = sum(1 for row in cell_rows if row["survived"])
        cell_distinct = distinct_count(cell_rows)
        per_window.append({
            "window": int(window), "n": len(cell_rows), "full": survived,
            "rate": round(survived / len(cell_rows), 6),
            "cp95": clopper_pearson(survived, len(cell_rows)),
            "episodes_expected": episodes_expected,
            "distinct": cell_distinct,
            "distinct_with_window": distinct_with_window(cell_rows),
            "distinct_replays_per_outcome": (round(len(cell_rows) / cell_distinct, 3)
                                             if cell_distinct else None),
            "by_seed": {str(seed): {
                "n": sum(1 for row in cell_rows
                         if int(row["policy_seed"]) == int(seed)),
                "full": sum(1 for row in cell_rows
                            if int(row["policy_seed"]) == int(seed)
                            and row["survived"])}
                for seed in SEEDS},
            "bombs_used_mean": round(statistics.fmean(
                [int(row["bombs_used"] or 0) for row in cell_rows]), 4),
            "alive_frames_mean": round(statistics.fmean(
                [float(row["alive_frames"] or 0) for row in cell_rows]), 4)})
    for seed in SEEDS:
        seed_rows = [row for row in rows if int(row["policy_seed"]) == int(seed)]
        if not seed_rows:
            continue
        survived = sum(1 for row in seed_rows if row["survived"])
        per_seed[str(seed)] = {
            "n": len(seed_rows), "full": survived,
            "rate": round(survived / len(seed_rows), 6),
            "cp95": clopper_pearson(survived, len(seed_rows)),
            "distinct": distinct_count(seed_rows),
            "bombs_used_mean": round(statistics.fmean(
                [int(row["bombs_used"] or 0) for row in seed_rows]), 4)}
    for window in WINDOWS:
        for seed in SEEDS:
            cell_rows = [row for row in rows
                         if int(row["window"]) == int(window)
                         and int(row["policy_seed"]) == int(seed)]
            if not cell_rows:
                continue
            survived = sum(1 for row in cell_rows if row["survived"])
            cell_distinct = distinct_count(cell_rows)
            per_cell.append({"window": int(window), "policy_seed": int(seed),
                             "n": len(cell_rows), "full": survived,
                             "rate": round(survived / len(cell_rows), 6),
                             "distinct": cell_distinct,
                             "void": bool(cell_distinct < DISTINCT_MIN),
                             "replays_per_outcome": (round(len(cell_rows)
                                                           / cell_distinct, 3)
                                                     if cell_distinct else None),
                             "bombs_used_mean": round(statistics.fmean(
                                 [int(row["bombs_used"] or 0)
                                  for row in cell_rows]), 4)})
    survived_all = sum(1 for row in rows if row["survived"])
    bomb_usage = _bomb_block(rows)
    distinct_pooled = distinct_count(rows)
    distinct_window_sum = sum(cell["distinct"] for cell in per_window)
    per_cell_distinct = [cell["distinct"] for cell in per_cell]
    void_cells = [{"window": int(cell["window"]),
                   "policy_seed": int(cell["policy_seed"]),
                   "n": int(cell["n"]), "full": int(cell["full"]),
                   "distinct": int(cell["distinct"])}
                  for cell in per_cell if cell["void"]]
    kept = {(cell["window"], cell["policy_seed"]) for cell in per_cell
            if not cell["void"]}
    valid_rows = [row for row in rows
                  if (int(row["window"]), int(row["policy_seed"])) in kept]
    valid_full = sum(1 for row in valid_rows if row["survived"])
    return {
        "arm": arm, "budget": int(budget), "bomb_mode": "sample",
        "files": files, "rows": len(rows), "expected_rows": expected,
        "episodes_per_window": episodes_expected,
        "complete": bool(len(rows) == expected),
        "per_window_table": per_window, "per_seed": per_seed,
        "per_cell": per_cell,
        "pooled": {"n": len(rows), "full": survived_all,
                   "rate": round(survived_all / len(rows), 6) if rows else None,
                   "cp95": clopper_pearson(survived_all, len(rows))},
        "pooled_valid": {
            "n": len(valid_rows), "full": valid_full,
            "rate": (round(valid_full / len(valid_rows), 6)
                     if valid_rows else None),
            "cp95": (clopper_pearson(valid_full, len(valid_rows))
                     if valid_rows else {}),
            "void_cells": len(void_cells), "cells_kept": len(kept),
            "note": (f"rows of the (seed, window) cells whose distinct >= "
                     f"{DISTINCT_MIN}; identical to `pooled` when no cell is "
                     f"void (prereg 搂2 P4)")},
        "bomb_usage": bomb_usage,
        "invalid_rows": sum(1 for row in rows if row.get("invalid")),
        "distinct": {
            "definition": ("(alive_frames, bombs_used, bomb_emits, "
                           "tuple(bomb_decisions)) 鈥?prereg 搂2 P4"),
            "rows": len(rows),
            "distinct_outcomes_pooled": distinct_pooled,
            "distinct_per_window_sum": distinct_window_sum,
            "distinct_replays_per_outcome": (round(len(rows) / distinct_pooled, 3)
                                             if distinct_pooled else None),
            "per_cell": {f"w{cell['window']}_s{cell['policy_seed']}":
                         int(cell["distinct"]) for cell in per_cell},
            "per_cell_min": min(per_cell_distinct) if per_cell_distinct else None,
            "per_cell_max": max(per_cell_distinct) if per_cell_distinct else None,
            "per_cell_mean": (round(statistics.fmean(per_cell_distinct), 4)
                              if per_cell_distinct else None),
            "min_required": DISTINCT_MIN,
            "cells_below_min": len(void_cells),
            "void_cells": void_cells,
            "kind": ("stochastic" if distinct_pooled
                     and len(rows) / distinct_pooled < 0.5 * episodes_expected
                     else "deterministic"),
            "note": ("a deterministic arm replays one trajectory per cell "
                     "(distinct = 1) and its row count is NOT a sample size; a "
                     f"cell with distinct < {DISTINCT_MIN} is void")},
        "cell_level": {
            "cells": len(per_cell),
            "full_cells": sum(1 for cell in per_cell if cell["full"] == cell["n"]),
            "zero_cells": sum(1 for cell in per_cell if cell["full"] == 0),
            "mean_cell_rate": round(statistics.fmean(
                [cell["rate"] for cell in per_cell]), 6) if per_cell else None,
            "min_distinct": (min(per_cell_distinct) if per_cell_distinct else None),
            "cells_below_distinct_min": len(void_cells)},
        "wall_s_total": round(sum(float(row["wall_s"] or 0) for row in rows), 3),
        "cells_complete": sum(1 for value in files.values()
                              if value["rows"] == episodes_expected),
    }


def _bomb_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    used = [int(row["bombs_used"] or 0) for row in rows]
    firsts = [int(row["first_bomb_decision"]) for row in rows
              if row.get("first_bomb_decision") is not None]
    lasts = [int(row["last_bomb_decision"]) for row in rows
             if row.get("last_bomb_decision") is not None]
    return {
        "episodes": len(rows),
        "counts": {str(k): sum(1 for v in used if v == k) for k in (0, 1, 2, 3)},
        "rate_by_count": {str(k): round(sum(1 for v in used if v == k) / len(rows), 6)
                          for k in (0, 1, 2, 3)} if rows else {},
        "mean": round(statistics.fmean(used), 4) if used else None,
        "median": round(statistics.median(used), 4) if used else None,
        "stdev": round(statistics.stdev(used), 4) if len(used) > 1 else None,
        "bombs_used_total": sum(used),
        "episodes_exhausted": sum(1 for row in rows
                                  if int(row["bombs_end"] or 0) == 0),
        "episodes_never_bombed": sum(1 for v in used if v == 0),
        "episodes_one_or_two": sum(1 for v in used if v in (1, 2)),
        "bomb_emits_total": sum(int(row["bomb_emits"] or 0) for row in rows),
        "first_bomb_decision": quantiles([float(v) for v in firsts])
        if firsts else {"n": 0},
        "last_bomb_decision": quantiles([float(v) for v in lasts])
        if lasts else {"n": 0},
        "alive_frames": quantiles([float(row["alive_frames"] or 0)
                                   for row in rows]),
        "min_nearest_px": quantiles([float(row["min_nearest_px"]) for row in rows
                                     if row.get("min_nearest_px") is not None]),
        "bomb_prob_mean": round(statistics.fmean(
            [float((row.get("bomb_prob") or {}).get("mean") or 0.0)
             for row in rows]), 6) if rows else None,
    }


def _curve(seed: int, bucket: int = 50) -> dict[str, Any]:
    rows = read_jsonl(train_rows_path(seed))
    buckets: list[dict[str, Any]] = []
    for start in range(0, len(rows), bucket):
        chunk = rows[start:start + bucket]
        if not chunk:
            continue
        rewards = [float(row["reward"]) for row in chunk]
        buckets.append({
            "episodes": [int(chunk[0]["episode"]), int(chunk[-1]["episode"])],
            "n": len(chunk),
            "reward_mean": round(statistics.fmean(rewards), 6),
            "survived": sum(1 for row in chunk if row["survived"]),
            "survival_rate": round(sum(1 for row in chunk if row["survived"])
                                   / len(chunk), 6),
            "bombs_used_mean": round(statistics.fmean(
                [int(row["bombs_used"] or 0) for row in chunk]), 4),
            "alive_frames_mean": round(statistics.fmean(
                [float(row["alive_frames"] or 0) for row in chunk]), 4),
            "entropy_mean": round(statistics.fmean(
                [float(row["entropy_mean"] or 0.0) for row in chunk]), 6)})
    if not rows:
        return {"seed": int(seed), "rows": 0, "buckets": [], "rise": None}
    half = len(rows) // 2
    first = statistics.fmean([float(row["reward"]) for row in rows[:half]])
    last = statistics.fmean([float(row["reward"]) for row in rows[half:]])
    return {"seed": int(seed), "rows": len(rows), "buckets": buckets,
            "first_half_reward": round(first, 6),
            "second_half_reward": round(last, 6),
            "rise": round(last - first, 6), "rises": bool(last > first),
            "survived_total": sum(1 for row in rows if row["survived"])}


# --------------------------------------------------------------------------- #
# mode: verdict / summary — P1-P5 (prereg §1.2) and the stop condition (§1.3)
# --------------------------------------------------------------------------- #
def _survived_bomb_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """P4 (prereg §1.2): bomb usage **among the episodes that survived**."""

    kept = [row for row in rows if row["survived"]]
    used = [int(row["bombs_used"] or 0) for row in kept]
    emits = [int(row["bomb_emits"] or 0) for row in kept]
    firsts = [int(row["first_bomb_decision"]) for row in kept
              if row.get("first_bomb_decision") is not None]
    return {
        "episodes": len(kept),
        "bombs_used_mean": round(statistics.fmean(used), 4) if used else None,
        "bombs_used_total": sum(used),
        "counts": {str(k): sum(1 for v in used if v == k)
                   for k in sorted(set(used))} if used else {},
        "bombs_used_zero": sum(1 for v in used if v == 0),
        "bombs_used_one_or_two": sum(1 for v in used if v in (1, 2)),
        "bomb_emits_mean": round(statistics.fmean(emits), 4) if emits else None,
        "bomb_emits_total": sum(emits),
        "first_bomb_decision": quantiles([float(v) for v in firsts])
        if firsts else {"n": 0},
        "alive_frames_mean": (round(statistics.fmean(
            [float(row["alive_frames"] or 0) for row in kept]), 4) if kept else None),
    }


def _first_bomb_block(rows: Sequence[dict[str, Any]],
                      budget: int = 900) -> dict[str, Any]:
    """**P3** (prereg §1.2): the ``first_bomb_decision`` median and distribution.

    The primary reading is over the episodes that actually bombed
    (``first_bomb_decision is not None``): a head that never bombs has no first
    bomb and cannot be said to have learned *when* to bomb, so counting those
    episodes as "late" would flatter it.  ``median_with_never_bombed_at_budget``
    is reported next to it as the opposite extreme (never-bombed = ``budget``),
    and ``degenerate`` flags the case where fewer than half the episodes bombed,
    in which case the median is not allowed to carry P3 on its own.
    """

    firsts = [int(row["first_bomb_decision"]) for row in rows
              if row.get("first_bomb_decision") is not None]
    sentinel = [int(row["first_bomb_decision"]) if row.get("first_bomb_decision")
                is not None else int(budget) for row in rows]
    buckets: tuple[tuple[str, int, int], ...] = (
        ("0", 0, 0), ("1-5", 1, 5), ("6-10", 6, 10), ("11-20", 11, 20),
        ("21-50", 21, 50), ("51-100", 51, 100), ("101-300", 101, 300),
        (">300", 301, 10 ** 9))
    distribution = {label: sum(1 for value in firsts if low <= value <= high)
                    for label, low, high in buckets}
    stats = quantiles([float(value) for value in firsts])
    sentinel_stats = quantiles([float(value) for value in sentinel])
    bombed = len(firsts)
    median = stats.get("median")
    degenerate = bool(rows and bombed / len(rows) < 0.5)
    median_gt = bool(median is not None and float(median) > P3_MEDIAN_THRESHOLD)
    return {
        "requirement": (f"P3: the median of `first_bomb_decision` must be "
                        f"> {P3_MEDIAN_THRESHOLD:g} (the head must stop bombing at "
                        f"decision 0)"),
        "definition": ("`first_bomb_decision` = the decision index of the episode's "
                       "first honoured bomb emission; median taken over the "
                       "episodes that bombed at least once"),
        "episodes": len(rows), "episodes_bombed": bombed,
        "episodes_never_bombed": len(rows) - bombed,
        "bombed_fraction": (round(bombed / len(rows), 6) if rows else None),
        "median": median, "mean": stats.get("mean"), "q1": stats.get("q1"),
        "q3": stats.get("q3"), "min": stats.get("min"), "max": stats.get("max"),
        "stdev": stats.get("stdev"), "n": stats.get("n"),
        "median_with_never_bombed_at_budget": sentinel_stats.get("median"),
        "distribution": distribution,
        "distribution_fraction": ({label: round(count / bombed, 6)
                                   for label, count in distribution.items()}
                                  if bombed else {}),
        "at_decision_zero": distribution["0"],
        "at_decision_zero_fraction": (round(distribution["0"] / bombed, 6)
                                      if bombed else None),
        "within_10": sum(value for label, value in distribution.items()
                         if label in ("0", "1-5", "6-10")),
        "after_10": (distribution["11-20"] + distribution["21-50"]
                     + distribution["51-100"] + distribution["101-300"]
                     + distribution[">300"]),
        "threshold": P3_MEDIAN_THRESHOLD,
        "median_gt_threshold": median_gt,
        "degenerate_low_bomb_fraction": degenerate,
        "passed": bool(median_gt and not degenerate),
    }


def _feature_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What the three explicit features actually looked like during evaluation."""

    present = [row["explicit_features"] for row in rows
               if isinstance(row.get("explicit_features"), dict)
               and row["explicit_features"].get("decisions")]
    mins = {key: [float(feat["min"][key]) for feat in present
                  if feat["min"].get(key) is not None]
            for key in ("f1", "f2", "f3")}
    maxs = {key: [float(feat["max"][key]) for feat in present
                  if feat["max"].get(key) is not None]
            for key in ("f1", "f2", "f3")}
    first_bombs = [entry for row in rows for entry in (row.get("first_bombs") or [])]
    return {
        "names": list(FEATURE_NAMES), "dim": FEATURE_DIM,
        "rows_with_features": len(present), "rows": len(rows),
        "all_in_unit_interval": bool(present and all(
            bool(feat.get("all_in_unit_interval")) for feat in present)),
        "per_row_min": {key: quantiles(values) for key, values in mins.items()},
        "per_row_max": {key: quantiles(values) for key, values in maxs.items()},
        "no_bullets_rows": sum(1 for feat in present
                               if int(feat.get("no_bullets_decisions") or 0) > 0),
        "first_bombs_logged": len(first_bombs),
        "first_bombs": first_bombs[:20],
        "first_bombs_f3": quantiles([float(entry["f3"]) for entry in first_bombs
                                     if entry.get("f3") is not None]),
        "first_bombs_nearest_px": quantiles(
            [float(entry["nearest_all_px"]) for entry in first_bombs
             if entry.get("nearest_all_px") is not None]),
    }


def _per_window_first_bomb(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for window in WINDOWS:
        cell = [row for row in rows if int(row["window"]) == int(window)]
        if not cell:
            continue
        block = _first_bomb_block(cell)
        survived = [row for row in cell if row["survived"]]
        out[str(window)] = {
            "n": len(cell), "episodes_bombed": block["episodes_bombed"],
            "median": block["median"], "mean": block["mean"],
            "at_decision_zero": block["at_decision_zero"],
            "distribution": block["distribution"],
            "median_survived": (_first_bomb_block(survived)["median"]
                                if survived else None),
            "n_survived": len(survived)}
    return out


def _head_digests() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for seed in SEEDS:
        path = head_ckpt_path(seed)
        out[str(seed)] = {"path": str(path), "exists": path.is_file(),
                          "sha256": sha256_file(path) if path.is_file() else None,
                          "bytes": path.stat().st_size if path.is_file() else None}
    return out


def mode_verdict(args: argparse.Namespace) -> int:
    """The round's single summary: P1-P5, the side-by-side, and the verdict.

    ``--mode summary`` is an alias so there is exactly **one** code path that
    produces the published numbers: a summary can never disagree with a verdict
    because they are the same file.  ``d3_gate.json`` is read from the round root
    and its ordering relative to this file is asserted (v8.27.0 was once misread
    as ``S-fork`` because a summary had been written 12 s *before* the gate).
    """

    out_path = pathlib.Path(args.out) if args.out else WORK / "summary.json"
    budget = 900
    block = _block(budget)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for window in WINDOWS:
            rows.extend(read_jsonl(eval_path(ARM, budget, window, seed)))
    usage_all = block.get("bomb_usage") or {}
    first_bomb = _first_bomb_block(rows, budget)
    arm_cell: dict[str, Any] = {
        "arm": ARM, "label": "F = obs_106 + [f1, f2, f3] (109 dims)",
        "reward": {"weights": list(REWARD_BY_ARM[ARM]),
                   "formula": "R = survived(900) + 0.5 * alive_frames / 1800",
                   "reward_survived": REWARD_BY_ARM[ARM][0],
                   "reward_alive_weight": REWARD_BY_ARM[ARM][1]},
        "budget": budget, "episodes_per_window": EVAL_EPISODES,
        "rows": block["rows"], "expected_rows": block["expected_rows"],
        "complete": block["complete"],
        "pooled": block["pooled"], "pooled_valid": block["pooled_valid"],
        "per_window": block["per_window_table"], "per_seed": block["per_seed"],
        "per_cell": block["per_cell"],
        "distinct": block["distinct"], "cell_level": block["cell_level"],
        "bomb_usage_all": usage_all,
        "bomb_usage_survived": _survived_bomb_block(rows),
        "bomb_usage_never_survived": _survived_bomb_block(
            [row for row in rows if not row["survived"]]),
        "first_bomb_decision": first_bomb,
        "first_bomb_decision_by_window": _per_window_first_bomb(rows),
        "explicit_features": _feature_block(rows),
        "invalid_rows": block["invalid_rows"],
        "wall_s_total": block["wall_s_total"], "files": block["files"],
        "heads": _head_digests(),
    }
    log(f"verdict {ARM} d{budget}: {block['pooled']['full']}/{block['pooled']['n']}"
        f" = {block['pooled']['rate']} distinct="
        f"{block['distinct'].get('distinct_outcomes_pooled')} "
        f"min_cell={block['distinct'].get('per_cell_min')} "
        f"first_bomb_median={first_bomb['median']}")

    sized = block["pooled_valid"] if block["pooled_valid"].get("n") else block["pooled"]
    p3 = first_bomb
    p3_passed = bool(p3["passed"])

    # -- P1 (primary): 900 decisions > 58/240 = 24.1667 %, one-sided Fisher ----
    p1_ref_full, p1_ref_n = P1_REFERENCE["pooled"]
    p1: dict[str, Any] = {
        "requirement": ("P1 (prereg §1.2, primary): pooled 900-decision (30 s) "
                        "survival must exceed v8.20.0's 60/40/20 point estimate "
                        "58/240 = 24.1667 %, one-sided Fisher p < "
                        f"{P1_ALPHA}; (seed, window) cells with distinct < "
                        f"{DISTINCT_MIN} excluded"),
        "budget": budget, "threshold": round(P1_THRESHOLD, 6), "alpha": P1_ALPHA,
        "observed": sized.get("rate"), "cp95": sized.get("cp95"),
        "n": sized.get("n"), "full": sized.get("full"),
        "observed_all_rows": block["pooled"].get("rate"),
        "n_all_rows": block["pooled"].get("n"),
        "reference": {"full": p1_ref_full, "n": p1_ref_n,
                      "rate": round(p1_ref_full / p1_ref_n, 6),
                      "arm": P1_REFERENCE["label"],
                      "source": P1_REFERENCE["source"]},
        "difference_pp": None, "fisher_greater": None, "fisher_less": None,
        "rate_gate": False, "fisher_significant": False,
        "complete": bool(block["complete"]), "passed": False,
    }
    if sized.get("n"):
        full, n = int(sized["full"]), int(sized["n"])
        p1["difference_pp"] = round(100.0 * (full / n - p1_ref_full / p1_ref_n), 4)
        p1["fisher_greater"] = fisher_greater(full, n - full, p1_ref_full,
                                              p1_ref_n - p1_ref_full)
        p1["fisher_less"] = fisher_less(full, n - full, p1_ref_full,
                                        p1_ref_n - p1_ref_full)
        p1["rate_gate"] = bool(full / n > P1_THRESHOLD)
        p1["fisher_significant"] = bool(
            float(p1["fisher_greater"].get("p_value") or 1.0) < P1_ALPHA)
        p1["passed"] = bool(p1["complete"] and p1["rate_gate"]
                            and p1["fisher_significant"])

    # -- P2: 900 decisions > 147/3600 = 4.0833 % (v8.24.0 D0, same framework) --
    p2_ref_full, p2_ref_n = P2_REFERENCE["pooled"]
    p2: dict[str, Any] = {
        "requirement": ("P2 (prereg §1.2): pooled 900-decision (30 s) survival must "
                        "exceed v8.24.0's D0 — the **same** framework with the same "
                        "sparse reward but the 106-dim head — 147/3,600 = 4.0833 %"),
        "budget": budget, "threshold": round(P2_THRESHOLD, 6),
        "reference": {"full": p2_ref_full, "n": p2_ref_n,
                      "rate": round(p2_ref_full / p2_ref_n, 6),
                      "arm": REF_D0_824["label"],
                      "source": REF_D0_824["source"]},
        "observed": sized.get("rate"), "cp95": sized.get("cp95"),
        "n": sized.get("n"), "full": sized.get("full"),
        "difference_pp": None, "fisher_greater": None, "fisher_less": None,
        "rate_gate": False, "passed": False,
    }
    if sized.get("n"):
        full, n = int(sized["full"]), int(sized["n"])
        p2["difference_pp"] = round(100.0 * (full / n - p2_ref_full / p2_ref_n), 4)
        p2["fisher_greater"] = fisher_greater(full, n - full, p2_ref_full,
                                              p2_ref_n - p2_ref_full)
        p2["fisher_less"] = fisher_less(full, n - full, p2_ref_full,
                                        p2_ref_n - p2_ref_full)
        p2["rate_gate"] = bool(full / n > P2_THRESHOLD)
        p2["passed"] = bool(p2["rate_gate"] and block["complete"])

    # -- P4: bomb usage on the episodes that survived -------------------------
    p4: dict[str, Any] = {
        "requirement": ("P4 (prereg §1.2): report the bomb-usage distribution and, "
                        "the mean bombs used **on the episodes that survived** "
                        "(plus `bomb_emits` per episode); descriptive, no pass/fail "
                        "gate"),
        "survived": arm_cell["bomb_usage_survived"],
        "never_survived": arm_cell["bomb_usage_never_survived"],
        "all_episodes": usage_all,
        "reference_v8_21": {"10s": 1.315, "20s": 2.964, "30s": 2.962,
                            "label": REF_RL821["label"]},
        "reported": True, "passed": None,
    }

    # -- P5: distinct on every (seed, window) cell ---------------------------
    info = arm_cell["distinct"]
    void_cells = info.get("void_cells") or []
    p5: dict[str, Any] = {
        "requirement": ("P5 (prereg §1.2): every (seed, window) cell must report its "
                        "`distinct` count on (alive_frames, bombs_used, bomb_emits, "
                        f"tuple(bomb_decisions)); distinct < {DISTINCT_MIN} makes the "
                        "cell void and excludes its rows from P1/P2"),
        "definition": ("(alive_frames, bombs_used, bomb_emits, tuple(bomb_decisions))"),
        "min_required": DISTINCT_MIN,
        "per_cell": info.get("per_cell"),
        "distinct_outcomes_pooled": info.get("distinct_outcomes_pooled"),
        "distinct_per_window_sum": info.get("distinct_per_window_sum"),
        "replays_per_outcome": info.get("distinct_replays_per_outcome"),
        "per_cell_min": info.get("per_cell_min"),
        "per_cell_max": info.get("per_cell_max"),
        "per_cell_mean": info.get("per_cell_mean"),
        "void_cells": void_cells, "cells_below_min": len(void_cells),
        "passed": bool(not void_cells and info.get("per_cell_min") is not None),
    }

    # -- the training curve (per 50 episodes, per seed and pooled) -----------
    curves = {str(seed): _curve(seed, 50) for seed in SEEDS}
    seeds_with_rows = [curves[str(seed)] for seed in SEEDS
                       if curves[str(seed)]["rows"]]
    pooled_buckets: list[dict[str, Any]] = []
    for index in range(max((len(curve["buckets"]) for curve in seeds_with_rows),
                           default=0)):
        chunk = [curve["buckets"][index] for curve in seeds_with_rows
                 if len(curve["buckets"]) > index]
        rows_n = sum(bucket["n"] for bucket in chunk)
        pooled_buckets.append({
            "episodes": chunk[0]["episodes"], "n": rows_n,
            "reward_mean": round(sum(bucket["reward_mean"] * bucket["n"]
                                     for bucket in chunk) / rows_n, 6),
            "survived": sum(bucket["survived"] for bucket in chunk),
            "survival_rate": round(sum(bucket["survived"] for bucket in chunk)
                                   / rows_n, 6),
            "bombs_used_mean": round(statistics.fmean(
                [bucket["bombs_used_mean"] for bucket in chunk]), 4),
            "alive_frames_mean": round(statistics.fmean(
                [bucket["alive_frames_mean"] for bucket in chunk]), 4),
            "entropy_mean": round(statistics.fmean(
                [bucket["entropy_mean"] for bucket in chunk]), 6)})
    train_rows = {str(seed): read_jsonl(train_rows_path(seed)) for seed in SEEDS}
    first_50 = [float(row["reward"]) for seed in SEEDS
                for row in train_rows[str(seed)][:50]]
    last_50 = [float(row["reward"]) for seed in SEEDS
               for row in train_rows[str(seed)][-50:]]
    first_bomb_train = _first_bomb_block(
        [{"first_bomb_decision": row.get("first_bomb_decision"),
          "bombs_used": row.get("bombs_used")}
         for seed in SEEDS for row in train_rows[str(seed)]], TRAIN_DECISIONS)
    train_curve: dict[str, Any] = {
        "requirement": ("the training curve: mean return, survival rate and bombs "
                        "used per 50 episodes, per seed and pooled"),
        "bucket_episodes": 50, "episodes_per_seed": TRAIN_EPISODES,
        "decisions_per_episode": TRAIN_DECISIONS,
        "seeds": curves, "pooled_buckets": pooled_buckets,
        "train_rows": {str(seed): len(train_rows[str(seed)]) for seed in SEEDS},
        "survived_total": {str(seed): sum(1 for row in train_rows[str(seed)]
                                          if row["survived"]) for seed in SEEDS},
        "first_50_reward": round(statistics.fmean(first_50), 6) if first_50 else None,
        "last_50_reward": round(statistics.fmean(last_50), 6) if last_50 else None,
        "rise": (round(statistics.fmean(last_50) - statistics.fmean(first_50), 6)
                 if first_50 and last_50 else None),
        "first_bomb_decision_in_training": first_bomb_train,
    }

    # -- the D3 fidelity gate: present, 5/5, and written BEFORE this file -----
    d3_path = WORK / "d3_gate.json"
    d3 = read_json(d3_path)
    d3_ok = bool(d3 and d3.get("all_pass"))
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    d3_created = str(d3.get("created_at") or "")
    ordering_ok = bool(d3_created and d3_created <= created_at)
    d3_info = {"path": str(d3_path), "present": bool(d3),
               "sha256": sha256_file(d3_path) if d3_path.is_file() else None,
               "created_at": d3_created, "summary_created_at": created_at,
               "written_before_summary": ordering_ok,
               "passed": d3.get("passed"), "total": d3.get("total"),
               "verdict": d3.get("verdict"), "tag": d3.get("tag"),
               "all_pass": d3_ok}

    # -- side by side with 4.08 % (v8.24 D0), 24.17 % (P1) and 60.83 % (G1) --
    def versus(ref_full: int, ref_n: int) -> dict[str, Any]:
        out: dict[str, Any] = {
            "pooled": {"full": sized.get("full"), "n": sized.get("n"),
                       "rate": sized.get("rate"), "cp95": sized.get("cp95")},
            "reference": {"full": ref_full, "n": ref_n,
                          "rate": round(ref_full / ref_n, 6)},
            "difference_pp": None, "fisher_greater": None, "fisher_less": None}
        if sized.get("n"):
            full, n = int(sized["full"]), int(sized["n"])
            out["difference_pp"] = round(100.0 * (full / n - ref_full / ref_n), 4)
            out["fisher_greater"] = fisher_greater(full, n - full, ref_full,
                                                   ref_n - ref_full)
            out["fisher_less"] = fisher_less(full, n - full, ref_full,
                                             ref_n - ref_full)
        return out

    g1_full, g1_n = REF_G1["900"]["pooled"]
    comparison: dict[str, Any] = {
        "arm": f"{ARM}|900", "budget": budget,
        "versus_D0_v8_24_4_08": {
            **versus(p2_ref_full, p2_ref_n),
            "label": REF_D0_824["label"], "source": REF_D0_824["source"]},
        "versus_P1_threshold_24_17": {
            **versus(p1_ref_full, p1_ref_n),
            "label": P1_REFERENCE["label"], "source": P1_REFERENCE["source"]},
        "versus_G1_v8_26_60_83": {
            **versus(g1_full, g1_n),
            "label": REF_G1["label"], "source": REF_G1["source"]},
        "versus_40_20_5_30s": {
            **versus(REF_30S_40_20_5["pooled"][0], REF_30S_40_20_5["pooled"][1]),
            "label": REF_30S_40_20_5["label"],
            "source": REF_30S_40_20_5["source"]},
    }

    # -- the stop condition (prereg §1.3) ------------------------------------
    complete = bool(block["complete"])
    if not complete:
        stop = "incomplete"
        stop_text = (f"only {block['rows']}/{block['expected_rows']} rows; no stop "
                     f"condition is declared on a partial table")
    elif not d3_ok:
        stop = "S-fork"
        stop_text = "the D3 fidelity gate is not 5/5; the whole round is void"
    elif not ordering_ok:
        stop = "S-fork"
        stop_text = ("d3_gate.json was not written before this summary; the gate "
                     "cannot certify a run it postdates")
    elif p1["passed"] and p3_passed:
        stop = "S-features-work"
        stop_text = ("P1 and P3 both hold: the explicit features beat 24.17 % at "
                     "30 s AND pushed the first bomb past decision 10 — the "
                     "representation was the bottleneck, so this route reopens")
    elif p1["passed"] and not p3_passed:
        stop = "S-features-partial"
        stop_text = ("P1 holds but P3 does not: the explicit features lifted the 30 s "
                     "rate above 24.17 % while the head still bombs at decision 0, so "
                     "the representation helped the rate but not the *timing* "
                     "behaviour (the prereg's table has no row for this case; it is "
                     "reported as the partial verdict)")
    elif p2["passed"]:
        stop = "S-features-partial"
        stop_text = ("P2 holds but P1 does not: better than the sparse/dense 106-dim "
                     "arms (4.08 %) but still far below the hand-written rule")
    else:
        stop = "S-representation-hopeless"
        stop_text = ("neither P1 nor P2 holds: changing the representation (three "
                     "explicit features) does NOT rescue 30 s bomb learning, so "
                     "'learning when to bomb' is closed inside this framework")

    table = [{
        "arm": ARM, "budget": budget,
        "pooled_full": block["pooled"]["full"],
        "pooled_n": block["pooled"]["n"],
        "pooled_rate": block["pooled"]["rate"],
        "cp95_low": block["pooled"]["cp95"].get("low"),
        "cp95_high": block["pooled"]["cp95"].get("high"),
        "rate_void_excluded": sized.get("rate"),
        "n_void_excluded": sized.get("n"),
        "distinct": info.get("distinct_outcomes_pooled"),
        "distinct_min_cell": info.get("per_cell_min"),
        "void_cells": info.get("cells_below_min"),
        "bombs_mean": usage_all.get("mean"),
        "bombs_mean_survived": arm_cell["bomb_usage_survived"]["bombs_used_mean"],
        "first_bomb_median": first_bomb["median"],
        "first_bomb_at_zero": first_bomb["at_decision_zero"],
        "complete": complete}]
    for bucket in block["per_window_table"]:
        table[0][f"w{bucket['window']}"] = f"{bucket['full']}/{bucket['n']}"
        table[0][f"w{bucket['window']}_distinct"] = bucket["distinct"]

    payload: dict[str, Any] = {
        "schema": SCHEMA_SUMMARY, "producer": PRODUCER, "status": "complete",
        "created_at": created_at,
        "prereg": {"path": PREREG_REL,
                   "sha256": (sha256_file(REPO / PREREG_REL)
                              if (REPO / PREREG_REL).is_file() else None)},
        "question": ("does appending three explicit features (bombs left / decisions "
                     "left / nearest-bullet distance) to the bomb head's input turn "
                     "'bomb at decision 0' into 'bomb when it matters' at 30 s (900 "
                     "decisions), beating the 24.17 % hand-written reference?"),
        "protocol": {
            "candidate_head": ("v8.6.0 bombdag.pt Policy(106,64,36), frozen, argmax: "
                               "cand = argmax_c max(logit[2c], logit[2c+1])"),
            "bomb_head": "Linear(109,64)->Tanh->Linear(64,2)",
            "bomb_head_input": (f"{FEATURE_DIM} = {OBS_DIM} (frozen obs) + "
                                f"{EXPLICIT_DIM} explicit: {list(FEATURE_NAMES)}"),
            "bomb_head_pretrain": (f"{BC_STEPS} BCE steps, batch {BC_BATCH} "
                                   f"({BC_POS_FRACTION:.0%} positives), Adam "
                                   f"{BC_LR:g}, one rule episode per window, "
                                   f"**on the 109-dim input**"),
            "learning_signal": ("REINFORCE, R = survived(900) + 0.5 * "
                                "alive_frames/1800 (v8.24.0's sparse form, for "
                                f"comparability), moving-average baseline (ema "
                                f"{BASELINE_EMA}), entropy {ENTROPY_COEF}, Adam "
                                f"{LR:g}, {TRAIN_EPISODES} episodes x "
                                f"{TRAIN_DECISIONS} decisions per seed"),
            "seeds": list(SEEDS), "windows": list(WINDOWS),
            "eval_budgets_decisions": list(EVAL_BUDGETS),
            "eval_episodes_per_window": EVAL_EPISODES,
            "eval_episodes_total": len(SEEDS) * len(WINDOWS) * EVAL_EPISODES,
            "decision_frames": DECISION_FRAMES, "prefix_frame": PREFIX_FRAMES,
            "loop": ("for seed in (0,1,2): for window in (11,12,17,18): for "
                     "episode in range(150)"),
            "episode_loop": "tools/rl/v830_bombpol.py::run_episode36 (imported)",
            "warm_start_rule": f"{RULE_NAME} = {RULE_LABEL} (through v8l0)"},
        "references": {"D0_v8_24": REF_D0_824, "G1_v8_26": REF_G1,
                       "60_40_20_30s": REF_30S_60_40_20,
                       "40_20_5_30s": REF_30S_40_20_5, "v8_21_RL": REF_RL821},
        "arm": arm_cell, "arms": {ARM: arm_cell},
        "comparison": comparison, "table": table,
        "train_curve": train_curve, "d3_gate": d3_info,
        "verdict": {
            "P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5,
            "stop_condition": stop, "stop_text": stop_text,
            "S-features-work": {"condition": "P1 and P3 hold",
                                "held": stop == "S-features-work"},
            "S-features-partial": {"condition": "P2 holds but P1 does not (also "
                                               "used when P1 holds but P3 does not)",
                                   "held": stop == "S-features-partial"},
            "S-representation-hopeless": {"condition": "neither P1 nor P2 holds",
                                          "held": stop ==
                                          "S-representation-hopeless"},
            "S-fork": {"condition": "the D3 fidelity gate fails / postdates the "
                                    "summary", "held": stop == "S-fork"}},
        "invalid_rows": block["invalid_rows"],
        "complete": complete,
        "notes": [
            "Only the 900-decision (30 s) tier was evaluated: the prereg fixes this "
            "round to it, so nothing is claimed about 20 s or 40 s.",
            "The 12 (seed, window) cells are not independent samples; the CP95 "
            "intervals are nominal and the cells are the effective n.",
            "Training and evaluation use the same 4 dev windows: no cross-window "
            "generalisation claim is made.",
            "D0 (v8.24.0, 4.08 %), the 24.17 % 60/40/20 reference and G1 "
            "(v8.26.0, 60.83 %) are published numbers, never re-run in this round.",
            "P3's median is taken over the episodes that bombed at least once; a "
            "never-bombing head is reported through "
            "`median_with_never_bombed_at_budget` and `degenerate_low_bomb_fraction` "
            "instead of being counted as 'late'.",
        ],
    }
    write_json(out_path, payload)
    write_json(WORK / "train_curve.json", {
        "schema": "v8u0-features-train-curve/1", "producer": PRODUCER,
        "created_at": created_at, "train_curve": train_curve})
    write_json(WORK / "bomb_usage.json", {
        "schema": "v8u0-features-bomb-usage/1", "producer": PRODUCER,
        "created_at": created_at, "P4": p4,
        "P3": first_bomb})
    write_json(WORK / "first_bomb_decision.json", {
        "schema": "v8u0-features-first-bomb-decision/1", "producer": PRODUCER,
        "created_at": created_at, "P3": first_bomb,
        "by_window": arm_cell["first_bomb_decision_by_window"],
        "in_training": first_bomb_train})
    log(f"verdict: P1={p1['passed']} ({p1['observed']}) P2={p2['passed']} "
        f"({p2['observed']}) P3={p3_passed} (median "
        f"{first_bomb['median']} vs > {P3_MEDIAN_THRESHOLD:g}, "
        f"at-0 {first_bomb['at_decision_zero']}/{first_bomb['episodes_bombed']}) "
        f"P4(reported) P5={p5['passed']} (min distinct {p5['per_cell_min']}) "
        f"d3={d3_ok} -> {stop}")
    log(f"  pooled {block['pooled']['full']}/{block['pooled']['n']} = "
        f"{block['pooled']['rate']} CP95=[{block['pooled']['cp95'].get('low')}, "
        f"{block['pooled']['cp95'].get('high')}] "
        f"bombs(survived)={arm_cell['bomb_usage_survived']['bombs_used_mean']} "
        f"distinct={info.get('distinct_outcomes_pooled')}")
    return 0 if block["invalid_rows"] == 0 else 4


# --------------------------------------------------------------------------- #
# mode: audio-audit / process-audit / publish
# --------------------------------------------------------------------------- #
def mode_audio(args: argparse.Namespace) -> int:
    out_path = pathlib.Path(args.out) if args.out else BASE_WORK / "audio_audit.json"
    # v8.28.0 is a single-arm round, so the audit covers the whole round root
    # (the private data dirs live under ``<round>/data/<tag>/w<window>``).
    root = BASE_WORK
    module = load_module("v431_audio_cfg_audit",
                         REPO / "tools" / "rl" / "v431_audio_cfg_audit.py")
    cfgs = sorted(root.rglob("th08.cfg")) if root.is_dir() else []
    rows: list[dict[str, Any]] = []
    for path in cfgs:
        try:
            decoded = module.decode(path)
        except OSError as exc:
            decoded = {"path": str(path), "error": str(exc)}
        if decoded:
            decoded["data_dir"] = str(path.parent)
            rows.append(decoded)
    ok = [row for row in rows if "error" not in row]
    muted = [row for row in ok if row["muted"]]
    audible = [row for row in ok if not row["muted"]]
    payload: dict[str, Any] = {
        "schema": SCHEMA_AUDIO, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("AGENTS.md §9.3: every private data dir of this round "
                        "carries musicMode=0 playSounds=0 musicVolume=0 "
                        "sfxVolume=0"),
        "scope": str(root), "cfg_files": len(cfgs), "decoded": len(ok),
        "errors": len(rows) - len(ok), "muted": len(muted),
        "audible": len(audible),
        "audible_paths": [row["path"] for row in audible], "rows": rows,
        "status": "complete"}
    payload["all_pass"] = bool(len(ok) > 0 and not audible)
    payload["status"] = "complete" if payload["all_pass"] else "failed"
    write_json(out_path, payload)
    log(f"audio-audit: {payload['cfg_files']} cfg, {payload['muted']} muted, "
        f"{payload['audible']} audible -> {out_path}")
    return 0 if payload["all_pass"] else 6


def mode_process(args: argparse.Namespace) -> int:
    out_path = (pathlib.Path(args.out) if args.out
                else BASE_WORK / "process_residue.json")
    census = engine_census()
    payload: dict[str, Any] = {
        "schema": SCHEMA_PROC, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("this round's engine processes must be 0 when it ends; "
                        "other rounds' processes are counted but never touched"),
        "needle": ".tmp_v8u0_features", "pgrep_returncode": census["pgrep_returncode"],
        "engine_processes_total": census["engine_processes_total"],
        "other_round_processes": census["other_round_processes"],
        "matching": census["matching"], "lines": census["lines_mine"]}
    payload["all_pass"] = bool(payload["matching"] == 0)
    payload["status"] = "complete" if payload["all_pass"] else "failed"
    write_json(out_path, payload)
    log(f"process-audit: {payload['matching']} own engine process(es) alive "
        f"({payload['other_round_processes']} other-round engines untouched) -> "
        f"{out_path}")
    return 0 if payload["all_pass"] else 6


def mode_publish(args: argparse.Namespace) -> int:
    import shutil

    root = pathlib.Path(args.out) if args.out else PUBLISH
    root.mkdir(parents=True, exist_ok=True)
    # (source, published name) — the head ckpt carries the prereg §4 name
    copies: list[tuple[pathlib.Path, str]] = []
    for budget in EVAL_BUDGETS:
        for seed in SEEDS:
            for window in WINDOWS:
                path = eval_path(ARM, budget, window, seed)
                copies.append((path, path.name))
    for seed in SEEDS:
        copies.append((head_ckpt_path(seed),
                       f"bomb_head_seed{seed}_d900.pt"))
        copies.append((bc_ckpt_path(seed), f"bomb_head_seed{seed}_bc.pt"))
        copies.append((train_rows_path(seed), f"seed{seed}.jsonl"))
        copies.append((WORK / "pretrain" / f"seed{seed}.json",
                       f"pretrain_seed{seed}.json"))
        copies.append((WORK / "train" / f"seed{seed}.json",
                       f"train_seed{seed}.json"))
    for name in ("summary.json", "train_smoke.json"):
        source = WORK / name
        if not source.is_file():
            continue
        copies.append((source, name))
    for name in ("d3_gate.json", "ckpt_assertions_start.json",
                 "ckpt_assertions_final.json", "audio_audit.json",
                 "process_residue.json", "preflight.json", "train_curve.json",
                 "bomb_usage.json", "first_bomb_decision.json",
                 "commands.tsv", "eval_exits.tsv", "deviations.md"):
        copies.append((BASE_WORK / name, name))
    copied: list[dict[str, Any]] = []
    for source, name in copies:
        if not source.is_file():
            continue
        target = root / name
        shutil.copy2(source, target)
        copied.append({"name": name, "source": str(source),
                       "sha256": sha256_file(target),
                       "bytes": target.stat().st_size})
    payload: dict[str, Any] = {
        "schema": SCHEMA_PUBLISH, "producer": PRODUCER, "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "publish_dir": str(root), "copied": copied, "count": len(copied)}
    write_json(root / "publish.json", payload)
    files: dict[str, Any] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root)).replace("\\", "/")
        if rel == "manifest.json":
            continue
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {"schema": SCHEMA_MANIFEST, "producer": PRODUCER,
                "created_at": payload["created_at"], "root": str(root),
                "prereg": PREREG_REL, "file_count": len(files),
                "total_bytes": sum(value["bytes"] for value in files.values()),
                "files": files}
    write_json(root / "manifest.json", manifest)
    log(f"publish: {len(copied)} artifacts + manifest ({len(files)} files) -> "
        f"{root}")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("v8.28.0 FEATURES — frozen candidate head (argmax) + a "
                     "2-action bomb head whose input is 106 obs + 3 explicit "
                     "features (109), trained by REINFORCE on 900-decision (30 s) "
                     "episodes with v8.24.0's sparse reward"))
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "assert-ckpt", "d3", "pretrain",
                                 "train", "train-smoke", "eval", "eval-smoke",
                                 "verdict", "summary", "audio-audit",
                                 "process-audit",
                                 "publish"))
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--arm", default="F", choices=ARMS)
    parser.add_argument("--arms", default=ARM)
    parser.add_argument("--budgets", default=",".join(str(b) for b in EVAL_BUDGETS))
    parser.add_argument("--decisions", type=int, default=TRAIN_DECISIONS)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--windows", default=",".join(str(w) for w in WINDOWS))
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--phase", default="start", choices=("start", "final"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--waited-s", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global _LOG_FH, ARM, WORK
    args = build_args(sys.argv[1:] if argv is None else argv)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.log:
        pathlib.Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        _LOG_FH = open(args.log, "a", encoding="utf-8")
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    ARM = str(args.arm)
    if ARM not in ARMS:
        raise BombRLError(f"--arm must be one of {ARMS}, got {ARM!r}")
    WORK = BASE_WORK
    WORK.mkdir(parents=True, exist_ok=True)
    BASE_WORK.mkdir(parents=True, exist_ok=True)
    log(f"mode={args.mode} python={sys.executable} pid={os.getpid()} "
        f"tag={args.tag!r} arm={args.arm} "
        f"budget={args.decisions} seed={args.policy_seed} "
        f"episodes={args.episodes}")
    try:
        if args.mode == "preflight":
            return mode_preflight(args)
        if args.mode == "assert-ckpt":
            return mode_assert(args)
        if args.mode == "d3":
            return mode_d3(args)
        if args.mode == "pretrain":
            return mode_pretrain(args)
        if args.mode == "train":
            return mode_train(args)
        if args.mode == "train-smoke":
            args.episodes = 1
            args.smoke = True
            return mode_train(args)
        if args.mode == "eval":
            return mode_eval(args)
        if args.mode == "eval-smoke":
            return mode_eval_smoke(args)
        if args.mode in ("verdict", "summary"):
            # one code path: the published summary **is** the verdict file
            return mode_verdict(args)
        if args.mode == "audio-audit":
            return mode_audio(args)
        if args.mode == "process-audit":
            return mode_process(args)
        if args.mode == "publish":
            return mode_publish(args)
        raise BombRLError(f"unknown mode {args.mode}")
    finally:
        if _LOG_FH is not None:
            _LOG_FH.flush()


if __name__ == "__main__":
    sys.exit(main())
