#!/usr/bin/env python3
"""v8.20.0 UNIFIED — the tightened 60/40/20 bomb rule re-measured at 600 / 900 /
1,200 decisions.

What this round asks
--------------------
``Document/Update/Alpha/v8.20.0/PREREGISTRATION_UNIFIED_001.md`` freezes **one**
arm (candidate head = v8.6.0 ``bombdag.pt`` ``Policy(106, 64, 36)``, bomb rule =
v8.19.0's S1 = **60 / 40 / 20 px**) and moves **only the decision budget**::

    U-600    600 decisions (20 s)   vs v8.12.0 H2s  236/240 = 98.3333 % (80/60/40)
    U-900    900 decisions (30 s)   vs v8.19.0 S1    58/240 =  24.1667 % (repeat)
    U-1200  1200 decisions (40 s)   vs R5          36/3,600 =   1.0000 % (80 px)

The arm itself is **v8.19.0 S1 verbatim** except for the budget
----------------------------------------------------------------
* policy: ``.tmp_v860_bombdag/ckpt/seed{seed}/bombdag.pt`` — ``Policy(106, 64,
  36)`` loaded by ``v860_bombdag.load_policy36`` (the same loader v8.6.0,
  v8.12.0's H2s, v8.14.0's HYBRID30 and v8.19.0's S1 use);
* candidate: ``score[c] = max(logit[2c], logit[2c+1])``, ``softmax(score / 1.0)``
  and **one sampled draw per decision** from one ``torch.Generator()`` seeded
  ``seed * 1000 + 7`` per process (created before the window loop, consumed once
  per decision across every window and episode);
* bomb: the external rule — ``nearest_all_px < THRESHOLDS[bombs_remaining]``
  with ``THRESHOLDS = {3: 60, 2: 40, 1: 20}`` **and** ``bombs_remaining > 0``;
  the distance itself is still measured by ``v800_bomb.nearest_bullet_distances``
  (v8.0.0's own code, not re-derived);
* masks: ``cand_to_mask(cand, bomb)`` on the decision's first frame and
  ``cand_to_mask(cand, 0)`` afterwards — inside the imported
  ``v830_bombpol.run_episode36``, not re-implemented here;
* start: PARK frame 8663; 2 frames per decision; ``budget`` decisions =
  ``2 * budget`` frames;
* loop: ``for seed in (0,1,2): for window in (11,12,17,18): for episode in
  range(20)`` — 3 x 4 x 20 = **240 episodes per budget**.

Modes
-----
``--mode preflight``     engine count on the box before any launch (AGENTS.md /
                         the "> 6 engines -> wait 5 min" rule), recorded as a
                         deviation log entry.
``--mode assert-ckpt``   engine sha256, the three v8.6.0 ``bombdag.pt``
                         (``9c6d5d4a`` / ``36a51090`` / ``885dde70``), the three
                         frozen v4.5.0 ``round2.pt`` (``4abee069`` / ``a93746bb``
                         / ``c69960fc``), the protected sources.  Fail closed.
``--mode d3``            the D3 fidelity gate (5 assertions) through the frozen
                         ``v432_fork_arm.py --mode fidelity``; a non-5/5 gate
                         voids the round.
``--mode smoke``         **one** episode (U-600 / seed0 / w11 / 600), then stop.
``--mode eval``          one ``--decisions`` x one ``--policy-seed``: 4 windows x
                         20 episodes, one ``eval_U{budget}_d{budget}_w{w}_seed{s}
                         .jsonl`` per window.
``--mode summary``       per-budget x per-window table + pooled rate + exact
                         Clopper-Pearson 95 % + bomb usage + P1/P2/P3 and the
                         S-unified / S-switch / S-mid stop condition.
``--mode audio-audit``   AGENTS.md §9.3 on this round's own data dirs.
``--mode process-audit`` this round's own engine processes must be 0.
``--mode publish``       copy the raw artifacts into
                         ``Document/Update/Alpha/v8.20.0/evidence/UNIFIED/`` and
                         write ``manifest.json``.

The ``--tag`` rule (v8.17.0's failure mode)
-------------------------------------------
Every process **must** pass its own unique ``--tag``.  v8.17.0 launched 18
evaluations inside the same second, all of them derived the data directory from
a timestamp, and 10 of 18 cells died on a missing ``th08.dat`` / an audible
``th08.cfg`` because they were sharing one private data dir.  Here the private
data dir is ``<work>/data/<tag>/w<window>`` and the engine log dir is
``<work>/runs/<tag>``, so a tag collision is the only way two processes can
collide; ``--tag`` is therefore recorded in every row and refused if empty.

Not in this file
----------------
* No training, no collection, no labels, no other arm, no pixel arm.
* No claim that the policy learned to bomb: the bomb is an external rule here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WORK = REPO / ".tmp_v8l0_unified"
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.20.0" / "evidence"
           / "UNIFIED")

SCHEMA_ASSERT = "v8l0-unified-ckpt-assertions/1"
SCHEMA_D3 = "v8l0-unified-d3-gate/1"
SCHEMA_SMOKE = "v8l0-unified-smoke/1"
SCHEMA_EPISODE = "v8l0-unified-episode/1"
SCHEMA_SEED = "v8l0-unified-budget-seed-progress/1"
SCHEMA_SUMMARY = "v8l0-unified-summary/1"
SCHEMA_AUDIO = "v8l0-unified-audio-audit/1"
SCHEMA_PROC = "v8l0-unified-process-audit/1"
SCHEMA_MANIFEST = "v8l0-unified-manifest/1"
SCHEMA_PUBLISH = "v8l0-unified-publish/1"
SCHEMA_PREFLIGHT = "v8l0-unified-preflight/1"
PRODUCER = ("tools/rl/v8l0_unified.py (live engine, HarnessClient role=shadow, "
            "episode loop = tools/rl/v830_bombpol.py::run_episode36)")
PREREG_REL = "Document/Update/Alpha/v8.20.0/PREREGISTRATION_UNIFIED_001.md"

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- the frozen protocol (prereg §1) -----------------------------------------
SEEDS: tuple[int, ...] = (0, 1, 2)
WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
EPISODES = 20
BUDGETS: tuple[int, ...] = (600, 900, 1200)
DECISION_FRAMES = 2
PREFIX_FRAMES = 8663
N_CAND = 18
N_ACTIONS = 36
TEMPERATURE = 1.0
GEN_CONST = 7

# -- the **fixed** rule: v8.19.0's S1, not swept this round (prereg §1) ------
RULE_NAME = "S1"
THRESHOLDS: dict[int, float] = {3: 60.0, 2: 40.0, 1: 20.0}
RULE_LABEL = "60/40/20"

# -- identities this round must find (fail closed) ---------------------------
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
    "tools/rl/v8d0_h2s.py",
    "tools/rl/v8k0_rulesweep30.py",
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)

# -- the references this round is placed beside (numbers only, never re-run) --
#: v8.12.0 H2s at 600 decisions with 80/60/40 — the same candidate head, the
#: same 12 (seed, window) cells, n = 240.  Re-derived below from
#: ``.tmp_v8d0_h2s/eval/eval_H2s_w{11,12,17,18}_seed{0,1,2}.jsonl`` (12 x 20
#: rows): per window 60/58/57/60, pooled **236/240 = 98.3333 %**.
H2S_600_REFERENCE: dict[str, Any] = {
    "arm": "v8.12.0 H2s — same Policy(106,64,36), rule 80/60/40 (3 -> 80, 2 -> 60, "
           "1 -> 40 px)",
    "budget_decisions": 600,
    "pooled": (236, 240),
    "per_window": {11: (60, 60), 12: (58, 60), 17: (57, 60), 18: (60, 60)},
    "source": (".tmp_v8d0_h2s/eval/eval_H2s_w{11,12,17,18}_seed{0,1,2}.jsonl "
               "(12 x 20 rows, re-derived here)"),
    "rerun_here": False,
}
#: v8.19.0 S1 at 900 decisions — an **identical** configuration to this round's
#: U-900 (same head, same 60/40/20, same generator seeds, same 12 cells), so
#: U-900 is a reproducibility check, not a new arm.  Re-derived below from
#: ``.tmp_v8k0_rulesweep30/eval/eval_S1_d900_w{11,12,17,18}_seed{0,1,2}.jsonl``
#: (12 x 20 rows): per window 16/11/11/20, pooled **58/240 = 24.1667 %**.
V819_S1_900_REFERENCE: dict[str, Any] = {
    "arm": "v8.19.0 S1 (60/40/20) at 900 decisions — identical configuration",
    "budget_decisions": 900,
    "pooled": (58, 240),
    "per_window": {11: (16, 60), 12: (11, 60), 17: (11, 60), 18: (20, 60)},
    "source": (".tmp_v8k0_rulesweep30/eval/eval_S1_d900_w{11,12,17,18}_"
               "seed{0,1,2}.jsonl (12 x 20 rows, re-derived here)"),
    "rerun_here": False,
}
#: The published 40 s cell the prereg puts U-1200 beside: the 80 px rule on the
#: same 4 windows x 300 episodes x 3 seeds = 3,600 episodes, from v8.1.0.
R5_1200_REFERENCE: dict[str, Any] = {
    "arm": "v8.1.0 A3 / R5 `80 px` at 1,200 decisions (the published 40 s cell)",
    "budget_decisions": 1200,
    "pooled": (36, 3600),
    "per_window": None,
    "source": ("Document/Update/Alpha/v8.1.0/evidence/BOMBFAR/ANALYSIS.md §97 "
               "(36/3,600 = 1.0000 %, CP95 [0.7013 %, 1.3818 %])"),
    "rerun_here": False,
}
P1_THRESHOLD = 0.95          # prereg §2 P1
P1_SWITCH_THRESHOLD = 0.90   # prereg §2.1 S-switch
P3_SUBSTANTIVE = 0.03        # prereg §2 P3
P2_ALPHA = 0.05

_LOG_FH: Any = None
_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class UnifiedError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v8l0] {message}"
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
        raise UnifiedError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v450():
    if "v450" not in _CACHE:
        module = load_module("v450_cf_label",
                             REPO / "tools" / "rl" / "v450_cf_label.py")
        module._LOG_FH = _LOG_FH
        _CACHE["v450"] = module
    return _CACHE["v450"]


def arm_a():
    if "arm_a" not in _CACHE:
        _CACHE["arm_a"] = load_module("v431_arm_a",
                                      REPO / "tools" / "rl" / "v431_arm_a.py")
    return _CACHE["arm_a"]


def m800():
    if "m800" not in _CACHE:
        module = load_module("v800_bomb", REPO / "tools" / "rl" / "v800_bomb.py")
        module._LOG_FH = _LOG_FH
        _CACHE["m800"] = module
    return _CACHE["m800"]


def v830():
    if "v830" not in _CACHE:
        module = load_module("v830_bombpol",
                             REPO / "tools" / "rl" / "v830_bombpol.py")
        module._LOG_FH = _LOG_FH
        _CACHE["v830"] = module
    return _CACHE["v830"]


def v860():
    if "v860" not in _CACHE:
        module = load_module("v860_bombdag",
                             REPO / "tools" / "rl" / "v860_bombdag.py")
        module._LOG_FH = _LOG_FH
        _CACHE["v860"] = module
    return _CACHE["v860"]


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
    """v4.5.0 §2.3: ``torch.Generator().manual_seed(seed * 1000 + 7)``."""

    return int(seed) * 1000 + GEN_CONST


def thresholds() -> dict[int, float]:
    return dict(THRESHOLDS)


def check_budget(budget: int) -> int:
    value = int(budget)
    if value not in BUDGETS:
        raise UnifiedError(f"this round fixes the budget to one of {BUDGETS}, "
                           f"got {value}")
    return value


def arm_of(budget: int) -> str:
    return f"U{int(budget)}"


def eval_file(budget: int, window: int, seed: int) -> pathlib.Path:
    budget = int(budget)
    return WORK / "eval" / (f"eval_U{budget}_d{budget}_w{int(window)}"
                            f"_seed{int(seed)}.jsonl")


# --------------------------------------------------------------------------- #
# exact Clopper-Pearson 95 % (pure Python, no scipy)
# --------------------------------------------------------------------------- #
def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _beta_quantile(p: float, a: float, b: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if betainc(a, b, mid) < p:
            low = mid
        else:
            high = mid
        if high - low < 1e-12:
            break
    return 0.5 * (low + high)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> dict[str, Any]:
    """Exact (Clopper-Pearson) two-sided 1-alpha interval for k / n."""

    if n == 0:
        return {"n": 0, "k": 0, "point": None, "low": None, "high": None}
    point = k / n
    low = 0.0 if k == 0 else _beta_quantile(alpha / 2.0, k, n - k + 1)
    high = 1.0 if k == n else _beta_quantile(1.0 - alpha / 2.0, k + 1, n - k)
    return {"n": n, "k": k, "point": round(point, 6),
            "low": round(low, 6), "high": round(high, 6),
            "confidence": 1.0 - alpha}


def intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool | None:
    if not left or not right or left.get("low") is None or right.get("low") is None:
        return None
    return bool(left["low"] <= right["high"] and right["low"] <= left["high"])


# --------------------------------------------------------------------------- #
# exact one-sided Fisher (pure Python, no scipy)
# --------------------------------------------------------------------------- #
def fisher_exact(a: int, b: int, c: int, d: int,
                 direction: str = "greater") -> dict[str, Any]:
    """One-sided Fisher exact test for the 2x2 table ``[[a, b], [c, d]]``.

    ``a``/``c`` are the survivals of the arm under test and of the reference,
    ``b``/``d`` their failures.  ``direction="greater"`` tests
    ``rate(arm) > rate(ref)`` -> ``p = P(X >= a)``; ``direction="less"`` tests
    ``rate(arm) < rate(ref)`` -> ``p = P(X <= a)``, with ``X ~ Hypergeometric
    (N = a+b+c+d, K = a+c, n = a+b)``.
    """

    n1, n2 = int(a) + int(b), int(c) + int(d)
    total = n1 + n2
    m = int(a) + int(c)
    out: dict[str, Any] = {
        "table": [[int(a), int(b)], [int(c), int(d)]],
        "arm": {"full": int(a), "n": n1,
                "rate": round(a / n1, 6) if n1 else None},
        "ref": {"full": int(c), "n": n2,
                "rate": round(c / n2, 6) if n2 else None},
        "difference_pp": (round(100.0 * (a / n1 - c / n2), 4)
                          if n1 and n2 else None),
        "alternative": f"arm {'>' if direction == 'greater' else '<'} ref "
                       f"(one-sided)",
        "p_value": None, "odds_ratio": None, "odds_ratio_haldane": None,
    }
    if n1 == 0 or n2 == 0 or total == 0:
        return out
    # support of X ~ Hypergeometric(N = total, K = m, n = n1)
    lower = max(0, n1 - (total - m))
    upper = min(n1, m)
    logs = [_log_comb(m, x) + _log_comb(total - m, n1 - x)
            for x in range(lower, upper + 1)]
    peak = max(logs)
    denominator = math.exp(_log_comb(total, n1) - peak)
    if direction == "greater":
        # P(X >= a) clipped to the support
        start = max(int(a), lower)
        if start > upper:
            out["p_value"] = 0.0 if int(a) > upper else 1.0
        else:
            numerator = sum(math.exp(logs[x - lower] - peak)
                            for x in range(start, upper + 1))
            out["p_value"] = float(min(1.0, numerator / denominator))
    else:
        # P(X <= a) clipped to the support
        top = min(int(a), upper)
        if top < lower:
            out["p_value"] = 0.0 if int(a) < lower else 1.0
        else:
            numerator = sum(math.exp(logs[x - lower] - peak)
                            for x in range(lower, top + 1))
            out["p_value"] = float(min(1.0, numerator / denominator))
    if int(b) * int(c) > 0:
        out["odds_ratio"] = float((int(a) * int(d)) / (int(b) * int(c)))
    out["odds_ratio_haldane"] = float(
        ((int(a) + 0.5) * (int(d) + 0.5)) / ((int(b) + 0.5) * (int(c) + 0.5)))
    return out


def fisher_greater(a: int, b: int, c: int, d: int) -> dict[str, Any]:
    return fisher_exact(a, b, c, d, "greater")


def fisher_less(a: int, b: int, c: int, d: int) -> dict[str, Any]:
    return fisher_exact(a, b, c, d, "less")


# --------------------------------------------------------------------------- #
# the arm: the H2s candidate sampler + the fixed 60/40/20 rule
# --------------------------------------------------------------------------- #
def process_generator(seed: int) -> Any:
    """The **one** generator of a process, created before the window loop.

    One ``torch.Generator`` per (budget, seed) process, consumed once per
    decision across every window and episode — the frozen H2s sampler.
    """

    import torch

    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    return generator


def candidate_sampled(logits: Any, generator: Any) -> tuple[int, float, float,
                                                            float]:
    """H2s: sample ``cand ~ softmax_c max(logit[2c], logit[2c+1])`` (T = 1.0)."""

    import torch

    pair = logits.view(N_CAND, 2)
    scores = pair.max(dim=1).values
    ordered = torch.sort(scores, descending=True).values
    best = float(ordered[0].item())
    second = float(ordered[1].item()) if ordered.numel() > 1 else float("nan")
    probs = torch.softmax(scores / TEMPERATURE, dim=-1)
    cand = int(torch.multinomial(probs, 1, generator=generator).item())
    return (cand, best, round(best - second, 6),
            round(float(probs[cand].item()), 6))


def new_audit(budget: int) -> dict[str, Any]:
    return {"cand_counts": [0] * N_CAND, "cand_prob": [],
            "threshold_counts": {str(key): 0 for key in (3, 2, 1, 0)},
            "rule": {"distance_true_bombs_positive": 0,
                     "distance_true_bombs_zero": 0,
                     "distance_false": 0},
            "emit_threshold_px": [], "emit_nearest_px": [],
            "arms_thresholds": {str(key): float(value)
                                for key, value in thresholds().items()},
            "budget": int(budget),
            "margins": [], "best_scores": []}


def rule_decision(state: dict[str, Any], hazards: Any) -> dict[str, Any]:
    """The fixed rule: ``nearest_all_px < THRESHOLDS[bombs]`` and ``bombs > 0``.

    The distance is v8.0.0's own ``nearest_bullet_distances`` (the measurement
    R5 uses); the threshold triple is v8.19.0's S1 = 60 / 40 / 20 px.
    """

    table = thresholds()
    distances = m800().nearest_bullet_distances(state, hazards)
    nearest = distances["nearest_all_px"]
    bombs_before = int(state["bombs_remaining"] or 0)
    threshold = table.get(bombs_before)
    distance_ok = bool(nearest is not None and threshold is not None
                       and float(nearest) < float(threshold))
    fired = bool(distance_ok and bombs_before > 0)
    return {"bomb": 1 if fired else 0, "nearest_all_px": nearest,
            "threshold_px": threshold, "bombs_remaining": bombs_before,
            "distance_ok": distance_ok}


def make_decide(*, policy: Any, generator: Any,
                audit: dict[str, Any]) -> Any:
    """``decide(state, hazards, decision) -> (cand, bomb)`` for the fixed rule."""

    import torch

    A = arm_a()

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        feats, _detail = A.build_features(state=state, hazards=hazards,
                                          decision_index=decision)
        with torch.no_grad():
            logits = policy(torch.from_numpy(feats).unsqueeze(0))[0]
        cand, best, margin, prob = candidate_sampled(logits, generator)
        rule = rule_decision(state, hazards)
        audit["cand_counts"][int(cand)] += 1
        audit["cand_prob"].append(float(prob))
        audit["margins"].append(float(margin))
        audit["best_scores"].append(float(best))
        key = str(int(rule["bombs_remaining"]))
        if key in audit["threshold_counts"]:
            audit["threshold_counts"][key] += 1
        if rule["distance_ok"]:
            if int(rule["bombs_remaining"]) > 0:
                audit["rule"]["distance_true_bombs_positive"] += 1
                audit["emit_threshold_px"].append(float(rule["threshold_px"]))
                audit["emit_nearest_px"].append(
                    None if rule["nearest_all_px"] is None
                    else float(rule["nearest_all_px"]))
            else:
                audit["rule"]["distance_true_bombs_zero"] += 1
        else:
            audit["rule"]["distance_false"] += 1
        return int(cand), int(rule["bomb"])

    return decide


def load_policy(seed: int) -> tuple[Any, pathlib.Path, str]:
    path = V860_ROOT / f"seed{int(seed)}" / "bombdag.pt"
    if not path.is_file():
        raise UnifiedError(f"seed {seed}: checkpoint not found: {path}")
    digest = sha256_file(path)
    expected = V860_SHA256[int(seed)]
    if digest != expected:
        raise UnifiedError(f"seed {seed}: {path} sha256 {digest[:16]} != "
                           f"{expected[:16]}")
    return v860().load_policy36(path, seed=seed), path, digest


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
    return v450().start_parent(slot=int(slot), window=int(window), engine=engine,
                               data_dir=data_dir, log_dir=log_dir)


def run_one_episode(*, parent: Any, episode: int, seed: int, budget: int,
                    policy: Any, generator: Any, tag: str) -> dict[str, Any]:
    """One episode through the frozen loop, with the fixed rule's audit."""

    audit = new_audit(budget)
    decide = make_decide(policy=policy, generator=generator, audit=audit)
    record = v830().run_episode36(parent, episode=int(episode),
                                  policy_seed=int(seed), decisions=int(budget),
                                  decide=decide, budget=int(budget),
                                  decision_frames=DECISION_FRAMES)
    probs = audit["cand_prob"] or [0.0]
    emit_ok = [bool(nearest is not None
                    and float(nearest) < float(threshold))
               for nearest, threshold in zip(audit["emit_nearest_px"],
                                             audit["emit_threshold_px"])]
    row = {
        "schema": SCHEMA_EPISODE, "producer": PRODUCER,
        "arm": arm_of(budget), "budget": int(budget),
        "rule": {"name": RULE_NAME, "label": RULE_LABEL,
                 "thresholds": {str(key): float(value)
                                for key, value in thresholds().items()}},
        "arm_thresholds": {str(key): float(value)
                           for key, value in thresholds().items()},
        "thresholds_label": RULE_LABEL,
        "tag": tag,
        "episode": int(episode), "window": int(parent.window),
        "policy_seed": int(seed), "budget_decisions": int(budget),
        "decision_frames": DECISION_FRAMES, "prefix_frame": PREFIX_FRAMES,
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
        "bomb_cands": record["bomb_cands"],
        "bomb_ignored": record["bomb_ignored"],
        "min_nearest_px": record["min_nearest_px"],
        "actions_sampled": record["actions_sampled"],
        "invalid": bool(record["invalid"]), "error": record["error"],
        "wall_s": record["wall_s"], "spawn_ms": record["spawn_ms"],
        "parent_alive": record["parent_alive"],
        # -- the arm's own audit: the candidate was SAMPLED and the bomb came
        #    from the fixed 60/40/20 triple (not from the policy's bomb head)
        "cand_counts": audit["cand_counts"],
        "distinct_cands": sum(1 for count in audit["cand_counts"] if count),
        "cand_prob": {"mean": round(statistics.fmean(probs), 6),
                      "min": round(min(probs), 6),
                      "max": round(max(probs), 6)},
        "cand_margin": quantiles(audit["margins"]),
        "cand_best_score": quantiles(audit["best_scores"]),
        "decisions_by_bombs_remaining": audit["threshold_counts"],
        "emit_threshold_px": audit["emit_threshold_px"],
        "rule_counts": audit["rule"],
        "samples_consumed": sum(audit["cand_counts"]),
    }
    row["checks"] = {
        # the loop calls ``decide`` once per measured decision; on a death the
        # fatal decision is measured but not counted in ``decisions_executed``
        # (``run_episode36`` breaks before its increment), hence the +1 slack.
        "one_sample_per_decision": bool(
            row["samples_consumed"] == row["decisions_measured"]
            and row["decisions_measured"] in (row["decisions_executed"],
                                              row["decisions_executed"] + 1)),
        "rule_bomb_matches_emits": bool(
            row["rule_counts"]["distance_true_bombs_positive"] == row["bomb_emits"]),
        "counter_bookkeeping_closes": bool(
            int(row["bombs_start"]) - int(row["bombs_end"])
            == int(row["bombs_used"] or 0)),
        "frames_consistent": bool(
            int(row["frames"]) - int(row["decisions_executed"]) * DECISION_FRAMES
            in ((0,) if row["survived"] else (1, 2))),
        # every emitted bomb really was inside the fixed threshold window
        "threshold_respected": bool(
            len(emit_ok) == int(row["bomb_emits"]) and all(emit_ok)),
    }
    if not all(row["checks"].values()):
        row["invalid"] = True
        row["error"] = row["error"] or f"audit check failed: {row['checks']}"
    return row


# --------------------------------------------------------------------------- #
# mode: preflight (engine census, AGENTS.md "> 6 engines -> wait 5 min")
# --------------------------------------------------------------------------- #
def engine_census() -> dict[str, Any]:
    proc = subprocess.run(["pgrep", "-af", "th08-modern"],
                          capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    engines = [line for line in lines
               if "th08-modern" in line and "pgrep" not in line
               and "rl-role" in line]
    mine = [line for line in engines if ".tmp_v8l0_unified" in line]
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
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("count the engine processes on the box before launching; "
                        "the project rule is '> 6 -> wait 5 minutes' and "
                        "'never pkill -f another round's processes'"),
        "phase": args.phase, "nproc": os.cpu_count(), "census": census,
        "history": history,
        "free_gb": None, "waited_s": float(args.waited_s or 0.0),
    }
    try:
        load = pathlib.Path("/proc/loadavg").read_text(encoding="utf-8").split()
        payload["loadavg"] = load[:3]
    except OSError:
        payload["loadavg"] = None
    write_json(out_path, payload)
    log(f"preflight {args.phase}: engines={census['engine_processes_total']} "
        f"(own={census['matching']}, other rounds="
        f"{census['other_round_processes']}) history={len(history)} "
        f"waited={payload['waited_s']}s -> {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# mode: assert-ckpt
# --------------------------------------------------------------------------- #
def mode_assert(args: argparse.Namespace) -> int:
    out_path = (pathlib.Path(args.out) if args.out
                else WORK / f"ckpt_assertions_{args.phase}.json")
    payload: dict[str, Any] = {
        "schema": SCHEMA_ASSERT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": args.phase,
        "requirement": ("the unified round's candidate head is the frozen v8.6.0 "
                        "`bombdag.pt` Policy(106,64,36) (9c6d5d4a / 36a51090 / "
                        "885dde70); its behaviour policy is the frozen v4.5.0 `S` "
                        "student (round2.pt = 4abee069 / a93746bb / c69960fc); "
                        "the engine and the protected sources must not have "
                        "moved"),
        "rule": {"name": RULE_NAME, "label": RULE_LABEL,
                 "thresholds": {str(k): float(v)
                                for k, v in thresholds().items()}},
        "budgets": list(BUDGETS),
        "engine": None, "v860_bombdag": {}, "v450_round2": {}, "protected": {},
        "self": {}, "prereg": {}, "all_pass": False,
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
        me = pathlib.Path(__file__).resolve()
        payload["self"] = {"path": str(me), "sha256": sha256_file(me)}
        prereg = REPO / PREREG_REL
        payload["prereg"] = {"path": PREREG_REL,
                             "sha256": (sha256_file(prereg)
                                        if prereg.is_file() else None)}
        payload["all_pass"] = bool(ok)
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
    write_json(out_path, payload)
    log(f"assert-ckpt {args.phase}: all_pass={payload['all_pass']} "
        f"engine={payload['engine'].get('match') if payload['engine'] else None} "
        f"-> {out_path}")
    return 0 if payload["all_pass"] else 2


# --------------------------------------------------------------------------- #
# mode: d3 (the 5-assertion fidelity gate, through the frozen v432 script)
# --------------------------------------------------------------------------- #
def mode_d3(args: argparse.Namespace) -> int:
    out_path = pathlib.Path(args.out) if args.out else WORK / "d3_gate.json"
    raw_path = WORK / "fidelity_d3.json"
    tag = args.tag or "v8l0-fidelity"
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
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("D3 fidelity, 5 assertions (FOR-001 §2.1-§2.5): HELLO "
                        "role=shadow, DEBUG threads=1 before/after PARK, 2,165 "
                        "checkpoints to frame 8663 with 0 mismatches, the R24 18 "
                        "constants 18/18, PARK frame=8663.  A non-5/5 gate voids "
                        "the round."),
        "tag": tag, "argv": argv, "exit": proc.returncode,
        "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()[-4000:],
        "wall_s": wall, "raw": str(raw_path),
        "raw_sha256": (sha256_file(raw_path) if raw_path.is_file() else None),
        "assertions_present": raw.get("assertions_present"),
        "checks": checks,
        "passed": sum(1 for value in checks.values() if value["passed"]),
        "total": len(checks), "verdict": raw.get("verdict"),
        "engine": raw.get("engine"), "data_dir": raw.get("data_dir"),
    }
    payload["all_pass"] = bool(proc.returncode == 0
                               and payload["passed"] == 5
                               and payload["total"] == 5
                               and "PASS" in str(payload["verdict"]))
    write_json(out_path, payload)
    for key in sorted(checks):
        log(f"d3 {key}: {'PASS' if checks[key]['passed'] else 'FAIL'}")
    log(f"d3: {payload['passed']}/{payload['total']} PASS "
        f"all_pass={payload['all_pass']} wall={wall}s -> {out_path}")
    return 0 if payload["all_pass"] else 2


# --------------------------------------------------------------------------- #
# mode: smoke (one episode: U-600 / seed0 / w11 / 600 decisions)
# --------------------------------------------------------------------------- #
def mode_smoke(args: argparse.Namespace) -> int:
    seed = int(args.policy_seed)
    windows = [int(part) for part in str(args.windows).split(",") if part.strip()]
    window = windows[0]
    budget = check_budget(args.decisions)
    tag = args.tag or f"v8l0-smoke-U{budget}-s{seed}-w{window}"
    out_path = pathlib.Path(args.out) if args.out else WORK / "smoke.json"
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    payload: dict[str, Any] = {
        "schema": SCHEMA_SMOKE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": (f"one episode of {arm_of(budget)} ({RULE_LABEL} px, "
                        f"seed{seed}/w{window}, {budget} decisions) with the "
                        f"sampler/rule audit, before the full 240-episode runs"),
        "arm": arm_of(budget), "budget": budget,
        "rule": {"name": RULE_NAME, "label": RULE_LABEL,
                 "thresholds": {str(k): float(v)
                                for k, v in thresholds().items()}},
        "arm_thresholds": {str(k): float(v) for k, v in thresholds().items()},
        "thresholds_label": RULE_LABEL,
        "policy_seed": seed, "window": window,
        "budget_decisions": budget, "episodes": 1,
        "decision_frames": DECISION_FRAMES, "tag": tag,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest,
                 "bytes": ckpt.stat().st_size},
        "row": None, "checks": {}, "all_pass": False, "error": None,
        "wall_s": None, "cfg": None, "gate": None, "data_dir": None,
    }
    parent: Any = None
    started = time.perf_counter()
    try:
        data_root = WORK / "data" / tag
        data_dir, data_info = v450().prepare_data_dir(data_root, window,
                                                      args.data_source)
        payload["data_dir"] = str(data_dir)
        payload["cfg"] = data_info["cfg"]
        parent, gate = start_parent(slot=0, window=window, engine=args.engine,
                                    data_dir=data_dir,
                                    log_dir=WORK / "runs" / tag)
        payload["gate"] = {key: gate[key] for key in (
            "hello_role_shadow", "park_ok", "park_frame", "threads_1",
            "all_pass")}
        payload["gate_prefix"] = {key: gate["prefix"][key] for key in (
            "hash_checks", "hash_mismatches", "final_frame")}
        generator = process_generator(seed)
        row = run_one_episode(parent=parent, episode=0, seed=seed,
                              budget=budget, policy=policy,
                              generator=generator, tag=tag)
        row["ckpt_sha256"] = digest
        row["engine_sha256"] = engine.get("sha256")
        payload["row"] = row
        payload["checks"] = {
            "survived": row["survived"],
            "bombs_used": row["bombs_used"],
            "alive_frames": row["alive_frames"],
            "frames": row["frames"],
            "decisions_executed": row["decisions_executed"],
            "bomb_emits": row["bomb_emits"],
            "bomb_decisions": row["bomb_decisions"],
            "cfg_muted_four_zero": payload["cfg"]["muted_four_zero"],
            "gate_all_pass": payload["gate"]["all_pass"],
            **row["checks"]}
        payload["all_pass"] = bool(
            not row["invalid"]
            and row["decisions_measured"] >= 1
            and row["samples_consumed"] >= 1
            and payload["checks"]["cfg_muted_four_zero"]
            and payload["checks"]["gate_all_pass"]
            and all(row["checks"].values()))
        log(f"SMOKE {arm_of(budget)} ({RULE_LABEL}) s{seed} w{window} ep0: "
            f"survived={row['survived']} bombs_used={row['bombs_used']} "
            f"alive_frames={row['alive_frames']} frames={row['frames']} "
            f"decisions={row['decisions_executed']} "
            f"bomb_emits={row['bomb_emits']} bomb_decisions={row['bomb_decisions']} "
            f"nearest_at_emit={row['nearest_px_at_emit']} "
            f"distinct_cands={row['distinct_cands']} checks={row['checks']}")
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
        log(f"SMOKE FAILED: {payload['error']}")
    finally:
        if parent is not None:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        payload["wall_s"] = round(time.perf_counter() - started, 3)
    write_json(out_path, payload)
    log(f"smoke: all_pass={payload['all_pass']} wall={payload['wall_s']}s "
        f"-> {out_path}")
    return 0 if payload["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: eval (one budget x one seed: 4 windows x 20 episodes)
# --------------------------------------------------------------------------- #
def mode_eval(args: argparse.Namespace) -> int:
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    episodes = int(args.episodes)
    if seed not in SEEDS:
        raise UnifiedError(f"--policy-seed must be one of {SEEDS}, got {seed}")
    if budget not in BUDGETS:
        raise UnifiedError(f"this round fixes --decisions to one of {BUDGETS}, "
                           f"got {budget}")
    if episodes != EPISODES:
        raise UnifiedError(f"this round fixes --episodes to {EPISODES} per "
                           f"window, got {episodes}")
    windows = tuple(int(part) for part in str(args.windows).split(",")
                    if part.strip())
    for window in windows:
        if window not in WINDOWS:
            raise UnifiedError(f"window {window} is not one of {WINDOWS}")
    if not str(args.tag or "").strip():
        raise UnifiedError(
            "--tag is mandatory: v8.17.0's 18-way launch shared one timestamp "
            "data dir and corrupted 10 of 18 cells; every process needs its "
            "own tag")
    tag = str(args.tag).strip()
    arm = arm_of(budget)
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    run_dir = WORK / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = WORK / "data" / tag
    progress_path = WORK / "eval" / f"U{budget}_d{budget}_s{seed}_progress.json"
    progress: dict[str, Any] = {
        "schema": SCHEMA_SEED, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": arm, "budget": budget,
        "rule": {"name": RULE_NAME, "label": RULE_LABEL,
                 "thresholds": {str(k): float(v)
                                for k, v in thresholds().items()}},
        "arm_thresholds": {str(k): float(v) for k, v in thresholds().items()},
        "thresholds_label": RULE_LABEL,
        "policy_seed": seed, "budget_decisions": budget,
        "episodes_per_window": episodes, "windows": list(windows), "tag": tag,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest},
        "generator": {"seed": generator_seed(seed),
                      "kind": ("one torch.Generator per (budget, seed) process, "
                               "consumed once per decision across every window "
                               "and episode")},
        "cells": {}, "status": "running", "error": None, "wall_s": None,
    }
    started = time.perf_counter()
    from touhou_rl.v420 import contracts as C

    generator = process_generator(seed)
    try:
        for window in windows:
            C.assert_seed_allowed(int(window), "train")
            out_path = eval_file(budget, window, seed)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data_dir, data_info = v450().prepare_data_dir(data_root, window,
                                                          args.data_source)
            if not data_info["cfg"]["muted_four_zero"]:
                raise UnifiedError(f"w{window}: the generated th08.cfg is not "
                                   f"muted: {data_info['cfg']}")
            log(f"{arm} ({RULE_LABEL}) s{seed} w{window}: cfg muted="
                f"{data_info['cfg']['muted_four_zero']} "
                f"sha256={data_info['cfg']['sha256'][:16]} tag={tag} "
                f"data_dir={data_dir}")
            parent, gate = start_parent(slot=int(window), window=window,
                                        engine=args.engine, data_dir=data_dir,
                                        log_dir=run_dir)
            rows: list[dict[str, Any]] = []
            started_window = time.perf_counter()
            try:
                with out_path.open("w", encoding="utf-8") as handle:
                    for episode in range(episodes):
                        row = run_one_episode(parent=parent, episode=episode,
                                              seed=seed, budget=budget,
                                              policy=policy,
                                              generator=generator, tag=tag)
                        row["ckpt_sha256"] = digest
                        row["engine_sha256"] = engine.get("sha256")
                        row["cfg_sha256"] = data_info["cfg"]["sha256"]
                        row["cfg_muted_four_zero"] = \
                            data_info["cfg"]["muted_four_zero"]
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        rows.append(row)
                        log(f"  {arm} s{seed} w{window} ep{episode}: "
                            f"survived={row['survived']} "
                            f"bombs_used={row['bombs_used']} "
                            f"alive={row['alive_frames']} "
                            f"distinct_cands={row['distinct_cands']} "
                            f"wall={row['wall_s']}s")
                        if row["invalid"]:
                            raise UnifiedError(
                                f"{arm} s{seed} w{window} ep{episode} invalid: "
                                f"{row['error']}")
            finally:
                try:
                    parent.close()
                except Exception:                                     # noqa: BLE001
                    pass
            survived = sum(1 for row in rows if row["survived"])
            progress["cells"][str(window)] = {
                "window": int(window), "episodes": len(rows),
                "survived": survived,
                "rate": round(survived / len(rows), 6) if rows else None,
                "bombs_used_total": sum(int(row["bombs_used"] or 0)
                                        for row in rows),
                "cfg_muted_four_zero": data_info["cfg"]["muted_four_zero"],
                "cfg_sha256": data_info["cfg"]["sha256"],
                "data_dir": str(data_dir),
                "gate": {key: gate[key] for key in (
                    "hello_role_shadow", "park_ok", "park_frame", "threads_1",
                    "all_pass")},
                "gate_prefix": {key: gate["prefix"][key] for key in (
                    "hash_checks", "hash_mismatches", "final_frame")},
                "wall_s": round(time.perf_counter() - started_window, 3),
                "out": str(out_path),
                "out_sha256": sha256_file(out_path),
                "out_bytes": out_path.stat().st_size}
            progress["status"] = "partial"
            progress["wall_s"] = round(time.perf_counter() - started, 3)
            write_json(progress_path, progress)
            log(f"{arm} s{seed} w{window}: {survived}/{len(rows)} survived "
                f"({progress['cells'][str(window)]['rate']}) wall="
                f"{progress['cells'][str(window)]['wall_s']}s")
        progress["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        progress["error"] = f"{type(exc).__name__}: {exc}"
        progress["status"] = "failed"
        log(f"EVAL FAILED {arm} s{seed}: {progress['error']}")
    finally:
        progress["wall_s"] = round(time.perf_counter() - started, 3)
        write_json(progress_path, progress)
    log(f"eval {arm} ({RULE_LABEL}) s{seed}: status={progress['status']} cells="
        f"{ {k: (v['survived'], v['episodes']) for k, v in progress['cells'].items()} } "
        f"wall={progress['wall_s']}s")
    return 0 if progress["status"] == "complete" else 4


# --------------------------------------------------------------------------- #
# mode: summary (the round's answer)
# --------------------------------------------------------------------------- #
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
        "bomb_emits_total": sum(int(row["bomb_emits"] or 0) for row in rows),
        "first_bomb_decision": quantiles([float(v) for v in firsts])
        if firsts else {"n": 0},
        "last_bomb_decision": quantiles([float(v) for v in lasts])
        if lasts else {"n": 0},
        "alive_frames": quantiles([float(row["alive_frames"] or 0)
                                   for row in rows]),
        "min_nearest_px": quantiles([float(row["min_nearest_px"])
                                     for row in rows
                                     if row.get("min_nearest_px") is not None]),
    }


def _reference_scan(kind: str) -> dict[str, Any]:
    """Re-derive a published reference from its raw rows when they are on disk."""

    if kind == "h2s600":
        template = (".tmp_v8d0_h2s/eval/eval_H2s_w{window}_seed{seed}.jsonl")
        fallback = H2S_600_REFERENCE
    elif kind == "v819s1":
        template = (".tmp_v8k0_rulesweep30/eval/eval_S1_d900_w{window}"
                    "_seed{seed}.jsonl")
        fallback = V819_S1_900_REFERENCE
    else:
        raise UnifiedError(f"unknown reference {kind!r}")
    rows: list[dict[str, Any]] = []
    frames: dict[str, Any] = {}
    for seed in SEEDS:
        for window in WINDOWS:
            path = REPO / template.format(window=window, seed=seed)
            cell = read_jsonl(path)
            frames[f"w{window}_seed{seed}"] = {
                "path": template.format(window=window, seed=seed),
                "exists": path.is_file(), "rows": len(cell),
                "full": sum(1 for row in cell if row.get("survived"))}
            rows.extend(cell)
    expected = len(SEEDS) * len(WINDOWS) * EPISODES
    if len(rows) != expected:
        return {"available": False, "reason": f"{len(rows)} rows on disk, "
                                              f"{expected} expected",
                "fallback_pooled": [int(fallback["pooled"][0]),
                                    int(fallback["pooled"][1])],
                "files": frames}
    per_window = {}
    for window in WINDOWS:
        cell = [row for row in rows if int(row["window"]) == int(window)]
        per_window[str(window)] = [sum(1 for row in cell if row["survived"]),
                                   len(cell)]
    full = sum(1 for row in rows if row.get("survived"))
    return {"available": True, "rows": len(rows), "full": full,
            "pooled": [full, len(rows)],
            "rate": round(full / len(rows), 6),
            "cp95": clopper_pearson(full, len(rows)),
            "per_window": per_window, "files": frames}


def _budget_block(budget: int, eval_dir: pathlib.Path) -> dict[str, Any]:
    expected = len(SEEDS) * len(WINDOWS) * EPISODES
    rows: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    for seed in SEEDS:
        for window in WINDOWS:
            path = eval_dir / (f"eval_U{budget}_d{budget}_w{window}"
                               f"_seed{seed}.jsonl")
            cell_rows = read_jsonl(path)
            files[f"w{window}_seed{seed}"] = {
                "name": path.name, "path": str(path), "exists": path.is_file(),
                "rows": len(cell_rows),
                "full": sum(1 for row in cell_rows if row.get("survived")),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None}
            rows.extend(cell_rows)
    per_window: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    per_cell: list[dict[str, Any]] = []
    for window in WINDOWS:
        window_rows = [row for row in rows if int(row["window"]) == int(window)]
        if not window_rows:
            continue
        survived = sum(1 for row in window_rows if row["survived"])
        per_window.append({
            "window": int(window), "n": len(window_rows), "full": survived,
            "rate": round(survived / len(window_rows), 6),
            "cp95": clopper_pearson(survived, len(window_rows)),
            "by_seed": {str(seed): {
                "n": sum(1 for row in window_rows
                         if int(row["policy_seed"]) == int(seed)),
                "full": sum(1 for row in window_rows
                            if int(row["policy_seed"]) == int(seed)
                            and row["survived"])}
                for seed in SEEDS
                if any(int(row["policy_seed"]) == int(seed)
                       for row in window_rows)},
            "bombs_used_mean": round(statistics.fmean(
                int(row["bombs_used"] or 0) for row in window_rows), 4),
            "alive_frames_mean": round(statistics.fmean(
                float(row["alive_frames"] or 0) for row in window_rows), 4),
        })
    for seed in SEEDS:
        seed_rows = [row for row in rows if int(row["policy_seed"]) == int(seed)]
        if not seed_rows:
            continue
        survived = sum(1 for row in seed_rows if row["survived"])
        per_seed[str(seed)] = {
            "n": len(seed_rows), "full": survived,
            "rate": round(survived / len(seed_rows), 6),
            "cp95": clopper_pearson(survived, len(seed_rows)),
            "bombs_used_mean": round(statistics.fmean(
                int(row["bombs_used"] or 0) for row in seed_rows), 4)}
    for window in WINDOWS:
        for seed in SEEDS:
            cell_rows = [row for row in rows
                         if int(row["window"]) == int(window)
                         and int(row["policy_seed"]) == int(seed)]
            if not cell_rows:
                continue
            survived = sum(1 for row in cell_rows if row["survived"])
            per_cell.append({"window": int(window), "policy_seed": int(seed),
                             "n": len(cell_rows), "full": survived,
                             "rate": round(survived / len(cell_rows), 6)})
    survived_all = sum(1 for row in rows if row["survived"])
    return {
        "arm": arm_of(budget), "budget": int(budget),
        "rule": {"name": RULE_NAME, "label": RULE_LABEL,
                 "thresholds": {str(k): float(v)
                                for k, v in thresholds().items()}},
        "thresholds_label": RULE_LABEL,
        "files": files, "rows": len(rows), "expected_rows": expected,
        "complete": bool(len(rows) == expected),
        "per_window_table": per_window,
        "per_seed": per_seed,
        "per_cell": per_cell,
        "pooled": {
            "n": len(rows), "full": survived_all,
            "rate": round(survived_all / len(rows), 6) if rows else None,
            "cp95": clopper_pearson(survived_all, len(rows)),
        },
        "bomb_usage": _bomb_block(rows),
        "invalid_rows": sum(1 for row in rows if row.get("invalid")),
        "wall_s_total": round(sum(float(row["wall_s"] or 0) for row in rows), 3),
        "cells_complete": sum(1 for value in files.values()
                              if value["rows"] == EPISODES),
    }


def _row_identity(eval_dir: pathlib.Path, reference_dir: pathlib.Path,
                  reference_prefix: str) -> dict[str, Any]:
    """Row-by-row identity of U-900 against v8.19.0 S1 (same 12 cells)."""

    fields = ("survived", "alive_frames", "bombs_used", "first_death_frame",
              "bomb_emits")
    mismatches = {key: 0 for key in fields}
    actions_mismatch = 0
    rows = 0
    cells = 0
    missing: list[str] = []
    for seed in SEEDS:
        for window in WINDOWS:
            mine = eval_dir / (f"eval_U900_d900_w{window}_seed{seed}.jsonl")
            ref = reference_dir / reference_prefix.format(window=window,
                                                          seed=seed)
            if not mine.is_file() or not ref.is_file():
                missing.append(f"w{window}_seed{seed}")
                continue
            left = read_jsonl(mine)
            right = read_jsonl(ref)
            if len(left) != len(right):
                missing.append(f"w{window}_seed{seed} (rows {len(left)} vs "
                               f"{len(right)})")
                continue
            cells += 1
            for row_left, row_right in zip(left, right):
                rows += 1
                for key in fields:
                    if row_left.get(key) != row_right.get(key):
                        mismatches[key] += 1
                if (row_left.get("actions_sampled") or []) != \
                        (row_right.get("actions_sampled") or []):
                    actions_mismatch += 1
    return {"rows": rows, "cells": cells, "missing": missing,
            "field_mismatches": mismatches,
            "episodes_with_different_actions_sampled": actions_mismatch,
            "identical": bool(rows == len(SEEDS) * len(WINDOWS) * EPISODES
                              and not missing
                              and not actions_mismatch
                              and not any(mismatches.values()))}


def mode_summary(args: argparse.Namespace) -> int:
    eval_dir = pathlib.Path(args.eval_dir) if args.eval_dir else WORK / "eval"
    out_path = pathlib.Path(args.out) if args.out else WORK / "summary.json"
    budgets = tuple(int(part) for part in str(args.budgets).split(",")
                    if part.strip())
    for budget in budgets:
        check_budget(budget)
    blocks: dict[str, Any] = {arm_of(b): _budget_block(b, eval_dir)
                              for b in budgets}
    h2s = _reference_scan("h2s600")
    v819 = _reference_scan("v819s1")
    r5_full, r5_n = R5_1200_REFERENCE["pooled"]

    def ref_of(name: str) -> tuple[int, int]:
        if name == "h2s600":
            pooled = h2s["pooled"] if h2s.get("available") else \
                H2S_600_REFERENCE["pooled"]
        elif name == "v819s1":
            pooled = v819["pooled"] if v819.get("available") else \
                V819_S1_900_REFERENCE["pooled"]
        else:
            pooled = R5_1200_REFERENCE["pooled"]
        return int(pooled[0]), int(pooled[1])

    # -- P1: U-600 >= 95 % -------------------------------------------------
    u600 = blocks.get(arm_of(600))
    p1_payload: dict[str, Any] = {
        "requirement": ("U-600 pooled survival >= 95 % (the tightened 60/40/20 "
                        "triple loses nothing at 20 s)"),
        "budget": 600, "threshold": P1_THRESHOLD,
        "observed": None, "cp95": None, "passed": False,
        "versus_h2s_80_60_40": None, "versus_h2s_fisher_less": None,
    }
    if u600 and u600["rows"]:
        pooled = u600["pooled"]
        full, n = int(pooled["full"]), int(pooled["n"])
        ref_full, ref_n = ref_of("h2s600")
        p1_payload["observed"] = pooled["rate"]
        p1_payload["cp95"] = pooled["cp95"]
        p1_payload["passed"] = bool(u600["complete"]
                                   and pooled["rate"] >= P1_THRESHOLD)
        p1_payload["versus_h2s_80_60_40"] = {
            "reference": {"full": ref_full, "n": ref_n,
                          "rate": round(ref_full / ref_n, 6),
                          "arm": H2S_600_REFERENCE["arm"]},
            "difference_pp": round(100.0 * (full / n - ref_full / ref_n), 4),
            "fisher_less": fisher_less(full, n - full, ref_full,
                                       ref_n - ref_full),
            "fisher_greater": fisher_greater(full, n - full, ref_full,
                                             ref_n - ref_full),
        }
        p1_payload["versus_h2s_fisher_less"] = \
            p1_payload["versus_h2s_80_60_40"]["fisher_less"]["p_value"]

    # -- P2: U-900 reproducibility against v8.19.0 S1 ------------------------
    u900 = blocks.get(arm_of(900))
    p2_payload: dict[str, Any] = {
        "requirement": ("U-900 must reproduce v8.19.0 S1 (58/240 = 24.1667 %): "
                        "the two exact CP95 intervals must overlap"),
        "budget": 900, "observed": None, "cp95": None,
        "reference": None, "reference_cp95": None,
        "cp95_overlap": None, "passed": False, "fisher_two_sided_note": None,
    }
    if u900 and u900["rows"]:
        pooled = u900["pooled"]
        full, n = int(pooled["full"]), int(pooled["n"])
        ref_full, ref_n = ref_of("v819s1")
        ref_cp = clopper_pearson(ref_full, ref_n)
        overlap = intervals_overlap(pooled["cp95"], ref_cp)
        p2_payload["observed"] = pooled["rate"]
        p2_payload["cp95"] = pooled["cp95"]
        p2_payload["reference"] = {"full": ref_full, "n": ref_n,
                                   "rate": round(ref_full / ref_n, 6),
                                   "arm": V819_S1_900_REFERENCE["arm"]}
        p2_payload["reference_cp95"] = ref_cp
        p2_payload["cp95_overlap"] = overlap
        p2_payload["difference_pp"] = round(
            100.0 * (full / n - ref_full / ref_n), 4)
        p2_payload["fisher_two_sided_note"] = {
            "greater": fisher_greater(full, n - full, ref_full,
                                      ref_n - ref_full)["p_value"],
            "less": fisher_less(full, n - full, ref_full,
                                ref_n - ref_full)["p_value"],
            "note": ("reported for description only; P2 is an interval-overlap "
                     "repeatability check, not a superiority test"),
        }
        p2_payload["passed"] = bool(u900["complete"] and overlap)
        p2_payload["row_identity_vs_v819_S1"] = _row_identity(
            eval_dir, REPO / ".tmp_v8k0_rulesweep30" / "eval",
            "eval_S1_d900_w{window}_seed{seed}.jsonl")

    # -- P3: U-1200 beside the published 1.00 % -----------------------------
    u1200 = blocks.get(arm_of(1200))
    p3_payload: dict[str, Any] = {
        "requirement": ("U-1200 reported beside R5's published 36/3,600 = "
                        "1.00 %; >= 3 % counts as a substantive improvement"),
        "budget": 1200, "observed": None, "cp95": None,
        "reference": {"full": int(r5_full), "n": int(r5_n),
                      "rate": round(r5_full / r5_n, 6),
                      "arm": R5_1200_REFERENCE["arm"],
                      "source": R5_1200_REFERENCE["source"]},
        "difference_pp": None, "substantive": False,
        "fisher_greater": None, "passed": False,
    }
    if u1200 and u1200["rows"]:
        pooled = u1200["pooled"]
        full, n = int(pooled["full"]), int(pooled["n"])
        p3_payload["observed"] = pooled["rate"]
        p3_payload["cp95"] = pooled["cp95"]
        p3_payload["difference_pp"] = round(
            100.0 * (full / n - r5_full / r5_n), 4)
        p3_payload["substantive"] = bool(full / n >= P3_SUBSTANTIVE)
        p3_payload["fisher_greater"] = fisher_greater(
            full, n - full, int(r5_full), int(r5_n) - int(r5_full))
        p3_payload["passed"] = bool(u1200["complete"])

    # -- the stop condition (prereg §2.1) ----------------------------------
    if not all(block["complete"] for block in blocks.values()):
        stop = "incomplete"
        stop_text = ("not every budget has 240 rows; the prereg's stop "
                     "conditions are not declared on a partial table")
    elif p1_payload["passed"]:
        stop = "S-unified"
        stop_text = ("P1 holds: one threshold triple (60/40/20) covers the "
                     "20 s / 30 s budgets; it becomes the recommended "
                     "deployment configuration")
    elif p1_payload["observed"] is not None and p1_payload["observed"] < \
            P1_SWITCH_THRESHOLD:
        stop = "S-switch"
        stop_text = ("P1 fails (< 90 %): the threshold must switch with the "
                     "budget -- 20 s uses 80/60/40, 30 s uses 60/40/20")
    else:
        stop = "S-mid"
        stop_text = ("P1 lands in 90-95 %: a small loss; Fisher decides whether "
                     "it is significantly below H2s's 98.3333 %")

    table = []
    for budget in budgets:
        block = blocks[arm_of(budget)]
        row = {"budget": budget, "arm": arm_of(budget),
               "rule": RULE_LABEL,
               "pooled_full": block["pooled"]["full"],
               "pooled_n": block["pooled"]["n"],
               "pooled_rate": block["pooled"]["rate"],
               "cp95_low": block["pooled"]["cp95"]["low"],
               "cp95_high": block["pooled"]["cp95"]["high"],
               "bombs_mean": block["bomb_usage"]["mean"],
               "bombs0": block["bomb_usage"]["counts"].get("0"),
               "bombs1": block["bomb_usage"]["counts"].get("1"),
               "bombs2": block["bomb_usage"]["counts"].get("2"),
               "bombs3": block["bomb_usage"]["counts"].get("3"),
               "complete": block["complete"]}
        for bucket in block["per_window_table"]:
            row[f"w{bucket['window']}"] = f"{bucket['full']}/{bucket['n']}"
        table.append(row)

    payload: dict[str, Any] = {
        "schema": SCHEMA_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": {"path": PREREG_REL,
                   "sha256": (sha256_file(REPO / PREREG_REL)
                              if (REPO / PREREG_REL).is_file() else None)},
        "question": ("does the tightened 60/40/20 bomb rule hold up when the "
                     "decision budget is moved 600 / 900 / 1,200, i.e. is one "
                     "threshold triple enough or must it switch with the "
                     "budget?"),
        "protocol": {
            "candidate_head": ("v8.6.0 bombdag.pt Policy(106,64,36), frozen; "
                               "score[c] = max(logit[2c], logit[2c+1]); "
                               "cand ~ softmax(score / 1.0), one draw per "
                               "decision from torch.Generator(seed*1000+7)"),
            "bomb": ("nearest_all_px < THRESHOLDS[bombs_remaining] "
                     "(60 / 40 / 20 px) and bombs_remaining > 0"),
            "rule": {"name": RULE_NAME, "label": RULE_LABEL,
                     "thresholds": {str(k): float(v)
                                    for k, v in thresholds().items()}},
            "seeds": list(SEEDS), "windows": list(WINDOWS),
            "episodes_per_window": EPISODES,
            "budgets_decisions": list(budgets),
            "decision_frames": DECISION_FRAMES,
            "prefix_frame": PREFIX_FRAMES,
            "frames_per_episode": {str(b): b * DECISION_FRAMES for b in budgets},
            "loop": ("for seed in (0,1,2): for window in (11,12,17,18): for "
                     "episode in range(20): run_one_episode(budget_decisions="
                     "b) for b in budgets"),
            "episode_loop": "tools/rl/v830_bombpol.py::run_episode36 (imported)",
        },
        "references": {
            "H2s_600_80_60_40": {"metadata": H2S_600_REFERENCE,
                                 "scan": h2s},
            "v819_S1_900_60_40_20": {"metadata": V819_S1_900_REFERENCE,
                                     "scan": v819},
            "R5_1200_80px": {"metadata": R5_1200_REFERENCE, "scan": None},
        },
        "budgets": blocks,
        "table": table,
        "verdict": {
            "P1": p1_payload,
            "P2": p2_payload,
            "P3": p3_payload,
            "P4": {"requirement": "bomb-usage distribution reported "
                                  "(descriptive)",
                   "bomb_usage": {arm: block["bomb_usage"]
                                  for arm, block in blocks.items()}},
            "stop_condition": stop,
            "stop_text": stop_text,
            "S-unified": {"condition": "P1 holds (>= 95 %)",
                          "held": stop == "S-unified"},
            "S-switch": {"condition": "P1 fails (< 90 %)",
                         "held": stop == "S-switch"},
            "S-mid": {"condition": "P1 in 90-95 %",
                      "held": stop == "S-mid"},
            "S-fork": {"condition": "the D3 fidelity gate fails",
                       "held": False},
        },
        "invalid_rows": sum(block["invalid_rows"] for block in blocks.values()),
        "notes": [
            "No training and no collection: this file only evaluates.",
            "The candidate head and the 60/40/20 rule are both frozen; only the "
            "decision budget moves.",
            "U-900 is a **reproducibility** check against v8.19.0 S1, not a new "
            "arm: same head, same rule, same generator seeds, same 12 cells.",
            "The 12 (seed, window) cells are not independent samples; the "
            "pooled rate carries its exact (Clopper-Pearson) interval.",
            "U-600 is compared with v8.12.0 H2s (80/60/40, same head, same 12 "
            "cells) and U-1200 with the published 80 px 40 s cell (v8.1.0 A3, "
            "36/3,600) -- different rules, so the budget is the only mover.",
        ],
    }
    write_json(out_path, payload)
    log(f"summary: budgets={list(blocks)} -> {out_path}")
    for budget in budgets:
        block = blocks[arm_of(budget)]
        pooled = block["pooled"]
        log(f"  {arm_of(budget)} ({RULE_LABEL}): pooled {pooled['full']}/"
            f"{pooled['n']} = {pooled['rate']} CP95="
            f"[{pooled['cp95']['low']}, {pooled['cp95']['high']}] "
            f"bombs_mean={block['bomb_usage']['mean']} "
            f"complete={block['complete']}")
        for bucket in block["per_window_table"]:
            log(f"    w{bucket['window']}: {bucket['full']}/{bucket['n']} = "
                f"{bucket['rate']} (bombs_mean={bucket['bombs_used_mean']})")
    log(f"  verdict: {stop} P1={p1_payload['passed']} "
        f"P2={p2_payload['passed']} P3={p3_payload['observed']}")
    return 0 if payload["invalid_rows"] == 0 else 4


# --------------------------------------------------------------------------- #
# mode: audio-audit / process-audit
# --------------------------------------------------------------------------- #
def mode_audio(args: argparse.Namespace) -> int:
    out_path = pathlib.Path(args.out) if args.out else WORK / "audio_audit.json"
    root = WORK / "data"
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
        "audible_paths": [row["path"] for row in audible],
        "rows": rows,
    }
    payload["all_pass"] = bool(len(ok) > 0 and not audible)
    write_json(out_path, payload)
    log(f"audio-audit: {payload['cfg_files']} cfg, {payload['muted']} muted, "
        f"{payload['audible']} audible -> {out_path}")
    return 0 if payload["all_pass"] else 6


def mode_process(args: argparse.Namespace) -> int:
    out_path = (pathlib.Path(args.out) if args.out
                else WORK / "process_residue.json")
    needle = ".tmp_v8l0_unified"
    census = engine_census()
    payload: dict[str, Any] = {
        "schema": SCHEMA_PROC, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("this round's engine processes must be 0 when it ends; "
                        "other rounds' processes are counted but never touched"),
        "needle": needle, "pgrep_returncode": census["pgrep_returncode"],
        "engine_processes_total": census["engine_processes_total"],
        "other_round_processes": census["other_round_processes"],
        "matching": census["matching"], "lines": census["lines_mine"]}
    payload["all_pass"] = bool(payload["matching"] == 0)
    write_json(out_path, payload)
    log(f"process-audit: {payload['matching']} own engine process(es) alive "
        f"({payload['other_round_processes']} other-round engine processes "
        f"exist and were not touched) -> {out_path}")
    return 0 if payload["all_pass"] else 6


# --------------------------------------------------------------------------- #
# mode: publish
# --------------------------------------------------------------------------- #
def mode_publish(args: argparse.Namespace) -> int:
    import shutil

    root = pathlib.Path(args.out) if args.out else PUBLISH
    root.mkdir(parents=True, exist_ok=True)
    budgets = tuple(int(part) for part in str(args.budgets).split(",")
                    if part.strip())
    copied: list[dict[str, Any]] = []
    sources: list[pathlib.Path] = []
    for budget in budgets:
        for seed in SEEDS:
            for window in WINDOWS:
                sources.append(eval_file(budget, window, seed))
    sources.extend([WORK / "summary.json", WORK / "d3_gate.json",
                    WORK / "ckpt_assertions_start.json",
                    WORK / "ckpt_assertions_final.json",
                    WORK / "audio_audit.json", WORK / "process_residue.json",
                    WORK / "smoke.json", WORK / "preflight_start.json",
                    WORK / "preflight_final.json", WORK / "preflight.json",
                    WORK / "commands.tsv", WORK / "eval_exits.tsv",
                    WORK / "RUN.md"])
    for source in sources:
        if not source.is_file():
            continue
        target = root / source.name
        shutil.copy2(source, target)
        copied.append({"name": source.name, "source": str(source),
                       "sha256": sha256_file(target),
                       "bytes": target.stat().st_size})
    payload: dict[str, Any] = {
        "schema": SCHEMA_PUBLISH, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "publish_dir": str(root), "copied": copied, "count": len(copied),
    }
    write_json(root / "publish.json", payload)
    files: dict[str, Any] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root)).replace("\\", "/")
        if rel in ("manifest.json",):
            continue
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {"schema": SCHEMA_MANIFEST, "producer": PRODUCER,
                "created_at": payload["created_at"], "root": str(root),
                "prereg": PREREG_REL,
                "file_count": len(files),
                "total_bytes": sum(value["bytes"] for value in files.values()),
                "files": files}
    write_json(root / "manifest.json", manifest)
    log(f"publish: {len(copied)} artifacts + manifest ({len(files)} files) "
        f"-> {root}")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("v8.20.0 UNIFIED — the fixed 60/40/20 bomb rule re-measured "
                     "at 600 / 900 / 1,200 decisions, 3 seeds x 4 windows x 20 "
                     "episodes per budget"))
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "assert-ckpt", "d3", "smoke",
                                 "eval", "summary", "audio-audit",
                                 "process-audit", "publish"))
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    parser.add_argument("--decisions", type=int, default=600)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--windows", default=",".join(str(w) for w in WINDOWS))
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--decision-frames", type=int, default=DECISION_FRAMES)
    parser.add_argument("--phase", default="start",
                        choices=("start", "final"))
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--waited-s", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global _LOG_FH
    args = build_args(sys.argv[1:] if argv is None else argv)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.log:
        pathlib.Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        _LOG_FH = open(args.log, "a", encoding="utf-8")
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    # one process one core: the batch-1 inference does not benefit from threads
    # and the box is shared with other rounds' evaluations
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    log(f"mode={args.mode} python={sys.executable} pid={os.getpid()} "
        f"tag={args.tag!r}")
    try:
        if args.mode == "preflight":
            return mode_preflight(args)
        if args.mode == "assert-ckpt":
            return mode_assert(args)
        if args.mode == "d3":
            return mode_d3(args)
        if args.mode == "smoke":
            return mode_smoke(args)
        if args.mode == "eval":
            return mode_eval(args)
        if args.mode == "summary":
            return mode_summary(args)
        if args.mode == "audio-audit":
            return mode_audio(args)
        if args.mode == "process-audit":
            return mode_process(args)
        if args.mode == "publish":
            return mode_publish(args)
        raise UnifiedError(f"unknown mode {args.mode}")
    finally:
        if _LOG_FH is not None:
            _LOG_FH.flush()


if __name__ == "__main__":
    sys.exit(main())
