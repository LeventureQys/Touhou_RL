#!/usr/bin/env python3
"""v8.25.0 T33TIERS -- apply the **35/15/3** triple to 20 s and 40 s as well

What this round asks
--------------------
``Document/Update/Alpha/v8.25.0/PREREGISTRATION_T33TIERS_001.md`` freezes a
**single-factor** design: the candidate head, the sampler and the hand-written
threshold triple are all held fixed at v8.23.0's winner **35 / 15 / 3 px**; the
**only** thing that moves is the decision budget::

    U3-600    35 / 15 / 3 px     600 decisions (20 s)    ref 60/40/20 @600 = 98.75 %
    U3-900    35 / 15 / 3 px     900 decisions (30 s)    ref v8.23 T3 = 117/240
    U3-1200   35 / 15 / 3 px   1,200 decisions (40 s)    ref 60/40/20 @1200 = 2.08 %

Each tier is 3 seeds x 4 windows x 20 episodes = 240 episodes.  Nine processes
in total (one tier x one seed each, all four windows inside the process so the
generator stream is kept), 720 episodes.

The three deltas from the v8.23.0 evaluator (``v8o0_ruletight.py``)
-------------------------------------------------------------------
1. **one** threshold triple (35/15/3) and **three** ``--decisions`` tiers
   (600 / 900 / 1200) instead of three arms at a fixed 900;
2. ``--tag`` -> ``v8q0-U3-<budget>-s<seed>`` and the work dir is
   ``.tmp_v8q0_t33tiers`` (own private data dirs, never v8.23.0's);
3. artifacts publish to ``Document/Update/Alpha/v8.25.0/evidence/T33TIERS/``.
   Every outcome table carries an explicit ``distinct`` column (the number of
   distinct candidate indices the sampler touched per episode).

Everything else -- the frozen ``bombdag.pt``, the softmax sampler with one draw
per decision from ``torch.Generator(seed*1000+7)``, the 4 windows, the 20
episodes, the 2 frames per decision, PARK frame 8663, the frozen
``run_episode36`` loop, the D3 fidelity gate, the audio gate and the tag rule --
is imported unchanged.

Endpoints (prereg §2)
--------------------
``P1``  U3-600 pooled >= **95 %** (not significantly *worse* than 98.75 %:
        one-sided Fisher "arm < ref" p > 0.05);
``P2``  U3-900's exact CP95 overlaps v8.23.0's T3 = 117/240 = 48.75 %;
``P3``  U3-1200 next to 60/40/20's 2.0833 % (>= 3 % counts as a real gain);
``P4``  the bomb-usage mean of all three tiers (the 30 s reference is 2.7417).

Stop conditions (prereg §2.1)
----------------------------
``S-unified``  P1 holds  -> 35/15/3 is **one configuration** for 20/30/40 s;
``S-switch``   P1 fails  -> the threshold must switch with the budget
               (20 s: 60/40/20 = 98.75 %; 30 s: 35/15/3 = 48.75 %);
``S-fork``     the D3 fidelity gate fails -> the whole round is void.

Modes
-----
``--mode preflight``      engine count before / after the launch (never
                          ``pkill -f`` another round's processes).
``--mode assert-ckpt``    engine sha256, the three v8.6.0 ``bombdag.pt``
                          (``9c6d5d4a`` / ``36a51090`` / ``885dde70``), the
                          three frozen v4.5.0 ``round2.pt`` (``4abee069`` /
                          ``a93746bb`` / ``c69960fc``), protected sources.
``--mode d3``             D3 fidelity gate (5 assertions) through the frozen
                          ``v432_fork_arm.py --mode fidelity``.
``--mode smoke``          **one** episode (U3-600 / seed0 / w11), then stop.
``--mode eval``           one ``--arm`` x one ``--policy-seed``: 4 windows x 20
                          episodes, one ``eval_{arm}_d{budget}_w{w}_seed{s}.jsonl``
                          per window.
``--mode summary``        per-tier x per-window table + pooled rate + exact
                          Clopper-Pearson 95 % + per-seed + bomb usage +
                          distinct + the P1/P2/P3/P4 verdict and the stop
                          condition.
``--mode audio-audit``    AGENTS.md §9.3 on this round's own data dirs.
``--mode process-audit``  this round's own engine processes must be 0.
``--mode publish``        copy the raw artifacts into
                          ``Document/Update/Alpha/v8.25.0/evidence/T33TIERS/``
                          and write ``manifest.json``.

The ``--tag`` rule (v8.17.0's failure mode)
-------------------------------------------
Every process **must** pass its own unique ``--tag``: the private data dir is
``<work>/data/<tag>/w<window>`` and the engine log dir is ``<work>/runs/<tag>``.
This file derives the canonical tag from the (budget, seed) pair and rejects
anything else, so two processes can never share a data dir.

Not in this file
----------------
* No training, no collection, no labels, no pixel arm, no other rule.
* No claim that the policy learned to bomb: the bomb is an external rule here.
* No re-test of 60/40/20 (its 600/900/1200 rows are re-derived on disk only).
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

WORK = REPO / ".tmp_v8q0_t33tiers"
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.25.0" / "evidence"
           / "T33TIERS")

SCHEMA_ASSERT = "v8q0-t33tiers-ckpt-assertions/1"
SCHEMA_D3 = "v8q0-t33tiers-d3-gate/1"
SCHEMA_SMOKE = "v8q0-t33tiers-smoke/1"
SCHEMA_EPISODE = "v8q0-t33tiers-episode/1"
SCHEMA_SEED = "v8q0-t33tiers-tier-seed-progress/1"
SCHEMA_SUMMARY = "v8q0-t33tiers-summary/1"
SCHEMA_AUDIO = "v8q0-t33tiers-audio-audit/1"
SCHEMA_PROC = "v8q0-t33tiers-process-audit/1"
SCHEMA_MANIFEST = "v8q0-t33tiers-manifest/1"
SCHEMA_PUBLISH = "v8q0-t33tiers-publish/1"
SCHEMA_PREFLIGHT = "v8q0-t33tiers-preflight/1"
PRODUCER = ("tools/rl/v8q0_t33tiers.py (live engine, HarnessClient role=shadow, "
            "episode loop = tools/rl/v830_bombpol.py::run_episode36)")
PREREG_REL = "Document/Update/Alpha/v8.25.0/PREREGISTRATION_T33TIERS_001.md"

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- the frozen protocol (prereg §1) -----------------------------------------
SEEDS: tuple[int, ...] = (0, 1, 2)
WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
EPISODES = 20
DECISION_FRAMES = 2
PREFIX_FRAMES = 8663
N_CAND = 18
N_ACTIONS = 36
TEMPERATURE = 1.0
GEN_CONST = 7

# -- the one rule this round uses (prereg §1) --------------------------------
RULE_THRESHOLDS: dict[int, float] = {3: 35.0, 2: 15.0, 1: 3.0}
RULE_LABEL = "35/15/3"
RULE_NAME = "U3"

# -- the single factor this round moves: the decision budget (prereg §1) -----
TIERS: tuple[int, ...] = (600, 900, 1200)
ARM_ORDER: tuple[str, ...] = tuple(f"{RULE_NAME}-{budget}" for budget in TIERS)
TIER_OF: dict[str, int] = {f"{RULE_NAME}-{budget}": int(budget)
                           for budget in TIERS}

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
    "tools/rl/v8l0_unified.py",
    "tools/rl/v8o0_ruletight.py",
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)

# -- the three published references (numbers only, never re-run here) --------
#: R-600 = v8.20.0's U600 (**60 / 40 / 20 px @ 600 decisions**): pooled
#: **237/240 = 98.75 %** (per window 60/59/58/60, per seed 80/79/78,
#: bombs mean 2.9083).
REF_600: dict[str, Any] = {
    "tier": "R-600",
    "arm": "R-600 = v8.20.0 U600 (60/40/20 px @ 600 decisions)",
    "budget_decisions": 600, "pooled": (237, 240),
    "per_window": {11: (60, 60), 12: (59, 60), 17: (58, 60), 18: (60, 60)},
    "per_seed": {0: (80, 80), 1: (79, 80), 2: (78, 80)},
    "bombs_mean": 2.9083,
    "template": (".tmp_v8l0_unified/eval/eval_U600_d600_w{window}"
                 "_seed{seed}.jsonl"),
    "rerun_here": False,
}
#: R-900 = v8.23.0's **T3 (35/15/3 @ 900 decisions)**, i.e. this round's own
#: 30 s point measured one round earlier: pooled **117/240 = 48.75 %**
#: (per window 32/32/18/35, per seed 38/42/37, bombs mean 2.7417).
REF_900: dict[str, Any] = {
    "tier": "R-900",
    "arm": "R-900 = v8.23.0 T3 (35/15/3 px @ 900 decisions)",
    "budget_decisions": 900, "pooled": (117, 240),
    "per_window": {11: (32, 60), 12: (32, 60), 17: (18, 60), 18: (35, 60)},
    "per_seed": {0: (38, 80), 1: (42, 80), 2: (37, 80)},
    "bombs_mean": 2.7417,
    "template": (".tmp_v8o0_ruletight/eval/eval_T3_d900_w{window}"
                 "_seed{seed}.jsonl"),
    "rerun_here": False,
}
#: R-1200 = v8.20.0's U1200 (**60 / 40 / 20 px @ 1,200 decisions**): pooled
#: **5/240 = 2.0833 %** (per window 4/0/0/1, per seed 1/2/2, bombs mean 3.0).
REF_1200: dict[str, Any] = {
    "tier": "R-1200",
    "arm": "R-1200 = v8.20.0 U1200 (60/40/20 px @ 1,200 decisions)",
    "budget_decisions": 1200, "pooled": (5, 240),
    "per_window": {11: (4, 60), 12: (0, 60), 17: (0, 60), 18: (1, 60)},
    "per_seed": {0: (1, 80), 1: (2, 80), 2: (2, 80)},
    "bombs_mean": 3.0,
    "template": (".tmp_v8l0_unified/eval/eval_U1200_d1200_w{window}"
                 "_seed{seed}.jsonl"),
    "rerun_here": False,
}
REFERENCE_OF: dict[int, dict[str, Any]] = {600: REF_600, 900: REF_900,
                                           1200: REF_1200}

P1_RATE = 0.95              # prereg §2 P1: U3-600 >= 95 %
P1_ALPHA = 0.05             # prereg §2 P1: not significantly worse (p > 0.05)
P3_SUBSTANTIVE = 0.03       # prereg §2 P3: >= 3 % counts as a real gain
P4_REFERENCE_MEAN = 2.7417  # v8.23.0's 30 s bomb-usage mean
CP95_CONFIDENCE = 0.95

_LOG_FH: Any = None
_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class T33Error(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v8q0] {message}"
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
        raise T33Error(f"cannot import {path}")
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


# --------------------------------------------------------------------------- #
# the arm spec: one rule, one budget
# --------------------------------------------------------------------------- #
def budget_of(arm: str) -> int:
    key = str(arm)
    if key not in TIER_OF:
        raise T33Error(f"--arm must be one of {list(ARM_ORDER)}, got {key!r}")
    return int(TIER_OF[key])


def arm_of(budget: int) -> str:
    value = int(budget)
    if value not in TIERS:
        raise T33Error(f"--decisions must be one of {list(TIERS)}, got {value}")
    return f"{RULE_NAME}-{value}"


def check_arm(arm: str) -> str:
    budget_of(arm)
    return str(arm)


def check_budget(budget: int, arm: str | None = None) -> int:
    """Validate the tier and, when an arm is given, that they agree."""

    arm_of(budget)              # raises unless the tier is one of TIERS
    value = int(budget)
    if arm is not None and budget_of(arm) != value:
        raise T33Error(f"--arm {arm!r} carries {budget_of(arm)} decisions but "
                       f"--decisions is {value}; the tier IS the budget")
    return value


def thresholds() -> dict[int, float]:
    return dict(RULE_THRESHOLDS)


def rule_block() -> dict[str, Any]:
    return {"name": RULE_NAME, "label": RULE_LABEL,
            "thresholds": {str(key): float(value)
                           for key, value in RULE_THRESHOLDS.items()}}


def canonical_tag(budget: int, seed: int) -> str:
    """v8.17.0's failure mode: every process needs its own data dir."""

    return f"v8q0-{RULE_NAME}-{int(budget)}-s{int(seed)}"


def check_tag(tag: str, budget: int, seed: int, *, smoke: bool = False) -> str:
    expected = canonical_tag(budget, seed)
    allowed = (f"{expected}-smoke",) if smoke else (expected,)
    value = str(tag or "").strip()
    if not value:
        return expected + ("-smoke" if smoke else "")
    if value not in allowed:
        raise T33Error(f"--tag must be {expected!r}"
                       f"{' or ' + repr(allowed[0]) if smoke else ''}, got "
                       f"{value!r}; the tag is derived from (budget, seed) so "
                       f"two processes can never share a data dir")
    return value


def eval_file(arm: str, window: int, seed: int) -> pathlib.Path:
    return WORK / "eval" / (f"eval_{check_arm(arm)}_d{budget_of(arm)}"
                            f"_w{int(window)}_seed{int(seed)}.jsonl")


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
    lower = max(0, n1 - (total - m))
    upper = min(n1, m)
    logs = [_log_comb(m, x) + _log_comb(total - m, n1 - x)
            for x in range(lower, upper + 1)]
    peak = max(logs)
    denominator = math.exp(_log_comb(total, n1) - peak)
    if direction == "greater":
        start = max(int(a), lower)
        if start > upper:
            out["p_value"] = 0.0 if int(a) > upper else 1.0
        else:
            numerator = sum(math.exp(logs[x - lower] - peak)
                            for x in range(start, upper + 1))
            out["p_value"] = float(min(1.0, numerator / denominator))
    else:
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
# the sampled candidate + the 35/15/3 rule
# --------------------------------------------------------------------------- #
def process_generator(seed: int) -> Any:
    """The **one** generator of a process, created before the window loop."""

    import torch

    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    return generator


def candidate_sampled(logits: Any, generator: Any) -> tuple[int, float, float,
                                                            float]:
    """H2s: sample ``cand ~ softmax_c max(logit[2c], logit[2c+1])`` (T = 1.0).

    Never argmax: §32.5 / v8.21.0 measured 7-16 pp lost by argmax.
    """

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


def new_audit() -> dict[str, Any]:
    return {"cand_counts": [0] * N_CAND, "cand_prob": [],
            "threshold_counts": {str(key): 0 for key in (3, 2, 1, 0)},
            "rule": {"distance_true_bombs_positive": 0,
                     "distance_true_bombs_zero": 0,
                     "distance_false": 0},
            "emit_threshold_px": [], "emit_nearest_px": [],
            "thresholds": {str(key): float(value)
                           for key, value in RULE_THRESHOLDS.items()},
            "margins": [], "best_scores": []}


def rule_decision(state: dict[str, Any], hazards: Any,
                  table: dict[int, float]) -> dict[str, Any]:
    """The rule: ``nearest_all_px < table[bombs]`` and ``bombs > 0``.

    The distance is v8.0.0's own ``nearest_bullet_distances`` (the measurement
    R5 uses); ``table`` is the 35/15/3 triple.
    """

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


def make_decide(*, policy: Any, generator: Any, audit: dict[str, Any]) -> Any:
    """``decide(state, hazards, decision) -> (cand, bomb)`` for this rule."""

    import torch

    A = arm_a()
    table = thresholds()

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        feats, _detail = A.build_features(state=state, hazards=hazards,
                                          decision_index=decision)
        with torch.no_grad():
            logits = policy(torch.from_numpy(feats).unsqueeze(0))[0]
        cand, best, margin, prob = candidate_sampled(logits, generator)
        rule = rule_decision(state, hazards, table)
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
        raise T33Error(f"seed {seed}: checkpoint not found: {path}")
    digest = sha256_file(path)
    expected = V860_SHA256[int(seed)]
    if digest != expected:
        raise T33Error(f"seed {seed}: {path} sha256 {digest[:16]} != "
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


def distinct_of(row: dict[str, Any]) -> int:
    """The number of distinct candidate indices this episode's sampler touched."""

    value = row.get("distinct_cands")
    if value is not None:
        return int(value)
    counts = row.get("cand_counts") or []
    return int(sum(1 for count in counts if count))


def distinct_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [distinct_of(row) for row in rows]
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None,
                "hist": {}, "quantiles": {"n": 0}}
    hist = {str(k): sum(1 for v in values if v == k)
            for k in range(0, N_CAND + 1)}
    hist = {key: value for key, value in hist.items() if value}
    return {"n": len(values),
            "mean": round(statistics.fmean(values), 4),
            "min": int(min(values)), "max": int(max(values)),
            "median": round(statistics.median(values), 4),
            "stdev": (round(statistics.stdev(values), 4)
                      if len(values) > 1 else None),
            "of_candidates": N_CAND, "hist": hist,
            "quantiles": quantiles([float(v) for v in values])}


def run_one_episode(*, parent: Any, episode: int, seed: int, budget: int,
                    policy: Any, generator: Any, tag: str) -> dict[str, Any]:
    """One episode through the frozen loop, with this rule's audit."""

    audit = new_audit()
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
    distinct_cands = int(sum(1 for count in audit["cand_counts"] if count))
    row = {
        "schema": SCHEMA_EPISODE, "producer": PRODUCER,
        "arm": arm_of(budget), "arm_label": RULE_LABEL, "budget": int(budget),
        "rule": rule_block(),
        "arm_thresholds": {str(key): float(value)
                           for key, value in RULE_THRESHOLDS.items()},
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
        # -- the rule's audit: the candidate was SAMPLED (not argmax) and the
        #    bomb came from the 35/15/3 triple (not from the policy's bomb head)
        "cand_counts": audit["cand_counts"],
        "distinct_cands": distinct_cands,
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
        # every emitted bomb really was inside this rule's threshold window
        "threshold_respected": bool(
            len(emit_ok) == int(row["bomb_emits"]) and all(emit_ok)),
        # the candidate was sampled from >= 2 distinct indices (not argmax)
        "candidate_sampled": bool(
            distinct_cands >= (2 if row["decisions_measured"] >= 2 else 1)),
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
    mine = [line for line in engines if ".tmp_v8q0_t33tiers" in line]
    other_rounds: dict[str, int] = {}
    for line in engines:
        marker = ".tmp_"
        if marker in line:
            stem = line.split(marker, 1)[1].split("/", 1)[0]
            other_rounds[f".tmp_{stem}"] = other_rounds.get(f".tmp_{stem}", 0) + 1
    return {"pgrep_returncode": proc.returncode,
            "engine_processes_total": len(engines),
            "other_round_processes": len(engines) - len(mine),
            "other_rounds": other_rounds,
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
                        "this round shares the box with .tmp_v8p0_bombrl900 and "
                        "must never pkill -f another round's processes"),
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
        f"{census['other_round_processes']} {census['other_rounds']}) "
        f"history={len(history)} waited={payload['waited_s']}s -> {out_path}")
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
        "requirement": ("the candidate head is the frozen v8.6.0 `bombdag.pt` "
                        "Policy(106,64,36) (9c6d5d4a / 36a51090 / 885dde70); its "
                        "behaviour policy is the frozen v4.5.0 `S` student "
                        "(round2.pt = 4abee069 / a93746bb / c69960fc); the "
                        "engine and the protected sources must not have moved"),
        "rule": rule_block(), "tiers": list(TIERS),
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
    tag = args.tag or "v8q0-fidelity"
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
# mode: smoke (one episode: U3-600 / seed0 / w11)
# --------------------------------------------------------------------------- #
def mode_smoke(args: argparse.Namespace) -> int:
    seed = int(args.policy_seed)
    arm = check_arm(args.arm)
    budget = check_budget(args.decisions, arm)
    windows = [int(part) for part in str(args.windows).split(",") if part.strip()]
    window = windows[0]
    tag = check_tag(args.tag, budget, seed, smoke=True)
    out_path = pathlib.Path(args.out) if args.out else WORK / "smoke.json"
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    payload: dict[str, Any] = {
        "schema": SCHEMA_SMOKE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": (f"one episode of {arm} ({RULE_LABEL} px, "
                        f"seed{seed}/w{window}, {budget} decisions) with the "
                        f"sampler/rule audit, before the full 240-episode runs"),
        "arm": arm, "arm_label": RULE_LABEL, "budget": budget,
        "rule": rule_block(),
        "arm_thresholds": {str(k): float(v)
                           for k, v in RULE_THRESHOLDS.items()},
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
        row = run_one_episode(parent=parent, episode=0, seed=seed, budget=budget,
                              policy=policy, generator=generator, tag=tag)
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
            "distinct_cands": row["distinct_cands"],
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
        log(f"SMOKE {arm} ({RULE_LABEL}) s{seed} w{window} ep0: "
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
# mode: eval (one tier x one seed: 4 windows x 20 episodes)
# --------------------------------------------------------------------------- #
def mode_eval(args: argparse.Namespace) -> int:
    seed = int(args.policy_seed)
    arm = check_arm(args.arm)
    budget = check_budget(args.decisions, arm)
    episodes = int(args.episodes)
    if seed not in SEEDS:
        raise T33Error(f"--policy-seed must be one of {SEEDS}, got {seed}")
    if episodes != EPISODES:
        raise T33Error(f"this round fixes --episodes to {EPISODES} per window, "
                       f"got {episodes}")
    windows = tuple(int(part) for part in str(args.windows).split(",")
                    if part.strip())
    for window in windows:
        if window not in WINDOWS:
            raise T33Error(f"window {window} is not one of {WINDOWS}")
    tag = check_tag(args.tag, budget, seed)
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    run_dir = WORK / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = WORK / "data" / tag
    progress_path = (WORK / "eval" /
                     f"{arm}_d{budget}_s{seed}_progress.json")
    progress: dict[str, Any] = {
        "schema": SCHEMA_SEED, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": arm, "arm_label": RULE_LABEL, "budget": budget,
        "rule": rule_block(),
        "arm_thresholds": {str(k): float(v)
                           for k, v in RULE_THRESHOLDS.items()},
        "thresholds_label": RULE_LABEL,
        "policy_seed": seed, "budget_decisions": budget,
        "episodes_per_window": episodes, "windows": list(windows), "tag": tag,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest},
        "generator": {"seed": generator_seed(seed),
                      "kind": ("one torch.Generator per (tier, seed) process, "
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
            out_path = eval_file(arm, window, seed)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data_dir, data_info = v450().prepare_data_dir(data_root, window,
                                                          args.data_source)
            if not data_info["cfg"]["muted_four_zero"]:
                raise T33Error(f"w{window}: the generated th08.cfg is not "
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
                            raise T33Error(
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
                "distinct_mean": round(statistics.fmean(
                    distinct_of(row) for row in rows), 4) if rows else None,
                "distinct_min": min((distinct_of(row) for row in rows),
                                    default=None),
                "distinct_max": max((distinct_of(row) for row in rows),
                                    default=None),
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
    log(f"eval {arm} ({RULE_LABEL}) s{seed}: status={progress['status']} "
        f"cells="
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


def _reference_scan(reference: dict[str, Any]) -> dict[str, Any]:
    """Re-derive a published reference tier from its 240 raw rows on disk."""

    template: str = str(reference["template"])
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
        return {"available": False,
                "reason": f"{len(rows)} rows on disk, {expected} expected",
                "fallback_pooled": [int(reference["pooled"][0]),
                                    int(reference["pooled"][1])],
                "files": frames}
    per_window = {}
    for window in WINDOWS:
        cell = [row for row in rows if int(row["window"]) == int(window)]
        per_window[str(window)] = [sum(1 for row in cell if row["survived"]),
                                   len(cell)]
    per_seed = {}
    for seed in SEEDS:
        cell = [row for row in rows if int(row["policy_seed"]) == int(seed)]
        per_seed[str(seed)] = [sum(1 for row in cell if row["survived"]),
                               len(cell)]
    full = sum(1 for row in rows if row.get("survived"))
    return {"available": True, "rows": len(rows), "full": full,
            "pooled": [full, len(rows)],
            "rate": round(full / len(rows), 6),
            "rate_pct": round(100.0 * full / len(rows), 4),
            "cp95": clopper_pearson(full, len(rows)),
            "bombs_mean": round(statistics.fmean(
                int(row["bombs_used"] or 0) for row in rows), 4),
            "distinct": distinct_block(rows),
            "per_window": per_window, "per_seed": per_seed, "files": frames}


def _arm_block(arm: str, eval_dir: pathlib.Path) -> dict[str, Any]:
    budget = budget_of(arm)
    expected = len(SEEDS) * len(WINDOWS) * EPISODES
    rows: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    for seed in SEEDS:
        for window in WINDOWS:
            path = eval_dir / (f"eval_{arm}_d{budget}_w{window}"
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
            "bombs_used_mean": round(statistics.fmean(
                int(row["bombs_used"] or 0) for row in window_rows), 4),
            "alive_frames_mean": round(statistics.fmean(
                float(row["alive_frames"] or 0) for row in window_rows), 4),
            "distinct": distinct_block(window_rows),
        })
    for seed in SEEDS:
        seed_rows = [row for row in rows if int(row["policy_seed"]) == int(seed)]
        if not seed_rows:
            continue
        survived = sum(1 for row in seed_rows if row["survived"])
        per_seed[str(seed)] = {
            "n": len(seed_rows), "full": survived,
            "rate": round(survived / len(seed_rows), 6),
            "rate_pct": round(100.0 * survived / len(seed_rows), 4),
            "cp95": clopper_pearson(survived, len(seed_rows)),
            "bombs_used_mean": round(statistics.fmean(
                int(row["bombs_used"] or 0) for row in seed_rows), 4),
            "distinct": distinct_block(seed_rows)}
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
                             "rate": round(survived / len(cell_rows), 6),
                             "distinct_mean": round(statistics.fmean(
                                 distinct_of(row) for row in cell_rows), 4)})
    survived_all = sum(1 for row in rows if row["survived"])
    seed_rates = [value["rate"] for value in per_seed.values()]
    spread = (round(100.0 * (max(seed_rates) - min(seed_rates)), 4)
              if len(seed_rates) > 1 else None)
    return {
        "arm": str(arm), "arm_label": RULE_LABEL, "budget": int(budget),
        "rule": rule_block(),
        "thresholds_label": RULE_LABEL,
        "files": files, "rows": len(rows), "expected_rows": expected,
        "complete": bool(len(rows) == expected),
        "per_window_table": per_window,
        "per_seed": per_seed,
        "per_seed_spread_pp": spread,
        "per_cell": per_cell,
        "pooled": {
            "n": len(rows), "full": survived_all,
            "rate": round(survived_all / len(rows), 6) if rows else None,
            "rate_pct": (round(100.0 * survived_all / len(rows), 4)
                         if rows else None),
            "cp95": clopper_pearson(survived_all, len(rows)),
        },
        "distinct": distinct_block(rows),
        "bomb_usage": _bomb_block(rows),
        "invalid_rows": sum(1 for row in rows if row.get("invalid")),
        "wall_s_total": round(sum(float(row["wall_s"] or 0) for row in rows), 3),
        "cells_complete": sum(1 for value in files.values()
                              if value["rows"] == EPISODES),
    }


def mode_summary(args: argparse.Namespace) -> int:
    eval_dir = pathlib.Path(args.eval_dir) if args.eval_dir else WORK / "eval"
    out_path = pathlib.Path(args.out) if args.out else WORK / "summary.json"
    arms = tuple(str(part) for part in str(args.arms).split(",") if part.strip())
    for arm in arms:
        check_arm(arm)
    blocks: dict[str, Any] = {arm: _arm_block(arm, eval_dir) for arm in arms}

    # -- the three published references (never re-run here) -----------------
    refs: dict[str, Any] = {}
    for budget in TIERS:
        reference = REFERENCE_OF[budget]
        scan = _reference_scan(reference)
        if scan.get("available"):
            full, n = int(scan["pooled"][0]), int(scan["pooled"][1])
        else:
            full, n = (int(reference["pooled"][0]),
                       int(reference["pooled"][1]))
        refs[str(budget)] = {
            "metadata": reference, "scan": scan,
            "tier": reference["tier"], "arm": reference["arm"],
            "pooled_used": [full, n], "rate_used": round(full / n, 6),
            "rate_pct_used": round(100.0 * full / n, 4),
            "cp95": clopper_pearson(full, n),
            "bombs_mean": reference["bombs_mean"],
            "per_seed_rate_pct": {
                seed: round(100.0 * counts[0] / counts[1], 4)
                for seed, counts in (scan.get("per_seed") or {}).items()},
            "per_window_table": [
                {"window": int(window), "full": counts[0], "n": counts[1],
                 "rate": round(counts[0] / counts[1], 6)}
                for window, counts in (scan.get("per_window") or {}).items()],
        }

    # -- P1: U3-600 must not be significantly WORSE than 98.75 % -----------
    verdicts: dict[str, Any] = {}
    for arm in arms:
        block = blocks[arm]
        budget = budget_of(arm)
        pooled = block["pooled"]
        full, n = int(pooled["full"]), int(pooled["n"])
        ref = refs[str(budget)]
        ref_full, ref_n = ref["pooled_used"]
        payload: dict[str, Any] = {
            "arm": arm, "arm_label": RULE_LABEL, "budget": budget,
            "reference": {"tier": ref["tier"], "arm": ref["arm"],
                          "pooled": [ref_full, ref_n],
                          "rate": ref["rate_used"],
                          "rate_pct": ref["rate_pct_used"],
                          "cp95": ref["cp95"]},
            "observed": pooled["rate"], "observed_pct": pooled["rate_pct"],
            "cp95": pooled["cp95"],
            "difference_pp": (round(100.0 * (full / n - ref["rate_used"]), 4)
                              if n else None),
            "fisher_greater": None, "fisher_less": None,
            "fisher_p_greater": None, "fisher_p_less": None,
            "cp95_overlaps_reference": intervals_overlap(pooled["cp95"],
                                                         ref["cp95"]),
            "distinct": block["distinct"],
        }
        if n:
            greater = fisher_greater(full, n - full, ref_full,
                                     ref_n - ref_full)
            less = fisher_less(full, n - full, ref_full, ref_n - ref_full)
            payload["fisher_greater"] = greater
            payload["fisher_less"] = less
            payload["fisher_p_greater"] = greater["p_value"]
            payload["fisher_p_less"] = less["p_value"]
        verdicts[arm] = payload

    # P1 -- the primary endpoint (prereg §2)
    p1_arm = arm_of(600)
    p1 = verdicts.get(p1_arm) or {}
    p1_rate_ok = bool(p1.get("observed") is not None
                      and p1["observed"] >= P1_RATE)
    p1_not_worse = bool(p1.get("fisher_p_less") is not None
                        and p1["fisher_p_less"] > P1_ALPHA)
    p1_passed = bool(p1_rate_ok and p1_not_worse
                     and blocks.get(p1_arm, {}).get("complete"))

    # P2 -- U3-900 must overlap v8.23.0's T3 = 117/240
    p2_arm = arm_of(900)
    p2 = verdicts.get(p2_arm) or {}
    p2_overlap = bool(p2.get("cp95_overlaps_reference"))
    p2_passed = bool(p2_overlap and blocks.get(p2_arm, {}).get("complete"))

    # P3 -- U3-1200 vs 60/40/20's 2.0833 % (>= 3 % is substantive)
    p3_arm = arm_of(1200)
    p3 = verdicts.get(p3_arm) or {}
    p3_rate = p3.get("observed")
    p3_passed = bool(p3_rate is not None and p3_rate >= P3_SUBSTANTIVE
                     and blocks.get(p3_arm, {}).get("complete"))
    p3_gain_pp = p3.get("difference_pp")

    # P4 -- bomb-usage distribution (descriptive)
    p4 = {arm: {"budget": budget_of(arm),
                "mean": blocks[arm]["bomb_usage"]["mean"],
                "median": blocks[arm]["bomb_usage"]["median"],
                "stdev": blocks[arm]["bomb_usage"]["stdev"],
                "counts": blocks[arm]["bomb_usage"]["counts"],
                "vs_30s_reference_pp": (
                    round(blocks[arm]["bomb_usage"]["mean"]
                          - P4_REFERENCE_MEAN, 4)
                    if blocks[arm]["bomb_usage"]["mean"] is not None else None)}
          for arm in arms}

    # -- stop conditions (prereg §2.1) -------------------------------------
    d3 = read_json(WORK / "d3_gate.json")
    d3_pass = bool(d3.get("all_pass"))
    all_complete = all(block["complete"] for block in blocks.values())
    if d3 and not d3_pass:
        primary = "S-fork"
        stop_text = ("the D3 fidelity gate is not 5/5; the round is void and no "
                     "threshold conclusion may be drawn")
    elif not all_complete:
        primary = "S-incomplete"
        stop_text = ("not every tier has 240 rows; the prereg's stop conditions "
                     "are not declared on a partial table")
    elif p1_passed:
        primary = "S-unified"
        stop_text = (
            f"P1 holds: {p1_arm} ({RULE_LABEL} @ 600 decisions) is "
            f"{p1['observed_pct']:.4f} % >= 95 % and one-sided Fisher "
            f"\"worse\" p = {p1['fisher_p_less']:.4f} > 0.05 against 60/40/20's "
            f"98.75 %.  35/15/3 is ONE configuration for 20/30/40 s: deploy it "
            f"at all three budgets")
    else:
        primary = "S-switch"
        why = []
        if not p1_rate_ok:
            why.append(f"{p1_arm} = {p1.get('observed_pct')} % < 95 %")
        if not p1_not_worse:
            why.append(f"one-sided Fisher \"worse\" p = "
                       f"{p1.get('fisher_p_less')} <= 0.05")
        stop_text = (
            f"P1 fails ({'; '.join(why)}): the threshold must switch with the "
            f"budget -- 20 s keeps 60/40/20 (98.75 %), 30 s uses 35/15/3 "
            f"(48.75 % at v8.23.0 and {p2.get('observed_pct')} % here), and "
            f"40 s is reported without a unified claim")

    # -- the side-by-side table (prereg §2) --------------------------------
    table = []
    for arm in arms:
        block = blocks[arm]
        budget = budget_of(arm)
        row = {"arm": arm, "tier": f"U3-{budget}", "rule": RULE_LABEL,
               "budget": budget,
               "pooled_full": block["pooled"]["full"],
               "pooled_n": block["pooled"]["n"],
               "pooled_rate": block["pooled"]["rate"],
               "pooled_rate_pct": block["pooled"]["rate_pct"],
               "cp95_low": block["pooled"]["cp95"]["low"],
               "cp95_high": block["pooled"]["cp95"]["high"],
               "distinct": block["distinct"]["mean"],
               "distinct_min": block["distinct"]["min"],
               "distinct_max": block["distinct"]["max"],
               "bombs_mean": block["bomb_usage"]["mean"],
               "bombs0": block["bomb_usage"]["counts"].get("0"),
               "bombs1": block["bomb_usage"]["counts"].get("1"),
               "bombs2": block["bomb_usage"]["counts"].get("2"),
               "bombs3": block["bomb_usage"]["counts"].get("3"),
               "seed_spread_pp": block["per_seed_spread_pp"],
               "reference": refs[str(budget)]["arm"],
               "reference_full": refs[str(budget)]["pooled_used"][0],
               "reference_n": refs[str(budget)]["pooled_used"][1],
               "reference_rate_pct": refs[str(budget)]["rate_pct_used"],
               "difference_pp": verdicts[arm]["difference_pp"],
               "fisher_p_greater": verdicts[arm]["fisher_p_greater"],
               "fisher_p_less": verdicts[arm]["fisher_p_less"],
               "cp95_overlaps_reference":
                   verdicts[arm]["cp95_overlaps_reference"],
               "complete": block["complete"]}
        for bucket in block["per_window_table"]:
            row[f"w{bucket['window']}"] = f"{bucket['full']}/{bucket['n']}"
            row[f"w{bucket['window']}_distinct"] = bucket["distinct"]["mean"]
        for seed in SEEDS:
            value = block["per_seed"].get(str(seed)) or {}
            row[f"s{seed}"] = (f"{value.get('full')}/{value.get('n')}"
                               if value else None)
            row[f"s{seed}_distinct"] = (value.get("distinct") or {}).get("mean")
        table.append(row)

    payload = {
        "schema": SCHEMA_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": {"path": PREREG_REL,
                   "sha256": (sha256_file(REPO / PREREG_REL)
                              if (REPO / PREREG_REL).is_file() else None)},
        "question": ("v8.23.0 established 35/15/3 as the highest, steadiest and "
                     "cheapest triple at 900 decisions (30 s), but only at "
                     "900.  Does the same triple still work at 600 (20 s) and "
                     "1,200 (40 s), i.e. is 35/15/3 ONE configuration?"),
        "protocol": {
            "candidate_head": ("v8.6.0 bombdag.pt Policy(106,64,36), frozen; "
                               "score[c] = max(logit[2c], logit[2c+1]); "
                               "cand ~ softmax(score / 1.0), one draw per "
                               "decision from torch.Generator(seed*1000+7)"),
            "bomb": ("nearest_all_px < THRESHOLDS[bombs_remaining] and "
                     "bombs_remaining > 0, THRESHOLDS = 35/15/3"),
            "rule": rule_block(),
            "tiers": {arm_of(budget): budget for budget in TIERS},
            "seeds": list(SEEDS), "windows": list(WINDOWS),
            "episodes_per_window": EPISODES,
            "decision_frames": DECISION_FRAMES,
            "prefix_frame": PREFIX_FRAMES,
            "frames_per_episode": {str(budget): budget * DECISION_FRAMES
                                   for budget in TIERS},
            "loop": ("for tier in (600, 900, 1200): for seed in (0,1,2): one "
                     "process; inside it for window in (11,12,17,18): for "
                     "episode in range(20)"),
            "episode_loop": "tools/rl/v830_bombpol.py::run_episode36 (imported)",
        },
        "reference": {refs[str(budget)]["tier"]: refs[str(budget)]
                      for budget in TIERS},
        "arms": blocks,
        "table": table,
        "verdict": {
            "P1": {"requirement": ("U3-600 pooled >= 95 % AND one-sided Fisher "
                                   "\"arm < 60/40/20 @600\" p > 0.05 (not "
                                   "significantly worse than 98.75 %)"),
                   "threshold_rate": P1_RATE, "alpha": P1_ALPHA,
                   "arm": p1_arm,
                   "observed_pct": p1.get("observed_pct"),
                   "cp95": p1.get("cp95"),
                   "reference_pct": refs["600"]["rate_pct_used"],
                   "fisher_p_less": p1.get("fisher_p_less"),
                   "fisher_p_greater": p1.get("fisher_p_greater"),
                   "rate_gate": p1_rate_ok, "not_worse": p1_not_worse,
                   "passed": p1_passed},
            "P2": {"requirement": ("U3-900's exact CP95 overlaps v8.23.0 T3 "
                                   "(117/240 = 48.75 %)"),
                   "arm": p2_arm,
                   "observed_pct": p2.get("observed_pct"),
                   "cp95": p2.get("cp95"),
                   "reference": {"tier": REF_900["tier"],
                                 "pooled": refs["900"]["pooled_used"],
                                 "rate_pct": refs["900"]["rate_pct_used"],
                                 "cp95": refs["900"]["cp95"]},
                   "overlaps": p2_overlap, "passed": p2_passed},
            "P3": {"requirement": (">= 3 % at 1,200 decisions counts as a real "
                                   "gain over 60/40/20's 2.0833 %"),
                   "arm": p3_arm,
                   "observed_pct": (round(100.0 * p3_rate, 4)
                                    if p3_rate is not None else None),
                   "reference_pct": refs["1200"]["rate_pct_used"],
                   "gain_pp": p3_gain_pp,
                   "cp95": p3.get("cp95"),
                   "fisher_p_greater": p3.get("fisher_p_greater"),
                   "substantive_threshold_pct": 100.0 * P3_SUBSTANTIVE,
                   "passed": p3_passed},
            "P4": {"requirement": ("bomb-usage distribution reported for all "
                                   "three tiers (descriptive; the 30 s "
                                   "reference mean is 2.7417)"),
                   "reference_mean_30s": P4_REFERENCE_MEAN,
                   "per_tier": p4, "reported": True},
            "stop_condition": primary,
            "stop_text": stop_text,
            "S-unified": {"condition": "P1 holds", "held": bool(
                d3_pass and all_complete and p1_passed)},
            "S-switch": {"condition": "P1 fails", "held": bool(
                d3_pass and all_complete and not p1_passed)},
            "S-fork": {"condition": "the D3 fidelity gate fails",
                       "held": bool(d3 and not d3_pass),
                       "d3_all_pass": d3_pass},
        },
        "distinct_note": ("`distinct` is the number of distinct candidate "
                          "indices (of 18) the softmax sampler touched inside "
                          "one episode; it is reported per tier, per window, "
                          "per seed and per cell so argmax-like collapse would "
                          "be visible."),
        "invalid_rows": sum(block["invalid_rows"] for block in blocks.values()),
        "notes": [
            "No training and no collection: this file only evaluates.",
            "The candidate head, the sampler and the 35/15/3 triple are frozen; "
            "only the decision budget moves.",
            "R-600 (60/40/20 @600 = 237/240) and R-1200 (60/40/20 @1200 = "
            "5/240) are NOT re-run: both are re-derived from their 240 raw rows "
            "on disk.  R-900 is v8.23.0's own T3 arm (117/240).",
            "P1's Fisher test is one-sided (arm < 60/40/20 @600); P2 and P3 are "
            "interval/comparison readouts; P4 is descriptive.",
            "The 12 (seed, window) cells are not independent samples; the "
            "pooled rate carries its exact (Clopper-Pearson) interval.",
            "No argmax reading is admissible anywhere (prereg §5, §32.5).",
        ],
    }
    write_json(out_path, payload)
    log(f"summary: arms={list(blocks)} -> {out_path}")
    for arm in arms:
        block = blocks[arm]
        pooled = block["pooled"]
        verdict = verdicts[arm]
        log(f"  {arm} ({RULE_LABEL}) d{block['budget']}: pooled "
            f"{pooled['full']}/{pooled['n']} = {pooled['rate_pct']} % CP95="
            f"[{pooled['cp95']['low']}, {pooled['cp95']['high']}] "
            f"distinct={block['distinct']['mean']} "
            f"bombs_mean={block['bomb_usage']['mean']} "
            f"spread={block['per_seed_spread_pp']}pp "
            f"vs_ref_p={verdict['fisher_p_less']} "
            f"overlap={verdict['cp95_overlaps_reference']} "
            f"complete={block['complete']}")
        for bucket in block["per_window_table"]:
            log(f"    w{bucket['window']}: {bucket['full']}/{bucket['n']} = "
                f"{bucket['rate']} (bombs_mean={bucket['bombs_used_mean']} "
                f"distinct={bucket['distinct']['mean']})")
    log(f"  P1={p1_passed} P2={p2_passed} P3={p3_passed} P4(reported) -> "
        f"verdict: {primary}")
    log(f"  {stop_text}")
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
    needle = ".tmp_v8q0_t33tiers"
    census = engine_census()
    payload: dict[str, Any] = {
        "schema": SCHEMA_PROC, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("this round's engine processes must be 0 when it ends; "
                        "other rounds' processes are counted but never touched"),
        "needle": needle, "pgrep_returncode": census["pgrep_returncode"],
        "engine_processes_total": census["engine_processes_total"],
        "other_round_processes": census["other_round_processes"],
        "other_rounds": census["other_rounds"],
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
    arms = tuple(str(part) for part in str(args.arms).split(",") if part.strip())
    copied: list[dict[str, Any]] = []
    sources: list[pathlib.Path] = []
    for arm in arms:
        for seed in SEEDS:
            for window in WINDOWS:
                sources.append(eval_file(arm, window, seed))
    for arm in arms:
        budget = budget_of(arm)
        for seed in SEEDS:
            sources.append(WORK / "eval" /
                           f"{arm}_d{budget}_s{seed}_progress.json")
    sources.extend([WORK / "summary.json", WORK / "d3_gate.json",
                    WORK / "ckpt_assertions_start.json",
                    WORK / "ckpt_assertions_final.json",
                    WORK / "audio_audit.json", WORK / "process_residue.json",
                    WORK / "smoke.json", WORK / "preflight_start.json",
                    WORK / "preflight_final.json", WORK / "preflight.json",
                    WORK / "commands.tsv", WORK / "eval_exits.tsv",
                    WORK / "RUN.md", WORK / "extra_numbers.json"])
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
        description=("v8.25.0 T33TIERS — apply the 35/15/3 bomb-threshold triple "
                     "at 600 / 900 / 1,200 decisions, 3 seeds x 4 windows x 20 "
                     "episodes per tier"))
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "assert-ckpt", "d3", "smoke",
                                 "eval", "summary", "audio-audit",
                                 "process-audit", "publish"))
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--arms", default=",".join(ARM_ORDER))
    parser.add_argument("--arm", default=ARM_ORDER[0])
    parser.add_argument("--decisions", type=int, default=TIERS[0])
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
    # and the box is shared with another round's evaluation
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    log(f"mode={args.mode} python={sys.executable} pid={os.getpid()} "
        f"tag={args.tag!r} arm={args.arm!r} decisions={args.decisions}")
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
        raise T33Error(f"unknown mode {args.mode}")
    finally:
        if _LOG_FH is not None:
            _LOG_FH.flush()


if __name__ == "__main__":
    sys.exit(main())
