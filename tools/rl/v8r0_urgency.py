#!/usr/bin/env python3
"""v8.26.0 URGENCY -- **urgency-adaptive** bomb thresholds for 20 / 30 / 40 s

What this round asks
--------------------
``Document/Update/Alpha/v8.26.0/PREREGISTRATION_URGENCY_001.md`` freezes a
**single-factor** design.  v8.25.0 proved that one *fixed* triple cannot cover
the three budgets (``S-switch``: 20 s wants 60/40/20 = 98.75 %, 30 s wants
35/15/3 = 48.75 %, 40 s wants 35/15/3 = 12.50 %).  The hypothesis here is that
the threshold should not depend on the *budget* but on **how many decisions one
remaining bomb still has to cover**::

    R       = budget_decisions - decisions_executed      # remaining decisions
    B       = max(bombs_left, 1)                         # remaining bombs
    urgency = R / B
    scale   = (urgency / 200.0) ** alpha                 # 200 = the 20 s opening
    thr[b]  = min(120.0, max(5.0, base[b] * scale))      # base = {3:35, 2:15, 1:3}

The bomb fires iff ``nearest_all_px < thr[bombs_left]`` -- the *same* rule as
every earlier round, with the constant triple replaced by the formula above.
The **only** factor that moves is ``alpha`` (the response strength).

Arms and tiers (prereg §1)
--------------------------
    G1   alpha = 0.50     G2   alpha = 1.00     G3   alpha = 1.50
    tiers: 600 (20 s) / 900 (30 s) / 1,200 (40 s) decisions

``G0`` (the fixed 35/15/3 triple) is **not re-run**: v8.25.0's 720 raw rows are
re-derived from ``.tmp_v8q0_t33tiers/eval``.  The 60/40/20 comparison arm is
re-derived from v8.20.0's raw rows in ``.tmp_v8l0_unified/eval``.

The four deltas from the v8.25.0 evaluator (``v8q0_t33tiers.py``)
----------------------------------------------------------------
1. the threshold is a **formula** (``urgency``/``scale``) instead of the fixed
   35/15/3 constant; every decision's urgency, scale and threshold is audited;
2. the three ``--decisions`` tiers (600 / 900 / 1200) are unchanged;
3. ``--tag`` -> ``v8r0-<arm>-d<budget>-s<seed>`` and the work dir is
   ``.tmp_v8r0_urgency`` (own private data dirs, never another round's);
4. artifacts publish to
   ``Document/Update/Alpha/v8.26.0/evidence/URGENCY/``.

Everything else -- the frozen ``bombdag.pt``, the softmax sampler with one draw
per decision from ``torch.Generator(seed*1000+7)``, the 4 windows, the 20
episodes, the 2 frames per decision, PARK frame 8663, the frozen
``run_episode36`` loop, the D3 fidelity gate, the audio gate and the tag rule --
is imported unchanged.

Endpoints (prereg §2)
--------------------
``P1``  **all three tiers at once** >= the best fixed threshold of that tier:
        20 s >= 98.75 % (60/40/20 @600), 30 s >= 48.75 % (35/15/3 @900),
        40 s >= 12.50 % (35/15/3 @1200);
``P2``  at least one tier **strictly better** than G0 (one-sided Fisher
        p < 0.05; prereg names 30 s > 48.75 % or 40 s > 12.50 %);
``P3``  the bomb usage of the three tiers, **survivors reported separately**
        (§35.12);
``P4``  ``distinct`` (distinct candidate indices touched) reported every cell.

Stop conditions (prereg §2.1)
----------------------------
``S-adaptive``  P1 and P2 both hold -> the urgency formula unifies 20/30/40 s;
``S-partial``   P2 holds, P1 does not -> per-tier best table is the deployment;
``S-null``      P2 does not hold -> the response is void, keep ``S-switch``;
``S-fork``      the D3 fidelity gate fails -> the round is void.

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
``--mode smoke``          **one** episode (G2 / 600 / seed0 / w11), then print
                          the formula values of the first 5 bomb emissions.
``--mode eval``           one ``--arm`` x one ``--decisions`` x one
                          ``--policy-seed``: 4 windows x 20 episodes.
``--mode summary``        three arms x three tiers: per-window / per-seed /
                          per-cell rates, exact CP95, ``distinct``, the bomb
                          usage split by outcome, the P1/P2/P3/P4 verdict and
                          the stop condition.
``--mode report``         ``RUN.md`` from ``summary.json``.
``--mode audio-audit``    AGENTS.md §9.3 on this round's own data dirs.
``--mode process-audit``  this round's own engine processes must be 0.
``--mode publish``        copy the raw artifacts into the evidence dir and
                          write ``manifest.json``.

The ``--tag`` rule (v8.17.0's failure mode)
-------------------------------------------
Every process **must** pass its own unique ``--tag``: the private data dir is
``<work>/data/<tag>/w<window>`` and the engine log dir is ``<work>/runs/<tag>``.
This file derives the canonical tag from the (arm, budget, seed) triple and
rejects anything else, so two processes can never share a data dir.

Not in this file
----------------
* No training, no collection, no labels, no pixel arm, no other rule.
* No claim that the policy learned to bomb: the bomb is an external rule here.
* No argmax reading anywhere (prereg §5, §32.5).
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

WORK = REPO / ".tmp_v8r0_urgency"
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.26.0" / "evidence"
           / "URGENCY")
G0_DIR = REPO / ".tmp_v8q0_t33tiers" / "eval"
FIX_DIR = REPO / ".tmp_v8l0_unified" / "eval"

SCHEMA_ASSERT = "v8r0-urgency-ckpt-assertions/1"
SCHEMA_D3 = "v8r0-urgency-d3-gate/1"
SCHEMA_SMOKE = "v8r0-urgency-smoke/1"
SCHEMA_EPISODE = "v8r0-urgency-episode/1"
SCHEMA_CELL = "v8r0-urgency-arm-tier-seed-progress/1"
SCHEMA_SUMMARY = "v8r0-urgency-summary/1"
SCHEMA_AUDIO = "v8r0-urgency-audio-audit/1"
SCHEMA_PROC = "v8r0-urgency-process-audit/1"
SCHEMA_MANIFEST = "v8r0-urgency-manifest/1"
SCHEMA_PUBLISH = "v8r0-urgency-publish/1"
SCHEMA_PREFLIGHT = "v8r0-urgency-preflight/1"
PRODUCER = ("tools/rl/v8r0_urgency.py (live engine, HarnessClient role=shadow, "
            "episode loop = tools/rl/v830_bombpol.py::run_episode36)")
PREREG_REL = "Document/Update/Alpha/v8.26.0/PREREGISTRATION_URGENCY_001.md"
MINE = ".tmp_v8r0_urgency"

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

# -- the formula (prereg §1, verbatim) ---------------------------------------
BASE_THRESHOLDS: dict[int, float] = {3: 35.0, 2: 15.0, 1: 3.0}
THR_FLOOR = 5.0            # prereg §1.1: 3 px is too tight, 5 px is proven
THR_CEIL = 120.0
URGENCY_BASE = 200.0       # 20 s opening: 600 decisions / 3 bombs
FORMULA = ("urgency = (budget_decisions - decisions_executed) / max(bombs_left, 1); "
           "scale = (urgency / 200.0) ** alpha; "
           "thr[b] = min(120.0, max(5.0, base[b] * scale)); "
           "base = {3: 35.0, 2: 15.0, 1: 3.0}; bomb iff nearest_all_px < thr[bombs_left]")

# -- the single factor this round moves: alpha (prereg §1) -------------------
ALPHA_OF: dict[str, float] = {"G1": 0.50, "G2": 1.00, "G3": 1.50}
ARM_ORDER: tuple[str, ...] = ("G1", "G2", "G3")
TIERS: tuple[int, ...] = (600, 900, 1200)
TIER_LABEL = {600: "20 s", 900: "30 s", 1200: "40 s"}

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
    "tools/rl/v8q0_t33tiers.py",
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)

# -- the two fixed-threshold arms this round compares against (never re-run) --
#: G0 = v8.25.0's own T33TIERS rows: the **fixed 35/15/3** triple at 600 / 900 /
#: 1,200 decisions (222/240 = 92.50 %, 117/240 = 48.75 %, 30/240 = 12.50 %).
G0_TEMPLATE = "eval_U3-{budget}_d{budget}_w{window}_seed{seed}.jsonl"
G0_PUBLISHED: dict[int, tuple[int, int]] = {600: (222, 240), 900: (117, 240),
                                            1200: (30, 240)}
G0_LABEL = "G0 = fixed 35/15/3 px (v8.25.0 T33TIERS)"
#: FIX = v8.20.0's U600/U900/U1200 rows: the **fixed 60/40/20** triple at the
#: same three budgets (237/240 = 98.75 %, 58/240 = 24.17 %, 5/240 = 2.08 %).
FIX_TEMPLATE = "eval_U{budget}_d{budget}_w{window}_seed{seed}.jsonl"
FIX_PUBLISHED: dict[int, tuple[int, int]] = {600: (237, 240), 900: (58, 240),
                                             1200: (5, 240)}
FIX_LABEL = "FIX = fixed 60/40/20 px (v8.20.0 U600/U900/U1200)"

#: P1's bar per tier = the **best** of the two fixed arms on that tier.
BAR_PUBLISHED: dict[int, tuple[int, int]] = {
    tier: max((G0_PUBLISHED[tier], FIX_PUBLISHED[tier]),
              key=lambda kn: kn[0] / kn[1])
    for tier in TIERS}
P2_ALPHA = 0.05             # prereg §2 P2: Fisher p < 0.05
CP95_CONFIDENCE = 0.95

_LOG_FH: Any = None
_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class UrgencyError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v8r0] {message}"
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
        raise UrgencyError(f"cannot import {path}")
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
# the arm spec: one alpha, one budget
# --------------------------------------------------------------------------- #
def alpha_of(arm: str) -> float:
    key = str(arm)
    if key not in ALPHA_OF:
        raise UrgencyError(f"--arm must be one of {list(ARM_ORDER)}, got {key!r}")
    return float(ALPHA_OF[key])


def check_arm(arm: str) -> str:
    alpha_of(arm)
    return str(arm)


def check_tier(budget: int) -> int:
    value = int(budget)
    if value not in TIERS:
        raise UrgencyError(f"--decisions must be one of {list(TIERS)}, got {value}")
    return value


def arm_label(arm: str) -> str:
    return f"URG-a{alpha_of(arm):.2f}"


def rule_block(alpha: float) -> dict[str, Any]:
    return {"name": "URG", "kind": "urgency-adaptive",
            "formula": FORMULA, "alpha": float(alpha),
            "base_thresholds": {str(key): float(value)
                                for key, value in BASE_THRESHOLDS.items()},
            "floor_px": THR_FLOOR, "ceil_px": THR_CEIL,
            "urgency_base": URGENCY_BASE}


def canonical_tag(arm: str, budget: int, seed: int) -> str:
    """v8.17.0's failure mode: every process needs its own data dir."""

    alpha_of(arm)
    check_tier(budget)
    return f"v8r0-{arm}-d{int(budget)}-s{int(seed)}"


def check_tag(tag: str, arm: str, budget: int, seed: int, *,
              smoke: bool = False) -> str:
    expected = canonical_tag(arm, budget, seed)
    allowed = (f"{expected}-smoke",) if smoke else (expected,)
    value = str(tag or "").strip()
    if not value:
        return expected + ("-smoke" if smoke else "")
    if value not in allowed:
        raise UrgencyError(
            f"--tag must be {allowed[0]!r}, got {value!r}; the tag is derived "
            f"from (arm, budget, seed) so two processes can never share a data "
            f"dir")
    return value


def eval_file(arm: str, budget: int, window: int, seed: int) -> pathlib.Path:
    return WORK / "eval" / (f"eval_{check_arm(arm)}_d{check_tier(budget)}"
                            f"_w{int(window)}_seed{int(seed)}.jsonl")


def progress_file(arm: str, budget: int, seed: int) -> pathlib.Path:
    return WORK / "eval" / (f"{check_arm(arm)}_d{check_tier(budget)}"
                            f"_s{int(seed)}_progress.json")


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
    """One-sided Fisher exact test for the 2x2 table ``[[a, b], [c, d]]``."""

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
# the formula and the sampled candidate
# --------------------------------------------------------------------------- #
def thresholds_for(*, budget_decisions: int, decisions_executed: int,
                   bombs_left: int, alpha: float
                   ) -> tuple[float, float, dict[int, float]]:
    """Prereg §1, verbatim: ``R / max(B,1)`` -> ``scale`` -> clipped triple.

    Returns ``(urgency, scale, thr)`` where ``thr`` is keyed by bombs remaining
    (1 / 2 / 3), exactly like the fixed 35/15/3 triple it replaces.
    """

    remaining = int(budget_decisions) - int(decisions_executed)
    if remaining < 0:
        raise UrgencyError(f"decisions_executed {decisions_executed} exceeds "
                           f"budget_decisions {budget_decisions}")
    bombs = max(int(bombs_left), 1)
    urgency = float(remaining) / float(bombs)
    scale = float((urgency / URGENCY_BASE) ** float(alpha))
    table = {size: float(min(THR_CEIL, max(THR_FLOOR, float(base) * scale)))
             for size, base in BASE_THRESHOLDS.items()}
    return urgency, scale, table


class Stat:
    """Tiny streaming mean / min / max (keeps the per-row JSON small)."""

    __slots__ = ("n", "total", "low", "high")

    def __init__(self) -> None:
        self.n = 0
        self.total = 0.0
        self.low: float | None = None
        self.high: float | None = None

    def add(self, value: float) -> None:
        value = float(value)
        self.n += 1
        self.total += value
        self.low = value if self.low is None else min(self.low, value)
        self.high = value if self.high is None else max(self.high, value)

    def block(self) -> dict[str, Any]:
        if not self.n:
            return {"n": 0}
        return {"n": self.n, "mean": round(self.total / self.n, 6),
                "min": round(float(self.low), 6),
                "max": round(float(self.high), 6)}


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
            "decisions_by_bombs": {str(key): 0 for key in (3, 2, 1, 0)},
            "rule": {"distance_true_bombs_positive": 0,
                     "distance_true_bombs_zero": 0,
                     "distance_false": 0},
            "urgency": Stat(), "scale": Stat(),
            "active_thr": {3: Stat(), 2: Stat(), 1: Stat()},
            "active_urgency": {3: Stat(), 2: Stat(), 1: Stat()},
            "active_scale": {3: Stat(), 2: Stat(), 1: Stat()},
            "emits": [], "thr_at_open": None, "bombs_at_open": None,
            "thr_bounds_ok": True, "formula_ok": True,
            "margins": [], "best_scores": []}


def rule_decision(state: dict[str, Any], hazards: Any,
                  table: dict[int, float]) -> dict[str, Any]:
    """The rule: ``nearest_all_px < table[bombs]`` and ``bombs > 0``.

    ``table`` is the urgency-scaled triple computed for this decision.
    """

    distances = m800().nearest_bullet_distances(state, hazards)
    nearest = distances["nearest_all_px"]
    bombs_before = int(state["bombs_remaining"] or 0)
    threshold = table.get(bombs_before) if bombs_before > 0 else None
    distance_ok = bool(nearest is not None and threshold is not None
                       and float(nearest) < float(threshold))
    fired = bool(distance_ok and bombs_before > 0)
    return {"bomb": 1 if fired else 0, "nearest_all_px": nearest,
            "threshold_px": threshold, "bombs_remaining": bombs_before,
            "distance_ok": distance_ok}


def make_decide(*, policy: Any, generator: Any, audit: dict[str, Any],
                budget: int, alpha: float) -> Any:
    """``decide(state, hazards, decision) -> (cand, bomb)`` for this formula.

    ``decision`` is the frozen loop's 0-based decision index, i.e. the number of
    decisions the engine has already executed when this one is taken -- so
    ``R = budget - decision`` is the **engine's** remaining-decision reading
    (prereg §1: "urgency 用真实剩余决策数").
    """

    import torch

    A = arm_a()

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        executed = int(decision)
        bombs_before = int(state["bombs_remaining"] or 0)
        urgency, scale, table = thresholds_for(
            budget_decisions=int(budget), decisions_executed=executed,
            bombs_left=bombs_before, alpha=float(alpha))
        feats, _detail = A.build_features(state=state, hazards=hazards,
                                          decision_index=decision)
        with torch.no_grad():
            logits = policy(torch.from_numpy(feats).unsqueeze(0))[0]
        cand, best, margin, prob = candidate_sampled(logits, generator)
        rule = rule_decision(state, hazards, table)
        for value in table.values():
            if not (THR_FLOOR - 1e-9 <= value <= THR_CEIL + 1e-9):
                audit["thr_bounds_ok"] = False
        audit["urgency"].add(urgency)
        audit["scale"].add(scale)
        if bombs_before in audit["active_thr"]:
            audit["active_thr"][bombs_before].add(float(table[bombs_before]))
            audit["active_urgency"][bombs_before].add(urgency)
            audit["active_scale"][bombs_before].add(scale)
        if executed == 0:
            audit["thr_at_open"] = {str(k): round(float(v), 6)
                                    for k, v in table.items()}
            audit["bombs_at_open"] = bombs_before
        audit["cand_counts"][int(cand)] += 1
        audit["cand_prob"].append(float(prob))
        audit["margins"].append(float(margin))
        audit["best_scores"].append(float(best))
        key = str(bombs_before)
        if key in audit["decisions_by_bombs"]:
            audit["decisions_by_bombs"][key] += 1
        if rule["distance_ok"]:
            if bombs_before > 0:
                audit["rule"]["distance_true_bombs_positive"] += 1
                # the formula must reproduce this very threshold
                expected_scale = float((urgency / URGENCY_BASE) ** float(alpha))
                expected = float(min(THR_CEIL, max(
                    THR_FLOOR,
                    float(BASE_THRESHOLDS[bombs_before]) * expected_scale)))
                if abs(expected - float(rule["threshold_px"])) > 1e-9:
                    audit["formula_ok"] = False
                audit["emits"].append({
                    "decision": int(executed),
                    "remaining_decisions": int(budget) - int(executed),
                    "bombs_before": int(bombs_before),
                    "urgency": round(urgency, 6), "scale": round(scale, 6),
                    "threshold_px": round(float(rule["threshold_px"]), 6),
                    "expected_threshold_px": round(expected, 6),
                    "nearest_all_px": (None if rule["nearest_all_px"] is None
                                       else round(float(rule["nearest_all_px"]),
                                                  6)),
                    "bomb": int(rule["bomb"])})
            else:
                audit["rule"]["distance_true_bombs_zero"] += 1
        else:
            audit["rule"]["distance_false"] += 1
        return int(cand), int(rule["bomb"])

    return decide


def load_policy(seed: int) -> tuple[Any, pathlib.Path, str]:
    path = V860_ROOT / f"seed{int(seed)}" / "bombdag.pt"
    if not path.is_file():
        raise UrgencyError(f"seed {seed}: checkpoint not found: {path}")
    digest = sha256_file(path)
    expected = V860_SHA256[int(seed)]
    if digest != expected:
        raise UrgencyError(f"seed {seed}: {path} sha256 {digest[:16]} != "
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
                    policy: Any, generator: Any, tag: str, alpha: float,
                    keep_trace: bool = False) -> dict[str, Any]:
    """One episode through the frozen loop, with this formula's audit."""

    audit = new_audit()
    decide = make_decide(policy=policy, generator=generator, audit=audit,
                         budget=budget, alpha=alpha)
    record = v830().run_episode36(parent, episode=int(episode),
                                  policy_seed=int(seed), decisions=int(budget),
                                  decide=decide, budget=int(budget),
                                  decision_frames=DECISION_FRAMES,
                                  keep_trace=bool(keep_trace))
    probs = audit["cand_prob"] or [0.0]
    emits = audit["emits"]
    emit_ok = [bool(item["nearest_all_px"] is not None
                    and float(item["nearest_all_px"])
                    < float(item["threshold_px"]))
               for item in emits]
    distinct_cands = int(sum(1 for count in audit["cand_counts"] if count))
    row = {
        "schema": SCHEMA_EPISODE, "producer": PRODUCER,
        "arm": str(getattr(parent, "arm", "") or ""),
        "arm_label": arm_label(str(getattr(parent, "arm", "G2") or "G2")),
        "alpha": float(alpha), "budget": int(budget),
        "rule": rule_block(alpha),
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
        # -- the formula's audit -------------------------------------------- #
        "cand_counts": audit["cand_counts"],
        "distinct_cands": distinct_cands,
        "cand_prob": {"mean": round(statistics.fmean(probs), 6),
                      "min": round(min(probs), 6),
                      "max": round(max(probs), 6)},
        "cand_margin": quantiles(audit["margins"]),
        "cand_best_score": quantiles(audit["best_scores"]),
        "decisions_by_bombs_remaining": audit["decisions_by_bombs"],
        "urgency": audit["urgency"].block(),
        "scale": audit["scale"].block(),
        "active_threshold_by_bombs": {
            str(key): audit["active_thr"][key].block() for key in (3, 2, 1)},
        "active_urgency_by_bombs": {
            str(key): audit["active_urgency"][key].block() for key in (3, 2, 1)},
        "active_scale_by_bombs": {
            str(key): audit["active_scale"][key].block() for key in (3, 2, 1)},
        "thr_at_open": audit["thr_at_open"],
        "bombs_at_open": audit["bombs_at_open"],
        "emits": emits,
        "emit_threshold_px": [item["threshold_px"] for item in emits],
        "emit_urgency": [item["urgency"] for item in emits],
        "emit_scale": [item["scale"] for item in emits],
        "emit_nearest_px": [item["nearest_all_px"] for item in emits],
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
        # every emitted bomb really was inside this decision's threshold window
        "threshold_respected": bool(
            len(emit_ok) == int(row["bomb_emits"]) and all(emit_ok)),
        # the candidate was sampled from >= 2 distinct indices (not argmax)
        "candidate_sampled": bool(
            distinct_cands >= (2 if row["decisions_measured"] >= 2 else 1)),
        # the formula stayed inside [5, 120] px at every decision
        "thresholds_in_bounds": bool(audit["thr_bounds_ok"]),
        # every emit's threshold equals base[b] * (urgency/200)**alpha, clipped
        "formula_reproduced": bool(audit["formula_ok"]),
    }
    if not all(row["checks"].values()):
        row["invalid"] = True
        row["error"] = row["error"] or f"audit check failed: {row['checks']}"
    return row


# --------------------------------------------------------------------------- #
# mode: preflight (engine census, never pkill)
# --------------------------------------------------------------------------- #
def engine_census() -> dict[str, Any]:
    proc = subprocess.run(["pgrep", "-af", "th08-modern"],
                          capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    engines = [line for line in lines
               if "th08-modern" in line and "pgrep" not in line
               and "rl-role" in line]
    mine = [line for line in engines if MINE in line]
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
                        "must never pkill -f another round's processes; at most "
                        "6 of this round's processes run at once"),
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
        "rule": {"base_thresholds": {str(k): v
                                     for k, v in BASE_THRESHOLDS.items()},
                 "floor_px": THR_FLOOR, "ceil_px": THR_CEIL,
                 "urgency_base": URGENCY_BASE,
                 "alphas": {arm: alpha_of(arm) for arm in ARM_ORDER}},
        "tiers": list(TIERS),
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
    tag = args.tag or "v8r0-fidelity"
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
# mode: smoke (G2 / 600 / seed0 / w11, first 5 bomb emissions printed)
# --------------------------------------------------------------------------- #
SMOKE_ARM = "G2"
SMOKE_BUDGET = 600
SMOKE_EMITS = 5
SMOKE_MAX_EPISODES = 3


def mode_smoke(args: argparse.Namespace) -> int:
    seed = int(args.policy_seed)
    arm = check_arm(args.arm)
    budget = check_tier(args.decisions)
    windows = [int(part) for part in str(args.windows).split(",") if part.strip()]
    window = windows[0]
    tag = check_tag(args.tag, arm, budget, seed, smoke=True)
    out_path = pathlib.Path(args.out) if args.out else WORK / "smoke.json"
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    alpha = alpha_of(arm)
    payload: dict[str, Any] = {
        "schema": SCHEMA_SMOKE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": (f"one smoke episode of {arm} (alpha={alpha:.2f}, "
                        f"seed{seed}/w{window}, {budget} decisions) that prints "
                        f"the urgency / scale / threshold of the first "
                        f"{SMOKE_EMITS} bomb emissions, before the full "
                        f"2,160-episode runs"),
        "arm": arm, "alpha": alpha, "arm_label": arm_label(arm),
        "budget": budget, "rule": rule_block(alpha),
        "policy_seed": seed, "window": window,
        "budget_decisions": budget, "episodes_requested": 1,
        "decision_frames": DECISION_FRAMES, "tag": tag,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest,
                 "bytes": ckpt.stat().st_size},
        "rows": [], "emissions": [], "first_5_emissions": [],
        "checks": {}, "all_pass": False, "error": None,
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
        # one smoke episode is the requirement; an episode can only spend 3
        # bombs, so up to 3 episodes are used to collect 5 emission samples.
        for episode in range(SMOKE_MAX_EPISODES):
            row = run_one_episode(parent=parent, episode=episode, seed=seed,
                                  budget=budget, policy=policy,
                                  generator=generator, tag=tag, alpha=alpha)
            row["arm"] = arm
            row["ckpt_sha256"] = digest
            row["engine_sha256"] = engine.get("sha256")
            payload["rows"].append(row)
            payload["emissions"].extend(
                dict(item, episode=int(episode)) for item in row["emits"])
            log(f"SMOKE {arm} (alpha={alpha:.2f}) s{seed} w{window} ep{episode}: "
                f"survived={row['survived']} bombs_used={row['bombs_used']} "
                f"alive_frames={row['alive_frames']} frames={row['frames']} "
                f"decisions={row['decisions_executed']} "
                f"decides={row['decisions_measured']} "
                f"bomb_emits={row['bomb_emits']} "
                f"distinct_cands={row['distinct_cands']} "
                f"thr_at_open={row['thr_at_open']} "
                f"urgency={row['urgency']} scale={row['scale']} "
                f"checks={row['checks']}")
            if len(payload["emissions"]) >= SMOKE_EMITS:
                break
        payload["first_5_emissions"] = payload["emissions"][:SMOKE_EMITS]
        if payload["first_5_emissions"]:
            log(f"SMOKE first {len(payload['first_5_emissions'])} bomb "
                f"emissions (urgency / scale / thr / nearest / R / B):")
            for index, item in enumerate(payload["first_5_emissions"]):
                log(f"  #{index + 1} ep{item['episode']} d{item['decision']}: "
                    f"urgency={item['urgency']} scale={item['scale']} "
                    f"thr={item['threshold_px']} nearest={item['nearest_all_px']} "
                    f"R={item['remaining_decisions']} B={item['bombs_before']}")
        else:
            log("SMOKE: no bomb was emitted in the smoke episodes")
        first = payload["rows"][0]
        payload["checks"] = {
            "survived": first["survived"],
            "bombs_used": first["bombs_used"],
            "alive_frames": first["alive_frames"],
            "frames": first["frames"],
            "decisions_executed": first["decisions_executed"],
            "bomb_emits": first["bomb_emits"],
            "bomb_decisions": first["bomb_decisions"],
            "distinct_cands": first["distinct_cands"],
            "cfg_muted_four_zero": payload["cfg"]["muted_four_zero"],
            "gate_all_pass": payload["gate"]["all_pass"],
            **first["checks"]}
        payload["all_pass"] = bool(
            len(payload["rows"]) >= 1
            and not any(row["invalid"] for row in payload["rows"])
            and first["decisions_measured"] >= 1
            and first["samples_consumed"] >= 1
            and payload["checks"]["cfg_muted_four_zero"]
            and payload["checks"]["gate_all_pass"]
            and all(first["checks"].values()))
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
    log(f"smoke: all_pass={payload['all_pass']} "
        f"episodes={len(payload['rows'])} "
        f"emissions={len(payload['emissions'])} wall={payload['wall_s']}s "
        f"-> {out_path}")
    return 0 if payload["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: eval (one arm x one tier x one seed: 4 windows x 20 episodes)
# --------------------------------------------------------------------------- #
def mode_eval(args: argparse.Namespace) -> int:
    seed = int(args.policy_seed)
    arm = check_arm(args.arm)
    budget = check_tier(args.decisions)
    alpha = alpha_of(arm)
    episodes = int(args.episodes)
    if seed not in SEEDS:
        raise UrgencyError(f"--policy-seed must be one of {SEEDS}, got {seed}")
    if episodes != EPISODES:
        raise UrgencyError(f"this round fixes --episodes to {EPISODES} per "
                           f"window, got {episodes}")
    windows = tuple(int(part) for part in str(args.windows).split(",")
                    if part.strip())
    for window in windows:
        if window not in WINDOWS:
            raise UrgencyError(f"window {window} is not one of {WINDOWS}")
    tag = check_tag(args.tag, arm, budget, seed)
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    run_dir = WORK / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = WORK / "data" / tag
    progress_path = progress_file(arm, budget, seed)
    progress: dict[str, Any] = {
        "schema": SCHEMA_CELL, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": arm, "arm_label": arm_label(arm), "alpha": alpha,
        "budget": budget, "rule": rule_block(alpha),
        "policy_seed": seed, "budget_decisions": budget,
        "episodes_per_window": episodes, "windows": list(windows), "tag": tag,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest},
        "generator": {"seed": generator_seed(seed),
                      "kind": ("one torch.Generator per (arm, tier, seed) "
                               "process, consumed once per decision across every "
                               "window and episode")},
        "cells": {}, "status": "running", "error": None, "wall_s": None,
    }
    started = time.perf_counter()
    from touhou_rl.v420 import contracts as C

    generator = process_generator(seed)
    try:
        for window in windows:
            C.assert_seed_allowed(int(window), "train")
            out_path = eval_file(arm, budget, window, seed)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data_dir, data_info = v450().prepare_data_dir(data_root, window,
                                                          args.data_source)
            if not data_info["cfg"]["muted_four_zero"]:
                raise UrgencyError(f"w{window}: the generated th08.cfg is not "
                                   f"muted: {data_info['cfg']}")
            log(f"{arm} (a{alpha:.2f}) d{budget} s{seed} w{window}: cfg muted="
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
                                              generator=generator, tag=tag,
                                              alpha=alpha)
                        row["arm"] = arm
                        row["ckpt_sha256"] = digest
                        row["engine_sha256"] = engine.get("sha256")
                        row["cfg_sha256"] = data_info["cfg"]["sha256"]
                        row["cfg_muted_four_zero"] = \
                            data_info["cfg"]["muted_four_zero"]
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        rows.append(row)
                        log(f"  {arm} d{budget} s{seed} w{window} ep{episode}: "
                            f"survived={row['survived']} "
                            f"bombs_used={row['bombs_used']} "
                            f"alive={row['alive_frames']} "
                            f"distinct_cands={row['distinct_cands']} "
                            f"emits={len(row['emits'])} "
                            f"wall={row['wall_s']}s")
                        if row["invalid"]:
                            raise UrgencyError(
                                f"{arm} d{budget} s{seed} w{window} ep{episode} "
                                f"invalid: {row['error']}")
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
            log(f"{arm} d{budget} s{seed} w{window}: {survived}/{len(rows)} "
                f"survived ({progress['cells'][str(window)]['rate']}) wall="
                f"{progress['cells'][str(window)]['wall_s']}s")
        progress["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        progress["error"] = f"{type(exc).__name__}: {exc}"
        progress["status"] = "failed"
        log(f"EVAL FAILED {arm} d{budget} s{seed}: {progress['error']}")
    finally:
        progress["wall_s"] = round(time.perf_counter() - started, 3)
        write_json(progress_path, progress)
    log(f"eval {arm} (a{alpha:.2f}) d{budget} s{seed}: "
        f"status={progress['status']} cells="
        f"{ {k: (v['survived'], v['episodes']) for k, v in progress['cells'].items()} } "
        f"wall={progress['wall_s']}s")
    return 0 if progress["status"] == "complete" else 4


# --------------------------------------------------------------------------- #
# summary helpers
# --------------------------------------------------------------------------- #
def _bomb_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Bomb usage, **split by outcome** (§35.12: survivors reported apart)."""

    def stats(cell: Sequence[dict[str, Any]]) -> dict[str, Any]:
        used = [int(row["bombs_used"] or 0) for row in cell]
        if not cell:
            return {"episodes": 0, "mean": None, "median": None, "stdev": None,
                    "counts": {}, "rate_by_count": {},
                    "bombs_used_total": 0, "episodes_exhausted": 0,
                    "episodes_never_bombed": 0}
        return {
            "episodes": len(cell),
            "counts": {str(k): sum(1 for v in used if v == k)
                       for k in (0, 1, 2, 3)},
            "rate_by_count": {str(k): round(sum(1 for v in used if v == k)
                                            / len(cell), 6) for k in (0, 1, 2, 3)},
            "mean": round(statistics.fmean(used), 4),
            "median": round(statistics.median(used), 4),
            "stdev": round(statistics.stdev(used), 4) if len(used) > 1 else None,
            "bombs_used_total": sum(used),
            "episodes_exhausted": sum(1 for row in cell
                                      if int(row["bombs_end"] or 0) == 0),
            "episodes_never_bombed": sum(1 for v in used if v == 0),
        }

    survived = [row for row in rows if row.get("survived")]
    died = [row for row in rows if not row.get("survived")]
    firsts = [int(row["first_bomb_decision"]) for row in rows
              if row.get("first_bomb_decision") is not None]
    lasts = [int(row["last_bomb_decision"]) for row in rows
             if row.get("last_bomb_decision") is not None]
    emit_thr = [float(v) for row in rows
                for v in (row.get("emit_threshold_px") or [])]
    emit_urg = [float(v) for row in rows
                for v in (row.get("emit_urgency") or [])]
    emit_scale = [float(v) for row in rows
                  for v in (row.get("emit_scale") or [])]
    emit_near = [float(v) for row in rows
                 for v in (row.get("emit_nearest_px") or []) if v is not None]
    return {
        "episodes": len(rows),
        "all": stats(rows),
        "survived": stats(survived),
        "died": stats(died),
        "bomb_emits_total": sum(int(row["bomb_emits"] or 0) for row in rows),
        "first_bomb_decision": (quantiles([float(v) for v in firsts])
                                if firsts else {"n": 0}),
        "last_bomb_decision": (quantiles([float(v) for v in lasts])
                               if lasts else {"n": 0}),
        "alive_frames": quantiles([float(row["alive_frames"] or 0)
                                   for row in rows]),
        "min_nearest_px": quantiles([float(row["min_nearest_px"]) for row in rows
                                     if row.get("min_nearest_px") is not None]),
        "emit_urgency": quantiles(emit_urg),
        "emit_scale": quantiles(emit_scale),
        "emit_threshold_px": quantiles(emit_thr),
        "emit_nearest_px": quantiles(emit_near),
    }


def _threshold_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The realised threshold schedule of a cell (proof the formula was live)."""

    def collect(field: str, key: str | None = None) -> dict[str, Any]:
        values = []
        for row in rows:
            node = row.get(field)
            if key is not None:
                node = (node or {}).get(key) or {}
            if node.get("n"):
                values.append(node)
        if not values:
            return {"n": 0}
        return {
            "n": sum(int(v["n"]) for v in values),
            "mean_of_means": round(statistics.fmean(float(v["mean"])
                                                    for v in values), 6),
            "min": round(min(float(v["min"]) for v in values), 6),
            "max": round(max(float(v["max"]) for v in values), 6)}

    opens = [row["thr_at_open"] for row in rows if row.get("thr_at_open")]
    return {
        "urgency": collect("urgency"),
        "scale": collect("scale"),
        "active_threshold_by_bombs": {
            key: collect("active_threshold_by_bombs", key)
            for key in ("3", "2", "1")},
        "thr_at_open": ({
            "distinct": sorted({json.dumps(item, sort_keys=True)
                                for item in opens}),
            "count": len(opens)} if opens else {"n": 0}),
    }


def _scan_reference(template: str, root: pathlib.Path,
                    published: dict[int, tuple[int, int]]
                    ) -> dict[int, dict[str, Any]]:
    """Re-derive a fixed-threshold arm from its 240 raw rows per tier."""

    out: dict[int, dict[str, Any]] = {}
    for tier in TIERS:
        rows: list[dict[str, Any]] = []
        files: dict[str, Any] = {}
        for seed in SEEDS:
            for window in WINDOWS:
                path = root / template.format(budget=tier, window=window,
                                              seed=seed)
                cell = read_jsonl(path)
                files[f"w{window}_seed{seed}"] = {
                    "path": str(path), "exists": path.is_file(),
                    "rows": len(cell),
                    "full": sum(1 for row in cell if row.get("survived"))}
                rows.extend(cell)
        expected = len(SEEDS) * len(WINDOWS) * EPISODES
        if len(rows) == expected:
            full = sum(1 for row in rows if row.get("survived"))
            per_window = {}
            for window in WINDOWS:
                cell = [row for row in rows if int(row["window"]) == int(window)]
                per_window[str(window)] = [sum(1 for r in cell if r["survived"]),
                                           len(cell)]
            per_seed = {}
            for seed in SEEDS:
                cell = [row for row in rows
                        if int(row["policy_seed"]) == int(seed)]
                per_seed[str(seed)] = [sum(1 for r in cell if r["survived"]),
                                       len(cell)]
            survived_rows = [row for row in rows if row.get("survived")]
            out[tier] = {
                "available": True, "rows": len(rows), "full": full,
                "n": len(rows), "pooled": [full, len(rows)],
                "rate": round(full / len(rows), 6),
                "rate_pct": round(100.0 * full / len(rows), 4),
                "cp95": clopper_pearson(full, len(rows)),
                "bombs_mean": round(statistics.fmean(
                    int(row["bombs_used"] or 0) for row in rows), 4),
                "bombs_mean_survived": (
                    round(statistics.fmean([int(row["bombs_used"] or 0)
                                            for row in survived_rows]), 4)
                    if survived_rows else None),
                "distinct": distinct_block(rows),
                "per_window": per_window, "per_seed": per_seed,
                "files": files}
        else:
            full, n = published[tier]
            out[tier] = {"available": False,
                         "reason": f"{len(rows)} rows on disk, {expected} "
                                   f"expected",
                         "rows": len(rows), "full": full, "n": n,
                         "pooled": [full, n],
                         "rate": round(full / n, 6),
                         "rate_pct": round(100.0 * full / n, 4),
                         "cp95": clopper_pearson(full, n),
                         "distinct": distinct_block(rows),
                         "files": files}
        log(f"  reference {root.name}/{template.format(budget=tier, window='*', seed='*')}: "
            f"{out[tier]['pooled'][0]}/{out[tier]['pooled'][1]} = "
            f"{out[tier]['rate_pct']} % "
            f"({'scanned' if out[tier]['available'] else 'published'})")
    return out


def _cell_block(arm: str, budget: int, eval_dir: pathlib.Path) -> dict[str, Any]:
    """One arm x one tier: 3 seeds x 4 windows x 20 episodes = 240 rows."""

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
        survived_cell = [row for row in window_rows if row["survived"]]
        per_window.append({
            "window": int(window), "n": len(window_rows), "full": survived,
            "rate": round(survived / len(window_rows), 6),
            "cp95": clopper_pearson(survived, len(window_rows)),
            "bombs_used_mean": round(statistics.fmean(
                int(row["bombs_used"] or 0) for row in window_rows), 4),
            "bombs_used_mean_survived": (
                round(statistics.fmean([int(row["bombs_used"] or 0)
                                        for row in survived_cell]), 4)
                if survived_cell else None),
            "alive_frames_mean": round(statistics.fmean(
                float(row["alive_frames"] or 0) for row in window_rows), 4),
            "distinct": distinct_block(window_rows)})
    for seed in SEEDS:
        seed_rows = [row for row in rows
                     if int(row["policy_seed"]) == int(seed)]
        if not seed_rows:
            continue
        survived = sum(1 for row in seed_rows if row["survived"])
        survived_cell = [row for row in seed_rows if row["survived"]]
        per_seed[str(seed)] = {
            "n": len(seed_rows), "full": survived,
            "rate": round(survived / len(seed_rows), 6),
            "rate_pct": round(100.0 * survived / len(seed_rows), 4),
            "cp95": clopper_pearson(survived, len(seed_rows)),
            "bombs_used_mean": round(statistics.fmean(
                int(row["bombs_used"] or 0) for row in seed_rows), 4),
            "bombs_used_mean_survived": (
                round(statistics.fmean([int(row["bombs_used"] or 0)
                                        for row in survived_cell]), 4)
                if survived_cell else None),
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
                             "bombs_used_mean": round(statistics.fmean(
                                 int(row["bombs_used"] or 0)
                                 for row in cell_rows), 4),
                             "distinct_mean": round(statistics.fmean(
                                 distinct_of(row) for row in cell_rows), 4)})
    survived_all = sum(1 for row in rows if row["survived"])
    seed_rates = [value["rate"] for value in per_seed.values()]
    spread = (round(100.0 * (max(seed_rates) - min(seed_rates)), 4)
              if len(seed_rates) > 1 else None)
    return {
        "arm": str(arm), "arm_label": arm_label(arm),
        "alpha": alpha_of(arm), "budget": int(budget),
        "tier_label": TIER_LABEL.get(int(budget)),
        "rule": rule_block(alpha_of(arm)),
        "files": files, "rows": len(rows), "expected_rows": expected,
        "complete": bool(len(rows) == expected),
        "cells_complete": sum(1 for value in files.values()
                              if value["rows"] == EPISODES),
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
        "thresholds_realised": _threshold_block(rows),
        "bomb_usage": _bomb_block(rows),
        "invalid_rows": sum(1 for row in rows if row.get("invalid")),
        "wall_s_total": round(sum(float(row["wall_s"] or 0) for row in rows), 3),
    }


def _fisher_row(observed: dict[str, Any], reference: dict[str, Any]
                ) -> dict[str, Any]:
    k, n = int(observed["full"]), int(observed["n"])
    rk, rn = int(reference["full"]), int(reference["n"])
    if not n or not rn:
        return {"p_greater": None, "p_less": None, "difference_pp": None,
                "table": None, "odds_ratio_haldane": None}
    greater = fisher_greater(k, n - k, rk, rn - rk)
    less = fisher_less(k, n - k, rk, rn - rk)
    return {"p_greater": greater["p_value"], "p_less": less["p_value"],
            "difference_pp": greater["difference_pp"],
            "table": [[k, n - k], [rk, rn - rk]],
            "odds_ratio_haldane": greater["odds_ratio_haldane"]}


def mode_summary(args: argparse.Namespace) -> int:
    eval_dir = pathlib.Path(args.eval_dir) if args.eval_dir else WORK / "eval"
    out_path = pathlib.Path(args.out) if args.out else WORK / "summary.json"
    arms = tuple(str(part) for part in str(args.arms).split(",") if part.strip())
    for arm in arms:
        check_arm(arm)
    tiers = tuple(check_tier(part) for part in str(args.tiers).split(",")
                  if part.strip())

    # -- the three arms x three tiers -------------------------------------- #
    blocks: dict[str, dict[str, Any]] = {}
    for arm in arms:
        blocks[arm] = {str(tier): _cell_block(arm, tier, eval_dir)
                       for tier in tiers}

    # -- the two fixed-threshold arms, re-derived from their raw rows ------- #
    log("summary: re-deriving the fixed arms from raw rows")
    g0 = _scan_reference(G0_TEMPLATE, G0_DIR, G0_PUBLISHED)
    fix = _scan_reference(FIX_TEMPLATE, FIX_DIR, FIX_PUBLISHED)
    bar: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        name, best = max([("G0", g0[tier]), ("FIX", fix[tier])],
                         key=lambda item: item[1]["full"] / item[1]["n"])
        bar[str(tier)] = {
            "tier": tier, "arm": name,
            "label": (G0_LABEL if name == "G0" else FIX_LABEL),
            "full": int(best["full"]), "n": int(best["n"]),
            "rate_pct": round(100.0 * int(best["full"]) / int(best["n"]), 4)}

    # -- per (arm, tier) verdict rows -------------------------------------- #
    rows_out: list[dict[str, Any]] = []
    for arm in arms:
        for tier in tiers:
            block = blocks[arm][str(tier)]
            observed = block["pooled"]
            k, n = int(observed["full"]), int(observed["n"])
            bar_row = bar[str(tier)]
            bk, bn = int(bar_row["full"]), int(bar_row["n"])
            bar_met = bool(n and k * bn >= bk * n)
            f_vs_g0 = _fisher_row(observed, g0[tier])
            f_vs_fix = _fisher_row(observed, fix[tier])
            f_vs_bar = _fisher_row(observed, bar_row)
            improved = bool(n and k * bn > bk * n
                            and (f_vs_bar["p_greater"] or 1.0) < P2_ALPHA)
            rows_out.append({
                "arm": arm, "arm_label": arm_label(arm),
                "alpha": alpha_of(arm), "tier_label": TIER_LABEL[tier],
                "budget": tier,
                "pooled_full": k, "pooled_n": n,
                "pooled_rate": observed["rate"],
                "pooled_rate_pct": observed["rate_pct"],
                "cp95": observed["cp95"],
                "distinct": block["distinct"],
                "bombs": {"all": block["bomb_usage"]["all"]["mean"],
                          "survived": block["bomb_usage"]["survived"]["mean"],
                          "died": block["bomb_usage"]["died"]["mean"]},
                "bar": bar_row, "bar_met": bar_met,
                "delta_vs_bar_pp": (round(100.0 * (k / n - bk / bn), 4)
                                    if n else None),
                "fisher_vs_bar": f_vs_bar,
                "fisher_vs_g0": f_vs_g0,
                "fisher_vs_fix": f_vs_fix,
                "improved_vs_bar": improved,
                "g0": {"full": g0[tier]["full"], "n": g0[tier]["n"],
                       "rate_pct": g0[tier]["rate_pct"],
                       "cp95": g0[tier]["cp95"]},
                "fix": {"full": fix[tier]["full"], "n": fix[tier]["n"],
                        "rate_pct": fix[tier]["rate_pct"],
                        "cp95": fix[tier]["cp95"]},
                "per_window": {f"w{bucket['window']}": [bucket["full"],
                                                        bucket["n"]]
                               for bucket in block["per_window_table"]},
                "per_window_distinct": {
                    f"w{bucket['window']}": bucket["distinct"]["mean"]
                    for bucket in block["per_window_table"]},
                "per_seed": {seed: [value["full"], value["n"]]
                             for seed, value in block["per_seed"].items()},
                "per_seed_distinct": {
                    seed: (value.get("distinct") or {}).get("mean")
                    for seed, value in block["per_seed"].items()},
                "per_seed_spread_pp": block["per_seed_spread_pp"],
                "per_cell": block["per_cell"],
                "cells_complete": block["cells_complete"],
                "complete": block["complete"],
                "invalid_rows": block["invalid_rows"],
            })

    # -- P1 / P2 / P3 / P4 -------------------------------------------------- #
    def arm_verdict(arm: str) -> dict[str, Any]:
        cells = [row for row in rows_out if row["arm"] == arm]
        complete = bool(all(row["complete"] for row in cells))
        p1 = bool(complete and all(row["bar_met"] for row in cells))
        named = [row for row in cells if int(row["budget"]) in (900, 1200)]
        p2_named_rows = [row for row in named if row["improved_vs_bar"]]
        p2 = bool(complete and p2_named_rows)
        return {"arm": arm, "alpha": alpha_of(arm), "complete": complete,
                "P1": p1, "P1_per_tier": {
                    f"d{row['budget']}": {"observed_pct": row["pooled_rate_pct"],
                                          "bar_pct": row["bar"]["rate_pct"],
                                          "bar_arm": row["bar"]["arm"],
                                          "met": row["bar_met"]}
                    for row in cells},
                "P2": p2,
                "P2_named_tiers": [row["budget"] for row in p2_named_rows],
                "P2_any_tier": bool(complete and any(row["improved_vs_bar"]
                                                     for row in cells)),
                "P2_improved_tiers": [row["budget"] for row in cells
                                      if row["improved_vs_bar"]],
                "min_delta_vs_bar_pp": (min(row["delta_vs_bar_pp"]
                                            for row in cells)
                                        if cells else None),
                "pooled_rate_pct_sum": round(sum(row["pooled_rate_pct"]
                                                 for row in cells), 4)}

    per_arm = {arm: arm_verdict(arm) for arm in arms}
    adaptive = [arm for arm in arms
                if per_arm[arm]["P1"] and per_arm[arm]["P2"]]
    partial = [arm for arm in arms
               if per_arm[arm]["P2"] and not per_arm[arm]["P1"]]
    any_p2 = any(per_arm[arm]["P2"] for arm in arms)
    any_p1 = any(per_arm[arm]["P1"] for arm in arms)
    p3 = {arm: {f"d{row['budget']}": row["bombs"]
                for row in rows_out if row["arm"] == arm} for arm in arms}
    p4 = {arm: {f"d{row['budget']}": row["distinct"]["mean"]
                for row in rows_out if row["arm"] == arm} for arm in arms}

    # -- stop condition (prereg §2.1) -------------------------------------- #
    d3 = read_json(WORK / "d3_gate.json")
    d3_pass = bool(d3.get("all_pass"))
    all_complete = all(row["complete"] for row in rows_out)
    if not d3:
        primary = "S-incomplete"
        stop_text = ("the D3 fidelity gate has not been run; no threshold "
                     "conclusion may be drawn")
    elif not d3_pass:
        primary = "S-fork"
        stop_text = ("the D3 fidelity gate is not 5/5; the round is void and no "
                     "threshold conclusion may be drawn")
    elif not all_complete:
        primary = "S-incomplete"
        stop_text = ("not every (arm, tier) cell has 240 rows; the prereg's stop "
                     "conditions are not declared on a partial table")
    elif adaptive:
        primary = "S-adaptive"
        stop_text = (
            f"P1 + P2 hold for {adaptive}: the urgency formula clears all three "
            f"fixed-threshold bars at once (20 s >= 98.75 %, 30 s >= 48.75 %, "
            f"40 s >= 12.50 %) and is strictly better than the fixed bar on "
            f"{ {arm: per_arm[arm]['P2_named_tiers'] for arm in adaptive} } "
            f"(one-sided Fisher p < 0.05).  The urgency-adaptive threshold is "
            f"the unified deployment configuration.")
    elif any_p2:
        pool = partial or list(arms)
        best = max(pool, key=lambda arm: per_arm[arm]["min_delta_vs_bar_pp"]
                   if per_arm[arm]["min_delta_vs_bar_pp"] is not None else -999)
        primary = "S-partial"
        stop_text = (
            f"P2 holds for {partial} but P1 does not: the adaptive threshold is "
            f"better than the fixed bar on "
            f"{ {arm: per_arm[arm]['P2_improved_tiers'] for arm in arms} } and "
            f"worse on the rest.  The deployment is the per-tier best table, "
            f"not a single configuration ({best} has the best worst-tier "
            f"margin: {per_arm[best]['min_delta_vs_bar_pp']} pp).")
    else:
        primary = "S-null"
        stop_text = (
            "P2 fails for every alpha: no tier is strictly better than its best "
            "fixed threshold (one-sided Fisher p < 0.05).  The urgency response "
            "is void; keep S-switch (20 s 60/40/20, 30 s 35/15/3).")

    # -- the side-by-side table -------------------------------------------- #
    table = []
    for row in rows_out:
        table.append({
            "arm": row["arm"], "alpha": row["alpha"],
            "tier": f"{row['budget']} ({row['tier_label']})",
            "budget": row["budget"],
            "pooled": f"{row['pooled_full']}/{row['pooled_n']}",
            "rate_pct": row["pooled_rate_pct"],
            "cp95": [row["cp95"]["low"], row["cp95"]["high"]],
            "bar": (f"{row['bar']['full']}/{row['bar']['n']} "
                    f"({row['bar']['rate_pct']} %)"),
            "delta_pp": row["delta_vs_bar_pp"],
            "fisher_p_greater": row["fisher_vs_bar"]["p_greater"],
            "fisher_p_less": row["fisher_vs_bar"]["p_less"],
            "bar_met": row["bar_met"],
            "distinct": row["distinct"]["mean"],
            "distinct_min": row["distinct"]["min"],
            "distinct_max": row["distinct"]["max"],
            "bombs_all": row["bombs"]["all"],
            "bombs_survived": row["bombs"]["survived"],
            "bombs_died": row["bombs"]["died"],
            "w11": f"{row['per_window']['w11'][0]}/"
                   f"{row['per_window']['w11'][1]}",
            "w12": f"{row['per_window']['w12'][0]}/"
                   f"{row['per_window']['w12'][1]}",
            "w17": f"{row['per_window']['w17'][0]}/"
                   f"{row['per_window']['w17'][1]}",
            "w18": f"{row['per_window']['w18'][0]}/"
                   f"{row['per_window']['w18'][1]}",
            "s0": f"{row['per_seed']['0'][0]}/{row['per_seed']['0'][1]}",
            "s1": f"{row['per_seed']['1'][0]}/{row['per_seed']['1'][1]}",
            "s2": f"{row['per_seed']['2'][0]}/{row['per_seed']['2'][1]}",
            "seed_spread_pp": row["per_seed_spread_pp"],
        })

    payload = {
        "schema": SCHEMA_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": {"path": PREREG_REL,
                   "sha256": (sha256_file(REPO / PREREG_REL)
                              if (REPO / PREREG_REL).is_file() else None)},
        "question": ("can ONE urgency-adaptive threshold triple "
                     "(thr[b] = base[b] * (R/max(B,1)/200)**alpha, clipped to "
                     "[5,120] px) clear the best FIXED threshold of every "
                     "budget at once -- 98.75 % at 20 s, 48.75 % at 30 s, "
                     "12.50 % at 40 s?"),
        "protocol": {
            "candidate_head": ("v8.6.0 bombdag.pt Policy(106,64,36), frozen; "
                               "score[c] = max(logit[2c], logit[2c+1]); "
                               "cand ~ softmax(score / 1.0), one draw per "
                               "decision from torch.Generator(seed*1000+7)"),
            "bomb": FORMULA,
            "alpha": {arm: alpha_of(arm) for arm in arms},
            "base_thresholds": {str(key): float(value)
                                for key, value in BASE_THRESHOLDS.items()},
            "floor_px": THR_FLOOR, "ceil_px": THR_CEIL,
            "urgency_base": URGENCY_BASE,
            "tiers": {str(tier): TIER_LABEL[tier] for tier in tiers},
            "seeds": list(SEEDS), "windows": list(WINDOWS),
            "episodes_per_window": EPISODES,
            "decision_frames": DECISION_FRAMES,
            "prefix_frame": PREFIX_FRAMES,
            "frames_per_episode": {str(tier): tier * DECISION_FRAMES
                                   for tier in tiers},
            "loop": ("for arm in (G1,G2,G3): for tier in (600,900,1200): for "
                     "seed in (0,1,2): one process; inside it for window in "
                     "(11,12,17,18): for episode in range(20)"),
            "episode_loop": "tools/rl/v830_bombpol.py::run_episode36 (imported)",
            "urgency_reading": ("decisions_executed = the frozen loop's 0-based "
                                "decision index, i.e. exactly the engine's "
                                "executed-decision count when the decision is "
                                "taken (prereg §1: real remaining decisions, not "
                                "remaining frames)"),
        },
        "reference": {
            "G0": {"label": G0_LABEL, "dir": str(G0_DIR),
                   "published": {str(k): v for k, v in G0_PUBLISHED.items()},
                   "tiers": {str(tier): g0[tier] for tier in tiers}},
            "FIX": {"label": FIX_LABEL, "dir": str(FIX_DIR),
                    "published": {str(k): v for k, v in FIX_PUBLISHED.items()},
                    "tiers": {str(tier): fix[tier] for tier in tiers}},
            "bar": bar,
            "bar_note": ("P1's bar per tier is the better of the two fixed "
                         "arms on that tier (prereg §2: 'the best fixed "
                         "threshold of that tier')"),
        },
        "arms": blocks,
        "table": table,
        "rows": rows_out,
        "verdict": {
            "P1": {"requirement": ("all three tiers at once >= the best fixed "
                                   "threshold of that tier (20 s >= 98.75 %, "
                                   "30 s >= 48.75 %, 40 s >= 12.50 %)"),
                   "per_arm": {arm: {"P1": per_arm[arm]["P1"],
                                     "per_tier": per_arm[arm]["P1_per_tier"]}
                               for arm in arms},
                   "arms_bar_met_all_tiers": [arm for arm in arms
                                              if per_arm[arm]["P1"]],
                   "passed": bool(any_p1)},
            "P2": {"requirement": ("at least one tier strictly better than the "
                                   "fixed bar (one-sided Fisher p < 0.05); "
                                   "prereg names 30 s > 48.75 % or "
                                   "40 s > 12.50 %"),
                   "alpha": P2_ALPHA,
                   "per_arm": {arm: {"P2": per_arm[arm]["P2"],
                                     "named_tiers":
                                         per_arm[arm]["P2_named_tiers"],
                                     "improved_tiers":
                                         per_arm[arm]["P2_improved_tiers"],
                                     "any_tier": per_arm[arm]["P2_any_tier"]}
                               for arm in arms},
                   "passed": bool(any_p2)},
            "P3": {"requirement": ("bomb usage of the three tiers, survivors "
                                   "reported separately (§35.12)"),
                   "per_arm_per_tier": p3, "reported": True},
            "P4": {"requirement": "distinct candidate indices reported in "
                                  "every cell (tier / window / seed / cell)",
                   "per_arm_per_tier_mean": p4, "reported": True},
            "stop_condition": primary,
            "stop_text": stop_text,
            "S-adaptive": {"condition": "P1 and P2 both hold", "held": bool(
                d3_pass and all_complete and adaptive),
                "arms": adaptive},
            "S-partial": {"condition": "P2 holds, P1 does not", "held": bool(
                d3_pass and all_complete and not adaptive and any_p2),
                "arms": partial},
            "S-null": {"condition": "P2 does not hold", "held": bool(
                d3_pass and all_complete and not any_p2)},
            "S-fork": {"condition": "the D3 fidelity gate fails",
                       "held": bool(d3 and not d3_pass),
                       "d3_all_pass": d3_pass},
        },
        "distinct_note": ("`distinct` is the number of distinct candidate "
                          "indices (of 18) the softmax sampler touched inside "
                          "one episode; it is reported per arm x tier, per "
                          "window, per seed and per cell so argmax-like collapse "
                          "would be visible."),
        "invalid_rows": sum(row["invalid_rows"] for row in rows_out),
        "notes": [
            "No training and no collection: this file only evaluates.",
            "The candidate head, the sampler and the base triple {3:35, 2:15, "
            "1:3} are frozen; only alpha (0.50 / 1.00 / 1.50) moves.",
            "G0 (fixed 35/15/3) and FIX (fixed 60/40/20) are NOT re-run: both "
            "are re-derived from their 240 raw rows per tier on disk.",
            "P1's and P2's Fisher tests are one-sided against the better fixed "
            "arm of that tier.",
            "The 12 (seed, window) cells are not independent samples; the "
            "pooled rate carries its exact (Clopper-Pearson) interval.",
            "No argmax reading is admissible anywhere (prereg §5, §32.5).",
        ],
    }
    write_json(out_path, payload)
    log(f"summary: arms={list(arms)} tiers={list(tiers)} -> {out_path}")
    for row in rows_out:
        log(f"  {row['arm']} (a{row['alpha']:.2f}) d{row['budget']}: "
            f"{row['pooled_full']}/{row['pooled_n']} = "
            f"{row['pooled_rate_pct']} % "
            f"CP95=[{row['cp95']['low']}, {row['cp95']['high']}] "
            f"bar={row['bar']['full']}/{row['bar']['n']} "
            f"({row['bar']['rate_pct']} %) delta={row['delta_vs_bar_pp']} pp "
            f"p_greater={row['fisher_vs_bar']['p_greater']} "
            f"met={row['bar_met']} distinct={row['distinct']['mean']} "
            f"bombs(all/surv/died)={row['bombs']['all']}/"
            f"{row['bombs']['survived']}/{row['bombs']['died']}")
        log(f"    w11={row['per_window']['w11']} w12={row['per_window']['w12']} "
            f"w17={row['per_window']['w17']} w18={row['per_window']['w18']} "
            f"s0={row['per_seed']['0']} s1={row['per_seed']['1']} "
            f"s2={row['per_seed']['2']} spread={row['per_seed_spread_pp']}pp")
    log(f"  P1={any_p1} P2={any_p2} P3(reported) P4(reported) -> verdict: "
        f"{primary}")
    log(f"  {stop_text}")
    return 0 if payload["invalid_rows"] == 0 else 4


# --------------------------------------------------------------------------- #
# mode: report (RUN.md from summary.json)
# --------------------------------------------------------------------------- #
def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def _num(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def mode_report(args: argparse.Namespace) -> int:
    summary = read_json(WORK / "summary.json")
    if not summary:
        raise UrgencyError("summary.json missing; run --mode summary first")
    out_path = pathlib.Path(args.out) if args.out else WORK / "RUN.md"
    verdict = summary.get("verdict") or {}
    d3 = read_json(WORK / "d3_gate.json")
    audio = read_json(WORK / "audio_audit.json")
    proc = read_json(WORK / "process_residue.json")
    smoke = read_json(WORK / "smoke.json")
    assert_start = read_json(WORK / "ckpt_assertions_start.json")
    assert_final = read_json(WORK / "ckpt_assertions_final.json")
    rows = summary.get("rows") or []
    lines: list[str] = []
    lines.append("# v8.26.0 URGENCY - urgency-adaptive bomb thresholds "
                 "(20 / 30 / 40 s)")
    lines.append("")
    lines.append(f"- generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    lines.append(f"- producer: `{PRODUCER}`")
    lines.append(f"- prereg: `{PREREG_REL}` "
                 f"sha256 `{(summary.get('prereg') or {}).get('sha256')}`")
    lines.append(f"- rule: `{FORMULA}`")
    lines.append("- arms: " + ", ".join(
        f"{arm} alpha={alpha_of(arm):.2f}"
        for arm in summary.get("protocol", {}).get("alpha", {})))
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    lines.append(f"- ckpt assertions (start): "
                 f"`{assert_start.get('all_pass')}`; (final): "
                 f"`{assert_final.get('all_pass')}`")
    lines.append(f"- D3 fidelity (5 assertions): "
                 f"`{d3.get('passed')}/{d3.get('total')}` "
                 f"all_pass=`{d3.get('all_pass')}`")
    lines.append(f"- smoke ({smoke.get('arm')} d{smoke.get('budget')} "
                 f"seed{smoke.get('policy_seed')} w{smoke.get('window')}): "
                 f"all_pass=`{smoke.get('all_pass')}`, "
                 f"episodes={len(smoke.get('rows') or [])}, "
                 f"emissions={len(smoke.get('emissions') or [])}")
    lines.append(f"- audio audit: muted=`{audio.get('muted')}` "
                 f"audible=`{audio.get('audible')}` "
                 f"all_pass=`{audio.get('all_pass')}`")
    lines.append(f"- process residue: own engines=`{proc.get('matching')}` "
                 f"all_pass=`{proc.get('all_pass')}`")
    lines.append(f"- invalid rows: `{summary.get('invalid_rows')}`")
    lines.append("")
    first5 = smoke.get("first_5_emissions") or []
    if first5:
        lines.append("## Smoke: first bomb emissions (formula live)")
        lines.append("")
        lines.append("| # | episode | decision | R | B | urgency | scale | "
                     "thr px | nearest px |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for index, item in enumerate(first5):
            lines.append(
                f"| {index + 1} | {item['episode']} | {item['decision']} | "
                f"{item['remaining_decisions']} | {item['bombs_before']} | "
                f"{item['urgency']:.4f} | {item['scale']:.4f} | "
                f"{item['threshold_px']:.2f} | "
                f"{_num(item['nearest_all_px'], 2)} |")
        lines.append("")
    lines.append("## Three arms x three tiers")
    lines.append("")
    lines.append("| arm | alpha | tier | survived | rate % | CP95 | "
                 "bar (best fixed) | delta pp | Fisher p(>) | Fisher p(<) | "
                 "bar met | distinct | bombs all | bombs survived | bombs died |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        cp = row["cp95"]
        lines.append(
            f"| {row['arm']} | {row['alpha']:.2f} | {row['budget']} "
            f"({row['tier_label']}) | {row['pooled_full']}/{row['pooled_n']} | "
            f"{_pct(row['pooled_rate_pct'])} | "
            f"[{_pct(100 * cp['low'])}, {_pct(100 * cp['high'])}] | "
            f"{row['bar']['full']}/{row['bar']['n']} "
            f"({_pct(row['bar']['rate_pct'])} %) | "
            f"{_num(row['delta_vs_bar_pp'], 2)} | "
            f"{_num(row['fisher_vs_bar']['p_greater'], 4)} | "
            f"{_num(row['fisher_vs_bar']['p_less'], 4)} | "
            f"{row['bar_met']} | {_num(row['distinct']['mean'], 2)} | "
            f"{_num(row['bombs']['all'], 2)} | "
            f"{_num(row['bombs']['survived'], 2)} | "
            f"{_num(row['bombs']['died'], 2)} |")
    lines.append("")
    lines.append("## Per window")
    lines.append("")
    lines.append("| arm | tier | w11 | w12 | w17 | w18 |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        cells = [f"{row['per_window'][f'w{w}'][0]}/"
                 f"{row['per_window'][f'w{w}'][1]}" for w in (11, 12, 17, 18)]
        lines.append(f"| {row['arm']} | {row['budget']} | " +
                     " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per seed")
    lines.append("")
    lines.append("| arm | tier | seed0 | seed1 | seed2 | spread pp |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        cells = [f"{row['per_seed'][str(s)][0]}/{row['per_seed'][str(s)][1]}"
                 for s in (0, 1, 2)]
        lines.append(f"| {row['arm']} | {row['budget']} | " +
                     " | ".join(cells) + " | "
                     f"{_num(row['per_seed_spread_pp'], 2)} |")
    lines.append("")
    lines.append("## Per cell (window x seed)")
    lines.append("")
    lines.append("| arm | tier | cell | survived | rate % | bombs mean | "
                 "distinct |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        for cell in row["per_cell"]:
            lines.append(
                f"| {row['arm']} | {row['budget']} | w{cell['window']}"
                f"/s{cell['policy_seed']} | {cell['full']}/{cell['n']} | "
                f"{_pct(100 * cell['rate'])} | "
                f"{_num(cell['bombs_used_mean'], 2)} | "
                f"{_num(cell['distinct_mean'], 2)} |")
    lines.append("")
    lines.append("## Verdict (prereg §2)")
    lines.append("")
    lines.append(f"- **P1** (all three tiers >= best fixed): "
                 f"`{verdict.get('P1', {}).get('passed')}`")
    for arm, node in (verdict.get("P1", {}).get("per_arm") or {}).items():
        tier_text = ", ".join(
            f"{key}: {_pct(value['observed_pct'])} vs "
            f"{_pct(value['bar_pct'])} ({'OK' if value['met'] else 'MISS'})"
            for key, value in (node.get("per_tier") or {}).items())
        lines.append(f"  - {arm}: {tier_text}")
    lines.append(f"- **P2** (at least one tier strictly better, Fisher "
                 f"p<0.05): `{verdict.get('P2', {}).get('passed')}`")
    for arm, node in (verdict.get("P2", {}).get("per_arm") or {}).items():
        lines.append(f"  - {arm}: named-tier better = "
                     f"{node.get('named_tiers')}, improved tiers = "
                     f"{node.get('improved_tiers')}")
    lines.append("- **P3** (bomb usage, survivors apart): reported per arm and "
                 "tier in the table above")
    lines.append(f"- **P4** (distinct per cell): reported for all "
                 f"`{len(rows)}` arm x tier rows, every window, seed and cell")
    lines.append("")
    lines.append(f"**Stop condition: `{verdict.get('stop_condition')}`**")
    lines.append("")
    lines.append(f"{verdict.get('stop_text')}")
    lines.append("")
    lines.append("## Threshold schedule actually realised")
    lines.append("")
    lines.append("| arm | tier | urgency min/mean/max | scale min/mean/max | "
                 "thr@B=3 mean [min,max] | thr@B=2 | thr@B=1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in summary.get("protocol", {}).get("alpha", {}):
        for tier in summary.get("protocol", {}).get("tiers", {}):
            block = ((summary.get("arms") or {}).get(arm) or {}).get(tier) or {}
            thr = block.get("thresholds_realised") or {}
            urgency = thr.get("urgency") or {}
            scale = thr.get("scale") or {}
            by_bombs = thr.get("active_threshold_by_bombs") or {}
            cells = []
            for key in ("3", "2", "1"):
                node = by_bombs.get(key) or {}
                cells.append(
                    "n/a" if not node.get("n")
                    else f"{_num(node.get('mean_of_means'), 2)} "
                         f"[{_num(node.get('min'), 2)}, "
                         f"{_num(node.get('max'), 2)}]")
            lines.append(
                f"| {arm} | {tier} | "
                f"{_num(urgency.get('min'), 2)}/"
                f"{_num(urgency.get('mean_of_means'), 2)}/"
                f"{_num(urgency.get('max'), 2)} | "
                f"{_num(scale.get('min'), 3)}/"
                f"{_num(scale.get('mean_of_means'), 3)}/"
                f"{_num(scale.get('max'), 3)} | " +
                " | ".join(cells) + " |")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report: {len(rows)} rows -> {out_path}")
    return 0


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
    census = engine_census()
    payload: dict[str, Any] = {
        "schema": SCHEMA_PROC, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("this round's engine processes must be 0 when it ends; "
                        "other rounds' processes are counted but never touched"),
        "needle": MINE, "pgrep_returncode": census["pgrep_returncode"],
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
    tiers = tuple(check_tier(part) for part in str(args.tiers).split(",")
                  if part.strip())
    copied: list[dict[str, Any]] = []
    sources: list[pathlib.Path] = []
    for arm in arms:
        for tier in tiers:
            for seed in SEEDS:
                for window in WINDOWS:
                    sources.append(eval_file(arm, tier, window, seed))
                sources.append(progress_file(arm, tier, seed))
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
        description=("v8.26.0 URGENCY - urgency-adaptive bomb thresholds "
                     "(thr[b] = base[b] * (R/max(B,1)/200)**alpha, clipped to "
                     "[5,120] px) at 600 / 900 / 1,200 decisions, "
                     "3 arms x 3 tiers x 3 seeds x 4 windows x 20 episodes"))
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "assert-ckpt", "d3", "smoke",
                                 "eval", "summary", "report", "audio-audit",
                                 "process-audit", "publish"))
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--arms", default=",".join(ARM_ORDER))
    parser.add_argument("--arm", default=ARM_ORDER[0])
    parser.add_argument("--tiers", default=",".join(str(t) for t in TIERS))
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
        if args.mode == "report":
            return mode_report(args)
        if args.mode == "audio-audit":
            return mode_audio(args)
        if args.mode == "process-audit":
            return mode_process(args)
        if args.mode == "publish":
            return mode_publish(args)
        raise UrgencyError(f"unknown mode {args.mode}")
    finally:
        if _LOG_FH is not None:
            _LOG_FH.flush()


if __name__ == "__main__":
    sys.exit(main())
