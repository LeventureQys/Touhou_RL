#!/usr/bin/env python3
"""v8.19.0 RULESWEEP30 — the R5 bomb-rule **thresholds** swept at 900 decisions.

What this round asks
--------------------
v8.19.0's prereg (``Document/Update/Alpha/v8.19.0/PREREGISTRATION_RULESWEEP30_001.md``)
records the 30 s ceiling of the pure script rule: R5 ``staged`` (3 -> 80 px,
2 -> 60 px, 1 -> 40 px) holds **634 / 3,600 = 17.6111 %** at 900 decisions
(``v8.4.0/evidence/BOMBRULE2/eval_R5_d900_seed{0,1,2}.jsonl``).  The rule has
never been swept on 30 s.  This file re-runs **one fixed candidate head** with
only the threshold triple moved:

    R0 (control, NOT re-run)  80 / 60 / 40      = the published 634/3,600
    S1                        60 / 40 / 20      later bombs (thrift)
    S2                       100 / 80 / 60      earlier bombs (rescue)
    S3                        40 / 30 / 20      very late bombs
    S4                       120 / 90 / 60      very early bombs

The arm itself is **v8.12.0 H2s verbatim** except for that triple
---------------------------------------------------------------
* policy: ``.tmp_v860_bombdag/ckpt/seed{seed}/bombdag.pt`` — ``Policy(106, 64,
  36)`` loaded by ``v860_bombdag.load_policy36`` (the same loader v8.6.0,
  v8.12.0's H2s and v8.14.0's HYBRID30 use);
* candidate: ``score[c] = max(logit[2c], logit[2c+1])``, ``softmax(score / 1.0)``
  and **one sampled draw per decision** from one ``torch.Generator()`` seeded
  ``seed * 1000 + 7`` per process (created before the window loop, consumed once
  per decision across every window and episode);
* bomb: the external rule — ``nearest_all_px < THRESHOLDS[bombs_remaining]``
  **and** ``bombs_remaining > 0``; the distance itself is still measured by
  ``v800_bomb.nearest_bullet_distances`` (v8.0.0's own code, not re-derived);
* masks: ``cand_to_mask(cand, bomb)`` on the decision's first frame and
  ``cand_to_mask(cand, 0)`` afterwards — inside the imported
  ``v830_bombpol.run_episode36``, not re-implemented here;
* start: PARK frame 8663; 2 frames per decision; 900 decisions = 1,800 frames;
* loop: ``for seed in (0,1,2): for window in (11,12,17,18): for episode in
  range(20)`` — 3 x 4 x 20 = **240 episodes per arm**.

Modes
-----
``--mode assert-ckpt``   engine sha256, the three v8.6.0 ``bombdag.pt``
                         (``9c6d5d4a`` / ``36a51090`` / ``885dde70``), the three
                         frozen v4.5.0 ``round2.pt`` (``4abee069`` / ``a93746bb``
                         / ``c69960fc``), the protected sources.  Fail closed.
``--mode d3``            the D3 fidelity gate (5 assertions) through the frozen
                         ``v432_fork_arm.py --mode fidelity``; a non-5/5 gate
                         voids the round.
``--mode smoke``         **one** episode (S1 / seed0 / w11 / 900), then stop:
                         prints ``survived`` / ``bombs_used`` / ``alive_frames``.
``--mode eval``          one ``--arm`` x one ``--policy-seed``: 4 windows x 20
                         episodes, one ``eval_{arm}_d900_w{w}_seed{s}.jsonl`` per
                         window.
``--mode summary``       per-arm x per-window table + pooled rate + exact
                         Clopper-Pearson 95 % + bomb usage + one-sided Fisher
                         against the published R5 cell -> ``summary.json``.
``--mode audio-audit``   AGENTS.md §9.3 on this round's own data dirs.
``--mode process-audit`` this round's own engine processes must be 0.
``--mode publish``       copy the raw artifacts into
                         ``Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/``
                         and write ``manifest.json``.

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

WORK = REPO / ".tmp_v8k0_rulesweep30"
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.19.0" / "evidence"
           / "RULESWEEP30")

SCHEMA_ASSERT = "v8k0-rulesweep30-ckpt-assertions/1"
SCHEMA_D3 = "v8k0-rulesweep30-d3-gate/1"
SCHEMA_SMOKE = "v8k0-rulesweep30-smoke/1"
SCHEMA_EPISODE = "v8k0-rulesweep30-episode/1"
SCHEMA_SEED = "v8k0-rulesweep30-arm-seed-progress/1"
SCHEMA_SUMMARY = "v8k0-rulesweep30-summary/1"
SCHEMA_AUDIO = "v8k0-rulesweep30-audio-audit/1"
SCHEMA_PROC = "v8k0-rulesweep30-process-audit/1"
SCHEMA_MANIFEST = "v8k0-rulesweep30-manifest/1"
SCHEMA_PUBLISH = "v8k0-rulesweep30-publish/1"
PRODUCER = ("tools/rl/v8k0_rulesweep30.py (live engine, HarnessClient "
            "role=shadow, episode loop = tools/rl/v830_bombpol.py::"
            "run_episode36)")
PREREG_REL = "Document/Update/Alpha/v8.19.0/PREREGISTRATION_RULESWEEP30_001.md"

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- the frozen protocol (prereg §1) -----------------------------------------
SEEDS: tuple[int, ...] = (0, 1, 2)
WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
EPISODES = 20
BUDGET = 900
DECISION_FRAMES = 2
PREFIX_FRAMES = 8663
N_CAND = 18
N_ACTIONS = 36
TEMPERATURE = 1.0
GEN_CONST = 7

# -- **the one moved factor**: the threshold triple (prereg §1) --------------
THRESHOLD_ARMS: dict[str, dict[int, float]] = {
    "R0": {3: 80.0, 2: 60.0, 1: 40.0},     # = published R5; NOT re-run here
    "S1": {3: 60.0, 2: 40.0, 1: 20.0},
    "S2": {3: 100.0, 2: 80.0, 1: 60.0},
    "S3": {3: 40.0, 2: 30.0, 1: 20.0},
    "S4": {3: 120.0, 2: 90.0, 1: 60.0},
}
SWEEP_ARMS: tuple[str, ...] = ("S1", "S2", "S3", "S4")
CONTROL_ARM = "R0"

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
    "tools/rl/v8e0_hybrid30.py",
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)

# -- the reference this round is placed beside (numbers only, no test) -------
#: R5 `staged` (80/60/40) at 900 decisions, re-derived from the published
#: ``Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R5_d900_seed{k}.jsonl``
#: (3 files x 1,200 rows): pooled 634/3,600 = 17.6111 %, per window
#: 198/143/107/186 out of 900, per seed 225/409/0 out of 1,200.
R5_REFERENCE: dict[str, Any] = {
    "arm": "v8.4.0 R5 `staged` — 3 -> 80 px, 2 -> 60 px, 1 -> 40 px, "
           "`nearest_all_px < threshold`, bombs > 0",
    "budget_decisions": 900,
    "pooled": (634, 3600),
    "per_window": {11: (198, 900), 12: (143, 900), 17: (107, 900),
                   18: (186, 900)},
    "per_seed": {0: (225, 1200), 1: (409, 1200), 2: (0, 1200)},
    "source": ("Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/"
               "eval_R5_d900_seed{0,1,2}.jsonl (3 x 1,200 rows, re-derived "
               "here; pooled 634/3,600 = 17.6111 %)"),
    "rerun_here": False,
}
P1_THRESHOLD = 0.24          # R5's 17.6111 % + 6.4 pp
P2_ALPHA = 0.05

_LOG_FH: Any = None
_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class RuleSweepError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v8k0] {message}"
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
        raise RuleSweepError(f"cannot import {path}")
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


def thresholds_of(arm: str) -> dict[int, float]:
    key = str(arm).upper()
    if key not in THRESHOLD_ARMS:
        raise RuleSweepError(f"unknown arm {arm!r}; arms are "
                             f"{sorted(THRESHOLD_ARMS)}")
    return dict(THRESHOLD_ARMS[key])


def thresholds_label(arm: str) -> str:
    triple = thresholds_of(arm)
    return f"{int(triple[3])}/{int(triple[2])}/{int(triple[1])}"


def eval_file(arm: str, window: int, seed: int,
              budget: int = BUDGET) -> pathlib.Path:
    return WORK / "eval" / (f"eval_{str(arm).upper()}_d{int(budget)}_w{int(window)}"
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


# --------------------------------------------------------------------------- #
# exact one-sided Fisher (pure Python, no scipy)
# --------------------------------------------------------------------------- #
def fisher_exact_greater(a: int, b: int, c: int, d: int) -> dict[str, Any]:
    """One-sided Fisher exact test for the 2x2 table [[a, b], [c, d]].

    ``a``/``c`` are the survivals of the arm under test and of the reference,
    ``b``/``d`` their failures.  The alternative is ``rate(arm) > rate(ref)``:
    ``p = P(X >= a)`` with ``X ~ Hypergeometric(N = a+b+c+d, K = a+c, n = a+b)``.
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
        "alternative": "arm > ref (one-sided)",
        "p_value": None, "odds_ratio": None, "odds_ratio_haldane": None,
    }
    if n1 == 0 or n2 == 0 or total == 0:
        return out
    upper = min(n1, m)
    if int(a) > upper:
        out["p_value"] = 0.0
    else:
        logs = [_log_comb(m, x) + _log_comb(total - m, n1 - x)
                for x in range(int(a), upper + 1)]
        peak = max(logs)
        numerator = sum(math.exp(value - peak) for value in logs)
        denominator = math.exp(_log_comb(total, n1) - peak)
        out["p_value"] = float(min(1.0, numerator / denominator))
    if int(b) * int(c) > 0:
        out["odds_ratio"] = float((int(a) * int(d)) / (int(b) * int(c)))
    out["odds_ratio_haldane"] = float(
        ((int(a) + 0.5) * (int(d) + 0.5)) / ((int(b) + 0.5) * (int(c) + 0.5)))
    return out


# --------------------------------------------------------------------------- #
# the arm: the H2s candidate sampler + the threshold-parameterised rule
# --------------------------------------------------------------------------- #
def process_generator(seed: int) -> Any:
    """The **one** generator of a process, created before the window loop.

    One ``torch.Generator`` per (arm, seed) process, consumed once per decision
    across every window and episode — the frozen H2s sampler.
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


def new_audit(arm: str) -> dict[str, Any]:
    return {"cand_counts": [0] * N_CAND, "cand_prob": [],
            "threshold_counts": {str(key): 0 for key in (3, 2, 1, 0)},
            "rule": {"distance_true_bombs_positive": 0,
                     "distance_true_bombs_zero": 0,
                     "distance_false": 0},
            "emit_threshold_px": [], "emit_nearest_px": [],
            "arms_thresholds": {str(key): float(value)
                                for key, value in thresholds_of(arm).items()},
            "margins": [], "best_scores": []}


def rule_decision(state: dict[str, Any], hazards: Any, arm: str) -> dict[str, Any]:
    """The swept rule: ``nearest_all_px < THRESHOLDS[bombs]`` and ``bombs > 0``.

    The distance is v8.0.0's own ``nearest_bullet_distances`` (the measurement
    R5 uses); only the threshold triple comes from this round's arm.
    """

    thresholds = thresholds_of(arm)
    distances = m800().nearest_bullet_distances(state, hazards)
    nearest = distances["nearest_all_px"]
    bombs_before = int(state["bombs_remaining"] or 0)
    threshold = thresholds.get(bombs_before)
    distance_ok = bool(nearest is not None and threshold is not None
                       and float(nearest) < float(threshold))
    fired = bool(distance_ok and bombs_before > 0)
    return {"bomb": 1 if fired else 0, "nearest_all_px": nearest,
            "threshold_px": threshold, "bombs_remaining": bombs_before,
            "distance_ok": distance_ok}


def make_decide(*, arm: str, policy: Any, generator: Any,
                audit: dict[str, Any]) -> Any:
    """``decide(state, hazards, decision) -> (cand, bomb)`` for one arm."""

    import torch

    A = arm_a()

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        feats, _detail = A.build_features(state=state, hazards=hazards,
                                          decision_index=decision)
        with torch.no_grad():
            logits = policy(torch.from_numpy(feats).unsqueeze(0))[0]
        cand, best, margin, prob = candidate_sampled(logits, generator)
        rule = rule_decision(state, hazards, arm)
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
        raise RuleSweepError(f"seed {seed}: checkpoint not found: {path}")
    digest = sha256_file(path)
    expected = V860_SHA256[int(seed)]
    if digest != expected:
        raise RuleSweepError(f"seed {seed}: {path} sha256 {digest[:16]} != "
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
                    arm: str, policy: Any, generator: Any,
                    tag: str) -> dict[str, Any]:
    """One episode through the frozen loop, with the swept rule's audit."""

    audit = new_audit(arm)
    decide = make_decide(arm=arm, policy=policy, generator=generator,
                         audit=audit)
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
        "schema": SCHEMA_EPISODE, "producer": PRODUCER, "arm": arm,
        "arm_thresholds": {str(key): float(value)
                           for key, value in thresholds_of(arm).items()},
        "thresholds_label": thresholds_label(arm),
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
        #    from this arm's threshold triple (not from the policy's bomb head)
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
        # every emitted bomb really was inside this arm's threshold window
        "threshold_respected": bool(
            len(emit_ok) == int(row["bomb_emits"]) and all(emit_ok)),
    }
    if not all(row["checks"].values()):
        row["invalid"] = True
        row["error"] = row["error"] or f"audit check failed: {row['checks']}"
    return row


# --------------------------------------------------------------------------- #
# mode: assert-ckpt
# --------------------------------------------------------------------------- #
def mode_assert(args: argparse.Namespace) -> int:
    out_path = (pathlib.Path(args.out) if args.out
                else WORK / f"ckpt_assertions_{args.phase}.json")
    payload: dict[str, Any] = {
        "schema": SCHEMA_ASSERT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": args.phase,
        "requirement": ("the sweep's candidate head is the frozen v8.6.0 "
                        "`bombdag.pt` Policy(106,64,36) (9c6d5d4a / 36a51090 / "
                        "885dde70); its behaviour policy is the frozen v4.5.0 `S` "
                        "student (round2.pt = 4abee069 / a93746bb / c69960fc); "
                        "the engine and the protected sources must not have "
                        "moved"),
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
    tag = args.tag or "v8k0-fidelity"
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
# mode: smoke (one episode: S1 / seed0 / w11 / 900 decisions)
# --------------------------------------------------------------------------- #
def mode_smoke(args: argparse.Namespace) -> int:
    seed = int(args.policy_seed)
    window = int(str(args.windows).split(",")[0])
    budget = int(args.decisions)
    arm = str(args.arm).upper()
    tag = args.tag or f"v8k0-smoke-{arm}-s{seed}-w{window}"
    out_path = pathlib.Path(args.out) if args.out else WORK / "smoke.json"
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    payload: dict[str, Any] = {
        "schema": SCHEMA_SMOKE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": (f"one episode of {arm} "
                        f"({thresholds_label(arm)} px, seed{seed}/w{window}, "
                        f"{budget} decisions) with the sampler/rule audit, before "
                        f"the full 240-episode sweep"),
        "arm": arm, "arm_thresholds": {str(k): float(v)
                                       for k, v in thresholds_of(arm).items()},
        "thresholds_label": thresholds_label(arm),
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
                              budget=budget, arm=arm, policy=policy,
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
        log(f"SMOKE {arm} ({thresholds_label(arm)}) s{seed} w{window} ep0: "
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
# mode: eval (one arm x one seed: 4 windows x 20 episodes)
# --------------------------------------------------------------------------- #
def mode_eval(args: argparse.Namespace) -> int:
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    episodes = int(args.episodes)
    arm = str(args.arm).upper()
    if arm not in SWEEP_ARMS:
        raise RuleSweepError(f"--arm must be one of {SWEEP_ARMS} "
                             f"(the control {CONTROL_ARM} is the published "
                             f"634/3,600 and is never re-run), got {arm}")
    if seed not in SEEDS:
        raise RuleSweepError(f"--policy-seed must be one of {SEEDS}, got {seed}")
    if budget != BUDGET:
        raise RuleSweepError(f"this round fixes --decisions to {BUDGET}, "
                             f"got {budget}")
    if episodes != EPISODES:
        raise RuleSweepError(f"this round fixes --episodes to {EPISODES} per "
                             f"window, got {episodes}")
    windows = tuple(int(part) for part in str(args.windows).split(",")
                    if part.strip())
    for window in windows:
        if window not in WINDOWS:
            raise RuleSweepError(f"window {window} is not one of {WINDOWS}")
    if not str(args.tag or "").strip():
        raise RuleSweepError(
            "--tag is mandatory: v8.17.0's 18-way launch shared one timestamp "
            "data dir and corrupted 10 of 18 cells; every process needs its "
            "own tag")
    tag = str(args.tag).strip()
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    run_dir = WORK / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = WORK / "data" / tag
    progress_path = WORK / "eval" / f"{arm}_d{budget}_s{seed}_progress.json"
    progress: dict[str, Any] = {
        "schema": SCHEMA_SEED, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": arm, "arm_thresholds": {str(k): float(v)
                                       for k, v in thresholds_of(arm).items()},
        "thresholds_label": thresholds_label(arm),
        "policy_seed": seed, "budget_decisions": budget,
        "episodes_per_window": episodes, "windows": list(windows), "tag": tag,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest},
        "generator": {"seed": generator_seed(seed),
                      "kind": ("one torch.Generator per (arm, seed) process, "
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
            out_path = eval_file(arm, window, seed, budget)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data_dir, data_info = v450().prepare_data_dir(data_root, window,
                                                          args.data_source)
            if not data_info["cfg"]["muted_four_zero"]:
                raise RuleSweepError(f"w{window}: the generated th08.cfg is not "
                                     f"muted: {data_info['cfg']}")
            log(f"{arm} ({thresholds_label(arm)}) s{seed} w{window}: cfg muted="
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
                                              seed=seed, budget=budget, arm=arm,
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
                            raise RuleSweepError(
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
    log(f"eval {arm} ({thresholds_label(arm)}) s{seed}: "
        f"status={progress['status']} cells="
        f"{ {k: (v['survived'], v['episodes']) for k, v in progress['cells'].items()} } "
        f"wall={progress['wall_s']}s")
    return 0 if progress["status"] == "complete" else 4


# --------------------------------------------------------------------------- #
# mode: summary (the sweep's answer)
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
        "mean": round(statistics.fmean(used), 4) if used else None,
        "median": round(statistics.median(used), 4) if used else None,
        "stdev": round(statistics.stdev(used), 4) if len(used) > 1 else None,
        "bombs_used_total": sum(used),
        "episodes_exhausted": sum(1 for row in rows
                                  if int(row["bombs_end"] or 0) == 0),
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


def _arm_block(arm: str, eval_dir: pathlib.Path) -> dict[str, Any]:
    expected = len(SEEDS) * len(WINDOWS) * EPISODES
    rows: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    for seed in SEEDS:
        for window in WINDOWS:
            path = eval_dir / (f"eval_{arm}_d{BUDGET}_w{window}"
                               f"_seed{seed}.jsonl")
            cell_rows = read_jsonl(path)
            files[f"w{window}_seed{seed}"] = {
                "path": str(path), "exists": path.is_file(),
                "rows": len(cell_rows),
                "full": sum(1 for row in cell_rows if row.get("survived")),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None}
            rows.extend(cell_rows)
    per_window: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    for window in WINDOWS:
        window_rows = [row for row in rows if int(row["window"]) == int(window)]
        if not window_rows:
            continue
        survived = sum(1 for row in window_rows if row["survived"])
        ref_full, ref_n = R5_REFERENCE["per_window"][int(window)]
        per_window[str(window)] = {
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
            "r5_reference": {"full": int(ref_full), "n": int(ref_n),
                             "rate": round(ref_full / ref_n, 6)},
            "fisher_vs_r5": fisher_exact_greater(
                survived, len(window_rows) - survived, int(ref_full),
                int(ref_n) - int(ref_full)),
            "arm_minus_r5_pp": round(
                100.0 * (survived / len(window_rows) - ref_full / ref_n), 4),
        }
    for seed in SEEDS:
        seed_rows = [row for row in rows if int(row["policy_seed"]) == int(seed)]
        if not seed_rows:
            continue
        survived = sum(1 for row in seed_rows if row["survived"])
        ref_full, ref_n = R5_REFERENCE["per_seed"][int(seed)]
        per_seed[str(seed)] = {
            "n": len(seed_rows), "full": survived,
            "rate": round(survived / len(seed_rows), 6),
            "cp95": clopper_pearson(survived, len(seed_rows)),
            "r5_reference": {"full": int(ref_full), "n": int(ref_n),
                             "rate": round(ref_full / ref_n, 6)}}
    survived_all = sum(1 for row in rows if row["survived"])
    ref_full, ref_n = R5_REFERENCE["pooled"]
    pooled_fisher = (fisher_exact_greater(survived_all, len(rows) - survived_all,
                                         int(ref_full), int(ref_n) - int(ref_full))
                     if rows else None)
    return {
        "arm": arm, "thresholds": {str(k): float(v)
                                   for k, v in thresholds_of(arm).items()},
        "thresholds_label": thresholds_label(arm),
        "files": files, "rows": len(rows), "expected_rows": expected,
        "complete": bool(len(rows) == expected),
        "per_window_table": [per_window[str(window)] for window in WINDOWS
                             if str(window) in per_window],
        "per_seed": per_seed,
        "pooled": {
            "n": len(rows), "full": survived_all,
            "rate": round(survived_all / len(rows), 6) if rows else None,
            "cp95": clopper_pearson(survived_all, len(rows)),
            "r5_reference": {"full": int(ref_full), "n": int(ref_n),
                             "rate": round(ref_full / ref_n, 6)},
            "fisher_vs_r5": pooled_fisher,
            "arm_minus_r5_pp": (round(100.0 * (survived_all / len(rows)
                                               - ref_full / ref_n), 4)
                                if rows else None),
        },
        "bomb_usage": _bomb_block(rows),
        "invalid_rows": sum(1 for row in rows if row.get("invalid")),
        "wall_s_total": round(sum(float(row["wall_s"] or 0) for row in rows), 3),
    }


def mode_summary(args: argparse.Namespace) -> int:
    eval_dir = pathlib.Path(args.eval_dir) if args.eval_dir else WORK / "eval"
    out_path = pathlib.Path(args.out) if args.out else WORK / "summary.json"
    arms = tuple(str(part).upper() for part in str(args.arms).split(",")
                 if part.strip())
    blocks: dict[str, Any] = {}
    for arm in arms:
        blocks[arm] = _arm_block(arm, eval_dir)
    # -- the verdict (prereg §2) -------------------------------------------
    p1_hits: list[str] = []
    p2_hits: list[str] = []
    significant: list[str] = []
    detail: dict[str, Any] = {}
    for arm, block in blocks.items():
        pooled = block["pooled"]
        fisher = pooled["fisher_vs_r5"] or {}
        p_value = fisher.get("p_value")
        complete = block["complete"]
        p1 = bool(complete and pooled["rate"] is not None
                  and pooled["rate"] >= P1_THRESHOLD)
        p2 = bool(complete and p_value is not None and p_value < P2_ALPHA)
        significant_any = bool(p2)
        if p1:
            p1_hits.append(arm)
        if p2:
            p2_hits.append(arm)
        if significant_any:
            significant.append(arm)
        detail[arm] = {
            "P1": {"requirement": f"pooled 900-decision rate >= {P1_THRESHOLD}",
                   "observed": pooled["rate"],
                   "cp95": pooled["cp95"],
                   "passed": p1},
            "P2": {"requirement": ("significantly better than the published R5 "
                                   "634/3,600 = 17.6111 % (one-sided Fisher, "
                                   "p < 0.05)"),
                   "p_value": p_value,
                   "difference_pp": fisher.get("difference_pp"),
                   "passed": p2},
            "complete": complete, "rows": block["rows"],
            "bomb_usage_mean": block["bomb_usage"]["mean"],
        }
    every_arm_complete = all(block["complete"] for block in blocks.values())
    rates = [block["pooled"]["rate"] for block in blocks.values()
             if block["pooled"]["rate"] is not None]
    best_arm = (max(blocks, key=lambda arm: blocks[arm]["pooled"]["rate"] or -1)
                if blocks else None)
    r5_rate = R5_REFERENCE["pooled"][0] / R5_REFERENCE["pooled"][1]
    s_flat = bool(rates and all(rate <= r5_rate for rate in rates)
                  and not significant)

    #: the prereg's monotonicity check runs late -> early: S3 (40/30/20),
    #: S1 (60/40/20), R0 (80/60/40, the published control), S2 (100/80/60),
    #: S4 (120/90/60).  Only the arms actually measured enter the sequence;
    #: R0 is read from the published cell, never re-run.
    order = ("S3", "S1", "R0", "S2", "S4")
    rate_of: dict[str, float | None] = {
        arm: block["pooled"]["rate"] for arm, block in blocks.items()}
    rate_of[CONTROL_ARM] = round(r5_rate, 6)
    sequence = [(arm, rate_of[arm]) for arm in order
                if rate_of.get(arm) is not None]
    monotone: bool | None = None
    if len(sequence) >= 3:
        monotone = bool(all(b >= a for (_left, a), (_right, b)
                            in zip(sequence, sequence[1:])))
    inversions = [{"from": sequence[index][0], "to": sequence[index + 1][0],
                   "rate_from": sequence[index][1],
                   "rate_to": sequence[index + 1][1]}
                  for index in range(len(sequence) - 1)
                  if sequence[index + 1][1] < sequence[index][1]]

    if not every_arm_complete:
        stop_condition = "incomplete"
    elif p1_hits and p2_hits:
        stop_condition = "S-win"
    elif s_flat:
        stop_condition = "S-flat"
    elif p2_hits:
        stop_condition = ("S-win* (P2 without P1: an arm beats R5 but stays "
                          f"below {P1_THRESHOLD})")
    elif monotone is False:
        stop_condition = "S-nonmono"
    else:
        stop_condition = "S-none"
    payload: dict[str, Any] = {
        "schema": SCHEMA_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": {"path": PREREG_REL,
                   "sha256": (sha256_file(REPO / PREREG_REL)
                              if (REPO / PREREG_REL).is_file() else None)},
        "question": ("does moving the R5 bomb-rule threshold triple raise the "
                     "pure-rule 30 s rate above 634/3,600 = 17.6111 %?"),
        "protocol": {
            "candidate_head": ("v8.6.0 bombdag.pt Policy(106,64,36), frozen; "
                               "score[c] = max(logit[2c], logit[2c+1]); "
                               "cand ~ softmax(score / 1.0), one draw per "
                               "decision from torch.Generator(seed*1000+7)"),
            "bomb": ("nearest_all_px < THRESHOLDS[bombs_remaining] and "
                     "bombs_remaining > 0"),
            "seeds": list(SEEDS), "windows": list(WINDOWS),
            "episodes_per_window": EPISODES,
            "budget_decisions": BUDGET,
            "decision_frames": DECISION_FRAMES,
            "prefix_frame": PREFIX_FRAMES,
            "frames_per_episode": BUDGET * DECISION_FRAMES,
            "loop": ("for seed in (0,1,2): for window in (11,12,17,18): for "
                     f"episode in range({EPISODES}): run_one_episode("
                     f"budget_decisions={BUDGET})"),
            "episode_loop": "tools/rl/v830_bombpol.py::run_episode36 (imported)",
            "arms": {arm: {"thresholds": {str(k): float(v)
                                          for k, v in thresholds_of(arm).items()},
                           "label": thresholds_label(arm)}
                     for arm in arms},
        },
        "control": {
            "arm": CONTROL_ARM, "thresholds": THRESHOLD_ARMS[CONTROL_ARM],
            "rerun": False, "source": R5_REFERENCE["source"],
            "pooled": {"full": int(R5_REFERENCE["pooled"][0]),
                       "n": int(R5_REFERENCE["pooled"][1]),
                       "rate": round(R5_REFERENCE["pooled"][0]
                                     / R5_REFERENCE["pooled"][1], 6)},
            "per_window": {str(k): {"full": int(v[0]), "n": int(v[1]),
                                    "rate": round(v[0] / v[1], 6)}
                           for k, v in R5_REFERENCE["per_window"].items()},
            "per_seed": {str(k): {"full": int(v[0]), "n": int(v[1]),
                                  "rate": round(v[0] / v[1], 6)}
                         for k, v in R5_REFERENCE["per_seed"].items()},
        },
        "arms": blocks,
        "table": [{
            "arm": arm,
            "thresholds": thresholds_label(arm),
            "w11": next((bucket["rate"] for bucket in block["per_window_table"]
                         if int(bucket["window"]) == 11), None),
            "w12": next((bucket["rate"] for bucket in block["per_window_table"]
                         if int(bucket["window"]) == 12), None),
            "w17": next((bucket["rate"] for bucket in block["per_window_table"]
                         if int(bucket["window"]) == 17), None),
            "w18": next((bucket["rate"] for bucket in block["per_window_table"]
                         if int(bucket["window"]) == 18), None),
            "pooled_full": block["pooled"]["full"],
            "pooled_n": block["pooled"]["n"],
            "pooled_rate": block["pooled"]["rate"],
            "cp95_low": block["pooled"]["cp95"]["low"],
            "cp95_high": block["pooled"]["cp95"]["high"],
            "bombs_mean": block["bomb_usage"]["mean"],
            "p_value_vs_r5": (block["pooled"]["fisher_vs_r5"] or {}).get(
                "p_value"),
        } for arm, block in blocks.items()],
        "verdict": {
            "P1": {"requirement": f"any arm >= {P1_THRESHOLD} (R5 + 6.4 pp)",
                   "hits": p1_hits, "passed": bool(p1_hits)},
            "P2": {"requirement": "that arm vs R5, one-sided Fisher p < 0.05",
                   "hits": p2_hits, "passed": bool(p2_hits)},
            "P3": {"requirement": "bomb-usage distribution reported "
                                  "(descriptive)",
                   "bomb_usage_mean": {arm: blocks[arm]["bomb_usage"]["mean"]
                                       for arm in blocks}},
            "P4": {"requirement": "per-window table reported (descriptive)"},
            "S-flat": {"condition": ("all arms <= 17.6111 % and none "
                                     "significant"),
                       "observed_rates": {arm: blocks[arm]["pooled"]["rate"]
                                          for arm in blocks},
                       "held": s_flat},
            "S-nonmono": {"order_late_to_early": list(order),
                          "sequence": [{"arm": arm, "rate": rate}
                                       for arm, rate in sequence],
                          "monotone": monotone,
                          "inversions": inversions,
                          "note": ("threshold vs rate is checked on the late -> "
                                   "early order S3, S1, R0 (published), S2, S4")},
            "best_arm": best_arm,
            "best_rate": (blocks[best_arm]["pooled"]["rate"]
                          if best_arm else None),
            "stop_condition": stop_condition,
            "per_arm": detail,
        },
        "invalid_rows": sum(block["invalid_rows"] for block in blocks.values()),
        "notes": [
            "No training and no collection: this file only evaluates.",
            "The candidate head is frozen; only the bomb-rule thresholds move.",
            "R0 (80/60/40) is the published 634/3,600 and is not re-run.",
            "The 12 (seed, window) cells are not independent samples; the "
            "pooled rate carries its exact (Clopper-Pearson) interval.",
            "Arms share the generator seed with the H2s round, so an arm is "
            "paired with H2s/R5 only up to the first divergent bomb decision.",
        ],
    }
    write_json(out_path, payload)
    log(f"summary: arms={list(blocks)} -> {out_path}")
    for arm, block in blocks.items():
        pooled = block["pooled"]
        fisher = pooled["fisher_vs_r5"] or {}
        log(f"  {arm} ({thresholds_label(arm)}): pooled {pooled['full']}/"
            f"{pooled['n']} = {pooled['rate']} CP95="
            f"[{pooled['cp95']['low']}, {pooled['cp95']['high']}] "
            f"p_vs_R5={fisher.get('p_value')} "
            f"bombs_mean={block['bomb_usage']['mean']} "
            f"complete={block['complete']}")
        for bucket in block["per_window_table"]:
            log(f"    w{bucket['window']}: {bucket['full']}/{bucket['n']} = "
                f"{bucket['rate']} (R5 {bucket['r5_reference']['rate']}, "
                f"p={bucket['fisher_vs_r5'].get('p_value')}, "
                f"bombs_mean={bucket['bombs_used_mean']})")
    log(f"  verdict: {stop_condition} P1={bool(p1_hits)} P2={bool(p2_hits)}")
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
    needle = ".tmp_v8k0_rulesweep30"
    proc = subprocess.run(["pgrep", "-af", "th08-modern"],
                          capture_output=True, text=True)
    all_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    # exclude this very pgrep line and the python driver lines: only the engine
    # binaries whose --data-dir points into this round's tree count as residue
    engines = [line for line in all_lines if "th08-modern" in line
               and "pgrep" not in line]
    mine = [line for line in engines if needle in line]
    payload: dict[str, Any] = {
        "schema": SCHEMA_PROC, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("this round's engine processes must be 0 when it ends; "
                        "other rounds' processes are counted but never touched"),
        "needle": needle, "pgrep_returncode": proc.returncode,
        "engine_processes_total": len(engines),
        "other_round_processes": len(engines) - len(mine),
        "matching": len(mine), "lines": mine}
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
    arms = tuple(str(part).upper() for part in str(args.arms).split(",")
                 if part.strip())
    copied: list[dict[str, Any]] = []
    sources: list[pathlib.Path] = []
    for arm in arms:
        for seed in SEEDS:
            for window in WINDOWS:
                sources.append(eval_file(arm, window, seed, BUDGET))
    sources.extend([WORK / "summary.json", WORK / "d3_gate.json",
                    WORK / "ckpt_assertions_start.json",
                    WORK / "ckpt_assertions_final.json",
                    WORK / "audio_audit.json", WORK / "process_residue.json",
                    WORK / "smoke.json", WORK / "commands.tsv",
                    WORK / "eval_exits.tsv", WORK / "preflight.txt",
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
        description=("v8.19.0 RULESWEEP30 — the R5 bomb-rule threshold triple "
                     "swept at 900 decisions, 3 seeds x 4 windows x 20 "
                     "episodes per arm"))
    parser.add_argument("--mode", required=True,
                        choices=("assert-ckpt", "d3", "smoke", "eval", "summary",
                                 "audio-audit", "process-audit", "publish"))
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--arm", default="S1")
    parser.add_argument("--arms", default=",".join(SWEEP_ARMS))
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--windows", default=",".join(str(w) for w in WINDOWS))
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--decisions", type=int, default=BUDGET)
    parser.add_argument("--decision-frames", type=int, default=DECISION_FRAMES)
    parser.add_argument("--phase", default="start", choices=("start", "final"))
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--tag", default="")
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
        raise RuleSweepError(f"unknown mode {args.mode}")
    finally:
        if _LOG_FH is not None:
            _LOG_FH.flush()


if __name__ == "__main__":
    sys.exit(main())
