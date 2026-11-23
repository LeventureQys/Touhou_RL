#!/usr/bin/env python3
"""v8.12.0 HYBRID_H2S — the **H2s** arm measured *per dev window*.

Why this file exists
--------------------
The H2s arm of v8.12.0 (`Document/Update/Alpha/v8.12.0/PREREGISTRATION_HYBRID_001.md`)
was implemented inside `tools/rl/v8c0_hybrid.py`, whose ``--mode eval`` uses the
v4.5.0/v8.6.0 protocol: **300 episodes per window** and the episode loop nested
inside the window loop.  The H2s cell was interrupted before it reached windows
12/17/18, so the numbers that survived are one window only and **cannot answer a
cross-window question**.  Re-running 3,600 episodes is neither affordable nor
necessary to answer it.

This file therefore re-runs **one** arm, with a deliberately minimal protocol:

    for seed in (0, 1, 2):            # torch.Generator seed*1000+7
      for window in (11, 12, 17, 18): # dev windows, PARK frame 8663
        for episode in range(20):     # 20 per widow, not 300
          run_one_episode(budget_decisions=600, decision_frames=2)

That is 3 x 4 x 20 = **240 episodes** (240 x 600 = 144,000 decisions, 288,000
frames) instead of 10,800.  Everything else is the frozen protocol.

H2s, verbatim (the supervising agent's definition)
--------------------------------------------------
* policy: ``.tmp_v860_bombdag/ckpt/seed{seed}/bombdag.pt`` — ``Policy(106, 64,
  36)`` loaded by ``v860_bombdag.load_policy36`` (the same loader v8.6.0 and
  v8c0's H2s use);
* candidate: ``score[c] = max(logit[2c], logit[2c+1])``, then
  ``softmax(score / 1.0)`` and **one sampled draw per decision** from
  ``torch.Generator().manual_seed(seed * 1000 + 7)``;
* bomb: **not** the policy's bomb head.  The external **R5 staged rule**
  (``tools/rl/v800_bomb.py``): ``nearest_all_px < {3: 80, 2: 60, 1: 40}[bombs_remaining]``
  **and** ``bombs_remaining > 0``;
* masks: ``cand_to_mask(cand, bomb)`` on the decision's first frame and
  ``cand_to_mask(cand, 0)`` afterwards (this is inside the imported
  ``v830_bombpol.run_episode36``, not re-implemented here);
* start: PARK frame 8663; 2 frames per decision; 600 decisions = 1,200 frames.

Nothing is re-implemented except this file's loop, its per-episode audit and the
cross-window summary; the engine plumbing is imported from the frozen modules and
the episode loop is ``v830_bombpol.run_episode36``.

Modes
-----
``--mode assert-ckpt``   engine sha256, the three v8.6.0 ``bombdag.pt``
                         (``9c6d5d4a`` / ``36a51090`` / ``885dde70``), the three
                         frozen v4.5.0 ``round2.pt`` (``4abee069`` / ``a93746bb``
                         / ``c69960fc``), the protected sources.  Fail closed.
``--mode d3``            the D3 fidelity gate (5 assertions) through the frozen
                         ``v432_fork_arm.py --mode fidelity``; a non-5/5 gate
                         voids the round.
``--mode smoke``         **one** episode (the first of seed0/w11), then stop:
                         prints ``survived`` / ``bombs_used`` / ``alive_frames``.
``--mode eval``          one ``--policy-seed``: 4 windows x ``--episodes`` (20).
``--mode summary``       cross-window table + pooled rate + Clopper-Pearson 95 %
                         + bomb-usage distribution -> ``summary.json``.
``--mode audio-audit``   AGENTS.md §9.3 on this round's own data dirs.
``--mode process-audit`` this round's own engine processes must be 0.
``--mode publish``       copy the raw artifacts into
                         ``Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/``
                         and write ``manifest.json``.

Reused, never re-implemented (imported by path)
-----------------------------------------------
* ``tools/rl/v450_cf_label.py`` — ``prepare_env`` (dummy audio, no display),
  ``prepare_data_dir``, ``start_parent`` (HELLO + 2,165-checkpoint prefix + PARK),
  ``check_engine``;
* ``tools/rl/v432_fork_arm.py`` — ``build_private_data_dir`` (a freshly generated
  and ``--inspect``ed muted ``th08.cfg`` before **every** engine start) and the
  D3 gate itself;
* ``tools/rl/v830_bombpol.py`` — **``run_episode36``**: the episode loop;
* ``tools/rl/v860_bombdag.py`` — ``load_policy36``;
* ``tools/rl/v800_bomb.py`` — the R5 rule (``rule_gate("staged")`` +
  ``nearest_bullet_distances``);
* ``tools/rl/v431_arm_a.py`` — ``build_features`` (106 dims) and ``Policy``;
* ``touhou_rl/v420/counterfactual._terminal_of`` — the death rule (inside the
  episode loop).

Not in this file
----------------
* No training, no collection, no labels, no other arm (H1/H2/H3), no pixel arm,
  no hyper-parameter search.
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

WORK = REPO / ".tmp_v8d0_h2s"
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.12.0" / "evidence"
           / "HYBRID_H2S")

SCHEMA_ASSERT = "v8d0-h2s-ckpt-assertions/1"
SCHEMA_D3 = "v8d0-h2s-d3-gate/1"
SCHEMA_SMOKE = "v8d0-h2s-smoke/1"
SCHEMA_EPISODE = "v8d0-h2s-episode/1"
SCHEMA_SEED = "v8d0-h2s-seed-progress/1"
SCHEMA_SUMMARY = "v8d0-h2s-summary/1"
SCHEMA_AUDIO = "v8d0-h2s-audio-audit/1"
SCHEMA_PROC = "v8d0-h2s-process-audit/1"
SCHEMA_MANIFEST = "v8d0-h2s-manifest/1"
SCHEMA_PUBLISH = "v8d0-h2s-publish/1"
PRODUCER = ("tools/rl/v8d0_h2s.py (live engine, HarnessClient role=shadow, "
            "episode loop = tools/rl/v830_bombpol.py::run_episode36)")
PREREG_REL = "Document/Update/Alpha/v8.12.0/PREREGISTRATION_HYBRID_001.md"

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- the arm and the protocol ------------------------------------------------
ARM = "H2s"
SEEDS: tuple[int, ...] = (0, 1, 2)
WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
EPISODES = 20
BUDGET = 600
DECISION_FRAMES = 2
PREFIX_FRAMES = 8663
N_CAND = 18
N_ACTIONS = 36
HIDDEN = 64
FEATURE_DIM = 106
TEMPERATURE = 1.0
GEN_CONST = 7
DIST_STAGES = (0, 1, 9, 10, 24, 25, 49, 50, 99, 100, 199, 200, 399, 400, 599)
R5_THRESHOLDS: dict[int, float] = {3: 80.0, 2: 60.0, 1: 40.0}

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
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)
#: the same-window R5 rates this arm is placed beside (v8.4.0 BOMBRULE2-001,
#: 600 decisions, the v4.5.0 `S` behaviour policy with its own stochastic
#: candidate).  Numbers only; this file runs no R5 episode.
R5_REFERENCE: dict[int, float] = {11: 0.811, 12: 0.792, 17: 0.704, 18: 0.777}

_LOG_FH: Any = None
_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class H2sError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v8d0] {message}"
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
        raise H2sError(f"cannot import {path}")
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
# H2s: the sampled candidate + the external R5 bomb
# --------------------------------------------------------------------------- #
def process_generator(seed: int) -> Any:
    """The **one** generator of a seed's process, created before the window loop.

    ``v860_bombdag.mode_eval`` and v8c0's H2s both create it once per process and
    consume it once per decision across every window and episode.  Re-seeding per
    episode (or per window) would turn the 20 episodes of a cell into replays of
    the first one, which is exactly the sampling the arm is about.
    """

    import torch

    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    return generator


def candidate_sampled(logits: Any, generator: Any) -> tuple[int, float, float,
                                                            float]:
    """H2s: sample ``cand ~ softmax_c max(logit[2c], logit[2c+1])`` (T = 1.0).

    One ``torch.multinomial`` draw per decision.  Returns
    ``(cand, best_score, margin_to_second, sampled_probability)``.
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


def r5_decision(state: dict[str, Any], hazards: Any,
                decision: int) -> dict[str, Any]:
    """The R5 bomb bit, computed by v8.0.0's own rule code.

    ``v800_bomb.nearest_bullet_distances`` is the distance measurement v8.0.0's
    loop uses; ``v800_bomb.rule_gate("staged")`` is the only place the 3 -> 80 /
    2 -> 60 / 1 -> 40 px threshold is resolved.  The gate is ``nearest <
    threshold`` **and** ``bombs_remaining > 0``.
    """

    distances = m800().nearest_bullet_distances(state, hazards)
    nearest = distances["nearest_all_px"]
    bombs_before = int(state["bombs_remaining"] or 0)
    gate = m800().rule_gate("staged", decision=int(decision),
                            bombs_before=bombs_before,
                            player_x=float(state.get("player_x") or 0.0))
    threshold = gate["threshold_px"]
    distance_ok = bool(nearest is not None and threshold is not None
                       and float(nearest) < float(threshold))
    fired = bool(distance_ok and bombs_before > 0)
    return {"bomb": 1 if fired else 0, "nearest_all_px": nearest,
            "threshold_px": threshold, "bombs_remaining": bombs_before,
            "distance_ok": distance_ok}


def new_audit() -> dict[str, Any]:
    return {"cand_counts": [0] * N_CAND, "cand_prob": [],
            "threshold_counts": {str(int(v)): 0 for v in R5_THRESHOLDS.values()},
            "threshold_other": 0,
            "r5": {"distance_true_bombs_positive": 0,
                   "distance_true_bombs_zero": 0,
                   "distance_false": 0},
            "margins": [], "best_scores": []}


def make_decide(*, policy: Any, generator: Any, audit: dict[str, Any]) -> Any:
    """``decide(state, hazards, decision) -> (cand, bomb)`` for H2s."""

    import torch

    A = arm_a()

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        feats, _detail = A.build_features(state=state, hazards=hazards,
                                          decision_index=decision)
        with torch.no_grad():
            logits = policy(torch.from_numpy(feats).unsqueeze(0))[0]
        cand, best, margin, prob = candidate_sampled(logits, generator)
        rule = r5_decision(state, hazards, decision)
        audit["cand_counts"][int(cand)] += 1
        audit["cand_prob"].append(float(prob))
        audit["margins"].append(float(margin))
        audit["best_scores"].append(float(best))
        key = str(int(rule["threshold_px"])) if rule["threshold_px"] is not None else None
        if key in audit["threshold_counts"]:
            audit["threshold_counts"][key] += 1
        else:
            audit["threshold_other"] += 1
        if rule["distance_ok"]:
            if int(rule["bombs_remaining"]) > 0:
                audit["r5"]["distance_true_bombs_positive"] += 1
            else:
                audit["r5"]["distance_true_bombs_zero"] += 1
        else:
            audit["r5"]["distance_false"] += 1
        return int(cand), int(rule["bomb"])

    return decide


def load_policy(seed: int) -> tuple[Any, pathlib.Path, str]:
    path = V860_ROOT / f"seed{int(seed)}" / "bombdag.pt"
    if not path.is_file():
        raise H2sError(f"H2s seed {seed}: checkpoint not found: {path}")
    digest = sha256_file(path)
    expected = V860_SHA256[int(seed)]
    if digest != expected:
        raise H2sError(f"H2s seed {seed}: {path} sha256 {digest[:16]} != "
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


def eval_path(window: int, seed: int) -> pathlib.Path:
    return WORK / "eval" / f"eval_H2s_w{int(window)}_seed{int(seed)}.jsonl"


# --------------------------------------------------------------------------- #
# mode: assert-ckpt
# --------------------------------------------------------------------------- #
def mode_assert(args: argparse.Namespace) -> int:
    out_path = (pathlib.Path(args.out) if args.out
                else WORK / f"ckpt_assertions_{args.phase}.json")
    payload: dict[str, Any] = {
        "schema": SCHEMA_ASSERT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": args.phase,
        "requirement": ("the three policies evaluated are the v8.6.0 "
                        "`bombdag.pt` (9c6d5d4a / 36a51090 / 885dde70); their "
                        "behaviour policy is the frozen v4.5.0 `S` student "
                        "(round2.pt = 4abee069 / a93746bb / c69960fc); the engine "
                        "and the protected sources must not have moved"),
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
                             "sha256": sha256_file(prereg) if prereg.is_file() else None}
        payload["all_pass"] = bool(ok)
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
    write_json(out_path, payload)
    log(f"assert-ckpt {args.phase}: all_pass={payload['all_pass']} "
        f"engine={payload['engine'].get('match') if payload['engine'] else None} "
        f"-> {out_path}")
    return 0 if payload["all_pass"] else 2


# --------------------------------------------------------------------------- #
# mode: d3 (the 5-assertion fidelity gate, run through the frozen v432 script)
# --------------------------------------------------------------------------- #
def mode_d3(args: argparse.Namespace) -> int:
    out_path = pathlib.Path(args.out) if args.out else WORK / "d3_gate.json"
    raw_path = WORK / "fidelity_d3.json"
    argv = [sys.executable, str(REPO / "tools" / "rl" / "v432_fork_arm.py"),
            "--mode", "fidelity", "--seeds", "12", "--learner", "frozen",
            "--work-dir", str(WORK / "work"), "--out", str(raw_path),
            "--tag", "v8d0-fidelity"]
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
        "argv": argv, "exit": proc.returncode, "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip()[-4000:], "wall_s": wall,
        "raw": str(raw_path), "raw_sha256": (sha256_file(raw_path)
                                             if raw_path.is_file() else None),
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
# mode: smoke (one episode: the first of seed0/w11)
# --------------------------------------------------------------------------- #
def run_one_episode(*, parent: Any, episode: int, seed: int, budget: int,
                    policy: Any, generator: Any) -> dict[str, Any]:
    """One episode through the frozen loop, with the H2s audit attached."""

    audit = new_audit()
    decide = make_decide(policy=policy, generator=generator, audit=audit)
    record = v830().run_episode36(parent, episode=int(episode),
                                  policy_seed=int(seed), decisions=int(budget),
                                  decide=decide, budget=int(budget),
                                  decision_frames=DECISION_FRAMES)
    probs = audit["cand_prob"] or [0.0]
    row = {
        "schema": SCHEMA_EPISODE, "producer": PRODUCER, "arm": ARM,
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
        "bombs_remaining_at_emit": record["bombs_remaining_at_emit"],
        "nearest_px_at_emit": record["nearest_px_at_emit"],
        "bomb_cands": record["bomb_cands"],
        "bomb_ignored": record["bomb_ignored"],
        "min_nearest_px": record["min_nearest_px"],
        "actions_sampled": record["actions_sampled"],
        "invalid": bool(record["invalid"]), "error": record["error"],
        "wall_s": record["wall_s"], "spawn_ms": record["spawn_ms"],
        "parent_alive": record["parent_alive"],
        # -- the arm's own audit: proves the candidate was SAMPLED and the bomb
        #    came from the R5 rule (not from the policy's bomb head)
        "cand_counts": audit["cand_counts"],
        "distinct_cands": sum(1 for count in audit["cand_counts"] if count),
        "cand_prob": {"mean": round(statistics.fmean(probs), 6),
                      "min": round(min(probs), 6),
                      "max": round(max(probs), 6)},
        "cand_margin": quantiles(audit["margins"]),
        "cand_best_score": quantiles(audit["best_scores"]),
        "threshold_counts": audit["threshold_counts"],
        "threshold_other": audit["threshold_other"],
        "r5_counts": audit["r5"],
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
        "r5_bomb_matches_emits": bool(
            row["r5_counts"]["distance_true_bombs_positive"] == row["bomb_emits"]),
        "counter_bookkeeping_closes": bool(
            int(row["bombs_start"]) - int(row["bombs_end"])
            == int(row["bombs_used"] or 0)),
        "frames_consistent": bool(
            int(row["frames"]) - int(row["decisions_executed"]) * DECISION_FRAMES
            in ((0,) if row["survived"] else (1, 2))),
    }
    if not all(row["checks"].values()):
        row["invalid"] = True
        row["error"] = row["error"] or f"audit check failed: {row['checks']}"
    return row


def mode_smoke(args: argparse.Namespace) -> int:
    seed = int(args.policy_seed)
    window = int(str(args.windows).split(",")[0])
    budget = int(args.decisions)
    out_path = pathlib.Path(args.out) if args.out else WORK / "smoke.json"
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    payload: dict[str, Any] = {
        "schema": SCHEMA_SMOKE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("one episode of H2s (seed0/w11, the first of its 20) with "
                        "the sampler/bomb audit, before the full 240-episode run"),
        "arm": ARM, "policy_seed": seed, "window": window,
        "budget_decisions": budget, "episodes": 1,
        "decision_frames": DECISION_FRAMES,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest,
                 "bytes": ckpt.stat().st_size},
        "row": None, "checks": {}, "all_pass": False, "error": None,
        "wall_s": None, "cfg": None, "gate": None, "data_dir": None,
    }
    parent: Any = None
    started = time.perf_counter()
    try:
        data_root = WORK / "data" / f"h2s-s{seed}"
        data_dir, data_info = v450().prepare_data_dir(data_root, window,
                                                      args.data_source)
        payload["data_dir"] = str(data_dir)
        payload["cfg"] = data_info["cfg"]
        parent, gate = start_parent(slot=0, window=window, engine=args.engine,
                                   data_dir=data_dir,
                                   log_dir=WORK / "runs" / "smoke")
        payload["gate"] = {key: gate[key] for key in (
            "hello_role_shadow", "park_ok", "park_frame", "threads_1", "all_pass")}
        payload["gate_prefix"] = {key: gate["prefix"][key] for key in (
            "hash_checks", "hash_mismatches", "final_frame")}
        generator = process_generator(seed)
        row = run_one_episode(parent=parent, episode=0, seed=seed, budget=budget,
                              policy=policy, generator=generator)
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
        log(f"SMOKE H2s s{seed} w{window} ep0: survived={row['survived']} "
            f"bombs_used={row['bombs_used']} alive_frames={row['alive_frames']} "
            f"frames={row['frames']} decisions={row['decisions_executed']} "
            f"bomb_emits={row['bomb_emits']} "
            f"distinct_cands={row['distinct_cands']} "
            f"checks={row['checks']}")
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
# mode: eval (one seed: 4 dev windows x 20 episodes)
# --------------------------------------------------------------------------- #
def mode_eval(args: argparse.Namespace) -> int:
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    episodes = int(args.episodes)
    if seed not in SEEDS:
        raise H2sError(f"--policy-seed must be one of {SEEDS}, got {seed}")
    if budget != BUDGET:
        raise H2sError(f"this round fixes --decisions to {BUDGET}, got {budget}")
    if episodes != EPISODES:
        raise H2sError(f"this round fixes --episodes to {EPISODES} per window, "
                       f"got {episodes}")
    windows = tuple(int(part) for part in str(args.windows).split(",")
                    if part.strip())
    for window in windows:
        if window not in WINDOWS:
            raise H2sError(f"window {window} is not one of {WINDOWS}")
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    run_dir = WORK / "runs" / f"h2s-s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = WORK / "eval" / f"seed{seed}_windows.json"
    progress: dict[str, Any] = {
        "schema": SCHEMA_SEED, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": ARM, "policy_seed": seed, "budget_decisions": budget,
        "episodes_per_window": episodes, "windows": list(windows),
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest},
        "generator": {"seed": generator_seed(seed),
                      "kind": ("one torch.Generator per seed process, consumed "
                               "once per decision across every window and "
                               "episode")},
        "cells": {}, "status": "running", "error": None, "wall_s": None,
    }
    started = time.perf_counter()
    from touhou_rl.v420 import contracts as C

    generator = process_generator(seed)
    try:
        for window in windows:
            C.assert_seed_allowed(int(window), "train")
            out_path = eval_path(window, seed)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if args.resume and out_path.is_file():
                existing = read_jsonl(out_path)
                if (len(existing) == episodes
                        and all(not row.get("invalid") for row in existing)):
                    progress["cells"][str(window)] = {
                        "window": int(window), "episodes": len(existing),
                        "survived": sum(1 for row in existing if row["survived"]),
                        "resumed": True, "out": str(out_path)}
                    log(f"H2s s{seed} w{window}: resume — {len(existing)} rows "
                        f"already on disk, skipped")
                    continue
            data_dir, data_info = v450().prepare_data_dir(
                WORK / "data" / f"h2s-s{seed}", window, args.data_source)
            log(f"H2s s{seed} w{window}: cfg muted="
                f"{data_info['cfg']['muted_four_zero']} "
                f"sha256={data_info['cfg']['sha256'][:16]}")
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
                                              policy=policy, generator=generator)
                        row["ckpt_sha256"] = digest
                        row["engine_sha256"] = engine.get("sha256")
                        row["cfg_sha256"] = data_info["cfg"]["sha256"]
                        row["cfg_muted_four_zero"] = \
                            data_info["cfg"]["muted_four_zero"]
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        rows.append(row)
                        log(f"  H2s s{seed} w{window} ep{episode}: "
                            f"survived={row['survived']}bombs_used={row['bombs_used']} "
                            f"alive={row['alive_frames']} "
                            f"distinct_cands={row['distinct_cands']} "
                            f"wall={row['wall_s']}s")
                        if row["invalid"]:
                            raise H2sError(
                                f"H2s s{seed} w{window} ep{episode} invalid: "
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
            log(f"H2s s{seed} w{window}: {survived}/{len(rows)} survived "
                f"({progress['cells'][str(window)]['rate']}) wall="
                f"{progress['cells'][str(window)]['wall_s']}s")
        progress["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        progress["error"] = f"{type(exc).__name__}: {exc}"
        progress["status"] = "failed"
        log(f"EVAL FAILED H2s s{seed}: {progress['error']}")
    finally:
        progress["wall_s"] = round(time.perf_counter() - started, 3)
        write_json(progress_path, progress)
    log(f"eval H2s s{seed}: status={progress['status']} cells="
        f"{ {k: (v['survived'], v['episodes']) for k, v in progress['cells'].items()} } "
        f"wall={progress['wall_s']}s")
    return 0 if progress["status"] == "complete" else 4


# --------------------------------------------------------------------------- #
# mode: summary (the cross-window answer)
# --------------------------------------------------------------------------- #
def _decision_histogram(values: Sequence[int]) -> dict[str, int]:
    out: dict[str, int] = {"0 (never)": 0}
    edges = ((1, 9), (10, 24), (25, 49), (50, 99), (100, 199), (200, 399),
             (400, 599))
    out["0 (never)"] = sum(1 for v in values if int(v) == 0)
    for low, high in edges:
        out[f"{low}-{high}"] = sum(1 for v in values if low <= int(v) <= high)
    return out


def mode_summary(args: argparse.Namespace) -> int:
    eval_dir = pathlib.Path(args.eval_dir) if args.eval_dir else WORK / "eval"
    out_path = pathlib.Path(args.out) if args.out else WORK / "summary.json"
    rows: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    for seed in SEEDS:
        for window in WINDOWS:
            path = eval_dir / f"eval_H2s_w{window}_seed{seed}.jsonl"
            cell_rows = read_jsonl(path)
            files[f"w{window}_seed{seed}"] = {
                "path": str(path), "exists": path.is_file(),
                "rows": len(cell_rows),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None}
            rows.extend(cell_rows)
    cells: list[dict[str, Any]] = []
    per_window: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    for window in WINDOWS:
        window_rows = [row for row in rows if int(row["window"]) == int(window)]
        by_seed: dict[str, Any] = {}
        for seed in SEEDS:
            cell_rows = [row for row in window_rows
                         if int(row["policy_seed"]) == int(seed)]
            if not cell_rows:
                continue
            survived = sum(1 for row in cell_rows if row["survived"])
            bombs = [int(row["bombs_used"] or 0) for row in cell_rows]
            signature = {(int(row["alive_frames"] or 0),
                          int(row["bombs_used"] or 0),
                          tuple(int(d) for d in row["bomb_decisions"]))
                         for row in cell_rows}
            cell = {
                "seed": int(seed), "window": int(window), "n": len(cell_rows),
                "full": survived,
                "rate": round(survived / len(cell_rows), 6),
                "cp95": clopper_pearson(survived, len(cell_rows)),
                "alive_frames": quantiles([float(row["alive_frames"] or 0)
                                           for row in cell_rows]),
                "bombs_used": {"mean": round(statistics.fmean(bombs), 4),
                               "total": sum(bombs),
                               "counts": {str(k): sum(1 for v in bombs if v == k)
                                          for k in (0, 1, 2, 3)}},
                "bomb_emits_total": sum(int(row["bomb_emits"]) for row in cell_rows),
                "distinct_trajectories": len(signature),
                "distinct_cands_mean": round(statistics.fmean(
                    float(row["distinct_cands"]) for row in cell_rows), 4),
                "invalid": sum(1 for row in cell_rows if row.get("invalid")),
                "wall_s_total": round(sum(float(row["wall_s"] or 0)
                                          for row in cell_rows), 3),
            }
            cells.append(cell)
            by_seed[str(seed)] = {"n": cell["n"], "full": cell["full"],
                                  "rate": cell["rate"]}
        if window_rows:
            survived = sum(1 for row in window_rows if row["survived"])
            per_window[str(window)] = {
                "window": int(window), "n": len(window_rows),
                "full": survived,
                "rate": round(survived / len(window_rows), 6),
                "cp95": clopper_pearson(survived, len(window_rows)),
                "by_seed": by_seed,
                "r5_reference_same_window": R5_REFERENCE.get(int(window)),
                "r5_minus_h2s": (None if R5_REFERENCE.get(int(window)) is None
                                 else round(R5_REFERENCE[int(window)]
                                            - survived / len(window_rows), 6)),
                "bombs_used_mean": round(statistics.fmean(
                    int(row["bombs_used"] or 0) for row in window_rows), 4),
                "alive_frames": quantiles([float(row["alive_frames"] or 0)
                                           for row in window_rows]),
            }
    for seed in SEEDS:
        seed_rows = [row for row in rows if int(row["policy_seed"]) == int(seed)]
        if not seed_rows:
            continue
        survived = sum(1 for row in seed_rows if row["survived"])
        per_seed[str(seed)] = {"n": len(seed_rows), "full": survived,
                               "rate": round(survived / len(seed_rows), 6)}
    survived_all = sum(1 for row in rows if row["survived"])
    bombs_all = [int(row["bombs_used"] or 0) for row in rows]
    first_bomb = [int(row["first_bomb_decision"]) for row in rows
                  if row.get("first_bomb_decision") is not None]
    bomb_decision_counts: dict[str, int] = {}
    for row in rows:
        for decision in row["bomb_decisions"]:
            key = str(int(decision))
            bomb_decision_counts[key] = bomb_decision_counts.get(key, 0) + 1
    expected = len(SEEDS) * len(WINDOWS) * EPISODES
    payload: dict[str, Any] = {
        "schema": SCHEMA_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": ARM,
        "arm_definition": {
            "policy": "v8.6.0 bombdag.pt, Policy(106, 64, 36)",
            "candidate": ("score[c] = max(logit[2c], logit[2c+1]); "
                          "cand ~ softmax(score / 1.0); one torch.Generator "
                          "(seed*1000+7) draw per decision"),
            "bomb": ("external R5 staged rule: nearest_all_px < "
                     "{3: 80, 2: 60, 1: 40}[bombs_remaining] and "
                     "bombs_remaining > 0"),
            "masks": ("cand_to_mask(cand, bomb) on the decision's first frame, "
                      "cand_to_mask(cand, 0) afterwards"),
        },
        "protocol": {"seeds": list(SEEDS), "windows": list(WINDOWS),
                     "episodes_per_window": EPISODES,
                     "budget_decisions": BUDGET,
                     "decision_frames": DECISION_FRAMES,
                     "prefix_frame": PREFIX_FRAMES,
                     "frames_per_episode": BUDGET * DECISION_FRAMES,
                     "loop": ("for seed in (0,1,2): for window in (11,12,17,18): "
                              "for episode in range(20): run_one_episode("
                              "budget_decisions=600)"),
                     "episode_loop": ("tools/rl/v830_bombpol.py::run_episode36 "
                                      "(imported)")},
        "files": files,
        "cells": cells,
        "per_seed": per_seed,
        "per_window_table": [per_window[str(window)] for window in WINDOWS
                             if str(window) in per_window],
        "pooled": {
            "n": len(rows), "expected_n": expected,
            "complete": bool(len(rows) == expected),
            "full": survived_all,
            "rate": round(survived_all / len(rows), 6) if rows else None,
            "cp95": clopper_pearson(survived_all, len(rows)),
        },
        "bomb_usage": {
            "episodes": len(rows),
            "counts": {str(k): sum(1 for v in bombs_all if v == k)
                       for k in (0, 1, 2, 3)},
            "mean": round(statistics.fmean(bombs_all), 4) if bombs_all else None,
            "median": round(statistics.median(bombs_all), 4) if bombs_all else None,
            "episodes_exhausted": sum(1 for row in rows if row["bombs_end"] == 0),
            "bomb_emits_total": sum(int(row["bomb_emits"]) for row in rows),
            "per_window_mean": {str(window): per_window[str(window)]["bombs_used_mean"]
                                for window in WINDOWS
                                if str(window) in per_window},
            "per_seed_mean": {str(seed): round(statistics.fmean(
                int(row["bombs_used"] or 0) for row in rows
                if int(row["policy_seed"]) == int(seed)), 4)
                for seed in SEEDS
                if any(int(row["policy_seed"]) == int(seed) for row in rows)},
            "first_bomb_decision": quantiles([float(v) for v in first_bomb])
            if first_bomb else {"n": 0},
            "bomb_decisions_per_episode": _decision_histogram(
                [len(row["bomb_decisions"]) for row in rows]),
            "bomb_decision_peak": (max(bomb_decision_counts.items(),
                                       key=lambda item: item[1])
                                   if bomb_decision_counts else None),
        },
        "invalid_rows": sum(1 for row in rows if row.get("invalid")),
        "reference": {
            "r5_same_window_rate": {str(k): v for k, v in R5_REFERENCE.items()},
            "r5_note": ("v8.4.0 R5 staged rule, 600 decisions, the v4.5.0 `S` "
                        "behaviour policy with its own stochastic candidate; "
                        "numbers taken from the v8.12.0 prereg §0 table"),
        },
        "notes": [
            "No training and no collection: this file only evaluates.",
            "H2s's bomb is an external rule, not the policy's bomb head.",
        ],
    }
    write_json(out_path, payload)
    log(f"summary: pooled {survived_all}/{len(rows)} = "
        f"{payload['pooled']['rate']} CP95={payload['pooled']['cp95']}")
    for window in WINDOWS:
        block = per_window.get(str(window))
        if block:
            log(f"  w{window}: {block['full']}/{block['n']} = "
                f"{block['rate']} (R5 {block['r5_reference_same_window']})")
    log(f"bomb_usage: {payload['bomb_usage']['counts']} mean="
        f"{payload['bomb_usage']['mean']} -> {out_path}")
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
    needle = ".tmp_v8d0_h2s"
    proc = subprocess.run(["pgrep", "-af", "th08-modern"],
                          capture_output=True, text=True)
    all_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    mine = [line for line in all_lines if needle in line]
    payload: dict[str, Any] = {
        "schema": SCHEMA_PROC, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("this round's engine processes must be 0 when it ends; "
                        "other rounds' processes are counted but never touched"),
        "needle": needle, "pgrep_returncode": proc.returncode,
        "engine_processes_total": len(all_lines),
        "other_round_processes": len(all_lines) - len(mine),
        "matching": len(mine), "lines": mine}
    payload["all_pass"] = bool(payload["matching"] == 0)
    write_json(out_path, payload)
    log(f"process-audit: {payload['matching']} own engine process(es) alive "
        f"({payload['other_round_processes']} other-round processes exist and "
        f"were not touched) -> {out_path}")
    return 0 if payload["all_pass"] else 6


# --------------------------------------------------------------------------- #
# mode: publish
# --------------------------------------------------------------------------- #
def mode_publish(args: argparse.Namespace) -> int:
    import shutil

    root = pathlib.Path(args.out) if args.out else PUBLISH
    root.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    sources: list[pathlib.Path] = []
    for seed in SEEDS:
        for window in WINDOWS:
            sources.append(eval_path(window, seed))
    sources.extend([WORK / "summary.json", WORK / "d3_gate.json",
                    WORK / "ckpt_assertions_start.json",
                    WORK / "ckpt_assertions_final.json",
                    WORK / "audio_audit.json",
                    WORK / "process_residue.json",
                    WORK / "determinism_check.json",
                    WORK / "commands.tsv",
                    WORK / "eval_exits.tsv",
                    WORK / "smoke.json"])
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
        "publish_dir": str(root), "copied": copied,
        "count": len(copied),
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
        description="v8.12.0 HYBRID_H2S — H2s per dev window (20 episodes each)")
    parser.add_argument("--mode", required=True,
                        choices=("assert-ckpt", "d3", "smoke", "eval", "summary",
                                 "audio-audit", "process-audit", "publish"))
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--windows", default=",".join(str(w) for w in WINDOWS))
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--decisions", type=int, default=BUDGET)
    parser.add_argument("--decision-frames", type=int, default=DECISION_FRAMES)
    parser.add_argument("--phase", default="start", choices=("start", "final"))
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global _LOG_FH
    args = build_args(sys.argv[1:] if argv is None else argv)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.log:
        _LOG_FH = open(args.log, "a", encoding="utf-8")
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    log(f"mode={args.mode} arm={ARM} python={sys.executable} pid={os.getpid()}")
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
        raise H2sError(f"unknown mode {args.mode}")
    finally:
        if _LOG_FH is not None:
            _LOG_FH.flush()


if __name__ == "__main__":
    sys.exit(main())
