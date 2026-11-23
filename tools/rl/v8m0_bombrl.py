#!/usr/bin/env python3
"""v8.21.0 BOMBRL — the candidate head is **frozen**; REINFORCE (terminal return
only) trains the *bomb head* (2 actions: bomb / no bomb).

What this round asks
--------------------
``Document/Update/Alpha/v8.21.0/PREREGISTRATION_BOMBRL_001.md`` freezes **one**
learnable object — ``BombHead(106 -> 64 -> 2)`` — and keeps everything else
exactly where v8.20.0's ``S-unified`` left it:

* **candidate head frozen**: ``.tmp_v860_bombdag/ckpt/seed{0,1,2}/bombdag.pt``
  (``Policy(106, 64, 36)``, loaded by ``v860_bombdag.load_policy36``), read out by
  **argmax** — ``cand = argmax_c max(logit[2c], logit[2c+1])`` — so no candidate
  sampling noise enters the bomb gradient (prereg §1);
* **bomb head**: ``Linear(106, 64) -> Tanh -> Linear(64, 2)``, warm-started from
  the 60/40/20 rule's own decisions with **200 BCE steps** (prereg §1);
* **learning signal**: REINFORCE from the **terminal return only**::

      R = survived(600 decisions) + 0.5 * alive_frames / 1200

  moving-average baseline, entropy regulariser 0.01, Adam 3e-4, **300 episodes x
  600 decisions per seed**, 3 seeds (prereg §1);
* **evaluation**: ``RL`` (softmax sampling of the bomb head) and ``RL-argmax``
  (deterministic bomb head) at 300 / 600 / 900 decisions x 3 seeds x 4 windows x
  300 episodes (prereg §1), loop ``seed -> window -> episode``;
* the episode loop, the masks, the death rule and the engine plumbing are the
  frozen ones (``v830_bombpol.run_episode36`` / ``v450_cf_label``), imported —
  not re-implemented.

Arms
----
``R0``        v8.20.0's published 60/40/20 rule (20 s 98.75 %, 30 s 24.17 %,
              40 s 2.08 %) — **numbers only, never re-run**.
``R0-argmax`` **extra control, not in the prereg** (see RUN.md §deviations): the
              same rule with the frozen candidate head read out by argmax.  The RL
              arms also use argmax candidates, so without this control the P1
              comparison would confound "bomb head learned to save bombs" with
              "the candidate readout changed from sampling to argmax".
``RL``        trained bomb head, softmax sampling.
``RL-argmax`` trained bomb head, argmax.

Modes
-----
``--mode preflight``      engine census before any launch ("> 6 engines -> wait
                          5 min", never ``pkill -f``).
``--mode assert-ckpt``    engine sha256 ``691bfd77``, the three ``bombdag.pt``
                          (``9c6d5d4a`` / ``36a51090`` / ``885dde70``), the three
                          v4.5.0 ``round2.pt`` (``4abee069`` / ``a93746bb`` /
                          ``c69960fc``), the protected sources, this file, the
                          prereg.  Fail closed.
``--mode d3``             the D3 fidelity gate (5 assertions) through the frozen
                          ``v432_fork_arm.py --mode fidelity``; a non-5/5 gate
                          voids the round (``S-fork``).
``--mode pretrain``       collect the 60/40/20 rule's decisions on the 4 dev
                          windows with argmax candidates, 200 balanced BCE steps,
                          save ``bomb_head_seed{s}_bc.pt`` + a clone probe.
``--mode train``          REINFORCE for ``--episodes`` (300) episodes, save
                          ``bomb_head_seed{s}.pt`` + ``train/seed{s}.jsonl``.
``--mode train-smoke``    one training episode; asserts the return is non-zero
                          and every gradient is finite.
``--mode eval``           one arm x one budget x one seed: 4 windows x 300
                          episodes -> ``eval_{arm}_d{budget}_w{w}_seed{s}.jsonl``.
                          ``--arm R0-argmax`` is the extra control (default 20
                          episodes per window, like v8.20.0's cells).
``--mode eval-smoke``     one episode; asserts the configured ``episodes_total``
                          is 300 and the cfg is muted.
``--mode summary``        per-arm x per-budget table, exact Clopper-Pearson 95 %,
                          one-sided Fisher vs R0, bomb-usage distribution, the
                          training curve, P1-P4 and the stop condition.
``--mode audio-audit``    AGENTS.md §9.3 on this round's own data dirs.
``--mode process-audit``  this round's own engine processes must be 0.
``--mode publish``        copy the raw artifacts into
                          ``Document/Update/Alpha/v8.21.0/evidence/BOMBRL/`` and
                          write ``manifest.json``.

The ``--tag`` rule
------------------
Every process **must** pass its own unique ``--tag``: the private data dir is
``<work>/data/<tag>/w<window>`` and the engine log dir is ``<work>/runs/<tag>``.
v8.17.0 lost 10 of 18 cells to one shared data directory; a tag is therefore
mandatory in every mode that starts an engine.

Not in this file
----------------
* No change to ``touhou_rl/**``, ``third_party/**``, any existing runner or any
  existing evidence directory.  The frozen modules are imported (or copied), the
  rule is called through ``v8l0_unified.rule_decision`` so the control is the
  published 60/40/20 behaviour bit for bit.
* No claim about resource *allocation*: this round learns only *when* to bomb.
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

WORK = REPO / ".tmp_v8m0_bombrl"
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.21.0" / "evidence"
           / "BOMBRL")

SCHEMA_PREFLIGHT = "v8m0-bombrl-preflight/1"
SCHEMA_ASSERT = "v8m0-bombrl-ckpt-assertions/1"
SCHEMA_D3 = "v8m0-bombrl-d3-gate/1"
SCHEMA_PRETRAIN = "v8m0-bombrl-pretrain/1"
SCHEMA_TRAIN = "v8m0-bombrl-train/1"
SCHEMA_TRAIN_SMOKE = "v8m0-bombrl-train-smoke/1"
SCHEMA_EVAL = "v8m0-bombrl-eval/1"
SCHEMA_EVAL_EPISODE = "v8m0-bombrl-episode/1"
SCHEMA_EVAL_SMOKE = "v8m0-bombrl-eval-smoke/1"
SCHEMA_SUMMARY = "v8m0-bombrl-summary/1"
SCHEMA_AUDIO = "v8m0-bombrl-audio-audit/1"
SCHEMA_PROC = "v8m0-bombrl-process-audit/1"
SCHEMA_PUBLISH = "v8m0-bombrl-publish/1"
SCHEMA_MANIFEST = "v8m0-bombrl-manifest/1"

PRODUCER = ("tools/rl/v8m0_bombrl.py (live engine, HarnessClient role=shadow, "
            "episode loop = tools/rl/v830_bombpol.py::run_episode36)")
PREREG_REL = "Document/Update/Alpha/v8.21.0/PREREGISTRATION_BOMBRL_001.md"

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- the frozen protocol (prereg §1) -----------------------------------------
SEEDS: tuple[int, ...] = (0, 1, 2)
WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
DECISION_FRAMES = 2
PREFIX_FRAMES = 8663
N_CAND = 18
N_ACTIONS = 36
FEATURE_DIM = 106
BOMB_HIDDEN = 64
N_BOMB = 2
GEN_CONST = 7

TRAIN_EPISODES = 300
TRAIN_DECISIONS = 600
TRAIN_FRAMES = TRAIN_DECISIONS * DECISION_FRAMES          # 1,200 frames = 20 s

EVAL_BUDGETS: tuple[int, ...] = (300, 600, 900)
EVAL_EPISODES = 300
CONTROL_EPISODES = 20

#: ``R = survived + 0.5 * alive_frames / 1200`` (prereg §1)
REWARD_SURVIVED = 1.0
REWARD_ALIVE_WEIGHT = 0.5
ALIVE_FRAMES_NORM = 1200.0

BASELINE_EMA = 0.95          # moving-average baseline (v431_arm_a's 0.95/0.05)
ENTROPY_COEF = 0.01          # prereg §1
LR = 3e-4                    # prereg §1
GRAD_CLIP = 1.0

BC_STEPS = 200               # prereg §1
BC_BATCH = 256
BC_LR = 1e-2                 # not fixed by the prereg; see RUN.md §deviations
#: one rule episode per dev window: with the frozen candidate head read out by
#: argmax the rule's trajectory is deterministic, so extra episodes are byte
#: duplicates of the same 600 decisions (verified: ep1 == ep0 on every window).
BC_EPISODES_PER_WINDOW = 1
BC_POS_FRACTION = 0.5        # balanced batches; see RUN.md §deviations

#: the control's rule: v8.20.0 U-*'s published 60/40/20 (read through v8l0)
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
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)

#: v8.20.0 UNIFIED (``S-unified``), the R0 arm P1/P3 are placed beside.
#: Re-derived from ``.tmp_v8l0_unified/eval/eval_U{budget}_d{budget}_w*_seed*
#: .jsonl`` (12 x 20 rows per budget).
R0_REFERENCE: dict[int, dict[str, Any]] = {
    600: {"pooled": (237, 240), "per_window": {11: (60, 60), 12: (59, 60),
                                               17: (58, 60), 18: (60, 60)},
          "arm": "v8.20.0 U-600 = 60/40/20, sampled candidates",
          "source": ".tmp_v8l0_unified/eval/eval_U600_d600_w*_seed*.jsonl"},
    900: {"pooled": (58, 240), "per_window": {11: (16, 60), 12: (11, 60),
                                              17: (11, 60), 18: (20, 60)},
          "arm": "v8.20.0 U-900 = 60/40/20, sampled candidates",
          "source": ".tmp_v8l0_unified/eval/eval_U900_d900_w*_seed*.jsonl"},
}
P1_THRESHOLD = 58.0 / 240.0   # 24.1667 %, R0's point estimate (prereg §2 P1)
P3_THRESHOLD = 0.95           # prereg §2 P3
P2_MEAN_THRESHOLD = 3.00      # prereg §2 P2

_LOG_FH: Any = None
_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# errors / logging / hashing / module loading
# --------------------------------------------------------------------------- #
class BombRLError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v8m0] {message}"
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


def reward_of(survived: bool, alive_frames: int) -> float:
    """``R = survived(600 decisions) + 0.5 * alive_frames / 1200`` (prereg §1)."""

    return (REWARD_SURVIVED * (1.0 if survived else 0.0)
            + REWARD_ALIVE_WEIGHT * float(alive_frames) / ALIVE_FRAMES_NORM)


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


def progress_path(arm: str, budget: int, seed: int) -> pathlib.Path:
    return WORK / "eval" / f"progress_{arm}_d{int(budget)}_s{int(seed)}.json"


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
    """``BombHead(106 -> 64 -> 2)``: ``Linear -> Tanh -> Linear(64, 2)``."""

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


def features_of(state: dict[str, Any], hazards: Any, decision: int) -> Any:
    import numpy as np

    feats, _detail = arm_a().build_features(state=state, hazards=hazards,
                                            decision_index=int(decision))
    return np.asarray(feats, dtype=np.float32)


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
            "bomb_probs": []}


def make_decide(*, policy: Any, head: Any, source: str, mode: str,
                generator: Any = None, audit: dict[str, Any] | None = None,
                collect: list[tuple[Any, int]] | None = None) -> Any:
    """``decide(state, hazards, decision) -> (cand, bomb)``.

    ``source`` is ``"bomb_head"`` (the learned 2-action head) or ``"rule"`` (the
    published 60/40/20 rule, used by the R0-argmax control only).  ``mode`` is
    ``"sample"`` (softmax sampling, the RL arm) or ``"argmax"``.  ``collect``
    records ``(features, bomb)`` for one REINFORCE/BC update.
    """

    import torch

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        feats = features_of(state, hazards, decision)
        cand, best, margin = candidate_argmax(policy, feats)
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
                         generator=generator, audit=audit, collect=collect)
    record = v830().run_episode36(parent, episode=int(episode),
                                  policy_seed=int(seed), decisions=int(budget),
                                  decide=decide, budget=int(budget),
                                  decision_frames=DECISION_FRAMES)
    probs = [p for p in audit["bomb_probs"] if p is not None] or [0.0]
    emit_ok = [bool(nearest is not None and threshold is not None
                    and float(nearest) < float(threshold))
               for nearest, threshold in zip(audit["emit_nearest_px"],
                                             audit["emit_threshold_px"])]
    row = {
        "schema": SCHEMA_EVAL_EPISODE, "producer": PRODUCER, "tag": tag,
        "arm": ("R0-argmax" if source == "rule" else
                ("RL" if mode == "sample" else "RL-argmax")),
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
    mine = [line for line in engines if ".tmp_v8m0_bombrl" in line]
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
                        "engine and the protected sources must not have moved"),
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
    tag = args.tag or "v8m0-fidelity"
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
        "requirement": ("warm-start BombHead(106->64->2) from the 60/40/20 "
                        "rule's own decisions with 200 BCE steps (prereg §1)"),
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
                           "shape": "Linear(106,64)->Tanh->Linear(64,2)"}
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
        "requirement": ("REINFORCE, terminal return only: R = survived(600) + "
                        "0.5 * alive_frames / 1200; moving-average baseline; "
                        "entropy 0.01; Adam 3e-4; 300 episodes x 600 decisions"),
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
                           "shape": "Linear(106,64)->Tanh->Linear(64,2)"}
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
        checks = {
            "episodes_expected": len(payload["episodes_run"]) == episodes,
            "return_nonzero": bool((first.get("reward") or 0.0) > 0.0),
            "alive_frames_positive": bool((first.get("alive_frames") or 0) > 0),
            "grad_finite": bool(payload["gradients"].get("finite_all")),
            "grad_nonzero": bool((first.get("grad_norm") or 0.0) > 0.0),
            "cfg_muted_four_zero": all(
                value.get("cfg_muted_four_zero") for value in
                payload.get("gates", {}).values()) if payload.get("gates") else False,
            "gate_all_pass": bool(payload.get("gates")) and all(
                value.get("all_pass") for value in payload["gates"].values()),
            "head_saved": payload["head"] is not None,
        }
        payload["checks"] = checks
        payload["all_pass"] = bool(all(checks.values()))
        write_json(out_path, payload)
        log(f"train-smoke: all_pass={payload['all_pass']} checks={checks}")
        ok = payload["all_pass"]
    log(f"train seed{seed}: status={payload['status']} "
        f"wall={payload['wall_s']}s -> {out_path}")
    return 0 if ok else 4


# --------------------------------------------------------------------------- #
# mode: eval (one arm x one budget x one seed: 4 windows x --episodes)
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
    if seed not in SEEDS:
        raise BombRLError(f"--policy-seed must be one of {SEEDS}, got {seed}")
    if budget not in EVAL_BUDGETS:
        raise BombRLError(f"this round fixes --decisions to {EVAL_BUDGETS}, got "
                          f"{budget}")
    if arm in ("RL", "RL-argmax") and episodes != EVAL_EPISODES:
        raise BombRLError(f"prereg §1 fixes {arm} to {EVAL_EPISODES} episodes per "
                          f"window, got {episodes}")
    if arm == "R0-argmax" and episodes < 1:
        raise BombRLError("--episodes must be >= 1")
    windows = tuple(int(part) for part in str(args.windows).split(",")
                    if part.strip())
    for window in windows:
        if window not in WINDOWS:
            raise BombRLError(f"window {window} is not one of {WINDOWS}")
    source = "rule" if arm == "R0-argmax" else "bomb_head"
    mode = ("argmax" if arm in ("R0-argmax", "RL-argmax") else "sample")
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    head = None
    head_info = None
    if source == "bomb_head":
        path = head_ckpt_path(seed)
        head = load_bomb_head(seed, path)
        head_info = {"path": str(path), "sha256": sha256_file(path)}
    generator = None
    if mode == "sample":
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
    progress_path_ = progress_path(arm, budget, seed)
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

    arm = str(args.arm or "RL")
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    window = 11
    tag = str(args.tag or "").strip()
    if not tag:
        raise BombRLError("--tag is mandatory in every mode that starts an engine")
    if arm not in ("RL", "RL-argmax", "R0-argmax"):
        raise BombRLError(f"unknown arm {arm!r}")
    source = "rule" if arm == "R0-argmax" else "bomb_head"
    mode = "argmax" if arm in ("R0-argmax", "RL-argmax") else "sample"
    out_path = pathlib.Path(args.out) if args.out else WORK / "eval_smoke.json"
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_policy(seed)
    head = None
    head_info = None
    if source == "bomb_head":
        path = head_ckpt_path(seed)
        head = load_bomb_head(seed, path)
        head_info = {"path": str(path), "sha256": sha256_file(path)}
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    configured_total = (CONTROL_EPISODES if arm == "R0-argmax"
                        else EVAL_EPISODES)
    payload: dict[str, Any] = {
        "schema": SCHEMA_EVAL_SMOKE, "producer": PRODUCER, "status": "failed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": tag,
        "requirement": ("one episode of the arm + confirmation that the eval "
                        "plan asks for 300 episodes per window"),
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
            "episodes_total_is_300": bool(configured_total == 300),
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
            payload["checks"]["episodes_total_is_300"]
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
def _block(arm: str, budget: int) -> dict[str, Any]:
    expected = len(SEEDS) * len(WINDOWS) * (
        CONTROL_EPISODES if arm == "R0-argmax" else EVAL_EPISODES)
    rows: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    episodes_expected = (CONTROL_EPISODES if arm == "R0-argmax"
                         else EVAL_EPISODES)
    for seed in SEEDS:
        for window in WINDOWS:
            path = eval_path(arm, budget, window, seed)
            cell = read_jsonl(path)
            files[f"w{window}_seed{seed}"] = {
                "name": path.name, "exists": path.is_file(),
                "rows": len(cell),
                "full": sum(1 for row in cell if row.get("survived")),
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
        per_window.append({
            "window": int(window), "n": len(cell_rows), "full": survived,
            "rate": round(survived / len(cell_rows), 6),
            "cp95": clopper_pearson(survived, len(cell_rows)),
            "episodes_expected": episodes_expected,
            "distinct_outcomes": len({
                (bool(row["survived"]), int(row["alive_frames"] or 0),
                 int(row["bombs_used"] or 0), int(row["bomb_emits"] or 0),
                 tuple(int(v) for v in (row.get("bomb_decisions") or [])))
                for row in cell_rows}),
            "distinct_surviving_trajectories": len({
                (int(row["alive_frames"] or 0), int(row["bombs_used"] or 0),
                 tuple(int(v) for v in (row.get("bomb_decisions") or [])))
                for row in cell_rows if row["survived"]}),
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
            per_cell.append({"window": int(window), "policy_seed": int(seed),
                             "n": len(cell_rows), "full": survived,
                             "rate": round(survived / len(cell_rows), 6),
                             "bombs_used_mean": round(statistics.fmean(
                                 [int(row["bombs_used"] or 0)
                                  for row in cell_rows]), 4),
                             "distinct_outcomes": len({
                                 (bool(row["survived"]),
                                  int(row["alive_frames"] or 0),
                                  int(row["bombs_used"] or 0),
                                  int(row["bomb_emits"] or 0),
                                  tuple(int(v) for v in
                                        (row.get("bomb_decisions") or [])))
                                 for row in cell_rows})})
    survived_all = sum(1 for row in rows if row["survived"])
    bomb_usage = _bomb_block(rows)
    distinct_pooled = len({
        (int(row["window"]), bool(row["survived"]),
         int(row["alive_frames"] or 0), int(row["bombs_used"] or 0),
         int(row["bomb_emits"] or 0),
         tuple(int(v) for v in (row.get("bomb_decisions") or [])))
        for row in rows})
    per_cell_distinct = [cell["distinct_outcomes"] for cell in per_cell]
    return {
        "arm": arm, "budget": int(budget), "bomb_mode": (
            "rule" if arm == "R0-argmax" else
            ("sample" if arm == "RL" else "argmax")),
        "files": files, "rows": len(rows), "expected_rows": expected,
        "episodes_per_window": episodes_expected,
        "complete": bool(len(rows) == expected),
        "cells_present": sum(1 for value in files.values() if value["rows"] > 0),
        "verdict_ready": bool(any(value["rows"] > 0 for value in files.values())),
        "sampling_complete": bool(len(rows) == expected),
        "per_window_table": per_window, "per_seed": per_seed,
        "per_cell": per_cell,
        "pooled": {"n": len(rows), "full": survived_all,
                   "rate": round(survived_all / len(rows), 6) if rows else None,
                   "cp95": clopper_pearson(survived_all, len(rows))},
        "bomb_usage": bomb_usage,
        "invalid_rows": sum(1 for row in rows if row.get("invalid")),
        "distinct": {
            "rows": len(rows),
            "distinct_outcomes_pooled": distinct_pooled,
            "distinct_replays_per_outcome": (round(len(rows) / distinct_pooled, 3)
                                             if distinct_pooled else None),
            "per_cell_min": min(per_cell_distinct) if per_cell_distinct else None,
            "per_cell_max": max(per_cell_distinct) if per_cell_distinct else None,
            "per_cell_mean": (round(statistics.fmean(per_cell_distinct), 4)
                              if per_cell_distinct else None),
            "kind": ("deterministic" if distinct_pooled
                     and len(rows) / distinct_pooled > 0.5 * episodes_expected
                     else "stochastic"),
            "note": ("an outcome is distinct on (window, survived, alive_frames, "
                     "bombs_used, bomb_emits, bomb_decisions); a deterministic arm "
                     "replays the same trajectory, so its row count is NOT a "
                     "sample size")},
        "cell_level": {
            "cells": len(per_cell),
            "full_cells": sum(1 for cell in per_cell if cell["full"] == cell["n"]),
            "zero_cells": sum(1 for cell in per_cell if cell["full"] == 0),
            "mean_cell_rate": round(statistics.fmean(
                [cell["rate"] for cell in per_cell]), 6) if per_cell else None,
            "max_distinct_outcomes": max(
                (cell["distinct_outcomes"] for cell in per_cell), default=None),
            "note": ("RL-argmax is deterministic per (seed, window) cell: its "
                     "3,600 rows are copies of 12 rollouts, so the pooled CP95 "
                     "is nominal only and the 12 cells are the effective n")},
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


def mode_summary(args: argparse.Namespace) -> int:
    arms = tuple(part.strip() for part in str(args.arms).split(",")
                 if part.strip())
    for arm in arms:
        if arm not in ("RL", "RL-argmax", "R0-argmax"):
            raise BombRLError(f"unknown arm {arm!r}")
    out_path = pathlib.Path(args.out) if args.out else WORK / "summary.json"
    budgets = tuple(int(part) for part in str(args.budgets).split(",")
                    if part.strip())
    for budget in budgets:
        if budget not in EVAL_BUDGETS:
            raise BombRLError(f"budget {budget} not in {EVAL_BUDGETS}")
    blocks: dict[str, Any] = {}
    for arm in arms:
        for budget in budgets:
            blocks[f"{arm}|{budget}"] = _block(arm, budget)
    curves = {str(seed): _curve(seed) for seed in SEEDS}

    def block(arm: str, budget: int) -> dict[str, Any]:
        return blocks.get(f"{arm}|{budget}", {"rows": 0, "complete": False,
                                             "pooled": {"n": 0, "full": 0,
                                                        "rate": None,
                                                        "cp95": {}},
                                             "bomb_usage": {},
                                             "per_window_table": [],
                                             "cell_level": {}})

    # -- P1: 30 s (900 decisions) above R0's 24.1667 % ----------------------
    rl900 = block("RL", 900)
    r0_full, r0_n = R0_REFERENCE[900]["pooled"]
    p1: dict[str, Any] = {
        "requirement": ("P1 (prereg §2): the RL arm's 900-decision (30 s) pooled "
                        "survival must exceed R0's point estimate 58/240 = "
                        "24.1667 %; one-sided Fisher + exact CP95"),
        "budget": 900, "threshold": round(P1_THRESHOLD, 6),
        "observed": rl900["pooled"]["rate"], "cp95": rl900["pooled"]["cp95"],
        "reference": {"full": r0_full, "n": r0_n,
                      "rate": round(r0_full / r0_n, 6),
                      "arm": R0_REFERENCE[900]["arm"]},
        "difference_pp": None, "fisher_greater": None, "fisher_less": None,
        "passed": False,
    }
    if rl900["rows"]:
        full, n = int(rl900["pooled"]["full"]), int(rl900["pooled"]["n"])
        p1["difference_pp"] = round(100.0 * (full / n - r0_full / r0_n), 4)
        p1["fisher_greater"] = fisher_greater(full, n - full, r0_full,
                                             r0_n - r0_full)
        p1["fisher_less"] = fisher_less(full, n - full, r0_full, r0_n - r0_full)
        p1["passed"] = bool(rl900["verdict_ready"] and full / n > r0_full / r0_n)

    # -- P2: bomb usage (does "1-2 bombs" appear; is the mean < 3.00) -------
    p2: dict[str, Any] = {
        "requirement": ("P2 (prereg §2): does the bomb-usage distribution contain "
                        "1-2-bomb episodes, and is the mean < 3.00"),
        "threshold_mean": P2_MEAN_THRESHOLD, "arms": {}, "passed": False,
        "hits": {},
    }
    for arm in ("RL", "RL-argmax"):
        for budget in budgets:
            cell = block(arm, budget)
            if not cell["rows"]:
                continue
            usage = cell["bomb_usage"]
            key = f"{arm}|{budget}"
            p2["arms"][key] = {
                "counts": usage["counts"], "mean": usage["mean"],
                "median": usage["median"], "one_or_two": usage["episodes_one_or_two"],
                "exhausted": usage["episodes_exhausted"],
                "mean_below_3": bool(usage["mean"] is not None
                                     and usage["mean"] < P2_MEAN_THRESHOLD),
                "has_one_or_two": bool(usage["episodes_one_or_two"] > 0)}
            p2["hits"][key] = bool(p2["arms"][key]["mean_below_3"]
                                   and p2["arms"][key]["has_one_or_two"])
    key900 = "RL|900"
    p2["primary"] = p2["arms"].get(key900)
    p2["passed"] = bool(p2["hits"].get(key900, False))

    # -- P3: 20 s (600 decisions) must stay >= 95 % ------------------------
    rl600 = block("RL", 600)
    p3: dict[str, Any] = {
        "requirement": ("P3 (prereg §2): the RL arm's 600-decision (20 s) pooled "
                        "survival must stay >= 95 % (R0 = 98.75 %)"),
        "budget": 600, "threshold": P3_THRESHOLD,
        "observed": rl600["pooled"]["rate"], "cp95": rl600["pooled"]["cp95"],
        "reference": {"full": R0_REFERENCE[600]["pooled"][0],
                      "n": R0_REFERENCE[600]["pooled"][1],
                      "rate": round(R0_REFERENCE[600]["pooled"][0]
                                    / R0_REFERENCE[600]["pooled"][1], 6),
                      "arm": R0_REFERENCE[600]["arm"]},
        "difference_pp": None, "fisher_less": None, "passed": False,
    }
    if rl600["rows"]:
        full, n = int(rl600["pooled"]["full"]), int(rl600["pooled"]["n"])
        ref_full, ref_n = R0_REFERENCE[600]["pooled"]
        p3["difference_pp"] = round(100.0 * (full / n - ref_full / ref_n), 4)
        p3["fisher_less"] = fisher_less(full, n - full, ref_full, ref_n - ref_full)
        p3["passed"] = bool(rl600["verdict_ready"] and full / n >= P3_THRESHOLD)

    # -- P4: the training curve --------------------------------------------
    all_curves = [curves[str(seed)] for seed in SEEDS if curves[str(seed)]["rows"]]
    pooled_buckets: list[dict[str, Any]] = []
    if all_curves:
        for index in range(max(len(curve["buckets"]) for curve in all_curves)):
            chunk = [curve["buckets"][index] for curve in all_curves
                     if len(curve["buckets"]) > index]
            rows_n = sum(bucket["n"] for bucket in chunk)
            rewards = sum(bucket["reward_mean"] * bucket["n"] for bucket in chunk)
            survived = sum(bucket["survived"] for bucket in chunk)
            pooled_buckets.append({
                "episodes": chunk[0]["episodes"], "n": rows_n,
                "reward_mean": round(rewards / rows_n, 6),
                "survived": survived,
                "survival_rate": round(survived / rows_n, 6),
                "bombs_used_mean": round(statistics.fmean(
                    [bucket["bombs_used_mean"] for bucket in chunk]), 4),
                "alive_frames_mean": round(statistics.fmean(
                    [bucket["alive_frames_mean"] for bucket in chunk]), 4),
                "entropy_mean": round(statistics.fmean(
                    [bucket["entropy_mean"] for bucket in chunk]), 6)})
    n_rows = sum(curve["rows"] for curve in all_curves)
    first_150 = statistics.fmean([float(row["reward"]) for seed in SEEDS
                                  for row in read_jsonl(train_rows_path(seed))[:150]]) \
        if all_curves else None
    last_150 = statistics.fmean([float(row["reward"]) for seed in SEEDS
                                 for row in read_jsonl(train_rows_path(seed))[-150:]]) \
        if all_curves else None
    rises = bool(first_150 is not None and last_150 is not None
                 and last_150 > first_150)
    p4: dict[str, Any] = {
        "requirement": ("P4 (diagnostic, prereg §2): training curve, mean return "
                        "per 50 episodes"),
        "seeds": curves, "pooled_buckets": pooled_buckets,
        "train_rows": n_rows,
        "first_150_reward": round(first_150, 6) if first_150 is not None else None,
        "last_150_reward": round(last_150, 6) if last_150 is not None else None,
        "rise": (round(last_150 - first_150, 6)
                 if (first_150 is not None and last_150 is not None) else None),
        "rises": rises,
        "flat": bool(not rises),
    }

    # -- the stop condition (prereg §2.1) ----------------------------------
    partial = bool(any(not cell["sampling_complete"] for cell in blocks.values()))
    if not all(cell["verdict_ready"] for cell in blocks.values()):
        stop = "incomplete"
        stop_text = ("at least one arm x budget is missing a (seed, window) cell; "
                     "the prereg's stop conditions are not declared on an "
                     "incomplete table")
    elif p3["observed"] is not None and p3["observed"] < P3_THRESHOLD:
        stop = "S-harm"
        stop_text = ("P3 fails (< 95 % at 20 s): the learned bomb head broke the "
                     "20 s tier; the next round adds a KL regulariser to the "
                     "60/40/20 behaviour")
    elif p1["passed"] and p2["passed"]:
        stop = "S-win"
        stop_text = ("P1 and P2 both hold: the terminal return teaches the bomb "
                     "head to save bombs; the next round moves to 900 decisions "
                     "with 1,000 episodes")
    elif p1["passed"] and not p2["passed"]:
        stop = "S-win (partial)"
        stop_text = ("P1 holds but P2 does not: the head survives longer at 30 s "
                     "without showing the 1-2-bomb saving pattern; reported as a "
                     "partial S-win, not as a saving result")
    elif p4["rises"]:
        stop = "S-nosave"
        stop_text = ("P1 fails while the training curve rises: the head learned to "
                     "live longer but still burns all three bombs; the next round "
                     "adds a -0.1 per-bomb penalty")
    else:
        stop = "S-flat"
        stop_text = ("the training curve does not rise (last 150 episodes <= "
                     "first 150): the REINFORCE signal is too weak at Adam 3e-4 "
                     "over 300 episodes; the next round moves to offline "
                     "return-weighted labels")
    if partial and stop != "incomplete":
        stop_text = (f"[PARTIAL SAMPLING: at least one (seed, window) cell holds "
                     f"fewer than {EVAL_EPISODES} episodes; every verdict below is "
                     f"declared on the rows actually on disk] " + stop_text)

    # -- attribution: how much of the drop is the candidate readout? ---------
    attribution: dict[str, Any] = {
        "question": ("the RL arms use the frozen candidate head read out by "
                     "argmax, while R0's published cells sampled candidates; the "
                     "extra R0-argmax control separates the two changes"),
        "candidate_readout_cost": {}, "learned_head_vs_rule": {},
        "conclusion": None}
    for budget in (600, 900):
        control = block("R0-argmax", budget)
        if not control["rows"]:
            continue
        ref_full, ref_n = R0_REFERENCE[budget]["pooled"]
        c_full, c_n = int(control["pooled"]["full"]), int(control["pooled"]["n"])
        attribution["candidate_readout_cost"][str(budget)] = {
            "R0_published": {"full": ref_full, "n": ref_n,
                             "rate": round(ref_full / ref_n, 6)},
            "R0_argmax_control": {"full": c_full, "n": c_n,
                                  "rate": round(c_full / c_n, 6)},
            "cost_pp": round(100.0 * (c_full / c_n - ref_full / ref_n), 4),
            "fisher_less": fisher_less(c_full, c_n - c_full, ref_full,
                                       ref_n - ref_full)["p_value"]}
    for budget in budgets:
        mine = block("RL", budget)
        control = block("R0-argmax", budget)
        if not mine["rows"] or not control["rows"]:
            continue
        a_full, a_n = int(mine["pooled"]["full"]), int(mine["pooled"]["n"])
        c_full, c_n = int(control["pooled"]["full"]), int(control["pooled"]["n"])
        attribution["learned_head_vs_rule"][str(budget)] = {
            "RL": {"full": a_full, "n": a_n, "rate": round(a_full / a_n, 6),
                   "bombs_mean": (mine["bomb_usage"] or {}).get("mean")},
            "R0_argmax": {"full": c_full, "n": c_n,
                          "rate": round(c_full / c_n, 6),
                          "bombs_mean": (control["bomb_usage"] or {}).get("mean")},
            "difference_pp": round(100.0 * (a_full / a_n - c_full / c_n), 4),
            "fisher_greater": fisher_greater(a_full, a_n - a_full, c_full,
                                             c_n - c_full)["p_value"],
            "fisher_less": fisher_less(a_full, a_n - a_full, c_full,
                                       c_n - c_full)["p_value"]}
    cost = attribution["candidate_readout_cost"]
    if cost:
        attribution["conclusion"] = (
            "with the candidate head frozen and read out by argmax, R0's own rule "
            f"loses {cost.get('600', {}).get('cost_pp')} pp at 600 decisions and "
            f"{cost.get('900', {}).get('cost_pp')} pp at 900 decisions; the learned "
            "bomb head is not worse than the rule under the same readout, so the "
            "loss of P1/P3 must be attributed to the frozen argmax candidate "
            "readout, not to the bomb head")

    table: list[dict[str, Any]] = []
    for arm in arms:
        for budget in budgets:
            cell = block(arm, budget)
            row = {"arm": arm, "budget": budget,
                   "pooled_full": cell["pooled"]["full"],
                   "pooled_n": cell["pooled"]["n"],
                   "pooled_rate": cell["pooled"]["rate"],
                   "cp95_low": cell["pooled"]["cp95"].get("low"),
                   "cp95_high": cell["pooled"]["cp95"].get("high"),
                   "bombs_mean": (cell["bomb_usage"] or {}).get("mean"),
                   "bombs0": (cell["bomb_usage"] or {}).get("counts", {}).get("0"),
                   "bombs1": (cell["bomb_usage"] or {}).get("counts", {}).get("1"),
                   "bombs2": (cell["bomb_usage"] or {}).get("counts", {}).get("2"),
                   "bombs3": (cell["bomb_usage"] or {}).get("counts", {}).get("3"),
                   "bombs_mean": (cell["bomb_usage"] or {}).get("mean"),
                   "distinct": cell.get("distinct", {}).get(
                       "distinct_outcomes_pooled"),
                   "distinct_per_cell_max": cell.get("distinct", {}).get(
                       "per_cell_max"),
                   "replay_factor": cell.get("distinct", {}).get(
                       "distinct_replays_per_outcome"),
                   "deterministic": bool(
                       cell.get("distinct", {}).get("kind") == "deterministic"),
                   "complete": cell["complete"]}
            for bucket in cell["per_window_table"]:
                row[f"w{bucket['window']}"] = f"{bucket['full']}/{bucket['n']}"
            table.append(row)

    payload: dict[str, Any] = {
        "schema": SCHEMA_SUMMARY, "producer": PRODUCER, "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": {"path": PREREG_REL,
                   "sha256": (sha256_file(REPO / PREREG_REL)
                              if (REPO / PREREG_REL).is_file() else None)},
        "question": ("with the candidate head frozen and read out by argmax, can a "
                     "2-action bomb head trained by REINFORCE from the terminal "
                     "return only learn to save bombs (1-2 instead of 3) without "
                     "losing the 20 s tier?"),
        "protocol": {
            "candidate_head": ("v8.6.0 bombdag.pt Policy(106,64,36), frozen, "
                               "argmax: cand = argmax_c max(logit[2c], "
                               "logit[2c+1])"),
            "bomb_head": "Linear(106,64)->Tanh->Linear(64,2)",
            "bomb_head_pretrain": (f"{BC_STEPS} BCE steps, batch {BC_BATCH} "
                                   f"({BC_POS_FRACTION:.0%} positives), Adam "
                                   f"{BC_LR:g}"),
            "learning_signal": ("REINFORCE, R = survived(600) + 0.5 * "
                                "alive_frames/1200, moving-average baseline "
                                f"(ema {BASELINE_EMA}), entropy {ENTROPY_COEF}, "
                                f"Adam {LR:g}, {TRAIN_EPISODES} episodes x "
                                f"{TRAIN_DECISIONS} decisions per seed"),
            "seeds": list(SEEDS), "windows": list(WINDOWS),
            "eval_budgets_decisions": list(EVAL_BUDGETS),
            "eval_episodes_per_window": {arm: (CONTROL_EPISODES
                                               if arm == "R0-argmax"
                                               else EVAL_EPISODES)
                                         for arm in arms},
            "decision_frames": DECISION_FRAMES, "prefix_frame": PREFIX_FRAMES,
            "loop": "for seed in (0,1,2): for window in (11,12,17,18): for " 
                    "episode in range(300)",
            "episode_loop": "tools/rl/v830_bombpol.py::run_episode36 (imported)"},
        "references": {"R0": {str(budget): {"pooled": R0_REFERENCE[budget]["pooled"],
                                            "per_window": R0_REFERENCE[budget][
                                                "per_window"],
                                            "arm": R0_REFERENCE[budget]["arm"],
                                            "source": R0_REFERENCE[budget]["source"]}
                              for budget in R0_REFERENCE},
                       "R0_at_300_decisions": "no published 10 s cell exists"},
        "arms": blocks, "table": table, "train_curve": p4,
        "partial_sampling": bool(partial),
        "attribution": attribution,
        "determinism": {key: cell.get("distinct") for key, cell in blocks.items()},
        "verdict": {"P1": p1, "P2": p2, "P3": p3, "P4": p4,
                    "stop_condition": stop, "stop_text": stop_text,
                    "S-win": {"condition": "P1 and P2 hold",
                              "held": stop == "S-win"},
                    "S-flat": {"condition": "the training curve does not rise",
                               "held": stop == "S-flat"},
                    "S-nosave": {"condition": "P1 fails while the curve rises",
                                 "held": stop == "S-nosave"},
                    "S-harm": {"condition": "P3 < 95 %",
                               "held": stop == "S-harm"},
                    "S-fork": {"condition": "the D3 fidelity gate fails",
                               "held": False}},
        "invalid_rows": sum(cell["invalid_rows"] for cell in blocks.values()),
        "notes": [
            "The candidate head is frozen and read out by argmax, so the RL arms "
            "and R0 differ in TWO ways (bomb policy and candidate readout).  The "
            "extra R0-argmax control isolates the readout.",
            "R0's published cells used sampled candidates, so P1's threshold is "
            "R0's own point estimate, as the prereg fixes it.",
            "The 12 (seed, window) cells are not independent samples; RL-argmax "
            "is deterministic per cell and its 3,600 rows are 12 rollouts.",
            "Training and evaluation use the same 4 dev windows: no cross-window "
            "generalisation claim is made.",
        ],
    }
    write_json(out_path, payload)
    usage_path = WORK / "bomb_usage.json"
    write_json(usage_path, {
        "schema": "v8m0-bombrl-bomb-usage/1", "producer": PRODUCER,
        "created_at": payload["created_at"],
        "arms": {key: cell["bomb_usage"] for key, cell in blocks.items()},
        "P2": p2})
    curve_path = WORK / "train_curve.json"
    write_json(curve_path, {
        "schema": "v8m0-bombrl-train-curve/1", "producer": PRODUCER,
        "created_at": payload["created_at"], "seeds": curves,
        "pooled_buckets": pooled_buckets,
        "first_150_reward": p4["first_150_reward"],
        "last_150_reward": p4["last_150_reward"], "rises": rises})
    log(f"summary: {len(blocks)} arm x budget blocks -> {out_path}")
    for arm in arms:
        for budget in budgets:
            cell = block(arm, budget)
            pooled = cell["pooled"]
            log(f"  {arm} d{budget}: pooled {pooled['full']}/{pooled['n']} = "
                f"{pooled['rate']} CP95=[{pooled['cp95'].get('low')}, "
                f"{pooled['cp95'].get('high')}] bombs_mean="
                f"{(cell['bomb_usage'] or {}).get('mean')} "
                f"complete={cell['complete']}")
    log(f"  P1={p1['passed']} ({p1['observed']}) P2={p2['passed']} "
        f"P3={p3['passed']} ({p3['observed']}) P4_rises={p4['rises']} "
        f"-> {stop}")
    return 0 if payload["invalid_rows"] == 0 else 4


# --------------------------------------------------------------------------- #
# mode: audio-audit / process-audit / publish
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
                else WORK / "process_residue.json")
    census = engine_census()
    payload: dict[str, Any] = {
        "schema": SCHEMA_PROC, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("this round's engine processes must be 0 when it ends; "
                        "other rounds' processes are counted but never touched"),
        "needle": ".tmp_v8m0_bombrl", "pgrep_returncode": census["pgrep_returncode"],
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
    # (source, published name): train/ and pretrain/ share basenames, so the
    # published name carries the directory to avoid a silent overwrite.
    pairs: list[tuple[pathlib.Path, str]] = []
    for arm in ("RL", "RL-argmax", "R0-argmax"):
        for budget in EVAL_BUDGETS:
            for seed in SEEDS:
                for window in WINDOWS:
                    path = eval_path(arm, budget, window, seed)
                    pairs.append((path, path.name))
    for seed in SEEDS:
        head = head_ckpt_path(seed)
        pairs.append((head, head.name))
        bc = bc_ckpt_path(seed)
        pairs.append((bc, bc.name))
        pairs.append((train_rows_path(seed), f"train_rows_seed{seed}.jsonl"))
        pairs.append((WORK / "pretrain" / f"seed{seed}.json",
                      f"pretrain_seed{seed}.json"))
        pairs.append((WORK / "train" / f"seed{seed}.json", f"train_seed{seed}.json"))
    for name in ("summary.json", "train_curve.json", "bomb_usage.json",
                 "d3_gate.json", "ckpt_assertions_start.json",
                 "ckpt_assertions_final.json", "audio_audit.json",
                 "process_residue.json", "train_smoke.json", "eval_smoke.json",
                 "preflight.json", "preflight_start.json", "preflight_final.json",
                 "commands.tsv", "eval_exits.tsv", "RUN.md"):
        pairs.append((WORK / name, name))
    copied: list[dict[str, Any]] = []
    for source, name in pairs:
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
        description=("v8.21.0 BOMBRL — frozen candidate head (argmax) + a 2-action "
                     "bomb head trained by terminal-return REINFORCE"))
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "assert-ckpt", "d3", "pretrain",
                                 "train", "train-smoke", "eval", "eval-smoke",
                                 "summary", "audio-audit", "process-audit",
                                 "publish"))
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--arm", default="RL",
                        choices=("RL", "RL-argmax", "R0-argmax"))
    parser.add_argument("--arms", default="RL,RL-argmax,R0-argmax")
    parser.add_argument("--budgets", default=",".join(str(b) for b in EVAL_BUDGETS))
    parser.add_argument("--decisions", type=int, default=600)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--windows", default=",".join(str(w) for w in WINDOWS))
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--phase", default="start", choices=("start", "final"))
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
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
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
        if args.mode == "summary":
            return mode_summary(args)
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
