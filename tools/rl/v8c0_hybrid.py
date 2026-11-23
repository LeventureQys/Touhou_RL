#!/usr/bin/env python3
"""v8.12.0 HYBRID-001 — **hybrid deployment**: the candidate direction comes from
the learned 36-action policy, the **bomb decision comes from the R5 rule**.

The executor for `Document/Update/Alpha/v8.12.0/PREREGISTRATION_HYBRID_001.md`
(FROZEN 2026-11-22, before any v8.12.0 evaluation).

What this round is
------------------
Five rounds of evidence say two different things at once:

* **"where to dodge" is learnable** — the v8.6.0 candidate head lifts the 20 s
  survival from 16-19 % (v4.5.0 / v8.3.0 / v8.5.0) to ~29 % and the 10 s figure
  to 99.94 %;
* **"when to bomb" is not learnable inside the H = 120 counterfactual-label
  frame** — every learned 36-action policy spends all three bombs (3.00 each),
  while the hand-written **R5 staged rule** reaches 77.11 % at 600 decisions and
  1.36-1.75 bombs per episode.

So there is a deployment plan that needs **no further training**:

    candidate  <- learned policy
    bomb       <- R5 rule (`nearest_all_px < 80/60/40 px` by `bombs_remaining`)

This file evaluates that plan and its two controls.  **Zero training, zero
collection.**  Three arms, one factor — where the bomb bit comes from:

===========  ===========================================  =====================
arm          candidate direction                          bomb decision
===========  ===========================================  =====================
``H1``       v8.6.0 36-action policy (control, = status     **the policy's own
             quo; known ~29 % / 3.00 bombs)                 bomb head** (sampled
                                                            from the 36-way
                                                            softmax)
``H2``       **the same policy's candidate head**:          **external R5 rule**
             ``cand = argmax_c max(logit[2c], logit[2c+1])``  (live ``HAZARDS``
                                                            -> ``nearest_all_px``)
``H3``       v8.3.0 36-action policy                        external R5 rule
===========  ===========================================  =====================

Protocol (prereg §1, unchanged from v4.5.0/v8.0.0/v8.3.0/v8.6.0): the four dev
windows ``(11, 12, 17, 18)``, PARK at frame **8663**, ``--decisions`` in
``{300, 600, 900}`` x ``3 seeds`` x ``300 episodes``, ``stochastic`` =
``softmax(z / 1.0)`` sampled by ``torch.multinomial`` from one
``torch.Generator`` seeded ``seed * 1000 + 7``.

**One property of H2/H3 that the prereg's own wording implies and this file
therefore records explicitly**: with ``argmax`` on the candidate and a
deterministic distance rule on the bomb, an H2/H3 episode has **no random
input** at all, so the 300 episodes of one (seed, window) cell are **replays of
one trajectory**.  Every summary therefore carries
``distinct_trajectories`` next to ``episodes`` so that a 3,600-episode pooled
rate is never mistaken for 3,600 independent samples (prereg §5: "不能把 12 格当
独立样本").

Modes
-----
``--mode preflight``        pins (sha256 of every imported file), the engine
                            gate, the three checkpoints of each arm, the action
                            table and the R5 thresholds.  Fail closed.
``--mode assert-ckpt --phase {start,final}``
                            engine sha256 + the frozen v4.5.0 ``round2.pt``
                            (``4abee069`` / ``a93746bb`` / ``c69960fc``) + the
                            **v8.6.0 ``bombdag.pt``** and **v8.3.0
                            ``round2.pt``** sha256 this round evaluates, plus
                            the protected sources and the prereg itself.
``--mode smoke``            **exactly one episode** of one arm with a
                            per-decision trace (prereg: smoke before any
                            parallel run): prints the bombs_remaining curve and
                            ``survived``.
``--mode eval``             one (arm, budget, seed): 4 dev windows x
                            ``--episodes`` episodes -> ``eval_{arm}_d{b}_seed{k}
                            .jsonl`` + ``.summary.json``.
``--mode crosscheck``       **H1's provenance proof**: run ``--episodes``
                            episodes of the H1 code path on one (seed, budget,
                            window) and compare every core field against the
                            row v8.6.0 recorded for the same episode index.  If
                            they are not bit-identical, H1 may not adopt the
                            v8.6.0 files and must be re-run.
``--mode adopt-h1``         copy the completed v8.6.0 eval grid
                            (``.tmp_v860_bombdag/eval/eval_d{b}_seed{k}.jsonl``)
                            into this round's ``eval/eval_H1_*`` namespace with a
                            provenance record (source sha256, status check,
                            episode recount).  The *closer* reader H1 is the
                            same protocol, the same checkpoints and the same
                            generator rule, so re-running 10,800 episodes to
                            reproduce numbers that already exist would spend the
                            round's budget on nothing.
``--mode diagnostics``      offline per-budget / per-(seed, window) counts,
                            means, medians, quantiles, the **bomb-usage
                            distribution** and the first/last bomb decision;
                            plus the distinct-trajectory count.  Descriptive
                            only — the tests live in ``--mode verdict``.
``--mode verdict``          P1-P5 of prereg §2: Clopper-Pearson 95 % intervals
                            and one-sided exact Fisher tests, computed **both**
                            on the literal 3,600-episode pool and on the
                            distinct-trajectory count (which is what the
                            pseudo-replication argument allows).
``--mode audio-audit``      AGENTS.md §9.3 on this round's **own** data dirs
                            (``v450_cf_label.mode_audit``, own ``--out``).
``--mode process-audit``    this round's own engine processes must be 0.
``--mode manifest --dir D`` sha256 + bytes of every file in D.
``--mode publish``          copy the raw artifacts into
                            ``Document/Update/Alpha/v8.12.0/evidence/HYBRID/``
                            and write the manifest.

Reused, never re-implemented (imported by path, sha256 pinned)
-------------------------------------------------------------
* ``tools/rl/v830_bombpol.py`` — **``run_episode36``** (the one episode loop all
  three arms share: this is what makes "the only difference is where the bomb
  bit comes from" a measured fact instead of a promise), ``split_action``,
  ``action_of``, ``action_masks``, ``policy36``, ``sample_action36``;
* ``tools/rl/v860_bombdag.py`` — ``load_policy36`` (the v8.6.0 checkpoint
  loader) and ``behaviour_ckpt``'s protected-source list;
* ``tools/rl/v800_bomb.py`` — **the R5 rule itself**: ``rule_gate`` with
  ``bomb_rule="staged"`` resolves ``bombs_remaining`` 3/2/1 to 80/60/40 px, and
  ``nearest_bullet_distances`` is the exact distance measurement v8.0.0's loop
  uses.  H2/H3 therefore do not re-type the rule; they call the frozen code.
* ``tools/rl/v450_cf_label.py`` — ``prepare_env`` (dummy audio, no display),
  ``prepare_data_dir``, ``start_parent`` (HELLO + the 2,165-checkpoint prefix +
  PARK), ``mode_audit``, ``check_engine``;
* ``tools/rl/v432_fork_arm.py`` — ``build_private_data_dir`` (muted ``th08.cfg``
  regenerated and ``--inspect``ed before every engine start); the **D3 gate**
  itself is run through ``v432 --mode fidelity`` (a separate command, exactly as
  v8.3.0/v8.6.0 did);
* ``tools/rl/v431_arm_a.py`` — ``build_features`` (106 dims) and ``Policy``;
* ``touhou_rl/v420/counterfactual._terminal_of`` — the death rule.

Failure policy (fail closed)
---------------------------
* a pinned import whose sha256 moved -> exit 2, nothing runs;
* D3 fidelity != 5/5, or a checkpoint whose sha256 moved -> exit 2, the round is
  void (S-fork);
* an invalid episode aborts the cell (rc 4) instead of being dropped;
* H2/H3 whose argmax is not a real maximum, or whose bomb bit disagrees with the
  R5 rule at the same decision, aborts the cell (rc 4).

Not in this file
----------------
* No training, no collection, no checkpoint output, no hyper-parameter search.
* No pixel arm.
* No claim that the policy "learned to bomb": in H2/H3 the bomb is an
  **external rule** (prereg §5).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WORK = REPO / ".tmp_v8c0_hybrid"
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.12.0" / "evidence"
           / "HYBRID")

SCHEMA_PREFLIGHT = "v8c0-hybrid-preflight/1"
SCHEMA_ASSERT = "v8c0-hybrid-ckpt-assertions/1"
SCHEMA_SMOKE = "v8c0-hybrid-smoke/1"
SCHEMA_EVAL = "v8c0-hybrid-eval-episode/1"
SCHEMA_EVAL_SUMMARY = "v8c0-hybrid-eval-summary/1"
SCHEMA_DIAG = "v8c0-hybrid-diagnostics/1"
SCHEMA_BOMB_USAGE = "v8c0-hybrid-bomb-usage/1"
SCHEMA_VERDICT = "v8c0-hybrid-verdict/1"
SCHEMA_AUDIT = "v8c0-hybrid-audio-audit/1"
SCHEMA_PROC = "v8c0-hybrid-process-audit/1"
SCHEMA_MANIFEST = "v8c0-hybrid-manifest/1"
SCHEMA_PUBLISH = "v8c0-hybrid-publish/1"
PRODUCER = ("tools/rl/v8c0_hybrid.py (live engine, HarnessClient role=shadow, "
            "episode loop = tools/rl/v830_bombpol.py::run_episode36)")
PREREG_REL = "Document/Update/Alpha/v8.12.0/PREREGISTRATION_HYBRID_001.md"
PREREG = REPO / PREREG_REL

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- pinned imports: the whole round is void if any of these moved ------------
PINNED: dict[str, str] = {
    "tools/rl/v450_cf_label.py":
        "b7ab0ebeee38df4fcbb5bbbdf5e453edc8e5547abb702473b0c09cb4e90d9947",
    "tools/rl/v432_fork_arm.py":
        "33235506077308549f7657e359381d8c38b1bf45ba4ede0d802448f2dbf0cede",
    "tools/rl/v431_arm_a.py":
        "96923930fed5d1df6ea398e858aa9fb123ed39a455c3ad7e2706230432445503",
    "tools/rl/v800_bomb.py":
        "5d85bcb016a0eebbfb0fd101b2406c1841d8481468e057104b67c57120c0d7bf",
    "tools/rl/v830_bombpol.py":
        "a8c6d3065c968dbd546e9b37f389d3f786404821b8b5fa53e25d520549c58e1c",
    "tools/rl/v860_bombdag.py":
        "3433ae743cc224cae436b6e2192654e30eabe5f05f9053218969e52bdc47cf8a",
    "touhou_rl/v420/contracts.py":
        "1357c69c32a74fda20553a3e6b290ca2102b7cc5c52e47e9fdc6eb9e08c946c1",
    "touhou_rl/v400/gym/actions.py":
        "a67145b2bec7c96b3aacaa2514e78cdadf50b056473d15b4ae8122c19c393eff",
}
PREREG_SHA256 = (
    "c937c1685af536b2d6aa9ac2cf378a9bfdcfb9603c5889516063be5d391636b4")

# -- frozen geometry / split (identical to the v4.5.0 -> v8.6.0 protocol) -----
PREFIX_FRAMES = 8663
DEV_WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
TRAIN_SEEDS: tuple[int, ...] = (0, 1, 2)
N_CAND = 18
N_ACTIONS = 36
BOMB_VALUES: tuple[int, ...] = (0, 1)
DECISION_FRAMES = 2
BUDGETS: tuple[int, ...] = (300, 600, 900)
EPISODES = 300
HIDDEN = 64
FEATURE_DIM = 106
TEMPERATURE = 1.0
GEN_CONST = 7

# -- the three arms (prereg §1: one factor, the source of the bomb bit) ------
ARMS: tuple[str, ...] = ("H1", "H2", "H3", "H2s")
ARM_SPEC: dict[str, dict[str, Any]] = {
    "H1": {"candidate": "v8.6.0 36-action policy, stochastic 36-way sample",
           "bomb": "the policy's own bomb head (sampled)",
           "ckpt_family": "v860", "policy": "v860-bombdag"},
    "H2": {"candidate": "v8.6.0 36-action policy, "
                        "argmax_c max(logit[2c], logit[2c+1])",
           "bomb": "external R5 staged rule (nearest_all_px < 80/60/40 px)",
           "ckpt_family": "v860", "policy": "v860-bombdag"},
    "H3": {"candidate": "v8.3.0 36-action policy, "
                        "argmax_c max(logit[2c], logit[2c+1])",
           "bomb": "external R5 staged rule (nearest_all_px < 80/60/40 px)",
           "ckpt_family": "v830", "policy": "v830-bombpol"},
    # -- added at the supervising agent's direction (see RUN.md "deviations"):
    #    the ONLY arm whose candidate is sampled, i.e. the only arm whose
    #    3,600-episode pool is 3,600 independent rollouts and the only one
    #    directly comparable to R5's 77.11 % (which was measured with a
    #    stochastic policy).
    "H2s": {"candidate": ("v8.6.0 36-action policy, "
                          "softmax_{c} max(logit[2c], logit[2c+1]) sampled "
                          "at temperature 1.0"),
            "bomb": "external R5 staged rule (nearest_all_px < 80/60/40 px)",
            "ckpt_family": "v860", "policy": "v860-bombdag"},
}
STOCHASTIC_ARMS: frozenset[str] = frozenset({"H1", "H2s"})
CKPT_ROOTS: dict[str, pathlib.Path] = {
    "v860": REPO / ".tmp_v860_bombdag" / "ckpt",
    "v830": REPO / ".tmp_v830_bombpol" / "ckpt",
}
CKPT_FILES: dict[str, str] = {"v860": "bombdag.pt", "v830": "round2.pt"}

# -- the frozen v4.5.0 `S` checkpoints (the behaviour policy of v8.3/v8.6) ---
FROZEN_ROUND2_SHA256: dict[int, str] = {
    0: "4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc",
    1: "a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4",
    2: "c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01",
}
V450_CKPT_ROOT = REPO / ".tmp_v450_cf" / "ckpt"

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

# -- the R5 rule as it is frozen in tools/rl/v800_bomb.py --------------------
R5_RULE = "staged"
R5_SOURCE = "tools/rl/v800_bomb.py:STAGED_THRESHOLDS (v8.4.0 BOMBRULE2-001 §1)"

# -- the references this round is placed beside (numbers only, no test) ------
REFERENCE = {
    "rule_r5": {"arm": "v8.4.0 R5 staged rule, v4.5.0 `S` behaviour policy",
                "rate": {300: 1.0, 600: 0.7711, 900: 0.1761},
                "bombs_used": {"min": 1.36, "max": 1.75},
                "source": "prereg §0 table (77.11 % / 1.36-1.75 bombs)"},
    "v830": {"arm": "v8.3.0 36-action policy, its own bomb head",
             "rate": {300: 0.9994, 600: None, 900: 0.0061},
             "rate_note": "99.94 % at 10 s / 0.61 % at 30 s",
             "bombs_used": 3.00,
             "source": "prereg §0 table"},
    "v860": {"arm": "v8.6.0 36-action policy, its own bomb head",
             "rate": {300: None, 600: None, 900: None},
             "rate_note": "~29 % at 20 s (three windows agree), 99.94 % at 10 s",
             "bombs_used": 3.00,
             "source": "prereg §0 table; the v8.6.0 d600/d900 grid is re-measured "
                       "this round as arm H1"},
}

_LOG_FH: Any = None
_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class HybridError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v8c0] {message}"
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
        raise HybridError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v450():
    if "v450" not in _CACHE:
        module = load_module("v450_cf_label", REPO / "tools" / "rl" / "v450_cf_label.py")
        module._LOG_FH = _LOG_FH
        _CACHE["v450"] = module
    return _CACHE["v450"]


def m432():
    if "m432" not in _CACHE:
        module = load_module("v432_fork_arm", REPO / "tools" / "rl" / "v432_fork_arm.py")
        module._LOG_FH = _LOG_FH
        _CACHE["m432"] = module
    return _CACHE["m432"]


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


def write_jsonl_line(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: pathlib.Path) -> dict[str, Any]:
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


def histogram(values: Sequence[int], edges: Sequence[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for edge in edges:
        out[f"<{int(edge)}"] = sum(1 for v in values if int(v) < int(edge))
    out[f">={int(edges[-1])}"] = sum(1 for v in values if int(v) >= int(edges[-1]))
    return out


def decision_histogram(values: Sequence[int]) -> dict[str, int]:
    edges = ((0, 0), (1, 9), (10, 24), (25, 49), (50, 99), (100, 199),
             (200, 399), (400, 799), (800, 10 ** 9))
    out: dict[str, int] = {}
    for low, high in edges:
        label = (str(low) if low == high else
                 (f">={low}" if high >= 10 ** 9 else f"{low}-{high}"))
        out[label] = sum(1 for v in values if low <= int(v) <= high)
    return out


def generator_seed(seed: int) -> int:
    """v4.5.0 §2.3's rule, verbatim: ``torch.Generator().manual_seed(seed*1000+7)``."""

    return int(seed) * 1000 + GEN_CONST


def check_pins() -> dict[str, Any]:
    out: dict[str, Any] = {"pinned": {}, "all_match": True}
    for rel, expected in PINNED.items():
        path = REPO / rel
        got = sha256_file(path) if path.is_file() else None
        match = got == expected
        out["pinned"][rel] = {"sha256": got, "expected": expected, "match": match}
        if not match:
            out["all_match"] = False
    got = sha256_file(PREREG) if PREREG.is_file() else None
    out["prereg"] = {"path": PREREG_REL, "sha256": got, "expected": PREREG_SHA256,
                     "match": got == PREREG_SHA256}
    if not out["prereg"]["match"]:
        out["all_match"] = False
    return out


def ckpt_path(arm: str, seed: int) -> pathlib.Path:
    family = ARM_SPEC[arm]["ckpt_family"]
    return CKPT_ROOTS[family] / f"seed{int(seed)}" / CKPT_FILES[family]


def policy_sha256(arm: str, seed: int) -> dict[str, Any]:
    path = ckpt_path(arm, seed)
    return {"path": str(path), "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None}


def parse_windows(text: str, *, allow_train: bool = False) -> tuple[int, ...]:
    values = tuple(int(part) for part in str(text).split(",") if part.strip())
    if not values:
        return DEV_WINDOWS
    allowed = (DEV_WINDOWS + (13, 14, 15, 16)) if allow_train else DEV_WINDOWS
    for window in values:
        if int(window) not in allowed:
            raise HybridError(f"window {window} is not allowed here "
                              f"(allowed {allowed})")
    return values


# --------------------------------------------------------------------------- #
# the arm's decision rule: the ONLY behavioural difference between the arms
# --------------------------------------------------------------------------- #
def load_arm_policy(arm: str, seed: int) -> tuple[Any, pathlib.Path, str]:
    """v8.6.0's ``load_policy36`` (== ``v830.policy36`` + ``load_state_dict``)."""

    path = ckpt_path(arm, seed)
    if not path.is_file():
        raise HybridError(f"{arm} seed {seed}: checkpoint not found: {path}")
    policy = v860().load_policy36(path, seed=seed)
    return policy, path, sha256_file(path)


def candidate_from_pairs(logits: Any) -> tuple[int, float, float]:
    """prereg §1 H2: ``cand = argmax_c max(logit[2c], logit[2c+1])``.

    ``action = cand * 2 + bomb`` (v8.3.0 §1), so the two logits of candidate ``c``
    sit at ``2c`` (no bomb) and ``2c + 1`` (bomb).  Taking the larger of the two
    asks the policy "how good is the best way to play candidate ``c``", which is
    exactly the question the R5 rule's bomb decision answers separately.

    Returns ``(cand, best_score, margin_to_second)``; the margin is recorded so a
    degenerate head (all candidates equal) is visible in the raw data.
    """

    import torch

    pair = logits.view(N_CAND, 2)
    scores = pair.max(dim=1).values
    cand = int(torch.argmax(scores).item())
    ordered = torch.sort(scores, descending=True).values
    best = float(ordered[0].item())
    second = float(ordered[1].item()) if ordered.numel() > 1 else float("nan")
    return cand, best, round(best - second, 6)


def candidate_sampled(logits: Any, generator: Any) -> tuple[int, float, float,
                                                             float]:
    """H2s: sample the candidate from ``softmax_c max(logit[2c], logit[2c+1])``.

    One ``torch.multinomial`` draw per decision from **one** generator seeded
    ``seed * 1000 + 7`` — the v4.5.0/v8.3.0/v8.6.0 sampling rule — so the arm is
    stochastic and its 3,600-episode pool is 3,600 independent rollouts.  The
    bomb bit is **not** taken from the policy: H2s's caller replaces it with the
    R5 rule's decision, so no logit's bomb head ever reaches the engine.

    Returns ``(cand, best_score, margin_to_second, sampled_probability)``.
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
    """The R5 bombs bit, computed by **v8.0.0's own rule code**.

    ``v800_bomb.nearest_bullet_distances`` is the distance measurement v8.0.0's
    episode loop uses (``nearest_all_px``), and ``v800_bomb.rule_gate`` with
    ``"staged"`` is the only place the 3 -> 80 / 2 -> 60 / 1 -> 40 px threshold is
    resolved.  The gate is v8.4.0's: ``nearest < threshold`` **and**
    ``bombs_remaining > 0``.
    """

    distances = m800().nearest_bullet_distances(state, hazards)
    nearest = distances["nearest_all_px"]
    bombs_before = int(state["bombs_remaining"] or 0)
    gate = m800().rule_gate(R5_RULE, decision=int(decision),
                            bombs_before=bombs_before,
                            player_x=float(state.get("player_x") or 0.0))
    threshold = gate["threshold_px"]
    distance_ok = bool(nearest is not None and threshold is not None
                       and float(nearest) < float(threshold))
    fired = bool(distance_ok and bombs_before > 0)
    return {"bomb": 1 if fired else 0, "nearest_all_px": nearest,
            "threshold_px": threshold, "bombs_remaining": bombs_before,
            "distance_ok": distance_ok, "bullets": distances["bullets"],
            "bullets_within_60": distances["bullets_within_60"]}


def make_decide(arm: str, *, policy: Any, generator: Any, box: dict[str, Any]) -> Any:
    """Return ``decide(state, hazards, decision) -> (cand, bomb)`` for one arm.

    ``generator`` is the **one** ``torch.Generator`` of the (arm, budget, seed)
    process — created and seeded once, outside the window/episode loops, exactly
    as ``v860_bombdag.mode_eval`` does it.  Re-seeding it per episode would make
    every episode of a cell a replay of the first one.

    ``box`` is reset by the caller for every episode; it collects the per-episode
    audit trail (which the shared ``run_episode36`` loop cannot know about).
    """

    import torch

    A = arm_a()
    stochastic = arm in STOCHASTIC_ARMS
    if stochastic and generator is None:
        raise HybridError(f"{arm} needs the (arm, budget, seed) generator")

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        feats, _detail = A.build_features(state=state, hazards=hazards,
                                          decision_index=decision)
        if arm == "H1":
            # exactly v8.6.0's decision: one 36-way stochastic sample
            action = int(v830().sample_action36(policy, feats, "stochastic",
                                               generator))
            cand, bomb = v830().split_action(action)
            box["sampled_action"].append(int(action))
            box["pair_margin"].append(None)
            return int(cand), int(bomb)
        with torch.no_grad():
            logits = policy(torch.from_numpy(feats).unsqueeze(0))[0]
        if arm == "H2s":
            # the supervising agent's H2s: the candidate is SAMPLED (so the
            # 3,600-episode pool is 3,600 independent rollouts) and the bomb
            # still comes only from the R5 rule
            cand, best, margin, prob = candidate_sampled(logits, generator)
            rule = r5_decision(state, hazards, decision)
            box["pair_margin"].append(margin)
            box["cand_best_score"].append(best)
            box["cand_prob"].append(prob)
            box["nearest_all_px"].append(rule["nearest_all_px"])
            box["threshold_px"].append(rule["threshold_px"])
            box["bombs_remaining_at_decision"].append(rule["bombs_remaining"])
            box["distance_ok"].append(rule["distance_ok"])
            return int(cand), int(rule["bomb"])
        cand, best, margin = candidate_from_pairs(logits)
        rule = r5_decision(state, hazards, decision)
        box["pair_margin"].append(margin)
        box["cand_best_score"].append(best)
        box["nearest_all_px"].append(rule["nearest_all_px"])
        box["threshold_px"].append(rule["threshold_px"])
        box["bombs_remaining_at_decision"].append(rule["bombs_remaining"])
        box["distance_ok"].append(rule["distance_ok"])
        return int(cand), int(rule["bomb"])

    return decide


def new_box() -> dict[str, Any]:
    return {"sampled_action": [], "pair_margin": [], "cand_best_score": [],
            "cand_prob": [], "nearest_all_px": [], "threshold_px": [],
            "bombs_remaining_at_decision": [], "distance_ok": []}


def process_generator(arm: str, seed: int) -> Any:
    """The **one** generator of an (arm, budget, seed) process (H1 / H2s).

    ``v860_bombdag.mode_eval`` creates it once, before the window loop, and it is
    consumed once per decision across all 4 windows x 300 episodes.  H1 must do
    the same or its episodes are replays of the first one — which is exactly what
    the first crosscheck run measured before this was fixed.
    """

    if arm not in STOCHASTIC_ARMS:
        return None
    import torch

    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    return generator


# --------------------------------------------------------------------------- #
# engine plumbing (all delegated to the frozen modules)
# --------------------------------------------------------------------------- #
def engine_gate(path: str) -> dict[str, Any]:
    return dict(v450().check_engine(path))


def prepare_env() -> dict[str, Any]:
    return dict(v450().prepare_env())


def start_parent(*, slot: int, window: int, engine: str,
                 data_dir: pathlib.Path, log_dir: pathlib.Path) -> tuple[Any,
                                                                        dict[str, Any]]:
    return v450().start_parent(slot=int(slot), window=int(window), engine=engine,
                               data_dir=data_dir, log_dir=log_dir)


def eval_path(root: pathlib.Path, arm: str, budget: int, seed: int) -> pathlib.Path:
    return root / "eval" / f"eval_{arm}_d{int(budget)}_seed{int(seed)}.jsonl"


# --------------------------------------------------------------------------- #
# mode: preflight
# --------------------------------------------------------------------------- #
def mode_preflight(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    engine = engine_gate(args.engine)
    pins = check_pins()
    rec: dict[str, Any] = {
        "schema": SCHEMA_PREFLIGHT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": {"path": PREREG_REL, "sha256": PREREG_SHA256, "frozen": True,
                   "single_factor": ("where the bomb bit comes from: the policy's "
                                     "own bomb head (H1) or the external R5 rule "
                                     "(H2, H3)")},
        "pins": pins, "engine": engine, "launch_env": prepare_env(),
        "feature_dim": int(arm_a().feature_dim()),
        "arms": {arm: {"candidate": ARM_SPEC[arm]["candidate"],
                       "bomb": ARM_SPEC[arm]["bomb"],
                       "policy": ARM_SPEC[arm]["policy"],
                       "checkpoints": {str(seed): policy_sha256(arm, seed)
                                       for seed in TRAIN_SEEDS}}
                 for arm in ARMS},
        "behaviour": {
            "v450_round2": {str(seed): {
                "path": str(V450_CKPT_ROOT / f"seed{seed}" / "round2.pt"),
                "expected": FROZEN_ROUND2_SHA256[seed],
                "sha256": (sha256_file(V450_CKPT_ROOT / f"seed{seed}" / "round2.pt")
                           if (V450_CKPT_ROOT / f"seed{seed}" / "round2.pt").is_file()
                           else None)}
                for seed in TRAIN_SEEDS},
            "bomb_rule": R5_RULE,
            "thresholds_px": {str(k): v for k, v in
                              m800().STAGED_THRESHOLDS.items()},
            "bomb_rule_source": R5_SOURCE,
            "bomb_rule_gate": "nearest_all_px < threshold and bombs_remaining > 0",
        },
        "action_space": {"candidates": N_CAND, "bombs": list(BOMB_VALUES),
                         "actions": N_ACTIONS,
                         "encoding": "action = cand * 2 + bomb",
                         "bomb1_semantics": ("cand_to_mask(cand, 1) on the "
                                             "decision's first frame, "
                                             "cand_to_mask(cand, 0) afterwards")},
        "frozen": {"DEV_WINDOWS": list(DEV_WINDOWS),
                   "train_seeds": list(TRAIN_SEEDS),
                   "budgets_decisions": list(BUDGETS),
                   "decision_frames": DECISION_FRAMES,
                   "episodes_per_window": EPISODES,
                   "temperature": TEMPERATURE, "feature_dim": FEATURE_DIM,
                   "prefix_frame": PREFIX_FRAMES,
                   "generator_seed_rule": "seed * 1000 + 7"},
        "episode_loop": "tools/rl/v830_bombpol.py::run_episode36 (imported)",
        "references": REFERENCE,
        "determinism_note": (
            "H2/H3 use argmax on the candidate and a deterministic distance rule "
            "on the bomb, so an episode has no random input; the 300 episodes of "
            "one (seed, window) cell are replays. --mode diagnostics reports "
            "distinct_trajectories next to episodes."),
    }
    missing = [f"{arm}/seed{seed}" for arm in ARMS for seed in TRAIN_SEEDS
               if not ckpt_path(arm, seed).is_file()]
    rec["missing_checkpoints"] = missing
    ok = (bool(pins["all_match"]) and bool(engine.get("match")) and not missing
          and int(rec["feature_dim"]) == FEATURE_DIM)
    rec["all_pass"] = ok
    out = pathlib.Path(args.out) if args.out else root / "preflight.json"
    write_json(out, rec)
    log(f"preflight: pins={pins['all_match']} engine={engine.get('match')} "
        f"feature_dim={rec['feature_dim']} missing_ckpt={len(missing)} "
        f"all_pass={ok} -> {out}")
    return 0 if ok else 2


# --------------------------------------------------------------------------- #
# mode: assert-ckpt
# --------------------------------------------------------------------------- #
def mode_assert(args: argparse.Namespace) -> int:
    out_path = (pathlib.Path(args.out) if args.out
                else pathlib.Path(args.work_dir) / "ckpt_assertions.json")
    payload: dict[str, Any] = {
        "schema": SCHEMA_ASSERT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": args.phase,
        "requirement": (
            "checklist item 2: the behaviour policy of the v8.3.0/v8.6.0 states is "
            "the frozen v4.5.0 `S` student (round2.pt = 4abee069… / a93746bb… / "
            "c69960fc…); the three policies evaluated here are the v8.6.0 "
            "bombdag.pt and the v8.3.0 bombpol round2.pt; the protected sources "
            "and the prereg must not have moved"),
        "engine": None, "v450_round2": {}, "policy_ckpts": {}, "protected": {},
        "v8c0_hybrid_py": {}, "prereg": {}, "all_pass": False,
    }
    try:
        payload["engine"] = engine_gate(args.engine)
        ok = bool(payload["engine"].get("match"))
        for seed in TRAIN_SEEDS:
            path = V450_CKPT_ROOT / f"seed{seed}" / "round2.pt"
            expected = FROZEN_ROUND2_SHA256[seed]
            digest = sha256_file(path) if path.is_file() else None
            match = bool(digest == expected)
            ok = ok and match
            payload["v450_round2"][str(seed)] = {
                "path": str(path), "exists": path.is_file(), "sha256": digest,
                "expected": expected, "expected_prefix": expected[:8],
                "match": match,
                "bytes": path.stat().st_size if path.is_file() else None}
        for arm in ARMS:
            payload["policy_ckpts"][arm] = {
                str(seed): policy_sha256(arm, seed) for seed in TRAIN_SEEDS}
            ok = ok and all(v["exists"] for v in
                            payload["policy_ckpts"][arm].values())
        for rel in PROTECTED_FILES:
            path = REPO / rel
            payload["protected"][rel] = {
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None}
        me = pathlib.Path(__file__).resolve()
        payload["v8c0_hybrid_py"] = {"path": str(me), "sha256": sha256_file(me)}
        payload["prereg"] = {
            "path": PREREG_REL,
            "sha256": sha256_file(PREREG) if PREREG.is_file() else None,
            "expected": PREREG_SHA256,
            "match": (PREREG.is_file()
                      and sha256_file(PREREG) == PREREG_SHA256)}
        ok = ok and payload["prereg"]["match"]
        payload["all_pass"] = bool(ok)
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
    write_json(out_path, payload)
    log(f"assert-ckpt {args.phase}: all_pass={payload['all_pass']} -> {out_path}")
    return 0 if payload["all_pass"] else 2


# --------------------------------------------------------------------------- #
# mode: smoke (prereg: one episode before any parallel run)
# --------------------------------------------------------------------------- #
def mode_smoke(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    arm = str(args.arm)
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    windows = parse_windows(args.windows, allow_train=True)[:1]
    window = windows[0]
    tag = args.tag or f"v8c0-smoke-{arm}-s{seed}-w{window}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    out_path = pathlib.Path(args.out) if args.out else root / "smoke.json"
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_arm_policy(arm, seed)

    payload: dict[str, Any] = {
        "schema": SCHEMA_SMOKE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL,
        "requirement": ("one episode of one arm, with a per-decision trace, before "
                        "any parallel run: does the bomb bit reach the engine and "
                        "does bombs_remaining drop?"),
        "arm": arm, "arm_spec": ARM_SPEC[arm], "policy_seed": seed,
        "window": int(window), "budget_decisions": budget,
        "episodes": int(args.episodes),
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest,
                 "bytes": ckpt.stat().st_size},
        "rows": [], "checks": {}, "all_pass": False, "error": None, "wall_s": None,
    }
    parents: list[Any] = []
    handle: Any = None
    started = time.perf_counter()
    try:
        data_dir, data_info = v450().prepare_data_dir(data_root, window,
                                                     args.data_source)
        payload["cfg"] = data_info["cfg"]
        payload["data_dir"] = str(data_dir)
        parent, gate = start_parent(slot=0, window=window, engine=args.engine,
                                    data_dir=data_dir, log_dir=run_dir)
        parents.append(parent)
        payload["gate"] = {key: gate[key] for key in (
            "hello_role_shadow", "park_ok", "park_frame", "threads_1", "all_pass")}
        payload["gate_prefix"] = gate["prefix"]
        generator = process_generator(arm, seed)
        for episode in range(int(args.episodes)):
            box = new_box()
            decide = make_decide(arm, policy=policy, generator=generator,
                                 box=box)
            record = v830().run_episode36(
                parent, episode=episode, policy_seed=seed, decisions=budget,
                decide=decide, budget=budget,
                decision_frames=DECISION_FRAMES, keep_trace=True)
            record["arm"] = arm
            record["schema"] = SCHEMA_SMOKE
            record["bomb_source"] = ARM_SPEC[arm]["bomb"]
            record["audit"] = {key: box[key] for key in (
                "pair_margin", "cand_best_score", "nearest_all_px",
                "threshold_px", "bombs_remaining_at_decision", "distance_ok")}
            payload["rows"].append(record)
            if record.get("invalid"):
                raise HybridError(f"smoke episode {episode} invalid: "
                                  f"{record.get('error')}")
            log(f"smoke {arm} s{seed} w{window} ep{episode}: frames="
                f"{record['frames']} bombs {record['bombs_start']}->"
                f"{record['bombs_end']} used={record['bombs_used']} "
                f"emits={record['bomb_emits']} survived={record['survived']} "
                f"bomb_decisions={record['bomb_decisions']}")
            # the bombs_remaining curve, per decision
            curve = []
            for row in (record.get("trace") or []):
                curve.append(int(row["bombs_remaining"]))
            payload["rows"][-1]["bombs_remaining_curve"] = curve
            log(f"  bombs_remaining curve ({len(curve)} decisions): "
                f"{compress_curve(curve)}")
        rows = payload["rows"]
        emits = sum(int(row["bomb_emits"]) for row in rows)
        used = sum(int(row["bombs_used"] or 0) for row in rows)
        payload["checks"] = {
            "episodes": len(rows),
            "episodes_valid": sum(1 for row in rows if not row["invalid"]),
            "survived": sum(1 for row in rows if row["survived"]),
            "bomb_emits_total": emits,
            "bombs_used_total": used,
            "bombs_used_per_episode": (round(used / len(rows), 4) if rows else None),
            "bomb_path_fired": bool(emits > 0),
            "bombs_remaining_decreased": bool(used > 0),
            "counter_bookkeeping_closes": bool(
                used == sum(int(row["bombs_start"]) - int(row["bombs_end"])
                            for row in rows)),
            "emits_honoured": sum(1 for row in rows
                                  if int(row["bomb_emits"])
                                  - int(row.get("bombs_ignored_emits") or 0) > 0),
        }
        payload["all_pass"] = bool(
            payload["checks"]["episodes_valid"] == len(rows)
            and payload["checks"]["bomb_path_fired"]
            and payload["checks"]["bombs_remaining_decreased"]
            and payload["checks"]["counter_bookkeeping_closes"])
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
        log(f"SMOKE FAILED {arm}: {payload['error']}")
    finally:
        for parent in parents:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        payload["wall_s"] = round(time.perf_counter() - started, 3)
    write_json(out_path, payload)
    log(f"smoke: {json.dumps(payload['checks'], ensure_ascii=False)} "
        f"all_pass={payload['all_pass']} -> {out_path}")
    return 0 if payload["all_pass"] else 4


def compress_curve(curve: Sequence[int]) -> str:
    """``[3,3,3,2,2,...]`` -> ``3x3, 2x10, ...`` (a readable bombs_remaining curve)."""

    if not curve:
        return "(empty)"
    parts: list[str] = []
    current = curve[0]
    count = 1
    for value in curve[1:]:
        if value == current:
            count += 1
        else:
            parts.append(f"{current}x{count}")
            current = value
            count = 1
    parts.append(f"{current}x{count}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# mode: eval (one arm x one budget x one seed, 4 dev windows)
# --------------------------------------------------------------------------- #
def mode_eval(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    arm = str(args.arm)
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    if arm not in ARMS:
        raise HybridError(f"--arm must be one of {ARMS}, got {arm}")
    if seed not in TRAIN_SEEDS:
        raise HybridError(f"--policy-seed must be one of {TRAIN_SEEDS}, got {seed}")
    if budget not in BUDGETS:
        raise HybridError(f"prereg §1 fixes --decisions to {BUDGETS}, got {budget}")
    if int(args.decision_frames) != DECISION_FRAMES:
        raise HybridError(f"prereg §1 keeps {DECISION_FRAMES} frames per decision")
    if int(args.episodes) < 1:
        raise HybridError(f"--episodes must be >= 1, got {args.episodes}")
    windows = parse_windows(args.windows)
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    tag = args.tag or f"v8c0-eval-{arm}-d{budget}-s{seed}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    out_path = (pathlib.Path(args.out) if args.out
                else eval_path(root, arm, budget, seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")
    policy, ckpt, digest = load_arm_policy(arm, seed)

    summary: dict[str, Any] = {
        "schema": SCHEMA_EVAL_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL, "mode": "eval", "arm": arm,
        "arm_spec": ARM_SPEC[arm], "policy_seed": seed,
        "budget_decisions": budget, "windows": list(windows),
        "episodes_per_window": int(args.episodes),
        "decision_frames": DECISION_FRAMES, "temperature": TEMPERATURE,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest,
                 "bytes": ckpt.stat().st_size},
        "sampler": ({
            "kind": "torch.multinomial on softmax(z/1) over 36 actions",
            "generator_seed": generator_seed(seed),
            "generator": "one torch.Generator per (arm, budget, seed) process, "
                         "windows in the frozen order"}
            if arm == "H1" else
            ({"kind": ("torch.multinomial on softmax(z/1) over the 18 pairwise "
                       "candidate maxima max(logit[2c], logit[2c+1])"),
              "generator_seed": generator_seed(seed),
              "generator": ("one torch.Generator per (arm, budget, seed) process, "
                            "windows in the frozen order; the bomb bit is NOT "
                            "sampled, it is the R5 rule's")}
             if arm == "H2s" else
             {"kind": "deterministic argmax over max(logit[2c], logit[2c+1])"
                      " + the R5 staged rule",
              "generator_seed": None,
              "generator": "none: this arm consumes no random numbers"})),
        "episode_loop": "tools/rl/v830_bombpol.py::run_episode36 (imported)",
        "bomb_rule": R5_RULE if arm != "H1" else None,
        "out": str(out_path), "per_window": {}, "status": "failed",
        "error": None, "wall_s": None,
    }
    parents: list[Any] = []
    handle = out_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    try:
        from touhou_rl.v420 import contracts as C

        generator = process_generator(arm, seed)
        for slot, window in enumerate(windows):
            C.assert_seed_allowed(int(window), "train")
            data_dir, data_info = v450().prepare_data_dir(
                data_root, window, args.data_source)
            log(f"{arm} d{budget} s{seed} w{window}: cfg muted="
                f"{data_info['cfg']['muted_four_zero']} "
                f"sha256={data_info['cfg']['sha256'][:16]}…")
            parent, gate = start_parent(slot=slot, window=window,
                                        engine=args.engine, data_dir=data_dir,
                                        log_dir=run_dir)
            parents.append(parent)
            rows: list[dict[str, Any]] = []
            for episode in range(int(args.episodes)):
                box = new_box()
                decide = make_decide(arm, policy=policy, generator=generator,
                                     box=box)
                record = v830().run_episode36(
                    parent, episode=episode, policy_seed=seed, decisions=budget,
                    decide=decide, budget=budget,
                    decision_frames=DECISION_FRAMES)
                record["schema"] = SCHEMA_EVAL
                record["producer"] = PRODUCER
                record["arm"] = arm
                record["cand_source"] = ARM_SPEC[arm]["candidate"]
                record["bomb_source"] = ARM_SPEC[arm]["bomb"]
                record["ckpt_sha256"] = digest
                record["out_path"] = str(out_path)
                # the arm's own audit trail (H2/H2s/H3 only); H1 carries the sample
                if arm == "H1":
                    record["sampled_action"] = box["sampled_action"]
                else:
                    record["pair_margin"] = box["pair_margin"]
                    record["cand_best_score"] = box["cand_best_score"]
                    record["nearest_all_px"] = box["nearest_all_px"]
                    record["threshold_px"] = box["threshold_px"]
                    record["bombs_remaining_at_decision"] = \
                        box["bombs_remaining_at_decision"]
                    record["distance_ok"] = box["distance_ok"]
                    if arm == "H2s":
                        record["cand_prob"] = box["cand_prob"]
                for drop in ("terminal_reply", "trace", "spawn_reply"):
                    record.pop(drop, None)
                write_jsonl_line(handle, record)
                handle.flush()
                rows.append(record)
                if record.get("invalid"):
                    raise HybridError(f"{arm} d{budget} seed {seed} window "
                                      f"{window} episode {episode} is invalid: "
                                      f"{record.get('error')}")
                if (episode + 1) % 25 == 0:
                    done = sum(1 for row in rows if row["survived"])
                    used = sum(int(row["bombs_used"] or 0) for row in rows)
                    log(f"  {arm} d{budget} s{seed} w{window}: {episode + 1}/"
                        f"{args.episodes} episodes, survived={done}, "
                        f"bombs_used={used}, wall="
                        f"{round(time.perf_counter() - started, 1)}s")
            summary["per_window"][str(window)] = window_block(window, rows,
                                                              data_info, data_dir,
                                                              gate)
            log(f"{arm} d{budget} seed {seed} window {window}: survived "
                f"{summary['per_window'][str(window)]['survived']}/{len(rows)} "
                f"mean_alive="
                f"{summary['per_window'][str(window)]['alive_frames']['mean']} "
                f"distinct={summary['per_window'][str(window)]['distinct_trajectories']}")
            # a partial summary after every window: an interrupted run must still
            # leave per-window numbers on disk, not a promise
            write_json(summary_path, {**summary, "status": "partial",
                                      "windows_done": len(summary["per_window"]),
                                      "wall_s": round(
                                          time.perf_counter() - started, 3)})
            parent.close()
            parents.remove(parent)
        summary["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["status"] = "failed"
        log(f"EVAL FAILED {arm} d{budget} seed {seed}: {summary['error']}")
    finally:
        handle.close()
        for parent in parents:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        summary["wall_s"] = round(time.perf_counter() - started, 3)
        episodes_total = sum(int(v["episodes"])
                             for v in summary["per_window"].values())
        survived_total = sum(int(v["survived"])
                             for v in summary["per_window"].values())
        summary["episodes_total"] = episodes_total
        summary["survived_total"] = survived_total
        summary["pooled_survived_fraction"] = (
            round(survived_total / episodes_total, 6) if episodes_total else None)
        summary["bombs_used_total"] = sum(
            int(v["bombs_used_total"]) for v in summary["per_window"].values())
        summary["bomb_emits_total"] = sum(
            int(v["bomb_emits_total"]) for v in summary["per_window"].values())
        summary["distinct_trajectories_total"] = sum(
            int(v["distinct_trajectories"])
            for v in summary["per_window"].values())
        if out_path.is_file():
            summary["out_sha256"] = sha256_file(out_path)
            summary["out_bytes"] = out_path.stat().st_size
        write_json(summary_path, summary)
    log(f"eval {arm} d{budget} seed {seed}: status={summary['status']} "
        f"survived={summary.get('survived_total')}/{summary.get('episodes_total')} "
        f"bombs_used={summary.get('bombs_used_total')} "
        f"wall={summary['wall_s']}s -> {out_path}")
    return 0 if summary["status"] == "complete" else 4


def window_block(window: int, rows: Sequence[dict[str, Any]],
                 data_info: dict[str, Any], data_dir: pathlib.Path,
                 gate: dict[str, Any]) -> dict[str, Any]:
    survived = sum(1 for row in rows if row["survived"])
    bombs_used = [int(row["bombs_used"] or 0) for row in rows]
    first = [int(row["first_bomb_decision"]) for row in rows
             if row.get("first_bomb_decision") is not None]
    last = [max(int(d) for d in row["bomb_decisions"]) for row in rows
            if row.get("bomb_decisions")]
    signatures = {(int(row["alive_frames"] or 0), int(row["bombs_used"] or 0),
                   tuple(int(d) for d in row["bomb_decisions"])) for row in rows}
    return {
        "window": int(window), "episodes": len(rows), "survived": survived,
        "survived_fraction": round(survived / len(rows), 6) if rows else None,
        "alive_frames": quantiles([float(row["alive_frames"] or 0) for row in rows]),
        "frames": quantiles([float(row["frames"]) for row in rows]),
        "bomb_episodes": sum(1 for row in rows if row["bomb_emits"]),
        "bomb_emits_total": sum(int(row["bomb_emits"]) for row in rows),
        "bombs_used_total": sum(bombs_used),
        "bombs_used_counts": {str(k): sum(1 for v in bombs_used if v == k)
                              for k in (0, 1, 2, 3)},
        "episodes_exhausted": sum(1 for row in rows if row["bombs_end"] == 0),
        "first_bomb_decision": quantiles([float(v) for v in first]) if first else {"n": 0},
        "last_bomb_decision": quantiles([float(v) for v in last]) if last else {"n": 0},
        "distinct_trajectories": len(signatures),
        "distinct_trajectory_note": ("the 300 episodes of this cell are replays if "
                                     "this is 1"),
        "cfg_muted": data_info["cfg"]["muted_four_zero"],
        "cfg_sha256": data_info["cfg"]["sha256"],
        "data_dir": str(data_dir),
        "gate": {key: gate[key] for key in (
            "hello_role_shadow", "park_ok", "park_frame", "threads_1", "all_pass")},
        "gate_prefix": {key: gate["prefix"][key] for key in (
            "hash_checks", "hash_mismatches", "final_frame")},
    }


# --------------------------------------------------------------------------- #
# mode: crosscheck (H1's code path == v8.6.0's recorded rows)
# --------------------------------------------------------------------------- #
CROSSCHECK_FIELDS = ("frames", "alive_frames", "first_death_offset",
                     "first_death_frame", "survived", "decisions_executed",
                     "terminal", "bombs_start", "bombs_end", "bombs_used",
                     "bomb_emits", "bomb_decisions", "bomb_ignored")


def v860_eval_path(budget: int, seed: int) -> pathlib.Path:
    return (REPO / ".tmp_v860_bombdag" / "eval"
            / f"eval_d{int(budget)}_seed{int(seed)}.jsonl")


def mode_crosscheck(args: argparse.Namespace) -> int:
    """Run H1 through **this** file and compare with v8.6.0's recorded row."""

    root = pathlib.Path(args.work_dir)
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    episodes = int(args.episodes)
    windows = parse_windows(args.windows)[:1]
    window = windows[0]
    tag = args.tag or f"v8c0-crosscheck-s{seed}-d{budget}-w{window}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    out_path = (pathlib.Path(args.out) if args.out
                else root / "crosscheck.json")
    source = v860_eval_path(budget, seed)
    if not source.is_file():
        raise HybridError(f"no v8.6.0 rows to compare against: {source}")
    recorded: list[dict[str, Any]] = []
    for row in read_jsonl(source):
        if int(row.get("window", -1)) == int(window):
            recorded.append(row)
    if len(recorded) < episodes:
        raise HybridError(f"{source} holds {len(recorded)} rows for window "
                          f"{window}, need {episodes}")
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    policy, ckpt, digest = load_arm_policy("H1", seed)

    payload: dict[str, Any] = {
        "schema": "v8c0-hybrid-crosscheck/1", "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL,
        "requirement": ("H1 is v8.6.0's eval arm; before adopting the v8.6.0 grid "
                        "for H1, this file's H1 code path must reproduce the "
                        "recorded rows bit for bit"),
        "reference": {"path": str(source), "sha256": sha256_file(source),
                      "rows_for_window": len(recorded)},
        "arm": "H1", "policy_seed": seed, "budget_decisions": budget,
        "window": int(window), "episodes": episodes,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest},
        "fields": list(CROSSCHECK_FIELDS),
        "rows": [], "mismatches": [], "all_pass": False, "error": None,
        "wall_s": None,
    }
    parents: list[Any] = []
    started = time.perf_counter()
    try:
        data_dir, data_info = v450().prepare_data_dir(data_root, window,
                                                     args.data_source)
        payload["cfg"] = data_info["cfg"]
        parent, gate = start_parent(slot=0, window=window, engine=args.engine,
                                    data_dir=data_dir, log_dir=run_dir)
        parents.append(parent)
        payload["gate"] = {key: gate[key] for key in (
            "hello_role_shadow", "park_ok", "park_frame", "threads_1", "all_pass")}
        generator = process_generator("H1", seed)
        for episode in range(episodes):
            box = new_box()
            decide = make_decide("H1", policy=policy, generator=generator,
                                 box=box)
            mine = v830().run_episode36(
                parent, episode=episode, policy_seed=seed, decisions=budget,
                decide=decide, budget=budget, decision_frames=DECISION_FRAMES)
            ref = recorded[episode]
            diffs = {field: {"mine": mine.get(field), "v860": ref.get(field)}
                     for field in CROSSCHECK_FIELDS
                     if mine.get(field) != ref.get(field)}
            if diffs:
                payload["mismatches"].append({"episode": int(episode),
                                              "differences": diffs})
            payload["rows"].append({"episode": int(episode),
                                    "identical": not diffs,
                                    "alive_frames": mine.get("alive_frames"),
                                    "v860_alive_frames": ref.get("alive_frames"),
                                    "bombs_used": mine.get("bombs_used"),
                                    "v860_bombs_used": ref.get("bombs_used")})
            log(f"  crosscheck ep{episode}: identical={not diffs} "
                f"alive={mine.get('alive_frames')}/{ref.get('alive_frames')}")
            if mine.get("invalid"):
                raise HybridError(f"crosscheck episode {episode} invalid: "
                                  f"{mine.get('error')}")
        payload["all_pass"] = bool(not payload["mismatches"]
                                   and len(payload["rows"]) == episodes)
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["all_pass"] = False
    finally:
        for parent in parents:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        payload["wall_s"] = round(time.perf_counter() - started, 3)
    write_json(out_path, payload)
    log(f"crosscheck: {episodes - len(payload['mismatches'])}/{episodes} identical "
        f"all_pass={payload['all_pass']} -> {out_path}")
    return 0 if payload["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: adopt-h1 (the v8.6.0 grid is H1)
# --------------------------------------------------------------------------- #
def mode_adopt_h1(args: argparse.Namespace) -> int:
    import shutil

    root = pathlib.Path(args.work_dir)
    out_path = (pathlib.Path(args.out) if args.out
                else root / "h1_adoption.json")
    crosscheck = read_json(root / "crosscheck.json") if (
        root / "crosscheck.json").is_file() else {}
    payload: dict[str, Any] = {
        "schema": "v8c0-hybrid-h1-adoption/1", "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "decision": ("H1 = the v8.6.0 36-action policy with its own bomb head, "
                     "i.e. exactly the arm v8.6.0's own eval grid measured.  That "
                     "grid finished on this machine minutes before this round "
                     "started (same protocol, same checkpoints, same generator "
                     "rule, same four dev windows), so its files ARE H1 and are "
                     "adopted rather than re-run"),
        "precondition": {
            "crosscheck_path": str(root / "crosscheck.json"),
            "crosscheck_all_pass": crosscheck.get("all_pass"),
            "crosscheck_episodes": crosscheck.get("episodes"),
            "crosscheck_mismatches": len(crosscheck.get("mismatches") or []),
            "required": True},
        "cells": {}, "all_pass": False,
    }
    if not crosscheck.get("all_pass"):
        payload["error"] = ("refusing to adopt: the H1 crosscheck did not pass, so "
                            "this file's H1 code path is not proven to be v8.6.0's")
        write_json(out_path, payload)
        log(f"adopt-h1 REFUSED: {payload['error']} -> {out_path}")
        return 4
    for budget in BUDGETS:
        for seed in TRAIN_SEEDS:
            source = v860_eval_path(budget, seed)
            summary_source = source.with_suffix(".summary.json")
            if not source.is_file() or not summary_source.is_file():
                raise HybridError(f"missing v8.6.0 eval artifact: {source}")
            rec = read_json(summary_source)
            if rec.get("status") != "complete":
                raise HybridError(f"{summary_source}: status {rec.get('status')}")
            rows = read_jsonl(source)
            if int(rec.get("episodes_total") or -1) != len(rows):
                raise HybridError(f"{source}: {len(rows)} rows but the summary "
                                  f"says {rec.get('episodes_total')}")
            target = eval_path(root, "H1", budget, seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            survived = sum(1 for row in rows if row["survived"])
            bombs = sum(int(row["bombs_used"] or 0) for row in rows)
            target_summary = target.with_suffix(".summary.json")
            write_json(target_summary, {
                "schema": SCHEMA_EVAL_SUMMARY, "producer": PRODUCER,
                "created_at": payload["created_at"], "prereg": PREREG_REL,
                "mode": "eval", "arm": "H1", "arm_spec": ARM_SPEC["H1"],
                "policy_seed": int(seed), "budget_decisions": int(budget),
                "windows": list(DEV_WINDOWS),
                "episodes_per_window": EPISODES,
                "decision_frames": DECISION_FRAMES, "temperature": TEMPERATURE,
                "adopted": True,
                "adopted_from": {
                    "jsonl": str(source), "jsonl_sha256": sha256_file(source),
                    "summary": str(summary_source),
                    "summary_sha256": sha256_file(summary_source),
                    "v860_status": rec.get("status"),
                    "v860_ckpt": rec.get("ckpt"),
                    "crosscheck_all_pass": crosscheck.get("all_pass"),
                    "crosscheck": str(root / "crosscheck.json"),
                    "windows": rec.get("windows"),
                    "sampler": rec.get("sampler")},
                "engine": rec.get("engine"),
                "ckpt": rec.get("ckpt"),
                "episodes_total": len(rows), "survived_total": survived,
                "pooled_survived_fraction": round(survived / len(rows), 6),
                "bombs_used_total": bombs,
                "bomb_emits_total": sum(int(row["bomb_emits"]) for row in rows),
                "out": str(target), "out_sha256": sha256_file(target),
                "out_bytes": target.stat().st_size,
                "per_window": rec.get("per_window"),
                "status": "complete", "error": None, "wall_s": 0.0})
            payload["cells"][f"H1_d{budget}_seed{seed}"] = {
                "source": str(source), "source_sha256": sha256_file(source),
                "target": str(target), "rows": len(rows),
                "survived": survived,
                "fraction": round(survived / len(rows), 6),
                "bombs_used_total": bombs}
            log(f"adopt-h1 d{budget} s{seed}: {survived}/{len(rows)} -> {target}")
    payload["all_pass"] = True
    write_json(out_path, payload)
    log(f"adopt-h1: all_pass=True -> {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# mode: diagnostics (offline, descriptive only)
# --------------------------------------------------------------------------- #
def collect_cells(root: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    cells: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        for budget in BUDGETS:
            rows: list[dict[str, Any]] = []
            for seed in TRAIN_SEEDS:
                path = eval_path(root, arm, budget, seed)
                for row in read_jsonl(path):
                    row["_seed_file"] = int(seed)
                    rows.append(row)
            cells[f"{arm}_d{budget}"] = rows
    return cells


def cell_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0}
    bombs_used = [int(row["bombs_used"] or 0) for row in rows]
    first = [int(row["first_bomb_decision"]) for row in rows
             if row.get("first_bomb_decision") is not None]
    last = [max(int(d) for d in row["bomb_decisions"]) for row in rows
            if row.get("bomb_decisions")]
    per_window: dict[str, Any] = {}
    for row in rows:
        key = str(int(row["window"]))
        bucket = per_window.setdefault(key, {"episodes": 0, "survived": 0,
                                             "alive": [], "bombs": []})
        bucket["episodes"] += 1
        bucket["survived"] += int(bool(row["survived"]))
        bucket["alive"].append(float(row["alive_frames"] or 0))
        bucket["bombs"].append(int(row["bombs_used"] or 0))
    windows = {key: {"episodes": v["episodes"], "survived": v["survived"],
                     "survived_fraction": round(v["survived"] / v["episodes"], 6),
                     "alive_frames": quantiles(v["alive"]),
                     "bombs_used_mean": round(statistics.fmean(v["bombs"]), 4)}
               for key, v in sorted(per_window.items())}
    per_seed: dict[str, Any] = {}
    for row in rows:
        key = str(int(row["_seed_file"]))
        bucket = per_seed.setdefault(key, {"episodes": 0, "survived": 0,
                                           "bombs": []})
        bucket["episodes"] += 1
        bucket["survived"] += int(bool(row["survived"]))
        bucket["bombs"].append(int(row["bombs_used"] or 0))
    seeds = {key: {"episodes": v["episodes"], "survived": v["survived"],
                   "survived_fraction": round(v["survived"] / v["episodes"], 6),
                   "bombs_used_mean": round(statistics.fmean(v["bombs"]), 4)}
             for key, v in sorted(per_seed.items())}
    signatures = {(int(row["window"]), int(row["_seed_file"]),
                   int(row["alive_frames"] or 0), int(row["bombs_used"] or 0),
                   tuple(int(d) for d in row["bomb_decisions"]))
                  for row in rows}
    return {
        "episodes": len(rows),
        "survived": sum(1 for row in rows if row["survived"]),
        "survived_fraction": round(
            sum(1 for row in rows if row["survived"]) / len(rows), 6),
        "alive_frames": quantiles([float(row["alive_frames"] or 0) for row in rows]),
        "decisions_executed": quantiles(
            [float(row["decisions_executed"]) for row in rows]),
        "bombs_used": {
            "distribution": {str(k): sum(1 for v in bombs_used if v == k)
                             for k in (0, 1, 2, 3)},
            "mean": round(statistics.fmean(bombs_used), 4),
            "quantiles": quantiles([float(v) for v in bombs_used]),
            "episodes_exhausted": sum(1 for v in bombs_used if v == 3),
        },
        "bomb_emits_total": sum(int(row["bomb_emits"]) for row in rows),
        "bomb_ignored_emits": sum(int(row.get("bombs_ignored_emits") or 0)
                                  for row in rows),
        "first_bomb_decision": (
            {"quantiles": quantiles([float(v) for v in first]),
             "histogram": decision_histogram(first),
             "episodes_with_a_bomb": len(first)} if first else {"n": 0}),
        "last_bomb_decision": (
            {"quantiles": quantiles([float(v) for v in last]),
             "histogram": decision_histogram(last)} if last else {"n": 0}),
        "distinct_trajectories": len(signatures),
        "distinct_trajectories_note": (
            "distinct over (window, seed, alive_frames, bombs_used, "
            "bomb_decisions): 12 for a deterministic arm means the pooled 3,600 "
            "episodes carry 12 independent rollouts"),
        "per_window": windows, "per_seed": seeds,
    }


def bomb_usage_block(cells: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, rows in cells.items():
        if not rows:
            continue
        bombs_used = [int(row["bombs_used"] or 0) for row in rows]
        first = [int(row["first_bomb_decision"]) for row in rows
                 if row.get("first_bomb_decision") is not None]
        thresholds: dict[str, int] = {}
        for row in rows:
            for value in (row.get("threshold_px") or []):
                thresholds[str(value)] = thresholds.get(str(value), 0) + 1
        out[key] = {
            "episodes": len(rows),
            "distribution": {str(k): sum(1 for v in bombs_used if v == k)
                             for k in (0, 1, 2, 3)},
            "mean": round(statistics.fmean(bombs_used), 4),
            "median": round(statistics.median(bombs_used), 4),
            "episodes_with_zero_bombs": sum(1 for v in bombs_used if v == 0),
            "episodes_at_three_bombs": sum(1 for v in bombs_used if v == 3),
            "first_bomb_decision_histogram": decision_histogram(first),
            "threshold_px_used_at_each_decision": thresholds,
            "sample_size_note": "descriptive only; prereg §5 applies",
        }
    return out


def mode_diagnostics(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    cells = collect_cells(root)
    payload: dict[str, Any] = {
        "schema": SCHEMA_DIAG, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL,
        "requirement": ("offline counts/means/medians/quantiles and the bomb-usage "
                        "distribution per arm per budget; NO test, NO interval "
                        "(prereg §2's tests live in --mode verdict)"),
        "arms": {arm: ARM_SPEC[arm] for arm in ARMS},
        "budgets": list(BUDGETS),
        "cells": {}, "determinism": {}, "references": REFERENCE,
    }
    for key, rows in cells.items():
        payload["cells"][key] = cell_block(rows)
    for key, block in payload["cells"].items():
        if block.get("episodes"):
            payload["determinism"][key] = {
                "episodes": block["episodes"],
                "distinct_trajectories": block["distinct_trajectories"],
                "replay_factor": round(block["episodes"]
                                       / max(block["distinct_trajectories"], 1), 3)}
    usage = bomb_usage_block(cells)
    out = pathlib.Path(args.out) if args.out else root / "diagnostics.json"
    write_json(out, payload)
    usage_out = (pathlib.Path(args.usage_out) if args.usage_out
                 else root / "bomb_usage.json")
    write_json(usage_out, {"schema": SCHEMA_BOMB_USAGE, "producer": PRODUCER,
                           "created_at": payload["created_at"],
                           "prereg": PREREG_REL, "cells": usage})
    for key in sorted(payload["cells"]):
        block = payload["cells"][key]
        if not block.get("episodes"):
            continue
        log(f"{key}: {block['survived']}/{block['episodes']} = "
            f"{100.0 * block['survived_fraction']:.2f}% "
            f"alive_mean={block['alive_frames']['mean']} "
            f"median={block['alive_frames']['median']} "
            f"bombs_mean={block['bombs_used']['mean']} "
            f"distinct={block['distinct_trajectories']}")
    log(f"diagnostics -> {out}, {usage_out}")
    return 0


# --------------------------------------------------------------------------- #
# exact statistics for --mode verdict (no scipy: pure Python)
# --------------------------------------------------------------------------- #
def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def fisher_one_sided_greater(a: int, b: int, c: int, d: int) -> float:
    """P(X >= a) for the 2x2 table [[a, b], [c, d]] (one-sided, greater)."""

    n = a + b + c + d
    row1 = a + b
    col1 = a + c
    if n == 0:
        return 1.0
    log_den = _log_comb(n, col1)
    total = 0.0
    upper = min(row1, col1)
    for k in range(a, upper + 1):
        log_num = _log_comb(row1, k) + _log_comb(n - row1, col1 - k)
        total += math.exp(log_num - log_den)
    return min(1.0, max(0.0, total))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes ``betacf``)."""

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
    """Regularised incomplete beta I_x(a, b)."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (_log_comb(0, 0)  # keep the lgamma helper in one place
                 + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
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


def mode_verdict(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    cells = collect_cells(root)
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERDICT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL,
        "statistics": ("Clopper-Pearson exact 95 % interval for a single rate; "
                       "one-sided exact Fisher for H2 > H1 (prereg §2).  Both are "
                       "reported on the literal episode pool AND on the "
                       "distinct-trajectory count, because an H2/H3 episode has no "
                       "random input and the 300 episodes of one cell are replays"),
        "criteria": {
            "P1": ("H2s d600 survived fraction >= 0.70 (the supervising agent's "
                   "re-scoped P1: H2s is the stochastic arm, so its pool is "
                   "3,600 independent episodes and is the only arm directly "
                   "comparable to R5's 77.11 %); H2 d600 is reported beside it as "
                   "the deterministic upper bound with distinct-n = 12"),
            "P2": "H2s > H1 at d600, one-sided Fisher p < 0.05",
            "P3": "H2s d900 survived fraction >= 0.15",
            "P4": "H2s bombs used per episode should be near R5's 600-decision "
                  "usage ({3: 3540, 2: 60} in the v8.1.0 record) (descriptive)",
            "P5": "H3 vs H2 (descriptive: candidate-head quality)",
        },
        "cells": {}, "tests": {}, "references": REFERENCE, "stops": {},
    }

    def block(arm: str, budget: int) -> dict[str, Any]:
        rows = cells.get(f"{arm}_d{budget}") or []
        if not rows:
            return {"episodes": 0}
        k = sum(1 for row in rows if row["survived"])
        n = len(rows)
        signatures = {(int(row["window"]), int(row["_seed_file"]),
                       int(row["alive_frames"] or 0), int(row["bombs_used"] or 0),
                       tuple(int(d) for d in row["bomb_decisions"]))
                      for row in rows}
        distinct_survived = 0
        seen: set[Any] = set()
        for row in rows:
            key = (int(row["window"]), int(row["_seed_file"]),
                   int(row["alive_frames"] or 0), int(row["bombs_used"] or 0),
                   tuple(int(d) for d in row["bomb_decisions"]))
            if key in seen:
                continue
            seen.add(key)
            distinct_survived += int(bool(row["survived"]))
        bombs = [int(row["bombs_used"] or 0) for row in rows]
        distinct_bombs: dict[Any, int] = {}
        for row in rows:
            key = (int(row["window"]), int(row["_seed_file"]),
                   int(row["alive_frames"] or 0), int(row["bombs_used"] or 0),
                   tuple(int(d) for d in row["bomb_decisions"]))
            distinct_bombs[key] = int(row["bombs_used"] or 0)
        return {
            "episodes": n, "survived": k,
            "survived_fraction": round(k / n, 6),
            "cp95_episode_pool": clopper_pearson(k, n),
            "distinct_trajectories": len(signatures),
            "distinct_survived": distinct_survived,
            "distinct_fraction": round(distinct_survived / len(signatures), 6),
            "cp95_distinct": clopper_pearson(distinct_survived, len(signatures)),
            "bombs_used_mean": round(statistics.fmean(bombs), 4),
            "bombs_used_distribution": {str(k): sum(1 for v in bombs if v == k)
                                        for k in (0, 1, 2, 3)},
            "bombs_used_distinct_mean": round(
                statistics.fmean(list(distinct_bombs.values())), 4),
        }

    for arm in ARMS:
        for budget in BUDGETS:
            payload["cells"][f"{arm}_d{budget}"] = block(arm, budget)

    def compare(arm_a_name: str, arm_b_name: str, budget: int) -> dict[str, Any]:
        a = payload["cells"].get(f"{arm_a_name}_d{budget}") or {}
        b = payload["cells"].get(f"{arm_b_name}_d{budget}") or {}
        if not a.get("episodes") or not b.get("episodes"):
            return {"available": False}
        out = {
            "available": True,
            "episode_pool": {
                "fisher_p_one_sided": round(fisher_one_sided_greater(
                    a["survived"], a["episodes"] - a["survived"],
                    b["survived"], b["episodes"] - b["survived"]), 8),
                "table": [[a["survived"], a["episodes"] - a["survived"]],
                          [b["survived"], b["episodes"] - b["survived"]]],
                "delta": round(a["survived_fraction"] - b["survived_fraction"], 6),
            },
            "distinct": {
                "fisher_p_one_sided": round(fisher_one_sided_greater(
                    a["distinct_survived"],
                    a["distinct_trajectories"] - a["distinct_survived"],
                    b["distinct_survived"],
                    b["distinct_trajectories"] - b["distinct_survived"]), 8),
                "table": [[a["distinct_survived"],
                           a["distinct_trajectories"] - a["distinct_survived"]],
                          [b["distinct_survived"],
                           b["distinct_trajectories"] - b["distinct_survived"]]],
                "delta": round(a["distinct_fraction"] - b["distinct_fraction"], 6),
            },
        }
        return out

    payload["tests"]["P2_H2s_vs_H1_d600"] = compare("H2s", "H1", 600)
    payload["tests"]["P5_H3_vs_H2_d600"] = compare("H3", "H2", 600)
    payload["tests"]["H2_vs_H1_d600"] = compare("H2", "H1", 600)
    payload["tests"]["H2s_vs_H2_d600"] = compare("H2s", "H2", 600)
    payload["tests"]["H3_vs_H1_d600"] = compare("H3", "H1", 600)
    payload["tests"]["H2s_vs_H1_d300"] = compare("H2s", "H1", 300)

    h2s_600 = payload["cells"].get("H2s_d600") or {}
    h2_600 = payload["cells"].get("H2_d600") or {}
    p1_ok = bool(h2s_600.get("survived_fraction") is not None
                 and h2s_600["survived_fraction"] >= 0.70)
    p1_ok_distinct = bool(h2s_600.get("distinct_fraction") is not None
                          and h2s_600["distinct_fraction"] >= 0.70)
    p2 = payload["tests"]["P2_H2s_vs_H1_d600"]
    p2_ok = bool(p2.get("available")
                 and p2["episode_pool"]["fisher_p_one_sided"] < 0.05
                 and p2["episode_pool"]["delta"] > 0)
    h2s_900 = payload["cells"].get("H2s_d900") or {}
    p3_ok = bool(h2s_900.get("survived_fraction") is not None
                 and h2s_900["survived_fraction"] >= 0.15)
    payload["P1"] = {
        "arm": "H2s (stochastic candidate + R5 rule)",
        "criterion": ">= 0.70",
        "value": h2s_600.get("survived_fraction"),
        "value_distinct": h2s_600.get("distinct_fraction"),
        "pass": p1_ok, "pass_on_distinct": p1_ok_distinct,
        "cp95_episode_pool": h2s_600.get("cp95_episode_pool"),
        "cp95_distinct": h2s_600.get("cp95_distinct"),
        "n_distinct": h2s_600.get("distinct_trajectories"),
        "H2_upper_bound": {
            "note": ("H2 is deterministic (argmax candidate), so its 300-episode "
                     "cells are replays; its effective n is the number of "
                     "(seed, window) cells"),
            "value": h2_600.get("survived_fraction"),
            "value_distinct": h2_600.get("distinct_fraction"),
            "n_distinct": h2_600.get("distinct_trajectories")},
        "vs_R5_77.11": {
            "delta": (round(h2s_600["survived_fraction"] - 0.7711, 6)
                      if h2s_600.get("survived_fraction") is not None else None),
            "fisher_p_one_sided_H2s_greater_than_R5": (
                round(fisher_one_sided_greater(
                    h2s_600["survived"], h2s_600["episodes"] - h2s_600["survived"],
                    2776, 824), 8)
                if h2s_600.get("survived") is not None else None),
            "R5_note": ("R5's 77.11 % is quoted from the prereg §0 table at 3,600 "
                        "episodes; the 2,776/824 split used for the test is "
                        "77.11 % of 3,600 rounded to whole episodes"),
            "comparable": ("yes: both arms sample their actions, so both pools "
                           "are independent episodes")},
    }
    payload["P2"] = {"criterion": "H2s > H1 (one-sided Fisher p < 0.05)",
                     "detail": p2, "pass": p2_ok}
    payload["P3"] = {"arm": "H2s", "criterion": ">= 0.15",
                     "value": h2s_900.get("survived_fraction"), "pass": p3_ok}
    payload["P4"] = {
        "arm": "H2s",
        "criterion": ("R5 at 600 decisions uses all three bombs in 3,540 of 3,600 "
                      "episodes and two in 60 (v8.1.0 record); the 1.36-1.75 "
                      "figure in prereg §0 is the 300-decision budget"),
        "distribution": h2s_600.get("bombs_used_distribution"),
        "value": h2s_600.get("bombs_used_mean"),
        "H2_value": h2_600.get("bombs_used_mean")}
    payload["P5"] = {"criterion": "descriptive",
                     "detail": payload["tests"]["P5_H3_vs_H2_d600"]}
    if h2s_600.get("survived_fraction") is None:
        payload["stops"] = {"stop": "none", "why": "H2s d600 not measured yet"}
    elif p1_ok:
        payload["stops"] = {"stop": "S-deploy",
                            "why": "H2s >= 70 % at 600 decisions (P1 holds): judge "
                                   "the hybrid system deployable and re-test it at "
                                   "30/40 s next round"}
    elif h2s_600["survived_fraction"] >= 0.50:
        payload["stops"] = {"stop": "S-gap",
                            "why": "50 % <= H2s < 70 %: judge the candidate head the "
                                   "bottleneck and DAgger the no-bombs-left states"}
    else:
        payload["stops"] = {"stop": "S-rule",
                            "why": "H2s < 50 %: judge rule bombing incompatible with "
                                   "the learned dodge and return to the pure rule"}
    out = pathlib.Path(args.out) if args.out else root / "verdict.json"
    write_json(out, payload)
    log(f"verdict: P1={payload['P1']['pass']} (H2s d600="
        f"{payload['P1']['value']}), P2={payload['P2']['pass']}, "
        f"P3={payload['P3']['pass']}, P4={payload['P4']['value']}, "
        f"stop={payload['stops']['stop']} -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# mode: audio-audit / process-audit / manifest / publish
# --------------------------------------------------------------------------- #
def mode_audio_audit(args: argparse.Namespace) -> int:
    namespace = argparse.Namespace(
        work_dir=args.work_dir,
        out=args.out or str(pathlib.Path(args.work_dir) / "audio_audit.json"))
    return v450().mode_audit(namespace)


def mode_process_audit(args: argparse.Namespace) -> int:
    needle = str(args.data_dir_filter)
    out = pathlib.Path(args.out) if args.out else WORK / "process_residue.json"
    proc = subprocess.run(["pgrep", "-af", args.engine_name],
                          capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if needle in line]
    payload = {
        "schema": SCHEMA_PROC, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("this round's checklist: 0 engine processes of this round "
                        "may survive its own run"),
        "engine_name": args.engine_name, "data_dir_filter": needle,
        "pgrep_returncode": proc.returncode,
        "processes_total_by_engine_name": len(
            [line for line in proc.stdout.splitlines() if line.strip()]),
        "matching": len(lines), "lines": lines}
    payload["all_pass"] = bool(payload["matching"] == 0)
    write_json(out, payload)
    log(f"process-audit: {payload['matching']} own engine process(es) alive -> {out}")
    return 0 if payload["all_pass"] else 6


def mode_manifest(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.dir)
    if not root.is_dir():
        raise HybridError(f"not a directory: {root}")
    out = pathlib.Path(args.out) if args.out else root / "manifest.json"
    files: dict[str, Any] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root)).replace("\\", "/")
        if path.resolve() == out.resolve():
            continue
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    payload = {"schema": SCHEMA_MANIFEST, "producer": PRODUCER,
               "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "root": str(root), "file_count": len(files),
               "excludes": str(out.relative_to(root)).replace("\\", "/"),
               "total_bytes": sum(v["bytes"] for v in files.values()),
               "files": files}
    write_json(out, payload)
    log(f"manifest: {len(files)} files, {payload['total_bytes']} bytes -> {out}")
    return 0


def mode_publish(args: argparse.Namespace) -> int:
    import shutil

    root = pathlib.Path(args.work_dir)
    dest = pathlib.Path(args.dir) if args.dir else PUBLISH
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    wanted: list[tuple[pathlib.Path, str]] = [
        (root / "preflight.json", "preflight.json"),
        (root / "ckpt_assertions.json", "ckpt_assertions.json"),
        (root / "ckpt_assertions_final.json", "ckpt_assertions_final.json"),
        (root / "fidelity_d3.json", "fidelity_d3.json"),
        (root / "smoke.json", "smoke.json"),
        (root / "diagnostics.json", "diagnostics.json"),
        (root / "bomb_usage.json", "bomb_usage.json"),
        (root / "verdict.json", "verdict.json"),
        (root / "audio_audit.json", "audio_audit.json"),
        (root / "process_residue.json", "process_residue.json"),
        (root / "crosscheck.json", "crosscheck.json"),
        (root / "h1_adoption.json", "h1_adoption.json"),
        (root / "probe" / "determinism_probe.json",
         "determinism_probe.json"),
        (root / "commands.tsv", "commands.tsv"),
        (root / "contention.tsv", "contention.tsv"),
    ]
    for name in ("exec.sh", "RUN.md"):
        wanted.append((root / name, name))
    for arm in ARMS:
        for budget in BUDGETS:
            for seed in TRAIN_SEEDS:
                path = root / "eval" / f"eval_{arm}_d{budget}_seed{seed}.jsonl"
                wanted.append((path, f"eval_{arm}_d{budget}_seed{seed}.jsonl"))
                wanted.append((path.with_suffix(".summary.json"),
                               f"eval_{arm}_d{budget}_seed{seed}.summary.json"))
    for source, name in wanted:
        if source.is_file():
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append({"source": str(source), "target": str(target),
                           "sha256": sha256_file(target),
                           "bytes": target.stat().st_size})
        else:
            log(f"publish: missing {source} (skipped)")
    payload = {"schema": SCHEMA_PUBLISH, "producer": PRODUCER,
               "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "prereg": PREREG_REL, "dest": str(dest), "files": copied,
               "file_count": len(copied),
               "v8c0_hybrid_sha256": sha256_file(pathlib.Path(__file__).resolve())}
    write_json(root / "publish.json", payload)
    mode_manifest(argparse.Namespace(dir=str(dest),
                                     out=str(dest / "manifest.json")))
    log(f"publish: {len(copied)} files -> {dest}")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    global _LOG_FH

    parser = argparse.ArgumentParser(
        description="v8.12.0 HYBRID-001: hybrid deployment — the candidate "
                    "direction comes from the learned 36-action policy, the bomb "
                    "decision from the R5 rule.")
    parser.add_argument("--mode", default="preflight",
                        choices=("preflight", "assert-ckpt", "smoke", "eval",
                                 "crosscheck", "adopt-h1", "diagnostics",
                                 "verdict", "audio-audit", "process-audit",
                                 "manifest", "publish"))
    parser.add_argument("--arm", default="H2", choices=ARMS)
    parser.add_argument("--phase", default="start",
                        choices=("start", "final"))
    parser.add_argument("--decisions", type=int, default=600,
                        help="decision budget (300 / 600 / 900)")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--windows", default="",
                        help="comma separated; empty = the four dev windows")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--decision-frames", type=int, default=DECISION_FRAMES)
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--work-dir", default=str(WORK))
    parser.add_argument("--out", default="")
    parser.add_argument("--usage-out", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--dir", default="")
    parser.add_argument("--engine-name", default="th08-modern")
    parser.add_argument("--data-dir-filter", default=str(WORK))
    args = parser.parse_args(argv)

    root = pathlib.Path(args.work_dir)
    root.mkdir(parents=True, exist_ok=True)
    _LOG_FH = (root / "run.log").open("a", encoding="utf-8")

    try:
        if args.mode == "preflight":
            return mode_preflight(args)
        if args.mode == "assert-ckpt":
            return mode_assert(args)
        if args.mode == "smoke":
            return mode_smoke(args)
        if args.mode == "eval":
            return mode_eval(args)
        if args.mode == "crosscheck":
            return mode_crosscheck(args)
        if args.mode == "adopt-h1":
            return mode_adopt_h1(args)
        if args.mode == "diagnostics":
            return mode_diagnostics(args)
        if args.mode == "verdict":
            return mode_verdict(args)
        if args.mode == "audio-audit":
            return mode_audio_audit(args)
        if args.mode == "process-audit":
            return mode_process_audit(args)
        if args.mode == "manifest":
            return mode_manifest(args)
        if args.mode == "publish":
            return mode_publish(args)
        raise HybridError(f"unknown mode {args.mode!r}")
    except HybridError as exc:
        log(f"FAILED: {exc}")
        return 2
    finally:
        if _LOG_FH is not None:
            _LOG_FH.flush()


if __name__ == "__main__":
    sys.exit(main())
