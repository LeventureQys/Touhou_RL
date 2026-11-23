#!/usr/bin/env python3
"""v8.6.0 BOMBDAGGER-001 — recollect the 36-way counterfactual labels with a
behaviour policy that **actually spends bombs**, so states whose
``bombs_remaining`` is 2 or 1 enter the label set.

What this is
------------
The executor for `Document/Update/Alpha/v8.6.0/PREREGISTRATION_BOMBDAGGER_001.md`
(FROZEN 2026-11-22, before any v8.6.0 label or training run).

v8.3.0's 5,166-state label set has ``bombs_remaining = {3.0: 5166}`` — every state
still holds all three bombs, because the behaviour policy of that round was the
frozen v4.5.0 ``S`` student, which **never presses the bomb**.  Consequently the
``bomb = 1`` half of the 36 labels was only ever measured on states with three
bombs left, and no model trained on it can learn *when to save a bomb*.

This round changes **one** factor — the behaviour policy that decides which states
get labelled:

    v8.3.0 / v8.5.0   v4.5.0 `S` (never bombs)          -> bombs always 3
    v8.6.0            v4.5.0 `S` + R5 staged rule       -> bombs in {3, 2, 1}

``R5`` is v8.4.0's ``--bomb-rule staged`` (``v800_bomb.rule_gate``): the rule
threshold is 80 px while ``bombs_remaining == 3``, 60 px at 2 and 40 px at 1
(``v800_bomb.STAGED_THRESHOLDS``), the trigger is ``nearest_all_px < threshold``
and the ``bombs_remaining > 0`` gate.  The label of every collected state is the
**unchanged** 36-way counterfactual label (18 candidates x bomb / no bomb,
H = 120, ``cand_to_mask(cand, bomb)`` on the label's first frame).

    behaviour = v4.5.0 `S` round2.pt of the same seed (4abee069 / a93746bb /
                c69960fc), **stochastic** (softmax(z/1), torch.Generator
                seed = seed*1000+7), rolled out live on train windows 13-16
    states    = the live states of those rollouts, every state labelled along
                all 36 actions, every state carrying its own bombs_remaining
    model     = Policy(106, 64, 36) — the frozen 2-layer shape
    train     = BCE over the 36 outputs, Adam 1e-3, batch 256, 200 epochs,
                3 seeds, **no auxiliary loss** (prereg §1: v8.5.0 proved the
                rule-distillation term adds nothing when bombs are always 3)
    eval      = 3 budgets (300 / 600 / 900 decisions) x 3 seeds x 4 dev windows
                x 300 episodes, stochastic, live engine

Modes
-----
==================================  =============================================
``--mode preflight``                 the pinned imports (sha256 of every file this
                                     round imports **and** of the frozen prereg),
                                     the feature dim (106), the action table and
                                     the frozen design, for the record.
``--mode assert-ckpt --phase P``     the three frozen v4.5.0 ``round2.pt``
                                     (``4abee069`` / ``a93746bb`` / ``c69960fc``)
                                     that are the behaviour-policy source of the
                                     v8.6.0 states, the engine sha256 and the
                                     sha256 of every protected source.
``--mode pilot``                     the cheap design check **before** spending the
                                     label budget: roll the behaviour policy (no
                                     labels) on the 12 (seed, window) cells and
                                     report how many states per episode carry
                                     ``bombs_remaining`` 3 / 2 / 1 / 0 and how many
                                     states that projects to for a given quota.
                                     Writes ``pilot.json``; it is a *design*
                                     diagnostic, not an outcome measurement.
``--mode collect --seed S --windows W``
                                     one (seed, window) cell: RESET + the frozen
                                     8663-frame prefix + PARK, then roll the
                                     behaviour policy for up to ``--decisions``
                                     decisions per episode; at every decision the
                                     state is labelled along all 36 actions and
                                     the **parent** then executes the behaviour
                                     action (``cand_to_mask(cand, bomb)`` on the
                                     decision's first frame).  States are written
                                     with ``bombs_remaining`` and the whole
                                     behaviour record (cand, bomb, nearest_px,
                                     threshold, rule), plus the four invariant
                                     fields of the executed branch
                                     (``bomb_spent`` / ``noop_due_to_invuln`` /
                                     ``bombs_before`` / ``bombs_after``) and the
                                     18-branch ``bomb1`` block with
                                     ``branch_kind``, ``anomaly_count`` and
                                     ``anomaly_cands``.  Stops at ``--quota``
                                     states for the cell or ``--max-episodes``.
``--mode stats``                     pool the cell files into
                                     ``labels_bombdag.jsonl`` and write
                                     ``labels_stats.json``: the
                                     **``bombs_remaining`` distribution** (P0),
                                     the decision-interval distribution, the
                                     bomb-usage distribution, and the two-branch
                                     bomb=1 invariant in **both** denominators
                                     (``noop_frac`` = noop / 18-per-state
                                     bomb=1 branches, ``noop_branch_frac`` =
                                     noop / 36-per-state branches,
                                     ``noop_state_frac`` = states with >=1 noop,
                                     ``anomaly_count``).  rc 4 if P0 fails
                                     (prereg S-data: the round is void).
``--mode train --seed S``            Policy(106,64,36), BCE, Adam 1e-3, batch 256,
                                     200 epochs, no early stopping, dev never read.
``--mode eval --decisions {300,600,900} --policy-seed S``
                                     the frozen v4.5.0 episode loop with a 36-way
                                     policy: 4 dev windows, 300 episodes each,
                                     ``stochastic`` = softmax(z/1) sample.
``--mode diagnostics``               offline per-budget / per-(seed,window) counts,
                                     means, medians, the **bomb-usage table**
                                     (0/1/2/3 bombs per episode), the first and
                                     last bomb decision, side by side with v8.3.0
                                     and R5.  **No test, no interval.**
``--mode train-summary``             the three train reports side by side.
``--mode audio-audit``               AGENTS.md §9.3 on this round's own data dirs.
``--mode process-audit``             this round's own engine processes must be 0.
``--mode manifest --dir D``          sha256 + bytes of every file in D.
``--mode publish``                   copy the raw artifacts into
                                     ``Document/Update/Alpha/v8.6.0/evidence/
                                     BOMBDAGGER/`` and write the manifest.

Reused, never re-implemented (imported by path, sha256 pinned; a moved file is
exit 2 and nothing runs)
----------------------------------------------------------------------------
* ``tools/rl/v450_cf_label.py`` — ``start_parent`` (HELLO + the 2,165-checkpoint
  prefix + PARK + DEBUG), ``prepare_env`` (dummy audio, no display),
  ``prepare_data_dir`` (a freshly generated muted ``th08.cfg`` per engine start),
  ``masks_for``, ``load_policy``, ``policy_action``, ``compact_observation``,
  ``Digest``, ``mode_audit`` (the audio audit), ``check_engine``;
* ``tools/rl/v432_fork_arm.py`` — ``ForkParent`` (warm RESET, prefix walk,
  SPAWN/CHILD/QUIT), ``build_private_data_dir``, ``_step_fields``, ``fields``;
* ``tools/rl/v431_arm_a.py`` — ``build_features`` (the frozen 106-dim
  observation) and ``Policy`` (the frozen 2-layer model);
* ``tools/rl/v800_bomb.py`` — ``nearest_bullet_distances`` and ``rule_gate``
  (R5's threshold schedule; the rule is **reused**, not re-implemented);
* ``tools/rl/v830_bombpol.py`` — ``label_state36`` / ``label_child_action`` (the
  36-way labeller, itself a proven copy of v4.5.0's), ``action_masks`` /
  ``split_action`` / ``action_of``, ``policy36`` (Policy(106,64,36)),
  ``run_episode36`` (the frozen evaluation loop with the policy's bomb bit),
  ``build_arrays`` (the 106-dim / 36-label matrices with the
  ``features_sha256`` re-check);
* ``touhou_rl/v420/counterfactual._terminal_of`` — the death rule of R24/C4;
* ``touhou_rl/v420/contracts.cand_to_mask`` and ``assert_seed_allowed``.

Failure policy (fail closed where the data would be wrong, record where the
engine is merely counter-intuitive)
-------------------------------------------------------------------
* a pinned import whose sha256 moved -> exit 2, nothing runs;
* a behaviour state whose 36 labels are incomplete, or whose 18 + 18 children did
  not all start at the parked parent's frame / state -> the cell is aborted
  (rc 4) and the partial file stays ``*.partial`` (a wrong label set cannot be
  repaired afterwards);
* the **bomb = 1 invariant is two-branched, never "must spend one"**: a branch
  passes when it spends a bomb (``bombs_after == bombs_before - 1``) **or** when
  it is a no-op inside a bomb's invulnerability window (``bombs_after ==
  bombs_before`` **and** its ``alive_frames`` equal the same candidate's
  ``bomb = 0`` arm).  Bombs do not stack — a press inside the window is ignored
  by the engine — so the second branch is legal and its rows are kept: they are
  the counterfactual pairs.  A branch that fits neither is counted in
  ``bomb1_anomalies`` / ``anomaly_kinds`` and reported, **never aborting the
  cell**; ``labels_stats.json`` reports ``noop_frac`` and ``anomaly_count``.
  Each label row carries ``bomb_spent``, ``noop_due_to_invuln``, ``bombs_before``
  and ``bombs_after`` for the branch the parent executed, plus the 18-branch
  ``bomb1`` block;
* a ``bomb = 0`` path that spends a bomb is counted (``bomb0_spent_a_bomb``,
  also surfaced in ``anomaly_kinds``) and logged, not fatal;
* P0 (``bombs_remaining`` 2 and 1 each >= 200 states) failing -> rc 4 from
  ``--mode stats`` and the round stops there (prereg S-data);
* every episode record carries ``invalid`` / ``error``; an invalid episode aborts
  the cell instead of being dropped.

Not in this file
----------------
* No statistics: counts, rates, means, medians, quantiles and distributions
  only.  P1-P3 belong to the independent analysis pass (prereg §2).
* No parameter search, no auxiliary loss, no dev reading during collection or
  training.
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

WORK = REPO / ".tmp_v860_bombdag"
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.6.0" / "evidence"
           / "BOMBDAGGER")
PREREG = REPO / "Document" / "Update" / "Alpha" / "v8.6.0" / (
    "PREREGISTRATION_BOMBDAGGER_001.md")
CKPT_ROOT = REPO / ".tmp_v450_cf" / "ckpt"

# -- pinned imports: the whole round is void if any of these moved ------------
PINNED = {
    "tools/rl/v450_cf_label.py":
        "b7ab0ebeee38df4fcbb5bbbdf5e453edc8e5547abb702473b0c09cb4e90d9947",
    "tools/rl/v432_fork_arm.py":
        "33235506077308549f7657e359381d8c38b1bf45ba4ede0d802448f2dbf0cede",
    "tools/rl/v431_arm_a.py":
        "96923930fed5d1df6ea398e858aa9fb123ed39a455c3ad7e2706230432445503",
    "tools/rl/v431_dense_driver.py":
        "1079d93a46ed77c72c9d684112da9ed8d6810797f8e16fc9e789c2fed3fb4f25",
    "tools/rl/v800_bomb.py":
        "5d85bcb016a0eebbfb0fd101b2406c1841d8481468e057104b67c57120c0d7bf",
    "tools/rl/v830_bombpol.py":
        "a8c6d3065c968dbd546e9b37f389d3f786404821b8b5fa53e25d520549c58e1c",
    "touhou_rl/v420/contracts.py":
        "1357c69c32a74fda20553a3e6b290ca2102b7cc5c52e47e9fdc6eb9e08c946c1",
    "touhou_rl/v400/gym/actions.py":
        "a67145b2bec7c96b3aacaa2514e78cdadf50b056473d15b4ae8122c19c393eff",
}
SCHEMA_PREFLIGHT = "v860-bombdag-preflight/1"
SCHEMA_ASSERT = "v860-bombdag-ckpt-assertions/1"
SCHEMA_PILOT = "v860-bombdag-pilot/1"
SCHEMA_LABEL = "v860-bombdag-label/1"
SCHEMA_COLLECT = "v860-bombdag-collect/1"
SCHEMA_STATS = "v860-bombdag-labels-stats/1"
SCHEMA_TRAIN = "v860-bombdag-train/1"
SCHEMA_TRAIN_SUMMARY = "v860-bombdag-train-summary/1"
SCHEMA_EVAL_SUMMARY = "v860-bombdag-eval-summary/1"
SCHEMA_DIAG = "v860-bombdag-diagnostics/1"
SCHEMA_BOMB_USAGE = "v860-bombdag-bomb-usage/1"
SCHEMA_AUDIT = "v860-bombdag-audio-audit/1"
SCHEMA_PROC = "v860-bombdag-process-audit/1"
SCHEMA_MANIFEST = "v860-bombdag-manifest/1"
SCHEMA_PUBLISH = "v860-bombdag-publish/1"
PRODUCER = ("tools/rl/v860_bombdag.py (live engine, HarnessClient role=shadow, "
            "behaviour policy = v4.5.0 `S` stochastic + R5 staged bomb rule)")
PREREG_REL = "Document/Update/Alpha/v8.6.0/PREREGISTRATION_BOMBDAGGER_001.md"

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- frozen geometry (unchanged from v4.5.0/v6.0.0/v8.0.0/v8.3.0) -------------
PREFIX_FRAMES = 8663
CHECKPOINTS_IN_PREFIX = 2165
R24_SEED = 12

# -- frozen split (prereg §1: the v4.5.0 / v6.0.0 / v8.3.0 protocol) ----------
TRAIN_WINDOWS: tuple[int, ...] = (13, 14, 15, 16)
DEV_WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
TRAIN_SEEDS: tuple[int, ...] = (0, 1, 2)

# -- frozen design (prereg §1) ----------------------------------------------
N_CAND = 18
N_ACTIONS = 36
BOMB_VALUES: tuple[int, ...] = (0, 1)
DECISION_FRAMES = 2
DECISIONS = 300
BUDGETS: tuple[int, ...] = (300, 600, 900)
H_DEFAULT = 120
EPISODES = 300
HIDDEN = 64
FEATURE_DIM = 106
LR = 1e-3
BATCH = 256
EPOCHS = 200
TEMPERATURE = 1.0
GEN_CONST = 7
DEFAULT_TIMEOUT = 300.0
P0_MIN_STATES = 200               # prereg §2 P0: bombs 2 and 1 each >= 200
QUOTA_DEFAULT = 420               # per (seed, window) cell; 12 x 420 = 5,040
MAX_EPISODES_DEFAULT = 10         # per cell
STATE_KEYS = ("frame", "player_x", "player_y", "player_power", "lives_remaining",
              "bombs_remaining", "graze", "deaths", "stage", "score")

# -- the behaviour policy: v4.5.0 `S` (the frozen DAgger student) ------------
FROZEN_ROUND2_SHA256: dict[int, str] = {
    0: "4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc",
    1: "a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4",
    2: "c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01",
}
PROTECTED_FILES = (
    "tools/rl/v432_fork_arm.py",
    "tools/rl/v450_cf_label.py",
    "tools/rl/v431_arm_a.py",
    "tools/rl/v800_bomb.py",
    "tools/rl/v830_bombpol.py",
    "tools/rl/v850_distill.py",
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)

# -- R5: v8.4.0 BOMBRULE2-001 §1's staged rule (reused through v800_bomb) -----
R5_RULE = "staged"
R5_THRESHOLDS: dict[int, float] = {3: 80.0, 2: 60.0, 1: 40.0}
R5_SOURCE = ("Document/Update/Alpha/v8.4.0/PREREGISTRATION_BOMBRULE2_001.md §1 "
             "(implemented in tools/rl/v800_bomb.py::rule_gate / "
             "STAGED_THRESHOLDS)")

# -- the two references this round is placed beside (numbers only, no test) --
BASELINE_V830 = {
    "arm": "v8.3.0 BOMBPOL-001 — behaviour policy v4.5.0 `S` (never bombs), the "
           "same 36-way labels on 5,166 states whose bombs_remaining is always 3",
    "episodes_per_budget": 3600,
    "survived": {300: 3598, 600: 700, 900: 22},
    "rate": {300: 0.999444, 600: 0.194444, 900: 0.006111},
    "bombs_used_per_episode": {"3": 3600},
    "first_bomb_decision_median": 0.0,
    "source": ("Document/Update/Alpha/v8.3.0/evidence/BOMBPOL/diagnostics.json "
               "(budgets 300/600/900)"),
}
RULE_R5 = {
    "arm": "v8.4.0 R5 `staged` — script rule: bombs 3 -> 80 px, 2 -> 60 px, "
           "1 -> 40 px, `nearest_all_px < threshold`, bombs > 0",
    "episodes_per_budget": 3600,
    "rate": {300: 1.0, 600: 0.7705, 900: 0.1938},
    "source": ("quoted from Document/Update/Alpha/v8.6.0/"
               "PREREGISTRATION_BOMBDAGGER_001.md §1's table (100 / 77.05 / "
               "19.38).  Cross-reference, not recomputed here: "
               "Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/RUN.md §0/§3 "
               "measured R5 at 600 decisions as 77.1111 % (2,776/3,600), 0.06 pp "
               "above the quoted 77.05 %; the published v8.4.0 tree carries no "
               "300/900-decision R5 table, so those two remain quoted only."),
}

_LOG_FH: Any = None
_MODS: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class BombDagError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v860] {message}"
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
        raise BombDagError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cached(name: str, path: pathlib.Path):
    module = _MODS.get(name)
    if module is None:
        module = load_module(name, path)
        module._LOG_FH = _LOG_FH
        _MODS[name] = module
    return module


def v450():
    """``tools/rl/v450_cf_label.py`` — imported, pinned, never modified."""

    module = _cached("v450_cf_label", REPO / "tools" / "rl" / "v450_cf_label.py")
    module.m432()._LOG_FH = _LOG_FH
    return module


def m432():
    return _cached("v432_fork_arm", REPO / "tools" / "rl" / "v432_fork_arm.py")


def arm_a():
    return _cached("v431_arm_a", REPO / "tools" / "rl" / "v431_arm_a.py")


def m800():
    return _cached("v800_bomb", REPO / "tools" / "rl" / "v800_bomb.py")


def v830():
    """``tools/rl/v830_bombpol.py`` — the proven 36-way labeller and eval loop."""

    return _cached("v830_bombpol", REPO / "tools" / "rl" / "v830_bombpol.py")


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


def write_jsonl_line(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def first_bomb_histogram(values: Sequence[int]) -> dict[str, int]:
    edges = ((0, 0), (1, 4), (5, 9), (10, 24), (25, 49), (50, 99), (100, 199),
             (200, 10 ** 9))
    out: dict[str, int] = {}
    for low, high in edges:
        label = (str(low) if low == high else
                 (f">={low}" if high >= 10 ** 9 else f"{low}-{high}"))
        out[label] = sum(1 for v in values if low <= int(v) <= high)
    return out


DECISION_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 24), (25, 49), (50, 99), (100, 149), (150, 199), (200, 249), (250, 299))


def decision_bucket(decision: int) -> str:
    for low, high in DECISION_BUCKETS:
        if low <= int(decision) <= high:
            return f"{low}-{high}"
    return f">={DECISION_BUCKETS[-1][1] + 1}"


def decision_histogram(values: Sequence[int]) -> dict[str, int]:
    out = {f"{low}-{high}": 0 for low, high in DECISION_BUCKETS}
    for value in values:
        out[decision_bucket(int(value))] = out.get(decision_bucket(int(value)), 0) + 1
    return out


def bombs_histogram(values: Sequence[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return out


def features_sha256(feats: Any) -> str:
    import numpy as np

    array = np.asarray(feats, dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def generator_seed(seed: int) -> int:
    """v4.5.0 §2.3's rule, verbatim: ``torch.Generator().manual_seed(seed*1000+7)``."""

    return int(seed) * 1000 + GEN_CONST


def check_pins() -> dict[str, Any]:
    out: dict[str, Any] = {"pinned": {}, "all_match": True,
                           "prereg": {}, "self": {}}
    for rel, expected in PINNED.items():
        path = REPO / rel
        got = sha256_file(path) if path.is_file() else None
        match = got == expected
        out["pinned"][rel] = {"sha256": got, "expected": expected, "match": match}
        if not match:
            out["all_match"] = False
    out["prereg"] = {"path": str(PREREG), "exists": PREREG.is_file(),
                     "sha256": sha256_file(PREREG) if PREREG.is_file() else None,
                     "bytes": PREREG.stat().st_size if PREREG.is_file() else None}
    if not out["prereg"]["exists"]:
        out["all_match"] = False
    me = pathlib.Path(__file__).resolve()
    out["self"] = {"path": str(me), "sha256": sha256_file(me),
                   "bytes": me.stat().st_size}
    return out


# --------------------------------------------------------------------------- #
# behaviour policy: v4.5.0 `S` (stochastic) + R5 (v800_bomb.rule_gate)
# --------------------------------------------------------------------------- #
def behaviour_ckpt(seed: int) -> pathlib.Path:
    return CKPT_ROOT / f"seed{int(seed)}" / "round2.pt"


def load_behaviour_policy(seed: int) -> tuple[Any, pathlib.Path, str]:
    """The frozen v4.5.0 ``S`` student of this seed, verified against its sha256."""

    path = behaviour_ckpt(seed)
    if not path.is_file():
        raise BombDagError(f"behaviour checkpoint not found: {path}")
    digest = sha256_file(path)
    expected = FROZEN_ROUND2_SHA256[int(seed)]
    if digest != expected:
        raise BombDagError(f"behaviour checkpoint {path} sha256 {digest[:16]} != "
                           f"the frozen {expected[:16]}")
    return v450().load_policy(path, seed=int(seed)), path, digest


def behaviour_decision(*, state: dict[str, Any], hazards: Any, policy: Any,
                       generator: Any, decision: int) -> dict[str, Any]:
    """One behaviour decision: the S candidate plus R5's bomb bit.

    The candidate comes from the frozen 18-way policy (``stochastic``, the same
    sampler v4.5.0's evaluation used); the bomb bit comes from **R5 as
    implemented in v800_bomb** — ``rule_gate("staged", ...)`` gives the
    per-``bombs_remaining`` threshold (80/60/40 px) and the trigger is
    ``nearest_all_px < threshold`` with ``bombs_remaining > 0``.
    """

    arm = arm_a()
    feats, _detail = arm.build_features(state=state, hazards=hazards,
                                        decision_index=int(decision))
    cand = v450().policy_action(policy, feats, "stochastic", generator)
    distances = m800().nearest_bullet_distances(state, hazards)
    nearest = distances["nearest_all_px"]
    bombs_before = int(state["bombs_remaining"] or 0)
    player_x = float(state.get("player_x") or 0.0)
    gate = m800().rule_gate(R5_RULE, decision=int(decision),
                            bombs_before=int(bombs_before), player_x=player_x)
    threshold = gate["threshold_px"]
    fired = bool(bombs_before > 0 and nearest is not None
                 and float(nearest) < float(threshold))
    return {"feats": feats, "features_sha256_k12": features_sha256(feats),
            "cand": int(cand), "bomb": int(fired),
            "action": int(cand) * 2 + int(fired),
            "nearest_all_px": nearest,
            "nearest_lethal_px": distances["nearest_lethal_px"],
            "bullets": distances["bullets"],
            "bullets_within_60": distances["bullets_within_60"],
            "bombs_before": int(bombs_before), "threshold_px": threshold,
            "rule": R5_RULE, "distances": distances}


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def cell_labels_path(root: pathlib.Path, seed: int, window: int) -> pathlib.Path:
    return root / "labels" / f"seed{int(seed)}_w{int(window)}.jsonl"


def cell_summary_path(root: pathlib.Path, seed: int, window: int) -> pathlib.Path:
    return root / "collect" / f"seed{int(seed)}_w{int(window)}.json"


def pooled_labels_path(root: pathlib.Path) -> pathlib.Path:
    return root / "labels_bombdag.jsonl"


def pooled_stats_path(root: pathlib.Path) -> pathlib.Path:
    return root / "labels_stats.json"


def policy_ckpt_path(root: pathlib.Path, seed: int) -> pathlib.Path:
    return root / "ckpt" / f"seed{int(seed)}" / "bombdag.pt"


def train_report_path(root: pathlib.Path, seed: int) -> pathlib.Path:
    return root / "train" / f"seed{int(seed)}.json"


def eval_path(root: pathlib.Path, budget: int, seed: int) -> pathlib.Path:
    return root / "eval" / f"eval_d{int(budget)}_seed{int(seed)}.jsonl"


# --------------------------------------------------------------------------- #
# engine / data dir plumbing (all delegated to the frozen modules)
# --------------------------------------------------------------------------- #
def engine_gate(path: str) -> dict[str, Any]:
    return dict(v450().check_engine(path))


def prepare_env() -> dict[str, Any]:
    return dict(v450().prepare_env())


def start_parent(*, slot: int, window: int, engine: str, data_dir: pathlib.Path,
                 log_dir: pathlib.Path) -> tuple[Any, dict[str, Any]]:
    return v450().start_parent(slot=int(slot), window=int(window), engine=engine,
                               data_dir=data_dir, log_dir=log_dir)


def parse_windows(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(text).split(",") if part.strip())


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
        "prereg": {"path": PREREG_REL, "frozen": True,
                   "single_factor": ("the behaviour policy that decides which states "
                                     "get labelled: v4.5.0 `S` only (bombs always 3) "
                                     "-> v4.5.0 `S` + R5 staged rule (bombs 3/2/1)")},
        "pins": pins, "engine": engine, "launch_env": prepare_env(),
        "feature_dim": int(arm_a().feature_dim()),
        "action_space": {"candidates": N_CAND, "bombs": list(BOMB_VALUES),
                         "actions": N_ACTIONS,
                         "encoding": "action = cand * 2 + bomb",
                         "bomb1_semantics": ("cand_to_mask(cand, 1) on the label's "
                                             "first frame, cand_to_mask(cand, 0) "
                                             "afterwards (v8.3.0 §1, unchanged)")},
        "behaviour": {
            "policy": "v4.5.0 `S` — .tmp_v450_cf/ckpt/seed{0,1,2}/round2.pt",
            "kind": "stochastic", "temperature": TEMPERATURE,
            "generator": "torch.Generator().manual_seed(seed*1000+7)",
            "checkpoints": {str(s): {"path": str(behaviour_ckpt(s)),
                                     "expected": FROZEN_ROUND2_SHA256[s]}
                            for s in TRAIN_SEEDS},
            "bomb_rule": R5_RULE, "thresholds_px": {str(k): v for k, v in
                                                    R5_THRESHOLDS.items()},
            "bomb_rule_source": R5_SOURCE,
            "bomb_rule_gate": ("nearest_all_px < threshold and bombs_remaining > 0"),
        },
        "frozen": {"TRAIN_WINDOWS": list(TRAIN_WINDOWS),
                   "DEV_WINDOWS": list(DEV_WINDOWS),
                   "train_seeds": list(TRAIN_SEEDS),
                   "H": H_DEFAULT, "budgets_decisions": list(BUDGETS),
                   "decisions_cap": DECISIONS, "decision_frames": DECISION_FRAMES,
                   "episodes_per_window": EPISODES,
                   "lr": LR, "batch": BATCH, "epochs": EPOCHS, "hidden": HIDDEN,
                   "p0_min_states": P0_MIN_STATES, "quota_default": QUOTA_DEFAULT,
                   "max_episodes_default": MAX_EPISODES_DEFAULT},
        "references": {"v830_bombpol": BASELINE_V830, "rule_r5": RULE_R5},
    }
    ok = bool(pins["all_match"]) and bool(engine.get("match"))
    rec["all_pass"] = ok
    out = pathlib.Path(args.out) if args.out else root / "preflight.json"
    write_json(out, rec)
    log(f"preflight: pins={pins['all_match']} engine={engine.get('match')} "
        f"feature_dim={rec['feature_dim']} all_pass={ok} -> {out}")
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
        "requirement": ("checklist item 1: the behaviour policy of every v8.6.0 "
                        "state is the frozen v4.5.0 `S` student, whose final "
                        "checkpoints are round2.pt = 4abee069… / a93746bb… / "
                        "c69960fc…; the protected sources must not have moved"),
        "engine": None, "ckpts": {}, "protected": {}, "v860_bombdag_py": {},
        "prereg": {}, "all_pass": False,
    }
    try:
        payload["engine"] = engine_gate(args.engine)
        ok = bool(payload["engine"].get("match"))
        for seed in TRAIN_SEEDS:
            path = behaviour_ckpt(seed)
            expected = FROZEN_ROUND2_SHA256[seed]
            digest = sha256_file(path) if path.is_file() else None
            match = bool(digest == expected)
            ok = ok and match
            payload["ckpts"][str(seed)] = {
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
        me = pathlib.Path(__file__).resolve()
        payload["v860_bombdag_py"] = {"path": str(me), "sha256": sha256_file(me)}
        payload["prereg"] = {"path": PREREG_REL,
                            "sha256": sha256_file(PREREG) if PREREG.is_file()
                            else None}
        ok = ok and PREREG.is_file()
        payload["all_pass"] = bool(ok)
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
    write_json(out_path, payload)
    log(f"assert-ckpt {args.phase}: all_pass={payload['all_pass']} -> {out_path}")
    return 0 if payload["all_pass"] else 2


# --------------------------------------------------------------------------- #
# mode: pilot (design check: how many states per bombs level does the roll-out
# produce?  no labels, no label budget spent)
# --------------------------------------------------------------------------- #
def mode_pilot(args: argparse.Namespace) -> int:
    import torch

    from touhou_rl.v400.gym import protocol as P
    from touhou_rl.v420 import counterfactual as cf

    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    windows = parse_windows(args.windows)
    episodes = int(args.episodes)
    decisions_cap = int(args.decisions)
    if seed not in TRAIN_SEEDS:
        raise BombDagError(f"--seed must be one of {TRAIN_SEEDS}, got {seed}")
    for window in windows:
        if window not in TRAIN_WINDOWS:
            raise BombDagError(f"window {window} is not a train window "
                               f"{TRAIN_WINDOWS}")
    pins = check_pins()
    if not pins["all_match"]:
        raise BombDagError(f"pinned import moved: {json.dumps(pins)[:800]}")
    engine = engine_gate(args.engine)
    tag = args.tag or f"v860-pilot-s{seed}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    out = pathlib.Path(args.out) if args.out else root / "pilot.json"

    policy, ckpt, ckpt_sha = load_behaviour_policy(seed)
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    rec: dict[str, Any] = {
        "schema": SCHEMA_PILOT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "purpose": ("design check before the label budget is spent: the bombs "
                    "distribution of the behaviour roll-out (no labels) and the "
                    "states it projects to for a given quota"),
        "seed": seed, "windows": list(windows), "episodes_per_window": episodes,
        "decisions_cap": decisions_cap,
        "behaviour": {"policy": str(ckpt), "sha256": ckpt_sha,
                      "kind": "stochastic",
                      "generator_seed": generator_seed(seed),
                      "bomb_rule": R5_RULE,
                      "thresholds_px": {str(k): v for k, v in
                                        R5_THRESHOLDS.items()}},
        "pins": pins, "engine": engine, "launch_env": prepare_env(),
        "cells": {}, "bombs_remaining_counts": {}, "status": "failed",
        "error": None, "wall_s": None,
    }
    started = time.perf_counter()
    parents: list[Any] = []
    try:
        for slot, window in enumerate(windows):
            masks, hashes, _meta = v450().masks_for(window)
            data_dir, data_info = v450().prepare_data_dir(
                data_root, window, args.data_source)
            parent, gate = start_parent(slot=slot, window=window,
                                        engine=args.engine, data_dir=data_dir,
                                        log_dir=run_dir)
            parents.append(parent)
            cell = {"window": int(window), "episodes": [],
                    "bombs_remaining_counts": {}, "decisions": 0,
                    "bomb_emits": 0, "deaths": 0,
                    "cfg_muted": data_info["cfg"]["muted_four_zero"],
                    "gate": {key: gate[key] for key in (
                        "hello_role_shadow", "park_ok", "park_frame",
                        "threads_1", "all_pass")}}
            state, _reply = parent.state()
            if int(state["frame"]) != PREFIX_FRAMES:
                raise BombDagError(f"w{window}: parked parent at frame "
                                   f"{state['frame']}, expected {PREFIX_FRAMES}")
            for episode in range(episodes):
                deaths0 = float(state["deaths"] or 0.0)
                stage0 = int(state["stage"] or 0)
                ep = {"episode": int(episode), "decisions": 0, "frames": 0,
                      "died": False, "terminal": None,
                      "bombs_start": int(state["bombs_remaining"] or 0),
                      "bombs_end": None, "bombs_remaining_counts": {},
                      "bomb_emits": 0, "thresholds_used": {},
                      "first_bomb_decision": None,
                      "last_bomb_decision": None,
                      "nearest_px_first": None}
                if episode > 0:
                    prefix = parent.walk_prefix(masks, hashes, window)
                    parent.park()
                    state, _reply = parent.state()
                    if int(state["frame"]) != PREFIX_FRAMES:
                        raise BombDagError(f"w{window} ep{episode}: warm RESET left "
                                           f"the parent at frame {state['frame']}")
                    deaths0 = float(state["deaths"] or 0.0)
                    stage0 = int(state["stage"] or 0)
                    ep["bombs_start"] = int(state["bombs_remaining"] or 0)
                    ep["prefix"] = {"hash_checks": prefix["hash_checks"],
                                    "hash_mismatches": prefix["hash_mismatches"],
                                    "final_frame": prefix["final_frame"]}
                died = False
                for decision in range(decisions_cap):
                    hazards = P.parse_hazards_reply(parent.ask("HAZARDS"))
                    got = behaviour_decision(state=state, hazards=hazards,
                                             policy=policy, generator=generator,
                                             decision=decision)
                    if ep["nearest_px_first"] is None:
                        ep["nearest_px_first"] = got["nearest_all_px"]
                    key = str(int(got["bombs_before"]))
                    ep["bombs_remaining_counts"][key] = (
                        ep["bombs_remaining_counts"].get(key, 0) + 1)
                    cell["bombs_remaining_counts"][key] = (
                        cell["bombs_remaining_counts"].get(key, 0) + 1)
                    rec["bombs_remaining_counts"][key] = (
                        rec["bombs_remaining_counts"].get(key, 0) + 1)
                    if got["bomb"]:
                        ep["bomb_emits"] += 1
                        cell["bomb_emits"] += 1
                        tkey = f"{float(got['threshold_px']):g}"
                        ep["thresholds_used"][tkey] = (
                            ep["thresholds_used"].get(tkey, 0) + 1)
                        if ep["first_bomb_decision"] is None:
                            ep["first_bomb_decision"] = int(decision)
                        ep["last_bomb_decision"] = int(decision)
                    first_mask, rest_mask = v830().action_masks(got["action"])
                    for frame_index in range(DECISION_FRAMES):
                        step = parent.step(first_mask if frame_index == 0
                                           else rest_mask)
                        if step["advanced"] == 1:
                            ep["frames"] += 1
                        reading, _reply = parent.state()
                        term = cf._terminal_of(
                            reading, deaths_start=deaths0, stage0=stage0,
                            hash_hex=(step["hash"] or ""),
                            frame=int(reading["frame"]), exited=step["exited"],
                            game_over=step["game_over"])
                        state = reading
                        if term is not None:
                            ep["terminal"] = str(term.get("terminal"))
                            if term.get("engine_dead"):
                                ep["died"] = True
                                cell["deaths"] += 1
                            died = True
                            break
                    ep["decisions"] += 1
                    cell["decisions"] += 1
                    if died:
                        break
                ep["bombs_end"] = int(state["bombs_remaining"] or 0)
                cell["episodes"].append(ep)
                if not died and ep["decisions"] >= decisions_cap:
                    # the episode walked the cap; the next one needs a warm RESET
                    pass
            rec["cells"][str(window)] = cell
            parent.close()
            parents.remove(parent)
            log(f"pilot w{window}: {cell['decisions']} decisions, bombs "
                f"{cell['bombs_remaining_counts']}, emits={cell['bomb_emits']}, "
                f"deaths={cell['deaths']}")
        rec["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["status"] = "failed"
        log(f"PILOT FAILED: {rec['error']}")
    finally:
        for parent in parents:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        rec["wall_s"] = round(time.perf_counter() - started, 3)
        total = sum(rec["bombs_remaining_counts"].values())
        rec["states_total"] = int(total)
        rec["fractions"] = {key: round(value / total, 6)
                            for key, value in rec["bombs_remaining_counts"].items()
                            } if total else {}
        rec["episodes_total"] = sum(len(v["episodes"])
                                    for v in rec["cells"].values())
        rec["projection"] = {
            "note": ("states with bombs 2 / 1 per 1,000 roll-out states; a quota of "
                     "N states yields N * fraction"),
            "per_1000_states": {key: round(1000.0 * value / total, 1)
                                for key, value in rec["bombs_remaining_counts"].items()
                                } if total else {},
            "states_for_200_each": (
                None if not total else {
                    "bombs2": int(math.ceil(
                        P0_MIN_STATES / max(rec["fractions"].get("2", 0.0), 1e-9))),
                    "bombs1": int(math.ceil(
                        P0_MIN_STATES / max(rec["fractions"].get("1", 0.0), 1e-9)))})}
        write_json(out, rec)
    log(f"pilot: {rec['status']} states={rec.get('states_total')} "
        f"bombs={rec['bombs_remaining_counts']} wall={rec['wall_s']}s -> {out}")
    return 0 if rec["status"] == "complete" else 4


# --------------------------------------------------------------------------- #
# mode: collect (one (seed, window) cell: roll the behaviour policy and label
# every state along all 36 actions)
# --------------------------------------------------------------------------- #
def mode_collect(args: argparse.Namespace) -> int:
    import torch

    from touhou_rl.v400.gym import protocol as P
    from touhou_rl.v400.labels.state import player_state_name
    from touhou_rl.v420 import counterfactual as cf

    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    windows = parse_windows(args.windows)
    horizon = int(args.horizon)
    decisions_cap = int(args.decisions)
    quota = int(args.quota)
    max_episodes = int(args.max_episodes)
    if seed not in TRAIN_SEEDS:
        raise BombDagError(f"--seed must be one of {TRAIN_SEEDS}, got {seed}")
    if horizon != H_DEFAULT:
        raise BombDagError(f"prereg §1 fixes H = {H_DEFAULT} for every label; got "
                           f"{horizon}")
    if decisions_cap != DECISIONS:
        raise BombDagError(f"the collection roll-out is capped at {DECISIONS} "
                           f"decisions per episode (the frozen protocol); got "
                           f"{decisions_cap}")
    for window in windows:
        if window not in TRAIN_WINDOWS:
            raise BombDagError(f"window {window} is not a train window "
                               f"{TRAIN_WINDOWS}")
    pins = check_pins()
    if not pins["all_match"]:
        raise BombDagError(f"pinned import moved: {json.dumps(pins)[:800]}")
    engine = engine_gate(args.engine)
    policy, ckpt, ckpt_sha = load_behaviour_policy(seed)
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    launch_env = prepare_env()

    for window in windows:
        tag = args.tag or f"v860-collect-s{seed}-w{window}"
        run_dir = root / "runs" / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        data_root = root / "data" / tag
        labels = (pathlib.Path(args.labels_out) if args.labels_out and len(windows) == 1
                  else cell_labels_path(root, seed, window))
        labels.parent.mkdir(parents=True, exist_ok=True)
        partial = labels.with_suffix(labels.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        summary_path = (pathlib.Path(args.summary_out)
                        if args.summary_out and len(windows) == 1
                        else cell_summary_path(root, seed, window))
        masks, hashes, _meta = v450().masks_for(window)
        rec: dict[str, Any] = {
            "schema": SCHEMA_COLLECT, "producer": PRODUCER,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "prereg": PREREG_REL, "seed": seed, "window": int(window),
            "quota": quota, "decisions_cap": decisions_cap,
            "max_episodes": max_episodes, "H": horizon,
            "actions_per_state": N_ACTIONS,
            "behaviour": {
                "policy": "v4.5.0 `S` (frozen DAgger student, round2.pt)",
                "ckpt": str(ckpt), "ckpt_sha256": ckpt_sha,
                "kind": "stochastic",
                "generator_seed": generator_seed(seed),
                "bomb_rule": R5_RULE,
                "thresholds_px": {str(k): v for k, v in R5_THRESHOLDS.items()},
                "bomb_rule_gate": ("nearest_all_px < threshold and "
                                   "bombs_remaining > 0"),
                "bomb_rule_source": R5_SOURCE,
                "mask_schedule": ("cand_to_mask(cand, bomb) on the decision's "
                                  "first frame, cand_to_mask(cand, 0) afterwards")},
            "bomb1_semantics": ("cand_to_mask(cand, 1) on the label's first frame, "
                               "cand_to_mask(cand, 0) on frames 2..H (v8.3.0 §1)"),
            "pins": pins, "engine": engine, "launch_env": launch_env,
            "labels_path": str(labels), "status": "failed", "error": None,
            "episodes": [], "states": 0, "bombs_remaining_counts": {},
            "bomb_emits": 0, "thresholds_used": {}, "decision_counts": {},
            "label_checks": {"labels_36": 0, "labels_missing": 0,
                             "baseline_frame_ok_total": 0,
                             "baseline_state_equal_parent_total": 0,
                             "spawn_ok_total": 0, "quit_ok_total": 0,
                             "bomb0_spent_a_bomb": 0,
                             "bomb1_press_honoured": 0,
                             "bomb1_press_ignored": 0,
                             "bomb1_noop_due_to_invuln": 0,
                             "bomb1_unspent_alive_differs": 0,
                             "bomb1_anomalies": 0,
                             "states_with_bomb1_anomaly": 0,
                             "anomaly_kinds": {},
                             "states_all_bomb1_noop": 0,
                             "states_some_bomb1_noop": 0,
                             "states_no_bomb1_noop": 0,
                             "noop_by_player_state": {},
                             "states_by_player_state": {},
                             "bomb1_decrement_frames": {},
                             "states_without_a_bomb_left": 0},
            "gate": {}, "cfg": {}, "wall_s": None, "state_costs_s": [],
        }
        handle = partial.open("w", encoding="utf-8")
        parents: list[Any] = []
        started = time.perf_counter()
        try:
            data_dir, data_info = v450().prepare_data_dir(
                data_root, window, args.data_source)
            rec["cfg"] = data_info["cfg"]
            rec["data_dir"] = str(data_dir)
            parent, gate = start_parent(slot=0, window=window,
                                        engine=args.engine, data_dir=data_dir,
                                        log_dir=run_dir)
            parents.append(parent)
            rec["gate"] = {key: gate[key] for key in (
                "hello_role_shadow", "park_ok", "park_frame", "threads_1",
                "all_pass")}
            rec["gate"]["prefix"] = {key: gate["prefix"][key] for key in (
                "hash_checks", "hash_mismatches", "final_frame", "wall_s")}
            state, _reply = parent.state()
            if int(state["frame"]) != PREFIX_FRAMES:
                raise BombDagError(f"w{window}: parked parent at frame "
                                   f"{state['frame']}, expected {PREFIX_FRAMES}")
            deaths0 = float(state["deaths"] or 0.0)
            stage0 = int(state["stage"] or 0)
            episode = 0
            while rec["states"] < quota and episode < max_episodes:
                ep: dict[str, Any] = {
                    "episode": int(episode), "decisions": 0, "frames": 0,
                    "died": False, "terminal": None,
                    "bombs_start": int(state["bombs_remaining"] or 0),
                    "bombs_end": None, "bombs_used": None, "bomb_emits": 0,
                    "states_labelled": 0,
                    "bombs_remaining_counts": {}, "first_bomb_decision": None,
                    "last_bomb_decision": None, "prefix": None}
                if episode > 0:
                    prefix = parent.walk_prefix(masks, hashes, window)
                    parent.park()
                    state, _reply = parent.state()
                    if int(state["frame"]) != PREFIX_FRAMES:
                        raise BombDagError(
                            f"w{window} ep{episode}: warm RESET left the parent at "
                            f"frame {state['frame']}")
                    deaths0 = float(state["deaths"] or 0.0)
                    stage0 = int(state["stage"] or 0)
                    ep["bombs_start"] = int(state["bombs_remaining"] or 0)
                    ep["prefix"] = {"hash_checks": prefix["hash_checks"],
                                    "hash_mismatches": prefix["hash_mismatches"],
                                    "final_frame": prefix["final_frame"]}
                died = False
                for decision in range(decisions_cap):
                    if rec["states"] >= quota:
                        break
                    state_frame = int(state["frame"])
                    hazards = P.parse_hazards_reply(parent.ask("HAZARDS"))
                    got = behaviour_decision(state=state, hazards=hazards,
                                             policy=policy, generator=generator,
                                             decision=decision)
                    key = str(int(got["bombs_before"]))
                    ep["bombs_remaining_counts"][key] = (
                        ep["bombs_remaining_counts"].get(key, 0) + 1)
                    rec["bombs_remaining_counts"][key] = (
                        rec["bombs_remaining_counts"].get(key, 0) + 1)
                    rec["decision_counts"][decision_bucket(decision)] = (
                        rec["decision_counts"].get(decision_bucket(decision), 0) + 1)
                    if got["bomb"]:
                        ep["bomb_emits"] += 1
                        rec["bomb_emits"] += 1
                        tkey = f"{float(got['threshold_px']):g}"
                        rec["thresholds_used"][tkey] = (
                            rec["thresholds_used"].get(tkey, 0) + 1)
                        if ep["first_bomb_decision"] is None:
                            ep["first_bomb_decision"] = int(decision)
                        ep["last_bomb_decision"] = int(decision)
                    park_reply = parent.park()
                    got_frame = m432().fields(park_reply).get("frame")
                    if got_frame != str(state_frame):
                        raise BombDagError(
                            f"w{window} d{decision}: PARK reports frame {got_frame}, "
                            f"the read state is at {state_frame}")
                    block, _children = v830().label_state36(
                        parent, state=state, horizon=horizon,
                        park_reply=park_reply)
                    # ---- fail-closed checks on the 36 labels of this state ----
                    if len(block["safe"]) != N_ACTIONS:
                        raise BombDagError(f"w{window} d{decision}: "
                                           f"{len(block['safe'])} labels")
                    if any(value is None for value in block["safe"]):
                        raise BombDagError(f"w{window} d{decision}: a label is None")
                    for action in range(N_ACTIONS):
                        safe = int(block["safe"][action])
                        alive = int(block["alive_frames"][action])
                        if (safe == 1) != (alive == horizon):
                            raise BombDagError(
                                f"w{window} d{decision} a{action}: safe={safe} but "
                                f"alive_frames={alive} at H={horizon}")
                    kids = block["children"]
                    if int(kids["baseline_frame_ok"]) != N_ACTIONS:
                        raise BombDagError(
                            f"w{window} d{decision}: only "
                            f"{kids['baseline_frame_ok']}/{N_ACTIONS} children started "
                            f"at the parked parent's frame")
                    if int(kids["spawn_ok"]) != N_ACTIONS \
                            or int(kids["quit_ok"]) != N_ACTIONS:
                        raise BombDagError(
                            f"w{window} d{decision}: spawn/quit "
                            f"{kids['spawn_ok']}/{kids['quit_ok']} of {N_ACTIONS}")
                    if int(kids["baseline_state_equal_parent"]) != N_ACTIONS:
                        raise BombDagError(
                            f"w{window} d{decision}: only "
                            f"{kids['baseline_state_equal_parent']}/{N_ACTIONS} "
                            f"children baseline states equal the parent's")
                    bombs_before = int(got["bombs_before"])
                    decrements = block["bomb_decrement_frame"]
                    bomb1_actions = [a for a in range(N_ACTIONS)
                                     if int(v830().split_action(a)[1]) == 1]
                    bomb0_actions = [a for a in range(N_ACTIONS)
                                     if int(v830().split_action(a)[1]) == 0]
                    # ------------------------------------------------------------------
                    # the two-branch bomb=1 invariant of BOMBDAGGER-001.
                    # ENGINE SEMANTICS (measured in v8.0.0 / v8.3.0 / v8.4.0 and again
                    # in v8.6.0, and the reason the first v8.6.0 collect aborted):
                    # **bombs do not stack** — pressing BTN_BOMB inside the
                    # invulnerability window of the previous bomb is ignored by the
                    # engine and spends nothing.  "bomb=1 must spend one bomb" is
                    # therefore FALSE inside that window.  A bomb=1 branch passes iff
                    #   (a) bombs_after == bombs_before - 1          (press honoured)
                    #   OR
                    #   (b) bombs_after == bombs_before AND its alive_frames equal the
                    #       same candidate's bomb=0 arm (action = cand*2+bomb, so the
                    #       sibling is action-1)                (no-op in the window).
                    # Anything else is counted as an **anomaly** and recorded, never
                    # aborted: the pair (bomb=0, bomb=1) is a legitimate counterfactual
                    # and dropping or aborting on it would bias the label set.
                    # ------------------------------------------------------------------
                    branches: dict[int, dict[str, Any]] = {}
                    for action in bomb1_actions:
                        before = int(block["bombs_start"][action])
                        after = int(block["bombs_end"][action])
                        alive1 = int(block["alive_frames"][action])
                        alive0 = int(block["alive_frames"][action - 1])
                        spent = bool(after == before - 1)
                        noop = bool(not spent and after == before and alive1 == alive0)
                        if spent:
                            kind = "spent"
                        elif noop:
                            kind = "noop_due_to_invuln"
                        elif after != before:
                            kind = "anomaly_counter_changed_not_minus_one"
                        else:
                            kind = "anomaly_counter_unchanged_alive_differs"
                        anomaly = bool(not spent and not noop)
                        if anomaly and decrements[action] is not None:
                            kind = "anomaly_press_seen_but_counter_restored"
                        branches[action] = {
                            "cand": int(v830().split_action(action)[0]),
                            "before": before, "after": after,
                            "alive": alive1, "alive_bomb0": alive0,
                            "decrement_frame": decrements[action],
                            "decrement_observed": decrements[action] is not None,
                            "spent": spent, "noop": noop,
                            "anomaly": anomaly, "kind": kind}
                    b1_before = [branches[a]["before"] for a in bomb1_actions]
                    b1_after = [branches[a]["after"] for a in bomb1_actions]
                    b1_alive = [branches[a]["alive"] for a in bomb1_actions]
                    b1_alive0 = [branches[a]["alive_bomb0"] for a in bomb1_actions]
                    b1_spent = [branches[a]["spent"] for a in bomb1_actions]
                    b1_noop = [branches[a]["noop"] for a in bomb1_actions]
                    b1_kind = [branches[a]["kind"] for a in bomb1_actions]
                    b1_alive_equal = [b1_alive[i] == b1_alive0[i]
                                      for i in range(len(bomb1_actions))]
                    honoured = sum(1 for flag in b1_spent if flag)
                    ignored = len(b1_spent) - honoured
                    noop_count = sum(1 for flag in b1_noop if flag)
                    anomaly_count = sum(1 for a in bomb1_actions
                                        if branches[a]["anomaly"])
                    anomaly_cands = [branches[a]["cand"] for a in bomb1_actions
                                     if branches[a]["anomaly"]]
                    b1_kinds: dict[str, int] = {}
                    for action in bomb1_actions:
                        kind = branches[action]["kind"]
                        b1_kinds[kind] = b1_kinds.get(kind, 0) + 1
                        dec = decrements[action]
                        key = str(dec) if dec is not None else "None"
                        rec["label_checks"]["bomb1_decrement_frames"][key] = (
                            rec["label_checks"]["bomb1_decrement_frames"].get(
                                key, 0) + 1)
                    for index in range(len(bomb1_actions)):
                        if not b1_spent[index] and not b1_alive_equal[index]:
                            rec["label_checks"]["bomb1_unspent_alive_differs"] += 1
                    # a bomb=0 path must never touch the counter (v8.0.0-v8.6.0: never
                    # observed); counted, never aborting the cell
                    for action in bomb0_actions:
                        if decrements[action] is not None:
                            rec["label_checks"]["bomb0_spent_a_bomb"] += 1
                    for kind, count in b1_kinds.items():
                        rec["label_checks"]["anomaly_kinds"][kind] = (
                            rec["label_checks"]["anomaly_kinds"].get(kind, 0) + count)
                    if rec["label_checks"]["bomb0_spent_a_bomb"]:
                        rec["label_checks"]["anomaly_kinds"][
                            "anomaly_bomb0_branch_spent_a_bomb"] = rec[
                                "label_checks"]["bomb0_spent_a_bomb"]
                    rec["label_checks"]["bomb1_press_honoured"] += honoured
                    rec["label_checks"]["bomb1_press_ignored"] += ignored
                    rec["label_checks"]["bomb1_noop_due_to_invuln"] += noop_count
                    rec["label_checks"]["bomb1_anomalies"] += anomaly_count
                    rec["label_checks"]["states_with_bomb1_anomaly"] += int(
                        anomaly_count > 0)
                    rec["label_checks"]["states_all_bomb1_noop"] += int(
                        honoured == 0)
                    rec["label_checks"]["states_some_bomb1_noop"] += int(
                        0 < noop_count < len(bomb1_actions))
                    rec["label_checks"]["states_no_bomb1_noop"] += int(
                        noop_count == 0)
                    player_state = int(state.get("player_state") or 0)
                    invuln_name = player_state_name(player_state)
                    rec["label_checks"]["noop_by_player_state"][
                        invuln_name] = rec["label_checks"][
                            "noop_by_player_state"].get(invuln_name, 0) + noop_count
                    rec["label_checks"]["states_by_player_state"][
                        invuln_name] = rec["label_checks"][
                            "states_by_player_state"].get(invuln_name, 0) + 1
                    rec["label_checks"]["labels_36"] += 1
                    rec["label_checks"]["baseline_frame_ok_total"] += N_ACTIONS
                    rec["label_checks"]["baseline_state_equal_parent_total"] += N_ACTIONS
                    rec["label_checks"]["spawn_ok_total"] += N_ACTIONS
                    rec["label_checks"]["quit_ok_total"] += N_ACTIONS
                    if bombs_before == 0:
                        rec["label_checks"]["states_without_a_bomb_left"] += 1
                    obs = v450().compact_observation(state, hazards)
                    row = {
                        "schema": SCHEMA_LABEL, "seed": seed,
                        "window": int(window), "episode": int(episode),
                        "decision": int(decision), "frame": state_frame,
                        "H": int(horizon), "actions": N_ACTIONS,
                        "bombs_remaining": int(bombs_before),
                        # ---- the four per-row fields of the invariant -------------
                        # the branch the parent actually executed
                        # (action = cand * 2 + bomb); every one of the 18 bomb=1
                        # branches of this state is in "bomb1" below.
                        "bombs_before": int(block["bombs_start"][int(got["action"])]),
                        "bombs_after": int(block["bombs_end"][int(got["action"])]),
                        "bomb_spent": bool(
                            int(block["bombs_end"][int(got["action"])])
                            == int(block["bombs_start"][int(got["action"])]) - 1),
                        "noop_due_to_invuln": bool(
                            int(got["bomb"]) == 1
                            and branches[int(got["action"])]["noop"]),
                        # ---- this state's 18 bomb=1 branches (aggregate) ----------
                        "bomb1_noop_count": int(noop_count),
                        "bomb1_anomaly_count": int(anomaly_count),
                        "bomb1_anomaly_cands": anomaly_cands,
                        "bomb1_anomaly_kinds": b1_kinds,
                        "bomb1_branch_kind": b1_kind,
                        "decision_bucket": decision_bucket(decision),
                        "behaviour": {
                            "policy": "v4.5.0 `S` round2.pt",
                            "ckpt": str(ckpt), "ckpt_sha256": ckpt_sha,
                            "kind": "stochastic",
                            "generator_seed": generator_seed(seed),
                            "cand": int(got["cand"]), "bomb": int(got["bomb"]),
                            "action": int(got["action"]),
                            "nearest_all_px": got["nearest_all_px"],
                            "nearest_lethal_px": got["nearest_lethal_px"],
                            "bullets": got["bullets"],
                            "bullets_within_60": got["bullets_within_60"],
                            "bomb_rule": R5_RULE,
                            "threshold_px": got["threshold_px"],
                            "threshold_source": ("tools/rl/v800_bomb.py::"
                                                 "rule_gate('staged')"),
                        },
                        "features_sha256_k12": got["features_sha256_k12"],
                        "safe": block["safe"],
                        "alive_frames": block["alive_frames"],
                        "bombs_used": block["bombs_used"],
                        "bombs_start": block["bombs_start"],
                        "bombs_end": block["bombs_end"],
                        "bomb_decrement_frame": block["bomb_decrement_frame"],
                        "bomb1_presses_honoured": int(honoured),
                        "bomb1_presses_ignored": int(ignored),
                        # ---- the bomb=1 audit of this state, per candidate ----
                        # ENGINE SEMANTICS (measured in v8.0.0/v8.3.0/v8.4.0 and
                        # again here): a bomb press is honoured only when no bomb
                        # is already active — inside a bomb's invulnerability
                        # window the press is a **no-op**, so bomb=1 and bomb=0 of
                        # the same candidate are the same act (`noop_due_to_invuln`
                        # = unspent AND the two arms lived exactly as long).
                        # Recorded, never dropped and never aborted on.
                        "bomb1": {
                            "cands": [int(v830().split_action(a)[0])
                                      for a in bomb1_actions],
                            "bombs_before": b1_before,
                            "bombs_after": b1_after,
                            "alive_frames": [int(block["alive_frames"][a])
                                             for a in bomb1_actions],
                            "alive_frames_bomb0": [int(block["alive_frames"][a - 1])
                                                   for a in bomb1_actions],
                            "decrement_frame": [decrements[a]
                                                for a in bomb1_actions],
                            "bomb_spent": b1_spent,
                            "alive_equal_to_bomb0": b1_alive_equal,
                            "noop_due_to_invuln": b1_noop,
                            "branch_kind": b1_kind,
                            "spent_fraction": round(honoured / len(b1_spent), 6),
                            "noop_count": int(noop_count),
                            "anomaly_count": int(anomaly_count),
                            "anomaly_cands": anomaly_cands,
                            "all_noop": bool(honoured == 0),
                            "any_noop": bool(noop_count > 0),
                        },
                        "invulnerability": {
                            "source": ("STATE player_state (u8): 3 = "
                                       "PLAYER_STATE_INVULNERABLE"),
                            "player_state": player_state,
                            "player_state_name": invuln_name,
                            "invulnerable": player_state == 3,
                        },
                        "park_reply": block["park_reply"],
                        "child_replies_digest": block["child_replies_digest"],
                        "child_reply_lines": block["child_reply_lines"],
                        "children": block["children"],
                        "label_wall_s": block["label_wall_s"],
                        "obs": obs,
                    }
                    write_jsonl_line(handle, row)
                    rec["states"] += 1
                    ep["states_labelled"] += 1
                    rec["state_costs_s"].append(float(block["label_wall_s"]))
                    # ---- the parent takes the behaviour action ----------------
                    first_mask, rest_mask = v830().action_masks(got["action"])
                    for frame_index in range(DECISION_FRAMES):
                        step = parent.step(first_mask if frame_index == 0
                                           else rest_mask)
                        if step["advanced"] == 1:
                            ep["frames"] += 1
                        reading, _reply = parent.state()
                        term = cf._terminal_of(
                            reading, deaths_start=deaths0, stage0=stage0,
                            hash_hex=(step["hash"] or ""),
                            frame=int(reading["frame"]), exited=step["exited"],
                            game_over=step["game_over"])
                        state = reading
                        if term is not None:
                            ep["terminal"] = str(term.get("terminal"))
                            if term.get("engine_dead"):
                                ep["died"] = True
                            died = True
                            break
                    ep["decisions"] += 1
                    if rec["states"] % 50 == 0:
                        log(f"  s{seed} w{window} ep{episode} d{decision}: "
                            f"{rec['states']}/{quota} states, bombs "
                            f"{rec['bombs_remaining_counts']}, mean_label="
                            f"{statistics.fmean(rec['state_costs_s']):.3f}s")
                    if died:
                        break
                ep["bombs_end"] = int(state["bombs_remaining"] or 0)
                ep["bombs_used"] = int(ep["bombs_start"]) - int(ep["bombs_end"])
                rec["episodes"].append(ep)
                episode += 1
                log(f"s{seed} w{window} ep{ep['episode']}: "
                    f"{ep['decisions']} decisions, {ep['states_labelled']} states, "
                    f"bombs {ep['bombs_start']}->{ep['bombs_end']}, "
                    f"emits={ep['bomb_emits']}, died={ep['died']}, "
                    f"terminal={ep['terminal']}")
            parent.close()
            parents.remove(parent)
            rec["status"] = "complete" if rec["states"] >= quota else "short"
            if rec["label_checks"]["bomb1_press_honoured"] <= 0:
                raise BombDagError(
                    f"w{window}: no bomb=1 label ever decremented the counter in "
                    f"{rec['states']} states — the bomb path is not working")
            # a bomb=0 label that spends a bomb is an accounting anomaly of the
            # labeller, recorded in label_checks["bomb0_spent_a_bomb"] and in
            # anomaly_kinds; it does NOT abort the cell (BOMBDAGGER-001: the
            # counter-dependent branches are recorded, never fatal).
            if rec["label_checks"]["bomb0_spent_a_bomb"]:
                log(f"w{window}: WARNING {rec['label_checks']['bomb0_spent_a_bomb']} "
                    f"bomb=0 label(s) spent a bomb (counted as an anomaly, "
                    f"cell kept)")
        except Exception as exc:                                      # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["status"] = "failed"
            log(f"COLLECT FAILED s{seed} w{window}: {rec['error']}")
        finally:
            handle.close()
            for parent in parents:
                try:
                    parent.close()
                except Exception:                                     # noqa: BLE001
                    pass
            rec["wall_s"] = round(time.perf_counter() - started, 3)
            rec["label_cost_s"] = quantiles(rec["state_costs_s"])
            del rec["state_costs_s"]
            rec["labels_total"] = int(rec["states"]) * N_ACTIONS
            if rec["status"] in ("complete", "short"):
                if labels.exists():
                    labels.unlink()
                partial.rename(labels)
                rec["labels_sha256"] = sha256_file(labels)
                rec["labels_bytes"] = labels.stat().st_size
            write_json(summary_path, rec)
        log(f"collect s{seed} w{window}: status={rec['status']} "
            f"states={rec['states']} bombs={rec['bombs_remaining_counts']} "
            f"wall={rec['wall_s']}s -> {labels}")
        if rec["status"] == "failed":
            return 4
    return 0


# --------------------------------------------------------------------------- #
# mode: stats (pool + the bombs_remaining distribution, P0)
# --------------------------------------------------------------------------- #
def mode_stats(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    cell_files = sorted((root / "labels").glob("seed*_w*.jsonl")) \
        if (root / "labels").is_dir() else []
    if not cell_files:
        raise BombDagError(f"no cell files under {root / 'labels'}")
    pooled = (pathlib.Path(args.labels_out) if args.labels_out
              else pooled_labels_path(root))
    out = pathlib.Path(args.out) if args.out else pooled_stats_path(root)
    pooled.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    per_cell: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    for path in cell_files:
        cell_rows = read_jsonl(path)
        if not cell_rows:
            log(f"stats: {path} is empty (skipped)")
            continue
        rows.extend(cell_rows)
        seed, window = int(cell_rows[0]["seed"]), int(cell_rows[0]["window"])
        key = f"s{seed}_w{window}"
        per_cell[key] = {
            "seed": seed, "window": window, "states": len(cell_rows),
            "bombs_remaining_counts": bombs_histogram(
                [r["bombs_remaining"] for r in cell_rows]),
            "episodes": len(sorted({int(r["episode"]) for r in cell_rows})),
            "path": str(path)}
        sources.append({"path": str(path), "sha256": sha256_file(path),
                        "bytes": path.stat().st_size, "states": len(cell_rows)})
    with pooled.open("w", encoding="utf-8") as handle:
        for row in rows:
            write_jsonl_line(handle, row)
    pooled_sha = sha256_file(pooled)

    states = len(rows)
    bombs_counts = bombs_histogram([r["bombs_remaining"] for r in rows])
    p0_bombs2 = int(bombs_counts.get("2", 0))
    p0_bombs1 = int(bombs_counts.get("1", 0))
    p0_pass = bool(p0_bombs2 >= P0_MIN_STATES and p0_bombs1 >= P0_MIN_STATES)
    decisions = [int(r["decision"]) for r in rows]
    bomb_bits = [int(r["behaviour"]["bomb"]) for r in rows]
    nearest = [float(r["behaviour"]["nearest_all_px"])
               for r in rows if r["behaviour"]["nearest_all_px"] is not None]
    thresholds: dict[str, int] = {}
    for row in rows:
        key = f"{float(row['behaviour']['threshold_px']):g}"
        thresholds[key] = thresholds.get(key, 0) + 1
    by_bombs: dict[str, Any] = {}
    for level in ("3", "2", "1", "0"):
        subset = [r for r in rows if str(int(r["bombs_remaining"])) == level]
        if not subset:
            by_bombs[level] = {"states": 0}
            continue
        emitted = sum(int(r["behaviour"]["bomb"]) for r in subset)
        nearest_sub = [float(r["behaviour"]["nearest_all_px"]) for r in subset
                       if r["behaviour"]["nearest_all_px"] is not None]
        by_bombs[level] = {
            "states": len(subset),
            "behaviour_bomb_emitted": emitted,
            "behaviour_bomb_rate": round(emitted / len(subset), 6),
            "nearest_px": quantiles(nearest_sub),
            "thresholds_used": sorted({f"{float(r['behaviour']['threshold_px']):g}"
                                       for r in subset}),
            "states_with_no_bomb_left": int(level) == 0,
            "decision_histogram": decision_histogram([int(r["decision"])
                                                      for r in subset])}
    # -- the 36-label contrast, split by bombs_remaining (descriptive only) ----
    def label_block(level: str | None) -> dict[str, Any]:
        subset = [r for r in rows if level is None
                  or str(int(r["bombs_remaining"])) == level]
        bomb0 = [a for a in range(N_ACTIONS) if int(v830().split_action(a)[1]) == 0]
        bomb1 = [a for a in range(N_ACTIONS) if int(v830().split_action(a)[1]) == 1]
        safe0 = [int(r["safe"][a]) for r in subset for a in bomb0]
        safe1 = [int(r["safe"][a]) for r in subset for a in bomb1]
        alive0 = [int(r["alive_frames"][a]) for r in subset for a in bomb0]
        alive1 = [int(r["alive_frames"][a]) for r in subset for a in bomb1]
        return {"states": len(subset),
                "bomb0": {"labels": len(safe0),
                          "safe_rate": (round(statistics.fmean(safe0), 6)
                                        if safe0 else None),
                          "alive_frames": quantiles(alive0)},
                "bomb1": {"labels": len(safe1),
                          "safe_rate": (round(statistics.fmean(safe1), 6)
                                        if safe1 else None),
                          "alive_frames": quantiles(alive1)},
                "bomb1_minus_bomb0_safe_rate": (
                    round(statistics.fmean(safe1) - statistics.fmean(safe0), 6)
                    if safe0 and safe1 else None)}

    # ---- the bomb=1 invariant, recomputed from the raw per-branch arrays -----
    # (the same two-branch rule the collect rolled out with; recomputed here so a
    #  label file is self-describing even without the collect summary)
    bomb1_actions = [a for a in range(N_ACTIONS)
                     if int(v830().split_action(a)[1]) == 1]
    inv_kinds: dict[str, int] = {}
    inv_by_bombs: dict[str, dict[str, Any]] = {}
    inv_branches = inv_spent = inv_noop = inv_anomaly = 0
    inv_states = state_any_noop = state_all_noop = 0
    beh_rows = beh_bomb_rows = beh_spent = beh_noop = beh_anomaly = 0
    for row in rows:
        level = str(int(row["bombs_remaining"]))
        cell = inv_by_bombs.setdefault(
            level, {"branches": 0, "spent": 0, "noop_due_to_invuln": 0,
                    "anomalies": 0, "states_any_noop": 0,
                    "states_all_18_noop": 0})
        beh_rows += 1
        inv_states += 1
        try:
            bomb_bit = int(row["behaviour"]["bomb"])
        except (KeyError, TypeError):
            bomb_bit = 0
        if bomb_bit == 1:
            beh_bomb_rows += 1
            if bool(row.get("bomb_spent", False)):
                beh_spent += 1
            elif bool(row.get("noop_due_to_invuln", False)):
                beh_noop += 1
            else:
                beh_anomaly += 1
        row_noop = 0
        for action in bomb1_actions:
            before = int(row["bombs_start"][action])
            after = int(row["bombs_end"][action])
            alive1 = int(row["alive_frames"][action])
            alive0 = int(row["alive_frames"][action - 1])
            spent = bool(after == before - 1)
            noop = bool(not spent and after == before and alive1 == alive0)
            inv_branches += 1
            cell["branches"] += 1
            if spent:
                inv_spent += 1
                cell["spent"] += 1
            elif noop:
                inv_noop += 1
                cell["noop_due_to_invuln"] += 1
                row_noop += 1
            else:
                inv_anomaly += 1
                cell["anomalies"] += 1
                kind = ("counter_changed_not_minus_one" if after != before
                        else "counter_unchanged_alive_differs")
                inv_kinds[kind] = inv_kinds.get(kind, 0) + 1
        if row_noop:
            state_any_noop += 1
            cell["states_any_noop"] += 1
        if row_noop == len(bomb1_actions):
            state_all_noop += 1
            cell["states_all_18_noop"] += 1
    for cell in inv_by_bombs.values():
        cell["states"] = cell["branches"] // len(bomb1_actions)
        cell["noop_frac"] = (round(cell["noop_due_to_invuln"] / cell["branches"], 6)
                             if cell["branches"] else None)
        cell["noop_state_frac"] = (round(cell["states_any_noop"] / cell["states"], 6)
                                   if cell["states"] else None)
        cell["noop_branch_frac"] = (
            round(cell["noop_due_to_invuln"] / (N_ACTIONS * cell["states"]), 6)
            if cell["states"] else None)
        cell["anomaly_frac"] = (round(cell["anomalies"] / cell["branches"], 6)
                                if cell["branches"] else None)
    noop_frac = round(inv_noop / inv_branches, 6) if inv_branches else None
    noop_state_frac = (round(state_any_noop / inv_states, 6)
                       if inv_states else None)
    noop_state_frac_all_18 = (round(state_all_noop / inv_states, 6)
                              if inv_states else None)
    noop_branch_frac = (round(inv_noop / (N_ACTIONS * inv_states), 6)
                        if inv_states else None)
    bomb1_invariant = {
        "rule": ("a bomb=1 branch passes iff bombs_after == bombs_before - 1 "
                 "(honoured press) OR bombs_after == bombs_before AND its "
                 "alive_frames equal the same candidate's bomb=0 arm (no-op inside "
                 "the previous bomb's invulnerability window: bombs do not stack). "
                 "A branch that fits neither is counted as an anomaly and kept; "
                 "noop_due_to_invuln rows are legitimate counterfactual pairs and "
                 "are never dropped"),
        "branches": inv_branches,
        "spent": inv_spent,
        "noop_due_to_invuln": inv_noop,
        "noop_frac": noop_frac,
        "noop_branch_frac": noop_branch_frac,
        "noop_state_frac": noop_state_frac,
        "noop_state_frac_all_18": noop_state_frac_all_18,
        "noop_definitions": {
            "noop_frac": ("noop bomb=1 branches / (18 * states) — the bomb=1 "
                          "branch denominator (backward-compatible name)"),
            "noop_branch_frac": ("noop bomb=1 branches / (36 * states) — the "
                                 "36-way action-branch denominator"),
            "noop_state_frac": ("states with at least one noop bomb=1 branch / "
                                "states (a state-level count: the 18 noop branches "
                                "of one state always come from the same press window)"),
            "noop_state_frac_all_18": ("states whose 18 bomb=1 branches are ALL "
                                       "noop / states"),
            "row_fields": ("each label row also carries bomb_spent / "
                           "noop_due_to_invuln / bombs_before / bombs_after for "
                           "the branch the behaviour policy executed"),
        },
        "anomalies": inv_anomaly,
        "anomaly_count": inv_anomaly,
        "anomaly_frac": (round(inv_anomaly / inv_branches, 6)
                         if inv_branches else None),
        "anomaly_kinds": inv_kinds,
        "by_bombs_remaining": inv_by_bombs,
        "rows": {
            "rows": beh_rows, "behaviour_bomb_rows": beh_bomb_rows,
            "behaviour_bomb_spent": beh_spent,
            "behaviour_bomb_noop_due_to_invuln": beh_noop,
            "behaviour_bomb_anomaly": beh_anomaly,
            "noop_frac_of_behaviour_bomb_rows": (
                round(beh_noop / beh_bomb_rows, 6) if beh_bomb_rows else None),
            "note": ("bomb_spent / noop_due_to_invuln / bombs_before / bombs_after "
                     "of each row describe the branch the parent actually executed; "
                     "the branch counts above are the 18 bomb=1 branches of every "
                     "state"),
        },
    }

    episodes = sum(len(sorted({int(r["episode"]) for r in cell_rows}))
                   for cell_rows in (read_jsonl(p) for p in cell_files))
    payload: dict[str, Any] = {
        "schema": SCHEMA_STATS, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL,
        "no_statistics": ("counts, rates, means, medians, quantiles and "
                          "distributions only; no test, no interval, no p-value"),
        "labels_file": {"path": str(pooled), "sha256": pooled_sha,
                        "bytes": pooled.stat().st_size, "states": states,
                        "labels_total": states * N_ACTIONS},
        "sources": sources,
        "states": states, "episodes": episodes,
        "H": sorted({int(r["H"]) for r in rows}),
        "actions": sorted({int(r["actions"]) for r in rows}),
        "windows": sorted({int(r["window"]) for r in rows}),
        "seeds": sorted({int(r["seed"]) for r in rows}),
        # ---- the key deliverable: the bombs_remaining distribution ----------
        "bombs_remaining": {
            "distribution": bombs_counts,
            "fractions": {key: round(value / states, 6)
                          for key, value in bombs_counts.items()},
            "distinct_values": sorted(int(key) for key in bombs_counts),
            "note": ("every v8.3.0 state had bombs_remaining = 3 (5,166/5,166); "
                     "this is the v8.6.0 behaviour policy's coverage"),
        },
        "p0": {
            "requirement": (f"prereg §2 P0: bombs_remaining 2 and 1 each >= "
                            f"{P0_MIN_STATES} states"),
            "bombs_remaining_2": p0_bombs2,
            "bombs_remaining_1": p0_bombs1,
            "min_required": P0_MIN_STATES,
            "pass": p0_pass,
            "stop_condition": ("S-data: if either is below the minimum the round "
                               "is void (prereg §2.1)"),
        },
        # ---- the bomb=1 invariant: noop is legal, anomalies are counted -------
        "noop_frac": noop_frac,
        "noop_branch_frac": noop_branch_frac,
        "noop_state_frac": noop_state_frac,
        "noop_state_frac_all_18": noop_state_frac_all_18,
        "anomaly_count": inv_anomaly,
        "bomb1_invariant": bomb1_invariant,
        # ---- the decision-interval distribution ------------------------------
        "decision_intervals": {
            "histogram": decision_histogram(decisions),
            "buckets": [f"{low}-{high}" for low, high in DECISION_BUCKETS],
            "decision_index": quantiles(decisions),
            "note": ("decision = index inside the episode (0-based); the roll-out "
                     f"is capped at {DECISIONS} decisions per episode"),
        },
        # ---- the bomb-usage distribution of the behaviour policy ------------
        "bomb_usage": {
            "states_with_behaviour_bomb": sum(bomb_bits),
            "states_without_behaviour_bomb": states - sum(bomb_bits),
            "behaviour_bomb_rate": round(sum(bomb_bits) / states, 6) if states else None,
            "thresholds_used": thresholds,
            "thresholds_px_definition": {str(k): v for k, v in
                                         R5_THRESHOLDS.items()},
            "bomb_rule": R5_RULE,
            "bomb_rule_source": R5_SOURCE,
            "by_bombs_remaining": by_bombs,
            "nearest_all_px": quantiles(nearest),
            "episodes": [{"seed": int(summary["seed"]),
                          "window": int(summary["window"]),
                          "episode": int(ep["episode"]),
                          "decisions": int(ep["decisions"]),
                          "states_labelled": int(ep["states_labelled"]),
                          "bombs_start": int(ep["bombs_start"]),
                          "bombs_end": int(ep["bombs_end"]),
                          "bombs_used": int(ep["bombs_used"]),
                          "bomb_emits": int(ep["bomb_emits"]),
                          "died": bool(ep["died"]), "terminal": ep["terminal"]}
                         for summary in (read_json(p) for p in sorted(
                             (root / "collect").glob("seed*_w*.json")))
                         for ep in summary.get("episodes", [])],
        },
        "labels": {"all": label_block(None), "bombs_3": label_block("3"),
                   "bombs_2": label_block("2"), "bombs_1": label_block("1"),
                   "bombs_0": label_block("0"),
                   "note": ("the same 36-way counterfactual label, now evaluated on "
                            "states with 2 and 1 bombs left; descriptive only")},
        "per_cell": per_cell,
        "shape": {"complete_36_label_rows": sum(
            1 for r in rows if len(r["safe"]) == N_ACTIONS
            and all(v in (0, 1) for v in r["safe"]))},
    }
    write_json(out, payload)
    log(f"stats: {states} states, bombs={bombs_counts}, P0="
        f"{'PASS' if p0_pass else 'FAIL'} (2:{p0_bombs2} 1:{p0_bombs1} >= "
        f"{P0_MIN_STATES}) -> {out}")
    log(f"stats: bomb=1 branches={inv_branches} spent={inv_spent} "
        f"noop={inv_noop} (noop_frac={noop_frac} noop_branch_frac="
        f"{noop_branch_frac} noop_state_frac={noop_state_frac}) "
        f"anomalies={inv_anomaly} kinds={inv_kinds or '{}'}")
    if not p0_pass:
        log("S-data: P0 is not met; the round is void and nothing is trained")
        return 4
    return 0


# --------------------------------------------------------------------------- #
# mode: train
# --------------------------------------------------------------------------- #
def mode_train(args: argparse.Namespace) -> int:
    import torch

    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    if seed not in TRAIN_SEEDS:
        raise BombDagError(f"--seed must be one of {TRAIN_SEEDS}, got {seed}")
    rows, info = load_pooled_rows(root)
    if not rows:
        raise BombDagError("no labelled rows")
    feats, labels, array_info = v830().build_arrays(rows)
    for row in rows:
        pass
    bombs_counts = bombs_histogram([r["bombs_remaining"] for r in rows])
    log(f"train seed={seed}: {len(rows)} states x {N_ACTIONS} actions, "
        f"dim={FEATURE_DIM}, bombs={bombs_counts}")

    x = torch.from_numpy(feats)
    y = torch.from_numpy(labels)
    policy = v830().policy36(seed=seed)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    import numpy as np

    rng = np.random.default_rng(seed)
    epoch_losses: list[float] = []
    batch = int(args.batch)
    started = time.perf_counter()
    for _epoch in range(int(EPOCHS)):
        perm = torch.from_numpy(rng.permutation(len(rows)))
        total = 0.0
        policy.train()
        for start in range(0, len(rows), batch):
            index = perm[start:start + batch]
            logits = policy(x.index_select(0, index))
            loss = loss_fn(logits, y.index_select(0, index))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(index.numel())
        epoch_losses.append(total / len(rows))
    wall = round(time.perf_counter() - started, 3)

    policy.eval()
    with torch.no_grad():
        logits = policy(x)
        final_bce = float(torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y))
        predicted = (torch.sigmoid(logits) >= 0.5).float()
        overall_acc = float((predicted == y).float().mean())
        per_action_acc = [float((predicted[:, action] == y[:, action]).float().mean())
                          for action in range(N_ACTIONS)]
        probs = torch.sigmoid(logits)
        bomb0 = [a for a in range(N_ACTIONS) if int(v830().split_action(a)[1]) == 0]
        bomb1 = [a for a in range(N_ACTIONS) if int(v830().split_action(a)[1]) == 1]
        mean_prob_bomb0 = float(probs[:, bomb0].mean())
        mean_prob_bomb1 = float(probs[:, bomb1].mean())

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else policy_ckpt_path(root, seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), str(ckpt))
    report_path = (pathlib.Path(args.out) if args.out
                   else train_report_path(root, seed))
    report: dict[str, Any] = {
        "schema": SCHEMA_TRAIN, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "seed": seed, "states": len(rows), "labels": int(len(rows) * N_ACTIONS),
        "feature_dim": int(FEATURE_DIM), "actions": int(N_ACTIONS),
        "bombs_remaining_distribution": bombs_counts,
        "model": {"class": "v431_arm_a.Policy with the output layer widened to 36",
                  "shape": f"Linear({FEATURE_DIM},{HIDDEN}) -> Tanh -> "
                           f"Linear({HIDDEN},36)",
                  "params": int(sum(p.numel() for p in policy.parameters())),
                  "source": "tools/rl/v830_bombpol.py::policy36 (imported)"},
        "labels_file": info, "features": array_info,
        "optimizer": {"name": "Adam", "lr": LR, "batch": batch, "epochs": EPOCHS,
                      "early_stopping": None, "auxiliary_loss": None,
                      "shuffle": "numpy default_rng(seed), one permutation/epoch"},
        "loss": "BCEWithLogitsLoss, mean over batch and 36 outputs",
        "init": {"torch_manual_seed": seed, "hidden": HIDDEN,
                 "feature_dim": int(FEATURE_DIM)},
        "epoch_losses": [round(value, 6) for value in epoch_losses],
        "loss_first_epoch": round(epoch_losses[0], 6),
        "loss_last_epoch": round(epoch_losses[-1], 6),
        "final_train_bce": round(final_bce, 6),
        "train_accuracy": round(overall_acc, 6),
        "train_accuracy_per_action": [round(v, 6) for v in per_action_acc],
        "mean_predicted_prob_bomb0": round(mean_prob_bomb0, 6),
        "mean_predicted_prob_bomb1": round(mean_prob_bomb1, 6),
        "ckpt": {"path": str(ckpt), "sha256": sha256_file(ckpt),
                 "bytes": ckpt.stat().st_size},
        "pins": check_pins(), "wall_s": wall, "deviations": [],
    }
    write_json(report_path, report)
    log(f"train seed={seed}: loss {report['loss_first_epoch']:.4f} -> "
        f"{report['loss_last_epoch']:.4f} (BCE {report['final_train_bce']:.4f}, "
        f"acc {overall_acc:.4f}) wall={wall}s -> {ckpt}")
    return 0


def load_pooled_rows(root: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pooled = pooled_labels_path(root)
    if not pooled.is_file():
        raise BombDagError(f"no pooled labels: {pooled}")
    rows = read_jsonl(pooled)
    stats_rec = read_json(pooled_stats_path(root)) if pooled_stats_path(root).is_file() else {}
    digest = sha256_file(pooled)
    info = {"path": str(pooled), "sha256": digest,
            "bytes": pooled.stat().st_size, "rows": len(rows),
            "recorded_sha256": (stats_rec.get("labels_file") or {}).get("sha256"),
            "sha256_match": digest == (stats_rec.get("labels_file") or {}).get("sha256")}
    if not info["sha256_match"]:
        raise BombDagError(f"{pooled} sha256 {digest[:16]} != the recorded "
                           f"{str(info['recorded_sha256'])[:16]}")
    return rows, info


# --------------------------------------------------------------------------- #
# mode: eval (one budget x one seed x 4 dev windows)
# --------------------------------------------------------------------------- #
def load_policy36(ckpt: pathlib.Path, *, seed: int) -> Any:
    import torch

    path = pathlib.Path(ckpt)
    if not path.is_file():
        raise BombDagError(f"policy checkpoint not found: {path}")
    policy = v830().policy36(seed=seed)
    policy.load_state_dict(torch.load(str(path), map_location="cpu",
                                      weights_only=True))
    policy.eval()
    return policy


def mode_eval(args: argparse.Namespace) -> int:
    import torch

    from touhou_rl.v420 import contracts as C

    root = pathlib.Path(args.work_dir)
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    if seed not in TRAIN_SEEDS:
        raise BombDagError(f"--policy-seed must be one of {TRAIN_SEEDS}, got {seed}")
    if budget not in BUDGETS:
        raise BombDagError(f"prereg §1 fixes --decisions to {BUDGETS}, got {budget}")
    if int(args.decision_frames) != DECISION_FRAMES:
        raise BombDagError(f"prereg §1 keeps {DECISION_FRAMES} frames per decision")
    if int(args.episodes) < 1:
        raise BombDagError(f"--episodes must be >= 1, got {args.episodes}")
    windows = parse_windows(args.windows)
    for window in windows:
        if int(window) not in DEV_WINDOWS:
            raise BombDagError(f"window {window} is not a dev window {DEV_WINDOWS}")
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    tag = args.tag or f"v860-eval-d{budget}-s{seed}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    out_path = (pathlib.Path(args.out) if args.out
                else eval_path(root, budget, seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")
    ckpt = pathlib.Path(args.ckpt) if args.ckpt else policy_ckpt_path(root, seed)
    digest = sha256_file(ckpt) if ckpt.is_file() else None
    policy = load_policy36(ckpt, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    arm = arm_a()

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        feats, _detail = arm.build_features(state=state, hazards=hazards,
                                            decision_index=decision)
        return v830().split_action(
            v830().sample_action36(policy, feats, args.kind, generator))

    summary: dict[str, Any] = {
        "schema": SCHEMA_EVAL_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "mode": "eval", "policy_seed": seed, "windows": list(windows),
        "episodes_per_window": int(args.episodes),
        "budget_decisions": budget, "decision_frames": DECISION_FRAMES,
        "temperature": TEMPERATURE, "kind": args.kind,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest,
                 "bytes": ckpt.stat().st_size if ckpt.is_file() else None},
        "sampler": {"kind": "torch.multinomial on softmax(z/1) over 36 actions",
                    "generator_seed": generator_seed(seed),
                    "generator": ("one torch.Generator per (budget, seed) process, "
                                  "windows in the frozen order 11,12,17,18")},
        "episode_loop": ("tools/rl/v830_bombpol.py::run_episode36 (imported): the "
                         "v8.0.0 loop with the policy's sampled bomb bit on the "
                         "decision's first frame"),
        "action_space": {"actions": N_ACTIONS,
                         "encoding": "action = cand * 2 + bomb"},
        "out": str(out_path), "per_window": {}, "status": "failed",
        "error": None, "wall_s": None,
    }
    parents: list[Any] = []
    handle = out_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    try:
        for slot, window in enumerate(windows):
            C.assert_seed_allowed(int(window), "train")
            data_dir, data_info = v450().prepare_data_dir(
                data_root, window, args.data_source)
            log(f"d{budget} s{seed} w{window}: cfg muted="
                f"{data_info['cfg']['muted_four_zero']}")
            parent, gate = start_parent(slot=slot, window=window,
                                        engine=args.engine, data_dir=data_dir,
                                        log_dir=run_dir)
            parents.append(parent)
            rows: list[dict[str, Any]] = []
            for episode in range(int(args.episodes)):
                record = v830().run_episode36(
                    parent, episode=episode, policy_seed=seed, decisions=budget,
                    decide=decide, budget=budget, decision_frames=DECISION_FRAMES)
                record["schema"] = "v860-bombdag-eval-episode/1"
                record["producer"] = PRODUCER
                record["eval_harness"] = ("tools/rl/v830_bombpol.py::run_episode36 "
                                          "(imported)")
                record.update({"ckpt_sha256": digest,
                               "generator_seed": generator_seed(seed)})
                record.pop("terminal_reply", None)
                record.pop("trace", None)
                record.pop("spawn_reply", None)
                write_jsonl_line(handle, record)
                rows.append(record)
                if record.get("invalid"):
                    raise BombDagError(f"d{budget} seed {seed} window {window} "
                                       f"episode {episode} is invalid: "
                                       f"{record.get('error')}")
                if (episode + 1) % 50 == 0:
                    done = sum(1 for row in rows if row["survived"])
                    used = sum(int(row["bombs_used"] or 0) for row in rows)
                    log(f"  d{budget} s{seed} w{window}: {episode + 1}/"
                        f"{args.episodes} episodes, survived={done}, "
                        f"bombs_used={used}")
            alive = [float(row["alive_frames"]) for row in rows]
            survived = sum(1 for row in rows if row["survived"])
            summary["per_window"][str(window)] = {
                "window": int(window), "episodes": len(rows),
                "survived": survived,
                "survived_fraction": round(survived / len(rows), 6) if rows else None,
                "alive_frames": quantiles(alive),
                "frames": quantiles([float(row["frames"]) for row in rows]),
                "bomb_episodes": sum(1 for row in rows if row["bomb_emits"]),
                "bomb_emits_total": sum(int(row["bomb_emits"]) for row in rows),
                "bombs_used_total": sum(int(row["bombs_used"] or 0) for row in rows),
                "episodes_exhausted": sum(1 for row in rows
                                          if row["bombs_end"] == 0),
                "cfg_muted": data_info["cfg"]["muted_four_zero"],
                "cfg_sha256": data_info["cfg"]["sha256"],
                "data_dir": str(data_dir),
                "gate": {key: gate[key] for key in (
                    "hello_role_shadow", "park_ok", "park_frame", "threads_1",
                    "all_pass")},
                "gate_prefix": {key: gate["prefix"][key] for key in (
                    "hash_checks", "hash_mismatches", "final_frame")},
            }
            log(f"d{budget} seed {seed} window {window}: survived {survived}/"
                f"{len(rows)} mean_alive="
                f"{summary['per_window'][str(window)]['alive_frames']['mean']}")
        summary["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["status"] = "failed"
        log(f"EVAL FAILED d{budget} seed {seed}: {summary['error']}")
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
        if out_path.is_file():
            summary["out_sha256"] = sha256_file(out_path)
            summary["out_bytes"] = out_path.stat().st_size
        write_json(summary_path, summary)
    log(f"eval d{budget} seed {seed}: status={summary['status']} "
        f"survived={summary.get('survived_total')}/{summary.get('episodes_total')} "
        f"bombs_used={summary.get('bombs_used_total')} wall={summary['wall_s']}s -> "
        f"{out_path}")
    return 0 if summary["status"] == "complete" else 4


# --------------------------------------------------------------------------- #
# mode: diagnostics
# --------------------------------------------------------------------------- #
def _episode_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return read_jsonl(path)


def bomb_usage_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    bombs_used = [int(row["bombs_used"] or 0) for row in rows]
    hist = bombs_histogram(bombs_used)
    counts = {str(k): int(hist.get(str(k), 0)) for k in (0, 1, 2, 3)}
    total = len(rows)
    first = [int(row["first_bomb_decision"]) for row in rows
             if row.get("first_bomb_decision") is not None]
    last = [max(int(d) for d in row["bomb_decisions"]) for row in rows
            if row.get("bomb_decisions")]
    return {
        "episodes": total,
        "episodes_by_bombs_used": counts,
        "fraction_by_bombs_used": {key: round(value / total, 6)
                                   for key, value in counts.items()} if total else {},
        "episodes_using_1_or_2_bombs": int(counts.get("1", 0) + counts.get("2", 0)),
        "episodes_using_3_bombs": int(counts.get("3", 0)),
        "episodes_using_0_bombs": int(counts.get("0", 0)),
        "bombs_spent_total": sum(bombs_used),
        "bombs_used_per_episode": quantiles(bombs_used),
        "bomb_emits_total": sum(int(row["bomb_emits"] or 0) for row in rows),
        "episodes_that_pressed_bomb": sum(1 for row in rows
                                          if int(row["bomb_emits"] or 0) > 0),
        "episodes_exhausted": sum(1 for row in rows if row["bombs_end"] == 0),
        "exhaustion_fraction": round(sum(1 for row in rows
                                         if row["bombs_end"] == 0) / total, 6)
        if total else None,
        "first_bomb_decision": quantiles(first),
        "first_bomb_decision_histogram": first_bomb_histogram(first),
        "last_bomb_decision": quantiles(last),
        "last_bomb_decision_histogram": first_bomb_histogram(last),
        "episodes_never_bombed": total - sum(1 for row in rows
                                             if int(row["bomb_emits"] or 0) > 0),
        "bombs_ignored_emits_total": sum(int(row.get("bombs_ignored_emits") or 0)
                                         for row in rows),
        "note": ("P3 is descriptive: the v8.3.0 arm used 3 bombs in 3,600/3,600 "
                 "episodes, so any episode with 1 or 2 is new"),
    }


def mode_diagnostics(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out = pathlib.Path(args.out) if args.out else root / "diagnostics.json"
    usage_out = (pathlib.Path(args.usage_out) if args.usage_out
                 else root / "bomb_usage.json")
    payload: dict[str, Any] = {
        "schema": SCHEMA_DIAG, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "no_statistics": ("counts, rates, means, medians, quantiles and "
                          "distributions only; no test, no interval, no p-value "
                          "(prereg §2 P1/P2 belong to the analysis pass)"),
        "budgets": {}, "per_cell": {},
        "references": {"v830_bombpol": BASELINE_V830, "rule_r5": RULE_R5},
        "budget_comparison_warning": ("prereg §5: the three budgets must not be "
                                      "compared by their means as if they were the "
                                      "same measurement"),
        "missing": [], "all_pass": True,
    }
    usage: dict[str, Any] = {
        "schema": SCHEMA_BOMB_USAGE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "budgets": {}, "missing": [],
    }
    for budget in BUDGETS:
        rows: list[dict[str, Any]] = []
        per_seed: dict[str, Any] = {}
        per_window: dict[str, Any] = {}
        for seed in TRAIN_SEEDS:
            path = eval_path(root, budget, seed)
            seed_rows = _episode_rows(path)
            if not seed_rows:
                payload["missing"].append(str(path))
                usage["missing"].append(str(path))
                payload["all_pass"] = False
                continue
            rows.extend(seed_rows)
            per_seed[str(seed)] = {
                "episodes": len(seed_rows),
                "survived": sum(1 for row in seed_rows if row["survived"]),
                "alive_frames": quantiles([row["alive_frames"]
                                           for row in seed_rows]),
                "bombs_used": bomb_usage_block(seed_rows),
                "path": str(path), "sha256": sha256_file(path)}
            for window in DEV_WINDOWS:
                cell = [row for row in seed_rows
                        if int(row["window"]) == int(window)]
                if not cell:
                    continue
                key = f"s{seed}_w{window}"
                per_window[key] = {
                    "seed": seed, "window": int(window), "episodes": len(cell),
                    "survived": sum(1 for row in cell if row["survived"]),
                    "alive_frames": quantiles([row["alive_frames"] for row in cell]),
                    "bombs_used_total": sum(int(row["bombs_used"] or 0)
                                            for row in cell)}
                payload["per_cell"][f"d{budget}_{key}"] = dict(per_window[key])
        if not rows:
            continue
        alive = [float(row["alive_frames"]) for row in rows]
        survived = sum(1 for row in rows if row["survived"])
        payload["budgets"][str(budget)] = {
            "budget_decisions": budget,
            "episodes": len(rows),
            "survived": survived,
            "survived_fraction": round(survived / len(rows), 6),
            "alive_frames": quantiles(alive),
            "alive_frames_histogram": histogram(
                [int(v) for v in alive],
                (1, 12, 24, 48, 60, 120, 300, 600, 900, 1200, 1800)),
            "per_seed": per_seed, "per_seed_window": per_window,
            "bomb": bomb_usage_block(rows),
            "reference_delta": {
                "vs_v830_survived_fraction": round(
                    survived / len(rows) - BASELINE_V830["rate"][budget], 6),
                "vs_r5_rule_survived_fraction": round(
                    survived / len(rows) - RULE_R5["rate"][budget], 6),
                "note": ("arithmetic differences of the quoted reference numbers; "
                         "no interval, no test"),
            },
        }
        usage["budgets"][str(budget)] = dict(payload["budgets"][str(budget)])
    write_json(out, payload)
    write_json(usage_out, usage)
    for budget in BUDGETS:
        rec = payload["budgets"].get(str(budget))
        if rec:
            log(f"d{budget}: survived {rec['survived']}/{rec['episodes']} = "
                f"{rec['survived_fraction']:.4%} mean_alive="
                f"{rec['alive_frames']['mean']} median="
                f"{rec['alive_frames']['median']} bombs/ep="
                f"{rec['bomb']['bombs_used_per_episode']['mean']} "
                f"1-2 bombs={rec['bomb']['episodes_using_1_or_2_bombs']} "
                f"exhausted={rec['bomb']['exhaustion_fraction']}")
    log(f"diagnostics -> {out} (+ {usage_out})")
    return 0 if payload["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: train-summary
# --------------------------------------------------------------------------- #
def mode_train_summary(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out: dict[str, Any] = {
        "schema": SCHEMA_TRAIN_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "design": ("BCE, Adam 1e-3, batch 256, 200 epochs, 3 seeds; one "
                   "checkpoint per seed trained on every collected state"),
        "labels": str(pooled_labels_path(root)),
        "seeds": {}, "missing": [], "all_pass": True,
    }
    for seed in TRAIN_SEEDS:
        path = train_report_path(root, seed)
        if not path.is_file():
            out["missing"].append(str(path))
            out["all_pass"] = False
            continue
        rec = read_json(path)
        out["seeds"][str(seed)] = {
            "report": str(path), "sha256": sha256_file(path),
            "states": rec["states"], "labels": rec["labels"],
            "bombs_remaining_distribution": rec.get("bombs_remaining_distribution"),
            "loss_first_epoch": rec["loss_first_epoch"],
            "loss_last_epoch": rec["loss_last_epoch"],
            "final_train_bce": rec["final_train_bce"],
            "train_accuracy": rec["train_accuracy"],
            "mean_predicted_prob_bomb0": rec.get("mean_predicted_prob_bomb0"),
            "mean_predicted_prob_bomb1": rec.get("mean_predicted_prob_bomb1"),
            "ckpt": rec["ckpt"], "wall_s": rec["wall_s"]}
    write_json(pathlib.Path(args.out) if args.out else root / "train_summary.json",
               out)
    log(f"train-summary: {len(out['seeds'])}/{len(TRAIN_SEEDS)} seeds -> "
        f"{args.out or (root / 'train_summary.json')}")
    return 0 if out["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: audio-audit / process-audit / manifest / publish
# --------------------------------------------------------------------------- #
def mode_audio_audit(args: argparse.Namespace) -> int:
    """``v450_cf_label.mode_audit`` on this round's ``--work-dir``, own ``--out``."""

    namespace = argparse.Namespace(
        work_dir=args.work_dir,
        out=args.out or str(pathlib.Path(args.work_dir) / "audio_audit.json"))
    return v450().mode_audit(namespace)


def mode_process_audit(args: argparse.Namespace) -> int:
    needle = str(args.data_dir_filter or WORK)
    out = pathlib.Path(args.out) if args.out else WORK / "process_residue.json"
    proc = subprocess.run(["pgrep", "-af", args.engine_name],
                          capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if needle in line]
    payload = {
        "schema": SCHEMA_PROC, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("this round's checklist: 0 engine processes of this round "
                        "may survive its own run (filtered by --data-dir-filter)"),
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
        raise BombDagError(f"not a directory: {root}")
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
    wanted = [
        (pooled_labels_path(root), "labels_bombdag.jsonl"),
        (pooled_stats_path(root), "labels_stats.json"),
        (root / "preflight.json", "preflight.json"),
        (root / "ckpt_assertions.json", "ckpt_assertions.json"),
        (root / "ckpt_assertions_final.json", "ckpt_assertions_final.json"),
        (root / "fidelity_d3.json", "fidelity_d3.json"),
        (root / "pilot.json", "pilot.json"),
        (root / "train_summary.json", "train_summary.json"),
        (root / "diagnostics.json", "diagnostics.json"),
        (root / "bomb_usage.json", "bomb_usage.json"),
        (root / "audio_audit.json", "audio_audit.json"),
        (root / "process_residue.json", "process_residue.json"),
        (root / "commands.tsv", "commands.tsv"),
    ]
    for name in ("exec.sh", "pilot.sh", "collect.sh", "train.sh", "eval.sh",
                 "RUN.md"):
        wanted.append((root / name, name))
    for seed in TRAIN_SEEDS:
        wanted.append((train_report_path(root, seed), f"train_seed{seed}.json"))
        wanted.append((policy_ckpt_path(root, seed), f"ckpt_seed{seed}.pt"))
    for budget in BUDGETS:
        for seed in TRAIN_SEEDS:
            path = eval_path(root, budget, seed)
            wanted.append((path, f"eval_d{budget}_seed{seed}.jsonl"))
            wanted.append((path.with_suffix(".summary.json"),
                           f"eval_d{budget}_seed{seed}.summary.json"))
    for path in sorted((root / "collect").glob("seed*_w*.json")):
        wanted.append((path, f"collect/{path.name}"))
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
               "v860_bombdag_sha256": sha256_file(pathlib.Path(__file__).resolve())}
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
        description="v8.6.0 BOMBDAGGER-001: recollect the 36-way labels from a "
                    "behaviour policy that spends bombs (v4.5.0 S + R5)")
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "assert-ckpt", "pilot", "collect",
                                 "stats", "train", "train-summary", "eval",
                                 "diagnostics", "audio-audit", "process-audit",
                                 "manifest", "publish"))
    parser.add_argument("--seed", type=int, default=0,
                        help="behaviour/training seed (0/1/2)")
    parser.add_argument("--policy-seed", type=int, default=None,
                        help="eval: alias of --seed")
    parser.add_argument("--windows", default="13,14,15,16")
    parser.add_argument("--decisions", type=int, default=DECISIONS,
                        help=f"eval budget {BUDGETS} / collect cap {DECISIONS}")
    parser.add_argument("--decision-frames", type=int, default=DECISION_FRAMES)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--horizon", type=int, default=H_DEFAULT)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--kind", default="stochastic",
                        choices=("stochastic", "greedy"))
    parser.add_argument("--quota", type=int, default=QUOTA_DEFAULT,
                        help="collect: states per (seed, window) cell")
    parser.add_argument("--max-episodes", type=int, default=MAX_EPISODES_DEFAULT)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--work-dir", default=str(WORK))
    parser.add_argument("--labels-out", default="")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--labels-file", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--usage-out", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--phase", default="start", choices=("start", "final"))
    parser.add_argument("--engine-name", default="th08-modern")
    parser.add_argument("--data-dir-filter", default="")
    parser.add_argument("--dir", default="")
    args = parser.parse_args(argv)

    if args.policy_seed is not None:
        args.seed = int(args.policy_seed)
    else:
        args.policy_seed = int(args.seed)

    engine_modes = ("pilot", "collect", "eval")
    if args.mode in engine_modes:
        prepare_env()
        tag = args.tag or f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
        args.tag = tag
        log_dir = pathlib.Path(args.work_dir) / "runs" / tag
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_FH = (log_dir / "run.log").open("a", encoding="utf-8")

    try:
        if args.mode == "preflight":
            return mode_preflight(args)
        if args.mode == "assert-ckpt":
            return mode_assert(args)
        if args.mode == "pilot":
            return mode_pilot(args)
        if args.mode == "collect":
            return mode_collect(args)
        if args.mode == "stats":
            return mode_stats(args)
        if args.mode == "train":
            return mode_train(args)
        if args.mode == "train-summary":
            return mode_train_summary(args)
        if args.mode == "eval":
            return mode_eval(args)
        if args.mode == "diagnostics":
            return mode_diagnostics(args)
        if args.mode == "audio-audit":
            return mode_audio_audit(args)
        if args.mode == "process-audit":
            return mode_process_audit(args)
        if args.mode == "manifest":
            return mode_manifest(args)
        return mode_publish(args)
    except BombDagError as exc:
        log(f"FAILED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
