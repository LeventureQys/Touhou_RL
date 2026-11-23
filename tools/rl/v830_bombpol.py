#!/usr/bin/env python3
"""v8.3.0 BOMBPOL-001 — a 36-action policy (18 candidates x bomb / no bomb) that
decides for itself when to bomb, trained on counterfactual labels.

What this is
------------
The executor for `Document/Update/Alpha/v8.3.0/PREREGISTRATION_BOMBPOL_001.md`
(FROZEN 2026-11-22, before any v8.3.0 label or training run).

v8.0.0/v8.1.0/v8.2.0 proved the *script* rule "bomb when a bullet is closer than
T px" moves the frozen v4.5.0 `S` policy from 13.4444 % (484/3,600) to 100 % at
300 decisions — but the bomb was **an external intervention**, the policy never
chose it.  This round widens the action space from 18 to **36**
(``action = cand * 2 + bomb``) and labels every state of the frozen v6.0.0 batch
along **all 36 actions**, so the policy can learn the bomb decision itself.

    states  = the SAME 5,166 states as `.tmp_v600_obsp/labels_pipe.jsonl`
              (located by (seed, window, decision, frame); the state itself is
              unchanged, only 18 more labels per state)
    labels  = 36 per state, each held for H = 120 frames, bomb=1 really presses
              BTN_BOMB on the label's first frame (prereg §1)
    model   = Policy(106, 64, 36) — same 2-layer shape as v4.5.0/v6.0.0
    train   = BCE, Adam 1e-3, batch 256, 200 epochs, 3 seeds (prereg §1)
    eval    = 3 budgets (300 / 600 / 900 decisions) x 3 seeds x 4 dev windows
              x 300 episodes, stochastic, live engine

Modes
-----
==================================  =============================================
``--mode preflight``                 pins (sha256 of every imported file **and**
                                     of the 5,166-state label file), feature dim
                                     (106), the frozen recount of that file
                                     (states / groups / windows), engine gate.
``--mode assert-ckpt --phase P``      engine sha256 + the three frozen v4.5.0
                                     ``round2.pt`` (``4abee069`` / ``a93746bb`` /
                                     ``c69960fc``) that are the behaviour-policy
                                     source of the 5,166 states, plus the sha256
                                     of every protected source.
``--mode selfcheck``                 two hard gates (prereg §2.1 S-label):
                                     (1) **18/18 at H = 600** — the bomb-aware
                                     labeller, with ``bomb = 0`` on all 18
                                     candidates, reproduces R24's 18
                                     ``first_death_offset`` bit for bit;
                                     (2) **36/36** — one live state labelled
                                     along all 36 actions: every label present,
                                     no ``None``, every ``bomb=1`` path really
                                     spent a bomb and no ``bomb=0`` path did;
                                     plus (if the pooled file exists) every
                                     collected row carries exactly 36 labels.
``--mode collect --seed S --round R`` one DAgger group: replay the recorded
                                     trajectory of one (seed, round) on the four
                                     train windows (the behaviour action of every
                                     decision is in the v6.0.0 row), assert the
                                     replayed state is the recorded state
                                     (frame + state dict + 106-dim features
                                     sha256), then label it along **36** actions.
``--mode merge``                     pool the nine group files into
                                     ``labels_bomb36.jsonl`` and write
                                     ``labels_bomb36_stats.json`` (36/36
                                     completeness, bomb=0 vs bomb=1 safe rate).
``--mode train --seed S``            Policy(106,64,36), BCE, Adam 1e-3, batch 256,
                                     200 epochs, no early stopping, dev never
                                     read.
``--mode eval --decisions {300,600,900} --policy-seed S``
                                     the frozen v4.5.0 episode loop with a 36-way
                                     policy: 4 dev windows, 300 episodes each,
                                     ``stochastic`` = softmax(z/1) sample, the
                                     sampled ``bomb`` bit is executed as
                                     ``cand_to_mask(cand, bomb)`` on the
                                     decision's first frame.
``--mode equiv``                     the control that ``run_episode36`` is the
                                     v8.0.0 loop: same window, same action
                                     sequence, bomb bit forced to 0, compared
                                     against ``v800_bomb.run_episode_bomb``
                                     (``--bomb-rule none``) on the live engine.
``--mode diagnostics``               offline per-budget / per-(seed,window)
                                     counts, means, medians, quantiles, the bomb
                                     usage table, the first-bomb-decision
                                     distribution and the exhaustion rate, side
                                     by side with the no-bomb baseline and the
                                     80 px rule.  **No test, no interval.**
``--mode audio-audit``               AGENTS.md §9.3 on this round's own data
                                     dirs; writes this round's own file only.
``--mode process-audit``             this round's own engine processes must be 0
                                     (``pgrep -af`` filtered by ``--data-dir``).
``--mode manifest --dir D``          sha256 + bytes of every file in D.
``--mode publish``                   copy the raw artifacts into
                                     ``Document/Update/Alpha/v8.3.0/evidence/
                                     BOMBPOL/`` and write the manifest.

Reused, never re-implemented (imported by path, sha256 pinned; a moved file is
exit 2 and nothing runs)
---------------------------------------------------------------------------
* ``tools/rl/v450_cf_label.py`` — ``start_parent`` (HELLO + the 2,165-checkpoint
  prefix + PARK + DEBUG), ``prepare_env`` (dummy audio, no display),
  ``prepare_data_dir``, ``masks_for``, ``policy_action``, ``Digest``,
  ``mode_audit`` (the audio audit), ``check_engine``;
* ``tools/rl/v432_fork_arm.py`` — ``ForkParent`` (warm RESET, SPAWN/CHILD/QUIT,
  ``walk_prefix``), ``build_private_data_dir`` (muted ``th08.cfg`` regenerated
  and ``--inspect``ed before every engine start), ``_step_fields``,
  ``fields``, and ``--mode fidelity`` (the D3 gate, run through v432 itself);
* ``tools/rl/v431_arm_a.py`` — ``build_features`` (the frozen 106-dim
  observation) and ``Policy`` (the frozen 2-layer model);
* ``tools/rl/v431_dense_driver.py`` — the recorded prefix masks/hashes;
* ``tools/rl/v800_bomb.py`` — ``run_episode_bomb`` and ``nearest_bullet_distances``
  (the ``--mode equiv`` reference);
* ``tools/rl/v600_obs_pipe.py`` — **not imported**: its ``PINNED`` table is stale
  (it pins v450 at ``978c7cca…`` and v432 at ``c018758b…``, both of which moved
  after v6.0.0 finished), so importing it would abort on its own pin check.  Its
  observation code is re-used through ``v431_arm_a.build_features`` (identical at
  K = 12, which v6.0.0 proved on 240 live states) and the rows of its output file
  are read **read-only**;
* ``touhou_rl/v420/counterfactual._terminal_of`` — the death rule of R24/C4.

Copied, not imported (a pinned file may not be edited this round)
-----------------------------------------------------------------
* ``label_child_action`` / ``label_state36`` — ``v450_cf_label.label_child`` /
  ``label_state`` line for line, with exactly two behavioural additions:
  (a) the mask is ``cand_to_mask(cand, bomb)`` on the first frame when
  ``bomb = 1`` and ``cand_to_mask(cand, 0)`` afterwards, (b) every child's
  ``bombs_remaining`` is read from the ``STATE`` reply the loop already fetches
  (no extra round trip) so the label row can carry ``bombs_used`` /
  ``bombs_remaining``.  Proven by ``--mode selfcheck`` gate 1 (18/18 R24) and by
  every collected row reproducing the v6.0.0 ``features_sha256_k12_ref``.
* ``run_episode36`` — ``v800_bomb.run_episode_bomb`` line for line, with the rule
  trigger replaced by the policy's sampled bomb bit.  Proven by
  ``--mode equiv`` on the live engine.
* ``Policy36`` — ``v431_arm_a.Policy`` with its output layer widened 18 -> 36
  (the frozen class hardcodes ``N_CAND = 18`` and may not be edited).

Failure policy (fail closed)
----------------------------
* a pinned import whose sha256 moved -> exit 2, nothing runs;
* D3 fidelity != 5/5, labeller != 18/18 at H = 600, a 36-path label that is
  incomplete, a bomb=1 path that did not spend a bomb on a fresh state, or a
  bomb=0 path that did -> exit 2 and the round is void (S-label / S-fork);
* a replayed state whose frame, state dict or ``features_sha256_k12_ref`` does
  not match the v6.0.0 row -> abort the group (rc 4) and the partial file stays
  ``*.partial``; nothing is retried into a plausible-looking number;
* fewer than 4,000 collected states -> rc 4 and the report says so (S-label);
* every episode record carries ``invalid`` / ``error``; an invalid episode
  aborts the cell (rc 4) instead of being dropped.

Not in this file
----------------
* No statistics: counts, means, medians, quantiles and distinct-value counts
  only.  P1-P4 belong to the independent analysis pass (prereg §2).
* No pixel arm, no parameter search, no dev reading during collection/training.
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

WORK = REPO / ".tmp_v830_bombpol"
PUBLISH = REPO / "Document" / "Update" / "Alpha" / "v8.3.0" / "evidence" / "BOMBPOL"
V600_ROOT = REPO / ".tmp_v600_obsp"
V600_LABELS = V600_ROOT / "labels_pipe.jsonl"
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
    "tools/rl/v600_obs_pipe.py":
        "edf02c6831bbb896e30c115b6f6805bde16c417b69a0e201308ffe3fe9c4fae4",
    "tools/rl/v800_bomb.py":
        "f884a4be1abebb2bc55ce93bd5d3cb1bea31b26dc3c26b2d98cdbfa15a0bb918",
    "touhou_rl/v420/contracts.py":
        "1357c69c32a74fda20553a3e6b290ca2102b7cc5c52e47e9fdc6eb9e08c946c1",
    "touhou_rl/v400/gym/actions.py":
        "a67145b2bec7c96b3aacaa2514e78cdadf50b056473d15b4ae8122c19c393eff",
}
# -- the 5,166-state batch this round re-labels (read-only) ------------------
V600_LABELS_SHA256 = (
    "eeb2f9d6429dbc1430d48b1d18d0b1c127e851d86e476fa26a70124e94e04260")
V600_LABELS_BYTES = 41374900
V600_LABELS_STATES = 5166
V600_EVIDENCE = REPO / "Document" / "Update" / "Alpha" / "v6.0.0" / "evidence" / "OBSP"

SCHEMA_LABEL = "v830-bombpol-label/1"
SCHEMA_SELFCHECK = "v830-bombpol-selfcheck/1"
SCHEMA_COLLECT = "v830-bombpol-collect/1"
SCHEMA_MERGE = "v830-bombpol-merge/1"
SCHEMA_LABEL_STATS = "v830-bombpol-label-stats/1"
SCHEMA_TRAIN = "v830-bombpol-train/1"
SCHEMA_TRAIN_SUMMARY = "v830-bombpol-train-summary/1"
SCHEMA_EVAL = "v830-bombpol-eval/1"
SCHEMA_EVAL_SUMMARY = "v830-bombpol-eval-summary/1"
SCHEMA_DIAG = "v830-bombpol-diagnostics/1"
SCHEMA_BOMB_USAGE = "v830-bombpol-bomb-usage/1"
SCHEMA_AUDIT = "v830-bombpol-audio-audit/1"
SCHEMA_PROC = "v830-bombpol-process-audit/1"
SCHEMA_MANIFEST = "v830-bombpol-manifest/1"
SCHEMA_PUBLISH = "v830-bombpol-publish/1"
SCHEMA_ASSERT = "v830-bombpol-ckpt-assertions/1"
SCHEMA_PREFLIGHT = "v830-bombpol-preflight/1"
SCHEMA_EQUIV = "v830-bombpol-equivalence/1"
SCHEMA_SMOKE = "v830-bombpol-smoke/1"
PRODUCER = "tools/rl/v830_bombpol.py (live engine, HarnessClient role=shadow)"
PREREG = "Document/Update/Alpha/v8.3.0/PREREGISTRATION_BOMBPOL_001.md"

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- frozen geometry (unchanged from v4.5.0/v6.0.0/v8.0.0) -------------------
PREFIX_FRAMES = 8663            # engine frame every parent PARKs at
CHECKPOINTS_IN_PREFIX = 2165    # recorded hashes checked by every prefix walk
R24_SEED = 12                   # the fidelity/selfcheck window
R24_OFFSETS: tuple[int, ...] = (
    195, 124, 58, 306, 286, 201, 315, 310, 176, 256, 118, 60, 361, 286, 269, 520,
    391, 304,
)
R24_SOURCE = "Document/Update/Alpha/v4.3.1/evidence/R24/c4_dense.json"

# -- frozen split (prereg §1: the v4.5.0 / v6.0.0 protocol) -----------------
TRAIN_WINDOWS: tuple[int, ...] = (13, 14, 15, 16)
DEV_WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
TRAIN_SEEDS: tuple[int, ...] = (0, 1, 2)
ROUNDS: tuple[int, ...] = (0, 1, 2)

# -- frozen design (prereg §1) ----------------------------------------------
N_CAND = 18                     # candidates (focus * 9 + move)
N_ACTIONS = 36                  # actions = cand * 2 + bomb
BOMB_VALUES: tuple[int, ...] = (0, 1)
DECISION_FRAMES = 2
DECISIONS = 300
BUDGETS: tuple[int, ...] = (300, 600, 900)
H_DEFAULT = 120                 # every collected label (60 decisions)
H_SELFCHECK = 600               # the R24 comparison only
SELFCHECK_ACTIONS = N_ACTIONS   # gate 2 needs all 36 on one live state
EPISODES = 300
HIDDEN = 64
FEATURE_DIM = 106               # v431_arm_a.feature_dim(): 10 + 6*12 + 20 + 4
LR = 1e-3
BATCH = 256
EPOCHS = 200
TEMPERATURE = 1.0               # stochastic evaluation only
GEN_CONST = 7                   # v4.5.0: torch.Generator().manual_seed(seed*1000+7)
DEFAULT_TIMEOUT = 300.0
MIN_STATES = 4000               # prereq §2.1 S-label
STATE_KEYS = ("frame", "player_x", "player_y", "player_power", "lives_remaining",
              "bombs_remaining", "graze", "deaths", "stage", "score")

# -- the three frozen v4.5.0 `S` checkpoints (the behaviour-policy source) ---
FROZEN_ROUND2_SHA256: dict[int, str] = {
    0: "4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc",
    1: "a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4",
    2: "c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01",
}
PROTECTED_FILES = (
    "tools/rl/v432_fork_arm.py",
    "tools/rl/v450_cf_label.py",
    "tools/rl/v431_arm_a.py",
    "tools/rl/v600_obs_pipe.py",
    "tools/rl/v800_bomb.py",
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)

# -- the two references this round is placed beside (numbers only, no test) --
BASELINE_NOBOMB = {
    "arm": "v4.5.0 `S`, 18 actions, bomb never pressed (v8.0.0 arm A0)",
    "episodes": 3600, "survived": 484, "rate": 0.134444,
    "mean_alive": 216.9714, "median_alive": 107.0,
    "budget_decisions": 300,
    "source": "Document/Update/Alpha/v8.0.0/evidence/BOMB/RUN.md §4",
}
RULE_80PX = {
    "arm": "v8.0.0 A3 / v8.1.0 A3 — script rule `nearest_px < 80 and bombs > 0`",
    "budget_decisions": [300, 600, 900],
    "episodes_per_budget": 3600,
    "survived": {300: 3600, 600: 3422, 900: 2322},
    "rate": {300: 1.0, 600: 0.9505, 900: 0.6451},
    "source": ("300 decisions: Document/Update/Alpha/v8.0.0/evidence/BOMB/RUN.md §4 "
               "(3,600/3,600).  600/900 decisions: **quoted from** "
               "Document/Update/Alpha/v8.3.0/PREREGISTRATION_BOMBPOL_001.md §0 "
               "(v8.1.0 BOMBFAR, still '收尾中' — its raw run in .tmp_v810_bomb was "
               "interrupted, no complete summary exists on disk, so these two "
               "numbers are NOT recomputed here)"),
}

_LOG_FH: Any = None
_M450: Any = None
_M432: Any = None
_ARM: Any = None
_M800: Any = None


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class BombPolError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v830] {message}"
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
        raise BombPolError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v450():
    """``tools/rl/v450_cf_label.py`` — imported, pinned, never modified."""

    global _M450
    if _M450 is None:
        _M450 = load_module("v450_cf_label", REPO / "tools" / "rl" / "v450_cf_label.py")
        _M450._LOG_FH = _LOG_FH
        _M450.m432()._LOG_FH = _LOG_FH
    return _M450


def m432():
    global _M432
    if _M432 is None:
        _M432 = load_module("v432_fork_arm", REPO / "tools" / "rl" / "v432_fork_arm.py")
        _M432._LOG_FH = _LOG_FH
    return _M432


def arm_a():
    """``tools/rl/v431_arm_a.py`` — the frozen 106-dim observation and model."""

    global _ARM
    if _ARM is None:
        _ARM = load_module("v431_arm_a", REPO / "tools" / "rl" / "v431_arm_a.py")
    return _ARM


def m800():
    """``tools/rl/v800_bomb.py`` — only for ``--mode equiv``'s reference loop."""

    global _M800
    if _M800 is None:
        _M800 = load_module("v800_bomb", REPO / "tools" / "rl" / "v800_bomb.py")
        _M800._LOG_FH = _LOG_FH
    return _M800


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


def first_bomb_histogram(values: Sequence[int]) -> dict[str, int]:
    """Buckets for "on which decision did the policy first press the bomb"."""

    edges = ((0, 0), (1, 4), (5, 9), (10, 24), (25, 49), (50, 99), (100, 199),
             (200, 10 ** 9))
    out: dict[str, int] = {}
    for low, high in edges:
        label = (str(low) if low == high else
                 (f">={low}" if high >= 10 ** 9 else f"{low}-{high}"))
        out[label] = sum(1 for v in values if low <= int(v) <= high)
    return out


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


def features_sha256(feats: Any) -> str:
    import numpy as np

    array = np.asarray(feats, dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def generator_seed(seed: int) -> int:
    """v4.5.0 §2.3's rule, verbatim: ``torch.Generator().manual_seed(seed*1000+7)``."""

    return int(seed) * 1000 + GEN_CONST


def check_pins() -> dict[str, Any]:
    """Every imported/protected file must still hash to the pinned value."""

    out: dict[str, Any] = {"pinned": {}, "all_match": True}
    for rel, expected in PINNED.items():
        path = REPO / rel
        got = sha256_file(path) if path.is_file() else None
        match = got == expected
        out["pinned"][rel] = {"sha256": got, "expected": expected, "match": match}
        if not match:
            out["all_match"] = False
    path = V600_LABELS
    got = sha256_file(path) if path.is_file() else None
    out["v600_labels"] = {"path": str(path), "sha256": got,
                          "expected": V600_LABELS_SHA256,
                          "bytes": path.stat().st_size if path.is_file() else None,
                          "match": got == V600_LABELS_SHA256}
    if not out["v600_labels"]["match"]:
        out["all_match"] = False
    return out


# --------------------------------------------------------------------------- #
# the 36-action space
# --------------------------------------------------------------------------- #
def action_of(cand: int, bomb: int) -> int:
    """``action = cand * 2 + bomb`` (prereg §1)."""

    if not 0 <= int(cand) < N_CAND:
        raise BombPolError(f"cand {cand} outside [0, {N_CAND})")
    if int(bomb) not in BOMB_VALUES:
        raise BombPolError(f"bomb {bomb} outside {BOMB_VALUES}")
    return int(cand) * 2 + int(bomb)


def split_action(action: int) -> tuple[int, int]:
    if not 0 <= int(action) < N_ACTIONS:
        raise BombPolError(f"action {action} outside [0, {N_ACTIONS})")
    cand, bomb = divmod(int(action), 2)
    return int(cand), int(bomb)


_MASK_TABLE: dict[int, tuple[int, int]] = {}


def action_masks(action: int) -> tuple[int, int]:
    """``(first-frame mask, remaining-frame mask)`` of one 36-way action.

    ``bomb = 1`` sends ``cand_to_mask(cand, 1)`` on the label's **first** frame —
    the prerequisite prereg §1 writes down — and the same candidate without the
    bomb bit on every later frame.  That is exactly what the evaluation loop
    sends (``v800_bomb.run_episode_bomb``: bomb on the decision's first frame,
    plain candidate after), so a ``safe`` label means "pressing the bomb on this
    decision's first frame is at least this good".
    """

    from touhou_rl.v420 import contracts as C

    key = int(action)
    cached = _MASK_TABLE.get(key)
    if cached is not None:
        return cached
    cand, bomb = split_action(key)
    first = int(C.cand_to_mask(cand, bomb))
    rest = int(C.cand_to_mask(cand, 0)) if bomb else first
    _MASK_TABLE[key] = (first, rest)
    return first, rest


def describe_actions() -> list[dict[str, Any]]:
    rows = []
    for action in range(N_ACTIONS):
        cand, bomb = split_action(action)
        first, rest = action_masks(action)
        rows.append({"action": action, "cand": cand, "bomb": bomb,
                     "first_frame_mask": first, "remaining_frame_mask": rest})
    return rows


# --------------------------------------------------------------------------- #
# the labeller: 36 children, H frames each  (copy of v450.label_child/label_state)
# --------------------------------------------------------------------------- #
def label_child_action(parent: Any, handle: int, action: int, horizon: int,
                       digest: Any, *,
                       parent_state: dict[str, Any] | None = None,
                       state_frame: int | None = None) -> dict[str, Any]:
    """Hold one 36-way action for up to ``horizon`` frames; report the death.

    ``safe = 1`` iff ``deaths`` did not increase inside the horizon (v4.5.0
    §2.1); ``alive_frames`` is the number of frames executed **before** the first
    death and ``horizon`` when there was none (R24's rule, reused through
    ``counterfactual._terminal_of``).  The bomb bookkeeping is read from the
    ``STATE`` reply this loop already fetches: ``bombs_start``,
    ``bombs_end``, ``bombs_used`` and ``bomb_decrement_frame`` (the frame index
    at which the counter first dropped — v8.0.0 measured 2 for an honoured
    press).
    """

    from touhou_rl.v400.labels.state import state_from_reply
    from touhou_rl.v420 import counterfactual as cf

    m = m432()
    cand, bomb = split_action(action)
    first_mask, rest_mask = action_masks(action)
    out: dict[str, Any] = {
        "action": int(action), "cand": int(cand), "bomb": int(bomb),
        "mask_first_frame": int(first_mask), "mask_remaining_frames": int(rest_mask),
        "handle": int(handle), "safe": None, "alive_frames": None,
        "frames_executed": 0, "terminal": None, "baseline_frame": None,
        "deaths_start": None, "bombs_start": None, "bombs_end": None,
        "bombs_used": None, "bomb_decrement_frame": None,
        "baseline_state_equal_parent": None, "error": None, "wall_s": None}
    started = time.perf_counter()
    try:
        reading, reply = parent.child_state(handle)
        digest.add(reply)
        deaths0 = float(reading["deaths"] or 0.0)
        stage0 = int(reading["stage"] or 0)
        bombs0 = int(reading["bombs_remaining"] or 0)
        out["deaths_start"] = deaths0
        out["bombs_start"] = bombs0
        out["baseline_frame"] = int(reading["frame"])
        if state_frame is not None:
            out["baseline_frame_ok"] = out["baseline_frame"] == int(state_frame)
            if not out["baseline_frame_ok"]:
                raise BombPolError(
                    f"action {action}: child starts at frame {out['baseline_frame']}, "
                    f"the parked parent is at {state_frame}")
        if parent_state is not None:
            out["baseline_state_equal_parent"] = dict(reading) == dict(parent_state)
        died = False
        bombs_end = bombs0
        for index in range(1, int(horizon) + 1):
            mask = first_mask if index == 1 else rest_mask
            reply = parent.child(handle, f"STEP {int(mask)} 1")
            digest.add(reply)
            step = m._step_fields(reply)
            if step["advanced"] == 1:
                out["frames_executed"] = index
            elif not (step["exited"] or step["game_over"]):
                raise BombPolError(f"action {action}: CHILD STEP advanced "
                                   f"{step['advanced']} frames: {reply!r}")
            state_reply = parent.child(handle, "STATE")
            digest.add(state_reply)
            reading = cf._require_state(state_reply, state_from_reply)
            remaining_now = int(reading["bombs_remaining"] or 0)
            bombs_end = remaining_now
            if out["bomb_decrement_frame"] is None and remaining_now < bombs0:
                out["bomb_decrement_frame"] = index
            term = cf._terminal_of(
                reading, deaths_start=deaths0, stage0=stage0,
                hash_hex=(step["hash"] or ""), frame=int(reading["frame"]),
                exited=step["exited"], game_over=step["game_over"])
            if term is not None:
                out["terminal"] = str(term.get("terminal"))
                if term.get("engine_dead"):
                    died = True
                    out["alive_frames"] = int(out["frames_executed"]) - 1
                else:
                    # stage_clear: no death inside the horizon, but the segment
                    # ended.  Recorded explicitly; never silently called a death.
                    out["alive_frames"] = int(horizon)
                break
        if out["alive_frames"] is None:
            out["alive_frames"] = int(horizon)
        out["safe"] = 0 if died else 1
        out["bombs_end"] = int(bombs_end)
        out["bombs_used"] = int(bombs0) - int(bombs_end)
    except Exception as exc:                                          # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["safe"] = None
        out["alive_frames"] = None
    out["wall_s"] = round(time.perf_counter() - started, 4)
    return out


def label_state36(parent: Any, *, state: dict[str, Any], horizon: int,
                  park_reply: str,
                  actions: Sequence[int] | None = None) -> tuple[dict[str, Any],
                                                                  list[dict[str, Any]]]:
    """One state: ``len(actions)`` spawns, horizons, quits, one digest."""

    m = m432()
    digest = v450().Digest()
    state_frame = int(state["frame"])
    order = (list(range(N_ACTIONS)) if actions is None else [int(a) for a in actions])
    children: list[dict[str, Any]] = []
    slots: list[int] = []
    started = time.perf_counter()
    for action in order:
        handle, spawn_reply = parent.spawn()
        digest.add(spawn_reply)
        slots.append(int(handle))
        run = label_child_action(parent, handle, action, horizon, digest,
                                 parent_state=state, state_frame=state_frame)
        quit_reply = parent.quit_child(handle)
        digest.add(quit_reply)
        run["spawn_reply"] = spawn_reply
        run["quit_reply"] = quit_reply
        if run["error"] is not None:
            raise BombPolError(f"action {action} on frame {state_frame}: "
                               f"{run['error']}")
        children.append(run)
    wall = round(time.perf_counter() - started, 4)
    spawn_ok = sum(1 for row in children
                   if str(row["spawn_reply"]).startswith("OK spawned=1"))
    quit_ok = sum(1 for row in children if str(row["quit_reply"]) == "OK bye")
    if spawn_ok != len(children) or quit_ok != len(children):
        raise BombPolError(f"frame {state_frame}: {spawn_ok}/{len(children)} spawns "
                           f"and {quit_ok}/{len(children)} quits succeeded")
    block = {
        "park_reply": park_reply,
        "frame": state_frame,
        "horizon": int(horizon),
        "actions": order,
        "child_replies_digest": digest.hexdigest(),
        "child_reply_lines": digest.lines,
        # the label itself: index a is action a = cand * 2 + bomb (prereg §1)
        "safe": [int(row["safe"]) for row in children],
        "alive_frames": [int(row["alive_frames"]) for row in children],
        "bombs_used": [int(row["bombs_used"]) for row in children],
        "bombs_start": [int(row["bombs_start"]) for row in children],
        "bombs_end": [int(row["bombs_end"]) for row in children],
        "bomb_decrement_frame": [row["bomb_decrement_frame"] for row in children],
        "children": {
            "handles": [row["handle"] for row in children],
            "masks_first_frame": [row["mask_first_frame"] for row in children],
            "masks_remaining_frames": [row["mask_remaining_frames"] for row in children],
            "frames_executed": [row["frames_executed"] for row in children],
            "terminals": [row["terminal"] for row in children],
            "baseline_frames": [row["baseline_frame"] for row in children],
            "wall_s": [row["wall_s"] for row in children],
            "spawn_ok": spawn_ok, "quit_ok": quit_ok,
            "baseline_frame_ok": sum(1 for row in children
                                     if row.get("baseline_frame_ok")),
            "baseline_state_equal_parent": sum(
                1 for row in children if row["baseline_state_equal_parent"]),
            "spawn_reply_example": children[0]["spawn_reply"] if children else None,
            "quit_reply_example": children[0]["quit_reply"] if children else None,
        },
        "label_wall_s": wall,
        "max_concurrent_children": 1,
        "child_slots_used": len(slots),
        "child_slots_cap": v450().MAX_CHILD_SLOTS,
    }
    return block, children


# --------------------------------------------------------------------------- #
# the model: Policy(106, 64, 36)
# --------------------------------------------------------------------------- #
def policy36(*, seed: int, feature_dim: int = FEATURE_DIM, hidden: int = HIDDEN) -> Any:
    """`Policy(106, 64, 36)` — the frozen class with its output widened to 36.

    ``v431_arm_a.Policy`` hardcodes ``N_CAND = 18`` outputs and may not be edited
    this round, so the frozen class is instantiated and only its last layer is
    replaced by ``Linear(hidden, 36)``:``Linear(dim, hidden) -> Tanh ->
    Linear(hidden, 36)``, i.e. the same shape, initialisation order and seed
    discipline as ``Policy(106, 64)``.
    """

    import torch

    torch.manual_seed(int(seed))
    policy = arm_a().Policy(int(feature_dim), int(hidden))
    policy.net[2] = torch.nn.Linear(int(hidden), N_ACTIONS)
    return policy


# --------------------------------------------------------------------------- #
# one episode: v800_bomb.run_episode_bomb's loop + the policy's bomb bit
# --------------------------------------------------------------------------- #
def run_episode36(parent: Any, *, episode: int, policy_seed: int, decisions: int,
                  decide: Any, budget: int,
                  decision_frames: int = DECISION_FRAMES,
                  keep_trace: bool = False) -> dict[str, Any]:
    """One ``SPAWN -> decisions x decision_frames frames -> QUIT`` episode.

    The single difference from ``v800_bomb.run_episode_bomb`` is where the bomb
    bit comes from: the policy's sampled 36-way action (``action = cand*2+bomb``)
    instead of a distance rule::

        frame 0        cand_to_mask(cand, 1)   if the policy sampled bomb = 1
        frame 0        cand_to_mask(cand, 0)   otherwise
        frames 1..k-1  cand_to_mask(cand, 0)   always

    Everything else — the spawn, the baseline ``STATE``, the death rule
    (``counterfactual._terminal_of``), the per-budget decision cap, the
    ``survived`` definition, ``first_death_offset = frames_at_death - 1`` and the
    mandatory ``QUIT`` — is the frozen loop, line for line.
    """

    from touhou_rl.v420 import counterfactual as cf
    from touhou_rl.v420 import contracts as C

    started = time.perf_counter()
    record: dict[str, Any] = {
        "schema": SCHEMA_EVAL, "kind": "episode", "mode": "eval",
        "budget_decisions": int(budget), "episode": int(episode),
        "policy_seed": int(policy_seed), "window": int(parent.window),
        "decisions_cap": int(decisions), "decision_frames": int(decision_frames),
        "return_decisions": 0.0, "survived": False, "wall_s": None, "frames": 0,
        "alive_frames": None, "first_death_offset": None,
        "first_death_frame": None, "terminal": None, "decisions_executed": 0,
        "invalid": False, "error": None, "spawn_ms": None, "template": parent.template,
        # -- the bomb audit of this episode (prereg §2 P4 — descriptive only)
        "bombs_start": None, "bombs_end": None, "bombs_used": None,
        "bomb_decisions": [], "bomb_frames": [], "bomb_actions": [],
        "bomb_cands": [], "bomb_emits": 0, "bomb_ignored": 0,
        "bomb_available_at_emit": [], "first_bomb_decision": None,
        "bombs_remaining_at_emit": [], "nearest_px_at_emit": [],
        "decisions_measured": 0, "min_nearest_px": None,
        "actions_sampled": [], "trace": [] if keep_trace else None,
    }
    handle: int | None = None
    try:
        t_spawn = time.perf_counter()
        handle, spawn_reply = parent.spawn()
        record["spawn_ms"] = round((time.perf_counter() - t_spawn) * 1000.0, 4)
        record["handle"] = int(handle)
        record["spawn_reply"] = spawn_reply

        state, _baseline_reply = parent.child_state(handle)
        deaths0 = float(state["deaths"] or 0.0)
        stage0 = int(state["stage"] or 0)
        record["bombs_start"] = int(state["bombs_remaining"] or 0)
        record["baseline"] = {"frame": int(state["frame"]), "deaths": deaths0,
                              "lives": float(state["lives_remaining"] or 0.0),
                              "stage": stage0,
                              "bombs": float(state["bombs_remaining"] or 0.0)}
        if int(state["frame"]) != PREFIX_FRAMES:
            raise BombPolError(f"spawned child starts at frame {state['frame']}, "
                               f"expected {PREFIX_FRAMES}")

        died_at: int | None = None
        terminal: str | None = None
        frames = 0
        mask_cache: dict[int, int] = {}
        for decision in range(int(decisions)):
            hazards = parent.child_hazards(handle)
            cand, bomb = decide(state, hazards, decision)
            if not 0 <= cand < N_CAND:
                raise BombPolError(f"decision {decision}: cand {cand} outside "
                                   f"[0,{N_CAND})")
            action = action_of(cand, bomb)
            if action not in mask_cache:
                first, rest = action_masks(action)
                mask_cache[action] = first
                mask_cache[-action - 1] = rest
            first_mask = mask_cache[action]
            rest_mask = mask_cache[-action - 1]
            record["actions_sampled"].append(int(action))
            bombs_before = int(state["bombs_remaining"] or 0)
            record["decisions_measured"] += 1
            nearest = None
            try:
                px = float(state.get("player_x") or 0.0)
                py = float(state.get("player_y") or 0.0)
                dists = [math.hypot(float(b.position[0]) - px,
                                    float(b.position[1]) - py)
                         for b in hazards.bullets]
                nearest = round(min(dists), 4) if dists else None
            except Exception:                                         # noqa: BLE001
                nearest = None
            if nearest is not None:
                record["min_nearest_px"] = (nearest
                                            if record["min_nearest_px"] is None
                                            else min(record["min_nearest_px"], nearest))

            emit = bool(bomb)
            if emit:
                record["bomb_emits"] += 1
                record["bomb_decisions"].append(int(decision))
                record["bomb_frames"].append(int(frames))
                record["bomb_actions"].append(int(action))
                record["bomb_cands"].append(int(cand))
                record["bombs_remaining_at_emit"].append(bombs_before)
                record["nearest_px_at_emit"].append(nearest)
                if record["first_bomb_decision"] is None:
                    record["first_bomb_decision"] = int(decision)
            if keep_trace:
                record["trace"].append(
                    {"decision": int(decision), "frame": int(state["frame"]),
                     "action": int(action), "cand": int(cand), "bomb": int(bomb),
                     "nearest_px": nearest, "bombs_remaining": bombs_before})

            died_here = False
            emit_decrement_at: int | None = None
            emit_last: int | None = None
            for frame_index in range(int(decision_frames)):
                mask = first_mask if frame_index == 0 else rest_mask
                step = m432()._step_fields(parent.child(handle, f"STEP {mask} 1"))
                if step["advanced"] == 1:
                    frames += 1
                reading, reply = parent.child_state(handle)
                term = cf._terminal_of(
                    reading, deaths_start=deaths0, stage0=stage0,
                    hash_hex=(step["hash"] or ""), frame=int(reading["frame"]),
                    exited=step["exited"], game_over=step["game_over"])
                state = reading
                if emit:
                    remaining_now = int(reading["bombs_remaining"] or 0)
                    emit_last = remaining_now
                    if emit_decrement_at is None and remaining_now < bombs_before:
                        emit_decrement_at = frame_index + 1
                if term is not None:
                    terminal = str(term.get("terminal"))
                    record["terminal_reply"] = reply
                    if term.get("engine_dead"):
                        died_at = frames
                    died_here = True
                    break
            if emit:
                honoured = emit_last is not None and emit_last < bombs_before
                record["bomb_available_at_emit"].append(bool(honoured))
                if not honoured:
                    record["bomb_ignored"] += 1
            if died_here:
                break
            record["decisions_executed"] = decision + 1

        record["return_decisions"] = float(record["decisions_executed"])
        record["survived"] = bool(died_at is None
                                  and record["decisions_executed"] >= int(decisions))
        record["frames"] = int(frames)
        record["first_death_offset"] = None if died_at is None else int(died_at) - 1
        record["first_death_frame"] = (None if died_at is None
                                       else PREFIX_FRAMES + int(died_at))
        record["alive_frames"] = (int(frames) if died_at is None
                                  else int(died_at) - 1)
        record["terminal"] = terminal
        record["bombs_end"] = int(state["bombs_remaining"] or 0)
        record["bombs_used"] = int(record["bombs_start"]) - int(record["bombs_end"])
        record["bombs_ignored_emits"] = int(record["bomb_ignored"])
    except Exception as exc:                                          # noqa: BLE001
        record["invalid"] = True
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if handle is not None:
            record["quit_reply"] = parent.quit_child(handle)
        record["wall_s"] = round(time.perf_counter() - started, 4)
        record["parent_alive"] = parent.alive()
    return record


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def labels_group_path(root: pathlib.Path, seed: int, rnd: int) -> pathlib.Path:
    return root / "labels" / f"seed{int(seed)}" / f"labels_round{int(rnd)}.jsonl"


def collect_summary_path(root: pathlib.Path, seed: int, rnd: int) -> pathlib.Path:
    return root / "collect" / f"seed{int(seed)}" / f"round{int(rnd)}.json"


def pooled_labels_path(root: pathlib.Path) -> pathlib.Path:
    return root / "labels_bomb36.jsonl"


def pooled_stats_path(root: pathlib.Path) -> pathlib.Path:
    return root / "labels_bomb36_stats.json"


def ckpt_path(root: pathlib.Path, seed: int) -> pathlib.Path:
    return root / "ckpt" / f"seed{int(seed)}" / "round2.pt"


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
    rec: dict[str, Any] = {
        "schema": SCHEMA_PREFLIGHT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "pins": check_pins(),
        "launch_env": prepare_env(),
        "action_space": {"candidates": N_CAND, "bombs": list(BOMB_VALUES),
                         "actions": N_ACTIONS,
                         "encoding": "action = cand * 2 + bomb",
                         "bomb1_semantics": ("cand_to_mask(cand, 1) on the label's "
                                             "first frame, cand_to_mask(cand, 0) "
                                             "afterwards (prereg §1)")},
        "frozen": {"TRAIN_WINDOWS": list(TRAIN_WINDOWS),
                   "DEV_WINDOWS": list(DEV_WINDOWS),
                   "train_seeds": list(TRAIN_SEEDS), "rounds": list(ROUNDS),
                   "H_default": H_DEFAULT, "H_selfcheck": H_SELFCHECK,
                   "budgets_decisions": list(BUDGETS),
                   "decisions_cap": DECISIONS, "decision_frames": DECISION_FRAMES,
                   "episodes_per_window": EPISODES,
                   "lr": LR, "batch": BATCH, "epochs": EPOCHS, "hidden": HIDDEN,
                   "temperature": TEMPERATURE, "feature_dim": FEATURE_DIM},
        "engine": {"path": args.engine},
        "v600_states": {"path": str(V600_LABELS), "states_expected": V600_LABELS_STATES},
        "v600_recount": {}, "action_table": describe_actions(),
        "references": {"nobomb": BASELINE_NOBOMB, "rule_80px": RULE_80PX},
    }
    ok = bool(rec["pins"]["all_match"])
    if V600_LABELS.is_file():
        rows = read_jsonl(V600_LABELS)
        groups: dict[str, int] = {}
        for row in rows:
            key = f"s{int(row['seed'])}_r{int(row['round'])}_w{int(row['window'])}"
            groups[key] = groups.get(key, 0) + 1
        rec["v600_recount"] = {
            "states": len(rows), "groups": len(groups), "per_group": groups,
            "windows": sorted({int(row["window"]) for row in rows}),
            "seeds": sorted({int(row["seed"]) for row in rows}),
            "rounds": sorted({int(row["round"]) for row in rows}),
            "states_expected_match": len(rows) == V600_LABELS_STATES,
            "labels_per_state": sorted({len(row["safe"]) for row in rows}),
            "H": sorted({int(row["H"]) for row in rows}),
            "bytes_match": V600_LABELS.stat().st_size == V600_LABELS_BYTES,
        }
        ok = ok and rec["v600_recount"]["states_expected_match"]
    else:
        rec["v600_recount"] = {"error": f"missing {V600_LABELS}"}
        ok = False
    if int(arm_a().feature_dim()) != FEATURE_DIM:
        rec["feature_dim_error"] = (f"v431_arm_a.feature_dim() = "
                                    f"{arm_a().feature_dim()} != {FEATURE_DIM}")
        ok = False
    rec["all_pass"] = ok
    out = pathlib.Path(args.out) if args.out else root / "preflight.json"
    write_json(out, rec)
    log(f"preflight: pins={rec['pins']['all_match']} "
        f"states={rec['v600_recount'].get('states')} "
        f"groups={rec['v600_recount'].get('groups')} all_pass={ok} -> {out}")
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
        "requirement": ("checklist item 1: the behaviour-policy source of the 5,166 "
                        "states is the frozen v4.5.0 DAgger chain, whose final "
                        "checkpoints are round2.pt = 4abee069…/a93746bb…/c69960fc…; "
                        "the protected sources must not have moved"),
        "why": ("the states of `.tmp_v600_obsp/labels_pipe.jsonl` were produced by "
                "round 0 = uniform random and rounds 1/2 = the previous round's "
                "greedy K=12 checkpoint of the same seed (v4.5.0 protocol), so the "
                "three frozen round2.pt are the end of that behaviour chain"),
        "engine": None, "ckpts": {}, "protected": {}, "v830_bomb_py": {},
        "all_pass": False,
    }
    try:
        payload["engine"] = engine_gate(args.engine)
        ok = bool(payload["engine"].get("match"))
        for seed in TRAIN_SEEDS:
            path = CKPT_ROOT / f"seed{seed}" / "round2.pt"
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
        payload["v830_bomb_py"] = {"path": str(me), "sha256": sha256_file(me)}
        payload["v600_labels"] = {
            "path": str(V600_LABELS),
            "sha256": sha256_file(V600_LABELS) if V600_LABELS.is_file() else None,
            "expected": V600_LABELS_SHA256,
            "match": (V600_LABELS.is_file()
                      and sha256_file(V600_LABELS) == V600_LABELS_SHA256)}
        ok = ok and payload["v600_labels"]["match"]
        payload["all_pass"] = bool(ok)
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
    write_json(out_path, payload)
    log(f"assert-ckpt {args.phase}: all_pass={payload['all_pass']} -> {out_path}")
    return 0 if payload["all_pass"] else 2


# --------------------------------------------------------------------------- #
# mode: selfcheck (S-label gate 1: 18/18 at H=600; gate 2: 36/36 complete)
# --------------------------------------------------------------------------- #
def mode_selfcheck(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out_path = pathlib.Path(args.out) if args.out else root / "labeller_selfcheck.json"
    tag = args.tag or f"v830-selfcheck-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    engine = engine_gate(args.engine)
    data_dir, data_info = v450().prepare_data_dir(data_root, R24_SEED,
                                                  args.data_source)
    log(f"selfcheck: data dir {data_dir} cfg muted="
        f"{data_info['cfg']['muted_four_zero']}")

    rec: dict[str, Any] = {
        "schema": SCHEMA_SELFCHECK, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG + " §2.1 (S-label) and §1 (36 actions)",
        "requirement": {
            "gate1_18of18_at_H600": (
                f"the bomb-aware labeller with bomb=0 on all {N_CAND} candidates "
                f"must reproduce R24's first_death_offset bit for bit"),
            "gate2_36of36": (
                f"one live state labelled along all {N_ACTIONS} actions must come "
                f"back complete (every label present, no None), every bomb=1 path "
                f"must really spend a bomb on a fresh state and no bomb=0 path may"),
        },
        "pins": check_pins(), "engine": engine, "data": data_info,
        "launch_env": prepare_env(), "window": R24_SEED,
        "park_frame_expected": PREFIX_FRAMES, "H_gate1": H_SELFCHECK,
        "H_gate2": H_DEFAULT, "reference": list(R24_OFFSETS),
        "reference_source": R24_SOURCE, "actions": N_ACTIONS,
        "gate1": {"rows": [], "matched": 0, "all_pass": False},
        "gate2": {"all_pass": False},
        "collected_file": {}, "all_pass": False, "error": None, "wall_s": None,
    }
    started = time.perf_counter()
    parent = None
    try:
        parent, gate = start_parent(slot=0, window=R24_SEED, engine=args.engine,
                                    data_dir=data_dir, log_dir=run_dir)
        rec["gate"] = {key: gate[key] for key in (
            "hello_role_shadow", "park_ok", "park_frame", "threads_1", "all_pass")}
        rec["gate_prefix"] = gate["prefix"]
        state, state_reply = parent.state()
        rec["parent_state"] = state
        rec["parent_state_reply"] = state_reply
        if int(state["frame"]) != PREFIX_FRAMES:
            raise BombPolError(f"parked parent reports frame {state['frame']}, "
                               f"expected {PREFIX_FRAMES}")

        # ---- gate 1: the labeller, H = 600, bomb = 0 on all 18 ------------
        # `start_parent` already PARKed; v4.5.0's selfcheck labels against that
        # very template without a second PARK, so gate 1 does exactly the same.
        rec["gate1"]["park_reply"] = gate["park_reply"]
        cand_actions = [action_of(cand, 0) for cand in range(N_CAND)]
        block, children = label_state36(parent, state=state, horizon=H_SELFCHECK,
                                        park_reply=gate["park_reply"],
                                        actions=cand_actions)
        rows = []
        for row in children:
            offset = row["alive_frames"] if row["safe"] == 0 else None
            rows.append({
                "cand": row["cand"], "action": row["action"], "bomb": row["bomb"],
                "mask": row["mask_first_frame"], "handle": row["handle"],
                "first_death_offset": offset,
                "r24_offset": R24_OFFSETS[row["cand"]],
                "match": offset == R24_OFFSETS[row["cand"]],
                "alive_frames": row["alive_frames"], "safe": row["safe"],
                "frames_executed": row["frames_executed"],
                "terminal": row["terminal"],
                "baseline_frame": row["baseline_frame"],
                "baseline_state_equal_parent": row["baseline_state_equal_parent"],
                "bombs_start": row["bombs_start"], "bombs_end": row["bombs_end"],
                "bombs_used": row["bombs_used"],
                "error": row["error"], "wall_s": row["wall_s"]})
        rec["gate1"]["rows"] = rows
        rec["gate1"]["block"] = {key: block[key] for key in (
            "frame", "horizon", "child_replies_digest", "child_reply_lines",
            "child_slots_used", "max_concurrent_children", "label_wall_s")}
        rec["gate1"]["children_block"] = block["children"]
        rec["gate1"]["matched"] = sum(1 for row in rows if row["match"])
        rec["gate1"]["all_pass"] = bool(rec["gate1"]["matched"] == N_CAND
                                        and all(row["error"] is None
                                                for row in children))
        rec["gate1"]["bombs_used_total"] = sum(int(row["bombs_used"] or 0)
                                               for row in children)
        for row in rows:
            log(f"  gate1 cand {row['cand']:2d}: offset={row['first_death_offset']} "
                f"r24={row['r24_offset']} match={row['match']}")
        if not rec["gate1"]["all_pass"]:
            raise BombPolError(f"S-label gate 1 failed: {rec['gate1']['matched']}"
                               f"/{N_CAND} offsets match")

        # ---- gate 2: 36/36 on one live state, H = 120 --------------------
        # the parent is parked on the same frame; every child starts there
        state2, _reply2 = parent.state()
        if int(state2["frame"]) != PREFIX_FRAMES:
            raise BombPolError(f"gate 2: parent moved to frame {state2['frame']}")
        park2 = parent.park()
        block2, children2 = label_state36(parent, state=state2, horizon=H_DEFAULT,
                                         park_reply=park2)
        labels_present = [row["alive_frames"] for row in children2]
        complete = bool(len(children2) == N_ACTIONS
                        and all(v is not None for v in labels_present)
                        and all(row["safe"] in (0, 1) for row in children2)
                        and all(row["error"] is None for row in children2))
        bomb1 = [row for row in children2 if row["bomb"] == 1]
        bomb0 = [row for row in children2 if row["bomb"] == 0]
        bombs0_start = int(bomb0[0]["bombs_start"]) if bomb0 else None
        rec["gate2"] = {
            "frame": int(state2["frame"]), "horizon": H_DEFAULT,
            "actions_requested": N_ACTIONS, "actions_labelled": len(children2),
            "labels_present": sum(1 for v in labels_present if v is not None),
            "complete": complete,
            "bomb1_paths": len(bomb1),
            "bomb1_paths_that_spent_a_bomb": sum(1 for row in bomb1
                                                 if int(row["bombs_used"] or 0) == 1),
            "bomb1_paths_that_did_not": [row["action"] for row in bomb1
                                         if int(row["bombs_used"] or 0) != 1],
            "bomb0_paths": len(bomb0),
            "bomb0_paths_that_spent_a_bomb": sum(1 for row in bomb0
                                                 if int(row["bombs_used"] or 0) != 0),
            "bombs_at_state": bombs0_start,
            "decrement_frames": sorted({row["bomb_decrement_frame"]
                                        for row in bomb1
                                        if row["bomb_decrement_frame"] is not None}),
            "bomb1_safe": sum(int(row["safe"]) for row in bomb1),
            "bomb0_safe": sum(int(row["safe"]) for row in bomb0),
            "bomb1_alive_frames": quantiles([row["alive_frames"] for row in bomb1]),
            "bomb0_alive_frames": quantiles([row["alive_frames"] for row in bomb0]),
            "rows": [{"action": row["action"], "cand": row["cand"],
                      "bomb": row["bomb"], "safe": row["safe"],
                      "alive_frames": row["alive_frames"],
                      "bombs_start": row["bombs_start"],
                      "bombs_end": row["bombs_end"],
                      "bombs_used": row["bombs_used"],
                      "bomb_decrement_frame": row["bomb_decrement_frame"],
                      "terminal": row["terminal"]} for row in children2],
            "block": {key: block2[key] for key in (
                "frame", "horizon", "child_replies_digest", "child_reply_lines",
                "child_slots_used", "label_wall_s")},
        }
        rec["gate2"]["all_pass"] = bool(
            complete
            and rec["gate2"]["bomb1_paths_that_spent_a_bomb"] == len(bomb1)
            and rec["gate2"]["bomb0_paths_that_spent_a_bomb"] == 0
            and len(bomb1) == N_CAND and len(bomb0) == N_CAND)
        log(f"  gate2: {rec['gate2']['actions_labelled']}/{N_ACTIONS} labels, "
            f"bomb1 spent {rec['gate2']['bomb1_paths_that_spent_a_bomb']}/{len(bomb1)}"
            f", bomb0 spent {rec['gate2']['bomb0_paths_that_spent_a_bomb']}")
        if not rec["gate2"]["all_pass"]:
            raise BombPolError(f"S-label gate 2 failed: {json.dumps(rec['gate2'], ensure_ascii=False)[:600]}")

        # ---- gate 3 (only if the pooled file already exists) -------------
        pooled = pooled_labels_path(root)
        if pooled.is_file():
            rows = read_jsonl(pooled)
            good = sum(1 for row in rows
                       if len(row["safe"]) == N_ACTIONS
                       and len(row["alive_frames"]) == N_ACTIONS
                       and all(v is not None for v in row["alive_frames"])
                       and all(v in (0, 1) for v in row["safe"])
                       and all((int(s) == 1) == (int(a) == int(row["H"]))
                               for s, a in zip(row["safe"], row["alive_frames"])))
            rec["collected_file"] = {
                "path": str(pooled), "rows": len(rows), "rows_with_36_labels": good,
                "sha256": sha256_file(pooled), "bytes": pooled.stat().st_size,
                "all_pass": bool(rows and good == len(rows)),
                "at_least_min_states": len(rows) >= MIN_STATES}
        else:
            rec["collected_file"] = {"path": str(pooled), "exists": False,
                                     "note": "checked after --mode merge"}
        rec["all_pass"] = bool(rec["gate1"]["all_pass"]
                               and rec["gate2"]["all_pass"]
                               and (rec["collected_file"].get("all_pass", True)))
    except Exception as exc:                                          # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["all_pass"] = False
    finally:
        if parent is not None:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        rec["wall_s"] = round(time.perf_counter() - started, 3)
    rec["verdict"] = (f"PASS: labeller {rec['gate1']['matched']}/{N_CAND} at "
                      f"H={H_SELFCHECK} and {rec['gate2'].get('actions_labelled')}/"
                      f"{N_ACTIONS} labels complete"
                      if rec["all_pass"] else
                      f"FAIL: gate1 {rec['gate1']['matched']}/{N_CAND}, gate2="
                      f"{json.dumps(rec.get('gate2', {}).get('complete'))}, "
                      f"{rec['error'] or 'mismatch'}")
    write_json(out_path, rec)
    log(f"selfcheck all_pass={rec['all_pass']} -> {out_path} ({rec['wall_s']}s)")
    log(rec["verdict"])
    return 0 if rec["all_pass"] else 2


# --------------------------------------------------------------------------- #
# mode: collect (one (seed, round) DAgger group, 4 windows, 36 labels/state)
# --------------------------------------------------------------------------- #
def load_v600_group(seed: int, rnd: int,
                    windows: Sequence[int]) -> dict[int, list[dict[str, Any]]]:
    """The recorded rows of one (seed, round), grouped by window, in order."""

    wanted = {int(w) for w in windows}
    groups: dict[int, list[dict[str, Any]]] = {int(w): [] for w in windows}
    for row in read_jsonl(V600_LABELS):
        if int(row["seed"]) != int(seed) or int(row["round"]) != int(rnd):
            continue
        if int(row["window"]) not in wanted:
            continue
        groups[int(row["window"])].append(row)
    for window in groups:
        groups[window].sort(key=lambda row: int(row["decision"]))
        decisions = [int(row["decision"]) for row in groups[window]]
        if decisions != list(range(len(decisions))):
            raise BombPolError(f"s{seed} r{rnd} w{window}: recorded decisions are "
                               f"not contiguous from 0: {decisions[:8]}…")
    return groups


def mode_collect(args: argparse.Namespace) -> int:
    """Replay one recorded trajectory per window and label every state 36 ways."""

    import numpy as np

    from touhou_rl.v400.gym import protocol as P
    from touhou_rl.v420 import contracts as C
    from touhou_rl.v420 import counterfactual as cf

    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    rnd = int(args.round)
    horizon = int(args.horizon)
    if seed not in TRAIN_SEEDS:
        raise BombPolError(f"--seed must be one of {TRAIN_SEEDS}, got {seed}")
    if rnd not in ROUNDS:
        raise BombPolError(f"--round must be one of {ROUNDS}, got {rnd}")
    if horizon != H_DEFAULT:
        raise BombPolError(f"prereg §1 fixes H = {H_DEFAULT} for every label; got "
                           f"{horizon}")
    windows = parse_windows(args.windows)
    for window in windows:
        if window not in TRAIN_WINDOWS:
            raise BombPolError(f"the 5,166 states live in windows "
                               f"{TRAIN_WINDOWS}, got {window}")
        C.assert_seed_allowed(int(window), "train")

    pins = check_pins()
    if not pins["all_match"]:
        raise BombPolError(f"pinned import moved: {json.dumps(pins)[:800]}")
    engine = engine_gate(args.engine)
    groups = load_v600_group(seed, rnd, windows)
    total_rows = sum(len(v) for v in groups.values())
    if total_rows == 0:
        raise BombPolError(f"s{seed} r{rnd}: no recorded rows in windows {windows}")

    tag = args.tag or f"v830-collect-s{seed}-r{rnd}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    labels = (pathlib.Path(args.labels_out) if args.labels_out
              else labels_group_path(root, seed, rnd))
    labels.parent.mkdir(parents=True, exist_ok=True)
    partial = labels.with_suffix(labels.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    summary_path = (pathlib.Path(args.summary_out) if args.summary_out
                    else collect_summary_path(root, seed, rnd))

    rec: dict[str, Any] = {
        "schema": SCHEMA_COLLECT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "seed": seed, "round": rnd, "windows": list(windows),
        "H": horizon, "actions_per_state": N_ACTIONS,
        "bomb1_semantics": ("cand_to_mask(cand, 1) on the label's first frame, "
                            "cand_to_mask(cand, 0) on frames 2..H"),
        "source_labels": {"path": str(V600_LABELS),
                          "sha256": pins["v600_labels"]["sha256"],
                          "rows_expected": total_rows},
        "state_locator": ("(seed, round, window, decision, frame): the recorded "
                          "behaviour_action of every decision is replayed in order "
                          "and the reached state must match the recorded frame, "
                          "state dict and features_sha256_k12_ref"),
        "pins": pins, "engine": engine, "launch_env": prepare_env(),
        "action_table": describe_actions(),
        "labels_path": str(labels), "status": "failed", "error": None,
        "windows_detail": [], "rows_written": 0, "states_matched": 0,
        "state_costs_s": [], "wall_s": None,
    }
    handle = partial.open("w", encoding="utf-8")
    parents: list[Any] = []
    started = time.perf_counter()
    try:
        for traj_index, window in enumerate(windows):
            rows = groups[window]
            if not rows:
                continue
            data_dir, data_info = v450().prepare_data_dir(
                data_root, window, args.data_source)
            log(f"s{seed} r{rnd} w{window}: {len(rows)} recorded states, cfg muted="
                f"{data_info['cfg']['muted_four_zero']}")
            parent, gate = start_parent(slot=traj_index, window=window,
                                        engine=args.engine, data_dir=data_dir,
                                        log_dir=run_dir)
            parents.append(parent)
            state, _reply = parent.state()
            if int(state["frame"]) != PREFIX_FRAMES:
                raise BombPolError(f"w{window}: parked parent at frame "
                                   f"{state['frame']}, expected {PREFIX_FRAMES}")
            deaths0 = float(state["deaths"] or 0.0)
            stage0 = int(state["stage"] or 0)
            window_rec: dict[str, Any] = {
                "window": int(window), "recorded_states": len(rows),
                "labelled_states": 0, "frames_replayed": 0,
                "state_equal": 0, "features_match": 0, "labels_written": 0,
                "safe_bomb0": 0, "safe_bomb1": 0,
                "bombs_used_bomb0": 0, "bombs_used_bomb1": 0,
                "terminal": None, "gate": {key: gate[key] for key in (
                    "hello_role_shadow", "park_ok", "park_frame", "threads_1",
                    "all_pass")},
                "gate_prefix": {key: gate["prefix"][key] for key in (
                    "hash_checks", "hash_mismatches", "final_frame", "wall_s")},
                "data_dir": str(data_dir), "cfg": data_info["cfg"],
            }
            costs: list[float] = []
            died = False
            for index, row in enumerate(rows):
                if int(row["decision"]) != index:
                    raise BombPolError(f"w{window}: row {index} carries decision "
                                       f"{row['decision']}")
                if int(state["frame"]) != int(row["frame"]):
                    raise BombPolError(
                        f"w{window} d{index}: replayed frame {state['frame']} != "
                        f"recorded frame {row['frame']}; the state cannot be "
                        f"reached by replaying the recorded behaviour actions")
                hazards = P.parse_hazards_reply(parent.ask("HAZARDS"))
                feats, _detail = arm_a().build_features(
                    state=state, hazards=hazards, decision_index=index)
                sha = features_sha256(feats)
                if sha != row["features_sha256_k12_ref"]:
                    raise BombPolError(
                        f"w{window} d{index}: replayed features {sha[:16]} != the "
                        f"recorded {row['features_sha256_k12_ref'][:16]}")
                # the recorded row keeps 10 of the live STATE's 25 fields, so
                # compare exactly those (a whole-dict comparison would be False
                # by construction and would measure nothing).
                stored_state = row["obs"]["state"]
                state_diff = [key for key, value in stored_state.items()
                              if key not in state or state[key] != value]
                state_equal = bool(not state_diff)
                park_reply = parent.park()
                got_frame = m432().fields(park_reply).get("frame")
                if got_frame != str(int(state["frame"])):
                    raise BombPolError(
                        f"w{window} d{index}: PARK reports frame {got_frame}, the "
                        f"read state is at {state['frame']}")
                block, _children = label_state36(parent, state=state,
                                                 horizon=horizon,
                                                 park_reply=park_reply)
                for action, safe in enumerate(block["safe"]):
                    alive = int(block["alive_frames"][action])
                    if (int(safe) == 1) != (alive == horizon):
                        raise BombPolError(
                            f"w{window} d{index} action {action}: safe={safe} but "
                            f"alive_frames={alive} at H={horizon}")
                bombs_used0 = sum(block["bombs_used"][action]
                                  for action in range(N_ACTIONS)
                                  if split_action(action)[1] == 0)
                bombs_used1 = sum(block["bombs_used"][action]
                                  for action in range(N_ACTIONS)
                                  if split_action(action)[1] == 1)
                out_row = {
                    "schema": SCHEMA_LABEL, "seed": seed, "round": rnd,
                    "window": int(window), "traj_index": int(traj_index),
                    "decision": int(index), "frame": int(state["frame"]),
                    "H": int(horizon), "actions": N_ACTIONS,
                    "features_sha256_k12": sha,
                    "features_sha256_k12_ref": row["features_sha256_k12_ref"],
                    "features_match_v600": bool(sha == row["features_sha256_k12_ref"]),
                    "state_equal_v600": state_equal,
                    "state_keys_compared": sorted(stored_state),
                    "state_keys_differ": state_diff,
                    "behaviour_action_v600": int(row["behavior_action"]),
                    "behaviour_mask_v600": int(row["behavior_mask"]),
                    "safe": block["safe"],
                    "alive_frames": block["alive_frames"],
                    "bombs_used": block["bombs_used"],
                    "bombs_start": block["bombs_start"],
                    "bombs_end": block["bombs_end"],
                    "bomb_decrement_frame": block["bomb_decrement_frame"],
                    "park_reply": block["park_reply"],
                    "child_replies_digest": block["child_replies_digest"],
                    "child_reply_lines": block["child_reply_lines"],
                    "children": block["children"],
                    "child_slots_used": block["child_slots_used"],
                    "max_concurrent_children": block["max_concurrent_children"],
                    "label_wall_s": block["label_wall_s"],
                    "obs": row["obs"],
                }
                write_jsonl_line(handle, out_row)
                rec["rows_written"] += 1
                window_rec["labelled_states"] += 1
                window_rec["labels_written"] += N_ACTIONS
                window_rec["state_equal"] += int(state_equal)
                window_rec["features_match"] += 1
                window_rec["safe_bomb0"] += sum(block["safe"][action]
                                                for action in range(N_ACTIONS)
                                                if split_action(action)[1] == 0)
                window_rec["safe_bomb1"] += sum(block["safe"][action]
                                                for action in range(N_ACTIONS)
                                                if split_action(action)[1] == 1)
                window_rec["bombs_used_bomb0"] += bombs_used0
                window_rec["bombs_used_bomb1"] += bombs_used1
                rec["states_matched"] += 1
                costs.append(float(block["label_wall_s"]))
                rec["state_costs_s"].append(float(block["label_wall_s"]))
                # ---- the parent takes the recorded behaviour action ----------
                mask = int(C.cand_to_mask(int(row["behavior_action"])))
                for _frame in range(DECISION_FRAMES):
                    step = parent.step(mask)
                    if step["advanced"] == 1:
                        window_rec["frames_replayed"] += 1
                    reading, _reply2 = parent.state()
                    term = cf._terminal_of(
                        reading, deaths_start=deaths0, stage0=stage0,
                        hash_hex=(step["hash"] or ""),
                        frame=int(reading["frame"]), exited=step["exited"],
                        game_over=step["game_over"])
                    state = reading
                    if term is not None:
                        window_rec["terminal"] = str(term.get("terminal"))
                        died = True
                        break
                if died and index != len(rows) - 1:
                    raise BombPolError(
                        f"w{window}: the replayed trajectory died at decision "
                        f"{index} but the recording has {len(rows)} states; the "
                        f"replay is not the recorded trajectory")
                if died:
                    break
                if (index + 1) % 50 == 0:
                    log(f"  w{window} d{index + 1}/{len(rows)}: "
                        f"mean_label={statistics.fmean(costs):.3f}s")
            window_rec["states_unreached"] = len(rows) - window_rec["labelled_states"]
            window_rec["mean_label_s"] = (round(statistics.fmean(costs), 4)
                                          if costs else None)
            window_rec["label_cost_s"] = quantiles(costs)
            window_rec["bomb0_safe_rate"] = (
                round(window_rec["safe_bomb0"] / (window_rec["labelled_states"]
                                                  * N_CAND), 6)
                if window_rec["labelled_states"] else None)
            window_rec["bomb1_safe_rate"] = (
                round(window_rec["safe_bomb1"] / (window_rec["labelled_states"]
                                                  * N_CAND), 6)
                if window_rec["labelled_states"] else None)
            rec["windows_detail"].append(window_rec)
            log(f"s{seed} r{rnd} w{window}: {window_rec['labelled_states']}/"
                f"{len(rows)} states, state_equal={window_rec['state_equal']}, "
                f"bomb0_safe={window_rec['bomb0_safe_rate']} "
                f"bomb1_safe={window_rec['bomb1_safe_rate']}")
            parent.close()
            parents.remove(parent)
        if rec["rows_written"] != total_rows:
            raise BombPolError(f"s{seed} r{rnd}: labelled {rec['rows_written']} of "
                               f"{total_rows} recorded states")
        rec["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["status"] = "failed"
        log(f"COLLECT FAILED s{seed} r{rnd}: {rec['error']}")
    finally:
        handle.close()
        for parent in parents:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        rec["wall_s"] = round(time.perf_counter() - started, 3)
        rec["states"] = int(rec["rows_written"])
        rec["labels_total"] = int(rec["rows_written"]) * N_ACTIONS
        rec["label_cost_s"] = quantiles(rec["state_costs_s"])
        del rec["state_costs_s"]
        if rec["status"] == "complete":
            if labels.exists():
                labels.unlink()
            partial.rename(labels)
            rec["labels_sha256"] = sha256_file(labels)
            rec["labels_bytes"] = labels.stat().st_size
        write_json(summary_path, rec)
    log(f"collect s{seed} r{rnd}: status={rec['status']} rows={rec['rows_written']} "
        f"mean_label={rec['label_cost_s'].get('mean')}s wall={rec['wall_s']}s")
    return 0 if rec["status"] == "complete" else 4


# --------------------------------------------------------------------------- #
# mode: merge (+ the 36-label completeness statistics)
# --------------------------------------------------------------------------- #
def label_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """36-label completeness and the bomb=0 vs bomb=1 contrast (descriptive)."""

    actions = N_ACTIONS
    bomb0 = [action for action in range(actions) if split_action(action)[1] == 0]
    bomb1 = [action for action in range(actions) if split_action(action)[1] == 1]
    safe0: list[int] = []
    safe1: list[int] = []
    alive0: list[int] = []
    alive1: list[int] = []
    states_all_safe0 = 0
    states_all_unsafe0 = 0
    states_all_safe1 = 0
    states_all_unsafe1 = 0
    bomb1_better = 0
    bomb1_equal = 0
    bomb1_worse = 0
    bombs_used_by_action = [0] * actions
    spent_bomb_counts = [0] * actions
    complete = 0
    for row in rows:
        good = (len(row["safe"]) == actions and len(row["alive_frames"]) == actions
                and all(v is not None for v in row["alive_frames"])
                and all(v in (0, 1) for v in row["safe"]))
        complete += int(good)
        if not good:
            continue
        safe = [int(v) for v in row["safe"]]
        alive = [int(v) for v in row["alive_frames"]]
        used = [int(v) for v in row.get("bombs_used", [0] * actions)]
        s0 = [safe[a] for a in bomb0]
        s1 = [safe[a] for a in bomb1]
        a0 = [alive[a] for a in bomb0]
        a1 = [alive[a] for a in bomb1]
        safe0.extend(s0)
        safe1.extend(s1)
        alive0.extend(a0)
        alive1.extend(a1)
        states_all_safe0 += int(all(s0))
        states_all_unsafe0 += int(not any(s0))
        states_all_safe1 += int(all(s1))
        states_all_unsafe1 += int(not any(s1))
        # per candidate: does bomb=1 keep you alive longer than bomb=0?
        for cand in range(N_CAND):
            if a1[cand] > a0[cand]:
                bomb1_better += 1
            elif a1[cand] == a0[cand]:
                bomb1_equal += 1
            else:
                bomb1_worse += 1
        for action in range(actions):
            bombs_used_by_action[action] += used[action]
            spent_bomb_counts[action] += int(used[action] > 0)
    states = len(rows)
    pairs = states * N_CAND
    return {
        "states": states,
        "rows_with_36_labels": complete,
        "complete_fraction": round(complete / states, 6) if states else None,
        "labels_total": complete * actions,
        "actions": actions,
        "H": sorted({int(row["H"]) for row in rows}),
        "windows": sorted({int(row["window"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "rounds": sorted({int(row["round"]) for row in rows}),
        "states_by_round": {str(r): sum(1 for row in rows if int(row["round"]) == r)
                            for r in ROUNDS},
        "states_by_window": {str(w): sum(1 for row in rows
                                         if int(row["window"]) == w)
                             for w in TRAIN_WINDOWS},
        "bomb0": {
            "actions": len(bomb0), "labels": len(safe0),
            "safe_rate": round(statistics.fmean(safe0), 6) if safe0 else None,
            "safe_labels": sum(safe0),
            "alive_frames": quantiles(alive0),
            "alive_frames_histogram": histogram(alive0, (1, 12, 24, 48, 60, 120)),
            "states_all_safe": states_all_safe0,
            "states_all_unsafe": states_all_unsafe0,
            "bomb_labels_that_spent_a_bomb": sum(
                1 for action in bomb0 if spent_bomb_counts[action]),
        },
        "bomb1": {
            "actions": len(bomb1), "labels": len(safe1),
            "safe_rate": round(statistics.fmean(safe1), 6) if safe1 else None,
            "safe_labels": sum(safe1),
            "alive_frames": quantiles(alive1),
            "alive_frames_histogram": histogram(alive1, (1, 12, 24, 48, 60, 120)),
            "states_all_safe": states_all_safe1,
            "states_all_unsafe": states_all_unsafe1,
            "bomb_labels_that_spent_a_bomb": sum(
                1 for action in bomb1 if spent_bomb_counts[action]),
            "bombs_spent_total": sum(bombs_used_by_action[action] for action in bomb1),
        },
        "contrast": {
            "unit": f"one (state, candidate) pair = {pairs} pairs",
            "bomb1_alive_gt_bomb0": bomb1_better,
            "bomb1_alive_eq_bomb0": bomb1_equal,
            "bomb1_alive_lt_bomb0": bomb1_worse,
            "safe_rate_delta_bomb1_minus_bomb0": (
                round(statistics.fmean(safe1) - statistics.fmean(safe0), 6)
                if safe0 and safe1 else None),
            "note": ("descriptive counts only; no test, no interval (prereg §2 "
                     "P1-P4 belong to the analysis pass)"),
        },
        "per_action": [
            {"action": action, "cand": split_action(action)[0],
             "bomb": split_action(action)[1],
             "safe_rate": (round(sum(1 for row in rows
                                     if int(row["safe"][action]) == 1)
                                / states, 6) if states else None),
             "labels_that_spent_a_bomb": spent_bomb_counts[action],
             "bombs_spent_total": bombs_used_by_action[action]}
            for action in range(actions)],
    }


def load_v600_index() -> dict[tuple[int, int, int, int], int]:
    """(seed, round, window, decision) -> frame, from the frozen v6.0.0 file."""

    index: dict[tuple[int, int, int, int], int] = {}
    for row in read_jsonl(V600_LABELS):
        index[(int(row["seed"]), int(row["round"]), int(row["window"]),
               int(row["decision"]))] = int(row["frame"])
    return index


def row_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (int(row["seed"]), int(row["round"]), int(row["window"]),
            int(row["decision"]))


def mode_merge(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    dest = pooled_labels_path(root)
    partial = dest.with_suffix(dest.suffix + ".partial")
    entries: list[dict[str, Any]] = []
    rows_total = 0
    missing: list[str] = []
    with partial.open("w", encoding="utf-8") as out:
        for seed in TRAIN_SEEDS:
            for rnd in ROUNDS:
                labels = labels_group_path(root, seed, rnd)
                summary = collect_summary_path(root, seed, rnd)
                if not labels.is_file() or not summary.is_file():
                    missing.append(f"s{seed}r{rnd}: {labels}")
                    continue
                rec = read_json(summary)
                if rec.get("status") != "complete":
                    missing.append(f"s{seed}r{rnd}: summary status "
                                   f"{rec.get('status')}")
                    continue
                digest = sha256_file(labels)
                if digest != rec.get("labels_sha256"):
                    raise BombPolError(f"s{seed}r{rnd}: labels sha256 {digest[:16]} "
                                       f"!= recorded {rec.get('labels_sha256','')[:16]}")
                count = 0
                for line in labels.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if len(row["safe"]) != N_ACTIONS:
                        raise BombPolError(f"s{seed}r{rnd}: a row carries "
                                           f"{len(row['safe'])} labels, not "
                                           f"{N_ACTIONS}")
                    out.write(line + "\n")
                    count += 1
                if count != int(rec["rows_written"]):
                    raise BombPolError(f"s{seed}r{rnd}: {count} rows != the "
                                       f"{rec['rows_written']} the summary recorded")
                rows_total += count
                entries.append({"seed": seed, "round": rnd, "path": str(labels),
                                "sha256": digest, "rows": count,
                                "bytes": labels.stat().st_size,
                                "summary": str(summary)})
                log(f"merge: s{seed} r{rnd} -> {count} rows")
    # the file object was closed by its `with` block; `partial` is a path
    if missing:
        raise BombPolError(f"cannot merge, incomplete groups: {missing}")
    if dest.exists():
        dest.unlink()
    partial.rename(dest)
    payload: dict[str, Any] = {
        "schema": SCHEMA_MERGE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "labels": str(dest), "sha256": sha256_file(dest),
        "bytes": dest.stat().st_size, "rows": rows_total,
        "labels_total": rows_total * N_ACTIONS,
        "entries": entries, "all_pass": False,
    }
    rows = read_jsonl(dest)
    stats = label_stats(rows)
    stats.update({"schema": SCHEMA_LABEL_STATS, "producer": PRODUCER,
                  "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "prereg": PREREG, "labels": str(dest),
                  "labels_sha256": payload["sha256"],
                  "labels_bytes": payload["bytes"],
                  "states_expected": V600_LABELS_STATES})
    payload["all_pass"] = bool(len(rows) == rows_total
                               and stats["rows_with_36_labels"] == rows_total
                               and len(rows) >= MIN_STATES)
    # -- the location check the round is built on: the merged file must carry
    #    exactly the v6.0.0 states, key by key, with the same 106-dim features
    v600 = load_v600_index()
    merged_keys = {row_key(row) for row in rows}
    payload["state_locator"] = {
        "v600_states": len(v600),
        "merged_states": len(merged_keys),
        "keys_missing_in_merged": len(set(v600) - merged_keys),
        "keys_not_in_v600": len(merged_keys - set(v600)),
        "keys_identical": bool(set(v600) == merged_keys),
        "features_sha256_equal_to_v600": sum(
            1 for row in rows
            if row["features_sha256_k12"] == row["features_sha256_k12_ref"]),
        "state_keys_10_equal_to_v600": sum(
            1 for row in rows if row.get("state_equal_v600")),
        "frames_equal_to_v600": sum(
            1 for row in rows if int(row["frame"]) == v600.get(row_key(row), -1)),
        "unit": "(seed, round, window, decision, frame)",
    }
    payload["all_pass"] = bool(payload["all_pass"]
                               and payload["state_locator"]["keys_identical"]
                               and payload["state_locator"][
                                   "features_sha256_equal_to_v600"] == rows_total)
    payload["label_stats"] = stats
    write_json(pooled_stats_path(root), stats)
    write_json(pathlib.Path(args.out) if args.out else root / "merge.json", payload)
    log(f"merge: {rows_total} states x {N_ACTIONS} labels -> {dest} "
        f"bomb0_safe={stats['bomb0']['safe_rate']} "
        f"bomb1_safe={stats['bomb1']['safe_rate']} all_pass={payload['all_pass']}")
    if len(rows) < MIN_STATES:
        log(f"STOP (S-label): {len(rows)} states < {MIN_STATES}")
        return 4
    return 0 if payload["all_pass"] else 2


# --------------------------------------------------------------------------- #
# mode: train
# --------------------------------------------------------------------------- #
def load_pooled_rows(root: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pooled = pooled_labels_path(root)
    if not pooled.is_file():
        raise BombPolError(f"no pooled labels: {pooled}")
    rows = read_jsonl(pooled)
    stats_rec = read_json(pooled_stats_path(root)) if pooled_stats_path(root).is_file() else {}
    info = {"path": str(pooled), "sha256": sha256_file(pooled),
            "bytes": pooled.stat().st_size, "rows": len(rows),
            "recorded_sha256": stats_rec.get("labels_sha256"),
            "sha256_match": sha256_file(pooled) == stats_rec.get("labels_sha256")}
    if not info["sha256_match"]:
        raise BombPolError(f"{pooled} sha256 {info['sha256'][:16]} != the recorded "
                           f"{str(info['recorded_sha256'])[:16]}")
    return rows, info


def build_arrays(rows: Sequence[dict[str, Any]]) -> tuple[Any, Any, dict[str, Any]]:
    """106-dim feature matrix and the 36-column label matrix, fail closed."""

    import numpy as np

    feats = np.zeros((len(rows), FEATURE_DIM), dtype=np.float32)
    labels = np.zeros((len(rows), N_ACTIONS), dtype=np.float32)
    bad_hash: list[int] = []
    bad_label: list[int] = []
    for index, row in enumerate(rows):
        obs = row["obs"]
        # the row stores the 64 nearest live bullets (v6.0.0's compact
        # observation), sorted by distance; `build_features` re-sorts the same
        # way and keeps the 12 nearest, so this reproduces v6.0.0's vector.
        hazards = v450().rebuild_hazards(obs)
        vector, _detail = arm_a().build_features(
            state=obs["state"], hazards=hazards,
            decision_index=int(row["decision"]))
        feats[index] = vector
        if features_sha256(vector) != row["features_sha256_k12"]:
            bad_hash.append(index)
        safe = row["safe"]
        if (len(safe) != N_ACTIONS
                or any(int(v) not in (0, 1) for v in safe)
                or any((int(s) == 1) != (int(a) == int(row["H"]))
                       for s, a in zip(safe, row["alive_frames"]))):
            bad_label.append(index)
        else:
            labels[index] = np.asarray(safe, dtype=np.float32)
    if bad_hash:
        raise BombPolError(f"{len(bad_hash)} of {len(rows)} rows do not reproduce "
                           f"their recorded features_sha256_k12 (first "
                           f"{bad_hash[0]}); refusing to train")
    if bad_label:
        raise BombPolError(f"{len(bad_label)} of {len(rows)} rows do not carry a "
                           f"valid 36-label vector (first {bad_label[0]})")
    info = {"rows": len(rows), "feature_dim": int(FEATURE_DIM),
            "actions": int(N_ACTIONS),
            "feature_sha256_reproduced": len(rows) - len(bad_hash),
            "label_vectors_valid": len(rows) - len(bad_label),
            "positive_rate": round(float(labels.mean()), 6),
            "positive_rate_bomb0": round(float(
                labels[:, [a for a in range(N_ACTIONS)
                          if split_action(a)[1] == 0]].mean()), 6),
            "positive_rate_bomb1": round(float(
                labels[:, [a for a in range(N_ACTIONS)
                          if split_action(a)[1] == 1]].mean()), 6)}
    return feats, labels, info


def mode_train(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    if seed not in TRAIN_SEEDS:
        raise BombPolError(f"--seed must be one of {TRAIN_SEEDS}, got {seed}")
    rows, info = load_pooled_rows(root)
    if not rows:
        raise BombPolError("no labelled rows")
    if len(rows) < MIN_STATES:
        raise BombPolError(f"S-label: {len(rows)} states < {MIN_STATES}")
    feats, labels, array_info = build_arrays(rows)
    log(f"train seed={seed}: {len(rows)} states x {N_ACTIONS} actions, "
        f"dim={FEATURE_DIM}; every features_sha256 reproduced")

    x = torch.from_numpy(feats)
    y = torch.from_numpy(labels)
    policy = policy36(seed=seed)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()
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
        mean_prob_bomb0 = float(probs[:, [a for a in range(N_ACTIONS)
                                         if split_action(a)[1] == 0]].mean())
        mean_prob_bomb1 = float(probs[:, [a for a in range(N_ACTIONS)
                                         if split_action(a)[1] == 1]].mean())

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else ckpt_path(root, seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), str(ckpt))
    report_path = (pathlib.Path(args.out) if args.out
                   else train_report_path(root, seed))
    report: dict[str, Any] = {
        "schema": SCHEMA_TRAIN, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "seed": seed, "states": len(rows), "labels": int(len(rows) * N_ACTIONS),
        "feature_dim": int(FEATURE_DIM), "actions": int(N_ACTIONS),
        "model": {"class": "v431_arm_a.Policy with the output layer widened to 36",
                  "shape": f"Linear({FEATURE_DIM},{HIDDEN}) -> Tanh -> "
                           f"Linear({HIDDEN},36)",
                  "params": int(sum(p.numel() for p in policy.parameters()))},
        "labels_file": info, "features": array_info,
        "optimizer": {"name": "Adam", "lr": LR, "batch": batch, "epochs": EPOCHS,
                      "early_stopping": None,
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
        f"acc {overall_acc:.4f}) wall={wall}s -> {ckpt} "
        f"sha256={report['ckpt']['sha256'][:16]}…")
    return 0


# --------------------------------------------------------------------------- #
# mode: eval (one budget x one seed x 4 dev windows)
# --------------------------------------------------------------------------- #
def load_policy36(ckpt: pathlib.Path, *, seed: int) -> Any:
    import torch

    path = pathlib.Path(ckpt)
    if not path.is_file():
        raise BombPolError(f"policy checkpoint not found: {path}")
    policy = policy36(seed=seed)
    policy.load_state_dict(torch.load(str(path), map_location="cpu",
                                      weights_only=True))
    policy.eval()
    return policy


def sample_action36(policy: Any, feats: Any, kind: str,
                    generator: Any = None) -> int:
    import torch

    with torch.no_grad():
        logits = policy(torch.from_numpy(feats).unsqueeze(0))[0]
        if kind == "greedy":
            return int(torch.argmax(logits).item())
        if kind != "stochastic":
            raise BombPolError(f"unknown action kind {kind!r}")
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        if generator is None:
            return int(torch.multinomial(probs, 1).item())
        return int(torch.multinomial(probs, 1, generator=generator).item())


def mode_eval(args: argparse.Namespace) -> int:
    import torch

    from touhou_rl.v420 import contracts as C

    root = pathlib.Path(args.work_dir)
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    if seed not in TRAIN_SEEDS:
        raise BombPolError(f"--policy-seed must be one of {TRAIN_SEEDS}, got {seed}")
    if budget not in BUDGETS:
        raise BombPolError(f"prereg §1 fixes --decisions to {BUDGETS}, got {budget}")
    if int(args.decision_frames) != DECISION_FRAMES:
        raise BombPolError(f"prereg §1 keeps {DECISION_FRAMES} frames per decision")
    if int(args.episodes) < 1:
        raise BombPolError(f"--episodes must be >= 1, got {args.episodes}")
    windows = parse_windows(args.windows)
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    tag = args.tag or f"v830-eval-d{budget}-s{seed}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    out_path = (pathlib.Path(args.out) if args.out
                else eval_path(root, budget, seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")
    ckpt = pathlib.Path(args.ckpt) if args.ckpt else ckpt_path(root, seed)
    digest = sha256_file(ckpt) if ckpt.is_file() else None
    policy = load_policy36(ckpt, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    arm = arm_a()

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        feats, _detail = arm.build_features(state=state, hazards=hazards,
                                            decision_index=decision)
        action = sample_action36(policy, feats, args.kind, generator)
        return split_action(action)

    summary: dict[str, Any] = {
        "schema": SCHEMA_EVAL_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "mode": "eval", "policy_seed": seed, "windows": list(windows),
        "episodes_per_window": int(args.episodes),
        "budget_decisions": budget, "decision_frames": DECISION_FRAMES,
        "temperature": TEMPERATURE, "kind": args.kind,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest, "bytes":
                 ckpt.stat().st_size if ckpt.is_file() else None},
        "sampler": {"kind": "torch.multinomial on softmax(z/1) over 36 actions",
                    "generator_seed": generator_seed(seed),
                    "generator": "one torch.Generator per (budget, seed) process, "
                                 "windows in the frozen order 11,12,17,18"},
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
                f"{data_info['cfg']['muted_four_zero']} "
                f"sha256={data_info['cfg']['sha256'][:16]}…")
            parent, gate = start_parent(slot=slot, window=window,
                                        engine=args.engine, data_dir=data_dir,
                                        log_dir=run_dir)
            parents.append(parent)
            rows: list[dict[str, Any]] = []
            for episode in range(int(args.episodes)):
                record = run_episode36(
                    parent, episode=episode, policy_seed=seed, decisions=budget,
                    decide=decide, budget=budget,
                    decision_frames=DECISION_FRAMES)
                record.update({"ckpt_sha256": digest,
                               "generator_seed": generator_seed(seed)})
                record.pop("terminal_reply", None)
                record.pop("trace", None)
                record.pop("spawn_reply", None)
                write_jsonl_line(handle, record)
                rows.append(record)
                if record.get("invalid"):
                    raise BombPolError(f"d{budget} seed {seed} window {window} "
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
                "first_bomb_decisions": sorted(
                    row["first_bomb_decision"] for row in rows
                    if row["first_bomb_decision"] is not None),
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
# mode: equiv (run_episode36 == v800_bomb.run_episode_bomb with bomb off)
# --------------------------------------------------------------------------- #
EQUIV_FIELDS = ("frames", "alive_frames", "first_death_offset", "first_death_frame",
                "return_decisions", "decisions_executed", "survived", "terminal",
                "bombs_start", "bombs_end", "bombs_used")


def mode_equiv(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    seed = int(args.policy_seed)
    windows = parse_windows(args.windows)
    episodes = int(args.episodes)
    budget = int(args.decisions)
    tag = args.tag or f"v830-equiv-s{seed}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    out_path = pathlib.Path(args.out) if args.out else root / "equiv_check.json"
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    ref_module = m800()

    def next_cand(window: int, episode: int, index: int) -> int:
        """A fixed, model-free action sequence both loops are given.

        Independent of any checkpoint on purpose: the equivalence control must be
        runnable before a single training run exists, and it must not depend on
        the thing it is meant to check.
        """

        return int((int(window) * 7 + int(episode) * 3 + int(index) * 5) % N_CAND)

    payload: dict[str, Any] = {
        "schema": SCHEMA_EQUIV, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG + " (no prereg claim: this is the copy's proof)",
        "requirement": ("`run_episode36` with the bomb bit forced to 0 must be the "
                        "v8.0.0 loop on the engine: the same action sequence must "
                        "give the same episode fields through both loops"),
        "reference": "tools/rl/v800_bomb.py::run_episode_bomb (--bomb-rule none)",
        "action_sequence": "cand = (window*7 + episode*3 + index*5) % 18 (model-free)",
        "engine": engine, "launch_env": launch_env,
        "policy_seed": seed, "windows": list(windows), "episodes": episodes,
        "budget_decisions": budget,
        "fields": list(EQUIV_FIELDS), "episodes_compared": [], "all_pass": False,
        "error": None, "wall_s": None,
    }
    parents: list[Any] = []
    started = time.perf_counter()
    try:
        for slot, window in enumerate(windows):
            data_dir, _info = v450().prepare_data_dir(data_root, window,
                                                      args.data_source)
            parent, gate = start_parent(slot=slot, window=window,
                                        engine=args.engine, data_dir=data_dir,
                                        log_dir=run_dir)
            parents.append(parent)
            if not gate["all_pass"]:
                raise BombPolError(f"w{window}: fidelity gate failed")
            for episode in range(episodes):
                mine = run_episode36(
                    parent, episode=episode, policy_seed=seed, decisions=budget,
                    decide=lambda state, hazards, decision: (
                        next_cand(window, episode, decision), 0),
                    budget=budget)
                seq_mine = list(mine.get("actions_sampled") or [])
                ref = ref_module.run_episode_bomb(
                    parent, mode="equiv", arm="A0", bomb_rule="none",
                    episode=episode, policy_seed=seed, decisions=budget,
                    decide=lambda state, hazards, decision: next_cand(
                        window, episode, decision),
                    keep_trace=True)
                seq_ref = [int(row["cand"]) for row in (ref.get("trace") or [])]
                ref.pop("trace", None)
                seq_mine_cands = [split_action(a)[0] for a in seq_mine]
                diffs = {field: {"mine": mine.get(field), "reference": ref.get(field)}
                         for field in EQUIV_FIELDS
                         if mine.get(field) != ref.get(field)}
                entry = {"window": int(window), "episode": int(episode),
                         "actions_compared": len(seq_mine_cands),
                         "action_sequence_identical": seq_mine_cands == seq_ref,
                         "field_differences": diffs,
                         "identical": bool(not diffs
                                           and seq_mine_cands == seq_ref)}
                payload["episodes_compared"].append(entry)
                log(f"  equiv w{window} ep{episode}: "
                    f"actions={len(seq_mine_cands)} "
                    f"identical={entry['identical']}")
        payload["all_pass"] = bool(payload["episodes_compared"]
                                   and all(e["identical"] for e
                                           in payload["episodes_compared"]))
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
    log(f"equiv: all_pass={payload['all_pass']} wall={payload['wall_s']}s -> "
        f"{out_path}")
    return 0 if payload["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: smoke (a few episodes before a whole budget's grid)
# --------------------------------------------------------------------------- #
def mode_smoke(args: argparse.Namespace) -> int:
    import torch

    root = pathlib.Path(args.work_dir)
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    windows = parse_windows(args.windows)
    tag = args.tag or f"v830-smoke-s{seed}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    out_path = pathlib.Path(args.out) if args.out else root / "smoke.json"
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    ckpt = pathlib.Path(args.ckpt) if args.ckpt else ckpt_path(root, seed)
    policy = load_policy36(ckpt, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    arm = arm_a()

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        feats, _detail = arm.build_features(state=state, hazards=hazards,
                                            decision_index=decision)
        return split_action(sample_action36(policy, feats, args.kind, generator))

    payload: dict[str, Any] = {
        "schema": SCHEMA_SMOKE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("before a whole grid: does the policy's bomb bit reach the "
                        "engine and does the counter drop?"),
        "engine": engine, "launch_env": launch_env, "policy_seed": seed,
        "budget_decisions": budget, "windows": list(windows),
        "episodes": int(args.episodes),
        "ckpt": {"path": str(ckpt),
                 "sha256": sha256_file(ckpt) if ckpt.is_file() else None},
        "rows": [], "checks": {}, "all_pass": False, "error": None, "wall_s": None,
    }
    parents: list[Any] = []
    started = time.perf_counter()
    try:
        for slot, window in enumerate(windows):
            data_dir, _info = v450().prepare_data_dir(data_root, window,
                                                      args.data_source)
            parent, gate = start_parent(slot=slot, window=window,
                                        engine=args.engine, data_dir=data_dir,
                                        log_dir=run_dir)
            parents.append(parent)
            for episode in range(int(args.episodes)):
                record = run_episode36(parent, episode=episode, policy_seed=seed,
                                       decisions=budget, decide=decide,
                                       budget=budget, keep_trace=True)
                record["window"] = int(window)
                payload["rows"].append(
                    {key: record[key] for key in (
                        "window", "episode", "frames", "alive_frames", "survived",
                        "decisions_executed", "bomb_emits", "bomb_decisions",
                        "bombs_start", "bombs_end", "bombs_used",
                        "bombs_remaining_at_emit", "bombs_ignored_emits",
                        "first_bomb_decision", "invalid", "error")})
                payload["rows"][-1]["trace"] = record["trace"]
        rows = payload["rows"]
        emits = sum(int(row["bomb_emits"]) for row in rows)
        used = sum(int(row["bombs_used"] or 0) for row in rows)
        payload["checks"] = {
            "bomb_path_fired": bool(emits > 0),
            "bombs_remaining_decreased": bool(used > 0),
            "emits_at_least_used": bool(emits >= used),
            "counter_bookkeeping_closes": bool(used == sum(
                int(row["bombs_start"]) - int(row["bombs_end"]) for row in rows)),
            "all_episodes_valid": bool(all(not row["invalid"] for row in rows)),
            "emits": emits, "bombs_spent": used,
            "episodes": len(rows),
        }
        payload["all_pass"] = bool(all(bool(v) for k, v in payload["checks"].items()
                                       if isinstance(v, bool)))
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for parent in parents:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        payload["wall_s"] = round(time.perf_counter() - started, 3)
    write_json(out_path, payload)
    log(f"smoke: checks={payload['checks']} all_pass={payload['all_pass']} -> "
        f"{out_path}")
    return 0 if payload["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: diagnostics (offline, descriptive only)
# --------------------------------------------------------------------------- #
def _episode_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return read_jsonl(path)


def mode_diagnostics(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out = pathlib.Path(args.out) if args.out else root / "diagnostics.json"
    usage_out = (pathlib.Path(args.usage_out) if args.usage_out
                 else root / "bomb_usage.json")
    payload: dict[str, Any] = {
        "schema": SCHEMA_DIAG, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "no_statistics": ("counts, rates, means, medians, quantiles and "
                          "distributions only; no test, no interval, no p-value "
                          "(prereg §5)"),
        "budgets": {}, "per_cell": {}, "references": {
            "nobomb": BASELINE_NOBOMB, "rule_80px": RULE_80PX},
        "budget_comparison_warning": ("prereg §5: the three budgets must not be "
                                      "compared by their means as if they were "
                                      "the same measurement"),
        "missing": [], "all_pass": True,
    }
    usage: dict[str, Any] = {
        "schema": SCHEMA_BOMB_USAGE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
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
                "bombs_used_total": sum(int(row["bombs_used"] or 0)
                                        for row in seed_rows),
                "path": str(path), "sha256": sha256_file(path),
            }
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
                                            for row in cell),
                    "episodes_exhausted": sum(1 for row in cell
                                              if row["bombs_end"] == 0),
                }
        if not rows:
            continue
        alive = [float(row["alive_frames"]) for row in rows]
        survived = sum(1 for row in rows if row["survived"])
        bomb_episodes = [row for row in rows if int(row["bomb_emits"] or 0) > 0]
        bombs_used = [int(row["bombs_used"] or 0) for row in rows]
        first_bomb = [int(row["first_bomb_decision"]) for row in rows
                      if row["first_bomb_decision"] is not None]
        exhausted = sum(1 for row in rows if int(row["bombs_end"] or 0) == 0)
        usage_hist: dict[str, int] = {}
        for value in bombs_used:
            usage_hist[str(int(value))] = usage_hist.get(str(int(value)), 0) + 1
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
            "bomb": {
                "episodes_that_pressed_bomb": len(bomb_episodes),
                "episodes_that_spent_a_bomb": sum(1 for value in bombs_used
                                                  if value > 0),
                "bomb_emits_total": sum(int(row["bomb_emits"] or 0)
                                        for row in rows),
                "bombs_spent_total": sum(bombs_used),
                "bombs_ignored_emits_total": sum(
                    int(row.get("bombs_ignored_emits") or 0) for row in rows),
                "bombs_used_per_episode": quantiles(bombs_used),
                "bombs_used_histogram": usage_hist,
                "episodes_exhausted": exhausted,
                "exhaustion_fraction": round(exhausted / len(rows), 6),
                "first_bomb_decision": quantiles(first_bomb),
                "first_bomb_decision_histogram": first_bomb_histogram(first_bomb),
                "episodes_never_bombed": len(rows) - len(bomb_episodes),
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
                f"exhausted={rec['bomb']['exhaustion_fraction']:.4%}")
    log(f"diagnostics -> {out} (+ {usage_out})")
    return 0 if payload["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: train-summary
# --------------------------------------------------------------------------- #
def mode_train_summary(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out: dict[str, Any] = {
        "schema": SCHEMA_TRAIN_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "design": ("BCE, Adam 1e-3, batch 256, 200 epochs, 3 seeds; one "
                   "checkpoint per seed trained on all 5,166 states (the same "
                   "state set v4.5.0's round 2 trained on)"),
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
            "loss_first_epoch": rec["loss_first_epoch"],
            "loss_last_epoch": rec["loss_last_epoch"],
            "final_train_bce": rec["final_train_bce"],
            "train_accuracy": rec["train_accuracy"],
            "mean_predicted_prob_bomb0": rec.get("mean_predicted_prob_bomb0"),
            "mean_predicted_prob_bomb1": rec.get("mean_predicted_prob_bomb1"),
            "ckpt": rec["ckpt"], "wall_s": rec["wall_s"],
        }
    write_json(pathlib.Path(args.out) if args.out else root / "train_summary.json",
               out)
    log(f"train-summary: {len(out['seeds'])}/{len(TRAIN_SEEDS)} seeds -> "
        f"{args.out or (root / 'train_summary.json')}")
    return 0 if out["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: audio-audit / process-audit / manifest
# --------------------------------------------------------------------------- #
def mode_audio_audit(args: argparse.Namespace) -> int:
    """``v450_cf_label.mode_audit`` on this round's ``--work-dir``, own ``--out``."""

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
        raise BombPolError(f"not a directory: {root}")
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


# --------------------------------------------------------------------------- #
# mode: publish
# --------------------------------------------------------------------------- #
def mode_publish(args: argparse.Namespace) -> int:
    import shutil

    root = pathlib.Path(args.work_dir)
    dest = pathlib.Path(args.dir) if args.dir else PUBLISH
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    wanted = [
        (pooled_labels_path(root), "labels_bomb36.jsonl"),
        (pooled_stats_path(root), "labels_bomb36_stats.json"),
        (root / "merge.json", "merge.json"),
        (root / "preflight.json", "preflight.json"),
        (root / "labeller_selfcheck.json", "labeller_selfcheck.json"),
        (root / "train_summary.json", "train_summary.json"),
        (root / "diagnostics.json", "diagnostics.json"),
        (root / "bomb_usage.json", "bomb_usage.json"),
        (root / "ckpt_assertions.json", "ckpt_assertions.json"),
        (root / "audio_audit.json", "audio_audit.json"),
        (root / "process_residue.json", "process_residue.json"),
        (root / "equiv_check.json", "equiv_check.json"),
        (root / "smoke.json", "smoke.json"),
        (root / "labels_vs_v600.json", "labels_vs_v600.json"),
        (root / "contention.tsv", "contention.tsv"),
        (root / "commands.tsv", "commands.tsv"),
        (root / "fidelity_d3.json", "fidelity_d3.json"),
        (root / "ckpt_assertions_final.json", "ckpt_assertions_final.json"),
    ]
    # the harness and the read-only helpers, so the round is reviewable
    for name in ("exec.sh", "chain.sh", "retry_group.sh", "stop_collect.sh",
                 "monitor.sh", "status.sh", "progress.sh", "grid_status.sh",
                 "show.py", "tables.py", "build_run.py", "compare_v600.py",
                 "draft_check.py", "RUN.md.tmpl"):
        wanted.append((root / name, f"scripts/{name}"))
    for seed in TRAIN_SEEDS:
        wanted.append((train_report_path(root, seed), f"train_seed{seed}.json"))
    for budget in BUDGETS:
        for seed in TRAIN_SEEDS:
            path = eval_path(root, budget, seed)
            wanted.append((path, f"eval_d{budget}_seed{seed}.jsonl"))
            wanted.append((path.with_suffix(".summary.json"),
                           f"eval_d{budget}_seed{seed}.summary.json"))
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
    for sub in ("logs", "runs"):
        source_dir = root / sub
        if not source_dir.is_dir():
            continue
        target_dir = dest / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(source_dir)
            target = target_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append({"source": str(path), "target": str(target),
                           "sha256": sha256_file(target),
                           "bytes": target.stat().st_size})
    payload = {"schema": SCHEMA_PUBLISH, "producer": PRODUCER,
               "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "prereg": PREREG, "dest": str(dest), "files": copied,
               "file_count": len(copied), "v830_bombpy_sha256":
               sha256_file(pathlib.Path(__file__).resolve())}
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
        description="v8.3.0 BOMBPOL-001: a 36-action policy that decides when to "
                    "bomb, trained on counterfactual labels")
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "assert-ckpt", "selfcheck",
                                 "collect", "merge", "train", "train-summary",
                                 "eval", "equiv", "smoke", "diagnostics",
                                 "audio-audit", "process-audit", "manifest",
                                 "publish"))
    parser.add_argument("--seed", type=int, default=0,
                        help="training seed (0/1/2)")
    parser.add_argument("--round", type=int, default=0, choices=ROUNDS,
                        help="collect: which DAgger round's states to re-label")
    parser.add_argument("--policy-seed", type=int, default=None,
                        help="eval/equiv/smoke: alias of --seed")
    parser.add_argument("--windows", default="13,14,15,16")
    parser.add_argument("--decisions", type=int, default=DECISIONS,
                        help=f"eval budget: one of {BUDGETS}")
    parser.add_argument("--decision-frames", type=int, default=DECISION_FRAMES)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--horizon", type=int, default=H_DEFAULT)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--kind", default="stochastic",
                        choices=("stochastic", "greedy"))
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--work-dir", default=str(WORK))
    parser.add_argument("--labels-out", default="")
    parser.add_argument("--summary-out", default="")
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

    engine_modes = ("selfcheck", "collect", "eval", "equiv", "smoke")
    if args.mode in engine_modes:
        prepare_env()
        tag = args.tag or f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
        args.tag = tag
        log_dir = pathlib.Path(args.work_dir) / "runs" / tag
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_FH = (log_dir / "run.log").open("a", encoding="utf-8")
        v450()._LOG_FH = _LOG_FH
        m432()._LOG_FH = _LOG_FH

    try:
        if args.mode == "preflight":
            return mode_preflight(args)
        if args.mode == "assert-ckpt":
            return mode_assert(args)
        if args.mode == "selfcheck":
            return mode_selfcheck(args)
        if args.mode == "collect":
            return mode_collect(args)
        if args.mode == "merge":
            return mode_merge(args)
        if args.mode == "train":
            return mode_train(args)
        if args.mode == "train-summary":
            return mode_train_summary(args)
        if args.mode == "eval":
            return mode_eval(args)
        if args.mode == "equiv":
            return mode_equiv(args)
        if args.mode == "smoke":
            return mode_smoke(args)
        if args.mode == "diagnostics":
            return mode_diagnostics(args)
        if args.mode == "audio-audit":
            return mode_audio_audit(args)
        if args.mode == "process-audit":
            return mode_process_audit(args)
        if args.mode == "manifest":
            return mode_manifest(args)
        return mode_publish(args)
    except BombPolError as exc:
        log(f"FAILED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
