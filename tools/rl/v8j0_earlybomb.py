#!/usr/bin/env python3
"""v8.18.0 EARLYBOMB-001 - sample **60-120 frames before the death**, so that the
``bomb = 1`` branch is an **ordinary bomb (1 spent)** and not a death bomb
(2 spent).

Why this round exists
---------------------
v8.17.0 (CONSERVE-001, ``--bomb-rule none``) finally produced armed labels
(``bombs_remaining = 3`` at every state) and its labels *did* express a bomb's
value (best ``bomb = 1`` > best ``bomb = 0``), yet the evaluated policy was worse
than v8.6.0 (20 s: 14.1 % vs 29.58 %).  The diagnosis in prereg v8.18.0 §0 is a
**timing** one, not a labelling one:

    v8.17.0 sampled ``[death_decision - 19 .. death_decision]`` (the last 20
    decisions = the last 40 frames).  All 600 states were ``player_state =
    DYING`` - inside the death animation - and the engine charged **two** bombs
    for a press there (3 -> 1: an ordinary bomb *plus* the deathbomb).

So the label taught "use a bomb 0.5 s before you die", the most expensive form of
the same resource, and the policy learned to burn all three bombs for one life.

This round moves **one** factor - the sampling distance - and freezes everything
else:

================================  ===============================================
v8.17.0                           **v8.18.0**
================================  ===============================================
``S`` + ``--bomb-rule none``      unchanged (``bomb = 0`` at every decision)
900 decisions, death-stop         unchanged (no-death episode discarded; a cell is
                                  skipped after 10 consecutive no-death episodes)
last 20 decisions (40 frames)     **``decisions_before_death`` in [30, 60]**
                                  (= 60-120 frames), one row of states per death
36-way labels x H = 120/240/600   unchanged (one 36-child run to H = 600;
                                  ``alive(H) = min(alive(600), H)``)
600 states                        unchanged (12 cells x 50)
================================  ===============================================

The sampling ladder (prereg §2.1 ``S-dead``)
---------------------------------------------
A death is detected when ``deaths`` increments, which happens **after** the death
animation, so ``decisions_before_death`` and "the player is still alive" are not
the same thing.  The window is therefore a *ladder*:

    early   decisions_before_death in [ 30,  60]   ( 60-120 frames before death)
    late    decisions_before_death in [ 60, 120]   (120-240 frames before death)
    deep    decisions_before_death in [120, 240]   (240-480 frames before death)

``--window auto`` measures, **inside the engine run**, the ``player_state`` of
every decision of the first death trajectory (pass 1 records it) and picks the
first rung whose ``ALIVE`` share reaches the P0a gate (>= 90 %), else the rung
with the highest share.  The choice and the full by-``decisions_before_death``
``player_state`` table are written to the cell summary, so the death-detection
lag is **measured** here rather than assumed.

The degenerate-window guard (``--min-death-decision``, default 200)
-------------------------------------------------------------------
The window is relative to *the* death, so a **late** death is not required by
the formula - but a death at decision 45 turns ``[30, 60]`` into decisions
0..15, i.e. the first 0.5 s after the frozen prefix.  There ``bomb = 0``
survives the H = 120 horizon by construction (the death is 60-90 decisions
away), so ``diff`` is a **structural** 0 and the states say nothing about the
question this round asks.  The 2026-11-22 smoke measured exactly that case
(``DIED at decision 45 -> window 0..15``), so an episode whose death comes
before ``--min-death-decision`` is **discarded** (``early_death_episodes``) and
the cell keeps searching; ``min_death_decision`` is written into every cell
summary and every state row, so the pool's composition stays auditable.  The
work order's relaxation to 120 (when late deaths are too rare) is a CLI switch,
not a silent redefinition: ``MIN_DEATH_DECISION_FALLBACK = 120``.

What is measured (prereg §2 P0a-P0d)
------------------------------------
* **P0a** ``ALIVE`` share of the sampled states (gate **>= 90 %**);
* **P0b** share with ``bombs_remaining > 0`` (gate **>= 50 %**; with
  ``--bomb-rule none`` this is 100 % by construction);
* **P0c** of the ``bomb = 1`` branches the engine **honoured** (the counter
  decremented), the share whose ``spent_amount`` is exactly **1** - the
  discriminating gate against v8.17.0 (gate **>= 90 %**);
* **P0d** share of states with ``best bomb1 > best bomb0`` at each horizon
  (gate **>= 20 %**).

Verdicts (prereg §2.1): ``S-dead`` (P0a fails -> move the window earlier),
``S-2bomb`` (P0c fails -> still two bombs), ``S-nodiff`` (P0a holds, P0d < 20 %
at every horizon -> publish statistics, do not train), ``train`` otherwise.

Two passes, and why
-------------------
The distance to the death is only known once the death has happened:

    pass 1  the behaviour trajectory (HAZARDS + PARK + 2 frames per decision),
            no labels: records every decision's ``player_state``,
            ``bombs_remaining``, ``nearest_all_px`` and executed action, and
            stops at the death;
    sample  the frozen rule: ``decisions_before_death`` in the chosen window;
    pass 2  the warm RESET, then the recorded actions are replayed **up to the
            last sampled decision** and only the sampled decisions are parked and
            labelled (36 children, H = 600, truncated to H = 120 / 240).

Replay is verified, never assumed: at **every** decision of pass 2 the live
``frame`` and ``nearest_all_px`` must equal pass 1's values, and any difference
aborts the cell (rc 4).  An episode that survives all 900 decisions is discarded
(prereg §1) and cannot contribute a state.

Modes
-----
==================================  =============================================
``--mode preflight``                 pinned imports (sha256 of every file this
                                     round imports **and** of the frozen prereg),
                                     feature dim, action table, frozen design
``--mode assert-ckpt --phase P``     the three frozen v4.5.0 ``round2.pt``
                                     (``4abee069`` / ``a93746bb`` / ``c69960fc``),
                                     the engine sha256 and the protected sources
``--mode fidelity``                  the **D3 gate** (5 assertions): delegated to
                                     ``v432_fork_arm.py --mode fidelity`` as a
                                     subprocess, never re-implemented
``--mode smoke``                     **one** death episode: the by-decision
                                     ``player_state`` table, the auto-chosen
                                     window, its states labelled 36 ways, and
                                     P0a-P0d printed --- the round's own gate
``--mode collect --seed S --windows W [--quota N --window R]``
                                     one (seed, window) death-trajectory cell
``--mode stats``                     pool the cells, compute P0a-P0d per horizon,
                                     write ``earlybomb_stats.json`` (rc 4 if the
                                     verdict is not ``train``)
``--mode train --arm {A,B} --seed S`` Policy(106,64,36), Adam 1e-3, batch 256,
                                     200 epochs, no early stopping, dev never read
``--mode train-summary``             the six training reports side by side
``--mode eval --arm {A,B} --decisions {300,600,900} --policy-seed S``
                                     the frozen v4.5.0 episode loop with a 36-way
                                     policy (``v830_bombpol.run_episode36``)
``--mode diagnostics``               offline per-arm / per-budget counts, rates and
                                     the bomb-usage table (P3).  **No test.**
``--mode audio-audit``               AGENTS.md §9.3 on this round's own data dirs
``--mode process-audit``             this round's own engine processes must be 0
``--mode manifest --dir D``          sha256 + bytes of every file in D
``--mode publish``                   copy the raw artifacts into
                                     ``Document/Update/Alpha/v8.18.0/evidence/
                                     EARLYBOMB/`` and write the manifest

Arm A's supervision is the label's own best-bomb decision
--------------------------------------------------------
``aux = 1 iff best_bomb1 > best_bomb0`` on the row's own horizon (the same
``diff`` P0d counts), computed from ``alive_frames`` alone - no rule, no
distance.  Arm A = ``BCE(36) + 0.5 * BCE_bomb(that bit)``; arm B = ``BCE(36)``.

Reused, never re-implemented (imported by path, sha256 pinned; a moved file is
exit 2 and nothing runs)
----------------------------------------------------------------------------
* ``tools/rl/v450_cf_label.py`` - ``prepare_data_dir`` (a freshly generated muted
  ``th08.cfg`` per engine start, AGENTS.md §9.3), ``masks_for``,
  ``compact_observation``, ``Digest``, ``mode_audit``, ``check_engine``,
  ``prepare_env``, ``start_parent``, ``rebuild_hazards``, ``policy_action``;
* ``tools/rl/v432_fork_arm.py`` - ``ForkParent`` (warm RESET, prefix walk,
  SPAWN/CHILD/QUIT), ``_step_fields``, ``fields``, ``--mode fidelity``;
* ``tools/rl/v431_arm_a.py`` - ``build_features`` (the frozen 106-dim vector);
* ``tools/rl/v800_bomb.py`` - ``nearest_bullet_distances`` (the live hazard
  measurement v8.16.0/v8.17.0 used, recorded per state);
* ``tools/rl/v830_bombpol.py`` - ``label_state36`` (the 36-way labeller),
  ``action_masks`` / ``split_action`` / ``action_of``, ``policy36``,
  ``run_episode36`` (the frozen evaluation loop), ``build_arrays``,
  ``sample_action36``;
* ``tools/rl/v860_bombdag.py`` - ``load_behaviour_policy``, ``behaviour_ckpt``,
  ``decision_bucket``;
* ``touhou_rl/v400/gym/protocol.parse_hazards_reply``, ``touhou_rl/v400/labels/
  state.player_state_name``, ``touhou_rl/v420/counterfactual._terminal_of``,
  ``touhou_rl/v420/contracts.cand_to_mask`` / ``assert_seed_allowed``.

Failure policy (fail closed where the data would be wrong, record where the
engine is merely counter-intuitive)
-------------------------------------------------------------------
* a pinned import whose sha256 moved -> exit 2, nothing runs;
* a pass-2 replay whose ``frame`` or ``nearest_all_px`` differs from pass 1 at
  any decision -> the cell is aborted (rc 4) and its file stays ``*.partial``;
* a labelled state whose 36 labels are incomplete, or whose 36 children did not
  all start at the parked parent's frame / state -> the cell is aborted (rc 4);
* the **bomb = 1 invariant is two-branched, never "must spend one"** (v8.6.0's
  measured engine semantics: a press inside a bomb's invulnerability window is a
  no-op).  Counted and reported, never aborting;
* P0a / P0b / P0c / P0d failing -> rc 4 from ``--mode stats`` and the round stops
  there (prereg §2.1): only the sampling statistics are published;
* every episode record carries ``died`` / ``terminal`` / ``error``; an invalid
  episode aborts the cell instead of being dropped.

Not in this file
----------------
* No statistics beyond counts, rates, means, medians, quantiles and
  distributions.  P1-P4 belong to the independent analysis pass (prereg §2).
* No parameter search, no early stopping, no dev reading during collection or
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

WORK = REPO / ".tmp_v8j0_earlybomb"
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.18.0" / "evidence"
           / "EARLYBOMB")
PREREG = REPO / "Document" / "Update" / "Alpha" / "v8.18.0" / (
    "PREREGISTRATION_EARLYBOMB_001.md")
PREREG_REL = "Document/Update/Alpha/v8.18.0/PREREGISTRATION_EARLYBOMB_001.md"
# v8.17.0's own pooled statistics (read-only reference: the round this one is
# placed beside; never written by this file)
V817_STATS = REPO / ".tmp_v8i0_nobomb" / "nobomb_stats.json"

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
    "tools/rl/v860_bombdag.py":
        "3433ae743cc224cae436b6e2192654e30eabe5f05f9053218969e52bdc47cf8a",
    "tools/rl/v400_make_cfg.py":
        "a8c2aabcdda5e5ad1d77e1df6b95d83ff7cab5243d1e757556903d83f1306817",
    "touhou_rl/v420/contracts.py":
        "1357c69c32a74fda20553a3e6b290ca2102b7cc5c52e47e9fdc6eb9e08c946c1",
    "touhou_rl/v420/counterfactual.py":
        "c1f8cae4c5ff8d36fc7d0ea4a5a908bb6bbe83cbc0218474ab647786d38eb9d8",
    "touhou_rl/v400/gym/actions.py":
        "a67145b2bec7c96b3aacaa2514e78cdadf50b056473d15b4ae8122c19c393eff",
    "touhou_rl/v400/gym/protocol.py":
        "9081402dcf699b8efebb473690218a66ebc48900971f0b127f39a31f4c156eb1",
}

SCHEMA_PREFLIGHT = "v8180-earlybomb-preflight/1"
SCHEMA_ASSERT = "v8180-earlybomb-ckpt-assertions/1"
SCHEMA_FIDELITY = "v8180-earlybomb-fidelity/1"
SCHEMA_LABEL = "v8180-earlybomb-label/1"
SCHEMA_COLLECT = "v8180-earlybomb-collect/1"
SCHEMA_SMOKE = "v8180-earlybomb-smoke/1"
SCHEMA_STATS = "v8180-earlybomb-stats/1"
SCHEMA_TRAIN = "v8180-earlybomb-train/1"
SCHEMA_TRAIN_SUMMARY = "v8180-earlybomb-train-summary/1"
SCHEMA_EVAL_SUMMARY = "v8180-earlybomb-eval-summary/1"
SCHEMA_DIAG = "v8180-earlybomb-diagnostics/1"
SCHEMA_BOMB_USAGE = "v8180-earlybomb-bomb-usage/1"
SCHEMA_AUDIT = "v8180-earlybomb-audio-audit/1"
SCHEMA_PROC = "v8180-earlybomb-process-audit/1"
SCHEMA_MANIFEST = "v8180-earlybomb-manifest/1"
SCHEMA_PUBLISH = "v8180-earlybomb-publish/1"
PRODUCER = ("tools/rl/v8j0_earlybomb.py (live engine, HarnessClient role=shadow, "
            "behaviour policy = v4.5.0 `S` stochastic + **no bomb at all**, "
            "900-decision death stop, sampling window "
            "`decisions_before_death` in [30, 60] = 60-120 frames before the "
            "death, 36-way labels at H = 120/240/600)")

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- frozen geometry (unchanged from v4.5.0 / v8.3.0 / v8.6.0 / v8.17.0) ------
PREFIX_FRAMES = 8663
CHECKPOINTS_IN_PREFIX = 2165
ENGINE_SHA256 = "691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6"

# -- frozen split (the v4.5.0 / v6.0.0 / v8.3.0 / v8.17.0 protocol) -----------
TRAIN_WINDOWS: tuple[int, ...] = (13, 14, 15, 16)
DEV_WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
TRAIN_SEEDS: tuple[int, ...] = (0, 1, 2)
SAMPLE_WINDOWS: tuple[int, ...] = (11, 12, 17, 18)

# -- frozen design (prereg v8.18.0 §1) ---------------------------------------
N_CAND = 18
N_ACTIONS = 36
BOMB_VALUES: tuple[int, ...] = (0, 1)
DECISION_FRAMES = 2
DECISIONS = 900                  # 30 s budget; death-stop
HORIZONS: tuple[int, ...] = (120, 240, 600)
H_LABEL_MAX = 600
H_DEFAULT = 120
NEAREST_RECORDED = 12
TARGET_STATES = 600
CELLS = 12                       # 3 seeds x 4 windows (11, 12, 17, 18)
QUOTA_PER_CELL = TARGET_STATES // CELLS                 # 50 -> 12 x 50 = 600
MAX_EPISODES_DEFAULT = 60
MAX_CONSECUTIVE_NO_DEATH = 10
EPISODES = 300
HIDDEN = 64
FEATURE_DIM = 106
LR = 1e-3
BATCH = 256
EPOCHS = 200
TEMPERATURE = 1.0
GEN_CONST = 7
DEFAULT_TIMEOUT = 300.0

# -- **the one moved factor**: the sampling distance to the death (prereg §1) --
WINDOW_SPECS: dict[str, tuple[int, int]] = {
    "tight": (20, 40),           # 40-80 frames before the death (2026-11-22:
                                 # the work order's window: > 40 frames = still
                                 # ALIVE, < 80 frames = bomb=0 dies inside H=120)
    "tighter": (10, 30),         # 20-60 frames (the work order's fallback)
    "early": (30, 60),           # 60-120 frames (prereg §1, superseded by `tight`
                                 # after the smoke's H=120 diff measured 0)
    "late": (60, 120),           # 120-240 frames (prereg §2.1 S-dead's shift)
    "deep": (120, 240),          # 240-480 frames (last rung of the ladder)
}
# -- the death-decision floor (a **degenerate-window** guard, see the docstring) --
# ``decisions_before_death`` in [30, 60] is only "60-120 frames before the death"
# if the death itself happens late enough: a death at decision 45 turns the
# window into decisions 0..15, i.e. the first 0.5 s after the frozen prefix,
# where ``bomb = 0`` simply survives the horizon and the diff is structurally 0.
# An episode whose death comes earlier than this floor is **discarded** (it
# cannot contribute a state) and the cell keeps searching; the threshold is
# recorded per cell and per state so the pool's composition is auditable.
MIN_DEATH_DECISION_DEFAULT = 300
MIN_DEATH_DECISION_FALLBACK = 120   # the work order's relaxation, recorded if used
MIN_DEATH_DECISION_SOURCE = (
    "the 2026-11-22 smoke: a death at decision 45 turned the [30, 60] window into "
    "decisions 0..15 (the first 0.5 s after the prefix), where bomb=0 survives the "
    "horizon by construction -> diff would be a structural 0, not a measurement; "
    "the work order therefore requires death_decision >= 300 (>= 120 if too rare, "
    "recorded in RUN.md)")
WINDOW_NAMES: tuple[str, ...] = ("tight", "tighter", "early", "late", "deep")
WINDOW_DEFAULT = "tight"
WINDOW_AUTO = "auto"
WINDOW_SOURCE = ("the 2026-11-22 smoke and the work order's follow-up: the "
                 "prereg §1 window [30, 60] (60-120 frames) measured H=120 diff "
                 "= 0 because bomb=0 still outlives the horizon from 82-120 "
                 "frames away, so the sampled window is [20, 40] "
                 "(decisions_before_death, 40-80 frames) with [10, 30] (20-60 "
                 "frames) as the fallback")
WINDOW_DESCRIPTION = {
    "tight": "decisions_before_death in [20, 40] = 40-80 frames before the death",
    "tighter": "decisions_before_death in [10, 30] = 20-60 frames before the death",
    "early": "decisions_before_death in [30, 60] = 60-120 frames before the death",
    "late": "decisions_before_death in [60, 120] = 120-240 frames before the death",
    "deep": "decisions_before_death in [120, 240] = 240-480 frames before the death",
}

# -- the behaviour policy: v4.5.0 `S`, and it never spends a bomb -------------
BOMB_RULE = "none"               # bomb = 0 at every decision (prereg §1)
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
    "tools/rl/v860_bombdag.py",
    "tools/rl/v8i0_nobomb.py",
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)

# -- P0a-P0d (prereg §2) ----------------------------------------------------
P0A_ALIVE_MIN = 0.90             # share of sampled states whose player_state is ALIVE
P0B_ARMED_MIN = 0.50             # share with bombs_remaining > 0
P0C_SPENT1_MIN = 0.90            # of honoured bomb=1 branches, spent == 1
P0D_DIFF_MIN = 0.20              # share of states with best bomb1 > best bomb0
BOMB_SEGMENTS: tuple[int, ...] = (3, 2, 1, 0)
ALIVE_STATE_NAME = "ALIVE"

# -- the two arms (prereg §1) -----------------------------------------------
ARMS: tuple[str, ...] = ("A", "B")
ARM_AUX_WEIGHT = {"A": 0.5, "B": 0.0}
ARM_AUX_TARGET_NAME = "best_bomb_argmax"
ARM_DESCRIPTION = {
    "A": ("BCE(36 路) + 0.5 x BCE_bomb(best-bomb: 1 iff best_bomb1 > best_bomb0 "
          "on that row's H，即 argmax alive_frames 落在 bomb=1 时取 1)"),
    "B": "仅 BCE(36 路) — 对照臂",
}
EVAL_BUDGETS = {"A": (300, 600, 900), "B": (300, 600, 900)}

# -- the references this round is placed beside (numbers only, no test) -----
BASELINE_V830 = {
    "arm": "v8.3.0 BOMBPOL-001 — behaviour policy v4.5.0 `S` (never bombs)",
    "episodes_per_budget": 3600,
    "rate": {300: 0.999444, 600: 0.194444, 900: 0.006111},
    "source": "Document/Update/Alpha/v8.3.0/evidence/BOMBPOL/diagnostics.json",
}
BASELINE_V860 = {
    "arm": "v8.6.0 BOMBDAG-001 — S + R5 rule supervision (DAgger on the rule)",
    "episodes_per_budget": 3600,
    "rate": {300: 0.999444, 600: 0.295833, 900: 0.029444},
    "source": ("Document/Update/Alpha/v8.6.0/evidence/BOMBDAG/diagnostics.json "
               "and the v8.11.0 RUN.md BASELINE table (99.94 / 29.58 / 2.94)"),
}
BASELINE_V817 = {
    "arm": "v8.17.0 CONSERVE-001 — death-adjacent (last 20 decisions) + none",
    "rate_20s": 0.141,
    "states_all_dying": True,
    "bombs_per_press": 2,
    "source": "prereg v8.18.0 §0 (20 s 14.1 %)",
}
RULE_R5 = {
    "arm": "v8.4.0 R5 `staged` — script rule: bombs 3 -> 80 px, 2 -> 60 px, "
           "1 -> 40 px, `nearest_all_px < threshold`, bombs > 0",
    "episodes_per_budget": 3600,
    "rate": {300: 1.0, 600: 0.7711, 900: 0.1761},
    "source": "Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/RUN.md §0/§3",
}

_LOG_FH: Any = None
_MODS: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class EarlyBombError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v8j0] {message}"
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
        raise EarlyBombError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cached(name: str, path: pathlib.Path):
    module = _MODS.get(name)
    if module is None:
        module = load_module(name, path)
        if hasattr(module, "_LOG_FH"):
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


def v860():
    """``tools/rl/v860_bombdag.py`` — the frozen v4.5.0 behaviour policy."""

    module = _cached("v860_bombdag", REPO / "tools" / "rl" / "v860_bombdag.py")
    module._LOG_FH = _LOG_FH
    return module


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


def count_map(values: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def features_sha256(feats: Any) -> str:
    import numpy as np

    array = np.asarray(feats, dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def generator_seed(seed: int) -> int:
    """v4.5.0 §2.3's rule, verbatim: ``torch.Generator().manual_seed(seed*1000+7)``."""

    return int(seed) * 1000 + GEN_CONST


def split_action(action: int) -> tuple[int, int]:
    """``action = cand * 2 + bomb`` (pinned by ``v830_bombpol.split_action``)."""

    cand, bomb = divmod(int(action), 2)
    return int(cand), int(bomb)


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
    out["prereg"] = {"path": PREREG_REL, "exists": PREREG.is_file(),
                     "sha256": sha256_file(PREREG) if PREREG.is_file() else None,
                     "bytes": PREREG.stat().st_size if PREREG.is_file() else None}
    if not out["prereg"]["exists"]:
        out["all_match"] = False
    me = pathlib.Path(__file__).resolve()
    out["self"] = {"path": str(me), "sha256": sha256_file(me),
                   "bytes": me.stat().st_size}
    return out


# --------------------------------------------------------------------------- #
# the frozen behaviour decision: `S` + never a bomb
# --------------------------------------------------------------------------- #
def behaviour_decision(*, state: dict[str, Any], hazards: Any, policy: Any,
                       generator: Any, decision: int) -> dict[str, Any]:
    """One behaviour decision under ``--bomb-rule none``: ``bomb = 0`` always.

    The candidate is the frozen 18-way ``S`` student sampled stochastically
    (``v450.policy_action``, the same sampler v4.5.0's evaluation used), exactly
    as v8.16.0 / v8.17.0 did.  Only the bomb bit is fixed: **never** (prereg §1),
    so every sampled state carries all three bombs.
    """

    arm = arm_a()
    feats, _detail = arm.build_features(state=state, hazards=hazards,
                                       decision_index=int(decision))
    cand = v450().policy_action(policy, feats, "stochastic", generator)
    distances = m800().nearest_bullet_distances(state, hazards)
    return {"feats": feats, "features_sha256_k12": features_sha256(feats),
            "cand": int(cand), "bomb": 0, "action": int(cand) * 2,
            "nearest_all_px": distances["nearest_all_px"],
            "nearest_lethal_px": distances["nearest_lethal_px"],
            "bullets": distances["bullets"],
            "bullets_within_60": distances["bullets_within_60"],
            "bombs_before": int(state["bombs_remaining"] or 0),
            "distances": distances}


def aux_best_bomb_target(row: dict[str, Any]) -> int:
    """Arm A's supervision bit: ``1`` iff the label's ``bomb = 1`` beats ``bomb = 0``.

    Computed from ``alive_frames`` alone (per the row's own horizon): the
    candidate with the best branch over both bomb bits is taken
    (``argmax_c max(alive[2c], alive[2c+1])``) and the bit is ``1`` iff that
    candidate's ``bomb = 1`` branch outlives its ``bomb = 0`` branch.  It needs
    no rule, no distance and no behaviour bit.
    """

    alive = [int(value) for value in row["alive_frames"]]
    if len(alive) != N_ACTIONS:
        raise EarlyBombError(f"{len(alive)} alive_frames != {N_ACTIONS}")
    best_cand = max(range(N_CAND),
                    key=lambda cand: max(alive[2 * cand], alive[2 * cand + 1]))
    return int(alive[2 * best_cand + 1] > alive[2 * best_cand])


# --------------------------------------------------------------------------- #
# **the one moved factor**: the sampling window
# --------------------------------------------------------------------------- #
def window_bounds(name: str) -> tuple[int, int]:
    if name not in WINDOW_SPECS:
        raise EarlyBombError(f"unknown window {name!r} (windows: "
                             f"{list(WINDOW_NAMES)} | {WINDOW_AUTO})")
    return WINDOW_SPECS[name]


def sample_decisions(death_decision: int, *, name: str) -> list[int]:
    """The decisions whose ``decisions_before_death`` lies in the window.

    ``decisions_before_death(d) = death_decision - d``, so the window
    ``[lo, hi]`` is ``d in [death - hi, death - lo]``, clipped at 0 (a state can
    never precede the prefix).  Ascending ``d`` = the natural replay order.
    """

    lo, hi = window_bounds(name)
    start = max(0, int(death_decision) - int(hi))
    end = int(death_decision) - int(lo)
    if end < 0:
        return []
    return list(range(start, end + 1))


def alive_share_for_window(trace: Sequence[dict[str, Any]], death_decision: int,
                           *, name: str) -> dict[str, Any]:
    """The ``player_state`` census of one candidate window, from pass 1's trace.

    ``trace`` holds one row per executed decision with the live ``player_state``
    read **before** the decision's frames, which is exactly the state a sampled
    row would be parked on.
    """

    lo, hi = window_bounds(name)
    chosen = sample_decisions(death_decision, name=name)
    states = count_map([trace[d]["player_state_name"] for d in chosen])
    total = len(chosen)
    alive = int(states.get(ALIVE_STATE_NAME, 0))
    return {"window": name, "lo": int(lo), "hi": int(hi),
            "decisions": total,
            "first_decision": (int(chosen[0]) if chosen else None),
            "last_decision": (int(chosen[-1]) if chosen else None),
            "player_state_counts": dict(sorted(states.items())),
            "alive": alive, "alive_fraction": (round(alive / total, 6)
                                               if total else None),
            "alive_gate": P0A_ALIVE_MIN,
            "alive_met": bool(total and (alive / total) >= P0A_ALIVE_MIN)}


def choose_window(trace: Sequence[dict[str, Any]], death_decision: int,
                  *, requested: str) -> tuple[str, list[dict[str, Any]]]:
    """``early`` -> ``late`` -> ``deep``: the first rung whose share reaches 90 %.

    Only used when ``--window auto`` (or with an explicit rung, in which case the
    census of every rung is still returned so the choice is auditable).  When no
    rung reaches the gate the **best** rung is returned with
    ``alive_met = False``, so P0a fails honestly instead of being redefined.
    """

    census = [alive_share_for_window(trace, death_decision, name=name)
              for name in WINDOW_NAMES]
    if requested != WINDOW_AUTO:
        return requested, census
    for block in census:
        if block["decisions"] and block["alive_met"]:
            return block["window"], census
    viable = [block for block in census if block["decisions"]]
    if not viable:
        return WINDOW_NAMES[0], census
    best = max(viable, key=lambda block: (block["alive_fraction"] or 0.0,
                                          -block["lo"]))
    return best["window"], census


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def cell_labels_path(root: pathlib.Path, seed: int, window: int) -> pathlib.Path:
    return root / "labels" / f"seed{int(seed)}_w{int(window)}.jsonl"


def cell_summary_path(root: pathlib.Path, seed: int, window: int) -> pathlib.Path:
    return root / "collect" / f"seed{int(seed)}_w{int(window)}.json"


def pooled_labels_path(root: pathlib.Path) -> pathlib.Path:
    return root / "labels_earlybomb.jsonl"


def pooled_stats_path(root: pathlib.Path) -> pathlib.Path:
    return root / "earlybomb_stats.json"


def policy_ckpt_path(root: pathlib.Path, arm: str, seed: int) -> pathlib.Path:
    return root / "ckpt" / str(arm) / f"seed{int(seed)}" / "earlybomb.pt"


def train_report_path(root: pathlib.Path, arm: str, seed: int) -> pathlib.Path:
    return root / "train" / f"train_{arm}_earlybomb_seed{int(seed)}.json"


def eval_path(root: pathlib.Path, arm: str, budget: int,
              seed: int) -> pathlib.Path:
    return (root / "eval" / str(arm)
            / f"eval_{arm}_earlybomb_d{int(budget)}_seed{int(seed)}.jsonl")


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


def nearest_bullets(state: dict[str, Any], hazards: Any,
                    count: int = NEAREST_RECORDED) -> list[dict[str, Any]]:
    """The ``count`` nearest live bullets, distance-ascending, with vx/vy."""

    px = float(state.get("player_x") or 0.0)
    py = float(state.get("player_y") or 0.0)
    rows: list[dict[str, Any]] = []
    for bullet in hazards.bullets:
        bx = float(bullet.position[0])
        by = float(bullet.position[1])
        dx = bx - px
        dy = by - py
        rows.append({
            "slot": int(bullet.slot),
            "x_px": round(bx, 4), "y_px": round(by, 4),
            "dx_px": round(dx, 4), "dy_px": round(dy, 4),
            "vx_px": round(float(bullet.velocity[0]), 4),
            "vy_px": round(float(bullet.velocity[1]), 4),
            "dist_px": round(math.hypot(dx, dy), 4),
            "lethal": bool(bullet.lethal),
            "state": int(bullet.state),
            "half_width_px": (None if bullet.half_width is None
                              else round(float(bullet.half_width), 4)),
        })
    rows.sort(key=lambda row: (row["dist_px"], row["slot"]))
    return rows[:int(count)]


# --------------------------------------------------------------------------- #
# the H sweep: one 36-child run to H = 600, three horizons by truncation
# --------------------------------------------------------------------------- #
def label_state_horizons(parent: Any, *, state: dict[str, Any],
                         park_reply: str,
                         horizons: Sequence[int] = HORIZONS
                         ) -> tuple[dict[str, Any], dict[int, dict[str, Any]],
                                    dict[str, Any]]:
    """Label one state's 36 actions **once** to ``max(horizons)`` and derive all.

    ``label_child_action`` reports ``alive_frames = death_frame - 1`` when the
    child dies inside the horizon and ``alive_frames = horizon`` when it does not
    (v4.5.0 §2.1 / R24, reused unchanged).  One action's mask sequence does
    **not** depend on the horizon, so

        alive_frames(H) = min(alive_frames(H_max), H)
        safe(H)         = 1  <=>  alive_frames(H) == H

    The derivation is asserted here; on the smoke state it is additionally
    verified against direct H = 120 / H = 240 runs (``verify_horizon_nesting``).
    """

    h_max = max(int(h) for h in horizons)
    if h_max != H_LABEL_MAX:
        raise EarlyBombError(f"the H sweep runs to H = {H_LABEL_MAX}, got {h_max}")
    block, children = v830().label_state36(parent, state=state, horizon=h_max,
                                           park_reply=park_reply)
    if len(block["alive_frames"]) != N_ACTIONS:
        raise EarlyBombError(f"{len(block['alive_frames'])} alive_frames != "
                             f"{N_ACTIONS}")
    alive_max = [int(value) for value in block["alive_frames"]]
    derived: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        h = int(horizon)
        if h > h_max:
            raise EarlyBombError(f"H={h} > H_max={h_max}")
        alive = [min(value, h) for value in alive_max]
        safe = [1 if value == h else 0 for value in alive]
        for action in range(N_ACTIONS):
            if (safe[action] == 1) != (alive[action] == h):
                raise EarlyBombError(f"H={h} a{action}: derived safe/alive "
                                     f"disagree")
        derived[h] = {
            "H": h, "alive_frames": alive, "safe": safe,
            "safe_actions": int(sum(safe)),
            "deaths": int(sum(1 for value in alive if value < h)),
            "max_alive_frames": int(max(alive)),
            "best_bomb0": int(max(alive[a] for a in range(N_ACTIONS)
                                 if split_action(a)[1] == 0)),
            "best_bomb1": int(max(alive[a] for a in range(N_ACTIONS)
                                 if split_action(a)[1] == 1)),
            "terminals": count_map(
                [str(row["terminal"]) for row in children
                 if row["terminal"] is not None
                 and int(row["alive_frames"]) < h]),
        }
    meta = {"H_max": int(h_max), "horizons": [int(h) for h in horizons],
            "horizon_source": ("alive(H) = min(alive(H_max), H); the mask "
                               "sequence of an action is horizon-independent"),
            "children": int(len(children)),
            "label_wall_s": block["label_wall_s"]}
    return block, derived, meta


def verify_horizon_nesting(parent: Any, *, state: dict[str, Any],
                           park_reply: str, block: dict[str, Any],
                           horizons: Sequence[int] = HORIZONS
                           ) -> dict[str, Any]:
    """Direct H = 120 / H = 240 runs vs. the truncation of the H = 600 run.

    Called on the smoke's states (``--verify-h 1``).  A single difference is a
    fail-closed error: it would mean the H sweep is not the truncation it claims
    to be.
    """

    h_max = max(int(h) for h in horizons)
    alive_max = [int(value) for value in block["alive_frames"]]
    out: dict[str, Any] = {"H_max": int(h_max), "checked": {}, "mismatches": 0,
                           "all_match": True}
    for horizon in horizons:
        h = int(horizon)
        if h >= h_max:
            continue
        direct, _children = v830().label_state36(parent, state=state,
                                                 horizon=h, park_reply=park_reply)
        direct_alive = [int(value) for value in direct["alive_frames"]]
        derived_alive = [min(value, h) for value in alive_max]
        mismatched = [a for a in range(N_ACTIONS)
                      if direct_alive[a] != derived_alive[a]]
        out["checked"][str(h)] = {
            "actions": N_ACTIONS,
            "mismatched_actions": len(mismatched),
            "first_mismatch": (None if not mismatched
                               else {"action": mismatched[0],
                                     "direct": direct_alive[mismatched[0]],
                                     "derived": derived_alive[mismatched[0]]}),
            "direct_safe_actions": int(sum(1 for v in direct_alive if v == h)),
            "derived_safe_actions": int(sum(1 for v in derived_alive if v == h)),
            "label_wall_s": direct["label_wall_s"],
        }
        out["mismatches"] += len(mismatched)
    out["all_match"] = bool(out["mismatches"] == 0)
    if not out["all_match"]:
        raise EarlyBombError(f"H-nesting verification failed: {out}")
    return out


# --------------------------------------------------------------------------- #
# the P0 screen (shared by --mode smoke and --mode stats)
# --------------------------------------------------------------------------- #
def state_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the three per-horizon rows of every state into one record.

    Fails closed on a state that does not carry all of ``HORIZONS``, on a
    ``safe`` / ``alive_frames`` disagreement, and on an ``alive_frames`` vector
    that is not the truncation of the H = 600 run.
    """

    by_state: dict[tuple[int, int, int, int], dict[int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        horizon = int(row["H"])
        if horizon not in HORIZONS:
            raise EarlyBombError(f"row {index}: H={horizon} is not one of "
                                 f"{list(HORIZONS)}")
        key = (int(row["seed"]), int(row["window"]), int(row["episode"]),
               int(row["decision"]))
        slot = by_state.setdefault(key, {})
        if horizon in slot:
            raise EarlyBombError(f"row {index}: state {key} has two H={horizon} "
                                 f"rows")
        slot[horizon] = row
    incomplete = [key for key, slot in by_state.items()
                  if set(slot) != set(HORIZONS)]
    if incomplete:
        raise EarlyBombError(f"{len(incomplete)} states do not carry all "
                             f"{list(HORIZONS)} horizons (first {incomplete[0]})")
    collapsed: list[dict[str, Any]] = []
    for key, slot in by_state.items():
        base = slot[int(H_LABEL_MAX)]
        alive_max = [int(value) for value in base["alive_frames"]]
        entry: dict[str, Any] = {
            "seed": int(base["seed"]), "window": int(base["window"]),
            "episode": int(base["episode"]), "decision": int(base["decision"]),
            "frame": int(base["frame"]),
            "player_state": int(base.get("player_state") or 0),
            "player_state_name": str(base.get("player_state_name") or "?"),
            "bombs_remaining": int(base["bombs_remaining"]),
            "decisions_before_death": int(base["decisions_before_death"]),
            "frames_before_death": int(base["frames_before_death"]),
            "nearest_all_px": base.get("nearest_all_px"),
            "bomb1_spent_amount": [int(v) for v in base["bomb1_spent_amount"]],
            "bomb1_honoured": int(base["bomb1_spent_honoured"]),
            "bomb1_spent1": int(base["bomb1_spent_one"]),
            "bomb1_spent2plus": int(base["bomb1_spent_two_or_more"]),
            "bomb1_noop": int(base["bomb1_noop_count"]),
            "by_H": {},
        }
        for horizon in HORIZONS:
            h = int(horizon)
            row = slot[h]
            alive = [int(value) for value in row["alive_frames"]]
            safe = [int(value) for value in row["safe"]]
            if len(alive) != N_ACTIONS or len(safe) != N_ACTIONS:
                raise EarlyBombError(f"state {key} H={h}: {len(alive)}/"
                                     f"{len(safe)} labels")
            if [min(value, h) for value in alive_max] != alive:
                raise EarlyBombError(f"state {key} H={h}: alive_frames is not "
                                     f"the truncation of the H={H_LABEL_MAX} run")
            for action in range(N_ACTIONS):
                if (safe[action] == 1) != (alive[action] == h):
                    raise EarlyBombError(f"state {key} H={h} a{action}: safe="
                                         f"{safe[action]} alive={alive[action]}")
            entry["by_H"][str(h)] = {
                "best_bomb0": int(max(alive[a] for a in range(N_ACTIONS)
                                      if split_action(a)[1] == 0)),
                "best_bomb1": int(max(alive[a] for a in range(N_ACTIONS)
                                      if split_action(a)[1] == 1)),
                "delta": int(max(alive[a] for a in range(N_ACTIONS)
                                 if split_action(a)[1] == 1)
                             - max(alive[a] for a in range(N_ACTIONS)
                                   if split_action(a)[1] == 0)),
                "max_alive": int(max(alive)),
                "critical": bool(max(alive) < h),
                "safe_actions": int(sum(safe)),
            }
        collapsed.append(entry)
    return collapsed


def p0_block(states: Sequence[dict[str, Any]],
             horizon: int | None = None) -> dict[str, Any]:
    """P0a / P0b / P0c (state level) and P0d (per horizon) of one sample."""

    total = len(states)
    alive = sum(1 for entry in states
                if entry["player_state_name"] == ALIVE_STATE_NAME)
    armed = sum(1 for entry in states if entry["bombs_remaining"] > 0)
    honoured = sum(entry["bomb1_honoured"] for entry in states)
    spent1 = sum(entry["bomb1_spent1"] for entry in states)
    spent2 = sum(entry["bomb1_spent2plus"] for entry in states)
    noop = sum(entry["bomb1_noop"] for entry in states)
    branches = sum(len(entry["bomb1_spent_amount"]) for entry in states)
    p0a = (alive / total) if total else None
    p0b = (armed / total) if total else None
    p0c = (spent1 / honoured) if honoured else None
    out: dict[str, Any] = {
        "states": total,
        "p0a": {"alive": alive, "fraction": (round(p0a, 6) if p0a is not None
                                             else None),
                "gate": P0A_ALIVE_MIN,
                "met": bool(total and p0a is not None and p0a >= P0A_ALIVE_MIN),
                "player_state_counts": count_map(
                    [entry["player_state_name"] for entry in states])},
        "p0b": {"armed": armed, "fraction": (round(p0b, 6) if p0b is not None
                                             else None),
                "gate": P0B_ARMED_MIN,
                "met": bool(total and p0b is not None and p0b >= P0B_ARMED_MIN),
                "bombs_remaining_counts": count_map(
                    [entry["bombs_remaining"] for entry in states])},
        "p0c": {"bomb1_branches": branches,
                "bomb1_honoured": honoured,
                "bomb1_spent_amount_counts": count_map(
                    [amount for entry in states
                     for amount in entry["bomb1_spent_amount"]]),
                "bomb1_spent_one": spent1,
                "bomb1_spent_two_or_more": spent2,
                "bomb1_noop_due_to_invuln": noop,
                "honoured_spent1_fraction": (round(p0c, 6) if p0c is not None
                                             else None),
                "gate": P0C_SPENT1_MIN,
                "met": bool(honoured and p0c is not None and p0c >= P0C_SPENT1_MIN)},
    }
    if horizon is not None:
        h = int(horizon)
        deltas = [entry["by_H"][str(h)] for entry in states]
        diff = sum(1 for item in deltas if item["delta"] > 0)
        equal = sum(1 for item in deltas if item["delta"] == 0)
        worse = sum(1 for item in deltas if item["delta"] < 0)
        critical = sum(1 for item in deltas if item["critical"])
        out["p0d"] = {
            "H": h, "states": total,
            "diff_states": diff,
            "fraction": (round(diff / total, 6) if total else None),
            "gate": P0D_DIFF_MIN,
            "met": bool(total and (diff / total) >= P0D_DIFF_MIN),
            "equal_states": equal, "worse_states": worse,
            "critical": critical,
            "critical_fraction": (round(critical / total, 6) if total else None),
            "delta_quantiles": quantiles([item["delta"] for item in deltas]),
            "best_bomb0_quantiles": quantiles([item["best_bomb0"]
                                               for item in deltas]),
            "best_bomb1_quantiles": quantiles([item["best_bomb1"]
                                               for item in deltas]),
            "mean_alive_over_36": (round(statistics.fmean(
                entry["max_alive"] for entry in deltas), 4) if total else None),
        }
    return out


def p0_summary(states: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """P0a-P0c once (state level) and P0d at every horizon."""

    base = p0_block(states)
    by_h = {str(int(h)): p0_block(states, horizon=int(h))["p0d"] for h in HORIZONS}
    return {"state_level": base, "by_H": by_h,
            "p0a_met": base["p0a"]["met"],
            "p0b_met": base["p0b"]["met"],
            "p0c_met": base["p0c"]["met"],
            "p0d_met_any": bool(any(block["met"] for block in by_h.values())),
            "p0d_horizons_met": [int(h) for h in HORIZONS
                                 if by_h[str(int(h))]["met"]],
            "all_four_met": bool(base["p0a"]["met"] and base["p0b"]["met"]
                                 and base["p0c"]["met"]
                                 and any(block["met"] for block in by_h.values()))}


def verdict_of(summary: dict[str, Any]) -> dict[str, Any]:
    """Prereg §2.1: S-dead / S-2bomb / S-nodiff / train."""

    if not summary["p0a_met"]:
        name = "S-dead"
        rule = ("P0a < 90 % ALIVE: the window is still inside the death "
                "animation; re-collect with the next rung of the ladder")
    elif not summary["p0c_met"]:
        name = "S-2bomb"
        rule = ("P0c < 90 % spent == 1: a press still costs two bombs; filter "
                "by player_state == ALIVE instead of by decisions_before_death")
    elif not summary["p0d_met_any"]:
        name = "S-nodiff"
        rule = ("P0a and P0c hold but no horizon reaches 20 % diff: a bomb one "
                "or two seconds early does not save inside H; publish the "
                "statistics, do not train")
    else:
        name = "train"
        rule = ("the smallest horizon reaching the P0d gate is the one to "
                "train with")
    return {"name": name, "rule": rule,
            "train_horizon": (min(summary["p0d_horizons_met"])
                              if name == "train" else None),
            "p0d_horizons_met": summary["p0d_horizons_met"]}


def choose_pool(states_all: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The prereg §2.1 screen: the whole sample, and the ``ALIVE`` sub-pool.

    The window is a *distance* to the death, so its lower rungs sit inside the
    death animation: ``player_state = DYING`` there, and a press costs **two**
    bombs (which is exactly v8.17.0's regime, measured again here as the
    ``spent == 2`` branches).  Prereg §2.1's ``S-2bomb`` remedy is to filter by
    ``player_state == ALIVE`` rather than by the distance, so both pools are
    measured:

    * ``p0_all``   - every sampled state (the window as prereg §1 defines it);
    * ``p0_alive`` - the states whose ``player_state`` is ``ALIVE``.

    The **chosen** pool is the full one when it already meets P0a and P0c, else
    the ``ALIVE`` pool (``filter = "player_state == ALIVE"``) - never silently:
    both are written into the statistics file.
    """

    alive_states = [entry for entry in states_all
                    if entry["player_state_name"] == ALIVE_STATE_NAME]
    summary_all = p0_summary(states_all)
    summary_alive = p0_summary(alive_states) if alive_states else None
    needs_filter = bool(not summary_all["p0a_met"] or not summary_all["p0c_met"])
    if needs_filter and summary_alive is not None:
        chosen, filter_name = summary_alive, ALIVE_STATE_NAME
    else:
        chosen, filter_name = summary_all, "none"
    verdict = verdict_of(chosen)
    verdict["filter"] = filter_name
    verdict["filter_reason"] = (
        "P0a/P0c fail on the whole window: the lower rungs are inside the death "
        "animation, where a press costs two bombs (prereg §2.1 S-2bomb)"
        if filter_name != "none" else
        "the whole window already meets P0a and P0c")
    return {"states_all": len(states_all), "states_alive": len(alive_states),
            "p0_all": summary_all, "p0_alive": summary_alive,
            "p0": chosen, "summary": chosen, "verdict": verdict,
            "filter": filter_name, "filtered_states": alive_states}


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
                   "single_factor": ("the sampling distance to the death: the "
                                     "last 20 decisions (v8.17.0) -> "
                                     "decisions_before_death in [30, 60]")},
        "pins": pins, "engine": engine, "launch_env": prepare_env(),
        "feature_dim": int(arm_a().feature_dim()),
        "action_space": {"candidates": N_CAND, "bombs": list(BOMB_VALUES),
                         "actions": N_ACTIONS,
                         "encoding": "action = cand * 2 + bomb",
                         "bomb1_semantics": ("cand_to_mask(cand, 1) on the "
                                             "label's first frame, "
                                             "cand_to_mask(cand, 0) afterwards "
                                             "(v8.3.0 §1, unchanged)")},
        "behaviour": {
            "policy": "v4.5.0 `S` — .tmp_v450_cf/ckpt/seed{0,1,2}/round2.pt",
            "kind": "stochastic", "temperature": TEMPERATURE,
            "generator": "torch.Generator().manual_seed(seed*1000+7)",
            "bomb_rule": BOMB_RULE,
            "bomb_rule_description": ("the behaviour never spends a bomb, so "
                                      "every sampled state carries all three "
                                      "(prereg §1)"),
            "checkpoints": {str(s): {"path": str(v860().behaviour_ckpt(s)),
                                     "expected": FROZEN_ROUND2_SHA256[s]}
                            for s in TRAIN_SEEDS},
            "roll_out": (f"roll until the player dies, cap {DECISIONS} decisions "
                         f"({DECISIONS * DECISION_FRAMES} frames = "
                         f"{DECISIONS * DECISION_FRAMES / 30.0:.1f} s at 30 fps); "
                         f"an episode that does not die is discarded")},
        "sampling": {
            "kind": "early-death-adjacent",
            "window_default": WINDOW_DEFAULT,
            "windows": {name: {"lo": WINDOW_SPECS[name][0],
                               "hi": WINDOW_SPECS[name][1],
                               "frames": (WINDOW_SPECS[name][0] * DECISION_FRAMES,
                                          WINDOW_SPECS[name][1] * DECISION_FRAMES),
                               "description": WINDOW_DESCRIPTION[name]}
                        for name in WINDOW_NAMES},
            "window_auto": ("early -> late -> deep: the first rung whose sampled "
                            "ALIVE share reaches P0a's 90 %, measured on pass 1's "
                            "per-decision player_state trace"),
            "window_source": WINDOW_SOURCE,
            "horizons": [int(h) for h in HORIZONS],
            "horizon_max": H_LABEL_MAX,
            "horizon_source": ("one 36-child run to H = 600; alive(H) = "
                               "min(alive(600), H), verified against direct "
                               "H = 120 / H = 240 runs on the smoke's states"),
            "windows": list(SAMPLE_WINDOWS),
            "max_consecutive_no_death": MAX_CONSECUTIVE_NO_DEATH,
            "target_states": TARGET_STATES,
            "cells": CELLS, "quota_per_cell": QUOTA_PER_CELL,
            "rule": ("decisions_before_death = death_decision - decision, kept in "
                     "the window of the rung"),
            "live_nearest": ("nearest_all_px = min over the live HAZARDS bullet "
                             "list at that decision (not the 12 stored bullets)"),
            "nearest_bullets_recorded": NEAREST_RECORDED,
            "two_pass": ("pass 1 rolls the trajectory to death and records every "
                         "decision's player_state / bombs_remaining / "
                         "nearest_all_px / action; pass 2 replays the recorded "
                         "actions up to the last sampled decision and labels only "
                         "the sampled ones"),
            "replay_check": ("frame and nearest_all_px must equal pass 1 at every "
                             "decision of pass 2; a difference aborts the cell")},
        "arms": {arm: {"loss": ARM_DESCRIPTION[arm],
                       "aux_weight": ARM_AUX_WEIGHT[arm],
                       "eval_budgets": list(EVAL_BUDGETS[arm])} for arm in ARMS},
        "p0": {"p0a_alive_min": P0A_ALIVE_MIN, "p0b_armed_min": P0B_ARMED_MIN,
               "p0c_spent1_min": P0C_SPENT1_MIN, "p0d_diff_min": P0D_DIFF_MIN},
        "frozen": {"TRAIN_WINDOWS": list(TRAIN_WINDOWS),
                   "DEV_WINDOWS": list(DEV_WINDOWS),
                   "SAMPLE_WINDOWS": list(SAMPLE_WINDOWS),
                   "train_seeds": list(TRAIN_SEEDS),
                   "H": H_DEFAULT, "horizons": [int(h) for h in HORIZONS],
                   "H_max": H_LABEL_MAX, "decisions_cap": DECISIONS,
                   "decision_frames": DECISION_FRAMES,
                   "episodes_per_window": EPISODES,
                   "lr": LR, "batch": BATCH, "epochs": EPOCHS, "hidden": HIDDEN,
                   "target_states": TARGET_STATES, "cells": CELLS,
                   "quota_per_cell": QUOTA_PER_CELL,
                   "max_episodes_default": MAX_EPISODES_DEFAULT,
                   "max_consecutive_no_death": MAX_CONSECUTIVE_NO_DEATH},
        "references": {"v830_bombpol": BASELINE_V830, "v860_bombdag": BASELINE_V860,
                       "v8170_conserve": BASELINE_V817, "rule_r5": RULE_R5},
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
    out_path = (pathlib.Path(args.out)
                if args.out else pathlib.Path(args.work_dir)
                / f"ckpt_assertions{'' if args.phase == 'start' else '_final'}.json")
    payload: dict[str, Any] = {
        "schema": SCHEMA_ASSERT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": args.phase,
        "requirement": ("checklist item 2: the behaviour policy of every v8.18.0 "
                        "state is the frozen v4.5.0 `S` student, whose final "
                        "checkpoints are round2.pt = 4abee069... / a93746bb... / "
                        "c69960fc...; the protected sources must not have moved"),
        "engine": None, "ckpts": {}, "protected": {}, "v8j0_earlybomb_py": {},
        "prereg": {}, "all_pass": False,
    }
    try:
        payload["engine"] = engine_gate(args.engine)
        ok = bool(payload["engine"].get("match"))
        for seed in TRAIN_SEEDS:
            path = v860().behaviour_ckpt(seed)
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
        payload["v8j0_earlybomb_py"] = {"path": str(me), "sha256": sha256_file(me)}
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
# mode: fidelity (the D3 gate, run through v432 itself)
# --------------------------------------------------------------------------- #
def engine_env() -> dict[str, str]:
    """AGENTS.md §9.3 on the child process: dummy audio, no display."""

    import os

    env = dict(os.environ)
    env["SDL_AUDIODRIVER"] = "dummy"
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"]
                                     if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def mode_fidelity(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out = pathlib.Path(args.out) if args.out else root / "fidelity_d3.json"
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "v8j0-fidelity"
    argv = [sys.executable, str(REPO / "tools" / "rl" / "v432_fork_arm.py"),
            "--mode", "fidelity", "--seeds", "12", "--learner", "frozen",
            "--work-dir", str(work), "--out", str(out), "--tag", tag]
    started = time.perf_counter()
    proc = subprocess.run(argv, capture_output=True, text=True,
                          env=engine_env())
    wall = round(time.perf_counter() - started, 3)
    payload: dict[str, Any] = {
        "schema": SCHEMA_FIDELITY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("the D3 fidelity gate (five assertions) must pass before "
                        "any label of this round is trusted; prereg §2.1 S-fork "
                        "voids the round otherwise"),
        "delegated_to": ("tools/rl/v432_fork_arm.py --mode fidelity (run as a "
                         "subprocess: the gate is v432's, not re-implemented "
                         "here)"),
        "argv": argv, "returncode": proc.returncode, "wall_s": wall,
        "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:],
        "out": str(out), "assertions": {}, "assertions_present": [],
        "all_pass": False, "verdict": None, "engine": None,
    }
    try:
        if out.is_file():
            rec = read_json(out)
            payload["assertions"] = {
                key: {"requirement": value.get("requirement"),
                      "passed": bool(value.get("passed"))}
                for key, value in sorted(rec.get("assertions", {}).items())}
            payload["assertions_present"] = sorted(rec.get("assertions", {}))
            payload["verdict"] = rec.get("verdict")
            payload["engine"] = rec.get("engine")
            payload["d3_all_pass"] = bool(rec.get("all_pass"))
            payload["d3_wall_s"] = rec.get("wall_s")
            payload["data_dir"] = rec.get("data_dir")
            payload["cfg"] = (rec.get("data") or {}).get("cfg")
            payload["launch_env_from_gate"] = rec.get("launch_env")
            payload["all_pass"] = bool(rec.get("all_pass"))
        else:
            payload["error"] = f"fidelity output not written: {out}"
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
    payload["five_assertions_present"] = bool(
        len(payload["assertions"]) == 5
        and all(item["passed"] for item in payload["assertions"].values()))
    write_json(root / "fidelity_d3_wrapper.json", payload)
    log(f"fidelity: rc={proc.returncode} assertions={len(payload['assertions'])} "
        f"all_passed={payload['five_assertions_present']} "
        f"d3_all_pass={payload.get('d3_all_pass')} wall={wall}s")
    return 0 if (payload["all_pass"] and payload["five_assertions_present"]) else 2


# --------------------------------------------------------------------------- #
# the cell runner: one (seed, window) death-trajectory cell
# --------------------------------------------------------------------------- #
def run_cell(*, seed: int, window: int, quota: int, max_episodes: int,
             window_name: str, decisions_cap: int, engine: str,
             data_source: str, root: pathlib.Path, tag: str,
             labels: pathlib.Path | None, summary_path: pathlib.Path | None,
             verify_h: bool, stop_after_death_episodes: int | None,
             schema: str, smoke: bool,
             min_death_decision: int = MIN_DEATH_DECISION_DEFAULT) -> dict[str, Any]:
    """Roll to death, sample the window, label 36 ways, write one cell.

    ``window_name`` may be ``auto``; the choice is then made from pass 1's
    per-decision ``player_state`` trace of the first death episode (the ``deep``
    rung is only used when ``early`` and ``late`` both miss the P0a gate).
    """

    import torch

    from touhou_rl.v400.gym import protocol as P
    from touhou_rl.v400.labels.state import player_state_name
    from touhou_rl.v420 import counterfactual as cf

    if seed not in TRAIN_SEEDS:
        raise EarlyBombError(f"--seed must be one of {TRAIN_SEEDS}, got {seed}")
    if window_name not in WINDOW_NAMES and window_name != WINDOW_AUTO:
        raise EarlyBombError(f"--window must be one of {list(WINDOW_NAMES)} or "
                             f"{WINDOW_AUTO!r}, got {window_name!r}")
    if decisions_cap != DECISIONS:
        raise EarlyBombError(f"the round fixes the trajectory budget at "
                             f"{DECISIONS} decisions (="
                             f"{DECISIONS * DECISION_FRAMES} frames = "
                             f"{DECISIONS * DECISION_FRAMES / 30.0:.0f} s) per "
                             f"episode; got {decisions_cap}")
    for index in (window,):
        if index not in SAMPLE_WINDOWS:
            raise EarlyBombError(f"window {index} is not one of the work order's "
                                 f"sampling windows {SAMPLE_WINDOWS}")
    pins = check_pins()
    if not pins["all_match"]:
        raise EarlyBombError(f"pinned import moved: {json.dumps(pins)[:800]}")
    engine_info = engine_gate(engine)
    policy, ckpt, ckpt_sha = v860().load_behaviour_policy(seed)
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    launch_env = prepare_env()
    bomb1_actions = [a for a in range(N_ACTIONS) if split_action(a)[1] == 1]
    bomb0_actions = [a for a in range(N_ACTIONS) if split_action(a)[1] == 0]

    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    labels = labels if labels is not None else cell_labels_path(root, seed, window)
    labels.parent.mkdir(parents=True, exist_ok=True)
    partial = labels.with_suffix(labels.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    summary_path = (summary_path if summary_path is not None
                    else cell_summary_path(root, seed, window))
    masks, hashes, _masks_meta = v450().masks_for(window)
    rec: dict[str, Any] = {
        "schema": schema, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL, "seed": seed, "window": int(window),
        "quota": int(quota), "decisions_cap": decisions_cap,
        "max_episodes": int(max_episodes), "H_max": H_LABEL_MAX,
        "horizons": list(HORIZONS), "decisions_budget": DECISIONS,
        "max_consecutive_no_death": MAX_CONSECUTIVE_NO_DEATH,
        "sampling_windows": list(SAMPLE_WINDOWS),
        "window_requested": window_name,
        "window_name": (window_name if window_name != WINDOW_AUTO else None),
        "window_lo": None, "window_hi": None,
        "window_census": None, "window_auto_choice": None,
        "window_source": WINDOW_SOURCE,
        "actions_per_state": N_ACTIONS,
        "state_rows_per_state": len(HORIZONS),
        "sampling": {"kind": "early-death-adjacent",
                     "rule": ("decisions_before_death = death_decision - "
                              "decision, kept inside the chosen window"),
                     "nearest_bullets_recorded": NEAREST_RECORDED,
                     "player_state_recorded_per_decision": True},
        "behaviour": {
            "policy": "v4.5.0 `S` (frozen DAgger student, round2.pt)",
            "ckpt": str(ckpt), "ckpt_sha256": ckpt_sha,
            "kind": "stochastic", "generator_seed": generator_seed(seed),
            "bomb_rule": BOMB_RULE,
            "bomb_rule_description": ("never: fired = 0 at every decision, so "
                                      "every sampled state has 3 bombs"),
            "roll_out": (f"until death, cap {decisions_cap} decisions "
                         f"({decisions_cap * DECISION_FRAMES} frames)"),
            "mask_schedule": ("cand_to_mask(cand, bomb) on the decision's first "
                              "frame, cand_to_mask(cand, 0) afterwards")},
        "bomb1_semantics": ("cand_to_mask(cand, 1) on the label's first frame, "
                            "cand_to_mask(cand, 0) on frames 2..H (v8.3.0 §1)"),
        "pins": pins, "engine": engine_info, "launch_env": launch_env,
        "labels_path": str(labels), "status": "failed", "error": None,
        "episodes": [], "states": 0, "death_episodes": 0,
        "no_death_episodes": 0, "invalid_episodes": 0,
        "min_death_decision": int(min_death_decision),
        "min_death_decision_source": MIN_DEATH_DECISION_SOURCE,
        "early_death_episodes": 0, "death_decisions_rejected": [],
        "states_by_H": {str(h): 0 for h in HORIZONS},
        "consecutive_no_death": 0, "consecutive_no_death_max": 0,
        "window_skipped_after": None, "skip_reason": None,
        "nesting": None, "nesting_verified": None,
        "replay_checks": 0, "replay_mismatches": 0,
        "player_state_counts": {}, "player_state_by_before_death": {},
        "bombs_remaining_counts": {}, "decisions_before_death_counts": {},
        "death_decision_quantiles": None, "bombs_at_death_counts": {},
        "label_checks": {"labels_36": 0, "bomb1_press_honoured": 0,
                         "bomb1_press_ignored": 0,
                         "bomb1_spent_one": 0, "bomb1_spent_two_or_more": 0,
                         "bomb1_spent_amounts": {}, "bomb1_noop_due_to_invuln": 0,
                         "bomb0_spent_a_bomb": 0,
                         "states_by_player_state": {},
                         "bomb1_anomalies": 0, "states_without_a_bomb_left": 0},
        "gate": {}, "cfg": {}, "wall_s": None, "state_costs_s": [],
        "pass_costs_s": {"pass1": [], "pass2": []},
    }
    handle = partial.open("w", encoding="utf-8")
    parents: list[Any] = []
    started = time.perf_counter()
    deaths0 = 0.0
    stage0 = 0
    state: dict[str, Any] = {}
    death_decisions: list[int] = []
    window_name_used = (window_name if window_name != WINDOW_AUTO else None)
    try:
        data_dir, data_info = v450().prepare_data_dir(data_root, window,
                                                      data_source)
        rec["cfg"] = data_info["cfg"]
        rec["data_dir"] = str(data_dir)
        parent, gate = start_parent(slot=0, window=window, engine=engine,
                                    data_dir=data_dir, log_dir=run_dir)
        parents.append(parent)
        rec["gate"] = {key: gate[key] for key in (
            "hello_role_shadow", "park_ok", "park_frame", "threads_1",
            "all_pass")}
        rec["gate"]["prefix"] = {key: gate["prefix"][key] for key in (
            "hash_checks", "hash_mismatches", "final_frame", "wall_s")}
        state, _reply = parent.state()
        if int(state["frame"]) != PREFIX_FRAMES:
            raise EarlyBombError(f"w{window}: parked parent at frame "
                                 f"{state['frame']}, expected {PREFIX_FRAMES}")
        deaths0 = float(state["deaths"] or 0.0)
        stage0 = int(state["stage"] or 0)
        episode = 0
        consecutive_no_death = 0
        consecutive_rejected = 0
        while rec["states"] < quota and episode < max_episodes:
            if (stop_after_death_episodes is not None
                    and rec["death_episodes"] >= int(stop_after_death_episodes)):
                log(f"s{seed} w{window}: stopping after "
                    f"{rec['death_episodes']} death episode(s)")
                break
            ep: dict[str, Any] = {
                "episode": int(episode), "decisions": 0, "frames": 0,
                "died": False, "terminal": None, "death_decision": None,
                "death_frames_executed": None,
                "bombs_start": int(state["bombs_remaining"] or 0),
                "bombs_end": None, "bombs_used": None, "bomb_emits": 0,
                "states_labelled": 0, "fires": 0,
                "bombs_remaining_counts": {}, "bomb_decision_counts": {},
                "player_state_counts": {}, "prefix": None, "window": None,
                "window_census": None, "pass1_s": None, "pass2_s": None,
                "replay_checks": 0, "replay_mismatches": 0,
                "player_state_trace": [], "death_player_state": None}
            if episode > 0:
                prefix = parent.walk_prefix(masks, hashes, window)
                parent.park()
                state, _reply = parent.state()
                if int(state["frame"]) != PREFIX_FRAMES:
                    raise EarlyBombError(
                        f"w{window} ep{episode}: warm RESET left the parent at "
                        f"frame {state['frame']}")
                deaths0 = float(state["deaths"] or 0.0)
                stage0 = int(state["stage"] or 0)
                ep["bombs_start"] = int(state["bombs_remaining"] or 0)
                ep["prefix"] = {"hash_checks": prefix["hash_checks"],
                                "hash_mismatches": prefix["hash_mismatches"],
                                "final_frame": prefix["final_frame"]}
            # ---------------- pass 1: the behaviour trajectory ---------------
            pass1_started = time.perf_counter()
            trace: list[dict[str, Any]] = []
            died = False
            for decision in range(decisions_cap):
                state_frame = int(state["frame"])
                player_state = int(state.get("player_state") or 0)
                player_name = player_state_name(player_state)
                hazards = P.parse_hazards_reply(parent.ask("HAZARDS"))
                got = behaviour_decision(state=state, hazards=hazards,
                                         policy=policy, generator=generator,
                                         decision=decision)
                park_reply = parent.park()
                got_frame = m432().fields(park_reply).get("frame")
                if got_frame != str(state_frame):
                    raise EarlyBombError(
                        f"w{window} d{decision}: PARK reports frame {got_frame}, "
                        f"the read state is at {state_frame}")
                trace.append({
                    "decision": int(decision), "frame": state_frame,
                    "player_state": player_state,
                    "player_state_name": player_name,
                    "nearest_all_px": got["nearest_all_px"],
                    "nearest_lethal_px": got["nearest_lethal_px"],
                    "bullets": int(got["bullets"]),
                    "bullets_within_60": int(got["bullets_within_60"]),
                    "bombs_before": int(got["bombs_before"]),
                    "cand": int(got["cand"]), "rule_bomb": 0,
                    "action": int(got["action"]),
                })
                ep["decisions"] += 1
                ep["player_state_trace"].append(
                    [int(decision), int(player_state), int(got["bombs_before"]),
                     got["nearest_all_px"]])
                ep["player_state_counts"][player_name] = (
                    ep["player_state_counts"].get(player_name, 0) + 1)
                rec["player_state_counts"][player_name] = (
                    rec["player_state_counts"].get(player_name, 0) + 1)
                key = str(int(got["bombs_before"]))
                ep["bombs_remaining_counts"][key] = (
                    ep["bombs_remaining_counts"].get(key, 0) + 1)
                rec["bombs_remaining_counts"][key] = (
                    rec["bombs_remaining_counts"].get(key, 0) + 1)
                first_mask, rest_mask = v830().action_masks(got["action"])
                terminal = None
                executed = 0
                for frame_index in range(DECISION_FRAMES):
                    step = parent.step(first_mask if frame_index == 0
                                       else rest_mask)
                    if step["advanced"] == 1:
                        ep["frames"] += 1
                        executed += 1
                    reading, _reply = parent.state()
                    terminal = cf._terminal_of(
                        reading, deaths_start=deaths0, stage0=stage0,
                        hash_hex=(step["hash"] or ""),
                        frame=int(reading["frame"]), exited=step["exited"],
                        game_over=step["game_over"])
                    state = reading
                    if terminal is not None:
                        ep["terminal"] = str(terminal.get("terminal"))
                        if terminal.get("engine_dead"):
                            ep["died"] = True
                        died = True
                        break
                if died:
                    ep["death_decision"] = int(decision)
                    ep["death_frames_executed"] = int(executed)
                    ep["death_player_state"] = player_name
                    break
            ep["pass1_s"] = round(time.perf_counter() - pass1_started, 3)
            rec["pass_costs_s"]["pass1"].append(ep["pass1_s"])
            if not ep["died"]:
                rec["no_death_episodes"] += 1
                consecutive_no_death += 1
                rec["consecutive_no_death"] = consecutive_no_death
                rec["consecutive_no_death_max"] = max(
                    int(rec["consecutive_no_death_max"]), consecutive_no_death)
                rec["episodes"].append(ep)
                episode += 1
                log(f"s{seed} w{window} ep{ep['episode']}: survived all "
                    f"{ep['decisions']} decisions, discarded "
                    f"({consecutive_no_death}/{MAX_CONSECUTIVE_NO_DEATH} "
                    f"consecutive no-death, pass1={ep['pass1_s']}s, bombs "
                    f"{ep['bombs_start']}->{state['bombs_remaining']})")
                if consecutive_no_death >= MAX_CONSECUTIVE_NO_DEATH:
                    rec["window_skipped_after"] = int(consecutive_no_death)
                    rec["skip_reason"] = (
                        f"{MAX_CONSECUTIVE_NO_DEATH} consecutive episodes did "
                        f"not die inside {decisions_cap} decisions")
                    log(f"s{seed} w{window}: SKIP the cell after "
                        f"{consecutive_no_death} consecutive no-death episodes "
                        f"(work order rule)")
                    break
                continue
            consecutive_no_death = 0
            rec["consecutive_no_death"] = 0
            # ---- the degenerate-window guard (work order, 2026-11-22) -------
            # a death at decision d < min_death_decision turns the window into
            # decisions 0..d-30, i.e. the first fraction of a second after the
            # prefix, where bomb = 0 survives the horizon by construction.  Such
            # an episode cannot contribute a state; the cell keeps searching.
            if int(ep["death_decision"]) < int(min_death_decision):
                rec["early_death_episodes"] += 1
                rec["death_decisions_rejected"].append(int(ep["death_decision"]))
                ep["rejected"] = "early_death"
                rec["episodes"].append(ep)
                episode += 1
                consecutive_rejected += 1
                rec["consecutive_rejected"] = consecutive_rejected
                rec["consecutive_rejected_max"] = max(
                    int(rec.get("consecutive_rejected_max") or 0),
                    consecutive_rejected)
                log(f"s{seed} w{window} ep{ep['episode']}: DIED at decision "
                    f"{ep['death_decision']} < min_death_decision="
                    f"{min_death_decision}; the window would be decisions "
                    f"0..{max(0, int(ep['death_decision']) - 30)} (the first "
                    f"{(max(0, int(ep['death_decision'])) - 30) * DECISION_FRAMES} "
                    f"frames after the prefix) -> DISCARDED "
                    f"({consecutive_rejected}/{MAX_CONSECUTIVE_NO_DEATH} "
                    f"consecutive rejections)")
                if consecutive_rejected >= MAX_CONSECUTIVE_NO_DEATH:
                    rec["window_skipped_after"] = int(consecutive_rejected)
                    rec["skip_reason"] = (
                        f"{MAX_CONSECUTIVE_NO_DEATH} consecutive episodes died "
                        f"earlier than decision {min_death_decision}, so no "
                        f"window state can be sampled in the mid/late game")
                    log(f"s{seed} w{window}: SKIP the cell after "
                        f"{consecutive_rejected} consecutive early deaths "
                        f"(min_death_decision={min_death_decision})")
                    break
                continue
            consecutive_rejected = 0
            rec["consecutive_rejected"] = 0
            # ---- the sampling ladder (prereg §2.1 S-dead) ------------------
            chosen_name, census = choose_window(trace, int(ep["death_decision"]),
                                               requested=window_name)
            if window_name_used is None:
                window_name_used = chosen_name
                rec["window_auto_choice"] = chosen_name
                log(f"s{seed} w{window}: window auto -> {chosen_name} "
                    f"({window_bounds(chosen_name)[0]}.."
                    f"{window_bounds(chosen_name)[1]} decisions; census " +
                    ", ".join(f"{b['window']}={b['alive_fraction']}"
                              for b in census) + ")")
            if chosen_name != window_name_used:
                log(f"s{seed} w{window} ep{episode}: this episode's own best rung "
                    f"is {chosen_name}, the cell uses {window_name_used} "
                    f"(one window per cell)")
            chosen = sample_decisions(int(ep["death_decision"]),
                                      name=str(window_name_used))
            ep["window_census"] = census
            rec["window_census"] = census
            rec["window_name"] = str(window_name_used)
            rec["window_lo"] = int(window_bounds(str(window_name_used))[0])
            rec["window_hi"] = int(window_bounds(str(window_name_used))[1])
            ep["window"] = {"name": str(window_name_used),
                            "first_decision": (int(chosen[0]) if chosen else None),
                            "last_decision": (int(chosen[-1]) if chosen else None),
                            "decisions": len(chosen),
                            "lo": rec["window_lo"], "hi": rec["window_hi"],
                            "frames": (rec["window_lo"] * DECISION_FRAMES,
                                       rec["window_hi"] * DECISION_FRAMES)}
            rec["death_episodes"] += 1
            death_decisions.append(int(ep["death_decision"]))
            rec["bombs_at_death_counts"][
                str(int(trace[int(ep["death_decision"])]["bombs_before"]))] = (
                rec["bombs_at_death_counts"].get(
                    str(int(trace[int(ep["death_decision"])]["bombs_before"])),
                    0) + 1)
            log(f"s{seed} w{window} ep{ep['episode']}: DIED at decision "
                f"{ep['death_decision']} ({ep['decisions']} decisions, "
                f"{ep['frames']} frames, death player_state="
                f"{ep['death_player_state']}), window {chosen[0] if chosen else None}"
                f"..{chosen[-1] if chosen else None} ({len(chosen)} decisions)")
            if not chosen:
                log(f"s{seed} w{window} ep{episode}: the death at decision "
                    f"{ep['death_decision']} is closer than the window's lower "
                    f"bound; no state can be sampled from it")
                rec["episodes"].append(ep)
                episode += 1
                continue
            # ---------------- pass 2: replay and label ----------------------
            pass2_started = time.perf_counter()
            prefix = parent.walk_prefix(masks, hashes, window)
            parent.park()
            state, _reply = parent.state()
            if int(state["frame"]) != PREFIX_FRAMES:
                raise EarlyBombError(
                    f"w{window} ep{episode}: pass-2 warm RESET left the parent "
                    f"at frame {state['frame']}, expected {PREFIX_FRAMES}")
            if prefix["hash_mismatches"] != 0:
                raise EarlyBombError(
                    f"w{window} ep{episode}: pass-2 prefix walk had "
                    f"{prefix['hash_mismatches']} hash mismatches")
            deaths0 = float(state["deaths"] or 0.0)
            stage0 = int(state["stage"] or 0)
            total = int(chosen[-1]) + 1
            chosen_set = set(chosen)
            for decision in range(total):
                entry = trace[decision]
                state_frame = int(state["frame"])
                hazards = P.parse_hazards_reply(parent.ask("HAZARDS"))
                distances = m800().nearest_bullet_distances(state, hazards)
                ep["replay_checks"] += 1
                rec["replay_checks"] += 1
                if state_frame != int(entry["frame"]):
                    rec["replay_mismatches"] += 1
                    ep["replay_mismatches"] += 1
                    raise EarlyBombError(
                        f"w{window} ep{episode} d{decision}: pass-2 frame "
                        f"{state_frame} != pass-1 frame {entry['frame']}")
                if distances["nearest_all_px"] != entry["nearest_all_px"]:
                    rec["replay_mismatches"] += 1
                    ep["replay_mismatches"] += 1
                    raise EarlyBombError(
                        f"w{window} ep{episode} d{decision}: pass-2 "
                        f"nearest_all_px {distances['nearest_all_px']} != "
                        f"pass-1 {entry['nearest_all_px']}")
                if int(state.get("player_state") or 0) != int(entry["player_state"]):
                    raise EarlyBombError(
                        f"w{window} ep{episode} d{decision}: pass-2 player_state "
                        f"{state.get('player_state')} != pass-1 "
                        f"{entry['player_state']}")
                if decision in chosen_set and rec["states"] < quota:
                    park_reply = parent.park()
                    got_frame = m432().fields(park_reply).get("frame")
                    if got_frame != str(state_frame):
                        raise EarlyBombError(
                            f"w{window} ep{episode} d{decision}: PARK reports "
                            f"frame {got_frame}, the read state is at "
                            f"{state_frame}")
                    block, derived, label_meta = label_state_horizons(
                        parent, state=state, park_reply=park_reply,
                        horizons=HORIZONS)
                    rec["nesting"] = dict(label_meta)
                    if verify_h and rec["nesting_verified"] is None:
                        nesting = verify_horizon_nesting(
                            parent, state=state, park_reply=park_reply,
                            block=block, horizons=HORIZONS)
                        nesting["seed"] = seed
                        nesting["window_index"] = int(window)
                        nesting["episode"] = int(episode)
                        nesting["decision"] = int(decision)
                        nesting["frame"] = state_frame
                        rec["nesting_verified"] = nesting
                        log(f"  nesting verified at d{decision}: "
                            f"{ {k: v['mismatched_actions'] for k, v in nesting['checked'].items()} } "
                            f"mismatches={nesting['mismatches']}")
                    # ---- fail-closed checks on this state's 36 labels ----
                    if len(block["safe"]) != N_ACTIONS:
                        raise EarlyBombError(
                            f"w{window} d{decision}: {len(block['safe'])} labels")
                    if any(value is None for value in block["safe"]):
                        raise EarlyBombError(
                            f"w{window} d{decision}: a label is None")
                    for action in range(N_ACTIONS):
                        if (int(block["safe"][action]) == 1) != (
                                int(block["alive_frames"][action]) == H_LABEL_MAX):
                            raise EarlyBombError(
                                f"w{window} d{decision} a{action}: safe/alive "
                                f"disagree at H={H_LABEL_MAX}")
                    for horizon_h, hblock in derived.items():
                        if len(hblock["safe"]) != N_ACTIONS \
                                or len(hblock["alive_frames"]) != N_ACTIONS:
                            raise EarlyBombError(
                                f"w{window} d{decision}: H={horizon_h} has "
                                f"{len(hblock['safe'])}/"
                                f"{len(hblock['alive_frames'])} labels")
                        for action in range(N_ACTIONS):
                            if (int(hblock["safe"][action]) == 1) != (
                                    int(hblock["alive_frames"][action])
                                    == int(horizon_h)):
                                raise EarlyBombError(
                                    f"w{window} d{decision} a{action}: "
                                    f"H={horizon_h} safe/alive disagree")
                        if int(hblock["max_alive_frames"]) > int(horizon_h):
                            raise EarlyBombError(
                                f"w{window} d{decision}: H={horizon_h} "
                                f"max_alive={hblock['max_alive_frames']}")
                    kids = block["children"]
                    if int(kids["baseline_frame_ok"]) != N_ACTIONS:
                        raise EarlyBombError(
                            f"w{window} d{decision}: only "
                            f"{kids['baseline_frame_ok']}/{N_ACTIONS} children "
                            f"started at the parked parent's frame")
                    if int(kids["spawn_ok"]) != N_ACTIONS \
                            or int(kids["quit_ok"]) != N_ACTIONS:
                        raise EarlyBombError(
                            f"w{window} d{decision}: spawn/quit "
                            f"{kids['spawn_ok']}/{kids['quit_ok']} of "
                            f"{N_ACTIONS}")
                    if int(kids["baseline_state_equal_parent"]) != N_ACTIONS:
                        raise EarlyBombError(
                            f"w{window} d{decision}: only "
                            f"{kids['baseline_state_equal_parent']}/{N_ACTIONS} "
                            f"children baseline states equal the parent's")
                    bombs_before = int(entry["bombs_before"])
                    action_executed = int(entry["action"])
                    decrements = block["bomb_decrement_frame"]
                    spent_amounts: dict[int, int] = {}
                    for action in bomb1_actions:
                        before = int(block["bombs_start"][action])
                        after = int(block["bombs_end"][action])
                        spent_amounts[action] = int(before) - int(after)
                    spent1 = sum(1 for action in bomb1_actions
                                 if spent_amounts[action] == 1)
                    spent2 = sum(1 for action in bomb1_actions
                                 if spent_amounts[action] >= 2)
                    honoured = sum(1 for action in bomb1_actions
                                   if spent_amounts[action] >= 1)
                    noop = sum(1 for action in bomb1_actions
                               if spent_amounts[action] == 0
                               and int(block["alive_frames"][action])
                               == int(block["alive_frames"][action - 1]))
                    anomalies = [
                        split_action(action)[0] for action in bomb1_actions
                        if spent_amounts[action] < 0
                        or (spent_amounts[action] == 0
                            and int(block["alive_frames"][action])
                            != int(block["alive_frames"][action - 1]))]
                    for action in bomb0_actions:
                        if decrements[action] is not None:
                            rec["label_checks"]["bomb0_spent_a_bomb"] += 1
                    rec["label_checks"]["bomb1_press_honoured"] += honoured
                    rec["label_checks"]["bomb1_press_ignored"] += (
                        len(bomb1_actions) - honoured)
                    rec["label_checks"]["bomb1_spent_one"] += spent1
                    rec["label_checks"]["bomb1_spent_two_or_more"] += spent2
                    rec["label_checks"]["bomb1_noop_due_to_invuln"] += noop
                    rec["label_checks"]["bomb1_anomalies"] += len(anomalies)
                    for action in bomb1_actions:
                        amount = str(spent_amounts[action])
                        rec["label_checks"]["bomb1_spent_amounts"][amount] = (
                            rec["label_checks"]["bomb1_spent_amounts"].get(
                                amount, 0) + 1)
                    player_state = int(state.get("player_state") or 0)
                    invuln_name = player_state_name(player_state)
                    rec["label_checks"]["states_by_player_state"][
                        invuln_name] = rec["label_checks"][
                            "states_by_player_state"].get(invuln_name, 0) + 1
                    rec["label_checks"]["labels_36"] += 1
                    if bombs_before == 0:
                        rec["label_checks"]["states_without_a_bomb_left"] += 1
                    feats, _detail = arm_a().build_features(
                        state=state, hazards=hazards,
                        decision_index=int(decision))
                    feats_sha = features_sha256(feats)
                    obs = v450().compact_observation(state, hazards)
                    nearest_rows = nearest_bullets(state, hazards)
                    before_death = int(ep["death_decision"]) - int(decision)
                    base_row = {
                        "schema": SCHEMA_LABEL, "seed": seed,
                        "window": int(window), "episode": int(episode),
                        "decision": int(decision), "frame": state_frame,
                        "H": None, "H_max": int(H_LABEL_MAX),
                        "horizons": [int(h) for h in HORIZONS],
                        "label_source": ("one 36-child run to H = 600; "
                                         "alive(H) = min(alive(600), H)"),
                        "actions": N_ACTIONS,
                        "bombs_remaining": int(bombs_before),
                        # ---- the moved factor, recorded per state ----------
                        "sampling_window": str(window_name_used),
                        "sampling_window_lo": int(rec["window_lo"]),
                        "sampling_window_hi": int(rec["window_hi"]),
                        "death_decision": int(ep["death_decision"]),
                        "decisions_before_death": int(before_death),
                        "frames_before_death": int(before_death
                                                   * DECISION_FRAMES),
                        "is_death_decision": bool(before_death == 0),
                        # ---- prereg §2 P0a: is the player still alive? -----
                        "player_state": int(player_state),
                        "player_state_name": invuln_name,
                        "player_state_alive": bool(invuln_name
                                                   == ALIVE_STATE_NAME),
                        "invulnerability": {
                            "source": ("STATE player_state (u8): 3 = "
                                       "PLAYER_STATE_INVULNERABLE"),
                            "player_state": int(player_state),
                            "player_state_name": invuln_name,
                            "invulnerable": player_state == 3},
                        # ---- the live hazard record ------------------------
                        "nearest_all_px": entry["nearest_all_px"],
                        "nearest_lethal_px": entry["nearest_lethal_px"],
                        "bullet_count": int(hazards.bullet_count),
                        "nearest_bullets": nearest_rows,
                        "nearest_bullets_recorded": len(nearest_rows),
                        "bullets_within_60": int(distances["bullets_within_60"]),
                        "rule_bomb": 0,
                        "rule": BOMB_RULE,
                        # ---- arm A's supervision bit (prereg §1) -----------
                        "aux_rule": ARM_AUX_TARGET_NAME,
                        "behaviour": {
                            "policy": "v4.5.0 `S` round2.pt",
                            "ckpt": str(ckpt), "ckpt_sha256": ckpt_sha,
                            "kind": "stochastic",
                            "generator_seed": generator_seed(seed),
                            "cand": int(entry["cand"]), "bomb": 0,
                            "action": int(action_executed),
                            "bomb_rule": BOMB_RULE,
                            "nearest_all_px": entry["nearest_all_px"],
                            "replay_verified": True},
                        "bombs_before": int(block["bombs_start"][action_executed]),
                        "bombs_after": int(block["bombs_end"][action_executed]),
                        "bomb_spent": bool(
                            int(block["bombs_end"][action_executed])
                            == int(block["bombs_start"][action_executed]) - 1),
                        # ---- P0c: what one press costs, branch by branch ---
                        "bomb1_spent_amount": [spent_amounts[a]
                                               for a in bomb1_actions],
                        "bomb1_spent_honoured": int(honoured),
                        "bomb1_spent_one": int(spent1),
                        "bomb1_spent_two_or_more": int(spent2),
                        "bomb1_noop_count": int(noop),
                        "bomb1_anomaly_cands": anomalies,
                        "bomb1": {
                            "cands": [split_action(a)[0] for a in bomb1_actions],
                            "bombs_before": [int(block["bombs_start"][a])
                                             for a in bomb1_actions],
                            "bombs_after": [int(block["bombs_end"][a])
                                            for a in bomb1_actions],
                            "spent_amount": [spent_amounts[a]
                                             for a in bomb1_actions],
                            "alive_frames": [int(block["alive_frames"][a])
                                             for a in bomb1_actions],
                            "alive_frames_bomb0": [
                                int(block["alive_frames"][a - 1])
                                for a in bomb1_actions],
                            "decrement_frame": [decrements[a]
                                                for a in bomb1_actions],
                            "spent_fraction": round(honoured
                                                    / len(bomb1_actions), 6)},
                        "bombs_used": block["bombs_used"],
                        "bombs_start": block["bombs_start"],
                        "bombs_end": block["bombs_end"],
                        "bomb_decrement_frame": block["bomb_decrement_frame"],
                        "decision_bucket": v860().decision_bucket(decision),
                        "features_sha256_k12": feats_sha,
                        "park_reply": block["park_reply"],
                        "child_replies_digest": block["child_replies_digest"],
                        "child_reply_lines": block["child_reply_lines"],
                        "children": block["children"],
                        "label_wall_s": block["label_wall_s"],
                        "obs": obs,
                    }
                    for horizon_h in HORIZONS:
                        hblock = derived[int(horizon_h)]
                        row = dict(base_row)
                        row["H"] = int(horizon_h)
                        row["safe"] = hblock["safe"]
                        row["alive_frames"] = hblock["alive_frames"]
                        row["max_alive_frames"] = int(hblock["max_alive_frames"])
                        row["safe_actions"] = int(hblock["safe_actions"])
                        row["critical"] = bool(
                            int(hblock["max_alive_frames"]) < int(horizon_h))
                        row["best_bomb0"] = int(hblock["best_bomb0"])
                        row["best_bomb1"] = int(hblock["best_bomb1"])
                        row["delta"] = int(hblock["best_bomb1"]
                                           - hblock["best_bomb0"])
                        row["aux_best_bomb"] = aux_best_bomb_target(row)
                        row["terminal_counts_at_H"] = hblock["terminals"]
                        write_jsonl_line(handle, row)
                        rec["states_by_H"][str(int(horizon_h))] = (
                            rec["states_by_H"].get(str(int(horizon_h)), 0) + 1)
                    handle.flush()
                    rec["states"] += 1
                    ep["states_labelled"] += 1
                    rec["state_costs_s"].append(float(block["label_wall_s"]))
                    rec["decisions_before_death_counts"][str(before_death)] = (
                        rec["decisions_before_death_counts"].get(
                            str(before_death), 0) + 1)
                    bucket = ("0" if before_death == 0 else
                              "1-14" if before_death < 15 else
                              "15-29" if before_death < 30 else
                              "30-59" if before_death < 60 else
                              "60-119" if before_death < 120 else "120+")
                    slot = rec["player_state_by_before_death"].setdefault(
                        bucket, {})
                    slot[invuln_name] = slot.get(invuln_name, 0) + 1
                    if rec["states"] % 10 == 0:
                        log(f"  s{seed} w{window} ep{episode} d{decision}: "
                            f"{rec['states']}/{quota} states, "
                            f"player_state={rec['label_checks']['states_by_player_state']}, "
                            f"mean_label="
                            f"{statistics.fmean(rec['state_costs_s']):.3f}s")
                    if rec["states"] >= quota:
                        break
                if decision >= int(chosen[-1]):
                    break
                first_mask, rest_mask = v830().action_masks(entry["action"])
                for frame_index in range(DECISION_FRAMES):
                    step = parent.step(first_mask if frame_index == 0
                                       else rest_mask)
                    if step["advanced"] == 1:
                        ep["frames"] += 1
                    reading, _reply = parent.state()
                    terminal = cf._terminal_of(
                        reading, deaths_start=deaths0, stage0=stage0,
                        hash_hex=(step["hash"] or ""),
                        frame=int(reading["frame"]), exited=step["exited"],
                        game_over=step["game_over"])
                    state = reading
                    if terminal is not None:
                        break
            ep["pass2_s"] = round(time.perf_counter() - pass2_started, 3)
            rec["pass_costs_s"]["pass2"].append(ep["pass2_s"])
            ep["bombs_end"] = int(state["bombs_remaining"] or 0)
            ep["bombs_used"] = int(ep["bombs_start"]) - int(ep["bombs_end"])
            rec["episodes"].append(ep)
            episode += 1
            log(f"s{seed} w{window} ep{ep['episode']}: DIED d"
                f"{ep['death_decision']}, {ep['decisions']} decisions, "
                f"{ep['states_labelled']} states labelled, replay_checks="
                f"{ep['replay_checks']}, pass1={ep['pass1_s']}s, "
                f"pass2={ep['pass2_s']}s")
            if rec["states"] >= quota:
                break
        parent.close()
        parents.remove(parent)
        if rec["states"] >= quota:
            rec["status"] = "complete"
        elif rec["window_skipped_after"] is not None:
            rec["status"] = "skipped"
        elif rec["states"] > 0:
            rec["status"] = "short"
        else:
            rec["status"] = "empty"
        if rec["states"] and rec["label_checks"]["bomb1_press_honoured"] <= 0 \
                and rec["label_checks"]["states_without_a_bomb_left"] \
                < rec["states"]:
            raise EarlyBombError(
                f"w{window}: no bomb=1 label ever decremented the counter in "
                f"{rec['states']} states although some had a bomb left - the "
                f"bomb path is not working")
        if rec["label_checks"]["bomb0_spent_a_bomb"]:
            log(f"w{window}: WARNING "
                f"{rec['label_checks']['bomb0_spent_a_bomb']} bomb=0 label(s) "
                f"spent a bomb (counted as an anomaly, cell kept)")
    except Exception as exc:                                          # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["status"] = "failed"
        log(f"CELL FAILED s{seed} w{window}: {rec['error']}")
    finally:
        handle.close()
        for parent in parents:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        rec["wall_s"] = round(time.perf_counter() - started, 3)
        rec["label_cost_s"] = quantiles(rec["state_costs_s"])
        del rec["state_costs_s"]
        rec["pass1_cost_s"] = quantiles(rec["pass_costs_s"]["pass1"])
        rec["pass2_cost_s"] = quantiles(rec["pass_costs_s"]["pass2"])
        del rec["pass_costs_s"]
        rec["labels_total"] = int(rec["states"]) * N_ACTIONS
        rec["label_rows"] = int(rec["states"]) * len(HORIZONS)
        rec["death_decision_quantiles"] = quantiles(death_decisions)
        if rec["status"] in ("complete", "short", "skipped", "empty"):
            if labels.exists():
                labels.unlink()
            partial.rename(labels)
            rec["labels_sha256"] = sha256_file(labels)
            rec["labels_bytes"] = labels.stat().st_size
        write_json(summary_path, rec)
    log(f"cell s{seed} w{window}: status={rec['status']} states={rec['states']} "
        f"rows={rec.get('label_rows')} window={rec.get('window_name')} "
        f"death_episodes={rec['death_episodes']} no_death="
        f"{rec['no_death_episodes']} "
        f"player_state={rec['label_checks']['states_by_player_state']} "
        f"spent={rec['label_checks']['bomb1_spent_amounts']} "
        f"wall={rec['wall_s']}s -> {labels}")
    return rec


# --------------------------------------------------------------------------- #
# mode: smoke (one death episode, the round's own gate)
# --------------------------------------------------------------------------- #
def mode_smoke(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    windows = parse_windows(args.windows)
    if len(windows) != 1:
        raise EarlyBombError(f"--mode smoke runs ONE window, got {windows}")
    window = int(windows[0])
    smoke_dir = root / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    labels = smoke_dir / f"labels_smoke_seed{seed}_w{window}.jsonl"
    summary_path = smoke_dir / f"smoke_seed{seed}_w{window}.json"
    rec = run_cell(seed=seed, window=window, quota=10_000, max_episodes=12,
                   window_name=str(args.window), decisions_cap=DECISIONS,
                   engine=args.engine, data_source=args.data_source, root=root,
                   tag=args.tag or f"v8j0-smoke-s{seed}-w{window}",
                   labels=labels, summary_path=summary_path,
                   verify_h=True, stop_after_death_episodes=1,
                   schema=SCHEMA_SMOKE, smoke=True,
                   min_death_decision=int(args.min_death_decision))
    if rec["status"] == "failed":
        log(f"smoke FAILED: {rec['error']}")
        return 4
    rows = read_jsonl(labels) if labels.is_file() else []
    states = state_rows(rows) if rows else []
    summary = p0_summary(states) if states else {"states": 0}
    verdict = verdict_of(summary) if states else {"name": "no-states",
                                                  "train_horizon": None}
    # the death-detection lag, measured on pass 1's per-decision trace
    trace = []
    if rec["episodes"]:
        trace = rec["episodes"][0].get("player_state_trace") or []
    lag = None
    for entry in reversed(trace):
        if entry[1] != 2:            # 2 = DYING
            lag = {"last_non_dying_decision": int(entry[0]),
                   "last_non_dying_player_state": int(entry[1]),
                   "last_non_dying_name": (list(rec["label_checks"]
                                                ["states_by_player_state"]))}
            break
    death_decision = (rec["episodes"][0].get("death_decision")
                      if rec["episodes"] else None)
    before_death = [entry["decisions_before_death"] for entry in states]
    decisions_range = {
        "n": len(before_death),
        "decisions_before_death_min": (min(before_death) if before_death else None),
        "decisions_before_death_max": (max(before_death) if before_death else None),
        "decisions_before_death_quantiles": quantiles(before_death),
        "frames_before_death_min": (min(before_death) * DECISION_FRAMES
                                    if before_death else None),
        "frames_before_death_max": (max(before_death) * DECISION_FRAMES
                                    if before_death else None),
        "sampled_decisions_min": (min(entry["decision"] for entry in states)
                                  if states else None),
        "sampled_decisions_max": (max(entry["decision"] for entry in states)
                                  if states else None),
        "death_decisions": [int(ep["death_decision"]) for ep in rec["episodes"]
                            if ep.get("death_decision") is not None],
        "episodes_run": len(rec["episodes"]),
        "episodes_rejected_early_death": rec["early_death_episodes"],
        "min_death_decision": rec["min_death_decision"],
    }
    lag_block = {"death_decision": death_decision, "trace_decisions": len(trace),
                 "probe": lag}
    if death_decision is not None:
        last_alive = [entry[0] for entry in trace if entry[1] == 0]
        lag_block["last_ALIVE_decision"] = (int(last_alive[-1]) if last_alive
                                            else None)
        lag_block["frames_of_DYING_before_detection"] = (
            int(death_decision) - int(last_alive[-1])
            if last_alive else None)
        lag_block["note"] = ("frames_of_DYING_before_detection = death_decision "
                             "- last decision whose player_state was ALIVE "
                             "(decisions), i.e. the death-detection lag the "
                             "prereg §2.1 S-dead asks to record")
    smoke_rec: dict[str, Any] = {
        "schema": SCHEMA_SMOKE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "requirement": ("prereg §2's P0 screen before any training: one death "
                        "episode, the sampled window's player_state / bombs / "
                        "bomb=1 spent distributions and the three horizons' "
                        "critical / diff"),
        "seed": seed, "window": window,
        "window_requested": str(args.window),
        "window_name": rec.get("window_name"),
        "window_census": rec.get("window_census"),
        "player_state_by_before_death": rec.get("player_state_by_before_death"),
        "player_state_counts_trajectory": rec.get("player_state_counts"),
        "death_detection_lag": lag_block,
        "decisions_before_death_range": decisions_range,
        "early_death_episodes": rec["early_death_episodes"],
        "min_death_decision": rec["min_death_decision"],
        "states": len(states),
        "p0": summary,
        "verdict": verdict,
        "label_checks": rec["label_checks"],
        "bombs_remaining_counts": rec["bombs_remaining_counts"],
        "decisions_before_death_counts": rec["decisions_before_death_counts"],
        "nesting_verified": rec.get("nesting_verified"),
        "cell": {"status": rec["status"], "wall_s": rec["wall_s"],
                 "error": rec.get("error"),
                 "labels_sha256": rec.get("labels_sha256"),
                 "pass1_cost_s": rec.get("pass1_cost_s"),
                 "pass2_cost_s": rec.get("pass2_cost_s"),
                 "label_cost_s": rec.get("label_cost_s")},
        "labels_path": str(labels),
    }
    write_json(root / "smoke.json", smoke_rec)
    # ---------------- the report ------------------------------------------
    log("=" * 78)
    log(f"SMOKE s{seed} w{window} window_requested={args.window} "
        f"-> used={rec.get('window_name')} "
        f"({rec.get('window_lo')}..{rec.get('window_hi')} decisions = "
        f"{(rec.get('window_lo') or 0) * DECISION_FRAMES}-"
        f"{(rec.get('window_hi') or 0) * DECISION_FRAMES} frames)")
    log(f"  death at decision {death_decision}; player_state over the whole "
        f"trajectory: {rec.get('player_state_counts')}")
    for block in (rec.get("window_census") or []):
        log(f"  census {block['window']:5s} [{block['lo']:3d},{block['hi']:3d}] "
            f"decisions={block['decisions']:3d} alive="
            f"{block['alive']}/{block['decisions']} = "
            f"{block['alive_fraction']} "
            f"{'PASS' if block['alive_met'] else 'fail'} "
            f"states={block['player_state_counts']}")
    log(f"  death-detection lag: {lag_block}")
    log(f"  sampled decisions_before_death: {decisions_range['decisions_before_death_min']}"
        f"..{decisions_range['decisions_before_death_max']} decisions = "
        f"{decisions_range['frames_before_death_min']}.."
        f"{decisions_range['frames_before_death_max']} frames "
        f"(median {decisions_range['decisions_before_death_quantiles'].get('median')}); "
        f"sampled decisions {decisions_range['sampled_decisions_min']}.."
        f"{decisions_range['sampled_decisions_max']}; "
        f"death decisions {decisions_range['death_decisions']}; "
        f"min_death_decision={decisions_range['min_death_decision']} "
        f"(rejected early deaths: {decisions_range['episodes_rejected_early_death']})")
    log(f"  sampled decisions_before_death histogram: "
        f"{rec.get('decisions_before_death_counts')}")
    log(f"  sampled states: {len(states)}")
    base = summary.get("state_level") or {}
    log(f"  P0a ALIVE          {base.get('p0a', {}).get('alive')}/"
        f"{base.get('p0a', {}).get('states', len(states))} = "
        f"{base.get('p0a', {}).get('fraction')} (gate {P0A_ALIVE_MIN:.0%}) -> "
        f"{'PASS' if base.get('p0a', {}).get('met') else 'FAIL'} "
        f"{base.get('p0a', {}).get('player_state_counts')}")
    log(f"  P0b bombs > 0      {base.get('p0b', {}).get('armed')}/"
        f"{len(states)} = {base.get('p0b', {}).get('fraction')} (gate "
        f"{P0B_ARMED_MIN:.0%}) -> "
        f"{'PASS' if base.get('p0b', {}).get('met') else 'FAIL'} "
        f"{base.get('p0b', {}).get('bombs_remaining_counts')}")
    log(f"  P0c spent == 1     {base.get('p0c', {}).get('bomb1_spent_one')}/"
        f"{base.get('p0c', {}).get('bomb1_honoured')} honoured = "
        f"{base.get('p0c', {}).get('honoured_spent1_fraction')} (gate "
        f"{P0C_SPENT1_MIN:.0%}) -> "
        f"{'PASS' if base.get('p0c', {}).get('met') else 'FAIL'} "
        f"amounts={base.get('p0c', {}).get('bomb1_spent_amount_counts')} "
        f"noop={base.get('p0c', {}).get('bomb1_noop_due_to_invuln')}")
    for horizon in HORIZONS:
        block = summary["by_H"][str(int(horizon))]
        log(f"  P0d H={int(horizon):3d}         {block['diff_states']}/"
            f"{block['states']} = {block['fraction']} (gate {P0D_DIFF_MIN:.0%}) "
            f"-> {'PASS' if block['met'] else 'fail'}; critical="
            f"{block['critical']} ({block['critical_fraction']}), equal="
            f"{block['equal_states']}, worse={block['worse_states']}, delta="
            f"{block['delta_quantiles']}")
    log(f"  verdict: {verdict['name']} — {verdict['rule']}")
    log("=" * 78)
    all_four = bool(summary.get("all_four_met"))
    return 0 if all_four else 4


# --------------------------------------------------------------------------- #
# mode: collect
# --------------------------------------------------------------------------- #
def mode_collect(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    windows = parse_windows(args.windows)
    rc = 0
    for window in windows:
        rec = run_cell(seed=seed, window=int(window), quota=int(args.quota),
                       max_episodes=int(args.max_episodes),
                       window_name=str(args.window), decisions_cap=DECISIONS,
                       engine=args.engine, data_source=args.data_source,
                       root=root,
                       tag=args.tag or f"v8j0-collect-s{seed}-w{window}",
                       labels=(pathlib.Path(args.labels_out)
                               if args.labels_out and len(windows) == 1 else None),
                       summary_path=(pathlib.Path(args.summary_out)
                                     if args.summary_out and len(windows) == 1
                                     else None),
                       verify_h=bool(int(getattr(args, "verify_h", 0) or 0)),
                       stop_after_death_episodes=None,
                       schema=SCHEMA_COLLECT, smoke=False,
                       min_death_decision=int(args.min_death_decision))
        if rec["status"] == "failed":
            rc = 4
    return rc


# --------------------------------------------------------------------------- #
# mode: stats (pool + the P0 screen)
# --------------------------------------------------------------------------- #
def mode_stats(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    cells = sorted((root / "labels").glob("seed*_w*.jsonl"))
    if not cells:
        raise EarlyBombError(f"no cell label files under {root / 'labels'}")
    rows: list[dict[str, Any]] = []
    per_cell: list[dict[str, Any]] = []
    for path in cells:
        cell_rows = read_jsonl(path)
        rows.extend(cell_rows)
        per_cell.append({
            "file": str(path), "sha256": sha256_file(path),
            "bytes": path.stat().st_size, "rows": len(cell_rows),
            "states": len({(int(row["seed"]), int(row["window"]),
                            int(row["episode"]), int(row["decision"]))
                           for row in cell_rows}),
            "seed": int(cell_rows[0]["seed"]) if cell_rows else None,
            "window": int(cell_rows[0]["window"]) if cell_rows else None,
            "sampling_window": (cell_rows[0].get("sampling_window")
                                if cell_rows else None),
            "episodes": len({int(row["episode"]) for row in cell_rows}),
            "by_H": {str(h): sum(1 for row in cell_rows if int(row["H"]) == int(h))
                     for h in HORIZONS},
        })
    pooled = pooled_labels_path(root)
    with pooled.open("w", encoding="utf-8") as handle:
        for row in rows:
            write_jsonl_line(handle, row)
    states = state_rows(rows)
    pool = choose_pool(states)
    summary = pool["p0"]
    verdict = pool["verdict"]
    windows_seen = sorted({str(cell["sampling_window"]) for cell in per_cell
                           if cell.get("sampling_window")})
    if len(windows_seen) > 1:
        raise EarlyBombError(f"the pooled cells used more than one sampling "
                            f"window: {windows_seen}; the P0 screen would not "
                            f"be comparable")
    by_segment = {}
    for bombs in BOMB_SEGMENTS:
        subset = [entry for entry in states
                  if entry["bombs_remaining"] == bombs]
        by_segment[str(bombs)] = (p0_block(subset) if subset else {"states": 0})
    by_player_state = {}
    for name in sorted({entry["player_state_name"] for entry in states}):
        subset = [entry for entry in states if entry["player_state_name"] == name]
        by_player_state[name] = p0_block(subset) if subset else {"states": 0}
    # ---- the training pool: the rows the verdict's filter selects -----------
    filter_name = pool["filter"]
    alive_pool = pool["filtered_states"]
    chosen_rows = ([row for row in rows if bool(row.get("player_state_alive"))]
                   if filter_name == ALIVE_STATE_NAME else rows)
    alive_labels = root / "labels_earlybomb_alive.jsonl"
    with alive_labels.open("w", encoding="utf-8") as handle:
        for row in chosen_rows:
            write_jsonl_line(handle, row)
    by_before_death = {}
    for label, lo, hi in (("30-39", 30, 39), ("40-49", 40, 49), ("50-60", 50, 60),
                          ("61-90", 61, 90), ("91-120", 91, 120),
                          ("121+", 121, 10 ** 9)):
        subset = [entry for entry in states
                  if lo <= entry["decisions_before_death"] <= hi]
        if subset:
            by_before_death[label] = p0_block(subset)
    cell_blocks: list[dict[str, Any]] = []
    bombs_at_death: dict[str, int] = {}
    for path in sorted((root / "collect").glob("seed*_w*.json")):
        cell = read_json(path)
        cell_blocks.append({
            "file": str(path), "seed": cell.get("seed"),
            "window": cell.get("window"),
            "sampling_window": cell.get("window_name"),
            "status": cell.get("status"), "states": cell.get("states"),
            "rows": cell.get("label_rows"),
            "death_episodes": cell.get("death_episodes"),
            "no_death_episodes": cell.get("no_death_episodes"),
            "window_skipped_after": cell.get("window_skipped_after"),
            "skip_reason": cell.get("skip_reason"),
            "decisions_cap": cell.get("decisions_cap"),
            "min_death_decision": cell.get("min_death_decision"),
            "early_death_episodes": cell.get("early_death_episodes"),
            "death_decisions_rejected": cell.get("death_decisions_rejected"),
            "states_by_H": cell.get("states_by_H"),
            "bombs_at_death_counts": cell.get("bombs_at_death_counts"),
            "death_decision_quantiles": cell.get("death_decision_quantiles"),
            "label_cost_s": cell.get("label_cost_s"),
            "nesting": cell.get("nesting_verified"),
            "wall_s": cell.get("wall_s"),
            "labels_sha256": cell.get("labels_sha256"),
            "labels_bytes": cell.get("labels_bytes"),
            "error": cell.get("error"),
            "label_checks": {key: (cell.get("label_checks") or {}).get(key)
                             for key in ("labels_36", "bomb1_press_honoured",
                                         "bomb1_spent_one",
                                         "bomb1_spent_two_or_more",
                                         "bomb1_spent_amounts",
                                         "bomb1_noop_due_to_invuln",
                                         "bomb0_spent_a_bomb",
                                         "states_without_a_bomb_left",
                                         "states_by_player_state")},
        })
        for key, count in (cell.get("bombs_at_death_counts") or {}).items():
            bombs_at_death[str(key)] = bombs_at_death.get(str(key), 0) + int(count)
    merged: dict[str, Any] = {}
    for cell in cell_blocks:
        for key in ("labels_36", "bomb1_press_honoured", "bomb1_spent_one",
                    "bomb1_spent_two_or_more", "bomb1_noop_due_to_invuln",
                    "bomb0_spent_a_bomb", "states_without_a_bomb_left"):
            merged[key] = merged.get(key, 0) + int(cell["label_checks"].get(key) or 0)
        for key in ("death_episodes", "no_death_episodes"):
            merged[key] = merged.get(key, 0) + int(cell.get(key) or 0)
        for key, count in ((cell["label_checks"].get("bomb1_spent_amounts")
                            or {}).items()):
            merged.setdefault("bomb1_spent_amounts", {})
            merged["bomb1_spent_amounts"][key] = (
                merged["bomb1_spent_amounts"].get(key, 0) + int(count))
        for key, count in ((cell["label_checks"].get("states_by_player_state")
                            or {}).items()):
            merged.setdefault("states_by_player_state", {})
            merged["states_by_player_state"][key] = (
                merged["states_by_player_state"].get(key, 0) + int(count))
    failed_cells = [cell for cell in cell_blocks if cell.get("status") == "failed"]
    payload: dict[str, Any] = {
        "schema": SCHEMA_STATS, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL,
        "requirement": ("prereg §2 P0a (ALIVE >= 90 %), P0b (bombs > 0 >= 50 %), "
                        "P0c (of the honoured bomb=1 branches, spent == 1 >= "
                        "90 %), P0d (best bomb1 > best bomb0 >= 20 % at some "
                        "horizon)"),
        "sampling": {
            "kind": "early-death-adjacent",
            "window_names": windows_seen,
            "window_bounds": {name: {"lo": WINDOW_SPECS[name][0],
                                     "hi": WINDOW_SPECS[name][1],
                                     "frames": (WINDOW_SPECS[name][0]
                                                * DECISION_FRAMES,
                                                WINDOW_SPECS[name][1]
                                                * DECISION_FRAMES)}
                              for name in windows_seen},
            "window_source": WINDOW_SOURCE,
            "behaviour": ("v4.5.0 `S` stochastic + never a bomb (--bomb-rule "
                          "none), rolled until death, cap "
                          f"{DECISIONS} decisions; an episode that does not die "
                          "is discarded"),
            "windows": list(SAMPLE_WINDOWS),
            "max_consecutive_no_death": MAX_CONSECUTIVE_NO_DEATH,
            "horizons": list(HORIZONS),
            "label_source": ("one 36-child run to H = 600; alive(H) = "
                             "min(alive(600), H)"),
        },
        "cells": per_cell, "cell_count": len(per_cell),
        "collect_cells": cell_blocks, "failed_cells": len(failed_cells),
        "states": len(states), "label_rows": len(rows),
        "labels": len(states) * N_ACTIONS,
        "labels_per_horizon": len(states) * N_ACTIONS,
        "episodes": len({(entry["seed"], entry["window"], entry["episode"])
                         for entry in states}),
        "definitions": {
            "p0a": "player_state == ALIVE at the sampled decision",
            "p0b": "bombs_remaining > 0 at the sampled decision",
            "p0c": ("of the 18 bomb=1 branches the engine honoured (the counter "
                    "decremented), the share whose spent_amount == 1"),
            "p0d": ("max(alive_frames over the 18 bomb=1 branches) > "
                    "max(alive_frames over the 18 bomb=0 branches) of the same "
                    "state at the same H"),
            "p0a_alive_min": P0A_ALIVE_MIN, "p0b_armed_min": P0B_ARMED_MIN,
            "p0c_spent1_min": P0C_SPENT1_MIN, "p0d_diff_min": P0D_DIFF_MIN},
        "p0": summary,
        "p0_all": pool["p0_all"],
        "p0_alive": pool["p0_alive"],
        "filter": {"name": filter_name,
                   "definition": ("player_state == ALIVE (prereg §2.1 S-2bomb)"
                                  if filter_name == ALIVE_STATE_NAME
                                  else "none (the whole window)"),
                   "reason": verdict.get("filter_reason"),
                   "states_all": pool["states_all"],
                   "states_alive": pool["states_alive"],
                   "alive_fraction": (round(pool["states_alive"]
                                            / pool["states_all"], 6)
                                      if pool["states_all"] else None),
                   "chosen_states": len(alive_pool)
                   if filter_name != "none" else pool["states_all"]},
        "by_player_state": by_player_state,
        "by_bombs_remaining": by_segment,
        "by_decisions_before_death": by_before_death,
        "player_state_counts": count_map([entry["player_state_name"]
                                          for entry in states]),
        "death_decision_counts": count_map([entry["decision"]
                                            + entry["decisions_before_death"]
                                            for entry in states]),
        "min_death_decisions_used": sorted({
            int(cell.get("min_death_decision") or 0) for cell in cell_blocks}),
        "early_death_episodes": sum(int(cell.get("early_death_episodes") or 0)
                                    for cell in cell_blocks),
        "bombs_remaining_counts": count_map([entry["bombs_remaining"]
                                             for entry in states]),
        "decisions_before_death_counts": count_map(
            [entry["decisions_before_death"] for entry in states]),
        "verdict": verdict,
        "bombs_at_death": {
            "counts": bombs_at_death,
            "counts_sorted": dict(sorted(bombs_at_death.items(), reverse=True)),
            "episodes": sum(bombs_at_death.values()),
            "requirement": ("with --bomb-rule none the player must die with all "
                            "three bombs, so the only key is '3'")},
        "label_checks": merged,
        "labels_file": {"path": str(pooled), "sha256": sha256_file(pooled),
                        "bytes": pooled.stat().st_size, "rows": len(rows)},
        "training_pool_file": {"path": str(alive_labels),
                               "sha256": sha256_file(alive_labels),
                               "bytes": alive_labels.stat().st_size,
                               "rows": len(chosen_rows),
                               "filter": filter_name},
        "reference_v817": BASELINE_V817,
    }
    if V817_STATS.is_file():
        payload["reference_v817"]["own_stats"] = {
            "path": str(V817_STATS), "sha256": sha256_file(V817_STATS)}
    write_json(pooled_stats_path(root), payload)
    base = summary["state_level"]
    full = pool["p0_all"]["state_level"]
    chosen_n = (len(alive_pool) if filter_name == ALIVE_STATE_NAME else len(states))
    log(f"stats: {len(states)} states, {len(rows)} rows from {len(per_cell)} "
        f"cells, windows={windows_seen}")
    log(f"  whole window: P0a {full['p0a']['alive']}/{len(states)} = "
        f"{full['p0a']['fraction']} ({full['p0a']['player_state_counts']}), "
        f"P0c {full['p0c']['bomb1_spent_one']}/{full['p0c']['bomb1_honoured']} "
        f"= {full['p0c']['honoured_spent1_fraction']} "
        f"({full['p0c']['bomb1_spent_amount_counts']})")
    log(f"  filter = {filter_name} ({verdict.get('filter_reason')}); "
        f"chosen pool = {chosen_n} states")
    log(f"  P0a ALIVE          {base['p0a']['alive']}/{chosen_n} = "
        f"{base['p0a']['fraction']} -> "
        f"{'PASS' if base['p0a']['met'] else 'FAIL'} "
        f"{base['p0a']['player_state_counts']}")
    log(f"  P0b bombs > 0      {base['p0b']['armed']}/{chosen_n} = "
        f"{base['p0b']['fraction']} -> "
        f"{'PASS' if base['p0b']['met'] else 'FAIL'} "
        f"{base['p0b']['bombs_remaining_counts']}")
    log(f"  P0c spent == 1     {base['p0c']['bomb1_spent_one']}/"
        f"{base['p0c']['bomb1_honoured']} = "
        f"{base['p0c']['honoured_spent1_fraction']} -> "
        f"{'PASS' if base['p0c']['met'] else 'FAIL'} "
        f"{base['p0c']['bomb1_spent_amount_counts']}")
    for horizon in HORIZONS:
        block = summary["by_H"][str(int(horizon))]
        log(f"  P0d H={int(horizon):3d}         {block['diff_states']}/"
            f"{block['states']} = {block['fraction']} -> "
            f"{'PASS' if block['met'] else 'fail'}; critical="
            f"{block['critical']} ({block['critical_fraction']})")
    log(f"  verdict: {verdict['name']} train_horizon={verdict['train_horizon']}")
    return 0 if verdict["name"] == "train" else 4


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def bomb_marginal_loss(logits: Any, target: Any) -> Any:
    """BCE of the **bomb marginal** against arm A's aux bit.

    The probability of *some* bomb action is the softmax mass on the 18 bomb=1
    outputs, so the aux term is a two-class cross-entropy on

        logit = logsumexp(logits[bomb = 1]) - logsumexp(logits[bomb = 0])

    against ``target``.  It never replaces the 36-way BCE; it is added with
    weight 0.5 (prereg §1, arm A only).
    """

    import torch

    bomb1 = [a for a in range(N_ACTIONS) if split_action(a)[1] == 1]
    bomb0 = [a for a in range(N_ACTIONS) if split_action(a)[1] == 0]
    margin = (torch.logsumexp(logits[:, bomb1], dim=1)
              - torch.logsumexp(logits[:, bomb0], dim=1))
    return torch.nn.functional.binary_cross_entropy_with_logits(margin, target)


def load_pooled_rows(root: pathlib.Path) -> tuple[list[dict[str, Any]],
                                                 dict[str, Any]]:
    pooled = pooled_labels_path(root)
    if not pooled.is_file():
        raise EarlyBombError(f"no pooled labels: {pooled}")
    rows = read_jsonl(pooled)
    stats_rec = (read_json(pooled_stats_path(root))
                 if pooled_stats_path(root).is_file() else {})
    digest = sha256_file(pooled)
    info = {"path": str(pooled), "sha256": digest,
            "bytes": pooled.stat().st_size, "rows": len(rows),
            "recorded_sha256": (stats_rec.get("labels_file") or {}).get("sha256")}
    info["sha256_match"] = digest == info["recorded_sha256"]
    if not info["sha256_match"]:
        raise EarlyBombError(f"{pooled} sha256 {digest[:16]} != the recorded "
                             f"{str(info['recorded_sha256'])[:16]}")
    return rows, info


def mode_train(args: argparse.Namespace) -> int:
    import torch

    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    arm = str(args.arm).upper()
    if seed not in TRAIN_SEEDS:
        raise EarlyBombError(f"--seed must be one of {TRAIN_SEEDS}, got {seed}")
    if arm not in ARMS:
        raise EarlyBombError(f"--arm must be one of {ARMS}, got {arm}")
    rows, info = load_pooled_rows(root)
    if not rows:
        raise EarlyBombError("no labelled rows")
    stats_rec = read_json(pooled_stats_path(root))
    verdict = stats_rec.get("verdict") or {}
    if verdict.get("name") != "train":
        raise EarlyBombError(f"the P0 screen did not pass (verdict="
                             f"{verdict.get('name')}); prereg §2.1 - no training")
    horizon = int(args.horizon or 0)
    if not horizon:
        horizon = int(verdict.get("train_horizon") or 0)
    if horizon not in HORIZONS:
        raise EarlyBombError(f"no horizon reached the P0d gate "
                             f"(verdict={verdict}); no training")
    p0 = stats_rec.get("p0") or {}
    if not (p0.get("p0a_met") and p0.get("p0b_met") and p0.get("p0c_met")):
        raise EarlyBombError(f"P0a/P0b/P0c did not pass ({p0.get('state_level')}): "
                             f"no training")
    rows = [row for row in rows if int(row["H"]) == int(horizon)]
    filter_name = str((stats_rec.get("filter") or {}).get("name") or "none")
    if filter_name == ALIVE_STATE_NAME:
        before = len(rows)
        rows = [row for row in rows if bool(row.get("player_state_alive"))]
        log(f"train: prereg §2.1 S-2bomb filter player_state == ALIVE: "
            f"{len(rows)}/{before} H={horizon} rows kept")
    if not rows:
        raise EarlyBombError(f"no H={horizon} rows in the pooled labels")
    log(f"train arm={arm} seed={seed} H={horizon}: {len(rows)} states "
        f"(horizons passing P0d: {p0.get('p0d_horizons_met')}; the smallest is "
        f"used)")
    feats, labels, array_info = v830().build_arrays(rows)
    aux_weight = float(ARM_AUX_WEIGHT[arm])
    targets = torch.tensor([float(aux_best_bomb_target(row)) for row in rows],
                           dtype=torch.float32)
    bombs_counts = count_map([row["bombs_remaining"] for row in rows])
    log(f"train arm={arm} seed={seed}: {len(rows)} states x {N_ACTIONS} actions, "
        f"dim={FEATURE_DIM}, bombs={bombs_counts}, best-bomb positives="
        f"{int(targets.sum())}, aux_weight={aux_weight}")

    x = torch.from_numpy(feats)
    y = torch.from_numpy(labels)
    policy = v830().policy36(seed=seed)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    import numpy as np

    rng = np.random.default_rng(seed)
    epoch_losses: list[float] = []
    epoch_aux: list[float] = []
    batch = int(args.batch)
    started = time.perf_counter()
    for _epoch in range(int(EPOCHS)):
        perm = torch.from_numpy(rng.permutation(len(rows)))
        total = 0.0
        total_aux = 0.0
        policy.train()
        for start in range(0, len(rows), batch):
            index = perm[start:start + batch]
            logits = policy(x.index_select(0, index))
            loss = loss_fn(logits, y.index_select(0, index))
            aux = torch.zeros(())
            if aux_weight > 0.0:
                aux = bomb_marginal_loss(logits, targets.index_select(0, index))
                loss = loss + aux_weight * aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(index.numel())
            total_aux += float(aux.detach()) * int(index.numel())
        epoch_losses.append(total / len(rows))
        epoch_aux.append(total_aux / len(rows))
    wall = round(time.perf_counter() - started, 3)

    policy.eval()
    with torch.no_grad():
        logits = policy(x)
        final_bce = float(torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y))
        final_aux = (float(bomb_marginal_loss(logits, targets))
                     if aux_weight > 0.0 else None)
        predicted = (torch.sigmoid(logits) >= 0.5).float()
        overall_acc = float((predicted == y).float().mean())
        per_action_acc = [float((predicted[:, action] == y[:, action]).float().mean())
                          for action in range(N_ACTIONS)]
        probs = torch.sigmoid(logits)
        bomb0 = [a for a in range(N_ACTIONS) if split_action(a)[1] == 0]
        bomb1 = [a for a in range(N_ACTIONS) if split_action(a)[1] == 1]
        mean_prob_bomb0 = float(probs[:, bomb0].mean())
        mean_prob_bomb1 = float(probs[:, bomb1].mean())
        bomb_mass = (torch.logsumexp(logits[:, bomb1], dim=1)
                     - torch.logsumexp(logits[:, bomb0], dim=1))
        bomb_prob = torch.sigmoid(bomb_mass)
        rule_predicted = (bomb_prob >= 0.5).float()
        aux_acc = float((rule_predicted == targets).float().mean())
        rule_true_positive = int(((rule_predicted == 1) & (targets == 1)).sum())
        rule_predicted_positive = int((rule_predicted == 1).sum())

    ckpt = (pathlib.Path(args.ckpt) if args.ckpt
            else policy_ckpt_path(root, arm, seed))
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), str(ckpt))
    report_path = (pathlib.Path(args.out) if args.out
                   else train_report_path(root, arm, seed))
    report: dict[str, Any] = {
        "schema": SCHEMA_TRAIN, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "arm": arm, "arm_description": ARM_DESCRIPTION[arm],
        "H": int(horizon), "H_label_source": rows[0].get("label_source"),
        "seed": seed, "states": len(rows), "labels": int(len(rows) * N_ACTIONS),
        "pool_filter": filter_name,
        "feature_dim": int(FEATURE_DIM), "actions": int(N_ACTIONS),
        "bombs_remaining_distribution": bombs_counts,
        "aux_target_positive_states": int(targets.sum()),
        "aux_rule": {"name": ARM_AUX_TARGET_NAME,
                     "definition": ("1 iff best_bomb1 > best_bomb0 on the row's "
                                    "own H (the label's best-bomb decision)"),
                     "positive_states": int(targets.sum()),
                     "source": ("tools/rl/v8j0_earlybomb.py::aux_best_bomb_target "
                                "(the row's own best_bomb0 / best_bomb1)")},
        "model": {"class": "v431_arm_a.Policy with the output layer widened to 36",
                  "shape": f"Linear({FEATURE_DIM},{HIDDEN}) -> Tanh -> "
                           f"Linear({HIDDEN},36)",
                  "params": int(sum(p.numel() for p in policy.parameters())),
                  "source": "tools/rl/v830_bombpol.py::policy36 (imported)"},
        "labels_file": info, "features": array_info,
        "optimizer": {"name": "Adam", "lr": LR, "batch": batch, "epochs": EPOCHS,
                      "early_stopping": None,
                      "shuffle": ("numpy default_rng(seed), one permutation/"
                                  "epoch")},
        "loss": {"main": "BCEWithLogitsLoss over the 36 outputs",
                 "aux": ("bomb marginal BCE against the best-bomb bit "
                         "(1 iff best_bomb1 > best_bomb0 on that row's H; "
                         "logsumexp over bomb=1 minus logsumexp over bomb=0)"
                         if aux_weight > 0.0 else None),
                 "aux_weight": aux_weight,
                 "total": ("main" if aux_weight == 0.0
                           else f"main + {aux_weight} * aux")},
        "init": {"torch_manual_seed": seed, "hidden": HIDDEN,
                 "feature_dim": int(FEATURE_DIM)},
        "epoch_losses": [round(value, 6) for value in epoch_losses],
        "epoch_aux_losses": ([round(value, 6) for value in epoch_aux]
                             if aux_weight > 0.0 else None),
        "loss_first_epoch": round(epoch_losses[0], 6),
        "loss_last_epoch": round(epoch_losses[-1], 6),
        "final_train_bce": round(final_bce, 6),
        "final_train_aux_bce": (None if final_aux is None else round(final_aux, 6)),
        "train_accuracy": round(overall_acc, 6),
        "train_accuracy_per_action": [round(v, 6) for v in per_action_acc],
        "mean_predicted_prob_bomb0": round(mean_prob_bomb0, 6),
        "mean_predicted_prob_bomb1": round(mean_prob_bomb1, 6),
        "aux_bomb_agreement": {"accuracy": round(aux_acc, 6),
                               "predicted_positive": rule_predicted_positive,
                               "true_positive": rule_true_positive,
                               "rule_positive": int(targets.sum()),
                               "rule": ARM_AUX_TARGET_NAME},
        "ckpt": {"path": str(ckpt), "sha256": sha256_file(ckpt),
                 "bytes": ckpt.stat().st_size},
        "pins": check_pins(), "wall_s": wall, "deviations": [],
    }
    write_json(report_path, report)
    log(f"train arm={arm} seed={seed}: loss {report['loss_first_epoch']:.4f} -> "
        f"{report['loss_last_epoch']:.4f} (BCE {report['final_train_bce']:.4f}, "
        f"acc {overall_acc:.4f}, aux-acc {aux_acc:.4f}) wall={wall}s -> {ckpt}")
    return 0


def mode_train_summary(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    summary: dict[str, Any] = {"schema": SCHEMA_TRAIN_SUMMARY, "producer": PRODUCER,
                               "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                               "prereg": PREREG_REL, "arms": {}}
    for arm in ARMS:
        summary["arms"][arm] = {}
        for seed in TRAIN_SEEDS:
            path = train_report_path(root, arm, seed)
            if not path.is_file():
                summary["arms"][arm][str(seed)] = {"missing": str(path)}
                continue
            report = read_json(path)
            summary["arms"][arm][str(seed)] = {
                "states": report["states"], "arm": report["arm"],
                "H": report["H"],
                "loss_first_epoch": report["loss_first_epoch"],
                "loss_last_epoch": report["loss_last_epoch"],
                "final_train_bce": report["final_train_bce"],
                "final_train_aux_bce": report["final_train_aux_bce"],
                "train_accuracy": report["train_accuracy"],
                "aux_bomb_agreement": report["aux_bomb_agreement"],
                "wall_s": report["wall_s"], "ckpt": report["ckpt"]}
    write_json(root / "train_summary.json", summary)
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            block = summary["arms"][arm].get(str(seed), {})
            if "missing" in block:
                continue
            log(f"train arm={arm} seed={seed}: bce={block['final_train_bce']:.4f} "
                f"acc={block['train_accuracy']:.4f} "
                f"aux_acc={block['aux_bomb_agreement']['accuracy']:.4f}")
    return 0


# --------------------------------------------------------------------------- #
# mode: eval (one arm x one budget x one policy seed x 4 dev windows)
# --------------------------------------------------------------------------- #
def load_policy36(ckpt: pathlib.Path, *, seed: int) -> Any:
    import torch

    path = pathlib.Path(ckpt)
    if not path.is_file():
        raise EarlyBombError(f"policy checkpoint not found: {path}")
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
    arm = str(args.arm).upper()
    budget = int(args.decisions)
    if seed not in TRAIN_SEEDS:
        raise EarlyBombError(f"--policy-seed must be one of {TRAIN_SEEDS}, got "
                             f"{seed}")
    if arm not in ARMS:
        raise EarlyBombError(f"--arm must be one of {ARMS}, got {arm}")
    if budget not in EVAL_BUDGETS[arm]:
        raise EarlyBombError(f"prereg §1 fixes arm {arm} to budgets "
                             f"{EVAL_BUDGETS[arm]}, got {budget}")
    if int(args.decision_frames) != DECISION_FRAMES:
        raise EarlyBombError(f"prereg §1 keeps {DECISION_FRAMES} frames per "
                             f"decision")
    if int(args.episodes) < 1:
        raise EarlyBombError(f"--episodes must be >= 1, got {args.episodes}")
    windows = parse_windows(args.windows)
    for window in windows:
        if int(window) not in DEV_WINDOWS:
            raise EarlyBombError(f"window {window} is not a dev window "
                                 f"{DEV_WINDOWS}")
    engine_info = engine_gate(args.engine)
    launch_env = prepare_env()
    # a **unique** tag per (arm, budget, seed): parallel evals must never share a
    # private data dir (v8.17.0's 18-way launch proved the failure mode: 10 of 18
    # cells died with `th08.dat` missing / a non-muted cfg because every process
    # used the same timestamp tag)
    tag = args.tag or f"v8j0-eval-{arm}-d{budget}-s{seed}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    out_path = (pathlib.Path(args.out) if args.out
                else eval_path(root, arm, budget, seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")
    ckpt = (pathlib.Path(args.ckpt) if args.ckpt
            else policy_ckpt_path(root, arm, seed))
    digest = sha256_file(ckpt) if ckpt.is_file() else None
    policy = load_policy36(ckpt, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    arm_mod = arm_a()

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        feats, _detail = arm_mod.build_features(state=state, hazards=hazards,
                                                decision_index=decision)
        return v830().split_action(
            v830().sample_action36(policy, feats, args.kind, generator))

    summary: dict[str, Any] = {
        "schema": SCHEMA_EVAL_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "mode": "eval", "arm": arm, "arm_description": ARM_DESCRIPTION[arm],
        "policy_seed": seed, "windows": list(windows),
        "episodes_per_window": int(args.episodes),
        "budget_decisions": budget, "decision_frames": DECISION_FRAMES,
        "temperature": TEMPERATURE, "kind": args.kind, "tag": tag,
        "engine": engine_info, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest,
                 "bytes": ckpt.stat().st_size if ckpt.is_file() else None},
        "sampler": {"kind": "torch.multinomial on softmax(z/1) over 36 actions",
                    "generator_seed": generator_seed(seed),
                    "generator": ("one torch.Generator per (arm, budget, seed) "
                                  "process, windows in the frozen order "
                                  "11,12,17,18")},
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
            if not data_info["cfg"]["muted_four_zero"]:
                raise EarlyBombError(f"w{window}: the generated th08.cfg is not "
                                     f"muted: {data_info['cfg']}")
            log(f"arm {arm} d{budget} s{seed} w{window}: cfg muted="
                f"{data_info['cfg']['muted_four_zero']} "
                f"sha256={str(data_info['cfg']['sha256'])[:12]} "
                f"data_dir={data_dir}")
            parent, gate = start_parent(slot=slot, window=window,
                                        engine=args.engine, data_dir=data_dir,
                                        log_dir=run_dir)
            parents.append(parent)
            rows: list[dict[str, Any]] = []
            for episode in range(int(args.episodes)):
                record = v830().run_episode36(
                    parent, episode=episode, policy_seed=seed, decisions=budget,
                    decide=decide, budget=budget, decision_frames=DECISION_FRAMES)
                record["schema"] = "v8180-earlybomb-eval-episode/1"
                record["producer"] = PRODUCER
                record["arm"] = arm
                record["eval_harness"] = ("tools/rl/v830_bombpol.py::"
                                          "run_episode36 (imported)")
                record.update({"ckpt_sha256": digest,
                               "generator_seed": generator_seed(seed)})
                record.pop("terminal_reply", None)
                record.pop("trace", None)
                record.pop("spawn_reply", None)
                write_jsonl_line(handle, record)
                rows.append(record)
                if record.get("invalid"):
                    raise EarlyBombError(
                        f"arm {arm} d{budget} seed {seed} window {window} "
                        f"episode {episode} is invalid: {record.get('error')}")
                if (episode + 1) % 50 == 0:
                    done = sum(1 for row in rows if row["survived"])
                    used = sum(int(row["bombs_used"] or 0) for row in rows)
                    log(f"  arm {arm} d{budget} s{seed} w{window}: {episode + 1}/"
                        f"{args.episodes} episodes, survived={done}, "
                        f"bombs_used={used}")
            alive = [float(row["alive_frames"]) for row in rows]
            survived = sum(1 for row in rows if row["survived"])
            summary["per_window"][str(window)] = {
                "window": int(window), "episodes": len(rows),
                "survived": survived,
                "survived_fraction": (round(survived / len(rows), 6)
                                      if rows else None),
                "alive_frames": quantiles(alive),
                "frames": quantiles([float(row["frames"]) for row in rows]),
                "bomb_episodes": sum(1 for row in rows if row["bomb_emits"]),
                "bomb_emits_total": sum(int(row["bomb_emits"]) for row in rows),
                "bombs_used_total": sum(int(row["bombs_used"] or 0) for row in rows),
                "episodes_exhausted": sum(1 for row in rows if row["bombs_end"] == 0),
                "bombs_used_counts": count_map([int(row["bombs_used"] or 0)
                                                for row in rows]),
                "cfg_muted": data_info["cfg"]["muted_four_zero"],
                "cfg_sha256": data_info["cfg"]["sha256"],
                "data_dir": str(data_dir),
                "gate": {key: gate[key] for key in (
                    "hello_role_shadow", "park_ok", "park_frame", "threads_1",
                    "all_pass")},
                "gate_prefix": {key: gate["prefix"][key] for key in (
                    "hash_checks", "hash_mismatches", "final_frame")},
            }
            log(f"arm {arm} d{budget} seed {seed} window {window}: survived "
                f"{survived}/{len(rows)} mean_alive="
                f"{summary['per_window'][str(window)]['alive_frames']['mean']}")
        summary["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["status"] = "failed"
        log(f"EVAL FAILED arm {arm} d{budget} seed {seed}: {summary['error']}")
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
    log(f"eval arm {arm} d{budget} seed {seed}: status={summary['status']} "
        f"survived={summary.get('survived_total')}/"
        f"{summary.get('episodes_total')} bombs_used="
        f"{summary.get('bombs_used_total')} wall={summary['wall_s']}s -> {out_path}")
    return 0 if summary["status"] == "complete" else 4


# --------------------------------------------------------------------------- #
# mode: diagnostics (offline; counts and distributions only, no test)
# --------------------------------------------------------------------------- #
def bomb_usage_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    used = [int(row.get("bombs_used") or 0) for row in rows]
    firsts = [int(row["first_bomb_decision"]) for row in rows
              if row.get("first_bomb_decision") is not None]
    lasts = [int(row["last_bomb_decision"]) for row in rows
             if row.get("last_bomb_decision") is not None]
    return {
        "episodes": len(rows),
        "bombs_used_counts": count_map(used),
        "bombs_used_quantiles": quantiles(used),
        "bombs_used_mean": (round(statistics.fmean(used), 4) if used else None),
        "episodes_using_1_or_2": sum(1 for value in used if 1 <= value <= 2),
        "episodes_using_0": sum(1 for value in used if value == 0),
        "episodes_using_3": sum(1 for value in used if value >= 3),
        "first_bomb_decision": quantiles(firsts),
        "last_bomb_decision": quantiles(lasts),
        "bomb_emits_total": sum(int(row.get("bomb_emits") or 0) for row in rows),
    }


def mode_diagnostics(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    payload: dict[str, Any] = {
        "schema": SCHEMA_DIAG, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "requirement": ("prereg §2 P1-P4 are counts and distributions here; the "
                        "test and the interval belong to the independent "
                        "analysis pass. No test, no interval in this file."),
        "note": ("the 12 (seed, window) cells are not independent samples; the "
                 "per-cell numbers are listed beside the pooled ones and must "
                 "not be averaged into a mean of means"),
        "arms": {}, "bomb_usage": {},
        "references": {"v830_bombpol": BASELINE_V830, "v860_bombdag": BASELINE_V860,
                       "v8170_conserve": BASELINE_V817, "rule_r5": RULE_R5},
    }
    for arm in ARMS:
        payload["arms"][arm] = {}
        payload["bomb_usage"][arm] = {}
        for budget in EVAL_BUDGETS[arm]:
            per_seed: dict[str, Any] = {}
            pooled_rows: list[dict[str, Any]] = []
            for seed in TRAIN_SEEDS:
                path = eval_path(root, arm, budget, seed)
                if not path.is_file():
                    per_seed[str(seed)] = {"missing": str(path)}
                    continue
                rows = read_jsonl(path)
                pooled_rows.extend(rows)
                survived = sum(1 for row in rows if row["survived"])
                per_seed[str(seed)] = {
                    "episodes": len(rows), "survived": survived,
                    "survived_fraction": (round(survived / len(rows), 6)
                                          if rows else None),
                    "alive_frames": quantiles([float(row["alive_frames"])
                                               for row in rows]),
                    "bombs_used": bomb_usage_block(rows),
                    "per_window": {
                        str(window): {
                            "episodes": sum(1 for row in rows
                                            if int(row["window"]) == window),
                            "survived": sum(1 for row in rows
                                            if int(row["window"]) == window
                                            and row["survived"]),
                        } for window in DEV_WINDOWS},
                }
            survived_all = sum(1 for row in pooled_rows if row["survived"])
            payload["arms"][arm][f"d{budget}"] = {
                "budget": budget, "per_seed": per_seed,
                "episodes_total": len(pooled_rows),
                "survived_total": survived_all,
                "pooled_survived_fraction": (round(survived_all / len(pooled_rows), 6)
                                             if pooled_rows else None),
                "bomb_usage_pooled": bomb_usage_block(pooled_rows),
            }
    for arm in ARMS:
        for budget in EVAL_BUDGETS[arm]:
            block = payload["arms"][arm].get(f"d{budget}", {})
            if block:
                payload["bomb_usage"][arm][f"d{budget}"] = block.get(
                    "bomb_usage_pooled")
    table: dict[str, Any] = {"v830": BASELINE_V830["rate"],
                             "v860": BASELINE_V860["rate"],
                             "v8170_20s": BASELINE_V817["rate_20s"],
                             "rule_r5": RULE_R5["rate"]}
    for arm in ARMS:
        table[f"arm_{arm}"] = {str(budget): payload["arms"][arm].get(
            f"d{budget}", {}).get("pooled_survived_fraction")
            for budget in EVAL_BUDGETS[arm]}
    payload["side_by_side_survived_fraction"] = table
    write_json(root / "diagnostics.json", payload)
    write_json(root / "bomb_usage.json", {
        "schema": SCHEMA_BOMB_USAGE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "requirement": ("prereg §2 P3: does the bomb-usage distribution contain "
                        "episodes that spend only 1-2 bombs, and is the mean < "
                        "3.00?"),
        "usage": payload["bomb_usage"]})
    for arm in ARMS:
        for budget in EVAL_BUDGETS[arm]:
            block = payload["arms"][arm].get(f"d{budget}", {})
            if not block:
                continue
            log(f"diagnostics arm {arm} d{budget}: "
                f"{block['survived_total']}/{block['episodes_total']} = "
                f"{(block['pooled_survived_fraction'] or 0):.4f}, bombs "
                f"{block['bomb_usage_pooled']['bombs_used_counts']} mean="
                f"{block['bomb_usage_pooled']['bombs_used_mean']}")
    return 0


# --------------------------------------------------------------------------- #
# mode: audio-audit / process-audit / manifest / publish
# --------------------------------------------------------------------------- #
def mode_audio_audit(args: argparse.Namespace) -> int:
    """``v450_cf_label.mode_audit`` on this round's own ``--work-dir``."""

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
                        "may survive its own run (filtered by "
                        "--data-dir-filter)"),
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
        raise EarlyBombError(f"not a directory: {root}")
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
        (pooled_labels_path(root), "labels_earlybomb.jsonl"),
        (root / "labels_earlybomb_alive.jsonl", "labels_earlybomb_alive.jsonl"),
        (root / "summary.json", "summary.json"),
        (pooled_stats_path(root), "earlybomb_stats.json"),
        (root / "smoke.json", "smoke.json"),
        (root / "preflight.json", "preflight.json"),
        (root / "ckpt_assertions.json", "ckpt_assertions.json"),
        (root / "ckpt_assertions_final.json", "ckpt_assertions_final.json"),
        (root / "fidelity_d3.json", "fidelity_d3.json"),
        (root / "fidelity_d3_wrapper.json", "fidelity_d3_wrapper.json"),
        (root / "train_summary.json", "train_summary.json"),
        (root / "diagnostics.json", "diagnostics.json"),
        (root / "bomb_usage.json", "bomb_usage.json"),
        (root / "audio_audit.json", "audio_audit.json"),
        (root / "process_residue.json", "process_residue.json"),
        (root / "commands.tsv", "commands.tsv"),
        (root / "work_manifest.json", "work_manifest.json"),
    ]
    for name in ("exec.sh", "RUN.md"):
        wanted.append((root / name, name))
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            wanted.append((train_report_path(root, arm, seed),
                           train_report_path(root, arm, seed).name))
            wanted.append((policy_ckpt_path(root, arm, seed),
                           f"ckpt_{arm}_earlybomb_seed{seed}.pt"))
        for budget in EVAL_BUDGETS[arm]:
            for seed in TRAIN_SEEDS:
                path = eval_path(root, arm, budget, seed)
                wanted.append((path, path.name))
                wanted.append((path.with_suffix(".summary.json"),
                               path.with_suffix(".summary.json").name))
    for path in sorted((root / "collect").glob("seed*_w*.json")):
        wanted.append((path, f"collect/{path.name}"))
    for path in sorted((root / "smoke").glob("smoke_*.json")):
        wanted.append((path, f"smoke/{path.name}"))
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
               "v8j0_earlybomb_sha256": sha256_file(
                   pathlib.Path(__file__).resolve())}
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
        description="v8.18.0 EARLYBOMB-001: sample 60-120 frames BEFORE a death "
                    "(the behaviour never bombs) and label them 36 ways, so that "
                    "bomb = 1 is an ordinary bomb (1 spent) and not a death bomb "
                    "(2 spent)")
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "assert-ckpt", "fidelity", "smoke",
                                 "collect", "stats", "train", "train-summary",
                                 "eval", "diagnostics", "audio-audit",
                                 "process-audit", "manifest", "publish"))
    parser.add_argument("--window", default=WINDOW_DEFAULT,
                        choices=(*WINDOW_NAMES, WINDOW_AUTO),
                        help=("collect/smoke: the sampling distance to the death. "
                              "early = decisions_before_death in [30,60] (prereg "
                              "§1, 60-120 frames) | late = [60,120] (§2.1 "
                              "S-dead's shift, 120-240 frames) | deep = [120,240] "
                              "| auto = the first rung whose sampled ALIVE share "
                              "reaches 90 %"))
    parser.add_argument("--seed", type=int, default=0,
                        help="behaviour/training seed (0/1/2)")
    parser.add_argument("--policy-seed", type=int, default=None,
                        help="eval: alias of --seed")
    parser.add_argument("--arm", default="A", choices=("A", "B", "a", "b"))
    parser.add_argument("--windows", default="11,12,17,18")
    parser.add_argument("--decisions", type=int, default=DECISIONS,
                        help="collect cap / eval budget")
    parser.add_argument("--decision-frames", type=int, default=DECISION_FRAMES)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--horizon", type=int, default=0,
                        help="train: which H to train with (0 = the verdict's "
                             "smallest H reaching the P0d gate)")
    parser.add_argument("--verify-h", type=int, default=0,
                        help="collect: 1 = verify H=120/240 against direct runs "
                             "on the first labelled state of the cell")
    parser.add_argument("--min-death-decision", type=int,
                        default=MIN_DEATH_DECISION_DEFAULT,
                        help=("collect/smoke: discard an episode whose death "
                              f"comes before this decision (default "
                              f"{MIN_DEATH_DECISION_DEFAULT}; the work order's "
                              f"relaxation is "
                              f"{MIN_DEATH_DECISION_FALLBACK}). A death at "
                              "decision 45 would turn the [30, 60] window into "
                              "decisions 0..15, where bomb = 0 survives by "
                              "construction"))
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--kind", default="stochastic",
                        choices=("stochastic", "greedy"))
    parser.add_argument("--quota", type=int, default=QUOTA_PER_CELL,
                        help="collect: labelled states per (seed, window) cell")
    parser.add_argument("--max-episodes", type=int, default=MAX_EPISODES_DEFAULT)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--work-dir", default=str(WORK))
    parser.add_argument("--labels-out", default="")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--out", default="")
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
    args.arm = str(args.arm).upper()

    engine_modes = ("smoke", "collect", "eval", "fidelity")
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
        if args.mode == "fidelity":
            return mode_fidelity(args)
        if args.mode == "smoke":
            return mode_smoke(args)
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
    except EarlyBombError as exc:
        log(f"FAILED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
