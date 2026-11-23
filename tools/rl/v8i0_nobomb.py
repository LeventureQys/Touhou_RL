#!/usr/bin/env python3
"""
v8.17.0 CONSERVE-001, **`none` branch** (NOBOMB) - sample the death
neighbourhood with a behaviour policy that **never spends a bomb**, so that every
sampled state is armed by construction (``bombs_remaining = 3``) and the 36-way
labels are finally asked a question that has an answer: *can a bomb that is still
in hand buy survival?*

Why this round exists
---------------------
v8.15 and v8.16 closed the mechanism from both sides on R5's trajectory:

    v8.15  the 10,008 v8.11 states (policy still alive)  -> critical = 0 / 10,008
           (some candidate always survived 120 frames: a bomb is never *needed*)
    v8.16  the last 20 decisions before a death          -> bombs_at_death = {0: 3}
           (R5 spends all three bombs before the death it cannot avoid, so the
            bomb = 1 branch is a no-op and ``best bomb=1 - best bomb=0`` is 0)

so on R5's trajectory a bomb is never the *marginal* resource: it is spent when
it is not needed and gone when it is.  This round moves **one** factor - the
behaviour policy's bomb rule - and leaves everything else frozen:

============================  =================================================
v8.16  (R5 behaviour)         staged 80 / 60 / 40 px by bombs_remaining
**v8.17.0**                   ``conserve25``: a bomb is spent **only** when
                              ``nearest_all_px < 25`` (and ``bombs_remaining > 0``)
                              -- or ``none``: never spend a bomb at all
frozen                        ``S`` stochastic behaviour, 900 decisions,
                              death-stop, last 20 decisions, 36-way labels at
                              H = 120 / 240 / 600, 4 windows x 3 seeds x 50
============================  =================================================

The expectation (prereg §1): under a 25 px rule the player walks into the death
with 2-3 bombs left, so the death neighbourhood finally contains states where
``bombs_remaining > 0`` and a bomb can pay for survival.  ``none`` is the purest
form of the same experiment (the player always dies with all three bombs) and is
the prereg's ``S-nobomb`` fallback if the conservative rule still spends them.

Both distance rules failed to leave a bomb in hand:

* v8.16.0 (R5, staged 80/60/40 px): ``bombs_at_death = {0: 3}``, and **all
  1,431** death-adjacent states had ``bombs_remaining = 0``;
* v8.17.0's 25 px CONSERVE smoke: ``bombs 3 -> 0`` with ``emits = 15`` and **all
  27** sampled states ``bombs_remaining = 0``.

so any *distance-triggered* rule spends the bombs before the death it cannot
avoid.  This file therefore takes the prereg's third and last option: **never
bomb at all**.  P0a then holds at 100 % by construction, and the round reduces to
P0b alone (``best bomb = 1 > best bomb = 0``) - the minimal, purest experiment.

Arm A's supervision follows: the aux bit is **the label's own best-bomb
decision** (``best_bomb1 > best_bomb0`` on the row's own horizon, i.e. exactly the
state-level ``diff`` that P0b counts), not a distance rule.  The 25 px rule's bit
is still computed and reported beside it for continuity, and the behaviour's own
bit is 0 everywhere (``--bomb-rule none``).

What is measured (prereg §2 P0a / P0b, both computed per horizon)
-----------------------------------------------------------------
* **P0a** ``armed_fraction``: share of the sampled states whose
  ``bombs_remaining > 0`` (gate: **>= 50 %**).  A state with 0 bombs left cannot
  express a bomb's value at all, so this is the precondition of the whole round.
* **P0b** ``diff_fraction_of_armed``: of those armed states, the share with
  ``best bomb=1 > best bomb=0`` at that horizon (gate: **>= 20 %**).  With
  ``--bomb-rule none`` the armed set is *every* state, so P0b is measured on the
  whole sample.
* both broken down by ``bombs_remaining`` (3 / 2 / 1 / 0) and by the decisions
  left before the death, per horizon, plus the ``bombs_at_death`` distribution
  from the cell summaries and the ``bombs_remaining`` distribution of the death
  neighbourhood itself.

Verdicts (prereg §2.1): ``S-nobomb`` if P0a fails (re-collect with
``--bomb-rule none``), ``S-nodiff`` if P0a holds but P0b < 20 % for every horizon
(publish the statistics, do not train), ``train`` otherwise - the **smallest**
horizon whose P0b reaches the gate is the one trained on.

The behaviour policy
--------------------
``S`` = the frozen v4.5.0 DAgger student ``round2.pt`` of the same seed
(``4abee069...`` / ``a93746bb...`` / ``c69960fc...``), sampled **stochastically**
(``torch.Generator().manual_seed(seed*1000+7)``), exactly as v4.5.0 / v8.3.0 /
v8.6.0 / v8.11.0 / v8.17.0 used it.  Only the bomb bit changes:

    --bomb-rule conserve25   fired = bombs_remaining > 0 and nearest_all_px < 25
    --bomb-rule none         fired = 0, always (all three bombs kept)
    --bomb-rule staged       v8.17.0's R5 (80/60/40 px), kept for continuity;
                             reused through tools/rl/v800_bomb.py::rule_gate

``nearest_all_px`` is ``v800_bomb.nearest_bullet_distances`` over the live
``HAZARDS`` bullet list - the same measurement R5 used, so the two rounds'
thresholds are comparable.

Arm A's supervision is **the label's own best-bomb decision**, whichever rule
collected the data: ``aux = 1 iff best_bomb1 > best_bomb0`` at the row's own
horizon - the same ``diff`` that P0b counts, recomputed per state from the 36
labels.  The 25 px conservative bit (``aux_bomb_conserve``, ``nearest_all_px < 25
and bombs_remaining > 0``) is still recorded per state and reported beside it, but
it is **not** arm A's target in this branch: with ``--bomb-rule none`` the
behaviour's own bit is 0 everywhere, so a rule-derived bit would reintroduce a
distance rule that this branch exists to remove.  Every bit is recorded per state
(``rule_bomb`` = the behaviour's = 0, ``aux_bomb_conserve`` = the 25 px rule,
``best_bomb0`` / ``best_bomb1`` / ``delta`` = the label's own answer).

Two passes, and why
-------------------
The "last 20 decisions before death" window is only known once the death has
happened, so each episode is walked twice:

    pass 1  the behaviour trajectory (HAZARDS + PARK + 2 frames per decision),
            no labels: records every decision's live ``nearest_all_px``, the
            threshold, the rule bit and the executed action, and stops at death;
    sample  the frozen rule: ``[death_decision - 19 .. death_decision]``;
    pass 2  the warm RESET, then the **recorded** actions are replayed and only
            the sampled decisions are parked and labelled (36 children, H = 600,
            truncated to H = 120 / 240).

Replay is verified, never assumed: at **every** decision of pass 2 the live
``frame`` and the live ``nearest_all_px`` must equal pass 1's values, and any
difference aborts the cell (rc 4).  An episode that survives all 900 decisions is
discarded (prereg §1: "不死就换下一条") and cannot contribute a state.

Modes
-----
==================================  =============================================
``--mode preflight``                 pinned imports (sha256 of every file this
                                     round imports **and** of the frozen prereg),
                                     feature dim (106), action table, frozen design
``--mode assert-ckpt --phase P``     the three frozen v4.5.0 ``round2.pt``
                                     (``4abee069`` / ``a93746bb`` / ``c69960fc``),
                                     the engine sha256 and every protected source
``--mode recall-v816``               **no engine**: recompute v8.16's
                                     ``bombs_at_death = {0: 3}`` and its P0a from
                                     the v8.16 pooled labels, so the negative
                                     baseline is measured here, not quoted
``--mode collect --seed S --windows W --quota N [--bomb-rule R]``
                                     one (seed, window) death-trajectory cell:
                                     roll to death (cap 900 decisions), sample
                                     the last 20 decisions, label 36 ways at
                                     H = 120/240/600 (three rows per state),
                                     write ``labels/seed{S}_w{W}.jsonl``
``--mode stats``                     pool the cells, compute P0a / P0b per horizon
                                     (rc 4 if P0a fails or no horizon reaches the
                                     P0b gate, prereg §2.1)
``--mode train --arm {A,B} --seed S`` Policy(106,64,36), Adam 1e-3, batch 256,
                                     200 epochs, no early stopping, dev never read
``--mode train-summary``             the six training reports side by side
``--mode eval --arm {A,B} --decisions {300,600,900} --policy-seed S``
                                     the frozen v4.5.0 episode loop with a 36-way
                                     policy (``v830_bombpol.run_episode36``):
                                     4 dev windows, 300 episodes each, stochastic
``--mode diagnostics``               offline per-arm / per-budget counts, rates and
                                     the **bomb-usage table** (P3).  **No test.**
``--mode audio-audit``               AGENTS.md §9.3 on this round's own data dirs
``--mode process-audit``             this round's own engine processes must be 0
``--mode manifest --dir D``          sha256 + bytes of every file in D
``--mode publish``                   copy the raw artifacts into
                                     ``Document/Update/Alpha/v8.17.0/evidence/
                                     CONSERVE/`` and write the manifest

Reused, never re-implemented (imported by path, sha256 pinned; a moved file is
exit 2 and nothing runs)
----------------------------------------------------------------------------
* ``tools/rl/v860_bombdag.py`` - ``load_behaviour_policy``, ``behaviour_ckpt``,
  ``decision_bucket`` and the frozen design constants (its ``behaviour_decision``
  is **not** used: the bomb rule is this round's factor, so the same body is
  reproduced here with ``--bomb-rule`` as a parameter, and the ``staged`` branch
  still goes through ``v800_bomb.rule_gate``).
* ``tools/rl/v450_cf_label.py`` - ``prepare_data_dir`` (a freshly generated muted
  ``th08.cfg`` per engine start, AGENTS.md §9.3), ``masks_for``,
  ``compact_observation``, ``Digest``, ``mode_audit``, ``check_engine``,
  ``prepare_env``, ``start_parent``.
* ``tools/rl/v432_fork_arm.py`` - ``ForkParent`` (warm RESET, prefix walk,
  SPAWN/CHILD/QUIT), ``_step_fields``, ``fields``.
* ``tools/rl/v431_arm_a.py`` - ``build_features`` (the frozen 106-dim
  observation) and ``Policy``.
* ``tools/rl/v800_bomb.py`` - ``nearest_bullet_distances`` and ``rule_gate``
  (R5's threshold schedule; the rule is **reused**, not re-implemented).
* ``tools/rl/v830_bombpol.py`` - ``label_state36`` (the 36-way labeller),
  ``action_masks`` / ``split_action`` / ``action_of``, ``policy36`` (the
  ``Policy(106,64,36)`` shape), ``run_episode36`` (the frozen evaluation loop),
  ``build_arrays`` (the 106-dim / 36-label matrices with the
  ``features_sha256`` re-check), ``sample_action36``.
* ``touhou_rl/v400/gym/protocol.parse_hazards_reply`` - the live ``HAZARDS``
  frame (the full bullet list, which is what ``nearest_all_px`` is taken over);
* ``touhou_rl/v420/counterfactual._terminal_of`` - the death rule of R24/C4;
* ``touhou_rl/v420/contracts.cand_to_mask`` and ``assert_seed_allowed``.

Failure policy (fail closed where the data would be wrong, record where the
engine is merely counter-intuitive)
-------------------------------------------------------------------
* a pinned import whose sha256 moved -> exit 2, nothing runs;
* a pass-2 replay whose ``frame`` or ``nearest_all_px`` differs from pass 1 at
  any decision -> the cell is aborted (rc 4) and its file stays ``*.partial``;
* a labelled state whose 36 labels are incomplete, or whose 18 + 18 children did
  not all start at the parked parent's frame / state -> the cell is aborted
  (rc 4);
* the **bomb = 1 invariant is two-branched, never "must spend one"** (v8.6.0's
  measured engine semantics: a press inside a bomb's invulnerability window is a
  no-op).  Counted and reported, never aborting;
* P0a failing, or P0b failing for every horizon -> rc 4 from ``--mode stats`` and
  the round stops there (prereg §2.1 ``S-nobomb`` / ``S-nodiff``): only the
  sampling statistics are published, no training, no survival numbers;
* every episode record carries ``died`` / ``invalid`` / ``error``; an invalid
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

WORK = REPO / ".tmp_v8i0_nobomb"
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.17.0" / "evidence"
           / "CONSERVE")
PREREG = REPO / "Document" / "Update" / "Alpha" / "v8.17.0" / (
    "PREREGISTRATION_CONSERVE_001.md")
PREREG_REL = "Document/Update/Alpha/v8.17.0/PREREGISTRATION_CONSERVE_001.md"
# v8.16.0's pooled labels (its `.tmp` pool; the evidence copy is written by its
# own publish step) - `--mode recall-v816` re-measures the negative baseline here
V816_LABELS = (REPO / ".tmp_v8g0_deathadj" / "labels_deathadj.jsonl")

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

SCHEMA_PREFLIGHT = "v8170-conserve-preflight/1"
SCHEMA_ASSERT = "v8170-conserve-ckpt-assertions/1"
SCHEMA_LABEL = "v8170-conserve-label/1"
SCHEMA_COLLECT = "v8170-conserve-collect/1"
SCHEMA_STATS = "v8170-conserve-stats/1"
SCHEMA_RECOUNT = "v8170-conserve-recall-v816/1"
SCHEMA_TRAIN = "v8170-conserve-train/1"
SCHEMA_TRAIN_SUMMARY = "v8170-conserve-train-summary/1"
SCHEMA_EVAL_SUMMARY = "v8170-conserve-eval-summary/1"
SCHEMA_DIAG = "v8170-conserve-diagnostics/1"
SCHEMA_BOMB_USAGE = "v8170-conserve-bomb-usage/1"
SCHEMA_AUDIT = "v8170-conserve-audio-audit/1"
SCHEMA_PROC = "v8170-conserve-process-audit/1"
SCHEMA_MANIFEST = "v8170-conserve-manifest/1"
SCHEMA_PUBLISH = "v8170-conserve-publish/1"
PRODUCER = ("tools/rl/v8i0_nobomb.py (live engine, HarnessClient role=shadow, "
            "behaviour policy = v4.5.0 `S` stochastic + **no bomb at all** "
            "(--bomb-rule none, bombs_remaining = 3 at every sampled state), "
            "900-decision death-adjacent sampling: the last 20 decisions before "
            "the death, 36-way labels at H = 120/240/600)")

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- frozen geometry (unchanged from v4.5.0/v6.0.0/v8.0.0/v8.3.0/v8.6.0) ------
PREFIX_FRAMES = 8663
CHECKPOINTS_IN_PREFIX = 2165
ENGINE_SHA256 = "691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6"

# -- frozen split (prereg §1: the v4.5.0 / v6.0.0 / v8.3.0 / v8.6.0 protocol) -
TRAIN_WINDOWS: tuple[int, ...] = (13, 14, 15, 16)
DEV_WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
TRAIN_SEEDS: tuple[int, ...] = (0, 1, 2)

# -- frozen design (prereg §1) ----------------------------------------------
N_CAND = 18
N_ACTIONS = 36
BOMB_VALUES: tuple[int, ...] = (0, 1)
DECISION_FRAMES = 2
DECISIONS = 900                  # 30 s budget (v8.16.0: R5 survives
                                 # 900 decisions only 17.61 % of the time, so
                                 # ~82 % of the episodes now reach a death
HORIZONS: tuple[int, ...] = (120, 240, 600)   # the H sweep (v8.16.0, frozen)
H_LABEL_MAX = 600                # one 36-child run to this horizon yields all
                                 # three H labels by alive(H) = min(alive(600), H)
H_DEFAULT = 120                  # v8.11/v8.15's horizon (kept as the default H)
H_P0_MIN_DIFF = 0.20             # the per-horizon "有差异" gate
SAMPLE_WINDOWS: tuple[int, ...] = (11, 12, 17, 18)   # the work order's windows
MAX_CONSECUTIVE_NO_DEATH = 10    # skip the (seed, window) cell after 10 in a row
EPISODES = 300
HIDDEN = 64
FEATURE_DIM = 106
LR = 1e-3
BATCH = 256
EPOCHS = 200
TEMPERATURE = 1.0
GEN_CONST = 7
DEFAULT_TIMEOUT = 300.0

# -- the one moved factor: death-adjacent sampling (prereg §1) ---------------
DEATH_WINDOW = 20                # the last N = 20 decisions before the death
DEATH_WINDOW_FRAMES = DEATH_WINDOW * DECISION_FRAMES    # = the last 40 frames
NEAREST_RECORDED = 12            # the nearest 12 bullets go into every state
TARGET_STATES = 600              # >= 600 states **per horizon** (12 x 50)
TARGET_DEATH_EPISODES = 30       # "约 30 条死亡 episode x 20 就够 600"
CELLS = 12                       # 3 seeds x 4 windows (11, 12, 17, 18)
QUOTA_PER_CELL = TARGET_STATES // CELLS                 # 50 -> 12 x 50 = 600
MAX_EPISODES_DEFAULT = 200       # the work order: `for episode in range(200)`

# -- the two arms (prereg §1) -----------------------------------------------
ARMS: tuple[str, ...] = ("A", "B")
ARM_AUX_WEIGHT = {"A": 0.5, "B": 0.0}
ARM_AUX_TARGET_NAME = "best_bomb_argmax"
ARM_DESCRIPTION = {
    "A": ("BCE(36 路) + 0.5 x BCE_bomb(best-bomb: max(alive|bomb=1) > "
          "max(alive|bomb=0) 的标签，即 argmax alive_frames 落在 bomb=1 时取 1)"),
    "B": "仅 BCE(36 路) — 对照臂",
}
EVAL_BUDGETS = {"A": (300, 600, 900), "B": (300, 600, 900)}

# -- P0a / P0b (prereg §2, computed per horizon) -----------------------------
P0A_ARMED_MIN = 0.50             # share of sampled states with bombs_remaining > 0
P0B_DIFF_MIN = 0.20              # of those armed states, share with bomb=1 better
P0_DIFF_MIN = P0B_DIFF_MIN       # the gate's old name, kept for the references
P0_CRITICAL_MIN = 0.50           # v8.16's precondition, still reported
BOMB_SEGMENTS: tuple[int, ...] = (3, 2, 1, 0)

# -- the actual behaviour policy: v4.5.0 `S` (the frozen DAgger student) ----
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
    "tools/rl/v850_distill.py",
    "tools/rl/v8b0_eventsample.py",
    "tools/rl/v8f0_critical.py",
    "tools/rl/v8g0_deathadj.py",      # this round copies it, never edits it
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)

# -- R5: v8.4.0 BOMBRULE2-001 §1's staged rule (reused through v800_bomb) ----
R5_RULE = "staged"
R5_THRESHOLDS: dict[int, float] = {3: 80.0, 2: 60.0, 1: 40.0}
R5_SOURCE = ("Document/Update/Alpha/v8.4.0/PREREGISTRATION_BOMBRULE2_001.md §1 "
             "(implemented in tools/rl/v800_bomb.py::rule_gate / "
             "STAGED_THRESHOLDS)")

# -- **the one moved factor**: the behaviour policy's bomb rule (prereg §1) ---
CONSERVE_THRESHOLD_PX = 25.0     # `conserve25`: a bomb only when a bullet is
                                 # this close (R5 fired at 80 / 60 / 40 px)
CONSERVE_RULE = "conserve25"
NO_BOMB_RULE = "none"
BOMB_RULES: tuple[str, ...] = (CONSERVE_RULE, NO_BOMB_RULE, R5_RULE)
BOMB_RULE_DEFAULT = NO_BOMB_RULE     # this branch: never spend a bomb
CONSERVE_SOURCE = ("Document/Update/Alpha/v8.17.0/"
                   "PREREGISTRATION_CONSERVE_001.md §1 "
                   "(the conservative bomb rule: `nearest_all_px < 25` and "
                   "`bombs_remaining > 0`; `--bomb-rule none` is §2.1's "
                   "S-nobomb fallback)")
BOMB_RULE_DESCRIPTION = {
    CONSERVE_RULE: ("threshold = 25 px for every bombs_remaining; fired = "
                    "bombs_remaining > 0 and nearest_all_px < 25"),
    NO_BOMB_RULE: ("fired = 0 always: the behaviour never spends a bomb, so the "
                   "player dies with all three (prereg §2.1 S-nobomb's purest "
                   "form)"),
    R5_RULE: ("v8.16.0's R5: bombs 3 -> 80 px, 2 -> 60 px, 1 -> 40 px, through "
              "tools/rl/v800_bomb.py::rule_gate (kept for continuity)"),
}

# -- the references this round is placed beside (numbers only, no test) -----
BASELINE_V830 = {
    "arm": "v8.3.0 BOMBPOL-001 — behaviour policy v4.5.0 `S` (never bombs)",
    "episodes_per_budget": 3600,
    "rate": {300: 0.999444, 600: 0.194444, 900: 0.006111},
    "source": ("Document/Update/Alpha/v8.3.0/evidence/BOMBPOL/diagnostics.json"),
}
BASELINE_V860 = {
    "arm": "v8.6.0 BOMBDAG-001 — S + R5 rule supervision (DAgger on the rule)",
    "episodes_per_budget": 3600,
    "rate": {300: 0.999444, 600: 0.295833, 900: 0.029444},
    "source": ("Document/Update/Alpha/v8.6.0/evidence/BOMBDAG/diagnostics.json "
               "and v8.11.0 RUN.md's BASELINE table (99.94 / 29.58 / 2.94)"),
}
RULE_R5 = {
    "arm": "v8.4.0 R5 `staged` — script rule: bombs 3 -> 80 px, 2 -> 60 px, "
           "1 -> 40 px, `nearest_all_px < threshold`, bombs > 0",
    "episodes_per_budget": 3600,
    "rate": {300: 1.0, 600: 0.7711, 900: 0.1761},
    "source": ("Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/RUN.md §0/§3 "
               "(600 decisions measured 77.1111 % = 2,776/3,600) and the quoted "
               "100 / 17.61 for 300 / 900 in v8.6.0's BASELINE table"),
}
V815_CRITICAL = {
    "arm": "v8.15.0 CRITICAL-001 — screen of the 10,008 v8.11 states",
    "states": 10008, "critical": 0,
    "source": ("Document/Update/Alpha/v8.15.0/evidence/CRITICAL/RUN.md "
               "(see --mode recall-v816 for this round's own baseline)"),
}

_LOG_FH: Any = None
_MODS: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class ConserveError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v8i0] {message}"
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
        raise ConserveError(f"cannot import {path}")
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
    """``tools/rl/v860_bombdag.py`` — the v8.6.0 behaviour policy (reused whole)."""

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
# **the one moved factor**: the behaviour decision, with the bomb rule as an
# argument (v860_bombdag.behaviour_decision in every other respect)
# --------------------------------------------------------------------------- #
def bomb_rule_threshold(bomb_rule: str, *, bombs_before: int,
                        player_x: float, decision: int) -> dict[str, Any]:
    """The threshold/non-distance gates of one bomb rule (prereg §1).

    ``staged`` is R5 and is delegated to ``v800_bomb.rule_gate`` unchanged (the
    rule is reused, not re-implemented); ``conserve25`` is this round's single
    moved factor: **one** threshold, 25 px, independent of ``bombs_remaining``;
    ``none`` never fires (``threshold_px`` is recorded as ``None``).
    """

    if bomb_rule == R5_RULE:
        return dict(m800().rule_gate(bomb_rule, decision=int(decision),
                                     bombs_before=int(bombs_before),
                                     player_x=float(player_x)))
    if bomb_rule == CONSERVE_RULE:
        return {"threshold_px": CONSERVE_THRESHOLD_PX, "decision_ok": True,
                "position_ok": True}
    if bomb_rule == NO_BOMB_RULE:
        return {"threshold_px": None, "decision_ok": False, "position_ok": False}
    raise ConserveError(f"unknown bomb rule {bomb_rule!r} "
                        f"(rules: {list(BOMB_RULES)})")


def behaviour_decision(*, state: dict[str, Any], hazards: Any, policy: Any,
                       generator: Any, decision: int, bomb_rule: str) -> dict[str, Any]:
    """One behaviour decision: the frozen ``S`` candidate plus this round's bomb.

    The candidate comes from the frozen 18-way policy (``stochastic``, the same
    sampler v4.5.0's evaluation used) and is **identical to v8.16.0's**: only the
    bomb bit changes, and every distance in the record comes from
    ``v800_bomb.nearest_bullet_distances`` (the same measurement R5 used).
    """

    arm = arm_a()
    feats, _detail = arm.build_features(state=state, hazards=hazards,
                                        decision_index=int(decision))
    cand = v450().policy_action(policy, feats, "stochastic", generator)
    distances = m800().nearest_bullet_distances(state, hazards)
    nearest = distances["nearest_all_px"]
    bombs_before = int(state["bombs_remaining"] or 0)
    player_x = float(state.get("player_x") or 0.0)
    gate = bomb_rule_threshold(bomb_rule, bombs_before=bombs_before,
                               player_x=player_x, decision=int(decision))
    threshold = gate["threshold_px"]
    if bomb_rule == NO_BOMB_RULE:
        fired = False
    else:
        fired = bool(bombs_before > 0 and nearest is not None
                     and float(nearest) < float(threshold))
    fired = bool(fired and bombs_before > 0)
    return {"feats": feats, "features_sha256_k12": features_sha256(feats),
            "cand": int(cand), "bomb": int(fired),
            "action": int(cand) * 2 + int(fired),
            "nearest_all_px": nearest,
            "nearest_lethal_px": distances["nearest_lethal_px"],
            "bullets": distances["bullets"],
            "bullets_within_60": distances["bullets_within_60"],
            "bombs_before": int(bombs_before), "threshold_px": threshold,
            "rule": bomb_rule, "distances": distances,
            "decision_ok": bool(gate["decision_ok"]),
            "position_ok": bool(gate["position_ok"])}


def best_bomb_target(row: dict[str, Any]) -> int:
    """Arm A's supervision bit in this branch: **the label's best-bomb decision**.

    The work order's definition (per state, on the row's own horizon): take the
    candidate whose **best** branch over both bomb bits is the best of the 18
    (``argmax_c max(alive[2c], alive[2c+1])``) and return ``1`` iff that
    candidate's ``bomb = 1`` branch outlives its ``bomb = 0`` branch
    (``alive[2c+1] > alive[2c]``), else ``0``.

    It is computed from the label row alone (``alive_frames`` is written by
    ``--mode collect``), so it needs no rule, no distance and no behaviour bit -
    and the behaviour's own bit is 0 everywhere in this branch, which is exactly
    why a rule-derived target would push "never bomb" against the labels.
    """

    alive = [int(value) for value in row["alive_frames"]]
    if len(alive) != N_ACTIONS:
        raise ConserveError(f"{len(alive)} alive_frames != {N_ACTIONS}")
    best_cand = max(range(N_CAND),
                    key=lambda cand: max(alive[2 * cand], alive[2 * cand + 1]))
    return int(alive[2 * best_cand + 1] > alive[2 * best_cand])


def aux_bomb_conserve(nearest_all_px: Any, bombs_remaining: int) -> int:
    """The **25 px conservative rule's** bit, from the row alone (reference only).

    ``1`` iff the state still has a bomb *and* a bullet is inside 25 px - the bit
    v8.17.0's ``conserve25`` collection would have fired at that decision.  In this
    branch it is **not** arm A's target (see ``best_bomb_target``): it is recorded
    and reported beside the label's own answer so the two can be compared, and it
    is never trained on.
    """

    if nearest_all_px is None:
        return 0
    return int(int(bombs_remaining) > 0
               and float(nearest_all_px) < CONSERVE_THRESHOLD_PX)


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def cell_labels_path(root: pathlib.Path, seed: int, window: int) -> pathlib.Path:
    return root / "labels" / f"seed{int(seed)}_w{int(window)}.jsonl"


def cell_summary_path(root: pathlib.Path, seed: int, window: int) -> pathlib.Path:
    return root / "collect" / f"seed{int(seed)}_w{int(window)}.json"


def pooled_labels_path(root: pathlib.Path) -> pathlib.Path:
    return root / "labels_nobomb.jsonl"


def pooled_stats_path(root: pathlib.Path) -> pathlib.Path:
    return root / "nobomb_stats.json"


def policy_ckpt_path(root: pathlib.Path, arm: str, seed: int) -> pathlib.Path:
    return root / "ckpt" / str(arm) / f"seed{int(seed)}" / "nobomb.pt"


def train_report_path(root: pathlib.Path, arm: str, seed: int) -> pathlib.Path:
    return root / "train" / f"train_{arm}_nobomb_seed{int(seed)}.json"


def eval_path(root: pathlib.Path, arm: str, budget: int,
              seed: int) -> pathlib.Path:
    return (root / "eval" / str(arm)
            / f"eval_{arm}_nobomb_d{int(budget)}_seed{int(seed)}.jsonl")


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
# the live hazard record (v8.11's, reused unchanged)
# --------------------------------------------------------------------------- #
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


def death_window(death_decision: int, *, length: int = DEATH_WINDOW) -> list[int]:
    """The frozen sample rule: the last ``length`` decisions before the death.

    ``[death_decision - length + 1 .. death_decision]`` clipped at 0.  The last
    element is the decision whose two frames contained the death; the first is
    ``2 * (length - 1)`` frames before it.
    """

    lo = max(0, int(death_decision) - int(length) + 1)
    return list(range(lo, int(death_decision) + 1))


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
    child dies inside the horizon and ``alive_frames = horizon`` when it does
    not (v4.5.0 §2.1 / R24, reused unchanged).  One action's mask sequence does
    **not** depend on the horizon (bomb bit on the first frame, the plain
    candidate on every later frame), so the H = 120 run is exactly the first
    120 frames of the H = 600 run and, frame for frame,

        alive_frames(H) = min(alive_frames(H_max), H)
        safe(H)         = 1  <=>  alive_frames(H) == H

    The derivation is asserted here; on the smoke state the caller additionally
    runs the direct H = 120 / H = 240 labels and compares them
    (``verify_horizon_nesting``), so the nesting is *measured*, not assumed.

    Returns ``(block_at_H_max, {H: {alive_frames, safe, ...}}, meta)``.
    """

    h_max = max(int(h) for h in horizons)
    if h_max != H_LABEL_MAX:
        raise ConserveError(f"the H sweep runs to H = {H_LABEL_MAX}, got {h_max}")
    block, children = v830().label_state36(parent, state=state, horizon=h_max,
                                           park_reply=park_reply)
    if len(block["alive_frames"]) != N_ACTIONS:
        raise ConserveError(f"{len(block['alive_frames'])} alive_frames != "
                            f"{N_ACTIONS}")
    alive_max = [int(value) for value in block["alive_frames"]]
    derived: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        h = int(horizon)
        if h > h_max:
            raise ConserveError(f"H={h} > H_max={h_max}")
        alive = [min(value, h) for value in alive_max]
        safe = [1 if value == h else 0 for value in alive]
        for action in range(N_ACTIONS):
            if (safe[action] == 1) != (alive[action] == h):
                raise ConserveError(f"H={h} a{action}: derived safe/alive "
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

    Called on the smoke state (and on the first state of a cell when
    ``--verify-h 1``).  A single difference is a fail-closed error: it would
    mean the H sweep is not the truncation it claims to be.
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
                                                 horizon=h,
                                                 park_reply=park_reply)
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
        raise ConserveError(f"H-nesting verification failed: {out}")
    return out


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
                   "single_factor": ("the behaviour policy's bomb rule: R5's "
                                     "staged 80/60/40 px (v8.16.0) -> the "
                                     "conservative 25 px rule (or none at all)")},
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
            "checkpoints": {str(s): {"path": str(v860().behaviour_ckpt(s)),
                                     "expected": FROZEN_ROUND2_SHA256[s]}
                            for s in TRAIN_SEEDS},
            "bomb_rule_default": BOMB_RULE_DEFAULT,
            "bomb_rules": {rule: {"description": BOMB_RULE_DESCRIPTION[rule],
                                  "gate": ("never" if rule == NO_BOMB_RULE else
                                           "nearest_all_px < threshold and "
                                           "bombs_remaining > 0")}
                           for rule in BOMB_RULES},
            "conserve_threshold_px": CONSERVE_THRESHOLD_PX,
            "r5_thresholds_px": {str(k): v for k, v in R5_THRESHOLDS.items()},
            "bomb_rule_source": CONSERVE_SOURCE,
            "reused_through": ("tools/rl/v8i0_nobomb.py::behaviour_decision "
                               "(v860_bombdag.behaviour_decision's body with the "
                               "bomb rule as a parameter; `staged` still goes "
                               "through v800_bomb.rule_gate)"),
            "aux_rule": {"name": CONSERVE_RULE,
                         "definition": ("arm A's BCE_bomb target: 1 iff "
                                        "bombs_remaining > 0 and "
                                        "nearest_all_px < 25, from the label row "
                                        "itself"),
                         "threshold_px": CONSERVE_THRESHOLD_PX},
            "roll_out": (f"roll until the player dies, cap {DECISIONS} decisions "
                         f"({DECISIONS * DECISION_FRAMES} frames = "
                         f"{DECISIONS * DECISION_FRAMES / 30.0:.1f} s at 30 fps); "
                         f"an episode that does not die is discarded"),
        },
        "sampling": {
            "kind": "death-adjacent",
            "window_decisions": DEATH_WINDOW,
            "window_frames": DEATH_WINDOW_FRAMES,
            "horizons": [int(h) for h in HORIZONS],
            "horizon_max": H_LABEL_MAX,
            "horizon_source": ("one 36-child run to H = 600; alive(H) = "
                               "min(alive(600), H), verified against direct "
                               "H = 120 / H = 240 runs on the smoke state"),
            "windows": list(SAMPLE_WINDOWS),
            "max_consecutive_no_death": MAX_CONSECUTIVE_NO_DEATH,
            "target_states_per_horizon": TARGET_STATES,
            "rule": ("the last N = 20 decisions before the death: "
                     "[death_decision - 19 .. death_decision]"),
            "live_nearest": ("nearest_all_px = min over the live HAZARDS bullet "
                             "list at that decision (not the 12 stored bullets)"),
            "nearest_bullets_recorded": NEAREST_RECORDED,
            "two_pass": ("pass 1 rolls the trajectory to death and records every "
                         "decision's nearest_all_px/threshold/rule bit; pass 2 "
                         "replays the recorded actions and labels only the last "
                         "20 decisions"),
            "replay_check": ("frame and nearest_all_px must equal pass 1 at every "
                             "decision of pass 2; a difference aborts the cell"),
        },
        "arms": {arm: {"loss": ARM_DESCRIPTION[arm],
                       "aux_weight": ARM_AUX_WEIGHT[arm],
                       "eval_budgets": list(EVAL_BUDGETS[arm])} for arm in ARMS},
        "frozen": {"TRAIN_WINDOWS": list(TRAIN_WINDOWS),
                   "DEV_WINDOWS": list(DEV_WINDOWS),
                   "SAMPLE_WINDOWS": list(SAMPLE_WINDOWS),
                   "train_seeds": list(TRAIN_SEEDS),
                   "H": H_DEFAULT, "horizons": [int(h) for h in HORIZONS],
                   "H_max": H_LABEL_MAX, "decisions_cap": DECISIONS,
                   "decision_frames": DECISION_FRAMES,
                   "episodes_per_window": EPISODES,
                   "lr": LR, "batch": BATCH, "epochs": EPOCHS, "hidden": HIDDEN,
                   "p0a_armed_min": P0A_ARMED_MIN,
                   "p0b_diff_min": P0B_DIFF_MIN,
                   "p0_critical_min_context": P0_CRITICAL_MIN,
                   "conserve_threshold_px": CONSERVE_THRESHOLD_PX,
                   "bomb_rule_default": BOMB_RULE_DEFAULT,
                   "target_states": TARGET_STATES,
                   "target_death_episodes": TARGET_DEATH_EPISODES,
                   "quota_per_cell": QUOTA_PER_CELL,
                   "max_episodes_default": MAX_EPISODES_DEFAULT,
                   "max_consecutive_no_death": MAX_CONSECUTIVE_NO_DEATH},
        "references": {"v830_bombpol": BASELINE_V830, "v860_bombdag": BASELINE_V860,
                       "rule_r5": RULE_R5, "v815_critical": V815_CRITICAL},
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
        "requirement": ("checklist item 2: the behaviour policy of every v8.17.0 "
                        "state is the frozen v4.5.0 `S` student, whose final "
                        "checkpoints are round2.pt = 4abee069… / a93746bb… / "
                        "c69960fc…; the protected sources must not have moved"),
        "engine": None, "ckpts": {}, "protected": {}, "v8i0_nobomb_py": {},
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
        payload["v8i0_nobomb_py"] = {"path": str(me), "sha256": sha256_file(me)}
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
# mode: recall-v816 (the negative baseline, recomputed here)
# --------------------------------------------------------------------------- #
def reduce_alive(alive: Sequence[int], *, horizon: int) -> dict[str, Any]:
    """Per-state reduction shared by ``stats`` and ``recall-v816``.

    ``critical`` = every one of the 36 candidates dies inside H frames
    (``max(alive_frames) < 120``); the bomb comparison is the best bomb = 1
    branch against the best bomb = 0 branch **of the same state**.
    """

    values = [int(v) for v in alive]
    if len(values) != N_ACTIONS:
        raise ConserveError(f"{len(values)} alive_frames != {N_ACTIONS}")
    bomb1 = [values[a] for a in range(N_ACTIONS) if split_action(a)[1] == 1]
    bomb0 = [values[a] for a in range(N_ACTIONS) if split_action(a)[1] == 0]
    if len(bomb1) != N_CAND or len(bomb0) != N_CAND:
        raise ConserveError(f"bomb halves are {len(bomb1)}/{len(bomb0)}")
    best1 = max(bomb1)
    best0 = max(bomb0)
    return {"max_alive": max(values), "min_alive": min(values),
            "mean_alive": statistics.fmean(values),
            "best_bomb1": best1, "best_bomb0": best0, "delta": best1 - best0,
            "critical": bool(max(values) < int(horizon)),
            "n_safe": sum(1 for value in values if value == int(horizon)),
            "all_36_safe": bool(all(value == int(horizon) for value in values)),
            "mirror_delta": max(values[0::2]) - max(values[1::2])}


def mode_recall(args: argparse.Namespace) -> int:
    """Recompute **v8.16.0's negative baseline** from its own pooled labels.

    v8.16.0 measured ``bombs_at_death = {0: 3}``: on R5's trajectory every death
    is reached with **no bomb left**, so ``bomb = 1`` is a no-op and P0a (and
    therefore P0b) cannot pass no matter how the labels are cut.  This round is
    only meaningful beside that number, so it is **recomputed here** from the
    published rows rather than quoted - with this round's own reduction and this
    round's own P0a / P0b definitions, at every horizon.
    """

    source = pathlib.Path(args.labels_file) if args.labels_file else V816_LABELS
    if not source.is_file():
        raise ConserveError(f"no v8.16.0 pooled label file: {source}")
    rows = 0
    per_h: dict[int, dict[str, Any]] = {
        int(h): {"states": 0, "armed": 0, "diff_armed": 0, "critical": 0,
                 "diff_critical": 0, "bombs": {}}
        for h in HORIZONS}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        horizon = int(row["H"])
        if horizon not in per_h:
            raise ConserveError(f"row {rows}: H={horizon} is not one of "
                                f"{list(HORIZONS)}")
        red = reduce_alive(row["alive_frames"], horizon=horizon)
        bombs = int(row["bombs_remaining"])
        block = per_h[horizon]
        block["states"] += 1
        block["bombs"][str(bombs)] = block["bombs"].get(str(bombs), 0) + 1
        if bombs > 0:
            block["armed"] += 1
            if red["delta"] > 0:
                block["diff_armed"] += 1
        if red["critical"]:
            block["critical"] += 1
            if red["delta"] > 0:
                block["diff_critical"] += 1
        rows += 1
    by_h: dict[str, Any] = {}
    for horizon, block in per_h.items():
        states = block["states"]
        armed = block["armed"]
        by_h[str(int(horizon))] = {
            "H": int(horizon), "states": states,
            "bombs_remaining_counts": dict(sorted(block["bombs"].items(),
                                                  reverse=True)),
            "armed": armed,
            "armed_fraction": (round(armed / states, 6) if states else None),
            "p0a_met": bool(states and (armed / states) >= P0A_ARMED_MIN),
            "diff_states_of_armed": block["diff_armed"],
            "diff_fraction_of_armed": (round(block["diff_armed"] / armed, 6)
                                       if armed else None),
            "critical": block["critical"],
            "critical_fraction": (round(block["critical"] / states, 6)
                                  if states else None),
            "diff_states_of_critical": block["diff_critical"],
            "diff_fraction_of_critical": (round(block["diff_critical"]
                                                / block["critical"], 6)
                                          if block["critical"] else None),
        }
    payload = {
        "schema": SCHEMA_RECOUNT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("the round's contrast: v8.16.0's death neighbourhood was "
                        "collected with R5's staged 80/60/40 px rule and must "
                        "measure P0a = 0 here, so that this round's P0a is "
                        "attributed to the bomb rule and not to the sampling"),
        "source": {"path": str(source), "sha256": sha256_file(source),
                   "bytes": source.stat().st_size, "rows": rows},
        "reference": {"round": "v8.16.0 DEATHADJ-001",
                      "measured": "bombs_at_death = {0: 3}",
                      "source": ("Document/Update/Alpha/v8.16.0/evidence/"
                                 "DEATHADJ/RUN.md (this round's smoke)")},
        "definitions": {"armed": "bombs_remaining > 0 (P0a)",
                        "diff": ("max(alive_frames over the 18 bomb=1 branches) > "
                                 "max(alive_frames over the 18 bomb=0 branches)"),
                        "p0a_armed_min": P0A_ARMED_MIN,
                        "p0b_diff_min": P0B_DIFF_MIN},
        "rows": rows, "by_H": by_h,
    }
    out = pathlib.Path(args.out) if args.out else (pathlib.Path(args.work_dir)
                                                   / "recall_v816.json")
    write_json(out, payload)
    for horizon in HORIZONS:
        block = by_h[str(int(horizon))]
        log(f"recall-v816 H={int(horizon)}: armed {block['armed']}/"
            f"{block['states']} (P0a "
            f"{'PASS' if block['p0a_met'] else 'FAIL'}), bombs="
            f"{block['bombs_remaining_counts']}")
    return 0


# --------------------------------------------------------------------------- #
# mode: collect (roll to death -> last 20 decisions -> 36-way labels)
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
    verify_h = bool(int(getattr(args, "verify_h", 0) or 0))
    if seed not in TRAIN_SEEDS:
        raise ConserveError(f"--seed must be one of {TRAIN_SEEDS}, got {seed}")
    if horizon not in (0, H_LABEL_MAX):
        raise ConserveError(f"the H sweep labels every state at {list(HORIZONS)} "
                            f"from one run to H_max = {H_LABEL_MAX}; got "
                            f"--horizon {horizon}")
    if decisions_cap != DECISIONS:
        raise ConserveError(f"the round fixes the trajectory budget at {DECISIONS} "
                            f"decisions (={DECISIONS * DECISION_FRAMES} frames = "
                            f"{DECISIONS * DECISION_FRAMES / 30.0:.0f} s) per "
                            f"episode; got {decisions_cap}")
    if int(args.window_decisions) != DEATH_WINDOW:
        raise ConserveError(f"prereg §1 fixes N = {DEATH_WINDOW} decisions before "
                            f"the death; got {args.window_decisions}")
    bomb_rule = str(args.bomb_rule)
    if bomb_rule not in BOMB_RULES:
        raise ConserveError(f"--bomb-rule must be one of {list(BOMB_RULES)}, got "
                            f"{bomb_rule!r}")
    for window in windows:
        if window not in SAMPLE_WINDOWS:
            raise ConserveError(f"window {window} is not one of the work order's "
                                f"sampling windows {SAMPLE_WINDOWS}")
    horizon = H_LABEL_MAX        # the one 36-child run the three labels come from
    pins = check_pins()
    if not pins["all_match"]:
        raise ConserveError(f"pinned import moved: {json.dumps(pins)[:800]}")
    engine = engine_gate(args.engine)
    policy, ckpt, ckpt_sha = v860().load_behaviour_policy(seed)
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    launch_env = prepare_env()
    bomb1_actions = [a for a in range(N_ACTIONS) if split_action(a)[1] == 1]
    bomb0_actions = [a for a in range(N_ACTIONS) if split_action(a)[1] == 0]

    for window in windows:
        tag = args.tag or f"v8i0-nobomb-collect-s{seed}-w{window}"
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
            "bomb_rule": bomb_rule,
            "conserve_threshold_px": CONSERVE_THRESHOLD_PX,
            "H_max": H_LABEL_MAX, "horizons": list(HORIZONS),
            "decisions_budget": DECISIONS,
            "max_consecutive_no_death": MAX_CONSECUTIVE_NO_DEATH,
            "sampling_windows": list(SAMPLE_WINDOWS),
            "window_decisions": DEATH_WINDOW,
            "window_frames": DEATH_WINDOW_FRAMES,
            "actions_per_state": N_ACTIONS,
            "state_rows_per_state": len(HORIZONS),
            "sampling": {"kind": "death-adjacent",
                         "window_decisions": DEATH_WINDOW,
                         "window_frames": DEATH_WINDOW_FRAMES,
                         "rule": ("the last N = 20 decisions before the death: "
                                  "[death_decision - 19 .. death_decision]"),
                         "nearest_bullets_recorded": NEAREST_RECORDED},
            "behaviour": {
                "policy": "v4.5.0 `S` (frozen DAgger student, round2.pt)",
                "ckpt": str(ckpt), "ckpt_sha256": ckpt_sha,
                "kind": "stochastic",
                "generator_seed": generator_seed(seed),
                "bomb_rule": bomb_rule,
                "bomb_rule_description": BOMB_RULE_DESCRIPTION[bomb_rule],
                "conserve_threshold_px": CONSERVE_THRESHOLD_PX,
                "thresholds_px": ({str(k): v for k, v in R5_THRESHOLDS.items()}
                                  if bomb_rule == R5_RULE else
                                  {str(b): CONSERVE_THRESHOLD_PX
                                   for b in BOMB_SEGMENTS}),
                "bomb_rule_gate": ("nearest_all_px < threshold and "
                                   "bombs_remaining > 0" if bomb_rule != NO_BOMB_RULE
                                   else "never (fired = 0 by definition)"),
                "bomb_rule_source": (R5_SOURCE if bomb_rule == R5_RULE
                                     else CONSERVE_SOURCE),
                "reused_through": ("tools/rl/v8i0_nobomb.py::behaviour_decision "
                                   "(v860_bombdag.behaviour_decision's body with "
                                   "the bomb rule as a parameter)"),
                "roll_out": (f"until death, cap {decisions_cap} decisions "
                             f"({decisions_cap * DECISION_FRAMES} frames = "
                             f"{decisions_cap * DECISION_FRAMES / 30.0:.0f} s)"),
                "mask_schedule": ("cand_to_mask(cand, bomb) on the decision's "
                                  "first frame, cand_to_mask(cand, 0) afterwards")},
            "aux_rule": {
                "name": CONSERVE_RULE,
                "definition": ("arm A's supervision bit, recorded per state as "
                               "aux_bomb_conserve: 1 iff bombs_remaining > 0 and "
                               "nearest_all_px < 25 - independent of --bomb-rule"),
                "threshold_px": CONSERVE_THRESHOLD_PX},
            "bomb1_semantics": ("cand_to_mask(cand, 1) on the label's first frame, "
                                "cand_to_mask(cand, 0) on frames 2..H (v8.3.0 §1)"),
            "pins": pins, "engine": engine, "launch_env": launch_env,
            "labels_path": str(labels), "status": "failed", "error": None,
            "episodes": [], "states": 0, "death_episodes": 0,
            "no_death_episodes": 0, "invalid_episodes": 0,
            "states_by_H": {str(h): 0 for h in HORIZONS},
            "consecutive_no_death": 0, "consecutive_no_death_max": 0,
            "window_skipped_after": None, "skip_reason": None,
            "nesting": None, "nesting_verified": None,
            "bombs_remaining_counts": {}, "positive_counts": {},
            "rule_bomb_counts": {}, "bomb_emits": 0, "thresholds_used": {},
            "decision_counts": {}, "decisions_before_death_counts": {},
            "death_decision_quantiles": None,
            "bombs_at_death_counts": {}, "death_states_by_bombs": {},
            "replay_checks": 0, "replay_mismatches": 0,
            "label_checks": {"labels_36": 0, "labels_missing": 0,
                             "baseline_frame_ok_total": 0,
                             "baseline_state_equal_parent_total": 0,
                             "spawn_ok_total": 0, "quit_ok_total": 0,
                             "bomb0_spent_a_bomb": 0,
                             "bomb1_press_honoured": 0,
                             "bomb1_press_ignored": 0,
                             "bomb1_spent_two_or_more": 0,
                             "bomb1_spent_amounts": {},
                             "bomb1_noop_due_to_invuln": 0,
                             "bomb1_unassigned": 0,
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
            "pass_costs_s": {"pass1": [], "pass2": []},
        }
        handle = partial.open("w", encoding="utf-8")
        parents: list[Any] = []
        started = time.perf_counter()
        deaths0 = 0.0
        stage0 = 0
        state: dict[str, Any] = {}
        death_decisions: list[int] = []
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
                raise ConserveError(f"w{window}: parked parent at frame "
                                    f"{state['frame']}, expected {PREFIX_FRAMES}")
            deaths0 = float(state["deaths"] or 0.0)
            stage0 = int(state["stage"] or 0)
            episode = 0
            consecutive_no_death = 0
            while rec["states"] < quota and episode < max_episodes:
                ep: dict[str, Any] = {
                    "episode": int(episode), "decisions": 0, "frames": 0,
                    "died": False, "terminal": None, "death_decision": None,
                    "death_frames_executed": None,
                    "bombs_start": int(state["bombs_remaining"] or 0),
                    "bombs_end": None, "bombs_used": None, "bomb_emits": 0,
                    "states_labelled": 0, "fires": 0,
                    "bombs_remaining_counts": {}, "first_bomb_decision": None,
                    "last_bomb_decision": None, "prefix": None,
                    "window": None, "pass1_s": None, "pass2_s": None,
                    "replay_checks": 0, "replay_mismatches": 0}
                if episode > 0:
                    prefix = parent.walk_prefix(masks, hashes, window)
                    parent.park()
                    state, _reply = parent.state()
                    if int(state["frame"]) != PREFIX_FRAMES:
                        raise ConserveError(
                            f"w{window} ep{episode}: warm RESET left the parent at "
                            f"frame {state['frame']}")
                    deaths0 = float(state["deaths"] or 0.0)
                    stage0 = int(state["stage"] or 0)
                    ep["bombs_start"] = int(state["bombs_remaining"] or 0)
                    ep["prefix"] = {"hash_checks": prefix["hash_checks"],
                                    "hash_mismatches": prefix["hash_mismatches"],
                                    "final_frame": prefix["final_frame"]}
                # ---------------- pass 1: the behaviour trajectory -----------
                pass1_started = time.perf_counter()
                trace: list[dict[str, Any]] = []
                died = False
                for decision in range(decisions_cap):
                    state_frame = int(state["frame"])
                    hazards = P.parse_hazards_reply(parent.ask("HAZARDS"))
                    got = behaviour_decision(
                        state=state, hazards=hazards, policy=policy,
                        generator=generator, decision=decision,
                        bomb_rule=bomb_rule)
                    park_reply = parent.park()
                    got_frame = m432().fields(park_reply).get("frame")
                    if got_frame != str(state_frame):
                        raise ConserveError(
                            f"w{window} d{decision}: PARK reports frame "
                            f"{got_frame}, the read state is at {state_frame}")
                    trace.append({
                        "decision": int(decision), "frame": state_frame,
                        "nearest_all_px": got["nearest_all_px"],
                        "nearest_lethal_px": got["nearest_lethal_px"],
                        "bullets": int(got["bullets"]),
                        "bullets_within_60": int(got["bullets_within_60"]),
                        "threshold_px": got["threshold_px"],
                        "bombs_before": int(got["bombs_before"]),
                        "cand": int(got["cand"]), "rule_bomb": int(got["bomb"]),
                        "action": int(got["action"]),
                    })
                    ep["decisions"] += 1
                    if got["bomb"]:
                        ep["bomb_emits"] += 1
                        rec["bomb_emits"] += 1
                        tkey = f"{float(got['threshold_px']):g}"
                        rec["thresholds_used"][tkey] = (
                            rec["thresholds_used"].get(tkey, 0) + 1)
                        if ep["first_bomb_decision"] is None:
                            ep["first_bomb_decision"] = int(decision)
                        ep["last_bomb_decision"] = int(decision)
                    key = str(int(got["bombs_before"]))
                    ep["bombs_remaining_counts"][key] = (
                        ep["bombs_remaining_counts"].get(key, 0) + 1)
                    rec["bombs_remaining_counts"][key] = (
                        rec["bombs_remaining_counts"].get(key, 0) + 1)
                    rec["decision_counts"][v860().decision_bucket(decision)] = (
                        rec["decision_counts"].get(
                            v860().decision_bucket(decision), 0) + 1)
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
                        break
                ep["fires"] = sum(1 for entry in trace if entry["rule_bomb"])
                ep["pass1_s"] = round(time.perf_counter() - pass1_started, 3)
                rec["pass_costs_s"]["pass1"].append(ep["pass1_s"])
                if not ep["died"]:
                    # prereg §1: an episode that does not die is discarded, and
                    # its tail is never sampled (that would be v8.11 again).
                    rec["no_death_episodes"] += 1
                    consecutive_no_death += 1
                    rec["consecutive_no_death"] = consecutive_no_death
                    rec["consecutive_no_death_max"] = max(
                        int(rec["consecutive_no_death_max"]),
                        consecutive_no_death)
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
                            f"{consecutive_no_death} consecutive no-death "
                            f"episodes (work order rule)")
                        break
                    continue
                consecutive_no_death = 0
                rec["consecutive_no_death"] = 0
                chosen = death_window(int(ep["death_decision"]))
                ep["window"] = {"first_decision": int(chosen[0]),
                                "last_decision": int(chosen[-1]),
                                "decisions": len(chosen),
                                "frames": len(chosen) * DECISION_FRAMES}
                rec["death_episodes"] += 1
                death_decisions.append(int(ep["death_decision"]))
                rec["bombs_at_death_counts"][
                    str(int(trace[int(ep["death_decision"])]["bombs_before"]))] = (
                    rec["bombs_at_death_counts"].get(
                        str(int(trace[int(ep["death_decision"])]["bombs_before"])),
                        0) + 1)
                log(f"s{seed} w{window} ep{ep['episode']}: DIED at decision "
                    f"{ep['death_decision']} ({ep['decisions']} decisions, "
                    f"{ep['frames']} frames, bombs "
                    f"{ep['bombs_start']}->"
                    f"{int(trace[int(ep['death_decision'])]['bombs_before'])}, "
                    f"emits={ep['bomb_emits']}), window {chosen[0]}..{chosen[-1]}")
                # ---------------- pass 2: replay and label -------------------
                pass2_started = time.perf_counter()
                prefix = parent.walk_prefix(masks, hashes, window)
                parent.park()
                state, _reply = parent.state()
                if int(state["frame"]) != PREFIX_FRAMES:
                    raise ConserveError(
                        f"w{window} ep{episode}: pass-2 warm RESET left the parent "
                        f"at frame {state['frame']}, expected {PREFIX_FRAMES}")
                if prefix["hash_mismatches"] != 0:
                    raise ConserveError(
                        f"w{window} ep{episode}: pass-2 prefix walk had "
                        f"{prefix['hash_mismatches']} hash mismatches")
                deaths0 = float(state["deaths"] or 0.0)
                stage0 = int(state["stage"] or 0)
                total = int(ep["death_decision"]) + 1
                for decision in range(total):
                    entry = trace[decision]
                    state_frame = int(state["frame"])
                    hazards = P.parse_hazards_reply(parent.ask("HAZARDS"))
                    distances = m800().nearest_bullet_distances(state, hazards)
                    # ---- the replay fidelity check: frame + live nearest ----
                    ep["replay_checks"] += 1
                    rec["replay_checks"] += 1
                    if state_frame != int(entry["frame"]):
                        rec["replay_mismatches"] += 1
                        ep["replay_mismatches"] += 1
                        raise ConserveError(
                            f"w{window} ep{episode} d{decision}: pass-2 frame "
                            f"{state_frame} != pass-1 frame {entry['frame']}")
                    if distances["nearest_all_px"] != entry["nearest_all_px"]:
                        rec["replay_mismatches"] += 1
                        ep["replay_mismatches"] += 1
                        raise ConserveError(
                            f"w{window} ep{episode} d{decision}: pass-2 "
                            f"nearest_all_px {distances['nearest_all_px']} != "
                            f"pass-1 {entry['nearest_all_px']}")
                    if decision in chosen and rec["states"] < quota:
                        park_reply = parent.park()
                        got_frame = m432().fields(park_reply).get("frame")
                        if got_frame != str(state_frame):
                            raise ConserveError(
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
                            nesting["window"] = int(window)
                            nesting["episode"] = int(episode)
                            nesting["decision"] = int(decision)
                            nesting["frame"] = state_frame
                            rec["nesting_verified"] = nesting
                            log(f"  nesting verified at d{decision}: "
                                f"{ {k: v['mismatched_actions'] for k, v in nesting['checked'].items()} } "
                                f"mismatches={nesting['mismatches']}")
                        # ---- fail-closed checks on this state's 36 labels ----
                        if len(block["safe"]) != N_ACTIONS:
                            raise ConserveError(
                                f"w{window} d{decision}: {len(block['safe'])} "
                                f"labels")
                        if any(value is None for value in block["safe"]):
                            raise ConserveError(
                                f"w{window} d{decision}: a label is None")
                        for action in range(N_ACTIONS):
                            safe = int(block["safe"][action])
                            alive = int(block["alive_frames"][action])
                            if (safe == 1) != (alive == horizon):
                                raise ConserveError(
                                    f"w{window} d{decision} a{action}: safe={safe} "
                                    f"but alive_frames={alive} at H={horizon}")
                        for horizon_h, hblock in derived.items():
                            if len(hblock["safe"]) != N_ACTIONS \
                                    or len(hblock["alive_frames"]) != N_ACTIONS:
                                raise ConserveError(
                                    f"w{window} d{decision}: H={horizon_h} has "
                                    f"{len(hblock['safe'])}/"
                                    f"{len(hblock['alive_frames'])} labels")
                            for action in range(N_ACTIONS):
                                if (int(hblock["safe"][action]) == 1) != (
                                        int(hblock["alive_frames"][action])
                                        == int(horizon_h)):
                                    raise ConserveError(
                                        f"w{window} d{decision} a{action}: "
                                        f"H={horizon_h} safe/alive disagree")
                            if int(hblock["max_alive_frames"]) > int(horizon_h):
                                raise ConserveError(
                                    f"w{window} d{decision}: H={horizon_h} "
                                    f"max_alive={hblock['max_alive_frames']}")
                        kids = block["children"]
                        if int(kids["baseline_frame_ok"]) != N_ACTIONS:
                            raise ConserveError(
                                f"w{window} d{decision}: only "
                                f"{kids['baseline_frame_ok']}/{N_ACTIONS} children "
                                f"started at the parked parent's frame")
                        if int(kids["spawn_ok"]) != N_ACTIONS \
                                or int(kids["quit_ok"]) != N_ACTIONS:
                            raise ConserveError(
                                f"w{window} d{decision}: spawn/quit "
                                f"{kids['spawn_ok']}/{kids['quit_ok']} of "
                                f"{N_ACTIONS}")
                        if int(kids["baseline_state_equal_parent"]) != N_ACTIONS:
                            raise ConserveError(
                                f"w{window} d{decision}: only "
                                f"{kids['baseline_state_equal_parent']}/{N_ACTIONS} "
                                f"children baseline states equal the parent's")
                        bombs_before = int(entry["bombs_before"])
                        action_executed = int(entry["action"])
                        decrements = block["bomb_decrement_frame"]
                        branches: dict[int, dict[str, Any]] = {}
                        for action in bomb1_actions:
                            before = int(block["bombs_start"][action])
                            after = int(block["bombs_end"][action])
                            alive1 = int(block["alive_frames"][action])
                            alive0 = int(block["alive_frames"][action - 1])
                            spent_amount = int(before) - int(after)
                            spent = bool(spent_amount >= 1)
                            noop = bool(spent_amount == 0 and after == before
                                        and alive1 == alive0)
                            if spent and spent_amount == 1:
                                kind = "spent"
                            elif spent:
                                # the engine spent two bombs on one press: the
                                # deathbomb of a state whose player is DYING
                                kind = "spent_deathbomb_2"
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
                                "cand": split_action(action)[0],
                                "before": before, "after": after,
                                "spent_amount": spent_amount,
                                "alive": alive1, "alive_bomb0": alive0,
                                "decrement_frame": decrements[action],
                                "spent": spent, "noop": noop,
                                "anomaly": anomaly, "kind": kind}
                        b1_kind = [branches[a]["kind"] for a in bomb1_actions]
                        b1_spent = [branches[a]["spent"] for a in bomb1_actions]
                        b1_noop = [branches[a]["noop"] for a in bomb1_actions]
                        b1_alive_equal = [branches[a]["alive"]
                                          == branches[a]["alive_bomb0"]
                                          for a in bomb1_actions]
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
                            dkey = str(dec) if dec is not None else "None"
                            rec["label_checks"]["bomb1_decrement_frames"][dkey] = (
                                rec["label_checks"]["bomb1_decrement_frames"].get(
                                    dkey, 0) + 1)
                        for index in range(len(bomb1_actions)):
                            if not b1_spent[index] and not b1_alive_equal[index]:
                                rec["label_checks"][
                                    "bomb1_unspent_alive_differs"] += 1
                        for action in bomb0_actions:
                            if decrements[action] is not None:
                                rec["label_checks"]["bomb0_spent_a_bomb"] += 1
                        for kind, count in b1_kinds.items():
                            rec["label_checks"]["anomaly_kinds"][kind] = (
                                rec["label_checks"]["anomaly_kinds"].get(kind, 0)
                                + count)
                        if rec["label_checks"]["bomb0_spent_a_bomb"]:
                            rec["label_checks"]["anomaly_kinds"][
                                "anomaly_bomb0_branch_spent_a_bomb"] = rec[
                                    "label_checks"]["bomb0_spent_a_bomb"]
                        rec["label_checks"]["bomb1_press_honoured"] += honoured
                        rec["label_checks"]["bomb1_press_ignored"] += ignored
                        two_bomb = sum(1 for a in bomb1_actions
                                       if branches[a]["spent_amount"] >= 2)
                        rec["label_checks"]["bomb1_spent_two_or_more"] += two_bomb
                        for action in bomb1_actions:
                            amount = str(branches[action]["spent_amount"])
                            rec["label_checks"]["bomb1_spent_amounts"][amount] = (
                                rec["label_checks"]["bomb1_spent_amounts"].get(
                                    amount, 0) + 1)
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
                        rec["label_checks"][
                            "baseline_state_equal_parent_total"] += N_ACTIONS
                        rec["label_checks"]["spawn_ok_total"] += N_ACTIONS
                        rec["label_checks"]["quit_ok_total"] += N_ACTIONS
                        if bombs_before == 0:
                            rec["label_checks"]["states_without_a_bomb_left"] += 1
                        feats, _detail = arm_a().build_features(
                            state=state, hazards=hazards,
                            decision_index=int(decision))
                        feats_sha = features_sha256(feats)
                        obs = v450().compact_observation(state, hazards)
                        nearest_rows = nearest_bullets(state, hazards)
                        rule_bomb = int(entry["rule_bomb"])
                        before_death = int(ep["death_decision"]) - int(decision)
                        base_row = {
                            "schema": SCHEMA_LABEL, "seed": seed,
                            "window": int(window), "episode": int(episode),
                            "decision": int(decision), "frame": state_frame,
                            "H": None,            # set per horizon below
                            "H_max": int(H_LABEL_MAX),
                            "horizons": [int(h) for h in HORIZONS],
                            "label_source": ("one 36-child run to H = 600; "
                                             "alive(H) = min(alive(600), H)"),
                            "actions": N_ACTIONS,
                            "bombs_remaining": int(bombs_before),
                            # ---- prereg §1's per-state record ------------
                            "nearest_all_px": entry["nearest_all_px"],
                            "nearest_lethal_px": entry["nearest_lethal_px"],
                            "bullet_count": int(hazards.bullet_count),
                            "nearest_bullets": nearest_rows,
                            "nearest_bullets_recorded": len(nearest_rows),
                            "bullets_within_60": int(distances["bullets_within_60"]),
                            "rule_bomb": rule_bomb,
                            "rule_bomb_bool": bool(rule_bomb),
                            "threshold_px": entry["threshold_px"],
                            "rule": bomb_rule,
                            # ---- arm A's own supervision bit (prereg §1) ----
                            "aux_bomb_conserve": aux_bomb_conserve(
                                entry["nearest_all_px"], bombs_before),
                            "aux_rule": CONSERVE_RULE,
                            "aux_threshold_px": CONSERVE_THRESHOLD_PX,
                            # ---- the moved factor, recorded per state -----
                            "sample_reason": ("death_window_end"
                                              if before_death == 0
                                              else "death_window"),
                            "death_decision": int(ep["death_decision"]),
                            "decisions_before_death": int(before_death),
                            "frames_before_death": int(before_death
                                                       * DECISION_FRAMES),
                            "is_death_decision": bool(before_death == 0),
                            "window_index": int(decision - chosen[0]),
                            "window_decisions": DEATH_WINDOW,
                            # ---- the branch the parent actually executes ----
                            "behaviour": {
                                "policy": "v4.5.0 `S` round2.pt",
                                "ckpt": str(ckpt), "ckpt_sha256": ckpt_sha,
                                "kind": "stochastic",
                                "generator_seed": generator_seed(seed),
                                "cand": int(entry["cand"]),
                                "bomb": int(rule_bomb),
                                "action": int(action_executed),
                                "nearest_all_px": entry["nearest_all_px"],
                                "nearest_lethal_px": entry["nearest_lethal_px"],
                                "bullets": int(entry["bullets"]),
                                "bullets_within_60": int(entry["bullets_within_60"]),
                                "bomb_rule": bomb_rule,
                                "threshold_px": entry["threshold_px"],
                                "threshold_source": (
                                    "tools/rl/v800_bomb.py::rule_gate('staged')"
                                    if bomb_rule == R5_RULE else
                                    "v8i0_nobomb.py::bomb_rule_threshold("
                                    f"{bomb_rule!r})"),
                                "replay_verified": True},
                            "bombs_before": int(block["bombs_start"][action_executed]),
                            "bombs_after": int(block["bombs_end"][action_executed]),
                            "bomb_spent": bool(
                                int(block["bombs_end"][action_executed])
                                == int(block["bombs_start"][action_executed]) - 1),
                            "noop_due_to_invuln": bool(
                                int(rule_bomb) == 1
                                and action_executed in branches
                                and branches[action_executed]["noop"]),
                            "bomb1_noop_count": int(noop_count),
                            "bomb1_anomaly_count": int(anomaly_count),
                            "bomb1_anomaly_cands": anomaly_cands,
                            "bomb1_anomaly_kinds": b1_kinds,
                            "bomb1_branch_kind": b1_kind,
                            "decision_bucket": v860().decision_bucket(decision),
                            "features_sha256_k12": feats_sha,
                            # the H = 600 child run's own bookkeeping (the bomb
                            # counter settles inside the first two frames, so it
                            # is horizon-independent; recorded once for audit)
                            "bombs_used": block["bombs_used"],
                            "bombs_start": block["bombs_start"],
                            "bombs_end": block["bombs_end"],
                            "bomb_decrement_frame": block["bomb_decrement_frame"],
                            "bomb1_presses_honoured": int(honoured),
                            "bomb1_presses_ignored": int(ignored),
                            "bomb1": {
                                "cands": [split_action(a)[0]
                                          for a in bomb1_actions],
                                "bombs_before": [branches[a]["before"]
                                                 for a in bomb1_actions],
                                "bombs_after": [branches[a]["after"]
                                                for a in bomb1_actions],
                                "spent_amount": [branches[a]["spent_amount"]
                                                 for a in bomb1_actions],
                                "alive_frames": [branches[a]["alive"]
                                                 for a in bomb1_actions],
                                "alive_frames_bomb0": [branches[a]["alive_bomb0"]
                                                       for a in bomb1_actions],
                                "decrement_frame": [branches[a]["decrement_frame"]
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
                                "any_noop": bool(noop_count > 0)},
                            "invulnerability": {
                                "source": ("STATE player_state (u8): 3 = "
                                           "PLAYER_STATE_INVULNERABLE"),
                                "player_state": player_state,
                                "player_state_name": invuln_name,
                                "invulnerable": player_state == 3},
                            "park_reply": block["park_reply"],
                            "child_replies_digest": block["child_replies_digest"],
                            "child_reply_lines": block["child_reply_lines"],
                            "children": block["children"],
                            "label_wall_s": block["label_wall_s"],
                            "obs": obs,
                        }
                        # ---- one row per horizon (the H sweep) --------------
                        for horizon_h in HORIZONS:
                            hblock = derived[int(horizon_h)]
                            row = dict(base_row)
                            row["H"] = int(horizon_h)
                            row["safe"] = hblock["safe"]
                            row["alive_frames"] = hblock["alive_frames"]
                            row["max_alive_frames"] = int(
                                hblock["max_alive_frames"])
                            row["safe_actions"] = int(hblock["safe_actions"])
                            row["critical"] = bool(
                                int(hblock["max_alive_frames"]) < int(horizon_h))
                            row["best_bomb0"] = int(hblock["best_bomb0"])
                            row["best_bomb1"] = int(hblock["best_bomb1"])
                            row["delta"] = int(hblock["best_bomb1"]
                                               - hblock["best_bomb0"])
                            row["bomb1_alive_at_H"] = [
                                int(hblock["alive_frames"][a])
                                for a in bomb1_actions]
                            row["bomb1_alive_bomb0_at_H"] = [
                                int(hblock["alive_frames"][a - 1])
                                for a in bomb1_actions]
                            row["terminal_counts_at_H"] = hblock["terminals"]
                            write_jsonl_line(handle, row)
                            rec["states_by_H"][str(int(horizon_h))] = (
                                rec["states_by_H"].get(str(int(horizon_h)), 0) + 1)
                        handle.flush()
                        rec["states"] += 1
                        ep["states_labelled"] += 1
                        rec["state_costs_s"].append(float(block["label_wall_s"]))
                        rec["rule_bomb_counts"][str(rule_bomb)] = (
                            rec["rule_bomb_counts"].get(str(rule_bomb), 0) + 1)
                        bkey = str(int(bombs_before))
                        if rule_bomb:
                            rec["positive_counts"][bkey] = (
                                rec["positive_counts"].get(bkey, 0) + 1)
                        rec["death_states_by_bombs"][bkey] = (
                            rec["death_states_by_bombs"].get(bkey, 0) + 1)
                        rec["decisions_before_death_counts"][str(before_death)] = (
                            rec["decisions_before_death_counts"].get(
                                str(before_death), 0) + 1)
                        if rec["states"] % 20 == 0:
                            log(f"  s{seed} w{window} ep{episode} d{decision}: "
                                f"{rec['states']}/{quota} states, bombs "
                                f"{rec['death_states_by_bombs']}, mean_label="
                                f"{statistics.fmean(rec['state_costs_s']):.3f}s")
                        if rec["states"] >= quota:
                            break
                    # ---- replay the recorded action ------------------------
                    if decision >= int(ep["death_decision"]):
                        break          # the death is inside this decision
                    first_mask, rest_mask = v830().action_masks(entry["action"])
                    replayed_died = False
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
                            replayed_died = True
                            break
                    if replayed_died:
                        break
                ep["pass2_s"] = round(time.perf_counter() - pass2_started, 3)
                rec["pass_costs_s"]["pass2"].append(ep["pass2_s"])
                ep["bombs_end"] = int(state["bombs_remaining"] or 0)
                ep["bombs_used"] = int(ep["bombs_start"]) - int(ep["bombs_end"])
                if ep["death_decision"] is not None:
                    death_decisions.append(int(ep["death_decision"]))
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
            else:
                rec["status"] = "short"
            if rec["states"] and rec["label_checks"]["bomb1_press_honoured"] <= 0 \
                    and rec["label_checks"]["states_without_a_bomb_left"] \
                    < rec["states"]:
                raise ConserveError(
                    f"w{window}: no bomb=1 label ever decremented the counter in "
                    f"{rec['states']} states although some had a bomb left — the "
                    f"bomb path is not working")
            if rec["label_checks"]["bomb0_spent_a_bomb"]:
                log(f"w{window}: WARNING "
                    f"{rec['label_checks']['bomb0_spent_a_bomb']} bomb=0 label(s) "
                    f"spent a bomb (counted as an anomaly, cell kept)")
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
            rec["pass1_cost_s"] = quantiles(rec["pass_costs_s"]["pass1"])
            rec["pass2_cost_s"] = quantiles(rec["pass_costs_s"]["pass2"])
            del rec["pass_costs_s"]
            rec["labels_total"] = int(rec["states"]) * N_ACTIONS
            rec["label_rows"] = int(rec["states"]) * len(HORIZONS)
            rec["labels_per_horizon"] = int(rec["states"]) * N_ACTIONS
            rec["death_decision_quantiles"] = quantiles(death_decisions)
            rec["death_episodes_expected_from_quota"] = round(
                rec["states"] / DEATH_WINDOW, 3)
            if rec["status"] in ("complete", "short", "skipped"):
                if labels.exists():
                    labels.unlink()
                partial.rename(labels)
                rec["labels_sha256"] = sha256_file(labels)
                rec["labels_bytes"] = labels.stat().st_size
            rec.pop("death_decisions", None)
            write_json(summary_path, rec)
        log(f"collect s{seed} w{window}: status={rec['status']} "
            f"states={rec['states']} rows={rec.get('label_rows')} "
            f"states_by_H={rec['states_by_H']} "
            f"death_episodes={rec['death_episodes']} "
            f"no_death={rec['no_death_episodes']} "
            f"bombs_at_death={rec['bombs_at_death_counts']} "
            f"wall={rec['wall_s']}s -> {labels}")
        if rec["status"] == "failed":
            return 4
    return 0


# --------------------------------------------------------------------------- #
# mode: stats (pool + the P0 numbers)
# --------------------------------------------------------------------------- #
def mode_stats(args: argparse.Namespace) -> int:
    """Pool the cells and run the P0a / P0b screen **once per horizon**.

    For every H in ``HORIZONS`` (120 / 240 / 600) the four numbers the work
    order asks for:

    1. ``critical_fraction`` = share of states with ``max(alive_frames) < H``;
    2. of those, ``diff_fraction_of_critical`` = share with
       ``best bomb=1 > best bomb=0`` (the gate, >= 20 %);
    3. the ``bombs_remaining`` segments (3 / 2 / 1 / 0), each with its own
       state count, critical count and diff count;
    4. the ``bombs_at_death`` distribution (the ``bombs_remaining`` at the
       decision whose frames contained the death), from the cell summaries.

    The verdict is ``S-nodiff`` iff **no** horizon reaches the diff gate;
    otherwise the **smallest** horizon that reaches it is the one to train with.
    """

    root = pathlib.Path(args.work_dir)
    cells = sorted((root / "labels").glob("seed*_w*.jsonl"))
    if not cells:
        raise ConserveError(f"no cell label files under {root / 'labels'}")
    rows: list[dict[str, Any]] = []
    per_cell: list[dict[str, Any]] = []
    for path in cells:
        cell_rows = read_jsonl(path)
        rows.extend(cell_rows)
        states_seen = {(int(row["seed"]), int(row["window"]),
                        int(row["episode"]), int(row["decision"]))
                       for row in cell_rows}
        per_cell.append({
            "file": str(path), "sha256": sha256_file(path),
            "bytes": path.stat().st_size, "rows": len(cell_rows),
            "states": len(states_seen),
            "seed": int(cell_rows[0]["seed"]) if cell_rows else None,
            "window": int(cell_rows[0]["window"]) if cell_rows else None,
            "episodes": len({int(row["episode"]) for row in cell_rows}),
            "by_H": {str(h): sum(1 for row in cell_rows
                                 if int(row["H"]) == int(h))
                     for h in HORIZONS},
            "positives": sum(1 for row in cell_rows if int(row["rule_bomb"])),
        })
    pooled = pooled_labels_path(root)
    with pooled.open("w", encoding="utf-8") as handle:
        for row in rows:
            write_jsonl_line(handle, row)

    # ---- collapse the three per-horizon rows of a state ---------------------
    by_state: dict[tuple[int, int, int, int], dict[int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        horizon = int(row["H"])
        if horizon not in HORIZONS:
            raise ConserveError(f"row {index}: H={horizon} is not one of "
                                f"{list(HORIZONS)}")
        key = (int(row["seed"]), int(row["window"]), int(row["episode"]),
               int(row["decision"]))
        slot = by_state.setdefault(key, {})
        if horizon in slot:
            raise ConserveError(f"row {index}: state {key} has two H={horizon} "
                                f"rows")
        slot[horizon] = row
    incomplete = [key for key, slot in by_state.items()
                  if set(slot) != set(HORIZONS)]
    if incomplete:
        raise ConserveError(f"{len(incomplete)} states do not carry all "
                            f"{list(HORIZONS)} horizons (first {incomplete[0]})")

    per_h: dict[int, list[dict[str, Any]]] = {int(h): [] for h in HORIZONS}
    nesting_mismatches = 0
    for key, slot in by_state.items():
        alive_max = [int(value) for value in slot[H_LABEL_MAX]["alive_frames"]]
        for horizon in HORIZONS:
            row = slot[int(horizon)]
            horizon = int(horizon)
            alive = [int(value) for value in row["alive_frames"]]
            safe = [int(value) for value in row["safe"]]
            if len(alive) != N_ACTIONS or len(safe) != N_ACTIONS:
                raise ConserveError(f"state {key} H={horizon}: "
                                    f"{len(alive)}/{len(safe)} labels")
            if [min(value, horizon) for value in alive_max] != alive:
                nesting_mismatches += 1
                raise ConserveError(
                    f"state {key} H={horizon}: alive_frames is not the "
                    f"truncation of the H={H_LABEL_MAX} run")
            for action in range(N_ACTIONS):
                if (safe[action] == 1) != (alive[action] == horizon):
                    raise ConserveError(
                        f"state {key} H={horizon} a{action}: safe="
                        f"{safe[action]} alive_frames={alive[action]}")
            red = reduce_alive(alive, horizon=horizon)
            red.update({"seed": int(row["seed"]), "window": int(row["window"]),
                        "episode": int(row["episode"]),
                        "decision": int(row["decision"]),
                        "bombs_remaining": int(row["bombs_remaining"]),
                        "rule_bomb": int(row["rule_bomb"]),
                        "decisions_before_death": int(
                            row["decisions_before_death"]),
                        "frames_before_death": int(row["frames_before_death"]),
                        "is_death_decision": bool(row["is_death_decision"]),
                        "nearest_all_px": row["nearest_all_px"]})
            per_h[horizon].append(red)

    def block_of(subset: Sequence[dict[str, Any]], horizon: int) -> dict[str, Any]:
        critical = [entry for entry in subset if entry["critical"]]
        diff_critical = [entry for entry in critical if entry["delta"] > 0]
        diff_all = [entry for entry in subset if entry["delta"] > 0]
        armed_states = [entry for entry in subset if entry["bombs_remaining"] > 0]
        diff_armed = [entry for entry in armed_states if entry["delta"] > 0]
        armed = len(armed_states)
        return {
            "H": int(horizon),
            "states": len(subset),
            "critical": len(critical),
            "critical_fraction": (round(len(critical) / len(subset), 6)
                                  if subset else None),
            "diff_states_of_critical": len(diff_critical),
            "diff_fraction_of_critical": (round(len(diff_critical) / len(critical), 6)
                                          if critical else None),
            "diff_states_all": len(diff_all),
            "diff_fraction_all": (round(len(diff_all) / len(subset), 6)
                                  if subset else None),
            # ---- P0a / P0b: the armed states (prereg §2) ------------------
            "armed": armed,
            "armed_fraction": (round(armed / len(subset), 6) if subset else None),
            "diff_states_of_armed": len(diff_armed),
            "diff_fraction_of_armed": (round(len(diff_armed) / armed, 6)
                                       if armed else None),
            "rule_positives": sum(1 for entry in subset if entry["rule_bomb"]),
            "rule_positive_fraction": (round(
                sum(1 for entry in subset if entry["rule_bomb"]) / len(subset), 6)
                if subset else None),
            "all_36_safe_states": sum(1 for entry in subset
                                      if entry["all_36_safe"]),
            "all_36_dead_states": sum(1 for entry in subset
                                      if entry["n_safe"] == 0),
            "mean_alive_frames_over_36": (round(statistics.fmean(
                entry["mean_alive"] for entry in subset), 4) if subset else None),
            "max_alive_frames": quantiles([entry["max_alive"] for entry in subset]),
            "best_bomb0": quantiles([entry["best_bomb0"] for entry in subset]),
            "best_bomb1": quantiles([entry["best_bomb1"] for entry in subset]),
            "delta_alive_frames": quantiles([entry["delta"] for entry in subset]),
            "bomb1_better": sum(1 for entry in subset if entry["delta"] > 0),
            "bomb1_equal": sum(1 for entry in subset if entry["delta"] == 0),
            "bomb1_worse": sum(1 for entry in subset if entry["delta"] < 0),
            "critical_at_death_decision": sum(1 for entry in critical
                                              if entry["is_death_decision"]),
            "states_at_death_decision": sum(1 for entry in subset
                                            if entry["is_death_decision"]),
        }

    by_h: dict[str, Any] = {}
    verdict_horizons: list[int] = []
    for horizon in HORIZONS:
        horizon = int(horizon)
        states = per_h[horizon]
        overall = block_of(states, horizon)
        by_bombs = {str(bombs): block_of(
            [entry for entry in states if entry["bombs_remaining"] == bombs],
            horizon) for bombs in BOMB_SEGMENTS}
        by_proximity = {}
        for label, lo, hi in (("0", 0, 0), ("1-4", 1, 4), ("5-9", 5, 9),
                              ("10-19", 10, 19)):
            by_proximity[label] = block_of(
                [entry for entry in states
                 if lo <= entry["decisions_before_death"] <= hi], horizon)
        # ---- P0a: are the sampled states still armed? (prereg §2) --------
        armed_fraction = overall["armed_fraction"] or 0.0
        p0a_met = bool(armed_fraction >= P0A_ARMED_MIN)
        # ---- P0b: of the armed ones, does a bomb buy survival? -----------
        diff_armed = overall["diff_fraction_of_armed"] or 0.0
        p0b_met = bool(int(overall["armed"]) > 0
                       and diff_armed >= P0B_DIFF_MIN)
        # v8.16.0's own precondition, kept in the record for continuity but
        # **not** gating this round (prereg §2 gates on P0a / P0b only)
        critical_fraction = overall["critical_fraction"] or 0.0
        critical_met = bool(critical_fraction >= P0_CRITICAL_MIN)
        if p0a_met and p0b_met:
            verdict_horizons.append(horizon)
        by_h[str(horizon)] = {
            "H": horizon,
            "states": len(states),
            "p0": {
                "armed": int(overall["armed"]),
                "armed_fraction": overall["armed_fraction"],
                "armed_fraction_gate": P0A_ARMED_MIN,
                "p0a_met": p0a_met,
                "diff_states_of_armed": int(overall["diff_states_of_armed"]),
                "diff_fraction_of_armed": overall["diff_fraction_of_armed"],
                "diff_fraction_gate": P0B_DIFF_MIN,
                "p0b_met": p0b_met,
                "pass": bool(p0a_met and p0b_met),
                "critical_fraction_context": overall["critical_fraction"],
                "critical_met_context": critical_met,
                "diff_fraction_of_critical_context":
                    overall["diff_fraction_of_critical"],
            },
            "overall": overall,
            "by_bombs_remaining": by_bombs,
            "by_decisions_before_death": by_proximity,
        }

    # ---- the death-end bombs (from the cell summaries, never per-row) -------
    bombs_at_death: dict[str, int] = {}
    cell_blocks: list[dict[str, Any]] = []
    for path in sorted((root / "collect").glob("seed*_w*.json")):
        cell = read_json(path)
        cell_blocks.append({
            "file": str(path), "seed": cell.get("seed"),
            "window": cell.get("window"), "status": cell.get("status"),
            "states": cell.get("states"), "rows": cell.get("label_rows"),
            "death_episodes": cell.get("death_episodes"),
            "no_death_episodes": cell.get("no_death_episodes"),
            "window_skipped_after": cell.get("window_skipped_after"),
            "skip_reason": cell.get("skip_reason"),
            "decisions_cap": cell.get("decisions_cap"),
            "states_by_H": cell.get("states_by_H"),
            "bombs_at_death_counts": cell.get("bombs_at_death_counts"),
            "death_decision_quantiles": cell.get("death_decision_quantiles"),
            "label_cost_s": cell.get("label_cost_s"),
            "nesting": cell.get("nesting_verified"),
            "wall_s": cell.get("wall_s"),
            "labels_sha256": cell.get("labels_sha256"),
            "labels_bytes": cell.get("labels_bytes"),
            "error": cell.get("error"),
            "label_checks": {key: cell.get("label_checks", {}).get(key)
                             for key in ("labels_36", "bomb1_press_honoured",
                                         "bomb1_noop_due_to_invuln",
                                         "bomb1_anomalies", "bomb0_spent_a_bomb",
                                         "states_without_a_bomb_left",
                                         "bomb1_spent_two_or_more",
                                         "states_by_player_state")},
        })
        for key, count in (cell.get("bombs_at_death_counts") or {}).items():
            bombs_at_death[str(key)] = bombs_at_death.get(str(key), 0) + int(count)

    merged_label_checks: dict[str, Any] = {}
    for cell in cell_blocks:
        for key in ("labels_36", "bomb1_press_honoured",
                    "bomb1_noop_due_to_invuln", "bomb1_anomalies",
                    "bomb0_spent_a_bomb", "states_without_a_bomb_left"):
            merged_label_checks[key] = (merged_label_checks.get(key, 0)
                                        + int(cell["label_checks"].get(key) or 0))
        for key in ("death_episodes", "no_death_episodes"):
            merged_label_checks[key] = (merged_label_checks.get(key, 0)
                                        + int(cell.get(key) or 0))
    total_states = len(by_state)
    # the rule the cells were collected with (every cell must agree; a mixed pool
    # would make P0a/P0b incomparable, so it is a fail-closed error)
    rules_seen = sorted({str(cell.get("bomb_rule")) for cell in cell_blocks
                         if cell.get("bomb_rule")})
    if len(rules_seen) > 1:
        raise ConserveError(f"the pooled cells used more than one bomb rule: "
                            f"{rules_seen}; P0a/P0b would not be comparable")
    bomb_rule = (rules_seen[0] if rules_seen
                 else str(getattr(args, "bomb_rule", BOMB_RULE_DEFAULT)))
    p0a_any = bool(any(by_h[str(int(h))]["p0"]["p0a_met"] for h in HORIZONS))
    if not p0a_any:
        verdict = "S-nobomb"
    elif verdict_horizons:
        verdict = "train"
    else:
        verdict = "S-nodiff"
    train_horizon = (min(verdict_horizons) if verdict_horizons else None)
    payload = {
        "schema": SCHEMA_STATS, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL,
        "requirement": ("P0a per horizon: >= 50 % of the sampled states must have "
                        "bombs_remaining > 0 (an armed state is the precondition "
                        "for a bomb to have any value); P0b: >= 20 % of those "
                        "armed states must have best bomb=1 > best bomb=0. "
                        "S-nobomb if P0a fails at every horizon, S-nodiff if P0a "
                        "holds but no horizon reaches the P0b gate"),
        "sampling": {"kind": "death-adjacent",
                     "window_decisions": DEATH_WINDOW,
                     "window_frames": DEATH_WINDOW_FRAMES,
                     "rule": ("the last N = 20 decisions before the death: "
                              "[death_decision - 19 .. death_decision]"),
                     "behaviour": ("v4.5.0 `S` stochastic + the round's bomb rule "
                                   "(see `bomb_rule`), rolled until death, cap "
                                   f"{DECISIONS} decisions "
                                   f"({DECISIONS * DECISION_FRAMES} frames); an "
                                   f"episode that does not die is discarded"),
                     "bomb_rule": bomb_rule,
                     "bomb_rule_description": BOMB_RULE_DESCRIPTION.get(
                         bomb_rule),
                     "bomb_rules_seen": sorted({str(row.get("rule"))
                                                for row in rows
                                                if row.get("rule") is not None}),
                     "windows": list(SAMPLE_WINDOWS),
                     "max_consecutive_no_death": MAX_CONSECUTIVE_NO_DEATH,
                     "horizons": list(HORIZONS),
                     "label_source": ("one 36-child run to H = 600; alive(H) = "
                                      "min(alive(600), H), verified in the smoke "
                                      "and in every cell with --verify-h 1")},
        "cells": per_cell, "cell_count": len(per_cell),
        "collect_cells": cell_blocks,
        "states": total_states, "label_rows": len(rows),
        "labels": total_states * N_ACTIONS,
        "labels_per_horizon": total_states * N_ACTIONS,
        "episodes": len({(int(entry[0]), int(entry[1]), int(entry[2]))
                         for entry in by_state}),
        "nesting_checks": {"states": total_states,
                           "mismatches": nesting_mismatches,
                           "alive_is_truncation": bool(nesting_mismatches == 0)},
        "definitions": {
            "armed": "bombs_remaining > 0 at the sampled decision (P0a)",
            "critical": ("max(alive_frames over the 36 actions) < H "
                         "(v8.16.0's precondition; context only, not a gate)"),
            "diff": ("max(alive_frames over the 18 bomb=1 branches) > "
                     "max(alive_frames over the 18 bomb=0 branches) of the same "
                     "state, at the same H (P0b, over the armed states)"),
            "p0a_armed_min": P0A_ARMED_MIN,
            "p0b_diff_min": P0B_DIFF_MIN,
            "p0_critical_min_context": P0_CRITICAL_MIN,
            "target_states_per_horizon": TARGET_STATES,
        },
        "by_H": by_h,
        "verdict": {
            "name": verdict,
            "p0a_gate": P0A_ARMED_MIN,
            "p0b_gate": P0B_DIFF_MIN,
            "horizons_reaching_gate": verdict_horizons,
            "train_horizon": train_horizon,
            "rule": ("the smallest H with P0a and P0b both met is the one to "
                     "train with; S-nobomb if P0a fails everywhere (re-collect "
                     "with --bomb-rule none); S-nodiff if P0a holds and no "
                     "horizon reaches the P0b gate"),
            "p0a_all_met": bool(all(by_h[str(int(h))]["p0"]["p0a_met"]
                                    for h in HORIZONS)),
            "p0_all_pass": bool(all(by_h[str(int(h))]["p0"]["pass"]
                                    for h in HORIZONS)),
        },
        "bombs_at_death": {
            "counts": bombs_at_death,
            "counts_sorted": dict(sorted(bombs_at_death.items(),
                                         reverse=True)),
            "episodes": sum(bombs_at_death.values()),
            "requirement": ("v8.16.0 measured {0: 3} (R5 spends every bomb "
                            "before the death); this round's rule must leave the "
                            "bomb in hand, so the counts must contain nonzero "
                            "keys"),
        },
        "pooled": {
            "bombs_remaining_counts": count_map(
                [entry["bombs_remaining"] for entry in per_h[int(H_DEFAULT)]]),
            "bombs_remaining_counts_by_H": {
                str(int(h)): count_map([entry["bombs_remaining"]
                                        for entry in per_h[int(h)]])
                for h in HORIZONS},
            "decisions_before_death_counts": count_map(
                [entry["decisions_before_death"]
                 for entry in per_h[int(H_DEFAULT)]]),
            "nearest_all_px": quantiles(
                [entry["nearest_all_px"] for entry in per_h[int(H_DEFAULT)]
                 if entry["nearest_all_px"] is not None]),
            "rule_bomb_counts": count_map(
                [entry["rule_bomb"] for entry in per_h[int(H_DEFAULT)]]),
            "death_decisions": quantiles(
                [entry["decision"] for entry in per_h[int(H_DEFAULT)]
                 if entry["is_death_decision"]]),
        },
        "label_checks": merged_label_checks,
        "labels_file": {"path": str(pooled), "sha256": sha256_file(pooled),
                        "bytes": pooled.stat().st_size, "rows": len(rows)},
        "reference_v815": V815_CRITICAL,
    }
    write_json(pooled_stats_path(root), payload)
    for horizon in HORIZONS:
        block = by_h[str(int(horizon))]
        log(f"stats H={int(horizon)}: P0a armed "
            f"{block['overall']['armed']}/{block['overall']['states']} = "
            f"{(block['p0']['armed_fraction'] or 0):.4f} (gate "
            f"{P0A_ARMED_MIN:.0%}) {'PASS' if block['p0']['p0a_met'] else 'FAIL'}"
            f"; P0b diff among armed {block['overall']['diff_states_of_armed']}/"
            f"{block['overall']['armed']} = "
            f"{(block['p0']['diff_fraction_of_armed'] or 0):.4f} (gate "
            f"{P0B_DIFF_MIN:.0%}) -> "
            f"{'PASS' if block['p0']['p0b_met'] else 'fail'}; critical(context) "
            f"{block['overall']['critical']}/{block['overall']['states']} = "
            f"{(block['p0']['critical_fraction_context'] or 0):.4f}; by bombs "
            + ", ".join(
                f"{bombs}={block['by_bombs_remaining'][str(bombs)]['armed']}/"
                f"{block['by_bombs_remaining'][str(bombs)]['states']}"
                f"(diff {block['by_bombs_remaining'][str(bombs)]['diff_states_of_armed']})"
                for bombs in BOMB_SEGMENTS))
    log(f"stats: {total_states} states, {len(rows)} rows from {len(per_cell)} "
        f"cells; bombs_at_death={payload['bombs_at_death']['counts_sorted']}; "
        f"verdict={verdict} train_horizon={train_horizon}")
    return 0 if verdict_horizons else 4


def load_pooled_rows(root: pathlib.Path) -> tuple[list[dict[str, Any]],
                                                 dict[str, Any]]:
    pooled = pooled_labels_path(root)
    if not pooled.is_file():
        raise ConserveError(f"no pooled labels: {pooled}")
    rows = read_jsonl(pooled)
    stats_rec = (read_json(pooled_stats_path(root))
                 if pooled_stats_path(root).is_file() else {})
    digest = sha256_file(pooled)
    info = {"path": str(pooled), "sha256": digest,
            "bytes": pooled.stat().st_size, "rows": len(rows),
            "recorded_sha256": (stats_rec.get("labels_file") or {}).get("sha256"),
            "sha256_match": digest == (stats_rec.get("labels_file") or {}).get(
                "sha256")}
    if not info["sha256_match"]:
        raise ConserveError(f"{pooled} sha256 {digest[:16]} != the recorded "
                            f"{str(info['recorded_sha256'])[:16]}")
    return rows, info


def bomb_marginal_loss(logits: Any, target: Any) -> Any:
    """BCE of the **bomb marginal** against the rule's bit (arm A's aux term).

    ``logits`` is the 36-way output of ``policy36`` (action = cand * 2 + bomb).
    The probability of *some* bomb action is the softmax mass on the 18 bomb=1
    outputs, so the aux term is a two-class cross-entropy on

        logit = logsumexp(logits[bomb = 1]) - logsumexp(logits[bomb = 0])

    against ``target`` (1 where R5 fires).  It never replaces the 36-way BCE; it
    is added with weight 0.5 (prereg §1, arm A only).
    """

    import torch

    bomb1 = [a for a in range(N_ACTIONS) if split_action(a)[1] == 1]
    bomb0 = [a for a in range(N_ACTIONS) if split_action(a)[1] == 0]
    margin = (torch.logsumexp(logits[:, bomb1], dim=1)
              - torch.logsumexp(logits[:, bomb0], dim=1))
    return torch.nn.functional.binary_cross_entropy_with_logits(margin, target)


def mode_train(args: argparse.Namespace) -> int:
    import torch

    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    arm = str(args.arm).upper()
    if seed not in TRAIN_SEEDS:
        raise ConserveError(f"--seed must be one of {TRAIN_SEEDS}, got {seed}")
    if arm not in ARMS:
        raise ConserveError(f"--arm must be one of {ARMS}, got {arm}")
    rows, info = load_pooled_rows(root)
    if not rows:
        raise ConserveError("no labelled rows")
    stats_rec = read_json(pooled_stats_path(root))
    verdict = stats_rec.get("verdict") or {}
    horizon = int(args.horizon or 0)
    if not horizon:
        horizon = int(verdict.get("train_horizon") or 0)
    if horizon not in HORIZONS:
        raise ConserveError(f"no horizon reached the P0a/P0b gates "
                            f"(verdict={verdict.get('name')}, horizons="
                            f"{verdict.get('horizons_reaching_gate')}); prereg "
                            f"§2.1 S-nobomb / S-nodiff - no training")
    horizon_block = (stats_rec.get("by_H") or {}).get(str(horizon)) or {}
    if not (horizon_block.get("p0") or {}).get("pass"):
        raise ConserveError(f"H={horizon} P0a/P0b does not pass "
                            f"({horizon_block.get('p0')}): no training")
    rows = [row for row in rows if int(row["H"]) == int(horizon)]
    if not rows:
        raise ConserveError(f"no H={horizon} rows in the pooled labels")
    log(f"train H={horizon}: {len(rows)} states (horizons passing P0a and P0b: "
        f"{verdict.get('horizons_reaching_gate')}; the smallest is used)")
    feats, labels, array_info = v830().build_arrays(rows)
    aux_weight = float(ARM_AUX_WEIGHT[arm])
    # arm A's aux target is **the label's own best-bomb decision** (this branch):
    # 1 iff the best bomb=1 branch outlives the best bomb=0 branch of that state at
    # this row's horizon - exactly the `diff` P0b counts.  `conserve_targets` keeps
    # the 25 px rule's bit for comparison and is never trained on.
    targets = torch.tensor([float(best_bomb_target(row)) for row in rows],
                           dtype=torch.float32)
    conserve_targets = torch.tensor(
        [float(aux_bomb_conserve(row.get("nearest_all_px"),
                                 int(row["bombs_remaining"]))) for row in rows],
        dtype=torch.float32)
    behaviour_targets = torch.tensor(
        [float(row.get("rule_bomb") or 0) for row in rows], dtype=torch.float32)
    aux_matches_behaviour = bool(int(targets.sum())
                                 == int(behaviour_targets.sum())
                                 and bool((targets == behaviour_targets).all()))
    bombs_counts = count_map([row["bombs_remaining"] for row in rows])
    log(f"train arm={arm} seed={seed}: {len(rows)} states x {N_ACTIONS} actions, "
        f"dim={FEATURE_DIM}, bombs={bombs_counts}, best-bomb positives="
        f"{int(targets.sum())}, 25px-rule positives="
        f"{int(conserve_targets.sum())}, behaviour positives="
        f"{int(behaviour_targets.sum())}, aux_weight={aux_weight}")

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
        rule_true_positive = int(((rule_predicted == 1)
                                  & (targets == 1)).sum())
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
        "feature_dim": int(FEATURE_DIM), "actions": int(N_ACTIONS),
        "bombs_remaining_distribution": bombs_counts,
        "aux_target_positive_states": int(targets.sum()),
        "aux_rule": {"name": ARM_AUX_TARGET_NAME,
                     "definition": ("1 iff best_bomb1 > best_bomb0 on the row's "
                                    "own H (the label's best-bomb decision)"),
                     "positive_states": int(targets.sum()),
                     "conserve25_positive_states": int(conserve_targets.sum()),
                     "conserve25_threshold_px": CONSERVE_THRESHOLD_PX,
                     "behaviour_positive_states": int(behaviour_targets.sum()),
                     "matches_behaviour_bits": aux_matches_behaviour,
                     "source": ("tools/rl/v8i0_nobomb.py::best_bomb_target (the "
                                "row's own best_bomb0 / best_bomb1; the 25 px "
                                "rule's bit is reported for comparison only)")},
        "model": {"class": "v431_arm_a.Policy with the output layer widened to 36",
                  "shape": f"Linear({FEATURE_DIM},{HIDDEN}) -> Tanh -> "
                           f"Linear({HIDDEN},36)",
                  "params": int(sum(p.numel() for p in policy.parameters())),
                  "source": "tools/rl/v830_bombpol.py::policy36 (imported)"},
        "labels_file": info, "features": array_info,
        "optimizer": {"name": "Adam", "lr": LR, "batch": batch, "epochs": EPOCHS,
                      "early_stopping": None, "shuffle": ("numpy default_rng(seed), "
                                                          "one permutation/epoch")},
        "loss": {"main": "BCEWithLogitsLoss over the 36 outputs",
                 "aux": ("bomb marginal BCE against the **best-bomb** bit "
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
                                "rule": ARM_AUX_TARGET_NAME,
                                "note": ("the aux target is the label's own "
                                         "best-bomb decision, not the behaviour's "
                                         "bit (which is 0 everywhere here)")},
        "ckpt": {"path": str(ckpt), "sha256": sha256_file(ckpt),
                 "bytes": ckpt.stat().st_size},
        "pins": check_pins(), "wall_s": wall, "deviations": [],
    }
    write_json(report_path, report)
    log(f"train arm={arm} seed={seed}: loss {report['loss_first_epoch']:.4f} -> "
        f"{report['loss_last_epoch']:.4f} (BCE {report['final_train_bce']:.4f}, "
        f"acc {overall_acc:.4f}, rule-acc {aux_acc:.4f}) wall={wall}s -> {ckpt}")
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
                "loss_first_epoch": report["loss_first_epoch"],
                "loss_last_epoch": report["loss_last_epoch"],
                "final_train_bce": report["final_train_bce"],
                "final_train_aux_bce": report["final_train_aux_bce"],
                "train_accuracy": report["train_accuracy"],
                "aux_bomb_agreement": report["aux_bomb_agreement"],
                "wall_s": report["wall_s"],
                "ckpt": report["ckpt"]}
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
# mode: eval (one budget x one seed x 4 dev windows)
# --------------------------------------------------------------------------- #
def load_policy36(ckpt: pathlib.Path, *, seed: int) -> Any:
    import torch

    path = pathlib.Path(ckpt)
    if not path.is_file():
        raise ConserveError(f"policy checkpoint not found: {path}")
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
        raise ConserveError(f"--policy-seed must be one of {TRAIN_SEEDS}, got "
                            f"{seed}")
    if arm not in ARMS:
        raise ConserveError(f"--arm must be one of {ARMS}, got {arm}")
    if budget not in EVAL_BUDGETS[arm]:
        raise ConserveError(f"prereg §1 fixes arm {arm} to budgets "
                            f"{EVAL_BUDGETS[arm]}, got {budget}")
    if int(args.decision_frames) != DECISION_FRAMES:
        raise ConserveError(f"prereg §1 keeps {DECISION_FRAMES} frames per "
                            f"decision")
    if int(args.episodes) < 1:
        raise ConserveError(f"--episodes must be >= 1, got {args.episodes}")
    windows = parse_windows(args.windows)
    for window in windows:
        if int(window) not in DEV_WINDOWS:
            raise ConserveError(f"window {window} is not a dev window "
                                f"{DEV_WINDOWS}")
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    tag = args.tag or f"v8i0-nobomb-eval-{arm}-d{budget}-s{seed}"
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
        "temperature": TEMPERATURE, "kind": args.kind,
        "engine": engine, "launch_env": launch_env,
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
            log(f"arm {arm} d{budget} s{seed} w{window}: cfg muted="
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
                record["schema"] = "v8170-conserve-eval-episode/1"
                record["producer"] = PRODUCER
                record["arm"] = arm
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
                    raise ConserveError(
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
                "survived_fraction": round(survived / len(rows), 6) if rows else None,
                "alive_frames": quantiles(alive),
                "frames": quantiles([float(row["frames"]) for row in rows]),
                "bomb_episodes": sum(1 for row in rows if row["bomb_emits"]),
                "bomb_emits_total": sum(int(row["bomb_emits"]) for row in rows),
                "bombs_used_total": sum(int(row["bombs_used"] or 0) for row in rows),
                "episodes_exhausted": sum(1 for row in rows
                                          if row["bombs_end"] == 0),
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
def eval_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


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
                        "test and the interval belong to the independent analysis "
                        "pass. No test, no interval in this file."),
        "note": ("the 12 (seed, window) cells are not independent samples; the "
                 "per-cell numbers are listed beside the pooled ones and must not "
                 "be averaged into a mean of means"),
        "arms": {}, "bomb_usage": {}, "references": {
            "v830_bombpol": BASELINE_V830, "v860_bombdag": BASELINE_V860,
            "rule_r5": RULE_R5},
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
                rows = eval_rows(path)
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
    # the side-by-side table (descriptive only)
    table: dict[str, Any] = {"v830": BASELINE_V830["rate"],
                             "v860": BASELINE_V860["rate"],
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
                        "episodes that spend only 1-2 bombs?"),
        "usage": payload["bomb_usage"]})
    for arm in ARMS:
        for budget in EVAL_BUDGETS[arm]:
            block = payload["arms"][arm].get(f"d{budget}", {})
            if not block:
                continue
            log(f"diagnostics arm {arm} d{budget}: "
                f"{block['survived_total']}/{block['episodes_total']} = "
                f"{(block['pooled_survived_fraction'] or 0):.4f}, bombs "
                f"{block['bomb_usage_pooled']['bombs_used_counts']}")
    return 0


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
        raise ConserveError(f"not a directory: {root}")
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
        (pooled_labels_path(root), "labels_nobomb.jsonl"),
        (pooled_stats_path(root), "nobomb_stats.json"),
        (root / "recall_v816.json", "nobomb_recall_v816.json"),
        (root / "preflight.json", "nobomb_preflight.json"),
        (root / "ckpt_assertions.json", "nobomb_ckpt_assertions.json"),
        (root / "ckpt_assertions_final.json", "nobomb_ckpt_assertions_final.json"),
        (root / "fidelity_d3.json", "nobomb_fidelity_d3.json"),
        (root / "train_summary.json", "nobomb_train_summary.json"),
        (root / "diagnostics.json", "nobomb_diagnostics.json"),
        (root / "bomb_usage.json", "nobomb_bomb_usage.json"),
        (root / "audio_audit.json", "nobomb_audio_audit.json"),
        (root / "process_residue.json", "nobomb_process_residue.json"),
        (root / "commands.tsv", "nobomb_commands.tsv"),
        (root / "work_manifest.json", "nobomb_work_manifest.json"),
    ]
    for name in ("exec.sh", "RUN.md"):
        wanted.append((root / name, f"nobomb_{name}"))
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            wanted.append((train_report_path(root, arm, seed),
                           train_report_path(root, arm, seed).name))
            wanted.append((policy_ckpt_path(root, arm, seed),
                           f"ckpt_{arm}_nobomb_seed{seed}.pt"))
        for budget in EVAL_BUDGETS[arm]:
            for seed in TRAIN_SEEDS:
                path = eval_path(root, arm, budget, seed)
                wanted.append((path, path.name))
                wanted.append((path.with_suffix(".summary.json"),
                               path.with_suffix(".summary.json").name))
    for path in sorted((root / "collect").glob("seed*_w*.json")):
        wanted.append((path, f"collect/nobomb_{path.name}"))
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
               "v8i0_nobomb_sha256": sha256_file(
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
        description="v8.17.0 CONSERVE-001: sample the last 20 decisions before a "
                    "death under a conservative bomb rule and label them 36 ways")
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "assert-ckpt", "recall-v816",
                                 "collect", "stats", "train", "train-summary",
                                 "eval", "diagnostics", "audio-audit",
                                 "process-audit", "manifest", "publish"))
    parser.add_argument("--bomb-rule", default=BOMB_RULE_DEFAULT,
                        choices=BOMB_RULES,
                        help=("collect: the behaviour policy's bomb rule. "
                              "conserve25 (prereg §1: only nearest_all_px < 25) "
                              "| none (prereg §2.1 S-nobomb: never bomb, the "
                              "player dies with all three) | staged (v8.16.0's R5, "
                              "for continuity)"))
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
                             "smallest H reaching the diff gate); collect must "
                             "leave it 0 or give H_max")
    parser.add_argument("--verify-h", type=int, default=0,
                        help="collect: 1 = verify H=120/240 against direct runs "
                             "on the first labelled state of the cell")
    parser.add_argument("--window-decisions", type=int, default=DEATH_WINDOW)
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
    parser.add_argument("--labels-file", default="")
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

    engine_modes = ("collect", "eval")
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
        if args.mode == "recall-v816":
            return mode_recall(args)
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
    except ConserveError as exc:
        log(f"FAILED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
