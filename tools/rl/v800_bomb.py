#!/usr/bin/env python3
"""v8.0.0 BOMB-001 — how many alive frames is a bomb worth?  A **script rule**, not learning.

Prereg: ``Document/Update/Alpha/v8.0.0/PREREGISTRATION_BOMB_001.md`` (FROZEN
2026-11-22, before any v8.0.0 result).

What this file is
-----------------
The executor for prereg §1's single-factor design.  The 18 candidates of the
frozen protocol are ``cand = focus * 9 + move`` with *shooting always on and
bombing always off* (``touhou_rl/v400/gym/actions.py:58-75``).  ``cand_to_mask``
already supports ``bomb=1``, so "allow the policy to bomb" needs no new action
space here — it needs one **rule**:

===========  ==========================================================
arm          bomb rule
===========  ==========================================================
``A0``       never bomb (``--bomb-rule none``; = the current protocol)
``A1``       bomb when the nearest bullet is closer than **40 px**
``A2``       ... **60 px**
``A3``       ... **80 px** (``--bomb-rule r40|r60|r80``)
``L1``       v8.2.0: bomb only when the nearest **lethal** bullet is
             **<= 3 px** away (``--bomb-rule lethal3``)
``L2``       ... **<= 5 px** (``--bomb-rule lethal5``)
``R0``       v8.4.0 control: the 80 px rule (= ``A3``, ``--bomb-rule r80``)
``R1``       conserve: 80 px, but **120 px** while ``bombs_remaining == 1``
``R2``       delay200: **no bomb at all** while ``decision < 200``, then 80 px
``R3``       double60: only when **>= 2 bullets** are within **60 px**
``R4``       wall: nearest **<= 80 px** **and** ``x < 60`` or ``x > 340``
``R5``       staged: 3 bombs -> 80 px, 2 -> 60 px, 1 -> 40 px
===========  ==========================================================

v8.2.0 (prereg ``Document/Update/Alpha/v8.2.0/PREREGISTRATION_BOMBLETH_001.md``)
------------------------------------------------------------------------------
The *only* v8.2.0 change is the two ``lethal3`` / ``lethal5`` rules and three
record fields.  A lethal rule tests ``nearest_lethal_px`` (the minimum over
``BulletRow.lethal`` bullets) instead of ``nearest_all_px``; the threshold test,
the "first frame of the decision, later frames bomb=0" mask schedule, the
``bombs_remaining > 0`` gate and every metric are unchanged.  Records keep the
v8.0.0 names and semantics (``emit_nearest_px``, ``emit_nearest_lethal_px``,
``decisions_below_threshold``, ``decisions_below_threshold_lethal``) and add
``trigger_distance`` (which distance the rule tested),
``decisions_below_threshold_all`` and ``decisions_below_threshold_trigger``.
For ``none``/``r40``/``r60``/``r80`` every pre-existing field keeps exactly its
v8.0.0 value (measured by the 60-episode A/B, ``Document/.../v8.2.0``).
``--arm`` may now name the arm directly (``--arm L3 --bomb-rule r80``), which is
how the v8.2.0 control arm is labelled; without ``--arm`` the label is derived
from the rule exactly as before.

v8.4.0 (prereg ``Document/Update/Alpha/v8.4.0/PREREGISTRATION_BOMBRULE2_001.md``)
--------------------------------------------------------------------------------
The *only* v8.4.0 change is the five rule variants of prereg §1 and the record
fields needed to audit them (they are additive; every v8.0.0/v8.2.0 field keeps
its name and — for the six pre-existing rules — its exact value, measured by the
60-episode A/B on ``--bomb-rule r80 --decisions 600``):

* ``conserve`` (R1): base 80 px; while ``bombs_remaining == 1`` the threshold is
  widened to **120 px** (the last bomb is only spent when it is that urgent);
* ``delay200`` (R2): **no bomb while ``decision < 200``**, then the 80 px rule;
* ``double60`` (R3): only when **>= 2 bullets** are inside **60 px**;
* ``wall`` (R4): nearest **<= 80 px** *and* the player is pinned to a side wall
  (``x < 60`` or ``x > 340``);
* ``staged`` (R5): the threshold is chosen by which bomb this would be —
  ``bombs_remaining == 3`` -> 80 px, ``2`` -> 60 px, ``1`` -> 40 px.

The distance test, the ``bombs_remaining > 0`` gate, the "first frame of the
decision, later frames ``bomb=0``" mask schedule and every metric are unchanged;
``rule_gate`` is the single place the per-decision threshold/gates are resolved.
Records add ``bomb_rule_params``, ``trigger_threshold_px`` (the threshold in
effect at each emit), ``emit_player_x``, ``emit_bullets_within_60``,
``decisions_gated_out`` and ``decisions_multi_within_60``; the trace rows used by
``--mode smoke`` add ``threshold_px`` / ``player_x`` / ``bullets_within_60`` /
``gated_out``.

Per decision (prereg §1): read the child's ``HAZARDS``, take the minimum
Euclidean distance from ``PLAYER`` to the bullet list, and if the rule fires and
``bombs_remaining > 0`` the decision's **first frame** is sent as
``cand_to_mask(cand, bomb=1)``; every later frame of that decision is
``cand_to_mask(cand, bomb=0)``.  A0 therefore reproduces the frozen evaluation
protocol exactly — and ``--mode equiv`` proves that on the engine rather than by
assertion (see below).

Evaluation protocol (prereg §1, unchanged from v4.5.0 §2.3)
----------------------------------------------------------
The ``S`` checkpoints of v4.5.0 (``.tmp_v450_cf/ckpt/seed{k}/round2.pt``), the
4 dev windows ``(11, 12, 17, 18)``, PARK at frame **8663**, **300 decisions /
600 frames**, **300 episodes per (seed, window)**, ``stochastic`` =
``softmax(z / 1.0)`` sampled by ``torch.multinomial`` from one
``torch.Generator`` seeded ``seed * 1000 + 7`` — v4.5.0's own RNG rule, imported
as :func:`v450_cf_label.policy_action` and :func:`v450_cf_label.load_policy`.

Modes
-----
``--mode smoke``      prereg §1's **smoke first**: 4 episodes of one arm, with a
                      per-decision trace (nearest distance, ``bombs_remaining``,
                      whether that decision emitted ``bomb=1``).  Writes
                      ``bomb_audit.json``'s ``smoke`` block **before** any full
                      run, so "the bomb path never fired" can never be discovered
                      after 14,400 episodes of work.
``--mode eval``       one (arm, seed): 4 dev windows x ``--episodes`` episodes ->
                      ``eval_{arm}_seed{k}.jsonl`` + ``.summary.json``.
``--mode equiv``      **engine-level equivalence control**: N episodes through
                      ``v432_fork_arm.run_episode`` (the frozen v4.5.0 path) and N
                      through this file's loop with ``--bomb-rule none``, both
                      from a freshly seeded generator; the episode fields and the
                      18-candidate action sequences must be identical.  This is
                      what makes A0 an honest control instead of a different code
                      path.
``--mode audit``      §2's bomb audit, offline, over the JSONL files of one arm
                      plus the smoke block: emits, decrements, ``bombs_used``,
                      exhaustion, and every invariant that proves the bomb really
                      fired.  Merges into the existing ``bomb_audit.json`` and
                      never touches another round's audit file.
``--mode audio-audit``decode every ``th08.cfg`` **this round** created
                      (``v450_cf_label.mode_audit``, own ``--out``).
``--mode diagnostics``offline counts/means/medians per arm, per (seed, window)
                      and pooled; bomb usage; **no statistics** — no test, no
                      p-value, no significance claim (prereg §5).
``--mode assert-ckpt``the three frozen v4.5.0 ``round2.pt`` sha256 + engine sha +
                      the protected sources, ``--phase start`` and ``--phase
                      final``.
``--mode process-audit``own engine processes still alive (``--data-dir`` filter);
                      must be 0.
``--mode manifest``   sha256 + size of every published file.

Red lines this file keeps (imported, never re-implemented)
---------------------------------------------------------
* observation ``tools/rl/v431_arm_a.py::build_features`` (106 dims) and model
  ``v431_arm_a.Policy`` — imported through ``v450_cf_label``;
* the stochastic action ``v450_cf_label.policy_action`` (``softmax(z / 1.0)``,
  ``torch.multinomial``) and ``v450_cf_label.load_policy`` — imported;
* the parked parent, ``SPAWN``/``CHILD``/``QUIT`` transport, the muted private
  data dir, the 2,165-checkpoint prefix walk and the per-window fidelity gate —
  ``v450_cf_label.prepare_env`` / ``prepare_data_dir`` / ``start_parent``, which
  use ``tools/rl/v432_fork_arm.py`` unchanged;
* ``STATE`` / ``STEP`` / ``HAZARDS`` decoding — ``v432_fork_arm._step_fields``,
  ``v450_cf_label``'s ``child_state``/``child_hazards`` and
  ``touhou_rl/v400/gym/protocol.py::parse_hazards_reply``;
* death semantics ``touhou_rl/v420/counterfactual.py::_terminal_of`` — a first
  death is the frame where ``globals.deaths`` increments, and the reported offset
  *excludes* that frame, so ``frames_executed == offset + 1``;
* the mask table ``touhou_rl/v420/contracts.py::cand_to_mask(cand, bomb)``
  (frozen v4.0.0 ``gym/actions.py`` table).

:func:`run_episode_bomb` is ``v432_fork_arm.run_episode``'s loop **copied** —
that file may not be modified (this round's hard constraint) — with exactly one
behavioural difference: the per-decision mask sequence.  ``--mode equiv`` measures
that the copy is equivalent when ``--bomb-rule none``, instead of promising it.

Not in this file
----------------
* No learning, no update, no checkpoint output (deterministic zero-training round).
* No statistics (prereg §2's P1/P2 live in the analysis pass, not here).
* No pixel arm (``PARK`` refuses under the GL path), no hyper-parameter search.
* No claim of "the policy learned to bomb": the bomb is an **external rule**.
"""

from __future__ import annotations

import argparse
import hashlib
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

WORK = REPO / ".tmp_v800_bomb"
CKPT_ROOT = REPO / ".tmp_v450_cf" / "ckpt"

SCHEMA_EVAL = "v800-bomb-eval/1"
SCHEMA_EVAL_SUMMARY = "v800-bomb-eval-summary/1"
SCHEMA_AUDIT = "v800-bomb-audit/1"
SCHEMA_SMOKE = "v800-bomb-smoke/1"
SCHEMA_DIAG = "v800-bomb-diagnostics/1"
SCHEMA_MANIFEST = "v800-bomb-manifest/1"
SCHEMA_PROC = "v800-bomb-process-audit/1"
SCHEMA_ASSERT = "v800-bomb-ckpt-assertions/1"
SCHEMA_EQUIV = "v800-bomb-equivalence/1"
PRODUCER = "tools/rl/v800_bomb.py (live engine run, HarnessClient role=shadow)"

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
ENGINE_SHA256 = "691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- frozen geometry / split (prereg §1: identical to v4.5.0's dev evaluation)
PREFIX_FRAMES = 8663
TARGET_FRAME = 8664
DEV_WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
N_CAND = 18
DECISIONS = 300
DECISION_FRAMES = 2
HIDDEN = 64
TEMPERATURE = 1.0
EPISODES = 300
DEFAULT_TIMEOUT = 300.0
GEN_CONST = 7                     # v4.5.0: torch.Generator().manual_seed(seed*1000+7)

# -- prereg §1's single factor: the bomb rule
BOMB_RULES: dict[str, float | None] = {"none": None, "r40": 40.0, "r60": 60.0,
                                       "r80": 80.0,
                                       # v8.2.0 BOMBLETH-001 §1: L1 / L2
                                       "lethal3": 3.0, "lethal5": 5.0,
                                       # v8.4.0 BOMBRULE2-001 §1: R0..R5; the value
                                       # is the rule's **base** threshold and
                                       # ``rule_gate`` resolves it per decision
                                       "conserve": 80.0, "delay200": 80.0,
                                       "double60": 60.0, "wall": 80.0,
                                       "staged": 80.0}
# -- v8.2.0: these rules test ``nearest_lethal_px``, every other rule tests
#    ``nearest_all_px`` (v8.0.0's literal prereg reading, untouched)
LETHAL_RULES: frozenset[str] = frozenset({"lethal3", "lethal5"})

# -- v8.4.0 BOMBRULE2-001 §1: the five rule variants, one constant per parameter
RULE2_RULES: frozenset[str] = frozenset({"conserve", "delay200", "double60",
                                         "wall", "staged"})
CONSERVE_LAST_BOMB_PX = 120.0     # R1: bombs_remaining == 1 -> widen to 120 px
DELAY200_FIRST_DECISION = 200     # R2: never bomb while decision < 200
DOUBLE60_PX = 60.0                # R3: "within 60 px"
DOUBLE60_COUNT = 2               # R3: ">= 2 bullets"
WALL_LEFT_PX = 60.0               # R4: x < 60 ...
WALL_RIGHT_PX = 340.0             # R4: ... or x > 340
STAGED_THRESHOLDS: dict[int, float] = {3: 80.0, 2: 60.0, 1: 40.0}   # R5
# -- the static parameters of each v8.4.0 rule, written into every record so the
#    audit never has to guess which numbers were in force
BOMB_RULE_PARAMS: dict[str, dict[str, Any]] = {
    "conserve": {"base_px": 80.0, "last_bomb_px": CONSERVE_LAST_BOMB_PX,
                 "note": "bombs_remaining == 1 -> 120 px"},
    "delay200": {"base_px": 80.0, "first_decision": DELAY200_FIRST_DECISION,
                 "note": "no bomb while decision < 200, then 80 px"},
    "double60": {"threshold_px": DOUBLE60_PX, "min_bullets": DOUBLE60_COUNT,
                 "note": ">= 2 bullets within 60 px"},
    "wall": {"base_px": 80.0, "left_px": WALL_LEFT_PX, "right_px": WALL_RIGHT_PX,
             "note": "nearest <= 80 px AND (x < 60 or x > 340)"},
    "staged": {"thresholds": {"3": 80.0, "2": 60.0, "1": 40.0},
               "note": "3 bombs -> 80 px, 2 -> 60 px, 1 -> 40 px"},
}

ARMS: dict[str, str] = {"A0": "none", "A1": "r40", "A2": "r60", "A3": "r80",
                        # v8.2.0 arms; L3 is the 80 px control of v8.0.0 §1
                        "L1": "lethal3", "L2": "lethal5", "L3": "r80",
                        # v8.4.0 arms; R0 is this round's 80 px control
                        "R0": "r80", "R1": "conserve", "R2": "delay200",
                        "R3": "double60", "R4": "wall", "R5": "staged"}
# -- v8.0.0 / v8.1.0's arm list; every loop below still iterates exactly this
ARM_ORDER: tuple[str, ...] = ("A0", "A1", "A2", "A3")
# -- v8.2.0's arms, separate so the v8.0.0 modes keep their exact behaviour
LETHAL_ARM_ORDER: tuple[str, ...] = ("L1", "L2", "L3")
# -- v8.4.0's arms, separate for the same reason
RULE2_ARM_ORDER: tuple[str, ...] = ("R0", "R1", "R2", "R3", "R4", "R5")
SEEDS: tuple[int, ...] = (0, 1, 2)

# -- the three frozen v4.5.0 ``S`` checkpoints this round evaluates
FROZEN_ROUND2_SHA256: dict[int, str] = {
    0: "4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc",
    1: "a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4",
    2: "c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01",
}
PROTECTED_FILES = (
    "tools/rl/v432_fork_arm.py",
    "tools/rl/v450_cf_label.py",
    "tools/rl/v431_arm_a.py",
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)
FROZEN_V450_EVAL = "eval/seed{seed}/eval_stochastic_dev.jsonl"

_LOG_FH: Any = None
_M450: Any = None
_M432: Any = None


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class BombError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v800] {message}"
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
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BombError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def m450():
    """``tools/rl/v450_cf_label.py`` — imported, never modified."""

    global _M450
    if _M450 is None:
        _M450 = load_module("v450_cf_label",
                            REPO / "tools" / "rl" / "v450_cf_label.py")
        _M450._LOG_FH = _LOG_FH
    return _M450


def m432():
    """``tools/rl/v432_fork_arm.py`` — imported, never modified."""

    global _M432
    if _M432 is None:
        _M432 = load_module("v432_fork_arm", REPO / "tools" / "rl" / "v432_fork_arm.py")
        _M432._LOG_FH = _LOG_FH
    return _M432


def arm_a():
    return m450().arm_a()


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_line(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def quantiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None,
                "min": None, "max": None}
    data = sorted(float(v) for v in values)
    if len(data) >= 2:
        q1, _q2, q3 = statistics.quantiles(data, n=4, method="inclusive")
    else:
        q1 = q3 = data[0]
    return {"n": len(data), "mean": round(statistics.fmean(data), 4),
            "median": round(statistics.median(data), 4), "q1": round(q1, 4),
            "q3": round(q3, 4), "min": round(data[0], 4), "max": round(data[-1], 4)}


def engine_gate(path: str) -> dict[str, Any]:
    from touhou_rl.v400.teacher.engine import check_engine_binary

    gate = dict(check_engine_binary(path, expected=ENGINE_SHA256))
    if not gate.get("match"):
        raise BombError(f"engine sha256 mismatch: {gate}")
    return gate


def rule_threshold(bomb_rule: str) -> float | None:
    if bomb_rule not in BOMB_RULES:
        raise BombError(f"unknown --bomb-rule {bomb_rule!r}; "
                        f"expected one of {sorted(BOMB_RULES)}")
    return BOMB_RULES[bomb_rule]


def trigger_is_lethal(bomb_rule: str) -> bool:
    """v8.2.0 §1: does this rule test the **lethal** minimum, or v8.0.0's all?

    ``lethal3`` / ``lethal5`` test ``nearest_lethal_px`` (only ``BulletRow.lethal``
    bullets); ``none`` / ``r40`` / ``r60`` / ``r80`` keep testing
    ``nearest_all_px``, which is why their behaviour is bit-identical to v8.0.0.
    """

    rule_threshold(bomb_rule)
    return bomb_rule in LETHAL_RULES


def resolve_arm(bomb_rule: str, requested: str) -> str:
    """The arm label (``--arm`` if given, else derived from the rule)."""

    if requested:
        known = ARMS.get(requested)
        if known is None:
            raise BombError(f"unknown --arm {requested!r}; expected one of "
                            f"{sorted(ARMS)}")
        if known != bomb_rule:
            raise BombError(f"--arm {requested} is {known!r}, not {bomb_rule!r}")
        return requested
    return arm_of(bomb_rule)


def arm_of(rule: str) -> str:
    # v8.0.0's order first, so ``r80`` still resolves to ``A3`` exactly as before
    for arm in ARM_ORDER:
        if ARMS[arm] == rule:
            return arm
    for arm in LETHAL_ARM_ORDER:
        if ARMS[arm] == rule:
            return arm
    # v8.4.0: ``r80`` never reaches here, so ``R0`` must be named with ``--arm``
    for arm in RULE2_ARM_ORDER:
        if ARMS[arm] == rule:
            return arm
    raise BombError(f"rule {rule!r} belongs to no arm")


def rule_gate(bomb_rule: str, *, decision: int, bombs_before: int,
              player_x: float) -> dict[str, Any]:
    """v8.4.0 §1: the per-decision threshold and the non-distance gates.

    This is the **only** place the five v8.4.0 rules differ from each other; the
    caller still applies the distance test (``< threshold``, ``<=`` for ``wall``,
    prereg §1's wording), the ``bombs_remaining > 0`` gate and the mask schedule.

    For ``none`` / ``r40`` / ``r60`` / ``r80`` / ``lethal3`` / ``lethal5`` this
    function is never called: those rules keep v8.0.0/v8.2.0's single comparison
    (which is what makes the A/B on ``--bomb-rule r80`` meaningful).
    """

    if bomb_rule not in RULE2_RULES:
        raise BombError(f"rule_gate called for non-v8.4.0 rule {bomb_rule!r}")
    threshold = rule_threshold(bomb_rule)
    decision_ok = True
    position_ok = True
    if bomb_rule == "conserve":
        # R1: 80 px by default; with the last bomb left, wait until 120 px
        if int(bombs_before) == 1:
            threshold = CONSERVE_LAST_BOMB_PX
    elif bomb_rule == "delay200":
        # R2: never bomb before decision 200, then the 80 px rule
        decision_ok = int(decision) >= DELAY200_FIRST_DECISION
    elif bomb_rule == "double60":
        # R3: 60 px; the ">= 2 bullets" half of the rule is the caller's count
        threshold = DOUBLE60_PX
    elif bomb_rule == "wall":
        # R4: 80 px, but only with the player pinned to a side wall
        position_ok = bool(float(player_x) < WALL_LEFT_PX
                           or float(player_x) > WALL_RIGHT_PX)
    elif bomb_rule == "staged":
        # R5: 3 bombs -> 80 px, 2 -> 60 px, 1 -> 40 px (bombs_remaining says
        # which bomb this would be)
        threshold = STAGED_THRESHOLDS.get(int(bombs_before), threshold)
    return {"threshold_px": threshold, "decision_ok": decision_ok,
            "position_ok": position_ok}



def parse_windows(text: str) -> tuple[int, ...]:
    if not str(text).strip():
        return DEV_WINDOWS
    values = tuple(int(part) for part in str(text).split(",") if part.strip())
    for window in values:
        if int(window) not in DEV_WINDOWS:
            raise BombError(f"window {window} is not a dev window {DEV_WINDOWS}")
    return values


def generator_seed(seed: int) -> int:
    """v4.5.0 §2.3's rule, verbatim: ``torch.Generator().manual_seed(seed*1000+7)``."""

    return int(seed) * 1000 + GEN_CONST


# --------------------------------------------------------------------------- #
# prereg §1: the bomb rule's distance measurement
# --------------------------------------------------------------------------- #
def nearest_bullet_distances(state: dict[str, Any], hazards: Any) -> dict[str, Any]:
    """Minimum Euclidean distance (px, 2-D playfield) from the player to a bullet.

    ``all`` is the literal prereg §1 reading — *the* ``HAZARDS`` bullet list — and
    is what the rule tests.  ``lethal`` restricts the same minimum to bullets that
    can actually kill (``BulletRow.lethal``: fired and collision not disabled); it
    is recorded on every decision so the two readings can be compared offline
    without re-running anything.
    """

    px = float(state.get("player_x") or 0.0)
    py = float(state.get("player_y") or 0.0)
    best_all: float | None = None
    best_lethal: float | None = None
    lethal_count = 0
    within_60 = 0
    for bullet in hazards.bullets:
        dx = float(bullet.position[0]) - px
        dy = float(bullet.position[1]) - py
        distance = math.hypot(dx, dy)
        if best_all is None or distance < best_all:
            best_all = distance
        if distance < DOUBLE60_PX:
            # v8.4.0 R3's ">= 2 bullets within 60 px"; recorded, not acted on,
            # by every other rule (so their behaviour is unchanged)
            within_60 += 1
        if getattr(bullet, "lethal", False):
            lethal_count += 1
            if best_lethal is None or distance < best_lethal:
                best_lethal = distance
    return {"bullets": len(hazards.bullets), "lethal": lethal_count,
            "nearest_all_px": None if best_all is None else round(best_all, 4),
            "nearest_lethal_px": None if best_lethal is None else round(best_lethal, 4),
            "bullets_within_60": within_60}


# --------------------------------------------------------------------------- #
# one episode: v432_fork_arm.run_episode's loop + the per-decision bomb frame
# --------------------------------------------------------------------------- #
def run_episode_bomb(parent: Any, *, mode: str, arm: str, bomb_rule: str,
                     episode: int, policy_seed: int, decisions: int,
                     decide: Any, decision_frames: int = DECISION_FRAMES,
                     keep_trace: bool = False) -> dict[str, Any]:
    """One ``SPAWN -> decisions x decision_frames frames -> QUIT`` episode.

    The single difference from ``v432_fork_arm.run_episode`` is the mask sent on
    each frame of a decision::

        frame 0        cand_to_mask(cand, 1)   if the rule fired and bombs > 0
        frame 0        cand_to_mask(cand, 0)   otherwise
        frames 1..k-1  cand_to_mask(cand, 0)   always

    Everything else — the spawn, the baseline ``STATE``, the death rule
    (``counterfactual._terminal_of``), the 300-decision cap, the reward shape
    (1 per surviving decision, 0 on the death decision), the ``survived``
    definition, ``first_death_offset = frames_at_death - 1`` and the mandatory
    ``QUIT`` — is the frozen loop, line for line.
    """

    from touhou_rl.v420 import counterfactual as cf
    from touhou_rl.v420 import contracts as C

    _threshold = rule_threshold(bomb_rule)
    _test_lethal = trigger_is_lethal(bomb_rule)
    _is_rule2 = bomb_rule in RULE2_RULES
    started = time.perf_counter()
    record: dict[str, Any] = {
        "schema": SCHEMA_EVAL, "kind": "episode", "mode": mode, "arm": arm,
        "bomb_rule": bomb_rule, "threshold_px": _threshold,
        # -- v8.2.0: which distance the rule tests, and both below-threshold
        #    readings under their own names (v8.0.0's two names keep their values)
        "trigger_distance": ("nearest_lethal_px" if _test_lethal
                             else "nearest_all_px"),
        "episode": int(episode), "policy_seed": int(policy_seed),
        "window": int(parent.window), "decisions_cap": int(decisions),
        "decision_frames": int(decision_frames), "return_decisions": 0.0,
        "survived": False, "wall_s": None, "frames": 0, "alive_frames": None,
        "first_death_offset": None, "first_death_frame": None, "terminal": None,
        "decisions_executed": 0, "invalid": False, "error": None,
        "spawn_ms": None, "template": parent.template,
        # -- the bomb audit of this episode (prereg §2 P3)
        "bombs_start": None, "bombs_end": None, "bombs_used": None,
        "bomb_emits": 0, "bomb_decisions": [], "bomb_frames": [],
        "emit_remaining_before": [], "emit_remaining_after_first_frame": [],
        "emit_remaining_after_last_frame": [], "emit_decremented": [],
        "emit_latency_frames": [],
        "emit_nearest_px": [], "emit_nearest_lethal_px": [],
        "decisions_below_threshold": 0, "decisions_below_threshold_lethal": 0,
        "decisions_below_threshold_all": 0,
        "decisions_below_threshold_trigger": 0,
        "min_nearest_px": None, "min_nearest_lethal_px": None,
        "decisions_measured": 0, "trace": [] if keep_trace else None,
        # -- v8.4.0's audit fields (additive; the six pre-existing rules keep the
        #    v8.0.0/v8.2.0 values of every field above)
        "bomb_rule_params": BOMB_RULE_PARAMS.get(bomb_rule),
        "trigger_threshold_px": [], "emit_player_x": [],
        "emit_bullets_within_60": [],
        "decisions_gated_out": 0, "decisions_multi_within_60": 0,
        "decisions_position_ok": 0,
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
            raise BombError(
                f"spawned child starts at frame {state['frame']}, expected "
                f"{PREFIX_FRAMES}")

        died_at: int | None = None
        terminal: str | None = None
        frames = 0
        base_mask_cache: dict[int, int] = {}
        for decision in range(int(decisions)):
            hazards = parent.child_hazards(handle)
            distances = nearest_bullet_distances(state, hazards)
            bombs_before = int(state["bombs_remaining"] or 0)
            cand = int(decide(state, hazards, decision))
            if not 0 <= cand < N_CAND:
                raise BombError(f"decision {decision}: cand {cand} outside [0,18)")
            if cand not in base_mask_cache:
                base_mask_cache[cand] = int(C.cand_to_mask(cand))
                base_mask_cache[-cand - 1] = int(C.cand_to_mask(cand, 1))
            base_mask = base_mask_cache[cand]
            bomb_mask = base_mask_cache[-cand - 1]

            nearest = distances["nearest_all_px"]
            nearest_lethal = distances["nearest_lethal_px"]
            trigger_px = nearest_lethal if _test_lethal else nearest
            record["decisions_measured"] += 1
            if nearest is not None:
                record["min_nearest_px"] = (nearest if record["min_nearest_px"] is None
                                            else min(record["min_nearest_px"], nearest))
            if nearest_lethal is not None:
                record["min_nearest_lethal_px"] = (
                    nearest_lethal if record["min_nearest_lethal_px"] is None
                    else min(record["min_nearest_lethal_px"], nearest_lethal))
            if _threshold is not None and nearest is not None and nearest < _threshold:
                # v8.2.0: the all-bullets reading, recorded under its own name
                record["decisions_below_threshold_all"] += 1
            if (nearest_lethal is not None and _threshold is not None
                    and nearest_lethal < _threshold):
                record["decisions_below_threshold_lethal"] += 1
            fired = False
            threshold_now = _threshold
            gate_decision_ok = True
            gate_position_ok = True
            gate_count_ok = True
            if _is_rule2:
                # v8.4.0: the rule's per-decision threshold and its extra gates
                gate = rule_gate(bomb_rule, decision=int(decision),
                                 bombs_before=bombs_before,
                                 player_x=float(state.get("player_x") or 0.0))
                threshold_now = gate["threshold_px"]
                gate_decision_ok = bool(gate["decision_ok"])
                gate_position_ok = bool(gate["position_ok"])
                gate_count_ok = bool(distances["bullets_within_60"] >= DOUBLE60_COUNT
                                     if bomb_rule == "double60" else True)
                if gate_position_ok:
                    record["decisions_position_ok"] += 1
                if not (gate_decision_ok and gate_position_ok and gate_count_ok):
                    record["decisions_gated_out"] += 1
                if int(distances["bullets_within_60"]) >= DOUBLE60_COUNT:
                    record["decisions_multi_within_60"] += 1
                distance_ok = False
                if nearest is not None and threshold_now is not None:
                    distance_ok = ((nearest <= threshold_now) if bomb_rule == "wall"
                                   else (nearest < threshold_now))
                if gate_decision_ok and gate_position_ok and gate_count_ok \
                        and distance_ok:
                    # the rule asked for a bomb on this decision (whether or not
                    # one was left); ``decisions_below_threshold`` keeps v8.0.0's
                    # "the rule fired" meaning for the rules that have one
                    record["decisions_below_threshold"] += 1
                    record["decisions_below_threshold_trigger"] += 1
                    fired = bombs_before > 0
            elif _threshold is not None and trigger_px is not None:
                if trigger_px < _threshold:
                    # v8.0.0 name and semantics: the distance *this rule* tests.
                    # For none/r40/r60/r80 that is nearest_all_px -> unchanged.
                    record["decisions_below_threshold"] += 1
                    record["decisions_below_threshold_trigger"] += 1
                    fired = bombs_before > 0

            emit = bool(fired)
            if emit:
                record["bomb_emits"] += 1
                record["bomb_decisions"].append(int(decision))
                record["bomb_frames"].append(int(frames))
                record["emit_remaining_before"].append(bombs_before)
                record["emit_nearest_px"].append(nearest)
                record["emit_nearest_lethal_px"].append(nearest_lethal)
                record["trigger_threshold_px"].append(threshold_now)
                record["emit_player_x"].append(
                    round(float(state.get("player_x") or 0.0), 4))
                record["emit_bullets_within_60"].append(
                    int(distances["bullets_within_60"]))
            if keep_trace:
                record["trace"].append(
                    {"decision": int(decision), "frame": int(state["frame"]),
                     "cand": cand, "nearest_px": nearest,
                     "nearest_lethal_px": nearest_lethal,
                     "bombs_remaining": bombs_before, "emitted": emit,
                     "bullets": distances["bullets"],
                     "threshold_px": threshold_now,
                     "player_x": round(float(state.get("player_x") or 0.0), 4),
                     "bullets_within_60": int(distances["bullets_within_60"]),
                     "gated_out": bool(_is_rule2 and not (
                         gate_decision_ok and gate_position_ok and gate_count_ok))})

            died_here = False
            emit_before = bombs_before
            emit_last: int | None = None
            emit_first: int | None = None
            emit_decrement_at: int | None = None
            for frame_index in range(int(decision_frames)):
                mask = bomb_mask if (emit and frame_index == 0) else base_mask
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
                    if frame_index == 0:
                        emit_first = remaining_now
                    if emit_decrement_at is None and remaining_now < emit_before:
                        emit_decrement_at = frame_index + 1
                if term is not None:
                    terminal = str(term.get("terminal"))
                    record["terminal_reply"] = reply
                    if term.get("engine_dead"):
                        died_at = frames
                    died_here = True
                    break
            if emit:
                record["emit_remaining_after_first_frame"].append(emit_first)
                record["emit_remaining_after_last_frame"].append(emit_last)
                record["emit_decremented"].append(
                    bool(emit_last is not None and emit_last < emit_before))
                record["emit_latency_frames"].append(emit_decrement_at)
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
# mode: smoke (prereg §1: prove the bomb path fires, before the full run)
# --------------------------------------------------------------------------- #
def mode_smoke(args: argparse.Namespace) -> int:
    import torch

    from touhou_rl.v420 import contracts as C

    V450 = m450()
    rule = str(args.bomb_rule)
    threshold = rule_threshold(rule)
    arm = resolve_arm(rule, str(getattr(args, "arm", "") or ""))
    seed = int(args.policy_seed)
    windows = parse_windows(args.windows)[:1]
    window = windows[0]
    episodes = int(args.episodes)
    if episodes < 1:
        raise BombError("--episodes must be >= 1")

    engine = engine_gate(args.engine)
    launch_env = V450.prepare_env()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = args.tag or f"smoke-{arm}-s{seed}-{stamp}"
    run_dir = pathlib.Path(args.work_dir) / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = pathlib.Path(args.work_dir) / "data" / tag
    out_path = (pathlib.Path(args.out) if args.out
                else pathlib.Path(args.work_dir) / "bomb_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else (CKPT_ROOT / f"seed{seed}"
                                                     / "round2.pt")
    if not ckpt.is_file():
        raise BombError(f"checkpoint not found: {ckpt}")
    digest = sha256_file(ckpt)
    expected = FROZEN_ROUND2_SHA256.get(seed)
    if digest != expected:
        raise BombError(f"seed {seed}: {ckpt} sha256 {digest} != the frozen "
                        f"round2 {expected}; refusing to run (checklist item 2)")
    policy = V450.load_policy(ckpt, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    A = arm_a()

    def decide(state: dict[str, Any], hazards: Any, decision: int) -> int:
        feats, _detail = A.build_features(state=state, hazards=hazards,
                                         decision_index=decision)
        return int(V450.policy_action(policy, feats, "stochastic", generator))

    rec: dict[str, Any] = {
        "schema": SCHEMA_SMOKE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": "Document/Update/Alpha/v8.0.0/PREREGISTRATION_BOMB_001.md §1",
        "requirement": ("prereg §1 smoke: before the 14,400-episode run, 4 episodes "
                        "of the widest rule must show that PLAYER.bombs_remaining "
                        "really decreases because a decision emitted BTN_BOMB"),
        "mode": "smoke", "arm": arm, "bomb_rule": rule, "threshold_px": threshold,
        "trigger_distance": ("nearest_lethal_px" if trigger_is_lethal(rule)
                             else "nearest_all_px"),
        "seed": seed, "window": int(window), "episodes": episodes,
        "decisions_cap": int(args.decisions), "decision_frames": int(args.decision_frames),
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest, "expected": expected,
                 "match": digest == expected},
        "sampler": {"kind": "torch.multinomial", "temperature": TEMPERATURE,
                    "generator_seed": generator_seed(seed)},
        "rows": [], "status": "failed", "error": None, "wall_s": None,
    }
    parents: list[Any] = []
    started = time.perf_counter()
    try:
        C.assert_seed_allowed(int(window), "train")
        data_dir, data_info = V450.prepare_data_dir(data_root, window, args.data_source)
        rec["data"] = {"data_dir": str(data_dir), "cfg": data_info["cfg"]}
        parent, gate = V450.start_parent(slot=0, window=window, engine=args.engine,
                                        data_dir=data_dir, log_dir=run_dir)
        parents.append(parent)
        rec["gate"] = {key: gate[key] for key in (
            "hello_role_shadow", "park_ok", "park_frame", "threads_1", "all_pass")}
        rec["gate_prefix"] = gate["prefix"]
        for episode in range(episodes):
            record = run_episode_bomb(
                parent, mode="smoke", arm=arm, bomb_rule=rule, episode=episode,
                policy_seed=seed, decisions=int(args.decisions), decide=decide,
                decision_frames=int(args.decision_frames), keep_trace=True)
            rec["rows"].append(record)
            if record.get("invalid"):
                raise BombError(f"smoke episode {episode} is invalid: "
                                f"{record.get('error')}")
            log(f"smoke {arm} s{seed} w{window} ep {episode}: frames="
                f"{record['frames']} bombs {record['bombs_start']}->"
                f"{record['bombs_end']} emits={record['bomb_emits']} "
                f"first_emit_decision="
                f"{record['bomb_decisions'][0] if record['bomb_decisions'] else None}")
        rec["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["status"] = "failed"
        log(f"SMOKE FAILED {arm} {rule}: {rec['error']}")
    finally:
        for parent in parents:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        rec["wall_s"] = round(time.perf_counter() - started, 3)

    rows = rec["rows"]
    emits = sum(int(row["bomb_emits"]) for row in rows)
    decremented = sum(sum(1 for flag in row["emit_decremented"] if flag) for row in rows)
    used = sum(int(row["bombs_used"] or 0) for row in rows)
    latencies = [int(value) for row in rows
                 for value in row["emit_latency_frames"] if value is not None]
    first_emit = next((row["bomb_decisions"][0] for row in rows
                       if row["bomb_decisions"]), None)
    first_emit_honoured = bool(
        rows and rows[0]["bomb_decisions"]
        and rows[0]["emit_decremented"]
        and rows[0]["emit_decremented"][0])
    out_row = {"schema": SCHEMA_SMOKE, "producer": PRODUCER,
               "created_at": rec["created_at"], "prereg": rec["prereg"],
               "requirement": rec["requirement"], "arm": arm, "bomb_rule": rule,
               "threshold_px": threshold,
               "trigger_distance": rec["trigger_distance"],
               "seed": seed, "window": int(window),
               "episodes": episodes, "engine": engine, "launch_env": launch_env,
               "ckpt": rec["ckpt"], "sampler": rec["sampler"],
               "gate": rec.get("gate"), "gate_prefix": rec.get("gate_prefix"),
               "cfg": rec.get("data", {}).get("cfg"),
               "status": rec["status"], "error": rec["error"],
               "wall_s": rec["wall_s"],
               "checks": {
                   "episodes": len(rows),
                   "bomb_emits_total": emits,
                   "emits_that_decremented_in_their_own_decision": decremented,
                   "emits_ignored_while_a_bomb_was_active": emits - decremented,
                   "bombs_used_total": used,
                   "bombs_start_total": sum(int(row["bombs_start"] or 0) for row in rows),
                   "bombs_end_total": sum(int(row["bombs_end"] or 0) for row in rows),
                   "decrement_latency_frames": {
                       "values": sorted(set(latencies)),
                       "min": min(latencies) if latencies else None,
                       "max": max(latencies) if latencies else None,
                       "note": ("frames after the BTN_BOMB frame at which "
                                "bombs_remaining first drops; measured, not assumed")},
                   "first_emit_decision_of_episode_0": (
                       rows[0]["bomb_decisions"][0]
                       if rows and rows[0]["bomb_decisions"] else None),
                   "first_emit_decision_any_episode": first_emit,
                   "first_emit_decremented": first_emit_honoured,
                   "bomb_path_fired": bool(emits > 0),
                   "bombs_remaining_decreased": bool(used > 0),
                   "counter_bookkeeping_closes": bool(emits > 0 and used == decremented),
                   "all_pass": bool(rec["status"] == "complete" and emits > 0
                                    and used > 0 and used == decremented
                                    and first_emit_honoured),
               },
               "episodes_detail": rows}
    write_json(out_path, out_row)
    # a durable copy of the smoke block: ``--mode audit`` merges the two, and a
    # later audit must never be the reason the smoke evidence disappears
    smoke_copy = pathlib.Path(args.work_dir) / "smoke" / "bomb_smoke.json"
    smoke_copy.parent.mkdir(parents=True, exist_ok=True)
    write_json(smoke_copy, out_row)
    out_row["smoke_copy"] = str(smoke_copy)
    log(f"smoke: emits={emits} decremented={decremented} used={used} "
        f"first_emit_decision={first_emit} -> {out_path}")
    return 0 if out_row["checks"]["all_pass"] else 7


# --------------------------------------------------------------------------- #
# mode: eval (one arm x one seed x 4 dev windows)
# --------------------------------------------------------------------------- #
def mode_eval(args: argparse.Namespace) -> int:
    import torch

    from touhou_rl.v420 import contracts as C

    V450 = m450()
    rule = str(args.bomb_rule)
    threshold = rule_threshold(rule)
    arm = resolve_arm(rule, str(getattr(args, "arm", "") or ""))
    seed = int(args.policy_seed)
    windows = parse_windows(args.windows)
    episodes = int(args.episodes)
    if int(args.decisions) < 1:
        raise BombError(f"--decisions must be >= 1, got {args.decisions}")
    if int(args.decision_frames) != DECISION_FRAMES:
        raise BombError(f"prereg §1 keeps {DECISION_FRAMES} frames per decision, "
                        f"got --decision-frames {args.decision_frames}")

    engine = engine_gate(args.engine)
    launch_env = V450.prepare_env()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = args.tag or f"eval-{arm}-s{seed}-{stamp}"
    run_dir = pathlib.Path(args.work_dir) / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = pathlib.Path(args.work_dir) / "data" / tag
    out_path = (pathlib.Path(args.out) if args.out
                else pathlib.Path(args.work_dir) / f"eval_{arm}_seed{seed}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else (CKPT_ROOT / f"seed{seed}"
                                                     / "round2.pt")
    if not ckpt.is_file():
        raise BombError(f"checkpoint not found: {ckpt}")
    digest = sha256_file(ckpt)
    expected = FROZEN_ROUND2_SHA256.get(seed)
    if digest != expected:
        raise BombError(f"seed {seed}: {ckpt} sha256 {digest} != the frozen "
                        f"round2 {expected}; refusing to run (checklist item 2)")
    policy = V450.load_policy(ckpt, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    A = arm_a()

    def decide(state: dict[str, Any], hazards: Any, decision: int) -> int:
        feats, _detail = A.build_features(state=state, hazards=hazards,
                                         decision_index=decision)
        return int(V450.policy_action(policy, feats, "stochastic", generator))

    summary: dict[str, Any] = {
        "schema": SCHEMA_EVAL_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": "Document/Update/Alpha/v8.0.0/PREREGISTRATION_BOMB_001.md §1",
        "mode": "eval", "arm": arm, "bomb_rule": rule, "threshold_px": threshold,
        "trigger_distance": ("nearest_lethal_px" if trigger_is_lethal(rule)
                             else "nearest_all_px"),
        "policy_seed": seed, "windows": list(windows),
        "episodes_per_window": episodes, "decisions_cap": int(args.decisions),
        "decision_frames": DECISION_FRAMES, "temperature": TEMPERATURE,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest, "expected": expected,
                 "match": digest == expected},
        "sampler": {"kind": "torch.multinomial", "generator_seed": generator_seed(seed),
                    "generator": "one torch.Generator per (arm, seed) process, "
                                 "windows in the frozen order 11,12,17,18"},
        "out": str(out_path), "per_window": {}, "status": "failed", "error": None,
        "wall_s": None,
    }
    parents: list[Any] = []
    handle = out_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    try:
        for slot, window in enumerate(windows):
            C.assert_seed_allowed(int(window), "train")
            data_dir, data_info = V450.prepare_data_dir(data_root, window,
                                                        args.data_source)
            log(f"{arm} s{seed} w{window}: cfg muted="
                f"{data_info['cfg']['muted_four_zero']} "
                f"sha256={data_info['cfg']['sha256'][:16]}…")
            parent, gate = V450.start_parent(slot=slot, window=window,
                                             engine=args.engine, data_dir=data_dir,
                                             log_dir=run_dir)
            parents.append(parent)
            rows: list[dict[str, Any]] = []
            for episode in range(episodes):
                record = run_episode_bomb(
                    parent, mode="eval", arm=arm, bomb_rule=rule, episode=episode,
                    policy_seed=seed, decisions=int(args.decisions), decide=decide,
                    decision_frames=DECISION_FRAMES)
                record.update({"ckpt_sha256": digest,
                               "generator_seed": generator_seed(seed)})
                record.pop("terminal_reply", None)
                record.pop("trace", None)
                record.pop("spawn_reply", None)
                write_line(handle, record)
                rows.append(record)
                if record.get("invalid"):
                    raise BombError(f"{arm} seed {seed} window {window} episode "
                                    f"{episode} is invalid: {record.get('error')}")
                if (episode + 1) % 50 == 0:
                    done = sum(1 for row in rows if row["survived"])
                    used = sum(int(row["bombs_used"] or 0) for row in rows)
                    log(f"  {arm} s{seed} w{window}: {episode + 1}/{episodes} "
                        f"episodes, survived={done}, bombs_used={used}")
            alive = [float(row["alive_frames"]) for row in rows]
            survived = sum(1 for row in rows if row["survived"])
            summary["per_window"][str(window)] = {
                "window": int(window), "arm": arm, "episodes": len(rows),
                "survived": survived,
                "survived_fraction": round(survived / len(rows), 6) if rows else None,
                "alive_frames": quantiles(alive),
                "frames": quantiles([float(row["frames"]) for row in rows]),
                "return_decisions": quantiles(
                    [float(row["return_decisions"]) for row in rows]),
                "bomb_emits_total": sum(int(row["bomb_emits"]) for row in rows),
                "bombs_used_total": sum(int(row["bombs_used"] or 0) for row in rows),
                "episodes_that_bombed": sum(1 for row in rows if row["bomb_emits"]),
                "episodes_exhausted": sum(1 for row in rows
                                          if row["bombs_end"] == 0),
                "cfg_muted": data_info["cfg"]["muted_four_zero"],
                "cfg_sha256": data_info["cfg"]["sha256"],
                "data_dir": str(data_dir),
                "gate": {key: gate[key] for key in (
                    "hello_role_shadow", "park_ok", "park_frame", "threads_1",
                    "all_pass")},
                "gate_prefix": gate["prefix"],
            }
            log(f"{arm} seed {seed} window {window}: survived {survived}/{len(rows)} "
                f"mean_alive={summary['per_window'][str(window)]['alive_frames']['mean']}")
        summary["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["status"] = "failed"
        log(f"EVAL FAILED {arm} seed {seed}: {summary['error']}")
    finally:
        handle.close()
        for parent in parents:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        summary["wall_s"] = round(time.perf_counter() - started, 3)
        episodes_total = sum(int(v["episodes"]) for v in summary["per_window"].values())
        survived_total = sum(int(v["survived"]) for v in summary["per_window"].values())
        summary["episodes_total"] = episodes_total
        summary["survived_total"] = survived_total
        summary["pooled_survived_fraction"] = (
            round(survived_total / episodes_total, 6) if episodes_total else None)
        summary["bomb_emits_total"] = sum(
            int(v["bomb_emits_total"]) for v in summary["per_window"].values())
        summary["bombs_used_total"] = sum(
            int(v["bombs_used_total"]) for v in summary["per_window"].values())
        if out_path.is_file():
            summary["out_sha256"] = sha256_file(out_path)
            summary["out_bytes"] = out_path.stat().st_size
        write_json(summary_path, summary)
    log(f"eval {arm} ({rule}) seed {seed}: status={summary['status']} "
        f"survived={summary.get('survived_total')}/{summary.get('episodes_total')} "
        f"emits={summary.get('bomb_emits_total')} wall={summary['wall_s']}s -> "
        f"{out_path}")
    return 0 if summary["status"] == "complete" else 4


# --------------------------------------------------------------------------- #
# mode: equiv (the A0 control really is the frozen v4.5.0 protocol)
# --------------------------------------------------------------------------- #
EQUIV_FIELDS = ("frames", "alive_frames", "first_death_offset", "first_death_frame",
                "return_decisions", "decisions_executed", "survived", "terminal")


def mode_equiv(args: argparse.Namespace) -> int:
    """Same ckpt, same seeds, same window: frozen ``run_episode`` vs this loop."""

    import torch

    from touhou_rl.v420 import contracts as C

    V450 = m450()
    m = m432()
    seed = int(args.policy_seed)
    window = parse_windows(args.windows)[0]
    episodes = int(args.episodes)
    engine = engine_gate(args.engine)
    launch_env = V450.prepare_env()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = args.tag or f"equiv-s{seed}-{stamp}"
    run_dir = pathlib.Path(args.work_dir) / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = pathlib.Path(args.work_dir) / "data" / tag
    out_path = (pathlib.Path(args.out) if args.out
                else pathlib.Path(args.work_dir) / "equiv_check.json")

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else (CKPT_ROOT / f"seed{seed}"
                                                     / "round2.pt")
    digest = sha256_file(ckpt)
    expected = FROZEN_ROUND2_SHA256.get(seed)
    if digest != expected:
        raise BombError(f"seed {seed}: ckpt sha256 {digest} != frozen {expected}")
    policy = V450.load_policy(ckpt, seed=seed)
    A = arm_a()

    frozen_actions: list[list[int]] = []
    record: dict[str, Any] = {
        "schema": SCHEMA_EQUIV, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("A0 must be the frozen v4.5.0 evaluation protocol, not a "
                        "second implementation: v432_fork_arm.run_episode and "
                        "v800_bomb.run_episode_bomb(--bomb-rule none) must produce "
                        "identical episode fields and identical action sequences"),
        "seed": seed, "window": int(window), "episodes": episodes,
        "ckpt": {"path": str(ckpt), "sha256": digest},
        "engine": engine, "launch_env": launch_env,
        "frozen_rows": [], "bomb_rows": [], "status": "failed", "error": None,
        "wall_s": None,
    }
    parents: list[Any] = []
    started = time.perf_counter()
    try:
        C.assert_seed_allowed(int(window), "train")
        data_dir, data_info = V450.prepare_data_dir(data_root, window, args.data_source)
        record["cfg"] = data_info["cfg"]
        parent, gate = V450.start_parent(slot=0, window=window, engine=args.engine,
                                        data_dir=data_dir, log_dir=run_dir)
        parents.append(parent)

        # ---- pass 1: the frozen loop, exactly as v4.5.0 --mode eval calls it ----
        gen = torch.Generator()
        gen.manual_seed(generator_seed(seed))
        actions: list[list[int]] = []
        for episode in range(episodes):
            current: list[int] = []

            def decide(state: dict[str, Any], hazards: Any, decision: int,
                       _current: list[int] = current) -> tuple[int, Any]:
                feats, _detail = A.build_features(state=state, hazards=hazards,
                                                 decision_index=decision)
                cand = int(V450.policy_action(policy, feats, "stochastic", gen))
                _current.append(cand)
                return cand, None

            row = m.run_episode(parent, mode="eval", episode=episode,
                                policy_seed=seed, decisions=DECISIONS,
                                decide=decide, learn=None,
                                decision_frames=DECISION_FRAMES)
            # the frozen loop has no ``alive_frames`` field; derive it from the two
            # fields it does report (R24's rule: the death frame is excluded, so
            # ``alive_frames == first_death_offset`` on a death)
            row["alive_frames"] = (row["frames"] if row["survived"]
                                   else row["first_death_offset"])
            # v800's loop additionally reports the bomb counters; the frozen loop
            # does not, so they are recorded separately instead of compared
            row["frozen_loop_reports_bombs"] = False
            row.pop("terminal_reply", None)
            row.pop("spawn_reply", None)
            record["frozen_rows"].append(row)
            actions.append(current)
            log(f"equiv frozen ep {episode}: frames={row['frames']} survived="
                f"{row['survived']} actions={len(current)}")
        frozen_actions = actions

        # ---- pass 2: this file's loop with the rule switched off ---------------
        gen2 = torch.Generator()
        gen2.manual_seed(generator_seed(seed))
        for episode in range(episodes):
            def decide2(state: dict[str, Any], hazards: Any, decision: int) -> int:
                feats, _detail = A.build_features(state=state, hazards=hazards,
                                                 decision_index=decision)
                return int(V450.policy_action(policy, feats, "stochastic", gen2))

            row = run_episode_bomb(parent, mode="equiv", arm="A0", bomb_rule="none",
                                   episode=episode, policy_seed=seed,
                                   decisions=DECISIONS, decide=decide2,
                                   decision_frames=DECISION_FRAMES, keep_trace=True)
            trace = row.pop("trace", None) or []
            row.pop("terminal_reply", None)
            row.pop("spawn_reply", None)
            record["bomb_rows"].append({**row, "trace_actions": [t["cand"] for t in trace]})
            log(f"equiv bomb   ep {episode}: frames={row['frames']} survived="
                f"{row['survived']} actions={len(trace)}")
        record["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["status"] = "failed"
        log(f"EQUIV FAILED: {record['error']}")
    finally:
        for parent in parents:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        record["wall_s"] = round(time.perf_counter() - started, 3)

    rows_frozen = record["frozen_rows"]
    rows_bomb = record["bomb_rows"]
    checks: list[dict[str, Any]] = []
    for index in range(min(len(rows_frozen), len(rows_bomb))):
        left, right = rows_frozen[index], rows_bomb[index]
        diffs = {key: {"frozen": left.get(key), "bomb": right.get(key)}
                 for key in EQUIV_FIELDS if left.get(key) != right.get(key)}
        same_actions = (frozen_actions[index] if index < len(frozen_actions) else None) == \
            right.get("trace_actions")
        checks.append({"episode": index, "fields_equal": not diffs, "diff": diffs,
                       "action_sequence_equal": same_actions,
                       "actions_compared": len(right.get("trace_actions") or []),
                       "match": bool(not diffs and same_actions)})
    record["checks"] = checks
    record["matched"] = sum(1 for row in checks if row["match"])
    record["all_pass"] = bool(record["status"] == "complete"
                              and len(rows_frozen) == episodes
                              and len(rows_bomb) == episodes
                              and record["matched"] == episodes)
    write_json(out_path, record)
    log(f"equiv: {record['matched']}/{episodes} episodes identical -> {out_path}")
    return 0 if record["all_pass"] else 8


# --------------------------------------------------------------------------- #
# mode: audit (prereg §2 P3, offline; owns bomb_audit.json)
# --------------------------------------------------------------------------- #
def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_episode_rows(root: pathlib.Path, arm: str) -> dict[str, list[dict[str, Any]]]:
    files = sorted(pathlib.Path(root).glob(f"*eval_{arm}_seed*.jsonl"))
    out: dict[str, list[dict[str, Any]]] = {}
    for path in files:
        out[path.name] = load_jsonl(path)
    return out


def mode_audit(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out_path = pathlib.Path(args.out) if args.out else root / "bomb_audit.json"
    existing: dict[str, Any] = {}
    if out_path.is_file():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:                                             # noqa: BLE001
            existing = {}
    # ``--mode smoke`` writes its block as the whole file (it must exist *before*
    # any full run); a later ``--mode audit`` keeps it under the ``smoke`` key.
    smoke = existing if existing.get("schema") == SCHEMA_SMOKE else existing.get("smoke")
    smoke_source = "bomb_audit.json (written by --mode smoke)"
    if smoke is None:
        for copy in (root / "smoke" / "bomb_smoke.json",
                     root.parent / "smoke" / "bomb_smoke.json"):
            if copy.is_file():
                smoke = json.loads(copy.read_text(encoding="utf-8"))
                smoke_source = str(copy)
                break
    payload: dict[str, Any] = {
        "schema": SCHEMA_AUDIT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": "Document/Update/Alpha/v8.0.0/PREREGISTRATION_BOMB_001.md §1/§2 P3",
        "requirement": ("prereg §2 P3 (descriptive only): per arm the bomb count, "
                        "the moment of each bomb, the remaining bombs, and the "
                        "fraction that ends at 0 — plus the invariants that prove "
                        "the bomb path really fired"),
        "rule_table": {arm: {"rule": rule, "threshold_px": BOMB_RULES[rule]}
                       for arm, rule in ARMS.items()},
        "distance_definition": (
            "min over hazards.bullets of hypot(bullet.x - player_x, bullet.y - "
            "player_y) in playfield px; nearest_lethal_px is the same minimum "
            "restricted to BulletRow.lethal; the rule tests nearest_px (prereg §1)"),
        "smoke": smoke,
        "smoke_source": smoke_source,
        "arms": {}, "status": "failed", "error": None, "wall_s": None,
    }
    started = time.perf_counter()
    try:
        for arm in ARM_ORDER:
            files = read_episode_rows(root, arm)
            if not files:
                continue
            block: dict[str, Any] = {
                "arm": arm, "bomb_rule": ARMS[arm],
                "threshold_px": BOMB_RULES[ARMS[arm]], "files": {},
                "episodes": 0, "emits": 0, "used": 0, "decremented": 0,
                "emits_without_decrement": 0,
                "latency_frames": {}, "emits_without_latency": 0,
                "episodes_that_bombed": 0, "episodes_exhausted": 0,
                "episodes_with_rules_but_no_bomb": 0,
                "emits_per_episode": [], "used_per_episode": [],
                "emit_decision_histogram": {}, "emit_frame_histogram": {},
                "bombs_start_histogram": {}, "first_emit_decision": [],
                "per_seed_window": {},
                "invariants": {},
            }
            for name, rows in files.items():
                block["files"][name] = {"episodes": len(rows)}
                for row in rows:
                    block["episodes"] += 1
                    emits = int(row.get("bomb_emits") or 0)
                    block["emits"] += emits
                    block["used"] += int(row.get("bombs_used") or 0)
                    block["decremented"] += sum(
                        1 for flag in row.get("emit_decremented") or [] if flag)
                    block["emits_without_decrement"] += sum(
                        1 for flag in row.get("emit_decremented") or [] if not flag)
                    for value in row.get("emit_latency_frames") or []:
                        if value is None:
                            block["emits_without_latency"] += 1
                        else:
                            key = str(int(value))
                            block["latency_frames"][key] = (
                                block["latency_frames"].get(key, 0) + 1)
                    block["emits_per_episode"].append(emits)
                    block["used_per_episode"].append(int(row.get("bombs_used") or 0))
                    if emits:
                        block["episodes_that_bombed"] += 1
                    if row.get("bombs_end") == 0:
                        block["episodes_exhausted"] += 1
                    if (ARMS[arm] != "none" and emits == 0
                            and int(row.get("decisions_below_threshold") or 0) == 0):
                        block["episodes_with_rules_but_no_bomb"] += 1
                    for decision in row.get("bomb_decisions") or []:
                        key = str(int(decision))
                        block["emit_decision_histogram"][key] = (
                            block["emit_decision_histogram"].get(key, 0) + 1)
                    for frame in row.get("bomb_frames") or []:
                        key = str(int(frame))
                        block["emit_frame_histogram"][key] = (
                            block["emit_frame_histogram"].get(key, 0) + 1)
                    start_key = str(int(row.get("bombs_start") or 0))
                    block["bombs_start_histogram"][start_key] = (
                        block["bombs_start_histogram"].get(start_key, 0) + 1)
                    if row.get("bomb_decisions"):
                        block["first_emit_decision"].append(
                            int(row["bomb_decisions"][0]))
                    cell = block["per_seed_window"].setdefault(
                        f"s{row.get('policy_seed')}-w{row.get('window')}",
                        {"episodes": 0, "emits": 0, "used": 0, "decremented": 0,
                         "episodes_that_bombed": 0, "episodes_exhausted": 0})
                    cell["episodes"] += 1
                    cell["emits"] += emits
                    cell["used"] += int(row.get("bombs_used") or 0)
                    cell["decremented"] += sum(
                        1 for flag in row.get("emit_decremented") or [] if flag)
                    if emits:
                        cell["episodes_that_bombed"] += 1
                    if row.get("bombs_end") == 0:
                        cell["episodes_exhausted"] += 1
            total = block["episodes"]
            emits_list = list(block["emits_per_episode"])
            used_list = list(block["used_per_episode"])
            block["emits_per_episode_stats"] = quantiles(
                [float(v) for v in emits_list])
            block["used_per_episode_stats"] = quantiles(
                [float(v) for v in used_list])
            block["first_emit_decision_stats"] = quantiles(
                [float(v) for v in block["first_emit_decision"]])
            block["emits_per_episode"] = {
                "distinct_values": sorted({int(v) for v in emits_list}),
                "counts": {str(value): emits_list.count(value)
                           for value in sorted(set(emits_list))},
            }
            block["used_per_episode"] = {
                "distinct_values": sorted({int(v) for v in used_list}),
                "counts": {str(value): used_list.count(value)
                           for value in sorted(set(used_list))},
            }
            block["fractions"] = {
                "episodes_that_bombed": round(block["episodes_that_bombed"] / total, 6)
                if total else None,
                "episodes_exhausted": round(block["episodes_exhausted"] / total, 6)
                if total else None,
            }
            if ARMS[arm] == "none":
                block["invariants"] = {
                    "no_bomb_emitted": block["emits"] == 0,
                    "bombs_used_zero": block["used"] == 0,
                    "bombs_untouched": block["episodes_exhausted"] == 0,
                }
            else:
                block["invariants"] = {
                    "bomb_path_fired": block["emits"] > 0,
                    "bombs_remaining_decreased": block["used"] > 0,
                    "counter_bookkeeping_closes": bool(
                        block["emits"] > 0
                        and block["used"] == block["decremented"]),
                    "emits_at_least_used": block["emits"] >= block["used"],
                    "used_at_most_eight_per_episode": all(
                        v <= 8 for v in block["used_per_episode"]["distinct_values"]),
                    "rule_fired_somewhere": block["episodes_that_bombed"] > 0,
                }
            block["all_pass"] = all(bool(v) for v in block["invariants"].values())
            payload["arms"][arm] = block
            log(f"audit {arm}: episodes={block['episodes']} emits={block['emits']} "
                f"used={block['used']} decremented={block['decremented']} "
                f"exhausted={block['episodes_exhausted']} "
                f"all_pass={block['all_pass']}")
        payload["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["status"] = "failed"
        log(f"AUDIT FAILED: {payload['error']}")
    payload["wall_s"] = round(time.perf_counter() - started, 3)
    payload["all_pass"] = bool(
        payload["status"] == "complete"
        and payload["arms"]
        and all(block["all_pass"] for block in payload["arms"].values()))
    write_json(out_path, payload)
    log(f"audit: {len(payload['arms'])} arm(s) -> {out_path}")
    return 0 if payload["all_pass"] else 5


# --------------------------------------------------------------------------- #
# mode: diagnostics (offline, descriptive only — no statistics)
# --------------------------------------------------------------------------- #
def mode_diagnostics(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out_path = pathlib.Path(args.out) if args.out else root / "diagnostics.json"

    def summary_of(path: pathlib.Path) -> dict[str, Any] | None:
        candidate = path.with_suffix(".summary.json")
        if not candidate.is_file():
            return None
        return json.loads(candidate.read_text(encoding="utf-8"))

    payload: dict[str, Any] = {
        "schema": SCHEMA_DIAG, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": "Document/Update/Alpha/v8.0.0/PREREGISTRATION_BOMB_001.md §2",
        "no_statistics": ("descriptive counts only: no test statistic, no p-value, "
                          "no confidence interval, no significance claim (prereg §5)"),
        "definitions": {
            "alive_frames": "frames executed before the first death; the death "
                            "frame itself is excluded (R24's offset rule), so "
                            "alive_frames == first_death_offset + 1 on a death and "
                            "== 600 when the episode survived",
            "frames": "frames executed including the death frame "
                      "(v432_fork_arm.run_episode's field)",
            "survived": "walked all 300 decisions (600 frames)",
        },
        "arms": {}, "pooled": {}, "a0_vs_v450": {}, "status": "failed",
        "error": None, "wall_s": None,
    }
    started = time.perf_counter()
    try:
        for arm in ARM_ORDER:
            files = sorted(root.glob(f"*eval_{arm}_seed*.jsonl"))
            if not files:
                continue
            block: dict[str, Any] = {"arm": arm, "bomb_rule": ARMS[arm],
                                     "threshold_px": BOMB_RULES[ARMS[arm]],
                                     "seeds": {}, "pooled": {}}
            pooled_rows: list[dict[str, Any]] = []
            for path in files:
                rows = load_jsonl(path)
                if not rows:
                    continue
                seed = int(rows[0]["policy_seed"])
                pooled_rows.extend(rows)
                per_window: dict[str, Any] = {}
                for window in DEV_WINDOWS:
                    cell = [row for row in rows if int(row["window"]) == int(window)]
                    if not cell:
                        continue
                    per_window[str(window)] = {
                        "episodes": len(cell),
                        "alive_frames": quantiles([float(r["alive_frames"]) for r in cell]),
                        "frames": quantiles([float(r["frames"]) for r in cell]),
                        "survived": sum(1 for r in cell if r["survived"]),
                        "survived_fraction": round(
                            sum(1 for r in cell if r["survived"]) / len(cell), 6),
                        "bomb_emits_total": sum(int(r["bomb_emits"]) for r in cell),
                        "bombs_used_total": sum(int(r["bombs_used"] or 0) for r in cell),
                        "episodes_that_bombed": sum(1 for r in cell if r["bomb_emits"]),
                        "episodes_exhausted": sum(1 for r in cell
                                                  if r["bombs_end"] == 0),
                    }
                block["seeds"][str(seed)] = {
                    "file": path.name, "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "summary": summary_of(path),
                    "episodes": len(rows),
                    "alive_frames": quantiles([float(r["alive_frames"]) for r in rows]),
                    "frames": quantiles([float(r["frames"]) for r in rows]),
                    "survived": sum(1 for r in rows if r["survived"]),
                    "survived_fraction": round(
                        sum(1 for r in rows if r["survived"]) / len(rows), 6),
                    "per_window": per_window,
                }
            total = len(pooled_rows)
            survived = sum(1 for r in pooled_rows if r["survived"])
            block["pooled"] = {
                "episodes": total,
                "seeds": sorted({int(r["policy_seed"]) for r in pooled_rows}),
                "windows": sorted({int(r["window"]) for r in pooled_rows}),
                "alive_frames": quantiles([float(r["alive_frames"]) for r in pooled_rows]),
                "frames": quantiles([float(r["frames"]) for r in pooled_rows]),
                "return_decisions": quantiles(
                    [float(r["return_decisions"]) for r in pooled_rows]),
                "survived": survived,
                "survived_fraction": round(survived / total, 6) if total else None,
                "failed": total - survived,
                "bomb_emits_total": sum(int(r["bomb_emits"]) for r in pooled_rows),
                "bombs_used_total": sum(int(r["bombs_used"] or 0) for r in pooled_rows),
                "episodes_that_bombed": sum(1 for r in pooled_rows if r["bomb_emits"]),
                "episodes_exhausted": sum(1 for r in pooled_rows
                                          if r["bombs_end"] == 0),
                "episodes_exhausted_fraction": round(
                    sum(1 for r in pooled_rows if r["bombs_end"] == 0) / total, 6)
                if total else None,
                "emits_per_episode": quantiles(
                    [float(r["bomb_emits"]) for r in pooled_rows]),
                "bombs_used_per_episode": quantiles(
                    [float(r["bombs_used"] or 0) for r in pooled_rows]),
                "decisions_below_threshold_total": sum(
                    int(r["decisions_below_threshold"]) for r in pooled_rows),
                "decisions_below_threshold_lethal_total": sum(
                    int(r["decisions_below_threshold_lethal"]) for r in pooled_rows),
                "decisions_measured_total": sum(
                    int(r["decisions_measured"]) for r in pooled_rows),
            }
            payload["pooled"][arm] = block["pooled"]
            payload["arms"][arm] = block

        # ---- A0 vs the frozen v4.5.0 stochastic dev evaluation ----------------
        ref_root = REPO / ".tmp_v450_cf"
        for seed in SEEDS:
            ref_path = ref_root / FROZEN_V450_EVAL.format(seed=seed)
            mine_path = root / f"eval_A0_seed{seed}.jsonl"
            entry: dict[str, Any] = {
                "seed": seed,
                "reference": str(ref_path), "reference_exists": ref_path.is_file(),
                "a0_file": str(mine_path), "a0_exists": mine_path.is_file(),
            }
            if ref_path.is_file() and mine_path.is_file():
                ref = load_jsonl(ref_path)
                mine = load_jsonl(mine_path)
                same_len = len(ref) == len(mine)
                diffs = 0
                compared = 0
                first: dict[str, Any] | None = None
                for index in range(min(len(ref), len(mine))):
                    left, right = ref[index], mine[index]
                    compared += 1
                    keys = ("window", "episode", "frames", "first_death_offset",
                            "return_decisions", "survived", "decisions_executed")
                    bad = {key: {"v450": left.get(key), "v800": right.get(key)}
                           for key in keys if left.get(key) != right.get(key)}
                    if bad:
                        diffs += 1
                        if first is None:
                            first = {"index": index, "diff": bad}
                entry.update({"reference_episodes": len(ref),
                              "a0_episodes": len(mine),
                              "same_length": same_len,
                              "compared": compared,
                              "episodes_with_differences": diffs,
                              "first_difference": first,
                              "identical": bool(same_len and diffs == 0)})
            payload["a0_vs_v450"][str(seed)] = entry
        payload["status"] = "complete"
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["status"] = "failed"
        log(f"DIAGNOSTICS FAILED: {payload['error']}")
    payload["wall_s"] = round(time.perf_counter() - started, 3)
    write_json(out_path, payload)
    log(f"diagnostics: {len(payload['arms'])} arm(s) -> {out_path}")
    return 0 if payload["status"] == "complete" else 4


# --------------------------------------------------------------------------- #
# mode: assert-ckpt (engine + the three frozen ckpts + protected sources)
# --------------------------------------------------------------------------- #
def mode_assert(args: argparse.Namespace) -> int:
    out_path = (pathlib.Path(args.out) if args.out
                else pathlib.Path(args.work_dir) / "ckpt_assertions.json")
    payload: dict[str, Any] = {
        "schema": SCHEMA_ASSERT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": args.phase,
        "requirement": ("checklist item 2: the S checkpoints this round evaluates "
                        "must be v4.5.0's frozen round2.pt "
                        "(4abee069…/a93746bb…/c69960fc…), and the protected sources "
                        "must not have moved"),
        "engine": None, "ckpts": {}, "protected": {}, "all_pass": False,
    }
    try:
        payload["engine"] = engine_gate(args.engine)
        ok = True
        for seed in SEEDS:
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
        payload["v800_bomb_py"] = {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": sha256_file(pathlib.Path(__file__).resolve())}
        payload["all_pass"] = bool(ok and payload["engine"].get("match"))
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
    write_json(out_path, payload)
    log(f"assert-ckpt {args.phase}: all_pass={payload['all_pass']} -> {out_path}")
    return 0 if payload["all_pass"] else 2


# --------------------------------------------------------------------------- #
# mode: audio-audit (AGENTS.md §9.3, this round's own data dirs only)
# --------------------------------------------------------------------------- #
def mode_audio_audit(args: argparse.Namespace) -> int:
    """``v450_cf_label.mode_audit`` on this round's ``--work-dir``, own ``--out``."""

    V450 = m450()
    namespace = argparse.Namespace(work_dir=args.work_dir,
                                   out=args.out or str(pathlib.Path(args.work_dir)
                                                       / "audio_audit.json"))
    code = V450.mode_audit(namespace)
    return code


# --------------------------------------------------------------------------- #
# mode: process-audit
# --------------------------------------------------------------------------- #
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
        "matching": len(lines), "lines": lines,
    }
    payload["all_pass"] = bool(payload["matching"] == 0)
    write_json(out, payload)
    log(f"process-audit: {payload['matching']} own engine process(es) alive -> {out}")
    return 0 if payload["all_pass"] else 6


# --------------------------------------------------------------------------- #
# mode: manifest
# --------------------------------------------------------------------------- #
def mode_manifest(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.dir)
    if not root.is_dir():
        raise BombError(f"not a directory: {root}")
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
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    global _LOG_FH

    parser = argparse.ArgumentParser(
        description="v8.0.0 BOMB-001: what is a bomb worth (script rule, no learning)")
    parser.add_argument("--mode", required=True,
                        choices=("smoke", "eval", "equiv", "audit", "audio-audit",
                                 "diagnostics", "assert-ckpt", "process-audit",
                                 "manifest"))
    parser.add_argument("--bomb-rule", default="none",
                        choices=tuple(BOMB_RULES),
                        help="prereg §1's single factor: none (A0) / r40 (A1) / "
                             "r60 (A2) / r80 (A3)")
    parser.add_argument("--arm", default="", help="optional cross-check of --bomb-rule")
    parser.add_argument("--seed", type=int, default=None,
                        help="alias for --policy-seed")
    parser.add_argument("--policy-seed", type=int, default=0,
                        help="the v4.5.0 training seed whose S ckpt is evaluated")
    parser.add_argument("--windows", default="11,12,17,18")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--decisions", type=int, default=DECISIONS,
                        help="decisions per episode (v8.0.0's frozen cap is the "
                             "default 300; v8.1.0 BOMBFAR passes 600/900/1200)")
    parser.add_argument("--decision-frames", type=int, default=DECISION_FRAMES)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--work-dir", default=str(WORK))
    parser.add_argument("--out", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--phase", default="start", choices=("start", "final"))
    parser.add_argument("--engine-name", default="th08-modern")
    parser.add_argument("--data-dir-filter", default="")
    parser.add_argument("--dir", default="")
    args = parser.parse_args(argv)

    if args.seed is not None:
        args.policy_seed = int(args.seed)
    if args.arm and args.arm in ARMS and ARMS[args.arm] != args.bomb_rule:
        log(f"--arm {args.arm} contradicts --bomb-rule {args.bomb_rule}; "
            f"refusing to run")
        return 2

    engine_modes = ("smoke", "eval", "equiv")
    if args.mode in engine_modes:
        m450().prepare_env()
    if args.mode in engine_modes:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = args.tag or f"{args.mode}-{stamp}"
        args.tag = tag
        log_dir = pathlib.Path(args.work_dir) / "runs" / tag
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_FH = (log_dir / "run.log").open("a", encoding="utf-8")
        m450()._LOG_FH = _LOG_FH
        m432()._LOG_FH = _LOG_FH

    try:
        if args.mode == "smoke":
            return mode_smoke(args)
        if args.mode == "eval":
            return mode_eval(args)
        if args.mode == "equiv":
            return mode_equiv(args)
        if args.mode == "audit":
            return mode_audit(args)
        if args.mode == "audio-audit":
            return mode_audio_audit(args)
        if args.mode == "diagnostics":
            return mode_diagnostics(args)
        if args.mode == "assert-ckpt":
            return mode_assert(args)
        if args.mode == "process-audit":
            return mode_process_audit(args)
        return mode_manifest(args)
    except BombError as exc:
        log(f"FAILED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
