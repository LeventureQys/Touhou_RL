"""v3.6.1 T2 protocol: per-strategy paired comparison on the strategy's own metric.

Why a new version
-----------------
v3.6.0 §6.2 gates T2 on G-1..G-4, where **G-3 is a survival gate**: ``ALL`` vs
``BASE`` on ``death_60`` must improve by +0.05.  The T1b diagnosis
(``v3.6.0/reports/T1b-G3诊断.md``) showed that no union of blocks - not even the
two that passed their own T1 gate - moves that number beyond +0.007, because
``ENEMY`` is measured by attributed bullet generation and ``POWER`` by power
change; neither is a survival quantity.  Keeping a survival gate as the entry
condition for a bullet-generation tactic therefore blocks T2 for a reason that has
nothing to do with the tactic.

This version replaces that entry condition with a **scope correction**:

    a strategy may enter T2 iff it passed its OWN T1 gate (absolute +
    increment, permutation p < 0.05) on the corpus where that gate was defined.

It is explicitly *not* a relaxation of G-3: G-3 stays in force for any claim about
"the seven tactics together improve survival", and no per-strategy threshold is
changed.  The correction is recorded as a new protocol because v3.6.0 forbids
editing its own pre-registration.

Admitted strategies (from ``T1-ablation-D3``, gate set ``d3``)
--------------------------------------------------------------
* ENEMY   - attributed_spawns_60 rho 0.5459 (gate 0.50), increment +0.1047 (gate +0.05), p 0.005
* POWER   - power_delta_60 increment +0.1535 (gate +0.05), p 0.005

Not admitted, and why:

| strategy | T1 verdict | reason |
|---|---|---|
| REGION | fail | increment -0.0024; conditional comparison infeasible (T1c/T1d) |
| LURE | fail | increment +0.0013; needs a multi-step label, not a single-decision one |
| BOSS | fail | increment -0.0645; block is redundant against BASE (D3 Q3) |
| EMERGENCY | underpowered | 30 bomb positives in the human corpus; needs a dying arm |
| HISTORY | fail | increment +0.0056 |

Design of the comparison
------------------------
* **Interleaved**, not two blocks: within one process the arms alternate episode by
  episode (baseline, tactic, baseline, tactic, ...).  Machine/game drift then hits
  both arms equally, and pairing is by episode index.
* **Same start state**: practice stage 1, Lunatic, team 0, frameskip 2, identical
  menu macro.  Team 0 is fixed on purpose - it is the only team whose movement
  constants are validated (4.0 / 2.0 px per frame), so no arm gets a wrong
  forward model.
* **n = 20 pairs** per strategy (the v3.6.0 §5.1 probe size), max_steps 3000.
* **Traces recorded** in both arms with ``slots``, so the primary metric is
  measured identically in both arms from the same offline pipeline.

Per-strategy metrics (pre-registered; primary in bold)
------------------------------------------------------
ENEMY (ClearEnemy):

* **attributed spawns per decision** (bullets spawned within 32 px of a visible
  enemy), relative change tactic vs baseline;
* target-enemy life drop per decision, kills, bullet count at +60/+240 frames;
* survival frames, deaths, bombs used;
* intent switches, wall-hug rate, action entropy, override/abort counts.

POWER (CollectPower):

* **power gain rate while clearance >= 6 px** (the safe-window gain);
* P items collected, power at first boss appearance;
* high-risk dash rate (moves that reduce clearance below 3 px within 10 frames);
* survival frames, deaths, bombs used.

Success thresholds (pre-registered)
-----------------------------------
ENEMY: attributed spawns per decision down **>=15%** with paired bootstrap
p < 0.05, **and** survival frames >= 90% of baseline, **and** deaths not more
than +10 percentage points.
POWER: safe-window power gain rate up **>=15%**, **and** the same two safety
conditions.

Safety dominates: a tactic that improves its own metric while failing either
safety condition is recorded as ``fail_safety`` regardless of the primary number.

Stopping lines
--------------
1. Fewer than 16 of 20 pairs usable (crash, max_steps truncation both arms
   asymmetric) -> ``underpowered``, no claim.
2. Any arm's complete rate < 95% -> re-check the recorder before analysing.
3. The baseline arm's attributed-spawn rate differing by more than 30% from the
   D3 corpus mean -> the setup changed; stop and compare setups before trusting
   the delta.
4. Never re-tune the aim weight, safety margin or clearance gate after seeing a
   result; they are fixed in ``touhou_rl/tactics.py`` and recorded in the report.

Deliverables
------------
``v3.6.1/{执行计划-T2.md, data/probe/t2-*/ , experiments/t2-*/ , reports/, evidence/}``
plus this module and its tests.
"""

from __future__ import annotations

import math

import numpy as np

from . import spec

# --- pre-registered constants (do not tune after seeing a result) -----------
CLEAR_ENEMY_AIM_WEIGHT = 3.0
CLEAR_ENEMY_SAFE_MARGIN = 10.0
COLLECT_POWER_MIN_CLEARANCE = 6.0
# The validated collection threshold from v3.5.0's P-contract experiment: the
# auto-collect line is y=128, and the controlled route moved up while
# player_y > 110 (the P-contract H-P3/H-P4/H-P5 all passed with that route).
COLLECT_POWER_TARGET_Y = 110.0
COLLECT_POWER_LINE = 128.0

TACTICS = ("none", "clear_enemy", "collect_power")


def _field(obj, name: str, default=None):
    """Read ``name`` from a dict or an attribute object.

    The offline pipeline sees trace dicts (``{"x": ..., "life": ...}``) while the
    live probe sees ``state.EnemyInfo`` / ``state.ItemInfo`` dataclasses.  This is
    the only place that difference is allowed to exist.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _item_type(item) -> int:
    """Item kind for both encodings: trace key ``type``, dataclass ``item_type``."""
    if isinstance(item, dict):
        return int(item.get("type", -1))
    return int(getattr(item, "item_type", -1))


def choose_target(enemies: list[dict], player_y: float) -> dict | None:
    """Highest-value visible enemy for a ClearEnemy intent.

    Pre-registered value (fixed before any T2 run):

        value(e) = boss_weight(e) * on_screen(e) * above_weight(e) / (1 + dist/100)

    * ``boss_weight`` = 1.0 for a boss, 0.4 otherwise (bosses are the dominant
      source of the bullets that kill you);
    * ``above_weight`` = 1.5 when the enemy is above the player (its bullets travel
      down into the path), 0.7 otherwise;
    * distance in pixels, so a far enemy is discounted but never ignored.

    Returns ``None`` when no enemy is alive.
    """
    best: dict | None = None
    best_score = -math.inf
    for enemy in enemies or []:
        life = float(_field(enemy, "life", 0.0) or 0.0)
        if life <= 0.0:
            continue
        ex = float(_field(enemy, "x", 0.0) or 0.0)
        ey = float(_field(enemy, "y", 0.0) or 0.0)
        dist = math.hypot(ex - spec.PLAYER_START_X, ey - player_y)
        weight = 1.0 if _field(enemy, "boss", False) else 0.4
        above = 1.5 if ey < player_y else 0.7
        score = weight * above / (1.0 + dist / 100.0)
        if score > best_score:
            best, best_score = enemy, score
    return best


def target_as_boss_array(target: dict | None) -> np.ndarray | None:
    """Feed the target's x to the pilot's existing aim term via its ``bosses`` arg.

    ``AvoidPilot._boss_x`` reads ``bosses[0, 0]`` and the aim preference pulls the
    path horizontally toward it, gated by ``aim_safe_margin``.  Substituting the
    target here is what makes ClearEnemy a *single-factor* change: the pilot's
    code, beam width, horizon and bomb logic are untouched.
    """
    if target is None:
        return None
    return np.asarray(
        [[float(_field(target, "x", 0.0) or 0.0), float(_field(target, "y", 0.0) or 0.0)]], dtype=np.float32
    )


def collect_power_override(
    power: float,
    player_y: float,
    items: list[dict],
    clearance: float,
    min_clearance: float = COLLECT_POWER_MIN_CLEARANCE,
    target_y: float = COLLECT_POWER_TARGET_Y,
) -> dict:
    """Conditional P collection, pre-registered in v3.6.0 §2.5 terms.

    Returns a dict with ``override`` (the action to force, or ``None``) and a
    ``reason`` that is recorded per decision, so the report can separate
    "aborted because unsafe" from "not applicable".

    Conditions (all must hold to override):

    * ``power < POWER_MAX`` (there is something to gain);
    * at least one on-screen power item exists;
    * the pilot's chosen action has clearance >= ``min_clearance`` pixels;
    * either the player is below the collection line (move up) or already above it
      (hold focus and let auto-collect run).

    The safety gate uses the pilot's *own* clearance for the action it already
    chose, not a separate plan: a tactic may not override a pilot that is already
    in trouble.
    """
    if power >= spec.POWER_MAX:
        return {"override": None, "reason": "power_max"}
    power_items = [
        item
        for item in (items or [])
        if _item_type(item) in spec.ITEM_POWER_TYPES and bool(_field(item, "onscreen", False))
    ]
    if not power_items:
        return {"override": None, "reason": "no_p_item"}
    if clearance < min_clearance:
        return {"override": None, "reason": "unsafe", "clearance": round(float(clearance), 2)}
    if player_y <= target_y:
        # Already in the collection band: hold focus and stop steering.
        return {
            "override": (0, 1, 1, 0),
            "reason": "hold_collect",
            "p_items": len(power_items),
            "clearance": round(float(clearance), 2),
        }
    return {
        "override": (1, 1, 1, 0),
        "reason": "up_collect",
        "p_items": len(power_items),
        "clearance": round(float(clearance), 2),
    }


def describe_target(target) -> dict:
    """Controller record for one ClearEnemy decision (trace-dict or dataclass)."""
    if target is None:
        return {"tactic": "clear_enemy", "abort_reason": "no_target"}
    return {
        "tactic": "clear_enemy",
        "target_slot": int(_field(target, "slot", -1)),
        "target_is_boss": bool(_field(target, "boss", False)),
        "target_x": round(float(_field(target, "x", 0.0) or 0.0), 2),
    }


# Per-pair randomization (v3.6.1 T2, pre-registered).
#
# The game is deterministic from the fixed practice start, so two episodes of the
# same arm are byte-identical (measured before this existed).  Policy-weight jitter
# was tried first and did *not* help: at stage start every action's clearance sits
# at the 60 px cap, so the ranking is "prefer larger y" for any positive
# bottom_bias - changing the magnitude never changes the order.
#
# The randomization therefore acts on the trajectory: a seeded warm-up of
# WARMUP_DECISIONS random moves, applied identically to both arms of a pair.  The
# pairing survives (same initial condition distribution per pair); the pairs stop
# being duplicates because the bullet-dense part of the stage starts from a
# different geometry.
WARMUP_DECISIONS = (20, 60)


def warmup_plan(seed: int | None, pair: int) -> list[tuple[int, int]]:
    """``(move, focus)`` for each warm-up decision; empty when unseeded.

    Identical for both arms of a pair because it is derived from ``seed + pair``.
    ``shoot`` stays on and ``bomb`` stays off: the warm-up randomizes position, it
    must not randomize resource usage (that would contaminate the bomb metrics).
    """
    if seed is None:
        return []
    rng = np.random.default_rng(int(seed) * 1000 + int(pair))
    count = int(rng.integers(WARMUP_DECISIONS[0], WARMUP_DECISIONS[1] + 1))
    return [(int(rng.integers(0, 9)), int(rng.integers(0, 2))) for _ in range(count)]


def jitter_kwargs(seed: int | None, pair: int) -> dict:
    """Deprecated no-op kept so old call sites and traces stay readable.

    Policy-weight jitter was measured to have no effect on a deterministic start;
    ``warmup_plan`` replaces it.  Returning an empty dict keeps the pilot
    configuration identical across pairs, which is now the honest thing to do.
    """
    return {}


def pilot_config_kwargs(tactic: str) -> dict:
    """PilotConfig overrides that define the ClearEnemy arm.

    ``none`` and ``collect_power`` keep the baseline pilot configuration, so the
    only difference in those comparisons is the collection override.
    """
    if tactic == "clear_enemy":
        return {"aim_weight": CLEAR_ENEMY_AIM_WEIGHT, "aim_safe_margin": CLEAR_ENEMY_SAFE_MARGIN}
    return {}


def validate_tactic(tactic: str) -> str:
    if tactic not in TACTICS:
        raise ValueError(f"unknown tactic {tactic!r}; expected one of {TACTICS}")
    return tactic
