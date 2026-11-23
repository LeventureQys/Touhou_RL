#!/usr/bin/env python3
"""v8.5.0 DISTILL-001 — distil the R5 rule into the 36-action policy.

What this is
------------
The executor for `Document/Update/Alpha/v8.5.0/PREREGISTRATION_DISTILL_001.md`
(FROZEN 2026-11-22, before any v8.5.0 training run).

v8.3.0 widened the action space to **36** (``action = cand * 2 + bomb``) and
labelled every one of the 5,166 frozen v6.0.0 states along all 36 actions.  The
label contrast is brutal — ``bomb = 1`` is safe on 99.83 % of its labels,
``bomb = 0`` on 54.58 % — so the BCE-trained policy learns "always bomb, at
once": it spends all three bombs on the first decisions and collapses from
99.94 % (300 decisions) to 19.44 % (600) and 0.61 % (900).

v8.4.0 then found a *rule* that spends the same three bombs but at the right
moments — **R5** (``staged``): ``bombs_remaining == 3`` -> 80 px,
``== 2`` -> 60 px, ``== 1`` -> 40 px, ``== 0`` -> never — which holds
77.05 % at 600 decisions instead of 19.44 %.

This round asks the prereg's single-factor question: **does adding the R5 rule
as extra supervision on the bomb head, on top of the frozen counterfactual
labels, recover the rule's timing while keeping the policy's 18-way dodging?**

    targets = R5 on every one of the 5,166 states, computed offline from `obs`
              (nearest bullet distance; no engine, no re-collection)
    model   = Policy(106, 64, 36), unchanged
    loss    = BCE(36 counterfactual labels) + **0.5 * BCE_bomb(R5)**
              where the bomb logit of candidate c is
              ``logit(action=2c+1) - logit(action=2c)``, i.e. the 36 outputs are
              reshaped to (18, 2) and the rule trains the *difference*
    train   = Adam 1e-3, batch 256, 200 epochs, 3 seeds (prereg §1; unchanged)
    eval    = 3 budgets (300 / 600 / 900 decisions) x 3 seeds x 4 dev windows
              x 300 episodes, stochastic, live engine

Modes
-----
==================================  =============================================
``--mode preflight``                 pins (sha256 of every imported file **and**
                                     of the frozen 36-label file), feature dim
                                     (106), the frozen recount of that file,
                                     engine gate.
``--mode assert-ckpt --phase P``      engine sha256 + the three frozen v4.5.0
                                     ``round2.pt`` (``4abee069`` / ``a93746bb`` /
                                     ``c69960fc``) that are the behaviour-policy
                                     source of the 5,166 states, plus the sha256
                                     of every protected source and of the frozen
                                     label file.
``--mode targets``                   R5 on all 5,166 states, offline, from
                                     ``obs``: nearest bullet distance, then
                                     ``bombs_remaining`` 3/2/1 -> 80/60/40 px
                                     (**strictly** ``<``, v8.4.0's own test;
                                     0 bombs -> never).  Writes
                                     ``rule_targets.jsonl`` (row-aligned with
                                     ``labels_bomb36.jsonl``) and
                                     ``rule_target_stats.json``.
``--mode train --seed S``            ``BCE(36) + 0.5 * BCE_bomb(R5)``, Adam 1e-3,
                                     batch 256, 200 epochs, no early stopping,
                                     dev never read; reports the total loss and
                                     **both parts** every epoch.
``--mode eval --decisions {300,600,900} --policy-seed S``
                                     v8.3.0's 36-action evaluation path
                                     (`v830_bombpol.run_episode36`, imported, not
                                     copied): the sampled ``bomb`` bit is
                                     executed as ``cand_to_mask(cand, bomb)`` on
                                     the decision's first frame; 4 dev windows,
                                     300 episodes each, ``stochastic``.
``--mode diagnostics``               offline per-budget / per-(seed,window)
                                     counts, means, medians, quantiles, the bomb
                                     usage table, the first/last bomb-decision
                                     distributions and the exhaustion rate, side
                                     by side with v8.3.0 and R5.  **No test, no
                                     interval.**
``--mode audio-audit``               AGENTS.md §9.3 on this round's own data
                                     dirs; writes this round's own file only.
``--mode process-audit``             this round's own engine processes must be 0
                                     (``pgrep -af`` filtered by ``--data-dir``).
``--mode ckpt-manifest``             sha256 + bytes of every checkpoint this
                                     round wrote (they stay in the work dir).
``--mode manifest --dir D``          sha256 + bytes of every file in D.
``--mode publish``                   copy the raw artifacts into
                                     ``Document/Update/Alpha/v8.5.0/evidence/
                                     DISTILL/`` and write the manifest.

Reused, never re-implemented, never edited (imported by path; sha256 recorded and
gated by ``--mode assert-ckpt``)
---------------------------------------------------------------------------
* ``tools/rl/v830_bombpol.py`` — ``run_episode36`` (the 36-action evaluation
  loop: ``cand_to_mask(cand, bomb)`` on the decision's first frame), ``policy36``
  (``Policy(106,64,36)``), ``load_policy36``, ``sample_action36``,
  ``build_arrays`` (106-dim features from the stored observations, with the
  recorded ``features_sha256_k12`` re-checked on every row), ``action_masks``,
  ``split_action``, ``quantiles`` / ``histogram`` / ``first_bomb_histogram``;
* ``tools/rl/v450_cf_label.py`` — ``prepare_env`` (dummy audio, no display),
  ``prepare_data_dir`` (muted ``th08.cfg`` regenerated + ``--inspect``ed before
  every engine start), ``start_parent``, ``check_engine``, ``rebuild_hazards``,
  ``mode_audit`` (the audio audit);
* ``tools/rl/v432_fork_arm.py`` — ``ForkParent`` and ``--mode fidelity`` (the D3
  gate, run through v432 itself);
* ``tools/rl/v431_arm_a.py`` — ``build_features`` and ``Policy``;
* ``touhou_rl/v420/contracts.py`` — ``cand_to_mask`` (only via v830).

Not in this file
----------------
* No statistics: counts, rates, means, medians, quantiles and distributions
  only.  P1-P4 belong to the independent analysis pass (prereg §2/§5).
* No new labels, no re-collection: the counterfactual labels are v8.3.0's, read
  read-only, and the engine is touched **only** by ``--mode eval``.
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

WORK = REPO / ".tmp_v850_distill"
PUBLISH = REPO / "Document" / "Update" / "Alpha" / "v8.5.0" / "evidence" / "DISTILL"
V830_WORK = REPO / ".tmp_v830_bombpol"
LABELS36 = V830_WORK / "labels_bomb36.jsonl"
CKPT_ROOT = REPO / ".tmp_v450_cf" / "ckpt"
M830_PATH = REPO / "tools" / "rl" / "v830_bombpol.py"

# -- the frozen v8.3.0 label batch this round trains on (read-only) ----------
LABELS36_SHA256 = (
    "aea2c7abee220997caee5c58fe7b4bbad6b4cbeb5d8549d2fc5efad046deed6e")
LABELS36_BYTES = 49602266
LABELS36_STATES = 5166

# -- pinned imports: recorded everywhere, gated by --mode assert-ckpt --------
#    (`v800_bomb.py` is listed for the record only: this round never imports it,
#     and v8.4.0 patched it after v8.3.0 closed, so its hash is not a gate.)
PINNED = {
    "tools/rl/v830_bombpol.py":
        "a8c6d3065c968dbd546e9b37f389d3f786404821b8b5fa53e25d520549c58e1c",
    "tools/rl/v450_cf_label.py":
        "b7ab0ebeee38df4fcbb5bbbdf5e453edc8e5547abb702473b0c09cb4e90d9947",
    "tools/rl/v432_fork_arm.py":
        "33235506077308549f7657e359381d8c38b1bf45ba4ede0d802448f2dbf0cede",
    "tools/rl/v431_arm_a.py":
        "96923930fed5d1df6ea398e858aa9fb123ed39a455c3ad7e2706230432445503",
    "tools/rl/v431_dense_driver.py":
        "1079d93a46ed77c72c9d684112da9ed8d6810797f8e16fc9e789c2fed3fb4f25",
    "touhou_rl/v420/contracts.py":
        "1357c69c32a74fda20553a3e6b290ca2102b7cc5c52e47e9fdc6eb9e08c946c1",
    "touhou_rl/v400/gym/actions.py":
        "a67145b2bec7c96b3aacaa2514e78cdadf50b056473d15b4ae8122c19c393eff",
}
RECORDED_NOT_GATED = ("tools/rl/v800_bomb.py",)
PROTECTED_FILES = tuple(PINNED) + RECORDED_NOT_GATED

# -- the three frozen v4.5.0 `S` checkpoints (the behaviour-policy source) ---
FROZEN_ROUND2_SHA256: dict[int, str] = {
    0: "4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc",
    1: "a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4",
    2: "c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01",
}

SCHEMA_PREFLIGHT = "v850-distill-preflight/1"
SCHEMA_ASSERT = "v850-distill-ckpt-assertions/1"
SCHEMA_TARGET = "v850-distill-rule-target/1"
SCHEMA_TARGET_STATS = "v850-distill-rule-target-stats/1"
SCHEMA_TRAIN = "v850-distill-train/1"
SCHEMA_TRAIN_SUMMARY = "v850-distill-train-summary/1"
SCHEMA_EVAL = "v850-distill-eval/1"
SCHEMA_EVAL_SUMMARY = "v850-distill-eval-summary/1"
SCHEMA_DIAG = "v850-distill-diagnostics/1"
SCHEMA_BOMB_USAGE = "v850-distill-bomb-usage/1"
SCHEMA_AUDIT = "v850-distill-audio-audit/1"
SCHEMA_PROC = "v850-distill-process-audit/1"
SCHEMA_MANIFEST = "v850-distill-manifest/1"
SCHEMA_PUBLISH = "v850-distill-publish/1"
SCHEMA_CKPT_MANIFEST = "v850-distill-ckpt-manifest/1"
PRODUCER = "tools/rl/v850_distill.py (live engine for eval; HarnessClient role=shadow)"
PREREG = "Document/Update/Alpha/v8.5.0/PREREGISTRATION_DISTILL_001.md"
R5_SOURCE = "Document/Update/Alpha/v8.4.0/PREREGISTRATION_BOMBRULE2_001.md §1 (R5)"

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- frozen geometry (unchanged from v4.5.0/v6.0.0/v8.0.0/v8.3.0) -----------
PREFIX_FRAMES = 8663

# -- frozen split (prereg §1: the v4.5.0 / v6.0.0 protocol) -----------------
TRAIN_WINDOWS: tuple[int, ...] = (13, 14, 15, 16)
DEV_WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
TRAIN_SEEDS: tuple[int, ...] = (0, 1, 2)

# -- frozen design (prereg §1) ----------------------------------------------
N_CAND = 18
N_ACTIONS = 36
DECISION_FRAMES = 2
BUDGETS: tuple[int, ...] = (300, 600, 900)
EPISODES = 300
FEATURE_DIM = 106
HIDDEN = 64
LR = 1e-3
BATCH = 256
EPOCHS = 200
LAMBDA_RULE = 0.5              # prereg §1: fixed, not scanned
TEMPERATURE = 1.0
MIN_STATES = 4000
H_DEFAULT = 120
R5_THRESHOLDS: dict[int, float] = {3: 80.0, 2: 60.0, 1: 40.0}   # v8.4.0 R5

# -- the two references this round is placed beside (numbers only, no test) --
V830_REFERENCE = {
    "arm": "v8.3.0 BOMBPOL-001 — 36-action policy, counterfactual labels only",
    "episodes_per_budget": 3600,
    "survived": {300: 3598, 600: 700, 900: 22},
    "rate": {300: 0.999444, 600: 0.194444, 900: 0.006111},
    "alive_mean": {300: 599.9906, 600: 990.7286, 900: 1029.8478},
    "alive_median": {300: 600.0, 600: 989.5, 900: 992.0},
    "bombs_used_per_episode": {300: 3.0, 600: 3.0, 900: 3.0},
    "first_bomb_decision_zero": {300: 3569, 600: 3560, 900: 3573},
    "exhaustion_fraction": {300: 1.0, 600: 1.0, 900: 1.0},
    "source": "Document/Update/Alpha/v8.3.0/evidence/BOMBPOL/RUN.md §5 and §7",
}
R5_REFERENCE = {
    "arm": "v8.4.0 BOMBRULE2-001 R5 (`staged`): 3->80 px, 2->60 px, 1->40 px",
    "rate": {300: 1.0, 600: 0.7705, 900: 0.1938},
    "rate_exact_600": 0.771111,
    "rate_exact_600_source": (
        "Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/RUN.md §0: 2,776 / 3,600 "
        "= 77.1111 % at 600 decisions"),
    "rate_300_source": (
        "Document/Update/Alpha/v8.0.0/evidence/BOMB/RUN.md §4 — the 80 px script "
        "rule is 3,600/3,600 at 300 decisions, and R5 fires on decision 0 in every "
        "episode, so its 300-decision arm is the same 100 %"),
    "rate_900_source": (
        "Document/Update/Alpha/v8.5.0/PREREGISTRATION_DISTILL_001.md §0: 19.38 % "
        "(n = 970, still '收尾中' when this prereg was frozen) — the 600/900 "
        "numbers this round is compared with are **quoted from the prereg**, not "
        "recomputed here"),
    "note": ("the prereg quotes 77.05 % / 19.38 %; v8.4.0's published 600-decision "
             "count is 77.1111 %. Both are reported, neither is recomputed"),
}

_LOG_FH: Any = None
_M830: Any = None


# --------------------------------------------------------------------------- #
# errors / logging / helpers
# --------------------------------------------------------------------------- #
class DistillError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v850] {message}"
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


def m830():
    """``tools/rl/v830_bombpol.py`` — imported, never edited (reuse, no copy)."""

    global _M830
    if _M830 is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("v830_bombpol", M830_PATH)
        if spec is None or spec.loader is None:
            raise DistillError(f"cannot import {M830_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _M830 = module
    return _M830


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quantiles(values: Sequence[float]) -> dict[str, Any]:
    return m830().quantiles(values)


def first_bomb_histogram(values: Sequence[int]) -> dict[str, int]:
    return m830().first_bomb_histogram(values)


def decision_histogram(values: Sequence[int]) -> dict[str, int]:
    """Same buckets as ``first_bomb_histogram`` — used for the *last* press too."""

    return m830().first_bomb_histogram(values)


# --------------------------------------------------------------------------- #
# pins / engine / paths
# --------------------------------------------------------------------------- #
def check_pins() -> dict[str, Any]:
    out: dict[str, Any] = {"pinned": {}, "recorded": {}, "all_gated_match": True}
    for rel, expected in PINNED.items():
        path = REPO / rel
        got = sha256_file(path) if path.is_file() else None
        match = got == expected
        out["pinned"][rel] = {"sha256": got, "expected": expected, "match": match}
        if not match:
            out["all_gated_match"] = False
    for rel in RECORDED_NOT_GATED:
        path = REPO / rel
        out["recorded"][rel] = {
            "sha256": sha256_file(path) if path.is_file() else None,
            "gated": False,
            "why": ("not imported by this round; v8.4.0 patched it after v8.3.0 "
                    "closed, so its hash is recorded, never gated"),
        }
    path = LABELS36
    got = sha256_file(path) if path.is_file() else None
    out["labels_bomb36"] = {
        "path": str(path), "sha256": got, "expected": LABELS36_SHA256,
        "bytes": path.stat().st_size if path.is_file() else None,
        "match": got == LABELS36_SHA256}
    if not out["labels_bomb36"]["match"]:
        out["all_gated_match"] = False
    return out


def engine_gate(path: str) -> dict[str, Any]:
    return dict(m830().v450().check_engine(path))


def prepare_env() -> dict[str, Any]:
    return dict(m830().prepare_env())


def targets_path(root: pathlib.Path) -> pathlib.Path:
    return root / "rule_targets.jsonl"


def targets_stats_path(root: pathlib.Path) -> pathlib.Path:
    return root / "rule_target_stats.json"


def ckpt_path(root: pathlib.Path, seed: int) -> pathlib.Path:
    return root / "ckpt" / f"seed{int(seed)}" / "policy36.pt"


def train_report_path(root: pathlib.Path, seed: int) -> pathlib.Path:
    return root / "train" / f"seed{int(seed)}.json"


def eval_path(root: pathlib.Path, budget: int, seed: int) -> pathlib.Path:
    return root / "eval" / f"eval_d{int(budget)}_seed{int(seed)}.jsonl"


def parse_windows(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(text).split(",") if part.strip())


# --------------------------------------------------------------------------- #
# the R5 rule, offline, from a stored observation
# --------------------------------------------------------------------------- #
def rule_of(bombs_remaining: int, nearest_px: float | None) -> dict[str, Any]:
    """v8.4.0 R5 on one state: ``3 -> 80 px``, ``2 -> 60 px``, ``1 -> 40 px``.

    The distance test is v8.4.0's own — **strictly** ``nearest < threshold``
    (``v800_bomb.run_episode_bomb``: ``(nearest <= threshold) if wall else
    (nearest < threshold)``, and R5 is not ``wall``) — and the ``bombs_remaining
    > 0`` gate is v8.0.0's ``fired = bombs_before > 0``: with no bomb left the
    rule cannot fire however close the bullet is.
    """

    bombs = int(bombs_remaining)
    threshold = R5_THRESHOLDS.get(bombs)
    if threshold is None:
        return {"bombs_remaining": bombs, "threshold_px": None, "rule_bomb": 0,
                "reason": "no bomb left" if bombs <= 0 else f"bombs={bombs} outside R5"}
    if nearest_px is None:
        return {"bombs_remaining": bombs, "threshold_px": threshold,
                "rule_bomb": 0, "reason": "no bullet in the observation"}
    fire = bool(float(nearest_px) < float(threshold))
    return {"bombs_remaining": bombs, "threshold_px": threshold,
            "rule_bomb": int(fire),
            "reason": ("nearest < threshold" if fire else "nearest >= threshold")}


def nearest_from_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Nearest bullet distance (px) of a stored observation, and its provenance.

    The stored ``obs['hazards']['bullets']`` is v6.0.0's compact observation: the
    **nearest** bullets (64 of them), and the frozen 106-dim feature vector is
    built from the 12 nearest of exactly this list.  ``--mode targets`` re-checks
    every row's ``features_sha256_k12`` (``v830_bombpol.build_arrays``) before it
    trusts the list, so the minimum computed here is the live minimum the rule
    would have measured.
    """

    state = obs["state"]
    px = float(state.get("player_x") or 0.0)
    py = float(state.get("player_y") or 0.0)
    bullets = obs["hazards"].get("bullets") or []
    dists = [math.hypot(float(b[0]) - px, float(b[1]) - py) for b in bullets]
    nearest_raw = min(dists) if dists else None
    return {
        "nearest_px": (round(nearest_raw, 4) if nearest_raw is not None else None),
        "bullets_stored": len(bullets),
        "bullet_count": int(obs["hazards"].get("bullet_count") or 0),
        "nearest_is_first": bool(
            dists and abs(dists[0] - min(dists)) < 1e-9),
        "sorted_by_distance": bool(
            all(dists[i] <= dists[i + 1] + 1e-9 for i in range(len(dists) - 1))),
    }


# --------------------------------------------------------------------------- #
# mode: preflight
# --------------------------------------------------------------------------- #
def mode_preflight(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    rec: dict[str, Any] = {
        "schema": SCHEMA_PREFLIGHT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "single_factor": ("prereg §1: does adding the R5 rule as extra supervision "
                          "on the bomb head help, with lambda fixed at 0.5?"),
        "pins": check_pins(), "launch_env": prepare_env(),
        "action_space": {"candidates": N_CAND, "bombs": [0, 1],
                         "actions": N_ACTIONS,
                         "encoding": "action = cand * 2 + bomb"},
        "loss": ("BCEWithLogits(36 counterfactual labels) + %g * "
                 "BCEWithLogits(bomb logit, rule_bomb); the bomb logit of "
                 "candidate c is logit(2c+1) - logit(2c)" % LAMBDA_RULE),
        "frozen": {"TRAIN_WINDOWS": list(TRAIN_WINDOWS),
                   "DEV_WINDOWS": list(DEV_WINDOWS),
                   "train_seeds": list(TRAIN_SEEDS),
                   "lambda": LAMBDA_RULE, "lr": LR, "batch": BATCH,
                   "epochs": EPOCHS, "hidden": HIDDEN,
                   "feature_dim": FEATURE_DIM, "H_labels": H_DEFAULT,
                   "budgets_decisions": list(BUDGETS),
                   "decisions_cap": 300, "decision_frames": DECISION_FRAMES,
                   "episodes_per_window": EPISODES, "temperature": TEMPERATURE},
        "r5": {"thresholds": {str(k): v for k, v in R5_THRESHOLDS.items()},
               "test": "nearest_all_px < threshold (strict), bombs_remaining > 0",
               "source": R5_SOURCE},
        "engine": {"path": args.engine},
        "labels_bomb36": {"path": str(LABELS36),
                          "states_expected": LABELS36_STATES},
        "recount": {}, "all_pass": False,
    }
    ok = bool(rec["pins"]["all_gated_match"])
    if LABELS36.is_file():
        rows = read_jsonl(LABELS36)
        rec["recount"] = {
            "states": len(rows),
            "states_expected_match": len(rows) == LABELS36_STATES,
            "bytes_match": LABELS36.stat().st_size == LABELS36_BYTES,
            "labels_per_state": sorted({len(row["safe"]) for row in rows}),
            "H": sorted({int(row["H"]) for row in rows}),
            "windows": sorted({int(row["window"]) for row in rows}),
            "seeds": sorted({int(row["seed"]) for row in rows}),
            "rounds": sorted({int(row["round"]) for row in rows}),
            "bombs_remaining_distribution": _counter(
                [int(row["obs"]["state"]["bombs_remaining"]) for row in rows]),
        }
        ok = ok and rec["recount"]["states_expected_match"] \
            and rec["recount"]["bytes_match"]
    else:
        rec["recount"] = {"error": f"missing {LABELS36}"}
        ok = False
    if int(m830().arm_a().feature_dim()) != FEATURE_DIM:
        rec["feature_dim_error"] = (f"v431_arm_a.feature_dim() = "
                                    f"{m830().arm_a().feature_dim()} != {FEATURE_DIM}")
        ok = False
    rec["ckpt_source"] = {
        str(seed): str(CKPT_ROOT / f"seed{seed}" / "round2.pt")
        for seed in TRAIN_SEEDS}
    rec["all_pass"] = ok
    out = pathlib.Path(args.out) if args.out else root / "preflight.json"
    write_json(out, rec)
    log(f"preflight: pins={rec['pins']['all_gated_match']} "
        f"states={rec['recount'].get('states')} all_pass={ok} -> {out}")
    return 0 if ok else 2


def _counter(values: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(int(value)) if isinstance(value, (int, float)) else str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda item: item[0]))


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
                        "the frozen 36-label file and this round's own executable "
                        "must not move"),
        "engine": None, "ckpts": {}, "protected": {}, "v850_distill_py": {},
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
                "bytes": path.stat().st_size if path.is_file() else None,
                "gated": rel in PINNED,
                "pinned": PINNED.get(rel),
                "match_pin": (sha256_file(path) == PINNED[rel]
                              if rel in PINNED and path.is_file() else None)}
        me = pathlib.Path(__file__).resolve()
        payload["v850_distill_py"] = {"path": str(me), "sha256": sha256_file(me)}
        payload["labels_bomb36"] = {
            "path": str(LABELS36),
            "sha256": sha256_file(LABELS36) if LABELS36.is_file() else None,
            "expected": LABELS36_SHA256,
            "match": (LABELS36.is_file()
                      and sha256_file(LABELS36) == LABELS36_SHA256)}
        ok = ok and payload["labels_bomb36"]["match"]
        payload["all_pass"] = bool(ok)
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
    write_json(out_path, payload)
    log(f"assert-ckpt {args.phase}: all_pass={payload['all_pass']} -> {out_path}")
    return 0 if payload["all_pass"] else 2


# --------------------------------------------------------------------------- #
# mode: targets  (R5 on every state, offline)
# --------------------------------------------------------------------------- #
def mode_targets(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out = pathlib.Path(args.out) if args.out else targets_path(root)
    stats_out = (pathlib.Path(args.stats_out) if args.stats_out
                 else targets_stats_path(root))
    pins = check_pins()
    rows = read_jsonl(LABELS36)
    rec: dict[str, Any] = {
        "schema": SCHEMA_TARGET_STATS, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "rule": {"name": "R5 (staged, v8.4.0)", "source": R5_SOURCE,
                 "thresholds_px": {str(k): v for k, v in R5_THRESHOLDS.items()},
                 "zero_bombs": "never",
                 "distance_test": "nearest_all_px < threshold (strict)",
                 "nearest_definition": ("min over the bullets of the stored "
                                        "observation, 2-D pixel distance to "
                                        "player_x/player_y")},
        "source": {"path": str(LABELS36), "sha256": pins["labels_bomb36"]["sha256"],
                   "expected": LABELS36_SHA256, "bytes": LABELS36.stat().st_size,
                   "states": len(rows), "read_only": True},
        "feature_reproduction": {}, "states": len(rows), "status": "failed",
        "error": None, "all_pass": False, "wall_s": None,
    }
    started = time.perf_counter()
    try:
        if not pins["all_gated_match"]:
            raise DistillError(f"pinned import moved: {json.dumps(pins)[:600]}")
        if len(rows) != LABELS36_STATES:
            raise DistillError(f"{len(rows)} states != {LABELS36_STATES}")
        # -- gate: the stored bullet lists must be the ones the frozen 106-dim
        #    observation was built from; v830's loader re-checks every row's
        #    features_sha256_k12 and refuses to train if one row disagrees.
        _feats, _labels, array_info = m830().build_arrays(rows)
        rec["feature_reproduction"] = dict(array_info)
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = out.with_suffix(out.suffix + ".partial")
        by_bombs: dict[str, dict[str, Any]] = {}
        by_threshold: dict[str, dict[str, Any]] = {}
        positives = 0
        nearest_all: list[float] = []
        first_is_min = 0
        sorted_rows = 0
        unsorted_by_bombs: dict[str, int] = {}
        with partial.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                obs = row["obs"]
                near = nearest_from_obs(obs)
                bombs = int(obs["state"]["bombs_remaining"] or 0)
                rule = rule_of(bombs, near["nearest_px"])
                record = {
                    "schema": SCHEMA_TARGET, "index": int(index),
                    "seed": int(row["seed"]), "round": int(row["round"]),
                    "window": int(row["window"]),
                    "decision": int(row["decision"]),
                    "frame": int(row["frame"]),
                    "H": int(row["H"]),
                    "nearest_px": near["nearest_px"],
                    "bombs_remaining": bombs,
                    "threshold_px": rule["threshold_px"],
                    "rule_bomb": int(rule["rule_bomb"]),
                    "reason": rule["reason"],
                    "bullets_stored": near["bullets_stored"],
                    "bullet_count": near["bullet_count"],
                    "nearest_is_first": near["nearest_is_first"],
                    "sorted_by_distance": near["sorted_by_distance"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                positives += int(rule["rule_bomb"])
                if near["nearest_px"] is not None:
                    nearest_all.append(float(near["nearest_px"]))
                first_is_min += int(near["nearest_is_first"])
                sorted_rows += int(near["sorted_by_distance"])
                if not near["sorted_by_distance"]:
                    key = str(bombs)
                    unsorted_by_bombs[key] = unsorted_by_bombs.get(key, 0) + 1
                slot = by_bombs.setdefault(str(bombs), {
                    "bombs_remaining": bombs, "states": 0, "rule_bomb_1": 0,
                    "nearest_px": []})
                slot["states"] += 1
                slot["rule_bomb_1"] += int(rule["rule_bomb"])
                if near["nearest_px"] is not None:
                    slot["nearest_px"].append(float(near["nearest_px"]))
                tkey = ("none" if rule["threshold_px"] is None
                        else f"{rule['threshold_px']:g}px")
                tslot = by_threshold.setdefault(tkey, {
                    "threshold_px": rule["threshold_px"], "states": 0,
                    "rule_bomb_1": 0})
                tslot["states"] += 1
                tslot["rule_bomb_1"] += int(rule["rule_bomb"])
        if out.exists():
            out.unlink()
        partial.rename(out)
        states = len(rows)
        for key, slot in by_bombs.items():
            slot["rule_bomb_1_fraction"] = round(slot["rule_bomb_1"]
                                                 / slot["states"], 6)
            slot["nearest_px"] = quantiles(slot["nearest_px"])
        rec.update({
            "targets_path": str(out), "targets_sha256": sha256_file(out),
            "targets_bytes": out.stat().st_size,
            "rule_bomb_1": positives,
            "rule_bomb_1_fraction": round(positives / states, 6),
            "rule_bomb_0_fraction": round(1 - positives / states, 6),
            "by_bombs_remaining": by_bombs,
            "by_threshold": by_threshold,
            "nearest_px": quantiles(nearest_all),
            "nearest_px_histogram": m830().histogram(
                [int(v) for v in nearest_all], (1, 5, 10, 20, 40, 60, 80, 120, 200)),
            "provenance": {
                "nearest_is_first": first_is_min,
                "sorted_by_distance": sorted_rows,
                "sorted_by_distance_fraction": round(sorted_rows / states, 6),
                "unsorted_rows_by_bombs_remaining": unsorted_by_bombs,
                "why_this_is_the_live_minimum": (
                    "the frozen 106-dim features are built from the 12 nearest of "
                    "exactly this stored list, and all 5,166 rows reproduce their "
                    "recorded features_sha256_k12 (above), so the nearest stored "
                    "bullet is the nearest live bullet"),
            },
            "status": "complete",
        })
        rec["all_pass"] = bool(
            rec["feature_reproduction"].get("feature_sha256_reproduced") == states
            and rec["feature_reproduction"].get("label_vectors_valid") == states
            and rec["targets_sha256"] == sha256_file(out)
            and states == LABELS36_STATES)
    except Exception as exc:                                          # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["status"] = "failed"
        rec["all_pass"] = False
        log(f"TARGETS FAILED: {rec['error']}")
    finally:
        rec["wall_s"] = round(time.perf_counter() - started, 3)
    write_json(stats_out, rec)
    log(f"targets: rule_bomb=1 {rec.get('rule_bomb_1')}/{rec.get('states')} = "
        f"{rec.get('rule_bomb_1_fraction')} all_pass={rec.get('all_pass')} -> {out}")
    return 0 if rec.get("all_pass") else 2


# --------------------------------------------------------------------------- #
# mode: train  (BCE36 + 0.5 * BCE_bomb)
# --------------------------------------------------------------------------- #
def load_training_arrays(root: pathlib.Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Features, 36-way labels, the R5 target vector (row-aligned) and its audit."""

    import numpy as np

    rows = read_jsonl(LABELS36)
    if len(rows) < MIN_STATES:
        raise DistillError(f"S-label: {len(rows)} states < {MIN_STATES}")
    path = targets_path(root)
    if not path.is_file():
        raise DistillError(f"no rule targets: {path}; run --mode targets first")
    stats = read_json(path.parent / targets_stats_path(root).name) \
        if targets_stats_path(root).is_file() else {}
    targets = read_jsonl(path)
    if len(targets) != len(rows):
        raise DistillError(f"{len(targets)} rule targets != {len(rows)} labelled "
                           f"states; the files are not row-aligned")
    mismatch = [i for i, (row, target) in enumerate(zip(rows, targets))
                if (int(target["seed"]) != int(row["seed"])
                    or int(target["round"]) != int(row["round"])
                    or int(target["window"]) != int(row["window"])
                    or int(target["decision"]) != int(row["decision"])
                    or int(target["frame"]) != int(row["frame"]))]
    if mismatch:
        raise DistillError(f"{len(mismatch)} rule targets do not locate the same "
                           f"state as the label row (first {mismatch[0]})")
    feats, labels, info = m830().build_arrays(rows)
    rule = np.asarray([int(t["rule_bomb"]) for t in targets], dtype=np.float32)
    info = dict(info)
    info.update({
        "rule_targets_path": str(path),
        "rule_targets_sha256": sha256_file(path),
        "rule_targets_recorded_sha256": stats.get("targets_sha256"),
        "rule_targets_sha256_match": (sha256_file(path)
                                      == stats.get("targets_sha256")),
        "rule_bomb_1": int(rule.sum()),
        "rule_bomb_1_fraction": round(float(rule.mean()), 6),
        "row_alignment": f"{len(targets)}/{len(rows)} rows located identically",
    })
    if not info["rule_targets_sha256_match"]:
        raise DistillError(f"{path} sha256 {info['rule_targets_sha256'][:16]} != "
                           f"the recorded "
                           f"{str(info['rule_targets_recorded_sha256'])[:16]}")
    return feats, labels, rule, info


def mode_train(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    if seed not in TRAIN_SEEDS:
        raise DistillError(f"--seed must be one of {TRAIN_SEEDS}, got {seed}")
    feats, labels, rule, info = load_training_arrays(root)
    rows = len(feats)
    log(f"train seed={seed}: {rows} states x {N_ACTIONS} labels + R5 on the bomb "
        f"head (rule_bomb=1 {info['rule_bomb_1']}/{rows}); every features_sha256 "
        f"reproduced")

    x = torch.from_numpy(feats)
    y = torch.from_numpy(labels)
    tgt_full = torch.from_numpy(rule).view(-1, 1).expand(-1, N_CAND).contiguous()
    policy = m830().policy36(seed=seed)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    bce = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    batch = int(args.batch)
    epoch_total: list[float] = []
    epoch_bce36: list[float] = []
    epoch_bce_bomb: list[float] = []
    started = time.perf_counter()
    for epoch in range(int(EPOCHS)):
        perm = torch.from_numpy(rng.permutation(rows))
        sum_total = sum_36 = sum_bomb = 0.0
        policy.train()
        for start in range(0, rows, batch):
            index = perm[start:start + batch]
            logits = policy(x.index_select(0, index))
            # the bomb logit of candidate c is logit(2c+1) - logit(2c): the 36
            # outputs are (18 candidates) x (2 bomb values), so reshaping is
            # exactly the pair-wise difference the prereg asks for.
            pair = logits.view(-1, N_CAND, 2)
            bomb_logit = pair[:, :, 1] - pair[:, :, 0]
            loss_36 = bce(logits, y.index_select(0, index))
            loss_bomb = bce(bomb_logit, tgt_full.index_select(0, index))
            loss = loss_36 + LAMBDA_RULE * loss_bomb
            opt.zero_grad()
            loss.backward()
            opt.step()
            count = int(index.numel())
            sum_total += float(loss.detach()) * count
            sum_36 += float(loss_36.detach()) * count
            sum_bomb += float(loss_bomb.detach()) * count
        epoch_total.append(sum_total / rows)
        epoch_bce36.append(sum_36 / rows)
        epoch_bce_bomb.append(sum_bomb / rows)
        if epoch % 25 == 0 or epoch == int(EPOCHS) - 1:
            log(f"  seed {seed} epoch {epoch}: total={epoch_total[-1]:.4f} "
                f"bce36={epoch_bce36[-1]:.4f} bce_bomb={epoch_bce_bomb[-1]:.4f}")
    wall = round(time.perf_counter() - started, 3)

    policy.eval()
    with torch.no_grad():
        logits = policy(x)
        pair = logits.view(-1, N_CAND, 2)
        bomb_logit = pair[:, :, 1] - pair[:, :, 0]
        final_36 = float(bce(logits, y))
        final_bomb = float(bce(bomb_logit, tgt_full))
        final_total = final_36 + LAMBDA_RULE * final_bomb
        predicted = (torch.sigmoid(logits) >= 0.5).float()
        overall_acc = float((predicted == y).float().mean())
        bomb_prob = torch.sigmoid(bomb_logit)
        bomb_pred = (bomb_prob >= 0.5).float()
        bomb_acc = float((bomb_pred == tgt_full).float().mean())
        rule1 = rule > 0.5
        bomb_prob_rule1 = (float(bomb_prob[torch.from_numpy(rule1)].mean())
                           if bool(rule1.any()) else None)
        bomb_prob_rule0 = (float(bomb_prob[torch.from_numpy(~rule1)].mean())
                           if bool((~rule1).any()) else None)
        # per-bombs_remaining mean predicted bomb probability (descriptive: does
        # the head reproduce R5's 80/60/40 shape?)
        bombs_column = [int(row["obs"]["state"]["bombs_remaining"])
                        for row in read_jsonl(LABELS36)]
        by_bombs: dict[str, Any] = {}
        for value in sorted(set(bombs_column)):
            mask = torch.tensor([b == value for b in bombs_column])
            if not bool(mask.any()):
                continue
            by_bombs[str(value)] = {
                "states": int(mask.sum()),
                "mean_p_bomb": round(float(bomb_prob[mask].mean()), 6),
                "share_predicted_bomb": round(
                    float(bomb_pred[mask].mean()), 6),
                "rule_bomb_1_fraction": round(
                    float(torch.from_numpy(rule)[mask].mean()), 6),
                "threshold_px": R5_THRESHOLDS.get(value),
            }

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else ckpt_path(root, seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), str(ckpt))
    report_path = (pathlib.Path(args.out) if args.out
                   else train_report_path(root, seed))
    report: dict[str, Any] = {
        "schema": SCHEMA_TRAIN, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "seed": seed, "states": rows, "labels": int(rows * N_ACTIONS),
        "feature_dim": int(FEATURE_DIM), "actions": int(N_ACTIONS),
        "model": {"class": "v431_arm_a.Policy with the output layer widened to 36 "
                           "(v830_bombpol.policy36, imported)",
                  "shape": f"Linear({FEATURE_DIM},{HIDDEN}) -> Tanh -> "
                           f"Linear({HIDDEN},36)",
                  "params": int(sum(p.numel() for p in policy.parameters()))},
        "inputs": info,
        "loss": {
            "formula": "BCE36 + 0.5 * BCE_bomb(R5)",
            "lambda": LAMBDA_RULE,
            "bce36": "BCEWithLogitsLoss over the 36 counterfactual labels, mean "
                     "over batch and 36 outputs",
            "bce_bomb": "logits.view(-1, 18, 2); bomb logit = logit(2c+1) - "
                        "logit(2c); BCEWithLogitsLoss against the state's rule_bomb "
                        "broadcast over the 18 candidates, mean over batch and 18",
        },
        "optimizer": {"name": "Adam", "lr": LR, "batch": batch, "epochs": EPOCHS,
                      "early_stopping": None,
                      "shuffle": "numpy default_rng(seed), one permutation/epoch"},
        "init": {"torch_manual_seed": seed, "hidden": HIDDEN,
                 "feature_dim": int(FEATURE_DIM)},
        "epoch_losses_total": [round(v, 6) for v in epoch_total],
        "epoch_losses_bce36": [round(v, 6) for v in epoch_bce36],
        "epoch_losses_bce_bomb": [round(v, 6) for v in epoch_bce_bomb],
        "loss_first_epoch": round(epoch_total[0], 6),
        "loss_last_epoch": round(epoch_total[-1], 6),
        "bce36_first_epoch": round(epoch_bce36[0], 6),
        "bce36_last_epoch": round(epoch_bce36[-1], 6),
        "bce_bomb_first_epoch": round(epoch_bce_bomb[0], 6),
        "bce_bomb_last_epoch": round(epoch_bce_bomb[-1], 6),
        "final_train_bce36": round(final_36, 6),
        "final_train_bce_bomb": round(final_bomb, 6),
        "final_train_total": round(final_total, 6),
        "train_accuracy": round(overall_acc, 6),
        "bomb_decision_accuracy_vs_rule": round(bomb_acc, 6),
        "mean_predicted_prob_bomb_on_rule1_states": bomb_prob_rule1,
        "mean_predicted_prob_bomb_on_rule0_states": bomb_prob_rule0,
        "bomb_probability_by_bombs_remaining": by_bombs,
        "ckpt": {"path": str(ckpt), "sha256": sha256_file(ckpt),
                 "bytes": ckpt.stat().st_size},
        "pins": check_pins(), "wall_s": wall, "deviations": [],
    }
    write_json(report_path, report)
    log(f"train seed={seed}: total {report['loss_first_epoch']:.4f} -> "
        f"{report['loss_last_epoch']:.4f} (bce36 {report['bce36_last_epoch']:.4f} "
        f"+ 0.5*{report['bce_bomb_last_epoch']:.4f}) acc={overall_acc:.4f} "
        f"bomb_acc_vs_rule={bomb_acc:.4f} wall={wall}s sha256="
        f"{report['ckpt']['sha256'][:16]}…")
    return 0


# --------------------------------------------------------------------------- #
# mode: eval  (v8.3.0's 36-action loop, imported)
# --------------------------------------------------------------------------- #
def mode_eval(args: argparse.Namespace) -> int:
    import torch

    from touhou_rl.v420 import contracts as C

    root = pathlib.Path(args.work_dir)
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    if seed not in TRAIN_SEEDS:
        raise DistillError(f"--policy-seed must be one of {TRAIN_SEEDS}, got {seed}")
    if budget not in BUDGETS:
        raise DistillError(f"prereg §1 fixes --decisions to {BUDGETS}, got {budget}")
    if int(args.decision_frames) != DECISION_FRAMES:
        raise DistillError(f"prereg §1 keeps {DECISION_FRAMES} frames per decision")
    if int(args.episodes) < 1:
        raise DistillError(f"--episodes must be >= 1, got {args.episodes}")
    windows = parse_windows(args.windows)
    pins = check_pins()
    if not pins["all_gated_match"]:
        raise DistillError(f"pinned import moved: {json.dumps(pins)[:600]}")
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    tag = args.tag or f"v850-eval-d{budget}-s{seed}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    out_path = (pathlib.Path(args.out) if args.out
                else eval_path(root, budget, seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")
    ckpt = pathlib.Path(args.ckpt) if args.ckpt else ckpt_path(root, seed)
    digest = sha256_file(ckpt) if ckpt.is_file() else None
    policy = m830().load_policy36(ckpt, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(m830().generator_seed(seed))
    arm = m830().arm_a()
    run_episode36 = m830().run_episode36

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        feats, _detail = arm.build_features(state=state, hazards=hazards,
                                            decision_index=decision)
        action = m830().sample_action36(policy, feats, args.kind, generator)
        return m830().split_action(action)

    summary: dict[str, Any] = {
        "schema": SCHEMA_EVAL_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "mode": "eval", "policy_seed": seed, "windows": list(windows),
        "episodes_per_window": int(args.episodes),
        "budget_decisions": budget, "decision_frames": DECISION_FRAMES,
        "temperature": TEMPERATURE, "kind": args.kind,
        "engine": engine, "launch_env": launch_env,
        "eval_impl": {
            "loop": "tools/rl/v830_bombpol.py::run_episode36 (imported, unchanged)",
            "action_execution": ("bomb=1 -> cand_to_mask(cand,1) on the decision's "
                                 "first frame, cand_to_mask(cand,0) afterwards; "
                                 "bomb=0 -> cand_to_mask(cand,0) throughout"),
            "v830_bombpol_sha256": pins["pinned"]["tools/rl/v830_bombpol.py"]["sha256"],
        },
        "ckpt": {"path": str(ckpt), "sha256": digest,
                 "bytes": ckpt.stat().st_size if ckpt.is_file() else None},
        "sampler": {"kind": "torch.multinomial on softmax(z/1) over 36 actions",
                    "generator_seed": m830().generator_seed(seed),
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
            data_dir, data_info = m830().v450().prepare_data_dir(
                data_root, window, args.data_source)
            log(f"d{budget} s{seed} w{window}: cfg muted="
                f"{data_info['cfg']['muted_four_zero']} "
                f"sha256={data_info['cfg']['sha256'][:16]}…")
            parent, gate = m830().start_parent(slot=slot, window=window,
                                               engine=args.engine,
                                               data_dir=data_dir,
                                               log_dir=run_dir)
            parents.append(parent)
            rows: list[dict[str, Any]] = []
            for episode in range(int(args.episodes)):
                record = run_episode36(parent, episode=episode, policy_seed=seed,
                                       decisions=budget, decide=decide,
                                       budget=budget,
                                       decision_frames=DECISION_FRAMES)
                record["schema"] = SCHEMA_EVAL
                record["ckpt_sha256"] = digest
                record["generator_seed"] = m830().generator_seed(seed)
                record["eval_impl"] = "tools/rl/v830_bombpol.py::run_episode36"
                record.pop("terminal_reply", None)
                record.pop("trace", None)
                record.pop("spawn_reply", None)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows.append(record)
                if record.get("invalid"):
                    raise DistillError(f"d{budget} seed {seed} window {window} "
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
# mode: diagnostics (offline, descriptive only)
# --------------------------------------------------------------------------- #
def _episode_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return read_jsonl(path)


def honoured_bomb_decisions(row: dict[str, Any]) -> list[int]:
    """The decisions on which a bomb was **actually spent** (the counter dropped).

    Every press is recorded as ``(decision, bombs_remaining_before)``.  A press is
    honoured exactly when the counter two frames later is one lower, i.e. when the
    **next** recorded press (or the episode's ``bombs_end``) sees a lower counter:
    a press while a bomb is burning is ignored by the game and leaves the counter
    where it was (v8.0.0/v8.3.0 measured 98.2 % ignored presses for this policy
    family).  This is therefore the arm's *effective* bomb timing, comparable with
    the R5 rule's emit decisions — not the press rate.
    """

    decisions = [int(v) for v in (row.get("bomb_decisions") or [])]
    before = row.get("bombs_remaining_at_emit") or []
    if not decisions or len(before) != len(decisions):
        return []
    honoured: list[int] = []
    for index, decision in enumerate(decisions):
        after = (int(before[index + 1]) if index + 1 < len(before)
                 else row.get("bombs_end"))
        if after is not None and int(after) < int(before[index]):
            honoured.append(int(decision))
    return honoured


def _bomb_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    bombs_used = [int(row["bombs_used"] or 0) for row in rows]
    first = [int(row["first_bomb_decision"]) for row in rows
             if row["first_bomb_decision"] is not None]
    last = [int(row["bomb_decisions"][-1]) for row in rows if row["bomb_decisions"]]
    honoured = [honoured_bomb_decisions(row) for row in rows]
    first_honoured = [values[0] for values in honoured if values]
    last_honoured = [values[-1] for values in honoured if values]
    derived_mismatch = sum(1 for values, used in zip(honoured, bombs_used)
                           if len(values) != used)
    hist: dict[str, int] = {}
    for value in bombs_used:
        hist[str(int(value))] = hist.get(str(int(value)), 0) + 1
    exhausted = sum(1 for row in rows if int(row["bombs_end"] or 0) == 0)
    return {
        "episodes": len(rows),
        "episodes_that_pressed_bomb": sum(1 for row in rows
                                          if int(row["bomb_emits"] or 0) > 0),
        "episodes_that_spent_a_bomb": sum(1 for v in bombs_used if v > 0),
        "bomb_emits_total": sum(int(row["bomb_emits"] or 0) for row in rows),
        "bombs_spent_total": sum(bombs_used),
        "bombs_ignored_emits_total": sum(int(row.get("bombs_ignored_emits") or 0)
                                         for row in rows),
        "bombs_used_per_episode": quantiles(bombs_used),
        "bombs_used_histogram": dict(sorted(hist.items())),
        "episodes_exhausted": exhausted,
        "exhaustion_fraction": round(exhausted / len(rows), 6) if rows else None,
        "first_bomb_decision": quantiles(first),
        "first_bomb_decision_histogram": first_bomb_histogram(first),
        "first_bomb_decision_zero": sum(1 for v in first if v == 0),
        "last_bomb_decision": quantiles(last),
        "last_bomb_decision_histogram": decision_histogram(last),
        "bomb_press_rate_per_decision": (
            round(sum(int(row["bomb_emits"] or 0) for row in rows)
                  / sum(int(row["decisions_measured"] or 0) for row in rows), 6)
            if rows else None),
        "honoured_bombs": {
            "unit": "one bomb the game actually spent (the counter dropped)",
            "derived_total": sum(len(values) for values in honoured),
            "bombs_used_total": sum(bombs_used),
            "episodes_where_derived_differs_from_bombs_used": derived_mismatch,
            "first_honoured_decision": quantiles(first_honoured),
            "first_honoured_decision_histogram": decision_histogram(first_honoured),
            "first_honoured_decision_zero": sum(1 for v in first_honoured if v == 0),
            "last_honoured_decision": quantiles(last_honoured),
            "last_honoured_decision_histogram": decision_histogram(last_honoured),
            "note": ("counts, means and quantiles of the decisions on which the "
                     "counter dropped; no test, no interval"),
        },
        "episodes_never_bombed": len(rows) - sum(1 for row in rows
                                                 if int(row["bomb_emits"] or 0) > 0),
        "mean_decisions_between_first_and_last_bomb": (
            round(statistics.fmean([b - a for a, b in
                                    zip([int(r["first_bomb_decision"]) for r in rows
                                         if r["first_bomb_decision"] is not None],
                                        [int(r["bomb_decisions"][-1]) for r in rows
                                         if r["bomb_decisions"]])]), 4)
            if all(row["bomb_decisions"] for row in rows) and rows else None),
    }


def mode_diagnostics(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out = pathlib.Path(args.out) if args.out else root / "diagnostics.json"
    usage_out = (pathlib.Path(args.usage_out) if args.usage_out
                 else root / "bomb_usage.json")
    targets_stats = (read_json(targets_stats_path(root))
                     if targets_stats_path(root).is_file() else {})
    train_reports = {}
    for seed in TRAIN_SEEDS:
        path = train_report_path(root, seed)
        if path.is_file():
            rec = read_json(path)
            train_reports[str(seed)] = {
                "loss_first_epoch": rec["loss_first_epoch"],
                "loss_last_epoch": rec["loss_last_epoch"],
                "bce36_last_epoch": rec["bce36_last_epoch"],
                "bce_bomb_last_epoch": rec["bce_bomb_last_epoch"],
                "final_train_bce36": rec["final_train_bce36"],
                "final_train_bce_bomb": rec["final_train_bce_bomb"],
                "final_train_total": rec["final_train_total"],
                "train_accuracy": rec["train_accuracy"],
                "bomb_decision_accuracy_vs_rule":
                    rec["bomb_decision_accuracy_vs_rule"],
                "bomb_probability_by_bombs_remaining":
                    rec["bomb_probability_by_bombs_remaining"],
                "ckpt": rec["ckpt"],
            }
    payload: dict[str, Any] = {
        "schema": SCHEMA_DIAG, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "no_statistics": ("counts, rates, means, medians, quantiles and "
                          "distributions only; no test, no interval, no p-value "
                          "(prereg §5)"),
        "design": {"loss": f"BCE36 + {LAMBDA_RULE}*BCE_bomb(R5)",
                   "seeds": list(TRAIN_SEEDS), "budgets": list(BUDGETS),
                   "episodes_per_window": EPISODES, "windows": list(DEV_WINDOWS)},
        "rule_targets": {key: targets_stats.get(key) for key in (
            "targets_sha256", "states", "rule_bomb_1", "rule_bomb_1_fraction",
            "rule_bomb_0_fraction", "by_bombs_remaining", "by_threshold",
            "nearest_px", "provenance")} if targets_stats else {},
        "training": train_reports,
        "budgets": {}, "per_cell": {}, "missing": [], "all_pass": True,
        "references": {"v830": V830_REFERENCE, "r5": R5_REFERENCE,
                       "nobomb_300": {
                           "rate": 0.134444, "survived": 484, "episodes": 3600,
                           "source": "Document/Update/Alpha/v8.0.0/evidence/"
                                     "BOMB/RUN.md §4 (v4.5.0 S, 18 actions, bomb "
                                     "never pressed)"}},
        "budget_comparison_warning": ("prereg §5: the three budgets must not be "
                                      "compared by their means as if they were the "
                                      "same measurement"),
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
                "survived_fraction": round(
                    sum(1 for row in seed_rows if row["survived"])
                    / len(seed_rows), 6),
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
                first = [int(row["first_bomb_decision"]) for row in cell
                         if row["first_bomb_decision"] is not None]
                last = [int(row["bomb_decisions"][-1]) for row in cell
                        if row["bomb_decisions"]]
                per_window[key] = {
                    "seed": seed, "window": int(window), "episodes": len(cell),
                    "survived": sum(1 for row in cell if row["survived"]),
                    "survived_fraction": round(
                        sum(1 for row in cell if row["survived"]) / len(cell), 6),
                    "alive_frames": quantiles([row["alive_frames"] for row in cell]),
                    "bombs_used_total": sum(int(row["bombs_used"] or 0)
                                            for row in cell),
                    "episodes_exhausted": sum(1 for row in cell
                                              if row["bombs_end"] == 0),
                    "first_bomb_decision_mean": (round(statistics.fmean(first), 4)
                                                 if first else None),
                    "last_bomb_decision_mean": (round(statistics.fmean(last), 4)
                                                if last else None),
                }
        if not rows:
            continue
        alive = [float(row["alive_frames"]) for row in rows]
        survived = sum(1 for row in rows if row["survived"])
        rec = {
            "budget_decisions": budget,
            "episodes": len(rows),
            "survived": survived,
            "survived_fraction": round(survived / len(rows), 6),
            "alive_frames": quantiles(alive),
            "alive_frames_histogram": m830().histogram(
                [int(v) for v in alive],
                (1, 12, 24, 48, 60, 120, 300, 600, 900, 1200, 1800)),
            "per_seed": per_seed, "per_seed_window": per_window,
            "bomb": _bomb_block(rows),
        }
        payload["budgets"][str(budget)] = rec
        usage["budgets"][str(budget)] = dict(rec)
    # -- the side-by-side table (descriptive arithmetic, no test) -----------
    side = {}
    for budget in BUDGETS:
        rec = payload["budgets"].get(str(budget))
        if not rec:
            continue
        side[str(budget)] = {
            "budget_decisions": budget,
            "v850": {"survived": rec["survived"], "episodes": rec["episodes"],
                     "rate": rec["survived_fraction"],
                     "alive_mean": rec["alive_frames"]["mean"],
                     "alive_median": rec["alive_frames"]["median"],
                     "bombs_used_per_episode":
                         rec["bomb"]["bombs_used_per_episode"]["mean"],
                     "first_bomb_decision_mean":
                         rec["bomb"]["first_bomb_decision"].get("mean"),
                     "last_bomb_decision_mean":
                         rec["bomb"]["last_bomb_decision"].get("mean"),
                     "first_honoured_bomb_decision_mean":
                         rec["bomb"]["honoured_bombs"]["first_honoured_decision"].get("mean"),
                     "last_honoured_bomb_decision_mean":
                         rec["bomb"]["honoured_bombs"]["last_honoured_decision"].get("mean"),
                     "bomb_press_rate_per_decision":
                         rec["bomb"]["bomb_press_rate_per_decision"],
                     "exhaustion_fraction": rec["bomb"]["exhaustion_fraction"]},
            "v830": {"rate": V830_REFERENCE["rate"][budget],
                     "survived": V830_REFERENCE["survived"][budget],
                     "alive_mean": V830_REFERENCE["alive_mean"][budget],
                     "alive_median": V830_REFERENCE["alive_median"][budget],
                     "bombs_used_per_episode":
                         V830_REFERENCE["bombs_used_per_episode"][budget],
                     "exhaustion_fraction":
                         V830_REFERENCE["exhaustion_fraction"][budget]},
            "r5": {"rate": R5_REFERENCE["rate"][budget],
                   "rate_source": (
                       R5_REFERENCE["rate_exact_600_source"] if budget == 600
                       else (R5_REFERENCE["rate_300_source"] if budget == 300
                             else R5_REFERENCE["rate_900_source"]))},
            "delta": {
                "vs_v830_rate_pp": round(
                    100 * (rec["survived_fraction"]
                           - V830_REFERENCE["rate"][budget]), 4),
                "vs_r5_rate_pp": round(
                    100 * (rec["survived_fraction"]
                           - R5_REFERENCE["rate"][budget]), 4),
                "note": ("percentage-point differences of raw rates; no test, no "
                         "interval (prereg §2 P1/P2 belong to the analysis pass)"),
            },
        }
    payload["side_by_side"] = side
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
                f"exhausted={rec['bomb']['exhaustion_fraction']:.4%} "
                f"first_bomb_mean={rec['bomb']['first_bomb_decision'].get('mean')} "
                f"last_bomb_mean={rec['bomb']['last_bomb_decision'].get('mean')}")
    log(f"diagnostics -> {out} (+ {usage_out})")
    return 0 if payload["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: train-summary (one table across the three seeds)
# --------------------------------------------------------------------------- #
def mode_train_summary(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out: dict[str, Any] = {
        "schema": SCHEMA_TRAIN_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "design": (f"loss = BCE36 + {LAMBDA_RULE}*BCE_bomb(R5), Adam {LR:g}, "
                   f"batch {BATCH}, {EPOCHS} epochs, 3 seeds; one checkpoint per "
                   f"seed trained on all {LABELS36_STATES} states"),
        "labels": str(LABELS36), "targets": str(targets_path(root)),
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
            "bce36_first_epoch": rec["bce36_first_epoch"],
            "bce36_last_epoch": rec["bce36_last_epoch"],
            "bce_bomb_first_epoch": rec["bce_bomb_first_epoch"],
            "bce_bomb_last_epoch": rec["bce_bomb_last_epoch"],
            "final_train_bce36": rec["final_train_bce36"],
            "final_train_bce_bomb": rec["final_train_bce_bomb"],
            "final_train_total": rec["final_train_total"],
            "train_accuracy": rec["train_accuracy"],
            "bomb_decision_accuracy_vs_rule":
                rec["bomb_decision_accuracy_vs_rule"],
            "ckpt": rec["ckpt"], "wall_s": rec["wall_s"],
        }
    write_json(pathlib.Path(args.out) if args.out else root / "train_summary.json",
               out)
    log(f"train-summary: {len(out['seeds'])}/{len(TRAIN_SEEDS)} seeds")
    return 0 if out["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: ckpt-manifest (checkpoints stay in the work dir, sha256 registered)
# --------------------------------------------------------------------------- #
def mode_ckpt_manifest(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out_path = (pathlib.Path(args.out) if args.out
                else root / "ckpt_manifest.json")
    files: dict[str, Any] = {}
    ckpt_root = root / "ckpt"
    if ckpt_root.is_dir():
        for path in sorted(p for p in ckpt_root.rglob("*") if p.is_file()):
            rel = str(path.relative_to(root)).replace("\\", "/")
            files[rel] = {"sha256": sha256_file(path),
                          "bytes": path.stat().st_size,
                          "path": str(path)}
    payload = {
        "schema": SCHEMA_CKPT_MANIFEST, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG,
        "requirement": ("checklist item 6: the three trained checkpoints stay in "
                        "`.tmp_v850_distill/` and their sha256 is registered here "
                        "and in every train report / eval summary"),
        "root": str(root / "ckpt"), "file_count": len(files),
        "total_bytes": sum(v["bytes"] for v in files.values()),
        "files": files,
        "input_ckpts": {str(seed): {
            "path": str(CKPT_ROOT / f"seed{seed}" / "round2.pt"),
            "sha256": FROZEN_ROUND2_SHA256[seed]} for seed in TRAIN_SEEDS},
    }
    write_json(out_path, payload)
    log(f"ckpt-manifest: {len(files)} checkpoint(s) -> {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# mode: audio-audit / process-audit / manifest
# --------------------------------------------------------------------------- #
def mode_audio_audit(args: argparse.Namespace) -> int:
    namespace = argparse.Namespace(
        work_dir=args.work_dir,
        out=args.out or str(pathlib.Path(args.work_dir) / "audio_audit.json"))
    rc = m830().v450().mode_audit(namespace)
    # relabel the file as this round's own artifact (same content, own schema)
    path = pathlib.Path(namespace.out)
    if path.is_file():
        payload = read_json(path)
        payload["schema"] = SCHEMA_AUDIT
        payload["producer"] = PRODUCER
        payload["prereg"] = PREREG
        payload["work_dir"] = str(pathlib.Path(args.work_dir))
        payload["all_pass"] = bool(payload.get("all_muted"))
        write_json(path, payload)
    return rc


def mode_process_audit(args: argparse.Namespace) -> int:
    needle = str(args.data_dir_filter)
    out = (pathlib.Path(args.out) if args.out
           else pathlib.Path(args.work_dir) / "process_residue.json")
    proc = subprocess.run(["pgrep", "-af", args.engine_name],
                          capture_output=True, text=True)
    all_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    lines = [line for line in all_lines if needle in line]
    payload = {
        "schema": SCHEMA_PROC, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("this round's checklist: 0 engine processes of this round "
                        "may survive its own run"),
        "engine_name": args.engine_name, "data_dir_filter": needle,
        "pgrep_returncode": proc.returncode,
        "processes_total_by_engine_name": len(all_lines),
        "matching": len(lines), "lines": lines,
        "all_engine_processes_seen": len(all_lines),
        "note": ("other rounds' engines are counted but never touched; only the "
                 "lines above carry this round's --data-dir"),
    }
    payload["all_pass"] = bool(payload["matching"] == 0)
    write_json(out, payload)
    log(f"process-audit: {payload['matching']} own engine process(es) alive "
        f"(of {len(all_lines)} on the box) -> {out}")
    return 0 if payload["all_pass"] else 6


def mode_manifest(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.dir)
    if not root.is_dir():
        raise DistillError(f"not a directory: {root}")
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
    wanted: list[tuple[pathlib.Path, str]] = [
        (targets_path(root), "rule_targets.jsonl"),
        (targets_stats_path(root), "rule_target_stats.json"),
        (root / "preflight.json", "preflight.json"),
        (root / "train_summary.json", "train_summary.json"),
        (root / "diagnostics.json", "diagnostics.json"),
        (root / "bomb_usage.json", "bomb_usage.json"),
        (root / "ckpt_assertions.json", "ckpt_assertions.json"),
        (root / "ckpt_assertions_final.json", "ckpt_assertions_final.json"),
        (root / "ckpt_manifest.json", "ckpt_manifest.json"),
        (root / "audio_audit.json", "audio_audit.json"),
        (root / "process_residue.json", "process_residue.json"),
        (root / "process_residue_final.json", "process_residue_final.json"),
        (root / "fidelity_d3.json", "fidelity_d3.json"),
        (root / "v830_honoured.json", "v830_honoured.json"),
        (root / "commands.tsv", "commands.tsv"),
    ]
    for name in ("exec.sh", "chain.sh", "chain2.sh", "grid.sh", "status.sh",
                 "finish.sh", "tables.py", "verify.py", "v830_honoured.py"):
        wanted.append((root / "scripts" / name, f"scripts/{name}"))
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
               "file_count": len(copied),
               "v850_distill_sha256": sha256_file(pathlib.Path(__file__).resolve())}
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
        description="v8.5.0 DISTILL-001: distil the R5 bomb rule into the "
                    "36-action policy (counterfactual labels + rule supervision)")
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "assert-ckpt", "targets", "train",
                                 "train-summary", "eval", "diagnostics",
                                 "ckpt-manifest", "audio-audit", "process-audit",
                                 "manifest", "publish"))
    parser.add_argument("--seed", type=int, default=0, help="training seed (0/1/2)")
    parser.add_argument("--policy-seed", type=int, default=None,
                        help="eval: alias of --seed")
    parser.add_argument("--windows", default="11,12,17,18")
    parser.add_argument("--decisions", type=int, default=300,
                        help=f"eval budget: one of {BUDGETS}")
    parser.add_argument("--decision-frames", type=int, default=DECISION_FRAMES)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--kind", default="stochastic",
                        choices=("stochastic", "greedy"))
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--work-dir", default=str(WORK))
    parser.add_argument("--out", default="")
    parser.add_argument("--stats-out", default="")
    parser.add_argument("--usage-out", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--phase", default="start", choices=("start", "final"))
    parser.add_argument("--engine-name", default="th08-modern")
    parser.add_argument("--data-dir-filter", default=".tmp_v850_distill")
    parser.add_argument("--dir", default="")
    args = parser.parse_args(argv)

    if args.policy_seed is not None:
        args.seed = int(args.policy_seed)
    else:
        args.policy_seed = int(args.seed)

    if args.mode in ("eval",):
        prepare_env()
        tag = args.tag or f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
        args.tag = tag
        log_dir = pathlib.Path(args.work_dir) / "runs" / tag
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_FH = (log_dir / "run.log").open("a", encoding="utf-8")
        m830().v450()._LOG_FH = _LOG_FH
        m830().m432()._LOG_FH = _LOG_FH
        m830()._LOG_FH = _LOG_FH

    try:
        if args.mode == "preflight":
            return mode_preflight(args)
        if args.mode == "assert-ckpt":
            return mode_assert(args)
        if args.mode == "targets":
            return mode_targets(args)
        if args.mode == "train":
            return mode_train(args)
        if args.mode == "train-summary":
            return mode_train_summary(args)
        if args.mode == "eval":
            return mode_eval(args)
        if args.mode == "diagnostics":
            return mode_diagnostics(args)
        if args.mode == "ckpt-manifest":
            return mode_ckpt_manifest(args)
        if args.mode == "audio-audit":
            return mode_audio_audit(args)
        if args.mode == "process-audit":
            return mode_process_audit(args)
        if args.mode == "manifest":
            return mode_manifest(args)
        return mode_publish(args)
    except DistillError as exc:
        log(f"FAILED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
