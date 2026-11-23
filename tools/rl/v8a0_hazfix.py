#!/usr/bin/env python3
"""v8.10.0 HAZFIX-001 — replay the v8.6.0 trajectories and record the *hazard*
fields they never stored, then (P0 permitting) distil the R5 rule.

Executor for `Document/Update/Alpha/v8.10.0/PREREGISTRATION_HAZFIX_001.md`
(FROZEN 2026-11-22, before any v8.10.0 replay, training or evaluation).

What this is
------------
The prereg records two defects it wants repaired:

  1. "v8.6.0's obs lost the nearest bullets" — the stored 12 bullets are
     ``selection.bullets_stored=12`` out of ``bullet_count=217``, and the v8.9.0
     recount measured the stored minimum as larger than the true minimum on
     1,843/4,155 rows;
  2. "the rule target cannot be computed offline from v8.6's labels".

This round **re-walks the same 5,040 states** with the same seeds / windows /
prefix / behaviour policy (``v4.5.0 S`` stochastic + R5 staged, the frozen v8.6.0
protocol), records the missing hazard fields per state, and reconciles the replay
against the v8.6.0 label rows state by state.

It re-labels **nothing**.  The 36-way counterfactual labels of v8.6.0 are reused
as they are; the 36-way labeller (``v830_bombpol.label_state36``) is never
called, so no child is spawned and the label semantics cannot move.

Modes
-----
======================  =======================================================
``--mode preflight``    the frozen prereg's sha256, this file's sha256, the
                        sha256 of every imported module, the engine, the feature
                        dim, the R5 table, and the v8.6.0 label file's sha256.
``--mode assert-ckpt``  the three frozen ``round2.pt`` (``4abee069`` /
                        ``a93746bb`` / ``c69960fc``) that are the behaviour
                        policy of every replayed state, plus the engine and the
                        protected sources.
``--mode replay``       one (seed, window) cell: HELLO + the frozen 8,663-frame
                        prefix + PARK, then roll the behaviour policy for up to
                        300 decisions per episode and 420 states per cell — the
                        same stop rule as ``v860_bombdag --mode collect`` — and
                        record, per state:
                        ``nearest_all_px`` (min over **all** ``HAZARDS``
                        bullets), ``nearest_index``, ``bullet_count``,
                        ``nearest_lethal_px``, the nearest 12 bullets sorted by
                        distance (with vx/vy), ``player_state``, ``laser_*``, and
                        the ``features_sha256_k12`` of the frozen 106-dim vector.
``--mode reconcile``    match the replay to the v8.6.0 label rows on
                        (seed, window, episode, decision) and report the
                        ``features_sha256_k12`` hit rate (prereg P0 #1), the
                        frame/bombs agreement, and the live ``nearest_all_px``
                        against the ``behaviour.nearest_all_px`` that v8.6.0
                        already recorded.
``--mode rule-stats``   apply R5 (3 -> 80 px, 2 -> 60 px, 1 -> 40 px, strict
                        ``<``, ``bombs > 0``) to the replayed ``nearest_all_px``
                        and count positives / negatives per segment (P0 #2).
``--mode p0``           the two P0 conditions and the prereg's verdict
                        (``go`` / ``S-data``).
``--mode audio-audit``  AGENTS.md 9.3 on this round's own data dirs.
``--mode process-audit``this round's own engine processes must be 0.
``--mode manifest``     sha256 + bytes of every file in a directory.
``--mode publish``      copy the raw artifacts into
                        ``Document/Update/Alpha/v8.10.0/evidence/HAZFIX/``.

Reused, never re-implemented (imported by path, sha256 pinned)
-------------------------------------------------------------
* ``tools/rl/v860_bombdag.py`` — the **replay loop** itself:
  ``load_behaviour_policy`` (the frozen ``round2.pt`` + its sha256),
  ``generator_seed`` (``seed*1000+7``), ``behaviour_decision`` (the frozen
  106-dim features of the same state, the stochastic 18-way candidate, R5's
  bomb bit, and the live ``nearest_all_px``), ``start_parent``,
  ``check_pins`` and the frozen constants ``TRAIN_WINDOWS`` / ``H_DEFAULT`` /
  ``DECISIONS`` / ``QUOTA_DEFAULT`` / ``MAX_EPISODES_DEFAULT`` /
  ``DECISION_FRAMES`` / ``R5_THRESHOLDS`` / ``FROZEN_ROUND2_SHA256``.
  The cell's stop rule, the warm RESET per episode, the prefix walk, the
  2-frame decision and the death rule are transcribed line for line from
  ``v860_bombdag --mode collect``; the only change is what is written per state.
* ``tools/rl/v450_cf_label.py`` — ``prepare_env`` (dummy audio, no display),
  ``prepare_data_dir`` (a freshly generated muted ``th08.cfg`` per engine start),
  ``masks_for``, ``mode_audit`` (the audio audit).
* ``tools/rl/v800_bomb.py`` — ``nearest_bullet_distances`` and ``rule_gate``
  (reached through ``v860_bombdag.behaviour_decision``).
* ``touhou_rl/v400/gym/protocol.py`` — ``parse_hazards_reply``; the parsed
  ``HAZARDS`` frame is the *only* source of the recorded hazard fields.
* ``touhou_rl/v400/labels/state.py`` — ``player_state_name``.

Not in this file
----------------
* No statistics: counts, rates, quantiles and distributions only.  P1-P4 belong
  to the independent analysis pass (prereg 2).
* No training and no evaluation: prereg ``S-data`` says a failed P0 stops the
  round after the replay reconciliation and the rule statistics.
* No file under ``touhou_rl/``, ``third_party/`` or an existing version
  directory is written, and no existing artifact is moved or deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import subprocess
import sys
import time
from typing import Any, Sequence

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WORK = REPO / ".tmp_v8a0_hazfix"
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.10.0" / "evidence"
           / "HAZFIX")
PREREG = (REPO / "Document" / "Update" / "Alpha" / "v8.10.0"
          / "PREREGISTRATION_HAZFIX_001.md")
PREREG_REL = "Document/Update/Alpha/v8.10.0/PREREGISTRATION_HAZFIX_001.md"

V860_PATH = REPO / "tools" / "rl" / "v860_bombdag.py"
V860_LABELS = REPO / ".tmp_v860_bombdag" / "labels_bombdag.jsonl"
#: sha256 of `.tmp_v860_bombdag/labels_bombdag.jsonl`, the file v8.9.0's
#: independent recount pinned.  The replay is reconciled against exactly this
#: file; a moved file is exit 2 and nothing runs.
V860_LABELS_SHA256 = (
    "e1f47805efdd37bb92d15c51898aacbef0ca13995a200d7d2e087cd919543c68")

# -- the pinned imports: the whole round is void if any of these moved ---------
PINNED = {
    "tools/rl/v860_bombdag.py":
        "3433ae743cc224cae436b6e2192654e30eabe5f05f9053218969e52bdc47cf8a",
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
    "touhou_rl/v420/contracts.py":
        "1357c69c32a74fda20553a3e6b290ca2102b7cc5c52e47e9fdc6eb9e08c946c1",
    "touhou_rl/v400/gym/actions.py":
        "a67145b2bec7c96b3aacaa2514e78cdadf50b056473d15b4ae8122c19c393eff",
}

SCHEMA_PREFLIGHT = "v8a0-hazfix-preflight/1"
SCHEMA_ASSERT = "v8a0-hazfix-ckpt-assertions/1"
SCHEMA_REPLAY = "v8a0-hazfix-replay/1"
SCHEMA_REPLAY_CELL = "v8a0-hazfix-replay-cell/1"
SCHEMA_RECONCILE = "v8a0-hazfix-reconciliation/1"
SCHEMA_RULE = "v8a0-hazfix-rule-target-stats/1"
SCHEMA_P0 = "v8a0-hazfix-p0/1"
SCHEMA_AUDIO = "v8a0-hazfix-audio-audit/1"
SCHEMA_PROC = "v8a0-hazfix-process-audit/1"
SCHEMA_MANIFEST = "v8a0-hazfix-manifest/1"
SCHEMA_PUBLISH = "v8a0-hazfix-publish/1"
PRODUCER = ("tools/rl/v8a0_hazfix.py (live engine, HarnessClient role=shadow, "
            "behaviour policy = v4.5.0 `S` stochastic + R5 staged, replayed to "
            "record the hazard fields v8.6.0 did not store)")

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- R5: v8.4.0's staged rule, unchanged (reached through v800_bomb) -----------
R5_RULE = "staged"
R5_THRESHOLDS: dict[int, float] = {3: 80.0, 2: 60.0, 1: 40.0}
R5_SOURCE = ("Document/Update/Alpha/v8.4.0/PREREGISTRATION_BOMBRULE2_001.md 1 "
             "(tools/rl/v800_bomb.py::rule_gate / STAGED_THRESHOLDS)")

# -- P0 (prereg 2): both conditions, or S-data -------------------------------
P0_MIN_HIT_RATE = 0.99
P0_MIN_POSITIVES = 50
P0_MIN_NEGATIVES = 50

_LOG_FH: Any = None
_MODS: dict[str, Any] = {}


class HazFixError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
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


def _cached(name: str, path: pathlib.Path) -> Any:
    if name not in _MODS:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise HazFixError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODS[name] = module
    return _MODS[name]


def v860() -> Any:
    """The v8.6.0 executor: the replay loop this round reuses."""

    return _cached("v860_bombdag", V860_PATH)


def v450() -> Any:
    return _cached("v450_cf_label", REPO / "tools" / "rl" / "v450_cf_label.py")


def v830() -> Any:
    return _cached("v830_bombpol", REPO / "tools" / "rl" / "v830_bombpol.py")


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


def write_jsonl_line(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def quantiles(values: Sequence[float]) -> dict[str, Any]:
    return dict(v860().quantiles([float(v) for v in values]))


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def replay_cell_path(root: pathlib.Path, seed: int, window: int) -> pathlib.Path:
    return root / "replay" / f"seed{int(seed)}_w{int(window)}.jsonl"


def replay_cell_summary_path(root: pathlib.Path, seed: int,
                             window: int) -> pathlib.Path:
    return root / "replay" / f"seed{int(seed)}_w{int(window)}.json"


def hazard_fields_path(root: pathlib.Path) -> pathlib.Path:
    return root / "evidence" / "hazard_fields.jsonl"


def reconciliation_path(root: pathlib.Path) -> pathlib.Path:
    return root / "evidence" / "replay_reconciliation.json"


def rule_stats_path(root: pathlib.Path) -> pathlib.Path:
    return root / "evidence" / "rule_target_stats.json"


def p0_path(root: pathlib.Path) -> pathlib.Path:
    return root / "evidence" / "p0.json"


# --------------------------------------------------------------------------- #
# the hazard fields this round exists to record
# --------------------------------------------------------------------------- #
def hazard_fields(*, state: dict[str, Any], hazards: Any,
                  behaviour: dict[str, Any]) -> dict[str, Any]:
    """Every distance the v8.6.0 obs did **not** carry, read off live HAZARDS.

    ``nearest_all_px`` is the quantity R5's ``rule_gate`` tests: the minimum
    Euclidean distance from the player to a bullet over the **whole**
    ``HAZARDS`` bullet list.  The stored obs only ever held the nearest 12
    **lethal** bullets (``compact_observation`` filters ``b.lethal``), which is
    why the two readings disagree on the states that carry a closer, harmless
    bullet.

    ``nearest12`` is the distance-sorted prefix (ties broken by slot index) with
    the same five numbers the obs stores plus the slot, the bullet state and the
    lethality flag, so a downstream arm can rebuild the frozen 106-dim vector
    from *this* file alone.
    """

    px = float(state.get("player_x") or 0.0)
    py = float(state.get("player_y") or 0.0)
    rows: list[tuple[float, int, Any]] = []
    nearest_lethal: float | None = None
    nearest_lethal_index: int | None = None
    for index, bullet in enumerate(hazards.bullets):
        dx = float(bullet.position[0]) - px
        dy = float(bullet.position[1]) - py
        distance = math.hypot(dx, dy)
        rows.append((distance, index, bullet))
        if getattr(bullet, "lethal", False) and (nearest_lethal is None
                                                or distance < nearest_lethal):
            nearest_lethal = distance
            nearest_lethal_index = index
    rows.sort(key=lambda item: (item[0], item[1]))
    nearest_all = rows[0][0] if rows else None
    nearest_index = rows[0][1] if rows else None
    nearest12 = [{"index": int(index),
                  "slot": int(getattr(bullet, "slot", -1)),
                  "x": float(bullet.position[0]),
                  "y": float(bullet.position[1]),
                  "vx": float(bullet.velocity[0]),
                  "vy": float(bullet.velocity[1]),
                  "half_width": float(getattr(bullet, "half_width", 0.0) or 0.0),
                  "dist_px": round(float(distance), 4),
                  "state": int(getattr(bullet, "state", -1)),
                  "lethal": bool(getattr(bullet, "lethal", False))}
                 for distance, index, bullet in rows[:12]]

    lasers = [lz for lz in getattr(hazards, "lasers", ()) or ()
              if getattr(lz, "in_use", False)]
    laser_nearest = None
    if lasers:
        laser_nearest = round(min(
            math.hypot(float(lz.position[0]) - px,
                       float(lz.position[1]) - py) for lz in lasers), 4)

    from touhou_rl.v400.labels.state import player_state_name

    player_state = int(state.get("player_state") or 0)
    return {
        "frame": int(state.get("frame") or 0),
        "bombs_remaining": int(state.get("bombs_remaining") or 0),
        "player_x": px, "player_y": py,
        "player_state": player_state,
        "player_state_name": player_state_name(player_state),
        "invulnerable": bool(player_state == 3),
        "bullet_count": int(len(hazards.bullets)),
        "hazards_bullet_count_field": int(getattr(hazards, "bullet_count", 0) or 0),
        "laser_count": int(getattr(hazards, "laser_count", 0) or 0),
        "laser_in_use": len(lasers),
        "laser_nearest_px": laser_nearest,
        "laser_note": ("distance to the nearest in-use laser's origin, not its "
                       "beam geometry"),
        "nearest_all_px": None if nearest_all is None else round(nearest_all, 4),
        "nearest_index": nearest_index,
        "nearest_lethal_px": (None if nearest_lethal is None
                              else round(nearest_lethal, 4)),
        "nearest_lethal_index": nearest_lethal_index,
        "nearest12": nearest12,
        "nearest12_count": len(nearest12),
        "lethal_bullets": sum(1 for _d, _i, b in rows if getattr(b, "lethal", False)),
        "features_sha256_k12": behaviour["features_sha256_k12"],
        "behaviour": {"cand": int(behaviour["cand"]), "bomb": int(behaviour["bomb"]),
                      "action": int(behaviour["action"]),
                      "threshold_px": float(behaviour["threshold_px"]),
                      "rule": R5_RULE},
    }


# --------------------------------------------------------------------------- #
# mode: preflight
# --------------------------------------------------------------------------- #
def mode_preflight(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    pins: dict[str, Any] = {"pinned": {}, "all_match": True}
    for rel, expected in PINNED.items():
        path = REPO / rel
        got = sha256_file(path) if path.is_file() else None
        match = got == expected
        pins["pinned"][rel] = {"sha256": got, "expected": expected,
                               "match": match}
        pins["all_match"] = bool(pins["all_match"] and match)
    labels_sha = sha256_file(V860_LABELS) if V860_LABELS.is_file() else None
    labels_ok = bool(labels_sha == V860_LABELS_SHA256)
    engine = dict(v860().engine_gate(args.engine))
    rec: dict[str, Any] = {
        "schema": SCHEMA_PREFLIGHT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": {"path": PREREG_REL, "exists": PREREG.is_file(),
                   "sha256": sha256_file(PREREG) if PREREG.is_file() else None,
                   "bytes": PREREG.stat().st_size if PREREG.is_file() else None},
        "self": {"path": str(pathlib.Path(__file__).resolve()),
                 "sha256": sha256_file(pathlib.Path(__file__).resolve())},
        "pins": pins, "engine": engine, "launch_env": v450().prepare_env(),
        "feature_dim": int(_cached("v431_arm_a",
                                   REPO / "tools" / "rl" / "v431_arm_a.py"
                                   ).feature_dim()),
        "r5": {"rule": R5_RULE,
               "thresholds_px": {str(k): v for k, v in R5_THRESHOLDS.items()},
               "gate": "nearest_all_px < threshold and bombs_remaining > 0",
               "source": R5_SOURCE},
        "replay": {
            "source": "tools/rl/v860_bombdag.py --mode collect (same seeds, "
                      "windows, prefix, behaviour policy, stop rule)",
            "v860_label_file": {"path": str(V860_LABELS.relative_to(REPO)
                                            ).replace("\\", "/"),
                                "sha256": labels_sha,
                                "expected": V860_LABELS_SHA256,
                                "match": labels_ok},
            "labels_reused_not_recollected": True,
            "relabeled_actions": 0},
        "p0": {"min_hit_rate": P0_MIN_HIT_RATE,
               "min_positives_per_segment": P0_MIN_POSITIVES,
               "min_negatives_per_segment": P0_MIN_NEGATIVES},
    }
    ok = bool(pins["all_match"]) and bool(engine.get("match")) and labels_ok
    rec["all_pass"] = ok
    out = pathlib.Path(args.out) if args.out else root / "preflight.json"
    write_json(out, rec)
    log(f"preflight: pins={pins['all_match']} engine={engine.get('match')} "
        f"labels={labels_ok} feature_dim={rec['feature_dim']} all_pass={ok} -> {out}")
    return 0 if ok else 2


# --------------------------------------------------------------------------- #
# mode: assert-ckpt
# --------------------------------------------------------------------------- #
def mode_assert(args: argparse.Namespace) -> int:
    out_path = (pathlib.Path(args.out) if args.out
                else pathlib.Path(args.work_dir) / "ckpt_assertions.json")
    frozen = dict(v860().FROZEN_ROUND2_SHA256)
    ckpt_root = REPO / ".tmp_v450_cf" / "ckpt"
    payload: dict[str, Any] = {
        "schema": SCHEMA_ASSERT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": args.phase,
        "requirement": ("the behaviour policy of every replayed state is the "
                        "frozen v4.5.0 `S` student of its seed: round2.pt = "
                        "4abee069 / a93746bb / c69960fc"),
        "engine": None, "ckpts": {}, "protected": {}, "v860_bombdag_py": {},
        "prereg": {}, "all_pass": False,
    }
    try:
        payload["engine"] = dict(v860().engine_gate(args.engine))
        ok = bool(payload["engine"].get("match"))
        for seed in sorted(frozen):
            path = ckpt_root / f"seed{int(seed)}" / "round2.pt"
            expected = frozen[seed]
            digest = sha256_file(path) if path.is_file() else None
            match = bool(digest == expected)
            ok = ok and match
            payload["ckpts"][str(seed)] = {
                "path": str(path), "exists": path.is_file(), "sha256": digest,
                "expected": expected, "expected_prefix": expected[:8],
                "match": match,
                "bytes": path.stat().st_size if path.is_file() else None}
        for rel in v860().PROTECTED_FILES:
            path = REPO / rel
            payload["protected"][rel] = {
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None}
        payload["v860_bombdag_py"] = {"path": str(V860_PATH),
                                      "sha256": sha256_file(V860_PATH)}
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
# mode: replay (v860_bombdag --mode collect, minus the labels, plus the hazards)
# --------------------------------------------------------------------------- #
def mode_replay(args: argparse.Namespace) -> int:
    import torch

    from touhou_rl.v400.gym import protocol as P
    from touhou_rl.v420 import counterfactual as cf

    mod = v860()
    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    windows = mod.parse_windows(args.windows)
    decisions_cap = int(args.decisions)
    quota = int(args.quota)
    max_episodes = int(args.max_episodes)
    if seed not in mod.TRAIN_SEEDS:
        raise HazFixError(f"--seed must be one of {mod.TRAIN_SEEDS}, got {seed}")
    if int(args.horizon) != mod.H_DEFAULT:
        raise HazFixError(f"prereg fixes H = {mod.H_DEFAULT}; got {args.horizon}")
    if decisions_cap != mod.DECISIONS:
        raise HazFixError(f"the roll-out is capped at {mod.DECISIONS} decisions "
                          f"per episode (the frozen protocol); got {decisions_cap}")
    for window in windows:
        if window not in mod.TRAIN_WINDOWS:
            raise HazFixError(f"window {window} is not a train window "
                              f"{mod.TRAIN_WINDOWS}")
    pins = mod.check_pins()
    if not pins["all_match"]:
        raise HazFixError(f"pinned import moved: {json.dumps(pins)[:800]}")
    engine = dict(mod.engine_gate(args.engine))
    v860_sha = sha256_file(V860_PATH)
    labels_sha = sha256_file(V860_LABELS) if V860_LABELS.is_file() else None
    if labels_sha != V860_LABELS_SHA256:
        raise HazFixError(f"the v8.6.0 label file moved: {labels_sha} != "
                          f"{V860_LABELS_SHA256}")
    policy, ckpt, ckpt_sha = mod.load_behaviour_policy(seed)
    generator = torch.Generator()
    generator.manual_seed(mod.generator_seed(seed))
    launch_env = v450().prepare_env()

    for window in windows:
        tag = args.tag or f"v8a0-replay-s{seed}-w{window}"
        run_dir = root / "runs" / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        data_root = root / "data" / tag
        out_path = replay_cell_path(root, seed, window)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial = out_path.with_suffix(out_path.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        summary_path = (pathlib.Path(args.summary_out)
                        if args.summary_out and len(windows) == 1
                        else replay_cell_summary_path(root, seed, window))
        masks, hashes, _meta = v450().masks_for(window)
        rec: dict[str, Any] = {
            "schema": SCHEMA_REPLAY_CELL, "producer": PRODUCER,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "prereg": PREREG_REL, "seed": seed, "window": int(window),
            "quota": quota, "decisions_cap": decisions_cap,
            "max_episodes": max_episodes, "H": int(args.horizon),
            "replay_of": ("tools/rl/v860_bombdag.py --mode collect --seed "
                          f"{seed} --windows {window} --quota {quota} --decisions "
                          f"{decisions_cap}"),
            "relabelled_actions": 0,
            "behaviour": {
                "policy": "v4.5.0 `S` (frozen DAgger student, round2.pt)",
                "ckpt": str(ckpt), "ckpt_sha256": ckpt_sha,
                "kind": "stochastic",
                "generator_seed": mod.generator_seed(seed),
                "bomb_rule": R5_RULE,
                "thresholds_px": {str(k): v for k, v in R5_THRESHOLDS.items()},
                "bomb_rule_gate": ("nearest_all_px < threshold and "
                                   "bombs_remaining > 0"),
                "bomb_rule_source": R5_SOURCE},
            "pins": pins, "engine": engine, "launch_env": launch_env,
            "v860_bombdag_sha256": v860_sha,
            "v860_labels_sha256": labels_sha,
            "source": {"masks": len(masks), "hashes": len(hashes)},
            "out_path": str(out_path), "status": "failed", "error": None,
            "episodes": [], "states": 0, "bombs_remaining_counts": {},
            "bomb_emits": 0, "thresholds_used": {}, "decision_counts": {},
            "player_state_counts": {}, "empty_bullet_states": 0,
            "states_with_laser": 0, "gate": {}, "cfg": {}, "wall_s": None,
        }
        handle = partial.open("w", encoding="utf-8")
        parents: list[Any] = []
        started = time.perf_counter()
        try:
            data_dir, data_info = v450().prepare_data_dir(
                data_root, window, args.data_source)
            rec["cfg"] = data_info["cfg"]
            rec["data_dir"] = str(data_dir)
            parent, gate = mod.start_parent(slot=0, window=window,
                                            engine=args.engine, data_dir=data_dir,
                                            log_dir=run_dir)
            parents.append(parent)
            rec["gate"] = {key: gate[key] for key in (
                "hello_role_shadow", "park_ok", "park_frame", "threads_1",
                "all_pass")}
            rec["gate"]["prefix"] = {key: gate["prefix"][key] for key in (
                "hash_checks", "hash_mismatches", "final_frame", "wall_s")}
            state, _reply = parent.state()
            if int(state["frame"]) != mod.PREFIX_FRAMES:
                raise HazFixError(f"w{window}: parked parent at frame "
                                  f"{state['frame']}, expected {mod.PREFIX_FRAMES}")
            deaths0 = float(state["deaths"] or 0.0)
            stage0 = int(state["stage"] or 0)
            episode = 0
            while rec["states"] < quota and episode < max_episodes:
                ep: dict[str, Any] = {
                    "episode": int(episode), "decisions": 0, "frames": 0,
                    "died": False, "terminal": None,
                    "bombs_start": int(state["bombs_remaining"] or 0),
                    "bombs_end": None, "bombs_used": None, "bomb_emits": 0,
                    "states_recorded": 0, "prefix": None}
                if episode > 0:
                    prefix = parent.walk_prefix(masks, hashes, window)
                    parent.park()
                    state, _reply = parent.state()
                    if int(state["frame"]) != mod.PREFIX_FRAMES:
                        raise HazFixError(
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
                    got = mod.behaviour_decision(state=state, hazards=hazards,
                                                 policy=policy,
                                                 generator=generator,
                                                 decision=decision)
                    # ---- the ONLY addition to the v8.6.0 collect loop --------
                    row = hazard_fields(state=state, hazards=hazards,
                                        behaviour=got)
                    row.update({
                        "schema": SCHEMA_REPLAY, "seed": seed,
                        "window": int(window), "episode": int(episode),
                        "decision": int(decision),
                        "state_frame": state_frame,
                        "behaviour_ckpt_sha256": ckpt_sha,
                        "generator_seed": mod.generator_seed(seed),
                        "v860_bombdag_sha256": v860_sha,
                    })
                    write_jsonl_line(handle, row)
                    key = str(int(got["bombs_before"]))
                    rec["bombs_remaining_counts"][key] = (
                        rec["bombs_remaining_counts"].get(key, 0) + 1)
                    rec["decision_counts"][mod.decision_bucket(decision)] = (
                        rec["decision_counts"].get(
                            mod.decision_bucket(decision), 0) + 1)
                    rec["player_state_counts"][row["player_state_name"]] = (
                        rec["player_state_counts"].get(
                            row["player_state_name"], 0) + 1)
                    if row["nearest_all_px"] is None:
                        rec["empty_bullet_states"] += 1
                    if row["laser_in_use"]:
                        rec["states_with_laser"] += 1
                    if got["bomb"]:
                        ep["bomb_emits"] += 1
                        rec["bomb_emits"] += 1
                        tkey = f"{float(got['threshold_px']):g}"
                        rec["thresholds_used"][tkey] = (
                            rec["thresholds_used"].get(tkey, 0) + 1)
                    rec["states"] += 1
                    ep["states_recorded"] += 1
                    # ---- the parent takes the behaviour action (unchanged) ---
                    first_mask, rest_mask = v830().action_masks(got["action"])
                    for frame_index in range(mod.DECISION_FRAMES):
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
                            f"{rec['bombs_remaining_counts']}")
                    if died:
                        break
                ep["bombs_end"] = int(state["bombs_remaining"] or 0)
                ep["bombs_used"] = int(ep["bombs_start"]) - int(ep["bombs_end"])
                rec["episodes"].append(ep)
                episode += 1
                log(f"s{seed} w{window} ep{ep['episode']}: {ep['decisions']} "
                    f"decisions, {ep['states_recorded']} states, bombs "
                    f"{ep['bombs_start']}->{ep['bombs_end']}, "
                    f"emits={ep['bomb_emits']}, died={ep['died']}, "
                    f"terminal={ep['terminal']}")
            parent.close()
            parents.remove(parent)
            rec["status"] = "complete" if rec["states"] >= quota else "short"
        except Exception as exc:                                      # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["status"] = "failed"
            log(f"REPLAY FAILED s{seed} w{window}: {rec['error']}")
        finally:
            handle.close()
            for parent in parents:
                try:
                    parent.close()
                except Exception:                                     # noqa: BLE001
                    pass
            rec["wall_s"] = round(time.perf_counter() - started, 3)
            if rec["status"] in ("complete", "short"):
                if out_path.exists():
                    out_path.unlink()
                partial.rename(out_path)
                rec["out_sha256"] = sha256_file(out_path)
                rec["out_bytes"] = out_path.stat().st_size
            write_json(summary_path, rec)
        log(f"replay s{seed} w{window}: status={rec['status']} "
            f"states={rec['states']} bombs={rec['bombs_remaining_counts']} "
            f"player={rec['player_state_counts']} empty={rec['empty_bullet_states']} "
            f"wall={rec['wall_s']}s -> {out_path}")
        if rec["status"] == "failed":
            return 4
    return 0


# --------------------------------------------------------------------------- #
# mode: reconcile (P0 #1)
# --------------------------------------------------------------------------- #
def mode_reconcile(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    cell_files = sorted((root / "replay").glob("seed*_w*.jsonl"))
    if not cell_files:
        raise HazFixError(f"no replay files under {root / 'replay'}")
    labels_sha = sha256_file(V860_LABELS)
    if labels_sha != V860_LABELS_SHA256:
        raise HazFixError(f"the v8.6.0 label file moved: {labels_sha}")
    replay: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    per_cell: dict[str, Any] = {}
    for path in cell_files:
        key_name = path.name
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                replay[(int(row["seed"]), int(row["window"]),
                        int(row["episode"]), int(row["decision"]))] = row
                count += 1
        per_cell[key_name] = {"rows": count, "sha256": sha256_file(path)}

    total = 0
    matched = 0
    feature_hits = 0
    frame_hits = 0
    bombs_hits = 0
    nearest_compared = 0
    nearest_equal = 0
    nearest_deltas: list[float] = []
    stored_deltas: list[float] = []
    missing: list[dict[str, Any]] = []
    feature_misses: list[dict[str, Any]] = []
    per_segment: dict[str, Any] = {}
    with V860_LABELS.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            key = (int(row["seed"]), int(row["window"]), int(row["episode"]),
                   int(row["decision"]))
            mine = replay.get(key)
            if mine is None:
                if len(missing) < 20:
                    missing.append({"seed": key[0], "window": key[1],
                                    "episode": key[2], "decision": key[3]})
                continue
            matched += 1
            if mine["features_sha256_k12"] == row["features_sha256_k12"]:
                feature_hits += 1
            elif len(feature_misses) < 20:
                feature_misses.append({
                    "seed": key[0], "window": key[1], "episode": key[2],
                    "decision": key[3],
                    "v860": row["features_sha256_k12"],
                    "replay": mine["features_sha256_k12"],
                    "v860_frame": row["frame"], "replay_frame": mine["frame"]})
            if int(mine["frame"]) == int(row["frame"]):
                frame_hits += 1
            if int(mine["bombs_remaining"]) == int(row["bombs_remaining"]):
                bombs_hits += 1
            stored = (row.get("behaviour") or {}).get("nearest_all_px")
            live = mine["nearest_all_px"]
            if stored is not None and live is not None:
                nearest_compared += 1
                delta = abs(float(stored) - float(live))
                nearest_deltas.append(delta)
                if delta < 1e-3:
                    nearest_equal += 1
            segment = per_segment.setdefault(str(int(row["bombs_remaining"])),
                                             {"n": 0, "feature_hits": 0})
            segment["n"] += 1
            if mine["features_sha256_k12"] == row["features_sha256_k12"]:
                segment["feature_hits"] += 1

    hit_rate = (feature_hits / total) if total else 0.0
    payload: dict[str, Any] = {
        "schema": SCHEMA_RECONCILE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL,
        "requirement": ("P0 #1: every replayed state must reproduce the v8.6.0 "
                        "label row's features_sha256_k12; hit rate >= "
                        f"{P0_MIN_HIT_RATE}"),
        "v860_labels": {"path": str(V860_LABELS), "sha256": labels_sha,
                        "states": total},
        "replay_cells": per_cell,
        "states_replayed": len(replay),
        "states_matched": matched,
        "features_sha256_k12": {"hits": feature_hits, "states": total,
                                "hit_rate": round(hit_rate, 6),
                                "threshold": P0_MIN_HIT_RATE,
                                "pass": bool(hit_rate >= P0_MIN_HIT_RATE)},
        "frame_equal": {"hits": frame_hits, "states": total,
                        "rate": round(frame_hits / total, 6) if total else None},
        "bombs_remaining_equal": {"hits": bombs_hits, "states": total,
                                  "rate": round(bombs_hits / total, 6)
                                  if total else None},
        "nearest_all_px_vs_v860_behaviour_field": {
            "compared": nearest_compared, "equal_1e-3": nearest_equal,
            "rate": round(nearest_equal / nearest_compared, 6)
            if nearest_compared else None,
            "abs_delta_px": quantiles(nearest_deltas)},
        "per_segment": per_segment,
        "missing_keys": missing, "feature_misses": feature_misses,
        "note": ("the v8.6.0 rows carry `behaviour.nearest_all_px`, computed at "
                 "collection time by v800_bomb.nearest_bullet_distances over the "
                 "whole HAZARDS list; the replayed value is measured "
                 "independently and compared here, but it is NOT the P0 "
                 "condition"),
    }
    out = pathlib.Path(args.out) if args.out else reconciliation_path(root)
    write_json(out, payload)
    log(f"reconcile: {feature_hits}/{total} feature hashes hit "
        f"({hit_rate:.4%}); frames {frame_hits}/{total}; nearest equal "
        f"{nearest_equal}/{nearest_compared} -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# mode: rule-stats (P0 #2)
# --------------------------------------------------------------------------- #
def rule_bomb(*, bombs_remaining: int, nearest_all_px: float | None) -> int:
    """R5 (`v800_bomb.rule_gate("staged")`) applied to a recorded distance.

    ``v8.4.0 BOMBRULE2-001 1``: the threshold is 80 px while ``bombs == 3``, 60 px
    at 2 and 40 px at 1; the trigger is ``nearest_all_px < threshold`` **and**
    ``bombs_remaining > 0``.  The comparison is strict, exactly as
    ``rule_gate``'s caller does it, and a state with no bullet at all never
    fires.
    """

    if nearest_all_px is None or int(bombs_remaining) <= 0:
        return 0
    threshold = R5_THRESHOLDS.get(int(bombs_remaining))
    if threshold is None:
        return 0
    return int(float(nearest_all_px) < float(threshold))


def mode_rule_stats(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    cell_files = sorted((root / "replay").glob("seed*_w*.jsonl"))
    if not cell_files:
        raise HazFixError(f"no replay files under {root / 'replay'}")
    segments: dict[int, dict[str, Any]] = {}
    nearest_all_by_bombs: dict[int, list[float]] = {}
    nearest_lethal_by_bombs: dict[int, list[float]] = {}
    rule_pos: dict[int, int] = {}
    rule_neg: dict[int, int] = {}
    total = 0
    empty = 0
    laser_rows = 0
    invuln_rows = 0
    player_states: dict[str, int] = {}
    for path in cell_files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total += 1
                bombs = int(row["bombs_remaining"])
                slot = segments.setdefault(bombs, {"n": 0, "pos": 0, "neg": 0,
                                                   "empty_bullets": 0,
                                                   "laser_rows": 0,
                                                   "invulnerable_rows": 0,
                                                   "nearest_all": [],
                                                   "nearest_lethal": [],
                                                   "distance_le_80": 0,
                                                   "distance_le_60": 0,
                                                   "distance_le_40": 0})
                slot["n"] += 1
                nearest = row["nearest_all_px"]
                lethal = row["nearest_lethal_px"]
                if nearest is None:
                    empty += 1
                    slot["empty_bullets"] += 1
                else:
                    slot["nearest_all"].append(float(nearest))
                    nearest_all_by_bombs.setdefault(bombs, []).append(float(nearest))
                    if float(nearest) <= 80.0:
                        slot["distance_le_80"] += 1
                    if float(nearest) <= 60.0:
                        slot["distance_le_60"] += 1
                    if float(nearest) <= 40.0:
                        slot["distance_le_40"] += 1
                if lethal is not None:
                    slot["nearest_lethal"].append(float(lethal))
                    nearest_lethal_by_bombs.setdefault(bombs, []).append(float(lethal))
                if row.get("laser_in_use"):
                    laser_rows += 1
                    slot["laser_rows"] += 1
                if row.get("invulnerable"):
                    invuln_rows += 1
                    slot["invulnerable_rows"] += 1
                name = str(row.get("player_state_name") or "UNKNOWN")
                player_states[name] = player_states.get(name, 0) + 1
                fired = rule_bomb(bombs_remaining=bombs, nearest_all_px=nearest)
                if fired:
                    slot["pos"] += 1
                    rule_pos[bombs] = rule_pos.get(bombs, 0) + 1
                else:
                    slot["neg"] += 1
                    rule_neg[bombs] = rule_neg.get(bombs, 0) + 1

    payload: dict[str, Any] = {
        "schema": SCHEMA_RULE, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL,
        "requirement": ("P0 #2: each of the three R5 segments must hold >= "
                        f"{P0_MIN_POSITIVES} positives and >= "
                        f"{P0_MIN_NEGATIVES} negatives"),
        "rule": {"name": R5_RULE,
                 "thresholds_px": {str(k): v for k, v in R5_THRESHOLDS.items()},
                 "gate": ("nearest_all_px < threshold and bombs_remaining > 0"),
                 "comparison": "strict <", "source": R5_SOURCE,
                 "distance_field": "nearest_all_px (all HAZARDS bullets)"},
        "states": total, "empty_bullet_states": empty,
        "states_with_laser": laser_rows, "invulnerable_states": invuln_rows,
        "player_state_counts": player_states,
        "segments": {}, "segments_pass": {},
    }
    for bombs in sorted(segments, reverse=True):
        slot = segments[bombs]
        payload["segments"][str(bombs)] = {
            "bombs_remaining": bombs, "n": slot["n"],
            "rule_pos": slot["pos"], "rule_neg": slot["neg"],
            "positive_rate": round(slot["pos"] / slot["n"], 6) if slot["n"] else None,
            "empty_bullets": slot["empty_bullets"],
            "laser_rows": slot["laser_rows"],
            "invulnerable_rows": slot["invulnerable_rows"],
            "nearest_all_px": quantiles(slot["nearest_all"]),
            "nearest_lethal_px": quantiles(slot["nearest_lethal"]),
            "distance_le_80_px": slot["distance_le_80"],
            "distance_le_60_px": slot["distance_le_60"],
            "distance_le_40_px": slot["distance_le_40"],
        }
        if bombs in R5_THRESHOLDS:
            payload["segments_pass"][str(bombs)] = {
                "pos_ok": bool(slot["pos"] >= P0_MIN_POSITIVES),
                "neg_ok": bool(slot["neg"] >= P0_MIN_NEGATIVES),
                "pass": bool(slot["pos"] >= P0_MIN_POSITIVES
                             and slot["neg"] >= P0_MIN_NEGATIVES)}
    payload["nearest_all_px_by_bombs"] = {
        str(b): quantiles(v) for b, v in sorted(nearest_all_by_bombs.items(),
                                                reverse=True)}
    payload["nearest_lethal_px_by_bombs"] = {
        str(b): quantiles(v) for b, v in sorted(nearest_lethal_by_bombs.items(),
                                                reverse=True)}
    needed = [str(b) for b in sorted(R5_THRESHOLDS, reverse=True)]
    missing = [b for b in needed if b not in payload["segments"]]
    payload["all_segments_present"] = {"needed": needed, "missing": missing,
                                       "pass": not missing}
    payload["p0_segment_pass"] = bool(
        not missing and all(payload["segments_pass"][b]["pass"] for b in needed))
    out = pathlib.Path(args.out) if args.out else rule_stats_path(root)
    write_json(out, payload)
    for bombs in needed:
        slot = payload["segments"].get(bombs)
        if slot:
            log(f"rule-stats bombs={bombs}: n={slot['n']} pos={slot['rule_pos']} "
                f"neg={slot['rule_neg']} rate={slot['positive_rate']} "
                f"pass={payload['segments_pass'][bombs]['pass']}")
    log(f"rule-stats: states={total} segments_pass={payload['p0_segment_pass']} "
        f"-> {out}")
    return 0


# --------------------------------------------------------------------------- #
# mode: p0 (both conditions, or S-data)
# --------------------------------------------------------------------------- #
def mode_p0(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    recon = json.loads(reconciliation_path(root).read_text(encoding="utf-8"))
    rules = json.loads(rule_stats_path(root).read_text(encoding="utf-8"))
    cond1 = bool(recon["features_sha256_k12"]["pass"])
    cond2 = bool(rules["p0_segment_pass"])
    payload = {
        "schema": SCHEMA_P0, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL,
        "condition_1_replay_identity": {
            "requirement": f"features_sha256_k12 hit rate >= {P0_MIN_HIT_RATE}",
            "hit_rate": recon["features_sha256_k12"]["hit_rate"],
            "hits": recon["features_sha256_k12"]["hits"],
            "states": recon["features_sha256_k12"]["states"],
            "pass": cond1},
        "condition_2_rule_segments": {
            "requirement": (f"each of the 3 segments >= {P0_MIN_POSITIVES} "
                            f"positives and >= {P0_MIN_NEGATIVES} negatives"),
            "segments": {k: {"rule_pos": v["rule_pos"], "rule_neg": v["rule_neg"],
                             "pass": rules["segments_pass"].get(k, {}).get("pass")}
                         for k, v in rules["segments"].items()},
            "pass": cond2},
        "p0_pass": bool(cond1 and cond2),
        "verdict": "go" if (cond1 and cond2) else "S-data",
        "stop_condition": None if (cond1 and cond2) else (
            "prereg 2.1 S-data: P0 does not hold -> publish the replay "
            "reconciliation and the rule statistics only, do not train"),
    }
    out = pathlib.Path(args.out) if args.out else p0_path(root)
    write_json(out, payload)
    log(f"p0: cond1={cond1} cond2={cond2} -> verdict={payload['verdict']} -> {out}")
    return 0 if payload["p0_pass"] else 5


# --------------------------------------------------------------------------- #
# audit / manifest / publish
# --------------------------------------------------------------------------- #
def mode_audio_audit(args: argparse.Namespace) -> int:
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
    all_engines = [line for line in proc.stdout.splitlines() if line.strip()]
    payload = {
        "schema": SCHEMA_PROC, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("0 engine processes of this round may survive its own "
                        "run (filtered by --data-dir-filter)"),
        "engine_name": args.engine_name, "data_dir_filter": needle,
        "pgrep_returncode": proc.returncode,
        "processes_total_by_engine_name": len(all_engines),
        "matching": len(lines), "lines": lines,
        "note": ("processes_total_by_engine_name counts every th08-modern on the "
                 "host, including other rounds' evaluations; only `matching` is "
                 "this round's own residue"),
    }
    payload["all_pass"] = bool(payload["matching"] == 0)
    write_json(out, payload)
    log(f"process-audit: {payload['matching']} own engine process(es) alive, "
        f"{payload['processes_total_by_engine_name']} on the host -> {out}")
    return 0 if payload["all_pass"] else 6


def mode_manifest(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.dir)
    if not root.is_dir():
        raise HazFixError(f"not a directory: {root}")
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
        (root / "preflight.json", "preflight.json"),
        (root / "ckpt_assertions.json", "ckpt_assertions.json"),
        (root / "ckpt_assertions_final.json", "ckpt_assertions_final.json"),
        (root / "fidelity_d3.json", "fidelity_d3.json"),
        (hazard_fields_path(root), "hazard_fields.jsonl"),
        (reconciliation_path(root), "replay_reconciliation.json"),
        (rule_stats_path(root), "rule_target_stats.json"),
        (p0_path(root), "p0.json"),
        (root / "audio_audit.json", "audio_audit.json"),
        (root / "audio_audit_work.json", "audio_audit_work.json"),
        (root / "audio_audit_smoke.json", "audio_audit_smoke.json"),
        (root / "process_residue.json", "process_residue.json"),
        (root / "commands.tsv", "commands.tsv"),
    ]
    for name in ("exec.sh", "RUN.md"):
        wanted.append((root / name, name))
    for path in sorted((root / "replay").glob("seed*_w*.json")):
        wanted.append((path, f"replay/{path.name}"))
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
               "v8a0_hazfix_sha256": sha256_file(
                   pathlib.Path(__file__).resolve()),
               "v860_bombdag_sha256": sha256_file(V860_PATH)}
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
        description="v8.10.0 HAZFIX-001: replay the v8.6.0 trajectories, record "
                    "the hazard fields they never stored, reconcile, and apply "
                    "R5's rule to the recovered nearest-bullet distance")
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "assert-ckpt", "replay",
                                 "reconcile", "rule-stats", "p0", "audio-audit",
                                 "process-audit", "manifest", "publish"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--windows", default="13,14,15,16")
    parser.add_argument("--decisions", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--quota", type=int, default=420)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--work-dir", default=str(WORK))
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--phase", default="start", choices=("start", "final"))
    parser.add_argument("--engine-name", default="th08-modern")
    parser.add_argument("--data-dir-filter", default="")
    parser.add_argument("--dir", default="")
    args = parser.parse_args(argv)

    if args.mode == "replay":
        v450().prepare_env()
        tag = args.tag or f"replay-{time.strftime('%Y%m%d-%H%M%S')}"
        args.tag = tag
        log_dir = pathlib.Path(args.work_dir) / "runs" / tag
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_FH = (log_dir / "run.log").open("a", encoding="utf-8")

    try:
        if args.mode == "preflight":
            return mode_preflight(args)
        if args.mode == "assert-ckpt":
            return mode_assert(args)
        if args.mode == "replay":
            return mode_replay(args)
        if args.mode == "reconcile":
            return mode_reconcile(args)
        if args.mode == "rule-stats":
            return mode_rule_stats(args)
        if args.mode == "p0":
            return mode_p0(args)
        if args.mode == "audio-audit":
            return mode_audio_audit(args)
        if args.mode == "process-audit":
            return mode_process_audit(args)
        if args.mode == "manifest":
            return mode_manifest(args)
        if args.mode == "publish":
            return mode_publish(args)
    except HazFixError as exc:
        log(f"ERROR {exc}")
        return 2
    finally:
        if _LOG_FH is not None:
            _LOG_FH.close()
            _LOG_FH = None
    return 2


if __name__ == "__main__":
    sys.exit(main())
