#!/usr/bin/env python3
"""v8.7.0 BOMBLONG-001 鈥?**long-horizon** (H = 600) counterfactual labels for the
bomb decision, so that spending a bomb finally costs something.

What this is
------------
The executor for `Document/Update/Alpha/v8.7.0/PREREGISTRATION_BOMBLONG_001.md`
(FROZEN 2026-11-22, before any v8.7.0 label or training run).

v8.6.0 measured the decisive fact (prereg 搂0): over **48,168** `bomb=1` vs
`bomb=0` pairs at H = 120 (2 s), *bombing is never worse* 鈥?0 pairs out of 48,168
where the bomb arm lived shorter.  It is sometimes better (2,513) and mostly
identical.  So at a 2-second horizon "press the bomb now" is a free action: the
price of spending one (no bomb left later) happens **outside** the label's
window.  The state distribution was not the bottleneck; the **credit horizon**
was.

This round changes exactly **one** factor 鈥?the horizon of the bomb decision's
label:

    candidate direction   36-way label at H = 120, reused from v8.6.0 (unchanged)
    bomb decision         **H = 600** (300 decisions x 2 frames = 10 s), freshly
                          collected: for every v8.6.0 state take its
                          short-horizon best candidate, then run **only** that
                          candidate's bomb = 0 and bomb = 1 arms for 600 frames

    states   = the v8.6.0 label rows (`.tmp_v860_bombdag/labels/*.jsonl[.partial]`,
               `bombs_remaining` in {3, 2, 1})
    model    = Policy(106, 64, 36) 鈥?the frozen 2-layer shape
    train    = Adam 1e-3, batch 256, 200 epochs, 3 seeds;
               **candidate head** = the v8.6.0 36-way short-horizon BCE;
               **bomb head** = the long-horizon 2-arm label on the selected
               candidate's two logits (`survive_600` BCE; falls back to
               `alive_frames_600 / 600` MSE if one class exceeds 90 %)
    eval     = 3 budgets (300 / 600 / 900 decisions) x 3 seeds x 4 dev windows
               x 300 episodes, stochastic, live engine, 36 actions

How the long-horizon label is collected (and why it is a replay, not a fresh roll-out)
-------------------------------------------------------------------------------------
A label row only stores the *observation* of a state, not the engine process that
was at it.  Reaching that exact state again therefore means replaying the v8.6.0
behaviour roll-out: RESTART from the window's recorded prefix (RESET + the frozen
8,663-frame walk with its 2,165 checked hashes), then execute, decision by
decision, **the action the v8.6.0 row recorded** (`behaviour.action`).  The
engine is deterministic (v4.3.2 搂2.3 checks 2,165 hashes per walk), so the
replay lands on bit-identical states 鈥?and that is *verified*, not assumed: at
every decision the replayed `(state, HAZARDS)` is compared with the row
(`compact_observation` must be equal **and** the 106-dim `features_sha256_k12`
must be equal **and** the frame must be equal).  A replay that does not land on
the recorded state aborts the cell (rc 4, partial file kept) 鈥?nothing is
labelled on a state it cannot prove.

At each verified state the parent is PARKed and two children are spawned from the
template: `action = cand * 2 + bomb` for `bomb` in {0, 1}, each held for H = 600
frames (`cand_to_mask(cand, bomb)` on the first frame, `cand_to_mask(cand, 0)`
afterwards 鈥?the frozen schedule).  This is `v830_bombpol.label_state36` with
`actions = [2*cand, 2*cand+1]` and `horizon = 600`, i.e. the proven labeller
called with a smaller action set and a longer horizon; nothing is re-implemented.

Modes
-----
==================================  =============================================
``--mode preflight``                 the pinned imports (sha256 of every file this
                                     round imports **and** of the frozen prereg),
                                     the feature dim (106), the action table, the
                                     frozen design, for the record.
``--mode assert-ckpt --phase P``     engine sha256 + the three frozen v4.5.0
                                     ``round2.pt`` (``4abee069`` / ``a93746bb`` /
                                     ``c69960fc``) that are the behaviour-policy
                                     source of the v8.6.0 states, plus the sha256
                                     of every protected source.
``--mode fidelity``                  the D3 gate (v4.3.2 搂2.1-搂2.5, five
                                     assertions) via ``v432_fork_arm.py --mode
                                     fidelity --seeds 12``, in this round's own
                                     work dir.  rc 2 unless all five pass.
``--mode select``                    **offline**.  Choose each state's
                                     short-horizon best candidate from the v8.6.0
                                     36-way labels (``score(c) =
                                     alive_frames[2c]``, the bomb = 0 arm; ties ->
                                     smallest index) and write ``states.jsonl``:
                                     the v8.6.0 row verbatim plus ``cand`` /
                                     ``cand_short_alive`` / ``cand_short_alive_bomb1``
                                     / ``source``.  Every source file's sha256 is
                                     recorded; the snapshot is what the rest of the
                                     round reads, so a later v8.6.0 write cannot
                                     change it.
``--mode collect --seed S --windows W``
                                     one (seed, window) cell: the verified replay
                                     above + the two H = 600 arms per state.
                                     Writes ``labels/seed{S}_w{W}.jsonl`` (the
                                     ``.partial`` is renamed only when the cell
                                     completed).
``--mode stats``                     pool the cells into ``labels_long.jsonl`` and
                                     write ``labels_long_stats.json``: the
                                     **P0 table** (per ``bombs_remaining``:
                                     bomb = 1 strictly longer / equal / shorter at
                                     H = 600), the ``survive_600`` cross-tab, the
                                     replay-verification counts, and the
                                     label-cost distribution.  **No test.**
                                     rc 4 if P0 fails (prereg 搂2.1 ``S-horizon``:
                                     the round publishes labels and stops).
``--mode train --seed S``            Policy(106,64,36): 36-way BCE on the v8.6.0
                                     short labels + the long-horizon bomb head.
``--mode train-summary``             the three train reports side by side.
``--mode eval --decisions {300,600,900} --policy-seed S``
                                     the frozen v4.5.0 episode loop with a 36-way
                                     policy (``v830_bombpol.run_episode36``,
                                     imported): 4 dev windows, 300 episodes each,
                                     ``stochastic`` = softmax(z/1) sample.
``--mode diagnostics``               offline per-budget / per-(seed,window) counts,
                                     means, medians, the **bomb-usage table**
                                     (0/1/2/3 bombs per episode), the first and
                                     last bomb decision, side by side with v8.3.0,
                                     v8.6.0 and R5.  **No test, no interval.**
``--mode audio-audit``               AGENTS.md 搂9.3 on this round's own data dirs.
``--mode process-audit``             this round's own engine processes must be 0.
``--mode manifest --dir D``          sha256 + bytes of every file in D.
``--mode publish``                   copy the raw artifacts into
                                     ``Document/Update/Alpha/v8.7.0/evidence/
                                     BOMBLONG/`` and write the manifest.

Reused, never re-implemented (imported by path, sha256 pinned; a moved file is
exit 2 and nothing runs)
----------------------------------------------------------------------------
* ``tools/rl/v450_cf_label.py`` 鈥?``start_parent`` (HELLO + the 2,165-checkpoint
  prefix + PARK + DEBUG), ``prepare_env`` (dummy audio, no display),
  ``prepare_data_dir`` (a freshly generated muted ``th08.cfg`` per engine start),
  ``masks_for``, ``compact_observation``, ``Digest``, ``mode_audit`` (the audio
  audit), ``check_engine``;
* ``tools/rl/v432_fork_arm.py`` 鈥?``ForkParent`` (RESET + prefix walk, PARK,
  SPAWN/CHILD/QUIT), ``fields``, ``_step_fields``, and ``--mode fidelity`` (the
  D3 gate, run through v432 itself);
* ``tools/rl/v431_arm_a.py`` 鈥?``build_features`` (the frozen 106-dim
  observation) and ``Policy`` (the frozen 2-layer model);
* ``tools/rl/v431_dense_driver.py`` 鈥?the recorded prefix masks/hashes;
* ``tools/rl/v830_bombpol.py`` 鈥?``label_state36`` / ``label_child_action`` (the
  36-way labeller, called here with 2 actions and H = 600), ``action_masks`` /
  ``split_action`` / ``action_of``, ``policy36`` (Policy(106,64,36)),
  ``run_episode36`` (the frozen evaluation loop), ``sample_action36``,
  ``load_policy36``, ``build_arrays`` (the 106-dim / 36-label matrices with the
  ``features_sha256`` re-check);
* ``tools/rl/v800_bomb.py`` 鈥?``nearest_bullet_distances`` (only inside the
  diagnostics' reference block; not needed for a label);
* ``touhou_rl/v420/counterfactual._terminal_of`` 鈥?the death rule of R24/C4;
* ``touhou_rl/v420/contracts.cand_to_mask`` and ``assert_seed_allowed``.

Failure policy (fail closed where the data would be wrong, record where the
engine is merely counter-intuitive)
-------------------------------------------------------------------
* a pinned import whose sha256 moved -> exit 2, nothing runs;
* a replay decision whose replayed observation or ``features_sha256_k12`` does
  not equal the v8.6.0 row's -> the cell is aborted (rc 4) and the partial file
  stays ``*.partial``;
* a row set whose decisions are not a contiguous ``0..k`` inside one episode, or
  whose frame is not the recorded one -> the cell is aborted (rc 4);
* a long label whose two arms did not both complete (``alive_frames`` missing,
  ``safe`` missing, ``baseline_frame_ok`` / ``spawn_ok`` / ``quit_ok`` short of
  2/2) -> the cell is aborted (rc 4);
* the bomb = 1 invariant is **two-branched, never "must spend one"** (v8.6.0's
  measured engine semantics: bombs do not stack, so a press inside the previous
  bomb's invulnerability window is a no-op and spends nothing).  Each arm is
  recorded with ``bombs_before`` / ``bombs_after`` / ``bomb_decrement_frame``;
  the branch kind is reported, **never** aborting the cell;
* P0 (the H = 600 label has no discrimination) failing -> rc 4 from
  ``--mode stats`` and the round stops there (prereg 搂2.1 ``S-horizon``);
* every evaluation record carries ``invalid`` / ``error``; an invalid episode
  aborts the cell instead of being dropped.

Not in this file
----------------
* No statistics: counts, rates, means, medians, quantiles and distributions
  only.  P1-P3 belong to the independent analysis pass (prereg 搂2).
* No parameter search, no auxiliary rule loss, no dev reading during collection
  or training.
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

WORK = REPO / ".tmp_v870_bomblong"
PUBLISH = (REPO / "Document" / "Update" / "Alpha" / "v8.7.0" / "evidence"
           / "BOMBLONG")
PREREG = REPO / "Document" / "Update" / "Alpha" / "v8.7.0" / (
    "PREREGISTRATION_BOMBLONG_001.md")
SHORT_ROOT = REPO / ".tmp_v860_bombdag" / "labels"
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
SCHEMA_PREFLIGHT = "v870-bomblong-preflight/1"
SCHEMA_ASSERT = "v870-bomblong-ckpt-assertions/1"
SCHEMA_FIDELITY = "v870-bomblong-fidelity/1"
SCHEMA_SELECT = "v870-bomblong-select/1"
SCHEMA_STATE = "v870-bomblong-state/1"
SCHEMA_LABEL = "v870-bomblong-label/1"
SCHEMA_COLLECT = "v870-bomblong-collect/1"
SCHEMA_STATS = "v870-bomblong-labels-long-stats/1"
SCHEMA_TRAIN = "v870-bomblong-train/1"
SCHEMA_TRAIN_SUMMARY = "v870-bomblong-train-summary/1"
SCHEMA_EVAL = "v870-bomblong-eval/1"
SCHEMA_EVAL_SUMMARY = "v870-bomblong-eval-summary/1"
SCHEMA_DIAG = "v870-bomblong-diagnostics/1"
SCHEMA_BOMB_USAGE = "v870-bomblong-bomb-usage/1"
SCHEMA_AUDIT = "v870-bomblong-audio-audit/1"
SCHEMA_PROC = "v870-bomblong-process-audit/1"
SCHEMA_MANIFEST = "v870-bomblong-manifest/1"
SCHEMA_PUBLISH = "v870-bomblong-publish/1"
PRODUCER = ("tools/rl/v870_bomblong.py (live engine, HarnessClient role=shadow, "
            "H=600 two-arm bomb label on the v8.6.0 short-horizon best candidate)")
PREREG_REL = "Document/Update/Alpha/v8.7.0/PREREGISTRATION_BOMBLONG_001.md"

ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"

# -- frozen geometry (unchanged from v4.5.0/v6.0.0/v8.3.0/v8.6.0) -------------
PREFIX_FRAMES = 8663
CHECKPOINTS_IN_PREFIX = 2165
R24_SEED = 12

# -- frozen split (prereg 搂1: the v4.5.0 / v6.0.0 / v8.3.0 / v8.6.0 protocol) --
TRAIN_WINDOWS: tuple[int, ...] = (13, 14, 15, 16)
DEV_WINDOWS: tuple[int, ...] = (11, 12, 17, 18)
TRAIN_SEEDS: tuple[int, ...] = (0, 1, 2)

# -- frozen design (prereg 搂1) ----------------------------------------------
N_CAND = 18
N_ACTIONS = 36
BOMB_VALUES: tuple[int, ...] = (0, 1)
DECISION_FRAMES = 2
SHORT_H = 120                    # the v8.6.0 candidate label (reused, unchanged)
LONG_DECISIONS = 300             # prereg 搂1: --decisions 300, 2 frames/decision
LONG_H = LONG_DECISIONS * DECISION_FRAMES      # = 600 frames = 10 s
BUDGETS: tuple[int, ...] = (300, 600, 900)
EPISODES = 300
HIDDEN = 64
FEATURE_DIM = 106
LR = 1e-3
BATCH = 256
EPOCHS = 200
TEMPERATURE = 1.0
GEN_CONST = 7
DEFAULT_TIMEOUT = 300.0
P0_THRESHOLD = 0.90              # prereg 搂2 P0: bomb1-longer fraction must be < 90 %
BALANCE_THRESHOLD = 0.90         # >90/10 -> the bomb head switches to MSE
BALANCE_FLOOR = 0.10
BOMB_HEAD_WEIGHT = 1.0           # loss = BCE36(short) + 1.0 * bomb_head
CND_RULE = ("score(cand) = alive_frames[cand*2] at H = 120 (the bomb = 0 arm of the "
            "v8.6.0 36-way label); argmax, ties -> smallest candidate index")

# -- the behaviour policy of the v8.6.0 states: v4.5.0 `S` (frozen, 3 seeds) ---
FROZEN_ROUND2_SHA256: dict[int, str] = {
    0: "4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc",
    1: "a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4",
    2: "c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01",
}
PROTECTED_FILES = (
    "tools/rl/v432_fork_arm.py",
    "tools/rl/v450_cf_label.py",
    "tools/rl/v431_arm_a.py",
    "tools/rl/v431_dense_driver.py",
    "tools/rl/v800_bomb.py",
    "tools/rl/v830_bombpol.py",
    "tools/rl/v850_distill.py",
    "tools/rl/v860_bombdag.py",
    "touhou_rl/v420/contracts.py",
    "touhou_rl/v400/gym/actions.py",
)

# -- the three references this round is placed beside (numbers only, no test) --
BASELINE_V830 = {
    "arm": ("v8.3.0 BOMBPOL-001 鈥?behaviour policy v4.5.0 `S` (never bombs), the "
            "same 36-way labels on 5,166 states whose bombs_remaining is always 3"),
    "episodes_per_budget": 3600,
    "survived": {300: 3598, 600: 700, 900: 22},
    "rate": {300: 0.999444, 600: 0.194444, 900: 0.006111},
    "bombs_used_per_episode": {"3": 3600},
    "first_bomb_decision_median": 0.0,
    "source": ("Document/Update/Alpha/v8.3.0/evidence/BOMBPOL/diagnostics.json "
               "(budgets 300/600/900)"),
}
BASELINE_V860 = {
    "arm": ("v8.6.0 BOMBDAGGER-001 鈥?36 actions, bombs in {3,2,1}, labels still "
            "H = 120; the round was stopped after collection and never trained or "
            "evaluated"),
    "episodes_per_budget": 0,
    "rate": {"300": None, "600": None, "900": None},
    "source": ("measured on this host: .tmp_v860_bombdag/{train,ckpt,eval}/ are "
               "empty and Document/Update/Alpha/v8.6.0/evidence/ does not exist, "
               "so there is **no v8.6.0 evaluation number** to place beside this "
               "round.  Recorded as not-available, never estimated."),
}
RULE_R5 = {
    "arm": ("v8.4.0 R5 `staged` 鈥?script rule: bombs 3 -> 80 px, 2 -> 60 px, "
            "1 -> 40 px, `nearest_all_px < threshold`, bombs > 0"),
    "episodes_per_budget": 3600,
    "rate": {300: 1.0, 600: 0.771111, 900: 0.176111},
    "source": ("quoted from the work order / Document/Update/Alpha/v8.4.0/"
               "evidence/BOMBRULE2/RUN.md: 100 % / 77.11 % / 17.61 %.  The v8.4.0 "
               "tree measured 600 decisions as 77.1111 % (2,776/3,600); the 300 and "
               "900 numbers are quoted, not recomputed here."),
}

_LOG_FH: Any = None
_MODS: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class BombLongError(RuntimeError):
    """The run cannot continue as specified (fail closed)."""


def log(message: str) -> None:
    line = f"[v870] {message}"
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
        raise BombLongError(f"cannot import {path}")
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
    """``tools/rl/v450_cf_label.py`` 鈥?imported, pinned, never modified."""

    module = _cached("v450_cf_label", REPO / "tools" / "rl" / "v450_cf_label.py")
    module.m432()._LOG_FH = _LOG_FH
    return module


def m432():
    return _cached("v432_fork_arm", REPO / "tools" / "rl" / "v432_fork_arm.py")


def arm_a():
    return _cached("v431_arm_a", REPO / "tools" / "rl" / "v431_arm_a.py")


def v830():
    """``tools/rl/v830_bombpol.py`` 鈥?the proven 36-way labeller and eval loop."""

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


def count_lines(path: pathlib.Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


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


def counter(values: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
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


def features_sha256(feats: Any) -> str:
    import numpy as np

    array = np.asarray(feats, dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def generator_seed(seed: int) -> int:
    """v4.5.0 搂2.3's rule, verbatim: ``torch.Generator().manual_seed(seed*1000+7)``."""

    return int(seed) * 1000 + GEN_CONST


def check_pins() -> dict[str, Any]:
    out: dict[str, Any] = {"pinned": {}, "all_match": True,
                           "prereg": {}, "self": {}, "short_root": {}}
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
    out["short_root"] = {"path": str(SHORT_ROOT),
                         "exists": SHORT_ROOT.is_dir()}
    return out


# --------------------------------------------------------------------------- #
# the minimal labeller primitive: hold one 36-way action for ``frames`` frames
# --------------------------------------------------------------------------- #
def hold_action(parent: Any, handle: int, action: int, frames: int, *,
                digest: Any = None,
                state_frame: int | None = None,
                parent_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Hold one action for up to ``frames`` frames on a spawned child.

    This is the whole labeller this round needs, written out instead of borrowed
    from ``v830_bombpol.label_state36`` (whose internal ``alive_frames`` has
    several assignment branches).  It is line-for-line the same protocol as
    ``v830_bombpol.label_child_action`` (the v4.5.0 labeller), which is what makes
    the resulting numbers comparable 鈥?and it is **checked against that function
    in ``--mode smoke``** rather than trusted:

        1. read the child's baseline ``STATE``  -> deaths / stage / bombs / frame
        2. for index in 1..frames:
               STEP first_mask if index == 1 else rest_mask
               STATE
               record bombs first decrease (else ``bomb_decrement_frame`` = None)
               counterfactual._terminal_of(...)  -> break on the first terminal
        3. alive_frames = frames_executed - 1 at an engine death, ``frames`` when
           the horizon was walked (R24's rule); safe = 0 iff it died

    Every returned field is a non-None int/bool except ``terminal`` and
    ``bomb_decrement_frame``; **``bomb_decrement_frame = None`` is legal** (a
    bomb = 1 press inside the previous bomb's invulnerability window is a no-op
    and spends nothing 鈥?v8.6.0's measured engine semantics), so nothing here
    ``int()``s it.
    """

    from touhou_rl.v400.labels.state import state_from_reply
    from touhou_rl.v420 import counterfactual as cf

    m = m432()
    cand, bomb = v830().split_action(int(action))
    first_mask, rest_mask = v830().action_masks(int(action))
    out: dict[str, Any] = {
        "action": int(action), "cand": int(cand), "bomb": int(bomb),
        "mask_first_frame": int(first_mask),
        "mask_remaining_frames": int(rest_mask),
        "handle": int(handle), "baseline_frame": None,
        "baseline_frame_ok": None, "baseline_state_equal_parent": None,
        "deaths_start": None, "bombs_start": None, "bombs_end": None,
        "bombs_used": None, "bomb_decrement_frame": None,
        "first_death_frame": None, "frames_executed": 0,
        "alive_frames": None, "safe": None, "died": False,
        "terminal": None, "terminal_frame": None}
    reading, reply = parent.child_state(handle)
    if digest is not None:
        digest.add(reply)
    deaths0 = float(reading["deaths"] or 0.0)
    stage0 = int(reading["stage"] or 0)
    bombs0 = int(reading["bombs_remaining"] or 0)
    out["deaths_start"] = deaths0
    out["bombs_start"] = bombs0
    out["bombs_end"] = bombs0
    out["baseline_frame"] = int(reading["frame"])
    if state_frame is not None:
        out["baseline_frame_ok"] = bool(out["baseline_frame"] == int(state_frame))
        if not out["baseline_frame_ok"]:
            raise BombLongError(
                f"action {action}: the child starts at frame "
                f"{out['baseline_frame']}, the parked parent is at {state_frame}")
    if parent_state is not None:
        out["baseline_state_equal_parent"] = bool(dict(reading) == dict(parent_state))
    died = False
    bombs_end = bombs0
    for index in range(1, int(frames) + 1):
        mask = first_mask if index == 1 else rest_mask
        reply = parent.child(handle, f"STEP {int(mask)} 1")
        if digest is not None:
            digest.add(reply)
        step = m._step_fields(reply)
        if step["advanced"] == 1:
            out["frames_executed"] = index
        elif not (step["exited"] or step["game_over"]):
            raise BombLongError(f"action {action}: CHILD STEP advanced "
                                f"{step['advanced']} frames: {reply!r}")
        state_reply = parent.child(handle, "STATE")
        if digest is not None:
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
            out["terminal"] = None if term.get("terminal") is None \
                else str(term.get("terminal"))
            out["terminal_frame"] = index
            if term.get("engine_dead"):
                died = True
                out["died"] = True
                out["first_death_frame"] = int(out["frames_executed"])
                out["alive_frames"] = int(out["frames_executed"]) - 1
            else:
                # stage_clear: no death inside the horizon, but the segment ended.
                # Recorded explicitly; never silently called a death.
                out["alive_frames"] = int(frames)
            break
    if out["alive_frames"] is None:
        out["alive_frames"] = int(frames)
    out["safe"] = 0 if died else 1
    out["bombs_end"] = int(bombs_end)
    out["bombs_used"] = int(bombs0) - int(bombs_end)
    return out


def validate_arm(arm: dict[str, Any], *, frames: int, where: str) -> None:
    """Fail fast on an arm whose numbers cannot be a 600-frame measurement."""

    alive = arm.get("alive_frames")
    if not isinstance(alive, int) or isinstance(alive, bool):
        raise BombLongError(f"{where}: alive_frames is {alive!r}, expected an int "
                            f"in [0, {frames}]")
    if not 0 <= alive <= int(frames):
        raise BombLongError(f"{where}: alive_frames = {alive} outside [0, {frames}]")
    for field in ("bombs_start", "bombs_end", "bombs_used", "frames_executed"):
        value = arm.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise BombLongError(f"{where}: {field} is {value!r}, expected an int")
    if arm.get("safe") not in (0, 1):
        raise BombLongError(f"{where}: safe is {arm.get('safe')!r}")
    if int(arm["frames_executed"]) < 1:
        raise BombLongError(f"{where}: frames_executed = "
                            f"{arm['frames_executed']} (the child never advanced)")
    if int(arm["bombs_used"]) not in (0, 1):
        raise BombLongError(f"{where}: bombs_used = {arm['bombs_used']} outside "
                            f"{{0, 1}}")
    if bool(arm["safe"] == 1) and int(arm["alive_frames"]) != int(frames):
        raise BombLongError(f"{where}: safe = 1 but alive_frames = "
                            f"{arm['alive_frames']} != {frames}")
    if arm.get("baseline_frame_ok") is False:
        raise BombLongError(f"{where}: the child did not start at the parked frame")
    if arm.get("bomb_decrement_frame") is not None:
        decrement = arm["bomb_decrement_frame"]
        if not isinstance(decrement, int) or not 1 <= int(decrement) <= int(frames):
            raise BombLongError(f"{where}: bomb_decrement_frame = {decrement!r} "
                                f"outside [1, {frames}]")
        if int(arm["bombs_end"]) != int(arm["bombs_start"]) - 1:
            raise BombLongError(
                f"{where}: a decrement was seen at frame {decrement} but "
                f"bombs stayed {arm['bombs_start']} -> {arm['bombs_end']}")
    elif int(arm["bombs_end"]) != int(arm["bombs_start"]):
        raise BombLongError(f"{where}: no decrement frame but bombs "
                            f"{arm['bombs_start']} -> {arm['bombs_end']}")


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def states_path(root: pathlib.Path) -> pathlib.Path:
    return root / "states.jsonl"


def select_report_path(root: pathlib.Path) -> pathlib.Path:
    return root / "select.json"


def cell_labels_path(root: pathlib.Path, seed: int, window: int) -> pathlib.Path:
    return root / "labels" / f"seed{int(seed)}_w{int(window)}.jsonl"


def cell_summary_path(root: pathlib.Path, seed: int, window: int) -> pathlib.Path:
    return root / "collect" / f"seed{int(seed)}_w{int(window)}.json"


def pooled_labels_path(root: pathlib.Path) -> pathlib.Path:
    return root / "labels_long.jsonl"


def pooled_stats_path(root: pathlib.Path) -> pathlib.Path:
    return root / "labels_long_stats.json"


def policy_ckpt_path(root: pathlib.Path, seed: int) -> pathlib.Path:
    return root / "ckpt" / f"seed{int(seed)}" / "bomblong.pt"


def train_report_path(root: pathlib.Path, seed: int) -> pathlib.Path:
    return root / "train" / f"seed{int(seed)}.json"


def eval_path(root: pathlib.Path, budget: int, seed: int) -> pathlib.Path:
    return root / "eval" / f"eval_d{int(budget)}_seed{int(seed)}.jsonl"


def behaviour_ckpt(seed: int) -> pathlib.Path:
    return CKPT_ROOT / f"seed{int(seed)}" / "round2.pt"


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
                   "single_factor": ("the horizon of the bomb decision's "
                                     "counterfactual label: H = 120 (2 s) -> "
                                     "H = 600 (10 s)")},
        "pins": pins, "engine": engine, "launch_env": prepare_env(),
        "feature_dim": int(arm_a().feature_dim()),
        "action_space": {"candidates": N_CAND, "bombs": list(BOMB_VALUES),
                         "actions": N_ACTIONS,
                         "encoding": "action = cand * 2 + bomb",
                         "bomb1_semantics": ("cand_to_mask(cand, 1) on the label's "
                                             "first frame, cand_to_mask(cand, 0) "
                                             "afterwards (v8.3.0 搂1, unchanged)")},
        "candidate_rule": CND_RULE,
        "frozen": {"TRAIN_WINDOWS": list(TRAIN_WINDOWS),
                   "DEV_WINDOWS": list(DEV_WINDOWS),
                   "train_seeds": list(TRAIN_SEEDS),
                   "SHORT_H": SHORT_H, "LONG_H": LONG_H,
                   "LONG_DECISIONS": LONG_DECISIONS,
                   "budgets_decisions": list(BUDGETS),
                   "decision_frames": DECISION_FRAMES,
                   "episodes_per_window": EPISODES,
                   "lr": LR, "batch": BATCH, "epochs": EPOCHS, "hidden": HIDDEN,
                   "p0_threshold": P0_THRESHOLD,
                   "balance_threshold": BALANCE_THRESHOLD,
                   "bomb_head_weight": BOMB_HEAD_WEIGHT},
        "short_label_source": {"root": str(SHORT_ROOT),
                               "schema": "v860-bombdag-label/1 (read-only)"},
        "references": {"v830_bombpol": BASELINE_V830, "v860_bombdagger": BASELINE_V860,
                       "rule_r5": RULE_R5},
    }
    ok = bool(pins["all_match"]) and bool(engine.get("match"))
    ok = ok and bool(pins["short_root"]["exists"])
    rec["all_pass"] = ok
    out = pathlib.Path(args.out) if args.out else root / "preflight.json"
    write_json(out, rec)
    log(f"preflight: pins={pins['all_match']} engine={engine.get('match')} "
        f"short_root={pins['short_root']['exists']} "
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
        "requirement": ("checklist item 2: the behaviour policy of every v8.6.0 "
                        "state is the frozen v4.5.0 `S` student, whose final "
                        "checkpoints are round2.pt = 4abee069鈥?/ a93746bb鈥?/ "
                        "c69960fc鈥? the protected sources must not have moved"),
        "engine": None, "ckpts": {}, "protected": {}, "v870_bomblong_py": {},
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
        payload["v870_bomblong_py"] = {"path": str(me), "sha256": sha256_file(me)}
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
def mode_fidelity(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out = pathlib.Path(args.out) if args.out else root / "fidelity_d3.json"
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "v870-fidelity"
    argv = [sys.executable, str(REPO / "tools" / "rl" / "v432_fork_arm.py"),
            "--mode", "fidelity", "--seeds", str(R24_SEED),
            "--learner", "frozen", "--work-dir", str(work),
            "--out", str(out), "--tag", tag]
    started = time.perf_counter()
    proc = subprocess.run(argv, capture_output=True, text=True,
                          env=_engine_env())
    wall = round(time.perf_counter() - started, 3)
    payload: dict[str, Any] = {
        "schema": SCHEMA_FIDELITY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requirement": ("checklist item 2: the D3 fidelity gate (v4.3.2 搂2.1-搂2.5, "
                        "five assertions) must pass before any label of this round "
                        "exists; prereg 搂2.1 S-fork voids the round otherwise"),
        "delegated_to": "tools/rl/v432_fork_arm.py --mode fidelity (imported as a "
                        "subprocess: the gate is v432's, not re-implemented here)",
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


def _engine_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["SDL_AUDIODRIVER"] = "dummy"
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"]
                                     if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


# --------------------------------------------------------------------------- #
# mode: select (offline: the short-horizon best candidate of every state)
# --------------------------------------------------------------------------- #
def short_source_for(seed: int, window: int) -> pathlib.Path | None:
    """The v8.6.0 label file of one cell, preferring whichever copy has more rows.

    The v8.6.0 collect renames ``*.jsonl.partial`` to ``*.jsonl`` when a cell
    completes, so a cell can legitimately have both; the one with more rows is
    the newer, and a row count is a cheap, exact tie-break.
    """

    best: tuple[pathlib.Path, int] | None = None
    for path in (SHORT_ROOT / f"seed{int(seed)}_w{int(window)}.jsonl",
                 SHORT_ROOT / f"seed{int(seed)}_w{int(window)}.jsonl.partial"):
        if not path.is_file():
            continue
        try:
            rows = count_lines(path)
        except OSError:
            continue
        if best is None or rows > best[1]:
            best = (path, rows)
    return None if best is None else best[0]


def choose_candidate(alive_frames: Sequence[int]) -> tuple[int, int, int]:
    """The short-horizon best candidate.

    ``score(cand) = alive_frames[cand * 2]`` 鈥?the candidate's **bomb = 0** arm of
    the v8.6.0 36-way label.  The bomb bit is what this round re-labels, so the
    candidate must be chosen in the world it is compared in (no bomb); scoring it
    with its bomb = 1 arm would let a bomb-shaped advantage pick the direction.
    Ties go to the smallest candidate index (prereg 搂1's rule).
    """

    scores = [int(alive_frames[cand * 2]) for cand in range(N_CAND)]
    best = max(range(N_CAND), key=lambda cand: (scores[cand], -cand))
    scores_bomb1 = [int(alive_frames[cand * 2 + 1]) for cand in range(N_CAND)]
    best_bomb1 = max(range(N_CAND), key=lambda cand: (scores_bomb1[cand], -cand))
    return int(best), int(scores[best]), int(best_bomb1)


def mode_select(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    pins = check_pins()
    if not pins["all_match"]:
        raise BombLongError(f"pinned import moved: {json.dumps(pins)[:800]}")
    if not SHORT_ROOT.is_dir():
        raise BombLongError(f"the v8.6.0 label root is missing: {SHORT_ROOT}")
    out_states = states_path(root)
    out_report = pathlib.Path(args.out) if args.out else select_report_path(root)
    out_states.parent.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    problems: list[str] = []
    total = 0
    bombs: dict[str, int] = {}
    cand_first: dict[str, int] = {}
    cand_alt: int = 0
    wrote = 0
    handle = out_states.open("w", encoding="utf-8")
    try:
        for seed in TRAIN_SEEDS:
            for window in TRAIN_WINDOWS:
                path = short_source_for(seed, window)
                if path is None:
                    problems.append(f"missing short labels for s{seed}_w{window}")
                    continue
                rows = read_jsonl(path)
                if not rows:
                    problems.append(f"empty short labels: {path}")
                    continue
                sources.append({"path": str(path), "sha256": sha256_file(path),
                                "bytes": path.stat().st_size, "rows": len(rows),
                                "seed": seed, "window": window})
                seen: set[tuple[int, int]] = set()
                episodes: dict[int, list[int]] = {}
                per_cell = {"seed": seed, "window": int(window),
                            "source": str(path), "rows": len(rows),
                            "bombs": {}, "cand_selected": {}, "cand_rule": CND_RULE}
                for index, row in enumerate(rows):
                    if int(row.get("H", -1)) != SHORT_H:
                        raise BombLongError(
                            f"{path} row {index}: H={row.get('H')} != {SHORT_H}")
                    if len(row.get("alive_frames", [])) != N_ACTIONS:
                        raise BombLongError(
                            f"{path} row {index}: {len(row.get('alive_frames', []))} "
                            f"alive_frames != {N_ACTIONS}")
                    if len(row.get("safe", [])) != N_ACTIONS:
                        raise BombLongError(
                            f"{path} row {index}: {len(row.get('safe', []))} safe != "
                            f"{N_ACTIONS}")
                    episode = int(row["episode"])
                    decision = int(row["decision"])
                    key = (episode, decision)
                    if key in seen:
                        problems.append(f"{path.name}: duplicate {key} (skipped)")
                        continue
                    seen.add(key)
                    episodes.setdefault(episode, []).append(decision)
                    cand, score, cand_b1 = choose_candidate(row["alive_frames"])
                    cand_alt += int(cand_b1 != cand)
                    bomb = int(row["bombs_remaining"])
                    bombs[str(bomb)] = bombs.get(str(bomb), 0) + 1
                    per_cell["bombs"][str(bomb)] = (
                        per_cell["bombs"].get(str(bomb), 0) + 1)
                    per_cell["cand_selected"][str(cand)] = (
                        per_cell["cand_selected"].get(str(cand), 0) + 1)
                    cand_first[str(cand)] = cand_first.get(str(cand), 0) + 1
                    record = dict(row)
                    record.update({
                        "schema": SCHEMA_STATE,
                        "source": {"path": str(path), "sha256": sources[-1]["sha256"],
                                   "row": int(index)},
                        "bombs_remaining": bomb,
                        "cand": int(cand),
                        "cand_short_alive": int(score),
                        "cand_short_alive_bomb1": int(
                            row["alive_frames"][cand * 2 + 1]),
                        "cand_argmax_bomb1": int(cand_b1),
                        "cand_rule": CND_RULE,
                        "short_alive_bomb0": [int(row["alive_frames"][c * 2])
                                              for c in range(N_CAND)],
                        "short_alive_bomb1": [int(row["alive_frames"][c * 2 + 1])
                                              for c in range(N_CAND)],
                    })
                    write_jsonl_line(handle, record)
                    wrote += 1
                    total += 1
                contiguous: dict[str, Any] = {}
                ok_contiguous = True
                for episode, decisions in sorted(episodes.items()):
                    decisions.sort()
                    expected = list(range(len(decisions)))
                    good = decisions == expected
                    ok_contiguous = ok_contiguous and good
                    contiguous[str(episode)] = {"rows": len(decisions),
                                                "decisions": len(decisions),
                                                "contiguous_from_0": good}
                per_cell["episodes"] = contiguous
                per_cell["decisions_contiguous"] = ok_contiguous
                if not ok_contiguous:
                    problems.append(f"s{seed}_w{window}: decisions not contiguous "
                                    f"from 0")
                cells[f"s{seed}_w{window}"] = per_cell
                log(f"select s{seed} w{window}: {len(rows)} rows, bombs "
                    f"{per_cell['bombs']}, episodes {len(episodes)}")
    finally:
        handle.close()
    report: dict[str, Any] = {
        "schema": SCHEMA_SELECT, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "purpose": ("choose each v8.6.0 state's short-horizon (H = 120) best "
                    "candidate; the long-horizon collection then only re-labels "
                    "that candidate's bomb = 0 / bomb = 1 arms"),
        "candidate_rule": CND_RULE,
        "states_file": {"path": str(out_states), "sha256": sha256_file(out_states),
                        "bytes": out_states.stat().st_size, "rows": wrote},
        "sources": sources, "cells": cells, "states": wrote,
        "bombs_remaining": bombs,
        "bombs_remaining_fractions": ({key: round(value / wrote, 6)
                                       for key, value in bombs.items()}
                                      if wrote else {}),
        "cand_selected": cand_first,
        "cand_argmax_bomb1_differs": cand_alt,
        "cand_argmax_bomb1_differs_fraction": (round(cand_alt / wrote, 6)
                                               if wrote else None),
        "problems": problems, "pins": pins,
        "all_pass": bool(wrote > 0 and not problems),
        "wall_s": None,
    }
    write_json(out_report, report)
    log(f"select: {wrote} states, bombs={bombs}, "
        f"argmax_bomb1_differs={cand_alt}, problems={len(problems)} -> {out_states}")
    return 0 if report["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: collect (one (seed, window) cell: the verified replay + 2 arms x H=600)
# --------------------------------------------------------------------------- #
def load_states(root: pathlib.Path) -> list[dict[str, Any]]:
    path = states_path(root)
    if not path.is_file():
        raise BombLongError(f"no state snapshot: run --mode select first ({path})")
    return read_jsonl(path)


def mode_collect(args: argparse.Namespace) -> int:
    from touhou_rl.v400.gym import protocol as P
    from touhou_rl.v420 import counterfactual as cf

    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    windows = parse_windows(args.windows)
    horizon = int(args.horizon)
    decisions_cap = int(args.decisions)
    limit = int(getattr(args, "limit_states", 0) or 0)
    if seed not in TRAIN_SEEDS:
        raise BombLongError(f"--seed must be one of {TRAIN_SEEDS}, got {seed}")
    if horizon != LONG_H:
        raise BombLongError(f"prereg 搂1 fixes H = {LONG_H} for the long label "
                            f"({LONG_DECISIONS} decisions x {DECISION_FRAMES} "
                            f"frames); got {horizon}")
    if decisions_cap != LONG_DECISIONS:
        raise BombLongError(f"prereg 搂1 fixes {LONG_DECISIONS} decisions; got "
                            f"{decisions_cap}")
    for window in windows:
        if window not in TRAIN_WINDOWS:
            raise BombLongError(f"window {window} is not a train window "
                                f"{TRAIN_WINDOWS}")
    pins = check_pins()
    if not pins["all_match"]:
        raise BombLongError(f"pinned import moved: {json.dumps(pins)[:800]}")
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    states = load_states(root)
    by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in states:
        by_cell.setdefault((int(row["seed"]), int(row["window"])), []).append(row)

    rc_all = 0
    for window in windows:
        tag = args.tag or f"v870-long-s{seed}-w{window}"
        run_dir = root / "runs" / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        data_root = root / "data" / tag
        labels = (pathlib.Path(args.labels_out)
                  if getattr(args, "labels_out", "") and len(windows) == 1
                  else cell_labels_path(root, seed, window))
        labels.parent.mkdir(parents=True, exist_ok=True)
        partial = labels.with_suffix(labels.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        summary_path = (pathlib.Path(args.summary_out)
                        if getattr(args, "summary_out", "") and len(windows) == 1
                        else cell_summary_path(root, seed, window))
        rows = sorted(by_cell.get((seed, int(window)), []),
                      key=lambda r: (int(r["episode"]), int(r["decision"])))
        if limit > 0:
            rows = rows[:limit]
        rec: dict[str, Any] = {
            "schema": SCHEMA_COLLECT, "producer": PRODUCER,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "prereg": PREREG_REL, "seed": seed, "window": int(window),
            "decisions_cap": decisions_cap, "H": horizon,
            "decision_frames": DECISION_FRAMES,
            "candidate_rule": CND_RULE,
            "arms_per_state": 2,
            "states_planned": len(rows), "limit_states": int(limit),
            "bomb1_semantics": ("cand_to_mask(cand, 1) on the arm's first frame, "
                                "cand_to_mask(cand, 0) on frames 2..H (v8.3.0 搂1)"),
            "replay": {"note": ("the v6.0.0/v8.6.0 roll-out is replayed action by "
                                "action from the recorded behaviour action of every "
                                "state; the replayed observation and the 106-dim "
                                "features sha256 are compared with the v8.6.0 row at "
                                "every decision"),
                       "actions_from": "row['behaviour']['action'] of the v8.6.0 "
                                       "label row (not re-sampled)",
                       "checks": {"obs_equal": 0, "features_sha256_equal": 0,
                                  "frame_equal": 0, "decisions": 0}},
            "pins": pins, "engine": engine, "launch_env": launch_env,
            "labels_path": str(labels), "status": "failed", "error": None,
            "episodes": [], "states": 0,
            "arm_checks": {"baseline_frame_ok": 0, "baseline_state_equal_parent": 0,
                           "spawn_ok": 0, "quit_ok": 0, "anomalies": 0,
                           "arm_kinds": {}, "bomb1_press_honoured": 0,
                           "bomb1_press_ignored": 0, "bomb0_spent_a_bomb": 0,
                           "decrement_frames": {"bomb0": {}, "bomb1": {}},
                           "survive_600_bomb0": 0, "survive_600_bomb1": 0},
            "gate": {}, "cfg": {}, "wall_s": None, "state_costs_s": [],
        }
        out_fh = partial.open("w", encoding="utf-8")
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
            masks, hashes, _meta = v450().masks_for(window)
            state, _reply = parent.state()
            if int(state["frame"]) != PREFIX_FRAMES:
                raise BombLongError(f"w{window}: parked parent at frame "
                                    f"{state['frame']}, expected {PREFIX_FRAMES}")
            episodes: dict[int, list[dict[str, Any]]] = {}
            for row in rows:
                episodes.setdefault(int(row["episode"]), []).append(row)
            for episode in sorted(episodes):
                ep: dict[str, Any] = {
                    "episode": int(episode), "rows": len(episodes[episode]),
                    "states": 0, "prefix": None, "terminal": None, "died": False,
                    "last_decision": None}
                if episode > 0:
                    prefix = parent.walk_prefix(masks, hashes, window)
                    parent.park()
                    state, _reply = parent.state()
                    if int(state["frame"]) != PREFIX_FRAMES:
                        raise BombLongError(
                            f"w{window} ep{episode}: warm RESET left the parent at "
                            f"frame {state['frame']}")
                    ep["prefix"] = {"hash_checks": prefix["hash_checks"],
                                    "hash_mismatches": prefix["hash_mismatches"],
                                    "final_frame": prefix["final_frame"]}
                # the death rule's baselines are captured at the episode start,
                # exactly as v8.6.0's collect does (bootstrap counterfactual._terminal_of)
                deaths0 = float(state["deaths"] or 0.0)
                stage0 = int(state["stage"] or 0)
                items = episodes[episode]
                for position, row in enumerate(items):
                    decision = int(row["decision"])
                    state_frame = int(state["frame"])
                    hazards = P.parse_hazards_reply(parent.ask("HAZARDS"))
                    obs = v450().compact_observation(state, hazards)
                    feats, _detail = arm_a().build_features(
                        state=state, hazards=hazards,
                        decision_index=int(decision))
                    digest = features_sha256(feats)
                    checks = rec["replay"]["checks"]
                    checks["decisions"] += 1
                    if obs == row["obs"]:
                        checks["obs_equal"] += 1
                    if digest == row["features_sha256_k12"]:
                        checks["features_sha256_equal"] += 1
                    if state_frame == int(row["frame"]):
                        checks["frame_equal"] += 1
                    if (obs != row["obs"] or digest != row["features_sha256_k12"]
                            or state_frame != int(row["frame"])):
                        raise BombLongError(
                            f"w{window} ep{episode} d{decision}: the replay did not "
                            f"land on the recorded state (frame {state_frame} vs "
                            f"{row['frame']}, obs_equal={obs == row['obs']}, "
                            f"features_equal="
                            f"{digest == row['features_sha256_k12']})")
                    cand = int(row["cand"])
                    park_reply = parent.park()
                    got_frame = m432().fields(park_reply).get("frame")
                    if got_frame != str(state_frame):
                        raise BombLongError(
                            f"w{window} ep{episode} d{decision}: PARK reports frame "
                            f"{got_frame}, the read state is at {state_frame}")
                    actions = [2 * cand, 2 * cand + 1]
                    reply_digest = v450().Digest()
                    arms: list[dict[str, Any]] = []
                    label_started = time.perf_counter()
                    for action in actions:
                        child, spawn_reply = parent.spawn()
                        reply_digest.add(spawn_reply)
                        arm = hold_action(parent, child, action, horizon,
                                          digest=reply_digest,
                                          state_frame=state_frame,
                                          parent_state=state)
                        quit_reply = parent.quit_child(child)
                        reply_digest.add(quit_reply)
                        arm["spawn_reply"] = spawn_reply
                        arm["quit_reply"] = quit_reply
                        validate_arm(arm, frames=horizon,
                                     where=(f"w{window} ep{episode} d{decision} "
                                            f"arm action={action} "
                                            f"(bomb={action % 2})"))
                        if not str(spawn_reply).startswith("OK spawned=1"):
                            raise BombLongError(
                                f"w{window} ep{episode} d{decision}: SPAWN "
                                f"refused: {spawn_reply!r}")
                        if str(quit_reply) != "OK bye":
                            raise BombLongError(
                                f"w{window} ep{episode} d{decision}: QUIT "
                                f"answer {quit_reply!r}")
                        arms.append(arm)
                    label_wall = round(time.perf_counter() - label_started, 4)
                    checks_arm = rec["arm_checks"]
                    checks_arm["baseline_frame_ok"] += sum(
                        1 for arm in arms if arm["baseline_frame_ok"])
                    checks_arm["baseline_state_equal_parent"] += sum(
                        1 for arm in arms if arm["baseline_state_equal_parent"])
                    checks_arm["spawn_ok"] += sum(
                        1 for arm in arms
                        if str(arm["spawn_reply"]).startswith("OK spawned=1"))
                    checks_arm["quit_ok"] += sum(
                        1 for arm in arms if str(arm["quit_reply"]) == "OK bye")
                    arm0, arm1 = arms
                    alive0 = int(arm0["alive_frames"])
                    alive1 = int(arm1["alive_frames"])
                    surv0 = int(arm0["safe"])
                    surv1 = int(arm1["safe"])
                    before0 = int(arm0["bombs_start"])
                    after0 = int(arm0["bombs_end"])
                    before1 = int(arm1["bombs_start"])
                    after1 = int(arm1["bombs_end"])
                    spent1 = bool(after1 == before1 - 1)
                    noop1 = bool(not spent1 and after1 == before1
                                 and alive1 == alive0)
                    anomaly = bool(not spent1 and not noop1)
                    kind = ("spent" if spent1 else
                            ("noop_due_to_invuln" if noop1 else
                             ("anomaly_counter_changed_not_minus_one"
                              if after1 != before1
                              else "anomaly_counter_unchanged_alive_differs")))
                    checks_arm["arm_kinds"][kind] = (
                        checks_arm["arm_kinds"].get(kind, 0) + 1)
                    checks_arm["anomalies"] += int(anomaly)
                    checks_arm["bomb1_press_honoured"] += int(spent1)
                    checks_arm["bomb1_press_ignored"] += int(not spent1)
                    # bomb_decrement_frame is None when that arm never decremented
                    # the counter (the bomb = 0 arm always; a bomb = 1 arm inside a
                    # previous bomb's invulnerability window).  None is a legal
                    # value and gets its own key, never int(None).
                    for arm, arm_name in ((arm0, "bomb0"), (arm1, "bomb1")):
                        decrement = arm["bomb_decrement_frame"]
                        key = "None" if decrement is None else str(int(decrement))
                        bucket = checks_arm["decrement_frames"][arm_name]
                        bucket[key] = bucket.get(key, 0) + 1
                    if arm0["bomb_decrement_frame"] is not None:
                        checks_arm["bomb0_spent_a_bomb"] += 1
                    checks_arm["survive_600_bomb0"] += surv0
                    checks_arm["survive_600_bomb1"] += surv1
                    if alive1 > alive0:
                        comparison = "bomb1_longer"
                    elif alive1 == alive0:
                        comparison = "equal"
                    else:
                        comparison = "bomb1_shorter"
                    out_row = {
                        "schema": SCHEMA_LABEL,
                        "seed": seed, "window": int(window),
                        "episode": int(episode), "decision": int(decision),
                        "frame": state_frame, "bombs_remaining": int(
                            row["bombs_remaining"]),
                        "H": int(horizon), "decisions": int(decisions_cap),
                        "decision_frames": DECISION_FRAMES,
                        "cand": cand, "cand_rule": CND_RULE,
                        "cand_short_alive": int(row["cand_short_alive"]),
                        "cand_short_alive_bomb1": int(
                            row["cand_short_alive_bomb1"]),
                        "actions": actions,
                        "alive_frames_600": [alive0, alive1],
                        "survive_600": [surv0, surv1],
                        "bombs_before": [before0, before1],
                        "bombs_after": [after0, after1],
                        "bombs_used": [int(arm0["bombs_used"]),
                                       int(arm1["bombs_used"])],
                        "bomb_decrement_frame": [
                            arm0["bomb_decrement_frame"],
                            arm1["bomb_decrement_frame"]],
                        "first_death_frame": [arm0["first_death_frame"],
                                              arm1["first_death_frame"]],
                        "terminals": [arm0["terminal"], arm1["terminal"]],
                        "terminal_frames": [arm0["terminal_frame"],
                                            arm1["terminal_frame"]],
                        "frames_executed": [int(arm0["frames_executed"]),
                                            int(arm1["frames_executed"])],
                        "alive_frames_600_bomb0": alive0,
                        "alive_frames_600_bomb1": alive1,
                        "survive_600_bomb0": surv0, "survive_600_bomb1": surv1,
                        "bombs_after_bomb0": after0, "bombs_after_bomb1": after1,
                        "delta_alive_600": int(alive1 - alive0),
                        "comparison": comparison,
                        "bomb1_branch_kind": kind,
                        "obs": obs,
                        "features_sha256_k12": digest,
                        "short_alive_bomb0": row["short_alive_bomb0"],
                        "short_alive_bomb1": row["short_alive_bomb1"],
                        "park_reply": park_reply,
                        "child_replies_digest": reply_digest.hexdigest(),
                        "child_reply_lines": reply_digest.lines,
                        "arms": arms,
                        "labeller": ("tools/rl/v870_bomblong.py::hold_action (this "
                                     "round's minimal primitive; checked against "
                                     "v830_bombpol.label_state36 in --mode smoke)"),
                        "label_wall_s": label_wall,
                    }
                    write_jsonl_line(out_fh, out_row)
                    out_fh.flush()
                    rec["states"] += 1
                    ep["states"] += 1
                    ep["last_decision"] = int(decision)
                    rec["state_costs_s"].append(float(label_wall))
                    if rec["states"] % 50 == 0:
                        log(f"  s{seed} w{window} ep{episode} d{decision}: "
                            f"{rec['states']}/{len(rows)} states, mean_label="
                            f"{statistics.fmean(rec['state_costs_s']):.3f}s, "
                            f"kinds={checks_arm['arm_kinds']}")
                    first_mask, rest_mask = v830().action_masks(
                        int(row["behaviour"]["action"]))
                    died = False
                    for frame_index in range(DECISION_FRAMES):
                        step = parent.step(first_mask if frame_index == 0
                                           else rest_mask)
                        reading, _reply = parent.state()
                        term = cf._terminal_of(
                            reading, deaths_start=deaths0, stage0=stage0,
                            hash_hex=(step["hash"] or ""),
                            frame=int(reading["frame"]), exited=step["exited"],
                            game_over=step["game_over"])
                        state = reading
                        if term is not None:
                            ep["terminal"] = str(term.get("terminal"))
                            ep["died"] = bool(term.get("engine_dead"))
                            died = True
                            break
                    if died and position != len(items) - 1:
                        raise BombLongError(
                            f"w{window} ep{episode}: the replay died at decision "
                            f"{decision} but the v8.6.0 cell has "
                            f"{len(items) - position - 1} more rows for this episode")
                rec["episodes"].append(ep)
                log(f"s{seed} w{window} ep{episode}: {ep['states']}/{ep['rows']} "
                    f"states, terminal={ep['terminal']}, "
                    f"died={ep['died']}")
            parent.close()
            parents.remove(parent)
            if rec["states"] >= len(rows):
                rec["status"] = "smoke" if limit > 0 else "complete"
            else:
                rec["status"] = "short"
        except Exception as exc:                                      # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["status"] = "failed"
            log(f"COLLECT FAILED s{seed} w{window}: {rec['error']}")
        finally:
            out_fh.close()
            for parent in parents:
                try:
                    parent.close()
                except Exception:                                     # noqa: BLE001
                    pass
            rec["wall_s"] = round(time.perf_counter() - started, 3)
            rec["label_cost_s"] = quantiles(rec["state_costs_s"])
            rec["states_planned"] = int(rec["states_planned"])
            del rec["state_costs_s"]
            if rec["status"] in ("complete", "short", "smoke"):
                if labels.exists():
                    labels.unlink()
                partial.rename(labels)
                rec["labels_sha256"] = sha256_file(labels)
                rec["labels_bytes"] = labels.stat().st_size
            write_json(summary_path, rec)
        log(f"collect s{seed} w{window}: status={rec['status']} "
            f"states={rec['states']}/{rec['states_planned']} "
            f"kinds={rec['arm_checks']['arm_kinds']} wall={rec['wall_s']}s -> {labels}")
        if rec["status"] == "failed":
            rc_all = 4
    return rc_all


# --------------------------------------------------------------------------- #
# mode: smoke (the single-state gate before any 12-way run)
# --------------------------------------------------------------------------- #
def mode_smoke(args: argparse.Namespace) -> int:
    """One or two states of one cell, through **both** labellers, compared.

    This is the pre-flight the 12-way run must pass:

    * the replay reaches the recorded state (obs / features / frame);
    * ``hold_action`` returns two fully-populated arms with no ``None`` in a
      numeric field and every value inside its legal range
      (``validate_arm`` is fail-fast);
    * the two arms' ``alive_frames`` / ``safe`` / ``bombs_start`` /
      ``bombs_end`` / ``bomb_decrement_frame`` are **identical** to
      ``v830_bombpol.label_state36``'s on the same parked state 鈥?i.e. the new
      primitive is the frozen labeller, bit for bit (the equivalence control).

    Nothing is written under ``labels/``: the output goes to ``smoke/``.
    """

    from touhou_rl.v400.gym import protocol as P
    from touhou_rl.v420 import counterfactual as cf

    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    windows = parse_windows(args.windows)
    if len(windows) != 1:
        raise BombLongError(f"--mode smoke needs exactly one window, got {windows}")
    window = windows[0]
    horizon = int(args.horizon)
    take = max(1, int(getattr(args, "states", 0) or 1))
    pins = check_pins()
    if not pins["all_match"]:
        raise BombLongError(f"pinned import moved: {json.dumps(pins)[:800]}")
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    rows = [row for row in load_states(root)
            if int(row["seed"]) == seed and int(row["window"]) == int(window)]
    rows.sort(key=lambda r: (int(r["episode"]), int(r["decision"])))
    rows = rows[:take]
    if not rows:
        raise BombLongError(f"no states for s{seed}_w{window}")
    tag = args.tag or f"v870-smoke-s{seed}-w{window}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    out = pathlib.Path(args.out) if args.out else root / "smoke" / (
        f"smoke_s{seed}_w{window}.json")
    payload: dict[str, Any] = {
        "schema": "v870-bomblong-smoke/1", "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "requirement": ("the gate before the 12-way collection: one state, both "
                        "labellers, identical numbers, no None in a numeric field"),
        "seed": seed, "window": int(window), "H": horizon,
        "decisions": int(args.decisions), "states_requested": take,
        "engine": engine, "launch_env": launch_env,
        "equivalence_reference": "v830_bombpol.label_state36 (imported, unmodified)",
        "states": [], "problems": [], "all_pass": False, "error": None,
        "wall_s": None,
    }
    parents: list[Any] = []
    started = time.perf_counter()
    try:
        data_dir, data_info = v450().prepare_data_dir(
            data_root, window, args.data_source)
        payload["cfg"] = data_info["cfg"]
        payload["data_dir"] = str(data_dir)
        parent, gate = start_parent(slot=0, window=window, engine=args.engine,
                                    data_dir=data_dir, log_dir=run_dir)
        parents.append(parent)
        payload["gate"] = {key: gate[key] for key in (
            "hello_role_shadow", "park_ok", "park_frame", "threads_1",
            "all_pass")}
        state, _reply = parent.state()
        if int(state["frame"]) != PREFIX_FRAMES:
            raise BombLongError(f"w{window}: parked parent at frame "
                                f"{state['frame']}, expected {PREFIX_FRAMES}")
        deaths0 = float(state["deaths"] or 0.0)
        stage0 = int(state["stage"] or 0)
        for row in rows:
            decision = int(row["decision"])
            state_frame = int(state["frame"])
            hazards = P.parse_hazards_reply(parent.ask("HAZARDS"))
            obs = v450().compact_observation(state, hazards)
            feats, _detail = arm_a().build_features(
                state=state, hazards=hazards, decision_index=decision)
            fhash = features_sha256(feats)
            item: dict[str, Any] = {
                "episode": int(row["episode"]), "decision": decision,
                "frame": state_frame, "bombs_remaining": int(
                    row["bombs_remaining"]), "cand": int(row["cand"]),
                "replay": {"frame_equal": bool(state_frame == int(row["frame"])),
                           "obs_equal": bool(obs == row["obs"]),
                           "features_sha256_equal": bool(
                               fhash == row["features_sha256_k12"]),
                           "features_sha256": fhash},
                "held": [], "reference": {}}
            if not all(item["replay"][key] for key in
                       ("frame_equal", "obs_equal", "features_sha256_equal")):
                raise BombLongError(f"smoke: the replay did not land on the "
                                    f"recorded state: {item['replay']}")
            cand = int(row["cand"])
            actions = [2 * cand, 2 * cand + 1]
            park_reply = parent.park()
            got_frame = m432().fields(park_reply).get("frame")
            if got_frame != str(state_frame):
                raise BombLongError(f"smoke: PARK reports frame {got_frame}, the "
                                    f"read state is at {state_frame}")
            reply_digest = v450().Digest()
            for action in actions:
                child, spawn_reply = parent.spawn()
                reply_digest.add(spawn_reply)
                arm = hold_action(parent, child, action, horizon,
                                  digest=reply_digest, state_frame=state_frame,
                                  parent_state=state)
                quit_reply = parent.quit_child(child)
                reply_digest.add(quit_reply)
                arm["spawn_reply"] = spawn_reply
                arm["quit_reply"] = quit_reply
                validate_arm(arm, frames=horizon,
                             where=f"smoke arm action={action} "
                                   f"(bomb={action % 2})")
                item["held"].append({key: arm[key] for key in (
                    "action", "cand", "bomb", "alive_frames", "safe",
                    "frames_executed", "first_death_frame", "bombs_start",
                    "bombs_end", "bombs_used", "bomb_decrement_frame",
                    "terminal", "baseline_frame", "baseline_frame_ok",
                    "baseline_state_equal_parent")})
            ref, _children = v830().label_state36(
                parent, state=state, horizon=horizon, park_reply=park_reply,
                actions=actions)
            item["reference"] = {key: ref[key] for key in (
                "safe", "alive_frames", "bombs_start", "bombs_end", "bombs_used",
                "bomb_decrement_frame")}
            mismatched = []
            for index, arm in enumerate(item["held"]):
                if int(arm["alive_frames"]) != int(ref["alive_frames"][index]):
                    mismatched.append(f"alive_frames[{index}] "
                                      f"{arm['alive_frames']} != "
                                      f"{ref['alive_frames'][index]}")
                if int(arm["safe"]) != int(ref["safe"][index]):
                    mismatched.append(f"safe[{index}] {arm['safe']} != "
                                      f"{ref['safe'][index]}")
                if int(arm["bombs_start"]) != int(ref["bombs_start"][index]):
                    mismatched.append(f"bombs_start[{index}]")
                if int(arm["bombs_end"]) != int(ref["bombs_end"][index]):
                    mismatched.append(f"bombs_end[{index}]")
                if arm["bomb_decrement_frame"] != ref["bomb_decrement_frame"][index]:
                    mismatched.append(
                        f"bomb_decrement_frame[{index}] "
                        f"{arm['bomb_decrement_frame']} != "
                        f"{ref['bomb_decrement_frame'][index]}")
            item["equivalence_mismatches"] = mismatched
            item["equivalence"] = not mismatched
            if mismatched:
                payload["problems"].append(
                    f"s{seed}_w{window} ep{row['episode']} d{decision}: "
                    f"hold_action != label_state36: {mismatched}")
            payload["states"].append(item)
            log(f"smoke s{seed} w{window} ep{row['episode']} d{decision}: "
                f"held alive={[a['alive_frames'] for a in item['held']]} "
                f"bombs_after={[a['bombs_end'] for a in item['held']]} "
                f"decrement={[a['bomb_decrement_frame'] for a in item['held']]} "
                f"reference alive={ref['alive_frames']} "
                f"equiv={item['equivalence']}")
            first_mask, rest_mask = v830().action_masks(
                int(row["behaviour"]["action"]))
            died = False
            for frame_index in range(DECISION_FRAMES):
                step = parent.step(first_mask if frame_index == 0 else rest_mask)
                reading, _reply = parent.state()
                term = cf._terminal_of(
                    reading, deaths_start=deaths0, stage0=stage0,
                    hash_hex=(step["hash"] or ""), frame=int(reading["frame"]),
                    exited=step["exited"], game_over=step["game_over"])
                state = reading
                if term is not None:
                    died = True
                    break
            if died and len(payload["states"]) < len(rows):
                raise BombLongError("smoke: the replay died before the last "
                                    "requested state")
        payload["all_pass"] = bool(payload["states"]
                                   and not payload["problems"])
    except Exception as exc:                                          # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["all_pass"] = False
        log(f"SMOKE FAILED: {payload['error']}")
    finally:
        for parent in parents:
            try:
                parent.close()
            except Exception:                                         # noqa: BLE001
                pass
        payload["wall_s"] = round(time.perf_counter() - started, 3)
        write_json(out, payload)
    log(f"smoke: all_pass={payload['all_pass']} states={len(payload['states'])} "
        f"problems={len(payload['problems'])} wall={payload['wall_s']}s -> {out}")
    return 0 if payload["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: stats (pool + P0: does the long horizon discriminate at all?)
# --------------------------------------------------------------------------- #
def p0_block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"states": len(rows)}
    if not rows:
        out.update({"bomb1_longer": 0, "equal": 0, "bomb1_shorter": 0,
                    "bomb1_longer_fraction": None})
        return out
    longer = sum(1 for row in rows if row["comparison"] == "bomb1_longer")
    equal = sum(1 for row in rows if row["comparison"] == "equal")
    shorter = sum(1 for row in rows if row["comparison"] == "bomb1_shorter")
    surv_both = sum(1 for row in rows
                    if row["survive_600_bomb0"] and row["survive_600_bomb1"])
    surv_only1 = sum(1 for row in rows
                     if row["survive_600_bomb1"] and not row["survive_600_bomb0"])
    surv_only0 = sum(1 for row in rows
                     if row["survive_600_bomb0"] and not row["survive_600_bomb1"])
    surv_neither = sum(1 for row in rows
                       if not row["survive_600_bomb0"]
                       and not row["survive_600_bomb1"])
    deltas = [int(row["delta_alive_600"]) for row in rows]
    out.update({
        "bomb1_longer": longer, "equal": equal, "bomb1_shorter": shorter,
        "bomb1_longer_fraction": round(longer / len(rows), 6),
        "equal_fraction": round(equal / len(rows), 6),
        "bomb1_shorter_fraction": round(shorter / len(rows), 6),
        "delta_alive_600": quantiles(deltas),
        "survive_600_crosstab": {
            "both": surv_both, "bomb1_only": surv_only1,
            "bomb0_only": surv_only0, "neither": surv_neither},
        "survive_600_rates": {
            "bomb0": round(statistics.fmean(
                [int(row["survive_600_bomb0"]) for row in rows]), 6),
            "bomb1": round(statistics.fmean(
                [int(row["survive_600_bomb1"]) for row in rows]), 6)},
        "alive_frames_600": {
            "bomb0": quantiles([row["alive_frames_600_bomb0"] for row in rows]),
            "bomb1": quantiles([row["alive_frames_600_bomb1"] for row in rows])},
        "bombs_after": {"bomb0": counter([row["bombs_after_bomb0"] for row in rows]),
                        "bomb1": counter([row["bombs_after_bomb1"] for row in rows])},
        "bomb1_branch_kind": counter([row["bomb1_branch_kind"] for row in rows]),
    })
    return out


def mode_stats(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    cell_files = sorted((root / "labels").glob("seed*_w*.jsonl")) \
        if (root / "labels").is_dir() else []
    if not cell_files:
        raise BombLongError(f"no cell files under {root / 'labels'}")
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
            "bombs_remaining_counts": counter([r["bombs_remaining"]
                                               for r in cell_rows]),
            "episodes": len({int(r["episode"]) for r in cell_rows}),
            "path": str(path)}
        sources.append({"path": str(path), "sha256": sha256_file(path),
                        "bytes": path.stat().st_size, "states": len(cell_rows)})
    with pooled.open("w", encoding="utf-8") as handle:
        for row in rows:
            write_jsonl_line(handle, row)
    pooled_sha = sha256_file(pooled)

    states = len(rows)
    bombs_counts = counter([r["bombs_remaining"] for r in rows])
    by_bombs: dict[str, Any] = {}
    for level in ("3", "2", "1", "0"):
        subset = [r for r in rows if str(int(r["bombs_remaining"])) == level]
        by_bombs[level] = p0_block(subset)
    key_subset = [r for r in rows if int(r["bombs_remaining"]) in (1, 2)]
    p0 = p0_block(key_subset)
    fraction = p0.get("bomb1_longer_fraction")
    p0_pass = bool(fraction is not None and fraction < P0_THRESHOLD)
    payload: dict[str, Any] = {
        "schema": SCHEMA_STATS, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "no_statistics": ("counts, rates, means, medians, quantiles and "
                          "distributions only; no test, no interval, no p-value"),
        "H": sorted({int(r["H"]) for r in rows}),
        "decisions": sorted({int(r["decisions"]) for r in rows}),
        "decision_frames": sorted({int(r["decision_frames"]) for r in rows}),
        "labels_file": {"path": str(pooled), "sha256": pooled_sha,
                        "bytes": pooled.stat().st_size, "states": states,
                        "arms": states * 2},
        "sources": sources,
        "states": states,
        "episodes": sum(int(v["episodes"]) for v in per_cell.values()),
        "windows": sorted({int(r["window"]) for r in rows}),
        "seeds": sorted({int(r["seed"]) for r in rows}),
        "bombs_remaining": {
            "distribution": bombs_counts,
            "fractions": {key: round(value / states, 6)
                          for key, value in bombs_counts.items()},
            "note": ("every v8.3.0 state had bombs_remaining = 3 (5,166/5,166); "
                     "these are the v8.6.0 behaviour policy's states")},
        # ---- the key deliverable: the H = 600 discrimination table -----------
        "p0": {
            "requirement": (f"prereg 搂2 P0: on states with bombs_remaining in "
                            f"{{1,2}}, the fraction where bomb = 1's "
                            f"alive_frames_600 is strictly greater than bomb = 0's "
                            f"must be < {P0_THRESHOLD:.0%}"),
            "definition": ("alive_frames_600 = the frames the arm survived before its "
                           "first death, or 600 when it never died inside the "
                           "horizon (R24's rule, reused)"),
            "on_bombs_1_and_2": p0,
            "bomb1_longer_fraction": fraction,
            "threshold": P0_THRESHOLD,
            "pass": p0_pass,
            "stop_condition": ("S-horizon: if the fraction is >= "
                               f"{P0_THRESHOLD:.0%} the long horizon does not "
                               "discriminate either; the round publishes the labels "
                               "and the statistics and stops (no training, no "
                               "evaluation)"),
        },
        "by_bombs_remaining": by_bombs,
        "comparison_short_vs_long": {
            "note": ("the same states, the same candidate, two horizons: the v8.6.0 "
                     "H = 120 pair (from the replayed row) and this round's H = 600 "
                     "pair.  Descriptive only."),
            "short_bomb1_longer": sum(
                1 for r in rows
                if int(r["short_alive_bomb1"][int(r["cand"])])
                > int(r["short_alive_bomb0"][int(r["cand"])])),
            "short_equal": sum(
                1 for r in rows
                if int(r["short_alive_bomb1"][int(r["cand"])])
                == int(r["short_alive_bomb0"][int(r["cand"])])),
            "short_bomb1_shorter": sum(
                1 for r in rows
                if int(r["short_alive_bomb1"][int(r["cand"])])
                < int(r["short_alive_bomb0"][int(r["cand"])])),
        },
        "per_cell": per_cell,
        "cost": {"label_wall_s": quantiles([r["label_wall_s"] for r in rows])},
        "per_cell_arm_kinds": counter([r["bomb1_branch_kind"] for r in rows]),
    }
    write_json(out, payload)
    log(f"stats: {states} states, bombs={bombs_counts}, P0 on bombs 1|2: "
        f"longer={p0.get('bomb1_longer')} equal={p0.get('equal')} "
        f"shorter={p0.get('bomb1_shorter')} fraction={fraction} -> "
        f"{'PASS' if p0_pass else 'FAIL'} (threshold {P0_THRESHOLD})")
    for level in ("3", "2", "1"):
        block = by_bombs[level]
        log(f"  bombs={level}: states={block.get('states')} "
            f"longer={block.get('bomb1_longer')} equal={block.get('equal')} "
            f"shorter={block.get('bomb1_shorter')} "
            f"frac={block.get('bomb1_longer_fraction')}")
    if not p0_pass:
        log("S-horizon: the H = 600 label does not discriminate; the round "
            "publishes labels and statistics and stops before training")
        return 4
    return 0


# --------------------------------------------------------------------------- #
# mode: train
# --------------------------------------------------------------------------- #
def load_long_by_key(root: pathlib.Path) -> dict[tuple[int, int, int, int],
                                                dict[str, Any]]:
    out: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for path in sorted((root / "labels").glob("seed*_w*.jsonl")):
        for row in read_jsonl(path):
            key = (int(row["seed"]), int(row["window"]), int(row["episode"]),
                   int(row["decision"]))
            out[key] = row
    return out


def mode_train(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    root = pathlib.Path(args.work_dir)
    seed = int(args.seed)
    if seed not in TRAIN_SEEDS:
        raise BombLongError(f"--seed must be one of {TRAIN_SEEDS}, got {seed}")
    states = load_states(root)
    if not states:
        raise BombLongError("no state snapshot")
    long_by_key = load_long_by_key(root)
    if not long_by_key:
        raise BombLongError("no long-horizon labels")
    pins = check_pins()
    if not pins["all_match"]:
        raise BombLongError(f"pinned import moved: {json.dumps(pins)[:800]}")

    feats, labels36, array_info = v830().build_arrays(states)
    index: dict[tuple[int, int, int, int], int] = {}
    for position, row in enumerate(states):
        index[(int(row["seed"]), int(row["window"]), int(row["episode"]),
               int(row["decision"]))] = position
    missing = [key for key in long_by_key if key not in index]
    if missing:
        raise BombLongError(f"{len(missing)} long labels have no short state "
                            f"(first {missing[0]})")
    n = len(states)
    long_mask = np.zeros((n,), dtype=bool)
    long_idx = np.zeros((n, 2), dtype=np.int64)
    long_survive = np.zeros((n, 2), dtype=np.float32)
    long_frames = np.zeros((n, 2), dtype=np.float32)
    for key, row in long_by_key.items():
        position = index[key]
        cand = int(row["cand"])
        long_mask[position] = True
        long_idx[position] = (cand * 2, cand * 2 + 1)
        long_survive[position] = (float(row["survive_600_bomb0"]),
                                  float(row["survive_600_bomb1"]))
        long_frames[position] = (float(row["alive_frames_600_bomb0"]) / float(LONG_H),
                                 float(row["alive_frames_600_bomb1"]) / float(LONG_H))
    positives = float(long_survive[long_mask].mean()) if long_mask.any() else None
    if positives is None:
        raise BombLongError("no long label joined to the state set")
    imbalance = max(positives, 1.0 - positives)
    target_kind = "bce_survive_600" if imbalance <= BALANCE_THRESHOLD else \
        "mse_alive_frames_600_normalised"
    log(f"train seed={seed}: {n} states, {int(long_mask.sum())} with a long label, "
        f"survive_600 positive rate {positives:.4f} -> bomb-head target "
        f"{target_kind}")

    x = torch.from_numpy(feats)
    y = torch.from_numpy(labels36)
    m = torch.from_numpy(long_mask)
    idx = torch.from_numpy(long_idx)
    target = torch.from_numpy(long_survive if target_kind == "bce_survive_600"
                              else long_frames)
    policy = v830().policy36(seed=seed)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    loss_short = torch.nn.BCEWithLogitsLoss()
    loss_long_bce = torch.nn.BCEWithLogitsLoss()
    loss_long_mse = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)
    batch = int(args.batch)
    epoch_losses: list[float] = []
    epoch_short: list[float] = []
    epoch_bomb: list[float] = []
    started = time.perf_counter()
    for _epoch in range(int(EPOCHS)):
        perm = torch.from_numpy(rng.permutation(n))
        total = 0.0
        total_short = 0.0
        total_bomb = 0.0
        bomb_rows = 0
        policy.train()
        for start in range(0, n, batch):
            selection = perm[start:start + batch]
            logits = policy(x.index_select(0, selection))
            short = loss_short(logits, y.index_select(0, selection))
            mask_batch = m.index_select(0, selection)
            bomb_term = torch.zeros((), dtype=torch.float32)
            if bool(mask_batch.any()):
                idx_batch = idx.index_select(0, selection)
                target_batch = target.index_select(0, selection)
                picked = logits.gather(1, idx_batch)[mask_batch]
                wanted = target_batch[mask_batch]
                if target_kind == "bce_survive_600":
                    bomb_term = loss_long_bce(picked, wanted)
                else:
                    bomb_term = loss_long_mse(torch.sigmoid(picked), wanted)
                bomb_rows += int(mask_batch.sum())
            loss = short + BOMB_HEAD_WEIGHT * bomb_term
            opt.zero_grad()
            loss.backward()
            opt.step()
            weight = int(selection.numel())
            total += float(loss.detach()) * weight
            total_short += float(short.detach()) * weight
            total_bomb += float(bomb_term.detach()) * weight
        epoch_losses.append(total / n)
        epoch_short.append(total_short / n)
        epoch_bomb.append(total_bomb / n)
    wall = round(time.perf_counter() - started, 3)

    policy.eval()
    with torch.no_grad():
        logits = policy(x)
        final_short = float(loss_short(logits, y))
        predicted = (torch.sigmoid(logits) >= 0.5).float()
        overall_acc = float((predicted == y).float().mean())
        per_action_acc = [float((predicted[:, action] == y[:, action]).float().mean())
                          for action in range(N_ACTIONS)]
        probs = torch.sigmoid(logits)
        mean_prob_bomb0 = float(probs[:, [a for a in range(N_ACTIONS)
                                          if a % 2 == 0]].mean())
        mean_prob_bomb1 = float(probs[:, [a for a in range(N_ACTIONS)
                                          if a % 2 == 1]].mean())
        picked_all = logits.gather(1, idx)[m]
        wanted_all = target[m]
        if target_kind == "bce_survive_600":
            final_bomb = float(loss_long_bce(picked_all, wanted_all))
        else:
            final_bomb = float(loss_long_mse(torch.sigmoid(picked_all), wanted_all))
        # the bomb-head sign agreement: does the arm the model prefers match the
        # arm the H = 600 label prefers?  descriptive only, no test
        prefer = (picked_all[:, 1] > picked_all[:, 0]).float()
        long_gap = (long_frames[m][:, 1] - long_frames[m][:, 0])
        label_prefer = (long_gap > 0).float()
        agree = float((prefer == label_prefer).float().mean())
        tied = int((long_gap == 0).sum())

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else policy_ckpt_path(root, seed)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), str(ckpt))
    report_path = (pathlib.Path(args.out) if args.out
                   else train_report_path(root, seed))
    report: dict[str, Any] = {
        "schema": SCHEMA_TRAIN, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "seed": seed, "states": n, "states_with_long_label": int(long_mask.sum()),
        "feature_dim": int(FEATURE_DIM), "actions": int(N_ACTIONS),
        "model": {"class": "v830_bombpol.policy36 (v431_arm_a.Policy widened to 36)",
                  "shape": f"Linear({FEATURE_DIM},{HIDDEN}) -> Tanh -> "
                           f"Linear({HIDDEN},36)",
                  "params": int(sum(p.numel() for p in policy.parameters()))},
        "candidate_head": {
            "target": "the v8.6.0 36-way short-horizon label (H = 120), unchanged",
            "loss": "BCEWithLogitsLoss over the 36 outputs",
            "states": n},
        "bomb_head": {
            "target": ("survive_600 (0/1) of the selected candidate's two arms"
                       if target_kind == "bce_survive_600" else
                       "alive_frames_600 / 600 of the selected candidate's two arms"),
            "target_kind": target_kind,
            "target_choice_rule": (f"BCE by default; switch to the normalised "
                                   f"alive_frames MSE when the positive rate leaves "
                                   f"[{BALANCE_FLOOR}, {BALANCE_THRESHOLD}]"),
            "survive_600_positive_rate": round(positives, 6),
            "imbalance": round(imbalance, 6),
            "states": int(long_mask.sum()),
            "arms": int(long_mask.sum()) * 2,
            "loss": ("BCEWithLogitsLoss on the two picked logits"
                     if target_kind == "bce_survive_600" else
                     "MSELoss between sigmoid(logit) and the normalised frames"),
            "weight": BOMB_HEAD_WEIGHT,
            "logits_used": "logit[cand*2] and logit[cand*2+1] of the selected cand"},
        "total_loss": ("BCE36(short) + "
                       f"{BOMB_HEAD_WEIGHT} * bomb_head"),
        "optimizer": {"name": "Adam", "lr": LR, "batch": batch, "epochs": EPOCHS,
                      "early_stopping": None,
                      "shuffle": "numpy default_rng(seed), one permutation/epoch"},
        "init": {"torch_manual_seed": seed, "hidden": HIDDEN,
                 "feature_dim": int(FEATURE_DIM)},
        "features": array_info,
        "epoch_losses": [round(value, 6) for value in epoch_losses],
        "epoch_losses_short": [round(value, 6) for value in epoch_short],
        "epoch_losses_bomb": [round(value, 6) for value in epoch_bomb],
        "loss_first_epoch": round(epoch_losses[0], 6),
        "loss_last_epoch": round(epoch_losses[-1], 6),
        "final_train_bce_36": round(final_short, 6),
        "final_train_bomb_head": round(final_bomb, 6),
        "train_accuracy": round(overall_acc, 6),
        "train_accuracy_per_action": [round(v, 6) for v in per_action_acc],
        "mean_predicted_prob_even": round(mean_prob_bomb0, 6),
        "mean_predicted_prob_odd": round(mean_prob_bomb1, 6),
        "bomb_head_agreement": {
            "note": ("descriptive only: the share of long-labelled states where the "
                     "arm with the higher logit is also the arm with the higher H = "
                     "600 alive_frames.  A tie in the label counts against the model."),
            "agreement": round(agree, 6), "states": int(long_mask.sum()),
            "label_ties": tied},
        "ckpt": {"path": str(ckpt), "sha256": sha256_file(ckpt),
                 "bytes": ckpt.stat().st_size},
        "pins": pins, "wall_s": wall,
        "deviations": (
            [] if target_kind == "bce_survive_600" else
            [f"bomb head target switched from survive_600 BCE to the normalised "
             f"alive_frames_600 MSE: the survive_600 positive rate is "
             f"{positives:.4f}, outside [{BALANCE_FLOOR}, {BALANCE_THRESHOLD}] "
             f"(the >90/10 imbalance rule of the work order)"]),
    }
    write_json(report_path, report)
    log(f"train seed={seed}: loss {report['loss_first_epoch']:.4f} -> "
        f"{report['loss_last_epoch']:.4f} (36-way BCE {final_short:.4f}, bomb head "
        f"{final_bomb:.4f} {target_kind}, agree={agree:.4f}) wall={wall}s -> {ckpt} "
        f"sha256={report['ckpt']['sha256'][:16]}鈥?)
    return 0


def mode_train_summary(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out_path = pathlib.Path(args.out) if args.out else root / "train_summary.json"
    out: dict[str, Any] = {
        "schema": SCHEMA_TRAIN_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "design": ("Adam 1e-3, batch 256, 200 epochs, 3 seeds; candidate head = the "
                   "v8.6.0 36-way short BCE, bomb head = the H = 600 two-arm label"),
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
            "states": rec["states"],
            "states_with_long_label": rec["states_with_long_label"],
            "loss_first_epoch": rec["loss_first_epoch"],
            "loss_last_epoch": rec["loss_last_epoch"],
            "final_train_bce_36": rec["final_train_bce_36"],
            "final_train_bomb_head": rec["final_train_bomb_head"],
            "bomb_head_target": rec["bomb_head"]["target_kind"],
            "survive_600_positive_rate": rec["bomb_head"]["survive_600_positive_rate"],
            "bomb_head_agreement": rec["bomb_head_agreement"]["agreement"],
            "train_accuracy": rec["train_accuracy"],
            "ckpt": rec["ckpt"], "wall_s": rec["wall_s"]}
    write_json(out_path, out)
    log(f"train-summary: {len(out['seeds'])}/{len(TRAIN_SEEDS)} seeds -> {out_path}")
    return 0 if out["all_pass"] else 4


# --------------------------------------------------------------------------- #
# mode: eval (one budget x one seed x 4 dev windows)
# --------------------------------------------------------------------------- #
def mode_eval(args: argparse.Namespace) -> int:
    import torch

    from touhou_rl.v420 import contracts as C

    root = pathlib.Path(args.work_dir)
    seed = int(args.policy_seed)
    budget = int(args.decisions)
    if seed not in TRAIN_SEEDS:
        raise BombLongError(f"--policy-seed must be one of {TRAIN_SEEDS}, got {seed}")
    if budget not in BUDGETS:
        raise BombLongError(f"prereg 搂1 fixes --decisions to {BUDGETS}, got {budget}")
    if int(args.decision_frames) != DECISION_FRAMES:
        raise BombLongError(f"prereg 搂1 keeps {DECISION_FRAMES} frames per decision")
    if int(args.episodes) < 1:
        raise BombLongError(f"--episodes must be >= 1, got {args.episodes}")
    windows = parse_windows(args.windows)
    engine = engine_gate(args.engine)
    launch_env = prepare_env()
    tag = args.tag or f"v870-eval-d{budget}-s{seed}"
    run_dir = root / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    data_root = root / "data" / tag
    out_path = (pathlib.Path(args.out) if args.out
                else eval_path(root, budget, seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")
    ckpt = pathlib.Path(args.ckpt) if args.ckpt else policy_ckpt_path(root, seed)
    digest = sha256_file(ckpt) if ckpt.is_file() else None
    policy = v830().load_policy36(ckpt, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(generator_seed(seed))
    arm = arm_a()

    def decide(state: dict[str, Any], hazards: Any,
               decision: int) -> tuple[int, int]:
        feats, _detail = arm.build_features(state=state, hazards=hazards,
                                            decision_index=decision)
        action = v830().sample_action36(policy, feats, args.kind, generator)
        return v830().split_action(action)

    summary: dict[str, Any] = {
        "schema": SCHEMA_EVAL_SUMMARY, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "mode": "eval", "policy_seed": seed, "windows": list(windows),
        "episodes_per_window": int(args.episodes),
        "budget_decisions": budget, "decision_frames": DECISION_FRAMES,
        "temperature": TEMPERATURE, "kind": args.kind,
        "engine": engine, "launch_env": launch_env,
        "ckpt": {"path": str(ckpt), "sha256": digest, "bytes":
                 ckpt.stat().st_size if ckpt.is_file() else None},
        "loop": "v830_bombpol.run_episode36 (imported, unmodified)",
        "sampler": {"kind": "torch.multinomial on softmax(z/1) over 36 actions",
                    "generator_seed": generator_seed(seed),
                    "generator": ("one torch.Generator per (budget, seed) process, "
                                  "windows in the frozen order 11,12,17,18")},
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
                f"sha256={data_info['cfg']['sha256'][:16]}鈥?)
            parent, gate = start_parent(slot=slot, window=window,
                                        engine=args.engine, data_dir=data_dir,
                                        log_dir=run_dir)
            parents.append(parent)
            rows: list[dict[str, Any]] = []
            for episode in range(int(args.episodes)):
                record = v830().run_episode36(
                    parent, episode=episode, policy_seed=seed, decisions=budget,
                    decide=decide, budget=budget,
                    decision_frames=DECISION_FRAMES)
                record["schema"] = SCHEMA_EVAL
                record["producer"] = PRODUCER
                record["loop"] = "v830_bombpol.run_episode36 (imported, unmodified)"
                record.update({"ckpt_sha256": digest,
                               "generator_seed": generator_seed(seed)})
                record.pop("terminal_reply", None)
                record.pop("trace", None)
                record.pop("spawn_reply", None)
                write_jsonl_line(handle, record)
                rows.append(record)
                if record.get("invalid"):
                    raise BombLongError(f"d{budget} seed {seed} window {window} "
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
                "last_bomb_decisions": sorted(
                    row["bomb_decisions"][-1] for row in rows
                    if row["bomb_decisions"]),
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
    return [row for row in read_jsonl(path) if row.get("kind") == "episode"]


def mode_diagnostics(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.work_dir)
    out = pathlib.Path(args.out) if args.out else root / "diagnostics.json"
    usage_out = (pathlib.Path(args.usage_out) if args.usage_out
                 else root / "bomb_usage.json")
    payload: dict[str, Any] = {
        "schema": SCHEMA_DIAG, "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prereg": PREREG_REL,
        "no_statistics": ("counts, rates, means, medians, quantiles and "
                          "distributions only; no test, no interval, no p-value"),
        "budgets": {}, "per_cell": {}, "references": {
            "v830_bombpol": BASELINE_V830, "v860_bombdagger": BASELINE_V860,
            "rule_r5": RULE_R5},
        "budget_comparison_warning": ("prereg 搂4: the three budgets must not be "
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
        last_bomb = [int(row["bomb_decisions"][-1]) for row in rows
                     if row.get("bomb_decisions")]
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
                "last_bomb_decision": quantiles(last_bomb),
                "last_bomb_decision_histogram": first_bomb_histogram(last_bomb),
                "episodes_never_bombed": len(rows) - len(bomb_episodes),
                "episodes_using_1_or_2_bombs": sum(1 for value in bombs_used
                                                   if 1 <= value <= 2),
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
# mode: audio-audit / process-audit / manifest / publish
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
                        "may survive its own run (filtered by --data-dir)"),
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
        raise BombLongError(f"not a directory: {root}")
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
        (pooled_labels_path(root), "labels_long.jsonl"),
        (pooled_stats_path(root), "labels_long_stats.json"),
        (select_report_path(root), "select.json"),
        (root / "preflight.json", "preflight.json"),
        (root / "ckpt_assertions.json", "ckpt_assertions.json"),
        (root / "ckpt_assertions_final.json", "ckpt_assertions_final.json"),
        (root / "fidelity_d3.json", "fidelity_d3.json"),
        (root / "fidelity_d3_wrapper.json", "fidelity_d3_wrapper.json"),
        (root / "p0.json", "p0.json"),
        (root / "train_summary.json", "train_summary.json"),
        (root / "diagnostics.json", "diagnostics.json"),
        (root / "bomb_usage.json", "bomb_usage.json"),
        (root / "audio_audit.json", "audio_audit.json"),
        (root / "process_residue.json", "process_residue.json"),
        (root / "run.sh", "scripts/run.sh"),
        (root / "commands.tsv", "commands.tsv"),
        (root / "RUN.md", "RUN.md"),
    ]
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
    for sub in ("logs", "runs", "collect"):
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
               "prereg": PREREG_REL, "dest": str(dest), "files": copied,
               "file_count": len(copied),
               "v870_bomblong_py_sha256":
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
        description="v8.7.0 BOMBLONG-001: long-horizon (H=600) counterfactual "
                    "labels for the bomb decision")
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "assert-ckpt", "fidelity", "select",
                                 "smoke", "collect", "stats", "train",
                                 "train-summary", "eval", "diagnostics",
                                 "audio-audit", "process-audit", "manifest",
                                 "publish"))
    parser.add_argument("--seed", type=int, default=0,
                        help="training seed (0/1/2)")
    parser.add_argument("--policy-seed", type=int, default=None,
                        help="eval: alias of --seed")
    parser.add_argument("--windows", default="13,14,15,16")
    parser.add_argument("--decisions", type=int, default=LONG_DECISIONS,
                        help=f"collect: {LONG_DECISIONS}; eval: one of {BUDGETS}")
    parser.add_argument("--decision-frames", type=int, default=DECISION_FRAMES)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--horizon", type=int, default=LONG_H)
    parser.add_argument("--states", type=int, default=1,
                        help="smoke: how many states of the cell to run")
    parser.add_argument("--limit-states", type=int, default=0,
                        help="collect: diagnostic only, stop each cell after N states")
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--kind", default="stochastic",
                        choices=("stochastic", "greedy"))
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--engine", default=ENGINE_DEFAULT)
    parser.add_argument("--data-source", default=DATA_SOURCE_DEFAULT)
    parser.add_argument("--work-dir", default=str(WORK))
    parser.add_argument("--labels-out", default="")
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

    engine_modes = ("fidelity", "smoke", "collect", "eval")
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
        if args.mode == "fidelity":
            return mode_fidelity(args)
        if args.mode == "select":
            return mode_select(args)
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
    except BombLongError as exc:
        log(f"FAILED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
