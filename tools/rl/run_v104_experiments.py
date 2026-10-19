"""v1.0.4 Stage3 / S3-4: first-round comparison and route decision (tool).

Five phases, one CLI (task book §4):

    --phase baseline    re-evaluate the frozen v1.0.4 baselines on the fixed A
                        validation split (7241 frames) without touching them
    --phase overfit     256 stratified centre frames from the A *training* split,
                        <=400 steps, to show whether the objective is learnable
    --phase throughput  10 warm-up + 100 measured batches on three paths
                        (perception / policy-frozen / policy-joint)
    --phase matrix      two perception runs (depth 1 / depth 3) feeding six policy
                        runs from the same-depth checkpoint bytes
    --phase gate        read-only G-v3 verdict into ``gate-v3.md``/``gate-v3.json``

Contract points implemented here (and only here):

* **Every training command this tool emits carries ``--max-val-batches 0``.**
  ``training_command`` has no parameter for it, so a caller cannot shorten the
  validation pass; a short pass leaves the 0-15 danger bucket empty on the real
  split and the trainer then refuses to write ``policy.pt`` (S3-3 §10.3), which
  would make the whole matrix unusable.
* Run directories are ``runs/v104-<experiment>-s<seed>``; an existing *valid*
  run is never overwritten.  A run whose recipe matches exactly is reused
  (``reused=true``), anything else is a ``v104:EVIDENCE`` stop.
* ``pilot-v1`` / ``v104-a01-72ep`` / ``v104-clean-24e12e`` and their aliases are
  read-only inputs; nothing here writes into a run directory it did not create.
* ``--screening`` and the smoke switches (``--smoke-epochs`` /
  ``--smoke-train-batches``) force ``acceptance_eligible=false`` in the summary
  and in the experiment table.  The formal matrix never uses them.

The 60 s/epoch design ceiling is *recorded*, not enforced as a stop: the user
ruled (question list §8 Q5) that depth 3 may continue at the measured
~72 s/epoch without refactoring the data path or changing the perception budget.
Every row therefore carries the measured epoch seconds and a
``budget_exceeded`` flag.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402  (after the sys.path fix so the repo copy is used)

from touhou_rl import obs, vision, vision_history, vision_losses, vision_metrics, vision_train  # noqa: E402
from touhou_rl.pipeline.manifest import DatasetManifest, ManifestError, file_digest, index_digest  # noqa: E402
from touhou_rl.vision import regions_to_batch  # noqa: E402
from touhou_rl.vision_data import VisionDataset  # noqa: E402
from touhou_rl.vmodels import load_vision_checkpoint, vision_arch  # noqa: E402
from touhou_rl.vmodels2 import VisionNetConfigV2, VisionScorerV2, heatmap_target  # noqa: E402

# --------------------------------------------------------------------------- #
# fixed identities of this stage
# --------------------------------------------------------------------------- #
PHASE_SCHEMA = "v104-s34-phase/1"
TABLE_SCHEMA = "v104-experiment-table/1"
GATE_SCHEMA = "v104-gate/1"

DATASET_A = ROOT / "third_party" / "th08" / "build" / "rl" / "datasets" / "pilot-ud0059-v2"
RUNS_ROOT = ROOT / "third_party" / "th08" / "build" / "rl" / "runs"
S34_DIR = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-4"
EVIDENCE_ROOT = S34_DIR / "evidence"

OLD_STATE_RUN = "v104-a01-72ep"
OLD_POLICY_RUN = "v104-clean-24e12e"
FROZEN_RUN_NAMES = ("pilot-v1", OLD_STATE_RUN, OLD_POLICY_RUN)

BATCH_SIZE = 96
SEED = 7
VAL_FRAMES = 7241
TRAIN_FRAMES = 65172
TRAIN_BATCHES = TRAIN_FRAMES // BATCH_SIZE            # 678 full batches (drop_last)
VAL_BATCHES = math.ceil(VAL_FRAMES / BATCH_SIZE)      # 76 = 75 * 96 + 41
VAL_INDEX_SHA256 = "e5d4ebbe6d8de05793520a393fc1fdbea2b1e2c4db9ff7c1632908340c76655f"
TEACHER_FINGERPRINT = "a52416f9342fa00a0185c061771d09c22ba735bbe01c83dd7910a6c3bc8ec0f6"

STATE_EPOCHS = 72
POLICY_EPOCHS = 12
SCREENING_STATE_EPOCHS = 24
POLICY_FREEZE_EPOCHS = 2
HEATMAP_SIGMA = 1.0          # v2 recipe: sigma 1 for perception and policy
PERCEPTION_LR = 3e-4

OVERFIT_TARGET_FRAMES = 256
OVERFIT_MAX_STEPS = 400
OVERFIT_REDUCTION = 0.30
WARMUP_BATCHES = 10
MEASURE_BATCHES = 100
#: Epoch 1 (and partly 2) of every run pays the one-time page/decode cache fill.
#: 設計 §5.1 excludes such one-time cost from the 60 s/epoch comparison.
COLD_START_EPOCHS = 2

BUDGET_SECONDS = 60.0
A01_POS_PX = 10.0
A01_DANGER_MEDIAN = 0.8
A02_BUCKET_REGRET = 1.5
BUCKETS = ((0.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, 120.0), (120.0, float("inf")))

#: The six declared configurations (task book §3).  Nothing here is tunable by
#: the caller: a pre-declared matrix is one seed-7 run per configuration.
EXPERIMENTS = {
    "E0": {"depth": 1, "objective": "legacy-reg", "rank_weight": 0.0, "regression_weight": 1.0},
    "E1": {"depth": 3, "objective": "legacy-reg", "rank_weight": 0.0, "regression_weight": 1.0},
    "E2": {"depth": 3, "objective": "centered-reg", "rank_weight": 0.0, "regression_weight": 1.0},
    "E3a": {"depth": 3, "objective": "rank", "rank_weight": 1.0, "regression_weight": 0.0},
    "E3b": {"depth": 3, "objective": "rank", "rank_weight": 1.0, "regression_weight": 0.1},
    "E3c": {"depth": 3, "objective": "rank", "rank_weight": 1.0, "regression_weight": 1.0},
}
EXPERIMENT_ORDER = ("E0", "E1", "E2", "E3a", "E3b", "E3c")

#: Perception runs are shared by depth: depth 1 only feeds E0, depth 3 feeds the
#: other five.  The perception stage computes no policy loss, so the objective it
#: records is a declaration (legacy-reg) and each policy run re-declares its own.
#: ``run_directory`` expands these to ``v104-state-d1-s7`` (``-smoke-s7`` for a
#: smoke run).
STATE_RUNS = {1: "state-d1", 3: "state-d3"}

#: Fallback epoch estimates (seconds) when no throughput evidence is available.
#: State numbers come from S3-3's measured steady state (depth 1 = 39.1 s,
#: depth 3 = 71.15 s, both including the full 7241-frame validation pass); the
#: policy number is that plus a 15% auxiliary-head allowance and is *nominal*.
FALLBACK_EPOCH_SECONDS = {
    "state": {1: 39.1, 3: 71.15},
    "policy": {1: 45.0, 3: 82.0},
    "source": "fallback: S3-3 steady-state epoch (39.1 s d1 / 71.15 s d3) + 15% policy overhead (nominal)",
}

INIT_FROM_PLACEHOLDER = "<state-pass.pt|state.pt>"


# --------------------------------------------------------------------------- #
# errors / small helpers
# --------------------------------------------------------------------------- #
def v104_error(code: str, detail: str) -> ValueError:
    """The one error shape of the v104 contract: ``v104:<CODE>: <detail>``."""
    return ValueError(f"v104:{code}: {detail}")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path) -> str:
    return file_digest(Path(path))


def rel(path) -> str:
    """Workspace-relative POSIX path when possible (portable identities)."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_json(path) -> dict:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise v104_error("EVIDENCE", f"{path} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise v104_error("EVIDENCE", f"{path} does not hold a JSON object")
    return payload


def write_json(path, payload) -> Path:
    """Atomic JSON write: a partial file must never look like finished evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)
    return path


def refuse_existing_output(path: Path) -> None:
    """A finished phase output is write-once; a crashed one may be resumed."""
    path = Path(path)
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise v104_error("EVIDENCE", f"{path} exists but is unreadable; refusing to overwrite it")
    status = payload.get("status") if isinstance(payload, dict) else None
    if status in ("running", "partial"):
        return  # resumable: the run directories decide what still has to happen
    raise v104_error("EVIDENCE", f"{path} already holds a finished phase output; refusing to overwrite it")


def fmt_weight(value: float) -> str:
    """``1.0 -> "1"``, ``0.1 -> "0.1"``: the trainer's float() accepts both."""
    number = float(value)
    if number == int(number):
        return str(int(number))
    return f"{number:g}"


def env_block() -> dict:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "numpy": np.__version__,
        "platform": sys.platform,
        "cwd": str(ROOT),
    }


def pick_device(requested: str) -> str:
    return "cuda" if requested == "cuda" and torch.cuda.is_available() else requested


# --------------------------------------------------------------------------- #
# training command builder (the --max-val-batches 0 rule lives here)
# --------------------------------------------------------------------------- #
def training_command(
    *,
    out_dir,
    depth: int,
    stage: str,
    objective: str,
    rank_weight: float,
    regression_weight: float,
    epochs_state: int = 0,
    epochs_policy: int = 0,
    init_from=None,
    max_train_batches: int = 0,
    device: str = "cuda",
    dataset=DATASET_A,
    note: str = "",
    python: str | None = None,
) -> list[str]:
    """One ``python -m touhou_rl.vision_train`` argv for a v104 matrix run.

    ``--max-val-batches 0`` is hard-coded and there is deliberately **no
    parameter** that could change it: a shortened validation pass empties the
    0-15 danger bucket on the real split, the trainer then refuses to write
    ``policy.pt`` (``stages.policy.best = {}``), and the matrix would be
    worthless (S3-3 §10.3).  ``test_training_command_always_requests_full_validation``
    locks this down.
    """
    if stage not in ("state", "policy"):
        raise v104_error("CONFIG", f"matrix stage must be state|policy, got {stage!r}")
    if depth not in (1, 3):
        raise v104_error("CONFIG", f"v104 allows depth 1 or 3, got {depth!r}")
    if stage == "state" and int(epochs_state) < 1:
        raise v104_error("CONFIG", f"state run needs epochs_state >= 1, got {epochs_state!r}")
    if stage == "policy" and int(epochs_policy) < 1:
        raise v104_error("CONFIG", f"policy run needs epochs_policy >= 1, got {epochs_policy!r}")
    if objective not in vision_losses.OBJECTIVES:
        raise v104_error("CONFIG", f"unknown objective {objective!r}")
    if objective == "rank" and float(rank_weight) != 1.0:
        raise v104_error("CONFIG", f"rank objective requires rank_weight=1, got {rank_weight!r}")
    command = [
        python or sys.executable,
        "-m",
        "touhou_rl.vision_train",
        "--dataset",
        rel(dataset),
        "--out",
        rel(out_dir),
        "--arch",
        "v2",
        "--stage",
        stage,
        "--depth",
        str(int(depth)),
        "--selection-contract",
        "v104",
        "--history-contract",
        "causal-segment-v1",
        "--policy-objective",
        objective,
        "--rank-weight",
        fmt_weight(rank_weight),
        "--regression-weight",
        fmt_weight(regression_weight),
        "--policy-freeze-epochs",
        str(POLICY_FREEZE_EPOCHS),
        "--heatmap-sigma",
        fmt_weight(HEATMAP_SIGMA),
        "--batch-size",
        str(BATCH_SIZE),
        "--seed",
        str(SEED),
        "--epochs-state",
        str(int(epochs_state)),
        "--epochs-policy",
        str(int(epochs_policy)),
        "--max-val-batches",
        "0",  # D9: always the full 7241-frame validation pass
        "--device",
        device,
    ]
    if int(max_train_batches):
        command.extend(["--max-train-batches", str(int(max_train_batches))])
    if init_from is not None:
        text = str(init_from)
        command.extend(["--init-from", text if text.startswith("<") else rel(text)])
    if note:
        command.extend(["--note", note])
    return command


def command_line(command) -> str:
    """Copy-pasteable Windows command line."""
    return subprocess.list2cmdline([str(part) for part in command])


def run_directory(experiment: str, *, smoke: bool = False) -> Path:
    """``runs/v104-<experiment>-s<seed>``; smoke runs get their own name space."""
    name = f"v104-{experiment}-s{SEED}" if not smoke else f"v104-{experiment}-smoke-s{SEED}"
    return RUNS_ROOT / name


# --------------------------------------------------------------------------- #
# matrix planning
# --------------------------------------------------------------------------- #
def matrix_plan(
    experiments=EXPERIMENT_ORDER,
    *,
    smoke: bool = False,
    smoke_epochs: int = 1,
    smoke_train_batches: int = 2,
    screening: bool = False,
    device: str = "cuda",
    estimates: dict | None = None,
    python: str | None = None,
) -> list[dict]:
    """The ordered runs of one matrix invocation (perception first, then policy).

    Two perception runs are shared: every depth-3 policy run starts from the same
    depth-3 checkpoint *bytes*, so the six configurations differ only in their
    objective - never in their initialisation or their data order.
    """
    wanted = set(experiments or ())
    selected = [name for name in EXPERIMENT_ORDER if name in wanted]
    if not selected:
        raise v104_error("CONFIG", "matrix needs at least one of E0|E1|E2|E3a|E3b|E3c")
    if smoke and screening:
        raise v104_error("CONFIG", "--smoke and --screening are different modes; pick one")
    epochs_state = 1 if smoke else (SCREENING_STATE_EPOCHS if screening else STATE_EPOCHS)
    epochs_policy = 1 if smoke else POLICY_EPOCHS
    max_train_batches = int(smoke_train_batches) if smoke else 0
    table = estimates or FALLBACK_EPOCH_SECONDS

    plans: list[dict] = []
    depths: list[int] = []
    for name in selected:
        depth = EXPERIMENTS[name]["depth"]
        if depth not in depths:
            depths.append(depth)
    for depth in depths:
        directory = run_directory(STATE_RUNS[depth], smoke=smoke)
        command = training_command(
            out_dir=directory,
            depth=depth,
            stage="state",
            objective="legacy-reg",
            rank_weight=0.0,
            regression_weight=1.0,
            epochs_state=epochs_state,
            epochs_policy=0,
            max_train_batches=max_train_batches,
            device=device,
            note=f"S3-4 matrix perception depth {depth} (state stage)",
            python=python,
        )
        per_epoch = float(table["state"][depth])
        plans.append(
            {
                "kind": "state",
                "experiment": f"state-d{depth}",
                "depth": depth,
                "epochs": epochs_state,
                "max_train_batches": max_train_batches,
                "run_dir": directory,
                "command": command,
                "seconds_per_epoch": per_epoch,
                "seconds_per_epoch_source": table.get("source", "fallback"),
                "estimated_seconds": per_epoch * epochs_state,
                "expected_checkpoint": directory / "state.pt",
            }
        )
    for name in selected:
        spec = EXPERIMENTS[name]
        depth = spec["depth"]
        source_plan = next(plan for plan in plans if plan["kind"] == "state" and plan["depth"] == depth)
        directory = run_directory(name, smoke=smoke)
        command = training_command(
            out_dir=directory,
            depth=depth,
            stage="policy",
            objective=spec["objective"],
            rank_weight=spec["rank_weight"],
            regression_weight=spec["regression_weight"],
            epochs_state=0,
            epochs_policy=epochs_policy,
            init_from=INIT_FROM_PLACEHOLDER,
            max_train_batches=max_train_batches,
            device=device,
            note=f"S3-4 matrix {name}: depth {depth}, {spec['objective']}, seed {SEED}",
            python=python,
        )
        per_epoch = float(table["policy"][depth])
        plans.append(
            {
                "kind": "policy",
                "experiment": name,
                "depth": depth,
                "objective": spec["objective"],
                "rank_weight": spec["rank_weight"],
                "regression_weight": spec["regression_weight"],
                "epochs": epochs_policy,
                "freeze_epochs": min(POLICY_FREEZE_EPOCHS, epochs_policy),
                "max_train_batches": max_train_batches,
                "run_dir": directory,
                "command": command,
                "state_plan_index": plans.index(source_plan),
                "seconds_per_epoch": per_epoch,
                "seconds_per_epoch_source": table.get("source", "fallback"),
                "estimated_seconds": per_epoch * epochs_policy,
                "expected_checkpoint": directory / "policy.pt",
            }
        )
    return plans


def plan_total_seconds(plans) -> float:
    return float(sum(plan["estimated_seconds"] for plan in plans))


def bind_init_from(plan: dict, source_checkpoint: Path) -> list[str]:
    """Rebuild a policy plan's argv with the *actual* selected state checkpoint."""
    command = list(plan["command"])
    index = command.index("--init-from")
    command[index + 1] = rel(source_checkpoint)
    return command


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def bucket_label(low: float, high: float) -> str:
    return f"{int(low)}-{int(high)}" if math.isfinite(high) else f"{int(low)}+"


def assign_bucket(distance_px: float) -> str:
    """Danger bucket of one nearest-bullet distance (``inf`` -> ``"120+"``)."""
    value = float(distance_px)
    if math.isnan(value):
        raise v104_error("DATA", "nearest-bullet distance is NaN")
    if math.isinf(value) and value > 0:
        return "120+"
    for low, high in BUCKETS:
        if low <= value < high:
            return bucket_label(low, high)
    raise v104_error("DATA", f"distance {value!r} fell outside every danger bucket")


def apportion_256(sizes, target: int = OVERFIT_TARGET_FRAMES) -> dict:
    """Split ``target`` frames over non-empty strata (task book §4, overfit).

    Rule: every non-empty stratum takes one first; the rest is apportioned by
    largest remainder proportional to the remaining capacity, and a stratum that
    runs out of frames hands its excess back to the next round (the loop).  The
    arithmetic is deterministic: ties break on the stratum order.
    """
    sizes = [int(size) for size in sizes]
    if any(size < 0 for size in sizes):
        raise v104_error("DATA", "stratum sizes must be >= 0")
    non_empty = [index for index, size in enumerate(sizes) if size > 0]
    if not non_empty:
        raise v104_error("DATA", "no non-empty stratum to sample from")
    if len(non_empty) > int(target):
        raise v104_error(
            "DATA",
            f"{len(non_empty)} non-empty strata exceed the fixed {int(target)}-frame budget",
        )
    allocation = {index: 0 for index in range(len(sizes))}
    for index in non_empty:
        allocation[index] = 1
    remaining = int(target) - len(non_empty)
    rounds = 0
    while remaining > 0:
        rounds += 1
        if rounds > len(sizes) + 2:  # pragma: no cover - guard against a logic slip
            raise v104_error("DATA", "apportionment did not converge")
        capacity = {index: sizes[index] - allocation[index] for index in non_empty}
        active = [index for index in non_empty if capacity[index] > 0]
        if not active:
            break
        total_capacity = sum(capacity[index] for index in active)
        take = min(remaining, total_capacity)
        exact = {index: take * capacity[index] / total_capacity for index in active}
        base = {index: int(math.floor(exact[index])) for index in active}
        leftover = take - sum(base.values())
        order = sorted(active, key=lambda index: (-(exact[index] - base[index]), index))
        for index in order[:leftover]:
            base[index] += 1
        for index in active:
            allocation[index] += base[index]
        remaining -= take
    return {"allocation": allocation, "unfilled": int(remaining)}


def overfit_entry_status(initial_regret, final_regret, threshold: float = OVERFIT_REDUCTION) -> dict:
    """G-v3 small-set entry: never divide by a degenerate initial regret.

    ``initial <= 0`` or non-finite is ``blocked`` with
    ``reason=degenerate_initial_regret``: zero regret means every candidate tied
    (the teacher cannot rank them), so "30% lower" is undefined - not a pass, and
    not a reason to change the sample or the threshold (design §5.2).
    """
    block = {
        "threshold": float(threshold),
        "initial_regret_px": initial_regret,
        "final_regret_px": final_regret,
        "reduction": None,
        "line_reached": False,
    }
    try:
        initial = float(initial_regret)
    except (TypeError, ValueError):
        block.update({"status": "blocked", "reason": "degenerate_initial_regret"})
        return block
    if not math.isfinite(initial) or initial <= 0.0:
        block.update({"status": "blocked", "reason": "degenerate_initial_regret"})
        return block
    try:
        final = float(final_regret)
    except (TypeError, ValueError):
        block.update({"status": "fail", "reason": "final_regret_unavailable"})
        return block
    if not math.isfinite(final):
        block.update({"status": "fail", "reason": "final_regret_non_finite"})
        return block
    reduction = (initial - final) / initial
    block["reduction"] = reduction
    block["line_reached"] = bool(reduction >= float(threshold) - 1e-12)
    block["status"] = "pass" if block["line_reached"] else "fail"
    block["reason"] = "reduction_at_least_30pct" if block["line_reached"] else "reduction_below_30pct"
    return block


def forecast_epoch_seconds(train_mean: float, val_mean: float) -> float:
    """``floor(65172/96) * train + ceil(7241/96) * val`` (task book §4)."""
    train_mean = float(train_mean)
    val_mean = float(val_mean)
    if train_mean < 0 or val_mean < 0:
        raise v104_error("CONFIG", "batch means must be >= 0")
    return TRAIN_BATCHES * train_mean + VAL_BATCHES * val_mean


def budget_exceeded(epoch_seconds) -> bool:
    """Record (never enforce) the 60 s/epoch design ceiling.

    The user ruled (question list §8 Q5) that depth 3 continues at the measured
    ~72 s/epoch, so an over-limit epoch is a *flag* on the row, not a stop.  This
    helper exists so the flag is computed one way everywhere and can be tested.

    The comparison is on the **steady state** (:func:`epoch_seconds_summary`): the
    design excludes the one-time cache/label build, which is exactly what makes the
    first two epochs of every run 2-3x slower.  Comparing the cold-start epoch
    would report a "+100%" overrun that no steady epoch actually has.
    """
    return epoch_seconds_summary(epoch_seconds)["steady_exceeds_ceiling"]


def budget_overrun(epoch_seconds) -> float | None:
    """Steady-state overrun as a fraction of the ceiling (``0.203`` = +20.3%)."""
    return epoch_seconds_summary(epoch_seconds)["steady_overrun_fraction"]


def epoch_seconds_summary(epoch_seconds, *, cold_start: int = COLD_START_EPOCHS) -> dict:
    """Per-epoch timing summary with the one-time warm-up epochs separated.

    設計 §5.1: "前3完整epoch核实≤60s/epoch（训练+全验证，不计一次性cache/labels）".
    Epoch 1 (and to a lesser degree epoch 2) pays the page-cache and decode-cache
    fill for the shards it touches, so it is reported separately and never used
    for the ceiling comparison.
    """
    values = [float(value) for value in epoch_seconds or []]
    skip = max(0, min(int(cold_start), max(0, len(values) - 1)))
    steady = values[skip:]
    steady_max = max(steady) if steady else None
    overrun = round(steady_max / BUDGET_SECONDS - 1.0, 4) if steady_max is not None else None
    return {
        "epochs": len(values),
        "cold_start_epochs": skip,
        "first_epoch_seconds": values[0] if values else None,
        "cold_start_seconds": values[:skip],
        "all_max_seconds": max(values) if values else None,
        "steady_mean_seconds": round(sum(steady) / len(steady), 2) if steady else None,
        "steady_max_seconds": steady_max,
        "steady_min_seconds": min(steady) if steady else None,
        "steady_exceeds_ceiling": bool(steady_max is not None and steady_max > BUDGET_SECONDS),
        "steady_overrun_fraction": overrun if (overrun is not None and overrun > 0) else (0.0 if steady_max is not None else None),
        "limit_seconds": BUDGET_SECONDS,
    }


def policy_mixed_epoch_seconds(
    frozen_epoch: float,
    joint_epoch: float,
    *,
    freeze: int = POLICY_FREEZE_EPOCHS,
    total: int = POLICY_EPOCHS,
) -> float:
    """Mean epoch of the real 12-epoch schedule (2 frozen + 10 joint)."""
    freeze = max(0, min(int(freeze), int(total)))
    return (freeze * float(frozen_epoch) + (total - freeze) * float(joint_epoch)) / int(total)


def estimates_from_throughput(path) -> dict:
    """Turn a ``throughput.json`` into per-epoch estimates for the matrix plan."""
    payload = read_json(path)
    paths = payload.get("paths") or {}
    perception = paths.get("perception") or {}
    frozen = paths.get("policy_frozen") or {}
    joint = paths.get("policy_joint") or {}
    if not perception or not frozen or not joint:
        raise v104_error("EVIDENCE", f"{path}: three throughput paths are required")
    state = {1: FALLBACK_EPOCH_SECONDS["state"][1], 3: FALLBACK_EPOCH_SECONDS["state"][3]}
    policy = {1: FALLBACK_EPOCH_SECONDS["policy"][1], 3: FALLBACK_EPOCH_SECONDS["policy"][3]}
    depth = int(payload.get("depth", 3))
    state_epoch = float(perception["forecast_epoch_seconds"])
    frozen_epoch = float(frozen["forecast_epoch_seconds"])
    joint_epoch = float(joint["forecast_epoch_seconds"])
    policy_epoch = policy_mixed_epoch_seconds(frozen_epoch, joint_epoch)
    state[depth] = state_epoch
    policy[depth] = policy_epoch
    extra = paths.get("perception_depth1_extra") or {}
    if extra.get("forecast_epoch_seconds") is not None:
        state[1] = float(extra["forecast_epoch_seconds"])
    per_key = {f"state.{level}": ("measured" if level == depth else "fallback") for level in (1, 3)}
    per_key.update({f"policy.{level}": ("measured" if level == depth else "fallback") for level in (1, 3)})
    if extra:
        per_key["state.1"] = "measured"
    return {
        "state": state,
        "policy": policy,
        "source": f"measured: {rel(path)}",
        "source_by_key": per_key,
        "measured": {
            "depth": depth,
            "state_epoch_seconds": state_epoch,
            "policy_frozen_epoch_seconds": frozen_epoch,
            "policy_joint_epoch_seconds": joint_epoch,
            "policy_mixed_epoch_seconds": policy_epoch,
        },
    }


def json_safe(value):
    """Recursively turn ``dict`` keys into strings and drop nothing else."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return rel(value)
    return value


# --------------------------------------------------------------------------- #
# dataset / checkpoint helpers
# --------------------------------------------------------------------------- #
def load_split(dataset_dir=DATASET_A):
    """``(manifest, manifest_path, train_indices, val_indices)`` for A."""
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise v104_error("DATA", f"no manifest at {manifest_path}")
    manifest = DatasetManifest.load(manifest_path)
    train, val = manifest.verify(dataset_dir)
    train = list(train)
    val = list(val)
    digest = index_digest(val)
    if digest != VAL_INDEX_SHA256:
        raise v104_error("COVERAGE", f"validation split digest {digest} != the frozen {VAL_INDEX_SHA256}")
    if len(val) != VAL_FRAMES:
        raise v104_error("COVERAGE", f"validation split has {len(val)} frames, expected {VAL_FRAMES}")
    return manifest, manifest_path, train, val


def teacher_cache(manifest, manifest_path, val_sha: str = VAL_INDEX_SHA256, dataset_dir=DATASET_A) -> dict:
    """The fingerprinted v104 teacher cache (never the unfingerprinted v1.0.3 one)."""
    cfg = vision_train.VisionTrainConfig(
        dataset=Path(dataset_dir),
        out_dir=ROOT / "third_party" / "th08" / "build" / "rl" / "runs" / "_s34_scratch",
        selection_contract="v104",
        history_contract=vision_history.HISTORY_CAUSAL_SEGMENT_V1,
    )
    with VisionDataset(Path(dataset_dir), history_contract=vision_history.HISTORY_CAUSAL_SEGMENT_V1) as dataset:
        cache = vision_train.load_or_build_teacher_values(
            cfg, dataset, manifest=manifest, val_index_sha256=val_sha, verbose=False
        )
    if cache["fingerprint"] != TEACHER_FINGERPRINT:
        raise v104_error(
            "TEACHER", f"teacher fingerprint {cache['fingerprint']} != the bound {TEACHER_FINGERPRINT}"
        )
    return cache


def load_checkpoint(path, device: str) -> dict:
    """Strict checkpoint load plus the identity this tool reports."""
    path = Path(path)
    if not path.is_file():
        raise v104_error("CHECKPOINT", f"checkpoint not found: {path}")
    model, extra = load_vision_checkpoint(path, map_location="cpu")
    extra = extra or {}
    arch = vision_arch(model)
    depth = int(getattr(model.config, "depth", 0) or 0)
    if depth < 1:
        raise v104_error("CHECKPOINT", f"{path}: config.depth {depth!r} is not positive")
    contract = extra.get("history_contract")
    history = (
        vision_history.normalize_history_contract(contract)
        if contract is not None
        else vision_history.HISTORY_LEGACY
    )
    sizes = {name: tuple(int(v) for v in size) for name, size in model.config.region_sizes.items()}
    model.to(device).eval()
    return {
        "path": path,
        "sha256": sha256_file(path),
        "model": model,
        "extra": extra,
        "arch": arch,
        "depth": depth,
        "history_contract": history,
        "legacy_history": contract is None,
        "region_sizes": sizes,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def action_tensor(result, *, where: str = "model.act") -> torch.Tensor:
    """The ``[N, 4]`` action tensor from a model's ``act`` return value.

    v1 and v2 ``act`` return the four-tuple ``(actions, logp, value, extra)``
    (設計 C1), so taking ``.shape`` of the return value is a ``tuple`` error - the
    first item is the action tensor.  A bare tensor is accepted too, because the
    contract of ``act`` is "the action comes first", not "it is always a tuple".
    """
    value = result
    if isinstance(value, (tuple, list)):
        if not value:
            raise v104_error("CHECKPOINT", f"{where} returned an empty tuple")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise v104_error(
            "CHECKPOINT", f"{where} returned {type(value).__name__}, expected a tensor of actions"
        )
    return value


def act_contract(result) -> str:
    """``tuple4`` / ``tensor``: what ``act`` returned, for the evidence record."""
    return "tuple4" if isinstance(result, (tuple, list)) else "tensor"


def batch_tensors(dataset: VisionDataset, chunk, depth: int, sizes, device: str):
    """The one preprocessing path: stack_indices -> region_frames -> regions_to_batch.

    At depth > 1 the distinct frames of the stack are decoded once and gathered
    back, mirroring the trainer's ``_gather`` (see the comment below).
    """
    requested = dataset.stack_indices(list(chunk), int(depth)).reshape(-1)
    if int(depth) == 1:
        frames = dataset.region_frames(requested.tolist(), sizes)
    else:
        # The trainer's ``_gather`` decodes each distinct frame once and gathers it
        # back: a 96-sample batch at depth 3 asks for 288 slots but touches ~100
        # distinct frames.  It is a private helper of another owner's module, so
        # the same rule is implemented here (value-identical, pinned by
        # ``test_gather_dedup_is_value_identical``); measuring the un-deduplicated
        # path would overstate an epoch by ~40% and make the 60 s comparison wrong.
        unique, inverse = np.unique(requested, return_inverse=True)
        frames = dataset.region_frames(unique.tolist(), sizes)
    stacked = {}
    for name, array in frames.items():
        if int(depth) == 1:
            stacked[name] = array.reshape(
                len(chunk), int(depth), sizes[name][0], sizes[name][1], vision.VISION_CHANNELS
            )
        else:
            stacked[name] = array[inverse].reshape(
                len(chunk), int(depth), sizes[name][0], sizes[name][1], vision.VISION_CHANNELS
            )
    return regions_to_batch(stacked, device)


def evaluate_state(checkpoint, val_indices, *, device="cuda", dataset_dir=DATASET_A, batch_size=BATCH_SIZE) -> dict:
    """Position / danger re-evaluation of a checkpoint over the whole A split."""
    loaded = load_checkpoint(checkpoint, device)
    started = time.perf_counter()
    dataset = VisionDataset(Path(dataset_dir), history_contract=loaded["history_contract"])
    try:
        preds, pred_danger, truths, true_danger, heatmaps, visited = [], [], [], [], [], []
        for start in range(0, len(val_indices), batch_size):
            chunk = [int(index) for index in val_indices[start : start + batch_size]]
            visited.extend(chunk)
            labels = dataset.labels(chunk)
            batch = batch_tensors(dataset, chunk, loaded["depth"], loaded["region_sizes"], device)
            with torch.no_grad():
                out = loaded["model"].decode_state(batch)
            preds.append(out["self_feat"].detach().cpu().numpy())
            pred_danger.append(out["danger_log"].detach().cpu().numpy())
            truths.append(np.asarray(labels["self_feat"], dtype=np.float64))
            true_danger.append(np.asarray(labels["danger"], dtype=np.float64))
            if "heatmap_logits" in out:
                heatmaps.append(out["heatmap_logits"].detach().cpu().numpy())
        metrics = vision_metrics.all_state_metrics(
            np.concatenate(preds),
            np.concatenate(pred_danger),
            np.concatenate(truths),
            np.concatenate(true_danger),
            heatmap_logits=np.concatenate(heatmaps) if heatmaps else None,
        )
        coverage = vision_train.coverage_record([int(index) for index in val_indices], visited)
    finally:
        dataset.close()
    return {
        "checkpoint": rel(loaded["path"]),
        "checkpoint_sha256": loaded["sha256"],
        "arch": loaded["arch"],
        "depth": loaded["depth"],
        "history_contract": loaded["history_contract"],
        "legacy_history": loaded["legacy_history"],
        "frames": len(visited),
        "coverage": coverage,
        "metrics": metrics,
        "seconds": round(time.perf_counter() - started, 2),
    }


def evaluate_policy(checkpoint, val_indices, teacher_values, *, device="cuda", dataset_dir=DATASET_A, batch_size=BATCH_SIZE) -> dict:
    """Full teacher-agreement re-evaluation (regret / buckets / top3) over A."""
    loaded = load_checkpoint(checkpoint, device)
    started = time.perf_counter()
    dataset = VisionDataset(Path(dataset_dir), history_contract=loaded["history_contract"])
    try:
        scores, truths, bullets = [], [], []
        for start in range(0, len(val_indices), batch_size):
            chunk = [int(index) for index in val_indices[start : start + batch_size]]
            labels = dataset.labels(chunk)
            batch = batch_tensors(dataset, chunk, loaded["depth"], loaded["region_sizes"], device)
            with torch.no_grad():
                predicted = loaded["model"].score(batch)
            scores.append(predicted.detach().cpu().numpy())
            truths.append(np.asarray(teacher_values[chunk], dtype=np.float32))
            bullets.append(np.asarray(labels["bullets"], dtype=np.float32))
        scores = np.concatenate(scores)
        metrics = vision_metrics.policy_metrics(scores, np.concatenate(truths), np.concatenate(bullets))
    finally:
        dataset.close()
    return {
        "checkpoint": rel(loaded["path"]),
        "checkpoint_sha256": loaded["sha256"],
        "arch": loaded["arch"],
        "depth": loaded["depth"],
        "frames": int(len(val_indices)),
        "metrics": metrics,
        "score_std": float(scores.std(axis=1).mean()),
        "seconds": round(time.perf_counter() - started, 2),
    }


def bucket_metric(metrics: dict, label: str = "0-15") -> dict | None:
    for entry in metrics.get("by_danger") or []:
        if entry.get("bucket") == label:
            return entry
    return None


def nearest_bullet_px(bullets: np.ndarray) -> np.ndarray:
    """Nearest valid bullet distance in px (identical formula to ``vision_metrics``).

    ``dx*192`` / ``dy*224`` in float64, ``inf`` when a frame has no valid slot.
    """
    frames = bullets.shape[0]
    if bullets.shape[1] == 0:
        return np.full(frames, np.inf, dtype=np.float64)
    dx = bullets[:, :, 0].astype(np.float64) * 192.0
    dy = bullets[:, :, 1].astype(np.float64) * 224.0
    valid = bullets[:, :, 5] > 0.5
    with np.errstate(invalid="ignore"):
        distance = np.sqrt(dx * dx + dy * dy)
    return np.where(valid, distance, np.inf).min(axis=1)


def model_for_depth(depth: int, device: str, *, seed: int = SEED) -> torch.nn.Module:
    """Fresh v2 model (the same geometry the trainer builds for ``--arch v2``)."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    return VisionScorerV2(VisionNetConfigV2(depth=int(depth), heatmap_gauss_sigma=HEATMAP_SIGMA)).to(device)


def train_config_for(experiment: str, device: str = "cuda") -> vision_train.VisionTrainConfig:
    spec = EXPERIMENTS[experiment]
    return vision_train.VisionTrainConfig(
        dataset=DATASET_A,
        out_dir=ROOT / "third_party" / "th08" / "build" / "rl" / "runs" / "_s34_scratch",
        arch="v2",
        stage="policy",
        epochs_state=0,
        epochs_policy=POLICY_EPOCHS,
        batch_size=BATCH_SIZE,
        depth=spec["depth"],
        seed=SEED,
        device=device,
        heatmap_sigma=HEATMAP_SIGMA,
        selection_contract="v104",
        history_contract=vision_history.HISTORY_CAUSAL_SEGMENT_V1,
        policy_objective=spec["objective"],
        rank_weight=spec["rank_weight"],
        regression_weight=spec["regression_weight"],
        policy_freeze_epochs=POLICY_FREEZE_EPOCHS,
    )


# --------------------------------------------------------------------------- #
# phase: baseline
# --------------------------------------------------------------------------- #
def phase_baseline(args) -> int:
    out = Path(args.out)
    primary = out / "baseline.json"
    refuse_existing_output(primary)
    old_state = RUNS_ROOT / OLD_STATE_RUN / "state.pt"
    old_policy_run = RUNS_ROOT / OLD_POLICY_RUN
    old_policy = old_policy_run / "policy.pt"
    old_state_summary = RUNS_ROOT / OLD_STATE_RUN / "vision-summary.json"
    old_policy_summary = old_policy_run / "vision-summary.json"
    print("[baseline] re-evaluate the frozen baselines on the fixed A split (read-only)")
    for path in (old_state, old_policy, old_state_summary, old_policy_summary):
        if not Path(path).is_file():
            raise v104_error("EVIDENCE", f"frozen baseline input missing: {rel(path)}")
        print(f"[baseline]   input {rel(path)}")
        print(f"[baseline]     sha256={sha256_file(path)}")
    if not args.execute:
        print("[baseline] dry run: pass --execute to re-evaluate (no run directory is created)")
        return 0

    device = pick_device(args.device)
    manifest, manifest_path, _train, val = load_split()
    cache = teacher_cache(manifest, manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    state_recheck = evaluate_state(old_state, val, device=device)
    policy_recheck = evaluate_policy(old_policy, val, cache["values"], device=device)
    old_state_payload = read_json(old_state_summary)
    old_policy_payload = read_json(old_policy_summary)
    state_best = (old_state_payload.get("stages") or {}).get("state", {}).get("best") or {}
    policy_best = (old_policy_payload.get("stages") or {}).get("policy", {}).get("best") or {}
    old_state_frames = int(state_best.get("val_frames") or 0)
    old_policy_frames = int(policy_best.get("val_frames") or 0)
    old_policy_bucket = bucket_metric(policy_best, "0-15") or {}

    payload = {
        "schema": PHASE_SCHEMA,
        "phase": "baseline",
        "status": "done",
        "created_at": now_iso(),
        "command": command_line([sys.executable, *sys.argv]),
        "cwd": str(ROOT),
        "execute": True,
        "dataset": rel(DATASET_A),
        "manifest": rel(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "val_index_sha256": VAL_INDEX_SHA256,
        "val_frames": len(val),
        "teacher": {
            "fingerprint": cache["fingerprint"],
            "cache": cache["path"],
            "cache_sha256": cache["cache_sha256"],
        },
        "device": device,
        "env": env_block(),
        "old": {
            "state": {
                "run": rel(RUNS_ROOT / OLD_STATE_RUN),
                "checkpoint": rel(old_state),
                "checkpoint_sha256": sha256_file(old_state),
                "summary": rel(old_state_summary),
                "summary_sha256": sha256_file(old_state_summary),
                "summary_best": state_best,
                "summary_val_frames": old_state_frames,
            },
            "policy": {
                "run": rel(old_policy_run),
                "checkpoint": rel(old_policy),
                "checkpoint_sha256": sha256_file(old_policy),
                "summary": rel(old_policy_summary),
                "summary_sha256": sha256_file(old_policy_summary),
                "summary_best": policy_best,
                "summary_val_frames": old_policy_frames,
            },
        },
        "recheck": {"state": state_recheck, "policy": policy_recheck},
        "comparison": {
            "state": {
                "old_summary_7200": {
                    "val_frames": old_state_frames,
                    "val_player_pos_px": state_best.get("val_player_pos_px"),
                    "val_player_pos_p95_px": state_best.get("val_player_pos_p95_px"),
                    "val_danger_corr_perframe_median": state_best.get("val_danger_corr_perframe_median"),
                    "val_danger_corr_perframe_valid": state_best.get("val_danger_corr_perframe_valid"),
                    "epoch": state_best.get("epoch"),
                },
                "recheck_7241": {
                    "frames": state_recheck["frames"],
                    "val_player_pos_px": state_recheck["metrics"].get("player_pos_px"),
                    "val_player_pos_p95_px": state_recheck["metrics"].get("player_pos_p95_px"),
                    "val_danger_corr_perframe_median": state_recheck["metrics"].get("danger_corr_perframe_median"),
                    "val_danger_corr_perframe_valid": state_recheck["metrics"].get("danger_corr_perframe_valid"),
                },
            },
            "policy": {
                "old_summary_7200": {
                    "val_frames": old_policy_frames,
                    "val_regret_px": policy_best.get("val_regret_px"),
                    "val_regret_px_bucket_0_15": old_policy_bucket.get("regret_px"),
                    "val_top1": policy_best.get("val_top1"),
                    "by_danger": policy_best.get("by_danger"),
                    "epoch": policy_best.get("epoch"),
                },
                "recheck_7241": {
                    "frames": policy_recheck["frames"],
                    "val_regret_px": policy_recheck["metrics"].get("regret_px"),
                    "val_regret_px_bucket_0_15": (bucket_metric(policy_recheck["metrics"], "0-15") or {}).get("regret_px"),
                    "val_top1": policy_recheck["metrics"].get("top1"),
                    "by_danger": policy_recheck["metrics"].get("by_danger"),
                },
            },
        },
        "frames_note": (
            "The frozen summaries' per-epoch rows report val_frames=7200 (the validation slice "
            "that existed when they were trained); this re-evaluation uses the fixed A validation "
            "split of 7241 frames.  7200 != 7241: both columns are listed side by side and must "
            "not be mixed.  The old summaries were read, never written."
        ),
        "new_E0": {
            "status": "pending",
            "note": "E0 is a new run of this stage (runs/v104-E0-s7); it is listed separately from "
            "this read-only re-evaluation and cannot stand in for the old baseline.",
            "source": "matrix phase / experiment-table.json",
            "row": _table_row_for("E0"),
        },
        "seconds": round(time.perf_counter() - started, 2),
        "overwrote_frozen_runs": False,
    }
    write_json(primary, payload)
    report = _baseline_report(payload)
    (out / "baseline.log").write_text(report, encoding="utf-8")
    print(report)
    print(f"[baseline] wrote {rel(primary)}")
    return 0


def _baseline_report(payload: dict) -> str:
    state = payload["comparison"]["state"]
    policy = payload["comparison"]["policy"]
    lines = [
        "baseline re-evaluation (read-only) vs the frozen summaries",
        f"  dataset            : {payload['dataset']}",
        f"  validation frames  : {payload['val_frames']}  sha256={payload['val_index_sha256']}",
        f"  old state ckpt     : {payload['old']['state']['checkpoint']}",
        f"                       sha256={payload['old']['state']['checkpoint_sha256']}",
        f"  old policy ckpt    : {payload['old']['policy']['checkpoint']}",
        f"                       sha256={payload['old']['policy']['checkpoint_sha256']}",
        "",
        "  state stage",
        f"    old summary (7200 frames): pos={state['old_summary_7200']['val_player_pos_px']} px  "
        f"danger_median={state['old_summary_7200']['val_danger_corr_perframe_median']}",
        f"    recheck     (7241 frames): pos={state['recheck_7241']['val_player_pos_px']} px  "
        f"danger_median={state['recheck_7241']['val_danger_corr_perframe_median']}",
        "  policy stage",
        f"    old summary (7200 frames): regret={policy['old_summary_7200']['val_regret_px']} px  "
        f"0-15={policy['old_summary_7200']['val_regret_px_bucket_0_15']} px  top1={policy['old_summary_7200']['val_top1']}",
        f"    recheck     (7241 frames): regret={policy['recheck_7241']['val_regret_px']} px  "
        f"0-15={policy['recheck_7241']['val_regret_px_bucket_0_15']} px  top1={policy['recheck_7241']['val_top1']}",
        "",
        "  policy buckets (7241-frame recheck)",
    ]
    for row in policy["recheck_7241"].get("by_danger") or []:
        lines.append(
            f"    {row['bucket']:>7}  frames={row['frames']:>4}  regret={row['regret_px']} px  top3={row['top3']}"
        )
    lines.extend(["", "  note: " + payload["frames_note"], ""])
    return "\n".join(lines)


def _table_row_for(experiment: str, *, screening=False) -> dict | None:
    path = S34_DIR / "experiment-table.json"
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except ValueError:
        return None
    for row in payload.get("rows") or []:
        if row.get("experiment") == experiment and bool(row.get("screening")) == bool(screening):
            return row
    return None


# --------------------------------------------------------------------------- #
# phase: overfit
# --------------------------------------------------------------------------- #
def stratified_center_frames(dataset: VisionDataset, train_indices, *, target=OVERFIT_TARGET_FRAMES, seed=SEED) -> dict:
    """Fixed ``stage x danger-bucket`` stratified centre frames from the A train split.

    Only training indices are eligible (a validation frame here would turn the
    probe into a generalisation claim).  Every stratum is sampled without
    replacement from a single ``numpy`` generator seeded with 7, in stratum order.
    """
    pools: dict[tuple[int, str], list[int]] = {}
    for start in range(0, len(train_indices), 8192):
        chunk = [int(index) for index in train_indices[start : start + 8192]]
        labels = dataset.labels(chunk)
        distances = nearest_bullet_px(np.asarray(labels["bullets"], dtype=np.float32))
        stages = np.asarray(labels["meta"], dtype=np.float64)[:, 1].astype(np.int64)
        for index, stage, distance in zip(chunk, stages.tolist(), distances.tolist()):
            pools.setdefault((int(stage), assign_bucket(distance)), []).append(index)
    keys = sorted(pools)
    sizes = [len(pools[key]) for key in keys]
    split = apportion_256(sizes, target)
    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    detail = []
    for position, key in enumerate(keys):
        want = int(split["allocation"][position])
        pool = np.asarray(pools[key], dtype=np.int64)
        if want:
            picked = np.sort(rng.choice(pool, size=want, replace=False)).astype(int).tolist()
        else:
            picked = []
        chosen.extend(picked)
        detail.append(
            {
                "stage": int(key[0]),
                "bucket": key[1],
                "available": int(pool.size),
                "allocated": want,
                "selected": len(picked),
                "indices": picked,
            }
        )
    chosen = sorted(chosen)
    if len(chosen) != len(set(chosen)):
        raise v104_error("DATA", "stratified sampling produced duplicate centre frames")
    if len(chosen) != int(target):
        raise v104_error("DATA", f"stratified sampling selected {len(chosen)} frames, expected {int(target)}")
    if set(chosen) - set(int(index) for index in train_indices):
        raise v104_error("DATA", "the overfit sample contains indices outside the training split")
    return {
        "target_frames": int(target),
        "frames": len(chosen),
        "indices": chosen,
        "indices_sha256": index_digest(chosen),
        "seed": int(seed),
        "replacement": False,
        "strata": detail,
        "unfilled": int(split["unfilled"]),
        "source_split": "train",
    }


def score_frames(model, dataset, indices, depth, sizes, device, batch=96) -> np.ndarray:
    """Policy scores ``[N, 18]`` for fixed indices (eval mode, no grad)."""
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            chunk = [int(index) for index in indices[start : start + batch]]
            scores.append(model.score(batch_tensors(dataset, chunk, depth, sizes, device)).detach().cpu().numpy())
    return np.concatenate(scores)


def regret_of(scores: np.ndarray, teacher: np.ndarray, bullets: np.ndarray, label: str | None = None) -> dict:
    """Mean raw teacher regret, optionally restricted to one danger bucket."""
    metrics = vision_metrics.policy_metrics(scores, teacher, bullets)
    if label is None:
        return {"regret_px": metrics["regret_px"], "frames": int(scores.shape[0]), "top1": metrics["top1"]}
    bucket = bucket_metric(metrics, label)
    if bucket is None:
        raise v104_error("DATA", f"no danger bucket {label!r} in the metric output")
    return {"regret_px": bucket["regret_px"], "frames": int(bucket["frames"]), "top1": bucket["top1"]}


def eval_overfit(model, dataset, indices, teacher, bullets, depth, sizes, device) -> dict:
    """Score the fixed 256 frames; report global and 0-15 regret plus score std."""
    scores = score_frames(model, dataset, indices, depth, sizes, device)
    return {
        "global": regret_of(scores, teacher, bullets),
        "bucket_0_15": regret_of(scores, teacher, bullets, "0-15"),
        "score_std": float(scores.std(axis=1).mean()),
    }


def phase_overfit(args) -> int:
    experiment = args.experiment[0] if args.experiment else None
    if not experiment:
        raise v104_error("CONFIG", "--phase overfit requires --experiment E0|E1|E2|E3a|E3b|E3c")
    if args.experiment and len(args.experiment) > 1:
        raise v104_error("CONFIG", "--phase overfit takes exactly one --experiment")
    spec = EXPERIMENTS[experiment]
    depth = spec["depth"]
    steps_limit = int(args.overfit_steps)
    if steps_limit < 1 or steps_limit > OVERFIT_MAX_STEPS:
        raise v104_error("CONFIG", f"--overfit-steps must be in 1..{OVERFIT_MAX_STEPS}, got {steps_limit}")
    out = Path(args.out)
    primary = out / "overfit.json"
    refuse_existing_output(primary)
    print(f"[overfit] {experiment}: depth {depth}, objective {spec['objective']}, <= {steps_limit} steps")
    if not args.execute:
        print("[overfit] dry run: pass --execute to sample 256 frames and train")
        return 0

    device = pick_device(args.device)
    if device == "cuda":
        torch.set_num_threads(1)
    manifest, manifest_path, train, val = load_split()
    cache = teacher_cache(manifest, manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dataset = VisionDataset(DATASET_A, history_contract=vision_history.HISTORY_CAUSAL_SEGMENT_V1)
    try:
        selection = stratified_center_frames(dataset, train)
        if set(selection["indices"]) & set(val):
            raise v104_error("DATA", "the overfit sample contains validation indices")
        indices = selection["indices"]
        labels = dataset.labels(indices)
        teacher = np.asarray(cache["values"][indices], dtype=np.float32)
        bullets = np.asarray(labels["bullets"], dtype=np.float32)
        model = model_for_depth(depth, device)
        cfg = train_config_for(experiment, device)
        optimizer, optim_record = vision_train.build_policy_optimizer(model, cfg, frozen=False)
        sizes = {name: tuple(int(v) for v in size) for name, size in model.config.region_sizes.items()}
        initial = eval_overfit(model, dataset, indices, teacher, bullets, depth, sizes, device)
        initial_regret = float(initial["global"]["regret_px"])
        initial_bucket = (
            float(initial["bucket_0_15"]["regret_px"]) if initial["bucket_0_15"]["frames"] else None
        )
        entry = overfit_entry_status(initial_regret, None)
        step_log: list[dict] = []
        rng = np.random.default_rng(SEED)
        cycles = 0
        order = np.arange(len(indices))
        per_cycle = max(1, math.ceil(len(indices) / BATCH_SIZE))
        if entry["status"] != "blocked":
            model.train()
            for step in range(1, steps_limit + 1):
                position = (step - 1) % per_cycle
                if position == 0:
                    order = rng.permutation(len(indices))
                    cycles += 1
                chunk_positions = order[position * BATCH_SIZE : (position + 1) * BATCH_SIZE]
                chunk = [indices[int(index)] for index in chunk_positions]
                teacher_px = torch.as_tensor(teacher[chunk_positions].astype(np.float32), device=device)
                batch = batch_tensors(dataset, chunk, depth, sizes, device)
                scores, _aux = vision_train.policy_forward(model, batch)
                loss, stats = vision_losses.policy_loss(
                    scores,
                    teacher_px,
                    spec["objective"],
                    0.5,
                    spec["rank_weight"],
                    spec["regression_weight"],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = 0.0
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        grad_norm += float(parameter.grad.detach().pow(2).sum())
                grad_norm = math.sqrt(grad_norm)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                row = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "rank_loss": stats["rank_loss"],
                    "regression_loss": stats["regression_loss"],
                    "valid_pairs": stats["valid_pairs"],
                    "frames_with_pairs": stats["frames_with_pairs"],
                    "frames_without_pairs": stats["frames_without_pairs"],
                    "score_std": float(scores.detach().std(dim=1).mean()),
                    "grad_norm": grad_norm,
                }
                step_log.append(row)
                if step == 1 or step % 50 == 0 or step == steps_limit:
                    measured = eval_overfit(model, dataset, indices, teacher, bullets, depth, sizes, device)
                    model.train()
                    row["eval"] = measured
                    print(
                        f"[overfit] step {step:>4} loss={row['loss']:.6f} "
                        f"regret={measured['global']['regret_px']:.4f} "
                        f"bucket0-15={measured['bucket_0_15']['regret_px']:.4f} "
                        f"pairs={stats['valid_pairs']} grad={grad_norm:.4f}"
                    )
        final = eval_overfit(model, dataset, indices, teacher, bullets, depth, sizes, device)
        entry = overfit_entry_status(initial_regret, float(final["global"]["regret_px"]))
        entry["bucket_0_15"] = overfit_entry_status(
            initial_bucket if initial_bucket is not None else float("nan"),
            float(final["bucket_0_15"]["regret_px"]) if final["bucket_0_15"]["frames"] else float("nan"),
        )
        gradients = [row["grad_norm"] for row in step_log]
        payload = {
            "schema": PHASE_SCHEMA,
            "phase": "overfit",
            "status": "done",
            "created_at": now_iso(),
            "command": command_line([sys.executable, *sys.argv]),
            "cwd": str(ROOT),
            "execute": True,
            "experiment": experiment,
            "depth": depth,
            "objective": spec["objective"],
            "rank_weight": spec["rank_weight"],
            "regression_weight": spec["regression_weight"],
            "dataset": rel(DATASET_A),
            "manifest": rel(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "val_index_sha256": VAL_INDEX_SHA256,
            "teacher": {"fingerprint": cache["fingerprint"], "cache_sha256": cache["cache_sha256"]},
            "history_contract": vision_history.HISTORY_CAUSAL_SEGMENT_V1,
            "initialisation": {
                "source": "fresh",
                "seed": SEED,
                "note": "The depth-3 state checkpoint is produced by the matrix phase, which runs "
                "after this probe; the G-v3 small-set criterion is a delta from the "
                "initialisation, so a deterministic fresh seed-7 initialisation is used and "
                "recorded here.  No cross-depth weights are transplanted (design C4).",
                "source_state_sha": None,
            },
            "optimizer": optim_record,
            "selection": selection,
            "steps_limit": steps_limit,
            "steps_run": len(step_log),
            "cycles": cycles,
            "batch_size": BATCH_SIZE,
            "drop_last": False,
            "aux_losses_used": False,
            "aux_note": "This probe trains the policy objective alone so the ranking signal is not "
            "masked by the auxiliary heads; the formal policy stage adds hm=.1/xy=1/"
            "danger=.2/self=.1 and is measured by the throughput phase.",
            "initial": initial,
            "final": final,
            "entry": entry,
            "steps": step_log,
            "gradients": {
                "finite": all(math.isfinite(value) for value in gradients),
                "nonzero": bool(gradients) and max(gradients) > 0.0,
                "first": gradients[0] if gradients else None,
                "last": gradients[-1] if gradients else None,
                "max": max(gradients) if gradients else None,
            },
            "device": device,
            "env": env_block(),
            "seconds": round(time.perf_counter() - started, 2),
        }
        write_json(primary, payload)
        lines = [
            f"overfit probe {experiment} (depth {depth}, {spec['objective']}, "
            f"rank={spec['rank_weight']}, reg={spec['regression_weight']})",
            f"  frames         : {payload['selection']['frames']} (train split only) "
            f"sha256={payload['selection']['indices_sha256']}",
            f"  steps          : {payload['steps_run']} (limit {steps_limit})",
            f"  initial regret : {payload['initial']['global']['regret_px']} px "
            f"(0-15 {payload['initial']['bucket_0_15']['regret_px']} px / "
            f"{payload['initial']['bucket_0_15']['frames']} frames)",
            f"  final regret   : {payload['final']['global']['regret_px']} px "
            f"(0-15 {payload['final']['bucket_0_15']['regret_px']} px / "
            f"{payload['final']['bucket_0_15']['frames']} frames)",
            f"  entry          : {entry['status']} ({entry['reason']}) reduction={entry['reduction']}",
            f"  gradients      : finite={payload['gradients']['finite']} "
            f"nonzero={payload['gradients']['nonzero']} max={payload['gradients']['max']}",
            f"  seconds        : {payload['seconds']}",
        ]
        (out / "overfit.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        print(f"[overfit] wrote {rel(primary)}")
        return 0
    finally:
        dataset.close()


# --------------------------------------------------------------------------- #
# phase: throughput
# --------------------------------------------------------------------------- #
class MeasuredStep:
    """One training step, split into data / forward+loss / backward+step."""

    REGION = "playfield"

    def __init__(self, model, optimizer, *, path: str, cfg, depth: int, device: str):
        self.model = model
        self.optimizer = optimizer
        self.path = path  # perception | policy
        self.cfg = cfg
        self.depth = int(depth)
        self.device = device
        self.mask = vision_train.self_aux_mask(device)
        self.heatmap_hw = tuple(int(v) for v in model.config.heatmap_hw)

    def sync(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _danger_target(self, labels):
        return torch.as_tensor(
            np.log1p(np.asarray(labels["danger"], dtype=np.float32) / vision_train.DANGER_SCALE),
            device=self.device,
        )

    def _heatmap_target(self, labels):
        height, width = self.heatmap_hw
        return torch.as_tensor(
            heatmap_target(
                np.asarray(labels["self_feat"])[:, 0],
                np.asarray(labels["self_feat"])[:, 1],
                height,
                width,
                sigma=float(self.cfg.heatmap_sigma),
            ),
            device=self.device,
        )

    def forward_loss(self, batch, labels):
        frames = int(batch[self.REGION].shape[0])
        truth_self = torch.as_tensor(np.asarray(labels["self_feat"], dtype=np.float32), device=self.device)
        target_danger = self._danger_target(labels).reshape(frames, obs.DANGER_ROWS, obs.DANGER_COLS)
        if self.path == "perception":
            out = self.model.decode_state(batch)
            log_probability = torch.log_softmax(out["heatmap_logits"].flatten(1), dim=-1)
            loss_heatmap = -(self._heatmap_target(labels) * log_probability).sum(-1).mean()
            loss_self = vision_train.masked_self_loss(out["self_feat"], truth_self, self.mask)
            loss_danger = torch.nn.functional.smooth_l1_loss(
                out["danger_log"].reshape(frames, obs.DANGER_ROWS, obs.DANGER_COLS), target_danger
            )
            loss_coord = vision_train.coordinate_loss(out["self_feat"], truth_self)
            return loss_heatmap + loss_self + loss_danger + loss_coord, {}
        scores, aux = vision_train.policy_forward(self.model, batch)
        loss_policy, stats = vision_losses.policy_loss(
            scores,
            torch.as_tensor(np.asarray(labels["teacher"], dtype=np.float32), device=self.device),
            self.cfg.policy_objective,
            self.cfg.rank_delta_px,
            self.cfg.rank_weight,
            self.cfg.regression_weight,
        )
        log_probability = torch.log_softmax(aux["heatmap_logits"].flatten(1), dim=-1)
        loss_heatmap = -(self._heatmap_target(labels) * log_probability).sum(-1).mean()
        loss_coord = vision_train.coordinate_loss(aux["self_feat"], truth_self)
        loss_danger = torch.nn.functional.smooth_l1_loss(
            aux["danger_log"].reshape(frames, obs.DANGER_ROWS, obs.DANGER_COLS), target_danger
        )
        loss_self = vision_train.masked_self_loss(aux["self_feat"], truth_self, self.mask)
        aux_loss = (
            float(self.cfg.policy_aux_heatmap) * loss_heatmap
            + float(self.cfg.policy_aux_coord) * loss_coord
            + float(self.cfg.policy_aux_danger) * loss_danger
            + float(self.cfg.policy_aux_self) * loss_self
        )
        return loss_policy + aux_loss, {"valid_pairs": stats["valid_pairs"]}

    def backward(self, loss) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad], 1.0
        )
        self.optimizer.step()


def measured_range(count: int, *, warmup: int = WARMUP_BATCHES, measure: int | None = None) -> range:
    """Batch indices that count toward a mean: the warm-up prefix is excluded.

    ``measure_path`` runs **two passes** over the same fixed batch cycle: the
    first pass warms the decode/page cache and is discarded, the second pass is
    measured.  Within the measured pass this helper still skips its own leading
    ``warmup`` batch equivalents (the GPU is already hot from the first pass, so
    this only drops the tail of the cycle for training: 110 offered, 10 skipped,
    100 measured).  Validation offers the epoch's own 76 batches, so 10 are
    skipped and 66 are measured - disclosed as
    ``val_batches_measured`` / ``val_batches_epoch``.
    """
    limit = int(count) if measure is None else min(int(count), int(warmup) + int(measure))
    return range(min(int(warmup), int(count)), limit)


def measure_path(
    name: str,
    *,
    experiment: str,
    depth: int,
    device: str,
    dataset: VisionDataset,
    train_batches,
    val_batches,
    teacher_values,
) -> dict:
    """Two passes over a fixed batch cycle: warm-up pass, then the measured pass."""
    spec = EXPERIMENTS[experiment]
    cfg = dataclasses.replace(train_config_for(experiment, device), depth=int(depth))
    model = model_for_depth(depth, device)
    if name == "perception":
        optimizer = torch.optim.AdamW(model.parameters(), lr=PERCEPTION_LR, weight_decay=1e-4)
        step = MeasuredStep(model, optimizer, path="perception", cfg=cfg, depth=depth, device=device)
    else:
        optimizer, _record = vision_train.build_policy_optimizer(
            model, cfg, frozen=(name == "policy_frozen")
        )
        step = MeasuredStep(model, optimizer, path="policy", cfg=cfg, depth=depth, device=device)
    sizes = {region: tuple(int(v) for v in size) for region, size in model.config.region_sizes.items()}

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    def load(chunk):
        labels = dataset.labels(chunk)
        packed = {
            "self_feat": np.asarray(labels["self_feat"], dtype=np.float32),
            "danger": np.asarray(labels["danger"], dtype=np.float32),
            "teacher": np.asarray(teacher_values[chunk], dtype=np.float32),
        }
        return batch_tensors(dataset, chunk, depth, sizes, device), packed

    train_measured = set(measured_range(len(train_batches), measure=MEASURE_BATCHES))
    val_measured = set(measured_range(len(val_batches)))
    for cycle in (0, 1):
        measured_pass = cycle == 1
        model.train()
        for index, chunk in enumerate(train_batches):
            measured = measured_pass and index in train_measured
            step.sync()
            start = time.perf_counter()
            batch, labels = load(chunk)
            step.sync()
            after_data = time.perf_counter()
            loss, _extras = step.forward_loss(batch, labels)
            step.sync()
            after_forward = time.perf_counter()
            step.backward(loss)
            step.sync()
            after_backward = time.perf_counter()
            if measured:
                train_rows.append(
                    {
                        "data": after_data - start,
                        "forward": after_forward - after_data,
                        "backward": after_backward - after_forward,
                        "total": after_backward - start,
                    }
                )
        model.eval()
        for index, chunk in enumerate(val_batches):
            measured = measured_pass and index in val_measured
            step.sync()
            start = time.perf_counter()
            batch, labels = load(chunk)
            step.sync()
            after_data = time.perf_counter()
            with torch.no_grad():
                if name == "perception":
                    model.decode_state(batch)
                else:
                    model.score(batch)
            step.sync()
            after_forward = time.perf_counter()
            if measured:
                val_rows.append(
                    {
                        "data": after_data - start,
                        "forward": after_forward - after_data,
                        "total": after_forward - start,
                    }
                )

    def mean(rows, key):
        return float(np.mean([row[key] for row in rows])) if rows else 0.0

    train_mean = mean(train_rows, "total")
    val_mean = mean(val_rows, "total")
    return {
        "path": name,
        "depth": int(depth),
        "objective": spec["objective"],
        "batch_size": BATCH_SIZE,
        "warmup_batches": WARMUP_BATCHES,
        "warmup_passes": 1,
        "measured_passes": 1,
        "train_batches_measured": len(train_rows),
        "train_batches_available": len(train_batches),
        "val_batches_measured": len(val_rows),
        "val_batches_epoch": VAL_BATCHES,
        "val_batches_available": len(val_batches),
        "measurement_note": "pass 1 over the fixed batch cycle warms the decode/page cache and is "
        "discarded; pass 2 is measured after the leading warm-up prefix.  The cycle is fixed "
        "(seed 7, shard-chunked) so every measured batch is a real training/validation batch.",
        "train": {
            "data_seconds": mean(train_rows, "data"),
            "forward_seconds": mean(train_rows, "forward"),
            "backward_seconds": mean(train_rows, "backward"),
            "total_seconds": train_mean,
            "batches_per_second": (1.0 / train_mean) if train_mean else 0.0,
        },
        "val": {
            "data_seconds": mean(val_rows, "data"),
            "forward_seconds": mean(val_rows, "forward"),
            "total_seconds": val_mean,
            "batches_per_second": (1.0 / val_mean) if val_mean else 0.0,
        },
        "forecast_epoch_seconds": forecast_epoch_seconds(train_mean, val_mean),
        "forecast_formula": f"{TRAIN_BATCHES}*train_total + {VAL_BATCHES}*val_total",
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0,
        },
        "device": device,
    }


def phase_throughput(args) -> int:
    experiment = args.experiment[0] if args.experiment else None
    if not experiment:
        raise v104_error("CONFIG", "--phase throughput requires --experiment E0|E1|E2|E3a|E3b|E3c")
    spec = EXPERIMENTS[experiment]
    depth = spec["depth"]
    out = Path(args.out)
    primary = out / "throughput.json"
    refuse_existing_output(primary)
    print(
        f"[throughput] {experiment}: depth {depth}; two passes over a fixed cycle, "
        f"pass 1 warm-up, pass 2 measured ({MEASURE_BATCHES} training batches + "
        f"the epoch's validation batches)"
    )
    if not args.execute:
        print("[throughput] dry run: pass --execute to time the three paths")
        return 0

    device = pick_device(args.device)
    if device == "cuda":
        torch.set_num_threads(1)
    manifest, manifest_path, train_idx, val_idx = load_split()
    cache = teacher_cache(manifest, manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dataset = VisionDataset(DATASET_A, history_contract=vision_history.HISTORY_CAUSAL_SEGMENT_V1)
    try:
        rng = np.random.default_rng(SEED)
        train_batches = list(
            vision_train.make_batches(
                train_idx,
                batch_size=BATCH_SIZE,
                shuffle=True,
                dataset=dataset,
                rng=rng,
                max_batches=WARMUP_BATCHES + MEASURE_BATCHES,
                drop_last=True,
            )
        )
        val_batches = list(
            vision_train.make_batches(
                val_idx,
                batch_size=BATCH_SIZE,
                shuffle=False,
                dataset=dataset,
                rng=None,
                max_batches=WARMUP_BATCHES + VAL_BATCHES,
                drop_last=False,
            )
        )
        if len(train_batches) < WARMUP_BATCHES + MEASURE_BATCHES:
            raise v104_error("DATA", f"only {len(train_batches)} training batches available")
        paths: dict[str, dict] = {}
        for name in ("perception", "policy_frozen", "policy_joint"):
            print(f"[throughput]   measuring {name} ...", flush=True)
            paths[name] = measure_path(
                name,
                experiment=experiment,
                depth=depth,
                device=device,
                dataset=dataset,
                train_batches=train_batches,
                val_batches=val_batches,
                teacher_values=cache["values"],
            )
            print(
                f"[throughput]   {name}: train "
                f"{paths[name]['train']['total_seconds'] * 1000:.2f} ms/batch, val "
                f"{paths[name]['val']['total_seconds'] * 1000:.2f} ms/batch, forecast "
                f"{paths[name]['forecast_epoch_seconds']:.2f} s/epoch"
            )
        if depth != 1:
            print("[throughput]   measuring perception at depth 1 (supplementary, for the E0 plan) ...", flush=True)
            extra = measure_path(
                "perception",
                experiment=experiment,
                depth=1,
                device=device,
                dataset=dataset,
                train_batches=train_batches,
                val_batches=val_batches,
                teacher_values=cache["values"],
            )
            extra["supplementary"] = True
            extra["note"] = (
                "Extra depth-1 perception measurement so the E0 plan has a same-session "
                "number; not one of the three contract paths."
            )
            paths["perception_depth1_extra"] = extra
            print(
                f"[throughput]   perception_depth1: train "
                f"{extra['train']['total_seconds'] * 1000:.2f} ms/batch, forecast "
                f"{extra['forecast_epoch_seconds']:.2f} s/epoch"
            )
        policy_mixed = policy_mixed_epoch_seconds(
            paths["policy_frozen"]["forecast_epoch_seconds"],
            paths["policy_joint"]["forecast_epoch_seconds"],
        )
        payload = {
            "schema": PHASE_SCHEMA,
            "phase": "throughput",
            "status": "done",
            "created_at": now_iso(),
            "command": command_line([sys.executable, *sys.argv]),
            "cwd": str(ROOT),
            "execute": True,
            "experiment": experiment,
            "depth": depth,
            "objective": spec["objective"],
            "batch_size": BATCH_SIZE,
            "dataset": rel(DATASET_A),
            "manifest": rel(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "val_index_sha256": VAL_INDEX_SHA256,
            "teacher": {"fingerprint": cache["fingerprint"], "cache_sha256": cache["cache_sha256"]},
            "history_contract": vision_history.HISTORY_CAUSAL_SEGMENT_V1,
            "paths": paths,
            "derived": {
                "state_epoch_seconds": paths["perception"]["forecast_epoch_seconds"],
                "policy_frozen_epoch_seconds": paths["policy_frozen"]["forecast_epoch_seconds"],
                "policy_joint_epoch_seconds": paths["policy_joint"]["forecast_epoch_seconds"],
                "policy_mixed_epoch_seconds": policy_mixed,
                "policy_mixed_formula": f"({POLICY_FREEZE_EPOCHS}*frozen + "
                f"{POLICY_EPOCHS - POLICY_FREEZE_EPOCHS}*joint)/{POLICY_EPOCHS}",
                "budget_seconds": BUDGET_SECONDS,
                "state_within_budget": bool(paths["perception"]["forecast_epoch_seconds"] <= BUDGET_SECONDS),
                "policy_joint_within_budget": bool(
                    paths["policy_joint"]["forecast_epoch_seconds"] <= BUDGET_SECONDS
                ),
                "depth1_state_epoch_seconds": (paths.get("perception_depth1_extra") or {}).get(
                    "forecast_epoch_seconds"
                ),
            },
            "device": device,
            "env": env_block(),
            "seconds": round(time.perf_counter() - started, 2),
            "cleanup_note": "Every path builds its own fresh model and optimizer; no checkpoint or run "
            "directory is written, so the shared matrix initialisation cannot be polluted.",
        }
    finally:
        dataset.close()
    write_json(primary, payload)
    lines = [f"throughput {experiment} (depth {depth}, batch {BATCH_SIZE}, two passes: warm-up + measured)"]
    for name, row in payload["paths"].items():
        lines.append(
            f"  {name:<24} train {row['train']['total_seconds'] * 1000:7.2f} ms "
            f"(data {row['train']['data_seconds'] * 1000:6.2f} / fwd "
            f"{row['train']['forward_seconds'] * 1000:6.2f} / bwd "
            f"{row['train']['backward_seconds'] * 1000:6.2f})  "
            f"val {row['val']['total_seconds'] * 1000:7.2f} ms  "
            f"forecast {row['forecast_epoch_seconds']:7.2f} s/epoch  "
            f"peak {row['memory']['peak_allocated_bytes'] / 2**20:8.1f} MiB alloc / "
            f"{row['memory']['peak_reserved_bytes'] / 2**20:8.1f} MiB reserved"
        )
    lines.append(
        f"  policy mixed epoch ({POLICY_FREEZE_EPOCHS} frozen + "
        f"{POLICY_EPOCHS - POLICY_FREEZE_EPOCHS} joint): "
        f"{payload['derived']['policy_mixed_epoch_seconds']:.2f} s"
    )
    lines.append(f"  seconds: {payload['seconds']}")
    (out / "throughput.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[throughput] wrote {rel(primary)}")
    return 0


# --------------------------------------------------------------------------- #
# phase: matrix
# --------------------------------------------------------------------------- #
def run_recipe_problem(summary: dict, plan: dict, *, screening: bool) -> str | None:
    """``None`` when an existing summary is exactly the run this plan wants."""
    if summary.get("schema") != "vision-summary/2":
        return f"schema {summary.get('schema')!r} != vision-summary/2"
    if summary.get("contract_version") != vision_train.CONTRACT_VERSION:
        return f"contract_version {summary.get('contract_version')!r}"
    recipe = summary.get("recipe") or {}
    if recipe.get("selection_contract") != "v104":
        return f"selection_contract {recipe.get('selection_contract')!r}"
    if recipe.get("history_contract") != vision_history.HISTORY_CAUSAL_SEGMENT_V1:
        return f"history_contract {recipe.get('history_contract')!r}"
    if int(recipe.get("depth", -1)) != int(plan["depth"]):
        return f"depth {recipe.get('depth')!r} != {plan['depth']}"
    if int(recipe.get("seed", -1)) != SEED:
        return f"seed {recipe.get('seed')!r}"
    if int(recipe.get("batch_size", -1)) != BATCH_SIZE:
        return f"batch_size {recipe.get('batch_size')!r}"
    if int(recipe.get("max_val_batches", -1)) != 0:
        return f"max_val_batches {recipe.get('max_val_batches')!r} (must be 0)"
    if bool(recipe.get("screening")) != bool(screening):
        return f"screening {recipe.get('screening')!r} does not match the requested mode"
    if int(recipe.get("max_train_batches", -1)) != int(plan["max_train_batches"]):
        return f"max_train_batches {recipe.get('max_train_batches')!r} != {plan['max_train_batches']}"
    if plan["kind"] == "state":
        if recipe.get("stage") != "state":
            return f"stage {recipe.get('stage')!r} != state"
        if int(recipe.get("epochs_state", -1)) != int(plan["epochs"]):
            return f"epochs_state {recipe.get('epochs_state')!r} != {plan['epochs']}"
    else:
        if recipe.get("stage") != "policy":
            return f"stage {recipe.get('stage')!r} != policy"
        if int(recipe.get("epochs_policy", -1)) != int(plan["epochs"]):
            return f"epochs_policy {recipe.get('epochs_policy')!r} != {plan['epochs']}"
        loss = (summary.get("loss") or {}).get("policy") or {}
        if loss.get("objective") != plan["objective"]:
            return f"objective {loss.get('objective')!r} != {plan['objective']}"
        if float(loss.get("rank_weight", -1)) != float(plan["rank_weight"]):
            return f"rank_weight {loss.get('rank_weight')!r}"
        if float(loss.get("regression_weight", -1)) != float(plan["regression_weight"]):
            return f"regression_weight {loss.get('regression_weight')!r}"
    return None


def epoch_seconds_of(summary: dict, kind: str) -> list[float]:
    history = ((summary.get("stages") or {}).get(kind) or {}).get("history") or []
    return [float(row.get("seconds", 0.0)) for row in history]


def run_subprocess(command, log_path: Path, *, prefix: str = "") -> int:
    """Run one training command, streaming into a log file; console lines are trimmed."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(command_line(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            trimmed = line.rstrip("\n")
            if trimmed.startswith("[vision") or "v104:" in trimmed or "Traceback" in trimmed:
                print(f"{prefix}{trimmed[:240]}", flush=True)
        code = process.wait()
        handle.write(f"\n[exit] {code}\n")
    return int(code)


def run_payload(run_dir: Path, plan: dict, *, log_path: Path, exit_code: int, seconds: float, reused: bool) -> dict:
    summary_path = run_dir / "vision-summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    epochs = epoch_seconds_of(summary, plan["kind"])
    block = (summary.get("stages") or {}).get(plan["kind"]) or {}
    best = block.get("best") or {}
    return {
        "experiment": plan["experiment"],
        "kind": plan["kind"],
        "depth": plan["depth"],
        "run_dir": rel(run_dir),
        "command": command_line(plan["command"]),
        "exit_code": int(exit_code),
        "seconds": round(seconds, 1),
        "reused": bool(reused),
        "log": rel(log_path),
        "summary": rel(summary_path),
        "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else "",
        "acceptance_eligible": bool(summary.get("acceptance_eligible")),
        "partial": bool(summary.get("partial")),
        "screening": bool((summary.get("recipe") or {}).get("screening")),
        "recipe": summary.get("recipe") or {},
        "contract_version": summary.get("contract_version"),
        "selection_rule": block.get("selection_rule"),
        "checkpoint": best.get("checkpoint"),
        "checkpoint_sha256": best.get("checkpoint_sha256"),
        "epochs": epochs,
        "max_epoch_seconds": max(epochs) if epochs else 0.0,
        "budget_exceeded": budget_exceeded(epochs),
        "coverage_val": (summary.get("coverage") or {}).get("val") or {},
        "source_state": summary.get("source_state"),
        "teacher": summary.get("teacher") or {},
        "metrics": {
            "state_best": best if plan["kind"] == "state" else None,
            "policy_best": best if plan["kind"] == "policy" else None,
            "by_danger": best.get("by_danger"),
        },
        "hardware": summary.get("hardware") or {},
        "parameters": summary.get("parameters"),
    }


def state_selection(run_dir: Path) -> dict:
    summary_path = run_dir / "vision-summary.json"
    summary = read_json(summary_path)
    best = ((summary.get("stages") or {}).get("state") or {}).get("best") or {}
    checkpoint = best.get("checkpoint")
    if not checkpoint:
        raise v104_error("CHECKPOINT", f"{rel(run_dir)}: state stage selected no checkpoint")
    path = run_dir / checkpoint
    if not path.is_file():
        raise v104_error("CHECKPOINT", f"{rel(path)} does not exist")
    sha = sha256_file(path)
    if best.get("checkpoint_sha256") and best["checkpoint_sha256"] != sha:
        raise v104_error(
            "CHECKPOINT", f"{rel(path)}: summary sha {best['checkpoint_sha256']} != actual {sha}"
        )
    return {
        "checkpoint": checkpoint,
        "path": path,
        "sha256": sha,
        "selection_rule": best.get("selection_rule"),
        "epoch": best.get("epoch"),
        "summary_sha256": sha256_file(summary_path),
    }


def phase_matrix(args) -> int:
    out = Path(args.out)
    primary = out / "matrix.json"
    refuse_existing_output(primary)
    requested = list(args.experiment or [])
    smoke = bool(args.smoke)
    screening = bool(args.screening)
    if smoke and not requested:
        raise v104_error("CONFIG", "--smoke requires an explicit --experiment (it is not a matrix run)")
    if args.throughput_json:
        estimates = estimates_from_throughput(args.throughput_json)
    elif (EVIDENCE_ROOT / "throughput-E3a" / "throughput.json").is_file():
        estimates = estimates_from_throughput(EVIDENCE_ROOT / "throughput-E3a" / "throughput.json")
    else:
        estimates = FALLBACK_EPOCH_SECONDS
    plans = matrix_plan(
        requested or EXPERIMENT_ORDER,
        smoke=smoke,
        smoke_epochs=args.smoke_epochs if args.smoke_epochs is not None else 1,
        smoke_train_batches=args.smoke_train_batches if args.smoke_train_batches is not None else 2,
        screening=screening,
        device=args.device,
        estimates=estimates,
    )
    print_matrix_plan(plans, estimates, smoke=smoke, screening=screening)
    if not args.execute:
        print("[matrix] dry run: no run directory is created.  Add --execute to run.")
        return 0
    if not smoke and not screening:
        for plan in plans:
            if "--max-train-batches" in plan["command"]:
                raise v104_error("CONFIG", "the formal matrix cannot carry a training limit")

    preflight: list[dict] = []
    for plan in plans:
        summary_path = plan["run_dir"] / "vision-summary.json"
        if summary_path.is_file():
            problem = run_recipe_problem(read_json(summary_path), plan, screening=smoke or screening)
            if problem is not None:
                raise v104_error(
                    "EVIDENCE",
                    f"{rel(plan['run_dir'])} already holds a run that is not this plan ({problem});"
                    " refusing to reuse or overwrite it",
                )
            preflight.append({"run_dir": rel(plan["run_dir"]), "reused": True})
        else:
            preflight.append({"run_dir": rel(plan["run_dir"]), "reused": False})
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    payload = {
        "schema": PHASE_SCHEMA,
        "phase": "matrix",
        "status": "running",
        "smoke": bool(smoke),
        "screening": bool(screening),
        "eligible_by_contract": bool(not smoke and not screening),
        "created_at": now_iso(),
        "command": command_line([sys.executable, *sys.argv]),
        "cwd": str(ROOT),
        "device": args.device,
        "estimates": json_safe(estimates),
        "preflight": preflight,
        "plan": [
            {
                "experiment": plan["experiment"],
                "kind": plan["kind"],
                "depth": plan["depth"],
                "epochs": plan["epochs"],
                "run_dir": rel(plan["run_dir"]),
                "estimated_seconds": round(plan["estimated_seconds"], 1),
                "command": command_line(plan["command"]),
            }
            for plan in plans
        ],
        "runs": [],
        "state_sources": {},
        "total_seconds": 0.0,
        "budget_limit_seconds": BUDGET_SECONDS,
    }
    write_json(primary, payload)
    started_all = time.perf_counter()
    state_sources: dict[int, dict] = {}
    val_indices = load_split()[3]
    try:
        for plan in plans:
            run_dir = plan["run_dir"]
            log_path = out / "logs" / f"{run_dir.name}.log"
            command = list(plan["command"])
            if plan["kind"] == "policy":
                source = state_sources.get(int(plan["depth"]))
                if source is None:
                    raise v104_error("CHECKPOINT", f"no state checkpoint for depth {plan['depth']}")
                command = bind_init_from(plan, source["path"])
                plan["command"] = command
            reused = (run_dir / "vision-summary.json").is_file()
            started = time.perf_counter()
            if reused:
                print(f"[matrix] reusing the valid run in {rel(run_dir)}")
                exit_code = 0
            else:
                print(f"[matrix] running {plan['experiment']} -> {rel(run_dir)}")
                print(f"[matrix]   {command_line(command)}")
                exit_code = run_subprocess(command, log_path, prefix="  ")
                if exit_code != 0:
                    raise v104_error(
                        "DATA",
                        f"{plan['experiment']} ({rel(run_dir)}) exited with {exit_code}; see {rel(log_path)}",
                    )
            seconds = time.perf_counter() - started
            row = run_payload(run_dir, plan, log_path=log_path, exit_code=exit_code, seconds=seconds, reused=reused)
            coverage = row["coverage_val"]
            if int(coverage.get("actual_frames", -1)) != VAL_FRAMES or not coverage.get("complete"):
                raise v104_error(
                    "COVERAGE",
                    f"{rel(run_dir)}: validation coverage {coverage.get('actual_frames')} frames,"
                    f" complete={coverage.get('complete')}; the matrix requires all {VAL_FRAMES}",
                )
            if row["partial"]:
                raise v104_error("DATA", f"{rel(run_dir)}: the run stopped partial ({row['summary']})")
            if plan["kind"] == "state":
                selected = state_selection(run_dir)
                state_sources[int(plan["depth"])] = {
                    "run": rel(run_dir),
                    "checkpoint": selected["checkpoint"],
                    "path": selected["path"],
                    "sha256": selected["sha256"],
                    "selection_rule": selected["selection_rule"],
                    "epoch": selected["epoch"],
                    "summary_sha256": selected["summary_sha256"],
                }
                row["selected_state_checkpoint"] = {
                    "checkpoint": selected["checkpoint"],
                    "sha256": selected["sha256"],
                    "selection_rule": selected["selection_rule"],
                }
            else:
                expected = state_sources[int(plan["depth"])]["sha256"]
                actual = (row.get("source_state") or {}).get("sha256")
                if actual != expected:
                    raise v104_error(
                        "CHECKPOINT",
                        f"{rel(run_dir)}: source_state.sha256 {actual} != the state checkpoint {expected}",
                    )
                row["source_state_sha256"] = actual
                print(f"[matrix]   re-evaluating the policy checkpoint's perception on 7241 frames ...")
                row["final_policy_perception"] = evaluate_state(
                    run_dir / (row["checkpoint"] or "policy.pt"), val_indices, device=pick_device(args.device)
                )
            payload["runs"].append(row)
            payload["state_sources"] = {
                str(depth): {key: (rel(value) if key == "path" else value) for key, value in source.items()}
                for depth, source in state_sources.items()
            }
            payload["total_seconds"] = round(time.perf_counter() - started_all, 1)
            write_json(primary, payload)
            print(
                f"[matrix]   done in {seconds:.1f} s; eligible={row['acceptance_eligible']} "
                f"max_epoch={row['max_epoch_seconds']:.1f} s"
            )
    except BaseException:
        payload["status"] = "partial"
        payload["total_seconds"] = round(time.perf_counter() - started_all, 1)
        write_json(primary, payload)
        raise
    payload["status"] = "done"
    payload["finished_at"] = now_iso()
    payload["total_seconds"] = round(time.perf_counter() - started_all, 1)
    payload["budget_summary"] = {
        "limit_seconds": BUDGET_SECONDS,
        "rows_over_limit": [
            row["experiment"] for row in payload["runs"] if row.get("budget_exceeded")
        ],
        "user_ruling": "question list §8 Q5: depth 3 may continue at the measured ~72 s/epoch; "
        "the 60 s ceiling is recorded, not enforced, this round",
    }
    write_json(primary, payload)
    table_path = write_experiment_table(payload, matrix_path=primary, out_dir=out)
    print(f"[matrix] wrote {rel(primary)} and {rel(table_path)}")
    print(f"[matrix] total wall clock {payload['total_seconds'] / 60.0:.1f} min")
    if smoke or screening:
        print("[matrix] SMOKE/SCREENING: acceptance_eligible=false; this is not a formal comparison")
    return 0


def print_matrix_plan(plans, estimates, *, smoke: bool, screening: bool) -> None:
    mode = "smoke" if smoke else ("screening" if screening else "formal")
    print(f"[matrix] {mode} plan, {len(plans)} runs, serial on one GPU")
    print(f"[matrix] epoch estimate source: {estimates.get('source')}")
    per_key = estimates.get("source_by_key") or {}
    if per_key:
        print(
            "[matrix] per-level estimate provenance: "
            + ", ".join(f"{key}={per_key[key]}" for key in sorted(per_key))
        )
    print(f"  {'experiment':<13}{'kind':<8}{'depth':<7}{'epochs':<8}{'s/epoch':>9}{'est. sec':>11}  run directory")
    for plan in plans:
        print(
            f"  {plan['experiment']:<13}{plan['kind']:<8}{plan['depth']:<7}{plan['epochs']:<8}"
            f"{plan['seconds_per_epoch']:>9.2f}{plan['estimated_seconds']:>11.0f}  {rel(plan['run_dir'])}"
        )
    total = plan_total_seconds(plans)
    print(f"  total estimate: {total:.0f} s = {total / 60.0:.1f} min = {total / 3600.0:.2f} h")
    print("[matrix] commands:")
    for plan in plans:
        print(f"  [{plan['experiment']}] {command_line(plan['command'])}")


def experiment_row(run: dict, payload: dict) -> dict:
    is_policy = run["kind"] == "policy"
    best = ((run.get("metrics") or {}).get("policy_best") if is_policy else (run.get("metrics") or {}).get("state_best")) or {}
    by_danger = (run.get("metrics") or {}).get("by_danger") or best.get("by_danger") or []
    source = (payload.get("state_sources") or {}).get(str(run["depth"])) or {}
    name = run["experiment"]
    spec = EXPERIMENTS.get(name)
    timing = epoch_seconds_summary(run.get("epochs"))
    budget = {
        "limit_seconds": BUDGET_SECONDS,
        "max_epoch_seconds": run.get("max_epoch_seconds"),
        "epoch_seconds": run.get("epochs"),
        "cold_start_epochs": timing["cold_start_epochs"],
        "cold_start_seconds": timing["cold_start_seconds"],
        "first_epoch_seconds": timing["first_epoch_seconds"],
        "steady_mean_seconds": timing["steady_mean_seconds"],
        "steady_max_seconds": timing["steady_max_seconds"],
        "exceeded": timing["steady_exceeds_ceiling"],
        "overrun_fraction": timing["steady_overrun_fraction"],
        "source": "measured per-epoch seconds from vision-summary.json",
        "note": "60 s/epoch is the design ceiling (設計 §5.1 excludes the one-time cache/label "
        "build, so epoch 1-2 are listed separately in cold_start_seconds and the comparison "
        "uses the steady epochs).  The user ruled (question list §8 Q5) that depth 3 may "
        "continue at the measured ~72 s/epoch without changing the data path, the "
        "perception rounds, the batch or the depth.  Recorded item by item; not a stop "
        "condition this round, and not a claim that the ceiling was met.",
    }
    return {
        "experiment": name,
        "kind": run["kind"],
        "depth": run["depth"],
        "objective": spec["objective"] if spec else "perception-aux-only",
        "weights": {
            "rank": spec["rank_weight"] if spec else 0.0,
            "regression": spec["regression_weight"] if spec else 0.0,
        },
        "seed": SEED,
        "screening": bool(payload.get("screening")),
        "smoke": bool(payload.get("smoke")),
        "contract": {
            "contract_version": run.get("contract_version"),
            "selection_contract": (run.get("recipe") or {}).get("selection_contract"),
            "history_contract": (run.get("recipe") or {}).get("history_contract"),
        },
        "source_state": {
            "run": source.get("run") if is_policy else None,
            "checkpoint": source.get("checkpoint") if is_policy else None,
            "sha256": run.get("source_state_sha256") if is_policy else None,
            "selection_rule": source.get("selection_rule") if is_policy else None,
            "initialisation": "strict load of the selected state checkpoint"
            if is_policy
            else "fresh seed-7 v2 build (the perception stage has no upstream checkpoint)",
        },
        "data": {
            "dataset": rel(DATASET_A),
            "val_index_sha256": VAL_INDEX_SHA256,
            "val_frames": VAL_FRAMES,
            "train_frames": TRAIN_FRAMES,
            "batch_size": BATCH_SIZE,
            "coverage_val": run.get("coverage_val"),
        },
        "teacher": run.get("teacher"),
        "epochs": {
            "measured": run.get("epochs"),
            "planned": STATE_EPOCHS if run["kind"] == "state" else POLICY_EPOCHS,
            "freeze_epochs": POLICY_FREEZE_EPOCHS if is_policy else 0,
        },
        "budget": budget,
        "eligible": bool(run.get("acceptance_eligible")),
        "checkpoint": {
            "name": run.get("checkpoint"),
            "sha256": run.get("checkpoint_sha256"),
            "selection_rule": run.get("selection_rule"),
        },
        "metrics": {
            "state_best": (run.get("metrics") or {}).get("state_best"),
            "policy_best": (run.get("metrics") or {}).get("policy_best"),
            "by_danger": by_danger,
            "policy_0_15": next(
                (entry for entry in by_danger if entry.get("bucket") == "0-15"), None
            )
            if is_policy
            else None,
        },
        "final_policy_perception": perception_summary(run.get("final_policy_perception")),
        "seconds": run.get("seconds"),
        "memory": (run.get("hardware") or {}).get("cuda_peak_memory_bytes"),
        "parameters": run.get("parameters"),
        "status": "partial" if run.get("partial") else "done",
        "log_paths": [run.get("log"), run.get("summary")],
        "run_dir": run.get("run_dir"),
        "summary_sha256": run.get("summary_sha256"),
        "reused": bool(run.get("reused")),
    }


def perception_summary(recheck: dict | None) -> dict | None:
    if not recheck:
        return None
    metrics = recheck.get("metrics") or {}
    return {
        "checkpoint": recheck.get("checkpoint"),
        "checkpoint_sha256": recheck.get("checkpoint_sha256"),
        "frames": recheck.get("frames"),
        "player_pos_px": metrics.get("player_pos_px"),
        "danger_corr_perframe_median": metrics.get("danger_corr_perframe_median"),
        "danger_corr_perframe_valid": metrics.get("danger_corr_perframe_valid"),
        "a01_pos_pass": bool(metrics.get("player_pos_px", 1e9) < A01_POS_PX),
        "a01_danger_pass": bool(metrics.get("danger_corr_perframe_median", -1e9) > A01_DANGER_MEDIAN),
    }


def write_experiment_table(matrix_payload: dict, *, matrix_path: Path, out_dir: Path) -> Path:
    """Append the matrix rows to ``Stage3/S3-4/experiment-table.json`` (history is kept)."""
    table_path = S34_DIR / "experiment-table.json"
    if table_path.is_file():
        table = read_json(table_path)
        if table.get("schema") != TABLE_SCHEMA:
            raise v104_error("EVIDENCE", f"{rel(table_path)} has an unknown schema")
    else:
        table = {"schema": TABLE_SCHEMA, "created_at": now_iso(), "rows": [], "runs": []}
    existing = {
        (row.get("experiment"), row.get("run_dir"), bool(row.get("smoke")))
        for row in table.get("rows") or []
    }
    for run in matrix_payload["runs"]:
        key = (run.get("experiment"), run.get("run_dir"), bool(matrix_payload.get("smoke")))
        if key in existing:
            continue
        table["rows"].append(experiment_row(run, matrix_payload))
        existing.add(key)
    table.setdefault("runs", []).append(
        {
            "matrix_evidence": rel(matrix_path),
            "created_at": matrix_payload.get("created_at"),
            "finished_at": matrix_payload.get("finished_at"),
            "smoke": bool(matrix_payload.get("smoke")),
            "screening": bool(matrix_payload.get("screening")),
            "total_seconds": matrix_payload.get("total_seconds"),
            "command": matrix_payload.get("command"),
        }
    )
    table["updated_at"] = now_iso()
    write_json(table_path, table)
    copy_path = Path(out_dir) / "experiment-table.json"
    if copy_path.resolve() != table_path.resolve():
        write_json(copy_path, table)
    return table_path


# --------------------------------------------------------------------------- #
# phase: gate
# --------------------------------------------------------------------------- #
def find_evidence(pattern: str, root: Path = EVIDENCE_ROOT) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else None


def parse_pytest_log(path: Path) -> dict:
    """Counts from a pytest output file.

    pytest only prints the counters that are non-zero: a clean run says
    ``554 passed in 18.83s`` with **no** ``failed`` token at all.  Treating a
    missing token as "unknown" turned a green run into ``fail`` (a real bug the
    first gate run hit), so a parsed run defaults ``failed``/``errors`` to 0 and
    the absence of a ``passed`` token stays "not parsed" (blocked).
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    passed = failed = errors = skipped = None
    for count, kind in re.findall(r"(\d+) (passed|failed|error|errors|skipped)", text):
        if kind == "passed":
            passed = int(count)
        elif kind == "failed":
            failed = int(count)
        elif kind in ("error", "errors"):
            errors = int(count)
        elif kind == "skipped":
            skipped = int(count)
    if passed is not None:
        failed = 0 if failed is None else failed
        errors = 0 if errors is None else errors
    return {
        "path": rel(path),
        "sha256": sha256_file(path),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "parsed": passed is not None,
    }


def phase_gate(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    matrix_path = Path(args.matrix) if args.matrix else find_evidence("matrix/matrix.json")
    baseline_path = Path(args.baseline) if args.baseline else find_evidence("baseline/baseline.json")
    throughput_path = Path(args.throughput) if args.throughput else find_evidence("throughput-*/throughput.json")
    table_path = S34_DIR / "experiment-table.json"
    items: dict[str, dict] = {}

    def add(key, status, **fields):
        items[key] = {"status": status, **fields}

    formal: list[dict] = []
    state_rows: dict[int, dict] = {}
    if not (table_path.is_file() and matrix_path and matrix_path.is_file()):
        add(
            "matrix",
            "blocked",
            reason="no finished matrix evidence (experiment-table.json + matrix.json are required)",
            table=rel(table_path) if table_path.is_file() else None,
            matrix=rel(matrix_path) if matrix_path else None,
        )
    else:
        table = read_json(table_path)
        formal = [
            row
            for row in table.get("rows") or []
            if not row.get("smoke") and not row.get("screening") and row.get("kind") == "policy"
        ]
        for row in table.get("rows") or []:
            if row.get("smoke") or row.get("screening") or row.get("kind") != "state":
                continue
            state_rows[int(row.get("depth", -1))] = row
        names = sorted({row["experiment"] for row in formal})
        missing = [name for name in EXPERIMENT_ORDER if name not in names]
        add(
            "matrix",
            "pass" if not missing else "fail",
            reason="six formal seed-7 configurations present" if not missing else f"missing {missing}",
            experiments=names,
            missing=missing,
            table=rel(table_path),
            table_sha256=sha256_file(table_path),
            matrix=rel(matrix_path),
            matrix_sha256=sha256_file(matrix_path),
            state_runs=sorted(f"state-d{depth}" for depth in state_rows),
        )

    if not formal:
        for key in (
            "coverage",
            "teacher",
            "history",
            "init",
            "a01_state",
            "a02_policy",
            "baseline",
            "overfit",
            "throughput",
            "v3_entry_throughput",
            "budget",
            "pytest",
            "startup",
            "route",
        ):
            add(key, "blocked", reason="matrix evidence incomplete")
        return write_gate(out, items, route="blocked", summary="matrix evidence is missing")

    coverage_ok = all(
        (row.get("data") or {}).get("coverage_val", {}).get("complete")
        and int((row.get("data") or {}).get("coverage_val", {}).get("actual_frames", -1)) == VAL_FRAMES
        and (row.get("data") or {}).get("coverage_val", {}).get("visited_index_sha256") == VAL_INDEX_SHA256
        for row in formal
    ) and all((row.get("data") or {}).get("val_index_sha256") == VAL_INDEX_SHA256 for row in formal)
    add(
        "coverage",
        "pass" if coverage_ok else "fail",
        reason="every formal run visited the fixed 7241-frame validation split exactly once"
        if coverage_ok
        else "at least one formal run did not cover the fixed validation split",
        rows=[
            {
                "experiment": row.get("experiment"),
                "actual_frames": (row.get("data") or {}).get("coverage_val", {}).get("actual_frames"),
                "complete": (row.get("data") or {}).get("coverage_val", {}).get("complete"),
                "visited_index_sha256": (row.get("data") or {}).get("coverage_val", {}).get(
                    "visited_index_sha256"
                ),
            }
            for row in formal
        ],
    )

    fingerprints = {(row.get("teacher") or {}).get("fingerprint") for row in formal}
    teacher_ok = len(fingerprints) == 1 and TEACHER_FINGERPRINT in fingerprints
    add(
        "teacher",
        "pass" if teacher_ok else "fail",
        reason="all formal runs share the bound teacher fingerprint"
        if teacher_ok
        else "teacher fingerprints differ or do not match the bound one",
        bound=TEACHER_FINGERPRINT,
        observed=sorted(str(value) for value in fingerprints),
    )

    contracts = {(row.get("contract") or {}).get("history_contract") for row in formal}
    states = {row.get("depth"): (row.get("source_state") or {}).get("sha256") for row in formal}
    history_ok = contracts == {vision_history.HISTORY_CAUSAL_SEGMENT_V1} and len(states) == 2
    add(
        "history",
        "pass" if history_ok else "fail",
        reason="all formal runs declare causal-segment-v1 and share one source checkpoint per depth"
        if history_ok
        else "history contract or per-depth source identity is inconsistent",
        history_contracts=sorted(str(value) for value in contracts),
        source_state_by_depth={str(key): value for key, value in states.items()},
    )

    init_ok = all(states.get(depth) for depth in (1, 3))
    add(
        "init",
        "pass" if init_ok else "fail",
        reason="depth-1 and depth-3 policy runs start from a same-depth state checkpoint byte for byte"
        if init_ok
        else "a depth has no recorded state checkpoint hash",
        source_state_by_depth={str(key): value for key, value in states.items()},
    )

    baseline = read_json(baseline_path) if baseline_path and baseline_path.is_file() else None
    if baseline is None:
        add("baseline", "blocked", reason="baseline.json missing")
    else:
        frames_ok = int(baseline.get("val_frames", 0)) == VAL_FRAMES
        add(
            "baseline",
            "pass" if frames_ok else "fail",
            reason="the frozen baselines were re-evaluated on the full 7241-frame split"
            if frames_ok
            else "the baseline recheck does not cover 7241 frames",
            path=rel(baseline_path),
            sha256=sha256_file(baseline_path),
            state=baseline.get("comparison", {}).get("state"),
            policy=baseline.get("comparison", {}).get("policy"),
        )

    overfit_path = find_evidence("overfit-*/overfit.json")
    overfit = read_json(overfit_path) if overfit_path else None
    if overfit is None:
        add("overfit", "blocked", reason="overfit.json missing")
    else:
        entry = overfit.get("entry") or {}
        add(
            "overfit",
            "pass" if entry.get("status") == "pass" else ("blocked" if entry.get("status") == "blocked" else "fail"),
            reason=entry.get("reason"),
            path=rel(overfit_path),
            sha256=sha256_file(overfit_path),
            experiment=overfit.get("experiment"),
            initial_regret_px=((overfit.get("initial") or {}).get("global") or {}).get("regret_px"),
            final_regret_px=((overfit.get("final") or {}).get("global") or {}).get("regret_px"),
            reduction=entry.get("reduction"),
            threshold=OVERFIT_REDUCTION,
            gradients=overfit.get("gradients"),
            indices_sha256=(overfit.get("selection") or {}).get("indices_sha256"),
        )

    throughput = read_json(throughput_path) if throughput_path and throughput_path.is_file() else None
    if throughput is None:
        add("throughput", "blocked", reason="throughput.json missing")
    else:
        derived = throughput.get("derived") or {}
        add(
            "throughput",
            "pass" if derived.get("state_epoch_seconds") is not None else "fail",
            reason="three paths measured; measured epoch seconds recorded against the 60 s ceiling",
            path=rel(throughput_path),
            sha256=sha256_file(throughput_path),
            state_epoch_seconds=derived.get("state_epoch_seconds"),
            policy_joint_epoch_seconds=derived.get("policy_joint_epoch_seconds"),
            within_budget=derived.get("state_within_budget"),
            user_ruling="question list §8 Q5: depth 3 may continue at the measured ~72 s/epoch; "
            "the 60 s ceiling is recorded, not enforced, this round",
        )

    add(
        "budget",
        "pass",
        reason="measured per-epoch seconds recorded for every formal run, with the one-time cold-start "
        "epochs separated from the steady state (設計 §5.1); this item only asserts that the record "
        "exists and is honest - whether <=60 s was met is the separate v3_entry_throughput item",
        limit_seconds=BUDGET_SECONDS,
        exceeds_design_ceiling=bool(
            [row for row in formal if (row.get("budget") or {}).get("exceeded")]
            or [row for row in state_rows.values() if (row.get("budget") or {}).get("exceeded")]
        ),
        rows={
            row.get("experiment"): {
                "steady_mean_seconds": (row.get("budget") or {}).get("steady_mean_seconds"),
                "steady_max_seconds": (row.get("budget") or {}).get("steady_max_seconds"),
                "cold_start_seconds": (row.get("budget") or {}).get("cold_start_seconds"),
                "exceeded": (row.get("budget") or {}).get("exceeded"),
                "overrun_fraction": (row.get("budget") or {}).get("overrun_fraction"),
            }
            for row in [*state_rows.values(), *formal]
        },
        user_ruling="question list §8 Q5: depth 3 may continue at the measured ~72 s/epoch; the 60 s "
        "ceiling is recorded, not enforced, this round",
    )

    throughput_steady = {}
    for row in [*state_rows.values(), *formal]:
        timing = (row.get("budget") or {})
        if timing.get("steady_mean_seconds") is None:
            continue
        throughput_steady[row.get("experiment")] = {
            "depth": row.get("depth"),
            "kind": row.get("kind"),
            "steady_mean_seconds": timing.get("steady_mean_seconds"),
            "steady_max_seconds": timing.get("steady_max_seconds"),
            "overrun_fraction": timing.get("overrun_fraction"),
        }
    depth3 = {
        name: value
        for name, value in throughput_steady.items()
        if int(value.get("depth", 0)) == 3
    }
    worst_depth3 = max((value["steady_max_seconds"] for value in depth3.values()), default=None)
    probes = (throughput or {}).get("derived") or {}
    add(
        "v3_entry_throughput",
        "fail" if (worst_depth3 is not None and worst_depth3 > BUDGET_SECONDS) else "pass",
        criterion="設計 §5.2 / 任务书 §5：v2 离线不足时，允许候选 v3 的条件之一为吞吐 <= 60 s/epoch"
        "（epoch 秒数 = 训练 + 全验证，排除一次性 cache/labels 构建）",
        measured={
            "depth3_steady_max_seconds": worst_depth3,
            "depth3_runs": depth3,
            "depth1_steady_max_seconds": max(
                (value["steady_max_seconds"] for value in throughput_steady.values() if int(value.get("depth", 0)) == 1),
                default=None,
            ),
            "throughput_probe_forecast_seconds": probes.get("state_epoch_seconds"),
            "s3_3_measured_seconds": 71.15,
        },
        limit_seconds=BUDGET_SECONDS,
        reason=(
            f"depth-3 steady epochs peak at {worst_depth3:.1f} s (>60 s by "
            f"{worst_depth3 / BUDGET_SECONDS - 1.0:+.1%}); the `<=60 s` v3 entry condition is therefore "
            "not met on the letter of the contract"
            if worst_depth3 is not None
            else "no depth-3 steady epoch timing available"
        ),
        needs_user_ruling=True,
        q5_note="Q5 裁决只允许 depth 3 以实测 ~72 s/epoch 继续本矩阵，不构成本条判据通过；"
        "本项按字面记 fail，不得扩大解释为吞吐门通过。",
    )

    if args.pytest_log:
        log_path = Path(args.pytest_log)
        if not log_path.is_file():
            raise v104_error("EVIDENCE", f"--pytest-log not found: {log_path}")
        info = parse_pytest_log(log_path)
        exit_code = getattr(args, "pytest_exit_code", None)
        ok = (
            info["parsed"]
            and info["failed"] == 0
            and (info["passed"] or 0) >= 178
            and (exit_code in (None, 0))
        )
        add(
            "pytest",
            "pass" if ok else "fail",
            reason="0 failed and at least 178 cases" if ok else "pytest did not report a clean run",
            command_exit=exit_code,
            command_exit_source="--pytest-exit-code" if exit_code is not None else "not supplied",
            **info,
        )
    else:
        add("pytest", "blocked", reason="no --pytest-log supplied; A-05 cannot be judged")

    ordered = route_order(formal)
    routed = ordered[0] if ordered else None

    # A-01 is judged on stages.state.best of the *source state run* (需求文档 §7:
    # "不读 final" and "A-01 仍按感知阶段 best 判定"), never on the policy checkpoint.
    star_state_row = state_rows.get(int(routed.get("depth", -1))) if routed else None
    if star_state_row is None:
        add("a01_state", "blocked", reason="no formal state row for the selected depth")
    else:
        state_best = (star_state_row.get("metrics") or {}).get("state_best") or {}
        position = state_best.get("val_player_pos_px")
        danger = state_best.get("val_danger_corr_perframe_median")
        valid = state_best.get("val_danger_corr_perframe_valid")
        a01_ok = bool(
            position is not None
            and danger is not None
            and float(position) < A01_POS_PX
            and float(danger) > A01_DANGER_MEDIAN
        )
        add(
            "a01_state",
            "pass" if a01_ok else "fail",
            criterion=f"val_player_pos_px < {A01_POS_PX} 且 val_danger_corr_perframe_median > {A01_DANGER_MEDIAN}"
            "（stages.state.best，7241 帧全验证）",
            measured={
                "val_player_pos_px": position,
                "val_danger_corr_perframe_median": danger,
                "val_danger_corr_perframe_valid": valid,
                "epoch": state_best.get("epoch"),
                "selection_rule": star_state_row.get("checkpoint", {}).get("selection_rule"),
            },
            checkpoint=star_state_row.get("run_dir"),
            checkpoint_sha256=star_state_row.get("checkpoint", {}).get("sha256"),
            reason="state best misses at least one A-01 bar" if not a01_ok else "A-01 met",
        )

    if routed is None:
        add("a02_policy", "blocked", reason="no policy row to select E*")
    else:
        star_best = (routed.get("metrics") or {}).get("policy_best") or {}
        bucket = ((routed.get("metrics") or {}).get("policy_0_15") or {})
        regret = bucket.get("regret_px")
        frames = bucket.get("frames")
        a02_ok = bool(regret is not None and frames and int(frames) > 0 and float(regret) < A02_BUCKET_REGRET)
        add(
            "a02_policy",
            "pass" if a02_ok else "fail",
            criterion=f"0-15 px 桶后悔 < {A02_BUCKET_REGRET} px（stages.policy.best.by_danger，空桶不算）",
            experiment=routed.get("experiment"),
            measured={
                "bucket_0_15_regret_px": regret,
                "bucket_0_15_frames": frames,
                "bucket_0_15_top3": bucket.get("top3"),
                "bucket_0_15_near_optimal_05": bucket.get("near_optimal_05"),
                "global_regret_px": star_best.get("val_regret_px"),
                "regret_p90_px": star_best.get("val_regret_px_p90_px") or star_best.get("val_regret_p90_px"),
                "epoch": star_best.get("epoch"),
            },
            checkpoint=routed.get("run_dir"),
            checkpoint_sha256=(routed.get("checkpoint") or {}).get("sha256"),
            reason="0-15 bucket regret is above the A-02 line" if not a02_ok else "A-02 met",
        )

    if routed is None:
        add("startup", "blocked", reason="no policy row to load")
    else:
        manifest_name = checkpoint_name(routed.get("checkpoint"))
        checkpoint = ROOT / routed["run_dir"] / manifest_name
        try:
            loaded = load_checkpoint(checkpoint, pick_device(args.device))
            dataset = VisionDataset(DATASET_A, history_contract=loaded["history_contract"])
            try:
                indices = load_split()[3][:4]
                batch = batch_tensors(dataset, indices, loaded["depth"], loaded["region_sizes"], pick_device(args.device))
                with torch.no_grad():
                    scores = loaded["model"].score(batch)
                    act_result = loaded["model"].act(batch)
                    actions = action_tensor(act_result)
            finally:
                dataset.close()
            finite = bool(torch.isfinite(scores).all())
            actions = actions.detach().cpu()
            legal = bool(
                actions.shape == (4, 4)
                and torch.equal(actions[:, 1], torch.ones_like(actions[:, 1]))
                and torch.equal(actions[:, 3], torch.zeros_like(actions[:, 3]))
                and bool(((actions[:, 0] >= 0) & (actions[:, 0] < 9)).all())
                and bool(((actions[:, 2] >= 0) & (actions[:, 2] < 2)).all())
            )
            add(
                "startup",
                "pass" if finite and legal else "fail",
                reason="real CLI run -> summary -> strict checkpoint load -> finite scores / legal actions"
                if finite and legal
                else "the loaded checkpoint produced non-finite scores or illegal actions",
                checkpoint=rel(checkpoint),
                checkpoint_sha256=loaded["sha256"],
                arch=loaded["arch"],
                depth=loaded["depth"],
                history_contract=loaded["history_contract"],
                scores_shape=list(scores.shape),
                scores_finite=finite,
                act_contract=act_contract(act_result),
                actions_shape=list(actions.shape),
                actions_dtype=str(actions.dtype).replace("torch.", ""),
                actions_legal=legal,
                actions_sample=actions.tolist(),
                experiment=routed["experiment"],
            )
        except (ValueError, RuntimeError) as exc:
            add("startup", "fail", reason=f"{type(exc).__name__}: {exc}", checkpoint=rel(checkpoint))

    route, route_reason, detail = route_decision(
        ordered,
        state_rows=state_rows,
        overfit_entry=(overfit or {}).get("entry"),
        overfit_gradients=(overfit or {}).get("gradients"),
        throughput_worst_steady=worst_depth3,
        coverage_ok=coverage_ok,
    )
    add(
        "route",
        detail.get("route_status", "pass"),
        reason=route_reason,
        route=route,
        **{key: value for key, value in detail.items() if key != "route_status"},
    )
    return write_gate(out, items, route=route, summary=route_reason)


def checkpoint_name(value, default: str = "policy.pt") -> str:
    """Checkpoint file name from a table field that is a string or a record."""
    if isinstance(value, dict):
        return str(value.get("name") or default)
    return str(value or default)


def route_order(rows) -> list[dict]:
    """Order by 0-15 bucket regret, then global regret, then earlier epoch (design §5.2)."""

    def key(row):
        policy_best = (row.get("metrics") or {}).get("policy_best") or {}
        bucket = ((row.get("metrics") or {}).get("policy_0_15") or {}).get("regret_px")
        if bucket is None:
            bucket = next(
                (
                    entry.get("regret_px")
                    for entry in (row.get("metrics") or {}).get("by_danger") or []
                    if entry.get("bucket") == "0-15"
                ),
                float("inf"),
            )
        global_regret = policy_best.get("val_regret_px")
        epoch = policy_best.get("epoch")
        return (
            float(bucket) if bucket is not None else float("inf"),
            float(global_regret) if global_regret is not None else float("inf"),
            int(epoch) if epoch is not None else 10**6,
        )

    return sorted(rows, key=key)


def route_decision(
    ordered,
    *,
    state_rows=None,
    overfit_entry=None,
    overfit_gradients=None,
    throughput_worst_steady=None,
    coverage_ok=True,
) -> tuple[str, str, dict]:
    """The written G-v3 verdict: v2-skip condition, or the candidate-v3 checklist.

    Three outcomes, all machine readable:

    * ``v2-sufficient`` - E* policy clears the 0-15 bucket line *and* its own full
      perception re-evaluation clears position<10 px and danger>0.8 (routing only,
      it never replaces A-01/A-02);
    * ``candidate-v3-blocked`` - v2 is insufficient, but at least one written v3
      entry condition fails; the user has to rule before any v3 work;
    * ``candidate-v3-ready`` - every written v3 entry condition is met.

    The ``<=60 s/epoch`` condition is judged on the letter of 設計 §5.2: the user's
    Q5 ruling allowed depth 3 to *keep running*, it did not pass this criterion,
    so an over-limit steady epoch makes the condition fail here.
    """
    if not ordered:
        return "blocked", "no formal configuration to route", {"route_status": "blocked"}
    star = ordered[0]
    policy_best = (star.get("metrics") or {}).get("policy_best") or {}
    bucket_row = (star.get("metrics") or {}).get("policy_0_15") or {}
    bucket = bucket_row.get("regret_px")
    frames = bucket_row.get("frames")
    perception = star.get("final_policy_perception") or {}
    source = star.get("source_state") or {}
    second = ordered[1] if len(ordered) > 1 else None
    second_bucket = (
        ((second.get("metrics") or {}).get("policy_0_15") or {}).get("regret_px") if second else None
    )
    state_row = (state_rows or {}).get(int(star.get("depth", -1))) or {}
    state_best = (state_row.get("metrics") or {}).get("state_best") or {}
    a01_state = bool(
        state_best.get("val_player_pos_px") is not None
        and state_best.get("val_danger_corr_perframe_median") is not None
        and float(state_best["val_player_pos_px"]) < A01_POS_PX
        and float(state_best["val_danger_corr_perframe_median"]) > A01_DANGER_MEDIAN
    )
    detail = {
        "selected_experiment": star.get("experiment"),
        "selection_rule": "0-15 桶后悔 -> 全局后悔 -> 较早 epoch（设计 §5.2）",
        "ranking": [
            {
                "experiment": row.get("experiment"),
                "bucket_0_15_regret_px": ((row.get("metrics") or {}).get("policy_0_15") or {}).get("regret_px"),
                "global_regret_px": ((row.get("metrics") or {}).get("policy_best") or {}).get("val_regret_px"),
                "epoch": ((row.get("metrics") or {}).get("policy_best") or {}).get("epoch"),
            }
            for row in ordered
        ],
        "selection_key": {
            "bucket_0_15_regret_px": bucket,
            "global_regret_px": policy_best.get("val_regret_px"),
            "epoch": policy_best.get("epoch"),
        },
        "selection_margin_to_second_px": (
            round(float(second_bucket) - float(bucket), 4)
            if (bucket is not None and second_bucket is not None)
            else None
        ),
        "policy_checkpoint_sha256": (star.get("checkpoint") or {}).get("sha256"),
        "source_state_sha256": source.get("sha256"),
        "source_state_checkpoint": source.get("checkpoint"),
        "policy_0_15_frame_count": frames,
        "policy_perception": perception,
        "a01_state": {
            "status": "pass" if a01_state else "fail",
            "val_player_pos_px": state_best.get("val_player_pos_px"),
            "val_danger_corr_perframe_median": state_best.get("val_danger_corr_perframe_median"),
            "checkpoint_sha256": (state_row.get("checkpoint") or {}).get("sha256"),
            "run": state_row.get("run_dir"),
        },
    }
    a02 = bucket is not None and float(bucket) < A02_BUCKET_REGRET and int(frames or 0) > 0
    a01_perception = bool(perception.get("a01_pos_pass")) and bool(perception.get("a01_danger_pass"))
    if a02 and a01_perception:
        detail["v2_skip_v3"] = True
        detail["v3_entry"] = None
        detail["route_status"] = "pass"
        return (
            "v2-sufficient",
            "E* clears the 0-15 bucket regret line for its policy and its own full perception "
            "re-evaluation satisfies position<10 px and danger median>0.8, so v2 need not enter "
            "v3; this is a routing condition only and does not replace A-01/A-02.",
            detail,
        )

    gradients = overfit_gradients or {}
    entry = overfit_entry or {}
    entry_status = entry.get("status")
    worst = throughput_worst_steady
    conditions = {
        "sequence_labels_coverage": {
            "status": "pass" if coverage_ok else "fail",
            "criterion": "序列对齐 / 标签 / 覆盖正确（7241 帧全验证、teacher 指纹一致、causal-segment-v1）",
            "measured": "coverage/teacher/history 三项见同名 item",
        },
        "small_set_regret_reduction_30pct": {
            "status": entry_status if entry_status in ("pass", "fail", "blocked") else "blocked",
            "criterion": "小集 rank 后悔相对初始化下降 >= 30%（含有效对与梯度）",
            "measured": {
                "reduction": entry.get("reduction"),
                "initial_regret_px": entry.get("initial_regret_px"),
                "final_regret_px": entry.get("final_regret_px"),
            },
        },
        "valid_pairs_gradient": {
            "status": "pass" if (gradients.get("finite") and gradients.get("nonzero")) else "blocked",
            "criterion": "有效对有限非零、梯度有限非零",
            "measured": gradients,
        },
        "throughput_le_60s": {
            "status": ("pass" if worst <= BUDGET_SECONDS else "fail") if worst is not None else "blocked",
            "criterion": "吞吐 <= 60 s/epoch（训练 + 全验证，排除一次性 cache/labels）",
            "measured": {
                "depth3_steady_max_seconds": worst,
                "limit_seconds": BUDGET_SECONDS,
            },
            "needs_user_ruling": bool(worst is not None and worst > BUDGET_SECONDS),
            "q5_note": "Q5 裁决允许 depth 3 以实测 ~72 s/epoch 继续跑，不等于本条通过；"
            "按字面记 fail，需用户裁决。",
        },
    }
    failed = [name for name, value in conditions.items() if value["status"] == "fail"]
    blocked = [name for name, value in conditions.items() if value["status"] == "blocked"]
    detail["v2_skip_v3"] = False
    detail["v3_entry"] = {
        "conditions": conditions,
        "failed": failed,
        "blocked": blocked,
        "status": "ready" if not failed and not blocked else "blocked",
        "needs_user_ruling": bool(failed or blocked),
    }
    if failed or blocked:
        detail["route_status"] = "blocked"
        return (
            "candidate-v3-blocked",
            "E* does not clear the v2-skip routing condition, and the written v3 entry checklist "
            f"is not complete (failed={failed}, blocked={blocked}).  In particular the <=60 s/epoch "
            "condition is not met on the letter of 設計 §5.2; the Q5 ruling covered continuing the "
            "depth-3 run, not this criterion, so v3 needs a new user ruling.  No v3 run is started "
            "by this tool.",
            detail,
        )
    detail["route_status"] = "pass"
    return (
        "candidate-v3-ready",
        "E* does not clear the v2-skip routing condition, but every written v3 entry condition "
        "(coverage/labels, small-set 30% line, valid-pair gradients, <=60 s/epoch) is met; v3 may "
        "be proposed.  No v3 run is started by this tool.",
        detail,
    )


def write_gate(out: Path, items: dict, *, route: str, summary: str) -> int:
    counts = {"pass": 0, "fail": 0, "blocked": 0}
    for item in items.values():
        counts[item.get("status", "blocked")] += 1
    exit_code = 2 if counts["blocked"] else (1 if counts["fail"] else 0)
    payload = {
        "schema": GATE_SCHEMA,
        "phase": "gate",
        "status": "done",
        "created_at": now_iso(),
        "command": command_line([sys.executable, *sys.argv]),
        "cwd": str(ROOT),
        "route": route,
        "route_reason": summary,
        "items": items,
        "counts": counts,
        "exit_code": exit_code,
        "v3_started": False,
        "v3_note": "This tool never starts a v3 run; G-v3 is a written verdict for the MainAgent.",
    }
    write_json(out / "gate.json", payload)
    write_json(out / "gate-v3.json", payload)
    if (S34_DIR / "gate-v3.json").resolve() != (out / "gate-v3.json").resolve():
        write_json(S34_DIR / "gate-v3.json", payload)
    markdown = gate_markdown(payload)
    (out / "gate-v3.md").write_text(markdown, encoding="utf-8")
    if (S34_DIR / "gate-v3.md").resolve() != (out / "gate-v3.md").resolve():
        (S34_DIR / "gate-v3.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return exit_code


def compact_measured(measured: dict, *, limit: int = 110) -> str:
    """One readable line of the ``measured`` block for the markdown table."""
    parts = []
    for name, value in (measured or {}).items():
        if isinstance(value, dict):
            pieces = []
            for key, inner in value.items():
                if isinstance(inner, dict) and inner.get("steady_max_seconds") is not None:
                    pieces.append(
                        f"{key} {inner.get('steady_mean_seconds')}/{inner.get('steady_max_seconds')} s"
                    )
                else:
                    pieces.append(f"{key}={inner}")
            text = "; ".join(pieces)
        elif isinstance(value, float):
            text = f"{value:.4g}"
        else:
            text = str(value)
        if len(text) > limit:
            text = text[: limit - 3] + "..."
        parts.append(f"{name}={text}")
    return ", ".join(parts)


def gate_markdown(payload: dict) -> str:
    lines = [
        "# S3-4 G-v3 门控判定（机器生成）",
        "",
        f"- 生成时间：{payload['created_at']}",
        f"- 命令：`{payload['command']}`",
        f"- 结论路线：`{payload['route']}`",
        f"- 计数：pass={payload['counts']['pass']} fail={payload['counts']['fail']} "
        f"blocked={payload['counts']['blocked']}",
        "",
        "## 逐项判定",
        "",
        "| 项 | 状态 | 理由 / 关键值 |",
        "| --- | --- | --- |",
    ]
    for key, item in payload["items"].items():
        reason = str(item.get("reason", ""))
        measured = item.get("measured")
        if isinstance(measured, dict) and measured:
            compact = compact_measured(measured)
            reason = f"{reason}；{compact}" if reason else compact
        reason = reason.replace("|", "/")
        lines.append(f"| {key} | {item.get('status')} | {reason} |")
    lines.extend(["", "## 路线", "", payload["route_reason"], ""])
    route_item = payload["items"].get("route") or {}
    if route_item.get("selected_experiment"):
        selection = route_item.get("selection_key") or {}
        margin = route_item.get("selection_margin_to_second_px")
        lines.extend(
            [
                f"- 预选 E*：{route_item['selected_experiment']}，0–15 桶后悔 "
                f"{selection.get('bucket_0_15_regret_px')} px，全局后悔 "
                f"{selection.get('global_regret_px')} px，epoch {selection.get('epoch')}"
                + (f"（与第二名相差 {margin} px）" if margin is not None else ""),
                f"- policy sha256：{route_item.get('policy_checkpoint_sha256')}",
                f"- source state sha256：{route_item.get('source_state_sha256')}",
            ]
        )
        ranking = route_item.get("ranking") or []
        if ranking:
            lines.extend(["", "| 排序 | 配置 | 0–15 桶后悔 px | 全局后悔 px | epoch |", "| ---: | --- | ---: | ---: | ---: |"])
            for position, row in enumerate(ranking, start=1):
                lines.append(
                    f"| {position} | {row.get('experiment')} | {row.get('bucket_0_15_regret_px')} | "
                    f"{row.get('global_regret_px')} | {row.get('epoch')} |"
                )
    entry = route_item.get("v3_entry") or {}
    if entry:
        lines.extend(
            [
                "",
                "## v3 进入条件清单（逐条，按字面）",
                "",
                "| 条件 | 状态 | 判据 | 实测 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for name, condition in (entry.get("conditions") or {}).items():
            measured = condition.get("measured")
            if isinstance(measured, dict):
                measured = ", ".join(f"{key}={value}" for key, value in measured.items())
            lines.append(
                f"| {name} | {condition.get('status')} | {condition.get('criterion')} | {measured} |"
            )
        lines.append("")
        lines.append(
            f"清单结论：**{entry.get('status')}**；未通过 {entry.get('failed')}，未判定 "
            f"{entry.get('blocked')}；需用户裁决={entry.get('needs_user_ruling')}。"
        )
    lines.extend(["", "本工具不会启动 v3；进入 v3 与否由 MainAgent 依此书面判定决定。", ""])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="v1.0.4 S3-4 first-round comparison and route decision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", required=True, choices=["baseline", "overfit", "throughput", "matrix", "gate"])
    parser.add_argument("--out", required=True, help="evidence directory (created only with --execute)")
    parser.add_argument(
        "--experiment",
        action="append",
        choices=list(EXPERIMENT_ORDER),
        default=None,
        help="configuration(s); required for overfit/throughput, optional subset for matrix",
    )
    parser.add_argument("--screening", action="store_true", help="matrix only: 24 perception epochs, ineligible")
    parser.add_argument("--execute", action="store_true", help="actually run; the default is dry run")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--dataset", default=str(DATASET_A), help="fixed to the A dataset; B is refused by this stage"
    )
    parser.add_argument("--smoke", action="store_true", help="matrix only: 1 epoch + a few batches per run")
    parser.add_argument("--smoke-epochs", type=int, default=None, help="smoke epochs per stage (default 1)")
    parser.add_argument("--smoke-train-batches", type=int, default=None, help="smoke training batches (default 2)")
    parser.add_argument("--overfit-steps", type=int, default=OVERFIT_MAX_STEPS, help=f"<= {OVERFIT_MAX_STEPS}")
    parser.add_argument("--throughput-json", default="", help="matrix: use this throughput.json for the estimates")
    parser.add_argument("--matrix", default="", help="gate: explicit matrix.json")
    parser.add_argument("--baseline", default="", help="gate: explicit baseline.json")
    parser.add_argument("--throughput", default="", help="gate: explicit throughput.json")
    parser.add_argument("--pytest-log", default="", help="gate: pytest output for A-05")
    parser.add_argument(
        "--pytest-exit-code",
        type=int,
        default=None,
        help="gate: the exit code of that pytest run (optional; recorded, never guessed)",
    )
    return parser


def run_phase(args) -> int:
    """Argument contract checks, then the phase itself."""
    if Path(args.dataset).resolve() != DATASET_A.resolve():
        raise v104_error(
            "DATA",
            f"this stage only runs on dataset A ({rel(DATASET_A)}); got {args.dataset!r}"
            " (pilot-ud0063 is a read-only cross-replay check and must never be trained on)",
        )
    if args.phase != "matrix" and (args.smoke or args.screening):
        raise v104_error(
            "CONFIG", f"--smoke/--screening are only valid with --phase matrix (phase {args.phase})"
        )
    if args.phase != "matrix" and (args.smoke_epochs is not None or args.smoke_train_batches is not None):
        raise v104_error("CONFIG", "--smoke-epochs/--smoke-train-batches are only valid with --phase matrix")
    if args.phase == "matrix":
        if args.smoke and args.screening:
            raise v104_error("CONFIG", "pick either --smoke or --screening")
        if not args.smoke and (args.smoke_epochs is not None or args.smoke_train_batches is not None):
            raise v104_error("CONFIG", "the smoke switches cannot be used without --smoke")
        if args.smoke and (args.smoke_epochs if args.smoke_epochs is not None else 1) < 1:
            raise v104_error("CONFIG", "--smoke-epochs must be >= 1")
        if args.smoke and (args.smoke_train_batches if args.smoke_train_batches is not None else 2) < 1:
            raise v104_error("CONFIG", "--smoke-train-batches must be >= 1")
    args.out = str(Path(args.out))
    if args.phase == "baseline":
        return phase_baseline(args)
    if args.phase == "overfit":
        return phase_overfit(args)
    if args.phase == "throughput":
        return phase_throughput(args)
    if args.phase == "matrix":
        return phase_matrix(args)
    return phase_gate(args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_phase(args)
    except ManifestError as error:
        print(str(error), file=sys.stderr, flush=True)
        return 2
    except ValueError as error:
        message = str(error)
        if message.startswith("v104:"):
            print(message, file=sys.stderr, flush=True)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
