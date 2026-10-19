"""v1.0.4 Stage3 / S3-7: refinement, R2/R3 ablations and the final freeze (tool).

One CLI, seven phases (task book §5)::

    --phase v3-matrix    E4 / E5 / E6 on ``v3-st`` (72 perception + 12 policy each)
    --phase hard-frames  R2's once-declared hard-frame weights: a train-split-only
                         per-frame native-px error pass over a frozen perception
                         checkpoint, written as NPZ + a separate JSON identity
    --phase r2           one hard-frame-weighted run on the winning architecture
    --phase r3           one learnable danger head run (v2, ``learnable-v104``)
    --phase integrate    one run with both R2 and R3, only after each helped alone
    --phase stability    the frozen recipe on seeds 17 / 27 (seed 7 stays the only
                         model that may be frozen / shipped)
    --phase freeze       ``Stage3/final-artifacts.json`` (schema ``v104-final/1``)

Contract points implemented here, and only here
------------------------------------------------
* **Dry run is the default.**  Nothing is created, no run directory is touched and
  no GPU work happens without ``--execute``.  The dry run prints the exact
  command line, the run directory, the recipe digest and the projected wall
  clock, so a later ``--execute`` cannot change the recipe.
* **A training phase eats a complete recipe.**  ``train_config`` must carry every
  effective ``VisionTrainConfig`` value (objective / weights / auxiliary terms /
  seed / budget / variant); unknown keys, missing keys and unexpressible values
  are ``v104:CONFIG``.  There are no hidden overrides: the command is generated
  from the validated recipe and the effective config is echoed back verbatim.
* **``--max-val-batches 0`` is enforced, not defaulted.**  A shortened validation
  pass empties the 0-15 danger bucket on the real split and the trainer then
  refuses to write ``policy.pt`` (S3-3 §10.3).
* **E4/E5/E6 have exactly one variable.**  ``local_enabled`` separates E4 from
  E5; E6 is E5 with the *policy-stage* auxiliary weights all zero (perception
  unchanged).  This is verified across the three recipes instead of trusted.
* **v3 needs the written G-v3 gate.**  ``--gate`` must point at the S3-4 verdict;
  a failed entry may only be waived if the recipe names the waiver (Q6 waives
  ``throughput_le_60s`` only) - the entry stays recorded as ``fail``.
* **R2 identities are JSON, never pickle**, and the weights are re-derived from
  the errors on load: 65172 training indices, no duplicate, no validation index,
  finite, ``e<10 -> 1``, ``10<=e<=30 -> 1.5``, ``e>30 -> 2``.
* **Freeze uses what the summaries selected.**  ``stages.state.best`` /
  ``stages.policy.best`` supply the checkpoint paths and their recorded sha256;
  ``best.pt`` is never guessed.  An existing *valid* ``final-artifacts.json`` is
  never overwritten, and a candidate below the hard lines may still be frozen
  (labelled, with the unmet items listed) - that is not "the version passed".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from touhou_rl import vision, vision_losses, vision_train  # noqa: E402
from touhou_rl.pipeline.manifest import (  # noqa: E402
    DatasetManifest,
    ManifestError,
    file_digest,
    index_digest,
)
from touhou_rl.vision import regions_to_batch  # noqa: E402
from touhou_rl.vision_data import VisionDataset  # noqa: E402
from touhou_rl.vmodels import load_vision_checkpoint, vision_arch  # noqa: E402

# --------------------------------------------------------------------------- #
# fixed identities of this stage
# --------------------------------------------------------------------------- #
PHASE_SCHEMA = "v104-refine-phase/1"
MATRIX_SCHEMA = "v104-refine-matrix/1"
RECIPE_SCHEMA = "v104-refine-recipe/1"
HARDFRAMES_SCHEMA = "v104-hard-frames/1"
THROUGHPUT_SCHEMA = "v104-refine-throughput/1"
FINAL_SCHEMA = "v104-final/1"
OUTPUT_SCHEMA = "v104-refine-output/1"

DATASET_A = ROOT / "third_party" / "th08" / "build" / "rl" / "datasets" / "pilot-ud0059-v2"
DATASET_B = ROOT / "third_party" / "th08" / "build" / "rl" / "datasets" / "pilot-ud0063"
RUNS_ROOT = ROOT / "third_party" / "th08" / "build" / "rl" / "runs"
STAGE3_DIR = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3"
S37_DIR = STAGE3_DIR / "S3-7"
FINAL_ARTIFACTS = STAGE3_DIR / "final-artifacts.json"

#: Read-only inputs frozen by earlier sub-stages (never written here).
FROZEN_RUN_NAMES = ("pilot-v1", "v104-a01-72ep", "v104-clean-24e12e")
OLD_STATE_CHECKPOINT = RUNS_ROOT / "v104-a01-72ep" / "state.pt"
S34_EXPERIMENT_TABLE = STAGE3_DIR / "S3-4" / "experiment-table.json"
GATE_V3 = STAGE3_DIR / "S3-4" / "gate-v3.json"

S37_BATCH_SIZE = 96
SEED = 7
STABILITY_SEEDS = (17, 27)
VAL_FRAMES = 7241
TRAIN_FRAMES = 65172
TRAIN_BATCHES = TRAIN_FRAMES // S37_BATCH_SIZE  # 678 full batches (drop_last)
VAL_BATCHES = math.ceil(VAL_FRAMES / S37_BATCH_SIZE)  # 76 = 75*96 + 41
VAL_INDEX_SHA256 = "e5d4ebbe6d8de05793520a393fc1fdbea2b1e2c4db9ff7c1632908340c76655f"
STATE_EPOCHS = 72
POLICY_EPOCHS = 12
#: 設計 §5.1 explicitly allows a 24-epoch perception screening run to save budget.
#: It is marked ``screening`` in the summary and can never be acceptance eligible,
#: and it never substitutes for the formal 72-epoch comparison.
SCREENING_STATE_EPOCHS = 24
BUDGET_SECONDS = 60.0
A01_POS_THRESHOLD = 10.0
A01_DANGER_THRESHOLD = 0.8
A02_BUCKET = "0-15"
A02_REGRET_THRESHOLD = 1.5

#: Fallback epoch estimates (seconds) when no throughput evidence exists.  These
#: are the v2 depth-3 steady-state numbers from S3-4; they are only used to print
#: a projection and every printed estimate names its source.
FALLBACK_EPOCH_SECONDS = {"v2": 72.2, "v3-st": None}

#: ``v3-matrix`` single-variable declaration (task book §3 / 設計 §5.3).  The
#: per-experiment delta is *data*, printed in the plan, and the tool refuses a
#: recipe set that differs in anything else.
V3_MATRIX = {
    "E4": {"local_enabled": False, "policy_aux_zero": False},
    "E5": {"local_enabled": True, "policy_aux_zero": False},
    "E6": {"local_enabled": True, "policy_aux_zero": True},
}
V3_MATRIX_ORDER = ("E4", "E5", "E6")
#: C3: perception auxiliary weights are 1/1/1/1 in every v3 configuration.
PERCEPTION_AUX = {"w_heatmap": 1.0, "w_coord": 1.0, "w_danger": 1.0, "w_self": 1.0}
#: C3: the standard policy auxiliary weights (E6 zeroes exactly these).
POLICY_AUX = {
    "policy_aux_heatmap": 0.1,
    "policy_aux_coord": 1.0,
    "policy_aux_danger": 0.2,
    "policy_aux_self": 0.1,
}

#: Recipe keys a training phase must state explicitly.  Anything else is a
#: hidden override; a missing key is a partial recipe.
REQUIRED_TRAIN_CONFIG = (
    "arch",
    "stage",
    "depth",
    "seed",
    "batch_size",
    "lr",
    "weight_decay",
    "epochs_state",
    "epochs_policy",
    "max_train_batches",
    "max_val_batches",
    "selection_contract",
    "history_contract",
    "policy_objective",
    "rank_delta_px",
    "rank_weight",
    "regression_weight",
    "w_heatmap",
    "w_coord",
    "w_danger",
    "w_self",
    "policy_aux_heatmap",
    "policy_aux_coord",
    "policy_aux_danger",
    "policy_aux_self",
    "policy_freeze_epochs",
    "encoder_lr",
    "policy_head_lr",
    "heatmap_sigma",
    "local_enabled",
    "danger_head_variant",
)
OPTIONAL_TRAIN_CONFIG = ("position_sample_weights", "note", "screening")

#: ``VisionTrainConfig`` field -> CLI flag.  A config key without a flag here can
#: never reach the trainer, so it is rejected instead of being silently applied.
TRAIN_FLAGS = {
    "arch": "--arch",
    "stage": "--stage",
    "depth": "--depth",
    "seed": "--seed",
    "batch_size": "--batch-size",
    "lr": "--lr",
    "weight_decay": "--weight-decay",
    "epochs_state": "--epochs-state",
    "epochs_policy": "--epochs-policy",
    "max_train_batches": "--max-train-batches",
    "max_val_batches": "--max-val-batches",
    "selection_contract": "--selection-contract",
    "history_contract": "--history-contract",
    "policy_objective": "--policy-objective",
    "rank_delta_px": "--rank-delta-px",
    "rank_weight": "--rank-weight",
    "regression_weight": "--regression-weight",
    "w_heatmap": "--w-heatmap",
    "w_coord": "--w-coord",
    "w_danger": "--w-danger",
    "w_self": "--w-self",
    "policy_aux_heatmap": "--policy-aux-heatmap",
    "policy_aux_coord": "--policy-aux-coord",
    "policy_aux_danger": "--policy-aux-danger",
    "policy_aux_self": "--policy-aux-self",
    "policy_freeze_epochs": "--policy-freeze-epochs",
    "encoder_lr": "--encoder-lr",
    "policy_head_lr": "--policy-head-lr",
    "heatmap_sigma": "--heatmap-sigma",
    "danger_head_variant": "--danger-head-variant",
    "position_sample_weights": "--position-sample-weights",
    "note": "--note",
}
#: ``local_enabled`` is a boolean flag pair, not a value.
BOOLEAN_FLAGS = {"local_enabled": ("--local-enabled", "--no-local-enabled")}
#: One-sided boolean switches: present means true, absent means false.
TRUE_ONLY_FLAGS = {"screening": "--screening"}


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


def resolve(path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


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
        return
    raise v104_error("EVIDENCE", f"{path} already holds a finished phase output; refusing to overwrite it")


def fmt_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def env_block() -> dict:
    import torch

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
    import torch

    return "cuda" if requested == "cuda" and torch.cuda.is_available() else requested


# --------------------------------------------------------------------------- #
# the recipe
# --------------------------------------------------------------------------- #
def recipe_digest(recipe: dict) -> str:
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    import hashlib

    return hashlib.sha256(canonical).hexdigest()


def load_recipe(path) -> dict:
    recipe = read_json(path)
    if recipe.get("schema") != RECIPE_SCHEMA:
        raise v104_error(
            "CONFIG", f"recipe schema {recipe.get('schema')!r} != {RECIPE_SCHEMA!r} ({path})"
        )
    if not isinstance(recipe.get("train_config"), dict):
        raise v104_error("CONFIG", f"recipe {path} has no train_config mapping")
    return recipe


def validate_recipe(recipe: dict, *, path, route: str, phase: str) -> dict:
    """Return the normalised plan for one training recipe (no side effects)."""
    train_config = dict(recipe["train_config"])
    known = set(REQUIRED_TRAIN_CONFIG) | set(OPTIONAL_TRAIN_CONFIG)
    unknown = sorted(set(train_config) - known)
    if unknown:
        raise v104_error(
            "CONFIG",
            f"recipe {path} has train_config key(s) {unknown} that this tool cannot express;"
            " a value it does not understand must not be silently dropped",
        )
    missing = [key for key in REQUIRED_TRAIN_CONFIG if key not in train_config]
    if missing:
        raise v104_error(
            "CONFIG", f"recipe {path} is incomplete: train_config is missing {missing}"
        )

    # Dataset / run directory / dataset identity live next to the config.
    dataset = resolve(recipe.get("dataset") or DATASET_A)
    if dataset.resolve() != DATASET_A.resolve():
        raise v104_error(
            "DATA",
            f"refine phases train on dataset A only ({rel(DATASET_A)}); got {rel(dataset)}"
            " (pilot-ud0063 must never be trained on)",
        )
    run_dir = resolve(recipe.get("run_dir") or "")
    if not recipe.get("run_dir"):
        raise v104_error("CONFIG", f"recipe {path} has no run_dir")
    if any(name in {part.lower() for part in run_dir.parts} for name in FROZEN_RUN_NAMES):
        raise v104_error("CHECKPOINT", f"run_dir {rel(run_dir)} is a frozen baseline path")

    # The trainer's own validator is the authority on legal values.
    normalised = dict(train_config)
    if normalised.get("position_sample_weights") in ("", None):
        normalised["position_sample_weights"] = None
    cfg = vision_train.VisionTrainConfig(
        dataset=dataset,
        out_dir=run_dir,
        **{key: value for key, value in normalised.items() if key not in ("note",)},
    )
    vision_train.validate_config(cfg)

    smoke = bool(recipe.get("smoke"))
    if int(train_config["max_val_batches"]) != 0:
        raise v104_error(
            "CONFIG",
            f"recipe {path} sets max_val_batches={train_config['max_val_batches']}; every boundary"
            " run needs the full 7241-frame validation (--max-val-batches 0)",
        )
    if int(train_config["seed"]) != SEED and phase != "stability":
        raise v104_error(
            "CONFIG", f"recipe {path} seeds {train_config['seed']}; this round pre-declares seed 7"
        )
    if int(train_config["max_train_batches"]) != 0 and not smoke:
        raise v104_error(
            "CONFIG",
            f"recipe {path} limits training batches; only the declared smoke recipe may do that"
            " (a limited run is non-eligible screening)",
        )
    if train_config["selection_contract"] != "v104":
        raise v104_error("CONFIG", f"recipe {path} must use selection_contract='v104'")
    screening = bool(train_config.get("screening", False))
    allowed_state_epochs = (0, STATE_EPOCHS) + ((SCREENING_STATE_EPOCHS,) if screening else ())
    if int(train_config["epochs_state"]) not in allowed_state_epochs and not smoke:
        raise v104_error(
            "CONFIG",
            f"recipe {path} trains {train_config['epochs_state']} perception epochs; the declared"
            f" budget is {STATE_EPOCHS} (a screening recipe may declare {SCREENING_STATE_EPOCHS}"
            " with screening=true)",
        )
    if int(train_config["epochs_state"]) == SCREENING_STATE_EPOCHS and not screening:
        raise v104_error(
            "CONFIG",
            f"recipe {path} trains {SCREENING_STATE_EPOCHS} perception epochs; that is the"
            " screening budget and needs screening=true",
        )
    if int(train_config["epochs_policy"]) not in (0, POLICY_EPOCHS) and not smoke:
        raise v104_error(
            "CONFIG",
            f"recipe {path} trains {train_config['epochs_policy']} policy epochs; the declared"
            f" budget is {POLICY_EPOCHS}",
        )
    if (int(train_config["epochs_state"]) + int(train_config["epochs_policy"])) < 1:
        raise v104_error("CONFIG", f"recipe {path} has no epochs at all")

    arch = train_config["arch"]
    if arch != route:
        raise v104_error(
            "CONFIG", f"recipe {path} arch {arch!r} != --route {route!r}"
        )

    source_state = recipe.get("source_state_checkpoint")
    source_sha = recipe.get("source_state_sha256")
    source_block = None
    if source_state:
        source_path = resolve(source_state)
        if not source_path.is_file():
            raise v104_error("CHECKPOINT", f"recipe {path}: source state checkpoint {source_state} is missing")
        actual = sha256_file(source_path)
        if source_sha is not None and source_sha != actual:
            raise v104_error(
                "CHECKPOINT",
                f"recipe {path}: source_state_sha256 {source_sha!r} != the file's {actual}",
            )
        source_block = {
            "plan": rel(source_path),
            "sha256": actual,
            "epochs": int(train_config["epochs_policy"]),
        }
        if train_config["stage"] == "policy" and int(train_config["epochs_policy"]) < 1:
            raise v104_error("CONFIG", f"recipe {path}: stage 'policy' needs policy epochs")
    elif train_config["stage"] == "policy":
        raise v104_error(
            "CONFIG",
            f"recipe {path}: stage 'policy' must name source_state_checkpoint + sha"
            " (the trainer refuses a silent fresh build)",
        )

    command = build_training_command(
        run_dir=run_dir, dataset=dataset, train_config=train_config
    )
    return {
        "experiment": str(recipe.get("experiment") or ""),
        "recipe_path": rel(path),
        "recipe_sha256": sha256_file(path),
        "recipe_digest": recipe_digest(recipe),
        "route": route,
        "phase": phase,
        "run_dir": rel(run_dir),
        "dataset": rel(dataset),
        "train_config": train_config,
        "source_state": source_block,
        "baseline": dict(recipe.get("baseline") or {}),
        "smoke": smoke,
        "screening": screening,
        "notes": str(recipe.get("notes") or ""),
        "command": command,
    }


def build_training_command(*, run_dir: Path, dataset: Path, train_config: dict) -> list[str]:
    """The exact ``python -m touhou_rl.vision_train`` argv of a validated recipe."""
    command = [
        sys.executable,
        "-m",
        "touhou_rl.vision_train",
        "--dataset",
        rel(dataset),
        "--out",
        rel(run_dir),
    ]
    for key in REQUIRED_TRAIN_CONFIG + OPTIONAL_TRAIN_CONFIG:
        if key not in train_config:
            continue
        value = train_config[key]
        if key in BOOLEAN_FLAGS:
            on, off = BOOLEAN_FLAGS[key]
            command.append(on if bool(value) else off)
        elif key in TRUE_ONLY_FLAGS:
            if bool(value):
                command.append(TRUE_ONLY_FLAGS[key])
        elif key == "position_sample_weights":
            if value:
                command.extend([TRAIN_FLAGS[key], str(value)])
        elif key == "note":
            if value:
                command.extend([TRAIN_FLAGS[key], str(value)])
        else:
            command.extend([TRAIN_FLAGS[key], fmt_value(value)])
    return command


def estimate_epoch_seconds(route: str, throughput: dict | None) -> dict:
    """``{perception, policy, source}`` epoch projections in seconds."""
    if throughput:
        paths = throughput.get("paths") or {}
        perception = (paths.get("perception") or {}).get("forecast_epoch_seconds")
        policy = (paths.get("policy_joint") or {}).get("forecast_epoch_seconds")
        if perception and policy:
            return {
                "perception": float(perception),
                "policy": float(policy),
                "source": f"throughput probe {throughput.get('path', '')}",
            }
    if route == "v2":
        value = FALLBACK_EPOCH_SECONDS["v2"]
        return {
            "perception": value,
            "policy": value,
            "source": "fallback: S3-4 measured v2 depth-3 steady epoch (nominal, 72.2 s)",
        }
    return {"perception": None, "policy": None, "source": "no v3 throughput evidence yet"}


def seconds_of(plan: dict, estimates: dict) -> float | None:
    perception = estimates.get("perception")
    policy = estimates.get("policy")
    if perception is None or policy is None:
        return None
    config = plan["train_config"]
    return float(config["epochs_state"]) * float(perception) + float(config["epochs_policy"]) * float(policy)


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "unknown (no v3 throughput evidence)"
    minutes = value / 60.0
    hours = minutes / 60.0
    if hours >= 1.0:
        return f"{value:.0f} s = {minutes:.1f} min = {hours:.2f} h"
    return f"{value:.0f} s = {minutes:.1f} min"


# --------------------------------------------------------------------------- #
# G-v3 gate check
# --------------------------------------------------------------------------- #
def check_v3_gate(gate_path, recipe: dict) -> dict:
    """Read the written S3-4 verdict; only declared waivers can relax a failure.

    The waiving recipe has to name both the conditions it waives and the ruling
    that waives them.  A mismatch (waiving something that passed, or leaving a
    failure unwaived) is refused: this tool never reinterprets the gate.
    """
    gate = read_json(resolve(gate_path))
    if not str(gate.get("schema", "")).startswith("v104-gate"):
        raise v104_error("EVIDENCE", f"{gate_path}: not a v104 gate document")
    items = gate.get("items")
    if not isinstance(items, dict):
        raise v104_error("EVIDENCE", f"{gate_path}: gate has no items mapping")
    entry = (items.get("route") or {}).get("v3_entry") or {}
    conditions = entry.get("conditions")
    if not isinstance(conditions, dict):
        raise v104_error(
            "EVIDENCE",
            f"{gate_path}: items.route.v3_entry.conditions is missing; the v3 entry checklist"
            " is not recorded and cannot be waived",
        )
    failed = sorted(
        name for name, row in conditions.items() if str((row or {}).get("status")) != "pass"
    )
    waiver = (recipe.get("gate") or {}).get("waiver") or {}
    declared = sorted(waiver.get("items") or [])
    by = str(waiver.get("by") or "")
    if failed and not by:
        raise v104_error(
            "EVIDENCE",
            f"G-v3 is blocked on {failed}; a v3 recipe must name the ruling that waives it"
            " (gate.waiver.by/items)",
        )
    if declared != failed:
        raise v104_error(
            "EVIDENCE",
            f"the recipe waives {declared} but the gate's failed entry is {failed};"
            " a waiver must match the verdict exactly",
        )
    statuses = {name: str((row or {}).get("status")) for name, row in conditions.items()}
    return {
        "path": rel(resolve(gate_path)),
        "sha256": sha256_file(resolve(gate_path)),
        "route": gate.get("route"),
        "exit_code": gate.get("exit_code"),
        "conditions": statuses,
        "failed": failed,
        "waived_by": by,
        "waiver_source": waiver.get("source"),
        "needs_user_ruling": bool(entry.get("needs_user_ruling")),
        "note": "failed entries stay recorded as fail; the waiver is printed, never re-scored",
    }


# --------------------------------------------------------------------------- #
# planning helpers
# --------------------------------------------------------------------------- #
def load_throughput_evidence(path) -> dict | None:
    if not path:
        return None
    payload = read_json(resolve(path))
    if payload.get("schema") != THROUGHPUT_SCHEMA:
        raise v104_error("EVIDENCE", f"{path}: not a {THROUGHPUT_SCHEMA} document")
    payload["path"] = rel(resolve(path))
    return payload


def planned_run(plan: dict, estimates: dict) -> dict:
    seconds = seconds_of(plan, estimates)
    return {
        **plan,
        "estimated_seconds": seconds,
        "estimated_wall_clock": fmt_seconds(seconds),
        "budget_seconds_per_epoch": BUDGET_SECONDS,
    }


def check_single_variable(plans: list[dict]) -> dict:
    """Prove E4/E5/E6 differ only where the declaration allows them to.

    A declared screening set may stop after E4/E5: E6 only zeroes the *policy*
    auxiliary weights and cannot change the perception stage, so a screening
    decision that only needs "does v3 beat v2 at 24 epochs" does not have to pay
    for it.  Anything else about the subset is still checked, and the result says
    which experiments were compared.
    """
    names = {plan["experiment"] for plan in plans}
    screening_only = all(plan.get("screening") for plan in plans)
    required = set(V3_MATRIX_ORDER)
    if screening_only and {"E4", "E5"} <= names:
        required = set(names)
    by_name = {plan["experiment"]: plan for plan in plans}
    missing = [name for name in V3_MATRIX_ORDER if name in required and name not in by_name]
    if missing:
        raise v104_error(
            "CONFIG", f"v3-matrix needs exactly the recipes {list(V3_MATRIX_ORDER)}; missing {missing}"
        )
    extra = sorted(set(by_name) - set(V3_MATRIX_ORDER))
    if extra:
        raise v104_error("CONFIG", f"v3-matrix got unexpected experiment(s) {extra}")

    expected_local = {name: V3_MATRIX[name]["local_enabled"] for name in V3_MATRIX_ORDER}
    for name, plan in by_name.items():
        config = plan["train_config"]
        if bool(config["local_enabled"]) != expected_local[name]:
            raise v104_error(
                "CONFIG",
                f"{name}: local_enabled={config['local_enabled']!r}; the declared value is"
                f" {expected_local[name]!r}",
            )
        for key, value in PERCEPTION_AUX.items():
            if float(config[key]) != float(value):
                raise v104_error(
                    "CONFIG", f"{name}: {key}={config[key]!r}; C3 fixes the perception auxiliary at {value}"
                )
        zero_policy_aux = V3_MATRIX[name]["policy_aux_zero"]
        for key, value in POLICY_AUX.items():
            expected = 0.0 if zero_policy_aux else float(value)
            if float(config[key]) != expected:
                raise v104_error(
                    "CONFIG",
                    f"{name}: {key}={config[key]!r}, but the declared delta for {name} is"
                    f" {expected} (E6 zeroes the policy auxiliary; E4/E5 keep C3's {value})",
                )

    reference = by_name["E5"]["train_config"]
    compared = [
        key
        for key in REQUIRED_TRAIN_CONFIG
        if key
        not in {
            "local_enabled",
            "policy_aux_heatmap",
            "policy_aux_coord",
            "policy_aux_danger",
            "policy_aux_self",
        }
    ]
    for name in sorted(set(by_name) - {"E5"}):
        config = by_name[name]["train_config"]
        differing = [key for key in compared if config[key] != reference[key]]
        if differing:
            raise v104_error(
                "CONFIG",
                f"{name} differs from E5 in {differing}; only local_enabled (E4) and the"
                " policy auxiliary weights (E6) may change",
            )
    return {
        "reference": "E5",
        "experiments": sorted(by_name),
        "screening_only": screening_only,
        "compared_fields": compared,
        "delta": {
            "E4": {"local_enabled": False},
            "E5": {"local_enabled": True},
            "E6": {"local_enabled": True, "policy_aux_all_zero": True},
        },
        "perception_aux_fixed": PERCEPTION_AUX,
        "policy_aux_reference": POLICY_AUX,
        "single_variable_verified": True,
    }


# --------------------------------------------------------------------------- #
# subprocess execution
# --------------------------------------------------------------------------- #
def run_training(command: list[str], log_path: Path) -> dict:
    """Run one training command, tee stdout+stderr to ``log_path``; return status."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"COMMAND: {' '.join(command)}\n")
        handle.flush()
        process = subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        handle.write(f"\nEXIT: {process.returncode}\nELAPSED_SECONDS: {elapsed:.2f}\n")
    return {
        "exit_code": int(process.returncode),
        "elapsed_seconds": round(elapsed, 2),
        "log": rel(log_path),
        "log_sha256": sha256_file(log_path),
    }


def run_summary(run_dir: Path) -> dict:
    """Read one finished run's summary and check the contract's coverage gate."""
    summary_path = run_dir / "vision-summary.json"
    if not summary_path.is_file():
        raise v104_error("COVERAGE", f"{rel(run_dir)} finished without vision-summary.json")
    summary = read_json(summary_path)
    coverage = (summary.get("coverage") or {}).get("val") or {}
    if summary.get("acceptance_eligible") and int(coverage.get("actual_frames", 0)) != VAL_FRAMES:
        raise v104_error(
            "COVERAGE",
            f"{rel(run_dir)} claims eligibility but validated {coverage.get('actual_frames')}"
            f" frames instead of {VAL_FRAMES}",
        )
    state_best = ((summary.get("stages") or {}).get("state") or {}).get("best") or {}
    policy_best = ((summary.get("stages") or {}).get("policy") or {}).get("best") or {}
    return {
        "run_dir": rel(run_dir),
        "summary": rel(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "schema": summary.get("schema"),
        "contract_version": summary.get("contract_version"),
        "arch": summary.get("arch"),
        "depth": summary.get("depth"),
        "parameters": summary.get("parameters"),
        "acceptance_eligible": summary.get("acceptance_eligible"),
        "partial": summary.get("partial"),
        "screening": (summary.get("recipe") or {}).get("screening"),
        "seed": (summary.get("recipe") or {}).get("seed"),
        "coverage": {"val": coverage},
        "epoch_seconds": [
            row.get("seconds")
            for stage in ("state", "policy")
            for row in (((summary.get("stages") or {}).get(stage) or {}).get("history") or [])
        ],
        "state_best": {
            key: state_best.get(key)
            for key in ("epoch", "val_player_pos_px", "val_danger_corr_perframe_median", "selection_rule", "checkpoint", "checkpoint_sha256")
            if key in state_best
        },
        "policy_best": {
            key: policy_best.get(key)
            for key in ("epoch", "val_regret_px_bucket_0_15", "val_regret_px", "selection_rule", "checkpoint", "checkpoint_sha256")
            if key in policy_best
        },
        "teacher_fingerprint": (summary.get("teacher") or {}).get("fingerprint"),
        "local_enabled": ((summary.get("input") or {}).get("local_enabled")),
        "danger_head_variant": ((summary.get("input") or {}).get("danger_head_variant")),
        "position_weights": summary.get("position_weights"),
    }


def execute_plans(args, plans: list[dict], output_path: Path, payload: dict, *, decorate=None) -> int:
    """Run every planned training command serially and record the results."""
    results = []
    status = "done"
    exit_code = 0
    for plan in plans:
        run_dir = resolve(plan["run_dir"])
        log_name = f"{plan['experiment'] or Path(plan['run_dir']).name}"
        if plan.get("smoke"):
            log_name = f"{log_name}-smoke"
        log_path = resolve(args.out) / "logs" / f"{log_name}.log"
        print(f"[refine] running {plan['run_dir']} ({plan['estimated_wall_clock']})", flush=True)
        outcome = run_training(plan["command"], log_path)
        row = {
            "experiment": plan["experiment"],
            "run_dir": plan["run_dir"],
            **outcome,
        }
        if outcome["exit_code"] != 0:
            status = "failed"
            exit_code = 1
            row["error"] = "training command failed"
            results.append(row)
            break
        try:
            row["result"] = run_summary(run_dir)
        except ValueError as error:
            status = "failed"
            exit_code = 1
            row["error"] = str(error)
            results.append(row)
            break
        if decorate is not None:
            row["gain_vs_baseline"] = decorate(row["result"], plan)
        results.append(row)
        print(
            f"[refine] {plan['experiment']} exit 0 in {outcome['elapsed_seconds']} s"
            f" -> {row['result']['summary_sha256'][:16]}",
            flush=True,
        )
    payload = {**payload, "results": results, "status": status, "exit_code": exit_code}
    write_json(output_path, payload)
    return exit_code


def execute_phase(args, *, phase_data: dict, plans: list[dict], output_name: str, decorate=None) -> int:
    out_dir = resolve(args.out)
    output_path = out_dir / output_name
    refuse_existing_output(output_path)
    payload = {
        "schema": OUTPUT_SCHEMA,
        "phase": args.phase,
        "status": "running",
        "created_at": now_iso(),
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(ROOT),
        "environment": env_block(),
        **phase_data,
    }
    write_json(output_path, payload)
    return execute_plans(args, plans, output_path, payload, decorate=decorate)


# --------------------------------------------------------------------------- #
# phase: v3-matrix (+ smoke / throughput)
# --------------------------------------------------------------------------- #
def phase_v3_matrix(args) -> int:
    if args.throughput:
        return phase_throughput(args)
    recipes = [load_recipe(path) for path in args.recipe or []]
    if not recipes:
        raise v104_error("CONFIG", "--phase v3-matrix needs --recipe (one per E4/E5/E6 recipe)")
    throughput = load_throughput_evidence(args.throughput_json) if args.throughput_json else None
    estimates = estimate_epoch_seconds("v3-st", throughput)
    base_plans = [
        validate_recipe(recipe, path=path, route="v3-st", phase="v3-matrix")
        for recipe, path in zip(recipes, args.recipe)
    ]
    single = check_single_variable(base_plans)
    plans = [planned_run(plan, estimates) for plan in base_plans]
    gate_path = args.gate or str(GATE_V3)
    gate_blocks = [check_v3_gate(gate_path, recipe) for recipe in recipes]
    total = sum(plan["estimated_seconds"] or 0.0 for plan in plans)

    payload = {
        "phase": "v3-matrix",
        "route": "v3-st",
        "gate": gate_blocks[0],
        "single_variable": single,
        "throughput_evidence": throughput,
        "estimates": {
            **estimates,
            "total_seconds": total,
            "total_wall_clock": fmt_seconds(total),
            "forecast_batches": {"train": TRAIN_BATCHES, "val": VAL_BATCHES},
        },
        "plans": plans,
        "smoke": bool(args.smoke),
        "note": "dry run: nothing is created and no GPU work happens without --execute",
    }
    print_plan(payload, as_json=args.print_json)
    if not args.execute:
        return 0
    return execute_phase(args, phase_data=payload, plans=plans, output_name="phase.json")


def print_plan(payload: dict, *, as_json: bool) -> None:
    """Human table by default; ``--print-json`` emits the same plan as JSON."""
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return
    if payload["phase"] == "v3-matrix":
        print_v3_plan(payload)
    else:
        print_single_plan(payload)


def print_v3_plan(payload: dict) -> None:
    print("=" * 78)
    print("v3-matrix plan (E4 / E5 / E6)")
    print("=" * 78)
    gate = payload["gate"]
    print(f"gate      : {gate['path']} (route={gate['route']}, exit={gate['exit_code']})")
    for name, status in sorted(gate["conditions"].items()):
        print(f"  {name:<32} {status}")
    if gate["failed"]:
        print(f"  waived  : {gate['failed']} by {gate['waived_by']} ({gate['waiver_source']})")
    single = payload["single_variable"]
    print(f"single var: local_enabled (E4 off / E5 on) + policy aux zero (E6) verified"
          f" across {single['compared_fields'].__len__()} compared fields")
    print(f"estimate  : {payload['estimates']['source']}")
    print(f"total     : {payload['estimates']['total_wall_clock']}")
    for plan in payload["plans"]:
        print("-" * 78)
        config = plan["train_config"]
        print(
            f"{plan['experiment']}: arch={config['arch']} depth={config['depth']}"
            f" local_enabled={config['local_enabled']}"
            f" policy_aux=({config['policy_aux_heatmap']},{config['policy_aux_coord']},"
            f"{config['policy_aux_danger']},{config['policy_aux_self']})"
        )
        print(f"  run_dir : {plan['run_dir']}")
        print(f"  epochs  : {config['epochs_state']} perception + {config['epochs_policy']} policy")
        print(f"  estimate: {plan['estimated_wall_clock']}")
        print(f"  recipe  : {plan['recipe_path']} sha256={plan['recipe_sha256'][:16]}")
        print(f"  command : {' '.join(plan['command'])}")


# --------------------------------------------------------------------------- #
# phase: throughput (v3 probe, measurement only)
# --------------------------------------------------------------------------- #
def _fixed_batches(
    dataset: VisionDataset, indices: list[int], count: int, seed: int = SEED, *, drop_last: bool = True
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    batches = list(
        vision_train.make_batches(
            indices,
            batch_size=S37_BATCH_SIZE,
            shuffle=True,
            dataset=dataset,
            rng=rng,
            max_batches=count,
            drop_last=drop_last,
        )
    )
    if len(batches) < count:
        raise v104_error("DATA", f"dataset produced {len(batches)} batches, needed {count}")
    return batches


def _gather_batch(dataset: VisionDataset, index: list[int], depth: int, sizes, device: str):
    """The trainer's own batch construction (same history, same preprocessing)."""
    stacked, labels = vision_train._gather(dataset, index, depth, sizes)  # noqa: SLF001
    batch = regions_to_batch({name: value.numpy() for name, value in stacked.items()}, device)
    return batch, labels


def phase_throughput(args) -> int:
    """10 warm-up + 100 measured batches on the perception and policy-joint paths."""
    import torch
    from touhou_rl import obs

    out_dir = resolve(args.out)
    output_path = out_dir / "throughput.json"
    refuse_existing_output(output_path)
    if not args.execute:
        payload = {
            "phase": "throughput",
            "route": "v3-st",
            "schema": THROUGHPUT_SCHEMA,
            "status": "dry-run",
            "plan": {
                "warmup_batches": 10,
                "measure_batches": 100,
                "val_batches": VAL_BATCHES,
                "train_batches_per_epoch": TRAIN_BATCHES,
                "paths": ["perception", "policy_joint"],
                "recipe": args.recipe,
            },
            "note": "dry run: no GPU work without --execute",
        }
        print(json.dumps(payload["plan"], ensure_ascii=False, indent=2))
        return 0
    device = pick_device(args.device)
    if args.recipe:
        recipe = load_recipe(args.recipe[0])
        config = dict(recipe["train_config"])
        arch = config["arch"]
    else:
        config = {
            "arch": "v3-st",
            "depth": 3,
            "batch_size": S37_BATCH_SIZE,
            "heatmap_sigma": 2.0,
            "local_enabled": True,
            "history_contract": "causal-segment-v1",
            "policy_objective": "rank",
            "rank_weight": 1.0,
            "regression_weight": 1.0,
            "rank_delta_px": 0.5,
            "pf_width": 32,
            "hud_width": 16,
            "trunk_hidden": 256,
        }
        arch = "v3-st"
    if arch != "v3-st":
        raise v104_error("CONFIG", f"--throughput measures the v3 paths; got arch {arch!r}")

    cfg = vision_train.VisionTrainConfig(
        dataset=DATASET_A,
        out_dir=out_dir / "throughput-scratch",
        arch="v3-st",
        stage="state",
        depth=int(config["depth"]),
        batch_size=int(config["batch_size"]),
        heatmap_sigma=float(config["heatmap_sigma"]),
        local_enabled=bool(config["local_enabled"]),
        history_contract=config["history_contract"],
        selection_contract="v104",
        policy_objective=config["policy_objective"],
        rank_weight=float(config["rank_weight"]),
        regression_weight=float(config["regression_weight"]),
        rank_delta_px=float(config["rank_delta_px"]),
        pf_width=int(config.get("pf_width", 32)),
        hud_width=int(config.get("hud_width", 16)),
        trunk_hidden=int(config.get("trunk_hidden", 256)),
        device=device,
    )
    vision_train.validate_config(cfg)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device == "cuda":
        torch.set_num_threads(1)
    model, sizes = vision_train._build_model(cfg, torch.device(device), use_init_from=False)  # noqa: SLF001
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 3e-4)), weight_decay=1e-4)
    dataset = VisionDataset(DATASET_A, history_contract=cfg.history_contract)
    try:
        total = len(dataset)
        train_idx, val_idx, val_sha, verified = vision_train._resolve_split(dataset, cfg, total)  # noqa: SLF001
        if val_sha != VAL_INDEX_SHA256:
            raise v104_error("COVERAGE", f"dataset split digest {val_sha!r} != A's {VAL_INDEX_SHA256!r}")
        train_batches = _fixed_batches(dataset, train_idx, 10 + 100)
        # The validation pass is 76 batches with a 41-frame tail; timing it with
        # drop_last would shave exactly that tail off the measurement.
        val_batches = _fixed_batches(dataset, val_idx, VAL_BATCHES, seed=SEED + 1, drop_last=False)
        mask = vision_train.self_aux_mask(device)
        sigma = float(model.config.heatmap_gauss_sigma)

        def perception_step(index: list[int], training: bool) -> None:
            batch, labels = _gather_batch(dataset, index, cfg.depth, sizes, device)
            count = len(index)
            truth = torch.as_tensor(labels["self_feat"], device=device)
            danger_target = torch.as_tensor(vision_train._danger_target(labels["danger"]), device=device)  # noqa: SLF001
            with torch.set_grad_enabled(training):
                out = model.decode_state(batch)
                grid_h, grid_w = model.config.heatmap_hw
                target_heatmap = torch.as_tensor(
                    _heatmap_target(labels["self_feat"][:, 0], labels["self_feat"][:, 1], grid_h, grid_w, sigma),
                    device=device,
                )
                loss_danger = torch.nn.functional.smooth_l1_loss(
                    out["danger_log"].reshape(count, obs.DANGER_ROWS, obs.DANGER_COLS),
                    danger_target.reshape(count, obs.DANGER_ROWS, obs.DANGER_COLS),
                )
                loss, _parts = vision_train.localisation_losses(
                    heatmap_per_sample=vision_train.heatmap_loss_per_sample(
                        out["heatmap_logits"], target_heatmap
                    ),
                    coord_per_sample=vision_train.coordinate_loss_per_sample(out["self_feat"], truth),
                    self_loss=vision_train.masked_self_loss(out["self_feat"], truth, mask),
                    danger_loss=loss_danger,
                    weights=None,
                    w_heatmap=1.0,
                    w_coord=1.0,
                    w_self=1.0,
                    w_danger=1.0,
                )
                if training:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()

        def policy_step(index: list[int], training: bool) -> None:
            batch, labels = _gather_batch(dataset, index, cfg.depth, sizes, device)
            count = len(index)
            truth = torch.as_tensor(labels["self_feat"], device=device)
            danger_target = torch.as_tensor(vision_train._danger_target(labels["danger"]), device=device)  # noqa: SLF001
            teacher = np.zeros((count, 18), dtype=np.float32)
            with torch.set_grad_enabled(training):
                scores, aux = vision_train.policy_forward(model, batch)
                teacher_px = torch.as_tensor(teacher, device=device)
                loss_policy, _stats = vision_losses.policy_loss(
                    scores, teacher_px, "rank", 0.5, 1.0, 1.0
                )
                grid_h, grid_w = model.config.heatmap_hw
                target_heatmap = torch.as_tensor(
                    _heatmap_target(labels["self_feat"][:, 0], labels["self_feat"][:, 1], grid_h, grid_w, sigma),
                    device=device,
                )
                loss_danger = torch.nn.functional.smooth_l1_loss(
                    aux["danger_log"].reshape(count, obs.DANGER_ROWS, obs.DANGER_COLS),
                    danger_target.reshape(count, obs.DANGER_ROWS, obs.DANGER_COLS),
                )
                aux_loss, _parts = vision_train.localisation_losses(
                    heatmap_per_sample=vision_train.heatmap_loss_per_sample(
                        aux["heatmap_logits"], target_heatmap
                    ),
                    coord_per_sample=vision_train.coordinate_loss_per_sample(aux["self_feat"], truth),
                    self_loss=vision_train.masked_self_loss(aux["self_feat"], truth, mask),
                    danger_loss=loss_danger,
                    weights=None,
                    w_heatmap=0.1,
                    w_coord=1.0,
                    w_self=0.1,
                    w_danger=0.2,
                )
                loss = loss_policy + aux_loss
                if training:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()

        def measure(step, name: str) -> dict:
            import torch as _torch

            for index in train_batches[:10]:
                step(index, True)
            if device == "cuda":
                _torch.cuda.synchronize()
            started = time.perf_counter()
            for index in train_batches[10:]:
                step(index, True)
            if device == "cuda":
                _torch.cuda.synchronize()
            train_seconds = (time.perf_counter() - started) / 100.0
            model.eval()
            with torch.no_grad():
                # Warm the validation frames once: their first decode is a one-time
                # page/decode-cache fill that 設計 §5.1 excludes from the epoch
                # comparison (S3-3/S3-4 did the same for the train batches).
                for index in val_batches:
                    step(index, False)
                if device == "cuda":
                    _torch.cuda.synchronize()
                started = time.perf_counter()
                for index in val_batches:
                    step(index, False)
                if device == "cuda":
                    _torch.cuda.synchronize()
                val_seconds = (time.perf_counter() - started) / len(val_batches)
            model.train()
            forecast = train_seconds * TRAIN_BATCHES + val_seconds * VAL_BATCHES
            return {
                "name": name,
                "train_ms_per_batch": round(train_seconds * 1000.0, 3),
                "val_ms_per_batch": round(val_seconds * 1000.0, 3),
                "forecast_epoch_seconds": round(forecast, 2),
                "forecast_formula": f"{TRAIN_BATCHES}*train + {VAL_BATCHES}*val",
                "warmup_batches": 10,
                "measured_batches": 100,
                "val_batches": len(val_batches),
                "within_budget_60s": bool(forecast <= BUDGET_SECONDS),
                "overrun_fraction": round(forecast / BUDGET_SECONDS - 1.0, 4),
            }

        if device == "cuda":
            import torch as _torch

            _torch.cuda.reset_peak_memory_stats()
        perception = measure(perception_step, "perception")
        policy = measure(policy_step, "policy_joint")
        import torch as _torch

        payload = {
            "schema": THROUGHPUT_SCHEMA,
            "phase": "throughput",
            "route": "v3-st",
            "status": "done",
            "created_at": now_iso(),
            "command": " ".join([sys.executable, *sys.argv]),
            "cwd": str(ROOT),
            "environment": env_block(),
            "device": device,
            "arch": arch,
            "depth": int(cfg.depth),
            "local_enabled": bool(cfg.local_enabled),
            "parameters": int(model.num_parameters()),
            "peak_memory_bytes": int(_torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
            "dataset": rel(DATASET_A),
            "val_index_sha256": val_sha,
            "manifest_verified": verified,
            "paths": {"perception": perception, "policy_joint": policy},
            "epochs": {"train_batches": TRAIN_BATCHES, "val_batches": VAL_BATCHES},
            "budget_seconds": BUDGET_SECONDS,
            "gate_implication": (
                "Q6 §3: if the v3 epoch is clearly above v2's 72 s (e.g. >100 s), stop and"
                " report the numbers before any formal v3 run"
            ),
        }
        write_json(output_path, payload)
        print(json.dumps(payload["paths"], ensure_ascii=False, indent=2))
        print(f"[refine] throughput written to {rel(output_path)}")
        return 0
    finally:
        dataset.close()


def _heatmap_target(rx, ry, grid_h, grid_w, sigma):
    """Local import of the shared target (kept out of the module import graph)."""
    from touhou_rl.vmodels2 import heatmap_target as target

    return target(rx, ry, grid_h, grid_w, sigma=sigma)


# --------------------------------------------------------------------------- #
# phase: hard-frames (R2's error source)
# --------------------------------------------------------------------------- #
def phase_hard_frames(args) -> int:
    if not args.source_state:
        raise v104_error("CONFIG", "--phase hard-frames needs --source-state <frozen perception checkpoint>")
    if args.history_contract not in vision_train.HISTORY_CONTRACTS:
        raise v104_error(
            "CONFIG",
            f"--history-contract must be one of {list(vision_train.HISTORY_CONTRACTS)},"
            f" got {args.history_contract!r}",
        )
    source_path = resolve(args.source_state)
    if not source_path.is_file():
        raise v104_error("CHECKPOINT", f"--source-state {args.source_state} does not exist")
    if any(name in {part.lower() for part in source_path.parts} for name in FROZEN_RUN_NAMES):
        # Reading is allowed (v2's R2 baseline is exactly the read-only old run);
        # writing is not, and this phase never writes into the run directory.
        pass
    out_dir = resolve(args.out)
    npz_path = out_dir / "hard-frames-a-train.npz"
    identity_path = out_dir / "hard-frames-a-train.json"
    source_sha = sha256_file(source_path)
    #: The error pass must stack frames exactly like the R2 run will, so the
    #: *declared* input contract wins over whatever the source checkpoint was
    #: trained with (that older contract is still recorded, see `source_...`).
    stack_contract = args.history_contract
    dataset = VisionDataset(DATASET_A, history_contract=stack_contract)
    try:
        total = len(dataset)
        manifest = DatasetManifest.load(DATASET_A / "manifest.json")
        train_idx, val_idx = manifest.verify(DATASET_A)
        val_sha = str(manifest.splits["val_index_sha256"])
        if val_sha != VAL_INDEX_SHA256:
            raise v104_error("DATA", f"dataset A split digest {val_sha!r} != {VAL_INDEX_SHA256!r}")
        if len(train_idx) != TRAIN_FRAMES or len(val_idx) != VAL_FRAMES:
            raise v104_error(
                "DATA",
                f"dataset A split is {len(train_idx)}/{len(val_idx)}, expected {TRAIN_FRAMES}/{VAL_FRAMES}",
            )
    finally:
        dataset.close()

    plan = {
        "schema": HARDFRAMES_SCHEMA,
        "phase": "hard-frames",
        "route": args.route,
        "source_state": rel(source_path),
        "source_state_sha256": source_sha,
        "dataset": rel(DATASET_A),
        "val_index_sha256": val_sha,
        "train_frames": len(train_idx),
        "val_frames": len(val_idx),
        "outputs": {"npz": rel(npz_path), "identity": rel(identity_path)},
        "thresholds": list(vision_train.POSITION_WEIGHT_THRESHOLDS),
        "weight_values": list(vision_train.POSITION_WEIGHT_VALUES),
        "note": (
            "v2 route: read-only v104-a01-72ep/state.pt as the error source (independent recipe)."
            if args.route == "v2"
            else "v3 route: this architecture's own frozen perception state (independent recipe)."
        ),
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print("[refine] dry run: nothing computed without --execute")
        return 0

    import torch

    device = pick_device(args.device)
    model, extra = load_vision_checkpoint(source_path, map_location="cpu")
    arch = vision_arch(model)
    if args.route != arch:
        raise v104_error(
            "CHECKPOINT", f"--route {args.route} but the checkpoint's arch is {arch!r}"
        )
    model.to(device).eval()
    source_history = str(extra.get("history_contract") or "legacy")
    dataset = VisionDataset(DATASET_A, history_contract=stack_contract)
    sizes = model.config.region_sizes
    depth = int(model.config.depth)
    errors = np.zeros(len(train_idx), dtype=np.float32)
    started = time.perf_counter()
    try:
        batches = list(
            vision_train.make_batches(
                train_idx,
                batch_size=S37_BATCH_SIZE,
                shuffle=False,
                max_batches=0,
                drop_last=False,
            )
        )
        position = 0
        with torch.no_grad():
            for index in batches:
                stacked, labels = vision_train._gather(dataset, index, depth, sizes)  # noqa: SLF001
                batch = regions_to_batch({name: value.numpy() for name, value in stacked.items()}, device)
                state = model.decode_state(batch)
                predicted = state["self_feat"][:, :2].detach().cpu().numpy().astype(np.float64)
                truth = labels["self_feat"][:, :2].astype(np.float64)
                if not np.isfinite(predicted).all():
                    raise v104_error("DATA", "the frozen checkpoint produced non-finite positions")
                residual = (
                    np.abs(predicted[:, 0] - truth[:, 0]) * 384.0
                    + np.abs(predicted[:, 1] - truth[:, 1]) * 448.0
                ) / 2.0
                errors[position : position + len(index)] = residual.astype(np.float32)
                position += len(index)
        if position != len(train_idx):
            raise v104_error("COVERAGE", f"scored {position} train frames, expected {len(train_idx)}")
    finally:
        dataset.close()
    if not np.isfinite(errors).all():
        raise v104_error("DATA", "per-frame errors contain non-finite values")
    if (errors < 0).any():
        raise v104_error("DATA", "per-frame errors contain negative values")
    weights = vision_train.position_weight_values(errors)
    train_indices = np.asarray(train_idx, dtype=np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, train_indices=train_indices, errors=errors, weights=weights)
    npz_sha = sha256_file(npz_path)
    histogram = {
        str(value): int(np.count_nonzero(weights == value)) for value in vision_train.POSITION_WEIGHT_VALUES
    }
    identity = {
        "schema": vision_train.POSITION_WEIGHTS_SCHEMA,
        "created_at": now_iso(),
        "producer": "tools/rl/refine_v104.py --phase hard-frames",
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(ROOT),
        "route": arch,
        "phase": "hard-frames",
        "dataset": rel(DATASET_A),
        "manifest_sha256": sha256_file(DATASET_A / "manifest.json"),
        "val_index_sha256": val_sha,
        "val_frames": len(val_idx),
        "source_state_checkpoint": rel(source_path),
        "source_state_sha256": source_sha,
        "source_state_epoch": extra.get("epoch"),
        "history_contract": stack_contract,
        "source_history_contract": source_history,
        "depth": depth,
        "roi": {name: [int(size[0]), int(size[1])] for name, size in sizes.items()},
        "frames": int(len(train_indices)),
        "train_indices_sha256": index_digest(sorted(int(index) for index in train_indices)),
        "npz": rel(npz_path),
        "npz_sha256": npz_sha,
        "error_metric": "(|dx|*384 + |dy|*448)/2 native px, model position vs privileged label",
        "thresholds": list(vision_train.POSITION_WEIGHT_THRESHOLDS),
        "weight_values": list(vision_train.POSITION_WEIGHT_VALUES),
        "error_mean_px": float(errors.astype(np.float64).mean()),
        "error_median_px": float(np.median(errors.astype(np.float64))),
        "error_p90_px": float(np.percentile(errors.astype(np.float64), 90)),
        "error_max_px": float(errors.astype(np.float64).max()),
        "frame_counts": histogram,
        "weight_sum": float(weights.astype(np.float64).sum()),
        "indices_are_train_only": True,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "device": device,
        "note": plan["note"],
    }
    write_json(identity_path, identity)
    # Self-check: reload through the trainer's loader, which is the authority.
    reloaded = vision_train.load_position_sample_weights(
        npz_path,
        train_indices=train_idx,
        val_indices=val_idx,
        dataset_frames=total,
        region_sizes={name: tuple(size) for name, size in sizes.items()},
        history_contract=stack_contract,
        source_state_sha256=source_sha,
    )
    check = {
        "npz_sha256": npz_sha,
        "identity_sha256": sha256_file(identity_path),
        "npz_bytes": int(npz_path.stat().st_size),
        "reload_ok": True,
        "reload_frame_counts": reloaded["frame_counts"],
        "weights_dtype": str(weights.dtype),
        "indices_dtype": str(train_indices.dtype),
        "unique_indices": int(len(set(int(index) for index in train_indices))),
        "no_duplicates": bool(len(set(int(index) for index in train_indices)) == len(train_indices)),
        "no_missing": bool(set(int(index) for index in train_indices) == set(int(index) for index in train_idx)),
        "overlaps_validation": int(
            len(set(int(index) for index in train_indices) & set(int(index) for index in val_idx))
        ),
        "all_finite": bool(np.isfinite(errors).all()),
        "thresholds_match": True,
    }
    print(json.dumps({**identity, "check": check}, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# phases: r2 / r3 / integrate / stability
# --------------------------------------------------------------------------- #
def _single_recipe_plan(args, *, route: str, phase: str) -> dict:
    if not args.recipe or len(args.recipe) != 1:
        raise v104_error("CONFIG", f"--phase {phase} needs exactly one --recipe")
    recipe_path = args.recipe[0]
    recipe = load_recipe(recipe_path)
    plan = validate_recipe(recipe, path=recipe_path, route=route, phase=phase)
    return plan


#: C1/C5 model input ROI (identical for v2 and v3): the R2 identity binds these.
MODEL_REGION_SIZES = {
    "playfield": (224, 192),
    "hud_score": (34, 216),
    "hud_status": (126, 216),
}


def _validated_weight_block(config: dict, *, source_sha: str | None) -> dict:
    """Run the *trainer's* loader on the recipe's weight table, at plan time.

    ``r2`` must carry a valid ``position_sample_weights``; proving that at plan
    time (instead of after 72 perception epochs) is the point.
    """
    npz_path = resolve(config["position_sample_weights"])
    identity_path = vision_train.position_weight_paths(npz_path)[1]
    if not npz_path.is_file():
        raise v104_error("DATA", f"position_sample_weights {rel(npz_path)} is missing")
    if not identity_path.is_file():
        raise v104_error("DATA", f"{rel(identity_path)} (the weight identity) is missing")
    manifest = DatasetManifest.load(DATASET_A / "manifest.json")
    try:
        train, val = manifest.verify(DATASET_A)
    except ManifestError as error:
        raise v104_error("DATA", f"dataset A does not verify: {error}") from error
    loaded = vision_train.load_position_sample_weights(
        npz_path,
        train_indices=train,
        val_indices=val,
        dataset_frames=len(train) + len(val),
        region_sizes=dict(MODEL_REGION_SIZES),
        history_contract=config["history_contract"],
        source_state_sha256=source_sha,
    )
    return {
        "path": loaded["path"],
        "sha256": loaded["sha256"],
        "identity_path": loaded["identity_path"],
        "identity_sha256": loaded["identity_sha256"],
        "train_frames": loaded["train_frames"],
        "frame_counts": loaded["frame_counts"],
        "errors_mean_px": loaded["errors_mean_px"],
        "errors_p90_px": loaded["errors_p90_px"],
        "reduction": "sum(w*per_sample)/sum(w) on heatmap+coordinate only; validation unweighted",
        "verified_at_plan_time": True,
    }


def _require_route_feature(plan: dict, *, phase: str) -> dict | None:
    """Phase-specific preconditions; returns the weight block when one is used."""
    config = plan["train_config"]
    weights_block = None
    if phase in ("r2", "integrate"):
        if not config.get("position_sample_weights"):
            raise v104_error(
                "CONFIG",
                f"phase {phase} needs a recipe whose train_config.position_sample_weights points"
                " at a validated hard-frames NPZ",
            )
        source_sha = (plan.get("source_state") or {}).get("sha256")
        weights_block = _validated_weight_block(config, source_sha=source_sha)
    if phase == "r3":
        if config["danger_head_variant"] != "learnable-v104":
            raise v104_error(
                "CONFIG", "phase r3 needs danger_head_variant='learnable-v104' in the recipe"
            )
        if config["arch"] != "v2":
            raise v104_error("CONFIG", "the R3 danger-head variant is a v2-only change (C5 keeps v3's head)")
        if config.get("position_sample_weights"):
            raise v104_error("CONFIG", "phase r3 must vary the danger head only; drop position_sample_weights")
    return weights_block


def phase_r2(args) -> int:
    plan = _single_recipe_plan(args, route=args.route, phase="r2")
    weights_block = _require_route_feature(plan, phase="r2")
    baseline = _require_baseline(plan, phase="r2")
    estimates = estimate_epoch_seconds(args.route, load_throughput_evidence(args.throughput_json))
    planned = planned_run(plan, estimates)
    payload = {
        "phase": "r2",
        "route": args.route,
        "plans": [planned],
        "baseline": baseline,
        "position_weights": weights_block,
        "estimates": estimated_block(estimates, [planned]),
        "note": "R2: only the heatmap/coordinate terms are weighted; validation is unweighted",
    }
    print_plan(payload, as_json=args.print_json)
    if not args.execute:
        return 0
    return execute_phase(
        args,
        phase_data=payload,
        plans=[planned],
        output_name="phase.json",
        decorate=lambda result, plan: gain_of("r2", result, baseline),
    )


def phase_r3(args) -> int:
    plan = _single_recipe_plan(args, route=args.route, phase="r3")
    _require_route_feature(plan, phase="r3")
    baseline = _require_baseline(plan, phase="r3")
    estimates = estimate_epoch_seconds(args.route, load_throughput_evidence(args.throughput_json))
    planned = planned_run(plan, estimates)
    payload = {
        "phase": "r3",
        "route": args.route,
        "plans": [planned],
        "baseline": baseline,
        "estimates": estimated_block(estimates, [planned]),
        "note": "R3: danger_head_variant=learnable-v104, every other recipe value fixed",
    }
    print_plan(payload, as_json=args.print_json)
    if not args.execute:
        return 0
    return execute_phase(
        args,
        phase_data=payload,
        plans=[planned],
        output_name="phase.json",
        decorate=lambda result, plan: gain_of("r3", result, baseline),
    )


def _require_baseline(plan: dict, *, phase: str) -> dict:
    """R2/R3 are judged against a declared comparison number, not a vibe."""
    baseline = plan.get("baseline") or {}
    key = "state_pos_px" if phase == "r2" else "danger_corr_perframe_median"
    if baseline.get(key) is None:
        raise v104_error(
            "CONFIG",
            f"phase {phase} needs the recipe's baseline.{key} (the source run's own metric) so the"
            " gain is decided against a recorded number",
        )
    return {
        "metric": "state val_player_pos_px" if phase == "r2" else "state val_danger_corr_perframe_median",
        "value": float(baseline[key]),
        "source": str(baseline.get("source") or ""),
        "recipe_key": f"baseline.{key}",
    }


def gain_of(phase: str, result: dict, baseline: dict) -> dict:
    """The phase's own improvement, computed from its summary and the baseline."""
    state_best = result.get("state_best") or {}
    if phase == "r2":
        value = state_best.get("val_player_pos_px")
        reference = baseline["value"]
        improved = value is not None and float(value) < float(reference)
        return {
            "metric": baseline["metric"],
            "value": value,
            "baseline": reference,
            "baseline_source": baseline["source"],
            "improved": bool(improved),
            "relative_change": (float(value) / float(reference) - 1.0)
            if value is not None and float(reference) != 0.0
            else None,
            "detail": f"state position {value} px vs declared baseline {reference} px",
        }
    value = state_best.get("val_danger_corr_perframe_median")
    reference = baseline["value"]
    improved = value is not None and float(value) > float(reference)
    return {
        "metric": baseline["metric"],
        "value": value,
        "baseline": reference,
        "baseline_source": baseline["source"],
        "improved": bool(improved),
        "absolute_change": (float(value) - float(reference)) if value is not None else None,
        "detail": f"danger per-frame median {value} vs declared baseline {reference}",
    }


def estimated_block(estimates: dict, plans: list[dict]) -> dict:
    total = sum(plan["estimated_seconds"] or 0.0 for plan in plans)
    return {
        **estimates,
        "total_seconds": total,
        "total_wall_clock": fmt_seconds(total if (estimates.get("perception") and estimates.get("policy")) else None),
    }


def print_single_plan(payload: dict) -> None:
    print("=" * 78)
    print(f"{payload['phase']} plan (route {payload['route']})")
    print("=" * 78)
    print(f"estimate: {payload['estimates']['source']}")
    print(f"total   : {payload['estimates']['total_wall_clock']}")
    for plan in payload["plans"]:
        config = plan["train_config"]
        print("-" * 78)
        print(
            f"{plan['experiment']}: arch={config['arch']} depth={config['depth']}"
            f" objective={config['policy_objective']}"
            f" variant={config['danger_head_variant']}"
            f" local_enabled={config['local_enabled']}"
        )
        print(f"  run_dir : {plan['run_dir']}")
        print(f"  epochs  : {config['epochs_state']} perception + {config['epochs_policy']} policy")
        print(f"  estimate: {plan['estimated_wall_clock']}")
        print(f"  recipe  : {plan['recipe_path']} sha256={plan['recipe_sha256'][:16]}")
        if plan["source_state"]:
            print(f"  source  : {plan['source_state']['plan']} sha256={plan['source_state']['sha256'][:16]}")
        print(f"  command : {' '.join(plan['command'])}")


def phase_integrate(args) -> int:
    """One combined run, only after R2 and R3 each helped on their own."""
    if not args.recipe or len(args.recipe) != 1:
        raise v104_error("CONFIG", "--phase integrate needs exactly one --recipe (the combined one)")
    recipe_path = args.recipe[0]
    recipe = load_recipe(recipe_path)
    plan = validate_recipe(recipe, path=recipe_path, route=args.route, phase="integrate")
    config = plan["train_config"]
    if config["danger_head_variant"] != "learnable-v104":
        raise v104_error("CONFIG", "the integrated recipe must carry danger_head_variant='learnable-v104'")
    if not config.get("position_sample_weights"):
        raise v104_error("CONFIG", "the integrated recipe must carry position_sample_weights")
    weights_block = _validated_weight_block(config, source_sha=(plan.get("source_state") or {}).get("sha256"))
    parent_paths = recipe.get("parents") or {}
    gains = {}
    for key in ("r2", "r3"):
        parent = parent_paths.get(key)
        if not parent:
            raise v104_error(
                "EVIDENCE",
                f"the integrated recipe must name parents.{key} (that phase's output JSON) so the"
                " tool can check the individual gain before combining",
            )
        payload = read_json(resolve(parent))
        results = payload.get("results") or []
        if payload.get("status") != "done" or not results:
            raise v104_error("EVIDENCE", f"parents.{key} is not a finished phase output")
        gain = results[-1].get("gain_vs_baseline") or {}
        if not gain.get("improved"):
            raise v104_error(
                "EVIDENCE",
                f"{key.upper()} alone showed no gain ({gain.get('detail')}); R2+R3 must not be"
                " combined without a single-change gain",
            )
        gains[key] = gain
    estimates = estimate_epoch_seconds(args.route, load_throughput_evidence(args.throughput_json))
    planned = planned_run(plan, estimates)
    payload = {
        "phase": "integrate",
        "route": args.route,
        "plans": [planned],
        "position_weights": weights_block,
        "estimates": estimated_block(estimates, [planned]),
        "parents": gains,
        "note": "both changes are declared; the joint run is not a single-variable comparison",
    }
    print_plan(payload, as_json=args.print_json)
    if not args.execute:
        return 0
    return execute_phase(args, phase_data=payload, plans=[planned], output_name="phase.json")


def phase_stability(args) -> int:
    if not args.recipe or len(args.recipe) != 1:
        raise v104_error("CONFIG", "--phase stability needs exactly one --recipe")
    recipe_path = args.recipe[0]
    base = load_recipe(recipe_path)
    plans = []
    for seed in STABILITY_SEEDS:
        variant = json.loads(json.dumps(base))
        variant["train_config"]["seed"] = int(seed)
        base_run = variant.get("run_dir") or ""
        if not base_run:
            raise v104_error("CONFIG", "the stability recipe needs run_dir")
        variant["run_dir"] = f"{base_run}-s{seed}"
        variant["experiment"] = f"{base.get('experiment') or 'stability'}-s{seed}"
        plans.append(
            validate_recipe(
                variant,
                path=recipe_path,
                route=args.route,
                phase="stability",
            )
        )
    estimates = estimate_epoch_seconds(args.route, load_throughput_evidence(args.throughput_json))
    planned = [planned_run(plan, estimates) for plan in plans]
    payload = {
        "phase": "stability",
        "route": args.route,
        "plans": planned,
        "estimates": estimated_block(estimates, planned),
        "note": "seeds 17/27 are stability evidence only; the frozen model stays seed 7",
    }
    print_plan(payload, as_json=args.print_json)
    if not args.execute:
        return 0
    return execute_phase(args, phase_data=payload, plans=planned, output_name="phase.json")


# --------------------------------------------------------------------------- #
# phase: freeze
# --------------------------------------------------------------------------- #
REQUIRED_FLAT = (
    "state_checkpoint",
    "state_checkpoint_sha256",
    "state_summary",
    "policy_checkpoint",
    "policy_checkpoint_sha256",
    "policy_summary",
    "teacher_cache",
    "teacher_fingerprint",
    "manifest_a",
    "manifest_a_sha256",
    "manifest_b",
    "manifest_b_sha256",
    "val_index_sha256",
    "arch",
    "depth",
    "seed",
    "experiment_table",
    "selection_reason",
    "frozen_at",
    "external_guard_uses",
)


def _summary_best(summary: dict, stage: str, *, where: str, summary_path: Path) -> dict:
    """The selected checkpoint of one stage, resolved the way the trainer records it.

    ``stages.<stage>.best.checkpoint`` is the *file name* inside the run directory
    (S3-3's contract), so a bare name is resolved against the summary's directory;
    a value that already carries a path is resolved from the workspace root.
    """
    if summary.get("schema") != "vision-summary/2":
        raise v104_error("EVIDENCE", f"{where}: schema {summary.get('schema')!r} != 'vision-summary/2'")
    if summary.get("contract_version") != "v104-A/1":
        raise v104_error(
            "EVIDENCE", f"{where}: contract_version {summary.get('contract_version')!r} != 'v104-A/1'"
        )
    best = ((summary.get("stages") or {}).get(stage) or {}).get("best") or {}
    if not best.get("checkpoint") or not best.get("checkpoint_sha256"):
        raise v104_error(
            "EVIDENCE",
            f"{where}: stages.{stage}.best has no checkpoint/checkpoint_sha256; there is nothing"
            " honest to freeze (best.pt is never guessed)",
        )
    recorded = str(best["checkpoint"]).replace("\\", "/")
    if Path(recorded).is_absolute():
        checkpoint = Path(recorded)
    elif "/" in recorded:
        checkpoint = resolve(recorded)
    else:
        checkpoint = Path(summary_path).parent / recorded
    if not checkpoint.is_file():
        raise v104_error("CHECKPOINT", f"{where}: {recorded} does not exist under {rel(Path(summary_path).parent)}")
    actual = sha256_file(checkpoint)
    if actual != best["checkpoint_sha256"]:
        raise v104_error(
            "CHECKPOINT",
            f"{where}: {recorded} hashes to {actual}, the summary records"
            f" {best['checkpoint_sha256']}",
        )
    return {"row": best, "path": checkpoint, "sha256": actual}


def _checkpoint_identity(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != "vision_scorer":
        raise v104_error("CHECKPOINT", f"{rel(path)} is not a vision checkpoint")
    arch = payload.get("arch") or "v1"
    config = payload.get("config") or {}
    depth = config.get("depth") if isinstance(config, dict) else None
    return {"arch": arch, "depth": depth, "extra": payload.get("extra") or {}}


def _manifest_dataset_dir(manifest, manifest_path: Path) -> Path:
    """Where a manifest's shards live.

    A's manifest sits inside its dataset directory, but B's *evaluation* manifest
    was written to an evidence directory (S3-1 built it read-only from the
    existing shards), so the shard directory comes from ``manifest.root``.
    """
    root = str(getattr(manifest, "root", "") or "")
    if root:
        candidate = Path(root)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.is_dir() and any(candidate.glob("*.npz")):
            return candidate
    return Path(manifest_path).parent


def phase_freeze(args) -> int:
    out_dir = resolve(args.out) if args.out else STAGE3_DIR
    target = out_dir / "final-artifacts.json"
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict) and existing.get("schema") == FINAL_SCHEMA and all(
            key in existing for key in REQUIRED_FLAT
        ):
            raise v104_error(
                "EVIDENCE",
                f"{rel(target)} already holds a valid {FINAL_SCHEMA}; refusing to overwrite the"
                " frozen list (a re-freeze needs a new revision, not a silent edit)",
            )
    required = {
        "--state-summary": args.state_summary,
        "--policy-summary": args.policy_summary,
        "--teacher-cache": args.teacher_cache,
        "--manifest-b": args.manifest_b,
        "--experiment-table": args.experiment_table,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise v104_error("CONFIG", f"freeze is missing required input(s) {missing}")

    state_path = resolve(args.state_summary)
    policy_path = resolve(args.policy_summary)
    state_summary = read_json(state_path)
    policy_summary = read_json(policy_path)
    state_best = _summary_best(state_summary, "state", where=rel(state_path), summary_path=state_path)
    policy_best = _summary_best(
        policy_summary, "policy", where=rel(policy_path), summary_path=policy_path
    )

    state_identity = _checkpoint_identity(state_best["path"])
    policy_identity = _checkpoint_identity(policy_best["path"])
    for label, identity, summary, where in (
        ("state", state_identity, state_summary, rel(state_path)),
        ("policy", policy_identity, policy_summary, rel(policy_path)),
    ):
        if identity["arch"] != summary.get("arch"):
            raise v104_error(
                "CHECKPOINT",
                f"{where}: checkpoint arch {identity['arch']!r} != summary arch {summary.get('arch')!r}",
            )
        if int(identity["depth"]) != int(summary.get("depth")):
            raise v104_error(
                "CHECKPOINT",
                f"{where}: checkpoint depth {identity['depth']!r} != summary depth {summary.get('depth')!r}",
            )
    if state_identity["arch"] != policy_identity["arch"]:
        raise v104_error(
            "CHECKPOINT",
            f"state arch {state_identity['arch']!r} != policy arch {policy_identity['arch']!r}",
        )
    arch = str(state_identity["arch"])
    depth = int(state_identity["depth"])

    seed_state = (state_summary.get("recipe") or {}).get("seed")
    seed_policy = (policy_summary.get("recipe") or {}).get("seed")
    seed = args.seed if args.seed is not None else seed_policy
    if seed != SEED:
        raise v104_error(
            "EVIDENCE",
            f"only the pre-selected seed {SEED} may be frozen (stability seeds 17/27 stay evidence)",
        )
    if seed_state is not None and int(seed_state) != int(seed):
        raise v104_error("EVIDENCE", f"state summary seed {seed_state} != policy summary seed {seed}")
    if seed_policy is not None and int(seed_policy) != int(seed):
        raise v104_error("EVIDENCE", f"policy summary seed {seed_policy} != {seed}")
    if int(args.external_guard_uses) != 0:
        raise v104_error(
            "EVIDENCE",
            f"external_guard_uses={args.external_guard_uses}; S3-7 freezes with 0 external guard uses",
        )

    val_state = (state_summary.get("data") or {}).get("val_index_sha256")
    val_policy = (policy_summary.get("data") or {}).get("val_index_sha256")
    if not val_state or not val_policy:
        raise v104_error("EVIDENCE", "both summaries must record data.val_index_sha256")
    if val_state != val_policy:
        raise v104_error(
            "DATA", f"state/policy validation digests differ ({val_state} != {val_policy})"
        )
    if val_state != VAL_INDEX_SHA256:
        raise v104_error(
            "DATA", f"validation digest {val_state!r} is not A's fixed {VAL_INDEX_SHA256!r}"
        )

    manifest_a_path = resolve(args.manifest_a) if args.manifest_a else DATASET_A / "manifest.json"
    manifest_b_path = resolve(args.manifest_b)
    for label, path in (("A", manifest_a_path), ("B", manifest_b_path)):
        if not path.is_file():
            raise v104_error("DATA", f"manifest {label} not found: {rel(path)}")
    manifest_a = DatasetManifest.load(manifest_a_path)
    try:
        _train_a, val_a = manifest_a.verify(_manifest_dataset_dir(manifest_a, manifest_a_path))
    except ManifestError as error:
        raise v104_error("DATA", f"manifest A does not verify: {error}") from error
    if index_digest(val_a) != val_state:
        raise v104_error(
            "DATA",
            f"manifest A's split digest {index_digest(val_a)!r} != the frozen {val_state!r}",
        )
    if len(val_a) != VAL_FRAMES:
        raise v104_error("DATA", f"manifest A holds {len(val_a)} validation frames, expected {VAL_FRAMES}")
    manifest_b = DatasetManifest.load(manifest_b_path)
    try:
        manifest_b.verify(_manifest_dataset_dir(manifest_b, manifest_b_path))
    except ManifestError as error:
        raise v104_error("DATA", f"manifest B does not verify: {error}") from error

    teacher_cache = resolve(args.teacher_cache)
    if not teacher_cache.is_file():
        raise v104_error("TEACHER", f"teacher cache not found: {rel(teacher_cache)}")
    teacher_block = policy_summary.get("teacher") or state_summary.get("teacher") or {}
    fingerprint = teacher_block.get("fingerprint") or args.teacher_fingerprint
    if not fingerprint:
        raise v104_error("TEACHER", "the summaries record no teacher fingerprint")
    recorded_cache = teacher_block.get("cache")
    if recorded_cache and rel(teacher_cache) != str(recorded_cache):
        raise v104_error(
            "TEACHER",
            f"--teacher-cache {rel(teacher_cache)} != the summary's {recorded_cache}",
        )
    sidecar = teacher_cache.with_suffix(".json")
    if sidecar.is_file():
        sidecar_payload = read_json(sidecar)
        sidecar_fp = sidecar_payload.get("fingerprint") or sidecar_payload.get("identity_sha256")
        if sidecar_fp and sidecar_fp != fingerprint:
            raise v104_error(
                "TEACHER",
                f"teacher fingerprint {fingerprint!r} != the cache sidecar's {sidecar_fp!r}",
            )
    elif teacher_block.get("cache_sha256") and teacher_block["cache_sha256"] != sha256_file(teacher_cache):
        raise v104_error("TEACHER", "teacher cache bytes do not match the summary's recorded sha256")

    table_path = resolve(args.experiment_table)
    if not table_path.is_file():
        raise v104_error("EVIDENCE", f"experiment table not found: {rel(table_path)}")
    selection_reason = args.selection_reason or ""
    if args.selection_reason_path:
        selection_reason = resolve(args.selection_reason_path).read_text(encoding="utf-8").strip()
    if not selection_reason:
        raise v104_error(
            "EVIDENCE",
            "freeze needs --selection-reason (or --selection-reason-path): the candidate's raw"
            " metrics, budget and rejected alternatives have to be written down",
        )

    unmet = readiness_items(state_summary, policy_summary)
    payload = {
        "schema": FINAL_SCHEMA,
        "state_checkpoint": rel(state_best["path"]),
        "state_checkpoint_sha256": state_best["sha256"],
        "state_summary": rel(state_path),
        "policy_checkpoint": rel(policy_best["path"]),
        "policy_checkpoint_sha256": policy_best["sha256"],
        "policy_summary": rel(policy_path),
        "teacher_cache": rel(teacher_cache),
        "teacher_fingerprint": fingerprint,
        "manifest_a": rel(manifest_a_path),
        "manifest_a_sha256": sha256_file(manifest_a_path),
        "manifest_b": rel(manifest_b_path),
        "manifest_b_sha256": sha256_file(manifest_b_path),
        "val_index_sha256": val_state,
        "arch": arch,
        "depth": depth,
        "seed": int(seed),
        "experiment_table": rel(table_path),
        "selection_reason": selection_reason,
        "frozen_at": now_iso(),
        "external_guard_uses": 0,
        "evidence": {},
        # Optional, never a substitute for a required key (設計 §6).
        "state_run": rel(state_best["path"].parent),
        "policy_run": rel(policy_best["path"].parent),
        "readiness": {
            "status": "hard-lines-met" if not unmet else "candidate-unmet-hard-lines",
            "unmet": unmet,
            "checked_at": now_iso(),
            "a01_state": {
                "checkpoint_sha256": state_best["sha256"],
                "player_pos_px": (state_best["row"] or {}).get("val_player_pos_px"),
                "danger_corr_perframe_median": (state_best["row"] or {}).get(
                    "val_danger_corr_perframe_median"
                ),
                "selection_rule": (state_best["row"] or {}).get("selection_rule"),
            },
            "a02_policy": {
                "checkpoint_sha256": policy_best["sha256"],
                "bucket_0_15_regret_px": (policy_best["row"] or {}).get("val_regret_px_bucket_0_15"),
                "global_regret_px": (policy_best["row"] or {}).get("val_regret_px"),
                "selection_rule": (policy_best["row"] or {}).get("selection_rule"),
            },
            "note": "freezing a below-threshold candidate is allowed; it is not a claim that v1.0.4 passed",
        },
    }
    missing_flat = [key for key in REQUIRED_FLAT if key not in payload]
    if missing_flat:
        raise v104_error("EVIDENCE", f"internal: frozen list is missing {missing_flat}")
    if not args.execute:
        print(json.dumps({key: payload[key] for key in REQUIRED_FLAT}, ensure_ascii=False, indent=2))
        print(
            f"[refine] dry run: would freeze {payload['arch']} depth {depth} seed {seed} into"
            f" {rel(target)} (readiness: {payload['readiness']['status']}); pass --execute to write"
        )
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(target, payload)
    print(json.dumps({key: payload[key] for key in REQUIRED_FLAT}, ensure_ascii=False, indent=2))
    print(f"[refine] frozen list written to {rel(target)} (readiness: {payload['readiness']['status']})")
    return 0


def readiness_items(state_summary: dict, policy_summary: dict) -> list[str]:
    """Which acceptance lines the frozen candidate does *not* meet (honest labels)."""
    unmet = []
    state_best = ((state_summary.get("stages") or {}).get("state") or {}).get("best") or {}
    position = state_best.get("val_player_pos_px")
    danger = state_best.get("val_danger_corr_perframe_median")
    if position is None or float(position) >= A01_POS_THRESHOLD:
        unmet.append(f"A-01 position {position} px >= {A01_POS_THRESHOLD} px")
    if danger is None or float(danger) <= A01_DANGER_THRESHOLD:
        unmet.append(f"A-01 danger per-frame median {danger} <= {A01_DANGER_THRESHOLD}")
    policy_best = ((policy_summary.get("stages") or {}).get("policy") or {}).get("best") or {}
    bucket = policy_best.get("val_regret_px_bucket_0_15")
    if bucket is None or float(bucket) >= A02_REGRET_THRESHOLD:
        unmet.append(f"A-02 {A02_BUCKET} bucket regret {bucket} px >= {A02_REGRET_THRESHOLD} px")
    return unmet


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="v1.0.4 S3-7 refinement, hard-frame weighting and the final freeze",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["v3-matrix", "hard-frames", "r2", "r3", "integrate", "stability", "freeze"],
    )
    parser.add_argument("--route", default="v2", choices=["v2", "v3-st"], help="model route (default v2)")
    parser.add_argument(
        "--out",
        default="",
        help="phase output directory (created only with --execute); freeze defaults to Stage3/",
    )
    parser.add_argument(
        "--recipe",
        action="append",
        default=None,
        help="complete recipe JSON; v3-matrix needs one per E4/E5/E6, the other phases exactly one",
    )
    parser.add_argument("--execute", action="store_true", help="actually run; the default is dry run")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="print the dry-run plan as JSON (same payload the phase would write)",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--gate", default="", help="v3 phases: the written G-v3 verdict (JSON)")
    parser.add_argument("--source-state", default="", help="hard-frames: the frozen perception checkpoint")
    parser.add_argument(
        "--history-contract",
        default="causal-segment-v1",
        choices=["legacy", "causal-segment-v1"],
        help="hard-frames: the stacking used for the error pass (must match the R2 run)",
    )
    parser.add_argument("--throughput", action="store_true", help="v3-matrix: measure the v3 throughput paths")
    parser.add_argument("--throughput-json", default="", help="use this throughput.json for the estimates")
    parser.add_argument("--smoke", action="store_true", help="v3-matrix: 1 perception + 1 policy epoch per config")
    parser.add_argument("--seed", type=int, default=None, help="freeze: the frozen seed (must be 7)")
    parser.add_argument("--state-summary", default="", help="freeze: perception stage summary")
    parser.add_argument("--policy-summary", default="", help="freeze: policy stage summary")
    parser.add_argument("--teacher-cache", default="", help="freeze: the fingerprinted teacher npz")
    parser.add_argument("--teacher-fingerprint", default="", help="freeze: override only if the summary has none")
    parser.add_argument("--manifest-a", default="", help="freeze: A manifest (default dataset A's)")
    parser.add_argument("--manifest-b", default="", help="freeze: B evaluation manifest")
    parser.add_argument("--experiment-table", default="", help="freeze: the experiment table JSON")
    parser.add_argument("--selection-reason", default="", help="freeze: why this candidate was selected")
    parser.add_argument("--selection-reason-path", default="", help="freeze: read that text from a file")
    parser.add_argument("--external-guard-uses", type=int, default=0, help="freeze: must be 0")
    return parser


def smoke_recipe_overrides(recipe: dict) -> dict:
    """The documented smoke shape: 1+1 epochs, 2 train batches, full validation."""
    variant = json.loads(json.dumps(recipe))
    variant["smoke"] = True
    config = variant["train_config"]
    config["epochs_state"] = 1 if int(config["epochs_state"]) else 0
    config["epochs_policy"] = 1 if int(config["epochs_policy"]) else 0
    config["max_train_batches"] = 2
    config["max_val_batches"] = 0
    config["policy_freeze_epochs"] = 1
    base_run = str(variant.get("run_dir") or "")
    if base_run and "-smoke" not in base_run:
        variant["run_dir"] = f"{base_run}-smoke"
    return variant


def run_phase(args) -> int:
    if args.phase != "v3-matrix" and args.throughput:
        raise v104_error("CONFIG", "--throughput is only valid with --phase v3-matrix")
    if args.phase != "v3-matrix" and args.smoke:
        raise v104_error("CONFIG", "--smoke is only valid with --phase v3-matrix")
    if args.phase == "v3-matrix" and args.route != "v3-st":
        raise v104_error("CONFIG", "--phase v3-matrix is the v3-st matrix; pass --route v3-st")
    if args.phase in ("v3-matrix", "hard-frames", "r2", "r3", "integrate", "stability") and not args.out:
        raise v104_error("CONFIG", f"--phase {args.phase} needs --out")
    if args.phase == "v3-matrix" and args.smoke:
        if not args.execute:
            raise v104_error(
                "CONFIG",
                "--smoke writes smoke recipes and run directories, so it needs --execute"
                " (the dry run never writes)",
            )
        if not args.recipe:
            raise v104_error("CONFIG", "--smoke still needs the three recipes (they define the config)")
        recipes = [load_recipe(path) for path in args.recipe]
        smoke_paths = []
        for path, recipe in zip(args.recipe, recipes):
            variant = smoke_recipe_overrides(recipe)
            smoke_path = resolve(args.out) / "recipes" / f"{Path(str(path)).stem}-smoke.json"
            write_json(smoke_path, variant)
            smoke_paths.append(str(smoke_path))
        print(f"[refine] smoke recipes written under {rel(resolve(args.out) / 'recipes')}")
        args.recipe = smoke_paths
    if args.phase == "v3-matrix":
        return phase_v3_matrix(args)
    if args.phase == "hard-frames":
        return phase_hard_frames(args)
    if args.phase == "r2":
        return phase_r2(args)
    if args.phase == "r3":
        return phase_r3(args)
    if args.phase == "integrate":
        return phase_integrate(args)
    if args.phase == "stability":
        return phase_stability(args)
    return phase_freeze(args)


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
