"""Write the S3-7 recipes (one JSON per phase) so every zone is pre-declared.

Run with the workspace interpreter after the tool exists:

    python Document/Update/Alpha/v1.0.4/Stage3/S3-7/evidence/make_recipes.py

The recipes are *inputs*, not evidence: they are written once and then frozen by
their own sha256 (recorded in the completion report and the dry-run table).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
S37 = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-7"
RECIPES = S37 / "recipes"
RUNS = "third_party/th08/build/rl/runs"
DATASET = "third_party/th08/build/rl/datasets/pilot-ud0059-v2"
GATE = "Document/Update/Alpha/v1.0.4/Stage3/S3-4/gate-v3.json"
HARD_FRAMES = "Document/Update/Alpha/v1.0.4/Stage3/S3-7/evidence/hard-frames/hard-frames-a-train.npz"

SCHEMA = "v104-refine-recipe/1"

#: C3: perception auxiliary weights are 1/1/1/1 in every configuration.
PERCEPTION_AUX = {"w_heatmap": 1.0, "w_coord": 1.0, "w_danger": 1.0, "w_self": 1.0}
#: C3: standard policy auxiliary weights.
POLICY_AUX = {
    "policy_aux_heatmap": 0.1,
    "policy_aux_coord": 1.0,
    "policy_aux_danger": 0.2,
    "policy_aux_self": 0.1,
}


def base_config(**overrides) -> dict:
    block = {
        "arch": "v2",
        "stage": "both",
        "depth": 3,
        "seed": 7,
        "batch_size": 96,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "epochs_state": 72,
        "epochs_policy": 12,
        "max_train_batches": 0,
        "max_val_batches": 0,
        "selection_contract": "v104",
        "history_contract": "causal-segment-v1",
        # E* (E3c) objective: rank 1.0 + centered-regression 1.0 (S3-4 §10.1).
        "policy_objective": "rank",
        "rank_delta_px": 0.5,
        "rank_weight": 1.0,
        "regression_weight": 1.0,
        "policy_freeze_epochs": 2,
        "encoder_lr": 1e-4,
        "policy_head_lr": 3e-4,
        # v2 recipe sigma is 1.0 (C4 "v2 sigma 1"), v3 records 2.0 (C5).
        "heatmap_sigma": 1.0,
        "local_enabled": True,
        "danger_head_variant": "legacy",
        **PERCEPTION_AUX,
        **POLICY_AUX,
    }
    block.update(overrides)
    return block


def v3_config(**overrides) -> dict:
    block = base_config(arch="v3-st", heatmap_sigma=2.0)
    block.update(overrides)
    return block


def recipe(phase: str, experiment: str, run_dir: str, config: dict, **extra) -> dict:
    payload = {
        "schema": SCHEMA,
        "phase": phase,
        "experiment": experiment,
        "route": config["arch"],
        "dataset": DATASET,
        "run_dir": run_dir,
        "train_config": config,
        "notes": "",
    }
    payload.update(extra)
    return payload


def write(name: str, payload: dict) -> Path:
    RECIPES.mkdir(parents=True, exist_ok=True)
    path = RECIPES / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    gate = {
        "path": GATE,
        "waiver": {
            "by": "Q6",
            "items": ["throughput_le_60s"],
            "source": "Document/Update/Alpha/v1.0.4/问题清单.md §8 Q6（用户裁决进入 v3）",
        },
    }
    written = []
    # E4/E5/E6: one declared variable each (local_enabled, then E6's policy aux).
    for name, local, zero_policy_aux in (
        ("E4", False, False),
        ("E5", True, False),
        ("E6", True, True),
    ):
        config = v3_config(local_enabled=local)
        if zero_policy_aux:
            for key in POLICY_AUX:
                config[key] = 0.0
        written.append(
            write(
                f"{name}.json",
                recipe("v3-matrix", name, f"{RUNS}/v104-v3-{name}-s7", config, gate=gate),
            )
        )
    # Screening (設計 §5.1): 24 perception + 12 policy, full validation, never
    # eligible and never a substitute for the formal 72-epoch comparison.
    for name, local in (("E4", False), ("E5", True)):
        config = v3_config(
            local_enabled=local,
            epochs_state=24,
            screening=True,
            note="screening 24+12, not a formal comparison",
        )
        written.append(
            write(
                f"{name}-screen.json",
                recipe(
                    "v3-matrix",
                    name,
                    f"{RUNS}/v104-{name}-screen-s7",
                    config,
                    gate=gate,
                    screening=True,
                    notes="screening 24+12, not a formal comparison",
                ),
            )
        )
    # R2: the E3c (E*) recipe plus the pre-declared hard-frame weights.
    r2 = base_config(position_sample_weights=HARD_FRAMES)
    written.append(
        write(
            "R2.json",
            recipe(
                "r2",
                "R2",
                f"{RUNS}/v104-R2-s7",
                r2,
                source_state_checkpoint="",
                baseline={
                    "state_pos_px": 13.0557,
                    "source": "runs/v104-state-d3-s7 state.pt (S3-4 state-d3 best ep55)",
                },
            ),
        )
    )
    # R3: the same recipe with only the danger head swapped.
    r3 = base_config(danger_head_variant="learnable-v104")
    written.append(
        write(
            "R3.json",
            recipe(
                "r3",
                "R3",
                f"{RUNS}/v104-R3-s7",
                r3,
                source_state_checkpoint="",
                baseline={
                    "danger_corr_perframe_median": 0.6198,
                    "source": "runs/v104-state-d3-s7 state.pt (S3-4 state-d3 best ep55)",
                },
            ),
        )
    )
    # Integrate: both changes, only runnable once each parent showed a gain.
    both = base_config(danger_head_variant="learnable-v104", position_sample_weights=HARD_FRAMES)
    written.append(
        write(
            "R2R3.json",
            recipe(
                "integrate",
                "R2R3",
                f"{RUNS}/v104-R2R3-s7",
                both,
                source_state_checkpoint="",
                parents={
                    "r2": "Document/Update/Alpha/v1.0.4/Stage3/S3-7/evidence/r2/phase.json",
                    "r3": "Document/Update/Alpha/v1.0.4/Stage3/S3-7/evidence/r3/phase.json",
                },
                notes="joint change of R2 and R3; not a single-variable comparison",
            ),
        )
    )
    # Stability: seeds 17/27 of the recipe that wins; pointed at E5 for now.
    stability = v3_config()
    written.append(
        write(
            "stability-E5.json",
            recipe(
                "stability",
                "v3-E5",
                f"{RUNS}/v104-v3-E5",
                stability,
                gate=gate,
                notes="seeds 17/27 only; the frozen model stays seed 7",
            ),
        )
    )
    for path in written:
        print(path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
