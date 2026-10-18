"""Pipeline stage 3 - train: consume a processed dataset and write a run report.

The stage's job is not to implement training (that lives in ``bc``, ``distill``
and ``vision_train``) but to make a training run *auditable*:

* it refuses to start without a manifest that verifies against the data on disk;
* it records which model kind, which hyper-parameters and which dataset revision
  produced the checkpoint;
* it copies the offline metrics that actually matter (game-unit errors, teacher
  agreement, regret) into one flat ``train-report.json`` next to the checkpoint.

Supported model kinds
---------------------
``privileged_bc``     behaviour cloning from expert actions (``touhou_rl.bc``)
``privileged_value``  teacher-value regression on privileged state (``touhou_rl.distill``)
``vision``            two-stage vision student (``touhou_rl.vision_train``)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..manifest import DatasetManifest

MODEL_KINDS = ("privileged_bc", "privileged_value", "vision")
MODALITY_FOR = {
    "privileged_bc": "privileged",
    "privileged_value": "privileged",
    "vision": "vision",
}


@dataclass
class TrainSpec:
    manifest: Path
    out_dir: Path
    model: str = "vision"
    epochs: int = 12
    batch_size: int = 128
    lr: float = 3e-4
    device: str = "cuda"
    seed: int = 7
    # vision-only knobs
    depth: int = 3
    width: int = 32
    trunk_hidden: int = 256
    vision_stage: str = "both"
    init_from: Path | None = None
    # privileged-only knobs
    self_hidden: int = 192
    bullet_hidden: int = 96
    max_batches_per_epoch: int = 0
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.model not in MODEL_KINDS:
            raise ValueError(f"model must be one of {MODEL_KINDS}")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")


def run_train(spec: TrainSpec) -> dict:
    spec.validate()
    manifest = DatasetManifest.load(Path(spec.manifest))
    expected = MODALITY_FOR[spec.model]
    if manifest.modality != expected:
        raise ValueError(
            f"model {spec.model} needs a {expected} dataset, manifest says {manifest.modality}"
        )
    dataset_dir = Path(manifest.root)
    train_idx, val_idx = manifest.verify(dataset_dir)
    if not train_idx:
        raise ValueError("manifest leaves no training frames")

    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if spec.model == "privileged_bc":
        from ...bc import BCConfig, train_bc
        from ...models import NetConfig

        summary = train_bc(
            BCConfig(
                dataset=dataset_dir,
                out_dir=out_dir,
                epochs=spec.epochs,
                batch_size=spec.batch_size,
                lr=spec.lr,
                device=spec.device,
                seed=spec.seed,
                max_steps_per_epoch=spec.max_batches_per_epoch or None,
                net=NetConfig(
                    self_hidden=spec.self_hidden,
                    bullet_hidden=spec.bullet_hidden,
                    trunk_hidden=spec.trunk_hidden,
                ),
            )
        )
    elif spec.model == "privileged_value":
        from ...distill import DistillConfig, train_distill
        from ...models import NetConfig

        summary = train_distill(
            DistillConfig(
                dataset=dataset_dir,
                out_dir=out_dir,
                epochs=spec.epochs,
                batch_size=spec.batch_size,
                lr=spec.lr,
                device=spec.device,
                seed=spec.seed,
                net=NetConfig(
                    self_hidden=spec.self_hidden,
                    bullet_hidden=spec.bullet_hidden,
                    trunk_hidden=spec.trunk_hidden,
                ),
            )
        )
    else:
        from ...vision_train import VisionTrainConfig, train

        summary = train(
            VisionTrainConfig(
                dataset=dataset_dir,
                out_dir=out_dir,
                stage=spec.vision_stage,
                epochs_state=spec.epochs,
                epochs_policy=spec.epochs,
                batch_size=spec.batch_size,
                lr=spec.lr,
                device=spec.device,
                seed=spec.seed,
                depth=spec.depth,
                width=spec.width,
                trunk_hidden=spec.trunk_hidden,
                init_from=spec.init_from,
                max_batches_per_epoch=spec.max_batches_per_epoch,
            )
        )

    report = {
        "stage": "train",
        "model": spec.model,
        "dataset": str(dataset_dir),
        "manifest": str(spec.manifest),
        "manifest_summary": manifest.summary(),
        "train_frames": len(train_idx),
        "val_frames": len(val_idx),
        "hyperparameters": {
            "epochs": spec.epochs,
            "batch_size": spec.batch_size,
            "lr": spec.lr,
            "seed": spec.seed,
            "depth": spec.depth if spec.model == "vision" else None,
            "width": spec.width if spec.model == "vision" else None,
            "trunk_hidden": spec.trunk_hidden,
            "vision_stage": spec.vision_stage if spec.model == "vision" else None,
            "init_from": str(spec.init_from) if spec.init_from else None,
            **spec.extra,
        },
        "elapsed_seconds": round(time.time() - started, 1),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": _flat_metrics(spec.model, summary),
        "summary": summary,
    }
    (out_dir / "train-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _flat_metrics(model: str, summary: dict) -> dict:
    """Pull the handful of numbers a report should quote, from each trainer's shape."""
    if model == "vision":
        stages = summary.get("stages", {})
        flat: dict = {}
        state = stages.get("state", {})
        if state.get("history"):
            last = state["history"][-1]
            flat.update(
                {
                    "state_val_loss": last.get("val_loss"),
                    "player_pos_px": last.get("val_player_pos_px"),
                    "danger_corr": last.get("val_danger_corr"),
                    "bullet_count_mae": last.get("val_bullet_count_mae"),
                }
            )
        policy = stages.get("policy", {})
        if policy:
            flat.update(
                {
                    "policy_best_regret_px": policy.get("best_val_regret_px"),
                    "policy_top1": (policy.get("final") or {}).get("top1"),
                }
            )
        return flat
    if model == "privileged_value":
        return {
            "val_top1": summary.get("val_top1"),
            "val_regret_px": summary.get("val_regret_px"),
            "best_val_loss": summary.get("best_val_loss"),
        }
    last = (summary.get("epochs") or [{}])[-1]
    return {
        "best_val_loss": summary.get("best_val_loss"),
        "move_acc": (last.get("acc") or {}).get("move"),
        "val_move_acc": (last.get("acc") or {}).get("val_move"),
    }
