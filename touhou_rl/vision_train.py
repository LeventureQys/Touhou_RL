"""Train the vision student: pixels -> privileged state, then pixels -> values.

Two stages, in this order on purpose
------------------------------------
``state``
    Frames -> the privileged observation (self position, resources, danger map).
    This is a supervised task with free labels, so it answers "does the vision
    path see the game at all" without touching the expensive action signal.  The
    *useful* metrics are the ones that map back to game units (pixels of player
    position, per-frame danger correlation), not the loss.

``policy``
    Frames -> the teacher's value for each of the 18 candidate actions.  The
    teacher is :mod:`touhou_rl.pilot`, which operates on the *stored privileged
    observation* - i.e. the vision student is distilled from the memory-reading
    teacher, exactly the intended role of the reverse-engineered data.

Architecture
------------
``--arch v1`` builds :class:`touhou_rl.vmodels.VisionScorer` (every region pooled
to 2x2, player position regressed by the state head).  ``--arch v2`` builds
:class:`touhou_rl.vmodels2.VisionScorerV2`: the playfield keeps a 28x24 grid and
the player position is read out of a heatmap by soft-argmax, so the state loss
gains a cross-entropy against a gaussian target (:func:`heatmap_target`) weighted
by ``--w-heatmap``.  ``--w-danger`` scales the danger term for both architectures.

Held-out set
------------
The validation split is no longer derived here.  It is read from
``<dataset>/manifest.json`` and *verified* (shard presence, byte sizes, shards
added after processing, split hash) by :meth:`DatasetManifest.verify`, so the
frames this run holds out are exactly the ones the report claims.  The old
``default_rng(seed).permutation(total)`` derivation survives only for synthetic
datasets, behind ``--allow-missing-manifest``, and the summary then records
``manifest_verified: false``.  Training-set sub-sampling (``--train-fraction``)
never touches the validation indices.

Model selection
---------------
The loss is not the selection key: it is dominated by the 128-cell danger term,
while the acceptance criteria read the position error and the per-frame danger
correlation separately.  So:

* ``state.pt``       written when ``val_player_pos_px`` reaches a new minimum;
* ``state-bestdanger.pt`` when ``val_danger_corr_perframe_median`` does;
* ``state-pass.pt``  written **once**, the first time an epoch satisfies both
  acceptance thresholds (``A01_POS_THRESHOLD`` and ``A01_PERFRAME_THRESHOLD``).
  Selecting ``state.pt`` by position alone can throw away the only epoch that
  met both bars, which would read as a false failure;
* ``policy.pt``      written when ``val_regret_px`` reaches a new minimum;
* ``best.pt``        compatibility copy of whichever checkpoint was written last.

``stages.<stage>.best`` in the summary is the *full* row of the epoch that the
saved checkpoint actually holds, so a report never pairs epoch 5's weights with
epoch 8's numbers.  ``stages.policy.final`` is kept for debugging only; acceptance
reads ``best``.

Batch limits
------------
``--max-train-batches`` and ``--max-val-batches`` are separate on purpose: a
screening run that also shrank the validation set would make two recipes'
``val_player_pos_px`` incomparable.

Label material
--------------
Teacher values are computed here rather than captured: they are a pure function
of the privileged observation, so recomputing them is free and lets the teacher
change without re-capturing video.

v1.0.4 opt-in contract (``--selection-contract v104``)
------------------------------------------------------
The defaults above are v1.0.3 behaviour, kept bit-identical so old runs stay
reproducible.  Passing ``--selection-contract v104`` turns on the written
contract of `璁捐鏂囨。` 搂4 C1-C6 / the S3-3 task book, and only then:

* ``history_contract=causal-segment-v1`` is forwarded to
  :class:`VisionDataset` (``legacy`` default keeps the v1.0.3 clamp);
* the policy loss is :func:`touhou_rl.vision_losses.policy_loss` (``legacy-reg``
  by default, so E0/E1 stay comparable), with explicit rank/regression weights;
* the perception auxiliary terms are added to the policy stage
  (heatmap .1 / coordinate 1 / danger .2 / self .1) and the self term is masked
  to the 19 original columns ``4..12`` + ``14..23``;
* the policy stage starts from the epoch actually selected by the state stage
  (``state-pass.pt`` when it exists, else ``state.pt``), never from the last
  in-memory epoch, and it is loaded strictly (arch/depth/ROI/history checked);
* the first ``policy_freeze_epochs`` policy epochs freeze the encoder/perception
  and train only the candidate head; the optimizer is then rebuilt with three
  parameter groups (head 3e-4, encoder/perception 1e-4);
* teacher values come from a fingerprinted cache
  ``<dataset>-teacher-values-v104-<fingerprint>.npz`` with a
  ``v104-teacher/1`` sidecar; the old unfingerprinted file is never trusted or
  overwritten;
* validation always covers every held-out frame (D9), the run records
  ``actual_frames`` / ``visited_index_sha256``, and any limit makes the run
  ``acceptance_eligible=false``;
* model selection is the contract's: ``state.best`` prefers the first joint
  pass, ``policy.best`` is the lexicographic minimum of
  ``(0-15 bucket regret, global regret, epoch)`` and an empty bucket cannot be
  selected;
* the summary additionally carries ``contract_version=v104-A/1`` plus recipe /
  data / teacher / input / loss / optimisation / coverage fingerprints.

Every rejection is a ``ValueError("v104:<CODE>: <detail>")`` whose known codes
reach the CLI as ``exit 2``; unknown exceptions keep their traceback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import obs, spec, vision, vision_losses, vision_metrics
from .pipeline.manifest import DatasetManifest, ManifestError, file_digest, index_digest
from .pilot import AvoidPilot, PilotConfig
from .vision import regions_to_batch
from .vision_data import VisionDataset
from .vision_split import stratified_prefix
from .vmodels import (
    VisionNetConfig,
    VisionScorer,
    load_vision_checkpoint,
    save_vision_checkpoint,
    vision_arch,
)
from .vmodels2 import (
    DANGER_HEAD_LEGACY,
    DANGER_HEAD_VARIANTS,
    VisionNetConfigV2,
    VisionScorerV2,
    decode_position,
    heatmap_target,
)

ROOT = Path(__file__).resolve().parents[1]
VALUE_CLIP = (-8.0, 40.0)
VALUE_SCALE = 10.0
DANGER_SCALE = 8.0
SCHEMA = "vision-summary/2"
N_CANDIDATES = 18
#: Trainer architectures.  ``v2`` stays the default; ``v3-st`` is the S3-5 model
#: (C5) and only reachable through the v1.0.4 contract.
ARCHES = ("v1", "v2", "v3-st")
ARCH_V3 = "v3-st"
#: R2 (瑷▓ 搂5.3): the only extra recipe this round - one pre-declared per-frame
#: weighting of the localisation terms.  The file is an NPZ with exactly
#: ``train_indices``/``errors``/``weights`` plus a separate JSON identity (no
#: pickle), so a loaded weight table cannot be mistaken for another run's.
POSITION_WEIGHTS_SCHEMA = "v104-position-weights/1"
POSITION_WEIGHT_THRESHOLDS = (10.0, 30.0)
POSITION_WEIGHT_VALUES = (1.0, 1.5, 2.0)
#: The v1.0.4 contract version recorded in every v104 summary/checkpoint.
CONTRACT_VERSION = "v104-A/1"
#: Sidecar schema of the fingerprinted teacher cache (task book 搂3).
TEACHER_SCHEMA = "v104-teacher/1"
TEACHER_LABEL_VERSION = "avoidpilot-clearance-px/1"
CANDIDATE_ORDER_VERSION = "focus9-move-v1"
SELECTION_CONTRACTS = ("legacy", "v104")
HISTORY_CONTRACTS = ("legacy", "causal-segment-v1")
#: A-01's two thresholds.  Module-level so the acceptance tooling, the sweep report
#: and this trainer quote the same numbers instead of three copies that drift.
A01_POS_THRESHOLD = 10.0
A01_PERFRAME_THRESHOLD = 0.8
#: v1.0.2's evidence baseline: several reports cite this directory, so writing into
#: it (or onto anything with the same name) would rewrite history.
FROZEN_RUN = ROOT / "third_party" / "th08" / "build" / "rl" / "runs" / "pilot-v1"
FROZEN_RUN_NAME = "pilot-v1"
RUNS_ROOT = ROOT / "third_party" / "th08" / "build" / "rl" / "runs"
#: v1.0.4 baselines are frozen too (璁捐鏂囨。 C4): the 72-epoch A-01 baseline and the
#: 24-epoch clean-data baseline.  Their sub-directories and name aliases count.
FROZEN_V104_NAMES = ("v104-a01-72ep", "v104-clean-24e12e")
#: v1.0.4's self auxiliary only supervises the *original* self_feat columns:
#: 4..12 (9 dims) plus 14..23 (10 dims) = 19.  Index 13 is the constant
#: stage-clear flag and 2/3 are reserved, so they are excluded (璁捐鏂囨。 C3).
SELF_AUX_INDICES = tuple(range(4, 13)) + tuple(range(14, 24))
CHECKPOINT_NAMES = ("state.pt", "state-pass.pt", "state-bestdanger.pt", "policy.pt")


# --------------------------------------------------------------------------- #
# errors / small helpers
# --------------------------------------------------------------------------- #
def _v104_error(code: str, detail: str) -> ValueError:
    """The one error shape of this contract: ``v104:<CODE>: <detail>``."""
    return ValueError(f"v104:{code}: {detail}")


def _sha256_file(path: Path) -> str:
    return file_digest(Path(path))


def _canonical_bytes(payload) -> bytes:
    """Canonical JSON: sorted keys, no padding, UTF-8 (used for fingerprints)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _fingerprint(payload) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _relative_root(path: Path) -> str:
    """Workspace-relative POSIX path when possible, so identities are portable."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _finite_weight(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _v104_error("CONFIG", f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise _v104_error("CONFIG", f"{name} must be finite and >= 0, got {value!r}")
    return number


def _positive(name: str, value) -> float:
    number = _finite_weight(name, value)
    if number <= 0.0:
        raise _v104_error("CONFIG", f"{name} must be > 0, got {value!r}")
    return number


# --------------------------------------------------------------------------- #
# configuration validation
# --------------------------------------------------------------------------- #
@dataclass
class LossPlan:
    """Effective policy loss configuration after the contract's normalisation."""

    objective: str
    rank_delta_px: float
    rank_weight: float
    regression_weight: float
    forced: bool

    def as_dict(self) -> dict:
        return {
            "objective": self.objective,
            "rank_delta_px": self.rank_delta_px,
            "rank_weight": self.rank_weight,
            "regression_weight": self.regression_weight,
            "weights_forced_by_contract": self.forced,
        }


def validate_config(cfg: "VisionTrainConfig") -> LossPlan:
    """Reject every illegal recipe *before* a single frame is decoded.

    The v1.0.3 defaults stay legal, so old command lines keep working.  The v104
    contract adds: the three enums, ``depth`` in ``{1, 3}``, positive learning
    rates / batch / epochs, non-negative finite weights, and the loss-weight
    conventions of :func:`touhou_rl.vision_losses.policy_loss`.
    """
    v104 = cfg.selection_contract == "v104"
    if cfg.selection_contract not in SELECTION_CONTRACTS:
        raise _v104_error(
            "CONFIG", f"selection_contract must be one of {list(SELECTION_CONTRACTS)}, got {cfg.selection_contract!r}"
        )
    if cfg.history_contract not in HISTORY_CONTRACTS:
        raise _v104_error(
            "CONFIG", f"history_contract must be one of {list(HISTORY_CONTRACTS)}, got {cfg.history_contract!r}"
        )
    if cfg.policy_objective not in vision_losses.OBJECTIVES:
        raise _v104_error(
            "CONFIG",
            f"policy_objective must be one of {list(vision_losses.OBJECTIVES)}, got {cfg.policy_objective!r}",
        )
    if cfg.stage not in ("state", "policy", "both"):
        raise _v104_error("CONFIG", f"stage must be state|policy|both, got {cfg.stage!r}")
    if cfg.arch not in ARCHES:
        raise _v104_error("CONFIG", f"arch must be one of {list(ARCHES)}, got {cfg.arch!r}")
    v3 = cfg.arch == ARCH_V3
    if cfg.arch not in ("v1", "v2"):
        if not v104:
            raise _v104_error(
                "CONFIG",
                f"arch={cfg.arch!r} is only reachable through selection_contract='v104'"
                " (the v1.0.3 contract builds v1/v2 only)",
            )
    if not isinstance(cfg.local_enabled, bool):
        raise _v104_error("CONFIG", f"local_enabled must be a bool, got {cfg.local_enabled!r}")
    if cfg.local_enabled is not True and not v3:
        raise _v104_error(
            "CONFIG",
            f"local_enabled=False is the v3-st E4 ablation and needs arch='v3-st', got {cfg.arch!r}",
        )
    if cfg.danger_head_variant not in DANGER_HEAD_VARIANTS:
        raise _v104_error(
            "CONFIG",
            f"danger_head_variant must be one of {list(DANGER_HEAD_VARIANTS)},"
            f" got {cfg.danger_head_variant!r}",
        )
    if cfg.danger_head_variant != DANGER_HEAD_LEGACY and cfg.arch != "v2":
        raise _v104_error(
            "CONFIG",
            f"danger_head_variant={cfg.danger_head_variant!r} is the v2 R3 variant;"
            f" arch={cfg.arch!r} keeps its own danger head",
        )
    if cfg.position_sample_weights is not None and not v104:
        raise _v104_error(
            "CONFIG",
            "--position-sample-weights is the R2 v104 recipe and requires"
            " selection_contract='v104'",
        )
    if not isinstance(cfg.depth, int) or cfg.depth < 1:
        raise _v104_error("CONFIG", f"depth must be a positive integer, got {cfg.depth!r}")
    if not isinstance(cfg.batch_size, int) or cfg.batch_size < 1:
        raise _v104_error("CONFIG", f"batch_size must be >= 1, got {cfg.batch_size!r}")
    for name in ("epochs_state", "epochs_policy", "max_train_batches", "max_val_batches"):
        value = getattr(cfg, name)
        if not isinstance(value, int) or value < 0:
            raise _v104_error("CONFIG", f"{name} must be an integer >= 0, got {value!r}")
    if cfg.stage in ("state", "both") and cfg.epochs_state < 1:
        raise _v104_error("CONFIG", f"stage {cfg.stage!r} needs epochs_state >= 1, got {cfg.epochs_state}")
    if cfg.stage in ("policy", "both") and cfg.epochs_policy < 1:
        raise _v104_error("CONFIG", f"stage {cfg.stage!r} needs epochs_policy >= 1, got {cfg.epochs_policy}")
    _positive("lr", cfg.lr)
    _positive("encoder_lr", cfg.encoder_lr)
    _positive("policy_head_lr", cfg.policy_head_lr)
    _finite_weight("weight_decay", cfg.weight_decay)
    _finite_weight("rank_delta_px", cfg.rank_delta_px)
    for name in (
        "w_heatmap",
        "w_self",
        "w_danger",
        "w_coord",
        "policy_aux_heatmap",
        "policy_aux_coord",
        "policy_aux_danger",
        "policy_aux_self",
    ):
        _finite_weight(name, getattr(cfg, name))
    if not isinstance(cfg.policy_freeze_epochs, int) or cfg.policy_freeze_epochs < 0:
        raise _v104_error(
            "CONFIG", f"policy_freeze_epochs must be an integer >= 0, got {cfg.policy_freeze_epochs!r}"
        )
    if cfg.selection_contract != "v104" and cfg.policy_objective != "legacy-reg":
        raise _v104_error(
            "CONFIG",
            f"policy_objective={cfg.policy_objective!r} requires selection_contract='v104'"
            " (the legacy contract's policy loss is the v1.0.3 regression)",
        )
    if v104:
        if cfg.arch not in ("v2", "v3-st"):
            raise _v104_error(
                "CONFIG",
                f"selection_contract='v104' requires arch='v2' (or 'v3-st'), got {cfg.arch!r}",
            )
        if v3:
            # C5 fixes the v3 geometry: depth 3, one spatial head, the fixed
            # danger head and a heatmap sigma that the model config records.
            if cfg.depth != 3:
                raise _v104_error(
                    "CONFIG", f"arch='v3-st' is depth 3 only (C5), got depth {cfg.depth}"
                )
            if float(cfg.heatmap_sigma) <= 0.0:
                raise _v104_error(
                    "CONFIG", f"heatmap_sigma must be > 0, got {cfg.heatmap_sigma!r}"
                )
        elif cfg.depth not in (1, 3):
            raise _v104_error("CONFIG", f"selection_contract='v104' allows depth 1 or 3, got {cfg.depth}")
        # A short screening run may legitimately have fewer epochs than the freeze
        # window; it then never reaches the joint phase and says so in the summary.
    if cfg.policy_objective == "rank":
        if float(cfg.rank_weight) != 1.0:
            raise _v104_error(
                "CONFIG", f"rank objective requires an explicit rank_weight=1, got {cfg.rank_weight!r}"
            )
        rank_weight = 1.0
        regression_weight = float(cfg.regression_weight)
        forced = False
    else:
        # The contract fixes (0, 1) for the regression objectives; a caller cannot
        # scale the regression term through these knobs (S3-2 搂7.4), and a silent
        # zero loss is exactly what the explicit check in policy_loss prevents.
        forced = (float(cfg.rank_weight) != 0.0) or (float(cfg.regression_weight) != 1.0)
        rank_weight = 0.0
        regression_weight = 1.0
    if rank_weight + regression_weight <= 0.0:
        raise _v104_error("CONFIG", "total policy loss weight must be > 0")
    return LossPlan(
        objective=cfg.policy_objective,
        rank_delta_px=float(cfg.rank_delta_px),
        rank_weight=rank_weight,
        regression_weight=regression_weight,
        forced=forced,
    )


@dataclass
class VisionTrainConfig:
    dataset: Path
    out_dir: Path
    arch: str = "v2"  # v1 | v2 | v3-st
    stage: str = "both"  # state | policy | both
    epochs_state: int = 8
    epochs_policy: int = 8
    batch_size: int = 96
    lr: float = 3e-4
    weight_decay: float = 1e-4
    depth: int = 1
    seed: int = 7
    device: str = "cuda"
    # split: the manifest decides the validation set; these only matter for the
    # synthetic-data fallback and for train-set sub-sampling.
    train_fraction: float = 1.0
    val_fraction: float = 0.1
    allow_missing_manifest: bool = False
    # deliberately separate limits (design doc 7.4)
    max_train_batches: int = 0
    max_val_batches: int = 0
    # pre-1.0.3 name, kept because ``pipeline train`` still passes it; it is read
    # only when ``max_train_batches`` is 0 (see ``train``).
    max_batches_per_epoch: int = 0
    # v2 recipe knobs (recorded in summary["config"] so a run can be reproduced)
    heatmap_sigma: float = 1.5
    w_heatmap: float = 1.0
    w_self: float = 1.0
    w_danger: float = 1.0
    w_coord: float = 1.0
    # torch intra-op threads; 0 = auto (1 on cuda). The model runs on the GPU, so the
    # default 14 spinning threads cost 15x the CPU for identical throughput (measured).
    torch_threads: int = 0
    # Playfield input size (H, W); None keeps the historical 224x192 (native 1/2).
    # 336x288 is exactly 3/4 of native 448x384, so the aspect ratio is preserved and
    # the heatmap cell drops from 16 to ~10.7 native px.
    playfield_hw: tuple[int, int] | None = None
    pf_width: int = 32
    hud_width: int = 16
    # v1 model geometry, kept so --arch v1 stays reproducible
    width: int = 32
    trunk_hidden: int = 256
    # Model input size per region; the archive stays native, so these can change
    # without re-collecting.
    model_regions: tuple[tuple[str, int, int], ...] = VisionNetConfig().model_regions
    init_from: Path | None = None
    teacher_values: Path | None = None
    note: str = ""
    # -- v1.0.4 opt-in contract (defaults reproduce v1.0.3 exactly) ----------
    selection_contract: str = "legacy"  # legacy | v104
    history_contract: str = "legacy"  # legacy | causal-segment-v1
    policy_objective: str = "legacy-reg"  # legacy-reg | centered-reg | rank
    rank_delta_px: float = 0.5
    rank_weight: float = 0.0
    regression_weight: float = 1.0
    policy_aux_heatmap: float = 0.1
    policy_aux_coord: float = 1.0
    policy_aux_danger: float = 0.2
    policy_aux_self: float = 0.1
    policy_freeze_epochs: int = 2
    encoder_lr: float = 1e-4
    policy_head_lr: float = 3e-4
    # -- v3-st (C5) / R3 / R2 ----------------------------------------------
    #: v3 only: ``False`` is the E4 ablation (every parameter slot kept, the
    #: local vector fed to fusion is all zeros - S3-5 implements the switch).
    local_enabled: bool = True
    #: v2 only (R3): ``legacy`` keeps the v1.0.3 danger head.
    danger_head_variant: str = DANGER_HEAD_LEGACY
    #: R2: per-frame localisation weights for A's *training* indices.
    position_sample_weights: Path | None = None
    #: A declared screening run (瑷▓ 搂5.1: 24 perception epochs may save budget,
    #: but they can never stand in for the formal 72).  Never acceptance eligible,
    #: even with an unrestricted, fully covered validation pass.
    screening: bool = False


# --------------------------------------------------------------------------- #
# teacher identity and its fingerprinted cache
# --------------------------------------------------------------------------- #
def _pilot_source_sha256() -> str:
    return _sha256_file(Path(__file__).resolve().with_name("pilot.py"))


def teacher_identity(
    dataset: VisionDataset,
    dataset_dir: Path,
    *,
    manifest: DatasetManifest | None,
    val_index_sha256: str,
) -> dict:
    """Everything the teacher values depend on, in canonical-JSON-able form.

    The identity binds the ordered shard list (name/frames/bytes/sha256), the
    manifest digest, the validation index digest, the candidate order, the whole
    ``PilotConfig``, the teacher source digest, the label dtype and the label
    version.  A change to any of them changes the fingerprint and therefore the
    cache path - which is the point: a cache can no longer be silently reused
    after the data or the teacher changed.
    """
    frames = {path: size for path, _start, size in dataset.offsets}
    disk = [path.name for path in dataset.paths]
    digest_source = "recomputed"
    if manifest is not None and [record.name for record in manifest.shards] == disk:
        shards = [
            {"name": record.name, "frames": int(record.frames), "bytes": int(record.bytes), "sha256": record.sha256}
            for record in manifest.shards
        ]
        # ``verify()`` has just checked every recorded byte count on disk, so the
        # manifest's digests are bound to the files actually being read.
        digest_source = "manifest"
    else:
        shards = [
            {
                "name": path.name,
                "frames": int(frames[path]),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for path in dataset.paths
        ]
    manifest_path = Path(dataset_dir) / "manifest.json"
    return {
        "schema": TEACHER_SCHEMA,
        "dataset": {
            "root": _relative_root(dataset_dir),
            "total_frames": int(len(dataset)),
            "shard_count": len(shards),
            "shards": shards,
            "shard_digest_source": digest_source,
        },
        "manifest_sha256": _sha256_file(manifest_path) if manifest_path.is_file() else "",
        "val_index_sha256": str(val_index_sha256 or ""),
        "candidate_order_version": CANDIDATE_ORDER_VERSION,
        "label_version": TEACHER_LABEL_VERSION,
        "teacher": {
            "name": "pilot",
            "class": "AvoidPilot",
            # The full config, so a switch_penalty change is part of the identity.
            "config": asdict(PilotConfig(switch_penalty=0.0)),
            "source_sha256": _pilot_source_sha256(),
        },
        "dtype": "float16",
        "value_shape": [int(len(dataset)), N_CANDIDATES],
        "score_semantics": "raw_clearance_px",
    }


def teacher_cache_path(dataset_dir: Path, fingerprint: str) -> Path:
    """Default cache: ``<dataset>-teacher-values-v104-<fingerprint>.npz``."""
    dataset_dir = Path(dataset_dir)
    return dataset_dir.parent / f"{dataset_dir.name}-teacher-values-v104-{fingerprint}.npz"


def teacher_sidecar_path(cache: Path) -> Path:
    return Path(str(cache) + ".json")


def teacher_audit(values: np.ndarray) -> dict:
    """Recomputable audit of the float16 labels, including the quantisation ties.

    The precision is *not* changed (璁捐鏂囨。 C3: "鏈疆榛樿淇濇寔鍘?float16"), so the
    ties float16 creates have to be reported instead: a pair of candidates whose
    teacher values collapse to the same half float carries no rank information and
    is excluded by the loss, which is only honest if the size of that set is known.
    """
    array = np.asarray(values)
    numeric = array.astype(np.float64)
    rows, cols = np.triu_indices(numeric.shape[1], 1)
    gaps = np.abs(numeric[:, rows] - numeric[:, cols])
    tied = gaps == 0.0
    best = numeric.max(axis=1)
    return {
        "frames": int(numeric.shape[0]),
        "candidates": int(numeric.shape[1]),
        "pairs_per_frame": int(gaps.shape[1]),
        "dtype": str(array.dtype),
        "value_min_px": float(numeric.min()),
        "value_max_px": float(numeric.max()),
        "pair_gap_p50_px": float(np.percentile(gaps, 50)),
        "pair_gap_mean_px": float(gaps.mean()),
        "pair_gap_p90_px": float(np.percentile(gaps, 90)),
        "tied_pairs": int(tied.sum()),
        "pair_tie_fraction": float(tied.mean()),
        "fully_tied_frames": int(tied.all(axis=1).sum()),
        "frames_with_a_tied_pair": int(tied.any(axis=1).sum()),
        "frame_min_gap_p50_px": float(np.percentile(gaps.min(axis=1), 50)),
        "random_choice_regret_mean_px": float((best - numeric.mean(axis=1)).mean()),
        "quantisation_policy": "float16 kept; ties audited and excluded from rank pairs",
    }


def _validate_teacher_values(values, total: int, source: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (total, N_CANDIDATES):
        raise _v104_error(
            "TEACHER", f"{source}: teacher values must be {(total, N_CANDIDATES)}, got {array.shape}"
        )
    if array.dtype != np.float16:
        raise _v104_error("TEACHER", f"{source}: teacher values must be float16, got {array.dtype}")
    if not np.isfinite(array.astype(np.float32)).all():
        raise _v104_error("TEACHER", f"{source}: teacher values contain non-finite entries")
    return array


def _compute_teacher_values(dataset: VisionDataset, verbose: bool = True) -> np.ndarray:
    """Labels-only recomputation with the fixed teacher (``switch_penalty=0``)."""
    pilot = AvoidPilot(PilotConfig(switch_penalty=0.0))
    out = np.zeros((len(dataset), N_CANDIDATES), dtype=np.float16)
    started = time.perf_counter()
    chunk = 2048
    for start in range(0, len(dataset), chunk):
        stop = min(len(dataset), start + chunk)
        # Labels only: decoding 2.3 M PNG frames just to compute teacher values
        # would cost more than the labelling itself.
        batch = dataset.labels(list(range(start, stop)))
        self_feat = batch["self_feat"].astype(np.float32)
        bullets = batch["bullets"].astype(np.float32)
        danger = batch["danger"]
        for offset in range(stop - start):
            observation = {
                "self_feat": self_feat[offset],
                "bullets": bullets[offset],
                "danger": danger[offset],
            }
            out[start + offset] = pilot.candidate_scores(observation).astype(np.float16)
        if verbose:
            print(f"[vision] teacher-labelled {stop}/{len(dataset)} frames", flush=True)
    if verbose:
        print(f"[vision] labelling took {time.perf_counter() - started:.1f}s", flush=True)
    return out


def _inspect_teacher_cache(path: Path, fingerprint: str, total: int) -> tuple[np.ndarray | None, str]:
    """``(values, reason)``; ``reason == 'ok'`` only when every check passed."""
    cache = Path(path)
    sidecar = teacher_sidecar_path(cache)
    if not cache.is_file():
        return None, "missing"
    if not sidecar.is_file():
        return None, "no-identity"
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "unreadable-sidecar"
    if payload.get("schema") != TEACHER_SCHEMA:
        return None, f"sidecar-schema:{payload.get('schema')!r}"
    if payload.get("fingerprint") != fingerprint:
        return None, "fingerprint"
    try:
        with np.load(cache) as data:
            if "values" not in data.files:
                return None, "no-values"
            values = np.asarray(data["values"])
    except (OSError, ValueError):
        return None, "unreadable-npz"
    if values.shape != (total, N_CANDIDATES):
        return None, "shape"
    if values.dtype != np.float16:
        return None, "dtype"
    if not np.isfinite(values.astype(np.float32)).all():
        return None, "nonfinite"
    if payload.get("cache_sha256") != _sha256_file(cache):
        return None, "digest"
    return values, "ok"


def _write_teacher_cache(path: Path, values: np.ndarray, identity: dict, fingerprint: str) -> str:
    """Verify-then-replace: temp npz, read-back equality, temp sidecar, atomic rename."""
    cache = Path(path)
    sidecar = teacher_sidecar_path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temp = cache.with_name(cache.name + f".tmp{os.getpid()}")
    temp_sidecar = teacher_sidecar_path(temp)
    try:
        with open(temp, "wb") as handle:
            np.savez_compressed(handle, values=values)
        with np.load(temp) as data:
            back = np.asarray(data["values"])
        if back.shape != values.shape or back.dtype != values.dtype or not np.array_equal(back, values):
            raise _v104_error("TEACHER", f"{temp.name}: read-back of the temporary cache did not match")
        digest = _sha256_file(temp)
        payload = {
            "schema": TEACHER_SCHEMA,
            "fingerprint": fingerprint,
            "cache_sha256": digest,
            "label_version": TEACHER_LABEL_VERSION,
            "candidate_order_version": CANDIDATE_ORDER_VERSION,
            "identity": identity,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        temp_sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, cache)
        os.replace(temp_sidecar, sidecar)
        return digest
    finally:
        for leftover in (temp, temp_sidecar):
            try:
                if leftover.exists():
                    leftover.unlink()
            except OSError:  # best effort; a stale temp is not a contract violation
                pass


def load_or_build_teacher_values(
    cfg: "VisionTrainConfig",
    dataset: VisionDataset,
    *,
    manifest: DatasetManifest | None,
    val_index_sha256: str,
    verbose: bool = True,
) -> dict:
    """Resolve the fingerprinted teacher cache, building it when necessary.

    ``--teacher-values`` is an assertion, not a hint: an explicit path that does
    not carry this run's fingerprint (old unfingerprinted file, different data,
    different teacher config) is a ``v104:TEACHER`` stop, never an overwrite.  The
    default path is derived from the fingerprint, so a rebuild there writes a new
    file and leaves the v1.0.3 cache untouched.
    """
    identity = teacher_identity(dataset, Path(cfg.dataset), manifest=manifest, val_index_sha256=val_index_sha256)
    fingerprint = _fingerprint(identity)
    explicit = cfg.teacher_values is not None
    path = Path(cfg.teacher_values) if explicit else teacher_cache_path(Path(cfg.dataset), fingerprint)
    values, reason = _inspect_teacher_cache(path, fingerprint, len(dataset))
    if explicit and reason != "ok":
        raise _v104_error(
            "TEACHER",
            f"--teacher-values {path} does not match the current teacher identity ({reason});"
            " refusing to reuse or overwrite it",
        )
    rebuilt = False
    if values is None:
        if verbose:
            why = f"rebuilding ({reason})" if reason != "missing" else "building"
            print(f"[vision] teacher cache {why}: {path}", flush=True)
        values = _compute_teacher_values(dataset, verbose=verbose)
        values = _validate_teacher_values(values, len(dataset), "recomputed teacher")
        digest = _write_teacher_cache(path, values, identity, fingerprint)
        rebuilt = True
    else:
        digest = _sha256_file(path)
        if verbose:
            print(f"[vision] teacher cache hit: {path}", flush=True)
    return {
        "values": values,
        "path": _relative_root(path),
        "path_absolute": str(path),
        "fingerprint": fingerprint,
        "cache_sha256": digest,
        "identity": identity,
        "identity_sha256": _fingerprint(identity),
        "explicit": explicit,
        "rebuilt": rebuilt,
        "audit": teacher_audit(values),
    }


def teacher_values_for(dataset: VisionDataset, cache: Path, verbose: bool = True) -> np.ndarray:
    """v1.0.3 path: unverified legacy cache, recomputed labels-only when absent.

    Kept byte-identical in behaviour for ``selection_contract='legacy'`` so old
    runs reproduce.  The v104 contract never reaches this function.
    """
    if cache.exists():
        with np.load(cache) as data:
            values = data["values"]
        if values.shape[0] == len(dataset):
            return values
    out = _compute_teacher_values(dataset, verbose=verbose)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, values=out)
    return out


# --------------------------------------------------------------------------- #
# frame gathering, batching and coverage
# --------------------------------------------------------------------------- #
def _gather(
    dataset: VisionDataset,
    indices: list[int],
    depth: int,
    region_sizes: dict[str, tuple[int, int]],
) -> tuple[dict[str, torch.Tensor], dict]:
    """Per-region frame tensors (decoded from the native archive) plus labels.

    The history stack repeats frames: a 96-sample batch at depth 3 asks for 288
    slots, but a shard-chunked batch of 96 samples drawn from 1-2 shards only
    spans ~100 distinct frames.  Decoding and copying the unique frames once and
    gathering them back is value-identical (same rows, same order, same dtype)
    and removes ~2/3 of the per-batch decode/copy traffic at depth 3.

    At depth 1 the stack *cannot* repeat a frame (``stack_indices[:, 0]`` is the
    sample itself), so the de-duplication would only add an ``np.unique`` sort and
    a full gather copy per region.  That was measured: +4 s per epoch over 679
    batches, ~10%, with zero bytes saved.  The one-slot case therefore keeps the
    direct path and stays byte-for-byte the v1.0.3 behaviour.
    """
    requested = dataset.stack_indices(indices, depth).reshape(-1)
    if depth == 1:
        frames = dataset.region_frames(requested.tolist(), region_sizes)
        stacked = {
            name: frames[name].reshape(
                len(indices), depth, region_sizes[name][0], region_sizes[name][1], vision.VISION_CHANNELS
            )
            for name in frames
        }
    else:
        unique, inverse = np.unique(requested, return_inverse=True)
        frames = dataset.region_frames(unique.tolist(), region_sizes)
        stacked = {
            name: frames[name][inverse].reshape(
                len(indices), depth, region_sizes[name][0], region_sizes[name][1], vision.VISION_CHANNELS
            )
            for name in frames
        }
    center = dataset.labels(indices)
    labels = {
        "self_feat": center["self_feat"].astype(np.float32),
        "danger": center["danger"].astype(np.float32),
        "action": center["action"].astype(np.int64),
        "bullets": center["bullets"].astype(np.float32),
    }
    return {name: torch.as_tensor(value, dtype=torch.uint8) for name, value in stacked.items()}, labels


def make_batches(
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool = False,
    dataset: VisionDataset | None = None,
    rng: np.random.Generator | None = None,
    max_batches: int = 0,
    drop_last: bool = True,
):
    """Yield batches of dataset indexes; ``drop_last`` decides the tail (D9).

    Training keeps ``drop_last=True`` (a short batch wastes a full optimiser step
    budget and changes the effective batch size), validation must pass
    ``drop_last=False``: 7241 frames at 96 per batch are 75 full batches plus a
    41-frame tail, and dropping the tail is exactly the v1.0.3 defect.
    """
    if batch_size < 1:
        raise _v104_error("CONFIG", f"batch_size must be >= 1, got {batch_size}")
    order = [int(index) for index in indices]
    if shuffle and dataset is not None:
        if rng is None:
            raise _v104_error("CONFIG", "a shuffled batch order needs a numpy Generator")
        # Shard-granular shuffle: identical per-epoch sample set, but each batch
        # reads 1-2 shard files instead of ~96.
        order = dataset.shard_chunked_order(order, rng)
    limit = len(order) - batch_size + 1 if drop_last else len(order)
    emitted = 0
    for start in range(0, max(0, limit), batch_size):
        yield order[start : start + batch_size]
        emitted += 1
        if max_batches and emitted >= max_batches:
            return


def coverage_record(expected_indices: list[int], visited_indices: list[int]) -> dict:
    """D9 accounting: did the epoch really visit every expected index exactly once?

    Both digests are taken over *sorted* index lists so they are comparable with
    the manifest's ``val_index_sha256`` (which is derived from the sorted split)
    regardless of the order the split happened to be built in.
    """
    expected = [int(index) for index in expected_indices]
    visited = [int(index) for index in visited_indices]
    counts: dict[int, int] = {}
    for index in visited:
        counts[index] = counts.get(index, 0) + 1
    duplicated = sum(1 for count in counts.values() if count > 1)
    missing = sum(1 for index in expected if index not in counts)
    expected_set = set(expected)
    unexpected = sum(1 for index in counts if index not in expected_set)
    ordered = sorted(visited)
    expected_digest = index_digest(sorted(expected))
    visited_digest = index_digest(ordered)
    return {
        "expected_frames": len(expected),
        "actual_frames": len(visited),
        "expected_index_sha256": expected_digest,
        "visited_index_sha256": visited_digest,
        "duplicated_indices": int(duplicated),
        "missing_indices": int(missing),
        "unexpected_indices": int(unexpected),
        "complete": (
            len(visited) == len(expected)
            and duplicated == 0
            and missing == 0
            and unexpected == 0
            and visited_digest == expected_digest
        ),
    }


def require_full_validation(record: dict) -> None:
    """The contract's coverage gate (exact message, exact code)."""
    if not record.get("complete"):
        raise _v104_error("COVERAGE", "validation indices are incomplete or duplicated")


def require_non_empty_split(train_indices: list[int], val_indices: list[int]) -> None:
    """An empty training or validation side is a data error, not a silent no-op."""
    if not train_indices:
        raise _v104_error("DATA", "training split is empty")
    if not val_indices:
        raise _v104_error("DATA", "validation split is empty")


def self_aux_mask(device: torch.device | str = "cpu") -> torch.Tensor:
    """Index tensor of the 19 self_feat columns the v104 auxiliary supervises."""
    return torch.as_tensor(SELF_AUX_INDICES, dtype=torch.long, device=device)


def masked_self_loss(predicted: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """SmoothL1 over the 19 supervised self columns (瑷▓ C3)."""
    return F.smooth_l1_loss(predicted.index_select(1, mask), truth.index_select(1, mask))


def _danger_target(danger: np.ndarray) -> np.ndarray:
    return np.log1p(danger / DANGER_SCALE)


# --------------------------------------------------------------------------- #
# R2: pre-declared hard-frame weights for the localisation terms
# --------------------------------------------------------------------------- #
#: R2 (瑷▓ 搂5.3, 鍟忛娓呭崟 S2-D06) - one pre-declared recipe, no online mining:
#:
#: * the error source is a *train-split only* pass of an already frozen
#:   perception checkpoint (v2: the read-only ``v104-a01-72ep/state.pt``
#:   baseline; v3: this architecture's own frozen perception state);
#: * the weights never touch the validation split and never change sampling;
#: * only the localisation terms are weighted: ``sum(w * per_sample) / sum(w)``
#:   for the heatmap cross-entropy and the coordinate MAE.  Danger and self keep
#:   their plain means, and the policy loss is untouched.
def position_weight_values(errors: np.ndarray) -> np.ndarray:
    """``e<10 -> 1``, ``10 <= e <= 30 -> 1.5``, ``e>30 -> 2`` (float32)."""
    values = np.asarray(errors, dtype=np.float64)
    low, high = POSITION_WEIGHT_THRESHOLDS
    easy, medium, hard = POSITION_WEIGHT_VALUES
    weights = np.full(values.shape, hard, dtype=np.float64)
    weights[values <= high] = medium
    weights[values < low] = easy
    return weights.astype(np.float32)


def position_weight_paths(path) -> tuple[Path, Path]:
    """``(npz, identity json)``: the identity is a sibling file, never a pickle."""
    npz = Path(path)
    return npz, npz.with_suffix(".json")


def _require_sha256(field: str, value, *, code: str = "DATA") -> str:
    """Identity fields that must be full sha256 digests (``v104:<code>`` otherwise)."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _v104_error(code, f"{field} must be a 64-char lowercase hex sha256, got {value!r}")
    return value


def load_position_sample_weights(
    path,
    *,
    train_indices: list[int],
    val_indices: list[int],
    dataset_frames: int,
    region_sizes: dict[str, tuple[int, int]],
    history_contract: str,
    source_state_sha256: str | None = None,
) -> dict:
    """Load + verify R2's per-frame weights; returns the full-length weight array.

    The identity is bound in a **separate JSON** (the task book forbids pickle):
    the npz only carries ``train_indices int64[n]`` / ``errors float32`` /
    ``weights float32``.  Verification, in order:

    1. dtypes and shapes are exactly the contract's;
    2. the stored indices are *exactly* A's training indices - no duplicate, no
       missing, no overlap with validation, no reordering assumption (the file's
       own ``train_indices`` values drive the scatter, never ``row == index``);
    3. errors are finite and >= 0, weights finite and in ``{1, 1.5, 2}``, and the
       weight of every frame is exactly what :func:`position_weight_values`
       derives from its error (the 10 / 30 px boundaries included);
    4. the JSON identity matches: this run's train-index digest, ROI, history
       contract, the error source checkpoint sha (when given), thresholds and
       the per-weight frame counts.
    """
    npz_path, identity_path = position_weight_paths(path)
    if not npz_path.is_file():
        raise _v104_error("DATA", f"position weight file not found: {npz_path}")
    try:
        with np.load(npz_path) as data:
            keys = set(data.keys())
            missing = {"train_indices", "errors", "weights"} - keys
            if missing:
                raise _v104_error(
                    "DATA",
                    f"{npz_path} is missing {sorted(missing)}; the contract fixes"
                    " train_indices/errors/weights",
                )
            indices = np.asarray(data["train_indices"])
            errors = np.asarray(data["errors"])
            weights = np.asarray(data["weights"])
    except _v104_error:
        raise
    except Exception as exc:  # noqa: BLE001 - an unreadable npz is a data error
        raise _v104_error("DATA", f"{npz_path} is unreadable: {exc}") from exc
    if indices.dtype != np.int64:
        raise _v104_error("DATA", f"train_indices dtype {indices.dtype} != int64")
    if errors.dtype != np.float32 or weights.dtype != np.float32:
        raise _v104_error(
            "DATA", f"errors/weights dtype must be float32, got {errors.dtype}/{weights.dtype}"
        )
    if indices.ndim != 1 or errors.ndim != 1 or weights.ndim != 1:
        raise _v104_error("DATA", "train_indices/errors/weights must all be 1-D")
    if not (indices.shape == errors.shape == weights.shape):
        raise _v104_error(
            "DATA",
            f"shape mismatch: indices {indices.shape}, errors {errors.shape}, weights {weights.shape}",
        )
    if not np.isfinite(errors).all():
        raise _v104_error("DATA", "errors contain non-finite values")
    if (errors < 0).any():
        raise _v104_error("DATA", "errors contain negative values")
    if not np.isfinite(weights).all():
        raise _v104_error("DATA", "weights contain non-finite values")

    expected_train = [int(index) for index in train_indices]
    expected_val = {int(index) for index in val_indices}
    stored = [int(index) for index in indices]
    if len(set(stored)) != len(stored):
        raise _v104_error("DATA", "train_indices contain duplicates")
    if set(stored) != set(expected_train):
        missing = sorted(set(expected_train) - set(stored))[:8]
        extra = sorted(set(stored) - set(expected_train))[:8]
        raise _v104_error(
            "DATA",
            f"train_indices are not exactly A's training split: {len(missing)} missing"
            f" (e.g. {missing}), {len(extra)} unexpected (e.g. {extra})",
        )
    if set(stored) & expected_val:
        raise _v104_error(
            "DATA", "train_indices overlap the validation split; validation must stay unweighted"
        )

    expected_weights = position_weight_values(errors)
    if not np.array_equal(expected_weights, weights):
        mismatch = int(np.count_nonzero(expected_weights != weights))
        first = int(np.argmax(expected_weights != weights))
        raise _v104_error(
            "DATA",
            f"weights do not follow the R2 buckets at {mismatch} frame(s); e.g. index"
            f" {stored[first]} has e={float(errors[first]):.4f} -> {float(expected_weights[first])}"
            f" but the file says {float(weights[first])}",
        )

    if not identity_path.is_file():
        raise _v104_error(
            "DATA", f"{identity_path} is missing; a weight table without its identity is not usable"
        )
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _v104_error("DATA", f"{identity_path} is unreadable: {exc}") from exc
    if not isinstance(identity, dict):
        raise _v104_error("DATA", f"{identity_path} does not hold a JSON object")
    if identity.get("schema") != POSITION_WEIGHTS_SCHEMA:
        raise _v104_error(
            "DATA", f"identity schema {identity.get('schema')!r} != {POSITION_WEIGHTS_SCHEMA!r}"
        )
    expected_digest = index_digest(sorted(expected_train))
    if identity.get("train_indices_sha256") != expected_digest:
        raise _v104_error(
            "DATA",
            "identity train_indices_sha256 does not describe this run's training split"
            f" ({identity.get('train_indices_sha256')!r} != {expected_digest!r})",
        )
    if int(identity.get("frames", -1)) != len(stored):
        raise _v104_error(
            "DATA", f"identity records {identity.get('frames')} frames, the npz holds {len(stored)}"
        )
    recorded_roi = identity.get("roi") or {}
    normalised_roi = {str(name): [int(size[0]), int(size[1])] for name, size in region_sizes.items()}
    recorded_normalised = {
        str(name): [int(size[0]), int(size[1])] for name, size in recorded_roi.items()
    }
    if recorded_normalised != normalised_roi:
        raise _v104_error("DATA", f"identity roi {recorded_roi} != this run's {normalised_roi}")
    if identity.get("history_contract") != history_contract:
        raise _v104_error(
            "DATA",
            f"identity history_contract {identity.get('history_contract')!r} != {history_contract!r}",
        )
    if list(identity.get("thresholds", [])) != list(POSITION_WEIGHT_THRESHOLDS):
        raise _v104_error(
            "DATA",
            f"identity thresholds {identity.get('thresholds')} != {list(POSITION_WEIGHT_THRESHOLDS)}",
        )
    if list(identity.get("weight_values", [])) != list(POSITION_WEIGHT_VALUES):
        raise _v104_error(
            "DATA",
            f"identity weight_values {identity.get('weight_values')} != {list(POSITION_WEIGHT_VALUES)}",
        )
    histogram = {str(value): int(np.count_nonzero(weights == value)) for value in POSITION_WEIGHT_VALUES}
    recorded_histogram = {
        str(key): int(value) for key, value in (identity.get("frame_counts") or {}).items()
    }
    if recorded_histogram != histogram:
        raise _v104_error("DATA", f"identity frame_counts {recorded_histogram} != the npz's {histogram}")
    if source_state_sha256 is not None:
        recorded_source = identity.get("source_state_sha256")
        if recorded_source != source_state_sha256:
            raise _v104_error(
                "CHECKPOINT",
                f"weight table was computed from {recorded_source!r}, this run freezes"
                f" {source_state_sha256!r}",
            )
    if identity.get("route") not in (None, "v2", ARCH_V3):
        raise _v104_error("DATA", f"identity route {identity.get('route')!r} is not v2/v3-st")

    table = np.ones(int(dataset_frames), dtype=np.float32)
    table[indices] = weights
    return {
        "weights": table,
        "path": _relative_root(npz_path),
        "sha256": _sha256_file(npz_path),
        "identity_path": _relative_root(identity_path),
        "identity_sha256": _sha256_file(identity_path),
        "identity": identity,
        "train_frames": len(stored),
        "frame_counts": histogram,
        "errors_mean_px": float(np.mean(errors.astype(np.float64))),
        "errors_p90_px": float(np.percentile(errors.astype(np.float64), 90)),
        "weight_sum": float(np.sum(table.astype(np.float64))),
    }


# --------------------------------------------------------------------------- #
# write protection
# --------------------------------------------------------------------------- #
def _refuse_frozen_baseline(out_dir: Path) -> None:
    """Refuse to write into any frozen evidence directory.

    v1.0.2's ``runs/pilot-v1`` (or a directory named so) is the oldest baseline;
    v1.0.4 adds ``runs/v104-a01-72ep`` and ``runs/v104-clean-24e12e``.  The name
    check catches the alias form regardless of which working directory it is
    typed in, the resolved-path check is the exact contract, and any path
    *component* equal to a frozen name catches sub-directories.
    """
    resolved = Path(out_dir).resolve()
    if resolved == FROZEN_RUN or FROZEN_RUN in resolved.parents or resolved.name.lower() == FROZEN_RUN_NAME:
        raise SystemExit("refusing to overwrite the v1.0.2 evidence baseline runs/pilot-v1")
    lowered = {part.lower() for part in resolved.parts}
    for name in FROZEN_V104_NAMES:
        frozen = RUNS_ROOT / name
        if resolved == frozen or frozen in resolved.parents or name in lowered:
            raise SystemExit(f"refusing to overwrite the frozen v1.0.4 baseline runs/{name}")


def _refuse_existing_run(out_dir: Path) -> None:
    """A v104 run directory is write-once: no silent overwrite of a real result."""
    out_dir = Path(out_dir)
    summary_path = out_dir / "vision-summary.json"
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict) and payload.get("schema") == SCHEMA and payload.get("stages"):
            raise _v104_error(
                "CHECKPOINT",
                f"run directory {out_dir} already holds a valid {SCHEMA}; refusing to overwrite it",
            )
    for name in CHECKPOINT_NAMES:
        if (out_dir / name).is_file():
            raise _v104_error(
                "CHECKPOINT", f"run directory {out_dir} already holds {name}; refusing to overwrite it"
            )


# --------------------------------------------------------------------------- #
# model selection keys (v104)
# --------------------------------------------------------------------------- #
def _bucket_regret(by_danger: list[dict], label: str = "0-15"):
    """``regret_px`` of one danger bucket, or ``None`` when the bucket is absent."""
    for row in by_danger:
        if row.get("bucket") == label:
            return row.get("regret_px")
    return None


def policy_selection_key(row: dict) -> tuple[float, float, int] | None:
    """``(0-15 bucket regret, global regret, epoch)`` or ``None`` when unusable.

    An empty 0-15 bucket cannot be selected: its zero-valued diagnostics are a
    placeholder for "no frame here", not a perfect score (瑷▓ C3/C4).  Returning
    ``None`` keeps such a row out of the running instead of letting it win.
    """
    regret = row.get("val_regret_px")
    if regret is None:
        return None
    bucket = None
    for entry in row.get("by_danger") or []:
        if entry.get("bucket") == "0-15":
            if int(entry.get("frames", 0)) > 0:
                bucket = float(entry["regret_px"])
            break
    if bucket is None:
        return None
    return (bucket, float(regret), int(row.get("epoch", 0)))


def state_selection_row(history: list[dict], pass_epoch: int | None) -> tuple[dict | None, str]:
    """Contract rule: first joint A-01 pass, else the best position row."""
    if pass_epoch is not None:
        for row in history:
            if row.get("epoch") == pass_epoch:
                return row, "joint-pass"
    best = None
    for row in history:
        position = row.get("val_player_pos_px")
        if position is None:
            continue
        if best is None or float(position) < float(best.get("val_player_pos_px")):
            best = row
    return best, ("position" if best is not None else "none")


def _checkpoint_record(path: Path) -> dict:
    path = Path(path)
    return {"checkpoint": path.name, "checkpoint_sha256": _sha256_file(path)}


# --------------------------------------------------------------------------- #
# policy parameter groups / forward
# --------------------------------------------------------------------------- #
def policy_parameter_groups(model: torch.nn.Module) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    """Split the model's parameters into encoder / perception / candidate head.

    For v2, ``encoder`` is the pixel backbone plus the fused trunk,
    ``perception`` is the shared state head (heatmap/self/danger decoder) and
    ``candidate_head`` is the 18-way scoring head including the candidate
    embedding.  The policy stage freezes the first two for
    ``policy_freeze_epochs`` epochs (瑷▓ C4).

    v3 has no candidate embedding (``policy_head`` scores the trunk directly) and
    its perception decoders are separate modules, so the same three roles are
    spelled out for its module names: ``policy_head`` is the candidate head,
    ``self_head``/``heat_conv``/``heatmap_head``/``location_mlp``/``danger`` are
    perception, and the playfield/local/global/HUD branches plus the fusion
    trunk are the encoder.  The roles, not the module names, are the contract.
    """
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "encoder": [],
        "perception": [],
        "candidate_head": [],
    }
    v3 = vision_arch(model) == ARCH_V3
    if v3:
        candidate_prefixes = ("policy_head.",)
        perception_prefixes = (
            "self_head.",
            "heat_conv.",
            "heatmap_head.",
            "location_mlp.",
            "danger.",
        )
    else:
        candidate_prefixes = ("candidate_embed.", "score_head.")
        perception_prefixes = ("self_head.",)
    for name, parameter in model.named_parameters():
        if name.startswith(candidate_prefixes):
            groups["candidate_head"].append((name, parameter))
        elif name.startswith(perception_prefixes):
            groups["perception"].append((name, parameter))
        else:
            groups["encoder"].append((name, parameter))
    return groups


def build_policy_optimizer(
    model: torch.nn.Module,
    cfg: "VisionTrainConfig",
    *,
    frozen: bool,
) -> tuple[torch.optim.Optimizer, dict]:
    """``(optimizer, record)``; rebuilds at the frozen -> joint switch.

    ``frozen=True`` puts only the candidate head in the optimizer and turns
    ``requires_grad`` off for everything else, so a forgotten zero-gradient path
    cannot silently train the encoder at head speed.
    """
    groups = policy_parameter_groups(model)
    learning_rates = {"encoder": float(cfg.encoder_lr), "perception": float(cfg.encoder_lr), "candidate_head": float(cfg.policy_head_lr)}
    record: dict = {"phase": "frozen" if frozen else "joint", "groups": {}, "trainable_parameters": 0}
    parameters = []
    for name in ("encoder", "perception", "candidate_head"):
        members = groups[name]
        trainable = (name == "candidate_head") if frozen else True
        for _parameter_name, parameter in members:
            parameter.requires_grad_(trainable)
        entry = {
            "lr": learning_rates[name] if trainable else 0.0,
            "tensors": len(members),
            "parameters": int(sum(parameter.numel() for _n, parameter in members)),
            "trainable": bool(trainable),
            "weight_decay": float(cfg.weight_decay),
        }
        record["groups"][name] = entry
        record["trainable_parameters"] += entry["parameters"] if trainable else 0
        if trainable and members:
            parameters.append(
                {
                    "params": [parameter for _n, parameter in members],
                    "lr": learning_rates[name],
                    "weight_decay": float(cfg.weight_decay),
                    "name": name,
                }
            )
    if not parameters:
        raise _v104_error("CONFIG", "policy optimizer has no trainable parameters")
    return torch.optim.AdamW(parameters), record


def policy_forward(model: torch.nn.Module, regions: dict[str, torch.Tensor]):
    """One shared encode -> ``(scores, aux)``; equals ``score()`` + ``decode_state()``.

    Calling ``score()`` and ``decode_state()`` separately would encode the pixels
    twice per step.  This mirrors :meth:`VisionScorerV2.score` and
    :meth:`VisionScorerV2.decode_state` exactly (a unit test pins the equality),
    so the auxiliary terms cost no extra forward.

    v3 must use its **single** ``forward()``: there,
    ``decode_state(features=...)`` deliberately recomputes the spatial maps from
    the pixels (C5), so the v2 idiom would decode every batch a second time.
    ``forward()`` already returns all seven C5 tensors; the mapping below is the
    v2 key layout the loss code consumes (``self_feat`` is ``self_pred``, whose
    columns 0/1 are the heatmap position).
    """
    if vision_arch(model) == ARCH_V3:
        out = model(regions)
        return out["policy_logits"], {
            "heatmap_logits": out["heatmap_logits"],
            "danger_log": out["danger_log"],
            "self_feat": out["self_pred"],
        }
    trunk, heatmap_logits, danger_log = model._encode(regions)
    scores = model.score(regions, features=trunk)
    rx, ry = decode_position(heatmap_logits)
    predicted = model.self_head(trunk)
    self_feat = torch.cat([rx.unsqueeze(-1), ry.unsqueeze(-1), predicted], dim=-1)
    return scores, {"heatmap_logits": heatmap_logits, "danger_log": danger_log, "self_feat": self_feat}


# --------------------------------------------------------------------------- #
# split / model build
# --------------------------------------------------------------------------- #
def _resolve_split(
    dataset: VisionDataset, cfg: VisionTrainConfig, total: int
) -> tuple[list[int], list[int], str, bool]:
    """``(train_idx, val_idx, val_index_sha256, manifest_verified)``.

    A manifest must exist and verify: the whole point of the file is that a later
    run cannot silently hold out a different tenth of the frames.  The fallback
    derivation is only reachable with ``--allow-missing-manifest`` (synthetic test
    data) and is flagged in the summary.
    """
    dataset_dir = Path(cfg.dataset)
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = DatasetManifest.load(manifest_path)
        train_idx, val_idx = manifest.verify(dataset_dir)
        if manifest.total_frames != total:
            raise ManifestError(
                f"dataset has {total} frames on disk but the manifest records {manifest.total_frames}"
            )
        val_sha = str(manifest.splits["val_index_sha256"])
        verified = True
    elif cfg.allow_missing_manifest:
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(total)
        n_val = max(1, int(total * cfg.val_fraction))
        val_idx = [int(index) for index in perm[:n_val]]
        train_idx = [int(index) for index in perm[n_val:]]
        val_sha = ""
        verified = False
    else:
        raise ManifestError(
            f"no manifest at {manifest_path}; run the process stage first"
            " (or pass --allow-missing-manifest for synthetic test data)"
        )

    if float(cfg.train_fraction) < 1.0:
        if not verified:
            # A fixed validation set is the premise of a data-volume curve; without
            # a verified manifest there is no such premise.
            raise SystemExit("--train-fraction requires a verified manifest")
        stage_map = dataset.stage_of(list(range(total)))
        train_idx = stratified_prefix(train_idx, stage_map, float(cfg.train_fraction), cfg.seed)
    return train_idx, val_idx, val_sha, verified


def _build_model(cfg: VisionTrainConfig, device: torch.device, *, use_init_from: bool = True):
    """``(model, region_sizes)``; ``--init-from`` replaces the fresh build entirely."""
    if use_init_from and cfg.init_from is not None:
        model, _extra = load_vision_checkpoint(cfg.init_from, map_location=device)
        print(f"[vision] warm start from {cfg.init_from}", flush=True)
        return model.to(device), model.config.region_sizes
    if cfg.arch == ARCH_V3:
        # C5's model is imported lazily: the v1/v2 paths must stay importable
        # without it, and an absent module is an explicit error, not a fallback.
        from .vmodels3 import VisionSpatialTemporalConfig, VisionSpatialTemporalPolicy

        net_config = VisionSpatialTemporalConfig(
            depth=cfg.depth,
            local_enabled=cfg.local_enabled,
            # The training target sigma is the model config's, not a second
            # trainer-side constant (C5 records 2.0; this keeps one source).
            heatmap_gauss_sigma=float(cfg.heatmap_sigma),
            history_contract=cfg.history_contract,
            pf_width=cfg.pf_width,
            hud_width=cfg.hud_width,
            trunk_hidden=cfg.trunk_hidden,
        )
        return VisionSpatialTemporalPolicy(net_config).to(device), net_config.region_sizes
    if cfg.arch == "v2":
        geometry = {}
        if cfg.playfield_hw is not None:
            # The playfield is where the small targets live; the HUD stays native.
            # 336x288 is exactly 3/4 of the native 448x384, so the aspect ratio is
            # preserved and the heatmap cell shrinks from 16 to ~10.7 native px.
            geometry["playfield_hw"] = tuple(cfg.playfield_hw)
        net_config = VisionNetConfigV2(
            depth=cfg.depth,
            trunk_hidden=cfg.trunk_hidden,
            pf_width=cfg.pf_width,
            hud_width=cfg.hud_width,
            heatmap_gauss_sigma=cfg.heatmap_sigma,
            danger_head_variant=cfg.danger_head_variant,
            **geometry,
        )
        return VisionScorerV2(net_config).to(device), net_config.region_sizes
    net_config = VisionNetConfig(
        depth=cfg.depth,
        width=cfg.width,
        trunk_hidden=cfg.trunk_hidden,
        model_regions=cfg.model_regions,
    )
    return VisionScorer(net_config).to(device), net_config.region_sizes


def _load_policy_source(
    cfg: VisionTrainConfig,
    out_dir: Path,
    model: torch.nn.Module,
    region_sizes: dict[str, tuple[int, int]],
    device: torch.device,
) -> dict:
    """Strictly load the checkpoint the state stage selected (v104 contract).

    ``stage=both``: ``state-pass.pt`` when the joint A-01 pass happened, else
    ``state.pt``.  ``stage=policy``: an explicit ``--init-from``.  Architecture,
    depth, ROI sizes and history contract must match; ``strict=True`` means a
    cross-depth or cross-architecture source is an error, never a partial load.
    """
    rule = "explicit-init-from"
    if cfg.stage == "both":
        source = None
        for name, candidate_rule in (("state-pass.pt", "joint-pass"), ("state.pt", "position")):
            if (Path(out_dir) / name).is_file():
                source = Path(out_dir) / name
                rule = candidate_rule
                break
        if source is None:
            raise _v104_error(
                "CHECKPOINT",
                f"stage 'both' finished without state-pass.pt or state.pt under {out_dir};"
                " the policy stage must start from the selected state checkpoint",
            )
    else:
        if cfg.init_from is None:
            raise _v104_error(
                "CONFIG",
                "stage 'policy' with selection_contract='v104' requires --init-from <state checkpoint>",
            )
        source = Path(cfg.init_from)
    if not source.is_file():
        raise _v104_error("CHECKPOINT", f"source state checkpoint {source} does not exist")
    loaded, extra = load_vision_checkpoint(source, map_location="cpu")
    if extra.get("arch") != cfg.arch:
        raise _v104_error(
            "CHECKPOINT", f"{source}: arch {extra.get('arch')!r} does not match cfg.arch {cfg.arch!r}"
        )
    if int(getattr(loaded.config, "depth", -1)) != int(cfg.depth):
        raise _v104_error(
            "CHECKPOINT", f"{source}: depth {getattr(loaded.config, 'depth', None)} does not match cfg.depth {cfg.depth}"
        )
    source_regions = {name: tuple(size) for name, size in loaded.config.region_sizes.items()}
    if source_regions != {name: tuple(size) for name, size in region_sizes.items()}:
        raise _v104_error("CHECKPOINT", f"{source}: ROI sizes {source_regions} do not match {region_sizes}")
    if extra.get("history_contract") != cfg.history_contract:
        raise _v104_error(
            "CHECKPOINT",
            f"{source}: history_contract {extra.get('history_contract')!r} does not match"
            f" cfg.history_contract {cfg.history_contract!r}",
        )
    model.load_state_dict(loaded.state_dict(), strict=True)
    model.to(device)
    return {
        "path": _relative_root(source),
        "sha256": _sha256_file(source),
        "epoch": extra.get("epoch"),
        "stage": extra.get("stage"),
        "selection_rule": rule,
        "strict_load": True,
        "history_contract": extra.get("history_contract"),
        "contract_version": extra.get("contract_version"),
    }


# --------------------------------------------------------------------------- #
# summary config block
# --------------------------------------------------------------------------- #
def _summary_config(cfg: VisionTrainConfig, max_train_batches: int, plan: LossPlan | None = None) -> dict:
    """Recipe block of the summary; ``max_train_batches`` separates screening runs.

    It records the **effective** training limit, so a run started through the
    deprecated ``max_batches_per_epoch`` alias is reported as a screening run too.
    """
    block = {
        "stage": cfg.stage,
        "epochs_state": cfg.epochs_state,
        "epochs_policy": cfg.epochs_policy,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "depth": cfg.depth,
        "max_train_batches": max_train_batches,
        "max_val_batches": cfg.max_val_batches,
        "heatmap_sigma": cfg.heatmap_sigma,
        "w_heatmap": cfg.w_heatmap,
        "w_coord": cfg.w_coord,
        "torch_threads": torch.get_num_threads(),
        "playfield_hw": list(cfg.playfield_hw) if cfg.playfield_hw else None,
        "w_self": cfg.w_self,
        "w_danger": cfg.w_danger,
        "pf_width": cfg.pf_width,
        "hud_width": cfg.hud_width,
        # below here: extra provenance, not part of the required minimum
        "seed": cfg.seed,
        "weight_decay": cfg.weight_decay,
        "val_fraction": cfg.val_fraction,
        "train_fraction": cfg.train_fraction,
        "allow_missing_manifest": cfg.allow_missing_manifest,
        "init_from": str(cfg.init_from) if cfg.init_from is not None else "",
        "selection_contract": cfg.selection_contract,
        "history_contract": cfg.history_contract,
        "policy_objective": cfg.policy_objective,
        "rank_delta_px": cfg.rank_delta_px,
        "rank_weight": cfg.rank_weight,
        "regression_weight": cfg.regression_weight,
        "policy_aux_heatmap": cfg.policy_aux_heatmap,
        "policy_aux_coord": cfg.policy_aux_coord,
        "policy_aux_danger": cfg.policy_aux_danger,
        "policy_aux_self": cfg.policy_aux_self,
        "policy_freeze_epochs": cfg.policy_freeze_epochs,
        "encoder_lr": cfg.encoder_lr,
        "policy_head_lr": cfg.policy_head_lr,
        "teacher_values": str(cfg.teacher_values) if cfg.teacher_values is not None else "",
        # v3 / R3 / R2 switches: recorded even when they hold the default, so a
        # recipe can be replayed without reading the command line back.
        "local_enabled": bool(cfg.local_enabled),
        "danger_head_variant": cfg.danger_head_variant,
        "position_sample_weights": str(cfg.position_sample_weights)
        if cfg.position_sample_weights is not None
        else "",
        "screening": bool(cfg.screening),
    }
    if plan is not None:
        block["effective_loss"] = plan.as_dict()
    return block


def coordinate_loss(pred_position, truth_position) -> torch.Tensor:
    """Mean absolute position error in pixels, scaled by 1/100.

    The heatmap is trained with cross-entropy against a 2-D Gaussian.  When the
    network is unsure, cross-entropy is minimised by a *broad* distribution, and
    the reported coordinate is the mean of that distribution - which drifts to the
    data prior.  This term optimises the quantity the metric actually reads, so
    hedging stops being free.  Measured on clean data: 42.35 -> 37.89 px.
    """
    residual = (
        torch.abs(pred_position[:, 0] - truth_position[:, 0]) * spec.PLAYFIELD_W
        + torch.abs(pred_position[:, 1] - truth_position[:, 1]) * spec.PLAYFIELD_H
    ) / 2.0
    return residual.mean() / 100.0


# --- per-sample localisation terms (R2 needs the un-reduced values) -------- #
def heatmap_loss_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``[B, H*W]`` logits/probability targets -> ``[B]`` cross-entropies."""
    log_probability = torch.log_softmax(logits.flatten(1), dim=-1)
    return -(target * log_probability).sum(-1)


def coordinate_loss_per_sample(pred_position: torch.Tensor, truth_position: torch.Tensor) -> torch.Tensor:
    """``[B, 2]`` positions -> ``[B]`` native-pixel MAE / 100."""
    residual = (
        torch.abs(pred_position[:, 0] - truth_position[:, 0]) * spec.PLAYFIELD_W
        + torch.abs(pred_position[:, 1] - truth_position[:, 1]) * spec.PLAYFIELD_H
    ) / 2.0
    return residual / 100.0


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``sum(w * v) / sum(w)`` - the R2 normalisation, never a batch mean."""
    total = weights.sum()
    if float(total) <= 0.0:
        raise _v104_error("DATA", "position sample weights sum to zero inside a batch")
    return (values * weights).sum() / total


def localisation_losses(
    *,
    heatmap_per_sample: torch.Tensor,
    coord_per_sample: torch.Tensor,
    self_loss: torch.Tensor,
    danger_loss: torch.Tensor,
    weights: torch.Tensor | None,
    w_heatmap: float,
    w_coord: float,
    w_self: float,
    w_danger: float,
) -> tuple[torch.Tensor, dict]:
    """Assemble the perception auxiliary terms, weighting **only** hm/xy.

    ``weights=None`` reproduces the v1.0.3/v2.0 reduction exactly (a plain batch
    mean per term).  With R2 weights the two localisation terms become
    ``sum(w * per_sample) / sum(w)``; danger and self keep their plain means, so
    a hard-frame recipe cannot quietly re-weight supervision it did not declare.
    """
    if weights is None:
        heatmap_term = heatmap_per_sample.mean()
        coord_term = coord_per_sample.mean()
    else:
        heatmap_term = weighted_mean(heatmap_per_sample, weights)
        coord_term = weighted_mean(coord_per_sample, weights)
    total = (
        float(w_heatmap) * heatmap_term
        + float(w_coord) * coord_term
        + float(w_self) * self_loss
        + float(w_danger) * danger_loss
    )
    return total, {
        "heatmap": heatmap_term,
        "coord": coord_term,
        "self": self_loss,
        "danger": danger_loss,
    }


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(cfg: VisionTrainConfig) -> dict:
    _refuse_frozen_baseline(cfg.out_dir)
    fraction = float(cfg.train_fraction)
    if not 0.0 < fraction <= 1.0:
        raise SystemExit(f"--train-fraction must be in (0, 1], got {cfg.train_fraction}")
    plan = validate_config(cfg)
    v104 = cfg.selection_contract == "v104"
    v3 = cfg.arch == ARCH_V3
    if v104:
        _refuse_existing_run(cfg.out_dir)
    # A v3 checkpoint records the manifest digest and the split digest (C6).  A run
    # without a manifest could not write an honest identity, so it must not start.
    if v3:
        manifest_path = Path(cfg.dataset) / "manifest.json"
        if not manifest_path.is_file():
            raise _v104_error(
                "CHECKPOINT",
                f"arch='v3-st' writes vision-checkpoint/3, which records manifest_sha;"
                f" no manifest at {manifest_path}",
            )
        if cfg.allow_missing_manifest:
            raise _v104_error(
                "CHECKPOINT",
                "arch='v3-st' refuses --allow-missing-manifest: the split identity would be empty",
            )
    # One effective training limit: the new name wins, the pre-1.0.3 name (still
    # passed by ``pipeline train``) is honoured when the new one is left at 0.
    max_train_batches = int(cfg.max_train_batches or cfg.max_batches_per_epoch)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    # The model runs on the GPU, so torch's intra-op pool has nothing to
    # parallelise and only spins.  Measured on this host (RTX 4070 SUPER,
    # batch 96): 21.1 ms wall / 3.8 ms CPU at 1 thread versus 20.9 ms wall /
    # 57.5 ms CPU at the default 14 threads - identical throughput, 15x the CPU.
    if cfg.torch_threads:
        torch.set_num_threads(int(cfg.torch_threads))
    elif device.type == "cuda":
        torch.set_num_threads(1)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    started_clock = time.perf_counter()
    dataset = VisionDataset(cfg.dataset, history_contract=cfg.history_contract)
    try:
        total = len(dataset)
        if total < 4:
            raise RuntimeError(f"dataset too small: {total} frames")
        train_idx, val_idx, val_sha, manifest_verified = _resolve_split(dataset, cfg, total)
        if v104:
            require_non_empty_split(train_idx, val_idx)
        print(
            f"[vision] {cfg.dataset}: {total} frames, train {len(train_idx)} (fraction {fraction}), "
            f"val {len(val_idx)}, arch {cfg.arch}, manifest_verified {manifest_verified}",
            flush=True,
        )
        manifest = DatasetManifest.load(Path(cfg.dataset) / "manifest.json") if (
            Path(cfg.dataset) / "manifest.json"
        ).is_file() else None
        stats = dataset.label_stats()

        model, region_sizes = _build_model(cfg, device, use_init_from=not v104)
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        shuffle_rng = np.random.default_rng(cfg.seed)
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        manifest_sha = (
            _sha256_file(Path(cfg.dataset) / "manifest.json")
            if (Path(cfg.dataset) / "manifest.json").is_file()
            else ""
        )
        # C5: for v3 the heatmap target sigma comes from the model config (which
        # `_build_model` filled from cfg.heatmap_sigma); v2 keeps cfg.heatmap_sigma.
        heatmap_sigma = (
            float(getattr(model.config, "heatmap_gauss_sigma", cfg.heatmap_sigma))
            if v3
            else float(cfg.heatmap_sigma)
        )

        summary = {
            "schema": SCHEMA,
            "arch": cfg.arch,
            "note": cfg.note,
            "config": _summary_config(cfg, max_train_batches, plan),
            "dataset": str(cfg.dataset),
            "dataset_stats": stats,
            "frames": total,
            "train_frames": len(train_idx),
            "val_frames": len(val_idx),
            "train_fraction": fraction,
            "manifest_verified": manifest_verified,
            "val_index_sha256": val_sha,
            "device": str(device),
            "depth": cfg.depth,
            "parameters": model.num_parameters(),
            "frame_shape": {
                name: [height, width, vision.VISION_CHANNELS]
                for name, (height, width) in region_sizes.items()
            },
            "started": started,
            "stages": {},
        }

        v104_context: dict = {}
        #: R2 state: ``None`` keeps every reduction a plain batch mean.
        position_weights: dict | None = None
        if v104 and cfg.position_sample_weights is not None:
            position_weights = load_position_sample_weights(
                cfg.position_sample_weights,
                train_indices=train_idx,
                val_indices=val_idx,
                dataset_frames=total,
                region_sizes=region_sizes,
                history_contract=cfg.history_contract,
            )
            summary["position_weights"] = {
                key: position_weights[key]
                for key in (
                    "path",
                    "sha256",
                    "identity_path",
                    "identity_sha256",
                    "train_frames",
                    "frame_counts",
                    "errors_mean_px",
                    "errors_p90_px",
                    "weight_sum",
                )
            }
            summary["position_weights"]["identity"] = position_weights["identity"]
            summary["position_weights"]["reduction"] = (
                "sum(w * per_sample) / sum(w) on the heatmap and coordinate terms only;"
                " danger/self/policy unweighted; validation unweighted"
            )
            print(
                f"[vision] R2 position weights {position_weights['path']} "
                f"(counts {position_weights['frame_counts']})",
                flush=True,
            )

        def loss_config_block() -> dict:
            """The C6 ``extra.loss_config`` identity of this run's objective."""
            return {
                "policy": plan.as_dict(),
                "aux_perception": {
                    "heatmap": float(cfg.w_heatmap),
                    "coord": float(cfg.w_coord),
                    "danger": float(cfg.w_danger),
                    "self": float(cfg.w_self),
                },
                "aux_policy": {
                    "heatmap": float(cfg.policy_aux_heatmap),
                    "coord": float(cfg.policy_aux_coord),
                    "danger": float(cfg.policy_aux_danger),
                    "self": float(cfg.policy_aux_self),
                },
                "self_mask_indices": list(SELF_AUX_INDICES),
                "danger_scale": DANGER_SCALE,
                "heatmap_sigma": heatmap_sigma,
                "position_sample_weights_sha256": position_weights["sha256"]
                if position_weights is not None
                else "",
            }

        def checkpoint_extra(stage_name: str, epoch: int, payload: dict | None = None) -> dict:
            extra = {
                "stage": stage_name,
                "epoch": epoch,
                "arch": cfg.arch,
                "train_fraction": fraction,
                "val_index_sha256": val_sha,
                "history_contract": cfg.history_contract,
            }
            if v104:
                extra["contract_version"] = CONTRACT_VERSION
                extra["selection_contract"] = "v104"
                extra["policy_objective"] = plan.objective
                extra["depth"] = int(cfg.depth)
                extra["region_sizes"] = {name: list(size) for name, size in region_sizes.items()}
                extra["local_enabled"] = bool(cfg.local_enabled)
                extra["danger_head_variant"] = cfg.danger_head_variant
                extra["heatmap_sigma"] = heatmap_sigma
                if position_weights is not None:
                    extra["position_sample_weights_sha256"] = position_weights["sha256"]
                extra.update(v104_context)
                if v3:
                    # C6's five identity keys.  ``source_state_sha`` is "" only for
                    # the perception stage (no initialisation source); the policy
                    # stage fills it from the checkpoint it actually loaded.
                    extra["source_state_sha"] = str(v104_context.get("source_state_sha256", ""))
                    extra["teacher_fingerprint"] = str(v104_context.get("teacher_fingerprint", ""))
                    extra["manifest_sha"] = manifest_sha
                    extra["val_index_sha256"] = val_sha
                    extra["loss_config"] = loss_config_block()
            extra.update(payload or {})
            return extra

        def epoch_batches(indices: list[int], training: bool, max_batches: int = 0):
            # D9: training keeps full batches, validation must not drop the tail.
            return make_batches(
                indices,
                batch_size=cfg.batch_size,
                shuffle=training,
                dataset=dataset,
                rng=shuffle_rng,
                max_batches=max_batches,
                drop_last=bool(training),
            )

        def state_row(epoch: int, train_row: dict, val_row: dict, seconds: float) -> dict:
            return {
                "epoch": epoch,
                "train_loss": train_row.get("loss"),
                "val_loss": val_row.get("loss"),
                "seconds": seconds,
                "val_frames": val_row.get("frames", 0),
                **{
                    f"val_{key}": value
                    for key, value in val_row.items()
                    if key not in ("loss", "frames")
                },
            }

        def policy_row(epoch: int, train_row: dict, val_row: dict, seconds: float) -> dict:
            by_danger = val_row.get("by_danger") or []
            row = {
                "epoch": epoch,
                "train_loss": train_row.get("loss"),
                "val_loss": val_row.get("loss"),
                "seconds": seconds,
                "val_frames": val_row.get("frames", 0),
                "val_top1": val_row.get("top1"),
                "val_regret_px": val_row.get("regret_px"),
                "val_regret_px_bucket_0_15": _bucket_regret(by_danger),
                "by_danger": by_danger,
            }
            if v104:
                row.update(
                    {
                        "val_top3": val_row.get("top3"),
                        "val_top3_tie": val_row.get("top3_tie"),
                        "val_near_optimal_05": val_row.get("near_optimal_05"),
                        "val_regret_p90_px": val_row.get("regret_p90_px"),
                        "val_score_std": val_row.get("score_std"),
                        "policy_loss_stats": val_row.get("loss_stats"),
                        "aux_losses": val_row.get("aux"),
                        "train_policy_loss_stats": train_row.get("loss_stats"),
                    }
                )
            return row

        def batch_position_weights(index: list[int]) -> torch.Tensor | None:
            """R2 weights of this batch, or ``None`` (plain means) when unused."""
            if position_weights is None:
                return None
            return torch.as_tensor(position_weights["weights"][index], dtype=torch.float32, device=device)

        def run_state_epoch(training: bool, indices: list[int], max_batches: int = 0) -> dict:
            model.train(training)
            total_loss = 0.0
            seen = 0
            preds: list[np.ndarray] = []
            truths: list[np.ndarray] = []
            pred_danger: list[np.ndarray] = []
            true_danger: list[np.ndarray] = []
            heatmaps: list[np.ndarray] = []
            visited: list[int] = []
            mask = self_aux_mask(device)
            for index in epoch_batches(indices, training, max_batches):
                visited.extend(index)
                regions, labels = _gather(dataset, index, cfg.depth, region_sizes)
                batch = regions_to_batch({name: value.numpy() for name, value in regions.items()}, device)
                batch_size = len(index)
                truth_self = torch.as_tensor(labels["self_feat"], device=device)
                target_danger = torch.as_tensor(_danger_target(labels["danger"]), device=device)
                with torch.set_grad_enabled(training):
                    out = model.decode_state(batch)
                    loss_danger = F.smooth_l1_loss(
                        out["danger_log"].reshape(batch_size, obs.DANGER_ROWS, obs.DANGER_COLS),
                        target_danger.reshape(batch_size, obs.DANGER_ROWS, obs.DANGER_COLS),
                    )
                    if "heatmap_logits" in out:  # v2 only; v1 has no heatmap_hw
                        grid_h, grid_w = model.config.heatmap_hw
                        target_heatmap = torch.as_tensor(
                            heatmap_target(
                                labels["self_feat"][:, 0],
                                labels["self_feat"][:, 1],
                                grid_h,
                                grid_w,
                                # C5: the v3 target sigma is the model config's.
                                sigma=heatmap_sigma,
                            ),
                            device=device,
                        )
                        if v104:
                            # 瑷▓ C3: the v104 auxiliary supervises only columns
                            # 4..12 + 14..23; columns 2/3 (reserved) and 13 (the
                            # constant stage-clear flag) carry no signal.
                            loss_self = masked_self_loss(out["self_feat"], truth_self, mask)
                        else:
                            loss_self = F.smooth_l1_loss(out["self_feat"][:, 2:24], truth_self[:, 2:24])
                        # Coordinate loss: cross-entropy alone is minimised by a *broad*
                        # distribution when the model is unsure, and the metric reads the
                        # mean of that distribution, so hedging is punished only by this
                        # term.  Measured on clean data: 42.35 -> 37.89 px at 8 epochs.
                        loss, parts = localisation_losses(
                            heatmap_per_sample=heatmap_loss_per_sample(
                                out["heatmap_logits"], target_heatmap
                            ),
                            # w_coord=0 disables the term without changing the other
                            # reductions (it was an exact zero before).
                            coord_per_sample=coordinate_loss_per_sample(
                                out["self_feat"], truth_self
                            ),
                            self_loss=loss_self,
                            danger_loss=loss_danger,
                            weights=batch_position_weights(index),
                            w_heatmap=cfg.w_heatmap,
                            w_coord=cfg.w_coord,
                            w_self=cfg.w_self,
                            w_danger=cfg.w_danger,
                        )
                        loss_heatmap = parts["heatmap"]
                        loss_coord = parts["coord"]
                    else:
                        loss_self = F.smooth_l1_loss(out["self_feat"], truth_self)
                        loss = loss_self + cfg.w_danger * loss_danger
                    if training:
                        optim.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optim.step()
                total_loss += float(loss.detach()) * batch_size
                seen += batch_size
                preds.append(out["self_feat"].detach().cpu().numpy())
                truths.append(labels["self_feat"])
                pred_danger.append(out["danger_log"].detach().cpu().numpy())
                true_danger.append(labels["danger"])
                if "heatmap_logits" in out:
                    heatmaps.append(out["heatmap_logits"].detach().cpu().numpy())
            coverage["train" if training else "val"] = {
                **coverage_record(indices, visited),
                "drop_last": bool(training),
            }
            if not training and v104 and not cfg.max_val_batches:
                require_full_validation(coverage["val"])
            if not preds:
                return {}
            result = {"loss": round(total_loss / max(1, seen), 5), "frames": seen}
            if not training:
                result.update(
                    vision_metrics.all_state_metrics(
                        np.concatenate(preds),
                        np.concatenate(pred_danger),
                        np.concatenate(truths),
                        np.concatenate(true_danger),
                        heatmap_logits=np.concatenate(heatmaps) if heatmaps else None,
                    )
                )
            return result

        def run_policy_epoch_legacy(
            training: bool, indices: list[int], values: np.ndarray, max_batches: int = 0
        ) -> dict:
            model.train(training)
            total_loss = 0.0
            seen = 0
            preds: list[np.ndarray] = []
            truths: list[np.ndarray] = []
            bullets: list[np.ndarray] = []
            visited: list[int] = []
            for index in epoch_batches(indices, training, max_batches):
                visited.extend(index)
                regions, labels = _gather(dataset, index, cfg.depth, region_sizes)
                batch = regions_to_batch({name: value.numpy() for name, value in regions.items()}, device)
                target = np.clip(values[index].astype(np.float32), *VALUE_CLIP) / VALUE_SCALE
                with torch.set_grad_enabled(training):
                    predicted = model.score(batch)
                    loss = F.smooth_l1_loss(predicted, torch.as_tensor(target, device=device))
                    if training:
                        optim.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optim.step()
                total_loss += float(loss.detach()) * len(index)
                seen += len(index)
                preds.append(predicted.detach().cpu().numpy())
                truths.append(values[index].astype(np.float32))
                bullets.append(labels["bullets"])
            coverage["train" if training else "val"] = {
                **coverage_record(indices, visited),
                "drop_last": bool(training),
            }
            if not training and v104 and not cfg.max_val_batches:
                require_full_validation(coverage["val"])
            if not preds:
                return {}
            result = {"loss": round(total_loss / max(1, seen), 5), "frames": seen}
            if not training:
                result.update(
                    vision_metrics.policy_metrics(
                        np.concatenate(preds), np.concatenate(truths), np.concatenate(bullets)
                    )
                )
            return result

        def run_policy_epoch_v104(
            training: bool,
            indices: list[int],
            values: np.ndarray,
            policy_optim: torch.optim.Optimizer,
            max_batches: int = 0,
        ) -> dict:
            """Policy epoch under the v104 contract: policy loss + masked auxiliaries."""
            model.train(training)
            mask = self_aux_mask(device)
            total_loss = 0.0
            seen = 0
            policy_loss_total = 0.0
            aux_total = {"heatmap": 0.0, "coord": 0.0, "danger": 0.0, "self": 0.0}
            stats_total = {
                "valid_pairs": 0,
                "frames_with_pairs": 0,
                "frames_without_pairs": 0,
                "rank_loss": 0.0,
                "regression_loss": 0.0,
            }
            preds: list[np.ndarray] = []
            truths: list[np.ndarray] = []
            bullets: list[np.ndarray] = []
            visited: list[int] = []
            for index in epoch_batches(indices, training, max_batches):
                visited.extend(index)
                regions, labels = _gather(dataset, index, cfg.depth, region_sizes)
                batch = regions_to_batch({name: value.numpy() for name, value in regions.items()}, device)
                batch_size = len(index)
                # float16 labels upcast: the quantisation (and its ties) is kept.
                teacher_px = torch.as_tensor(values[index].astype(np.float32), device=device)
                truth_self = torch.as_tensor(labels["self_feat"], device=device)
                target_danger = torch.as_tensor(_danger_target(labels["danger"]), device=device)
                with torch.set_grad_enabled(training):
                    scores, aux = policy_forward(model, batch)
                    loss_policy, stats = vision_losses.policy_loss(
                        scores,
                        teacher_px,
                        plan.objective,
                        plan.rank_delta_px,
                        plan.rank_weight,
                        plan.regression_weight,
                    )
                    grid_h, grid_w = model.config.heatmap_hw
                    target_heatmap = torch.as_tensor(
                        heatmap_target(
                            labels["self_feat"][:, 0],
                            labels["self_feat"][:, 1],
                            grid_h,
                            grid_w,
                            # C5: the v3 target sigma is the model config's.
                            sigma=heatmap_sigma,
                        ),
                        device=device,
                    )
                    loss_danger = F.smooth_l1_loss(
                        aux["danger_log"].reshape(batch_size, obs.DANGER_ROWS, obs.DANGER_COLS),
                        target_danger.reshape(batch_size, obs.DANGER_ROWS, obs.DANGER_COLS),
                    )
                    loss_self = masked_self_loss(aux["self_feat"], truth_self, mask)
                    aux_loss, aux_parts = localisation_losses(
                        heatmap_per_sample=heatmap_loss_per_sample(
                            aux["heatmap_logits"], target_heatmap
                        ),
                        coord_per_sample=coordinate_loss_per_sample(aux["self_feat"], truth_self),
                        self_loss=loss_self,
                        danger_loss=loss_danger,
                        weights=batch_position_weights(index),
                        w_heatmap=cfg.policy_aux_heatmap,
                        w_coord=cfg.policy_aux_coord,
                        w_self=cfg.policy_aux_self,
                        w_danger=cfg.policy_aux_danger,
                    )
                    loss_heatmap = aux_parts["heatmap"]
                    loss_coord = aux_parts["coord"]
                    loss = loss_policy + aux_loss
                    if training:
                        policy_optim.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
                        )
                        policy_optim.step()
                total_loss += float(loss.detach()) * batch_size
                policy_loss_total += float(loss_policy.detach()) * batch_size
                aux_total["heatmap"] += float(loss_heatmap.detach()) * batch_size
                aux_total["coord"] += float(loss_coord.detach()) * batch_size
                aux_total["danger"] += float(loss_danger.detach()) * batch_size
                aux_total["self"] += float(loss_self.detach()) * batch_size
                for key in stats_total:
                    stats_total[key] += stats[key]
                seen += batch_size
                preds.append(scores.detach().cpu().numpy())
                truths.append(values[index].astype(np.float32))
                bullets.append(labels["bullets"])
            coverage["train" if training else "val"] = {
                **coverage_record(indices, visited),
                "drop_last": bool(training),
            }
            if not training and v104 and not cfg.max_val_batches:
                require_full_validation(coverage["val"])
            if not preds:
                return {}
            predicted_scores = np.concatenate(preds)
            result = {
                "loss": round(total_loss / max(1, seen), 5),
                "frames": seen,
                "policy_loss": round(policy_loss_total / max(1, seen), 6),
                "loss_stats": {
                    "objective": plan.objective,
                    "rank_loss": round(stats_total["rank_loss"] / max(1, seen), 6),
                    "regression_loss": round(stats_total["regression_loss"] / max(1, seen), 6),
                    "valid_pairs": int(stats_total["valid_pairs"]),
                    "frames_with_pairs": int(stats_total["frames_with_pairs"]),
                    "frames_without_pairs": int(stats_total["frames_without_pairs"]),
                },
                "aux": {key: round(value / max(1, seen), 6) for key, value in aux_total.items()},
                "score_std": float(predicted_scores.std(axis=1).mean()),
            }
            if not training:
                # policy_metrics is a numpy interface (S3-2 搂7.4): CUDA tensors must
                # be moved to the host first, which the predictions already are.
                result.update(
                    vision_metrics.policy_metrics(
                        predicted_scores, np.concatenate(truths), np.concatenate(bullets)
                    )
                )
            return result

        def candidate_values() -> dict:
            """Teacher target array plus the provenance block the summary records."""
            if not v104:
                if not dataset.has_value:
                    cache = cfg.teacher_values or (
                        Path(cfg.dataset).parent / f"{Path(cfg.dataset).name}-teacher-values.npz"
                    )
                    return {"values": teacher_values_for(dataset, cache), "block": {}}
                values = np.zeros((total, N_CANDIDATES), dtype=np.float16)
                for start in range(0, total, 4096):
                    stop = min(total, start + 4096)
                    values[start:stop] = dataset.labels(list(range(start, stop)))["value"]
                return {"values": values, "block": {}}
            # v104 never trusts `manifest.has_teacher_values` or a shard `value`
            # array: identity comes from the fingerprinted sidecar only.
            cache = load_or_build_teacher_values(
                cfg, dataset, manifest=manifest, val_index_sha256=val_sha
            )
            block = {
                "cache": cache["path"],
                "fingerprint": cache["fingerprint"],
                "cache_sha256": cache["cache_sha256"],
                "explicit_path": cache["explicit"],
                "rebuilt": cache["rebuilt"],
                "dtype": "float16",
                "identity": cache["identity"],
                "audit": cache["audit"],
                "manifest_has_teacher_values": bool(manifest.has_teacher_values) if manifest else False,
                "shard_value_used": False,
            }
            return {"values": cache["values"], "block": block}

        coverage: dict = {"val": {}, "train": {}}
        acceptance_eligible = (
            v104 and not max_train_batches and not cfg.max_val_batches and not cfg.screening
        )
        early_teacher: dict | None = None

        def save_best_best_pt(stage_label: str, epoch: int) -> None:
            """``best.pt`` compatibility copy; v3 needs the full C6 identity too."""
            extra = (
                checkpoint_extra(stage_label, epoch)
                if v3
                else {"stage": stage_label, "epoch": epoch}
            )
            save_vision_checkpoint(cfg.out_dir / "best.pt", model, extra)

        try:
            if v104 and v3 and cfg.stage in ("state", "both"):
                # A v3 state checkpoint must record the teacher fingerprint (C6),
                # so the fingerprinted cache is resolved before the first save.
                early_teacher = candidate_values()
                summary["teacher"] = early_teacher["block"]
                v104_context["teacher_fingerprint"] = str(early_teacher["block"].get("fingerprint", ""))
            if cfg.stage in ("state", "both"):
                best_pos = float("inf")
                best_pos_row: dict | None = None
                best_danger = -float("inf")
                best_danger_row: dict | None = None
                pass_epoch: int | None = None
                history = []
                for epoch in range(1, cfg.epochs_state + 1):
                    started_epoch = time.perf_counter()
                    train_row = run_state_epoch(True, train_idx, max_train_batches)
                    with torch.no_grad():
                        val_row = run_state_epoch(False, val_idx, cfg.max_val_batches)
                    row = state_row(epoch, train_row, val_row, round(time.perf_counter() - started_epoch, 1))
                    history.append(row)
                    print(f"[vision:state] {json.dumps(row)}", flush=True)
                    position = row.get("val_player_pos_px")
                    danger = row.get("val_danger_corr_perframe_median")
                    if position is not None and position < best_pos:
                        best_pos = position
                        best_pos_row = dict(row)
                        save_vision_checkpoint(
                            cfg.out_dir / "state.pt", model, checkpoint_extra("state", epoch, val_row)
                        )
                        save_best_best_pt("state", epoch)
                    if danger is not None and danger > best_danger:
                        best_danger = danger
                        best_danger_row = dict(row)
                        save_vision_checkpoint(
                            cfg.out_dir / "state-bestdanger.pt",
                            model,
                            checkpoint_extra("state-danger", epoch, val_row),
                        )
                        save_best_best_pt("state", epoch)
                    if (
                        pass_epoch is None
                        and position is not None
                        and danger is not None
                        and position < A01_POS_THRESHOLD
                        and danger > A01_PERFRAME_THRESHOLD
                    ):
                        # Written once: a later epoch that is better at only one of the
                        # two bars must not erase the joint pass.
                        pass_epoch = epoch
                        save_vision_checkpoint(
                            cfg.out_dir / "state-pass.pt",
                            model,
                            checkpoint_extra(
                                "state-pass",
                                epoch,
                                {"a01": {"pos_px": position, "perframe_median": danger, "epoch": epoch}},
                            ),
                        )
                        save_best_best_pt("state-pass", epoch)
                state_block = {
                    "history": history,
                    "best": best_pos_row or {},
                    "final": history[-1] if history else {},
                    # extra provenance: which epoch the other two checkpoints hold
                    "best_danger": best_danger_row or {},
                    "pass_epoch": pass_epoch,
                }
                if v104:
                    selected, rule = state_selection_row(history, pass_epoch)
                    if selected is not None:
                        chosen = dict(selected)
                        chosen["selection_rule"] = rule
                        chosen.update(
                            _checkpoint_record(
                                cfg.out_dir / ("state-pass.pt" if rule == "joint-pass" else "state.pt")
                            )
                        )
                        state_block["best"] = chosen
                        state_block["selection_rule"] = rule
                    else:
                        state_block["best"] = {}
                        state_block["selection_rule"] = "none"
                    if best_pos_row is not None:
                        position_row = dict(best_pos_row)
                        position_row.update(_checkpoint_record(cfg.out_dir / "state.pt"))
                        state_block["best_position"] = position_row
                    if best_danger_row is not None:
                        danger_row = dict(best_danger_row)
                        danger_row.update(_checkpoint_record(cfg.out_dir / "state-bestdanger.pt"))
                        state_block["best_danger"] = danger_row
                    state_block["pass_epoch"] = pass_epoch
                summary["stages"]["state"] = state_block

            if cfg.stage in ("policy", "both"):
                teacher = early_teacher if early_teacher is not None else candidate_values()
                values = teacher["values"]
                if v104:
                    summary["teacher"] = teacher["block"]
                if v104:
                    source_state = _load_policy_source(cfg, cfg.out_dir, model, region_sizes, device)
                    summary["source_state"] = source_state
                    v104_context["source_state_sha256"] = source_state["sha256"]
                    v104_context["teacher_fingerprint"] = summary.get("teacher", {}).get("fingerprint", "")
                    print(f"[vision] policy source state: {source_state['path']}", flush=True)
                best_key: tuple | None = None
                best_row: dict | None = None
                history = []
                policy_optim, optim_record = (None, {})
                if v104:
                    frozen_phase = cfg.policy_freeze_epochs > 0
                    policy_optim, optim_record = build_policy_optimizer(model, cfg, frozen=frozen_phase)
                    summary["optim"] = {
                        "weight_decay": float(cfg.weight_decay),
                        "encoder_lr": float(cfg.encoder_lr),
                        "policy_head_lr": float(cfg.policy_head_lr),
                        "freeze_epochs": int(cfg.policy_freeze_epochs),
                        "initial": optim_record,
                        "switch_epoch": int(cfg.policy_freeze_epochs) + 1
                        if cfg.policy_freeze_epochs < cfg.epochs_policy
                        else None,
                        "joint_phase_reached": bool(cfg.epochs_policy > cfg.policy_freeze_epochs),
                        "rebuilds": [],
                    }
                for epoch in range(1, cfg.epochs_policy + 1):
                    if v104 and epoch == cfg.policy_freeze_epochs + 1:
                        # Frozen -> joint: rebuild so every group carries its own LR
                        # instead of relying on a per-parameter override.
                        policy_optim, optim_record = build_policy_optimizer(model, cfg, frozen=False)
                        summary["optim"]["joint"] = optim_record
                        summary["optim"]["rebuilds"].append(
                            {"epoch": epoch, "from": "frozen", "to": "joint", "groups": optim_record["groups"]}
                        )
                        print(f"[vision:policy] optimizer rebuilt for joint training at epoch {epoch}", flush=True)
                    started_epoch = time.perf_counter()
                    if v104:
                        train_row = run_policy_epoch_v104(
                            True, train_idx, values, policy_optim, max_train_batches
                        )
                        with torch.no_grad():
                            val_row = run_policy_epoch_v104(
                                False, val_idx, values, policy_optim, cfg.max_val_batches
                            )
                    else:
                        train_row = run_policy_epoch_legacy(True, train_idx, values, max_train_batches)
                        with torch.no_grad():
                            val_row = run_policy_epoch_legacy(False, val_idx, values, cfg.max_val_batches)
                    row = policy_row(epoch, train_row, val_row, round(time.perf_counter() - started_epoch, 1))
                    history.append(row)
                    print(f"[vision:policy] {json.dumps(row)}", flush=True)
                    if not v104:
                        # Model selection uses **regret**, not loss.  Measured on this
                        # dataset: loss kept falling while top-1 agreement and regret got
                        # worse after epoch 4, so selecting by loss picks a strictly
                        # worse policy (it is more confident about the wrong action).
                        regret = row.get("val_regret_px")
                        if regret is not None and regret < (best_key[1] if best_key else float("inf")):
                            best_key = (None, regret, epoch)
                            best_row = dict(row)
                            save_vision_checkpoint(
                                cfg.out_dir / "policy.pt", model, checkpoint_extra("policy", epoch, row)
                            )
                            save_best_best_pt("policy", epoch)
                    else:
                        key = policy_selection_key(row)
                        if key is not None and (best_key is None or key < best_key):
                            best_key = key
                            best_row = dict(row)
                            save_vision_checkpoint(
                                cfg.out_dir / "policy.pt",
                                model,
                                checkpoint_extra("policy", epoch, {"selection_key": list(key)}),
                            )
                            save_best_best_pt("policy", epoch)
                policy_block = {
                    "history": history,
                    "best": best_row or {},
                    # last epoch only; acceptance reads ``best``, never this
                    "final": history[-1] if history else {},
                }
                if v104:
                    if best_row is not None:
                        chosen = dict(best_row)
                        chosen["selection_rule"] = "bucket-0-15-regret"
                        chosen["selection_key"] = list(best_key)
                        chosen.update(_checkpoint_record(cfg.out_dir / "policy.pt"))
                        policy_block["best"] = chosen
                    else:
                        policy_block["best"] = {}
                    policy_block["selection_rule"] = (
                        "bucket-0-15-regret" if best_row is not None else "none"
                    )
                summary["stages"]["policy"] = policy_block
        except BaseException as error:
            summary["partial"] = True
            summary["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            if v104:
                summary.setdefault("partial", False)
                summary["contract_version"] = CONTRACT_VERSION
                # Eligibility is not a label someone types: it needs a run without
                # batch limits *and* a validation pass that really visited every
                # held-out index exactly once (D9).  The training record is audited
                # but never gates: drop_last deliberately keeps full batches only.
                summary["acceptance_eligible"] = bool(
                    acceptance_eligible and coverage.get("val", {}).get("complete")
                )
                summary["recipe"] = {
                    "selection_contract": cfg.selection_contract,
                    "history_contract": cfg.history_contract,
                    "stage": cfg.stage,
                    "epochs_state": cfg.epochs_state,
                    "epochs_policy": cfg.epochs_policy,
                    "batch_size": cfg.batch_size,
                    "depth": cfg.depth,
                    "seed": cfg.seed,
                    "max_train_batches": max_train_batches,
                    "max_val_batches": cfg.max_val_batches,
                    "screening": bool(max_train_batches or cfg.max_val_batches or cfg.screening),
                    "train_fraction": fraction,
                }
                summary["data"] = {
                    "dataset": _relative_root(cfg.dataset),
                    "frames": total,
                    "train_frames": len(train_idx),
                    "val_frames": len(val_idx),
                    "val_index_sha256": val_sha,
                    "manifest_sha256": _sha256_file(Path(cfg.dataset) / "manifest.json")
                    if (Path(cfg.dataset) / "manifest.json").is_file()
                    else "",
                    "manifest_verified": manifest_verified,
                    "history_contract": cfg.history_contract,
                    "history_audit_depth3_available": True,
                }
                summary["input"] = {
                    "contract": "rgb255-v1",
                    "arch": cfg.arch,
                    "depth": int(cfg.depth),
                    "regions": {name: list(size) for name, size in region_sizes.items()},
                    "frame_shape": {
                        name: [height, width, vision.VISION_CHANNELS]
                        for name, (height, width) in region_sizes.items()
                    },
                    "candidate_order_version": CANDIDATE_ORDER_VERSION,
                    "score_semantics": "preference_logits",
                    # v3 E4/E5 ablation, R3 head variant and the single source of
                    # the heatmap target sigma (C5: from the model config for v3).
                    "local_enabled": bool(cfg.local_enabled),
                    "danger_head_variant": cfg.danger_head_variant,
                    "heatmap_sigma": heatmap_sigma,
                }
                summary["loss"] = {
                    "policy": plan.as_dict(),
                    "aux_perception": {
                        "heatmap": float(cfg.w_heatmap),
                        "coord": float(cfg.w_coord),
                        "danger": float(cfg.w_danger),
                        "self": float(cfg.w_self),
                    },
                    "aux_policy": {
                        "heatmap": float(cfg.policy_aux_heatmap),
                        "coord": float(cfg.policy_aux_coord),
                        "danger": float(cfg.policy_aux_danger),
                        "self": float(cfg.policy_aux_self),
                    },
                    "self_mask_indices": list(SELF_AUX_INDICES),
                    "self_mask_dims": len(SELF_AUX_INDICES),
                    "danger_scale": DANGER_SCALE,
                    "heatmap_sigma": heatmap_sigma,
                }
                summary["coverage"] = coverage
                summary["hardware"] = {
                    "device": str(device),
                    "torch_threads": torch.get_num_threads(),
                    "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated())
                    if device.type == "cuda"
                    else 0,
                    "elapsed_seconds": round(time.perf_counter() - started_clock, 2),
                }
            summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            (cfg.out_dir / "vision-summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    finally:
        dataset.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the vision student")
    parser.add_argument("--dataset", required=True, help="dataset directory (manifest.json + shards)")
    parser.add_argument("--out", required=True, help="experiment directory; frozen baselines are refused")
    parser.add_argument(
        "--arch",
        default="v2",
        choices=list(ARCHES),
        help="vision architecture: v1 | v2 (default) | v3-st (v104 contract only)",
    )
    parser.add_argument("--stage", default="both", choices=["state", "policy", "both"])
    parser.add_argument("--epochs-state", type=int, default=8)
    parser.add_argument("--epochs-policy", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (both stages); exposed so a v104 recipe can state it",
    )
    parser.add_argument("--depth", type=int, default=1, help="frame stack depth (input channels = 3*depth)")
    parser.add_argument("--device", default="cuda", help="cuda or cpu; falls back to cpu when cuda is absent")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="fraction of the training split to use; requires a verified manifest when < 1.0",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="only used by the --allow-missing-manifest fallback split",
    )
    parser.add_argument(
        "--max-train-batches", type=int, default=0, help="0 = unlimited; limits training only"
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="0 = unlimited; limits validation only (0 for experiments)",
    )
    parser.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=None,
        help="deprecated alias of --max-train-batches (ignored when --max-train-batches is non-zero)",
    )
    parser.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="derive the split on the fly; synthetic test data only",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--heatmap-sigma", type=float, default=1.5, help="v2 heatmap target sigma, in cells")
    parser.add_argument("--w-heatmap", type=float, default=1.0, help="v2 heatmap loss weight")
    parser.add_argument("--w-coord", type=float, default=1.0, help="v2 coordinate loss weight (0 disables)")
    parser.add_argument("--torch-threads", type=int, default=0, help="torch intra-op threads (0 = auto: 1 on cuda)")
    parser.add_argument("--playfield-hw", default="", help="playfield input size HxW, e.g. 336x288 (default 224x192)")
    parser.add_argument("--w-self", type=float, default=1.0, help="v2 self-head loss weight")
    parser.add_argument("--w-danger", type=float, default=1.0, help="danger loss weight (v1 and v2)")
    parser.add_argument("--pf-width", type=int, default=32, help="v2 playfield first-layer channels")
    parser.add_argument("--hud-width", type=int, default=16, help="v2 HUD first-layer channels")
    parser.add_argument("--width", type=int, default=32, help="v1 first-layer channels")
    parser.add_argument("--trunk-hidden", type=int, default=256)
    parser.add_argument("--init-from", default="", help="optional warm-start checkpoint")
    parser.add_argument("--note", default="", help="free-form note written to summary['note']")
    # v1.0.4 opt-in contract (defaults = v1.0.3 behaviour)
    parser.add_argument(
        "--selection-contract",
        default="legacy",
        choices=list(SELECTION_CONTRACTS),
        help="legacy = v1.0.3 selection; v104 = the written v1.0.4 contract",
    )
    parser.add_argument(
        "--history-contract",
        default="legacy",
        choices=list(HISTORY_CONTRACTS),
        help="legacy = v1.0.3 shard clamp; causal-segment-v1 = the v1.0.4 causal segments",
    )
    parser.add_argument(
        "--policy-objective",
        default="legacy-reg",
        choices=list(vision_losses.OBJECTIVES),
        help="policy loss: legacy-reg | centered-reg | rank (requires --selection-contract v104)",
    )
    parser.add_argument("--rank-delta-px", type=float, default=0.5)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--policy-aux-heatmap", type=float, default=0.1)
    parser.add_argument("--policy-aux-coord", type=float, default=1.0)
    parser.add_argument("--policy-aux-danger", type=float, default=0.2)
    parser.add_argument("--policy-aux-self", type=float, default=0.1)
    parser.add_argument("--policy-freeze-epochs", type=int, default=2)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--policy-head-lr", type=float, default=3e-4)
    parser.add_argument(
        "--teacher-values",
        default="",
        help="explicit teacher cache; must carry this run's v104 fingerprint or the run stops",
    )
    parser.add_argument(
        "--local-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="v3-st only: --no-local-enabled is the E4 ablation (zero local vector, same slots)",
    )
    parser.add_argument(
        "--danger-head-variant",
        default=DANGER_HEAD_LEGACY,
        choices=list(DANGER_HEAD_VARIANTS),
        help="v2 only (R3): legacy (default) | learnable-v104",
    )
    parser.add_argument(
        "--position-sample-weights",
        default="",
        help="R2: NPZ with A's train_indices/errors/weights plus its JSON identity",
    )
    parser.add_argument(
        "--screening",
        action="store_true",
        help="declare a screening run (24 perception epochs): never acceptance eligible",
    )
    args = parser.parse_args(argv)

    max_train_batches = args.max_train_batches
    if args.max_batches_per_epoch is not None and max_train_batches == 0:
        max_train_batches = args.max_batches_per_epoch

    cfg = VisionTrainConfig(
        dataset=Path(args.dataset),
        out_dir=Path(args.out),
        arch=args.arch,
        stage=args.stage,
        epochs_state=args.epochs_state,
        epochs_policy=args.epochs_policy,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        depth=args.depth,
        # ``--seed`` used to be accepted by argparse and then dropped here, so
        # every run silently trained with the dataclass default (7).  Pinned by
        # test_v104_refine.py::test_cli_seed_reaches_the_config and
        # ::test_cli_flags_are_not_silently_ignored.
        seed=args.seed,
        device=args.device,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        allow_missing_manifest=args.allow_missing_manifest,
        max_train_batches=max_train_batches,
        max_val_batches=args.max_val_batches,
        heatmap_sigma=args.heatmap_sigma,
        w_heatmap=args.w_heatmap,
        w_self=args.w_self,
        w_danger=args.w_danger,
        # Same defect class as ``--seed`` above: ``--w-coord`` was parsed and then
        # dropped, so ``cfg.w_coord`` was stuck at 1.0 and the documented
        # "0 disables" switch did nothing.  Pinned by
        # test_v104_refine.py::test_cli_flags_are_not_silently_ignored and
        # ::test_every_cli_flag_is_read_when_building_the_config.
        w_coord=args.w_coord,
        torch_threads=args.torch_threads,
        playfield_hw=(tuple(int(v) for v in args.playfield_hw.lower().split('x')) if args.playfield_hw else None),
        pf_width=args.pf_width,
        hud_width=args.hud_width,
        width=args.width,
        trunk_hidden=args.trunk_hidden,
        init_from=Path(args.init_from) if args.init_from else None,
        teacher_values=Path(args.teacher_values) if args.teacher_values else None,
        note=args.note,
        selection_contract=args.selection_contract,
        history_contract=args.history_contract,
        policy_objective=args.policy_objective,
        rank_delta_px=args.rank_delta_px,
        rank_weight=args.rank_weight,
        regression_weight=args.regression_weight,
        policy_aux_heatmap=args.policy_aux_heatmap,
        policy_aux_coord=args.policy_aux_coord,
        policy_aux_danger=args.policy_aux_danger,
        policy_aux_self=args.policy_aux_self,
        policy_freeze_epochs=args.policy_freeze_epochs,
        encoder_lr=args.encoder_lr,
        policy_head_lr=args.policy_head_lr,
        local_enabled=bool(args.local_enabled),
        danger_head_variant=args.danger_head_variant,
        position_sample_weights=Path(args.position_sample_weights)
        if args.position_sample_weights
        else None,
        screening=bool(args.screening),
    )
    try:
        summary = train(cfg)
    except ManifestError as error:
        # A known manifest/IO contract failure: message to stderr, exit 2, no
        # traceback (瑷▓ C6).  Unknown exceptions keep their traceback.
        print(str(error), file=sys.stderr, flush=True)
        return 2
    except ValueError as error:
        message = str(error)
        if message.startswith("v104:"):
            print(message, file=sys.stderr, flush=True)
            return 2
        raise
    print(json.dumps({k: v for k, v in summary.items() if k != "stages"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
