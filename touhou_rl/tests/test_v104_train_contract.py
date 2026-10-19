"""v1.0.4 training contract (S3-3): D9 coverage, teacher identity, v104 selection.

Why a second training test file
-------------------------------
``test_vision_train.py`` pins the v1.0.3 behaviour that must stay reproducible
(legacy selection, manifest gate, batch limits, the old checkpoint names).  This
file pins the *opt-in* v1.0.4 contract of `设计文档` §4 C1-C6 and the S3-3 task
book, which only activates with ``selection_contract='v104'``:

* **D9** - 7241 validation frames at batch 96 are 76 batches (75 x 96 + 41), the
  run records ``actual_frames`` / ``visited_index_sha256``, a full validation
  raises the exact ``v104:COVERAGE`` error when an index is missed, and any batch
  limit marks the run ``acceptance_eligible=false``;
* **teacher identity** - values come from
  ``<dataset>-teacher-values-v104-<fingerprint>.npz`` with a ``v104-teacher/1``
  sidecar; the fingerprint binds shard digests, manifest/split digests, the
  candidate order, the full ``PilotConfig`` and the teacher source; the old
  unfingerprinted cache is neither trusted nor overwritten, and an explicit
  ``--teacher-values`` that does not match is a ``v104:TEACHER`` stop;
* **loss / auxiliary** - non-rank objectives are forced to (0, 1), a rank
  objective without an explicit weight 1 is refused, the self auxiliary is masked
  to the 19 original columns ``4..12`` + ``14..23``, and the fused policy forward
  returns exactly what ``score()`` + ``decode_state()`` return;
* **freeze -> joint** - the first ``policy_freeze_epochs`` epochs train only the
  candidate head, the optimizer is rebuilt at the switch with the contract's
  learning rates, and a 1-epoch screening run says it never reached the joint
  phase;
* **selection** - ``state.best`` prefers the first joint A-01 pass, ``policy.pt``
  is chosen by ``(0-15 bucket regret, global regret, earlier epoch)``, an empty
  bucket cannot be selected, and every ``best`` row carries
  ``selection_rule/checkpoint/checkpoint_sha256`` that match the bytes on disk;
* **write protection** - an existing valid summary/checkpoint and the frozen
  ``pilot-v1`` / ``v104-a01-72ep`` / ``v104-clean-24e12e`` directories are refused;
* **error format** - ``v104:<CODE>: <detail>``, the CLI maps known errors to
  exit 2 while unknown exceptions keep their traceback.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from touhou_rl import obs, vision, vision_metrics, vision_train
from touhou_rl.pipeline.manifest import DatasetManifest, ShardRecord, file_digest, index_digest, split_indices
from touhou_rl.vision_data import META_FIELDS, VisionDataset, VisionShardWriter
from touhou_rl.vmodels import load_vision_checkpoint
from touhou_rl.vmodels2 import VisionNetConfigV2, VisionScorerV2

REGION_HW = {"playfield": (56, 48), "hud_score": (17, 54), "hud_status": (31, 54)}
STAGES = (0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2)
FRAMES = 30
#: The real A split, referenced by the task book and the requirements doc.
A_VAL_INDEX_SHA256 = "e5d4ebbe6d8de05793520a393fc1fdbea2b1e2c4db9ff7c1632908340c76655f"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _png(value: int, height: int, width: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = value % 256
    image[: max(1, height // 4), : max(1, width // 4), 1] = (value * 3) % 256
    return vision.encode_png(image)


def _observation(index: int) -> obs.Observation:
    rng = np.random.default_rng(2000 + index)
    self_feat = rng.random(obs.N_SELF, dtype=np.float32) * 0.5
    bullets = rng.random((obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float32).astype(np.float16)
    bullets[:, 5] = (rng.random(obs.N_BULLETS) > 0.25).astype(np.float16)
    return obs.Observation(
        self_feat=self_feat,
        bullets=bullets,
        danger_map=rng.integers(0, 8, size=(obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8),
    )


def _dataset(root: Path, *, frames: int = FRAMES, pixel_offset: int = 0, shard_size: int = 6) -> Path:
    """Tiny three-region dataset with a causal history (meta frame delta = 2)."""
    dataset_dir = root / "ds"
    writer = VisionShardWriter(dataset_dir, shard_size=shard_size, tag="v104")
    for index in range(frames):
        observation = _observation(index)
        meta = {name: 0 for name in META_FIELDS}
        # frame step 2 keeps ``causal-segment-v1`` segments alive inside a stage,
        # so depth-3 stacks contain real history and not only clamped repeats.
        meta["frame"] = 2 * index
        meta["stage"] = STAGES[index % len(STAGES)]
        regions = {
            name: _png(pixel_offset + index * 9 + offset, *hw)
            for offset, (name, hw) in enumerate(REGION_HW.items())
        }
        writer.add(regions, observation, (index % 9, 1, 0, 0), meta)
    writer.close()
    return dataset_dir


def _write_manifest(dataset_dir: Path, *, val_fraction: float = 0.2) -> DatasetManifest:
    records = []
    for path in sorted(dataset_dir.glob("*.npz")):
        with np.load(path) as data:
            frames = int(data["action"].shape[0])
        records.append(
            ShardRecord(name=path.name, frames=frames, bytes=path.stat().st_size, sha256=file_digest(path))
        )
    manifest = DatasetManifest.build(
        modality="vision",
        root=dataset_dir,
        shards=records,
        frame_shape=None,
        has_teacher_values=True,
        teacher={},
        seed=7,
        val_fraction=val_fraction,
        stats={},
        provenance={},
    )
    manifest.save(dataset_dir / "manifest.json")
    return manifest


def _cfg(dataset_dir: Path, out_dir: Path, **overrides) -> vision_train.VisionTrainConfig:
    values = {
        "dataset": dataset_dir,
        "out_dir": out_dir,
        "arch": "v2",
        "stage": "state",
        "epochs_state": 1,
        "epochs_policy": 1,
        "batch_size": 2,
        "depth": 1,
        "device": "cpu",
        "val_fraction": 0.2,
        "allow_missing_manifest": True,
        "max_train_batches": 2,
        "max_val_batches": 0,
        "pf_width": 8,
        "hud_width": 4,
        "width": 8,
        "trunk_hidden": 32,
        "playfield_hw": (56, 48),
        "policy_freeze_epochs": 0,
    }
    values.update(overrides)
    return vision_train.VisionTrainConfig(**values)


def _v104(dataset_dir: Path, out_dir: Path, **overrides) -> vision_train.VisionTrainConfig:
    overrides.setdefault("selection_contract", "v104")
    return _cfg(dataset_dir, out_dir, **overrides)


def _extra(path: Path) -> dict:
    return load_vision_checkpoint(path, map_location="cpu")[1]


def _fingerprint_of(dataset_dir: Path, *, val_sha: str = "") -> str:
    manifest_path = Path(dataset_dir) / "manifest.json"
    manifest = DatasetManifest.load(manifest_path) if manifest_path.is_file() else None
    with VisionDataset(dataset_dir) as dataset:
        identity = vision_train.teacher_identity(
            dataset, dataset_dir, manifest=manifest, val_index_sha256=val_sha
        )
    return vision_train._fingerprint(identity)


def _bucket_row(label: str, frames: int, regret: float) -> dict:
    return {
        "bucket": label,
        "frames": frames,
        "top1": 0.0,
        "top3": 0.0,
        "top3_tie": 0.0,
        "near_optimal_05": 0.0,
        "regret_px": regret if frames else 0.0,
        "regret_p90_px": regret if frames else 0.0,
    }


def _policy_metrics_script(rows: list[tuple[float, float, int]]) -> list[dict]:
    """``[(bucket_0_15_regret, global_regret, bucket_frames)]`` -> metric dicts."""
    scripted = []
    for bucket_regret, global_regret, bucket_frames in rows:
        scripted.append(
            {
                "top1": 0.1,
                "top3": 0.2,
                "top3_tie": 0.3,
                "near_optimal_05": 0.05,
                "regret_px": global_regret,
                "regret_p90_px": global_regret + 1.0,
                "by_danger": [
                    _bucket_row("0-15", bucket_frames, bucket_regret),
                    _bucket_row("15-30", 2, 3.0),
                    _bucket_row("30-60", 0, 0.0),
                    _bucket_row("60-120", 0, 0.0),
                    _bucket_row("120+", 0, 0.0),
                ],
            }
        )
    return scripted


def _fake_state_metrics(pos_px: float, median: float) -> dict:
    return {
        "self_mae": 0.1,
        "player_pos_px": pos_px,
        "player_pos_p95_px": pos_px * 2.0,
        "ray_mae": 0.2,
        "bullet_count_mae": 3.0,
        "danger_mae_log": 0.3,
        "danger_corr": 0.4,
        "danger_corr_perframe_mean": median,
        "danger_corr_perframe_median": median,
        "danger_corr_perframe_valid": 6,
        "danger_iou": 0.5,
        "danger_cell_acc": 0.6,
        "heatmap_hit_rate": 0.7,
    }


# --------------------------------------------------------------------------- #
# D9: batching, tail coverage, coverage hashes
# --------------------------------------------------------------------------- #
def test_a_validation_split_is_7241_frames_in_76_batches_with_a_41_tail():
    """The exact numbers D9 is about, derived from the frozen A split."""
    _train, val = split_indices(72413, 7, 0.1)
    assert len(val) == 7241
    batches = list(vision_train.make_batches(val, batch_size=96, drop_last=False))
    assert len(batches) == 76
    assert [len(batch) for batch in batches[-2:]] == [96, 41]
    visited = [index for batch in batches for index in batch]
    record = vision_train.coverage_record(val, visited)
    assert record["complete"] is True
    assert record["actual_frames"] == 7241
    assert record["visited_index_sha256"] == index_digest(val) == A_VAL_INDEX_SHA256


def test_drop_last_training_keeps_only_full_batches():
    train, _val = split_indices(72413, 7, 0.1)
    batches = list(vision_train.make_batches(train, batch_size=96, drop_last=True))
    assert len(batches) == len(train) // 96 == 678
    assert {len(batch) for batch in batches} == {96}
    assert sum(len(batch) for batch in batches) == 678 * 96
    # the tail belongs to validation accounting, not to the optimiser
    assert 678 * 96 < len(train)


def test_small_and_limited_batches():
    assert list(vision_train.make_batches([0, 1, 2], batch_size=5, drop_last=True)) == []
    assert list(vision_train.make_batches([0, 1, 2], batch_size=5, drop_last=False)) == [[0, 1, 2]]
    limited = list(vision_train.make_batches(list(range(100)), batch_size=10, drop_last=False, max_batches=2))
    assert len(limited) == 2
    with pytest.raises(ValueError) as error:
        list(vision_train.make_batches([0], batch_size=0))
    assert "v104:CONFIG" in str(error.value)


def test_coverage_detects_duplicates_and_missing_indices():
    expected = [0, 1, 2, 3]
    duplicated = vision_train.coverage_record(expected, [0, 1, 2, 2])
    assert duplicated["complete"] is False
    assert duplicated["duplicated_indices"] == 1
    assert duplicated["actual_frames"] == 4
    missing = vision_train.coverage_record(expected, [0, 1, 2])
    assert missing["complete"] is False
    assert missing["missing_indices"] == 1
    assert missing["visited_index_sha256"] != missing["expected_index_sha256"]
    with pytest.raises(ValueError) as error:
        vision_train.require_full_validation(duplicated)
    assert str(error.value) == "v104:COVERAGE: validation indices are incomplete or duplicated"


def test_empty_splits_are_data_errors():
    with pytest.raises(ValueError) as error:
        vision_train.require_non_empty_split([], [1, 2])
    assert str(error.value) == "v104:DATA: training split is empty"
    with pytest.raises(ValueError) as error:
        vision_train.require_non_empty_split([1, 2], [])
    assert str(error.value) == "v104:DATA: validation split is empty"
    vision_train.require_non_empty_split([1], [2])  # the legal case does not raise


def test_validation_loss_is_weighted_by_the_actual_batch_size(tmp_path, monkeypatch):
    """A 4-frame and a 2-frame validation batch must not have equal weight."""
    dataset_dir = _dataset(tmp_path / "a")
    real_smooth_l1 = F.smooth_l1_loss
    scripted = [1.0, 4.0]
    calls: list[float] = []

    def fake_smooth_l1(predicted, target, **kwargs):
        if torch.is_grad_enabled():  # training batches keep the real loss
            return real_smooth_l1(predicted, target, **kwargs)
        value = scripted[min(len(scripted) - 1, len(calls))]
        calls.append(value)
        return torch.tensor(value, dtype=torch.float32)

    monkeypatch.setattr(vision_train.F, "smooth_l1_loss", fake_smooth_l1)
    summary = vision_train.train(
        _cfg(
            dataset_dir,
            tmp_path / "run",
            stage="policy",
            epochs_policy=1,
            batch_size=4,
            max_train_batches=1,
            max_val_batches=0,
        )
    )
    row = summary["stages"]["policy"]["history"][0]
    assert row["val_frames"] == 6
    assert calls == [1.0, 4.0]  # batches of 4 and 2
    # (4*1.0 + 2*4.0) / 6 = 2.0; the unweighted mean (2.5) is what a batch-average bug reports
    assert row["val_loss"] == pytest.approx(2.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# configuration / loss weights / self mask
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"selection_contract": "bogus"}, "selection_contract"),
        ({"history_contract": "bogus"}, "history_contract"),
        ({"policy_objective": "bogus"}, "policy_objective"),
        ({"selection_contract": "v104", "arch": "v1"}, "requires arch='v2'"),
        ({"selection_contract": "v104", "depth": 2}, "depth 1 or 3"),
        ({"depth": 0}, "depth must be a positive integer"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"lr": 0.0}, "lr must be > 0"),
        ({"encoder_lr": -1.0}, "encoder_lr"),
        ({"policy_aux_self": -0.1}, "policy_aux_self"),
        ({"policy_freeze_epochs": -1}, "policy_freeze_epochs"),
        ({"epochs_state": 0, "stage": "state"}, "needs epochs_state >= 1"),
        ({"policy_objective": "rank"}, "requires selection_contract='v104'"),
        (
            {"selection_contract": "v104", "policy_objective": "rank", "rank_weight": 0.0},
            "rank_weight=1",
        ),
    ],
)
def test_illegal_recipes_are_v104_config_errors(tmp_path, overrides, fragment):
    cfg = _cfg(tmp_path / "ds", tmp_path / "run", **overrides)
    with pytest.raises(ValueError) as error:
        vision_train.validate_config(cfg)
    message = str(error.value)
    assert message.startswith("v104:CONFIG: ")
    assert fragment in message


def test_non_rank_objectives_force_regression_weights(tmp_path):
    """S3-2 §7.4: a regression objective must be called with (rank=0, reg=1)."""
    cfg = _cfg(
        tmp_path / "ds",
        tmp_path / "run",
        selection_contract="v104",
        policy_objective="legacy-reg",
        rank_weight=1.0,
        regression_weight=0.4,
    )
    plan = vision_train.validate_config(cfg)
    assert (plan.rank_weight, plan.regression_weight) == (0.0, 1.0)
    assert plan.forced is True
    rank = vision_train.validate_config(
        _cfg(
            tmp_path / "ds",
            tmp_path / "run",
            selection_contract="v104",
            policy_objective="rank",
            rank_weight=1.0,
            regression_weight=0.1,
        )
    )
    assert (rank.rank_weight, rank.regression_weight) == (1.0, 0.1)
    assert rank.forced is False


def test_self_auxiliary_is_masked_to_the_nineteen_original_columns():
    mask = vision_train.self_aux_mask()
    assert mask.dtype == torch.long
    assert mask.tolist() == list(range(4, 13)) + list(range(14, 24))
    assert len(mask) == 19
    assert {2, 3, 13}.isdisjoint(set(mask.tolist()))
    torch.manual_seed(0)
    predicted = torch.randn(4, 24, requires_grad=True)
    truth = torch.randn(4, 24)
    masked = vision_train.masked_self_loss(predicted, truth, mask)
    manual = F.smooth_l1_loss(predicted.index_select(1, mask), truth.index_select(1, mask))
    assert float(masked.detach()) == float(manual.detach())
    # the old [2:24] slice is a different loss, so the mask is really applied
    legacy_slice = F.smooth_l1_loss(predicted[:, 2:24], truth[:, 2:24])
    assert float(legacy_slice.detach()) != float(masked.detach())
    masked.backward()
    gradient = predicted.grad.abs().sum(dim=0)
    assert float(gradient[2]) == 0.0 and float(gradient[3]) == 0.0 and float(gradient[13]) == 0.0
    assert float(gradient[4]) > 0.0 and float(gradient[23]) > 0.0


def test_policy_forward_equals_the_public_model_api():
    """The shared encode must not change what score()/decode_state() return."""
    model = VisionScorerV2(
        VisionNetConfigV2(depth=3, pf_width=8, hud_width=4, trunk_hidden=32, playfield_hw=(56, 48))
    ).eval()
    regions = {
        "playfield": torch.rand(2, 3, 56, 48, 3),
        "hud_score": torch.rand(2, 3, 34, 216, 3),
        "hud_status": torch.rand(2, 3, 126, 216, 3),
    }
    with torch.no_grad():
        scores, aux = vision_train.policy_forward(model, regions)
        public_scores = model.score(regions)
        state = model.decode_state(regions)
    assert torch.equal(scores, public_scores)
    assert torch.equal(aux["heatmap_logits"], state["heatmap_logits"])
    assert torch.equal(aux["danger_log"], state["danger_log"])
    assert torch.equal(aux["self_feat"], state["self_feat"])


# --------------------------------------------------------------------------- #
# teacher identity and the fingerprinted cache
# --------------------------------------------------------------------------- #
def test_teacher_fingerprint_binds_data_split_and_teacher_source(tmp_path, monkeypatch):
    dataset_dir = _dataset(tmp_path / "a")
    baseline = _fingerprint_of(dataset_dir, val_sha="split-1")
    assert _fingerprint_of(dataset_dir, val_sha="split-1") == baseline
    assert _fingerprint_of(dataset_dir, val_sha="split-2") != baseline
    other = _dataset(tmp_path / "b", pixel_offset=37)
    with VisionDataset(dataset_dir) as dataset:
        base_identity = vision_train.teacher_identity(
            dataset, dataset_dir, manifest=None, val_index_sha256="split-1"
        )
    with VisionDataset(other) as dataset:
        other_identity = vision_train.teacher_identity(
            dataset, other, manifest=None, val_index_sha256="split-1"
        )
    # the bytes that would be decoded are part of the identity, not just the path
    assert [row["sha256"] for row in other_identity["dataset"]["shards"]] != [
        row["sha256"] for row in base_identity["dataset"]["shards"]
    ]
    assert _fingerprint_of(other, val_sha="split-1") != baseline
    monkeypatch.setattr(vision_train, "_pilot_source_sha256", lambda: "0" * 64)
    assert _fingerprint_of(dataset_dir, val_sha="split-1") != baseline


def test_teacher_identity_lists_the_ordered_shards_and_full_pilot_config(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    with VisionDataset(dataset_dir) as dataset:
        identity = vision_train.teacher_identity(
            dataset, dataset_dir, manifest=None, val_index_sha256="split"
        )
    assert identity["schema"] == "v104-teacher/1"
    assert identity["candidate_order_version"] == "focus9-move-v1"
    assert identity["dtype"] == "float16"
    assert identity["value_shape"] == [FRAMES, 18]
    assert identity["val_index_sha256"] == "split"
    names = [row["name"] for row in identity["dataset"]["shards"]]
    assert names == sorted(names) and len(names) == len(list(dataset_dir.glob("*.npz")))
    for row in identity["dataset"]["shards"]:
        assert set(row) == {"name", "frames", "bytes", "sha256"}
        assert len(row["sha256"]) == 64
    assert identity["teacher"]["config"]["switch_penalty"] == 0.0
    assert len(identity["dataset"]["shards"][0]["sha256"]) == 64


def _teacher_cache_and_fingerprint(dataset_dir: Path, cfg) -> tuple[Path, str, dict]:
    with VisionDataset(dataset_dir) as dataset:
        identity = vision_train.teacher_identity(
            dataset, dataset_dir, manifest=None, val_index_sha256=""
        )
        fingerprint = vision_train._fingerprint(identity)
    return vision_train.teacher_cache_path(dataset_dir, fingerprint), fingerprint, identity


def test_teacher_cache_is_fingerprinted_reused_and_written_atomically(tmp_path, capsys):
    dataset_dir = _dataset(tmp_path / "a")
    cfg = _v104(dataset_dir, tmp_path / "run", stage="policy", epochs_policy=1)
    with VisionDataset(dataset_dir) as dataset:
        first = vision_train.load_or_build_teacher_values(
            cfg, dataset, manifest=None, val_index_sha256=""
        )
        second = vision_train.load_or_build_teacher_values(
            cfg, dataset, manifest=None, val_index_sha256=""
        )
    assert first["rebuilt"] is True
    assert second["rebuilt"] is False
    cache = Path(first["path_absolute"])
    assert "-teacher-values-v104-" in cache.name
    assert cache != Path(str(dataset_dir) + "-teacher-values.npz")
    sidecar = Path(str(cache) + ".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["schema"] == "v104-teacher/1"
    assert payload["fingerprint"] == first["fingerprint"]
    assert payload["cache_sha256"] == file_digest(cache) == first["cache_sha256"]
    assert payload["identity"]["dtype"] == "float16"
    assert payload["identity"]["teacher"]["class"] == "AvoidPilot"
    assert np.array_equal(first["values"], second["values"])
    assert first["values"].dtype == np.float16
    # no temp files survived the atomic write
    assert list(cache.parent.glob("*.tmp*")) == []
    assert first["audit"]["pairs_per_frame"] == 153


def test_explicit_teacher_values_without_identity_are_refused(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    legacy = tmp_path / "legacy-teacher-values.npz"
    np.savez_compressed(legacy, values=np.zeros((FRAMES, 18), dtype=np.float16))
    before = legacy.read_bytes()
    cfg = _v104(
        dataset_dir, tmp_path / "run", stage="policy", epochs_policy=1, teacher_values=legacy
    )
    with VisionDataset(dataset_dir) as dataset:
        with pytest.raises(ValueError) as error:
            vision_train.load_or_build_teacher_values(
                cfg, dataset, manifest=None, val_index_sha256=""
            )
    message = str(error.value)
    assert message.startswith("v104:TEACHER: ")
    assert "does not match the current teacher identity" in message
    assert "no-identity" in message
    assert legacy.read_bytes() == before
    assert not Path(str(legacy) + ".json").exists()


def test_teacher_cache_with_the_same_shape_but_different_values_is_rejected(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    cache, fingerprint, _identity = _teacher_cache_and_fingerprint(dataset_dir, None)
    values = np.zeros((FRAMES, 18), dtype=np.float16)
    _digest = vision_train._write_teacher_cache(cache, values, {"identity": "test"}, fingerprint)
    # tamper with the payload but keep the shape: the sidecar digest must catch it
    with np.load(cache) as data:
        tampered = np.array(data["values"])
    tampered[0, 0] = 12.5
    np.savez_compressed(cache, values=tampered)
    cfg = _v104(
        dataset_dir, tmp_path / "run", stage="policy", epochs_policy=1, teacher_values=cache
    )
    with VisionDataset(dataset_dir) as dataset:
        with pytest.raises(ValueError) as error:
            vision_train.load_or_build_teacher_values(
                cfg, dataset, manifest=None, val_index_sha256=""
            )
    assert "digest" in str(error.value)
    assert str(error.value).startswith("v104:TEACHER: ")


def test_teacher_cache_with_non_finite_values_is_refused(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    cache, fingerprint, identity = _teacher_cache_and_fingerprint(dataset_dir, None)
    values = np.zeros((FRAMES, 18), dtype=np.float16)
    values[3, 4] = np.inf
    np.savez_compressed(cache, values=values)
    payload = {
        "schema": "v104-teacher/1",
        "fingerprint": fingerprint,
        "cache_sha256": file_digest(cache),
        "identity": identity,
    }
    Path(str(cache) + ".json").write_text(json.dumps(payload), encoding="utf-8")
    cfg = _v104(
        dataset_dir, tmp_path / "run", stage="policy", epochs_policy=1, teacher_values=cache
    )
    with VisionDataset(dataset_dir) as dataset:
        with pytest.raises(ValueError) as error:
            vision_train.load_or_build_teacher_values(
                cfg, dataset, manifest=None, val_index_sha256=""
            )
    assert "nonfinite" in str(error.value)


def test_teacher_values_validation_rejects_shape_dtype_and_non_finite():
    good = np.zeros((4, 18), dtype=np.float16)
    assert vision_train._validate_teacher_values(good, 4, "test") is good
    for bad, fragment in (
        (np.zeros((3, 18), dtype=np.float16), "must be"),
        (np.zeros((4, 18), dtype=np.float32), "float16"),
        (np.full((4, 18), np.nan, dtype=np.float16), "non-finite"),
    ):
        with pytest.raises(ValueError) as error:
            vision_train._validate_teacher_values(bad, 4, "test")
        assert fragment in str(error.value)
        assert str(error.value).startswith("v104:TEACHER: ")


def test_legacy_teacher_cache_is_neither_trusted_nor_overwritten(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    legacy = Path(str(dataset_dir) + "-teacher-values.npz")
    np.savez_compressed(legacy, values=np.arange(FRAMES * 18, dtype=np.float16).reshape(FRAMES, 18))
    before = legacy.read_bytes()
    cfg = _v104(dataset_dir, tmp_path / "run", stage="policy", epochs_policy=1)
    with VisionDataset(dataset_dir) as dataset:
        cache = vision_train.load_or_build_teacher_values(
            cfg, dataset, manifest=None, val_index_sha256=""
        )
    assert legacy.read_bytes() == before
    assert Path(cache["path_absolute"]) != legacy
    assert cache["rebuilt"] is True
    # the v1.0.3 file holds a ramp; the v104 cache must hold real teacher values
    assert not np.array_equal(np.asarray(cache["values"]), np.asarray(np.arange(FRAMES * 18, dtype=np.float16).reshape(FRAMES, 18)))
    # and pointing --teacher-values at the old file is the explicit refusal
    explicit = _v104(
        dataset_dir, tmp_path / "run2", stage="policy", epochs_policy=1, teacher_values=legacy
    )
    with VisionDataset(dataset_dir) as dataset:
        with pytest.raises(ValueError) as error:
            vision_train.load_or_build_teacher_values(
                explicit, dataset, manifest=None, val_index_sha256=""
            )
    assert "no-identity" in str(error.value)


def test_teacher_audit_reports_the_float16_ties():
    values = np.zeros((5, 18), dtype=np.float16)
    values[0, :] = np.float16(10.0)
    values[1, :] = np.linspace(0.0, 17.0, 18, dtype=np.float16)
    audit = vision_train.teacher_audit(values)
    assert audit["frames"] == 5
    assert audit["pairs_per_frame"] == 153
    assert audit["fully_tied_frames"] == 4  # frame 0 plus the three all-zero rows
    assert audit["frames_with_a_tied_pair"] == 4
    assert audit["pair_tie_fraction"] == pytest.approx(4 * 153 / (5 * 153))
    assert audit["dtype"] == "float16"
    assert audit["random_choice_regret_mean_px"] > 0.0


# --------------------------------------------------------------------------- #
# selection rules
# --------------------------------------------------------------------------- #
def test_state_selection_prefers_the_first_joint_pass():
    history = [
        {"epoch": 1, "val_player_pos_px": 9.0},
        {"epoch": 2, "val_player_pos_px": 8.0},
    ]
    row, rule = vision_train.state_selection_row(history, pass_epoch=1)
    assert rule == "joint-pass"
    assert row["epoch"] == 1
    row, rule = vision_train.state_selection_row(history, pass_epoch=None)
    assert rule == "position"
    assert row["epoch"] == 2
    assert vision_train.state_selection_row([], pass_epoch=None) == (None, "none")


def test_policy_selection_key_orders_bucket_then_global_then_epoch():
    def row(epoch: int, bucket: float, glob: float, frames: int = 4) -> dict:
        return {
            "epoch": epoch,
            "val_regret_px": glob,
            "by_danger": [
                _bucket_row("0-15", frames, bucket),
                _bucket_row("15-30", 2, 3.0),
            ],
        }

    better_bucket = vision_train.policy_selection_key(row(2, 1.0, 9.0))
    worse_bucket = vision_train.policy_selection_key(row(1, 1.5, 1.0))
    assert better_bucket < worse_bucket
    better_global = vision_train.policy_selection_key(row(2, 1.0, 9.0))
    worse_global = vision_train.policy_selection_key(row(1, 1.0, 10.0))
    assert better_global < worse_global
    earlier = vision_train.policy_selection_key(row(1, 1.0, 9.0))
    later = vision_train.policy_selection_key(row(3, 1.0, 9.0))
    assert earlier < later
    # an empty 0-15 bucket has placeholder zeros and must never win
    assert vision_train.policy_selection_key(row(1, 0.0, 0.0, frames=0)) is None


def test_v104_policy_selection_uses_the_danger_bucket(tmp_path, monkeypatch):
    dataset_dir = _dataset(tmp_path / "a")
    script = _policy_metrics_script([(2.0, 5.0, 3), (1.0, 9.0, 3), (1.0, 9.0, 3)])

    def fake_policy_metrics(pred, teacher, bullets):
        return dict(script[min(len(script) - 1, fake_policy_metrics.calls)])

    fake_policy_metrics.calls = 0

    def wrapper(pred, teacher, bullets):
        result = fake_policy_metrics(pred, teacher, bullets)
        fake_policy_metrics.calls += 1
        return result

    monkeypatch.setattr(vision_metrics, "policy_metrics", wrapper)
    out = tmp_path / "run"
    summary = vision_train.train(
        _v104(
            dataset_dir,
            out,
            stage="both",
            epochs_state=1,
            epochs_policy=3,
            max_train_batches=1,
            max_val_batches=0,
        )
    )
    history = summary["stages"]["policy"]["history"]
    assert [row["epoch"] for row in history] == [1, 2, 3]
    best = summary["stages"]["policy"]["best"]
    assert best["epoch"] == 2  # lower 0-15 bucket regret beats epoch 3's tie
    assert best["selection_rule"] == "bucket-0-15-regret"
    assert best["checkpoint"] == "policy.pt"
    assert best["checkpoint_sha256"] == file_digest(out / "policy.pt")
    assert best["selection_key"] == [1.0, 9.0, 2]
    assert _extra(out / "policy.pt")["epoch"] == 2
    assert summary["stages"]["policy"]["selection_rule"] == "bucket-0-15-regret"


def test_v104_empty_bucket_cannot_be_selected(tmp_path, monkeypatch):
    dataset_dir = _dataset(tmp_path / "a")
    script = _policy_metrics_script([(0.0, 4.0, 0), (0.0, 3.0, 0)])

    def fake_policy_metrics(pred, teacher, bullets):
        return dict(script[min(len(script) - 1, fake_policy_metrics.calls)])

    fake_policy_metrics.calls = 0

    def wrapper(pred, teacher, bullets):
        result = fake_policy_metrics(pred, teacher, bullets)
        fake_policy_metrics.calls += 1
        return result

    monkeypatch.setattr(vision_metrics, "policy_metrics", wrapper)
    out = tmp_path / "run"
    summary = vision_train.train(
        _v104(
            dataset_dir,
            out,
            stage="both",
            epochs_state=1,
            epochs_policy=2,
            max_train_batches=1,
        )
    )
    assert summary["stages"]["policy"]["best"] == {}
    assert summary["stages"]["policy"]["selection_rule"] == "none"
    assert not (out / "policy.pt").exists()
    assert summary["stages"]["policy"]["final"]["epoch"] == 2


def test_legacy_policy_ignores_buckets_and_still_selects(tmp_path, monkeypatch):
    """The old contract selects by global regret; the default must not change."""
    dataset_dir = _dataset(tmp_path / "a")
    script = _policy_metrics_script([(0.0, 5.0, 0), (0.0, 3.0, 0)])

    def fake_policy_metrics(pred, teacher, bullets):
        return dict(script[min(len(script) - 1, fake_policy_metrics.calls)])

    fake_policy_metrics.calls = 0

    def wrapper(pred, teacher, bullets):
        result = fake_policy_metrics(pred, teacher, bullets)
        fake_policy_metrics.calls += 1
        return result

    monkeypatch.setattr(vision_metrics, "policy_metrics", wrapper)
    out = tmp_path / "run"
    summary = vision_train.train(
        _cfg(dataset_dir, out, stage="policy", epochs_policy=2, max_train_batches=1)
    )
    assert summary["stages"]["policy"]["best"]["epoch"] == 2
    assert (out / "policy.pt").is_file()
    assert "selection_rule" not in summary["stages"]["policy"]["best"]


# --------------------------------------------------------------------------- #
# policy schedule: freeze -> joint, shared init, strict source checks
# --------------------------------------------------------------------------- #
def test_build_policy_optimizer_freezes_encoder_and_perception(tmp_path):
    cfg = _v104(tmp_path / "ds", tmp_path / "run", stage="policy")
    model = VisionScorerV2(
        VisionNetConfigV2(depth=3, pf_width=8, hud_width=4, trunk_hidden=32, playfield_hw=(56, 48))
    )
    optimizer, record = vision_train.build_policy_optimizer(model, cfg, frozen=True)
    head = ("candidate_embed.", "score_head.")
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith(head), name
    assert record["phase"] == "frozen"
    assert record["groups"]["encoder"]["trainable"] is False
    assert record["groups"]["encoder"]["lr"] == 0.0
    assert record["groups"]["perception"]["trainable"] is False
    assert record["groups"]["candidate_head"]["trainable"] is True
    assert record["groups"]["candidate_head"]["lr"] == 3e-4
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["name"] == "candidate_head"

    joint, joint_record = vision_train.build_policy_optimizer(model, cfg, frozen=False)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert len(joint.param_groups) == 3
    assert {group["name"]: group["lr"] for group in joint.param_groups} == {
        "encoder": 1e-4,
        "perception": 1e-4,
        "candidate_head": 3e-4,
    }
    assert joint_record["trainable_parameters"] == sum(p.numel() for p in model.parameters())


def test_v104_three_policy_epochs_cross_the_freeze_boundary(tmp_path, monkeypatch):
    dataset_dir = _dataset(tmp_path / "a")
    created: list[int] = []
    real_adamw = torch.optim.AdamW

    def counting_adamw(*args, **kwargs):
        optimizer = real_adamw(*args, **kwargs)
        created.append(len(optimizer.param_groups))
        return optimizer

    monkeypatch.setattr(torch.optim, "AdamW", counting_adamw)
    out = tmp_path / "run"
    summary = vision_train.train(
        _v104(
            dataset_dir,
            out,
            stage="both",
            epochs_state=1,
            epochs_policy=3,
            policy_freeze_epochs=2,
            max_train_batches=1,
            max_val_batches=0,
        )
    )
    optim = summary["optim"]
    assert optim["initial"]["phase"] == "frozen"
    assert optim["initial"]["groups"]["encoder"]["trainable"] is False
    assert optim["initial"]["groups"]["perception"]["trainable"] is False
    assert optim["joint"]["phase"] == "joint"
    assert optim["joint"]["groups"]["encoder"]["lr"] == 1e-4
    assert optim["joint"]["groups"]["encoder"]["trainable"] is True
    assert optim["switch_epoch"] == 3
    assert optim["joint_phase_reached"] is True
    assert [row["epoch"] for row in optim["rebuilds"]] == [3]
    # one state optimizer + one frozen policy optimizer + one joint rebuild
    assert created == [1, 1, 3]  # parameter-group counts of each construction
    assert [row["epoch"] for row in summary["stages"]["policy"]["history"]] == [1, 2, 3]


def test_v104_single_policy_epoch_does_not_claim_joint_training(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    out = tmp_path / "run"
    summary = vision_train.train(
        _v104(
            dataset_dir,
            out,
            stage="both",
            epochs_state=1,
            epochs_policy=1,
            policy_freeze_epochs=2,
            max_train_batches=1,
            max_val_batches=1,
        )
    )
    assert summary["optim"]["joint_phase_reached"] is False
    assert summary["optim"]["switch_epoch"] is None
    assert "joint" not in summary["optim"]
    assert summary["optim"]["rebuilds"] == []


def test_v104_policy_starts_from_the_selected_state_checkpoint(tmp_path, monkeypatch):
    dataset_dir = _dataset(tmp_path / "a")
    state_script = [_fake_state_metrics(9.0, 0.85), _fake_state_metrics(9.5, 0.5)]

    def fake_all_state_metrics(pred_self, pred_danger_log, truth_self, truth_danger, *, heatmap_logits=None):
        return dict(state_script[min(len(state_script) - 1, fake_all_state_metrics.calls)])

    fake_all_state_metrics.calls = 0

    def state_wrapper(*args, **kwargs):
        result = fake_all_state_metrics(*args, **kwargs)
        fake_all_state_metrics.calls += 1
        return result

    policy_script = _policy_metrics_script([(1.0, 5.0, 3)])

    def fake_policy_metrics(pred, teacher, bullets):
        return dict(policy_script[0])

    monkeypatch.setattr(vision_metrics, "all_state_metrics", state_wrapper)
    monkeypatch.setattr(vision_metrics, "policy_metrics", fake_policy_metrics)
    out = tmp_path / "run"
    summary = vision_train.train(
        _v104(
            dataset_dir,
            out,
            stage="both",
            epochs_state=2,
            epochs_policy=1,
            max_train_batches=1,
            max_val_batches=0,
        )
    )
    state_best = summary["stages"]["state"]["best"]
    assert state_best["epoch"] == 1
    assert state_best["selection_rule"] == "joint-pass"
    assert state_best["checkpoint"] == "state-pass.pt"
    assert state_best["checkpoint_sha256"] == file_digest(out / "state-pass.pt")
    # the policy stage loaded those bytes, not the last state epoch (epoch 2)
    source = summary["source_state"]
    assert source["path"].endswith("state-pass.pt")
    assert source["epoch"] == 1
    assert source["sha256"] == file_digest(out / "state-pass.pt")
    assert source["strict_load"] is True
    extra = _extra(out / "policy.pt")
    assert extra["source_state_sha256"] == source["sha256"]
    assert extra["contract_version"] == "v104-A/1"
    assert extra["history_contract"] == "legacy"
    assert summary["stages"]["state"]["best_position"]["epoch"] == 1
    assert summary["stages"]["state"]["pass_epoch"] == 1


def test_v104_policy_rejects_a_cross_depth_source(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    first = vision_train.train(
        _v104(
            dataset_dir,
            tmp_path / "state-run",
            stage="state",
            epochs_state=1,
            depth=1,
            max_train_batches=1,
        )
    )
    assert (tmp_path / "state-run" / "state.pt").is_file()
    assert first["depth"] == 1
    with pytest.raises(ValueError) as error:
        vision_train.train(
            _v104(
                dataset_dir,
                tmp_path / "policy-run",
                stage="policy",
                depth=3,
                epochs_policy=1,
                init_from=tmp_path / "state-run" / "state.pt",
            )
        )
    message = str(error.value)
    assert message.startswith("v104:CHECKPOINT: ")
    assert "depth" in message


def test_v104_policy_rejects_a_source_with_a_different_history_contract(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    vision_train.train(
        _v104(
            dataset_dir,
            tmp_path / "state-run",
            stage="state",
            epochs_state=1,
            max_train_batches=1,
        )
    )
    with pytest.raises(ValueError) as error:
        vision_train.train(
            _v104(
                dataset_dir,
                tmp_path / "policy-run",
                stage="policy",
                history_contract="causal-segment-v1",
                epochs_policy=1,
                init_from=tmp_path / "state-run" / "state.pt",
            )
        )
    message = str(error.value)
    assert message.startswith("v104:CHECKPOINT: ")
    assert "history_contract" in message


def test_v104_policy_only_stage_requires_init_from(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    with pytest.raises(ValueError) as error:
        vision_train.train(
            _v104(dataset_dir, tmp_path / "run", stage="policy", epochs_policy=1)
        )
    assert "requires --init-from" in str(error.value)


# --------------------------------------------------------------------------- #
# run directory write protection
# --------------------------------------------------------------------------- #
def test_v104_run_refuses_an_existing_summary_or_checkpoint(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    out = tmp_path / "run"
    vision_train.train(_v104(dataset_dir, out, stage="state", epochs_state=1))
    with pytest.raises(ValueError) as error:
        vision_train.train(_v104(dataset_dir, out, stage="state", epochs_state=1))
    assert error.value.args[0].startswith("v104:CHECKPOINT: ")
    assert "refusing to overwrite" in str(error.value)

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "policy.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError) as error:
        vision_train.train(_v104(dataset_dir, bare, stage="state", epochs_state=1))
    assert "policy.pt" in str(error.value)


def test_legacy_contract_still_allows_rerunning_a_directory(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    out = tmp_path / "run"
    first = vision_train.train(_cfg(dataset_dir, out, stage="state", epochs_state=1))
    second = vision_train.train(_cfg(dataset_dir, out, stage="state", epochs_state=1))
    assert first["schema"] == second["schema"] == "vision-summary/2"
    assert "contract_version" not in second


@pytest.mark.parametrize(
    "relative",
    [
        "third_party/th08/build/rl/runs/v104-a01-72ep",
        "third_party/th08/build/rl/runs/v104-a01-72ep/nested",
        "third_party/th08/build/rl/runs/v104-clean-24e12e",
        "third_party/th08/build/rl/runs/v104-clean-24e12e/s7",
    ],
)
def test_v104_frozen_baselines_are_refused(tmp_path, relative):
    dataset_dir = _dataset(tmp_path / "a")
    with pytest.raises(SystemExit) as error:
        vision_train.train(_v104(dataset_dir, Path(relative), stage="state", epochs_state=1))
    assert "refusing to overwrite the frozen v1.0.4 baseline" in str(error.value)


# --------------------------------------------------------------------------- #
# summary contract
# --------------------------------------------------------------------------- #
def test_v104_summary_carries_every_contract_block(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    manifest = _write_manifest(dataset_dir)
    out = tmp_path / "run"
    summary = vision_train.train(
        _v104(
            dataset_dir,
            out,
            stage="both",
            epochs_state=1,
            epochs_policy=1,
            depth=3,
            history_contract="causal-segment-v1",
            policy_objective="rank",
            rank_weight=1.0,
            regression_weight=0.0,
            allow_missing_manifest=False,
            max_train_batches=1,
            max_val_batches=0,
        )
    )
    assert summary["schema"] == "vision-summary/2"
    assert summary["contract_version"] == "v104-A/1"
    assert summary["acceptance_eligible"] is False  # --max-train-batches 1
    for key in (
        "recipe",
        "data",
        "teacher",
        "input",
        "loss",
        "optim",
        "coverage",
        "source_state",
        "hardware",
        "partial",
    ):
        assert key in summary, key
    assert summary["data"]["val_index_sha256"] == manifest.splits["val_index_sha256"]
    assert summary["data"]["manifest_sha256"] == file_digest(dataset_dir / "manifest.json")
    assert summary["data"]["history_contract"] == "causal-segment-v1"
    assert summary["data"]["manifest_verified"] is True
    assert summary["recipe"]["screening"] is True
    assert summary["input"]["candidate_order_version"] == "focus9-move-v1"
    assert summary["input"]["depth"] == 3
    assert summary["loss"]["policy"]["objective"] == "rank"
    assert (summary["loss"]["policy"]["rank_weight"], summary["loss"]["policy"]["regression_weight"]) == (1.0, 0.0)
    assert summary["loss"]["self_mask_dims"] == 19
    assert summary["loss"]["aux_policy"]["heatmap"] == 0.1
    assert summary["loss"]["aux_policy"]["danger"] == 0.2
    assert summary["teacher"]["fingerprint"] == _fingerprint_of(
        dataset_dir, val_sha=manifest.splits["val_index_sha256"]
    )
    assert summary["teacher"]["cache_sha256"] == file_digest(Path(summary["teacher"]["cache"]))
    assert summary["teacher"]["shard_value_used"] is False
    assert summary["teacher"]["audit"]["pairs_per_frame"] == 153
    assert summary["coverage"]["val"]["actual_frames"] == summary["val_frames"] == 6
    assert summary["coverage"]["val"]["complete"] is True
    # the training limit is recorded honestly: one batch of two frames, not "all"
    assert summary["coverage"]["train"]["expected_frames"] == 24
    assert summary["coverage"]["train"]["actual_frames"] == 2
    assert summary["coverage"]["train"]["complete"] is False
    assert summary["coverage"]["val"]["visited_index_sha256"] == manifest.splits["val_index_sha256"]
    state_best = summary["stages"]["state"]["best"]
    assert {"selection_rule", "checkpoint", "checkpoint_sha256"} <= set(state_best)
    assert state_best["checkpoint_sha256"] == file_digest(out / state_best["checkpoint"])
    policy_best = summary["stages"]["policy"]["best"]
    if policy_best:
        assert {"selection_rule", "checkpoint", "checkpoint_sha256"} <= set(policy_best)
        assert policy_best["checkpoint_sha256"] == file_digest(out / policy_best["checkpoint"])
    config = summary["config"]
    assert config["selection_contract"] == "v104"
    assert config["history_contract"] == "causal-segment-v1"
    assert config["policy_objective"] == "rank"
    assert config["effective_loss"]["weights_forced_by_contract"] is False
    assert summary["parameters"] > 0


def test_v104_limited_validation_is_recorded_and_not_claimable(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    out = tmp_path / "run"
    summary = vision_train.train(
        _v104(
            dataset_dir,
            out,
            stage="state",
            epochs_state=1,
            max_train_batches=0,
            max_val_batches=1,
        )
    )
    coverage = summary["coverage"]
    assert coverage["val"]["actual_frames"] == 2  # one batch of two
    assert coverage["val"]["complete"] is False
    assert coverage["val"]["visited_index_sha256"] != summary["val_index_sha256"]
    assert summary["acceptance_eligible"] is False


def test_v104_full_validation_covers_every_held_out_frame(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    manifest = _write_manifest(dataset_dir)
    out = tmp_path / "run"
    summary = vision_train.train(
        _v104(
            dataset_dir,
            out,
            stage="state",
            epochs_state=1,
            batch_size=4,
            allow_missing_manifest=False,
            max_train_batches=0,
            max_val_batches=0,
        )
    )
    coverage = summary["coverage"]["val"]
    assert coverage["expected_frames"] == coverage["actual_frames"] == 6
    assert coverage["complete"] is True
    assert coverage["visited_index_sha256"] == manifest.splits["val_index_sha256"]
    assert summary["acceptance_eligible"] is True


def test_v104_drop_last_does_not_make_a_full_run_ineligible(tmp_path):
    """Training may keep only full batches; eligibility reads the validation side."""
    dataset_dir = _dataset(tmp_path / "a")
    _write_manifest(dataset_dir)  # 24 train / 6 val
    out = tmp_path / "run"
    summary = vision_train.train(
        _v104(
            dataset_dir,
            out,
            stage="state",
            epochs_state=1,
            batch_size=5,
            allow_missing_manifest=False,
            max_train_batches=0,
            max_val_batches=0,
        )
    )
    assert summary["coverage"]["train"]["expected_frames"] == 24
    assert summary["coverage"]["train"]["actual_frames"] == 20  # four full batches of five
    assert summary["coverage"]["train"]["complete"] is False
    assert summary["coverage"]["train"]["drop_last"] is True
    assert summary["coverage"]["val"]["actual_frames"] == 6  # five plus the one-frame tail
    assert summary["coverage"]["val"]["complete"] is True
    assert summary["acceptance_eligible"] is True


def test_legacy_summary_has_no_v104_blocks(tmp_path):
    dataset_dir = _dataset(tmp_path / "a")
    out = tmp_path / "run"
    summary = vision_train.train(_cfg(dataset_dir, out, stage="both", epochs_state=1, epochs_policy=1))
    for key in (
        "contract_version",
        "recipe",
        "data",
        "teacher",
        "input",
        "loss",
        "optim",
        "coverage",
        "source_state",
        "acceptance_eligible",
        "partial",
    ):
        assert key not in summary, key
    assert "selection_rule" not in summary["stages"]["state"]["best"]
    assert summary["stages"]["policy"]["best"]["epoch"] >= 1


# --------------------------------------------------------------------------- #
# CLI error contract
# --------------------------------------------------------------------------- #
def test_cli_maps_v104_errors_to_exit_2(tmp_path, capsys):
    dataset_dir = _dataset(tmp_path / "a")
    code = vision_train.main(
        [
            "--dataset",
            str(dataset_dir),
            "--out",
            str(tmp_path / "run"),
            "--selection-contract",
            "v104",
            "--depth",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.startswith("v104:CONFIG: ")


def test_cli_maps_manifest_errors_to_exit_2(tmp_path, capsys):
    dataset_dir = _dataset(tmp_path / "a")  # no manifest.json on purpose
    code = vision_train.main(["--dataset", str(dataset_dir), "--out", str(tmp_path / "run")])
    captured = capsys.readouterr()
    assert code == 2
    assert "no manifest" in captured.err


def test_cli_keeps_unknown_errors_as_tracebacks(tmp_path, monkeypatch):
    def explode(cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(vision_train, "train", explode)
    with pytest.raises(RuntimeError):
        vision_train.main(["--dataset", "x", "--out", "y"])
