"""Unit tests for the v2 evaluation metrics and the R3 training-set sub-sampling.

These pin the properties whose silent breakage would corrupt a whole experiment:

* ``bullet_count_mae`` reads index 14 (``active_count/700``).  v1 read index 13,
  which is ``stage_clear_flag`` and is 1.0 on every recorded frame, so the old
  "15.1 bullets" number was measuring a constant;
* player position error keeps the v1 ``(|dx|*384 + |dy|*448)/2`` form, not the
  Euclidean norm, so 70.97 px stays comparable;
* per-frame danger correlation drops frames whose 128 cells are constant instead
  of filling them with 0.0 (filling would drag the median toward "no correlation");
* the R3 sub-samples are nested (0.25 within 0.5 within 1.0) under one seed, which
  is what makes the data-scaling curve measure data volume instead of sampling noise.

Everything here is pure numpy: no torch, no CUDA, no game window.
"""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl import obs, vision
from touhou_rl import vision_metrics as vm
from touhou_rl.vision_data import VisionDataset, VisionShardWriter
from touhou_rl.vision_split import stratified_prefix


# -- player position ------------------------------------------------------
def test_position_mae_matches_v1_formula():
    truth = np.zeros((4, 24), dtype=np.float32)
    pred = truth.copy()
    pred[:, 0] = 0.01
    pred[:, 1] = -0.02  # the sign must not matter
    mean_px, p95_px = vm.player_position_mae_px(pred, truth)
    expected = (0.01 * 384.0 + 0.02 * 448.0) / 2.0
    assert mean_px == pytest.approx(expected, rel=1e-6)
    assert p95_px == pytest.approx(expected, rel=1e-6)
    # Euclidean distance would report 9.749 px here and inflate every v1 baseline.
    assert mean_px != pytest.approx(9.7494, abs=1e-2)


def test_position_mae_p95_and_empty():
    assert vm.player_position_mae_px(np.zeros((0, 24)), np.zeros((0, 24))) == (0.0, 0.0)
    truth = np.zeros((16, 24), dtype=np.float32)
    pred = truth.copy()
    pred[:, 0] = 0.05
    mean_px, p95_px = vm.player_position_mae_px(pred, truth)
    assert mean_px == p95_px  # constant error distribution
    assert mean_px == pytest.approx(0.05 * 384.0 / 2.0, rel=1e-9)


# -- bullet count ---------------------------------------------------------
def test_bullet_count_uses_index_14():
    truth = np.zeros((8, 24), dtype=np.float32)
    truth[:, 13] = 1.0
    truth[:, 14] = 0.5
    pred = np.zeros((8, 24), dtype=np.float32)
    # Reading index 13 would report 700.0 here; index 14 reports 0.5 * 700.
    assert vm.bullet_count_mae(pred, truth) == pytest.approx(350.0, rel=1e-9)


def test_bullet_count_index13_is_ignored():
    truth = np.zeros((8, 24), dtype=np.float32)
    truth[:, 13] = 1.0
    truth[:, 14] = 0.5
    changed = truth.copy()
    changed[:, 13] = 0.0
    pred = np.full((8, 24), 0.25, dtype=np.float32)
    baseline = vm.bullet_count_mae(pred, truth)
    assert baseline == pytest.approx(175.0, rel=1e-9)
    assert vm.bullet_count_mae(pred, changed) == baseline


# -- danger map -----------------------------------------------------------
def test_danger_corr_global_matches_numpy():
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 6, size=(12, 16, 8)).astype(np.uint8)
    pred = rng.random((12, 128)) * 0.5
    target = np.log1p(truth.astype(np.float64) / 8.0).reshape(12, -1)
    expected = float(np.corrcoef(pred.ravel(), target.ravel())[0, 1])
    got = vm.danger_metrics(pred, truth)["danger_corr"]
    assert got == pytest.approx(round(expected, 4), rel=1e-6)
    assert abs(got - expected) <= 5e-5  # 4-decimal reporting contract


def test_danger_perframe_excludes_constant_frames():
    rng = np.random.default_rng(5)
    truth = rng.integers(0, 7, size=(8, 16, 8)).astype(np.uint8)
    truth[2] = 0
    truth[6] = 0
    pred = rng.random((8, 16, 8)) * 0.4
    metrics = vm.danger_metrics(pred.reshape(8, 128), truth)
    assert metrics["danger_corr_perframe_valid"] == 6
    assert not np.isnan(metrics["danger_corr_perframe_median"])
    target = np.log1p(truth.astype(np.float64) / 8.0).reshape(8, 128)
    manual = [
        float(np.corrcoef(pred.reshape(8, 128)[i], target[i])[0, 1]) for i in (0, 1, 3, 4, 5, 7)
    ]
    assert metrics["danger_corr_perframe_mean"] == pytest.approx(float(np.mean(manual)), abs=1e-3)
    assert metrics["danger_corr_perframe_median"] == pytest.approx(float(np.median(manual)), abs=1e-3)


def test_danger_perframe_all_invalid_returns_zero():
    truth = np.zeros((5, 16, 8), dtype=np.uint8)
    pred = np.full((5, 128), 0.3)
    metrics = vm.danger_metrics(pred, truth)
    assert metrics["danger_corr_perframe_valid"] == 0
    for key in (
        "danger_corr",
        "danger_corr_perframe_mean",
        "danger_corr_perframe_median",
        "danger_iou",
        "danger_cell_acc",
    ):
        assert metrics[key] == 0.0


def test_danger_iou_perfect_and_empty():
    rng = np.random.default_rng(11)
    truth = rng.integers(0, 7, size=(6, 16, 8)).astype(np.uint8)
    pred = np.log1p(truth.astype(np.float64) / 8.0)
    assert vm.danger_metrics(pred, truth)["danger_iou"] == pytest.approx(1.0, abs=1e-6)
    empty = vm.danger_metrics(np.zeros((6, 128)), np.zeros((6, 16, 8), dtype=np.uint8))
    assert empty["danger_iou"] == 0.0


def test_danger_metrics_accept_both_shapes():
    rng = np.random.default_rng(13)
    truth = rng.integers(0, 9, size=(10, 16, 8)).astype(np.uint8)
    pred = rng.random((10, 16, 8)) * 0.6
    assert vm.danger_metrics(pred, truth) == vm.danger_metrics(pred.reshape(10, 128), truth)


# -- policy ---------------------------------------------------------------
def test_policy_metrics_bucket_labels_and_empty_buckets():
    frames = 6
    pred = np.zeros((frames, 18))
    pred[:, 0] = 1.0
    teacher = np.zeros((frames, 18))
    teacher[:, 0] = 5.0
    bullets = np.zeros((frames, 64, 6), dtype=np.float32)  # every slot invalid
    metrics = vm.policy_metrics(pred, teacher, bullets)
    assert [bucket["bucket"] for bucket in metrics["by_danger"]] == [
        "0-15",
        "15-30",
        "30-60",
        "60-120",
        "120+",
    ]
    assert [bucket["frames"] for bucket in metrics["by_danger"]] == [0, 0, 0, 0, frames]
    assert metrics["by_danger"][4]["top1"] == 1.0
    assert metrics["by_danger"][4]["regret_px"] == 0.0
    for bucket in metrics["by_danger"][:4]:
        assert bucket["top1"] == 0.0
        assert bucket["regret_px"] == 0.0


def test_policy_metrics_regret_and_top1():
    pred = np.zeros((2, 18))
    teacher = np.zeros((2, 18))
    # frame 0: model prefers 7, teacher prefers 3 -> wrong, regret 100 - 40 = 60 px
    pred[0, 7] = 1.0
    teacher[0, 3] = 100.0
    teacher[0, 7] = 40.0
    # frame 1: model prefers 2, teacher prefers 2 -> right, regret 0
    pred[1, 2] = 1.0
    teacher[1, 2] = 200.0
    teacher[1, 5] = 150.0
    bullets = np.zeros((2, 64, 6), dtype=np.float32)
    bullets[0, 0, 0] = 5.0 / 192.0  # 5 px -> bucket 0-15
    bullets[0, 0, 5] = 1.0
    bullets[1, 0, 0] = 40.0 / 192.0  # 40 px -> bucket 30-60
    bullets[1, 0, 5] = 1.0
    metrics = vm.policy_metrics(pred, teacher, bullets)
    assert metrics["top1"] == pytest.approx(0.5, rel=1e-9)
    assert metrics["regret_px"] == pytest.approx(30.0, rel=1e-6)
    assert metrics["by_danger"][0]["frames"] == 1
    assert metrics["by_danger"][0]["top1"] == 0.0
    assert metrics["by_danger"][0]["regret_px"] == pytest.approx(60.0, rel=1e-6)
    assert metrics["by_danger"][2]["frames"] == 1
    assert metrics["by_danger"][2]["top1"] == 1.0
    assert metrics["by_danger"][2]["regret_px"] == 0.0
    assert metrics["by_danger"][1]["frames"] == 0


# -- v1.0.4 policy diagnostics (top3 / tie / near-optimal / p90) ----------
def _no_bullets(frames: int) -> np.ndarray:
    return np.zeros((frames, 64, 6), dtype=np.float32)


def test_policy_metrics_top3_strict_versus_tie_expanded():
    pred = np.zeros((1, 18))
    pred[0, 9] = 1.0  # the student picks candidate 9
    teacher = np.zeros((1, 18))
    teacher[0, [0, 2, 5, 9]] = 10.0  # four-way tie at the top
    metrics = vm.policy_metrics(pred, teacher, _no_bullets(1))
    # stable (-value, index) order puts the top three at 0, 2, 5: 9 is outside
    assert metrics["top3"] == 0.0
    # every candidate whose raw value is >= the third-best value (10.0) is tied
    assert metrics["top3_tie"] == 1.0
    assert metrics["top1"] == 0.0
    assert metrics["regret_px"] == 0.0
    assert metrics["near_optimal_05"] == 1.0
    bucket = metrics["by_danger"][4]  # no valid bullet -> 120+
    assert bucket["frames"] == 1
    assert bucket["top3"] == 0.0
    assert bucket["top3_tie"] == 1.0
    assert bucket["near_optimal_05"] == 1.0


def test_policy_metrics_top3_hit_and_stable_order():
    pred = np.zeros((2, 18))
    teacher = np.zeros((2, 18))
    # frame 0: teacher order 4 > 2 > 0; the student picks 2 -> inside the top three
    teacher[0, 4], teacher[0, 2], teacher[0, 0] = 5.0, 4.0, 3.0
    pred[0, 2] = 1.0
    # frame 1: the student picks the fourth best -> outside
    teacher[1, 4], teacher[1, 2], teacher[1, 0], teacher[1, 7] = 5.0, 4.0, 3.0, 2.0
    pred[1, 7] = 1.0
    metrics = vm.policy_metrics(pred, teacher, _no_bullets(2))
    assert metrics["top3"] == pytest.approx(0.5, abs=1e-9)
    assert metrics["top3_tie"] == pytest.approx(0.5, abs=1e-9)  # no tie at 3rd place
    assert metrics["top1"] == 0.0
    assert metrics["regret_px"] == pytest.approx(2.0, abs=1e-6)  # (1 + 3) / 2


def test_policy_metrics_all_tied_teacher_is_a_full_tie_set():
    pred = np.zeros((1, 18))
    pred[0, 13] = 1.0
    teacher = np.full((1, 18), 2.0)
    metrics = vm.policy_metrics(pred, teacher, _no_bullets(1))
    assert metrics["top3"] == 0.0  # the strict first three are 0, 1, 2
    assert metrics["top3_tie"] == 1.0  # every candidate is tied with the third best


def test_policy_metrics_near_optimal_boundary():
    pred = np.zeros((3, 18))
    teacher = np.zeros((3, 18))
    teacher[0, 0], teacher[0, 1] = 10.0, 9.5  # regret 0.5 -> near optimal
    pred[0, 1] = 1.0
    teacher[1, 0], teacher[1, 1] = 10.0, 9.4  # regret 0.6 -> not
    pred[1, 1] = 1.0
    teacher[2, 0], teacher[2, 1] = 10.0, 10.0  # regret 0.0 -> near optimal
    pred[2, 1] = 1.0
    metrics = vm.policy_metrics(pred, teacher, _no_bullets(3))
    assert metrics["near_optimal_05"] == pytest.approx(2.0 / 3.0, abs=5e-5)
    regrets = np.array([0.5, 0.6, 0.0])
    assert metrics["regret_px"] == pytest.approx(round(float(regrets.mean()), 3), abs=1e-6)
    assert metrics["regret_p90_px"] == pytest.approx(
        round(float(np.percentile(regrets, 90)), 3), abs=1e-6
    )


def test_policy_metrics_regret_uses_raw_unclipped_teacher_px():
    pred = np.zeros((2, 18))
    teacher = np.zeros((2, 18))
    teacher[0, 0], teacher[0, 5] = 100.0, -20.0  # clipped this would be 48 px
    pred[0, 5] = 1.0
    teacher[1, 0], teacher[1, 5] = 3.0, 2.0
    pred[1, 5] = 1.0
    metrics = vm.policy_metrics(pred, teacher, _no_bullets(2))
    assert metrics["regret_px"] == pytest.approx(60.5, abs=1e-6)  # (120 + 1) / 2
    assert metrics["near_optimal_05"] == 0.0


def test_policy_metrics_per_bucket_new_keys():
    pred = np.zeros((2, 18))
    teacher = np.zeros((2, 18))
    teacher[0, 0], teacher[0, 2] = 10.0, 9.0
    pred[0, 2] = 1.0  # top3 hit, regret 1.0
    teacher[1, 0], teacher[1, 2] = 10.0, 1.0
    pred[1, 2] = 1.0  # top3 hit, regret 9.0
    bullets = _no_bullets(2)
    bullets[0, 0, 0], bullets[0, 0, 5] = 5.0 / 192.0, 1.0  # 5 px -> 0-15
    bullets[1, 0, 0], bullets[1, 0, 5] = 45.0 / 192.0, 1.0  # 45 px -> 30-60
    metrics = vm.policy_metrics(pred, teacher, bullets)
    keys = {
        "bucket",
        "frames",
        "top1",
        "top3",
        "top3_tie",
        "near_optimal_05",
        "regret_px",
        "regret_p90_px",
    }
    for bucket in metrics["by_danger"]:
        assert set(bucket) == keys
    near, mid = metrics["by_danger"][0], metrics["by_danger"][2]
    assert near["frames"] == mid["frames"] == 1
    assert near["top3"] == mid["top3"] == 1.0
    assert near["regret_px"] == pytest.approx(1.0, abs=1e-6)
    assert mid["regret_px"] == pytest.approx(9.0, abs=1e-6)
    assert near["regret_p90_px"] == pytest.approx(1.0, abs=1e-6)
    assert set(metrics) == {
        "top1",
        "top3",
        "top3_tie",
        "near_optimal_05",
        "regret_px",
        "regret_p90_px",
        "by_danger",
    }


def test_policy_metrics_distance_uses_both_axis_scales():
    bullets = _no_bullets(1)
    bullets[0, 0, 0] = 12.0 / 192.0
    bullets[0, 0, 1] = 16.0 / 224.0
    bullets[0, 0, 5] = 1.0  # sqrt(12^2 + 16^2) = 20 px -> 15-30
    metrics = vm.policy_metrics(np.zeros((1, 18)), np.zeros((1, 18)), bullets)
    assert metrics["by_danger"][1]["frames"] == 1
    assert metrics["by_danger"][0]["frames"] == 0


def test_policy_metrics_distance_boundary_15px_is_not_in_first_bucket():
    pred = np.zeros((2, 18))
    teacher = np.zeros((2, 18))
    bullets = _no_bullets(2)
    bullets[0, 0, 0], bullets[0, 0, 5] = 15.0 / 192.0, 1.0  # exactly 15 px
    bullets[1, 0, 0], bullets[1, 0, 5] = 14.9 / 192.0, 1.0  # 14.9 px
    metrics = vm.policy_metrics(pred, teacher, bullets)
    assert metrics["by_danger"][0]["frames"] == 1  # only the 14.9 px frame
    assert metrics["by_danger"][1]["frames"] == 1  # the 15.0 px frame
    assert [bucket["frames"] for bucket in metrics["by_danger"]] == [1, 1, 0, 0, 0]


def test_policy_metrics_empty_frames_keep_new_keys():
    result = vm.policy_metrics(
        np.zeros((0, 18)), np.zeros((0, 18)), np.zeros((0, 64, 6), dtype=np.float32)
    )
    for key in ("top1", "top3", "top3_tie", "near_optimal_05", "regret_px", "regret_p90_px"):
        assert result[key] == 0.0
    assert len(result["by_danger"]) == 5
    for bucket in result["by_danger"]:
        assert bucket["frames"] == 0
        for key in ("top1", "top3", "top3_tie", "near_optimal_05", "regret_px", "regret_p90_px"):
            assert bucket[key] == 0.0


def test_policy_metrics_rejects_bad_shape_frame_count_and_non_finite():
    pred = np.zeros((2, 18))
    teacher = np.zeros((2, 18))
    bullets = _no_bullets(2)
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.policy_metrics(np.zeros((2, 17)), teacher, bullets)
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.policy_metrics(pred, np.zeros((2, 19)), bullets)
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.policy_metrics(pred, np.zeros((3, 18)), bullets)
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.policy_metrics(pred, teacher, np.zeros((2, 6)))
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.policy_metrics(pred, teacher, np.zeros((3, 64, 6)))
    bad_pred = pred.copy()
    bad_pred[0, 0] = np.nan
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.policy_metrics(bad_pred, teacher, bullets)
    bad_teacher = teacher.copy()
    bad_teacher[0, 1] = np.inf
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.policy_metrics(pred, bad_teacher, bullets)
    bad_bullets = bullets.copy()
    bad_bullets[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.policy_metrics(pred, teacher, bad_bullets)



# -- heatmap --------------------------------------------------------------
def test_heatmap_hit_rate_tolerance():
    truth = np.zeros((2, 24), dtype=np.float32)
    truth[:, 0] = (12 + 0.5) / 24.0
    truth[:, 1] = (14 + 0.5) / 28.0
    logits = np.zeros((2, 28, 24))
    logits[0, 15, 13] = 1.0  # one cell away on both axes -> hit
    logits[1, 16, 12] = 1.0  # two rows away -> miss
    assert vm.heatmap_hit_rate(logits, truth) == pytest.approx(0.5, rel=1e-9)
    exact = np.zeros((1, 28, 24))
    exact[0, 14, 12] = 1.0
    assert vm.heatmap_hit_rate(exact, truth[:1]) == pytest.approx(1.0, rel=1e-9)


def test_heatmap_hit_rate_uses_actual_grid_size():
    """A 56x48 head must be scored against its own grid, not the 28x24 v2 grid."""
    truth = np.zeros((3, 24), dtype=np.float32)
    truth[:, 0] = (30 + 0.5) / 48.0
    truth[:, 1] = (20 + 0.5) / 56.0
    logits = np.zeros((3, 56, 48))
    logits[0, 20, 30] = 1.0  # exact cell
    logits[1, 21, 31] = 1.0  # one cell away on both axes -> hit
    logits[2, 23, 30] = 1.0  # three rows away -> miss
    assert vm.heatmap_hit_rate(logits, truth) == pytest.approx(2.0 / 3.0, abs=5e-5)
    # the v1.0.3 hard-coded 28x24 grid scores this fixture 0, so the number
    # above cannot pass by accident: its truth cell is row 10 / col 15, ten rows
    # away from the (0, 0) peak below
    legacy_grid = np.zeros((1, 28, 24))
    legacy_grid[0, 0, 0] = 1.0
    assert vm.heatmap_hit_rate(legacy_grid, truth[:1]) == 0.0


def test_heatmap_hit_rate_28x24_and_56x48_both_hit():
    truth = np.zeros((1, 24), dtype=np.float32)
    truth[0, 0] = (12 + 0.5) / 24.0
    truth[0, 1] = (14 + 0.5) / 28.0
    small = np.zeros((1, 28, 24))
    small[0, 14, 12] = 1.0
    large = np.zeros((1, 56, 48))
    large[0, 28, 24] = 1.0
    assert vm.heatmap_hit_rate(small, truth) == 1.0
    assert vm.heatmap_hit_rate(large, truth) == 1.0


def test_heatmap_hit_rate_rejects_bad_shape_and_non_finite():
    truth = np.zeros((1, 24), dtype=np.float32)
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.heatmap_hit_rate(np.zeros((28, 24)), truth)  # logits must be [N, H, W]
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.heatmap_hit_rate(np.zeros((2, 28, 24)), truth)  # frame count mismatch
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.heatmap_hit_rate(np.zeros((1, 0, 24)), truth)  # empty grid
    logits = np.zeros((1, 28, 24))
    logits[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.heatmap_hit_rate(logits, truth)
    assert vm.heatmap_hit_rate(np.zeros((0, 28, 24)), np.zeros((0, 24), dtype=np.float32)) == 0.0
    assert vm.heatmap_hit_rate(np.zeros((0, 56, 48)), np.zeros((0, 24), dtype=np.float32)) == 0.0



# -- aggregation ----------------------------------------------------------
def test_all_state_metrics_keys():
    rng = np.random.default_rng(17)
    frames = 6
    truth_self = rng.random((frames, 24)).astype(np.float32)
    truth_self[:, 14] = 0.4
    pred_self = truth_self + rng.normal(0, 0.01, size=(frames, 24)).astype(np.float32)
    truth_danger = rng.integers(0, 6, size=(frames, 16, 8)).astype(np.uint8)
    pred_danger = np.log1p(truth_danger / 8.0) + rng.normal(0, 0.01, size=(frames, 16, 8))
    metrics = vm.all_state_metrics(pred_self, pred_danger, truth_self, truth_danger)
    assert set(metrics) == {
        "self_mae",
        "player_pos_px",
        "player_pos_p95_px",
        "ray_mae",
        "bullet_count_mae",
        "danger_mae_log",
        "danger_corr",
        "danger_corr_perframe_mean",
        "danger_corr_perframe_median",
        "danger_corr_perframe_valid",
        "danger_iou",
        "danger_cell_acc",
    }
    with_heatmap = vm.all_state_metrics(
        pred_self,
        pred_danger,
        truth_self,
        truth_danger,
        heatmap_logits=rng.random((frames, 28, 24)),
    )
    assert set(with_heatmap) == set(metrics) | {"heatmap_hit_rate"}


def test_metrics_never_return_nan():
    empty_self = np.zeros((0, 24), dtype=np.float32)
    empty_danger_pred = np.zeros((0, 128))
    empty_danger_truth = np.zeros((0, 16, 8), dtype=np.uint8)
    assert vm.player_position_mae_px(empty_self, empty_self) == (0.0, 0.0)
    assert vm.bullet_count_mae(empty_self, empty_self) == 0.0
    assert vm.ray_mae(empty_self, empty_self) == 0.0
    assert vm.self_mae(empty_self, empty_self) == 0.0
    danger = vm.danger_metrics(empty_danger_pred, empty_danger_truth)
    assert danger["danger_corr_perframe_valid"] == 0
    for key, value in danger.items():
        if key != "danger_corr_perframe_valid":
            assert value == 0.0
    policy = vm.policy_metrics(
        np.zeros((0, 18)), np.zeros((0, 18)), np.zeros((0, 64, 6), dtype=np.float32)
    )
    assert policy["top1"] == 0.0
    assert policy["regret_px"] == 0.0
    assert len(policy["by_danger"]) == 5
    assert vm.heatmap_hit_rate(np.zeros((0, 28, 24)), empty_self) == 0.0
    combined = vm.all_state_metrics(
        empty_self, empty_danger_pred, empty_self, empty_danger_truth
    )
    for key, value in combined.items():
        assert not (isinstance(value, float) and np.isnan(value)), key


def test_all_state_metrics_rejects_non_finite_predictions():
    """A ``nan`` prediction must fail, not become a 0.0 that passes A-01."""
    truth_self = np.zeros((4, 24), dtype=np.float32)
    truth_danger = np.zeros((4, 16, 8), dtype=np.uint8)
    pred_self = np.zeros((4, 24), dtype=np.float32)
    pred_danger = np.zeros((4, 16, 8))
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.player_position_mae_px(np.array([[np.nan] + [0.0] * 23]), truth_self[:1])
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.self_mae(np.array([[np.inf] + [0.0] * 23]), truth_self[:1])
    nan_self = pred_self.copy()
    nan_self[1, 0] = np.nan
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.all_state_metrics(nan_self, pred_danger, truth_self, truth_danger)
    inf_danger = pred_danger.copy()
    inf_danger[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.all_state_metrics(pred_self, inf_danger, truth_self, truth_danger)
    nan_heatmap = np.zeros((4, 28, 24))
    nan_heatmap[2, 5, 5] = np.nan
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.all_state_metrics(
            pred_self,
            pred_danger,
            truth_self,
            truth_danger,
            heatmap_logits=nan_heatmap,
        )
    bad_truth = truth_self.copy()
    bad_truth[0, 1] = np.inf
    with pytest.raises(ValueError, match="v104:DATA"):
        vm.all_state_metrics(pred_self, pred_danger, bad_truth, truth_danger)


def test_all_state_metrics_accepts_56x48_heatmap():
    frames = 3
    truth_self = np.zeros((frames, 24), dtype=np.float32)
    truth_self[:, 0] = (24 + 0.5) / 48.0
    truth_self[:, 1] = (28 + 0.5) / 56.0
    heatmap = np.zeros((frames, 56, 48))
    heatmap[:, 28, 24] = 1.0
    metrics = vm.all_state_metrics(
        truth_self,
        np.zeros((frames, 16, 8)),
        truth_self,
        np.zeros((frames, 16, 8), dtype=np.uint8),
        heatmap_logits=heatmap,
    )
    assert metrics["heatmap_hit_rate"] == 1.0


# -- stratified sub-sampling ---------------------------------------------
def test_stratified_prefix_is_nested():
    stage_map = np.repeat(np.array([1, 2, 3], dtype=np.int64), 100)
    train = list(range(300))
    quarter = stratified_prefix(train, stage_map, 0.25, seed=7)
    half = stratified_prefix(train, stage_map, 0.5, seed=7)
    full = stratified_prefix(train, stage_map, 1.0, seed=7)
    assert len(quarter) == 75
    assert len(half) == 150
    assert set(quarter) <= set(half) <= set(full)
    assert full == train
    assert set(full) == set(train)
    # same seed, same answer
    assert quarter == stratified_prefix(train, stage_map, 0.25, seed=7)
    assert quarter == sorted(quarter)


def test_stratified_prefix_is_stratified():
    stage_map = np.repeat(np.array([2, 5, 9], dtype=np.int64), 100)
    train = list(range(300))
    picked = stratified_prefix(train, stage_map, 0.5, seed=7)
    counts = np.bincount(stage_map[np.asarray(picked)], minlength=10)
    assert [int(counts[stage]) for stage in (2, 5, 9)] == [50, 50, 50]
    # a non-contiguous training set (the real case: 10% held out for validation)
    subset = [index for index in range(300) if index % 10 != 0]
    subset_picked = stratified_prefix(subset, stage_map, 0.5, seed=7)
    subset_counts = np.bincount(stage_map[np.asarray(subset_picked)], minlength=10)
    assert [int(subset_counts[stage]) for stage in (2, 5, 9)] == [45, 45, 45]
    assert set(subset_picked) <= set(subset)


@pytest.mark.parametrize("fraction", [0.0, 1.5, -0.1])
def test_stratified_prefix_rejects_bad_fraction(fraction):
    stage_map = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="fraction must be in"):
        stratified_prefix(list(range(10)), stage_map, fraction)


# -- dataset metadata -----------------------------------------------------
def _observation(seed: int = 0) -> obs.Observation:
    rng = np.random.default_rng(seed)
    return obs.Observation(
        self_feat=rng.random(obs.N_SELF, dtype=np.float32),
        bullets=rng.random((obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float32).astype(np.float16),
        danger_map=rng.integers(0, 8, size=(obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.uint8),
    )


def _regions(value: int) -> dict[str, bytes]:
    sizes = {"playfield": (48, 64), "hud_score": (8, 32), "hud_status": (16, 32)}
    return {
        name: vision.encode_png(np.full((height, width, 3), value, dtype=np.uint8))
        for name, (height, width) in sizes.items()
    }


def test_stage_of_matches_meta(tmp_path, monkeypatch):
    stages = [2, 2, 2, 2, 3, 3, 3]
    writer = VisionShardWriter(tmp_path, shard_size=3, tag="t")
    for index, stage in enumerate(stages):
        writer.add(
            _regions(index * 10),
            _observation(index),
            (0, 1, 0, 0),
            {
                "frame": index,
                "stage": stage,
                "score": 0,
                "graze": 0,
                "lives": 0,
                "bombs": 0,
                "power": 0,
                "deaths": 0,
            },
        )
    writer.close()
    dataset = VisionDataset(tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("stage_of must read meta only, never decode a PNG")

    monkeypatch.setattr(vision, "decode_png", boom)
    picked = dataset.stage_of([0, 4, 6])
    assert picked.dtype == np.int64
    assert list(picked) == [2, 3, 3]
    assert list(dataset.stage_of(np.arange(7))) == stages
    assert dataset.stage_of(np.asarray([2, 5])).shape == (2,)
