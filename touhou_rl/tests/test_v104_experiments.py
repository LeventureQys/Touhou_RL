"""Unit tests for the v1.0.4 S3-4 tool ``tools/rl/run_v104_experiments.py``.

What is pinned here, because none of it can be inferred from a green long run:

* the six declared configurations are exactly the task book's table (depth,
  objective, rank weight, regression weight), and the matrix always runs two
  perception runs first and then six policy runs from the *same-depth*
  checkpoint;
* **every** emitted training command carries ``--max-val-batches 0`` and
  ``training_command`` has no parameter that could change it - a shortened
  validation pass would empty the 0-15 danger bucket and the trainer would then
  refuse to write ``policy.pt`` (S3-3 §10.3);
* the overfit sampler takes 256 centre frames from the *training* split only,
  apportions them deterministically with capacity recycling, and never divides
  by a degenerate (``<= 0`` or non-finite) initial regret;
* the epoch forecast is ``floor(65172/96) * train + ceil(7241/96) * val`` and the
  60 s/epoch ceiling is *recorded* as a flag, not enforced as a stop (user ruling
  on question list §8 Q5);
* finished evidence is write-once, the matrix refuses a run directory whose
  recipe is not the planned one, and the gate blocks - rather than passes - when
  the matrix evidence is missing.

Every fixture is synthetic: no game, no training, no real dataset.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_tool():
    """Import ``tools/rl/run_v104_experiments.py`` as a module (tools is no package)."""
    module_name = "v104_experiments_tool"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = ROOT / "tools" / "rl" / "run_v104_experiments.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


@pytest.fixture(autouse=True)
def _never_launch_training(monkeypatch):
    """A unit test must never spend GPU time: any real launch is a hard error.

    The refusal tests stop inside the pre-flight, so reaching the launcher means
    the contract under test did not hold.
    """
    def boom(*_args, **_kwargs):
        raise AssertionError("a unit test tried to launch a real training run")

    monkeypatch.setattr(TOOL, "run_subprocess", boom)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
class _FakeDataset:
    """Minimal ``labels``-only stand-in: 3 stages x 5 danger buckets."""

    def __init__(self, total: int = 600):
        self.total = int(total)

    def labels(self, indices):
        count = len(indices)
        bullets = np.zeros((count, 64, 6), dtype=np.float32)
        stage = np.zeros((count, 8), dtype=np.float32)
        distances = [5.0, 20.0, 45.0, 90.0, float("inf")]
        for row, index in enumerate(indices):
            value = int(index)
            stage[row, 1] = float(1 + value % 3)
            distance = distances[value % 5]
            if np.isfinite(distance):
                bullets[row, 0, 0] = distance * 0.6 / 192.0
                bullets[row, 0, 1] = distance * 0.8 / 224.0
                bullets[row, 0, 5] = 1.0  # the validity flag
        return {
            "self_feat": np.zeros((count, 24), dtype=np.float32),
            "danger": np.zeros((count, 16, 8), dtype=np.uint8),
            "bullets": bullets,
            "meta": stage,
            "action": np.zeros((count, 4), dtype=np.int64),
        }


def _summary(**overrides):
    """A minimal valid ``vision-summary/2`` payload for recipe checks."""
    payload = {
        "schema": "vision-summary/2",
        "contract_version": "v104-A/1",
        "recipe": {
            "selection_contract": "v104",
            "history_contract": "causal-segment-v1",
            "stage": "policy",
            "depth": 3,
            "seed": 7,
            "batch_size": 96,
            "max_val_batches": 0,
            "max_train_batches": 0,
            "screening": False,
            "epochs_policy": 12,
        },
        "loss": {"policy": {"objective": "rank", "rank_weight": 1.0, "regression_weight": 0.0}},
        "stages": {"policy": {"history": [{"seconds": 71.2}], "best": {}}},
    }
    for key, value in overrides.items():
        payload[key] = value
    return payload


# --------------------------------------------------------------------------- #
# the six declared configurations
# --------------------------------------------------------------------------- #
def test_the_six_configurations_match_the_task_book():
    assert TOOL.EXPERIMENT_ORDER == ("E0", "E1", "E2", "E3a", "E3b", "E3c")
    expected = {
        "E0": (1, "legacy-reg", 0.0, 1.0),
        "E1": (3, "legacy-reg", 0.0, 1.0),
        "E2": (3, "centered-reg", 0.0, 1.0),
        "E3a": (3, "rank", 1.0, 0.0),
        "E3b": (3, "rank", 1.0, 0.1),
        "E3c": (3, "rank", 1.0, 1.0),
    }
    for name, (depth, objective, rank, regression) in expected.items():
        spec = TOOL.EXPERIMENTS[name]
        assert (spec["depth"], spec["objective"], spec["rank_weight"], spec["regression_weight"]) == (
            depth,
            objective,
            rank,
            regression,
        )


def test_matrix_plan_has_two_perception_runs_then_six_policy_runs():
    plans = TOOL.matrix_plan(estimates=TOOL.FALLBACK_EPOCH_SECONDS)
    assert len(plans) == 8
    assert [plan["kind"] for plan in plans[:2]] == ["state", "state"]
    assert [plan["depth"] for plan in plans[:2]] == [1, 3]
    assert [plan["experiment"] for plan in plans[2:]] == list(TOOL.EXPERIMENT_ORDER)
    assert plans[2]["depth"] == 1  # E0 is the depth-1 configuration
    assert {plan["depth"] for plan in plans[3:]} == {3}
    assert plans[0]["epochs"] == TOOL.STATE_EPOCHS == 72
    assert all(plan["epochs"] == TOOL.POLICY_EPOCHS == 12 for plan in plans[2:])


def test_policy_plans_start_from_their_own_depth_state_run():
    plans = TOOL.matrix_plan(estimates=TOOL.FALLBACK_EPOCH_SECONDS)
    for plan in plans[2:]:
        source = plans[plan["state_plan_index"]]
        assert source["kind"] == "state"
        assert source["depth"] == plan["depth"]
        assert TOOL.INIT_FROM_PLACEHOLDER in plan["command"]
    bound = TOOL.bind_init_from(plans[3], Path("third_party/th08/build/rl/runs/v104-state-d3-s7/state.pt"))
    index = bound.index("--init-from")
    assert bound[index + 1].endswith("v104-state-d3-s7/state.pt")
    assert "--depth" in bound and bound[bound.index("--depth") + 1] == "3"


def test_every_matrix_command_requests_the_full_validation_pass():
    for plan in TOOL.matrix_plan(estimates=TOOL.FALLBACK_EPOCH_SECONDS):
        command = plan["command"]
        assert "--max-val-batches" in command
        values = [command[i + 1] for i, part in enumerate(command) if part == "--max-val-batches"]
        assert values == ["0"], plan["experiment"]


def test_training_command_has_no_validation_limit_parameter():
    parameters = inspect.signature(TOOL.training_command).parameters
    assert "max_val_batches" not in parameters
    assert "max_train_batches" in parameters  # the training limit stays explicit (smoke)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"stage": "state", "epochs_state": 0}, "epochs_state"),
        ({"stage": "policy", "epochs_policy": 0}, "epochs_policy"),
        ({"depth": 2, "stage": "state", "epochs_state": 1}, "depth"),
        ({"objective": "rank", "rank_weight": 0.0, "stage": "state", "epochs_state": 1}, "rank_weight"),
        ({"objective": "nope", "stage": "state", "epochs_state": 1}, "objective"),
        ({"stage": "both", "epochs_state": 1}, "stage"),
    ],
)
def test_training_command_rejects_illegal_recipes(kwargs, message):
    base = {
        "out_dir": ROOT / "third_party/th08/build/rl/runs/v104-E0-s7",
        "depth": 3,
        "stage": "state",
        "objective": "legacy-reg",
        "rank_weight": 0.0,
        "regression_weight": 1.0,
        "epochs_state": 1,
    }
    base.update(kwargs)
    with pytest.raises(ValueError) as error:
        TOOL.training_command(**base)
    assert str(error.value).startswith("v104:CONFIG:")
    assert message in str(error.value)


def test_rank_configurations_carry_their_declared_weights():
    plans = {plan["experiment"]: plan for plan in TOOL.matrix_plan(estimates=TOOL.FALLBACK_EPOCH_SECONDS)}
    for name, rank, regression in (("E3a", "1", "0"), ("E3b", "1", "0.1"), ("E3c", "1", "1")):
        command = plans[name]["command"]
        assert command[command.index("--policy-objective") + 1] == "rank"
        assert command[command.index("--rank-weight") + 1] == rank
        assert command[command.index("--regression-weight") + 1] == regression
    assert plans["E2"]["command"][plans["E2"]["command"].index("--policy-objective") + 1] == "centered-reg"
    for plan in plans.values():
        command = plan["command"]
        assert command[command.index("--selection-contract") + 1] == "v104"
        assert command[command.index("--history-contract") + 1] == "causal-segment-v1"
        assert command[command.index("--seed") + 1] == "7"
        assert command[command.index("--batch-size") + 1] == "96"


def test_smoke_plan_is_short_and_uses_its_own_run_directories():
    plans = TOOL.matrix_plan(
        ["E0", "E3a"],
        smoke=True,
        smoke_epochs=1,
        smoke_train_batches=2,
        estimates=TOOL.FALLBACK_EPOCH_SECONDS,
    )
    assert [plan["experiment"] for plan in plans] == ["state-d1", "state-d3", "E0", "E3a"]
    assert [plan["run_dir"].name for plan in plans] == [
        "v104-state-d1-smoke-s7",
        "v104-state-d3-smoke-s7",
        "v104-E0-smoke-s7",
        "v104-E3a-smoke-s7",
    ]
    for plan in plans:
        assert plan["epochs"] == 1
        assert "-smoke-" in plan["run_dir"].name
        command = plan["command"]
        assert command[command.index("--max-train-batches") + 1] == "2"
        assert command[command.index("--max-val-batches") + 1] == "0"
    assert all(plan["seconds_per_epoch"] > 0 for plan in plans)


def test_screening_uses_24_perception_epochs_and_no_training_limit():
    plans = TOOL.matrix_plan(["E0"], screening=True, estimates=TOOL.FALLBACK_EPOCH_SECONDS)
    state = plans[0]
    assert state["epochs"] == TOOL.SCREENING_STATE_EPOCHS == 24
    assert "--max-train-batches" not in state["command"]
    assert plans[1]["epochs"] == TOOL.POLICY_EPOCHS
    with pytest.raises(ValueError):
        TOOL.matrix_plan(["E0"], smoke=True, screening=True)


# --------------------------------------------------------------------------- #
# CLI contract
# --------------------------------------------------------------------------- #
def _cli(*argv):
    return TOOL.main([str(part) for part in argv])


def test_dry_run_creates_no_directory(tmp_path, capsys):
    out = tmp_path / "matrix-dry"
    assert _cli("--phase", "matrix", "--out", out) == 0
    assert not out.exists()
    text = capsys.readouterr().out
    assert "formal plan" in text
    assert "total estimate" in text
    for name in TOOL.EXPERIMENT_ORDER:
        assert f"v104-{name}-s7" in text


def test_dry_run_reports_every_run_and_its_seconds(tmp_path, capsys):
    assert _cli("--phase", "matrix", "--out", tmp_path / "m") == 0
    text = capsys.readouterr().out
    assert "v104-state-d1-s7" in text and "v104-state-d3-s7" in text
    assert text.count("touhou_rl.vision_train") == 8


def test_smoke_requires_an_experiment(tmp_path, capsys):
    assert _cli("--phase", "matrix", "--out", tmp_path / "s", "--smoke") == 2
    assert "--smoke requires an explicit --experiment" in capsys.readouterr().err


def test_smoke_switches_are_matrix_only(tmp_path, capsys):
    assert _cli("--phase", "overfit", "--experiment", "E3a", "--out", tmp_path / "o", "--smoke-epochs", 1) == 2
    assert "--phase matrix" in capsys.readouterr().err


def test_smoke_switches_need_the_smoke_flag(tmp_path, capsys):
    assert (
        _cli("--phase", "matrix", "--out", tmp_path / "m", "--smoke-epochs", "1", "--experiment", "E0") == 2
    )
    assert "without --smoke" in capsys.readouterr().err


def test_dataset_b_is_refused(tmp_path, capsys):
    other = ROOT / "third_party" / "th08" / "build" / "rl" / "datasets" / "pilot-ud0063"
    assert _cli("--phase", "overfit", "--experiment", "E3a", "--out", tmp_path / "o", "--dataset", other) == 2
    message = capsys.readouterr().err
    assert message.startswith("v104:DATA:")
    assert "pilot-ud0063" in message


def test_overfit_requires_an_experiment(tmp_path, capsys):
    assert _cli("--phase", "overfit", "--out", tmp_path / "o") == 2
    assert "requires --experiment" in capsys.readouterr().err


def test_overfit_rejects_more_than_400_steps(tmp_path, capsys):
    assert _cli("--phase", "overfit", "--experiment", "E3a", "--out", tmp_path / "o", "--overfit-steps", 401) == 2
    assert "1..400" in capsys.readouterr().err


def test_gather_dedup_is_value_identical():
    """The de-duplicated depth-3 stack must equal the naive one, value for value."""
    total = 100
    depth = 3

    class _HistoryDataset:
        def stack_indices(self, indices, stack_depth):
            rows = []
            for index in indices:
                rows.append([max(0, int(index) - offset) for offset in range(stack_depth)][::-1])
            return np.asarray(rows, dtype=np.int64)

        def region_frames(self, indices, sizes):
            out = {}
            for name, (height, width) in sizes.items():
                array = np.zeros((len(indices), height, width, 3), dtype=np.uint8)
                for row, index in enumerate(indices):
                    array[row] = (int(index) % 251) + 1  # distinguishable per frame
                out[name] = array
            return out

    sizes = {"playfield": (4, 3), "hud_score": (2, 2)}
    chunk = list(range(total - 6, total))
    deduped = TOOL.batch_tensors(_HistoryDataset(), chunk, depth, sizes, "cpu")
    dataset = _HistoryDataset()
    history = dataset.stack_indices(chunk, depth)
    frames = dataset.region_frames(history.reshape(-1).tolist(), sizes)
    naive_stacked = {
        name: frames[name].reshape(len(chunk), depth, height, width, 3)
        for name, (height, width) in sizes.items()
    }
    naive = TOOL.regions_to_batch(naive_stacked, "cpu")
    import torch

    for name in sizes:
        assert torch.equal(deduped[name], naive[name]), name
    # the de-duplication really happened (18 slots -> a smaller distinct set)
    assert np.unique(history.reshape(-1)).size < history.size


# --------------------------------------------------------------------------- #
# overfit sampling
# --------------------------------------------------------------------------- #
def test_apportion_fills_the_budget_and_respects_capacity():
    sizes = [1, 4, 10, 30, 200, 500]
    split = TOOL.apportion_256(sizes, 256)
    allocation = split["allocation"]
    assert sum(allocation.values()) == 256
    assert split["unfilled"] == 0
    for index, size in enumerate(sizes):
        assert 1 <= allocation[index] <= size
    # everything proportional: the biggest stratum must get the most frames
    assert allocation[5] == max(allocation.values())


def test_apportion_returns_the_excess_when_capacity_runs_out():
    split = TOOL.apportion_256([5, 5], 256)
    assert split["allocation"] == {0: 5, 1: 5}
    assert split["unfilled"] == 246


def test_apportion_refuses_more_strata_than_the_budget():
    with pytest.raises(ValueError) as error:
        TOOL.apportion_256([1] * 257, 256)
    assert str(error.value).startswith("v104:DATA:")
    with pytest.raises(ValueError):
        TOOL.apportion_256([0, 0], 256)


def test_apportion_is_deterministic():
    sizes = [3, 7, 11, 19, 23]
    assert TOOL.apportion_256(sizes, 256) == TOOL.apportion_256(sizes, 256)


@pytest.mark.parametrize(
    "distance, bucket",
    [
        (0.0, "0-15"),
        (14.999, "0-15"),
        (15.0, "15-30"),
        (29.999, "15-30"),
        (30.0, "30-60"),
        (60.0, "60-120"),
        (119.999, "60-120"),
        (120.0, "120+"),
        (float("inf"), "120+"),
    ],
)
def test_danger_bucket_boundaries(distance, bucket):
    assert TOOL.assign_bucket(distance) == bucket


def test_bucket_nan_is_a_data_error():
    with pytest.raises(ValueError) as error:
        TOOL.assign_bucket(float("nan"))
    assert str(error.value).startswith("v104:DATA:")


def test_stratified_center_frames_uses_only_training_indices():
    dataset = _FakeDataset(total=600)
    train = list(range(450, 600)) + list(range(0, 300))  # deliberately unsorted
    selection = TOOL.stratified_center_frames(dataset, train, target=256, seed=7)
    assert selection["frames"] == 256
    assert len(set(selection["indices"])) == 256
    assert set(selection["indices"]) <= set(train)
    assert selection["indices"] == sorted(selection["indices"])
    assert selection["indices_sha256"] == TOOL.index_digest(selection["indices"])
    assert selection["replacement"] is False
    assert selection["source_split"] == "train"
    # every stage x bucket stratum that exists is represented
    strata = {(row["stage"], row["bucket"]) for row in selection["strata"]}
    assert len(strata) == 15
    assert all(row["allocated"] >= 1 for row in selection["strata"])


def test_stratified_selection_is_deterministic():
    dataset = _FakeDataset(total=600)
    train = list(range(0, 300))
    first = TOOL.stratified_center_frames(dataset, train, target=256, seed=7)
    second = TOOL.stratified_center_frames(dataset, train, target=256, seed=7)
    assert first["indices"] == second["indices"]
    assert first["indices_sha256"] == second["indices_sha256"]


def test_stratified_selection_handles_a_tiny_stratum():
    dataset = _FakeDataset(total=600)
    train = list(range(0, 300)) + [301]
    selection = TOOL.stratified_center_frames(dataset, train, target=256, seed=7)
    assert selection["frames"] == 256
    assert set(selection["indices"]) <= set(train)
    assert len(set(selection["indices"])) == 256


# --------------------------------------------------------------------------- #
# overfit entry rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("initial", [0.0, -0.5, float("nan"), float("inf"), None, "x"])
def test_overfit_entry_blocks_a_degenerate_initial_regret(initial):
    entry = TOOL.overfit_entry_status(initial, 1.0)
    assert entry["status"] == "blocked"
    assert entry["reason"] == "degenerate_initial_regret"
    assert entry["reduction"] is None
    assert entry["line_reached"] is False


@pytest.mark.parametrize(
    "initial, final, status",
    [
        (5.0, 3.4, "pass"),   # 32% lower
        (5.0, 3.5, "pass"),   # exactly 30%
        (5.0, 4.0, "fail"),   # 20% lower
        (5.0, 5.0, "fail"),   # no movement
        (5.0, float("nan"), "fail"),
    ],
)
def test_overfit_entry_measures_the_thirty_percent_line(initial, final, status):
    entry = TOOL.overfit_entry_status(initial, final)
    assert entry["status"] == status
    assert entry["threshold"] == TOOL.OVERFIT_REDUCTION


# --------------------------------------------------------------------------- #
# throughput arithmetic
# --------------------------------------------------------------------------- #
def test_forecast_uses_floor_and_ceil_batch_counts():
    assert TOOL.TRAIN_BATCHES == 65172 // 96 == 678
    assert TOOL.VAL_BATCHES == -(-7241 // 96) == 76
    assert TOOL.forecast_epoch_seconds(0.001, 0.002) == pytest.approx(678 * 0.001 + 76 * 0.002)


def test_measured_range_excludes_the_warmup_prefix():
    assert list(TOOL.measured_range(110, measure=100)) == list(range(10, 110))
    assert len(list(TOOL.measured_range(110, measure=100))) == 100
    assert list(TOOL.measured_range(76)) == list(range(10, 76))
    assert list(TOOL.measured_range(5)) == []
    assert TOOL.WARMUP_BATCHES == 10 and TOOL.MEASURE_BATCHES == 100


def test_policy_mixed_epoch_is_the_twelve_epoch_average():
    mixed = TOOL.policy_mixed_epoch_seconds(40.0, 90.0)
    assert mixed == pytest.approx((2 * 40.0 + 10 * 90.0) / 12)
    assert TOOL.policy_mixed_epoch_seconds(50.0, 50.0) == pytest.approx(50.0)


def test_budget_ceiling_is_recorded_not_enforced():
    assert TOOL.budget_exceeded([39.0, 39.2]) is False
    assert TOOL.budget_exceeded([60.0]) is False
    assert TOOL.budget_exceeded([71.15]) is True
    assert TOOL.budget_overrun([71.15]) == pytest.approx(0.1858, abs=1e-3)
    assert TOOL.budget_overrun([39.0]) == 0.0
    assert TOOL.budget_exceeded([]) is False


def test_budget_uses_the_steady_state_not_the_cold_start_epoch():
    """Epoch 1-2 pay the one-time cache fill; 設計 §5.1 excludes that from the ceiling."""
    depth1 = [105.0, 65.1, 40.6, 40.3, 40.9, 40.6, 40.3]
    depth3 = [114.7, 86.5, 69.6, 69.0, 68.8, 68.8, 69.2]
    assert TOOL.epoch_seconds_summary(depth1)["all_max_seconds"] == 105.0
    assert TOOL.epoch_seconds_summary(depth1)["steady_max_seconds"] == 40.9
    assert TOOL.budget_exceeded(depth1) is False  # cold start alone must not flag a run
    assert TOOL.budget_exceeded(depth3) is True  # 69.2 s steady is genuinely above 60
    summary = TOOL.epoch_seconds_summary(depth3)
    assert summary["steady_exceeds_ceiling"] is True
    assert summary["cold_start_epochs"] == 2
    assert summary["cold_start_seconds"] == [114.7, 86.5]
    assert summary["steady_max_seconds"] == 69.6
    assert summary["steady_overrun_fraction"] == pytest.approx(69.6 / 60.0 - 1.0, abs=1e-4)
    assert TOOL.epoch_seconds_summary([])["steady_max_seconds"] is None


def test_epoch_seconds_summary_handles_short_runs():
    assert TOOL.epoch_seconds_summary([100.0])["cold_start_epochs"] == 0
    assert TOOL.epoch_seconds_summary([100.0])["steady_max_seconds"] == 100.0
    assert TOOL.epoch_seconds_summary([50.0, 52.0])["cold_start_epochs"] == 1
    assert TOOL.epoch_seconds_summary([50.0, 52.0])["steady_max_seconds"] == 52.0


# --------------------------------------------------------------------------- #
# act contract (v1/v2 return a four-tuple)
# --------------------------------------------------------------------------- #
def test_action_tensor_takes_the_first_item_of_the_act_tuple():
    import torch

    actions = torch.zeros((4, 4), dtype=torch.int64)
    result = (actions, torch.zeros(4), torch.zeros(4), torch.zeros(4))
    assert TOOL.act_contract(result) == "tuple4"
    assert TOOL.action_tensor(result) is actions
    assert TOOL.act_contract(actions) == "tensor"
    assert TOOL.action_tensor(actions) is actions


@pytest.mark.parametrize("result", [(), "actions", None, 3])
def test_action_tensor_rejects_a_broken_act_return(result):
    with pytest.raises(ValueError) as error:
        TOOL.action_tensor(result)
    assert str(error.value).startswith("v104:CHECKPOINT:")


def test_estimates_from_throughput_feed_the_matrix_plan(tmp_path):
    payload = {
        "depth": 3,
        "paths": {
            "perception": {"forecast_epoch_seconds": 72.0},
            "policy_frozen": {"forecast_epoch_seconds": 60.0},
            "policy_joint": {"forecast_epoch_seconds": 90.0},
            "perception_depth1_extra": {"forecast_epoch_seconds": 40.0},
        },
    }
    path = tmp_path / "throughput.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    estimates = TOOL.estimates_from_throughput(path)
    assert estimates["state"][3] == 72.0
    assert estimates["state"][1] == 40.0
    assert estimates["policy"][3] == pytest.approx((2 * 60.0 + 10 * 90.0) / 12)
    plans = TOOL.matrix_plan(estimates=estimates)
    state = next(plan for plan in plans if plan["kind"] == "state" and plan["depth"] == 3)
    assert state["estimated_seconds"] == pytest.approx(72.0 * 72)


# --------------------------------------------------------------------------- #
# write-once evidence and recipe reuse
# --------------------------------------------------------------------------- #
def test_refuses_to_overwrite_finished_evidence(tmp_path):
    path = tmp_path / "baseline.json"
    TOOL.write_json(path, {"status": "done"})
    with pytest.raises(ValueError) as error:
        TOOL.refuse_existing_output(path)
    assert str(error.value).startswith("v104:EVIDENCE:")
    TOOL.write_json(path, {"status": "running"})
    TOOL.refuse_existing_output(path)  # a crashed run may be resumed
    TOOL.refuse_existing_output(tmp_path / "absent.json")


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, None),
        ({"recipe": {**_summary()["recipe"], "depth": 1}}, "depth"),
        ({"recipe": {**_summary()["recipe"], "epochs_policy": 24}}, "epochs_policy"),
        ({"recipe": {**_summary()["recipe"], "max_val_batches": 96}}, "max_val_batches"),
        ({"recipe": {**_summary()["recipe"], "screening": True}}, "screening"),
        ({"recipe": {**_summary()["recipe"], "seed": 17}}, "seed"),
        ({"loss": {"policy": {"objective": "legacy-reg", "rank_weight": 0.0, "regression_weight": 1.0}}}, "objective"),
    ],
)
def test_run_recipe_problem_catches_a_different_recipe(overrides, expected):
    plan = {
        "kind": "policy",
        "depth": 3,
        "epochs": 12,
        "max_train_batches": 0,
        "objective": "rank",
        "rank_weight": 1.0,
        "regression_weight": 0.0,
    }
    problem = TOOL.run_recipe_problem(_summary(**overrides), plan, screening=False)
    if expected is None:
        assert problem is None
    else:
        assert problem is not None and expected in problem


def test_matrix_refuses_a_run_directory_with_a_different_recipe(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(TOOL, "RUNS_ROOT", tmp_path / "runs")
    run_dir = tmp_path / "runs" / "v104-state-d1-smoke-s7"
    run_dir.mkdir(parents=True)
    TOOL.write_json(run_dir / "vision-summary.json", _summary(recipe={**_summary()["recipe"], "depth": 3}))
    code = _cli(
        "--phase", "matrix", "--out", tmp_path / "m", "--smoke", "--experiment", "E0", "--execute"
    )
    assert code == 2
    message = capsys.readouterr().err
    assert message.startswith("v104:EVIDENCE:")
    assert "refusing to reuse or overwrite" in message
    assert not (tmp_path / "m" / "matrix.json").exists()


# --------------------------------------------------------------------------- #
# experiment table
# --------------------------------------------------------------------------- #
def test_experiment_table_appends_without_rewriting_history(tmp_path, monkeypatch):
    monkeypatch.setattr(TOOL, "S34_DIR", tmp_path / "S3-4")
    payload = {
        "smoke": False,
        "screening": False,
        "created_at": "2026-10-19T00:00:00+08:00",
        "finished_at": "2026-10-19T01:00:00+08:00",
        "command": "cmd",
        "total_seconds": 100.0,
        "state_sources": {"3": {"run": "runs/v104-state-d3-s7", "checkpoint": "state-pass.pt", "sha256": "a" * 64}},
        "runs": [
            {
                "experiment": "E3a",
                "kind": "policy",
                "depth": 3,
                "run_dir": "runs/v104-E3a-s7",
                "epochs": [71.0, 71.5],
                "max_epoch_seconds": 71.5,
                "budget_exceeded": True,
                "acceptance_eligible": True,
                "partial": False,
                "checkpoint": "policy.pt",
                "checkpoint_sha256": "b" * 64,
                "source_state_sha256": "a" * 64,
                "selection_rule": "bucket-0-15-regret",
                "coverage_val": {"actual_frames": 7241, "complete": True},
                "teacher": {"fingerprint": TOOL.TEACHER_FINGERPRINT},
                "recipe": {"selection_contract": "v104", "history_contract": "causal-segment-v1"},
                "contract_version": "v104-A/1",
                "metrics": {
                    "policy_best": {"val_regret_px": 4.0, "epoch": 3},
                    "by_danger": [{"bucket": "0-15", "frames": 900, "regret_px": 1.2, "top3": 0.4}],
                },
                "final_policy_perception": {
                    "checkpoint": "policy.pt",
                    "checkpoint_sha256": "b" * 64,
                    "frames": 7241,
                    "metrics": {"player_pos_px": 9.0, "danger_corr_perframe_median": 0.85},
                },
                "seconds": 10.0,
                "parameters": 1,
                "summary_sha256": "c" * 64,
                "log": "log",
                "summary": "summary",
                "reused": False,
            }
        ],
    }
    table_path = TOOL.write_experiment_table(payload, matrix_path=tmp_path / "m.json", out_dir=tmp_path / "out")
    table = json.loads(table_path.read_text(encoding="utf-8"))
    assert table["schema"] == TOOL.TABLE_SCHEMA
    assert len(table["rows"]) == 1
    row = table["rows"][0]
    assert row["experiment"] == "E3a"
    assert row["budget"]["exceeded"] is True
    assert row["budget"]["overrun_fraction"] == pytest.approx(0.1917, abs=1e-3)
    assert row["eligible"] is True
    assert row["metrics"]["policy_0_15"]["regret_px"] == 1.2
    assert row["final_policy_perception"]["a01_pos_pass"] is True
    assert row["contract"]["history_contract"] == "causal-segment-v1"
    # the same run twice must not duplicate the row
    TOOL.write_experiment_table(payload, matrix_path=tmp_path / "m.json", out_dir=tmp_path / "out")
    assert len(json.loads(table_path.read_text(encoding="utf-8"))["rows"]) == 1


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #
def test_gate_blocks_without_matrix_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(TOOL, "S34_DIR", tmp_path / "S3-4")
    monkeypatch.setattr(TOOL, "EVIDENCE_ROOT", tmp_path / "S3-4" / "evidence")
    out = tmp_path / "gate"
    code = _cli("--phase", "gate", "--out", out)
    assert code == 2
    payload = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    assert payload["route"] == "blocked"
    assert payload["v3_started"] is False
    assert payload["counts"]["pass"] == 0
    assert all(item["status"] == "blocked" for item in payload["items"].values())
    assert (out / "gate-v3.md").is_file()
    assert (tmp_path / "S3-4" / "gate-v3.md").is_file()


def _policy_row(name, bucket, position, danger, epoch):
    return {
        "experiment": name,
        "kind": "policy",
        "depth": 3,
        "run_dir": f"runs/v104-{name}-s7",
        "metrics": {
            "policy_best": {"val_regret_px": 4.0, "epoch": epoch},
            "policy_0_15": {"frames": 900, "regret_px": bucket},
            "by_danger": [{"bucket": "0-15", "frames": 900, "regret_px": bucket}],
        },
        "final_policy_perception": {
            "a01_pos_pass": position < 10.0,
            "a01_danger_pass": danger > 0.8,
            "player_pos_px": position,
            "danger_corr_perframe_median": danger,
        },
        "source_state": {"sha256": "a" * 64},
        "checkpoint": {"sha256": "b" * 64},
    }


def test_gate_route_prefers_v2_when_bucket_and_perception_pass():
    rows = [_policy_row("E3a", 1.2, 9.0, 0.85, 3), _policy_row("E1", 4.0, 12.0, 0.7, 5)]
    ordered = TOOL.route_order(rows)
    assert ordered[0]["experiment"] == "E3a"
    route, reason, detail = TOOL.route_decision(ordered)
    assert route == "v2-sufficient"
    assert detail["v2_skip_v3"] is True
    assert detail["selection_key"]["bucket_0_15_regret_px"] == 1.2
    assert "does not replace A-01/A-02" in reason


def test_gate_route_falls_back_when_a_v3_entry_condition_fails():
    """A-02 not met + the <=60 s condition failed -> candidate-v3 but blocked."""
    rows = [_policy_row("E3a", 3.1, 8.6, 0.63, 9)]
    route, reason, detail = TOOL.route_decision(
        TOOL.route_order(rows),
        state_rows={3: {"metrics": {"state_best": {"val_player_pos_px": 13.06, "val_danger_corr_perframe_median": 0.62}}}},
        overfit_entry={"status": "pass", "reduction": 0.478},
        overfit_gradients={"finite": True, "nonzero": True},
        throughput_worst_steady=72.1,
        coverage_ok=True,
    )
    assert route == "candidate-v3-blocked"
    assert detail["v2_skip_v3"] is False
    assert detail["route_status"] == "blocked"
    entry = detail["v3_entry"]
    assert entry["conditions"]["small_set_regret_reduction_30pct"]["status"] == "pass"
    assert entry["conditions"]["valid_pairs_gradient"]["status"] == "pass"
    assert entry["conditions"]["sequence_labels_coverage"]["status"] == "pass"
    condition = entry["conditions"]["throughput_le_60s"]
    assert condition["status"] == "fail"
    assert condition["needs_user_ruling"] is True
    assert "Q5" in condition["q5_note"]
    assert entry["failed"] == ["throughput_le_60s"]
    assert entry["needs_user_ruling"] is True
    assert detail["a01_state"]["status"] == "fail"
    assert "needs a new user ruling" in reason
    # A-02 passes but the policy's own perception does not: still not v2-sufficient
    rows = [_policy_row("E3a", 1.0, 12.0, 0.85, 3)]
    route, _reason, detail = TOOL.route_decision(TOOL.route_order(rows))
    assert route == "candidate-v3-blocked"
    assert detail["policy_perception"]["a01_pos_pass"] is False


def test_gate_route_is_ready_when_every_v3_entry_condition_holds():
    rows = [_policy_row("E3a", 3.1, 8.6, 0.63, 9)]
    route, reason, detail = TOOL.route_decision(
        TOOL.route_order(rows),
        state_rows={3: {"metrics": {"state_best": {"val_player_pos_px": 13.0, "val_danger_corr_perframe_median": 0.62}}}},
        overfit_entry={"status": "pass", "reduction": 0.478},
        overfit_gradients={"finite": True, "nonzero": True},
        throughput_worst_steady=58.0,
        coverage_ok=True,
    )
    assert route == "candidate-v3-ready"
    assert detail["route_status"] == "pass"
    assert detail["v3_entry"]["status"] == "ready"
    assert detail["v3_entry"]["needs_user_ruling"] is False
    assert "<=60 s/epoch" in reason or "60 s/epoch" in reason


def test_gate_route_blocks_on_a_degenerate_small_set():
    rows = [_policy_row("E3a", 3.1, 8.6, 0.63, 9)]
    route, reason, detail = TOOL.route_decision(
        TOOL.route_order(rows),
        overfit_entry={"status": "blocked", "reason": "degenerate_initial_regret"},
        overfit_gradients={"finite": True, "nonzero": True},
        throughput_worst_steady=58.0,
    )
    assert route == "candidate-v3-blocked"
    assert detail["v3_entry"]["conditions"]["small_set_regret_reduction_30pct"]["status"] == "blocked"
    assert detail["v3_entry"]["blocked"] == ["small_set_regret_reduction_30pct"]
    assert "blocked=" in reason


def test_gate_route_treats_a_missing_measurement_as_blocked():
    rows = [_policy_row("E3a", 3.1, 8.6, 0.63, 9)]
    route, _reason, detail = TOOL.route_decision(TOOL.route_order(rows))
    assert route == "candidate-v3-blocked"
    assert detail["v3_entry"]["conditions"]["throughput_le_60s"]["status"] == "blocked"


def test_route_order_uses_bucket_then_global_then_epoch():
    def row(name, bucket, global_regret, epoch):
        return {
            "experiment": name,
            "metrics": {
                "policy_best": {"val_regret_px": global_regret, "epoch": epoch},
                "policy_0_15": {"regret_px": bucket},
            },
        }

    rows = [
        row("A", 2.0, 3.0, 1),
        row("B", 1.5, 9.0, 9),
        row("C", 1.5, 4.0, 9),
        row("D", 1.5, 4.0, 2),
    ]
    ordered = [item["experiment"] for item in TOOL.route_order(rows)]
    assert ordered == ["D", "C", "B", "A"]


def test_pytest_log_parser_reads_the_counts(tmp_path):
    log = tmp_path / "pytest.txt"
    log.write_text("..................................................\n482 passed, 3 skipped in 42.59s\n", encoding="utf-8")
    info = TOOL.parse_pytest_log(log)
    assert info["passed"] == 482
    assert info["skipped"] == 3
    assert info["failed"] == 0  # a clean run prints no "failed" token at all
    assert info["parsed"] is True
    log.write_text("2 failed, 480 passed in 20s\n", encoding="utf-8")
    info = TOOL.parse_pytest_log(log)
    assert info["failed"] == 2 and info["passed"] == 480
    log.write_text("554 passed in 18.83s\n", encoding="utf-8")
    info = TOOL.parse_pytest_log(log)
    assert (info["parsed"], info["passed"], info["failed"], info["errors"]) == (True, 554, 0, 0)
    log.write_text("no tests ran\n", encoding="utf-8")
    assert TOOL.parse_pytest_log(log)["parsed"] is False


def test_gate_requires_a_pytest_log_to_judge_a05(tmp_path, monkeypatch):
    monkeypatch.setattr(TOOL, "S34_DIR", tmp_path / "S3-4")
    monkeypatch.setattr(TOOL, "EVIDENCE_ROOT", tmp_path / "S3-4" / "evidence")
    (tmp_path / "S3-4").mkdir(parents=True)
    TOOL.write_json(
        tmp_path / "S3-4" / "experiment-table.json",
        {"schema": TOOL.TABLE_SCHEMA, "rows": [_policy_row(name, 1.0, 9.0, 0.9, 1) for name in TOOL.EXPERIMENT_ORDER]},
    )
    out = tmp_path / "gate"
    code = _cli("--phase", "gate", "--out", out, "--matrix", tmp_path / "S3-4" / "experiment-table.json")
    payload = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    assert payload["items"]["pytest"]["status"] == "blocked"
    assert code == 2
