"""Tests for the v3.4.0 Stage 0 offline gate (``touhou_rl.stage0_offline``)."""

import json
import math

import numpy as np
import pytest

from touhou_rl import chunk_safety, entity_obs, options
from touhou_rl import stage0_offline as s0

SNAPSHOT_FRAME = 100
PLAYER = [192.0, 384.0]


def _snapshot(
    *,
    bullets=None,
    player=None,
    frame=SNAPSHOT_FRAME,
    stage=0,
    player_state=3,
    complete=True,
    schema=s0.SNAPSHOT_SCHEMA,
):
    bullets = bullets if bullets is not None else [([210.0, 384.0], [-2.0, 0.0], 1.0)]
    return {
        "schema": schema,
        "frame": frame,
        "stage": stage,
        "complete": complete,
        "world_complete": False,
        "player_state": player_state,
        "player_position": list(player if player is not None else PLAYER),
        "player_half": [0.825, 0.825],
        "bullets": {
            "position": [list(item[0]) for item in bullets],
            "velocity": [list(item[1]) for item in bullets],
            "half": [float(item[2]) for item in bullets],
        },
    }


def _row(
    frame,
    delta,
    *,
    move=0,
    focus=0,
    executed_move=None,
    executed_focus=None,
    snapshot=None,
    terminated=False,
    remaining_bullet_offset=0.0,
):
    bullets = [([240.0 - remaining_bullet_offset, 384.0], [-2.0, 0.0], 1.0)]
    return {
        "frame": frame,
        "frame_delta": delta,
        "stage": 0,
        "terminated": terminated,
        "teacher_action": [move, 1, focus, 0],
        "executed_action": [
            move if executed_move is None else executed_move,
            1,
            0 if executed_focus is None else executed_focus,
            0,
        ],
        "raw_safety_snapshot": snapshot if snapshot is not None else _snapshot(bullets=bullets),
    }


def _write_trace(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_candidate_steps_match_options_order():
    steps = s0.candidate_steps()
    assert steps.shape == (18, 2)
    assert np.allclose(steps, options.STEPS, atol=1e-9)
    # focus*9 + move order: candidate 0 idle, 1 north, 9 focused idle.
    assert np.allclose(steps[0], 0.0)
    assert np.allclose(steps[1], [0.0, -4.0])
    assert np.allclose(steps[9], 0.0)


def test_clearance_matches_chunk_safety_at_durations():
    """The 10-frame horizon must agree with the shipping chunk model's 5-decision one."""
    snapshot = _snapshot(
        bullets=[
            ([215.0, 384.0], [-3.0, 0.0], 1.0),
            ([150.0, 300.0], [1.5, 1.5], 2.0),
            ([260.0, 420.0], [-1.0, 2.0], 0.5),
        ]
    )
    shipping = chunk_safety.predict(
        snapshot,
        chunk_safety.ChunkSafetyConfig(frameskip=2, speed_diag=2.8284, focus_speed_diag=1.4142),
    )
    mine = s0.label_snapshot(
        snapshot,
        horizons=(2, 6, 10),
        config=s0.GeometryConfig(speed_diag=2.8284, focus_speed_diag=1.4142),
    )
    assert mine["valid"] is True
    # durations 1/3/5 decisions with frameskip 2 are frames 2/6/10
    assert np.allclose(mine["clearance"], shipping["clearance"], atol=1e-4)
    assert np.array_equal(mine["safe"], shipping["clearance"] > 0)


def test_clearance_marks_the_blocking_direction_unsafe():
    # A bullet sitting exactly where "north" (candidate 1) leads must be unsafe.
    snapshot = _snapshot(bullets=[([192.0, 374.0], [0.0, 0.0], 1.0)])
    labelled = s0.label_snapshot(snapshot, horizons=(10,))
    north = labelled["clearance"][1, 0]
    idle = labelled["clearance"][0, 0]
    assert north <= 0.0
    assert idle > 0.0
    assert not bool(labelled["safe"][1, 0])
    assert bool(labelled["safe"][0, 0])


def test_clearance_ignores_bullets_outside_the_horizon():
    far = _snapshot(bullets=[([600.0, 900.0], [0.0, 0.0], 1.0)])
    labelled = s0.label_snapshot(far, horizons=s0.HORIZONS)
    assert np.all(labelled["clearance"] == s0.CLEARANCE_CAP)


def test_clearance_without_bullets_is_all_safe():
    empty = _snapshot(bullets=[])
    labelled = s0.label_snapshot(empty)
    assert np.all(labelled["clearance"] == s0.CLEARANCE_CAP)
    assert np.all(labelled["safe"])


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda item: item.update(schema="other/1"), "schema"),
        (lambda item: item.update(complete=False), "incomplete"),
        (lambda item: item.update(player_state=2), "player_state"),
        (lambda item: item["bullets"].update(half=[1.0, 2.0]), "malformed"),
        (lambda item: item.update(player_position=[float("nan"), 0.0]), "malformed"),
    ],
)
def test_invalid_snapshots_are_rejected_with_a_reason(mutate, reason):
    snapshot = _snapshot()
    mutate(snapshot)
    assert s0.parse_snapshot(snapshot) is None
    labelled = s0.label_snapshot(snapshot)
    assert labelled["valid"] is False
    assert labelled["reason"] == reason


def test_label_keeps_the_parsed_arrays_for_reuse():
    labelled = s0.label_snapshot(_snapshot())
    assert labelled["valid"] is True
    player, position, velocity, half, stage = labelled["parsed"]
    assert player.shape == (2,)
    assert position.shape == velocity.shape == (1, 2)
    assert half.shape == (1,)
    assert stage == 0


# ---------------------------------------------------------------------------
# targets and metrics
# ---------------------------------------------------------------------------


def test_clearance_target_roundtrip_keeps_ordering_and_bounds():
    values = np.array([0.0, 0.5, 5.0, 30.0, 60.0])
    target = s0.clearance_target(values)
    assert np.all((target >= 0.0) & (target <= 1.0))
    assert np.all(np.diff(target) > 0)
    restored = s0.clearance_from_target(target)
    assert np.allclose(restored, values)
    # Anything above the cap saturates at the cap instead of exceeding 1.
    assert s0.clearance_target(np.array([120.0]))[0] == pytest.approx(1.0)


def test_survival_targets_are_thresholds():
    remaining = np.array([0, 59, 60, 239, 240, 9999])
    labels = s0.survival_targets(remaining)
    assert labels.shape == (6, 2)
    assert labels[:, 0].tolist() == [0, 0, 1, 1, 1, 1]
    assert labels[:, 1].tolist() == [0, 0, 0, 0, 1, 1]


def test_rankdata_averages_ties():
    ranks = s0.rankdata([10.0, 20.0, 20.0, 30.0])
    assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_detects_monotone_and_reverse_orders():
    increasing = np.arange(50, dtype=np.float64)
    assert s0.spearman(increasing, increasing**3) == pytest.approx(1.0)
    decreasing = -increasing
    assert s0.spearman(increasing, decreasing) == pytest.approx(-1.0)
    assert s0.spearman(increasing, np.ones_like(increasing)) is None
    assert s0.spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None


def test_roc_auc_is_the_probability_of_correct_ordering():
    labels = np.array([0, 0, 1, 1])
    assert s0.roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert s0.roc_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert s0.roc_auc(labels, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert s0.roc_auc(np.zeros(4), np.arange(4.0)) is None


def test_permutation_p_is_small_for_a_separating_score():
    rng = np.random.default_rng(0)
    labels = np.array([True] * 30 + [False] * 30)
    scores = rng.normal(0.0, 1.0, 60)
    scores[:30] += 5.0
    assert s0.permutation_p(labels, scores, samples=100) < 0.05
    assert s0.permutation_p(np.ones(6, dtype=bool), np.arange(6.0)) is None


def test_subsample_respects_seed_and_bounds():
    pool = np.arange(100)
    first = s0._subsample(pool, 10, 11)
    second = s0._subsample(pool, 10, 11)
    assert np.array_equal(first, second)
    assert first.size == 10
    assert np.array_equal(s0._subsample(pool, 200, 11), pool)


# ---------------------------------------------------------------------------
# trace -> cache
# ---------------------------------------------------------------------------


def test_candidate_of_action_maps_focus_and_move():
    assert s0.candidate_of_action([0, 1, 0, 0]) == 0
    assert s0.candidate_of_action([5, 1, 0, 0]) == 5
    assert s0.candidate_of_action([2, 1, 1, 0]) == 11
    assert s0.candidate_of_action([9, 1, 0, 0]) == -1
    assert s0.candidate_of_action([0, 1, 2, 0]) == -1
    assert s0.candidate_of_action(None) == -1
    assert s0.candidate_of_action("nope") == -1


def test_split_by_group_holds_every_nth_sorted_episode():
    groups = [f"g{index:02d}" for index in range(9)]
    train, validation = s0.split_by_group(groups)
    assert groups[3] in set(np.array(groups)[validation])
    assert groups[7] in set(np.array(groups)[validation])
    assert validation.size == 2
    assert train.size + validation.size == len(groups)
    assert set(train) & set(validation) == set()
    # Order-independent: shuffling the rows must not change which episodes are held.
    shuffled_train, shuffled_validation = s0.split_by_group(list(reversed(groups)))
    assert set(groups[index] for index in validation) == set(
        list(reversed(groups))[index] for index in shuffled_validation
    )


def _build_two_corpora(tmp_path, episodes=8, rows_per_episode=4):
    rollouts = tmp_path / "rollouts"
    teacher = tmp_path / "teacher"
    for episode in range(episodes):
        rows = []
        for decision in range(rows_per_episode):
            frame = decision * 100
            terminal = decision == rows_per_episode - 1
            rows.append(
                _row(
                    frame,
                    100,
                    move=decision % 9,
                    focus=(episode + decision) % 2,
                    executed_move=(decision + 1) % 9,
                    terminated=terminal,
                    remaining_bullet_offset=5.0 * decision,
                )
            )
        target = teacher if episode >= episodes - 2 else rollouts
        _write_trace(target / f"episode-{episode:03d}-decisions.jsonl", rows)
    return rollouts, teacher


def test_build_cache_writes_arrays_labels_and_summary(tmp_path):
    rollouts, teacher = _build_two_corpora(tmp_path)
    cache = tmp_path / "stage0" / "cache.npz"
    summary = s0.build_cache(
        [("rollouts", rollouts), ("teacher", teacher)],
        cache,
        progress=lambda *_args, **_kwargs: None,
    )
    assert cache.exists()
    assert (tmp_path / "stage0" / "cache.summary.json").exists()

    data = s0.load_cache(cache)
    for name in ("bullets", "meta", "grid", "clearance", "teacher", "executed", "group", "remaining", "split"):
        assert name in data
    states = data["bullets"].shape[0]
    assert states == summary["decisions"] == 8 * 4
    assert data["bullets"].shape[1:] == (entity_obs.K_BULLETS, entity_obs.BULLET_FEATURES)
    assert data["clearance"].shape == (states, len(s0.HORIZONS), 18)
    assert data["grid"].shape[1:] == (entity_obs.GRID_CHANNELS, entity_obs.GRID_ROWS, entity_obs.GRID_COLS)
    assert data["teacher"].dtype == np.int16
    assert data["clearance"].dtype == np.float16

    # Remaining frames: rows are 0/100/200/300 with a 100-frame final step -> end 400.
    episode_remaining = data["remaining"][:4].tolist()
    assert episode_remaining == [400, 300, 200, 100]
    # Every row here keeps its geometric labels: the shortest remaining span (100
    # frames) is still longer than the 40-frame horizon, so the forced-unsafe rule
    # does not fire.  That rule has its own test below.
    assert set(data["split"].tolist()) <= {0, 1}
    assert data["split"].sum() > 0
    assert summary["episodes"] == 8
    assert summary["validation_episodes"] == 2
    assert summary["corpora"]["rollouts"]["kept"] == 6 * 4
    assert summary["corpora"]["teacher"]["kept"] == 2 * 4
    assert summary["valid_fraction"] == pytest.approx(1.0)

    # The deterministic hold-out must round-trip through the cache.
    held = s0.held_out_groups(data["group_names"])
    names_by_row = data["group_names"][data["group"].astype(np.int64)]
    expected = np.array([str(name) in held for name in names_by_row.tolist()])
    assert np.array_equal(expected, data["split"].astype(bool))


def test_death_before_horizon_forces_every_candidate_unsafe(tmp_path):
    rollouts = tmp_path / "rollouts"
    rows = [
        _row(0, 10, move=0),
        _row(10, 25, move=1),
        _row(35, 0, move=2, terminated=True),
    ]
    _write_trace(rollouts / "episode-001-decisions.jsonl", rows)
    cache = tmp_path / "cache.npz"
    s0.build_cache([("rollouts", rollouts)], cache, progress=lambda *_a, **_k: None)
    data = s0.load_cache(cache)
    # end frame 35 -> remaining 35/25/0; the 40-frame horizon is forced unsafe for
    # every row, the shorter horizons only for the terminal row.
    assert data["remaining"].tolist() == [35, 25, 0]
    clearance = data["clearance"].astype(np.float64)
    assert np.all(clearance[2] == 0.0)
    assert np.all(clearance[:, 2, :] == 0.0)  # H=40: death inside the horizon
    assert not np.all(clearance[0, 0, :] == 0.0)  # H=10: geometry kept
    assert not np.all(clearance[1, 1, :] == 0.0)  # H=20: geometry kept
    assert np.all(clearance[1, 2, :] == 0.0)


def test_build_cache_counts_dropped_rows(tmp_path):
    rollouts = tmp_path / "rollouts"
    good = _row(0, 10)
    bad = _row(10, 10, snapshot=_snapshot(complete=False))
    worse = _row(20, 10, snapshot=_snapshot(schema="other/1"))
    _write_trace(rollouts / "episode-001-decisions.jsonl", [good, bad, worse])
    cache = tmp_path / "cache.npz"
    summary = s0.build_cache([("rollouts", rollouts)], cache, progress=lambda *_a, **_k: None)
    assert summary["decisions"] == 1
    assert summary["parsed_rows"] == 3
    assert summary["dropped"] == {"incomplete": 1, "schema": 1}
    assert summary["valid_fraction"] == pytest.approx(1 / 3, abs=1e-4)


def test_build_cache_caps_rows_for_smoke_runs(tmp_path):
    rollouts, _teacher = _build_two_corpora(tmp_path)
    cache = tmp_path / "cache.npz"
    summary = s0.build_cache(
        [("rollouts", rollouts)], cache, max_rows_per_group=1, progress=lambda *_a, **_k: None
    )
    assert summary["decisions"] == 6


def test_build_cache_rejects_empty_corpus(tmp_path):
    rollouts = tmp_path / "rollouts"
    _write_trace(rollouts / "episode-001-decisions.jsonl", [_row(0, 10, snapshot=_snapshot(complete=False))])
    with pytest.raises(ValueError):
        s0.build_cache([("rollouts", rollouts)], tmp_path / "cache.npz", progress=lambda *_a, **_k: None)
    with pytest.raises(ValueError):
        s0.build_cache([("rollouts", tmp_path / "missing")], tmp_path / "cache.npz", progress=lambda *_a, **_k: None)


def test_trace_files_requires_the_expected_name(tmp_path):
    root = tmp_path / "rollouts"
    root.mkdir()
    (root / "notes.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        s0.trace_files(root)


# ---------------------------------------------------------------------------
# models: smoke tests on a tiny cache (CPU, one epoch)
# ---------------------------------------------------------------------------


def _tiny_run(tmp_path):
    rollouts, teacher = _build_two_corpora(tmp_path)
    cache = tmp_path / "cache.npz"
    s0.build_cache([("rollouts", rollouts), ("teacher", teacher)], cache, progress=lambda *_a, **_k: None)
    data = s0.load_cache(cache)
    train_index, validation_index = s0.split_by_group(
        [str(data["group_names"][int(group)]) for group in data["group"].tolist()]
    )
    out = tmp_path / "out"
    value = s0.train_value(
        data,
        train_index,
        validation_index,
        out,
        epochs=1,
        batch_size=4,
        device="cpu",
        hidden=32,
        progress=lambda *_a, **_k: None,
    )
    q = s0.train_q(
        data,
        train_index,
        validation_index,
        out,
        epochs=1,
        batch_size=4,
        device="cpu",
        hidden=32,
        progress=lambda *_a, **_k: None,
    )
    return data, value, q, out


def test_training_writes_reloadable_checkpoints(tmp_path):
    torch = pytest.importorskip("torch")
    data, value, q, out = _tiny_run(tmp_path)
    assert value["history"][0]["train_loss"] == pytest.approx(value["history"][0]["train_loss"])
    assert math.isfinite(value["best"]["validation_loss"])
    assert math.isfinite(q["best"]["validation_loss"])
    payload = torch.load(out / "value.pt", map_location="cpu", weights_only=False)
    assert payload["schema"] == s0.SCHEMA
    assert payload["kind"] == "value"
    assert payload["survival_frames"] == list(s0.SURVIVAL_FRAMES)
    q_payload = torch.load(out / "q.pt", map_location="cpu", weights_only=False)
    assert q_payload["kind"] == "candidate_q"
    assert q_payload["horizons"] == list(s0.HORIZONS)


def test_evaluate_returns_the_pre_registered_fields(tmp_path):
    pytest.importorskip("torch")
    data, value, q, _out = _tiny_run(tmp_path)
    metrics = s0.evaluate(
        data, value["checkpoint"], q["checkpoint"], device="cpu", progress=lambda *_a, **_k: None
    )
    assert metrics["summary"]["validation_states"] == 8
    assert metrics["summary"]["validation_episodes"] == 2
    assert metrics["summary"]["validation_teacher_states"] == 4
    assert set(metrics["q"]["metrics"]) == {"10", "20", "40"}
    assert "spearman_per_episode_mean" in metrics["value"]
    assert "survive_60_auc" in metrics["value"]
    assert "q_argmax_vs_teacher" in metrics["agreement"]
    assert "auc" in metrics["death_risk"]
    assert "executed_clearance_vs_death60_auc" in metrics["label_validity"]


def test_predictions_are_permutation_safe(tmp_path):
    """The held-out index order must not change per-state predictions."""
    torch = pytest.importorskip("torch")
    data, value, q, _out = _tiny_run(tmp_path)
    indices = np.flatnonzero(data["split"] == 1)
    straight, _logits = s0._predict_value(value["checkpoint"], data, indices, "cpu")
    shuffled, _logits2 = s0._predict_value(value["checkpoint"], data, indices[::-1], "cpu")
    assert np.allclose(straight, shuffled, atol=1e-6)
    straight_q = s0._predict_q(q["checkpoint"], data, indices, "cpu")
    shuffled_q = s0._predict_q(q["checkpoint"], data, indices[::-1], "cpu")
    assert np.allclose(straight_q, shuffled_q, atol=1e-6)


def test_checkpoint_kind_is_enforced(tmp_path):
    pytest.importorskip("torch")
    data, value, q, _out = _tiny_run(tmp_path)
    with pytest.raises(ValueError):
        s0._predict_q(value["checkpoint"], data, np.flatnonzero(data["split"] == 1), "cpu")
    with pytest.raises(ValueError):
        s0._predict_value(q["checkpoint"], data, np.flatnonzero(data["split"] == 1), "cpu")


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def _metrics(**overrides):
    metrics = {
        "summary": {"validation_episodes": 20, "validation_states": 1000},
        "value": {"spearman_per_episode_mean": 0.71},
        "q": {"metrics": {"10": {"safe_auc": 0.81}, "20": {"safe_auc": 0.83}, "40": {"safe_auc": 0.79}}},
        "agreement": {"q_argmax_vs_teacher": 0.56},
        "death_risk": {"auc": 0.78, "p_value": 0.001},
        "label_validity": {"executed_clearance_vs_death60_auc": 0.62},
    }
    for path, value in overrides.items():
        section, key = path.split("__")
        metrics[section][key] = value
    return metrics


def _data_summary(**overrides):
    summary = {
        "valid_fraction": 0.97,
        "corpora": {"rollouts": {"kept": 900}, "teacher": {"kept": 100}},
    }
    summary.update(overrides)
    return summary


def test_verdict_passes_when_every_gate_passes():
    result = s0.verdict(_metrics(), _data_summary())
    assert result["passed"] is True
    assert result["verdict"] == "pass"
    assert result["failed"] == []


@pytest.mark.parametrize(
    "metrics, data, failed",
    [
        (_metrics(value__spearman_per_episode_mean=0.59), _data_summary(), "S0-1_value_spearman"),
        (_metrics(value__spearman_per_episode_mean=None), _data_summary(), "S0-1_value_spearman"),
        (
            _metrics(q__metrics={"10": {"safe_auc": 0.81}, "20": {"safe_auc": 0.74}, "40": {"safe_auc": 0.79}}),
            _data_summary(),
            "S0-2_q_safe_auc",
        ),
        (_metrics(agreement__q_argmax_vs_teacher=0.49), _data_summary(), "S0-3_teacher_agreement"),
        (_metrics(agreement__q_argmax_vs_teacher=None), _data_summary(), "S0-3_teacher_agreement"),
        (_metrics(death_risk__auc=0.69), _data_summary(), "S0-4_death_ranking"),
        (_metrics(death_risk__p_value=0.02), _data_summary(), "S0-4_death_ranking"),
        (_metrics(), _data_summary(valid_fraction=0.89), "S0-0_data_integrity"),
        (_metrics(), _data_summary(corpora={"rollouts": {"kept": 900}}), "S0-0_data_integrity"),
        (
            _metrics(summary__validation_episodes=9),
            _data_summary(),
            "S0-0_data_integrity",
        ),
    ],
)
def test_verdict_fails_on_each_pre_registered_gate(metrics, data, failed):
    result = s0.verdict(metrics, data)
    assert result["passed"] is False
    assert failed in result["failed"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_root_specs_accepts_labels_and_rejects_duplicates():
    parsed = s0.parse_root_specs([r"rollouts=C:\data\a", r"teacher=C:\data\b"])
    assert parsed == [("rollouts", r"C:\data\a"), ("teacher", r"C:\data\b")]
    bare = s0.parse_root_specs([r"C:\data\a"])
    assert bare == [("root0", r"C:\data\a")]
    with pytest.raises(ValueError):
        s0.parse_root_specs(["a=one", "a=two"])
    with pytest.raises(ValueError):
        s0.parse_root_specs(["=missing-name"])


def test_cli_build_then_train_smoke(tmp_path):
    pytest.importorskip("torch")
    rollouts, teacher = _build_two_corpora(tmp_path)
    cache = tmp_path / "cache.npz"
    assert (
        s0.main(
            [
                "--build",
                "--root",
                f"rollouts={rollouts}",
                "--root",
                f"teacher={teacher}",
                "--cache",
                str(cache),
            ]
        )
        == 0
    )
    out = tmp_path / "out"
    assert (
        s0.main(
            [
                "--train",
                "--cache",
                str(cache),
                "--out",
                str(out),
                "--device",
                "cpu",
                "--value-epochs",
                "1",
                "--q-epochs",
                "1",
                "--batch-size",
                "4",
                "--hidden",
                "32",
            ]
        )
        == 0
    )
    report = json.loads((out / "stage0-report.json").read_text(encoding="utf-8"))
    assert report["schema"] == s0.SCHEMA
    assert report["gates"]["verdict"] in ("pass", "fail")
    assert set(report["gates"]["gates"]) == {
        "S0-0_data_integrity",
        "S0-1_value_spearman",
        "S0-2_q_safe_auc",
        "S0-3_teacher_agreement",
        "S0-4_death_ranking",
    }
    assert report["data"]["validation_states"] == 8


def test_train_rejects_a_cache_whose_split_was_tampered_with(tmp_path):
    pytest.importorskip("torch")
    rollouts, teacher = _build_two_corpora(tmp_path)
    cache = tmp_path / "cache.npz"
    s0.main(["--build", "--root", f"rollouts={rollouts}", "--root", f"teacher={teacher}", "--cache", str(cache)])
    data = s0.load_cache(cache)
    data["split"] = np.zeros_like(data["split"])
    np.savez_compressed(cache, **data)
    with pytest.raises(ValueError):
        s0.main(["--train", "--cache", str(cache), "--out", str(tmp_path / "out"), "--device", "cpu"])
