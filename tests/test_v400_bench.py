"""Contract tests only: synthetic records below are NOT game results."""
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
import pytest
from touhou_rl.v400.bench.judge import PixelObs, policy_view, judge
from touhou_rl.v400.gym import env as gym_env
from touhou_rl.v400.bench.suites import (DEV_SEEDS, CONFIRM_SEEDS, STAGE_NAMES, SUITES,
                                         stage_name, stage_ordinal)
from touhou_rl.v400.bench.stats import bootstrap, paired
from touhou_rl.v400.bench.results import load_results, row_stage, validate, summarize

#: Alias kept so the vocabulary test below reads unambiguously.
stage_names = STAGE_NAMES


def test_benchmark_registered_suites_and_disjoint_seeds():
    assert set(SUITES) >= {"S1N", "S1L", "PSN", "PSL", "RUNN"}
    assert not set(DEV_SEEDS) & set(CONFIRM_SEEDS)


def test_benchmark_policy_view_is_pixels_and_actions_only():
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    view = policy_view(pixels, (0, 3))
    assert isinstance(view, PixelObs)
    assert {field.name for field in fields(view)} == {"pixels", "prev_actions"}
    pixels[:] = 42
    assert not view.pixels.any()


def test_benchmark_judge_reads_real_env_interface_with_fake_shared_block(monkeypatch, tmp_path):
    class Client:
        def __init__(self, *args, **kwargs):
            pass

    class Block:
        def state(self):
            import struct
            values = np.zeros(25, dtype=np.uint32)
            for index, value in ((gym_env.F_DEATHS, 2.0),
                                 (gym_env.F_BOMBS, 1.0), (gym_env.F_LIVES, 3.0)):
                values[index] = struct.unpack("<I", struct.pack("<f", value))[0]
            values[gym_env.F_SCORE] = 12345
            return values

        def header(self):
            return {"frame": 500}

    monkeypatch.setattr(gym_env, "HarnessClient", Client)
    game = gym_env.Th08Env(tmp_path / "harness", tmp_path, shm_name="fake", privileged=False)
    with pytest.raises(RuntimeError, match="live shared block"):
        judge(game)
    game._shm = Block()
    verdict = judge(game)
    assert (verdict.deaths, verdict.bombs, verdict.lives, verdict.score, verdict.frame) == (2.0, 1.0, 3.0, 12345, 500)


def test_benchmark_episode_statistics_and_pairing():
    result = bootstrap([0, 1, 1], samples=100, seed=1)
    assert result["n"] == 3 and result["lo"] <= result["mean"] <= result["hi"]
    assert paired([3, 4], [1, 2], samples=100, seed=1)["mean"] == 2
    with pytest.raises(ValueError):
        paired([1], [2, 3])


def test_benchmark_rejects_unmeasured_and_empty_results():
    with pytest.raises(ValueError, match="incomplete"):
        validate({"score": 123})
    assert "No measured episodes" in summarize([])


def _synthetic_episode(**overrides):
    """Contract fixture only: satisfies the reader, never a real engine result."""
    row = {
        "ckpt": "none", "agent": "still", "suite": "S1N", "seed": 1000,
        "harness_commit": "f563df58c44037b44208c393a86540b434b0c83b",
        "privilege": "deploy_pixel", "frames": 1800, "stages": [1],
        "deaths": 1.0, "bombs_used": 0.0, "lives_end": 2.0, "score": 12345,
        "cleared": False, "end": "frame_cap", "wall_s": 2.5,
        "started_at": "2026-11-08T15:56:06.529046+00:00",
        "status": "ok", "measured": True,
    }
    row.update(overrides)
    return row


def test_bench_results_skip_explicit_void_records(tmp_path):
    audit = {
        "schema": "v400-wp7-bench-invalidation/1", "status": "invalid",
        "measured": False, "created_at": "2026-11-08T15:32:26.962501+00:00",
        "reason": "audit-only record", "notes": "excluded from the scoreboard",
        "invalidated_rows": [{"seed": 1002}],
        "invalidated_started_at": ["2026-11-08T15:57:00+00:00"],
    }
    records = (
        _synthetic_episode(),
        audit,
        _synthetic_episode(seed=1001, started_at="2026-11-08T15:56:10+00:00"),
        _synthetic_episode(seed=1002, started_at="2026-11-08T15:57:00+00:00"),
        _synthetic_episode(seed=1003, started_at="2026-11-08T15:58:00+00:00", measured=False),
        _synthetic_episode(seed=1004, started_at="2026-11-08T15:59:00+00:00", status="void"),
    )
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    rows = load_results(path)
    assert [row["seed"] for row in rows] == [1000, 1001]
    report = summarize(rows)
    assert report.startswith("# TH08-Bench measured results")
    # EXP-P0-077: a row that records no stage is labelled `unrecorded` in both
    # stage columns instead of being assumed to be any particular stage.
    assert "| still | S1N | unrecorded | unrecorded | unrecorded | unrecorded | 2 |" in report
    assert "no measured episodes" not in report


def test_bench_suites_declare_zero_based_stage_indexes():
    """EXP-P0-077: ``stages`` are engine ``stage_index`` values, not ordinals."""
    s1n = SUITES["S1N"]
    assert s1n.stage_indexes == (0,) and s1n.stages == (0,)
    assert s1n.stage_index == 0 and s1n.stage_name == "Stage1"
    assert s1n.stage_names == ("Stage1",)
    # A 1-based reading would put S1N on the real Stage 2; that must never be true.
    assert 1 not in s1n.stage_indexes
    assert stage_name(0) == "Stage1" and stage_name(1) == "Stage2"
    assert stage_ordinal(0) == "Stage 1" and stage_ordinal(1) == "Stage 2"
    assert stage_ordinal(7) == "Stage 8"
    for suite_name in ("PSN", "PSL", "RUNN"):
        assert SUITES[suite_name].stage_indexes == tuple(range(8))
        assert SUITES[suite_name].stage_index is None
        assert SUITES[suite_name].stage_name is None
    with pytest.raises(ValueError):
        stage_name(-1)
    with pytest.raises(ValueError):
        stage_name(9)


def test_bench_suite_stage_names_match_the_measured_reference_names():
    """The cap-side and suite-side stage names must stay one vocabulary."""
    from touhou_rl.v400.bench.human_reference import STAGE_NAMES as reference_names
    assert stage_names == reference_names


def test_bench_summarize_keeps_stages_in_separate_cells():
    """Rows of one suite that ran different stages are never averaged together."""
    stage1 = _synthetic_episode(seed=1000, stage_index=0, stage_name="Stage1", stage_observed=0,
                                frames=1800, score=5000)
    stage2 = _synthetic_episode(seed=1001, stage_index=1, stage_name="Stage2", stage_observed=1,
                                frames=1800, score=40000)
    observed_only = _synthetic_episode(seed=1002, stage_observed=1, frames=1800, score=40000)
    report = summarize([stage1, stage2, observed_only])
    assert "| still | S1N | 0 | Stage1 | unrecorded | declared | 1 |" in report
    assert "| still | S1N | 1 | Stage2 | unrecorded | declared | 1 |" in report
    # The pre-EXP-P0-077 row is grouped by the engine's own ``stage_observed``.
    assert "| still | S1N | 1 | Stage2 | unrecorded | stage_observed | 1 |" in report


def test_bench_results_validate_enforces_the_zero_based_stage_pairing():
    validate(_synthetic_episode(stage_index=0, stage_name="Stage1"))
    validate(_synthetic_episode(stage_index=1, stage_name="Stage2"))
    # A 1-based label under a 0-based index is exactly the EXP-P0-077 defect.
    with pytest.raises(ValueError, match="does not match 0-based stage_index"):
        validate(_synthetic_episode(stage_index=0, stage_name="Stage2"))
    with pytest.raises(ValueError, match="stage_name without a stage_index"):
        validate(_synthetic_episode(stage_name="Stage1"))
    with pytest.raises(ValueError, match="0-based engine stage index"):
        validate(_synthetic_episode(stage_index=9, stage_name="Extra"))
    with pytest.raises(ValueError, match="0-based engine stage index"):
        validate(_synthetic_episode(stage_index=True, stage_name="Stage2"))


def test_bench_row_stage_prefers_the_declared_index_and_never_guesses():
    assert row_stage({"stage_index": 0, "stage_name": "Stage1"}) == (0, "Stage1", "declared")
    assert row_stage({"stage_observed": 1}) == (1, "Stage2", "stage_observed")
    assert row_stage({"stage_index": 0, "stage_observed": 1}) == (0, "Stage1", "declared")
    assert row_stage({}) == (None, None, "unrecorded")
    assert row_stage({"stage_observed": 99}) == (None, None, "unrecorded")


def test_bench_results_still_reject_unmarked_incomplete_records(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"score": 123}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete measured episode record"):
        load_results(path)


def test_bench_results_skip_invalid_rows_in_repository_log():
    path = (Path(__file__).resolve().parents[1]
            / "Document/Update/Alpha/v4.0.0/bench/results.jsonl")
    if not path.is_file():
        pytest.skip("repository results.jsonl not present")
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    invalidated = {stamp for record in raw if record.get("status") == "invalid"
                   for stamp in record.get("invalidated_started_at") or ()}
    rows = load_results(path)  # must not raise on the historic status="invalid" audit line
    assert rows, "expected at least one measured episode"
    assert all(row["status"] == "ok" for row in rows)
    assert not [row for row in rows if row.get("started_at") in invalidated]
    assert summarize(rows).startswith("# TH08-Bench measured results")
