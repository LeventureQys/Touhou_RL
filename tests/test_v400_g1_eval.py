"""Unit tests for :mod:`tools.rl.v400_g1_eval` (no engine, no subprocess, no WSL).

The evaluator's job is arithmetic and bookkeeping, so the tests pin exactly the
places where a G1 report could silently drift: the cap derivation, the Wilson
interval, the strict ``stage_clear``/``max_frames`` vocabulary, and the two
denominators the task asks for ("completed runs" vs "all declared runs").
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.rl import v400_g1_eval as g1  # noqa: E402


# ---------------------------------------------------------------------------
# caps
# ---------------------------------------------------------------------------
def test_s1n_cap_is_ceil_of_one_and_a_half_stage1():
    human = g1.load_human_reference()
    cap, source = g1.s1n_cap(human)
    assert human[0]["gameplay_frames"] == 13258
    assert cap == 19887
    assert math.ceil(1.5 * 13258) == 19887
    assert "13258" in source


def test_psn_caps_use_measured_normal_frames_and_flag_the_missing_ones():
    human = g1.load_human_reference()
    caps = {row["stage"]: row for row in g1.psn_caps(human)}
    assert sorted(caps) == list(range(8))
    assert caps[0]["cap"] == 19887  # Stage1 13258
    assert caps[1]["cap"] == 27788  # Stage2 18525 -> ceil(27787.5)
    assert caps[2]["cap"] == 30014  # Stage3 20009 -> ceil(30013.5)
    assert caps[3]["cap"] == 41583  # Stage4A 27722
    assert caps[5]["cap"] == 38513  # Stage5 25675 -> ceil(38512.5)
    assert caps[7]["cap"] == 54605  # Stage6B 36403 -> ceil(54604.5)
    # no measured human play on normal for Stage4B / Stage6A
    assert caps[4]["missing_human_reference"] is True
    assert caps[6]["missing_human_reference"] is True
    assert caps[4]["cap"] == g1.MISSING_STAGE_CAP == 45000
    assert caps[6]["cap"] == 45000
    assert caps[4]["proxy_note"] and "41583" in caps[4]["proxy_note"]
    assert caps[4]["proxy_note"] and "Stage4A" in caps[4]["proxy_note"]
    assert caps[6]["proxy_note"] and "54605" in caps[6]["proxy_note"]
    assert caps[6]["proxy_note"] and "Stage6B" in caps[6]["proxy_note"]
    # the fallback is labelled as such, never passed off as a measured cap
    assert caps[4]["cap_source"].startswith("fallback")
    assert caps[6]["cap_source"].startswith("fallback")
    # every measured stage keeps its human frame count visible in the source text
    for stage in (0, 1, 2, 3, 5, 7):
        assert caps[stage]["missing_human_reference"] is False
        assert str(caps[stage]["human_frames"]) in caps[stage]["cap_source"]


def test_cap_from_human_never_rounds_down():
    assert g1.cap_from_human(1) == 2  # ceil(1.5)
    assert g1.cap_from_human(2) == 3
    assert g1.cap_from_human(13258) == 19887


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------
def test_wilson_interval_matches_the_closed_form():
    low, high = g1.wilson_interval(18, 20)
    assert low == pytest.approx(0.698966, abs=1e-6)
    assert high == pytest.approx(0.972134, abs=1e-6)
    assert low < 18 / 20 < high


def test_wilson_interval_edges():
    assert g1.wilson_interval(0, 0) is None
    low, high = g1.wilson_interval(20, 20)
    assert low < 1.0 and high == pytest.approx(1.0)
    low, high = g1.wilson_interval(0, 20)
    assert low == pytest.approx(0.0) and 0.0 < high < 0.2


# ---------------------------------------------------------------------------
# command construction
# ---------------------------------------------------------------------------
def test_teacher_command_is_the_task_cli(tmp_path):
    command = g1.teacher_command(
        stage=0,
        seed=1000,
        cap=19887,
        out_path=tmp_path / "x.jsonl",
        python="/root/v400-venv/bin/python",
    )
    assert command[0] == "/root/v400-venv/bin/python"
    assert command[1] == "tools/rl/v400_teacher.py"
    assert "--max-frames" in command and command[command.index("--max-frames") + 1] == "19887"
    assert "--stop-on-death" in command
    assert command[command.index("--stage") + 1] == "0"
    assert command[command.index("--difficulty") + 1] == "normal"
    assert command[command.index("--seed") + 1] == "1000"
    assert command[command.index("--mode") + 1] == "full"
    assert "--quiet" in command
    assert "--out" in command and command[command.index("--out") + 1].endswith("x.jsonl")
    # decisions must not become the binding stop before max-frames
    decisions = int(command[command.index("--decisions") + 1])
    assert decisions * 4 > 19887


def test_teacher_command_carries_the_worker_data_dir(tmp_path):
    command = g1.teacher_command(
        stage=3,
        seed=1,
        cap=41583,
        out_path=tmp_path / "x.jsonl",
        python="python3",
        data_dir="/root/v400-data/g1/inst3",
    )
    assert command[command.index("--data-dir") + 1] == "/root/v400-data/g1/inst3"
    # absent when not asked for, so the serial default is bit-identical to before
    plain = g1.teacher_command(stage=3, seed=1, cap=41583, out_path=tmp_path / "y.jsonl", python="python3")
    assert "--data-dir" not in plain


def _fake_args(**overrides):
    import argparse

    base = dict(data_dir="/root/v400-data/p1s1/game", data_dirs=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_parallel_jobs_refuse_a_shared_data_dir():
    assert g1._data_dirs(_fake_args(), 1) == ["/root/v400-data/p1s1/game"]
    with pytest.raises(SystemExit):
        g1._data_dirs(_fake_args(), 6)
    with pytest.raises(SystemExit):
        g1._data_dirs(_fake_args(data_dirs="/root/a,/root/b"), 6)
    six = ",".join(f"/root/v400-data/g1/inst{i}" for i in range(6))
    dirs = g1._data_dirs(_fake_args(data_dirs=six), 6)
    assert dirs == [f"/root/v400-data/g1/inst{i}" for i in range(6)]


# ---------------------------------------------------------------------------
# report parsing (alive != clear)
# ---------------------------------------------------------------------------
def _write_run(path: Path, *, stop_reason="stage_changed", stage_clear=True, deaths=0, frames=13676,
               boss_life_curve=None, enemy_count_curve=None):
    report = {
        "kind": "report",
        "status": "ok",
        "stop_reason": stop_reason,
        "stage": 0,
        "stage_clear": stage_clear,
        "frames_committed": frames,
        "survival_frames": frames,
        "deaths_observed": deaths,
        "decisions": 3,
        "bombs_spent": 0.0,
        "bomb_decisions": 0,
        "realtime_multiple": 0.5711,
        "stage_at_stop": 1 if stage_clear else 0,
        "engine": {"sha256": "f" * 64, "match": True},
        "boss_decisions": 0,
        "enemies_available": True,
        "consistency": {"agree_alive": 1},
    }
    lines = [json.dumps(report)]
    for index in range(3):
        boss_life = (boss_life_curve or [60000, 60000, 0])[index]
        count = (enemy_count_curve or [2, 2, 0])[index]
        rows = []
        if boss_life is not None:
            rows.append({"slot": 1, "boss": True, "life": boss_life, "x": 0.0, "y": 0.0})
        for extra in range(max(0, count - len(rows))):
            rows.append({"slot": 4 + extra, "boss": False, "life": 100, "x": 1.0, "y": 1.0})
        lines.append(
            json.dumps(
                {
                    "kind": "decision",
                    "decision": index,
                    "enemies": {
                        "available": True,
                        "count": count,
                        "boss_count": 1 if rows and rows[0]["boss"] else 0,
                        "boss_life": boss_life,
                        "rows": rows,
                    },
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_run_takes_the_game_verdicts(tmp_path):
    path = tmp_path / "run.jsonl"
    _write_run(path, stop_reason="stage_changed", stage_clear=True, deaths=0, frames=13676)
    parsed = g1.parse_run(path)
    assert parsed["stage_clear"] is True
    assert parsed["stop_reason"] == "stage_changed"
    assert parsed["frames"] == 13676
    assert parsed["deaths"] == 0
    assert parsed["engine_sha_match"] is True


def test_alive_at_frame_cap_is_not_a_clear(tmp_path):
    path = tmp_path / "run.jsonl"
    _write_run(path, stop_reason="max_frames", stage_clear=False, deaths=0, frames=19887)
    report = g1.read_report(path)
    assert report["stage_clear"] is False
    assert report["stop_reason"] == "max_frames"
    parsed = g1.parse_run(path)
    assert parsed["stage_clear"] is False


def test_boss_analysis_separates_present_from_killed(tmp_path):
    killed = tmp_path / "killed.jsonl"
    _write_run(killed, boss_life_curve=[60000, 60000, 0], enemy_count_curve=[2, 2, 0])
    out = g1.boss_analysis(killed)
    assert out["boss_present_decisions"] == 3
    assert out["boss_life_first"] == 60000
    assert out["boss_life_min"] == 0
    assert out["boss_zero_life_rows"] == 1
    assert out["boss_kill_evidence"] == "life_reached_zero"

    survived = tmp_path / "survived.jsonl"
    _write_run(survived, boss_life_curve=[60000, 50000, 40000], enemy_count_curve=[2, 2, 2])
    out = g1.boss_analysis(survived)
    assert out["boss_kill_evidence"] == "present_at_stop"
    assert out["boss_life_min"] == 40000

    absent = tmp_path / "absent.jsonl"
    _write_run(absent, boss_life_curve=[None, None, None], enemy_count_curve=[0, 0, 0])
    assert g1.boss_analysis(absent)["boss_kill_evidence"] == "not_observed"


# ---------------------------------------------------------------------------
# summaries: two denominators
# ---------------------------------------------------------------------------
def _s1n_record(seed, *, clear=False, deaths=0, frames=13676, wall=400.0, exit_code=0,
                stop_reason="stage_changed", completed=True):
    return {
        "case_id": f"s1n_seed{seed}",
        "suite": "s1n",
        "seed": seed,
        "completed": completed,
        "exit_code": exit_code,
        "stage_clear": clear,
        "deaths": deaths,
        "frames": frames,
        "wall_s": wall,
        "stop_reason": stop_reason,
    }


def test_summarize_s1n_reports_both_denominators_and_is_partial():
    records = [
        _s1n_record(1000, clear=True),
        _s1n_record(1001, clear=False, deaths=1, stop_reason="death", exit_code=0),
        _s1n_record(1002, completed=False, exit_code=-9),
    ]
    summary = g1.summarize_s1n(records, declared=20)
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["not_run"] == 17
    assert summary["clears"] == 1
    assert summary["clear_rate_completed"] == pytest.approx(0.5)
    assert summary["clear_rate_declared"] == pytest.approx(1 / 20)
    assert summary["partial"] is True
    assert summary["deaths_mean_per_episode"] == pytest.approx(0.5)
    assert summary["episodes_with_death"] == 1
    assert summary["exit_codes"] == {"0": 2}


def test_summarize_s1n_full_suite_is_not_partial():
    records = [_s1n_record(1000 + i, clear=(i < 18)) for i in range(20)]
    summary = g1.summarize_s1n(records, declared=20)
    assert summary["clears"] == 18
    assert summary["clear_rate_completed"] == pytest.approx(0.9)
    assert summary["wilson95_completed"][0] < 0.9 < summary["wilson95_completed"][1]
    assert summary["partial"] is False


def test_summarize_psn_keeps_unrun_stages_visible():
    caps = g1.psn_caps(g1.load_human_reference())
    records = [
        {"stage": 0, "completed": True, "stage_clear": True, "deaths": 0, "frames": 13676,
         "stop_reason": "stage_changed", "wall_s": 400.0, "exit_code": 0, "bombs_spent": 0.0,
         "boss": {"boss_present_decisions": 10, "boss_kill_evidence": "life_reached_zero",
                  "boss_life_first": 60000, "boss_life_min": 0}},
        {"stage": 1, "completed": True, "stage_clear": False, "deaths": 1, "frames": 9000,
         "stop_reason": "death", "wall_s": 300.0, "exit_code": 0, "bombs_spent": 1.0,
         "boss": {"boss_present_decisions": 5, "boss_kill_evidence": "present_at_stop",
                  "boss_life_first": 60000, "boss_life_min": 30000}},
    ]
    summary = g1.summarize_psn(records, caps)
    assert summary["stages_ran"] == 2
    assert summary["stages_declared"] == 8
    assert summary["deaths_mean_per_stage"] == pytest.approx(0.5)
    assert summary["stages_at_or_below_half"] == 1
    assert summary["clears"] == 1
    assert summary["partial"] is True
    stage4 = summary["stages"][4]
    assert stage4["ran"] is False
    assert stage4["cap"] == 45000


def test_markdown_tables_carry_every_stage():
    caps = g1.psn_caps(g1.load_human_reference())
    summary = g1.summarize_psn([], caps)
    text = g1.markdown_psn(summary)
    for stage in range(8):
        assert f"| {stage} |" in text
    s1n_text = g1.markdown_s1n([_s1n_record(1000, clear=True)], g1.summarize_s1n([_s1n_record(1000, clear=True)], 20))
    assert "| 1 | 1000 |" in s1n_text


# ---------------------------------------------------------------------------
# resume / append
# ---------------------------------------------------------------------------
def test_records_append_and_resume(tmp_path):
    path = tmp_path / "s1n_runs.jsonl"
    g1.append_record(path, _s1n_record(1000, clear=True))
    g1.append_record(path, _s1n_record(1001, completed=False, exit_code=-9))
    records = g1.load_records(path)
    assert [row["seed"] for row in records] == [1000, 1001]
    assert g1._already_done(records) == {"s1n_seed1000"}


def test_missing_reference_fails_closed(tmp_path):
    with pytest.raises(SystemExit):
        g1.s1n_cap({})
