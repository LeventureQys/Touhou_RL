"""Unit tests for the v1.0.3 experiment tools (``v103_sweep`` / ``check_a01``).

These run without the game and without training: every summary here is synthetic
JSON in a temporary directory.  What is pinned is the part that must never drift,
because a silently charitable verdict is worse than a red test:

* A-01 is judged on **every** epoch of ``stages.state.history`` (an earlier epoch
  can meet both thresholds while a later epoch wins on position alone), and only
  on ``val_danger_corr_perframe_median`` - a missing per-frame key is ``blocked``,
  never a fallback to the pooled ``danger_corr``;
* the primary run comes from a full v2 run, not from a screening run, and an
  explicit ``--primary`` wins over inference;
* A-03 is a ratio rule with a fixed status vocabulary, and ``r4_authorized`` stays
  ``false`` no matter what A-03 concludes;
* the sweep refuses a recipe that would shrink validation, and never rewrites a
  finished directory or duplicates its table row.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(tool: str):
    """Import ``tools/rl/<tool>.py`` as a module (``tools`` is not a package)."""
    module_name = f"v103_tools_{tool}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = ROOT / "tools" / "rl" / f"{tool}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SWEEP = _load("v103_sweep")
CHECK = _load("check_a01")


# --------------------------------------------------------------------------- #
# synthetic summaries
# --------------------------------------------------------------------------- #
def _state_row(epoch: int, pos=None, perf=None, global_corr=None, drop_perf: bool = False) -> dict:
    row = {
        "epoch": epoch,
        "train_loss": 0.10,
        "val_loss": 0.12,
        "seconds": 60.0,
        "val_frames": 7247,
        "val_player_pos_px": pos,
        "val_danger_corr": global_corr,
    }
    if not drop_perf:
        row["val_danger_corr_perframe_median"] = perf
    return row


def _policy_best(regret_0_15: float = 1.2) -> dict:
    return {
        "epoch": 8,
        "val_top1": 0.50,
        "val_regret_px": 2.0,
        "by_danger": [
            {"bucket": "0-15", "frames": 900, "top1": 0.4, "regret_px": regret_0_15},
            {"bucket": "15-30", "frames": 700, "top1": 0.5, "regret_px": 3.0},
        ],
    }


def _best_from_history(history: list[dict]) -> dict:
    """Simulate ``state.pt`` selection: the epoch with the lowest position error."""
    scored = [row for row in history if isinstance(row.get("val_player_pos_px"), (int, float))]
    best = min(scored, key=lambda row: row["val_player_pos_px"])
    return {
        "epoch": best["epoch"],
        "val_player_pos_px": best["val_player_pos_px"],
        "val_danger_corr_perframe_median": best.get("val_danger_corr_perframe_median"),
    }


def _summary(
    history: list[dict],
    *,
    arch: str = "v2",
    train_fraction: float = 1.0,
    max_train_batches: int = 0,
    with_policy: bool = True,
    policy_best: dict | None = None,
    policy_final: dict | None = None,
    train_frames: int = 65225,
) -> dict:
    stages = {"state": {"history": history, "best": _best_from_history(history)}}
    if with_policy:
        stages["policy"] = {
            "history": [{"epoch": 8, "val_top1": 0.5, "val_regret_px": 2.0, "by_danger": []}],
            "best": policy_best or _policy_best(),
            "final": policy_final or _policy_best(),
        }
    return {
        "schema": "vision-summary/2",
        "arch": arch,
        "note": "",
        "config": {
            "stage": "both" if with_policy else "state",
            "epochs_state": 8,
            "epochs_policy": 8,
            "batch_size": 96,
            "lr": 0.0003,
            "depth": 1,
            "max_train_batches": max_train_batches,
            "max_val_batches": 0,
        },
        "dataset": "synthetic",
        "frames": 72472,
        "train_frames": train_frames,
        "val_frames": 7247,
        "train_fraction": train_fraction,
        "stages": stages,
    }


def _write(directory: Path, name: str, payload: dict) -> str:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _verdict(tmp_path: Path, argv: list[str]) -> tuple[int, dict]:
    out = tmp_path / "acceptance.json"
    code = CHECK.main([*argv, "--out", str(out)])
    return code, json.loads(out.read_text(encoding="utf-8"))


def _passing_history() -> list[dict]:
    return [_state_row(1, pos=8.0, perf=0.82, global_corr=0.5)]


def _r3_point(pos, regret=1.2) -> dict:
    """One R3 run: state quality in ``pos``, policy quality in ``regret``."""
    return _summary([_state_row(1, pos=pos, perf=0.9)], policy_best=_policy_best(regret))


# --------------------------------------------------------------------------- #
# recipes file helpers (sweep)
# --------------------------------------------------------------------------- #
def _recipe(rid: str = "R1-a", arch: str = "v2", args: dict | None = None, note: str = "") -> dict:
    if args is None:
        args = {
            "stage": "both",
            "epochs_state": 8,
            "epochs_policy": 8,
            "batch_size": 96,
            "depth": 1,
            "train_fraction": 1.0,
            "max_train_batches": 0,
            "max_val_batches": 0,
        }
    return {"id": rid, "arch": arch, "note": note, "args": args}


def _recipe_file(tmp_path: Path, recipes: list[dict], name: str = "recipes.json", **overrides) -> str:
    payload = {
        "dataset": "third_party/th08/build/rl/datasets/synthetic",
        "runs_root": str(tmp_path / "runs"),
        "table": str(tmp_path / "实验表.md"),
        "recipes": recipes,
    }
    payload.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _table_rows(table_path: Path) -> list[str]:
    lines = table_path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.startswith("|") and "---" not in line]


# --------------------------------------------------------------------------- #
# A-01
# --------------------------------------------------------------------------- #
def test_a01_pass(tmp_path):
    path = _write(tmp_path, "exp-05/vision-summary.json", _summary(_passing_history()))
    code, doc = _verdict(tmp_path, ["--summary", path])
    assert doc["a01"]["status"] == "pass"
    assert doc["a01"]["epoch"] == 1
    assert doc["a01"]["pos_px"] == 8.0
    assert doc["a01"]["perframe_median"] == 0.82
    assert doc["a01"]["thresholds"] == {"pos_px": 10.0, "perframe_median": 0.8}
    assert doc["primary"]["checkpoint"] == "state-pass.pt"
    assert code == 0


def test_a01_fail_on_position(tmp_path):
    """``pos_px == 10.0`` is not ``< 10.0``: the threshold is strict."""
    history = [_state_row(1, pos=10.0, perf=0.9)]
    path = _write(tmp_path, "exp-05/vision-summary.json", _summary(history))
    code, doc = _verdict(tmp_path, ["--summary", path])
    assert doc["a01"]["status"] == "fail"
    assert doc["a01"]["epoch"] is None
    assert code == 1


def test_a01_fail_on_perframe_even_if_global_corr_high(tmp_path):
    """A pooled correlation of 0.95 must not rescue a per-frame median of 0.5."""
    history = [_state_row(1, pos=8.0, perf=0.5, global_corr=0.95)]
    path = _write(tmp_path, "exp-05/vision-summary.json", _summary(history))
    code, doc = _verdict(tmp_path, ["--summary", path])
    assert doc["a01"]["status"] == "fail"
    assert doc["a01"]["best_pos_px"] == 8.0
    assert code == 1


def test_a01_blocked_when_perframe_missing(tmp_path):
    """No per-frame key means blocked; the pooled ``danger_corr`` is not a substitute."""
    history = [_state_row(1, pos=8.0, perf=None, global_corr=0.95, drop_perf=True)]
    path = _write(tmp_path, "exp-05/vision-summary.json", _summary(history))
    code, doc = _verdict(tmp_path, ["--summary", path])
    assert doc["a01"]["status"] == "blocked"
    assert "val_danger_corr_perframe_median" in doc["a01"]["reason"]
    assert "danger_corr" in doc["a01"]["reason"]
    assert doc["a01"]["pos_px"] is None
    assert code == 1


def test_a01_pass_on_non_best_epoch(tmp_path):
    """Epoch 1 meets both bars, epoch 2 wins on position alone: still a pass."""
    history = [
        _state_row(1, pos=9.5, perf=0.85),
        _state_row(2, pos=8.0, perf=0.70),
    ]
    path = _write(tmp_path, "exp-05/vision-summary.json", _summary(history))
    code, doc = _verdict(tmp_path, ["--summary", path])
    assert doc["a01"]["status"] == "pass"
    assert doc["a01"]["epoch"] == 1
    assert doc["a01"]["pos_px"] == 9.5
    assert doc["primary"]["epoch"] == 1
    assert doc["primary"]["checkpoint"] == "state-pass.pt"
    assert doc["primary"]["pos_px"] == doc["a01"]["pos_px"]
    assert doc["primary"]["perframe_median"] == doc["a01"]["perframe_median"]
    assert code == 0


def test_a01_fail_reports_closest_epoch(tmp_path):
    history = [
        _state_row(1, pos=9.9, perf=0.50),
        _state_row(2, pos=15.0, perf=0.75),
    ]
    path = _write(tmp_path, "exp-05/vision-summary.json", _summary(history))
    code, doc = _verdict(tmp_path, ["--summary", path])
    assert doc["a01"]["status"] == "fail"
    assert doc["a01"]["best_pos_px"] == 9.9
    assert doc["a01"]["best_perframe_median"] == 0.50
    assert doc["primary"]["checkpoint"] == "state.pt"
    assert doc["primary"]["epoch"] == 1
    assert code == 1


# --------------------------------------------------------------------------- #
# primary run selection
# --------------------------------------------------------------------------- #
def test_primary_selection_prefers_lowest_pos_among_v2_full(tmp_path):
    full = _write(
        tmp_path,
        "exp-05/vision-summary.json",
        _summary([_state_row(1, pos=9.0, perf=0.85)]),
    )
    half = _write(
        tmp_path,
        "exp-06/vision-summary.json",
        _summary([_state_row(1, pos=5.0, perf=0.9)], train_fraction=0.5),
    )
    v1 = _write(
        tmp_path,
        "exp-07/vision-summary.json",
        _summary([_state_row(1, pos=3.0, perf=0.9)], arch="v1"),
    )
    code, doc = _verdict(tmp_path, ["--summary", full, "--summary", half, "--summary", v1])
    assert doc["primary"]["path"] == full
    assert doc["primary"]["arch"] == "v2"
    assert doc["a01"]["status"] == "pass"
    assert code == 0


def test_primary_excludes_screening_runs(tmp_path):
    """A cheap screening run must not become the run that A-01/A-02 judge."""
    screening = _write(
        tmp_path,
        "exp-01/vision-summary.json",
        _summary(
            [_state_row(1, pos=5.0, perf=0.9)],
            max_train_batches=160,
            with_policy=False,
        ),
    )
    full = _write(
        tmp_path,
        "exp-05/vision-summary.json",
        _summary([_state_row(1, pos=9.0, perf=0.85)]),
    )
    code, doc = _verdict(tmp_path, ["--summary", screening, "--summary", full])
    assert doc["primary"]["path"] == full

    # With only the screening run left there is no primary run at all.
    code, doc = _verdict(tmp_path, ["--summary", screening])
    assert doc["primary"]["path"] is None
    assert doc["a01"]["status"] == "blocked"
    assert doc["a02"]["status"] == "blocked"
    assert "no full v2 run" in doc["a01"]["reason"]
    assert "no full v2 run" in doc["a02"]["reason"]
    assert code == 1


def test_explicit_primary_overrides_inference(tmp_path):
    better = _write(
        tmp_path,
        "exp-04/vision-summary.json",
        _summary([_state_row(1, pos=4.0, perf=0.9)]),
    )
    chosen = _write(
        tmp_path,
        "exp-05/vision-summary.json",
        _summary([_state_row(1, pos=9.0, perf=0.85)]),
    )
    code, doc = _verdict(tmp_path, ["--summary", better, "--summary", chosen, "--primary", chosen])
    assert doc["primary"]["path"] == chosen
    assert doc["a01"]["pos_px"] == 9.0
    assert code == 0


def test_primary_without_stages_state_exit_2(tmp_path):
    path = _write(tmp_path, "broken/vision-summary.json", {"schema": "vision-summary/2", "arch": "v2"})
    out = tmp_path / "acceptance.json"
    code = CHECK.main(["--primary", path, "--out", str(out)])
    assert code == 2
    assert not out.exists()


# --------------------------------------------------------------------------- #
# A-02
# --------------------------------------------------------------------------- #
def test_a02_reads_bucket_from_best_not_final(tmp_path):
    summary = _summary(
        _passing_history(),
        policy_best=_policy_best(1.2),
        policy_final=_policy_best(9.9),
    )
    path = _write(tmp_path, "exp-05/vision-summary.json", summary)
    code, doc = _verdict(tmp_path, ["--summary", path])
    assert doc["a02"]["status"] == "pass"
    assert doc["a02"]["regret_px_bucket_0_15"] == 1.2
    assert doc["primary"]["regret_px_bucket_0_15"] == 1.2
    assert code == 0


def test_a02_blocked_when_primary_has_no_policy(tmp_path):
    summary = _summary(_passing_history(), with_policy=False)
    path = _write(tmp_path, "exp-05/vision-summary.json", summary)
    code, doc = _verdict(tmp_path, ["--primary", path])
    assert doc["a01"]["status"] == "pass"
    assert doc["a02"]["status"] == "blocked"
    assert "no policy stage" in doc["a02"]["reason"]
    assert doc["a02"]["regret_px_bucket_0_15"] is None
    assert code == 1


def test_a02_blocked_when_bucket_missing(tmp_path):
    best = _policy_best()
    best["by_danger"] = [{"bucket": "15-30", "frames": 700, "top1": 0.5, "regret_px": 3.0}]
    path = _write(tmp_path, "exp-05/vision-summary.json", _summary(_passing_history(), policy_best=best))
    code, doc = _verdict(tmp_path, ["--summary", path])
    assert doc["a02"]["status"] == "blocked"
    assert "0-15" in doc["a02"]["reason"]
    assert code == 1


# --------------------------------------------------------------------------- #
# A-03
# --------------------------------------------------------------------------- #
def _r3_argv(tmp_path: Path, points: dict, primary: str) -> list[str]:
    argv = ["--primary", primary]
    for fraction, summary in points.items():
        argv.extend(["--r3", f"{fraction:.2f}={_write(tmp_path, f'r3-{fraction:.2f}/vision-summary.json', summary)}"])
    return argv


def test_a03_plateau(tmp_path):
    primary = _write(tmp_path, "exp-05/vision-summary.json", _summary(_passing_history()))
    argv = _r3_argv(
        tmp_path,
        {0.25: _r3_point(18.0), 0.50: _r3_point(12.0, regret=4.0), 1.00: _r3_point(11.5, regret=3.9)},
        primary,
    )
    code, doc = _verdict(tmp_path, argv)
    assert doc["a01"]["status"] == "pass"
    assert doc["a03"]["status"] == "plateau"
    assert doc["a03"]["gain_pos"] is False
    assert doc["a03"]["gain_regret"] is False
    assert len(doc["a03"]["points"]) == 3
    assert doc["r4_authorized"] is False
    assert code == 0


def test_a03_has_gain_by_position(tmp_path):
    """``15.0 <= 0.85 * 20.0``: the 100% point is worth 3 more replays, not R4."""
    primary = _write(tmp_path, "exp-05/vision-summary.json", _summary(_passing_history()))
    argv = _r3_argv(
        tmp_path,
        {0.50: _r3_point(20.0, regret=4.0), 1.00: _r3_point(15.0, regret=4.0)},
        primary,
    )
    code, doc = _verdict(tmp_path, argv)
    assert doc["a03"]["status"] == "has_gain"
    assert doc["a03"]["gain_pos"] is True
    assert doc["r4_authorized"] is False
    assert "3 more replays" in doc["note"]
    assert code == 0


def test_a03_blocked_when_a01_fails(tmp_path):
    primary = _write(
        tmp_path,
        "exp-05/vision-summary.json",
        _summary([_state_row(1, pos=14.0, perf=0.6)]),
    )
    argv = _r3_argv(
        tmp_path,
        {0.50: _r3_point(20.0), 1.00: _r3_point(15.0)},
        primary,
    )
    code, doc = _verdict(tmp_path, argv)
    assert doc["a01"]["status"] == "fail"
    assert doc["a03"]["status"] == "blocked"
    assert "A-01 not met" in doc["a03"]["reason"]
    assert doc["r4_authorized"] is False
    assert code == 1


def test_a03_incomplete(tmp_path):
    primary = _write(tmp_path, "exp-05/vision-summary.json", _summary(_passing_history()))
    argv = _r3_argv(tmp_path, {0.25: _r3_point(22.0), 1.00: _r3_point(15.0)}, primary)
    code, doc = _verdict(tmp_path, argv)
    assert doc["a03"]["status"] == "incomplete"
    assert "0.50" in doc["a03"]["reason"]
    assert doc["a03"]["gain_pos"] is None
    assert code == 0


def test_a03_point_carries_frames_and_rule(tmp_path):
    primary = _write(tmp_path, "exp-05/vision-summary.json", _summary(_passing_history(), train_frames=65225))
    argv = _r3_argv(
        tmp_path,
        {0.50: _r3_point(20.0), 1.00: _r3_point(15.0)},
        primary,
    )
    code, doc = _verdict(tmp_path, argv)
    point = [row for row in doc["a03"]["points"] if row["fraction"] == 1.00][0]
    assert point["train_frames"] == 65225
    assert point["val_frames"] == 7247
    assert doc["a03"]["rule"] == "pos(1.00) <= 0.85*pos(0.50) or regret(1.00) <= 0.90*regret(0.50)"
    assert doc["inputs"]["r3"]["1.00"].endswith("vision-summary.json")
    assert code == 0


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #
def test_missing_file_exit_code_2(tmp_path):
    missing = str(tmp_path / "nope" / "vision-summary.json")
    out = tmp_path / "acceptance.json"
    assert CHECK.main(["--summary", missing, "--out", str(out)]) == 2
    assert not out.exists()

    existing = _write(tmp_path, "exp-05/vision-summary.json", _summary(_passing_history()))
    assert CHECK.main(["--primary", existing, "--r3", f"0.50={missing}", "--out", str(out)]) == 2
    assert not out.exists()


def test_broken_json_exit_code_2(tmp_path):
    path = tmp_path / "exp-05/vision-summary.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert CHECK.main(["--summary", str(path)]) == 2


def test_r4_authorized_is_false_without_out_file(tmp_path, capsys):
    path = _write(tmp_path, "exp-05/vision-summary.json", _summary(_passing_history()))
    code = CHECK.main(["--summary", path])
    assert code == 0
    printed = capsys.readouterr().out
    assert '"r4_authorized": false' in printed


# --------------------------------------------------------------------------- #
# sweep: validation
# --------------------------------------------------------------------------- #
def test_sweep_rejects_max_val_batches(tmp_path):
    args = {"stage": "state", "max_train_batches": 160, "max_val_batches": 20}
    recipes = _recipe_file(tmp_path, [_recipe(args=args)])
    with pytest.raises(SystemExit) as excinfo:
        SWEEP.main(["--recipes", recipes, "--dry-run"])
    assert "validation must always be full" in str(excinfo.value)


def test_sweep_rejects_unknown_recipe_arg(tmp_path):
    recipes = _recipe_file(tmp_path, [_recipe(args={"stage": "both", "width": 32})])
    with pytest.raises(SystemExit) as excinfo:
        SWEEP.main(["--recipes", recipes, "--dry-run"])
    assert "unknown recipe arg" in str(excinfo.value)
    assert "width" in str(excinfo.value)


def test_sweep_rejects_pilot_v1_runs_root(tmp_path):
    runs_root = tmp_path / "third_party" / "th08" / "build" / "rl" / "runs" / "pilot-v1"
    recipes = _recipe_file(tmp_path, [_recipe()], runs_root=str(runs_root))
    with pytest.raises(SystemExit) as excinfo:
        SWEEP.main(["--recipes", recipes, "--dry-run"])
    assert "pilot-v1" in str(excinfo.value)


def test_sweep_rejects_unknown_recipe_id(tmp_path):
    recipes = _recipe_file(tmp_path, [_recipe()])
    with pytest.raises(SystemExit) as excinfo:
        SWEEP.main(["--recipes", recipes, "--only", "R9-z", "--dry-run"])
    assert "unknown recipe id" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# sweep: dry run and real runs (training is stubbed)
# --------------------------------------------------------------------------- #
def test_sweep_dry_run_does_not_write(tmp_path, capsys):
    recipes = _recipe_file(tmp_path, [_recipe("R1-a"), _recipe("R1-b", args={"stage": "state", "max_train_batches": 160})])
    report = tmp_path / "sweep-report.json"
    code = SWEEP.main(["--recipes", recipes, "--dry-run", "--out-json", str(report)])
    assert code == 0
    assert not (tmp_path / "实验表.md").exists()
    assert not (tmp_path / "runs").exists()

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert [entry["id"] for entry in payload["entries"]] == ["R1-a", "R1-b"]
    assert [entry["out_dir"] for entry in payload["entries"]] == [
        (tmp_path / "runs" / "v103-exp-01").as_posix(),
        (tmp_path / "runs" / "v103-exp-02").as_posix(),
    ]
    for entry in payload["entries"]:
        assert entry["status"] == "dry-run"
        assert entry["metrics"] == {}
        # Validation is always full, even when a recipe forgets to say so.
        assert "--max-val-batches 0" in entry["command"]

    printed = capsys.readouterr().out
    assert "touhou_rl.vision_train" in printed
    assert "--max-train-batches 160" in printed
    assert "--train-fraction 1.0" in printed


def _fake_execute(summary_for: dict[str, dict] | None = None, fail_dirs: tuple[str, ...] = ()):
    """Stub for ``subprocess.run``: writes a summary, or fails, per directory name."""
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> int:
        calls.append(list(cmd))
        out_dir = Path(cmd[cmd.index("--out") + 1])
        if out_dir.name in fail_dirs:
            return 1
        payload = (summary_for or {}).get(out_dir.name)
        if payload is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "vision-summary.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        return 0

    return runner, calls


def test_sweep_appends_one_row_per_recipe(tmp_path, monkeypatch):
    recipes = _recipe_file(tmp_path, [_recipe("R1-a"), _recipe("R1-b")])
    runner, calls = _fake_execute(
        {
            "v103-exp-01": _summary([_state_row(1, pos=8.0, perf=0.82)]),
            "v103-exp-02": _summary([_state_row(1, pos=12.5, perf=0.70)], with_policy=False),
        }
    )
    monkeypatch.setattr(SWEEP, "_execute", runner)
    report = tmp_path / "sweep-report.json"
    assert SWEEP.main(["--recipes", recipes, "--out-json", str(report)]) == 0
    assert len(calls) == 2

    rows = _table_rows(tmp_path / "实验表.md")
    assert rows[0] == SWEEP.TABLE_HEADER
    assert len(rows) == 3
    assert "| 01 | R1-a | v2 | 65225 | 0 | 8.00 | 0.820 | 1.20 |" in rows[1]
    assert "| 02 | R1-b | v2 | 65225 | 0 | 12.50 | 0.700 | n/a |" in rows[2]
    assert rows[1].endswith("v103-exp-01/vision-summary.json |")

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["entries"][0]["metrics"]["pos_px"] == 8.0
    assert payload["entries"][1]["metrics"]["regret_0_15"] is None


def test_sweep_rerun_skips_finished_and_keeps_table(tmp_path, monkeypatch):
    """Re-entry: a finished directory is not retrained and its row is not duplicated."""
    recipes = _recipe_file(tmp_path, [_recipe("R1-a"), _recipe("R1-b")])
    payloads = {
        "v103-exp-01": _summary([_state_row(1, pos=8.0, perf=0.82)]),
        "v103-exp-02": _summary([_state_row(1, pos=9.0, perf=0.85)]),
    }
    report = tmp_path / "sweep-report.json"
    runner, calls = _fake_execute(payloads)
    monkeypatch.setattr(SWEEP, "_execute", runner)
    assert SWEEP.main(["--recipes", recipes, "--out-json", str(report)]) == 0
    assert len(calls) == 2

    runner_two, calls_two = _fake_execute(payloads)
    monkeypatch.setattr(SWEEP, "_execute", runner_two)
    assert SWEEP.main(["--recipes", recipes, "--out-json", str(report)]) == 0
    assert calls_two == []
    assert len(_table_rows(tmp_path / "实验表.md")) == 3
    assert len(json.loads(report.read_text(encoding="utf-8"))["entries"]) == 2


def test_sweep_continues_after_failure(tmp_path, monkeypatch):
    recipes = _recipe_file(tmp_path, [_recipe("R1-a"), _recipe("R1-b")])
    runner, calls = _fake_execute(
        {"v103-exp-02": _summary([_state_row(1, pos=9.0, perf=0.85)])},
        fail_dirs=("v103-exp-01",),
    )
    monkeypatch.setattr(SWEEP, "_execute", runner)
    report = tmp_path / "sweep-report.json"
    assert SWEEP.main(["--recipes", recipes, "--out-json", str(report)]) == 1

    entries = json.loads(report.read_text(encoding="utf-8"))["entries"]
    assert [entry["status"] for entry in entries] == ["failed", "ok"]
    assert entries[0]["returncode"] == 1
    assert entries[1]["returncode"] == 0
    rows = _table_rows(tmp_path / "实验表.md")
    assert len(rows) == 3
    assert "n/a" in rows[1]


def test_sweep_numbering_continues_across_recipe_files(tmp_path, monkeypatch):
    """A second recipe file continues the numbering instead of overwriting batch one."""
    first = _recipe_file(tmp_path, [_recipe("R1-a")], name="recipes.json")
    payloads = {
        "v103-exp-01": _summary([_state_row(1, pos=8.0, perf=0.82)]),
        "v103-exp-02": _summary([_state_row(1, pos=7.0, perf=0.86)]),
    }
    runner, _ = _fake_execute(payloads)
    monkeypatch.setattr(SWEEP, "_execute", runner)
    report = tmp_path / "sweep-report.json"
    assert SWEEP.main(["--recipes", first, "--out-json", str(report)]) == 0

    second = _recipe_file(tmp_path, [_recipe("R2-a", args={"stage": "state", "max_train_batches": 0})], name="recipes-2.json")
    runner_two, calls_two = _fake_execute(payloads)
    monkeypatch.setattr(SWEEP, "_execute", runner_two)
    assert SWEEP.main(["--recipes", second, "--out-json", str(report)]) == 0
    assert calls_two[0][calls_two[0].index("--out") + 1].endswith("v103-exp-02")

    entries = json.loads(report.read_text(encoding="utf-8"))["entries"]
    assert [entry["id"] for entry in entries] == ["R1-a", "R2-a"]
    assert len(_table_rows(tmp_path / "实验表.md")) == 3
