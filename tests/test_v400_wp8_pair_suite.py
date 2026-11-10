"""Tests for the three-replay WP8 paired-pixel suite (``v400_wp8_pair_suite``).

Everything here is offline: the suite's arithmetic, its fail-closed guards and
its montage layout are exercised without starting the game.  The synthetic
fixtures below are *test inputs only* -- they are never written to an evidence
directory and they never stand in for a real capture.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "rl"))

import v400_wp8_pair_suite as suite  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, "RGBA").save(path)


def _windows_manifest(root: Path, frames: list[int], *, bullets: int = 7) -> Path:
    rows = []
    for index, frame in enumerate(frames):
        _png(root / "frames" / f"{frame:05d}.png", np.zeros((480, 640, 4), dtype=np.uint8) + index)
        rows.append(
            {
                "logicFrame": frame,
                "image": f"frames/{frame:05d}.png",
                "state_sha256": "0" * 64,
                "state_evidence": {
                    "active_bullets": bullets,
                    "player_x": 100.0 + index,
                    "player_y": 200.0,
                    "player_state": 0,
                    "score": 1000 + index,
                    "lives": 2.0,
                    "power": 1.5,
                    "stage": 3,
                },
            }
        )
    path = root / "windows-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"source": "windows-fence", "replay_sha256": "a" * 64, "frames": rows}),
        encoding="utf-8",
    )
    return path


def _linux_manifest(root: Path, frames: list[int], *, shift: int = 0) -> Path:
    rows = []
    for index, frame in enumerate(frames):
        _png(root / "frames" / f"{frame:05d}.png", np.zeros((480, 640, 4), dtype=np.uint8) + index)
        fields = {
            "f0": 500 + index,
            "f1": 3,
            "f2": frame,
            "f3": frame,
            "f4": 1000 + index + shift,
            "f5": 0,
            "f6": float_bits(0.0),
            "f7": float_bits(2.0),
            "f8": float_bits(3.0),
            "f9": float_bits(1.5),
            "f10": float_bits(100.0 + index),
            "f11": float_bits(200.0),
            "f12": float_bits(0.0),
            "f13": 0,
            "f14": 1234,
            "f15": 0,
            "f16": 0,
            "f17": 0,
            "f18": 0,
            "f19": 0,
            "f20": 0,
            "f21": 0,
            "f22": 6,
            "f23": 6,
            "f24": 1,
        }
        rows.append(
            {
                "logicFrame": frame,
                "image": f"frames/{frame:05d}.png",
                "state_sha256": "1" * 64,
                "state_evidence": {"fields": fields},
            }
        )
    path = root / "linux-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"source": "linux-harness", "replay_sha256": "a" * 64, "frames": rows}),
        encoding="utf-8",
    )
    return path


def _report(root: Path, frames: list[int], *, playfield: float, hud: float, aux: float) -> Path:
    rows = []
    for frame in frames:
        rows.append(
            {
                "logicFrame": frame,
                "regions": {
                    "playfield": {
                        "mean_absolute_difference": playfield,
                        "different_pixel_ratio": 0.25,
                        "max_channel_difference": 9,
                    },
                    "hud": {
                        "mean_absolute_difference": hud,
                        "different_pixel_ratio": 0.10,
                        "max_channel_difference": 251,
                    },
                },
                "aux_regions": {
                    suite.AUX_KEY: {
                        "mean_absolute_difference": aux,
                        "different_pixel_ratio": 0.05,
                        "max_channel_difference": 12,
                    }
                },
            }
        )
    path = root / "pixel-report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "v400-wp8-paired-pixel-audit/2",
                "replay_sha256": "a" * 64,
                "state_pairing": "same-logic-frame",
                "state_consistency": "not-asserted",
                "pairs": len(rows),
                "aggregate": {
                    "playfield": {
                        "mean_of_frame_mean_absolute_difference": playfield,
                        "min_frame_mean_absolute_difference": playfield,
                        "max_frame_mean_absolute_difference": playfield,
                        "mean_different_pixel_ratio": 0.25,
                        "max_frame_different_pixel_ratio": 0.25,
                        "max_channel_difference": 9,
                    },
                    "hud": {
                        "mean_of_frame_mean_absolute_difference": hud,
                        "min_frame_mean_absolute_difference": hud,
                        "max_frame_mean_absolute_difference": hud,
                        "mean_different_pixel_ratio": 0.10,
                        "max_frame_different_pixel_ratio": 0.10,
                        "max_channel_difference": 251,
                    },
                    suite.AUX_KEY: {
                        "mean_of_frame_mean_absolute_difference": aux,
                        "min_frame_mean_absolute_difference": aux,
                        "max_frame_mean_absolute_difference": aux,
                        "mean_different_pixel_ratio": 0.05,
                        "max_frame_different_pixel_ratio": 0.05,
                        "max_channel_difference": 12,
                    },
                },
                "frames": rows,
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# window / manifest guards
# ---------------------------------------------------------------------------
def test_windows_logic_frames_accepts_a_consecutive_window(tmp_path):
    manifest = _windows_manifest(tmp_path, list(range(1000, 1010)))
    assert suite.windows_logic_frames(manifest) == list(range(1000, 1010))


def test_windows_logic_frames_rejects_a_gap(tmp_path):
    manifest = _windows_manifest(tmp_path, [1000, 1001, 1003])
    with pytest.raises(ValueError, match="not consecutive"):
        suite.windows_logic_frames(manifest)


def test_windows_logic_frames_rejects_duplicates(tmp_path):
    manifest = _windows_manifest(tmp_path, [1000, 1001, 1001])
    with pytest.raises(ValueError, match="not strictly increasing"):
        suite.windows_logic_frames(manifest)


def test_windows_logic_frames_rejects_foreign_source(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"source": "linux-harness", "frames": [{"logicFrame": 1}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a windows-fence manifest"):
        suite.windows_logic_frames(path)


def test_require_frame_count_is_exact(tmp_path):
    suite.require_frame_count([1, 2, 3], 3)
    with pytest.raises(ValueError, match="require 4"):
        suite.require_frame_count([1, 2, 3], 4)


# ---------------------------------------------------------------------------
# cross-side field checks
# ---------------------------------------------------------------------------
def test_cross_check_reports_every_field_equal(tmp_path):
    frames = list(range(10, 15))
    win = _windows_manifest(tmp_path / "w", frames)
    lin = _linux_manifest(tmp_path / "l", frames)
    result = suite.state_cross_check(win, lin)
    assert result["frames"] == 5
    assert result["mismatchCount"] == 0
    for name, entry in result["fields"].items():
        assert entry == {"equal": 5, "total": 5}, name
    assert "not a full state-equality proof" in result["note"].lower()


def test_cross_check_detects_a_score_divergence(tmp_path):
    frames = list(range(10, 15))
    win = _windows_manifest(tmp_path / "w", frames)
    lin = _linux_manifest(tmp_path / "l", frames, shift=1)
    result = suite.state_cross_check(win, lin)
    assert result["mismatchCount"] == 5
    assert result["fields"]["score"]["equal"] == 0
    assert result["fields"]["player_x"]["equal"] == 5
    assert result["mismatches"][0]["failed"] == ["score"]


def test_cross_check_rejects_different_frame_sets(tmp_path):
    win = _windows_manifest(tmp_path / "w", [10, 11, 12])
    lin = _linux_manifest(tmp_path / "l", [10, 11, 13])
    with pytest.raises(ValueError, match="logicFrame sets differ"):
        suite.state_cross_check(win, lin)


def test_cross_check_compares_float_bit_patterns_not_tolerance(tmp_path):
    """A one-ulp neighbour must be reported: the check is exact, not a tolerance."""
    frames = [1]
    win = _windows_manifest(tmp_path / "w", frames)
    lin = _linux_manifest(tmp_path / "l", frames)
    data = json.loads(lin.read_text(encoding="utf-8"))
    data["frames"][0]["state_evidence"]["fields"]["f10"] = float_bits(100.0) + 1
    lin.write_text(json.dumps(data), encoding="utf-8")
    result = suite.state_cross_check(win, lin)
    assert result["fields"]["player_x"]["equal"] == 0
    assert result["mismatchCount"] == 1


def test_cross_check_checks_difficulty_and_shot_against_the_replay(tmp_path):
    frames = [1, 2]
    win = _windows_manifest(tmp_path / "w", frames)
    lin = _linux_manifest(tmp_path / "l", frames)
    ok = suite.state_cross_check(win, lin, expected_difficulty=1, expected_shot=6)
    assert ok["difficulty"]["linuxFramesDiffering"] == []
    assert ok["shot"]["equal"] is True
    bad = suite.state_cross_check(win, lin, expected_difficulty=3, expected_shot=9)
    assert bad["difficulty"]["linuxFramesDiffering"] == [1]
    assert bad["shot"]["equal"] is False


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def test_summarise_report_orders_the_worst_frames_by_the_spec_metric(tmp_path):
    frames = list(range(5))
    report = _report(tmp_path, frames, playfield=0.1, hud=0.6, aux=0.2)
    summary = suite.summarise_report(report)
    assert summary["pairs"] == 5
    assert summary["logicFrameRange"] == [0, 4]
    assert len(summary["worstBySpecMetric"]) == 5
    assert summary["worstBySpecMetric"][0]["hudMad"] == pytest.approx(0.6)


def test_summarise_report_rejects_an_unknown_schema(tmp_path):
    path = tmp_path / "pixel-report.json"
    path.write_text(json.dumps({"schema": "something-else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected schema"):
        suite.summarise_report(path)


def test_combine_summaries_pools_by_frame_count(tmp_path):
    a = suite.summarise_report(_report(tmp_path / "a", list(range(4)), playfield=0.10, hud=0.30, aux=0.05))
    b = suite.summarise_report(_report(tmp_path / "b", list(range(8)), playfield=0.40, hud=0.90, aux=0.15))
    combined = suite.combine_summaries([a, b])
    assert combined["frames"] == 12
    assert combined["replays"] == 2
    # (4*0.10 + 8*0.40) / 12
    assert combined["playfield"]["pooled_mean_of_frame_mean_absolute_difference"] == pytest.approx(
        (4 * 0.10 + 8 * 0.40) / 12
    )
    assert combined["playfield"]["per_replay_mean"] == [0.1, 0.4]
    assert combined["hud"]["max_channel_difference"] == 251


def test_select_review_frames_covers_every_replay_and_the_global_worst():
    rows = [
        {"replay": "a", "logicFrame": 1, "playfieldMad": 0.10, "wholeFrameExcludingFpsMad": 0.20},
        {"replay": "a", "logicFrame": 2, "playfieldMad": 0.90, "wholeFrameExcludingFpsMad": 0.10},
        {"replay": "b", "logicFrame": 3, "playfieldMad": 0.30, "wholeFrameExcludingFpsMad": 0.40},
        {"replay": "c", "logicFrame": 4, "playfieldMad": 0.20, "wholeFrameExcludingFpsMad": 0.05},
        {"replay": "c", "logicFrame": 5, "playfieldMad": 0.70, "wholeFrameExcludingFpsMad": 0.06},
    ]
    chosen = suite.select_review_frames(rows, global_top=2)
    per_replay = {(row["replay"], row["logicFrame"]) for row in chosen if row["reason"].startswith("worst whole-frame")}
    assert per_replay == {("a", 1), ("b", 3), ("c", 5)}
    global_rows = [row for row in chosen if row["reason"].startswith("global")]
    # Global picks are the highest playfield MADs not already chosen: (a,2) then
    # (c,4) -- (c,5) and (b,3) are already in as per-replay worsts.
    assert [(row["replay"], row["logicFrame"]) for row in global_rows] == [("a", 2), ("c", 4)]
    keys = [(row["replay"], row["logicFrame"]) for row in chosen]
    assert len(keys) == len(set(keys)), "a frame must not be selected twice"


def test_select_review_frames_never_picks_a_hud_only_frame_for_every_replay():
    """A huge HUD number must not crowd out the per-replay content frames."""
    rows = [
        {"replay": "a", "logicFrame": 1, "playfieldMad": 0.01, "wholeFrameExcludingFpsMad": 0.02},
        {"replay": "b", "logicFrame": 2, "playfieldMad": 0.02, "wholeFrameExcludingFpsMad": 0.03},
    ]
    chosen = suite.select_review_frames(rows, global_top=0)
    assert {row["logicFrame"] for row in chosen} == {1, 2}


# ---------------------------------------------------------------------------
# montage layout
# ---------------------------------------------------------------------------
def test_build_review_image_composes_three_panels_and_a_zoom(tmp_path):
    win = np.zeros((480, 640, 3), dtype=np.uint8)
    lin = np.zeros((480, 640, 3), dtype=np.uint8)
    lin[100:140, 200:240] = 255
    win_path = tmp_path / "win.png"
    lin_path = tmp_path / "lin.png"
    Image.fromarray(np.dstack([win, np.full((480, 640, 1), 255, np.uint8)]), "RGBA").save(win_path)
    Image.fromarray(np.dstack([lin, np.full((480, 640, 1), 255, np.uint8)]), "RGBA").save(lin_path)
    out = tmp_path / "review.png"
    info = suite.build_review_image(win_path, lin_path, out, label="frame 42")
    with Image.open(out) as image:
        assert image.size == (640 * 3 + 16, 480 + 8 + 160)
    assert info["zoomBox"]["scale"] == 4
    assert info["diffGain"] == 8
    # The worst block must contain the injected rectangle.
    y, x = info["zoomBox"]["y"], info["zoomBox"]["x"]
    assert y[0] <= 100 < y[1] and x[0] <= 200 < x[1]


def test_build_review_image_rejects_a_non_rgba_png(tmp_path):
    path = tmp_path / "rgb.png"
    Image.new("RGB", (640, 480)).save(path)
    with pytest.raises(ValueError, match="expected PNG RGBA 640x480"):
        suite.build_review_image(path, path, tmp_path / "out.png", label="x")


def test_difference_image_amplifies_and_clamps():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.zeros((4, 4, 3), dtype=np.uint8)
    b[0, 0] = 100
    diff = suite.difference_image(a, b, gain=8)
    assert int(diff[0, 0].max()) == 255  # clamped
    b[1, 1] = 2
    diff = suite.difference_image(a, b, gain=8)
    assert int(diff[1, 1].max()) == 16


# ---------------------------------------------------------------------------
# shared-template guards (the incident EXP-P0-058 records)
# ---------------------------------------------------------------------------
def test_parse_sha256sum_reads_basename_and_hex():
    text = (
        "60e49abc8f0a955aa8fa883769a30c66913417be453632c9a8a99b0b8a5c5191  /root/v400-data/game/score.dat\n"
        "48bf7621da5879ef5ec1392fcac2cc0a85c313912a24fb33b4cddccf0d28f07a  th08.cfg\n"
        "garbage line\n"
    )
    parsed = suite.parse_sha256sum(text)
    assert parsed["score.dat"] == "60e49abc8f0a955aa8fa883769a30c66913417be453632c9a8a99b0b8a5c5191"
    assert parsed["th08.cfg"] == "48bf7621da5879ef5ec1392fcac2cc0a85c313912a24fb33b4cddccf0d28f07a"
    assert len(parsed) == 2


def test_shared_template_status_detects_a_dirty_score_dat():
    clean = suite.shared_template_status(dict(suite.EXPECTED_SHARED_TEMPLATE))
    assert clean["clean"] is True
    assert clean["differing"] == []
    dirty = dict(suite.EXPECTED_SHARED_TEMPLATE)
    dirty["score.dat"] = "0" * 64
    status = suite.shared_template_status(dirty)
    assert status["clean"] is False
    assert status["differing"] == ["score.dat"]


def test_prepare_wsl_runtime_never_points_at_the_shared_dir(tmp_path, monkeypatch):
    """The runtime dir must be per-case and must copy from ``.orig``."""
    calls: list[list[str]] = []

    class FakeRunner:
        def run(self, name, argv, *, timeout=600.0, cwd=None):
            calls.append(argv)
            return {
                "name": name,
                "command": " ".join(argv),
                "exitCode": 0,
                "wallSeconds": 0.0,
                "stdoutTail": "60e49abc8f0a955aa8fa883769a30c66913417be453632c9a8a99b0b8a5c5191  score.dat\n"
                "48bf7621da5879ef5ec1392fcac2cc0a85c313912a24fb33b4cddccf0d28f07a  th08.cfg\n",
                "stderrTail": "",
            }

    info = suite.prepare_wsl_runtime("case-x", FakeRunner())
    script = calls[0][-1]
    assert suite.WSL_SHARED_DATA in script  # only as the source of the copies
    assert f"{suite.WSL_RUNTIME_ROOT}/case-x/game" in script
    assert info["pristineEqual"] is True
    assert info["runtimeDataDir"] == f"{suite.WSL_RUNTIME_ROOT}/case-x/game"
    # Every mutable file is copied from its .orig counterpart, never linked.
    for name in suite.SHARED_TEMPLATE_FILES:
        assert f"cp -f '{suite.WSL_SHARED_DATA}/{name}.orig'" in script
    # The two big data blobs are symlinks, not copies.
    for name in suite.SHARED_TEMPLATE_LINKS:
        assert f"ln -sfn '{suite.WSL_SHARED_DATA}/{name}'" in script


def test_linux_capture_passes_the_private_data_dir(tmp_path, monkeypatch):
    """A regression guard for the dirty-template incident: --data-dir is private."""
    commands: list[list[str]] = []

    class FakeRunner:
        def run(self, name, argv, *, timeout=600.0, cwd=None):
            commands.append(argv)
            stdout = ""
            if name.endswith("-linux-capture"):
                stdout = "{}"
            if name.endswith("-runtime"):
                stdout = (
                    "60e49abc8f0a955aa8fa883769a30c66913417be453632c9a8a99b0b8a5c5191  score.dat\n"
                    "48bf7621da5879ef5ec1392fcac2cc0a85c313912a24fb33b4cddccf0d28f07a  th08.cfg\n"
                )
            if name.endswith("-stage-sha"):
                stdout = f"{'a' * 64}  /root/x.rpy\n"
            return {"name": name, "command": " ".join(argv), "exitCode": 0, "wallSeconds": 0.0,
                    "stdoutTail": stdout, "stderrTail": ""}

    spec = {"name": "case-y", "replay": "th08_replay/Th8_ud2970.rpy", "stage": 3, "startLogicFrame": 1}
    win = _windows_manifest(tmp_path / "w", [10, 11])
    # stage_replay_in_wsl compares the staged sha with the local file; the fake
    # reports a fake hash, so patch the helper to keep this test about --data-dir.
    monkeypatch.setattr(suite, "sha256_file", lambda path: "a" * 64)
    info = suite.linux_capture(spec, win, tmp_path / "case-y", FakeRunner())
    capture_command = next(argv for argv in commands if " capture " in argv[-1])
    joined = " ".join(capture_command)
    assert f"--data-dir '{suite.WSL_RUNTIME_ROOT}/case-y/game'" in joined
    assert f"--data-dir {suite.WSL_SHARED_DATA}" not in joined
    assert info["captureExitCode"] == 0
    assert info["runtime"]["runtimeDataDir"] == f"{suite.WSL_RUNTIME_ROOT}/case-y/game"


# ---------------------------------------------------------------------------
# spec loading
# ---------------------------------------------------------------------------
def test_load_specs_accepts_the_builtin_three_replays():
    specs = suite.load_specs(None)
    assert len(specs) == 3
    assert len({spec["name"] for spec in specs}) == 3
    # G0.4 forbids "all three windows are the Stage 1 title card".
    assert len({(spec["replay"], spec["stage"]) for spec in specs}) == 3


def test_load_specs_rejects_duplicate_names(tmp_path):
    path = tmp_path / "specs.json"
    path.write_text(
        json.dumps(
            [
                {"name": "a", "replay": "th08_replay/Th8_ud2970.rpy", "stage": 3, "startLogicFrame": 10},
                {"name": "a", "replay": "th08_replay/Th8_ud0059.rpy", "stage": 5, "startLogicFrame": 10},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="names must be unique"):
        suite.load_specs(path)


def test_load_specs_rejects_a_missing_replay(tmp_path):
    path = tmp_path / "specs.json"
    path.write_text(
        json.dumps([{"name": "a", "replay": "th08_replay/nope.rpy", "stage": 0, "startLogicFrame": 0}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="replay missing"):
        suite.load_specs(path)


def test_cmd_plan_rejects_a_stage_the_replay_does_not_contain(tmp_path, capsys):
    specs = tmp_path / "specs.json"
    specs.write_text(
        json.dumps(
            [
                {
                    "name": "bad",
                    # Th8_ud2970 has no Stage 4B (index 4); it took the 4A route.
                    "replay": "th08_replay/Th8_ud2970.rpy",
                    "stage": 4,
                    "startLogicFrame": 0,
                }
            ]
        ),
        encoding="utf-8",
    )
    code = suite.main(["--out", str(tmp_path / "o"), "--specs", str(specs), "plan"])
    captured = capsys.readouterr()
    assert code == 3
    assert "stage 4 not in replay stages" in captured.err


# ---------------------------------------------------------------------------
# step recording
# ---------------------------------------------------------------------------
def test_step_runner_records_exit_code_and_command(tmp_path):
    runner = suite.StepRunner(tmp_path / "steps.json")
    record = runner.run("true", [sys.executable, "-c", "print('hi')"])
    assert record["exitCode"] == 0
    assert "hi" in record["stdoutTail"]
    assert "print" in record["command"]
    payload = json.loads((tmp_path / "steps.json").read_text(encoding="utf-8"))
    assert payload["steps"][0]["name"] == "true"


def test_step_runner_records_a_failing_exit_code(tmp_path):
    runner = suite.StepRunner(tmp_path / "steps.json")
    record = runner.run("false", [sys.executable, "-c", "raise SystemExit(7)"])
    assert record["exitCode"] == 7
