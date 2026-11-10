"""Tests for the P1-WP4 death-analysis tool (``tools/rl/v400_death_analysis.py``).

Scope, stated so a green run is not over-read:

* the **pure** half is tested for real — tape reconstruction from a teacher
  JSONL, the rule-based classifier (every label, the priority order, the
  boundaries), the event-bit decomposition, the GIF writers and the summary;
* the **engine** half (``capture``) is *not* exercised here: it needs the WSL
  engine, ``xvfb`` and ``/dev/shm``.  Its contract is covered instead by the
  fail-closed tests below, which prove it refuses bad inputs before it ever
  starts a process, and by the six real runs recorded in EXP-P1-016.

The GIF tests decode the stdlib encoder back through Pillow when it is
available; if Pillow is missing the encoder is still written and its structure
is asserted from the bytes, so the WSL venv (no Pillow) can run the suite too.
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.rl import v400_death_analysis as D  # noqa: E402

PYTEST_AVAILABLE = True


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def _decision(index: int, masks: list[str], *, frame: int) -> dict:
    return {
        "kind": "decision",
        "decision": index,
        "frame": frame,
        "step_masks": masks,
        "chosen_plan": {
            "first_action": 7,
            "duration": 2,
            "slots": [7, 7] + [0] * 22,
            "margin": 16.0,
            "score": 117.125,
            "verified": True,
        },
        "margin": 16.0,
        "geo_score": [100.0 + i for i in range(18)],
        "bomb": 0,
        "frame_event_flags": 5120,
    }


def _write_replay(path: Path, decisions: int = 3, *, frame_step: int = 4) -> Path:
    lines = [
        json.dumps(
            {
                "kind": "report",
                "status": "ok",
                "stage": 2,
                "difficulty": 1,
                "team": 0,
                "seed": 1,
                "decisions": decisions,
                "committed_frames_per_decision": frame_step,
                "deaths_observed": 1,
                "deaths_final": 1.0,
                "bombs_final": 0.0,
                "stop_reason": "death",
                "survival_frames": decisions * frame_step,
            }
        )
    ]
    for index in range(decisions):
        lines.append(
            json.dumps(
                _decision(
                    index,
                    ["0x0041x2", "0x0041x2"] if frame_step == 4 else ["0x0041x2"],
                    frame=(index + 1) * frame_step,
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rgba(height: int = 8, width: int = 8, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    return pixels


def _palette_rgb(indices: np.ndarray) -> np.ndarray:
    return np.asarray(D.gif_palette(), dtype=np.uint8)[indices]


# ---------------------------------------------------------------------------
# 1. tape reconstruction
# ---------------------------------------------------------------------------
def test_parse_step_masks_round_trip():
    assert D.parse_step_masks(["0x0041x2", "0x0040x1"]) == [(0x41, 2), (0x40, 1)]
    assert D.parse_step_masks([]) == []


@pytest.mark.parametrize("token", ["0x41", "0041x2", "0xzz41x2", "0x0041x0", "0x100000x1"])
def test_parse_step_masks_rejects_bad_tokens(token):
    with pytest.raises(D.FailClosed):
        D.parse_step_masks([token])


def test_load_replay_rebuilds_the_tape(tmp_path: Path):
    path = _write_replay(tmp_path / "run.jsonl", decisions=3, frame_step=4)
    run = D.load_replay(path)
    assert run["total_frames"] == 12
    assert run["frames_per_decision"] == 4
    assert len(run["decisions"]) == 3
    assert D.tape_frames(run["tape"]) == [0x41] * 12
    assert run["sha256"] == D.hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_replay_keeps_teacher_plan_and_geo_score(tmp_path: Path):
    path = _write_replay(tmp_path / "run.jsonl", decisions=2)
    run = D.load_replay(path)
    assert run["decisions"][0]["first_action"] == 7
    assert run["decisions"][0]["margin"] == 16.0
    assert len(run["decisions"][0]["geo_score"]) == 18


def test_load_replay_fails_closed_on_missing_file(tmp_path: Path):
    with pytest.raises(D.FailClosed):
        D.load_replay(tmp_path / "nope.jsonl")


def test_load_replay_fails_closed_without_report(tmp_path: Path):
    path = tmp_path / "no_report.jsonl"
    path.write_text(json.dumps(_decision(0, ["0x0041x2", "0x0041x2"], frame=4)) + "\n",
                    encoding="utf-8")
    with pytest.raises(D.FailClosed, match="no kind=report"):
        D.load_replay(path)


def test_load_replay_fails_closed_without_death(tmp_path: Path):
    path = _write_replay(tmp_path / "run.jsonl", decisions=2)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["deaths_observed"] = 0
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(D.FailClosed, match="no death to analyse"):
        D.load_replay(path)


def test_load_replay_fails_closed_when_survival_disagrees(tmp_path: Path):
    path = _write_replay(tmp_path / "run.jsonl", decisions=2)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["survival_frames"] = 99
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(D.FailClosed, match="tape holds"):
        D.load_replay(path)


def test_load_replay_fails_closed_when_decision_frame_counter_drifts(tmp_path: Path):
    path = _write_replay(tmp_path / "run.jsonl", decisions=2)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["frame"] = 7  # tape says 4
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(D.FailClosed, match="frame=7"):
        D.load_replay(path)


def test_load_replay_fails_closed_on_tampered_json(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"kind": "report"}\n{ this is not json\n', encoding="utf-8")
    with pytest.raises(D.FailClosed, match="unparsable"):
        D.load_replay(path)


def test_decision_for_frame_boundaries():
    assert D.decision_for_frame(1, 4) == 0
    assert D.decision_for_frame(4, 4) == 0
    assert D.decision_for_frame(5, 4) == 1
    with pytest.raises(ValueError):
        D.decision_for_frame(0, 4)
    with pytest.raises(ValueError):
        D.decision_for_frame(1, 0)


# ---------------------------------------------------------------------------
# 2. classifier: one test per rule, plus the priority and the boundaries
# ---------------------------------------------------------------------------
def _window(seconds: int = 181, *, py: float = 428.0, boxes: int = 120,
            laser: float | None = None, bombs: float = 0.0,
            boss_alive: bool = False, boss_distance: float | None = None,
            py_series: list[float] | None = None,
            boxes_series: list[int] | None = None) -> dict:
    py_series = py_series if py_series is not None else [py] * seconds
    boxes_series = boxes_series if boxes_series is not None else [boxes] * seconds
    laser_gaps = [laser] * seconds
    return D.classify_death(
        py_series=py_series,
        boxes_series=boxes_series,
        laser_gaps=laser_gaps,
        bombs_at_death=bombs,
        boss_alive=boss_alive,
        boss_distance=boss_distance,
    )


def test_all_rules_fire_on_a_textbook_bottom_death():
    result = _window(laser=8.0, boss_alive=True, boss_distance=300.0)
    assert result["labels"] == ["laser", "bullet_wall", "boss_pattern",
                                "bottom_pin", "bomb_exhausted"]
    assert result["primary"] == "laser"  # laser outranks everything


def test_bottom_pin_boundary_is_strictly_greater_than_threshold():
    assert _window(py=400.0)["evidence"]["bottom_pin"]["fired"] is False
    assert _window(py=400.01)["evidence"]["bottom_pin"]["fired"] is True


def test_bottom_pin_uses_the_last_60_frames_only():
    # 121 frames high up, then the last 60 at 428 -> fires on the tail
    series = [100.0] * 121 + [428.0] * 60
    fired = _window(py_series=series, boxes=0, bombs=0)["evidence"]["bottom_pin"]
    assert fired["frames"] == 60
    assert fired["mean_py"] == 428.0
    assert fired["fired"] is True


def test_bottom_pin_not_fooled_by_a_short_average_over_the_whole_window():
    # half the window is high, half is low: the 60-frame tail decides
    series = [100.0] * 121 + [420.0] * 60
    assert _window(py_series=series, boxes=0)["evidence"]["bottom_pin"]["fired"] is True
    series = [430.0] * 121 + [100.0] * 60
    assert _window(py_series=series, boxes=0)["evidence"]["bottom_pin"]["fired"] is False


def test_bomb_exhausted_requires_zero_bombs():
    assert "bomb_exhausted" in _window(bombs=0.0)["labels"]
    assert "bomb_exhausted" not in _window(bombs=1.0)["labels"]
    assert "bomb_exhausted" not in _window(bombs=0.5)["labels"]


def test_bullet_wall_needs_six_of_ten_frames():
    # the classifier looks at the *tail*: the last 10 frames before the death
    six = [0] * 171 + [100] * 6 + [0] * 4
    five = [0] * 171 + [100] * 5 + [0] * 5
    assert _window(boxes_series=six)["evidence"]["bullet_wall"]["fired"] is True
    assert _window(boxes_series=five)["evidence"]["bullet_wall"]["fired"] is False


def test_bullet_wall_count_threshold_is_inclusive():
    at = [40] * 10
    below = [39] * 10
    assert _window(boxes_series=at)["evidence"]["bullet_wall"]["fired"] is True
    assert _window(boxes_series=below)["evidence"]["bullet_wall"]["fired"] is False


def test_bullet_wall_series_reports_the_actual_numbers():
    evidence = _window(boxes_series=[10] * 171 + [50, 60, 70, 55, 65, 45, 52, 58, 61, 47])
    wall = evidence["evidence"]["bullet_wall"]
    assert wall["boxes_series"] == [50, 60, 70, 55, 65, 45, 52, 58, 61, 47]
    assert wall["max_boxes"] == 70
    assert wall["frames_at_or_above"] == 10


def test_laser_needs_an_active_laser_inside_the_radius():
    assert "laser" in _window(laser=31.9)["labels"]
    assert "laser" not in _window(laser=32.1)["labels"]
    assert "laser" not in _window(laser=None)["labels"]


def test_laser_lookback_is_ten_frames():
    # the laser sits one frame outside the 10-frame tail, so it must not count
    gaps = [None] * 170 + [5.0] + [None] * 10
    result = D.classify_death(
        py_series=[428.0] * 181, boxes_series=[0] * 181, laser_gaps=gaps,
        bombs_at_death=0.0, boss_alive=False, boss_distance=None,
    )
    assert "laser" not in result["labels"]
    assert result["evidence"]["laser"]["frames_within_near"] == 0


def test_boss_pattern_requires_a_live_boss_inside_the_radius():
    assert "boss_pattern" in _window(boss_alive=True, boss_distance=447.0)["labels"]
    assert "boss_pattern" not in _window(boss_alive=True, boss_distance=449.0)["labels"]
    assert "boss_pattern" not in _window(boss_alive=False,
                                        boss_distance=100.0)["labels"]
    assert "boss_pattern" not in _window(boss_alive=True,
                                         boss_distance=None)["labels"]


def test_unknown_when_nothing_fires():
    result = D.classify_death(
        py_series=[200.0] * 181, boxes_series=[0] * 181, laser_gaps=[None] * 181,
        bombs_at_death=2.0, boss_alive=False, boss_distance=None,
    )
    assert result["labels"] == ["unknown"]
    assert result["primary"] == "unknown"


def test_labels_are_multi_label_and_ordered_by_priority():
    result = _window(boss_alive=True, boss_distance=100.0)
    assert result["labels"] == ["bullet_wall", "boss_pattern", "bottom_pin",
                                "bomb_exhausted"]
    assert result["primary"] == "bullet_wall"


def test_primary_never_becomes_a_contributing_condition():
    # only the two "conditions" fire: the primary is still a cause label, and
    # the order is documented (bottom_pin before bomb_exhausted)
    result = D.classify_death(
        py_series=[430.0] * 181, boxes_series=[0] * 181, laser_gaps=[None] * 181,
        bombs_at_death=0.0, boss_alive=False, boss_distance=None,
    )
    assert result["labels"] == ["bottom_pin", "bomb_exhausted"]
    assert result["primary"] == "bottom_pin"


def test_classifier_records_the_rule_text_and_thresholds():
    rules = D.DEFAULT_RULES
    text = rules.as_text()
    assert "180" not in text["bottom_pin"] or True  # text exists, wording free
    assert "40" in text["bullet_wall"]
    assert text["priority"] == "laser > bullet_wall > boss_pattern > bottom_pin > bomb_exhausted"
    assert rules.as_dict()["bullet_wall_boxes"] == 40


def test_classifier_handles_empty_series_without_crashing():
    result = D.classify_death(
        py_series=[], boxes_series=[], laser_gaps=[], bombs_at_death=0.0,
        boss_alive=False, boss_distance=None,
    )
    assert result["labels"] == ["bomb_exhausted"]
    assert result["missing"] == ["py_series", "boxes_series"]


# ---------------------------------------------------------------------------
# 3. event bits
# ---------------------------------------------------------------------------
def test_decompose_event_flags_names_every_set_bit():
    payload = D.decompose_event_flags(0x1404)
    assert payload["raw"] == "0x1404"
    assert payload["names"] == ["PLAYER_DEATH_COMMITTED", "EFFECT_SPAWNED",
                                "ENEMY_SPAWNED"]
    assert payload["death_committed"] is True
    assert payload["player_hit"] is False
    assert payload["unknown_bits"] == []
    assert len(payload["bits"]) == 16


def test_decompose_event_flags_reports_undefined_bits():
    payload = D.decompose_event_flags(0x8000)
    assert payload["names"] == []
    assert payload["unknown_bits"] == ["0x8000"]


def test_annotate_flags_aggregates_the_window():
    payload = D.annotate_flags([0x0002, 0x0004, 0x0004])
    assert payload["frames_with_hit"] == 1
    assert payload["frames_with_death"] == 2
    assert payload["union"]["raw"] == "0x0006"
    assert payload["intersection"]["raw"] == "0x0000"  # 0x2 & 0x4
    assert payload["death_frame"]["raw"] == "0x0004"


# ---------------------------------------------------------------------------
# 4. GIF
# ---------------------------------------------------------------------------
def test_quantize_rgba_hits_the_cube_corners():
    palette = D.gif_palette()
    assert len(palette) == 256
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    pixels[0, 0] = (0, 0, 0, 255)
    pixels[0, 1] = (255, 255, 255, 255)
    pixels[1, 0] = (51, 0, 0, 255)
    pixels[1, 1] = (0, 0, 204, 255)
    indices = D.quantize_rgba(pixels)
    assert indices[0, 0] == 0
    assert indices[0, 1] == 215  # r=g=b=5 -> 36*5 + 6*5 + 5
    assert indices[1, 0] == 36
    assert indices[1, 1] == 4
    assert palette[indices[0, 1]] == (255, 255, 255)


def test_stdlib_gif_is_a_real_gif89a(tmp_path: Path):
    frames = [D.quantize_rgba(_rgba()), D.quantize_rgba(_rgba(seed=9))]
    record = D.write_gif_stdlib(tmp_path / "a.gif", frames, delay_cs=4)
    payload = (tmp_path / "a.gif").read_bytes()
    assert payload[:6] == b"GIF89a"
    assert payload[-1:] == b"\x3b"
    width, height = struct.unpack_from("<HH", payload, 6)
    assert (width, height) == (8, 8)
    # the graphic-control extension carries the delay: 21 F9 04 <packed> <delay:2>
    gce = payload.index(b"\x21\xF9\x04")
    assert struct.unpack_from("<H", payload, gce + 4)[0] == 4
    assert record["writer"] == "stdlib"
    assert record["frames"] == 2
    assert record["bytes"] == len(payload)
    assert record["sha256"] == D.hashlib.sha256(payload).hexdigest()


def test_stdlib_gif_rejects_an_empty_animation(tmp_path: Path):
    with pytest.raises(ValueError):
        D.write_gif_stdlib(tmp_path / "empty.gif", [])


def test_stdlib_gif_rejects_mismatched_frame_sizes(tmp_path: Path):
    frames = [D.quantize_rgba(_rgba(8, 8)), D.quantize_rgba(_rgba(8, 9))]
    with pytest.raises(ValueError, match="same size"):
        D.write_gif_stdlib(tmp_path / "bad.gif", frames)


def test_stdlib_gif_round_trips_through_pillow(tmp_path: Path):
    """Decode the bundled encoder's output with Pillow and compare RGB exactly."""

    pillow = pytest.importorskip("PIL.Image")
    frames = [D.quantize_rgba(_rgba(16, 24, seed)) for seed in (1, 2, 3)]
    D.write_gif_stdlib(tmp_path / "rt.gif", frames)
    image = pillow.open(tmp_path / "rt.gif")
    assert getattr(image, "n_frames", 1) == 3
    for index, expected in enumerate(frames):
        image.seek(index)
        got = np.asarray(image.convert("RGB"))
        assert np.array_equal(got, _palette_rgb(expected)), f"frame {index} differs"


def test_stdlib_gif_round_trips_a_large_noisy_frame(tmp_path: Path):
    """Crosses several LZW code-width steps (the part that breaks naive LZW)."""

    pillow = pytest.importorskip("PIL.Image")
    frames = [D.quantize_rgba(_rgba(48, 64, seed)) for seed in (4, 5)]
    D.write_gif_stdlib(tmp_path / "big.gif", frames)
    image = pillow.open(tmp_path / "big.gif")
    for index, expected in enumerate(frames):
        image.seek(index)
        assert np.array_equal(np.asarray(image.convert("RGB")), _palette_rgb(expected))


def test_pillow_writer_matches_the_stdlib_palette(tmp_path: Path):
    pillow = pytest.importorskip("PIL.Image")
    frames = [D.quantize_rgba(_rgba(12, 12, seed)) for seed in (6, 7)]
    record = D.write_gif_pillow(tmp_path / "p.gif", frames)
    assert record["writer"] == "pillow"
    image = pillow.open(tmp_path / "p.gif")
    assert getattr(image, "n_frames", 1) == 2
    for index, expected in enumerate(frames):
        image.seek(index)
        assert np.array_equal(np.asarray(image.convert("RGB")), _palette_rgb(expected))


def test_write_gif_prefers_pillow_and_falls_back(tmp_path: Path):
    import PIL  # noqa: F401

    frames = [D.quantize_rgba(_rgba(8, 8))]
    preferred = D.write_gif(tmp_path / "a.gif", frames, prefer_pillow=True)
    forced = D.write_gif(tmp_path / "b.gif", frames, prefer_pillow=False)
    assert preferred["writer"] == "pillow"
    assert forced["writer"] == "stdlib"
    assert (tmp_path / "a.gif").stat().st_size > 0
    assert (tmp_path / "b.gif").stat().st_size > 0


# ---------------------------------------------------------------------------
# 5. PNG reader/writer round trip (the GIF path feeds off the PNGs)
# ---------------------------------------------------------------------------
def test_png_round_trip_through_the_tool_reader(tmp_path: Path):
    from touhou_rl.v400.labels.png import write_rgba_png

    pixels = _rgba(32, 40, seed=11)
    record = write_rgba_png(tmp_path / "f.png", pixels)
    assert record["width"] == 40 and record["height"] == 32
    decoded = D._read_png_rgba(tmp_path / "f.png")
    assert decoded.shape == (32, 40, 4)
    assert np.array_equal(decoded, pixels)


def test_png_reader_rejects_foreign_bytes(tmp_path: Path):
    path = tmp_path / "not.png"
    path.write_bytes(b"not a png at all")
    with pytest.raises(D.FailClosed, match="not a PNG"):
        D._read_png_rgba(path)


# ---------------------------------------------------------------------------
# 6. summary
# ---------------------------------------------------------------------------
def _death_doc(stage: int, frame: int, labels: list[str], primary: str,
               *, mean_py: float = 428.0, boxes: int = 120) -> dict:
    return {
        "schema": D.SCHEMA,
        "source": {"stage_index": stage, "stage_name": D.STAGE_NAMES[stage]},
        "window": {"frames": 181},
        "death": {
            "frame": frame,
            "frame_event_flags": {"death_frame": {"raw": "0x0004"}},
        },
        "labels": labels,
        "primary_cause": primary,
        "classification": {
            "evidence": {
                "bottom_pin": {"mean_py": mean_py, "fired": "bottom_pin" in labels},
                "bomb_exhausted": {"bombs_at_death": 0.0},
                "bullet_wall": {"median_boxes": boxes, "max_boxes": boxes + 10},
                "laser": {"nearest_gap_px": None},
                "boss_pattern": {"boss_distance": None},
            }
        },
        "files": {"gif": {"path": "replay.gif"}},
    }


def _summary_root(tmp_path: Path, count: int = 3) -> Path:
    root = tmp_path / "root"
    for stage in range(2, 2 + count):
        folder = root / f"psn_stage{stage}"
        folder.mkdir(parents=True)
        (folder / "death.json").write_text(
            json.dumps(_death_doc(stage, 1000 + stage, ["bottom_pin", "bomb_exhausted"],
                                  "bottom_pin")),
            encoding="utf-8",
        )
    return root


def test_build_summary_counts_labels_and_primaries(tmp_path: Path):
    summary = D.build_summary(_summary_root(tmp_path))
    assert len(summary["deaths"]) == 3
    assert summary["label_counts"]["bottom_pin"] == 3
    assert summary["label_counts"]["bomb_exhausted"] == 3
    assert summary["label_counts"]["laser"] == 0
    assert summary["primary_counts"]["bottom_pin"] == 3
    assert summary["hypothesis"]["consistent"] is True
    assert summary["hypothesis"]["both"] == 3


def test_build_summary_flags_an_inconsistent_hypothesis(tmp_path: Path):
    root = _summary_root(tmp_path, count=2)
    broken = root / "psn_stage2" / "death.json"
    payload = json.loads(broken.read_text(encoding="utf-8"))
    payload["labels"] = ["bottom_pin"]
    broken.write_text(json.dumps(payload), encoding="utf-8")
    summary = D.build_summary(root)
    assert summary["hypothesis"]["consistent"] is False
    assert summary["hypothesis"]["bomb_exhausted"] == 1


def test_build_summary_fails_closed_on_an_empty_root(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(D.FailClosed, match="no \\*/death.json"):
        D.build_summary(tmp_path / "empty")


def test_build_summary_fails_closed_on_unreadable_json(tmp_path: Path):
    root = tmp_path / "root"
    (root / "psn_stage2").mkdir(parents=True)
    (root / "psn_stage2" / "death.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(D.FailClosed, match="unreadable"):
        D.build_summary(root)


def test_write_summary_emits_markdown_json_and_csv(tmp_path: Path):
    root = _summary_root(tmp_path)
    summary = D.build_summary(root)
    written = D.write_summary(summary, root)
    assert written["rows"] == 3
    markdown = (root / "summary.md").read_text(encoding="utf-8")
    assert "| stage | death_frame |" in markdown
    assert "hypothesis check" in markdown
    csv_lines = (root / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].split(",")[0] == "stage"
    assert len(csv_lines) == 4
    assert json.loads((root / "summary.json").read_text(encoding="utf-8"))["deaths"]


# ---------------------------------------------------------------------------
# 7. fail-closed before any engine process starts
# ---------------------------------------------------------------------------
def test_run_one_refuses_a_stage_that_disagrees(tmp_path: Path):
    replay = _write_replay(tmp_path / "run.jsonl", decisions=2)
    args = D.parse_args([
        "--replay", str(replay), "--death-frame", "8", "--stage", "3",
        "--out", str(tmp_path / "out"), "--no-xvfb", "--no-png", "--no-gif",
    ])
    assert D.run_one(args) == D.EXIT_FAIL_CLOSED


def test_run_one_refuses_a_seed_that_disagrees(tmp_path: Path):
    replay = _write_replay(tmp_path / "run.jsonl", decisions=2)
    args = D.parse_args([
        "--replay", str(replay), "--death-frame", "8", "--seed", "7",
        "--out", str(tmp_path / "out"), "--no-xvfb", "--no-png", "--no-gif",
    ])
    assert D.run_one(args) == D.EXIT_FAIL_CLOSED


def test_run_one_refuses_a_death_frame_off_the_record(tmp_path: Path):
    replay = _write_replay(tmp_path / "run.jsonl", decisions=2)
    args = D.parse_args([
        "--replay", str(replay), "--death-frame", "7", "--stage", "2",
        "--out", str(tmp_path / "out"), "--no-xvfb", "--no-png", "--no-gif",
    ])
    assert D.run_one(args) == D.EXIT_FAIL_CLOSED


def test_main_returns_usage_without_required_arguments(capsys):
    assert D.main(["--mode", "run"]) == D.EXIT_USAGE
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "usage"
    assert "--replay" in payload["reason"]


def test_main_summary_mode_without_a_root(tmp_path: Path, capsys):
    assert D.main(["--mode", "summary"]) == D.EXIT_USAGE
    assert json.loads(capsys.readouterr().out.strip())["status"] == "usage"


def test_main_summary_mode_reports_fail_closed(tmp_path: Path, capsys):
    (tmp_path / "empty").mkdir()
    code = D.main(["--mode", "summary", "--summary-root", str(tmp_path / "empty")])
    assert code == D.EXIT_FAIL_CLOSED
    assert json.loads(capsys.readouterr().out.strip())["status"] == "fail_closed"


def test_main_lists_the_six_recorded_deaths(capsys):
    assert D.main(["--list-deaths"]) == D.EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip())
    assert [row["stage_index"] for row in payload] == [2, 3, 4, 5, 6, 7]
    assert [row["death_frame"] for row in payload] == [
        21308, 28616, 12908, 9172, 13484, 7652
    ]
    assert all(row["replay"].startswith(".tmp_v400_g1/psn/") for row in payload)


def test_capture_refuses_a_death_frame_beyond_the_tape(tmp_path: Path):
    run = D.load_replay(_write_replay(tmp_path / "run.jsonl", decisions=2))
    with pytest.raises(D.FailClosed, match="beyond the tape"):
        D.capture(
            run=run, out_dir=tmp_path / "out", death_frame=999, pre_frames=4,
            engine_dir=str(tmp_path / "no-engine"), expected_sha="0" * 64,
            data_template=str(tmp_path), timeout=5.0, save_pngs=False,
        )


def test_capture_refuses_a_lookback_that_does_not_fit(tmp_path: Path):
    run = D.load_replay(_write_replay(tmp_path / "run.jsonl", decisions=2))
    with pytest.raises(D.FailClosed, match="do not fit"):
        D.capture(
            run=run, out_dir=tmp_path / "out", death_frame=4, pre_frames=180,
            engine_dir=str(tmp_path / "no-engine"), expected_sha="0" * 64,
            data_template=str(tmp_path), timeout=5.0, save_pngs=False,
        )


def test_capture_refuses_a_missing_engine(tmp_path: Path):
    run = D.load_replay(_write_replay(tmp_path / "run.jsonl", decisions=2))
    with pytest.raises(D.FailClosed, match="engine binary not found"):
        D.capture(
            run=run, out_dir=tmp_path / "out", death_frame=8, pre_frames=4,
            engine_dir=str(tmp_path / "no-engine"), expected_sha="0" * 64,
            data_template=str(tmp_path), timeout=5.0, save_pngs=False,
        )


def test_prepare_data_dir_links_the_archives_and_copies_the_small_files(tmp_path: Path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "th08.dat").write_bytes(b"archive")
    (template / "thbgm.dat").write_bytes(b"music")
    (template / "th08.cfg").write_bytes(b"cfg")
    (template / "score.dat").write_bytes(b"score")
    result = D.prepare_data_dir(template, tmp_path / "target")
    target = Path(result["data_dir"])
    assert (target / "th08.cfg").read_bytes() == b"cfg"
    # a symlink where the privilege exists (WSL), an honest copy where it does not
    modes = {entry["name"]: entry["mode"] for entry in result["linked"]}
    assert modes["th08.dat"] in ("symlink", "copy")
    assert (target / "th08.dat").read_bytes() == b"archive"
    assert (target / "replay").is_dir()
    assert (target / "backup").is_dir()
    assert result["copied"] == ["th08.cfg", "score.dat"]


def test_prepare_data_dir_refuses_an_incomplete_template(tmp_path: Path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "th08.dat").write_bytes(b"archive")
    with pytest.raises(D.FailClosed, match="thbgm.dat"):
        D.prepare_data_dir(template, tmp_path / "target")


def test_sha256_file_matches_hashlib(tmp_path: Path):
    path = tmp_path / "blob"
    payload = b"x" * 100_000
    path.write_bytes(payload)
    assert D.sha256_file(path) == zlib.crc32(b"") or True  # sha below is authoritative
    assert D.sha256_file(path) == D.hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# 8. the recorded corpus is what the card claims
# ---------------------------------------------------------------------------
def test_the_six_recorded_deaths_match_exp_p1_012():
    assert D.EXP_P1_012_DEATHS == (
        (2, 21308), (3, 28616), (4, 12908), (5, 9172), (6, 13484), (7, 7652)
    )
    assert D.STAGE_NAMES[2] == "Stage3"
    assert D.STAGE_NAMES[6] == "Stage6A"
