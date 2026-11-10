"""Tests for the P2-WP3 reward detector + G2 metrics (EXP-P2-003).

Everything here is offline: synthetic shards are written with the collector's own
plane layout, no engine, no GPU, no dataset.  The point is to pin the two things
that can silently produce a wrong G2 number 閳?a leaky split and a mis-computed
metric 閳?and to pin the fail-closed paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.rl import v400_p2_bulk_collect as bulk
from tools.rl import v400_p2_reward_detector as det


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def write_shard(root: Path, *, name: str, stage: int, seed: int, frames: int,
                death_at: tuple[int, ...] = (), bomb_at: tuple[int, ...] = (),
                hit_at: tuple[int, ...] = (), source: str = "dive",
                label_version: str = det.LABEL_VERSION,
                engine: str = "a" * 64, torn_tail: bool = False,
                lives: tuple[int, ...] | None = None,
                bombs: tuple[int, ...] | None = None,
                score: tuple[int, ...] | None = None) -> dict:
    """One synthetic shard attempt; labels are the only source of truth."""

    attempt = root / "shards" / name / "attempt-01"
    attempt.mkdir(parents=True)
    sizes = bulk.frame_bytes()
    for plane, size in sizes.items():
        (attempt / f"{plane}.u8").write_bytes(b"\0" * (size * frames))
    rows = []
    for index in range(frames):
        flags = 0
        if index in set(death_at):
            flags |= 0x4
        if index in set(bomb_at):
            flags |= 0x1
        if index in set(hit_at):
            flags |= 0x2
        row = {
            "i": index,
            "f": index,
            "go": 0,
            "ev": flags,
            "msk": 1,
            "li": (lives[index] if lives else 2),
            "bo": (bombs[index] if bombs else 3),
            "sc": (score[index] if score else 100 + index),
            "gr": 0,
            "po": 0,
            "ti": 0,
            "pw": 0,
            "dx": 0,
            "px": 0.0,
            "py": 0.0,
            "ps": 0,
            "death": bool(flags & 0x4),
            "bomb": bool(flags & 0x1),
            "hit": bool(flags & 0x2),
            "dying": bool(flags & 0x4),
        }
        rows.append(json.dumps(row, separators=(",", ":")))
    payload = "\n".join(rows) + "\n"
    if torn_tail:
        payload += '{"i": 999, "f": 999, "de'
    (attempt / "labels.jsonl").write_text(payload, encoding="utf-8")
    meta = {
        "schema": bulk.DATASET_SCHEMA,
        "shard": name,
        "attempt": "attempt-01",
        "spec": {"source": source, "stage": stage, "difficulty": 2, "seed": seed,
                 "frames": frames, "hold": 6, "bomb_prob": 0.0, "warmup": 400,
                 "lp_threads": 4, "tag": "", "key": name},
        "engine_sha256": engine,
        "label_version": label_version,
        "complete": True,
        "frames_written": frames,
        "counts": {"death": len(death_at), "bomb": len(bomb_at), "hit": len(hit_at)},
    }
    (attempt / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {
        "shard": name,
        "attempt": "attempt-01",
        "dir": str(attempt),
        "frames": frames,
        "stage": stage,
        "difficulty": 2,
        "seed": seed,
        "source": source,
        "complete": True,
        "engine_sha256": engine,
        "label_version": label_version,
    }


def write_index(root: Path, rows: list[dict]) -> Path:
    path = root / "index.json"
    path.write_text(
        json.dumps(
            {
                "schema": det.INDEX_SCHEMA,
                "engine_sha256": rows[0]["engine_sha256"] if rows else None,
                "label_version": det.LABEL_VERSION,
                "frames_total": sum(row["frames"] for row in rows),
                "shards": rows,
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def test_wilson_and_rule_of_three_bounds():
    assert det.wilson_lower_bound(0, 0) == 0.0
    assert det.rule_of_three_lower_bound(0) == 0.0
    # 3 failures out of n: 1 - 3/n is exactly the rule-of-three bound
    assert det.rule_of_three_lower_bound(300) == pytest.approx(0.99, abs=1e-12)
    assert det.rule_of_three_lower_bound(1000) == pytest.approx(0.997, abs=1e-12)
    assert det.rule_of_three_lower_bound(299) < 0.99 < det.rule_of_three_lower_bound(301)
    # Wilson is monotone in the number of successes at a fixed n
    bounds = [det.wilson_lower_bound(k, 100) for k in (90, 95, 99, 100)]
    assert bounds == sorted(bounds)
    # closed-form check for a perfect proportion: p - z^2/(n+z^2) * (1 + ...) -> 0.9630
    assert det.wilson_lower_bound(100, 100) == pytest.approx(0.9630, abs=5e-4)


def test_proportion_methods_and_gates():
    perfect = det.proportion(299, 299, zero_errors=True)
    assert perfect["method"] == "rule-of-three"
    assert perfect["value"] == 1.0
    assert perfect["pass_99"] is False  # 1 - 3/299 < 0.99, honestly reported
    assert det.proportion(300, 300, zero_errors=True)["pass_99"] is True
    measured = det.proportion(999, 1000)
    assert measured["method"] == "wilson"
    assert measured["pass_995"] is False


def test_binary_scores_counts():
    scores = det.binary_scores(tp=90, fp=10, fn=10)
    assert scores["precision"]["value"] == pytest.approx(0.9)
    assert scores["recall"]["value"] == pytest.approx(0.9)
    assert scores["precision"]["n"] == 100
    assert scores["recall"]["n"] == 100
    assert scores["f1"] == pytest.approx(0.9)


def test_event_starts_and_matching_are_one_to_one():
    assert det.event_starts([0, 1, 1, 0, 0, 1]) == [1, 5]
    assert det.event_starts([]) == []
    match = det.match_events([10, 40], [8, 12, 55], before=10, after=10)
    assert match["true_events"] == 2
    assert match["matched"] == 1          # 40 is 25 frames from 55 -> outside
    assert match["latencies"] == [-2]     # 8 detects the event at 10, two frames early
    assert match["unmatched_predictions"] == 2
    # one prediction can never cover two events
    assert det.match_events([10, 12], [11], before=10, after=10)["matched"] == 1


def test_latency_summary_gate():
    summary = det.latency_summary([0, 3, 10])
    assert summary["median"] == 3 and summary["max"] == 10
    assert summary["within_limit"] is True
    assert det.latency_summary([0, 11])["within_limit"] is False
    assert det.latency_summary([])["within_limit"] is None


# ---------------------------------------------------------------------------
# storage / dataset
# ---------------------------------------------------------------------------
def test_torn_tail_keeps_only_whole_frames(tmp_path):
    row = write_shard(tmp_path, name="dive-s0d2-seed1-n40", stage=0, seed=1,
                      frames=40, torn_tail=True)
    attempt = Path(row["dir"])
    labels = bulk.read_labels(attempt)
    assert len(labels) == 40
    assert bulk.shard_frame_count(attempt) == 40


def test_shard_frame_count_is_limited_by_planes(tmp_path):
    row = write_shard(tmp_path, name="dive-s0d2-seed2-n40", stage=0, seed=2, frames=40)
    attempt = Path(row["dir"])
    (attempt / "play96.u8").write_bytes(b"\0" * 1)  # truncated plane
    assert bulk.shard_frame_count(attempt) == 0


def test_index_roundtrip_and_split_has_no_leakage(tmp_path):
    rows = [
        write_shard(tmp_path, name="dive-s0-seed1-n40", stage=0, seed=1, frames=40,
                    death_at=(10, 20)),
        write_shard(tmp_path, name="dive-s5-seed2-n40", stage=5, seed=2, frames=40),
        write_shard(tmp_path, name="dive-s6-seed3-n40", stage=6, seed=3, frames=40),
    ]
    index = det.load_index(write_index(tmp_path, rows))
    entries = det.shard_entries(index)
    split = det.build_split(entries, stage_splits={"train": (0,), "val": (5,), "test": (6,)})
    assert split["frames"] == {"train": 40, "val": 40, "test": 40}
    seen: dict[str, str] = {}
    for row in split["shards"]:
        key = f"{row['shard']}/{row['attempt']}"
        assert key not in seen, "a shard appears in two splits"
        seen[key] = row["split"]
    assert split["engine_sha256"] == "a" * 64


def test_split_refuses_unassigned_stage(tmp_path):
    rows = [write_shard(tmp_path, name="dive-s0-seed1-n10", stage=0, seed=1, frames=10)]
    entries = det.shard_entries(det.load_index(write_index(tmp_path, rows)))
    with pytest.raises(det.DetectorDataError, match="assigned to no split"):
        det.build_split(entries, stage_splits={"train": (1,), "val": (2,), "test": (3,)})


def test_split_refuses_mixed_engine_sha(tmp_path):
    rows = [
        write_shard(tmp_path, name="dive-s0-seed1-n10", stage=0, seed=1, frames=10),
        write_shard(tmp_path, name="dive-s6-seed2-n10", stage=6, seed=2, frames=10,
                    engine="b" * 64),
    ]
    entries = det.shard_entries(det.load_index(write_index(tmp_path, rows)))
    with pytest.raises(det.DetectorDataError, match="mixes engine SHAs"):
        det.build_split(entries, stage_splits={"train": (0,), "val": (5,), "test": (6,)})


def test_split_refuses_empty_split(tmp_path):
    rows = [write_shard(tmp_path, name="dive-s0-seed1-n10", stage=0, seed=1, frames=10)]
    entries = det.shard_entries(det.load_index(write_index(tmp_path, rows)))
    with pytest.raises(det.DetectorDataError, match="has no frames"):
        det.build_split(entries, stage_splits={"train": (0,), "val": (5,), "test": (6,)})


def test_load_index_fail_closed(tmp_path):
    with pytest.raises(det.DetectorDataError, match="not found"):
        det.load_index(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "wrong", "label_version": det.LABEL_VERSION}))
    with pytest.raises(det.DetectorDataError, match="schema"):
        det.load_index(bad)
    version = tmp_path / "v.json"
    version.write_text(json.dumps({"schema": det.INDEX_SCHEMA, "label_version": "p2-s4-1"}))
    with pytest.raises(det.DetectorDataError, match="label_version"):
        det.load_index(version)


def test_shard_entries_reject_short_planes(tmp_path):
    row = write_shard(tmp_path, name="dive-s0-seed9-n40", stage=0, seed=9, frames=40)
    index = json.loads(write_index(tmp_path, [row]).read_text())
    index["shards"][0]["frames"] = 999  # index claims more than the planes hold
    with pytest.raises(det.DetectorDataError, match="whole frames"):
        det.shard_entries(index)


# ---------------------------------------------------------------------------
# targets / sampling
# ---------------------------------------------------------------------------
def test_death_label_is_one_frame_ahead(tmp_path):
    row = write_shard(tmp_path, name="dive-s0-seed1-n10", stage=0, seed=1, frames=10,
                      death_at=(5,))
    labels = bulk.read_labels(Path(row["dir"]))
    targets = det.build_targets([labels], [(0, frame) for frame in range(9)])
    # frame 4 predicts the death that happens at frame 5
    assert targets["death"][4] == 1
    assert targets["death"].sum() == 1


def test_targets_encode_score_sign_and_range(tmp_path):
    row = write_shard(tmp_path, name="dive-s0-seed1-n6", stage=0, seed=1, frames=6,
                      score=(100, 150, 150, 90, 90, 200), lives=(2, 2, 2, 2, 2, 9),
                      bombs=(3, 3, 3, 3, 3, 3))
    labels = bulk.read_labels(Path(row["dir"]))
    targets = det.build_targets([labels], [(0, frame) for frame in range(6)])
    # 0 = fell, 1 = same, 2 = rose; the last frame's successor is itself -> same
    assert targets["score"].tolist() == [2, 1, 0, 1, 2, 1]
    assert targets["out_of_range"]["lives"] == 1


def test_sampling_keeps_every_death_positive(tmp_path):
    rows, labels = [], []
    for index, stage in enumerate((0, 1)):
        row = write_shard(tmp_path, name=f"dive-s{stage}-seed{index}-n40", stage=stage,
                          seed=index, frames=40, death_at=(5, 25))
        rows.append(row)
        labels.append(bulk.read_labels(Path(row["dir"])))
    sampling = det.sample_indices(labels, max_samples=8, rng_seed=0)
    assert sampling["positives"] == 4
    assert len(sampling["samples"]) <= 8
    picked = {pair for pair in sampling["samples"]}
    expected = {(0, 4), (0, 24), (1, 4), (1, 24)}
    assert expected <= picked


def test_build_arrays_shapes(tmp_path):
    row = write_shard(tmp_path, name="dive-s0-seed1-n20", stage=0, seed=1, frames=20)
    rows = [row]
    samples = [(0, frame) for frame in range(0, 20, 2)]
    arrays = det.build_arrays(rows, samples, resolution="play64")
    assert arrays["play"].shape == (10, det.STACK, 64, 64)
    assert arrays["hud"].shape == (10, *det.HUD_SHAPE)
    assert arrays["play"].dtype.name == "uint8"


# ---------------------------------------------------------------------------
# model / verdict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("resolution", ["play96", "play64"])
def test_model_forward_shapes_and_param_parity(resolution):
    torch = pytest.importorskip("torch")
    shape = bulk.plane_shapes()[resolution]
    model = det.build_model(play_shape=shape, hud_shape=det.HUD_SHAPE)
    out = model(
        torch.zeros(2, det.STACK, *shape),
        torch.zeros(2, 1, *det.HUD_SHAPE),
    )
    assert out["death"].shape == (2, 2)
    assert out["bomb"].shape == (2, 2)
    assert out["lives"].shape == (2, det.LIVES_CLASSES)
    assert out["bombs"].shape == (2, det.BOMBS_CLASSES)
    assert out["score"].shape == (2, det.SCORE_CLASSES)
    assert det.parameter_count(model) > 100_000


def test_verdict_marks_unmet_gates():
    block = {
        "threshold": 0.5,
        "death_frame_level": {"tp": 50, "fp": 500, "fn": 50,
                              "precision": det.proportion(50, 550),
                              "recall": det.proportion(50, 100),
                              "f1": 0.15},
        "death_event_level": {
            "true_events": 40, "pred_events": 900, "matched": 38,
            "matched_predictions": 38,
            "recall": det.proportion(38, 40), "precision": det.proportion(38, 900),
            "latency": det.latency_summary([0, 1, 2]),
        },
        "lives_accuracy": det.proportion(999, 1000),
        "bombs_accuracy": det.proportion(1000, 1000),
        "score_sign_accuracy": {**det.proportion(1000, 1000), "negative_class_samples": 0},
    }
    verdict = det.g2_verdict(block)
    gates = {row["gate"]: row for row in verdict["gates"]}
    assert gates["death recall (frame level)"]["pass"] is False
    assert gates["death precision (event level)"]["pass"] is False
    assert gates["lives read accuracy"]["pass"] is False
    assert gates["bombs read accuracy"]["pass"] is True
    assert gates["score delta sign accuracy"]["pass"] is True
    assert verdict["all_pass"] is False
    assert verdict["score_negative_samples"] == 0


def test_report_renders_every_gate(tmp_path):
    metrics = {
        "schema": det.METRICS_SCHEMA,
        "resolution": "play96",
        "verdict": det.g2_verdict(
            {
                "threshold": 0.5,
                "death_frame_level": {"precision": det.proportion(100, 100),
                                      "recall": det.proportion(100, 100),
                                      "tp": 100, "fp": 0, "fn": 0, "f1": 1.0},
                "death_event_level": {
                    "true_events": 100, "pred_events": 100, "matched": 100,
                    "matched_predictions": 100,
                    "recall": det.proportion(100, 100),
                    "precision": det.proportion(100, 100),
                    "latency": det.latency_summary([0, 1]),
                },
                "lives_accuracy": det.proportion(1000, 1000),
                "bombs_accuracy": det.proportion(1000, 1000),
                "score_sign_accuracy": {**det.proportion(1000, 1000),
                                        "negative_class_samples": 12},
            }
        ),
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(metrics))
    out = tmp_path / "g2.md"
    assert det.main(["report", "--metrics", str(path), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "death latency" in text
    assert "\u8fbe\u6807" in text

# ---------------------------------------------------------------------------
# EXP-P2-004 v1: HUD-sequence event heads, digit classification, latency
# ---------------------------------------------------------------------------
def test_hud_diff_events_fire_on_every_lives_decrement():
    """A lives decrement is the death event signal; equality/up is never one."""

    assert det.hud_diff_events([2, 2, 1, 1, 0, 0]) == [
        False, False, True, False, True, False]
    assert det.hud_diff_events([0, 0, 0]) == [False, False, False]
    # a life *gained* is not a death
    assert det.hud_diff_events([1, 2]) == [False, False]
    assert det.hud_diff_events([]) == []


def test_hud_diff_events_bombs_match_the_bomb_count():
    bombs = [3, 3, 2, 2, 2, 1]
    flags = det.hud_diff_events(bombs)
    assert [i for i, f in enumerate(flags) if f] == [2, 5]


def test_score_glyph_boxes_are_nine_disjoint_in_plane_cells():
    boxes = det.score_glyph_boxes()
    assert len(boxes) == det.HUD_SCORE_DIGITS == 9
    for r0, r1, c0, c1 in boxes:
        assert 0 <= r0 < r1 <= det.HUD_SHAPE[0]
        assert 0 <= c0 < c1 <= det.HUD_SHAPE[1]
        assert c1 - c0 == det.HUD_SCORE_PITCH
    # consecutive glyph cells never overlap and cover one pitch each
    for left, right in zip(boxes, boxes[1:]):
        assert left[3] == right[2]


def test_digit_roundtrip_and_nine_digit_format():
    for value in (0, 7, 119687, 999_999_999):
        digits = det.digits_of(value)
        assert len(digits) == 9
        assert det.decode_digits(digits) == value
    assert det.digits_of(119687) == [0, 0, 0, 1, 1, 9, 6, 8, 7]
    assert det.digits_of(-5) == [0] * 9
    assert det.digits_of(2_000_000_000) == [9] * 9


def test_sign_series_three_classes():
    assert det.sign_series([5, 5, 9, 3, 3, 4]) == [0, 0, 1, -1, 0, 1]


def test_causal_windows_never_look_ahead():
    np = pytest.importorskip("numpy")
    vector = np.arange(20, dtype="float32").reshape(10, 2)
    windows = det.causal_windows(vector, history=3)
    assert windows.shape == (10, 4, 2)
    before = windows[5].copy()
    mutated = vector.copy()
    mutated[6:] += 1000.0          # everything strictly after frame 5
    after = det.causal_windows(mutated, history=3)[5]
    assert np.array_equal(before, after)


def test_death_hud_lag_is_inside_the_latency_gate():
    """death at t, lives decrement at t+4: event recall 1.0, max latency 4."""

    true = [False] * 40
    true[10] = True
    pred = [False] * 40
    pred[14] = True               # HUD reading changed four frames later
    metrics = det.event_metrics(true, pred, max_latency=10)
    assert metrics["event_level"]["recall"]["value"] == 1.0
    assert metrics["frame_level"]["recall"]["value"] == 0.0  # not the same frame
    assert metrics["latency"]["max"] == 4
    assert metrics["latency"]["within_limit"] is True


def test_latency_gate_fails_past_ten_frames():
    true = [False] * 40
    true[5] = True
    pred = [False] * 40
    pred[20] = True               # 15 frames late
    metrics = det.event_metrics(true, pred, max_latency=10)
    assert metrics["event_level"]["recall"]["value"] == 0.0
    assert metrics["latency"]["n"] == 0


def test_event_metrics_precision_counts_false_alarms():
    true = [False] * 30
    for t in (5, 25):
        true[t] = True
    pred = [False] * 30
    for t in (9, 12, 29):
        pred[t] = True
    metrics = det.event_metrics(true, pred, max_latency=10)
    # 5 -> 9 matched, 25 -> 29 matched, 12 is a false alarm
    assert metrics["event_level"]["recall"]["value"] == 1.0
    assert metrics["event_level"]["precision"]["value"] == pytest.approx(2 / 3)
    assert metrics["frames"]["fp"] == 3   # all three predictions are off-frame


def test_reader_accuracy_separates_settled_from_rolling_score():
    """The label is globals.score; the drawn counter rolls, so sign drifts."""

    np = pytest.importorskip("numpy")
    n = 60
    score = np.full(n, 1000, dtype="int64")
    score[40:] = 1075                      # one true jump at t=40
    digits = np.stack([np.asarray(det.digits_of(int(v))) for v in score])
    # the picture lags: it shows the old value for a while, then catches up
    shown = np.stack([np.asarray(det.digits_of(int(v))) for v in score])
    shown[40:50] = np.asarray(det.digits_of(1000))
    readings = {
        "lives": np.eye(det.LIVES_CLASSES)[np.full(n, 2)],
        "bombs": np.eye(det.BOMBS_CLASSES)[np.full(n, 3)],
        "digits": shown,
        "decoded": np.asarray([det.decode_digits(row) for row in shown]),
    }
    labels = {"lives": np.full(n, 2), "bombs": np.full(n, 3), "score": score}
    result = det.reader_accuracy(readings, labels)
    assert result["lives"]["value"] == 1.0
    assert result["bombs"]["value"] == 1.0
    assert result["score_integer"]["value"] == pytest.approx(50 / 60)
    assert result["score_sign_negative_class_samples"] == 0
    assert result["score_settled_frames"] + result["score_rolling_frames"] == n
    # every settled frame is read correctly, the misses are inside the roll
    assert result["score_integer_settled"]["value"] >= result["score_integer"]["value"]


def test_v1_verdict_covers_every_g2_gate():
    block = {
        "death": det.event_metrics([False, True, False, True], [False, False, True, True]),
        "bomb": det.event_metrics([False, True, False, False], [False, True, False, False]),
        "readers": {
            "lives": det.proportion(10, 10),
            "bombs": det.proportion(10, 10),
            "score_digit_accuracy": det.proportion(90, 90),
            "score_integer": det.proportion(10, 10),
            "score_sign": det.proportion(10, 10),
            "score_integer_settled": det.proportion(10, 10),
            "score_sign_settled": det.proportion(10, 10),
            "score_sign_negative_class_samples": 0,
        },
    }
    verdict = det.g2_verdict_v1(block)
    gates = {row["gate"] for row in verdict["gates"]}
    for required in (
        "death recall (frame level)", "death precision (frame level)",
        "death recall (event level)", "death precision (event level)",
        "death latency (max, frames)", "bomb recall (event level)",
        "lives read accuracy", "bombs read accuracy",
        "score delta sign accuracy",
    ):
        assert required in gates
    # one prediction for two true events -> recall 0.5, so the gate must fail
    assert verdict["all_pass"] is False
    assert verdict["score_negative_samples"] == 0


def test_v1_report_marks_the_untestable_negative_class(tmp_path):
    block = {
        "death": det.event_metrics([True], [True]),
        "bomb": det.event_metrics([True], [True]),
        "readers": {
            "lives": det.proportion(10, 10),
            "bombs": det.proportion(10, 10),
            "score_digit_accuracy": det.proportion(90, 90),
            "score_integer": det.proportion(10, 10),
            "score_sign": det.proportion(10, 10),
            "score_integer_settled": det.proportion(10, 10),
            "score_sign_settled": det.proportion(10, 10),
            "score_sign_negative_class_samples": 0,
        },
    }
    metrics = {
        "schema": det.METRICS_SCHEMA + "-v1",
        "resolution": "hud-sequence",
        "verdict": det.g2_verdict_v1(block),
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    out = tmp_path / "g2.md"
    assert det.main(["report-v1", "--metrics", str(path), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "death latency" in text
    assert "score \u8d1f\u5411\u6837\u672c" in text
    assert "\u4e0d\u53ef\u6d4b" in text


# ---------------------------------------------------------------------------
# EXP-P2-005: displayScore labels, measured cell origin, per-shard sign
# ---------------------------------------------------------------------------
def test_score_glyph_boxes_measured_origin_matches_the_gui_anchor():
    """Gui.cpp:1299 draws at x=488 -> band column 8, so the measured boxes start
    there; the legacy EXP-P2-004 boxes start five pixels to the left."""

    measured = det.score_glyph_boxes(det.HUD_SCORE_X0_MEASURED)
    assert det.HUD_SCORE_X0_MEASURED == 8
    assert measured[0] == (det.HUD_SCORE_ROWS[0], det.HUD_SCORE_ROWS[1], 8, 21)
    assert measured[-1] == (det.HUD_SCORE_ROWS[0], det.HUD_SCORE_ROWS[1], 112, 125)
    assert measured[-1][3] <= det.HUD_SHAPE[1]
    legacy = det.score_glyph_boxes()
    assert legacy[0][2] == det.HUD_SCORE_X0 == 3
    assert det.HUD_SCORE_X0_MEASURED - det.HUD_SCORE_X0 == 5


def test_hud_crops_honours_the_cell_origin():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    batch = np.zeros((1, *det.HUD_SHAPE), dtype=np.uint8)
    batch[0, det.HUD_SCORE_ROWS[0]:det.HUD_SCORE_ROWS[1], 3:8] = 255   # left of the glyph
    legacy = det.hud_crops(batch)["score"]
    measured = det.hud_crops(batch, score_x0=det.HUD_SCORE_X0_MEASURED)["score"]
    assert float(legacy[0, 0].max()) == 1.0      # legacy cell [3, 16) sees it
    assert float(measured[0, 0].max()) == 0.0    # measured cell [8, 21) does not


def test_score_accuracy_counts_digits_integer_and_sign():
    np = pytest.importorskip("numpy")
    labels = np.asarray([100, 100, 100, 150, 150], dtype="int64")
    digits = np.stack([np.asarray(det.digits_of(int(v))) for v in labels])
    result = det.score_accuracy(labels.copy(), digits, labels, [(0, 5)])
    assert result["digit_accuracy"]["value"] == 1.0
    assert result["integer_accuracy"]["value"] == 1.0
    assert result["sign_accuracy"]["value"] == 1.0
    assert result["sign_true_counts"] == {"down": 0, "same": 4, "up": 1}

    broken = digits.copy()
    broken[2, 8] = (broken[2, 8] + 1) % 10          # one wrong digit on frame 2
    decoded = np.asarray([det.decode_digits(row) for row in broken], dtype="int64")
    result = det.score_accuracy(decoded, broken, labels, [(0, 5)])
    assert result["integer_accuracy"]["value"] == pytest.approx(4 / 5)
    assert result["digit_accuracy"]["value"] == pytest.approx(44 / 45)
    assert result["per_digit"][8]["value"] == pytest.approx(4 / 5)


def test_score_accuracy_computes_signs_inside_each_shard():
    """A new shard starts its score at 0, so a flat series invents a ``-1``."""

    np = pytest.importorskip("numpy")
    labels = np.asarray([100] * 5 + [0] * 5, dtype="int64")
    digits = np.stack([np.asarray(det.digits_of(int(v))) for v in labels])
    result = det.score_accuracy(labels.copy(), digits, labels, [(0, 5), (5, 10)])
    assert result["sign_true_counts"]["down"] == 0
    assert result["sign_true_counts"]["same"] == 10
    assert result["sign_accuracy"]["value"] == 1.0
    # the artifact the per-shard computation removes, kept visible as evidence
    assert det.sign_series(labels).count(-1) == 1


def test_load_display_score_reads_relabel_rows_and_marks_gaps(tmp_path):
    np = pytest.importorskip("numpy")
    shard_root = tmp_path / "relabel" / "shards"
    shard_root.mkdir(parents=True)
    rows = [
        {"i": 0, "ds": 10, "gs": 12, "age": 2},
        {"i": 1, "ds": 11, "gs": 12, "age": 1},
        {"i": 2, "ds": 12, "gs": 12, "age": 0},
    ]
    (shard_root / "s1.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    shards = [{"shard": "s1", "dir": str(tmp_path / "x")}]
    display, globals_score, ages = det.load_display_score(
        tmp_path / "relabel", shards, limits=[4]
    )
    assert display.tolist() == [10, 11, 12, -1]
    assert globals_score.tolist() == [12, 12, 12, -1]
    assert ages.tolist() == [2, 1, 0, -1]

    with pytest.raises(det.DetectorDataError, match="no displayScore rows"):
        det.load_display_score(tmp_path / "relabel", [{"shard": "missing", "dir": "x"}], limits=[1])


def test_pixel_consistency_gate_needs_the_wilson_lower_bound():
    """100 % on 1000 frames is enough for a 99.5 % lower bound; 99.4 % is not."""

    perfect = det.proportion(1000, 1000)
    assert perfect["lower_bound_95"] > 0.995
    almost = det.proportion(994, 1000)
    assert almost["lower_bound_95"] < 0.995
    assert det.proportion(995, 1000)["lower_bound_95"] < 0.995


# ---------------------------------------------------------------------------
# EXP-P2-006: observable-scope death gate, game_over gate, two-column verdict
# ---------------------------------------------------------------------------
def _death_shard(observable: int, hidden: int) -> dict:
    """A shard with ``observable`` lives>0 deaths and ``hidden`` lives==0 ones."""

    li = [3]
    death = [False]
    for _ in range(observable):
        death.extend([True, False, False, False, False])
        li.extend([li[-1]] * 5)
        li[-1] = li[-2] - 1          # the +4 frame decrement the HUD shows
    for _ in range(hidden):
        death.extend([True, False, False, False, False])
        li.extend([li[-1]] * 5)      # lives already 0: nothing moves
    return {"li": li, "death": death}


def test_observable_death_scopes_separate_the_ceiling_from_the_gate():
    shards = [_death_shard(2, 1) for _ in range(10)]
    block = det.observable_death_scopes(shards)
    assert block["events"] == 30
    assert block["observable_events"] == 20
    assert block["unobservable_events"] == 10
    assert block["predictions"] == 20
    # every pixel-visible death is caught, the hidden ones are out of scope
    assert block["observable"]["recall"]["value"] == 1.0
    assert block["observable"]["precision"]["value"] == 1.0
    # ... but in the original scope the ceiling is 20/30
    assert block["as_written"]["recall"]["value"] == pytest.approx(2 / 3)
    assert block["latency"]["n"] == 20
    assert block["latency"]["max"] == 4


def test_two_scope_verdict_requires_350_observable_events():
    """Wilson LB >= 0.99 needs n >= 386 at zero failures, so 350 is the floor."""

    small = det.observable_death_scopes([_death_shard(2, 0) for _ in range(150)])
    assert small["observable_events"] == 300
    verdict = det.g2_verdict_v2(
        {
            "death_scopes": small,
            "bomb": det.event_metrics([True] * 400, [True] * 400)["event_level"],
            "readers": {
                "lives": det.proportion(1000, 1000),
                "bombs": det.proportion(1000, 1000),
                "score_sign": det.proportion(1000, 1000),
            },
            "game_over": det.game_over_metrics([{"hgo": [0, 1, 1, 0]}]),
        }
    )
    sample_row = {row["gate"]: row for row in verdict["rows"]}["death observable sample size"]
    assert sample_row["amended"]["n"] == 300
    assert sample_row["amended"]["pass"] is False
    assert verdict["amended_pass"] is False
    assert verdict["verdict"]["amended"] == "未达"

    big = det.observable_death_scopes([_death_shard(2, 0) for _ in range(200)])
    assert big["observable_events"] == 400
    assert big["observable"]["recall"]["lower_bound_95"] > 0.99
    assert big["observable"]["precision"]["lower_bound_95"] > 0.99


def test_game_over_gate_counts_onsets_and_fails_closed_without_samples():
    shards = [
        {"hgo": [0, 0, 1, 1, 1, 0], "go_pred": [0, 0, 0, 1, 1, 0]},   # onset at 2
        {"hgo": [0, 1, 1], "go_pred": [0, 1, 1]},                     # onset at 1
    ]
    block = det.game_over_metrics(shards)
    assert block["events"] == 2
    assert block["predictions"] == 2
    assert block["matched"] == 2
    assert block["measured"] is True
    assert block["recall"]["value"] == 1.0
    assert block["recall"]["n"] == 2
    assert block["recall"]["lower_bound_95"] < 0.99      # n=2 can never gate

    empty = det.game_over_metrics([{"hgo": [0, 0, 0], "go_pred": [0, 0, 0]}])
    assert empty["measured"] is False
    assert empty["recall"]["n"] == 0
    # a detector that never fires is not a 100 % detector
    silent = det.game_over_metrics([{"hgo": [0, 1, 1], "go_pred": [0, 0, 0]}])
    assert silent["recall"]["value"] == 0.0
    assert silent["precision"]["n"] == 0


def test_v2_verdict_marks_game_over_as_out_of_scope_for_the_written_criteria():
    block = {
        "death_scopes": det.observable_death_scopes([_death_shard(2, 1) for _ in range(200)]),
        "bomb": det.event_metrics([True, False] * 400, [True, False] * 400)["event_level"],
        "readers": {
            "lives": det.proportion(1000, 1000),
            "bombs": det.proportion(1000, 1000),
            "score_sign": det.proportion(1000, 1000),
        },
        "game_over": det.game_over_metrics([{"hgo": [0, 1, 1]}, {"hgo": [0, 0, 1]}]),
    }
    verdict = det.g2_verdict_v2(block)
    rows = {row["gate"]: row for row in verdict["rows"]}
    # the original G2 criteria (12-P2 §6) have no game_over row: not applicable
    assert rows["game_over recall (event level)"]["as_written"]["applicable"] is False
    assert rows["game_over recall (event level)"]["amended"]["applicable"] is True
    # as written the death recall ceiling bites: 400/600 observable
    assert rows["death recall (event level)"]["as_written"]["pass"] is False
    assert rows["death recall (event level)"]["amended"]["pass"] is True
    assert rows["bomb recall (event level)"]["amended"]["pass"] is True
    assert verdict["as_written_pass"] is False
    # game_over has only 2 events -> its own 99 % lower bound cannot be met
    assert rows["game_over recall (event level)"]["amended"]["pass"] is False
    assert verdict["amended_pass"] is False
    assert verdict["verdict"]["signed"] is False
    assert set(verdict["verdict"]) == {"as_written", "amended", "signed"}


def test_v2_verdict_passes_the_amended_column_only_when_every_row_clears():
    """400 observable deaths + 400 game-over onsets: the amendment would pass."""

    block = {
        "death_scopes": det.observable_death_scopes([_death_shard(2, 0) for _ in range(200)]),
        "bomb": det.event_metrics([True, False] * 400, [True, False] * 400)["event_level"],
        "readers": {
            # readers need n >= 769 for a 0.995 Wilson lower bound at zero failures
            "lives": det.proportion(1000, 1000),
            "bombs": det.proportion(1000, 1000),
            "score_sign": det.proportion(1000, 1000),
        },
        "game_over": det.game_over_metrics([{"hgo": [1, 0] * 400, "go_pred": [1, 0] * 400}]),
    }
    verdict = det.g2_verdict_v2(block)
    assert verdict["scope"]["game_over_events"] == 400
    assert verdict["as_written_pass"] is True     # 100 % is reachable with no hidden deaths
    assert verdict["amended_pass"] is True
    assert verdict["verdict"]["amended"] == "PASS-with-amendment（待负责人签署）"
    assert verdict["verdict"]["signed"] is False



def test_v2_verdict_fails_closed_on_zero_samples():
    block = {
        "death_scopes": det.observable_death_scopes([{"li": [1, 1, 1], "death": [False] * 3}]),
        "bomb": det.event_metrics([False] * 10, [False] * 10)["event_level"],
        "readers": {
            "lives": det.proportion(0, 0),
            "bombs": det.proportion(0, 0),
            "score_sign": det.proportion(0, 0),
        },
        "game_over": det.game_over_metrics([{"hgo": [0, 0]}]),
    }
    verdict = det.g2_verdict_v2(block)
    for row in verdict["rows"]:
        if row["amended"]["applicable"]:
            assert row["amended"]["pass"] is False, row["gate"]
    assert verdict["amended_pass"] is False
    assert verdict["verdict"]["as_written"] == "未达"
    assert verdict["verdict"]["amended"] == "未达"


def test_report_v2_renders_both_columns(tmp_path):
    block = {
        "death_scopes": det.observable_death_scopes([_death_shard(2, 1) for _ in range(200)]),
        "bomb": det.event_metrics([True, False] * 400, [True, False] * 400)["event_level"],
        "readers": {
            "lives": det.proportion(1000, 1000),
            "bombs": det.proportion(1000, 1000),
            "score_sign": det.proportion(1000, 1000),
        },
        "game_over": det.game_over_metrics([{"hgo": [0, 1, 1]}]),
    }
    metrics = {"schema": det.METRICS_SCHEMA + "-v2", "block": block,
               "verdict": det.g2_verdict_v2(block)}
    path = tmp_path / "m2.json"
    path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "g2-v2.md"
    assert det.main(["report-v2", "--metrics", str(path), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "按原文口径" in text
    assert "限定口径" in text
    assert "game_over" in text
    assert "PASS-with-amendment" not in text  # fail-closed: not all amended rows pass

    block["game_over"] = det.game_over_metrics(
        [{"hgo": [1, 0] * 400, "go_pred": [1, 0] * 400}])
    metrics = {"schema": det.METRICS_SCHEMA + "-v2", "block": block,
               "verdict": det.g2_verdict_v2(block)}
    path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
    assert det.main(["report-v2", "--metrics", str(path), "--out", str(out)]) == 0
    assert "PASS-with-amendment" in out.read_text(encoding="utf-8")


def test_report_v2_exits_nonzero_when_no_scope_block(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"schema": det.METRICS_SCHEMA + "-v2"}), encoding="utf-8")
    out = tmp_path / "g2-empty.md"
    assert det.main(["report-v2", "--metrics", str(path), "--out", str(out)]) == 2
    assert "未达" in out.read_text(encoding="utf-8")


def test_collector_hud_flag_changes_the_shard_identity_and_roundtrips():
    plain = bulk.ShardSpec(source="dive", stage=3, difficulty=3, seed=9200, frames=620)
    hud = bulk.ShardSpec(source="dive", stage=3, difficulty=3, seed=9200, frames=620,
                         hud=True)
    assert "hud" not in plain.key and hud.key.endswith("-hud")
    assert hud.key != plain.key
    restored = bulk.ShardSpec(**{k: v for k, v in hud.as_dict().items() if k != "key"})
    assert restored.hud is True
    assert restored.key == hud.key
    # the default P2 path is unchanged: hud defaults off and the key has no suffix
    assert plain.as_dict()["hud"] is False
    assert plain.key == "dive-s3d3-seed9200-n620-bp0.02"



# ---------------------------------------------------------------------------
# EXP-P2-007: bomb gate in two scopes, stage_clear event bit, WP5 online gate
# ---------------------------------------------------------------------------
def _bomb_shard(observable: int, hidden: int) -> dict:
    """A shard with ``observable`` bombs that spend a bomb and ``hidden`` that do not."""

    bo = [3]
    bomb = [False]
    for _ in range(observable):
        bomb.extend([True, False, False, False, False])
        bo.extend([bo[-1]] * 5)
        bo[-1] = bo[-2] - 1          # the +4 frame decrement the HUD shows
    for _ in range(hidden):
        bomb.extend([True, False, False, False, False])
        bo.extend([bo[-1]] * 5)      # bombs already 0 / press refused
    return {"bo": bo, "bomb": bomb}


def test_observable_bomb_scopes_separate_the_physical_ceiling():
    shards = [_bomb_shard(2, 1) for _ in range(10)]
    block = det.observable_bomb_scopes(shards)
    assert block["events"] == 30
    assert block["observable_events"] == 20
    assert block["unobservable_events"] == 10
    assert block["predictions"] == 20
    assert block["observable"]["recall"]["value"] == 1.0
    assert block["observable"]["precision"]["value"] == 1.0
    assert block["as_written"]["recall"]["value"] == pytest.approx(2 / 3)
    assert block["latency"]["max"] == 4
    assert "counter" in block["detector"]


def test_bomb_gate_needs_386_observable_events_for_a_99_percent_lower_bound():
    small = det.observable_bomb_scopes([_bomb_shard(2, 0) for _ in range(100)])
    assert small["observable_events"] == 200
    assert small["observable"]["recall"]["lower_bound_95"] < 0.99
    verdict = det.g2_verdict_v2(
        {
            "death_scopes": det.observable_death_scopes([_death_shard(2, 0) for _ in range(200)]),
            "bomb_scopes": small,
            "readers": {
                "lives": det.proportion(1000, 1000),
                "bombs": det.proportion(1000, 1000),
                "score_sign": det.proportion(1000, 1000),
            },
            "game_over": det.game_over_metrics([{"hgo": [1, 0] * 400, "go_pred": [1, 0] * 400}]),
        }
    )
    rows = {row["gate"]: row for row in verdict["rows"]}
    assert rows["bomb observable sample size"]["amended"]["n"] == 200
    assert rows["bomb observable sample size"]["amended"]["pass"] is False
    assert verdict["amended_pass"] is False

    big = det.observable_bomb_scopes([_bomb_shard(2, 0) for _ in range(200)])
    assert big["observable_events"] == 400
    assert big["observable"]["recall"]["lower_bound_95"] > 0.99
    assert big["observable"]["precision"]["lower_bound_95"] > 0.99
    verdict = det.g2_verdict_v2(
        {
            "death_scopes": det.observable_death_scopes([_death_shard(2, 0) for _ in range(200)]),
            "bomb_scopes": big,
            "readers": {
                "lives": det.proportion(1000, 1000),
                "bombs": det.proportion(1000, 1000),
                "score_sign": det.proportion(1000, 1000),
            },
            "game_over": det.game_over_metrics([{"hgo": [1, 0] * 400, "go_pred": [1, 0] * 400}]),
        }
    )
    rows = {row["gate"]: row for row in verdict["rows"]}
    assert rows["bomb recall (event level)"]["amended"]["pass"] is True
    assert rows["bomb observable sample size"]["amended"]["pass"] is True


def test_stage_clear_metrics_read_the_events_bit_and_measure_the_proxy_gap():
    """The bit is the label; STATE.f1 only moves once the clear screen ends."""

    bit = det.STAGE_CLEAR_BIT
    assert bit == 0x0080
    # the engine's own event enum has no 0x0080 member: the harness bit cannot collide
    assert bit not in (0x0001, 0x0002, 0x0004, 0x0020, 0x0040,
                       0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000)
    frames = 200
    ev = [0] * frames
    stage = [0] * frames
    for index in range(120, 160):
        ev[index] = bit
    for index in range(160, frames):
        stage[index] = 1
    block = det.stage_clear_metrics([{"ev": ev, "stage": stage}])
    assert block["events"] == 1
    assert block["measured"] is False          # no pixel-side detector exists
    assert block["recall"]["n"] == 0           # never a fabricated 100 %
    assert block["proxy_latency"]["n"] == 1
    assert block["proxy_latency"]["max"] == 40  # STATE.f1 is 40 frames late
    assert "no pixel-side" in block["detector"]


def test_v2_verdict_adds_the_stage_clear_row_only_when_the_bit_exists():
    base = {
        "death_scopes": det.observable_death_scopes([_death_shard(2, 0) for _ in range(200)]),
        "bomb": det.event_metrics([True, False] * 400, [True, False] * 400)["event_level"],
        "readers": {
            "lives": det.proportion(1000, 1000),
            "bombs": det.proportion(1000, 1000),
            "score_sign": det.proportion(1000, 1000),
        },
        "game_over": det.game_over_metrics([{"hgo": [1, 0] * 400, "go_pred": [1, 0] * 400}]),
    }
    without = det.g2_verdict_v2(base)
    assert not any("stage_clear" in row["gate"] for row in without["rows"])
    assert without["amended_pass"] is True     # old metrics files render unchanged

    block = dict(base)
    block["stage_clear"] = det.stage_clear_metrics(
        [{"ev": [det.STAGE_CLEAR_BIT] + [0] * 99}])
    with_bit = det.g2_verdict_v2(block)
    rows = {row["gate"]: row for row in with_bit["rows"]}
    assert "stage_clear recall (event level)" in rows
    # one event can never clear a 99 % lower bound
    assert rows["stage_clear recall (event level)"]["amended"]["pass"] is False
    assert with_bit["amended_pass"] is False


def _online_payload(episodes: int, frames: int, *, perfect: bool = True) -> dict:
    def entry(n: int, successes: int) -> dict:
        return det.proportion(successes, n)

    total = 400 if perfect else 300
    return {
        "schema": "v400-p2-007-wp5-online-1",
        "combined": {
            "episodes": episodes,
            "frames": frames,
            "death": {"recall": entry(total, total), "precision": entry(total, total)},
            "bomb": {"recall": entry(total, total), "precision": entry(total, total)},
            "game_over": {"recall": entry(total, total), "precision": entry(total, total)},
        },
    }


def test_online_validation_schema_needs_three_episodes_and_five_thousand_frames():
    short = det.online_validation_gate(_online_payload(2, 4000))
    assert short["measured"] is True
    assert short["long_enough"] is False
    assert short["pass"] is False
    assert all(row["pass"] is False for row in short["rows"])
    assert "流太短" in short["rows"][0]["note"]

    ok = det.online_validation_gate(_online_payload(3, 5000))
    assert ok["long_enough"] is True
    assert len(ok["rows"]) == 6
    assert ok["pass"] is True

    empty = det.online_validation_gate({})
    assert empty["measured"] is False
    assert empty["pass"] is False
    assert all(row["n"] == 0 and row["pass"] is False for row in empty["rows"])


def test_online_validation_wilson_gate_is_not_a_point_estimate():
    """100 % on 6 events is not a 99 % lower bound; 400 events is."""

    payload = _online_payload(3, 5000)
    payload["combined"]["death"] = {
        "recall": det.proportion(6, 6), "precision": det.proportion(6, 6)}
    few = det.online_validation_gate(payload)
    rows = {row["gate"]: row for row in few["rows"]}
    assert rows["online death recall"]["value"] == 1.0
    assert rows["online death recall"]["pass"] is False
    assert few["pass"] is False

# ---------------------------------------------------------------------------
# EXP-P2-008: pixel-side stage_clear detector (play64 gradient flip)
# ---------------------------------------------------------------------------
def _play64_window(*, edge: int = 60, length: int = 181, top: float = 40.0,
                   bottom: float = 10.0, gameplay_top: float = 10.0,
                   gameplay_bottom: float = 40.0):
    """A synthetic play64 stack: gameplay is bottom-lit, the clear screen top-lit."""

    import numpy as np

    frames = np.zeros((length, 64, 64), dtype=np.uint8)
    frames[:, :, :] = np.linspace(gameplay_top, gameplay_bottom, 64, dtype=np.float32)[None, :, None]
    frames[edge:] = np.linspace(top, bottom, 64, dtype=np.float32)[None, :, None]
    return frames


def _write_stage_clear_event(root: Path, case: str, stage: int, *, edge: int = 60,
                             length: int = 181, play=None) -> Path:
    import numpy as np

    directory = root / f"{case}-s{stage}"
    directory.mkdir(parents=True)
    play = _play64_window(edge=edge, length=length) if play is None else play
    level = np.zeros(length, dtype=np.uint8)
    level[edge:edge + 60] = 1
    np.savez_compressed(directory / "frames.npz", play=play, level=level,
                        hud=np.zeros((length, 56, 160), dtype=np.uint8))
    (directory / "meta.json").write_text(json.dumps({
        "case": case, "stage": stage, "edgeFrame": edge,
        "window": [edge - 60, edge + 120],
        "replaySha256": "0" * 64, "binarySha256": "0" * 64,
    }), encoding="utf-8")
    return directory


def test_stage_clear_vgrad_flips_between_gameplay_and_clear_screen():
    import numpy as np

    gameplay = np.zeros((1, 64, 64), dtype=np.uint8)
    gameplay[0] = np.linspace(10.0, 40.0, 64)[:, None]
    clear = np.zeros((1, 64, 64), dtype=np.uint8)
    clear[0] = np.linspace(40.0, 10.0, 64)[:, None]

    assert det.stage_clear_vgrad(gameplay)[0] < -10
    assert det.stage_clear_vgrad(clear)[0] > 10
    assert bool(det.stage_clear_pixel_flags(gameplay)[0]) is False
    assert bool(det.stage_clear_pixel_flags(clear)[0]) is True

    with pytest.raises(ValueError):
        det.stage_clear_vgrad(np.zeros((2, 96, 112), dtype=np.uint8))


def test_stage_clear_predictions_need_sustain_and_respect_refractory():
    import numpy as np

    window = _play64_window(edge=60, length=181)
    predictions = det.stage_clear_pixel_predictions(window, jump=4.0, floor=3.0,
                                                    sustain=5, refractory=150)
    assert predictions == [60]

    # a one-frame blip must not count as a clear screen
    blip = _play64_window(edge=60, length=181)
    blip[20:21] = np.linspace(40.0, 10.0, 64, dtype=np.float32)[None, :, None]
    assert det.stage_clear_pixel_predictions(blip, jump=4.0, floor=3.0, sustain=5,
                                             refractory=150) == [60]

    # two clear screens closer than the refractory: only the first is reported
    twice = _play64_window(edge=60, length=181)
    twice[80:] = np.linspace(10.0, 40.0, 64, dtype=np.float32)[None, :, None]
    twice[90:] = np.linspace(40.0, 10.0, 64, dtype=np.float32)[None, :, None]
    assert det.stage_clear_pixel_predictions(twice, jump=4.0, floor=3.0, sustain=5,
                                             refractory=150) == [60]


def test_stage_clear_metrics_are_fail_closed_and_size_aware(tmp_path: Path):
    import numpy as np

    perfect = det._stage_clear_eval([{"case": "c1", "stage": 0, "level": np.array([0] * 60 + [1] * 61 + [0] * 60),
                                      "play": _play64_window(edge=60), "edge": 60}])
    assert perfect["events"] == 1 and perfect["matched"] == 1
    assert perfect["recall"]["value"] == 1.0
    assert perfect["precision"]["value"] == 1.0
    assert perfect["latency"]["max"] == 0

    silent = det._stage_clear_eval([{"case": "c1", "stage": 0, "level": np.array([1, 1, 1]),
                                     "play": _play64_window(edge=60)[:3], "edge": 0}])
    assert silent["recall"]["value"] == 0.0

    empty = det._stage_clear_eval([])
    assert empty["events"] == 0
    assert empty["recall"]["n"] == 0
    assert empty["precision"]["n"] == 0

    root = tmp_path / "events"
    # deterministic whole-replay split: pick one name per side, without guessing
    cases = {split: None for split in ("train", "test")}
    for index in range(200):
        name = f"case-{index:03d}"
        side = det.split_stage_clear_cases([name])[name]
        if cases.get(side) is None:
            cases[side] = name
        if cases.get("train") and cases.get("test"):
            break
    _write_stage_clear_event(root, cases["train"], 0)
    _write_stage_clear_event(root, cases["test"], 0)
    block = det.stage_clear_corpus_gate(root)
    assert block["measured"] is True
    # primary numbers are the 3-fold out-of-fold aggregate over both directories
    assert block["events"] == 2
    assert block["test"]["events"] == 1              # the single 70/30 split scores only test
    assert block["recall"]["value"] == 1.0
    # one held-out event can never reach a 99 % Wilson lower bound
    assert block["recall"]["lower_bound_95"] < 0.99
    assert block["train_events"] == 1
    assert block["split"].startswith("sha256(case)")

    nothing = det.stage_clear_corpus_gate(tmp_path / "missing")
    assert nothing["measured"] is False
    assert nothing["recall"]["n"] == 0


def test_g2_stage_clear_row_needs_recall_and_precision_lower_bounds():
    block = {
        "death_scopes": det.observable_death_scopes([_death_shard(2, 0) for _ in range(200)]),
        "bomb": det.event_metrics([True, False] * 400, [True, False] * 400)["event_level"],
        "readers": {"lives": det.proportion(1000, 1000),
                    "bombs": det.proportion(1000, 1000),
                    "score_sign": det.proportion(1000, 1000)},
        "game_over": det.game_over_metrics([{"hgo": [1, 0] * 400, "go_pred": [1, 0] * 400}]),
        "online": {"episodes": 3, "frames": 5000, "pass": True, "long_enough": True},
        "stage_clear": {
            "measured": True, "events": 1000, "detector": "vgrad sign flip",
            "recall": det.proportion(1000, 1000),
            "precision": det.proportion(1000, 1000),
        },
    }
    verdict = det.g2_verdict_v2(block)
    rows = {row["gate"]: row for row in verdict["rows"]}
    row = rows["stage_clear recall (event level)"]
    assert row["amended"]["pass"] is True
    assert row["amended"]["n"] == 1000
    assert "精确率" in row["amended"]["note"]

    # 100 % on 100 events is below the 99 % lower bound: the row must not pass
    block["stage_clear"] = {"measured": True, "events": 100, "detector": "vgrad sign flip",
                            "recall": det.proportion(100, 100),
                            "precision": det.proportion(100, 100)}
    small = {row["gate"]: row for row in det.g2_verdict_v2(block)["rows"]}
    assert small["stage_clear recall (event level)"]["amended"]["value"] == 1.0
    assert small["stage_clear recall (event level)"]["amended"]["pass"] is False

    # zero samples are not a pass
    block["stage_clear"] = {"measured": True, "events": 0,
                            "recall": det.proportion(0, 0),
                            "precision": det.proportion(0, 0)}
    zero = {row["gate"]: row for row in det.g2_verdict_v2(block)["rows"]}
    assert zero["stage_clear recall (event level)"]["amended"]["pass"] is False


def test_g2_two_column_table_keeps_fourteen_rows_with_measured_stage_clear():
    block = {
        "death_scopes": det.observable_death_scopes([_death_shard(2, 0) for _ in range(200)]),
        "bomb_scopes": det.observable_bomb_scopes(
            [{"bo": [3, 2, 2, 1] * 100, "bomb": [True, False, False, False] * 100}]),
        "readers": {"lives": det.proportion(1000, 1000),
                    "bombs": det.proportion(1000, 1000),
                    "score_sign": det.proportion(1000, 1000)},
        "game_over": det.game_over_metrics([{"hgo": [1, 0] * 400, "go_pred": [1, 0] * 400}]),
        "online": {"episodes": 3, "frames": 5000, "pass": False, "long_enough": True},
        "stage_clear": {"measured": True, "events": 1000, "detector": "vgrad sign flip",
                        "recall": det.proportion(1000, 1000),
                        "precision": det.proportion(1000, 1000)},
    }
    verdict = det.g2_verdict_v2(block)
    assert len(verdict["rows"]) == 14
    table = det.render_v2_table(verdict)
    assert "stage_clear recall (event level)" in table
    assert table.count("|") > 14 * 3


# ---------------------------------------------------------------------------
# EXP-P2-008: pixel-side death detector (player sprite NCC collapse)
# ---------------------------------------------------------------------------
def _sprite_planes(*, length: int = 40, alive: tuple[int, int], gap: tuple[int, int] | None = None,
                   row: int = 60, col: int = 48):
    """play96 stack with a textured 5x5 player sprite, absent on ``gap`` frames."""

    import numpy as np

    planes = np.zeros((length, 112, 96), dtype=np.uint8)
    planes[:, :, :] = np.linspace(0, 30, 96, dtype=np.float32)[None, None, :]
    checker = np.array([[200, 240, 200, 240, 200],
                        [240, 200, 240, 200, 240],
                        [200, 240, 200, 240, 200],
                        [240, 200, 240, 200, 240],
                        [200, 240, 200, 240, 200]], dtype=np.uint8)
    gap_start, gap_end = gap if gap is not None else (-1, -1)
    for index in range(*alive):
        if gap_start <= index < gap_end:
            continue
        planes[index, row - 2:row + 3, col - 2:col + 3] = checker
    return planes


def test_pixel_death_ncc_collapses_when_the_sprite_disappears():
    positions = [(60, 48)] * 40
    planes = _sprite_planes(alive=(0, 40), gap=(20, 35))
    ncc = det.player_patch_ncc(planes, positions)
    assert ncc[1] > 0.99                       # sprite unchanged
    assert ncc[20] < 0.5                       # sprite gone
    predictions = det.pixel_death_predictions(ncc, low=0.55, high=0.9)
    assert predictions == [20]                 # one fire, the respawn blink is refractory

    metrics = det.pixel_death_metrics([{"ncc": ncc, "events": [20]}])
    assert metrics["events"] == 1 and metrics["matched"] == 1
    assert metrics["recall"]["value"] == 1.0
    assert metrics["precision"]["value"] == 1.0
    assert metrics["latency"]["max"] == 0

    with pytest.raises(ValueError):
        det.player_patch_ncc(planes.reshape(40, 96, 112), positions)


def test_pixel_death_metrics_count_extra_fires_in_precision():
    # two sprite losses 120 frames apart: the second is outside the 90-frame
    # respawn refractory, so it is a second prediction and must hurt precision
    positions = [(60, 48)] * 160
    planes = _sprite_planes(length=160, alive=(0, 160), gap=(20, 35))
    planes[130:160] = _sprite_planes(length=30, alive=(0, 30), gap=(0, 30))
    ncc = det.player_patch_ncc(planes, positions)
    predictions = det.pixel_death_predictions(ncc, low=0.55, high=0.9)
    assert len(predictions) >= 2
    metrics = det.pixel_death_metrics([{"ncc": ncc, "events": [20]}],
                                      low=0.55, high=0.9)
    assert metrics["matched"] == 1
    assert metrics["predictions"] >= 2
    assert metrics["precision"]["value"] < 1.0

    silent = det.pixel_death_metrics([{"ncc": ncc, "events": []}])
    assert silent["recall"]["n"] == 0
    assert silent["recall"]["value"] == 0.0


# ---------------------------------------------------------------------------
# EXP-P2-009: stage_clear topped up to the 386-event Wilson threshold
# ---------------------------------------------------------------------------
def test_g2_stage_clear_row_clears_the_386_event_threshold():
    """386 zero-failure events is where the 99 % Wilson bound is actually reached.

    ``LB = n / (n + 3.8416)``, so the exact boundary is n = 381 (380 -> 0.989992,
    381 -> 0.990018).  386 is the project's stated target because it leaves
    margin above that boundary; 131 is the EXP-P2-008 sample and caps at 97.15 %.
    """

    assert det.proportion(380, 380)["lower_bound_95"] < 0.99
    assert det.proportion(381, 381)["lower_bound_95"] >= 0.99
    assert det.proportion(386, 386)["lower_bound_95"] >= 0.99
    assert det.proportion(386, 386)["lower_bound_95"] > det.proportion(381, 381)["lower_bound_95"]
    assert det.proportion(131, 131)["lower_bound_95"] < 0.99

    block = {
        "death_scopes": det.observable_death_scopes([_death_shard(2, 0) for _ in range(200)]),
        "bomb_scopes": det.observable_bomb_scopes(
            [{"bo": [3, 2, 2, 1] * 100, "bomb": [True, False, False, False] * 100}]),
        "readers": {"lives": det.proportion(1000, 1000),
                    "bombs": det.proportion(1000, 1000),
                    "score_sign": det.proportion(1000, 1000)},
        "game_over": det.game_over_metrics([{"hgo": [1, 0] * 400, "go_pred": [1, 0] * 400}]),
        "online": {"episodes": 3, "frames": 5000, "pass": True, "long_enough": True},
        "stage_clear": {"measured": True, "events": 386, "detector": "vgrad jump",
                        "recall": det.proportion(386, 386),
                        "precision": det.proportion(386, 386)},
    }
    rows = {row["gate"]: row for row in det.g2_verdict_v2(block)["rows"]}
    row = rows["stage_clear recall (event level)"]
    assert row["amended"]["n"] == 386
    assert row["amended"]["pass"] is True
    # the combined row takes the *worse* of recall and precision
    block["stage_clear"]["precision"] = det.proportion(380, 380)
    rows = {row["gate"]: row for row in det.g2_verdict_v2(block)["rows"]}
    assert rows["stage_clear recall (event level)"]["amended"]["pass"] is False


def test_stage_clear_cross_validation_folds_agree_within_one_corpus(tmp_path: Path):
    """Every fold is scored by a criterion fitted on replays it never saw.

    EXP-P2-009 does **not** re-tune anything, so the property that has to hold is
    that the 3 folds independently land on the same criterion and that each
    held-out fold is scored completely (``matched == events``, no miss).
    """

    import hashlib

    def fold_of(case: str) -> int:
        return hashlib.sha256(case.encode()).digest()[1] % 3

    # pick the first two replays per fold so all three folds are populated
    buckets: dict[int, list[str]] = {}
    for index in range(200):
        name = f"case-{index:03d}"
        buckets.setdefault(fold_of(name), []).append(name)
        if len(buckets) == 3 and all(len(chunk) >= 2 for chunk in buckets.values()):
            break
    assert len(buckets) == 3 and all(len(chunk) >= 2 for chunk in buckets.values()), buckets

    root = tmp_path / "events"
    edges = (40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)
    selected = [name for chunk in buckets.values() for name in chunk[:2]]
    assert len(selected) == 6, selected
    layout = [(case, stage, edges[2 * offset + stage])
              for offset, case in enumerate(selected)
              for stage in (0, 1)]
    for case, stage, edge in layout:
        _write_stage_clear_event(root, case, stage, edge=edge)
    edges = [edge for _, _, edge in layout]

    block = det.stage_clear_corpus_gate(root)
    assert block["measured"] is True
    assert block["events"] == len(edges)
    assert block["recall"]["value"] == 1.0
    assert block["precision"]["value"] == 1.0

    folds = block["cross_validation"]["folds"]
    assert len(folds) == 3
    chosen = {(row["jump"], row["floor"], row["sustain"]) for row in folds}
    assert len(chosen) == 1, f"folds disagree on the criterion: {chosen}"
    for row in folds:
        assert row["held_out"] == row["events"] > 0
        assert row["matched"] == row["events"]
        assert row["predictions"] == row["events"]
        assert row["misses"] == []
    assert sum(row["events"] for row in folds) == len(edges)
    # the aggregate out-of-fold numbers are what the G2 row consumes
    assert block["cross_validation"]["events"] == len(edges)
    assert block["cross_validation"]["matched"] == len(edges)


def test_stage_clear_coverage_report_schema_and_zero_fill(tmp_path: Path):
    """The coverage block must show a 0 for a difficulty/stage never sampled."""

    from tools.rl import v400_stage_clear_replay as replay

    plan = {
        "available": [
            {"case": "a", "difficulty": "Easy", "stage": 0, "stageFrames": 100},
            {"case": "b", "difficulty": "Easy", "stage": 0, "stageFrames": 120},
            {"case": "c", "difficulty": "Easy", "stage": 1, "stageFrames": 300},
            {"case": "d", "difficulty": "Lunatic", "stage": 1, "stageFrames": 200},
            {"case": "e", "difficulty": "Extra", "stage": 8, "stageFrames": 900},
        ]
    }
    captured = [
        {"case": "a", "stage": 0, "difficulty": "Easy", "stageFrames": 100},
        {"case": "b", "stage": 0, "difficulty": "Easy", "stageFrames": 120},
        {"case": "c", "stage": 1, "difficulty": "Easy", "stageFrames": 300},
    ]
    coverage = replay.coverage_block(captured, plan)
    assert coverage["schema"] == "v400-stage-clear-coverage/1"
    assert coverage["capturedEvents"] == 3
    assert coverage["difficulties"]["Easy"] == {"captured": 3, "available": 3, "frames": 520}
    assert coverage["difficulties"]["Lunatic"]["captured"] == 0
    assert coverage["difficulties"]["Lunatic"]["available"] == 1
    assert "Extra" in coverage["zeroDifficulties"]
    assert "Lunatic" in coverage["zeroDifficulties"]
    assert coverage["stages"]["0"] == {"captured": 2, "available": 2, "frames": 220}
    assert "8" in coverage["zeroStages"]
    assert coverage["difficultyStage"] == {"Easy/s0": 2, "Easy/s1": 1}
    # distinct (difficulty, stage) buckets in the corpus, not the event count
    assert coverage["availableUniverse"] == 4

    # no plan -> counts still work, the "available" side is explicitly null
    bare = replay.coverage_block(captured, None)
    assert bare["difficulties"]["Easy"]["captured"] == 3
    assert bare["difficulties"]["Easy"]["available"] is None
    assert bare["zeroDifficulties"] == ["Normal", "Hard", "Lunatic", "Extra"]
