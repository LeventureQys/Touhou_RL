"""EXP-P2-010 / WP5 pixel-side death CNN: contract tests (pure numpy, no torch).

The detector's *identification* numbers live in the offline/online artifacts;
these tests pin the parts that must never drift:

* the causal window (no future frame, clipping at the start of a run),
* the label bands (positives, unknown dissolve tail, respawn hard negatives),
* the split (whole run, stratified, no run in two splits, fail closed),
* the alarm rule (rising edge, confirmation delay added *after* the event),
* the event/latency matching (one-to-one, ``before=0`` by default),
* the online schema and the fail-closed gate,
* the CLI wiring of ``index``/``split``/``gate`` and their exit codes.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tools.rl import v400_pixel_death as pd


# ---------------------------------------------------------------------------
# window construction
# ---------------------------------------------------------------------------
def test_window_is_causal_and_ordered():
    indices = pd.window_indices(10)
    assert indices == list(range(4, 11))
    assert max(indices) == 10, "window must end at t, never look ahead"
    assert indices == sorted(indices)


def test_window_clips_at_run_start_without_wrapping():
    assert pd.window_indices(0, frames=100) == [0, 0, 0, 0, 0, 0, 0]
    assert pd.window_indices(2, frames=100) == [0, 0, 0, 0, 0, 1, 2]
    assert pd.window_indices(6, frames=100) == [0, 1, 2, 3, 4, 5, 6]
    with pytest.raises(pd.PixelDeathError):
        pd.window_indices(-1, frames=100)
    with pytest.raises(pd.PixelDeathError):
        pd.window_indices(0, frames=0)


def test_window_valid_rejects_short_runs():
    assert pd.window_valid(pd.WINDOW - 1)
    assert not pd.window_valid(pd.WINDOW - 2)


def test_stack_window_copies_the_right_frames():
    planes = np.arange(10 * 3 * 4, dtype=np.uint8).reshape(10, 3, 4)
    stack = pd.stack_window(planes, 7)
    assert stack.shape == (pd.WINDOW, 3, 4)
    assert np.array_equal(stack[-1], planes[7])
    assert np.array_equal(stack[0], planes[1])
    stack[0, 0, 0] = 255
    assert planes[1, 0, 0] != 255, "stack_window must copy, not alias the memmap"


def test_frame_channels_matches_batch_channels():
    generator = np.random.default_rng(0)
    planes = generator.integers(0, 255, size=(40, *pd.PLAY96_SHAPE), dtype=np.uint8)
    single = pd.frame_channels(planes, 20)
    batch = pd.batch_channels(planes, [20])
    assert single.shape == (pd.CHANNELS, *pd.PLAY96_SHAPE)
    assert np.allclose(single, batch[0], atol=1e-6)


def test_batch_channels_is_vectorised_over_ends():
    generator = np.random.default_rng(1)
    planes = generator.integers(0, 255, size=(50, *pd.PLAY96_SHAPE), dtype=np.uint8)
    ends = [6, 7, 12, 33]
    batch = pd.batch_channels(planes, ends)
    assert batch.shape == (len(ends), pd.CHANNELS, *pd.PLAY96_SHAPE)
    for position, end in enumerate(ends):
        assert np.allclose(batch[position], pd.frame_channels(planes, end), atol=1e-6)
    empty = pd.batch_channels(planes, [])
    assert empty.shape[0] == 0


def test_collapse_channel_is_zero_on_a_static_screen():
    planes = np.full((20, *pd.PLAY96_SHAPE), 90, dtype=np.uint8)
    channels = pd.frame_channels(planes, 12)
    assert np.allclose(channels[1:], 0.0)


def test_collapse_channel_sees_motion_dying():
    generator = np.random.default_rng(2)
    planes = generator.integers(0, 255, size=(20, *pd.PLAY96_SHAPE), dtype=np.uint8).astype(
        np.int16
    )
    planes[10:] = 128                       # screen freezes at frame 10
    planes = np.clip(planes, 0, 255).astype(np.uint8)
    moving = pd.frame_channels(planes, 9)
    frozen = pd.frame_channels(planes, 14)
    assert np.abs(moving[4]).mean() > np.abs(frozen[4]).mean()


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def test_label_bands_positive_unknown_and_respawn():
    events = [100]
    label, usable = pd.frame_labels(200, events)
    assert label[100] == 1 and label[104] == 1
    assert label[99] == 0 and label[105] == 0
    # pre-onset band is unusable, not negative
    assert not usable[100 - pd.EDGE] and not usable[99]
    # dissolve tail is unusable for the normal (+4) death ...
    assert not usable[105] and not usable[108]
    # ... but the respawn blink stays usable as a hard negative
    assert usable[100 + pd.RESPAWN_AFTER[0]] and usable[100 + pd.RESPAWN_AFTER[1]]
    assert label[100 + pd.RESPAWN_AFTER[0]] == 0
    # a positive frame is never unusable
    assert usable[100] and usable[104]


def test_slow_dissolve_tail_is_unusable_for_bomb_empty_events():
    label, usable = pd.frame_labels(300, [100], slow_events=[100], slow_after=29)
    assert not usable[105] and not usable[120] and not usable[129]
    # the +29 tail covers the respawn band too, so a slow event contributes no
    # respawn hard negatives -- it contributes its own 30-frame unusable tail.
    assert not usable[100 + pd.RESPAWN_AFTER[0]]
    assert usable[130] and usable[140]
    assert label[100] == 1 and label[104] == 1 and label[130] == 0


def test_window_start_is_never_usable():
    _label, usable = pd.frame_labels(50, [])
    assert not usable[: pd.WINDOW].any()


def test_build_samples_keeps_labels_aligned_with_samples(tmp_path):
    # EXP-P2-010 §2.3 regression: sorting must carry the label, otherwise the
    # trainer is fed a mislabelled corpus and every metric below is noise.
    shards = _corpus(tmp_path, per_stratum=4)
    rows = [row.as_row() | {"events": [20 + index],
                            "bomb_empty_events": []}
            for index, row in enumerate(shards)]
    payload = pd.build_samples(rows, seed=1, negatives_per_positive=2.0, jitter=2)
    samples = payload["samples"]
    labels = payload["labels"]
    assert len(samples) == len(labels)
    positives = {sample for sample, label in zip(samples, labels) if label == 1}
    negatives = {sample for sample, label in zip(samples, labels) if label == 0}
    assert positives and negatives
    assert not positives & negatives, "a sample cannot be both positive and negative"
    for shard_index, frame in positives:
        expected = pd.positive_sample_frames(
            int(rows[shard_index]["frames"]), rows[shard_index]["events"]
        )
        assert frame in expected, "a positive sample is not inside a positive window"
    for shard_index, frame in negatives:
        label, usable = pd.frame_labels(int(rows[shard_index]["frames"]),
                                       rows[shard_index]["events"])
        assert usable[frame] and not label[frame], "a negative sample is labelled positive"
    assert samples == sorted(samples)


def test_positive_sample_frames_are_causal():
    frames = pd.positive_sample_frames(500, [0, 300])
    assert frames == [300, 301, 302, 303, 304]
    assert pd.positive_sample_frames(500, [10]) == [10, 11, 12, 13, 14]
    assert all(frame >= pd.WINDOW - 1 for frame in frames)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def _fake_shard(tmp_path, name, *, root="r1", stage=3, frames=40, deaths=(), bombs=()):
    directory = Path(tmp_path) / "shards" / name / "attempt-01"
    directory.mkdir(parents=True, exist_ok=True)
    planes = np.zeros((frames, *pd.PLAY96_SHAPE), dtype=np.uint8)
    planes.tofile(directory / "play96.u8")
    rows = []
    for index in range(frames):
        row = {"i": index, "death": index in deaths}
        if index in deaths:
            row["bo"] = 0 if index in bombs else 3
            row["li"] = 2
        rows.append(row)
    (directory / "labels.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return directory


def test_scan_root_reads_the_collector_contract(tmp_path):
    _fake_shard(tmp_path / "root", "dive-s3d2-seed1-n40-w00", deaths=[20, 30], bombs=[30])
    shards = pd.scan_root(tmp_path / "root")
    assert len(shards) == 1
    shard = shards[0]
    assert shard.stage == 3 and shard.frames == 40
    assert shard.deaths == [20, 30]
    assert shard.death_bombs == [3, 0]
    assert pd.death_events(shard.deaths) == [20, 30]
    assert pd.bomb_empty_events(shard) == [30]


def test_scan_root_ignores_a_shard_without_planes(tmp_path):
    directory = Path(tmp_path) / "root" / "shards" / "x-s3d1-seed1-n10-w00" / "attempt-01"
    directory.mkdir(parents=True)
    (directory / "labels.jsonl").write_text('{"i": 0, "death": true}\n', encoding="utf-8")
    assert pd.scan_root(tmp_path / "root") == []


def test_death_events_merge_a_multi_frame_flag():
    assert pd.death_events([10, 11, 12, 40]) == [10, 40]


def _corpus(tmp_path, *, per_stratum=6):
    shards = []
    for root in ("g2c-go2", "g2c-bomb2"):
        for stage in (2, 3):
            for index in range(per_stratum):
                name = f"dive-s{stage}d3-seed{index}-n40-w00"
                bombs = index % 2 == 1
                directory = _fake_shard(
                    tmp_path / root, name, root=root, stage=stage, frames=40,
                    deaths=(20 + index,), bombs=(20 + index,) if bombs else (),
                )
                shards.append(
                    pd.Shard(dir=directory, root=root, shard=name,
                             attempt="attempt-01", stage=stage, frames=40,
                             deaths=[20 + index],
                             death_bombs=[0 if bombs else 3],
                             death_lives=[2])
                )
    return shards


def test_split_keeps_whole_runs_and_no_leakage(tmp_path):
    shards = _corpus(tmp_path)
    split = pd.build_split(shards, seed=7, min_test_events=1)
    assert split["schema"] == pd.SPLIT_SCHEMA
    assignments = [(row["root"], row["shard"], row["attempt"], row["split"])
                   for row in split["shards"]]
    runs = {}
    for root, shard, attempt, name in assignments:
        runs.setdefault(f"{root}/{shard}/{attempt}", set()).add(name)
    assert all(len(names) == 1 for names in runs.values()), "a run leaked across splits"
    assert len(assignments) == len(shards)
    assert {name for names in runs.values() for name in names} == {"train", "val", "test"}
    for name in ("train", "val", "test"):
        assert split["summary"][name]["shards"] == len(pd.split_rows(split, name))


def test_split_is_deterministic_for_a_seed(tmp_path):
    shards = _corpus(tmp_path)
    first = pd.build_split(shards, seed=11, min_test_events=1)
    second = pd.build_split(shards, seed=11, min_test_events=1)
    assert first["summary"] == second["summary"]
    assert [row["shard"] for row in first["shards"]] == [
        row["shard"] for row in second["shards"]
    ]


def test_split_fails_closed_on_bad_input(tmp_path):
    shards = _corpus(tmp_path)
    with pytest.raises(pd.PixelDeathError):
        pd.build_split(shards, fractions=(0.5, 0.5, 0.5))
    with pytest.raises(pd.PixelDeathError):
        pd.build_split(shards, fractions=(1.0, 0.0, 0.0))
    with pytest.raises(pd.PixelDeathError):
        pd.build_split(shards, min_test_events=10 ** 6)
    with pytest.raises(pd.PixelDeathError):
        pd.build_split([])
    with pytest.raises(pd.PixelDeathError):
        pd.split_rows({"shards": []}, "test")


def test_split_keeps_the_bomb_empty_slice_on_both_sides(tmp_path):
    shards = _corpus(tmp_path, per_stratum=8)
    split = pd.build_split(shards, seed=3, min_test_events=1)
    assert split["summary"]["train"]["bomb_empty_events"] > 0
    assert split["summary"]["test"]["bomb_empty_events"] > 0


# ---------------------------------------------------------------------------
# alarms + latency
# ---------------------------------------------------------------------------
def test_predict_events_is_a_rising_edge():
    probs = [0.0] * 5 + [0.9, 0.95, 0.9] + [0.0] * 5 + [0.9] + [0.0] * 5
    alarms = pd.predict_events(probs, threshold=0.5, refractory=0, delay=0, lookback=1)
    assert alarms == [5, 13]


def test_predict_events_never_fires_before_the_rise():
    probs = [0.0] * 3 + [0.9] * 4
    for delay in (0, 1, 2):
        alarms = pd.predict_events(probs, threshold=0.5, refractory=0, delay=delay,
                                   lookback=1)
        assert alarms == [3 + delay], "the confirmation delay must be added, never subtracted"


def test_predict_events_needs_a_lookback_of_history():
    probs = [0.0, 0.0, 0.9, 0.9]
    assert pd.predict_events(probs, threshold=0.5, refractory=0, delay=0) == []
    assert pd.predict_events(probs, threshold=0.5, refractory=0, delay=0,
                             lookback=1) == [2]


def test_predict_events_refractory_and_validation():
    probs = [0.0, 0.9, 0.0, 0.9, 0.0, 0.0, 0.9]
    assert pd.predict_events(probs, threshold=0.5, refractory=0, delay=0,
                             lookback=1) == [1, 3, 6]
    assert pd.predict_events(probs, threshold=0.5, refractory=2, delay=0,
                             lookback=1) == [1, 6]
    with pytest.raises(pd.PixelDeathError):
        pd.predict_events(probs, threshold=1.5)
    with pytest.raises(pd.PixelDeathError):
        pd.predict_events(probs, threshold=0.5, refractory=-1)
    with pytest.raises(pd.PixelDeathError):
        pd.predict_events([[0.1]], threshold=0.5)
    with pytest.raises(pd.PixelDeathError):
        pd.PixelDeathError and pd.predict_events(probs, threshold=0.5, prominence=-0.1)


def test_prominence_suppresses_a_flat_ramp():
    ramp = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    eased = pd.predict_events(ramp, threshold=0.5, refractory=0, delay=0, lookback=2)
    assert eased and eased[0] >= 4
    gated = pd.predict_events(ramp, threshold=0.5, refractory=0, delay=0, lookback=2,
                              prominence=0.4)
    assert gated == [], "a smooth ramp must not clear a 0.4 prominence gate"
    spike = [0.05, 0.05, 0.05, 0.05, 0.95, 0.05, 0.05]
    assert pd.predict_events(spike, threshold=0.5, prominence=0.4, lookback=2) == [5]


def test_alarm_rule_streaming_matches_the_batch_rule():
    generator = np.random.default_rng(5)
    probs = (generator.random(400) ** 3).tolist()
    for kwargs in ({"threshold": 0.3}, {"threshold": 0.6, "prominence": 0.2},
                   {"threshold": 0.5, "refractory": 30, "delay": 2}):
        rule = pd.AlarmRule(**kwargs)
        streamed = []
        for prob in probs:
            frame = rule.push(prob)
            if frame is not None:
                streamed.append(frame)
        assert streamed == pd.predict_events(probs, **kwargs)


def test_match_events_is_one_to_one_and_rejects_early_fires():
    match = pd.match_events([100], [99, 101])
    assert match["matched"] == 1 and match["latencies"] == [1]
    assert match["unmatched_predictions"] == 1, "the pre-event alarm is a false positive"
    tolerant = pd.match_events([100], [99], before=2)
    assert tolerant["matched"] == 1 and tolerant["latencies"] == [-1]


def test_match_events_window_and_shared_prediction():
    assert pd.match_events([100], [111])["matched"] == 0
    assert pd.match_events([100], [110])["matched"] == 1
    shared = pd.match_events([100, 104], [102])
    assert shared["matched"] == 1, "one prediction cannot cover two events"
    assert shared["unmatched_predictions"] == 0


def test_latency_summary_reports_median_p95_and_max():
    summary = pd.latency_summary([0, 1, 2, 3, 10])
    assert summary["n"] == 5 and summary["median"] == 2
    assert summary["max"] == 10 and summary["within_limit"]
    assert pd.latency_summary([0, 11])["within_limit"] is False
    assert pd.latency_summary([])["n"] == 0
    assert pd.latency_summary([-1, 0])["early_detections"] == 1


def test_proportion_carries_the_wilson_ceiling():
    perfect = pd.proportion(9, 9)
    assert perfect["value"] == 1.0 and perfect["lower_bound_95"] < 1.0
    assert perfect["lower_bound_ceiling"] == pytest.approx(1 - 3 / 9)
    assert not perfect["pass_99"]
    big = pd.proportion(500, 500)
    assert big["lower_bound_95"] >= 0.99 and big["pass_99"]
    assert pd.proportion(0, 0)["lower_bound_95"] == 0.0


# ---------------------------------------------------------------------------
# online schema + gate
# ---------------------------------------------------------------------------
def _episode(index, *, frames=2400, deaths=(500, 900), alarms=None, latencies=None):
    alarms = alarms if alarms is not None else [death + 1 for death in deaths]
    latencies = latencies if latencies is not None else [1 for _ in deaths]
    return {
        "episode": index,
        "frames": frames,
        "deathEvents": list(deaths),
        "alarms": list(alarms),
        "latencies": list(latencies),
        "inferenceMs": 3.5,
    }


def _big_episodes(count=5):
    return [_episode(index, deaths=tuple(range(20, 2400, 6))) for index in range(count)]


def test_online_gate_accepts_a_clean_payload():
    payload = {"schema": pd.ONLINE_SCHEMA, "episodes": _big_episodes()}
    gate = pd.online_gate(payload, min_episodes=5, min_frames=2400)
    assert gate["deaths"] == 5 * len(tuple(range(20, 2400, 6)))
    assert gate["alarms"] == gate["deaths"] == gate["matched"]
    assert gate["passed"], gate["failures"]
    assert gate["latency"]["max"] == 1


def test_online_gate_fails_closed_on_shape():
    with pytest.raises(pd.PixelDeathError):
        pd.online_gate({"schema": "something-else", "episodes": []})
    with pytest.raises(pd.PixelDeathError):
        pd.online_gate({"schema": pd.ONLINE_SCHEMA, "episodes": []})
    with pytest.raises(pd.PixelDeathError):
        pd.online_gate({"schema": pd.ONLINE_SCHEMA, "episodes": [{"frames": 2400}]})
    with pytest.raises(pd.PixelDeathError):
        pd.online_gate(["not", "a", "dict"])


def test_online_gate_rejects_short_or_empty_episodes():
    payload = {"schema": pd.ONLINE_SCHEMA,
               "episodes": [_episode(index, frames=100) for index in range(5)]}
    gate = pd.online_gate(payload, min_episodes=5, min_frames=2400)
    assert not gate["passed"]
    assert any("frames" in failure for failure in gate["failures"])
    empty = {"schema": pd.ONLINE_SCHEMA,
             "episodes": [_episode(index, deaths=()) for index in range(5)]}
    gate = pd.online_gate(empty, min_episodes=5, min_frames=2400)
    assert not gate["passed"]
    assert any("no death event" in failure for failure in gate["failures"])


def test_online_gate_rejects_too_few_episodes_and_bad_latency():
    payload = {"schema": pd.ONLINE_SCHEMA, "episodes": [_episode(0)]}
    gate = pd.online_gate(payload, min_episodes=5, min_frames=2400)
    assert not gate["passed"]
    assert any("episodes" in failure for failure in gate["failures"])
    slow = {"schema": pd.ONLINE_SCHEMA,
            "episodes": [_episode(index, alarms=[501], latencies=[41]) for index in range(5)]}
    gate = pd.online_gate(slow, min_episodes=5, min_frames=2400)
    assert not gate["passed"]
    assert gate["latency"]["max"] == 41
    assert any("latency" in failure for failure in gate["failures"])


def test_online_gate_rejects_early_alarms():
    payload = {"schema": pd.ONLINE_SCHEMA,
               "episodes": [_episode(index, alarms=[499], latencies=[-1])
                            for index in range(5)]}
    gate = pd.online_gate(payload, min_episodes=5, min_frames=2400)
    assert not gate["passed"]
    assert any("before the event" in failure for failure in gate["failures"])


def test_online_gate_cannot_sign_a_small_sample():
    # 10/10 events is a perfect point estimate, but the Wilson LB stays far
    # under 0.99: this is the arithmetic that keeps the G2 row unsigned at the
    # required 5x2400-frame budget (n=10, ceiling 1-3/n = 0.70).
    payload = {"schema": pd.ONLINE_SCHEMA,
               "episodes": [_episode(index) for index in range(5)]}
    gate = pd.online_gate(payload, min_episodes=5, min_frames=2400)
    assert gate["recall"]["value"] == 1.0
    assert gate["recall"]["lower_bound_95"] < 0.99
    assert gate["recall"]["lower_bound_ceiling"] == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# offline report + CLI
# ---------------------------------------------------------------------------
def _offline_payload(recall_successes=500, recall_trials=500, precision_successes=500,
                     precision_trials=500, latency_max=1):
    def block(successes, trials, extra=None):
        return {**pd.proportion(successes, trials), **(extra or {})}

    return {
        "schema": pd.OFFLINE_SCHEMA,
        "split": {"schema": pd.SPLIT_SCHEMA, "machine": "whole shard", "seed": 1,
                  "summary": {}},
        "checkpoint": {"parameters": 96_000},
        "test": {
            "events": recall_trials,
            "alarms": precision_trials,
            "recall": block(recall_successes, recall_trials),
            "precision": block(precision_successes, precision_trials),
            "latency": pd.latency_summary([0] * (precision_trials - 1) + [latency_max]),
            "bombEmpty": {"events": 20, "matched": 20,
                          "recall": block(20, 20), "latency": pd.latency_summary([1] * 20)},
        },
        "val": None,
    }


def test_offline_report_requires_a_test_event():
    with pytest.raises(pd.PixelDeathError):
        pd.offline_report(split={"schema": pd.SPLIT_SCHEMA}, checkpoint={},
                          test={"recall": {"n": 0}})


def test_gate_command_combines_offline_and_online(tmp_path):
    offline = _offline_payload()
    offline_path = tmp_path / "offline.json"
    offline_path.write_text(json.dumps(offline), encoding="utf-8")
    online_path = tmp_path / "online.json"
    online_path.write_text(
        json.dumps({"schema": pd.ONLINE_SCHEMA, "episodes": _big_episodes()}),
        encoding="utf-8",
    )
    out = tmp_path / "gate.json"
    code = pd.main(["gate", "--offline", str(offline_path), "--online", str(online_path),
                    "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and payload["passed"] and payload["row14"]["passed"]

    # a slow online latency must fail the same command
    slow_path = tmp_path / "slow.json"
    slow_path.write_text(
        json.dumps({"schema": pd.ONLINE_SCHEMA,
                    "episodes": [_episode(index, alarms=[501], latencies=[31])
                                 for index in range(5)]}),
        encoding="utf-8",
    )
    code = pd.main(["gate", "--offline", str(offline_path), "--online", str(slow_path),
                    "--out", str(out)])
    assert code == 1
    assert not json.loads(out.read_text(encoding="utf-8"))["passed"]

    # without an online artifact the row cannot be signed
    code = pd.main(["gate", "--offline", str(offline_path), "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and payload["online"]["failures"]


def test_gate_command_rejects_a_foreign_offline_schema(tmp_path):
    path = tmp_path / "offline.json"
    path.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")
    with pytest.raises(pd.PixelDeathError):
        pd.main(["gate", "--offline", str(path), "--out", str(tmp_path / "gate.json")])


def test_split_command_round_trip(tmp_path):
    # index -> split through the CLI on a tiny generated corpus
    root = tmp_path / "corpus"
    for index in range(6):
        _fake_shard(root, f"dive-s3d3-seed{index}-n40-w00", frames=40,
                    deaths=(20 + index,), bombs=(25,) if index % 2 else ())
    index_path = tmp_path / "index.json"
    assert pd.main(["index", "--root", str(root), "--out", str(index_path)]) == 0
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["schema"] == pd.INDEX_SCHEMA and payload["totals"]["shards"] == 6
    split_path = tmp_path / "split.json"
    assert pd.main(["split", "--index", str(index_path), "--out", str(split_path),
                    "--min-test-events", "1"]) == 0
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assert split["schema"] == pd.SPLIT_SCHEMA
    assert sum(split["summary"][name]["shards"] for name in ("train", "val", "test")) == 6


def test_index_command_fails_closed_on_an_empty_root(tmp_path):
    with pytest.raises(pd.PixelDeathError):
        pd.main(["index", "--root", str(tmp_path), "--out", str(tmp_path / "index.json")])


# ---------------------------------------------------------------------------
# EXP-P2-011: a checkpoint trained before a feature leg was added must still
# load and score on exactly the legs it saw (the shipped WP5 CNN is 5-channel,
# the current feature builder emits 6).
# ---------------------------------------------------------------------------
def _planes(frames: int = 40):
    generator = np.random.default_rng(7)
    return generator.integers(0, 255, size=(frames, *pd.PLAY96_SHAPE), dtype=np.uint8)


def test_frame_channels_trims_to_the_leading_legs():
    planes = _planes()
    full = pd.frame_channels(planes, 20)
    trimmed = pd.frame_channels(planes, 20, channels=5)
    assert full.shape == (pd.CHANNELS,) + pd.PLAY96_SHAPE
    assert trimmed.shape == (5,) + pd.PLAY96_SHAPE
    assert np.array_equal(trimmed, full[:5])
    assert not np.array_equal(full[5], full[0])
    for bad in (0, pd.CHANNELS + 1):
        with pytest.raises(pd.PixelDeathError):
            pd.frame_channels(planes, 20, channels=bad)


def test_batch_channels_trims_to_the_leading_legs():
    planes = _planes()
    ends = [10, 20, 30]
    full = pd.batch_channels(planes, ends)
    trimmed = pd.batch_channels(planes, ends, channels=5)
    assert full.shape == (3, pd.CHANNELS) + pd.PLAY96_SHAPE
    assert trimmed.shape == (3, 5) + pd.PLAY96_SHAPE
    assert np.array_equal(trimmed, full[:, :5])
    with pytest.raises(pd.PixelDeathError):
        pd.batch_channels(planes, ends, channels=99)


def test_load_checkpoint_infers_the_channel_count_from_the_state_dict(tmp_path):
    torch = pytest.importorskip("torch")
    model = pd.build_model(width=8, channels=5)
    payload = {
        "schema": "v400-wp5-pixel-ckpt-1",
        "state": model.state_dict(),
        "config": {"width": 8, "auxCells": 0},
        "valThreshold": 0.7,
    }
    path = tmp_path / "ckpt.pt"
    torch.save(payload, path)
    loaded, meta = pd.load_checkpoint(path)
    assert meta["config"]["channels"] == 5
    assert loaded.body[0].weight.shape[1] == 5
    features = pd.frame_channels(_planes(), 20, channels=5)
    scores = pd._scores(loaded.to("cpu"), features[None], device=torch.device("cpu"),
                        batch_size=1)
    assert scores.shape == (1,)
