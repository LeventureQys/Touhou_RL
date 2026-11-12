"""T2a perception probe tests (EXP-P3-010).

No engine, no game, no collected data: every test builds its own frames, so the
suite proves *the probe's own* invariants rather than the machine's state:

* the native luminance equals obs-v1's, and the native 6x7 block mean
  reproduces ``obs_spec.encode_planes`` byte for byte (input A cannot drift);
* the 24x24 window really is a crop of the frozen P2 grid, snapped to 4 px,
  with out-of-grid cells marked invalid instead of scored as empty;
* input D's window anchor lines up so the 96x96 scored window sits exactly at
  ``[16:112]`` of the 128x128 crop;
* the head has **no** global pooling and the same parameter count for A/B/C;
* the pre-registered selection rule (smallest qualifying input, 0.90 floor,
  0.95 of best, D gated on the locator's 4 px median error) is implemented as
  written, including the stop-loss.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from touhou_rl.v400.student import obs_spec
from touhou_rl.v400.student.probe import inputs, labels, native
from touhou_rl.v400.student.probe.collect import (
    CollectError, ReplayRow, load_replay_rows,
)
from touhou_rl.v400.student.probe.net import (
    F1Counts, OccupancyHead, ProbeModel, count_cells, decode_locator,
    locator_error_px, locator_target, player_cell_coords, cell_coords_to_playfield,
    window_logits, HeadShapeError,
)
from touhou_rl.v400.student.probe.train import (
    TEST_SPLIT, TRAIN_SPLIT, select_input,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _frame(seed: int = 0, *, bullets: bool = True) -> np.ndarray:
    """A 640x480 RGBA8 readback with structure inside and outside the field."""

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 40, size=(480, 640, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    if bullets:
        for _ in range(40):
            x = int(rng.integers(32, 32 + 384 - 3))
            y = int(rng.integers(16, 16 + 448 - 3))
            pixels[y:y + 3, x:x + 3, :3] = 255
    return pixels


def _grid_from_cells(cells: np.ndarray) -> np.ndarray:
    """Expand a 24x24 window back into a 96x112 grid at ``(row0, col0)``."""

    grid = np.zeros((labels.GRID_HEIGHT, labels.GRID_WIDTH), dtype=bool)
    return grid, cells


# ---------------------------------------------------------------------------
# native pixels
# ---------------------------------------------------------------------------
def test_gray_playfield_matches_obs_spec_luminance():
    pixels = _frame(3)
    native_field = native.gray_playfield(pixels)
    assert native_field.shape == (448, 384)
    assert native_field.dtype == np.uint8
    x0, y0 = obs_spec.PLAYFIELD_ORIGIN
    crop = pixels[y0:y0 + 448, x0:x0 + 384]
    expected = obs_spec._luminance(crop).astype(np.uint8)
    assert np.array_equal(native_field, expected)


def test_block_mean_reproduces_obs_v1_play64():
    """Input A must be obs-v1's plane, not a second implementation of it."""

    pixels = _frame(5)
    fields = native.encode_native(pixels)
    assert fields["play64"].shape == obs_spec.PLAY_SHAPE
    assert np.array_equal(fields["play64"], obs_spec.encode_planes(pixels)[0])
    assert fields["play96x112"].shape == (112, 96)
    assert fields["play192x224"].shape == (224, 192)
    # and the native field is the same picture, not a resampled one
    assert fields["native"].shape == (448, 384)


def test_encode_native_refuses_a_frame_without_a_playfield():
    small = np.zeros((240, 320, 4), dtype=np.uint8)
    with pytest.raises(native.NativeFrameError):
        native.gray_playfield(small)
    with pytest.raises(native.NativeFrameError):
        native.gray_playfield(np.zeros((480, 640, 3), dtype=np.uint8))


def test_block_mean_requires_exact_blocks():
    native_field = np.zeros((448, 384), dtype=np.uint8)
    with pytest.raises(native.NativeFrameError):
        native.block_mean(native_field, (100, 100))


# ---------------------------------------------------------------------------
# window labels
# ---------------------------------------------------------------------------
def test_snap_anchor_is_on_the_frozen_cell_grid():
    for x, y in ((100.0, 200.0), (0.0, 0.0), (383.9, 447.9), (192.3, 233.7)):
        anchor_x, anchor_y = labels.snap_anchor(x, y)
        assert anchor_x % labels.CELL_PX == 0
        assert anchor_y % labels.CELL_PX == 0
        # the ship is inside the window, within half a cell of its centre
        assert abs((anchor_x + labels.WINDOW_PX // 2) - x) <= labels.CELL_PX / 2
        assert abs((anchor_y + labels.WINDOW_PX // 2) - y) <= labels.CELL_PX / 2


def test_window_label_is_a_crop_of_the_p2_grid():
    occupied = np.zeros((labels.GRID_HEIGHT, labels.GRID_WIDTH), dtype=bool)
    occupied[10:14, 20:24] = True
    distance = np.full((labels.GRID_HEIGHT, labels.GRID_WIDTH), 255.0, dtype=np.float32)
    distance[5:6, 15:16] = 1.0
    # player at the centre of cell (11, 21)
    player_x = 21 * 4 + 2.0
    player_y = 11 * 4 + 2.0
    window = labels.window_label(occupied, distance, player_x, player_y)
    assert window.valid.all()
    col0, row0 = labels.cell_origin(player_x, player_y)
    # The window is centred on the ship, so the marked cells sit above it.
    assert (row0, col0) == (0, 10)
    assert np.array_equal(
        window.cells,
        occupied[row0:row0 + 24, col0:col0 + 24])
    assert window.positives == int(occupied[row0:row0 + 24, col0:col0 + 24].sum())
    assert window.danger_cells == int(
        (distance[row0:row0 + 24, col0:col0 + 24] <= labels.DANGER_MARGIN_PX).sum())
    assert window.danger_cells == 1


def test_window_label_marks_out_of_grid_cells_invalid():
    occupied = np.zeros((labels.GRID_HEIGHT, labels.GRID_WIDTH), dtype=bool)
    window = labels.window_label(occupied, None, 2.0, 3.0)
    assert not window.valid.all()
    assert window.scored_cells < 24 * 24
    assert window.danger_cells == 0
    # cells outside the grid are never counted as positives
    assert window.cells[~window.valid].sum() == 0


def test_min_distance_takes_the_elementwise_minimum():
    a = np.full((112, 96), 10.0, dtype=np.float32)
    b = np.full((112, 96), 4.0, dtype=np.float32)
    b[5, 5] = 0.0
    stacked = labels.min_distance([a, b])
    assert stacked is not None
    assert stacked[0, 0] == 4.0
    assert stacked[5, 5] == 0.0
    assert labels.min_distance([]) is None


def test_window_pack_roundtrip():
    window = labels.window_label(
        np.zeros((112, 96), dtype=bool), None, 100.0, 200.0)
    packed = window.pack()
    assert packed["cells"].dtype == np.uint8
    assert packed["cells"].size == (24 * 24 + 7) // 8
    assert np.array_equal(labels.unpack_window(packed["cells"]), window.cells)


# ---------------------------------------------------------------------------
# input tensors
# ---------------------------------------------------------------------------
def test_input_shapes_and_areas():
    fields = native.encode_native(_frame(7))
    anchor_x, anchor_y = labels.snap_anchor(200.0, 250.0)
    shapes = {}
    for name in ("A", "B", "C", "D"):
        tensor = inputs.build_input(name, fields, anchor_x=anchor_x, anchor_y=anchor_y)
        shapes[name] = tensor.shape
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0
        assert shapes[name] == inputs.INPUT_SPECS[name].shape
    assert inputs.smallest_first() == ["A", "B", "D", "C"]


def test_input_d_needs_the_anchor():
    fields = native.encode_native(_frame(8))
    with pytest.raises(ValueError):
        inputs.build_input("D", fields)


def test_input_d_window_alignment():
    """The 96x96 scored window is exactly ``[16:112]`` of D's 128x128 crop."""

    native_field = np.zeros((448, 384), dtype=np.uint8)
    # a single marker at native (200, 250) plus a unique border pattern
    native_field[250, 200] = 255
    anchor_x, anchor_y = labels.snap_anchor(200.0, 250.0)
    window = inputs.local_window(native_field, anchor_x, anchor_y)
    assert window.shape == (128, 128)
    row = 250 - (anchor_y - 16)
    col = 200 - (anchor_x - 16)
    assert window[row, col] == 255
    assert int(window.sum()) == 255
    # the marker is inside the scored window
    assert 16 <= row < 112 and 16 <= col < 112
    rows, cols = inputs.label_crop_spec("D", (128, 128), anchor_x, anchor_y)
    assert rows == slice(16, 112) and cols == slice(16, 112)


def test_input_d_zero_pads_outside_the_playfield():
    native_field = np.full((448, 384), 200, dtype=np.uint8)
    anchor_x, anchor_y = labels.snap_anchor(0.0, 0.0)
    window = inputs.local_window(native_field, anchor_x, anchor_y)
    assert window.shape == (128, 128)
    # a ship at the corner still gets a full centred window, padded with 0
    assert int((window == 0).sum()) > 0
    assert int((window == 200).sum()) > 0


def test_window_cell_slice_matches_the_label_grid():
    anchor_x, anchor_y = labels.snap_anchor(123.0, 321.0)
    rows, cols = inputs.window_cell_slice(anchor_x, anchor_y)
    assert rows.stop - rows.start == labels.WINDOW_CELLS
    assert cols.stop - cols.start == labels.WINDOW_CELLS
    assert rows.start == anchor_y // labels.CELL_PX


# ---------------------------------------------------------------------------
# the head: no global pooling, identical parameter counts
# ---------------------------------------------------------------------------
def test_head_has_no_global_pooling():
    import torch

    model = ProbeModel("A")
    names = [name for name, _ in model.named_modules()]
    assert not any("AdaptiveAvgPool" in name for name in names)
    # a 1x1 pooling would collapse the field; prove the output keeps its grid
    head = OccupancyHead(in_channels=1)
    out = head(torch.zeros(1, 1, 64, 64))
    assert out.shape[-2:] == (64, 64)
    assert out.numel() > 1


def test_parameter_counts_match_across_whole_field_inputs():
    counts = {name: ProbeModel(name).head.parameter_count() for name in ("A", "B", "C")}
    assert len(set(counts.values())) == 1
    assert counts["A"] < ProbeModel("D").head.parameter_count()


def test_model_forward_shapes():
    import torch

    anchors = torch.tensor([[40, 30], [10, 5], [0, 0]])
    for name in ("A", "B", "C", "D"):
        spec = inputs.INPUT_SPECS[name]
        model = ProbeModel(name)
        out = model(torch.zeros(3, *spec.shape), anchors)
        assert out.shape == (3, 24, 24), (name, out.shape)


def test_model_rejects_a_wrong_shape():
    import torch

    model = ProbeModel("A")
    with pytest.raises(HeadShapeError):
        model(torch.zeros(1, 1, 96, 112), torch.zeros(1, 2, dtype=torch.long))


def test_window_logits_picks_the_anchored_cells():
    import torch

    cells = torch.zeros(2, 1, 112, 96)
    cells[0, 0, 50, 60] = 7.0
    cells[1, 0, 10 + 5, 20 + 6] = 3.0
    anchors = torch.tensor([[50 - 5, 60 - 6], [10, 20]])
    out = window_logits(cells, name="B", anchors=anchors)
    assert out.shape == (2, 24, 24)
    assert float(out[0, 5, 6]) == 7.0
    assert float(out[1, 5, 6]) == 3.0
    assert float(out[0].max()) == 7.0


# ---------------------------------------------------------------------------
# pixel locator
# ---------------------------------------------------------------------------
def test_locator_coordinate_roundtrip():
    for x, y in ((0.0, 0.0), (192.0, 224.0), (383.0, 447.0)):
        u, v = player_cell_coords(x, y)
        back = cell_coords_to_playfield(u, v)
        assert back == pytest.approx((x, y), abs=1e-4)


def test_locator_target_and_decode_roundtrip():
    for x, y in ((100.0, 200.0), (12.0, 30.0), (360.0, 430.0)):
        blob = locator_target(x, y)
        assert blob.shape == (64, 64)
        assert blob.max() == pytest.approx(1.0)
        decoded = decode_locator(blob)
        assert locator_error_px(decoded, (x, y)) < 4.0


def test_locator_error_is_euclidean_pixels():
    assert locator_error_px((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# F1 bookkeeping and the pre-registered rule
# ---------------------------------------------------------------------------
def test_count_cells_masks_invalid_cells():
    target = np.zeros((4, 4), dtype=bool)
    target[0, 0] = True
    predicted = np.ones((4, 4), dtype=bool)
    valid = np.zeros((4, 4), dtype=bool)
    valid[0, :2] = True
    valid[1, 0] = True
    counts = count_cells(predicted, target, valid)
    assert counts.tp == 1
    assert counts.fp == 2
    assert counts.fn == 0
    assert counts.as_dict()["f1"] == pytest.approx(2 * 1 / (2 * 1 + 2))


def test_f1_counts_are_additive():
    a = F1Counts(tp=3, fp=1, fn=2, tn=10)
    b = F1Counts(tp=1, fp=1, fn=1, tn=1)
    total = a.add(b)
    assert (total.tp, total.fp, total.fn, total.tn) == (4, 2, 3, 11)
    assert total.f1 == pytest.approx(2 * 4 / (2 * 4 + 2 + 3))


def _result(f1: float) -> dict:
    return {"micro": {"f1": f1}}


def test_selection_prefers_the_smallest_qualifying_input():
    decision = select_input({"A": _result(0.80), "B": _result(0.95),
                             "C": _result(0.97), "D": _result(0.96)})
    assert decision["f1"]["B"] == pytest.approx(0.95)
    assert decision["chosen"] == "B"
    assert decision["stop_loss"] is False


def test_selection_floor_rejects_everything_below_090():
    decision = select_input({"A": _result(0.50), "B": _result(0.60),
                             "C": _result(0.70), "D": _result(0.80)})
    assert decision["chosen"] is None
    assert decision["stop_loss"] is True


def test_selection_share_of_best_rejects_a_far_behind_input():
    # best 1.00, so the 0.95 share bar is 0.95: A at 0.94 does not qualify.
    decision = select_input({"A": _result(0.94), "B": _result(1.00),
                             "C": _result(0.99), "D": _result(0.98)})
    assert decision["qualified"]["A"] is False
    assert decision["chosen"] == "B"
    assert "0.95" in decision["reasons"]["A"]


def test_selection_disqualifies_d_when_the_locator_is_worse_than_4px():
    locator = {"median_px": 4.5, "gate_passed": False}
    decision = select_input({"A": _result(0.91), "B": _result(0.96),
                             "C": _result(0.97), "D": _result(1.00)},
                            locator=locator)
    assert decision["qualified"]["D"] is False
    assert decision["chosen"] == "B"
    assert "locator" in decision["reasons"]["D"]


def test_selection_disqualifies_d_without_a_locator_measurement():
    decision = select_input({"A": _result(0.95), "B": _result(0.96),
                             "C": _result(0.97), "D": _result(0.99)})
    assert decision["qualified"]["D"] is False
    assert decision["chosen"] == "A"


def test_selection_notes_resolution_is_not_the_bottleneck():
    locator = {"median_px": 2.0, "gate_passed": True}
    decision = select_input({"A": _result(0.93), "B": _result(0.94),
                             "C": _result(0.95), "D": _result(0.96)},
                            locator=locator)
    assert decision["chosen"] == "A"
    assert decision["resolution_not_the_bottleneck"] is True


def test_split_is_the_pre_registered_one():
    assert list(TRAIN_SPLIT) == list(range(1, 15))
    assert list(TEST_SPLIT) == [15, 16]


# ---------------------------------------------------------------------------
# D0 replay plan parsing (no engine)
# ---------------------------------------------------------------------------
def _write_labels(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_load_replay_rows_accepts_both_hash_spellings(tmp_path):
    path = tmp_path / "labels.jsonl"
    _write_labels(path, [
        {"kind": "decision", "decision": 0, "step_frames": [65, 65, 65, 65],
         "hash_pre": "7f04c7361749ed9f", "hash_post": "e3b18537a740bc49"},
        {"kind": "decision", "decision": 1, "step_frames": [1, 2, 3, 4],
         "hash_pre": {"hash": "aaaaaaaaaaaaaaaa", "frames": 4},
         "hash_post": {"hash": "bbbbbbbbbbbbbbbb", "frames": 8}},
        {"kind": "report", "status": "ok"},
    ])
    rows = load_replay_rows(path)
    assert len(rows) == 2
    assert rows[0] == ReplayRow(0, (65, 65, 65, 65), "7f04c7361749ed9f",
                                "e3b18537a740bc49", 0, 4)
    assert rows[1].hash_pre == "aaaaaaaaaaaaaaaa"
    assert rows[1].frame_post == 8


def test_load_replay_rows_fails_closed(tmp_path):
    path = tmp_path / "labels.jsonl"
    _write_labels(path, [
        {"kind": "decision", "decision": 0, "step_frames": [1, 2],
         "hash_pre": "7f04c7361749ed9f", "hash_post": "e3b18537a740bc49"}])
    with pytest.raises(CollectError):
        load_replay_rows(path)
    _write_labels(path, [
        {"kind": "decision", "decision": 0, "step_frames": [1, 2, 3, 4],
         "hash_pre": "not-a-hash", "hash_post": "e3b18537a740bc49"}])
    with pytest.raises(CollectError):
        load_replay_rows(path)
    _write_labels(path, [
        {"kind": "decision", "decision": 1, "step_frames": [1, 2, 3, 4],
         "hash_pre": "7f04c7361749ed9f", "hash_post": "e3b18537a740bc49"}])
    with pytest.raises(CollectError):
        load_replay_rows(path)


def test_load_replay_rows_honours_limit(tmp_path):
    path = tmp_path / "labels.jsonl"
    _write_labels(path, [
        {"kind": "decision", "decision": i, "step_frames": [1, 2, 3, 4],
         "hash_pre": "%016x" % (i + 1), "hash_post": "%016x" % (i + 2)}
        for i in range(5)])
    assert len(load_replay_rows(path, limit=3)) == 3


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------
def _load_tool():
    path = REPO_ROOT / "tools" / "rl" / "v400_p3_probe.py"
    spec = importlib.util.spec_from_file_location("v400_p3_probe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_seed_parsing_and_shard_lookup(tmp_path):
    tool = _load_tool()
    assert tool.parse_seeds("1-3,7") == [1, 2, 3, 7]
    with pytest.raises(Exception):
        tool.parse_seeds("2,2")
    shard_root = tmp_path / "shards"
    (shard_root / "dual-s0d1seed5").mkdir(parents=True)
    (shard_root / "dual-s0d1seed5" / "shard.json").write_text("{}", encoding="utf-8")
    assert tool.shard_dir_for(shard_root, 5).name == "dual-s0d1seed5"
    with pytest.raises(FileNotFoundError):
        tool.shard_dir_for(shard_root, 6)


def test_cli_dry_run_does_not_touch_anything(tmp_path, capsys):
    tool = _load_tool()
    shard_root = tmp_path / "shards"
    shard = shard_root / "dual-s0d1seed1"
    shard.mkdir(parents=True)
    (shard / "shard.json").write_text("{}", encoding="utf-8")
    exit_code = tool.main([
        "collect", "--seeds", "1", "--dry-run",
        "--shard-root", str(shard_root),
        "--data-template", str(tmp_path / "nope"),
        "--out-dir", str(tmp_path / "out"),
        "--report-dir", str(tmp_path / "reports"),
    ])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["binary_exists"] is False
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "reports").exists()


def test_cli_help_exits_zero():
    tool = _load_tool()
    with pytest.raises(SystemExit) as info:
        tool.main(["--help"])
    assert info.value.code == 0


# ---------------------------------------------------------------------------
# end-to-end (tiny): a synthetic shard goes through train + evaluate
# ---------------------------------------------------------------------------
def _write_synthetic_shard(directory: Path, seed: int, count: int) -> None:
    rng = np.random.default_rng(seed)
    playfield = np.zeros((count, 448, 384), dtype=np.uint8)
    cells = np.zeros((count, 24, 24), dtype=bool)
    valid = np.ones((count, 24, 24), dtype=bool)
    danger = np.zeros((count, 24, 24), dtype=bool)
    anchors = np.zeros((count, 2), dtype=np.int16)
    players = np.zeros((count, 2), dtype=np.float32)
    for row in range(count):
        x = float(rng.uniform(60, 324))
        y = float(rng.uniform(60, 388))
        blobs = rng.integers(0, 2, size=6)
        for index, on in enumerate(blobs):
            if not on:
                continue
            bx = int(np.clip(x + rng.integers(-30, 30), 0, 380))
            by = int(np.clip(y + rng.integers(-30, 30), 0, 444))
            playfield[row, by:by + 4, bx:bx + 4] = 255
        anchor_x, anchor_y = labels.snap_anchor(x, y)
        anchors[row] = (anchor_y // 4, anchor_x // 4)
        players[row] = (x, y)
        # the label is the presence of a bright blob: learnable from pixels
        for r in range(24):
            for c in range(24):
                px = anchor_x + c * 4
                py = anchor_y + r * 4
                if 0 <= py < 448 - 4 and 0 <= px < 384 - 4:
                    if playfield[row, py:py + 4, px:px + 4].any():
                        cells[row, r, c] = True
    np.save(directory / f"native_s{seed}.npy", playfield)
    np.savez_compressed(
        directory / f"labels_s{seed}.npz",
        cells=np.packbits(cells.reshape(count, -1), axis=1),
        valid=np.packbits(valid.reshape(count, -1), axis=1),
        danger=np.packbits(danger.reshape(count, -1), axis=1),
        anchors=anchors, players=players,
        hurt=np.tile(np.asarray([-0.8, -0.8, 0.8, 0.8], dtype=np.float32), (count, 1)),
        would_die=np.zeros((count, 2), dtype=np.int8),
        danger_frames=np.zeros(count, dtype=np.int8),
        instant_frames=np.arange(count, dtype=np.int64),
        observed_frames=np.arange(count, dtype=np.int64),
        hashes_pre=np.asarray(["0" * 16] * count, dtype="U16"),
        hashes_post=np.asarray(["1" * 16] * count, dtype="U16"),
        stage_start=np.asarray([0, 1, 0, seed], dtype=np.int64),
    )


def test_training_end_to_end_on_a_synthetic_shard(tmp_path):
    from touhou_rl.v400.student.probe.train import (
        evaluate_probe, naive_baselines, train_probe)

    torch = pytest.importorskip("torch")
    for seed in (1, 2):
        _write_synthetic_shard(tmp_path, seed, 24)
    trained = train_probe("A", train_dir=tmp_path, seeds=(1,),
                          train_seed=0, epochs=1, batch=8, device="cpu")
    evaluation = evaluate_probe(trained["model"], "A", test_dir=tmp_path,
                                seeds=(2,), batch=8, device="cpu")
    assert 0.0 <= evaluation["micro"]["f1"] <= 1.0
    assert evaluation["micro"]["tp"] >= 0
    assert evaluation["anchor_source"] == "truth"
    assert set(evaluation["per_seed"]) == {"2"}
    baseline = naive_baselines(test_dir=tmp_path, seeds=(2,))
    assert baseline["never"]["f1"] == 0.0
    assert baseline["always"]["f1"] > 0.0
    for shard in trained["shards"]:
        shard.close()


# ---------------------------------------------------------------------------
# D+loc: predicted anchors, and the resumable merge
# ---------------------------------------------------------------------------
def test_evaluate_probe_rejects_predicted_anchors_for_whole_field_inputs(tmp_path):
    from touhou_rl.v400.student.probe.train import TrainError, evaluate_probe

    pytest.importorskip("torch")
    _write_synthetic_shard(tmp_path, 2, 8)
    with pytest.raises(TrainError):
        evaluate_probe(object(), "A", test_dir=tmp_path, seeds=(2,),
                       anchor_cells={2: np.zeros((8, 2), dtype=np.int64)})


def test_evaluate_probe_accepts_predicted_anchors_for_d(tmp_path):
    """D+loc must run with shifted anchors and still count cells, not crash."""

    from touhou_rl.v400.student.probe.train import evaluate_probe

    pytest.importorskip("torch")
    _write_synthetic_shard(tmp_path, 1, 16)
    _write_synthetic_shard(tmp_path, 2, 16)
    from touhou_rl.v400.student.probe.train import train_probe

    trained = train_probe("D", train_dir=tmp_path, seeds=(1,), train_seed=0,
                          epochs=1, batch=8, device="cpu")
    labels = np.load(tmp_path / "labels_s2.npz")
    shifted = labels["anchors"].astype(np.int64) + np.asarray([1, 0])
    evaluation = evaluate_probe(trained["model"], "D", test_dir=tmp_path,
                                seeds=(2,), batch=8, device="cpu",
                                anchor_cells={2: shifted})
    assert evaluation["anchor_source"] == "locator"
    assert evaluation["windows"] == 16
    assert 0.0 <= evaluation["micro"]["f1"] <= 1.0
    for shard in trained["shards"]:
        shard.close()


def _fake_run(f1: float, precision: float, recall: float) -> dict:
    return {"test": {"micro": {"f1": f1, "precision": precision,
                               "recall": recall}}}


def _write_input_report(directory: Path, name: str, values: list[float]) -> None:
    runs = [_fake_run(value, value + 0.01, value - 0.02) for value in values]
    payload = {
        "input": name,
        "runs": runs,
        "micro_f1": {"mean": sum(values) / len(values), "min": min(values),
                     "max": max(values),
                     "values": [round(value, 6) for value in values]},
    }
    (directory / f"train-{name}.json").write_text(
        json.dumps(payload), encoding="utf-8")


def test_micro_summary_reports_sd_and_precision_recall():
    tool = _load_tool()
    summary = tool._micro_summary(_write_input_payload([0.80, 0.90, 1.00]))
    assert summary["mean"] == pytest.approx(0.90)
    assert summary["sd"] == pytest.approx(0.10)
    assert summary["range"] == pytest.approx(0.20)
    assert summary["precision"] == [0.8, 0.9, 1.0]


def _write_input_payload(values: list[float]) -> dict:
    return {"input": "A", "runs": [_fake_run(value, value, value)
                                   for value in values],
            "micro_f1": {"mean": sum(values) / len(values)}}


def test_merge_assembles_four_inputs_and_applies_the_rule(tmp_path):
    tool = _load_tool()
    source = tmp_path / "source"
    source.mkdir()
    _write_input_report(source, "A", [0.50, 0.54, 0.58])
    _write_input_report(source, "B", [0.85, 0.86, 0.87])
    _write_input_report(source, "C", [0.92, 0.93, 0.94])
    _write_input_report(source, "D", [0.95, 0.96, 0.97])
    (source / "locator.json").write_text(json.dumps(
        {"median_px": 1.5, "gate_px": 4.0, "gate_passed": True}),
        encoding="utf-8")
    exit_code = tool.main([
        "merge", "--run-dir", str(tmp_path), "--report-dir", str(source),
        "--search-dir", str(source), "--no-baselines"])
    assert exit_code == 0
    payload = json.loads((source / "train-all.json").read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert set(payload["results"]) == {"A", "B", "C", "D"}
    assert payload["decision"]["chosen"] == "D"  # A<B<D<C by input pixels
    assert payload["decision"]["qualified"]["C"] is True
    assert payload["decision"]["qualified"]["B"] is False
    assert payload["decision"]["stop_loss"] is False
    assert payload["micro_f1"]["C"]["mean"] == pytest.approx(0.93)


def test_merge_fails_closed_when_an_input_is_missing(tmp_path):
    tool = _load_tool()
    source = tmp_path / "source"
    source.mkdir()
    _write_input_report(source, "A", [0.50])
    _write_input_report(source, "B", [0.85])
    exit_code = tool.main([
        "merge", "--run-dir", str(tmp_path), "--report-dir", str(source),
        "--search-dir", str(source), "--no-baselines"])
    assert exit_code == tool.EXIT_FAILED
    assert not (source / "train-all.json").is_file()

