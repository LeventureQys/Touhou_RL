"""EXP-P1-014 D0 dual channel: alignment gate, decision mapping, roundtrip, resume.

Everything here runs **without an engine**.  The pixel arrays are synthetic
(constant or ramp-valued), and every test that writes a shard says so: these
tests prove the *code path* (hash-alignment gate, frame/decision arithmetic, the
EXP-P1-013 writer/loader roundtrip, shard resume refusal), never that a real D0
dataset exists.  A real D0 shard additionally needs ``channel_a.json`` and
``channel_b.json`` from two real engine runs whose checkpoints matched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "tools" / "rl")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import v400_d0_dual as dual  # noqa: E402

from touhou_rl.v400.student import dataset, obs_spec  # noqa: E402
from touhou_rl.v400.teacher import obs  # noqa: E402


FRAMES_PER_DECISION = dual.DEFAULT_FRAMES_PER_DECISION


# ---------------------------------------------------------------------------
# fixtures / helpers (synthetic, labelled as such)
# ---------------------------------------------------------------------------
def _hash(index: int) -> str:
    return f"{index:016x}"


def label_rows(count: int = 3, *, start: int = 0) -> list[dict]:
    rows = []
    for index in range(count):
        pre = index * FRAMES_PER_DECISION + start
        rows.append({
            "decision": index,
            "frame": pre + FRAMES_PER_DECISION,
            "frame_pre": pre,
            "hash_pre": {"hash": _hash(1000 + pre), "frames": pre},
            "frame_post": pre + FRAMES_PER_DECISION,
            "hash_post": {"hash": _hash(1000 + pre + FRAMES_PER_DECISION),
                          "frames": pre + FRAMES_PER_DECISION},
            "step_frames": [1, 2, 4, 8],
            "scores": [float(value) for value in range(obs_spec.N_MOVE_CLASSES)],
            "verified_mask": [bool(value % 2) for value in range(obs_spec.N_MOVE_CLASSES)],
            "chosen_plan": {"first_action": 1, "slots": [1, 2]},
            "margin": 0.5,
            "bomb": 0,
            "stage": 0,
            "difficulty": 1,
            "frame_event_flags": 0,
            "shoot_release": False,
            "bomb_death_available": True,
        })
    return rows


def synthetic_pixels(count: int) -> dict[str, np.ndarray]:
    """SYNTHETIC planes with a distinct value per stored render frame."""

    play = np.zeros((count, obs_spec.FRAME_STACK, *obs_spec.PLAY_SHAPE), dtype=np.uint8)
    hud = np.zeros((count, *obs_spec.HUD_SHAPE), dtype=np.uint8)
    render_frames = np.zeros((count, obs_spec.FRAME_STACK), dtype=np.int64)
    pads = np.zeros(count, dtype=np.int64)
    decision_frames = np.zeros(count, dtype=np.int64)
    for index in range(count):
        decision_frame = index * FRAMES_PER_DECISION
        decision_frames[index] = decision_frame
        if index == 0:
            pads[index] = obs_spec.FRAME_STACK - 1
            render_frames[index] = [0, 0, 0, 0]
        else:
            render_frames[index] = list(
                range(decision_frame - obs_spec.FRAME_STACK, decision_frame))
        play[index] = np.arange(obs_spec.FRAME_STACK, dtype=np.uint8)[:, None, None]
        hud[index] = np.uint8(index + 1)
    return {
        "play": play,
        "hud": hud,
        "render_frames": render_frames,
        "pads": pads,
        "decision_frames": decision_frames,
    }


def alignment_doc(rows: list[dict], *, mismatch_at: int | None = None,
                  mismatch_boundary: str = "pre") -> dict:
    pairs = []
    for row in rows:
        for boundary, frame, checkpoint in (
            ("pre", row["frame_pre"], row["hash_pre"]),
            ("post", row["frame_post"], row["hash_post"]),
        ):
            value = checkpoint["hash"]
            env_hash = value
            if (mismatch_at is not None and row["decision"] == mismatch_at
                    and boundary == mismatch_boundary):
                env_hash = _hash(999999)
            pairs.append({
                "decision": row["decision"],
                "boundary": boundary,
                "frame": frame,
                "shadow": value,
                "env": env_hash,
                "env_frames": frame,
                "equal": env_hash == value,
                "source": "A/B",
            })
    return {
        "schema": "v400-d0-dual-alignment-1",
        "method": "hash-checkpoint",
        "decisions": len(rows),
        "boundaries": len(pairs),
        "checkpoints_equal": sum(1 for row in pairs if row["equal"]),
        "checkpoints_mismatched": sum(1 for row in pairs if not row["equal"]),
        "first_mismatch": next((row for row in pairs if not row["equal"]), None),
        "pairs": pairs,
        "pairs_total": len(pairs),
        "all_equal": all(row["equal"] for row in pairs),
        "sha256": dual._canonical_sha256(pairs),
    }


def args_for(tmp_path: Path, *, count: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        decisions=count,
        stage=0,
        difficulty=1,
        team=0,
        seed=1,
        mode="fast",
        out_dir=str(tmp_path / "d0"),
        alignment_json=str(tmp_path / "alignment.json"),
        shard_key="dual-test",
        max_pairs=512,
    )


def write_alignment(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def channel_reports() -> tuple[dict, dict]:
    a = {"decisions": 3, "labels_sha256": "a" * 64, "wall_s": 1.0, "exit_code": 0,
         "engine_binary": "/build/rl_hazard2/th08-modern",
         "engine_sha256": "e" * 64, "channel": "A", "role": "shadow"}
    b = {"decisions": 3, "frames_read": 12, "pixel_ms_per_frame": 1.0, "wall_s": 1.0,
         "engine_binary": "/build/rl_hazard2/th08-modern",
         "engine_sha256": "e" * 64, "channel": "B", "role": "env"}
    return a, b


def build_shard(tmp_path: Path, *, count: int = 3, mismatch_at: int | None = None):
    rows = label_rows(count)
    write_alignment(tmp_path / "alignment.json", alignment_doc(rows, mismatch_at=mismatch_at))
    args = args_for(tmp_path, count=count)
    a, b = channel_reports()
    summary = dual.assemble(args, rows, synthetic_pixels(count), a, b)
    return summary, rows, args


# ---------------------------------------------------------------------------
# channel A parsing is strict
# ---------------------------------------------------------------------------
def _write_labels(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {"kind": "decision", "schema": "v400-teacher-obs-1", **row}
            handle.write(json.dumps(payload) + "\n")


def test_load_labels_accepts_the_checkpoint_record(tmp_path):
    path = tmp_path / "labels.jsonl"
    _write_labels(path, label_rows(3))
    rows = dual.load_labels(path)
    assert [row["frame_pre"] for row in rows] == [0, 4, 8]
    assert [row["frame_post"] for row in rows] == [4, 8, 12]
    assert rows[0]["step_frames"] == [1, 2, 4, 8]
    assert len(rows[0]["scores"]) == obs_spec.N_MOVE_CLASSES


def test_load_labels_rejects_a_row_without_per_frame_masks(tmp_path):
    rows = label_rows(2)
    rows[1]["step_frames"] = [1, 2]
    path = tmp_path / "labels.jsonl"
    _write_labels(path, rows)
    with pytest.raises(dual.DualError, match="per-frame masks"):
        dual.load_labels(path)


def test_load_labels_rejects_a_non_hash_checkpoint(tmp_path):
    rows = label_rows(2)
    rows[1]["hash_post"] = {"hash": "not-a-hash", "frames": 8}
    path = tmp_path / "labels.jsonl"
    _write_labels(path, rows)
    with pytest.raises(dual.DualError, match="not an engine HASH"):
        dual.load_labels(path)


def test_load_labels_rejects_frame_and_decision_disagreement(tmp_path):
    rows = label_rows(2)
    rows[1]["hash_pre"] = {"hash": _hash(1), "frames": 3}
    path = tmp_path / "labels.jsonl"
    _write_labels(path, rows)
    with pytest.raises(dual.DualError, match="hash_pre frames"):
        dual.load_labels(path)


# ---------------------------------------------------------------------------
# the alignment gate itself
# ---------------------------------------------------------------------------
def test_pair_equality_is_hash_and_frame_together():
    good = dual._pair(3, "pre", 12, _hash(5), {"hash": _hash(5), "frames": 12}, "A/B")
    assert good["equal"] is True
    wrong_hash = dual._pair(3, "pre", 12, _hash(5), {"hash": _hash(6), "frames": 12}, "A/B")
    assert wrong_hash["equal"] is False
    wrong_frame = dual._pair(3, "pre", 12, _hash(5), {"hash": _hash(5), "frames": 13}, "A/B")
    assert wrong_frame["equal"] is False


def test_alignment_fails_closed_and_never_writes_a_shard(tmp_path):
    rows = label_rows(3)
    write_alignment(tmp_path / "alignment.json", alignment_doc(rows, mismatch_at=1))
    args = args_for(tmp_path)
    a, b = channel_reports()
    with pytest.raises(dual.AlignError, match="mismatched checkpoint"):
        dual.assemble(args, rows, synthetic_pixels(3), a, b)
    assert not (tmp_path / "d0" / "shards").exists()


def test_a_single_mismatch_is_the_first_mismatch(tmp_path):
    rows = label_rows(4)
    doc = alignment_doc(rows, mismatch_at=2)
    assert doc["checkpoints_equal"] == doc["boundaries"] - 1
    first = doc["first_mismatch"]
    assert first["decision"] == 2 and first["boundary"] == "pre"
    assert first["frame"] == 8 and first["shadow"] != first["env"]
    assert doc["all_equal"] is False


def test_shard_assembly_refuses_a_shadow_that_moved_between_decisions(tmp_path):
    rows = label_rows(3)
    rows[1]["hash_pre"] = {"hash": _hash(424242), "frames": 4}  # breaks post(0) == pre(1)
    write_alignment(tmp_path / "alignment.json", alignment_doc(rows))
    args = args_for(tmp_path)
    a, b = channel_reports()
    with pytest.raises(dual.AlignError, match="channel A moved"):
        dual.assemble(args, rows, synthetic_pixels(3), a, b)
    assert not (tmp_path / "d0" / "shards").exists()


# ---------------------------------------------------------------------------
# decision -> frame mapping, and the writer/loader contract
# ---------------------------------------------------------------------------
def test_decision_index_maps_to_frames_d_minus_4_to_d_minus_1(tmp_path):
    summary, _, _ = build_shard(tmp_path, count=4)
    shard = dataset.load_teacher_shard(summary["dir"])
    assert len(shard) == 4
    for index in range(1, 4):
        sample = shard[index]
        observation = shard.records[index]["obs"]
        expected = [index * FRAMES_PER_DECISION - 4 + offset for offset in range(4)]
        assert observation["render_frames"] == expected
        assert observation["newest_render_frame"] + 1 == index * FRAMES_PER_DECISION
        assert observation["p3_label_frame"] == index * FRAMES_PER_DECISION
        assert observation["pad"] == 0
        assert int(sample["play"].shape[0]) == obs_spec.FRAME_STACK
    first = shard.records[0]["obs"]
    assert first["pad"] == obs_spec.FRAME_STACK - 1
    assert first["render_frames"] == [0, 0, 0, 0]
    # The plane row really is the one the label was recorded at.
    assert int(shard.play[2][0][0][0]) == 0
    assert int(shard.hud[2][0][0]) == 3


def test_dataset_roundtrip_loads_n_samples_with_obs_shapes(tmp_path):
    summary, rows, args = build_shard(tmp_path, count=5)
    loaded = dataset.load_teacher_samples(
        [Path(summary["dir"]) / "labels.jsonl"],
        min_samples=5,
        obs_root=Path(args.out_dir),
        require_observations=True,
    )
    assert loaded["count"] == 5
    assert np.asarray(loaded["samples"][0]["scores"]).shape == (obs_spec.N_MOVE_CLASSES,)
    assert np.asarray(loaded["samples"][0]["verified_mask"]).shape == (obs_spec.N_MOVE_CLASSES,)
    assert np.asarray(loaded["samples"][0]["obs_arrays"]["play"]).shape == (
        obs_spec.FRAME_STACK, *obs_spec.PLAY_SHAPE)
    assert np.asarray(loaded["samples"][0]["obs_arrays"]["hud"]).shape == obs_spec.HUD_SHAPE
    assert loaded["obs_spec_digest"] == obs.FROZEN_OBS_SPEC_DIGEST
    assert loaded["obs_unresolved"] == []
    assert loaded["alignment"] == dataset.TEACHER_OBSERVATION_TIME
    normalized = dataset.load_teacher_shard(summary["dir"]).normalized(1)
    assert normalized["play"].dtype == np.float32
    assert normalized["hud"].dtype == np.float32


def test_shard_json_records_both_channels_and_the_hash_alignment(tmp_path):
    summary, rows, _ = build_shard(tmp_path, count=3)
    payload = json.loads((Path(summary["dir"]) / "shard.json").read_text(encoding="utf-8"))
    assert payload["label_channel"] == "shadow"
    assert payload["pixel_channel"] == "env"
    assert payload["alignment_method"] == "hash-checkpoint"
    assert payload["alignment"]["pixel_source"] == "env_replay"
    assert payload["alignment"]["hash_alignment"] == "hash-checkpoint"
    checkpoints = payload["alignment"]["hash_checkpoints"]
    assert checkpoints["method"] == "hash-checkpoint"
    assert checkpoints["boundaries"] == 2 * 3
    assert checkpoints["checkpoints_equal"] == 2 * 3
    assert checkpoints["first_mismatch"] is None
    assert payload["engine_shadow"]["role"] == "shadow"
    assert payload["engine_env"]["role"] == "env"
    assert payload["engine_shadow"]["sha256"] == payload["engine_env"]["sha256"]
    # The frozen P3 alignment contract the loader checks is still intact.
    for key in ("observation_time", "stack", "stack_order", "decision_frames",
                "p3_window_last_row", "p3_window"):
        assert key in payload["alignment"]
    manifest = dataset.load_teacher_manifest(Path(summary["dir"]).parent.parent)
    assert manifest["totals"]["shards"] == 1
    assert manifest["totals"]["decisions"] == 3


def test_shard_assembly_refuses_to_overwrite_or_resume(tmp_path):
    build_shard(tmp_path, count=3)
    args = args_for(tmp_path)
    a, b = channel_reports()
    with pytest.raises(obs.ObservationError, match="already exists"):
        dual.assemble(args, label_rows(3), synthetic_pixels(3), a, b)
    partial = Path(args.out_dir) / "shards" / ".dual-test.partial"
    assert not partial.exists()


def test_labels_only_cli_refuses_the_pixel_and_legacy_modes():
    spec = importlib.util.spec_from_file_location(
        "v400_teacher_d0_test", ROOT / "tools" / "rl" / "v400_teacher.py")
    teacher = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(teacher)
    base = ["--decisions", "1", "--stage", "0", "--difficulty", "1"]
    assert teacher.main([*base, "--labels-only", "--save-observations", "x"]) == 3
    assert teacher.main([*base, "--labels-only", "--follow-stage"]) == 3
    assert teacher.main([*base, "--labels-only", "--allow-legacy-restore"]) == 3
    parsed = teacher.parse_args([*base, "--labels-only"])
    assert parsed.labels_only is True


# ---------------------------------------------------------------------------
# EXP-P1-018 scale-up: engine slots, shard resume, early stop, assemble lock
# ---------------------------------------------------------------------------
def test_engine_slot_counts_only_this_batchs_engines(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(stdout="11\n12\n13\n")

    monkeypatch.setattr(dual.subprocess, "run", fake_run)
    assert dual.engine_processes("th08-modern --data-dir /root/v400-data/d0s") == 3
    assert calls[-1][:2] == ["pgrep", "-f"]
    assert calls[-1][2] == "th08-modern --data-dir /root/v400-data/d0s"
    assert dual.engine_processes() == 3
    assert calls[-1][:2] == ["pgrep", "-x"]


def test_engine_slot_waits_for_a_limit_then_gives_up(monkeypatch):
    counts = iter([4, 4])
    monkeypatch.setattr(dual, "engine_processes", lambda pattern=None: next(counts))
    with pytest.raises(SystemExit) as failure:
        dual.wait_for_engine_slot(limit=4, pattern="x", timeout=0.0, log=lambda *a: None)
    assert "ENGINE_BUSY_AFTER_0S" in str(failure.value)


def test_engine_slot_returns_under_the_limit_without_waiting(monkeypatch):
    monkeypatch.setattr(dual, "engine_processes", lambda pattern=None: 2)
    slot = dual.wait_for_engine_slot(limit=4, pattern="x", timeout=0.0, log=lambda *a: None)
    assert slot["limit"] == 4 and slot["pattern"] == "x"
    assert slot["samples"] == [{"t": 0.0, "engines": 2}]


def test_resume_report_accepts_a_finished_shard(tmp_path):
    summary, _, args = build_shard(tmp_path, count=4)
    resumed = dual.shard_resume_report(summary["dir"])
    assert resumed is not None
    assert resumed["decisions"] == 4 and resumed["resumed"] is True


def test_resume_report_refuses_a_staging_directory(tmp_path):
    partial = tmp_path / "d0" / "shards" / ".dual-test.partial"
    partial.mkdir(parents=True)
    assert dual.shard_resume_report(partial) is None
    assert dual.shard_resume_report(tmp_path / "nowhere") is None


def test_resume_report_refuses_a_corrupt_shard(tmp_path):
    summary, _, _ = build_shard(tmp_path, count=3)
    (Path(summary["dir"]) / "obs" / "play64.npy").write_bytes(b"not a npy")
    assert dual.shard_resume_report(summary["dir"]) is None


def test_main_skips_a_completed_shard_without_touching_the_engine(tmp_path, monkeypatch):
    summary, _, args = build_shard(tmp_path, count=3)

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("a completed shard must not start a channel")

    monkeypatch.setattr(dual, "run_labels", explode)
    monkeypatch.setattr(dual, "run_pixels", explode)
    monkeypatch.setattr(dual.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    code = dual.main([
        "--phase", "all", "--decisions", "3", "--stage", "0", "--difficulty", "1",
        "--seed", "1", "--shard-key", "dual-test",
        "--work-dir", str(tmp_path / "work"), "--out-dir", str(tmp_path / "d0"),
        "--engine-limit", "4", "--engine-pattern", "nothing",
    ])
    assert code == 0


def test_main_refuses_to_start_next_to_an_interrupted_staging_dir(tmp_path):
    partial = tmp_path / "d0" / "shards" / ".dual-test.partial"
    partial.mkdir(parents=True)
    with pytest.raises(dual.DualError, match="interrupted run"):
        dual.main([
            "--phase", "all", "--decisions", "3", "--stage", "0", "--difficulty", "1",
            "--seed", "1", "--shard-key", "dual-test",
            "--work-dir", str(tmp_path / "work"), "--out-dir", str(tmp_path / "d0"),
        ])
    assert partial.exists()  # evidence is kept, never deleted


def _fake_teacher(tmp_path: Path, *, rows: int, requested: int) -> Path:
    """A stand-in channel A that writes ``rows`` decisions and a report row."""

    script = tmp_path / "fake_teacher.py"
    script.write_text(
        "import argparse, json, pathlib\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--decisions', type=int)\n"
        "p.add_argument('--stage'); p.add_argument('--difficulty'); p.add_argument('--team')\n"
        "p.add_argument('--seed'); p.add_argument('--mode'); p.add_argument('--labels-only',\n"
        "               action='store_true'); p.add_argument('--out'); p.add_argument('--data-dir')\n"
        "p.add_argument('--engine'); p.add_argument('--expected-sha')\n"
        "p.add_argument('--quiet', action='store_true')\n"
        "p.add_argument('--max-frames', type=int); p.add_argument('--stop-on-death',\n"
        "               action='store_true')\n"
        "a = p.parse_args()\n"
        f"rows = {rows}\n"
        "handle = open(a.out, 'w')\n"
        "for index in range(rows):\n"
        "    pre = index * 4\n"
        "    row = {'kind': 'decision', 'schema': 'v400-teacher-obs-1',\n"
        "           'decision': index, 'frame': pre + 4,\n"
        "           'hash_pre': {'hash': f'{1000 + pre:016x}', 'frames': pre},\n"
        "           'hash_post': {'hash': f'{1004 + pre:016x}', 'frames': pre + 4},\n"
        "           'step_frames': [1, 2, 4, 8],\n"
        "           'scores': [float(v) for v in range(18)],\n"
        "           'verified_mask': [bool(v % 2) for v in range(18)],\n"
        "           'chosen_plan': None, 'margin': None, 'bomb': 0, 'stage': 0,\n"
        "           'difficulty': 1, 'frame_event_flags': 0, 'shoot_release': False,\n"
        "           'bomb_death_available': False}\n"
        "    handle.write(json.dumps(row) + '\\n')\n"
        "handle.write(json.dumps({'kind': 'report', 'status': 'ok',\n"
        "                         'stop_reason': 'stage_changed', 'stage_clear': True,\n"
        f"                         'decisions': rows, 'frames_committed': rows * 4,\n"
        f"                         'deaths_observed': 0}}) + '\\n')\n"
        "handle.close()\n",
        encoding="utf-8")

    def args():
        return SimpleNamespace(
            decisions=requested, stage=0, difficulty=1, team=0, seed=1, mode="fast",
            teacher_tool=str(script), repo_root=str(tmp_path), engine=str(tmp_path),
            binary=None, expected_sha="e" * 64, data_template=str(tmp_path / "template"),
            data_root=str(tmp_path / "data"), work_dir=str(tmp_path),
            labels_jsonl=str(tmp_path / "labels.jsonl"), max_frames=requested * 4,
            stop_on_death=True, allow_early_stop=True,
            out_dir=str(tmp_path / "d0"), shard_key="dual-test", max_pairs=512,
        )

    template = tmp_path / "template"
    template.mkdir(exist_ok=True)
    for name in ("th08.dat", "thbgm.dat"):
        (template / name).write_bytes(b"stub")
    for name in ("th08.cfg", "score.dat"):
        (template / name).write_bytes(b"stub")
    return args


def test_early_stop_accepts_fewer_decisions_and_records_the_real_count(tmp_path, monkeypatch):
    factory = _fake_teacher(tmp_path, rows=5, requested=9)
    # The data dir is a symlink farm (Linux); keep this test to the label contract.
    monkeypatch.setattr(dual, "prepare_data_dir",
                        lambda *a, **k: {"data_dir": "stub", "reused": False})
    report = dual.run_labels(factory())
    assert report["decisions"] == 5
    assert report["decisions_requested"] == 9
    assert report["stopped_early"] is True
    assert report["stop_reason"] == "stage_changed"
    assert report["stage_clear"] is True
    labels = dual.load_labels(tmp_path / "labels.jsonl")
    assert len(labels) == 5


def test_early_stop_is_refused_without_the_flag(tmp_path, monkeypatch):
    factory = _fake_teacher(tmp_path, rows=5, requested=9)
    monkeypatch.setattr(dual, "prepare_data_dir",
                        lambda *a, **k: {"data_dir": "stub", "reused": False})
    args = factory()
    args.allow_early_stop = False
    with pytest.raises(dual.DualError, match="requested 9"):
        dual.run_labels(args)


def test_assemble_holds_a_cross_process_lock_and_keeps_each_shards_report(tmp_path):
    rows = label_rows(3)
    write_alignment(tmp_path / "alignment.json", alignment_doc(rows))
    args = args_for(tmp_path)
    a, b = channel_reports()
    with dual.out_dir_lock(args.out_dir):
        first = dual.assemble(args, rows, synthetic_pixels(3), a, b)
    lock = Path(args.out_dir) / ".assemble.lock"
    assert lock.is_file()
    args.shard_key = "dual-second"
    write_alignment(tmp_path / "alignment.json", alignment_doc(rows))
    with dual.out_dir_lock(args.out_dir):
        second = dual.assemble(args, rows, synthetic_pixels(3), a, b)
    manifest = dataset.load_teacher_manifest(Path(args.out_dir))
    assert manifest["totals"]["shards"] == 2
    assert manifest["totals"]["decisions"] == 6
    assert first["dir"] != second["dir"]


def test_early_stop_shard_meta_records_the_stop_verdict(tmp_path):
    rows = label_rows(4)
    write_alignment(tmp_path / "alignment.json", alignment_doc(rows))
    args = args_for(tmp_path, count=4)
    a, b = channel_reports()
    a = {**a, "decisions_requested": 9, "stopped_early": True,
         "stop_reason": "stage_changed", "stage_clear": True, "deaths_observed": 0,
         "frames_committed": 16}
    summary = dual.assemble(args, rows, synthetic_pixels(4), a, b)
    payload = json.loads((Path(summary["dir"]) / "shard.json").read_text(encoding="utf-8"))
    assert payload["decisions"] == 4
    assert payload["decisions_requested"] == 9
    assert payload["stopped_early"] is True
    assert payload["stop_reason"] == "stage_changed"
    assert payload["stage_clear"] is True


# ---------------------------------------------------------------------------
# EXP-P1-018: the stage-clear boundary is excluded, never faked
# ---------------------------------------------------------------------------
def _clear_channel_a(base: dict, **over) -> dict:
    return {**base, "stop_reason": "stage_changed", "stage_clear": True,
            "decisions_requested": 9, "stopped_early": True, **over}


def test_stage_clear_commit_is_trimmed_and_every_kept_checkpoint_matches(tmp_path):
    rows = label_rows(4)
    doc = alignment_doc(rows, mismatch_at=3, mismatch_boundary="post")
    write_alignment(tmp_path / "alignment.json", doc)
    assert doc["checkpoints_mismatched"] == 1
    args = args_for(tmp_path, count=3)
    a, b = channel_reports()
    # assemble() only trims when --trim-transition is on.
    args.trim_transition = True
    summary = dual.assemble(args, rows, synthetic_pixels(4), _clear_channel_a(a), b)
    assert summary["decisions"] == 3
    payload = json.loads((Path(summary["dir"]) / "shard.json").read_text(encoding="utf-8"))
    trim = payload["transition_trim"]
    assert trim["decision"] == 3
    assert trim["checkpoints_before_trim"] == 8
    assert trim["checkpoints_mismatched_before_trim"] == 1
    assert trim["checkpoints_after_trim"] == 6
    assert trim["decisions_after_trim"] == 3
    assert trim["shadow_post"] != trim["env_post"]
    hash_block = payload["alignment"]["hash_checkpoints"]
    assert hash_block["boundaries"] == 6
    assert hash_block["checkpoints_equal"] == 6
    assert hash_block["checkpoints_mismatched"] == 0
    assert hash_block["first_mismatch"] is None
    loaded = dataset.load_teacher_samples([Path(summary["dir"]) / "labels.jsonl"],
                                          min_samples=3, obs_root=Path(args.out_dir),
                                          require_observations=True)
    assert loaded["count"] == 3


def test_stage_clear_trim_accepts_pixels_that_are_already_one_shorter(tmp_path):
    """Channel B never appends the sample whose ``post`` failed (3,370 vs 3,371)."""

    rows = label_rows(4)
    write_alignment(tmp_path / "alignment.json",
                    alignment_doc(rows, mismatch_at=3, mismatch_boundary="post"))
    args = args_for(tmp_path, count=3)
    args.trim_transition = True
    a, b = channel_reports()
    summary = dual.assemble(args, rows, synthetic_pixels(3), _clear_channel_a(a), b)
    assert summary["decisions"] == 3
    payload = json.loads((Path(summary["dir"]) / "shard.json").read_text(encoding="utf-8"))
    assert payload["transition_trim"]["decisions_before_trim"] == 4
    assert payload["transition_trim"]["decisions_after_trim"] == 3
    assert len(dataset.load_teacher_shard(summary["dir"])) == 3


def test_stage_clear_trim_is_refused_without_the_flag(tmp_path):
    rows = label_rows(4)
    write_alignment(tmp_path / "alignment.json",
                    alignment_doc(rows, mismatch_at=3, mismatch_boundary="post"))
    args = args_for(tmp_path, count=3)
    a, b = channel_reports()
    with pytest.raises(dual.AlignError, match="mismatched checkpoint"):
        dual.assemble(args, rows, synthetic_pixels(4), _clear_channel_a(a), b)
    assert not (tmp_path / "d0" / "shards").exists()


def test_stage_clear_trim_never_hides_a_mismatch_before_the_last_decision(tmp_path):
    rows = label_rows(4)
    write_alignment(tmp_path / "alignment.json",
                    alignment_doc(rows, mismatch_at=2, mismatch_boundary="post"))
    args = args_for(tmp_path, count=3)
    args.trim_transition = True
    a, b = channel_reports()
    with pytest.raises(dual.AlignError, match="mismatched checkpoint"):
        dual.assemble(args, rows, synthetic_pixels(4), _clear_channel_a(a), b)
    assert not (tmp_path / "d0" / "shards").exists()


def test_stage_clear_trim_is_refused_when_the_teacher_did_not_clear(tmp_path):
    rows = label_rows(4)
    write_alignment(tmp_path / "alignment.json",
                    alignment_doc(rows, mismatch_at=3, mismatch_boundary="post"))
    args = args_for(tmp_path, count=3)
    args.trim_transition = True
    a, b = channel_reports()
    a = {**a, "stop_reason": "max_frames", "stage_clear": False}
    with pytest.raises(dual.AlignError, match="mismatched checkpoint"):
        dual.assemble(args, rows, synthetic_pixels(4), a, b)
    assert not (tmp_path / "d0" / "shards").exists()
