"""T3 D1 collector: plan arithmetic, trajectory take-over, shard roundtrip, fail-closed.

Everything here runs **without an engine**.  The pixel arrays are synthetic and the
"recorded corpus" is a hand-built D0-schema directory, so these tests prove the *code
path* (reachability -> plan -> shard spec -> rebased obs-v3 shard -> loader gate,
plus every fail-closed branch), never that a real D1 dataset exists.  The live
evidence is produced by the WSL runs recorded in EXP-P3-012.
"""

from __future__ import annotations

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

import v400_d1_collect as collect  # noqa: E402

from touhou_rl.v400.gym.trajectory import TrajectoryStartError  # noqa: E402
from touhou_rl.v400.student import dataset, obs_spec, obs_spec_v3, obs_v3_dataset  # noqa: E402
from touhou_rl.v400.teacher import obs  # noqa: E402


FRAMES = collect.FRAMES_PER_DECISION


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _hash(index: int) -> str:
    return f"{index:016x}"


def plan_args(tmp_path: Path, **over) -> SimpleNamespace:
    base = dict(
        stages="0,1", seeds="1-2", start_mix="shard",
        frames_target_total=2000, frames_target_per_stage=1000,
        decisions_per_shard=100, plan_limit_per_stage=0, plan_name="unit",
        difficulty=1, team=0, mode="fast", engine="/root/none/rl_v3",
        expected_sha="0" * 64, data_template="/root/v400-data/game",
        timeout=10.0, trajectory_bias=0.65, trajectory_boss_weight=0.25,
        boss_start=8, corpus_root=str(tmp_path / "no-corpus"),
        corpus_roots=[], work_dir=str(tmp_path / "work"),
        out_dir=str(tmp_path / "out"), data_root_prefix="/root/v400-data/d1",
        probe_cap_frames=800,
    )
    base.update(over)
    return SimpleNamespace(**base)


def reachability(*rows) -> dict:
    return {"schema": collect.REACHABILITY_SCHEMA,
            "stages": list(rows)}


def reach(stage: int, frames: int, *, clear: bool = False, death: int | None = None,
          lower_bound: bool = False) -> dict:
    return {"stage": stage, "status": "ok", "frames": frames,
            "reachable_frames": frames, "reachable_is_lower_bound": lower_bound,
            "capped": lower_bound, "stage_clear": clear, "death_frame": death,
            "stop_reason": "stage_changed" if clear else ("death" if death else "max_frames"),
            "teacher_fps": 30.0}


def normalized_rows(count: int, *, action: int = 1,
                    hash_base: int = 1000) -> list[dict]:
    """The row shape ``v400_d0_dual.load_labels`` returns (hash fields are strings)."""

    rows = []
    for index in range(count):
        pre = index * FRAMES
        rows.append({
            "decision": index,
            "frame": pre + FRAMES,
            "frame_pre": pre,
            "hash_pre": _hash(hash_base + pre),
            "frame_post": pre + FRAMES,
            "hash_post": _hash(hash_base + pre + FRAMES),
            "step_frames": [1, 2, 4, 8],
            "scores": [float(value) for value in range(obs_spec.N_MOVE_CLASSES)],
            "verified_mask": [bool(value % 2) for value in range(obs_spec.N_MOVE_CLASSES)],
            "chosen_plan": {"first_action": action, "slots": [action]},
            "margin": 0.5,
            "bomb": 0,
            "stage": 0,
            "difficulty": 1,
            "frame_event_flags": None,
            "shoot_release": False,
            "bomb_death_available": False,
        })
    return rows


def corpus_shard(root: Path, *, key: str, stage: int = 0, seed: int = 1,
                 decisions: int = 6) -> Path:
    """A minimal recorded shard ``load_d0_shard`` accepts (no obs planes needed)."""

    shard_dir = root / "shards" / key
    shard_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    pairs = []
    for index in range(decisions):
        frame = index * FRAMES
        pre = _hash(2000 + frame)
        post = _hash(2000 + frame + FRAMES)
        labels.append({"kind": "decision", "decision": index, "frame": frame,
                       "step_frames": [1, 2, 4, 8],
                       "hash_pre": pre, "hash_post": post})
        pairs.append({"decision": index, "boundary": "pre", "frame": frame,
                      "shadow": pre, "env": pre, "equal": True})
        pairs.append({"decision": index, "boundary": "post", "frame": frame + FRAMES,
                      "shadow": post, "env": post, "equal": True})
    (shard_dir / "labels.jsonl").write_text(
        "\n".join(json.dumps(row) for row in labels) + "\n", encoding="utf-8")
    (shard_dir / "shard.json").write_text(json.dumps({
        "schema": dataset.TEACHER_SHARD_SCHEMA,
        "shard": key,
        "complete": True,
        "frames_per_decision": FRAMES,
        "reset_command": f"RESET {stage} 1 0 {seed} 0",
        "start_mode": "shard",
        "decisions": decisions,
        "hash_checkpoint_pairs": pairs,
    }, indent=2), encoding="utf-8")
    return shard_dir


def write_obs_v3_shard(tmp_path: Path, *, start_mode: str, start_frame: int = 0,
                       count: int = 3, key: str | None = None):
    """Write a real obs-v3 shard through the collector's own row writer."""

    key = key or f"d1-s0d1-n{count}-seed1-{start_mode}"
    out = tmp_path / "out"
    rows = normalized_rows(count)
    history, info = collect.dual().v3_action_history(rows)
    meta = {
        "stage": 0, "difficulty": 1, "team": 0, "seed": 1, "mode": "fast",
        "frames_per_decision": FRAMES,
        "decisions_requested": count, "start_mode": start_mode,
        "start_frame": start_frame, "stage_index": 0,
        "label_channel": "shadow", "pixel_channel": "env",
        "alignment_method": "hash-checkpoint", "obs_requested": "v3",
        "obs_planes": [plane.meta_key for plane in obs.obs_version_spec("v3").planes],
        "obs_bytes_per_decision_expected": obs_spec_v3.BYTES_PER_DECISION,
        "reset_command": "RESET 0 1 0 1 0", "act_hist": info,
    }
    writer = obs.TeacherObsShardWriter(out, key=key, meta=meta, obs_version="v3")
    rng = np.random.default_rng(7)
    for index in range(count):
        relative = index * FRAMES
        absolute = start_frame + relative
        window = obs.decision_window(absolute)
        sample = {
            "play": rng.integers(0, 255, size=(obs_spec.FRAME_STACK, *obs_spec_v3.PLAY_SHAPE),
                                 dtype=np.uint8),
            "hud": rng.integers(0, 255, size=obs_spec.HUD_SHAPE, dtype=np.uint8),
            "render_frames": list(window["render_frames"]), "window": window,
            "pad": 0, "episode_start": False, "seed_frame": None,
        }
        if start_mode == "shard" and index == 0:
            sample.update({"render_frames": [0, 0, 0, 0], "pad": 3,
                           "episode_start": True})
        if start_mode == "trajectory":
            sample = collect._rebase_sample(sample, start_frame)
        collect._add_sample(writer, index, relative, rows[index], sample,
                            history[index], start_frame)
    pairs = []
    for index in range(count):
        pairs.append({"decision": index, "boundary": "pre",
                      "frame": start_frame + index * FRAMES,
                      "shadow": rows[index]["hash_pre"],
                      "env": rows[index]["hash_pre"], "equal": True, "source": "A/B"})
        pairs.append({"decision": index, "boundary": "post",
                      "frame": start_frame + (index + 1) * FRAMES,
                      "shadow": rows[index]["hash_post"],
                      "env": rows[index]["hash_post"], "equal": True, "source": "A/B"})
    writer.meta["alignment_extra"] = {
        "pixel_source": "env_replay", "label_source": "shadow_teacher",
        "start_mode": start_mode, "start_frame": start_frame,
        "hash_alignment": "hash-checkpoint",
        "hash_checkpoints": {"method": "hash-checkpoint", "boundaries": len(pairs),
                            "decisions": count, "checkpoints_equal": len(pairs),
                            "checkpoints_mismatched": 0, "first_mismatch": None,
                            "trimmed_transition": None},
    }
    writer.meta["hash_checkpoint_pairs"] = pairs
    writer.meta["decisions"] = count
    writer.meta["frames"] = count * FRAMES
    summary = writer.finalize(report={"d1": {"start_mode": start_mode}})
    return out, summary


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def test_stage_list_keeps_the_task_order():
    assert collect.parse_stage_list("0,1,3,2,4,5,6,7") == [0, 1, 3, 2, 4, 5, 6, 7]
    assert collect.parse_stage_list(" 0 , 0 ,1") == [0, 1]


def test_stage_list_refuses_an_engine_stage_out_of_range():
    with pytest.raises(collect.D1Error):
        collect.parse_stage_list("0,9")


def test_seed_list_expands_ranges():
    assert collect.parse_seed_list("1-3,7") == [1, 2, 3, 7]
    with pytest.raises(collect.D1Error):
        collect.parse_seed_list("5-2")


def test_start_mix_keeps_the_cycle_and_refuses_unknown_modes():
    assert collect.parse_start_mix("shard,trajectory") == ["shard", "trajectory"]
    assert collect.parse_start_mix("shard") == ["shard"]
    with pytest.raises(collect.D1Error):
        collect.parse_start_mix("shard,human")


def test_cli_defaults_are_the_task_ones():
    args = collect.parse_args(["--phase", "plan"])
    assert args.stages == "0,1,3,2,4,5,6,7"
    assert args.workers == 4
    assert args.frames_target_total == 2_000_000
    assert args.frames_target_per_stage == 150_000
    assert args.start_mix == "shard"
    assert args.obs == "v3"
    assert args.mode == "fast"
    assert args.difficulty == 1
    assert args.expected_sha.startswith("691bfd77")


# ---------------------------------------------------------------------------
# plan arithmetic: reachability -> shards, no synthetic frames
# ---------------------------------------------------------------------------
def test_plan_never_asks_for_more_frames_than_the_stage_reachable(tmp_path):
    args = plan_args(tmp_path, decisions_per_shard=500, frames_target_per_stage=1000,
                     frames_target_total=2000)
    plan = collect.build_plan(args, reachability(reach(0, 13480, clear=True),
                                                 reach(1, 1200, death=1200)),
                              {})
    assert plan["status"] == "ok"
    by_key = {spec["stage"]: spec for spec in plan["shards"]}
    # stage 0: 13,480 frames reachable -> 500 decisions fit; one shard of 1,000 frames
    assert by_key[0]["decisions"] == 500
    assert by_key[0]["frames_expected"] == 2000
    # stage 1: only 1,200 frames reachable -> 300 decisions, 1,200 frames
    assert by_key[1]["decisions"] == 300
    assert by_key[1]["frames_expected"] == 1200
    for row in plan["stages_detail"]:
        assert row["frames_per_shard"] <= row["reachable_frames"]
        assert row["frames_per_shard"] % FRAMES == 0


def test_plan_records_a_coverage_shortfall_instead_of_padding(tmp_path):
    args = plan_args(tmp_path, frames_target_per_stage=10_000,
                     frames_target_total=20_000, plan_limit_per_stage=1,
                     decisions_per_shard=100)
    plan = collect.build_plan(args, reachability(reach(0, 13_480, clear=True),
                                                 reach(1, 13_480, clear=True)), {})
    detail = {row["stage"]: row for row in plan["stages_detail"]}
    # one shard of 400 frames planned, 10,000 asked -> 9,600 short, on purpose
    assert detail[0]["shards_planned"] == 1
    assert detail[0]["frames_planned"] == 400
    assert detail[0]["coverage_shortfall"] == 9_600
    assert plan["coverage"]["shortfall_frames"] == 19_200
    # no shard claims the missing frames
    assert plan["frames_planned"] == 800
    assert plan["frames_planned"] <= plan["targets"]["total"]


def test_plan_full_size_derives_the_shard_count_from_reachability(tmp_path):
    args = plan_args(tmp_path, frames_target_per_stage=150_000,
                     frames_target_total=300_000, decisions_per_shard=4000)
    plan = collect.build_plan(args, reachability(reach(0, 13_480, clear=True),
                                                 reach(1, 40_000, lower_bound=True)), {})
    detail = {row["stage"]: row for row in plan["stages_detail"]}
    # stage 0: 4000 decisions/shard x 4 = 16,000 frames > 13,480 reachable -> 3,370
    assert detail[0]["frames_per_shard"] == 13_480
    assert detail[0]["shards_needed"] == 12  # ceil(150,000 / 13,480)
    # stage 1: the cap binds, not the reachability
    assert detail[1]["frames_per_shard"] == 16_000
    assert detail[1]["shards_needed"] == 10  # ceil(150,000 / 16,000)


def test_plan_refuses_to_run_with_a_stage_that_has_no_reachability(tmp_path):
    args = plan_args(tmp_path)
    plan = collect.build_plan(args, reachability(reach(0, 1000)), {})
    assert plan["status"] == "blocked"
    assert any("stage 1" in error for error in plan["errors"])


def test_plan_limit_per_stage_is_recorded_as_the_shortfall_reason(tmp_path):
    args = plan_args(tmp_path, plan_limit_per_stage=2, decisions_per_shard=100,
                     frames_target_per_stage=10_000, frames_target_total=20_000)
    plan = collect.build_plan(args, reachability(reach(0, 5000), reach(1, 5000)), {})
    detail = {row["stage"]: row for row in plan["stages_detail"]}
    assert detail[0]["shards_planned"] == 2
    assert detail[0]["coverage_shortfall_reason"] == "plan_limit_per_stage"


# ---------------------------------------------------------------------------
# trajectory starts: corpus index, take-over frame, plan wiring
# ---------------------------------------------------------------------------
def test_corpus_index_only_takes_shard_start_shards(tmp_path):
    root = tmp_path / "corpus"
    corpus_shard(root, key="d0-a", stage=0, seed=1)
    corpus_shard(root, key="d0-b", stage=1, seed=2)
    # a trajectory shard is re-based and must never become somebody's corpus
    (root / "shards" / "d1-traj").mkdir(parents=True)
    (root / "shards" / "d1-traj" / "shard.json").write_text(json.dumps({
        "complete": True, "start_mode": "trajectory", "shard": "d1-traj"}), encoding="utf-8")
    args = plan_args(tmp_path, corpus_root=str(root))
    index = collect.index_corpora(args)
    assert sorted(index) == [0, 1]
    assert [row["seed"] for row in index[0]] == [1]
    assert index[0][0]["source"] == "d0"


def test_takeover_frame_leaves_room_for_the_whole_shard(tmp_path):
    from touhou_rl.v400.gym.trajectory import load_d0_shard

    root = tmp_path / "corpus"
    shard_dir = corpus_shard(root, key="d0-a", decisions=6)
    traj = load_d0_shard(shard_dir)
    frame = collect.choose_takeover_frame(
        traj, decisions=1, rng=__import__("random").Random(0), bias=0.0,
        boss_start=None, boss_weight=0.0)
    assert frame % FRAMES == 0
    assert frame + FRAMES <= traj.frames
    with pytest.raises(collect.D1Error):
        collect.choose_takeover_frame(traj, decisions=7, rng=__import__("random").Random(0),
                                      bias=0.0, boss_start=None, boss_weight=0.0)


def test_plan_fails_closed_when_a_trajectory_start_has_no_corpus(tmp_path):
    args = plan_args(tmp_path, start_mix="shard,trajectory",
                     frames_target_per_stage=400, frames_target_total=800,
                     decisions_per_shard=1)
    plan = collect.build_plan(args, reachability(reach(0, 1000), reach(1, 1000)), {})
    assert plan["status"] == "blocked"
    assert any("trajectory" in error for error in plan["errors"])
    # the first shard of each stage is still a plain shard start
    assert {spec["start_mode"] for spec in plan["shards"]} == {"shard"}


def test_plan_assigns_a_verified_takeover_point_from_the_corpus(tmp_path):
    root = tmp_path / "corpus"
    corpus_shard(root, key="d0-a", stage=0, seed=1, decisions=6)
    args = plan_args(tmp_path, stages="0", start_mix="shard,trajectory",
                     frames_target_per_stage=8, frames_target_total=8,
                     decisions_per_shard=1, boss_start=8, corpus_root=str(root))
    corpora = collect.index_corpora(args)
    plan = collect.build_plan(args, reachability(reach(0, 1000), reach(1, 1000)), corpora)
    trajectory = [spec for spec in plan["shards"] if spec["start_mode"] == "trajectory"]
    assert len(trajectory) == 1
    spec = trajectory[0]
    assert spec["seed"] == 1  # the corpus seed, not the plan seed cycle
    assert spec["target_frame"] % FRAMES == 0
    assert spec["prefix_decisions"] == spec["target_frame"] // FRAMES
    assert spec["teacher_decisions"] == spec["prefix_decisions"] + spec["decisions"]
    assert spec["expected_start_hash"] == _hash(2000 + spec["target_frame"])


def test_trajectory_prefix_must_reproduce_the_recorded_tape(tmp_path):
    root = tmp_path / "corpus"
    shard_dir = corpus_shard(root, key="d0-a", decisions=6)
    spec = {"key": "d1-traj", "target_frame": 8, "corpus": {"dir": str(shard_dir)},
            "decisions": 1}
    labels = normalized_rows(4)
    labels[1]["step_frames"] = [9, 9, 9, 9]  # diverges at decision 1
    with pytest.raises(collect.AlignmentError):
        collect._verify_trajectory_prefix(spec, labels)
    labels = normalized_rows(4)
    labels[0]["hash_pre"] = "0" * 16
    with pytest.raises(collect.AlignmentError):
        collect._verify_trajectory_prefix(spec, labels)
    kept, proof = collect._verify_trajectory_prefix(spec, normalized_rows(4, hash_base=2000))
    assert len(kept) == 2 and proof["decisions"] == 2 and proof["verified"] is True


def test_trajectory_prefix_index_must_have_masks(tmp_path):
    root = tmp_path / "corpus"
    shard_dir = corpus_shard(root, key="d0-a", decisions=6)
    spec = {"key": "d1-traj", "target_frame": 8, "corpus": {"dir": str(shard_dir)},
            "decisions": 1}
    with pytest.raises(collect.D1Error):
        collect._verify_trajectory_prefix(spec, normalized_rows(1))


# ---------------------------------------------------------------------------
# shard writing: shard-start and re-based take-over shards load back
# ---------------------------------------------------------------------------
def test_shard_start_shard_loads_through_the_v3_gate(tmp_path):
    out, summary = write_obs_v3_shard(tmp_path, start_mode="shard", count=3)
    shard = obs_v3_dataset.load_teacher_shard_v3(Path(summary["dir"]))
    assert len(shard) == 3
    assert shard.meta["start_mode"] == "shard"
    assert shard.meta["decisions"] == 3
    assert shard.meta["frames"] == 3 * FRAMES
    assert shard.meta["obs_spec_version"] == "obs-v3"
    assert shard.meta["obs_spec_digest"] == obs_spec_v3.obs_spec_v3_digest()
    assert shard.play.shape == (3, obs_spec.FRAME_STACK, *obs_spec_v3.PLAY_SHAPE)
    loaded = obs_v3_dataset.load_teacher_samples_v3(
        [Path(summary["dir"]) / "labels.jsonl"], min_samples=3, obs_root=out,
        require_observations=True)
    assert loaded["count"] == 3
    assert loaded["obs_unresolved"] == []


def test_takeover_shard_rebases_frames_to_zero_and_still_loads(tmp_path):
    out, summary = write_obs_v3_shard(tmp_path, start_mode="trajectory", start_frame=8,
                                      count=2)
    shard = obs_v3_dataset.load_teacher_shard_v3(Path(summary["dir"]))
    assert len(shard) == 2
    assert shard.meta["start_mode"] == "trajectory"
    assert shard.meta["start_frame"] == 8
    assert shard.meta["stage_frames"] == [0, 2 * FRAMES]
    # decision 0 is the take-over state itself: 4 real renders, no padding
    first = shard.records[0]
    assert first["obs"]["pad"] == 0
    assert first["obs"]["render_frames"] == list(obs.decision_window(0)["render_frames"])
    loaded = obs_v3_dataset.load_teacher_samples_v3(
        [Path(summary["dir"]) / "labels.jsonl"], min_samples=2, obs_root=out,
        require_observations=True)
    assert loaded["count"] == 2
    assert loaded["obs_shapes"]["play"] == list(obs_spec_v3.PLAY_SHAPE)


def test_rebase_moves_the_observation_window_by_the_takeover_frame():
    window = obs.decision_window(8)
    sample = {"play": np.zeros((4, *obs_spec_v3.PLAY_SHAPE), dtype=np.uint8),
              "hud": np.zeros(obs_spec.HUD_SHAPE, dtype=np.uint8),
              "render_frames": list(window["render_frames"]), "window": window,
              "pad": 0}
    rebased = collect._rebase_sample(sample, 8)
    assert rebased["window"]["decision_frame"] == 0
    assert rebased["render_frames"] == [-4, -3, -2, -1]
    assert rebased["pad"] == 0


# ---------------------------------------------------------------------------
# probe: reachability record shapes
# ---------------------------------------------------------------------------
def _probe_with(monkeypatch, record: dict, *, cap: int = 800) -> dict:
    out = {"report": record, "labels": [{} for _ in range(record["decisions"])],
           "labels_jsonl": "unused"}
    monkeypatch.setattr(collect, "run_channel_a", lambda *a, **k: out)
    return collect.probe_stage({"stage": 2, "seed": 3, "mode": "fast", "difficulty": 1},
                               cap_frames=cap, work_dir=Path("unused"),
                               data_root="unused")


def test_probe_records_a_death_as_the_reachable_length(monkeypatch):
    record = _probe_with(monkeypatch, {
        "decisions": 150, "stop_reason": "death", "stage_clear": False,
        "deaths_observed": 1, "frames_committed": 600, "wall_s": 20.0,
    })
    assert record["status"] == "ok"
    assert record["frames"] == 600
    assert record["reachable_frames"] == 600
    assert record["death_frame"] == 600
    assert record["capped"] is False
    assert record["reachable_is_lower_bound"] is False
    assert record["teacher_fps"] == 30.0


def test_probe_marks_a_capped_episode_as_a_lower_bound(monkeypatch):
    record = _probe_with(monkeypatch, {
        "decisions": 200, "stop_reason": "max_frames", "stage_clear": False,
        "deaths_observed": 0, "frames_committed": 800, "wall_s": 40.0,
    })
    assert record["frames"] == 800
    assert record["capped"] is True
    assert record["reachable_is_lower_bound"] is True


def test_probe_records_a_failure_instead_of_crashing(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("engine busy")

    monkeypatch.setattr(collect, "run_channel_a", boom)
    record = collect.probe_stage({"stage": 4, "seed": 1, "mode": "fast", "difficulty": 1},
                                 cap_frames=800, work_dir=Path("unused"),
                                 data_root="unused")
    assert record["status"] == "failed"
    assert "engine busy" in record["error"]


# ---------------------------------------------------------------------------
# run: the engine budget and the resume ledger
# ---------------------------------------------------------------------------
def test_run_refuses_more_than_four_engines(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "plan.unit.json").write_text(json.dumps({"shards": []}), encoding="utf-8")
    args = collect.parse_args(["--phase", "run", "--workers", "6",
                               "--work-dir", str(work)])
    with pytest.raises(SystemExit):
        collect.phase_run(args)


# ---------------------------------------------------------------------------
# full-plan estimate
# ---------------------------------------------------------------------------
def test_full_plan_estimate_reports_disk_against_d05(tmp_path):
    args = plan_args(tmp_path)
    plan = collect.build_plan(
        plan_args(tmp_path, frames_target_per_stage=150_000,
                  frames_target_total=300_000, decisions_per_shard=4000),
        reachability(reach(0, 13_480, clear=True), reach(1, 40_000, death=40_000)),
        {})
    gate = {"frames": 4000, "decisions": 1000,
            "shard_reports": [
                {"obs_bytes": 52_112 * 500, "labels_bytes": 500 * 900,
                 "shard_json_bytes": 500 * 200,
                 "detail": {"teacher_fps": 30.0, "env_fps": 60.0, "wall_s": 100.0}},
                {"obs_bytes": 52_112 * 500, "labels_bytes": 500 * 900,
                 "shard_json_bytes": 500 * 200,
                 "detail": {"teacher_fps": 30.0, "env_fps": 60.0, "wall_s": 100.0}},
            ]}
    estimate = collect.estimate_full_plan(args, gate, reachability(
        reach(0, 13_480, clear=True), reach(1, 40_000, death=40_000)), plan)
    assert estimate["shards_needed"] == plan["stages_detail"][0]["shards_needed"] \
        + plan["stages_detail"][1]["shards_needed"]
    assert estimate["disk"]["gb"] > 0
    assert estimate["disk"]["d05_fraction"] < 0.8
    assert estimate["disk"]["triggers_d05_alert"] is False
    assert set(estimate["wall_s"]) == {"4", "8", "12", "16"}


# ---------------------------------------------------------------------------
# a recorded shard is a valid trajectory corpus (stages 1-7 plan)
# ---------------------------------------------------------------------------
def test_a_d1_shard_start_shard_can_be_the_next_trajectory_corpus(tmp_path):
    out, summary = write_obs_v3_shard(tmp_path, start_mode="shard", count=4,
                                      key="d1-s0d1-n4-seed1-shard")
    args = plan_args(tmp_path, corpus_root=str(tmp_path / "none"),
                     out_dir=str(out), start_mix="shard,trajectory",
                     frames_target_per_stage=400, frames_target_total=800,
                     decisions_per_shard=1)
    index = collect.index_corpora(args)
    assert index.get(0), "the finished D1 shard must be a corpus"
    assert index[0][0]["source"] == "d1"
    # the re-based shard.json arithmetic is what a corpus loader needs
    from touhou_rl.v400.gym.trajectory import load_d0_shard

    traj = load_d0_shard(Path(summary["dir"]))
    assert traj.frames == 4 * FRAMES
    assert len(traj.hashes) == 5  # frames 0,4,8,12,16
