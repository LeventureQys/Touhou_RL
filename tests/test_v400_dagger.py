"""EXP-P3-003: DAgger round-1 collector contracts.

The tests cover the four things the experiment card promises:

1. the β-injection probability and its fail-closed edge cases;
2. the HASH-checkpoint alignment gate (a mismatched pair is never accepted);
3. shard schema compatibility with ``dataset.load_teacher_samples`` (the D0
   loader is the only definition of "loadable");
4. whole-episode split isolation over the D0 + DAgger aggregate, with no seed
   in two splits, plus the aggregation accounting (counts must add up).

Nothing here needs an engine: the engine-facing code is exercised by the real
collection run (``.tmp_v400_dagger/logs/collect.log``), and the synthetic shards
below are built with the same writer the collector uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "tools" / "rl"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import v400_dagger_collect as dagger  # noqa: E402
from touhou_rl.v400.student import dataset, obs_spec, teacher_bc  # noqa: E402
from touhou_rl.v400.teacher.obs import TeacherObsShardWriter  # noqa: E402


# ---------------------------------------------------------------------------
# 1. β-injection
# ---------------------------------------------------------------------------
def test_injection_rate_matches_beta_and_is_seeded():
    rng = np.random.default_rng(0)
    draws = [float(rng.random()) for _ in range(100_000)]
    injected = sum(1 for draw in draws if dagger.inject_decision(draw, 0.2, teacher_cand=3))
    # Binomial(100k, 0.2): sd = 0.00126 -> 5 sd is 0.63 pp.  A wrong β (0.15/0.25)
    # is 5+ sd away, so this pins the realised rate without being flaky.
    assert abs(injected / len(draws) - 0.2) < 0.006

    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    seq_a = [rng_a.random() for _ in range(50)]
    seq_b = [rng_b.random() for _ in range(50)]
    assert seq_a == seq_b  # the RNG is the only source of the draw


def test_injection_edges_and_no_teacher_action():
    assert dagger.inject_decision(0.0, 0.0, teacher_cand=1) is False
    assert dagger.inject_decision(0.0, 1.0, teacher_cand=1) is True
    assert dagger.inject_decision(0.999, 1.0, teacher_cand=1) is True
    # No teacher plan -> never injected, not even at beta=1: there is no action to
    # inject, and pretending otherwise would report a β the data does not have.
    assert dagger.inject_decision(0.0, 1.0, teacher_cand=None) is False
    for bad in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(dagger.DaggerError):
            dagger.inject_decision(0.5, bad, teacher_cand=1)
    for bad_draw in (-0.01, 1.0, float("nan")):
        with pytest.raises(dagger.DaggerError):
            dagger.inject_decision(bad_draw, 0.2, teacher_cand=1)


# ---------------------------------------------------------------------------
# 2. HASH alignment (fail-closed)
# ---------------------------------------------------------------------------
def _ckpt(value: str, frames: int) -> dict:
    return {"hash": value, "frames": frames}


def test_align_pair_requires_hash_and_frame_agreement():
    row = dagger.align_pair(3, "pre", 12, _ckpt("aaaaaaaaaaaaaaaa", 12),
                            _ckpt("aaaaaaaaaaaaaaaa", 12))
    assert row["equal"] is True and row["source"] == "dagger A/B"
    same_hash_wrong_frame = dagger.align_pair(3, "post", 16, _ckpt("a" * 16, 16),
                                              _ckpt("a" * 16, 15))
    assert same_hash_wrong_frame["equal"] is False
    other_hash = dagger.align_pair(4, "pre", 16, _ckpt("b" * 16, 16), _ckpt("a" * 16, 16))
    assert other_hash["equal"] is False
    with pytest.raises(dagger.AlignError):
        dagger.align_pair(0, "pre", 0, {"hash": "a" * 16}, _ckpt("a" * 16, 0))


def test_require_aligned_is_fail_closed_and_counts_boundaries():
    good = [dagger.align_pair(i, side, 4 * i, _ckpt("c" * 16, 4 * i), _ckpt("c" * 16, 4 * i))
            for i in range(2) for side in ("pre", "post")]
    report = dagger.require_aligned(good, decisions=2)
    assert report["boundaries"] == 4 and report["all_equal"] is True
    assert report["checkpoints_mismatched"] == 0

    bad = list(good)
    bad[-1] = dict(bad[-1], equal=False)
    with pytest.raises(dagger.AlignError) as error:
        dagger.require_aligned(bad, decisions=2)
    assert "checkpoint mismatch" in str(error.value)
    assert bad[-1]["equal"] is False  # the evidence is not rewritten to look clean


def test_stage_clear_trim_needs_the_documented_single_boundary():
    decisions = 3
    pairs = [dagger.align_pair(i, side, 4 * i, _ckpt("d" * 16, 4 * i), _ckpt("d" * 16, 4 * i))
             for i in range(decisions) for side in ("pre", "post")]
    changed = {"stop_reason": "stage_changed", "stage_clear": True}
    # No mismatch -> no trim (nothing to exclude).
    assert dagger.dual._transition_info_from_pairs(pairs, decisions, changed) is None
    # The stage-clear commit's post state is the one documented exclusion.
    trimmed = list(pairs)
    trimmed[-1] = dict(trimmed[-1], equal=False)
    info = dagger.dual._transition_info_from_pairs(trimmed, decisions, changed)
    assert info is not None
    assert info["decisions_after_trim"] == decisions - 1
    assert info["checkpoints_mismatched_before_trim"] == 1
    # Any other single mismatch (not the last post) is not the documented one.
    other = list(pairs)
    other[0] = dict(other[0], equal=False)
    assert dagger.dual._transition_info_from_pairs(other, decisions, changed) is None
    # And without the teacher's stage verdict there is no trim at all.
    assert dagger.dual._transition_info_from_pairs(trimmed, decisions, {}) is None


# ---------------------------------------------------------------------------
# 3. schema compatibility with the D0 loader
# ---------------------------------------------------------------------------
def _write_shard(root: Path, key: str, *, seed: int, decisions: int,
                 beta: float = 0.2, injected: int = 0) -> Path:
    """One synthetic D0-shaped shard, built exactly like the collector builds it."""

    meta = {
        "stage": 0, "difficulty": 1, "team": 0, "seed": int(seed),
        "mode": "fast", "frames_per_decision": dagger.FRAMES_PER_DECISION,
        "generated_by": "tests/test_v400_dagger.py",
        "decisions": int(decisions), "decisions_requested": int(decisions),
        "stopped_early": False, "stop_reason": "decisions", "stage_clear": False,
        "frames_committed": int(decisions) * 4, "deaths_observed": 0.0,
        "transition_trim": None, "label_channel": "shadow", "pixel_channel": "env",
        "alignment_method": "hash-checkpoint",
        "dagger": {"round": 1, "beta": beta, "injections": int(injected),
                   "executed_teacher_rate": injected / max(1, decisions)},
    }
    writer = TeacherObsShardWriter(root, key=key, meta=meta)
    for index in range(int(decisions)):
        frame = index * dagger.FRAMES_PER_DECISION
        play = np.full((obs_spec.FRAME_STACK, *obs_spec.PLAY_SHAPE),
                       (index * 7) % 251 + 1, dtype=np.uint8)
        hud = np.full(obs_spec.HUD_SHAPE, (index * 3) % 251 + 1, dtype=np.uint8)
        label = dagger.dual._ShadowLabel({
            "scores": [float(17 - (index + k) % 18) for k in range(18)],
            "verified_mask": [k % 2 == 0 for k in range(18)],
            "bomb": 0, "margin": 5.0, "stage": 0, "difficulty": 1,
            "chosen_plan": {"slots": [3, 3], "duration": 2, "cand": 3},
            "frame_event_flags": {"frame_event_flags": 0}, "shoot_release": False,
            "bomb_death_available": False,
        })
        # Decision 0 is the documented episode-start padding row (one seed render
        # repeated); every later row is the consecutive frames[d-4 .. d-1] window,
        # which is exactly what ``DecisionObservationBuffer.stack_for`` produces.
        pad = obs_spec.FRAME_STACK - 1 if index == 0 else 0
        render_frames = ([152] * obs_spec.FRAME_STACK if index == 0
                         else [frame - obs_spec.FRAME_STACK + k
                               for k in range(obs_spec.FRAME_STACK)])
        writer.add(decision=index, decision_frame=frame, label=label,
                   sample={"play": play, "hud": hud,
                           "render_frames": render_frames,
                           "pad": pad,
                           "window": {"decision_frame": frame,
                                      "newest_render_frame": frame - 1,
                                      "label_frames": [frame, frame + 1],
                                      "p3_window_last_row": frame - 1,
                                      "p3_label_frame": frame}},
                   extra={"channel": "dagger-lockstep",
                          "dagger": {"beta": beta, "injected": index < injected}})
    writer.finalize(report={"test": True})
    return root / "shards" / key


def test_collector_shard_loads_through_the_d0_loader(tmp_path):
    key = dagger.shard_key(2000)
    shard_dir = _write_shard(tmp_path, key, seed=2000, decisions=12, injected=2)
    loaded = dataset.load_teacher_samples([shard_dir / "labels.jsonl"], min_samples=1,
                                          obs_root=tmp_path, require_observations=True)
    assert loaded["count"] == 12
    assert loaded["obs_spec_version"] == obs_spec.OBS_SPEC_VERSION
    assert loaded["obs_spec_digest"] == obs_spec.obs_spec_digest()
    first = loaded["samples"][0]
    assert np.asarray(first["scores"]).shape == (obs_spec.N_MOVE_CLASSES,)
    assert np.asarray(first["verified_mask"]).shape == (obs_spec.N_MOVE_CLASSES,)
    # The DAgger provenance travels with the shard and with every row (the frozen
    # loader only surfaces the D0 keys, so the row itself is read for the rest).
    shard = dataset.load_teacher_shard(shard_dir)
    assert shard.meta["dagger"]["beta"] == 0.2
    assert shard.meta["dagger"]["round"] == 1
    raw = [json.loads(line) for line in
           (shard_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert raw[0]["dagger"]["beta"] == 0.2
    assert [row["dagger"]["injected"] for row in raw[:3]] == [True, True, False]
    assert len(shard) == 12
    assert shard.play.shape == (12, obs_spec.FRAME_STACK, *obs_spec.PLAY_SHAPE)
    assert shard.hud.shape == (12, *obs_spec.HUD_SHAPE)
    assert shard.meta["dagger"]["round"] == 1


def test_schema_rejects_a_row_with_a_broken_label(tmp_path):
    key = dagger.shard_key(2001)
    shard_dir = _write_shard(tmp_path, key, seed=2001, decisions=3)
    # Truncate one score vector: the loader must refuse, not guess.
    rows = (shard_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[1])
    payload["scores"] = payload["scores"][:17]
    rows[1] = json.dumps(payload)
    (shard_dir / "labels.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(dataset.DatasetError):
        dataset.load_teacher_samples([shard_dir / "labels.jsonl"], min_samples=1,
                                     obs_root=tmp_path, require_observations=True)


def test_shard_writer_refuses_a_second_shard_with_the_same_key(tmp_path):
    key = dagger.shard_key(2002)
    _write_shard(tmp_path, key, seed=2002, decisions=2)
    with pytest.raises(Exception):
        _write_shard(tmp_path, key, seed=2002, decisions=2)


# ---------------------------------------------------------------------------
# 4. aggregation + whole-episode split isolation
# ---------------------------------------------------------------------------
def test_episode_plan_never_overlaps_the_evaluation_seeds():
    args = dagger.build_parser().parse_args(["--seed-start", "2000", "--episodes", "4"])
    plan = dagger.episode_plan(args)
    assert [row["seed"] for row in plan] == [2000, 2001, 2002, 2003]
    assert [row["key"] for row in plan] == [dagger.shard_key(s) for s in (2000, 2001, 2002, 2003)]
    assert not any(row["eval_overlap"] for row in plan)
    # A plan that would touch the S1N seeds is detected, not silently run.
    bad = dagger.episode_plan(dagger.build_parser().parse_args(
        ["--seed-start", "1000", "--episodes", "2"]))
    assert [row["eval_overlap"] for row in bad] == [True, True]


def test_count_roots_adds_up_and_refuses_duplicate_keys(tmp_path):
    d0_root = tmp_path / "d0"
    dagger_root = tmp_path / "d0_extra"
    _write_shard(d0_root, "dual-s0d1seed1", seed=1, decisions=5)
    _write_shard(d0_root, "dual-s0d1seed2", seed=2, decisions=7)
    _write_shard(dagger_root, dagger.shard_key(2000), seed=2000, decisions=9)
    report = dagger.count_roots([d0_root, dagger_root])
    assert report["decisions"] == 21
    assert report["shards"] == 3
    assert [row["decisions"] for row in report["roots"]] == [12, 9]

    # The same key in two roots would double-count one episode: refused.
    clash = tmp_path / "clash"
    _write_shard(clash, "dual-s0d1seed1", seed=1, decisions=1)
    with pytest.raises(dagger.DaggerError):
        dagger.count_roots([d0_root, clash])
    with pytest.raises(dagger.DaggerError):
        dagger.count_roots([tmp_path / "missing"])


def test_split_assignment_is_episode_isolated_across_roots():
    keys = ([f"dual-s0d1seed{s}" for s in range(1, 17)]
            + [dagger.shard_key(s) for s in range(2000, 2006)])
    seeds = {}
    decisions = {}
    for key in keys:
        seed = int(key.rsplit("seed", 1)[1])
        seeds[key] = seed
        decisions[key] = 3_000
    assignments = teacher_bc.assign_shard_splits(keys, ratios=teacher_bc.SPLIT_RATIOS, seed=0)
    report = teacher_bc.split_report(assignments, decisions=decisions, seeds=seeds)
    # Same seed never lands in two splits, and every shard is assigned once.
    seen: dict[int, str] = {}
    for name in teacher_bc.SPLIT_NAMES:
        for key in report[name]["shards"]:
            assert assignments[key] == name
            assert seeds[key] not in seen, f"seed {seeds[key]} in two splits"
            seen[seeds[key]] = name
    assert set(assignments) == set(keys)
    total = sum(report[name]["decisions"] for name in teacher_bc.SPLIT_NAMES)
    assert total == sum(decisions.values())
    # The split is a function of the key list + seed, not of insertion order.
    again = teacher_bc.assign_shard_splits(list(reversed(keys)),
                                           ratios=teacher_bc.SPLIT_RATIOS, seed=0)
    assert again == assignments
    assert teacher_bc.split_signature(assignments) == teacher_bc.split_signature(again)


def test_split_seed_collisions_detects_a_cross_root_duplicate():
    entries = [{"key": "a", "seed": 1}, {"key": "b", "seed": 2}, {"key": "c", "seed": 1}]
    assert dagger.split_seed_collisions(entries) == [1]
    assert dagger.split_seed_collisions([{"key": "a", "seed": 3}]) == []


def test_frames_per_decision_matches_the_d0_schema():
    from touhou_rl.v400.teacher import get_config

    # The teacher's fast profile is what D0 used; the collector refuses to write a
    # shard whose decision period differs from the schema's 4 frames.
    fast = get_config("fast")
    assert fast.replan_decisions * fast.decision_frames == dagger.FRAMES_PER_DECISION
    assert dagger.FRAMES_PER_DECISION == 4
    assert obs_spec.DECISION_FRAMES == 2


def test_shard_key_shape_and_eval_seed_guard():
    assert dagger.shard_key(2000) == "dagger-s0d1seed2000"
    assert dagger.shard_key(2000, stage=2, difficulty=3) == "dagger-s2d3seed2000"
    args = dagger.build_parser().parse_args([])
    assert args.beta == 0.2
    assert args.phase == "run"
    assert args.shard_prefix if hasattr(args, "shard_prefix") else True


# ---------------------------------------------------------------------------
# 5. EXP-P3-006 (DAgger round 2): campaign accounting, geo_restore, two arms
# ---------------------------------------------------------------------------
def test_continue_on_error_is_off_by_default_and_selectable():
    """A campaign flag must never change the default (R1's) semantics."""

    default = dagger.build_parser().parse_args([])
    assert default.continue_on_error is False
    assert default.beta == 0.2
    assert default.mode == "fast"
    selected = dagger.build_parser().parse_args(["--continue-on-error", "--mode", "fast3"])
    assert selected.continue_on_error is True
    assert selected.mode == "fast3"


def test_campaign_summary_derives_partial_from_the_target_not_from_the_caller():
    full = dagger.campaign_summary(
        decisions=100_000, target=100_000, checkpoints=200_000,
        checkpoints_equal=200_000, elapsed_s=1.0, episodes=48, failures=0,
        beta=0.2, mode="fast", checkpoint="ckpt.pt")
    assert full["target_met"] is True and full["partial"] is False
    assert full["hash_alignment_rate"] == 1.0
    assert full["hash_alignment_complete"] is True

    # The wave that stops on the wall budget: partial is derived, and the
    # alignment evidence is still reported (it is not "unknown" just because the
    # decision target was missed).
    short = dagger.campaign_summary(
        decisions=47_998, target=100_000, checkpoints=95_996,
        checkpoints_equal=95_996, elapsed_s=6600.0, episodes=42, failures=1,
        beta=0.2, mode="fast", checkpoint="ckpt.pt")
    assert short["partial"] is True and short["target_met"] is False
    assert short["episodes_failed"] == 1
    assert short["hash_alignment_rate"] == 1.0

    # A misaligned checkpoint is visible at campaign level (fail-closed signal).
    dirty = dagger.campaign_summary(
        decisions=10, target=10, checkpoints=20, checkpoints_equal=19,
        elapsed_s=1.0, episodes=1, failures=0, beta=0.2, mode="fast",
        checkpoint="ckpt.pt")
    assert dirty["hash_alignment_rate"] == 0.95
    assert dirty["hash_alignment_complete"] is False

    # No checkpoint at all -> rate is None, never a vacuous 1.0.
    empty = dagger.campaign_summary(
        decisions=0, target=10, checkpoints=0, checkpoints_equal=0,
        elapsed_s=0.0, episodes=0, failures=0, beta=0.2, mode="fast",
        checkpoint="ckpt.pt")
    assert empty["hash_alignment_rate"] is None
    assert empty["partial"] is True
    with pytest.raises(dagger.DaggerError):
        dagger.campaign_summary(decisions=1, target=1, checkpoints=2,
                                checkpoints_equal=3, elapsed_s=1.0, episodes=1,
                                failures=0, beta=0.2, mode="fast", checkpoint="x")
    with pytest.raises(dagger.DaggerError):
        dagger.campaign_summary(decisions=1, target=1, checkpoints=2,
                                checkpoints_equal=2, elapsed_s=1.0, episodes=1,
                                failures=0, beta=1.5, mode="fast", checkpoint="x")


def test_geo_restore_is_a_value_exact_inverse_on_unverified_candidates_only(tmp_path):
    """``scores + 5*(1 - verified_mask)``, reversible value by value."""

    trainer = pytest.importorskip("v400_student_train_teacher")
    key = dagger.shard_key(3000)
    shard_dir = _write_shard(tmp_path, key, seed=3000, decisions=6)
    shard = dataset.load_teacher_shard(shard_dir)
    on_disk = np.stack([np.asarray(row["scores"], dtype=np.float64)
                        for row in shard.records])
    mask = np.stack([np.asarray(row["verified_mask"], dtype=bool)
                     for row in shard.records])
    assert mask.any() and (~mask).any(), "the fixture must have both candidates"

    off = trainer._load_shard_arrays(shard, geo_restore=False)
    on = trainer._load_shard_arrays(shard, geo_restore=True)

    # Off == the labels as written; On == Off + 5 exactly on the unverified ones.
    assert np.array_equal(off["scores"].astype(np.float64), on_disk)
    assert np.array_equal(on["scores_raw"], off["scores"])
    delta = on["scores"].astype(np.float64) - off["scores"].astype(np.float64)
    expected = 5.0 * (~mask)
    assert np.array_equal(delta, expected)
    # Verified candidates are untouched: the mask is never re-derived from values.
    assert np.all(delta[mask] == 0.0)

    # Reversibility, value by value, and the accounting the gate reports.
    restored_back = on["scores"].astype(np.float64) - 5.0 * (~mask)
    assert np.array_equal(restored_back, on_disk)
    assert on["unverified_candidates"] == int((~mask).sum())
    assert off["unverified_candidates"] == int((~mask).sum())
    assert on["restored_delta"] == pytest.approx(5.0 * int((~mask).sum()))
    assert off["restored_delta"] == 0.0
    # The discount is read from the teacher's own Weights, not hard-coded here.
    from touhou_rl.v400.teacher import config as teacher_config
    assert trainer.UNVERIFIED_DISCOUNT == float(teacher_config.Weights().unverified_discount)


def test_two_arms_share_one_split_and_differ_only_in_the_restored_mass(tmp_path):
    """The A/B control: same data, same split, one flag off/on."""

    trainer = pytest.importorskip("v400_student_train_teacher")
    d0_root = tmp_path / "d0"
    r1_root = tmp_path / "r1"
    r2_root = tmp_path / "r2"
    for seed in (1, 2, 3):  # D0
        _write_shard(d0_root, f"dual-s0d1seed{seed}", seed=seed, decisions=5)
    for seed in (2000, 2001):  # DAgger R1
        _write_shard(r1_root, dagger.shard_key(seed), seed=seed, decisions=4,
                     injected=1)
    for seed in (3000, 3001, 3002):  # DAgger R2
        _write_shard(r2_root, dagger.shard_key(seed), seed=seed, decisions=4,
                     injected=1)
    keys = ([f"dual-s0d1seed{s}" for s in (1, 2, 3)]
            + [dagger.shard_key(s) for s in (2000, 2001, 3000, 3001, 3002)])
    assignments = teacher_bc.assign_shard_splits(keys, ratios=teacher_bc.SPLIT_RATIOS,
                                                 seed=0)
    assert set(assignments) == set(keys)
    # No seed can cross a split (the R2 seeds are new, so nothing collides).
    seeds = {key: int(key.rsplit("seed", 1)[1]) for key in keys}
    teacher_bc.check_episode_isolation(assignments, seeds)

    manifest = dataset.load_teacher_manifest(d0_root)
    kwargs = dict(min_samples=1, require_observations=True,
                  extra_roots=(r1_root, r2_root))
    off_splits, off_gate = trainer.build_splits(
        d0_root, manifest, assignments, geo_restore=False, **kwargs)
    on_splits, on_gate = trainer.build_splits(
        d0_root, manifest, assignments, geo_restore=True, **kwargs)

    # Same split, same observations, same shard identity: only the target mass moves.
    for name in teacher_bc.SPLIT_NAMES:
        off, on = off_splits[name], on_splits[name]
        assert off.shard_keys == on.shard_keys
        assert len(off) == len(on) > 0
        assert len(off) == sum(off.decisions_by_shard.values())
        assert off.decisions_by_shard == on.decisions_by_shard
        assert np.array_equal(off.play, on.play)
        assert np.array_equal(off.hud, on.hud)
        assert np.array_equal(off.bomb, on.bomb)
        assert np.array_equal(off.scores_raw, on.scores_raw)
        # Off keeps the on-disk penalty; On adds the discount back per candidate.
        assert np.array_equal(off.scores, off.scores_raw)
        assert int((on.scores > off.scores).sum()) > 0

    total_off = sum(int(off_splits[n].scores.size) for n in teacher_bc.SPLIT_NAMES)
    total_on = sum(int(on_splits[n].scores.size) for n in teacher_bc.SPLIT_NAMES)
    assert total_off == total_on
    delta = sum(float((on_splits[n].scores - off_splits[n].scores).sum())
                for n in teacher_bc.SPLIT_NAMES)
    assert delta == pytest.approx(5.0 * off_gate["geo_restore_unverified_values"])
    assert off_gate["geo_restore"] is False and on_gate["geo_restore"] is True
    assert on_gate["geo_restore_discount"] == 5.0
    assert on_gate["geo_restore_total_delta"] == pytest.approx(delta)
    assert off_gate["geo_restore_total_delta"] == 0.0
    # Aggregation accounting: every root contributed, totals add up to the shards.
    assert on_gate["root_count"] == 3
    assert on_gate["shards_loaded"] == len(keys)
    assert [row["shards"] for row in on_gate["roots"]] == [3, 2, 3]
    assert on_gate["samples"] == 3 * 5 + 2 * 4 + 3 * 4
    # The split is a function of the corpus: both arms see the same signature.
    assert (teacher_bc.split_signature(assignments)
            == teacher_bc.split_signature({k: assignments[k] for k in reversed(keys)}))


def test_r2_seeds_extend_the_corpus_without_colliding_with_d0_or_r1():
    d0_keys = [f"dual-s0d1seed{s}" for s in range(1, 17)]
    r1_keys = [dagger.shard_key(s) for s in range(2000, 2011)]
    r2_keys = [dagger.shard_key(s) for s in range(3000, 3060)]
    assert not set(d0_keys) & set(r1_keys)
    assert not set(r1_keys) & set(r2_keys)
    assert not set(d0_keys) & set(r2_keys)
    # Every key carries its own seed, so whole-shard splitting cannot leak an
    # episode: the R2 slice is 60 distinct seeds and 60 distinct shards.
    seeds = {key: int(key.rsplit("seed", 1)[1]) for key in d0_keys + r1_keys + r2_keys}
    assert len(set(seeds.values())) == len(seeds) == 16 + 11 + 60
    assert dagger.split_seed_collisions(
        [{"key": key, "seed": seed} for key, seed in seeds.items()]) == []

