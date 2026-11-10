"""EXP-P3-004 — the label-signal 2x2: what the four arms' labels actually are.

The experiment's claim is that ``verify_k=1`` turns the teacher's argmax into
"which action was sampled for verification" (EXP-P3-003 §8.3).  Four things in
the pipeline make that claim testable, and each one is pinned here:

1. **``fast3`` is registered and is ``fast`` plus ``verify_k=3``, nothing else**
   — otherwise the 2x2's verify_k factor would be confounded with the search
   budget (``v2a`` changes horizon *and* beam *and* weights *and* the bomb
   window).  ``fast`` / ``v2a`` must stay byte-identical.
2. **``geo_restore`` is value-exact**: the label stores
   ``geo[c] - 5.0 * (1 - verified[c])`` (``teacher/label.py:161-163``) and the
   restore adds the same 5.0 back to exactly the unverified candidates.  The
   numbered example in the task card is checked against the real D0 row
   (``scores[7] = 117.125`` is *verified*, so restoring it is a no-op; the 17
   unverified candidates each move by exactly +5.0).
3. **the restore flattens the soft target when only one candidate is verified**
   — this is the mechanism that makes the arm ``geo_restore=on, verify_k=1``
   collapse to chance, so it is asserted numerically (target entropy up, max
   probability near 1/18) rather than described.
4. **macro recall and the closed-loop observation window** — macro recall is the
   metric the arms are compared on, and the evaluation window is the fix that
   makes the closed loop see the same 4 *consecutive* frames the student was
   trained on (instead of 4 frames strided 2 apart, spanning 8).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from touhou_rl.v400.student import obs_spec
from touhou_rl.v400.teacher import config as teacher_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "tools" / "rl" / "v400_student_train_teacher.py"
EVAL_TOOL = REPO_ROOT / "tools" / "rl" / "v400_student_eval.py"
D0_ROOT = REPO_ROOT / ".tmp_v400_d0" / "d0"


def _load(name: str, path: Path):
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # ``dataclasses`` resolves ``str`` annotations through
    # ``sys.modules[cls.__module__]``, so a module loaded by path must register
    # itself before it executes (the trainer defines ``SplitArrays``).
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture(scope="module")
def trainer():
    return _load("v400_train_teacher_signal", TRAINER)


@pytest.fixture(scope="module")
def evaluator():
    return _load("v400_student_eval_signal", EVAL_TOOL)


# ---------------------------------------------------------------------------
# 1. the fast3 profile
# ---------------------------------------------------------------------------
def test_fast3_is_fast_with_three_verified_plans_and_nothing_else():
    fast = teacher_config.get_config("fast")
    fast3 = teacher_config.get_config("fast3")
    assert fast3.mode == "fast3"
    assert fast.verify_k == 1 and fast3.verify_k == 3
    # Every other search / scoring knob is identical: the 2x2 must change one
    # factor, so a differing horizon or weight would invalidate the contrast.
    for name in ("horizon", "beam", "max_iterations", "replan_decisions",
                 "decision_frames", "durations", "border_x", "border_y",
                 "record_items", "bomb_active", "bomb_death",
                 "bomb_rule1_max_death_frame", "bomb_rule1_min_bombs_left",
                 "bomb_rule1_max_margin", "clip_y", "shoot_always",
                 "shoot_pulse_period"):
        assert getattr(fast3, name) == getattr(fast, name), name
    assert fast3.weights == fast.weights
    # 11-P1 §2.1: one decision is still four committed frames.
    assert fast3.replan_decisions * fast3.decision_frames == 4
    # The registry exposes it under exactly that name, and the candidate list of
    # *scoring* profiles is untouched (fast3 is a labelling profile).
    assert teacher_config.CONFIGS["fast3"] is fast3
    assert teacher_config.get_config("fast") is teacher_config.FAST
    assert "fast3" not in {profile.mode for profile in teacher_config.PROFILES}


def test_fast_and_v2a_keep_their_meaning():
    """Adding the third label profile may not redefine the two that were there."""

    fast = teacher_config.FAST
    v2a = teacher_config.V2A
    assert (fast.mode, fast.horizon, fast.beam, fast.verify_k, fast.max_iterations) == (
        "fast", 32, 24, 1, 1)
    assert (v2a.mode, v2a.horizon, v2a.beam, v2a.verify_k, v2a.max_iterations) == (
        "v2a", 48, 64, 3, 3)
    assert (v2a.weights.bottom_penalty, v2a.weights.bottom_extra_penalty) == (1.5, 2.0)
    assert v2a.bomb_rule1_max_death_frame == 12
    # The three label-profile knobs are one value apart, and the discount the
    # restore undoes is the same constant in every profile.
    for profile in (teacher_config.FAST, teacher_config.FAST3, teacher_config.V2A,
                    teacher_config.FULL, teacher_config.V3B):
        assert profile.weights.unverified_discount == 5.0


def test_the_teacher_cli_and_the_d0_pipeline_accept_the_new_profile():
    cli = _load("v400_teacher_cli_signal", REPO_ROOT / "tools" / "rl" / "v400_teacher.py")
    assert cli.parse_args(["--mode", "fast3"]).mode == "fast3"
    assert cli.parse_args([]).mode == "full"  # the default is still full
    dual = _load("v400_d0_dual_cli_signal", REPO_ROOT / "tools" / "rl" / "v400_d0_dual.py")
    args = dual.parse_args(["--mode", "fast3", "--decisions", "7"])
    assert (args.mode, args.decisions) == ("fast3", 7)
    # The choice list is the teacher registry itself, so a profile can never be
    # spelled two ways (the D0 pipeline used to accept only full/fast).
    with pytest.raises(SystemExit):
        dual.parse_args(["--mode", "fast33"])


# ---------------------------------------------------------------------------
# 2. geo restore: value-exact, mask-driven
# ---------------------------------------------------------------------------
def _first_decision_row(shard: str = "dual-s0d1seed1") -> dict:
    path = D0_ROOT / "shards" / shard / "labels.jsonl"
    if not path.is_file():
        pytest.skip(f"{path} is not present on this machine")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("kind") == "decision":
                return payload
    raise AssertionError(f"{path} holds no decision row")


def test_geo_restore_on_the_real_d0_first_row(trainer):
    """The task card's numbered example, checked against the frozen D0 label.

    Task text: "D0 第一行 index 7 应从 122.03 还原为 117.03".  The recorded row
    says index 7 is ``117.125`` **and verified**, so the restore is an identity
    there (there is no discount to undo); the values that do move are the other
    17, by exactly +5.0.  The direction is therefore +5, not -5: ``scores[c]``
    already carries ``geo[c] - 5`` for unverified candidates.
    """

    row = _first_decision_row()
    scores = np.asarray(row["scores"], dtype=np.float64)
    mask = np.asarray(row["verified_mask"], dtype=bool)
    assert scores.shape == (obs_spec.N_MOVE_CLASSES,)
    assert int(mask.sum()) == 1  # fast/verify_k=1: exactly one action is verified
    assert bool(mask[7]) is True
    assert scores[7] == pytest.approx(117.125, abs=1e-9)
    restored = trainer.geo_restore_scores(scores, mask)
    # verified -> untouched (this is the "117.03 stays 117.03" half of the example)
    assert restored[7] == scores[7]
    # the chosen plan's own geo_score is the verified number: the two agree
    plan = row["chosen_plan"]
    assert int(plan["first_action"]) == int(np.argmax(scores)) == 7
    assert plan["geo_score"] == pytest.approx(float(scores[7]))
    # unverified -> exactly +5.0, e.g. index 0: 111.791667 -> 116.791667
    assert scores[0] == pytest.approx(111.791667, abs=1e-6)
    assert restored[0] == pytest.approx(116.791667, abs=1e-6)
    delta = restored - scores
    assert np.allclose(delta[~mask], teacher_config.Weights().unverified_discount,
                       atol=0.0, rtol=0.0)
    assert np.all(delta[mask] == 0.0)


def test_geo_restore_only_touches_unverified_values(trainer):
    rng = np.random.default_rng(11)
    scores = np.round(rng.normal(112.0, 0.4, size=(6, obs_spec.N_MOVE_CLASSES)), 6)
    mask = np.zeros_like(scores, dtype=bool)
    for row, cand in enumerate((0, 3, 7, 7, 12, 17)):
        scores[row, cand] = 117.0 + row
        mask[row, cand] = True
    restored = trainer.geo_restore_scores(scores, mask)
    discount = trainer.UNVERIFIED_DISCOUNT
    assert discount == 5.0
    # exact, not approximately: adding a constant is exact in float64
    assert np.array_equal(restored[mask], scores[mask])
    assert np.array_equal(restored[~mask], scores[~mask] + discount)
    # and the round trip lands back on the on-disk label
    assert np.array_equal(restored - discount * (~mask), scores)
    # bitwise: every element is either unchanged or +5.0, nothing else
    changed = ~np.isclose(restored, scores)
    assert np.array_equal(changed, ~mask)


def test_geo_restore_rejects_malformed_input(trainer):
    scores = np.zeros((2, obs_spec.N_MOVE_CLASSES))
    with pytest.raises(ValueError):
        trainer.geo_restore_scores(scores, np.zeros((2, obs_spec.N_MOVE_CLASSES - 1), bool))
    with pytest.raises(ValueError):
        trainer.geo_restore_scores(np.zeros((2, obs_spec.N_MOVE_CLASSES - 1)),
                                   np.zeros((2, obs_spec.N_MOVE_CLASSES - 1), bool))
    with pytest.raises(ValueError):
        trainer.geo_restore_scores(scores, np.zeros_like(scores, dtype=bool),
                                   discount=float("nan"))


def test_geo_restore_flattens_a_one_of_eighteen_target(trainer):
    """Why the arm exists: with verify_k=1 the restore removes almost all signal."""

    row = _first_decision_row()
    scores = np.asarray(row["scores"], dtype=np.float64)
    mask = np.asarray(row["verified_mask"], dtype=bool)
    restored = trainer.geo_restore_scores(scores, mask)
    tau = 2.0

    def softmax(values):
        shifted = np.exp((values - values.max()) / tau)
        return shifted / shifted.sum()

    before = softmax(scores)
    after = softmax(restored)
    assert before.max() > 0.40           # the 5.0 spike dominates the target
    assert after.max() < 0.08            # ~1/18 = 0.0556: the target is flat
    assert after.max() < before.max() / 5.0
    entropy = lambda p: float(-(p * np.log(p)).sum())  # noqa: E731
    assert entropy(after) > entropy(before) + 0.5
    # ln(18) = 2.890: the restored target is essentially the uniform distribution
    assert entropy(after) > np.log(obs_spec.N_MOVE_CLASSES) - 0.01
    # the geometric spread that survives is two orders below the discount
    geometric_gap = float(np.sort(restored)[-1] - np.sort(restored)[-2])
    assert geometric_gap < teacher_config.Weights().unverified_discount / 20.0
    # ... and the argmax is unchanged by the restore (it is the verified action),
    # which is exactly why top-1 alone can look healthy while the target is flat.
    assert int(np.argmax(after)) == int(np.argmax(before)) == 7


# ---------------------------------------------------------------------------
# 3. macro recall
# ---------------------------------------------------------------------------
def test_macro_recall_is_hand_checkable(trainer):
    teacher = np.array([0, 0, 0, 1, 1, 2])
    predicted = np.array([0, 0, 1, 1, 2, 2])
    report = trainer.macro_recall(predicted, teacher)
    # class 0: 2/3, class 1: 1/2, class 2: 1/1 -> macro = 0.7222...
    assert report["macro_recall"] == pytest.approx((2 / 3 + 1 / 2 + 1.0) / 3, abs=1e-12)
    assert report["classes_present"] == 3
    assert report["per_class_recall"][:3] == [
        pytest.approx(2 / 3), pytest.approx(1 / 2), pytest.approx(1.0)]
    assert report["per_class_support"][:4] == [3, 2, 1, 0]
    assert report["majority_class"] == 0
    assert report["majority_class_baseline"] == pytest.approx(3 / 6)
    assert report["predicted_classes"] == [0, 1, 2]


def test_macro_recall_averages_only_over_present_classes(trainer):
    # The absent class 2 must not drag the mean down with a 0/0 recall.
    # teacher [0,0,1] vs pred [0,1,1]: class 0 -> 1/2, class 1 -> 1/1.
    report = trainer.macro_recall(np.array([0, 1, 1]), np.array([0, 0, 1]))
    assert report["classes_present"] == 2
    assert report["macro_recall"] == pytest.approx(0.75)
    # a degenerate "always predict majority" model: 1 present class is 1.0 for
    # that class but the minority class is 0 -> macro is not fooled by accuracy
    constant = trainer.macro_recall(np.zeros(8, dtype=int),
                                    np.array([0, 0, 0, 0, 0, 0, 5, 5]))
    assert constant["macro_recall"] == pytest.approx(0.5)
    assert constant["majority_class_baseline"] == pytest.approx(0.75)
    assert constant["predicted_classes"] == [0]


def test_macro_recall_rejects_bad_arguments(trainer):
    with pytest.raises(ValueError):
        trainer.macro_recall(np.array([0, 1]), np.array([0]))
    with pytest.raises(ValueError):
        trainer.macro_recall(np.array([], dtype=int), np.array([], dtype=int))
    with pytest.raises(ValueError):
        trainer.macro_recall(np.array([0, obs_spec.N_MOVE_CLASSES]), np.array([0, 1]))


def test_evaluate_reports_top1_macro_recall_kl_and_bomb(trainer):
    torch = pytest.importorskip("torch")
    from touhou_rl.v400.student.model import StudentNet

    samples = 12
    play = np.random.default_rng(5).integers(0, 256, size=(samples, 4, 64, 64), dtype=np.uint8)
    hud = np.random.default_rng(6).integers(0, 256, size=(samples, 56, 160), dtype=np.uint8)
    scores = np.full((samples, obs_spec.N_MOVE_CLASSES), 112.0, dtype=np.float32)
    scores[:, 7] += 5.0                       # the D0 label shape
    bomb = np.ones(samples, dtype=np.int8)
    arrays = trainer.SplitArrays(
        name="synthetic", play=play, hud=hud, scores=scores, bomb=bomb,
        verified_count=np.ones(samples, dtype=np.int16),
        shard_index=np.zeros(samples, dtype=np.int16),
        shard_keys=["synthetic"], decisions_by_shard={"synthetic": samples},
        scores_raw=scores.copy())
    torch.manual_seed(0)
    net = StudentNet(width=48, hidden=256)
    metrics = trainer.evaluate(arrays, net, np.arange(samples), tau=2.0, device="cpu",
                               batch_size=8, torch=torch)
    for key in ("top1", "macro_recall", "kl", "bomb_accuracy", "predicted_classes",
                "macro_recall_per_class", "majority_class_baseline"):
        assert key in metrics, key
    assert metrics["samples"] == samples
    assert 0.0 <= metrics["macro_recall"] <= 1.0
    assert metrics["macro_recall_classes_present"] == 1
    assert metrics["macro_recall"] == (metrics["top1"] if 7 in metrics["predicted_classes"]
                                       else 0.0)
    assert metrics["kl"] > 0.0 and np.isfinite(metrics["kl"])


# ---------------------------------------------------------------------------
# 4. the closed-loop observation window
# ---------------------------------------------------------------------------
def test_decision_steps_keep_the_frame_ruler_and_double_the_renders(evaluator):
    assert evaluator.DECISION_FRAMES == 2
    assert obs_spec.FRAME_STACK == 4
    consecutive = evaluator.decision_steps("consecutive", 2)
    strided = evaluator.decision_steps("strided", 2)
    assert consecutive == (2, 1)     # two single-frame STEPs, two renders
    assert strided == (1, 2)         # one two-frame STEP, one render
    # the frame budget per decision is identical: only the renders differ
    for mode in ("consecutive", "strided"):
        units, want = evaluator.decision_steps(mode, 2)
        assert units * want == 2
    # the last decision against the frame cap may ask for a single frame
    assert evaluator.decision_steps("consecutive", 1) == (1, 1)
    assert evaluator.decision_steps("strided", 1) == (1, 1)
    with pytest.raises(evaluator.EvalError):
        evaluator.decision_steps("dense", 2)
    with pytest.raises(evaluator.EvalError):
        evaluator.decision_steps("consecutive", 0)


def _window_stack(mode: str, *, total_frames: int = 8) -> list[int]:
    """Which frame indices end up in the 4-deep stack, for each window mode.

    Mirrors the loop: a decision spends ``DECISION_FRAMES`` game frames and pushes
    one render per STEP unit.  The frames are labelled 1..N so the stack can be
    read back as "which frames the network saw".
    """

    buffer = obs_spec.ObservationBuffer()
    frame = 0
    while frame < total_frames:
        budget = min(2, total_frames - frame)
        units, want = (budget, 1) if mode == "consecutive" else (1, budget)
        for _ in range(units):
            frame += want
            blank_play = np.full(obs_spec.PLAY_SHAPE, frame % 256, dtype=np.uint8)
            blank_hud = np.zeros(obs_spec.HUD_SHAPE, dtype=np.uint8)
            buffer.push(blank_play, blank_hud)
    stack = buffer.stack()
    return [int(plane[0, 0]) for plane in stack]


def test_consecutive_window_is_the_training_window(evaluator):
    consecutive = _window_stack("consecutive")
    strided = _window_stack("strided")
    # training: DecisionObservationBuffer.stack_for(4k) = frames[4k-4..4k-1]
    assert consecutive == [5, 6, 7, 8]
    assert consecutive == list(range(consecutive[-1] - 3, consecutive[-1] + 1))
    # legacy: one render per decision -> consecutive pushes are 2 frames apart,
    # so 4 pushes span (FRAME_STACK - 1) * DECISION_FRAMES + 1 = 7 frames
    assert strided == [2, 4, 6, 8]
    assert strided[-1] - strided[0] == (obs_spec.FRAME_STACK - 1) * evaluator.DECISION_FRAMES
    assert strided != consecutive


def test_the_eval_cli_defaults_to_the_aligned_window(evaluator):
    import inspect

    assert evaluator.OBS_WINDOWS == ("consecutive", "strided")
    assert inspect.signature(evaluator.run_episode).parameters["obs_window"].default == \
        "consecutive"
    parser = evaluator.build_parser()
    base = ["--engine", "e", "--data-dir", "d", "--checkpoint", "c"]
    assert parser.parse_args(base).obs_window == "consecutive"
    assert parser.parse_args([*base, "--obs-window", "strided"]).obs_window == "strided"
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--obs-window", "dense"])


def _aligned_record() -> dict:
    return {
        "schema": "v400-student-eval-episode-1",
        "checkpoint": ".tmp_v400_p3sig/arms/armA_k1_geooff.pt",
        "checkpoint_sha256": "a" * 64,
        "obs_spec_version": obs_spec.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec.obs_spec_digest(),
        "seed": 1000,
        "suite": "S1N",
        "stage": 0,
        "difficulty": "normal",
        "max_frames": 19887,
        "frames": 1234,
        "deaths": 1.0,
        "score": 4321,
        "end": "death_stop",
        "privilege": "deploy_pixel",
        "wall_s": 1.5,
        "engine_sha256": "b" * 64,
        "policy_input": "play64 4-frame stack + hud strip; no privileged state",
        "not_a_gate": True,
        "obs_window": "consecutive",
        "renders": 618,
        "window_alignment": {"mode": "consecutive", "frames_per_decision": 2,
                            "renders_per_decision": 2, "stack_span_frames": 4,
                            "matches_training_obs": True},
    }


def test_eval_record_pins_the_window_it_was_measured_with(evaluator):
    record = _aligned_record()
    evaluator.validate_episode_record(record)
    wrong = json.loads(json.dumps(record))
    wrong["window_alignment"]["renders_per_decision"] = 1
    with pytest.raises(evaluator.EvalError):
        evaluator.validate_episode_record(wrong)
    unknown = json.loads(json.dumps(record))
    unknown["obs_window"] = "dense"
    with pytest.raises(evaluator.EvalError):
        evaluator.validate_episode_record(unknown)
    # the legacy row is still valid, so old artifacts keep loading
    legacy = json.loads(json.dumps(record))
    legacy["obs_window"] = "strided"
    legacy["window_alignment"].update({"mode": "strided", "renders_per_decision": 1,
                                       "stack_span_frames": 8,
                                       "matches_training_obs": False})
    evaluator.validate_episode_record(legacy)


def test_summary_carries_the_window_forward(evaluator):
    summary = evaluator.summarise([_aligned_record()])
    assert summary["obs_window"] == "consecutive"
    assert summary["window_alignment"]["stack_span_frames"] == 4
    assert "strided" in summary["window_note"]
