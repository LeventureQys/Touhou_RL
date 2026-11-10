"""EXP-P3-008 obs-v2 contract tests: digest, history reconstruction, GRU, eval schema.

Run (Windows)::

    $env:PYTHONPATH='D:\\WorkShop\\Touhou_RL'
    python -m pytest -p no:cacheprovider --basetemp .tmp_v400_obsv2/pytest -q \
      tests/test_v400_obs_v2.py

What is locked here, and why each lock exists:

* **obs-v1 must not move** — the file bytes *and* the canonical digest are literals.
  obs-v2 is additive, so a v1 change is a contract break even if it looks harmless.
* **obs-v2 digest is frozen** — a literal sha256 over the canonical spec; a change to
  any obs-v2 field (including the reserved-input statement) must bump the spec, not
  slip through.
* **action history is causal and boundary-cleared** — the window holds ``a[t-8..t-1]``,
  is cleared at a shard start and at a stage change, and never contains ``a[t]``.
* **GRU packing / BPTT** — the batch builder maps step ``s`` to decision ``end-(7-s)``
  (no reordering bug), ``truncate_bptt`` really cuts the gradient of the prefix, and
  the deployment cache reproduces the batch forward's last step.
* **eval schema** — an obs-v2 episode validates against the obs-v2 spec and is refused
  against obs-v1, so the two rulers can never be mixed in one summary.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from touhou_rl.v400.student import model_v2, obs_spec, obs_spec_v2, obs_v2_train

REPO_ROOT = Path(__file__).resolve().parents[1]
OBS_SPEC_FILE = REPO_ROOT / "touhou_rl" / "v400" / "student" / "obs_spec.py"
#: Byte-lock of ``touhou_rl/v400/student/obs_spec.py`` (obs-v1, 2026-11-09).
OBS_V1_FILE_SHA256 = "0d47daf1bddc2b6fb8a33eb54a558f2d40f4a8c3b3d878155035d914d455578b"
OBS_V1_DIGEST = "24f0ddace59104938dbe991c0d8efbb6a8cd21dfd5a29ebcb6d023b5b364a381"
OBS_V2_DIGEST = "95110743993c090e2b30fa40ac349a350e74021b35464e3204c784a69ae4dba3"
OBS_V2_ROOT = REPO_ROOT / ".tmp_v400_obsv2" / "obsv2"


def _load_tool(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def eval_tool():
    return _load_tool("v400_student_eval_obs_v2", REPO_ROOT / "tools" / "rl" /
                      "v400_student_eval.py")


# ---------------------------------------------------------------------------
# 1. obs-v1 frozen; obs-v2 additive
# ---------------------------------------------------------------------------
def test_obs_v1_file_bytes_and_digest_are_frozen():
    import hashlib

    assert hashlib.sha256(OBS_SPEC_FILE.read_bytes()).hexdigest() == OBS_V1_FILE_SHA256, (
        "obs-spec.py (obs-v1) changed; obs-v2 is additive and must not edit v1")
    assert obs_spec.OBS_SPEC_VERSION == "obs-v1"
    assert obs_spec.obs_spec_digest() == OBS_V1_DIGEST


def test_obs_v2_digest_is_frozen():
    assert obs_spec_v2.OBS_SPEC_VERSION == "obs-v2"
    assert obs_spec_v2.obs_spec_v2_digest() == OBS_V2_DIGEST
    assert obs_spec_v2.OBS_V1_DIGEST == OBS_V1_DIGEST


def test_obs_v2_is_obs_v1_plus_only_the_action_history():
    v1 = obs_spec.obs_spec()
    v2 = obs_spec_v2.obs_spec_v2()
    added = set(v2) - set(v1)
    assert added == {"supersedes", "action_history", "self_state", "danger_summary"}
    assert v2["version"] == "obs-v2" and v1["version"] == "obs-v1"
    assert v2["excluded_inputs"] != v1["excluded_inputs"]
    # Every inherited section is untouched, value for value.
    for key in v1:
        if key in ("version", "excluded_inputs"):
            continue
        assert v2[key] == v1[key], f"obs-v2 changed inherited field {key}"
    assert v2["supersedes"] == {"version": "obs-v1", "digest": OBS_V1_DIGEST}
    assert v2["action_history"]["decisions"] == 8
    assert v2["action_history"]["dims"] == 144
    assert v2["action_history"]["excludes_current_action"] is True
    # The two optional inputs are declared absent, not silently dropped.
    assert v2["self_state"]["present"] is False
    assert v2["danger_summary"]["present"] is False
    assert "action history" in json.dumps(v2["excluded_inputs"]).lower()


def test_check_obs_spec_v2_fails_closed_on_a_v1_checkpoint():
    with pytest.raises(obs_spec_v2.ObsSpecV2Mismatch):
        obs_spec_v2.check_obs_spec_v2("obs-v1", OBS_V1_DIGEST)
    with pytest.raises(obs_spec_v2.ObsSpecV2Mismatch):
        obs_spec_v2.check_obs_spec_v2("obs-v2", OBS_V1_DIGEST)
    obs_spec_v2.check_obs_spec_v2("obs-v2", OBS_V2_DIGEST)


# ---------------------------------------------------------------------------
# 2. action history: encoding, causality, boundaries
# ---------------------------------------------------------------------------
def test_history_holds_the_previous_eight_actions_oldest_first_and_never_the_current():
    actions = np.arange(20, dtype=np.int16) % obs_spec.N_MOVE_CLASSES
    history = obs_spec_v2.build_action_history(actions)
    assert history.shape == (20, 8, 18)
    assert history.dtype == np.uint8
    for t in range(20):
        expected = list(actions[max(0, t - 8):t])
        rows = history[t]
        # the trailing ``len(expected)`` rows are the real actions, oldest first
        for offset, action in enumerate(expected):
            row = rows[8 - len(expected) + offset]
            assert int(row.argmax()) == int(action)
            assert int(row.sum()) == 1
        # rows before the history starts are all-zero padding, never action 0
        for row in rows[:8 - len(expected)]:
            assert int(row.sum()) == 0


def test_history_clears_at_the_stage_boundary():
    actions = np.full(12, 7, dtype=np.int16)
    stage = np.array([0] * 6 + [1] * 6, dtype=np.int32)
    history = obs_spec_v2.build_action_history(actions, stage)
    # decision 6 is the first of the new stage: no history at all
    assert history[6].sum() == 0
    # decision 7 only knows decision 6's action (not 0..5)
    filled = np.argwhere(history[7].sum(axis=-1) > 0).ravel()
    assert filled.tolist() == [7]
    # decision 12 (outside this shard) would only know 8..11
    history2 = obs_spec_v2.build_action_history(np.full(14, 3, dtype=np.int16),
                                               np.array([0] * 6 + [1] * 8, dtype=np.int32))
    # 7 actions (6..12) fit into the window, so the first row is the only padding
    assert history2[13].sum(axis=-1).tolist() == [0, 1, 1, 1, 1, 1, 1, 1]


def test_history_is_cleared_at_the_shard_start_of_every_shard():
    first = obs_spec_v2.build_action_history(np.full(10, 5, dtype=np.int16))
    # a separate shard with the same actions must not inherit the previous one
    assert first[0].sum() == 0
    assert first[7].sum(axis=-1).tolist() == [0, 1, 1, 1, 1, 1, 1, 1]
    assert first[8].sum() == 8


def test_executed_action_is_read_from_each_shard_kind():
    teacher_rows = [{"chosen_plan": {"first_action": 3}, "stage": 0},
                    {"chosen_plan": {"first_action": 4}, "stage": 0}]
    cand, stage = obs_spec_v2.reconstruct_from_label_rows(teacher_rows)
    assert cand.tolist() == [3, 4]
    assert stage.tolist() == [0, 0]

    dagger_rows = [{"chosen_plan": {"first_action": 3}, "stage": 0,
                    "dagger": {"executed_cand": 11, "executed_bomb": 0}},
                   {"chosen_plan": {"first_action": 4}, "stage": 0,
                    "dagger": {"executed_cand": 12, "executed_bomb": 1}}]
    cand, _ = obs_spec_v2.reconstruct_from_label_rows(dagger_rows)
    # the executed action wins: the trajectory did 11/12, not the teacher's 3/4
    assert cand.tolist() == [11, 12]


def test_history_refuses_a_row_with_no_reconstructible_action():
    with pytest.raises(obs_spec_v2.ObsSpecV2Error, match="neither"):
        obs_spec_v2.reconstruct_from_label_rows([{"stage": 0}])


def test_history_rejects_out_of_range_and_bad_shapes():
    with pytest.raises(obs_spec_v2.ObsSpecV2Error):
        obs_spec_v2.build_action_history(np.array([18], dtype=np.int16))
    with pytest.raises(obs_spec_v2.ObsSpecV2Error):
        obs_spec_v2.check_action_history(np.zeros((3, 8, 18), dtype=np.float32))
    bad = np.zeros((3, 8, 18), dtype=np.uint8)
    bad[1, 2, 3] = 2
    with pytest.raises(obs_spec_v2.ObsSpecV2Error):
        obs_spec_v2.check_action_history(bad)


def test_deployment_buffer_matches_training_reconstruction():
    actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    training = obs_spec_v2.build_action_history(np.asarray(actions, dtype=np.int16))
    buffer = obs_spec_v2.ActionHistoryBuffer()
    for t, action in enumerate(actions):
        # at decision t the buffer holds a[0..t-1]
        assert np.array_equal(buffer.rows(), training[t]), f"row {t} differs"
        buffer.push(action)
    buffer.reset()
    assert buffer.rows().sum() == 0


# ---------------------------------------------------------------------------
# 3. GRU: sequence packing, BPTT truncation, deployment cache
# ---------------------------------------------------------------------------
def _split(count: int = 20, *, with_stage_boundary: bool = False) -> obs_v2_train.SplitArraysV2:
    play = np.zeros((count, 4, 64, 64), dtype=np.uint8)
    for index in range(count):
        play[index] = index % 256
    hud = np.zeros((count, 56, 160), dtype=np.uint8)
    hist = np.zeros((count, 8, 18), dtype=np.uint8)
    stage = np.zeros(count, dtype=np.int32)
    if with_stage_boundary:
        stage[count // 2:] = 1
    shard_index = np.zeros(count, dtype=np.int16)
    boundary = np.zeros(count, dtype=bool)
    boundary[0] = True
    if with_stage_boundary:
        boundary[count // 2] = True
    return obs_v2_train.SplitArraysV2(
        name="unit", play=play, hud=hud, hist=hist, stage=stage,
        scores=np.zeros((count, 18), dtype=np.float32),
        bomb=np.zeros(count, dtype=np.int8),
        verified_count=np.zeros(count, dtype=np.int16),
        shard_index=shard_index,
        segment=(np.cumsum(boundary) - 1).astype(np.int32),
        shard_keys=["unit"], decisions_by_shard={"unit": count})


def test_gru_batch_packs_step_s_to_decision_end_minus_seven_plus_s():
    import torch

    arrays = _split(20)
    ends = np.array([10, 19], dtype=np.int64)
    play, hud, hist, scores, bomb = obs_v2_train.batch_sequence(
        arrays, ends, 8, "cpu", torch)
    assert play.shape == (8, 2, 4, 64, 64)
    assert hud.shape == (8, 2, 1, 56, 160)
    assert hist.shape == (8, 2, 8, 18)
    assert scores.shape == (8, 2, 18)
    assert bomb.shape == (8, 2)
    for step in range(8):
        for column, end in enumerate(ends):
            # step 7 is the decision itself; step 0 is seven decisions earlier
            assert int(round(float(play[step, column, 0, 0, 0].item()) * 255)) == \
                int((end - 7 + step) % 256)


def test_gru_window_ends_never_cross_a_stage_or_shard_boundary():
    arrays = _split(20, with_stage_boundary=True)
    ends = obs_v2_train.valid_window_ends(arrays, 8)
    assert ends.min() == 7
    # decision 10 would need 3..10, which straddles the boundary at 10 -> refused;
    # decision 11 needs 4..11, still straddling; only windows fully inside the second
    # segment (10..19) survive, i.e. ends >= 17.
    assert 10 not in ends.tolist()
    assert 11 not in ends.tolist()
    assert ends.tolist() == [7, 8, 9, 17, 18, 19]


def test_truncate_bptt_cuts_the_prefix_gradient_only():
    import torch

    torch.manual_seed(0)
    model = model_v2.StudentGRU(width=8, hidden=16, hist_hidden=8, gru_input=16)
    model.eval()
    play = torch.rand(8, 2, 4, 64, 64, requires_grad=True)
    hud = torch.rand(8, 2, 1, 56, 160)
    hist = torch.zeros(8, 2, 8, 18)
    sequence = model.encode(play, hud, hist)
    truncated = obs_v2_train.truncate_bptt(sequence, 4, 8)
    # the tensor keeps its 8 steps; what is cut is the gradient, checked below
    assert truncated.shape == sequence.shape
    output, _ = model.gru(truncated)
    logits = model.move_head(output)
    logits[4:].pow(2).sum().backward()
    per_step = play.grad.abs().sum(dim=(1, 2, 3, 4))
    assert float(per_step[:4].max()) == 0.0, "detached steps must get no gradient"
    assert float(per_step[4:].min()) > 0.0, "the loss steps must reach the encoder"


def test_deployment_cache_reproduces_the_batch_forward_last_step():
    import torch

    torch.manual_seed(1)
    model = model_v2.StudentGRU(width=8, hidden=16, hist_hidden=8, gru_input=16)
    model.eval()
    play = torch.rand(8, 1, 4, 64, 64)
    hud = torch.rand(8, 1, 1, 56, 160)
    hist = torch.zeros(8, 1, 8, 18)
    batch_move, batch_bomb = model(play, hud, hist)
    cache = model_v2.GRUCache(model.seq_len)
    step_move, step_bomb = None, None
    with torch.no_grad():
        for index in range(8):
            step_move, step_bomb = model.act_step(play[index], hud[index], hist[index],
                                                  cache)
    assert step_move == int(batch_move[-1, 0].argmax())
    assert step_bomb == int(batch_bomb[-1, 0].argmax())


def test_deployment_cache_pads_the_episode_start_by_repeating_the_first_decision():
    import torch

    torch.manual_seed(2)
    model = model_v2.StudentGRU(width=8, hidden=16, hist_hidden=8, gru_input=16)
    cache = model_v2.GRUCache(8)
    embedding = torch.randn(1, 16)
    window = cache.push(embedding)
    assert window.shape == (8, 1, 16)
    for step in range(8):
        assert torch.equal(window[step], embedding)
    window2 = cache.push(torch.randn(1, 16))
    assert torch.equal(window2[6], embedding)


def test_deployment_api_takes_the_8x18_window_not_the_flat_144():
    """Regression: the eval loop feeds ``(B,8,18)``; a flat ``(B,144)`` must be refused."""

    import torch

    mlp = model_v2.StudentNetV2(width=8, hidden=16, hist_hidden=8)
    mlp.eval()
    play = torch.rand(1, 4, 64, 64)
    hud = torch.rand(1, 1, 56, 160)
    window = torch.zeros(1, 8, 18)
    cand, bomb = mlp.act(play, hud, window)
    assert 0 <= cand < obs_spec.N_MOVE_CLASSES and bomb in (0, 1)
    with pytest.raises(ValueError, match="act_hist"):
        mlp.act(play, hud, torch.zeros(1, 144))

    gru = model_v2.StudentGRU(width=8, hidden=16, hist_hidden=8, gru_input=16)
    gru.eval()
    cache = model_v2.GRUCache(gru.seq_len)
    cand, bomb = gru.act_step(play, hud, window, cache)
    assert 0 <= cand < obs_spec.N_MOVE_CLASSES and bomb in (0, 1)


def test_gru_evaluation_refuses_windows_before_the_split_start():
    import torch

    torch.manual_seed(3)
    model = model_v2.StudentGRU(width=8, hidden=16, hist_hidden=8, gru_input=16)
    arrays = _split(20)
    logits, bombs, used = obs_v2_train.predict_gru(
        model, arrays, np.arange(20), device="cpu", batch_size=8, seq_len=8, torch=torch)
    # indices 0..6 have no 8-decision window; they must be refused, never wrapped
    assert used.tolist() == list(range(7, 20))
    assert logits.shape == (13, obs_spec.N_MOVE_CLASSES)
    assert bombs.shape == (13, obs_spec.N_BOMB_CLASSES)


def test_parameter_budgets():
    mlp = model_v2.StudentNetV2(width=48, hidden=256, hist_hidden=64)
    gru = model_v2.StudentGRU(width=48, hidden=256, hist_hidden=64, gru_input=128)
    assert mlp.n_parameters() <= 300_000, "M2 must stay under the 300k parameter cap"
    assert gru.gru.num_layers == 1, "single-layer GRU"
    assert gru.gru.hidden_size == 256
    assert gru.seq_len == 8
    assert mlp.config()["arch"] == "mlp" and gru.config()["arch"] == "gru"


def test_obs_v2_checkpoint_round_trip_and_spec_mismatch(tmp_path):
    import torch

    model = model_v2.StudentNetV2(width=8, hidden=16, hist_hidden=8)
    path = model_v2.save_checkpoint_v2(tmp_path / "ckpt.pt", model,
                                       train={"tau": 2.0}, provenance={})
    loaded, payload = model_v2.load_checkpoint_v2(path)
    assert payload["obs_spec_version"] == "obs-v2"
    assert payload["obs_spec_digest"] == OBS_V2_DIGEST
    assert payload["parameters"] == model.n_parameters()
    assert loaded.config() == model.config()

    tampered = torch.load(path, map_location="cpu", weights_only=False)
    tampered["obs_spec_digest"] = OBS_V1_DIGEST
    torch.save(tampered, tmp_path / "bad.pt")
    with pytest.raises(obs_spec_v2.ObsSpecV2Mismatch):
        model_v2.load_checkpoint_v2(tmp_path / "bad.pt")

    # an obs-v1 checkpoint is refused by the obs-v2 loader, not silently reinterpreted
    from touhou_rl.v400.student.model import save_checkpoint

    v1_path = save_checkpoint(tmp_path / "v1.pt",
                              model_v2.StudentNetV2(width=8, hidden=8, hist_hidden=8))
    with pytest.raises(ValueError):
        model_v2.load_checkpoint_v2(v1_path)


# ---------------------------------------------------------------------------
# 4. evaluation schema: an obs-v2 episode is validated against obs-v2 only
# ---------------------------------------------------------------------------
def _episode(eval_tool, **overrides) -> dict:
    record = {
        "schema": eval_tool.EPISODE_SCHEMA,
        "checkpoint": "ckpt.pt",
        "checkpoint_sha256": "0" * 64,
        "obs_spec_version": obs_spec_v2.OBS_SPEC_VERSION,
        "obs_spec_digest": OBS_V2_DIGEST,
        "seed": 1000,
        "suite": "S1N",
        "stage": 0,
        "difficulty": "normal",
        "max_frames": 19887,
        "frames": 1816,
        "deaths": 1.0,
        "score": 12345,
        "end": "death_stop",
        "privilege": "deploy_pixel",
        "policy_input": "play64 + hud + act_hist",
        "wall_s": 20.0,
        "engine_sha256": "1" * 64,
        "not_a_gate": True,
        "obs_window": "consecutive",
        "window_alignment": {"renders_per_decision": 2},
        "stop_on_death": True,
        "exit": "controller",
        "stage_start": 0,
        "stage_end": 0,
        "cleared": False,
    }
    record.update(overrides)
    return record


def test_eval_schema_accepts_an_obs_v2_episode_only_under_obs_v2(eval_tool):
    record = _episode(eval_tool)
    eval_tool.validate_episode_record(
        record, expected_spec=(obs_spec_v2.OBS_SPEC_VERSION, OBS_V2_DIGEST))
    # the default expectation is obs-v1, so the same record must be refused there
    with pytest.raises(eval_tool.EvalError, match="obs spec"):
        eval_tool.validate_episode_record(record)


def test_eval_schema_still_rejects_a_mixed_window_ruler(eval_tool):
    record = _episode(eval_tool, window_alignment={"renders_per_decision": 1})
    with pytest.raises(eval_tool.EvalError, match="render"):
        eval_tool.validate_episode_record(
            record, expected_spec=(obs_spec_v2.OBS_SPEC_VERSION, OBS_V2_DIGEST))


def test_eval_parser_exposes_the_obs_v2_flag(eval_tool):
    args = eval_tool.build_parser().parse_args(
        ["--engine", "e", "--data-dir", "d", "--checkpoint", "c", "--obs-v2"])
    assert args.obs_v2 is True
    assert args.obs_window == "consecutive"
    default = eval_tool.build_parser().parse_args(
        ["--engine", "e", "--data-dir", "d", "--checkpoint", "c"])
    assert default.obs_v2 is False


# ---------------------------------------------------------------------------
# 5. the built obs-v2 root (present on this machine; skipped elsewhere)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not (OBS_V2_ROOT / "manifest.json").is_file(),
                    reason="obs-v2 root not built on this machine")
def test_built_obs_v2_root_matches_the_v1_corpus():
    from touhou_rl.v400.student.obs_v2_dataset import load_obs_v2_shards

    manifest = json.loads((OBS_V2_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["obs_spec_version"] == "obs-v2"
    assert manifest["obs_spec_digest"] == OBS_V2_DIGEST
    assert manifest["base_obs_spec_digest"] == OBS_V1_DIGEST
    # D0 53,512 + DAgger R1 23,400
    assert manifest["totals"]["decisions"] == 76_912
    assert manifest["totals"]["shards"] == 27
    shards = load_obs_v2_shards(OBS_V2_ROOT)
    assert len(shards) == 27
    assert sum(shard.decisions for shard in shards.values()) == 76_912
    payload_bytes = 0
    header_bytes = 0
    for shard in shards.values():
        assert shard.act_hist.shape == (shard.decisions, 8, 18)
        assert shard.stage.shape == (shard.decisions,)
        assert shard.meta["v1_planes"], "the v1 plane hashes must be recorded"
        path = shard.dir / "act_hist.npy"
        payload_bytes += shard.decisions * 144
        # .npy carries a header; the totals record file bytes, so split them out
        header_bytes += path.stat().st_size - shard.decisions * 144
    assert payload_bytes == 76_912 * 144
    assert manifest["totals"]["act_hist_bytes"] == payload_bytes + header_bytes
