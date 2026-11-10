"""Contract tests for the P4 RL prototype (EXP-P4-001).

Everything here runs offline: the engine is replaced by
:class:`touhou_rl.v400.rl.env.FakeBackend`, so the reward rules, the termination
rules, the action encoding, the vectorised determinism and the CI/claim
integration are checkable without a display, a GPU or a 47 MB game archive.

The one thing a fake cannot check is whether the real engine reproduces the
published per-seed frames; that is ``tools/rl/v400_rl_train.py --validate-ruler``
and it is reported in the experiment card, not asserted here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from touhou_rl.v400.gym.actions import N_CAND, cand_to_mask
from touhou_rl.v400.rl import env as rl_env
from touhou_rl.v400.rl.env import (
    BC_BASELINE_MAGNITUDE,
    END_DEATH_STOP,
    END_ENGINE_EXIT,
    END_FRAME_CAP,
    END_GAME_OVER,
    END_STAGE_CLEAR,
    FakeBackend,
    N_ACTIONS,
    REWARD_DEATH,
    REWARD_SCORE_PER_POINT,
    REWARD_STAGE_CLEAR,
    REWARD_SURVIVE_PER_FRAME,
    RlEnv,
    RlEnvError,
    VecRlEnv,
    action_to_cand_bomb,
    cand_bomb_to_action,
    reward_scale_report,
    reward_scale_table,
)

REPO = Path(__file__).resolve().parents[1]


def make_env(script, *, max_frames=100, stop_on_death=True, start=None) -> RlEnv:
    backend = FakeBackend(script, start=start)
    return RlEnv(backend, max_frames=max_frames, stop_on_death=stop_on_death)


def play(env: RlEnv, actions, seed: int = 1000):
    obs, info = env.reset(seed)
    trace = []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append({"reward": reward, "terminated": terminated,
                      "truncated": truncated, "info": info})
        if terminated or truncated:
            break
    return trace


# --------------------------------------------------------------------------- #
# action mapping
# --------------------------------------------------------------------------- #

def test_action_mapping_is_the_frozen_cand_times_bomb_grid():
    assert N_ACTIONS == N_CAND * 2 == 36
    assert len(rl_env.ACTION_NAMES) == N_ACTIONS
    seen = set()
    for action in range(N_ACTIONS):
        cand, bomb = action_to_cand_bomb(action)
        assert 0 <= cand < N_CAND
        assert bomb in (0, 1)
        assert cand_bomb_to_action(cand, bomb) == action
        # The mask is the frozen encoder's, not a second table.
        assert rl_env.cand_to_mask(cand, bomb) == cand_to_mask(cand, bomb)
        seen.add(rl_env.cand_to_mask(cand, bomb))
    # 18 move-focus masks x bomb on/off: every joint action is a distinct mask.
    assert len(seen) == N_ACTIONS


@pytest.mark.parametrize("bad", [-1, N_ACTIONS, 999])
def test_action_out_of_range_fails_closed(bad):
    with pytest.raises(RlEnvError):
        action_to_cand_bomb(bad)
    env = make_env([{"advanced": 1}] * 4)
    env.reset(1000)
    with pytest.raises(RlEnvError):
        env.step(bad)


def test_bomb_half_of_the_grid_is_reachable():
    bombs = {action_to_cand_bomb(action)[1] for action in range(N_ACTIONS)}
    assert bombs == {0, 1}
    # cand 3, bomb 1 -> STUDENT_ACTIONS order index 7
    assert cand_bomb_to_action(3, 1) == 7


# --------------------------------------------------------------------------- #
# reward terms
# --------------------------------------------------------------------------- #

def test_survive_term_is_one_per_logic_frame():
    env = make_env([{"advanced": 1}] * 10)
    trace = play(env, [0, 0])
    assert len(trace) == 2
    for row in trace:
        assert row["reward"] == pytest.approx(2 * REWARD_SURVIVE_PER_FRAME)
    assert trace[-1]["info"]["frames"] == 4


def test_score_term_is_delta_over_one_thousand():
    script = [{"advanced": 1}, {"advanced": 1, "state": FakeBackend.patch(score=5000)},
              {"advanced": 1}, {"advanced": 1, "state": FakeBackend.patch(score=9000)}]
    env = make_env(script)
    trace = play(env, [0, 0])
    first, second = trace[0], trace[1]
    assert first["reward"] == pytest.approx(2.0 + 5000 * REWARD_SCORE_PER_POINT)
    assert second["reward"] == pytest.approx(2.0 + 4000 * REWARD_SCORE_PER_POINT)


def test_death_term_is_minus_one_hundred_and_ends_the_episode_with_stop_on_death():
    script = [{"advanced": 1}, {"advanced": 1},
              {"advanced": 1, "state": FakeBackend.patch(deaths=1.0, lives=2.0)}]
    env = make_env(script, max_frames=100, stop_on_death=True)
    trace = play(env, [0, 0])
    last = trace[-1]
    assert last["terminated"] is True
    assert last["info"]["end"] == END_DEATH_STOP
    assert last["info"]["step_reward_terms"]["death"] == pytest.approx(REWARD_DEATH)
    # survive(1) + death(-100) + the 3 bombs still in hand at the terminal step.
    assert last["reward"] == pytest.approx(1.0 + REWARD_DEATH + 3.0)


def test_death_without_stop_on_death_continues_until_lives_run_out():
    script = [{"advanced": 1},
              {"advanced": 1, "state": FakeBackend.patch(deaths=1.0, lives=1.0)},
              {"advanced": 1, "state": FakeBackend.patch(deaths=1.0, lives=1.0)},
              {"advanced": 1, "state": FakeBackend.patch(deaths=2.0, lives=0.0)}]
    env = make_env(script, max_frames=100, stop_on_death=False)
    trace = play(env, [0, 0, 0, 0, 0])
    assert trace[0]["terminated"] is False
    assert trace[0]["info"]["deaths"] == pytest.approx(1.0)
    assert trace[0]["info"]["step_reward_terms"]["death"] == pytest.approx(REWARD_DEATH)
    last = trace[-1]
    assert last["terminated"] is True
    assert last["info"]["end"] == END_GAME_OVER


def test_bomb_remaining_is_paid_only_on_the_terminal_transition():
    script = [{"advanced": 1}, {"advanced": 1},
              {"advanced": 1, "state": FakeBackend.patch(deaths=1.0, lives=2.0)}]
    env = make_env(script, start={"deaths": 0.0, "lives": 3.0, "bombs": 2.0,
                                  "score": 0.0, "stage": 0.0})
    trace = play(env, [0, 0])
    assert trace[0]["info"]["step_reward_terms"]["bomb_remaining"] == 0.0
    assert trace[-1]["info"]["step_reward_terms"]["bomb_remaining"] == pytest.approx(2.0)


def test_stage_clear_pays_a_thousand_and_terminates():
    script = [{"advanced": 1}, {"advanced": 1, "state": FakeBackend.patch(stage=1)}]
    env = make_env(script, max_frames=100)
    trace = play(env, [0])
    last = trace[-1]
    assert last["terminated"] is True
    assert last["info"]["end"] == END_STAGE_CLEAR
    terms = last["info"]["step_reward_terms"]
    assert terms["stage_clear"] == pytest.approx(REWARD_STAGE_CLEAR)
    assert last["info"]["cleared"] is True


def test_engine_exit_is_its_own_end_label():
    script = [{"advanced": 1, "exit": True}]
    env = make_env(script)
    trace = play(env, [0])
    assert trace[-1]["info"]["end"] == END_ENGINE_EXIT
    assert trace[-1]["terminated"] is True


def test_frame_cap_truncates_and_reports_frame_cap():
    env = make_env([{"advanced": 1}] * 50, max_frames=5)
    trace = play(env, [0] * 10)
    assert trace[-1]["truncated"] is True
    assert trace[-1]["terminated"] is False
    assert trace[-1]["info"]["end"] == END_FRAME_CAP
    assert trace[-1]["info"]["frames"] == 5


# --------------------------------------------------------------------------- #
# observation contract
# --------------------------------------------------------------------------- #

def test_observation_is_the_frozen_obs_v1_tensor_pair():
    env = make_env([{"advanced": 1}] * 8)
    obs, _info = env.reset(1000)
    assert set(obs) == {"play", "hud"}
    assert obs["play"].shape == (4, 64, 64) and obs["play"].dtype == np.float32
    assert obs["hud"].shape == (1, 56, 160) and obs["hud"].dtype == np.float32
    assert obs["play"].min() >= 0.0 and obs["play"].max() <= 1.0
    obs2, *_ = env.step(0)
    assert obs2["play"].shape == obs["play"].shape


def test_privileged_state_is_opt_in_and_flagged():
    env = make_env([{"advanced": 1}] * 4)
    env.reset(1000)
    _obs, _r, _t, _tr, info = env.step(0)
    assert "state" not in info
    assert info["privileged_diagnostic"] is False
    assert info["privilege"] == "training_engine_truth_not_in_observation"

    env2 = RlEnv(FakeBackend([{"advanced": 1}] * 4), max_frames=100,
                 privileged_state=True)
    env2.reset(1000)
    _obs, _r, _t, _tr, info2 = env2.step(0)
    assert "state" in info2 and info2["privileged_diagnostic"] is True
    assert info2["state"]["lives_remaining"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# episode lifecycle
# --------------------------------------------------------------------------- #

def test_reset_reuses_a_fake_engine_but_counts_a_restart():
    backend = FakeBackend([{"advanced": 1}] * 8)
    env = RlEnv(backend, max_frames=100)
    env.reset(1000)
    assert backend.restarts == 0
    env.reset(1001)
    # The real engine supports one RESET per process (EXP-P4-001), so a new
    # episode must restart it; the fake counts the restart the same way.
    assert backend.restarts == 1
    assert backend.reset_count == 2


def test_episode_return_accumulates_the_terms():
    script = [{"advanced": 1},
              {"advanced": 1, "state": FakeBackend.patch(score=2000)},
              {"advanced": 1, "state": FakeBackend.patch(deaths=1.0, lives=2.0)}]
    env = make_env(script, start={"deaths": 0.0, "lives": 3.0, "bombs": 1.0,
                                  "score": 0.0, "stage": 0.0})
    trace = play(env, [0, 0])
    last = trace[-1]["info"]
    expected = 3.0 + 2000 * REWARD_SCORE_PER_POINT + REWARD_DEATH + 1.0
    assert last["episode_return"] == pytest.approx(expected)
    assert last["reward_terms"]["survive"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# parallel determinism
# --------------------------------------------------------------------------- #

def _deterministic_script(frames: int = 40):
    script = []
    for index in range(frames):
        step = {"advanced": 1}
        if index % 17 == 16:
            step["state"] = FakeBackend.patch(score=500 * (index + 1))
        script.append(step)
    return script


def test_parallel_envs_are_deterministic_across_two_runs():
    def run_once():
        vec = VecRlEnv([RlEnv(FakeBackend(_deterministic_script()), max_frames=60)
                        for _ in range(4)])
        try:
            results = vec.reset_all([1000, 1001, 1002, 1003])
            payload = [[float(row[0]["play"].sum()), float(row[0]["hud"].sum())]
                       for row in results]
            for _ in range(12):
                outputs, _new = vec.step_all_with_reset([0, 1, 2, 3], [2000] * 4)
                payload.append([[row[1], row[2], row[3],
                                 float(row[0]["play"].sum())] for row in outputs])
            return payload
        finally:
            vec.close()

    assert run_once() == run_once()


def test_vec_env_preserves_order_and_rejects_wrong_arity():
    vec = VecRlEnv([RlEnv(FakeBackend(_deterministic_script()), max_frames=60)
                    for _ in range(3)])
    try:
        vec.reset_all([1000, 1001, 1002])
        with pytest.raises(RlEnvError):
            vec.step_all([0, 0])
        outputs = vec.step_all([0, 1, 2])
        assert len(outputs) == 3
    finally:
        vec.close()


# --------------------------------------------------------------------------- #
# reward table
# --------------------------------------------------------------------------- #

def test_reward_table_covers_every_paid_term_and_the_shares():
    report = reward_scale_report()
    names = [term["name"] for term in report["terms"]]
    assert names == ["survive", "score", "death", "bomb_remaining", "stage_clear"]
    assert set(report["bc_baseline_magnitude"]) == set(names)
    shares = report["bc_baseline_share"]
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)
    # Survival dominates by design; the card relies on this number.
    assert shares["survive"] > 0.9
    # m1 uses all three bombs, so hoarding is worth nothing to the baseline.
    assert BC_BASELINE_MAGNITUDE["bomb_remaining"] == 0.0
    assert len(reward_scale_table()) == 5


# --------------------------------------------------------------------------- #
# CI report + claim integration (the P3 criterion template)
# --------------------------------------------------------------------------- #

def _write_arm(root: Path, label: str, frames) -> Path:
    arm = root / label
    arm.mkdir(parents=True, exist_ok=True)
    (arm / "episodes.jsonl").write_text(
        "".join(json.dumps({"seed": 1000 + index, "frames": int(value),
                            "end": "death_stop"}) + "\n"
                for index, value in enumerate(frames)), encoding="utf-8")
    (arm / "summary.json").write_text(
        json.dumps({"label": label, "episodes": len(list(frames)),
                    "mean_frames": float(np.mean(list(frames)))}), encoding="utf-8")
    return arm


def _run_claim(args, tmp_path):
    return subprocess.run(
        [sys.executable, str(REPO / "tools" / "rl" / "v400_eval_power.py"), "claim", *args],
        capture_output=True, text=True, cwd=str(REPO), timeout=300)


def test_ci_report_has_mean_sd_se_bootstrap_ci_for_twenty_episodes(tmp_path):
    frames = [2000 + 40 * index for index in range(20)]
    arm = _write_arm(tmp_path, "rl", frames)
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "rl" / "v400_eval_power.py"),
         "baseline", "--run", str(arm),
         "--json", str(tmp_path / "baseline.json")],
        capture_output=True, text=True, cwd=str(REPO), timeout=300)
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    stats = payload["arm"]
    assert stats["n"] == 20
    for key in ("mean", "sd", "se", "ci_low", "ci_high", "median", "min", "max"):
        assert stats[key] is not None
    assert stats["ci_low"] < stats["mean"] < stats["ci_high"]
    assert payload["protocol_floor_episodes"] == 20


def test_claim_refuses_an_arm_without_a_ci(tmp_path):
    good = _write_arm(tmp_path, "good", [5000] * 20)
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "summary.json").write_text(
        json.dumps({"label": "bare", "mean_frames": 2248.85}), encoding="utf-8")
    result = _run_claim(["--a", f"good={good}", "--b", f"bare={bare}"], tmp_path)
    assert result.returncode == 2, result.stdout
    payload = json.loads(result.stdout)
    assert "无法判定" in payload["verdict"]


def test_claim_finds_an_effect_when_the_two_cis_are_disjoint(tmp_path):
    low = _write_arm(tmp_path, "low", [2000 + 10 * index for index in range(20)])
    high = _write_arm(tmp_path, "high", [4000 + 10 * index for index in range(20)])
    result = _run_claim(["--a", f"high={high}", "--b", f"low={low}"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "有效应"
    assert payload["ci_overlap"] is False
    assert payload["delta"] > 0


def test_claim_says_no_verdict_when_the_cis_overlap(tmp_path):
    same_a = _write_arm(tmp_path, "same_a", [2249, 2251] * 10)
    same_b = _write_arm(tmp_path, "same_b", [2250, 2250] * 10)
    result = _run_claim(["--a", f"same_a={same_a}", "--b", f"same_b={same_b}"], tmp_path)
    assert result.returncode == 2, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ci_overlap"] is True
    assert "无法判定" in payload["verdict"]
