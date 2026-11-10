"""EXP-P3-009: the evaluation protocol must be as testable as the policy.

Four contracts, each of which the previous five-episode ruler violated:

1. ``--seeds`` parses ranges/lists and **refuses** nonsense instead of silently
   evaluating a different set of episodes.
2. ``--repeats`` really re-runs every seed and the summary aggregates over
   (seed, repeat) without pretending the repeats are extra seeds.
3. A reported row carries ``sd`` / ``se`` / a bootstrap **CI**, and the CI is
   reproducible and brackets the mean.
4. The variance decomposition separates ``σ²_ckpt`` from ``σ²_episode``, and the
   criterion template **refuses** a comparison that has no CI — "no verdict" is a
   first-class outcome, not an error message.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_PATH = REPO_ROOT / "tools" / "rl" / "v400_student_eval.py"
POWER_PATH = REPO_ROOT / "tools" / "rl" / "v400_eval_power.py"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ev():
    return _load("v400_student_eval_power_case", EVAL_PATH)


@pytest.fixture(scope="module")
def pw():
    return _load("v400_eval_power_case", POWER_PATH)


def _episodes(frames, *, seeds=None, level="death_stop"):
    """Minimal but schema-shaped episode records for aggregator tests."""

    records = []
    for index, value in enumerate(frames):
        seed = seeds[index] if seeds else 1000 + index
        records.append({
            "frames": int(value), "seed": seed, "repeat": 0, "score": 1,
            "deaths": 1.0, "end": level, "cleared": False, "stage": 0,
            "difficulty": "normal", "max_frames": 19887, "stop_on_death": True,
            "obs_window": "consecutive", "window_alignment": None,
            "action_histogram": None,
        })
    return records


# --------------------------------------------------------------------------- #
# 1. seed specs
# --------------------------------------------------------------------------- #

def test_seed_range_becomes_twenty_episodes(ev):
    seeds = ev.parse_seeds("1000-1019")
    assert seeds == list(range(1000, 1020))
    assert len(seeds) == 20


def test_seed_list_range_mix_preserves_order_and_dedupes(ev):
    assert ev.parse_seeds("1000,1002-1004,1001") == [1000, 1002, 1003, 1004, 1001]
    # a duplicate seed is one episode, not two
    assert ev.parse_seeds("1000-1002,1001") == [1000, 1001, 1002]


@pytest.mark.parametrize("spec", ["", "   ", "1000-", "-1000", "1019-1000",
                                 "abc", "1000-abc", "1000-9999", "-5"])
def test_bad_seed_specs_fail_closed(ev, spec):
    with pytest.raises(ev.EvalError):
        ev.parse_seeds(spec)


def test_seed_cap_is_enforced(ev):
    assert len(ev.parse_seeds("0-255")) == ev.MAX_SEEDS
    with pytest.raises(ev.EvalError):
        ev.parse_seeds("0-256")


def test_default_cli_seed_spec_is_the_historical_five(ev):
    args = ev.build_parser().parse_args(
        ["--engine", "e", "--data-dir", "d", "--checkpoint", "c"])
    assert ev.parse_seeds(args.seeds) == [1000, 1001, 1002, 1003, 1004]
    assert args.repeats == 1


def test_cli_accepts_range_and_repeats(ev):
    args = ev.build_parser().parse_args(
        ["--engine", "e", "--data-dir", "d", "--checkpoint", "c",
         "--seeds", "1000-1019", "--repeats", "2"])
    assert len(ev.parse_seeds(args.seeds)) == 20
    assert args.repeats == 2


# --------------------------------------------------------------------------- #
# 2. repeat aggregation
# --------------------------------------------------------------------------- #

def test_repeats_are_labelled_and_counted_but_not_extra_seeds(ev):
    records = []
    for repeat in range(2):
        for seed, frames in ((1000, 100), (1001, 200), (1002, 300)):
            record = _episodes([frames], seeds=[seed])[0]
            record["repeat"] = repeat
            records.append(record)
    summary = ev.summarise(records, bootstrap_iters=500)
    assert summary["episodes"] == 6
    assert summary["seed_count"] == 3
    assert summary["repeats"] == 2
    assert summary["seed_list"] == [1000, 1001, 1002]
    assert [row["repeats"] for row in summary["per_seed"]] == [2, 2, 2]
    assert summary["per_seed"][0]["frames"] == [100, 100]
    # the mean is over episodes, not over seeds: 6 episodes at 100/200/300
    assert summary["mean_frames"] == pytest.approx(200.0)
    assert summary["frames"] == [100, 200, 300, 100, 200, 300]


def test_repeat_zero_is_recorded_on_the_episode(ev):
    record = _episodes([123])[0]
    assert record["repeat"] == 0
    # the validator type-checks the field when present
    ev.validate_episode_record({
        "schema": ev.EPISODE_SCHEMA, "checkpoint": "c", "checkpoint_sha256": "0" * 64,
        "obs_spec_version": ev.obs_spec.OBS_SPEC_VERSION,
        "obs_spec_digest": ev.obs_spec.obs_spec_digest(), "seed": 1000,
        "suite": "S1N", "stage": 0, "difficulty": "normal", "max_frames": 10,
        "frames": 5, "deaths": 1.0, "score": 1, "end": "death_stop",
        "privilege": "deploy_pixel", "wall_s": 1.0, "engine_sha256": "0" * 64,
        "policy_input": "p", "not_a_gate": True, "repeat": 3,
    })


def test_error_episodes_are_not_measurements(ev):
    records = _episodes([100, 200])
    broken = _episodes([999])[0]
    broken["end"] = "error"
    records.append(broken)
    summary = ev.summarise(records, bootstrap_iters=200)
    assert summary["episodes"] == 3
    assert summary["measured"] == 2
    assert summary["mean_frames"] == pytest.approx(150.0)


# --------------------------------------------------------------------------- #
# 3. CI arithmetic
# --------------------------------------------------------------------------- #

def test_bootstrap_ci_is_reproducible_and_brackets_the_mean(ev):
    values = [1200, 1800, 900, 2600, 1500, 1750, 1100, 2050]
    first = ev.bootstrap_ci(values, iters=3000, seed=7)
    second = ev.bootstrap_ci(values, iters=3000, seed=7)
    assert first == second
    low, high = first
    assert low < float(np.mean(values)) < high
    assert low < high


def test_bootstrap_ci_equals_the_point_for_one_episode(ev):
    assert ev.bootstrap_ci([1500], iters=100) == (1500.0, 1500.0)


def test_bootstrap_ci_narrows_with_more_episodes(ev):
    rng = np.random.default_rng(0)
    small = rng.normal(2000, 400, 10)
    large = rng.normal(2000, 400, 200)
    lo_s, hi_s = ev.bootstrap_ci(small, iters=4000, seed=1)
    lo_l, hi_l = ev.bootstrap_ci(large, iters=4000, seed=1)
    assert (hi_l - lo_l) < (hi_s - lo_s) / 2.0


def test_frames_statistics_reports_sd_se_and_ci(ev):
    stats = ev.frames_statistics([100, 200, 300, 400, 500])
    assert stats["n"] == 5
    assert stats["mean"] == pytest.approx(300.0)
    assert stats["sd"] == pytest.approx(158.11388300841898)
    assert stats["se"] == pytest.approx(70.71067811865476)
    assert stats["ci_low"] < 300.0 < stats["ci_high"]
    assert stats["ci_level"] == ev.CI_LEVEL
    assert stats["ci_method"] == "percentile-bootstrap"


def test_one_episode_has_no_interval(ev):
    stats = ev.frames_statistics([7])
    assert stats["n"] == 1
    assert stats["sd"] is None and stats["se"] is None


def test_bad_bootstrap_arguments_fail_closed(ev):
    with pytest.raises(ev.EvalError):
        ev.bootstrap_ci([], iters=10)
    with pytest.raises(ev.EvalError):
        ev.bootstrap_ci([1, 2, 3], level=1.5)
    with pytest.raises(ev.EvalError):
        ev.bootstrap_ci([1, 2, 3], iters=0)


def test_summary_carries_the_criterion_and_an_interval(ev):
    summary = ev.summarise(_episodes(list(range(1000, 1020))), bootstrap_iters=500)
    assert summary["criterion"]["this_summary_has_ci"] is True
    assert summary["criterion"]["min_episodes_per_arm"] >= 20
    assert "不重叠" in summary["criterion"]["template"]
    assert summary["ci_low"] < summary["mean_frames"] < summary["ci_high"]
    assert summary["sd_frames"] > 0


# --------------------------------------------------------------------------- #
# 4. variance components
# --------------------------------------------------------------------------- #

def test_variance_components_recover_a_known_two_level_model(pw):
    rng = np.random.default_rng(11)
    sd_ckpt, sd_ep, n = 400.0, 200.0, 40
    groups = {}
    for index in range(12):
        offset = rng.normal(0.0, sd_ckpt)
        groups[f"ckpt{index}"] = rng.normal(2000 + offset, sd_ep, n)
    components = pw.variance_components(groups)
    assert components["k"] == 12
    assert components["n_total"] == 12 * n
    assert components["sigma2_episode"] == pytest.approx(sd_ep ** 2, rel=0.35)
    assert components["sigma2_checkpoint"] == pytest.approx(sd_ckpt ** 2, rel=0.6)
    # the whole point of the decomposition
    assert components["se_of_arm_mean_with_n"]["20"] > math.sqrt(
        components["sigma2_episode"] / 20)


def test_variance_components_truncate_a_negative_checkpoint_term(pw):
    rng = np.random.default_rng(3)
    groups = {f"c{index}": rng.normal(1000, 100, 30) for index in range(5)}
    components = pw.variance_components(groups)
    assert components["sigma2_checkpoint"] == 0.0
    assert components["sigma2_checkpoint_raw"] <= 0.0
    assert components["icc"] == pytest.approx(0.0)


def test_variance_components_need_two_checkpoints(pw):
    with pytest.raises(pw.PowerError):
        pw.variance_components({"only": [1.0, 2.0, 3.0]})


def test_variance_components_need_two_episodes_per_checkpoint(pw):
    with pytest.raises(pw.PowerError):
        pw.variance_components({"a": [1.0], "b": [2.0]})


# --------------------------------------------------------------------------- #
# 5. criterion template
# --------------------------------------------------------------------------- #

def _arm(label, mean, low, high, n=20, sd=100.0):
    return {"label": label, "mean": mean, "ci_low": low, "ci_high": high,
            "n": n, "sd": sd}


def test_disjoint_intervals_are_an_effect(pw):
    verdict = pw.compare(_arm("a", 2000, 1900, 2100), _arm("b", 1400, 1300, 1500))
    assert verdict["verdict"] == pw.VERDICT_EFFECT
    assert verdict["delta"] == pytest.approx(600.0)
    assert verdict["ci_overlap"] is False


def test_overlapping_intervals_have_no_verdict(pw):
    verdict = pw.compare(_arm("a", 2000, 1850, 2150), _arm("b", 1700, 1620, 1880))
    assert verdict["verdict"] == pw.VERDICT_OVERLAP
    assert verdict["ci_overlap"] is True
    assert verdict["overlap_interval"] == [1850.0, 1880.0]


def test_a_comparison_without_a_ci_is_refused(pw):
    verdict = pw.compare(_arm("a", 2000, None, None), _arm("b", 1400, 1300, 1500))
    assert verdict["verdict"] == pw.VERDICT_NO_CI
    assert verdict["delta"] is None
    assert any("no CI" in problem for problem in verdict["problems"])


def test_five_episode_arms_are_flagged_below_the_protocol_floor(pw):
    verdict = pw.compare(_arm("a", 2000, 1900, 2100, n=5),
                         _arm("b", 1400, 1300, 1500, n=5))
    assert len(verdict["warnings"]) == 2


def test_checkpoint_floor_can_veto_a_disjoint_pair(pw):
    verdict = pw.apply_checkpoint_floor(447.8, sigma2_ckpt=860243.0)
    assert verdict["resolvable"] is False
    assert verdict["resolution_limit"] > 447.8


def test_claim_cli_refuses_to_bypass_the_ci_requirement(pw, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        pw.main(["claim", "--a", f"x={tmp_path}", "--b", f"y={tmp_path}",
                 "--allow-missing-ci"])
    assert excinfo.value.code not in (0, None)


# --------------------------------------------------------------------------- #
# 6. power arithmetic
# --------------------------------------------------------------------------- #

def test_normal_helpers_round_trip(pw):
    assert pw.norm_cdf(0.0) == pytest.approx(0.5)
    assert pw.norm_cdf(1.959963985) == pytest.approx(0.975, abs=1e-6)
    assert pw.norm_ppf(0.975) == pytest.approx(1.959963985, abs=1e-6)
    assert pw.norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(pw.PowerError):
        pw.norm_ppf(1.5)


def test_alpha_005_power_080_needs_2_8_sigmas(pw):
    assert pw.z_critical(0.05) == pytest.approx(1.959963985, abs=1e-6)
    se = 100.0
    delta = se * (pw.z_critical(0.05) + pw.norm_ppf(0.8))
    assert delta == pytest.approx(280.1585, abs=1e-3)
    assert pw.achievable_power(delta, se, alpha=0.05) == pytest.approx(0.8, abs=1e-6)


def test_power_is_alpha_at_zero_delta_and_one_at_infinite_delta(pw):
    assert pw.achievable_power(0.0, 100.0, alpha=0.05) == pytest.approx(0.05, abs=1e-9)
    assert pw.achievable_power(1e6, 100.0, alpha=0.05) == pytest.approx(1.0, abs=1e-9)


def test_se_of_difference_matches_the_analytic_form(pw):
    se = pw.se_of_difference(1000.0, 40000.0, k=4, n=25)
    assert se == pytest.approx(math.sqrt(2 * (1000 / 4 + 40000 / (4 * 25))))
    # more checkpoints help more than more episodes, given a checkpoint term
    assert pw.se_of_difference(1000.0, 40000.0, k=8, n=25) < \
           pw.se_of_difference(1000.0, 40000.0, k=4, n=25)
    assert pw.se_of_difference(1000.0, 0.0, k=2, n=10 ** 9) == pytest.approx(
        math.sqrt(2 * 500.0), rel=1e-6)


def test_required_design_is_a_monotone_floor(pw):
    small = pw.required_design(0.0, 122414.0, 550.0)
    large = pw.required_design(0.0, 122414.0, 300.0)
    assert small["feasible"] and large["feasible"]
    assert large["cheapest_by_cost"]["episodes"] > small["cheapest_by_cost"]["episodes"]
    assert large["cheapest_by_cost"]["n"] > small["cheapest_by_cost"]["n"]


def test_a_huge_checkpoint_term_makes_the_design_infeasible(pw):
    report = pw.required_design(860243.0, 122414.0, 550.0, max_checkpoints=6)
    assert report["se_floor_with_many_checkpoints"] > report["se_required"]
    assert report["cheapest"] is None
    assert all(row["n"] is None for row in report["rows"])


def test_required_design_reports_the_checkpoint_hunger(pw):
    report = pw.required_design(200000.0, 122414.0, 550.0, max_checkpoints=16)
    table = {row["k"]: row for row in report["rows"]}
    # 2σ²_ckpt / k must fall under se_required² by itself: k > 2·200000/38534
    assert table[10]["n"] is None
    assert table[11]["n"] is not None  # one checkpoint can never reach the target
    assert table[16]["n"] < table[11]["n"]


# --------------------------------------------------------------------------- #
# 7. reading recorded arms
# --------------------------------------------------------------------------- #

def _write_arm(root: pathlib.Path, frames, *, seed_base=1000, label=None):
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, value in enumerate(frames):
        record = _episodes([value], seeds=[seed_base + index])[0]
        records.append(record)
    (root / "episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    summary = {"frames": [float(value) for value in frames],
               "episodes": len(frames),
               "mean_frames": float(np.mean(frames)) if frames else None,
               "label": label or root.name}
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def test_load_arm_reads_episodes_and_skips_failures(pw, tmp_path):
    root = _write_arm(tmp_path / "arm", [1000, 1200, 1400])
    with (root / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        broken = _episodes([999], seeds=[1009])[0]
        broken["end"] = "error"
        handle.write(json.dumps(broken) + "\n")
    arm = pw.load_arm(root)
    assert arm["n"] == 3
    assert arm["mean"] == pytest.approx(1200.0)
    assert arm["ci_low"] < arm["mean"] < arm["ci_high"]


def test_load_arm_without_episodes_or_ci_yields_no_interval(pw, tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    (root / "summary.json").write_text(
        json.dumps({"mean_frames": 1234.0, "episodes": 5}), encoding="utf-8")
    arm = pw.load_arm(root)
    assert arm["ci_low"] is None
    assert pw.compare(arm, arm)["verdict"] == pw.VERDICT_NO_CI


# --------------------------------------------------------------------------- #
# 8. CLI end to end
# --------------------------------------------------------------------------- #

def test_cli_noise_and_power_round_trip(pw, tmp_path, capsys):
    rng = np.random.default_rng(5)
    runs = []
    for index in range(4):
        offset = index * 350.0
        frames = (2000 + offset + rng.normal(0, 250, 20)).astype(int).tolist()
        _write_arm(tmp_path / f"arm{index}", frames)
        runs += ["--runs", f"arm{index}={tmp_path / f'arm{index}'}"]
    noise_json = tmp_path / "noise.json"
    assert pw.main(["noise", *runs, "--iters", "1000",
                    "--json", str(noise_json)]) == 0
    payload = json.loads(noise_json.read_text(encoding="utf-8"))
    components = payload["variance_components_all"]
    assert components["k"] == 4
    assert components["sigma2_episode"] > 0
    assert components["sigma2_checkpoint"] > 0

    # --same-config takes a comma-separated group (and is reported separately)
    grouped = tmp_path / "noise_grouped.json"
    assert pw.main(["noise", *runs, "--iters", "1000", "--same-config", "arm0,arm1",
                    "--json", str(grouped)]) == 0
    same = json.loads(grouped.read_text(encoding="utf-8"))[
        "variance_components_same_config"]
    assert same is not None and same["k"] == 2
    assert same["sigma2_episode"] > 0

    power_json = tmp_path / "power.json"
    assert pw.main(["power", "--noise", str(noise_json),
                    "--deltas", "300,550", "--json", str(power_json)]) == 0
    power = json.loads(power_json.read_text(encoding="utf-8"))
    assert len(power["designs"]) == 2
    assert power["designs"][1]["delta"] == 550.0


def test_cli_claim_exit_codes(pw, tmp_path):
    _write_arm(tmp_path / "high", [2000, 2100, 2200, 2300, 2400] * 4)
    _write_arm(tmp_path / "low", [900, 1000, 1100, 1200, 1300] * 4)
    assert pw.main(["claim", "--a", f"high={tmp_path / 'high'}",
                    "--b", f"low={tmp_path / 'low'}"]) == 0

    # condition 3: the same disjoint pair is refused once the measured
    # checkpoint variance says a retrain moves more than the delta
    assert pw.main(["claim", "--a", f"high={tmp_path / 'high'}",
                    "--b", f"low={tmp_path / 'low'}",
                    "--sigma2-checkpoint", "900000"]) == 2

    _write_arm(tmp_path / "same", [2000, 2100, 2200, 2300, 2400] * 4)
    assert pw.main(["claim", "--a", f"high={tmp_path / 'high'}",
                    "--b", f"same={tmp_path / 'same'}"]) == 2

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "summary.json").write_text(json.dumps({"mean_frames": 1500.0}),
                                       encoding="utf-8")
    assert pw.main(["claim", "--a", f"high={tmp_path / 'high'}",
                    "--b", f"bare={bare}"]) == 2


def test_cli_baseline_reports_the_checkpoint_level_interval(pw, tmp_path):
    _write_arm(tmp_path / "arm", [1000, 1200, 1400, 1600, 1800] * 4)
    path = tmp_path / "baseline.json"
    assert pw.main(["baseline", "--run", str(tmp_path / "arm"),
                    "--sigma2-checkpoint", "250000", "--iters", "1000",
                    "--json", str(path)]) == 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    interval = payload["checkpoint_level_interval"]
    assert interval["ci_low"] < payload["arm"]["mean"] < interval["ci_high"]
    assert interval["ci_high"] - interval["ci_low"] > 0
    assert "不重叠" in payload["criterion"]
