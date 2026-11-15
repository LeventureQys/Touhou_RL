"""Check v430 planning arithmetic, coverage, links and source integrity.

This never imports the game or learner. --write exclusively creates a new
plan_checks.json; existing reports and historical evidence are never changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote


HERE = Path(__file__).resolve().parent
VERSION = HERE.parent
ROOT = VERSION.parents[3]
REPORT = HERE / "plan_checks.json"
CHECKS: list[str] = []


def require(condition: bool, label: str) -> None:
    if not condition:
        raise ValueError(label)
    CHECKS.append(label)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_lower(successes: int, n: int, z: float = 1.959963984540054) -> float:
    p = successes / n
    return (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n)


def git(*args: str) -> dict:
    process = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, encoding="utf-8", errors="replace", check=False)
    require(process.returncode == 0, f"git {' '.join(args)} exit zero")
    return {"exit_code": process.returncode, "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip()}


def actor_parameter_estimate(model: dict, action: dict) -> int:
    """Arithmetic for the explicitly specified candidate, not a built network."""
    enc = model["encoder_candidate"]
    count = 0
    for name in ("global", "play", "hud"):
        stream = enc[name]
        for inputs, outputs, kernel, _stride in stream["convs"]:
            count += outputs * (inputs*kernel*kernel + 1) + 2*outputs
        flattened = stream["convs"][-1][1] * math.prod(stream["adaptive_pool_hw"])
        count += (flattened + 1) * stream["project_features"]
    history = enc["history"]
    count += (history["flattened_features"] + 1) * history["project_features"]
    hidden = model["hidden_size"]
    count += (enc["fusion_features"] + 1)*hidden + 2*hidden
    count += 6*hidden*hidden + 6*hidden
    count += (hidden+1)*sum(action["heads"].values())
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    require((ROOT / "AGENTS.md").is_file(), "repository root exists")
    spec_path = VERSION / "PLAN_SPEC.json"
    spec = read_json(spec_path)
    require(spec["schema"] == "v430-training-plan/2", "known plan schema")
    require(spec["scope"] == "planning_only" and not spec["execution_started"], "planning only")
    require(not spec["implementation_available"] and not spec["training_authorized_by_this_document"], "no implied implementation or authorization")
    require(spec["legacy_wp05_status"] == "paused", "WP05 remains paused")
    require(spec["capability_status"] == "not_run", "no new capability claimed")

    strategies = spec["strategies"]
    require([s["id"] for s in strategies] == [f"S{i:02}" for i in range(1, 9)], "eight unique strategies")
    require(sum(s["micro_train"] for s in strategies) == spec["datasets"]["MICRO-10"]["events"] == 10, "micro train allocation")
    require(sum(s["micro_dev"] for s in strategies) == spec["datasets"]["MICRO-DEV-30"]["events"] == 30, "micro dev allocation")
    power = spec["power"]
    require([s["id"] for s in power["subtypes"]] == [f"P{i}" for i in range(1, 6)], "five power behaviors")
    require(sum(s["events"] for s in power["subtypes"]) == spec["datasets"]["POWER-30"]["events"], "POWER-30 allocation")
    require(sum(spec["datasets"]["TACT-30"]["categories"].values()) == 30, "TACT-30 allocation")
    require(spec["datasets"]["MICRO-DEV-30"]["disjoint_training_ancestors"], "micro holdout lineage")
    require(not spec["datasets"]["CONFIRM"]["used_for_training_or_selection"], "confirmation isolation")
    require(not spec["datasets"]["BRANCH-32"]["unknown_is_death"], "unknown branch outcomes preserved")
    require(power["validate_own_path"] == ["approach", "pickup_or_wait", "exit", "next_threat"], "power own complete trajectory")

    action = spec["action"]
    require(math.prod(action["heads"].values()) == action["joint_count"] == 72, "joint action space arithmetic")
    require(not action["forced_shoot"] and not action["truth_action_mask"], "action privilege boundary")
    require(action["owner_exclusive"] and action["session_actor_loss_mask"] == 0, "session action ownership")
    obs = spec["observation"]["O2"]
    image_bytes = sum(math.prod(obs[k]) for k in ("global_rgb_chw", "play_gray_chw", "hud_gray_chw"))
    require(image_bytes == obs["image_bytes_per_decision"] == 474112, "observation byte arithmetic")
    require(obs["global_rgb_chw"][1:] == [240, 320], "global dimensions")
    require(obs["play_gray_chw"][1:] == [obs["play_crop_xywh"][3]//2, obs["play_crop_xywh"][2]//2], "play crop dimensions")
    require(obs["hud_gray_chw"][1:] == [obs["hud_crop_xywh"][3], obs["hud_crop_xywh"][2]], "HUD crop dimensions")
    for crop_name in ("play_crop_xywh", "hud_crop_xywh"):
        x, y, width, height = obs[crop_name]
        source_width, source_height = spec["observation"]["source_rgb_wh"]
        require(0 <= x < x+width <= source_width and 0 <= y < y+height <= source_height, f"{crop_name} inside source")
    model = spec["model"]
    enc = model["encoder_candidate"]
    require(math.prod(obs["executed_key_history_shape"]) + math.prod(obs["history_valid_shape"]) + 1 == enc["history"]["flattened_features"], "history feature arithmetic")
    require(sum(enc[k]["project_features"] for k in ("global", "play", "hud", "history")) == enc["fusion_features"], "fusion feature arithmetic")
    params = actor_parameter_estimate(model, action)
    lo, hi = model["parameter_range_millions"]
    require(lo*1e6 <= params <= hi*1e6, "candidate actor within development parameter range")

    ppo = spec["ppo"]
    require(ppo["hidden_size"] == model["hidden_size"] == enc["fusion_projection"], "memory dimensions agree")
    require(ppo["rollout_decisions_per_env"] % ppo["loss_unroll_decisions"] == 0, "rollout divisible by unroll")
    require(ppo["loss_unroll_decisions"] * ppo["sequences_per_minibatch"] == ppo["effective_minibatch_steps"], "minibatch arithmetic")
    require(ppo["num_envs"] * ppo["rollout_decisions_per_env"] % ppo["effective_minibatch_steps"] == 0, "minibatches partition rollout")
    require(0 < ppo["lambda_at_two_frames"] <= 1, "lambda valid")
    require(not ppo["off_policy_demos_in_ppo_ratio"] and not ppo["privileged_value_gradient_to_actor"], "PPO data and gradient isolation")
    require(ppo["duration_source_R1_R2"] == "shared external capture/controller timing only", "R1 R2 timing matched without engine truth")
    require(spec["modes"]["R1"]["critic"] == "pixel" and spec["modes"]["R2"]["critic"] == "same pixel critic as R1", "formal critic comparison matched")
    require(not spec["modes"]["R1"]["judge_access"], "R1 judge isolation")
    reward = ppo["main_reward"]
    require(not reward["nonfinal_death_terminal"] and reward["raw_pickup"] == reward["raw_kill"] == reward["per_step_survival"] == 0, "main reward semantics")
    for name in ("pretrain", "bc"):
        cfg = spec[name]
        require(cfg["check_updates"] == sorted(set(cfg["check_updates"])) and max(cfg["check_updates"]) == cfg["max_updates"], f"{name} check schedule")
    for name in ("early_frame_shares", "late_frame_shares", "aggregation_decision_shares", "offline_tactical_content_shares"):
        shares = spec["curriculum"][name]
        require(all(0 <= v <= 1 for v in shares.values()) and math.isclose(sum(shares.values()), 1), f"{name} sums to one")
    require(spec["curriculum"]["late_frame_shares"]["natural"] >= 0.7, "natural start curriculum exit")
    require(spec["curriculum"]["student_execution_teacher_beta"] == 0, "student autonomy statistics")
    generic = spec["observation"]["O_G"]
    require(not generic["fixed_game_specific_crops"] and generic["source_training_required"] and generic["separate_digest_and_model"], "generic transfer has separate source training contract")
    require(generic["actor_fusion_features"] == enc["global"]["project_features"] + enc["play"]["project_features"] + enc["history"]["project_features"], "generic fusion excludes game-specific HUD branch")

    micro, budget = spec["micro"], spec["budgets"]
    micro_cost = len(micro["arms"])*len(micro["train_seeds"])*micro["frames_per_arm_seed"]
    require(micro_cost == micro["total_new_frames"] == budget["micro_new_frames"], "micro training budget")
    require(len(ppo["train_seeds"])*ppo["cumulative_frame_tiers_per_seed"][0] == budget["main_new_frames"], "main training budget")
    allocated = budget["bootstrap_new_frames"] + budget["micro_new_frames"] + budget["main_new_frames"]
    require(allocated == budget["allocated_training_frames"] <= budget["training_frame_cap"], "allocated training fits cap")
    increment = len(ppo["train_seeds"])*(ppo["cumulative_frame_tiers_per_seed"][1] - ppo["cumulative_frame_tiers_per_seed"][0])
    require(increment == budget["future_increment_1m_to_5m_across_3_seeds"], "cumulative tier incremental cost")
    require(math.isclose(budget["new_persistent_gib"] * budget["stop_new_long_tasks_resource_fraction"], budget["stop_new_persistent_tasks_gib"]), "disk reserve threshold")
    require(not budget["unknown_prior_consumption_is_zero"] and not budget["automatic_budget_extension"] and not budget["automatic_asset_deletion"], "resource integrity")

    evaluation = spec["evaluation"]
    wave = evaluation["first_wave"]
    dev_episodes = wave["train_seeds"]*(wave["early_checkpoints"]*wave["episodes_each_early_checkpoint"]+wave["episodes_at_1m"])
    require(dev_episodes == wave["development_episodes"] == 195, "development episode count")
    require(wave["early_checkpoints"] + 1 == wave["checkpoints_per_seed"], "development checkpoint count")
    require(wave["checkpoint_interval_frames"]*wave["checkpoints_per_seed"] == ppo["cumulative_frame_tiers_per_seed"][0], "evaluation aligns with frame tier")
    max_eval_frames = (dev_episodes + wave["max_confirm_candidates"]*wave["confirm_episodes"])*wave["s1_frame_cap_candidate"]
    require(max_eval_frames == wave["s1_total_worst_case_frames"] < budget["evaluation_frame_cap"], "successful-policy evaluation fits budget")
    formal = evaluation["formal"]
    lower_bound = wilson_lower(formal["min_successes"], formal["episodes"])
    require(lower_bound >= formal["wilson_min_lower_bound"], "formal count meets Wilson threshold")
    require(formal["episodes"] == wave["confirm_episodes"] == spec["datasets"]["CONFIRM"]["episodes"], "formal denominators consistent")
    adapt = evaluation["adaptation"]
    adapt_cost = adapt["clean_families"]*len(adapt["trainable_arms"])*adapt["train_seeds"]*adapt["cumulative_frame_tiers"][-1]
    require(adapt_cost == adapt["total_at_20m"] == 360000000, "adaptation real interaction cost")
    require(all(v == "not_run_for_v430" for v in evaluation["new_capability_gates"].values()), "all capability gates unrun")

    phases = spec["phases"]
    require([p["id"] for p in phases] == [f"ST{i:02}" for i in range(10)], "ten ordered phases")
    packages = {f"W{i:02}" for i in range(13)}
    require({w for p in phases for w in p["work_packages"]} == packages, "all thirteen work packages covered")
    require(all(p["status"] == "not_run" for p in phases), "all phases unrun")

    documents = {name: (VERSION/name).read_text(encoding="utf-8-sig") for name in spec["documents"]}
    for name, content in documents.items():
        require("\ufffd" not in content, f"{name} valid decoded text")
        require(all(line == line.rstrip() for line in content.splitlines()), f"{name} no trailing whitespace")
        require(sum(line.startswith("```") for line in content.splitlines()) % 2 == 0, f"{name} fenced blocks balanced")
    for name in ("README.md", "TRAINING_PLAN.md", "DEVELOPMENT_GUIDE.md"):
        require(spec["revision"] in documents[name], f"{name} revision matches")
    for strategy in strategies:
        require(f'| {strategy["id"]} |' in documents["TRAINING_PLAN.md"], f'{strategy["id"]} documented')
    for phase in phases:
        require(f'| {phase["id"]} |' in documents["TRAINING_PLAN.md"], f'{phase["id"]} documented')
    cli = spec["future_cli"]
    require(not cli["implemented"], "future CLI clearly unimplemented")
    for command in cli["subcommands"]:
        require(f"python {cli['entry']} {command} " in documents["DEVELOPMENT_GUIDE.md"], f"CLI {command} documented")
    require(set(cli["config_owners"]) == packages and set(cli["test_owners"]) == packages, "all config and test owners listed")
    configs = [cfg for group in cli["config_owners"].values() for cfg in group]
    require(len(configs) == len(set(configs)), "config ownership unique")
    for cfg in configs:
        require(cfg in documents["EXECUTION_PLAN.md"], f"{cfg} in work package whitelist")
    for group in cli["test_owners"].values():
        for suffix in group:
            require(f"test_v430_{suffix}.py" in documents["DEVELOPMENT_GUIDE.md"], f"test {suffix} assigned")
    require("8M" in documents["TRAINING_PLAN.md"] and "5,866,665" in documents["TRAINING_PLAN.md"], "evaluation reserve documented")
    require("R2-AC" in documents["README.md"] and "相同像素critic" in documents["README.md"], "R2-AC separated in roadmap")

    link_count = 0
    for name, content in documents.items():
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", content):
            target = match.group(1).strip().strip("<>")
            if "://" in target or target.startswith("#"):
                continue
            target = unquote(target.split("#", 1)[0])
            resolved = (VERSION/target).resolve()
            require(resolved.exists() or (args.write and resolved == REPORT), f"local link {name}: {target}")
            link_count += 1

    old_report = HERE / "review_checks.json"
    baseline = read_json(old_report)
    for relative, expected_sha in baseline["source_sha256"].items():
        require(sha(ROOT/relative) == expected_sha, f"historical SHA unchanged: {relative}")
    for relative in ("touhou_rl/v400/gym/env.py", "touhou_rl/v400/gym/protocol.py", "touhou_rl/v400/gym/shm.py",
                     "touhou_rl/v400/gym/replay_starts.py", "touhou_rl/v400/gym/trajectory.py", "touhou_rl/v400/gym/trajectory_verify.py",
                     "touhou_rl/v400/teacher/engine.py", "touhou_rl/v400/teacher/decide.py", "touhou_rl/v400/teacher/search.py", "touhou_rl/v400/teacher/verify.py",
                     "touhou_rl/v400/student/obs_spec_v3.py", "touhou_rl/v400/student/model_v3.py",
                     "touhou_rl/v400/rl/env.py", "touhou_rl/v400/rl/policy.py", "touhou_rl/tactics.py"):
        require((ROOT/relative).is_file(), f"reuse source exists: {relative}")

    document_hashes = {name: sha(VERSION/name) for name in [*spec["documents"], "PLAN_SPEC.json"]}
    if REPORT.exists():
        previous = read_json(REPORT)
        require(previous["document_sha256"] == document_hashes, "documents match existing plan report")
        require(previous["historical_review_sha256"] == sha(old_report), "historical review report unchanged")
    head = git("rev-parse", "HEAD")
    tracked_diff = git("diff", "--name-only")
    staged_diff = git("diff", "--cached", "--name-only")
    status = git("status", "--short", "--untracked-files=normal")
    result = {
        "schema": "v430-plan-checks/1", "spec_revision": spec["revision"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "planning arithmetic, document coverage and immutable source checks only",
        "command": "python Document/Update/Alpha/v4.3.0/evidence/verify_plan.py" + (" --write" if args.write else ""),
        "verification_exit_code": 0, "checks_passed": len(CHECKS), "checks": CHECKS,
        "document_sha256": document_hashes, "checker_sha256": sha(Path(__file__)),
        "historical_review_sha256": sha(old_report), "historical_source_hashes_checked": len(baseline["source_sha256"]),
        "local_links_checked": link_count, "git_head": head["stdout"], "tracked_diff": tracked_diff,
        "staged_diff": staged_diff, "git_status": status,
        "arithmetic": {"candidate_actor_parameters_estimate": params,
                       "observation_image_bytes_per_decision": image_bytes,
                       "uint8_rollout_image_bytes": image_bytes*ppo["num_envs"]*ppo["rollout_decisions_per_env"],
                       "allocated_training_frames": allocated, "s1_evaluation_worst_case_frames": max_eval_frames,
                       "evaluation_frames_reserved_for_other_checks": budget["evaluation_frame_cap"]-max_eval_frames,
                       "formal_wilson_lower_at_min_successes": lower_bound,
                       "adaptation_target_frames_at_20m": adapt_cost,
                       "adaptation_real_hours_at_60hz": adapt_cost/(60*3600)},
        "capability_tests_run": 0, "training_runs_started": 0,
        "not_performed": ["business implementation", "network instantiation", "game execution", "training",
                          "Windows deployment", "formal capability confirmation", "asset-host verification"],
    }
    if args.write:
        with REPORT.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps({"status": "plan_checks_passed", "exit_code": 0, "checks_passed": len(CHECKS),
                      "documents": len(documents), "local_links_checked": link_count,
                      "historical_source_hashes_checked": len(baseline["source_sha256"]),
                      "report_created": args.write, "arithmetic": result["arithmetic"],
                      "capability_tests_run": 0, "training_runs_started": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
