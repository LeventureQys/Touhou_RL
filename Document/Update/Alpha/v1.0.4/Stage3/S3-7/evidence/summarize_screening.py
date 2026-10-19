"""Extract the screening comparison table from the two real summaries.

Reads ``runs/v104-E{4,5}-screen-s7/vision-summary.json`` (read-only) and prints /
writes the numbers the S3-7 report quotes: state best (position, danger median),
policy best (0-15 bucket regret, global regret, top3), coverage, epoch seconds.

Usage::

    python Document/Update/Alpha/v1.0.4/Stage3/S3-7/evidence/summarize_screening.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
RUNS = ROOT / "third_party" / "th08" / "build" / "rl" / "runs"
S37 = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-7"
#: S3-4's v2 reference numbers (same fixed validation split).
V2_REFERENCE = {
    "E2": {"bucket": 3.050, "global": 2.499, "top3": 0.4508, "sha": "32c28f9ba31d59f9"},
    "E3c": {"bucket": 3.049, "global": 2.635, "top3": 0.4196, "sha": "d29c7fa6cfcc2370"},
}


def epoch_seconds(rows: list[dict], cold_start: int = 2) -> dict:
    seconds = [float(row.get("seconds") or 0.0) for row in rows]
    steady = seconds[cold_start:]
    return {
        "all": seconds,
        "cold_start": seconds[:cold_start],
        "steady_mean": round(sum(steady) / len(steady), 2) if steady else None,
        "steady_max": round(max(steady), 2) if steady else None,
    }


def one(name: str) -> dict:
    run_dir = RUNS / f"v104-{name}-screen-s7"
    summary = json.loads((run_dir / "vision-summary.json").read_text(encoding="utf-8"))
    state = summary["stages"]["state"]
    policy = summary["stages"]["policy"]
    state_best = state.get("best") or {}
    policy_best = policy.get("best") or {}
    return {
        "experiment": name,
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "arch": summary.get("arch"),
        "depth": summary.get("depth"),
        "parameters": summary.get("parameters"),
        "screening": (summary.get("recipe") or {}).get("screening"),
        "acceptance_eligible": summary.get("acceptance_eligible"),
        "partial": summary.get("partial"),
        "coverage_val": (summary.get("coverage") or {}).get("val"),
        "state_epochs": epoch_seconds(state.get("history") or []),
        "policy_epochs": epoch_seconds(policy.get("history") or []),
        "local_enabled": (summary.get("input") or {}).get("local_enabled"),
        "heatmap_sigma": (summary.get("input") or {}).get("heatmap_sigma"),
        "state_best": {
            "epoch": state_best.get("epoch"),
            "selection_rule": state_best.get("selection_rule"),
            "val_player_pos_px": state_best.get("val_player_pos_px"),
            "val_player_pos_p95_px": state_best.get("val_player_pos_p95_px"),
            "val_danger_corr_perframe_median": state_best.get("val_danger_corr_perframe_median"),
            "val_danger_corr_perframe_valid": state_best.get("val_danger_corr_perframe_valid"),
            "val_danger_corr": state_best.get("val_danger_corr"),
            "checkpoint": state_best.get("checkpoint"),
            "checkpoint_sha256": state_best.get("checkpoint_sha256"),
            "state_pass_written": (state.get("pass_epoch") is not None),
            "pass_epoch": state.get("pass_epoch"),
        },
        "policy_best": {
            "epoch": policy_best.get("epoch"),
            "selection_rule": policy_best.get("selection_rule"),
            "val_regret_px_bucket_0_15": policy_best.get("val_regret_px_bucket_0_15"),
            "val_regret_px": policy_best.get("val_regret_px"),
            "val_top1": policy_best.get("val_top1"),
            "val_top3": policy_best.get("val_top3"),
            "val_top3_tie": policy_best.get("val_top3_tie"),
            "val_near_optimal_05": policy_best.get("val_near_optimal_05"),
            "val_regret_p90_px": policy_best.get("val_regret_p90_px"),
            "by_danger": policy_best.get("by_danger"),
            "checkpoint": policy_best.get("checkpoint"),
            "checkpoint_sha256": policy_best.get("checkpoint_sha256"),
        },
        "final_epoch_seconds": {
            "state": (state.get("history") or [{}])[-1].get("seconds"),
            "policy": (policy.get("history") or [{}])[-1].get("seconds"),
        },
    }


def main() -> int:
    payload = {
        "schema": "v104-s37-screening/1",
        "dataset": "third_party/th08/build/rl/datasets/pilot-ud0059-v2",
        "val_index_sha256": "e5d4ebbe6d8de05793520a393fc1fdbea2b1e2c4db9ff7c1632908340c76655f",
        "val_frames": 7241,
        "recipe": "24 perception + 12 policy, screening, seed 7, batch 96, full validation",
        "v2_reference": V2_REFERENCE,
        "runs": [],
    }
    for name in ("E4", "E5"):
        payload["runs"].append(one(name))
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    target = S37 / "evidence" / "screening-summary.json"
    target.write_text(text + "\n", encoding="utf-8")
    for row in payload["runs"]:
        state = row["state_best"]
        policy = row["policy_best"]
        print(
            f"{row['experiment']}: state ep{state['epoch']} pos={state['val_player_pos_px']} "
            f"p95={state['val_player_pos_p95_px']} danger={state['val_danger_corr_perframe_median']} "
            f"| policy ep{policy['epoch']} bucket={policy['val_regret_px_bucket_0_15']} "
            f"global={policy['val_regret_px']} top3={policy['val_top3']} "
            f"| state steady={row['state_epochs']['steady_mean']} s "
            f"policy steady={row['policy_epochs']['steady_mean']} s"
        )
    print(f"written {target.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
