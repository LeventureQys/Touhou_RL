"""Build ``Stage3/S3-7/v3-experiment-table.json`` from the real v3 summaries.

One row per v3 run (formal 72+12 and the earlier 24+12 screening), read *only*
from ``runs/v104-v3-<E>-s7`` / ``runs/v104-E<E>-screen-s7``.  The table is
append-only: existing rows are preserved verbatim and a row whose key
(``experiment|run_kind|run_dir``) already exists is never rewritten.

Usage::

    python Document/Update/Alpha/v1.0.4/Stage3/S3-7/evidence/make_v3_table.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
RUNS = ROOT / "third_party" / "th08" / "build" / "rl" / "runs"
S37 = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-7"
TABLE = S37 / "v3-experiment-table.json"

SCHEMA = "v104-v3-experiment-table/1"
BUDGET_SECONDS = 60.0
#: Formal rows first, then the screening rows that preceded them, then the two
#: stability runs (append-only: new rows go to the end, old rows never move).
#:
#: ``v104-v3-E5-s17`` is the run whose directory says ``s17`` but whose trainer
#: summary records ``recipe.seed = config.seed = 7``; ``requested_seed`` keeps the
#: label visible instead of silently relabelling the run as seed 17.
#: ``v104-v3-E5-seed17`` is the **genuine** seed-17 run made after the ``--seed``
#: wiring fix, and its summary records 17 (that is the end-to-end proof).
SOURCES = (
    ("E4", "formal", "v104-v3-E4-s7", {}),
    ("E5", "formal", "v104-v3-E5-s7", {}),
    ("E6", "formal", "v104-v3-E6-s7", {}),
    ("E4", "screening", "v104-E4-screen-s7", {}),
    ("E5", "screening", "v104-E5-screen-s7", {}),
    (
        "E5",
        "stability",
        "v104-v3-E5-s17",
        {
            "requested_seed": 17,
            "seed_identity_mismatch": True,
            "seed_note": (
                "directory and log label say s17/seed17, but the trainer summary records"
                " recipe.seed=7 and config.seed=7, and its config block is identical to"
                " runs/v104-v3-E5-s7: this is a same-seed replicate, not a seed-17 run"
            ),
            "replicate_of": "third_party/th08/build/rl/runs/v104-v3-E5-s7",
        },
    ),
    (
        "E5",
        "stability",
        "v104-v3-E5-seed17",
        {
            "requested_seed": 17,
            "seed_identity_mismatch": False,
            "seed_note": (
                "genuine seed-17 run after the --seed wiring fix: the summary records"
                " recipe.seed=17 and config.seed=17, which is the end-to-end proof that"
                " the CLI flag now reaches VisionTrainConfig"
            ),
            "paired_with": "third_party/th08/build/rl/runs/v104-v3-E5-s7",
        },
    ),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def epoch_block(history: list[dict], *, cold_start: int = 2) -> dict:
    seconds = [float(row.get("seconds") or 0.0) for row in history]
    steady = seconds[cold_start:]
    mean = round(sum(steady) / len(steady), 2) if steady else None
    peak = round(max(steady), 2) if steady else None
    return {
        "epochs": len(seconds),
        "cold_start_seconds": seconds[:cold_start],
        "steady_mean_seconds": mean,
        "steady_max_seconds": peak,
        "all_seconds": seconds,
        "budget": {
            "limit_seconds": BUDGET_SECONDS,
            "exceeded": (peak is not None and peak > BUDGET_SECONDS),
            "overrun_fraction": round(peak / BUDGET_SECONDS - 1.0, 4) if peak else None,
            "note": (
                "Q6 waiver: the <=60 s v3 entry condition stays fail (waived_by=Q6);"
                " the overrun is recorded, never silently accepted"
            ),
        },
    }


def row_for(experiment: str, kind: str, run_name: str, extra: dict | None = None) -> dict:
    run_dir = RUNS / run_name
    summary_path = run_dir / "vision-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    state = summary["stages"]["state"]
    policy = summary["stages"]["policy"]
    state_best = state.get("best") or {}
    policy_best = policy.get("best") or {}
    buckets = policy_best.get("by_danger") or []
    bucket = next((item for item in buckets if item.get("bucket") == "0-15"), {})
    coverage = (summary.get("coverage") or {}).get("val") or {}
    aux = (summary.get("loss") or {}).get("aux_policy") or {}
    return {
        "experiment": experiment,
        "run_kind": kind,
        "route": summary["arch"],
        "depth": int(summary["depth"]),
        "local_enabled": bool((summary.get("input") or {}).get("local_enabled")),
        "policy_aux": {
            "heatmap": aux.get("heatmap"),
            "coord": aux.get("coord"),
            "danger": aux.get("danger"),
            "self": aux.get("self"),
        },
        "perception_aux": (summary.get("loss") or {}).get("aux_perception"),
        "heatmap_sigma": (summary.get("input") or {}).get("heatmap_sigma"),
        "seed": (summary.get("recipe") or {}).get("seed"),
        "parameters": summary.get("parameters"),
        "source_state_sha": (summary.get("source_state") or {}).get("sha256"),
        "source_state_checkpoint": (summary.get("source_state") or {}).get("checkpoint")
        or (summary.get("source_state") or {}).get("path"),
        "source_state_rule": (summary.get("source_state") or {}).get("selection_rule"),
        "teacher_fingerprint": (summary.get("teacher") or {}).get("fingerprint"),
        "val_index_sha256": (summary.get("data") or {}).get("val_index_sha256"),
        "epochs": {
            "state": (summary.get("recipe") or {}).get("epochs_state"),
            "policy": (summary.get("recipe") or {}).get("epochs_policy"),
        },
        "wall_seconds": (summary.get("hardware") or {}).get("elapsed_seconds"),
        "epoch_seconds": {
            "perception": epoch_block(state.get("history") or []),
            "policy": epoch_block(policy.get("history") or []),
        },
        "eligible": summary.get("acceptance_eligible"),
        "screening": (summary.get("recipe") or {}).get("screening"),
        "partial": summary.get("partial"),
        "status": "done" if not summary.get("partial") else "partial",
        "coverage": {
            "expected_frames": coverage.get("expected_frames"),
            "actual_frames": coverage.get("actual_frames"),
            "complete": coverage.get("complete"),
            "visited_index_sha256": coverage.get("visited_index_sha256"),
        },
        "raw": {
            "state": {
                "best_epoch": state_best.get("epoch"),
                "selection_rule": state_best.get("selection_rule"),
                "pass_epoch": state.get("pass_epoch"),
                "player_pos_px": state_best.get("val_player_pos_px"),
                "player_pos_p95_px": state_best.get("val_player_pos_p95_px"),
                "danger_corr_perframe_median": state_best.get("val_danger_corr_perframe_median"),
                "danger_corr_perframe_valid": state_best.get("val_danger_corr_perframe_valid"),
                "danger_corr": state_best.get("val_danger_corr"),
                "checkpoint": state_best.get("checkpoint"),
                "checkpoint_sha256": state_best.get("checkpoint_sha256"),
            },
            "policy": {
                "best_epoch": policy_best.get("epoch"),
                "selection_rule": policy_best.get("selection_rule"),
                "bucket_0_15_regret_px": policy_best.get("val_regret_px_bucket_0_15"),
                "bucket_0_15_frames": bucket.get("frames"),
                "bucket_0_15_top3": bucket.get("top3"),
                "global_regret_px": policy_best.get("val_regret_px"),
                "top1": policy_best.get("val_top1"),
                "top3": policy_best.get("val_top3"),
                "top3_tie": policy_best.get("val_top3_tie"),
                "near_optimal_05": policy_best.get("val_near_optimal_05"),
                "regret_p90_px": policy_best.get("val_regret_p90_px"),
                "by_danger": buckets,
                "checkpoint": policy_best.get("checkpoint"),
                "checkpoint_sha256": policy_best.get("checkpoint_sha256"),
            },
        },
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "summary": summary_path.relative_to(ROOT).as_posix(),
        "summary_sha256": digest(summary_path),
        **(extra or {}),
    }


def row_key(row: dict) -> str:
    return f"{row['experiment']}|{row.get('run_kind')}|{row.get('run_dir')}"


def main() -> int:
    existing: list[dict] = []
    header: dict = {}
    if TABLE.is_file():
        payload = json.loads(TABLE.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise SystemExit(f"{TABLE} has schema {payload.get('schema')!r}; refusing to touch it")
        existing = list(payload.get("rows") or [])
        header = {key: value for key, value in payload.items() if key != "rows"}
    seen = {row_key(row) for row in existing}
    appended = []
    for experiment, kind, run_name, extra in SOURCES:
        row = row_for(experiment, kind, run_name, extra)
        if row_key(row) in seen:
            continue
        appended.append(row)
        seen.add(row_key(row))
    rows = existing + appended
    payload = {
        "schema": SCHEMA,
        "dataset": "third_party/th08/build/rl/datasets/pilot-ud0059-v2",
        "val_index_sha256": "e5d4ebbe6d8de05793520a393fc1fdbea2b1e2c4db9ff7c1632908340c76655f",
        "val_frames": 7241,
        "read_only_inputs": [
            "third_party/th08/build/rl/runs/v104-v3-E4-s7",
            "third_party/th08/build/rl/runs/v104-v3-E5-s7",
            "third_party/th08/build/rl/runs/v104-v3-E6-s7",
            "third_party/th08/build/rl/runs/v104-E4-screen-s7",
            "third_party/th08/build/rl/runs/v104-E5-screen-s7",
            "third_party/th08/build/rl/runs/v104-v3-E5-s17",
            "third_party/th08/build/rl/runs/v104-v3-E5-seed17",
        ],
        "append_only": True,
        "generated_by": "Document/Update/Alpha/v1.0.4/Stage3/S3-7/evidence/make_v3_table.py",
        **{key: value for key, value in header.items() if key in ("first_written_at",)},
        "rows": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    TABLE.write_text(text + "\n", encoding="utf-8")
    print(f"{len(existing)} existing + {len(appended)} appended = {len(rows)} rows -> {TABLE.relative_to(ROOT).as_posix()}")
    for row in rows:
        raw = row["raw"]
        print(
            f"  {row['experiment']:<3} {row['run_kind']:<9} local={row['local_enabled']!s:<5}"
            f" pos={raw['state']['player_pos_px']:<8} danger={raw['state']['danger_corr_perframe_median']:<7}"
            f" bucket={raw['policy']['bucket_0_15_regret_px']:<6} global={raw['policy']['global_regret_px']:<6}"
            f" top3={raw['policy']['top3']:<7} wall={row['wall_seconds']} s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
