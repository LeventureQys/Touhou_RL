"""Cross-check every number the S3-7 report quotes against the run summaries.

The report is prose; the summaries and ``v3-experiment-table.json`` are data.  This
script re-reads the data and asserts the exact figures the report states (tables in
§11.5/§11.6/§11.10/§11.12 and the percentages in §11.7), so a transcription typo
cannot survive unnoticed.  Exit 0 = all assertions hold.

Rows are keyed by the run directory's basename because there are now two
``stability`` rows (the mislabelled same-seed replicate and the genuine seed-17
run); ``(experiment, run_kind)`` is no longer unique.

Usage::

    python Document/Update/Alpha/v1.0.4/Stage3/S3-7/evidence/verify_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
TABLE = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-7" / "v3-experiment-table.json"

#: The numbers as written in the report, keyed by the run directory's basename.
EXPECTED = {
    "v104-v3-E4-s7": {
        "pos": 2.3314,
        "danger": 0.8006,
        "bucket": 1.899,
        "global": 1.28,
        "top3": 0.5959,
        "top1": 0.3506,
        "near": 0.6194,
        "p90": 4.188,
        "frames": 1076,
        "wall": 8490.61,
        "perception_steady_mean": 100.54,
        "perception_steady_max": 103.6,
        "policy_steady_mean": 103.66,
        "policy_steady_max": 103.9,
        "state_sha_prefix": "01bef067",
        "policy_sha_prefix": "b50577dd",
    },
    "v104-v3-E5-s7": {
        "pos": 2.4066,
        "danger": 0.8166,
        "bucket": 1.271,
        "global": 0.87,
        "top3": 0.6601,
        "top1": 0.4006,
        "near": 0.6904,
        "p90": 2.898,
        "frames": 1076,
        "wall": 8735.00,
        "perception_steady_mean": 103.73,
        "perception_steady_max": 104.7,
        "policy_steady_mean": 105.64,
        "policy_steady_max": 106.5,
        "state_sha_prefix": "00a2d284",
        "policy_sha_prefix": "74abde8a",
    },
    "v104-v3-E6-s7": {
        "pos": 2.1873,
        "danger": 0.8037,
        "bucket": 1.519,
        "global": 1.013,
        "top3": 0.6354,
        "top1": 0.3801,
        "near": 0.6687,
        "p90": 3.406,
        "frames": 1076,
        "wall": 8662.98,
        "perception_steady_mean": 102.57,
        "perception_steady_max": 104.3,
        "policy_steady_mean": 105.75,
        "policy_steady_max": 106.8,
        "state_sha_prefix": "ec3dda35",
        "policy_sha_prefix": "035c7d16",
    },
    "v104-E4-screen-s7": {
        "pos": 2.3558,
        "danger": 0.7595,
        "bucket": 2.082,
        "global": 1.451,
        "top3": 0.5815,
        "frames": 1076,
        "wall": 3814.74,
    },
    "v104-E5-screen-s7": {
        "pos": 2.4602,
        "danger": 0.7467,
        "bucket": 1.355,
        "global": 0.961,
        "top3": 0.6441,
        "frames": 1076,
        "wall": 3846.57,
    },
    # §11.10: the run whose directory says s17 but whose summary records seed 7.
    "v104-v3-E5-s17": {
        "pos": 2.1918,
        "danger": 0.8167,
        "bucket": 1.231,
        "global": 0.883,
        "top3": 0.6637,
        "top1": 0.3964,
        "near": 0.6879,
        "p90": 2.945,
        "frames": 1076,
        "wall": 8546.07,
        "perception_steady_mean": 101.31,
        "perception_steady_max": 105.6,
        "policy_steady_mean": 103.76,
        "policy_steady_max": 104.1,
        "state_sha_prefix": "e2ba07c2",
        "policy_sha_prefix": "c1d8f538",
    },
    # §11.12: the genuine seed-17 run made after the --seed wiring fix.
    "v104-v3-E5-seed17": {
        "pos": 2.2863,
        "danger": 0.8082,
        "bucket": 1.383,
        "global": 0.924,
        "top3": 0.652,
        "top1": 0.3845,
        "near": 0.679,
        "p90": 3.062,
        "frames": 1076,
        "wall": 8543.96,
        "perception_steady_mean": 101.28,
        "perception_steady_max": 113.4,
        "policy_steady_mean": 103.8,
        "policy_steady_max": 104.2,
        "state_sha_prefix": "0caac000",
        "policy_sha_prefix": "5fabcc9c",
    },
}
#: §11.10 identity: the replicate records seed 7 even though its label asks for 17.
REPLICATE_KEY = "v104-v3-E5-s17"
REPLICATE_EXPECT = {
    "seed": 7,
    "requested_seed": 17,
    "seed_identity_mismatch": True,
    "screening": False,
    "eligible": True,
    "source_state_rule": "joint-pass",
}
#: §11.12 identity: the genuine seed-17 run really records 17.
GENUINE_KEY = "v104-v3-E5-seed17"
GENUINE_EXPECT = {
    "seed": 17,
    "requested_seed": 17,
    "seed_identity_mismatch": False,
    "screening": False,
    "eligible": True,
    "source_state_rule": "joint-pass",
}
REPLICATE_SUMMARY = "third_party/th08/build/rl/runs/v104-v3-E5-s17/vision-summary.json"
REFERENCE_SUMMARY = "third_party/th08/build/rl/runs/v104-v3-E5-s7/vision-summary.json"
GENUINE_SUMMARY = "third_party/th08/build/rl/runs/v104-v3-E5-seed17/vision-summary.json"
#: §11.10.2 deltas between the frozen seed-7 run and the same-seed replicate.
EXPECTED_REPLICATE_DELTA = {
    "pos_abs": -0.2148,
    "pos_rel": -8.93,
    "danger_abs": 0.0001,
    "bucket_abs": -0.0400,
    "bucket_rel": -3.15,
    "global_rel": 1.49,
    "top3_rel": 0.55,
    "pooled_corr": (0.8722, 0.8752),
    "p95": (7.8724, 7.6771),
}
#: §11.12 comparisons against the frozen seed-7 run.
EXPECTED_SEED17 = {
    "pos_abs": -0.1203,
    "pos_rel": -5.00,
    "danger_abs": -0.0084,
    "bucket_abs": 0.1120,
    "bucket_rel": 8.81,
    "global_rel": 6.21,
    "top3_rel": -1.23,
    "danger_margin_seed7": 0.0166,
    "danger_margin_seed17": 0.0082,
    "a02_margin_seed7": 0.229,
    "a02_margin_seed17": 0.117,
    "joint_pass_epochs": (28, 30),
}
#: Report §11.7 / §11.6 percentages (percent, 1 decimal).
EXPECTED_PERCENT = {
    "local_bucket": -33.1,
    "local_global": -32.0,
    "local_top3": 10.8,
    "aux_bucket": -16.3,
    "aux_global": -14.1,
    "aux_top3": 3.9,
    "v2_bucket": -58.3,
    "v2_global": -67.0,
    "v2_top3": 57.3,
    "screen_bucket_E4": -8.8,
    "screen_bucket_E5": -6.2,
    "state_pos_local": 0.0752,
}
TOTAL_WALL = 25888.59


def main() -> int:
    payload = json.loads(TABLE.read_text(encoding="utf-8"))
    rows = {Path(row["run_dir"]).name: row for row in payload["rows"]}
    failures: list[str] = []
    if len(rows) != len(payload["rows"]):
        failures.append("two rows share a run directory; the table must not merge runs")

    for key, expected in EXPECTED.items():
        row = rows.get(key)
        if row is None:
            failures.append(f"{key}: missing row")
            continue
        raw = row["raw"]
        actual = {
            "pos": raw["state"]["player_pos_px"],
            "danger": raw["state"]["danger_corr_perframe_median"],
            "bucket": raw["policy"]["bucket_0_15_regret_px"],
            "global": raw["policy"]["global_regret_px"],
            "top3": raw["policy"]["top3"],
            "top1": raw["policy"]["top1"],
            "near": raw["policy"]["near_optimal_05"],
            "p90": raw["policy"]["regret_p90_px"],
            "frames": raw["policy"]["bucket_0_15_frames"],
            "wall": row["wall_seconds"],
            "perception_steady_mean": row["epoch_seconds"]["perception"]["steady_mean_seconds"],
            "perception_steady_max": row["epoch_seconds"]["perception"]["steady_max_seconds"],
            "policy_steady_mean": row["epoch_seconds"]["policy"]["steady_mean_seconds"],
            "policy_steady_max": row["epoch_seconds"]["policy"]["steady_max_seconds"],
            "state_sha_prefix": (raw["state"]["checkpoint_sha256"] or "")[:8],
            "policy_sha_prefix": (raw["policy"]["checkpoint_sha256"] or "")[:8],
        }
        for field, value in expected.items():
            got = actual.get(field)
            if field.endswith("_prefix"):
                if got != value:
                    failures.append(f"{key} {field}: table {got!r} != report {value!r}")
                continue
            if got is None or abs(float(got) - float(value)) > 1e-9:
                failures.append(f"{key} {field}: table {got!r} != report {value!r}")

    # §11.9(a): the epoch-1 divergence claim comes from the un-rounded first row.
    for name, expected in (
        ("v104-v3-E5-s7", 31.0962),
        ("v104-v3-E6-s7", 22.5038),
        ("v104-v3-E5-s17", 22.9099),
        ("v104-v3-E5-seed17", 23.1583),
    ):
        row = rows[name]
        summary = json.loads((ROOT / row["summary"]).read_text(encoding="utf-8"))
        first = summary["stages"]["state"]["history"][0]["val_player_pos_px"]
        if abs(float(first) - expected) > 1e-9:
            failures.append(f"{name} ep1 pos: summary {first!r} != report {expected!r}")

    # Percentages quoted in §11.7 / §11.6.
    formal = {
        name: rows[name]["raw"]
        for name in ("v104-v3-E4-s7", "v104-v3-E5-s7", "v104-v3-E6-s7")
    }

    def pct(new, old):
        return round((new / old - 1.0) * 100.0, 1)

    percent = {
        "local_bucket": pct(formal["v104-v3-E5-s7"]["policy"]["bucket_0_15_regret_px"], formal["v104-v3-E4-s7"]["policy"]["bucket_0_15_regret_px"]),
        "local_global": pct(formal["v104-v3-E5-s7"]["policy"]["global_regret_px"], formal["v104-v3-E4-s7"]["policy"]["global_regret_px"]),
        "local_top3": pct(formal["v104-v3-E5-s7"]["policy"]["top3"], formal["v104-v3-E4-s7"]["policy"]["top3"]),
        "aux_bucket": pct(formal["v104-v3-E5-s7"]["policy"]["bucket_0_15_regret_px"], formal["v104-v3-E6-s7"]["policy"]["bucket_0_15_regret_px"]),
        "aux_global": pct(formal["v104-v3-E5-s7"]["policy"]["global_regret_px"], formal["v104-v3-E6-s7"]["policy"]["global_regret_px"]),
        "aux_top3": pct(formal["v104-v3-E5-s7"]["policy"]["top3"], formal["v104-v3-E6-s7"]["policy"]["top3"]),
        "v2_bucket": pct(formal["v104-v3-E5-s7"]["policy"]["bucket_0_15_regret_px"], 3.049),
        "v2_global": pct(formal["v104-v3-E5-s7"]["policy"]["global_regret_px"], 2.635),
        "v2_top3": pct(formal["v104-v3-E5-s7"]["policy"]["top3"], 0.4196),
        "screen_bucket_E4": pct(formal["v104-v3-E4-s7"]["policy"]["bucket_0_15_regret_px"], rows["v104-E4-screen-s7"]["raw"]["policy"]["bucket_0_15_regret_px"]),
        "screen_bucket_E5": pct(formal["v104-v3-E5-s7"]["policy"]["bucket_0_15_regret_px"], rows["v104-E5-screen-s7"]["raw"]["policy"]["bucket_0_15_regret_px"]),
        "state_pos_local": round(formal["v104-v3-E5-s7"]["state"]["player_pos_px"] - formal["v104-v3-E4-s7"]["state"]["player_pos_px"], 4),
    }
    for field, expected in EXPECTED_PERCENT.items():
        if abs(percent[field] - expected) > 0.05:
            failures.append(f"{field}: computed {percent[field]} != report {expected}")

    total = sum(
        rows[name]["wall_seconds"] for name in ("v104-v3-E4-s7", "v104-v3-E5-s7", "v104-v3-E6-s7")
    )
    if abs(total - TOTAL_WALL) > 0.01:
        failures.append(f"total wall: {total} != report {TOTAL_WALL}")

    # §11.10: identity of the s17-labelled run and its deltas vs the frozen seed-7 run.
    replicate = rows[REPLICATE_KEY]
    for field, expected in REPLICATE_EXPECT.items():
        if replicate.get(field) != expected:
            failures.append(f"replicate {field}: table {replicate.get(field)!r} != report {expected!r}")
    replicate_summary = json.loads((ROOT / REPLICATE_SUMMARY).read_text(encoding="utf-8"))
    reference_summary = json.loads((ROOT / REFERENCE_SUMMARY).read_text(encoding="utf-8"))
    if replicate_summary["recipe"]["seed"] != 7 or replicate_summary["config"]["seed"] != 7:
        failures.append(
            "replicate summary no longer records seed 7: "
            f"{replicate_summary['recipe']['seed']}/{replicate_summary['config']['seed']}"
        )
    differing = [
        key
        for key in set(replicate_summary["config"]) | set(reference_summary["config"])
        if replicate_summary["config"].get(key) != reference_summary["config"].get(key)
    ]
    if differing:
        failures.append(f"replicate config differs from E5-s7 in {sorted(differing)}")
    rep_state = replicate_summary["stages"]["state"]["best"]
    ref_state = reference_summary["stages"]["state"]["best"]
    rep_policy = replicate_summary["stages"]["policy"]["best"]
    ref_policy = reference_summary["stages"]["policy"]["best"]
    delta = {
        "pos_abs": round(rep_state["val_player_pos_px"] - ref_state["val_player_pos_px"], 4),
        "pos_rel": round((rep_state["val_player_pos_px"] / ref_state["val_player_pos_px"] - 1) * 100, 2),
        "danger_abs": round(
            rep_state["val_danger_corr_perframe_median"] - ref_state["val_danger_corr_perframe_median"], 4
        ),
        "bucket_abs": round(
            rep_policy["val_regret_px_bucket_0_15"] - ref_policy["val_regret_px_bucket_0_15"], 4
        ),
        "bucket_rel": round(
            (rep_policy["val_regret_px_bucket_0_15"] / ref_policy["val_regret_px_bucket_0_15"] - 1) * 100, 2
        ),
        "global_rel": round((rep_policy["val_regret_px"] / ref_policy["val_regret_px"] - 1) * 100, 2),
        "top3_rel": round((rep_policy["val_top3"] / ref_policy["val_top3"] - 1) * 100, 2),
        "pooled_corr": (ref_state["val_danger_corr"], rep_state["val_danger_corr"]),
        "p95": (ref_state["val_player_pos_p95_px"], rep_state["val_player_pos_p95_px"]),
    }
    for field, expected in EXPECTED_REPLICATE_DELTA.items():
        got = delta[field]
        if isinstance(expected, tuple):
            if any(abs(float(a) - float(b)) > 1e-4 for a, b in zip(got, expected)):
                failures.append(f"replicate {field}: {got} != report {expected}")
        elif abs(float(got) - float(expected)) > 0.005:
            failures.append(f"replicate {field}: {got} != report {expected}")
    if replicate_summary["stages"]["state"].get("pass_epoch") != 28:
        failures.append("replicate joint-pass epoch is not 28")

    # §11.12: the genuine seed-17 run, its identity and the margins the report quotes.
    genuine = rows[GENUINE_KEY]
    for field, expected in GENUINE_EXPECT.items():
        if genuine.get(field) != expected:
            failures.append(f"seed17 {field}: table {genuine.get(field)!r} != report {expected!r}")
    genuine_summary = json.loads((ROOT / GENUINE_SUMMARY).read_text(encoding="utf-8"))
    if genuine_summary["recipe"]["seed"] != 17 or genuine_summary["config"]["seed"] != 17:
        failures.append(
            "seed-17 summary does not record 17: "
            f"{genuine_summary['recipe']['seed']}/{genuine_summary['config']['seed']}"
        )
    seed17_state = genuine_summary["stages"]["state"]["best"]
    seed17_policy = genuine_summary["stages"]["policy"]["best"]
    seed17 = {
        "pos_abs": round(seed17_state["val_player_pos_px"] - ref_state["val_player_pos_px"], 4),
        "pos_rel": round((seed17_state["val_player_pos_px"] / ref_state["val_player_pos_px"] - 1) * 100, 2),
        "danger_abs": round(seed17_state["val_danger_corr_perframe_median"] - ref_state["val_danger_corr_perframe_median"], 4),
        "bucket_abs": round(seed17_policy["val_regret_px_bucket_0_15"] - ref_policy["val_regret_px_bucket_0_15"], 4),
        "bucket_rel": round((seed17_policy["val_regret_px_bucket_0_15"] / ref_policy["val_regret_px_bucket_0_15"] - 1) * 100, 2),
        "global_rel": round((seed17_policy["val_regret_px"] / ref_policy["val_regret_px"] - 1) * 100, 2),
        "top3_rel": round((seed17_policy["val_top3"] / ref_policy["val_top3"] - 1) * 100, 2),
        "danger_margin_seed7": round(ref_state["val_danger_corr_perframe_median"] - 0.8, 4),
        "danger_margin_seed17": round(seed17_state["val_danger_corr_perframe_median"] - 0.8, 4),
        "a02_margin_seed7": round(1.5 - ref_policy["val_regret_px_bucket_0_15"], 4),
        "a02_margin_seed17": round(1.5 - seed17_policy["val_regret_px_bucket_0_15"], 4),
        "joint_pass_epochs": (
            int(reference_summary["stages"]["state"]["pass_epoch"]),
            int(genuine_summary["stages"]["state"]["pass_epoch"]),
        ),
    }
    for field, expected in EXPECTED_SEED17.items():
        got = seed17[field]
        if isinstance(expected, tuple):
            if tuple(int(a) for a in got) != tuple(int(b) for b in expected):
                failures.append(f"seed17 {field}: {got} != report {expected}")
        elif field.startswith(("danger_margin", "a02_margin")):
            if abs(float(got) - float(expected)) > 1e-4:
                failures.append(f"seed17 {field}: {got} != report {expected}")
        elif abs(float(got) - float(expected)) > 0.005:
            failures.append(f"seed17 {field}: {got} != report {expected}")

    print(f"rows checked: {len(EXPECTED)}; percentages checked: {len(EXPECTED_PERCENT)}")
    print(f"formal total wall: {total:.2f} s = {total / 3600:.2f} h")
    print(
        f"replicate identity: seed={replicate['seed']} requested={replicate['requested_seed']}"
        f" mismatch={replicate['seed_identity_mismatch']}; config identical to E5-s7"
    )
    print(f"replicate deltas: {json.dumps(delta, ensure_ascii=False)}")
    print(
        f"genuine seed-17 identity: seed={genuine['seed']} requested={genuine['requested_seed']}"
        f" mismatch={genuine['seed_identity_mismatch']}"
    )
    print(f"seed17 vs seed7: {json.dumps(seed17, ensure_ascii=False)}")
    if failures:
        print("FAILURES:")
        for line in failures:
            print("  -", line)
        return 1
    print("ALL REPORT NUMBERS MATCH THE SUMMARIES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
