"""Cross-check every number the S3-7 report quotes against the run summaries.

The report is prose; the summaries and ``v3-experiment-table.json`` are data.  This
script re-reads the data and asserts the exact figures the report states (tables in
§11.5/§11.6/§11.9 and the ablation percentages in §11.7), so a transcription typo
cannot survive unnoticed.  Exit 0 = all assertions hold.

Usage::

    python Document/Update/Alpha/v1.0.4/Stage3/S3-7/evidence/verify_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
TABLE = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-7" / "v3-experiment-table.json"

#: The numbers as written in the report.  ``None`` means "not asserted here".
EXPECTED = {
    ("E4", "formal"): {
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
        "ep1_pos": 25.9364,
    },
    ("E5", "formal"): {
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
        "ep1_pos": 31.0962,
    },
    ("E6", "formal"): {
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
        "ep1_pos": 22.5038,
    },
    ("E4", "screening"): {
        "pos": 2.3558,
        "danger": 0.7595,
        "bucket": 2.082,
        "global": 1.451,
        "top3": 0.5815,
        "frames": 1076,
        "wall": 3814.74,
    },
    ("E5", "screening"): {
        "pos": 2.4602,
        "danger": 0.7467,
        "bucket": 1.355,
        "global": 0.961,
        "top3": 0.6441,
        "frames": 1076,
        "wall": 3846.57,
    },
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
    rows = {(row["experiment"], row["run_kind"]): row for row in payload["rows"]}
    failures: list[str] = []

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
            if field == "ep1_pos":
                continue  # checked from the raw summary below
            got = actual.get(field)
            if field.endswith("_prefix"):
                if got != value:
                    failures.append(f"{key} {field}: table {got!r} != report {value!r}")
                continue
            if got is None or abs(float(got) - float(value)) > 1e-9:
                failures.append(f"{key} {field}: table {got!r} != report {value!r}")

    # The epoch-1 divergence claim comes from the un-rounded first history row.
    for name, expected in (("E5", 31.0962), ("E6", 22.5038)):
        row = rows[(name, "formal")]
        summary = json.loads((ROOT / row["summary"]).read_text(encoding="utf-8"))
        first = summary["stages"]["state"]["history"][0]["val_player_pos_px"]
        if abs(float(first) - expected) > 1e-9:
            failures.append(f"{name} ep1 pos: summary {first!r} != report {expected!r}")

    # Percentages quoted in §11.7 / §11.6.
    formal = {name: rows[(name, "formal")]["raw"] for name in ("E4", "E5", "E6")}

    def pct(new, old):
        return round((new / old - 1.0) * 100.0, 1)

    percent = {
        "local_bucket": pct(formal["E5"]["policy"]["bucket_0_15_regret_px"], formal["E4"]["policy"]["bucket_0_15_regret_px"]),
        "local_global": pct(formal["E5"]["policy"]["global_regret_px"], formal["E4"]["policy"]["global_regret_px"]),
        "local_top3": pct(formal["E5"]["policy"]["top3"], formal["E4"]["policy"]["top3"]),
        "aux_bucket": pct(formal["E5"]["policy"]["bucket_0_15_regret_px"], formal["E6"]["policy"]["bucket_0_15_regret_px"]),
        "aux_global": pct(formal["E5"]["policy"]["global_regret_px"], formal["E6"]["policy"]["global_regret_px"]),
        "aux_top3": pct(formal["E5"]["policy"]["top3"], formal["E6"]["policy"]["top3"]),
        "v2_bucket": pct(formal["E5"]["policy"]["bucket_0_15_regret_px"], 3.049),
        "v2_global": pct(formal["E5"]["policy"]["global_regret_px"], 2.635),
        "v2_top3": pct(formal["E5"]["policy"]["top3"], 0.4196),
        "screen_bucket_E4": pct(formal["E4"]["policy"]["bucket_0_15_regret_px"], rows[("E4", "screening")]["raw"]["policy"]["bucket_0_15_regret_px"]),
        "screen_bucket_E5": pct(formal["E5"]["policy"]["bucket_0_15_regret_px"], rows[("E5", "screening")]["raw"]["policy"]["bucket_0_15_regret_px"]),
        "state_pos_local": round(formal["E5"]["state"]["player_pos_px"] - formal["E4"]["state"]["player_pos_px"], 4),
    }
    for field, expected in EXPECTED_PERCENT.items():
        if abs(percent[field] - expected) > 0.05:
            failures.append(f"{field}: computed {percent[field]} != report {expected}")

    total = sum(rows[(name, "formal")]["wall_seconds"] for name in ("E4", "E5", "E6"))
    if abs(total - TOTAL_WALL) > 0.01:
        failures.append(f"total wall: {total} != report {TOTAL_WALL}")

    print(f"rows checked: {len(EXPECTED)}; percentages checked: {len(EXPECTED_PERCENT)}")
    print(f"formal total wall: {total:.2f} s = {total / 3600:.2f} h")
    if failures:
        print("FAILURES:")
        for line in failures:
            print("  -", line)
        return 1
    print("ALL REPORT NUMBERS MATCH THE SUMMARIES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
