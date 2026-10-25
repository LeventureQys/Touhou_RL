import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

report = json.loads(
    pathlib.Path("Document/Update/Alpha/v3.6.0/evidence/region-stratified-fixedwidth.json").read_text(
        encoding="utf-8"
    )
)
detail = report["bin_detail_full_data"]
print(f"{'bin':>3} {'lo':>6} {'hi':>6} {'n':>6} {'pos':>5} {'auc':>7} {'base_auc':>9} kept")
for row in detail["rows"]:
    auc = row["auc"] if row["auc"] is not None else float("nan")
    base = row["baseline_auc"] if row["baseline_auc"] is not None else float("nan")
    print(
        f"{row['bin']:>3} {row['lo']:>6.1f} {row['hi']:>6.1f} {row['n']:>6} {row['positives']:>5} "
        f"{auc:>7.3f} {base:>9.3f} {row['kept']}"
    )
print("dropped positives", detail["dropped_positives"], "/", detail["positives_total"])
print()
print("verdict:", report["verdict"], "-", report["reason"])
print(
    "LOEO stratified region AUC",
    round(report["stratified"]["auc"], 4),
    "baseline",
    round(report["stratified"]["baseline_auc"], 4),
    "p",
    round(report["permutation"]["p"], 4),
)
