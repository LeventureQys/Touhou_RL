"""Self-check for the T1b G-3 diagnosis: only the G-3 cell may move."""

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE = pathlib.Path("Document/Update/Alpha/v3.6.0/experiments/T1b-g3-diagnosis")
TAGS = ("ALL", "DIAG-noNEG", "DIAG-CORE")
runs = {tag: json.loads((BASE / tag / "ablation.json").read_text(encoding="utf-8")) for tag in TAGS}

reference = {name: row["verdict"] for name, row in runs["ALL"]["verdicts"]["per_strategy"].items()}
reference_deltas = {(c["owner"], c["label"]): c["delta"] for c in runs["ALL"]["cells"]}

ok = True
for tag, report in runs.items():
    verdicts = {name: row["verdict"] for name, row in report["verdicts"]["per_strategy"].items()}
    same_verdicts = verdicts == reference
    deltas = {(c["owner"], c["label"]): c["delta"] for c in report["cells"]}
    diffs = [
        (key, reference_deltas[key], deltas[key])
        for key in reference_deltas
        if (reference_deltas[key] is None) != (deltas[key] is None)
        or (
            reference_deltas[key] is not None
            and deltas[key] is not None
            and abs(reference_deltas[key] - deltas[key]) > 1e-9
        )
    ]
    print(f"{tag:12s} verdicts identical={same_verdicts} cell-delta diffs={len(diffs)}")
    if not same_verdicts or diffs:
        ok = False
        print("   ", verdicts, diffs[:5])

print()
print(f"{'tag':12s} {'delta':>8} {'delta_ep':>9} {'p':>7} {'blocks':>6} {'G-3':>6}")
for tag, report in runs.items():
    gate = list(report["global_cells"].values())[0]
    print(
        f"{tag:12s} {gate['delta']:>+8.4f} {gate['delta_ep']:>+9.4f} {gate['permutation_p']:>7.4f} "
        f"{len(report['global_blocks']):>6} {str(report['verdicts']['gates']['G-3']['pass']):>6}"
    )

print()
print("SELF-CHECK PASS" if ok else "SELF-CHECK FAIL: a restricted union changed per-strategy cells")
sys.exit(0 if ok else 1)
