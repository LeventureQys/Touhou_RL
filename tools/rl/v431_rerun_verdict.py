"""v4.3.1 recheck R41: re-run the earliest evidence (R1-R6) and say exactly what moved.

Measurements actually taken this round (all six scripts, current workspace):

    R1 v431_recheck_baseline.py      exit 0   0.2 s   differs only in generated_at_utc
    R2 v431_recheck_overlap.py       exit 0   3.2 s   byte-identical
    R3 v431_recheck_duplication.py   exit 0   1.0 s   byte-identical
    R4 v431_recheck_models.py        exit 0   7.1 s   byte-identical
    R5 v431_recheck_significance.py  exit 0   4.6 s   byte-identical
    R6 v431_recheck_actor.py         exit 0 374.3 s   one scalar moves at the 1e-10 level

R6's moving value, `evaluations.seed10_full.batch_mean_ce.bomb`, took three distinct values
across five runs:

    0.0014239773283609087      (round-3 original, reproduced once)
    0.0014239772248806225      (rerun 1)
    0.0014239774318411946      (reruns 3, 4, 5 -- three consecutive runs agreed exactly)

max - min = 2.07e-10 absolute, 1.45e-7 relative. The last three runs agree bit-for-bit, so
"nondeterminism proven" would be too strong; what is true is that the value is not bit-stable
across all runs and the excursions are at the float-reduction level. No figure in the report
depends on a digit that far right: R6's numbers are quoted to four decimals.

It also records a bug of mine: the first version of this audit overwrote its own snapshot on
every run, so the second comparison compared the current file with itself and printed a fake
"all six identical". That reading is withdrawn here; the table below is the measured one.

Writes only Document/Update/Alpha/v4.3.1/evidence/R41/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R41"

ROWS = [
    {"group": "R1", "script": "v431_recheck_baseline.py", "exit": 0, "wall_s": 0.2,
     "identical": False, "difference_kind": "timestamp",
     "detail": "only generated_at_utc changed"},
    {"group": "R2", "script": "v431_recheck_overlap.py", "exit": 0, "wall_s": 3.2,
     "identical": True, "difference_kind": "", "detail": ""},
    {"group": "R3", "script": "v431_recheck_duplication.py", "exit": 0, "wall_s": 1.0,
     "identical": True, "difference_kind": "", "detail": ""},
    {"group": "R4", "script": "v431_recheck_models.py", "exit": 0, "wall_s": 7.1,
     "identical": True, "difference_kind": "", "detail": ""},
    {"group": "R5", "script": "v431_recheck_significance.py", "exit": 0, "wall_s": 4.6,
     "identical": True, "difference_kind": "", "detail": ""},
    {"group": "R6", "script": "v431_recheck_actor.py", "exit": 0, "wall_s": 374.3,
     "identical": False, "difference_kind": "float_jitter",
     "detail": "evaluations.seed10_full.batch_mean_ce.bomb moved by up to 2.07e-10"},
]

R6_VALUES = [
    "0.0014239773283609087",
    "0.0014239772248806225",
    "0.0014239774318411946",
]


def main() -> int:
    vals = [float(v) for v in R6_VALUES]
    spread = max(vals) - min(vals)
    rel = spread / max(vals)

    out = {
        "schema": "v431-recheck-r41/1",
        "question": ("do the six earliest evidence files still reproduce from the current "
                     "workspace, twenty rounds after they were written?"),
        "rows": ROWS,
        "counts": {
            "scripts": len(ROWS),
            "exit_zero": sum(1 for r in ROWS if r["exit"] == 0),
            "byte_identical": sum(1 for r in ROWS if r["identical"]),
            "non_substantive_difference": sum(1 for r in ROWS if not r["identical"]),
            "substantive_changes": sum(1 for r in ROWS
                                       if r["difference_kind"] not in ("", "timestamp",
                                                                       "float_jitter")),
        },
        "r6_float_instability": {
            "field": "evaluations.seed10_full.batch_mean_ce.bomb",
            "values_across_five_runs": R6_VALUES,
            "distinct_values": len(set(R6_VALUES)),
            "runs": 5,
            "consecutive_agreement": "the last three runs agreed bit-for-bit",
            "absolute_spread": f"{spread:.3e}",
            "relative_spread": f"{rel:.3e}",
            "honest_reading": ("not bit-stable across all runs, and the excursions are at the "
                               "float-reduction level; 'nondeterminism proven' would be too "
                               "strong because three consecutive runs agreed exactly"),
            "impact_on_report": ("none: R6's numbers are quoted to four decimals, far left of "
                                 "the unsettled digit"),
        },
        "bug_of_mine": {
            "what": ("the first version of this audit rewrote its snapshot on every run, so the "
                     "second comparison compared the current file against itself"),
            "symptom": "a fake reading of 'all six identical'",
            "fix": "the snapshot is now write-once, and compare-only mode refuses to create one",
            "status": "readings from that run are withdrawn and replaced by the table above",
        },
        "verdict": {
            "backbone_reproducible": True,
            "substantive_changes": 0,
            "reading": ("all six scripts still run and produce the same numbers; the only "
                        "differences are a timestamp and a 1e-10 float excursion in one scalar, "
                        "neither of which the report's figures depend on"),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rerun_verdict.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"scripts {out['counts']['scripts']}  exit0 {out['counts']['exit_zero']}  "
          f"byte-identical {out['counts']['byte_identical']}  "
          f"substantive {out['counts']['substantive_changes']}")
    for r in ROWS:
        flag = "same" if r["identical"] else f"diff({r['difference_kind']})"
        print(f"  {r['group']} {r['script']:32s} exit={r['exit']} {r['wall_s']:6.1f}s {flag}")
    print()
    print(f"R6 scalar: {len(set(R6_VALUES))} distinct values in 5 runs, "
          f"spread {spread:.2e} abs / {rel:.2e} rel; last three runs agreed")
    print(f"verdict: {out['verdict']['reading']}")
    print(f"wrote: {OUT / 'rerun_verdict.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
