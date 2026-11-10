"""TH08-Bench scoreboard from existing measured episode JSONL only.

Stage numbering (EXP-P0-077)
----------------------------
``results.jsonl`` rows may declare the **0-based** engine ``stage_index`` they ran
(``RESET <stage>`` / ``STATE.f1``; ``stage_index 0`` is the human-visible Stage 1).
Rows recorded before EXP-P0-077 carry no declared index, so the reader falls back
to the engine's own ``stage_observed`` and labels the cell ``stage_observed``.  A
row whose stage cannot be established is reported as ``unrecorded`` instead of
being folded into an assumed stage.

``--stage-index`` keeps only the rows of the requested 0-based stage, which is how
the pre-fix ``S1N`` rows (measured on ``stage_index 1`` = Stage 2) are kept out of
a Stage-1 table without editing or deleting them.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from touhou_rl.v400.bench.results import load_results, row_stage, summarize

DEFAULT_RESULTS = Path("Document/Update/Alpha/v4.0.0/bench/results.jsonl")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", action="store_true", required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage-index", type=int, action="append", default=None,
                        metavar="N",
                        help="only summarize rows whose 0-based engine stage_index is N "
                             "(repeatable); rows without a declared stage_index are matched "
                             "by the engine's own stage_observed")
    args = parser.parse_args(argv)
    try:
        rows = load_results(args.results)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if args.stage_index:
        wanted = set(args.stage_index)
        rows = [row for row in rows if row_stage(row)[0] in wanted]
    if not rows:
        parser.error("no measured episodes available; refusing empty scoreboard")
    report = summarize(rows)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
