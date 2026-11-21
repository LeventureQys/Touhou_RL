"""v4.3.1 recheck R43: is the corrected baseline robust to the smoothing choice?

The original review demanded this explicitly (`ROUTE_REVIEW_2026-11-15.md:73`):

    正式重评应同时保留：训练集频率概率基线的 CE、恒猜训练集多数类的准确率、模型逐头 CE 与
    准确率、逐类表现，以及真实闭环成功率。缺失类别的平滑方案须预先固定。

My R1 computes both a plain train-frequency MLE and a Laplace-smoothed variant, but the report
quotes the MLE numbers without saying how sensitive they are. If a head has a class that never
appears in train, MLE gives it probability 0 and the CE becomes +inf (or, for an eval set that
also lacks it, trivially 0) -- which is exactly the trap the review warned about.

So this measures, from the stored R1 evidence:

  * how many classes each head is missing in train;
  * |MLE - Laplace| for every baseline, and the maximum;
  * whether that uncertainty is small relative to the effect the report claims (+0.067..+0.128
    nats for the 16x model).

Writes only Document/Update/Alpha/v4.3.1/evidence/R43/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
OUT = EV / "R43"
# the CE advantage of the 16x model over the corrected baseline, from R5 (paired bootstrap)
EFFECT_NATS = [0.0673, 0.0736, 0.1106, 0.1187, 0.1280]


def main() -> int:
    r1 = json.loads((EV / "R1" / "baseline_recheck.json").read_text(encoding="utf-8"))
    rows = []
    for name, b in r1["baselines"].items():
        tc = b.get("train_counts") or []
        mle = b.get("ce_train_mle")
        lap = b.get("ce_train_laplace")
        delta = abs(float(lap) - float(mle)) if (lap is not None and mle is not None) else None
        rows.append({
            "baseline": name,
            "classes": len(tc),
            "missing_in_train": sum(1 for x in tc if x == 0),
            "ce_train_mle": mle,
            "ce_train_laplace": lap,
            "abs_delta": round(delta, 8) if delta is not None else None,
            "degenerate": bool(tc) and sum(1 for x in tc if x > 0) == 1,
        })

    move = [r for r in rows if r["baseline"].endswith("__move")]
    degen = [r for r in rows if r["degenerate"]]
    max_delta_move = max((r["abs_delta"] for r in move), default=None)
    min_effect = min(EFFECT_NATS)
    out = {
        "schema": "v431-recheck-r43/1",
        "question": ("is the corrected baseline sensitive to the smoothing scheme the original "
                     "review demanded be pre-fixed?"),
        "requirement_quoted": ("缺失类别的平滑方案须预先固定 (ROUTE_REVIEW_2026-11-15.md:73)"),
        "rows": rows,
        "summary": {
            "move_heads_no_missing_classes": all(r["missing_in_train"] == 0 for r in move),
            "max_abs_delta_on_move": max_delta_move,
            "degenerate_heads": [r["baseline"] for r in degen],
            "smallest_claimed_effect_nats": min_effect,
            "effect_over_smoothing_uncertainty": (round(min_effect / max_delta_move, 1)
                                                  if max_delta_move else None),
        },
        "verdict": {
            "baseline_robust_to_smoothing": None,
            "reading": None,
        },
    }
    v = out["verdict"]
    ok = (out["summary"]["move_heads_no_missing_classes"]
          and max_delta_move is not None and max_delta_move < 1e-3)
    v["baseline_robust_to_smoothing"] = ok
    v["reading"] = (
        "the head the report quotes (`move`) has no class missing from train, so MLE is "
        f"well-defined, and Laplace changes it by at most {max_delta_move:.2e} nats -- about "
        f"{out['summary']['effect_over_smoothing_uncertainty']}x smaller than the smallest claimed "
        f"effect ({min_effect} nats). The only heads where the choice matters are `shoot`/`bomb`, "
        "which are degenerate (a single label value) and already reported as zero-signal "
        "instrument properties" if ok else
        "the baseline is NOT robust to the smoothing choice; the reported CE values need to be "
        "restated with a declared scheme")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smoothing.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'baseline':>22s} {'classes':>7s} {'missing':>7s} {'MLE':>10s} {'Laplace':>10s} "
          f"{'|delta|':>10s}")
    print("-" * 72)
    for r in rows:
        print(f"{r['baseline']:>22s} {r['classes']:7d} {r['missing_in_train']:7d} "
              f"{r['ce_train_mle']:10.6f} {r['ce_train_laplace']:10.6f} "
              f"{r['abs_delta']:10.2e}")
    print()
    print(f"move heads have no missing class: "
          f"{out['summary']['move_heads_no_missing_classes']}")
    print(f"max |MLE - Laplace| on move     : {max_delta_move:.2e} nats")
    print(f"smallest claimed effect (R5)    : {min_effect} nats "
          f"-> {out['summary']['effect_over_smoothing_uncertainty']}x larger")
    print(f"degenerate heads                : {out['summary']['degenerate_heads']}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'smoothing.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
