"""v4.3.1 recheck R7f: is the teacher `margin` comparable across tapes?

R7e showed the `margin < 8` criterion never fires on seeds 10/11 (max tight slots = 0
over the whole tape) but fires on seed12. Before either keeping or dropping the
criterion, this checks whether the margin distributions are even on the same scale.
If they are not, the criterion measures the labelling instrument rather than the tape
and must not be used to choose hold-out segments.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOT = REPO / ".tmp_v420_data"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R7"
SEEDS = {"seed12": 12, "seed10": 10, "seed11": 11}


def main() -> None:
    out = {}
    for name, seed in SEEDS.items():
        shard = ROOT / "T04" / "dataset" / f"d1-s0d1-n3300-seed{seed}-shard"
        slots = [json.loads(ln) for ln in
                 (shard / "slots.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        margin = np.array([float(s["margin"]) for s in slots])
        scores = np.array([[float(x) for x in s["scores"]] for s in slots], dtype=np.float64)
        top2 = np.sort(scores, axis=1)[:, -2:]
        gap = top2[:, 1] - top2[:, 0]
        label_sources = {}
        for s in slots:
            label_sources[s.get("label_source")] = label_sources.get(s.get("label_source"), 0) + 1
        out[name] = {
            "slots": len(slots),
            "margin_min": round(float(margin.min()), 4),
            "margin_p1": round(float(np.percentile(margin, 1)), 4),
            "margin_p5": round(float(np.percentile(margin, 5)), 4),
            "margin_median": round(float(np.median(margin)), 4),
            "margin_max": round(float(margin.max()), 4),
            "margin_n_unique": int(len(np.unique(margin))),
            "n_below_8": int((margin < 8.0).sum()),
            "n_below_1": int((margin < 1.0).sum()),
            "top2_gap_min": round(float(gap.min()), 4),
            "top2_gap_median": round(float(np.median(gap)), 4),
            "top2_gap_n_below_8": int((gap < 8.0).sum()),
            "label_source_counts": label_sources,
            "score_min": round(float(scores.min()), 4),
            "score_max": round(float(scores.max()), 4),
        }
        r = out[name]
        print(f"== {name}: margin min {r['margin_min']:.3f} p1 {r['margin_p1']:.3f} "
              f"p5 {r['margin_p5']:.3f} median {r['margin_median']:.3f} max {r['margin_max']:.3f}")
        print(f"   unique margins {r['margin_n_unique']}  below8 {r['n_below_8']}  "
              f"top2gap below8 {r['top2_gap_n_below_8']}  "
              f"score range [{r['score_min']}, {r['score_max']}]")
        print(f"   label_sources {label_sources}")

    (OUT / "margin_comparability.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print()
    print(f"wrote: {OUT / 'margin_comparability.json'}")


if __name__ == "__main__":
    main()
