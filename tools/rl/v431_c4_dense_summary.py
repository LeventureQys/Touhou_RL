"""v4.3.1 recheck R24: the C4 gate PASSES in the dense regime at 10 s.

R20 rejected the sparse early regime (0/4 states admissible: 6-9 constant actions outlived
the whole 3 s segment). R23 then showed the dense regime is reachable for the state arm by
walking the tape with `step_frame` instead of `replay`. This runs the same C4 rule there:

    window  : seed 12, frame 8,664 (inside R8's dense window; a recorded block boundary)
    segment : 10 s = 300 decisions x 2 frames = 600 frames (top of the 3-10 s range)
    rule    : hold each of the 18 candidates for the whole segment
    admissible iff NO constant action survives

Result: **18/18 constants die inside the segment**. No survivor. The window is admissible.

The death offsets also give the exact constraint on segment length: the longest-surviving
constant reaches frame `max_offset`, so any segment longer than that is admissible. That
turns "pick a segment length" from a guess into a number.

Writes only Document/Update/Alpha/v4.3.1/evidence/R24/.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DIR = REPO / ".tmp_v431_seg001" / "full" / "raw" / "c4dense"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R24"
SEED = 12
FRAME = 8664
SEGMENT_FRAMES = 600
FPS = 60.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(DIR.glob(f"c4dense-seed{SEED}-f{FRAME}-cand*.json"))
    if not files:
        print(f"no C4-dense episodes under {DIR}", file=sys.stderr)
        return 2

    rows = []
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        cand = int(rec["policy_spec"].split(":", 1)[1])
        rows.append({
            "cand": cand,
            "terminal": rec["terminal"],
            "first_death_offset": rec["first_death_offset"],
            "frames_executed": rec["frames_executed"],
            "alive_120": rec["alive_120"],
            "alive_180": rec["alive_180"],
            "restore_hash_checks": (rec.get("restore") or {}).get("hash_checks"),
            "restore_hash_mismatches": (rec.get("restore") or {}).get("hash_mismatches"),
            "wall_s": rec["wall_s"],
            "error": rec["error"],
        })
    rows.sort(key=lambda r: r["cand"])

    survivors = [r["cand"] for r in rows if r["terminal"] is None]
    offsets = [r["first_death_offset"] for r in rows
               if r["first_death_offset"] is not None]
    max_off = max(offsets) if offsets else None
    min_admissible_frames = (max_off + 1) if max_off is not None else None

    out = {
        "schema": "v431-recheck-r24/1",
        "window": {"seed": SEED, "frame": FRAME, "dense": FRAME >= 8000},
        "segment": {"frames": SEGMENT_FRAMES,
                    "seconds": round(SEGMENT_FRAMES / FPS, 2),
                    "decisions": SEGMENT_FRAMES // 2},
        "rule": "hold each of the 18 candidates for the whole segment; admissible iff none survives",
        "counts": {
            "candidates": len(rows),
            "complete": sum(1 for r in rows if r["error"] is None),
            "survivors": len(survivors),
            "surviving_candidates": survivors,
            "errors": sum(1 for r in rows if r["error"] is not None),
        },
        "admissible": len(survivors) == 0,
        "death_offsets": {
            "min": min(offsets) if offsets else None,
            "median": statistics.median(offsets) if offsets else None,
            "max": max_off,
            "max_seconds": round(max_off / FPS, 2) if max_off is not None else None,
            "all": offsets,
        },
        "segment_length_constraint": {
            "min_admissible_frames": min_admissible_frames,
            "min_admissible_seconds": (round(min_admissible_frames / FPS, 2)
                                       if min_admissible_frames else None),
            "reading": ("any segment longer than the best constant's survival is admissible at "
                        "this window; the 10 s segment clears it by "
                        f"{SEGMENT_FRAMES - (max_off or 0)} frames"),
            "margin_frames": SEGMENT_FRAMES - (max_off or 0),
            "margin_fraction": round((SEGMENT_FRAMES - (max_off or 0)) / SEGMENT_FRAMES, 4),
        },
        "restore_gate": {
            "hash_checks_all": all((r["restore_hash_checks"] or 0) == 2165 for r in rows),
            "hash_mismatches_total": sum(r["restore_hash_mismatches"] or 0 for r in rows),
        },
        "cost": {
            "episodes": len(rows),
            "mean_wall_s": round(sum(r["wall_s"] for r in rows) / len(rows), 1),
            "total_wall_s": round(sum(r["wall_s"] for r in rows), 1),
        },
        "verdict": {
            "c4_passes_dense_10s": len(survivors) == 0,
            "contrast_with_sparse_3s": "R20 found 0/4 admissible at 3 s in the early regime",
            "consequence": ("arm A may proceed on this window at a segment length above "
                            f"{round(min_admissible_frames / FPS, 2)} s"),
        },
        "caveats": [
            "one window, one seed: admissibility is per (window, segment length), not a "
            "property of the dense regime in general",
            "the pixel arms (B/C) still cannot run here: obs-v3 needs the action-history "
            "buffer and `_pair_cand` refuses split-mask decisions (R22/R23)",
            "the best constant survives 520 frames, so a policy that simply holds it and then "
            "switches once would also clear 600 frames; the gate rules out 'one constant key' "
            "but not 'one constant key plus one change'",
        ],
        "rows": rows,
    }

    (OUT / "c4_dense.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    c = out["counts"]
    print(f"window seed {SEED} frame {FRAME}   segment {SEGMENT_FRAMES} frames "
          f"({out['segment']['seconds']} s)")
    print(f"candidates      : {c['candidates']}  complete {c['complete']}  errors {c['errors']}")
    print(f"survivors       : {c['survivors']} {survivors}")
    print(f"ADMISSIBLE      : {out['admissible']}")
    d = out["death_offsets"]
    print(f"death offsets   : min {d['min']}  median {d['median']}  max {d['max']} "
          f"({d['max_seconds']} s)")
    k = out["segment_length_constraint"]
    print(f"min admissible segment : {k['min_admissible_frames']} frames "
          f"({k['min_admissible_seconds']} s); 10 s clears by {k['margin_frames']} frames "
          f"({k['margin_fraction']:.1%})")
    print(f"hash gate       : all 2,165 checks per episode = {out['restore_gate']['hash_checks_all']}, "
          f"mismatches {out['restore_gate']['hash_mismatches_total']}")
    print(f"cost            : {out['cost']['episodes']} episodes, "
          f"mean {out['cost']['mean_wall_s']} s, total {out['cost']['total_wall_s']} s")
    print()
    print(f"{'cand':>4s} {'terminal':>13s} {'death_off':>10s} {'alive120':>9s} {'alive180':>9s}")
    print("-" * 52)
    for r in rows:
        print(f"{r['cand']:4d} {str(r['terminal']):>13s} {str(r['first_death_offset']):>10s} "
              f"{str(r['alive_120']):>9s} {str(r['alive_180']):>9s}")
    print()
    print(f"wrote: {OUT / 'c4_dense.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
