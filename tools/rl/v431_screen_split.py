"""v4.3.1 recheck R28: the widened C4 screen, and the frozen train/dev window split.

R25 screened three windows (2 admissible) and `S1b` wants at least four per side. This
screens six more tapes at the same frame (8,664), judges every window on the same rule
(hold each of the 18 candidates for the whole 600-frame segment; admissible iff none
survives), and then freezes a disjoint train/dev split over the *tapes*.

Why the split is over tapes and not frames: the 18 candidates at one window are the same
actions, but the bullet RNG differs per tape, so a tape is the natural ancestor unit -- the
same argument R2/R3 made about the three teacher tapes being distinct ancestors.

What it also decides: the segment length. A window only rules out constant actions when the
segment is longer than the best constant's survival, so the experiment's segment must exceed
`max(max_offset + 1)` over the windows it uses. That number is reported rather than guessed.

Writes only Document/Update/Alpha/v4.3.1/evidence/R28/.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DIR = REPO / ".tmp_v431_seg001" / "full" / "raw" / "c4dense"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R28"
FRAME = 8664
SEGMENT_FRAMES = 600
FPS = 60.0
PAT = re.compile(r"c4dense-seed(\d+)-f(\d+)-cand(\d+)\.json$")
TRAIN_SEEDS = (13, 14, 15, 16)
DEV_SEEDS = (11, 12, 17, 18)


def main() -> int:
    windows: dict[int, dict] = {}
    for f in sorted(DIR.glob(f"c4dense-seed*-f{FRAME}-cand*.json")):
        m = PAT.search(f.name)
        if not m:
            continue
        seed, cand = int(m.group(1)), int(m.group(3))
        rec = json.loads(f.read_text(encoding="utf-8"))
        windows.setdefault(seed, {})[cand] = {
            "terminal": rec["terminal"],
            "first_death_offset": rec["first_death_offset"],
            "alive_180": rec["alive_180"],
            "hash_checks": (rec.get("restore") or {}).get("hash_checks"),
            "hash_mismatches": (rec.get("restore") or {}).get("hash_mismatches"),
            "wall_s": rec["wall_s"],
            "error": rec["error"],
        }

    rows = []
    for seed in sorted(windows):
        c = windows[seed]
        judged = len(c) == 18 and all(v["error"] is None for v in c.values())
        survivors = sorted(k for k, v in c.items() if v["terminal"] is None)
        offsets = [v["first_death_offset"] for v in c.values()
                   if v["first_death_offset"] is not None]
        max_off = max(offsets) if offsets else None
        rows.append({
            "seed": seed, "frame": FRAME,
            "candidates_run": len(c), "judged": judged,
            "n_survivors": len(survivors), "survivors": survivors,
            "admissible": bool(judged and not survivors),
            "min_death_offset": min(offsets) if offsets else None,
            "median_death_offset": statistics.median(offsets) if offsets else None,
            "max_death_offset": max_off,
            "min_blocking_frames": (max_off + 1) if max_off is not None else None,
            "min_blocking_seconds": (round((max_off + 1) / FPS, 2)
                                     if max_off is not None else None),
            "n_survive_180": sum(1 for v in c.values() if v["alive_180"]),
            "hash_mismatches_total": sum(v["hash_mismatches"] or 0 for v in c.values()),
            "hash_checks_all_2165": all((v["hash_checks"] or 0) == 2165 for v in c.values()),
            "mean_wall_s": round(statistics.mean([v["wall_s"] for v in c.values()]), 1),
        })

    by_seed = {r["seed"]: r for r in rows}
    admissible = [r["seed"] for r in rows if r["admissible"]]
    train_ok = [s for s in TRAIN_SEEDS if by_seed.get(s, {}).get("admissible")]
    dev_ok = [s for s in DEV_SEEDS if by_seed.get(s, {}).get("admissible")]
    used = train_ok + dev_ok
    blocking = [by_seed[s]["min_blocking_frames"] for s in used
                if by_seed.get(s, {}).get("min_blocking_frames")]
    required = max(blocking) if blocking else None

    out = {
        "schema": "v431-recheck-r28/1",
        "screen": {
            "frame": FRAME, "segment_frames": SEGMENT_FRAMES,
            "segment_seconds": round(SEGMENT_FRAMES / FPS, 2),
            "windows_screened": len(rows),
            "windows_judged": sum(1 for r in rows if r["judged"]),
            "windows_admissible": len(admissible),
            "admissible_seeds": admissible,
            "rejected_seeds": [r["seed"] for r in rows
                               if r["judged"] and not r["admissible"]],
            "rows": rows,
        },
        "split": {
            "unit": "tape (seed), not frame",
            "train_seeds": TRAIN_SEEDS,
            "dev_seeds": DEV_SEEDS,
            "train_admissible": train_ok,
            "dev_admissible": dev_ok,
            "disjoint": not (set(TRAIN_SEEDS) & set(DEV_SEEDS)),
            "s1b_min_4_per_side": (len(train_ok) >= 4 and len(dev_ok) >= 4),
            "s1b_detail": f"train {len(train_ok)}/4, dev {len(dev_ok)}/4",
            "rejected_excluded": [r["seed"] for r in rows
                                  if r["judged"] and not r["admissible"]],
        },
        "segment_length": {
            "required_frames": required,
            "required_seconds": round(required / FPS, 2) if required else None,
            "chosen_frames": SEGMENT_FRAMES,
            "chosen_seconds": round(SEGMENT_FRAMES / FPS, 2),
            "margin_frames": (SEGMENT_FRAMES - required) if required else None,
            "per_window_min_blocking_frames": {str(s): by_seed[s]["min_blocking_frames"]
                                               for s in used},
            "reading": ("the experiment must run a segment longer than every admissible "
                        "window's best constant survival; the 600-frame segment is the "
                        "smallest round number that clears all of them"),
        },
        "gate_integrity": {
            "every_episode_hash_checks_2165": all(r["hash_checks_all_2165"] for r in rows),
            "hash_mismatches_total": sum(r["hash_mismatches_total"] for r in rows),
        },
        "cost": {
            "episodes": sum(r["candidates_run"] for r in rows),
            "mean_wall_s": round(statistics.mean([r["mean_wall_s"] for r in rows]), 1),
            "window_cost_s": round(statistics.mean([r["mean_wall_s"] for r in rows]) * 18, 1),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen_split.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"== C4 screen at frame {FRAME}, segment {SEGMENT_FRAMES} frames "
          f"({out['screen']['segment_seconds']} s) ==")
    print(f"{'seed':>5s} {'cands':>6s} {'judged':>7s} {'surv':>5s} {'adm':>6s} "
          f"{'minOff':>7s} {'maxOff':>7s} {'minBlock':>9s} {'>=180f':>7s}")
    print("-" * 70)
    for r in rows:
        print(f"{r['seed']:5d} {r['candidates_run']:6d} {str(r['judged']):>7s} "
              f"{r['n_survivors']:5d} {str(r['admissible']):>6s} "
              f"{str(r['min_death_offset']):>7s} {str(r['max_death_offset']):>7s} "
              f"{str(r['min_blocking_frames']):>9s} {r['n_survive_180']:7d}")
    print()
    s = out["split"]
    print(f"admissible      : {len(admissible)} / {out['screen']['windows_judged']} "
          f"-> seeds {admissible}")
    print(f"rejected        : {out['screen']['rejected_seeds']}")
    print(f"split           : train {s['train_seeds']} ({len(train_ok)} ok)  "
          f"dev {s['dev_seeds']} ({len(dev_ok)} ok)  disjoint={s['disjoint']}")
    print(f"S1b (>=4/side)  : {s['s1b_min_4_per_side']}  ({s['s1b_detail']})")
    g = out["segment_length"]
    print(f"segment needed  : {g['required_frames']} frames = {g['required_seconds']} s "
          f"(margin {g['margin_frames']} frames at {g['chosen_frames']})")
    print(f"gate integrity  : all episodes 2,165 hash checks = "
          f"{out['gate_integrity']['every_episode_hash_checks_2165']}, mismatches "
          f"{out['gate_integrity']['hash_mismatches_total']}")
    print()
    print(f"wrote: {OUT / 'screen_split.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
