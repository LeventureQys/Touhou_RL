"""v4.3.1 recheck R25: C4 across tapes, and what the D-state arm can actually observe.

Part A — C4 replication. R24 found the seed12 window at frame 8,664 admissible (18/18
constants die inside 600 frames). One window is one observation, so the same rule is
repeated on seed10 and seed11 at the same frame. This aggregates every
`c4dense-seed<S>-f<F>-cand<C>.json` into one table and decides admissibility per window.

Part B — the observation channel. The prereg's arm A is described as "engine-truth
privileged state: player position, power, **boss HP**, **nearest-bullet geometry**". The
STATE reply (`touhou_rl/v400/labels/state.py: FIELD_NAMES`) has 25 fields and **none of
them describes a bullet or a boss**. So the arm as written cannot be built from STATE:
a bullet-aware D-state arm would need the `HAZARDS` channel, which is a *child* verb
(`CHILD_SNAPSHOT_VERBS`) whose availability on an env-role session is not established
here. That is recorded as an open question, not assumed either way.

Writes only Document/Update/Alpha/v4.3.1/evidence/R25/.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DIR = REPO / ".tmp_v431_seg001" / "full" / "raw" / "c4dense"
STATE_PY = REPO / "touhou_rl" / "v400" / "labels" / "state.py"
PROTO = REPO / "touhou_rl" / "v400" / "gym" / "protocol.py"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R25"
SEGMENT_FRAMES = 600
FPS = 60.0
PAT = re.compile(r"c4dense-seed(\d+)-f(\d+)-cand(\d+)\.json$")


def state_fields() -> list[str]:
    text = STATE_PY.read_text(encoding="utf-8")
    start = text.find("FIELD_NAMES")
    start = text.find("(", start)
    end = text.find("\n)", start)
    block = text[start:end]
    return re.findall(r'"([a-z_0-9]+)"', block)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="",
                    help="comma list; blank = every seed found. Use to regenerate a "
                         "historical snapshot instead of absorbing later windows.")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    only = {int(s) for s in args.seeds.split(",") if s.strip()}
    OUT.mkdir(parents=True, exist_ok=True)
    windows: dict[tuple[int, int], dict] = {}
    for f in sorted(DIR.glob("c4dense-seed*-f*-cand*.json")):
        m = PAT.search(f.name)
        if not m:
            continue
        seed, frame, cand = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if only and seed not in only:
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        w = windows.setdefault((seed, frame), {"cands": {}})
        w["cands"][cand] = {
            "terminal": rec["terminal"],
            "first_death_offset": rec["first_death_offset"],
            "frames_executed": rec["frames_executed"],
            "alive_180": rec["alive_180"],
            "hash_checks": (rec.get("restore") or {}).get("hash_checks"),
            "hash_mismatches": (rec.get("restore") or {}).get("hash_mismatches"),
            "wall_s": rec["wall_s"],
            "error": rec["error"],
        }

    rows = []
    for (seed, frame), w in sorted(windows.items()):
        c = w["cands"]
        survivors = sorted(k for k, v in c.items()
                           if v["terminal"] is None and v["error"] is None)
        offsets = [v["first_death_offset"] for v in c.values()
                   if v["first_death_offset"] is not None]
        max_off = max(offsets) if offsets else None
        judged = len(c) == 18 and all(v["error"] is None for v in c.values())
        rows.append({
            "seed": seed, "frame": frame,
            "candidates_run": len(c),
            "complete": sum(1 for v in c.values() if v["error"] is None),
            "judged": judged,
            "survivors": survivors,
            "n_survivors": len(survivors),
            # a partial window is NOT admissible-by-default: it simply is not judged yet
            "admissible": judged and len(survivors) == 0,
            "min_death_offset": min(offsets) if offsets else None,
            "median_death_offset": statistics.median(offsets) if offsets else None,
            "max_death_offset": max_off,
            "min_admissible_frames": (max_off + 1) if max_off is not None else None,
            "min_admissible_seconds": (round((max_off + 1) / FPS, 2)
                                       if max_off is not None else None),
            "n_survive_180": sum(1 for v in c.values() if v["alive_180"]),
            "hash_checks_all_2165": all((v["hash_checks"] or 0) == 2165 for v in c.values()),
            "hash_mismatches_total": sum(v["hash_mismatches"] or 0 for v in c.values()),
            "mean_wall_s": round(statistics.mean([v["wall_s"] for v in c.values()]), 1),
        })

    fields = state_fields()
    bullet_like = [f for f in fields if any(k in f for k in
                                           ("bullet", "hazard", "enemy", "boss", "shot_obs"))]
    proto = PROTO.read_text(encoding="utf-8")
    child_verbs = re.search(r"CHILD_SNAPSHOT_VERBS\s*=\s*\(([^)]*)\)", proto)
    child_verbs_list = re.findall(r'"([A-Z]+)"', child_verbs.group(1)) if child_verbs else []

    out = {
        "schema": "v431-recheck-r25/1",
        "partA_c4_across_tapes": {
            "segment_frames": SEGMENT_FRAMES,
            "segment_seconds": round(SEGMENT_FRAMES / FPS, 2),
            "windows": rows,
            "n_windows": len(rows),
            "n_judged": sum(1 for r in rows if r["judged"]),
            "n_admissible": sum(1 for r in rows if r["admissible"]),
            "admissible_seeds": sorted(r["seed"] for r in rows if r["admissible"]),
            "rejected_seeds": sorted(r["seed"] for r in rows
                                     if r["judged"] and not r["admissible"]),
            "partial_seeds": sorted(r["seed"] for r in rows if not r["judged"]),
            "all_judged_admissible": (all(r["admissible"] for r in rows if r["judged"])
                                      if any(r["judged"] for r in rows) else False),
            # kept for R25's frozen checks: "every judged window admissible"
            "all_admissible": (all(r["admissible"] for r in rows if r["judged"])
                               if any(r["judged"] for r in rows) else False),
            "verdict": None,
        },
        "partB_observation_channel": {
            "state_fields": fields,
            "state_field_count": len(fields),
            "state_has_bullet_or_boss_field": bool(bullet_like),
            "bullet_like_fields": bullet_like,
            "child_snapshot_verbs": child_verbs_list,
            "prereg_arm_a_description": ("player position, power, boss HP, nearest-bullet "
                                         "geometry"),
            "gap": ("STATE exposes player position/resources only; it carries no bullet or "
                    "boss field. A bullet-aware D-state arm needs the HAZARDS channel, and "
                    "HAZARDS is listed among the CHILD snapshot verbs - its availability on "
                    "the env-role session used here is NOT established."),
            "options": [
                "A-STATE: build the arm from the 25 STATE fields only; cheap, no new "
                "plumbing, but blind to bullets - it can only learn 'where not to stand'",
                "A-HAZARDS: probe whether an env-role session answers HAZARDS; if it does, "
                "the arm can see bullets, but this needs one cheap engine probe first",
            ],
            "status": "open - not resolved in this round",
        },
    }
    a = out["partA_c4_across_tapes"]
    a["verdict"] = (
        f"{a['n_admissible']} of {a['n_judged']} judged windows admissible "
        f"(admissible seeds {a['admissible_seeds']}, rejected {a['rejected_seeds']}, "
        f"partial {a['partial_seeds']}); admissibility is window-specific, not a regime "
        f"property")

    target = pathlib.Path(args.out) if args.out else (OUT / "c4_multi.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"== C4 across tapes (segment {SEGMENT_FRAMES} frames = "
          f"{out['partA_c4_across_tapes']['segment_seconds']} s) ==")
    print(f"{'seed':>5s} {'frame':>6s} {'cands':>6s} {'judged':>7s} {'survivors':>10s} "
          f"{'adm':>5s} "
          f"{'minOff':>7s} {'maxOff':>7s} {'minAdmF':>8s} {'>=180f':>7s} {'wall':>7s}")
    print("-" * 80)
    for r in rows:
        print(f"{r['seed']:5d} {r['frame']:6d} {r['candidates_run']:6d} "
              f"{str(r['judged']):>7s} {r['n_survivors']:10d} "
              f"{str(r['admissible']):>5s} "
              f"{str(r['min_death_offset']):>7s} {str(r['max_death_offset']):>7s} "
              f"{str(r['min_admissible_frames']):>8s} {r['n_survive_180']:7d} "
              f"{r['mean_wall_s']:7.1f}")
    print()
    print(f"judged windows    : {a['n_judged']} / {a['n_windows']} "
          f"(admissible {a['n_admissible']})")
    print(f"admissible seeds  : {a['admissible_seeds']}")
    print(f"rejected seeds    : {a['rejected_seeds']}")
    print(f"partial seeds     : {a['partial_seeds']}")
    print(f"verdict           : {a['verdict']}")
    print()
    b = out["partB_observation_channel"]
    print("== observation channel ==")
    print(f"  STATE fields          : {b['state_field_count']}")
    print(f"  bullets/boss in STATE : {b['state_has_bullet_or_boss_field']} "
          f"({b['bullet_like_fields']})")
    print(f"  child snapshot verbs  : {b['child_snapshot_verbs']}")
    print(f"  status                : {b['status']}")
    print()
    print(f"wrote: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
